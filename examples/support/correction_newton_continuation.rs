//! Shared setup and checks for the two public-facade continuation examples.

use RustedSciThe::numerical::BVP_CorrectionNewton::{
    CorrectionNewtonMatrixLayout, CorrectionNewtonOptions, CorrectionNewtonParameterLoopOutcome,
    CorrectionNewtonSymbolicFrontend, CorrectionNewtonTelemetryMode,
};
use RustedSciThe::numerical::BVP_api::{
    BvpBoundaryCondition, BvpBoundarySide, BvpParameterLoopReport, BvpProblem, BvpSolver,
    BvpSolverConfig,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DMatrix;

pub struct Model {
    pub names: Vec<&'static str>,
    pub equations: Vec<&'static str>,
    pub right: f64,
    pub boundaries: Vec<(BvpBoundarySide, f64)>,
    pub initial: Vec<f64>,
    pub parameter: &'static str,
    pub parameters: Vec<Vec<f64>>,
    pub rhs: fn(f64, &[f64]) -> Vec<f64>,
    pub check: fn(f64, &[f64], &DMatrix<f64>),
}

pub fn run(model: Model) -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let steps = args.next().map_or(Ok(256), |arg| arg.parse::<usize>())?;
    if steps < 16 || args.next().is_some() {
        return Err("usage: example [intervals >= 16]; default: 256".into());
    }
    let mesh: Vec<_> = (0..=steps)
        .map(|node| model.right * node as f64 / steps as f64)
        .collect();
    let conditions = model
        .boundaries
        .iter()
        .enumerate()
        .map(|(state, &(side, value))| BvpBoundaryCondition { state, side, value })
        .collect();
    let problem = BvpProblem::new(
        model
            .equations
            .iter()
            .map(|eq| Expr::parse_expression(eq))
            .collect(),
        model.names.iter().map(|name| (*name).into()).collect(),
        "x",
        mesh.clone(),
        conditions,
    )?
    .with_external_parameters(vec![model.parameter.into()], model.parameters[0].clone())?;
    let options = CorrectionNewtonOptions {
        symbolic_frontend: CorrectionNewtonSymbolicFrontend::AtomView,
        matrix_layout: CorrectionNewtonMatrixLayout::BandedAuto,
        absolute_tolerance: 1e-8,
        default_relative_tolerance: 1e-8,
        bounds: model
            .names
            .iter()
            .map(|name| ((*name).into(), (-10.0, 10.0)))
            .collect(),
        telemetry_mode: CorrectionNewtonTelemetryMode::Timings,
        ..CorrectionNewtonOptions::default()
    };
    // State-major reduced guess: left-constrained states omit node 0;
    // right-constrained states omit node `steps`. Output is full node-major.
    let initial_guess = DMatrix::from_fn(model.names.len(), steps, |state, _| model.initial[state]);
    let mut solver = BvpSolver::new(
        problem,
        BvpSolverConfig::CorrectionNewton {
            options,
            initial_guess,
        },
    )?;
    let BvpParameterLoopReport::CorrectionNewton(report) =
        solver.solve_parameter_loop(&model.parameters)?
    else {
        unreachable!("configured the CorrectionNewton route")
    };
    assert_eq!(
        report.points.len(),
        10,
        "the whole schedule must be attempted"
    );
    let mut last_success = None;
    let mut successful = 0;
    for (index, point) in report.points.iter().enumerate() {
        assert_eq!(point.index, index);
        assert_eq!(point.parameters, model.parameters[index]);
        assert_eq!(
            point.guess_source_index, last_success,
            "only successful profiles become checkpoints"
        );
        match &point.outcome {
            CorrectionNewtonParameterLoopOutcome::Converged {
                mesh: actual_mesh,
                values,
                report,
            } => {
                assert_eq!(actual_mesh.len(), mesh.len());
                assert!(
                    actual_mesh
                        .iter()
                        .zip(&mesh)
                        .all(|(a, b)| (a - b).abs() < 1e-14)
                );
                assert_eq!(values.shape(), (mesh.len(), model.names.len()));
                assert!(values.iter().all(|v| v.is_finite() && v.abs() <= 10.0));
                for (state, &(side, expected)) in model.boundaries.iter().enumerate() {
                    let node = match side {
                        BvpBoundarySide::Left => 0,
                        BvpBoundarySide::Right => steps,
                    };
                    assert!((values[(node, state)] - expected).abs() < 1e-12);
                }
                // Independent evaluation of the fixed-grid trapezoidal equations.
                let mut residual = 0.0_f64;
                for node in 0..steps {
                    let left: Vec<_> = values.row(node).iter().copied().collect();
                    let right: Vec<_> = values.row(node + 1).iter().copied().collect();
                    let fl = (model.rhs)(point.parameters[0], &left);
                    let fr = (model.rhs)(point.parameters[0], &right);
                    let h = actual_mesh[node + 1] - actual_mesh[node];
                    for state in 0..model.names.len() {
                        residual = residual.max(
                            (right[state] - left[state] - 0.5 * h * (fl[state] + fr[state])).abs(),
                        );
                    }
                }
                assert!(residual < 1e-7, "discrete residual: {residual:e}");
                assert!(report.correction_norm.is_some_and(|norm| norm < 1.0));
                (model.check)(point.parameters[0], actual_mesh, values);
                println!(
                    "point={index} {}={:.6}, guess={:?}, nodes={}, iterations={}, residual={residual:.3e}",
                    model.parameter,
                    point.parameters[0],
                    point.guess_source_index,
                    actual_mesh.len(),
                    report.iterations
                );
                println!("  telemetry: {:?}", report.telemetry);
                last_success = Some(index);
                successful += 1;
            }
            CorrectionNewtonParameterLoopOutcome::Failed { error, .. } => {
                eprintln!("point={index} failed: {error}; checkpoint remains {last_success:?}");
                // Do not advance last_success: the following point must reuse it.
            }
        }
    }
    assert_eq!(
        successful, 10,
        "the default sweep should converge at every point"
    );
    Ok(())
}
