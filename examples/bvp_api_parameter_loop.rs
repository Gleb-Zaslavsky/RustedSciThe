//! Use the typed BVP facade for a prepared BVP_sci fixed-input sweep.

use RustedSciThe::numerical::BVP_api::{
    BvpBoundaryCondition, BvpBoundarySide, BvpParameterLoopReport, BvpProblem,
    BvpSciSparseFactorization, BvpSolver, BvpSolverConfig,
};
use RustedSciThe::numerical::BVP_sci::new::{
    BvpSciAssembly, BvpSciExecution, BvpSciOptions, BvpSciTelemetryMode,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DMatrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // y' = rate*y, y(0)=1, with exact solution exp(rate*x).
    let mesh = (0..41).map(|index| index as f64 / 40.0).collect::<Vec<_>>();
    let guess = DMatrix::from_row_slice(
        1,
        mesh.len(),
        &mesh.iter().map(|x| (0.5 * x).exp()).collect::<Vec<_>>(),
    );
    let problem = BvpProblem::new(
        vec![Expr::parse_expression("rate * y")],
        vec!["y".into()],
        "x",
        mesh,
        vec![BvpBoundaryCondition {
            state: 0,
            side: BvpBoundarySide::Left,
            value: 1.0,
        }],
    )?
    .with_external_parameters(vec!["rate".into()], vec![0.5])?;
    let mut solver = BvpSolver::new(
        problem,
        BvpSolverConfig::Sci {
            options: BvpSciOptions::default()
                .with_execution(BvpSciExecution::Lambdify)
                .with_assembly(BvpSciAssembly::AtomViewNative)
                .with_tolerance(1.0e-6)
                .with_telemetry(BvpSciTelemetryMode::Counters),
            sparse_factorization: BvpSciSparseFactorization::StandardFaer,
            aot_config: None,
            initial_guess: guess,
        },
    )?;

    let report = solver.solve_parameter_loop(&[vec![0.5], vec![0.75], vec![1.0]])?;
    let BvpParameterLoopReport::Sci(report) = report else {
        unreachable!("the configured route is BVP_sci")
    };
    for point in report.points {
        match point.result {
            Ok(solution) => {
                let terminal = solution.y.last().copied().unwrap_or(f64::NAN);
                println!(
                    "rate={:.2}: y(1)={terminal:.8}, RHS calls={}",
                    point.external_parameters[0], point.telemetry.rhs_calls
                );
            }
            Err(error) => eprintln!(
                "rate={:.2}: point failed: {error}",
                point.external_parameters[0]
            ),
        }
    }
    Ok(())
}
