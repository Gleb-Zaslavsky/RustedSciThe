//! Reuse one prepared six-state combustion BVP across an ordered heat-input sweep.
//!
//! This is a substantial sparse BVP_sci workload. Each point changes only the
//! fixed `heat_scale` input and warm-starts from the last successful profile.

use RustedSciThe::numerical::BVP_api::{
    BvpBoundaryCondition, BvpBoundarySide, BvpParameterLoopReport, BvpProblem,
    BvpSciSparseFactorization, BvpSolver, BvpSolverConfig,
};
use RustedSciThe::numerical::BVP_sci::new::{
    BvpSciAssembly, BvpSciExecution, BvpSciMatrixLayout, BvpSciOptions, BvpSciStatus,
    BvpSciTelemetryMode,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DMatrix;

const STATES: [&str; 6] = ["Teta", "q", "C0", "J0", "C1", "J1"];

fn combustion_rhs() -> Vec<Expr> {
    let [teta, q, c0, j0, _c1, j1] = Expr::parse_vector_expression(STATES.to_vec())
        .try_into()
        .expect("combustion model has six states");
    let heat_scale = Expr::parse_expression("heat_scale");
    let rate = Expr::Const(1.3e5)
        * Expr::exp(
            Expr::Const(-5000.0 * 4.184)
                / (Expr::Const(8.314) * (teta * Expr::Const(600.0) + Expr::Const(600.0))),
        )
        * c0
        * Expr::Const((34.2 / 1000.0) * 2.0e6 / (8.314 * 1500.0) / 0.342);
    vec![
        q.clone() / Expr::Const(0.07),
        q * Expr::Const(0.0090168)
            - heat_scale
                * Expr::Const(3000.0e3 * 0.034 * (3.0e-4_f64).powi(2) / 600.0)
                * rate.clone(),
        j0.clone() / Expr::Const(2.88e-4),
        j0 * Expr::Const(1.50e-3)
            + rate.clone()
                * Expr::Const((34.2 / 1000.0) * 2.0e6 / (8.314 * 1500.0) * (3.0e-4_f64).powi(2)),
        j1.clone() / Expr::Const(2.88e-4),
        j1 * Expr::Const(1.50e-3)
            + rate * Expr::Const((34.2 / 1000.0) * 2.0e6 / (8.314 * 1500.0) * (3.0e-4_f64).powi(2)),
    ]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let nodes = 1_000;
    let mesh = (0..nodes)
        .map(|index| index as f64 / (nodes - 1) as f64)
        .collect::<Vec<_>>();
    let problem = BvpProblem::new(
        combustion_rhs(),
        STATES.iter().map(|state| (*state).to_owned()).collect(),
        "x",
        mesh,
        vec![
            BvpBoundaryCondition {
                state: 0,
                side: BvpBoundarySide::Left,
                value: 400.0 / 600.0,
            },
            BvpBoundaryCondition {
                state: 1,
                side: BvpBoundarySide::Right,
                value: 1e-10,
            },
            BvpBoundaryCondition {
                state: 2,
                side: BvpBoundarySide::Left,
                value: 1.0,
            },
            BvpBoundaryCondition {
                state: 3,
                side: BvpBoundarySide::Right,
                value: 1e-7,
            },
            BvpBoundaryCondition {
                state: 4,
                side: BvpBoundarySide::Left,
                value: 1e-3,
            },
            BvpBoundaryCondition {
                state: 5,
                side: BvpBoundarySide::Right,
                value: 1e-7,
            },
        ],
    )?
    .with_external_parameters(vec!["heat_scale".into()], vec![1.0])?;
    let options = BvpSciOptions::default()
        .with_execution(BvpSciExecution::Lambdify)
        .with_assembly(BvpSciAssembly::AtomViewNative)
        .with_matrix_layout(BvpSciMatrixLayout::Sparse)
        .with_tolerance(1e-6)
        .with_limits(nodes + 64, 20, 4)
        .with_telemetry(BvpSciTelemetryMode::Counters);
    let mut solver = BvpSolver::new(
        problem,
        BvpSolverConfig::Sci {
            options,
            sparse_factorization: BvpSciSparseFactorization::StandardFaer,
            aot_config: None,
            initial_guess: DMatrix::from_element(STATES.len(), nodes, 0.99),
        },
    )?;

    let values = (0..10)
        .map(|index| vec![1.0 + index as f64 * 0.005])
        .collect::<Vec<_>>();
    let BvpParameterLoopReport::Sci(report) = solver.solve_parameter_loop(&values)? else {
        unreachable!("the configured route is BVP_sci")
    };
    for point in report.points {
        match point.result {
            Ok(solution) if solution.status == BvpSciStatus::Success => println!(
                "heat_scale={:.3}: nodes={}, Newton iterations={}, RHS calls={}, residual={:.3e}",
                point.external_parameters[0],
                solution.x.len(),
                solution.iterations,
                point.telemetry.rhs_calls,
                solution.residual_norm
            ),
            Ok(solution) => eprintln!(
                "heat_scale={:.3}: solver status {:?}",
                point.external_parameters[0], solution.status
            ),
            Err(error) => eprintln!(
                "heat_scale={:.3}: failed: {error}",
                point.external_parameters[0]
            ),
        }
    }
    Ok(())
}
