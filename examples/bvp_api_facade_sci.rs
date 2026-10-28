//! Minimal typed BVP facade example using the production BVP_sci route.
//!
//! Run with `cargo run --example bvp_api_facade_sci`.

use RustedSciThe::numerical::BVP_api::{
    BvpBoundaryCondition, BvpBoundarySide, BvpProblem, BvpSciSparseFactorization, BvpSolution,
    BvpSolver, BvpSolverConfig, BvpSolverTelemetrySnapshot,
};
use RustedSciThe::numerical::BVP_sci::new::{
    BvpSciAssembly, BvpSciExecution, BvpSciOptions, BvpSciTelemetryMode,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DMatrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // y' = -y with y(0) = 1; the solver determines the right endpoint value.
    let problem = BvpProblem::new(
        vec![Expr::parse_expression("-y")],
        vec!["y".into()],
        "x",
        vec![0.0, 0.25, 0.5, 0.75, 1.0],
        vec![BvpBoundaryCondition {
            state: 0,
            side: BvpBoundarySide::Left,
            value: 1.0,
        }],
    )?;

    // BVP_sci takes one initial-guess column per mesh point. Damped and Frozen
    // have their own native layouts, which are represented by separate config
    // variants rather than coerced into one common option structure.
    let config = BvpSolverConfig::Sci {
        options: BvpSciOptions::default()
            .with_assembly(BvpSciAssembly::ExprLegacy)
            .with_execution(BvpSciExecution::Lambdify)
            .with_telemetry(BvpSciTelemetryMode::Counters),
        sparse_factorization: BvpSciSparseFactorization::StandardFaer,
        aot_config: None,
        initial_guess: DMatrix::from_row_slice(1, 5, &[1.0, 0.78, 0.61, 0.47, 0.37]),
    };
    let mut solver = BvpSolver::new(problem, config)?;

    match solver.solve()? {
        BvpSolution::Sci(solution) => {
            println!(
                "y(1) = {:.8}; Newton iterations = {}",
                solution.y[4], solution.iterations
            );
        }
        BvpSolution::Damped { .. } | BvpSolution::Frozen { .. } => {
            unreachable!("the configured route is BVP_sci")
        }
    }

    let data = solver.postprocess_dataset()?;
    println!("postprocessing data shape: {:?}", data.values.shape());
    if let BvpSolverTelemetrySnapshot::Sci(snapshot) = solver.telemetry_snapshot() {
        println!("native RHS evaluations: {}", snapshot.rhs_calls);
    }
    Ok(())
}
