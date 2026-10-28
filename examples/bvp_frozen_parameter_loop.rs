//! Ordered fixed-parameter loop with Frozen BVP and last-successful guess reuse.
//!
//! The middle point deliberately leaves the real domain of `sqrt(alpha)`.
//! Its typed native failure is retained in the report, and the final point is
//! solved from the first successful profile instead of the failed iterate.
//!
//! Run with `cargo run --example bvp_frozen_parameter_loop`.

use std::collections::HashMap;

use RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_frozen::{
    FrozenParameterLoopOutcome, FrozenSolverOptions, NRBVP,
};
use RustedSciThe::numerical::BVP_Damp::telemetry::BvpTelemetryMode;
use RustedSciThe::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DMatrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = FrozenSolverOptions::sparse_frozen()
        .with_tolerance(1e-9)
        .with_bvp_telemetry_mode(BvpTelemetryMode::Counters);
    let backend = options
        .generated_backend_config
        .clone()
        .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly));
    let options = options.with_generated_backend_config(backend);

    let mut solver = NRBVP::new_with_options(
        vec![Expr::parse_expression("alpha^0.5")],
        DMatrix::zeros(1, 6),
        vec!["y".to_owned()],
        "x".to_owned(),
        HashMap::from([("y".to_owned(), vec![(0, 0.0)])]),
        0.0,
        1.0,
        6,
        options,
    );
    solver.try_set_parameter_names(Some(&["alpha"]))?;

    // The order is intentional. The invalid middle point should not erase the
    // successful profile from alpha=1, which is reused for alpha=4.
    let report = solver.try_solve_parameter_loop(&[vec![1.0], vec![-1.0], vec![4.0]])?;
    if let Some(error) = report.stop_error {
        return Err(format!("parameter loop stopped while restoring solver state: {error}").into());
    }

    let mut converged = 0;
    for point in report.points {
        match point.outcome {
            FrozenParameterLoopOutcome::Converged {
                mesh,
                values,
                telemetry,
            } => {
                converged += 1;
                println!(
                    "point {}: alpha={}, converged, nodes={}, y(1)={:.6}, iterations={}",
                    point.index,
                    point.parameters[0],
                    mesh.len(),
                    values[(mesh.len() - 1, 0)],
                    telemetry.counters.iterations,
                );
            }
            FrozenParameterLoopOutcome::NotConverged { telemetry } => println!(
                "point {}: alpha={}, did not converge ({} iterations)",
                point.index, point.parameters[0], telemetry.counters.iterations,
            ),
            FrozenParameterLoopOutcome::Failed { error, .. } => println!(
                "point {}: alpha={}, native solve failed: {error}",
                point.index, point.parameters[0],
            ),
        }
    }
    assert_eq!(
        converged, 2,
        "the first and last parameter points should solve"
    );
    Ok(())
}
