//! Ordered fixed-parameter sweep with adaptive Damped BVP checkpoints.
//!
//! Each successful point carries both its refined mesh and profile forward.
//! A deliberately invalid middle parameter is reported without contaminating
//! the checkpoint used by the final point.
//!
//! Run with `cargo run --example bvp_damped_parameter_loop`.

use std::collections::HashMap;

use RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_damped::{
    AdaptiveGridConfig, DampedParameterLoopOutcome, DampedSolverOptions, NRBVP, SolverParams,
};
use RustedSciThe::numerical::BVP_Damp::grid_api::GridRefinementMethod;
use RustedSciThe::numerical::BVP_Damp::telemetry::BvpTelemetryMode;
use RustedSciThe::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DMatrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut options = DampedSolverOptions::dense_damped()
        .with_abs_tolerance(1e-9)
        .with_rel_tolerance(HashMap::from([("y".to_owned(), 1e-8)]))
        .with_bounds(HashMap::from([("y".to_owned(), (-10.0, 10.0))]))
        .with_max_iterations(12)
        .with_strategy_params(Some(SolverParams {
            adaptive: Some(AdaptiveGridConfig {
                version: 1,
                max_refinements: 1,
                grid_method: GridRefinementMethod::DoublePoints,
            }),
            ..SolverParams::default()
        }))
        .with_bvp_telemetry_mode(BvpTelemetryMode::Counters);
    options.generated_backend_config = options
        .generated_backend_config
        .clone()
        .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly));

    let mut solver = NRBVP::new_with_options(
        vec![Expr::parse_expression("alpha^0.5")],
        DMatrix::zeros(1, 4),
        vec!["y".to_owned()],
        "x".to_owned(),
        HashMap::from([("y".to_owned(), vec![(0, 0.0)])]),
        0.0,
        1.0,
        4,
        options,
    );
    solver.try_set_parameter_names(Some(&["alpha"]))?;
    solver.dont_save_log(true);

    let report = solver.try_solve_parameter_loop(&[vec![1.0], vec![-1.0], vec![4.0]])?;
    if let Some(error) = report.stop_error {
        return Err(format!("parameter loop stopped during safe recovery: {error}").into());
    }

    let mut converged = 0;
    for point in report.points {
        match point.outcome {
            DampedParameterLoopOutcome::Converged {
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
            DampedParameterLoopOutcome::NotConverged { telemetry } => println!(
                "point {}: alpha={}, did not converge ({} iterations)",
                point.index, point.parameters[0], telemetry.counters.iterations,
            ),
            DampedParameterLoopOutcome::Failed { error, .. } => println!(
                "point {}: alpha={}, native solve failed: {error}",
                point.index, point.parameters[0],
            ),
        }
    }
    assert_eq!(converged, 2);
    Ok(())
}
