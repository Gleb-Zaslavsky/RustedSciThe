//! Run fixed-input parameter loops through the typed BVP facade.
//!
//! Damped keeps the last successful adaptive mesh/profile checkpoint. Frozen
//! keeps one fixed mesh and carries only the last successful profile. A failed
//! point is reported without discarding the earlier successful checkpoint.
//!
//! Run with `cargo run --example bvp_api_damped_frozen_parameter_loop`.

use std::collections::HashMap;

use RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_damped::{
    DampedParameterLoopOutcome, DampedSolverOptions,
};
use RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_frozen::{
    FrozenParameterLoopOutcome, FrozenSolverOptions,
};
use RustedSciThe::numerical::BVP_Damp::telemetry::BvpTelemetryMode;
use RustedSciThe::numerical::BVP_api::{
    BvpBoundaryCondition, BvpBoundarySide, BvpParameterLoopReport, BvpProblem, BvpSolver,
    BvpSolverConfig,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DMatrix;

fn problem() -> Result<BvpProblem, Box<dyn std::error::Error>> {
    // y' = 1/rate, y(0) = 0; therefore y(1) = 1/rate.
    Ok(BvpProblem::new(
        vec![Expr::parse_expression("1 / rate")],
        vec!["y".into()],
        "x",
        (0..9).map(|i| i as f64 / 8.0).collect(),
        vec![BvpBoundaryCondition {
            state: 0,
            side: BvpBoundarySide::Left,
            value: 0.0,
        }],
    )?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The zero-valued middle input intentionally causes a point-level native
    // evaluation failure; the final point should still use the last success.
    let sweep = [vec![1.0], vec![0.0], vec![4.0]];

    let mut damped = BvpSolver::new(
        problem()?,
        BvpSolverConfig::Damped {
            options: DampedSolverOptions::default()
                .with_bounds(HashMap::from([("y".to_owned(), (-10.0, 10.0))]))
                .with_rel_tolerance(HashMap::from([("y".to_owned(), 1.0e-8)]))
                .with_bvp_telemetry_mode(BvpTelemetryMode::Counters),
            // Damped stores one guess column per interval (8 columns).
            initial_guess: DMatrix::zeros(1, 8),
        },
    )?;
    damped.bind_parameters(vec!["rate".into()], vec![1.0])?;
    let BvpParameterLoopReport::Damped(report) = damped.solve_parameter_loop(&sweep)? else {
        unreachable!("the selected facade route is Damped")
    };
    for point in report.points {
        match point.outcome {
            DampedParameterLoopOutcome::Converged {
                mesh,
                values,
                telemetry,
            } => println!(
                "Damped rate={}: y(1)={:.6}, nodes={}, iterations={}",
                point.parameters[0],
                values[(mesh.len() - 1, 0)],
                mesh.len(),
                telemetry.counters.iterations,
            ),
            DampedParameterLoopOutcome::Failed { error, .. } => {
                println!("Damped rate={}: point failed: {error}", point.parameters[0])
            }
            DampedParameterLoopOutcome::NotConverged { telemetry } => println!(
                "Damped rate={}: did not converge after {} iterations",
                point.parameters[0], telemetry.counters.iterations
            ),
        }
    }
    if let Some(error) = report.stop_error {
        return Err(format!("Damped loop could not recover safely: {error}").into());
    }

    let mut frozen = BvpSolver::new(
        problem()?,
        BvpSolverConfig::Frozen {
            options: FrozenSolverOptions::default()
                .with_bvp_telemetry_mode(BvpTelemetryMode::Counters),
            // Frozen stores one guess column per node (9 columns).
            initial_guess: DMatrix::zeros(1, 9),
        },
    )?;
    frozen.bind_parameters(vec!["rate".into()], vec![1.0])?;
    let BvpParameterLoopReport::Frozen(report) = frozen.solve_parameter_loop(&sweep)? else {
        unreachable!("the selected facade route is Frozen")
    };
    for point in report.points {
        match point.outcome {
            FrozenParameterLoopOutcome::Converged {
                mesh,
                values,
                telemetry,
            } => println!(
                "Frozen rate={}: y(1)={:.6}, nodes={}, iterations={}",
                point.parameters[0],
                values[(mesh.len() - 1, 0)],
                mesh.len(),
                telemetry.counters.iterations,
            ),
            FrozenParameterLoopOutcome::Failed { error, .. } => {
                println!("Frozen rate={}: point failed: {error}", point.parameters[0])
            }
            FrozenParameterLoopOutcome::NotConverged { telemetry } => println!(
                "Frozen rate={}: did not converge after {} iterations",
                point.parameters[0], telemetry.counters.iterations
            ),
        }
    }
    if let Some(error) = report.stop_error {
        return Err(format!("Frozen loop could not recover safely: {error}").into());
    }
    Ok(())
}
