//! Modern typed guide for nonlinear systems.
//!
//! This example keeps symbolic preparation separate from numeric solving:
//! prepare the equations once, bind several parameter vectors, and inspect
//! solver statistics for every solve.
//!
//! Run with:
//!
//! ```text
//! cargo run --example nonlinear_systems_modern_guide
//! ```

use RustedSciThe::numerical::Nonlinear_systems::prelude::{
    Bounds, DampedNewtonMethod, DiagnosticsOptions, NonlinearSolverMethod,
    PreparedSymbolicNonlinearProblem, SolveOptions, SymbolicProblemOptions,
};
use nalgebra::DVector;

fn solve_options(bounds: Option<Bounds>) -> SolveOptions {
    SolveOptions {
        tolerance: 1e-10,
        max_iterations: 64,
        bounds,
        diagnostics: DiagnosticsOptions {
            collect_history: false,
            collect_statistics: true,
            enable_logging: false,
            ..DiagnosticsOptions::default()
        },
        ..SolveOptions::default()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Modern nonlinear-system guide ===");
    parameter_sweep()?;
    bounded_system()?;
    Ok(())
}

fn parameter_sweep() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n1. Prepare once, bind several parameter vectors");
    println!("   equations: a*x + y - 3 = 0, x - y = 0");

    let prepared = PreparedSymbolicNonlinearProblem::from_strings(
        vec!["a*x + y - 3".to_string(), "x - y".to_string()],
        SymbolicProblemOptions::new()
            .with_variables(vec!["x".to_string(), "y".to_string()])
            .with_equation_parameters(vec!["a".to_string()]),
    )?;
    println!("   prepared backend: {:?}", prepared.backend_kind());
    println!(
        "   parameter schema: {:?}",
        prepared.parameter_schema().map(|schema| schema.names())
    );

    let method = NonlinearSolverMethod::DampedNewton(DampedNewtonMethod::default());
    for (a, initial) in [(1.0, [1.0, 1.0]), (2.0, [1.0, 1.0]), (0.5, [1.0, 1.0])] {
        let bound = prepared.bind_values(DVector::from_vec(vec![a]))?;
        let result = method.clone().solve(
            &bound,
            DVector::from_vec(initial.to_vec()),
            solve_options(None),
        )?;
        println!(
            "   a={a:.2}: x={:.8}, y={:.8}, iterations={}, residual_calls={}, jacobian_calls={}, attempts={}",
            result.x[0],
            result.x[1],
            result.statistics.iterations,
            result.statistics.residual_evaluations,
            result.statistics.jacobian_evaluations,
            result.statistics.attempts.len(),
        );
        if let Some(attempt) = result.statistics.attempts.last() {
            println!(
                "      last attempt: iteration={}, R/J={}/{}, refresh/reuse={}/{}, factor/linear={}/{}, accepted/rejected={}/{}",
                attempt.iteration,
                attempt.residual_evaluations,
                attempt.jacobian_evaluations,
                attempt.jacobian_refreshes,
                attempt.jacobian_reuses,
                attempt.linear_factorizations,
                attempt.linear_solves,
                attempt.accepted_steps,
                attempt.rejected_steps,
            );
        }
    }
    println!("   symbolic preparation is not repeated for these bindings");
    Ok(())
}

fn bounded_system() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n2. Bound-aware solve with symbolic Jacobian");
    println!("   equations: x^2 - 4 = 0, y^2 - 1 = 0");

    let problem = PreparedSymbolicNonlinearProblem::from_strings(
        vec!["x^2 - 4".to_string(), "y^2 - 1".to_string()],
        SymbolicProblemOptions::new().with_variables(vec!["x".to_string(), "y".to_string()]),
    )?;
    let bound = problem.bind_without_parameters()?;
    let bounds = Bounds::new(vec![(0.0, 3.0), (0.0, 2.0)])?;
    let result = NonlinearSolverMethod::DampedNewton(DampedNewtonMethod::default()).solve(
        &bound,
        DVector::from_vec(vec![0.5, 0.5]),
        solve_options(Some(bounds)),
    )?;
    println!(
        "   solution: x={:.8}, y={:.8}; final residual norm={:.3e}",
        result.x[0],
        result.x[1],
        result.residual.norm(),
    );
    println!("   bounds are enforced by SolveOptions, not by clipping the reported result");
    Ok(())
}
