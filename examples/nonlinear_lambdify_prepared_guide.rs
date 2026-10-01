//! Prepared Lambdify guide for parameterized nonlinear systems.
//!
//! Preparation and runtime binding are separate lifecycle stages:
//! `PreparedSymbolicNonlinearProblem` owns symbolic differentiation and the
//! compiled scalar callbacks, while each `bind_values` call creates a cheap
//! solver-facing view. The same example is run with sequential and thresholded
//! parallel callback execution.
//!
//! Run with:
//!
//! ```text
//! cargo run --example nonlinear_lambdify_prepared_guide
//! ```

use RustedSciThe::numerical::Nonlinear_systems::prelude::{
    DampedNewtonMethod, DiagnosticsOptions, JacobianProvider, LambdifyExecutionPolicy,
    NonlinearProblem, NonlinearSolverMethod, PreparedSymbolicNonlinearProblem, SolveOptions,
    SolveResult, SymbolicProblemOptions,
};
use nalgebra::{DMatrix, DVector};

fn print_diagnostics(result: &SolveResult) {
    let statistics = &result.statistics;
    let milliseconds = |duration: std::time::Duration| duration.as_secs_f64() * 1e3;

    println!("  diagnostics availability: {:?}", statistics.availability);
    println!(
        "  iterations={} accepted={} rejected={} reusable_trials={}",
        statistics.iterations,
        statistics.accepted_steps,
        statistics.rejected_steps,
        statistics.reusable_trial_points,
    );
    println!(
        "  residual calls={} (state={}, trial={}), time_ms={:.3}",
        statistics.residual_evaluations,
        statistics.state_residual_evaluations,
        statistics.trial_residual_evaluations,
        milliseconds(statistics.residual_duration),
    );
    println!(
        "  Jacobian calls={} (state={}, trial={}), time_ms={:.3}",
        statistics.jacobian_evaluations,
        statistics.state_jacobian_evaluations,
        statistics.trial_jacobian_evaluations,
        milliseconds(statistics.jacobian_duration),
    );
    println!(
        "  linear solves={} factorizations={}, linear_ms={:.3} (factor_ms={:.3}, solve_ms={:.3}), total_ms={:.3}",
        statistics.linear_solves,
        statistics.linear_factorizations,
        milliseconds(statistics.linear_solve_duration),
        milliseconds(statistics.linear_factorization_duration),
        milliseconds(statistics.linear_system_solve_duration),
        milliseconds(statistics.total_duration),
    );
}

fn solve_options() -> SolveOptions {
    SolveOptions {
        tolerance: 1e-11,
        max_iterations: 50,
        diagnostics: DiagnosticsOptions {
            collect_history: false,
            collect_statistics: true,
            enable_logging: false,
            ..DiagnosticsOptions::default()
        },
        ..SolveOptions::default()
    }
}

fn run_policy(policy: LambdifyExecutionPolicy) -> Result<(), Box<dyn std::error::Error>> {
    println!("\npolicy: {policy:?}");
    let prepared = PreparedSymbolicNonlinearProblem::from_strings(
        vec!["x^2 - a".to_string(), "y - b*x".to_string()],
        SymbolicProblemOptions::new()
            .with_variables(vec!["x".to_string(), "y".to_string()])
            .with_equation_parameters(vec!["a".to_string(), "b".to_string()])
            .with_lambdify_backend()
            .with_lambdify_execution_policy(policy),
    )?;

    println!("prepared backend: {:?}", prepared.backend_kind());
    println!(
        "prepared policy: {:?}",
        prepared.lambdify_execution_policy()
    );
    println!(
        "parameter schema: {:?}",
        prepared.parameter_schema().map(|s| s.names())
    );

    let method = NonlinearSolverMethod::DampedNewton(DampedNewtonMethod::default());
    for (parameters, initial) in [([4.0, 3.0], [1.0, 3.0]), ([9.0, 2.0], [2.0, 4.0])] {
        let bound = prepared.bind_values(DVector::from_vec(parameters.to_vec()))?;
        let point = DVector::from_vec(initial.to_vec());
        let mut residual = DVector::zeros(2);
        let mut jacobian = DMatrix::zeros(2, 2);

        // These buffers are reused for both callbacks. Binding parameters does
        // not rebuild the symbolic Jacobian or the Lambdify closures.
        bound.residual_into(&point, &mut residual)?;
        bound.jacobian_into(&point, &mut jacobian)?;
        let result = method.clone().solve(&bound, point, solve_options())?;
        println!(
            "parameters={parameters:?}; residual={residual:?}; J00={:.3}; solution=[{:.8}, {:.8}]; R/J={}/{}, residual_norm={:.3e}",
            jacobian[(0, 0)],
            result.x[0],
            result.x[1],
            result.statistics.residual_evaluations,
            result.statistics.jacobian_evaluations,
            result.residual.norm(),
        );
        print_diagnostics(&result);
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Prepared/new Lambdify nonlinear-system guide ===");
    println!("System: x^2 - a = 0, y - b*x = 0");
    println!("Prepare once, bind many parameter vectors, and select callback execution policy.");
    run_policy(LambdifyExecutionPolicy::Sequential)?;
    run_policy(LambdifyExecutionPolicy::Parallel { min_work: 1 })?;
    Ok(())
}
