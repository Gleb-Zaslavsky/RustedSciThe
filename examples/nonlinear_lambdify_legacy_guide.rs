//! Compatibility guide for the legacy Lambdify nonlinear-system API.
//!
//! The compatibility object prepares the symbolic residual and Jacobian once,
//! then allows parameter values to be updated in place. This is useful for
//! existing applications that already own a `SymbolicNonlinearProblem`.
//!
//! Run with:
//!
//! ```text
//! cargo run --example nonlinear_lambdify_legacy_guide
//! ```

use RustedSciThe::numerical::Nonlinear_systems::prelude::{
    DampedNewtonMethod, DiagnosticsOptions, JacobianProvider, NonlinearProblem,
    NonlinearSolverMethod, SolveOptions, SolveResult, SymbolicNonlinearProblem,
    SymbolicProblemOptions,
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Legacy Lambdify nonlinear-system guide ===");
    println!("System: x^2 - a = 0, y - b*x = 0");
    println!("The symbolic Jacobian is prepared once and reused after parameter updates.");

    // The parameter order is part of the contract: [a, b] matches the values
    // passed to set_parameter_values below.
    let mut problem = SymbolicNonlinearProblem::from_strings_with_options(
        vec!["x^2 - a".to_string(), "y - b*x".to_string()],
        SymbolicProblemOptions::new()
            .with_variables(vec!["x".to_string(), "y".to_string()])
            .with_equation_parameters(vec!["a".to_string(), "b".to_string()])
            .with_equation_parameter_values(DVector::from_vec(vec![4.0, 3.0]))
            .with_lambdify_backend(),
    )?;

    println!("backend: {:?}", problem.backend_kind());
    println!("variables: {:?}", problem.variables());
    println!(
        "parameters: {:?}",
        problem.parameter_schema().map(|s| s.names())
    );
    println!(
        "reusable output path: residual={}, jacobian={}",
        problem.supports_residual_into(),
        problem.supports_jacobian_into()
    );

    let method = NonlinearSolverMethod::DampedNewton(DampedNewtonMethod::default());
    let mut residual = DVector::zeros(2);
    let mut jacobian = DMatrix::zeros(2, 2);

    for (parameters, initial) in [([4.0, 3.0], [1.0, 3.0]), ([9.0, 2.0], [2.0, 4.0])] {
        // No symbolic differentiation or lambdification happens here. Only
        // the validated numeric binding changes on the already prepared object.
        problem.set_parameter_values(DVector::from_vec(parameters.to_vec()))?;
        let x = DVector::from_vec(initial.to_vec());
        problem.residual_into(&x, &mut residual)?;
        problem.jacobian_into(&x, &mut jacobian)?;
        println!("parameters={parameters:?}; residual={residual:?}; jacobian={jacobian:?}");

        let result = method.clone().solve(&problem, x, solve_options())?;
        println!(
            "  solution=[{:.8}, {:.8}], iterations={}, R/J={}/{}, residual_norm={:.3e}",
            result.x[0],
            result.x[1],
            result.statistics.iterations,
            result.statistics.residual_evaluations,
            result.statistics.jacobian_evaluations,
            result.residual.norm(),
        );
        print_diagnostics(&result);
    }

    println!("One compatibility problem was reused for both parameter sets.");
    Ok(())
}
