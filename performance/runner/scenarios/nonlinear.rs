use crate::error::RunnerError;
use crate::registry::ScenarioOutput;
use RustedSciThe::numerical::Nonlinear_systems::prelude::{
    NewtonMethod, NonlinearSolverMethod, PreparationTelemetryMode,
    PreparedSymbolicNonlinearProblem, SolveOptions, SymbolicLambdifyFrontend,
    SymbolicProblemOptions, TerminationReason,
};
use nalgebra::DVector;
use std::hint::black_box;
use std::time::Instant;

const DIMENSION: usize = 16;

pub fn run_quadratic_chain() -> Result<ScenarioOutput, RunnerError> {
    let full_started = Instant::now();
    let target = DVector::from_iterator(
        DIMENSION,
        (0..DIMENSION).map(|index| 1.0 + index as f64 * 0.01),
    );
    let initial = &target * 0.9;
    let variables = (0..DIMENSION)
        .map(|index| format!("x{index}"))
        .collect::<Vec<_>>();
    let equations = variables
        .iter()
        .enumerate()
        .map(|(index, variable)| {
            let mut equation = format!("p0*({variable}^2-{:.17})", target[index].powi(2));
            if index > 0 {
                equation.push_str(&format!("+0.02*(x{}-{:.17})", index - 1, target[index - 1]));
            }
            if index + 1 < DIMENSION {
                equation.push_str(&format!("+0.02*(x{}-{:.17})", index + 1, target[index + 1]));
            }
            equation
        })
        .collect::<Vec<_>>();

    let prepare_started = Instant::now();
    let problem = PreparedSymbolicNonlinearProblem::from_strings(
        equations,
        SymbolicProblemOptions::new()
            .with_variables(variables)
            .with_equation_parameters(vec!["p0".into()])
            .with_equation_parameter_values(DVector::from_vec(vec![1.0]))
            .with_lambdify_frontend(SymbolicLambdifyFrontend::ExprLegacy)
            .with_preparation_telemetry(PreparationTelemetryMode::Disabled),
    )
    .map_err(|error| RunnerError::Scenario(format!("nonlinear preparation: {error}")))?;
    let preparation_ms = prepare_started.elapsed().as_secs_f64() * 1_000.0;

    let bind_started = Instant::now();
    let bound = problem
        .bind_values(DVector::from_vec(vec![1.0]))
        .map_err(|error| RunnerError::Scenario(format!("nonlinear parameter binding: {error}")))?;
    let binding_ms = bind_started.elapsed().as_secs_f64() * 1_000.0;

    let solve_started = Instant::now();
    let result = NonlinearSolverMethod::Newton(NewtonMethod)
        .solve(
            &bound,
            initial,
            SolveOptions {
                tolerance: 1.0e-10,
                max_iterations: 80,
                ..SolveOptions::default()
            },
        )
        .map_err(|error| RunnerError::Scenario(format!("nonlinear Newton solve: {error}")))?;
    let solve_ms = solve_started.elapsed().as_secs_f64() * 1_000.0;
    let scenario_wall_ms = full_started.elapsed().as_secs_f64() * 1_000.0;

    let max_solution_error = result
        .x
        .iter()
        .zip(target.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    if result.termination != TerminationReason::Converged
        || !result.residual_norm.is_finite()
        || result.residual_norm > 1.0e-8
        || max_solution_error > 1.0e-6
    {
        return Err(RunnerError::Scenario(format!(
            "nonlinear quadratic chain failed correctness: termination={:?}, residual={}, max_error={max_solution_error:e}",
            result.termination, result.residual_norm
        )));
    }
    black_box((&result.x, &result.residual));

    Ok(ScenarioOutput {
        metrics: vec![
            ("preparation_ms".into(), "ms".into(), preparation_ms),
            ("parameter_binding_ms".into(), "ms".into(), binding_ms),
            ("solve_ms".into(), "ms".into(), solve_ms),
            ("scenario_wall_ms".into(), "ms".into(), scenario_wall_ms),
        ],
        native_scopes: Vec::new(),
        counters: vec![
            ("state_dimension".into(), DIMENSION as u64),
            ("newton_iterations".into(), result.iterations as u64),
            (
                "residual_evaluations".into(),
                result.statistics.residual_evaluations as u64,
            ),
            (
                "jacobian_evaluations".into(),
                result.statistics.jacobian_evaluations as u64,
            ),
        ],
        diagnostics: vec![
            (
                "residual_norm".into(),
                "absolute".into(),
                result.residual_norm,
            ),
            (
                "max_solution_abs_error".into(),
                "absolute".into(),
                max_solution_error,
            ),
        ],
    })
}
