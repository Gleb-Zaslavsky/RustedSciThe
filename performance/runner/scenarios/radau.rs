use crate::error::RunnerError;
use crate::registry::{Scenario, ScenarioOutput};
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;
use RustedSciThe::numerical::ivp_workloads::{build_workload, WorkloadKind};
use RustedSciThe::numerical::Radau::{
    RadauAotConfig, RadauConfig, RadauExecution, RadauExecutionPolicy, RadauFrontend,
    RadauMatrixLayout, RadauOutputPolicy, RadauProblem, RadauSolver, RadauTelemetryMode,
};

const DIMENSION: usize = 16;
const HORIZON: f64 = 0.01;

fn artifact_root() -> Result<PathBuf, RunnerError> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("performance")
        .join("local")
        .join("aot")
        .join("radau");
    fs::create_dir_all(&path)?;
    Ok(path)
}

fn assemble_workload(
    frontend: RadauFrontend,
    layout: RadauMatrixLayout,
    execution: RadauExecution,
    policy: RadauExecutionPolicy,
    telemetry: RadauTelemetryMode,
) -> Result<(RadauSolver, Vec<f64>, Vec<f64>), RunnerError> {
    let workload = build_workload(WorkloadKind::DiffusionChain, DIMENSION);
    let variable_refs = workload
        .variables
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let jacobian = (frontend == RadauFrontend::ExprLegacy).then(|| {
        workload
            .equations
            .iter()
            .flat_map(|equation| {
                variable_refs
                    .iter()
                    .map(move |variable| equation.diff(variable))
            })
            .collect()
    });
    let mut problem = RadauProblem::new(
        workload.equations,
        workload.variables,
        workload.time_variable,
    )
    .with_parameters(workload.parameter_names);
    if let Some(jacobian) = jacobian {
        problem = problem.with_jacobian(jacobian);
    }
    let aot = if execution == RadauExecution::Aot {
        Some(RadauAotConfig::build_if_missing_release(artifact_root()?).with_c_compiler("tcc"))
    } else {
        None
    };
    let config = RadauConfig {
        t_bound: HORIZON,
        first_step: Some(HORIZON * 0.1),
        max_step: HORIZON * 0.25,
        rtol: 1.0e-7,
        atol: 1.0e-10,
        max_steps: 2_000,
        max_newton_iterations: 8,
        max_retries: 24,
        execution,
        frontend,
        matrix_layout: layout,
        telemetry,
        execution_policy: policy,
        output: RadauOutputPolicy::FinalOnly,
        aot,
        ..RadauConfig::default()
    };
    let solver = RadauSolver::prepare(problem, config)
        .map_err(|error| RunnerError::Scenario(format!("Radau preparation: {error}")))?;
    Ok((
        solver,
        workload.initial_state.as_slice().to_vec(),
        workload.parameter_values.as_slice().to_vec(),
    ))
}

fn solve_workload(
    frontend: RadauFrontend,
    layout: RadauMatrixLayout,
    execution: RadauExecution,
    policy: RadauExecutionPolicy,
    diagnostic_only: bool,
) -> Result<ScenarioOutput, RunnerError> {
    let full_started = Instant::now();
    let setup_started = Instant::now();
    let (mut solver, initial_state, parameters) =
        assemble_workload(frontend, layout, execution, policy, RadauTelemetryMode::Off)?;
    let solver_setup_ms = setup_started.elapsed().as_secs_f64() * 1_000.0;

    let solve_started = Instant::now();
    let solution = solver
        .solve_with_parameters(&initial_state, &parameters)
        .map_err(|error| RunnerError::Scenario(format!("Radau solve: {error}")))?;
    let solve_ms = solve_started.elapsed().as_secs_f64() * 1_000.0;
    let scenario_wall_ms = full_started.elapsed().as_secs_f64() * 1_000.0;

    validate_solution(&solution.y, solution.t, solution.accepted_steps)?;
    let max_abs = solution
        .y
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    black_box(&solution.y);

    if !diagnostic_only {
        return Ok(ScenarioOutput {
            metrics: vec![
                ("solver_setup_ms".into(), "ms/off".into(), solver_setup_ms),
                ("solve_ms".into(), "ms/off".into(), solve_ms),
                ("scenario_wall_ms".into(), "ms/off".into(), scenario_wall_ms),
            ],
            native_scopes: Vec::new(),
            counters: Vec::new(),
            diagnostics: vec![
                (
                    "accepted_steps".into(),
                    "count".into(),
                    solution.accepted_steps as f64,
                ),
                (
                    "rejected_steps".into(),
                    "count".into(),
                    solution.rejected_steps as f64,
                ),
                ("max_abs_state".into(), "state".into(), max_abs),
            ],
        });
    }

    // Stage timings are collected by a separate instrumented solve. They are
    // diagnostics only and never contaminate the telemetry-Off solve timer.
    let (mut diagnostic_solver, diagnostic_y0, diagnostic_parameters) = assemble_workload(
        frontend,
        layout,
        execution,
        policy,
        RadauTelemetryMode::Timings,
    )?;
    let diagnostic_solution = diagnostic_solver
        .solve_with_parameters(&diagnostic_y0, &diagnostic_parameters)
        .map_err(|error| RunnerError::Scenario(format!("Radau timed diagnostic solve: {error}")))?;
    validate_solution(
        &diagnostic_solution.y,
        diagnostic_solution.t,
        diagnostic_solution.accepted_steps,
    )?;
    let telemetry_state_max_diff = solution
        .y
        .iter()
        .zip(&diagnostic_solution.y)
        .map(|(plain, timed)| (plain - timed).abs())
        .fold(0.0, f64::max);
    if solution.accepted_steps != diagnostic_solution.accepted_steps
        || solution.rejected_steps != diagnostic_solution.rejected_steps
        || telemetry_state_max_diff > 1.0e-12
    {
        return Err(RunnerError::Scenario(format!(
            "Radau telemetry changed the trajectory: accepted {} vs {}, rejected {} vs {}, max state diff={telemetry_state_max_diff:e}",
            solution.accepted_steps,
            diagnostic_solution.accepted_steps,
            solution.rejected_steps,
            diagnostic_solution.rejected_steps
        )));
    }
    let telemetry = diagnostic_solution.telemetry();

    let mut metrics = Vec::new();
    for (stage, elapsed_ms) in &telemetry.timings_ms {
        // Radau reports zero-initialized entries for stages that did not run.
        // Omit them rather than presenting absence as a measured 0 ms.
        if elapsed_ms.is_finite() && *elapsed_ms > 0.0 {
            metrics.push((
                format!("stage_{stage}"),
                "ms/instrumented".into(),
                *elapsed_ms,
            ));
        }
    }
    let native_scopes = metrics
        .iter()
        .filter_map(|(metric_id, _, _)| {
            let key = metric_id.strip_prefix("stage_")?;
            let scope = telemetry.timing_scopes.get(key)?;
            let kind = match scope.kind {
                RustedSciThe::numerical::Radau::api::RadauTelemetryScopeKind::Inclusive => {
                    "inclusive"
                }
                RustedSciThe::numerical::Radau::api::RadauTelemetryScopeKind::Child => "child",
                RustedSciThe::numerical::Radau::api::RadauTelemetryScopeKind::Standalone => {
                    "standalone"
                }
            };
            Some(crate::registry::NativeMetricScope {
                metric_id: metric_id.clone(),
                stage: key.to_owned(),
                kind: kind.into(),
                parent: scope.parent.map(str::to_owned),
            })
        })
        .collect();

    Ok(ScenarioOutput {
        metrics,
        native_scopes,
        counters: vec![
            ("state_dimension".into(), solution.y.len() as u64),
            ("accepted_steps".into(), solution.accepted_steps as u64),
            ("rejected_steps".into(), solution.rejected_steps as u64),
            ("attempts".into(), solution.attempts as u64),
        ],
        diagnostics: vec![
            ("final_time".into(), "time".into(), solution.t),
            ("max_abs_state".into(), "state".into(), max_abs),
            (
                "telemetry_state_max_diff".into(),
                "absolute".into(),
                telemetry_state_max_diff,
            ),
        ],
    })
}

fn validate_solution(y: &[f64], time: f64, accepted_steps: usize) -> Result<(), RunnerError> {
    if !time.is_finite()
        || (time - HORIZON).abs() > HORIZON * 1.0e-10
        || y.len() != DIMENSION
        || y.iter().any(|value| !value.is_finite())
        || accepted_steps == 0
    {
        return Err(RunnerError::Scenario(format!(
            "Radau diffusion route returned invalid result: t={time}, n={}, accepted_steps={accepted_steps}",
            y.len()
        )));
    }
    Ok(())
}

pub fn run_robertson() -> Result<ScenarioOutput, RunnerError> {
    solve_robertson()
}

fn solve_robertson() -> Result<ScenarioOutput, RunnerError> {
    let full_started = Instant::now();
    let setup_started = Instant::now();
    let workload = build_workload(WorkloadKind::Robertson, 3);
    let variable_refs = workload
        .variables
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let jacobian = workload
        .equations
        .iter()
        .flat_map(|equation| {
            variable_refs
                .iter()
                .map(move |variable| equation.diff(variable))
        })
        .collect();
    let problem = RadauProblem::new(
        workload.equations,
        workload.variables,
        workload.time_variable,
    )
    .with_parameters(workload.parameter_names)
    .with_jacobian(jacobian);
    let horizon = 0.002;
    let config = RadauConfig {
        t_bound: horizon,
        first_step: Some(horizon * 0.1),
        max_step: horizon * 0.25,
        rtol: 1.0e-7,
        atol: 1.0e-10,
        max_steps: 2_000,
        max_newton_iterations: 8,
        max_retries: 24,
        execution: RadauExecution::Lambdify,
        frontend: RadauFrontend::ExprLegacy,
        matrix_layout: RadauMatrixLayout::Dense,
        telemetry: RadauTelemetryMode::Off,
        execution_policy: RadauExecutionPolicy::Sequential,
        output: RadauOutputPolicy::FinalOnly,
        ..RadauConfig::default()
    };
    let mut solver = RadauSolver::prepare(problem, config)
        .map_err(|error| RunnerError::Scenario(format!("Radau preparation: {error}")))?;
    let setup_ms = setup_started.elapsed().as_secs_f64() * 1_000.0;
    let solve_started = Instant::now();
    let solution = solver
        .solve_with_parameters(
            workload.initial_state.as_slice(),
            workload.parameter_values.as_slice(),
        )
        .map_err(|error| RunnerError::Scenario(format!("Radau Robertson solve: {error}")))?;
    let solve_ms = solve_started.elapsed().as_secs_f64() * 1_000.0;
    let scenario_wall_ms = full_started.elapsed().as_secs_f64() * 1_000.0;
    if (solution.t - horizon).abs() > horizon * 1.0e-10
        || solution.y.len() != 3
        || solution.y.iter().any(|value| !value.is_finite())
        || solution.accepted_steps == 0
    {
        return Err(RunnerError::Scenario(format!(
            "Radau Robertson returned invalid result: t={}, n={}, accepted_steps={}",
            solution.t,
            solution.y.len(),
            solution.accepted_steps
        )));
    }
    let mass_error = (solution.y.iter().sum::<f64>() - 1.0).abs();
    if mass_error > 1.0e-6 {
        return Err(RunnerError::Scenario(format!(
            "Radau Robertson mass invariant failed: {mass_error:e}"
        )));
    }
    black_box(&solution.y);
    Ok(ScenarioOutput {
        metrics: vec![
            ("solver_setup_ms".into(), "ms".into(), setup_ms),
            ("solve_ms".into(), "ms".into(), solve_ms),
            ("scenario_wall_ms".into(), "ms".into(), scenario_wall_ms),
        ],
        native_scopes: Vec::new(),
        counters: vec![
            ("state_dimension".into(), solution.y.len() as u64),
            ("accepted_steps".into(), solution.accepted_steps as u64),
            ("rejected_steps".into(), solution.rejected_steps as u64),
            ("attempts".into(), solution.attempts as u64),
        ],
        diagnostics: vec![
            ("final_time".into(), "time".into(), solution.t),
            (
                "mass_invariant_abs_error".into(),
                "absolute".into(),
                mass_error,
            ),
        ],
    })
}

macro_rules! declare_route {
    ($name:ident, $frontend:expr, $layout:expr, $execution:expr, $policy:expr) => {
        pub fn $name() -> Result<ScenarioOutput, RunnerError> {
            solve_workload($frontend, $layout, $execution, $policy, false)
        }
    };
}

pub fn diagnose(scenario: &Scenario) -> Result<ScenarioOutput, RunnerError> {
    let frontend = match scenario.frontend {
        "ExprLegacy" => RadauFrontend::ExprLegacy,
        "AtomView" => RadauFrontend::AtomViewNative,
        other => {
            return Err(RunnerError::Scenario(format!(
                "unsupported Radau frontend {other}"
            )));
        }
    };
    let layout = match scenario.linear_algebra {
        "Dense" => RadauMatrixLayout::Dense,
        "Sparse" => RadauMatrixLayout::Sparse,
        "Banded(1,1)" => RadauMatrixLayout::Banded { lower: 1, upper: 1 },
        other => {
            return Err(RunnerError::Scenario(format!(
                "unsupported Radau layout {other}"
            )));
        }
    };
    let execution = match scenario.execution {
        "Lambdify" => RadauExecution::Lambdify,
        "AOT(warm-cache)" => RadauExecution::Aot,
        other => {
            return Err(RunnerError::Scenario(format!(
                "unsupported Radau execution {other}"
            )));
        }
    };
    let policy = execution_policy(scenario.parallel_policy)?;
    solve_workload(frontend, layout, execution, policy, true)
}

fn execution_policy(label: &str) -> Result<RadauExecutionPolicy, RunnerError> {
    match label {
        "Sequential" => Ok(RadauExecutionPolicy::Sequential),
        // The registry uses the concise label; the explicit form is retained
        // for callers that want the threshold visible in scenario metadata.
        "Parallel" | "Parallel(min_work=1)" => Ok(RadauExecutionPolicy::Parallel { min_work: 1 }),
        other => Err(RunnerError::Scenario(format!(
            "unsupported Radau policy {other}"
        ))),
    }
}

// These exports are intentionally explicit: the registry uses stable scenario
// IDs while each wrapper fixes exactly one supported comparison route.
declare_route!(
    lambdify_expr_dense_seq,
    RadauFrontend::ExprLegacy,
    RadauMatrixLayout::Dense,
    RadauExecution::Lambdify,
    RadauExecutionPolicy::Sequential
);
declare_route!(
    lambdify_expr_sparse_seq,
    RadauFrontend::ExprLegacy,
    RadauMatrixLayout::Sparse,
    RadauExecution::Lambdify,
    RadauExecutionPolicy::Sequential
);
declare_route!(
    lambdify_expr_banded_seq,
    RadauFrontend::ExprLegacy,
    RadauMatrixLayout::Banded { lower: 1, upper: 1 },
    RadauExecution::Lambdify,
    RadauExecutionPolicy::Sequential
);
declare_route!(
    lambdify_atom_dense_seq,
    RadauFrontend::AtomViewNative,
    RadauMatrixLayout::Dense,
    RadauExecution::Lambdify,
    RadauExecutionPolicy::Sequential
);
declare_route!(
    lambdify_atom_sparse_seq,
    RadauFrontend::AtomViewNative,
    RadauMatrixLayout::Sparse,
    RadauExecution::Lambdify,
    RadauExecutionPolicy::Sequential
);
declare_route!(
    lambdify_atom_banded_seq,
    RadauFrontend::AtomViewNative,
    RadauMatrixLayout::Banded { lower: 1, upper: 1 },
    RadauExecution::Lambdify,
    RadauExecutionPolicy::Sequential
);
declare_route!(
    lambdify_expr_dense_par,
    RadauFrontend::ExprLegacy,
    RadauMatrixLayout::Dense,
    RadauExecution::Lambdify,
    RadauExecutionPolicy::Parallel { min_work: 1 }
);

declare_route!(
    lambdify_expr_sparse_par,
    RadauFrontend::ExprLegacy,
    RadauMatrixLayout::Sparse,
    RadauExecution::Lambdify,
    RadauExecutionPolicy::Parallel { min_work: 1 }
);
declare_route!(
    lambdify_expr_banded_par,
    RadauFrontend::ExprLegacy,
    RadauMatrixLayout::Banded { lower: 1, upper: 1 },
    RadauExecution::Lambdify,
    RadauExecutionPolicy::Parallel { min_work: 1 }
);
declare_route!(
    lambdify_atom_dense_par,
    RadauFrontend::AtomViewNative,
    RadauMatrixLayout::Dense,
    RadauExecution::Lambdify,
    RadauExecutionPolicy::Parallel { min_work: 1 }
);
declare_route!(
    lambdify_atom_sparse_par,
    RadauFrontend::AtomViewNative,
    RadauMatrixLayout::Sparse,
    RadauExecution::Lambdify,
    RadauExecutionPolicy::Parallel { min_work: 1 }
);
declare_route!(
    lambdify_atom_banded_par,
    RadauFrontend::AtomViewNative,
    RadauMatrixLayout::Banded { lower: 1, upper: 1 },
    RadauExecution::Lambdify,
    RadauExecutionPolicy::Parallel { min_work: 1 }
);

declare_route!(
    aot_expr_dense_seq,
    RadauFrontend::ExprLegacy,
    RadauMatrixLayout::Dense,
    RadauExecution::Aot,
    RadauExecutionPolicy::Sequential
);
declare_route!(
    aot_expr_sparse_seq,
    RadauFrontend::ExprLegacy,
    RadauMatrixLayout::Sparse,
    RadauExecution::Aot,
    RadauExecutionPolicy::Sequential
);
declare_route!(
    aot_expr_banded_seq,
    RadauFrontend::ExprLegacy,
    RadauMatrixLayout::Banded { lower: 1, upper: 1 },
    RadauExecution::Aot,
    RadauExecutionPolicy::Sequential
);
declare_route!(
    aot_atom_dense_seq,
    RadauFrontend::AtomViewNative,
    RadauMatrixLayout::Dense,
    RadauExecution::Aot,
    RadauExecutionPolicy::Sequential
);
declare_route!(
    aot_atom_sparse_seq,
    RadauFrontend::AtomViewNative,
    RadauMatrixLayout::Sparse,
    RadauExecution::Aot,
    RadauExecutionPolicy::Sequential
);
declare_route!(
    aot_atom_banded_seq,
    RadauFrontend::AtomViewNative,
    RadauMatrixLayout::Banded { lower: 1, upper: 1 },
    RadauExecution::Aot,
    RadauExecutionPolicy::Sequential
);
declare_route!(
    aot_expr_dense_par,
    RadauFrontend::ExprLegacy,
    RadauMatrixLayout::Dense,
    RadauExecution::Aot,
    RadauExecutionPolicy::Parallel { min_work: 1 }
);
declare_route!(
    aot_expr_sparse_par,
    RadauFrontend::ExprLegacy,
    RadauMatrixLayout::Sparse,
    RadauExecution::Aot,
    RadauExecutionPolicy::Parallel { min_work: 1 }
);
declare_route!(
    aot_expr_banded_par,
    RadauFrontend::ExprLegacy,
    RadauMatrixLayout::Banded { lower: 1, upper: 1 },
    RadauExecution::Aot,
    RadauExecutionPolicy::Parallel { min_work: 1 }
);
declare_route!(
    aot_atom_dense_par,
    RadauFrontend::AtomViewNative,
    RadauMatrixLayout::Dense,
    RadauExecution::Aot,
    RadauExecutionPolicy::Parallel { min_work: 1 }
);
declare_route!(
    aot_atom_sparse_par,
    RadauFrontend::AtomViewNative,
    RadauMatrixLayout::Sparse,
    RadauExecution::Aot,
    RadauExecutionPolicy::Parallel { min_work: 1 }
);
declare_route!(
    aot_atom_banded_par,
    RadauFrontend::AtomViewNative,
    RadauMatrixLayout::Banded { lower: 1, upper: 1 },
    RadauExecution::Aot,
    RadauExecutionPolicy::Parallel { min_work: 1 }
);

#[cfg(test)]
mod tests {
    use super::execution_policy;
    use RustedSciThe::numerical::Radau::RadauExecutionPolicy;

    #[test]
    fn registry_parallel_label_resolves_to_the_route_threshold() {
        assert_eq!(
            execution_policy("Parallel").unwrap(),
            RadauExecutionPolicy::Parallel { min_work: 1 }
        );
        assert_eq!(
            execution_policy("Parallel(min_work=1)").unwrap(),
            RadauExecutionPolicy::Parallel { min_work: 1 }
        );
        assert_eq!(
            execution_policy("Sequential").unwrap(),
            RadauExecutionPolicy::Sequential
        );
        assert!(execution_policy("Auto").is_err());
    }
}
