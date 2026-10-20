use crate::error::RunnerError;
use crate::registry::{Scenario, ScenarioOutput};
use RustedSciThe::numerical::LSODE2::{
    IvpColdStage, IvpLambdifyExecutionPolicy, IvpTelemetry, Lsode2AotProfile, Lsode2AotToolchain,
    Lsode2ControllerConfig, Lsode2LinearSystemStructure, Lsode2ProblemConfig,
    Lsode2ResidualJacobianSource, Lsode2Solver, Lsode2SymbolicAssemblyBackend,
    Lsode2SymbolicExecutionMode,
};
use RustedSciThe::numerical::ivp_workloads::{WorkloadKind, build_workload};
use RustedSciThe::symbolic::symbolic_ivp_generated::{
    SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
};
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

const DIMENSION: usize = 128;
const T_BOUND: f64 = 0.01;

/// Keep the original small Robertson case as a stable, inexpensive sanity gate.
pub fn run_robertson() -> Result<ScenarioOutput, RunnerError> {
    let full_started = Instant::now();
    let workload = build_workload(WorkloadKind::Robertson, 3);
    let config = Lsode2ProblemConfig::new(
        workload.equations,
        workload.variables,
        workload.time_variable,
        0.0,
        workload.initial_state,
        0.25,
        0.02,
        1.0e-7,
        1.0e-9,
    )
    .with_controller(Lsode2ControllerConfig::bdf_only())
    .with_faithful_bdf_solve(200_000, 200_000)
    .with_telemetry(IvpTelemetry::disabled());

    let setup_started = Instant::now();
    let mut solver = Lsode2Solver::new(config).map_err(|error| {
        RunnerError::Scenario(format!("LSODE2 Robertson construction: {error}"))
    })?;
    let setup_ms = setup_started.elapsed().as_secs_f64() * 1_000.0;
    let prepare_started = Instant::now();
    solver
        .prepare()
        .map_err(|error| RunnerError::Scenario(format!("LSODE2 Robertson prepare: {error}")))?;
    let prepare_ms = prepare_started.elapsed().as_secs_f64() * 1_000.0;
    let solve_started = Instant::now();
    let solution = solver
        .solve_with_summary()
        .map_err(|error| RunnerError::Scenario(format!("LSODE2 Robertson solve: {error}")))?;
    let solve_ms = solve_started.elapsed().as_secs_f64() * 1_000.0;
    let scenario_wall_ms = full_started.elapsed().as_secs_f64() * 1_000.0;

    if !matches!(
        solution.status.as_str(),
        "finished" | "finished_native_faithful" | "finished_native_faithful_partial"
    ) || solution
        .final_t
        .is_none_or(|time| (time - 0.25).abs() > 1.0e-10)
        || solution.variable_count != 3
        || solution
            .final_y
            .as_ref()
            .is_none_or(|state| state.len() != 3 || state.iter().any(|value| !value.is_finite()))
    {
        return Err(RunnerError::Scenario(format!(
            "LSODE2 Robertson returned invalid result: status={}, final_t={:?}, n={}",
            solution.status, solution.final_t, solution.variable_count
        )));
    }
    let final_state = solution.final_y.as_ref().expect("checked final state");
    let mass_error = (final_state.iter().sum::<f64>() - 1.0).abs();
    if mass_error > 1.0e-5 {
        return Err(RunnerError::Scenario(format!(
            "LSODE2 Robertson mass invariant failed: {mass_error:e}"
        )));
    }
    black_box(final_state);

    Ok(ScenarioOutput {
        metrics: vec![
            ("solver_setup_ms".into(), "ms".into(), setup_ms),
            ("prepare_ms".into(), "ms".into(), prepare_ms),
            ("solve_ms".into(), "ms".into(), solve_ms),
            ("scenario_wall_ms".into(), "ms".into(), scenario_wall_ms),
        ],
        native_scopes: Vec::new(),
        counters: vec![
            ("state_dimension".into(), solution.variable_count as u64),
            ("trajectory_points".into(), solution.time_points as u64),
        ],
        diagnostics: vec![
            (
                "final_time".into(),
                "time".into(),
                solution.final_t.unwrap(),
            ),
            (
                "mass_invariant_abs_error".into(),
                "absolute".into(),
                mass_error,
            ),
        ],
    })
}

#[derive(Clone, Copy)]
enum MatrixRoute {
    Dense,
    Sparse,
    Banded,
}

#[derive(Clone, Copy)]
enum ExecutionRoute {
    LambdifySequential,
    LambdifyParallel,
    LambdifyAuto,
    AotWhole,
    AotParallel2,
}

impl ExecutionRoute {
    fn label(self) -> &'static str {
        match self {
            Self::LambdifySequential => "Lambdify / Sequential",
            Self::LambdifyParallel => "Lambdify / Parallel",
            Self::LambdifyAuto => "Lambdify / Auto",
            Self::AotWhole => "AOT / Whole",
            Self::AotParallel2 => "AOT / Parallel2",
        }
    }

    fn is_aot(self) -> bool {
        matches!(self, Self::AotWhole | Self::AotParallel2)
    }

    fn lambdify_policy(self) -> IvpLambdifyExecutionPolicy {
        match self {
            Self::LambdifySequential | Self::AotWhole | Self::AotParallel2 => {
                IvpLambdifyExecutionPolicy::Sequential
            }
            Self::LambdifyParallel => IvpLambdifyExecutionPolicy::Parallel { min_work: 1 },
            Self::LambdifyAuto => IvpLambdifyExecutionPolicy::Auto { min_work: 1 },
        }
    }

    fn symbolic_execution(self) -> Lsode2SymbolicExecutionMode {
        if self.is_aot() {
            Lsode2SymbolicExecutionMode::Aot {
                toolchain: Lsode2AotToolchain::CTcc,
                profile: Lsode2AotProfile::Release,
            }
        } else {
            Lsode2SymbolicExecutionMode::LambdifyExpr
        }
    }
}

fn artifact_root() -> Result<PathBuf, RunnerError> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("performance")
        .join("local")
        .join("aot")
        .join("lsode2");
    fs::create_dir_all(&path)?;
    Ok(path)
}

fn generated_backend() -> Result<SymbolicIvpGeneratedBackendConfig, RunnerError> {
    Ok(SymbolicIvpGeneratedBackendConfig::new()
        .with_output_parent_dir(Some(artifact_root()?))
        .with_build_policy(SymbolicIvpAotBuildPolicy::BuildIfMissing {
            profile: RustedSciThe::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile::Release,
        })
        .with_c_tcc())
}

fn make_config(
    assembly: Lsode2SymbolicAssemblyBackend,
    matrix: MatrixRoute,
    execution: ExecutionRoute,
    telemetry: IvpTelemetry,
) -> Result<Lsode2ProblemConfig, RunnerError> {
    let workload = build_workload(WorkloadKind::DiffusionChain, DIMENSION);
    let mut config = Lsode2ProblemConfig::new(
        workload.equations,
        workload.variables,
        workload.time_variable,
        0.0,
        workload.initial_state,
        T_BOUND,
        0.001,
        1.0e-7,
        1.0e-10,
    )
    .with_controller(Lsode2ControllerConfig::bdf_only())
    .with_faithful_bdf_solve(200_000, 200_000)
    .with_lambdify_execution_policy(execution.lambdify_policy())
    .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
        assembly,
        execution: execution.symbolic_execution(),
    })
    .with_telemetry(telemetry);

    if !workload.parameter_names.is_empty() {
        config = config
            .with_equation_parameters(workload.parameter_names)
            .with_equation_parameter_values(workload.parameter_values);
    }

    config = match matrix {
        MatrixRoute::Dense => config.with_dense_symbolic_backend(),
        MatrixRoute::Sparse => config.with_native_sparse_faer_backend(),
        MatrixRoute::Banded => config
            .with_native_banded_faithful_backend()
            .with_linear_system_structure(Lsode2LinearSystemStructure::Banded { kl: 1, ku: 1 }),
    };

    if execution.is_aot() {
        let artifact_dir = artifact_root()?;
        config = match matrix {
            MatrixRoute::Dense => config.with_dense_aot_c_tcc(&artifact_dir),
            MatrixRoute::Sparse => {
                config.with_native_sparse_faer_generated_backend(generated_backend()?)
            }
            MatrixRoute::Banded => {
                config.with_native_banded_faithful_generated_backend(generated_backend()?)
            }
        };
        if matches!(execution, ExecutionRoute::AotParallel2) {
            config = config.with_aot_parallel_chunking(2);
        }
    }
    Ok(config)
}

fn solve_route(
    assembly: Lsode2SymbolicAssemblyBackend,
    matrix: MatrixRoute,
    execution: ExecutionRoute,
    diagnostic_only: bool,
) -> Result<ScenarioOutput, RunnerError> {
    let full_started = Instant::now();
    let setup_started = Instant::now();
    let mut solver = Lsode2Solver::new(make_config(
        assembly,
        matrix,
        execution,
        IvpTelemetry::disabled(),
    )?)
    .map_err(|error| RunnerError::Scenario(format!("LSODE2 construction: {error}")))?;
    let setup_ms = setup_started.elapsed().as_secs_f64() * 1_000.0;

    let prepare_started = Instant::now();
    solver
        .prepare()
        .map_err(|error| RunnerError::Scenario(format!("LSODE2 prepare: {error}")))?;
    let prepare_ms = prepare_started.elapsed().as_secs_f64() * 1_000.0;
    let solve_started = Instant::now();
    let solution = solver
        .solve_with_summary()
        .map_err(|error| RunnerError::Scenario(format!("LSODE2 solve: {error}")))?;
    let solve_ms = solve_started.elapsed().as_secs_f64() * 1_000.0;
    let scenario_wall_ms = full_started.elapsed().as_secs_f64() * 1_000.0;
    validate_solution(&solution, execution)?;
    let final_state = solution
        .final_y
        .as_ref()
        .expect("validated LSODE2 summary has final state");
    black_box(final_state);

    if !diagnostic_only {
        return Ok(ScenarioOutput {
            metrics: vec![
                ("solver_setup_ms".into(), "ms/off".into(), setup_ms),
                ("prepare_ms".into(), "ms/off".into(), prepare_ms),
                ("solve_ms".into(), "ms/off".into(), solve_ms),
                ("scenario_wall_ms".into(), "ms/off".into(), scenario_wall_ms),
            ],
            native_scopes: Vec::new(),
            counters: Vec::new(),
            diagnostics: vec![
                (
                    "accepted_steps".into(),
                    "count".into(),
                    solution.evaluation_telemetry.accepted_steps as f64,
                ),
                (
                    "final_state_min".into(),
                    "state".into(),
                    final_state.iter().copied().fold(f64::INFINITY, f64::min),
                ),
            ],
        });
    }

    // Stage/counter data comes from a separate instrumented solve so the gate
    // remains an uninstrumented full-lifecycle measurement.
    let mut diagnostic = Lsode2Solver::new(make_config(
        assembly,
        matrix,
        execution,
        IvpTelemetry::detailed(),
    )?)
    .map_err(|error| RunnerError::Scenario(format!("LSODE2 diagnostic construction: {error}")))?;
    diagnostic
        .prepare()
        .map_err(|error| RunnerError::Scenario(format!("LSODE2 diagnostic prepare: {error}")))?;
    let diagnostic_solution = diagnostic
        .solve_with_summary()
        .map_err(|error| RunnerError::Scenario(format!("LSODE2 diagnostic solve: {error}")))?;
    validate_solution(&diagnostic_solution, execution)?;
    let diagnostic_state = diagnostic_solution
        .final_y
        .as_ref()
        .expect("validated diagnostic summary has final state");
    let trajectory_diff = final_state
        .iter()
        .zip(diagnostic_state.iter())
        .map(|(plain, timed)| (plain - timed).abs())
        .fold(0.0_f64, f64::max);
    let snapshot = diagnostic.telemetry_snapshot();
    let evaluation = diagnostic_solution.evaluation_telemetry;
    if trajectory_diff > 1.0e-11
        || solution.status != diagnostic_solution.status
        || evaluation.accepted_steps == 0
    {
        return Err(RunnerError::Scenario(format!(
            "LSODE2 telemetry changed route behavior: status={}/{}, accepted={}, trajectory diff={trajectory_diff:e}",
            solution.status, diagnostic_solution.status, evaluation.accepted_steps
        )));
    }
    if execution.is_aot()
        && (snapshot.execution != RustedSciThe::symbolic::ivp_telemetry::IvpTelemetryExecution::Aot
            || snapshot.aot_runtime_ready == 0)
    {
        return Err(RunnerError::Scenario(format!(
            "LSODE2 {} did not publish a ready AOT runtime",
            execution.label()
        )));
    }

    let mut metrics = vec![
        (
            "stage_residual_ms".into(),
            "ms/instrumented".into(),
            evaluation.residual_ms_total,
        ),
        (
            "stage_jacobian_ms".into(),
            "ms/instrumented".into(),
            evaluation.jacobian_ms_total,
        ),
        (
            "stage_linear_ms".into(),
            "ms/instrumented".into(),
            evaluation.linear_ms_total,
        ),
    ];
    for stage in [
        IvpColdStage::SolverPreparation,
        IvpColdStage::SymbolicJacobian,
        IvpColdStage::AtomJacobianPreparation,
        IvpColdStage::AotLowering,
        IvpColdStage::AotBuild,
        IvpColdStage::AotLink,
    ] {
        let timing = snapshot.cold_stage(stage);
        if timing.recorded {
            metrics.push((
                format!("stage_cold_{stage:?}_ms"),
                "ms/instrumented".into(),
                timing.elapsed.as_secs_f64() * 1_000.0,
            ));
        }
    }

    Ok(ScenarioOutput {
        metrics,
        native_scopes: Vec::new(),
        counters: vec![
            ("state_dimension".into(), solution.variable_count as u64),
            ("trajectory_points".into(), solution.time_points as u64),
            ("accepted_steps".into(), evaluation.accepted_steps as u64),
            ("rejected_steps".into(), evaluation.rejected_steps as u64),
            (
                "residual_evaluations".into(),
                evaluation.residual_evaluations as u64,
            ),
            (
                "jacobian_evaluations".into(),
                evaluation.jacobian_evaluations as u64,
            ),
            ("linear_solves".into(), evaluation.linear_solves as u64),
            (
                "aot_parallel_dispatches".into(),
                snapshot.aot_parallel_dispatches,
            ),
            (
                "lambdify_parallel_dispatches".into(),
                snapshot.parallel_dispatches,
            ),
        ],
        diagnostics: vec![
            (
                "final_time".into(),
                "time".into(),
                solution.final_t.unwrap(),
            ),
            (
                "max_abs_state".into(),
                "state".into(),
                solution.max_abs_solution,
            ),
            (
                "telemetry_trajectory_max_diff".into(),
                "absolute".into(),
                trajectory_diff,
            ),
        ],
    })
}

fn validate_solution(
    summary: &RustedSciThe::numerical::LSODE2::Lsode2SolveSummary,
    execution: ExecutionRoute,
) -> Result<(), RunnerError> {
    if !matches!(
        summary.status.as_str(),
        "finished" | "finished_native_faithful" | "finished_native_faithful_partial"
    ) || summary
        .final_t
        .is_none_or(|time| (time - T_BOUND).abs() > 1.0e-10)
        || summary.variable_count != DIMENSION
        || summary.final_y.as_ref().is_none_or(|state| {
            state.len() != DIMENSION || state.iter().any(|value| !value.is_finite())
        })
    {
        return Err(RunnerError::Scenario(format!(
            "LSODE2 {} returned invalid result: status={}, final_t={:?}, n={}",
            execution.label(),
            summary.status,
            summary.final_t,
            summary.variable_count
        )));
    }
    let state = summary.final_y.as_ref().expect("checked final state");
    let minimum = state.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = state.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if minimum < -1.0e-8 || maximum > 2.0 + 1.0e-8 {
        return Err(RunnerError::Scenario(format!(
            "LSODE2 diffusion maximum-principle bounds failed: min={minimum:e}, max={maximum:e}"
        )));
    }
    Ok(())
}

macro_rules! declare_route {
    ($name:ident, $assembly:expr, $matrix:expr, $execution:expr) => {
        pub fn $name() -> Result<ScenarioOutput, RunnerError> {
            solve_route($assembly, $matrix, $execution, false)
        }
    };
}

pub fn diagnose(scenario: &Scenario) -> Result<ScenarioOutput, RunnerError> {
    let assembly = match scenario.frontend {
        "ExprLegacy" => Lsode2SymbolicAssemblyBackend::ExprLegacy,
        "AtomView" => Lsode2SymbolicAssemblyBackend::AtomView,
        other => {
            return Err(RunnerError::Scenario(format!(
                "unsupported LSODE2 frontend {other}"
            )));
        }
    };
    let matrix = match scenario.linear_algebra {
        "Dense" => MatrixRoute::Dense,
        "Sparse" => MatrixRoute::Sparse,
        "Banded(1,1)" => MatrixRoute::Banded,
        other => {
            return Err(RunnerError::Scenario(format!(
                "unsupported LSODE2 layout {other}"
            )));
        }
    };
    let execution =
        execution_route(scenario.execution, scenario.parallel_policy).ok_or_else(|| {
            RunnerError::Scenario(format!("unsupported LSODE2 route {}", scenario.id))
        })?;
    solve_route(assembly, matrix, execution, true)
}

fn execution_route(execution: &str, policy: &str) -> Option<ExecutionRoute> {
    match (execution, policy) {
        ("Lambdify", "Sequential") => Some(ExecutionRoute::LambdifySequential),
        ("Lambdify", "Parallel(min_work=1)") => Some(ExecutionRoute::LambdifyParallel),
        ("Lambdify", "Auto(min_work=1)") => Some(ExecutionRoute::LambdifyAuto),
        ("AOT(warm-cache)", "Whole(default chunks)") => Some(ExecutionRoute::AotWhole),
        ("AOT(warm-cache)", "Parallel2") => Some(ExecutionRoute::AotParallel2),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_registered_lsode2_matrix_policy_has_an_execution_route() {
        let scenarios = crate::registry::SCENARIOS.iter().filter(|scenario| {
            scenario.solver == "lsode2" && scenario.groups.contains(&"lsode2-matrix")
        });

        let mut count = 0;
        for scenario in scenarios {
            count += 1;
            assert!(
                execution_route(scenario.execution, scenario.parallel_policy).is_some(),
                "unhandled LSODE2 execution metadata for {}: {}/{}",
                scenario.id,
                scenario.execution,
                scenario.parallel_policy
            );
        }
        assert_eq!(count, 30, "LSODE2 route matrix changed; review its parser");
    }

    #[test]
    fn unknown_lsode2_execution_policy_is_rejected() {
        assert!(execution_route("Lambdify", "Parallel").is_none());
        assert!(execution_route("AOT(warm-cache)", "Whole").is_none());
    }
}

declare_route!(
    diff_expr_dense_lambdify_seq,
    Lsode2SymbolicAssemblyBackend::ExprLegacy,
    MatrixRoute::Dense,
    ExecutionRoute::LambdifySequential
);
declare_route!(
    diff_atom_dense_lambdify_seq,
    Lsode2SymbolicAssemblyBackend::AtomView,
    MatrixRoute::Dense,
    ExecutionRoute::LambdifySequential
);
declare_route!(
    diff_expr_sparse_lambdify_seq,
    Lsode2SymbolicAssemblyBackend::ExprLegacy,
    MatrixRoute::Sparse,
    ExecutionRoute::LambdifySequential
);
declare_route!(
    diff_atom_sparse_lambdify_seq,
    Lsode2SymbolicAssemblyBackend::AtomView,
    MatrixRoute::Sparse,
    ExecutionRoute::LambdifySequential
);
declare_route!(
    diff_expr_banded_lambdify_seq,
    Lsode2SymbolicAssemblyBackend::ExprLegacy,
    MatrixRoute::Banded,
    ExecutionRoute::LambdifySequential
);
declare_route!(
    diff_atom_banded_lambdify_seq,
    Lsode2SymbolicAssemblyBackend::AtomView,
    MatrixRoute::Banded,
    ExecutionRoute::LambdifySequential
);

declare_route!(
    diff_expr_dense_lambdify_parallel,
    Lsode2SymbolicAssemblyBackend::ExprLegacy,
    MatrixRoute::Dense,
    ExecutionRoute::LambdifyParallel
);
declare_route!(
    diff_atom_dense_lambdify_parallel,
    Lsode2SymbolicAssemblyBackend::AtomView,
    MatrixRoute::Dense,
    ExecutionRoute::LambdifyParallel
);
declare_route!(
    diff_expr_sparse_lambdify_parallel,
    Lsode2SymbolicAssemblyBackend::ExprLegacy,
    MatrixRoute::Sparse,
    ExecutionRoute::LambdifyParallel
);
declare_route!(
    diff_atom_sparse_lambdify_parallel,
    Lsode2SymbolicAssemblyBackend::AtomView,
    MatrixRoute::Sparse,
    ExecutionRoute::LambdifyParallel
);
declare_route!(
    diff_expr_banded_lambdify_parallel,
    Lsode2SymbolicAssemblyBackend::ExprLegacy,
    MatrixRoute::Banded,
    ExecutionRoute::LambdifyParallel
);
declare_route!(
    diff_atom_banded_lambdify_parallel,
    Lsode2SymbolicAssemblyBackend::AtomView,
    MatrixRoute::Banded,
    ExecutionRoute::LambdifyParallel
);

declare_route!(
    diff_expr_dense_lambdify_auto,
    Lsode2SymbolicAssemblyBackend::ExprLegacy,
    MatrixRoute::Dense,
    ExecutionRoute::LambdifyAuto
);
declare_route!(
    diff_atom_dense_lambdify_auto,
    Lsode2SymbolicAssemblyBackend::AtomView,
    MatrixRoute::Dense,
    ExecutionRoute::LambdifyAuto
);
declare_route!(
    diff_expr_sparse_lambdify_auto,
    Lsode2SymbolicAssemblyBackend::ExprLegacy,
    MatrixRoute::Sparse,
    ExecutionRoute::LambdifyAuto
);
declare_route!(
    diff_atom_sparse_lambdify_auto,
    Lsode2SymbolicAssemblyBackend::AtomView,
    MatrixRoute::Sparse,
    ExecutionRoute::LambdifyAuto
);
declare_route!(
    diff_expr_banded_lambdify_auto,
    Lsode2SymbolicAssemblyBackend::ExprLegacy,
    MatrixRoute::Banded,
    ExecutionRoute::LambdifyAuto
);
declare_route!(
    diff_atom_banded_lambdify_auto,
    Lsode2SymbolicAssemblyBackend::AtomView,
    MatrixRoute::Banded,
    ExecutionRoute::LambdifyAuto
);

declare_route!(
    diff_expr_dense_aot_whole,
    Lsode2SymbolicAssemblyBackend::ExprLegacy,
    MatrixRoute::Dense,
    ExecutionRoute::AotWhole
);
declare_route!(
    diff_atom_dense_aot_whole,
    Lsode2SymbolicAssemblyBackend::AtomView,
    MatrixRoute::Dense,
    ExecutionRoute::AotWhole
);
declare_route!(
    diff_expr_sparse_aot_whole,
    Lsode2SymbolicAssemblyBackend::ExprLegacy,
    MatrixRoute::Sparse,
    ExecutionRoute::AotWhole
);
declare_route!(
    diff_atom_sparse_aot_whole,
    Lsode2SymbolicAssemblyBackend::AtomView,
    MatrixRoute::Sparse,
    ExecutionRoute::AotWhole
);
declare_route!(
    diff_expr_banded_aot_whole,
    Lsode2SymbolicAssemblyBackend::ExprLegacy,
    MatrixRoute::Banded,
    ExecutionRoute::AotWhole
);
declare_route!(
    diff_atom_banded_aot_whole,
    Lsode2SymbolicAssemblyBackend::AtomView,
    MatrixRoute::Banded,
    ExecutionRoute::AotWhole
);

declare_route!(
    diff_expr_dense_aot_parallel2,
    Lsode2SymbolicAssemblyBackend::ExprLegacy,
    MatrixRoute::Dense,
    ExecutionRoute::AotParallel2
);
declare_route!(
    diff_atom_dense_aot_parallel2,
    Lsode2SymbolicAssemblyBackend::AtomView,
    MatrixRoute::Dense,
    ExecutionRoute::AotParallel2
);
declare_route!(
    diff_expr_sparse_aot_parallel2,
    Lsode2SymbolicAssemblyBackend::ExprLegacy,
    MatrixRoute::Sparse,
    ExecutionRoute::AotParallel2
);
declare_route!(
    diff_atom_sparse_aot_parallel2,
    Lsode2SymbolicAssemblyBackend::AtomView,
    MatrixRoute::Sparse,
    ExecutionRoute::AotParallel2
);
declare_route!(
    diff_expr_banded_aot_parallel2,
    Lsode2SymbolicAssemblyBackend::ExprLegacy,
    MatrixRoute::Banded,
    ExecutionRoute::AotParallel2
);
declare_route!(
    diff_atom_banded_aot_parallel2,
    Lsode2SymbolicAssemblyBackend::AtomView,
    MatrixRoute::Banded,
    ExecutionRoute::AotParallel2
);
