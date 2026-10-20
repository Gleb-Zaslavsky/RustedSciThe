//! Direct Python adapter for the established damped Newton BVP solver.
//!
//! The adapter intentionally owns only Python-facing validation, option
//! translation and detached result data.  Discretization, symbolic lowering,
//! generated-backend selection and the nonlinear iteration remain in the
//! native `NRBVP` implementation.

use std::collections::HashMap;

use nalgebra::DMatrix;
use pyo3::prelude::*;

use rustedscithe_core::numerical::BVP_Damp::BvpLambdifyTelemetryMode;
use rustedscithe_core::numerical::BVP_Damp::NR_Damp_solver_damped::{
    AdaptiveGridConfig, DampedSolverOptions, NRBVP, SolverParams,
};
use rustedscithe_core::numerical::BVP_Damp::NR_Damp_solver_frozen::{
    FrozenSolverOptions, NRBVP as FrozenNRBVP,
};
use rustedscithe_core::numerical::BVP_Damp::generated_solver_handoff::{
    AotBuildPolicy, AotBuildProfile, AotExecutionPolicy, GeneratedBackendConfig,
};
use rustedscithe_core::numerical::BVP_Damp::grid_api::GridRefinementMethod;
use rustedscithe_core::numerical::BVP_Damp::telemetry::{
    BvpCallbackStage, BvpLoggingConfig, BvpLoggingMode, BvpTelemetryMode, BvpTelemetrySnapshot,
};
use rustedscithe_core::somelinalg::banded::{
    FallbackPolicy, LinearSolverConfig, LinearSolverPolicy,
};
use rustedscithe_core::symbolic::bvp::legacy::BvpBackendIntegrationError;
use rustedscithe_core::symbolic::bvp::legacy::BvpSymbolicAssemblyBackend;
use rustedscithe_core::symbolic::bvp::telemetry::BvpLambdifyExecutionPolicy;
use rustedscithe_core::symbolic::bvp::telemetry::BvpLambdifyTelemetrySnapshot;
use rustedscithe_core::symbolic::codegen::codegen_aot_driver::AotCodegenBackend;
use rustedscithe_core::symbolic::codegen::codegen_aot_resolution::AotResolver;
use rustedscithe_core::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
use rustedscithe_core::symbolic::codegen::codegen_orchestrator::ParallelExecutorConfig;
use rustedscithe_core::symbolic::codegen::codegen_provider_api::MatrixBackend;

use crate::contract::SolverOptions;
use crate::errors::{
    ExhaustionError, InvalidConfigurationError, PreparationError, SolverError, SolverPanicState,
    UnsupportedOptionError, annotate_native, catch_panic, native_variant,
};
use crate::expr::PyExpr;
use crate::telemetry::write_report;

fn finite_values(values: &[f64], label: &str) -> PyResult<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(InvalidConfigurationError::new_err(format!(
            "{label} must contain only finite values"
        )));
    }
    Ok(())
}

fn validate_names(names: &[String], label: &str, allow_empty: bool) -> PyResult<()> {
    if (!allow_empty && names.is_empty()) || names.iter().any(|name| name.trim().is_empty()) {
        return Err(InvalidConfigurationError::new_err(format!(
            "{label} must contain at least one non-empty name"
        )));
    }
    let mut seen = std::collections::HashSet::with_capacity(names.len());
    if names.iter().any(|name| !seen.insert(name.as_str())) {
        return Err(InvalidConfigurationError::new_err(format!(
            "{label} must contain unique names"
        )));
    }
    Ok(())
}

fn map_native_error(error: BvpBackendIntegrationError) -> PyErr {
    let native_kind = native_variant(&error);
    let message = error.to_string();
    annotate_native(
        match error {
            BvpBackendIntegrationError::InvalidProblem { .. }
            | BvpBackendIntegrationError::InvalidSolverConfiguration { .. }
            | BvpBackendIntegrationError::InvalidLogLevel { .. }
            | BvpBackendIntegrationError::LogFileCreationFailed { .. } => {
                InvalidConfigurationError::new_err(message)
            }
            BvpBackendIntegrationError::CompiledAotRequiredButUnavailable { .. }
            | BvpBackendIntegrationError::AutomaticAotBuildRequested { .. }
            | BvpBackendIntegrationError::AutomaticAotRebuildRequested { .. }
            | BvpBackendIntegrationError::AutomaticAotBuildFailed { .. }
            | BvpBackendIntegrationError::CompiledAotRuntimeUnavailable { .. }
            | BvpBackendIntegrationError::AtomAotPlanPreparationFailed { .. }
            | BvpBackendIntegrationError::AotLifecycleFailure { .. } => {
                PreparationError::new_err(message)
            }
            _ => SolverError::new_err(message),
        },
        "solve",
        "bvp_damp",
        "native",
        &native_kind,
    )
}

fn map_telemetry(value: &str) -> PyResult<(BvpTelemetryMode, BvpLambdifyTelemetryMode)> {
    match value {
        "off" => Ok((BvpTelemetryMode::Off, BvpLambdifyTelemetryMode::Off)),
        "counters" => Ok((
            BvpTelemetryMode::Counters,
            BvpLambdifyTelemetryMode::Counters,
        )),
        "timings" => Ok((BvpTelemetryMode::Timings, BvpLambdifyTelemetryMode::Timings)),
        "detailed" => Ok((
            BvpTelemetryMode::Detailed,
            BvpLambdifyTelemetryMode::Detailed,
        )),
        other => Err(InvalidConfigurationError::new_err(format!(
            "unknown BVP_Damp telemetry mode '{other}'"
        ))),
    }
}

fn matrix_backend(value: &str) -> PyResult<(String, MatrixBackend)> {
    match value {
        "dense" => Ok(("Dense".to_owned(), MatrixBackend::Dense)),
        "sparse" => Ok(("Sparse".to_owned(), MatrixBackend::SparseCol)),
        "banded" => Ok(("Banded".to_owned(), MatrixBackend::Banded)),
        other => Err(UnsupportedOptionError::new_err(format!(
            "unsupported BVP_Damp matrix layout '{other}'"
        ))),
    }
}

fn frontend(value: &str) -> PyResult<BvpSymbolicAssemblyBackend> {
    match value {
        "expr_legacy" => Ok(BvpSymbolicAssemblyBackend::ExprLegacy),
        "atom_native" => Ok(BvpSymbolicAssemblyBackend::AtomView),
        other => Err(UnsupportedOptionError::new_err(format!(
            "unsupported BVP_Damp frontend '{other}'"
        ))),
    }
}

fn execution_policy(value: &str) -> PyResult<BvpLambdifyExecutionPolicy> {
    match value {
        "sequential" => Ok(BvpLambdifyExecutionPolicy::Sequential),
        "parallel" => Ok(BvpLambdifyExecutionPolicy::Parallel { min_work: 0 }),
        "auto" => Ok(BvpLambdifyExecutionPolicy::Auto { min_work: 0 }),
        other => Err(InvalidConfigurationError::new_err(format!(
            "unknown BVP_Damp execution policy '{other}'"
        ))),
    }
}

fn build_policy(options: &SolverOptions) -> AotBuildPolicy {
    let profile = if options.aot_profile() == "debug" {
        AotBuildProfile::Debug
    } else {
        AotBuildProfile::Release
    };
    match options.aot_policy() {
        "require_prebuilt" => AotBuildPolicy::RequirePrebuilt,
        "rebuild_always" => AotBuildPolicy::RebuildAlways { profile },
        _ => AotBuildPolicy::BuildIfMissing { profile },
    }
}

fn grid_method(value: &str, parameters: &[f64]) -> PyResult<GridRefinementMethod> {
    finite_values(parameters, "grid_parameters")?;
    let normalized = value.trim().to_ascii_lowercase().replace(['-', ' '], "_");
    let wrong_count = |expected: usize| {
        InvalidConfigurationError::new_err(format!(
            "grid_method '{normalized}' expects {expected} grid_parameters, got {}",
            parameters.len()
        ))
    };
    match normalized.as_str() {
        "double_points" | "double" => {
            if !parameters.is_empty() {
                return Err(wrong_count(0));
            }
            Ok(GridRefinementMethod::DoublePoints)
        }
        "easy" => {
            if parameters.len() != 1 {
                return Err(wrong_count(1));
            }
            Ok(GridRefinementMethod::Easy(parameters[0]))
        }
        "pearson" => {
            if parameters.len() != 2 {
                return Err(wrong_count(2));
            }
            Ok(GridRefinementMethod::Pearson(parameters[0], parameters[1]))
        }
        "grcar_smooke" | "grcar" => {
            if parameters.len() != 3 {
                return Err(wrong_count(3));
            }
            Ok(GridRefinementMethod::GrcarSmooke(
                parameters[0],
                parameters[1],
                parameters[2],
            ))
        }
        "sci" | "scipy" => {
            if !parameters.is_empty() {
                return Err(wrong_count(0));
            }
            Ok(GridRefinementMethod::Sci())
        }
        "two_point" | "twopoint" => {
            if parameters.len() != 3 {
                return Err(wrong_count(3));
            }
            Ok(GridRefinementMethod::TwoPoint(
                parameters[0],
                parameters[1],
                parameters[2],
            ))
        }
        _ => Err(UnsupportedOptionError::new_err(format!(
            "unsupported BVP_Damp grid_method '{normalized}'; expected double_points, easy, pearson, grcar_smooke, sci or two_point"
        ))),
    }
}

fn strategy_params(
    max_jacobian_refreshes: Option<usize>,
    max_damping_iterations: Option<usize>,
    damping_factor: Option<f64>,
    max_refinements: Option<usize>,
    grid_method_name: Option<&str>,
    grid_parameters: Option<Vec<f64>>,
) -> PyResult<SolverParams> {
    let mut params = SolverParams::default();
    if let Some(value) = max_jacobian_refreshes {
        if value == 0 {
            return Err(InvalidConfigurationError::new_err(
                "max_jacobian_refreshes must be non-zero",
            ));
        }
        params.max_jac = Some(value);
    }
    if let Some(value) = max_damping_iterations {
        if value == 0 {
            return Err(InvalidConfigurationError::new_err(
                "max_damping_iterations must be non-zero",
            ));
        }
        params.max_damp_iter = Some(value);
    }
    if let Some(value) = damping_factor {
        if !value.is_finite() || !(0.0 < value && value < 1.0) {
            return Err(InvalidConfigurationError::new_err(
                "damping_factor must be finite and strictly between 0 and 1",
            ));
        }
        params.damp_factor = Some(value);
    }
    let grid_parameters = grid_parameters.unwrap_or_default();
    if let Some(name) = grid_method_name {
        let max_refinements = max_refinements.ok_or_else(|| {
            InvalidConfigurationError::new_err(
                "max_refinements is required when grid_method is selected",
            )
        })?;
        let method = grid_method(name, &grid_parameters)?;
        params.adaptive = Some(AdaptiveGridConfig {
            version: 1,
            max_refinements,
            grid_method: method,
        });
    } else if !grid_parameters.is_empty() || max_refinements.is_some() {
        return Err(InvalidConfigurationError::new_err(
            "grid_parameters and max_refinements require grid_method",
        ));
    }
    Ok(params)
}

fn aot_execution_policy(options: &SolverOptions) -> AotExecutionPolicy {
    match options.execution_policy() {
        "sequential" => AotExecutionPolicy::SequentialOnly,
        "parallel" => AotExecutionPolicy::Parallel(ParallelExecutorConfig::default()),
        _ => AotExecutionPolicy::Auto,
    }
}

fn linear_solver_config(
    options: &SolverOptions,
    policy: Option<&str>,
    fallback: Option<&str>,
    iterative_refinement_steps: usize,
) -> PyResult<Option<LinearSolverConfig>> {
    if options.matrix_layout() != "banded" {
        if policy.is_some() || fallback.is_some() || iterative_refinement_steps != 0 {
            return Err(UnsupportedOptionError::new_err(
                "banded linear solver settings require matrix_layout='banded'",
            ));
        }
        return Ok(None);
    }
    let policy = match policy.unwrap_or("auto") {
        "auto" => LinearSolverPolicy::Auto,
        "block_tridiagonal" => LinearSolverPolicy::ForceBlockTridiagonal,
        "block_tridiagonal_consistent" => LinearSolverPolicy::ForceBlockTridiagonalConsistent,
        "banded" => LinearSolverPolicy::ForceBanded,
        "general_banded_partial_pivot" => LinearSolverPolicy::ForceGeneralBandedPartialPivot,
        "faer_sparse" => LinearSolverPolicy::ForceFaerSparse,
        other => {
            return Err(UnsupportedOptionError::new_err(format!(
                "unsupported banded_linear_solver '{other}'"
            )));
        }
    };
    let fallback = match fallback.unwrap_or("to_faer_sparse") {
        "never" => FallbackPolicy::Never,
        "to_faer_sparse" => FallbackPolicy::ToFaerSparse,
        other => {
            return Err(UnsupportedOptionError::new_err(format!(
                "unsupported banded_fallback '{other}'"
            )));
        }
    };
    Ok(Some(LinearSolverConfig {
        policy,
        fallback,
        iterative_refinement_steps,
    }))
}

fn backend_config(
    options: &SolverOptions,
    logging_mode: BvpLoggingMode,
    logging_max_events: usize,
    banded_linear_solver: Option<&str>,
    banded_fallback: Option<&str>,
    iterative_refinement_steps: usize,
) -> PyResult<GeneratedBackendConfig> {
    let assembly = frontend(options.frontend())?;
    let (_, matrix) = matrix_backend(options.matrix_layout())?;
    let policy = if options.execution() == "aot" {
        BackendSelectionPolicy::AotOnly
    } else {
        BackendSelectionPolicy::LambdifyOnly
    };
    let (bvp_telemetry, lambdify_telemetry) = map_telemetry(options.telemetry())?;
    if options.aot_output_dir().is_some() {
        return Err(UnsupportedOptionError::new_err(
            "BVP_Damp does not currently expose a native AOT output-directory override; use the native registry/resolver configuration",
        ));
    }
    let mut config = GeneratedBackendConfig::new()
        .with_symbolic_assembly_backend(assembly)
        .with_matrix_backend_override(matrix)
        .with_backend_policy_override(Some(policy))
        .with_aot_build_policy(build_policy(options))
        .with_aot_execution_policy(aot_execution_policy(options))
        .with_lambdify_execution_policy(execution_policy(options.execution_policy())?)
        .with_lambdify_telemetry_mode(lambdify_telemetry)
        .with_bvp_telemetry_mode(bvp_telemetry)
        .with_bvp_logging_config(
            BvpLoggingConfig::new(logging_mode).with_max_events(logging_max_events),
        );
    config = match options.aot_toolchain() {
        "gcc" | "tcc" => config
            .with_aot_codegen_backend(AotCodegenBackend::C)
            .with_aot_c_compiler(options.aot_toolchain()),
        "zig" => config.with_aot_codegen_backend(AotCodegenBackend::Zig),
        "rust" => config.with_aot_codegen_backend(AotCodegenBackend::Rust),
        _ => unreachable!("SolverOptions validates the AOT toolchain"),
    };
    if let Some(path) = options.aot_handoff_path() {
        let resolver = AotResolver::read_handoff(path).map_err(|error| {
            PreparationError::new_err(format!("could not read AOT handoff '{path}': {error}"))
        })?;
        config = config.with_resolver(Some(resolver));
    }
    if options.sparse_factorization() != "standard_faer" {
        return Err(UnsupportedOptionError::new_err(
            "BVP_Damp direct API supports only its native standard linear policy; sparse_factorization is a BVP_sci option",
        ));
    }
    if let Some(linear_config) = linear_solver_config(
        options,
        banded_linear_solver,
        banded_fallback,
        iterative_refinement_steps,
    )? {
        config = config.with_banded_linear_solver_config(linear_config);
    }
    Ok(config)
}

fn options_for_solver(
    options: &SolverOptions,
    scheme: &str,
    tolerance: f64,
    max_iterations: usize,
    bounds: HashMap<String, (f64, f64)>,
    state_variables: &[String],
    strategy_params: SolverParams,
    linear_sys_method: Option<&str>,
    loglevel: Option<&str>,
    logging_mode: BvpLoggingMode,
    logging_max_events: usize,
    banded_linear_solver: Option<&str>,
    banded_fallback: Option<&str>,
    iterative_refinement_steps: usize,
) -> PyResult<DampedSolverOptions> {
    let (method, _) = matrix_backend(options.matrix_layout())?;
    let mut solver_options = match options.matrix_layout() {
        "dense" => DampedSolverOptions::dense_damped(),
        "sparse" => DampedSolverOptions::sparse_damped(),
        "banded" => DampedSolverOptions::banded_damped(),
        _ => unreachable!("matrix_backend validates layout"),
    };
    solver_options.method = method;
    solver_options = solver_options
        .with_scheme_name(scheme)
        .with_abs_tolerance(tolerance)
        // BVP_Damp requires a per-state relative tolerance in addition to the
        // scalar absolute tolerance.  The Python surface deliberately keeps a
        // compact scalar tolerance, so apply it consistently to every state.
        .with_rel_tolerance(
            state_variables
                .iter()
                .cloned()
                .map(|name| (name, tolerance))
                .collect(),
        )
        .with_max_iterations(max_iterations)
        .with_bounds(bounds)
        .with_strategy_params(Some(strategy_params))
        .with_loglevel(loglevel.map(str::to_owned))
        .with_generated_backend_config(backend_config(
            options,
            logging_mode,
            logging_max_events,
            banded_linear_solver,
            banded_fallback,
            iterative_refinement_steps,
        )?);
    solver_options.linear_sys_method = linear_sys_method.map(str::to_owned);
    Ok(solver_options)
}

/// Convert Python's `[node][state]` initial guess into the core's
/// `[state, node]` nalgebra matrix without changing node/state ordering.
fn flatten_node_major(
    initial_state: Vec<f64>,
    nodes: usize,
    state_dimension: usize,
) -> PyResult<DMatrix<f64>> {
    let expected = nodes.checked_mul(state_dimension).ok_or_else(|| {
        InvalidConfigurationError::new_err("mesh/state dimensions overflow usize")
    })?;
    if initial_state.len() != expected {
        return Err(InvalidConfigurationError::new_err(format!(
            "initial_state must contain {expected} node-major values, got {}",
            initial_state.len()
        )));
    }
    finite_values(&initial_state, "initial_state")?;
    let mut matrix = DMatrix::zeros(state_dimension, nodes);
    for node in 0..nodes {
        for state in 0..state_dimension {
            matrix[(state, node)] = initial_state[node * state_dimension + state];
        }
    }
    Ok(matrix)
}

/// Copy the core's `[state, node]` matrix back to detached Python
/// `[node][state]` rows, matching the public BVP result convention.
fn detached_values(matrix: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..matrix.ncols())
        .map(|node| {
            (0..matrix.nrows())
                .map(|state| matrix[(state, node)])
                .collect()
        })
        .collect()
}

fn snapshot_maps(
    telemetry: &BvpTelemetrySnapshot,
    legacy_counters: Option<&HashMap<String, usize>>,
) -> (HashMap<String, u64>, HashMap<String, f64>) {
    let native = &telemetry.counters;
    let mut counters: HashMap<String, u64> = [
        ("iterations", native.iterations),
        ("scalar_evaluations", native.scalar_evaluations),
        ("residual_calls", native.residual_calls),
        ("residual_requests", native.residual_requests),
        ("jacobian_requests", native.jacobian_requests),
        ("jacobian_recalculations", native.jacobian_recalculations),
        ("residual_chunks", native.residual_chunks),
        ("jacobian_chunks", native.jacobian_chunks),
        ("conversions", native.conversions),
        ("copies", native.copies),
        ("damping_trials", native.damping_trials),
        ("damping_rejections", native.damping_rejections),
        ("linear_solves", native.linear_solves),
        ("factorizations", native.factorizations),
        ("factorization_cache_hits", native.factorization_cache_hits),
        (
            "factorization_invalidations",
            native.factorization_invalidations,
        ),
        ("rhs_solves", native.rhs_solves),
        ("grid_refinements", native.grid_refinements),
    ]
    .into_iter()
    .map(|(name, value)| (name.to_owned(), value))
    .collect();
    if let Some(legacy) = legacy_counters {
        for (name, value) in legacy {
            let stable_name = match name.as_str() {
                "number of grid points" => "grid_points",
                "length of y vector" => "state_vector_length",
                "number of jacobian elements" => "jacobian_elements",
                _ => continue,
            };
            counters.insert(stable_name.to_owned(), *value as u64);
        }
    }
    let mut timings: HashMap<String, f64> = [
        ("total_ms", telemetry.timings.total),
        ("residual_ms", telemetry.timings.residual),
        ("jacobian_ms", telemetry.timings.jacobian),
        ("linear_system_ms", telemetry.timings.linear_system),
        ("factorization_ms", telemetry.timings.factorization),
        ("rhs_solve_ms", telemetry.timings.rhs_solve),
        (
            "symbolic_operations_ms",
            telemetry.timings.symbolic_operations,
        ),
        ("grid_refinement_ms", telemetry.timings.grid_refinement),
    ]
    .into_iter()
    .map(|(name, duration)| (name.to_owned(), duration.as_secs_f64() * 1000.0))
    .collect();
    if !telemetry.telemetry_mode.collects_counters() {
        counters.clear();
    }
    if !telemetry.telemetry_mode.collects_timings() {
        timings.clear();
    }
    (counters, timings)
}

fn snapshot_detail_maps(
    telemetry: &BvpTelemetrySnapshot,
) -> (
    HashMap<String, f64>,
    HashMap<String, u64>,
    HashMap<String, u64>,
) {
    let timings_enabled = telemetry.telemetry_mode.collects_timings();
    let counters_enabled = telemetry.telemetry_mode.collects_counters();
    let mut scope_timings_ms = HashMap::new();
    let mut scope_events = HashMap::new();
    let scopes = [
        ("solve", &telemetry.scopes.solve),
        ("mesh_revision", &telemetry.scopes.mesh_revision),
        ("iteration", &telemetry.scopes.iteration),
        ("damping_trial", &telemetry.scopes.damping_trial),
    ];
    if timings_enabled {
        for (name, scope) in scopes {
            scope_timings_ms.insert(format!("{name}_ms"), scope.elapsed.as_secs_f64() * 1000.0);
        }
        for stage in &telemetry.timings.callback_stage_timings {
            let name = match stage.stage {
                BvpCallbackStage::ResidualInputPrep => "residual_input_prep",
                BvpCallbackStage::ResidualArgs => "residual_args",
                BvpCallbackStage::ResidualValues => "residual_values",
                BvpCallbackStage::ResidualBoxing => "residual_boxing",
                BvpCallbackStage::JacobianInputPrep => "jacobian_input_prep",
                BvpCallbackStage::JacobianArgs => "jacobian_args",
                BvpCallbackStage::JacobianValues => "jacobian_values",
                BvpCallbackStage::JacobianAssembly => "jacobian_assembly",
                BvpCallbackStage::Other => "other",
            };
            scope_timings_ms.insert(
                format!("callback_{name}_ms"),
                stage.elapsed.as_secs_f64() * 1000.0,
            );
        }
    }
    if counters_enabled {
        for (name, scope) in [
            ("mesh_revision", &telemetry.scopes.mesh_revision),
            ("iteration", &telemetry.scopes.iteration),
            ("damping_trial", &telemetry.scopes.damping_trial),
        ] {
            scope_events.insert(format!("{name}_events"), scope.events);
        }
    }

    let mut storage_bytes = HashMap::new();
    if telemetry.telemetry_mode != BvpTelemetryMode::Off {
        for (name, value) in [
            ("jacobian_values", telemetry.storage.jacobian_values),
            ("jacobian_indices", telemetry.storage.jacobian_indices),
            ("factor", telemetry.storage.factor),
            ("scratch", telemetry.storage.scratch),
        ] {
            if let Some(value) = value {
                storage_bytes.insert(name.to_owned(), value);
            }
        }
    }
    (scope_timings_ms, scope_events, storage_bytes)
}

fn snapshot_callback_maps(
    telemetry: &BvpTelemetrySnapshot,
) -> (HashMap<String, u64>, HashMap<String, f64>) {
    let mut counters = HashMap::new();
    let mut timings_ms = HashMap::new();
    let mut add_lambdify = |route: &str, snapshot: Option<BvpLambdifyTelemetrySnapshot>| {
        let Some(snapshot) = snapshot else { return };
        if matches!(
            snapshot.mode,
            BvpLambdifyTelemetryMode::Counters | BvpLambdifyTelemetryMode::Detailed
        ) {
            counters.insert(format!("{route}_residual_calls"), snapshot.residual_calls);
            counters.insert(format!("{route}_jacobian_calls"), snapshot.jacobian_calls);
            counters.insert(
                format!("{route}_parallel_dispatches"),
                snapshot.parallel_dispatches,
            );
            counters.insert(
                format!("{route}_sequential_dispatches"),
                snapshot.sequential_dispatches,
            );
        }
        if matches!(
            snapshot.mode,
            BvpLambdifyTelemetryMode::Timings | BvpLambdifyTelemetryMode::Detailed
        ) {
            timings_ms.insert(
                format!("{route}_residual_ms"),
                snapshot.residual_elapsed.as_secs_f64() * 1000.0,
            );
            timings_ms.insert(
                format!("{route}_jacobian_ms"),
                snapshot.jacobian_elapsed.as_secs_f64() * 1000.0,
            );
        }
    };
    add_lambdify("expr_legacy", telemetry.legacy_lambdify);
    add_lambdify("atom_view", telemetry.atom_lambdify);

    if let Some(snapshot) = telemetry.direct_banded_jacobian {
        if telemetry.telemetry_mode.collects_counters() {
            counters.extend([
                ("direct_banded_calls".to_owned(), snapshot.calls),
                ("direct_banded_work_items".to_owned(), snapshot.work_items),
                ("direct_banded_errors".to_owned(), snapshot.errors),
                (
                    "direct_banded_evaluator_calls".to_owned(),
                    snapshot.evaluator_calls,
                ),
                (
                    "direct_banded_storage_writes".to_owned(),
                    snapshot.storage_writes,
                ),
                (
                    "direct_banded_parallel_dispatches".to_owned(),
                    snapshot.parallel_dispatches,
                ),
                (
                    "direct_banded_sequential_dispatches".to_owned(),
                    snapshot.sequential_dispatches,
                ),
                (
                    "direct_banded_diagonal_dispatches".to_owned(),
                    snapshot.diagonal_dispatches,
                ),
                (
                    "direct_banded_entry_dispatches".to_owned(),
                    snapshot.entry_dispatches,
                ),
                (
                    "direct_banded_effective_task_count".to_owned(),
                    snapshot.effective_task_count,
                ),
                (
                    "prepared_atom_evaluators".to_owned(),
                    snapshot.prepared_atom_evaluators,
                ),
                (
                    "prepared_atom_nodes".to_owned(),
                    snapshot.prepared_atom_nodes,
                ),
                (
                    "prepared_atom_add_nodes".to_owned(),
                    snapshot.prepared_atom_add_nodes,
                ),
                (
                    "prepared_atom_mul_nodes".to_owned(),
                    snapshot.prepared_atom_mul_nodes,
                ),
                (
                    "prepared_atom_powi_nodes".to_owned(),
                    snapshot.prepared_atom_powi_nodes,
                ),
                (
                    "prepared_atom_pow_nodes".to_owned(),
                    snapshot.prepared_atom_pow_nodes,
                ),
                (
                    "prepared_atom_builtin_nodes".to_owned(),
                    snapshot.prepared_atom_builtin_nodes,
                ),
                (
                    "prepared_atom_custom_nodes".to_owned(),
                    snapshot.prepared_atom_custom_nodes,
                ),
            ]);
        }
        if telemetry.telemetry_mode.collects_timings() {
            timings_ms.extend([
                (
                    "direct_banded_elapsed_ms".to_owned(),
                    snapshot.elapsed.as_secs_f64() * 1000.0,
                ),
                (
                    "direct_banded_argument_prepare_ms".to_owned(),
                    snapshot.argument_prepare_elapsed.as_secs_f64() * 1000.0,
                ),
                (
                    "direct_banded_evaluator_ms".to_owned(),
                    snapshot.evaluator_elapsed.as_secs_f64() * 1000.0,
                ),
                (
                    "direct_banded_storage_write_ms".to_owned(),
                    snapshot.storage_write_elapsed.as_secs_f64() * 1000.0,
                ),
                (
                    "direct_banded_assembly_alloc_ms".to_owned(),
                    snapshot.assembly_alloc_elapsed.as_secs_f64() * 1000.0,
                ),
            ]);
        }
    }
    (counters, timings_ms)
}

fn snapshot_preparation_timings(telemetry: &BvpTelemetrySnapshot) -> HashMap<String, f64> {
    let mut timings = HashMap::new();
    if !telemetry.telemetry_mode.collects_timings() {
        return timings;
    }
    if let Some(snapshot) = telemetry.generation {
        for (name, duration) in [
            ("generation_total_ms", snapshot.total),
            ("generation_discretization_ms", snapshot.discretization),
            (
                "generation_symbolic_jacobian_ms",
                snapshot.symbolic_jacobian,
            ),
            ("generation_find_bandwidth_ms", snapshot.find_bandwidth),
            (
                "generation_backend_selection_ms",
                snapshot.backend_selection,
            ),
            ("generation_runtime_binding_ms", snapshot.runtime_binding),
            (
                "generation_lambdify_jacobian_compile_ms",
                snapshot.lambdify_jacobian_compile,
            ),
            (
                "generation_lambdify_residual_compile_ms",
                snapshot.lambdify_residual_compile,
            ),
        ] {
            timings.insert(name.to_owned(), duration.as_secs_f64() * 1000.0);
        }
    }
    if let Some(snapshot) = telemetry.atom_discretization {
        for (name, duration) in [
            ("atom_boundary_conditions_ms", snapshot.boundary_conditions),
            ("atom_discretization_ms", snapshot.discretization),
            (
                "atom_boundary_application_ms",
                snapshot.boundary_application,
            ),
            ("atom_flat_list_ms", snapshot.flat_list),
            ("atom_consistency_ms", snapshot.consistency),
            (
                "atom_bounds_and_tolerances_ms",
                snapshot.bounds_and_tolerances,
            ),
            ("atom_total_ms", snapshot.total),
        ] {
            timings.insert(name.to_owned(), duration.as_secs_f64() * 1000.0);
        }
    }
    timings
}

fn telemetry_mode_name(mode: BvpTelemetryMode) -> &'static str {
    match mode {
        BvpTelemetryMode::Off => "off",
        BvpTelemetryMode::Counters => "counters",
        BvpTelemetryMode::Timings => "timings",
        BvpTelemetryMode::Detailed => "detailed",
    }
}

/// Detached result for one BVP_Damp solve.
///
/// The mesh and values are copied out of the mutable native solver. Values
/// retain BVP_Damp's state-major orientation `(state_dimension, mesh_nodes)`;
/// use the NumPy accessors when this layout is more convenient for numerical
/// postprocessing. Telemetry and file-export methods operate only on this
/// detached snapshot and do not rerun the solver.
#[pyclass(name = "BvpDampSolution", module = "rustedscithe")]
#[derive(Clone)]
pub struct BvpDampSolution {
    solver_name: String,
    mesh: Vec<f64>,
    values: Vec<Vec<f64>>,
    parameters: Vec<f64>,
    status: String,
    telemetry_mode: String,
    counters: HashMap<String, u64>,
    timings_ms: HashMap<String, f64>,
    scope_timings_ms: HashMap<String, f64>,
    scope_events: HashMap<String, u64>,
    storage_bytes: HashMap<String, u64>,
    callback_counters: HashMap<String, u64>,
    callback_timings_ms: HashMap<String, f64>,
    preparation_timings_ms: HashMap<String, f64>,
    storage_estimated: Option<bool>,
}

#[pymethods]
impl BvpDampSolution {
    /// Return the detached mesh coordinates in increasing order.
    pub fn mesh(&self) -> Vec<f64> {
        self.mesh.clone()
    }

    /// Return states in native BVP_Damp orientation: `(state, mesh_node)`.
    pub fn values(&self) -> Vec<Vec<f64>> {
        self.values.clone()
    }

    /// Owned NumPy array with shape `(state_dimension, mesh_nodes)`.
    pub fn values_array<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        crate::numpy_support::matrix(py, &self.values)
    }

    /// Owned NumPy array containing the accepted mesh.
    pub fn mesh_array<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f64>> {
        crate::numpy_support::vector(py, &self.mesh)
    }

    #[getter]
    /// Values of declared model parameters for this detached solution.
    pub fn parameters(&self) -> Vec<f64> {
        self.parameters.clone()
    }

    #[getter]
    /// Native solver status captured when this solution was produced.
    pub fn status(&self) -> &str {
        &self.status
    }

    #[getter]
    /// Native telemetry collection policy used for this solve.
    pub fn telemetry_mode(&self) -> &str {
        &self.telemetry_mode
    }

    /// Return the counter snapshot collected during this solve.
    pub fn telemetry_counters(&self) -> HashMap<String, u64> {
        self.counters.clone()
    }

    /// Return stage timings in milliseconds for this solve.
    pub fn telemetry_timings_ms(&self) -> HashMap<String, f64> {
        self.timings_ms.clone()
    }

    /// Return elapsed time for native lifecycle scopes and callback sub-stages.
    /// Empty maps mean that this telemetry mode did not collect timings.
    pub fn telemetry_scope_timings_ms(&self) -> HashMap<String, f64> {
        self.scope_timings_ms.clone()
    }

    /// Return event counts grouped by native lifecycle scope.
    /// Empty maps mean that this telemetry mode did not collect counters.
    pub fn telemetry_scope_events(&self) -> HashMap<String, u64> {
        self.scope_events.clone()
    }

    /// Return available matrix/factor storage values in bytes.
    /// Missing keys mean that the native backend does not expose that value.
    pub fn telemetry_storage_bytes(&self) -> HashMap<String, u64> {
        self.storage_bytes.clone()
    }

    /// Whether the available storage values are estimates rather than direct
    /// allocator measurements; `None` means no storage values are available.
    pub fn telemetry_storage_is_estimated(&self) -> Option<bool> {
        self.storage_estimated
    }

    /// Return frontend-specific callback counts and dispatch decisions.
    pub fn telemetry_callback_counters(&self) -> HashMap<String, u64> {
        self.callback_counters.clone()
    }

    /// Return frontend-specific callback and direct Banded evaluator timings.
    pub fn telemetry_callback_timings_ms(&self) -> HashMap<String, f64> {
        self.callback_timings_ms.clone()
    }

    /// Return available AtomView and symbolic-generation preparation timings.
    /// Empty maps indicate that this route did not publish such a snapshot or
    /// that timing collection was disabled.
    pub fn telemetry_preparation_timings_ms(&self) -> HashMap<String, f64> {
        self.preparation_timings_ms.clone()
    }

    /// Return a small stable result descriptor for notebooks and batch tools.
    ///
    /// Mesh and values remain available through their typed accessors; this
    /// method is intentionally metadata-only and never serializes the state
    /// matrix into a second representation.
    pub fn diagnostics(&self) -> HashMap<String, String> {
        HashMap::from([
            ("solver".to_owned(), self.solver_name.clone()),
            ("status".to_owned(), self.status.clone()),
            ("telemetry_mode".to_owned(), self.telemetry_mode.clone()),
            ("nodes".to_owned(), self.mesh.len().to_string()),
            ("state_dimension".to_owned(), self.values.len().to_string()),
            (
                "parameter_count".to_owned(),
                self.parameters.len().to_string(),
            ),
        ])
    }

    /// Render counters and millisecond timings as one compact text report.
    pub fn telemetry_report(&self) -> String {
        let mut counters = self.counters.iter().collect::<Vec<_>>();
        counters.sort_by_key(|(name, _)| *name);
        let mut timings = self.timings_ms.iter().collect::<Vec<_>>();
        timings.sort_by_key(|(name, _)| *name);
        let mut callbacks = self.callback_counters.iter().collect::<Vec<_>>();
        callbacks.sort_by_key(|(name, _)| *name);
        let mut callback_timings = self.callback_timings_ms.iter().collect::<Vec<_>>();
        callback_timings.sort_by_key(|(name, _)| *name);
        let mut scopes = self.scope_timings_ms.iter().collect::<Vec<_>>();
        scopes.sort_by_key(|(name, _)| *name);
        let mut storage = self.storage_bytes.iter().collect::<Vec<_>>();
        storage.sort_by_key(|(name, _)| *name);
        format!(
            "{} status={}; counters={}; timings_ms={}; callback_counters={}; callback_timings_ms={}; scopes_ms={}; storage_bytes={}",
            self.solver_name,
            self.status,
            counters
                .iter()
                .map(|(name, value)| format!("{name}={value}"))
                .collect::<Vec<_>>()
                .join(","),
            timings
                .iter()
                .map(|(name, value)| format!("{name}={value:.6}"))
                .collect::<Vec<_>>()
                .join(","),
            callbacks
                .iter()
                .map(|(name, value)| format!("{name}={value}"))
                .collect::<Vec<_>>()
                .join(","),
            callback_timings
                .iter()
                .map(|(name, value)| format!("{name}={value:.6}"))
                .collect::<Vec<_>>()
                .join(","),
            scopes
                .iter()
                .map(|(name, value)| format!("{name}={value:.6}"))
                .collect::<Vec<_>>()
                .join(","),
            storage
                .iter()
                .map(|(name, value)| format!("{name}={value}"))
                .collect::<Vec<_>>()
                .join(","),
        )
    }

    /// Write the compact telemetry report to `path` without solver log output.
    pub fn write_telemetry_report(&self, path: &str) -> PyResult<()> {
        write_report(path, &self.telemetry_report())
    }

    /// Write one compact CSV row per mesh node without capturing solver logs.
    pub fn write_solution_csv(&self, path: &str) -> PyResult<()> {
        let state_dimension = self.values.len();
        let aligned = self.values.iter().all(|row| row.len() == self.mesh.len());
        let mut report = if aligned {
            String::from("mesh")
        } else {
            String::from("node_index")
        };
        for state in 0..state_dimension {
            report.push_str(&format!(",y{state}"));
        }
        report.push('\n');
        let row_count = if aligned {
            self.mesh.len()
        } else {
            self.values.iter().map(Vec::len).max().unwrap_or(0)
        };
        for node in 0..row_count {
            if aligned {
                report.push_str(&format!("{:.17e}", self.mesh[node]));
            } else {
                report.push_str(&node.to_string());
            }
            for state in 0..state_dimension {
                if let Some(value) = self.values[state].get(node) {
                    report.push_str(&format!(",{value:.17e}"));
                } else {
                    report.push(',');
                }
            }
            report.push('\n');
        }
        write_report(path, &report)
    }

    /// Write a human-readable tabular state snapshot for simple postprocessing.
    pub fn write_solution_txt(&self, path: &str) -> PyResult<()> {
        let state_dimension = self.values.len();
        let aligned = self.values.iter().all(|row| row.len() == self.mesh.len());
        let mut report = if aligned {
            String::from("BVP_Damp solution\nmesh")
        } else {
            String::from("BVP_Damp solution\nnode_index")
        };
        for state in 0..state_dimension {
            report.push_str(&format!("\ty{state}"));
        }
        report.push('\n');
        let row_count = if aligned {
            self.mesh.len()
        } else {
            self.values.iter().map(Vec::len).max().unwrap_or(0)
        };
        for node in 0..row_count {
            if aligned {
                report.push_str(&format!("{:.17e}", self.mesh[node]));
            } else {
                report.push_str(&node.to_string());
            }
            for state in 0..state_dimension {
                if let Some(value) = self.values[state].get(node) {
                    report.push_str(&format!("\t{value:.17e}"));
                } else {
                    report.push('\t');
                }
            }
            report.push('\n');
        }
        write_report(path, &report)
    }
}

/// Direct Python adapter for the native damped Newton BVP solver.
///
/// Boundary conditions map each state name to `(endpoint_index, value)` pairs,
/// where endpoint index is `0` (left) or `1` (right). `initial_state` is a
/// flattened node-major vector corresponding to shape
/// `(mesh_nodes, state_dimension)`; the adapter reshapes it for the core.
/// Solver controls such as refinement scheme, damping limits, linear solver,
/// bounds, logging and telemetry are forwarded to the native BVP_Damp API.
/// Use `prepare()` and `solve_prepared()` for an explicit prepared lifecycle,
/// then rebind values or restart the iterate without rebuilding the model.
#[pyclass(name = "BvpDampSolver", module = "rustedscithe", unsendable)]
pub struct BvpDampSolver {
    solver: NRBVP,
    frontend: String,
    execution: String,
    matrix_layout: String,
    bandwidth: Option<(usize, usize)>,
    prepared: bool,
    panic_state: SolverPanicState,
}

#[pymethods]
impl BvpDampSolver {
    /// Build a damped-Newton BVP from equations, boundary observations and an
    /// initial mesh guess.
    ///
    /// `expressions` has one derivative RHS per state variable. Every boundary
    /// condition key must name a state; its pairs select endpoint `0` (left)
    /// or `1` (right) and its value. `initial_state` is flattened in node-major
    /// order, so each node's state values appear consecutively. The binding
    /// reshapes it into the core's state-major matrix. `parameters`/
    /// `parameter_values` declare and bind symbolic input values. `options`
    /// selects frontend, execution and matrix layout. The
    /// remaining keyword options forward native refinement, damping, logging,
    /// bounds and linear-solver policies; unsupported combinations fail
    /// explicitly rather than silently falling back.
    #[new]
    #[pyo3(signature=(expressions, state_variables, boundary_conditions, mesh, initial_state, *, parameters=None, parameter_values=None, options=None, independent_variable="x", scheme="forward", strategy="damped", tolerance=1e-6, max_iterations=25, max_jacobian_refreshes=None, max_damping_iterations=None, damping_factor=None, max_refinements=None, grid_method=None, grid_parameters=None, linear_sys_method=None, loglevel=None, logging_mode="off", logging_max_events=256, banded_linear_solver=None, banded_fallback=None, iterative_refinement_steps=0, bounds=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        expressions: Vec<PyExpr>,
        state_variables: Vec<String>,
        boundary_conditions: HashMap<String, Vec<(usize, f64)>>,
        mesh: Vec<f64>,
        initial_state: Vec<f64>,
        parameters: Option<Vec<String>>,
        parameter_values: Option<Vec<f64>>,
        options: Option<PyRef<'_, SolverOptions>>,
        independent_variable: &str,
        scheme: &str,
        strategy: &str,
        tolerance: f64,
        max_iterations: usize,
        max_jacobian_refreshes: Option<usize>,
        max_damping_iterations: Option<usize>,
        damping_factor: Option<f64>,
        max_refinements: Option<usize>,
        grid_method: Option<&str>,
        grid_parameters: Option<Vec<f64>>,
        linear_sys_method: Option<&str>,
        loglevel: Option<&str>,
        logging_mode: &str,
        logging_max_events: usize,
        banded_linear_solver: Option<&str>,
        banded_fallback: Option<&str>,
        iterative_refinement_steps: usize,
        bounds: Option<HashMap<String, (f64, f64)>>,
    ) -> PyResult<Self> {
        validate_names(&state_variables, "state_variables", false)?;
        let parameters = parameters.unwrap_or_default();
        validate_names(&parameters, "parameters", true)?;
        if independent_variable.trim().is_empty() {
            return Err(InvalidConfigurationError::new_err(
                "independent_variable must not be empty",
            ));
        }
        if expressions.len() != state_variables.len() {
            return Err(InvalidConfigurationError::new_err(format!(
                "expected one RHS expression per state variable, got {} expressions for {} variables",
                expressions.len(),
                state_variables.len()
            )));
        }
        if state_variables
            .iter()
            .chain(parameters.iter())
            .any(|name| name == independent_variable)
        {
            return Err(InvalidConfigurationError::new_err(
                "independent_variable must not collide with state_variables or parameters",
            ));
        }
        if !matches!(scheme, "forward" | "trapezoid") {
            return Err(UnsupportedOptionError::new_err(
                "BVP_Damp scheme must be 'forward' or 'trapezoid'",
            ));
        }
        if strategy != "damped" {
            return Err(UnsupportedOptionError::new_err(
                "BVP_Damp direct Python API currently supports strategy='damped' only; strategy='frozen' remains an explicit stub",
            ));
        }
        if let Some(method) = linear_sys_method {
            if method.trim().is_empty() {
                return Err(InvalidConfigurationError::new_err(
                    "linear_sys_method must not be empty",
                ));
            }
        }
        let normalized_loglevel = match loglevel {
            None | Some("off") | Some("none") => None,
            Some("debug") => Some("debug"),
            Some("info") => Some("info"),
            Some("warn") => Some("warn"),
            Some("error") => Some("error"),
            Some(value) => {
                return Err(UnsupportedOptionError::new_err(format!(
                    "unsupported BVP_Damp loglevel '{value}'; expected off, debug, info, warn or error"
                )));
            }
        };
        let logging_mode = match logging_mode.trim().to_ascii_lowercase().as_str() {
            "off" => BvpLoggingMode::Off,
            "warnings" | "warn" => BvpLoggingMode::Warnings,
            "detailed" | "debug" => BvpLoggingMode::Detailed,
            other => {
                return Err(UnsupportedOptionError::new_err(format!(
                    "unsupported BVP_Damp logging_mode '{other}'; expected off, warnings or detailed"
                )));
            }
        };
        if !tolerance.is_finite() || tolerance <= 0.0 || max_iterations == 0 {
            return Err(InvalidConfigurationError::new_err(
                "tolerance must be finite and positive and max_iterations must be non-zero",
            ));
        }
        if mesh.len() < 2 || !mesh.windows(2).all(|pair| pair[1] > pair[0]) {
            return Err(InvalidConfigurationError::new_err(
                "mesh must contain at least two strictly increasing nodes",
            ));
        }
        finite_values(&mesh, "mesh")?;
        let parameter_values = parameter_values.unwrap_or_else(|| vec![0.0; parameters.len()]);
        if parameter_values.len() != parameters.len() {
            return Err(InvalidConfigurationError::new_err(format!(
                "parameter_values must contain {} values, got {}",
                parameters.len(),
                parameter_values.len()
            )));
        }
        finite_values(&parameter_values, "parameter_values")?;
        let expected_symbols = std::iter::once(independent_variable.to_owned())
            .chain(state_variables.iter().cloned())
            .chain(parameters.iter().cloned())
            .collect::<std::collections::HashSet<_>>();
        for (index, expression) in expressions.iter().enumerate() {
            if let Some(unknown) = expression
                .inner
                .extract_variables()
                .1
                .iter()
                .find(|name| !expected_symbols.contains(*name))
            {
                return Err(InvalidConfigurationError::new_err(format!(
                    "RHS expression {index} references undeclared symbol '{unknown}'"
                )));
            }
        }
        finite_boundary_conditions(&boundary_conditions)?;
        let bounds = bounds.ok_or_else(|| {
            InvalidConfigurationError::new_err(
                "BVP_Damp requires explicit bounds for every state variable",
            )
        })?;
        validate_bounds(&bounds, &state_variables)?;
        let options = match options {
            Some(value) => value.clone(),
            None => SolverOptions::new(None, None, None, None, None, None, None, None, None)?,
        };
        options.validate_for("bvp_damp")?;
        let _ = matrix_backend(options.matrix_layout())?;
        let bandwidth = match (options.band_lower(), options.band_upper()) {
            (Some(lower), Some(upper)) if options.matrix_layout() == "banded" => {
                Some((lower, upper))
            }
            (None, None) if options.matrix_layout() != "banded" => None,
            (None, None) => {
                return Err(InvalidConfigurationError::new_err(
                    "Banded BVP_Damp requires options.set_banded_width(lower, upper)",
                ));
            }
            _ if options.matrix_layout() != "banded" => {
                return Err(UnsupportedOptionError::new_err(
                    "banded widths are only valid with matrix_layout='banded'",
                ));
            }
            _ => {
                return Err(InvalidConfigurationError::new_err(
                    "band_lower and band_upper must be set together",
                ));
            }
        };
        let strategy_params = strategy_params(
            max_jacobian_refreshes,
            max_damping_iterations,
            damping_factor,
            max_refinements,
            grid_method,
            grid_parameters,
        )?;
        let damp_options = options_for_solver(
            &options,
            scheme,
            tolerance,
            max_iterations,
            bounds,
            &state_variables,
            strategy_params,
            linear_sys_method,
            normalized_loglevel,
            logging_mode,
            logging_max_events,
            banded_linear_solver,
            banded_fallback,
            iterative_refinement_steps,
        )?;
        let initial_guess = flatten_node_major(initial_state, mesh.len(), state_variables.len())?;
        let route = format!(
            "{}:{}:{}",
            options.frontend(),
            options.execution(),
            options.matrix_layout()
        );
        let solver = catch_panic("construct", "bvp_damp", &route, || {
            let mut solver = NRBVP::new_with_options(
                expressions
                    .into_iter()
                    .map(|expression| expression.inner)
                    .collect(),
                initial_guess,
                state_variables,
                independent_variable.to_owned(),
                boundary_conditions,
                mesh[0],
                mesh.last().copied().ok_or_else(|| {
                    InvalidConfigurationError::new_err("mesh must contain at least two nodes")
                })?,
                mesh.len(),
                damp_options,
            );
            if !parameters.is_empty() {
                let parameter_refs = parameters.iter().map(String::as_str).collect::<Vec<_>>();
                solver
                    .try_set_params(Some(parameter_refs.as_slice()))
                    .map_err(map_native_error)?;
                solver
                    .try_set_param_values(Some(parameter_values))
                    .map_err(map_native_error)?;
            }
            Ok(solver)
        })?;
        Ok(Self {
            solver,
            frontend: options.frontend().to_owned(),
            execution: options.execution().to_owned(),
            matrix_layout: options.matrix_layout().to_owned(),
            bandwidth,
            prepared: false,
            panic_state: SolverPanicState::default(),
        })
    }

    /// Solve the BVP and return detached values plus telemetry.
    pub fn solve(&mut self) -> PyResult<BvpDampSolution> {
        let route = format!(
            "{}:{}:{}",
            self.frontend, self.execution, self.matrix_layout
        );
        let panic_state = self.panic_state.clone();
        panic_state.run("solve", "bvp_damp", &route, || self.solve_inner())
    }

    /// Prepare symbolic callbacks and the selected runtime without solving.
    ///
    /// This is the cold-to-prepared lifecycle boundary used by continuation:
    /// subsequent `set_parameters` calls can update numeric bindings without
    /// rebuilding the symbolic graph.
    pub fn prepare(&mut self) -> PyResult<()> {
        let route = format!(
            "{}:{}:{}",
            self.frontend, self.execution, self.matrix_layout
        );
        let panic_state = self.panic_state.clone();
        panic_state.run("prepare", "bvp_damp", &route, || {
            self.solver
                .try_eq_generate(None, self.bandwidth)
                .map_err(map_native_error)?;
            self.prepared = true;
            Ok(())
        })
    }

    /// Solve using callbacks prepared by [`Self::prepare`].
    pub fn solve_prepared(&mut self) -> PyResult<BvpDampSolution> {
        let route = format!(
            "{}:{}:{}",
            self.frontend, self.execution, self.matrix_layout
        );
        let panic_state = self.panic_state.clone();
        panic_state.run("solve", "bvp_damp", &route, || {
            self.solver
                .try_solver_prepared()
                .map_err(map_native_error)?;
            self.prepared = true;
            self.detached_solution()
        })
    }

    /// Replace numeric parameter values while retaining the prepared graph.
    pub fn set_parameters(&mut self, values: Vec<f64>) -> PyResult<()> {
        finite_values(&values, "parameter_values")?;
        if values.len() != self.solver.param_names.len() {
            return Err(InvalidConfigurationError::new_err(format!(
                "parameter_values must contain {} values, got {}",
                self.solver.param_names.len(),
                values.len()
            )));
        }
        let route = format!(
            "{}:{}:{}",
            self.frontend, self.execution, self.matrix_layout
        );
        let panic_state = self.panic_state.clone();
        panic_state.run("continuation", "bvp_damp", &route, || {
            self.solver
                .try_set_param_values(Some(values))
                .map_err(map_native_error)
        })
    }

    /// Reset the Newton initial guess while retaining prepared callbacks.
    /// Numeric Jacobian and factor state are invalidated by the native solver.
    pub fn restart(&mut self, initial_state: Vec<f64>) -> PyResult<()> {
        let initial_guess =
            flatten_node_major(initial_state, self.solver.n_steps, self.solver.values.len())?;
        let route = format!(
            "{}:{}:{}",
            self.frontend, self.execution, self.matrix_layout
        );
        let panic_state = self.panic_state.clone();
        panic_state.run("restart", "bvp_damp", &route, || {
            self.solver
                .try_set_initial_guess(initial_guess)
                .map_err(map_native_error)
        })
    }

    /// Install a warm-start iterate without invalidating prepared callbacks.
    pub fn set_prepared_iterate(&mut self, initial_state: Vec<f64>) -> PyResult<()> {
        let iterate =
            flatten_node_major(initial_state, self.solver.n_steps, self.solver.values.len())?;
        let route = format!(
            "{}:{}:{}",
            self.frontend, self.execution, self.matrix_layout
        );
        let panic_state = self.panic_state.clone();
        panic_state.run("restart", "bvp_damp", &route, || {
            self.solver
                .try_set_prepared_iterate(iterate)
                .map_err(map_native_error)
        })
    }

    #[getter]
    /// Declared value-only parameter names, in binding order.
    pub fn parameter_names(&self) -> Vec<String> {
        self.solver.param_names.clone()
    }

    #[getter]
    /// Current values corresponding to `parameter_names`.
    pub fn parameter_values(&self) -> Vec<f64> {
        self.solver.param_values.clone().unwrap_or_default()
    }

    #[getter]
    /// Symbolic frontend selected for the prepared BVP problem.
    pub fn frontend(&self) -> &str {
        &self.frontend
    }

    #[getter]
    /// Runtime route selected for the prepared BVP problem.
    pub fn execution(&self) -> &str {
        &self.execution
    }

    #[getter]
    /// Linear-system layout selected for the BVP solver.
    pub fn matrix_layout(&self) -> &str {
        &self.matrix_layout
    }

    /// Return the effective route selected at construction time.
    pub fn provenance(&self) -> HashMap<String, String> {
        let mut values = HashMap::from([
            ("solver".to_owned(), "bvp_damp".to_owned()),
            ("strategy".to_owned(), "damped".to_owned()),
            ("frontend".to_owned(), self.frontend.clone()),
            ("execution".to_owned(), self.execution.clone()),
            ("matrix_layout".to_owned(), self.matrix_layout.clone()),
            (
                "prepared_bandwidth".to_owned(),
                self.bandwidth.map_or_else(
                    || "none".to_owned(),
                    |(lower, upper)| format!("{lower},{upper}"),
                ),
            ),
        ]);
        values.insert("prepared".to_owned(), self.prepared.to_string());
        values
    }
}

impl BvpDampSolver {
    fn solve_inner(&mut self) -> PyResult<BvpDampSolution> {
        if self.bandwidth.is_some() {
            self.solver
                .try_eq_generate(None, self.bandwidth)
                .map_err(map_native_error)?;
            self.solver
                .try_solver_prepared()
                .map_err(map_native_error)?;
        } else {
            self.solver.try_solve().map_err(map_native_error)?;
        }
        self.prepared = true;
        self.detached_solution()
    }

    fn detached_solution(&self) -> PyResult<BvpDampSolution> {
        let result = self.solver.get_result().ok_or_else(|| {
            SolverError::new_err("BVP_Damp completed without a detached solution")
        })?;
        let stats = self.solver.get_statistics();
        let (counters, timings_ms) = snapshot_maps(&stats.telemetry, None);
        let (scope_timings_ms, scope_events, storage_bytes) =
            snapshot_detail_maps(&stats.telemetry);
        let (callback_counters, callback_timings_ms) = snapshot_callback_maps(&stats.telemetry);
        let preparation_timings_ms = snapshot_preparation_timings(&stats.telemetry);
        let storage_estimated =
            (!storage_bytes.is_empty()).then_some(stats.telemetry.storage.estimated);
        Ok(BvpDampSolution {
            solver_name: "bvp_damp".to_owned(),
            mesh: self.solver.x_mesh.iter().copied().collect(),
            values: detached_values(&result),
            parameters: self.solver.param_values.clone().unwrap_or_default(),
            status: "finished".to_owned(),
            telemetry_mode: telemetry_mode_name(stats.telemetry.telemetry_mode).to_owned(),
            counters,
            timings_ms,
            scope_timings_ms,
            scope_events,
            storage_bytes,
            callback_counters,
            callback_timings_ms,
            preparation_timings_ms,
            storage_estimated,
        })
    }
}

/// Direct Python adapter for the native fixed-mesh Naive Newton BVP route.
///
/// Unlike [`BvpDampSolver`], this route uses the separate Frozen solver core:
/// it accepts a uniform mesh, has no damping/bounds/adaptive-refinement policy,
/// and retains the native prepared parameter-rebind lifecycle.
#[pyclass(name = "BvpDampNaiveSolver", module = "rustedscithe", unsendable)]
pub struct BvpDampNaiveSolver {
    solver: FrozenNRBVP,
    frontend: String,
    execution: String,
    matrix_layout: String,
    prepared: bool,
    panic_state: SolverPanicState,
}

#[pymethods]
impl BvpDampNaiveSolver {
    /// Create a fixed-mesh Naive Newton BVP.
    ///
    /// `mesh` must be strictly increasing and uniformly spaced because that is
    /// the mesh contract of the native Frozen/Naive implementation. Initial
    /// values are node-major `(mesh_node, state)` and boundary locations use
    /// the native endpoint selector `0` or `1`.
    #[new]
    #[pyo3(signature=(expressions, state_variables, boundary_conditions, mesh, initial_state, *, parameters=None, parameter_values=None, options=None, independent_variable="x", scheme="forward", tolerance=1e-6, max_iterations=25, logging_mode="off", logging_max_events=256, linear_sys_method=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        expressions: Vec<PyExpr>,
        state_variables: Vec<String>,
        boundary_conditions: HashMap<String, Vec<(usize, f64)>>,
        mesh: Vec<f64>,
        initial_state: Vec<f64>,
        parameters: Option<Vec<String>>,
        parameter_values: Option<Vec<f64>>,
        options: Option<PyRef<'_, SolverOptions>>,
        independent_variable: &str,
        scheme: &str,
        tolerance: f64,
        max_iterations: usize,
        logging_mode: &str,
        logging_max_events: usize,
        linear_sys_method: Option<&str>,
    ) -> PyResult<Self> {
        validate_names(&state_variables, "state_variables", false)?;
        let parameters = parameters.unwrap_or_default();
        validate_names(&parameters, "parameters", true)?;
        if state_variables
            .iter()
            .any(|state| parameters.contains(state))
        {
            return Err(InvalidConfigurationError::new_err(
                "state_variables and parameters must not contain the same symbol names",
            ));
        }
        if independent_variable.trim().is_empty()
            || state_variables
                .iter()
                .chain(parameters.iter())
                .any(|name| name == independent_variable)
        {
            return Err(InvalidConfigurationError::new_err(
                "independent_variable must be non-empty and distinct from states and parameters",
            ));
        }
        if expressions.len() != state_variables.len() {
            return Err(InvalidConfigurationError::new_err(format!(
                "expected one RHS expression per state variable, got {} expressions for {} variables",
                expressions.len(),
                state_variables.len()
            )));
        }
        if !matches!(scheme, "forward" | "trapezoid") {
            return Err(UnsupportedOptionError::new_err(
                "BVP_Damp Naive scheme must be 'forward' or 'trapezoid'",
            ));
        }
        if !tolerance.is_finite() || tolerance <= 0.0 || max_iterations == 0 {
            return Err(InvalidConfigurationError::new_err(
                "tolerance must be finite and positive and max_iterations must be non-zero",
            ));
        }
        if mesh.len() < 2 || !mesh.windows(2).all(|pair| pair[1] > pair[0]) {
            return Err(InvalidConfigurationError::new_err(
                "mesh must contain at least two strictly increasing nodes",
            ));
        }
        finite_values(&mesh, "mesh")?;
        let step = (mesh[mesh.len() - 1] - mesh[0]) / (mesh.len() - 1) as f64;
        if mesh.windows(2).any(|pair| {
            let local = pair[1] - pair[0];
            (local - step).abs() > 1e-10 * step.abs().max(1.0)
        }) {
            return Err(UnsupportedOptionError::new_err(
                "BVP_Damp Naive currently requires a uniform mesh; the native Frozen core constructs its mesh from endpoints and point count",
            ));
        }
        finite_boundary_conditions(&boundary_conditions)?;
        if boundary_conditions.len() != state_variables.len()
            || boundary_conditions
                .keys()
                .any(|name| !state_variables.contains(name))
        {
            return Err(InvalidConfigurationError::new_err(
                "boundary_conditions must provide endpoint constraints for declared state variables",
            ));
        }
        let expected_symbols = std::iter::once(independent_variable.to_owned())
            .chain(state_variables.iter().cloned())
            .chain(parameters.iter().cloned())
            .collect::<std::collections::HashSet<_>>();
        for (index, expression) in expressions.iter().enumerate() {
            if let Some(unknown) = expression
                .inner
                .extract_variables()
                .1
                .iter()
                .find(|name| !expected_symbols.contains(*name))
            {
                return Err(InvalidConfigurationError::new_err(format!(
                    "RHS expression {index} references undeclared symbol '{unknown}'"
                )));
            }
        }
        if linear_sys_method.is_some_and(|value| value.trim().is_empty()) {
            return Err(InvalidConfigurationError::new_err(
                "linear_sys_method must not be empty",
            ));
        }
        let logging_mode = match logging_mode.trim().to_ascii_lowercase().as_str() {
            "off" => BvpLoggingMode::Off,
            "warnings" | "warn" => BvpLoggingMode::Warnings,
            "detailed" | "debug" => BvpLoggingMode::Detailed,
            other => {
                return Err(UnsupportedOptionError::new_err(format!(
                    "unsupported BVP_Damp logging_mode '{other}'; expected off, warnings or detailed"
                )));
            }
        };
        let parameter_values = parameter_values.unwrap_or_else(|| vec![0.0; parameters.len()]);
        if parameter_values.len() != parameters.len() {
            return Err(InvalidConfigurationError::new_err(format!(
                "parameter_values must contain {} values, got {}",
                parameters.len(),
                parameter_values.len()
            )));
        }
        finite_values(&parameter_values, "parameter_values")?;
        let options = match options {
            Some(value) => value.clone(),
            None => SolverOptions::new(None, None, None, None, None, None, None, None, None)?,
        };
        options.validate_for("bvp_damp")?;
        let (method, _) = matrix_backend(options.matrix_layout())?;
        // The fixed-mesh native callback path currently expects nalgebra's
        // dense-vector ABI here; reject the known AtomNative+AOT+Dense
        // mismatch before it can panic across the Python boundary.
        if options.frontend() == "atom_native"
            && options.execution() == "aot"
            && options.matrix_layout() == "dense"
        {
            return Err(UnsupportedOptionError::new_err(
                "fixed-mesh BVP_Damp Naive/Frozen does not support the AtomNative+AOT+Dense route; use ExprLegacy+AOT, AtomNative+Lambdify, or a supported structured AOT layout",
            ));
        }
        if options.band_lower().is_some() || options.band_upper().is_some() {
            return Err(UnsupportedOptionError::new_err(
                "BVP_Damp Naive infers its band structure from the generated Jacobian and does not consume explicit band widths",
            ));
        }
        let config = backend_config(&options, logging_mode, logging_max_events, None, None, 0)?;
        let mut native_options = FrozenSolverOptions::dense_naive();
        native_options.method = method;
        native_options.scheme = scheme.to_owned();
        native_options.tolerance = tolerance;
        native_options.max_iterations = max_iterations;
        native_options.linear_sys_method = linear_sys_method.map(str::to_owned);
        native_options.generated_backend_config = config;
        let initial_guess = flatten_node_major(initial_state, mesh.len(), state_variables.len())?;
        let route = format!(
            "{}:{}:{}",
            options.frontend(),
            options.execution(),
            options.matrix_layout()
        );
        let mut solver = catch_panic("construct", "bvp_damp_naive", &route, || {
            let mut solver = FrozenNRBVP::new_with_options(
                expressions
                    .into_iter()
                    .map(|expression| expression.inner)
                    .collect(),
                initial_guess,
                state_variables,
                independent_variable.to_owned(),
                boundary_conditions,
                mesh[0],
                mesh[mesh.len() - 1],
                mesh.len(),
                native_options,
            );
            if !parameters.is_empty() {
                let parameter_refs = parameters.iter().map(String::as_str).collect::<Vec<_>>();
                solver
                    .try_set_params(Some(parameter_refs.as_slice()))
                    .map_err(map_native_error)?;
                solver
                    .try_set_param_values(Some(parameter_values))
                    .map_err(map_native_error)?;
            }
            Ok(solver)
        })?;
        solver.dont_save_log(true);
        Ok(Self {
            solver,
            frontend: options.frontend().to_owned(),
            execution: options.execution().to_owned(),
            matrix_layout: options.matrix_layout().to_owned(),
            prepared: false,
            panic_state: SolverPanicState::default(),
        })
    }

    /// Prepare symbolic callbacks and the native linear runtime.
    pub fn prepare(&mut self) -> PyResult<()> {
        let route = self.route();
        let panic_state = self.panic_state.clone();
        panic_state.run("prepare", "bvp_damp_naive", &route, || {
            self.solver.try_eq_generate().map_err(map_native_error)?;
            self.prepared = true;
            Ok(())
        })
    }

    /// Solve from the configured initial guess, preparing callbacks if needed.
    pub fn solve(&mut self) -> PyResult<BvpDampSolution> {
        let route = self.route();
        let panic_state = self.panic_state.clone();
        panic_state.run("solve", "bvp_damp_naive", &route, || {
            let result = self.solver.try_solver().map_err(map_native_error)?;
            self.prepared = true;
            self.finish_solution(result, "bvp_damp_naive")
        })
    }

    /// Solve using a previously prepared symbolic/runtime plan.
    pub fn solve_prepared(&mut self) -> PyResult<BvpDampSolution> {
        let route = self.route();
        let panic_state = self.panic_state.clone();
        panic_state.run("solve", "bvp_damp_naive", &route, || {
            if !self.prepared {
                return Err(SolverError::new_err(
                    "BVP_Damp Naive has no prepared plan; call prepare() first",
                ));
            }
            let result = self
                .solver
                .try_solver_prepared()
                .map_err(map_native_error)?;
            self.finish_solution(result, "bvp_damp_naive")
        })
    }

    /// Rebind numeric parameters without rebuilding the symbolic plan.
    pub fn set_parameters(&mut self, values: Vec<f64>) -> PyResult<()> {
        finite_values(&values, "parameter_values")?;
        if values.len() != self.solver.param_names.len() {
            return Err(InvalidConfigurationError::new_err(format!(
                "parameter_values must contain {} values, got {}",
                self.solver.param_names.len(),
                values.len()
            )));
        }
        let route = self.route();
        let panic_state = self.panic_state.clone();
        panic_state.run("continuation", "bvp_damp_naive", &route, || {
            self.solver
                .try_set_param_values(Some(values))
                .map_err(map_native_error)
        })
    }

    #[getter]
    /// Names of the declared numeric parameters, in binding order.
    pub fn parameter_names(&self) -> Vec<String> {
        self.solver.param_names.clone()
    }

    #[getter]
    /// Current numeric values corresponding to `parameter_names`.
    pub fn parameter_values(&self) -> Vec<f64> {
        self.solver.param_values.clone().unwrap_or_default()
    }

    #[getter]
    /// Symbolic preparation frontend selected for this solver.
    pub fn frontend(&self) -> &str {
        &self.frontend
    }

    #[getter]
    /// Callback execution route selected for this solver.
    pub fn execution(&self) -> &str {
        &self.execution
    }

    #[getter]
    /// Linear algebra layout selected for the generated BVP system.
    pub fn matrix_layout(&self) -> &str {
        &self.matrix_layout
    }

    /// Describe the selected native algorithm and current lifecycle state.
    pub fn provenance(&self) -> HashMap<String, String> {
        HashMap::from([
            ("solver".to_owned(), "bvp_damp_naive".to_owned()),
            ("strategy".to_owned(), "Naive".to_owned()),
            ("frontend".to_owned(), self.frontend.clone()),
            ("execution".to_owned(), self.execution.clone()),
            ("matrix_layout".to_owned(), self.matrix_layout.clone()),
            ("prepared".to_owned(), self.prepared.to_string()),
        ])
    }
}

impl BvpDampNaiveSolver {
    fn route(&self) -> String {
        format!(
            "{}:{}:{}",
            self.frontend, self.execution, self.matrix_layout
        )
    }

    fn finish_solution(
        &self,
        result: Option<nalgebra::DVector<f64>>,
        solver_name: &str,
    ) -> PyResult<BvpDampSolution> {
        if result.is_none() || self.solver.result.is_none() {
            return Err(ExhaustionError::new_err(format!(
                "BVP_Damp Naive did not converge in {} iterations",
                self.solver.max_iterations
            )));
        }
        let matrix = self.solver.get_result().ok_or_else(|| {
            SolverError::new_err("BVP_Damp Naive completed without a result matrix")
        })?;
        let stats = self.solver.get_statistics();
        let (counters, timings_ms) = snapshot_maps(&stats.telemetry, Some(&stats.counters));
        let (scope_timings_ms, scope_events, storage_bytes) =
            snapshot_detail_maps(&stats.telemetry);
        let (callback_counters, callback_timings_ms) = snapshot_callback_maps(&stats.telemetry);
        let preparation_timings_ms = snapshot_preparation_timings(&stats.telemetry);
        let storage_estimated =
            (!storage_bytes.is_empty()).then_some(stats.telemetry.storage.estimated);
        // Frozen's compatibility postprocessor returns node-major storage;
        // preserve the stable Python solution convention `(state, node)`.
        let values = (0..matrix.ncols())
            .map(|state| {
                (0..matrix.nrows())
                    .map(|node| matrix[(node, state)])
                    .collect()
            })
            .collect();
        Ok(BvpDampSolution {
            solver_name: solver_name.to_owned(),
            mesh: self.solver.x_mesh.iter().copied().collect(),
            values,
            parameters: self.solver.param_values.clone().unwrap_or_default(),
            status: "finished".to_owned(),
            telemetry_mode: telemetry_mode_name(stats.telemetry.telemetry_mode).to_owned(),
            counters,
            timings_ms,
            scope_timings_ms,
            scope_events,
            storage_bytes,
            callback_counters,
            callback_timings_ms,
            preparation_timings_ms,
            storage_estimated,
        })
    }
}

/// Direct Python adapter for BVP_Damp's Frozen-Jacobian Newton strategy.
///
/// This is a distinct native strategy from both the damped solver and Naive
/// Newton. It shares the fixed uniform-mesh, symbolic, layout and prepared
/// parameter lifecycle of the native Frozen solver core.
#[pyclass(name = "BvpDampFrozenSolver", module = "rustedscithe", unsendable)]
pub struct BvpDampFrozenSolver {
    inner: BvpDampNaiveSolver,
    policy: String,
}

#[pymethods]
impl BvpDampFrozenSolver {
    /// Create a fixed-mesh Frozen Newton BVP with an explicit Jacobian reuse policy.
    ///
    /// `policy` selects one native strategy: `Frozen_naive`, `every_m`,
    /// `at_high_morm`, `at_low_speed`, or `complex`. Policy values are passed
    /// in `policy_parameters` using the native order and shape.
    #[new]
    #[pyo3(signature=(expressions, state_variables, boundary_conditions, mesh, initial_state, *, parameters=None, parameter_values=None, options=None, independent_variable="x", scheme="forward", tolerance=1e-6, max_iterations=25, logging_mode="off", logging_max_events=256, linear_sys_method=None, policy="Frozen_naive", policy_parameters=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        expressions: Vec<PyExpr>,
        state_variables: Vec<String>,
        boundary_conditions: HashMap<String, Vec<(usize, f64)>>,
        mesh: Vec<f64>,
        initial_state: Vec<f64>,
        parameters: Option<Vec<String>>,
        parameter_values: Option<Vec<f64>>,
        options: Option<PyRef<'_, SolverOptions>>,
        independent_variable: &str,
        scheme: &str,
        tolerance: f64,
        max_iterations: usize,
        logging_mode: &str,
        logging_max_events: usize,
        linear_sys_method: Option<&str>,
        policy: &str,
        policy_parameters: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let strategy_params = frozen_policy_parameters(policy, policy_parameters)?;
        let mut inner = BvpDampNaiveSolver::new(
            expressions,
            state_variables,
            boundary_conditions,
            mesh,
            initial_state,
            parameters,
            parameter_values,
            options,
            independent_variable,
            scheme,
            tolerance,
            max_iterations,
            logging_mode,
            logging_max_events,
            linear_sys_method,
        )?;
        // The native struct exposes these fields for its legacy API. Set them
        // before preparation so the generated plan captures the Frozen route.
        inner.solver.strategy = "Frozen".to_owned();
        inner.solver.strategy_params = Some(strategy_params);
        Ok(Self {
            inner,
            policy: policy.to_owned(),
        })
    }

    /// Prepare symbolic callbacks and the Frozen linear-runtime plan.
    pub fn prepare(&mut self) -> PyResult<()> {
        let route = self.inner.route();
        let panic_state = self.inner.panic_state.clone();
        panic_state.run("prepare", "bvp_damp_frozen", &route, || {
            self.inner
                .solver
                .try_eq_generate()
                .map_err(map_native_error)?;
            self.inner.prepared = true;
            Ok(())
        })
    }

    /// Solve from the configured initial guess, preparing the native plan if needed.
    pub fn solve(&mut self) -> PyResult<BvpDampSolution> {
        let route = self.inner.route();
        let panic_state = self.inner.panic_state.clone();
        panic_state.run("solve", "bvp_damp_frozen", &route, || {
            let result = self.inner.solver.try_solver().map_err(map_native_error)?;
            self.inner.prepared = true;
            self.inner.finish_solution(result, "bvp_damp_frozen")
        })
    }

    /// Solve with the previously prepared callbacks and numeric structure.
    pub fn solve_prepared(&mut self) -> PyResult<BvpDampSolution> {
        let route = self.inner.route();
        let panic_state = self.inner.panic_state.clone();
        panic_state.run("solve", "bvp_damp_frozen", &route, || {
            if !self.inner.prepared {
                return Err(SolverError::new_err(
                    "BVP_Damp Frozen has no prepared plan; call prepare() first",
                ));
            }
            let result = self
                .inner
                .solver
                .try_solver_prepared()
                .map_err(map_native_error)?;
            self.inner.finish_solution(result, "bvp_damp_frozen")
        })
    }

    /// Rebind numeric parameters while retaining the symbolic prepared plan.
    pub fn set_parameters(&mut self, values: Vec<f64>) -> PyResult<()> {
        finite_values(&values, "parameter_values")?;
        if values.len() != self.inner.solver.param_names.len() {
            return Err(InvalidConfigurationError::new_err(format!(
                "parameter_values must contain {} values, got {}",
                self.inner.solver.param_names.len(),
                values.len()
            )));
        }
        let route = self.inner.route();
        let panic_state = self.inner.panic_state.clone();
        panic_state.run("continuation", "bvp_damp_frozen", &route, || {
            self.inner
                .solver
                .try_set_param_values(Some(values))
                .map_err(map_native_error)
        })
    }

    #[getter]
    /// Names of the declared numeric parameters, in binding order.
    pub fn parameter_names(&self) -> Vec<String> {
        self.inner.solver.param_names.clone()
    }

    #[getter]
    /// Current numeric values corresponding to `parameter_names`.
    pub fn parameter_values(&self) -> Vec<f64> {
        self.inner.solver.param_values.clone().unwrap_or_default()
    }

    #[getter]
    /// Symbolic frontend selected for this Frozen problem.
    pub fn frontend(&self) -> &str {
        &self.inner.frontend
    }

    #[getter]
    /// Callback execution route selected for this Frozen problem.
    pub fn execution(&self) -> &str {
        &self.inner.execution
    }

    #[getter]
    /// Matrix layout selected for the generated BVP system.
    pub fn matrix_layout(&self) -> &str {
        &self.inner.matrix_layout
    }

    /// Describe the selected Frozen policy and current native route.
    pub fn provenance(&self) -> HashMap<String, String> {
        let mut data = self.inner.provenance();
        data.insert("solver".to_owned(), "bvp_damp_frozen".to_owned());
        data.insert("strategy".to_owned(), "Frozen".to_owned());
        data.insert("frozen_policy".to_owned(), self.policy.clone());
        data
    }
}

fn frozen_policy_parameters(
    policy: &str,
    parameters: Option<Vec<f64>>,
) -> PyResult<HashMap<String, Option<Vec<f64>>>> {
    let expected_len = match policy {
        "Frozen_naive" => None,
        "every_m" | "at_high_morm" | "at_low_speed" => Some(1),
        "complex" => Some(3),
        other => {
            return Err(UnsupportedOptionError::new_err(format!(
                "unsupported Frozen policy '{other}'; expected Frozen_naive, every_m, at_high_morm, at_low_speed or complex"
            )));
        }
    };
    let values = match expected_len {
        None if parameters.is_none() => None,
        None => {
            return Err(InvalidConfigurationError::new_err(
                "Frozen_naive does not accept policy_parameters",
            ));
        }
        Some(expected) => {
            let values = parameters.ok_or_else(|| {
                InvalidConfigurationError::new_err(format!(
                    "Frozen policy '{policy}' requires {expected} policy parameter(s)"
                ))
            })?;
            if values.len() != expected || values.iter().any(|value| !value.is_finite()) {
                return Err(InvalidConfigurationError::new_err(format!(
                    "Frozen policy '{policy}' requires {expected} finite policy parameter(s)"
                )));
            }
            match policy {
                "every_m" if values[0] <= 0.0 || values[0].fract() != 0.0 => {
                    return Err(InvalidConfigurationError::new_err(
                        "every_m requires a positive integer iteration period",
                    ));
                }
                "at_high_morm" if values[0] <= 0.0 => {
                    return Err(InvalidConfigurationError::new_err(
                        "at_high_morm requires a positive norm threshold",
                    ));
                }
                "at_low_speed" if values[0] > 1.0 => {
                    return Err(InvalidConfigurationError::new_err(
                        "at_low_speed requires a speed rate no greater than 1",
                    ));
                }
                "complex"
                    if values[0] <= 0.0
                        || values[0].fract() != 0.0
                        || values[1] <= 0.0
                        || values[2] > 1.0 =>
                {
                    return Err(InvalidConfigurationError::new_err(
                        "complex expects [positive integer period, positive norm threshold, speed rate <= 1]",
                    ));
                }
                _ => {}
            }
            Some(values)
        }
    };
    Ok(HashMap::from([(policy.to_owned(), values)]))
}

fn finite_boundary_conditions(conditions: &HashMap<String, Vec<(usize, f64)>>) -> PyResult<()> {
    for (name, entries) in conditions {
        if name.trim().is_empty() {
            return Err(InvalidConfigurationError::new_err(
                "boundary condition variable names must not be empty",
            ));
        }
        for (position, value) in entries {
            if *position > 1 || !value.is_finite() {
                return Err(InvalidConfigurationError::new_err(
                    "boundary conditions require position 0 or 1 and finite values",
                ));
            }
        }
    }
    Ok(())
}

fn validate_bounds(
    bounds: &HashMap<String, (f64, f64)>,
    state_variables: &[String],
) -> PyResult<()> {
    for state in state_variables {
        let Some((lower, upper)) = bounds.get(state) else {
            return Err(InvalidConfigurationError::new_err(format!(
                "bounds must contain state variable '{state}'"
            )));
        };
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(InvalidConfigurationError::new_err(format!(
                "bounds for '{state}' must be finite and ordered as lower <= upper"
            )));
        }
    }
    Ok(())
}
