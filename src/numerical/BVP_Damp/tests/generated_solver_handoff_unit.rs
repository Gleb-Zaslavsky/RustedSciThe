//! Unit tests extracted from the production module.
//!
//! Keeping these tests in a separate file keeps solver implementation and
//! test-only story machinery independently navigable.

use super::{
    AotBuildPolicy, AotChunkingPolicy, AotCompileConfig, AotExecutionPolicy,
    BandedGeneratedBackendMode, DampedSolverBuildRequest, FrozenSolverBuildRequest,
    GeneratedBackendConfig, SparseGeneratedBackendMode,
};
use crate::symbolic::bvp::aot_telemetry::BvpAotTelemetryMode;
use crate::symbolic::bvp::telemetry::{BvpLambdifyExecutionPolicy, BvpLambdifyTelemetryMode};
use crate::symbolic::codegen::CodegenIR::AtomOptimizationProfile;
use crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend;
use crate::symbolic::codegen::codegen_aot_registry::AotRegistry;
use crate::symbolic::codegen::codegen_aot_resolution::AotResolver;
use crate::symbolic::codegen::codegen_aot_runtime_link::{
    LinkedResidualChunk, LinkedSparseAotBackend, LinkedSparseJacobianChunk,
    register_linked_sparse_backend, unregister_linked_sparse_backend,
};
use crate::symbolic::codegen::codegen_backend_selection::{
    BackendSelectionPolicy, SelectedBackendKind,
};
use crate::symbolic::codegen::codegen_manifest::PreparedProblemManifest;
use crate::symbolic::codegen::codegen_orchestrator::{
    ParallelExecutorConfig, ParallelFallbackPolicy,
};
use crate::symbolic::codegen::codegen_provider_api::{MatrixBackend, PreparedProblem};
use crate::symbolic::codegen::rust_backend::codegen_aot_build::{AotBuildProfile, AotBuildRequest};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions_BVP::{
    BvpBackendIntegrationError, BvpSymbolicAssemblyBackend, Jacobian,
};
use faer::Col;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicUsize, Ordering},
};
use std::time::{SystemTime, UNIX_EPOCH};

fn linked_runtime_registry_test_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn real_bvp_inputs() -> (
    Vec<Expr>,
    Vec<String>,
    String,
    HashMap<String, Vec<(usize, f64)>>,
) {
    let eq_system = vec![Expr::parse_expression("y-z"), Expr::parse_expression("z^3")];
    let values = vec!["y".to_string(), "z".to_string()];
    let arg = "x".to_string();
    let mut border_conditions = HashMap::new();
    border_conditions.insert("y".to_string(), vec![(0, 0.0), (1, 0.0)]);
    border_conditions.insert("z".to_string(), vec![(0, 1.0), (1, 1.0)]);
    (eq_system, values, arg, border_conditions)
}

#[test]
fn lambdify_telemetry_handles_survive_solver_handoff() {
    let (eq_system, values, arg, _) = real_bvp_inputs();
    let border_conditions = HashMap::from([
        ("y".to_string(), vec![(0, 0.0)]),
        ("z".to_string(), vec![(0, 1.0)]),
    ]);
    let request = DampedSolverBuildRequest {
        eq_system,
        values,
        arg,
        param_names: None,
        param_values: None,
        t0: 0.0,
        n_steps: Some(4),
        h: Some(0.25),
        mesh: None,
        border_conditions,
        bounds: None,
        rel_tolerance: None,
        scheme: "forward".to_string(),
        method: "Dense".to_string(),
        bandwidth: None,
        backend_policy: BackendSelectionPolicy::LambdifyOnly,
        resolver: None,
        aot_execution_policy: AotExecutionPolicy::Auto,
        aot_build_policy: AotBuildPolicy::UseIfAvailable,
        aot_compile_config: AotCompileConfig::default(),
        aot_codegen_backend: AotCodegenBackend::Rust,
        aot_c_compiler: None,
        aot_chunking_policy: AotChunkingPolicy::default(),
        atom_optimization_profile: AtomOptimizationProfile::Full,
        symbolic_assembly_backend: BvpSymbolicAssemblyBackend::ExprLegacy,
        matrix_backend_override: None,
        banded_linear_solver_config: crate::somelinalg::banded::LinearSolverConfig::default(),
        lambdify_telemetry_mode: BvpLambdifyTelemetryMode::Detailed,
        aot_telemetry_mode: BvpAotTelemetryMode::Off,
        lambdify_execution_policy: BvpLambdifyExecutionPolicy::default(),
    };

    let mut state = request
        .generate()
        .expect("dense Lambdify handoff should build");
    let y = nalgebra::DVector::from_fn(state.variable_string.len(), |index, _| {
        0.2 + index as f64 * 0.01
    });
    let _ = state.fun.call(0.0, &y);
    state
        .jac
        .as_mut()
        .expect("dense Lambdify Jacobian should be present")
        .call(0.0, &y);

    let telemetry = state
        .legacy_lambdify_telemetry
        .as_ref()
        .expect("handoff should preserve the live ExprLegacy telemetry handle")
        .snapshot();
    assert!(telemetry.residual_calls >= 1);
    assert!(telemetry.jacobian_calls >= 1);
    assert!(telemetry.residual_elapsed > std::time::Duration::ZERO);
    assert!(telemetry.jacobian_elapsed > std::time::Duration::ZERO);
}

fn parameterized_bvp_inputs() -> (
    Vec<Expr>,
    Vec<String>,
    String,
    Vec<String>,
    HashMap<String, Vec<(usize, f64)>>,
) {
    let eq_system = vec![
        Expr::parse_expression("a*(y-z)"),
        Expr::parse_expression("a*z^3"),
    ];
    let values = vec!["y".to_string(), "z".to_string()];
    let arg = "x".to_string();
    let params = vec!["a".to_string()];
    let mut border_conditions = HashMap::new();
    border_conditions.insert("y".to_string(), vec![(0, 0.0), (1, 0.0)]);
    border_conditions.insert("z".to_string(), vec![(0, 1.0), (1, 1.0)]);
    (eq_system, values, arg, params, border_conditions)
}

fn banded_aot_lifecycle_inputs() -> (
    Vec<Expr>,
    Vec<String>,
    String,
    HashMap<String, Vec<(usize, f64)>>,
) {
    let eq_system = vec![Expr::parse_expression("v"), Expr::parse_expression("-u")];
    let values = vec!["u".to_string(), "v".to_string()];
    let arg = "x".to_string();
    let mut border_conditions = HashMap::new();
    border_conditions.insert("u".to_string(), vec![(0, 0.0)]);
    border_conditions.insert("v".to_string(), vec![(0, 1.0)]);
    (eq_system, values, arg, border_conditions)
}

#[test]
fn transient_aot_failure_classifier_marks_windows_lock_and_spawn_failures() {
    assert!(super::is_transient_aot_infra_failure(
        "The process cannot access the file because it is being used by another process"
    ));
    assert!(super::is_transient_aot_infra_failure(
        "failed to spawn build runner: Access is denied"
    ));
    assert!(super::is_missing_aot_toolchain_failure(
        "failed to spawn build runner: The system cannot find the file specified. (os error 2)"
    ));
    assert!(!super::is_transient_aot_infra_failure(
        "failed to spawn build runner: The system cannot find the file specified. (os error 2)"
    ));
    assert!(!super::is_transient_aot_infra_failure(
        "error[E0425]: cannot find value `x` in this scope"
    ));
}

#[test]
fn execute_aot_build_with_retry_retries_transient_failures_only() {
    let transient_attempts = AtomicUsize::new(0);
    super::execute_aot_build_with_retry(
        || {
            let attempt = transient_attempts.fetch_add(1, Ordering::SeqCst);
            if attempt == 0 {
                Err("failed to spawn build runner: sharing violation".to_string())
            } else {
                Ok((true, Some(0), String::new(), String::new()))
            }
        },
        "test transient retry",
    )
    .expect("transient infrastructure failure should be retried");
    assert_eq!(transient_attempts.load(Ordering::SeqCst), 2);

    let deterministic_attempts = AtomicUsize::new(0);
    let err = super::execute_aot_build_with_retry(
        || {
            deterministic_attempts.fetch_add(1, Ordering::SeqCst);
            Ok((
                false,
                Some(1),
                String::new(),
                "error: expected expression".to_string(),
            ))
        },
        "test deterministic failure",
    )
    .expect_err("deterministic compiler failure should not be retried");
    assert_eq!(deterministic_attempts.load(Ordering::SeqCst), 1);
    assert!(err.contains("deterministic build failure"));

    let missing_toolchain_attempts = AtomicUsize::new(0);
    let err = super::execute_aot_build_with_retry(
            || {
                missing_toolchain_attempts.fetch_add(1, Ordering::SeqCst);
                Err(
                    "failed to spawn build runner: The system cannot find the file specified. (os error 2)"
                        .to_string(),
                )
            },
            "test missing toolchain",
        )
        .expect_err("missing toolchain should be reported without transient retries");
    assert_eq!(missing_toolchain_attempts.load(Ordering::SeqCst), 1);
    assert!(err.contains("missing or unreachable toolchain"));
    assert!(err.contains("check PATH"));
}

#[test]
fn sparse_build_if_missing_release_defaults_to_dev_fastest_compile_preset() {
    let config = super::GeneratedBackendConfig::sparse_build_if_missing_release();
    assert_eq!(
        config.symbolic_assembly_backend,
        BvpSymbolicAssemblyBackend::AtomView
    );
    assert_eq!(
        config.aot_build_policy,
        super::AotBuildPolicy::BuildIfMissing {
            profile: super::AotBuildProfile::Release,
        }
    );
    assert_eq!(config.aot_compile_config, AotCompileConfig::dev_fastest());
    assert_eq!(config.aot_codegen_backend, AotCodegenBackend::C);
    assert_eq!(config.aot_c_compiler.as_deref(), Some("tcc"));
}

#[test]
fn public_default_selects_atomview_and_tcc_aot() {
    let config = super::GeneratedBackendConfig::default();

    assert_eq!(
        config.symbolic_assembly_backend,
        BvpSymbolicAssemblyBackend::AtomView
    );
    assert_eq!(config.aot_codegen_backend, AotCodegenBackend::C);
    assert_eq!(config.aot_c_compiler.as_deref(), Some("tcc"));
    assert_eq!(
        config.aot_build_policy,
        super::AotBuildPolicy::UseIfAvailable
    );
}

#[test]
fn sparse_presets_allow_explicit_exprlegacy_compatibility_override() {
    let config = GeneratedBackendConfig::from_sparse_mode(SparseGeneratedBackendMode::Defaults)
        .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy);

    assert_eq!(
        config.symbolic_assembly_backend,
        BvpSymbolicAssemblyBackend::ExprLegacy
    );
    assert_eq!(
        config.effective_backend_policy("Sparse"),
        BackendSelectionPolicy::PreferAotThenLambdify
    );
}

fn baseline_sparse_residual(args: &Col<f64>) -> Vec<f64> {
    let (eq_system, values, arg, border_conditions) = real_bvp_inputs();
    let mut staged = Jacobian::new();
    staged.discretization_system_BVP_par(
        eq_system,
        values,
        arg,
        0.0,
        Some(6),
        None,
        None,
        border_conditions,
        None,
        None,
        "forward".to_string(),
    );
    staged.calc_jacobian_parallel_smart_optimized();
    let prepared_bridge = staged.prepare_sparse_aot_problem(
            "eval_bvp_residual",
            "eval_bvp_sparse_values",
            crate::symbolic::codegen::codegen_runtime_api::recommended_residual_chunking_for_parallelism(
                staged.vector_of_functions.len(),
                4,
            ),
            crate::symbolic::codegen::codegen_runtime_api::recommended_row_chunking_for_parallelism(
                staged.vector_of_functions.len(),
                4,
            ),
        );
    let input_names = prepared_bridge
        .variable_names
        .iter()
        .map(|name| name.as_str())
        .collect::<Vec<_>>();
    let args_vec = args.iter().copied().collect::<Vec<_>>();
    prepared_bridge
        .residuals
        .iter()
        .map(|expr| expr.lambdify_borrowed_thread_safe(&input_names)(&args_vec))
        .collect()
}

fn unique_test_artifact_dir(problem_key: &str) -> PathBuf {
    let sanitized = problem_key
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' => ch,
            _ => '_',
        })
        .collect::<String>();
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after unix epoch")
        .as_nanos();
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("test-artifacts")
        .join("generated-solver-handoff")
        .join(format!("{}-{}-{}", sanitized, std::process::id(), nonce))
}

fn register_offset_linked_backend(
    offset_residual: f64,
    offset_jacobian: f64,
) -> (AotResolver, String) {
    register_offset_linked_backend_with_chunk_offsets(
        offset_residual,
        offset_jacobian,
        offset_residual,
        offset_jacobian,
    )
}

fn register_offset_linked_backend_with_chunk_offsets(
    offset_residual: f64,
    offset_jacobian: f64,
    chunk_offset_residual: f64,
    chunk_offset_jacobian: f64,
) -> (AotResolver, String) {
    let (eq_system, values, arg, border_conditions) = real_bvp_inputs();
    let mut staged = Jacobian::new();
    staged.discretization_system_BVP_par(
        eq_system,
        values,
        arg,
        0.0,
        Some(6),
        None,
        None,
        border_conditions,
        None,
        None,
        "forward".to_string(),
    );
    staged.calc_jacobian_parallel_smart_optimized();
    let prepared_bridge = staged.prepare_sparse_aot_problem(
            "eval_bvp_residual",
            "eval_bvp_sparse_values",
            crate::symbolic::codegen::codegen_runtime_api::recommended_residual_chunking_for_parallelism(
                staged.vector_of_functions.len(),
                4,
            ),
            crate::symbolic::codegen::codegen_runtime_api::recommended_row_chunking_for_parallelism(
                staged.vector_of_functions.len(),
                4,
            ),
        );
    let problem_key = prepared_bridge.problem_key();
    let input_name_strings = prepared_bridge
        .variable_names
        .iter()
        .cloned()
        .collect::<Vec<_>>();
    let residual_input_name_strings = input_name_strings.clone();
    let jacobian_input_name_strings = input_name_strings.clone();
    let residual_exprs = prepared_bridge.residuals.clone();
    let sparse_exprs = prepared_bridge.sparse_entries.clone();
    let residual_len = prepared_bridge.shape.0;
    let shape = prepared_bridge.shape;
    let nnz = sparse_exprs.len();
    let prepared_sparse = prepared_bridge.as_prepared_problem();
    let residual_chunks = prepared_sparse
        .residual_plan
        .chunks
        .iter()
        .map(|chunk| {
            let input_name_strings = input_name_strings.clone();
            let chunk_exprs = chunk.residuals.iter().cloned().collect::<Vec<_>>();
            LinkedResidualChunk::new(
                chunk.output_offset,
                chunk.residuals.len(),
                Arc::new(move |args, out| {
                    let input_names = input_name_strings
                        .iter()
                        .map(|name| name.as_str())
                        .collect::<Vec<_>>();
                    for (slot, expr) in out.iter_mut().zip(chunk_exprs.iter()) {
                        *slot = expr.lambdify_borrowed_thread_safe(&input_names)(args)
                            + chunk_offset_residual;
                    }
                }),
            )
        })
        .collect::<Vec<_>>();
    let jacobian_value_chunks = prepared_sparse
        .jacobian_plan
        .chunks
        .iter()
        .map(|chunk| {
            let input_name_strings = input_name_strings.clone();
            let chunk_entries = chunk
                .entries
                .iter()
                .map(|entry| (entry.row, entry.col, entry.expr.clone()))
                .collect::<Vec<_>>();
            LinkedSparseJacobianChunk::new(
                chunk.value_offset,
                chunk.entries.len(),
                Arc::new(move |args, out| {
                    let input_names = input_name_strings
                        .iter()
                        .map(|name| name.as_str())
                        .collect::<Vec<_>>();
                    for (slot, (_, _, expr)) in out.iter_mut().zip(chunk_entries.iter()) {
                        *slot = expr.lambdify_borrowed_thread_safe(&input_names)(args)
                            + chunk_offset_jacobian;
                    }
                }),
            )
        })
        .collect::<Vec<_>>();

    register_linked_sparse_backend(
        LinkedSparseAotBackend::new(
            problem_key.clone(),
            residual_len,
            shape,
            nnz,
            Arc::new(move |args, out| {
                let input_names = residual_input_name_strings
                    .iter()
                    .map(|name| name.as_str())
                    .collect::<Vec<_>>();
                for (slot, expr) in out.iter_mut().zip(residual_exprs.iter()) {
                    *slot =
                        expr.lambdify_borrowed_thread_safe(&input_names)(args) + offset_residual;
                }
            }),
            Arc::new(move |args, out| {
                let input_names = jacobian_input_name_strings
                    .iter()
                    .map(|name| name.as_str())
                    .collect::<Vec<_>>();
                for (slot, (_, _, expr)) in out.iter_mut().zip(sparse_exprs.iter()) {
                    *slot =
                        expr.lambdify_borrowed_thread_safe(&input_names)(args) + offset_jacobian;
                }
            }),
        )
        .with_chunked_evaluators(residual_chunks, jacobian_value_chunks),
    );

    let prepared = PreparedProblem::sparse(prepared_bridge.as_prepared_problem());
    let manifest = PreparedProblemManifest::from(&prepared);
    let dir = unique_test_artifact_dir(&problem_key);
    fs::create_dir_all(&dir).expect("workspace test dir should be creatable");
    let build = AotBuildRequest::new(
        prepared_bridge.generated_aot_crate(
            "generated_bvp_solver_handoff_fixture",
            "generated_bvp_solver_handoff_module",
        ),
        dir.as_path(),
        AotBuildProfile::Release,
    )
    .materialize()
    .expect("build request should materialize");
    fs::create_dir_all(&build.artifact_dir).expect("artifact dir should be creatable");
    fs::write(&build.expected_rlib, b"fake rlib").expect("expected rlib should be writable");

    let mut registry = AotRegistry::new();
    registry.register_materialized_build(manifest, &build);
    (AotResolver::new(registry), problem_key)
}

fn register_parameterized_linked_backend() -> (AotResolver, String) {
    let (eq_system, values, arg, params, border_conditions) = parameterized_bvp_inputs();
    let param_refs = params.iter().map(|name| name.as_str()).collect::<Vec<_>>();
    let mut staged = Jacobian::new();
    staged.set_params(Some(param_refs.as_slice()));
    staged.discretization_system_BVP_par(
        eq_system,
        values,
        arg,
        0.0,
        Some(6),
        None,
        None,
        border_conditions,
        None,
        None,
        "forward".to_string(),
    );
    staged.calc_jacobian_parallel_smart_optimized();
    let prepared_bridge = staged.prepare_sparse_aot_problem(
            "eval_bvp_residual",
            "eval_bvp_sparse_values",
            crate::symbolic::codegen::codegen_runtime_api::recommended_residual_chunking_for_parallelism(
                staged.vector_of_functions.len(),
                4,
            ),
            crate::symbolic::codegen::codegen_runtime_api::recommended_row_chunking_for_parallelism(
                staged.vector_of_functions.len(),
                4,
            ),
        );
    let problem_key = prepared_bridge.problem_key();
    let input_name_strings = prepared_bridge
        .param_names
        .iter()
        .chain(prepared_bridge.variable_names.iter())
        .cloned()
        .collect::<Vec<_>>();
    let residual_input_name_strings = input_name_strings.clone();
    let jacobian_input_name_strings = input_name_strings.clone();
    let residual_exprs = prepared_bridge.residuals.clone();
    let sparse_exprs = prepared_bridge.sparse_entries.clone();
    let residual_len = prepared_bridge.shape.0;
    let shape = prepared_bridge.shape;
    let nnz = sparse_exprs.len();

    register_linked_sparse_backend(LinkedSparseAotBackend::new(
        problem_key.clone(),
        residual_len,
        shape,
        nnz,
        Arc::new(move |args, out| {
            let input_names = residual_input_name_strings
                .iter()
                .map(|name| name.as_str())
                .collect::<Vec<_>>();
            for (slot, expr) in out.iter_mut().zip(residual_exprs.iter()) {
                *slot = expr.lambdify_borrowed_thread_safe(&input_names)(args);
            }
        }),
        Arc::new(move |args, out| {
            let input_names = jacobian_input_name_strings
                .iter()
                .map(|name| name.as_str())
                .collect::<Vec<_>>();
            for (slot, (_, _, expr)) in out.iter_mut().zip(sparse_exprs.iter()) {
                *slot = expr.lambdify_borrowed_thread_safe(&input_names)(args);
            }
        }),
    ));

    let prepared = PreparedProblem::sparse(prepared_bridge.as_prepared_problem());
    let manifest = PreparedProblemManifest::from(&prepared);
    let dir = unique_test_artifact_dir(&problem_key);
    fs::create_dir_all(&dir).expect("workspace test dir should be creatable");
    let build = AotBuildRequest::new(
        prepared_bridge.generated_aot_crate(
            "generated_bvp_solver_parameterized_fixture",
            "generated_bvp_solver_parameterized_module",
        ),
        dir.as_path(),
        AotBuildProfile::Release,
    )
    .materialize()
    .expect("parameterized build request should materialize");
    fs::create_dir_all(&build.artifact_dir).expect("artifact dir should be creatable");
    fs::write(&build.expected_rlib, b"fake rlib").expect("expected rlib should be writable");

    let mut registry = AotRegistry::new();
    registry.register_materialized_build(manifest, &build);
    (AotResolver::new(registry), problem_key)
}

#[test]
fn damped_solver_handoff_prefers_callable_linked_aot_backend() {
    let _guard = linked_runtime_registry_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let (resolver, problem_key) = register_offset_linked_backend(100.0, 200.0);
    let (eq_system, values, arg, border_conditions) = real_bvp_inputs();
    let mut probe = Jacobian::new();
    let execution = probe.generate_BVP_with_backend_selection(
        eq_system.clone(),
        values.clone(),
        arg.clone(),
        None,
        0.0,
        None,
        Some(6),
        None,
        None,
        border_conditions.clone(),
        None,
        None,
        "forward".to_string(),
        "Sparse".to_string(),
        Some((2, 2)),
        BackendSelectionPolicy::PreferAotThenLambdify,
        Some(&resolver),
    );
    assert_eq!(
        execution.selected().prepared_problem.problem_key(),
        problem_key
    );
    assert_eq!(
        execution.selected().effective_backend,
        crate::symbolic::codegen::codegen_backend_selection::SelectedBackendKind::AotCompiled
    );
    let request = DampedSolverBuildRequest {
        eq_system,
        values,
        arg,
        param_names: None,
        param_values: None,
        t0: 0.0,
        n_steps: Some(6),
        h: None,
        mesh: None,
        border_conditions,
        bounds: None,
        rel_tolerance: None,
        scheme: "forward".to_string(),
        method: "Sparse".to_string(),
        bandwidth: Some((2, 2)),
        backend_policy: BackendSelectionPolicy::PreferAotThenLambdify,
        resolver: Some(resolver.clone()),
        aot_execution_policy: AotExecutionPolicy::Auto,
        aot_build_policy: AotBuildPolicy::UseIfAvailable,
        aot_compile_config: AotCompileConfig::default(),
        aot_codegen_backend: AotCodegenBackend::Rust,
        aot_c_compiler: None,
        aot_chunking_policy: AotChunkingPolicy::default(),
        atom_optimization_profile: AtomOptimizationProfile::Full,
        symbolic_assembly_backend: BvpSymbolicAssemblyBackend::ExprLegacy,
        matrix_backend_override: None,
        banded_linear_solver_config: crate::somelinalg::banded::LinearSolverConfig::default(),
        lambdify_telemetry_mode: BvpLambdifyTelemetryMode::Off,
        aot_telemetry_mode: BvpAotTelemetryMode::Off,
        lambdify_execution_policy: BvpLambdifyExecutionPolicy::default(),
    };

    let state = request.generate().expect("damped handoff should build");
    let y = Col::from_fn(state.variable_string.len(), |index| {
        0.2 + index as f64 * 0.01
    });
    let residual = state.fun.call(0.0, &y).to_DVectorType();
    let baseline_residual = baseline_sparse_residual(&y);

    for (actual, expected) in residual.iter().zip(baseline_residual.iter()) {
        assert!((actual - (expected + 100.0)).abs() < 1e-10);
    }
    assert!(state.jac.is_some(), "jacobian callback should be present");
    assert!(
        state
            .runtime_diagnostics
            .get("generated.handoff.initial_generate_wall_ms")
            .and_then(|value| value.parse::<f64>().ok())
            .is_some_and(|value| value >= 0.0),
        "damped handoff must expose the initial generated-backend wall-clock stage"
    );
    assert!(
        state
            .runtime_diagnostics
            .get("generated.handoff.initial.symbolic_jacobian_time_ms")
            .and_then(|value| value.parse::<f64>().ok())
            .is_some_and(|value| value >= 0.0),
        "damped handoff must preserve the internal symbolic Jacobian stage timing"
    );
    for stage in [
        "symbolic_jacobian_variable_sets_time_ms",
        "symbolic_jacobian_row_differentiation_time_ms",
        "symbolic_jacobian_dense_cache_materialize_time_ms",
        "symbolic_jacobian_sparse_cache_flatten_time_ms",
    ] {
        let key = format!("generated.handoff.initial.{stage}");
        assert!(
            state
                .runtime_diagnostics
                .get(&key)
                .and_then(|value| value.parse::<f64>().ok())
                .is_some_and(|value| value >= 0.0),
            "damped handoff must expose detailed symbolic Jacobian stage {key}"
        );
    }

    unregister_linked_sparse_backend(&problem_key);
}

#[test]
fn frozen_solver_handoff_prefers_callable_linked_aot_backend() {
    let _guard = linked_runtime_registry_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let (resolver, problem_key) = register_offset_linked_backend(50.0, 75.0);
    let (eq_system, values, arg, border_conditions) = real_bvp_inputs();
    let mut probe = Jacobian::new();
    let execution = probe.generate_BVP_with_backend_selection(
        eq_system.clone(),
        values.clone(),
        arg.clone(),
        None,
        0.0,
        None,
        Some(6),
        None,
        None,
        border_conditions.clone(),
        None,
        None,
        "forward".to_string(),
        "Sparse".to_string(),
        Some((2, 2)),
        BackendSelectionPolicy::PreferAotThenLambdify,
        Some(&resolver),
    );
    assert_eq!(
        execution.selected().prepared_problem.problem_key(),
        problem_key
    );
    assert_eq!(
        execution.selected().effective_backend,
        crate::symbolic::codegen::codegen_backend_selection::SelectedBackendKind::AotCompiled
    );
    let request = FrozenSolverBuildRequest {
        eq_system,
        values,
        arg,
        param_names: None,
        param_values: None,
        t0: 0.0,
        n_steps: Some(6),
        h: None,
        mesh: None,
        border_conditions,
        scheme: "forward".to_string(),
        method: "Sparse".to_string(),
        bandwidth: Some((2, 2)),
        backend_policy: BackendSelectionPolicy::PreferAotThenLambdify,
        resolver: Some(resolver.clone()),
        aot_execution_policy: AotExecutionPolicy::Auto,
        aot_build_policy: AotBuildPolicy::UseIfAvailable,
        aot_compile_config: AotCompileConfig::default(),
        aot_codegen_backend: AotCodegenBackend::Rust,
        aot_c_compiler: None,
        aot_chunking_policy: AotChunkingPolicy::default(),
        atom_optimization_profile: AtomOptimizationProfile::Full,
        symbolic_assembly_backend: BvpSymbolicAssemblyBackend::ExprLegacy,
        matrix_backend_override: None,
        banded_linear_solver_config: crate::somelinalg::banded::LinearSolverConfig::default(),
        lambdify_telemetry_mode: BvpLambdifyTelemetryMode::Off,
        aot_telemetry_mode: BvpAotTelemetryMode::Off,
        lambdify_execution_policy: BvpLambdifyExecutionPolicy::default(),
    };

    let state = request.generate().expect("frozen handoff should build");
    let y = Col::from_fn(state.variable_string.len(), |index| {
        0.3 + index as f64 * 0.02
    });
    let residual = state.fun.call(0.0, &y).to_DVectorType();
    let baseline_residual = baseline_sparse_residual(&y);

    for (actual, expected) in residual.iter().zip(baseline_residual.iter()) {
        assert!((actual - (expected + 50.0)).abs() < 1e-10);
    }
    assert!(state.jac.is_some(), "jacobian callback should be present");
    assert_eq!(state.selected_backend, SelectedBackendKind::AotCompiled);
    assert!(
        state
            .runtime_diagnostics
            .contains_key("generated.handoff.initial_generate_wall_ms"),
        "frozen handoff must export the same initial generation stage as damped handoff"
    );
    assert_eq!(
        state
            .runtime_diagnostics
            .get("aot.runtime.execution_policy")
            .map(String::as_str),
        Some("Auto"),
        "frozen handoff must keep linked runtime callback diagnostics"
    );

    unregister_linked_sparse_backend(&problem_key);
}

#[test]
fn damped_solver_handoff_parallel_policy_uses_chunked_linked_backend_callbacks() {
    let _guard = linked_runtime_registry_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let (resolver, problem_key) =
        register_offset_linked_backend_with_chunk_offsets(100.0, 200.0, 300.0, 400.0);
    let (eq_system, values, arg, border_conditions) = real_bvp_inputs();
    let request = DampedSolverBuildRequest {
        eq_system,
        values,
        arg,
        param_names: None,
        param_values: None,
        t0: 0.0,
        n_steps: Some(6),
        h: None,
        mesh: None,
        border_conditions,
        bounds: None,
        rel_tolerance: None,
        scheme: "forward".to_string(),
        method: "Sparse".to_string(),
        bandwidth: Some((2, 2)),
        backend_policy: BackendSelectionPolicy::PreferAotThenLambdify,
        resolver: Some(resolver),
        aot_execution_policy: AotExecutionPolicy::Parallel(ParallelExecutorConfig {
            jobs_per_worker: 1,
            max_residual_jobs: Some(2),
            max_sparse_jobs: Some(2),
            fallback_policy: ParallelFallbackPolicy::Never,
        }),
        aot_build_policy: AotBuildPolicy::UseIfAvailable,
        aot_compile_config: AotCompileConfig::default(),
        aot_codegen_backend: AotCodegenBackend::Rust,
        aot_c_compiler: None,
        aot_chunking_policy: AotChunkingPolicy::default(),
        atom_optimization_profile: AtomOptimizationProfile::Full,
        symbolic_assembly_backend: BvpSymbolicAssemblyBackend::ExprLegacy,
        matrix_backend_override: None,
        banded_linear_solver_config: crate::somelinalg::banded::LinearSolverConfig::default(),
        lambdify_telemetry_mode: BvpLambdifyTelemetryMode::Off,
        aot_telemetry_mode: BvpAotTelemetryMode::Off,
        lambdify_execution_policy: BvpLambdifyExecutionPolicy::default(),
    };

    let state = request
        .generate()
        .expect("parallel damped handoff should build");
    let y = Col::from_fn(state.variable_string.len(), |index| {
        0.2 + index as f64 * 0.01
    });
    let residual = state.fun.call(0.0, &y).to_DVectorType();
    let baseline_residual = baseline_sparse_residual(&y);

    for (actual, expected) in residual.iter().zip(baseline_residual.iter()) {
        assert!((actual - (expected + 300.0)).abs() < 1e-10);
    }
    assert_eq!(
        state
            .runtime_diagnostics
            .get("aot.runtime.execution_policy")
            .map(String::as_str),
        Some("Parallel")
    );
    assert_eq!(
        state
            .runtime_diagnostics
            .get("aot.runtime.parallel_requested")
            .map(String::as_str),
        Some("true")
    );
    assert!(
        state
            .runtime_diagnostics
            .get("aot.auto.min_work_per_job")
            .and_then(|value| value.parse::<usize>().ok())
            .is_some_and(|work| work > 0),
        "runtime diagnostics should include the machine-aware Auto threshold"
    );
    assert!(
        state
            .runtime_diagnostics
            .contains_key("aot.auto.sparse_jacobian.reason"),
        "runtime diagnostics should explain the Auto sparse-Jacobian decision"
    );
    assert_eq!(
        state
            .runtime_diagnostics
            .get("aot.runtime.residual.actual_jobs")
            .map(String::as_str),
        Some("2")
    );
    assert_eq!(
        state
            .runtime_diagnostics
            .get("aot.runtime.residual.fallback")
            .map(String::as_str),
        Some("false")
    );
    assert_eq!(
        state
            .runtime_diagnostics
            .get("aot.runtime.residual.fallback_reason")
            .map(String::as_str),
        Some("none")
    );
    assert!(
        state
            .runtime_diagnostics
            .get("aot.runtime.residual.work_per_job")
            .and_then(|value| value.parse::<usize>().ok())
            .is_some_and(|work| work > 0),
        "parallel residual diagnostics must expose non-zero work_per_job"
    );
    assert_eq!(
        state
            .runtime_diagnostics
            .get("aot.runtime.sparse_jacobian.actual_jobs")
            .map(String::as_str),
        Some("2")
    );
    assert_eq!(
        state
            .runtime_diagnostics
            .get("aot.runtime.sparse_jacobian.fallback")
            .map(String::as_str),
        Some("false")
    );
    assert_eq!(
        state
            .runtime_diagnostics
            .get("aot.runtime.sparse_jacobian.fallback_reason")
            .map(String::as_str),
        Some("none")
    );
    assert!(
        state
            .runtime_diagnostics
            .get("aot.runtime.sparse_jacobian.work_per_job")
            .and_then(|value| value.parse::<usize>().ok())
            .is_some_and(|work| work > 0),
        "parallel sparse-Jacobian diagnostics must expose non-zero work_per_job"
    );

    unregister_linked_sparse_backend(&problem_key);
}

#[test]
fn require_prebuilt_errors_when_compiled_backend_is_missing() {
    let (eq_system, values, arg, border_conditions) = real_bvp_inputs();
    let request = DampedSolverBuildRequest {
        eq_system,
        values,
        arg,
        param_names: None,
        param_values: None,
        t0: 0.0,
        n_steps: Some(6),
        h: None,
        mesh: None,
        border_conditions,
        bounds: None,
        rel_tolerance: None,
        scheme: "forward".to_string(),
        method: "Sparse".to_string(),
        bandwidth: Some((2, 2)),
        backend_policy: BackendSelectionPolicy::PreferAotThenLambdify,
        resolver: None,
        aot_execution_policy: AotExecutionPolicy::Auto,
        aot_build_policy: AotBuildPolicy::RequirePrebuilt,
        aot_compile_config: AotCompileConfig::default(),
        aot_codegen_backend: AotCodegenBackend::Rust,
        aot_c_compiler: None,
        aot_chunking_policy: AotChunkingPolicy::default(),
        atom_optimization_profile: AtomOptimizationProfile::Full,
        symbolic_assembly_backend: BvpSymbolicAssemblyBackend::ExprLegacy,
        matrix_backend_override: None,
        banded_linear_solver_config: crate::somelinalg::banded::LinearSolverConfig::default(),
        lambdify_telemetry_mode: BvpLambdifyTelemetryMode::Off,
        aot_telemetry_mode: BvpAotTelemetryMode::Off,
        lambdify_execution_policy: BvpLambdifyExecutionPolicy::default(),
    };

    let err = request
        .generate()
        .err()
        .expect("require-prebuilt should reject missing compiled AOT");
    assert!(matches!(
        err,
        BvpBackendIntegrationError::CompiledAotRequiredButUnavailable {
            effective_backend: SelectedBackendKind::AotMissing,
            ..
        }
    ));
}

#[test]
fn generated_backend_config_can_select_non_rust_codegen_backend() {
    let config = super::GeneratedBackendConfig::sparse_build_if_missing_release()
        .with_aot_codegen_backend(AotCodegenBackend::C);

    assert_eq!(config.aot_codegen_backend, AotCodegenBackend::C);
    assert_eq!(
        config.aot_build_policy,
        AotBuildPolicy::BuildIfMissing {
            profile: super::AotBuildProfile::Release,
        }
    );
}

#[test]
fn banded_generated_backend_defaults_select_faithful_lapack_without_refinement() {
    let config = GeneratedBackendConfig::from_banded_mode(BandedGeneratedBackendMode::Defaults);

    assert_eq!(
        config.symbolic_assembly_backend,
        BvpSymbolicAssemblyBackend::AtomView
    );
    assert_eq!(
        config.matrix_backend_override,
        Some(crate::symbolic::codegen::codegen_provider_api::MatrixBackend::Banded)
    );
    assert_eq!(
        config.banded_linear_solver_config.policy,
        crate::somelinalg::banded::LinearSolverPolicy::ForceBanded
    );
    assert_eq!(
        config
            .banded_linear_solver_config
            .iterative_refinement_steps,
        0
    );
    assert_eq!(config.effective_method("Sparse"), "Banded");
    assert_eq!(
        config.effective_backend_policy("Banded"),
        BackendSelectionPolicy::PreferAotThenLambdify
    );
}

#[test]
fn banded_lambdify_preset_keeps_banded_matrix_backend() {
    let config = GeneratedBackendConfig::from_banded_mode(BandedGeneratedBackendMode::Lambdify);

    assert_eq!(
        config.symbolic_assembly_backend,
        BvpSymbolicAssemblyBackend::AtomView
    );
    assert_eq!(config.effective_method("Sparse"), "Banded");
    assert_eq!(
        config.effective_backend_policy("Banded"),
        BackendSelectionPolicy::LambdifyOnly
    );
    assert_eq!(
        config
            .banded_linear_solver_config
            .iterative_refinement_steps,
        0
    );
}

#[test]
fn banded_presets_allow_explicit_exprlegacy_compatibility_override() {
    let config = GeneratedBackendConfig::from_banded_mode(BandedGeneratedBackendMode::Lambdify)
        .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy);

    assert_eq!(
        config.symbolic_assembly_backend,
        BvpSymbolicAssemblyBackend::ExprLegacy
    );
    assert_eq!(config.effective_method("Sparse"), "Banded");
    assert_eq!(
        config.effective_backend_policy("Banded"),
        BackendSelectionPolicy::LambdifyOnly
    );
}

#[test]
fn build_if_missing_materializes_generated_crate_and_keeps_callable_runtime() {
    let _guard = linked_runtime_registry_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let (eq_system, values, arg, border_conditions) = real_bvp_inputs();
    let mut staged = Jacobian::new();
    staged.discretization_system_BVP_par(
        eq_system.clone(),
        values.clone(),
        arg.clone(),
        0.0,
        Some(6),
        None,
        None,
        border_conditions.clone(),
        None,
        None,
        "forward".to_string(),
    );
    staged.calc_jacobian_parallel_smart_optimized();
    let prepared_bridge = staged.prepare_sparse_aot_problem(
            "eval_bvp_residual",
            "eval_bvp_sparse_values",
            crate::symbolic::codegen::codegen_runtime_api::recommended_residual_chunking_for_parallelism(
                staged.vector_of_functions.len(),
                4,
            ),
            crate::symbolic::codegen::codegen_runtime_api::recommended_row_chunking_for_parallelism(
                staged.vector_of_functions.len(),
                4,
            ),
        );
    let problem_key = prepared_bridge.problem_key();
    let request = FrozenSolverBuildRequest {
        eq_system,
        values,
        arg,
        param_names: None,
        param_values: None,
        t0: 0.0,
        n_steps: Some(6),
        h: None,
        mesh: None,
        border_conditions,
        scheme: "forward".to_string(),
        method: "Sparse".to_string(),
        bandwidth: Some((2, 2)),
        backend_policy: BackendSelectionPolicy::PreferAotThenLambdify,
        resolver: None,
        aot_execution_policy: AotExecutionPolicy::Auto,
        aot_build_policy: AotBuildPolicy::BuildIfMissing {
            profile: crate::numerical::BVP_Damp::generated_solver_handoff::AotBuildProfile::Release,
        },
        aot_compile_config: AotCompileConfig::default(),
        aot_codegen_backend: AotCodegenBackend::Rust,
        aot_c_compiler: None,
        aot_chunking_policy: AotChunkingPolicy::default(),
        atom_optimization_profile: AtomOptimizationProfile::Full,
        symbolic_assembly_backend: BvpSymbolicAssemblyBackend::ExprLegacy,
        matrix_backend_override: None,
        banded_linear_solver_config: crate::somelinalg::banded::LinearSolverConfig::default(),
        lambdify_telemetry_mode: BvpLambdifyTelemetryMode::Off,
        aot_telemetry_mode: BvpAotTelemetryMode::Off,
        lambdify_execution_policy: BvpLambdifyExecutionPolicy::default(),
    };

    let state = request
        .generate()
        .expect("build-if-missing should build artifact and keep runtime callable");
    assert_eq!(
        state.selected_backend,
        SelectedBackendKind::AotCompiled,
        "BuildIfMissing should use the freshly built compiled backend in the current handoff, not only in a later solve"
    );
    assert!(
        state.jac.is_some(),
        "jacobian callback should stay callable"
    );
    let updated_resolver = state
        .updated_resolver
        .as_ref()
        .expect("build-if-missing should return an updated resolver snapshot");
    let resolved = updated_resolver.resolve_by_problem_key(&problem_key);
    assert!(
        resolved.is_compiled(),
        "updated resolver should see the freshly built compiled artifact"
    );

    let expected_rlib = &resolved.registered.expected_rlib;
    assert!(
        expected_rlib.exists(),
        "build-if-missing should produce compiled rlib at {}",
        expected_rlib.display()
    );
}

#[test]
fn banded_build_if_missing_then_require_prebuilt_keeps_compiled_backend() {
    let _guard = linked_runtime_registry_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let (eq_system, values, arg, border_conditions) = banded_aot_lifecycle_inputs();

    let make_request =
        |resolver: Option<AotResolver>, build_policy: AotBuildPolicy| DampedSolverBuildRequest {
            eq_system: eq_system.clone(),
            values: values.clone(),
            arg: arg.clone(),
            param_names: None,
            param_values: None,
            t0: 0.0,
            n_steps: Some(8),
            h: None,
            mesh: None,
            border_conditions: border_conditions.clone(),
            bounds: None,
            rel_tolerance: None,
            scheme: "forward".to_string(),
            method: "Banded".to_string(),
            bandwidth: Some((6, 6)),
            backend_policy: BackendSelectionPolicy::PreferAotThenLambdify,
            resolver,
            aot_execution_policy: AotExecutionPolicy::SequentialOnly,
            aot_build_policy: build_policy,
            aot_compile_config: AotCompileConfig::dev_fastest(),
            aot_codegen_backend: AotCodegenBackend::Rust,
            aot_c_compiler: None,
            aot_chunking_policy: AotChunkingPolicy::default(),
            atom_optimization_profile: AtomOptimizationProfile::Full,
            symbolic_assembly_backend: BvpSymbolicAssemblyBackend::AtomView,
            matrix_backend_override: Some(MatrixBackend::Banded),
            banded_linear_solver_config:
                crate::somelinalg::banded::LinearSolverConfig::faithful_banded(),
            lambdify_telemetry_mode: BvpLambdifyTelemetryMode::Off,
            aot_telemetry_mode: BvpAotTelemetryMode::Off,
            lambdify_execution_policy: BvpLambdifyExecutionPolicy::default(),
        };

    let built_state = make_request(
        None,
        AotBuildPolicy::BuildIfMissing {
            profile: super::AotBuildProfile::Debug,
        },
    )
    .generate()
    .expect("banded BuildIfMissing should materialize a callable generated backend");
    assert_eq!(
        built_state.selected_backend,
        SelectedBackendKind::AotCompiled,
        "banded BuildIfMissing must use the compiled backend immediately"
    );
    assert!(
        built_state.jac.is_some(),
        "compiled banded handoff must provide a Jacobian callback"
    );
    assert!(
        built_state
            .runtime_diagnostics
            .contains_key("generated.handoff.post_build_rebind_wall_ms"),
        "freshly built AOT backend must be attached by direct runtime rebinding"
    );
    assert!(
        !built_state
            .runtime_diagnostics
            .contains_key("generated.handoff.post_build_regenerate_wall_ms"),
        "freshly built AOT backend must not trigger a second symbolic generation pass"
    );

    let resolver = built_state
        .updated_resolver
        .expect("banded BuildIfMissing should return resolver snapshot for reuse");
    let strict_state = make_request(Some(resolver), AotBuildPolicy::RequirePrebuilt)
        .generate()
        .expect("banded RequirePrebuilt should reuse the resolver without Lambdify fallback");
    assert_eq!(
        strict_state.selected_backend,
        SelectedBackendKind::AotCompiled,
        "banded RequirePrebuilt must stay on the compiled backend"
    );
    assert!(
        strict_state.jac.is_some(),
        "prebuilt banded handoff must keep the Jacobian callback available"
    );
}

#[test]
fn frozen_banded_build_if_missing_then_require_prebuilt_keeps_compiled_backend() {
    let (eq_system, values, arg, border_conditions) = banded_aot_lifecycle_inputs();

    let make_request =
        |resolver: Option<AotResolver>, build_policy: AotBuildPolicy| FrozenSolverBuildRequest {
            eq_system: eq_system.clone(),
            values: values.clone(),
            arg: arg.clone(),
            param_names: None,
            param_values: None,
            t0: 0.0,
            n_steps: Some(8),
            h: None,
            mesh: None,
            border_conditions: border_conditions.clone(),
            scheme: "forward".to_string(),
            method: "Banded".to_string(),
            bandwidth: Some((6, 6)),
            backend_policy: BackendSelectionPolicy::PreferAotThenLambdify,
            resolver,
            aot_execution_policy: AotExecutionPolicy::SequentialOnly,
            aot_build_policy: build_policy,
            aot_compile_config: AotCompileConfig::dev_fastest(),
            aot_codegen_backend: AotCodegenBackend::Rust,
            aot_c_compiler: None,
            aot_chunking_policy: AotChunkingPolicy::default(),
            atom_optimization_profile: AtomOptimizationProfile::Full,
            symbolic_assembly_backend: BvpSymbolicAssemblyBackend::AtomView,
            matrix_backend_override: Some(MatrixBackend::Banded),
            banded_linear_solver_config:
                crate::somelinalg::banded::LinearSolverConfig::faithful_banded(),
            lambdify_telemetry_mode: BvpLambdifyTelemetryMode::Off,
            aot_telemetry_mode: BvpAotTelemetryMode::Off,
            lambdify_execution_policy: BvpLambdifyExecutionPolicy::default(),
        };

    let built_state = make_request(
        None,
        AotBuildPolicy::BuildIfMissing {
            profile: super::AotBuildProfile::Debug,
        },
    )
    .generate()
    .expect("frozen banded BuildIfMissing should materialize a callable backend");
    assert_eq!(
        built_state.selected_backend,
        SelectedBackendKind::AotCompiled,
        "frozen banded BuildIfMissing must use the compiled backend immediately"
    );
    assert!(
        built_state.jac.is_some(),
        "compiled frozen banded handoff must provide a Jacobian callback"
    );
    assert!(
        built_state
            .runtime_diagnostics
            .contains_key("generated.handoff.post_build_rebind_wall_ms"),
        "freshly built frozen AOT backend must report direct runtime rebinding"
    );

    let resolver = built_state
        .updated_resolver
        .expect("frozen banded BuildIfMissing should return resolver snapshot for reuse");
    let strict_state = make_request(Some(resolver), AotBuildPolicy::RequirePrebuilt)
        .generate()
        .expect("frozen banded RequirePrebuilt should reuse compiled backend");
    assert_eq!(
        strict_state.selected_backend,
        SelectedBackendKind::AotCompiled,
        "frozen banded RequirePrebuilt must stay on the compiled backend"
    );
    assert!(
        strict_state.jac.is_some(),
        "prebuilt frozen banded handoff must keep Jacobian callback available"
    );
    assert!(
        strict_state
            .runtime_diagnostics
            .contains_key("generated.handoff.initial_generate_wall_ms"),
        "prebuilt frozen AOT handoff must preserve lifecycle diagnostics"
    );
}

#[test]
fn require_prebuilt_errors_when_artifact_is_registered_but_not_built() {
    let (eq_system, values, arg, border_conditions) = real_bvp_inputs();
    let mut staged = Jacobian::new();
    staged.discretization_system_BVP_par(
        eq_system.clone(),
        values.clone(),
        arg.clone(),
        0.0,
        Some(6),
        None,
        None,
        border_conditions.clone(),
        None,
        None,
        "forward".to_string(),
    );
    staged.calc_jacobian_parallel_smart_optimized();
    let prepared_bridge = staged.prepare_sparse_aot_problem(
            "eval_bvp_residual",
            "eval_bvp_sparse_values",
            crate::symbolic::codegen::codegen_runtime_api::recommended_residual_chunking_for_parallelism(
                staged.vector_of_functions.len(),
                4,
            ),
            crate::symbolic::codegen::codegen_runtime_api::recommended_row_chunking_for_parallelism(
                staged.vector_of_functions.len(),
                4,
            ),
        );
    let prepared = PreparedProblem::sparse(prepared_bridge.as_prepared_problem());
    let manifest = PreparedProblemManifest::from(&prepared);
    let dir = unique_test_artifact_dir(&prepared_bridge.problem_key());
    fs::create_dir_all(&dir).expect("workspace test dir should be creatable");
    let build = AotBuildRequest::new(
        prepared_bridge.generated_aot_crate(
            "generated_bvp_solver_registered_only_fixture",
            "generated_bvp_solver_registered_only_module",
        ),
        dir.as_path(),
        AotBuildProfile::Release,
    )
    .materialize()
    .expect("build request should materialize");

    let mut registry = AotRegistry::new();
    registry.register_materialized_build(manifest, &build);
    let resolver = AotResolver::new(registry);

    let request = DampedSolverBuildRequest {
        eq_system,
        values,
        arg,
        param_names: None,
        param_values: None,
        t0: 0.0,
        n_steps: Some(6),
        h: None,
        mesh: None,
        border_conditions,
        bounds: None,
        rel_tolerance: None,
        scheme: "forward".to_string(),
        method: "Sparse".to_string(),
        bandwidth: Some((2, 2)),
        backend_policy: BackendSelectionPolicy::PreferAotThenLambdify,
        resolver: Some(resolver),
        aot_execution_policy: AotExecutionPolicy::Auto,
        aot_build_policy: AotBuildPolicy::RequirePrebuilt,
        aot_compile_config: AotCompileConfig::default(),
        aot_codegen_backend: AotCodegenBackend::Rust,
        aot_c_compiler: None,
        aot_chunking_policy: AotChunkingPolicy::default(),
        atom_optimization_profile: AtomOptimizationProfile::Full,
        symbolic_assembly_backend: BvpSymbolicAssemblyBackend::ExprLegacy,
        matrix_backend_override: None,
        banded_linear_solver_config: crate::somelinalg::banded::LinearSolverConfig::default(),
        lambdify_telemetry_mode: BvpLambdifyTelemetryMode::Off,
        aot_telemetry_mode: BvpAotTelemetryMode::Off,
        lambdify_execution_policy: BvpLambdifyExecutionPolicy::default(),
    };

    let err = request
        .generate()
        .err()
        .expect("require-prebuilt should reject not-built compiled AOT");
    assert!(matches!(
        err,
        BvpBackendIntegrationError::CompiledAotRequiredButUnavailable {
            effective_backend: SelectedBackendKind::AotRegisteredButNotBuilt,
            ..
        }
    ));
}

#[test]
fn damped_solver_handoff_reuses_compiled_backend_when_only_param_values_change() {
    let _guard = linked_runtime_registry_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let (resolver, problem_key) = register_parameterized_linked_backend();
    let (eq_system, values, arg, params, border_conditions) = parameterized_bvp_inputs();
    let param_refs = params.iter().map(|name| name.as_str()).collect::<Vec<_>>();

    let mut probe_a = Jacobian::new();
    probe_a.set_params(Some(param_refs.as_slice()));
    probe_a.set_param_values(Some(vec![1.0]));
    let execution_a = probe_a.generate_BVP_with_backend_selection(
        eq_system.clone(),
        values.clone(),
        arg.clone(),
        Some(param_refs.as_slice()),
        0.0,
        None,
        Some(6),
        None,
        None,
        border_conditions.clone(),
        None,
        None,
        "forward".to_string(),
        "Sparse".to_string(),
        Some((2, 2)),
        BackendSelectionPolicy::PreferAotThenLambdify,
        Some(&resolver),
    );

    let mut probe_b = Jacobian::new();
    probe_b.set_params(Some(param_refs.as_slice()));
    probe_b.set_param_values(Some(vec![3.0]));
    let execution_b = probe_b.generate_BVP_with_backend_selection(
        eq_system.clone(),
        values.clone(),
        arg.clone(),
        Some(param_refs.as_slice()),
        0.0,
        None,
        Some(6),
        None,
        None,
        border_conditions.clone(),
        None,
        None,
        "forward".to_string(),
        "Sparse".to_string(),
        Some((2, 2)),
        BackendSelectionPolicy::PreferAotThenLambdify,
        Some(&resolver),
    );

    assert_eq!(
        execution_a.selected().prepared_problem.problem_key(),
        problem_key
    );
    assert_eq!(
        execution_b.selected().prepared_problem.problem_key(),
        problem_key
    );

    let request_a = DampedSolverBuildRequest {
        eq_system: eq_system.clone(),
        values: values.clone(),
        arg: arg.clone(),
        param_names: Some(params.clone()),
        param_values: Some(vec![1.0]),
        t0: 0.0,
        n_steps: Some(6),
        h: None,
        mesh: None,
        border_conditions: border_conditions.clone(),
        bounds: None,
        rel_tolerance: None,
        scheme: "forward".to_string(),
        method: "Sparse".to_string(),
        bandwidth: Some((2, 2)),
        backend_policy: BackendSelectionPolicy::PreferAotThenLambdify,
        resolver: Some(resolver.clone()),
        aot_execution_policy: AotExecutionPolicy::Auto,
        aot_build_policy: AotBuildPolicy::RequirePrebuilt,
        aot_compile_config: AotCompileConfig::default(),
        aot_codegen_backend: AotCodegenBackend::Rust,
        aot_c_compiler: None,
        aot_chunking_policy: AotChunkingPolicy::default(),
        atom_optimization_profile: AtomOptimizationProfile::Full,
        symbolic_assembly_backend: BvpSymbolicAssemblyBackend::ExprLegacy,
        matrix_backend_override: None,
        banded_linear_solver_config: crate::somelinalg::banded::LinearSolverConfig::default(),
        lambdify_telemetry_mode: BvpLambdifyTelemetryMode::Off,
        aot_telemetry_mode: BvpAotTelemetryMode::Off,
        lambdify_execution_policy: BvpLambdifyExecutionPolicy::default(),
    };

    let request_b = DampedSolverBuildRequest {
        eq_system,
        values,
        arg,
        param_names: Some(params),
        param_values: Some(vec![3.0]),
        t0: 0.0,
        n_steps: Some(6),
        h: None,
        mesh: None,
        border_conditions,
        bounds: None,
        rel_tolerance: None,
        scheme: "forward".to_string(),
        method: "Sparse".to_string(),
        bandwidth: Some((2, 2)),
        backend_policy: BackendSelectionPolicy::PreferAotThenLambdify,
        resolver: Some(resolver),
        aot_execution_policy: AotExecutionPolicy::Auto,
        aot_build_policy: AotBuildPolicy::RequirePrebuilt,
        aot_compile_config: AotCompileConfig::default(),
        aot_codegen_backend: AotCodegenBackend::Rust,
        aot_c_compiler: None,
        aot_chunking_policy: AotChunkingPolicy::default(),
        atom_optimization_profile: AtomOptimizationProfile::Full,
        symbolic_assembly_backend: BvpSymbolicAssemblyBackend::ExprLegacy,
        matrix_backend_override: None,
        banded_linear_solver_config: crate::somelinalg::banded::LinearSolverConfig::default(),
        lambdify_telemetry_mode: BvpLambdifyTelemetryMode::Off,
        aot_telemetry_mode: BvpAotTelemetryMode::Off,
        lambdify_execution_policy: BvpLambdifyExecutionPolicy::default(),
    };

    let state_a = request_a
        .generate()
        .expect("parameterized prebuilt backend should be callable for first param set");
    let state_b = request_b
        .generate()
        .expect("parameterized prebuilt backend should be callable for second param set");

    let y = Col::from_fn(state_a.variable_string.len(), |index| {
        0.2 + index as f64 * 0.01
    });
    let residual_a = state_a.fun.call(0.0, &y).to_DVectorType();
    let residual_b = state_b.fun.call(0.0, &y).to_DVectorType();
    let max_diff = residual_a
        .iter()
        .zip(residual_b.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff > 1e-8,
        "changing only param_values should change residuals without rebuild"
    );

    unregister_linked_sparse_backend(&problem_key);
}
