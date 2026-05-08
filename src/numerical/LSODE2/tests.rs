use super::{
    Lsode2BackendConfig, Lsode2ControllerConfig, Lsode2DstodaState, Lsode2ErrorControlConfig,
    Lsode2ErrorController, Lsode2IterationMode, Lsode2JacobianBackend,
    Lsode2LinearSolverBackend, Lsode2LinearSolverChoice, Lsode2LinearSolverPolicy,
    Lsode2LinearSystemStructure, Lsode2MethodFamily, Lsode2NativeExecutionConfig,
    Lsode2NativeIntegrationLimits, Lsode2NativeStatistics, Lsode2ProblemConfig,
    Lsode2ResidualJacobianSource, Lsode2RetryAction, Lsode2RuntimeState, Lsode2Solver,
    Lsode2StepControlConfig, Lsode2StepCycle, Lsode2SwitchReason, Lsode2SwitchTelemetry,
    Lsode2SymbolicAssemblyBackend, Lsode2SymbolicExecutionMode, Lsode2Tolerance,
};
use crate::symbolic::codegen::codegen_aot_runtime_link::{
    LinkedDenseAotBackend, LinkedResidualAotBackend, LinkedSparseAotBackend,
    register_linked_dense_backend, register_linked_residual_backend,
    register_linked_sparse_backend, unregister_linked_dense_backend,
    unregister_linked_residual_backend, unregister_linked_sparse_backend,
};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp::{
    IvpSymbolicAssemblyBackend, SymbolicIvpProblemOptions, prepare_symbolic_ivp_problem,
    prepare_symbolic_ivp_residual_problem,
};
use crate::symbolic::symbolic_ivp_generated::{
    SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
    prepare_generated_symbolic_ivp_sparse_backend,
};
use nalgebra::{DMatrix, DVector};
use std::any::Any;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

fn exponential_decay_config() -> Lsode2ProblemConfig {
    Lsode2ProblemConfig::new(
        vec![Expr::parse_expression("-y")],
        vec!["y".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0]),
        1.0,
        0.02,
        1e-6,
        1e-8,
    )
}

// --- Adams parity micro-tests (C. Adams numeric path) ---

#[test]
fn parity_adams_correction_divergence_detection_micro() {
    // Use the correction controller directly to assert DSTODA-style divergence
    let controller = crate::numerical::LSODE2::Lsode2CorrectionController::scalar(
        1.0e-3,
        1.0e-6,
        crate::numerical::LSODE2::Lsode2CorrectionControlConfig::default(),
    )
    .expect("build correction controller");

    let diverged = controller
        .assess_iteration(2, &[1.0], &[1.0e-2], &[1.0e-2], Some(1.0e-4), Some(0.5), 2)
        .expect("assessment should run");

    assert_eq!(
        diverged.status,
        crate::numerical::LSODE2::Lsode2CorrectionStatus::Diverged,
        "expected DSTODA-style divergence (DEL > 2*DELP) to be detected"
    );
}

#[test]
fn parity_adams_pdlast_and_sm1_limits_rh() {
    use crate::numerical::LSODE2::{
        Lsode2ErrorControlConfig, Lsode2ErrorController, Lsode2RuntimeState,
        Lsode2StepControlConfig, Lsode2StepCycle,  Lsode2Tolerance,
        Lsode2CorrectionAssessment, Lsode2CorrectionStatus,
    };
    use super::step_cycle::{
    Lsode2StepMethod,
};
    // Build an Adams-like step cycle and exercise PDEST/PDLAST behaviour
    let state = Lsode2RuntimeState::new(0.0, &[1.0], 0.1, 3, Lsode2StepControlConfig::default())
        .expect("runtime state");
    let error_control = Lsode2ErrorController::new(
        Lsode2Tolerance::scalar(1.0e-3, 1.0e-6),
        Lsode2ErrorControlConfig::default(),
    )
    .expect("error controller");

    let mut cycle = Lsode2StepCycle::new_with_method(state, error_control, Lsode2StepMethod::AdamsLike);

    cycle.record_adams_lipschitz_estimate_from_assessment(&Lsode2CorrectionAssessment {
        order: 1,
        iteration: 2,
        weighted_norm: 0.0,
        accumulated_weighted_norm: 0.0,
        previous_weighted_norm: None,
        previous_rate_max: None,
        convergence_ratio: None,
        convergence_rate_estimate: None,
        rate_max_estimate: None,
        pdest_candidate: Some(5.0),
        convergence_measure: 0.0,
        status: Lsode2CorrectionStatus::Converged,
        local_error: vec![0.0],
        needs_jacobian_refresh: false,
    });

    // After recording, PDEST and PDLAST should reflect the estimate
    assert_eq!(cycle.adams_pdest(), 5.0);
    assert_eq!(cycle.adams_pdlast(), 5.0);

    // Now run a post-accept order selection which should clear PDEST but keep PDLAST
    let _ = cycle.select_post_accept_order(&[0.905], 1.0, None).expect("select order");
    assert_eq!(cycle.adams_pdest(), 0.0, "PDEST should be cleared after RH selection");
    assert_eq!(cycle.adams_pdlast(), 5.0, "PDLAST should retain last nonzero estimate");
}

fn parameterized_decay_config() -> Lsode2ProblemConfig {
    Lsode2ProblemConfig::new(
        vec![Expr::parse_expression("a*y")],
        vec!["y".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0]),
        1.0,
        0.02,
        1e-6,
        1e-8,
    )
    .with_equation_parameters(vec!["a".to_string()])
    .with_equation_parameter_values(DVector::from_vec(vec![-1.0]))
}

fn stiff_relaxation_config() -> Lsode2ProblemConfig {
    Lsode2ProblemConfig::new(
        vec![Expr::parse_expression("-10000*(y-cos(t))-sin(t)")],
        vec!["y".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0]),
        1.0,
        0.25,
        1e-6,
        1e-8,
    )
    .with_first_step(Some(0.25))
}

fn robertson_stiff_native_sparse_config() -> Lsode2ProblemConfig {
    Lsode2ProblemConfig::new(
        vec![
            Expr::parse_expression("-0.04*y1 + 1e4*y2*y3"),
            Expr::parse_expression("0.04*y1 - 1e4*y2*y3 - 3e7*y2*y2"),
            Expr::parse_expression("3e7*y2*y2"),
        ],
        vec!["y1".to_string(), "y2".to_string(), "y3".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0, 0.0, 0.0]),
        4.0e3,
        1.0e6,
        1.0e-4,
        1.0e-8,
    )
    .with_native_sparse_faer_backend()
    .with_analytical_callbacks(
        |_t, y: &DVector<f64>| {
            let y1 = y[0];
            let y2 = y[1];
            let y3 = y[2];
            DVector::from_vec(vec![
                -0.04 * y1 + 1.0e4 * y2 * y3,
                0.04 * y1 - 1.0e4 * y2 * y3 - 3.0e7 * y2 * y2,
                3.0e7 * y2 * y2,
            ])
        },
        |_t, y: &DVector<f64>| {
            let y2 = y[1];
            let y3 = y[2];
            DMatrix::from_row_slice(
                3,
                3,
                &[
                    -0.04,
                    1.0e4 * y3,
                    1.0e4 * y2,
                    0.04,
                    -1.0e4 * y3 - 6.0e7 * y2,
                    -1.0e4 * y2,
                    0.0,
                    6.0e7 * y2,
                    0.0,
                ],
            )
        },
    )
    .with_native_execution(Lsode2NativeExecutionConfig::native_solve(200_000, 200_000))
}

fn nonsteady_chemical_kinetics_native_sparse_config() -> Lsode2ProblemConfig {
    // Consecutive irreversible reactions:
    // A -> B -> C with disparate time scales.
    let k1 = 1.0e4_f64;
    let k2 = 1.0e-2_f64;
    Lsode2ProblemConfig::new(
        vec![
            Expr::parse_expression("-1e4*A"),
            Expr::parse_expression("1e4*A-1e-2*B"),
            Expr::parse_expression("1e-2*B"),
        ],
        vec!["A".to_string(), "B".to_string(), "C".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0, 0.0, 0.0]),
        10.0,
        1.0e6,
        1.0e-6,
        1.0e-10,
    )
    .with_native_sparse_faer_backend()
    .with_analytical_callbacks(
        move |_t, y: &DVector<f64>| {
            let a = y[0];
            let b = y[1];
            DVector::from_vec(vec![-k1 * a, k1 * a - k2 * b, k2 * b])
        },
        move |_t, _y: &DVector<f64>| {
            DMatrix::from_row_slice(
                3,
                3,
                &[
                    -k1, 0.0, 0.0, //
                    k1, -k2, 0.0, //
                    0.0, k2, 0.0,
                ],
            )
        },
    )
    .with_first_step(Some(1.0e-6))
    .with_native_execution(Lsode2NativeExecutionConfig::native_solve(200_000, 200_000))
}

fn make_bdf_cycle_for_parity(step_config: Lsode2StepControlConfig) -> Lsode2StepCycle {
    let state = Lsode2RuntimeState::new(0.0, &[1.0], 1.0, 3, step_config)
        .expect("LSODE2 runtime state should initialize for parity test");
    let error_control = Lsode2ErrorController::new(
        Lsode2Tolerance::scalar(1.0e-3, 1.0e-6),
        Lsode2ErrorControlConfig::default(),
    )
    .expect("LSODE2 error-control config should be valid for parity test");
    Lsode2StepCycle::new(state, error_control)
}

fn unique_aot_diag_output_dir(prefix: &str) -> PathBuf {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_nanos();
    PathBuf::from(format!(
        "target/lsode2-tests/aot-diag/{prefix}/pid{}_{}",
        std::process::id(),
        now
    ))
}

fn unique_test_tag(prefix: &str) -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_nanos();
    format!("{prefix}_pid{}_{}", std::process::id(), now)
}

struct DenseBackendUnregisterGuard {
    key: String,
}

impl DenseBackendUnregisterGuard {
    fn new(key: String) -> Self {
        Self { key }
    }
}

impl Drop for DenseBackendUnregisterGuard {
    fn drop(&mut self) {
        unregister_linked_dense_backend(self.key.as_str());
    }
}

struct SparseBackendUnregisterGuard {
    key: String,
}

impl SparseBackendUnregisterGuard {
    fn new(key: String) -> Self {
        Self { key }
    }
}

impl Drop for SparseBackendUnregisterGuard {
    fn drop(&mut self) {
        unregister_linked_sparse_backend(self.key.as_str());
    }
}

struct ResidualBackendUnregisterGuard {
    key: String,
}

impl ResidualBackendUnregisterGuard {
    fn new(key: String) -> Self {
        Self { key }
    }
}

impl Drop for ResidualBackendUnregisterGuard {
    fn drop(&mut self) {
        unregister_linked_residual_backend(self.key.as_str());
    }
}

fn sparse_tcc_symbolic_aot_config(
    output_parent_dir: PathBuf,
    artifact_tag: &str,
) -> Lsode2ProblemConfig {
    let generated_backend =
        SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
            .with_c_tcc()
            .with_crate_name_override(Some(format!(
                "generated_lsode2_diag_sparse_tcc_{artifact_tag}"
            )))
            .with_module_name_override(Some(format!(
                "generated_lsode2_diag_sparse_tcc_{artifact_tag}"
            )));

    let mut config = exponential_decay_config()
        .with_linear_system_structure(Lsode2LinearSystemStructure::Sparse)
        .with_linear_solver_policy(Lsode2LinearSolverPolicy::Force(
            Lsode2LinearSolverChoice::FaerSparseLu,
        ))
        .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
            execution: Lsode2SymbolicExecutionMode::Aot {
                toolchain: super::Lsode2AotToolchain::CTcc,
                profile: super::Lsode2AotProfile::Release,
            },
        });
    config.backend.generated_backend = generated_backend;
    config
}

fn panic_payload_to_string(payload: Box<dyn Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<String>() {
        return s.clone();
    }
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        return (*s).to_string();
    }
    "panic payload is not a string".to_string()
}

fn classify_aot_tcc_diag_error(message: &str) -> &'static str {
    if message.contains("Permission denied")
        || message.contains("could not write")
        || message.contains("access is denied")
    {
        "permission_denied_or_file_lock"
    } else if message.contains("GeneratedBackendFailure") {
        "generated_backend_failure"
    } else if message.contains("is not recognized as an internal or external command")
        || message.contains("command not found")
        || message.contains("No such file or directory")
    {
        "toolchain_not_available"
    } else {
        "other"
    }
}

#[derive(Debug, Clone)]
struct AotTccDiagOutcome {
    label: String,
    status_kind: String,
    error_class: String,
    detail: String,
}

fn run_sparse_tcc_aot_diag_case(label: &str, config: Lsode2ProblemConfig) -> AotTccDiagOutcome {
    let result = catch_unwind_quiet(|| {
        let mut solver = Lsode2Solver::new(config).map_err(|e| format!("new_error({e})"))?;
        solver
            .prepare()
            .map_err(|e| format!("prepare_error({e})"))?;
        solver.solve().map_err(|e| format!("solve_error({e})"))?;
        Ok::<(), String>(())
    });

    match result {
        Ok(Ok(())) => AotTccDiagOutcome {
            label: label.to_string(),
            status_kind: "finished".to_string(),
            error_class: "-".to_string(),
            detail: "-".to_string(),
        },
        Ok(Err(message)) => AotTccDiagOutcome {
            label: label.to_string(),
            status_kind: if message.starts_with("new_error(") {
                "new_error".to_string()
            } else if message.starts_with("prepare_error(") {
                "prepare_error".to_string()
            } else {
                "solve_error".to_string()
            },
            error_class: classify_aot_tcc_diag_error(&message).to_string(),
            detail: message,
        },
        Err(payload) => {
            let message = panic_payload_to_string(payload);
            AotTccDiagOutcome {
                label: label.to_string(),
                status_kind: "panic".to_string(),
                error_class: classify_aot_tcc_diag_error(&message).to_string(),
                detail: message,
            }
        }
    }
}

fn catch_unwind_quiet<F, R>(f: F) -> std::thread::Result<R>
where
    F: FnOnce() -> R,
{
    static PANIC_HOOK_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let lock = PANIC_HOOK_LOCK.get_or_init(|| Mutex::new(()));
    let _guard = lock.lock().expect("panic hook lock should not be poisoned");
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    std::panic::set_hook(previous);
    result
}

fn assert_exponential_decay_solve(config: Lsode2ProblemConfig) {
    let mut solver = Lsode2Solver::new(config).expect("dense symbolic LSODE2 config should build");
    solver
        .solve()
        .expect("LSODE2 dense symbolic solve should finish");

    let (t, y) = solver.get_result();
    assert!(
        matches!(
            solver.status(),
            "finished" | "finished_native_faithful" | "finished_native_faithful_partial"
        ),
        "unexpected LSODE2 solve status: {}",
        solver.status()
    );
    assert!(!t.is_empty());
    assert_eq!(y.ncols(), 1);

    let y_final = y[(y.nrows() - 1, 0)];
    let expected = (-1.0_f64).exp();
    assert!(
        (y_final - expected).abs() < 1e-4,
        "exp decay mismatch: got={y_final:e}, expected={expected:e}"
    );
}

fn solve_exponential_decay(config: Lsode2ProblemConfig) -> Lsode2Solver {
    let mut solver = Lsode2Solver::new(config).expect("LSODE2 config should build");
    solver.solve().expect("LSODE2 solve should finish");
    solver
}

fn to_ivp_assembly_backend(assembly: Lsode2SymbolicAssemblyBackend) -> IvpSymbolicAssemblyBackend {
    match assembly {
        Lsode2SymbolicAssemblyBackend::ExprLegacy => IvpSymbolicAssemblyBackend::ExprLegacy,
        Lsode2SymbolicAssemblyBackend::AtomView => IvpSymbolicAssemblyBackend::AtomView,
    }
}

fn run_prelinked_dense_aot_case(assembly: Lsode2SymbolicAssemblyBackend, key_tag: &str) {
    let generated_backend = SymbolicIvpGeneratedBackendConfig::require_prebuilt()
        .with_crate_name_override(Some(format!(
            "generated_lsode2_prelinked_dense_parity_{key_tag}"
        )))
        .with_module_name_override(Some(format!(
            "generated_lsode2_prelinked_dense_parity_{key_tag}"
        )));
    let source = Lsode2ResidualJacobianSource::Symbolic {
        assembly,
        execution: Lsode2SymbolicExecutionMode::Aot {
            toolchain: super::Lsode2AotToolchain::CTcc,
            profile: super::Lsode2AotProfile::Release,
        },
    };
    let mut config = exponential_decay_config()
        .with_residual_jacobian_source(source)
        .with_linear_system_structure(Lsode2LinearSystemStructure::Dense)
        .with_linear_solver_policy(Lsode2LinearSolverPolicy::Auto);
    config.backend.generated_backend = generated_backend.clone();

    let problem = prepare_symbolic_ivp_problem(
        config.eq_system.clone(),
        config.values.clone(),
        config.arg.clone(),
        SymbolicIvpProblemOptions::new()
            .with_aot_options(generated_backend.aot_options)
            .with_symbolic_assembly_backend(to_ivp_assembly_backend(assembly)),
    )
    .expect("dense symbolic IVP problem should prepare for prelinked AOT parity test");
    let problem_key = problem
        .prepare_dense_aot_problem(generated_backend.aot_options)
        .problem_key();

    register_linked_dense_backend(LinkedDenseAotBackend::new(
        problem_key.clone(),
        1,
        (1, 1),
        Arc::new(move |args: &[f64], out: &mut [f64]| {
            assert!(args.len() >= 2, "IVP dense AOT args should be [t, y...]");
            assert_eq!(out.len(), 1);
            out[0] = -args[1];
        }),
        Arc::new(move |_args: &[f64], out: &mut [f64]| {
            assert_eq!(out.len(), 1);
            out[0] = -1.0;
        }),
    ));
    let _dense_guard = DenseBackendUnregisterGuard::new(problem_key.clone());

    let y_final = (|| {
        let mut solver =
            Lsode2Solver::new(config).expect("LSODE2 dense AOT parity config should build");
        solver
            .solve()
            .expect("LSODE2 dense AOT parity solve should finish");
        let (_, y) = solver.get_result();
        y[(y.nrows() - 1, 0)]
    })();
    let expected = (-1.0_f64).exp();
    assert!(
        (y_final - expected).abs() < 1e-4,
        "dense AOT parity mismatch ({key_tag}): got={y_final:e}, expected={expected:e}"
    );
}

fn run_prelinked_sparse_or_banded_aot_case(
    assembly: Lsode2SymbolicAssemblyBackend,
    structure: Lsode2LinearSystemStructure,
    key_tag: &str,
) {
    let generated_backend = SymbolicIvpGeneratedBackendConfig::require_prebuilt()
        .with_crate_name_override(Some(format!(
            "generated_lsode2_prelinked_sparse_parity_{key_tag}"
        )))
        .with_module_name_override(Some(format!(
            "generated_lsode2_prelinked_sparse_parity_{key_tag}"
        )));
    let source = Lsode2ResidualJacobianSource::Symbolic {
        assembly,
        execution: Lsode2SymbolicExecutionMode::Aot {
            toolchain: super::Lsode2AotToolchain::CTcc,
            profile: super::Lsode2AotProfile::Release,
        },
    };
    let mut config = exponential_decay_config()
        .with_residual_jacobian_source(source)
        .with_linear_system_structure(structure)
        .with_linear_solver_policy(Lsode2LinearSolverPolicy::Auto);
    config.backend.generated_backend = generated_backend.clone();

    let mut options = SymbolicIvpProblemOptions::new()
        .with_aot_options(generated_backend.aot_options)
        .with_symbolic_assembly_backend(to_ivp_assembly_backend(assembly));
    if let Some(parameters) = config.equation_parameters.clone() {
        options = options.with_equation_parameters(parameters);
    }
    if let Some(values) = config.equation_parameter_values.clone() {
        options = options.with_equation_parameter_values(values);
    }
    let prepared_sparse = prepare_generated_symbolic_ivp_sparse_backend(
        config.eq_system.clone(),
        config.values.clone(),
        config.arg.clone(),
        options,
        {
            let mut probe_backend = generated_backend.clone();
            probe_backend.build_policy = SymbolicIvpAotBuildPolicy::UseIfAvailable;
            probe_backend
        },
    )
    .expect("sparse generated backend should prepare for LSODE2 AOT parity case");
    let sparse_problem_key = prepared_sparse.problem_key.clone();
    let nnz = prepared_sparse.jacobian_structure.nnz();

    let residual_problem = prepare_symbolic_ivp_residual_problem(
        config.eq_system.clone(),
        config.values.clone(),
        config.arg.clone(),
        SymbolicIvpProblemOptions::new()
            .with_aot_options(generated_backend.aot_options)
            .with_symbolic_assembly_backend(to_ivp_assembly_backend(assembly)),
    )
    .expect("residual-only symbolic IVP problem should prepare for LSODE2 AOT parity case");
    let residual_problem_key = residual_problem
        .prepare_residual_aot_problem(generated_backend.aot_options)
        .problem_key();

    let jacobian_calls = Arc::new(AtomicUsize::new(0));
    let jacobian_calls_for_backend = Arc::clone(&jacobian_calls);
    register_linked_residual_backend(LinkedResidualAotBackend::new(
        residual_problem_key.clone(),
        1,
        Arc::new(move |args: &[f64], out: &mut [f64]| {
            assert!(args.len() >= 2, "IVP residual AOT args should be [t, y...]");
            assert_eq!(out.len(), 1);
            out[0] = -args[1];
        }),
    ));
    register_linked_sparse_backend(LinkedSparseAotBackend::new(
        sparse_problem_key.clone(),
        1,
        (1, 1),
        nnz,
        Arc::new(move |args: &[f64], out: &mut [f64]| {
            assert!(args.len() >= 2, "IVP sparse AOT args should be [t, y...]");
            assert_eq!(out.len(), 1);
            out[0] = -args[1];
        }),
        Arc::new(move |_args: &[f64], out: &mut [f64]| {
            jacobian_calls_for_backend.fetch_add(1, Ordering::Relaxed);
            assert_eq!(out.len(), nnz);
            out.fill(-1.0);
        }),
    ));
    let _sparse_guard = SparseBackendUnregisterGuard::new(sparse_problem_key.clone());
    let _residual_guard = ResidualBackendUnregisterGuard::new(residual_problem_key.clone());

    let y_final = (|| {
        let mut solver = Lsode2Solver::new(config).expect("LSODE2 AOT parity config should build");
        solver
            .solve()
            .expect("LSODE2 AOT parity solve should finish");
        let (_, y) = solver.get_result();
        y[(y.nrows() - 1, 0)]
    })();

    let expected = (-1.0_f64).exp();
    assert!(
        (y_final - expected).abs() < 1e-4,
        "AOT parity mismatch ({key_tag}): got={y_final:e}, expected={expected:e}"
    );
    assert!(
        jacobian_calls.load(Ordering::Relaxed) > 0,
        "AOT parity case should call linked sparse Jacobian values ({key_tag})"
    );
}

#[test]
fn lsode2_default_controller_is_faithful_lsode_fixed_bdf() {
    let config = exponential_decay_config();
    assert_eq!(config.controller.mode, super::Lsode2ControllerMode::BdfOnly);

    let auto = config.clone().with_automatic_adams_bdf_controller();
    assert_eq!(
        auto.controller.mode,
        super::Lsode2ControllerMode::AutomaticAdamsBdf
    );
}

#[test]
fn lsode2_dense_symbolic_bdf_solves_exponential_decay() {
    assert_exponential_decay_solve(exponential_decay_config());
}

#[test]
fn lsode2_sparse_faer_bdf_solves_exponential_decay() {
    assert_exponential_decay_solve(exponential_decay_config().with_native_sparse_faer_backend());
}

#[test]
fn lsode2_banded_faithful_bdf_solves_exponential_decay() {
    assert_exponential_decay_solve(
        exponential_decay_config().with_native_banded_faithful_backend(),
    );
}

#[test]
fn lsode2_dense_aot_config_surface_builds_solver_without_native_backend() {
    let config = exponential_decay_config().with_dense_aot_c_tcc("target/lsode2-tests");
    let solver = Lsode2Solver::new(config).expect("dense AOT LSODE2 config should build");

    assert_eq!(
        solver.config().backend.linear_solver_backend,
        Lsode2LinearSolverBackend::Dense
    );
    assert_eq!(
        solver
            .config()
            .backend
            .generated_backend
            .aot_c_compiler
            .as_deref(),
        Some("tcc")
    );
}

#[test]
fn lsode2_resolved_plan_auto_selects_solver_from_structure() {
    let dense_plan = exponential_decay_config()
        .with_linear_system_structure(Lsode2LinearSystemStructure::Dense)
        .with_linear_solver_policy(Lsode2LinearSolverPolicy::Auto)
        .resolve_plan();
    assert_eq!(dense_plan.linear_solver, Lsode2LinearSolverChoice::DenseLu);
    assert_eq!(
        dense_plan.linear_solver_reason,
        "auto_from_linear_structure_dense"
    );

    let sparse_plan = exponential_decay_config()
        .with_linear_system_structure(Lsode2LinearSystemStructure::Sparse)
        .with_linear_solver_policy(Lsode2LinearSolverPolicy::Auto)
        .resolve_plan();
    assert_eq!(
        sparse_plan.linear_solver,
        Lsode2LinearSolverChoice::FaerSparseLu
    );
    assert_eq!(
        sparse_plan.linear_solver_reason,
        "auto_from_linear_structure_sparse"
    );

    let banded_plan = exponential_decay_config()
        .with_linear_system_structure(Lsode2LinearSystemStructure::Banded { kl: 2, ku: 2 })
        .with_linear_solver_policy(Lsode2LinearSolverPolicy::Auto)
        .resolve_plan();
    assert_eq!(
        banded_plan.linear_solver,
        Lsode2LinearSolverChoice::LapackFaithfulBandedLu
    );
    assert_eq!(
        banded_plan.linear_solver_reason,
        "auto_from_linear_structure_banded"
    );
}

#[test]
fn lsode2_resolved_plan_force_policy_overrides_structure() {
    let config = exponential_decay_config()
        .with_linear_system_structure(Lsode2LinearSystemStructure::Dense)
        .with_linear_solver_policy(Lsode2LinearSolverPolicy::Force(
            Lsode2LinearSolverChoice::FaerSparseLu,
        ));
    let plan = config.resolve_plan();

    assert_eq!(plan.linear_solver, Lsode2LinearSolverChoice::FaerSparseLu);
    assert_eq!(plan.linear_solver_reason, "forced_by_linear_solver_policy");
    assert_eq!(
        config.backend.linear_solver_backend,
        Lsode2LinearSolverBackend::SparseFaer
    );
}

#[test]
fn lsode2_resolved_plan_analytical_sparse_forced_faer_is_visible() {
    let plan = exponential_decay_config()
        .with_native_sparse_faer_backend()
        .with_analytical_callbacks(
            |_t, y: &DVector<f64>| DVector::from_vec(vec![-y[0]]),
            |_t, _y: &DVector<f64>| DMatrix::from_row_slice(1, 1, &[-1.0]),
        )
        .with_faithful_bdf_solve(4096, 4096)
        .resolve_plan();

    assert_eq!(plan.source, Lsode2ResidualJacobianSource::Analytical);
    assert_eq!(plan.structure, Lsode2LinearSystemStructure::Sparse);
    assert_eq!(plan.linear_solver, Lsode2LinearSolverChoice::FaerSparseLu);
    assert_eq!(plan.linear_solver_reason, "forced_by_linear_solver_policy");
}

#[test]
fn lsode2_resolved_plan_analytical_banded_forced_faithful_is_visible() {
    let plan = exponential_decay_config()
        .with_native_banded_faithful_backend()
        .with_analytical_callbacks(
            |_t, y: &DVector<f64>| DVector::from_vec(vec![-y[0]]),
            |_t, _y: &DVector<f64>| DMatrix::from_row_slice(1, 1, &[-1.0]),
        )
        .with_faithful_bdf_solve(4096, 4096)
        .resolve_plan();

    assert_eq!(plan.source, Lsode2ResidualJacobianSource::Analytical);
    assert_eq!(
        plan.structure,
        Lsode2LinearSystemStructure::Banded { kl: 0, ku: 0 }
    );
    assert_eq!(
        plan.linear_solver,
        Lsode2LinearSolverChoice::LapackFaithfulBandedLu
    );
    assert_eq!(plan.linear_solver_reason, "forced_by_linear_solver_policy");
}

#[test]
fn lsode2_can_set_symbolic_atom_view_source_in_config_surface() {
    let config = exponential_decay_config().with_residual_jacobian_source(
        Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::AtomView,
            execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
        },
    );
    let plan = config.resolve_plan();
    assert_eq!(
        plan.source,
        Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::AtomView,
            execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
        }
    );
    assert_eq!(
        config.backend.jacobian_backend,
        Lsode2JacobianBackend::SymbolicGenerated
    );
}

#[test]
fn lsode2_with_backend_keeps_existing_symbolic_assembly_backend() {
    let config = exponential_decay_config()
        .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::AtomView,
            execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
        })
        .with_backend(Lsode2BackendConfig::native_sparse_faer());

    assert_eq!(
        config.residual_jacobian_source,
        Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::AtomView,
            execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
        }
    );
}

#[test]
fn lsode2_atom_view_lambdify_solves_for_dense_sparse_and_banded_structures() {
    let source = Lsode2ResidualJacobianSource::Symbolic {
        assembly: Lsode2SymbolicAssemblyBackend::AtomView,
        execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
    };

    let dense = exponential_decay_config()
        .with_residual_jacobian_source(source)
        .with_linear_system_structure(Lsode2LinearSystemStructure::Dense)
        .with_linear_solver_policy(Lsode2LinearSolverPolicy::Auto);
    assert_exponential_decay_solve(dense);

    let sparse = exponential_decay_config()
        .with_residual_jacobian_source(source)
        .with_linear_system_structure(Lsode2LinearSystemStructure::Sparse)
        .with_linear_solver_policy(Lsode2LinearSolverPolicy::Auto);
    assert_exponential_decay_solve(sparse);

    let banded = exponential_decay_config()
        .with_residual_jacobian_source(source)
        .with_linear_system_structure(Lsode2LinearSystemStructure::Banded { kl: 1, ku: 1 })
        .with_linear_solver_policy(Lsode2LinearSolverPolicy::Auto);
    assert_exponential_decay_solve(banded);
}

#[test]
fn lsode2_symbolic_aot_policy_maps_to_generated_backend_settings() {
    let source = Lsode2ResidualJacobianSource::Symbolic {
        assembly: Lsode2SymbolicAssemblyBackend::AtomView,
        execution: Lsode2SymbolicExecutionMode::Aot {
            toolchain: super::Lsode2AotToolchain::CTcc,
            profile: super::Lsode2AotProfile::Debug,
        },
    };
    let config = exponential_decay_config()
        .with_residual_jacobian_source(source)
        .with_linear_system_structure(Lsode2LinearSystemStructure::Sparse)
        .with_linear_solver_policy(Lsode2LinearSolverPolicy::Auto);

    assert_eq!(
        config.backend.generated_backend.aot_codegen_backend,
        crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend::C
    );
    assert_eq!(
        config.backend.generated_backend.aot_c_compiler.as_deref(),
        Some("tcc")
    );
    assert_eq!(
        config.backend.generated_backend.build_policy,
        SymbolicIvpAotBuildPolicy::BuildIfMissing {
            profile:
                crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile::Debug
        }
    );
    assert_eq!(
        config.residual_jacobian_source,
        Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::AtomView,
            execution: Lsode2SymbolicExecutionMode::Aot {
                toolchain: super::Lsode2AotToolchain::CTcc,
                profile: super::Lsode2AotProfile::Debug,
            },
        }
    );
}

#[test]
fn lsode2_sparse_native_path_records_residual_jacobian_and_lu_stats() {
    let solver = solve_exponential_decay(
        exponential_decay_config().with_backend(
            Lsode2BackendConfig::default()
                .with_linear_solver_backend(Lsode2LinearSolverBackend::SparseFaer),
        ),
    );
    let stats = solver.native_statistics();

    assert!(
        stats.native_residual_calls > 0,
        "native sparse path should call residual"
    );
    assert!(
        stats.native_jacobian_calls > 0,
        "native sparse path should record native Jacobian evaluations"
    );
    assert!(
        stats.native_jacobian_ms_total.is_finite(),
        "native sparse Jacobian timing should be finite"
    );
    assert!(
        stats.native_linear_solve_calls > 0,
        "native sparse path should factor Newton systems"
    );
}

#[test]
fn lsode2_banded_native_path_records_residual_jacobian_and_lu_stats() {
    let solver = solve_exponential_decay(
        exponential_decay_config().with_backend(
            Lsode2BackendConfig::default()
                .with_linear_solver_backend(Lsode2LinearSolverBackend::BandedFaithful),
        ),
    );
    let stats = solver.native_statistics();

    assert!(
        stats.native_residual_calls > 0,
        "native banded path should call residual"
    );
    assert!(
        stats.native_jacobian_calls > 0,
        "native banded path should record native Jacobian evaluations"
    );
    assert!(
        stats.native_jacobian_ms_total.is_finite(),
        "native banded Jacobian timing should be finite"
    );
    assert!(
        stats.native_linear_solve_calls > 0,
        "native banded path should factor Newton systems"
    );
}

#[test]
fn lsode2_prepare_is_idempotent_and_separate_from_solve() {
    let mut solver =
        Lsode2Solver::new(exponential_decay_config().with_native_banded_faithful_backend())
            .expect("LSODE2 native banded config should build");

    assert!(!solver.is_prepared());
    solver.prepare().expect("LSODE2 prepare should succeed");
    assert!(solver.is_prepared());

    let stats_after_first_prepare = solver.statistics();
    assert_eq!(stats_after_first_prepare.backend_prepare_calls, 1);
    assert_eq!(stats_after_first_prepare.solve_calls, 0);
    assert_eq!(stats_after_first_prepare.bdf_nlu_total, 0);

    solver
        .prepare()
        .expect("second LSODE2 prepare should be a no-op");
    assert_eq!(solver.statistics().backend_prepare_calls, 1);

    solver.solve().expect("prepared LSODE2 solve should finish");
    assert!(
        matches!(
            solver.status(),
            "finished_native_faithful" | "finished_native_faithful_partial"
        ),
        "unexpected status after native faithful solve: {}",
        solver.status()
    );
    assert_eq!(solver.statistics().backend_prepare_calls, 1);
}

#[test]
fn lsode2_native_statistics_track_prepare_solve_and_controller_decision() {
    let mut solver =
        Lsode2Solver::new(exponential_decay_config().with_native_banded_faithful_backend())
            .expect("LSODE2 native banded config should build");

    solver.prepare().expect("LSODE2 prepare should succeed");
    solver.solve().expect("LSODE2 solve should finish");

    let native = solver.native_statistics();
    assert_eq!(native.backend_prepare_calls, 1);
    assert_eq!(native.solve_calls, 1);
    assert_eq!(native.algorithm_decision_calls, 1);
    assert_eq!(native.preferred_bdf_count, 1);
    assert_eq!(native.executed_bdf_count, 1);
    assert_eq!(native.bridge_prepare_calls, 1);
    assert_eq!(native.bridge_step_calls, 0);
    assert_eq!(native.bridge_bdf_nlu_total, 0);
    assert!(native.native_step_attempts > 0);
    assert!(native.native_residual_calls > 0);
    assert!(native.native_jacobian_calls > 0);
    assert!(native.native_linear_solve_calls > 0);
    let probe = solver
        .native_step_probe()
        .expect("native banded backend should expose a native step probe");
    assert!(probe.iterations > 0);
    assert!(probe.attempted_steps > 0);
    assert!(probe.accepted_steps > 0);
    assert!(
        probe.accepted_steps > 0 || probe.rejected_steps > 0,
        "native preflight should end in either accepted or rejected step attempts"
    );
    assert!(probe.h_trial > 0.0);
    assert!(probe.t_trial > solver.config().t0);
    assert!(probe.final_t >= solver.config().t0);
}

#[test]
fn lsode2_solve_with_summary_reports_final_state_and_statistics() {
    let mut solver =
        Lsode2Solver::new(exponential_decay_config().with_native_sparse_faer_backend())
            .expect("LSODE2 native sparse config should build");

    let summary = solver
        .solve_with_summary()
        .expect("LSODE2 solve with summary should finish");

    assert_eq!(summary.method, "bdf");
    assert_eq!(summary.jacobian_backend, "symbolic_generated");
    assert_eq!(summary.linear_solver_backend, "faer_sparse_lu");
    assert!(
        summary.status == "finished_native_faithful"
            || summary.status == "finished_native_faithful_partial",
        "unexpected summary status: {}",
        summary.status
    );
    assert!(summary.time_points > 0);
    assert_eq!(summary.variable_count, 1);
    let final_t = summary.final_t.expect("final t should be available");
    assert!(final_t.is_finite(), "final t should be finite");
    assert!(
        (final_t - 1.0).abs() <= 0.02,
        "final t should land within one configured max_step of t_bound, got={final_t:e}"
    );

    let final_y = summary.final_y.expect("final y should be available");
    let expected = (-1.0_f64).exp();
    assert!(
        (final_y[0] - expected).abs() < 1e-4,
        "summary final y mismatch: got={:e}, expected={:e}",
        final_y[0],
        expected
    );
    assert!(summary.max_abs_solution >= final_y[0].abs());
    assert_eq!(summary.algorithm.controller_mode, "bdf_only");
    assert_eq!(summary.algorithm.active_family, "bdf");
    assert_eq!(summary.algorithm.preferred_family, "bdf");
    assert_eq!(summary.algorithm.executed_family, Some("bdf"));
    assert_eq!(summary.algorithm.switch_reason, "fixed_controller");
    assert!(!summary.algorithm.switch_uses_fallback);
    assert!(!summary.algorithm.method_switching_enabled);
    assert!(summary.algorithm.bdf_current_order.is_some());
    assert_eq!(summary.algorithm.bdf_max_order_cap, Some(5));
    assert!(summary.algorithm.bdf_equal_step_count.is_some());
    assert_eq!(summary.statistics.backend_prepare_calls, 0);
    assert_eq!(summary.statistics.solve_calls, 0);
    assert_eq!(summary.statistics.step_calls, 0);
    assert_eq!(summary.native_statistics.backend_prepare_calls, 0);
    assert_eq!(summary.native_statistics.solve_calls, 1);
    assert_eq!(summary.native_statistics.preferred_bdf_count, 1);
    assert_eq!(summary.native_statistics.executed_bdf_count, 1);
    assert_eq!(summary.native_statistics.bridge_prepare_calls, 0);
    assert_eq!(summary.native_statistics.bridge_bdf_nlu_total, 0);
    let probe = summary
        .native_step_probe
        .as_ref()
        .expect("native sparse summary should carry a step probe");
    assert!(probe.iterations > 0);
    assert!(probe.attempted_steps > 0);
    assert!(probe.accepted_steps > 0);
    assert!(
        probe.accepted_steps > 0 || probe.rejected_steps > 0,
        "native preflight should end in either accepted or rejected step attempts"
    );
    assert!(probe.h_trial > 0.0);
    assert!(probe.t_trial > 0.0);
    assert!(probe.final_t >= 0.0);
    assert!(summary.native_integration_preview.is_none());
    assert!(summary.native_integration_solve.is_some());
}

#[test]
fn lsode2_dense_backend_skips_native_step_probe() {
    let mut solver =
        Lsode2Solver::new(exponential_decay_config()).expect("LSODE2 dense config should build");

    solver.solve().expect("LSODE2 dense solve should finish");

    assert!(
        solver.native_step_probe().is_none(),
        "dense bridge path should not claim a native sparse/banded probe"
    );
    let native = solver.native_statistics();
    assert_eq!(native.native_step_attempts, 0);
    assert_eq!(native.native_residual_calls, 0);
    assert_eq!(native.native_jacobian_calls, 0);
    assert_eq!(native.native_linear_solve_calls, 0);
}

#[test]
fn lsode2_sparse_native_step_probe_records_solver_level_probe() {
    let mut solver =
        Lsode2Solver::new(exponential_decay_config().with_native_sparse_faer_backend())
            .expect("LSODE2 native sparse config should build");

    solver.solve().expect("LSODE2 sparse solve should finish");

    let probe = solver
        .native_step_probe()
        .expect("native sparse backend should expose a step probe");
    assert!(probe.iterations > 0);
    assert!(probe.attempted_steps > 0);
    assert!(probe.accepted_steps > 0);
    assert!(
        probe.accepted_steps > 0 || probe.rejected_steps > 0,
        "native preflight should end in either accepted or rejected step attempts"
    );
    assert!(probe.h_trial > 0.0);
    assert!(probe.t_trial > solver.config().t0);
    assert!(probe.final_t >= solver.config().t0);

    let native = solver.native_statistics();
    assert!(native.native_step_attempts > 0);
    assert!(native.native_residual_calls > 0);
    assert!(native.native_jacobian_calls > 0);
    assert!(native.native_linear_solve_calls > 0);
}

#[test]
fn lsode2_dense_native_integration_preview_skips_cleanly() {
    let mut solver =
        Lsode2Solver::new(exponential_decay_config()).expect("LSODE2 dense config should build");

    let summary = solver
        .run_native_integration_preview(Lsode2NativeIntegrationLimits::new(4, 2))
        .expect("dense native preview should not error");

    assert!(summary.is_none());
    let native = solver.native_statistics();
    assert_eq!(native.algorithm_decision_calls, 1);
    assert_eq!(native.native_step_attempts, 0);
    assert_eq!(native.native_residual_calls, 0);
    assert_eq!(native.native_jacobian_calls, 0);
    assert_eq!(native.native_linear_solve_calls, 0);
}

#[test]
fn lsode2_sparse_native_integration_preview_returns_solver_level_summary() {
    let mut solver =
        Lsode2Solver::new(exponential_decay_config().with_native_sparse_faer_backend())
            .expect("LSODE2 native sparse config should build");

    let summary = solver
        .run_native_integration_preview(Lsode2NativeIntegrationLimits::new(4, 2))
        .expect("native sparse preview should run")
        .expect("native sparse preview should return a summary");

    assert!(summary.attempted_steps > 0);
    assert_eq!(
        summary.attempted_steps,
        summary.accepted_steps + summary.rejected_steps
    );
    assert!(summary.total_iterations > 0);
    assert!(summary.first_report.predicted.t_trial > solver.config().t0);
    assert!(summary.final_t >= solver.config().t0);
    assert_eq!(summary.final_y.len(), 1);

    let native = solver.native_statistics();
    assert_eq!(native.algorithm_decision_calls, 1);
    assert!(native.native_step_attempts > 0);
    assert!(native.native_residual_calls > 0);
    assert!(native.native_jacobian_calls > 0);
    assert!(native.native_linear_solve_calls > 0);
}

#[test]
fn lsode2_sparse_native_integration_preview_can_run_explicit_adams_order1_family() {
    let mut solver =
        Lsode2Solver::new(exponential_decay_config().with_native_sparse_faer_backend())
            .expect("LSODE2 native sparse config should build");

    let summary = solver
        .run_native_integration_preview_for_family(
            Lsode2NativeIntegrationLimits::new(5, 3),
            Lsode2MethodFamily::Adams,
        )
        .expect("native sparse Adams-order1 preview should run")
        .expect("native sparse Adams-order1 preview should return a summary");

    assert!(summary.attempted_steps > 0);
    assert!(summary.accepted_steps > 0);
    assert_eq!(summary.first_report.predicted.order, 1);
    assert!(summary.last_report.predicted.order >= 1);
}

#[test]
fn lsode2_native_preview_auto_method_can_switch_to_adams_order1_after_probe_warmup() {
    let mut solver = Lsode2Solver::new(
        exponential_decay_config()
            .with_native_sparse_faer_backend()
            .with_controller(
                Lsode2ControllerConfig::automatic_adams_bdf().with_method_switch_probe_steps(1),
            ),
    )
    .expect("LSODE2 automatic-controller sparse config should build");

    let first = solver
        .run_native_integration_preview(Lsode2NativeIntegrationLimits::new(32, 16))
        .expect("first native preview should run")
        .expect("first native preview should return a summary");
    assert!(
        first.accepted_steps >= 2,
        "probe warmup for method_switch_probe_steps=1 needs at least two accepted steps"
    );

    let second = solver
        .run_native_integration_preview(Lsode2NativeIntegrationLimits::new(6, 3))
        .expect("second native preview should run")
        .expect("second native preview should return a summary");
    assert_eq!(second.first_report.predicted.order, 1);
    assert!(second.last_report.predicted.order >= 1);
}

#[test]
fn lsode2_solve_can_use_configured_native_preview_before_bridge() {
    let mut solver = Lsode2Solver::new(
        exponential_decay_config()
            .with_native_sparse_faer_backend()
            .with_native_execution(Lsode2NativeExecutionConfig::preview_before_bridge(4, 2)),
    )
    .expect("LSODE2 native-preview config should build");

    let summary = solver
        .solve_with_summary()
        .expect("LSODE2 solve with configured native preview should finish");

    assert_eq!(summary.status, "finished");
    let preview = summary
        .native_integration_preview
        .as_ref()
        .expect("configured solve should expose native integration preview");
    assert!(preview.attempted_steps > 0);
    assert_eq!(
        preview.attempted_steps,
        preview.accepted_steps + preview.rejected_steps
    );
    assert!(preview.total_iterations > 0);
    assert!(preview.final_t >= solver.config().t0);

    let probe = summary
        .native_step_probe
        .as_ref()
        .expect("configured solve should also expose legacy-compatible probe view");
    assert_eq!(probe.attempted_steps, preview.attempted_steps);
    assert_eq!(probe.accepted_steps, preview.accepted_steps);
    assert_eq!(probe.rejected_steps, preview.rejected_steps);

    assert_eq!(summary.native_statistics.algorithm_decision_calls, 1);
    assert!(summary.native_statistics.native_step_attempts > 0);
    assert!(summary.native_statistics.bridge_bdf_nlu_total > 0);
    assert!(summary.native_integration_solve.is_none());
}

#[test]
fn lsode2_default_native_execution_prefers_faithful_bdf_solve() {
    let config = exponential_decay_config().with_native_sparse_faer_backend();
    match config.native_execution {
        Lsode2NativeExecutionConfig::NativeSolve {
            max_step_attempts,
            max_accepted_steps,
        } => {
            assert_eq!(max_step_attempts, 200_000);
            assert_eq!(max_accepted_steps, 200_000);
        }
        other => panic!("expected faithful-native default execution, got {other:?}"),
    }
}

#[test]
fn lsode2_solve_can_use_configured_faithful_native_solve() {
    let mut solver = Lsode2Solver::new(
        exponential_decay_config()
            .with_native_sparse_faer_backend()
            .with_faithful_bdf_solve(128, 128),
    )
    .expect("LSODE2 faithful native-solve config should build");

    let summary = solver
        .solve_with_summary()
        .expect("LSODE2 faithful native solve should finish");

    assert!(
        summary.status == "finished_native_faithful"
            || summary.status == "finished_native_faithful_partial"
    );
    assert!(summary.native_integration_preview.is_none());
    let native_solve = summary
        .native_integration_solve
        .as_ref()
        .expect("faithful native solve should expose native integration summary");
    assert!(native_solve.attempted_steps > 0);
    assert!(
        native_solve.attempted_steps <= native_solve.accepted_steps + native_solve.rejected_steps
    );
    assert!(native_solve.accepted_steps > 0);
    assert_eq!(
        native_solve.accepted_t_history.len(),
        native_solve.accepted_y_history.len()
    );

    let final_t = summary
        .final_t
        .expect("native solve should provide final t");
    assert!(
        final_t >= solver.config().t0 && final_t <= solver.config().t_bound,
        "native solve final t should stay inside the integration interval, got={final_t:e}"
    );
    let final_y = summary
        .final_y
        .expect("native solve should provide final y");
    assert!(
        final_y[0].is_finite(),
        "native faithful solve should keep its state finite"
    );

    let (t, y) = solver.get_result();
    assert_eq!(t.len(), native_solve.accepted_t_history.len());
    assert_eq!(y.nrows(), native_solve.accepted_y_history.len());
    assert_eq!(solver.status(), summary.status);

    assert_eq!(summary.native_statistics.algorithm_decision_calls, 1);
    assert!(summary.native_statistics.native_step_attempts > 0);
    assert_eq!(summary.native_statistics.bridge_bdf_nlu_total, 0);
    assert_eq!(summary.statistics.backend_prepare_calls, 0);
    assert_eq!(summary.statistics.solve_calls, 0);
}

#[test]
fn lsode2_faithful_native_solve_falls_back_to_bridge_on_dense_backend() {
    let mut solver = Lsode2Solver::new(exponential_decay_config().with_faithful_bdf_solve(8, 8))
        .expect("dense LSODE2 config should build");

    let summary = solver
        .solve_with_summary()
        .expect("dense backend should fall back to bridge solve");
    assert_eq!(summary.status, "finished");
    assert!(summary.native_integration_solve.is_none());
    assert!(summary.native_statistics.bridge_bdf_nlu_total > 0);
}

#[test]
fn lsode2_algorithm_controller_snapshot_reports_auto_mode_bridge_fallback_honestly() {
    let mut solver = Lsode2Solver::new(
        exponential_decay_config()
            .with_controller(Lsode2ControllerConfig::automatic_adams_bdf())
            .with_native_banded_faithful_backend(),
    )
    .expect("LSODE2 automatic-controller config should build");

    let summary = solver
        .solve_with_summary()
        .expect("LSODE2 solve should finish through current BDF engine");

    assert!(
        summary.status == "finished_native_faithful"
            || summary.status == "finished_native_faithful_partial",
        "unexpected automatic-controller status: {}",
        summary.status
    );
    assert_eq!(summary.algorithm.controller_mode, "automatic_adams_bdf");
    assert_eq!(summary.algorithm.active_family, "adams");
    assert_eq!(summary.algorithm.executed_family, Some("adams"));
    assert!(summary.algorithm.method_switching_enabled);
    assert_eq!(summary.algorithm.preferred_family, "adams");
    assert!(
        summary.algorithm.switch_reason == "switch_probe_warmup"
            || summary.algorithm.switch_reason == "switch_advantage_not_met"
            || summary.algorithm.switch_reason == "insufficient_cost_evidence",
        "automatic controller should keep Adams during warmup or until cost evidence is available; got={}",
        summary.algorithm.switch_reason
    );
    assert!(!summary.algorithm.switch_uses_fallback);

    let stiff_decision = solver.algorithm_switch_decision_with_telemetry(
        Lsode2SwitchTelemetry::default()
            .with_stiffness_ratio(summary.algorithm.stiffness_ratio_threshold),
    );
    assert_eq!(stiff_decision.preferred_family, Lsode2MethodFamily::Bdf);
    assert_eq!(
        stiff_decision.reason,
        Lsode2SwitchReason::StiffnessSuspected
    );
    assert!(!stiff_decision.uses_fallback);
}

#[test]
fn lsode2_dstoda_switch_choreography_label_replay_reason_cost_stiff_gates() {
    let config = Lsode2ControllerConfig::automatic_adams_bdf()
        .with_method_switch_probe_steps(1)
        .with_stiffness_ratio_threshold(10.0)
        .with_convergence_failure_threshold(2)
        .with_min_cost_samples_for_switch(3);
    let caps = super::algorithm::Lsode2ControllerExecutionCapabilities {
        adams_engine_available: true,
    };

    // 1) ICOUNT-like warmup: keep current family.
    let warmup = config.switch_decision_with_probe_gate_and_capabilities_and_current_family(
        Lsode2SwitchTelemetry::quiet_nonstiff(),
        Some(false),
        caps,
        Some(Lsode2MethodFamily::Adams),
    );
    assert_eq!(warmup.preferred_family, Lsode2MethodFamily::Adams);
    assert_eq!(warmup.reason, Lsode2SwitchReason::SwitchProbeWarmup);

    // 2) Probe open + partial DSTODA step-cap telemetry: hold on step-advantage gate.
    let partial_step_adv =
        config.switch_decision_with_probe_gate_and_capabilities_and_current_family(
            Lsode2SwitchTelemetry::quiet_nonstiff()
                .with_adams_step_size_cap_estimate(1.0)
                .with_bdf_step_size_cap_estimate(1.0),
            Some(true),
            caps,
            Some(Lsode2MethodFamily::Adams),
        );
    assert_eq!(partial_step_adv.preferred_family, Lsode2MethodFamily::Adams);
    assert_eq!(
        partial_step_adv.reason,
        Lsode2SwitchReason::SwitchAdvantageNotMet
    );

    // 3) Probe open + cost telemetry but missing sample count gate.
    let insufficient_cost =
        config.switch_decision_with_probe_gate_and_capabilities_and_current_family(
            Lsode2SwitchTelemetry::quiet_nonstiff()
                .with_adams_step_cost_estimate(2.0)
                .with_bdf_step_cost_estimate(1.0)
                .with_adams_cost_samples(1)
                .with_bdf_cost_samples(3),
            Some(true),
            caps,
            Some(Lsode2MethodFamily::Adams),
        );
    assert_eq!(insufficient_cost.preferred_family, Lsode2MethodFamily::Adams);
    assert_eq!(
        insufficient_cost.reason,
        Lsode2SwitchReason::InsufficientCostEvidence
    );

    // 4) Probe open + enough cost evidence: prefer BDF by cost.
    let cost_bdf = config.switch_decision_with_probe_gate_and_capabilities_and_current_family(
        Lsode2SwitchTelemetry::quiet_nonstiff()
            .with_adams_step_cost_estimate(2.0)
            .with_bdf_step_cost_estimate(1.0)
            .with_adams_cost_samples(3)
            .with_bdf_cost_samples(3),
        Some(true),
        caps,
        Some(Lsode2MethodFamily::Adams),
    );
    assert_eq!(cost_bdf.preferred_family, Lsode2MethodFamily::Bdf);
    assert_eq!(cost_bdf.reason, Lsode2SwitchReason::CostPreferenceBdf);

    // 5) Stiffness gate dominates independently of probe state.
    let stiff = config.switch_decision_with_probe_gate_and_capabilities_and_current_family(
        Lsode2SwitchTelemetry::quiet_nonstiff().with_stiffness_ratio(10.0),
        Some(false),
        caps,
        Some(Lsode2MethodFamily::Adams),
    );
    assert_eq!(stiff.preferred_family, Lsode2MethodFamily::Bdf);
    assert_eq!(stiff.reason, Lsode2SwitchReason::StiffnessSuspected);

    // 6) Convergence-trouble override gate.
    let conv = config.switch_decision_with_probe_gate_and_capabilities_and_current_family(
        Lsode2SwitchTelemetry::quiet_nonstiff().with_convergence_failures(2),
        Some(true),
        caps,
        Some(Lsode2MethodFamily::Adams),
    );
    assert_eq!(conv.preferred_family, Lsode2MethodFamily::Bdf);
    assert_eq!(conv.reason, Lsode2SwitchReason::ConvergenceTrouble);
}

#[test]
fn lsode2_rejects_adams_only_on_bridge_path() {
    let config = exponential_decay_config().with_adams_only_controller();
    let plan = config.controller.execution_plan();
    assert!(!plan.is_executable_now());
    assert!(plan.requires_adams_engine);

    let err = match Lsode2Solver::new(config) {
        Ok(_) => panic!("Adams-only mode should not silently execute through BDF"),
        Err(err) => err,
    };

    let message = err.to_string();
    assert!(message.contains("Adams-only controller"));
    assert!(message.contains("requires native Adams execution support"));
}

#[test]
fn lsode2_allows_adams_only_with_faithful_native_solve() {
    let mut solver = Lsode2Solver::new(
        exponential_decay_config()
            .with_adams_only_controller()
            .with_native_sparse_faer_backend()
            .with_faithful_bdf_solve(512, 512),
    )
    .expect("adams-only should be allowed on native faithful execution path");

    let summary = solver
        .solve_with_summary()
        .expect("adams-only native faithful solve should complete");

    assert!(
        summary.status == "finished_native_faithful"
            || summary.status == "finished_native_faithful_partial"
    );
    assert_eq!(summary.algorithm.controller_mode, "adams_only");
    assert_eq!(summary.algorithm.preferred_family, "adams");
    assert_eq!(summary.algorithm.switch_reason, "fixed_controller");
    assert!(summary.native_integration_solve.is_some());
}

#[test]
fn lsode2_automatic_native_nonstiff_uses_cost_aware_family_selection_after_probe_warmup() {
    let mut solver = Lsode2Solver::new(
        exponential_decay_config()
            .with_automatic_adams_bdf_controller()
            .with_native_sparse_faer_backend()
            .with_faithful_bdf_solve(256, 128),
    )
    .expect("automatic non-stiff native config should build");

    let first = solver
        .solve_with_summary()
        .expect("first non-stiff native solve should finish");
    assert!(
        first.status == "finished_native_faithful"
            || first.status == "finished_native_faithful_partial"
    );
    assert_eq!(
        first.algorithm.preferred_family, "adams",
        "first non-stiff automatic run should stay in Adams warmup (LSODA-style start); got reason={}",
        first.algorithm.switch_reason
    );
    assert!(first.native_statistics.native_step_accepts > 0);
    assert!(
        solver
            .run_native_integration_preview_for_family(
                Lsode2NativeIntegrationLimits::new(64, 32),
                Lsode2MethodFamily::Adams,
            )
            .expect("non-stiff adams probe should run")
            .is_some(),
        "non-stiff adams probe should produce a native summary"
    );

    let second = solver
        .solve_with_summary()
        .expect("second non-stiff native solve should finish");
    assert!(
        second.status == "finished_native_faithful"
            || second.status == "finished_native_faithful_partial"
    );
    assert_eq!(second.algorithm.controller_mode, "automatic_adams_bdf");
    assert!(
        second.algorithm.switch_reason == "cost_preference_adams"
            || second.algorithm.switch_reason == "cost_preference_bdf"
            || second.algorithm.switch_reason == "nonstiff_preference"
            || second.algorithm.switch_reason == "convergence_trouble"
            || second.algorithm.switch_reason == "stiffness_suspected"
            || second.algorithm.switch_reason == "switch_advantage_not_met"
            || second.algorithm.switch_reason == "insufficient_cost_evidence",
        "automatic non-stiff path should switch by ODEPACK-style signals (cost/stiffness/convergence), not heuristic fallback; got={}",
        second.algorithm.switch_reason
    );
    assert_eq!(
        second.algorithm.executed_family,
        Some(second.algorithm.preferred_family),
        "native automatic path should execute the same family it prefers"
    );
    if second.algorithm.preferred_family == "adams" {
        assert!(
            second.native_statistics.native_adams_cost_samples > 0,
            "non-stiff automatic switch to Adams should be backed by recorded native Adams cost evidence"
        );
    } else {
        assert!(
            second.native_statistics.native_bdf_cost_samples > 0,
            "non-stiff automatic switch to BDF should be backed by recorded native BDF cost evidence"
        );
    }
}

#[test]
fn lsode2_automatic_native_stiff_keeps_bdf_family() {
    let mut solver = Lsode2Solver::new(
        stiff_relaxation_config()
            .with_controller(
                Lsode2ControllerConfig::automatic_adams_bdf().with_method_switch_probe_steps(1),
            )
            .with_native_sparse_faer_backend()
            .with_faithful_bdf_solve(512, 256),
    )
    .expect("automatic stiff native config should build");

    let first = solver
        .solve_with_summary()
        .expect("first stiff native solve should finish");
    assert!(
        first.status == "finished_native_faithful"
            || first.status == "finished_native_faithful_partial"
    );
    assert!(
        solver
            .run_native_integration_preview_for_family(
                Lsode2NativeIntegrationLimits::new(96, 48),
                Lsode2MethodFamily::Bdf,
            )
            .expect("stiff bdf probe should run")
            .is_some(),
        "stiff bdf probe should produce a native summary"
    );

    let second = solver
        .solve_with_summary()
        .expect("second stiff native solve should finish");
    assert!(
        second.status == "finished_native_faithful"
            || second.status == "finished_native_faithful_partial"
    );
    assert_eq!(second.algorithm.controller_mode, "automatic_adams_bdf");
    assert!(
        second.native_statistics.native_bdf_cost_samples > 0,
        "stiff path expected at least one native BDF cost sample before automatic decision"
    );
    assert!(
        second.algorithm.preferred_family == "adams" || second.algorithm.preferred_family == "bdf",
        "stiff automatic path should expose a valid method family, got={}",
        second.algorithm.preferred_family
    );
    assert_eq!(
        second.algorithm.executed_family,
        Some(second.algorithm.preferred_family)
    );
    // NOTE:
    // LSODA-first choreography starts on Adams and can remain on Adams during
    // warmup/step-advantage hold windows even on stiff setups.
    // We therefore assert parity on controller semantics and evidence, not on
    // a hard-coded family label.
    assert!(
        second.algorithm.switch_reason == "convergence_trouble"
            || second.algorithm.switch_reason == "stiffness_suspected"
            || second.algorithm.switch_reason == "cost_preference_bdf"
            || second.algorithm.switch_reason == "switch_advantage_not_met"
            || second.algorithm.switch_reason == "switch_probe_warmup"
            || second.algorithm.switch_reason == "insufficient_cost_evidence",
        "unexpected auto-switch reason on stiff path: {}",
        second.algorithm.switch_reason
    );
    assert!(
        second.native_statistics.native_bdf_cost_samples > 0,
        "stiff automatic switch should be backed by recorded native BDF cost evidence"
    );
}

#[test]
fn lsode2_rejects_invalid_controller_order_caps() {
    let err = match Lsode2Solver::new(
        exponential_decay_config()
            .with_controller(Lsode2ControllerConfig::bdf_only().with_max_bdf_order(0)),
    ) {
        Ok(_) => panic!("invalid BDF max order should be rejected"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("BDF max order"));

    let err = match Lsode2Solver::new(
        exponential_decay_config()
            .with_controller(Lsode2ControllerConfig::bdf_only().with_max_adams_order(13)),
    ) {
        Ok(_) => panic!("invalid Adams max order should be rejected"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("Adams max order"));
}

#[test]
fn lsode2_controller_bdf_order_cap_reaches_low_level_bdf_engine() {
    let mut solver = Lsode2Solver::new(
        exponential_decay_config()
            .with_controller(Lsode2ControllerConfig::bdf_only().with_max_bdf_order(2))
            .with_native_sparse_faer_backend(),
    )
    .expect("LSODE2 capped-order config should build");

    solver
        .prepare()
        .expect("LSODE2 prepare should install capped BDF engine");

    let snapshot = solver.algorithm_snapshot();
    assert_eq!(snapshot.max_bdf_order, 2);
    assert_eq!(snapshot.bdf_current_order, Some(1));
    assert_eq!(solver.bdf_max_order_cap(), 2);
}

#[test]
fn lsode2_sparse_native_path_solves_parameterized_decay() {
    let solver = solve_exponential_decay(
        parameterized_decay_config().with_backend(
            Lsode2BackendConfig::default()
                .with_linear_solver_backend(Lsode2LinearSolverBackend::SparseFaer),
        ),
    );

    let (_, y) = solver.get_result();
    let y_final = y[(y.nrows() - 1, 0)];
    let expected = (-1.0_f64).exp();
    assert!(
        (y_final - expected).abs() < 1e-4,
        "parameterized sparse decay mismatch: got={y_final:e}, expected={expected:e}"
    );
}

#[test]
fn lsode2_banded_native_path_solves_parameterized_decay() {
    let solver = solve_exponential_decay(
        parameterized_decay_config().with_backend(
            Lsode2BackendConfig::default()
                .with_linear_solver_backend(Lsode2LinearSolverBackend::BandedFaithful),
        ),
    );

    let (_, y) = solver.get_result();
    let y_final = y[(y.nrows() - 1, 0)];
    let expected = (-1.0_f64).exp();
    assert!(
        (y_final - expected).abs() < 1e-4,
        "parameterized banded decay mismatch: got={y_final:e}, expected={expected:e}"
    );
}

#[test]
fn lsode2_rejects_parameter_value_count_mismatch() {
    let config = parameterized_decay_config()
        .with_equation_parameter_values(DVector::from_vec(vec![-1.0, -2.0]));

    let err = match Lsode2Solver::new(config) {
        Ok(_) => panic!("parameter count mismatch should be rejected"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("expected 1 parameter values"));
}

#[test]
fn lsode2_sparse_native_path_uses_updated_parameter_values_before_solve() {
    let config = parameterized_decay_config()
        .with_equation_parameter_values(DVector::from_vec(vec![-2.0]))
        .with_backend(
            Lsode2BackendConfig::default()
                .with_linear_solver_backend(Lsode2LinearSolverBackend::SparseFaer),
        );
    let mut solver = Lsode2Solver::new(config).expect("LSODE2 config should build");
    solver
        .set_parameter_values(DVector::from_vec(vec![-1.0]))
        .expect("parameter update should succeed before solve");
    solver.solve().expect("LSODE2 solve should finish");

    let (_, y) = solver.get_result();
    let y_final = y[(y.nrows() - 1, 0)];
    let expected = (-1.0_f64).exp();
    assert!(
        (y_final - expected).abs() < 1e-4,
        "updated sparse parameter mismatch: got={y_final:e}, expected={expected:e}"
    );
}

#[test]
fn lsode2_native_path_uses_updated_parameter_values_after_prepare() {
    let config = parameterized_decay_config()
        .with_equation_parameter_values(DVector::from_vec(vec![-2.0]))
        .with_native_sparse_faer_backend();
    let mut solver = Lsode2Solver::new(config).expect("LSODE2 config should build");
    solver.prepare().expect("LSODE2 prepare should succeed");
    solver
        .set_parameter_values(DVector::from_vec(vec![-1.0]))
        .expect("parameter update should succeed after prepare");
    assert!(
        !solver.is_prepared(),
        "parameter updates should invalidate cached prepared BDF state"
    );
    solver.solve().expect("LSODE2 solve should finish");

    let (_, y) = solver.get_result();
    let y_final = y[(y.nrows() - 1, 0)];
    let expected = (-1.0_f64).exp();
    assert!(
        (y_final - expected).abs() < 1e-4,
        "post-prepare parameter update mismatch: got={y_final:e}, expected={expected:e}"
    );
    assert_eq!(solver.statistics().backend_prepare_calls, 1);
}

#[test]
fn lsode2_rejects_planned_but_unwired_fd_jacobian_backend() {
    let config = Lsode2ProblemConfig::new(
        vec![Expr::parse_expression("-y")],
        vec!["y".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0]),
        1.0,
        0.05,
        1e-6,
        1e-8,
    )
    .with_backend(
        Lsode2BackendConfig::default()
            .with_jacobian_backend(Lsode2JacobianBackend::FiniteDifference),
    );

    let err = match Lsode2Solver::new(config) {
        Ok(_) => panic!("FD Jacobian is a planned backend"),
        Err(err) => err,
    };
    assert!(
        err.to_string()
            .contains("finite-difference Jacobians are planned")
    );
}

#[test]
fn lsode2_analytical_native_sparse_backend_solves_exponential_decay() {
    let config = exponential_decay_config()
        .with_native_sparse_faer_backend()
        .with_analytical_callbacks(
            |_t, y: &DVector<f64>| DVector::from_vec(vec![-y[0]]),
            |_t, _y: &DVector<f64>| DMatrix::from_row_slice(1, 1, &[-1.0]),
        )
        .with_faithful_bdf_solve(4096, 4096);

    let mut solver =
        Lsode2Solver::new(config).expect("analytical native sparse LSODE2 config should build");
    let summary = solver
        .solve_with_summary()
        .expect("analytical native sparse LSODE2 solve should finish");

    assert!(
        summary.status == "finished_native_faithful"
            || summary.status == "finished_native_faithful_partial",
        "unexpected analytical native sparse status: {}",
        summary.status
    );
    let final_t = summary
        .final_t
        .expect("analytical solve should produce final t");
    assert!(
        final_t > 0.99,
        "native analytical solve should reach t_bound"
    );
    let y_final = summary
        .final_y
        .as_ref()
        .expect("analytical solve should produce final y")[0];
    let expected = (-1.0_f64).exp();
    assert!(
        (y_final - expected).abs() < 1e-4,
        "analytical native sparse decay mismatch: got={y_final:e}, expected={expected:e}"
    );
}

#[test]
fn lsode2_analytical_native_banded_backend_solves_exponential_decay() {
    let config = exponential_decay_config()
        .with_native_banded_faithful_backend()
        .with_analytical_callbacks(
            |_t, y: &DVector<f64>| DVector::from_vec(vec![-y[0]]),
            |_t, _y: &DVector<f64>| DMatrix::from_row_slice(1, 1, &[-1.0]),
        )
        .with_faithful_bdf_solve(4096, 4096);

    let mut solver =
        Lsode2Solver::new(config).expect("analytical native banded LSODE2 config should build");
    let summary = solver
        .solve_with_summary()
        .expect("analytical native banded LSODE2 solve should finish");

    assert!(
        summary.status == "finished_native_faithful"
            || summary.status == "finished_native_faithful_partial",
        "unexpected analytical native banded status: {}",
        summary.status
    );
    let final_t = summary
        .final_t
        .expect("analytical solve should produce final t");
    assert!(
        final_t > 0.99,
        "native analytical solve should reach t_bound"
    );
    let y_final = summary
        .final_y
        .as_ref()
        .expect("analytical solve should produce final y")[0];
    let expected = (-1.0_f64).exp();
    assert!(
        (y_final - expected).abs() < 1e-4,
        "analytical native banded decay mismatch: got={y_final:e}, expected={expected:e}"
    );
}

#[test]
fn lsode2_rejects_analytical_route_without_callbacks() {
    let config = exponential_decay_config()
        .with_native_sparse_faer_backend()
        .with_faithful_bdf_solve(4096, 4096)
        .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Analytical);

    let err = match Lsode2Solver::new(config) {
        Ok(_) => panic!("analytical route without callbacks should be rejected"),
        Err(err) => err,
    };
    assert!(
        err.to_string()
            .contains("analytical route requires residual and jacobian callbacks")
    );
}

#[test]
fn lsode2_rejects_analytical_route_without_native_execution_mode() {
    let config = exponential_decay_config()
        .with_native_sparse_faer_backend()
        .with_bridge_solve()
        .with_analytical_callbacks(
            |_t, y: &DVector<f64>| DVector::from_vec(vec![-y[0]]),
            |_t, _y: &DVector<f64>| DMatrix::from_row_slice(1, 1, &[-1.0]),
        );

    let err = match Lsode2Solver::new(config) {
        Ok(_) => panic!("analytical route should require native execution mode"),
        Err(err) => err,
    };
    assert!(
        err.to_string()
            .contains("analytical route is currently native-only")
    );
}

#[test]
fn lsode2_native_linear_backend_accepts_residual_generated_aot_config() {
    let config = exponential_decay_config().with_backend(
        Lsode2BackendConfig::native_banded_faithful().with_generated_backend(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release("target/lsode2-tests"),
        ),
    );

    let solver = Lsode2Solver::new(config)
        .expect("native banded path should accept residual-only generated AOT config");
    assert_eq!(
        solver.config().backend.linear_solver_backend,
        Lsode2LinearSolverBackend::BandedFaithful
    );
}

#[test]
fn lsode2_native_bdf_robertson_tracks_lsode_reference_scale() {
    // Fortran LSODE reference (sample output, t = 4e3):
    // y1 ≈ 1.831701e-01, y2 ≈ 8.940379e-07, y3 ≈ 8.168290e-01.
    let config = robertson_stiff_native_sparse_config();
    let mut solver =
        Lsode2Solver::new(config).expect("Robertson native sparse LSODE2 config should build");
    let summary = solver
        .solve_with_summary()
        .expect("Robertson native sparse LSODE2 solve should finish");

    assert!(
        summary.status == "finished_native_faithful"
            || summary.status == "finished_native_faithful_partial",
        "unexpected native Robertson status: {}",
        summary.status
    );
    let final_t = summary
        .final_t
        .expect("native Robertson solve should produce final t");
    assert!(
        final_t > 3.9e3,
        "native Robertson solve should reach near t_bound=4e3, got t={final_t:e}"
    );

    let y = summary
        .final_y
        .as_ref()
        .expect("native Robertson solve should produce final state");
    let y1 = y[0];
    let y2 = y[1];
    let y3 = y[2];

    assert!(
        (y1 - 1.831701e-1).abs() < 5.0e-3,
        "Robertson y1 mismatch at t≈4e3: got={y1:e}"
    );
    assert!(
        (y3 - 8.168290e-1).abs() < 5.0e-3,
        "Robertson y3 mismatch at t≈4e3: got={y3:e}"
    );
    assert!(
        y2.abs() < 1.0e-4,
        "Robertson y2 should stay near trace scale at t≈4e3, got={y2:e}"
    );

    let mass = y1 + y2 + y3;
    assert!(
        (mass - 1.0).abs() < 2.0e-3,
        "Robertson mass balance drift too large: y1+y2+y3={mass:e}"
    );
}

#[test]
fn lsode2_faithful_native_bdf_robertson_multi_run_stability() {
    let mut worst_mass_drift = 0.0_f64;
    let mut worst_y1_drift = 0.0_f64;
    let mut worst_y3_drift = 0.0_f64;

    for _ in 0..3 {
        let config =
            robertson_stiff_native_sparse_config().with_faithful_bdf_solve(200_000, 200_000);
        let mut solver = Lsode2Solver::new(config)
            .expect("Robertson faithful native LSODE2 config should build");
        let summary = solver
            .solve_with_summary()
            .expect("Robertson faithful native LSODE2 solve should finish");

        assert!(
            summary.status == "finished_native_faithful"
                || summary.status == "finished_native_faithful_partial",
            "unexpected faithful native Robertson status: {}",
            summary.status
        );

        let final_t = summary
            .final_t
            .expect("faithful native Robertson solve should produce final t");
        assert!(
            final_t > 3.9e3,
            "faithful native Robertson solve should reach near t_bound=4e3, got t={final_t:e}"
        );

        let y = summary
            .final_y
            .as_ref()
            .expect("faithful native Robertson solve should produce final state");
        let y1 = y[0];
        let y2 = y[1];
        let y3 = y[2];
        let mass = y1 + y2 + y3;

        worst_mass_drift = worst_mass_drift.max((mass - 1.0).abs());
        worst_y1_drift = worst_y1_drift.max((y1 - 1.831701e-1).abs());
        worst_y3_drift = worst_y3_drift.max((y3 - 8.168290e-1).abs());

        assert!(y2.abs() < 1.0e-4, "Robertson y2 drift too large: {y2:e}");
        assert!(
            summary.native_statistics.native_step_attempts > 0
                && summary.native_statistics.native_linear_solve_calls > 0,
            "faithful native Robertson run should perform native steps and linear solves"
        );
    }

    assert!(
        worst_y1_drift < 5.0e-3 && worst_y3_drift < 5.0e-3 && worst_mass_drift < 2.0e-3,
        "faithful native Robertson stability drift too large: y1={worst_y1_drift:e}, y3={worst_y3_drift:e}, mass={worst_mass_drift:e}"
    );
}

#[test]
fn lsode2_faithful_native_bdf_stiff_relaxation_matches_cosine_tail() {
    let config = stiff_relaxation_config()
        .with_native_banded_faithful_backend()
        .with_faithful_bdf_solve(32_768, 32_768);
    let mut solver = Lsode2Solver::new(config)
        .expect("stiff-relaxation faithful native banded config should build");
    let summary = solver
        .solve_with_summary()
        .expect("stiff-relaxation faithful native banded solve should finish");

    assert!(
        summary.status == "finished_native_faithful"
            || summary.status == "finished_native_faithful_partial",
        "unexpected faithful native stiff-relaxation status: {}",
        summary.status
    );
    let final_t = summary
        .final_t
        .expect("stiff-relaxation solve should produce final t");
    assert!(
        final_t > 0.99,
        "stiff-relaxation should reach near t_bound=1.0"
    );

    let y_final = summary
        .final_y
        .as_ref()
        .expect("stiff-relaxation solve should produce final y")[0];
    let expected = final_t.cos();
    assert!(
        (y_final - expected).abs() < 2.0e-3,
        "faithful native stiff-relaxation mismatch: got={y_final:e}, expected={expected:e}"
    );
}

#[test]
fn lsode2_automatic_native_robertson_records_stiff_switch_telemetry_and_native_stats() {
    let mut solver = Lsode2Solver::new(
        robertson_stiff_native_sparse_config()
            .with_controller(
                Lsode2ControllerConfig::automatic_adams_bdf()
                    .with_method_switch_probe_steps(1)
                    .with_stiffness_ratio_threshold(1.0),
            )
            .with_faithful_bdf_solve(200_000, 200_000),
    )
    .expect("automatic Robertson config should build");

    let summary = solver
        .solve_with_summary()
        .expect("automatic Robertson solve should finish");
    assert!(
        summary.status == "finished_native_faithful"
            || summary.status == "finished_native_faithful_partial"
    );
    assert!(
        summary.algorithm.preferred_family == "adams" || summary.algorithm.preferred_family == "bdf",
        "stiff Robertson should expose a valid family in auto mode, got={}",
        summary.algorithm.preferred_family
    );
    assert!(
        summary.algorithm.switch_reason == "switch_probe_warmup"
            || summary.algorithm.switch_reason == "switch_advantage_not_met"
            || summary.algorithm.switch_reason == "stiffness_suspected"
            || summary.algorithm.switch_reason == "convergence_trouble"
            || summary.algorithm.switch_reason == "cost_preference_bdf",
        "unexpected stiff Robertson switch reason: {}",
        summary.algorithm.switch_reason
    );
    assert!(
        summary.native_statistics.native_jacobian_calls > 0
            && summary.native_statistics.native_linear_solve_calls > 0
    );
}

#[test]
fn lsode2_automatic_native_stiff_relaxation_can_force_real_bdf_switch() {
    let mut solver = Lsode2Solver::new(
        stiff_relaxation_config()
            .with_native_banded_faithful_backend()
            .with_controller(
                Lsode2ControllerConfig::automatic_adams_bdf()
                    .with_method_switch_probe_steps(1)
                    .with_convergence_failure_threshold(1)
                    .with_rejection_threshold(1),
            )
            .with_faithful_bdf_solve(65_536, 65_536),
    )
    .expect("automatic stiff-relaxation config should build");

    let summary = solver
        .solve_with_summary()
        .expect("automatic stiff-relaxation solve should finish");
    assert!(
        summary.status == "finished_native_faithful"
            || summary.status == "finished_native_faithful_partial"
    );
    assert!(
        summary.native_statistics.preferred_bdf_count > 0,
        "stiff-relaxation should produce BDF preference when convergence/rejection gate is active"
    );
    assert!(
        summary.native_statistics.executed_bdf_count > 0,
        "stiff-relaxation should execute BDF steps when BDF is preferred"
    );
}

#[test]
fn lsode2_automatic_native_nonsteady_kinetics_switches_to_bdf_and_keeps_mass() {
    let mut solver = Lsode2Solver::new(
        nonsteady_chemical_kinetics_native_sparse_config()
            .with_controller(
                Lsode2ControllerConfig::automatic_adams_bdf()
                    .with_method_switch_probe_steps(1)
                    .with_stiffness_ratio_threshold(1.0)
                    .with_convergence_failure_threshold(1)
                    .with_rejection_threshold(1),
            )
            .with_faithful_bdf_solve(200_000, 200_000),
    )
    .expect("automatic non-steady kinetics config should build");

    let summary = solver
        .solve_with_summary()
        .expect("automatic non-steady kinetics solve should finish");
    assert!(
        summary.status == "finished_native_faithful"
            || summary.status == "finished_native_faithful_partial"
    );
    assert!(
        summary.native_statistics.preferred_bdf_count > 0,
        "stiff non-steady kinetics should produce at least one BDF preference"
    );
    assert!(
        summary.native_statistics.executed_bdf_count > 0,
        "stiff non-steady kinetics should execute BDF path"
    );

    let y = summary
        .final_y
        .as_ref()
        .expect("non-steady kinetics should expose final state");
    let mass = y[0] + y[1] + y[2];
    assert!(
        (mass - 1.0).abs() < 5.0e-4,
        "mass conservation drift too large for A->B->C kinetics: {mass:e}"
    );
    assert!(
        y[0] >= -1.0e-10 && y[1] >= -1.0e-10 && y[2] >= -1.0e-10,
        "species should stay non-negative up to tolerance: A={:e}, B={:e}, C={:e}",
        y[0],
        y[1],
        y[2]
    );
}

#[test]
fn lsode2_native_residual_aot_presets_select_expected_backends() {
    let sparse_tcc =
        exponential_decay_config().with_native_sparse_faer_aot_c_tcc("target/lsode2-tests");
    assert_eq!(
        sparse_tcc.backend.linear_solver_backend,
        Lsode2LinearSolverBackend::SparseFaer
    );
    assert_eq!(
        sparse_tcc
            .backend
            .generated_backend
            .aot_c_compiler
            .as_deref(),
        Some("tcc")
    );

    let banded_gcc =
        exponential_decay_config().with_native_banded_faithful_aot_c_gcc("target/lsode2-tests");
    assert_eq!(
        banded_gcc.backend.linear_solver_backend,
        Lsode2LinearSolverBackend::BandedFaithful
    );
    assert_eq!(
        banded_gcc
            .backend
            .generated_backend
            .aot_c_compiler
            .as_deref(),
        Some("gcc")
    );

    let banded_zig =
        exponential_decay_config().with_native_banded_faithful_aot_zig("target/lsode2-tests");
    assert_eq!(
        banded_zig.backend.linear_solver_backend,
        Lsode2LinearSolverBackend::BandedFaithful
    );
    assert!(
        banded_zig
            .backend
            .generated_backend
            .aot_c_compiler
            .is_none(),
        "Zig preset should not carry a C compiler override"
    );
}

#[test]
fn lsode2_native_banded_path_can_use_prelinked_residual_aot_backend() {
    let generated_backend = SymbolicIvpGeneratedBackendConfig::require_prebuilt()
        .with_crate_name_override(Some("generated_lsode2_prelinked_banded_test".to_string()))
        .with_module_name_override(Some("generated_lsode2_prelinked_banded_test".to_string()));
    let config = exponential_decay_config()
        .with_native_banded_faithful_generated_backend(generated_backend.clone());
    let residual_problem = prepare_symbolic_ivp_residual_problem(
        config.eq_system.clone(),
        config.values.clone(),
        config.arg.clone(),
        SymbolicIvpProblemOptions::new().with_aot_options(generated_backend.aot_options),
    )
    .expect("residual problem should prepare");
    let problem_key = residual_problem
        .prepare_residual_aot_problem(generated_backend.aot_options)
        .problem_key();
    let mut sparse_probe_backend = generated_backend.clone();
    sparse_probe_backend.build_policy = SymbolicIvpAotBuildPolicy::UseIfAvailable;
    let prepared_sparse = prepare_generated_symbolic_ivp_sparse_backend(
        config.eq_system.clone(),
        config.values.clone(),
        config.arg.clone(),
        SymbolicIvpProblemOptions::new().with_aot_options(generated_backend.aot_options),
        sparse_probe_backend,
    )
    .expect("sparse generated backend should produce one stable problem key");
    let residual_calls = Arc::new(AtomicUsize::new(0));
    let residual_calls_for_backend = Arc::clone(&residual_calls);

    register_linked_residual_backend(LinkedResidualAotBackend::new(
        problem_key.clone(),
        1,
        Arc::new(move |args: &[f64], out: &mut [f64]| {
            residual_calls_for_backend.fetch_add(1, Ordering::Relaxed);
            assert_eq!(out.len(), 1);
            assert!(args.len() >= 2, "IVP residual AOT args should be [t, y...]");
            out[0] = -args[1];
        }),
    ));
    register_linked_sparse_backend(LinkedSparseAotBackend::new(
        prepared_sparse.problem_key.clone(),
        1,
        (1, 1),
        prepared_sparse.jacobian_structure.nnz(),
        Arc::new(move |args: &[f64], out: &mut [f64]| {
            assert_eq!(out.len(), 1);
            out[0] = -args[1];
        }),
        Arc::new(move |_args: &[f64], out: &mut [f64]| {
            out.fill(-1.0);
        }),
    ));

    let result = (|| {
        let mut solver = Lsode2Solver::new(config).expect("LSODE2 config should build");
        solver.solve().expect("LSODE2 solve should finish");
        let (_, y) = solver.get_result();
        y[(y.nrows() - 1, 0)]
    })();

    unregister_linked_residual_backend(problem_key.as_str());
    unregister_linked_sparse_backend(prepared_sparse.problem_key.as_str());

    let expected = (-1.0_f64).exp();
    assert!(
        (result - expected).abs() < 1e-4,
        "prelinked residual AOT banded solve mismatch: got={result:e}, expected={expected:e}"
    );
    assert!(
        residual_calls.load(Ordering::Relaxed) > 0,
        "prelinked residual backend should be called during solve"
    );
}

#[test]
fn lsode2_native_sparse_symbolic_aot_uses_prelinked_sparse_jacobian_backend() {
    let generated_backend = SymbolicIvpGeneratedBackendConfig::require_prebuilt()
        .with_crate_name_override(Some(
            "generated_lsode2_prelinked_sparse_jacobian".to_string(),
        ))
        .with_module_name_override(Some(
            "generated_lsode2_prelinked_sparse_jacobian".to_string(),
        ));
    let mut probe_backend = generated_backend.clone();
    probe_backend.build_policy = SymbolicIvpAotBuildPolicy::UseIfAvailable;

    let mut config = exponential_decay_config()
        .with_native_sparse_faer_generated_backend(generated_backend.clone())
        .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
            execution: Lsode2SymbolicExecutionMode::Aot {
                toolchain: super::Lsode2AotToolchain::CTcc,
                profile: super::Lsode2AotProfile::Release,
            },
        });
    config.backend.generated_backend = generated_backend;

    let mut options = SymbolicIvpProblemOptions::new();
    if let Some(parameters) = config.equation_parameters.clone() {
        options = options.with_equation_parameters(parameters);
    }
    if let Some(values) = config.equation_parameter_values.clone() {
        options = options.with_equation_parameter_values(values);
    }
    options = options.with_aot_options(config.backend.generated_backend.aot_options);
    options = options.with_symbolic_assembly_backend(
        crate::symbolic::symbolic_ivp::IvpSymbolicAssemblyBackend::ExprLegacy,
    );

    let prepared_sparse = prepare_generated_symbolic_ivp_sparse_backend(
        config.eq_system.clone(),
        config.values.clone(),
        config.arg.clone(),
        options,
        probe_backend,
    )
    .expect("sparse generated preparation should produce a stable problem key");
    let problem_key = prepared_sparse.problem_key.clone();
    let nnz = prepared_sparse.jacobian_structure.nnz();
    let residual_problem = prepare_symbolic_ivp_residual_problem(
        config.eq_system.clone(),
        config.values.clone(),
        config.arg.clone(),
        SymbolicIvpProblemOptions::new()
            .with_aot_options(config.backend.generated_backend.aot_options)
            .with_symbolic_assembly_backend(
                crate::symbolic::symbolic_ivp::IvpSymbolicAssemblyBackend::ExprLegacy,
            ),
    )
    .expect("residual-only symbolic IVP problem should prepare");
    let residual_problem_key = residual_problem
        .prepare_residual_aot_problem(config.backend.generated_backend.aot_options)
        .problem_key();

    let jacobian_calls = Arc::new(AtomicUsize::new(0));
    let jacobian_calls_for_backend = Arc::clone(&jacobian_calls);
    register_linked_residual_backend(LinkedResidualAotBackend::new(
        residual_problem_key.clone(),
        1,
        Arc::new(move |args: &[f64], out: &mut [f64]| {
            assert!(args.len() >= 2, "IVP residual AOT args should be [t, y...]");
            assert_eq!(out.len(), 1);
            out[0] = -args[1];
        }),
    ));
    register_linked_sparse_backend(LinkedSparseAotBackend::new(
        problem_key.clone(),
        1,
        (1, 1),
        nnz,
        Arc::new(move |args: &[f64], out: &mut [f64]| {
            assert!(args.len() >= 2, "IVP sparse AOT args should be [t, y...]");
            assert_eq!(out.len(), 1);
            out[0] = -args[1];
        }),
        Arc::new(move |_args: &[f64], out: &mut [f64]| {
            jacobian_calls_for_backend.fetch_add(1, Ordering::Relaxed);
            assert_eq!(out.len(), nnz);
            out.fill(-1.0);
        }),
    ));

    let y_final = (|| {
        let mut solver =
            Lsode2Solver::new(config).expect("LSODE2 sparse symbolic AOT should build");
        solver
            .solve()
            .expect("LSODE2 sparse symbolic AOT solve should finish");
        let (_, y) = solver.get_result();
        y[(y.nrows() - 1, 0)]
    })();

    unregister_linked_sparse_backend(problem_key.as_str());
    unregister_linked_residual_backend(residual_problem_key.as_str());

    let expected = (-1.0_f64).exp();
    assert!(
        (y_final - expected).abs() < 1e-4,
        "prelinked sparse AOT Jacobian solve mismatch: got={y_final:e}, expected={expected:e}"
    );
    assert!(
        jacobian_calls.load(Ordering::Relaxed) > 0,
        "prelinked sparse AOT Jacobian evaluator should be called by LSODE2 solve"
    );
}

#[test]
fn lsode2_native_sparse_symbolic_aot_atom_view_uses_prelinked_sparse_jacobian_backend() {
    let generated_backend = SymbolicIvpGeneratedBackendConfig::require_prebuilt()
        .with_crate_name_override(Some(
            "generated_lsode2_prelinked_sparse_jacobian_atom".to_string(),
        ))
        .with_module_name_override(Some(
            "generated_lsode2_prelinked_sparse_jacobian_atom".to_string(),
        ));
    let mut probe_backend = generated_backend.clone();
    probe_backend.build_policy = SymbolicIvpAotBuildPolicy::UseIfAvailable;

    let mut config = exponential_decay_config()
        .with_native_sparse_faer_generated_backend(generated_backend.clone())
        .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::AtomView,
            execution: Lsode2SymbolicExecutionMode::Aot {
                toolchain: super::Lsode2AotToolchain::CTcc,
                profile: super::Lsode2AotProfile::Release,
            },
        });
    config.backend.generated_backend = generated_backend;

    let mut options = SymbolicIvpProblemOptions::new();
    if let Some(parameters) = config.equation_parameters.clone() {
        options = options.with_equation_parameters(parameters);
    }
    if let Some(values) = config.equation_parameter_values.clone() {
        options = options.with_equation_parameter_values(values);
    }
    options = options.with_aot_options(config.backend.generated_backend.aot_options);
    options = options.with_symbolic_assembly_backend(
        crate::symbolic::symbolic_ivp::IvpSymbolicAssemblyBackend::AtomView,
    );

    let prepared_sparse = prepare_generated_symbolic_ivp_sparse_backend(
        config.eq_system.clone(),
        config.values.clone(),
        config.arg.clone(),
        options,
        probe_backend,
    )
    .expect("sparse generated preparation (AtomView) should produce a stable problem key");
    let problem_key = prepared_sparse.problem_key.clone();
    let nnz = prepared_sparse.jacobian_structure.nnz();
    let residual_problem = prepare_symbolic_ivp_residual_problem(
        config.eq_system.clone(),
        config.values.clone(),
        config.arg.clone(),
        SymbolicIvpProblemOptions::new()
            .with_aot_options(config.backend.generated_backend.aot_options)
            .with_symbolic_assembly_backend(
                crate::symbolic::symbolic_ivp::IvpSymbolicAssemblyBackend::AtomView,
            ),
    )
    .expect("residual-only symbolic IVP problem (AtomView) should prepare");
    let residual_problem_key = residual_problem
        .prepare_residual_aot_problem(config.backend.generated_backend.aot_options)
        .problem_key();

    let jacobian_calls = Arc::new(AtomicUsize::new(0));
    let jacobian_calls_for_backend = Arc::clone(&jacobian_calls);
    register_linked_residual_backend(LinkedResidualAotBackend::new(
        residual_problem_key.clone(),
        1,
        Arc::new(move |args: &[f64], out: &mut [f64]| {
            assert!(args.len() >= 2, "IVP residual AOT args should be [t, y...]");
            assert_eq!(out.len(), 1);
            out[0] = -args[1];
        }),
    ));
    register_linked_sparse_backend(LinkedSparseAotBackend::new(
        problem_key.clone(),
        1,
        (1, 1),
        nnz,
        Arc::new(move |args: &[f64], out: &mut [f64]| {
            assert!(args.len() >= 2, "IVP sparse AOT args should be [t, y...]");
            assert_eq!(out.len(), 1);
            out[0] = -args[1];
        }),
        Arc::new(move |_args: &[f64], out: &mut [f64]| {
            jacobian_calls_for_backend.fetch_add(1, Ordering::Relaxed);
            assert_eq!(out.len(), nnz);
            out.fill(-1.0);
        }),
    ));

    let y_final = (|| {
        let mut solver =
            Lsode2Solver::new(config).expect("LSODE2 sparse symbolic AOT (AtomView) should build");
        solver
            .solve()
            .expect("LSODE2 sparse symbolic AOT (AtomView) solve should finish");
        let (_, y) = solver.get_result();
        y[(y.nrows() - 1, 0)]
    })();

    unregister_linked_sparse_backend(problem_key.as_str());
    unregister_linked_residual_backend(residual_problem_key.as_str());

    let expected = (-1.0_f64).exp();
    assert!(
        (y_final - expected).abs() < 1e-4,
        "prelinked sparse AOT Jacobian solve mismatch (AtomView): got={y_final:e}, expected={expected:e}"
    );
    assert!(
        jacobian_calls.load(Ordering::Relaxed) > 0,
        "prelinked sparse AOT Jacobian evaluator (AtomView) should be called by LSODE2 solve"
    );
}

#[test]
fn lsode2_native_banded_symbolic_aot_atom_view_uses_prelinked_sparse_jacobian_backend() {
    let generated_backend = SymbolicIvpGeneratedBackendConfig::require_prebuilt()
        .with_crate_name_override(Some(
            "generated_lsode2_prelinked_banded_jacobian_atom".to_string(),
        ))
        .with_module_name_override(Some(
            "generated_lsode2_prelinked_banded_jacobian_atom".to_string(),
        ));
    let mut probe_backend = generated_backend.clone();
    probe_backend.build_policy = SymbolicIvpAotBuildPolicy::UseIfAvailable;

    let mut config = exponential_decay_config()
        .with_native_banded_faithful_generated_backend(generated_backend.clone())
        .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::AtomView,
            execution: Lsode2SymbolicExecutionMode::Aot {
                toolchain: super::Lsode2AotToolchain::CTcc,
                profile: super::Lsode2AotProfile::Release,
            },
        });
    config.backend.generated_backend = generated_backend;

    let mut options = SymbolicIvpProblemOptions::new();
    if let Some(parameters) = config.equation_parameters.clone() {
        options = options.with_equation_parameters(parameters);
    }
    if let Some(values) = config.equation_parameter_values.clone() {
        options = options.with_equation_parameter_values(values);
    }
    options = options.with_aot_options(config.backend.generated_backend.aot_options);
    options = options.with_symbolic_assembly_backend(
        crate::symbolic::symbolic_ivp::IvpSymbolicAssemblyBackend::AtomView,
    );

    let prepared_sparse = prepare_generated_symbolic_ivp_sparse_backend(
        config.eq_system.clone(),
        config.values.clone(),
        config.arg.clone(),
        options,
        probe_backend,
    )
    .expect("banded sparse-generated preparation (AtomView) should produce a stable problem key");
    let problem_key = prepared_sparse.problem_key.clone();
    let nnz = prepared_sparse.jacobian_structure.nnz();
    let residual_problem = prepare_symbolic_ivp_residual_problem(
        config.eq_system.clone(),
        config.values.clone(),
        config.arg.clone(),
        SymbolicIvpProblemOptions::new()
            .with_aot_options(config.backend.generated_backend.aot_options)
            .with_symbolic_assembly_backend(
                crate::symbolic::symbolic_ivp::IvpSymbolicAssemblyBackend::AtomView,
            ),
    )
    .expect("residual-only symbolic IVP problem (AtomView, banded) should prepare");
    let residual_problem_key = residual_problem
        .prepare_residual_aot_problem(config.backend.generated_backend.aot_options)
        .problem_key();

    let jacobian_calls = Arc::new(AtomicUsize::new(0));
    let jacobian_calls_for_backend = Arc::clone(&jacobian_calls);
    register_linked_residual_backend(LinkedResidualAotBackend::new(
        residual_problem_key.clone(),
        1,
        Arc::new(move |args: &[f64], out: &mut [f64]| {
            assert!(args.len() >= 2, "IVP residual AOT args should be [t, y...]");
            assert_eq!(out.len(), 1);
            out[0] = -args[1];
        }),
    ));
    register_linked_sparse_backend(LinkedSparseAotBackend::new(
        problem_key.clone(),
        1,
        (1, 1),
        nnz,
        Arc::new(move |args: &[f64], out: &mut [f64]| {
            assert!(args.len() >= 2, "IVP sparse AOT args should be [t, y...]");
            assert_eq!(out.len(), 1);
            out[0] = -args[1];
        }),
        Arc::new(move |_args: &[f64], out: &mut [f64]| {
            jacobian_calls_for_backend.fetch_add(1, Ordering::Relaxed);
            assert_eq!(out.len(), nnz);
            out.fill(-1.0);
        }),
    ));

    let y_final = (|| {
        let mut solver =
            Lsode2Solver::new(config).expect("LSODE2 banded symbolic AOT (AtomView) should build");
        solver
            .solve()
            .expect("LSODE2 banded symbolic AOT (AtomView) solve should finish");
        let (_, y) = solver.get_result();
        y[(y.nrows() - 1, 0)]
    })();

    unregister_linked_sparse_backend(problem_key.as_str());
    unregister_linked_residual_backend(residual_problem_key.as_str());

    let expected = (-1.0_f64).exp();
    assert!(
        (y_final - expected).abs() < 1e-4,
        "prelinked banded AOT Jacobian solve mismatch (AtomView): got={y_final:e}, expected={expected:e}"
    );
    assert!(
        jacobian_calls.load(Ordering::Relaxed) > 0,
        "prelinked sparse AOT Jacobian evaluator (AtomView, banded) should be called by LSODE2 solve"
    );
}

#[test]
fn lsode2_fortran_labels_replay_quality_gate() {
    let mut dstoda = Lsode2DstodaState::default();
    let mut stats = Lsode2NativeStatistics::default();

    let mut record = |state: &Lsode2DstodaState| {
        stats.record_dstoda_flags(
            state.jacobian_currency(),
            state.ipup(),
            state.ipup_trigger(),
            state.kflag(),
            state.icf(),
            state.iret(),
            state.redo_stage(),
        );
    };

    // 1) baseline accepted-like control state.
    dstoda.mark_jacobian_current(0);
    record(&dstoda);

    // 2) error-test failure path.
    dstoda.record_error_test_failure();
    record(&dstoda);

    // 3) repeated-error reset choreography.
    dstoda.record_repeated_error_test_reset();
    record(&dstoda);

    // 4) stale-J one-shot same-step refresh branch.
    dstoda.mark_jacobian_stale();
    let _ = dstoda.decide_after_corrector_failure(Lsode2IterationMode::JacobianBased);
    record(&dstoda);

    // 5) post-refresh no-recover branch.
    dstoda.mark_jacobian_current(0);
    let _ = dstoda.decide_after_corrector_failure(Lsode2IterationMode::JacobianBased);
    record(&dstoda);

    // 6) terminal repeated convergence branch.
    dstoda.record_repeated_convergence_failure();
    record(&dstoda);

    // 7) predictor-driven IPUP reason replay.
    dstoda.record_step_accepted();
    dstoda.mark_jacobian_current(0);
    dstoda.set_coefficient_ratio(1.31);
    dstoda.maybe_request_jacobian_update_before_predict(1, Lsode2IterationMode::JacobianBased);
    record(&dstoda);

    // 8) forced restart-with-derivative-refresh branch.
    dstoda.mark_jacobian_current(0);
    dstoda.record_repeated_error_test_failure();
    record(&dstoda);

    assert_eq!(stats.native_kflag_ok_count, 2);
    assert_eq!(stats.native_kflag_error_test_failure_count, 2);
    assert_eq!(stats.native_kflag_repeated_convergence_failure_count, 1);
    assert_eq!(stats.native_icf_refresh_requested_count, 1);
    assert_eq!(stats.native_icf_refresh_did_not_recover_count, 2);
    assert_eq!(stats.native_iret_retry_after_error_test_failure_count, 1);
    assert_eq!(stats.native_iret_restart_with_derivative_refresh_count, 1);
    assert_eq!(stats.native_redo_corrector_refresh_same_step_count, 1);
    assert_eq!(stats.native_redo_corrector_failure_retry_count, 2);
    assert_eq!(stats.native_redo_error_test_retry_count, 1);
    assert_eq!(stats.native_redo_repeated_error_reset_count, 1);
    assert_eq!(stats.native_ipup_trigger_predictor_rc_ccmax_count, 1);

    // terminal KFLAG=-2 class must be represented.
    assert!(stats.native_kflag_repeated_convergence_failure_count > 0);
}

#[test]
fn lsode2_dstoda_terminal_convergence_reason_groups_map_to_kflag_minus_two_class() {
    // Branch A: repeated convergence failures (MXNCF-like terminal).
    let mut mxncf_cycle = make_bdf_cycle_for_parity(Lsode2StepControlConfig {
        max_convergence_failures: 1,
        ..Lsode2StepControlConfig::default()
    });
    mxncf_cycle.state_mut().set_step_size(1.0).unwrap();
    let mxncf_retry = mxncf_cycle.reject_after_nonlinear_failure().unwrap();
    assert_eq!(
        mxncf_retry.action,
        Lsode2RetryAction::FailRepeatedConvergenceFailures
    );
    assert_eq!(mxncf_cycle.kflag_code(), -2);
    assert_eq!(mxncf_cycle.iredo().code(), 1);

    // Branch B: HMIN guard terminal from convergence-retract path.
    let mut hmin_cycle = make_bdf_cycle_for_parity(Lsode2StepControlConfig {
        h_min: 0.3,
        max_convergence_failures: 10,
        ..Lsode2StepControlConfig::default()
    });
    hmin_cycle.state_mut().set_step_size(1.0).unwrap();
    let hmin_retry = hmin_cycle.reject_after_nonlinear_failure().unwrap();
    assert_eq!(hmin_retry.action, Lsode2RetryAction::FailStepSizeUnderflow);
    assert_eq!(hmin_cycle.kflag_code(), -2);
    assert_eq!(hmin_cycle.iredo().code(), 1);
}

#[test]
fn lsode2_backend_parity_checklist_lambdify_and_aot_exprlegacy_and_atomview() {
    let session_tag = unique_test_tag("aot_parity");
    let assemblies = [
        Lsode2SymbolicAssemblyBackend::ExprLegacy,
        Lsode2SymbolicAssemblyBackend::AtomView,
    ];
    let structures = [
        Lsode2LinearSystemStructure::Dense,
        Lsode2LinearSystemStructure::Sparse,
        Lsode2LinearSystemStructure::Banded { kl: 1, ku: 1 },
    ];

    // Lambdify parity surface: all assembly/structure combinations should solve.
    for assembly in assemblies {
        for structure in structures {
            let source = Lsode2ResidualJacobianSource::Symbolic {
                assembly,
                execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
            };
            let config = exponential_decay_config()
                .with_residual_jacobian_source(source)
                .with_linear_system_structure(structure)
                .with_linear_solver_policy(Lsode2LinearSolverPolicy::Auto);
            assert_exponential_decay_solve(config);
        }
    }

    // AOT parity surface: run prelinked runtime callbacks for dense/sparse/banded.
    for assembly in assemblies {
        let assembly_tag = match assembly {
            Lsode2SymbolicAssemblyBackend::ExprLegacy => "exprlegacy",
            Lsode2SymbolicAssemblyBackend::AtomView => "atomview",
        };

        run_prelinked_dense_aot_case(
            assembly,
            &format!("{session_tag}_{assembly_tag}_dense"),
        );
        run_prelinked_sparse_or_banded_aot_case(
            assembly,
            Lsode2LinearSystemStructure::Sparse,
            &format!("{session_tag}_{assembly_tag}_sparse"),
        );
        run_prelinked_sparse_or_banded_aot_case(
            assembly,
            Lsode2LinearSystemStructure::Banded { kl: 1, ku: 1 },
            &format!("{session_tag}_{assembly_tag}_banded"),
        );
    }
}

#[test]
fn lsode2_aot_tcc_sparse_cold_warm_diagnostic() {
    let out_base = unique_aot_diag_output_dir("sparse_tcc_cold_warm");
    let shared_tag = "shared_artifact";

    let cold = run_sparse_tcc_aot_diag_case(
        "cold(shared artifact)",
        sparse_tcc_symbolic_aot_config(out_base.clone(), shared_tag),
    );
    let warm_reuse = run_sparse_tcc_aot_diag_case(
        "warm(reuse same artifact)",
        sparse_tcc_symbolic_aot_config(out_base.clone(), shared_tag),
    );
    let warm_unique = run_sparse_tcc_aot_diag_case(
        "warm(unique artifact)",
        sparse_tcc_symbolic_aot_config(out_base, "unique_artifact"),
    );

    let rows = [&cold, &warm_reuse, &warm_unique];
    println!(
        "[LSODE2 debug] AOT+tcc sparse cold/warm diagnostic; focus: infra lock vs numerical path"
    );
    println!("label | status_kind | error_class | detail");
    println!("--------------------------------------------------------------------------------");
    for row in rows {
        println!(
            "{} | {} | {} | {}",
            row.label, row.status_kind, row.error_class, row.detail
        );
    }

    for row in rows {
        let known = matches!(
            row.error_class.as_str(),
            "-" | "permission_denied_or_file_lock"
                | "generated_backend_failure"
                | "toolchain_not_available"
        );
        assert!(
            known,
            "unexpected diagnostic class for {}: status={} class={} detail={}",
            row.label, row.status_kind, row.error_class, row.detail
        );
    }
}

// --- Additional parity micro-tests ---

#[test]
fn parity_hmin_guard_terminal() {
    use crate::numerical::LSODE2::{Lsode2StepControlConfig};

    // Create a cycle with h_min set to 1.0 and initial h just slightly above
    // the HMIN*1.00001 guard to trigger the guard path on reject.
    let mut cycle = {
        let state = crate::numerical::LSODE2::Lsode2RuntimeState::new(
            0.0,
            &[1.0],
            1.000009,
            3,
            Lsode2StepControlConfig { h_min: 1.0, ..Default::default() },
        )
        .expect("runtime state");
        let ec = crate::numerical::LSODE2::Lsode2ErrorController::new(
            crate::numerical::LSODE2::Lsode2Tolerance::scalar(1.0e-3, 1.0e-6),
            crate::numerical::LSODE2::Lsode2ErrorControlConfig::default(),
        )
        .expect("error controller");
        crate::numerical::LSODE2::Lsode2StepCycle::new(state, ec)
    };

    let outcome = cycle
        .finish_with_local_error(1.0, &[0.9], &[1.0e-1])
        .expect("finish should return outcome");

    match outcome {
        crate::numerical::LSODE2::Lsode2StepCycleOutcome::Rejected { retry, .. } => {
            assert_eq!(retry.action, crate::numerical::LSODE2::Lsode2RetryAction::FailStepSizeUnderflow);
            assert_eq!(cycle.kflag(), crate::numerical::LSODE2::Lsode2Kflag::ErrorTestFailure);
        }
        other => panic!("expected HMIN-guard rejection, got {other:?}"),
    }
}

#[test]
fn parity_mxncg_terminal_reached() {
    use crate::numerical::LSODE2::{Lsode2StepControlConfig};

    // Configure cycle with max_error_test_failures small so repeated rejects hit terminal
    let mut cycle = {
        let state = crate::numerical::LSODE2::Lsode2RuntimeState::new(
            0.0,
            &[1.0],
            0.5,
            3,
            Lsode2StepControlConfig { max_error_test_failures: 2, ..Default::default() },
        )
        .expect("runtime state");
        let ec = crate::numerical::LSODE2::Lsode2ErrorController::new(
            crate::numerical::LSODE2::Lsode2Tolerance::scalar(1.0e-3, 1.0e-6),
            crate::numerical::LSODE2::Lsode2ErrorControlConfig::default(),
        )
        .expect("error controller");
        crate::numerical::LSODE2::Lsode2StepCycle::new(state, ec)
    };

    // Cause repeated error-test failures
    cycle.state_mut().reject_after_error_test_with_hint(0.5, 3).unwrap();
    cycle.state_mut().reject_after_error_test_with_hint(0.25, 3).unwrap();

    let outcome = cycle
        .finish_with_local_error(0.1, &[0.9], &[1.0e-1])
        .expect("finish outcome");

    match outcome {
        crate::numerical::LSODE2::Lsode2StepCycleOutcome::Rejected { retry, .. } => {
            assert_eq!(retry.action, crate::numerical::LSODE2::Lsode2RetryAction::FailRepeatedErrorTestFailures);
            assert_eq!(cycle.kflag(), crate::numerical::LSODE2::Lsode2Kflag::RepeatedErrorTestFailure);
        }
        other => panic!("expected repeated error-test terminal, got {other:?}"),
    }
}

#[test]
fn parity_lmax_order_reduction() {
    use crate::numerical::LSODE2::{Lsode2StepControlConfig};

    // Create a state with max_order=3 and ensure order selection never exceeds it
    let mut cycle = {
        let state = crate::numerical::LSODE2::Lsode2RuntimeState::new(
            0.0,
            &[1.0],
            0.1,
            3, // max_order (LMAX equivalent)
            Lsode2StepControlConfig::default(),
        )
        .expect("runtime state");
        let ec = crate::numerical::LSODE2::Lsode2ErrorController::new(
            crate::numerical::LSODE2::Lsode2Tolerance::scalar(1.0e-3, 1.0e-6),
            crate::numerical::LSODE2::Lsode2ErrorControlConfig::default(),
        )
        .expect("error controller");
        crate::numerical::LSODE2::Lsode2StepCycle::new(state, ec)
    };

    // Call select_post_accept_order and assert the returned order respects the runtime max_order
    let decision = cycle.select_post_accept_order(&[0.9], 1.0, None).expect("select order");
    assert!(decision.order_new <= cycle.state().max_order(), "new order must not exceed configured max_order");
}
