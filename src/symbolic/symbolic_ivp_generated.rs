//! High-level generated-backend orchestration for shared IVP symbolic problems.
//!
//! This module is the user-facing lifecycle layer above:
//! - shared IVP symbolic preparation,
//! - dense AOT preparation,
//! - backend-agnostic materialized build requests,
//! - resolver reuse,
//! - and linked compiled dense backends (`Rust` / `C` / `Zig`).
//!
//! Practical policy note:
//! unlike large sparse BVP pipelines, dense IVP Jacobians are usually much smaller,
//! are built once, and are then reused across many implicit steps and Newton
//! iterations. Because of that, IVP defaults deliberately bias towards
//! better runtime throughput (`C + gcc`) instead of the cheapest possible build.
//!
//! Practical guidance from current IVP comparisons:
//! - `Lambdify` remains the safest default for small IVP systems and many BDF
//!   scenarios where Jacobians are rebuilt rarely and residuals dominate.
//! - `C + tcc` is the most practical compiled choice when startup latency still
//!   matters but you want a native dense backend, especially for larger
//!   Backward Euler problems.
//! - `C + gcc` is the runtime-oriented compiled choice and is worth trying when
//!   you expect many repeated dense implicit solves on the same problem.
//! - `Zig` is available and can be competitive, but today the most polished
//!   IVP-facing choices are still `Lambdify`, `C + tcc`, and `C + gcc`.

use crate::symbolic::bvp::atom_aot::AtomAotMatrixLayout;
use crate::symbolic::codegen::c_backend::codegen_c_aot_build::CAotCompileConfig;
use crate::symbolic::codegen::c_backend::codegen_c_aot_registry::register_c_build_in_registry;
use crate::symbolic::codegen::c_backend::codegen_c_aot_runtime_link::{
    register_generated_c_banded_backend, register_generated_c_dense_backend,
    register_generated_c_residual_backend, register_generated_c_sparse_backend,
};
use crate::symbolic::codegen::codegen_aot_driver::{
    AotBuildPreset, AotCodegenBackend, ExecutedGeneratedAotBuild, GeneratedAotBuildRequest,
    GeneratedAotBuildResult, generated_aot_artifact_from_prepared_problem,
    generated_aot_build_request_from_artifact,
};
use crate::symbolic::codegen::codegen_aot_lifecycle::AotArtifactState;
use crate::symbolic::codegen::codegen_aot_resolution::{AotResolutionStatus, AotResolver};
use crate::symbolic::codegen::codegen_aot_runtime_link::{
    LinkedDenseAotBackend, LinkedJacobianLayout, LinkedResidualAotBackend, LinkedSparseAotBackend,
    register_generated_banded_cdylib_backend, register_generated_dense_cdylib_backend,
    register_generated_residual_cdylib_backend, register_generated_sparse_cdylib_backend,
    resolve_linked_dense_backend, resolve_linked_residual_backend, resolve_linked_sparse_backend,
};
use crate::symbolic::codegen::codegen_provider_api::{
    BackendKind, MatrixBackend, PreparedProblem, PreparedSparseProblem,
};
use crate::symbolic::codegen::codegen_runtime_api::{
    ResidualChunkingStrategy, SparseJacobianStructure,
};
use crate::symbolic::codegen::codegen_tasks::{
    IvpResidualTask, SparseChunkingStrategy, SparseExprEntry, SparseJacobianTask,
};
use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use crate::symbolic::codegen::zig_backend::codegen_zig_aot_registry::register_zig_build_in_registry;
use crate::symbolic::codegen::zig_backend::codegen_zig_aot_runtime_link::{
    register_generated_zig_banded_backend, register_generated_zig_dense_backend,
    register_generated_zig_residual_backend, register_generated_zig_sparse_backend,
};
use crate::symbolic::ivp_telemetry::{IvpColdStage, IvpTelemetry};
use crate::symbolic::symbolic_ivp::{
    IvpBackendError, PreparedSymbolicIvpProblem, PreparedSymbolicIvpResidualProblem,
    SymbolicIvpAotOptions, SymbolicIvpProblemOptions, prepare_symbolic_ivp_problem,
    prepare_symbolic_ivp_residual_problem,
};
use crate::symbolic::symbolic_ivp_aot::{
    PreparedSymbolicIvpAtomAotProblem, generated_aot_artifact_from_symbolic_ivp_atom_problem,
    generated_aot_artifact_from_symbolic_ivp_residual_problem,
    prepared_atom_aot_problem_from_residual_problem,
    prepared_atom_aot_problem_from_symbolic_ivp_problem_with_layout,
    try_generated_aot_artifact_from_symbolic_ivp_problem,
};
use log::{debug, info, warn};
use std::fmt;
use std::path::{Path, PathBuf};

use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::sleep;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

static REBUILD_OUTPUT_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Aggregated runtime/setup statistics for one symbolic IVP solver instance.
#[derive(Debug, Clone, Default)]
pub struct IvpBackendStatistics {
    pub backend_prepare_calls: usize,
    pub backend_prepare_ms_total: f64,
    pub solve_calls: usize,
    pub solve_ms_total: f64,
    pub step_calls: usize,
    pub nonlinear_solve_calls: usize,
    pub nonlinear_iterations_total: usize,
    pub residual_calls: usize,
    pub residual_ms_total: f64,
    pub jacobian_calls: usize,
    pub jacobian_ms_total: f64,
    pub bdf_nfev_total: usize,
    pub bdf_njev_total: usize,
    pub bdf_nlu_total: usize,
}

impl IvpBackendStatistics {
    pub fn record_backend_prepare_duration(&mut self, duration: Duration) {
        self.backend_prepare_calls += 1;
        self.backend_prepare_ms_total += duration.as_secs_f64() * 1_000.0;
    }

    pub fn record_solve_duration(&mut self, duration: Duration) {
        self.solve_calls += 1;
        self.solve_ms_total += duration.as_secs_f64() * 1_000.0;
    }

    pub fn record_residual_duration(&mut self, duration: Duration) {
        self.residual_calls += 1;
        self.residual_ms_total += duration.as_secs_f64() * 1_000.0;
    }

    pub fn record_jacobian_duration(&mut self, duration: Duration) {
        self.jacobian_calls += 1;
        self.jacobian_ms_total += duration.as_secs_f64() * 1_000.0;
    }

    pub fn avg_residual_ms(&self) -> Option<f64> {
        (self.residual_calls > 0).then(|| self.residual_ms_total / self.residual_calls as f64)
    }

    pub fn avg_jacobian_ms(&self) -> Option<f64> {
        (self.jacobian_calls > 0).then(|| self.jacobian_ms_total / self.jacobian_calls as f64)
    }

    pub fn avg_nonlinear_iterations(&self) -> Option<f64> {
        (self.nonlinear_solve_calls > 0)
            .then(|| self.nonlinear_iterations_total as f64 / self.nonlinear_solve_calls as f64)
    }

    pub fn table_report(&self) -> String {
        format!(
            "prepare_calls={} prepare_ms_total={:.3} solve_calls={} solve_ms_total={:.3} steps={} nonlinear_solves={} nonlinear_iters_total={} nonlinear_iters_avg={:.3} residual_calls={} residual_ms_total={:.3} residual_ms_avg={:.6} jacobian_calls={} jacobian_ms_total={:.3} jacobian_ms_avg={:.6} bdf[nfev/njev/nlu]={}/{}/{}",
            self.backend_prepare_calls,
            self.backend_prepare_ms_total,
            self.solve_calls,
            self.solve_ms_total,
            self.step_calls,
            self.nonlinear_solve_calls,
            self.nonlinear_iterations_total,
            self.avg_nonlinear_iterations().unwrap_or(0.0),
            self.residual_calls,
            self.residual_ms_total,
            self.avg_residual_ms().unwrap_or(0.0),
            self.jacobian_calls,
            self.jacobian_ms_total,
            self.avg_jacobian_ms().unwrap_or(0.0),
            self.bdf_nfev_total,
            self.bdf_njev_total,
            self.bdf_nlu_total,
        )
    }
}

/// High-level generated-backend mode for dense IVP problems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DenseIvpGeneratedBackendMode {
    /// Prefer compiled AOT when available and otherwise keep lambdify.
    #[default]
    Defaults,
    /// Require a prebuilt compiled AOT backend.
    RequirePrebuilt,
    /// Build a release AOT artifact when it is missing.
    ///
    /// For IVP this is intentionally runtime-oriented: the default emitted backend
    /// is `C + gcc`, because the generated dense callbacks are typically reused many
    /// times after the initial build.
    BuildIfMissingRelease,
}

/// Build policy for symbolic IVP generated backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SymbolicIvpAotBuildPolicy {
    /// Use compiled AOT only when an artifact is already available.
    #[default]
    UseIfAvailable,
    /// Require an existing compiled artifact.
    RequirePrebuilt,
    /// Build the generated crate if the compiled artifact is missing.
    BuildIfMissing { profile: AotBuildProfile },
    /// Always rebuild the generated crate.
    RebuildAlways { profile: AotBuildProfile },
}

/// High-level result of backend selection for one IVP symbolic problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectedSymbolicIvpBackendKind {
    Lambdify,
    AotCompiled,
    AotRegisteredButNotBuilt,
    AotMissing,
}

/// Emits one cache decision on a cold generated-backend selection path.
///
/// The phase is static so selection diagnostics do not allocate a formatted
/// detail string. This helper is never called by callbacks.
fn log_aot_cache_selection(
    telemetry: &IvpTelemetry,
    route: &'static str,
    problem_key: &str,
    phase: &'static str,
    selection: SelectedSymbolicIvpBackendKind,
) {
    let event = if selection == SelectedSymbolicIvpBackendKind::AotCompiled {
        crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::CacheHit
    } else {
        crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::CacheMiss
    };
    telemetry.log_aot_event(event, route, problem_key, phase);
}

/// Measures one generated-backend cold stage without retaining a scope or
/// allocating on the callback path. The helper is also used by `Off` mode:
/// `start_cold_stage` then returns `None` and the recording call is a no-op.
#[inline]
fn measure_cold_stage<T>(
    telemetry: &IvpTelemetry,
    stage: IvpColdStage,
    operation: impl FnOnce() -> T,
) -> T {
    let started = telemetry.start_cold_stage(stage);
    let result = operation();
    telemetry.record_cold_stage(stage, started);
    result
}

#[inline]
fn measure_optional_cold_stage<T>(
    telemetry: Option<&IvpTelemetry>,
    stage: IvpColdStage,
    operation: impl FnOnce() -> T,
) -> T {
    match telemetry {
        Some(telemetry) => measure_cold_stage(telemetry, stage, operation),
        None => operation(),
    }
}

/// User-facing configuration for symbolic IVP generated backend orchestration.
#[derive(Debug, Clone)]
pub struct SymbolicIvpGeneratedBackendConfig {
    /// Optional resolver snapshot reused across calls.
    pub resolver: Option<AotResolver>,
    /// Dense IVP AOT runtime-plan chunking options.
    pub aot_options: SymbolicIvpAotOptions,
    /// Residual chunking used by generated residual paths.
    pub residual_chunking_strategy: ResidualChunkingStrategy,
    /// Sparse Jacobian chunking used by sparse/banded native-AOT Jacobian paths.
    pub sparse_jacobian_chunking_strategy: SparseChunkingStrategy,
    /// Lifecycle build policy.
    pub build_policy: SymbolicIvpAotBuildPolicy,
    /// Backend used to emit generated dense IVP artifacts.
    pub aot_codegen_backend: AotCodegenBackend,
    /// Optional explicit C compiler override for `C` AOT backends.
    pub aot_c_compiler: Option<String>,
    /// Parent directory where generated crates should be materialized.
    pub output_parent_dir: Option<PathBuf>,
    /// Optional explicit generated crate name.
    pub crate_name_override: Option<String>,
    /// Optional explicit generated module name.
    pub module_name_override: Option<String>,
}

impl Default for SymbolicIvpGeneratedBackendConfig {
    fn default() -> Self {
        Self {
            resolver: None,
            aot_options: SymbolicIvpAotOptions::default(),
            residual_chunking_strategy: ResidualChunkingStrategy::Whole,
            sparse_jacobian_chunking_strategy: SparseChunkingStrategy::Whole,
            build_policy: SymbolicIvpAotBuildPolicy::default(),
            aot_codegen_backend: AotCodegenBackend::default(),
            aot_c_compiler: None,
            output_parent_dir: None,
            crate_name_override: None,
            module_name_override: None,
        }
    }
}

impl SymbolicIvpGeneratedBackendConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn defaults() -> Self {
        Self::new()
    }

    pub fn require_prebuilt() -> Self {
        Self::new().with_build_policy(SymbolicIvpAotBuildPolicy::RequirePrebuilt)
    }

    /// Recommended practical IVP default for compiled repeated solves:
    /// keep release-quality codegen and prefer `C + gcc` runtime throughput.
    pub fn build_if_missing_release(output_parent_dir: impl Into<PathBuf>) -> Self {
        Self::new()
            .with_output_parent_dir(Some(output_parent_dir.into()))
            .with_build_policy(SymbolicIvpAotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Release,
            })
            .with_c_gcc()
    }

    pub fn from_mode(mode: DenseIvpGeneratedBackendMode) -> Self {
        match mode {
            DenseIvpGeneratedBackendMode::Defaults => Self::defaults(),
            DenseIvpGeneratedBackendMode::RequirePrebuilt => Self::require_prebuilt(),
            DenseIvpGeneratedBackendMode::BuildIfMissingRelease => Self::new()
                .with_build_policy(SymbolicIvpAotBuildPolicy::BuildIfMissing {
                    profile: AotBuildProfile::Release,
                })
                .with_c_gcc(),
        }
    }

    pub fn with_resolver(mut self, resolver: Option<AotResolver>) -> Self {
        self.resolver = resolver;
        self
    }

    /// Removes every registered generated artifact from the current resolver snapshot.
    ///
    /// This is a conservative explicit cleanup path for story/debug/cold-build workflows.
    /// It does not unload dynamic libraries or touch live callbacks.
    pub fn cleanup_registered_aot_artifacts(&mut self) -> std::io::Result<usize> {
        let Some(resolver) = self.resolver.as_mut() else {
            return Ok(0);
        };

        let problem_keys = resolver.registry().problem_keys();
        let mut removed = 0;
        for problem_key in problem_keys {
            if resolver.cleanup_artifact_by_problem_key(&problem_key)? {
                removed += 1;
            }
        }
        Ok(removed)
    }

    pub fn with_aot_options(mut self, aot_options: SymbolicIvpAotOptions) -> Self {
        self.aot_options = aot_options;
        self
    }

    pub fn with_residual_chunking_strategy(
        mut self,
        residual_chunking_strategy: ResidualChunkingStrategy,
    ) -> Self {
        self.residual_chunking_strategy = residual_chunking_strategy;
        self
    }

    pub fn with_sparse_jacobian_chunking_strategy(
        mut self,
        sparse_jacobian_chunking_strategy: SparseChunkingStrategy,
    ) -> Self {
        self.sparse_jacobian_chunking_strategy = sparse_jacobian_chunking_strategy;
        self
    }

    pub fn with_build_policy(mut self, build_policy: SymbolicIvpAotBuildPolicy) -> Self {
        self.build_policy = build_policy;
        self
    }

    pub fn with_aot_codegen_backend(mut self, backend: AotCodegenBackend) -> Self {
        self.aot_codegen_backend = backend;
        self
    }

    pub fn with_aot_c_compiler(mut self, compiler: impl Into<String>) -> Self {
        self.aot_c_compiler = Some(compiler.into());
        self
    }

    pub fn with_output_parent_dir(mut self, output_parent_dir: Option<PathBuf>) -> Self {
        self.output_parent_dir = output_parent_dir;
        self
    }

    pub fn with_crate_name_override(mut self, crate_name_override: Option<String>) -> Self {
        self.crate_name_override = crate_name_override;
        self
    }

    pub fn with_module_name_override(mut self, module_name_override: Option<String>) -> Self {
        self.module_name_override = module_name_override;
        self
    }

    fn output_parent_dir(&self) -> Result<&Path, SymbolicIvpGeneratedError> {
        self.output_parent_dir
            .as_deref()
            .ok_or(SymbolicIvpGeneratedError::AotBuildOutputDirMissing)
    }

    /// Uses `C + tcc` when faster bootstrap is more important than peak runtime.
    pub fn with_c_tcc(self) -> Self {
        self.with_aot_codegen_backend(AotCodegenBackend::C)
            .with_aot_c_compiler("tcc")
    }

    /// Uses `C + gcc` when runtime throughput matters more than startup cost.
    pub fn with_c_gcc(self) -> Self {
        self.with_aot_codegen_backend(AotCodegenBackend::C)
            .with_aot_c_compiler("gcc")
    }

    /// Recommended dense IVP compiled path when one Jacobian will be reused
    /// across many implicit steps and Newton iterations.
    pub fn for_repeated_solves(self) -> Self {
        self.with_c_gcc()
    }

    /// Uses `Rust` for dense IVP generated artifacts.
    ///
    /// This is mostly a reference / compatibility backend for IVP. The default
    /// practical repeated-solve policy prefers `C + gcc`.
    pub fn with_rust(self) -> Self {
        let mut config = self.with_aot_codegen_backend(AotCodegenBackend::Rust);
        config.aot_c_compiler = None;
        config
    }

    /// Uses Zig for dense IVP generated artifacts.
    pub fn with_zig(self) -> Self {
        let mut config = self.with_aot_codegen_backend(AotCodegenBackend::Zig);
        config.aot_c_compiler = None;
        config
    }
}

/// Errors surfaced by the high-level IVP generated-backend layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymbolicIvpGeneratedError {
    IvpBackend(IvpBackendError),
    CompiledAotArtifactMissing(String),
    CompiledAotArtifactNotBuilt(String),
    CompiledAotRuntimeUnavailable(String),
    AotLifecycle(crate::symbolic::codegen::codegen_aot_lifecycle::AotLifecycleError),
    AotBuildOutputDirMissing,
    AotBuildFailed(String),
}

impl fmt::Display for SymbolicIvpGeneratedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IvpBackend(err) => write!(f, "{err}"),
            Self::CompiledAotArtifactMissing(message)
            | Self::CompiledAotArtifactNotBuilt(message)
            | Self::CompiledAotRuntimeUnavailable(message)
            | Self::AotBuildFailed(message) => write!(f, "{message}"),
            Self::AotLifecycle(error) => write!(f, "{error}"),
            Self::AotBuildOutputDirMissing => {
                write!(
                    f,
                    "symbolic IVP generated backend build requested without output directory"
                )
            }
        }
    }
}

impl std::error::Error for SymbolicIvpGeneratedError {}

impl From<IvpBackendError> for SymbolicIvpGeneratedError {
    fn from(value: IvpBackendError) -> Self {
        Self::IvpBackend(value)
    }
}

impl From<crate::symbolic::codegen::codegen_aot_lifecycle::AotLifecycleError>
    for SymbolicIvpGeneratedError
{
    fn from(value: crate::symbolic::codegen::codegen_aot_lifecycle::AotLifecycleError) -> Self {
        Self::AotLifecycle(value)
    }
}

fn executed_process_output(executed: &ExecutedGeneratedAotBuild) -> (Option<i32>, String, String) {
    match executed {
        ExecutedGeneratedAotBuild::Rust(result) => (
            result.status_code,
            result.stdout.clone(),
            result.stderr.clone(),
        ),
        ExecutedGeneratedAotBuild::C(result) => (
            result.status_code,
            result.stdout.clone(),
            result.stderr.clone(),
        ),
        ExecutedGeneratedAotBuild::Zig(result) => (
            result.status_code,
            result.stdout.clone(),
            result.stderr.clone(),
        ),
    }
}

fn is_transient_aot_infra_failure(text: &str) -> bool {
    let low = text.to_ascii_lowercase();
    low.contains("permission denied")
        || low.contains("access is denied")
        || low.contains("being used by another process")
        || low.contains("resource busy")
        || low.contains("temporarily unavailable")
        || low.contains("failed to spawn build runner")
        || low.contains("could not write")
        || low.contains("file is locked")
        || low.contains("sharing violation")
}

fn retry_exhausted_aot_message(
    build_context: &str,
    attempts: usize,
    detail: &str,
    transient: bool,
) -> String {
    let class = if transient {
        "transient infrastructure failure"
    } else {
        "deterministic build failure"
    };
    format!(
        "AOT build execution failed ({build_context}) after {attempts} attempt(s); classified as {class}. \
If this is a transient infrastructure failure, check stale compiler processes, file locks and antivirus/indexer interference; \
otherwise inspect compiler stdout/stderr below.\n\
detail:\n{detail}"
    )
}

fn execute_generated_build_with_retry(
    build: &GeneratedAotBuildResult,
    build_context: &str,
    telemetry: Option<&IvpTelemetry>,
    route: &'static str,
    problem_key: &str,
) -> Result<ExecutedGeneratedAotBuild, SymbolicIvpGeneratedError> {
    const MAX_ATTEMPTS: usize = 3;
    let mut last_failure: Option<String> = None;
    let mut last_transient = false;
    let mut quarantine_attempted = false;
    let mut attempts_completed = 0usize;

    for attempt in 1..=MAX_ATTEMPTS {
        attempts_completed = attempt;
        if let Some(telemetry) = telemetry {
            telemetry.record_aot_build_attempt(attempt > 1);
        }
        debug!(
            target: "rustedscithe::symbolic::aot",
            "executing generated AOT build context={} attempt={}/{}",
            build_context,
            attempt,
            MAX_ATTEMPTS
        );
        match build.execute() {
            Ok(executed) => {
                if executed.succeeded() {
                    if let Some(telemetry) = telemetry {
                        telemetry.record_aot_build_result(true);
                    }
                    return Ok(executed);
                }
                let (status, stdout, stderr) = executed_process_output(&executed);
                let detail = format!(
                    "command={}\nstatus={status:?}\nstdout:\n{stdout}\nstderr:\n{stderr}",
                    build.command_line()
                );
                let transient = is_transient_aot_infra_failure(&detail);
                last_transient = transient;
                last_failure = Some(detail);
                if transient && attempt < MAX_ATTEMPTS {
                    quarantine_attempted = true;
                    match crate::symbolic::codegen::codegen_aot_lifecycle::quarantine_generated_tree(
                        &build.workdir(),
                        problem_key,
                    ) {
                        Ok(Some(_)) => {
                            if let Some(telemetry) = telemetry {
                                telemetry.log_aot_event(
                                    crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::Quarantined,
                                    route,
                                    problem_key,
                                    "transient build tree quarantined before retry",
                                );
                            }
                        }
                        Ok(None) => {}
                        Err(error) => {
                            last_failure = Some(error.to_string());
                            last_transient = false;
                            break;
                        }
                    }
                    if let Some(telemetry) = telemetry {
                        telemetry.log_aot_event(
                            crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::Retry,
                            route,
                            problem_key,
                            "transient build failure; retrying",
                        );
                    }
                    sleep(Duration::from_millis((attempt as u64) * 120));
                    continue;
                }
            }
            Err(err) => {
                let detail = format!("command={}\nerror={}", build.command_line(), err);
                let transient = is_transient_aot_infra_failure(&detail);
                last_transient = transient;
                last_failure = Some(detail);
                if transient && attempt < MAX_ATTEMPTS {
                    quarantine_attempted = true;
                    match crate::symbolic::codegen::codegen_aot_lifecycle::quarantine_generated_tree(
                        &build.workdir(),
                        problem_key,
                    ) {
                        Ok(Some(_)) => {
                            if let Some(telemetry) = telemetry {
                                telemetry.log_aot_event(
                                    crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::Quarantined,
                                    route,
                                    problem_key,
                                    "transient build tree quarantined before retry",
                                );
                            }
                        }
                        Ok(None) => {}
                        Err(error) => {
                            last_failure = Some(error.to_string());
                            last_transient = false;
                            break;
                        }
                    }
                    if let Some(telemetry) = telemetry {
                        telemetry.log_aot_event(
                            crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::Retry,
                            route,
                            problem_key,
                            "transient build failure; retrying",
                        );
                    }
                    sleep(Duration::from_millis((attempt as u64) * 120));
                    continue;
                }
            }
        }
        break;
    }

    let detail = last_failure.unwrap_or_else(|| "unknown build failure".to_string());
    if let Some(telemetry) = telemetry {
        telemetry.record_aot_build_result(false);
    }
    let mut diagnostics =
        crate::symbolic::codegen::codegen_aot_lifecycle::AotFailureDiagnostics::new(
            crate::symbolic::codegen::codegen_aot_lifecycle::AotLifecycleStage::Build,
            if last_transient {
                crate::symbolic::codegen::codegen_aot_lifecycle::AotFailureKind::RetryExhausted
            } else {
                crate::symbolic::codegen::codegen_aot_lifecycle::AotFailureKind::Compiler
            },
            problem_key,
            retry_exhausted_aot_message(build_context, attempts_completed, &detail, last_transient),
        );
    diagnostics.attempts = attempts_completed as u32;
    diagnostics.root_kind = Some(if last_transient {
        crate::symbolic::codegen::codegen_aot_lifecycle::AotFailureKind::Lock
    } else {
        crate::symbolic::codegen::codegen_aot_lifecycle::AotFailureKind::Compiler
    });
    diagnostics.quarantine_attempted = quarantine_attempted;
    Err(SymbolicIvpGeneratedError::AotLifecycle(
        crate::symbolic::codegen::codegen_aot_lifecycle::AotLifecycleError::new(diagnostics),
    ))
}

fn runtime_registration_error_message(
    route: &str,
    backend: AotCodegenBackend,
    problem_key: &str,
    artifact_path: &std::path::Path,
    err: String,
) -> crate::symbolic::codegen::codegen_aot_lifecycle::AotLifecycleError {
    let detail = format!(
        "symbolic IVP {route} compiled AOT artifact could not be registered as a linked runtime \
(backend={backend:?}; problem_key={problem_key}; artifact_path={}). \
This usually means dynamic loading failed, the artifact is stale/incompatible, or the file is still locked by another process. \
Rebuild the artifact, check toolchain ABI compatibility, and make sure no stale process keeps the library open. detail: {err}",
        artifact_path.display()
    );
    crate::symbolic::codegen::codegen_aot_lifecycle::AotLifecycleError::new(
        crate::symbolic::codegen::codegen_aot_lifecycle::AotFailureDiagnostics::new(
            crate::symbolic::codegen::codegen_aot_lifecycle::AotLifecycleStage::Link,
            crate::symbolic::codegen::codegen_aot_lifecycle::AotFailureKind::Link,
            problem_key,
            detail,
        ),
    )
}

fn materialization_error(
    route: &str,
    problem_key: &str,
    error: std::io::Error,
) -> SymbolicIvpGeneratedError {
    SymbolicIvpGeneratedError::AotLifecycle(
        crate::symbolic::codegen::codegen_aot_lifecycle::AotLifecycleError::new(
            crate::symbolic::codegen::codegen_aot_lifecycle::AotFailureDiagnostics::new(
                crate::symbolic::codegen::codegen_aot_lifecycle::AotLifecycleStage::Materialized,
                crate::symbolic::codegen::codegen_aot_lifecycle::AotFailureKind::Io,
                problem_key,
                format!("route={route}; generated artifact materialization failed: {error}"),
            ),
        ),
    )
}

fn aot_context_message(
    route: &str,
    problem_key: &str,
    config: &SymbolicIvpGeneratedBackendConfig,
) -> String {
    let compiler = config.aot_c_compiler.as_deref().unwrap_or("-");
    let output_parent = config
        .output_parent_dir
        .as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "-".to_string());
    format!(
        "route={route}; problem_key={problem_key}; build_policy={:?}; codegen_backend={:?}; c_compiler={compiler}; output_parent_dir={output_parent}",
        config.build_policy, config.aot_codegen_backend
    )
}

fn missing_aot_message(
    route: &str,
    problem_key: &str,
    config: &SymbolicIvpGeneratedBackendConfig,
) -> String {
    format!(
        "symbolic IVP {route} AOT artifact is missing ({context}). For strict RequirePrebuilt, run BuildIfMissing/RebuildAlways once with the same problem, frontend, matrix backend, toolchain and chunking policy, then reuse the returned resolver.",
        context = aot_context_message(route, problem_key, config)
    )
}

fn not_built_aot_message(
    route: &str,
    problem_key: &str,
    config: &SymbolicIvpGeneratedBackendConfig,
) -> String {
    format!(
        "symbolic IVP {route} AOT artifact is registered but the expected compiled file is not present ({context}). Rebuild the artifact or clean the stale resolver entry before using RequirePrebuilt.",
        context = aot_context_message(route, problem_key, config)
    )
}

fn runtime_unavailable_aot_message(
    route: &str,
    problem_key: &str,
    config: &SymbolicIvpGeneratedBackendConfig,
) -> String {
    format!(
        "symbolic IVP {route} compiled AOT artifact exists but no linked runtime is registered ({context}). Re-register the generated artifact or rebuild it in this process before using RequirePrebuilt.",
        context = aot_context_message(route, problem_key, config)
    )
}

/// Result of preparing one IVP symbolic problem through the high-level
/// generated-backend layer.
pub struct PreparedGeneratedSymbolicIvpProblem {
    pub problem: PreparedSymbolicIvpProblem,
    pub selected_backend: SelectedSymbolicIvpBackendKind,
    pub updated_resolver: Option<AotResolver>,
    pub build_result: Option<GeneratedAotBuildResult>,
    runtime_owner: Option<PreparedIvpAotRuntime>,
}

impl PreparedGeneratedSymbolicIvpProblem {
    pub fn into_problem(self) -> PreparedSymbolicIvpProblem {
        self.problem
    }

    /// Returns the unified AOT owner when this result selected a linked
    /// compiled runtime. The older resolver/build fields remain compatibility
    /// snapshots for callers that still inspect them separately.
    pub fn aot_runtime(&self) -> Option<&PreparedIvpAotRuntime> {
        self.runtime_owner.as_ref()
    }

    /// Fallible production accessor that rejects a stale linked runtime before
    /// it is handed to an IVP solver. The infallible accessor above remains a
    /// compatibility view for callers that already own the lifecycle.
    pub fn try_aot_runtime(
        &self,
    ) -> Result<Option<&PreparedIvpAotRuntime>, PreparedIvpRuntimeError> {
        self.runtime_owner
            .as_ref()
            .map(|runtime| runtime.validate().map(|()| runtime))
            .transpose()
    }
}

/// Result of preparing one residual-only IVP symbolic problem through the
/// high-level generated-backend layer.
pub struct PreparedGeneratedSymbolicIvpResidualProblem {
    pub problem: PreparedSymbolicIvpResidualProblem,
    pub selected_backend: SelectedSymbolicIvpBackendKind,
    pub updated_resolver: Option<AotResolver>,
    pub build_result: Option<GeneratedAotBuildResult>,
    runtime_owner: Option<PreparedIvpAotRuntime>,
}

impl PreparedGeneratedSymbolicIvpResidualProblem {
    pub fn into_problem(self) -> PreparedSymbolicIvpResidualProblem {
        self.problem
    }

    /// Returns the unified AOT owner when this result selected a linked
    /// compiled runtime.
    pub fn aot_runtime(&self) -> Option<&PreparedIvpAotRuntime> {
        self.runtime_owner.as_ref()
    }

    /// Fallible production accessor that validates artifact identity and
    /// resolver readiness before a residual-only runtime is used.
    pub fn try_aot_runtime(
        &self,
    ) -> Result<Option<&PreparedIvpAotRuntime>, PreparedIvpRuntimeError> {
        self.runtime_owner
            .as_ref()
            .map(|runtime| runtime.validate().map(|()| runtime))
            .transpose()
    }
}

/// Result of preparing one sparse-IVP generated backend (residual + sparse
/// Jacobian values) through the high-level lifecycle.
///
/// This is the LSODE2-oriented AOT path where Jacobian values are produced by
/// compiled callbacks instead of lambdified symbolic closures.
pub struct PreparedGeneratedSymbolicIvpSparseBackend {
    pub problem_key: String,
    pub selected_backend: SelectedSymbolicIvpBackendKind,
    pub linked_backend: Option<LinkedSparseAotBackend>,
    pub jacobian_structure: SparseJacobianStructure,
    /// The immutable preparation/lifecycle telemetry stream used to build
    /// this generated sparse or compact-Banded runtime. Keeping it on the
    /// prepared result makes cold-stage performance reports possible without
    /// reaching into private symbolic preparation state.
    pub telemetry: IvpTelemetry,
    pub updated_resolver: Option<AotResolver>,
    pub build_result: Option<GeneratedAotBuildResult>,
    runtime_owner: Option<PreparedIvpAotRuntime>,
}

impl PreparedGeneratedSymbolicIvpSparseBackend {
    /// Returns the unified AOT owner when this result selected a linked
    /// compiled runtime.
    pub fn aot_runtime(&self) -> Option<&PreparedIvpAotRuntime> {
        self.runtime_owner.as_ref()
    }

    /// Fallible production accessor that validates artifact identity and
    /// resolver readiness before a sparse or compact-Banded runtime is used.
    pub fn try_aot_runtime(
        &self,
    ) -> Result<Option<&PreparedIvpAotRuntime>, PreparedIvpRuntimeError> {
        self.runtime_owner
            .as_ref()
            .map(|runtime| runtime.validate().map(|()| runtime))
            .transpose()
    }
}

#[derive(Clone)]
enum PreparedIvpLinkedRuntime {
    Dense(LinkedDenseAotBackend),
    Sparse(LinkedSparseAotBackend),
    Residual(LinkedResidualAotBackend),
}

/// Unified owner for one prepared linked AOT runtime and its lifecycle
/// identity. Compatibility result fields remain available, but new lifecycle
/// code should keep this owner intact so artifact identity, resolver snapshot,
/// build metadata and linked callbacks travel together.
pub struct PreparedIvpAotRuntime {
    problem_key: String,
    selected_backend: SelectedSymbolicIvpBackendKind,
    resolver: Option<AotResolver>,
    build_result: Option<GeneratedAotBuildResult>,
    linked: PreparedIvpLinkedRuntime,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreparedIvpRuntimeError {
    LinkedProblemKeyMismatch {
        expected: String,
        actual: String,
    },
    InvalidLinkedLayout {
        runtime: &'static str,
        message: String,
    },
    ArtifactManifestMismatch {
        problem_key: String,
        registered_key: String,
        manifest_key: String,
    },
    ArtifactInvalidated {
        problem_key: String,
        state: AotArtifactState,
        detail: String,
    },
    ArtifactMissing {
        problem_key: String,
    },
    ArtifactNotReady {
        problem_key: String,
        status: AotResolutionStatus,
    },
}

impl fmt::Display for PreparedIvpRuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LinkedProblemKeyMismatch { expected, actual } => write!(
                f,
                "prepared IVP AOT linked runtime key mismatch: owner={expected}, linked={actual}"
            ),
            Self::InvalidLinkedLayout { runtime, message } => write!(
                f,
                "prepared IVP AOT {runtime} linked layout is invalid: {message}"
            ),
            Self::ArtifactManifestMismatch {
                problem_key,
                registered_key,
                manifest_key,
            } => write!(
                f,
                "prepared IVP AOT artifact manifest mismatch for {problem_key}: registered key={registered_key}, manifest key={manifest_key}"
            ),
            Self::ArtifactInvalidated {
                problem_key,
                state,
                detail,
            } => write!(
                f,
                "prepared IVP AOT artifact is invalidated for {problem_key}: state={state:?}; {detail}"
            ),
            Self::ArtifactMissing { problem_key } => {
                write!(
                    f,
                    "prepared IVP AOT artifact is missing for problem key {problem_key}"
                )
            }
            Self::ArtifactNotReady {
                problem_key,
                status,
            } => write!(
                f,
                "prepared IVP AOT artifact is not ready for problem key {problem_key}: {status:?}"
            ),
        }
    }
}

impl std::error::Error for PreparedIvpRuntimeError {}

impl PreparedIvpAotRuntime {
    fn new(
        problem_key: String,
        selected_backend: SelectedSymbolicIvpBackendKind,
        resolver: Option<AotResolver>,
        build_result: Option<GeneratedAotBuildResult>,
        linked: PreparedIvpLinkedRuntime,
    ) -> Self {
        Self {
            problem_key,
            selected_backend,
            resolver,
            build_result,
            linked,
        }
    }

    pub fn problem_key(&self) -> &str {
        &self.problem_key
    }

    pub fn selected_backend(&self) -> SelectedSymbolicIvpBackendKind {
        self.selected_backend
    }

    pub fn resolver(&self) -> Option<&AotResolver> {
        self.resolver.as_ref()
    }

    pub fn build_result(&self) -> Option<&GeneratedAotBuildResult> {
        self.build_result.as_ref()
    }

    pub fn linked_runtime_kind(&self) -> &'static str {
        match self.linked {
            PreparedIvpLinkedRuntime::Dense(_) => "dense",
            PreparedIvpLinkedRuntime::Sparse(_) => "sparse",
            PreparedIvpLinkedRuntime::Residual(_) => "residual",
        }
    }

    fn linked_problem_key(&self) -> &str {
        match &self.linked {
            PreparedIvpLinkedRuntime::Dense(linked) => linked.problem_key.as_str(),
            PreparedIvpLinkedRuntime::Sparse(linked) => linked.problem_key.as_str(),
            PreparedIvpLinkedRuntime::Residual(linked) => linked.problem_key.as_str(),
        }
    }

    fn validate_linked_layout(&self) -> Result<(), PreparedIvpRuntimeError> {
        fn validate_chunks(
            runtime: &'static str,
            stage: &'static str,
            expected: usize,
            chunks: impl Iterator<Item = (usize, usize)>,
        ) -> Result<(), PreparedIvpRuntimeError> {
            let mut cursor = 0usize;
            let mut count = 0usize;
            for (offset, len) in chunks {
                if offset != cursor {
                    return Err(PreparedIvpRuntimeError::InvalidLinkedLayout {
                        runtime,
                        message: format!(
                            "{stage} chunk {count} starts at {offset}, expected contiguous offset {cursor}"
                        ),
                    });
                }
                let end = offset.checked_add(len).ok_or_else(|| {
                    PreparedIvpRuntimeError::InvalidLinkedLayout {
                        runtime,
                        message: format!("{stage} chunk {count} range overflows"),
                    }
                })?;
                if end > expected {
                    return Err(PreparedIvpRuntimeError::InvalidLinkedLayout {
                        runtime,
                        message: format!(
                            "{stage} chunk {count} ends at {end}, beyond output length {expected}"
                        ),
                    });
                }
                cursor = end;
                count += 1;
            }
            if count > 0 && cursor != expected {
                return Err(PreparedIvpRuntimeError::InvalidLinkedLayout {
                    runtime,
                    message: format!(
                        "{stage} chunks cover [0, {cursor}), expected complete [0, {expected})"
                    ),
                });
            }
            Ok(())
        }

        match &self.linked {
            PreparedIvpLinkedRuntime::Dense(linked) => {
                if linked.residual_len != linked.shape.0 {
                    return Err(PreparedIvpRuntimeError::InvalidLinkedLayout {
                        runtime: "dense",
                        message: format!(
                            "residual length {} does not match Jacobian row count {}",
                            linked.residual_len, linked.shape.0
                        ),
                    });
                }
                let jacobian_len = linked.shape.0.checked_mul(linked.shape.1).ok_or_else(|| {
                    PreparedIvpRuntimeError::InvalidLinkedLayout {
                        runtime: "dense",
                        message: "Jacobian shape product overflows".to_string(),
                    }
                })?;
                validate_chunks(
                    "dense",
                    "residual",
                    linked.residual_len,
                    linked
                        .residual_chunks
                        .iter()
                        .map(|chunk| (chunk.output_offset, chunk.output_len)),
                )?;
                validate_chunks(
                    "dense",
                    "Jacobian",
                    jacobian_len,
                    linked
                        .jacobian_chunks
                        .iter()
                        .map(|chunk| (chunk.value_offset, chunk.value_len)),
                )
            }
            PreparedIvpLinkedRuntime::Sparse(linked) => {
                if linked.residual_len != linked.shape.0 {
                    return Err(PreparedIvpRuntimeError::InvalidLinkedLayout {
                        runtime: "sparse",
                        message: format!(
                            "residual length {} does not match Jacobian row count {}",
                            linked.residual_len, linked.shape.0
                        ),
                    });
                }
                let jacobian_len = linked.jacobian_output_len().map_err(|error| {
                    PreparedIvpRuntimeError::InvalidLinkedLayout {
                        runtime: "sparse",
                        message: error.to_string(),
                    }
                })?;
                if matches!(linked.jacobian_layout, LinkedJacobianLayout::ExplicitValues)
                    && jacobian_len != linked.nnz
                {
                    return Err(PreparedIvpRuntimeError::InvalidLinkedLayout {
                        runtime: "sparse",
                        message: format!(
                            "explicit Jacobian output length {jacobian_len} does not match nnz {}",
                            linked.nnz
                        ),
                    });
                }
                validate_chunks(
                    "sparse",
                    "residual",
                    linked.residual_len,
                    linked
                        .residual_chunks
                        .iter()
                        .map(|chunk| (chunk.output_offset, chunk.output_len)),
                )?;
                validate_chunks(
                    "sparse",
                    "Jacobian",
                    jacobian_len,
                    linked
                        .jacobian_value_chunks
                        .iter()
                        .map(|chunk| (chunk.value_offset, chunk.value_len)),
                )
            }
            PreparedIvpLinkedRuntime::Residual(linked) => validate_chunks(
                "residual",
                "residual",
                linked.residual_len,
                linked
                    .residual_chunks
                    .iter()
                    .map(|chunk| (chunk.output_offset, chunk.output_len)),
            ),
        }
    }

    /// Validates that the owner and its linked runtime still refer to the
    /// same artifact and that an attached resolver still sees a compiled
    /// publication. This is intentionally fallible so callers can reject a
    /// stale runtime before handing it to a solver.
    pub fn validate(&self) -> Result<(), PreparedIvpRuntimeError> {
        let actual = self.linked_problem_key();
        if actual != self.problem_key {
            return Err(PreparedIvpRuntimeError::LinkedProblemKeyMismatch {
                expected: self.problem_key.clone(),
                actual: actual.to_string(),
            });
        }
        self.validate_linked_layout()?;
        if let Some(resolver) = &self.resolver {
            let resolved = resolver.resolve_by_problem_key(&self.problem_key);
            if resolved.status == AotResolutionStatus::Missing {
                return Err(PreparedIvpRuntimeError::ArtifactMissing {
                    problem_key: self.problem_key.clone(),
                });
            }
            let registered = &resolved.registered;
            let manifest_key = registered.manifest_problem_key();
            if registered.problem_key != self.problem_key || registered.problem_key != manifest_key
            {
                return Err(PreparedIvpRuntimeError::ArtifactManifestMismatch {
                    problem_key: self.problem_key.clone(),
                    registered_key: registered.problem_key.clone(),
                    manifest_key,
                });
            }
            let inspection = registered.inspect_artifact();
            if matches!(
                inspection.state,
                AotArtifactState::Partial | AotArtifactState::Stale
            ) {
                return Err(PreparedIvpRuntimeError::ArtifactInvalidated {
                    problem_key: self.problem_key.clone(),
                    state: inspection.state,
                    detail: format!(
                        "marker_exists={}, static_output_exists={}, dynamic_output_exists={}",
                        inspection.marker_exists,
                        inspection.static_output_exists,
                        inspection.dynamic_output_exists
                    ),
                });
            }
            if resolved.status != AotResolutionStatus::Compiled {
                return Err(PreparedIvpRuntimeError::ArtifactNotReady {
                    problem_key: self.problem_key.clone(),
                    status: resolved.status,
                });
            }
        }
        Ok(())
    }
}

fn generated_names(
    problem_key: &str,
    config: &SymbolicIvpGeneratedBackendConfig,
) -> (String, String) {
    let suffix = generated_artifact_suffix(problem_key);
    let crate_name = config
        .crate_name_override
        .clone()
        .unwrap_or_else(|| format!("ivp_dense_{suffix}"));
    let module_name = config
        .module_name_override
        .clone()
        .unwrap_or_else(|| format!("ivp_dense_mod_{suffix}"));
    (crate_name, module_name)
}

fn generated_residual_names(
    problem_key: &str,
    config: &SymbolicIvpGeneratedBackendConfig,
) -> (String, String) {
    let suffix = generated_artifact_suffix(problem_key);
    let crate_name = config
        .crate_name_override
        .clone()
        .unwrap_or_else(|| format!("ivp_res_{suffix}"));
    let module_name = config
        .module_name_override
        .clone()
        .unwrap_or_else(|| format!("ivp_res_mod_{suffix}"));
    (crate_name, module_name)
}

fn generated_sparse_names(
    problem_key: &str,
    config: &SymbolicIvpGeneratedBackendConfig,
) -> (String, String) {
    let suffix = generated_artifact_suffix(problem_key);
    let crate_name = config
        .crate_name_override
        .clone()
        .unwrap_or_else(|| format!("ivp_sp_{suffix}"));
    let module_name = config
        .module_name_override
        .clone()
        .unwrap_or_else(|| format!("ivp_sp_mod_{suffix}"));
    (crate_name, module_name)
}

fn generated_artifact_suffix(problem_key: &str) -> String {
    problem_key
        .chars()
        .take(16)
        .collect::<String>()
        .replace('-', "_")
}

fn rebuild_isolated_output_parent_dir(base: &Path, route: &str, problem_key: &str) -> PathBuf {
    let short_key = problem_key
        .chars()
        .take(16)
        .collect::<String>()
        .replace('-', "_");
    let route = route.replace(|ch: char| !ch.is_ascii_alphanumeric(), "_");
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let seq = REBUILD_OUTPUT_COUNTER.fetch_add(1, Ordering::Relaxed);
    base.join(format!("rebuild_{route}_{short_key}_{pid}_{nanos}_{seq}"))
}

fn output_parent_dir_for_requested_build(
    config: &SymbolicIvpGeneratedBackendConfig,
    route: &str,
    problem_key: &str,
) -> Result<PathBuf, SymbolicIvpGeneratedError> {
    let base = config.output_parent_dir()?.to_path_buf();
    if matches!(
        config.build_policy,
        SymbolicIvpAotBuildPolicy::RebuildAlways { .. }
    ) {
        Ok(rebuild_isolated_output_parent_dir(
            &base,
            route,
            problem_key,
        ))
    } else {
        Ok(base)
    }
}

fn dense_problem_key(
    problem: &PreparedSymbolicIvpProblem,
    options: SymbolicIvpAotOptions,
) -> Result<String, SymbolicIvpGeneratedError> {
    if problem.native_atoms().is_some() {
        return Ok(
            prepared_atom_aot_problem_from_symbolic_ivp_problem_with_layout(
                problem,
                options,
                AtomAotMatrixLayout::Dense {
                    rows: problem.equations.len(),
                    cols: problem.variables.len(),
                },
            )?
            .problem_key(),
        );
    }
    Ok(problem.prepare_dense_aot_problem(options).problem_key())
}

fn select_backend(
    problem: &PreparedSymbolicIvpProblem,
    resolver: Option<&AotResolver>,
    options: SymbolicIvpAotOptions,
) -> Result<SelectedSymbolicIvpBackendKind, SymbolicIvpGeneratedError> {
    let problem_key = dense_problem_key(problem, options)?;
    if let Some(linked) = resolve_linked_dense_backend(problem_key.as_str()) {
        if linked.problem_key == problem_key {
            return Ok(SelectedSymbolicIvpBackendKind::AotCompiled);
        }
    }

    Ok(match resolver {
        Some(resolver) => match resolver.resolve_by_problem_key(problem_key.as_str()).status {
            AotResolutionStatus::Missing => SelectedSymbolicIvpBackendKind::AotMissing,
            AotResolutionStatus::RegisteredButNotBuilt => {
                SelectedSymbolicIvpBackendKind::AotRegisteredButNotBuilt
            }
            AotResolutionStatus::Compiled => SelectedSymbolicIvpBackendKind::AotCompiled,
        },
        None => SelectedSymbolicIvpBackendKind::AotMissing,
    })
}

fn select_residual_backend(
    problem: &PreparedSymbolicIvpResidualProblem,
    resolver: Option<&AotResolver>,
    options: SymbolicIvpAotOptions,
) -> SelectedSymbolicIvpBackendKind {
    let prepared = problem.prepare_residual_aot_problem(options);
    let problem_key = prepared.problem_key();
    if let Some(linked) = resolve_linked_residual_backend(problem_key.as_str()) {
        if linked.problem_key == problem_key {
            return SelectedSymbolicIvpBackendKind::AotCompiled;
        }
    }

    match resolver {
        Some(resolver) => match resolver.resolve_by_problem_key(problem_key.as_str()).status {
            AotResolutionStatus::Missing => SelectedSymbolicIvpBackendKind::AotMissing,
            AotResolutionStatus::RegisteredButNotBuilt => {
                SelectedSymbolicIvpBackendKind::AotRegisteredButNotBuilt
            }
            AotResolutionStatus::Compiled => SelectedSymbolicIvpBackendKind::AotCompiled,
        },
        None => SelectedSymbolicIvpBackendKind::AotMissing,
    }
}

fn select_sparse_backend(
    problem: &PreparedSparseProblem<'_>,
    resolver: Option<&AotResolver>,
) -> SelectedSymbolicIvpBackendKind {
    let problem_key =
        crate::symbolic::codegen::codegen_manifest::PreparedProblemManifest::from(problem)
            .problem_key();
    select_sparse_backend_by_key(problem_key.as_str(), resolver)
}

fn select_sparse_backend_by_key(
    problem_key: &str,
    resolver: Option<&AotResolver>,
) -> SelectedSymbolicIvpBackendKind {
    if let Some(linked) = resolve_linked_sparse_backend(problem_key) {
        if linked.problem_key == problem_key {
            return SelectedSymbolicIvpBackendKind::AotCompiled;
        }
    }

    match resolver {
        Some(resolver) => match resolver.resolve_by_problem_key(problem_key).status {
            AotResolutionStatus::Missing => SelectedSymbolicIvpBackendKind::AotMissing,
            AotResolutionStatus::RegisteredButNotBuilt => {
                SelectedSymbolicIvpBackendKind::AotRegisteredButNotBuilt
            }
            AotResolutionStatus::Compiled => SelectedSymbolicIvpBackendKind::AotCompiled,
        },
        None => SelectedSymbolicIvpBackendKind::AotMissing,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResolvedIvpAotBuildAction {
    /// Reuse an already linked/compiled artifact, or fall back to Lambdify
    /// when the policy allows it.
    ReuseOrFallback,
    /// Materialize/build only when the selected artifact is not compiled.
    BuildIfMissing,
    /// Build in an isolated output directory even when a compiled artifact
    /// is already available.
    RebuildAlways,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ResolvedIvpAotPlan {
    policy: SymbolicIvpAotBuildPolicy,
    initial_selection: SelectedSymbolicIvpBackendKind,
    action: ResolvedIvpAotBuildAction,
    profile: Option<AotBuildProfile>,
    preset: Option<AotBuildPreset>,
}

impl ResolvedIvpAotPlan {
    fn resolve(
        config: &SymbolicIvpGeneratedBackendConfig,
        initial_selection: SelectedSymbolicIvpBackendKind,
    ) -> Self {
        let (action, profile) = match config.build_policy {
            SymbolicIvpAotBuildPolicy::UseIfAvailable
            | SymbolicIvpAotBuildPolicy::RequirePrebuilt => {
                (ResolvedIvpAotBuildAction::ReuseOrFallback, None)
            }
            SymbolicIvpAotBuildPolicy::BuildIfMissing { profile } => {
                (ResolvedIvpAotBuildAction::BuildIfMissing, Some(profile))
            }
            SymbolicIvpAotBuildPolicy::RebuildAlways { profile } => {
                (ResolvedIvpAotBuildAction::RebuildAlways, Some(profile))
            }
        };
        let action = match action {
            ResolvedIvpAotBuildAction::BuildIfMissing
                if initial_selection == SelectedSymbolicIvpBackendKind::AotCompiled =>
            {
                ResolvedIvpAotBuildAction::ReuseOrFallback
            }
            other => other,
        };
        let preset = match profile {
            Some(AotBuildProfile::Debug) => Some(AotBuildPreset::DevFastest),
            Some(AotBuildProfile::Release) => Some(AotBuildPreset::Production),
            None => None,
        };
        Self {
            policy: config.build_policy,
            initial_selection,
            action,
            profile,
            preset,
        }
    }

    fn should_build(self) -> bool {
        matches!(
            self.action,
            ResolvedIvpAotBuildAction::BuildIfMissing | ResolvedIvpAotBuildAction::RebuildAlways
        )
    }

    fn profile(self) -> Option<AotBuildProfile> {
        self.profile
    }

    fn preset(self) -> Option<AotBuildPreset> {
        self.preset
    }
}

fn register_ivp_build_result_in_registry(
    resolver_snapshot: Option<AotResolver>,
    manifest: crate::symbolic::codegen::codegen_manifest::PreparedProblemManifest,
    build: &GeneratedAotBuildResult,
) -> Result<AotResolver, SymbolicIvpGeneratedError> {
    let mut registry = resolver_snapshot
        .as_ref()
        .map(|resolver| resolver.registry().clone())
        .unwrap_or_default();

    match build {
        GeneratedAotBuildResult::Rust(result) => {
            registry.register_materialized_build(manifest, result);
        }
        GeneratedAotBuildResult::C(result) => {
            register_c_build_in_registry(&mut registry, manifest, result);
        }
        GeneratedAotBuildResult::Zig(result) => {
            register_zig_build_in_registry(&mut registry, manifest, result);
        }
    }

    Ok(AotResolver::new(registry))
}

fn register_ivp_runtime_backend(
    route: &str,
    backend: AotCodegenBackend,
    resolver: &AotResolver,
    problem_key: &str,
) -> Result<(), SymbolicIvpGeneratedError> {
    let resolved = resolver.resolve_by_problem_key(problem_key);
    let artifact_path = resolved.registered.expected_cdylib.clone();
    match backend {
        AotCodegenBackend::Rust => register_generated_dense_cdylib_backend(&resolved.registered)
            .map(|_| ())
            .map_err(|err| {
                SymbolicIvpGeneratedError::AotLifecycle(runtime_registration_error_message(
                    route,
                    backend,
                    problem_key,
                    &artifact_path,
                    err,
                ))
            }),
        AotCodegenBackend::C => register_generated_c_dense_backend(&resolved.registered)
            .map(|_| ())
            .map_err(|err| {
                SymbolicIvpGeneratedError::AotLifecycle(runtime_registration_error_message(
                    route,
                    backend,
                    problem_key,
                    &artifact_path,
                    err,
                ))
            }),
        AotCodegenBackend::Zig => register_generated_zig_dense_backend(&resolved.registered)
            .map(|_| ())
            .map_err(|err| {
                SymbolicIvpGeneratedError::AotLifecycle(runtime_registration_error_message(
                    route,
                    backend,
                    problem_key,
                    &artifact_path,
                    err,
                ))
            }),
    }
}

fn register_ivp_residual_runtime_backend(
    route: &str,
    backend: AotCodegenBackend,
    resolver: &AotResolver,
    problem_key: &str,
) -> Result<(), SymbolicIvpGeneratedError> {
    let resolved = resolver.resolve_by_problem_key(problem_key);
    let artifact_path = resolved.registered.expected_cdylib.clone();
    match backend {
        AotCodegenBackend::Rust => register_generated_residual_cdylib_backend(&resolved.registered)
            .map(|_| ())
            .map_err(|err| {
                SymbolicIvpGeneratedError::AotLifecycle(runtime_registration_error_message(
                    route,
                    backend,
                    problem_key,
                    &artifact_path,
                    err,
                ))
            }),
        AotCodegenBackend::C => register_generated_c_residual_backend(&resolved.registered)
            .map(|_| ())
            .map_err(|err| {
                SymbolicIvpGeneratedError::AotLifecycle(runtime_registration_error_message(
                    route,
                    backend,
                    problem_key,
                    &artifact_path,
                    err,
                ))
            }),
        AotCodegenBackend::Zig => register_generated_zig_residual_backend(&resolved.registered)
            .map(|_| ())
            .map_err(|err| {
                SymbolicIvpGeneratedError::AotLifecycle(runtime_registration_error_message(
                    route,
                    backend,
                    problem_key,
                    &artifact_path,
                    err,
                ))
            }),
    }
}

fn register_ivp_sparse_runtime_backend(
    route: &str,
    backend: AotCodegenBackend,
    resolver: &AotResolver,
    problem_key: &str,
) -> Result<(), SymbolicIvpGeneratedError> {
    let resolved = resolver.resolve_by_problem_key(problem_key);
    let artifact_path = resolved.registered.expected_cdylib.clone();
    match backend {
        AotCodegenBackend::Rust => register_generated_sparse_cdylib_backend(&resolved.registered)
            .map(|_| ())
            .map_err(|err| {
                SymbolicIvpGeneratedError::AotLifecycle(runtime_registration_error_message(
                    route,
                    backend,
                    problem_key,
                    &artifact_path,
                    err,
                ))
            }),
        AotCodegenBackend::C => register_generated_c_sparse_backend(&resolved.registered)
            .map(|_| ())
            .map_err(|err| {
                SymbolicIvpGeneratedError::AotLifecycle(runtime_registration_error_message(
                    route,
                    backend,
                    problem_key,
                    &artifact_path,
                    err,
                ))
            }),
        AotCodegenBackend::Zig => register_generated_zig_sparse_backend(&resolved.registered)
            .map(|_| ())
            .map_err(|err| {
                SymbolicIvpGeneratedError::AotLifecycle(runtime_registration_error_message(
                    route,
                    backend,
                    problem_key,
                    &artifact_path,
                    err,
                ))
            }),
    }
}

fn register_ivp_banded_runtime_backend(
    route: &str,
    backend: AotCodegenBackend,
    resolver: &AotResolver,
    problem_key: &str,
) -> Result<(), SymbolicIvpGeneratedError> {
    let resolved = resolver.resolve_by_problem_key(problem_key);
    let artifact_path = resolved.registered.expected_cdylib.clone();
    let result = match backend {
        AotCodegenBackend::Rust => {
            register_generated_banded_cdylib_backend(&resolved.registered).map(|_| ())
        }
        AotCodegenBackend::C => {
            register_generated_c_banded_backend(&resolved.registered).map(|_| ())
        }
        AotCodegenBackend::Zig => {
            register_generated_zig_banded_backend(&resolved.registered).map(|_| ())
        }
    };
    result.map_err(|err| {
        SymbolicIvpGeneratedError::AotLifecycle(runtime_registration_error_message(
            route,
            backend,
            problem_key,
            &artifact_path,
            err,
        ))
    })
}

/// Reconnects an AtomView-native sparse or compact-Banded runtime from the
/// resolver when this process does not yet have a linked registry entry.
///
/// The resolver is the durable lifecycle boundary: it describes the generated
/// manifest and compiled output, while the linked registry is deliberately
/// process-local.  Keeping this reconnect step here makes `RequirePrebuilt`
/// behave the same after a process restart as it does after an in-process
/// build, without changing the callback ABI or creating language-specific
/// solver branches.
fn reconnect_ivp_native_sparse_runtime_backend(
    route: &str,
    backend: AotCodegenBackend,
    resolver: Option<&AotResolver>,
    problem_key: &str,
    layout: crate::symbolic::bvp::atom_aot::AtomAotMatrixLayout,
) -> Result<Option<LinkedSparseAotBackend>, SymbolicIvpGeneratedError> {
    if let Some(linked) = resolve_linked_sparse_backend(problem_key) {
        debug!(
            target: "rustedscithe::symbolic::aot",
            "linked sparse AOT runtime already available route={} key={}",
            route,
            problem_key
        );
        return Ok(Some(linked));
    }

    let Some(resolver) = resolver else {
        return Ok(None);
    };
    let resolved = resolver.resolve_by_problem_key(problem_key);
    if !resolved.is_compiled() {
        return Ok(None);
    }

    debug!(
        target: "rustedscithe::symbolic::aot",
        "reconnecting sparse AOT runtime route={} backend={:?} key={}",
        route,
        backend,
        problem_key
    );

    let result = if matches!(
        layout,
        crate::symbolic::bvp::atom_aot::AtomAotMatrixLayout::BandedCompact { .. }
    ) {
        register_ivp_banded_runtime_backend(route, backend, resolver, problem_key)
    } else {
        register_ivp_sparse_runtime_backend(route, backend, resolver, problem_key)
    };
    result?;
    Ok(resolve_linked_sparse_backend(problem_key))
}

/// Reconnects the residual callback from a compiled shared artifact when the
/// process-local residual registry is empty.
fn reconnect_ivp_native_residual_runtime_backend(
    route: &str,
    backend: AotCodegenBackend,
    resolver: Option<&AotResolver>,
    problem_key: &str,
) -> Result<
    Option<crate::symbolic::codegen::codegen_aot_runtime_link::LinkedResidualAotBackend>,
    SymbolicIvpGeneratedError,
> {
    if let Some(linked) = resolve_linked_residual_backend(problem_key) {
        debug!(
            target: "rustedscithe::symbolic::aot",
            "linked residual AOT runtime already available route={} key={}",
            route,
            problem_key
        );
        return Ok(Some(linked));
    }

    let Some(resolver) = resolver else {
        return Ok(None);
    };
    let resolved = resolver.resolve_by_problem_key(problem_key);
    if !resolved.is_compiled() {
        return Ok(None);
    }

    debug!(
        target: "rustedscithe::symbolic::aot",
        "reconnecting residual AOT runtime route={} backend={:?} key={}",
        route,
        backend,
        problem_key
    );

    register_ivp_residual_runtime_backend(route, backend, resolver, problem_key)?;
    Ok(resolve_linked_residual_backend(problem_key))
}

fn perform_requested_build(
    problem: &PreparedSymbolicIvpProblem,
    config: &SymbolicIvpGeneratedBackendConfig,
    resolver_snapshot: Option<AotResolver>,
) -> Result<(Option<GeneratedAotBuildResult>, Option<AotResolver>), SymbolicIvpGeneratedError> {
    let preset =
        match ResolvedIvpAotPlan::resolve(config, SelectedSymbolicIvpBackendKind::AotMissing)
            .preset()
        {
            Some(preset) => preset,
            None => return Ok((None, resolver_snapshot)),
        };

    let native_prepared = if problem.native_atoms().is_some() {
        Some(measure_cold_stage(
            &problem.telemetry,
            IvpColdStage::AtomJacobianPreparation,
            || {
                prepared_atom_aot_problem_from_symbolic_ivp_problem_with_layout(
                    problem,
                    config.aot_options,
                    AtomAotMatrixLayout::Dense {
                        rows: problem.equations.len(),
                        cols: problem.variables.len(),
                    },
                )
            },
        )?)
    } else {
        None
    };
    let (problem_key, manifest) = if let Some(prepared) = native_prepared.as_ref() {
        (prepared.problem_key(), prepared.manifest())
    } else {
        let prepared = problem.prepare_dense_aot_problem(config.aot_options);
        (prepared.problem_key(), prepared.manifest())
    };
    let route = if native_prepared.is_some() {
        "dense-atom-native"
    } else {
        "dense-expr-legacy"
    };
    problem.telemetry.log_aot_event(
        crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::Planned,
        route,
        problem_key.as_str(),
        "dense AOT build requested",
    );
    let (crate_name, module_name) = generated_names(&problem_key, config);
    info!(
        "Materializing symbolic IVP dense {:?} AOT build '{}' with preset {:?}",
        config.aot_codegen_backend, crate_name, preset
    );
    let artifact = measure_cold_stage(&problem.telemetry, IvpColdStage::AotLowering, || {
        if let Some(prepared) = native_prepared.as_ref() {
            generated_aot_artifact_from_symbolic_ivp_atom_problem(
                &crate_name,
                &module_name,
                prepared,
                config.aot_codegen_backend,
            )
        } else {
            try_generated_aot_artifact_from_symbolic_ivp_problem(
                &crate_name,
                &module_name,
                problem,
                config.aot_options,
                config.aot_codegen_backend,
            )
        }
    })?;
    let output_parent_dir =
        output_parent_dir_for_requested_build(config, "dense", problem_key.as_str())?;
    let mut request = measure_cold_stage(
        &problem.telemetry,
        IvpColdStage::AotSourceGeneration,
        || generated_aot_build_request_from_artifact(artifact, output_parent_dir, preset),
    );
    if let (GeneratedAotBuildRequest::C(c_request), Some(compiler)) =
        (&mut request, config.aot_c_compiler.as_ref())
    {
        let compile_config = match preset {
            AotBuildPreset::Production => CAotCompileConfig::production(),
            AotBuildPreset::FastBuild => CAotCompileConfig::fast_build(),
            AotBuildPreset::DevFastest => CAotCompileConfig::dev_fastest(),
        }
        .with_compiler(compiler.clone());
        *c_request = c_request.clone().with_compile_config(compile_config);
    }
    let build = measure_cold_stage(&problem.telemetry, IvpColdStage::AotMaterialization, || {
        request
            .materialize()
            .map_err(|err| materialization_error("dense", problem_key.as_str(), err))
    })?;
    problem.telemetry.log_aot_event(
        crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::Materialized,
        route,
        problem_key.as_str(),
        "dense source artifact materialized",
    );
    problem.telemetry.log_aot_event(
        crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::BuildStarted,
        route,
        problem_key.as_str(),
        "compiler build started",
    );

    let executed = measure_cold_stage(&problem.telemetry, IvpColdStage::AotBuild, || {
        execute_generated_build_with_retry(
            &build,
            &format!(
                "ivp-dense backend={:?} key={}",
                config.aot_codegen_backend, problem_key
            ),
            Some(&problem.telemetry),
            route,
            problem_key.as_str(),
        )
    });
    if let Err(error) = executed {
        problem.telemetry.log_aot_event(
            crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::BuildFailed,
            route,
            problem_key.as_str(),
            "compiler build failed",
        );
        return Err(error);
    }
    problem.telemetry.log_aot_event(
        crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::BuildSucceeded,
        route,
        problem_key.as_str(),
        "compiler build completed",
    );

    problem.telemetry.log_aot_event(
        crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::LinkStarted,
        route,
        problem_key.as_str(),
        "link/load started",
    );
    let resolver = measure_cold_stage(&problem.telemetry, IvpColdStage::AotLink, || {
        register_ivp_build_result_in_registry(resolver_snapshot, manifest, &build)
    })?;
    measure_cold_stage(&problem.telemetry, IvpColdStage::AotPublication, || {
        register_ivp_runtime_backend(
            "dense",
            config.aot_codegen_backend,
            &resolver,
            problem_key.as_str(),
        )
    })?;
    problem.telemetry.log_aot_event(
        crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::Linked,
        route,
        problem_key.as_str(),
        "linked runtime registered",
    );
    problem.telemetry.log_aot_event(
        crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::Published,
        route,
        problem_key.as_str(),
        "linked runtime published",
    );
    Ok((Some(build), Some(resolver)))
}

fn perform_requested_residual_build(
    problem: &PreparedSymbolicIvpResidualProblem,
    config: &SymbolicIvpGeneratedBackendConfig,
    resolver_snapshot: Option<AotResolver>,
) -> Result<(Option<GeneratedAotBuildResult>, Option<AotResolver>), SymbolicIvpGeneratedError> {
    let preset =
        match ResolvedIvpAotPlan::resolve(config, SelectedSymbolicIvpBackendKind::AotMissing)
            .preset()
        {
            Some(preset) => preset,
            None => return Ok((None, resolver_snapshot)),
        };

    let prepared = problem.prepare_residual_aot_problem(config.aot_options);
    let problem_key = prepared.problem_key();
    problem.telemetry.log_aot_event(
        crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::Planned,
        "residual-only",
        problem_key.as_str(),
        "residual AOT build requested",
    );
    let (crate_name, module_name) = generated_residual_names(&prepared.problem_key(), config);
    info!(
        "Materializing symbolic IVP residual-only {:?} AOT build '{}' with preset {:?}",
        config.aot_codegen_backend, crate_name, preset
    );
    let artifact = measure_cold_stage(&problem.telemetry, IvpColdStage::AotLowering, || {
        generated_aot_artifact_from_symbolic_ivp_residual_problem(
            &crate_name,
            &module_name,
            problem,
            config.aot_options,
            config.aot_codegen_backend,
        )
    });
    let output_parent_dir = output_parent_dir_for_requested_build(
        config,
        "residual-only",
        prepared.problem_key().as_str(),
    )?;
    let mut request = measure_cold_stage(
        &problem.telemetry,
        IvpColdStage::AotSourceGeneration,
        || generated_aot_build_request_from_artifact(artifact, output_parent_dir, preset),
    );
    if let (GeneratedAotBuildRequest::C(c_request), Some(compiler)) =
        (&mut request, config.aot_c_compiler.as_ref())
    {
        let compile_config = match preset {
            AotBuildPreset::Production => CAotCompileConfig::production(),
            AotBuildPreset::FastBuild => CAotCompileConfig::fast_build(),
            AotBuildPreset::DevFastest => CAotCompileConfig::dev_fastest(),
        }
        .with_compiler(compiler.clone());
        *c_request = c_request.clone().with_compile_config(compile_config);
    }
    let build = measure_cold_stage(&problem.telemetry, IvpColdStage::AotMaterialization, || {
        request
            .materialize()
            .map_err(|err| materialization_error("residual-only", problem_key.as_str(), err))
    })?;
    problem.telemetry.log_aot_event(
        crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::Materialized,
        "residual-only",
        problem_key.as_str(),
        "residual source artifact materialized",
    );
    problem.telemetry.log_aot_event(
        crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::BuildStarted,
        "residual-only",
        problem_key.as_str(),
        "compiler build started",
    );

    let executed = measure_cold_stage(&problem.telemetry, IvpColdStage::AotBuild, || {
        execute_generated_build_with_retry(
            &build,
            &format!(
                "ivp-residual backend={:?} key={}",
                config.aot_codegen_backend,
                prepared.problem_key()
            ),
            Some(&problem.telemetry),
            "residual-only",
            problem_key.as_str(),
        )
    });
    if let Err(error) = executed {
        problem.telemetry.log_aot_event(
            crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::BuildFailed,
            "residual-only",
            problem_key.as_str(),
            "compiler build failed",
        );
        return Err(error);
    }
    problem.telemetry.log_aot_event(
        crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::BuildSucceeded,
        "residual-only",
        problem_key.as_str(),
        "compiler build completed",
    );

    problem.telemetry.log_aot_event(
        crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::LinkStarted,
        "residual-only",
        problem_key.as_str(),
        "link/load started",
    );
    let resolver = measure_cold_stage(&problem.telemetry, IvpColdStage::AotLink, || {
        register_ivp_build_result_in_registry(resolver_snapshot, prepared.manifest(), &build)
    })?;
    measure_cold_stage(&problem.telemetry, IvpColdStage::AotPublication, || {
        register_ivp_residual_runtime_backend(
            "residual-only",
            config.aot_codegen_backend,
            &resolver,
            problem_key.as_str(),
        )
    })?;
    problem.telemetry.log_aot_event(
        crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::Linked,
        "residual-only",
        problem_key.as_str(),
        "linked runtime registered",
    );
    problem.telemetry.log_aot_event(
        crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::Published,
        "residual-only",
        problem_key.as_str(),
        "linked runtime published",
    );
    Ok((Some(build), Some(resolver)))
}

fn perform_requested_sparse_build(
    problem: &PreparedSparseProblem<'_>,
    config: &SymbolicIvpGeneratedBackendConfig,
    resolver_snapshot: Option<AotResolver>,
) -> Result<(Option<GeneratedAotBuildResult>, Option<AotResolver>), SymbolicIvpGeneratedError> {
    let preset =
        match ResolvedIvpAotPlan::resolve(config, SelectedSymbolicIvpBackendKind::AotMissing)
            .preset()
        {
            Some(preset) => preset,
            None => return Ok((None, resolver_snapshot)),
        };

    let manifest =
        crate::symbolic::codegen::codegen_manifest::PreparedProblemManifest::from(problem);
    let problem_key = manifest.problem_key();
    let (crate_name, module_name) = generated_sparse_names(&problem_key, config);
    info!(
        "Materializing symbolic IVP sparse {:?} AOT build '{}' with preset {:?}",
        config.aot_codegen_backend, crate_name, preset
    );
    let prepared_problem = PreparedProblem::sparse(problem.clone());
    let artifact = generated_aot_artifact_from_prepared_problem(
        &crate_name,
        &module_name,
        &prepared_problem,
        config.aot_codegen_backend,
    );
    let output_parent_dir =
        output_parent_dir_for_requested_build(config, "sparse", problem_key.as_str())?;
    let mut request =
        generated_aot_build_request_from_artifact(artifact, output_parent_dir, preset);
    if let (GeneratedAotBuildRequest::C(c_request), Some(compiler)) =
        (&mut request, config.aot_c_compiler.as_ref())
    {
        let compile_config = match preset {
            AotBuildPreset::Production => CAotCompileConfig::production(),
            AotBuildPreset::FastBuild => CAotCompileConfig::fast_build(),
            AotBuildPreset::DevFastest => CAotCompileConfig::dev_fastest(),
        }
        .with_compiler(compiler.clone());
        *c_request = c_request.clone().with_compile_config(compile_config);
    }
    let build = request
        .materialize()
        .map_err(|err| materialization_error("sparse-expr-legacy", problem_key.as_str(), err))?;

    execute_generated_build_with_retry(
        &build,
        &format!(
            "ivp-sparse backend={:?} key={}",
            config.aot_codegen_backend, problem_key
        ),
        None,
        "sparse-expr-legacy",
        problem_key.as_str(),
    )?;

    let resolver = register_ivp_build_result_in_registry(resolver_snapshot, manifest, &build)?;
    register_ivp_sparse_runtime_backend(
        "sparse",
        config.aot_codegen_backend,
        &resolver,
        problem_key.as_str(),
    )?;
    Ok((Some(build), Some(resolver)))
}

fn perform_requested_native_sparse_build(
    problem: &PreparedSymbolicIvpAtomAotProblem,
    config: &SymbolicIvpGeneratedBackendConfig,
    resolver_snapshot: Option<AotResolver>,
) -> Result<(Option<GeneratedAotBuildResult>, Option<AotResolver>), SymbolicIvpGeneratedError> {
    let preset =
        match ResolvedIvpAotPlan::resolve(config, SelectedSymbolicIvpBackendKind::AotMissing)
            .preset()
        {
            Some(preset) => preset,
            None => return Ok((None, resolver_snapshot)),
        };

    let manifest = problem.manifest();
    let problem_key = manifest.problem_key();
    if let Some(telemetry) = problem.telemetry() {
        telemetry.log_aot_event(
            crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::Planned,
            "sparse-atom-native",
            problem_key.as_str(),
            "native AOT build requested",
        );
    }
    let (crate_name, module_name) = generated_sparse_names(&problem_key, config);
    info!(
        "Materializing AtomView-native symbolic IVP sparse {:?} AOT build '{}' with preset {:?}",
        config.aot_codegen_backend, crate_name, preset
    );
    let artifact = if let Some(telemetry) = problem.telemetry() {
        measure_cold_stage(telemetry, IvpColdStage::AotLowering, || {
            generated_aot_artifact_from_symbolic_ivp_atom_problem(
                &crate_name,
                &module_name,
                problem,
                config.aot_codegen_backend,
            )
        })?
    } else {
        generated_aot_artifact_from_symbolic_ivp_atom_problem(
            &crate_name,
            &module_name,
            problem,
            config.aot_codegen_backend,
        )?
    };
    if let Some(telemetry) = problem.telemetry() {
        telemetry.log_aot_event(
            crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::Materialized,
            "sparse-atom-native",
            problem_key.as_str(),
            "native source artifact materialized",
        );
    }
    let output_parent_dir =
        output_parent_dir_for_requested_build(config, "sparse-atom-native", problem_key.as_str())?;
    let mut request = if let Some(telemetry) = problem.telemetry() {
        measure_cold_stage(telemetry, IvpColdStage::AotSourceGeneration, || {
            generated_aot_build_request_from_artifact(artifact, output_parent_dir, preset)
        })
    } else {
        generated_aot_build_request_from_artifact(artifact, output_parent_dir, preset)
    };
    if let (GeneratedAotBuildRequest::C(c_request), Some(compiler)) =
        (&mut request, config.aot_c_compiler.as_ref())
    {
        let compile_config = match preset {
            AotBuildPreset::Production => CAotCompileConfig::production(),
            AotBuildPreset::FastBuild => CAotCompileConfig::fast_build(),
            AotBuildPreset::DevFastest => CAotCompileConfig::dev_fastest(),
        }
        .with_compiler(compiler.clone());
        *c_request = c_request.clone().with_compile_config(compile_config);
    }
    let materialized = problem.telemetry().map(|telemetry| {
        measure_cold_stage(telemetry, IvpColdStage::AotMaterialization, || {
            request.materialize().map_err(|err| {
                materialization_error("sparse-atom-native", problem_key.as_str(), err)
            })
        })
    });
    let build = if let Some(materialized) = materialized {
        materialized
    } else {
        request
            .materialize()
            .map_err(|err| materialization_error("sparse-atom-native", problem_key.as_str(), err))
    };
    let build = match build {
        Ok(build) => build,
        Err(error) => {
            if let Some(telemetry) = problem.telemetry() {
                telemetry.log_aot_event(
                    crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::BuildFailed,
                    "sparse-atom-native",
                    problem_key.as_str(),
                    "artifact materialization failed",
                );
            }
            return Err(error);
        }
    };

    let build_started = problem.telemetry().and_then(|telemetry| {
        telemetry.log_aot_event(
            crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::BuildStarted,
            "sparse-atom-native",
            problem_key.as_str(),
            "compiler build started",
        );
        telemetry.start_cold_stage(IvpColdStage::AotBuild)
    });

    let executed = execute_generated_build_with_retry(
        &build,
        &format!(
            "ivp-sparse-atom-native backend={:?} key={}",
            config.aot_codegen_backend, problem_key
        ),
        problem.telemetry(),
        "sparse-atom-native",
        problem_key.as_str(),
    );
    if let Some(telemetry) = problem.telemetry() {
        telemetry.record_cold_stage(IvpColdStage::AotBuild, build_started);
    }
    match executed {
        Ok(_) => {
            if let Some(telemetry) = problem.telemetry() {
                telemetry.log_aot_event(
                    crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::BuildSucceeded,
                    "sparse-atom-native",
                    problem_key.as_str(),
                    "compiler build completed",
                );
            }
        }
        Err(error) => {
            if let Some(telemetry) = problem.telemetry() {
                telemetry.log_aot_event(
                    crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::BuildFailed,
                    "sparse-atom-native",
                    problem_key.as_str(),
                    "compiler build failed",
                );
            }
            return Err(error.into());
        }
    }

    let link_started = problem.telemetry().and_then(|telemetry| {
        telemetry.log_aot_event(
            crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::LinkStarted,
            "sparse-atom-native",
            problem_key.as_str(),
            "link/load started",
        );
        telemetry.start_cold_stage(crate::symbolic::ivp_telemetry::IvpColdStage::AotLink)
    });
    if let Some(telemetry) = problem.telemetry() {
        telemetry.record_aot_link_attempt();
    }
    let resolver = register_ivp_build_result_in_registry(resolver_snapshot, manifest, &build);
    let resolver = match resolver {
        Ok(resolver) => resolver,
        Err(error) => {
            if let Some(telemetry) = problem.telemetry() {
                telemetry.record_cold_stage(
                    crate::symbolic::ivp_telemetry::IvpColdStage::AotLink,
                    link_started,
                );
                telemetry.record_aot_link_result(false);
                telemetry.log_aot_event(
                    crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::LinkFailed,
                    "sparse-atom-native",
                    problem_key.as_str(),
                    "artifact registration failed",
                );
            }
            return Err(error);
        }
    };
    let is_compact_banded = matches!(
        problem.plan().matrix_layout(),
        crate::symbolic::bvp::atom_aot::AtomAotMatrixLayout::BandedCompact { .. }
    );
    let publish = || {
        let linked = if is_compact_banded {
            register_ivp_banded_runtime_backend(
                "banded-atom-native",
                config.aot_codegen_backend,
                &resolver,
                problem_key.as_str(),
            )
        } else {
            register_ivp_sparse_runtime_backend(
                "sparse-atom-native",
                config.aot_codegen_backend,
                &resolver,
                problem_key.as_str(),
            )
        };
        // The Atom-native artifact carries residual and Jacobian symbols in one
        // manifest. Publish both callbacks from that one load so a solver does
        // not rebuild a second residual-only artifact before its Jacobian setup.
        linked.and_then(|()| {
            register_ivp_residual_runtime_backend(
                "residual-atom-native",
                config.aot_codegen_backend,
                &resolver,
                problem_key.as_str(),
            )
        })
    };
    let linked =
        measure_optional_cold_stage(problem.telemetry(), IvpColdStage::AotPublication, publish);
    if let Some(telemetry) = problem.telemetry() {
        telemetry.record_cold_stage(
            crate::symbolic::ivp_telemetry::IvpColdStage::AotLink,
            link_started,
        );
        telemetry.record_aot_link_result(linked.is_ok());
        if linked.is_ok() {
            telemetry.log_aot_event(
                crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::Linked,
                "sparse-atom-native",
                problem_key.as_str(),
                "linked callbacks registered",
            );
        }
        telemetry.log_aot_event(
            if linked.is_ok() {
                crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::Published
            } else {
                crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::LinkFailed
            },
            "sparse-atom-native",
            problem_key.as_str(),
            if linked.is_ok() {
                "linked callbacks published"
            } else {
                "linked callback publication failed"
            },
        );
    }
    linked?;
    Ok((Some(build), Some(resolver)))
}

fn prepare_generated_symbolic_ivp_native_sparse_backend(
    baseline_problem: &PreparedSymbolicIvpResidualProblem,
    config: &SymbolicIvpGeneratedBackendConfig,
) -> Result<PreparedGeneratedSymbolicIvpSparseBackend, SymbolicIvpGeneratedError> {
    let rows = baseline_problem.equations.len();
    let cols = baseline_problem.variables.len();
    prepare_generated_symbolic_ivp_native_backend(
        baseline_problem,
        config,
        crate::symbolic::bvp::atom_aot::AtomAotMatrixLayout::SparseCsc { rows, cols, nnz: 0 },
    )
}

fn prepare_generated_symbolic_ivp_native_banded_backend(
    baseline_problem: &PreparedSymbolicIvpResidualProblem,
    config: &SymbolicIvpGeneratedBackendConfig,
    kl: usize,
    ku: usize,
) -> Result<PreparedGeneratedSymbolicIvpSparseBackend, SymbolicIvpGeneratedError> {
    let rows = baseline_problem.equations.len();
    let cols = baseline_problem.variables.len();
    prepare_generated_symbolic_ivp_native_backend(
        baseline_problem,
        config,
        crate::symbolic::bvp::atom_aot::AtomAotMatrixLayout::BandedCompact {
            rows,
            cols,
            kl,
            ku,
            slots: 0,
        },
    )
}

fn prepare_generated_symbolic_ivp_native_backend(
    baseline_problem: &PreparedSymbolicIvpResidualProblem,
    config: &SymbolicIvpGeneratedBackendConfig,
    requested_layout: crate::symbolic::bvp::atom_aot::AtomAotMatrixLayout,
) -> Result<PreparedGeneratedSymbolicIvpSparseBackend, SymbolicIvpGeneratedError> {
    let native = measure_cold_stage(
        &baseline_problem.telemetry,
        IvpColdStage::AtomJacobianPreparation,
        || {
            prepared_atom_aot_problem_from_residual_problem(
                baseline_problem,
                config.residual_chunking_strategy,
                config.sparse_jacobian_chunking_strategy,
                requested_layout,
            )
        },
    )
    .map_err(SymbolicIvpGeneratedError::from)?;
    let problem_key = native.problem_key();
    let initial_selection = measure_cold_stage(
        &baseline_problem.telemetry,
        IvpColdStage::AotCacheLookup,
        || select_sparse_backend_by_key(&problem_key, config.resolver.as_ref()),
    );
    log_aot_cache_selection(
        &baseline_problem.telemetry,
        "sparse-atom-native",
        &problem_key,
        "phase=initial-selection",
        initial_selection,
    );
    baseline_problem
        .telemetry
        .record_aot_resolution(initial_selection == SelectedSymbolicIvpBackendKind::AotCompiled);
    let resolved_plan = ResolvedIvpAotPlan::resolve(config, initial_selection);
    let (build_result, resolver_snapshot) = if resolved_plan.should_build() {
        perform_requested_native_sparse_build(&native, config, config.resolver.clone())?
    } else {
        (None, config.resolver.clone())
    };
    let final_selection = measure_cold_stage(
        &baseline_problem.telemetry,
        IvpColdStage::AotCacheLookup,
        || select_sparse_backend_by_key(&problem_key, resolver_snapshot.as_ref()),
    );
    log_aot_cache_selection(
        &baseline_problem.telemetry,
        "sparse-atom-native",
        &problem_key,
        "phase=final-selection",
        final_selection,
    );
    baseline_problem
        .telemetry
        .record_aot_resolution(final_selection == SelectedSymbolicIvpBackendKind::AotCompiled);
    let layout = native.plan().matrix_layout();
    let (rows, cols) = layout.shape();
    let jacobian_structure = SparseJacobianStructure {
        rows,
        cols,
        row_indices: native
            .plan()
            .jacobian_entries()
            .iter()
            .map(|entry| entry.row)
            .collect(),
        col_indices: native
            .plan()
            .jacobian_entries()
            .iter()
            .map(|entry| entry.col)
            .collect(),
    };

    match final_selection {
        SelectedSymbolicIvpBackendKind::AotCompiled => {
            let had_process_local_runtime = resolve_linked_sparse_backend(&problem_key).is_some();
            let linked_backend = reconnect_ivp_native_sparse_runtime_backend(
                "sparse-atom-native",
                config.aot_codegen_backend,
                resolver_snapshot.as_ref(),
                &problem_key,
                *layout,
            )?;
            if linked_backend.is_none() {
                return Err(SymbolicIvpGeneratedError::CompiledAotRuntimeUnavailable(
                    runtime_unavailable_aot_message("sparse-atom-native", &problem_key, config),
                ));
            }
            if !had_process_local_runtime {
                baseline_problem.telemetry.record_aot_reconnect();
                baseline_problem.telemetry.log_aot_event(
                    crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::Linked,
                    "sparse-atom-native",
                    &problem_key,
                    "compiled runtime reconnected",
                );
            }
            baseline_problem.telemetry.record_aot_runtime_ready();
            baseline_problem.telemetry.log_aot_event(
                crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::RuntimeReady,
                "sparse-atom-native",
                &problem_key,
                "runtime ready for solver handoff",
            );
            let runtime_owner = linked_backend.clone().map(|linked| {
                PreparedIvpAotRuntime::new(
                    problem_key.clone(),
                    final_selection,
                    resolver_snapshot.clone(),
                    build_result.clone(),
                    PreparedIvpLinkedRuntime::Sparse(linked),
                )
            });
            Ok(PreparedGeneratedSymbolicIvpSparseBackend {
                problem_key,
                selected_backend: final_selection,
                linked_backend,
                jacobian_structure,
                telemetry: baseline_problem.telemetry.clone(),
                updated_resolver: resolver_snapshot,
                build_result,
                runtime_owner,
            })
        }
        SelectedSymbolicIvpBackendKind::AotRegisteredButNotBuilt => {
            Err(SymbolicIvpGeneratedError::CompiledAotArtifactNotBuilt(
                not_built_aot_message("sparse-atom-native", &problem_key, config),
            ))
        }
        SelectedSymbolicIvpBackendKind::AotMissing | SelectedSymbolicIvpBackendKind::Lambdify => {
            if matches!(
                config.build_policy,
                SymbolicIvpAotBuildPolicy::RequirePrebuilt
            ) {
                Err(SymbolicIvpGeneratedError::CompiledAotArtifactMissing(
                    missing_aot_message("sparse-atom-native", &problem_key, config),
                ))
            } else {
                Ok(PreparedGeneratedSymbolicIvpSparseBackend {
                    problem_key,
                    selected_backend: SelectedSymbolicIvpBackendKind::Lambdify,
                    linked_backend: None,
                    jacobian_structure,
                    telemetry: baseline_problem.telemetry.clone(),
                    updated_resolver: resolver_snapshot,
                    build_result,
                    runtime_owner: None,
                })
            }
        }
    }
}

/// Builds one sparse-IVP generated backend (residual + sparse Jacobian values)
/// through the high-level lifecycle.
///
/// This path is intended for solvers like LSODE2 that want compiled Jacobian
/// value callbacks for sparse/banded Newton systems.
pub fn prepare_generated_symbolic_ivp_sparse_backend(
    equations: Vec<crate::symbolic::symbolic_engine::Expr>,
    variables: Vec<String>,
    time_arg: String,
    options: SymbolicIvpProblemOptions,
    config: SymbolicIvpGeneratedBackendConfig,
) -> Result<PreparedGeneratedSymbolicIvpSparseBackend, SymbolicIvpGeneratedError> {
    let use_atom_native_aot = options.symbolic_assembly_backend
        == crate::symbolic::symbolic_ivp::IvpSymbolicAssemblyBackend::AtomView;
    if use_atom_native_aot {
        let baseline_problem =
            prepare_symbolic_ivp_residual_problem(equations, variables, time_arg, options)?;
        return prepare_generated_symbolic_ivp_native_sparse_backend(&baseline_problem, &config);
    }

    let baseline_problem = prepare_symbolic_ivp_problem(equations, variables, time_arg, options)?;
    let aot_source = &baseline_problem;
    let variable_refs = aot_source
        .variables
        .iter()
        .map(|value| value.as_str())
        .collect::<Vec<_>>();
    let parameter_refs = aot_source.equation_parameters.as_ref().map(|parameters| {
        parameters
            .iter()
            .map(|value| value.as_str())
            .collect::<Vec<_>>()
    });
    let mut sparse_param_refs = Vec::with_capacity(1 + parameter_refs.as_ref().map_or(0, Vec::len));
    sparse_param_refs.push(aot_source.time_arg.as_str());
    if let Some(params) = parameter_refs.as_ref() {
        sparse_param_refs.extend(params.iter().copied());
    }

    let sparse_entries = aot_source
        .symbolic_jacobian
        .iter()
        .enumerate()
        .flat_map(|(row, jac_row)| {
            jac_row.iter().enumerate().filter_map(move |(col, expr)| {
                if expr.is_zero() {
                    None
                } else {
                    Some(SparseExprEntry { row, col, expr })
                }
            })
        })
        .collect::<Vec<_>>();

    let residual_plan = IvpResidualTask {
        fn_name: "generated_ivp_residual_eval",
        time_arg: aot_source.time_arg.as_str(),
        residuals: &aot_source.equations,
        variables: &variable_refs,
        params: parameter_refs.as_deref(),
    }
    .runtime_plan(config.aot_options.residual_strategy);

    let shape = (
        aot_source.symbolic_jacobian.len(),
        aot_source
            .symbolic_jacobian
            .first()
            .map_or(0, |row| row.len()),
    );

    let sparse_plan = SparseJacobianTask {
        fn_name: "generated_ivp_jacobian_values_eval",
        shape,
        entries: &sparse_entries,
        variables: &variable_refs,
        params: Some(sparse_param_refs.as_slice()),
    }
    .runtime_plan(config.sparse_jacobian_chunking_strategy);
    let jacobian_structure = sparse_plan.structure.clone();

    let prepared_sparse = PreparedSparseProblem::new(
        BackendKind::Aot,
        MatrixBackend::SparseCol,
        residual_plan,
        sparse_plan,
    );

    let initial_selection =
        measure_cold_stage(&aot_source.telemetry, IvpColdStage::AotCacheLookup, || {
            select_sparse_backend(&prepared_sparse, config.resolver.as_ref())
        });
    let resolved_plan = ResolvedIvpAotPlan::resolve(&config, initial_selection);
    let (build_result, resolver_snapshot) = if resolved_plan.should_build() {
        perform_requested_sparse_build(&prepared_sparse, &config, config.resolver.clone())?
    } else {
        (None, config.resolver.clone())
    };

    let final_selection =
        measure_cold_stage(&aot_source.telemetry, IvpColdStage::AotCacheLookup, || {
            select_sparse_backend(&prepared_sparse, resolver_snapshot.as_ref())
        });
    let problem_key =
        crate::symbolic::codegen::codegen_manifest::PreparedProblemManifest::from(&prepared_sparse)
            .problem_key();

    let mut linked_backend = resolve_linked_sparse_backend(problem_key.as_str());
    if linked_backend.is_none() {
        if let Some(resolver) = resolver_snapshot.as_ref() {
            let _ = register_ivp_sparse_runtime_backend(
                "sparse",
                config.aot_codegen_backend,
                resolver,
                problem_key.as_str(),
            );
            linked_backend = resolve_linked_sparse_backend(problem_key.as_str());
        }
    }

    match final_selection {
        SelectedSymbolicIvpBackendKind::AotCompiled => {
            if linked_backend.is_none() {
                return Err(SymbolicIvpGeneratedError::CompiledAotRuntimeUnavailable(
                    runtime_unavailable_aot_message("sparse", problem_key.as_str(), &config),
                ));
            }
            let runtime_owner = linked_backend.clone().map(|linked| {
                PreparedIvpAotRuntime::new(
                    problem_key.clone(),
                    final_selection,
                    resolver_snapshot.clone(),
                    build_result.clone(),
                    PreparedIvpLinkedRuntime::Sparse(linked),
                )
            });
            Ok(PreparedGeneratedSymbolicIvpSparseBackend {
                problem_key,
                selected_backend: final_selection,
                linked_backend,
                jacobian_structure,
                telemetry: aot_source.telemetry.clone(),
                updated_resolver: resolver_snapshot,
                build_result,
                runtime_owner,
            })
        }
        SelectedSymbolicIvpBackendKind::AotRegisteredButNotBuilt => {
            Err(SymbolicIvpGeneratedError::CompiledAotArtifactNotBuilt(
                not_built_aot_message("sparse", problem_key.as_str(), &config),
            ))
        }
        SelectedSymbolicIvpBackendKind::AotMissing | SelectedSymbolicIvpBackendKind::Lambdify => {
            match config.build_policy {
                SymbolicIvpAotBuildPolicy::RequirePrebuilt => {
                    Err(SymbolicIvpGeneratedError::CompiledAotArtifactMissing(
                        missing_aot_message("sparse", problem_key.as_str(), &config),
                    ))
                }
                _ => Ok(PreparedGeneratedSymbolicIvpSparseBackend {
                    problem_key,
                    selected_backend: SelectedSymbolicIvpBackendKind::Lambdify,
                    linked_backend: None,
                    jacobian_structure,
                    telemetry: aot_source.telemetry.clone(),
                    updated_resolver: resolver_snapshot,
                    build_result,
                    runtime_owner: None,
                }),
            }
        }
    }
}

/// Builds an AtomView-native compact-Banded IVP backend.
///
/// The bandwidth is part of the public linear-system contract. Requiring it
/// here avoids a second symbolic differentiation pass merely to infer layout;
/// callers that do not know the bandwidth should continue using the sparse
/// route until a separate inference phase is explicitly requested.
pub fn prepare_generated_symbolic_ivp_banded_backend(
    equations: Vec<crate::symbolic::symbolic_engine::Expr>,
    variables: Vec<String>,
    time_arg: String,
    bandwidth: (usize, usize),
    options: SymbolicIvpProblemOptions,
    config: SymbolicIvpGeneratedBackendConfig,
) -> Result<PreparedGeneratedSymbolicIvpSparseBackend, SymbolicIvpGeneratedError> {
    if options.symbolic_assembly_backend
        != crate::symbolic::symbolic_ivp::IvpSymbolicAssemblyBackend::AtomView
    {
        return Err(SymbolicIvpGeneratedError::AotBuildFailed(
            "compact-Banded generated IVP backend requires AtomView assembly".to_string(),
        ));
    }
    let baseline_problem =
        prepare_symbolic_ivp_residual_problem(equations, variables, time_arg, options)?;
    prepare_generated_symbolic_ivp_native_banded_backend(
        &baseline_problem,
        &config,
        bandwidth.0,
        bandwidth.1,
    )
}

/// Builds one shared IVP symbolic problem through the high-level generated-backend layer.
pub fn prepare_generated_symbolic_ivp_problem(
    equations: Vec<crate::symbolic::symbolic_engine::Expr>,
    variables: Vec<String>,
    time_arg: String,
    options: SymbolicIvpProblemOptions,
    config: SymbolicIvpGeneratedBackendConfig,
) -> Result<PreparedGeneratedSymbolicIvpProblem, SymbolicIvpGeneratedError> {
    let baseline_problem = prepare_symbolic_ivp_problem(equations, variables, time_arg, options)?;
    let aot_source = &baseline_problem;
    let initial_selection = measure_cold_stage(
        &baseline_problem.telemetry,
        IvpColdStage::AotCacheLookup,
        || select_backend(aot_source, config.resolver.as_ref(), config.aot_options),
    )?;
    let problem_key = dense_problem_key(aot_source, config.aot_options)?;
    log_aot_cache_selection(
        &baseline_problem.telemetry,
        if aot_source.native_atoms().is_some() {
            "dense-atom-native"
        } else {
            "dense-expr-legacy"
        },
        problem_key.as_str(),
        "phase=initial-selection",
        initial_selection,
    );
    baseline_problem
        .telemetry
        .record_aot_resolution(initial_selection == SelectedSymbolicIvpBackendKind::AotCompiled);
    let resolved_plan = ResolvedIvpAotPlan::resolve(&config, initial_selection);
    let (build_result, resolver_snapshot) = if resolved_plan.should_build() {
        perform_requested_build(aot_source, &config, config.resolver.clone())?
    } else {
        (None, config.resolver.clone())
    };

    let final_selection = measure_cold_stage(
        &baseline_problem.telemetry,
        IvpColdStage::AotCacheLookup,
        || select_backend(aot_source, resolver_snapshot.as_ref(), config.aot_options),
    )?;
    log_aot_cache_selection(
        &baseline_problem.telemetry,
        if aot_source.native_atoms().is_some() {
            "dense-atom-native"
        } else {
            "dense-expr-legacy"
        },
        problem_key.as_str(),
        "phase=final-selection",
        final_selection,
    );
    baseline_problem
        .telemetry
        .record_aot_resolution(final_selection == SelectedSymbolicIvpBackendKind::AotCompiled);
    match final_selection {
        SelectedSymbolicIvpBackendKind::Lambdify | SelectedSymbolicIvpBackendKind::AotMissing => {
            match config.build_policy {
                SymbolicIvpAotBuildPolicy::RequirePrebuilt => {
                    Err(SymbolicIvpGeneratedError::CompiledAotArtifactMissing(
                        missing_aot_message("dense", problem_key.as_str(), &config),
                    ))
                }
                _ => Ok(PreparedGeneratedSymbolicIvpProblem {
                    problem: baseline_problem,
                    selected_backend: SelectedSymbolicIvpBackendKind::Lambdify,
                    updated_resolver: resolver_snapshot,
                    build_result,
                    runtime_owner: None,
                }),
            }
        }
        SelectedSymbolicIvpBackendKind::AotRegisteredButNotBuilt => {
            Err(SymbolicIvpGeneratedError::CompiledAotArtifactNotBuilt(
                not_built_aot_message("dense", problem_key.as_str(), &config),
            ))
        }
        SelectedSymbolicIvpBackendKind::AotCompiled => {
            let problem_key = dense_problem_key(aot_source, config.aot_options)?;
            baseline_problem.telemetry.record_aot_link_attempt();
            if let Some(linked) = resolve_linked_dense_backend(problem_key.as_str()) {
                let problem = baseline_problem;
                problem.telemetry.record_aot_link_result(true);
                problem.telemetry.record_aot_runtime_ready();
                problem.telemetry.log_aot_event(
                    crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::RuntimeReady,
                    if problem.native_atoms().is_some() {
                        "dense-atom-native"
                    } else {
                        "dense-expr-legacy"
                    },
                    problem_key.as_str(),
                    "runtime ready for solver handoff",
                );
                let runtime_owner = PreparedIvpAotRuntime::new(
                    problem_key,
                    SelectedSymbolicIvpBackendKind::AotCompiled,
                    resolver_snapshot.clone(),
                    build_result.clone(),
                    PreparedIvpLinkedRuntime::Dense(linked.clone()),
                );
                Ok(PreparedGeneratedSymbolicIvpProblem {
                    problem: problem.into_linked_dense_backend(linked),
                    selected_backend: SelectedSymbolicIvpBackendKind::AotCompiled,
                    updated_resolver: resolver_snapshot,
                    build_result,
                    runtime_owner: Some(runtime_owner),
                })
            } else {
                match config.build_policy {
                    SymbolicIvpAotBuildPolicy::RequirePrebuilt => {
                        Err(SymbolicIvpGeneratedError::CompiledAotRuntimeUnavailable(
                            runtime_unavailable_aot_message("dense", problem_key.as_str(), &config),
                        ))
                    }
                    _ => {
                        warn!(
                            "Symbolic IVP compiled AOT artifact exists but no linked runtime is registered; falling back to lambdify"
                        );
                        Ok(PreparedGeneratedSymbolicIvpProblem {
                            problem: baseline_problem,
                            selected_backend: SelectedSymbolicIvpBackendKind::Lambdify,
                            updated_resolver: resolver_snapshot,
                            build_result,
                            runtime_owner: None,
                        })
                    }
                }
            }
        }
    }
}

/// Builds one residual-only IVP symbolic problem through the high-level
/// generated-backend layer.
///
/// This is the LSODE2 native sparse/banded path: residuals may come from
/// Lambdify or AOT, while Jacobian storage is supplied by the native symbolic
/// sparse/banded callback installed by the solver.
pub fn prepare_generated_symbolic_ivp_residual_problem(
    equations: Vec<crate::symbolic::symbolic_engine::Expr>,
    variables: Vec<String>,
    time_arg: String,
    options: SymbolicIvpProblemOptions,
    config: SymbolicIvpGeneratedBackendConfig,
) -> Result<PreparedGeneratedSymbolicIvpResidualProblem, SymbolicIvpGeneratedError> {
    prepare_generated_symbolic_ivp_residual_problem_with_layout(
        equations, variables, time_arg, options, config, None,
    )
}

/// Builds an AtomView-native residual AOT problem using the same compact
/// Banded layout that the solver will later request for its Jacobian.
///
/// Keeping this layout in the residual preparation call prevents one solve
/// from materializing a sparse artifact first and then a second compact-Banded
/// artifact during Jacobian preparation.
pub fn prepare_generated_symbolic_ivp_banded_residual_problem(
    equations: Vec<crate::symbolic::symbolic_engine::Expr>,
    variables: Vec<String>,
    time_arg: String,
    bandwidth: (usize, usize),
    options: SymbolicIvpProblemOptions,
    config: SymbolicIvpGeneratedBackendConfig,
) -> Result<PreparedGeneratedSymbolicIvpResidualProblem, SymbolicIvpGeneratedError> {
    if options.symbolic_assembly_backend
        != crate::symbolic::symbolic_ivp::IvpSymbolicAssemblyBackend::AtomView
    {
        return Err(SymbolicIvpGeneratedError::AotBuildFailed(
            "compact-Banded generated IVP residual backend requires AtomView assembly".to_string(),
        ));
    }
    prepare_generated_symbolic_ivp_residual_problem_with_layout(
        equations,
        variables,
        time_arg,
        options,
        config,
        Some(bandwidth),
    )
}

fn prepare_generated_symbolic_ivp_residual_problem_with_layout(
    equations: Vec<crate::symbolic::symbolic_engine::Expr>,
    variables: Vec<String>,
    time_arg: String,
    options: SymbolicIvpProblemOptions,
    config: SymbolicIvpGeneratedBackendConfig,
    native_banded_layout: Option<(usize, usize)>,
) -> Result<PreparedGeneratedSymbolicIvpResidualProblem, SymbolicIvpGeneratedError> {
    let is_atom_view = options.symbolic_assembly_backend
        == crate::symbolic::symbolic_ivp::IvpSymbolicAssemblyBackend::AtomView;
    let baseline_problem =
        prepare_symbolic_ivp_residual_problem(equations, variables, time_arg, options)?;

    // AOT AtomView residuals and Jacobians must share one generated artifact.
    // The ordinary UseIfAvailable path remains on the prepared Lambdify
    // residual and must not pay for a Jacobian/layout preparation it will not
    // consume.
    if is_atom_view
        && !matches!(
            config.build_policy,
            SymbolicIvpAotBuildPolicy::UseIfAvailable
        )
    {
        let native = match native_banded_layout {
            Some((kl, ku)) => prepare_generated_symbolic_ivp_native_banded_backend(
                &baseline_problem,
                &config,
                kl,
                ku,
            )?,
            None => {
                prepare_generated_symbolic_ivp_native_sparse_backend(&baseline_problem, &config)?
            }
        };
        match native.selected_backend {
            SelectedSymbolicIvpBackendKind::AotCompiled => {
                let linked = reconnect_ivp_native_residual_runtime_backend(
                    "residual-atom-native",
                    config.aot_codegen_backend,
                    native.updated_resolver.as_ref(),
                    native.problem_key.as_str(),
                )?
                .ok_or_else(|| {
                    SymbolicIvpGeneratedError::CompiledAotRuntimeUnavailable(
                        runtime_unavailable_aot_message(
                            "residual-atom-native",
                            native.problem_key.as_str(),
                            &config,
                        ),
                    )
                })?;
                let runtime_owner = PreparedIvpAotRuntime::new(
                    native.problem_key.clone(),
                    SelectedSymbolicIvpBackendKind::AotCompiled,
                    native.updated_resolver.clone(),
                    native.build_result.clone(),
                    PreparedIvpLinkedRuntime::Residual(linked.clone()),
                );
                return Ok(PreparedGeneratedSymbolicIvpResidualProblem {
                    problem: baseline_problem.into_linked_residual_backend(linked),
                    selected_backend: SelectedSymbolicIvpBackendKind::AotCompiled,
                    updated_resolver: native.updated_resolver,
                    build_result: native.build_result,
                    runtime_owner: Some(runtime_owner),
                });
            }
            SelectedSymbolicIvpBackendKind::AotRegisteredButNotBuilt => {
                return Err(SymbolicIvpGeneratedError::CompiledAotArtifactNotBuilt(
                    not_built_aot_message("residual-atom-native", &native.problem_key, &config),
                ));
            }
            SelectedSymbolicIvpBackendKind::AotMissing
            | SelectedSymbolicIvpBackendKind::Lambdify => {
                if matches!(
                    config.build_policy,
                    SymbolicIvpAotBuildPolicy::RequirePrebuilt
                ) {
                    return Err(SymbolicIvpGeneratedError::CompiledAotArtifactMissing(
                        missing_aot_message("residual-atom-native", &native.problem_key, &config),
                    ));
                }
                return Ok(PreparedGeneratedSymbolicIvpResidualProblem {
                    problem: baseline_problem,
                    selected_backend: SelectedSymbolicIvpBackendKind::Lambdify,
                    updated_resolver: native.updated_resolver,
                    build_result: native.build_result,
                    runtime_owner: None,
                });
            }
        }
    }

    let initial_selection = measure_cold_stage(
        &baseline_problem.telemetry,
        IvpColdStage::AotCacheLookup,
        || {
            select_residual_backend(
                &baseline_problem,
                config.resolver.as_ref(),
                config.aot_options,
            )
        },
    );
    let problem_key = baseline_problem
        .prepare_residual_aot_problem(config.aot_options)
        .problem_key();
    log_aot_cache_selection(
        &baseline_problem.telemetry,
        if is_atom_view {
            "residual-atom-native"
        } else {
            "residual-expr-legacy"
        },
        problem_key.as_str(),
        "phase=initial-selection",
        initial_selection,
    );
    baseline_problem
        .telemetry
        .record_aot_resolution(initial_selection == SelectedSymbolicIvpBackendKind::AotCompiled);
    let resolved_plan = ResolvedIvpAotPlan::resolve(&config, initial_selection);
    let (build_result, resolver_snapshot) = if resolved_plan.should_build() {
        perform_requested_residual_build(&baseline_problem, &config, config.resolver.clone())?
    } else {
        (None, config.resolver.clone())
    };

    let final_selection = measure_cold_stage(
        &baseline_problem.telemetry,
        IvpColdStage::AotCacheLookup,
        || {
            select_residual_backend(
                &baseline_problem,
                resolver_snapshot.as_ref(),
                config.aot_options,
            )
        },
    );
    log_aot_cache_selection(
        &baseline_problem.telemetry,
        if is_atom_view {
            "residual-atom-native"
        } else {
            "residual-expr-legacy"
        },
        problem_key.as_str(),
        "phase=final-selection",
        final_selection,
    );
    baseline_problem
        .telemetry
        .record_aot_resolution(final_selection == SelectedSymbolicIvpBackendKind::AotCompiled);
    match final_selection {
        SelectedSymbolicIvpBackendKind::Lambdify | SelectedSymbolicIvpBackendKind::AotMissing => {
            match config.build_policy {
                SymbolicIvpAotBuildPolicy::RequirePrebuilt => {
                    Err(SymbolicIvpGeneratedError::CompiledAotArtifactMissing(
                        missing_aot_message("residual-only", problem_key.as_str(), &config),
                    ))
                }
                _ => Ok(PreparedGeneratedSymbolicIvpResidualProblem {
                    problem: baseline_problem,
                    selected_backend: SelectedSymbolicIvpBackendKind::Lambdify,
                    updated_resolver: resolver_snapshot,
                    build_result,
                    runtime_owner: None,
                }),
            }
        }
        SelectedSymbolicIvpBackendKind::AotRegisteredButNotBuilt => {
            Err(SymbolicIvpGeneratedError::CompiledAotArtifactNotBuilt(
                not_built_aot_message("residual-only", problem_key.as_str(), &config),
            ))
        }
        SelectedSymbolicIvpBackendKind::AotCompiled => {
            let prepared = baseline_problem.prepare_residual_aot_problem(config.aot_options);
            let problem_key = prepared.problem_key();
            baseline_problem.telemetry.record_aot_link_attempt();
            let had_process_local_runtime =
                resolve_linked_residual_backend(problem_key.as_str()).is_some();
            if let Some(linked) = reconnect_ivp_native_residual_runtime_backend(
                "residual-only",
                config.aot_codegen_backend,
                resolver_snapshot.as_ref(),
                problem_key.as_str(),
            )? {
                baseline_problem.telemetry.record_aot_link_result(true);
                if !had_process_local_runtime {
                    baseline_problem.telemetry.record_aot_reconnect();
                    baseline_problem.telemetry.log_aot_event(
                        crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::Linked,
                        "residual-only",
                        problem_key.as_str(),
                        "compiled runtime reconnected",
                    );
                }
                baseline_problem.telemetry.record_aot_runtime_ready();
                baseline_problem.telemetry.log_aot_event(
                    crate::symbolic::ivp_telemetry::IvpAotLifecycleEvent::RuntimeReady,
                    "residual-only",
                    problem_key.as_str(),
                    "runtime ready for solver handoff",
                );
                let runtime_owner = PreparedIvpAotRuntime::new(
                    problem_key.to_string(),
                    SelectedSymbolicIvpBackendKind::AotCompiled,
                    resolver_snapshot.clone(),
                    build_result.clone(),
                    PreparedIvpLinkedRuntime::Residual(linked.clone()),
                );
                Ok(PreparedGeneratedSymbolicIvpResidualProblem {
                    problem: baseline_problem.into_linked_residual_backend(linked),
                    selected_backend: SelectedSymbolicIvpBackendKind::AotCompiled,
                    updated_resolver: resolver_snapshot,
                    build_result,
                    runtime_owner: Some(runtime_owner),
                })
            } else {
                match config.build_policy {
                    SymbolicIvpAotBuildPolicy::RequirePrebuilt => {
                        Err(SymbolicIvpGeneratedError::CompiledAotRuntimeUnavailable(
                            runtime_unavailable_aot_message(
                                "residual-only",
                                problem_key.as_str(),
                                &config,
                            ),
                        ))
                    }
                    _ => {
                        baseline_problem.telemetry.record_aot_link_result(false);
                        warn!(
                            "Symbolic IVP residual-only compiled AOT artifact exists but no linked runtime is registered; falling back to lambdify"
                        );
                        Ok(PreparedGeneratedSymbolicIvpResidualProblem {
                            problem: baseline_problem,
                            selected_backend: SelectedSymbolicIvpBackendKind::Lambdify,
                            updated_resolver: resolver_snapshot,
                            build_result,
                            runtime_owner: None,
                        })
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::codegen::codegen_aot_registry::AotRegistry;
    use crate::symbolic::codegen::codegen_aot_runtime_link::{
        LinkedDenseAotBackend, LinkedDenseJacobianChunk, register_linked_dense_backend,
        resolve_linked_residual_backend, resolve_linked_sparse_backend,
        unregister_linked_dense_backend, unregister_linked_residual_backend,
        unregister_linked_sparse_backend,
    };
    use crate::symbolic::ivp_telemetry::{IvpColdStage, IvpTelemetry, IvpTelemetryRoute};
    use crate::symbolic::symbolic_engine::Expr;
    use crate::symbolic::symbolic_ivp::{IvpBackendKind, IvpSymbolicAssemblyBackend};
    use nalgebra::DVector;
    use std::fs;
    use std::sync::Arc;
    use tempfile::tempdir;

    fn sample_problem() -> (Vec<Expr>, Vec<String>, String, SymbolicIvpProblemOptions) {
        sample_problem_with_offset(0.0)
    }

    fn sample_problem_with_offset(
        offset: f64,
    ) -> (Vec<Expr>, Vec<String>, String, SymbolicIvpProblemOptions) {
        (
            vec![
                Expr::parse_expression(&format!("a*t + y + b*z + {offset:.17}")),
                Expr::parse_expression("c*y - z + b*t"),
            ],
            vec!["y".to_string(), "z".to_string()],
            "t".to_string(),
            SymbolicIvpProblemOptions::new()
                .with_equation_parameters(vec!["a".to_string(), "b".to_string(), "c".to_string()])
                .with_equation_parameter_values(DVector::from_vec(vec![2.0, -0.5, 3.0])),
        )
    }

    #[test]
    fn generated_ivp_default_artifact_names_stay_windows_toolchain_friendly() {
        let config = SymbolicIvpGeneratedBackendConfig::defaults();
        let problem_key = "534b35bdcbb8f0ed-with-extra-long-tail";

        let (dense_crate, dense_module) = generated_names(problem_key, &config);
        let (residual_crate, residual_module) = generated_residual_names(problem_key, &config);
        let (sparse_crate, sparse_module) = generated_sparse_names(problem_key, &config);

        for name in [
            dense_crate.as_str(),
            dense_module.as_str(),
            residual_crate.as_str(),
            residual_module.as_str(),
            sparse_crate.as_str(),
            sparse_module.as_str(),
        ] {
            assert!(
                name.len() <= 32,
                "default IVP AOT names should stay short for deep Windows cold-build paths: {name}"
            );
        }
        assert!(dense_crate.starts_with("ivp_dense_"));
        assert!(residual_crate.starts_with("ivp_res_"));
        assert!(sparse_crate.starts_with("ivp_sp_"));

        let override_config = SymbolicIvpGeneratedBackendConfig::defaults()
            .with_crate_name_override(Some("custom_generated_crate".to_string()))
            .with_module_name_override(Some("custom_generated_module".to_string()));
        assert_eq!(
            generated_residual_names(problem_key, &override_config),
            (
                "custom_generated_crate".to_string(),
                "custom_generated_module".to_string()
            )
        );
    }

    #[test]
    fn generated_ivp_defaults_fall_back_to_lambdify_when_aot_is_missing() {
        let (equations, variables, time_arg, options) = sample_problem();
        let prepared = prepare_generated_symbolic_ivp_problem(
            equations,
            variables,
            time_arg,
            options,
            SymbolicIvpGeneratedBackendConfig::defaults(),
        )
        .expect("defaults should fall back to lambdify");

        assert_eq!(
            prepared.selected_backend,
            SelectedSymbolicIvpBackendKind::Lambdify
        );
        assert_eq!(prepared.problem.backend_kind, IvpBackendKind::Lambdify);
        assert!(
            prepared
                .try_aot_runtime()
                .expect("Lambdify fallback has no stale AOT owner")
                .is_none()
        );
    }

    #[test]
    fn generated_atomview_use_if_available_stays_native_when_aot_is_not_requested() {
        let (equations, variables, time_arg, options) = sample_problem();
        let telemetry = IvpTelemetry::counters();
        let options = options
            .with_symbolic_assembly_backend(IvpSymbolicAssemblyBackend::AtomView)
            .with_telemetry(telemetry.clone());

        let prepared = prepare_generated_symbolic_ivp_problem(
            equations,
            variables,
            time_arg,
            options,
            SymbolicIvpGeneratedBackendConfig::defaults(),
        )
        .expect("native AtomView should prepare without an AOT compatibility pass");

        assert_eq!(
            prepared.selected_backend,
            SelectedSymbolicIvpBackendKind::Lambdify
        );
        assert_eq!(prepared.problem.backend_kind, IvpBackendKind::Lambdify);
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.route, IvpTelemetryRoute::AtomViewNative);
        assert_eq!(snapshot.cold_stage(IvpColdStage::AtomToExpr).calls, 0);
        assert!(snapshot.cold_stage(IvpColdStage::ExprToAtom).calls > 0);
    }

    #[test]
    fn generated_atomview_sparse_route_prepares_native_aot_identity_without_expr_jacobian() {
        let (equations, variables, time_arg, options) = sample_problem();
        let prepared = prepare_generated_symbolic_ivp_sparse_backend(
            equations,
            variables,
            time_arg,
            options.with_symbolic_assembly_backend(IvpSymbolicAssemblyBackend::AtomView),
            SymbolicIvpGeneratedBackendConfig::defaults(),
        )
        .expect("AtomView sparse route should fall back to native lambdify");

        assert_eq!(
            prepared.selected_backend,
            SelectedSymbolicIvpBackendKind::Lambdify
        );
        assert_eq!(prepared.jacobian_structure.rows, 2);
        assert_eq!(prepared.jacobian_structure.cols, 2);
        assert_eq!(prepared.jacobian_structure.nnz(), 4);
        assert!(prepared.problem_key != "");
    }

    #[test]
    fn generated_atomview_aot_handoff_reuses_the_single_native_atom_payload() {
        let (equations, variables, time_arg, options) = sample_problem();
        let telemetry = IvpTelemetry::detailed();
        let residual_problem =
            crate::symbolic::symbolic_ivp::prepare_symbolic_ivp_residual_problem(
                equations,
                variables,
                time_arg,
                options
                    .with_symbolic_assembly_backend(IvpSymbolicAssemblyBackend::AtomView)
                    .with_telemetry(telemetry.clone()),
            )
            .expect("native AtomView residual preparation should succeed");

        let _native = prepared_atom_aot_problem_from_residual_problem(
            &residual_problem,
            ResidualChunkingStrategy::Whole,
            SparseChunkingStrategy::Whole,
            crate::symbolic::bvp::atom_aot::AtomAotMatrixLayout::SparseCsc {
                rows: 2,
                cols: 2,
                nnz: 0,
            },
        )
        .expect("native AOT handoff should reuse the prepared payload");

        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.cold_stage(IvpColdStage::ExprToAtom).calls, 1);
        assert_eq!(snapshot.cold_stage(IvpColdStage::AtomToExpr).calls, 0);
        assert_eq!(snapshot.cold_stage(IvpColdStage::SparsePattern).calls, 1);
        assert_eq!(snapshot.cold_stage(IvpColdStage::LayoutPlanning).calls, 1);
    }

    #[test]
    fn generated_ivp_require_prebuilt_surfaces_missing_artifact() {
        let (equations, variables, time_arg, options) = sample_problem();
        let result = prepare_generated_symbolic_ivp_problem(
            equations,
            variables,
            time_arg,
            options,
            SymbolicIvpGeneratedBackendConfig::require_prebuilt(),
        );

        match result {
            Err(SymbolicIvpGeneratedError::CompiledAotArtifactMissing(message)) => {
                assert!(message.contains("route=dense"));
                assert!(message.contains("problem_key="));
                assert!(message.contains("build_policy=RequirePrebuilt"));
                assert!(message.contains("codegen_backend="));
                assert!(message.contains("output_parent_dir="));
                assert!(message.contains("BuildIfMissing/RebuildAlways"));
            }
            Err(other) => panic!("expected missing compiled artifact error, got {other}"),
            Ok(_) => panic!("missing prebuilt artifact should be surfaced"),
        }
    }

    #[test]
    fn generated_ivp_aot_diagnostic_messages_include_context() {
        let config = SymbolicIvpGeneratedBackendConfig::require_prebuilt()
            .with_c_tcc()
            .with_output_parent_dir(Some(PathBuf::from("target/ivp-diagnostics")));
        let message = missing_aot_message("sparse", "abc123", &config);

        assert!(message.contains("route=sparse"));
        assert!(message.contains("problem_key=abc123"));
        assert!(message.contains("build_policy=RequirePrebuilt"));
        assert!(message.contains("codegen_backend=C"));
        assert!(message.contains("c_compiler=tcc"));
        assert!(message.contains("target/ivp-diagnostics"));

        let not_built = not_built_aot_message("residual-only", "def456", &config);
        assert!(not_built.contains("registered but the expected compiled file is not present"));
        assert!(not_built.contains("problem_key=def456"));

        let runtime = runtime_unavailable_aot_message("dense", "ghi789", &config);
        assert!(runtime.contains("compiled AOT artifact exists but no linked runtime"));
        assert!(runtime.contains("problem_key=ghi789"));

        let retry = retry_exhausted_aot_message(
            "ivp-dense backend=C key=abc123",
            3,
            "failed to spawn build runner",
            true,
        );
        assert!(retry.contains("after 3 attempt(s)"));
        assert!(retry.contains("transient infrastructure failure"));
        assert!(retry.contains("stale compiler processes"));

        let load = runtime_registration_error_message(
            "dense",
            AotCodegenBackend::C,
            "abc123",
            std::path::Path::new("target/example.dll"),
            "load failed".to_string(),
        );
        let load_message = load.to_string();
        assert!(load_message.contains("symbolic IVP dense"));
        assert!(load_message.contains("problem_key=abc123"));
        assert!(load_message.contains("artifact_path=target/example.dll"));
        assert!(load_message.contains("stale/incompatible"));
    }

    #[test]
    fn generated_ivp_materialization_failure_preserves_typed_diagnostics() {
        let error = materialization_error(
            "dense",
            "materialization-key",
            std::io::Error::new(std::io::ErrorKind::PermissionDenied, "read-only output"),
        );
        match error {
            SymbolicIvpGeneratedError::AotLifecycle(error) => {
                assert_eq!(
                    error.diagnostics.stage,
                    crate::symbolic::codegen::codegen_aot_lifecycle::AotLifecycleStage::Materialized
                );
                assert_eq!(
                    error.diagnostics.kind,
                    crate::symbolic::codegen::codegen_aot_lifecycle::AotFailureKind::Io
                );
                assert_eq!(error.diagnostics.artifact_key, "materialization-key");
                assert!(error.diagnostics.detail.contains("dense"));
                assert!(error.diagnostics.detail.contains("read-only output"));
            }
            other => panic!("expected typed materialization failure, got {other:?}"),
        }
    }

    #[test]
    fn prepared_ivp_aot_runtime_validation_reports_linked_key_mismatch() {
        let linked = LinkedDenseAotBackend::new(
            "linked-key",
            1,
            (1, 1),
            Arc::new(|_, output| output[0] = 0.0),
            Arc::new(|_, output| output[0] = 1.0),
        );
        let runtime = PreparedIvpAotRuntime::new(
            "owner-key".to_string(),
            SelectedSymbolicIvpBackendKind::AotCompiled,
            None,
            None,
            PreparedIvpLinkedRuntime::Dense(linked),
        );

        assert_eq!(
            runtime.validate(),
            Err(PreparedIvpRuntimeError::LinkedProblemKeyMismatch {
                expected: "owner-key".to_string(),
                actual: "linked-key".to_string(),
            })
        );
    }

    #[test]
    fn prepared_ivp_aot_runtime_validation_rejects_incomplete_chunks() {
        let linked = LinkedDenseAotBackend::new(
            "chunk-key",
            2,
            (2, 1),
            Arc::new(|_, output| output.fill(0.0)),
            Arc::new(|_, output| output.fill(1.0)),
        )
        .with_chunked_evaluators(
            Vec::new(),
            vec![LinkedDenseJacobianChunk::new(
                0,
                1,
                Arc::new(|_, output| output[0] = 1.0),
            )],
        );
        let runtime = PreparedIvpAotRuntime::new(
            "chunk-key".to_string(),
            SelectedSymbolicIvpBackendKind::AotCompiled,
            None,
            None,
            PreparedIvpLinkedRuntime::Dense(linked),
        );

        let error = runtime
            .validate()
            .expect_err("incomplete Jacobian chunks must be rejected before execution");
        assert!(matches!(
            error,
            PreparedIvpRuntimeError::InvalidLinkedLayout {
                runtime: "dense",
                ..
            }
        ));
    }

    #[test]
    fn prepared_ivp_aot_runtime_validation_reports_missing_artifact() {
        let linked = LinkedDenseAotBackend::new(
            "missing-key",
            1,
            (1, 1),
            Arc::new(|_, output| output[0] = 0.0),
            Arc::new(|_, output| output[0] = 1.0),
        );
        let runtime = PreparedIvpAotRuntime::new(
            "missing-key".to_string(),
            SelectedSymbolicIvpBackendKind::AotCompiled,
            Some(AotResolver::new(AotRegistry::new())),
            None,
            PreparedIvpLinkedRuntime::Dense(linked),
        );

        assert_eq!(
            runtime.validate(),
            Err(PreparedIvpRuntimeError::ArtifactMissing {
                problem_key: "missing-key".to_string(),
            })
        );
    }

    #[test]
    fn prepared_ivp_aot_runtime_validation_reports_registered_but_not_built() {
        let (equations, variables, time_arg, options) = sample_problem_with_offset(0.375);
        let dir = tempdir().expect("tempdir should exist");
        let baseline = prepare_symbolic_ivp_problem(equations, variables, time_arg, options)
            .expect("symbolic IVP preparation should succeed");
        let prepared = baseline.prepare_dense_aot_problem(SymbolicIvpAotOptions::default());
        let generic = crate::symbolic::codegen::codegen_provider_api::PreparedProblem::dense(
            prepared.as_prepared_problem(),
        );
        let crate_spec =
            crate::symbolic::codegen::codegen_aot_driver::generated_aot_crate_from_prepared_problem(
                "registered_not_built_fixture",
                "registered_not_built_module",
                &generic,
            );
        let build =
            crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildRequest::new(
                crate_spec,
                dir.path(),
                AotBuildProfile::Debug,
            )
            .materialize()
            .expect("AOT fixture should materialize");
        let mut registry = AotRegistry::new();
        registry.register_materialized_build(prepared.manifest(), &build);
        let resolver = AotResolver::new(registry);
        let runtime_key = prepared.problem_key();

        let linked = LinkedDenseAotBackend::new(
            runtime_key.clone(),
            2,
            (2, 2),
            Arc::new(|_, output| output.fill(0.0)),
            Arc::new(|_, output| output.fill(1.0)),
        );
        let runtime = PreparedIvpAotRuntime::new(
            runtime_key.clone(),
            SelectedSymbolicIvpBackendKind::AotCompiled,
            Some(resolver),
            None,
            PreparedIvpLinkedRuntime::Dense(linked),
        );

        assert_eq!(
            runtime.validate(),
            Err(PreparedIvpRuntimeError::ArtifactNotReady {
                problem_key: runtime_key,
                status: AotResolutionStatus::RegisteredButNotBuilt,
            })
        );
    }

    #[test]
    fn prepared_ivp_aot_runtime_rejects_registered_artifact_after_schema_change() {
        let (equations, variables, time_arg, options) = sample_problem_with_offset(0.0);
        let (changed_equations, changed_variables, changed_time_arg, changed_options) =
            sample_problem_with_offset(1.0);
        let dir = tempdir().expect("tempdir should exist");
        let baseline = prepare_symbolic_ivp_problem(equations, variables, time_arg, options)
            .expect("baseline symbolic IVP preparation should succeed");
        let changed = prepare_symbolic_ivp_problem(
            changed_equations,
            changed_variables,
            changed_time_arg,
            changed_options,
        )
        .expect("changed symbolic IVP preparation should succeed");
        let baseline_prepared =
            baseline.prepare_dense_aot_problem(SymbolicIvpAotOptions::default());
        let changed_prepared = changed.prepare_dense_aot_problem(SymbolicIvpAotOptions::default());
        assert_ne!(
            baseline_prepared.problem_key(),
            changed_prepared.problem_key(),
            "schema changes must produce a new artifact identity"
        );

        let baseline_generic =
            crate::symbolic::codegen::codegen_provider_api::PreparedProblem::dense(
                baseline_prepared.as_prepared_problem(),
            );
        let crate_spec =
            crate::symbolic::codegen::codegen_aot_driver::generated_aot_crate_from_prepared_problem(
                "schema_invalidation_fixture",
                "schema_invalidation_module",
                &baseline_generic,
            );
        let build =
            crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildRequest::new(
                crate_spec,
                dir.path(),
                AotBuildProfile::Debug,
            )
            .materialize()
            .expect("baseline AOT fixture should materialize");
        let mut registry = AotRegistry::new();
        registry.register_materialized_build(baseline_prepared.manifest(), &build);
        let runtime_key = changed_prepared.problem_key();
        let linked = LinkedDenseAotBackend::new(
            runtime_key.clone(),
            2,
            (2, 2),
            Arc::new(|_, output| output.fill(0.0)),
            Arc::new(|_, output| output.fill(1.0)),
        );
        let runtime = PreparedIvpAotRuntime::new(
            runtime_key.clone(),
            SelectedSymbolicIvpBackendKind::AotCompiled,
            Some(AotResolver::new(registry)),
            None,
            PreparedIvpLinkedRuntime::Dense(linked),
        );

        assert_eq!(
            runtime.validate(),
            Err(PreparedIvpRuntimeError::ArtifactMissing {
                problem_key: runtime_key,
            })
        );
    }

    #[test]
    fn prepared_ivp_aot_runtime_rejects_stale_published_output_without_marker() {
        let (equations, variables, time_arg, options) = sample_problem_with_offset(0.625);
        let dir = tempdir().expect("tempdir should exist");
        let baseline = prepare_symbolic_ivp_problem(equations, variables, time_arg, options)
            .expect("symbolic IVP preparation should succeed");
        let prepared = baseline.prepare_dense_aot_problem(SymbolicIvpAotOptions::default());
        let generic = crate::symbolic::codegen::codegen_provider_api::PreparedProblem::dense(
            prepared.as_prepared_problem(),
        );
        let crate_spec =
            crate::symbolic::codegen::codegen_aot_driver::generated_aot_crate_from_prepared_problem(
                "stale_publication_fixture",
                "stale_publication_module",
                &generic,
            );
        let build =
            crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildRequest::new(
                crate_spec,
                dir.path(),
                AotBuildProfile::Debug,
            )
            .materialize()
            .expect("AOT fixture should materialize");
        let mut registry = AotRegistry::new();
        registry.register_materialized_build(prepared.manifest(), &build);
        fs::remove_file(&build.written.manifest_rs)
            .expect("test should be able to remove the publication marker");
        fs::create_dir_all(&build.artifact_dir).expect("artifact dir should exist");
        fs::write(&build.expected_rlib, b"stale output")
            .expect("test should be able to create a stale output");

        let runtime_key = prepared.problem_key();
        let linked = LinkedDenseAotBackend::new(
            runtime_key.clone(),
            2,
            (2, 2),
            Arc::new(|_, output| output.fill(0.0)),
            Arc::new(|_, output| output.fill(1.0)),
        );
        let runtime = PreparedIvpAotRuntime::new(
            runtime_key.clone(),
            SelectedSymbolicIvpBackendKind::AotCompiled,
            Some(AotResolver::new(registry)),
            None,
            PreparedIvpLinkedRuntime::Dense(linked),
        );

        let error = runtime
            .validate()
            .expect_err("output without its marker must not be reused");
        assert!(matches!(
            error,
            PreparedIvpRuntimeError::ArtifactInvalidated {
                problem_key,
                state: AotArtifactState::Stale,
                ..
            } if problem_key == runtime_key
        ));
    }

    #[test]
    fn generated_ivp_rebuild_always_uses_isolated_output_parent_dirs() {
        let base = PathBuf::from("target/ivp-rebuild-lock-safety");
        let stable = SymbolicIvpGeneratedBackendConfig::build_if_missing_release(base.clone());
        let stable_dir =
            output_parent_dir_for_requested_build(&stable, "sparse", "abcdef0123456789")
                .expect("stable build path should resolve");
        assert_eq!(stable_dir, base);

        let rebuild = SymbolicIvpGeneratedBackendConfig::new()
            .with_output_parent_dir(Some(base.clone()))
            .with_build_policy(SymbolicIvpAotBuildPolicy::RebuildAlways {
                profile: AotBuildProfile::Release,
            });
        let first = output_parent_dir_for_requested_build(&rebuild, "sparse", "abcdef0123456789")
            .expect("first rebuild path should resolve");
        let second = output_parent_dir_for_requested_build(&rebuild, "sparse", "abcdef0123456789")
            .expect("second rebuild path should resolve");

        assert!(first.starts_with(&base));
        assert!(second.starts_with(&base));
        assert_ne!(
            first, second,
            "RebuildAlways must not overwrite a possibly loaded AOT artifact"
        );
        assert!(
            first
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.contains("rebuild_sparse_abcdef0123456789")),
            "isolated rebuild path should remain diagnosable: {:?}",
            first
        );
    }

    #[test]
    fn generated_ivp_resolved_aot_plan_separates_build_policies() {
        let missing = SelectedSymbolicIvpBackendKind::AotMissing;
        let compiled = SelectedSymbolicIvpBackendKind::AotCompiled;

        let use_if_available =
            ResolvedIvpAotPlan::resolve(&SymbolicIvpGeneratedBackendConfig::defaults(), missing);
        assert_eq!(
            use_if_available.action,
            ResolvedIvpAotBuildAction::ReuseOrFallback
        );
        assert!(!use_if_available.should_build());
        assert_eq!(use_if_available.profile(), None);
        assert_eq!(use_if_available.preset(), None);

        let require_prebuilt = ResolvedIvpAotPlan::resolve(
            &SymbolicIvpGeneratedBackendConfig::require_prebuilt(),
            missing,
        );
        assert_eq!(
            require_prebuilt.action,
            ResolvedIvpAotBuildAction::ReuseOrFallback
        );
        assert!(!require_prebuilt.should_build());
        assert_eq!(
            require_prebuilt.policy,
            SymbolicIvpAotBuildPolicy::RequirePrebuilt
        );

        let build_if_missing = ResolvedIvpAotPlan::resolve(
            &SymbolicIvpGeneratedBackendConfig::new().with_build_policy(
                SymbolicIvpAotBuildPolicy::BuildIfMissing {
                    profile: AotBuildProfile::Debug,
                },
            ),
            missing,
        );
        assert_eq!(
            build_if_missing.action,
            ResolvedIvpAotBuildAction::BuildIfMissing
        );
        assert!(build_if_missing.should_build());
        assert_eq!(build_if_missing.profile(), Some(AotBuildProfile::Debug));
        assert_eq!(build_if_missing.preset(), Some(AotBuildPreset::DevFastest));

        let build_if_missing_with_compiled = ResolvedIvpAotPlan::resolve(
            &SymbolicIvpGeneratedBackendConfig::new().with_build_policy(
                SymbolicIvpAotBuildPolicy::BuildIfMissing {
                    profile: AotBuildProfile::Release,
                },
            ),
            compiled,
        );
        assert_eq!(
            build_if_missing_with_compiled.action,
            ResolvedIvpAotBuildAction::ReuseOrFallback
        );
        assert!(!build_if_missing_with_compiled.should_build());
        assert_eq!(
            build_if_missing_with_compiled.preset(),
            Some(AotBuildPreset::Production)
        );

        let rebuild = ResolvedIvpAotPlan::resolve(
            &SymbolicIvpGeneratedBackendConfig::new().with_build_policy(
                SymbolicIvpAotBuildPolicy::RebuildAlways {
                    profile: AotBuildProfile::Debug,
                },
            ),
            compiled,
        );
        assert_eq!(rebuild.action, ResolvedIvpAotBuildAction::RebuildAlways);
        assert!(rebuild.should_build());
        assert_eq!(rebuild.initial_selection, compiled);
    }

    #[test]
    fn generated_ivp_backend_config_selects_c_and_zig_backends() {
        let c_tcc = SymbolicIvpGeneratedBackendConfig::defaults().with_c_tcc();
        assert_eq!(c_tcc.aot_codegen_backend, AotCodegenBackend::C);
        assert_eq!(c_tcc.aot_c_compiler.as_deref(), Some("tcc"));

        let c_gcc = SymbolicIvpGeneratedBackendConfig::defaults().with_c_gcc();
        assert_eq!(c_gcc.aot_codegen_backend, AotCodegenBackend::C);
        assert_eq!(c_gcc.aot_c_compiler.as_deref(), Some("gcc"));

        let zig = SymbolicIvpGeneratedBackendConfig::defaults()
            .with_c_tcc()
            .with_zig();
        assert_eq!(zig.aot_codegen_backend, AotCodegenBackend::Zig);
        assert_eq!(zig.aot_c_compiler, None);

        let rust = SymbolicIvpGeneratedBackendConfig::defaults()
            .with_c_gcc()
            .with_rust();
        assert_eq!(rust.aot_codegen_backend, AotCodegenBackend::Rust);
        assert_eq!(rust.aot_c_compiler, None);
    }

    #[test]
    fn generated_ivp_build_if_missing_release_prefers_c_gcc() {
        let config =
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release("target/generated-ivp");

        assert_eq!(config.aot_codegen_backend, AotCodegenBackend::C);
        assert_eq!(config.aot_c_compiler.as_deref(), Some("gcc"));
        assert_eq!(
            config.build_policy,
            SymbolicIvpAotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Release
            }
        );
    }

    #[test]
    fn generated_ivp_build_if_missing_materializes_build_and_updates_resolver() {
        let (equations, variables, time_arg, options) = sample_problem_with_offset(0.125);
        let dir = tempdir().expect("tempdir should exist");
        let prepared = prepare_generated_symbolic_ivp_problem(
            equations,
            variables,
            time_arg,
            options,
            SymbolicIvpGeneratedBackendConfig::defaults()
                .with_build_policy(SymbolicIvpAotBuildPolicy::BuildIfMissing {
                    profile: AotBuildProfile::Debug,
                })
                .with_output_parent_dir(Some(dir.path().to_path_buf())),
        )
        .expect("build-if-missing should succeed");

        assert_eq!(
            prepared.selected_backend,
            SelectedSymbolicIvpBackendKind::AotCompiled
        );
        assert!(prepared.build_result.is_some());
        let runtime = prepared
            .try_aot_runtime()
            .expect("compiled Dense owner should validate against its linked artifact")
            .expect("compiled Dense preparation should retain one runtime owner");
        assert_eq!(runtime.linked_runtime_kind(), "dense");
        assert_eq!(
            runtime.selected_backend(),
            SelectedSymbolicIvpBackendKind::AotCompiled
        );
        let resolver = prepared
            .updated_resolver
            .expect("build should update resolver");
        assert_eq!(resolver.registry().len(), 1);
    }

    #[test]
    fn generated_atomview_sparse_build_if_missing_materializes_native_artifact() {
        let (equations, variables, time_arg, options) = sample_problem_with_offset(0.125);
        let dir = tempdir().expect("tempdir should exist");
        let telemetry = IvpTelemetry::detailed();
        let prepared = prepare_generated_symbolic_ivp_sparse_backend(
            equations,
            variables,
            time_arg,
            options
                .with_symbolic_assembly_backend(IvpSymbolicAssemblyBackend::AtomView)
                .with_telemetry(telemetry.clone()),
            SymbolicIvpGeneratedBackendConfig::defaults()
                .with_residual_chunking_strategy(ResidualChunkingStrategy::ByOutputCount {
                    max_outputs_per_chunk: 1,
                })
                .with_sparse_jacobian_chunking_strategy(SparseChunkingStrategy::ByNonZeroCount {
                    max_entries_per_chunk: 2,
                })
                .with_build_policy(SymbolicIvpAotBuildPolicy::BuildIfMissing {
                    profile: AotBuildProfile::Debug,
                })
                .with_output_parent_dir(Some(dir.path().to_path_buf())),
        )
        .expect("native AtomView sparse build-if-missing should succeed");

        assert_eq!(
            prepared.selected_backend,
            SelectedSymbolicIvpBackendKind::AotCompiled
        );
        assert_eq!(prepared.jacobian_structure.nnz(), 4);
        assert!(prepared.build_result.is_some());
        let runtime = prepared
            .try_aot_runtime()
            .expect("compiled Sparse owner should validate against its linked artifact")
            .expect("compiled Sparse preparation should retain one runtime owner");
        assert_eq!(runtime.linked_runtime_kind(), "sparse");
        assert_eq!(runtime.problem_key(), prepared.problem_key.as_str());
        let resolver = prepared
            .updated_resolver
            .expect("native sparse build should update resolver");
        assert_eq!(resolver.registry().len(), 1);

        let linked = prepared
            .linked_backend
            .expect("native sparse build should publish a linked callback");
        assert_eq!(linked.residual_chunks.len(), 2);
        assert_eq!(linked.jacobian_value_chunks.len(), 2);
        let args = [0.25, 2.0, -0.5, 3.0, 1.0, 2.0];
        let mut residual = [0.0; 2];
        linked
            .try_residual_eval(&args, &mut residual)
            .expect("native residual callback should accept the prepared ABI");
        assert!((residual[0] - 0.625).abs() < 1.0e-12);
        assert!((residual[1] - 0.875).abs() < 1.0e-12);

        let mut jacobian = [0.0; 4];
        linked
            .try_jacobian_values_eval(&args, &mut jacobian)
            .expect("native sparse Jacobian callback should accept the prepared ABI");
        for (actual, expected) in jacobian.iter().zip([1.0, -0.5, 3.0, -1.0]) {
            assert!((actual - expected).abs() < 1.0e-12);
        }

        let mut residual_chunk = [0.0; 1];
        linked
            .try_residual_chunk_eval(1, &args, &mut residual_chunk)
            .expect("native residual chunk callback should be linked");
        assert!((residual_chunk[0] - 0.875).abs() < 1.0e-12);

        let mut jacobian_chunk = [0.0; 2];
        linked
            .try_jacobian_chunk_eval(1, &args, &mut jacobian_chunk)
            .expect("native Jacobian chunk callback should be linked");
        assert!((jacobian_chunk[0] - 3.0).abs() < 1.0e-12);
        assert!((jacobian_chunk[1] + 1.0).abs() < 1.0e-12);

        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.cold_stage(IvpColdStage::ExprToAtom).calls, 1);
        assert_eq!(snapshot.cold_stage(IvpColdStage::AtomToExpr).calls, 0);
        assert_eq!(
            snapshot.cold_stage(IvpColdStage::AotMaterialization).calls,
            1
        );
        assert_eq!(snapshot.cold_stage(IvpColdStage::AotBuild).calls, 1);
        assert_eq!(snapshot.cold_stage(IvpColdStage::AotLink).calls, 1);
    }

    #[test]
    fn generated_atomview_banded_build_if_missing_publishes_compact_slots() {
        let (equations, variables, time_arg, options) = sample_problem_with_offset(0.125);
        let dir = tempdir().expect("tempdir should exist");
        let telemetry = IvpTelemetry::detailed();
        let prepared = prepare_generated_symbolic_ivp_banded_backend(
            equations,
            variables,
            time_arg,
            (1, 1),
            options
                .with_symbolic_assembly_backend(IvpSymbolicAssemblyBackend::AtomView)
                .with_telemetry(telemetry.clone()),
            SymbolicIvpGeneratedBackendConfig::defaults()
                .with_build_policy(SymbolicIvpAotBuildPolicy::BuildIfMissing {
                    profile: AotBuildProfile::Debug,
                })
                .with_output_parent_dir(Some(dir.path().to_path_buf())),
        )
        .expect("native AtomView compact-Banded build should succeed");

        assert_eq!(
            prepared.selected_backend,
            SelectedSymbolicIvpBackendKind::AotCompiled
        );
        let linked = prepared
            .linked_backend
            .expect("compact-Banded build should publish a linked callback");
        assert_eq!(
            linked.jacobian_layout,
            crate::symbolic::codegen::codegen_aot_runtime_link::LinkedJacobianLayout::BandedCompact {
                rows: 2,
                cols: 2,
                kl: 1,
                ku: 1,
            }
        );

        let args = [0.25, 2.0, -0.5, 3.0, 1.0, 2.0];
        let mut values = [0.0; 6];
        linked
            .try_jacobian_values_eval(&args, &mut values)
            .expect("compact-Banded callback should accept its complete slot buffer");
        let banded = crate::somelinalg::banded::storage::Banded::from_vec(2, 1, 1, values.to_vec())
            .expect("compact callback output should form a valid Banded matrix");
        assert!((banded[(0, 0)] - 1.0).abs() < 1.0e-12);
        assert!((banded[(0, 1)] + 0.5).abs() < 1.0e-12);
        assert!((banded[(1, 0)] - 3.0).abs() < 1.0e-12);
        assert!((banded[(1, 1)] + 1.0).abs() < 1.0e-12);

        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.cold_stage(IvpColdStage::ExprToAtom).calls, 1);
        assert_eq!(snapshot.cold_stage(IvpColdStage::AtomToExpr).calls, 0);
        assert_eq!(
            snapshot.cold_stage(IvpColdStage::AotMaterialization).calls,
            1
        );
        assert_eq!(snapshot.cold_stage(IvpColdStage::AotBuild).calls, 1);
        assert_eq!(snapshot.cold_stage(IvpColdStage::AotLink).calls, 1);
    }

    #[test]
    fn generated_atomview_banded_residual_reuses_compact_artifact_on_require_prebuilt() {
        let (equations, variables, time_arg, options) = sample_problem_with_offset(0.126);
        let dir = tempdir().expect("tempdir should exist");
        let build_config = SymbolicIvpGeneratedBackendConfig::defaults()
            .with_build_policy(SymbolicIvpAotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Debug,
            })
            .with_output_parent_dir(Some(dir.path().to_path_buf()));
        let built = prepare_generated_symbolic_ivp_banded_residual_problem(
            equations.clone(),
            variables.clone(),
            time_arg.clone(),
            (1, 1),
            options
                .clone()
                .with_symbolic_assembly_backend(IvpSymbolicAssemblyBackend::AtomView),
            build_config.clone(),
        )
        .expect("native AtomView compact-Banded residual build should succeed");

        assert_eq!(
            built.selected_backend,
            SelectedSymbolicIvpBackendKind::AotCompiled
        );
        assert!(
            built.build_result.is_some(),
            "first compact-Banded residual preparation should build"
        );
        let residual = built
            .problem
            .try_evaluate_residual(0.25, &DVector::from_vec(vec![1.0, 2.0]))
            .expect("linked compact-Banded residual callback should evaluate");
        assert!((residual[0] - 0.626).abs() < 1.0e-12);
        assert!((residual[1] - 0.875).abs() < 1.0e-12);

        // Simulate the process-local linked registries being empty while the
        // durable resolver/artifact handoff survives. RequirePrebuilt must
        // reconnect both callbacks from the compiled shared artifact rather
        // than requiring a caller-side registration hook.
        let problem_key = built
            .updated_resolver
            .as_ref()
            .and_then(|resolver| resolver.registry().problem_keys().into_iter().next())
            .expect("the build should publish one resolver artifact");
        unregister_linked_sparse_backend(problem_key.as_str());
        unregister_linked_residual_backend(problem_key.as_str());

        let require_config = build_config
            .with_resolver(built.updated_resolver.clone())
            .with_build_policy(SymbolicIvpAotBuildPolicy::RequirePrebuilt);
        let reused = prepare_generated_symbolic_ivp_banded_residual_problem(
            equations,
            variables,
            time_arg,
            (1, 1),
            options.with_symbolic_assembly_backend(IvpSymbolicAssemblyBackend::AtomView),
            require_config,
        )
        .expect("RequirePrebuilt should reuse the compact-Banded residual artifact");
        assert_eq!(
            reused.selected_backend,
            SelectedSymbolicIvpBackendKind::AotCompiled
        );
        assert!(
            reused.build_result.is_none(),
            "RequirePrebuilt reuse must not materialize a second build"
        );
        let reused_residual = reused
            .problem
            .try_evaluate_residual(0.25, &DVector::from_vec(vec![1.0, 2.0]))
            .expect("reused compact-Banded residual callback should evaluate");
        assert_eq!(residual, reused_residual);
    }

    #[test]
    fn generated_atomview_banded_require_prebuilt_reconnects_empty_process_registries() {
        let (equations, variables, time_arg, options) = sample_problem_with_offset(0.127);
        let dir = tempdir().expect("tempdir should exist");
        let telemetry = IvpTelemetry::counters();
        let build_config = SymbolicIvpGeneratedBackendConfig::defaults()
            .with_build_policy(SymbolicIvpAotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Debug,
            })
            .with_output_parent_dir(Some(dir.path().to_path_buf()));
        let built = prepare_generated_symbolic_ivp_banded_residual_problem(
            equations.clone(),
            variables.clone(),
            time_arg.clone(),
            (1, 1),
            options
                .clone()
                .with_symbolic_assembly_backend(IvpSymbolicAssemblyBackend::AtomView)
                .with_telemetry(telemetry.clone()),
            build_config.clone(),
        )
        .expect("initial compact-Banded build should succeed");
        let resolver = built
            .updated_resolver
            .clone()
            .expect("initial build should return a resolver");
        let problem_key = resolver
            .registry()
            .problem_keys()
            .into_iter()
            .next()
            .expect("initial build should publish one artifact");

        unregister_linked_sparse_backend(problem_key.as_str());
        unregister_linked_residual_backend(problem_key.as_str());
        assert!(resolve_linked_sparse_backend(problem_key.as_str()).is_none());
        assert!(resolve_linked_residual_backend(problem_key.as_str()).is_none());

        let reused = prepare_generated_symbolic_ivp_banded_residual_problem(
            equations,
            variables,
            time_arg,
            (1, 1),
            options
                .with_symbolic_assembly_backend(IvpSymbolicAssemblyBackend::AtomView)
                .with_telemetry(telemetry.clone()),
            build_config
                .with_resolver(Some(resolver))
                .with_build_policy(SymbolicIvpAotBuildPolicy::RequirePrebuilt),
        )
        .expect("RequirePrebuilt should reconnect linked callbacks from the artifact");

        assert_eq!(
            reused.selected_backend,
            SelectedSymbolicIvpBackendKind::AotCompiled
        );
        assert!(reused.build_result.is_none());
        let residual = reused
            .problem
            .try_evaluate_residual(0.25, &DVector::from_vec(vec![1.0, 2.0]))
            .expect("reconnected residual callback should evaluate");
        assert!((residual[0] - 0.627).abs() < 1.0e-12);
        assert!((residual[1] - 0.875).abs() < 1.0e-12);
        assert!(resolve_linked_sparse_backend(problem_key.as_str()).is_some());
        assert!(resolve_linked_residual_backend(problem_key.as_str()).is_some());
        let snapshot = telemetry.snapshot();
        assert!(snapshot.aot_reconnects >= 1);
        assert!(snapshot.aot_runtime_ready >= 2);
    }

    #[test]
    fn generated_atomview_aot_compiler_failure_returns_typed_partial_diagnostics() {
        let (equations, variables, time_arg, options) = sample_problem_with_offset(0.25);
        let dir = tempdir().expect("tempdir should exist");
        let telemetry = IvpTelemetry::detailed();
        let result = prepare_generated_symbolic_ivp_sparse_backend(
            equations,
            variables,
            time_arg,
            options
                .with_symbolic_assembly_backend(IvpSymbolicAssemblyBackend::AtomView)
                .with_telemetry(telemetry.clone()),
            SymbolicIvpGeneratedBackendConfig::defaults()
                .with_c_tcc()
                .with_aot_c_compiler("rustedscithe-missing-aot-compiler")
                .with_build_policy(SymbolicIvpAotBuildPolicy::BuildIfMissing {
                    profile: AotBuildProfile::Debug,
                })
                .with_output_parent_dir(Some(dir.path().to_path_buf())),
        );

        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("missing compiler must fail the typed AOT boundary"),
        };
        let message = error.to_string();
        assert!(
            message.contains("AOT build"),
            "unexpected AOT failure: {message}"
        );
        assert!(
            message.contains("rustedscithe-missing-aot-compiler"),
            "compiler identity was lost from AOT diagnostics: {message}"
        );
        assert!(matches!(
            &error,
            SymbolicIvpGeneratedError::AotLifecycle(lifecycle)
                if lifecycle.diagnostics.stage
                    == crate::symbolic::codegen::codegen_aot_lifecycle::AotLifecycleStage::Build
                    && lifecycle.diagnostics.kind
                        == crate::symbolic::codegen::codegen_aot_lifecycle::AotFailureKind::Compiler
                    && lifecycle.diagnostics.attempts == 1
        ));
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.cold_stage(IvpColdStage::ExprToAtom).calls, 1);
        assert_eq!(snapshot.cold_stage(IvpColdStage::AtomToExpr).calls, 0);
        assert_eq!(
            snapshot.cold_stage(IvpColdStage::AotMaterialization).calls,
            1
        );
        assert_eq!(snapshot.cold_stage(IvpColdStage::AotBuild).calls, 1);
        assert_eq!(snapshot.cold_stage(IvpColdStage::AotLink).calls, 0);
        assert_eq!(snapshot.aot_build_attempts, 1);
        assert_eq!(snapshot.aot_build_retries, 0);
        assert_eq!(snapshot.aot_build_successes, 0);
        assert_eq!(snapshot.aot_build_failures, 1);
    }

    #[test]
    fn generated_atomview_require_prebuilt_reports_missing_link_output_typed() {
        let (equations, variables, time_arg, options) = sample_problem_with_offset(0.251);
        let dir = tempdir().expect("tempdir should exist");
        let build_config = SymbolicIvpGeneratedBackendConfig::defaults()
            .with_build_policy(SymbolicIvpAotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Debug,
            })
            .with_output_parent_dir(Some(dir.path().to_path_buf()));
        let built = prepare_generated_symbolic_ivp_sparse_backend(
            equations.clone(),
            variables.clone(),
            time_arg.clone(),
            options
                .clone()
                .with_symbolic_assembly_backend(IvpSymbolicAssemblyBackend::AtomView),
            build_config.clone(),
        )
        .expect("initial AtomView build should succeed");
        let resolver = built
            .updated_resolver
            .clone()
            .expect("initial build should return a resolver");
        let problem_key = built.problem_key.clone();
        let artifact = resolver
            .registry()
            .get_by_problem_key(problem_key.as_str())
            .expect("resolver should retain the generated artifact")
            .clone();
        drop(built);
        unregister_linked_sparse_backend(problem_key.as_str());
        unregister_linked_residual_backend(problem_key.as_str());

        fs::remove_file(&artifact.expected_cdylib)
            .expect("failure injection should remove the compiled dynamic output");
        let result = prepare_generated_symbolic_ivp_sparse_backend(
            equations,
            variables,
            time_arg,
            options.with_symbolic_assembly_backend(IvpSymbolicAssemblyBackend::AtomView),
            build_config
                .with_resolver(Some(resolver))
                .with_build_policy(SymbolicIvpAotBuildPolicy::RequirePrebuilt),
        );
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("RequirePrebuilt must reject a missing linked output"),
        };
        let message = error.to_string();
        assert!(
            matches!(
                error,
                SymbolicIvpGeneratedError::AotLifecycle(ref lifecycle)
                    if lifecycle.diagnostics.kind
                        == crate::symbolic::codegen::codegen_aot_lifecycle::AotFailureKind::Link
            ),
            "missing cdylib must be a typed link/load lifecycle failure: {message}"
        );
        assert!(message.contains(problem_key.as_str()));
        assert!(message.contains("cdylib"));
    }

    #[test]
    fn generated_ivp_reuses_updated_resolver_with_linked_runtime() {
        let (equations, variables, time_arg, options) = sample_problem_with_offset(0.25);
        let dir = tempdir().expect("tempdir should exist");
        let first = prepare_generated_symbolic_ivp_problem(
            equations.clone(),
            variables.clone(),
            time_arg.clone(),
            options.clone(),
            SymbolicIvpGeneratedBackendConfig::defaults()
                .with_build_policy(SymbolicIvpAotBuildPolicy::BuildIfMissing {
                    profile: AotBuildProfile::Debug,
                })
                .with_output_parent_dir(Some(dir.path().to_path_buf())),
        )
        .expect("first build should succeed");
        let resolver = first
            .updated_resolver
            .clone()
            .expect("build should produce resolver");

        let baseline = prepare_symbolic_ivp_problem(equations, variables, time_arg, options)
            .expect("baseline IVP problem should prepare");
        let prepared = baseline.prepare_dense_aot_problem(SymbolicIvpAotOptions::default());
        let problem_key = prepared.problem_key();

        register_linked_dense_backend(LinkedDenseAotBackend::new(
            problem_key.clone(),
            2,
            (2, 2),
            Arc::new(|args: &[f64], out: &mut [f64]| {
                let t = args[0];
                let a = args[1];
                let b = args[2];
                let c = args[3];
                let y = args[4];
                let z = args[5];
                out[0] = a * t + y + b * z + 0.25;
                out[1] = c * y - z + b * t;
            }),
            Arc::new(|args: &[f64], out: &mut [f64]| {
                let c = args[3];
                out[0] = 1.0;
                out[1] = args[2];
                out[2] = c;
                out[3] = -1.0;
            }),
        ));

        let second = prepare_generated_symbolic_ivp_problem(
            vec![
                Expr::parse_expression("a*t + y + b*z + 0.25"),
                Expr::parse_expression("c*y - z + b*t"),
            ],
            vec!["y".to_string(), "z".to_string()],
            "t".to_string(),
            SymbolicIvpProblemOptions::new()
                .with_equation_parameters(vec!["a".to_string(), "b".to_string(), "c".to_string()])
                .with_equation_parameter_values(DVector::from_vec(vec![2.0, -0.5, 3.0])),
            SymbolicIvpGeneratedBackendConfig::defaults().with_resolver(Some(resolver)),
        )
        .expect("second prepare should reuse resolver and linked runtime");

        assert_eq!(
            second.selected_backend,
            SelectedSymbolicIvpBackendKind::AotCompiled
        );
        assert_eq!(second.problem.backend_kind, IvpBackendKind::Aot);
        unregister_linked_dense_backend(problem_key.as_str());
    }

    #[test]
    fn generated_ivp_cleanup_registered_aot_artifacts_is_safe_without_registered_artifacts() {
        let mut config = SymbolicIvpGeneratedBackendConfig::defaults();
        assert_eq!(config.cleanup_registered_aot_artifacts().unwrap(), 0);

        config = config.with_resolver(Some(AotResolver::new(AotRegistry::new())));
        assert_eq!(config.cleanup_registered_aot_artifacts().unwrap(), 0);
    }

    #[test]
    fn generated_ivp_transient_infra_failure_detector_matches_known_lock_and_spawn_signatures() {
        assert!(is_transient_aot_infra_failure(
            "tcc: error: could not write 'x.dll': Permission denied"
        ));
        assert!(is_transient_aot_infra_failure(
            "error: failed to spawn build runner target\\...\\build.zig"
        ));
        assert!(is_transient_aot_infra_failure(
            "The process cannot access the file because it is being used by another process"
        ));
        assert!(!is_transient_aot_infra_failure(
            "symbolic parse error: unknown variable q"
        ));
    }
}
