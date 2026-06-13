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

use crate::symbolic::codegen::c_backend::codegen_c_aot_build::CAotCompileConfig;
use crate::symbolic::codegen::c_backend::codegen_c_aot_registry::register_c_build_in_registry;
use crate::symbolic::codegen::c_backend::codegen_c_aot_runtime_link::{
    register_generated_c_dense_backend, register_generated_c_residual_backend,
    register_generated_c_sparse_backend,
};
use crate::symbolic::codegen::codegen_aot_driver::{
    generated_aot_artifact_from_prepared_problem, generated_aot_build_request_from_artifact,
    AotBuildPreset, AotCodegenBackend, ExecutedGeneratedAotBuild, GeneratedAotBuildRequest,
    GeneratedAotBuildResult,
};
use crate::symbolic::codegen::codegen_aot_resolution::{AotResolutionStatus, AotResolver};
use crate::symbolic::codegen::codegen_aot_runtime_link::{
    register_generated_dense_cdylib_backend, register_generated_residual_cdylib_backend,
    register_generated_sparse_cdylib_backend, resolve_linked_dense_backend,
    resolve_linked_residual_backend, resolve_linked_sparse_backend, LinkedSparseAotBackend,
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
    register_generated_zig_dense_backend, register_generated_zig_residual_backend,
    register_generated_zig_sparse_backend,
};
use crate::symbolic::symbolic_ivp::{
    prepare_symbolic_ivp_problem, prepare_symbolic_ivp_residual_problem, IvpBackendError,
    PreparedSymbolicIvpProblem, PreparedSymbolicIvpResidualProblem, SymbolicIvpAotOptions,
    SymbolicIvpProblemOptions,
};
use crate::symbolic::symbolic_ivp_aot::{
    generated_aot_artifact_from_symbolic_ivp_problem,
    generated_aot_artifact_from_symbolic_ivp_residual_problem,
};
use log::{info, warn};
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
) -> Result<ExecutedGeneratedAotBuild, SymbolicIvpGeneratedError> {
    const MAX_ATTEMPTS: usize = 3;
    let mut last_failure: Option<String> = None;
    let mut last_transient = false;

    for attempt in 1..=MAX_ATTEMPTS {
        match build.execute() {
            Ok(executed) => {
                if executed.succeeded() {
                    return Ok(executed);
                }
                let (status, stdout, stderr) = executed_process_output(&executed);
                let detail = format!("status={status:?}\nstdout:\n{stdout}\nstderr:\n{stderr}");
                let transient = is_transient_aot_infra_failure(&detail);
                last_transient = transient;
                last_failure = Some(detail);
                if transient && attempt < MAX_ATTEMPTS {
                    sleep(Duration::from_millis((attempt as u64) * 120));
                    continue;
                }
            }
            Err(err) => {
                let detail = err.to_string();
                let transient = is_transient_aot_infra_failure(&detail);
                last_transient = transient;
                last_failure = Some(detail);
                if transient && attempt < MAX_ATTEMPTS {
                    sleep(Duration::from_millis((attempt as u64) * 120));
                    continue;
                }
            }
        }
        break;
    }

    let detail = last_failure.unwrap_or_else(|| "unknown build failure".to_string());
    Err(SymbolicIvpGeneratedError::AotBuildFailed(
        retry_exhausted_aot_message(build_context, MAX_ATTEMPTS, &detail, last_transient),
    ))
}

fn runtime_registration_error_message(
    route: &str,
    backend: AotCodegenBackend,
    problem_key: &str,
    artifact_path: &std::path::Path,
    err: String,
) -> String {
    format!(
        "symbolic IVP {route} compiled AOT artifact could not be registered as a linked runtime \
(backend={backend:?}; problem_key={problem_key}; artifact_path={}). \
This usually means dynamic loading failed, the artifact is stale/incompatible, or the file is still locked by another process. \
Rebuild the artifact, check toolchain ABI compatibility, and make sure no stale process keeps the library open. detail: {err}",
        artifact_path.display()
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
}

impl PreparedGeneratedSymbolicIvpProblem {
    pub fn into_problem(self) -> PreparedSymbolicIvpProblem {
        self.problem
    }
}

/// Result of preparing one residual-only IVP symbolic problem through the
/// high-level generated-backend layer.
pub struct PreparedGeneratedSymbolicIvpResidualProblem {
    pub problem: PreparedSymbolicIvpResidualProblem,
    pub selected_backend: SelectedSymbolicIvpBackendKind,
    pub updated_resolver: Option<AotResolver>,
    pub build_result: Option<GeneratedAotBuildResult>,
}

impl PreparedGeneratedSymbolicIvpResidualProblem {
    pub fn into_problem(self) -> PreparedSymbolicIvpResidualProblem {
        self.problem
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
    pub updated_resolver: Option<AotResolver>,
    pub build_result: Option<GeneratedAotBuildResult>,
}

fn generated_names(
    problem_key: &str,
    config: &SymbolicIvpGeneratedBackendConfig,
) -> (String, String) {
    let suffix = problem_key
        .chars()
        .take(16)
        .collect::<String>()
        .replace('-', "_");
    let crate_name = config
        .crate_name_override
        .clone()
        .unwrap_or_else(|| format!("generated_ivp_dense_{suffix}"));
    let module_name = config
        .module_name_override
        .clone()
        .unwrap_or_else(|| format!("generated_ivp_dense_module_{suffix}"));
    (crate_name, module_name)
}

fn generated_residual_names(
    problem_key: &str,
    config: &SymbolicIvpGeneratedBackendConfig,
) -> (String, String) {
    let suffix = problem_key
        .chars()
        .take(16)
        .collect::<String>()
        .replace('-', "_");
    let crate_name = config
        .crate_name_override
        .clone()
        .unwrap_or_else(|| format!("generated_ivp_residual_{suffix}"));
    let module_name = config
        .module_name_override
        .clone()
        .unwrap_or_else(|| format!("generated_ivp_residual_module_{suffix}"));
    (crate_name, module_name)
}

fn generated_sparse_names(
    problem_key: &str,
    config: &SymbolicIvpGeneratedBackendConfig,
) -> (String, String) {
    let suffix = problem_key
        .chars()
        .take(16)
        .collect::<String>()
        .replace('-', "_");
    let crate_name = config
        .crate_name_override
        .clone()
        .unwrap_or_else(|| format!("generated_ivp_sparse_{suffix}"));
    let module_name = config
        .module_name_override
        .clone()
        .unwrap_or_else(|| format!("generated_ivp_sparse_module_{suffix}"));
    (crate_name, module_name)
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

fn select_backend(
    problem: &PreparedSymbolicIvpProblem,
    resolver: Option<&AotResolver>,
    options: SymbolicIvpAotOptions,
) -> SelectedSymbolicIvpBackendKind {
    let prepared = problem.prepare_dense_aot_problem(options);
    let problem_key = prepared.problem_key();
    if let Some(linked) = resolve_linked_dense_backend(problem_key.as_str()) {
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
    if let Some(linked) = resolve_linked_sparse_backend(problem_key.as_str()) {
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

fn should_build_for_selection(
    config: &SymbolicIvpGeneratedBackendConfig,
    selected_backend: SelectedSymbolicIvpBackendKind,
) -> bool {
    match config.build_policy {
        SymbolicIvpAotBuildPolicy::UseIfAvailable | SymbolicIvpAotBuildPolicy::RequirePrebuilt => {
            false
        }
        SymbolicIvpAotBuildPolicy::BuildIfMissing { .. } => {
            selected_backend != SelectedSymbolicIvpBackendKind::AotCompiled
        }
        SymbolicIvpAotBuildPolicy::RebuildAlways { .. } => true,
    }
}

fn build_profile(policy: SymbolicIvpAotBuildPolicy) -> Option<AotBuildProfile> {
    match policy {
        SymbolicIvpAotBuildPolicy::BuildIfMissing { profile }
        | SymbolicIvpAotBuildPolicy::RebuildAlways { profile } => Some(profile),
        SymbolicIvpAotBuildPolicy::UseIfAvailable | SymbolicIvpAotBuildPolicy::RequirePrebuilt => {
            None
        }
    }
}

fn build_preset(policy: SymbolicIvpAotBuildPolicy) -> Option<AotBuildPreset> {
    match build_profile(policy) {
        Some(AotBuildProfile::Debug) => Some(AotBuildPreset::DevFastest),
        Some(AotBuildProfile::Release) => Some(AotBuildPreset::Production),
        None => None,
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
                SymbolicIvpGeneratedError::AotBuildFailed(runtime_registration_error_message(
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
                SymbolicIvpGeneratedError::AotBuildFailed(runtime_registration_error_message(
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
                SymbolicIvpGeneratedError::AotBuildFailed(runtime_registration_error_message(
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
                SymbolicIvpGeneratedError::AotBuildFailed(runtime_registration_error_message(
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
                SymbolicIvpGeneratedError::AotBuildFailed(runtime_registration_error_message(
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
                SymbolicIvpGeneratedError::AotBuildFailed(runtime_registration_error_message(
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
                SymbolicIvpGeneratedError::AotBuildFailed(runtime_registration_error_message(
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
                SymbolicIvpGeneratedError::AotBuildFailed(runtime_registration_error_message(
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
                SymbolicIvpGeneratedError::AotBuildFailed(runtime_registration_error_message(
                    route,
                    backend,
                    problem_key,
                    &artifact_path,
                    err,
                ))
            }),
    }
}

fn perform_requested_build(
    problem: &PreparedSymbolicIvpProblem,
    config: &SymbolicIvpGeneratedBackendConfig,
    resolver_snapshot: Option<AotResolver>,
) -> Result<(Option<GeneratedAotBuildResult>, Option<AotResolver>), SymbolicIvpGeneratedError> {
    let preset = match build_preset(config.build_policy) {
        Some(preset) => preset,
        None => return Ok((None, resolver_snapshot)),
    };

    let prepared = problem.prepare_dense_aot_problem(config.aot_options);
    let (crate_name, module_name) = generated_names(&prepared.problem_key(), config);
    info!(
        "Materializing symbolic IVP dense {:?} AOT build '{}' with preset {:?}",
        config.aot_codegen_backend, crate_name, preset
    );
    let artifact = generated_aot_artifact_from_symbolic_ivp_problem(
        &crate_name,
        &module_name,
        problem,
        config.aot_options,
        config.aot_codegen_backend,
    );
    let output_parent_dir =
        output_parent_dir_for_requested_build(config, "dense", prepared.problem_key().as_str())?;
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
        .map_err(|err| SymbolicIvpGeneratedError::AotBuildFailed(err.to_string()))?;

    execute_generated_build_with_retry(
        &build,
        &format!(
            "ivp-dense backend={:?} key={}",
            config.aot_codegen_backend,
            prepared.problem_key()
        ),
    )?;

    let resolver =
        register_ivp_build_result_in_registry(resolver_snapshot, prepared.manifest(), &build)?;
    register_ivp_runtime_backend(
        "dense",
        config.aot_codegen_backend,
        &resolver,
        prepared.problem_key().as_str(),
    )?;
    Ok((Some(build), Some(resolver)))
}

fn perform_requested_residual_build(
    problem: &PreparedSymbolicIvpResidualProblem,
    config: &SymbolicIvpGeneratedBackendConfig,
    resolver_snapshot: Option<AotResolver>,
) -> Result<(Option<GeneratedAotBuildResult>, Option<AotResolver>), SymbolicIvpGeneratedError> {
    let preset = match build_preset(config.build_policy) {
        Some(preset) => preset,
        None => return Ok((None, resolver_snapshot)),
    };

    let prepared = problem.prepare_residual_aot_problem(config.aot_options);
    let (crate_name, module_name) = generated_residual_names(&prepared.problem_key(), config);
    info!(
        "Materializing symbolic IVP residual-only {:?} AOT build '{}' with preset {:?}",
        config.aot_codegen_backend, crate_name, preset
    );
    let artifact = generated_aot_artifact_from_symbolic_ivp_residual_problem(
        &crate_name,
        &module_name,
        problem,
        config.aot_options,
        config.aot_codegen_backend,
    );
    let output_parent_dir = output_parent_dir_for_requested_build(
        config,
        "residual-only",
        prepared.problem_key().as_str(),
    )?;
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
        .map_err(|err| SymbolicIvpGeneratedError::AotBuildFailed(err.to_string()))?;

    execute_generated_build_with_retry(
        &build,
        &format!(
            "ivp-residual backend={:?} key={}",
            config.aot_codegen_backend,
            prepared.problem_key()
        ),
    )?;

    let resolver =
        register_ivp_build_result_in_registry(resolver_snapshot, prepared.manifest(), &build)?;
    register_ivp_residual_runtime_backend(
        "residual-only",
        config.aot_codegen_backend,
        &resolver,
        prepared.problem_key().as_str(),
    )?;
    Ok((Some(build), Some(resolver)))
}

fn perform_requested_sparse_build(
    problem: &PreparedSparseProblem<'_>,
    config: &SymbolicIvpGeneratedBackendConfig,
    resolver_snapshot: Option<AotResolver>,
) -> Result<(Option<GeneratedAotBuildResult>, Option<AotResolver>), SymbolicIvpGeneratedError> {
    let preset = match build_preset(config.build_policy) {
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
        .map_err(|err| SymbolicIvpGeneratedError::AotBuildFailed(err.to_string()))?;

    execute_generated_build_with_retry(
        &build,
        &format!(
            "ivp-sparse backend={:?} key={}",
            config.aot_codegen_backend, problem_key
        ),
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
    let baseline_problem = prepare_symbolic_ivp_problem(equations, variables, time_arg, options)?;
    let variable_refs = baseline_problem
        .variables
        .iter()
        .map(|value| value.as_str())
        .collect::<Vec<_>>();
    let parameter_refs = baseline_problem
        .equation_parameters
        .as_ref()
        .map(|parameters| {
            parameters
                .iter()
                .map(|value| value.as_str())
                .collect::<Vec<_>>()
        });
    let mut sparse_param_refs = Vec::with_capacity(1 + parameter_refs.as_ref().map_or(0, Vec::len));
    sparse_param_refs.push(baseline_problem.time_arg.as_str());
    if let Some(params) = parameter_refs.as_ref() {
        sparse_param_refs.extend(params.iter().copied());
    }

    let sparse_entries = baseline_problem
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
        time_arg: baseline_problem.time_arg.as_str(),
        residuals: &baseline_problem.equations,
        variables: &variable_refs,
        params: parameter_refs.as_deref(),
    }
    .runtime_plan(config.aot_options.residual_strategy);

    let shape = (
        baseline_problem.symbolic_jacobian.len(),
        baseline_problem
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

    let initial_selection = select_sparse_backend(&prepared_sparse, config.resolver.as_ref());
    let (build_result, resolver_snapshot) =
        if should_build_for_selection(&config, initial_selection) {
            perform_requested_sparse_build(&prepared_sparse, &config, config.resolver.clone())?
        } else {
            (None, config.resolver.clone())
        };

    let final_selection = select_sparse_backend(&prepared_sparse, resolver_snapshot.as_ref());
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
            Ok(PreparedGeneratedSymbolicIvpSparseBackend {
                problem_key,
                selected_backend: final_selection,
                linked_backend,
                jacobian_structure,
                updated_resolver: resolver_snapshot,
                build_result,
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
                    updated_resolver: resolver_snapshot,
                    build_result,
                }),
            }
        }
    }
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
    let initial_selection = select_backend(
        &baseline_problem,
        config.resolver.as_ref(),
        config.aot_options,
    );
    let (build_result, resolver_snapshot) =
        if should_build_for_selection(&config, initial_selection) {
            perform_requested_build(&baseline_problem, &config, config.resolver.clone())?
        } else {
            (None, config.resolver.clone())
        };

    let final_selection = select_backend(
        &baseline_problem,
        resolver_snapshot.as_ref(),
        config.aot_options,
    );
    let problem_key = baseline_problem
        .prepare_dense_aot_problem(config.aot_options)
        .problem_key();
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
                }),
            }
        }
        SelectedSymbolicIvpBackendKind::AotRegisteredButNotBuilt => {
            Err(SymbolicIvpGeneratedError::CompiledAotArtifactNotBuilt(
                not_built_aot_message("dense", problem_key.as_str(), &config),
            ))
        }
        SelectedSymbolicIvpBackendKind::AotCompiled => {
            let prepared = baseline_problem.prepare_dense_aot_problem(config.aot_options);
            let problem_key = prepared.problem_key();
            if let Some(linked) = resolve_linked_dense_backend(problem_key.as_str()) {
                Ok(PreparedGeneratedSymbolicIvpProblem {
                    problem: baseline_problem.into_linked_dense_backend(linked),
                    selected_backend: SelectedSymbolicIvpBackendKind::AotCompiled,
                    updated_resolver: resolver_snapshot,
                    build_result,
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
    let baseline_problem =
        prepare_symbolic_ivp_residual_problem(equations, variables, time_arg, options)?;
    let initial_selection = select_residual_backend(
        &baseline_problem,
        config.resolver.as_ref(),
        config.aot_options,
    );
    let (build_result, resolver_snapshot) =
        if should_build_for_selection(&config, initial_selection) {
            perform_requested_residual_build(&baseline_problem, &config, config.resolver.clone())?
        } else {
            (None, config.resolver.clone())
        };

    let final_selection = select_residual_backend(
        &baseline_problem,
        resolver_snapshot.as_ref(),
        config.aot_options,
    );
    let problem_key = baseline_problem
        .prepare_residual_aot_problem(config.aot_options)
        .problem_key();
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
            if let Some(linked) = resolve_linked_residual_backend(problem_key.as_str()) {
                Ok(PreparedGeneratedSymbolicIvpResidualProblem {
                    problem: baseline_problem.into_linked_residual_backend(linked),
                    selected_backend: SelectedSymbolicIvpBackendKind::AotCompiled,
                    updated_resolver: resolver_snapshot,
                    build_result,
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
                        warn!(
                            "Symbolic IVP residual-only compiled AOT artifact exists but no linked runtime is registered; falling back to lambdify"
                        );
                        Ok(PreparedGeneratedSymbolicIvpResidualProblem {
                            problem: baseline_problem,
                            selected_backend: SelectedSymbolicIvpBackendKind::Lambdify,
                            updated_resolver: resolver_snapshot,
                            build_result,
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
    use crate::symbolic::codegen::codegen_aot_runtime_link::{
        register_linked_dense_backend, unregister_linked_dense_backend, LinkedDenseAotBackend,
    };
    use crate::symbolic::symbolic_engine::Expr;
    use crate::symbolic::symbolic_ivp::IvpBackendKind;
    use nalgebra::DVector;
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
        assert!(load.contains("symbolic IVP dense"));
        assert!(load.contains("problem_key=abc123"));
        assert!(load.contains("artifact_path=target/example.dll"));
        assert!(load.contains("stale/incompatible"));
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
        let resolver = prepared
            .updated_resolver
            .expect("build should update resolver");
        assert_eq!(resolver.registry().len(), 1);
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
