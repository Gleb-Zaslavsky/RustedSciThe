use crate::numerical::BVP_Damp::BVP_traits::{Fun, Jac};
use crate::somelinalg::banded::LinearSolverConfig;
use crate::symbolic::codegen::c_backend::codegen_c_aot_build::{
    CAotBuildProfile, CAotBuildRequest, CAotCompileConfig,
};
use crate::symbolic::codegen::c_backend::codegen_c_aot_registry::register_c_build_in_registry;
use crate::symbolic::codegen::c_backend::codegen_c_aot_runtime_link::{
    register_generated_c_banded_backend, register_generated_c_sparse_backend,
};
use crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend;
use crate::symbolic::codegen::codegen_aot_registry::{AotRegistry, RegisteredAotArtifact};
use crate::symbolic::codegen::codegen_aot_resolution::AotResolver;
use crate::symbolic::codegen::codegen_aot_runtime_link::{
    register_generated_banded_cdylib_backend, register_generated_sparse_cdylib_backend,
    resolve_linked_sparse_backend, unregister_linked_sparse_backend,
};
use crate::symbolic::codegen::codegen_backend_selection::{
    BackendSelectionPolicy, SelectedBackendKind,
};
use crate::symbolic::codegen::codegen_manifest::PreparedProblemManifest;
use crate::symbolic::codegen::codegen_orchestrator::ParallelExecutorConfig;
use crate::symbolic::codegen::codegen_provider_api::{BackendKind, MatrixBackend};
use crate::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy;
use crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy;
use crate::symbolic::codegen::rust_backend::codegen_aot_build::{
    AotBuildProfile as LifecycleBuildProfile, AotBuildRequest,
    AotCompileConfig as LifecycleAotCompileConfig,
};
use crate::symbolic::codegen::zig_backend::codegen_zig_aot_build::{
    ZigAotBuildProfile, ZigAotBuildRequest,
};
use crate::symbolic::codegen::zig_backend::codegen_zig_aot_registry::register_zig_build_in_registry;
use crate::symbolic::codegen::zig_backend::codegen_zig_aot_runtime_link::{
    register_generated_zig_banded_backend, register_generated_zig_sparse_backend,
};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions_BVP::{
    BvpBackendIntegrationError, BvpGeneratedAotCrateBreakdown, BvpLegacySolverBundle,
    BvpSparseExecutionPlan, BvpSparseSolverBundle, BvpSymbolicAssemblyBackend, Jacobian,
};
use log::{error, info};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::thread::sleep;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

fn bvp_aot_lifecycle_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn insert_elapsed_ms(diagnostics: &mut HashMap<String, String>, key: &str, began: Instant) {
    diagnostics.insert(
        key.to_string(),
        format!("{:.6}", began.elapsed().as_secs_f64() * 1_000.0),
    );
}

fn copy_prepare_diagnostics(
    target: &mut HashMap<String, String>,
    bundle: &BvpSparseSolverBundle,
    phase: &str,
) {
    for (key, value) in bundle.runtime_diagnostics() {
        if let Some(stage) = key.strip_prefix("generated.prepare.") {
            target.insert(format!("generated.handoff.{phase}.{stage}"), value.clone());
        }
    }
}

fn append_aot_artifact_breakdown(
    diagnostics: &mut HashMap<String, String>,
    breakdown: &BvpGeneratedAotCrateBreakdown,
) {
    for (key, value) in [
        ("module_ms", breakdown.module_build_ms),
        ("module_init_ms", breakdown.prepared_module_init_ms),
        ("residual_lower_ms", breakdown.prepared_residual_blocks_ms),
        ("jacobian_lower_ms", breakdown.prepared_jacobian_blocks_ms),
        ("source_emit_ms", breakdown.language_source_emit_ms),
        ("c_header_ms", breakdown.c_header_emit_ms),
        ("packaging_ms", breakdown.artifact_packaging_ms),
    ] {
        diagnostics.insert(
            format!("generated.aot.artifact.{key}"),
            format!("{value:.6}"),
        );
    }
}

fn append_registered_artifact_contract(
    diagnostics: &mut HashMap<String, String>,
    artifact: &RegisteredAotArtifact,
) {
    diagnostics.insert(
        "generated.aot.artifact.problem_key".to_string(),
        artifact.problem_key.clone(),
    );
    diagnostics.insert(
        "generated.aot.artifact.manifest_key".to_string(),
        artifact.manifest_problem_key(),
    );
    diagnostics.insert(
        "generated.aot.artifact.manifest_key_matches".to_string(),
        artifact.manifest_key_matches().to_string(),
    );
    diagnostics.insert(
        "generated.aot.artifact.manifest_file".to_string(),
        artifact.manifest_file.display().to_string(),
    );
    diagnostics.insert(
        "generated.aot.artifact.manifest_file_exists".to_string(),
        artifact.manifest_file_exists().to_string(),
    );
    diagnostics.insert(
        "generated.aot.artifact.expected_cdylib".to_string(),
        artifact.expected_cdylib.display().to_string(),
    );
    diagnostics.insert(
        "generated.aot.artifact.expected_cdylib_exists".to_string(),
        artifact.expected_cdylib.exists().to_string(),
    );
    diagnostics.insert(
        "generated.aot.artifact.expected_rlib".to_string(),
        artifact.expected_rlib.display().to_string(),
    );
    diagnostics.insert(
        "generated.aot.artifact.expected_rlib_exists".to_string(),
        artifact.expected_rlib.exists().to_string(),
    );
    diagnostics.insert(
        "generated.aot.artifact.contract_issues".to_string(),
        artifact.lifecycle_contract_issues().join(" | "),
    );
}

fn is_transient_aot_infra_failure(text: &str) -> bool {
    let low = text.to_ascii_lowercase();
    low.contains("permission denied")
        || low.contains("access is denied")
        || low.contains("being used by another process")
        || low.contains("resource busy")
        || low.contains("temporarily unavailable")
        || low.contains("failed to spawn")
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
        "BVP AOT build execution failed ({build_context}) after {attempts} attempt(s); classified as {class}. \
If this is a transient infrastructure failure, check stale compiler processes, file locks and antivirus/indexer interference; \
otherwise inspect compiler stdout/stderr below.\n\
detail:\n{detail}"
    )
}

fn execute_aot_build_with_retry<F>(mut execute: F, build_context: &str) -> Result<(), String>
where
    F: FnMut() -> Result<(bool, Option<i32>, String, String), String>,
{
    const MAX_ATTEMPTS: usize = 3;
    let mut last_failure: Option<String> = None;
    let mut last_transient = false;
    let mut attempts_used = 0usize;

    for attempt in 1..=MAX_ATTEMPTS {
        attempts_used = attempt;
        match execute() {
            Ok((true, _, _, _)) => return Ok(()),
            Ok((false, status, stdout, stderr)) => {
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
                let transient = is_transient_aot_infra_failure(&err);
                last_transient = transient;
                last_failure = Some(err);
                if transient && attempt < MAX_ATTEMPTS {
                    sleep(Duration::from_millis((attempt as u64) * 120));
                    continue;
                }
            }
        }
        break;
    }

    let detail = last_failure.unwrap_or_else(|| "unknown build failure".to_string());
    Err(retry_exhausted_aot_message(
        build_context,
        attempts_used,
        &detail,
        last_transient,
    ))
}

/// Unified callback/metadata handoff for the damped sparse BVP solver.
pub struct DampedGeneratedSolverState {
    /// Residual callback ready for the solver runtime loop.
    pub fun: Box<dyn Fun>,
    /// Jacobian callback ready for the solver runtime loop.
    pub jac: Option<Box<dyn Jac>>,
    /// Per-unknown bounds on the discretized state vector.
    pub bounds_vec: Vec<(f64, f64)>,
    /// Per-unknown relative tolerances on the discretized state vector.
    pub rel_tolerance_vec: Vec<f64>,
    /// Flattened symbolic variable names in solver input order.
    pub variable_string: Vec<String>,
    /// Jacobian bandwidth metadata used by sparse linear solves.
    pub bandwidth: (usize, usize),
    /// Boundary condition positions and values in the flattened state vector.
    pub bc_position_and_value: Vec<(usize, usize, f64)>,
    /// Updated resolver snapshot that includes any newly materialized AOT artifact.
    pub updated_resolver: Option<AotResolver>,
    /// Backend branch that actually supplied the runtime callbacks.
    pub selected_backend: SelectedBackendKind,
    /// Runtime diagnostics for generated callback execution.
    pub runtime_diagnostics: HashMap<String, String>,
}

/// Unified callback/metadata handoff for the frozen sparse BVP solver.
pub struct FrozenGeneratedSolverState {
    /// Residual callback ready for the solver runtime loop.
    pub fun: Box<dyn Fun>,
    /// Jacobian callback ready for the solver runtime loop.
    pub jac: Option<Box<dyn Jac>>,
    /// Flattened symbolic variable names in solver input order.
    pub variable_string: Vec<String>,
    /// Jacobian bandwidth metadata used by sparse linear solves.
    pub bandwidth: (usize, usize),
    /// Updated resolver snapshot that includes any newly materialized AOT artifact.
    pub updated_resolver: Option<AotResolver>,
    /// Backend branch that actually supplied the runtime callbacks.
    pub selected_backend: SelectedBackendKind,
    /// Runtime diagnostics for generated callback execution.
    pub runtime_diagnostics: HashMap<String, String>,
}

/// Applies a damped generated handoff state to a solver-specific runtime object.
pub trait ApplyDampedGeneratedSolverState {
    /// Stores callbacks and metadata produced by the generated handoff layer.
    fn apply_generated_solver_state(&mut self, state: DampedGeneratedSolverState);
}

/// Applies a frozen generated handoff state to a solver-specific runtime object.
pub trait ApplyFrozenGeneratedSolverState {
    /// Stores callbacks and metadata produced by the generated handoff layer.
    fn apply_generated_solver_state(&mut self, state: FrozenGeneratedSolverState);
}

/// Builds a complete damped solver handoff request from a solver runtime object.
pub trait BuildDampedSolverRequest {
    /// Creates the symbolic-to-generated request consumed by the shared handoff layer.
    fn build_solver_request(
        &mut self,
        mesh: Option<Vec<f64>>,
        bandwidth: Option<(usize, usize)>,
    ) -> DampedSolverBuildRequest;
}

/// Builds a complete frozen solver handoff request from a solver runtime object.
pub trait BuildFrozenSolverRequest {
    /// Creates the symbolic-to-generated request consumed by the shared handoff layer.
    fn build_solver_request(&self) -> FrozenSolverBuildRequest;
}

/// User-facing configuration for generated backend selection in BVP solvers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AotBuildProfile {
    #[default]
    Release,
    Debug,
}

/// Solver-facing rustc/codegen configuration for generated AOT crate builds.
pub type AotCompileConfig = LifecycleAotCompileConfig;

/// Solver-level build policy for generated AOT artifacts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AotBuildPolicy {
    #[default]
    UseIfAvailable,
    BuildIfMissing {
        profile: AotBuildProfile,
    },
    RequirePrebuilt,
    RebuildAlways {
        profile: AotBuildProfile,
    },
}

impl AotBuildPolicy {
    /// Short stable label for logging and user-facing diagnostics.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::UseIfAvailable => "UseIfAvailable",
            Self::BuildIfMissing { .. } => "BuildIfMissing",
            Self::RequirePrebuilt => "RequirePrebuilt",
            Self::RebuildAlways { .. } => "RebuildAlways",
        }
    }
}

/// Solver-level execution policy for compiled AOT callbacks.
#[derive(Clone, Debug, PartialEq, Default)]
pub enum AotExecutionPolicy {
    #[default]
    Auto,
    SequentialOnly,
    Parallel(ParallelExecutorConfig),
}

impl AotExecutionPolicy {
    /// Short stable label for logging and user-facing diagnostics.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::SequentialOnly => "SequentialOnly",
            Self::Parallel(_) => "Parallel",
        }
    }
}

/// Optional chunking overrides surfaced at solver setup level.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct AotChunkingPolicy {
    /// Optional residual chunking override.
    pub residual: Option<ResidualChunkingStrategy>,
    /// Optional sparse Jacobian chunking override.
    pub sparse_jacobian: Option<SparseChunkingStrategy>,
}

impl AotChunkingPolicy {
    /// Creates an empty chunking policy that keeps backend defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a policy from explicit residual and sparse Jacobian strategies.
    pub fn with_parts(
        residual: Option<ResidualChunkingStrategy>,
        sparse_jacobian: Option<SparseChunkingStrategy>,
    ) -> Self {
        Self {
            residual,
            sparse_jacobian,
        }
    }
}

/// User-facing configuration for generated backend selection in BVP solvers.
#[derive(Clone, Default)]
pub struct GeneratedBackendConfig {
    /// Optional explicit backend policy override.
    pub backend_policy_override: Option<BackendSelectionPolicy>,
    /// Optional compiled AOT resolver snapshot.
    pub resolver: Option<AotResolver>,
    /// Solver-level execution policy for compiled AOT callbacks.
    pub aot_execution_policy: AotExecutionPolicy,
    /// Solver-level build policy for generated AOT artifacts.
    pub aot_build_policy: AotBuildPolicy,
    /// Optional compile-time rustc/codegen overrides for generated AOT artifacts.
    pub aot_compile_config: AotCompileConfig,
    /// Codegen backend used to emit generated AOT artifacts.
    pub aot_codegen_backend: AotCodegenBackend,
    /// Optional explicit C compiler for C AOT backends, e.g. `gcc` or `tcc`.
    pub aot_c_compiler: Option<String>,
    /// Optional chunking overrides for residual and sparse Jacobian generation.
    pub aot_chunking_policy: AotChunkingPolicy,
    /// Symbolic assembly backend used before lambdify/AOT lowering.
    pub symbolic_assembly_backend: BvpSymbolicAssemblyBackend,
    /// Optional matrix backend override for the modern generated BVP path.
    pub matrix_backend_override: Option<MatrixBackend>,
    /// Native linear solver configuration used by the generated banded runtime path.
    pub banded_linear_solver_config: LinearSolverConfig,
}

/// High-level sparse generated-backend modes exposed at solver setup level.
///
/// These modes intentionally hide backend-selection and build-policy details from
/// typical solver users while still allowing advanced callers to drop down to
/// [`GeneratedBackendConfig`] when they need finer control.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SparseGeneratedBackendMode {
    /// Prefer compiled AOT when available and otherwise fall back to lambdify.
    #[default]
    Defaults,
    /// Require a previously built sparse AOT artifact.
    RequirePrebuilt,
    /// Build a release sparse AOT artifact when it is missing.
    BuildIfMissingRelease,
}

impl SparseGeneratedBackendMode {
    /// Converts the high-level sparse mode into a concrete generated-backend configuration.
    pub fn generated_backend_config(self) -> GeneratedBackendConfig {
        match self {
            Self::Defaults => GeneratedBackendConfig::sparse_defaults(),
            Self::RequirePrebuilt => GeneratedBackendConfig::sparse_require_prebuilt(),
            Self::BuildIfMissingRelease => {
                GeneratedBackendConfig::sparse_build_if_missing_release()
            }
        }
    }
}

/// High-level banded generated-backend modes exposed at solver setup level.
///
/// These presets select the `Banded` matrix backend and route native linear
/// solves to the faithful LAPACK-style banded LU solver with `refine = 0`.
/// Advanced users can still override compiler, chunking, build policy, or the
/// native linear solver through [`GeneratedBackendConfig`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BandedGeneratedBackendMode {
    /// Prefer compiled AOT when available and otherwise fall back to lambdify.
    #[default]
    Defaults,
    /// Force the lambdify callback path while keeping the native banded matrix
    /// and faithful LAPACK-style linear solver backend.
    Lambdify,
    /// Build a release AOT artifact when it is missing.
    BuildIfMissingRelease,
}

impl BandedGeneratedBackendMode {
    /// Converts the high-level banded mode into a concrete generated-backend configuration.
    pub fn generated_backend_config(self) -> GeneratedBackendConfig {
        match self {
            Self::Defaults => GeneratedBackendConfig::banded_defaults(),
            Self::Lambdify => GeneratedBackendConfig::banded_lambdify_defaults(),
            Self::BuildIfMissingRelease => {
                GeneratedBackendConfig::banded_build_if_missing_release()
            }
        }
    }
}

impl GeneratedBackendConfig {
    /// Creates an empty generated-backend configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates the default sparse/BVP generated-backend configuration.
    ///
    /// Practical guidance:
    /// - good general default when you do not want to commit to a specific backend,
    /// - uses `AtomView`, whose sparse-first symbolic Jacobian route removes
    ///   the legacy row-differentiation bottleneck on measured BVP systems,
    /// - still prefers compiled AOT when available,
    /// - otherwise falls back to the established lambdify path.
    pub fn sparse_defaults() -> Self {
        Self::new()
            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
            .with_backend_policy_override(Some(BackendSelectionPolicy::PreferAotThenLambdify))
    }

    /// Creates a sparse generated-backend configuration that requires a prebuilt AOT artifact.
    pub fn sparse_require_prebuilt() -> Self {
        Self::sparse_defaults().with_aot_build_policy(AotBuildPolicy::RequirePrebuilt)
    }

    /// Creates a sparse generated-backend configuration that builds a release artifact on demand.
    ///
    /// Practical guidance:
    /// - best fit for interactive "build on first use" workflows,
    /// - defaults to the fast `DevFastest` compile preset,
    /// - still backend-agnostic until you explicitly choose Rust/C/Zig.
    pub fn sparse_build_if_missing_release() -> Self {
        Self::sparse_defaults()
            .with_aot_build_policy(AotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Release,
            })
            // Build-if-missing is primarily an interactive/on-demand workflow, so default to
            // the fastest practical compile preset instead of the heaviest production codegen.
            .with_aot_compile_dev_fastest()
    }

    /// Creates a generated-backend configuration from a high-level sparse mode.
    pub fn from_sparse_mode(mode: SparseGeneratedBackendMode) -> Self {
        mode.generated_backend_config()
    }

    /// Creates the default banded/BVP generated-backend configuration.
    ///
    /// Practical guidance:
    /// - selects the generated `Banded` matrix path,
    /// - uses faithful LAPACK-style banded LU as the native linear solver,
    /// - uses `AtomView`, whose sparse-first symbolic Jacobian route avoids the
    ///   expensive legacy row-differentiation pass on banded BVP systems,
    /// - keeps `refine = 0` because current BVP workloads do not benefit from
    ///   the extra correction pass,
    /// - prefers compiled AOT when available and falls back to lambdify.
    pub fn banded_defaults() -> Self {
        Self::sparse_defaults()
            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
            .with_matrix_backend_override(MatrixBackend::Banded)
            .with_banded_linear_solver_config(LinearSolverConfig::faithful_banded())
    }

    /// Creates a banded configuration that explicitly uses lambdify callbacks.
    ///
    /// This is the simple callback path: `AtomView` generated `Banded` matrix
    /// assembly plus faithful LAPACK-style native banded solves. Callers that
    /// need compatibility comparison can explicitly override `ExprLegacy`.
    pub fn banded_lambdify_defaults() -> Self {
        Self::banded_defaults()
            .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
    }

    /// Creates a banded configuration that builds a release AOT artifact on demand.
    pub fn banded_build_if_missing_release() -> Self {
        Self::banded_defaults()
            .with_aot_build_policy(AotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Release,
            })
            .with_aot_compile_dev_fastest()
    }

    /// Creates a generated-backend configuration from a high-level banded mode.
    pub fn from_banded_mode(mode: BandedGeneratedBackendMode) -> Self {
        mode.generated_backend_config()
    }

    /// Creates a configuration from explicit policy and resolver values.
    pub fn with_parts(
        backend_policy_override: Option<BackendSelectionPolicy>,
        resolver: Option<AotResolver>,
    ) -> Self {
        Self {
            backend_policy_override,
            resolver,
            aot_execution_policy: AotExecutionPolicy::Auto,
            aot_build_policy: AotBuildPolicy::UseIfAvailable,
            aot_compile_config: AotCompileConfig::default(),
            aot_codegen_backend: AotCodegenBackend::Rust,
            aot_c_compiler: None,
            aot_chunking_policy: AotChunkingPolicy::default(),
            symbolic_assembly_backend: BvpSymbolicAssemblyBackend::ExprLegacy,
            matrix_backend_override: None,
            banded_linear_solver_config: crate::somelinalg::banded::LinearSolverConfig::default(),
        }
    }

    /// Sets an explicit backend policy override.
    pub fn with_backend_policy_override(
        mut self,
        backend_policy_override: Option<BackendSelectionPolicy>,
    ) -> Self {
        self.backend_policy_override = backend_policy_override;
        self
    }

    /// Sets an optional compiled AOT resolver snapshot.
    pub fn with_resolver(mut self, resolver: Option<AotResolver>) -> Self {
        self.resolver = resolver;
        self
    }

    /// Sets the AOT execution policy exposed at solver level.
    pub fn with_aot_execution_policy(mut self, policy: AotExecutionPolicy) -> Self {
        self.aot_execution_policy = policy;
        self
    }

    /// Sets the AOT build policy exposed at solver level.
    pub fn with_aot_build_policy(mut self, policy: AotBuildPolicy) -> Self {
        self.aot_build_policy = policy;
        self
    }

    /// Sets compile-time rustc/codegen overrides for generated AOT artifacts.
    pub fn with_aot_compile_config(mut self, config: AotCompileConfig) -> Self {
        self.aot_compile_config = config;
        self
    }

    /// Selects the backend used to emit generated AOT artifacts.
    pub fn with_aot_codegen_backend(mut self, backend: AotCodegenBackend) -> Self {
        self.aot_codegen_backend = backend;
        self
    }

    /// Selects an explicit C compiler for C AOT backends.
    pub fn with_aot_c_compiler(mut self, compiler: impl Into<String>) -> Self {
        self.aot_c_compiler = Some(compiler.into());
        self
    }

    /// Uses AtomView symbolic assembly plus on-demand `gcc`-compiled C AOT.
    ///
    /// Practical guidance:
    /// - strong choice when runtime throughput matters more than bootstrap latency,
    /// - especially useful for repeated solves on the same symbolic problem.
    pub fn sparse_atomview_build_if_missing_release_gcc() -> Self {
        Self::sparse_build_if_missing_release()
            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
            .with_aot_codegen_backend(AotCodegenBackend::C)
            .with_aot_c_compiler("gcc")
    }

    /// Uses AtomView symbolic assembly plus on-demand `tcc`-compiled C AOT.
    ///
    /// Practical guidance:
    /// - strongest compiled choice for low-latency bootstrap,
    /// - currently the most practical repeated-solve backend once you expect
    ///   roughly `2-3` solves or more on large combustion-style BVPs.
    pub fn sparse_atomview_build_if_missing_release_tcc() -> Self {
        Self::sparse_build_if_missing_release()
            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
            .with_aot_codegen_backend(AotCodegenBackend::C)
            .with_aot_c_compiler("tcc")
    }

    /// Uses AtomView symbolic assembly plus on-demand Zig-compiled sparse AOT.
    pub fn sparse_atomview_build_if_missing_release_zig() -> Self {
        Self::sparse_build_if_missing_release()
            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
            .with_aot_codegen_backend(AotCodegenBackend::Zig)
    }

    /// User-facing alias for the practical repeated-solve recommendation:
    /// AtomView symbolic assembly plus `tcc`-compiled C AOT.
    pub fn sparse_atomview_for_repeated_solves() -> Self {
        Self::sparse_atomview_build_if_missing_release_tcc()
    }

    /// Uses AtomView symbolic assembly plus on-demand `gcc`-compiled banded C AOT.
    pub fn banded_atomview_build_if_missing_release_gcc() -> Self {
        Self::banded_build_if_missing_release()
            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
            .with_aot_codegen_backend(AotCodegenBackend::C)
            .with_aot_c_compiler("gcc")
    }

    /// Uses AtomView symbolic assembly plus on-demand `tcc`-compiled banded C AOT.
    ///
    /// This is usually the quickest compiled bootstrap path for large BVP
    /// experiments when the C toolchain is available.
    pub fn banded_atomview_build_if_missing_release_tcc() -> Self {
        Self::banded_build_if_missing_release()
            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
            .with_aot_codegen_backend(AotCodegenBackend::C)
            .with_aot_c_compiler("tcc")
    }

    /// Uses AtomView symbolic assembly plus on-demand Zig-compiled banded AOT.
    pub fn banded_atomview_build_if_missing_release_zig() -> Self {
        Self::banded_build_if_missing_release()
            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
            .with_aot_codegen_backend(AotCodegenBackend::Zig)
    }

    /// User-facing alias for the practical repeated-solve banded recommendation:
    /// AtomView symbolic assembly plus `tcc`-compiled C AOT, backed by faithful
    /// LAPACK-style native banded LU.
    pub fn banded_atomview_for_repeated_solves() -> Self {
        Self::banded_atomview_build_if_missing_release_tcc()
    }

    /// Uses the default production-oriented compile settings for generated AOT artifacts.
    pub fn with_aot_compile_production(self) -> Self {
        self.with_aot_compile_config(AotCompileConfig::production())
    }

    /// Uses the faster-build compromise preset for generated AOT artifacts.
    pub fn with_aot_compile_fast_build(self) -> Self {
        self.with_aot_compile_config(AotCompileConfig::fast_build())
    }

    /// Uses the fastest developer-oriented compile preset for generated AOT artifacts.
    pub fn with_aot_compile_dev_fastest(self) -> Self {
        self.with_aot_compile_config(AotCompileConfig::dev_fastest())
    }

    /// Sets solver-level chunking overrides for generated AOT plans.
    pub fn with_aot_chunking_policy(mut self, policy: AotChunkingPolicy) -> Self {
        self.aot_chunking_policy = policy;
        self
    }

    /// Sets the symbolic assembly backend used before backend lowering.
    pub fn with_symbolic_assembly_backend(mut self, backend: BvpSymbolicAssemblyBackend) -> Self {
        self.symbolic_assembly_backend = backend;
        self
    }

    /// Overrides the matrix backend used by the generated BVP path.
    ///
    /// This keeps the outer solver method stable (`Sparse` on the user-facing
    /// options surface) while allowing the generated symbolic/codegen stack to
    /// target a different matrix representation such as native `Banded`.
    pub fn with_matrix_backend_override(mut self, backend: MatrixBackend) -> Self {
        self.matrix_backend_override = Some(backend);
        self
    }

    /// Overrides the native linear solver configuration used by generated banded callbacks.
    pub fn with_banded_linear_solver_config(mut self, config: LinearSolverConfig) -> Self {
        self.banded_linear_solver_config = config;
        self
    }

    /// Resolves the effective legacy method string seen by the symbolic BVP
    /// preparation layer.
    pub fn effective_method(&self, fallback_method: &str) -> String {
        self.matrix_backend_override
            .map(|backend| match backend {
                MatrixBackend::Banded => "Banded",
                MatrixBackend::Dense => "Dense",
                MatrixBackend::CsMat => "Sparse_1",
                MatrixBackend::CsMatrix => "Sparse_2",
                MatrixBackend::SparseCol | MatrixBackend::ValuesOnly => "Sparse",
            })
            .unwrap_or(fallback_method)
            .to_string()
    }

    /// Returns the effective backend policy for a given solver method.
    pub fn effective_backend_policy(&self, method: &str) -> BackendSelectionPolicy {
        self.backend_policy_override
            .unwrap_or_else(|| backend_policy_for_method(method))
    }
}

/// Input bundle used to build a complete damped solver handoff state.
pub struct DampedSolverBuildRequest {
    /// Symbolic right-hand sides before BVP discretization.
    pub eq_system: Vec<Expr>,
    /// Names of unknown variables.
    pub values: Vec<String>,
    /// Optional symbolic parameter names that affect evaluation but are not Newton unknowns.
    pub param_names: Option<Vec<String>>,
    /// Current numeric values for `param_names`.
    pub param_values: Option<Vec<f64>>,
    /// Independent variable name.
    pub arg: String,
    /// Left boundary value of the independent variable.
    pub t0: f64,
    /// Number of discretization steps when a uniform mesh is used.
    pub n_steps: Option<usize>,
    /// Uniform mesh spacing when the mesh is not given explicitly.
    pub h: Option<f64>,
    /// Optional user-supplied mesh.
    pub mesh: Option<Vec<f64>>,
    /// Boundary conditions passed to the BVP symbolic builder.
    pub border_conditions: HashMap<String, Vec<(usize, f64)>>,
    /// Optional per-variable bounds metadata.
    pub bounds: Option<HashMap<String, (f64, f64)>>,
    /// Optional per-variable relative tolerance metadata.
    pub rel_tolerance: Option<HashMap<String, f64>>,
    /// Discretization scheme name.
    pub scheme: String,
    /// Matrix backend/method selector used by the legacy BVP module.
    pub method: String,
    /// Optional sparse bandwidth hint.
    pub bandwidth: Option<(usize, usize)>,
    /// Preferred backend branch for sparse symbolic generation.
    pub backend_policy: BackendSelectionPolicy,
    /// Optional resolver snapshot used to detect compiled AOT artifacts.
    pub resolver: Option<AotResolver>,
    /// Solver-level execution policy carried into generated backend setup.
    pub aot_execution_policy: AotExecutionPolicy,
    /// Solver-level build policy carried into generated backend setup.
    pub aot_build_policy: AotBuildPolicy,
    /// Optional compile-time rustc/codegen overrides carried into generated backend setup.
    pub aot_compile_config: AotCompileConfig,
    /// Codegen backend used to emit generated AOT artifacts.
    pub aot_codegen_backend: AotCodegenBackend,
    /// Optional explicit C compiler for C AOT backends.
    pub aot_c_compiler: Option<String>,
    /// Optional chunking overrides carried into generated backend setup.
    pub aot_chunking_policy: AotChunkingPolicy,
    /// Symbolic assembly backend used before lambdify/AOT lowering.
    pub symbolic_assembly_backend: BvpSymbolicAssemblyBackend,
    /// Optional explicit matrix backend override for generated sparse/banded handoff.
    pub matrix_backend_override: Option<MatrixBackend>,
    /// Native linear solver configuration used by the generated banded runtime path.
    pub banded_linear_solver_config: LinearSolverConfig,
}

impl DampedSolverBuildRequest {
    /// Builds a solver-ready callback and metadata state.
    pub fn generate(self) -> Result<DampedGeneratedSolverState, BvpBackendIntegrationError> {
        generate_damped_solver_state(
            self.eq_system,
            self.values,
            self.param_names,
            self.param_values,
            self.arg,
            self.t0,
            self.n_steps,
            self.h,
            self.mesh,
            self.border_conditions,
            self.bounds,
            self.rel_tolerance,
            self.scheme,
            self.method,
            self.bandwidth,
            self.backend_policy,
            self.resolver.as_ref(),
            self.aot_execution_policy,
            self.aot_build_policy,
            self.aot_compile_config,
            self.aot_codegen_backend,
            self.aot_c_compiler,
            self.aot_chunking_policy,
            self.symbolic_assembly_backend,
            self.matrix_backend_override,
            self.banded_linear_solver_config,
        )
    }
}

/// Builds a damped solver state and applies it to the target runtime object.
pub fn try_build_and_apply_damped_solver_state<T: ApplyDampedGeneratedSolverState>(
    target: &mut T,
    request: DampedSolverBuildRequest,
    context: &str,
) -> Result<(), BvpBackendIntegrationError> {
    info!("{context}: generating damped solver state");
    match request.generate() {
        Ok(state) => {
            target.apply_generated_solver_state(state);
            info!("{context}: damped solver state applied");
            Ok(())
        }
        Err(err) => {
            error!("{context}: {err:?}");
            Err(err)
        }
    }
}

/// Builds a damped solver state and applies it to the target runtime object.
pub fn build_and_apply_damped_solver_state<T: ApplyDampedGeneratedSolverState>(
    target: &mut T,
    request: DampedSolverBuildRequest,
    context: &str,
) {
    try_build_and_apply_damped_solver_state(target, request, context)
        .unwrap_or_else(|err| panic!("{context}: {err:?}"));
}

/// Asks a solver runtime object to build its damped request, then generates and applies the state.
pub fn try_generate_and_apply_damped_solver_state<
    T: BuildDampedSolverRequest + ApplyDampedGeneratedSolverState,
>(
    target: &mut T,
    mesh: Option<Vec<f64>>,
    bandwidth: Option<(usize, usize)>,
    context: &str,
) -> Result<(), BvpBackendIntegrationError> {
    let request = target.build_solver_request(mesh, bandwidth);
    try_build_and_apply_damped_solver_state(target, request, context)
}

/// Asks a solver runtime object to build its damped request, then generates and applies the state.
pub fn generate_and_apply_damped_solver_state<
    T: BuildDampedSolverRequest + ApplyDampedGeneratedSolverState,
>(
    target: &mut T,
    mesh: Option<Vec<f64>>,
    bandwidth: Option<(usize, usize)>,
    context: &str,
) {
    try_generate_and_apply_damped_solver_state(target, mesh, bandwidth, context)
        .unwrap_or_else(|err| panic!("{context}: {err:?}"));
}

/// Input bundle used to build a complete frozen solver handoff state.
pub struct FrozenSolverBuildRequest {
    /// Symbolic right-hand sides before BVP discretization.
    pub eq_system: Vec<Expr>,
    /// Names of unknown variables.
    pub values: Vec<String>,
    /// Independent variable name.
    pub arg: String,
    /// Optional symbolic parameter names used by residual/Jacobian generation.
    pub param_names: Option<Vec<String>>,
    /// Current numeric values for `param_names`.
    pub param_values: Option<Vec<f64>>,
    /// Left boundary value of the independent variable.
    pub t0: f64,
    /// Number of discretization steps when a uniform mesh is used.
    pub n_steps: Option<usize>,
    /// Uniform mesh spacing when the mesh is not given explicitly.
    pub h: Option<f64>,
    /// Optional user-supplied mesh.
    pub mesh: Option<Vec<f64>>,
    /// Boundary conditions passed to the BVP symbolic builder.
    pub border_conditions: HashMap<String, Vec<(usize, f64)>>,
    /// Discretization scheme name.
    pub scheme: String,
    /// Matrix backend/method selector used by the legacy BVP module.
    pub method: String,
    /// Optional sparse bandwidth hint.
    pub bandwidth: Option<(usize, usize)>,
    /// Preferred backend branch for sparse symbolic generation.
    pub backend_policy: BackendSelectionPolicy,
    /// Optional resolver snapshot used to detect compiled AOT artifacts.
    pub resolver: Option<AotResolver>,
    /// Solver-level execution policy carried into generated backend setup.
    pub aot_execution_policy: AotExecutionPolicy,
    /// Solver-level build policy carried into generated backend setup.
    pub aot_build_policy: AotBuildPolicy,
    /// Optional compile-time rustc/codegen overrides carried into generated backend setup.
    pub aot_compile_config: AotCompileConfig,
    /// Codegen backend used to emit generated AOT artifacts.
    pub aot_codegen_backend: AotCodegenBackend,
    /// Optional explicit C compiler for C AOT backends.
    pub aot_c_compiler: Option<String>,
    /// Optional chunking overrides carried into generated backend setup.
    pub aot_chunking_policy: AotChunkingPolicy,
    /// Symbolic assembly backend used before lambdify/AOT lowering.
    pub symbolic_assembly_backend: BvpSymbolicAssemblyBackend,
    /// Optional explicit matrix backend override for generated sparse/banded handoff.
    pub matrix_backend_override: Option<MatrixBackend>,
    /// Native linear solver configuration used by the generated banded runtime path.
    pub banded_linear_solver_config: LinearSolverConfig,
}

impl FrozenSolverBuildRequest {
    /// Builds a solver-ready callback and metadata state.
    pub fn generate(self) -> Result<FrozenGeneratedSolverState, BvpBackendIntegrationError> {
        generate_frozen_solver_state(
            self.eq_system,
            self.values,
            self.arg,
            self.param_names,
            self.param_values,
            self.t0,
            self.n_steps,
            self.h,
            self.mesh,
            self.border_conditions,
            self.scheme,
            self.method,
            self.bandwidth,
            self.backend_policy,
            self.resolver.as_ref(),
            self.aot_execution_policy,
            self.aot_build_policy,
            self.aot_compile_config,
            self.aot_codegen_backend,
            self.aot_c_compiler,
            self.aot_chunking_policy,
            self.symbolic_assembly_backend,
            self.matrix_backend_override,
            self.banded_linear_solver_config,
        )
    }
}

/// Builds a frozen solver state and applies it to the target runtime object.
pub fn try_build_and_apply_frozen_solver_state<T: ApplyFrozenGeneratedSolverState>(
    target: &mut T,
    request: FrozenSolverBuildRequest,
    context: &str,
) -> Result<(), BvpBackendIntegrationError> {
    info!("{context}: generating frozen solver state");
    match request.generate() {
        Ok(state) => {
            target.apply_generated_solver_state(state);
            info!("{context}: frozen solver state applied");
            Ok(())
        }
        Err(err) => {
            error!("{context}: {err:?}");
            Err(err)
        }
    }
}

/// Builds a frozen solver state and applies it to the target runtime object.
pub fn build_and_apply_frozen_solver_state<T: ApplyFrozenGeneratedSolverState>(
    target: &mut T,
    request: FrozenSolverBuildRequest,
    context: &str,
) {
    try_build_and_apply_frozen_solver_state(target, request, context)
        .unwrap_or_else(|err| panic!("{context}: {err:?}"));
}

/// Asks a solver runtime object to build its frozen request, then generates and applies the state.
pub fn try_generate_and_apply_frozen_solver_state<
    T: BuildFrozenSolverRequest + ApplyFrozenGeneratedSolverState,
>(
    target: &mut T,
    context: &str,
) -> Result<(), BvpBackendIntegrationError> {
    let request = target.build_solver_request();
    try_build_and_apply_frozen_solver_state(target, request, context)
}

/// Asks a solver runtime object to build its frozen request, then generates and applies the state.
pub fn generate_and_apply_frozen_solver_state<
    T: BuildFrozenSolverRequest + ApplyFrozenGeneratedSolverState,
>(
    target: &mut T,
    context: &str,
) {
    try_generate_and_apply_frozen_solver_state(target, context)
        .unwrap_or_else(|err| panic!("{context}: {err:?}"));
}

/// Builds a complete callback/metadata handoff for the damped solver.
///
/// The sparse mainline goes through the modern bundle path while non-sparse
/// modes still use the legacy Jacobian handoff for compatibility.
#[allow(clippy::too_many_arguments)]
pub fn generate_damped_solver_state(
    eq_system: Vec<Expr>,
    values: Vec<String>,
    param_names: Option<Vec<String>>,
    param_values: Option<Vec<f64>>,
    arg: String,
    t0: f64,
    n_steps: Option<usize>,
    h: Option<f64>,
    mesh: Option<Vec<f64>>,
    border_conditions: HashMap<String, Vec<(usize, f64)>>,
    bounds: Option<HashMap<String, (f64, f64)>>,
    rel_tolerance: Option<HashMap<String, f64>>,
    scheme: String,
    method: String,
    bandwidth: Option<(usize, usize)>,
    backend_policy: BackendSelectionPolicy,
    resolver: Option<&AotResolver>,
    aot_execution_policy: AotExecutionPolicy,
    aot_build_policy: AotBuildPolicy,
    aot_compile_config: AotCompileConfig,
    aot_codegen_backend: AotCodegenBackend,
    aot_c_compiler: Option<String>,
    aot_chunking_policy: AotChunkingPolicy,
    symbolic_assembly_backend: BvpSymbolicAssemblyBackend,
    matrix_backend_override: Option<MatrixBackend>,
    banded_linear_solver_config: LinearSolverConfig,
) -> Result<DampedGeneratedSolverState, BvpBackendIntegrationError> {
    let handoff_begin = Instant::now();
    let mut handoff_diagnostics = HashMap::new();
    let param_name_refs = parameter_name_refs(param_names.as_ref());
    if matches!(method.as_str(), "Sparse" | "Banded")
        || matches!(matrix_backend_override, Some(MatrixBackend::Banded))
    {
        let lifecycle_lock_begin = Instant::now();
        let _lifecycle_guard = aot_lifecycle_needs_serialization(backend_policy, aot_build_policy)
            .then(|| {
                bvp_aot_lifecycle_lock()
                    .lock()
                    .expect("BVP AOT lifecycle lock poisoned")
            });
        if _lifecycle_guard.is_some() {
            insert_elapsed_ms(
                &mut handoff_diagnostics,
                "generated.aot.lifecycle_lock_wait_ms",
                lifecycle_lock_begin,
            );
        }
        let retry_eq_system = eq_system.clone();
        let retry_values = values.clone();
        let retry_arg = arg.clone();
        let retry_param_values = param_values.clone();
        let retry_mesh = mesh.clone();
        let retry_border_conditions = border_conditions.clone();
        let retry_bounds = bounds.clone();
        let retry_rel_tolerance = rel_tolerance.clone();
        let retry_scheme = scheme.clone();
        let retry_method = method.clone();
        let jacobian_instance = prepared_bvp_jacobian(
            symbolic_assembly_backend,
            param_name_refs.as_deref(),
            param_values.clone(),
            banded_linear_solver_config,
        );
        let original_backend_policy = backend_policy;
        let backend_policy =
            effective_backend_policy_for_build(original_backend_policy, aot_build_policy);
        let initial_generate_begin = Instant::now();
        let mut bundle = try_generate_sparse_bundle(
            jacobian_instance,
            eq_system,
            values,
            arg,
            param_name_refs.as_deref(),
            t0,
            n_steps,
            h,
            mesh,
            border_conditions,
            bounds,
            rel_tolerance,
            scheme,
            method,
            bandwidth,
            backend_policy,
            resolver,
            aot_chunking_policy,
        )?;
        insert_elapsed_ms(
            &mut handoff_diagnostics,
            "generated.handoff.initial_generate_wall_ms",
            initial_generate_begin,
        );
        copy_prepare_diagnostics(&mut handoff_diagnostics, &bundle, "initial");
        if let Some(auto_chunking_policy) = auto_codegen_chunking_policy_for_sparse_bundle(
            &bundle,
            original_backend_policy,
            &aot_execution_policy,
            aot_build_policy,
            aot_chunking_policy,
        ) {
            let jacobian_instance = prepared_bvp_jacobian(
                symbolic_assembly_backend,
                param_name_refs.as_deref(),
                retry_param_values.clone(),
                banded_linear_solver_config,
            );
            let auto_regenerate_begin = Instant::now();
            bundle = try_generate_sparse_bundle(
                jacobian_instance,
                retry_eq_system.clone(),
                retry_values.clone(),
                retry_arg.clone(),
                param_name_refs.as_deref(),
                t0,
                n_steps,
                h,
                retry_mesh.clone(),
                retry_border_conditions.clone(),
                retry_bounds.clone(),
                retry_rel_tolerance.clone(),
                retry_scheme.clone(),
                retry_method.clone(),
                bandwidth,
                backend_policy,
                resolver,
                auto_chunking_policy,
            )?;
            insert_elapsed_ms(
                &mut handoff_diagnostics,
                "generated.handoff.auto_chunk_regenerate_wall_ms",
                auto_regenerate_begin,
            );
            copy_prepare_diagnostics(&mut handoff_diagnostics, &bundle, "auto_chunk");
        }
        let build_policy_begin = Instant::now();
        let (bundle, updated_resolver) = enforce_build_policy_on_sparse_bundle(
            bundle,
            resolver,
            aot_build_policy,
            aot_compile_config,
            aot_codegen_backend,
            aot_c_compiler,
            &mut handoff_diagnostics,
        )?;
        insert_elapsed_ms(
            &mut handoff_diagnostics,
            "generated.handoff.build_policy_wall_ms",
            build_policy_begin,
        );
        let bundle = match updated_resolver.as_ref() {
            Some(updated_resolver_ref) if backend_policy_targets_aot(original_backend_policy) => {
                let rebind_begin = Instant::now();
                let bound = activate_compiled_sparse_bundle_after_aot_build(
                    bundle,
                    updated_resolver_ref,
                    aot_codegen_backend,
                )?;
                insert_elapsed_ms(
                    &mut handoff_diagnostics,
                    "generated.handoff.post_build_rebind_wall_ms",
                    rebind_begin,
                );
                bound
            }
            _ => bundle,
        };
        let execution_bind_begin = Instant::now();
        let mut bundle = apply_execution_policy_to_sparse_bundle(bundle, aot_execution_policy)?;
        insert_elapsed_ms(
            &mut handoff_diagnostics,
            "generated.handoff.execution_bind_wall_ms",
            execution_bind_begin,
        );
        insert_elapsed_ms(
            &mut handoff_diagnostics,
            "generated.handoff.total_wall_ms",
            handoff_begin,
        );
        bundle.runtime_diagnostics.extend(handoff_diagnostics);
        Ok(damped_state_from_sparse_solver_bundle(
            bundle,
            updated_resolver,
        ))
    } else {
        let jacobian_instance = prepared_bvp_jacobian(
            symbolic_assembly_backend,
            param_name_refs.as_deref(),
            param_values,
            banded_linear_solver_config,
        );
        let legacy_bundle = jacobian_instance.generate_legacy_solver_bundle_with_params(
            eq_system,
            values,
            arg,
            param_name_refs.as_deref(),
            t0,
            None,
            n_steps,
            h,
            mesh,
            border_conditions,
            bounds,
            rel_tolerance,
            scheme,
            method,
            bandwidth,
        );
        Ok(damped_state_from_legacy_solver_bundle(legacy_bundle))
    }
}

/// Builds a complete callback/metadata handoff for the frozen solver.
///
/// The sparse mainline goes through the modern bundle path while non-sparse
/// modes still use the legacy Jacobian handoff for compatibility.
#[allow(clippy::too_many_arguments)]
pub fn generate_frozen_solver_state(
    eq_system: Vec<Expr>,
    values: Vec<String>,
    arg: String,
    param_names: Option<Vec<String>>,
    param_values: Option<Vec<f64>>,
    t0: f64,
    n_steps: Option<usize>,
    h: Option<f64>,
    mesh: Option<Vec<f64>>,
    border_conditions: HashMap<String, Vec<(usize, f64)>>,
    scheme: String,
    method: String,
    bandwidth: Option<(usize, usize)>,
    backend_policy: BackendSelectionPolicy,
    resolver: Option<&AotResolver>,
    aot_execution_policy: AotExecutionPolicy,
    aot_build_policy: AotBuildPolicy,
    aot_compile_config: AotCompileConfig,
    aot_codegen_backend: AotCodegenBackend,
    aot_c_compiler: Option<String>,
    aot_chunking_policy: AotChunkingPolicy,
    symbolic_assembly_backend: BvpSymbolicAssemblyBackend,
    matrix_backend_override: Option<MatrixBackend>,
    banded_linear_solver_config: LinearSolverConfig,
) -> Result<FrozenGeneratedSolverState, BvpBackendIntegrationError> {
    let handoff_begin = Instant::now();
    let mut handoff_diagnostics = HashMap::new();
    let param_name_refs = parameter_name_refs(param_names.as_ref());
    if matches!(method.as_str(), "Sparse" | "Banded")
        || matches!(matrix_backend_override, Some(MatrixBackend::Banded))
    {
        let lifecycle_lock_begin = Instant::now();
        let _lifecycle_guard = aot_lifecycle_needs_serialization(backend_policy, aot_build_policy)
            .then(|| {
                bvp_aot_lifecycle_lock()
                    .lock()
                    .expect("BVP AOT lifecycle lock poisoned")
            });
        if _lifecycle_guard.is_some() {
            insert_elapsed_ms(
                &mut handoff_diagnostics,
                "generated.aot.lifecycle_lock_wait_ms",
                lifecycle_lock_begin,
            );
        }
        let retry_eq_system = eq_system.clone();
        let retry_values = values.clone();
        let retry_arg = arg.clone();
        let retry_param_values = param_values.clone();
        let retry_mesh = mesh.clone();
        let retry_border_conditions = border_conditions.clone();
        let retry_scheme = scheme.clone();
        let retry_method = method.clone();
        let jacobian_instance = prepared_bvp_jacobian(
            symbolic_assembly_backend,
            param_name_refs.as_deref(),
            param_values.clone(),
            banded_linear_solver_config,
        );
        let original_backend_policy = backend_policy;
        let backend_policy =
            effective_backend_policy_for_build(original_backend_policy, aot_build_policy);
        let initial_generate_begin = Instant::now();
        let mut bundle = try_generate_sparse_bundle(
            jacobian_instance,
            eq_system,
            values,
            arg,
            param_name_refs.as_deref(),
            t0,
            n_steps,
            h,
            mesh,
            border_conditions,
            None,
            None,
            scheme,
            method,
            bandwidth,
            backend_policy,
            resolver,
            aot_chunking_policy,
        )?;
        insert_elapsed_ms(
            &mut handoff_diagnostics,
            "generated.handoff.initial_generate_wall_ms",
            initial_generate_begin,
        );
        copy_prepare_diagnostics(&mut handoff_diagnostics, &bundle, "initial");
        if let Some(auto_chunking_policy) = auto_codegen_chunking_policy_for_sparse_bundle(
            &bundle,
            original_backend_policy,
            &aot_execution_policy,
            aot_build_policy,
            aot_chunking_policy,
        ) {
            let jacobian_instance = prepared_bvp_jacobian(
                symbolic_assembly_backend,
                param_name_refs.as_deref(),
                retry_param_values.clone(),
                banded_linear_solver_config,
            );
            let auto_regenerate_begin = Instant::now();
            bundle = try_generate_sparse_bundle(
                jacobian_instance,
                retry_eq_system.clone(),
                retry_values.clone(),
                retry_arg.clone(),
                param_name_refs.as_deref(),
                t0,
                n_steps,
                h,
                retry_mesh.clone(),
                retry_border_conditions.clone(),
                None,
                None,
                retry_scheme.clone(),
                retry_method.clone(),
                bandwidth,
                backend_policy,
                resolver,
                auto_chunking_policy,
            )?;
            insert_elapsed_ms(
                &mut handoff_diagnostics,
                "generated.handoff.auto_chunk_regenerate_wall_ms",
                auto_regenerate_begin,
            );
            copy_prepare_diagnostics(&mut handoff_diagnostics, &bundle, "auto_chunk");
        }
        let build_policy_begin = Instant::now();
        let (bundle, updated_resolver) = enforce_build_policy_on_sparse_bundle(
            bundle,
            resolver,
            aot_build_policy,
            aot_compile_config,
            aot_codegen_backend,
            aot_c_compiler,
            &mut handoff_diagnostics,
        )?;
        insert_elapsed_ms(
            &mut handoff_diagnostics,
            "generated.handoff.build_policy_wall_ms",
            build_policy_begin,
        );
        let bundle = match updated_resolver.as_ref() {
            Some(updated_resolver_ref) if backend_policy_targets_aot(original_backend_policy) => {
                let rebind_begin = Instant::now();
                let bound = activate_compiled_sparse_bundle_after_aot_build(
                    bundle,
                    updated_resolver_ref,
                    aot_codegen_backend,
                )?;
                insert_elapsed_ms(
                    &mut handoff_diagnostics,
                    "generated.handoff.post_build_rebind_wall_ms",
                    rebind_begin,
                );
                bound
            }
            _ => bundle,
        };
        let execution_bind_begin = Instant::now();
        let mut bundle = apply_execution_policy_to_sparse_bundle(bundle, aot_execution_policy)?;
        insert_elapsed_ms(
            &mut handoff_diagnostics,
            "generated.handoff.execution_bind_wall_ms",
            execution_bind_begin,
        );
        insert_elapsed_ms(
            &mut handoff_diagnostics,
            "generated.handoff.total_wall_ms",
            handoff_begin,
        );
        bundle.runtime_diagnostics.extend(handoff_diagnostics);
        Ok(frozen_state_from_sparse_solver_bundle(
            bundle,
            updated_resolver,
        ))
    } else {
        let jacobian_instance = prepared_bvp_jacobian(
            symbolic_assembly_backend,
            param_name_refs.as_deref(),
            param_values,
            banded_linear_solver_config,
        );
        let legacy_bundle = jacobian_instance.generate_legacy_solver_bundle_with_params(
            eq_system,
            values,
            arg,
            param_name_refs.as_deref(),
            t0,
            None,
            n_steps,
            h,
            mesh,
            border_conditions,
            None,
            None,
            scheme,
            method,
            bandwidth,
        );
        Ok(frozen_state_from_legacy_solver_bundle(legacy_bundle))
    }
}

/// Returns the current default backend policy for solver-side sparse handoff.
///
/// Sparse mainline is now allowed to prefer compiled AOT artifacts when they
/// become available, while still falling back cleanly to lambdified callbacks.
pub fn backend_policy_for_method(method: &str) -> BackendSelectionPolicy {
    if method == "Sparse" {
        BackendSelectionPolicy::PreferAotThenLambdify
    } else {
        BackendSelectionPolicy::LambdifyOnly
    }
}

fn backend_policy_targets_aot(policy: BackendSelectionPolicy) -> bool {
    matches!(
        policy,
        BackendSelectionPolicy::AotOnly
            | BackendSelectionPolicy::PreferAotThenLambdify
            | BackendSelectionPolicy::PreferAotThenNumeric
    )
}

fn aot_lifecycle_needs_serialization(
    _backend_policy: BackendSelectionPolicy,
    build_policy: AotBuildPolicy,
) -> bool {
    matches!(
        build_policy,
        AotBuildPolicy::BuildIfMissing { .. }
            | AotBuildPolicy::RequirePrebuilt
            | AotBuildPolicy::RebuildAlways { .. }
    )
}

fn effective_backend_policy_for_build(
    backend_policy: BackendSelectionPolicy,
    build_policy: AotBuildPolicy,
) -> BackendSelectionPolicy {
    match build_policy {
        AotBuildPolicy::UseIfAvailable => backend_policy,
        AotBuildPolicy::BuildIfMissing { .. } | AotBuildPolicy::RebuildAlways { .. } => {
            if backend_policy_targets_aot(backend_policy) {
                BackendSelectionPolicy::AotOnly
            } else {
                backend_policy
            }
        }
        AotBuildPolicy::RequirePrebuilt => {
            if backend_policy_targets_aot(backend_policy) {
                BackendSelectionPolicy::AotOnly
            } else {
                backend_policy
            }
        }
    }
}

fn parameter_name_refs(param_names: Option<&Vec<String>>) -> Option<Vec<&str>> {
    param_names.map(|names| names.iter().map(|name| name.as_str()).collect())
}

fn prepared_bvp_jacobian(
    symbolic_assembly_backend: BvpSymbolicAssemblyBackend,
    param_name_refs: Option<&[&str]>,
    param_values: Option<Vec<f64>>,
    banded_linear_solver_config: LinearSolverConfig,
) -> Jacobian {
    let mut jacobian_instance = Jacobian::new();
    jacobian_instance.set_symbolic_assembly_backend(symbolic_assembly_backend);
    jacobian_instance.set_params(param_name_refs);
    jacobian_instance.set_param_values(param_values);
    jacobian_instance.set_banded_linear_solver_config(banded_linear_solver_config);
    jacobian_instance
}

#[allow(clippy::too_many_arguments)]
fn try_generate_sparse_bundle(
    jacobian_instance: Jacobian,
    eq_system: Vec<Expr>,
    values: Vec<String>,
    arg: String,
    param_name_refs: Option<&[&str]>,
    t0: f64,
    n_steps: Option<usize>,
    h: Option<f64>,
    mesh: Option<Vec<f64>>,
    border_conditions: HashMap<String, Vec<(usize, f64)>>,
    bounds: Option<HashMap<String, (f64, f64)>>,
    rel_tolerance: Option<HashMap<String, f64>>,
    scheme: String,
    method: String,
    bandwidth: Option<(usize, usize)>,
    backend_policy: BackendSelectionPolicy,
    resolver: Option<&AotResolver>,
    aot_chunking_policy: AotChunkingPolicy,
) -> Result<BvpSparseSolverBundle, BvpBackendIntegrationError> {
    match (
        aot_chunking_policy.residual,
        aot_chunking_policy.sparse_jacobian,
    ) {
        (None, None) => jacobian_instance.try_generate_sparse_solver_bundle_with_backend_selection(
            eq_system,
            values,
            arg,
            param_name_refs,
            t0,
            None,
            n_steps,
            h,
            mesh,
            border_conditions,
            bounds,
            rel_tolerance,
            scheme,
            method,
            bandwidth,
            backend_policy,
            resolver,
        ),
        (residual, sparse_jacobian) => jacobian_instance
            .try_generate_sparse_solver_bundle_with_backend_selection_and_chunking(
                eq_system,
                values,
                arg,
                param_name_refs,
                t0,
                None,
                n_steps,
                h,
                mesh,
                border_conditions,
                bounds,
                rel_tolerance,
                scheme,
                method,
                bandwidth,
                backend_policy,
                resolver,
                residual.unwrap_or(ResidualChunkingStrategy::Whole),
                sparse_jacobian.unwrap_or(SparseChunkingStrategy::Whole),
            ),
    }
}

fn build_policy_can_materialize_auto_chunked_artifact(build_policy: AotBuildPolicy) -> bool {
    matches!(
        build_policy,
        AotBuildPolicy::BuildIfMissing { .. } | AotBuildPolicy::RebuildAlways { .. }
    )
}

fn auto_codegen_chunking_policy_for_sparse_bundle(
    bundle: &BvpSparseSolverBundle,
    original_backend_policy: BackendSelectionPolicy,
    execution_policy: &AotExecutionPolicy,
    build_policy: AotBuildPolicy,
    requested_chunking: AotChunkingPolicy,
) -> Option<AotChunkingPolicy> {
    if requested_chunking != AotChunkingPolicy::default()
        || !matches!(execution_policy, AotExecutionPolicy::Auto)
        || !backend_policy_targets_aot(original_backend_policy)
        || !build_policy_can_materialize_auto_chunked_artifact(build_policy)
    {
        return None;
    }

    let auto_plan = bundle
        .execution
        .selected()
        .prepared_problem
        .auto_parallel_plan();
    if !matches!(
        auto_plan.execution_mode,
        crate::symbolic::codegen::codegen_orchestrator::AutoExecutionMode::Parallel
    ) {
        info!(
            "Auto AOT codegen kept whole callbacks for problem_key={} because residual_reason={}, sparse_reason={}, residual_work/job={}/{}, sparse_work/job={}/{}",
            bundle.execution.selected().problem_key(),
            auto_plan.residual_stage.reason.as_str(),
            auto_plan.sparse_stage.reason.as_str(),
            auto_plan.residual_stage.work_per_job,
            auto_plan.residual_stage.min_work_per_job,
            auto_plan.sparse_stage.work_per_job,
            auto_plan.sparse_stage.min_work_per_job
        );
        return None;
    }

    info!(
        "Auto AOT codegen selected chunked callbacks for problem_key={} with residual_chunking={:?}, sparse_chunking={:?}, residual_jobs={}, sparse_jobs={}, residual_work/job={}, sparse_work/job={}, workers={}",
        bundle.execution.selected().problem_key(),
        auto_plan.residual_chunking,
        auto_plan.sparse_chunking,
        auto_plan.residual_stage.jobs,
        auto_plan.sparse_stage.jobs,
        auto_plan.residual_stage.work_per_job,
        auto_plan.sparse_stage.work_per_job,
        auto_plan.workers
    );
    Some(AotChunkingPolicy::with_parts(
        Some(auto_plan.residual_chunking),
        Some(auto_plan.sparse_chunking),
    ))
}

fn activate_compiled_sparse_bundle_after_aot_build(
    mut bundle: BvpSparseSolverBundle,
    updated_resolver: &AotResolver,
    aot_codegen_backend: AotCodegenBackend,
) -> Result<BvpSparseSolverBundle, BvpBackendIntegrationError> {
    let problem_key = bundle.execution.selected().problem_key();
    let resolution = updated_resolver.resolve_by_problem_key(problem_key.as_str());
    if !resolution.is_compiled() {
        return Err(
            BvpBackendIntegrationError::CompiledAotRequiredButUnavailable {
                problem_key,
                effective_backend: bundle.effective_backend(),
            },
        );
    }

    let mut selected = bundle.execution.selected().clone();
    selected.requested_backend = BackendKind::Aot;
    selected.effective_backend = SelectedBackendKind::AotCompiled;
    selected.aot_resolution = Some(resolution);
    bundle.execution = BvpSparseExecutionPlan::AotCompiled(selected);

    if !try_link_sparse_runtime_from_resolution(
        &bundle,
        Some(updated_resolver),
        aot_codegen_backend,
    ) || !bundle.rebind_linked_runtime_callbacks(None, None)
    {
        return Err(BvpBackendIntegrationError::CompiledAotRuntimeUnavailable { problem_key });
    }

    Ok(bundle)
}

fn sanitize_generated_name(input: &str) -> String {
    input
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' => ch.to_ascii_lowercase(),
            _ => '_',
        })
        .collect::<String>()
}

fn build_output_parent_for_problem(problem_key: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("generated-aot")
        .join(sanitize_generated_name(problem_key))
}

fn unique_build_output_parent_for_problem(problem_key: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after unix epoch")
        .as_nanos();
    build_output_parent_for_problem(problem_key)
        .join(format!("build-{}-{nonce}", std::process::id()))
}

fn to_lifecycle_build_profile(profile: AotBuildProfile) -> LifecycleBuildProfile {
    match profile {
        AotBuildProfile::Release => LifecycleBuildProfile::Release,
        AotBuildProfile::Debug => LifecycleBuildProfile::Debug,
    }
}

fn to_c_build_profile(profile: AotBuildProfile) -> CAotBuildProfile {
    match profile {
        AotBuildProfile::Release => CAotBuildProfile::Release,
        AotBuildProfile::Debug => CAotBuildProfile::Debug,
    }
}

fn to_zig_build_profile(profile: AotBuildProfile) -> ZigAotBuildProfile {
    match profile {
        AotBuildProfile::Release => ZigAotBuildProfile::ReleaseFast,
        AotBuildProfile::Debug => ZigAotBuildProfile::Debug,
    }
}

fn to_c_compile_config(compile_config: &AotCompileConfig) -> CAotCompileConfig {
    if *compile_config == AotCompileConfig::dev_fastest() {
        CAotCompileConfig::dev_fastest()
    } else if *compile_config == AotCompileConfig::fast_build() {
        CAotCompileConfig::fast_build()
    } else {
        CAotCompileConfig::production()
    }
}

fn register_sparse_runtime_from_registered_artifact(
    artifact: &crate::symbolic::codegen::codegen_aot_registry::RegisteredAotArtifact,
    matrix_backend: MatrixBackend,
    backend: AotCodegenBackend,
) -> Result<(), String> {
    match backend {
        AotCodegenBackend::Rust => match matrix_backend {
            MatrixBackend::Banded => register_generated_banded_cdylib_backend(artifact).map(|_| ()),
            _ => register_generated_sparse_cdylib_backend(artifact).map(|_| ()),
        },
        AotCodegenBackend::C => match matrix_backend {
            MatrixBackend::Banded => register_generated_c_banded_backend(artifact).map(|_| ()),
            _ => register_generated_c_sparse_backend(artifact).map(|_| ()),
        },
        AotCodegenBackend::Zig => match matrix_backend {
            MatrixBackend::Banded => register_generated_zig_banded_backend(artifact).map(|_| ()),
            _ => register_generated_zig_sparse_backend(artifact).map(|_| ()),
        },
    }
}

fn rust_sparse_aot_build_request(
    bundle: &BvpSparseSolverBundle,
    problem_key: &str,
    profile: AotBuildProfile,
    compile_config: AotCompileConfig,
) -> (AotBuildRequest, BvpGeneratedAotCrateBreakdown) {
    let selected = bundle.execution.selected();
    let backend_label = match selected.matrix_backend {
        MatrixBackend::Banded => "banded",
        _ => "sparse",
    };
    let crate_name = format!(
        "generated_bvp_{}_{}",
        backend_label,
        sanitize_generated_name(problem_key)
    );
    let module_name = format!(
        "generated_bvp_module_{}_{}",
        backend_label,
        sanitize_generated_name(problem_key)
    );
    let output_parent_dir = unique_build_output_parent_for_problem(problem_key);
    let (artifact, breakdown) = selected
        .prepared_problem
        .generated_aot_artifact_with_breakdown_for_matrix_backend(
            &crate_name,
            &module_name,
            AotCodegenBackend::Rust,
            selected.matrix_backend,
        );
    let request = AotBuildRequest::new(
        artifact
            .into_rust_crate()
            .expect("Rust backend must emit GeneratedAotCrate"),
        output_parent_dir,
        to_lifecycle_build_profile(profile),
    )
    .with_compile_config(compile_config);
    (request, breakdown)
}

fn c_sparse_aot_build_request(
    bundle: &BvpSparseSolverBundle,
    problem_key: &str,
    profile: AotBuildProfile,
    compile_config: CAotCompileConfig,
) -> (CAotBuildRequest, BvpGeneratedAotCrateBreakdown) {
    let selected = bundle.execution.selected();
    let backend_label = match selected.matrix_backend {
        MatrixBackend::Banded => "banded",
        _ => "sparse",
    };
    let library_name = format!(
        "generated_bvp_{}_{}",
        backend_label,
        sanitize_generated_name(problem_key)
    );
    let module_name = format!(
        "generated_bvp_module_{}_{}",
        backend_label,
        sanitize_generated_name(problem_key)
    );
    let output_parent_dir = unique_build_output_parent_for_problem(problem_key);
    let (artifact, breakdown) = selected
        .prepared_problem
        .generated_aot_artifact_with_breakdown_for_matrix_backend(
            &library_name,
            &module_name,
            AotCodegenBackend::C,
            selected.matrix_backend,
        );
    let library_spec = match artifact {
        crate::symbolic::codegen::codegen_aot_driver::GeneratedAotArtifact::C(library) => library,
        _ => unreachable!("C backend must emit GeneratedCAotLibrary"),
    };
    let request =
        CAotBuildRequest::new(library_spec, output_parent_dir, to_c_build_profile(profile))
            .with_compile_config(compile_config);
    (request, breakdown)
}

fn zig_sparse_aot_build_request(
    bundle: &BvpSparseSolverBundle,
    problem_key: &str,
    profile: AotBuildProfile,
) -> (ZigAotBuildRequest, BvpGeneratedAotCrateBreakdown) {
    let selected = bundle.execution.selected();
    let backend_label = match selected.matrix_backend {
        MatrixBackend::Banded => "banded",
        _ => "sparse",
    };
    let library_name = format!(
        "generated_bvp_{}_{}",
        backend_label,
        sanitize_generated_name(problem_key)
    );
    let module_name = format!(
        "generated_bvp_module_{}_{}",
        backend_label,
        sanitize_generated_name(problem_key)
    );
    let output_parent_dir = unique_build_output_parent_for_problem(problem_key);
    let (artifact, breakdown) = selected
        .prepared_problem
        .generated_aot_artifact_with_breakdown_for_matrix_backend(
            &library_name,
            &module_name,
            AotCodegenBackend::Zig,
            selected.matrix_backend,
        );
    let library_spec = match artifact {
        crate::symbolic::codegen::codegen_aot_driver::GeneratedAotArtifact::Zig(library) => library,
        _ => unreachable!("Zig backend must emit GeneratedZigAotLibrary"),
    };
    (
        ZigAotBuildRequest::new(
            library_spec,
            output_parent_dir,
            to_zig_build_profile(profile),
        ),
        breakdown,
    )
}

fn try_materialize_and_build_sparse_aot_bundle(
    bundle: &BvpSparseSolverBundle,
    resolver: Option<&AotResolver>,
    profile: AotBuildProfile,
    compile_config: AotCompileConfig,
    aot_codegen_backend: AotCodegenBackend,
    aot_c_compiler: Option<String>,
    diagnostics: &mut HashMap<String, String>,
) -> Result<AotResolver, BvpBackendIntegrationError> {
    let selected = bundle.execution.selected();
    let problem_key = selected.problem_key();
    let manifest_begin = Instant::now();
    let manifest = PreparedProblemManifest::from(
        &selected
            .prepared_problem
            .as_prepared_problem_for_matrix_backend(selected.matrix_backend),
    );
    insert_elapsed_ms(diagnostics, "generated.aot.manifest_ms", manifest_begin);
    let mut registry = resolver
        .map(|existing| existing.registry().clone())
        .unwrap_or_else(AotRegistry::new);
    match aot_codegen_backend {
        AotCodegenBackend::Rust => {
            let artifact_begin = Instant::now();
            let (request, breakdown) =
                rust_sparse_aot_build_request(bundle, &problem_key, profile, compile_config);
            append_aot_artifact_breakdown(diagnostics, &breakdown);
            insert_elapsed_ms(
                diagnostics,
                "generated.aot.artifact_wall_ms",
                artifact_begin,
            );
            info!(
                "materializing sparse Rust AOT crate for problem_key={} with build_profile={:?}",
                problem_key, profile
            );
            let materialize_begin = Instant::now();
            let build = request.materialize().map_err(|err| {
                BvpBackendIntegrationError::AutomaticAotBuildFailed {
                    problem_key: problem_key.clone(),
                    message: err.to_string(),
                }
            })?;
            insert_elapsed_ms(
                diagnostics,
                "generated.aot.materialize_ms",
                materialize_begin,
            );
            info!(
                "executing sparse Rust AOT build for problem_key={} in crate_dir={}",
                problem_key,
                build.written.crate_dir.display()
            );
            let compile_link_begin = Instant::now();
            let build_context = format!(
                "bvp {:?} Rust key={} output={}",
                selected.matrix_backend,
                problem_key,
                build.written.crate_dir.display()
            );
            execute_aot_build_with_retry(
                || {
                    let executed = build.execute().map_err(|err| err.to_string())?;
                    Ok((
                        executed.succeeded(),
                        executed.status_code,
                        executed.stdout.clone(),
                        executed.stderr.clone(),
                    ))
                },
                &build_context,
            )
            .map_err(|message| {
                BvpBackendIntegrationError::AutomaticAotBuildFailed {
                    problem_key: problem_key.clone(),
                    message,
                }
            })?;
            insert_elapsed_ms(
                diagnostics,
                "generated.aot.compile_link_ms",
                compile_link_begin,
            );
            let registration_begin = Instant::now();
            let registered = registry
                .register_materialized_build(manifest, &build)
                .clone();
            append_registered_artifact_contract(diagnostics, &registered);
            let runtime_registration = match selected.matrix_backend {
                MatrixBackend::Banded => register_generated_banded_cdylib_backend(&registered),
                _ => register_generated_sparse_cdylib_backend(&registered),
            };
            if let Err(err) = runtime_registration {
                error!(
                    "{:?} Rust AOT build succeeded for problem_key={} but runtime cdylib registration failed: {}",
                    selected.matrix_backend, problem_key, err
                );
            }
            insert_elapsed_ms(
                diagnostics,
                "generated.aot.register_link_ms",
                registration_begin,
            );
        }
        AotCodegenBackend::C => {
            let mut c_compile = match profile {
                AotBuildProfile::Debug => CAotCompileConfig::dev_fastest(),
                AotBuildProfile::Release => to_c_compile_config(&compile_config),
            };
            if let Some(compiler) = aot_c_compiler {
                c_compile = c_compile.with_compiler(compiler);
            }
            let artifact_begin = Instant::now();
            let (request, breakdown) =
                c_sparse_aot_build_request(bundle, &problem_key, profile, c_compile);
            append_aot_artifact_breakdown(diagnostics, &breakdown);
            insert_elapsed_ms(
                diagnostics,
                "generated.aot.artifact_wall_ms",
                artifact_begin,
            );
            info!(
                "materializing sparse C AOT library for problem_key={} with build_profile={:?}",
                problem_key, profile
            );
            let materialize_begin = Instant::now();
            let build = request.materialize().map_err(|err| {
                BvpBackendIntegrationError::AutomaticAotBuildFailed {
                    problem_key: problem_key.clone(),
                    message: err.to_string(),
                }
            })?;
            insert_elapsed_ms(
                diagnostics,
                "generated.aot.materialize_ms",
                materialize_begin,
            );
            info!(
                "executing sparse C AOT build for problem_key={} in library_dir={}",
                problem_key,
                build.written.library_dir.display()
            );
            let compile_link_begin = Instant::now();
            let build_context = format!(
                "bvp {:?} C key={} output={}",
                selected.matrix_backend,
                problem_key,
                build.written.library_dir.display()
            );
            execute_aot_build_with_retry(
                || {
                    let executed = build.execute().map_err(|err| err.to_string())?;
                    Ok((
                        executed.succeeded(),
                        executed.status_code,
                        executed.stdout.clone(),
                        executed.stderr.clone(),
                    ))
                },
                &build_context,
            )
            .map_err(|message| {
                BvpBackendIntegrationError::AutomaticAotBuildFailed {
                    problem_key: problem_key.clone(),
                    message,
                }
            })?;
            insert_elapsed_ms(
                diagnostics,
                "generated.aot.compile_link_ms",
                compile_link_begin,
            );
            let registration_begin = Instant::now();
            let registered = register_c_build_in_registry(&mut registry, manifest, &build).clone();
            append_registered_artifact_contract(diagnostics, &registered);
            let runtime_registration = match selected.matrix_backend {
                MatrixBackend::Banded => register_generated_c_banded_backend(&registered),
                _ => register_generated_c_sparse_backend(&registered),
            };
            if let Err(err) = runtime_registration {
                error!(
                    "{:?} C AOT build succeeded for problem_key={} but runtime registration failed: {}",
                    selected.matrix_backend, problem_key, err
                );
            }
            insert_elapsed_ms(
                diagnostics,
                "generated.aot.register_link_ms",
                registration_begin,
            );
        }
        AotCodegenBackend::Zig => {
            let artifact_begin = Instant::now();
            let (request, breakdown) = zig_sparse_aot_build_request(bundle, &problem_key, profile);
            append_aot_artifact_breakdown(diagnostics, &breakdown);
            insert_elapsed_ms(
                diagnostics,
                "generated.aot.artifact_wall_ms",
                artifact_begin,
            );
            info!(
                "materializing sparse Zig AOT library for problem_key={} with build_profile={:?}",
                problem_key, profile
            );
            let materialize_begin = Instant::now();
            let build = request.materialize().map_err(|err| {
                BvpBackendIntegrationError::AutomaticAotBuildFailed {
                    problem_key: problem_key.clone(),
                    message: err.to_string(),
                }
            })?;
            insert_elapsed_ms(
                diagnostics,
                "generated.aot.materialize_ms",
                materialize_begin,
            );
            info!(
                "executing sparse Zig AOT build for problem_key={} in library_dir={}",
                problem_key,
                build.written.library_dir.display()
            );
            let compile_link_begin = Instant::now();
            let build_context = format!(
                "bvp {:?} Zig key={} output={}",
                selected.matrix_backend,
                problem_key,
                build.written.library_dir.display()
            );
            execute_aot_build_with_retry(
                || {
                    let executed = build.execute().map_err(|err| err.to_string())?;
                    Ok((
                        executed.succeeded(),
                        executed.status_code,
                        executed.stdout.clone(),
                        executed.stderr.clone(),
                    ))
                },
                &build_context,
            )
            .map_err(|message| {
                BvpBackendIntegrationError::AutomaticAotBuildFailed {
                    problem_key: problem_key.clone(),
                    message,
                }
            })?;
            insert_elapsed_ms(
                diagnostics,
                "generated.aot.compile_link_ms",
                compile_link_begin,
            );
            let registration_begin = Instant::now();
            let registered =
                register_zig_build_in_registry(&mut registry, manifest, &build).clone();
            append_registered_artifact_contract(diagnostics, &registered);
            let runtime_registration = match selected.matrix_backend {
                MatrixBackend::Banded => register_generated_zig_banded_backend(&registered),
                _ => register_generated_zig_sparse_backend(&registered),
            };
            if let Err(err) = runtime_registration {
                error!(
                    "{:?} Zig AOT build succeeded for problem_key={} but runtime registration failed: {}",
                    selected.matrix_backend, problem_key, err
                );
            }
            insert_elapsed_ms(
                diagnostics,
                "generated.aot.register_link_ms",
                registration_begin,
            );
        }
    }
    info!(
        "sparse {:?} AOT build succeeded for problem_key={} with profile={:?}",
        aot_codegen_backend, problem_key, profile
    );
    Ok(AotResolver::new(registry))
}

fn try_link_sparse_runtime_from_resolution(
    bundle: &BvpSparseSolverBundle,
    resolver: Option<&AotResolver>,
    aot_codegen_backend: AotCodegenBackend,
) -> bool {
    let problem_key = bundle.execution.selected().problem_key();
    if resolve_linked_sparse_backend(problem_key.as_str()).is_some() {
        return true;
    }

    let resolved = bundle
        .resolved_aot_artifact()
        .cloned()
        .or_else(|| resolver.map(|value| value.resolve_by_problem_key(problem_key.as_str())));
    let Some(resolved) = resolved else {
        return false;
    };
    if !resolved.is_compiled() {
        error!(
            "resolved sparse generated {:?} artifact for problem_key={} is not compiled/callable by contract: {}",
            aot_codegen_backend,
            problem_key,
            resolved.registered.lifecycle_contract_summary()
        );
        return false;
    }

    match register_sparse_runtime_from_registered_artifact(
        &resolved.registered,
        bundle.execution.selected().matrix_backend,
        aot_codegen_backend,
    ) {
        Ok(_) => true,
        Err(err) => {
            error!(
                "failed to register sparse generated {:?} runtime for problem_key={}: {}; artifact_contract={}",
                aot_codegen_backend,
                problem_key,
                err,
                resolved.registered.lifecycle_contract_summary()
            );
            false
        }
    }
}

fn enforce_build_policy_on_sparse_bundle(
    bundle: BvpSparseSolverBundle,
    resolver: Option<&AotResolver>,
    build_policy: AotBuildPolicy,
    compile_config: AotCompileConfig,
    aot_codegen_backend: AotCodegenBackend,
    aot_c_compiler: Option<String>,
    diagnostics: &mut HashMap<String, String>,
) -> Result<(BvpSparseSolverBundle, Option<AotResolver>), BvpBackendIntegrationError> {
    let problem_key = bundle.execution.selected().problem_key();
    let effective_backend = bundle.effective_backend();
    info!(
        "enforcing sparse AOT build policy {} for problem_key={} with effective_backend={:?}",
        build_policy.as_str(),
        problem_key,
        effective_backend
    );

    match build_policy {
        AotBuildPolicy::UseIfAvailable => {
            let runtime_available = bundle.is_runtime_callable()
                || (matches!(effective_backend, SelectedBackendKind::AotCompiled)
                    && try_link_sparse_runtime_from_resolution(
                        &bundle,
                        resolver,
                        aot_codegen_backend,
                    ));
            if matches!(effective_backend, SelectedBackendKind::AotCompiled) && !runtime_available {
                error!(
                    "compiled sparse AOT backend resolved for problem_key={} but runtime callbacks are unavailable",
                    problem_key
                );
                Err(BvpBackendIntegrationError::CompiledAotRuntimeUnavailable { problem_key })
            } else {
                info!(
                    "reusing available sparse backend for problem_key={} under UseIfAvailable",
                    problem_key
                );
                Ok((bundle, None))
            }
        }
        AotBuildPolicy::RequirePrebuilt => {
            let runtime_available = bundle.is_runtime_callable()
                || (matches!(effective_backend, SelectedBackendKind::AotCompiled)
                    && try_link_sparse_runtime_from_resolution(
                        &bundle,
                        resolver,
                        aot_codegen_backend,
                    ));
            if !matches!(effective_backend, SelectedBackendKind::AotCompiled) {
                error!(
                    "RequirePrebuilt requested for problem_key={} but compiled backend is not available: {:?}",
                    problem_key, effective_backend
                );
                Err(
                    BvpBackendIntegrationError::CompiledAotRequiredButUnavailable {
                        problem_key,
                        effective_backend,
                    },
                )
            } else if !runtime_available {
                error!(
                    "RequirePrebuilt succeeded in resolution for problem_key={} but runtime callbacks are unavailable",
                    problem_key
                );
                Err(BvpBackendIntegrationError::CompiledAotRuntimeUnavailable { problem_key })
            } else {
                info!(
                    "using prebuilt sparse AOT backend for problem_key={}",
                    problem_key
                );
                Ok((bundle, None))
            }
        }
        AotBuildPolicy::BuildIfMissing { .. } => {
            let runtime_available = bundle.is_runtime_callable()
                || (matches!(effective_backend, SelectedBackendKind::AotCompiled)
                    && try_link_sparse_runtime_from_resolution(
                        &bundle,
                        resolver,
                        aot_codegen_backend,
                    ));
            if matches!(effective_backend, SelectedBackendKind::AotCompiled) && runtime_available {
                info!(
                    "compiled sparse AOT backend already callable for problem_key={}, skipping build",
                    problem_key
                );
                Ok((bundle, None))
            } else {
                let profile = match build_policy {
                    AotBuildPolicy::BuildIfMissing { profile } => profile,
                    _ => unreachable!(),
                };
                // On Windows, an already loaded generated cdylib keeps the .dll file locked.
                // If the compiled artifact is present but not currently callable for this bundle,
                // drop any stale linked backend before rebuilding so cargo can overwrite it.
                let _ = unregister_linked_sparse_backend(problem_key.as_str());
                info!(
                    "compiled sparse AOT backend missing or not callable for problem_key={}, building with profile={:?}",
                    problem_key, profile
                );
                let updated_resolver = try_materialize_and_build_sparse_aot_bundle(
                    &bundle,
                    resolver,
                    profile,
                    compile_config.clone(),
                    aot_codegen_backend,
                    aot_c_compiler.clone(),
                    diagnostics,
                )?;
                if bundle.is_runtime_callable()
                    || try_link_sparse_runtime_from_resolution(
                        &bundle,
                        Some(&updated_resolver),
                        aot_codegen_backend,
                    )
                {
                    info!(
                        "compiled sparse AOT backend became callable after BuildIfMissing for problem_key={}",
                        problem_key
                    );
                    Ok((bundle, Some(updated_resolver)))
                } else {
                    error!(
                        "BuildIfMissing completed for problem_key={} but runtime callbacks are still unavailable",
                        problem_key
                    );
                    Err(BvpBackendIntegrationError::AutomaticAotBuildRequested { problem_key })
                }
            }
        }
        AotBuildPolicy::RebuildAlways { .. } => {
            let profile = match build_policy {
                AotBuildPolicy::RebuildAlways { profile } => profile,
                _ => unreachable!(),
            };
            // Forced rebuild must also unload any previously linked generated cdylib;
            // otherwise Windows denies replacing the existing .dll on disk.
            let _ = unregister_linked_sparse_backend(problem_key.as_str());
            info!(
                "forcing sparse AOT rebuild for problem_key={} with profile={:?}",
                problem_key, profile
            );
            let updated_resolver = try_materialize_and_build_sparse_aot_bundle(
                &bundle,
                resolver,
                profile,
                compile_config,
                aot_codegen_backend,
                aot_c_compiler,
                diagnostics,
            )?;
            if bundle.is_runtime_callable()
                || try_link_sparse_runtime_from_resolution(
                    &bundle,
                    Some(&updated_resolver),
                    aot_codegen_backend,
                )
            {
                info!(
                    "compiled sparse AOT backend remains callable after forced rebuild for problem_key={}",
                    problem_key
                );
                Ok((bundle, Some(updated_resolver)))
            } else {
                error!(
                    "forced sparse AOT rebuild completed for problem_key={} but runtime callbacks are unavailable",
                    problem_key
                );
                Err(BvpBackendIntegrationError::AutomaticAotRebuildRequested { problem_key })
            }
        }
    }
}

fn apply_execution_policy_to_sparse_bundle(
    mut bundle: BvpSparseSolverBundle,
    policy: AotExecutionPolicy,
) -> Result<BvpSparseSolverBundle, BvpBackendIntegrationError> {
    let problem_key = bundle.execution.selected().problem_key();
    info!(
        "applying sparse AOT execution policy {} for problem_key={} with effective_backend={:?}",
        policy.as_str(),
        problem_key,
        bundle.effective_backend()
    );
    match policy {
        AotExecutionPolicy::Auto => {
            if matches!(bundle.effective_backend(), SelectedBackendKind::AotCompiled) {
                let auto_plan = bundle
                    .execution
                    .selected()
                    .prepared_problem
                    .auto_parallel_plan();
                if let Some(config) = auto_plan.executor_config {
                    if !bundle.rebind_linked_runtime_callbacks(None, Some(config)) {
                        return Err(BvpBackendIntegrationError::CompiledAotRuntimeUnavailable {
                            problem_key,
                        });
                    }
                    bundle.refresh_linked_runtime_diagnostics(policy.as_str(), Some(config));
                    info!(
                        "auto-selected sparse parallel runtime binding for problem_key={} with residual_jobs={:?}, sparse_jobs={:?}, residual_chunking={:?}, sparse_chunking={:?}, residual_reason={}, sparse_reason={}, residual_work_per_job={}, sparse_work_per_job={}, min_work_per_job={}, workers={}",
                        problem_key,
                        config.max_residual_jobs,
                        config.max_sparse_jobs,
                        auto_plan.residual_chunking,
                        auto_plan.sparse_chunking,
                        auto_plan.residual_stage.reason.as_str(),
                        auto_plan.sparse_stage.reason.as_str(),
                        auto_plan.residual_stage.work_per_job,
                        auto_plan.sparse_stage.work_per_job,
                        auto_plan.min_work_per_job,
                        auto_plan.workers
                    );
                } else if !bundle.rebind_linked_runtime_callbacks(None, None) {
                    return Err(BvpBackendIntegrationError::CompiledAotRuntimeUnavailable {
                        problem_key,
                    });
                } else {
                    bundle.refresh_linked_runtime_diagnostics(policy.as_str(), None);
                    info!(
                        "auto-selected sequential sparse runtime binding for problem_key={} with residual_reason={}, sparse_reason={}, residual_work_per_job={}, sparse_work_per_job={}, min_work_per_job={} and workers={}",
                        problem_key,
                        auto_plan.residual_stage.reason.as_str(),
                        auto_plan.sparse_stage.reason.as_str(),
                        auto_plan.residual_stage.work_per_job,
                        auto_plan.sparse_stage.work_per_job,
                        auto_plan.min_work_per_job,
                        auto_plan.workers
                    );
                }
            } else {
                bundle.refresh_linked_runtime_diagnostics(policy.as_str(), None);
                info!(
                    "keeping default sparse runtime callback binding for problem_key={}",
                    problem_key
                );
            }
        }
        AotExecutionPolicy::SequentialOnly => {
            if matches!(bundle.effective_backend(), SelectedBackendKind::AotCompiled)
                && !bundle.rebind_linked_runtime_callbacks(None, None)
            {
                return Err(BvpBackendIntegrationError::CompiledAotRuntimeUnavailable {
                    problem_key,
                });
            }
            bundle.refresh_linked_runtime_diagnostics(policy.as_str(), None);
            info!(
                "bound sparse runtime callbacks sequentially for problem_key={}",
                problem_key
            );
        }
        AotExecutionPolicy::Parallel(config) => {
            if matches!(bundle.effective_backend(), SelectedBackendKind::AotCompiled)
                && !bundle.rebind_linked_runtime_callbacks(None, Some(config))
            {
                return Err(BvpBackendIntegrationError::CompiledAotRuntimeUnavailable {
                    problem_key,
                });
            }
            bundle.refresh_linked_runtime_diagnostics(policy.as_str(), Some(config));
            info!(
                "bound sparse runtime callbacks in parallel for problem_key={}",
                problem_key
            );
        }
    }
    Ok(bundle)
}

/// Converts the assembled sparse symbolic/provider bundle into a damped solver runtime state.
pub fn damped_state_from_sparse_solver_bundle(
    bundle: BvpSparseSolverBundle,
    updated_resolver: Option<AotResolver>,
) -> DampedGeneratedSolverState {
    let selected_backend = bundle.effective_backend();
    let bounds_vec = bundle.bounds_vec.clone().unwrap_or_default();
    let rel_tolerance_vec = bundle.rel_tolerance_vec.clone().unwrap_or_default();
    let variable_string = bundle.variable_string.clone();
    let bandwidth = bundle.bandwidth.unwrap_or((0, 0));
    let bc_position_and_value = bundle.bc_position_and_value.clone();
    let runtime_diagnostics = bundle.runtime_diagnostics().clone();

    let (fun, jac) = bundle
        .into_runtime_callbacks()
        .unwrap_or_else(|| panic!("BVP sparse solver bundle did not provide runtime callbacks"));

    DampedGeneratedSolverState {
        fun,
        jac: Some(jac),
        bounds_vec,
        rel_tolerance_vec,
        variable_string,
        bandwidth,
        bc_position_and_value,
        updated_resolver,
        selected_backend,
        runtime_diagnostics,
    }
}

/// Compatibility helper for callers that still hold a legacy [`Jacobian`] instance.
///
/// New code should prefer bundle-based handoff paths instead of converting from
/// raw Jacobian state directly.
pub fn damped_state_from_legacy_jacobian(
    jacobian_instance: Jacobian,
) -> DampedGeneratedSolverState {
    damped_state_from_legacy_solver_bundle(jacobian_instance.into_legacy_solver_bundle())
}

/// Converts a centralized legacy symbolic bundle into the damped solver runtime state.
pub fn damped_state_from_legacy_solver_bundle(
    bundle: BvpLegacySolverBundle,
) -> DampedGeneratedSolverState {
    DampedGeneratedSolverState {
        fun: bundle.residual_function,
        jac: bundle.jacobian_function,
        bounds_vec: bundle.bounds_vec.unwrap_or_default(),
        rel_tolerance_vec: bundle.rel_tolerance_vec.unwrap_or_default(),
        variable_string: bundle.variable_string,
        bandwidth: bundle.bandwidth.unwrap_or((0, 0)),
        bc_position_and_value: bundle.bc_position_and_value,
        updated_resolver: None,
        selected_backend: SelectedBackendKind::Lambdify,
        runtime_diagnostics: HashMap::new(),
    }
}

/// Converts the assembled sparse symbolic/provider bundle into a frozen solver runtime state.
pub fn frozen_state_from_sparse_solver_bundle(
    bundle: BvpSparseSolverBundle,
    updated_resolver: Option<AotResolver>,
) -> FrozenGeneratedSolverState {
    let selected_backend = bundle.effective_backend();
    let variable_string = bundle.variable_string.clone();
    let bandwidth = bundle.bandwidth.unwrap_or((0, 0));
    let runtime_diagnostics = bundle.runtime_diagnostics().clone();

    let (fun, jac) = bundle.into_runtime_callbacks().unwrap_or_else(|| {
        panic!("Frozen BVP sparse solver bundle did not provide runtime callbacks")
    });

    FrozenGeneratedSolverState {
        fun,
        jac: Some(jac),
        variable_string,
        bandwidth,
        updated_resolver,
        selected_backend,
        runtime_diagnostics,
    }
}

/// Compatibility helper for callers that still hold a legacy [`Jacobian`] instance.
///
/// New code should prefer bundle-based handoff paths instead of converting from
/// raw Jacobian state directly.
pub fn frozen_state_from_legacy_jacobian(
    jacobian_instance: Jacobian,
) -> FrozenGeneratedSolverState {
    frozen_state_from_legacy_solver_bundle(jacobian_instance.into_legacy_solver_bundle())
}

/// Converts a centralized legacy symbolic bundle into the frozen solver runtime state.
pub fn frozen_state_from_legacy_solver_bundle(
    bundle: BvpLegacySolverBundle,
) -> FrozenGeneratedSolverState {
    FrozenGeneratedSolverState {
        fun: bundle.residual_function,
        jac: bundle.jacobian_function,
        variable_string: bundle.variable_string,
        bandwidth: bundle.bandwidth.unwrap_or((0, 0)),
        updated_resolver: None,
        selected_backend: SelectedBackendKind::Lambdify,
        runtime_diagnostics: HashMap::new(),
    }
}
//=============================================================================================
// TESTS
//=============================================================================================
#[cfg(test)]
mod tests {
    use super::{
        AotBuildPolicy, AotChunkingPolicy, AotCompileConfig, AotExecutionPolicy,
        BandedGeneratedBackendMode, DampedSolverBuildRequest, FrozenSolverBuildRequest,
        GeneratedBackendConfig, SparseGeneratedBackendMode,
    };
    use crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend;
    use crate::symbolic::codegen::codegen_aot_registry::AotRegistry;
    use crate::symbolic::codegen::codegen_aot_resolution::AotResolver;
    use crate::symbolic::codegen::codegen_aot_runtime_link::{
        register_linked_sparse_backend, unregister_linked_sparse_backend, LinkedResidualChunk,
        LinkedSparseAotBackend, LinkedSparseJacobianChunk,
    };
    use crate::symbolic::codegen::codegen_backend_selection::{
        BackendSelectionPolicy, SelectedBackendKind,
    };
    use crate::symbolic::codegen::codegen_manifest::PreparedProblemManifest;
    use crate::symbolic::codegen::codegen_orchestrator::{
        ParallelExecutorConfig, ParallelFallbackPolicy,
    };
    use crate::symbolic::codegen::codegen_provider_api::{MatrixBackend, PreparedProblem};
    use crate::symbolic::codegen::rust_backend::codegen_aot_build::{
        AotBuildProfile, AotBuildRequest,
    };
    use crate::symbolic::symbolic_engine::Expr;
    use crate::symbolic::symbolic_functions_BVP::{
        BvpBackendIntegrationError, BvpSymbolicAssemblyBackend, Jacobian,
    };
    use faer::Col;
    use std::collections::HashMap;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use std::time::{SystemTime, UNIX_EPOCH};

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
        assert_eq!(config.aot_codegen_backend, AotCodegenBackend::Rust);
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
                        *slot = expr.lambdify_borrowed_thread_safe(&input_names)(args)
                            + offset_residual;
                    }
                }),
                Arc::new(move |args, out| {
                    let input_names = jacobian_input_name_strings
                        .iter()
                        .map(|name| name.as_str())
                        .collect::<Vec<_>>();
                    for (slot, (_, _, expr)) in out.iter_mut().zip(sparse_exprs.iter()) {
                        *slot = expr.lambdify_borrowed_thread_safe(&input_names)(args)
                            + offset_jacobian;
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
            symbolic_assembly_backend: BvpSymbolicAssemblyBackend::ExprLegacy,
            matrix_backend_override: None,
            banded_linear_solver_config: crate::somelinalg::banded::LinearSolverConfig::default(),
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
            symbolic_assembly_backend: BvpSymbolicAssemblyBackend::ExprLegacy,
            matrix_backend_override: None,
            banded_linear_solver_config: crate::somelinalg::banded::LinearSolverConfig::default(),
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
            symbolic_assembly_backend: BvpSymbolicAssemblyBackend::ExprLegacy,
            matrix_backend_override: None,
            banded_linear_solver_config: crate::somelinalg::banded::LinearSolverConfig::default(),
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
            symbolic_assembly_backend: BvpSymbolicAssemblyBackend::ExprLegacy,
            matrix_backend_override: None,
            banded_linear_solver_config: crate::somelinalg::banded::LinearSolverConfig::default(),
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
                profile:
                    crate::numerical::BVP_Damp::generated_solver_handoff::AotBuildProfile::Release,
            },
            aot_compile_config: AotCompileConfig::default(),
            aot_codegen_backend: AotCodegenBackend::Rust,
            aot_c_compiler: None,
            aot_chunking_policy: AotChunkingPolicy::default(),
            symbolic_assembly_backend: BvpSymbolicAssemblyBackend::ExprLegacy,
            matrix_backend_override: None,
            banded_linear_solver_config: crate::somelinalg::banded::LinearSolverConfig::default(),
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
        let (eq_system, values, arg, border_conditions) = banded_aot_lifecycle_inputs();

        let make_request = |resolver: Option<AotResolver>, build_policy: AotBuildPolicy| {
            DampedSolverBuildRequest {
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
                symbolic_assembly_backend: BvpSymbolicAssemblyBackend::AtomView,
                matrix_backend_override: Some(MatrixBackend::Banded),
                banded_linear_solver_config:
                    crate::somelinalg::banded::LinearSolverConfig::faithful_banded(),
            }
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

        let make_request = |resolver: Option<AotResolver>, build_policy: AotBuildPolicy| {
            FrozenSolverBuildRequest {
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
                symbolic_assembly_backend: BvpSymbolicAssemblyBackend::AtomView,
                matrix_backend_override: Some(MatrixBackend::Banded),
                banded_linear_solver_config:
                    crate::somelinalg::banded::LinearSolverConfig::faithful_banded(),
            }
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
            symbolic_assembly_backend: BvpSymbolicAssemblyBackend::ExprLegacy,
            matrix_backend_override: None,
            banded_linear_solver_config: crate::somelinalg::banded::LinearSolverConfig::default(),
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
            symbolic_assembly_backend: BvpSymbolicAssemblyBackend::ExprLegacy,
            matrix_backend_override: None,
            banded_linear_solver_config: crate::somelinalg::banded::LinearSolverConfig::default(),
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
            symbolic_assembly_backend: BvpSymbolicAssemblyBackend::ExprLegacy,
            matrix_backend_override: None,
            banded_linear_solver_config: crate::somelinalg::banded::LinearSolverConfig::default(),
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
}
