use crate::numerical::BVP_Damp::BVP_traits::{Fun, Jac};
use crate::symbolic::codegen_aot_build::{
    AotBuildProfile as LifecycleBuildProfile, AotBuildRequest,
};
use crate::symbolic::codegen_aot_driver::generated_aot_crate_from_prepared_sparse_problem;
use crate::symbolic::codegen_aot_registry::AotRegistry;
use crate::symbolic::codegen_aot_resolution::AotResolver;
use crate::symbolic::codegen_backend_selection::{BackendSelectionPolicy, SelectedBackendKind};
use crate::symbolic::codegen_manifest::PreparedProblemManifest;
use crate::symbolic::codegen_orchestrator::ParallelExecutorConfig;
use crate::symbolic::codegen_runtime_api::ResidualChunkingStrategy;
use crate::symbolic::codegen_tasks::SparseChunkingStrategy;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions_BVP::{
    BvpBackendIntegrationError, BvpLegacySolverBundle, BvpSparseSolverBundle, Jacobian,
};
use log::{error, info};
use std::collections::HashMap;
use std::path::PathBuf;

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
    /// Optional chunking overrides for residual and sparse Jacobian generation.
    pub aot_chunking_policy: AotChunkingPolicy,
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

impl GeneratedBackendConfig {
    /// Creates an empty generated-backend configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates the default sparse/BVP production-oriented generated-backend configuration.
    ///
    /// This prefers compiled AOT when available and otherwise falls back to the
    /// established lambdify path without requiring the caller to know the exact
    /// backend-selection enum variant.
    pub fn sparse_defaults() -> Self {
        Self::new()
            .with_backend_policy_override(Some(BackendSelectionPolicy::PreferAotThenLambdify))
    }

    /// Creates a sparse generated-backend configuration that requires a prebuilt AOT artifact.
    pub fn sparse_require_prebuilt() -> Self {
        Self::sparse_defaults().with_aot_build_policy(AotBuildPolicy::RequirePrebuilt)
    }

    /// Creates a sparse generated-backend configuration that builds a release artifact on demand.
    pub fn sparse_build_if_missing_release() -> Self {
        Self::sparse_defaults().with_aot_build_policy(AotBuildPolicy::BuildIfMissing {
            profile: AotBuildProfile::Release,
        })
    }

    /// Creates a generated-backend configuration from a high-level sparse mode.
    pub fn from_sparse_mode(mode: SparseGeneratedBackendMode) -> Self {
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
            aot_chunking_policy: AotChunkingPolicy::default(),
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

    /// Sets solver-level chunking overrides for generated AOT plans.
    pub fn with_aot_chunking_policy(mut self, policy: AotChunkingPolicy) -> Self {
        self.aot_chunking_policy = policy;
        self
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
    /// Optional chunking overrides carried into generated backend setup.
    pub aot_chunking_policy: AotChunkingPolicy,
}

impl DampedSolverBuildRequest {
    /// Builds a solver-ready callback and metadata state.
    pub fn generate(self) -> Result<DampedGeneratedSolverState, BvpBackendIntegrationError> {
        generate_damped_solver_state(
            self.eq_system,
            self.values,
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
            self.aot_chunking_policy,
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
    /// Optional chunking overrides carried into generated backend setup.
    pub aot_chunking_policy: AotChunkingPolicy,
}

impl FrozenSolverBuildRequest {
    /// Builds a solver-ready callback and metadata state.
    pub fn generate(self) -> Result<FrozenGeneratedSolverState, BvpBackendIntegrationError> {
        generate_frozen_solver_state(
            self.eq_system,
            self.values,
            self.arg,
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
            self.aot_chunking_policy,
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
    aot_chunking_policy: AotChunkingPolicy,
) -> Result<DampedGeneratedSolverState, BvpBackendIntegrationError> {
    if method == "Sparse" {
        let jacobian_instance = Jacobian::new();
        let backend_policy = effective_backend_policy_for_build(backend_policy, aot_build_policy);
        let bundle = match (
            aot_chunking_policy.residual,
            aot_chunking_policy.sparse_jacobian,
        ) {
            (None, None) => jacobian_instance
                .try_generate_sparse_solver_bundle_with_backend_selection(
                    eq_system,
                    values,
                    arg,
                    None,
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
                )?,
            (residual, sparse_jacobian) => jacobian_instance
                .try_generate_sparse_solver_bundle_with_backend_selection_and_chunking(
                    eq_system,
                    values,
                    arg,
                    None,
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
                )?,
        };
        let (bundle, updated_resolver) =
            enforce_build_policy_on_sparse_bundle(bundle, resolver, aot_build_policy)?;
        let bundle = apply_execution_policy_to_sparse_bundle(bundle, aot_execution_policy)?;
        Ok(damped_state_from_sparse_solver_bundle(
            bundle,
            updated_resolver,
        ))
    } else {
        let legacy_bundle = Jacobian::new().generate_legacy_solver_bundle_with_params(
            eq_system,
            values,
            arg,
            None,
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
    aot_chunking_policy: AotChunkingPolicy,
) -> Result<FrozenGeneratedSolverState, BvpBackendIntegrationError> {
    if method == "Sparse" {
        let jacobian_instance = Jacobian::new();
        let backend_policy = effective_backend_policy_for_build(backend_policy, aot_build_policy);
        let bundle = match (
            aot_chunking_policy.residual,
            aot_chunking_policy.sparse_jacobian,
        ) {
            (None, None) => jacobian_instance
                .try_generate_sparse_solver_bundle_with_backend_selection(
                    eq_system,
                    values,
                    arg,
                    None,
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
                    backend_policy,
                    resolver,
                )?,
            (residual, sparse_jacobian) => jacobian_instance
                .try_generate_sparse_solver_bundle_with_backend_selection_and_chunking(
                    eq_system,
                    values,
                    arg,
                    None,
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
                    backend_policy,
                    resolver,
                    residual.unwrap_or(ResidualChunkingStrategy::Whole),
                    sparse_jacobian.unwrap_or(SparseChunkingStrategy::Whole),
                )?,
        };
        let (bundle, updated_resolver) =
            enforce_build_policy_on_sparse_bundle(bundle, resolver, aot_build_policy)?;
        let bundle = apply_execution_policy_to_sparse_bundle(bundle, aot_execution_policy)?;
        Ok(frozen_state_from_sparse_solver_bundle(
            bundle,
            updated_resolver,
        ))
    } else {
        let legacy_bundle = Jacobian::new().generate_legacy_solver_bundle_with_params(
            eq_system,
            values,
            arg,
            None,
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

fn effective_backend_policy_for_build(
    backend_policy: BackendSelectionPolicy,
    build_policy: AotBuildPolicy,
) -> BackendSelectionPolicy {
    match build_policy {
        AotBuildPolicy::UseIfAvailable
        | AotBuildPolicy::BuildIfMissing { .. }
        | AotBuildPolicy::RebuildAlways { .. } => backend_policy,
        AotBuildPolicy::RequirePrebuilt => {
            if backend_policy_targets_aot(backend_policy) {
                BackendSelectionPolicy::AotOnly
            } else {
                backend_policy
            }
        }
    }
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

fn to_lifecycle_build_profile(profile: AotBuildProfile) -> LifecycleBuildProfile {
    match profile {
        AotBuildProfile::Release => LifecycleBuildProfile::Release,
        AotBuildProfile::Debug => LifecycleBuildProfile::Debug,
    }
}

fn try_materialize_and_build_sparse_aot_bundle(
    bundle: &BvpSparseSolverBundle,
    resolver: Option<&AotResolver>,
    profile: AotBuildProfile,
) -> Result<AotResolver, BvpBackendIntegrationError> {
    let selected = bundle.execution.selected();
    let problem_key = selected.prepared_problem.problem_key();
    let crate_name = format!(
        "generated_bvp_sparse_{}",
        sanitize_generated_name(&problem_key)
    );
    let module_name = format!(
        "generated_bvp_module_{}",
        sanitize_generated_name(&problem_key)
    );
    let output_parent_dir = build_output_parent_for_problem(&problem_key);
    let crate_spec = generated_aot_crate_from_prepared_sparse_problem(
        &crate_name,
        &module_name,
        &selected.prepared_problem.as_prepared_problem(),
    );
    let request = AotBuildRequest::new(
        crate_spec,
        output_parent_dir,
        to_lifecycle_build_profile(profile),
    );
    info!(
        "materializing sparse AOT crate for problem_key={} with build_profile={:?}",
        problem_key, profile
    );
    let build = request.materialize().map_err(|err| {
        BvpBackendIntegrationError::AutomaticAotBuildFailed {
            problem_key: problem_key.clone(),
            message: err.to_string(),
        }
    })?;
    info!(
        "executing sparse AOT build for problem_key={} in crate_dir={}",
        problem_key,
        build.written.crate_dir.display()
    );
    let executed =
        build
            .execute()
            .map_err(|err| BvpBackendIntegrationError::AutomaticAotBuildFailed {
                problem_key: problem_key.clone(),
                message: err.to_string(),
            })?;
    if executed.succeeded() {
        info!(
            "sparse AOT build succeeded for problem_key={} with profile={:?}",
            problem_key, profile
        );
        let manifest = PreparedProblemManifest::from(
            &crate::symbolic::codegen_provider_api::PreparedProblem::sparse(
                selected.prepared_problem.as_prepared_problem(),
            ),
        );
        let mut registry = resolver
            .map(|existing| existing.registry().clone())
            .unwrap_or_else(AotRegistry::new);
        registry.register_materialized_build(manifest, &build);
        Ok(AotResolver::new(registry))
    } else {
        error!(
            "sparse AOT build failed for problem_key={} with status={:?}",
            problem_key, executed.status_code
        );
        Err(BvpBackendIntegrationError::AutomaticAotBuildFailed {
            problem_key,
            message: format!(
                "cargo build failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
                executed.status_code, executed.stdout, executed.stderr
            ),
        })
    }
}

fn enforce_build_policy_on_sparse_bundle(
    bundle: BvpSparseSolverBundle,
    resolver: Option<&AotResolver>,
    build_policy: AotBuildPolicy,
) -> Result<(BvpSparseSolverBundle, Option<AotResolver>), BvpBackendIntegrationError> {
    let problem_key = bundle.execution.selected().prepared_problem.problem_key();
    let effective_backend = bundle.effective_backend();
    info!(
        "enforcing sparse AOT build policy {} for problem_key={} with effective_backend={:?}",
        build_policy.as_str(),
        problem_key,
        effective_backend
    );

    match build_policy {
        AotBuildPolicy::UseIfAvailable => {
            if matches!(effective_backend, SelectedBackendKind::AotCompiled)
                && !bundle.is_runtime_callable()
            {
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
            } else if !bundle.is_runtime_callable() {
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
            if matches!(effective_backend, SelectedBackendKind::AotCompiled)
                && bundle.is_runtime_callable()
            {
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
                info!(
                    "compiled sparse AOT backend missing or not callable for problem_key={}, building with profile={:?}",
                    problem_key, profile
                );
                let updated_resolver =
                    try_materialize_and_build_sparse_aot_bundle(&bundle, resolver, profile)?;
                if bundle.is_runtime_callable() {
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
            info!(
                "forcing sparse AOT rebuild for problem_key={} with profile={:?}",
                problem_key, profile
            );
            let updated_resolver =
                try_materialize_and_build_sparse_aot_bundle(&bundle, resolver, profile)?;
            if bundle.is_runtime_callable() {
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
    let problem_key = bundle.execution.selected().prepared_problem.problem_key();
    info!(
        "applying sparse AOT execution policy {} for problem_key={} with effective_backend={:?}",
        policy.as_str(),
        problem_key,
        bundle.effective_backend()
    );
    match policy {
        AotExecutionPolicy::Auto => {
            info!(
                "keeping default sparse runtime callback binding for problem_key={}",
                problem_key
            );
        }
        AotExecutionPolicy::SequentialOnly => {
            if matches!(bundle.effective_backend(), SelectedBackendKind::AotCompiled)
                && !bundle.rebind_linked_runtime_callbacks(None, None)
            {
                return Err(BvpBackendIntegrationError::CompiledAotRuntimeUnavailable {
                    problem_key,
                });
            }
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
    let bounds_vec = bundle.bounds_vec.clone().unwrap_or_default();
    let rel_tolerance_vec = bundle.rel_tolerance_vec.clone().unwrap_or_default();
    let variable_string = bundle.variable_string.clone();
    let bandwidth = bundle.bandwidth.unwrap_or((0, 0));
    let bc_position_and_value = bundle.bc_position_and_value.clone();

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
    }
}

/// Converts the assembled sparse symbolic/provider bundle into a frozen solver runtime state.
pub fn frozen_state_from_sparse_solver_bundle(
    bundle: BvpSparseSolverBundle,
    updated_resolver: Option<AotResolver>,
) -> FrozenGeneratedSolverState {
    let variable_string = bundle.variable_string.clone();
    let bandwidth = bundle.bandwidth.unwrap_or((0, 0));

    let (fun, jac) = bundle.into_runtime_callbacks().unwrap_or_else(|| {
        panic!("Frozen BVP sparse solver bundle did not provide runtime callbacks")
    });

    FrozenGeneratedSolverState {
        fun,
        jac: Some(jac),
        variable_string,
        bandwidth,
        updated_resolver,
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
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AotBuildPolicy, AotChunkingPolicy, AotExecutionPolicy, DampedSolverBuildRequest,
        FrozenSolverBuildRequest, build_output_parent_for_problem, sanitize_generated_name,
    };
    use crate::symbolic::codegen_aot_build::{AotBuildProfile, AotBuildRequest};
    use crate::symbolic::codegen_aot_driver::generated_aot_crate_from_prepared_problem;
    use crate::symbolic::codegen_aot_registry::AotRegistry;
    use crate::symbolic::codegen_aot_resolution::AotResolver;
    use crate::symbolic::codegen_aot_runtime_link::{
        LinkedResidualChunk, LinkedSparseAotBackend, LinkedSparseJacobianChunk,
        register_linked_sparse_backend, unregister_linked_sparse_backend,
    };
    use crate::symbolic::codegen_backend_selection::{BackendSelectionPolicy, SelectedBackendKind};
    use crate::symbolic::codegen_manifest::PreparedProblemManifest;
    use crate::symbolic::codegen_orchestrator::{ParallelExecutorConfig, ParallelFallbackPolicy};
    use crate::symbolic::codegen_provider_api::PreparedProblem;
    use crate::symbolic::symbolic_engine::Expr;
    use crate::symbolic::symbolic_functions_BVP::{BvpBackendIntegrationError, Jacobian};
    use faer::Col;
    use std::collections::HashMap;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Arc;
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
            crate::symbolic::codegen_runtime_api::recommended_residual_chunking_for_parallelism(
                staged.vector_of_functions.len(),
                4,
            ),
            crate::symbolic::codegen_runtime_api::recommended_row_chunking_for_parallelism(
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
            crate::symbolic::codegen_runtime_api::recommended_residual_chunking_for_parallelism(
                staged.vector_of_functions.len(),
                4,
            ),
            crate::symbolic::codegen_runtime_api::recommended_row_chunking_for_parallelism(
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
            generated_aot_crate_from_prepared_problem(
                "generated_bvp_solver_handoff_fixture",
                "generated_bvp_solver_handoff_module",
                &prepared,
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
            crate::symbolic::codegen_backend_selection::SelectedBackendKind::AotCompiled
        );
        let request = DampedSolverBuildRequest {
            eq_system,
            values,
            arg,
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
            aot_chunking_policy: AotChunkingPolicy::default(),
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
            crate::symbolic::codegen_backend_selection::SelectedBackendKind::AotCompiled
        );
        let request = FrozenSolverBuildRequest {
            eq_system,
            values,
            arg,
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
            aot_chunking_policy: AotChunkingPolicy::default(),
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
            aot_chunking_policy: AotChunkingPolicy::default(),
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

        unregister_linked_sparse_backend(&problem_key);
    }

    #[test]
    fn require_prebuilt_errors_when_compiled_backend_is_missing() {
        let (eq_system, values, arg, border_conditions) = real_bvp_inputs();
        let request = DampedSolverBuildRequest {
            eq_system,
            values,
            arg,
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
            aot_chunking_policy: AotChunkingPolicy::default(),
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
            crate::symbolic::codegen_runtime_api::recommended_residual_chunking_for_parallelism(
                staged.vector_of_functions.len(),
                4,
            ),
            crate::symbolic::codegen_runtime_api::recommended_row_chunking_for_parallelism(
                staged.vector_of_functions.len(),
                4,
            ),
        );
        let problem_key = prepared_bridge.problem_key();
        let request = FrozenSolverBuildRequest {
            eq_system,
            values,
            arg,
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
            aot_chunking_policy: AotChunkingPolicy::default(),
        };

        let state = request
            .generate()
            .expect("build-if-missing should build artifact and keep runtime callable");
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

        let build_dir = build_output_parent_for_problem(&problem_key);
        let crate_name = format!(
            "generated_bvp_sparse_{}",
            sanitize_generated_name(&problem_key)
        );
        let expected_rlib = build_dir
            .join(&crate_name)
            .join("target")
            .join("release")
            .join(format!("lib{crate_name}.rlib"));
        assert!(
            expected_rlib.exists(),
            "build-if-missing should produce compiled rlib at {}",
            expected_rlib.display()
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
            crate::symbolic::codegen_runtime_api::recommended_residual_chunking_for_parallelism(
                staged.vector_of_functions.len(),
                4,
            ),
            crate::symbolic::codegen_runtime_api::recommended_row_chunking_for_parallelism(
                staged.vector_of_functions.len(),
                4,
            ),
        );
        let prepared = PreparedProblem::sparse(prepared_bridge.as_prepared_problem());
        let manifest = PreparedProblemManifest::from(&prepared);
        let dir = unique_test_artifact_dir(&prepared_bridge.problem_key());
        fs::create_dir_all(&dir).expect("workspace test dir should be creatable");
        let build = AotBuildRequest::new(
            generated_aot_crate_from_prepared_problem(
                "generated_bvp_solver_registered_only_fixture",
                "generated_bvp_solver_registered_only_module",
                &prepared,
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
            aot_chunking_policy: AotChunkingPolicy::default(),
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
}
