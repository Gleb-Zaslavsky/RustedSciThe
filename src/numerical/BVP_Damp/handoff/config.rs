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
    /// Typed symbolic/backend preparation stages captured before the solve.
    pub generation_telemetry: Option<BvpGenerationTelemetrySnapshot>,
    /// Typed Atom-native preparation telemetry, when the selected route used it.
    pub atom_discretization_telemetry:
        Option<crate::symbolic::bvp::telemetry::BvpAtomDiscretizationTelemetrySnapshot>,
    /// Runtime callback telemetry for the ExprLegacy Lambdify route.
    pub legacy_lambdify_telemetry: Option<BvpLambdifyTelemetry>,
    /// Runtime callback telemetry for the AtomView Lambdify route.
    pub atom_lambdify_telemetry: Option<BvpLambdifyTelemetry>,
    /// Immutable snapshot of the direct no-Mutex Banded callback telemetry.
    pub direct_banded_jacobian_telemetry: Option<BvpDirectJacobianTelemetry>,
    /// Shared typed AOT telemetry handle for live post-solve snapshots.
    pub aot_telemetry: Option<BvpAotTelemetry>,
    /// Numeric parameter binding shared by prepared Lambdify callbacks.
    pub(crate) parameter_binding:
        Option<crate::symbolic::bvp::parameter_binding::BvpParameterBindingHandle>,
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
    /// Typed symbolic/backend preparation stages captured before the solve.
    pub generation_telemetry: Option<BvpGenerationTelemetrySnapshot>,
    /// Typed Atom-native preparation telemetry, when the selected route used it.
    pub atom_discretization_telemetry:
        Option<crate::symbolic::bvp::telemetry::BvpAtomDiscretizationTelemetrySnapshot>,
    /// Runtime callback telemetry for the ExprLegacy Lambdify route.
    pub legacy_lambdify_telemetry: Option<BvpLambdifyTelemetry>,
    /// Runtime callback telemetry for the AtomView Lambdify route.
    pub atom_lambdify_telemetry: Option<BvpLambdifyTelemetry>,
    /// Immutable snapshot of the direct no-Mutex Banded callback telemetry.
    pub direct_banded_jacobian_telemetry: Option<BvpDirectJacobianTelemetry>,
    /// Numeric parameter binding shared by prepared Lambdify callbacks.
    pub(crate) parameter_binding:
        Option<crate::symbolic::bvp::parameter_binding::BvpParameterBindingHandle>,
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
#[derive(Clone, Debug)]
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
    /// AtomView lowering optimization profile used when materializing AOT artifacts.
    pub atom_optimization_profile: AtomOptimizationProfile,
    /// Symbolic assembly backend used before lambdify/AOT lowering.
    pub symbolic_assembly_backend: BvpSymbolicAssemblyBackend,
    /// Optional matrix backend override for the modern generated BVP path.
    pub matrix_backend_override: Option<MatrixBackend>,
    /// Native linear solver configuration used by the generated banded runtime path.
    pub banded_linear_solver_config: LinearSolverConfig,
    /// Runtime collection mode for Lambdify callback telemetry. This has no
    /// effect on compiled AOT callbacks.
    pub lambdify_telemetry_mode: BvpLambdifyTelemetryMode,
    /// Runtime collection mode for AtomView AOT preparation and generated
    /// callback lifecycle telemetry. `Off` is the zero-overhead default.
    pub aot_telemetry_mode: BvpAotTelemetryMode,
    /// Runtime execution policy for pure Lambdify residual/Jacobian callbacks.
    /// This is independent from AOT execution policy.
    pub lambdify_execution_policy: BvpLambdifyExecutionPolicy,
    /// Opt-in structured decision logging for the BVP solver runtime.
    pub bvp_logging_mode: BvpLoggingMode,
    /// Maximum number of retained typed decision events per solve.
    pub bvp_logging_max_events: usize,
    /// Collection policy for solver counters and stage timings.
    pub bvp_telemetry_mode: BvpTelemetryMode,
}

impl Default for GeneratedBackendConfig {
    fn default() -> Self {
        // The public default is the production BVP route: AtomView for
        // symbolic preparation and tcc for on-demand C AOT.  Rust and
        // ExprLegacy remain explicit compatibility/diagnostic choices.
        Self {
            backend_policy_override: None,
            resolver: None,
            aot_execution_policy: AotExecutionPolicy::Auto,
            aot_build_policy: AotBuildPolicy::UseIfAvailable,
            aot_compile_config: AotCompileConfig::default(),
            aot_codegen_backend: AotCodegenBackend::C,
            aot_c_compiler: Some("tcc".to_string()),
            aot_chunking_policy: AotChunkingPolicy::default(),
            atom_optimization_profile: AtomOptimizationProfile::Full,
            symbolic_assembly_backend: BvpSymbolicAssemblyBackend::AtomView,
            matrix_backend_override: None,
            banded_linear_solver_config: LinearSolverConfig::default(),
            lambdify_telemetry_mode: BvpLambdifyTelemetryMode::Off,
            aot_telemetry_mode: BvpAotTelemetryMode::Off,
            lambdify_execution_policy: BvpLambdifyExecutionPolicy::default(),
            bvp_logging_mode: BvpLoggingMode::Off,
            bvp_logging_max_events: BvpLoggingConfig::default().max_events,
            bvp_telemetry_mode: BvpTelemetryMode::Counters,
        }
    }
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

    /// Sets the runtime telemetry mode for Lambdify callbacks.
    pub fn with_lambdify_telemetry_mode(mut self, mode: BvpLambdifyTelemetryMode) -> Self {
        self.lambdify_telemetry_mode = mode;
        self
    }

    /// Sets typed telemetry for the AtomView-native AOT route.
    pub fn with_aot_telemetry_mode(mut self, mode: BvpAotTelemetryMode) -> Self {
        self.aot_telemetry_mode = mode;
        self
    }

    /// Selects how pure Lambdify residual/Jacobian callbacks execute.
    ///
    /// This policy is separate from AOT execution and is applied only when
    /// the selected route is Lambdify. The default preserves the historical
    /// parallel callback behavior.
    pub fn with_lambdify_execution_policy(mut self, policy: BvpLambdifyExecutionPolicy) -> Self {
        self.lambdify_execution_policy = policy;
        self
    }

    /// Enables typed solver decision logging without changing numerical code.
    pub fn with_bvp_logging_mode(mut self, mode: BvpLoggingMode) -> Self {
        self.bvp_logging_mode = mode;
        self
    }

    /// Sets the bounded capacity of the typed decision trace.
    pub fn with_bvp_logging_max_events(mut self, max_events: usize) -> Self {
        self.bvp_logging_max_events = max_events;
        self
    }

    /// Sets the complete structured logging policy.
    pub fn with_bvp_logging_config(mut self, config: BvpLoggingConfig) -> Self {
        self.bvp_logging_mode = config.mode;
        self.bvp_logging_max_events = config.max_events;
        self
    }

    /// Selects the solver telemetry collection policy. `Off` avoids counter
    /// increments, timers, and callback-stage collection on the hot path.
    pub fn with_bvp_telemetry_mode(mut self, mode: BvpTelemetryMode) -> Self {
        self.bvp_telemetry_mode = mode;
        self
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
            aot_codegen_backend: AotCodegenBackend::C,
            aot_c_compiler: Some("tcc".to_string()),
            aot_chunking_policy: AotChunkingPolicy::default(),
            atom_optimization_profile: AtomOptimizationProfile::Full,
            symbolic_assembly_backend: BvpSymbolicAssemblyBackend::AtomView,
            matrix_backend_override: None,
            banded_linear_solver_config: crate::somelinalg::banded::LinearSolverConfig::default(),
            lambdify_telemetry_mode: BvpLambdifyTelemetryMode::Off,
            aot_telemetry_mode: BvpAotTelemetryMode::Off,
            lambdify_execution_policy: BvpLambdifyExecutionPolicy::default(),
            bvp_logging_mode: BvpLoggingMode::Off,
            bvp_logging_max_events: BvpLoggingConfig::default().max_events,
            bvp_telemetry_mode: BvpTelemetryMode::Counters,
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

    /// Sets the AtomView lowering optimization profile for generated AOT artifacts.
    ///
    /// `Full` preserves the production default. `NoCse` is intended for
    /// correctness/performance comparisons around structural common
    /// subexpression elimination.
    pub fn with_atom_optimization_profile(mut self, profile: AtomOptimizationProfile) -> Self {
        self.atom_optimization_profile = profile;
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
                // The historical nalgebra `CsMatrix` BVP runtime backend was
                // never completed. Keep old provider-level requests on the
                // production sparse route instead of selecting the removed
                // `Sparse_2` trait path.
                MatrixBackend::CsMatrix => "Sparse",
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
