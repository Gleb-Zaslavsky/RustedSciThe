/// Configuration parameters for the damped Newton solver
///
/// Controls damping behavior, Jacobian reuse strategy, and adaptive grid refinement
#[derive(Debug, Clone, PartialEq)]
pub struct SolverParams {
    /// Maximum iterations before Jacobian recalculation (default: 3)
    pub max_jac: Option<usize>,
    /// Maximum damping iterations per Newton step (default: 5)
    pub max_damp_iter: Option<usize>,
    /// Factor for reducing damping coefficient (default: 0.5)
    pub damp_factor: Option<f64>,
    /// Adaptive grid refinement configuration
    pub adaptive: Option<AdaptiveGridConfig>,
}

/// Configuration for adaptive grid refinement
///
/// Defines when and how to refine the computational mesh
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveGridConfig {
    /// Refinement criterion version (currently only version 1 supported)
    pub version: usize,
    /// Maximum number of grid refinements allowed
    pub max_refinements: usize,
    /// Grid refinement algorithm to use
    pub grid_method: GridRefinementMethod,
}

/// Discretization scheme used to assemble the BVP residual.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BvpDerivativeScheme {
    /// One-sided forward discretization used by the legacy default route.
    Forward,
    /// Trapezoidal/collocation-style discretization using both interval endpoints.
    Trapezoid,
}

impl BvpDerivativeScheme {
    pub(crate) fn as_legacy_str(self) -> &'static str {
        match self {
            Self::Forward => DEFAULT_FORWARD_SCHEME,
            Self::Trapezoid => "trapezoid",
        }
    }
}

impl Default for SolverParams {
    fn default() -> Self {
        Self {
            max_jac: Some(3),
            max_damp_iter: Some(5),
            damp_factor: Some(0.5),
            adaptive: None,
        }
    }
}

/// User-facing setup options for the damped BVP solver.
#[derive(Clone)]
pub struct DampedSolverOptions {
    /// Discretization scheme name.
    pub scheme: String,
    /// Nonlinear solver strategy name.
    pub strategy: String,
    /// Optional detailed strategy configuration.
    pub strategy_params: Option<SolverParams>,
    /// Optional linear-system method override.
    pub linear_sys_method: Option<String>,
    /// Matrix backend/method selector.
    pub method: String,
    /// Absolute convergence tolerance.
    pub abs_tolerance: f64,
    /// Optional per-variable relative tolerances.
    pub rel_tolerance: Option<HashMap<String, f64>>,
    /// Maximum nonlinear iterations.
    pub max_iterations: usize,
    /// Optional per-variable bounds.
    pub bounds: Option<HashMap<String, (f64, f64)>>,
    /// Optional logging level.
    pub loglevel: Option<String>,
    /// Generated-backend configuration used by sparse solver paths.
    pub generated_backend_config: GeneratedBackendConfig,
}

impl DampedSolverOptions {
    /// Creates damped solver options from explicit values.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        scheme: String,
        strategy: String,
        strategy_params: Option<SolverParams>,
        linear_sys_method: Option<String>,
        method: String,
        abs_tolerance: f64,
        rel_tolerance: Option<HashMap<String, f64>>,
        max_iterations: usize,
        bounds: Option<HashMap<String, (f64, f64)>>,
        loglevel: Option<String>,
    ) -> Self {
        Self {
            scheme,
            strategy,
            strategy_params,
            linear_sys_method,
            method,
            abs_tolerance,
            rel_tolerance,
            max_iterations,
            bounds,
            loglevel,
            generated_backend_config: GeneratedBackendConfig::default(),
        }
    }

    /// Attaches an explicit generated-backend configuration.
    pub fn with_generated_backend_config(mut self, config: GeneratedBackendConfig) -> Self {
        self.generated_backend_config = config;
        self
    }

    /// Selects the runtime telemetry level for Lambdify callbacks.
    ///
    /// The default is [`BvpLambdifyTelemetryMode::Off`]. Use `Counters` for
    /// cheap call counts or `Detailed` when callback wall-clock timings are
    /// needed for a diagnostic/story run.
    pub fn with_lambdify_telemetry_mode(mut self, mode: BvpLambdifyTelemetryMode) -> Self {
        self.generated_backend_config = self
            .generated_backend_config
            .with_lambdify_telemetry_mode(mode);
        self
    }

    /// Selects typed solver decision logging for diagnostic runs.
    pub fn with_bvp_logging_mode(mut self, mode: BvpLoggingMode) -> Self {
        self.generated_backend_config = self.generated_backend_config.with_bvp_logging_mode(mode);
        self
    }

    /// Configures bounded typed decision logging for the solver runtime.
    pub fn with_bvp_logging_config(mut self, config: BvpLoggingConfig) -> Self {
        self.generated_backend_config = self
            .generated_backend_config
            .with_bvp_logging_config(config);
        self
    }

    /// Selects whether solver counters and stage timers are collected.
    pub fn with_bvp_telemetry_mode(mut self, mode: BvpTelemetryMode) -> Self {
        self.generated_backend_config = self.generated_backend_config.with_bvp_telemetry_mode(mode);
        self
    }

    /// Returns options with an explicit AtomView AOT optimization profile.
    ///
    /// The default is `AtomOptimizationProfile::Full`, which preserves the
    /// historical CSE-enabled pipeline. Diagnostic profiles such as `NoCse`
    /// are useful for correctness and performance A/B checks.
    pub fn with_atom_optimization_profile(mut self, profile: AtomOptimizationProfile) -> Self {
        self.generated_backend_config = self
            .generated_backend_config
            .with_atom_optimization_profile(profile);
        self
    }

    /// Selects the residual discretization scheme with a typed public API.
    ///
    /// The solver still stores the legacy string internally because the older
    /// symbolic/numeric assembly layers use string flags. New user code should
    /// prefer this method over passing raw `"forward"` / `"trapezoid"` strings.
    pub fn with_scheme(mut self, scheme: BvpDerivativeScheme) -> Self {
        self.scheme = scheme.as_legacy_str().to_string();
        self
    }

    /// Selects the legacy forward derivative discretization.
    pub fn forward_derivative(self) -> Self {
        self.with_scheme(BvpDerivativeScheme::Forward)
    }

    /// Selects the trapezoid derivative discretization.
    pub fn trapezoid_derivative(self) -> Self {
        self.with_scheme(BvpDerivativeScheme::Trapezoid)
    }

    /// Compatibility escape hatch for legacy/custom scheme strings.
    pub fn with_scheme_name(mut self, scheme: impl Into<String>) -> Self {
        self.scheme = scheme.into();
        self
    }

    /// Overrides the native linear solver configuration used by generated banded callbacks.
    pub fn with_banded_linear_solver_config(mut self, config: LinearSolverConfig) -> Self {
        self.generated_backend_config = self
            .generated_backend_config
            .with_banded_linear_solver_config(config);
        self
    }

    /// Overrides detailed nonlinear strategy parameters.
    pub fn with_strategy_params(mut self, strategy_params: Option<SolverParams>) -> Self {
        self.strategy_params = strategy_params;
        self
    }

    /// Overrides the absolute convergence tolerance.
    pub fn with_abs_tolerance(mut self, abs_tolerance: f64) -> Self {
        self.abs_tolerance = abs_tolerance;
        self
    }

    /// Overrides per-variable relative tolerances.
    pub fn with_rel_tolerance(mut self, rel_tolerance: HashMap<String, f64>) -> Self {
        self.rel_tolerance = Some(rel_tolerance);
        self
    }

    /// Overrides the nonlinear iteration limit.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Overrides per-variable bounds.
    pub fn with_bounds(mut self, bounds: HashMap<String, (f64, f64)>) -> Self {
        self.bounds = Some(bounds);
        self
    }

    /// Overrides the solver log level.
    pub fn with_loglevel(mut self, loglevel: Option<String>) -> Self {
        self.loglevel = loglevel;
        self
    }

    /// Attaches a high-level sparse generated-backend mode.
    pub fn with_sparse_generated_backend_mode(mut self, mode: SparseGeneratedBackendMode) -> Self {
        self.generated_backend_config = GeneratedBackendConfig::from_sparse_mode(mode);
        self
    }

    /// Attaches a high-level banded generated-backend mode.
    ///
    /// Banded modes keep the outer nonlinear solver unchanged, but route the
    /// generated callback stack through native `Banded` matrix assembly and the
    /// faithful LAPACK-style banded LU linear solver with `refine = 0`.
    pub fn with_banded_generated_backend_mode(mut self, mode: BandedGeneratedBackendMode) -> Self {
        self.generated_backend_config = GeneratedBackendConfig::from_banded_mode(mode);
        self
    }

    /// Selects the symbolic assembly backend used before lambdify/AOT lowering.
    pub fn with_symbolic_assembly_backend(mut self, backend: BvpSymbolicAssemblyBackend) -> Self {
        self.generated_backend_config = self
            .generated_backend_config
            .with_symbolic_assembly_backend(backend);
        self
    }

    /// Selects the matrix backend through the typed API.
    ///
    /// The historical `method: String` field is retained for compatibility,
    /// while the normalized generated-backend configuration becomes the source
    /// of truth for new callers.
    pub fn with_matrix_backend(mut self, backend: MatrixBackend) -> Self {
        self.generated_backend_config = self
            .generated_backend_config
            .with_matrix_backend_override(backend);
        self
    }

    /// Creates production-oriented sparse damped solver options with standard defaults.
    ///
    /// This is the preferred starting point for most BVP users:
    /// - `forward` discretization
    /// - `Damped` nonlinear strategy
    /// - `Sparse` matrix backend
    /// - generated backend defaults that prefer AOT and fall back to lambdify
    pub fn sparse_damped() -> Self {
        Self::default().with_sparse_generated_backend_mode(SparseGeneratedBackendMode::Defaults)
    }

    /// Creates production-oriented banded damped solver options with standard defaults.
    ///
    /// This selects generated `Banded` matrix callbacks and faithful
    /// LAPACK-style native banded LU (`refine = 0`) while preserving the
    /// regular damped Newton nonlinear strategy.
    pub fn banded_damped() -> Self {
        Self::default().with_banded_generated_backend_mode(BandedGeneratedBackendMode::Defaults)
    }

    /// Creates production-oriented dense damped solver options with standard defaults.
    ///
    /// This is the preferred starting point for dense BVP users that do not
    /// need sparse/AOT-specific behavior.
    pub fn dense_damped() -> Self {
        Self {
            method: default_dense_method_name(),
            ..Self::default()
        }
    }

    /// Uses the standard sparse generated-backend defaults.
    pub fn with_sparse_generated_backend_defaults(self) -> Self {
        self.with_sparse_generated_backend_mode(SparseGeneratedBackendMode::Defaults)
    }

    /// Uses the standard banded generated-backend defaults.
    pub fn with_banded_generated_backend_defaults(self) -> Self {
        self.with_banded_generated_backend_mode(BandedGeneratedBackendMode::Defaults)
    }

    /// Uses lambdify callbacks with generated `Banded` matrix assembly.
    pub fn with_banded_lambdify(self) -> Self {
        self.with_banded_generated_backend_mode(BandedGeneratedBackendMode::Lambdify)
    }

    /// Builds a banded release AOT backend on demand.
    pub fn with_banded_aot_build_if_missing_release(self) -> Self {
        self.with_banded_generated_backend_mode(BandedGeneratedBackendMode::BuildIfMissingRelease)
    }

    /// Requires a prebuilt sparse generated backend.
    pub fn with_sparse_aot_require_prebuilt(self) -> Self {
        self.with_sparse_generated_backend_mode(SparseGeneratedBackendMode::RequirePrebuilt)
    }

    /// Builds a sparse release AOT backend on demand.
    pub fn with_sparse_aot_build_if_missing_release(self) -> Self {
        self.with_sparse_generated_backend_mode(SparseGeneratedBackendMode::BuildIfMissingRelease)
    }

    /// Uses AtomView symbolic assembly plus on-demand `gcc`-compiled sparse C AOT.
    ///
    /// Prefer this when runtime throughput matters more than startup latency.
    pub fn with_sparse_atomview_c_gcc(self) -> Self {
        self.with_generated_backend_config(
            GeneratedBackendConfig::sparse_atomview_build_if_missing_release_gcc(),
        )
    }

    /// Uses AtomView symbolic assembly plus on-demand `tcc`-compiled sparse C AOT.
    ///
    /// Prefer this for practical repeated-solve workflows on the same large BVP.
    pub fn with_sparse_atomview_c_tcc(self) -> Self {
        self.with_generated_backend_config(
            GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc(),
        )
    }

    /// User-facing alias for the currently recommended repeated-solve compiled path.
    pub fn with_sparse_atomview_for_repeated_solves(self) -> Self {
        self.with_generated_backend_config(
            GeneratedBackendConfig::sparse_atomview_for_repeated_solves(),
        )
    }

    /// Uses AtomView symbolic assembly plus on-demand `gcc`-compiled banded C AOT.
    pub fn with_banded_atomview_c_gcc(self) -> Self {
        self.with_generated_backend_config(
            GeneratedBackendConfig::banded_atomview_build_if_missing_release_gcc(),
        )
    }

    /// Uses AtomView symbolic assembly plus on-demand `tcc`-compiled banded C AOT.
    pub fn with_banded_atomview_c_tcc(self) -> Self {
        self.with_generated_backend_config(
            GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc(),
        )
    }

    /// Uses AtomView symbolic assembly plus on-demand Zig-compiled banded AOT.
    pub fn with_banded_atomview_zig(self) -> Self {
        self.with_generated_backend_config(
            GeneratedBackendConfig::banded_atomview_build_if_missing_release_zig(),
        )
    }

    /// User-facing alias for the currently recommended repeated-solve banded path.
    pub fn with_banded_atomview_for_repeated_solves(self) -> Self {
        self.with_generated_backend_config(
            GeneratedBackendConfig::banded_atomview_for_repeated_solves(),
        )
    }
}

impl Default for DampedSolverOptions {
    fn default() -> Self {
        Self {
            scheme: default_forward_scheme_name(),
            strategy: "Damped".to_string(),
            strategy_params: Some(SolverParams::default()),
            linear_sys_method: None,
            method: default_sparse_method_name(),
            abs_tolerance: 1e-6,
            rel_tolerance: None,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            bounds: None,
            loglevel: None,
            generated_backend_config: GeneratedBackendConfig::default(),
        }
    }
}
