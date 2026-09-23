/// User-facing setup options for the frozen BVP solver.
#[derive(Clone)]
pub struct FrozenSolverOptions {
    /// Residual discretization scheme name.
    pub scheme: String,
    /// Nonlinear solver strategy name.
    pub strategy: String,
    /// Optional strategy-specific parameters.
    pub strategy_params: Option<HashMap<String, Option<Vec<f64>>>>,
    /// Optional linear-system method override.
    pub linear_sys_method: Option<String>,
    /// Matrix backend/method selector.
    pub method: String,
    /// Absolute convergence tolerance.
    pub tolerance: f64,
    /// Maximum nonlinear iterations.
    pub max_iterations: usize,
    /// Generated-backend configuration used by sparse solver paths.
    pub generated_backend_config: GeneratedBackendConfig,
}

impl FrozenSolverOptions {
    /// Creates frozen solver options from explicit values.
    pub fn new(
        strategy: String,
        strategy_params: Option<HashMap<String, Option<Vec<f64>>>>,
        linear_sys_method: Option<String>,
        method: String,
        tolerance: f64,
        max_iterations: usize,
    ) -> Self {
        Self {
            scheme: default_forward_scheme_name(),
            strategy,
            strategy_params,
            linear_sys_method,
            method,
            tolerance,
            max_iterations,
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
    /// The stored value is still the legacy string consumed by the symbolic
    /// discretization layer, but user code should prefer this method over raw
    /// `"forward"` / `"trapezoid"` strings.
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
    pub fn with_strategy_params(
        mut self,
        strategy_params: Option<HashMap<String, Option<Vec<f64>>>>,
    ) -> Self {
        self.strategy_params = strategy_params;
        self
    }

    /// Overrides the absolute convergence tolerance.
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Overrides the nonlinear iteration limit.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Overrides the linear-system method.
    pub fn with_linear_sys_method(mut self, linear_sys_method: Option<String>) -> Self {
        self.linear_sys_method = linear_sys_method;
        self
    }

    /// Attaches a high-level sparse generated-backend mode.
    pub fn with_sparse_generated_backend_mode(mut self, mode: SparseGeneratedBackendMode) -> Self {
        self.generated_backend_config = GeneratedBackendConfig::from_sparse_mode(mode);
        self
    }

    /// Attaches a high-level banded generated-backend mode.
    ///
    /// Banded modes route generated callbacks through native `Banded` matrix
    /// assembly and faithful LAPACK-style banded LU with `refine = 0`.
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

    /// Creates production-oriented sparse frozen solver options with standard defaults.
    ///
    /// This is the preferred starting point for most sparse/frozen BVP users.
    pub fn sparse_frozen() -> Self {
        Self::default().with_sparse_generated_backend_mode(SparseGeneratedBackendMode::Defaults)
    }

    /// Creates production-oriented banded frozen solver options with standard defaults.
    ///
    /// This selects generated `Banded` matrix callbacks and faithful
    /// LAPACK-style native banded LU (`refine = 0`) while preserving the frozen
    /// Newton strategy.
    pub fn banded_frozen() -> Self {
        Self::default().with_banded_generated_backend_mode(BandedGeneratedBackendMode::Defaults)
    }

    /// Creates production-oriented dense frozen solver options with standard defaults.
    pub fn dense_frozen() -> Self {
        Self {
            method: default_dense_method_name(),
            ..Self::default()
        }
    }

    /// Creates production-oriented dense solver options for the naive strategy.
    pub fn dense_naive() -> Self {
        Self {
            strategy: "Naive".to_string(),
            strategy_params: None,
            ..Self::dense_frozen()
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

impl Default for FrozenSolverOptions {
    fn default() -> Self {
        Self {
            scheme: default_forward_scheme_name(),
            strategy: "Frozen".to_string(),
            strategy_params: Some(HashMap::from([("Frozen_naive".to_string(), None)])),
            linear_sys_method: None,
            method: default_sparse_method_name(),
            tolerance: 1e-6,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            generated_backend_config: GeneratedBackendConfig::default(),
        }
    }
}
