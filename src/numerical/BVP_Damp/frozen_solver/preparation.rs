impl NRBVP {
    /// Returns the resource state paired with the prepared Frozen runtime.
    #[allow(dead_code)]
    pub(crate) fn prepared_resource_snapshot_for_diagnostics(&self) -> BvpPreparedResourceSnapshot {
        self.factor_owner.resource_snapshot()
    }

    /// Returns one coherent prepared-plan/resource diagnostic snapshot.
    #[allow(dead_code)]
    pub(crate) fn prepared_runtime_snapshot_for_diagnostics(&self) -> BvpPreparedRuntimeSnapshot {
        self.factor_owner
            .runtime_snapshot(&self.prepared_runtime_revision)
    }

    /// Computes the identity captured by a prepared Frozen runtime.
    ///
    /// This check is performed only at the prepared-solve boundary. It catches
    /// direct writes to historical public fields without adding work to the
    /// residual/Jacobian callbacks.
    fn prepared_plan_fingerprint(&self) -> PreparedPlanFingerprint {
        let mut hash = 0xcbf2_9ce4_8422_2325;
        fingerprint_debug(&mut hash, &self.eq_system);
        fingerprint_debug(&mut hash, &self.initial_guess.as_slice());
        fingerprint_debug(&mut hash, &self.values);
        fingerprint_debug(&mut hash, &self.arg);
        fingerprint_debug(&mut hash, &self.BorderConditions);
        fingerprint_debug(&mut hash, &(self.t0, self.t_end, self.n_steps));
        fingerprint_debug(&mut hash, &self.scheme);
        fingerprint_debug(&mut hash, &self.strategy);
        fingerprint_debug(&mut hash, &self.strategy_params);
        fingerprint_debug(&mut hash, &self.linear_sys_method);
        fingerprint_debug(&mut hash, &self.method);
        fingerprint_debug(&mut hash, &self.tolerance);
        fingerprint_debug(&mut hash, &self.max_iterations);
        fingerprint_debug(&mut hash, &self.param_names);
        fingerprint_debug(&mut hash, &self.param_values);
        fingerprint_debug(&mut hash, &self.x_mesh.as_slice());
        fingerprint_callback_ptr(&mut hash, Some(self.fun.as_ref()));
        fingerprint_callback_ptr(&mut hash, self.jac.as_deref());
        fingerprint_debug(&mut hash, &self.generated_backend_config);
        fingerprint_debug(&mut hash, &self.generated_backend_selected_backend);
        fingerprint_bytes(&mut hash, b"bvp-frozen-prepared-plan-v1");
        PreparedPlanFingerprint(hash)
    }

    /// Drops all state derived from the current Jacobian.
    ///
    /// Frozen reuse is valid only while the callback inputs and discretization
    /// remain unchanged. Parameter/continuation changes must therefore clear
    /// both the matrix and its owned factor, rather than merely forcing the
    /// next iteration to recalculate the Jacobian.
    fn invalidate_linear_runtime(&mut self) {
        self.prepared_runtime_revision.factor_invalidated();
        let had_owned_factor = self.factor_owner.has_factor();
        self.factor_owner.invalidate_numeric_jacobian();
        if had_owned_factor {
            self.telemetry_counters.record_factorization_invalidation();
        }
        self.jac_recalc = true;
        self.m = 0;
        self.error_old = 0.0;
    }

    #[inline]
    fn effective_runtime_method(&self) -> String {
        self.generated_backend_config.effective_method(&self.method)
    }

    pub fn new(
        eq_system: Vec<Expr>,        //
        initial_guess: DMatrix<f64>, // initial guess
        values: Vec<String>,
        arg: String,
        BorderConditions: HashMap<String, Vec<(usize, f64)>>,
        t0: f64,
        t_end: f64,
        n_steps: usize,
        strategy: String,
        strategy_params: Option<HashMap<String, Option<Vec<f64>>>>,
        linear_sys_method: Option<String>,
        method: String,
        tolerance: f64,        // tolerance
        max_iterations: usize, // max number of iterations
    ) -> NRBVP {
        //jacobian: Jacobian, initial_guess: Vec<f64>, tolerance: f64, max_iterations: usize, max_error: f64, result: Option<Vec<f64>>
        let y0 = default_placeholder_y();

        let fun0: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>> =
            Box::new(|_x, y: &DVector<f64>| y.clone());
        let boxed_fun: Box<dyn Fun> = Box::new(FunEnum::Dense(fun0));
        let x_mesh = frozen_point_mesh(t0, t_end, n_steps);
        // let fun0 =  Box::new( |x, y: &DVector<f64>| y.clone() );
        NRBVP {
            eq_system,
            initial_guess: initial_guess.clone(),
            values,
            arg,
            param_names: Vec::new(),
            param_values: None,
            BorderConditions,
            t0,
            t_end,
            n_steps,
            scheme: BvpDerivativeScheme::Forward.as_legacy_str().to_string(),
            tolerance,
            strategy,
            strategy_params,
            linear_sys_method,
            method,
            max_iterations,
            max_error: 0.0,
            result: None,
            x_mesh,
            fun: boxed_fun,
            jac: None,
            p: 0.0,
            y: y0,
            m: 0,
            factor_owner: BvpPreparedRuntime::new(),
            prepared_runtime_revision: BvpRuntimeRevision::default(),
            jac_recalc: true,
            error_old: 0.0,
            variable_string: Vec::new(), // vector of indexed variable names
            bandwidth: (0, 0),
            generated_backend_config: GeneratedBackendConfig::default(),
            generated_backend_selected_backend: None,
            generated_backend_runtime_diagnostics: HashMap::new(),
            telemetry_counters: BvpTelemetryRecorder::default(),
            generation_telemetry: None,
            atom_discretization_telemetry: None,
            legacy_lambdify_telemetry: None,
            atom_lambdify_telemetry: None,
            direct_banded_jacobian_telemetry: None,
            parameter_binding: None,
            custom_timer: CustomTimer::new(),
            no_reports: false,
        }
    }

    /// Creates a new solver instance with an explicit generated-backend configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_generated_backend_config(
        eq_system: Vec<Expr>,
        initial_guess: DMatrix<f64>,
        values: Vec<String>,
        arg: String,
        BorderConditions: HashMap<String, Vec<(usize, f64)>>,
        t0: f64,
        t_end: f64,
        n_steps: usize,
        strategy: String,
        strategy_params: Option<HashMap<String, Option<Vec<f64>>>>,
        linear_sys_method: Option<String>,
        method: String,
        tolerance: f64,
        max_iterations: usize,
        generated_backend_config: GeneratedBackendConfig,
    ) -> NRBVP {
        Self::new(
            eq_system,
            initial_guess,
            values,
            arg,
            BorderConditions,
            t0,
            t_end,
            n_steps,
            strategy,
            strategy_params,
            linear_sys_method,
            method,
            tolerance,
            max_iterations,
        )
        .with_generated_backend_config(generated_backend_config)
    }

    /// Creates a new solver instance with a high-level sparse generated-backend mode.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_sparse_generated_backend_mode(
        eq_system: Vec<Expr>,
        initial_guess: DMatrix<f64>,
        values: Vec<String>,
        arg: String,
        BorderConditions: HashMap<String, Vec<(usize, f64)>>,
        t0: f64,
        t_end: f64,
        n_steps: usize,
        strategy: String,
        strategy_params: Option<HashMap<String, Option<Vec<f64>>>>,
        linear_sys_method: Option<String>,
        method: String,
        tolerance: f64,
        max_iterations: usize,
        mode: SparseGeneratedBackendMode,
    ) -> NRBVP {
        Self::new_with_generated_backend_config(
            eq_system,
            initial_guess,
            values,
            arg,
            BorderConditions,
            t0,
            t_end,
            n_steps,
            strategy,
            strategy_params,
            linear_sys_method,
            method,
            tolerance,
            max_iterations,
            GeneratedBackendConfig::from_sparse_mode(mode),
        )
    }

    /// Creates a solver from a grouped options object instead of many positional arguments.
    ///
    /// This is the preferred public construction path for new code. The other
    /// constructor variants are retained as compatibility entrypoints.
    pub fn new_with_options(
        eq_system: Vec<Expr>,
        initial_guess: DMatrix<f64>,
        values: Vec<String>,
        arg: String,
        BorderConditions: HashMap<String, Vec<(usize, f64)>>,
        t0: f64,
        t_end: f64,
        n_steps: usize,
        options: FrozenSolverOptions,
    ) -> NRBVP {
        Self::new_with_generated_backend_config(
            eq_system,
            initial_guess,
            values,
            arg,
            BorderConditions,
            t0,
            t_end,
            n_steps,
            options.strategy,
            options.strategy_params,
            options.linear_sys_method,
            options.method,
            options.tolerance,
            options.max_iterations,
            options.generated_backend_config,
        )
        .with_scheme_name(options.scheme)
    }

    /// Returns a solver configured with the provided generated-backend settings.
    pub fn with_generated_backend_config(mut self, config: GeneratedBackendConfig) -> Self {
        self.telemetry_counters.set_logging_config(
            crate::numerical::BVP_Damp::telemetry::BvpLoggingConfig {
                mode: config.bvp_logging_mode,
                max_events: config.bvp_logging_max_events,
            },
        );
        self.telemetry_counters
            .set_telemetry_mode(config.bvp_telemetry_mode);
        self.custom_timer
            .set_telemetry_mode(config.bvp_telemetry_mode);
        self.generated_backend_config = config;
        self
    }

    /// Returns a solver configured with the selected residual discretization scheme.
    pub fn with_scheme(mut self, scheme: BvpDerivativeScheme) -> Self {
        self.scheme = scheme.as_legacy_str().to_string();
        self
    }

    /// Returns a solver configured with the legacy forward derivative discretization.
    pub fn forward_derivative(self) -> Self {
        self.with_scheme(BvpDerivativeScheme::Forward)
    }

    /// Returns a solver configured with the trapezoid derivative discretization.
    pub fn trapezoid_derivative(self) -> Self {
        self.with_scheme(BvpDerivativeScheme::Trapezoid)
    }

    /// Compatibility escape hatch for legacy/custom scheme strings.
    pub fn with_scheme_name(mut self, scheme: impl Into<String>) -> Self {
        self.scheme = scheme.into();
        self
    }

    /// Returns a solver configured with a high-level sparse generated-backend mode.
    pub fn with_sparse_generated_backend_mode(mut self, mode: SparseGeneratedBackendMode) -> Self {
        self.generated_backend_config = GeneratedBackendConfig::from_sparse_mode(mode);
        self
    }

    /// Returns a solver configured with the selected symbolic assembly backend.
    pub fn with_symbolic_assembly_backend(mut self, backend: BvpSymbolicAssemblyBackend) -> Self {
        self.generated_backend_config = self
            .generated_backend_config
            .with_symbolic_assembly_backend(backend);
        self
    }

    /// Returns a solver configured with the standard sparse generated-backend defaults.
    pub fn with_sparse_generated_backend_defaults(self) -> Self {
        self.with_sparse_generated_backend_mode(SparseGeneratedBackendMode::Defaults)
    }

    /// Returns a solver configured to require a prebuilt sparse AOT backend.
    pub fn with_sparse_aot_require_prebuilt(self) -> Self {
        self.with_sparse_generated_backend_mode(SparseGeneratedBackendMode::RequirePrebuilt)
    }

    /// Returns a solver configured to build a sparse release AOT backend on demand.
    pub fn with_sparse_aot_build_if_missing_release(self) -> Self {
        self.with_sparse_generated_backend_mode(SparseGeneratedBackendMode::BuildIfMissingRelease)
    }

    /// Returns a solver configured for AtomView symbolic assembly plus `gcc`-compiled sparse C AOT.
    pub fn with_sparse_atomview_c_gcc(self) -> Self {
        self.with_generated_backend_config(
            GeneratedBackendConfig::sparse_atomview_build_if_missing_release_gcc(),
        )
    }

    /// Returns a solver configured for AtomView symbolic assembly plus `tcc`-compiled sparse C AOT.
    pub fn with_sparse_atomview_c_tcc(self) -> Self {
        self.with_generated_backend_config(
            GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc(),
        )
    }

    /// Returns a solver configured for the recommended repeated-solve compiled path.
    pub fn with_sparse_atomview_for_repeated_solves(self) -> Self {
        self.with_generated_backend_config(
            GeneratedBackendConfig::sparse_atomview_for_repeated_solves(),
        )
    }

    /// Returns a solver with an explicit generated-backend policy override.
    pub fn with_backend_policy_override(
        mut self,
        backend_policy: Option<BackendSelectionPolicy>,
    ) -> Self {
        self.generated_backend_config.backend_policy_override = backend_policy;
        self
    }

    /// Returns a solver with an explicit generated-backend resolver snapshot.
    pub fn with_aot_resolver(mut self, resolver: Option<AotResolver>) -> Self {
        self.generated_backend_config.resolver = resolver;
        self
    }

    /// Returns a solver with an explicit solver-level AOT execution policy.
    pub fn with_aot_execution_policy(mut self, policy: AotExecutionPolicy) -> Self {
        self.generated_backend_config.aot_execution_policy = policy;
        self
    }

    /// Returns a solver with an explicit solver-level AOT build policy.
    pub fn with_aot_build_policy(mut self, policy: AotBuildPolicy) -> Self {
        self.generated_backend_config.aot_build_policy = policy;
        self
    }

    /// Returns a solver with explicit solver-level AOT chunking overrides.
    pub fn with_aot_chunking_policy(mut self, policy: AotChunkingPolicy) -> Self {
        self.generated_backend_config.aot_chunking_policy = policy;
        self
    }

    /// Returns a solver with an explicit AtomView AOT optimization profile.
    pub fn with_atom_optimization_profile(mut self, profile: AtomOptimizationProfile) -> Self {
        self.generated_backend_config = self
            .generated_backend_config
            .with_atom_optimization_profile(profile);
        self
    }

    /// Replaces the Frozen mesh through the fallible lifecycle boundary.
    ///
    /// Frozen stores `n_steps` mesh points, so at least two points are needed
    /// to construct a finite interval. The old prepared callbacks, Jacobian
    /// and factor are never reused after a mesh change; callers must invoke
    /// [`NRBVP::try_eq_generate`] before `try_solver_prepared` again.
    pub fn try_set_mesh(
        &mut self,
        t0: f64,
        t_end: f64,
        n_steps: usize,
    ) -> Result<(), BvpBackendIntegrationError> {
        if !t0.is_finite() || !t_end.is_finite() || t_end <= t0 {
            return Err(BvpBackendIntegrationError::InvalidProblem {
                field: "interval".to_string(),
                message: format!("expected finite t_end > t0, got [{t0}, {t_end}]"),
            });
        }
        if n_steps < 2 {
            return Err(BvpBackendIntegrationError::InvalidProblem {
                field: "n_steps".to_string(),
                message: format!("Frozen mesh requires at least 2 points, got {n_steps}"),
            });
        }

        let new_mesh = frozen_point_mesh(t0, t_end, n_steps);
        let mesh_changed = self.t0 != t0
            || self.t_end != t_end
            || self.n_steps != n_steps
            || self.x_mesh.as_slice() != new_mesh.as_slice();
        if !mesh_changed {
            return Ok(());
        }

        self.invalidate_linear_runtime();
        self.prepared_runtime_revision.mesh_changed();
        self.jac = None;
        self.factor_owner.clear_layout();
        self.variable_string.clear();
        self.result = None;

        let previous_guess = self.initial_guess.clone();
        self.initial_guess = DMatrix::from_fn(self.values.len(), n_steps, |row, col| {
            if row < previous_guess.nrows() && col < previous_guess.ncols() {
                previous_guess[(row, col)]
            } else {
                0.0
            }
        });
        self.t0 = t0;
        self.t_end = t_end;
        self.n_steps = n_steps;
        self.x_mesh = new_mesh;
        self.y = Vectors_type_casting(
            &DVector::zeros(self.values.len() * n_steps),
            self.generated_backend_config.effective_method(&self.method),
        );
        Ok(())
    }

    /// Basic methods to set the equation system

    ///Set system of equations with vector of symbolic expressions
    pub fn task_check(&self) {
        if self.t_end < self.t0 {
            panic!("Frozen BVP task check failed: t_end must be greater than t0");
        }

        if self.n_steps < 2 {
            panic!("Frozen BVP task check failed: n_steps must be at least 2");
        }
        if self.max_iterations < 1 {
            panic!("Frozen BVP task check failed: max_iterations must be greater than 1");
        }
        let (m, n) = self.initial_guess.shape();
        if m != self.values.len() {
            panic!(
                "Frozen BVP task check failed: initial guess row count must match number of unknowns, rows = {}, values = {}",
                m,
                self.values.len()
            );
        }
        if n != self.n_steps {
            panic!(
                "Frozen BVP task check failed: initial guess column count must equal number of steps"
            );
        }
        if self.tolerance < 0.0 {
            panic!("Frozen BVP task check failed: tolerance must be greater than 0.0");
        }
        if self.max_error < 0.0 {
            panic!("Frozen BVP task check failed: max_error must be greater than 0.0");
        }
        if self.BorderConditions.is_empty() {
            panic!("Frozen BVP task check failed: boundary conditions must be specified");
        }
        if self.BorderConditions.len() != self.values.len() {
            panic!(
                "Frozen BVP task check failed: boundary conditions must be specified for each unknown"
            );
        }
    }

    /// Fallible counterpart of [`NRBVP::task_check`] for typed callers.
    ///
    /// The historical `task_check()` and `eq_generate()` methods remain
    /// compatibility panic wrappers; generated preparation uses this method so
    /// malformed task documents and strategy parameters stay inside `Result`.
    pub fn try_task_check(&self) -> Result<(), BvpBackendIntegrationError> {
        let invalid_problem =
            |field: &str, message: String| BvpBackendIntegrationError::InvalidProblem {
                field: field.to_string(),
                message,
            };
        let invalid_option = |field: &str, value: String, message: String| {
            BvpBackendIntegrationError::InvalidSolverConfiguration {
                field: field.to_string(),
                value,
                message,
            }
        };

        if self.values.is_empty() {
            return Err(invalid_problem(
                "values",
                "at least one unknown is required".into(),
            ));
        }
        if self.initial_guess.shape() != (self.values.len(), self.n_steps) {
            return Err(invalid_problem(
                "initial_guess",
                format!(
                    "shape {:?} must be ({}, {})",
                    self.initial_guess.shape(),
                    self.values.len(),
                    self.n_steps
                ),
            ));
        }
        if !self.t0.is_finite() || !self.t_end.is_finite() || self.t_end <= self.t0 {
            return Err(invalid_problem(
                "interval",
                format!(
                    "expected finite t_end > t0, got [{}, {}]",
                    self.t0, self.t_end
                ),
            ));
        }
        if self.n_steps < 2 {
            return Err(invalid_problem(
                "n_steps",
                format!("expected n_steps >= 2, got {}", self.n_steps),
            ));
        }
        if self.max_iterations < 1 {
            return Err(invalid_option(
                "max_iterations",
                self.max_iterations.to_string(),
                "expected max_iterations >= 1".into(),
            ));
        }
        if !self.tolerance.is_finite() || self.tolerance <= 0.0 {
            return Err(invalid_option(
                "tolerance",
                self.tolerance.to_string(),
                "expected a finite positive tolerance".into(),
            ));
        }
        if !matches!(
            self.scheme.to_ascii_lowercase().as_str(),
            "forward" | "trapezoid" | "trapezoidal"
        ) {
            return Err(invalid_option(
                "scheme",
                self.scheme.clone(),
                "supported values are forward and trapezoid".into(),
            ));
        }
        let effective_method = self.generated_backend_config.effective_method(&self.method);
        if BvpMatrixBackend::from_legacy_method(&effective_method).is_none() {
            return Err(invalid_option(
                "method",
                effective_method,
                "unknown matrix backend".into(),
            ));
        }
        match self.strategy.as_str() {
            "Naive" => {
                if self.strategy_params.is_some() {
                    return Err(invalid_option(
                        "strategy_params",
                        format!("{:?}", self.strategy_params),
                        "Naive strategy does not accept strategy parameters".into(),
                    ));
                }
            }
            "Frozen" => {
                let Some(params) = self.strategy_params.as_ref() else {
                    return Err(invalid_option(
                        "strategy_params",
                        "missing".into(),
                        "Frozen strategy requires exactly one strategy parameter".into(),
                    ));
                };
                if params.len() != 1 {
                    return Err(invalid_option(
                        "strategy_params",
                        format!("{params:?}"),
                        "Frozen strategy requires exactly one strategy parameter".into(),
                    ));
                }
                let (name, value) = params.iter().next().expect("len checked above");
                let valid = match (name.as_str(), value.as_ref()) {
                    ("Frozen_naive", None) => true,
                    ("every_m", Some(values)) => {
                        values.len() == 1 && values[0].is_finite() && values[0] > 0.0
                    }
                    ("at_high_morm", Some(values)) => {
                        values.len() == 1 && values[0].is_finite() && values[0] > 0.0
                    }
                    ("at_low_speed", Some(values)) => {
                        values.len() == 1 && values[0].is_finite() && values[0] <= 1.0
                    }
                    ("complex", Some(values)) => {
                        values.len() == 3 && values.iter().all(|value| value.is_finite())
                    }
                    _ => false,
                };
                if !valid {
                    return Err(invalid_option(
                        "strategy_params",
                        format!("{params:?}"),
                        "unsupported Frozen strategy parameter shape or value".into(),
                    ));
                }
            }
            strategy => {
                return Err(invalid_option(
                    "strategy",
                    strategy.to_string(),
                    "supported values are Frozen and Naive".into(),
                ));
            }
        }
        if self.BorderConditions.is_empty()
            || self.BorderConditions.len() != self.values.len()
            || self
                .BorderConditions
                .keys()
                .any(|name| !self.values.iter().any(|value| value == name))
        {
            return Err(invalid_problem(
                "boundary_conditions",
                format!(
                    "expected one boundary-condition entry per unknown, got {} for {} unknowns",
                    self.BorderConditions.len(),
                    self.values.len()
                ),
            ));
        }
        Ok(())
    }

    pub fn try_eq_generate(&mut self) -> Result<(), BvpBackendIntegrationError> {
        self.try_task_check()?;
        let effective_method = self.generated_backend_config.effective_method(&self.method);
        let effective_policy = self
            .generated_backend_config
            .effective_backend_policy(&effective_method);
        if effective_policy == BackendSelectionPolicy::NumericOnly {
            return Err(BvpBackendIntegrationError::PipelinePanicked(
                "NumericOnly is intentionally not available for the frozen BVP solver; use the damped solver with numeric_rhs for pure numeric finite-difference discretization, or use symbolic Lambdify/AOT with Frozen"
                    .to_string(),
            ));
        }
        try_generate_and_apply_frozen_solver_state(
            self,
            "building frozen BVP generated solver state",
        )
    }

    /// Compatibility-only wrapper over [`NRBVP::try_eq_generate`].
    ///
    /// Prefer the fallible `try_*` entrypoint in new code so backend/build/runtime
    /// errors stay typed all the way to the caller.
    pub fn eq_generate(&mut self) {
        self.try_eq_generate().unwrap_or_else(|err| {
            panic!("Frozen BVP generated solver state build failed: {err:?}")
        });
    } // end of method eq_generate

    /// Installs an optional compiled AOT resolver used by backend selection.
    pub fn set_aot_resolver(&mut self, resolver: Option<AotResolver>) {
        let mut config = self.generated_backend_config.clone();
        config.resolver = resolver;
        self.set_generated_backend_config(config);
    }

    /// Installs the solver-level AOT execution policy.
    pub fn set_aot_execution_policy(&mut self, policy: AotExecutionPolicy) {
        let mut config = self.generated_backend_config.clone();
        config.aot_execution_policy = policy;
        self.set_generated_backend_config(config);
    }

    /// Returns the configured solver-level AOT execution policy.
    pub fn aot_execution_policy(&self) -> &AotExecutionPolicy {
        &self.generated_backend_config.aot_execution_policy
    }

    /// Installs the solver-level AOT build policy.
    pub fn set_aot_build_policy(&mut self, policy: AotBuildPolicy) {
        let mut config = self.generated_backend_config.clone();
        config.aot_build_policy = policy;
        self.set_generated_backend_config(config);
    }

    /// Returns the configured solver-level AOT build policy.
    pub fn aot_build_policy(&self) -> AotBuildPolicy {
        self.generated_backend_config.aot_build_policy
    }

    /// Installs explicit solver-level AOT chunking overrides.
    pub fn set_aot_chunking_policy(&mut self, policy: AotChunkingPolicy) {
        let mut config = self.generated_backend_config.clone();
        config.aot_chunking_policy = policy;
        self.set_generated_backend_config(config);
    }

    /// Returns the configured solver-level AOT chunking overrides.
    pub fn aot_chunking_policy(&self) -> AotChunkingPolicy {
        self.generated_backend_config.aot_chunking_policy
    }

    /// Installs an explicit AtomView AOT optimization profile.
    pub fn set_atom_optimization_profile(&mut self, profile: AtomOptimizationProfile) {
        let mut config = self.generated_backend_config.clone();
        config.atom_optimization_profile = profile;
        self.set_generated_backend_config(config);
    }

    /// Returns the configured AtomView AOT optimization profile.
    pub fn atom_optimization_profile(&self) -> AtomOptimizationProfile {
        self.generated_backend_config.atom_optimization_profile
    }

    /// Returns the configured compiled AOT resolver, if present.
    pub fn aot_resolver(&self) -> Option<&AotResolver> {
        self.generated_backend_config.resolver.as_ref()
    }

    /// Installs an explicit generated-backend selection policy override.
    pub fn set_backend_policy_override(&mut self, backend_policy: Option<BackendSelectionPolicy>) {
        let mut config = self.generated_backend_config.clone();
        config.backend_policy_override = backend_policy;
        self.set_generated_backend_config(config);
    }

    /// Returns the configured generated-backend selection policy override, if present.
    pub fn backend_policy_override(&self) -> Option<BackendSelectionPolicy> {
        self.generated_backend_config.backend_policy_override
    }

    /// Installs the complete generated-backend configuration in one call.
    pub fn set_generated_backend_config(&mut self, config: GeneratedBackendConfig) {
        self.invalidate_linear_runtime();
        self.telemetry_counters.set_logging_config(
            crate::numerical::BVP_Damp::telemetry::BvpLoggingConfig {
                mode: config.bvp_logging_mode,
                max_events: config.bvp_logging_max_events,
            },
        );
        self.telemetry_counters
            .set_telemetry_mode(config.bvp_telemetry_mode);
        self.custom_timer
            .set_telemetry_mode(config.bvp_telemetry_mode);
        self.generated_backend_config = config;
        self.prepared_runtime_revision.configuration_changed();
    }

    /// Changes Lambdify runtime telemetry for the next generated callback build.
    pub fn set_lambdify_telemetry_mode(&mut self, mode: BvpLambdifyTelemetryMode) {
        let config = self
            .generated_backend_config
            .clone()
            .with_lambdify_telemetry_mode(mode);
        self.set_generated_backend_config(config);
    }

    /// Changes typed solver decision logging for subsequent runtime events.
    pub fn set_bvp_logging_mode(&mut self, mode: BvpLoggingMode) {
        let config = self
            .generated_backend_config
            .clone()
            .with_bvp_logging_mode(mode);
        self.set_generated_backend_config(config);
    }

    /// Replaces the complete bounded typed decision logging policy.
    pub fn set_bvp_logging_config(&mut self, config: BvpLoggingConfig) {
        self.set_generated_backend_config(
            self.generated_backend_config
                .clone()
                .with_bvp_logging_config(config),
        );
    }

    /// Changes solver counter/timer collection for subsequent solves.
    pub fn set_bvp_telemetry_mode(&mut self, mode: BvpTelemetryMode) {
        self.generated_backend_config.bvp_telemetry_mode = mode;
        self.telemetry_counters.set_telemetry_mode(mode);
        self.custom_timer.set_telemetry_mode(mode);
    }

    /// Returns the Lambdify callback telemetry policy used for new callbacks.
    pub fn lambdify_telemetry_mode(&self) -> BvpLambdifyTelemetryMode {
        self.generated_backend_config.lambdify_telemetry_mode
    }

    /// Replaces boundary conditions through the revision-tracked API.
    ///
    /// The public field remains for source compatibility; prepared callers
    /// should use this setter so factors and generated callbacks are invalidated
    /// when the physical boundary problem changes.
    pub fn set_boundary_conditions(&mut self, conditions: HashMap<String, Vec<(usize, f64)>>) {
        if self.BorderConditions != conditions {
            self.BorderConditions = conditions;
            self.invalidate_linear_runtime();
            self.prepared_runtime_revision.problem_changed();
            self.result = None;
        }
    }

    /// Fallible typed setter for symbolic parameter names.
    pub fn try_set_params(
        &mut self,
        params: Option<&[&str]>,
    ) -> Result<(), BvpBackendIntegrationError> {
        let param_names: Vec<String> = params
            .map(|items| items.iter().map(|name| (*name).to_string()).collect())
            .unwrap_or_default();
        if let Some(values) = self.param_values.as_ref() {
            if values.len() != param_names.len() {
                return Err(BvpBackendIntegrationError::InvalidSolverConfiguration {
                    field: "param_names".to_string(),
                    value: format!("{} names", param_names.len()),
                    message: format!(
                        "param_values length {} must match param_names length {}",
                        values.len(),
                        param_names.len()
                    ),
                });
            }
        }
        if self.param_names != param_names {
            self.param_names = param_names;
            self.parameter_binding = None;
            self.invalidate_linear_runtime();
            self.prepared_runtime_revision.parameters_changed();
        }
        Ok(())
    }

    /// Compatibility wrapper for [`Self::try_set_params`].
    pub fn set_params(&mut self, params: Option<&[&str]>) {
        self.try_set_params(params)
            .unwrap_or_else(|error| panic!("invalid BVP parameter names: {error:?}"));
    }

    /// Fallible typed setter for the numeric values of symbolic parameters.
    /// Parameters are evaluator inputs, not Newton unknowns and are never
    /// included in the symbolic derivative layout.
    pub fn try_set_param_values(
        &mut self,
        values: Option<Vec<f64>>,
    ) -> Result<(), BvpBackendIntegrationError> {
        if let Some(values_ref) = values.as_ref() {
            if values_ref.len() != self.param_names.len() {
                return Err(BvpBackendIntegrationError::InvalidSolverConfiguration {
                    field: "param_values".to_string(),
                    value: format!("{} values", values_ref.len()),
                    message: format!(
                        "expected exactly {} values for declared symbolic parameters",
                        self.param_names.len()
                    ),
                });
            }
        }
        if self.param_values != values {
            self.param_values = values;
            if let Some(binding) = &self.parameter_binding {
                binding.replace(self.param_values.clone());
                let prepared_fingerprint = self.prepared_plan_fingerprint();
                self.prepared_runtime_revision
                    .refresh_prepared_fingerprint(prepared_fingerprint);
                self.factor_owner
                    .refresh_prepared_binding(prepared_fingerprint);
            } else {
                self.prepared_runtime_revision.parameters_changed();
            }
            self.invalidate_linear_runtime();
        }
        Ok(())
    }

    /// Compatibility wrapper for [`Self::try_set_param_values`].
    pub fn set_param_values(&mut self, values: Option<Vec<f64>>) {
        self.try_set_param_values(values)
            .unwrap_or_else(|error| panic!("invalid BVP parameter values: {error:?}"));
    }

    /// Sets the symbolic assembly backend used before lambdify/AOT lowering.
    pub fn set_symbolic_assembly_backend(&mut self, backend: BvpSymbolicAssemblyBackend) {
        let config = self
            .generated_backend_config
            .clone()
            .with_symbolic_assembly_backend(backend);
        self.set_generated_backend_config(config);
    }

    /// Installs a high-level sparse generated-backend mode on an existing solver.
    pub fn set_sparse_generated_backend_mode(&mut self, mode: SparseGeneratedBackendMode) {
        self.set_generated_backend_config(GeneratedBackendConfig::from_sparse_mode(mode));
    }

    /// Installs the standard sparse generated-backend defaults.
    pub fn set_sparse_generated_backend_defaults(&mut self) {
        self.set_sparse_generated_backend_mode(SparseGeneratedBackendMode::Defaults);
    }

    /// Installs AtomView symbolic assembly plus `gcc`-compiled sparse C AOT.
    pub fn set_sparse_atomview_c_gcc(&mut self) {
        self.set_generated_backend_config(
            GeneratedBackendConfig::sparse_atomview_build_if_missing_release_gcc(),
        );
    }

    /// Installs AtomView symbolic assembly plus `tcc`-compiled sparse C AOT.
    pub fn set_sparse_atomview_c_tcc(&mut self) {
        self.set_generated_backend_config(
            GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc(),
        );
    }

    /// Installs the recommended repeated-solve compiled path.
    pub fn set_sparse_atomview_for_repeated_solves(&mut self) {
        self.set_generated_backend_config(
            GeneratedBackendConfig::sparse_atomview_for_repeated_solves(),
        );
    }

    /// Returns the full generated-backend configuration.
    pub fn generated_backend_config(&self) -> &GeneratedBackendConfig {
        &self.generated_backend_config
    }

    /// Returns the normalized configuration and the backend selected so far.
    /// This is read-only and never triggers equation generation.
    pub fn resolved_plan(&self) -> crate::numerical::BVP_Damp::resolved_plan::BvpResolvedPlan {
        let mut plan =
            crate::numerical::BVP_Damp::resolved_plan::BvpResolvedPlan::from_common_for_solver(
                &self.generated_backend_config,
                &self.method,
                &self.scheme,
                crate::numerical::BVP_Damp::resolved_plan::strategy_from_name(&self.strategy),
            );
        if let Some(selected) = self.generated_backend_selected_backend {
            plan = plan.with_selected_backend(selected);
        }
        plan
    }

    /// Removes registered generated AOT artifact directories owned by this solver resolver.
    ///
    /// This is an explicit lifecycle operation for cold-build/story/debug workflows. Call it only
    /// after compiled callbacks from this solver are no longer needed; the method does not try to
    /// unregister process-local linked callbacks or unload dynamic libraries.
    pub fn cleanup_registered_aot_artifacts(&mut self) -> std::io::Result<usize> {
        cleanup_registered_aot_artifacts(&mut self.generated_backend_config)
    }

    /// Returns accumulated operation statistics and actual backend diagnostics.
    pub fn get_statistics(&self) -> FrozenBvpStatistics {
        let telemetry_counters = self.telemetry_counters.snapshot();
        let mut counters = telemetry_counters.to_legacy_map();
        if let Some(jac) = self.factor_owner.jacobian() {
            let shape = jac.shape();
            counters.insert("number of jacobian elements".to_string(), shape.0 * shape.1);
        }
        counters.insert("length of y vector".to_string(), self.y.len());
        counters.insert("number of grid points".to_string(), self.x_mesh.len());
        let mut diagnostics = self.generated_backend_runtime_diagnostics.clone();
        self.append_generated_backend_diagnostics(&mut diagnostics);
        let timings = self.custom_timer.snapshot();
        let scopes = self
            .telemetry_counters
            .scopes_snapshot(&timings, &telemetry_counters);
        let storage = self
            .factor_owner
            .old_jac
            .as_ref()
            .map(|jac| {
                crate::numerical::BVP_Damp::telemetry::BvpStorageBytes::for_solver_method(
                    &self.effective_runtime_method(),
                    jac.shape().0,
                    jac.shape().1,
                    self.factor_owner.bandwidth(),
                )
            })
            .unwrap_or_default();
        FrozenBvpStatistics {
            counters,
            timers: self.custom_timer.get_all(),
            diagnostics,
            telemetry: BvpTelemetrySnapshot {
                telemetry_mode: self.telemetry_counters.telemetry_mode(),
                counters: telemetry_counters,
                timings,
                scopes,
                storage,
                plan: Some(self.resolved_plan()),
                log_events: self.telemetry_counters.log_events_snapshot(),
                log_events_dropped: self.telemetry_counters.dropped_log_events(),
                solve_id: self.telemetry_counters.solve_id(),
                logging_config: self.telemetry_counters.logging_config(),
                atom_discretization: self.atom_discretization_telemetry,
                generation: self.generation_telemetry,
                legacy_lambdify: self
                    .legacy_lambdify_telemetry
                    .as_ref()
                    .map(BvpLambdifyTelemetry::snapshot),
                atom_lambdify: self
                    .atom_lambdify_telemetry
                    .as_ref()
                    .map(BvpLambdifyTelemetry::snapshot),
                direct_banded_jacobian: self
                    .direct_banded_jacobian_telemetry
                    .as_ref()
                    .map(crate::symbolic::bvp::telemetry::BvpDirectJacobianTelemetry::snapshot),
            },
        }
    }

    fn append_generated_backend_diagnostics(&self, diagnostics: &mut HashMap<String, String>) {
        let config = &self.generated_backend_config;
        let effective_method = self.effective_runtime_method();
        diagnostics.insert(
            "generated.backend_policy".to_string(),
            format!("{:?}", config.effective_backend_policy(&effective_method)),
        );
        diagnostics.insert("generated.effective_method".to_string(), effective_method);
        diagnostics.insert(
            "generated.selected_backend".to_string(),
            self.generated_backend_selected_backend
                .map(|backend| format!("{backend:?}"))
                .unwrap_or_else(|| "not_generated".to_string()),
        );
        diagnostics.insert(
            "generated.symbolic_assembly_backend".to_string(),
            format!("{:?}", config.symbolic_assembly_backend),
        );
        diagnostics.insert(
            "generated.matrix_backend_override".to_string(),
            config
                .matrix_backend_override
                .map(|backend| format!("{backend:?}"))
                .unwrap_or_else(|| "none".to_string()),
        );
        diagnostics.insert(
            "aot.build_policy".to_string(),
            config.aot_build_policy.as_str().to_string(),
        );
        diagnostics.insert(
            "aot.execution_policy".to_string(),
            config.aot_execution_policy.as_str().to_string(),
        );
        diagnostics.insert(
            "aot.codegen_backend".to_string(),
            format!("{:?}", config.aot_codegen_backend),
        );
        diagnostics.insert(
            "aot.c_compiler".to_string(),
            config
                .aot_c_compiler
                .clone()
                .unwrap_or_else(|| "none".to_string()),
        );
        diagnostics.insert(
            "aot.chunking.residual".to_string(),
            config
                .aot_chunking_policy
                .residual
                .map(|strategy| format!("{strategy:?}"))
                .unwrap_or_else(|| "default".to_string()),
        );
        diagnostics.insert(
            "aot.chunking.sparse_jacobian".to_string(),
            config
                .aot_chunking_policy
                .sparse_jacobian
                .map(|strategy| format!("{strategy:?}"))
                .unwrap_or_else(|| "default".to_string()),
        );
    }

    /// Returns the selected symbolic assembly backend.
    pub fn symbolic_assembly_backend(&self) -> BvpSymbolicAssemblyBackend {
        self.generated_backend_config.symbolic_assembly_backend
    }
    pub fn set_new_step(&mut self, p: f64, y: Box<dyn VectorType>, initial_guess: DMatrix<f64>) {
        self.invalidate_linear_runtime();
        self.p = p;
        self.y = y;
        self.initial_guess = initial_guess;
    }
    pub fn set_p(&mut self, p: f64) {
        if self.p != p {
            self.invalidate_linear_runtime();
        }
        self.p = p;
    }
}
