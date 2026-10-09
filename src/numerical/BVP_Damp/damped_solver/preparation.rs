impl NRBVP {
    /// Returns the prepared-plan identity for internal correctness stories.
    ///
    /// This is metadata-only and does not expose callback ownership or mutable
    /// runtime state to callers.
    #[allow(dead_code)]
    pub(crate) fn prepared_plan_for_diagnostics(&self) -> Option<BvpPreparedPlan> {
        self.prepared_runtime_revision.prepared_plan()
    }

    /// Returns the resource state paired with the prepared-plan metadata.
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

    /// Computes the identity captured by a prepared runtime.
    ///
    /// The public solver fields are retained for compatibility and can still
    /// be mutated directly.  Revision-tracked setters remain the fast path;
    /// this audit catches direct changes at the prepared-solve boundary.  No
    /// fingerprinting happens while residuals/Jacobians are evaluated.
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
        fingerprint_debug(&mut hash, &self.abs_tolerance);
        fingerprint_debug(&mut hash, &self.rel_tolerance);
        fingerprint_debug(&mut hash, &self.max_iterations);
        fingerprint_debug(&mut hash, &self.Bounds);
        fingerprint_debug(&mut hash, &self.loglevel);
        fingerprint_debug(&mut hash, &self.param_names);
        fingerprint_debug(&mut hash, &self.param_values);
        fingerprint_debug(&mut hash, &self.x_mesh.as_slice());
        fingerprint_debug(&mut hash, &self.new_grid_enabled);
        fingerprint_debug(&mut hash, &self.grid_refinemens);
        fingerprint_debug(&mut hash, &self.factor_owner.bandwidth());
        fingerprint_callback_ptr(&mut hash, Some(self.fun.as_ref()));
        fingerprint_callback_ptr(&mut hash, self.jac.as_deref());
        fingerprint_debug(&mut hash, &self.generated_backend_selected_backend);
        fingerprint_debug(&mut hash, &self.resolved_plan());
        fingerprint_bytes(&mut hash, b"bvp-damped-prepared-plan-v1");
        PreparedPlanFingerprint(hash)
    }

    #[inline]
    fn effective_runtime_method(&self) -> String {
        self.generated_backend_config.effective_method(&self.method)
    }

    fn parse_log_level(&self) -> Result<LevelFilter, BvpBackendIntegrationError> {
        match self.loglevel.as_deref() {
            Some("debug") | Some("info") => Ok(LevelFilter::Info),
            Some("warn") => Ok(LevelFilter::Warn),
            Some("error") => Ok(LevelFilter::Error),
            Some(level) => Err(BvpBackendIntegrationError::InvalidLogLevel {
                level: level.to_string(),
            }),
            None => Ok(LevelFilter::Info),
        }
    }

    /// Creates a new NRBVP solver instance
    ///
    /// # Arguments
    /// * `eq_system` - System of ODEs as symbolic expressions
    /// * `initial_guess` - Initial solution guess as matrix (variables Р“вЂ” grid points)
    /// * `values` - Names of unknown variables
    /// * `arg` - Independent variable name (usually time or space)
    /// * `BorderConditions` - Boundary conditions for each variable
    /// * `t0` - Initial value of independent variable
    /// * `t_end` - Final value of independent variable
    /// * `n_steps` - Number of grid points
    /// * `scheme` - Discretization scheme ("trapezoid", etc.)
    /// * `strategy` - Solver strategy ("Damped", "Naive", "Frozen")
    /// * `strategy_params` - Solver configuration parameters
    /// * `linear_sys_method` - Linear system solver method
    /// * `method` - Matrix backend ("Dense" or "Sparse")
    /// * `abs_tolerance` - Absolute convergence tolerance
    /// * `rel_tolerance` - Relative tolerance for each variable
    /// * `max_iterations` - Maximum Newton iterations
    /// * `Bounds` - Solution bounds for each variable
    /// * `loglevel` - Logging level ("debug", "info", "warn", "error")
    pub fn new(
        eq_system: Vec<Expr>,
        initial_guess: DMatrix<f64>,
        values: Vec<String>,
        arg: String,
        BorderConditions: HashMap<String, Vec<(usize, f64)>>,
        t0: f64,
        t_end: f64,
        n_steps: usize,
        scheme: String,
        strategy: String,
        strategy_params: Option<SolverParams>,
        linear_sys_method: Option<String>,
        method: String,
        abs_tolerance: f64,
        rel_tolerance: Option<HashMap<String, f64>>,
        max_iterations: usize,
        Bounds: Option<HashMap<String, (f64, f64)>>,
        loglevel: Option<String>,
    ) -> NRBVP {
        //jacobian: Jacobian, initial_guess: Vec<f64>, tolerance: f64, max_iterations: usize, max_error: f64, result: Option<Vec<f64>>
        let y0 = default_placeholder_y();

        let fun0: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>> =
            Box::new(|_x, y: &DVector<f64>| y.clone());
        let boxed_fun: Box<dyn Fun> = Box::new(FunEnum::Dense(fun0));
        let x_mesh = damped_interval_mesh(t0, t_end, n_steps);

        // let fun0 =  Box::new( |x, y: &DVector<f64>| y.clone() );
        let new_grid_enabled_: bool = if let Some(ref params) = strategy_params {
            params.adaptive.is_some()
        } else {
            false
        };
        NRBVP {
            eq_system,
            initial_guess: initial_guess.clone(),
            values,
            arg,
            BorderConditions,
            t0,
            t_end,
            n_steps,
            abs_tolerance,
            rel_tolerance,
            scheme,
            strategy,
            strategy_params,
            linear_sys_method,
            method,
            max_iterations,
            max_error: 0.0,
            Bounds,
            loglevel,
            param_names: Vec::new(),
            param_values: None,
            no_reports: false,
            result: None,
            full_result: None,
            x_mesh,
            fun: boxed_fun,
            BC_position_and_value: Vec::new(),
            jac: None,
            p: 0.0,
            y: y0,
            m: 0,
            factor_owner: BvpPreparedRuntime::new(),
            prepared_runtime_revision: BvpRuntimeRevision::default(),
            jac_recalc: true,
            error_old: 0.0,

            bounds_vec: Vec::new(),
            rel_tolerance_vec: Vec::new(),
            variable_string: Vec::new(),
            adaptive: false,
            new_grid_enabled: new_grid_enabled_,
            grid_refinemens: 0,
            prepared_iterate: false,
            number_of_refined_intervals: 0,
            bandwidth: (0, 0),
            generated_backend_config: GeneratedBackendConfig::default(),
            numeric_rhs: None,
            numeric_jacobian: None,
            generated_backend_selected_backend: None,
            generated_backend_runtime_diagnostics: HashMap::new(),
            telemetry_counters: BvpTelemetryRecorder::default(),
            generation_telemetry: None,
            atom_discretization_telemetry: None,
            legacy_lambdify_telemetry: None,
            atom_lambdify_telemetry: None,
            direct_banded_jacobian_telemetry: None,
            aot_telemetry: None,
            parameter_binding: None,
            nodes_added: Vec::new(),
            custom_timer: CustomTimer::new(),
        }
    }
    pub fn default() -> NRBVP {
        NRBVP::new(
            vec![],
            DMatrix::zeros(0, 0),
            vec![],
            "".to_string(),
            HashMap::new(),
            0.0,
            0.0,
            0,
            "".to_string(),
            "".to_string(),
            None,
            None,
            "".to_string(),
            0.0,
            None,
            0,
            None,
            None,
        )
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
        scheme: String,
        strategy: String,
        strategy_params: Option<SolverParams>,
        linear_sys_method: Option<String>,
        method: String,
        abs_tolerance: f64,
        rel_tolerance: Option<HashMap<String, f64>>,
        max_iterations: usize,
        Bounds: Option<HashMap<String, (f64, f64)>>,
        loglevel: Option<String>,
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
            scheme,
            strategy,
            strategy_params,
            linear_sys_method,
            method,
            abs_tolerance,
            rel_tolerance,
            max_iterations,
            Bounds,
            loglevel,
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
        scheme: String,
        strategy: String,
        strategy_params: Option<SolverParams>,
        linear_sys_method: Option<String>,
        method: String,
        abs_tolerance: f64,
        rel_tolerance: Option<HashMap<String, f64>>,
        max_iterations: usize,
        Bounds: Option<HashMap<String, (f64, f64)>>,
        loglevel: Option<String>,
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
            scheme,
            strategy,
            strategy_params,
            linear_sys_method,
            method,
            abs_tolerance,
            rel_tolerance,
            max_iterations,
            Bounds,
            loglevel,
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
        options: DampedSolverOptions,
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
            options.scheme,
            options.strategy,
            options.strategy_params,
            options.linear_sys_method,
            options.method,
            options.abs_tolerance,
            options.rel_tolerance,
            options.max_iterations,
            options.bounds,
            options.loglevel,
            options.generated_backend_config,
        )
    }

    /// Creates a pure-numeric Damped solver that uses the RHS closure as the
    /// source of truth and an explicit finite-difference Newton Jacobian.
    ///
    /// Uses [`DampedSolverOptions::sparse_damped`] plus the provided bounds and
    /// per-variable tolerances. Use [`NRBVP::new_numeric_fd_with_options`] when
    /// you need Banded, custom nonlinear parameters, or logging settings.
    #[allow(clippy::too_many_arguments)]
    pub fn new_numeric_fd<
        F: Fn(f64, &DVector<f64>, Option<&[f64]>) -> DVector<f64> + Send + Sync + 'static,
    >(
        initial_guess: DMatrix<f64>,
        values: Vec<String>,
        arg: String,
        border_conditions: HashMap<String, Vec<(usize, f64)>>,
        t0: f64,
        t_end: f64,
        n_steps: usize,
        bounds: HashMap<String, (f64, f64)>,
        rel_tolerance: HashMap<String, f64>,
        rhs: F,
    ) -> NRBVP {
        Self::new_numeric_fd_with_options(
            initial_guess,
            values,
            arg,
            border_conditions,
            t0,
            t_end,
            n_steps,
            DampedSolverOptions::sparse_damped()
                .with_bounds(bounds)
                .with_rel_tolerance(rel_tolerance),
            rhs,
        )
    }

    /// Creates a pure-numeric Damped solver that uses the RHS closure as the
    /// source of truth and an explicit finite-difference Newton Jacobian.
    ///
    /// This wrapper exists so users do not have to pass `Vec::new()` as a
    /// symbolic equation placeholder when they intentionally choose the numeric
    /// route.
    #[allow(clippy::too_many_arguments)]
    pub fn new_numeric_fd_with_options<
        F: Fn(f64, &DVector<f64>, Option<&[f64]>) -> DVector<f64> + Send + Sync + 'static,
    >(
        initial_guess: DMatrix<f64>,
        values: Vec<String>,
        arg: String,
        border_conditions: HashMap<String, Vec<(usize, f64)>>,
        t0: f64,
        t_end: f64,
        n_steps: usize,
        options: DampedSolverOptions,
        rhs: F,
    ) -> NRBVP {
        Self::new_numeric_with_optional_jacobian(
            initial_guess,
            values,
            arg,
            border_conditions,
            t0,
            t_end,
            n_steps,
            options,
            Arc::new(rhs),
            None,
        )
    }

    /// Creates a pure-numeric Damped solver with a user-provided continuous
    /// RHS Jacobian `df/dy`.
    ///
    /// The closure returns the small per-node Jacobian of the continuous RHS.
    /// The large discretized Newton Jacobian is assembled by
    /// `numeric_discretization`, preserving the selected matrix backend.
    ///
    /// Uses [`DampedSolverOptions::sparse_damped`] plus the provided bounds and
    /// per-variable tolerances. Use [`NRBVP::new_numeric_with_jacobian_options`]
    /// when you need Banded, custom nonlinear parameters, or logging settings.
    #[allow(clippy::too_many_arguments)]
    pub fn new_numeric_with_jacobian<
        F: Fn(f64, &DVector<f64>, Option<&[f64]>) -> DVector<f64> + Send + Sync + 'static,
        J: Fn(f64, &DVector<f64>, Option<&[f64]>) -> DMatrix<f64> + Send + Sync + 'static,
    >(
        initial_guess: DMatrix<f64>,
        values: Vec<String>,
        arg: String,
        border_conditions: HashMap<String, Vec<(usize, f64)>>,
        t0: f64,
        t_end: f64,
        n_steps: usize,
        bounds: HashMap<String, (f64, f64)>,
        rel_tolerance: HashMap<String, f64>,
        rhs: F,
        jacobian: J,
    ) -> NRBVP {
        Self::new_numeric_with_jacobian_options(
            initial_guess,
            values,
            arg,
            border_conditions,
            t0,
            t_end,
            n_steps,
            DampedSolverOptions::sparse_damped()
                .with_bounds(bounds)
                .with_rel_tolerance(rel_tolerance),
            rhs,
            jacobian,
        )
    }

    /// Creates a pure-numeric Damped solver with a user-provided continuous
    /// RHS Jacobian `df/dy`.
    ///
    /// The closure returns the small per-node Jacobian of the continuous RHS.
    /// The large discretized Newton Jacobian is assembled by
    /// `numeric_discretization`, preserving the selected matrix backend.
    #[allow(clippy::too_many_arguments)]
    pub fn new_numeric_with_jacobian_options<
        F: Fn(f64, &DVector<f64>, Option<&[f64]>) -> DVector<f64> + Send + Sync + 'static,
        J: Fn(f64, &DVector<f64>, Option<&[f64]>) -> DMatrix<f64> + Send + Sync + 'static,
    >(
        initial_guess: DMatrix<f64>,
        values: Vec<String>,
        arg: String,
        border_conditions: HashMap<String, Vec<(usize, f64)>>,
        t0: f64,
        t_end: f64,
        n_steps: usize,
        options: DampedSolverOptions,
        rhs: F,
        jacobian: J,
    ) -> NRBVP {
        Self::new_numeric_with_optional_jacobian(
            initial_guess,
            values,
            arg,
            border_conditions,
            t0,
            t_end,
            n_steps,
            options,
            Arc::new(rhs),
            Some(Arc::new(jacobian)),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_numeric_with_optional_jacobian(
        initial_guess: DMatrix<f64>,
        values: Vec<String>,
        arg: String,
        border_conditions: HashMap<String, Vec<(usize, f64)>>,
        t0: f64,
        t_end: f64,
        n_steps: usize,
        mut options: DampedSolverOptions,
        rhs: NumericBvpRhs,
        jacobian: Option<NumericBvpJacobian>,
    ) -> NRBVP {
        options.generated_backend_config = options
            .generated_backend_config
            .with_backend_policy_override(Some(BackendSelectionPolicy::NumericOnly));
        Self::new_with_options(
            Vec::new(),
            initial_guess,
            values,
            arg,
            border_conditions,
            t0,
            t_end,
            n_steps,
            options,
        )
        .with_numeric_rhs_arc(rhs)
        .with_numeric_jacobian_arc(jacobian)
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

    /// Replaces boundary conditions through the revision-tracked API.
    ///
    /// Direct mutation of the historical public `BorderConditions` field is
    /// retained for compatibility, but prepared callers should use this
    /// method so stale callbacks and factors cannot survive a physical change.
    pub fn set_boundary_conditions(&mut self, conditions: HashMap<String, Vec<(usize, f64)>>) {
        if self.BorderConditions != conditions {
            self.BorderConditions = conditions;
            self.invalidate_linear_runtime();
            self.prepared_runtime_revision.problem_changed();
            self.BC_position_and_value.clear();
            self.result = None;
            self.full_result = None;
        }
    }

    /// Fallible typed setter for symbolic parameter names.
    pub fn try_set_params(
        &mut self,
        params: Option<&[&str]>,
    ) -> Result<(), BvpBackendIntegrationError> {
        let param_names: Vec<String> = params
            .map(|names| names.iter().map(|name| (*name).to_string()).collect())
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

    /// Fallible typed setter for the current numeric parameter binding.
    ///
    /// Parameters are evaluator inputs, not Newton unknowns, and are not
    /// differentiated symbolically. Prepared Lambdify callbacks replace the
    /// numeric binding in place; symbolic preparation and closure compilation
    /// are retained. Numeric factors still invalidate because Jacobian values
    /// changed.
    pub fn try_set_param_values(
        &mut self,
        values: Option<Vec<f64>>,
    ) -> Result<(), BvpBackendIntegrationError> {
        if let Some(ref values) = values {
            if values.len() != self.param_names.len() {
                return Err(BvpBackendIntegrationError::InvalidSolverConfiguration {
                    field: "param_values".to_string(),
                    value: format!("{} values", values.len()),
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

    /// Replaces the callbacks of an already prepared runtime through an
    /// explicit lifecycle operation.
    ///
    /// Direct writes to the public compatibility fields `fun` and `jac` are
    /// intentionally detected as stale by [`Self::try_solver_prepared`].
    /// Tests and integrations that deliberately install callbacks with the
    /// same prepared shape must use this method instead. The callback
    /// generation is advanced, the numeric Jacobian/factor is dropped, and a
    /// fresh fingerprint is published for the explicit replacement.
    pub fn try_replace_prepared_callbacks(
        &mut self,
        fun: Box<dyn Fun>,
        jac: Option<Box<dyn Jac>>,
    ) -> Result<(), BvpBackendIntegrationError> {
        if self.prepared_runtime_revision.prepared_plan().is_none() {
            return Err(BvpBackendIntegrationError::PreparedRuntimeInvalidated {
                reason: "runtime callbacks can only be replaced after try_eq_generate".to_string(),
            });
        }

        self.fun = fun;
        self.jac = jac;
        self.prepared_runtime_revision.callbacks_changed();
        self.factor_owner.clear_numeric_jacobian();

        let fingerprint = self.prepared_plan_fingerprint();
        self.prepared_runtime_revision
            .mark_prepared_with_fingerprint(fingerprint);
        self.factor_owner.publish_prepared_binding(fingerprint);
        Ok(())
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

    /// Fallible Damped mesh mutation boundary.
    ///
    /// Damped stores `n_steps` intervals and therefore requires at least two
    /// intervals for the existing solver contract. A changed mesh clears all
    /// callback-derived layout and linear resources before the next prepare.
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
        if n_steps <= 1 {
            return Err(BvpBackendIntegrationError::InvalidProblem {
                field: "n_steps".to_string(),
                message: format!("Damped mesh requires at least 2 intervals, got {n_steps}"),
            });
        }

        let new_mesh = damped_interval_mesh(t0, t_end, n_steps);
        let mesh_changed = self.t0 != t0
            || self.t_end != t_end
            || self.n_steps != n_steps
            || self.x_mesh.as_slice() != new_mesh.as_slice();
        if mesh_changed {
            self.invalidate_linear_runtime();
            self.prepared_runtime_revision.mesh_changed();
            self.jac = None;
            self.bounds_vec.clear();
            self.rel_tolerance_vec.clear();
            self.factor_owner.clear_layout();
            self.variable_string.clear();
            self.result = None;
            self.full_result = None;
            self.grid_refinemens = 0;
            self.number_of_refined_intervals = 0;
            self.nodes_added.clear();

            let previous_guess = self.initial_guess.clone();
            self.initial_guess = DMatrix::from_fn(self.values.len(), n_steps, |row, col| {
                if row < previous_guess.nrows() && col < previous_guess.ncols() {
                    previous_guess[(row, col)]
                } else {
                    0.0
                }
            });
            self.n_steps = n_steps;
            self.y = Vectors_type_casting(
                &DVector::zeros(self.values.len() * n_steps),
                self.effective_runtime_method(),
            );
        }
        self.x_mesh = new_mesh;
        self.t0 = t0;
        self.t_end = t_end;
        Ok(())
    }

    /// Compatibility wrapper for the historical infallible mesh setter.
    pub fn set_mesh(&mut self, t0: f64, t_end: f64, n_steps: usize) {
        self.try_set_mesh(t0, t_end, n_steps)
            .unwrap_or_else(|error| panic!("Damped BVP mesh update failed: {error}"));
    }

    /// Installs a pure numeric RHS callback used by the `NumericOnly` backend route.
    ///
    /// This route is available for the damped Newton BVP solver only. It bypasses
    /// symbolic BVP assembly: the continuous RHS closure is discretized by
    /// `numeric_discretization`. If no numeric Jacobian closure is installed,
    /// the Newton Jacobian is approximated by finite differences. Use
    /// [`NRBVP::set_numeric_jacobian`] or
    /// [`NRBVP::new_numeric_with_jacobian_options`] for an explicit analytical
    /// continuous RHS Jacobian.
    ///
    /// The callback must return a derivative vector with length `values.len()`.
    pub fn set_numeric_rhs(&mut self, rhs: Option<NumericBvpRhs>) {
        self.invalidate_linear_runtime();
        self.prepared_runtime_revision.callbacks_changed();
        self.numeric_rhs = rhs;
    }

    /// Installs a continuous RHS Jacobian callback for the `NumericOnly` route.
    ///
    /// The callback returns `df/dy` at one mesh node. The solver assembles the
    /// global discretized Newton Jacobian from this local Jacobian, boundary
    /// conditions, and the selected finite-difference scheme.
    pub fn set_numeric_jacobian(&mut self, jacobian: Option<NumericBvpJacobian>) {
        self.invalidate_linear_runtime();
        self.prepared_runtime_revision.callbacks_changed();
        self.numeric_jacobian = jacobian;
    }

    /// Builder-style helper to install an already boxed/arc-ed numeric RHS.
    pub fn with_numeric_rhs_arc(mut self, rhs: NumericBvpRhs) -> Self {
        self.numeric_rhs = Some(rhs);
        self
    }

    /// Builder-style helper to install an already boxed/arc-ed numeric Jacobian.
    pub fn with_numeric_jacobian_arc(mut self, jacobian: Option<NumericBvpJacobian>) -> Self {
        self.numeric_jacobian = jacobian;
        self
    }

    /// Builder-style helper to install a pure numeric RHS callback.
    ///
    /// See [`NRBVP::set_numeric_rhs`] for the route contract.
    pub fn with_numeric_rhs<
        F: Fn(f64, &DVector<f64>, Option<&[f64]>) -> DVector<f64> + Send + Sync + 'static,
    >(
        mut self,
        rhs: F,
    ) -> Self {
        self.numeric_rhs = Some(Arc::new(rhs));
        self
    }

    /// Builder-style helper to install a continuous RHS Jacobian callback.
    pub fn with_numeric_jacobian<
        J: Fn(f64, &DVector<f64>, Option<&[f64]>) -> DMatrix<f64> + Send + Sync + 'static,
    >(
        mut self,
        jacobian: J,
    ) -> Self {
        self.numeric_jacobian = Some(Arc::new(jacobian));
        self
    }

    /// Returns true when a pure numeric RHS callback is installed.
    pub fn has_numeric_rhs(&self) -> bool {
        self.numeric_rhs.is_some()
    }

    /// Returns true when a pure numeric continuous RHS Jacobian callback is installed.
    pub fn has_numeric_jacobian(&self) -> bool {
        self.numeric_jacobian.is_some()
    }
    /// Basic methods to set the equation system

    /// Validates solver configuration and input parameters
    ///
    /// Performs comprehensive checks on problem dimensions, boundary conditions,
    /// tolerances, and bounds to ensure the problem is well-posed.
    ///
    /// # Panics
    /// Panics if any validation check fails with descriptive error message
    pub fn task_check(&self) {
        assert_eq!(
            self.initial_guess.len(), //grid length =  number of unknowns
            self.n_steps * self.values.len(),
            "lenght of initial guess {} should be equal to n_steps*values, {}, {} ",
            self.initial_guess.len(),
            self.x_mesh.len(),
            self.values.len()
        );
        assert!(self.t_end > self.t0, "t_end must be greater than t0");
        assert!(self.n_steps > 1, "n_steps must be greater than 1");
        assert!(
            self.max_iterations > 1,
            "max_iterations must be greater than 1"
        );
        let (m, n) = self.initial_guess.shape();
        if m != self.values.len() {
            panic!(
                "m must be equal to the length of the argument, m= {}, arg = {}",
                m,
                self.arg.len()
            );
        }
        assert_eq!(n, self.n_steps, "n must be equal to the number of steps");
        assert!(
            self.abs_tolerance > 0.0,
            "tolerance must be greater than 0.0"
        );

        assert!(
            !self.BorderConditions.is_empty(),
            "BorderConditions must be specified"
        );
        let total_conditions: usize = self.BorderConditions.values().map(|v| v.len()).sum();
        assert_eq!(
            total_conditions,
            self.values.len(),
            "Total number of boundary conditions ({}) must equal number of variables ({})",
            total_conditions,
            self.values.len()
        );
        assert!(
            !self.Bounds.is_none(),
            "Bounds must be specified for each value"
        );
        let bound_keys_vec = self
            .Bounds
            .clone()
            .unwrap()
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(
            bound_keys_vec.len(),
            self.values.len(),
            "Bounds must be specified for each value"
        );
        // check if initial guess values are inside bunds defined for certain values
        if self.result.is_none() {
            // check of does the guess fits into bounds must be enable only at the beginning (at result == None)
            // we will find ourselves in this place again when the command to recalculate the lattice is given, and the result of the previous
            //iteration may go beyond the boundaries and we must make sure that this fact does not stop the program, therefore
            if_initial_guess_inside_bounds(&self.initial_guess, &self.Bounds, self.values.clone());
        }
        assert!(
            !self.rel_tolerance.is_none(),
            "rel_tolerance must be specified for each value"
        );

        // Validation is implicit in the struct design - if adaptive is Some, grid_method is always present
    }

    /// Fallible counterpart of [`NRBVP::task_check`] for typed callers.
    ///
    /// Compatibility callers may keep using `task_check()`, but the production
    /// `try_*` path must reject malformed user input without panicking or
    /// terminating the process.
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
        if self.n_steps <= 1 {
            return Err(invalid_problem(
                "n_steps",
                format!("expected n_steps > 1, got {}", self.n_steps),
            ));
        }
        if self.max_iterations <= 1 {
            return Err(invalid_option(
                "max_iterations",
                self.max_iterations.to_string(),
                "expected max_iterations > 1".into(),
            ));
        }
        if !self.abs_tolerance.is_finite() || self.abs_tolerance <= 0.0 {
            return Err(invalid_option(
                "abs_tolerance",
                self.abs_tolerance.to_string(),
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
        if !self.strategy.eq_ignore_ascii_case("damped") {
            return Err(invalid_option(
                "strategy",
                self.strategy.clone(),
                "damped solver requires strategy=Damped".into(),
            ));
        }
        if self.BorderConditions.is_empty() {
            return Err(invalid_problem(
                "boundary_conditions",
                "at least one boundary condition is required".into(),
            ));
        }
        let total_conditions: usize = self.BorderConditions.values().map(Vec::len).sum();
        if total_conditions != self.values.len()
            || self
                .BorderConditions
                .keys()
                .any(|name| !self.values.iter().any(|value| value == name))
        {
            return Err(invalid_problem(
                "boundary_conditions",
                format!(
                    "expected one total condition per unknown; got {} for {} unknowns",
                    total_conditions,
                    self.values.len()
                ),
            ));
        }
        let bounds = self.Bounds.as_ref().ok_or_else(|| {
            invalid_problem("bounds", "bounds are required for the damped solver".into())
        })?;
        if bounds.len() != self.values.len() {
            return Err(invalid_problem(
                "bounds",
                format!(
                    "expected bounds for {} unknowns, got {}",
                    self.values.len(),
                    bounds.len()
                ),
            ));
        }
        for (row, name) in self.values.iter().enumerate() {
            let Some(&(lower, upper)) = bounds.get(name) else {
                return Err(invalid_problem(
                    "bounds",
                    format!("missing bounds for unknown {name}"),
                ));
            };
            if !lower.is_finite() || !upper.is_finite() || lower > upper {
                return Err(invalid_problem(
                    "bounds",
                    format!("invalid interval for {name}: [{lower}, {upper}]"),
                ));
            }
            if self.result.is_none() {
                for value in self.initial_guess.row(row).iter() {
                    if !value.is_finite() || *value < lower || *value > upper {
                        return Err(invalid_problem(
                            "initial_guess",
                            format!("value {value} for {name} is outside [{lower}, {upper}]"),
                        ));
                    }
                }
            }
        }
        let rel_tolerance = self.rel_tolerance.as_ref().ok_or_else(|| {
            invalid_option(
                "rel_tolerance",
                "missing".into(),
                "relative tolerances are required for the damped solver".into(),
            )
        })?;
        if rel_tolerance.len() != self.values.len()
            || self.values.iter().any(|name| {
                rel_tolerance
                    .get(name)
                    .map(|value| !value.is_finite() || *value <= 0.0)
                    .unwrap_or(true)
            })
        {
            return Err(invalid_option(
                "rel_tolerance",
                format!("{rel_tolerance:?}"),
                "expected one finite positive tolerance per unknown".into(),
            ));
        }
        Ok(())
    }
    /// Generates discretized system and Jacobian from symbolic expressions
    ///
    /// This is the core symbolic-to-numerical transformation that:
    /// 1. Discretizes the ODE system using finite differences
    /// 2. Generates analytical Jacobian matrix
    /// 3. Creates function closures for residual and Jacobian evaluation
    /// 4. Sets up boundary condition handling
    ///
    /// # Arguments
    /// * `mesh_` - Optional custom mesh points (if None, uniform mesh is used)
    /// * `bandwidth` - Optional Jacobian bandwidth for sparse matrices
    pub fn try_eq_generate(
        &mut self,
        mesh_: Option<Vec<f64>>,
        bandwidth: Option<(usize, usize)>,
    ) -> Result<(), BvpBackendIntegrationError> {
        // Memory inspection is advisory diagnostics, not part of the typed
        // problem-validation contract. In particular, sysinfo/platform
        // backends must never turn malformed input or restricted CI hosts into
        // a process-level failure.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            task_check_mem(self.n_steps, self.values.len(), &self.method);
        }));
        self.try_task_check()?;

        let effective_method = self.generated_backend_config.effective_method(&self.method);
        let effective_policy = self
            .generated_backend_config
            .effective_backend_policy(&effective_method);
        if effective_policy == BackendSelectionPolicy::NumericOnly {
            if let Some(rhs) = self.numeric_rhs.clone() {
                if let Some(mesh) = mesh_.clone() {
                    self.x_mesh = DVector::from_vec(mesh);
                }
                let mesh_vec: Vec<f64> = self.x_mesh.iter().copied().collect();
                let bounds = self.Bounds.as_ref().ok_or_else(|| {
                    BvpBackendIntegrationError::PipelinePanicked(
                        "pure numeric BVP route requires Bounds".to_string(),
                    )
                })?;
                let rel_tolerance = self.rel_tolerance.as_ref().ok_or_else(|| {
                    BvpBackendIntegrationError::PipelinePanicked(
                        "pure numeric BVP route requires rel_tolerance".to_string(),
                    )
                })?;
                let state = build_numeric_generated_solver_state(
                    rhs,
                    self.numeric_jacobian.clone(),
                    &effective_method,
                    self.scheme.as_str(),
                    &self.values,
                    &self.BorderConditions,
                    bounds,
                    rel_tolerance,
                    self.n_steps,
                    &mesh_vec,
                    bandwidth.or_else(|| {
                        if self.factor_owner.bandwidth() == (0, 0) {
                            None
                        } else {
                            Some(self.factor_owner.bandwidth())
                        }
                    }),
                    self.param_values.clone(),
                )
                .map_err(BvpBackendIntegrationError::PipelinePanicked)?;
                self.apply_generated_solver_state(state);
                return Ok(());
            }
            return Err(BvpBackendIntegrationError::PipelinePanicked(
                "NumericOnly BVP route requires a numeric_rhs closure; symbolic lambdify/AOT routes must use LambdifyOnly, AotOnly, or PreferAotThenLambdify"
                    .to_string(),
            ));
        }

        try_generate_and_apply_damped_solver_state(
            self,
            mesh_,
            bandwidth,
            "building damped BVP generated solver state",
        )
    }

    /// Compatibility-only wrapper over [`NRBVP::try_eq_generate`].
    ///
    /// Prefer the fallible `try_*` entrypoint in new code so backend/build/runtime
    /// errors stay typed all the way to the caller.
    pub fn eq_generate(&mut self, mesh_: Option<Vec<f64>>, bandwidth: Option<(usize, usize)>) {
        self.try_eq_generate(mesh_, bandwidth)
            .unwrap_or_else(|err| panic!("BVP generated solver state build failed: {err:?}"));
    } // end of method eq_generate
    /// Updates solver state for new iteration step
    ///
    /// Used internally during grid refinement to set new problem parameters
    pub fn set_new_step(&mut self, p: f64, y: Box<dyn VectorType>, initial_guess: DMatrix<f64>) {
        self.invalidate_linear_runtime();
        self.p = p;
        self.y = y;
        self.initial_guess = initial_guess;
    }

    /// Replace the Newton initial guess without rebuilding symbolic callbacks.
    ///
    /// This is the state-side half of a prepared continuation restart. The
    /// callback graph and generated backend remain valid; only numeric factor
    /// state and the current iterate are discarded.
    pub fn try_set_initial_guess(
        &mut self,
        initial_guess: DMatrix<f64>,
    ) -> Result<(), BvpBackendIntegrationError> {
        let expected = (self.values.len(), self.n_steps);
        if initial_guess.shape() != expected {
            return Err(BvpBackendIntegrationError::InvalidProblem {
                field: "initial_guess".to_string(),
                message: format!(
                    "expected shape {}x{}, got {}x{}",
                    expected.0,
                    expected.1,
                    initial_guess.nrows(),
                    initial_guess.ncols()
                ),
            });
        }
        if let Some(index) = initial_guess.iter().position(|value| !value.is_finite()) {
            return Err(BvpBackendIntegrationError::NonFiniteCallbackValue {
                stage: "initial_guess".to_string(),
                index,
            });
        }
        let flattened = DVector::from_vec(initial_guess.iter().copied().collect());
        self.initial_guess = initial_guess;
        self.result = None;
        self.full_result = None;
        self.grid_refinemens = 0;
        self.prepared_iterate = false;
        self.y = Vectors_type_casting(&flattened, self.effective_runtime_method());
        self.invalidate_linear_runtime();
        Ok(())
    }

    /// Install a warm-start Newton iterate without changing the prepared plan.
    ///
    /// Unlike [`Self::try_set_initial_guess`], this method does not mutate the
    /// structural initial-guess fingerprint. It is therefore suitable for a
    /// prepared parameter continuation where callbacks remain valid and only
    /// the current numeric iterate changes.
    pub fn try_set_prepared_iterate(
        &mut self,
        iterate: DMatrix<f64>,
    ) -> Result<(), BvpBackendIntegrationError> {
        let expected = (self.values.len(), self.n_steps);
        if iterate.shape() != expected {
            return Err(BvpBackendIntegrationError::InvalidProblem {
                field: "prepared_iterate".to_string(),
                message: format!(
                    "expected shape {}x{}, got {}x{}",
                    expected.0,
                    expected.1,
                    iterate.nrows(),
                    iterate.ncols()
                ),
            });
        }
        if let Some(index) = iterate.iter().position(|value| !value.is_finite()) {
            return Err(BvpBackendIntegrationError::NonFiniteCallbackValue {
                stage: "prepared_iterate".to_string(),
                index,
            });
        }
        let flattened = DVector::from_vec(iterate.iter().copied().collect());
        self.y = Vectors_type_casting(&flattened, self.effective_runtime_method());
        self.prepared_iterate = true;
        self.invalidate_linear_runtime();
        Ok(())
    }

    /// Sets the parameter value (typically time or spatial coordinate)
    pub fn set_p(&mut self, p: f64) {
        if self.p != p {
            self.invalidate_linear_runtime();
        }
        self.p = p;
    }

    /// Drops numeric Jacobian/factor state after an input or continuation change.
    ///
    /// A symbolic callback may remain reusable, but its numeric values and any
    /// owned direct factor are only valid for the previous parameter/state
    /// snapshot. This keeps the Damped cache contract aligned with Frozen.
    fn invalidate_linear_runtime(&mut self) {
        self.prepared_runtime_revision.factor_invalidated();
        let had_owned_factor = self.factor_owner.has_factor();
        let old_jac_was_factorized = self
            .factor_owner
            .old_jac
            .as_ref()
            .map(|jacobian| jacobian.factorization_ready())
            .unwrap_or(false);
        self.factor_owner.invalidate_numeric_jacobian();
        if had_owned_factor || old_jac_was_factorized {
            self.telemetry_counters.record_factorization_invalidation();
        }
        self.jac_recalc = true;
        self.m = 0;
        self.error_old = 0.0;
    }

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

    /// Returns accumulated nonlinear/callback/linear-solve statistics for the current solver.
    pub fn get_statistics(&self) -> DampedBvpStatistics {
        let telemetry_counters = self.telemetry_counters.snapshot();
        let mut counters = telemetry_counters.to_legacy_map();
        if let Some(jac) = self.factor_owner.jacobian() {
            let jac_shape = jac.shape();
            let matrix_weight = checkmem(&**jac);
            counters.insert("jacobian memory, MB".to_string(), matrix_weight as usize);
            counters.insert(
                "number of jacobian elements".to_string(),
                jac_shape.0 * jac_shape.1,
            );
        }
        counters.insert("length of y vector".to_string(), self.y.len() as usize);
        counters.insert(
            "number of grid points".to_string(),
            self.x_mesh.len() as usize,
        );
        let timers = self.custom_timer.get_all();
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
        let mut diagnostics = self.generated_backend_runtime_diagnostics.clone();
        self.append_generated_backend_diagnostics(&mut diagnostics);
        if let Some(telemetry) = &self.aot_telemetry {
            telemetry
                .snapshot()
                .append_compatibility_diagnostics(&mut diagnostics);
        }
        DampedBvpStatistics {
            counters,
            timers,
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
        let backend_policy = config.effective_backend_policy(&effective_method);
        diagnostics.insert("generated.effective_method".to_string(), effective_method);
        diagnostics.insert(
            "generated.backend_policy".to_string(),
            format!("{backend_policy:?}"),
        );
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

        match &config.resolver {
            Some(resolver) => {
                let problem_keys = resolver.registry().problem_keys();
                diagnostics.insert("aot.resolver.present".to_string(), "true".to_string());
                diagnostics.insert(
                    "aot.resolver.entries".to_string(),
                    problem_keys.len().to_string(),
                );
                diagnostics.insert(
                    "aot.resolver.problem_keys".to_string(),
                    if problem_keys.is_empty() {
                        "none".to_string()
                    } else {
                        problem_keys.join(",")
                    },
                );
            }
            None => {
                diagnostics.insert("aot.resolver.present".to_string(), "false".to_string());
                diagnostics.insert("aot.resolver.entries".to_string(), "0".to_string());
                diagnostics.insert("aot.resolver.problem_keys".to_string(), "none".to_string());
            }
        }
    }

    /// Returns the selected symbolic assembly backend.
    pub fn symbolic_assembly_backend(&self) -> BvpSymbolicAssemblyBackend {
        self.generated_backend_config.symbolic_assembly_backend
    }
    /////////////////////
    /// Computes Newton step using cached inverse Jacobian
    ///
    /// More efficient than `step()` when Jacobian doesn't need recalculation
    pub fn step_with_inv_Jac(&self, p: f64, y: &dyn VectorType) -> Box<dyn VectorType> {
        let fun = &self.fun;
        self.telemetry_counters.record_residual_call();
        let F_k = fun.call(p, y);
        let inv_J_k = self
            .factor_owner
            .old_jac
            .as_ref()
            .expect("Damped BVP inverse-Jacobian step requires a cached Jacobian factor")
            .clone_box();
        let undamped_step_k: Box<dyn VectorType> = inv_J_k.mul(&*F_k);
        undamped_step_k
    }

    /// Recalculates and inverts Jacobian matrix if needed
    ///
    /// Updates the cached Jacobian and resets iteration counter
    pub fn recalc_and_inverse_Jac(&mut self) {
        if self.jac_recalc {
            let p = self.p;
            let y = &*self.y;
            log::info!("\n \n JACOBIAN (RE)CALCULATED! \n \n");
            let inv_J_k = if let Some(jac_function) = self.jac.as_mut() {
                let jac_matrix = jac_function.call(p, y);
                jac_function.inv(&*jac_matrix, self.abs_tolerance, self.max_iterations)
            } else {
                let y_runtime =
                    Vectors_type_casting(&y.to_DVectorType(), self.effective_runtime_method());
                let jac_matrix = finite_difference_jacobian(&*self.fun, p, &*y_runtime, 1e-8);
                let inv = jac_matrix
                    .to_DMatrixType()
                    .try_inverse()
                    .unwrap_or_else(|| {
                        panic!("Damped BVP FD Jacobian inversion failed: singular matrix")
                    });
                Box::new(inv)
            };
            let had_owned_factor = self.factor_owner.borrow().is_some();
            if had_owned_factor
                || self
                    .factor_owner
                    .old_jac
                    .as_ref()
                    .map(|jacobian| jacobian.factorization_ready())
                    .unwrap_or(false)
            {
                self.telemetry_counters.record_factorization_invalidation();
            }
            self.factor_owner.replace_numeric_jacobian(inv_J_k);
            self.m = 0;
            self.telemetry_counters.record_jacobian_recalculation();
        }
    }
    ////////////////////////////////////////////////////////////////

    /// Compatibility wrapper around [`Self::try_recalc_jacobian`].
    #[allow(dead_code)]
    fn recalc_jacobian(&mut self) {
        self.try_recalc_jacobian()
            .unwrap_or_else(|error| panic!("Damped BVP Jacobian recalculation failed: {error:?}"));
    }

    /// Fallible Jacobian recalculation used by the solver's `try_*` path.
    ///
    /// Shape validation uses the backend's native `shape()` metadata. It does
    /// not convert a matrix to a dense temporary merely to validate a callback.
    fn try_recalc_jacobian(&mut self) -> Result<(), BvpBackendIntegrationError> {
        if self.jac_recalc {
            let p = self.p;
            let y = &*self.y;
            info!("\n \n JACOBIAN (RE)CALCULATED! \n \n");
            let begin = Instant::now();
            self.custom_timer.jac_tic();
            let jac_matrix = if let Some(jac_function) = self.jac.as_mut() {
                match jac_function.try_call(p, y) {
                    Ok(jacobian) => jacobian,
                    Err(error) => {
                        self.custom_timer.jac_tac();
                        return Err(BvpBackendIntegrationError::CallbackExecutionFailed {
                            stage: "Jacobian".to_string(),
                            message: error.to_string(),
                        });
                    }
                }
            } else {
                let y_runtime =
                    Vectors_type_casting(&y.to_DVectorType(), self.effective_runtime_method());
                match try_finite_difference_jacobian(&*self.fun, p, &*y_runtime, 1e-8) {
                    Ok(jacobian) => jacobian,
                    Err(error) => {
                        self.custom_timer.jac_tac();
                        return Err(BvpBackendIntegrationError::CallbackExecutionFailed {
                            stage: "Jacobian/finite-difference residual".to_string(),
                            message: error.to_string(),
                        });
                    }
                }
            };
            let (actual_rows, actual_columns) = jac_matrix.shape();
            let expected = y.len();
            if actual_rows != expected || actual_columns != expected {
                self.custom_timer.jac_tac();
                return Err(BvpBackendIntegrationError::CallbackShapeMismatch {
                    stage: "Jacobian".to_string(),
                    expected_rows: expected,
                    expected_columns: expected,
                    actual_rows,
                    actual_columns,
                });
            }
            info!("jacobian recalculation time: ");
            let elapsed = begin.elapsed();
            elapsed_time(elapsed);
            self.custom_timer.jac_tac();
            let had_owned_factor = self.factor_owner.borrow().is_some();
            if had_owned_factor
                || self
                    .factor_owner
                    .old_jac
                    .as_ref()
                    .map(|jacobian| jacobian.factorization_ready())
                    .unwrap_or(false)
            {
                self.telemetry_counters.record_factorization_invalidation();
            }
            self.factor_owner.replace_numeric_jacobian(jac_matrix);
            let prepared_owner = self.factor_owner.jacobian().and_then(|jacobian| {
                prepare_factor_owner_runtime(
                    jacobian.as_ref(),
                    self.factor_owner.bandwidth(),
                    self.linear_sys_method.as_deref(),
                )
            });
            self.factor_owner.replace_factor(prepared_owner);
            self.prepared_runtime_revision
                .mark_numeric_jacobian_current();
            if self.factor_owner.borrow().is_some() {
                self.prepared_runtime_revision.mark_factor_current();
            }
            self.m = 0;
            self.telemetry_counters.record_jacobian_recalculation();
        }
        Ok(())
    }

    /// Recalculates and validates the prepared Jacobian through the public
    /// typed boundary.
    ///
    /// This is the fallible counterpart of the historical internal
    /// recalculation path. It is intentionally a thin wrapper: validation,
    /// factor invalidation and telemetry remain owned by the solver runtime.
    /// New integrations should use this method instead of relying on the
    /// panic-based compatibility solver methods.
    pub fn try_recalculate_jacobian(&mut self) -> Result<(), BvpBackendIntegrationError> {
        self.try_recalc_jacobian()
    }
}
