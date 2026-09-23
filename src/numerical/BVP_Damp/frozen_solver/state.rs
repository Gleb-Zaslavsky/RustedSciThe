pub struct NRBVP {
    pub eq_system: Vec<Expr>,
    pub initial_guess: DMatrix<f64>,
    pub values: Vec<String>,
    pub arg: String,
    pub param_names: Vec<String>,
    pub param_values: Option<Vec<f64>>,
    pub BorderConditions: HashMap<String, Vec<(usize, f64)>>,
    pub t0: f64,
    pub t_end: f64,
    pub n_steps: usize,
    pub scheme: String,
    pub strategy: String,
    pub strategy_params: Option<HashMap<String, Option<Vec<f64>>>>,
    pub linear_sys_method: Option<String>,
    pub method: String,
    pub tolerance: f64,
    pub max_iterations: usize,
    pub max_error: f64,
    pub result: Option<DVector<f64>>,
    pub x_mesh: DVector<f64>,

    pub fun: Box<dyn Fun>,
    pub jac: Option<Box<dyn Jac>>,
    pub p: f64,
    pub y: Box<dyn VectorType>,
    m: usize, // iteration counter without jacobian recalculation
    /// Internal prepared factor for direct Dense/faer Frozen solves.
    /// Banded keeps ownership inside `BandedMatrixType`; legacy callers never
    /// see this field or the factor-owner runtime.
    /// Common prepared-runtime owner for reusable Dense/faer factors.
    ///
    /// Banded ownership remains inside `BandedMatrixType`; callbacks and
    /// mesh/layout are separate migration slices.
    factor_owner: BvpPreparedRuntime,
    prepared_runtime_revision: BvpRuntimeRevision,
    jac_recalc: bool,
    error_old: f64,
    variable_string: Vec<String>, // vector of indexed variable names
    bandwidth: (usize, usize),
    generated_backend_config: GeneratedBackendConfig,
    generated_backend_selected_backend: Option<SelectedBackendKind>,
    generated_backend_runtime_diagnostics: HashMap<String, String>,
    telemetry_counters: BvpTelemetryRecorder,
    generation_telemetry: Option<BvpGenerationTelemetrySnapshot>,
    atom_discretization_telemetry:
        Option<crate::symbolic::bvp::telemetry::BvpAtomDiscretizationTelemetrySnapshot>,
    legacy_lambdify_telemetry: Option<BvpLambdifyTelemetry>,
    atom_lambdify_telemetry: Option<BvpLambdifyTelemetry>,
    direct_banded_jacobian_telemetry:
        Option<crate::symbolic::bvp::telemetry::BvpDirectJacobianTelemetry>,
    parameter_binding: Option<crate::symbolic::bvp::parameter_binding::BvpParameterBindingHandle>,
    custom_timer: CustomTimer,
    no_reports: bool,
}

impl ApplyFrozenGeneratedSolverState for NRBVP {
    fn apply_generated_solver_state(&mut self, state: FrozenGeneratedSolverState) {
        self.prepared_runtime_revision.callbacks_changed();
        self.prepared_runtime_revision.artifact_changed();
        if matches!(state.selected_backend, SelectedBackendKind::AotCompiled) {
            self.prepared_runtime_revision.linked_runtime_changed();
        }
        if let Some(updated_resolver) = state.updated_resolver.clone() {
            self.generated_backend_config.resolver = Some(updated_resolver);
        }
        self.fun = state.fun;
        self.jac = state.jac;
        self.factor_owner
            .replace_layout(state.variable_string.clone(), state.bandwidth);
        self.variable_string = state.variable_string;
        self.bandwidth = state.bandwidth;
        self.generated_backend_selected_backend = Some(state.selected_backend);
        let backend_fallback = matches!(
            self.generated_backend_config
                .effective_backend_policy(&self.method),
            BackendSelectionPolicy::PreferAotThenLambdify
                | BackendSelectionPolicy::PreferAotThenNumeric
        ) && matches!(
            state.selected_backend,
            SelectedBackendKind::Lambdify | SelectedBackendKind::Numeric
        );
        self.telemetry_counters.record_backend_selection(
            match state.selected_backend {
                SelectedBackendKind::Numeric => 1,
                SelectedBackendKind::Lambdify => 2,
                SelectedBackendKind::AotCompiled => 3,
                SelectedBackendKind::AotRegisteredButNotBuilt => 4,
                SelectedBackendKind::AotMissing => 5,
            },
            backend_fallback,
        );
        self.generated_backend_runtime_diagnostics = state.runtime_diagnostics;
        self.generation_telemetry = state.generation_telemetry;
        self.atom_discretization_telemetry = state.atom_discretization_telemetry;
        self.legacy_lambdify_telemetry = state.legacy_lambdify_telemetry;
        self.atom_lambdify_telemetry = state.atom_lambdify_telemetry;
        self.direct_banded_jacobian_telemetry = state.direct_banded_jacobian_telemetry;
        self.parameter_binding = state.parameter_binding;
        self.invalidate_linear_runtime();
        let prepared_fingerprint = self.prepared_plan_fingerprint();
        self.prepared_runtime_revision
            .mark_prepared_with_fingerprint(prepared_fingerprint);
        self.factor_owner
            .publish_prepared_binding(prepared_fingerprint);
    }
}

impl BuildFrozenSolverRequest for NRBVP {
    fn build_solver_request(&self) -> FrozenSolverBuildRequest {
        let h = (self.t_end - self.t0) / self.n_steps as f64;
        let effective_method = self.generated_backend_config.effective_method(&self.method);
        FrozenSolverBuildRequest {
            eq_system: self.eq_system.clone(),
            values: self.values.clone(),
            arg: self.arg.clone(),
            param_names: (!self.param_names.is_empty()).then(|| self.param_names.clone()),
            param_values: self.param_values.clone(),
            t0: self.t0,
            n_steps: Some(self.n_steps),
            h: Some(h),
            mesh: None,
            border_conditions: self.BorderConditions.clone(),
            scheme: self.scheme.clone(),
            method: effective_method.clone(),
            bandwidth: None,
            backend_policy: self
                .generated_backend_config
                .effective_backend_policy(&effective_method),
            resolver: self.generated_backend_config.resolver.clone(),
            aot_execution_policy: self.generated_backend_config.aot_execution_policy.clone(),
            aot_build_policy: self.generated_backend_config.aot_build_policy,
            aot_compile_config: self.generated_backend_config.aot_compile_config.clone(),
            aot_codegen_backend: self.generated_backend_config.aot_codegen_backend,
            aot_c_compiler: self.generated_backend_config.aot_c_compiler.clone(),
            aot_chunking_policy: self.generated_backend_config.aot_chunking_policy,
            atom_optimization_profile: self.generated_backend_config.atom_optimization_profile,
            symbolic_assembly_backend: self.generated_backend_config.symbolic_assembly_backend,
            matrix_backend_override: self.generated_backend_config.matrix_backend_override,
            banded_linear_solver_config: self.generated_backend_config.banded_linear_solver_config,
            lambdify_telemetry_mode: self.generated_backend_config.lambdify_telemetry_mode,
            aot_telemetry_mode: self.generated_backend_config.aot_telemetry_mode,
            lambdify_execution_policy: self.generated_backend_config.lambdify_execution_policy,
        }
    }
}
