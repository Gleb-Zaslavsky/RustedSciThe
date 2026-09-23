use log::{error, info, warn};

use super::BVP_utils::checkmem;

/// Main solver structure for damped Newton-Raphson method
///
/// Contains all problem parameters, solution state, and solver configuration.
/// This is the central structure that orchestrates the entire BVP solving process.
pub struct NRBVP {
    pub eq_system: Vec<Expr>, // the system of ODEs defined in the symbolic format
    pub initial_guess: DMatrix<f64>, // initial guess s - matrix with number of rows equal to the number of unknown vars, and number of columns equal to the number of steps
    pub values: Vec<String>,         //unknown variables
    pub arg: String,                 // time or coordinate
    pub BorderConditions: HashMap<String, Vec<(usize, f64)>>, // hashmap where keys are variable names and values are vectors of tuples with the index of the boundary condition (0 for inititial condition 1 for ending condition) and the value.
    pub t0: f64,                                              // initial value of argument
    pub t_end: f64,                                           // end of argument
    pub n_steps: usize,                                       // number of  steps
    pub scheme: String,                                       // name of the numerical scheme
    pub strategy: String,                                     // name of the strategy
    pub strategy_params: Option<SolverParams>,                // solver parameters
    pub linear_sys_method: Option<String>,                    // method for solving linear system
    pub method: String,     // define crate using for matrices and vectors
    pub abs_tolerance: f64, // relative tolerance

    pub rel_tolerance: Option<HashMap<String, f64>>, // absolute tolerance - hashmap of the var names and values of tolerance for them
    pub max_iterations: usize,                       // maximum number of iterations
    pub max_error: f64,
    pub Bounds: Option<HashMap<String, (f64, f64)>>, // hashmap where keys are variable names and values are tuples with lower and upper bounds.
    pub loglevel: Option<String>,
    pub param_names: Vec<String>, // symbolic parameter names used by RHS but not solved by Newton
    pub param_values: Option<Vec<f64>>, // current numeric values for param_names
    no_reports: bool,
    // thets all user defined  parameters
    //
    pub result: Option<DVector<f64>>, // result vector of calculation
    pub full_result: Option<DMatrix<f64>>,
    pub x_mesh: DVector<f64>,
    pub fun: Box<dyn Fun>, // vector representing the discretized sysytem
    pub jac: Option<Box<dyn Jac>>, // matrix function of Jacobian
    pub p: f64,            // parameter
    pub y: Box<dyn VectorType>, // iteration vector
    m: usize,              // iteration counter without jacobian recalculation
    pub BC_position_and_value: Vec<(usize, usize, f64)>, //  where keys are positions of boundary conditions in the global vector and values are the boundary condition values.
    /// Common prepared-runtime owner for reusable Dense/faer factors.
    ///
    /// The historical field name is retained internally during migration;
    /// callbacks and mesh/layout are still migrated in separate ownership
    /// slices.
    factor_owner: BvpPreparedRuntime,
    prepared_runtime_revision: BvpRuntimeRevision,
    jac_recalc: bool,            //flag indicating if jacobian should be recalculated
    error_old: f64,              // error of previous iteration
    bounds_vec: Vec<(f64, f64)>, //vector of bounds for each of the unkown variables (discretized vector)
    rel_tolerance_vec: Vec<f64>, // vector of relative tolerance for each of the unkown variables
    variable_string: Vec<String>, // vector of indexed variable names
    #[allow(dead_code)]
    adaptive: bool, // flag indicating if adaptive grid should be used
    pub new_grid_enabled: bool,  //flag indicating if the grid should be refined
    grid_refinemens: usize,      //
    number_of_refined_intervals: usize, //number of refined intervals
    bandwidth: (usize, usize),   //bandwidth
    generated_backend_config: GeneratedBackendConfig, // generated backend selection config
    numeric_rhs: Option<NumericBvpRhs>, // pure numeric RHS source for NumericOnly route
    numeric_jacobian: Option<NumericBvpJacobian>, // optional continuous RHS Jacobian
    generated_backend_selected_backend: Option<SelectedBackendKind>,
    generated_backend_runtime_diagnostics: HashMap<String, String>,
    /// Shared native AOT telemetry handle; diagnostics are materialized only
    /// when statistics are requested, outside callback hot paths.
    aot_telemetry: Option<BvpAotTelemetry>,
    telemetry_counters: BvpTelemetryRecorder,
    generation_telemetry: Option<BvpGenerationTelemetrySnapshot>,
    atom_discretization_telemetry:
        Option<crate::symbolic::bvp::telemetry::BvpAtomDiscretizationTelemetrySnapshot>,
    legacy_lambdify_telemetry: Option<BvpLambdifyTelemetry>,
    atom_lambdify_telemetry: Option<BvpLambdifyTelemetry>,
    direct_banded_jacobian_telemetry:
        Option<crate::symbolic::bvp::telemetry::BvpDirectJacobianTelemetry>,
    parameter_binding: Option<crate::symbolic::bvp::parameter_binding::BvpParameterBindingHandle>,
    nodes_added: Vec<usize>,
    custom_timer: CustomTimer,
}

impl ApplyDampedGeneratedSolverState for NRBVP {
    fn apply_generated_solver_state(&mut self, state: DampedGeneratedSolverState) {
        // A regenerated callback bundle invalidates both the factor and the
        // numeric Jacobian produced by the previous bundle.
        self.prepared_runtime_revision.callbacks_changed();
        self.prepared_runtime_revision.artifact_changed();
        if matches!(state.selected_backend, SelectedBackendKind::AotCompiled) {
            self.prepared_runtime_revision.linked_runtime_changed();
        }
        self.factor_owner.clear_numeric_jacobian();
        if let Some(updated_resolver) = state.updated_resolver.clone() {
            self.generated_backend_config.resolver = Some(updated_resolver);
        }
        self.fun = state.fun;
        self.jac = state.jac;
        self.bounds_vec = state.bounds_vec;
        self.rel_tolerance_vec = state.rel_tolerance_vec;
        self.factor_owner
            .replace_layout(state.variable_string.clone(), state.bandwidth);
        self.variable_string = state.variable_string;
        self.bandwidth = state.bandwidth;
        self.BC_position_and_value = state.bc_position_and_value;
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
        self.aot_telemetry = state.aot_telemetry;
        self.generation_telemetry = state.generation_telemetry;
        self.atom_discretization_telemetry = state.atom_discretization_telemetry;
        self.legacy_lambdify_telemetry = state.legacy_lambdify_telemetry;
        self.atom_lambdify_telemetry = state.atom_lambdify_telemetry;
        self.direct_banded_jacobian_telemetry = state.direct_banded_jacobian_telemetry;
        self.parameter_binding = state.parameter_binding;
        let prepared_fingerprint = self.prepared_plan_fingerprint();
        self.prepared_runtime_revision
            .mark_prepared_with_fingerprint(prepared_fingerprint);
        self.factor_owner
            .publish_prepared_binding(prepared_fingerprint);
    }
}

impl BuildDampedSolverRequest for NRBVP {
    fn build_solver_request(
        &mut self,
        mesh_: Option<Vec<f64>>,
        bandwidth: Option<(usize, usize)>,
    ) -> DampedSolverBuildRequest {
        let (h, n_steps, mesh) = if mesh_.is_none() {
            let h = Some((self.t_end - self.t0) / self.n_steps as f64);
            let n_steps = Some(self.n_steps);
            (h, n_steps, None)
        } else {
            self.x_mesh = DVector::from_vec(mesh_.clone().unwrap());
            (None, None, mesh_)
        };
        let effective_method = self.generated_backend_config.effective_method(&self.method);

        DampedSolverBuildRequest {
            eq_system: self.eq_system.clone(),
            values: self.values.clone(),
            param_names: (!self.param_names.is_empty()).then(|| self.param_names.clone()),
            param_values: self.param_values.clone(),
            arg: self.arg.clone(),
            t0: self.t0,
            n_steps,
            h,
            mesh,
            border_conditions: self.BorderConditions.clone(),
            bounds: self.Bounds.clone(),
            rel_tolerance: self.rel_tolerance.clone(),
            scheme: self.scheme.clone(),
            method: effective_method.clone(),
            bandwidth,
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
