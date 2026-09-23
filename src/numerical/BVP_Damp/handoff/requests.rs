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
    /// AtomView lowering optimization profile carried into generated AOT setup.
    pub atom_optimization_profile: AtomOptimizationProfile,
    /// Symbolic assembly backend used before lambdify/AOT lowering.
    pub symbolic_assembly_backend: BvpSymbolicAssemblyBackend,
    /// Optional explicit matrix backend override for generated sparse/banded handoff.
    pub matrix_backend_override: Option<MatrixBackend>,
    /// Native linear solver configuration used by the generated banded runtime path.
    pub banded_linear_solver_config: LinearSolverConfig,
    /// Runtime collection mode for Lambdify callbacks.
    pub lambdify_telemetry_mode: BvpLambdifyTelemetryMode,
    /// Runtime collection mode for AtomView AOT preparation and lifecycle.
    pub aot_telemetry_mode: BvpAotTelemetryMode,
    /// Runtime execution policy for pure Lambdify residual/Jacobian callbacks.
    pub lambdify_execution_policy: BvpLambdifyExecutionPolicy,
}

// The request carries the pure-Lambdify execution policy separately from AOT
// execution policy so both solver variants can select the callback regime.
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
            self.atom_optimization_profile,
            self.symbolic_assembly_backend,
            self.matrix_backend_override,
            self.banded_linear_solver_config,
            self.lambdify_telemetry_mode,
            self.aot_telemetry_mode,
            self.lambdify_execution_policy,
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
    /// AtomView lowering optimization profile carried into generated AOT setup.
    pub atom_optimization_profile: AtomOptimizationProfile,
    /// Symbolic assembly backend used before lambdify/AOT lowering.
    pub symbolic_assembly_backend: BvpSymbolicAssemblyBackend,
    /// Optional explicit matrix backend override for generated sparse/banded handoff.
    pub matrix_backend_override: Option<MatrixBackend>,
    /// Native linear solver configuration used by the generated banded runtime path.
    pub banded_linear_solver_config: LinearSolverConfig,
    /// Runtime collection mode for Lambdify callbacks.
    pub lambdify_telemetry_mode: BvpLambdifyTelemetryMode,
    /// Runtime collection mode for AtomView AOT preparation and lifecycle.
    pub aot_telemetry_mode: BvpAotTelemetryMode,
    /// Runtime execution policy for pure Lambdify residual/Jacobian callbacks.
    pub lambdify_execution_policy: BvpLambdifyExecutionPolicy,
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
            self.atom_optimization_profile,
            self.symbolic_assembly_backend,
            self.matrix_backend_override,
            self.banded_linear_solver_config,
            self.lambdify_telemetry_mode,
            self.aot_telemetry_mode,
            self.lambdify_execution_policy,
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
