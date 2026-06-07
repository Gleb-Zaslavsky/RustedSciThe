use crate::symbolic::symbolic_engine::Expr;
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;
use std::time::Instant;

use crate::Utils::logger::save_matrix_to_file;
use crate::Utils::plots::plots;
use crate::Utils::postprocessing::{
    PostprocessDataset, PostprocessError, PostprocessPlan, PostprocessReport,
};
use crate::numerical::BVP_Damp::BVP_traits::{
    Fun, FunEnum, Jac, MatrixType, VectorType, Vectors_type_casting,
};
use crate::numerical::BVP_Damp::BVP_utils::*;
use crate::numerical::BVP_Damp::NR_Damp_solver_damped::BvpDerivativeScheme;
use crate::numerical::BVP_Damp::generated_solver_handoff::{
    AotBuildPolicy, AotChunkingPolicy, AotExecutionPolicy, ApplyFrozenGeneratedSolverState,
    BandedGeneratedBackendMode, BuildFrozenSolverRequest, FrozenGeneratedSolverState,
    FrozenSolverBuildRequest, GeneratedBackendConfig, SparseGeneratedBackendMode,
    try_generate_and_apply_frozen_solver_state,
};
use crate::numerical::BVP_Damp::solver_common::{
    DEFAULT_MAX_ITERATIONS, cleanup_registered_aot_artifacts, default_dense_method_name,
    default_forward_scheme_name, default_placeholder_y, default_sparse_method_name,
    frozen_point_mesh,
};
use crate::somelinalg::banded::LinearSolverConfig;
use crate::symbolic::codegen::CodegenIR::AtomOptimizationProfile;
use crate::symbolic::codegen::codegen_aot_resolution::AotResolver;
use crate::symbolic::codegen::codegen_backend_selection::{
    BackendSelectionPolicy, SelectedBackendKind,
};
use crate::symbolic::symbolic_functions_BVP::{
    BvpBackendIntegrationError, BvpSymbolicAssemblyBackend,
};

use chrono::Local;

use log::info;

use simplelog::*;

use std::fs::File;

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

/// Runtime counters, timers, and generated-backend diagnostics for Frozen Newton.
#[derive(Clone, Debug)]
pub struct FrozenBvpStatistics {
    pub counters: HashMap<String, usize>,
    pub timers: HashMap<String, String>,
    pub diagnostics: HashMap<String, String>,
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
    old_jac: Option<Box<dyn MatrixType>>,
    jac_recalc: bool,
    error_old: f64,
    variable_string: Vec<String>, // vector of indexed variable names
    bandwidth: (usize, usize),
    generated_backend_config: GeneratedBackendConfig,
    generated_backend_selected_backend: Option<SelectedBackendKind>,
    generated_backend_runtime_diagnostics: HashMap<String, String>,
    calc_statistics: HashMap<String, usize>,
    custom_timer: CustomTimer,
    no_reports: bool,
}

impl ApplyFrozenGeneratedSolverState for NRBVP {
    fn apply_generated_solver_state(&mut self, state: FrozenGeneratedSolverState) {
        if let Some(updated_resolver) = state.updated_resolver.clone() {
            self.generated_backend_config.resolver = Some(updated_resolver);
        }
        self.fun = state.fun;
        self.jac = state.jac;
        self.variable_string = state.variable_string;
        self.bandwidth = state.bandwidth;
        self.generated_backend_selected_backend = Some(state.selected_backend);
        self.generated_backend_runtime_diagnostics = state.runtime_diagnostics;
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
        }
    }
}

impl NRBVP {
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
            old_jac: None,
            jac_recalc: true,
            error_old: 0.0,
            variable_string: Vec::new(), // vector of indexed variable names
            bandwidth: (0, 0),
            generated_backend_config: GeneratedBackendConfig::default(),
            generated_backend_selected_backend: None,
            generated_backend_runtime_diagnostics: HashMap::new(),
            calc_statistics: HashMap::from([
                ("number of iterations".to_string(), 0),
                ("number of jacobians recalculations".to_string(), 0),
                ("number of solving linear systems".to_string(), 0),
            ]),
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

    /// Basic methods to set the equation system

    ///Set system of equations with vector of symbolic expressions
    pub fn task_check(&self) {
        if self.t_end < self.t0 {
            panic!("Frozen BVP task check failed: t_end must be greater than t0");
        }

        if self.n_steps < 1 {
            panic!("Frozen BVP task check failed: n_steps must be greater than 1");
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

    pub fn try_eq_generate(&mut self) -> Result<(), BvpBackendIntegrationError> {
        self.task_check();
        strategy_check(&self.strategy, &self.strategy_params);
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
        self.generated_backend_config.resolver = resolver;
    }

    /// Installs the solver-level AOT execution policy.
    pub fn set_aot_execution_policy(&mut self, policy: AotExecutionPolicy) {
        self.generated_backend_config.aot_execution_policy = policy;
    }

    /// Returns the configured solver-level AOT execution policy.
    pub fn aot_execution_policy(&self) -> &AotExecutionPolicy {
        &self.generated_backend_config.aot_execution_policy
    }

    /// Installs the solver-level AOT build policy.
    pub fn set_aot_build_policy(&mut self, policy: AotBuildPolicy) {
        self.generated_backend_config.aot_build_policy = policy;
    }

    /// Returns the configured solver-level AOT build policy.
    pub fn aot_build_policy(&self) -> AotBuildPolicy {
        self.generated_backend_config.aot_build_policy
    }

    /// Installs explicit solver-level AOT chunking overrides.
    pub fn set_aot_chunking_policy(&mut self, policy: AotChunkingPolicy) {
        self.generated_backend_config.aot_chunking_policy = policy;
    }

    /// Returns the configured solver-level AOT chunking overrides.
    pub fn aot_chunking_policy(&self) -> AotChunkingPolicy {
        self.generated_backend_config.aot_chunking_policy
    }

    /// Installs an explicit AtomView AOT optimization profile.
    pub fn set_atom_optimization_profile(&mut self, profile: AtomOptimizationProfile) {
        self.generated_backend_config.atom_optimization_profile = profile;
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
        self.generated_backend_config.backend_policy_override = backend_policy;
    }

    /// Returns the configured generated-backend selection policy override, if present.
    pub fn backend_policy_override(&self) -> Option<BackendSelectionPolicy> {
        self.generated_backend_config.backend_policy_override
    }

    /// Installs the complete generated-backend configuration in one call.
    pub fn set_generated_backend_config(&mut self, config: GeneratedBackendConfig) {
        self.generated_backend_config = config;
    }

    /// Installs the symbolic parameter names used by the generated residual/Jacobian.
    pub fn set_params(&mut self, params: Option<&[&str]>) {
        self.param_names = params
            .map(|items| items.iter().map(|name| (*name).to_string()).collect())
            .unwrap_or_default();
        if let Some(values) = self.param_values.as_ref() {
            assert_eq!(
                values.len(),
                self.param_names.len(),
                "param_values length must match param_names length"
            );
        }
    }

    /// Installs the numeric values for the already-declared symbolic parameters.
    pub fn set_param_values(&mut self, values: Option<Vec<f64>>) {
        if let Some(values_ref) = values.as_ref() {
            assert_eq!(
                values_ref.len(),
                self.param_names.len(),
                "param_values length must match param_names length"
            );
        }
        self.param_values = values;
    }

    /// Sets the symbolic assembly backend used before lambdify/AOT lowering.
    pub fn set_symbolic_assembly_backend(&mut self, backend: BvpSymbolicAssemblyBackend) {
        self.generated_backend_config = self
            .generated_backend_config
            .clone()
            .with_symbolic_assembly_backend(backend);
    }

    /// Installs a high-level sparse generated-backend mode on an existing solver.
    pub fn set_sparse_generated_backend_mode(&mut self, mode: SparseGeneratedBackendMode) {
        self.generated_backend_config = GeneratedBackendConfig::from_sparse_mode(mode);
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
        let mut counters = self.calc_statistics.clone();
        if let Some(jac) = &self.old_jac {
            let shape = jac.shape();
            counters.insert("number of jacobian elements".to_string(), shape.0 * shape.1);
        }
        counters.insert("length of y vector".to_string(), self.y.len());
        counters.insert("number of grid points".to_string(), self.x_mesh.len());
        let mut diagnostics = self.generated_backend_runtime_diagnostics.clone();
        self.append_generated_backend_diagnostics(&mut diagnostics);
        FrozenBvpStatistics {
            counters,
            timers: self.custom_timer.get_all(),
            diagnostics,
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
        self.p = p;
        self.y = y;
        self.initial_guess = initial_guess;
    }
    pub fn set_p(&mut self, p: f64) {
        self.p = p;
    }

    ///Newton-Raphson method
    /// realize iteration of Newton-Raphson - calculate new iteration vector by using Jacobian matrix
    pub fn iteration(&mut self) -> Box<dyn VectorType> {
        let p = self.p;
        let y = &*self.y;
        let fun = &self.fun;
        let fun_begin = Instant::now();
        let new_fun = fun.call(p, y);
        self.custom_timer.append_to_fun_time(fun_begin.elapsed());
        let jac = self
            .jac
            .as_mut()
            .expect("Frozen BVP iteration requires an installed Jacobian callback");
        let now = Instant::now();

        let new_j = if self.jac_recalc {
            info!("\n \n JACOBIAN (RE)CALCULATED! \n \n");
            let begin = Instant::now();
            self.custom_timer.jac_tic();
            let new_j = jac.call(p, y);
            info!("jacobian recalculation time: ");
            let elapsed = begin.elapsed();
            elapsed_time(elapsed);
            self.custom_timer.jac_tac();
            self.old_jac = Some(new_j.clone_box());
            self.m = 0;
            *self
                .calc_statistics
                .entry("number of jacobians recalculations".to_string())
                .or_insert(0) += 1;
            new_j
        } else {
            self.m = self.m + 1;
            self.old_jac
                .as_ref()
                .expect(
                    "Frozen BVP iteration requires a cached Jacobian matrix when reuse is enabled",
                )
                .clone_box()
        };

        //   println!("new fun = {:?}", &new_fun);
        let linear_begin = Instant::now();
        let delta: Box<dyn VectorType> = new_j.solve_sys(
            &*new_fun,
            self.linear_sys_method.clone(),
            self.tolerance,
            self.max_iterations,
            self.bandwidth,
            y,
        );
        self.custom_timer
            .append_to_linear_sys_time(linear_begin.elapsed());
        *self
            .calc_statistics
            .entry("number of solving linear systems".to_string())
            .or_insert(0) += 1;
        let elapsed = now.elapsed();
        elapsed_time(elapsed);
        //  println!(" \n \n dy= {:?}", &delta);
        // element wise subtraction
        let new_y: Box<dyn VectorType> = y - &*delta;

        new_y
    }
    pub fn main_loop(&mut self) -> Option<DVector<f64>> {
        self.try_main_loop().unwrap_or_else(|err| {
            panic!("Frozen BVP main loop failed during fallible runtime path: {err:?}")
        })
    }

    /// Fallible internal Newton loop used by [`NRBVP::try_solver`].
    pub fn try_main_loop(&mut self) -> Result<Option<DVector<f64>>, BvpBackendIntegrationError> {
        info!("solving system of equations with Newton-Raphson method! \n \n");
        let y: DMatrix<f64> = self.initial_guess.clone();
        let y: Vec<f64> = y.iter().cloned().collect();
        let y: DVector<f64> = DVector::from_vec(y);
        self.y = Vectors_type_casting(&y.clone(), self.method.clone());
        let mut i = 0;

        while i < self.max_iterations {
            *self
                .calc_statistics
                .entry("number of iterations".to_string())
                .or_insert(0) += 1;
            let new_y = self.iteration();
            let y1 = new_y.subtract(&*self.y);
            let dy: Box<dyn VectorType> = y1.clone_box();

            let error = dy.norm();
            self.jac_recalc = frozen_jac_recalc(
                &self.strategy,
                &self.strategy_params,
                &self.old_jac,
                self.m,
                error,
                self.error_old,
            );
            self.error_old = error;
            info!(" \n \n error = {:?} \n \n", &error);
            if error < self.tolerance {
                log::info!("converged in {} iterations, error = {}", i, error);
                self.result = Some(new_y.to_DVectorType());
                self.max_error = error;
                return Ok(Some(new_y.to_DVectorType()));
            } else {
                let new_y: Box<dyn VectorType> = new_y.clone_box();
                self.y = new_y;
                i += 1;
            }
        }
        Ok(None)
    }
    /// Fallible solve path without logging setup.
    ///
    /// This is the preferred entrypoint for internal/runtime callers that want
    /// typed backend and execution errors but do not need the higher-level
    /// logging wrapper provided by [`NRBVP::try_solve`].
    pub fn try_solver(&mut self) -> Result<Option<DVector<f64>>, BvpBackendIntegrationError> {
        // TODO! сравнить явный мэш с неявным
        // let test_mesh = Some((0..100).map(|x| 0.01 * x as f64).collect::<Vec<f64>>());
        self.custom_timer.start();
        self.custom_timer.symbolic_operations_tic();
        self.try_eq_generate()?;
        self.custom_timer.symbolic_operations_tac();
        let begin = Instant::now();
        let res = self.try_main_loop()?;
        let end = begin.elapsed();
        elapsed_time(end);

        Ok(res)
    }
    /// Compatibility-only wrapper over [`NRBVP::try_solver`].
    ///
    /// New production-facing code should call [`NRBVP::try_solver`] or
    /// [`NRBVP::try_solve`] instead.
    pub fn solver(&mut self) -> Option<DVector<f64>> {
        self.try_solver()
            .unwrap_or_else(|err| panic!("Frozen BVP solver failed before Newton loop: {err:?}"))
    }

    /// Main public fallible solve entrypoint with logging support.
    ///
    /// This is the preferred production-facing solve path.
    pub fn try_solve(&mut self) -> Result<Option<DVector<f64>>, BvpBackendIntegrationError> {
        let logger_instance = if self.no_reports {
            let logger_instance = CombinedLogger::init(vec![TermLogger::new(
                LevelFilter::Info,
                Config::default(),
                TerminalMode::Mixed,
                ColorChoice::Auto,
            )]);
            logger_instance
        } else {
            let date_and_time = Local::now().format("%Y-%m-%d_%H-%M");
            let name = format!("log_{}.txt", date_and_time);
            let file = File::create(&name).map_err(|err| {
                BvpBackendIntegrationError::LogFileCreationFailed {
                    path: name.clone(),
                    message: err.to_string(),
                }
            })?;
            let logger_instance = CombinedLogger::init(vec![
                TermLogger::new(
                    LevelFilter::Info,
                    Config::default(),
                    TerminalMode::Mixed,
                    ColorChoice::Auto,
                ),
                WriteLogger::new(LevelFilter::Info, Config::default(), file),
            ]);
            logger_instance
        };
        match logger_instance {
            Ok(()) => {
                let res = self.try_solver()?;
                log::info!("Program ended");
                Ok(res)
            }
            Err(_) => self.try_solver(),
        }
    }
    /// Compatibility-only wrapper over [`NRBVP::try_solve`].
    ///
    /// New code should prefer [`NRBVP::try_solve`] so AOT/logging/runtime failures
    /// remain typed instead of turning into a panic.
    pub fn solve(&mut self) -> Option<DVector<f64>> {
        self.try_solve().unwrap_or_else(|err| {
            panic!("Frozen BVP solve failed before convergence loop: {err:?}")
        })
    }
    pub fn dont_save_log(&mut self, dont_save_log: bool) {
        self.no_reports = dont_save_log;
    }
    pub fn save_to_file(&self) {
        //let date_and_time = Local::now().format("%Y-%m-%d_%H-%M-%S");
        let result_DMatrix = self
            .get_result()
            .expect("Frozen BVP save_to_file requires a computed solution matrix");
        let _ = save_matrix_to_file(
            &result_DMatrix,
            &self.values,
            "result.txt",
            &self.x_mesh,
            &self.arg,
        );
    }
    pub fn get_result(&self) -> Option<DMatrix<f64>> {
        let number_of_Ys = self.values.len();
        let n_steps = self.n_steps;
        let vector_of_results = self
            .result
            .clone()
            .expect("Frozen BVP get_result requires a converged solution vector")
            .clone();
        let matrix_of_results: DMatrix<f64> =
            DMatrix::from_column_slice(number_of_Ys, n_steps, vector_of_results.clone().as_slice())
                .transpose();
        let permutted_results = matrix_of_results;
        Some(permutted_results)
    }

    /// Converts the computed solution into the unified postprocessing dataset.
    pub fn postprocess_dataset(&self) -> Result<PostprocessDataset, PostprocessError> {
        if self.result.is_none() {
            return Err(PostprocessError::InvalidDataset(
                "Frozen BVP postprocess_dataset requires a converged solution vector".to_string(),
            ));
        }
        let values = self.get_result().ok_or_else(|| {
            PostprocessError::InvalidDataset(
                "Frozen BVP postprocess_dataset requires a converged solution vector".to_string(),
            )
        })?;
        PostprocessDataset::new(
            self.arg.clone(),
            self.values.clone(),
            self.x_mesh.clone(),
            values,
        )
    }

    /// Executes a declarative postprocessing plan using the modern facade.
    pub fn execute_postprocessing(
        &self,
        plan: &PostprocessPlan,
    ) -> Result<PostprocessReport, PostprocessError> {
        let dataset = self.postprocess_dataset()?;
        plan.execute(&dataset)
    }

    pub fn plot_result(&self) {
        let number_of_Ys = self.values.len();
        let n_steps = self.n_steps;
        let vector_of_results = self
            .result
            .clone()
            .expect("Frozen BVP plot_result requires a converged solution vector")
            .clone();
        let matrix_of_results: DMatrix<f64> =
            DMatrix::from_column_slice(number_of_Ys, n_steps, vector_of_results.clone().as_slice())
                .transpose();
        for _col in matrix_of_results.column_iter() {
            //   println!( "{:?}", DVector::from_column_slice(_col.as_slice()) );
        }
        info!(
            "matrix of results has shape {:?}",
            matrix_of_results.shape()
        );
        info!("length of x mesh : {:?}", n_steps);
        info!("number of Ys: {:?}", number_of_Ys);
        plots(
            self.arg.clone(),
            self.values.clone(),
            self.x_mesh.clone(),
            matrix_of_results,
        );
        info!("result plotted");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::BVP_Damp::generated_solver_handoff::{
        AotBuildPolicy, AotBuildProfile, AotChunkingPolicy, AotExecutionPolicy,
        BandedGeneratedBackendMode, GeneratedBackendConfig, SparseGeneratedBackendMode,
    };
    use crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend;
    use crate::symbolic::codegen::codegen_aot_registry::AotRegistry;
    use crate::symbolic::codegen::codegen_aot_resolution::AotResolver;
    use crate::symbolic::codegen::codegen_aot_runtime_link::{
        LinkedSparseAotBackend, register_linked_sparse_backend, unregister_linked_sparse_backend,
    };
    use crate::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
    use crate::symbolic::codegen::codegen_orchestrator::{
        ParallelExecutorConfig, ParallelFallbackPolicy,
    };
    use crate::symbolic::codegen::codegen_provider_api::MatrixBackend;
    use crate::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy;
    use crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy;
    use crate::symbolic::symbolic_functions_BVP::BvpBackendIntegrationError;
    use faer::Col;
    use nalgebra::{DMatrix, DVector};
    use std::sync::Arc;

    fn sparse_surface_test_solver() -> NRBVP {
        NRBVP::new_with_options(
            vec![Expr::parse_expression("z"), Expr::parse_expression("-y")],
            DMatrix::from_element(2, 4, 0.1),
            vec!["y".to_string(), "z".to_string()],
            "x".to_string(),
            HashMap::from([
                ("y".to_string(), vec![(0, 0.0)]),
                ("z".to_string(), vec![(0, 1.0)]),
            ]),
            0.0,
            1.0,
            4,
            FrozenSolverOptions::sparse_frozen(),
        )
    }

    fn sparse_surface_test_solver_with_naive_strategy() -> NRBVP {
        let options = FrozenSolverOptions::sparse_frozen()
            .with_strategy_params(Some(HashMap::from([("Frozen_naive".to_string(), None)])))
            .with_tolerance(1e-6)
            .with_max_iterations(10);
        NRBVP::new_with_options(
            vec![Expr::parse_expression("y-z"), Expr::parse_expression("-z")],
            DMatrix::from_element(2, 8, 0.5),
            vec!["z".to_string(), "y".to_string()],
            "x".to_string(),
            HashMap::from([
                ("z".to_string(), vec![(0usize, 1.0f64)]),
                ("y".to_string(), vec![(1usize, 1.0f64)]),
            ]),
            0.0,
            1.0,
            8,
            options,
        )
    }

    fn frozen_linear_solver(n_steps: usize, options: FrozenSolverOptions) -> NRBVP {
        // y'' = 0 represented as y' = z, z' = 0 with
        // y(0)=0 and z(0)=1. This gives y=x, z=1 and is exact for
        // the forward first-order BVP stencil, so backend coverage is not
        // polluted by discretization error.
        let values = vec!["y".to_string(), "z".to_string()];
        let t0 = 0.0;
        let t_end = 1.0;
        let h = (t_end - t0) / n_steps as f64;
        let mut guess = vec![0.0; values.len() * n_steps];
        for i in 0..n_steps {
            let x = t0 + (i as f64) * h;
            guess[i * values.len()] = x;
            guess[i * values.len() + 1] = 1.0;
        }

        NRBVP::new_with_options(
            vec![Expr::parse_expression("z"), Expr::parse_expression("0.0")],
            DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(guess).as_slice()),
            values,
            "x".to_string(),
            HashMap::from([
                ("y".to_string(), vec![(0usize, 0.0f64)]),
                ("z".to_string(), vec![(0usize, 1.0f64)]),
            ]),
            t0,
            t_end,
            n_steps,
            options,
        )
    }

    fn assert_frozen_linear_solution_quality(
        solver: &NRBVP,
        n_steps: usize,
        rms_tol: f64,
        max_abs_tol: f64,
    ) {
        let solution = solver
            .get_result()
            .expect("frozen BVP solver should store a solution matrix");
        assert_eq!(solution.nrows(), n_steps);
        assert_eq!(solution.ncols(), 2);

        let h = 1.0 / n_steps as f64;
        let mut sq_sum = 0.0;
        let mut max_abs = 0.0;
        for i in 0..solution.nrows() {
            // Frozen stores the reduced unknown vector. With both linear BVP
            // boundary conditions placed at the left edge, row 0 corresponds
            // to the first free mesh point, not to the boundary itself.
            let x = (i + 1) as f64 * h;
            let y_err = (solution[(i, 0)] - x).abs();
            let z_err = (solution[(i, 1)] - 1.0).abs();
            assert!(
                solution[(i, 0)].is_finite(),
                "frozen y contains non-finite values"
            );
            assert!(
                solution[(i, 1)].is_finite(),
                "frozen z contains non-finite values"
            );
            let err = y_err.max(z_err);
            sq_sum += err * err;
            max_abs = f64::max(max_abs, err);
        }
        let rms = (sq_sum / solution.nrows() as f64).sqrt();
        assert!(
            rms <= rms_tol,
            "frozen linear BVP RMS error too large: rms={rms:e}, tol={rms_tol:e}"
        );
        assert!(
            max_abs <= max_abs_tol,
            "frozen linear BVP max error too large: max={max_abs:e}, tol={max_abs_tol:e}"
        );
    }

    #[derive(Debug)]
    struct FrozenCombustionStoryRow {
        source: &'static str,
        variant: &'static str,
        total_ms: f64,
        solution_diff: f64,
        symbolic_ms: f64,
        linear_ms: f64,
        jacobian_ms: f64,
        residual_ms: f64,
        initial_generate_ms: f64,
        initial_symbolic_jacobian_ms: f64,
        post_build_rebind_ms: f64,
        compile_link_ms: f64,
        residual_jobs: f64,
        jacobian_jobs: f64,
        iterations: usize,
        linear_solves: usize,
        jacobian_rebuilds: usize,
        selected_backend: String,
        build_policy: String,
    }

    fn frozen_combustion_solver(
        n_steps: usize,
        matrix: &'static str,
        config: GeneratedBackendConfig,
    ) -> NRBVP {
        let names = vec!["Teta", "q", "C0", "J0", "C1", "J1"];
        let unknowns = Expr::parse_vector_expression(names.clone());
        let teta = unknowns[0].clone();
        let q = unknowns[1].clone();
        let c0 = unknowns[2].clone();
        let j0 = unknowns[3].clone();
        let j1 = unknowns[5].clone();

        let dt = Expr::Const(600.0);
        let t_scale = Expr::Const(600.0);
        let lambda = Expr::Const(0.07);
        let q_heat = Expr::Const(3000.0 * 1e3 * 0.034);
        let a = Expr::Const(1.3e5);
        let e = Expr::Const(5000.0 * 4.184);
        let m = Expr::Const(34.2 / 1000.0);
        let gas_r = Expr::Const(8.314);
        let ro_m = Expr::Const((34.2 / 1000.0) * 2e6 / (8.314 * 1500.0));
        let qm = Expr::Const((3e-4_f64).powi(2) / 600.0);
        let qs = Expr::Const((3e-4_f64).powi(2));
        let ro_d = Expr::Const(2.88e-4);
        let pe_d = Expr::Const(1.50e-3);
        let rate = a
            * Expr::exp(-e / (gas_r * (teta.clone() * t_scale.clone() + dt.clone())))
            * c0.clone()
            * (ro_m.clone() / Expr::Const(0.342));
        let eqs = vec![
            q.clone() / lambda,
            q * Expr::Const(0.0090168) - q_heat * rate.clone() * qm,
            j0.clone() / ro_d.clone(),
            j0 * pe_d.clone()
                - (m.clone() * Expr::Const(-1.0) * rate.clone() * ro_m.clone() / m.clone())
                    * qs.clone(),
            j1.clone() / ro_d,
            j1 * pe_d - (m.clone() * rate * ro_m / m) * qs,
        ];
        let boundary_conditions = HashMap::from([
            ("Teta".to_string(), vec![(0, (1000.0 - 600.0) / 600.0)]),
            ("q".to_string(), vec![(1, 1e-10)]),
            ("C0".to_string(), vec![(0, 1.0)]),
            ("J0".to_string(), vec![(1, 1e-7)]),
            ("C1".to_string(), vec![(0, 1e-3)]),
            ("J1".to_string(), vec![(1, 1e-7)]),
        ]);
        let initial_guess = DMatrix::from_element(names.len(), n_steps, 0.99);
        let options = match matrix {
            "Sparse" => FrozenSolverOptions::sparse_frozen(),
            "Banded" => FrozenSolverOptions::banded_frozen(),
            _ => panic!("unsupported Frozen combustion story matrix route: {matrix}"),
        }
        .with_generated_backend_config(config)
        .with_tolerance(1e-6)
        .with_max_iterations(100);
        let mut solver = NRBVP::new_with_options(
            eqs,
            initial_guess,
            names.iter().map(|name| (*name).to_string()).collect(),
            "x".to_string(),
            boundary_conditions,
            0.0,
            1.0,
            n_steps,
            options,
        );
        solver.dont_save_log(true);
        solver
    }

    fn frozen_story_timer_ms(stats: &FrozenBvpStatistics, prefix: &str) -> f64 {
        stats
            .timers
            .iter()
            .find(|(name, _)| name.starts_with(prefix))
            .and_then(|(name, value)| {
                let value = value
                    .split(',')
                    .next_back()
                    .and_then(|raw| raw.trim().parse::<f64>().ok())?;
                Some(if name.contains("ms") {
                    value
                } else {
                    value * 1_000.0
                })
            })
            .unwrap_or(f64::NAN)
    }

    fn frozen_story_diagnostic_ms(stats: &FrozenBvpStatistics, key: &str) -> f64 {
        stats
            .diagnostics
            .get(key)
            .and_then(|value| value.parse::<f64>().ok())
            .unwrap_or(f64::NAN)
    }

    fn frozen_story_diagnostic_string(stats: &FrozenBvpStatistics, key: &str) -> String {
        stats
            .diagnostics
            .get(key)
            .cloned()
            .unwrap_or_else(|| "-".to_string())
    }

    fn frozen_story_linf_diff(lhs: &DMatrix<f64>, rhs: &DMatrix<f64>) -> f64 {
        lhs.iter()
            .zip(rhs.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f64, f64::max)
    }

    fn run_frozen_combustion_story_row(
        n_steps: usize,
        matrix: &'static str,
        source: &'static str,
        variant: &'static str,
        config: GeneratedBackendConfig,
        baseline: Option<&DMatrix<f64>>,
    ) -> (
        FrozenCombustionStoryRow,
        DMatrix<f64>,
        GeneratedBackendConfig,
    ) {
        let begin = Instant::now();
        let mut solver = frozen_combustion_solver(n_steps, matrix, config);
        solver.try_solve().unwrap_or_else(|err| {
            panic!("{source}/{variant} Frozen combustion solve failed: {err:?}")
        });
        collect_frozen_story_row(begin, solver, source, variant, baseline)
    }

    fn collect_frozen_story_row(
        begin: Instant,
        solver: NRBVP,
        source: &'static str,
        variant: &'static str,
        baseline: Option<&DMatrix<f64>>,
    ) -> (
        FrozenCombustionStoryRow,
        DMatrix<f64>,
        GeneratedBackendConfig,
    ) {
        let total_ms = begin.elapsed().as_secs_f64() * 1_000.0;
        let solution = solver
            .get_result()
            .expect("successful Frozen solve should expose its result");
        assert!(
            solution.iter().all(|value| value.is_finite()),
            "{source}/{variant} returned non-finite values"
        );
        let stats = solver.get_statistics();
        let row = FrozenCombustionStoryRow {
            source,
            variant,
            total_ms,
            solution_diff: baseline
                .map(|reference| frozen_story_linf_diff(&solution, reference))
                .unwrap_or(0.0),
            symbolic_ms: frozen_story_timer_ms(&stats, "Symbolic Operations"),
            linear_ms: frozen_story_timer_ms(&stats, "Linear System"),
            jacobian_ms: frozen_story_timer_ms(&stats, "Jacobian"),
            residual_ms: frozen_story_timer_ms(&stats, "Function"),
            initial_generate_ms: frozen_story_diagnostic_ms(
                &stats,
                "generated.handoff.initial_generate_wall_ms",
            ),
            initial_symbolic_jacobian_ms: frozen_story_diagnostic_ms(
                &stats,
                "generated.handoff.initial.symbolic_jacobian_time_ms",
            ),
            post_build_rebind_ms: frozen_story_diagnostic_ms(
                &stats,
                "generated.handoff.post_build_rebind_wall_ms",
            ),
            compile_link_ms: frozen_story_diagnostic_ms(&stats, "generated.aot.compile_link_ms"),
            residual_jobs: frozen_story_diagnostic_ms(&stats, "aot.runtime.residual.actual_jobs"),
            jacobian_jobs: frozen_story_diagnostic_ms(
                &stats,
                "aot.runtime.sparse_jacobian.actual_jobs",
            ),
            iterations: stats.counters["number of iterations"],
            linear_solves: stats.counters["number of solving linear systems"],
            jacobian_rebuilds: stats.counters["number of jacobians recalculations"],
            selected_backend: frozen_story_diagnostic_string(&stats, "generated.selected_backend"),
            build_policy: frozen_story_diagnostic_string(&stats, "aot.build_policy"),
        };
        (row, solution, solver.generated_backend_config().clone())
    }

    fn frozen_polynomial_two_point_solver(n_steps: usize, config: GeneratedBackendConfig) -> NRBVP {
        // Non-combustion nonlinear BVP with exact solution y = 1 + x^2:
        // y' = z,
        // z' = 2 + 0.1 * (y - (1 + x^2))^2.
        //
        // Frozen currently requires a boundary-condition key for every state
        // variable, so we use y(-1) and z(1), both taken from the exact profile.
        let values = vec!["y".to_string(), "z".to_string()];
        let t0 = -1.0;
        let t_end = 1.0;
        let h = (t_end - t0) / n_steps as f64;
        let mut guess = vec![0.0; values.len() * n_steps];
        for i in 0..n_steps {
            let x = t0 + i as f64 * h;
            let y = 1.0 + x * x;
            let z = 2.0 * x;
            guess[i * values.len()] = y;
            guess[i * values.len() + 1] = z;
        }

        let y_left = 2.0;
        let z_right = 2.0;
        let options = FrozenSolverOptions::banded_frozen()
            .with_generated_backend_config(config)
            .with_tolerance(1e-6)
            .with_max_iterations(40);
        let mut solver = NRBVP::new_with_options(
            vec![
                Expr::parse_expression("z"),
                Expr::parse_expression("2.0 + 0.1*(y - (1.0 + x*x))^2"),
            ],
            DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(guess).as_slice()),
            values,
            "x".to_string(),
            HashMap::from([
                ("y".to_string(), vec![(0usize, y_left)]),
                ("z".to_string(), vec![(1usize, z_right)]),
            ]),
            t0,
            t_end,
            n_steps,
            options,
        );
        solver.dont_save_log(true);
        solver
    }

    fn run_frozen_polynomial_story_row(
        n_steps: usize,
        source: &'static str,
        variant: &'static str,
        config: GeneratedBackendConfig,
        baseline: Option<&DMatrix<f64>>,
    ) -> (
        FrozenCombustionStoryRow,
        DMatrix<f64>,
        GeneratedBackendConfig,
    ) {
        let begin = Instant::now();
        let mut solver = frozen_polynomial_two_point_solver(n_steps, config);
        let solve_result = solver.try_solve().unwrap_or_else(|err| {
            panic!("{source}/{variant} Frozen nonlinear polynomial solve failed: {err:?}")
        });
        assert!(
            solve_result.is_some(),
            "{source}/{variant} Frozen nonlinear polynomial solve did not converge within the configured iteration/tolerance budget"
        );
        collect_frozen_story_row(begin, solver, source, variant, baseline)
    }

    fn frozen_story_mean_std(values: &[f64]) -> (f64, f64) {
        let finite = values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        if finite.is_empty() {
            return (f64::NAN, f64::NAN);
        }
        let mean = finite.iter().sum::<f64>() / finite.len() as f64;
        let variance = finite
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f64>()
            / finite.len() as f64;
        (mean, variance.sqrt())
    }

    fn print_frozen_combustion_story(title: &str, rows: &[FrozenCombustionStoryRow]) {
        println!("[BVP Frozen story] {title}: correctness/backend selection");
        println!("source   | variant    | selected_backend | build_policy    | solve_diff");
        println!("{}", "-".repeat(82));
        for row in rows {
            println!(
                "{:<8} | {:<10} | {:<16} | {:<15} | {:.6e}",
                row.source, row.variant, row.selected_backend, row.build_policy, row.solution_diff
            );
        }
        println!();
        println!("[BVP Frozen story] {title}: wall-clock and Newton stages; milliseconds");
        println!(
            "source   | variant    | total_ms | symbolic_ms | linear_ms | jac_ms | fun_ms | iters | linsys | jac_re"
        );
        println!("{}", "-".repeat(118));
        for row in rows {
            println!(
                "{:<8} | {:<10} | {:>8.3} | {:>11.3} | {:>9.3} | {:>6.3} | {:>6.3} | {:>5} | {:>6} | {:>6}",
                row.source,
                row.variant,
                row.total_ms,
                row.symbolic_ms,
                row.linear_ms,
                row.jacobian_ms,
                row.residual_ms,
                row.iterations,
                row.linear_solves,
                row.jacobian_rebuilds,
            );
        }
        println!();
        println!(
            "[BVP Frozen story] {title}: generated handoff and compiled callback stages; milliseconds"
        );
        println!(
            "source   | variant    | initial_generate | initial_sym_jac | rebind_ms | compile_link | res_jobs | jac_jobs"
        );
        println!("{}", "-".repeat(120));
        for row in rows {
            println!(
                "{:<8} | {:<10} | {:>16.3} | {:>15.3} | {:>9.3} | {:>12.3} | {:>8.3} | {:>8.3}",
                row.source,
                row.variant,
                row.initial_generate_ms,
                row.initial_symbolic_jacobian_ms,
                row.post_build_rebind_ms,
                row.compile_link_ms,
                row.residual_jobs,
                row.jacobian_jobs,
            );
        }
        println!();
        println!("[BVP Frozen story] {title}: repeated-run summary; milliseconds");
        println!(
            "source   | variant    | total_ms mean+/-std | symbolic_ms mean+/-std | linear_ms mean+/-std | max_solution_diff"
        );
        println!("{}", "-".repeat(126));
        let mut identities = rows
            .iter()
            .map(|row| (row.source, row.variant))
            .collect::<Vec<_>>();
        identities.sort_unstable();
        identities.dedup();
        for (source, variant) in identities {
            let selected = rows
                .iter()
                .filter(|row| row.source == source && row.variant == variant)
                .collect::<Vec<_>>();
            let (total_mean, total_std) =
                frozen_story_mean_std(&selected.iter().map(|row| row.total_ms).collect::<Vec<_>>());
            let (symbolic_mean, symbolic_std) = frozen_story_mean_std(
                &selected
                    .iter()
                    .map(|row| row.symbolic_ms)
                    .collect::<Vec<_>>(),
            );
            let (linear_mean, linear_std) = frozen_story_mean_std(
                &selected.iter().map(|row| row.linear_ms).collect::<Vec<_>>(),
            );
            let max_diff = selected
                .iter()
                .map(|row| row.solution_diff)
                .fold(0.0_f64, f64::max);
            println!(
                "{:<8} | {:<10} | {:>9.3} +/- {:<9.3} | {:>12.3} +/- {:<9.3} | {:>10.3} +/- {:<9.3} | {:.6e}",
                source,
                variant,
                total_mean,
                total_std,
                symbolic_mean,
                symbolic_std,
                linear_mean,
                linear_std,
                max_diff,
            );
        }
    }

    fn frozen_tcc_config(
        matrix: &'static str,
        policy: AotBuildPolicy,
        chunking: AotChunkingPolicy,
        execution: AotExecutionPolicy,
    ) -> GeneratedBackendConfig {
        let config = match matrix {
            "Sparse" => GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc(),
            "Banded" => GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc(),
            _ => panic!("unsupported Frozen tcc story matrix route: {matrix}"),
        };
        config
            .with_aot_codegen_backend(AotCodegenBackend::C)
            .with_aot_c_compiler("tcc")
            .with_aot_compile_dev_fastest()
            .with_aot_chunking_policy(chunking)
            .with_aot_execution_policy(execution)
            .with_aot_build_policy(policy)
    }

    fn frozen_whole_chunking() -> AotChunkingPolicy {
        AotChunkingPolicy::with_parts(
            Some(ResidualChunkingStrategy::Whole),
            Some(SparseChunkingStrategy::Whole),
        )
    }

    fn frozen_chunk4_execution() -> (AotChunkingPolicy, AotExecutionPolicy) {
        (
            AotChunkingPolicy::with_parts(
                Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 4 }),
                Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 4 }),
            ),
            AotExecutionPolicy::Parallel(ParallelExecutorConfig {
                jobs_per_worker: 1,
                max_residual_jobs: Some(4),
                max_sparse_jobs: Some(4),
                fallback_policy: ParallelFallbackPolicy::Never,
            }),
        )
    }

    #[test]
    #[ignore = "heavy Frozen combustion-1000 Banded AtomView Lambdify vs tcc whole/chunk4 end-to-end story; run in release with --nocapture"]
    fn frozen_combustion_1000_banded_atomview_lambdify_vs_tcc_aot_end_to_end_story() {
        let n_steps = 1_000;
        let repetitions = 2;
        let mut rows = Vec::new();
        let (chunk4, parallel) = frozen_chunk4_execution();

        for _ in 0..repetitions {
            let (baseline_row, baseline, _) = run_frozen_combustion_story_row(
                n_steps,
                "Banded",
                "Lambdify",
                "AtomView",
                GeneratedBackendConfig::banded_lambdify_defaults(),
                None,
            );
            rows.push(baseline_row);

            let (whole_row, _, _) = run_frozen_combustion_story_row(
                n_steps,
                "Banded",
                "AOT",
                "tcc/whole",
                frozen_tcc_config(
                    "Banded",
                    AotBuildPolicy::RebuildAlways {
                        profile: AotBuildProfile::Release,
                    },
                    frozen_whole_chunking(),
                    AotExecutionPolicy::SequentialOnly,
                ),
                Some(&baseline),
            );
            rows.push(whole_row);

            let (chunked_row, _, _) = run_frozen_combustion_story_row(
                n_steps,
                "Banded",
                "AOT",
                "tcc/chunk4",
                frozen_tcc_config(
                    "Banded",
                    AotBuildPolicy::RebuildAlways {
                        profile: AotBuildProfile::Release,
                    },
                    chunk4,
                    parallel.clone(),
                ),
                Some(&baseline),
            );
            rows.push(chunked_row);
        }

        print_frozen_combustion_story(
            "combustion-1000 Banded AtomView Lambdify vs tcc AOT cold routes",
            &rows,
        );
        assert!(
            rows.iter().all(|row| row.solution_diff <= 1e-5),
            "Frozen AOT variants must remain equivalent to the Lambdify baseline"
        );
        assert!(
            rows.iter()
                .filter(|row| row.source == "AOT")
                .all(|row| row.selected_backend == "AotCompiled"),
            "Frozen AOT cold routes must execute freshly compiled callbacks"
        );
        assert!(
            rows.iter()
                .filter(|row| row.variant == "tcc/chunk4")
                .all(|row| row.residual_jobs > 1.0 && row.jacobian_jobs > 1.0),
            "Frozen chunk4 route must expose real callback-level parallel execution"
        );
    }

    #[test]
    #[ignore = "heavy Frozen combustion-1000 AOT artifact lifecycle story: BuildIfMissing followed by strict RequirePrebuilt reuse"]
    fn frozen_combustion_1000_banded_atomview_tcc_build_then_require_prebuilt_story() {
        let n_steps = 1_000;
        let (baseline_row, baseline, _) = run_frozen_combustion_story_row(
            n_steps,
            "Banded",
            "Lambdify",
            "AtomView",
            GeneratedBackendConfig::banded_lambdify_defaults(),
            None,
        );
        let (built_row, _, built_config) = run_frozen_combustion_story_row(
            n_steps,
            "Banded",
            "AOT",
            "build",
            frozen_tcc_config(
                "Banded",
                AotBuildPolicy::BuildIfMissing {
                    profile: AotBuildProfile::Release,
                },
                frozen_whole_chunking(),
                AotExecutionPolicy::SequentialOnly,
            ),
            Some(&baseline),
        );
        assert!(
            built_config.resolver.is_some(),
            "BuildIfMissing must leave a resolver snapshot for strict reuse"
        );
        let strict_config = built_config.with_aot_build_policy(AotBuildPolicy::RequirePrebuilt);
        let mut rows = vec![baseline_row, built_row];
        for _ in 0..3 {
            let (prebuilt_row, _, _) = run_frozen_combustion_story_row(
                n_steps,
                "Banded",
                "AOT",
                "prebuilt",
                strict_config.clone(),
                Some(&baseline),
            );
            rows.push(prebuilt_row);
        }

        print_frozen_combustion_story(
            "combustion-1000 Banded AtomView tcc BuildIfMissing -> RequirePrebuilt lifecycle",
            &rows,
        );
        assert!(
            rows.iter().all(|row| row.solution_diff <= 1e-5),
            "Frozen lifecycle rows must remain equivalent to Lambdify"
        );
        assert!(
            rows.iter()
                .filter(|row| row.source == "AOT")
                .all(|row| row.selected_backend == "AotCompiled"),
            "both built and strict prebuilt rows must execute compiled callbacks"
        );
        assert!(
            rows.iter()
                .filter(|row| row.variant == "prebuilt")
                .all(|row| row.build_policy == "RequirePrebuilt"),
            "warm rows must be strict RequirePrebuilt executions, not fallback builds"
        );
    }

    #[test]
    #[ignore = "heavy Frozen combustion-1000 Sparse AtomView tcc artifact lifecycle: BuildIfMissing followed by strict RequirePrebuilt reuse"]
    fn frozen_combustion_1000_sparse_atomview_tcc_build_then_require_prebuilt_story() {
        let n_steps = 1_000;
        let sparse_lambdify = GeneratedBackendConfig::sparse_defaults()
            .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly));
        let (baseline_row, baseline, _) = run_frozen_combustion_story_row(
            n_steps,
            "Sparse",
            "Lambdify",
            "AtomView",
            sparse_lambdify,
            None,
        );
        let (built_row, _, built_config) = run_frozen_combustion_story_row(
            n_steps,
            "Sparse",
            "AOT",
            "build",
            frozen_tcc_config(
                "Sparse",
                AotBuildPolicy::BuildIfMissing {
                    profile: AotBuildProfile::Release,
                },
                frozen_whole_chunking(),
                AotExecutionPolicy::SequentialOnly,
            ),
            Some(&baseline),
        );
        assert!(
            built_config.resolver.is_some(),
            "Sparse BuildIfMissing must leave a resolver snapshot for strict reuse"
        );
        let strict_config = built_config.with_aot_build_policy(AotBuildPolicy::RequirePrebuilt);
        let mut rows = vec![baseline_row, built_row];
        for _ in 0..3 {
            let (prebuilt_row, _, _) = run_frozen_combustion_story_row(
                n_steps,
                "Sparse",
                "AOT",
                "prebuilt",
                strict_config.clone(),
                Some(&baseline),
            );
            rows.push(prebuilt_row);
        }

        print_frozen_combustion_story(
            "combustion-1000 Sparse AtomView tcc BuildIfMissing -> RequirePrebuilt lifecycle",
            &rows,
        );
        assert!(
            rows.iter().all(|row| row.solution_diff <= 1e-5),
            "Frozen Sparse lifecycle rows must remain equivalent to Lambdify"
        );
        assert!(
            rows.iter()
                .filter(|row| row.source == "AOT")
                .all(|row| row.selected_backend == "AotCompiled"),
            "Frozen Sparse build and strict prebuilt rows must execute compiled callbacks"
        );
        assert!(
            rows.iter()
                .filter(|row| row.variant == "prebuilt")
                .all(|row| {
                    row.build_policy == "RequirePrebuilt" && row.compile_link_ms.is_nan()
                }),
            "Frozen Sparse prebuilt rows must neither fall back nor compile again"
        );
    }

    #[test]
    #[ignore = "Frozen non-combustion nonlinear polynomial BVP: Banded AtomView Lambdify vs tcc BuildIfMissing -> RequirePrebuilt"]
    fn frozen_polynomial_banded_atomview_tcc_build_then_require_prebuilt_story() {
        let n_steps = 80;
        let (baseline_row, baseline, _) = run_frozen_polynomial_story_row(
            n_steps,
            "Lambdify",
            "AtomView",
            GeneratedBackendConfig::banded_lambdify_defaults(),
            None,
        );
        let (built_row, _, built_config) = run_frozen_polynomial_story_row(
            n_steps,
            "AOT",
            "build",
            frozen_tcc_config(
                "Banded",
                AotBuildPolicy::BuildIfMissing {
                    profile: AotBuildProfile::Release,
                },
                frozen_whole_chunking(),
                AotExecutionPolicy::SequentialOnly,
            ),
            Some(&baseline),
        );
        assert!(
            built_config.resolver.is_some(),
            "Polynomial BuildIfMissing must leave a resolver snapshot for strict reuse"
        );
        let strict_config = built_config.with_aot_build_policy(AotBuildPolicy::RequirePrebuilt);
        let mut rows = vec![baseline_row, built_row];
        for _ in 0..2 {
            let (prebuilt_row, _, _) = run_frozen_polynomial_story_row(
                n_steps,
                "AOT",
                "prebuilt",
                strict_config.clone(),
                Some(&baseline),
            );
            rows.push(prebuilt_row);
        }

        print_frozen_combustion_story(
            "nonlinear polynomial BVP Banded AtomView tcc BuildIfMissing -> RequirePrebuilt lifecycle",
            &rows,
        );
        assert!(
            rows.iter().all(|row| row.solution_diff <= 1e-6),
            "Frozen nonlinear polynomial lifecycle rows must remain equivalent to Lambdify"
        );
        assert!(
            rows.iter()
                .filter(|row| row.source == "AOT")
                .all(|row| row.selected_backend == "AotCompiled"),
            "Frozen nonlinear polynomial build and strict prebuilt rows must execute compiled callbacks"
        );
        assert!(
            rows.iter()
                .filter(|row| row.variant == "prebuilt")
                .all(|row| {
                    row.build_policy == "RequirePrebuilt" && row.compile_link_ms.is_nan()
                }),
            "Frozen nonlinear polynomial prebuilt rows must neither fall back nor compile again"
        );
    }

    #[test]
    fn generated_backend_surface_builder_methods_update_solver_config() {
        let solver = sparse_surface_test_solver()
            .with_backend_policy_override(Some(BackendSelectionPolicy::PreferAotThenLambdify))
            .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
            .with_aot_build_policy(AotBuildPolicy::RequirePrebuilt)
            .with_aot_chunking_policy(AotChunkingPolicy::with_parts(
                Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 2 }),
                Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 3 }),
            ))
            .with_atom_optimization_profile(AtomOptimizationProfile::NoCse);

        assert_eq!(
            solver.backend_policy_override(),
            Some(BackendSelectionPolicy::PreferAotThenLambdify)
        );
        assert_eq!(
            solver.aot_execution_policy(),
            &AotExecutionPolicy::SequentialOnly
        );
        assert_eq!(solver.aot_build_policy(), AotBuildPolicy::RequirePrebuilt);
        assert_eq!(
            solver.aot_chunking_policy(),
            AotChunkingPolicy::with_parts(
                Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 2 }),
                Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 3 }),
            )
        );
        assert_eq!(
            solver.atom_optimization_profile(),
            AtomOptimizationProfile::NoCse
        );
    }

    #[test]
    fn symbolic_assembly_backend_is_exposed_on_solver_surface() {
        let mut solver = sparse_surface_test_solver()
            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView);

        assert_eq!(
            solver.symbolic_assembly_backend(),
            BvpSymbolicAssemblyBackend::AtomView
        );
        assert_eq!(
            solver.generated_backend_config().symbolic_assembly_backend,
            BvpSymbolicAssemblyBackend::AtomView
        );

        solver.set_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy);
        assert_eq!(
            solver.symbolic_assembly_backend(),
            BvpSymbolicAssemblyBackend::ExprLegacy
        );
        assert_eq!(
            solver.generated_backend_config().symbolic_assembly_backend,
            BvpSymbolicAssemblyBackend::ExprLegacy
        );
    }

    #[test]
    fn banded_frozen_lambdify_mode_sets_banded_matrix_and_lambdify_policy() {
        let options = FrozenSolverOptions::banded_frozen().with_banded_lambdify();

        assert_eq!(options.method, "Sparse");
        assert_eq!(
            options.generated_backend_config.symbolic_assembly_backend,
            BvpSymbolicAssemblyBackend::AtomView
        );
        assert_eq!(
            options.generated_backend_config.backend_policy_override,
            Some(BackendSelectionPolicy::LambdifyOnly)
        );
        assert_eq!(
            options.generated_backend_config.matrix_backend_override,
            Some(MatrixBackend::Banded)
        );
        assert_eq!(
            options.generated_backend_config.symbolic_assembly_backend,
            BvpSymbolicAssemblyBackend::AtomView
        );
        assert_eq!(
            options.generated_backend_config.aot_build_policy,
            AotBuildPolicy::UseIfAvailable
        );
    }

    #[test]
    fn banded_frozen_generated_backend_mode_build_if_missing_sets_release_aot_policy() {
        let options = FrozenSolverOptions::banded_frozen()
            .with_banded_generated_backend_mode(BandedGeneratedBackendMode::BuildIfMissingRelease);

        assert_eq!(
            options.generated_backend_config.backend_policy_override,
            Some(BackendSelectionPolicy::PreferAotThenLambdify)
        );
        assert_eq!(
            options.generated_backend_config.matrix_backend_override,
            Some(MatrixBackend::Banded)
        );
        assert_eq!(
            options.generated_backend_config.aot_build_policy,
            AotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Release
            }
        );
    }

    #[test]
    fn frozen_sparse_lambdify_linear_bvp_solves_against_exact_profile() {
        let options = FrozenSolverOptions::sparse_frozen()
            .with_generated_backend_config(
                GeneratedBackendConfig::sparse_defaults()
                    .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly)),
            )
            .with_strategy_params(Some(HashMap::from([("Frozen_naive".to_string(), None)])))
            .with_tolerance(1e-8)
            .with_max_iterations(20);
        let mut solver = frozen_linear_solver(24, options);
        solver.dont_save_log(true);

        solver
            .try_solve()
            .expect("sparse frozen lambdify linear BVP should solve");

        assert_frozen_linear_solution_quality(&solver, 24, 1e-10, 1e-9);
        assert!(
            solver.jac.is_some(),
            "sparse frozen route should prepare a Jacobian"
        );
        assert!(
            !solver.variable_string.is_empty(),
            "sparse frozen route should prepare reduced variable metadata"
        );
    }

    #[test]
    fn frozen_banded_default_atomview_lambdify_linear_bvp_solves_against_exact_profile() {
        let options = FrozenSolverOptions::banded_frozen()
            .with_banded_lambdify()
            .with_strategy_params(Some(HashMap::from([("Frozen_naive".to_string(), None)])))
            .with_tolerance(1e-8)
            .with_max_iterations(20);
        assert_eq!(
            options.generated_backend_config.symbolic_assembly_backend,
            BvpSymbolicAssemblyBackend::AtomView
        );
        let mut solver = frozen_linear_solver(24, options);
        solver.dont_save_log(true);

        solver
            .try_solve()
            .expect("default banded frozen AtomView lambdify linear BVP should solve");

        assert_frozen_linear_solution_quality(&solver, 24, 1e-7, 1e-6);
        assert!(
            solver.jac.is_some(),
            "banded frozen route should prepare a Jacobian"
        );
        assert!(
            solver.bandwidth.0 + solver.bandwidth.1 > 0,
            "banded frozen route should expose non-empty bandwidth metadata"
        );
        let stats = solver.get_statistics();
        assert_eq!(
            stats.diagnostics.get("generated.selected_backend"),
            Some(&"Lambdify".to_string()),
            "Frozen statistics must report the backend that supplied callbacks"
        );
        assert_eq!(
            stats.diagnostics.get("generated.symbolic_assembly_backend"),
            Some(&"AtomView".to_string())
        );
        assert!(
            stats
                .diagnostics
                .contains_key("generated.handoff.initial_generate_wall_ms"),
            "Frozen must preserve symbolic handoff timing diagnostics"
        );
        assert!(
            stats.counters["number of iterations"] > 0
                && stats.counters["number of jacobians recalculations"] > 0
                && stats.counters["number of solving linear systems"] > 0,
            "Frozen end-to-end solve must expose its Newton work counters: {:?}",
            stats.counters
        );
        assert!(
            stats
                .timers
                .keys()
                .any(|key| key.starts_with("Symbolic Operations")),
            "Frozen end-to-end solve must expose backend preparation timing"
        );
    }

    #[test]
    fn try_eq_generate_surfaces_missing_prebuilt_aot_as_typed_error() {
        let mut solver =
            sparse_surface_test_solver_with_naive_strategy().with_sparse_aot_require_prebuilt();

        let err = solver
            .try_eq_generate()
            .expect_err("try_eq_generate should return a typed AOT availability error");

        assert!(matches!(
            err,
            BvpBackendIntegrationError::CompiledAotRequiredButUnavailable { .. }
        ));
    }

    #[test]
    fn try_solve_surfaces_missing_prebuilt_aot_as_typed_error() {
        let mut solver =
            sparse_surface_test_solver_with_naive_strategy().with_sparse_aot_require_prebuilt();
        solver.dont_save_log(true);

        let err = solver
            .try_solve()
            .expect_err("try_solve should return a typed AOT availability error");

        assert!(matches!(
            err,
            BvpBackendIntegrationError::CompiledAotRequiredButUnavailable { .. }
        ));
    }

    #[test]
    fn sparse_generated_backend_presets_are_exposed_on_solver_surface() {
        let solver = sparse_surface_test_solver().with_sparse_aot_build_if_missing_release();

        assert_eq!(
            solver.backend_policy_override(),
            Some(BackendSelectionPolicy::PreferAotThenLambdify)
        );
        assert_eq!(
            solver.aot_build_policy(),
            AotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Release
            }
        );
    }

    #[test]
    fn sparse_generated_backend_mode_is_exposed_on_solver_surface() {
        let mut solver = sparse_surface_test_solver()
            .with_sparse_generated_backend_mode(SparseGeneratedBackendMode::RequirePrebuilt);

        assert_eq!(
            solver.backend_policy_override(),
            Some(BackendSelectionPolicy::PreferAotThenLambdify)
        );
        assert_eq!(solver.aot_build_policy(), AotBuildPolicy::RequirePrebuilt);

        solver.set_sparse_generated_backend_mode(SparseGeneratedBackendMode::BuildIfMissingRelease);
        assert_eq!(
            solver.aot_build_policy(),
            AotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Release
            }
        );
    }

    #[test]
    fn sparse_frozen_options_preset_sets_production_defaults() {
        let options = FrozenSolverOptions::sparse_frozen();

        assert_eq!(options.scheme, "forward");
        assert_eq!(options.strategy, "Frozen");
        assert_eq!(options.method, "Sparse");
        assert_eq!(
            options.generated_backend_config.symbolic_assembly_backend,
            BvpSymbolicAssemblyBackend::AtomView
        );
        assert_eq!(
            options.generated_backend_config.backend_policy_override,
            Some(BackendSelectionPolicy::PreferAotThenLambdify)
        );
        assert_eq!(
            options.generated_backend_config.aot_build_policy,
            AotBuildPolicy::UseIfAvailable
        );
        assert_eq!(
            options.generated_backend_config.aot_execution_policy,
            AotExecutionPolicy::Auto
        );
        assert_eq!(
            options.generated_backend_config.aot_chunking_policy,
            AotChunkingPolicy::default()
        );
    }

    #[test]
    fn banded_frozen_options_preset_uses_auto_aot_chunking_defaults() {
        let options = FrozenSolverOptions::banded_frozen();

        assert_eq!(
            options.generated_backend_config.matrix_backend_override,
            Some(MatrixBackend::Banded)
        );
        assert_eq!(
            options.generated_backend_config.aot_execution_policy,
            AotExecutionPolicy::Auto
        );
        assert_eq!(
            options.generated_backend_config.aot_chunking_policy,
            AotChunkingPolicy::default()
        );
    }

    #[test]
    fn frozen_options_scheme_builder_methods_set_legacy_scheme_flag() {
        let options = FrozenSolverOptions::sparse_frozen().trapezoid_derivative();
        assert_eq!(options.scheme, "trapezoid");

        let options = options.forward_derivative();
        assert_eq!(options.scheme, "forward");

        let options = options.with_scheme(BvpDerivativeScheme::Trapezoid);
        assert_eq!(options.scheme, "trapezoid");

        let options = options.with_scheme_name("custom-experimental");
        assert_eq!(options.scheme, "custom-experimental");
    }

    #[test]
    fn frozen_solver_scheme_builder_methods_feed_generated_request() {
        let solver = sparse_surface_test_solver().trapezoid_derivative();
        assert_eq!(solver.scheme, "trapezoid");
        assert_eq!(solver.build_solver_request().scheme, "trapezoid");

        let solver = solver.forward_derivative();
        assert_eq!(solver.scheme, "forward");
        assert_eq!(solver.build_solver_request().scheme, "forward");

        let solver = solver.with_scheme(BvpDerivativeScheme::Trapezoid);
        assert_eq!(solver.scheme, "trapezoid");
        assert_eq!(solver.build_solver_request().scheme, "trapezoid");

        let solver = solver.with_scheme_name("custom-experimental");
        assert_eq!(solver.scheme, "custom-experimental");
        assert_eq!(solver.build_solver_request().scheme, "custom-experimental");
    }

    #[test]
    fn dense_frozen_options_preset_sets_dense_defaults() {
        let options = FrozenSolverOptions::dense_frozen();

        assert_eq!(options.strategy, "Frozen");
        assert_eq!(options.method, "Dense");
    }

    #[test]
    fn dense_naive_options_preset_sets_dense_defaults() {
        let options = FrozenSolverOptions::dense_naive();

        assert_eq!(options.strategy, "Naive");
        assert!(options.strategy_params.is_none());
        assert_eq!(options.method, "Dense");
    }

    #[test]
    fn constructor_style_sparse_generated_backend_mode_sets_solver_config() {
        let solver = NRBVP::new_with_sparse_generated_backend_mode(
            vec![Expr::parse_expression("z"), Expr::parse_expression("-y")],
            DMatrix::from_element(2, 4, 0.1),
            vec!["y".to_string(), "z".to_string()],
            "x".to_string(),
            HashMap::from([
                ("y".to_string(), vec![(0, 0.0)]),
                ("z".to_string(), vec![(0, 1.0)]),
            ]),
            0.0,
            1.0,
            4,
            "Frozen".to_string(),
            None,
            None,
            "Sparse".to_string(),
            1e-6,
            10,
            SparseGeneratedBackendMode::RequirePrebuilt,
        );

        assert_eq!(
            solver.backend_policy_override(),
            Some(BackendSelectionPolicy::PreferAotThenLambdify)
        );
        assert_eq!(solver.aot_build_policy(), AotBuildPolicy::RequirePrebuilt);
    }

    #[test]
    fn options_style_solver_setup_sets_sparse_generated_backend_mode() {
        let options = FrozenSolverOptions::sparse_frozen().with_sparse_aot_require_prebuilt();

        let solver = NRBVP::new_with_options(
            vec![Expr::parse_expression("z"), Expr::parse_expression("-y")],
            DMatrix::from_element(2, 4, 0.1),
            vec!["y".to_string(), "z".to_string()],
            "x".to_string(),
            HashMap::from([
                ("y".to_string(), vec![(0, 0.0)]),
                ("z".to_string(), vec![(0, 1.0)]),
            ]),
            0.0,
            1.0,
            4,
            options,
        );

        assert_eq!(
            solver.backend_policy_override(),
            Some(BackendSelectionPolicy::PreferAotThenLambdify)
        );
        assert_eq!(solver.aot_build_policy(), AotBuildPolicy::RequirePrebuilt);
    }

    #[test]
    fn test_newton_raphson_solver() {
        // Define a simple equation: x^2 - 4 = 0
        let eq1 = Expr::parse_expression("y-z");
        let eq2 = Expr::parse_expression("-z");
        let eq_system = vec![eq1, eq2];

        let values = vec!["z".to_string(), "y".to_string()];
        let arg = "x".to_string();
        let tolerance = 1e-2;
        let max_iterations = 100;

        let t0 = 0.0;
        let t_end = 1.0;
        let n_steps = 100;
        let ones = vec![1.0; values.len() * n_steps];
        let initial_guess: DMatrix<f64> =
            DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(ones).as_slice());
        let mut BorderConditions = HashMap::new();
        BorderConditions.insert("z".to_string(), vec![(0usize, 1.0f64)]);
        BorderConditions.insert("y".to_string(), vec![(1usize, 1.0f64)]);
        assert_eq!(&eq_system.len(), &2);
        let options = FrozenSolverOptions::dense_naive()
            .with_tolerance(tolerance)
            .with_max_iterations(max_iterations);
        let mut nr = NRBVP::new_with_options(
            eq_system,
            initial_guess,
            values,
            arg,
            BorderConditions,
            t0,
            t_end,
            n_steps,
            options,
        );
        nr.try_eq_generate()
            .expect("dense frozen solver should generate through the fallible API");

        assert_eq!(nr.eq_system.len(), 2);
        nr.dont_save_log(true);
        // Solve the equation at t=0 with initial guess y=[2.0]
        //    nr.set_new_step(0.0, DVector::from_element(1, 2.0), DVector::from_element(1, 2.0));
        let _solution = nr
            .try_solve()
            .expect("dense frozen solver should solve through the fallible API")
            .unwrap();
    }

    #[test]
    fn sparse_eq_generate_uses_bundle_handoff_without_breaking_metadata() {
        let mut solver = sparse_surface_test_solver_with_naive_strategy();

        solver
            .try_eq_generate()
            .expect("sparse frozen handoff should generate through the fallible API");

        assert!(solver.jac.is_some());
        assert!(!solver.variable_string.is_empty());
        assert!(solver.bandwidth.0 + solver.bandwidth.1 > 0);
        let _ = &solver.fun;
    }

    #[test]
    fn numeric_only_is_rejected_for_frozen_solver_instead_of_lambdify_fallback() {
        let mut solver = sparse_surface_test_solver_with_naive_strategy();
        solver.set_backend_policy_override(Some(BackendSelectionPolicy::NumericOnly));

        let err = solver
            .try_eq_generate()
            .expect_err("Frozen NumericOnly must not silently fall back to symbolic lambdify");
        assert!(matches!(
            err,
            BvpBackendIntegrationError::PipelinePanicked(message)
                if message.contains("not available for the frozen BVP solver")
        ));
    }

    #[test]
    fn build_solver_request_carries_optional_aot_resolver() {
        let mut solver = sparse_surface_test_solver_with_naive_strategy();
        solver.set_aot_resolver(Some(AotResolver::new(AotRegistry::new())));

        let request = solver.build_solver_request();
        assert!(request.resolver.is_some());
    }

    #[test]
    fn build_solver_request_carries_parameter_names_and_values() {
        let mut solver = sparse_surface_test_solver_with_naive_strategy();
        solver.set_params(Some(&["alpha", "beta"]));
        solver.set_param_values(Some(vec![1.5, -0.25]));

        let request = solver.build_solver_request();
        assert_eq!(
            request.param_names,
            Some(vec!["alpha".to_string(), "beta".to_string()])
        );
        assert_eq!(request.param_values, Some(vec![1.5, -0.25]));
    }

    #[test]
    #[should_panic(expected = "param_values length must match param_names length")]
    fn solver_surface_rejects_parameter_value_length_mismatch() {
        let mut solver = sparse_surface_test_solver_with_naive_strategy();
        solver.set_params(Some(&["alpha", "beta"]));
        solver.set_param_values(Some(vec![1.5]));
    }

    #[test]
    fn build_solver_request_uses_backend_policy_override_when_present() {
        let mut solver = sparse_surface_test_solver_with_naive_strategy();
        solver.set_backend_policy_override(Some(BackendSelectionPolicy::NumericOnly));

        let request = solver.build_solver_request();
        assert_eq!(request.backend_policy, BackendSelectionPolicy::NumericOnly);
    }

    #[test]
    fn generated_backend_config_is_exposed_as_user_facing_solver_setting() {
        let mut solver = sparse_surface_test_solver_with_naive_strategy();

        let config = GeneratedBackendConfig::with_parts(
            Some(BackendSelectionPolicy::NumericOnly),
            Some(AotResolver::new(AotRegistry::new())),
        );
        solver.set_generated_backend_config(config);

        let request = solver.build_solver_request();
        assert_eq!(request.backend_policy, BackendSelectionPolicy::NumericOnly);
        assert!(request.resolver.is_some());
        assert_eq!(
            solver.generated_backend_config().backend_policy_override,
            Some(BackendSelectionPolicy::NumericOnly)
        );
    }

    #[test]
    fn generated_backend_config_can_be_applied_during_solver_construction() {
        let solver = sparse_surface_test_solver_with_naive_strategy()
            .with_generated_backend_config(GeneratedBackendConfig::with_parts(
                Some(BackendSelectionPolicy::NumericOnly),
                Some(AotResolver::new(AotRegistry::new())),
            ));

        assert_eq!(
            solver.generated_backend_config().backend_policy_override,
            Some(BackendSelectionPolicy::NumericOnly)
        );
        assert!(solver.generated_backend_config().resolver.is_some());
    }

    #[test]
    fn cleanup_registered_aot_artifacts_is_safe_without_registered_artifacts() {
        let mut solver = sparse_surface_test_solver_with_naive_strategy();
        assert_eq!(solver.cleanup_registered_aot_artifacts().unwrap(), 0);

        solver.set_aot_resolver(Some(AotResolver::new(AotRegistry::new())));
        assert_eq!(solver.cleanup_registered_aot_artifacts().unwrap(), 0);
    }

    #[test]
    fn build_solver_request_carries_surface_aot_policies() {
        let config = GeneratedBackendConfig::new()
            .with_backend_policy_override(Some(BackendSelectionPolicy::PreferAotThenLambdify))
            .with_resolver(Some(AotResolver::new(AotRegistry::new())))
            .with_aot_execution_policy(AotExecutionPolicy::Parallel(ParallelExecutorConfig {
                jobs_per_worker: 2,
                max_residual_jobs: Some(4),
                max_sparse_jobs: Some(2),
                fallback_policy: ParallelFallbackPolicy::Never,
            }))
            .with_aot_build_policy(AotBuildPolicy::RequirePrebuilt)
            .with_aot_chunking_policy(AotChunkingPolicy::with_parts(
                Some(ResidualChunkingStrategy::ByOutputCount {
                    max_outputs_per_chunk: 6,
                }),
                Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 3 }),
            ))
            .with_atom_optimization_profile(AtomOptimizationProfile::NoCse);

        let solver =
            sparse_surface_test_solver_with_naive_strategy().with_generated_backend_config(config);

        let request = solver.build_solver_request();
        assert_eq!(request.aot_build_policy, AotBuildPolicy::RequirePrebuilt);
        assert_eq!(
            request.aot_chunking_policy.residual,
            Some(ResidualChunkingStrategy::ByOutputCount {
                max_outputs_per_chunk: 6
            })
        );
        assert_eq!(
            request.aot_chunking_policy.sparse_jacobian,
            Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 3 })
        );
        assert_eq!(
            request.atom_optimization_profile,
            AtomOptimizationProfile::NoCse
        );
        match request.aot_execution_policy {
            AotExecutionPolicy::Parallel(inner) => {
                assert_eq!(inner.jobs_per_worker, 2);
                assert_eq!(inner.max_residual_jobs, Some(4));
                assert_eq!(inner.max_sparse_jobs, Some(2));
            }
            other => std::panic!("expected parallel execution policy, got {other:?}"),
        }
        let _ = AotBuildProfile::Debug;
    }

    #[test]
    fn eq_generate_build_if_missing_saves_compiled_resolver_for_next_request() {
        let mut solver = sparse_surface_test_solver_with_naive_strategy()
            .with_generated_backend_config(
                GeneratedBackendConfig::new()
                    .with_backend_policy_override(Some(
                        BackendSelectionPolicy::PreferAotThenLambdify,
                    ))
                    .with_aot_build_policy(AotBuildPolicy::BuildIfMissing {
                        profile: AotBuildProfile::Release,
                    }),
            );

        solver
            .try_eq_generate()
            .expect("build-if-missing path should generate through the fallible API");

        let saved_resolver = solver
            .generated_backend_config()
            .resolver
            .as_ref()
            .expect("first build-if-missing run should save updated resolver");
        assert!(
            !saved_resolver.registry().is_empty(),
            "first build-if-missing run should register at least one compiled artifact"
        );

        let updated_config = solver
            .generated_backend_config()
            .clone()
            .with_aot_build_policy(AotBuildPolicy::RequirePrebuilt);
        solver.set_generated_backend_config(updated_config);

        let next_request = solver.build_solver_request();
        assert!(next_request.resolver.is_some());

        let next_state = next_request.generate().expect(
            "next request should reuse the saved compiled resolver and generate successfully",
        );
        assert!(
            next_state.jac.is_some(),
            "successful generation through the saved resolver should still provide a Jacobian callback"
        );
    }

    #[test]
    fn second_eq_generate_reuses_saved_resolver_and_runs_linked_compiled_backend() {
        let values = vec!["z".to_string(), "y".to_string()];
        let n_steps = 8;
        let mut solver = sparse_surface_test_solver_with_naive_strategy()
            .with_backend_policy_override(Some(BackendSelectionPolicy::PreferAotThenLambdify))
            .with_aot_build_policy(AotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Release,
            });

        solver
            .try_eq_generate()
            .expect("first sparse generation should succeed through the fallible API");
        let saved_resolver = solver
            .generated_backend_config()
            .resolver
            .clone()
            .expect("first build-if-missing run should save updated resolver");

        let y = Col::from_fn(values.len() * n_steps, |index| 0.3 + index as f64 * 0.02);
        let baseline = solver.fun.call(0.0, &y).to_DVectorType();

        let problem_keys = saved_resolver.registry().problem_keys();
        assert_eq!(
            problem_keys.len(),
            1,
            "build-if-missing should register exactly one artifact for this isolated test"
        );
        let problem_key = problem_keys[0].clone();
        let resolved = saved_resolver.resolve_by_problem_key(&problem_key);
        assert!(
            resolved.is_compiled(),
            "saved resolver should see compiled artifact"
        );

        let baseline_values: Vec<f64> = baseline.iter().copied().collect();
        register_linked_sparse_backend(LinkedSparseAotBackend::new(
            problem_key.clone(),
            resolved.registered.manifest.io.residual_len,
            (
                resolved.registered.manifest.io.jacobian_rows,
                resolved.registered.manifest.io.jacobian_cols,
            ),
            resolved.registered.manifest.io.jacobian_nnz.unwrap_or(0),
            Arc::new(move |_args, out| {
                for (dst, src) in out.iter_mut().zip(baseline_values.iter()) {
                    *dst = *src + 55.0;
                }
            }),
            Arc::new(move |_args, out| {
                for (index, value) in out.iter_mut().enumerate() {
                    *value = 900.0 + index as f64;
                }
            }),
        ));

        solver.set_aot_build_policy(AotBuildPolicy::RequirePrebuilt);
        solver
            .try_eq_generate()
            .expect("second sparse generation should reuse resolver through the fallible API");
        let residual = solver.fun.call(0.0, &y).to_DVectorType();

        for (actual, expected) in residual.iter().zip(baseline.iter()) {
            assert!((actual - (expected + 55.0)).abs() < 1e-10);
        }

        unregister_linked_sparse_backend(&problem_key);
    }
}
