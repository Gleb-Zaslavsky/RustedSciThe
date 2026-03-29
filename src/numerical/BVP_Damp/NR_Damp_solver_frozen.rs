use crate::symbolic::symbolic_engine::Expr;
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;
use std::time::Instant;

use crate::Utils::logger::save_matrix_to_file;
use crate::Utils::plots::plots;
use crate::numerical::BVP_Damp::BVP_traits::{
    Fun, FunEnum, Jac, MatrixType, VectorType, Vectors_type_casting,
};
use crate::numerical::BVP_Damp::BVP_utils::*;
use crate::numerical::BVP_Damp::generated_solver_handoff::{
    AotBuildPolicy, AotChunkingPolicy, AotExecutionPolicy, ApplyFrozenGeneratedSolverState,
    BuildFrozenSolverRequest, FrozenGeneratedSolverState, FrozenSolverBuildRequest,
    GeneratedBackendConfig, SparseGeneratedBackendMode, try_generate_and_apply_frozen_solver_state,
};
use crate::symbolic::codegen_aot_resolution::AotResolver;
use crate::symbolic::codegen_backend_selection::BackendSelectionPolicy;
use crate::symbolic::symbolic_functions_BVP::BvpBackendIntegrationError;

use chrono::Local;

use log::info;

use simplelog::*;

use std::fs::File;

/// User-facing setup options for the frozen BVP solver.
#[derive(Clone)]
pub struct FrozenSolverOptions {
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

    /// Creates production-oriented sparse frozen solver options with standard defaults.
    ///
    /// This is the preferred starting point for most sparse/frozen BVP users.
    pub fn sparse_frozen() -> Self {
        Self::default().with_sparse_generated_backend_mode(SparseGeneratedBackendMode::Defaults)
    }

    /// Creates production-oriented dense frozen solver options with standard defaults.
    pub fn dense_frozen() -> Self {
        Self {
            method: "Dense".to_string(),
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

    /// Requires a prebuilt sparse generated backend.
    pub fn with_sparse_aot_require_prebuilt(self) -> Self {
        self.with_sparse_generated_backend_mode(SparseGeneratedBackendMode::RequirePrebuilt)
    }

    /// Builds a sparse release AOT backend on demand.
    pub fn with_sparse_aot_build_if_missing_release(self) -> Self {
        self.with_sparse_generated_backend_mode(SparseGeneratedBackendMode::BuildIfMissingRelease)
    }
}

impl Default for FrozenSolverOptions {
    fn default() -> Self {
        Self {
            strategy: "Frozen".to_string(),
            strategy_params: Some(HashMap::from([("Frozen_naive".to_string(), None)])),
            linear_sys_method: None,
            method: "Sparse".to_string(),
            tolerance: 1e-6,
            max_iterations: 25,
            generated_backend_config: GeneratedBackendConfig::default(),
        }
    }
}

pub struct NRBVP {
    pub eq_system: Vec<Expr>,
    pub initial_guess: DMatrix<f64>,
    pub values: Vec<String>,
    pub arg: String,
    pub BorderConditions: HashMap<String, Vec<(usize, f64)>>,
    pub t0: f64,
    pub t_end: f64,
    pub n_steps: usize,
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
    }
}

impl BuildFrozenSolverRequest for NRBVP {
    fn build_solver_request(&self) -> FrozenSolverBuildRequest {
        let h = (self.t_end - self.t0) / self.n_steps as f64;
        FrozenSolverBuildRequest {
            eq_system: self.eq_system.clone(),
            values: self.values.clone(),
            arg: self.arg.clone(),
            t0: self.t0,
            n_steps: Some(self.n_steps),
            h: Some(h),
            mesh: None,
            border_conditions: self.BorderConditions.clone(),
            scheme: "forward".to_string(),
            method: self.method.clone(),
            bandwidth: None,
            backend_policy: self
                .generated_backend_config
                .effective_backend_policy(&self.method),
            resolver: self.generated_backend_config.resolver.clone(),
            aot_execution_policy: self.generated_backend_config.aot_execution_policy.clone(),
            aot_build_policy: self.generated_backend_config.aot_build_policy,
            aot_chunking_policy: self.generated_backend_config.aot_chunking_policy,
        }
    }
}

impl NRBVP {
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
        let y0 = Box::new(DVector::from_vec(vec![0.0, 0.0]));

        let fun0: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>> =
            Box::new(|_x, y: &DVector<f64>| y.clone());
        let boxed_fun: Box<dyn Fun> = Box::new(FunEnum::Dense(fun0));
        let h = (t_end - t0) / (n_steps - 1) as f64;
        let T_list: Vec<f64> = (0..n_steps)
            .map(|i| t0 + (i as f64) * h)
            .collect::<Vec<_>>();
        // let fun0 =  Box::new( |x, y: &DVector<f64>| y.clone() );
        NRBVP {
            eq_system,
            initial_guess: initial_guess.clone(),
            values,
            arg,
            BorderConditions,
            t0,
            t_end,
            n_steps,
            tolerance,
            strategy,
            strategy_params,
            linear_sys_method,
            method,
            max_iterations,
            max_error: 0.0,
            result: None,
            x_mesh: DVector::from_vec(T_list),
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
    }

    /// Returns a solver configured with the provided generated-backend settings.
    pub fn with_generated_backend_config(mut self, config: GeneratedBackendConfig) -> Self {
        self.generated_backend_config = config;
        self
    }

    /// Returns a solver configured with a high-level sparse generated-backend mode.
    pub fn with_sparse_generated_backend_mode(mut self, mode: SparseGeneratedBackendMode) -> Self {
        self.generated_backend_config = GeneratedBackendConfig::from_sparse_mode(mode);
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

    /// Installs a high-level sparse generated-backend mode on an existing solver.
    pub fn set_sparse_generated_backend_mode(&mut self, mode: SparseGeneratedBackendMode) {
        self.generated_backend_config = GeneratedBackendConfig::from_sparse_mode(mode);
    }

    /// Installs the standard sparse generated-backend defaults.
    pub fn set_sparse_generated_backend_defaults(&mut self) {
        self.set_sparse_generated_backend_mode(SparseGeneratedBackendMode::Defaults);
    }

    /// Returns the full generated-backend configuration.
    pub fn generated_backend_config(&self) -> &GeneratedBackendConfig {
        &self.generated_backend_config
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
        let new_fun = fun.call(p, y);
        let jac = self
            .jac
            .as_mut()
            .expect("Frozen BVP iteration requires an installed Jacobian callback");
        let now = Instant::now();

        let new_j = if self.jac_recalc {
            info!("\n \n JACOBIAN (RE)CALCULATED! \n \n");
            let begin = Instant::now();
            let new_j = jac.call(p, y);
            info!("jacobian recalculation time: ");
            let elapsed = begin.elapsed();
            elapsed_time(elapsed);
            // println!(" \n \n new_j = {:?} ", jac_rowwise_printing(&*&new_j) );
            self.old_jac = Some(new_j.clone_box());
            self.m = 0;
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
        let delta: Box<dyn VectorType> = new_j.solve_sys(
            &*new_fun,
            self.linear_sys_method.clone(),
            self.tolerance,
            self.max_iterations,
            self.bandwidth,
            y,
        );
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
        self.try_eq_generate()?;
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
        GeneratedBackendConfig, SparseGeneratedBackendMode,
    };
    use crate::symbolic::codegen_aot_registry::AotRegistry;
    use crate::symbolic::codegen_aot_resolution::AotResolver;
    use crate::symbolic::codegen_aot_runtime_link::{
        LinkedSparseAotBackend, register_linked_sparse_backend, unregister_linked_sparse_backend,
    };
    use crate::symbolic::codegen_backend_selection::BackendSelectionPolicy;
    use crate::symbolic::codegen_orchestrator::{ParallelExecutorConfig, ParallelFallbackPolicy};
    use crate::symbolic::codegen_runtime_api::ResidualChunkingStrategy;
    use crate::symbolic::codegen_tasks::SparseChunkingStrategy;
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

    #[test]
    fn generated_backend_surface_builder_methods_update_solver_config() {
        let solver = sparse_surface_test_solver()
            .with_backend_policy_override(Some(BackendSelectionPolicy::PreferAotThenLambdify))
            .with_aot_execution_policy(AotExecutionPolicy::SequentialOnly)
            .with_aot_build_policy(AotBuildPolicy::RequirePrebuilt)
            .with_aot_chunking_policy(AotChunkingPolicy::with_parts(
                Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 2 }),
                Some(SparseChunkingStrategy::ByTargetChunkCount { target_chunks: 3 }),
            ));

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

        assert_eq!(options.strategy, "Frozen");
        assert_eq!(options.method, "Sparse");
        assert_eq!(
            options.generated_backend_config.backend_policy_override,
            Some(BackendSelectionPolicy::PreferAotThenLambdify)
        );
        assert_eq!(
            options.generated_backend_config.aot_build_policy,
            AotBuildPolicy::UseIfAvailable
        );
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
    fn build_solver_request_carries_optional_aot_resolver() {
        let mut solver = sparse_surface_test_solver_with_naive_strategy();
        solver.set_aot_resolver(Some(AotResolver::new(AotRegistry::new())));

        let request = solver.build_solver_request();
        assert!(request.resolver.is_some());
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
            ));

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

        let err = next_request.generate().err().expect(
            "next request should now see compiled artifact and fail only at runtime linking",
        );
        assert!(matches!(
            err,
            BvpBackendIntegrationError::CompiledAotRuntimeUnavailable { .. }
        ));
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
