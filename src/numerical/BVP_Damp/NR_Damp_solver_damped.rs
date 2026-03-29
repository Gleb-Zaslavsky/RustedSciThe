//! # Damped Newton-Raphson Solver for Boundary Value Problems
//!
//! ## Module Purpose
//! This module implements a robust damped Newton-Raphson method for solving systems of nonlinear
//! boundary value problems (BVPs). It's the core solver for stiff ODEs with adaptive grid refinement
//! capabilities, making it essential for the entire RustedSciThe project.
//!
//! ## Key Features
//! - **Damped Newton Method**: Prevents divergence by controlling step size through damping coefficients
//! - **Adaptive Grid Refinement**: Automatically refines mesh where solution changes rapidly
//! - **Jacobian Reuse Strategy**: Optimizes performance by intelligently reusing Jacobian matrices
//! - **Boundary Constraint Handling**: Ensures solution stays within physical bounds
//! - **Multiple Matrix Backends**: Supports both dense and sparse matrix operations
//!
//! ## Core Structures
//! - [`NRBVP`]: Main solver struct containing all problem parameters and solution state
//! - [`SolverParams`]: Configuration for damping parameters and adaptive grid settings
//! - [`AdaptiveGridConfig`]: Specific settings for grid refinement algorithms
//!
//! ## Algorithm Overview
//! The solver uses a modified Newton method with damping to ensure convergence:
//! 1. Compute undamped Newton step: `J(x_k) * Δx = -F(x_k)`
//! 2. Apply boundary constraints to determine maximum step size
//! 3. Use damping coefficient λ to control step: `x_{k+1} = x_k + λ * Δx`
//! 4. Accept step if residual decreases, otherwise reduce λ and retry
//!
//! ## Interesting Code Features
//! - **Trait-based abstraction**: Uses `VectorType` and `MatrixType` traits for backend flexibility
//! - **Macro-based timing**: Custom macros for performance profiling of different operations
//! - **Adaptive strategy pattern**: Configurable Jacobian recalculation strategies
//! - **Memory-aware design**: Checks system memory before allocating large matrices
//! - **Comprehensive logging**: Detailed logging with configurable levels for debugging
//!
//! ## Performance Tips
//! - Use sparse matrices for large problems (>1000 unknowns)
//! - Enable adaptive grid refinement for problems with boundary layers
//! - Tune damping parameters based on problem stiffness
//! - Monitor Jacobian age to balance accuracy vs performance
//!
//! ## References
//! - Cantera MultiNewton solver (MultiNewton.cpp)
//! - TWOPNT Fortran solver ("The Twopnt Program for Boundary Value Problems" by J. F. Grcar)
//! - Chemkin Theory Manual p.261
use crate::symbolic::symbolic_engine::Expr;
use chrono::Local;

use crate::Utils::logger::{save_matrix_to_csv, save_matrix_to_file};
use crate::Utils::plots::{plots, plots_gnulot, plots_terminal};
use crate::numerical::BVP_Damp::BVP_traits::{
    Fun, FunEnum, Jac, MatrixType, VectorType, Vectors_type_casting,
};
use crate::numerical::BVP_Damp::BVP_utils::{
    CustomTimer, construct_full_solution, elapsed_time, extract_unknown_variables, task_check_mem,
};
use crate::numerical::BVP_Damp::BVP_utils_damped::{
    bound_step_Cantera2, convergence_condition, if_initial_guess_inside_bounds, jac_recalc,
};
use crate::numerical::BVP_Damp::generated_solver_handoff::{
    AotBuildPolicy, AotChunkingPolicy, AotExecutionPolicy, ApplyDampedGeneratedSolverState,
    BuildDampedSolverRequest, DampedGeneratedSolverState, DampedSolverBuildRequest,
    GeneratedBackendConfig, SparseGeneratedBackendMode, try_generate_and_apply_damped_solver_state,
};
use crate::symbolic::codegen_aot_resolution::AotResolver;
use crate::symbolic::codegen_backend_selection::BackendSelectionPolicy;
use crate::symbolic::symbolic_functions_BVP::BvpBackendIntegrationError;
use core::panic;
use nalgebra::{DMatrix, DVector};
use simplelog::LevelFilter;
use simplelog::*;
use std::collections::HashMap;
use std::fs::File;
use std::time::Instant;
use tabled::{builder::Builder, settings::Style};

use crate::numerical::BVP_Damp::grid_api::{GridRefinementMethod, new_grid};

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

    /// Creates production-oriented dense damped solver options with standard defaults.
    ///
    /// This is the preferred starting point for dense BVP users that do not
    /// need sparse/AOT-specific behavior.
    pub fn dense_damped() -> Self {
        Self {
            method: "Dense".to_string(),
            ..Self::default()
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

impl Default for DampedSolverOptions {
    fn default() -> Self {
        Self {
            scheme: "forward".to_string(),
            strategy: "Damped".to_string(),
            strategy_params: Some(SolverParams::default()),
            linear_sys_method: None,
            method: "Sparse".to_string(),
            abs_tolerance: 1e-6,
            rel_tolerance: None,
            max_iterations: 25,
            bounds: None,
            loglevel: None,
            generated_backend_config: GeneratedBackendConfig::default(),
        }
    }
}

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
    old_jac: Option<Box<dyn MatrixType>>,
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
    calc_statistics: HashMap<String, usize>,
    nodes_added: Vec<usize>,
    custom_timer: CustomTimer,
}

impl ApplyDampedGeneratedSolverState for NRBVP {
    fn apply_generated_solver_state(&mut self, state: DampedGeneratedSolverState) {
        if let Some(updated_resolver) = state.updated_resolver.clone() {
            self.generated_backend_config.resolver = Some(updated_resolver);
        }
        self.fun = state.fun;
        self.jac = state.jac;
        self.bounds_vec = state.bounds_vec;
        self.rel_tolerance_vec = state.rel_tolerance_vec;
        self.variable_string = state.variable_string;
        self.bandwidth = state.bandwidth;
        self.BC_position_and_value = state.bc_position_and_value;
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

        DampedSolverBuildRequest {
            eq_system: self.eq_system.clone(),
            values: self.values.clone(),
            arg: self.arg.clone(),
            t0: self.t0,
            n_steps,
            h,
            mesh,
            border_conditions: self.BorderConditions.clone(),
            bounds: self.Bounds.clone(),
            rel_tolerance: self.rel_tolerance.clone(),
            scheme: self.scheme.clone(),
            method: self.method.clone(),
            bandwidth,
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
    /// * `initial_guess` - Initial solution guess as matrix (variables × grid points)
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
        let y0 = Box::new(DVector::from_vec(vec![0.0, 0.0]));

        let fun0: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>> =
            Box::new(|_x, y: &DVector<f64>| y.clone());
        let boxed_fun: Box<dyn Fun> = Box::new(FunEnum::Dense(fun0));
        let h = (t_end - t0) / n_steps as f64;
        let T_list: Vec<f64> = (0..n_steps + 1)
            .map(|i| t0 + (i as f64) * h)
            .collect::<Vec<_>>();

        // let fun0 =  Box::new( |x, y: &DVector<f64>| y.clone() );
        let new_grid_enabled_: bool = if let Some(ref params) = strategy_params {
            params.adaptive.is_some()
        } else {
            false
        };
        let vec_of_tuples = vec![
            ("number of iterations".to_string(), 0),
            ("number of solving linear systems".to_string(), 0),
            ("number of jacobians recalculations".to_string(), 0),
            ("number of grid refinements".to_string(), 0),
        ];
        let Hashmap_statistics: HashMap<String, usize> = vec_of_tuples.into_iter().collect();
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
            no_reports: false,
            result: None,
            full_result: None,
            x_mesh: DVector::from_vec(T_list),
            fun: boxed_fun,
            BC_position_and_value: Vec::new(),
            jac: None,
            p: 0.0,
            y: y0,
            m: 0,
            old_jac: None,
            jac_recalc: true,
            error_old: 0.0,

            bounds_vec: Vec::new(),
            rel_tolerance_vec: Vec::new(),
            variable_string: Vec::new(),
            adaptive: false,
            new_grid_enabled: new_grid_enabled_,
            grid_refinemens: 0,
            number_of_refined_intervals: 0,
            bandwidth: (0, 0),
            generated_backend_config: GeneratedBackendConfig::default(),
            calc_statistics: Hashmap_statistics,
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

    pub fn set_mesh(&mut self, t0: f64, t_end: f64, n_steps: usize) {
        let h = (t_end - t0) / n_steps as f64;
        let T_list: Vec<f64> = (0..n_steps + 1)
            .map(|i| t0 + (i as f64) * h)
            .collect::<Vec<_>>();

        self.x_mesh = DVector::from_vec(T_list);
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
        task_check_mem(self.n_steps, self.values.len(), &self.method);
        self.task_check();
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
        self.p = p;
        self.y = y;
        self.initial_guess = initial_guess;
    }

    /// Sets the parameter value (typically time or spatial coordinate)
    pub fn set_p(&mut self, p: f64) {
        self.p = p;
    }

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
    /////////////////////
    /// Computes Newton step using cached inverse Jacobian
    ///
    /// More efficient than `step()` when Jacobian doesn't need recalculation
    pub fn step_with_inv_Jac(&self, p: f64, y: &dyn VectorType) -> Box<dyn VectorType> {
        let fun = &self.fun;
        let F_k = fun.call(p, y);
        let inv_J_k = self
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
            let jac_function = self.jac.as_mut().expect(
                "Damped BVP Jacobian recalculation requires an installed Jacobian callback",
            );
            let jac_matrix = jac_function.call(p, y);
            let inv_J_k = jac_function.inv(&*jac_matrix, self.abs_tolerance, self.max_iterations);
            self.old_jac = Some(inv_J_k);
            self.m = 0;
            *self
                .calc_statistics
                .entry("number of jacobians recalculations".to_string())
                .or_insert(0) += 1;
        }
    }
    ////////////////////////////////////////////////////////////////

    /// Internal method for Jacobian recalculation with timing
    ///
    /// Recalculates Jacobian matrix when needed and updates performance statistics
    fn recalc_jacobian(&mut self) {
        if self.jac_recalc {
            let p = self.p;
            let y = &*self.y;
            info!("\n \n JACOBIAN (RE)CALCULATED! \n \n");
            let begin = Instant::now();
            self.custom_timer.jac_tic();
            let jac_function = self.jac.as_mut().expect(
                "Damped BVP timed Jacobian recalculation requires an installed Jacobian callback",
            );
            let jac_matrix = jac_function.call(p, y);
            // println!(" \n \n new_j = {:?} ", jac_rowwise_printing(&*&new_j) );
            info!("jacobian recalculation time: ");
            let elapsed = begin.elapsed();
            elapsed_time(elapsed);
            self.custom_timer.jac_tac();
            self.old_jac = Some(jac_matrix);
            self.m = 0;
            *self
                .calc_statistics
                .entry("number of jacobians recalculations".to_string())
                .or_insert(0) += 1;
        }
    }

    /// Computes undamped Newton step and solves linear system
    ///
    /// Core method that:
    /// 1. Evaluates residual function F(y)
    /// 2. Solves J(y) * Δy = -F(y) for Newton step
    /// 3. Returns step and timing information
    ///
    /// # Returns
    /// Tuple of (Newton step, (function time, linear system time))
    pub fn step(
        &self,
        p: f64,
        y: &dyn VectorType,
    ) -> (
        Box<dyn VectorType>,
        (std::time::Duration, std::time::Duration),
    ) {
        let fun_time_start = Instant::now();
        let fun = &self.fun;
        let F_k = fun.call(p, y);
        let fun_time_end = fun_time_start.elapsed();
        let J_k = self
            .old_jac
            .as_ref()
            .expect("Damped BVP Newton step requires a cached Jacobian matrix");
        assert_eq!(
            F_k.len(),
            J_k.shape().0,
            "length of F_k {} and number of rows in J_k {} must be equal",
            F_k.len(),
            J_k.shape().0
        );
        assert_eq!(
            J_k.shape().0,
            J_k.shape().1,
            "J_k must be a square matrix, but got shape {:?}",
            J_k.shape()
        );
        assert_eq!(
            F_k.len(),
            J_k.shape().1,
            "length of F_k {} and number of columns in J_k {} must be equal",
            F_k.len(),
            J_k.shape().1
        );
        let residual_norm = F_k.norm();
        info!("\n \n residual norm = {:?} ", residual_norm);
        // jac_rowwise_printing(&J_k);
        //    println!(" \n \n F_k = {:?} \n \n", F_k.to_DVectorType());
        for el in F_k.iterate() {
            if el.is_nan() {
                error!("\n \n NaN in undamped step residual function \n \n");
                panic!("Damped BVP step failed: residual vector contains NaN before linear solve")
            }
        }
        // solving equation J_k*dy_k=-F_k for undamped dy_k, but Lambda*dy_k - is dumped step
        let linear_sys_time_start = Instant::now();
        let undamped_step_k: Box<dyn VectorType> = J_k.solve_sys(
            &*F_k,
            self.linear_sys_method.clone(),
            self.abs_tolerance,
            self.max_iterations,
            self.bandwidth,
            y,
        );
        //  info!("linear system solution {},\n {} \n {}", undamped_step_k.to_DVectorType(), F_k.to_DVectorType(), J_k.to_DMatrixType());
        let linear_sys_time_end = linear_sys_time_start.elapsed();
        for el in undamped_step_k.iterate() {
            if el.is_nan() {
                log::error!("\n \n NaN in damped step deltaY \n \n");
                panic!("Damped BVP step failed: Newton update contains NaN after linear solve")
            }
        }
        let pair_of_times = (fun_time_end, linear_sys_time_end);
        (undamped_step_k, pair_of_times)
    }

    /// Performs damped Newton step with line search
    ///
    /// Implements the core damping algorithm:
    /// 1. Computes undamped Newton step
    /// 2. Applies boundary constraints to determine maximum step size
    /// 3. Uses line search with damping to ensure residual decreases
    /// 4. Returns status code and accepted step
    ///
    /// # Returns
    /// * `(1, Some(step))` - Converged solution found
    /// * `(0, Some(step))` - Step accepted, continue iterations
    /// * `(-2, None)` - No acceptable damping coefficient found
    /// * `(-3, None)` - Step violates bounds
    pub fn damped_step(&mut self) -> (i32, Option<Box<dyn VectorType>>) {
        // macro for saving times
        macro_rules! save_operation_times {
            ($self:expr, $pair_of_times:expr) => {
                let (fun_time, linear_sys_time) = $pair_of_times;
                $self.custom_timer.append_to_fun_time(fun_time);
                $self
                    .custom_timer
                    .append_to_linear_sys_time(linear_sys_time);
            };
        }
        //_________________________________________________________________
        let p = self.p;
        let now = Instant::now();
        // compute the undamped Newton step
        let y_k_minus_1 = &*self.y;
        let (undamped_step_k_minus_1, pair_of_times) = self.step(p, y_k_minus_1);
        // saving times of corresponding operations
        save_operation_times!(self, pair_of_times);
        *self
            .calc_statistics
            .entry("number of solving linear systems".to_string())
            .or_insert(0) += 1;

        let fbound = bound_step_Cantera2(y_k_minus_1, &*undamped_step_k_minus_1, &self.bounds_vec);
        if fbound.is_nan() {
            error!("\n \n fbound is NaN \n \n");
            panic!("Damped BVP damping failed: boundary step factor is NaN")
        }
        if fbound.is_infinite() {
            error!("\n \n fbound is infinite \n \n");
            panic!("Damped BVP damping failed: boundary step factor is infinite")
        }
        // let fbound =1.0;
        info!("\n \n fboundary  = {}", fbound);
        let mut lambda = 1.0 * fbound;
        // if fbound is very small, then x0 is already close to the boundary and
        // step0 points out of the allowed domain. In this case, the Newton
        // algorithm fails, so return an error condition.
        if fbound < 1e-10 {
            log::warn!(
                "\n  No damped step can be taken without violating solution component bounds."
            );
            return (-3, None);
        }

        let maxDampIter = self
            .strategy_params
            .as_ref()
            .and_then(|p| p.max_damp_iter)
            .unwrap_or(5);
        let DampFacor = self
            .strategy_params
            .as_ref()
            .and_then(|p| p.damp_factor)
            .unwrap_or(0.5);

        let mut S_k: Option<f64> = None;
        let mut damped_step_result: Option<Box<dyn VectorType>> = None;
        let mut conv: f64 = 0.0;

        // compute the weighted norm of the undamped step size (s0 in C++ code) - calculated OUTSIDE the loop
        let s0 = undamped_step_k_minus_1.norm();

        let mut k = 0;
        while k < maxDampIter {
            if k > 1 {
                info!("\n \n damped_step number {} ", k);
            }
            info!("\n \n Damping coefficient = {}", lambda);

            // step the solution by the damped step size: x_{k+1} = x_k + alpha_k*step_k
            let damped_step_k = undamped_step_k_minus_1.mul_float(lambda);
            let y_k: Box<dyn VectorType> = y_k_minus_1 - &*damped_step_k;

            // compute the next undamped step that would result if x1 is accepted
            // J(x_k)^-1 F(x_k+1)
            let (undamped_step_k, pair_of_times) = self.step(p, &*y_k);
            // saving times of corresponding operations
            save_operation_times!(self, pair_of_times);
            *self
                .calc_statistics
                .entry("number of solving linear systems".to_string())
                .or_insert(0) += 1;

            // compute the weighted norm of step1 (s1 in C++ code)
            let s1 = undamped_step_k.norm();
            self.error_old = s1;
            info!("\n \n L2 norm of undamped step = {}", s1);
            let convergence_cond_for_step =
                convergence_condition(&*y_k, &self.abs_tolerance, &self.rel_tolerance_vec);

            // If the norm of s1 is less than the norm of s0, then accept this
            // damping coefficient. Also accept it if this step would result in a
            // converged solution. Otherwise, decrease the damping coefficient and
            // try again.
            let elapsed = now.elapsed();
            elapsed_time(elapsed);

            // C++ acceptance criteria: if (s1 < 1.0 || s1 < s0)
            if (s1 < 1.0) || (s1 < s0) {
                // The criterion for accepting is that the undamped steps decrease in
                // magnitude, This prevents the iteration from stepping away from the region where there is good reason to believe a solution lies
                S_k = Some(s1);
                damped_step_result = Some(y_k.clone_box());
                conv = convergence_cond_for_step;
                break;
            }
            // if fail this criterion we must reject it and retries the step with a reduced damping parameter
            lambda = lambda / (2.0f64.powf(k as f64 + DampFacor));
            info!("damping coefficient decreased to {}", lambda);
            S_k = Some(s1);

            k += 1;
        }

        if k < maxDampIter {
            let step_norm = S_k.expect(
                "Damped BVP damping invariant violated: accepted damping step did not record a step norm",
            );
            // if there is a damping coefficient found (so max damp steps not exceeded)
            if step_norm > conv {
                //found damping coefficient but not converged yet
                info!("\n \n  Damping coefficient found (solution has not converged yet)");
                info!(
                    "\n \n  step norm =  {}, weight norm = {}, convergence condition = {}",
                    self.error_old, step_norm, conv
                );
                (0, damped_step_result)
            } else {
                info!("\n \n  Damping coefficient found (solution has converged)");
                info!(
                    "\n \n step norm =  {}, weight norm = {}, convergence condition = {}",
                    self.error_old, step_norm, conv
                );
                (1, damped_step_result)
            }
        } else {
            //  if we have reached max damping iterations without finding a damping coefficient we must reject the step
            warn!("\n \n  No damping coefficient found (max damping iterations reached)");
            (-2, None)
        }
    } // end of damped step
    pub fn calc_residual(&self, y: Box<dyn VectorType>) -> f64 {
        let fun = &self.fun;
        let F_k = fun.call(self.p, &*y);
        F_k.norm()
    }
    /// Main iteration loop for damped Newton-Raphson method
    ///
    /// Orchestrates the complete solution process:
    /// 1. Newton iterations with damping
    /// 2. Jacobian reuse strategy
    /// 3. Adaptive grid refinement when needed
    /// 4. Convergence checking
    ///
    /// # Returns
    /// Solution vector if converged, None if failed
    pub fn main_loop_damped(&mut self) -> Option<DVector<f64>> {
        self.try_main_loop_damped().unwrap_or_else(|err| {
            panic!("Damped BVP main loop failed during fallible runtime path: {err:?}")
        })
    }

    /// Fallible internal Newton loop used by [`NRBVP::try_solver`].
    pub fn try_main_loop_damped(
        &mut self,
    ) -> Result<Option<DVector<f64>>, BvpBackendIntegrationError> {
        ////////////////////////////////////////////////////////////////////////
        info!("\n \n solving system of equations with Newton-Raphson method! \n \n");
        info!("{:?}", self.initial_guess.shape());
        if self.grid_refinemens == 0 {
            let y: DMatrix<f64> = self.initial_guess.clone();
            //  println!("new y = {} \n \n", &y);
            let y: Vec<f64> = y.iter().cloned().collect();
            let y: DVector<f64> = DVector::from_vec(y);
            self.result = Some(y.clone()); // save into result in case the very first iteration
            // with the current n_steps will go wrong and we shall need grid refinement
            self.y = Vectors_type_casting(&y.clone(), self.method.clone());
        } else {
        }
        let initial_res_nornal = self.calc_residual(self.y.clone_box());
        info!("norm of the initial residual = {}", initial_res_nornal);
        // println!("y = {:?}", &y);
        let mut nJacReeval = 0;
        let mut i = 0;
        while i < self.max_iterations {
            self.jac_recalc = jac_recalc(
                &self.strategy_params,
                self.m,
                &self.old_jac,
                &mut self.jac_recalc,
            );
            self.recalc_jacobian();
            self.m += 1;
            i += 1; // increment the number of iterations
            *self
                .calc_statistics
                .entry("number of iterations".to_string())
                .or_insert(0) += 1;
            let (status, damped_step_result) = self.damped_step();

            if status == 0 {
                // status == 0 means convergence is not reached yet we're going to another iteration
                let y_k_plus_1 = match damped_step_result {
                    Some(y_k_plus_1) => y_k_plus_1,
                    _ => {
                        error!("\n \n y_k_plus_1 is None");
                        panic!(
                            "Damped BVP main loop invariant violated: accepted step returned no updated state"
                        )
                    }
                };
                self.y = y_k_plus_1;
                self.jac_recalc = false;
            }
            // status == 0
            else if status == 1 {
                // status == 1 means convergence is reached, save the result
                info!("\n \n Solution has converged, breaking the loop!");

                let y_k_plus_1 = match damped_step_result {
                    Some(y_k_plus_1) => y_k_plus_1,
                    _ => {
                        panic!(
                            "Damped BVP main loop invariant violated: converged step returned no updated state"
                        )
                    }
                };
                let resid_norm = self.calc_residual(y_k_plus_1.clone_box());
                info!("residual norm of the solution = {}", resid_norm);
                let result = Some(y_k_plus_1.to_DVectorType()); // save the successful result of the iteration
                // before refining in case it will go wrong
                self.result = result.clone();
                info!(
                    "\n \n solution found for the current grid {}",
                    &self.result.clone().unwrap().len()
                );

                // if flag for new grid is up we must call adaptive grid refinement
                if self.new_grid_enabled
                    && self
                        .strategy_params
                        .as_ref()
                        .map_or(false, |p| p.adaptive.is_some())
                {
                    info!("solving with new grid!");
                    return self.try_solve_with_new_grid();
                } else {
                    // if adapive is None then we just return the result
                    info!("returning the result");

                    return Ok(result);
                };
            //  self.max_error = error; // ???
            }
            // status == 1
            else if status < 0 {
                //negative means convergence is not reached yet, damped step is not accepted
                if self.m > 1 {
                    // if we have already tried 2 times with same Jacobian we must recalculate Jacobian
                    self.jac_recalc = true;
                    info!(
                        "\n \n status <0, recalculating Jacobian flag up! Jacobian age = {} \n \n",
                        self.m
                    );
                    if nJacReeval > 3 {
                        break;
                    }
                    nJacReeval += 1;
                } else {
                    info!("\n \n Jacobian age {} =<1 \n \n", self.m);
                    //  self.new_grid_enabled = true;
                    break;
                }
            } // status <0

            info!("\n \n end of iteration {} with jac age {} \n \n", i, self.m);
        }

        // all iterations, recalculations of Jacobian were unsuccessful
        // only that can help - grid refinement

        if self.new_grid_enabled
            && self
                .strategy_params
                .as_ref()
                .map_or(false, |p| p.adaptive.is_some())
        {
            info!("\n \n iterations unsuccessful, calling solve_with_new_grid \n \n");
            return self.try_solve_with_new_grid();
        }

        Ok(None)
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                      functions to create a new grid and recalculate with new grid
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    /// Determines if grid refinement is needed based on solution analysis
    ///
    /// Checks refinement criteria and maximum refinement limits
    pub fn we_need_refinement(&mut self) {
        if let Some(ref params) = self.strategy_params {
            if let Some(ref adaptive_config) = params.adaptive {
                let mut res = match adaptive_config.version {
                    1 => {
                        if self.number_of_refined_intervals == 0 {
                            log::info!(
                                "\n \n number of marked intervals is 0, no new grid is needed \n \n"
                            );
                            false
                        } else {
                            log::info!(
                                "\n \n number of marked intervals is {}, new grid is needed \n \n",
                                self.number_of_refined_intervals
                            );
                            true
                        }
                    }
                    _ => panic!(
                        "Damped BVP adaptive grid refinement failed: unsupported adaptive version"
                    ),
                };

                if adaptive_config.max_refinements <= self.grid_refinemens {
                    info!(
                        "maximum number of grid refinements {} reached {}",
                        adaptive_config.max_refinements, self.grid_refinemens
                    );
                    res = false;
                }
                self.new_grid_enabled = res;
            }
        }
    }

    /// Creates refined mesh based on solution gradients and error estimates
    ///
    /// # Returns
    /// Tuple of (new mesh points, interpolated initial guess, number of refined intervals)
    fn create_new_grid(&mut self) -> (Vec<f64>, DVector<f64>, usize) {
        info!("================GRID REFINEMENT===================");
        let y = self.result.clone().unwrap().clone_box();
        let y_DVector = y.to_DVectorType();
        let number_of_Ys = self.values.len();
        let n_steps = self.n_steps;

        // dbg!(&y_DMatrix.);
        let method = self
            .strategy_params
            .as_ref()
            .and_then(|p| p.adaptive.as_ref())
            .map(|a| a.grid_method.clone())
            .expect("Grid method must be specified when adaptive is enabled");

        self.custom_timer.grid_refinement_tic();

        let BC_position_and_value = self.BC_position_and_value.clone();
        let full_result_vector = construct_full_solution(y_DVector.clone(), BC_position_and_value);

        let y_DMatrix =
            DMatrix::from_column_slice(number_of_Ys, n_steps + 1, full_result_vector.as_slice());
        for (value, row) in self.values.iter().zip(y_DMatrix.clone().row_iter()) {
            let row: Vec<f64> = row.iter().cloned().collect();
            log::debug!(
                "Initial guess for {}: {:?} of len {}",
                value,
                row,
                row.len()
            );
        }

        let residuals: Option<DVector<f64>> = match method {
            GridRefinementMethod::Sci() => {
                // compute residuals on the current grid
                let fun = &self.fun;
                let p = self.p;
                let y_dvector = self.result.clone().unwrap();
                let y = crate::numerical::BVP_Damp::BVP_traits::Vectors_type_casting(
                    &y_dvector,
                    self.method.clone(),
                );
                let residuals = fun.call(p, &*y);
                let residuals = residuals.to_DVectorType();
                Some(residuals)
            }
            _ => None,
        };

        let (new_mesh, initial_guess, number_of_nonzero_keys) = new_grid(
            method,
            &y_DMatrix,
            &self.x_mesh,
            self.abs_tolerance,
            residuals,
        );

        let initial_guess = extract_unknown_variables(
            initial_guess,
            &self.BC_position_and_value,
            number_of_nonzero_keys,
        );
        assert_eq!(
            initial_guess.len(),
            (new_mesh.len() - 1) * self.values.len(),
            "Initial guess size mismatch after grid refinement"
        );

        //  dbg!(&initial_guess);

        info!("================GRID REFINEMENT ENDED===================");
        self.custom_timer.grid_refinement_tac();
        (new_mesh, initial_guess, number_of_nonzero_keys)
    }

    /// Continues solving on refined grid
    ///
    /// Updates solver state with new mesh and restarts Newton iterations
    fn try_solve_with_new_grid(
        &mut self,
    ) -> Result<Option<DVector<f64>>, BvpBackendIntegrationError> {
        let (new_mesh, initial_guess, number_of_nonzero_keys) = self.create_new_grid();
        self.custom_timer.grid_refinement_tac();
        self.number_of_refined_intervals = number_of_nonzero_keys;
        self.nodes_added.push(number_of_nonzero_keys);
        let initial_guess_matrix = DMatrix::from_column_slice(
            self.values.len(),
            new_mesh.len() - 1,
            initial_guess.as_slice(),
        );

        self.initial_guess = initial_guess_matrix;
        self.y = Vectors_type_casting(&initial_guess, self.method.clone());
        self.x_mesh = DVector::from_vec(new_mesh.clone());
        self.grid_refinemens += 1;
        info!(
            "\n \n grid refinement counter = {} \n \n",
            self.grid_refinemens
        );
        *self
            .calc_statistics
            .entry("number of grid refinements".to_string())
            .or_insert(0) += 1;
        self.we_need_refinement();

        if number_of_nonzero_keys > 0 {
            // Clear old Jacobian completely to avoid dimension mismatch
            self.old_jac = None;
            self.jac = None; // Clear Jacobian function as well
            self.jac_recalc = true; // Force Jacobian recalculation for new grid
            self.m = 0; // Reset Jacobian age counter

            // Clear other cached state that might be invalid for new grid
            self.bounds_vec.clear();
            self.rel_tolerance_vec.clear();
            self.variable_string.clear();

            // Update grid parameters
            self.n_steps = new_mesh.len() - 1;

            info!(
                "new guess of shape {} {}",
                self.initial_guess.nrows(),
                self.initial_guess.ncols()
            );
            info!("new mesh length {}", new_mesh.len());

            self.custom_timer.symbolic_operations_tic();
            // Regenerate system with new grid - no need to recalculate bandwidth
            self.try_eq_generate(Some(new_mesh), Some(self.bandwidth))?;
            self.custom_timer.symbolic_operations_tac();
        } else {
            info!("no new grid needed - returning to main loop");
            return Ok(None);
        }
        self.jac_recalc = true;
        self.try_main_loop_damped()
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                       main functions to start the solver
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    /// Internal solver method with timing and statistics
    ///
    /// Coordinates the complete solution process:
    /// 1. Symbolic system generation
    /// 2. Newton iteration loop
    /// 3. Result processing and statistics
    ///
    /// # Returns
    /// Solution vector if successful, None if failed
    /// Fallible solve path without logging setup.
    ///
    /// This is the preferred entrypoint for internal/runtime callers that want
    /// typed backend and execution errors but do not need the higher-level
    /// logging wrapper provided by [`NRBVP::try_solve`].
    pub fn try_solver(&mut self) -> Result<Option<DVector<f64>>, BvpBackendIntegrationError> {
        self.custom_timer.start();
        self.custom_timer.symbolic_operations_tic();
        self.try_eq_generate(None, None)?;
        self.custom_timer.symbolic_operations_tac();
        let begin = Instant::now();
        let res = self.try_main_loop_damped()?;
        let end = begin.elapsed();
        elapsed_time(end);
        let time = end.as_secs_f64() as usize;
        self.handle_result();
        self.calc_statistics
            .insert("time elapsed, s".to_string(), time);
        self.calc_statistics();
        self.custom_timer.get_all();
        Ok(res)
    }

    /// Compatibility-only wrapper over [`NRBVP::try_solver`].
    ///
    /// New production-facing code should call [`NRBVP::try_solver`] or
    /// [`NRBVP::try_solve`] instead.
    pub fn solver(&mut self) -> Option<DVector<f64>> {
        self.try_solver()
            .unwrap_or_else(|err| panic!("BVP solver failed before Newton loop: {err:?}"))
    }

    /// Main public interface for solving BVP
    ///
    /// Wrapper that handles logging configuration and calls internal solver.
    /// Supports configurable logging levels and automatic log file generation.
    ///
    /// # Returns
    /// Solution vector if successful, None if failed
    pub fn try_solve(&mut self) -> Result<Option<DVector<f64>>, BvpBackendIntegrationError> {
        let is_logging_disabled = self
            .loglevel
            .as_ref()
            .map(|level| level == "off" || level == "none")
            .unwrap_or(false);

        if is_logging_disabled {
            self.try_solver()
        } else {
            let log_option = self.parse_log_level()?;
            let logger_instance = if self.no_reports {
                // don't want to save txt report
                let logger_instance = CombinedLogger::init(vec![TermLogger::new(
                    log_option,
                    Config::default(),
                    TerminalMode::Mixed,
                    ColorChoice::Auto,
                )]);
                logger_instance
            } else {
                // want to save txt report
                let date_and_time = Local::now().format("%Y-%m-%d_%H-%M-%S");
                let name = format!("log_{}.txt", date_and_time);
                let file = File::create(&name).map_err(|err| {
                    BvpBackendIntegrationError::LogFileCreationFailed {
                        path: name.clone(),
                        message: err.to_string(),
                    }
                })?;
                let logger_instance = CombinedLogger::init(vec![
                    TermLogger::new(
                        log_option,
                        Config::default(),
                        TerminalMode::Mixed,
                        ColorChoice::Auto,
                    ),
                    WriteLogger::new(log_option, Config::default(), file),
                ]);
                logger_instance
            };
            match logger_instance {
                Ok(()) => {
                    let res = self.try_solver()?;
                    info!(" \n \n Program ended");
                    Ok(res)
                }
                Err(_) => self.try_solver(), //end Error
            } // end mat 
        }
    }

    /// Compatibility-only wrapper over [`NRBVP::try_solve`].
    ///
    /// New code should prefer [`NRBVP::try_solve`] so AOT/logging/runtime failures
    /// remain typed instead of turning into a panic.
    pub fn solve(&mut self) -> Option<DVector<f64>> {
        self.try_solve()
            .unwrap_or_else(|err| panic!("BVP solve failed before convergence loop: {err:?}"))
    }
    pub fn dont_save_log(&mut self, dont_save_log: bool) {
        self.no_reports = dont_save_log;
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                     functions to return and save result in different formats
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// Saves solution to text file with formatted output
    ///
    /// # Arguments
    /// * `filename` - Optional filename (defaults to "result.txt")
    pub fn save_to_file(&self, filename: Option<String>) {
        let name = if let Some(name) = filename {
            format!("{}.txt", name)
        } else {
            "result.txt".to_string()
        };
        let result_DMatrix = self
            .get_result()
            .expect("Damped BVP save_to_file requires a computed full solution matrix");
        let _ = save_matrix_to_file(
            &result_DMatrix,
            &self.values,
            &name,
            &self.x_mesh,
            &self.arg,
        );
    }

    /// Saves solution to CSV file for data analysis
    ///
    /// # Arguments
    /// * `filename` - Optional filename (defaults to "result_table")
    pub fn save_to_csv(&self, filename: Option<String>) {
        let name = if let Some(name) = filename {
            name
        } else {
            "result_table".to_string()
        };
        let result_DMatrix = self
            .get_result()
            .expect("Damped BVP save_to_csv requires a computed full solution matrix");
        let _ = save_matrix_to_csv(
            &result_DMatrix,
            &self.values,
            &name,
            &self.x_mesh,
            &self.arg,
        );
    }

    /// Returns the complete solution matrix including boundary conditions
    ///
    /// # Returns
    /// Matrix where rows are grid points and columns are variables
    pub fn get_result(&self) -> Option<DMatrix<f64>> {
        self.full_result.clone()
    }

    /// Processes raw solution vector into full solution matrix
    ///
    /// Reconstructs complete solution by adding boundary conditions
    /// and reshaping into proper matrix format
    pub fn handle_result(&mut self) {
        let number_of_Ys = self.values.len();
        let n_steps = self.n_steps;
        let vector_of_results = self
            .result
            .clone()
            .expect("Damped BVP handle_result requires a converged solution vector")
            .clone();

        let BC_position_and_value = self.BC_position_and_value.clone();
        let full_results_vector = construct_full_solution(vector_of_results, BC_position_and_value);
        let full_results: DMatrix<f64> =
            DMatrix::from_column_slice(number_of_Ys, n_steps + 1, full_results_vector.as_slice());
        let full_results = full_results.transpose();
        info!("matrix of results has shape {:?}", full_results.shape());
        info!("length of x mesh : {:?}", n_steps);
        info!("number of Ys: {:?}", number_of_Ys);
        self.full_result = Some(full_results.clone());
    }
    /// Creates plots using gnuplot backend
    ///
    /// Generates publication-quality plots of the solution
    /// Requires gnuplot to be installed and in PATH
    pub fn gnuplot_result(&self) {
        let permutted_results = self
            .full_result
            .clone()
            .expect("Damped BVP gnuplot_result requires a computed full solution matrix");
        plots_gnulot(
            self.arg.clone(),
            self.values.clone(),
            self.x_mesh.clone(),
            permutted_results,
        );
        info!("result plotted");
    }

    /// Creates plots using plotters crate
    ///
    /// Generates solution plots with embedded Rust plotting
    pub fn plot_result(&self) {
        let permutted_results = self
            .full_result
            .clone()
            .expect("Damped BVP plot_result requires a computed full solution matrix");
        plots(
            self.arg.clone(),
            self.values.clone(),
            self.x_mesh.clone(),
            permutted_results,
        );
        info!("result plotted");
    }
    pub fn plot_result_in_terminal(&self) {
        let permutted_results = self
            .full_result
            .clone()
            .expect("Damped BVP plot_result_in_terminal requires a computed full solution matrix");
        plots_terminal(
            self.arg.clone(),
            self.values.clone(),
            self.x_mesh.clone(),
            permutted_results,
        );
        info!("result plotted");
    }
    /// Computes and displays solver performance statistics
    ///
    /// Shows memory usage, iteration counts, and timing information
    fn calc_statistics(&self) {
        let mut stats = self.calc_statistics.clone();
        if let Some(jac) = &self.old_jac {
            let jac_shape = self
                .old_jac
                .as_ref()
                .expect("Damped BVP calc_statistics requires a cached Jacobian when statistics say one is available")
                .shape();
            let matrix_weight = checkmem(&**jac);
            stats.insert("jacobian memory, MB".to_string(), matrix_weight as usize);
            stats.insert(
                "number of jacobian elements".to_string(),
                jac_shape.0 * jac_shape.1,
            );
        }
        stats.insert("length of y vector".to_string(), self.y.len() as usize);
        stats.insert(
            "number of grid points".to_string(),
            self.x_mesh.len() as usize,
        );
        let mut table = Builder::from(stats).build();
        table.with(Style::modern_rounded());
        info!("\n \n CALC STATISTICS \n \n {}", table.to_string());

        // What nodes were added per refinement

        let nodes_added_table: Vec<Vec<String>> = self
            .nodes_added
            .iter()
            .enumerate()
            .map(|(idx, val)| vec![idx.to_string(), val.to_string()])
            .collect();

        info!("\n \n NODES ADDED PER REFINEMENT \n \n");
        let mut table = Builder::from(nodes_added_table).build();
        table.with(Style::modern_rounded());
        info!("\n {} \n", table.to_string());
    }

    ////////////////////////////////////////////////////////////
    // Utility methods
    ////////////////////////////////////////////////////////////

    /// Updates internal timing statistics
    ///
    /// Used internally to accumulate timing data for performance analysis
    pub fn step_with_timer(&mut self, pair_of_times: (std::time::Duration, std::time::Duration)) {
        let (fun_time, linear_sys_time) = pair_of_times;
        self.custom_timer.append_to_fun_time(fun_time);
        self.custom_timer.append_to_linear_sys_time(linear_sys_time);
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
    use nalgebra::DMatrix;
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
            DampedSolverOptions::sparse_damped(),
        )
    }

    fn sparse_surface_test_solver_with_tolerances() -> NRBVP {
        let bounds = HashMap::from([
            ("z".to_string(), (-10.0, 10.0)),
            ("y".to_string(), (-10.0, 10.0)),
        ]);
        let rel_tolerance = HashMap::from([("z".to_string(), 1e-4), ("y".to_string(), 1e-4)]);
        let options = DampedSolverOptions {
            strategy_params: Some(SolverParams::default()),
            abs_tolerance: 1e-6,
            rel_tolerance: Some(rel_tolerance),
            max_iterations: 10,
            bounds: Some(bounds),
            ..DampedSolverOptions::sparse_damped()
        };
        NRBVP::new_with_options(
            vec![Expr::parse_expression("y-z"), Expr::parse_expression("-z")],
            DMatrix::from_element(2, 5, 0.25),
            vec!["z".to_string(), "y".to_string()],
            "x".to_string(),
            HashMap::from([
                ("z".to_string(), vec![(0usize, 1.0f64)]),
                ("y".to_string(), vec![(1usize, 1.0f64)]),
            ]),
            0.0,
            1.0,
            5,
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
    fn sparse_damped_options_preset_sets_production_defaults() {
        let options = DampedSolverOptions::sparse_damped();

        assert_eq!(options.scheme, "forward");
        assert_eq!(options.strategy, "Damped");
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
    fn dense_damped_options_preset_sets_dense_defaults() {
        let options = DampedSolverOptions::dense_damped();

        assert_eq!(options.scheme, "forward");
        assert_eq!(options.strategy, "Damped");
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
            "forward".to_string(),
            "Damped".to_string(),
            None,
            None,
            "Sparse".to_string(),
            1e-6,
            None,
            10,
            None,
            None,
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
        let options = DampedSolverOptions::sparse_damped().with_sparse_aot_require_prebuilt();

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
    fn sparse_eq_generate_uses_bundle_handoff_without_breaking_metadata() {
        let values = vec!["z".to_string(), "y".to_string()];
        let n_steps = 5;
        let mut solver = sparse_surface_test_solver_with_tolerances();

        solver
            .try_eq_generate(None, None)
            .expect("sparse handoff should generate through the fallible API");

        assert!(solver.jac.is_some());
        assert!(!solver.variable_string.is_empty());
        assert!(!solver.BC_position_and_value.is_empty());
        assert_eq!(solver.bounds_vec.len(), values.len() * n_steps);
        assert_eq!(solver.rel_tolerance_vec.len(), values.len() * n_steps);
        let _ = &solver.fun;
    }

    #[test]
    fn build_solver_request_carries_optional_aot_resolver() {
        let mut solver = sparse_surface_test_solver_with_tolerances();
        solver.set_aot_resolver(Some(AotResolver::new(AotRegistry::new())));

        let request = solver.build_solver_request(None, None);
        assert!(request.resolver.is_some());
    }

    #[test]
    fn build_solver_request_uses_backend_policy_override_when_present() {
        let mut solver = sparse_surface_test_solver_with_tolerances();
        solver.set_backend_policy_override(Some(BackendSelectionPolicy::NumericOnly));

        let request = solver.build_solver_request(None, None);
        assert_eq!(request.backend_policy, BackendSelectionPolicy::NumericOnly);
    }

    #[test]
    fn generated_backend_config_is_exposed_as_user_facing_solver_setting() {
        let mut solver = sparse_surface_test_solver_with_tolerances();

        let config = GeneratedBackendConfig::with_parts(
            Some(BackendSelectionPolicy::NumericOnly),
            Some(AotResolver::new(AotRegistry::new())),
        );
        solver.set_generated_backend_config(config);

        let request = solver.build_solver_request(None, None);
        assert_eq!(request.backend_policy, BackendSelectionPolicy::NumericOnly);
        assert!(request.resolver.is_some());
        assert_eq!(
            solver.generated_backend_config().backend_policy_override,
            Some(BackendSelectionPolicy::NumericOnly)
        );
    }

    #[test]
    fn generated_backend_config_can_be_applied_during_solver_construction() {
        let solver = sparse_surface_test_solver_with_tolerances().with_generated_backend_config(
            GeneratedBackendConfig::with_parts(
                Some(BackendSelectionPolicy::NumericOnly),
                Some(AotResolver::new(AotRegistry::new())),
            ),
        );

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
            .with_aot_build_policy(AotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Release,
            })
            .with_aot_chunking_policy(AotChunkingPolicy::with_parts(
                Some(ResidualChunkingStrategy::ByOutputCount {
                    max_outputs_per_chunk: 8,
                }),
                Some(SparseChunkingStrategy::ByRowCount { rows_per_chunk: 4 }),
            ));

        let mut solver =
            sparse_surface_test_solver_with_tolerances().with_generated_backend_config(config);

        let request = solver.build_solver_request(None, None);
        assert_eq!(
            request.aot_build_policy,
            AotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Release
            }
        );
        assert_eq!(
            request.aot_chunking_policy.residual,
            Some(ResidualChunkingStrategy::ByOutputCount {
                max_outputs_per_chunk: 8
            })
        );
        assert_eq!(
            request.aot_chunking_policy.sparse_jacobian,
            Some(SparseChunkingStrategy::ByRowCount { rows_per_chunk: 4 })
        );
        match request.aot_execution_policy {
            AotExecutionPolicy::Parallel(inner) => {
                assert_eq!(inner.jobs_per_worker, 2);
                assert_eq!(inner.max_residual_jobs, Some(4));
                assert_eq!(inner.max_sparse_jobs, Some(2));
            }
            other => std::panic!("expected parallel execution policy, got {other:?}"),
        }
    }

    #[test]
    fn eq_generate_build_if_missing_saves_compiled_resolver_for_next_request() {
        let mut solver = sparse_surface_test_solver_with_tolerances()
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
            .try_eq_generate(None, None)
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

        let next_request = solver.build_solver_request(None, None);
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
        let n_steps = 5;
        let mut solver = sparse_surface_test_solver_with_tolerances()
            .with_backend_policy_override(Some(BackendSelectionPolicy::PreferAotThenLambdify))
            .with_aot_build_policy(AotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Release,
            });

        solver
            .try_eq_generate(None, None)
            .expect("first sparse generation should succeed through the fallible API");
        let saved_resolver = solver
            .generated_backend_config()
            .resolver
            .clone()
            .expect("first build-if-missing run should save updated resolver");

        let y = Col::from_fn(values.len() * n_steps, |index| 0.2 + index as f64 * 0.01);
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
                    *dst = *src + 123.0;
                }
            }),
            Arc::new(move |_args, out| {
                for (index, value) in out.iter_mut().enumerate() {
                    *value = 700.0 + index as f64;
                }
            }),
        ));

        solver.set_aot_build_policy(AotBuildPolicy::RequirePrebuilt);
        solver
            .try_eq_generate(None, None)
            .expect("second sparse generation should reuse resolver through the fallible API");
        let residual = solver.fun.call(0.0, &y).to_DVectorType();

        for (actual, expected) in residual.iter().zip(baseline.iter()) {
            assert!((actual - (expected + 123.0)).abs() < 1e-10);
        }

        unregister_linked_sparse_backend(&problem_key);
    }

    #[test]
    fn try_eq_generate_surfaces_missing_prebuilt_aot_as_typed_error() {
        let mut solver =
            sparse_surface_test_solver_with_tolerances().with_sparse_aot_require_prebuilt();

        let err = solver
            .try_eq_generate(None, None)
            .expect_err("try_eq_generate should return a typed AOT availability error");

        assert!(matches!(
            err,
            BvpBackendIntegrationError::CompiledAotRequiredButUnavailable { .. }
        ));
    }

    #[test]
    fn try_solve_surfaces_missing_prebuilt_aot_as_typed_error() {
        let mut solver =
            sparse_surface_test_solver_with_tolerances().with_sparse_aot_require_prebuilt();
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
    fn try_solve_surfaces_invalid_loglevel_as_typed_error() {
        let mut solver = sparse_surface_test_solver_with_tolerances();
        solver.loglevel = Some("trace".to_string());

        let err = solver
            .try_solve()
            .expect_err("invalid loglevel should be returned as a typed error");

        assert!(matches!(
            err,
            BvpBackendIntegrationError::InvalidLogLevel { ref level } if level == "trace"
        ));
    }
}
