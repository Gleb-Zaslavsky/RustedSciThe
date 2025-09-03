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
use crate::symbolic::symbolic_functions_BVP::Jacobian;
use chrono::Local;

use crate::Utils::logger::{save_matrix_to_csv, save_matrix_to_file};
use crate::Utils::plots::{plots, plots_gnulot};
use crate::numerical::BVP_Damp::BVP_traits::{
    Fun, FunEnum, Jac, MatrixType, VectorType, Vectors_type_casting,
};
use crate::numerical::BVP_Damp::BVP_utils::{
    CustomTimer, construct_full_solution, elapsed_time, extract_unknown_variables, task_check_mem,
};
use crate::numerical::BVP_Damp::BVP_utils_damped::{
    bound_step_Cantera2, convergence_condition, if_initial_guess_inside_bounds, jac_recalc,
};
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
    new_grid_enabled: bool,      //flag indicating if the grid should be refined
    grid_refinemens: usize,      //
    number_of_refined_intervals: usize, //number of refined intervals
    bandwidth: (usize, usize),   //bandwidth
    calc_statistics: HashMap<String, usize>,
    nodes_added: Vec<usize>,
    custom_timer: CustomTimer,
}

impl NRBVP {
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
            calc_statistics: Hashmap_statistics,
            nodes_added: Vec::new(),
            custom_timer: CustomTimer::new(),
        }
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
    pub fn eq_generate(&mut self, mesh_: Option<Vec<f64>>, bandwidth: Option<(usize, usize)>) {
        // check if memory is enough for
        task_check_mem(self.n_steps, self.values.len(), &self.method);
        // check if user specified task is correct
        self.task_check();
        // strategy_check(&self.strategy, &self.strategy_params);
        let mut jacobian_instance = Jacobian::new();
        // mesh of t's can be defined directly or by size of step -h, and number of points
        let (h, n_steps, mesh) = if mesh_.is_none() {
            // case of mesh not defined directly
            let h = Some((self.t_end - self.t0) / self.n_steps as f64);
            let n_steps = Some(self.n_steps);
            (h, n_steps, None)
        } else {
            // case of mesh defined directly
            self.x_mesh = DVector::from_vec(mesh_.clone().unwrap());
            (None, None, mesh_)
        };
        let scheme = self.scheme.clone();

        jacobian_instance.generate_BVP(
            self.eq_system.clone(),
            self.values.clone(),
            self.arg.clone(),
            self.t0.clone(),
            None,
            n_steps.clone(),
            h,
            mesh,
            self.BorderConditions.clone(),
            self.Bounds.clone(),
            self.rel_tolerance.clone(),
            scheme.clone(),
            self.method.clone(),
            bandwidth,
        );

        //     info("Jacobian = {:?}", jacobian_instance.readable_jacobian);
        let fun = jacobian_instance.residiual_function;

        let jac = jacobian_instance.jac_function;

        self.fun = fun;

        self.jac = jac;
        self.bounds_vec = jacobian_instance.bounds.unwrap();
        self.rel_tolerance_vec = jacobian_instance.rel_tolerance_vec.unwrap();
        self.variable_string = jacobian_instance.variable_string;
        self.bandwidth = jacobian_instance.bandwidth.unwrap();
        self.BC_position_and_value = jacobian_instance.BC_pos_n_values;
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
    /////////////////////
    /// Computes Newton step using cached inverse Jacobian
    ///
    /// More efficient than `step()` when Jacobian doesn't need recalculation
    pub fn step_with_inv_Jac(&self, p: f64, y: &dyn VectorType) -> Box<dyn VectorType> {
        let fun = &self.fun;
        let F_k = fun.call(p, y);
        let inv_J_k = self.old_jac.as_ref().unwrap().clone_box();
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
            let jac_function = self.jac.as_mut().unwrap();
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
            let jac_function = self.jac.as_mut().unwrap();
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
        let J_k = self.old_jac.as_ref().unwrap();
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
                panic!()
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
                panic!()
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
            panic!()
        }
        if fbound.is_infinite() {
            error!("\n \n fbound is infinite \n \n");
            panic!()
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
            // if there is a damping coefficient found (so max damp steps not exceeded)
            if S_k.unwrap() > conv {
                //found damping coefficient but not converged yet
                info!("\n \n  Damping coefficient found (solution has not converged yet)");
                info!(
                    "\n \n  step norm =  {}, weight norm = {}, convergence condition = {}",
                    self.error_old,
                    S_k.unwrap(),
                    conv
                );
                (0, damped_step_result)
            } else {
                info!("\n \n  Damping coefficient found (solution has converged)");
                info!(
                    "\n \n step norm =  {}, weight norm = {}, convergence condition = {}",
                    self.error_old,
                    S_k.unwrap(),
                    conv
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
                        panic!()
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
                        panic!(" \n \n y_k_plus_1 is None")
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
                    self.solve_with_new_grid()
                } else {
                    // if adapive is None then we just return the result
                    info!("returning the result");

                    return result;
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
            self.solve_with_new_grid()
        }

        None
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
                    _ => panic!("Unsupported adaptive version"),
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
    fn solve_with_new_grid(&mut self) {
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
            self.eq_generate(Some(new_mesh), Some(self.bandwidth));
            self.custom_timer.symbolic_operations_tac();
        } else {
            info!("no new grid needed - returning to main loop");
            return;
        }
        self.jac_recalc = true;
        self.main_loop_damped();
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
    pub fn solver(&mut self) -> Option<DVector<f64>> {
        self.custom_timer.start();
        self.custom_timer.symbolic_operations_tic();
        self.eq_generate(None, None);
        self.custom_timer.symbolic_operations_tac();
        let begin = Instant::now();
        let res = self.main_loop_damped();
        let end = begin.elapsed();
        elapsed_time(end);
        let time = end.as_secs_f64() as usize;
        self.handle_result();
        self.calc_statistics
            .insert("time elapsed, s".to_string(), time);
        self.calc_statistics();
        self.custom_timer.get_all();
        res
    }

    /// Main public interface for solving BVP
    ///
    /// Wrapper that handles logging configuration and calls internal solver.
    /// Supports configurable logging levels and automatic log file generation.
    ///
    /// # Returns
    /// Solution vector if successful, None if failed
    pub fn solve(&mut self) -> Option<DVector<f64>> {
        let is_logging_disabled = self
            .loglevel
            .as_ref()
            .map(|level| level == "off" || level == "none")
            .unwrap_or(false);

        if is_logging_disabled {
            let res = self.solver();
            res
        } else {
            let loglevel = self.loglevel.clone();
            let log_option = if let Some(level) = loglevel {
                match level.as_str() {
                    "debug" => LevelFilter::Info,
                    "info" => LevelFilter::Info,
                    "warn" => LevelFilter::Warn,
                    "error" => LevelFilter::Error,
                    _ => panic!("loglevel must be debug, info, warn or error"),
                }
            } else {
                LevelFilter::Info
            };
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
                let logger_instance = CombinedLogger::init(vec![
                    TermLogger::new(
                        log_option,
                        Config::default(),
                        TerminalMode::Mixed,
                        ColorChoice::Auto,
                    ),
                    WriteLogger::new(log_option, Config::default(), File::create(name).unwrap()),
                ]);
                logger_instance
            };
            match logger_instance {
                Ok(()) => {
                    let res = self.solver();
                    info!(" \n \n Program ended");
                    res
                }
                Err(_) => {
                    let res = self.solver();
                    res
                } //end Error
            } // end mat 
        }
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
        let result_DMatrix = self.get_result().unwrap();
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
        let result_DMatrix = self.get_result().unwrap();
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
        let vector_of_results = self.result.clone().unwrap().clone();

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
        let permutted_results = self.full_result.clone().unwrap();
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
        let permutted_results = self.full_result.clone().unwrap();
        plots(
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
            let jac_shape = self.old_jac.as_ref().unwrap().shape();
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

/* */
