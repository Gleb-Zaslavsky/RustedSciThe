//!
//! # BVP Symbolic Wrapper Module
//!
//! This module provides a high-level wrapper for solving Boundary Value Problems (BVP)
//! using symbolic expressions. It combines symbolic computation with numerical BVP solving
//! to provide an easy-to-use interface for complex differential equation systems.
//!
//! ## Key Features
//! - **Symbolic to Numerical**: Converts symbolic ODE expressions to efficient numerical functions
//! - **Automatic Jacobian**: Generates analytical Jacobians from symbolic expressions
//! - **Bounds Support**: Applies variable bounds to prevent numerical issues (e.g., log(0))
//! - **Flexible Mesh**: Supports both uniform and custom mesh definitions
//! - **Multiple Solvers**: Integrates with advanced BVP solvers with error control
//! - **Result Analysis**: Provides plotting, saving, and statistical analysis of solutions
//!
//! ## Main Structure: `BVPwrap`
//! The `BVPwrap` struct is the main interface that:
//! 1. Takes symbolic ODE expressions as strings or `Expr` objects
//! 2. Converts them to numerical functions with bounds handling
//! 3. Generates analytical Jacobians for faster convergence
//! 4. Solves the BVP using state-of-the-art numerical methods
//! 5. Provides results in various formats (matrices, plots, files)
//!
//! ## Usage Example
//! ```rust,ignore
//! // Define ODE system: y'' = -y, y(0) = 0, y'(0) = 1
//! let equations = vec!["z".to_string(), "-y".to_string()];
//! let variables = vec!["y".to_string(), "z".to_string()];
//! let boundary_conditions = HashMap::from([
//!     ("y".to_string(), vec![(0, 0.0)]),  // y(0) = 0
//!     ("z".to_string(), vec![(0, 1.0)]),  // z(0) = 1
//! ]);
//!
//! let mut solver = BVPwrap::new(
//!     None, Some(0.0), Some(π), Some(100),
//!     Expr::parse_vector_expression(equations),
//!     variables, vec![], None, boundary_conditions,
//!     "x".to_string(), 1e-6, 1000, initial_guess
//! );
//!
//! solver.solve();
//! solver.plot_result();
//! ```
//!
//! ## Function Overview
//! - `new()`: Creates a new BVP solver instance with mesh and boundary conditions
//! - `set_additional_parameters()`: Configures Jacobian usage and variable bounds
//! - `solve()`: Main solving function with logging support
//! - `eq_generate()`: Converts symbolic expressions to numerical functions
//! - `BC_closure_creater()`: Creates boundary condition functions from HashMap
//! - `plot_result()`, `save_to_file()`: Result visualization and export
//!
use crate::numerical::BVP_Damp::BVP_utils::{CustomTimer, elapsed_time};
use crate::numerical::BVP_sci::BVP_sci_faer::{
    BCFunction, BCJacobian, BVPResult, ODEFunction, ODEJacobian, faer_col, faer_dense_mat,
    solve_bvp,
};
use crate::numerical::BVP_sci::BVP_sci_symbolic_functions::Jacobian_sci_faer;
use crate::symbolic::symbolic_engine::Expr;
use chrono::Local;
use log::{error, info};
use nalgebra::{DMatrix, DVector};
use simplelog::LevelFilter;
use simplelog::*;
use std::collections::HashMap;
use std::fs::File;
use crate::numerical::BVP_sci::BVP_sci_utils::size_of_jacobian;
use crate::Utils::logger::{save_matrix_to_csv, save_matrix_to_file};
use crate::Utils::plots::{plots, plots_gnulot};
use faer::mat::Mat;
use std::time::Instant;
use tabled::{builder::Builder, settings::Style};
pub struct BVPwrap {
    pub eq_system: Vec<Expr>,
    ///equation system
    pub values: Vec<String>,
    ///unknown variables
    pub param: Vec<String>,
    ///parameters
    pub arg: String,
    /// time or coordinate
    pub t0: Option<f64>,
    /// start time or coordinate
    pub t_end: Option<f64>,
    /// end time or coordinate
    pub n_steps: Option<usize>,
    /// number of steps
    pub x_mesh_col: faer_col,
    pub x_mesh: DVector<f64>,
    /// mesh points
    pub param_values: Option<DVector<f64>>,
    /// parameters values
    pub param_col: Option<faer_col>,
    /// parameters as column vector
    pub BoundaryConditions: HashMap<String, Vec<(usize, f64)>>,
    /// boundary conditions function
    pub BC_function: Option<Box<BCFunction>>,
    /// boundary condition function
    pub tolerance: f64,
    ///  Map{var_name: {index, value}} index 0=minimum of variable, 1=maximum of variable
    pub Bounds: Option<HashMap<String, Vec<(usize, f64)>>>,
    /// maximum nodes of mesh
    pub max_nodes: usize,
    pub initial_guess: DMatrix<f64>,
    /// initial guess for the solution
    pub initial_guess_mat: faer_dense_mat,
    /// initial guess as faer_dense_mat
    /// Numerical Jacobian function closure
    pub jac_function: Option<Box<ODEJacobian>>,
    /// Numerical residual function closure
    pub residual_function: Box<ODEFunction>,
    pub use_analytical_jacobian: bool,
    pub loglevel: Option<String>,
    pub result: BVPResult,
    full_result: Option<DMatrix<f64>>,
    custom_timer: CustomTimer,
}
impl BVPwrap {
    pub fn new(
        x_mesh_set: Option<DVector<f64>>,
        t0: Option<f64>,
        t_end: Option<f64>,
        n_steps: Option<usize>,
        eq_system: Vec<Expr>,
        values: Vec<String>,
        param: Vec<String>,
        param_values: Option<DVector<f64>>,
        BoundaryConditions: HashMap<String, Vec<(usize, f64)>>,
        arg: String,
        tolerance: f64,
        max_nodes: usize,
        initial_guess: DMatrix<f64>,
    ) -> Self {
        // mesh can be set either by (start, end, steps) or by a vector of points
        let x_mesh_col = if let Some(x_mesh) = x_mesh_set {
            faer_col::from_iter(x_mesh.iter().map(|&x| x))
        } else {
            if t0.is_some() && t_end.is_some() && n_steps.is_some() {
                let start = t0.unwrap();
                let end = t_end.unwrap();
                let steps = n_steps.unwrap();
                faer_col::from_fn(steps, |i| {
                    start + i as f64 * (end - start) / (steps - 1) as f64
                })
            } else {
                panic!("Either provide a mesh vector or t0, t_end, and n_steps");
            }
        };
        // DVector from faer_col
        let x_mesh: DVector<f64> =
            DVector::from_iterator(x_mesh_col.shape().0, x_mesh_col.iter().map(|&x| x));
        let bc_closure = Self::BC_closure_creater(
            BoundaryConditions.clone(), // default empty HashMap for boundary conditions
            values.clone(),
        );
        // turning parameters into a faer column vector
        let param_col = if let Some(param_values) = param_values.clone() {
            Some(faer_col::from_iter(param_values.iter().map(|&x| x)))
        } else {
            None
        };
        // turning initial guess into a faer_dense_mat
        let initial_guess_mat =
            faer_dense_mat::from_fn(initial_guess.shape().0, initial_guess.shape().1, |i, j| {
                initial_guess[(i, j)]
            });
        BVPwrap {
            eq_system: eq_system,
            values: values,
            param: param,
            arg: arg,

            t0: None,
            t_end: None,
            n_steps: None,
            x_mesh_col: x_mesh_col,
            x_mesh: x_mesh,
            param_values: param_values,
            param_col: param_col,
            tolerance: tolerance,
            Bounds: None,
            BoundaryConditions: BoundaryConditions,
            BC_function: bc_closure,
            max_nodes: max_nodes,
            initial_guess: initial_guess,
            initial_guess_mat: initial_guess_mat,
            jac_function: None,
            residual_function: Box::new(|_, _, _| Mat::zeros(0, 0)),
            result: BVPResult::default(),
            full_result: None,
            use_analytical_jacobian: true,
            loglevel: Some("info".to_string()), // default log level
            custom_timer: CustomTimer::new(),
        }
    }
    pub fn set_additional_parameters(
        &mut self,
        use_analytical_jacobian: Option<bool>,
        Bounds: Option<HashMap<String, Vec<(usize, f64)>>>,
        loglevel: Option<String>
    ) {
        if let Some(use_analytical_jacobian) = use_analytical_jacobian {
            if use_analytical_jacobian {
                println!("\n Using analytical Jacobian \n");
            } else {
                println!("\n Using numerical Jacobian \n");
            }
            self.use_analytical_jacobian = use_analytical_jacobian;
        } else {
            println!("\n Using analytical Jacobian \n");
        }
        self.Bounds = Bounds;
        if let Some(level) = loglevel{
            self.loglevel = Some(level)
        }
    }
    pub fn check_task(&self) {
        assert_eq!(
            self.eq_system.len(),
            self.values.len(),
            "eq_system and values must have the same length"
        );

        assert!(!self.values.is_empty(), "values must not be empty");

        assert!(!self.arg.is_empty(), "arg must not be empty");

        assert!(
            self.x_mesh_col.shape().0 > 1,
            "x_mesh must have at least 2 points"
        );

        if self.t_end.is_some() && self.t0.is_some() {
            assert!(self.t_end > self.t0, "t_end must be greater than t0");
        }

        if self.n_steps.is_some() {
            assert!(self.n_steps.unwrap() > 1, "n_steps must be greater than 1");
        }

        assert!(self.tolerance > 0.0, "tolerance must be greater than 0.0");
        assert!(
            !self.BoundaryConditions.is_empty(),
            "BoundaryConditions must be specified"
        );
        let total_bcs: usize = self.BoundaryConditions.values().map(|v| v.len()).sum();
        assert_eq!(
            total_bcs,
            self.values.len(),
            "Total boundary conditions must equal number of variables"
        );

        // Check if params are provided, param_values should match
        if !self.param.is_empty() {
            if let Some(param_values) = &self.param_values {
                assert_eq!(
                    param_values.len(),
                    self.param.len(),
                    "param_values must be specified for each param"
                );
            } else {
                panic!("param_values must be provided when params are specified");
            }
        }

        assert_eq!(
            self.initial_guess.nrows(),
            self.values.len(),
            "initial_guess rows must match number of variables"
        );
        assert_eq!(
            self.initial_guess.ncols(),
            self.x_mesh_col.shape().0,
            "initial_guess cols must match mesh points"
        );
    }
    /// Check if all necessary components are created
    pub fn is_all_created(&self) {
        assert!(self.x_mesh_col.shape().0 > 0, "x_mesh_col must be created");
        assert!(self.x_mesh.shape().0 > 0, "x_mesh must be created");
        assert!(
            self.initial_guess_mat.shape().0 > 0,
            "initial_guess_mat must be created"
        );
        assert!(
            self.initial_guess_mat.shape().1 > 0,
            "initial_guess_mat must have columns"
        );

        assert!(self.BC_function.is_some(), "BC_function must be created");
        if self.use_analytical_jacobian {
            assert!(self.jac_function.is_some(), "jac_function must be created");
        }

        // Check parameter consistency (params are optional)
        if !self.param.is_empty() {
            assert!(
                self.param_col.is_some(),
                "param_col must be created when params exist"
            );
            if let Some(param_col) = &self.param_col {
                assert_eq!(
                    param_col.shape().0,
                    self.param.len(),
                    "param_col size must match param names"
                );
            }
        }

        // Verify mesh consistency
        assert_eq!(
            self.x_mesh_col.shape().0,
            self.x_mesh.len(),
            "x_mesh_col and x_mesh must have same length"
        );
    }
    /// BC set as HashMap with key as variable name and value as Vec<(position, value)>
    /// where position: 0=left boundary, 1=right boundary
    pub fn BC_closure_creater(
        map_of_bc: HashMap<String, Vec<(usize, f64)>>,
        values: Vec<String>,
    ) -> Option<Box<BCFunction>> {
        if map_of_bc.is_empty() {
            return None;
        }

        let bc_closure = move |ya: &faer_col, yb: &faer_col, _p: &faer_col| {
            let mut bc_residuals = Vec::new();

            for var_name in &values {
                if let Some(conditions) = map_of_bc.get(var_name) {
                    let var_idx = values.iter().position(|v| v == var_name).unwrap();
                    for &(boundary_idx, target_value) in conditions {
                        let residual = match boundary_idx {
                            0 => ya[var_idx] - target_value, // left boundary
                            1 => yb[var_idx] - target_value, // right boundary
                            _ => panic!("Boundary index must be 0 (left) or 1 (right)"),
                        };
                        bc_residuals.push(residual);
                    }
                }
            }

            faer_col::from_fn(bc_residuals.len(), |i| bc_residuals[i])
        };

        Some(Box::new(bc_closure))
    }
    /// Wrapper function to solve BVP with error handling
    pub fn solve_bvp_wrap(
        &mut self,
        fun: &ODEFunction,
        bc: &BCFunction,
        x: faer_col,
        y: faer_dense_mat,
        p: Option<faer_col>,
        _s: Option<faer_dense_mat>, // Singular term not implemented
        fun_jac: Option<&ODEJacobian>,
        bc_jac: Option<&BCJacobian>,
        tol: f64,
        max_nodes: usize,
        verbose: u8,
        bc_tol: Option<f64>,
        custom_timer: CustomTimer,
    ) {
        let begin = Instant::now();

        info!("BVP solver started");
        let bvpres = solve_bvp(
            fun,
            bc,
            x,
            y,
            p,
            _s,
            fun_jac,
            bc_jac,
            tol,
            max_nodes,
            verbose,
            bc_tol,
            Some(custom_timer),
        );

        match bvpres {
            Ok(res) => {
                info!("BVP solved successfully");
                self.result = res.clone();
                self.convert_result();
                let timer = self.result.custom_timer.clone();
                timer.get_all();
                let end = begin.elapsed();
                elapsed_time(end);
                let time = end.as_secs_f64() as usize;
                // println!("{:?}", self.result.calc_statistics);
                let calc_statistics = &mut self.result.calc_statistics;
                calc_statistics.insert("time elapsed, s".to_string(), time);
                self.calc_statistics();
            }
            Err(e) => {
                error!("Error solving BVP: {}", e);
            }
        }
    }
    pub fn solver(&mut self) {
        let begin = Instant::now();
        self.custom_timer.start();
        self.custom_timer.symbolic_operations_tic();
        let (jacobian, residual_function, bc_func) = self.eq_generate();
        self.custom_timer.symbolic_operations_tac();

        // Extract needed fields before mutable borrow

        let x_mesh_col = self.x_mesh_col.clone();
        let initial_guess_mat = self.initial_guess_mat.clone();
        let param_col = self.param_col.clone();
        // let jac_func = Some(self.jac_function.as_ref().unwrap());
        let tolerance = self.tolerance;
        let max_nodes = self.max_nodes;
        let custom_timer = self.custom_timer.clone();
        self.solve_bvp_wrap(
            &residual_function,
            bc_func.as_ref().unwrap(),
            x_mesh_col,
            initial_guess_mat,
            param_col,
            None, // Singular term not implemented
            jacobian.as_deref(),
            None,
            tolerance,
            max_nodes,
            2,    // verbose level
            None, // bc_tol
            custom_timer.clone(),
        );
        let end = begin.elapsed();
        // let's now calculate paramters of jacobian
        let res = self.result.clone();
        let x_mesh = res.x.clone();
        let y = res.y.clone();
        let p =  res.p.clone();
        let p = if let Some(p) = p {
           p 
        } else {
             faer_col::zeros(0)
        };
        let jacfunc = jacobian.unwrap()(
            &x_mesh,
            &y,
            &p
            ).0;
        size_of_jacobian(jacfunc);
        elapsed_time(end);
    }

    /// wrapper around solver function to implement logging

    pub fn solve(&mut self) {
        let is_logging_disabled = self
            .loglevel
            .as_ref()
            .map(|level| level == "off" || level == "none")
            .unwrap_or(false);

        if is_logging_disabled {
            self.solver();
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

            match logger_instance {
                Ok(()) => {
                    self.solver();
                    info!(" \n \n Program ended");
                }
                Err(_) => {
                    self.solver();
                } //end Error
            } // end mat
        }
    }

    pub fn eq_generate(
        &mut self,
    ) -> (
        Option<Box<ODEJacobian>>,
        Box<ODEFunction>,
        Option<Box<BCFunction>>,
    ) {
        self.check_task();
        let mut jac_instance = Jacobian_sci_faer::new();
        jac_instance.generate_BVP(
            self.eq_system.clone(),
            self.values.clone(),
            self.param.clone(),
            self.arg.clone(),
            None,
        );
        let jacobian: Option<Box<ODEJacobian>> = if self.use_analytical_jacobian {
            jac_instance.jac_function
        } else {
            None
        };
        self.jac_function = jacobian;
        self.residual_function = jac_instance.residual_function;
        self.is_all_created();
        info!("BVP equations generated");
        info!("BVP symbolic Jacobian generated");
        let mut jac_instance = Jacobian_sci_faer::new();
        jac_instance.generate_BVP(
            self.eq_system.clone(),
            self.values.clone(),
            self.param.clone(),
            self.arg.clone(),
            self.Bounds.clone(),
        );
        let jacobian: Option<Box<ODEJacobian>> = if self.use_analytical_jacobian {
            jac_instance.jac_function
        } else {
            None
        };
        let residual: Box<ODEFunction> = jac_instance.residual_function;
        let bc_func =
            Self::BC_closure_creater(self.BoundaryConditions.clone(), self.values.clone());
        (jacobian, residual, bc_func)
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                     functions to return and save result in different formats
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // returns a tuple of a dense matrix and a vector
    fn convert_result(&mut self) {
        let result = &self.result;
        let x = result.x.clone();
        let x_len = x.shape().0;
        let x: DVector<f64> = DVector::from_iterator(x_len, x.iter().map(|x| *x));
        let y = result.y.clone();
        let (nrows, ncols) = y.shape();
        let dense = y;
        let mut dmatrix = DMatrix::zeros(ncols, nrows);
        for (i, row) in dense.row_iter().enumerate() {
            let row = row.to_owned().iter().map(|x| *x).collect::<Vec<f64>>();
            dmatrix.column_mut(i).copy_from(&DVector::from_vec(row));
        }
        self.full_result = Some(dmatrix.clone());
        self.x_mesh = x.clone();
    }

    pub fn save_to_file(&self, filename: Option<String>) {
        //let date_and_time = Local::now().format("%Y-%m-%d_%H-%M-%S");
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
    pub fn get_result(&self) -> Option<DMatrix<f64>> {
        self.full_result.clone()
    }

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

    fn calc_statistics(&self) {
        let mut stats = self.result.calc_statistics.clone();

        stats.insert(
            "number of grid points".to_string(),
            self.x_mesh.len() as usize,
        );
        let mut table = Builder::from(stats).build();
        table.with(Style::modern_rounded());
        info!("\n \n CALC STATISTICS \n \n {}", table.to_string());
    }
}
