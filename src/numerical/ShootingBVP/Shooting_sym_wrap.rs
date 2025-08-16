//! # Symbolic Shooting Method Wrapper for Boundary Value Problems
//!
//! This module provides a high-level interface for solving boundary value problems (BVPs)
//! using the shooting method with symbolic equation definitions and various IVP solvers.
//!
//! ## Overview
//!
//! The shooting method converts a BVP into an initial value problem (IVP) by guessing
//! unknown initial conditions and iteratively adjusting them until boundary conditions
//! are satisfied. This wrapper allows you to:
//!
//! - Define ODEs symbolically using string expressions
//! - Use any available IVP solver (RK45, BDF, Radau, etc.)
//! - Handle mixed boundary conditions (Dirichlet/Neumann)
//! - Automatically generate equation closures from symbolic expressions
//!
//! ## Supported Boundary Conditions
//!
//! - **Dirichlet**: y(x) = value (position specified)
//! - **Neumann**: y'(x) = value (derivative specified)
//! - **Mixed**: Combinations of Dirichlet and Neumann at different boundaries
//!
//! ## Variable Convention
//!
//! For second-order ODEs converted to first-order systems:
//! - `y0` represents the function value y(x)
//! - `y1` represents the derivative y'(x)
//!
//! ## Usage Example
//!
//! ```rust, ignore
//! use std::collections::HashMap;
//! use RustedSciThe::numerical::ShootingBVP::Shooting_sym_wrap::BVPShooting;
//! use RustedSciThe::numerical::ODE_api2::{SolverType, SolverParam};
//! use RustedSciThe::symbolic::symbolic_engine::Expr;
//!
//! // Define second-order ODE: y'' + y = 0
//! // Convert to first-order system: y0' = y1, y1' = -y0
//! let eq_vec = vec![
//!     Expr::parse_expression("y1"),    // y0' = y1
//!     Expr::parse_expression("-y0")   // y1' = -y0
//! ];
//! let values = vec!["y0".to_string(), "y1".to_string()];
//! let arg = "x".to_string();
//!
//! // Set boundary conditions: y(0) = 1, y(π/2) = 0
//! let mut boundary_conditions = HashMap::new();
//! boundary_conditions.insert("y0".to_string(), vec![(0, 1.0), (1, 0.0)]);
//!
//! let borders = (0.0, std::f64::consts::PI / 2.0);
//!
//! // Create BVP instance
//! let mut bvp = BVPShooting::new(eq_vec, values, arg, boundary_conditions, borders);
//!
//! // Solve with RK45 (simple method)
//! bvp.simple_solve(0.0, 1e-6, 100, 0.01);
//!
//! // Or solve with specific IVP solver
//! let params = HashMap::new();
//! bvp.solve_with_certain_ivp(
//!     0.0, 1e-6, 100, 0.01,
//!     SolverType::NonStiff("RK45".to_string()),
//!     params
//! );
//!
//! // Get results
//! let solution = bvp.get_solution();
//! let x_mesh = bvp.get_x();
//! let y_matrix = bvp.get_y();
//! ```

use crate::numerical::ODE_api2::{SolverParam, SolverType, UniversalODESolver};
use crate::numerical::ShootingBVP::Shooting_simple::{
    BoundaryCondition, BoundaryConditionType, BoundaryValueProblem, RK45_ivp_solver_and_mesh,
    ShootingMethodResult, ShootingMethodSolver,
};

use crate::symbolic::symbolic_engine::Expr;
use log::{debug, info};
use nalgebra::{DMatrix, DVector};
use simplelog::*;
use std::collections::HashMap;
/// Main structure for solving boundary value problems using the shooting method
/// with symbolic equation definitions and configurable IVP solvers.
pub struct BVPShooting {
    /// Vector of symbolic expressions representing the ODE system
    pub eq_vec: Vec<Expr>,
    /// Names of the dependent variables (e.g., ["y0", "y1"] for y and y')
    pub values: Vec<String>,
    /// Name of the independent variable (e.g., "x" or "t")
    pub arg: String,
    /// Boundary conditions map: variable_name -> [(boundary_index, value)]
    /// boundary_index: 0 = left boundary, 1 = right boundary
    pub BoundaryConditions: HashMap<String, Vec<(usize, f64)>>,
    /// Internal shooting method solver instance
    pub solver: ShootingMethodSolver,
    /// Type of IVP solver to use (RK45, BDF, Radau, etc.)
    pub ivp_solver: SolverType,
    /// Parameters for the IVP solver
    pub solver_params: HashMap<String, SolverParam>,
    /// Domain boundaries (start, end)
    pub borders: (f64, f64),
    /// Solution matrix (rows = variables, columns = time steps)
    pub solution: DMatrix<f64>,
    /// Complete solution result including mesh and boundary values
    pub full_sol: ShootingMethodResult,
}
impl BVPShooting {
    /// Create a new BVP shooting method solver instance.
    ///
    /// # Arguments
    /// * `eq_vec` - Vector of symbolic expressions representing the ODE system
    /// * `values` - Names of dependent variables (e.g., ["y0", "y1"])
    /// * `arg` - Name of independent variable (e.g., "x")
    /// * `BoundaryConditions` - Map of boundary conditions
    /// * `borders` - Domain boundaries (start, end)
    pub fn new(
        eq_vec: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        BoundaryConditions: HashMap<String, Vec<(usize, f64)>>,
        borders: (f64, f64),
    ) -> Self {
        // init_logger();
        BVPShooting {
            eq_vec,
            values,
            arg,
            solver: ShootingMethodSolver::new(),
            ivp_solver: SolverType::BDF,
            solver_params: HashMap::new(),
            BoundaryConditions: BoundaryConditions,
            borders: borders,
            solution: DMatrix::zeros(0, 0),
            full_sol: ShootingMethodResult::default(),
        }
    }
    /// Set the IVP solver type and parameters.
    ///
    /// # Arguments
    /// * `ivp_solver` - Type of IVP solver (RK45, BDF, Radau, etc.)
    /// * `params` - Solver-specific parameters
    pub fn set_parameters(&mut self, ivp_solver: SolverType, params: HashMap<String, SolverParam>) {
        self.ivp_solver = ivp_solver;
        self.solver_params = params;
    }
    pub fn generate_eq_closure(&self) -> Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>> {
        let eq_vec = self.eq_vec.clone();
        let values = self.values.clone();
        let arg = self.arg.clone();
        let n = eq_vec.len();
        let residual = Box::new(move |t: f64, y: &DVector<f64>| -> DVector<f64> {
            let mut args = Vec::with_capacity(1 + n);
            args.push(t);
            args.extend(y.iter().cloned());
            let mut res = DVector::zeros(n);
            let mut all_var_names = vec![arg.as_str()];
            all_var_names.extend(values.iter().map(|s| s.as_str()));
            for i in 0..n {
                res[i] = eq_vec[i].eval_expression(all_var_names.clone(), &args);
            }
            res
        });
        residual
    }
    //

    pub fn BC_create(&self) -> (BoundaryCondition, BoundaryCondition) {
        let values = &self.values;
        let map_of_bc = &self.BoundaryConditions;

        let mut left_bc: Option<BoundaryCondition> = None;
        let mut right_bc: Option<BoundaryCondition> = None;

        // Process all boundary conditions
        for var_name in values {
            if let Some(conditions) = map_of_bc.get(var_name) {
                let var_idx = values.iter().position(|v| v == var_name).unwrap();
                for (boundary_idx, value) in conditions {
                    // For shooting method: y0 is position (Dirichlet), y1 is derivative (Neumann)
                    let bc_type: BoundaryConditionType = if var_idx == 0 {
                        BoundaryConditionType::Dirichlet // y0 -> position boundary condition
                    } else {
                        BoundaryConditionType::Neumann // y1 -> derivative boundary condition
                    };

                    let bc = BoundaryCondition {
                        value: *value,
                        bc_type: bc_type,
                    };

                    if *boundary_idx == 0 {
                        // 0 means left boundary - only set if not already set
                        if left_bc.is_none() {
                            left_bc = Some(bc);
                        }
                    } else {
                        // 1 means right boundary - only set if not already set
                        if right_bc.is_none() {
                            right_bc = Some(bc);
                        }
                    }
                }
            }
        }

        // Return with defaults if not set
        let left = left_bc.unwrap_or(BoundaryCondition {
            value: 0.0,
            bc_type: BoundaryConditionType::Dirichlet,
        });
        let right = right_bc.unwrap_or(BoundaryCondition {
            value: 0.0,
            bc_type: BoundaryConditionType::Dirichlet,
        });

        debug!("Left BC: value={}, type={:?}", left.value, left.bc_type);
        debug!("Right BC: value={}, type={:?}", right.value, right.bc_type);

        (left, right)
    }
    /// Solve the BVP using the shooting method with built-in RK45 IVP solver.
    ///
    /// This is the simplest interface that uses a fixed RK45 solver for the IVP.
    ///
    /// # Arguments
    /// * `initial_guess` - Initial guess for unknown boundary condition
    /// * `tolerance` - Convergence tolerance for shooting method
    /// * `max_iterations` - Maximum iterations for shooting method
    /// * `step_size` - Step size for RK45 integration
    pub fn simple_solve(
        &mut self,
        initial_guess: f64,
        tolerance: f64,
        max_iterations: usize,
        step_size: f64,
    ) {
        let ode_system = self.generate_eq_closure();
        let (a, b) = self.borders;
        let (left_bc, right_bc) = self.BC_create();

        let problem = BoundaryValueProblem::new(ode_system, a, b, left_bc, right_bc);

        let mut solver = ShootingMethodSolver {
            initial_guess,
            tolerance,
            max_iterations,
            step_size,
            result: ShootingMethodResult::default(),
        };
        let result = solver.solve(&problem, |x0, y0, x_end, h| {
            RK45_ivp_solver_and_mesh(x0, y0, x_end, h, &problem.ode_system)
        });
        match result {
            Ok(result_struct) => {
                self.full_sol = result_struct.clone();
                let solution = result_struct.y;
                self.solution = solution;
            }
            Err(e) => {
                println!("Error: {}", e);
            }
        }
        //self.solution = solution;
    }

    /// Solve the BVP using the shooting method with a specified IVP solver.
    ///
    /// This method allows you to choose any available IVP solver (RK45, BDF, Radau, etc.)
    /// and configure its parameters.
    ///
    /// # Arguments
    /// * `initial_guess` - Initial guess for unknown boundary condition
    /// * `tolerance` - Convergence tolerance for shooting method
    /// * `max_iterations` - Maximum iterations for shooting method
    /// * `step_size` - Step size for integration
    /// * `ivp_solver` - Type of IVP solver to use
    /// * `params` - Parameters specific to the chosen IVP solver
    pub fn solve_with_certain_ivp(
        &mut self,
        initial_guess: f64,
        tolerance: f64,
        max_iterations: usize,
        step_size: f64,
        ivp_solver: SolverType,
        params: HashMap<String, SolverParam>,
    ) {
        let ode_system = self.generate_eq_closure();
        let (t0, tend) = self.borders;
        let (left_bc, right_bc) = self.BC_create();

        let problem = BoundaryValueProblem::new(ode_system, t0, tend, left_bc, right_bc);
        let ivp_closure = Self::creating_IVP_closure(
            self.eq_vec.clone(),
            self.values.clone(),
            self.arg.clone(),
            ivp_solver.clone(),
            params,
        );

        let mut solver = ShootingMethodSolver {
            initial_guess,
            tolerance,
            max_iterations,
            step_size,
            result: ShootingMethodResult::default(),
        };

        let result = solver.solve(&problem, |x0, y0, x_end, h| {
            let (y_matrix, x_vec) = ivp_closure(x0, y0, x_end, h);
            (y_matrix, x_vec)
        });

        match result {
            Ok(result_struct) => {
                //  println!("str{:?}", result_struct);
                self.full_sol = result_struct.clone();
                let solution = result_struct.y;
                self.solution = solution;
            }
            Err(e) => {
                println!("Error: {}", e);
            }
        }
    }
    /// Get the complete solution result including mesh, solution matrix, and boundary values.
    pub fn get_solution(&self) -> ShootingMethodResult {
        return self.full_sol.clone();
    }

    /// Get the spatial mesh (x-coordinates) of the solution.
    pub fn get_x(&self) -> DVector<f64> {
        return self.full_sol.x_mesh.clone();
    }

    /// Get the solution matrix where rows are variables and columns are spatial points.
    pub fn get_y(&self) -> DMatrix<f64> {
        return self.full_sol.y.clone();
    }
    pub fn IVP_solver(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        solver_type: SolverType,
        params: HashMap<String, SolverParam>,
        step: f64,
    ) -> Result<(DVector<f64>, DMatrix<f64>), String> {
        let mut solver =
            UniversalODESolver::new(eq_system, values, arg, solver_type, t0, y0, t_bound);
        let mut params = params;
        let add_step_size = HashMap::from([("step_size".to_string(), SolverParam::Float(step))]);
        params.extend(add_step_size);
        info!("solving with params {:?}", params);
        solver.set_parameters(params);
        solver.initialize();
        solver.solve();
        let (x, y) = solver.get_result();
        match (x, y) {
            (Some(x_vec), Some(y_mat)) => Ok((x_vec, y_mat)),
            _ => Err("IVP solver failed to produce results".to_string()),
        }
    }

    pub fn creating_IVP_closure(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        solver_type: SolverType,
        params: HashMap<String, SolverParam>,
    ) -> Box<dyn Fn(f64, DVector<f64>, f64, f64) -> (DMatrix<f64>, DVector<f64>)> {
        let f: Box<dyn Fn(f64, DVector<f64>, f64, f64) -> (DMatrix<f64>, DVector<f64>) + '_> =
            Box::new(
                move |t0: f64, ivp_initial_condition: DVector<f64>, t_end, step| {
                    info!("\n ivp_initial_condition: {:}", ivp_initial_condition);
                    match Self::IVP_solver(
                        eq_system.clone(),
                        values.clone(),
                        arg.clone(),
                        t0,
                        ivp_initial_condition,
                        t_end,
                        solver_type.clone(),
                        params.clone(),
                        step,
                    ) {
                        Ok((x_vec, y_mat)) => {
                            let y_mat = y_mat.transpose();
                            (y_mat, x_vec)
                        }
                        Err(e) => {
                            eprintln!("IVP solver error: {}", e);
                            panic!("IVP solver failed: {}", e);
                        }
                    }
                },
            );
        f
    }
}
#[allow(dead_code)]
fn init_logger() {
    let _ = SimpleLogger::init(LevelFilter::Debug, Config::default());
}
