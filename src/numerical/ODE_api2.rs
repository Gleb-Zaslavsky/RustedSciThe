//! # Universal ODE Solver API
//!
//! This module provides a unified interface for solving ordinary differential equations (ODEs)
//! using different numerical methods. It supports both stiff and non-stiff problems.
//!
//! ## Supported Solvers
//!
//! ### Non-stiff Methods (for smooth, non-stiff problems)
//! - **RK45**: 4th-order Runge-Kutta method with adaptive step size
//! - **DOPRI**: Dormand-Prince method (high-order explicit)
//! - **AB4**: Adams-Bashforth 4th order method
//!
//! ### Stiff Methods (for stiff/implicit problems)
//! - **Radau**: Implicit Runge-Kutta method (Order 3, 5, or 7)
//! - **BDF**: Backward Differentiation Formula (variable order)
//! - **Backward Euler**: Simple implicit method
//!
//! ## Quick Start Guide
//!
//! ### Step 1: Define your ODE system
//! ```rust, ignore
//! use RustedSciThe::symbolic::symbolic_engine::Expr;
//! use nalgebra::DVector;
//!
//! // Example: y' = -2*y + sin(t), y(0) = 1
//! let eq1 = Expr::parse_expression("-2*y + sin(t)");
//! let eq_system = vec![eq1];
//! let values = vec!["y".to_string()];
//! let arg = "t".to_string();
//! let t0 = 0.0;
//! let y0 = DVector::from_vec(vec![1.0]);
//! let t_bound = 5.0;
//! ```
//!
//! ### Step 2: Choose and create a solver
//!
//! #### For non-stiff problems (smooth solutions):
//! ```rust, ignore
//! use RustedSciThe::numerical::ODE_api2::UniversalODESolver;
//!
//! // RK45 - good general purpose solver
//! let mut solver = UniversalODESolver::rk45(
//!     eq_system, values, arg, t0, y0, t_bound, 1e-4 // step_size
//! );
//!
//! // DOPRI - high accuracy explicit method
//! let mut solver = UniversalODESolver::dopri(
//!     eq_system, values, arg, t0, y0, t_bound, 1e-4
//! );
//! ```
//!
//! #### For stiff problems (rapid changes, chemical kinetics, etc.):
//! ```rust, ignore
//! use RustedSciThe::numerical::Radau::Radau_main::RadauOrder;
//!
//! // BDF - best for very stiff problems
//! let mut solver = UniversalODESolver::bdf(
//!     eq_system, values, arg, t0, y0, t_bound,
//!     1e-3, // max_step
//!     1e-6, // rtol
//!     1e-8  // atol
//! );
//!
//! // Radau - high-order implicit method
//! let mut solver = UniversalODESolver::radau(
//!     eq_system, values, arg, RadauOrder::Order5, t0, y0, t_bound,
//!     1e-6, // tolerance
//!     50,   // max_iterations
//!     Some(1e-3) // step_size (optional)
//! );
//! ```
//!
//! ### Step 3: Solve and get results
//! ```rust, ignore
//! solver.solve();
//! let (t_result, y_result) = solver.get_result();
//!
//! // Plot results
//! solver.plot_result();
//!
//! // Save to file (if supported)
//! solver.save_result().unwrap();
//! ```
//!
//! ## Advanced Usage
//!
//! ### Custom parameters for specific solvers:
//! ```rust, ignore
//! use RustedSciThe::numerical::ODE_api2::{UniversalODESolver, SolverType};
//!
//! let mut solver = UniversalODESolver::new(
//!     eq_system, values, arg, SolverType::Radau(RadauOrder::Order3),
//!     t0, y0, t_bound
//! );
//!
//! // Set Radau-specific parameters
//!
//! solver.set_tolerance(1e-8);
//! solver.set_parallel(true);  // Enable parallel processing
//! solver.set_max_iterations(100);
//!
//! solver.solve();
//! ```
//!
//! ## Solver Selection Guide
//!
//! | Problem Type | Recommended Solver | When to Use |
//! |--------------|-------------------|-------------|
//! | Smooth, non-stiff | RK45 or DOPRI | General purpose, oscillatory systems |
//! | Mildly stiff | Backward Euler | Simple implicit problems |
//! | Very stiff | BDF | Chemical kinetics, fast transients |
//! | High accuracy needed | Radau Order5/7 | Precision-critical applications |
//! | Large systems | BDF with sparse | Many variables, sparse Jacobian |
//!
//! ## System of Equations Example
//! ```rust, ignore
//! // System: y1' = -2*y1 + y2, y2' = y1 - 2*y2
//! RustedSciThe::numerical::ODE_api2::UniversalODESolver;
//! use nalgebra::{DMatrix, DVector};
//! let eq1 = Expr::parse_expression("-2*y1 + y2");
//! let eq2 = Expr::parse_expression("y1 - 2*y2");
//! let eq_system = vec![eq1, eq2];
//! let values = vec!["y1".to_string(), "y2".to_string()];
//! let y0 = DVector::from_vec(vec![1.0, 0.0]);
//!
//! let mut solver = UniversalODESolver::rk45(
//!     eq_system, values, "t".to_string(), 0.0, y0, 10.0, 1e-4
//! );
//! solver.solve();
//! ```

use crate::symbolic::symbolic_engine::Expr;
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

// Import all solver types
use crate::numerical::BDF::BDF_api::ODEsolver as BDFSolver;
use crate::numerical::BE::BE;
use crate::numerical::NonStiff_api::nonstiffODE;
use crate::numerical::Radau::Radau_main::{Radau, RadauOrder};

/// Universal ODE solver that can handle all solver types
pub struct UniversalODESolver {
    // Common parameters
    eq_system: Vec<Expr>,
    values: Vec<String>,
    arg: String,
    t0: f64,
    y0: DVector<f64>,
    t_bound: f64,

    // Solver type and instance
    solver_type: SolverType,
    solver_instance: Option<SolverInstance>,

    // Results
    t_result: Option<DVector<f64>>,
    y_result: Option<DMatrix<f64>>,

    // Solver-specific parameters
    solver_params: HashMap<String, SolverParam>,
}

/// Enum for different solver types
#[derive(Clone, Debug)]
pub enum SolverType {
    NonStiff(String), // RK45, DOPRI, AB4
    Radau(RadauOrder),
    BDF,
    BackwardEuler,
}

/// Enum for solver instances
pub enum SolverInstance {
    NonStiff(nonstiffODE),
    Radau(Radau),
    BDF(BDFSolver),
    BE(BE),
}

/// Enum for different parameter types
#[derive(Clone, Debug)]
pub enum SolverParam {
    Float(f64),
    Int(usize),
    Bool(bool),
    OptionalFloat(Option<f64>),
    OptionalInt(Option<usize>),
    OptionalMatrix(Option<DMatrix<f64>>),
}

impl UniversalODESolver {
    /// Create a new universal ODE solver
    pub fn new(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        solver_type: SolverType,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
    ) -> Self {
        UniversalODESolver {
            eq_system,
            values,
            arg,
            t0,
            y0,
            t_bound,
            solver_type,
            solver_instance: None,
            t_result: None,
            y_result: None,
            solver_params: HashMap::new(),
        }
    }

    /// Set solver-specific parameters
    pub fn set_parameter(&mut self, key: &str, value: SolverParam) {
        self.solver_params.insert(key.to_string(), value);
    }

    pub fn set_parameters(&mut self, params: HashMap<String, SolverParam>) {
        self.solver_params.extend(params);
    }

    /// Set step size (common parameter)
    pub fn set_step_size(&mut self, step: f64) {
        self.set_parameter("step_size", SolverParam::Float(step));
    }

    /// Set tolerance (for stiff solvers)
    pub fn set_tolerance(&mut self, tolerance: f64) {
        self.set_parameter("tolerance", SolverParam::Float(tolerance));
    }

    /// Set maximum iterations (for iterative solvers)
    pub fn set_max_iterations(&mut self, max_iter: usize) {
        self.set_parameter("max_iterations", SolverParam::Int(max_iter));
    }

    /// Set relative tolerance (for BDF)
    pub fn set_rtol(&mut self, rtol: f64) {
        self.set_parameter("rtol", SolverParam::Float(rtol));
    }

    /// Set absolute tolerance (for BDF)
    pub fn set_atol(&mut self, atol: f64) {
        self.set_parameter("atol", SolverParam::Float(atol));
    }

    /// Set maximum step size (for BDF)
    pub fn set_max_step(&mut self, max_step: f64) {
        self.set_parameter("max_step", SolverParam::Float(max_step));
    }

    /// Set first step (for BDF)
    pub fn set_first_step(&mut self, first_step: Option<f64>) {
        self.set_parameter("first_step", SolverParam::OptionalFloat(first_step));
    }

    /// Set vectorized flag (for BDF)
    pub fn set_vectorized(&mut self, vectorized: bool) {
        self.set_parameter("vectorized", SolverParam::Bool(vectorized));
    }

    /// Set Jacobian sparsity (for BDF)
    pub fn set_jac_sparsity(&mut self, jac_sparsity: Option<DMatrix<f64>>) {
        self.set_parameter("jac_sparsity", SolverParam::OptionalMatrix(jac_sparsity));
    }

    /// Set parallel processing (for Radau)
    pub fn set_parallel(&mut self, parallel: bool) {
        self.set_parameter("parallel", SolverParam::Bool(parallel));
    }

    /// Initialize the solver based on type and parameters
    pub fn initialize(&mut self) {
        match &self.solver_type {
            SolverType::NonStiff(method) => {
                let step = self.get_float_param("step_size").unwrap_or(1e-3);

                let mut solver = nonstiffODE::new(
                    self.eq_system.clone(),
                    self.values.clone(),
                    self.arg.clone(),
                    method.clone(),
                    self.t0,
                    self.y0.clone(),
                    self.t_bound,
                    step,
                );

                solver.generate();
                self.solver_instance = Some(SolverInstance::NonStiff(solver));
            }

            SolverType::Radau(order) => {
                let mut solver = Radau::new(order.clone());

                let tolerance = self.get_float_param("tolerance").unwrap_or(1e-6);
                let max_iterations = self.get_int_param("max_iterations").unwrap_or(50);
                let step = self.get_optional_float_param("step_size");

                solver.set_initial(
                    self.eq_system.clone(),
                    self.values.clone(),
                    self.arg.clone(),
                    tolerance,
                    max_iterations,
                    step,
                    self.t0,
                    self.t_bound,
                    self.y0.clone(),
                );

                // Set parallel processing if specified
                if let Some(parallel) = self.get_bool_param("parallel") {
                    solver.set_parallel(parallel);
                }

                self.solver_instance = Some(SolverInstance::Radau(solver));
            }

            SolverType::BDF => {
                let max_step = self.get_float_param("max_step").unwrap_or(1e-3);
                let rtol = self.get_float_param("rtol").unwrap_or(1e-5);
                let atol = self.get_float_param("atol").unwrap_or(1e-5);
                let vectorized = self.get_bool_param("vectorized").unwrap_or(false);
                let first_step = self.get_optional_float_param("first_step");
                let jac_sparsity = self.get_optional_matrix_param("jac_sparsity");
                //  println!("max step {}", max_step);
                // println!("params {:?}", self.solver_params);
                let mut solver = BDFSolver::new(
                    self.eq_system.clone(),
                    self.values.clone(),
                    self.arg.clone(),
                    "BDF".to_string(),
                    self.t0,
                    self.y0.clone(),
                    self.t_bound,
                    max_step,
                    rtol,
                    atol,
                    jac_sparsity,
                    vectorized,
                    first_step,
                );

                solver.generate();
                self.solver_instance = Some(SolverInstance::BDF(solver));
            }

            SolverType::BackwardEuler => {
                let mut solver = BE::new();

                let tolerance = self.get_float_param("tolerance").unwrap_or(1e-6);
                let max_iterations = self.get_int_param("max_iterations").unwrap_or(50);
                let step = self.get_optional_float_param("step_size");

                solver.set_initial(
                    self.eq_system.clone(),
                    self.values.clone(),
                    self.arg.clone(),
                    tolerance,
                    max_iterations,
                    step,
                    self.t0,
                    self.t_bound,
                    self.y0.clone(),
                );

                self.solver_instance = Some(SolverInstance::BE(solver));
            }
        }
    }

    /// Solve the ODE system
    pub fn solve(&mut self) {
        if self.solver_instance.is_none() {
            self.initialize();
        }

        match self.solver_instance.as_mut().unwrap() {
            SolverInstance::NonStiff(solver) => {
                solver.solve();
                let (t_result, y_result) = solver.get_result();
               // let y_result = y_result.transpose();
                self.t_result = Some(t_result);
                self.y_result = Some(y_result);
            }

            SolverInstance::Radau(solver) => {
                solver.solve();
                let (t_result, y_result) = solver.get_result();
                self.t_result = t_result;
                self.y_result = y_result;
            }

            SolverInstance::BDF(solver) => {
                solver.solve();
                let (t_result, y_result) = solver.get_result();
                self.t_result = Some(t_result);
                self.y_result = Some(y_result);
            }

            SolverInstance::BE(solver) => {
                solver.solve();
                let (t_result, y_result) = solver.get_result();
                self.t_result = t_result;
                self.y_result = y_result;
            }
        }
    }

    /// Get results
    pub fn get_result(&self) -> (Option<DVector<f64>>, Option<DMatrix<f64>>) {
        (self.t_result.clone(), self.y_result.clone())
    }

    /// Plot results
    pub fn plot_result(&self) {
        if let (Some(_t_result), Some(_y_result)) = (&self.t_result, &self.y_result) {
            match self.solver_instance.as_ref().unwrap() {
                SolverInstance::NonStiff(solver) => solver.plot_result(),
                SolverInstance::Radau(solver) => solver.plot_result(),
                SolverInstance::BDF(solver) => solver.plot_result(),
                SolverInstance::BE(solver) => solver.plot_result(),
            }
        }
    }

    /// Save results
    pub fn save_result(&self) -> Result<(), Box<dyn std::error::Error>> {
        match self.solver_instance.as_ref().unwrap() {
            SolverInstance::NonStiff(solver) => solver.save_result(),
            SolverInstance::BDF(solver) => solver.save_result(),
            _ => Ok(()), // Other solvers don't have save_result implemented
        }
    }

    // Helper methods for parameter extraction
    fn get_float_param(&self, key: &str) -> Option<f64> {
        match self.solver_params.get(key) {
            Some(SolverParam::Float(val)) => Some(*val),
            _ => None,
        }
    }

    fn get_int_param(&self, key: &str) -> Option<usize> {
        match self.solver_params.get(key) {
            Some(SolverParam::Int(val)) => Some(*val),
            _ => None,
        }
    }

    fn get_bool_param(&self, key: &str) -> Option<bool> {
        match self.solver_params.get(key) {
            Some(SolverParam::Bool(val)) => Some(*val),
            _ => None,
        }
    }

    fn get_optional_float_param(&self, key: &str) -> Option<f64> {
        match self.solver_params.get(key) {
            Some(SolverParam::OptionalFloat(val)) => *val,
            Some(SolverParam::Float(val)) => Some(*val),
            _ => None,
        }
    }

    fn get_optional_matrix_param(&self, key: &str) -> Option<DMatrix<f64>> {
        match self.solver_params.get(key) {
            Some(SolverParam::OptionalMatrix(val)) => val.clone(),
            _ => None,
        }
    }
}

/// Convenience functions for creating solvers
impl UniversalODESolver {
    /// Create RK45 solver
    pub fn rk45(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        step_size: f64,
    ) -> Self {
        let mut solver = Self::new(
            eq_system,
            values,
            arg,
            SolverType::NonStiff("RK45".to_string()),
            t0,
            y0,
            t_bound,
        );
        solver.set_step_size(step_size);
        solver
    }

    /// Create DOPRI solver
    pub fn dopri(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        step_size: f64,
    ) -> Self {
        let mut solver = Self::new(
            eq_system,
            values,
            arg,
            SolverType::NonStiff("DOPRI".to_string()),
            t0,
            y0,
            t_bound,
        );
        solver.set_step_size(step_size);
        solver
    }
    /// Create DOPRI solver
    pub fn ab4(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        step_size: f64,
    ) -> Self {
        let mut solver = Self::new(
            eq_system,
            values,
            arg,
            SolverType::NonStiff("AB4".to_string()),
            t0,
            y0,
            t_bound,
        );
        solver.set_step_size(step_size);
        solver
    }

    /// Create Radau solver
    pub fn radau(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        order: RadauOrder,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        tolerance: f64,
        max_iterations: usize,
        step_size: Option<f64>,
    ) -> Self {
        let mut solver = Self::new(
            eq_system,
            values,
            arg,
            SolverType::Radau(order),
            t0,
            y0,
            t_bound,
        );
        solver.set_tolerance(tolerance);
        solver.set_max_iterations(max_iterations);
        if let Some(step) = step_size {
            solver.set_step_size(step);
        }
        solver
    }

    /// Create BDF solver
    pub fn bdf(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        max_step: f64,
        rtol: f64,
        atol: f64,
    ) -> Self {
        let mut solver = Self::new(eq_system, values, arg, SolverType::BDF, t0, y0, t_bound);
        solver.set_max_step(max_step);
        solver.set_rtol(rtol);
        solver.set_atol(atol);
        solver
    }

    /// Create Backward Euler solver
    pub fn backward_euler(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        tolerance: f64,
        max_iterations: usize,
        step_size: Option<f64>,
    ) -> Self {
        let mut solver = Self::new(
            eq_system,
            values,
            arg,
            SolverType::BackwardEuler,
            t0,
            y0,
            t_bound,
        );
        solver.set_tolerance(tolerance);
        solver.set_max_iterations(max_iterations);
        if let Some(step) = step_size {
            solver.set_step_size(step);
        }
        solver
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Tests
/// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;


    #[test]
    fn test_universal_rk45() {
        let eq1 = Expr::parse_expression("-y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 1.0;
        let step_size = 1e-4;

        let mut solver =
            UniversalODESolver::rk45(eq_system, values, arg, t0, y0, t_bound, step_size);

        solver.solve();
        let (t_result, y_result) = solver.get_result();

        assert!(t_result.is_some());
        assert!(y_result.is_some());

        let y_final = y_result.clone().unwrap()[(y_result.as_ref().unwrap().nrows() - 1, 0)];
        let expected = (-1.0_f64).exp();
        assert_relative_eq!(y_final, expected, epsilon = 1e-2);
    }
    #[test]
    fn test_rk45_exponential() {
        // y'' - y = 0,
        // y0' =y1,
        // y1' = y0
        // y(0) = 0 y'(0) = 1
        // solution y(x) = 1/2 (e^x - e^(-x))

        let eq_system = vec![Expr::parse_expression("y1"), Expr::parse_expression("y0")];
        let values = vec!["y0".to_string(), "y1".to_string()];
        let arg = "x".to_string();
        let t0 = 0.0;

        let y0 = DVector::from_vec(vec![0.0, 1.0]);
        let t_bound = 0.5;
        let step = 1e-3;
        let solver_type = SolverType::NonStiff("RK45".to_owned());
        let mut solver =
            UniversalODESolver::new(eq_system, values, arg, solver_type, t0, y0, t_bound);
        let mut params = HashMap::new();
        let add_step_size = HashMap::from([("step_size".to_string(), SolverParam::Float(step))]);
        params.extend(add_step_size);

        solver.set_parameters(params);
        solver.initialize();
        solver.solve();
        let (t_result, y_result) = solver.get_result();
        let y_final = y_result.clone().unwrap();
        let x_mesh = t_result.clone().unwrap();
        let y0: DVector<f64> = y_final.column(0).into();
        // println!("{:?}", y0);
        for i in 0..y0.len() {
            let y = y0[i];
            let x = x_mesh[i];
            let expected = 0.5 * (x.exp() - (-x).exp());
            assert_relative_eq!(y, expected, epsilon = 1e-4);
        }
    }
    #[test]
    fn test_ab4_cos() {


 // y'' + y = 0,
 // y0' =y1,
 // y1' =- y0 
 // y(0) = 1, y'(0) = 0 (solution: y = cos(x))
        let eq_system = vec![Expr::parse_expression("y1"), Expr::parse_expression("-y0")];
        let values = vec!["y0".to_string(), "y1".to_string()];
        let arg = "x".to_string();
        let t0 = 0.0;
                                                                    // cos(0)=1, -sin(0)=0 
        let y0 = DVector::from_vec(vec![1.0, 0.0]);
        let t_bound = std::f64::consts::PI ;
        let step = 1e-5;
        let solver_type = SolverType::NonStiff("AB4".to_owned());
        let mut solver =
            UniversalODESolver::new(eq_system, values, arg, solver_type, t0, y0, t_bound);
        let mut params = HashMap::new();
        let add_step_size = HashMap::from([("step_size".to_string(), SolverParam::Float(step))]);
        params.extend(add_step_size);

        solver.set_parameters(params);
        solver.initialize();
        solver.solve();
        let (t_result, y_result) = solver.get_result();
        let y_final = y_result.clone().unwrap();
        let x_mesh = t_result.clone().unwrap();
        let y0: DVector<f64> = y_final.column(0).into();
        let y1: DVector<f64> = y_final.column(1).into();
       // println!("{:?} \n {:?}", y0, y1);
        for i in 0..y0.len() {
            let y = y0[i];
            let x = x_mesh[i];
            let expected = x.cos();
            let expected1 = -x.sin();
            assert_relative_eq!(y, expected, epsilon = 1e-4);
            assert_relative_eq!(y1[i], expected1, epsilon = 1e-4);
        }
    }
    #[test]
    fn test_rk45_cos2() {


 // y'' + y = 0,
 // y0' =y1,
 // y1' =- y0 
 // y(0) = 1, y'(0) = 0 (solution: y = cos(x))
        let eq_system = vec![Expr::parse_expression("y1"), Expr::parse_expression("-y0")];
        let values = vec!["y0".to_string(), "y1".to_string()];
        let arg = "x".to_string();
        let t0 = 0.0;
                                                                    // cos(0)=1, -sin(0)=0 
        let y0 = DVector::from_vec(vec![1.0, 0.0]);
        let t_bound = std::f64::consts::PI ;
        let step = 1e-5;
        let solver_type = SolverType::NonStiff("RK45".to_owned());
        let mut solver =
            UniversalODESolver::new(eq_system, values, arg, solver_type, t0, y0, t_bound);
        let mut params = HashMap::new();
        let add_step_size = HashMap::from([("step_size".to_string(), SolverParam::Float(step))]);
        params.extend(add_step_size);

        solver.set_parameters(params);
        solver.initialize();
        solver.solve();
        let (t_result, y_result) = solver.get_result();
        let y_final = y_result.clone().unwrap();
        let x_mesh = t_result.clone().unwrap();
        let y0: DVector<f64> = y_final.column(0).into();
        let y1: DVector<f64> = y_final.column(1).into();
       // println!("{:?} \n {:?}", y0, y1);
        for i in 0..y0.len() {
            let y = y0[i];
            let x = x_mesh[i];
            let expected = x.cos();
            let expected1 = -x.sin();
            assert_relative_eq!(y, expected, epsilon = 1e-4);
            assert_relative_eq!(y1[i], expected1, epsilon = 1e-4);
        }
    }
    #[test]
    fn test_universal_radau() {
        let eq1 = Expr::parse_expression("-y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 0.5;

        let mut solver = UniversalODESolver::radau(
            eq_system,
            values,
            arg,
            RadauOrder::Order3,
            t0,
            y0,
            t_bound,
            1e-6,
            50,
            Some(1e-3),
        );

        solver.solve();
        let (t_result, y_result) = solver.get_result();

        assert!(t_result.is_some());
        assert!(y_result.is_some());
    }

    #[test]
    fn test_universal_bdf() {
        let eq1 = Expr::parse_expression("-y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 0.5;

        let mut solver =
            UniversalODESolver::bdf(eq_system, values, arg, t0, y0, t_bound, 1e-3, 1e-5, 1e-5);

        solver.solve();
        let (t_result, y_result) = solver.get_result();

        assert!(t_result.is_some());
        assert!(y_result.is_some());
    }
    #[test]
    fn test_bdf_exponential() {
        // y'' - y = 0,
        // y0' =y1,
        // y1' = y0
        // y(0) = 0 y' = 1
        // solution y(x) = 1/2 (e^x - e^(-x))

        let eq_system = vec![Expr::parse_expression("y1"), Expr::parse_expression("y0")];
        let values = vec!["y0".to_string(), "y1".to_string()];
        let arg = "x".to_string();
        let t0 = 0.0;

        let y0 = DVector::from_vec(vec![0.0, 1.0]);
        let t_bound = 0.5;

        let mut solver =
            UniversalODESolver::bdf(eq_system, values, arg, t0, y0, t_bound, 1e-3, 1e-5, 1e-5);

        solver.solve();
        let (t_result, y_result) = solver.get_result();
        let y_final = y_result.clone().unwrap();
        let x_mesh = t_result.clone().unwrap();
        let y0: DVector<f64> = y_final.column(0).into();
        // println!("{:?}", y0);
        for i in 0..y0.len() {
            let y = y0[i];
            let x = x_mesh[i];
            let expected = 0.5 * (x.exp() - (-x).exp());
            assert_relative_eq!(y, expected, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_bdf_linear() {
        // y'' = 0, y(0) = 0, y(1) = 1 (solution: y = x)
        // y0' = y1
        // y1' = 0
        //
        let eq_vec = vec![Expr::parse_expression("y1"), Expr::parse_expression("0")];
        let values = vec!["y0".to_string(), "y1".to_string()];
        let arg = "x".to_string();

        let y0 = DVector::from_vec(vec![0.0, 0.9997384083916304]);
        let t0 = 0.0;
        let t_bound = 1.0;

        let mut solver =
            UniversalODESolver::bdf(eq_vec, values, arg, t0, y0, t_bound, 1e-3, 1e-5, 1e-5);

        solver.solve();
        let (t_result, y_result) = solver.get_result();
        let y_final = y_result.clone().unwrap();
        let x_mesh = t_result.clone().unwrap();
        let y0: DVector<f64> = y_final.column(0).into();
        // println!("{:?}", y0);
        for i in 0..y0.len() {
            let y = y0[i];
            let x = x_mesh[i];
            let expected = x;
            assert_relative_eq!(y, expected, epsilon = 1e-3);
        }
    }
    #[test]
    fn test_universal_backward_euler() {
        let eq1 = Expr::parse_expression("-y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 0.5;

        let mut solver = UniversalODESolver::backward_euler(
            eq_system,
            values,
            arg,
            t0,
            y0,
            t_bound,
            1e-6,
            50,
            Some(1e-3),
        );

        solver.solve();
        let (t_result, y_result) = solver.get_result();

        assert!(t_result.is_some());
        assert!(y_result.is_some());
    }
    #[test]
    fn test_direct_setting() {
        let eq1 = Expr::parse_expression("-y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 0.5;

        let mut solver = UniversalODESolver::new(
            eq_system,
            values,
            arg,
            SolverType::Radau(RadauOrder::Order3),
            t0,
            y0,
            t_bound,
        );
        solver.set_max_iterations(100);
        solver.set_tolerance(1e-6);
        solver.set_step_size(1e-3);
        solver.initialize();
        solver.solve();
        let (t_result, y_result) = solver.get_result();

        assert!(t_result.is_some());
        assert!(y_result.is_some());
    }
}
