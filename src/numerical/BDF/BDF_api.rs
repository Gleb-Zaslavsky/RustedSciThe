//! # BDF ODE Solver API
//!
//! High-level interface for solving ordinary differential equations using the
//! Backward Differentiation Formula (BDF) method with symbolic expression support.
//!
//! ## Overview
//!
//! This module provides a user-friendly API for solving stiff and non-stiff ODEs
//! by combining symbolic expression parsing with the robust BDF numerical solver.
//! The solver automatically generates analytical Jacobians from symbolic expressions,
//! significantly improving performance and accuracy for stiff systems.
//!
//! ## Key Features
//!
//! - **Symbolic Integration**: Parse string expressions into ODEs
//! - **Automatic Jacobian**: Generate analytical Jacobians from symbolic expressions
//! - **Adaptive Methods**: Variable order (1-5) and step size control
//! - **Stop Conditions**: Terminate integration when variables reach target values
//! - **Result Export**: Save results to CSV and generate plots
//! - **Comprehensive Testing**: Extensive test suite with analytical comparisons
//!
//! ## Main Components
//!
//! ### ODEsolver
//! The primary struct that orchestrates the entire solving process:
//! - Parses symbolic expressions into numerical functions
//! - Generates analytical Jacobians automatically
//! - Manages BDF solver instance and integration loop
//! - Handles result storage and export
//!
//! ### Key Methods
//! - `new()`: Create solver with problem parameters
//! - `solve()`: Complete integration from t0 to t_bound
//! - `set_stop_condition()`: Set early termination conditions
//! - `get_result()`: Retrieve time and solution arrays
//! - `plot_result()`: Generate solution plots
//! - `save_result()`: Export results to CSV
//!
//! ## Usage Example
//!
//! ```rust
//! use crate::numerical::BDF::BDF_api::ODEsolver;
//! use crate::symbolic::symbolic_engine::Expr;
//! use nalgebra::DVector;
//!
//! // Define Van der Pol oscillator: y1' = y2, y2' = μ(1-y1²)y2 - y1
//! let eq1 = Expr::parse_expression("y2");
//! let eq2 = Expr::parse_expression("5*(1-y1*y1)*y2 - y1");
//! let eq_system = vec![eq1, eq2];
//! let values = vec!["y1".to_string(), "y2".to_string()];
//!
//! // Create solver
//! let mut solver = ODEsolver::new(
//!     eq_system,                    // System of ODEs
//!     values,                       // Variable names
//!     "t".to_string(),             // Independent variable
//!     "BDF".to_string(),           // Method
//!     0.0,                         // t0
//!     DVector::from_vec(vec![2.0, 0.0]), // y0
//!     10.0,                        // t_bound
//!     0.01,                        // max_step
//!     1e-6,                        // rtol
//!     1e-8,                        // atol
//!     None,                        // jac_sparsity
//!     false,                       // vectorized
//!     None,                        // first_step
//! );
//!
//! // Solve and get results
//! solver.solve();
//! let (t_result, y_result) = solver.get_result();
//!
//! // Optional: plot and save results
//! solver.plot_result();
//! solver.save_result().unwrap();
//! ```
//!
//! ## Advanced Features
//!
//! ### Stop Conditions
//! Terminate integration when variables reach specific values:
//! ```rust, ignore
//! let mut stop_condition = HashMap::new();
//! stop_condition.insert("y1".to_string(), 0.0);
//! solver.set_stop_condition(stop_condition);
//! ```
//!
//! ### Symbolic Expression Support
//! The solver supports complex mathematical expressions:
//! - Basic operations: `+`, `-`, `*`, `/`, `^`
//! - Functions: `sin`, `cos`, `exp`, `log`, `sqrt`
//! - Variables: Any alphanumeric identifier
//! - Constants: Numerical values
//!
//! ### Automatic Jacobian Generation
//! The solver automatically computes analytical Jacobians ∂f/∂y from symbolic
//! expressions, which is crucial for:
//! - **Stiff systems**: Implicit methods require accurate Jacobians
//! - **Performance**: Analytical Jacobians are much faster than finite differences
//! - **Accuracy**: Exact derivatives improve Newton convergence
//!
//! ## Implementation Notes
//!
//! ### Matrix Flattening Strategy
//! The solver uses an efficient matrix flattening approach for result storage:
//! ```rust, ignore
//! // Convert Vec<DVector<f64>> to DMatrix<f64>
//! let mut flat_vec: Vec<f64> = Vec::new();
//! for vector in y.iter() {
//!     flat_vec.extend(vector);
//! }
//! let y_res = DMatrix::from_vec(cols, rows, flat_vec).transpose();
//! ```
//!
//! ### Status Management
//! Integration status is tracked throughout the process:
//! - `"running"`: Integration in progress
//! - `"finished"`: Successfully reached t_bound
//! - `"failed"`: Integration failed (step size too small, etc.)
//! - `"stopped_by_condition"`: Terminated by user-defined stop condition
//!
//! ### Error Handling
//! The solver implements robust error handling:
//! - Graceful degradation when Newton iteration fails
//! - Automatic step size reduction for difficult regions
//! - Clear error messages for debugging

use crate::numerical::BDF::BDF_solver::BDF;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions::Jacobian;
extern crate nalgebra as na;
use crate::Utils::plots::plots;
use crate::numerical::BDF::common::NumberOrVec;
use na::{DMatrix, DVector};

use csv::Writer;
use std::collections::HashMap;
use std::time::Instant;

/// High-level ODE solver interface with symbolic expression support.
///
/// This struct provides a complete solution for solving ODEs defined as symbolic
/// expressions, automatically generating numerical functions and analytical Jacobians.
///
/// # Workflow
/// 1. **Setup**: Define ODE system as symbolic expressions
/// 2. **Generation**: Convert expressions to numerical functions and Jacobians
/// 3. **Integration**: Use BDF method with adaptive order/step control
/// 4. **Results**: Store, plot, and export solution data
///
/// # Key Features
/// - Automatic Jacobian generation from symbolic expressions
/// - Adaptive BDF method (orders 1-5) for stiff problems
/// - Stop conditions for event detection
/// - Built-in plotting and CSV export capabilities
pub struct ODEsolver {
    /// System of ODEs as symbolic expressions (e.g., ["y2", "-y1"])
    eq_system: Vec<Expr>,
    /// Variable names corresponding to solution components (e.g., ["y1", "y2"])
    values: Vec<String>,
    /// Independent variable name (typically "t" for time)
    arg: String,
    /// Numerical method identifier ("BDF" for this implementation)
    method: String,
    /// Initial time t₀
    t0: f64,
    /// Initial solution vector y₀
    y0: DVector<f64>,
    /// Final integration time
    t_bound: f64,
    /// Maximum allowed step size
    max_step: f64,
    /// Relative error tolerance
    rtol: f64,
    /// Absolute error tolerance
    atol: f64,
    /// Optional Jacobian sparsity pattern (not currently used)
    #[allow(dead_code)]
    jac_sparsity: Option<DMatrix<f64>>,
    /// Whether the ODE function supports vectorized evaluation
    vectorized: bool,
    /// Optional initial step size (auto-selected if None)
    first_step: Option<f64>,

    /// Current integration status: "running", "finished", "failed", "stopped_by_condition"
    status: String,
    /// Internal BDF solver instance
    Solver_instance: BDF,
    /// Optional error message from failed integration
    message: Option<String>,

    /// Time points of computed solution
    t_result: DVector<f64>,
    /// Solution matrix: rows = time points, columns = variables
    y_result: DMatrix<f64>,
    /// Optional stop conditions: variable_name → target_value
    stop_condition: Option<HashMap<String, f64>>,
}
impl ODEsolver {
    /// Creates a new ODE solver with the specified parameters.
    ///
    /// # Parameters
    /// * `eq_system` - Vector of symbolic expressions defining dy/dt = f(t,y)
    /// * `values` - Variable names corresponding to solution components
    /// * `arg` - Independent variable name (usually "t")
    /// * `method` - Solver method ("BDF" for this implementation)
    /// * `t0` - Initial time
    /// * `y0` - Initial solution vector
    /// * `t_bound` - Final integration time
    /// * `max_step` - Maximum step size
    /// * `rtol` - Relative tolerance for error control
    /// * `atol` - Absolute tolerance for error control
    /// * `jac_sparsity` - Optional Jacobian sparsity pattern
    /// * `vectorized` - Whether ODE function supports vectorized calls
    /// * `first_step` - Optional initial step size
    ///
    /// # Returns
    /// New ODEsolver instance ready for integration
    ///
    /// # Example
    /// ```rust, ignore
    /// let eq_system = vec![Expr::parse_expression("y2"),
    ///                      Expr::parse_expression("-y1")];
    /// let values = vec!["y1".to_string(), "y2".to_string()];
    /// let solver = ODEsolver::new(
    ///     eq_system, values, "t".to_string(), "BDF".to_string(),
    ///     0.0, DVector::from_vec(vec![1.0, 0.0]), 10.0,
    ///     0.01, 1e-6, 1e-8, None, false, None
    /// );
    /// ```
    pub fn new(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        method: String,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        max_step: f64,
        rtol: f64,
        atol: f64,
        jac_sparsity: Option<DMatrix<f64>>,
        vectorized: bool,
        first_step: Option<f64>,
    ) -> Self {
        let New = BDF::new();

        ODEsolver {
            eq_system,
            values,
            arg,
            method,
            t0,
            y0,
            t_bound,
            max_step,
            rtol,
            atol,

            jac_sparsity,
            vectorized,
            first_step,
            status: "running".to_string(),
            Solver_instance: New,
            message: None,

            t_result: DVector::zeros(1),
            y_result: DMatrix::zeros(1, 1),
            stop_condition: None,
        }
    }

    /// Sets stop conditions for early termination of integration.
    ///
    /// Integration will stop when any variable reaches its target value
    /// within the absolute tolerance.
    ///
    /// # Parameters
    /// * `stop_condition` - Map of variable names to target values
    ///
    /// # Example
    /// ```rust, ignore
    /// let mut stop_condition = HashMap::new();
    /// stop_condition.insert("y1".to_string(), 0.0);
    /// solver.set_stop_condition(stop_condition);
    /// ```
    pub fn set_stop_condition(&mut self, stop_condition: HashMap<String, f64>) {
        self.stop_condition = Some(stop_condition);
    }

    /// Checks if any stop condition has been met.
    ///
    /// # Parameters
    /// * `y` - Current solution vector
    ///
    /// # Returns
    /// `true` if any variable has reached its target value within tolerance
    fn check_stop_condition(&self, y: &DVector<f64>) -> bool {
        if let Some(ref conditions) = self.stop_condition {
            for (var_name, target_value) in conditions {
                if let Some(var_index) = self.values.iter().position(|v| v == var_name) {
                    let current_value = y[var_index];
                    if (current_value - target_value).abs() <= self.atol {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Generates numerical functions and Jacobian from symbolic expressions.
    ///
    /// This method:
    /// 1. Creates a Jacobian instance for symbolic processing
    /// 2. Converts symbolic expressions to numerical functions
    /// 3. Generates analytical Jacobian matrix function
    /// 4. Initializes the BDF solver with these functions
    ///
    /// # Implementation Details
    /// Uses the symbolic engine to automatically compute ∂f/∂y analytically,
    /// which is crucial for stiff problem performance.
    pub fn generate(&mut self) {
        let mut Jacobian_instance = Jacobian::new();
        Jacobian_instance.generate_IVP_ODEsolver(
            self.eq_system.clone(),
            self.values.clone(),
            self.arg.clone(),
        );
        let fun = Jacobian_instance.lambdified_functions_IVP_DVector;
        let jac = Jacobian_instance.function_jacobian_IVP_DMatrix;
        if self.method == "BDF" {
            let mut Solver_instance = BDF::new();
            Solver_instance.set_initial(
                fun,
                self.t0,
                self.y0.clone(),
                self.t_bound,
                self.max_step,
                NumberOrVec::Number(self.rtol),
                NumberOrVec::Number(self.atol),
                Some(jac),
                None,
                self.vectorized,
                self.first_step,
            );
            self.Solver_instance = Solver_instance;
        }
    }
    /// Performs a single integration step.
    ///
    /// This method wraps the BDF solver's step implementation and manages
    /// the integration status. It handles:
    /// - Boundary detection (reaching t_bound)
    /// - Error handling and status updates
    /// - Direction checking for integration completion
    ///
    /// # Status Updates
    /// - "finished": Successfully reached t_bound or boundary
    /// - "failed": Step failed (convergence issues, step size too small)
    /// - "running": Integration continues normally
    pub fn step(&mut self) {
        //  let (success, message_) =self.Solver_instance._step_impl();

        // Analogue of step function in https://github.com/scipy/scipy/blob/main/scipy/integrate/_ivp/base.py
        let t = self.Solver_instance.t;
        if t == self.t_bound {
            self.Solver_instance.t_old = Some(t);

            self.status = "finished".to_string();
        } else {
            let (success, message_) = self.Solver_instance._step_impl();
            if let Some(message_str) = message_ {
                self.message = Some(message_str.to_string());
            } else {
                self.message = None;
            }

            if success == false {
                self.status = "failed".to_string();
            } else {
                self.Solver_instance.t_old = Some(t);
                let _status: String = "running".to_string();
                if self.Solver_instance.direction * (self.Solver_instance.t - self.t_bound) >= 0.0 {
                    self.status = "finished".to_string();
                }
            }
        }
    }
    #[warn(unused_assignments)]
    /// Main integration loop that drives the solution from t0 to t_bound.
    ///
    /// This method implements the complete integration algorithm:
    /// 1. **Step Loop**: Repeatedly calls step() until completion
    /// 2. **Status Monitoring**: Tracks integration progress and failures
    /// 3. **Stop Conditions**: Checks user-defined termination criteria
    /// 4. **Data Collection**: Stores solution points for output
    /// 5. **Matrix Assembly**: Converts solution vectors to result matrices
    ///
    /// # Performance Features
    /// - **Efficient Storage**: Uses vector extension for minimal allocations
    /// - **Matrix Flattening**: Optimized conversion from Vec<DVector> to DMatrix
    /// - **Timing**: Reports integration time for performance analysis
    ///
    /// # Matrix Assembly Algorithm
    /// ```text
    /// flat_vec = [y₁(t₁), y₂(t₁), ..., yₙ(t₁), y₁(t₂), y₂(t₂), ..., yₙ(tₘ)]
    /// y_result = reshape(flat_vec, n_vars, n_times).transpose()
    /// ```
    pub fn main_loop(&mut self) -> () {
        // Analogue of https://github.com/scipy/scipy/blob/main/scipy/integrate/_ivp/ivp.py
        let start = Instant::now();
        let mut integr_status: Option<i8> = None;
        let mut y: Vec<DVector<f64>> = Vec::new();
        let mut t: Vec<f64> = Vec::new();
        let mut _i: i64 = 0;
        while integr_status.is_none() {
            self.step();
            let _status: i8 = 0;
            _i += 1;
            if self.status == "finished".to_string() {
                integr_status = Some(0)
            } else if self.status == "failed".to_string() {
                integr_status = Some(-1);
                break;
            }
            // Check stop condition before storing solution
            if self.check_stop_condition(&self.Solver_instance.y) {
                self.status = "stopped_by_condition".to_string();
                integr_status = Some(0);
            }

            t.push(self.Solver_instance.t);
            y.push(self.Solver_instance.y.clone());
        }

        let rows = &y.len();
        let cols = &y[0].len();

        let mut flat_vec: Vec<f64> = Vec::new();
        for vector in y.iter() {
            flat_vec.extend(vector)
        }
        let y_res: DMatrix<f64> = DMatrix::from_vec(*cols, *rows, flat_vec).transpose();
        let t_res = DVector::from_vec(t);
        let duration = start.elapsed();
        println!("Program took {} milliseconds to run", duration.as_millis());

        self.t_result = t_res.clone();
        self.y_result = y_res.clone();
    }

    /// Solves the ODE system from t0 to t_bound.
    ///
    /// This is the main entry point that orchestrates the complete solution process:
    /// 1. **Generate**: Convert symbolic expressions to numerical functions
    /// 2. **Integrate**: Run the main integration loop
    ///
    /// After calling this method, use `get_result()` to retrieve the solution.
    ///
    /// # Example
    /// ```rust, ignore
    /// solver.solve();
    /// let (t_result, y_result) = solver.get_result();
    /// ```
    pub fn solve(&mut self) -> () {
        self.generate();
        self.main_loop();
    }

    /// Generates plots of the solution using the built-in plotting utility.
    ///
    /// Creates time-series plots for all solution variables.
    /// Requires the solution to be computed first via `solve()`.
    pub fn plot_result(&self) -> () {
        plots(
            self.arg.clone(),
            self.values.clone(),
            self.t_result.clone(),
            self.y_result.clone(),
        );
        println!("result plotted");
    }

    /// Returns the computed solution data.
    ///
    /// # Returns
    /// * `DVector<f64>` - Time points
    /// * `DMatrix<f64>` - Solution matrix (rows = time, columns = variables)
    ///
    /// # Example
    /// ```rust, ignore
    /// let (t_result, y_result) = solver.get_result();
    /// println!("Final time: {}", t_result[t_result.len()-1]);
    /// println!("Final solution: {:?}", y_result.row(y_result.nrows()-1));
    /// ```
    pub fn get_result(&self) -> (DVector<f64>, DMatrix<f64>) {
        (self.t_result.clone(), self.y_result.clone())
    }

    /// Returns the current integration status.
    ///
    /// # Possible Values
    /// - `"running"`: Integration in progress
    /// - `"finished"`: Successfully completed
    /// - `"failed"`: Integration failed
    /// - `"stopped_by_condition"`: Terminated by stop condition
    ///
    /// # Returns
    /// Reference to the status string
    pub fn get_status(&self) -> &String {
        &self.status
    }

    /// Saves the solution results to a CSV file.
    ///
    /// The CSV format includes:
    /// - First row: Column headers (time variable + solution variables)
    /// - Subsequent rows: Time points and corresponding solution values
    ///
    /// # File Format
    /// ```csv
    /// t,y1,y2,...
    /// 0.0,1.0,0.0,...
    /// 0.01,0.999,0.01,...
    /// ```
    ///
    /// # Returns
    /// `Result<(), Box<dyn std::error::Error>>` - Success or file I/O error

    pub fn save_result(&self) -> Result<(), Box<dyn std::error::Error>> {
        let path = format!(
            "f:\\RUST\\RustProjects_\\RustedSciThe3\\src\\numerical\\results\\{}+{}.csv",
            self.arg,
            self.values.join("+")
        );
        let mut wtr = Writer::from_path(path)?;

        // Write column titles
        wtr.write_record(&[&self.arg, "values"])?;

        // Write time column
        wtr.write_record(self.t_result.iter().map(|&x| x.to_string()))?;

        // Write y columns
        for (i, col) in self.y_result.column_iter().enumerate() {
            let col_name = format!("{}", &self.values[i]);
            wtr.write_record(&[
                &col_name,
                &col.iter()
                    .map(|&x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            ])?;
        }
        print!("result saved");
        wtr.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::symbolic_engine::Expr;
    use std::collections::HashMap;

    #[test]
    fn test_bdf_riccati_equation() {
        // Riccati equation: y' = y^2 - t^2, y(0) = 1
        // Highly nonlinear with known analytical behavior
        let eq1 = Expr::parse_expression("y*y - t*t");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let method = "BDF".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 0.5;
        let max_step = 0.001;
        let rtol = 1e-8;
        let atol = 1e-10;

        let mut solver = ODEsolver::new(
            eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false,
            None,
        );

        solver.solve();
        assert_eq!(solver.get_status(), "finished");

        let (t_result, y_result) = solver.get_result();
        // Verify solution remains bounded and smooth
        for i in 0..t_result.len() {
            assert!(y_result[(i, 0)].is_finite());
            assert!(y_result[(i, 0)] > 0.0); // Should remain positive
        }
    }

    #[test]
    fn test_bdf_van_der_pol_oscillator() {
        // Van der Pol oscillator: y1' = y2, y2' = μ(1-y1^2)y2 - y1
        // Highly nonlinear system with μ = 5 (stiff)
        let eq1 = Expr::parse_expression("y2");
        let eq2 = Expr::parse_expression("5*(1-y1*y1)*y2 - y1");
        let eq_system = vec![eq1, eq2];
        let values = vec!["y1".to_string(), "y2".to_string()];
        let arg = "t".to_string();
        let method = "BDF".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![2.0, 0.0]);
        let t_bound = 5.0;
        let max_step = 0.01;
        let rtol = 1e-6;
        let atol = 1e-8;

        let mut solver = ODEsolver::new(
            eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false,
            None,
        );

        solver.solve();
        assert_eq!(solver.get_status(), "finished");

        let (_, y_result) = solver.get_result();
        // Van der Pol should exhibit limit cycle behavior
        assert!(y_result[(y_result.nrows() - 1, 0)].abs() < 3.0); // Bounded oscillation
    }

    #[test]
    fn test_bdf_bernoulli_equation() {
        // Bernoulli equation: y' + y = y^3, y(0) = 0.5
        // Analytical solution: y = 1/sqrt(3*exp(2*t) + 1)
        let eq1 = Expr::parse_expression("y*y*y - y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let method = "BDF".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![0.5]);
        let t_bound = 0.3;
        let max_step = 0.001;
        let rtol = 1e-8;
        let atol = 1e-10;

        let mut solver = ODEsolver::new(
            eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false,
            None,
        );

        solver.solve();
        assert_eq!(solver.get_status(), "finished");

        let (t_result, y_result) = solver.get_result();

        // Compare with analytical solution at final time
        let t_final = t_result[t_result.len() - 1];
        let y_analytical = 1.0 / (3.0 * (2.0 * t_final).exp() + 1.0).sqrt();
        let y_numerical = y_result[(y_result.nrows() - 1, 0)];

        assert!(
            (y_numerical - y_analytical).abs() < 1e-4,
            "Numerical: {}, Analytical: {}, Error: {}",
            y_numerical,
            y_analytical,
            (y_numerical - y_analytical).abs()
        );
    }

    #[test]
    fn test_bdf_logistic_equation() {
        // Logistic equation: y' = r*y*(1-y/K), y(0) = y0
        // Analytical solution: y = K*y0*exp(r*t)/(K + y0*(exp(r*t) - 1))
        let eq1 = Expr::parse_expression("2*y*(1-y/10)");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let method = "BDF".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 3.0;
        let max_step = 0.01;
        let rtol = 1e-8;
        let atol = 1e-10;

        let mut solver = ODEsolver::new(
            eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false,
            None,
        );

        solver.solve();
        assert_eq!(solver.get_status(), "finished");

        let (t_result, y_result) = solver.get_result();

        // Compare with analytical solution
        let r = 2.0;
        let k = 10.0;
        let y0_val = 1.0;

        for i in 0..t_result.len() {
            let t = t_result[i];
            let y_analytical = k * y0_val * (r * t).exp() / (k + y0_val * ((r * t).exp() - 1.0));
            let y_numerical = y_result[(i, 0)];

            assert!(
                (y_numerical - y_analytical).abs() < 1e-5,
                "At t={}: Numerical: {}, Analytical: {}, Error: {}",
                t,
                y_numerical,
                y_analytical,
                (y_numerical - y_analytical).abs()
            );
        }
    }

    #[test]
    fn test_bdf_pendulum_equation() {
        // Nonlinear pendulum: θ'' + sin(θ) = 0
        // Rewritten as system: θ' = ω, ω' = -sin(θ)
        let eq1 = Expr::parse_expression("omega");
        let eq2 = Expr::parse_expression("-sin(theta)");
        let eq_system = vec![eq1, eq2];
        let values = vec!["theta".to_string(), "omega".to_string()];
        let arg = "t".to_string();
        let method = "BDF".to_string();
        let t0 = 0.0;
        // at t = 0 θ(0)=1, omega(0) = θ'(0)=1
        let y0 = DVector::from_vec(vec![1.0, 0.0]); // Small angle approximation
        let t_bound = 1.0;
        let max_step = 0.001;
        let rtol = 1e-6;
        let atol = 1e-8;

        let mut solver = ODEsolver::new(
            eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false,
            None,
        );

        solver.solve();
        assert_eq!(solver.get_status(), "finished");

        let (_, y_result) = solver.get_result();

        // Energy conservation 0.5 *θ'^2 = C+cos(θ), C = - cos(1)
        // 0.5 *θ'^2 = cos(θ)- cos(1)
        //  so 0.5 *θ'^2 - (cos(θ)- cos(1)) must be close to 0 evarywhere
        println!(
            "1st and last teta {}, {}",
            y_result[(0, 0)],
            y_result[(y_result.nrows() - 1, 0)]
        );
        println!(
            "1st and last omega {}, {}",
            y_result[(0, 1)],
            y_result[(y_result.nrows() - 1, 1)]
        );
        let final_theta = y_result[(y_result.nrows() - 1, 0)];
        let final_omega = y_result[(y_result.nrows() - 1, 1)];
        let final_energy = 0.5 * final_omega.powi(2) - (1.0_f64.cos() - final_theta.cos());

        assert!(
            final_energy.abs() < 1e-3,
            "Energy not conserved: Initial: {}",
            final_energy
        );
    }

    #[test]
    fn test_bdf_lorenz_system() {
        // Lorenz system: x' = σ(y-x), y' = x(ρ-z)-y, z' = xy-βz
        let eq1 = Expr::parse_expression("10*(y-x)");
        let eq2 = Expr::parse_expression("x*(28-z)-y");
        let eq3 = Expr::parse_expression("x*y-8*z/3");
        let eq_system = vec![eq1, eq2, eq3];
        let values = vec!["x".to_string(), "y".to_string(), "z".to_string()];
        let arg = "t".to_string();
        let method = "BDF".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        let t_bound = 5.0;
        let max_step = 0.001;
        let rtol = 1e-8;
        let atol = 1e-10;

        let mut solver = ODEsolver::new(
            eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false,
            None,
        );

        solver.solve();
        assert_eq!(solver.get_status(), "finished");

        let (_, y_result) = solver.get_result();

        // Verify chaotic behavior remains bounded
        for i in 0..y_result.nrows() {
            assert!(y_result[(i, 0)].abs() < 50.0); // x bounded
            assert!(y_result[(i, 1)].abs() < 50.0); // y bounded
            assert!(y_result[(i, 2)] > 0.0 && y_result[(i, 2)] < 50.0); // z positive and bounded
        }
    }

    #[test]
    fn test_bdf_stiff_chemical_reaction() {
        // Stiff chemical kinetics: A -> B -> C
        // y1' = -k1*y1, y2' = k1*y1 - k2*y2, y3' = k2*y2
        // with k1 = 1, k2 = 1000 (stiff)
        let eq1 = Expr::parse_expression("-y1");
        let eq2 = Expr::parse_expression("y1 - 1000*y2");
        let eq3 = Expr::parse_expression("1000*y2");
        let eq_system = vec![eq1, eq2, eq3];
        let values = vec!["y1".to_string(), "y2".to_string(), "y3".to_string()];
        let arg = "t".to_string();
        let method = "BDF".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let t_bound = 2.0;
        let max_step = 0.01;
        let rtol = 1e-6;
        let atol = 1e-8;

        let mut solver = ODEsolver::new(
            eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false,
            None,
        );

        solver.solve();
        assert_eq!(solver.get_status(), "finished");

        let (t_result, y_result) = solver.get_result();

        // Mass conservation: y1 + y2 + y3 = 1
        let final_sum = y_result[(y_result.nrows() - 1, 0)]
            + y_result[(y_result.nrows() - 1, 1)]
            + y_result[(y_result.nrows() - 1, 2)];
        assert!(
            (final_sum - 1.0).abs() < 1e-6,
            "Mass not conserved: {}",
            final_sum
        );

        // At t_bound, y1 should be approximately exp(-t_bound)
        let t_final = t_result[t_result.len() - 1];
        let y1_analytical = (-t_final).exp();
        let y1_numerical = y_result[(y_result.nrows() - 1, 0)];
        assert!((y1_numerical - y1_analytical).abs() < 1e-4);
    }

    // Stop condition tests
    #[test]
    fn test_bdf_stop_condition_single_variable() {
        let eq1 = Expr::parse_expression("y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let method = "BDF".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 10.0;
        let max_step = 0.01;
        let rtol = 1e-6;
        let atol = 1e-3;

        let mut solver = ODEsolver::new(
            eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false,
            None,
        );

        let mut stop_condition = HashMap::new();
        stop_condition.insert("y".to_string(), 2.0);
        solver.set_stop_condition(stop_condition);

        solver.solve();

        assert_eq!(solver.get_status(), "stopped_by_condition");
        let (_, y_result) = solver.get_result();
        let final_y = y_result[(y_result.nrows() - 1, 0)];
        assert!((final_y - 2.0).abs() <= atol);
    }

    #[test]
    fn test_bdf_stop_condition_multiple_variables() {
        let eq1 = Expr::parse_expression("y2");
        let eq2 = Expr::parse_expression("-y1");
        let eq_system = vec![eq1, eq2];
        let values = vec!["y1".to_string(), "y2".to_string()];
        let arg = "t".to_string();
        let method = "BDF".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0, 0.0]);
        let t_bound = 10.0;
        let max_step = 0.01;
        let rtol = 1e-6;
        let atol = 1e-3;

        let mut solver = ODEsolver::new(
            eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false,
            None,
        );

        let mut stop_condition = HashMap::new();
        stop_condition.insert("y1".to_string(), 0.0);
        solver.set_stop_condition(stop_condition);

        solver.solve();

        assert_eq!(solver.get_status(), "stopped_by_condition");
        let (_, y_result) = solver.get_result();
        let final_y1 = y_result[(y_result.nrows() - 1, 0)];
        assert!(final_y1.abs() <= atol);
    }

    #[test]
    fn test_bdf_no_stop_condition() {
        let eq1 = Expr::parse_expression("-y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let method = "BDF".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 1.0;
        let max_step = 0.1;
        let rtol = 1e-6;
        let atol = 1e-6;

        let mut solver = ODEsolver::new(
            eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false,
            None,
        );

        solver.solve();

        assert_eq!(solver.get_status(), "finished");
        let (t_result, _) = solver.get_result();
        let final_t = t_result[t_result.len() - 1];
        assert!((final_t - t_bound).abs() <= max_step);
    }

    #[test]
    fn test_bdf_stop_condition_nonlinear() {
        let eq1 = Expr::parse_expression("y*y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let method = "BDF".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 10.0;
        let max_step = 0.01;
        let rtol = 1e-6;
        let atol = 1e-3;

        let mut solver = ODEsolver::new(
            eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false,
            None,
        );

        let mut stop_condition = HashMap::new();
        stop_condition.insert("y".to_string(), 1.5);
        solver.set_stop_condition(stop_condition);

        solver.solve();

        assert_eq!(solver.get_status(), "stopped_by_condition");
        let (_, y_result) = solver.get_result();
        let final_y = y_result[(y_result.nrows() - 1, 0)];
        assert!((final_y - 1.5).abs() <= atol);
    }
}
