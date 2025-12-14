//! Symbolic wrapper for Levenberg-Marquardt optimization algorithm.
//!
//! This module provides a high-level interface for solving nonlinear least squares problems
//! using symbolic expressions. It automatically generates analytical Jacobians and provides
//! logging capabilities similar to the NR solver.

use crate::numerical::optimization::LM_optimization::LevenbergMarquardt;
use crate::numerical::optimization::problem_LM::LeastSquaresProblem;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions::Jacobian;
use log::info;
use nalgebra::{DMatrix, DVector};
use simplelog::*;
use std::collections::HashMap;

/// Symbolic wrapper for Levenberg-Marquardt algorithm.
/// Solves nonlinear least squares problems using symbolic expressions with analytical Jacobians.
pub struct LM {
    /// Jacobian instance containing symbolic functions and their derivatives
    pub jacobian: Jacobian,
    /// Vector of symbolic equations to solve
    pub eq_system: Vec<Expr>,
    /// Variable names in the equations
    pub values: Vec<String>,
    /// Optional parameter names for parametric equations
    pub parameters: Option<Vec<String>>,
    /// Initial guess for the solution
    pub initial_guess: Vec<f64>,
    /// Maximum number of iterations
    pub max_iterations: Option<usize>,
    /// Convergence tolerance for parameters
    pub tolerance: Option<f64>,
    /// Function value tolerance
    pub f_tolerance: Option<f64>,
    /// Gradient tolerance
    pub g_tolerance: Option<f64>,
    /// Whether to scale diagonal elements
    pub scale_diag: Option<bool>,
    /// Solution vector
    pub result: Option<DVector<f64>>,
    /// Solution mapped to variable names
    pub map_of_solutions: Option<HashMap<String, f64>>,
    /// Logging level (debug, info, warn, error, off, none)
    pub loglevel: Option<String>,
}

impl LM {
    /// Creates a new LM solver instance with default settings.
    pub fn new() -> Self {
        LM {
            jacobian: Jacobian::new(),
            eq_system: Vec::new(),
            values: Vec::new(),
            parameters: None,
            initial_guess: Vec::new(),
            tolerance: None,
            f_tolerance: None,
            g_tolerance: None,
            scale_diag: None,
            max_iterations: None,
            result: None,
            map_of_solutions: None,
            loglevel: Some("info".to_string()),
        }
    }
    pub fn set_loglevel(&mut self, loglevel: String) {
        self.loglevel = Some(loglevel);
    }
    /// Sets up the equation system with unknowns, parameters, and solver options.
    pub fn set_equation_system(
        &mut self,
        eq_system: Vec<Expr>,
        unknowns: Option<Vec<String>>,
        parameters: Option<Vec<String>>,
        initial_guess: Vec<f64>,
        tolerance: Option<f64>,
        f_tolerance: Option<f64>,
        g_tolerance: Option<f64>,
        scale_diag: Option<bool>,
        max_iterations: Option<usize>,
    ) {
        self.eq_system = eq_system.clone();
        self.initial_guess = initial_guess;
        self.tolerance = tolerance;
        self.g_tolerance = g_tolerance;
        self.max_iterations = max_iterations;
        self.f_tolerance = f_tolerance;
        self.scale_diag = scale_diag;
        self.parameters = parameters;
        let values = if let Some(values) = unknowns {
            values
        } else {
            let mut args: Vec<String> = eq_system
                .iter()
                .map(|x| x.all_arguments_are_variables())
                .flatten()
                .collect::<Vec<String>>();
            args.sort();
            args.dedup();

            assert!(!args.is_empty(), "No variables found in the equations.");
            assert_eq!(
                args.len() == eq_system.len(),
                true,
                "Equation system and vector of variables should have the same length."
            );

            args
        };
        self.values = values.clone();
        assert!(
            !self.initial_guess.is_empty(),
            "Initial guess should not be empty."
        );
        if let Some(tolerance) = tolerance {
            assert!(
                tolerance >= 0.0,
                "Tolerance should be a non-negative number."
            );
        }
        if let Some(max_iterations) = max_iterations {
            assert!(
                max_iterations > 0,
                "Max iterations should be a positive number."
            );
        }
        if let Some(g_tolerance) = g_tolerance {
            assert!(
                g_tolerance >= 0.0,
                "Gradient tolerance should be a non-negative number."
            );
        }
        if let Some(f_tolerance) = f_tolerance {
            assert!(
                f_tolerance >= 0.0,
                "Function tolerance should be a non-negative number."
            );
        }
    }

    /// Parses string equations and sets up the system.
    pub fn eq_generate_from_str(
        &mut self,
        eq_system_string: Vec<String>,
        unknowns: Option<Vec<String>>,
        parameters: Option<Vec<String>>,
        initial_guess: Vec<f64>,
        tolerance: Option<f64>, // tolerance: f64
        f_tolerance: Option<f64>,
        g_tolerance: Option<f64>,
        scale_diag: Option<bool>,
        max_iterations: Option<usize>, // max_iterations: usize,
    ) {
        let eq_system = eq_system_string
            .iter()
            .map(|x| Expr::parse_expression(x))
            .collect::<Vec<Expr>>();
        self.set_equation_system(
            eq_system,
            unknowns,
            parameters,
            initial_guess,
            tolerance,
            f_tolerance,
            g_tolerance,
            scale_diag,
            max_iterations,
        );
    }

    /// Generates symbolic Jacobian and function representations.
    pub fn eq_generate(&mut self) {
        let eq_system = self.eq_system.clone();
        let mut Jacobian_instance = Jacobian::new();
        let args = self.values.clone();
        let args: Vec<&str> = args.iter().map(|x| x.as_str()).collect();
        Jacobian_instance.set_vector_of_functions(eq_system);
        Jacobian_instance.set_variables(args.clone());
        Jacobian_instance.calc_jacobian();
        Jacobian_instance.lambdify_jacobian_DMatrix_parallel();
        Jacobian_instance.lambdify_vector_funvector_DVector();
        assert_eq!(
            Jacobian_instance.vector_of_variables.len(),
            self.initial_guess.len(),
            "Initial guess and vector of variables should have the same length."
        );
        self.jacobian = Jacobian_instance;
    }
    /// Generates symbolic Jacobian for parametric equations.
    pub fn eq_generate_with_params(&mut self) {
        let eq_system = self.eq_system.clone();
        let mut Jacobian_instance = Jacobian::new();
        let args = self.values.clone();
        let args: Vec<&str> = args.iter().map(|x| x.as_str()).collect();
        Jacobian_instance.set_vector_of_functions(eq_system);
        let params = self
            .parameters
            .clone()
            .expect("for a problem with params - params must be set!");
        Jacobian_instance.set_params(params);
        Jacobian_instance.set_variables(args.clone());
        Jacobian_instance.calc_jacobian();
        Jacobian_instance.lambdify_jacobian_DMatrix_with_parameters_parallel();
        Jacobian_instance.lambdify_vector_funvector_DVector_with_parameters_parallel();
        assert_eq!(
            Jacobian_instance.vector_of_variables.len(),
            self.initial_guess.len(),
            "Initial guess and vector of variables should have the same length."
        );
        self.jacobian = Jacobian_instance;
    }
    /// Solves the nonlinear system with optional logging.
    pub fn solve(&mut self) {
        let is_logging_disabled = self
            .loglevel
            .as_ref()
            .map(|level| level == "off" || level == "none")
            .unwrap_or(false);

        if is_logging_disabled {
       
            self.solve_internal();
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

            let logger_instance = CombinedLogger::init(vec![TermLogger::new(
                log_option,
                Config::default(),
                TerminalMode::Mixed,
                ColorChoice::Auto,
            )]);

            match logger_instance {
                Ok(()) => {
               
                    self.solve_internal();
                    info!("Program ended");
                }
                Err(_) => {
               
                    self.solve_internal();
                }
            }
        }
    }

    /// Internal solver implementation without logging setup.
     fn solve_internal(&mut self) {
        let residual = |x: &DVector<f64>| -> DVector<f64> {
            let residual = &self.jacobian.lambdified_function_DVector;
            let residual = residual(x);
            residual.clone()
        };
        let jacobian = |x: &DVector<f64>| -> DMatrix<f64> {
            let jacobian = &self.jacobian.lambdified_jacobian_DMatrix;
            let jacobian = jacobian(x);
            jacobian.clone()
        };
        let problem = NonlinearSystem::new(
            DVector::from_vec(self.initial_guess.clone()),
            residual,
            jacobian,
        );
        let LM = LevenbergMarquardt::new();
        let LM = if let Some(max_iterations) = self.max_iterations {
            let LM = LM.with_patience(max_iterations);
            LM
        } else {
            LM
        };
        let LM = if let Some(tolerance) = self.tolerance {
            let LM = LM.with_xtol(tolerance);
            LM
        } else {
            LM
        };
        let LM = if let Some(g_tolerance) = self.g_tolerance {
            let LM = LM.with_gtol(g_tolerance);
            LM
        } else {
            LM
        };
        let LM = if let Some(f_tolerance) = self.f_tolerance {
            let LM = LM.with_ftol(f_tolerance);
            LM
        } else {
            LM
        };
        let (result, report) = LM.minimize(problem);
        info!("Nonlinear System Example:");
        info!("Termination: {:?}", report.termination);
        info!("Evaluations: {}", report.number_of_evaluations);
        info!("Final objective: {}", report.objective_function);
        info!("Final params: {:?}", result.params());
        if report.termination.was_successful() {
            let solution = result.params();
            self.result = Some(solution.clone());
            let solution: Vec<f64> = solution.data.into();
            let unknowns = self.values.clone();
            let map_of_solutions: HashMap<String, f64> = unknowns
                .iter()
                .zip(solution.iter())
                .map(|(k, v)| (k.to_string(), *v))
                .collect();

            let map_of_solutions = map_of_solutions;
            info!("Map of solutions: {:?}", map_of_solutions);
            self.map_of_solutions = Some(map_of_solutions);
        }
    }

    /// Solves parametric system without modifying self, returns solution map and vector.
    pub fn solve_with_params_unmut_internal(
        &self,
        params: Vec<f64>,
    ) -> (Option<HashMap<String, f64>>, Option<DVector<f64>>) {
        let params_vec = DVector::from_vec(params);
        let residual = |x: &DVector<f64>| -> DVector<f64> {
            let residual = &self.jacobian.lambdified_function_with_params;
            residual(&params_vec, x)
        };

        let jacobian = |x: &DVector<f64>| -> DMatrix<f64> {
            let jacobian = &self.jacobian.lambdified_jacobian_DMatrix_with_params;
            jacobian(&params_vec, x)
        };
        let problem = NonlinearSystem::new(
            DVector::from_vec(self.initial_guess.clone()),
            residual,
            jacobian,
        );
        let LM = LevenbergMarquardt::new();
        let LM = if let Some(max_iterations) = self.max_iterations {
            let LM = LM.with_patience(max_iterations);
            LM
        } else {
            LM
        };
        let LM = if let Some(tolerance) = self.tolerance {
            let LM = LM.with_xtol(tolerance);
            LM
        } else {
            LM
        };
        let LM = if let Some(g_tolerance) = self.g_tolerance {
            let LM = LM.with_gtol(g_tolerance);
            LM
        } else {
            LM
        };
        let LM = if let Some(f_tolerance) = self.f_tolerance {
            let LM = LM.with_ftol(f_tolerance);
            LM
        } else {
            LM
        };
        let (result, report) = LM.minimize(problem);
        info!("Nonlinear System Example:");
        info!("Termination: {:?}", report.termination);
        info!("Evaluations: {}", report.number_of_evaluations);
        info!("Final objective: {}", report.objective_function);
        info!("Final params: {:?}", result.params());
        if report.termination.was_successful() {
            let solution_: DVector<f64> = result.params();
            // self.result = Some(solution.clone());
            let solution: Vec<f64> = solution_.clone().data.into();
            let unknowns = self.values.clone();
            let map_of_solutions: HashMap<String, f64> = unknowns
                .iter()
                .zip(solution.iter())
                .map(|(k, v)| (k.to_string(), *v))
                .collect();

            let map_of_solutions: HashMap<String, f64> = map_of_solutions;
            info!("Map of solutions: {:?}", map_of_solutions);
            return (Some(map_of_solutions), Some(solution_));
        } else {
            (None, None)
        }
    }

    /// Solves parametric system with given parameter values and optional logging.
    pub fn solve_with_params_unmut(&self, params: Vec<f64>)  -> (Option<HashMap<String, f64>>, Option<DVector<f64>>) {
        let is_logging_disabled = self
            .loglevel
            .as_ref()
            .map(|level| level == "off" || level == "none")
            .unwrap_or(false);

        let (map_of_solutions, solution) = if is_logging_disabled {
     
            self.solve_with_params_unmut_internal(params)
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

            let logger_instance = CombinedLogger::init(vec![TermLogger::new(
                log_option,
                Config::default(),
                TerminalMode::Mixed,
                ColorChoice::Auto,
            )]);

            match logger_instance {
                Ok(()) => {
            
                    let result = self.solve_with_params_unmut_internal(params);
                    info!("Program ended");
                    result
                }
                Err(_) => {
              
                    self.solve_with_params_unmut_internal(params)
                }
            }
        };
     (map_of_solutions, solution)
    }

    pub fn  solve_with_params(&mut self, params: Vec<f64>){

        let (map_of_solutions, solution) = self.solve_with_params_unmut(params);
        self.map_of_solutions = map_of_solutions;
        self.result = solution;
    }
}
//generic LeastSquaresProblem implementation that accepts closures for residuals and Jacobian:

/// Generic nonlinear system that wraps residual and Jacobian functions.
/// Used as adapter between closures and LeastSquaresProblem trait.
pub struct NonlinearSystem<R, J>
where
    R: Fn(&DVector<f64>) -> DVector<f64>,
    J: Fn(&DVector<f64>) -> DMatrix<f64>,
{
    /// Current parameter values
    params: DVector<f64>,
    /// Residual function closure
    residuals_fn: R,
    /// Jacobian function closure
    jacobian_fn: J,
}

impl<R, J> NonlinearSystem<R, J>
where
    R: Fn(&DVector<f64>) -> DVector<f64>,
    J: Fn(&DVector<f64>) -> DMatrix<f64>,
{
    /// Creates a new nonlinear system with function closures and initial parameter guess.
    pub fn new(initial_guess: DVector<f64>, residuals_fn: R, jacobian_fn: J) -> Self {
        Self {
            params: initial_guess,
            residuals_fn,
            jacobian_fn,
        }
    }
}

impl<R, J> LeastSquaresProblem for NonlinearSystem<R, J>
where
    R: Fn(&DVector<f64>) -> DVector<f64>,
    J: Fn(&DVector<f64>) -> DMatrix<f64>,
{
    fn set_params(&mut self, x: &DVector<f64>) {
        self.params.copy_from(x);
    }

    fn params(&self) -> DVector<f64> {
        self.params.clone()
    }

    fn residuals(&self) -> Option<DVector<f64>> {
        Some((self.residuals_fn)(&self.params))
    }

    fn jacobian(&self) -> Option<DMatrix<f64>> {
        Some((self.jacobian_fn)(&self.params))
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////TESTS////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
///
///             TESTS FOR  Generic nonlinear system solver that accepts closures for residuals and Jacobian
#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::optimization::LM_optimization::LevenbergMarquardt;

    #[test]
    fn test_nonlinear_system_example() {
        // Example: Solve the system:
        // x^2 + y^2 - 1 = 0
        // x - y = 0
        // Solution should be approximately (√2/2, √2/2) or (-√2/2, -√2/2)

        let initial_guess = DVector::from_vec(vec![0.5, 0.3]);

        // Define residuals function
        let residuals_fn = |params: &DVector<f64>| -> DVector<f64> {
            let x = params[0];
            let y = params[1];
            DVector::from_vec(vec![
                x * x + y * y - 1.0, // x^2 + y^2 - 1 = 0
                x - y,               // x - y = 0
            ])
        };

        // Define Jacobian function
        let jacobian_fn = |params: &DVector<f64>| -> DMatrix<f64> {
            let x = params[0];
            let y = params[1];
            DMatrix::from_row_slice(
                2,
                2,
                &[
                    2.0 * x,
                    2.0 * y, // ∂/∂x(x^2+y^2-1), ∂/∂y(x^2+y^2-1)
                    1.0,
                    -1.0, // ∂/∂x(x-y),       ∂/∂y(x-y)
                ],
            )
        };

        let problem = NonlinearSystem::new(initial_guess, residuals_fn, jacobian_fn);
        let (result, report) = LevenbergMarquardt::new().minimize(problem);

        println!("Nonlinear System Example:");
        println!("Termination: {:?}", report.termination);
        println!("Evaluations: {}", report.number_of_evaluations);
        println!("Final objective: {}", report.objective_function);
        println!("Final params: {:?}", result.params());

        let final_params = result.params();
        let expected = (2.0_f64).sqrt() / 2.0; // √2/2 ≈ 0.707

        // Check that we found a solution close to (√2/2, √2/2)
        assert!((final_params[0].abs() - expected).abs() < 1e-6);
        assert!((final_params[1].abs() - expected).abs() < 1e-6);
        assert!((final_params[0] - final_params[1]).abs() < 1e-10); // x ≈ y
    }

    #[test]
    fn test_simple_quadratic_system() {
        // Solve: x^2 - 4 = 0, solution should be x = ±2
        let initial_guess = DVector::from_vec(vec![1.0]);

        let residuals_fn = |params: &DVector<f64>| -> DVector<f64> {
            let x = params[0];
            DVector::from_vec(vec![x * x - 4.0])
        };

        let jacobian_fn = |params: &DVector<f64>| -> DMatrix<f64> {
            let x = params[0];
            DMatrix::from_row_slice(1, 1, &[2.0 * x])
        };

        let problem = NonlinearSystem::new(initial_guess, residuals_fn, jacobian_fn);
        let (result, report) = LevenbergMarquardt::new().minimize(problem);

        println!("\nSimple Quadratic System:");
        println!("Termination: {:?}", report.termination);
        println!("Final params: {:?}", result.params());

        let final_params = result.params();
        assert!((final_params[0].abs() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_complex_nonlinear_system() {
        // More complex system:
        // sin(x) + cos(y) - 1 = 0
        // x^2 + y^2 - 1 = 0

        let initial_guess = DVector::from_vec(vec![0.5, 0.5]);

        let residuals_fn = |params: &DVector<f64>| -> DVector<f64> {
            let x = params[0];
            let y = params[1];
            DVector::from_vec(vec![x.sin() + y.cos() - 1.0, x * x + y * y - 1.0])
        };

        let jacobian_fn = |params: &DVector<f64>| -> DMatrix<f64> {
            let x = params[0];
            let y = params[1];
            DMatrix::from_row_slice(
                2,
                2,
                &[
                    x.cos(),
                    -y.sin(), // ∂/∂x(sin(x)+cos(y)-1), ∂/∂y(sin(x)+cos(y)-1)
                    2.0 * x,
                    2.0 * y, // ∂/∂x(x^2+y^2-1),       ∂/∂y(x^2+y^2-1)
                ],
            )
        };

        let problem = NonlinearSystem::new(initial_guess, residuals_fn, jacobian_fn);
        let (result, report) = LevenbergMarquardt::new().with_tol(1e-12).minimize(problem);

        println!("\nComplex Nonlinear System:");
        println!("Termination: {:?}", report.termination);
        println!("Final params: {:?}", result.params());
        println!("Final objective: {}", report.objective_function);

        // Verify the solution satisfies the equations
        let final_params = result.params();
        let x = final_params[0];
        let y = final_params[1];

        let residual1 = x.sin() + y.cos() - 1.0;
        let residual2 = x * x + y * y - 1.0;

        assert!(residual1.abs() < 1e-10);
        assert!(residual2.abs() < 1e-10);
    }
}
/////////////////////////////////////////////////////////////////////////////////////
///   
#[cfg(test)]
mod tests2 {
    use super::*;
    use crate::symbolic::symbolic_engine::Expr;
    use std::vec;
    #[test]
    fn test_nonlinear_system_example() {
        let vec_of_str = vec!["x^2 + y^2 - 1".to_string(), "x - y".to_string()];
        let initial_guess = vec![0.5, 0.5];
        let values = vec!["x".to_string(), "y".to_string()];
        let mut LM = LM::new();
        LM.eq_generate_from_str(
            vec_of_str,
            Some(values),
            None,
            initial_guess,
            None,
            None,
            None,
            None,
            None,
        );
        LM.eq_generate();
        LM.solve();
    }
    #[test]
    fn test_with_params() {
        // Solve: a*x^2 + b*y^2 - 1 = 0, x - y = 0 with params a=1, b=1
        let vec_of_str = vec!["a*x^2 + b*y^2 - 1".to_string(), "x - y".to_string()];
        let initial_guess = vec![0.5, 0.5];
        let values = vec!["x".to_string(), "y".to_string()];
        let params = vec!["a".to_string(), "b".to_string()];
        let mut LM = LM::new();
        LM.eq_generate_from_str(
            vec_of_str,
            Some(values),
            Some(params),
            initial_guess,
            None,
            None,
            None,
            None,
            None,
        );
        LM.set_loglevel("info".to_string());
        LM.eq_generate_with_params();
        LM.solve_with_params(vec![1.0, 1.0]);
        let map = LM.map_of_solutions.unwrap();
        let expected = (2.0_f64).sqrt() / 2.0;
        assert!((map["x"].abs() - expected).abs() < 1e-6);
        assert!((map["y"].abs() - expected).abs() < 1e-6);
    }

    #[test]
    fn chemical_equations() {
        let symbolic = Expr::Symbols("N0, N1, N2, Np, Lambda0, Lambda1");

        let dGm0 = Expr::Const(8.314 * 8.0e4); //8.314 * 8.0e3   
        let dG0 = Expr::Const(-450.0e3);
        let dG1 = Expr::Const(-150.0e3);
        let dG2 = Expr::Const(-50e3);
        let N0 = symbolic[0].clone();
        let N1 = symbolic[1].clone();
        let N2 = symbolic[2].clone();
        let Np = symbolic[3].clone();
        let Lambda0 = symbolic[4].clone();
        let Lambda1 = symbolic[5].clone();

        let RT = Expr::Const(8.314) * Expr::Const(3250.0);
        let eq_mu = vec![
            Lambda0.clone()
                + Expr::Const(2.0) * Lambda1.clone()
                + (dG0.clone() + RT.clone() * Expr::ln(N0.clone() / Np.clone())) / dGm0.clone(),
            Lambda0
                + Lambda1.clone()
                + (dG1 + RT.clone() * Expr::ln(N1.clone() / Np.clone())) / dGm0.clone(),
            Expr::Const(2.0) * Lambda1
                + (dG2 + RT * Expr::ln(N2.clone() / Np.clone())) / dGm0.clone(),
        ];
        let eq_sum_mole_numbers = vec![N0.clone() + N1.clone() + N2.clone() - Np.clone()];
        let composition_eq = vec![
            N0.clone() + N1.clone() - Expr::Const(0.999),
            Expr::Const(2.0) * N0.clone() + N1.clone() + Expr::Const(2.0) * N2 - Expr::Const(1.501),
        ];

        let mut full_system_sym = Vec::new();
        full_system_sym.extend(eq_mu.clone());
        full_system_sym.extend(eq_sum_mole_numbers.clone());
        full_system_sym.extend(composition_eq.clone());

        let full_system_sym: Vec<Expr> = full_system_sym.iter().map(|x| x.clone().simplify()).collect();

        for eq in &full_system_sym {
            println!("eq: {}", eq.clone().pretty_print());
        }
        // solver
        let initial_guess = vec![0.1, 0.1, 0.2, 0.3, 2.0, 2.0];
        let unknowns: Vec<String> = symbolic.iter().map(|x| x.to_string()).collect();
        let mut LM = LM::new();
        LM.set_loglevel("none".to_string());
        LM.set_equation_system(
            full_system_sym.clone(),
            Some(unknowns.clone()),
            None,
            initial_guess,
            None,
            Some(1e-6),
            Some(1e-6),
            Some(true),
            None,
        );
        LM.eq_generate();
        LM.solve();
        let map_of_solutions = LM.map_of_solutions.unwrap();

        let N0 = map_of_solutions.get("N0").unwrap();
        let N1 = map_of_solutions.get("N1").unwrap();
        let N2 = map_of_solutions.get("N2").unwrap();
        let Np = map_of_solutions.get("Np").unwrap();
        let _Lambda0 = map_of_solutions.get("Lambda0").unwrap();
        let _Lambda1 = map_of_solutions.get("Lambda1").unwrap();
        let d1 = *N0 + *N1 - 0.999;
        let d2 = N0 + N1 + N2 - Np;
        let d3 = 2.0 * N0 + N1 + 2.0 * N2 - 1.501;
        println!("d1: {}", d1);
        println!("d2: {}", d2);
        println!("d3: {}", d3);
        println!("map_of_solutions: {:?}", map_of_solutions);
        assert!(d1.abs() < 1e-3);
        assert!(d2.abs() < 1e-2);
        assert!(d3.abs() < 1e-2);
    }
}
