use crate::numerical::optimization::LM_optimization::LevenbergMarquardt;
use crate::numerical::optimization::problem_LM::LeastSquaresProblem;
use crate::numerical::optimization::sym_wrapper::NonlinearSystem;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions::Jacobian;
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

/// 1D fitting using Levenberg-Marquardt algorithm
/// This is a wrapper around the Levenberg-Marquardt algorithm.
/// It takes a symbolic expression and a set of data points and fits the expression to the data points.
pub struct Fitting {
    pub x_data: Vec<f64>,     // x data
    pub y_data: Vec<f64>,     // y data
    pub jacobian: Jacobian, // instance of Jacobian struct, contains jacobian matrix function and equation functions
    pub equations: Vec<Expr>, // equations to fit, flattened row-major when there are many
    pub arg: String,
    pub unknown_coeffs: Vec<String>,   // vector of variables
    pub initial_guess: Vec<f64>,       // initial guess
    pub max_iterations: Option<usize>, // maximum number of iterations
    pub tolerance: Option<f64>,        // tolerance
    pub f_tolerance: Option<f64>,
    pub g_tolerance: Option<f64>, // gradient tolerance
    pub scale_diag: Option<bool>,
    pub result: Option<DVector<f64>>,
    pub map_of_solutions: Option<HashMap<String, f64>>,
    pub r_ssquared: Option<f64>,
}

impl Fitting {
    pub fn new() -> Self {
        Fitting {
            x_data: Vec::new(),
            y_data: Vec::new(),
            jacobian: Jacobian::new(),
            equations: vec![Expr::parse_expression("0")],
            unknown_coeffs: Vec::new(),
            arg: String::new(),
            initial_guess: Vec::new(),
            tolerance: None,
            f_tolerance: None,
            g_tolerance: None,
            scale_diag: None,
            max_iterations: None,
            result: None,
            map_of_solutions: None,
            r_ssquared: None,
        }
    }

    /// Builder pattern: Set x data
    pub fn with_x_data(mut self, x_data: Vec<f64>) -> Self {
        self.x_data = x_data;
        self
    }

    /// Builder pattern: Set y data
    pub fn with_y_data(mut self, y_data: Vec<f64>) -> Self {
        self.y_data = y_data;
        self
    }

    /// Builder pattern: Set data (x and y together)
    pub fn with_data(mut self, x_data: Vec<f64>, y_data: Vec<f64>) -> Self {
        self.x_data = x_data;
        self.y_data = y_data;
        self
    }

    /// Builder pattern: Set equation from Expr
    pub fn with_equation(mut self, eq: Expr) -> Self {
        self.equations = vec![eq];
        self
    }

    /// Builder pattern: Set target equations from a vector of Expr values.
    ///
    /// The equations are flattened in row-major order when residuals and
    /// predictions are generated.
    pub fn with_equations(mut self, eq_system: Vec<Expr>) -> Self {
        self.equations = eq_system;
        self
    }

    /// Builder pattern: Set equation from string
    pub fn with_equation_str(mut self, eq_string: String) -> Self {
        self.equations = vec![Expr::parse_expression(&eq_string)];
        self
    }

    /// Builder pattern: Set target equations from a vector of strings.
    pub fn with_equations_str(mut self, eq_system_string: Vec<String>) -> Self {
        self.equations = eq_system_string
            .iter()
            .map(|x| Expr::parse_expression(x))
            .collect::<Vec<_>>();
        self
    }

    /// Builder pattern: Set polynomial equation of given degree
    pub fn with_polynomial(mut self, degree: usize, arg: String) -> Self {
        let (eq, unknowns) = Expr::polyval(degree, &arg);
        self.equations = vec![eq];
        self.unknown_coeffs = unknowns;
        self.arg = arg;
        self
    }

    /// Builder pattern: Set unknown coefficients
    pub fn with_unknowns(mut self, unknowns: Vec<String>) -> Self {
        self.unknown_coeffs = unknowns;
        self
    }

    /// Builder pattern: Set argument variable
    pub fn with_arg(mut self, arg: String) -> Self {
        self.arg = arg;
        self
    }

    /// Builder pattern: Set initial guess
    pub fn with_initial_guess(mut self, initial_guess: Vec<f64>) -> Self {
        self.initial_guess = initial_guess;
        self
    }

    /// Builder pattern: Set tolerance
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = Some(tolerance);
        self
    }

    /// Builder pattern: Set function tolerance
    pub fn with_f_tolerance(mut self, f_tolerance: f64) -> Self {
        self.f_tolerance = Some(f_tolerance);
        self
    }

    /// Builder pattern: Set gradient tolerance
    pub fn with_g_tolerance(mut self, g_tolerance: f64) -> Self {
        self.g_tolerance = Some(g_tolerance);
        self
    }

    /// Builder pattern: Set scale diagonal
    pub fn with_scale_diag(mut self, scale_diag: bool) -> Self {
        self.scale_diag = Some(scale_diag);
        self
    }

    /// Builder pattern: Set max iterations
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = Some(max_iterations);
        self
    }

    /// Builder pattern: Build and fit (generates Jacobian and solves)
    pub fn build(mut self) -> Self {
        self.validate_and_infer();
        self.eq_generate();
        self.solve();
        self
    }

    /// Validate inputs and infer unknowns if not provided
    fn validate_and_infer(&mut self) {
        assert!(!self.x_data.is_empty(), "X data cannot be empty.");
        assert!(!self.y_data.is_empty(), "Y data cannot be empty.");
        assert!(
            !self.initial_guess.is_empty(),
            "Initial guess cannot be empty."
        );
        assert!(!self.arg.is_empty(), "Argument variable cannot be empty.");

        let equations = self.active_equations();
        let expected_y_len = self.x_data.len() * equations.len();
        assert_eq!(
            self.y_data.len(),
            expected_y_len,
            "Y data must have length x_data.len() * equation_count."
        );

        if self.unknown_coeffs.is_empty() {
            let mut args: Vec<String> = equations
                .iter()
                .flat_map(|eq| eq.all_arguments_are_variables())
                .collect();
            args.sort();
            args.dedup();
            // Remove the independent variable from unknowns
            args.retain(|x| x != &self.arg);
            assert!(
                !args.is_empty(),
                "No unknown coefficients found in equation."
            );
            self.unknown_coeffs = args;
        }

        assert_eq!(
            self.unknown_coeffs.len(),
            self.initial_guess.len(),
            "Initial guess length must match number of unknown coefficients."
        );
    }
    pub fn set_fitting(
        &mut self,
        x_data: Vec<f64>,
        y_data: Vec<f64>,
        eq: Expr,
        unknowns: Option<Vec<String>>,
        arg: String,
        initial_guess: Vec<f64>,
        tolerance: Option<f64>,
        f_tolerance: Option<f64>,
        g_tolerance: Option<f64>,
        scale_diag: Option<bool>,
        max_iterations: Option<usize>,
    ) {
        self.x_data = x_data;
        self.y_data = y_data;
        self.equations = vec![eq.clone()];
        self.arg = arg;
        self.initial_guess = initial_guess;
        self.tolerance = tolerance;
        self.g_tolerance = g_tolerance;
        self.max_iterations = max_iterations;
        self.f_tolerance = f_tolerance;
        self.scale_diag = scale_diag;
        let values = if let Some(values) = unknowns {
            values
        } else {
            let mut args: Vec<String> = eq.all_arguments_are_variables();
            args.sort();
            args.dedup();

            args
        };
        self.unknown_coeffs = values.clone();
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
    /// set fitting function as a vector of expressions
    pub fn set_fitting_system(
        &mut self,
        x_data: Vec<f64>,
        y_data: Vec<f64>,
        eq_system: Vec<Expr>,
        unknowns: Option<Vec<String>>,
        arg: String,
        initial_guess: Vec<f64>,
        tolerance: Option<f64>,
        f_tolerance: Option<f64>,
        g_tolerance: Option<f64>,
        scale_diag: Option<bool>,
        max_iterations: Option<usize>,
    ) {
        self.x_data = x_data;
        self.y_data = y_data;
        self.equations = eq_system.clone();
        self.arg = arg;
        self.initial_guess = initial_guess;
        self.tolerance = tolerance;
        self.g_tolerance = g_tolerance;
        self.max_iterations = max_iterations;
        self.f_tolerance = f_tolerance;
        self.scale_diag = scale_diag;
        let values = if let Some(values) = unknowns {
            values
        } else {
            let mut args: Vec<String> = eq_system
                .iter()
                .flat_map(|x| x.all_arguments_are_variables())
                .collect();
            args.sort();
            args.dedup();
            args
        };
        self.unknown_coeffs = values.clone();
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
    /// set fitting function as a string
    pub fn fitting_generate_from_str(
        &mut self,
        x_data: Vec<f64>,
        y_data: Vec<f64>,
        eq_string: String,
        unknowns: Option<Vec<String>>,
        arg: String,
        initial_guess: Vec<f64>,
        tolerance: Option<f64>, // tolerance: f64
        f_tolerance: Option<f64>,
        g_tolerance: Option<f64>,
        scale_diag: Option<bool>,
        max_iterations: Option<usize>, // max_iterations: usize,
    ) {
        let eq = Expr::parse_expression(&eq_string);

        self.set_fitting(
            x_data,
            y_data,
            eq,
            unknowns,
            arg,
            initial_guess,
            tolerance,
            f_tolerance,
            g_tolerance,
            scale_diag,
            max_iterations,
        );
    }
    /// set fitting function as a vector of expressions
    pub fn fitting_generate_from_vec(
        &mut self,
        x_data: Vec<f64>,
        y_data: Vec<f64>,
        eq_system: Vec<Expr>,
        unknowns: Option<Vec<String>>,
        arg: String,
        initial_guess: Vec<f64>,
        tolerance: Option<f64>,
        f_tolerance: Option<f64>,
        g_tolerance: Option<f64>,
        scale_diag: Option<bool>,
        max_iterations: Option<usize>,
    ) {
        self.set_fitting_system(
            x_data,
            y_data,
            eq_system,
            unknowns,
            arg,
            initial_guess,
            tolerance,
            f_tolerance,
            g_tolerance,
            scale_diag,
            max_iterations,
        );
    }
    /// fit with polynomial of certain degree
    pub fn poly_fitting(
        &mut self,
        x_data: Vec<f64>,
        y_data: Vec<f64>,
        degree: usize,
        arg: String,
        initial_guess: Vec<f64>,
        tolerance: Option<f64>,
        f_tolerance: Option<f64>,
        g_tolerance: Option<f64>,
        scale_diag: Option<bool>,
        max_iterations: Option<usize>,
    ) {
        // create polynomial equation
        let (eq, unknowns) = Expr::polyval(degree, &arg);
        println!("polynom: {}", eq);
        self.set_fitting(
            x_data,
            y_data,
            eq,
            Some(unknowns),
            arg,
            initial_guess,
            tolerance,
            f_tolerance,
            g_tolerance,
            scale_diag,
            max_iterations,
        );
    }
    pub fn fit_linear(&mut self, x_data: Vec<f64>, y_data: Vec<f64>, guess: (f64, f64)) {
        self.initial_guess = vec![guess.0, guess.1];
        self.poly_fitting(
            x_data.clone(),
            y_data.clone(),
            1,
            "x".to_string(),
            self.initial_guess.clone(),
            None,
            None,
            None,
            None,
            None,
        );
    }
    pub fn eq_generate(&mut self) {
        let eq = self.active_equations();
        let arg = self.arg.clone();
        let x_data = self.x_data.clone();
        let y_data = self.y_data.clone();
        let eq = create_residiual_vec(&eq, arg, x_data, y_data);
        let mut Jacobian_instance = Jacobian::new();
        let unknown_coeffs = self.unknown_coeffs.clone();
        let unknown_coeffs: Vec<&str> = unknown_coeffs.iter().map(|x| x.as_str()).collect();
        Jacobian_instance.set_vector_of_functions(eq);
        Jacobian_instance.set_variables(unknown_coeffs.clone());
        Jacobian_instance.calc_jacobian();
        Jacobian_instance.jacobian_generate(unknown_coeffs.clone());
        Jacobian_instance.lambdify_funcvector(unknown_coeffs);
        assert_eq!(
            Jacobian_instance.vector_of_variables.len(),
            self.initial_guess.len(),
            "Initial guess and vector of variables should have the same length."
        );
        self.jacobian = Jacobian_instance;
    }
    pub fn solve(&mut self) {
        let residual = |x: &DVector<f64>| -> DVector<f64> {
            let residual = &self
                .jacobian
                .evaluate_funvector_lambdified_DVector_unmut(x.clone().data.into());
            residual.clone()
        };
        let jacobian = |x: &DVector<f64>| -> DMatrix<f64> {
            let jacobian = &self
                .jacobian
                .evaluate_func_jacobian_DMatrix_unmut(x.clone().data.into());
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
        println!("Nonlinear System Example:");
        println!("Termination: {:?}", report.termination);
        println!("Evaluations: {}", report.number_of_evaluations);
        println!("Final objective: {}", report.objective_function);
        println!("Final params: {:?}", result.params());
        if report.termination.was_successful() {
            let solution = result.params();
            self.result = Some(solution.clone());
            let solution: Vec<f64> = solution.data.into();
            let unknowns = self.unknown_coeffs.clone();
            let map_of_solutions: HashMap<String, f64> = unknowns
                .iter()
                .zip(solution.iter())
                .map(|(k, v)| (k.to_string(), *v))
                .collect();

            let map_of_solutions = map_of_solutions;
            println!("Map of solutions: {:?}", map_of_solutions);
            self.map_of_solutions = Some(map_of_solutions);
            self.compare_with_data();
        }
    }
    /// for those who din't want to mess with multiple parameters

    pub fn easy_fitting(
        &mut self,
        x_data: Vec<f64>,
        y_data: Vec<f64>,
        eq_string: String,
        unknowns: Option<Vec<String>>,
        arg: String,
        initial_guess: Vec<f64>,
    ) {
        self.fitting_generate_from_str(
            x_data,
            y_data,
            eq_string,
            unknowns,
            arg,
            initial_guess,
            None,
            None,
            None,
            None,
            None,
        );
        self.eq_generate();
        self.solve();
    }

    fn compare_with_data(&mut self) {
        let x_data = self.x_data.clone();
        let y_data = self.y_data.clone();
        let eq = self.active_equations();
        let map_of_solutions = self.map_of_solutions.clone().unwrap();
        let y_pred = evaluate_equations(&eq, &x_data, &map_of_solutions);
        // calculate r squared
        let r_squared = r_squared(&y_data, &y_pred);
        self.r_ssquared = Some(r_squared);
        println!("R squared: {}", r_squared);
    }
    /// extrapolate or interpolate function for arbitrary x values
    pub fn extra_interpolate(&self, x_values: Vec<f64>) -> Vec<f64> {
        let eq = self.active_equations();
        let map_of_solutions = self.map_of_solutions.clone().unwrap();
        evaluate_equations(&eq, &x_values, &map_of_solutions)
    }
    pub fn get_r_squared(&self) -> Option<f64> {
        self.r_ssquared
    }

    /// Return the coefficient of determination for the last successful fit.
    pub fn r_squared(&self) -> Option<f64> {
        self.get_r_squared()
    }

    /// Short alias for [`r_squared`](Self::r_squared).
    pub fn r2(&self) -> Option<f64> {
        self.r_squared()
    }

    pub fn get_map_of_solutions(&self) -> Option<HashMap<String, f64>> {
        self.map_of_solutions.clone()
    }

    /// Return the fitted parameter map in a backend-friendly form.
    ///
    /// This is the preferred accessor for new code because it reads the same
    /// way on the classic LM path and on the higher-level wrappers.
    pub fn solution_map(&self) -> Option<HashMap<String, f64>> {
        self.get_map_of_solutions()
    }

    fn active_equations(&self) -> Vec<Expr> {
        self.equations.clone()
    }
}

fn create_residiual_vec(
    eq_system: &[Expr],
    arg: String,
    x_data: Vec<f64>,
    y_data: Vec<f64>,
) -> Vec<Expr> {
    let mut residual_vec = Vec::new();
    let x_len = x_data.len();
    assert_eq!(
        y_data.len(),
        eq_system.len() * x_len,
        "Flattened target data must match equation count times x data length."
    );
    for (eq_idx, eq) in eq_system.iter().enumerate() {
        let y_offset = eq_idx * x_len;
        for i in 0..x_len {
            let eq_i = eq.clone().set_variable(&arg, x_data[i]);
            let residual = eq_i.clone() - Expr::Const(y_data[y_offset + i]);
            residual_vec.push(residual);
        }
    }
    residual_vec
}

fn evaluate_equations(
    eq_system: &[Expr],
    x_values: &[f64],
    map_of_solutions: &HashMap<String, f64>,
) -> Vec<f64> {
    let mut y_pred = Vec::with_capacity(eq_system.len() * x_values.len());
    for eq in eq_system {
        let eq = eq.clone().set_variable_from_map(map_of_solutions);
        let eq_fun = eq.lambdify1D();
        for x in x_values {
            y_pred.push(eq_fun(*x));
        }
    }
    y_pred
}

pub fn r_squared(y_data: &Vec<f64>, y_pred: &Vec<f64>) -> f64 {
    let y_mean = y_data.iter().sum::<f64>() / y_data.len() as f64;
    let ss_tot = y_data.iter().map(|y| (y - y_mean).powi(2)).sum::<f64>();
    let ss_res = y_data
        .iter()
        .zip(y_pred.iter())
        .map(|(y, y_pred)| (y - y_pred).powi(2))
        .sum::<f64>();
    let r_squared = 1.0 - ss_res / ss_tot;
    r_squared
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn linear_fitting_test() {
        let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_data = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let initial_guess = vec![1.0, 1.0];
        let unknown_coeffs = vec!["a".to_string(), "b".to_string()];
        let eq = "a * x + b".to_string();
        let mut sym_fitting = Fitting::new();
        sym_fitting.fitting_generate_from_str(
            x_data,
            y_data,
            eq,
            Some(unknown_coeffs),
            "x".to_string(),
            initial_guess,
            None,
            None,
            None,
            None,
            None,
        );
        sym_fitting.eq_generate();
        sym_fitting.solve();
        let map_of_solutions = sym_fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map_of_solutions["a"], 2.0, epsilon = 1e-6);
        assert_relative_eq!(map_of_solutions["b"], 0.0, epsilon = 1e-6);
    }
    #[test]
    fn quadratic_fitting_test() {
        let x_data = (0..100).map(|x| x as f64).collect::<Vec<f64>>();
        let quadratic_function = |x: f64| 5.0 * x * x + 2.0 * x + 100.0;
        let y_data = x_data
            .iter()
            .map(|&x| quadratic_function(x))
            .collect::<Vec<f64>>();
        let initial_guess = vec![1.0, 1.0, 1.0];
        let unknown_coeffs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let eq = "a * x^2.0 + b * x + c".to_string();
        let mut sym_fitting = Fitting::new();
        sym_fitting.fitting_generate_from_str(
            x_data,
            y_data,
            eq,
            Some(unknown_coeffs),
            "x".to_string(),
            initial_guess,
            None,
            None,
            None,
            None,
            None,
        );
        sym_fitting.eq_generate();
        sym_fitting.solve();
        let map_of_solutions = sym_fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map_of_solutions["a"], 5.0, epsilon = 1e-6);
        assert_relative_eq!(map_of_solutions["b"], 2.0, epsilon = 1e-6);
        assert_relative_eq!(map_of_solutions["c"], 100.0, epsilon = 1e-6);
    }
    #[test]
    fn exp_fitting_test() {
        let x_data = (0..20).map(|x| x as f64).collect::<Vec<f64>>();
        let exp_function = |x: f64| (1e-1 * x).exp() + 10.0;
        let y_data = x_data
            .iter()
            .map(|&x| exp_function(x))
            .collect::<Vec<f64>>();
        let initial_guess = vec![1.0, 1.0];
        let unknown_coeffs = vec!["a".to_string(), "b".to_string()];
        let eq = " exp(a*x) + b".to_string();
        let mut sym_fitting = Fitting::new();
        sym_fitting.fitting_generate_from_str(
            x_data,
            y_data,
            eq,
            Some(unknown_coeffs),
            "x".to_string(),
            initial_guess,
            None,
            None,
            None,
            None,
            None,
        );
        sym_fitting.eq_generate();
        sym_fitting.solve();
        let map_of_solutions = sym_fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map_of_solutions["a"], 1e-1, epsilon = 1e-6);
        assert_relative_eq!(map_of_solutions["b"], 10.0, epsilon = 1e-6);
    }
    #[test]
    fn easy_fitting_test() {
        let x_data = (0..100).map(|x| x as f64).collect::<Vec<f64>>();
        let quadratic_function = |x: f64| 5.0 * x * x + 2.0 * x + 100.0;
        let y_data = x_data
            .iter()
            .map(|&x| quadratic_function(x))
            .collect::<Vec<f64>>();
        let initial_guess = vec![1.0, 1.0, 1.0];
        let unknown_coeffs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let eq = "a * x^2.0 + b * x + c".to_string();
        let mut sym_fitting = Fitting::new();
        sym_fitting.easy_fitting(
            x_data,
            y_data,
            eq,
            Some(unknown_coeffs),
            "x".to_string(),
            initial_guess,
        );
        let map_of_solutions = sym_fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map_of_solutions["a"], 5.0, epsilon = 1e-6);
        assert_relative_eq!(map_of_solutions["b"], 2.0, epsilon = 1e-6);
        assert_relative_eq!(map_of_solutions["c"], 100.0, epsilon = 1e-6);
    }

    #[test]
    fn polinomial_fitting_test() {
        let x_data = (0..100).map(|x| x as f64).collect::<Vec<f64>>();
        let polynomial_function = |x: f64| 5.0 * x * x * x + 2.0 * x * x + 100.0 * x + 1000.0;
        let y_data = x_data
            .iter()
            .map(|&x| polynomial_function(x))
            .collect::<Vec<f64>>();
        let initial_guess = vec![1.0, 1.0, 1.0, 1.0];

        let mut sym_fitting = Fitting::new();
        sym_fitting.poly_fitting(
            x_data,
            y_data,
            3,
            "x".to_string(),
            initial_guess,
            None,
            None,
            None,
            None,
            None,
        );
        sym_fitting.eq_generate();
        sym_fitting.solve();
        let map_of_solutions = sym_fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map_of_solutions["c3"], 5.0, epsilon = 1e-6);
        assert_relative_eq!(map_of_solutions["c2"], 2.0, epsilon = 1e-6);
        assert_relative_eq!(map_of_solutions["c1"], 100.0, epsilon = 1e-6);
        assert_relative_eq!(map_of_solutions["c0"], 1000.0, epsilon = 1e-6);
    }
    #[test]
    fn test_linear_fit() {
        let x_data = (0..100).map(|x| x as f64).collect::<Vec<f64>>();
        let y_data = x_data.iter().map(|&x| 5.0 * x + 2.0).collect::<Vec<f64>>();
        let initial_guess = (1.0, 1.0);
        let mut sym_fitting = Fitting::new();
        sym_fitting.fit_linear(x_data, y_data, initial_guess);
        sym_fitting.eq_generate();
        sym_fitting.solve();
        let map_of_solutions = sym_fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map_of_solutions["c1"], 5.0, epsilon = 1e-6);
        assert_relative_eq!(map_of_solutions["c0"], 2.0, epsilon = 1e-6);
    }
    #[test]
    fn test_noisy_linear_fit() {
        use rand::Rng;
        let x_data = (0..1000).map(|x| x as f64).collect::<Vec<f64>>();
        // to y data add some noise from -0.05 to 0.05
        let y_data = x_data
            .iter()
            .map(|&x| 5.0 * x + 2.0 + rand::random_range(-0.1..0.1))
            .collect::<Vec<f64>>();
        println!("noisy y_data: {:?}", y_data);
        let initial_guess = (1.0, 1.0);
        let mut sym_fitting = Fitting::new();
        sym_fitting.fit_linear(x_data, y_data, initial_guess);
        sym_fitting.eq_generate();
        sym_fitting.solve();
        let map_of_solutions = sym_fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map_of_solutions["c1"], 5.0, epsilon = 1e-2);
        assert_relative_eq!(map_of_solutions["c0"], 2.0, epsilon = 1e-2);
    }

    #[test]
    fn test_noisy_linear_experimatal() {
        let x_data = vec![
            0.0 + 41.35,
            1.75 + 41.35,
            4.85 + 41.35,
            6.0 + 41.35,
            11.2 + 41.35,
        ];

        let y_data = vec![-0.69, -2.24, -5.47, -6.47, -11.86];
        println!("noisy y_data: {:?}", y_data);
        let initial_guess = (1.0, 1.0);
        let mut sym_fitting = Fitting::new();
        sym_fitting.fit_linear(x_data, y_data, initial_guess);
        sym_fitting.eq_generate();
        sym_fitting.solve();
        let map_of_solutions = sym_fitting.map_of_solutions.unwrap();
        println!("{:?}", map_of_solutions);
    }

    // ========== Builder Pattern Tests ==========

    #[test]
    fn test_builder_linear_string() {
        let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_data = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let fitting = Fitting::new()
            .with_data(x_data, y_data)
            .with_equation_str("a * x + b".to_string())
            .with_arg("x".to_string())
            .with_unknowns(vec!["a".to_string(), "b".to_string()])
            .with_initial_guess(vec![1.0, 1.0])
            .build();

        let map = fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map["a"], 2.0, epsilon = 1e-6);
        assert_relative_eq!(map["b"], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_builder_linear_native_expr() {
        let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_data = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        // Native symbolic construction
        let vars = Expr::Symbols("a, x, b");
        let a = vars[0].clone();
        let x = vars[1].clone();
        let b = vars[2].clone();
        let eq = a * x + b;

        let fitting = Fitting::new()
            .with_data(x_data, y_data)
            .with_equation(eq)
            .with_arg("x".to_string())
            .with_unknowns(vec!["a".to_string(), "b".to_string()])
            .with_initial_guess(vec![1.0, 1.0])
            .build();

        let map = fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map["a"], 2.0, epsilon = 1e-6);
        assert_relative_eq!(map["b"], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_builder_vector_equations_native_expr() {
        let x_data = vec![1.0, 2.0, 3.0, 4.0];
        let mut y_data = x_data.iter().map(|&x| 2.0 * x + 1.0).collect::<Vec<f64>>();
        y_data.extend(x_data.iter().map(|&x| 3.0 * x * x + 4.0));

        let vars = Expr::Symbols("a, b, c, d, x");
        let a = vars[0].clone();
        let b = vars[1].clone();
        let c = vars[2].clone();
        let d = vars[3].clone();
        let x = vars[4].clone();

        let eq_system = vec![
            a.clone() * x.clone() + b.clone(),
            c * x.clone().pow(Expr::Const(2.0)) + d,
        ];

        let fitting = Fitting::new()
            .with_data(x_data, y_data)
            .with_equations(eq_system)
            .with_arg("x".to_string())
            .with_unknowns(vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ])
            .with_initial_guess(vec![1.0, 1.0, 1.0, 1.0])
            .build();

        let map = fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map["a"], 2.0, epsilon = 1e-6);
        assert_relative_eq!(map["b"], 1.0, epsilon = 1e-6);
        assert_relative_eq!(map["c"], 3.0, epsilon = 1e-6);
        assert_relative_eq!(map["d"], 4.0, epsilon = 1e-6);
    }

    #[test]
    fn test_vector_equations_imperative_api() {
        let x_data = vec![1.0, 2.0, 3.0];
        let mut y_data = x_data.iter().map(|&x| 1.5 * x + 0.5).collect::<Vec<f64>>();
        y_data.extend(x_data.iter().map(|&x| 2.0 * x * x + 3.0));

        let vars = Expr::Symbols("a, b, c, d, x");
        let a = vars[0].clone();
        let b = vars[1].clone();
        let c = vars[2].clone();
        let d = vars[3].clone();
        let x = vars[4].clone();

        let eq_system = vec![
            a.clone() * x.clone() + b.clone(),
            c * x.clone().pow(Expr::Const(2.0)) + d,
        ];

        let mut fitting = Fitting::new();
        fitting.fitting_generate_from_vec(
            x_data,
            y_data,
            eq_system,
            Some(vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ]),
            "x".to_string(),
            vec![1.0, 1.0, 1.0, 1.0],
            None,
            None,
            None,
            None,
            None,
        );
        fitting.eq_generate();
        fitting.solve();

        let map = fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map["a"], 1.5, epsilon = 1e-6);
        assert_relative_eq!(map["b"], 0.5, epsilon = 1e-6);
        assert_relative_eq!(map["c"], 2.0, epsilon = 1e-6);
        assert_relative_eq!(map["d"], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_builder_quadratic_native_expr() {
        let x_data = (0..50).map(|x| x as f64).collect::<Vec<f64>>();
        let y_data = x_data
            .iter()
            .map(|&x| 3.0 * x * x + 2.0 * x + 1.0)
            .collect::<Vec<f64>>();

        // Native symbolic construction
        let vars = Expr::Symbols("a, b, c, x");
        let a = vars[0].clone();
        let b = vars[1].clone();
        let c = vars[2].clone();
        let x = vars[3].clone();

        let eq = a * x.clone().pow(Expr::Const(2.0)) + b * x + c;

        let fitting = Fitting::new()
            .with_data(x_data, y_data)
            .with_equation(eq)
            .with_arg("x".to_string())
            .with_unknowns(vec!["a".to_string(), "b".to_string(), "c".to_string()])
            .with_initial_guess(vec![1.0, 1.0, 1.0])
            .build();

        let map = fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map["a"], 3.0, epsilon = 1e-6);
        assert_relative_eq!(map["b"], 2.0, epsilon = 1e-6);
        assert_relative_eq!(map["c"], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_builder_exponential_native_expr() {
        let x_data = (0..20).map(|x| x as f64).collect::<Vec<f64>>();
        let y_data = x_data
            .iter()
            .map(|&x| (0.1 * x).exp() + 5.0)
            .collect::<Vec<f64>>();

        // Native symbolic construction
        let vars = Expr::Symbols("a, x, b");
        let a = vars[0].clone();
        let x = vars[1].clone();
        let b = vars[2].clone();

        let eq = Expr::exp(a * x) + b;

        let fitting = Fitting::new()
            .with_data(x_data, y_data)
            .with_equation(eq)
            .with_arg("x".to_string())
            .with_unknowns(vec!["a".to_string(), "b".to_string()])
            .with_initial_guess(vec![0.05, 1.0])
            .with_tolerance(1e-8)
            .build();

        let map = fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map["a"], 0.1, epsilon = 1e-6);
        assert_relative_eq!(map["b"], 5.0, epsilon = 1e-6);
    }

    #[test]
    fn test_builder_logarithmic_native_expr() {
        let x_data = (1..30).map(|x| x as f64).collect::<Vec<f64>>();
        let y_data = x_data
            .iter()
            .map(|&x| 2.0 * x.ln() + 3.0)
            .collect::<Vec<f64>>();

        // Native symbolic construction
        let vars = Expr::Symbols("a, x, b");
        let a = vars[0].clone();
        let x = vars[1].clone();
        let b = vars[2].clone();

        let eq = a * Expr::ln(x) + b;

        let fitting = Fitting::new()
            .with_data(x_data, y_data)
            .with_equation(eq)
            .with_arg("x".to_string())
            .with_unknowns(vec!["a".to_string(), "b".to_string()])
            .with_initial_guess(vec![1.0, 1.0])
            .build();

        let map = fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map["a"], 2.0, epsilon = 1e-6);
        assert_relative_eq!(map["b"], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_builder_trigonometric_native_expr() {
        use std::f64::consts::PI;
        let x_data = (0..100).map(|x| x as f64 * PI / 50.0).collect::<Vec<f64>>();
        let y_data = x_data
            .iter()
            .map(|&x| 2.0 * x.sin() + 1.0)
            .collect::<Vec<f64>>();

        // Native symbolic construction
        let vars = Expr::Symbols("a, x, b");
        let a = vars[0].clone();
        let x = vars[1].clone();
        let b = vars[2].clone();
        let eq = a * Expr::sin(Box::new(x)) + b;

        let fitting = Fitting::new()
            .with_data(x_data, y_data)
            .with_equation(eq)
            .with_arg("x".to_string())
            .with_unknowns(vec!["a".to_string(), "b".to_string()])
            .with_initial_guess(vec![1.0, 0.5])
            .with_tolerance(1e-8)
            .with_g_tolerance(1e-8)
            .with_f_tolerance(1e-8)
            .with_max_iterations(100)
            .build();

        let map = fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map["a"], 2.0, epsilon = 1e-6);
        assert_relative_eq!(map["b"], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_builder_polynomial() {
        let x_data = (0..50).map(|x| x as f64).collect::<Vec<f64>>();
        let y_data = x_data
            .iter()
            .map(|&x| 2.0 * x.powi(3) + 3.0 * x.powi(2) + 4.0 * x + 5.0)
            .collect::<Vec<f64>>();

        let fitting = Fitting::new()
            .with_data(x_data, y_data)
            .with_polynomial(3, "x".to_string())
            .with_initial_guess(vec![1.0, 1.0, 1.0, 1.0])
            .build();

        let map = fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map["c3"], 2.0, epsilon = 1e-6);
        assert_relative_eq!(map["c2"], 3.0, epsilon = 1e-6);
        assert_relative_eq!(map["c1"], 4.0, epsilon = 1e-6);
        assert_relative_eq!(map["c0"], 5.0, epsilon = 1e-6);
    }

    #[test]
    fn test_builder_complex_native_expr() {
        let x_data = (1..30).map(|x| x as f64 * 0.1).collect::<Vec<f64>>();
        let y_data = x_data
            .iter()
            .map(|&x| 2.0 * x.ln() + 0.5 * (0.3 * x).exp() + 1.0)
            .collect::<Vec<f64>>();

        // Native symbolic construction with multiple operations
        let vars = Expr::Symbols("a, b, c, d, x");
        let a = vars[0].clone();
        let b = vars[1].clone();
        let c = vars[2].clone();
        let d = vars[3].clone();
        let x = vars[4].clone();

        let eq = a * Expr::ln(x.clone()) + b * Expr::exp(c * x) + d;

        let fitting = Fitting::new()
            .with_data(x_data, y_data)
            .with_equation(eq)
            .with_arg("x".to_string())
            .with_unknowns(vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ])
            .with_initial_guess(vec![1.5, 0.3, 0.2, 0.5])
            .with_tolerance(1e-6)
            .with_max_iterations(300)
            .build();

        let map = fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map["a"], 2.0, epsilon = 1e-3);
        assert_relative_eq!(map["b"], 0.5, epsilon = 1e-3);
        assert_relative_eq!(map["c"], 0.3, epsilon = 1e-3);
        assert_relative_eq!(map["d"], 1.0, epsilon = 1e-3);
    }

    #[test]
    fn test_builder_power_law_native_expr() {
        // Use smaller range and simpler power for more stable fitting
        let x_data = (1..50).map(|x| x as f64).collect::<Vec<f64>>();
        let y_data = x_data
            .iter()
            .map(|&x| 2.0 * x.powf(0.5) + 1.0)
            .collect::<Vec<f64>>();

        // Native symbolic construction: a * x^b + c
        let vars = Expr::Symbols("a, b, c, x");
        let a = vars[0].clone();
        let b = vars[1].clone();
        let c = vars[2].clone();
        let x = vars[3].clone();

        let eq = a * x.pow(b) + c;

        let fitting = Fitting::new()
            .with_data(x_data, y_data)
            .with_equation(eq)
            .with_arg("x".to_string())
            .with_unknowns(vec!["a".to_string(), "b".to_string(), "c".to_string()])
            .with_initial_guess(vec![1.5, 0.4, 0.5])
            .with_tolerance(1e-6)
            .with_max_iterations(300)
            .build();

        let map = fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map["a"], 2.0, epsilon = 1e-4);
        assert_relative_eq!(map["b"], 0.5, epsilon = 1e-4);
        assert_relative_eq!(map["c"], 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_builder_power_law() {
        // Use smaller range and simpler power for more stable fitting
        let x_data = (1..50).map(|x| x as f64).collect::<Vec<f64>>();
        let y_data = x_data
            .iter()
            .map(|&x| 2.0 * x.powf(0.5) + 1.0)
            .collect::<Vec<f64>>();

        let fitting = Fitting::new()
            .with_data(x_data, y_data)
            .with_equation_str("a*x^b + c".to_string())
            .with_arg("x".to_string())
            .with_unknowns(vec!["a".to_string(), "b".to_string(), "c".to_string()])
            .with_initial_guess(vec![1.5, 0.4, 0.5])
            .with_tolerance(1e-6)
            .with_max_iterations(3000)
            .build();

        let map = fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map["a"], 2.0, epsilon = 1e-4);
        assert_relative_eq!(map["b"], 0.5, epsilon = 1e-4);
        assert_relative_eq!(map["c"], 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_builder_rational_function_native_expr() {
        let x_data = (1..30).map(|x| x as f64).collect::<Vec<f64>>();
        let y_data = x_data
            .iter()
            .map(|&x| (2.0 * x + 3.0) / (x + 1.0))
            .collect::<Vec<f64>>();

        // Native symbolic construction: (a*x + b) / (x + c)
        let vars = Expr::Symbols("a, b, c, x");
        let a = vars[0].clone();
        let b = vars[1].clone();
        let c = vars[2].clone();
        let x = vars[3].clone();

        let eq = (a * x.clone() + b) / (x + c);

        let fitting = Fitting::new()
            .with_data(x_data, y_data)
            .with_equation(eq)
            .with_arg("x".to_string())
            .with_unknowns(vec!["a".to_string(), "b".to_string(), "c".to_string()])
            .with_initial_guess(vec![1.5, 2.0, 0.5])
            .with_tolerance(1e-8)
            .build();

        let map = fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map["a"], 2.0, epsilon = 1e-5);
        assert_relative_eq!(map["b"], 3.0, epsilon = 1e-5);
        assert_relative_eq!(map["c"], 1.0, epsilon = 1e-5);
    }
}
