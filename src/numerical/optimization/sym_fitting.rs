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
    pub x_data: Vec<f64>,   // x data
    pub y_data: Vec<f64>,   // y data
    pub jacobian: Jacobian, // instance of Jacobian struct, contains jacobian matrix function and equation functions
    pub eq: Expr,           //equation  to fit
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
            eq: Expr::parse_expression("0"),
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
        self.eq = eq.clone();
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
        let eq = self.eq.clone();
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
        let eq = self.eq.clone();
        let map_of_solutions = self.map_of_solutions.clone().unwrap();
        // set numerical values of fittedd parameters
        let eq = eq.set_variable_from_map(&map_of_solutions);
        // turn symbolic equation into function
        let eq_fun = eq.lambdify1D();
        // calculate predicted y values
        let y_pred = x_data.iter().map(|x| eq_fun(*x)).collect::<Vec<f64>>();
        // calculate r squared
        let r_squared = r_squared(&y_data, &y_pred);
        self.r_ssquared = Some(r_squared);
        println!("R squared: {}", r_squared);
    }

    pub fn get_r_squared(&self) -> Option<f64> {
        self.r_ssquared
    }

    pub fn get_map_of_solutions(&self) -> Option<HashMap<String, f64>> {
        self.map_of_solutions.clone()
    }
}

fn create_residiual_vec(eq: &Expr, arg: String, x_data: Vec<f64>, y_data: Vec<f64>) -> Vec<Expr> {
    // let y_i = Expr::IndexedVars(x_data.len(), "y").0;
    let mut residual_vec = Vec::new();
    for i in 0..x_data.len() {
        let eq_i = eq.clone().set_variable(&arg, x_data[i]);
        let residual = eq_i.clone() - Expr::Const(y_data[i]);
        residual_vec.push(residual);
    }
    residual_vec
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
            .map(|&x| 5.0 * x + 2.0 + rand::rng().random_range(-0.1..0.1))
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
}
