//! Optimization algorithms:
//! - Levenberg-Marquardt algorithm for:
//!         - solving nonlinear equation system
//!         - fitting data  
//! LM uses analytical Jacobian matrix
//! - H.P.Gavin's Levenberg-Marquardt algorithm
//! - scalar root finding:
//!         - bisection method
//!         - Newton-Raphson method
//!         - secant method
//!         - Brent's method
//! 
#[allow(non_snake_case)]
///here is main loop for solving nonlinear equation system with Levenberg-Marquardt algorithm
pub mod LM_optimization;
/// some special cases of fitting
pub mod fitting_features;
#[allow(non_snake_case)]
/// main function to solve nonlinear equation system with Levenberg-Marquardt algorithm
pub mod problem_LM;
/// some linear algebra functions for solving nonlinear equation system
#[allow(non_snake_case)]
pub mod qr_LM;
/// nonlinear equation system solver implementation for fittting data. Fiitting function is symbolic expression.
///
/// Example#1
/// ```
///  use approx::assert_relative_eq;
/// use RustedSciThe::numerical::optimization::sym_fitting::Fitting;
///   // creating test data to fit
/// let x_data = (0..20).map(|x| x as f64).collect::<Vec<f64>>();
///        let exp_function = |x: f64| (1e-1 * x).exp() + 10.0;
///        let y_data = x_data
///            .iter()
///            .map(|&x| exp_function(x))
///            .collect::<Vec<f64>>();
///        let initial_guess = vec![1.0, 1.0];
///        let unknown_coeffs = vec!["a".to_string(), "b".to_string()];
///        let eq = " exp(a*x) + b".to_string();
///        let mut sym_fitting = Fitting::new();
///        sym_fitting.fitting_generate_from_str(
///            x_data,
///            y_data,
///            eq,
///            Some(unknown_coeffs),
///            "x".to_string(),
///            initial_guess,
///            None,
///            None,
///            None,
///            None,
///            None,
///        );
///        sym_fitting.eq_generate();
///        sym_fitting.solve();
///        let map_of_solutions = sym_fitting.map_of_solutions.unwrap();
///        assert_relative_eq!(map_of_solutions["a"], 1e-1, epsilon = 1e-6);
///        assert_relative_eq!(map_of_solutions["b"], 10.0, epsilon = 1e-6);
///```
/// Example#2
/// ```
///  use approx::assert_relative_eq;
/// use RustedSciThe::numerical::optimization::sym_fitting::Fitting;
///        let x_data = (0..100).map(|x| x as f64).collect::<Vec<f64>>();
///        let quadratic_function = |x: f64| 5.0 * x * x + 2.0 * x + 100.0;
///        let y_data = x_data
///            .iter()
///            .map(|&x| quadratic_function(x))
///            .collect::<Vec<f64>>();
///        let initial_guess = vec![1.0, 1.0, 1.0];
///        let unknown_coeffs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
///        let eq = "a * x^2.0 + b * x + c".to_string();
///        let mut sym_fitting = Fitting::new();
///        sym_fitting.easy_fitting(
///            x_data,
///            y_data,
///            eq,
///            Some(unknown_coeffs),
///            "x".to_string(),
///            initial_guess,
///        );
///        let map_of_solutions = sym_fitting.map_of_solutions.unwrap();
///        assert_relative_eq!(map_of_solutions["a"], 5.0, epsilon = 1e-6);
///        assert_relative_eq!(map_of_solutions["b"], 2.0, epsilon = 1e-6);
///        assert_relative_eq!(map_of_solutions["c"], 100.0, epsilon = 1e-6);
///```
pub mod sym_fitting;
/// solver of nonlinear equation system with Levenberg-Marquardt algorithm is
/// used with residual functions that are symbolic expressions and jacobian functions that are calculated analytically
/// 
/// Example#1
/// ```
///  use approx::assert_relative_eq;
/// use RustedSciThe::numerical::optimization::sym_wrapper::LM;
///         let vec_of_str = vec!["x^2 + y^2 - 1".to_string(), "x - y".to_string()];
///        let initial_guess = vec![0.5, 0.5];
///        let values = vec!["x".to_string(), "y".to_string()];
///        let mut LM = LM::new();
///        LM.eq_generate_from_str(
///            vec_of_str,
///            Some(values),
///            initial_guess,
///            None,
///            None,
///            None,
///            None,
///            None,
///        );
///        LM.eq_generate();
///        LM.solve();
///  ```        
pub mod sym_wrapper;
/// trust region subproblem solver for Levenberg-Marquardt algorithm
#[allow(non_snake_case)]
pub mod trust_region_LM;
/// some utility functions for solving nonlinear equation system with Levenberg-Marquardt algorithm
pub mod utils;
/// not production ready
pub mod Gavin_chi;

/// interpolation and extrapolation of data
pub mod inter_n_extrapolate;
 mod lm_gavin;
/// H.P.Gavin's Levenberg-Marquardt algorithm
/// 
/// Example#1
/// ```
///  use RustedSciThe::numerical::optimization::lm_gavin2::{LevenbergMarquardt, PolynomialModel};
/// use nalgebra::DVector;
/// use nalgebra::dvector;
///  use crate::RustedSciThe::numerical::optimization::lm_gavin2::ObjectiveFunction;
///  // you can define your own objective function
///   let model = PolynomialModel::new(2); // Quadratic
///        let mut lm = LevenbergMarquardt::new(model);
///
///        let t = DVector::from_vec((0..10).map(|i| i as f64).collect());
///        let p_true = dvector![1.0, 2.0, 0.5]; // 1 + 2x + 0.5x^2
///        let y_true = lm.objective_fn.evaluate(&t, &p_true);
///
///        let p_initial = dvector![0.8, 1.8, 0.4];
///
///        match lm.lm(p_initial, &t, &y_true) {
///            Ok((p_fitted, red_x2, _sigma_p, _sigma_y, _corr_p, _r_sq, _cvg_hst)) => {
///                println!("Fitted polynomial parameters: {:?}", p_fitted);
///                println!("Reduced chi-squared: {}", red_x2);
///                assert!(red_x2 < 1e-10);
///            }
///            Err(e) => panic!("Polynomial LM fitting failed: {}", e),
///        }
///  ```
pub mod lm_gavin2;
/// using Bisection, Secant, Newton, and Brent methods to find the minimum of a scalar function of one variable
/// 
/// Example#1
/// ```
/// use RustedSciThe:: numerical::optimization::minimize_scalar::{ScalarRootFinder, RootFindingMethod, approx_equal};
///         let solver = ScalarRootFinder::new();
///
///        // Solve x^3 - x - 1 = 0, root approximately at x = 1.324717957
///        let result = solver
///            .solve_symbolic_str(
///                "x^3 - (x + 1)",
///                "x",
///                RootFindingMethod::Bisection,
///                1.5,                // initial guess
///                Some((-10.0, 3.0)), // search range
///                None,
///            )
///            .unwrap();
///        println!("result.root: {}", result.root);
///        let expected_root = 1.324717957244746;
///        assert!(approx_equal(result.root, expected_root, 1e-9));
///        assert!(result.converged);
///  ```
/// 
/// Example#2
/// ```
///  use RustedSciThe::numerical::optimization::minimize_scalar::{ScalarRootFinder, RootFindingMethod, approx_equal};
///         let solver = ScalarRootFinder::new();
///
///        // Solve exp(x) - 2 = 0, root at x = ln(2)
///        let result = solver
///            .solve_symbolic_str(
///                "exp(x) - 2",
///                "x",
///                RootFindingMethod::NewtonRaphson,
///                1.0, // initial guess
///                None,
///                None,
///            )
///            .unwrap();
///
///        let expected_root = 2.0_f64.ln();
///        assert!(approx_equal(result.root, expected_root, 1e-10));
///        assert!(result.converged);
///    
///  ```
pub mod minimize_scalar;
