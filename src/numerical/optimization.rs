//! Optimization algorithms:
//! - Levenberg-Marquardt algorithm for:
//!         - solving nonlinear equation system
//!         - fitting data  
//! LM uses analytical Jacobian matrix
//! - H.P.Gavin's Levenberg-Marquardt algorithm
//!
/// not production ready
pub mod Gavin_chi;
/// Compatibility re-export; interpolation now lives in `numerical::interpolation`.
pub use crate::numerical::interpolation::ppoly as PPoly;
/// some special cases of fitting
pub mod fitting_features;
/// Compatibility re-export; interpolation now lives in `numerical::interpolation`.
pub use crate::numerical::interpolation::inter_n_extrapolate;
pub mod kinetic_fitting;
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
///        sym_fitting.solve().expect("fit should converge");
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
///        ).expect("fit should converge");
///        let map_of_solutions = sym_fitting.map_of_solutions.unwrap();
///        assert_relative_eq!(map_of_solutions["a"], 5.0, epsilon = 1e-6);
///        assert_relative_eq!(map_of_solutions["b"], 2.0, epsilon = 1e-6);
///        assert_relative_eq!(map_of_solutions["c"], 100.0, epsilon = 1e-6);
///```
pub mod sym_fitting;
pub mod universal_fitting;
pub mod varpro;
pub use universal_fitting::{Method, UniversalFitting, UniversalFittingResult};
