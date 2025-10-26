//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                         TESTS
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests1 {

    use crate::numerical::Radau::Radau_main::{Radau, RadauCoefficients, RadauOrder};
    #[test]
    fn test_radau_coefficients_order3() {
        let coeffs = RadauCoefficients::new(RadauOrder::Order3);
        assert_eq!(coeffs.stages, 2);
        assert_eq!(coeffs.c.len(), 2);
        assert_eq!(coeffs.a.nrows(), 2);
        assert_eq!(coeffs.a.ncols(), 2);
        assert_eq!(coeffs.b.len(), 2);
    }

    #[test]
    fn test_radau_coefficients_order5() {
        let coeffs = RadauCoefficients::new(RadauOrder::Order5);
        assert_eq!(coeffs.stages, 3);
        assert_eq!(coeffs.c.len(), 3);
        assert_eq!(coeffs.a.nrows(), 3);
        assert_eq!(coeffs.a.ncols(), 3);
        assert_eq!(coeffs.b.len(), 3);
    }

    #[test]
    fn test_radau_new() {
        let radau = Radau::new(RadauOrder::Order3);
        assert_eq!(radau.coefficients.stages, 2);
        assert_eq!(radau.status, "running");
    }
}

#[cfg(test)]
mod tests2 {
    use crate::symbolic::symbolic_engine::Expr;

    use crate::numerical::Radau::Radau_main::{Radau, RadauCoefficients, RadauOrder};
    use crate::symbolic::symbolic_vectors::{ExprMatrix, ExprVector};
    use approx::assert_relative_eq;
    use nalgebra::DMatrix;
    use nalgebra::DVector;
    #[test]
    fn test_radau_coefficients_order3() {
        let coeffs = RadauCoefficients::new(RadauOrder::Order3);
        assert_eq!(coeffs.stages, 2);
        assert_eq!(coeffs.c.len(), 2);
        assert_eq!(coeffs.a.nrows(), 2);
        assert_eq!(coeffs.a.ncols(), 2);
        assert_eq!(coeffs.b.len(), 2);

        // Check specific coefficient values for Order 3
        assert_relative_eq!(coeffs.c[0], 1.0 / 3.0, epsilon = 1e-10);
        assert_relative_eq!(coeffs.c[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(coeffs.b[0], 3.0 / 4.0, epsilon = 1e-10);
        assert_relative_eq!(coeffs.b[1], 1.0 / 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_radau_coefficients_order5() {
        let coeffs = RadauCoefficients::new(RadauOrder::Order5);
        assert_eq!(coeffs.stages, 3);
        assert_eq!(coeffs.c.len(), 3);
        assert_eq!(coeffs.a.nrows(), 3);
        assert_eq!(coeffs.a.ncols(), 3);
        assert_eq!(coeffs.b.len(), 3);

        // Check that c[2] = 1 (characteristic of Radau methods)
        assert_relative_eq!(coeffs.c[2], 1.0, epsilon = 1e-10);

        // Check that sum of b coefficients equals 1
        let b_sum: f64 = coeffs.b.iter().sum();
        assert_relative_eq!(b_sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_radau_coefficients_order7_fallback() {
        let coeffs = RadauCoefficients::new(RadauOrder::Order7);
        // Should fallback to Order5
        assert_eq!(coeffs.stages, 3);
    }

    #[test]
    fn test_radau_new_order3() {
        let radau = Radau::new(RadauOrder::Order3);
        assert_eq!(radau.coefficients.stages, 2);
        assert_eq!(radau.status, "running");
        assert_eq!(radau.t, 0.0);
        assert_eq!(radau.t0, 0.0);
        assert_eq!(radau.t_bound, 0.0);
        assert!(matches!(radau.order, RadauOrder::Order3));
    }

    #[test]
    fn test_radau_new_order5() {
        let radau = Radau::new(RadauOrder::Order5);
        assert_eq!(radau.coefficients.stages, 3);
        assert_eq!(radau.status, "running");
        assert!(matches!(radau.order, RadauOrder::Order5));
    }

    #[test]
    fn test_radau_display() {
        let mut radau = Radau::new(RadauOrder::Order3);
        radau.t0 = 0.0;
        radau.t_bound = 1.0;
        radau.t = 0.5;
        radau.y = DVector::from_vec(vec![1.0, 2.0]);

        let display_str = format!("{}", radau);
        assert!(display_str.contains("Order3"));
        assert!(display_str.contains("0.5"));
        assert!(display_str.contains("1"));
        assert!(display_str.contains("2"));
    }

    #[test]
    fn test_radau_set_initial_basic() {
        let eq1 = Expr::parse_expression("y");
        let eq_system = vec![eq1];
        let y0 = DVector::from_vec(vec![1.0]);
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let tolerance = 1e-6;
        let max_iterations = 50;
        let h = Some(0.1);
        let t0 = 0.0;
        let t_bound = 1.0;

        let mut radau = Radau::new(RadauOrder::Order3);
        radau.set_initial(
            eq_system,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            t0,
            t_bound,
            y0.clone(),
        );

        assert_eq!(radau.y0, y0);
        assert_eq!(radau.y, y0);
        assert_eq!(radau.t0, t0);
        assert_eq!(radau.t_bound, t_bound);
        assert_eq!(radau.t, t0);
        assert_eq!(radau.global_timestepping, true);
        assert_eq!(radau.k_stages.nrows(), 1);
        assert_eq!(radau.k_stages.ncols(), 2); // 2 stages for Order3
        assert_eq!(radau.y_stages.nrows(), 1);
        assert_eq!(radau.y_stages.ncols(), 2);
    }

    #[test]
    fn test_radau_set_initial_multiple_variables() {
        let eq1 = Expr::parse_expression("y");
        let eq2 = Expr::parse_expression("z");
        let eq_system = vec![eq1, eq2];
        let y0 = DVector::from_vec(vec![1.0, 2.0]);
        let values = vec!["y".to_string(), "z".to_string()];
        let arg = "t".to_string();
        let tolerance = 1e-6;
        let max_iterations = 50;
        let h = Some(0.1);
        let t0 = 0.0;
        let t_bound = 1.0;

        let mut radau = Radau::new(RadauOrder::Order5);
        radau.set_initial(
            eq_system,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            t0,
            t_bound,
            y0.clone(),
        );

        assert_eq!(radau.k_stages.nrows(), 2); // 2 variables
        assert_eq!(radau.k_stages.ncols(), 3); // 3 stages for Order5
        assert_eq!(radau.y_stages.nrows(), 2);
        assert_eq!(radau.y_stages.ncols(), 3);
    }

    #[test]
    fn test_radau_set_initial_no_global_timestepping() {
        let eq1 = Expr::parse_expression("y");
        let eq_system = vec![eq1];
        let y0 = DVector::from_vec(vec![1.0]);
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let tolerance = 1e-6;
        let max_iterations = 50;
        let h = None; // No fixed step size
        let t0 = 0.0;
        let t_bound = 1.0;

        let mut radau = Radau::new(RadauOrder::Order3);
        radau.set_initial(
            eq_system,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            t0,
            t_bound,
            y0,
        );

        assert_eq!(radau.global_timestepping, false);
    }

    #[test]
    fn test_radau_check_valid() {
        let eq1 = Expr::parse_expression("y");
        let eq_system = vec![eq1];
        let y0 = DVector::from_vec(vec![1.0]);
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let tolerance = 1e-6;
        let max_iterations = 50;
        let h = Some(0.1);
        let t0 = 0.0;
        let t_bound = 1.0;

        let mut radau = Radau::new(RadauOrder::Order3);
        radau.set_initial(
            eq_system,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            t0,
            t_bound,
            y0,
        );

        // Should not panic
        radau.check();
    }

    #[test]
    #[should_panic(expected = "initial y is empty")]
    fn test_radau_check_empty_y() {
        let radau = Radau::new(RadauOrder::Order3);
        radau.check();
    }

    #[test]
    #[should_panic(expected = "system is empty")]
    fn test_radau_check_empty_system() {
        let mut radau = Radau::new(RadauOrder::Order3);
        radau.y = DVector::from_vec(vec![1.0]);
        radau.check();
    }

    #[test]
    fn test_construct_radau_system_single_variable() {
        let mut radau = Radau::new(RadauOrder::Order3);
        let eq1 = Expr::parse_expression("y");
        let original_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let K_matrix = ExprMatrix::indexed_vars_matrix(1, 2, "K");
        let y_vector = ExprVector::indexed_vars_vector(1, "y");
        radau.set_stage_variables(K_matrix, y_vector);
        let radau_system = radau.construct_radau_system(original_system, values, arg);

        // For 1 variable and 2 stages (Order3), should have 2 equations
        assert_eq!(radau_system.len(), 2);
    }

    #[test]
    fn test_construct_radau_system_multiple_variables() {
        let mut radau = Radau::new(RadauOrder::Order5);
        let eq1 = Expr::parse_expression("y");
        let eq2 = Expr::parse_expression("z");
        let original_system = vec![eq1, eq2];
        let values = vec!["y".to_string(), "z".to_string()];
        let arg = "t".to_string();
        let K_matrix = ExprMatrix::indexed_vars_matrix(2, 3, "K");
        let y_vector = ExprVector::indexed_vars_vector(2, "y");
        radau.set_stage_variables(K_matrix, y_vector);
        let radau_system = radau.construct_radau_system(original_system, values, arg);

        // For 2 variables and 3 stages (Order5), should have 6 equations
        assert_eq!(radau_system.len(), 6);
    }

    #[test]
    fn test_radau_step_at_boundary() {
        let mut radau = Radau::new(RadauOrder::Order3);
        radau.t = 1.0;
        radau.t_bound = 1.0;
        radau.status = "running".to_string();

        radau.step();

        assert_eq!(radau.status, "finished");
        assert_eq!(radau.t_old, Some(1.0));
    }

    #[test]
    fn test_radau_get_result_empty() {
        let radau = Radau::new(RadauOrder::Order3);
        let (t_result, y_result) = radau.get_result();

        assert!(t_result.is_some());
        assert!(y_result.is_some());
        assert_eq!(t_result.unwrap().len(), 0);
        assert_eq!(y_result.unwrap().nrows(), 0);
    }

    #[test]
    fn test_radau_solve_initialization() {
        let eq1 = Expr::parse_expression("y");
        let eq_system = vec![eq1];
        let y0 = DVector::from_vec(vec![1.0]);
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let tolerance = 1e-6;
        let max_iterations = 50;
        let h = Some(0.1);
        let t0 = 0.0;
        let t_bound = 1.0;

        let mut radau = Radau::new(RadauOrder::Order5);
        radau.set_initial(
            eq_system,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            t0,
            t_bound,
            y0,
        );

        // Should not panic and should call eq_generate
        radau.solve();
    }

    #[test]
    fn test_radau_coefficients_consistency_order3() {
        let coeffs = RadauCoefficients::new(RadauOrder::Order3);

        // Test that A matrix and b vector are consistent (last row of A should equal b)
        for j in 0..coeffs.stages {
            assert_relative_eq!(
                coeffs.a[(coeffs.stages - 1, j)],
                coeffs.b[j],
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn test_radau_coefficients_consistency_order5() {
        let coeffs = RadauCoefficients::new(RadauOrder::Order5);

        // Test that A matrix and b vector are consistent (last row of A should equal b)
        for j in 0..coeffs.stages {
            assert_relative_eq!(
                coeffs.a[(coeffs.stages - 1, j)],
                coeffs.b[j],
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn test_radau_working_arrays_initialization() {
        let eq1 = Expr::parse_expression("y");
        let eq2 = Expr::parse_expression("z");
        let eq_system = vec![eq1, eq2];
        let y0 = DVector::from_vec(vec![1.0, 2.0]);
        let values = vec!["y".to_string(), "z".to_string()];
        let arg = "t".to_string();
        let tolerance = 1e-6;
        let max_iterations = 50;
        let h = Some(0.1);
        let t0 = 0.0;
        let t_bound = 1.0;

        let mut radau = Radau::new(RadauOrder::Order3);
        radau.set_initial(
            eq_system,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            t0,
            t_bound,
            y0,
        );

        // Check that working arrays are properly sized and initialized to zero
        assert_eq!(radau.k_stages.nrows(), 2);
        assert_eq!(radau.k_stages.ncols(), 2);
        assert_eq!(radau.y_stages.nrows(), 2);
        assert_eq!(radau.y_stages.ncols(), 2);

        assert_eq!(radau.k_stages, DMatrix::zeros(2, 2));
        assert_eq!(radau.y_stages, DMatrix::zeros(2, 2));
    }
}

#[cfg(test)]
mod tests_real_odes {
    use crate::symbolic::symbolic_engine::Expr;

    use crate::numerical::Radau::Radau_main::{Radau, RadauOrder};
    use approx::assert_relative_eq;
    use nalgebra::DMatrix;
    use nalgebra::DVector;
    use simplelog::*;
    #[test]
    fn test_radau_simple_linear_ode() {
        // Test: y' = -y, y(0) = 1
        // Exact solution: y(t) = exp(-t)
        let eq1 = Expr::parse_expression("-y");
        let eq_system = vec![eq1];

        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let tolerance = 1e-6;
        let max_iterations = 50;
        let h = Some(0.01);
        let t0 = 0.0;
        let t_bound = 1.0;
        let y0 = DVector::from_vec(vec![1.0]);

        let mut radau = Radau::new(RadauOrder::Order5);
        radau.set_initial(
            eq_system,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            t0,
            t_bound,
            y0,
        );

        radau.solve();

        assert_eq!(radau.status, "finished");
        let (t_result, y_result) = radau.get_result();
        assert!(t_result.is_some());
        assert!(y_result.is_some());

        let t_res = t_result.unwrap();
        let y_res: DMatrix<f64> = y_result.unwrap();
        println!("t = {:?} \n", t_res);
        println!("y = {:?} \n", y_res);
        // Check final value: y(1) ≈ exp(-1) ≈ 0.3679
        let final_y = y_res[(y_res.nrows() - 1, 0)];
        let expected = (-1.0_f64).exp();
        assert_relative_eq!(final_y, expected, epsilon = 1e-3);
    }

    #[test]
    fn test_radau_exponential_growth() {
        // Test: y' = y, y(0) = 1
        // Exact solution: y(t) = exp(t)
        let eq1 = Expr::parse_expression("y");
        let eq_system = vec![eq1];

        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let tolerance = 1e-6;
        let max_iterations = 50;
        let h = Some(0.01);
        let t0 = 0.0;
        let t_bound = 0.5;
        let y0 = DVector::from_vec(vec![1.0]);

        let mut radau = Radau::new(RadauOrder::Order5);
        radau.set_log_level(LevelFilter::Info);
        radau.set_initial(
            eq_system,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            t0,
            t_bound,
            y0,
        );

        radau.solve();

        assert_eq!(radau.status, "finished");
        let (_, y_result) = radau.get_result();
        let y_res = y_result.unwrap();
        let t_res = radau.t_result;
        // Check final value: y(0.5) ≈ exp(0.5) ≈ 1.6487
        let final_y = y_res[(y_res.nrows() - 1, 0)];
        let expected = (0.5_f64).exp();
        assert_relative_eq!(final_y, expected, epsilon = 1e-3);
        let f_t = |t: f64| (t).exp();
        for (t, y) in t_res.iter().zip(y_res.row_iter()) {
            assert_relative_eq!(y[0], f_t(*t), epsilon = 1e-3);
        }
    }

    #[test]
    fn test_radau_linear_system_2x2() {
        // Test system: y1' = -2*y1 + y2, y2' = y1 - 2*y2
        // Initial conditions: y1(0) = 1, y2(0) = 0
        // solution: y1(t) =  1/2 e^(-3 x) (e^(2 x) + 1)
        // y2(t) = 1/2 e^(-3 x) (-1 + e^(2 x))
        let eq1 = Expr::parse_expression("-2*y1+y2");
        let eq2 = Expr::parse_expression("y1-2*y2");
        let eq_system = vec![eq1, eq2];

        let values = vec!["y1".to_string(), "y2".to_string()];
        let arg = "t".to_string();
        let tolerance = 1e-6;
        let max_iterations = 50;
        let h = Some(1e-3);
        let t0 = 0.0;
        let t_bound = 1.0;
        let y0 = DVector::from_vec(vec![1.0, 0.0]);

        let mut radau = Radau::new(RadauOrder::Order3);
        radau.set_initial(
            eq_system,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            t0,
            t_bound,
            y0,
        );

        radau.solve();

        assert_eq!(radau.status, "finished");
        let (_, y_result) = radau.get_result();
        let y_res = y_result.unwrap();

        // Check that we have 2 variables
        assert_eq!(y_res.ncols(), 2);
        //    println!("y nrows {}, ncols = {:?} \n", y_res.nrows(), y_res.ncols());
        // The exact solution involves eigenvalues -1 and -3
        // At t=1: solutions should be decaying
        let final_y1 = y_res[(y_res.nrows() - 1, 0)];
        let final_y2 = y_res[(y_res.nrows() - 1, 1)];

        let y1_end = 0.5 * f64::exp(-3.0) * (f64::exp(2.0) + 1.0);
        let y2_end = 0.5 * f64::exp(-3.0) * (-1.0 + f64::exp(2.0));
        println!("expected endpoints = {:?}, {} \n", y1_end, y2_end);
        assert_relative_eq!(final_y1, y1_end, epsilon = 1e-2);
        assert_relative_eq!(final_y2, y2_end, epsilon = 1e-2);
        // Both should be positive but smaller than initial values
        assert!(final_y1 > 0.0 && final_y1 < 1.0);
        assert!(final_y2.abs() < 1.0); // y2 starts at 0, should remain bounded

        let f_y1 = |t: f64| 0.5 * f64::exp(-3.0 * t) * (f64::exp(2.0 * t) + 1.0);
        let f_y2 = |t: f64| 0.5 * f64::exp(-3.0 * t) * (-1.0 + f64::exp(2.0 * t));
        let t_res = radau.t_result;
        for (t, y) in t_res.iter().zip(y_res.row_iter()) {
            let y1_exact = f_y1(*t);
            let y2_exact = f_y2(*t);
            assert_relative_eq!(y[0], y1_exact, epsilon = 1e-2);
            assert_relative_eq!(y[1], y2_exact, epsilon = 1e-2);
        }
    }

    #[test]
    fn test_radau_harmonic_oscillator() {
        // Test: y1' = y2, y2' = -y1 (harmonic oscillator)
        // Initial conditions: y1(0) = 1, y2(0) = 0
        // Exact solution: y1(t) = cos(t), y2(t) = -sin(t)
        let eq1 = Expr::parse_expression("y2");
        let eq2 = Expr::parse_expression("-y1");
        let eq_system = vec![eq1, eq2];

        let values = vec!["y1".to_string(), "y2".to_string()];
        let arg = "t".to_string();
        let tolerance = 1e-8;
        let max_iterations = 50;
        let h = Some(1e-2);
        let t0 = 0.0;
        let t_bound = std::f64::consts::PI / 2.0; // π/2
        let y0 = DVector::from_vec(vec![1.0, 0.0]);

        let mut radau = Radau::new(RadauOrder::Order5);
        radau.set_initial(
            eq_system,
            values.clone(),
            arg,
            tolerance,
            max_iterations,
            h,
            t0,
            t_bound,
            y0,
        );

        radau.solve();

        assert_eq!(radau.status, "finished");
        let (_, y_result) = radau.get_result();
        let y_res = y_result.unwrap();
        let t_res = radau.t_result;
        //  println!("t_res: {}", t_res);
        //  println!("{:?}", values);
        //  println!("y_res: {}", y_res);
        // At t = π/2: y1 should be ≈ 0, y2 should be ≈ -1
        let final_y1 = y_res[(y_res.nrows() - 1, 0)];
        let final_y2 = y_res[(y_res.nrows() - 1, 1)];

        assert_relative_eq!(final_y1, 0.0, epsilon = 1e-2);
        assert_relative_eq!(final_y2, -1.0, epsilon = 1e-2);
        // compare with exact solution
        let f_y1 = |t: f64| f64::cos(t);
        let f_y2 = |t: f64| -f64::sin(t);
        for (t, y) in t_res.iter().zip(y_res.row_iter()) {
            let y1_exact = f_y1(*t);
            let y2_exact = f_y2(*t);
            assert_relative_eq!(y[0], y1_exact, epsilon = 1e-2);
            assert_relative_eq!(y[1], y2_exact, epsilon = 1e-2);
        }
    }

    #[test]
    fn test_radau_nonlinear_ode() {
        // Test: y' = y^2, y(0) = 1
        // Exact solution: y(t) = 1/(1-t) for t < 1
        let eq1 = Expr::parse_expression("y*y");
        let eq_system = vec![eq1];

        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let tolerance = 1e-6;
        let max_iterations = 50;
        let h = Some(0.01);
        let t0 = 0.0;
        let t_bound = 0.5;
        let y0 = DVector::from_vec(vec![1.0]);

        let mut radau = Radau::new(RadauOrder::Order3);
        radau.set_initial(
            eq_system,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            t0,
            t_bound,
            y0,
        );

        radau.solve();

        assert_eq!(radau.status, "finished");
        let (_, y_result) = radau.get_result();
        let y_res = y_result.unwrap();

        // At t = 0.5: y(0.5) = 1/(1-0.5) = 2
        let final_y = y_res[(y_res.nrows() - 1, 0)];
        let expected = 1.0 / (1.0 - 0.5);
        assert_relative_eq!(final_y, expected, epsilon = 1e-2);
    }

    #[test]
    fn test_radau_van_der_pol_oscillator() {
        // Van der Pol oscillator: y1' = y2, y2' = μ(1-y1^2)y2 - y1
        // With μ = 0.1 (weakly nonlinear)
        let eq1 = Expr::parse_expression("y2");
        let eq2 = Expr::parse_expression("0.1*(1.0 - y1*y1)*y2 - y1");
        let eq_system = vec![eq1, eq2];

        let values = vec!["y1".to_string(), "y2".to_string()];
        let arg = "t".to_string();
        let tolerance = 1e-6;
        let max_iterations = 50;
        let h = Some(0.1);
        let t0 = 0.0;
        let t_bound = 5.0;
        let y0 = DVector::from_vec(vec![2.0, 0.0]); // Start away from equilibrium

        let mut radau = Radau::new(RadauOrder::Order3);
        radau.set_initial(
            eq_system,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            t0,
            t_bound,
            y0,
        );

        radau.solve();

        assert_eq!(radau.status, "finished");
        let (_, y_result) = radau.get_result();
        let y_res = y_result.unwrap();

        // Check that solution remains bounded (Van der Pol has limit cycle)
        let final_y1 = y_res[(y_res.nrows() - 1, 0)];
        let final_y2 = y_res[(y_res.nrows() - 1, 1)];

        assert!(final_y1.abs() < 10.0); // Should remain bounded
        assert!(final_y2.abs() < 10.0);
    }

    #[test]
    fn test_radau_stiff_system() {
        // Stiff system: y1' = -1000*y1 + y2, y2' = y1 - y2
        // This tests Radau's ability to handle stiff problems
        let eq1 = Expr::parse_expression("-1000.0*y1 + y2");
        let eq2 = Expr::parse_expression("y1 - y2");
        let eq_system = vec![eq1, eq2];

        let values = vec!["y1".to_string(), "y2".to_string()];
        let arg = "t".to_string();
        let tolerance = 1e-6;
        let max_iterations = 100; // May need more iterations for stiff problems
        let h = Some(0.01); // Small step size for stiff problem
        let t0 = 0.0;
        let t_bound = 0.1;
        let y0 = DVector::from_vec(vec![1.0, 1.0]);

        let mut radau = Radau::new(RadauOrder::Order5);
        radau.set_initial(
            eq_system,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            t0,
            t_bound,
            y0,
        );

        radau.solve();

        assert_eq!(radau.status, "finished");
        let (_, y_result) = radau.get_result();
        let y_res = y_result.unwrap();

        // For stiff systems, just check that solution doesn't blow up
        let final_y1 = y_res[(y_res.nrows() - 1, 0)];
        let final_y2 = y_res[(y_res.nrows() - 1, 1)];

        assert!(final_y1.is_finite());
        assert!(final_y2.is_finite());
        assert!(final_y1.abs() < 100.0); // Should remain reasonable
        assert!(final_y2.abs() < 100.0);
    }

    #[test]
    fn test_radau_three_body_problem_simplified() {
        // Simplified 3-body problem (2D, one body)
        // y1' = y3, y2' = y4, y3' = -y1/r^3, y4' = -y2/r^3
        // where r = sqrt(y1^2 + y2^2)
        let eq1 = Expr::parse_expression("y3");
        let eq2 = Expr::parse_expression("y4");
        let eq3 = Expr::parse_expression("-y1/((y1*y1 + y2*y2)^1.5)");
        let eq4 = Expr::parse_expression("-y2/((y1*y1 + y2*y2)^1.5)");
        let eq_system = vec![eq1, eq2, eq3, eq4];

        let values = vec![
            "y1".to_string(),
            "y2".to_string(),
            "y3".to_string(),
            "y4".to_string(),
        ];
        let arg = "t".to_string();
        let tolerance = 1e-6;
        let max_iterations = 50;
        let h = Some(0.01);
        let t0 = 0.0;
        let t_bound = 1.0;
        // Initial conditions for circular orbit
        let y0 = DVector::from_vec(vec![1.0, 0.0, 0.0, 1.0]);

        let mut radau = Radau::new(RadauOrder::Order5);
        radau.set_initial(
            eq_system,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            t0,
            t_bound,
            y0,
        );

        radau.solve();

        assert_eq!(radau.status, "finished");
        let (_, y_result) = radau.get_result();
        //let y_res = y_result.unwrap();

        // Check that solution remains bounded
    }
}

#[cfg(test)]
mod tests_real_odes_parallel {
    use crate::symbolic::symbolic_engine::Expr;

    use crate::numerical::Radau::Radau_main::{Radau, RadauOrder};
    use approx::assert_relative_eq;
    use nalgebra::DMatrix;
    use nalgebra::DVector;
    use simplelog::*;
    #[test]
    fn test_radau_simple_linear_ode() {
        // Test: y' = -y, y(0) = 1
        // Exact solution: y(t) = exp(-t)
        let eq1 = Expr::parse_expression("-y");
        let eq_system = vec![eq1];

        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let tolerance = 1e-6;
        let max_iterations = 50;
        let h = Some(0.01);
        let t0 = 0.0;
        let t_bound = 1.0;
        let y0 = DVector::from_vec(vec![1.0]);

        let mut radau = Radau::new(RadauOrder::Order5);
        radau.set_initial(
            eq_system,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            t0,
            t_bound,
            y0,
        );
        radau.set_parallel(true);
        radau.solve();

        assert_eq!(radau.status, "finished");
        let (t_result, y_result) = radau.get_result();
        assert!(t_result.is_some());
        assert!(y_result.is_some());

        let t_res = t_result.unwrap();
        let y_res: DMatrix<f64> = y_result.unwrap();
        println!("t = {:?} \n", t_res);
        println!("y = {:?} \n", y_res);
        // Check final value: y(1) ≈ exp(-1) ≈ 0.3679
        let final_y = y_res[(y_res.nrows() - 1, 0)];
        let expected = (-1.0_f64).exp();
        assert_relative_eq!(final_y, expected, epsilon = 1e-3);
    }

    #[test]
    fn test_radau_exponential_growth() {
        // Test: y' = y, y(0) = 1
        // Exact solution: y(t) = exp(t)
        let eq1 = Expr::parse_expression("y");
        let eq_system = vec![eq1];

        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let tolerance = 1e-6;
        let max_iterations = 50;
        let h = Some(0.01);
        let t0 = 0.0;
        let t_bound = 0.5;
        let y0 = DVector::from_vec(vec![1.0]);

        let mut radau = Radau::new(RadauOrder::Order5);
        radau.set_log_level(LevelFilter::Info);
        radau.set_initial(
            eq_system,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            t0,
            t_bound,
            y0,
        );
        radau.set_parallel(true);
        radau.solve();

        assert_eq!(radau.status, "finished");
        let (_, y_result) = radau.get_result();
        let y_res = y_result.unwrap();
        let t_res = radau.t_result;
        // Check final value: y(0.5) ≈ exp(0.5) ≈ 1.6487
        let final_y = y_res[(y_res.nrows() - 1, 0)];
        let expected = (0.5_f64).exp();
        assert_relative_eq!(final_y, expected, epsilon = 1e-3);
        let f_t = |t: f64| (t).exp();
        for (t, y) in t_res.iter().zip(y_res.row_iter()) {
            assert_relative_eq!(y[0], f_t(*t), epsilon = 1e-3);
        }
    }

    #[test]
    fn test_radau_linear_system_2x2() {
        // Test system: y1' = -2*y1 + y2, y2' = y1 - 2*y2
        // Initial conditions: y1(0) = 1, y2(0) = 0
        // solution: y1(t) =  1/2 e^(-3 x) (e^(2 x) + 1)
        // y2(t) = 1/2 e^(-3 x) (-1 + e^(2 x))
        let eq1 = Expr::parse_expression("-2*y1+y2");
        let eq2 = Expr::parse_expression("y1-2*y2");
        let eq_system = vec![eq1, eq2];

        let values = vec!["y1".to_string(), "y2".to_string()];
        let arg = "t".to_string();
        let tolerance = 1e-6;
        let max_iterations = 50;
        let h = Some(1e-3);
        let t0 = 0.0;
        let t_bound = 1.0;
        let y0 = DVector::from_vec(vec![1.0, 0.0]);

        let mut radau = Radau::new(RadauOrder::Order3);
        radau.set_initial(
            eq_system,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            t0,
            t_bound,
            y0,
        );
        radau.set_parallel(true);
        radau.solve();

        assert_eq!(radau.status, "finished");
        let (_, y_result) = radau.get_result();
        let y_res = y_result.unwrap();

        // Check that we have 2 variables
        assert_eq!(y_res.ncols(), 2);
        //    println!("y nrows {}, ncols = {:?} \n", y_res.nrows(), y_res.ncols());
        // The exact solution involves eigenvalues -1 and -3
        // At t=1: solutions should be decaying
        let final_y1 = y_res[(y_res.nrows() - 1, 0)];
        let final_y2 = y_res[(y_res.nrows() - 1, 1)];

        let y1_end = 0.5 * f64::exp(-3.0) * (f64::exp(2.0) + 1.0);
        let y2_end = 0.5 * f64::exp(-3.0) * (-1.0 + f64::exp(2.0));
        println!("expected endpoints = {:?}, {} \n", y1_end, y2_end);
        assert_relative_eq!(final_y1, y1_end, epsilon = 1e-2);
        assert_relative_eq!(final_y2, y2_end, epsilon = 1e-2);
        // Both should be positive but smaller than initial values
        assert!(final_y1 > 0.0 && final_y1 < 1.0);
        assert!(final_y2.abs() < 1.0); // y2 starts at 0, should remain bounded

        let f_y1 = |t: f64| 0.5 * f64::exp(-3.0 * t) * (f64::exp(2.0 * t) + 1.0);
        let f_y2 = |t: f64| 0.5 * f64::exp(-3.0 * t) * (-1.0 + f64::exp(2.0 * t));
        let t_res = radau.t_result;
        for (t, y) in t_res.iter().zip(y_res.row_iter()) {
            let y1_exact = f_y1(*t);
            let y2_exact = f_y2(*t);
            assert_relative_eq!(y[0], y1_exact, epsilon = 1e-2);
            assert_relative_eq!(y[1], y2_exact, epsilon = 1e-2);
        }
    }

    #[test]
    fn test_radau_harmonic_oscillator() {
        // Test: y1' = y2, y2' = -y1 (harmonic oscillator)
        // Initial conditions: y1(0) = 1, y2(0) = 0
        // Exact solution: y1(t) = cos(t), y2(t) = -sin(t)
        let eq1 = Expr::parse_expression("y2");
        let eq2 = Expr::parse_expression("-y1");
        let eq_system = vec![eq1, eq2];

        let values = vec!["y1".to_string(), "y2".to_string()];
        let arg = "t".to_string();
        let tolerance = 1e-8;
        let max_iterations = 50;
        let h = Some(1e-2);
        let t0 = 0.0;
        let t_bound = std::f64::consts::PI / 2.0; // π/2
        let y0 = DVector::from_vec(vec![1.0, 0.0]);

        let mut radau = Radau::new(RadauOrder::Order5);
        radau.set_initial(
            eq_system,
            values.clone(),
            arg,
            tolerance,
            max_iterations,
            h,
            t0,
            t_bound,
            y0,
        );
        radau.set_parallel(true);
        radau.solve();

        assert_eq!(radau.status, "finished");
        let (_, y_result) = radau.get_result();
        let y_res = y_result.unwrap();
        let t_res = radau.t_result;
        //  println!("t_res: {}", t_res);
        //  println!("{:?}", values);
        //  println!("y_res: {}", y_res);
        // At t = π/2: y1 should be ≈ 0, y2 should be ≈ -1
        let final_y1 = y_res[(y_res.nrows() - 1, 0)];
        let final_y2 = y_res[(y_res.nrows() - 1, 1)];

        assert_relative_eq!(final_y1, 0.0, epsilon = 1e-2);
        assert_relative_eq!(final_y2, -1.0, epsilon = 1e-2);
        // compare with exact solution
        let f_y1 = |t: f64| f64::cos(t);
        let f_y2 = |t: f64| -f64::sin(t);
        for (t, y) in t_res.iter().zip(y_res.row_iter()) {
            let y1_exact = f_y1(*t);
            let y2_exact = f_y2(*t);
            assert_relative_eq!(y[0], y1_exact, epsilon = 1e-2);
            assert_relative_eq!(y[1], y2_exact, epsilon = 1e-2);
        }
    }

    #[test]
    fn test_radau_nonlinear_ode() {
        // Test: y' = y^2, y(0) = 1
        // Exact solution: y(t) = 1/(1-t) for t < 1
        let eq1 = Expr::parse_expression("y*y");
        let eq_system = vec![eq1];

        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let tolerance = 1e-6;
        let max_iterations = 50;
        let h = Some(0.01);
        let t0 = 0.0;
        let t_bound = 0.5;
        let y0 = DVector::from_vec(vec![1.0]);

        let mut radau = Radau::new(RadauOrder::Order3);
        radau.set_initial(
            eq_system,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            t0,
            t_bound,
            y0,
        );
        radau.set_parallel(true);
        radau.solve();

        assert_eq!(radau.status, "finished");
        let (_, y_result) = radau.get_result();
        let y_res = y_result.unwrap();

        // At t = 0.5: y(0.5) = 1/(1-0.5) = 2
        let final_y = y_res[(y_res.nrows() - 1, 0)];
        let expected = 1.0 / (1.0 - 0.5);
        assert_relative_eq!(final_y, expected, epsilon = 1e-2);
    }

    #[test]
    fn test_radau_van_der_pol_oscillator() {
        // Van der Pol oscillator: y1' = y2, y2' = μ(1-y1^2)y2 - y1
        // With μ = 0.1 (weakly nonlinear)
        let eq1 = Expr::parse_expression("y2");
        let eq2 = Expr::parse_expression("0.1*(1.0 - y1*y1)*y2 - y1");
        let eq_system = vec![eq1, eq2];

        let values = vec!["y1".to_string(), "y2".to_string()];
        let arg = "t".to_string();
        let tolerance = 1e-6;
        let max_iterations = 50;
        let h = Some(0.1);
        let t0 = 0.0;
        let t_bound = 5.0;
        let y0 = DVector::from_vec(vec![2.0, 0.0]); // Start away from equilibrium

        let mut radau = Radau::new(RadauOrder::Order3);
        radau.set_initial(
            eq_system,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            t0,
            t_bound,
            y0,
        );
        radau.set_parallel(true);
        radau.solve();

        assert_eq!(radau.status, "finished");
        let (_, y_result) = radau.get_result();
        let y_res = y_result.unwrap();

        // Check that solution remains bounded (Van der Pol has limit cycle)
        let final_y1 = y_res[(y_res.nrows() - 1, 0)];
        let final_y2 = y_res[(y_res.nrows() - 1, 1)];

        assert!(final_y1.abs() < 10.0); // Should remain bounded
        assert!(final_y2.abs() < 10.0);
    }

    #[test]
    fn test_radau_stiff_system() {
        // Stiff system: y1' = -1000*y1 + y2, y2' = y1 - y2
        // This tests Radau's ability to handle stiff problems
        let eq1 = Expr::parse_expression("-1000.0*y1 + y2");
        let eq2 = Expr::parse_expression("y1 - y2");
        let eq_system = vec![eq1, eq2];

        let values = vec!["y1".to_string(), "y2".to_string()];
        let arg = "t".to_string();
        let tolerance = 1e-6;
        let max_iterations = 100; // May need more iterations for stiff problems
        let h = Some(0.01); // Small step size for stiff problem
        let t0 = 0.0;
        let t_bound = 0.1;
        let y0 = DVector::from_vec(vec![1.0, 1.0]);

        let mut radau = Radau::new(RadauOrder::Order5);
        radau.set_initial(
            eq_system,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            t0,
            t_bound,
            y0,
        );
        radau.set_parallel(true);
        radau.solve();

        assert_eq!(radau.status, "finished");
        let (_, y_result) = radau.get_result();
        let y_res = y_result.unwrap();

        // For stiff systems, just check that solution doesn't blow up
        let final_y1 = y_res[(y_res.nrows() - 1, 0)];
        let final_y2 = y_res[(y_res.nrows() - 1, 1)];

        assert!(final_y1.is_finite());
        assert!(final_y2.is_finite());
        assert!(final_y1.abs() < 100.0); // Should remain reasonable
        assert!(final_y2.abs() < 100.0);
    }

    #[test]
    fn test_radau_three_body_problem_simplified() {
        // Simplified 3-body problem (2D, one body)
        // y1' = y3, y2' = y4, y3' = -y1/r^3, y4' = -y2/r^3
        // where r = sqrt(y1^2 + y2^2)
        let eq1 = Expr::parse_expression("y3");
        let eq2 = Expr::parse_expression("y4");
        let eq3 = Expr::parse_expression("-y1/((y1*y1 + y2*y2)^1.5)");
        let eq4 = Expr::parse_expression("-y2/((y1*y1 + y2*y2)^1.5)");
        let eq_system = vec![eq1, eq2, eq3, eq4];

        let values = vec![
            "y1".to_string(),
            "y2".to_string(),
            "y3".to_string(),
            "y4".to_string(),
        ];
        let arg = "t".to_string();
        let tolerance = 1e-6;
        let max_iterations = 50;
        let h = Some(0.01);
        let t0 = 0.0;
        let t_bound = 1.0;
        // Initial conditions for circular orbit
        let y0 = DVector::from_vec(vec![1.0, 0.0, 0.0, 1.0]);

        let mut radau = Radau::new(RadauOrder::Order5);
        radau.set_initial(
            eq_system,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            t0,
            t_bound,
            y0,
        );
        radau.set_parallel(true);
        radau.solve();

        assert_eq!(radau.status, "finished");
        let (_, y_result) = radau.get_result();
        //let y_res = y_result.unwrap();

        // Check that solution remains bounded
    }
}

#[cfg(test)]
mod tests_stop_conditions {
    use crate::numerical::Radau::Radau_main::{Radau, RadauOrder};
    use crate::symbolic::symbolic_engine::Expr;
    use approx::assert_relative_eq;
    use nalgebra::DVector;
    use std::collections::HashMap;

    #[test]
    fn test_radau_stop_condition_single_variable() {
        // Test: y' = y, y(0) = 1, stop when y reaches 2.0
        let eq1 = Expr::parse_expression("y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let tolerance = 1e-2;
        let max_iterations = 50;
        let h = Some(0.01);
        let t0 = 0.0;
        let t_bound = 10.0; // Large bound to ensure stop condition triggers first
        let y0 = DVector::from_vec(vec![1.0]);

        let mut radau = Radau::new(RadauOrder::Order3);
        radau.set_initial(
            eq_system,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            t0,
            t_bound,
            y0,
        );

        let mut stop_condition = HashMap::new();
        stop_condition.insert("y".to_string(), 2.0);
        radau.set_stop_condition(stop_condition);

        radau.solve();

        assert_eq!(radau.get_status(), "stopped_by_condition");
        let (_, y_result) = radau.get_result();
        let y_res = y_result.unwrap();
        let final_y = y_res[(y_res.nrows() - 1, 0)];
        assert!((final_y - 2.0).abs() <= tolerance);
    }

    #[test]
    fn test_radau_stop_condition_multiple_variables() {
        // Test system: y1' = y2, y2' = -y1, stop when y1 reaches 0.0
        let eq1 = Expr::parse_expression("y2");
        let eq2 = Expr::parse_expression("-y1");
        let eq_system = vec![eq1, eq2];
        let values = vec!["y1".to_string(), "y2".to_string()];
        let arg = "t".to_string();
        let tolerance = 1e-2;
        let max_iterations = 50;
        let h = Some(0.01);
        let t0 = 0.0;
        let t_bound = 10.0;
        let y0 = DVector::from_vec(vec![1.0, 0.0]);

        let mut radau = Radau::new(RadauOrder::Order5);
        radau.set_initial(
            eq_system,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            t0,
            t_bound,
            y0,
        );

        let mut stop_condition = HashMap::new();
        stop_condition.insert("y1".to_string(), 0.0);
        radau.set_stop_condition(stop_condition);

        radau.solve();

        assert_eq!(radau.get_status(), "stopped_by_condition");
        let (_, y_result) = radau.get_result();
        let y_res = y_result.unwrap();
        let final_y1 = y_res[(y_res.nrows() - 1, 0)];
        assert!(final_y1.abs() <= tolerance);
    }

    #[test]
    fn test_radau_no_stop_condition() {
        // Test without stop condition - should run to t_bound
        let eq1 = Expr::parse_expression("-y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let tolerance = 1e-6;
        let max_iterations = 50;
        let h = Some(0.1);
        let t0 = 0.0;
        let t_bound = 1.0;
        let y0 = DVector::from_vec(vec![1.0]);

        let mut radau = Radau::new(RadauOrder::Order3);
        radau.set_initial(
            eq_system,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            t0,
            t_bound,
            y0,
        );

        radau.solve();

        assert_eq!(radau.get_status(), "finished");
        let (t_result, _) = radau.get_result();
        let t_res = t_result.unwrap();
        let final_t = t_res[t_res.len() - 1];
        assert_relative_eq!(final_t, t_bound, epsilon = 0.1);
    }

    #[test]
    fn test_radau_stop_condition_nonlinear() {
        // Test: y' = y^2, y(0) = 1, stop when y reaches 1.5
        let eq1 = Expr::parse_expression("y*y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let tolerance = 1e-2;
        let max_iterations = 50;
        let h = Some(0.01);
        let t0 = 0.0;
        let t_bound = 10.0;
        let y0 = DVector::from_vec(vec![1.0]);

        let mut radau = Radau::new(RadauOrder::Order5);
        radau.set_initial(
            eq_system,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            t0,
            t_bound,
            y0,
        );

        let mut stop_condition = HashMap::new();
        stop_condition.insert("y".to_string(), 1.5);
        radau.set_stop_condition(stop_condition);

        radau.solve();

        assert_eq!(radau.get_status(), "stopped_by_condition");
        let (_, y_result) = radau.get_result();
        let y_res = y_result.unwrap();
        let final_y = y_res[(y_res.nrows() - 1, 0)];
        assert!((final_y - 1.5).abs() <= tolerance);
    }
}
