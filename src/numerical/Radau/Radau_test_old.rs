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

#[cfg(test)]
mod tests_statistics_and_options {
    use crate::numerical::Radau::Radau_main::{Radau, RadauOrder, RadauSolverOptions};
    use crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend;
    use crate::symbolic::symbolic_engine::Expr;
    use crate::symbolic::symbolic_ivp_generated::DenseIvpGeneratedBackendMode;
    use nalgebra::DMatrix;
    use nalgebra::DVector;
    use std::process::Command;
    use std::time::Instant;

    struct RadauComparisonRow {
        label: &'static str,
        total_ms: f64,
        prepare_ms: f64,
        solve_ms: f64,
        residual_ms_avg: f64,
        jacobian_ms_avg: f64,
        step_calls: usize,
        newton_solves: usize,
        newton_iters_avg: f64,
    }

    fn stiff_radau_options(parallel: bool) -> RadauSolverOptions {
        let eq_system = vec![
            Expr::parse_expression("-1000.0*y1 + y2"),
            Expr::parse_expression("y1 - y2"),
        ];
        let values = vec!["y1".to_string(), "y2".to_string()];

        RadauSolverOptions::new(
            RadauOrder::Order5,
            eq_system,
            values,
            "t".to_string(),
            1e-6,
            100,
            Some(0.01),
            0.0,
            0.1,
            DVector::from_vec(vec![1.0, 1.0]),
        )
        .with_parallel(parallel)
    }

    fn linear_decay_radau_options(parallel: bool) -> RadauSolverOptions {
        let eq_system = vec![Expr::parse_expression("-y")];
        let values = vec!["y".to_string()];

        RadauSolverOptions::new(
            RadauOrder::Order5,
            eq_system,
            values,
            "t".to_string(),
            1e-6,
            50,
            Some(0.01),
            0.0,
            1.0,
            DVector::from_vec(vec![1.0]),
        )
        .with_parallel(parallel)
    }

    fn solve_with_measurement(
        label: &'static str,
        options: RadauSolverOptions,
    ) -> RadauComparisonRow {
        let mut solver = Radau::new_with_options(options);
        let total_start = Instant::now();
        solver.solve();
        let total_ms = total_start.elapsed().as_secs_f64() * 1_000.0;
        let stats = solver.get_statistics();

        assert_eq!(
            solver.get_status(),
            "finished",
            "Radau comparison scenario `{label}` should finish successfully"
        );

        RadauComparisonRow {
            label,
            total_ms,
            prepare_ms: stats.backend_prepare_ms_total,
            solve_ms: stats.solve_ms_total,
            residual_ms_avg: stats.avg_residual_ms().unwrap_or(0.0),
            jacobian_ms_avg: stats.avg_jacobian_ms().unwrap_or(0.0),
            step_calls: stats.step_calls,
            newton_solves: stats.newton_solve_calls,
            newton_iters_avg: stats.avg_newton_iterations().unwrap_or(0.0),
        }
    }

    fn print_comparison_table(scenario: &str, rows: &[RadauComparisonRow]) {
        println!(
            "[Radau compare] scenario={}, variants={}",
            scenario,
            rows.len()
        );
        println!(
            "variant        | total_ms | setup_ms | solve_ms | residual_ms(avg) | jacobian_ms(avg) | steps | newton_solves | newton_iters(avg)"
        );
        println!(
            "---------------------------------------------------------------------------------------------------------------------------------"
        );
        for row in rows {
            println!(
                "{:<14} | {:>8.3} | {:>8.3} | {:>8.3} | {:>16.6} | {:>16.6} | {:>5} | {:>13} | {:>16.3}",
                row.label,
                row.total_ms,
                row.prepare_ms,
                row.solve_ms,
                row.residual_ms_avg,
                row.jacobian_ms_avg,
                row.step_calls,
                row.newton_solves,
                row.newton_iters_avg,
            );
        }
    }

    fn tcc_is_available() -> bool {
        if let Ok(explicit) = std::env::var("RUSTEDSCITHE_TCC") {
            return std::path::Path::new(&explicit).is_file();
        }
        Command::new("tcc").arg("-v").output().is_ok()
    }

    #[test]
    fn test_radau_new_with_options_preserves_parallel_flag() {
        let eq_system = vec![Expr::parse_expression("-y")];
        let values = vec!["y".to_string()];
        let options = RadauSolverOptions::new(
            RadauOrder::Order3,
            eq_system,
            values,
            "t".to_string(),
            1e-6,
            25,
            Some(0.1),
            0.0,
            0.5,
            DVector::from_vec(vec![1.0]),
        )
        .with_parallel(true);

        let solver = Radau::new_with_options(options);
        assert!(solver.parallel);
        assert!(solver.newton.parallel);
    }

    #[test]
    fn test_radau_statistics_are_collected_after_solve() {
        let eq_system = vec![Expr::parse_expression("-y")];
        let values = vec!["y".to_string()];
        let options = RadauSolverOptions::new(
            RadauOrder::Order3,
            eq_system,
            values,
            "t".to_string(),
            1e-6,
            25,
            Some(0.1),
            0.0,
            0.5,
            DVector::from_vec(vec![1.0]),
        );

        let mut solver = Radau::new_with_options(options);
        solver.solve();

        let stats = solver.get_statistics();
        let report = solver.statistics_report();
        assert!(stats.backend_prepare_calls >= 1);
        assert!(stats.solve_calls >= 1);
        assert!(stats.step_calls >= 1);
        assert!(stats.newton_solve_calls >= 1);
        assert!(stats.newton_iterations_total >= 1);
        assert!(stats.residual_calls >= 1);
        assert!(stats.jacobian_calls >= 1);
        assert!(stats.linear_solves >= 1);
        assert!(stats.lu_factorizations >= 1);
        assert!(report.contains("prepare_calls="));
        assert!(report.contains("newton_solves="));
    }

    #[test]
    fn test_radau_statistics_report_for_stiff_system() {
        let mut solver = Radau::new_with_options(stiff_radau_options(false));
        solver.solve();

        let stats = solver.get_statistics();
        let report = solver.statistics_report();
        println!("[Radau stats] sequential stiff-system => {}", report);

        assert_eq!(solver.get_status(), "finished");
        assert!(stats.backend_prepare_ms_total >= 0.0);
        assert!(stats.solve_ms_total > 0.0);
        assert!(stats.step_calls > 0);
        assert!(stats.newton_solve_calls >= stats.step_calls);
        assert!(stats.newton_iterations_total >= stats.newton_solve_calls);
        assert!(stats.residual_calls >= stats.newton_iterations_total);
        assert!(stats.jacobian_calls >= stats.newton_iterations_total);
        assert!(stats.avg_residual_ms().unwrap_or(0.0) >= 0.0);
        assert!(stats.avg_jacobian_ms().unwrap_or(0.0) >= 0.0);
        assert!(report.contains("residual_ms_avg="));
        assert!(report.contains("jacobian_ms_avg="));
    }

    #[test]
    fn test_radau_parallel_and_sequential_statistics_remain_consistent() {
        let mut sequential = Radau::new_with_options(stiff_radau_options(false));
        sequential.solve();
        let sequential_stats = sequential.get_statistics();
        let (_, sequential_result) = sequential.get_result();
        let sequential_y: DMatrix<f64> = sequential_result.expect("sequential result must exist");

        let mut parallel = Radau::new_with_options(stiff_radau_options(true));
        parallel.solve();
        let parallel_stats = parallel.get_statistics();
        let (_, parallel_result) = parallel.get_result();
        let parallel_y: DMatrix<f64> = parallel_result.expect("parallel result must exist");

        println!(
            "[Radau stats] sequential => {}",
            sequential.statistics_report()
        );
        println!(
            "[Radau stats] parallel   => {}",
            parallel.statistics_report()
        );

        assert_eq!(sequential.get_status(), "finished");
        assert_eq!(parallel.get_status(), "finished");
        assert_eq!(sequential_y.shape(), parallel_y.shape());
        assert_eq!(sequential_stats.step_calls, parallel_stats.step_calls);
        assert_eq!(
            sequential_stats.newton_solve_calls,
            parallel_stats.newton_solve_calls
        );
        assert_eq!(
            sequential_stats.residual_calls,
            parallel_stats.residual_calls
        );
        assert_eq!(
            sequential_stats.jacobian_calls,
            parallel_stats.jacobian_calls
        );

        let max_solution_diff = sequential_y
            .iter()
            .zip(parallel_y.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_solution_diff < 1e-8,
            "parallel and sequential Radau should stay numerically aligned, diff={max_solution_diff:e}"
        );
    }

    #[test]
    fn test_radau_solver_facing_comparison_table() {
        let linear_rows = vec![
            solve_with_measurement("seq-linear", linear_decay_radau_options(false)),
            solve_with_measurement("par-linear", linear_decay_radau_options(true)),
        ];
        print_comparison_table("linear-decay-1", &linear_rows);

        let stiff_rows = vec![
            solve_with_measurement("seq-stiff", stiff_radau_options(false)),
            solve_with_measurement("par-stiff", stiff_radau_options(true)),
        ];
        print_comparison_table("stiff-2", &stiff_rows);

        assert_eq!(linear_rows.len(), 2);
        assert_eq!(stiff_rows.len(), 2);
        assert!(linear_rows.iter().all(|row| row.step_calls > 0));
        assert!(stiff_rows.iter().all(|row| row.newton_solves > 0));
    }

    #[test]
    fn test_radau_generated_backend_surface_keeps_selected_c_backend() {
        let solver = Radau::new(RadauOrder::Order5)
            .with_dense_generated_backend_c_tcc("target/generated-ivp-tests")
            .with_dense_generated_backend_mode(DenseIvpGeneratedBackendMode::BuildIfMissingRelease);

        assert_eq!(
            solver.generated_backend_config().aot_codegen_backend,
            AotCodegenBackend::C
        );
        assert_eq!(
            solver.generated_backend_config().aot_c_compiler.as_deref(),
            Some("tcc")
        );
    }

    #[test]
    fn test_radau_new_with_options_installs_generated_backend_mode() {
        let solver = Radau::new_with_options(
            linear_decay_radau_options(false)
                .with_dense_generated_backend_mode(DenseIvpGeneratedBackendMode::RequirePrebuilt),
        );

        assert_eq!(
            solver.generated_backend_config().build_policy,
            crate::symbolic::symbolic_ivp_generated::SymbolicIvpAotBuildPolicy::RequirePrebuilt
        );
        assert_eq!(
            solver.newton.generated_backend_config().build_policy,
            crate::symbolic::symbolic_ivp_generated::SymbolicIvpAotBuildPolicy::RequirePrebuilt
        );
    }

    #[test]
    fn test_radau_generated_backend_ctcc_smoke_solve() {
        if !tcc_is_available() {
            println!("[Radau generated backend] skipping C-tcc smoke test: compiler not available");
            return;
        }

        let mut solver = Radau::new_with_options(
            linear_decay_radau_options(false)
                .with_dense_generated_backend_c_tcc("target/generated-radau-tests"),
        );

        solver
            .try_solve()
            .expect("Radau generated backend C-tcc smoke solve should succeed");

        assert_eq!(solver.get_status(), "finished");
        assert!(solver.get_statistics().backend_prepare_calls >= 1);
    }
}

#[cfg(test)]
mod tests_generated_backend_compare {
    use crate::numerical::Radau::Radau_main::{Radau, RadauOrder, RadauSolverOptions};
    use crate::symbolic::codegen::codegen_runtime_api::{
        recommended_dense_jacobian_chunking_for_parallelism,
        recommended_residual_chunking_for_parallelism,
    };
    use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
    use crate::symbolic::symbolic_engine::Expr;
    use crate::symbolic::symbolic_ivp::SymbolicIvpAotOptions;
    use crate::symbolic::symbolic_ivp_generated::{
        DenseIvpGeneratedBackendMode, SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
    };
    use nalgebra::DVector;
    use std::path::PathBuf;
    use std::process::Command;
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    #[derive(Clone)]
    struct RadauScenario {
        label: &'static str,
        equations: Vec<Expr>,
        values: Vec<String>,
        y0: DVector<f64>,
        tolerance: f64,
        max_iterations: usize,
        h: f64,
        t0: f64,
        t_bound: f64,
    }

    struct RadauBackendCompareRow {
        variant: &'static str,
        total: Duration,
        setup_ms: f64,
        solve_ms: f64,
        residual_ms_avg: f64,
        jacobian_ms_avg: f64,
        residual_calls: usize,
        jacobian_calls: usize,
        step_calls: usize,
        newton_solves: usize,
        newton_iters_avg: f64,
        max_abs_solution: f64,
        solution_diff: f64,
        status: &'static str,
    }

    #[derive(Clone, Copy)]
    enum Toolchain {
        Ctcc,
        Cgcc,
        Zig,
        Rust,
    }

    impl Toolchain {
        fn label(self) -> &'static str {
            match self {
                Self::Ctcc => "AOT-C-tcc",
                Self::Cgcc => "AOT-C-gcc",
                Self::Zig => "AOT-Zig",
                Self::Rust => "AOT-Rust",
            }
        }
    }

    #[derive(Clone, Copy)]
    enum ChunkingMode {
        Whole,
        Parallel2,
    }

    impl ChunkingMode {
        fn label(self) -> &'static str {
            match self {
                Self::Whole => "whole",
                Self::Parallel2 => "parallel(auto,x2)",
            }
        }
    }

    fn stiff_2_scenario() -> RadauScenario {
        RadauScenario {
            label: "stiff-2",
            equations: vec![
                Expr::parse_expression("-1000.0*y1 + y2"),
                Expr::parse_expression("y1 - y2"),
            ],
            values: vec!["y1".to_string(), "y2".to_string()],
            y0: DVector::from_vec(vec![1.0, 1.0]),
            tolerance: 1e-6,
            max_iterations: 100,
            h: 0.01,
            t0: 0.0,
            t_bound: 0.1,
        }
    }

    fn robertson_3_scenario() -> RadauScenario {
        RadauScenario {
            label: "robertson-3",
            equations: vec![
                Expr::parse_expression("-0.04*y1 + 1.0e4*y2*y3"),
                Expr::parse_expression("0.04*y1 - 1.0e4*y2*y3 - 3.0e7*y2^2"),
                Expr::parse_expression("3.0e7*y2^2"),
            ],
            values: vec!["y1".to_string(), "y2".to_string(), "y3".to_string()],
            y0: DVector::from_vec(vec![1.0, 0.0, 0.0]),
            tolerance: 1e-9,
            max_iterations: 100,
            h: 0.0025,
            t0: 0.0,
            t_bound: 20.0,
        }
    }

    fn hires_8_scenario() -> RadauScenario {
        RadauScenario {
            label: "hires-8",
            equations: vec![
                Expr::parse_expression("-1.71*y1 + 0.43*y2 + 8.32*y3 + 0.0007"),
                Expr::parse_expression("1.71*y1 - 8.75*y2"),
                Expr::parse_expression("-10.03*y3 + 0.43*y4 + 0.035*y5"),
                Expr::parse_expression("8.32*y2 + 1.71*y3 - 1.12*y4"),
                Expr::parse_expression("-1.745*y5 + 0.43*y6 + 0.43*y7"),
                Expr::parse_expression("-280.0*y6*y8 + 0.69*y4 + 1.71*y5 - 0.43*y6 + 0.69*y7"),
                Expr::parse_expression("280.0*y6*y8 - 1.81*y7"),
                Expr::parse_expression("-280.0*y6*y8 + 1.81*y7"),
            ],
            values: vec![
                "y1".to_string(),
                "y2".to_string(),
                "y3".to_string(),
                "y4".to_string(),
                "y5".to_string(),
                "y6".to_string(),
                "y7".to_string(),
                "y8".to_string(),
            ],
            y0: DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0057]),
            tolerance: 1e-8,
            max_iterations: 120,
            h: 0.005,
            t0: 0.0,
            t_bound: 20.0,
        }
    }

    fn scenario_options(scenario: &RadauScenario) -> RadauSolverOptions {
        RadauSolverOptions::new(
            RadauOrder::Order5,
            scenario.equations.clone(),
            scenario.values.clone(),
            "t".to_string(),
            scenario.tolerance,
            scenario.max_iterations,
            Some(scenario.h),
            scenario.t0,
            scenario.t_bound,
            scenario.y0.clone(),
        )
    }

    fn command_exists(cmd: &str, probe_arg: &str) -> bool {
        Command::new(cmd).arg(probe_arg).output().is_ok()
    }

    fn tcc_available() -> bool {
        if let Ok(explicit) = std::env::var("RUSTEDSCITHE_TCC") {
            return std::path::Path::new(&explicit).is_file();
        }
        command_exists("tcc", "-v")
    }

    fn gcc_available() -> bool {
        if let Ok(explicit) = std::env::var("RUSTEDSCITHE_GCC") {
            return std::path::Path::new(&explicit).is_file();
        }
        command_exists("gcc", "--version")
    }

    fn zig_available() -> bool {
        command_exists("zig", "version")
    }

    fn toolchain_available(toolchain: Toolchain) -> bool {
        match toolchain {
            Toolchain::Ctcc => tcc_available(),
            Toolchain::Cgcc => gcc_available(),
            Toolchain::Zig => zig_available(),
            Toolchain::Rust => true,
        }
    }

    fn chunking_options(var_count: usize, mode: ChunkingMode) -> SymbolicIvpAotOptions {
        match mode {
            ChunkingMode::Whole => SymbolicIvpAotOptions::default(),
            ChunkingMode::Parallel2 => SymbolicIvpAotOptions {
                residual_strategy: recommended_residual_chunking_for_parallelism(var_count, 2),
                jacobian_strategy: recommended_dense_jacobian_chunking_for_parallelism(
                    var_count, 2,
                ),
            },
        }
    }

    fn unique_output_root(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        PathBuf::from(format!(
            "target/generated-radau-heavy-story/{prefix}/pid{}_{}",
            std::process::id(),
            nanos
        ))
    }

    fn make_generated_backend_config(
        out_dir: PathBuf,
        toolchain: Toolchain,
        chunking: ChunkingMode,
        var_count: usize,
    ) -> SymbolicIvpGeneratedBackendConfig {
        let base = SymbolicIvpGeneratedBackendConfig::from_mode(
            DenseIvpGeneratedBackendMode::BuildIfMissingRelease,
        )
        .with_output_parent_dir(Some(out_dir))
        .with_build_policy(SymbolicIvpAotBuildPolicy::BuildIfMissing {
            profile: AotBuildProfile::Release,
        })
        .with_aot_options(chunking_options(var_count, chunking));

        match toolchain {
            Toolchain::Ctcc => base.with_c_tcc(),
            Toolchain::Cgcc => base.with_c_gcc(),
            Toolchain::Zig => base.with_zig(),
            Toolchain::Rust => base.with_rust(),
        }
    }

    fn scenario_options_with_generated_backend(
        scenario: &RadauScenario,
        config: SymbolicIvpGeneratedBackendConfig,
    ) -> RadauSolverOptions {
        scenario_options(scenario).with_generated_backend_config(config)
    }

    fn make_solver_quiet(mut solver: Radau) -> Radau {
        solver.set_console_logging(false);
        solver.disable_logging();
        solver
    }

    fn max_abs_vector(v: &DVector<f64>) -> f64 {
        v.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()))
    }

    fn max_abs_diff(a: &DVector<f64>, b: &DVector<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .fold(0.0_f64, |acc, (lhs, rhs)| acc.max((lhs - rhs).abs()))
    }

    fn run_solver(
        label: &'static str,
        solver: &mut Radau,
        baseline_solution: Option<&DVector<f64>>,
    ) -> (RadauBackendCompareRow, DVector<f64>) {
        let total_start = Instant::now();
        let outcome = solver.try_solve();
        let total = total_start.elapsed();
        let stats = solver.get_statistics();
        let (_, result) = solver.get_result();
        let final_solution = result
            .and_then(|matrix| {
                if matrix.nrows() == 0 {
                    None
                } else {
                    Some(matrix.row(matrix.nrows() - 1).transpose().into_owned())
                }
            })
            .unwrap_or_else(|| DVector::from_element(solver.y.len(), f64::NAN));
        let max_abs_solution = max_abs_vector(&final_solution);
        let solution_diff = baseline_solution
            .map(|baseline| max_abs_diff(&final_solution, baseline))
            .unwrap_or(0.0);

        (
            RadauBackendCompareRow {
                variant: label,
                total,
                setup_ms: stats.backend_prepare_ms_total,
                solve_ms: stats.solve_ms_total,
                residual_ms_avg: stats.avg_residual_ms().unwrap_or(0.0),
                jacobian_ms_avg: stats.avg_jacobian_ms().unwrap_or(0.0),
                residual_calls: stats.residual_calls,
                jacobian_calls: stats.jacobian_calls,
                step_calls: stats.step_calls,
                newton_solves: stats.newton_solve_calls,
                newton_iters_avg: stats.avg_newton_iterations().unwrap_or(0.0),
                max_abs_solution,
                solution_diff,
                status: if outcome.is_ok() && solver.get_status() == "finished" {
                    "finished"
                } else {
                    "failed"
                },
            },
            final_solution,
        )
    }

    fn print_compare_table(scenario: &RadauScenario, rows: &[RadauBackendCompareRow]) {
        println!(
            "[Radau backend compare] scenario={}, vars={}, variants={}",
            scenario.label,
            scenario.values.len(),
            rows.len()
        );
        println!(
            "variant      |   total_ms |   setup_ms |   solve_ms | residual_ms(avg) | jacobian_ms(avg) | max_abs_solution | status"
        );
        println!(
            "----------------------------------------------------------------------------------------------------------------------"
        );
        for row in rows {
            println!(
                "{:<12} | {:>10.3} | {:>10.3} | {:>10.3} | {:>16.6} | {:>16.6} | {:>16.6e} | {}",
                row.variant,
                row.total.as_secs_f64() * 1_000.0,
                row.setup_ms,
                row.solve_ms,
                row.residual_ms_avg,
                row.jacobian_ms_avg,
                row.max_abs_solution,
                row.status,
            );
        }
        println!(
            "{:<12} | {:>14} | {:>16} | {:>10} | {:>14} | {:>16}",
            "variant",
            "solution_diff",
            "speedup_vs_lambdify",
            "steps",
            "newton_solves",
            "newton_iters(avg)"
        );
        println!(
            "----------------------------------------------------------------------------------------------------------------------"
        );
        let lambdify_total_ms = rows
            .iter()
            .find(|row| row.variant == "Lambdify")
            .map(|row| row.total.as_secs_f64() * 1_000.0)
            .unwrap_or(0.0);
        for row in rows {
            let speedup = if row.variant == "Lambdify" {
                1.0
            } else {
                lambdify_total_ms / (row.total.as_secs_f64() * 1_000.0)
            };
            println!(
                "{:<12} | {:>14.6e} | {:>15.3}x | {:>10} | {:>14} | {:>16.3}",
                row.variant,
                row.solution_diff,
                speedup,
                row.step_calls,
                row.newton_solves,
                row.newton_iters_avg,
            );
        }
        println!(
            "{:<12} | {:>10} | {:>10} | {:>16} | {:>16}",
            "variant", "res_calls", "jac_calls", "residual_ms(total)", "jacobian_ms(total)"
        );
        println!(
            "----------------------------------------------------------------------------------------------------------------------"
        );
        for row in rows {
            println!(
                "{:<12} | {:>10} | {:>10} | {:>16.3} | {:>16.3}",
                row.variant,
                row.residual_calls,
                row.jacobian_calls,
                row.residual_ms_avg * row.residual_calls as f64,
                row.jacobian_ms_avg * row.jacobian_calls as f64,
            );
        }
        if let Some(lambdify) = rows.iter().find(|row| row.variant == "Lambdify") {
            let residual_total = lambdify.residual_ms_avg * lambdify.residual_calls as f64;
            let jacobian_total = lambdify.jacobian_ms_avg * lambdify.jacobian_calls as f64;
            let dominant = if residual_total > 2.0 * jacobian_total {
                "residual-dominated"
            } else if jacobian_total > 2.0 * residual_total {
                "jacobian-dominated"
            } else {
                "mixed"
            };
            let best_total = rows
                .iter()
                .min_by(|lhs, rhs| lhs.total.cmp(&rhs.total))
                .expect("rows should not be empty");
            let best_solve = rows
                .iter()
                .min_by(|lhs, rhs| lhs.solve_ms.total_cmp(&rhs.solve_ms))
                .expect("rows should not be empty");
            println!(
                "[Radau backend compare] summary: dominant_hot_path={}, best_total={}, best_solve={}, baseline_residual_ms_total={:.3}, baseline_jacobian_ms_total={:.3}",
                dominant, best_total.variant, best_solve.variant, residual_total, jacobian_total
            );
        }
        println!(
            "[Radau backend compare] finished scenario `{}`",
            scenario.label
        );
    }

    fn run_compare_for_scenario(scenario: &RadauScenario) {
        let mut rows = Vec::new();

        let mut lambdify = make_solver_quiet(Radau::new_with_options(scenario_options(scenario)));
        let (lambdify_row, baseline_solution) = run_solver("Lambdify", &mut lambdify, None);
        rows.push(lambdify_row);

        if gcc_available() {
            let mut gcc_solver = make_solver_quiet(
                Radau::new_with_options(scenario_options(scenario))
                    .with_dense_generated_backend_c_gcc("target/generated-radau-tests"),
            );
            let (row, _) = run_solver("C-gcc", &mut gcc_solver, Some(&baseline_solution));
            rows.push(row);
        }

        if tcc_available() {
            let mut tcc_solver = make_solver_quiet(
                Radau::new_with_options(scenario_options(scenario))
                    .with_dense_generated_backend_c_tcc("target/generated-radau-tests"),
            );
            let (row, _) = run_solver("C-tcc", &mut tcc_solver, Some(&baseline_solution));
            rows.push(row);
        }

        if zig_available() {
            let mut zig_solver = make_solver_quiet(
                Radau::new_with_options(scenario_options(scenario))
                    .with_dense_generated_backend_zig("target/generated-radau-tests"),
            );
            let (row, _) = run_solver("Zig", &mut zig_solver, Some(&baseline_solution));
            rows.push(row);
        }

        print_compare_table(scenario, &rows);
        assert!(!rows.is_empty());
        assert!(rows.iter().all(|row| row.status == "finished"));
    }

    fn print_production_like_table(scenario: &RadauScenario, rows: &[RadauBackendCompareRow]) {
        println!(
            "[Radau production-like] scenario={}, vars={}, variants={}",
            scenario.label,
            scenario.values.len(),
            rows.len()
        );
        println!(
            "variant      |     total_ms | max_abs_solution | solution_diff_vs_lambdify | status"
        );
        println!(
            "--------------------------------------------------------------------------------------"
        );
        for row in rows {
            println!(
                "{:<12} | {:>12.3} | {:>16.6e} | {:>24.6e} | {}",
                row.variant,
                row.total.as_secs_f64() * 1_000.0,
                row.max_abs_solution,
                row.solution_diff,
                row.status,
            );
        }
        if let Some(best_total) = rows.iter().min_by(|lhs, rhs| lhs.total.cmp(&rhs.total)) {
            println!(
                "[Radau production-like] best_total={} scenario={}",
                best_total.variant, scenario.label
            );
        }
        println!(
            "[Radau production-like] finished scenario `{}`",
            scenario.label
        );
    }

    fn run_production_like_for_scenario(scenario: &RadauScenario) {
        let mut rows = Vec::new();

        let mut lambdify = make_solver_quiet(Radau::new_with_options(scenario_options(scenario)));
        let (lambdify_row, baseline_solution) = run_solver("Lambdify", &mut lambdify, None);
        rows.push(lambdify_row);

        if gcc_available() {
            let mut gcc_solver = make_solver_quiet(
                Radau::new_with_options(scenario_options(scenario))
                    .with_dense_generated_backend_c_gcc("target/generated-radau-tests"),
            );
            let (row, _) = run_solver("C-gcc", &mut gcc_solver, Some(&baseline_solution));
            rows.push(row);
        }

        if tcc_available() {
            let mut tcc_solver = make_solver_quiet(
                Radau::new_with_options(scenario_options(scenario))
                    .with_dense_generated_backend_c_tcc("target/generated-radau-tests"),
            );
            let (row, _) = run_solver("C-tcc", &mut tcc_solver, Some(&baseline_solution));
            rows.push(row);
        }

        if zig_available() {
            let mut zig_solver = make_solver_quiet(
                Radau::new_with_options(scenario_options(scenario))
                    .with_dense_generated_backend_zig("target/generated-radau-tests"),
            );
            let (row, _) = run_solver("Zig", &mut zig_solver, Some(&baseline_solution));
            rows.push(row);
        }

        print_production_like_table(scenario, &rows);
        assert!(!rows.is_empty());
        assert!(rows.iter().all(|row| row.status == "finished"));
    }

    #[test]
    #[ignore]
    fn radau_dense_aot_heavy_toolchain_chunking_matrix_story() {
        let scenarios = vec![robertson_3_scenario(), hires_8_scenario()];
        println!(
            "[Radau AOT heavy] dense toolchain+chunking matrix; all time columns are milliseconds"
        );
        println!(
            "scenario    | route        | chunking         | total_ms | setup_ms | solve_ms | final_diff_vs_lambdify | residual_calls | jacobian_calls | steps | status"
        );
        println!(
            "----------------------------------------------------------------------------------------------------------------------------------------------------------------"
        );

        let mut any_finished = false;

        for scenario in &scenarios {
            let mut baseline_solver =
                make_solver_quiet(Radau::new_with_options(scenario_options(scenario)));
            let (baseline_row, baseline_solution) =
                run_solver("Lambdify", &mut baseline_solver, None);
            any_finished |= baseline_row.status == "finished";
            println!(
                "{:<11} | {:<12} | {:<16} | {:>8.3} | {:>8.3} | {:>8.3} | {:>22.3e} | {:>14} | {:>14} | {:>5} | {}",
                scenario.label,
                baseline_row.variant,
                "n/a",
                baseline_row.total.as_secs_f64() * 1_000.0,
                baseline_row.setup_ms,
                baseline_row.solve_ms,
                baseline_row.solution_diff,
                baseline_row.residual_calls,
                baseline_row.jacobian_calls,
                baseline_row.step_calls,
                baseline_row.status
            );

            for toolchain in [
                Toolchain::Ctcc,
                Toolchain::Cgcc,
                Toolchain::Zig,
                Toolchain::Rust,
            ] {
                if !toolchain_available(toolchain) {
                    println!(
                        "[Radau AOT heavy] skipping {} on scenario {}: compiler/runtime unavailable",
                        toolchain.label(),
                        scenario.label
                    );
                    continue;
                }

                for chunking in [ChunkingMode::Whole, ChunkingMode::Parallel2] {
                    let out_dir = unique_output_root(&format!(
                        "{}_{}_{}",
                        scenario.label,
                        toolchain.label(),
                        chunking.label()
                    ));
                    let config = make_generated_backend_config(
                        out_dir,
                        toolchain,
                        chunking,
                        scenario.values.len().max(1),
                    );
                    let mut solver = make_solver_quiet(Radau::new_with_options(
                        scenario_options_with_generated_backend(scenario, config),
                    ));
                    let (row, _) =
                        run_solver(toolchain.label(), &mut solver, Some(&baseline_solution));
                    any_finished |= row.status == "finished";
                    println!(
                        "{:<11} | {:<12} | {:<16} | {:>8.3} | {:>8.3} | {:>8.3} | {:>22.3e} | {:>14} | {:>14} | {:>5} | {}",
                        scenario.label,
                        row.variant,
                        chunking.label(),
                        row.total.as_secs_f64() * 1_000.0,
                        row.setup_ms,
                        row.solve_ms,
                        row.solution_diff,
                        row.residual_calls,
                        row.jacobian_calls,
                        row.step_calls,
                        row.status
                    );

                    assert_eq!(
                        row.status,
                        "finished",
                        "Radau dense AOT route failed: scenario={} route={} chunking={}",
                        scenario.label,
                        row.variant,
                        chunking.label()
                    );
                    assert!(
                        row.solution_diff <= 1e-7 * (1.0 + baseline_solution.amax()),
                        "Radau dense AOT parity drift too large: scenario={} route={} chunking={} diff={:e}",
                        scenario.label,
                        row.variant,
                        chunking.label(),
                        row.solution_diff
                    );
                }
            }
        }

        assert!(
            any_finished,
            "at least one heavy dense Radau route should finish"
        );
    }

    #[test]
    #[ignore]
    fn radau_generated_backend_end_to_end_compare_table() {
        let scenarios = vec![
            stiff_2_scenario(),
            robertson_3_scenario(),
            hires_8_scenario(),
        ];
        for scenario in scenarios {
            run_compare_for_scenario(&scenario);
        }
    }

    #[test]
    #[ignore]
    fn radau_production_like_end_to_end_compare_table() {
        let scenarios = vec![
            stiff_2_scenario(),
            robertson_3_scenario(),
            hires_8_scenario(),
        ];
        for scenario in scenarios {
            run_production_like_for_scenario(&scenario);
        }
    }
}
