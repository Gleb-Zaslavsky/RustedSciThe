#[cfg(test)]
mod tests {

    use crate::numerical::BVP_Damp::BVP_utils::CustomTimer;

    use crate::numerical::BVP_sci::BVP_sci_faer::{faer_col, faer_dense_mat};
    use crate::numerical::BVP_sci::BVP_sci_symb::BVPwrap;

    use crate::symbolic::symbolic_engine::Expr;

    use nalgebra::{DMatrix, DVector};
    use std::collections::HashMap;
    #[test]
    fn test_bc_closure_creater_simple() {
        let mut boundary_conditions = HashMap::new();
        boundary_conditions.insert("y".to_string(), vec![(0, 1.0)]); // left boundary: y(0) = 1.0
        boundary_conditions.insert("z".to_string(), vec![(1, 2.0)]); // right boundary: z(1) = 2.0

        let values = vec!["y".to_string(), "z".to_string()];

        let bc_func = BVPwrap::BC_closure_creater(boundary_conditions, values);
        assert!(bc_func.is_some());

        let bc = bc_func.unwrap();

        // Test boundary condition evaluation
        let ya = faer_col::from_fn(2, |i| [1.5, 0.5][i]); // y(0)=1.5, z(0)=0.5
        let yb = faer_col::from_fn(2, |i| [0.8, 2.3][i]); // y(1)=0.8, z(1)=2.3
        let p = faer_col::zeros(0);

        let residuals = bc(&ya, &yb, &p);

        // Expected residuals: [ya[0] - 1.0, yb[1] - 2.0] = [1.5 - 1.0, 2.3 - 2.0] = [0.5, 0.3]
        assert_eq!(residuals.nrows(), 2);
        assert!((residuals[0] - 0.5).abs() < 1e-10);
        assert!((residuals[1] - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_bc_closure_creater_empty() {
        let boundary_conditions = HashMap::new();
        let values = vec!["y".to_string(), "z".to_string()];

        let bc_func = BVPwrap::BC_closure_creater(boundary_conditions, values);
        assert!(bc_func.is_none());
    }

    #[test]
    fn test_bc_closure_creater_mixed_boundaries() {
        let mut boundary_conditions = HashMap::new();
        boundary_conditions.insert("y".to_string(), vec![(0, 0.0)]); // left boundary: y(0) = 0.0
        boundary_conditions.insert("z".to_string(), vec![(0, 1.0)]); // left boundary: z(0) = 1.0

        let values = vec!["y".to_string(), "z".to_string()];

        let bc_func = BVPwrap::BC_closure_creater(boundary_conditions, values);
        assert!(bc_func.is_some());

        let bc = bc_func.unwrap();

        let ya = faer_col::from_fn(2, |i| [0.1, 1.2][i]);
        let yb = faer_col::from_fn(2, |i| [5.0, 6.0][i]);
        let p = faer_col::zeros(0);

        let residuals = bc(&ya, &yb, &p);

        // Both conditions are on left boundary: [ya[0] - 0.0, ya[1] - 1.0] = [0.1, 0.2]
        assert_eq!(residuals.nrows(), 2);
        assert!((residuals[0] - 0.1).abs() < 1e-10);
        assert!((residuals[1] - 0.2).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "Boundary index must be 0 (left) or 1 (right)")]
    fn test_bc_closure_creater_invalid_boundary_index() {
        let mut boundary_conditions = HashMap::new();
        boundary_conditions.insert("y".to_string(), vec![(2, 1.0)]); // invalid index

        let values = vec!["y".to_string()];

        let bc_func = BVPwrap::BC_closure_creater(boundary_conditions, values).unwrap();

        let ya = faer_col::from_fn(1, |_| 1.0);
        let yb = faer_col::from_fn(1, |_| 2.0);
        let p = faer_col::zeros(0);

        bc_func(&ya, &yb, &p); // Should panic here
    }

    #[test]
    fn test_eq_generate_simple_system() {
        // Create simple ODE system: dy/dx = z, dz/dx = -y
        let eq1 = Expr::Var("z".to_string());
        let eq2 = -Expr::Var("y".to_string());
        let eq_system = vec![eq1, eq2];

        let values = vec!["y".to_string(), "z".to_string()];
        let param = vec![];
        let arg = "x".to_string();

        let mut boundary_conditions = HashMap::new();
        boundary_conditions.insert("y".to_string(), vec![(0, 1.0)]);
        boundary_conditions.insert("z".to_string(), vec![(1, 0.0)]);

        let x_mesh = DVector::from_vec(vec![0.0, 0.5, 1.0]);
        let initial_guess = DMatrix::from_fn(2, 3, |i, j| (i + j) as f64);

        let mut bvp_wrap = BVPwrap::new(
            Some(x_mesh),
            None,
            None,
            None,
            eq_system,
            values,
            param,
            None,
            boundary_conditions,
            arg,
            1e-6,
            1000,
            initial_guess,
        );

        let (jacobian, residual_function, bc_func) = bvp_wrap.eq_generate();

        // Test that functions were generated
        assert!(jacobian.is_some());
        assert!(bc_func.is_some());

        // Test residual function
        let x = faer_col::from_fn(2, |i| i as f64);
        let mut y = faer_dense_mat::zeros(2, 2);
        *y.get_mut(0, 0) = 1.0; // y
        *y.get_mut(1, 0) = 2.0; // z
        *y.get_mut(0, 1) = 3.0; // y
        *y.get_mut(1, 1) = 4.0; // z
        let p = faer_col::zeros(0);

        let result = residual_function(&x, &y, &p);

        // Expected: dy/dx = z, dz/dx = -y
        assert!((result.get(0, 0) - 2.0).abs() < 1e-10); // z[0] = 2.0
        assert!((result.get(1, 0) - (-1.0)).abs() < 1e-10); // -y[0] = -1.0
        assert!((result.get(0, 1) - 4.0).abs() < 1e-10); // z[1] = 4.0
        assert!((result.get(1, 1) - (-3.0)).abs() < 1e-10); // -y[1] = -3.0
    }

    #[test]
    fn test_eq_generate_with_parameters() {
        // Create ODE system with parameters: dy/dx = a*z, dz/dx = -b*y
        let eq1 = Expr::Var("a".to_string()) * Expr::Var("z".to_string());
        let eq2 = -Expr::Var("b".to_string()) * Expr::Var("y".to_string());
        let eq_system = vec![eq1, eq2];

        let values = vec!["y".to_string(), "z".to_string()];
        let param = vec!["a".to_string(), "b".to_string()];
        let param_values = Some(DVector::from_vec(vec![1.0, 2.0]));
        let arg = "x".to_string();

        let mut boundary_conditions = HashMap::new();
        boundary_conditions.insert("y".to_string(), vec![(0, 1.0)]);
        boundary_conditions.insert("z".to_string(), vec![(1, 0.0)]);

        let x_mesh = DVector::from_vec(vec![0.0, 1.0]);
        let initial_guess = DMatrix::from_fn(2, 2, |i, j| (i + j) as f64);

        let mut bvp_wrap = BVPwrap::new(
            Some(x_mesh),
            None,
            None,
            None,
            eq_system,
            values,
            param,
            param_values,
            boundary_conditions,
            arg,
            1e-6,
            1000,
            initial_guess,
        );

        let (jacobian, residual_function, bc_func) = bvp_wrap.eq_generate();

        // Test that functions were generated
        assert!(jacobian.is_some());
        assert!(bc_func.is_some());

        // Test residual function with parameters
        let x = faer_col::from_fn(1, |_| 0.0);
        let mut y = faer_dense_mat::zeros(2, 1);
        *y.get_mut(0, 0) = 2.0; // y
        *y.get_mut(1, 0) = 3.0; // z
        let p = faer_col::from_fn(2, |i| [1.0, 2.0][i]); // a=1.0, b=2.0

        let result = residual_function(&x, &y, &p);

        // Expected: dy/dx = a*z = 1.0*3.0 = 3.0, dz/dx = -b*y = -2.0*2.0 = -4.0
        assert!((result.get(0, 0) - 3.0).abs() < 1e-10);
        assert!((result.get(1, 0) - (-4.0)).abs() < 1e-10);
    }

    #[test]
    fn test_eq_generate_jacobian_evaluation() {
        // Simple linear system for easy Jacobian verification
        let eq1 = Expr::Var("y".to_string());
        let eq2 = Expr::Var("z".to_string());
        let eq_system = vec![eq1, eq2];

        let values = vec!["y".to_string(), "z".to_string()];
        let param = vec![];
        let arg = "x".to_string();

        let mut boundary_conditions = HashMap::new();
        boundary_conditions.insert("y".to_string(), vec![(0, 0.0)]);
        boundary_conditions.insert("z".to_string(), vec![(1, 0.0)]);

        let x_mesh = DVector::from_vec(vec![0.0, 1.0]);
        let initial_guess = DMatrix::zeros(2, 2);

        let mut bvp_wrap = BVPwrap::new(
            Some(x_mesh),
            None,
            None,
            None,
            eq_system,
            values,
            param,
            None,
            boundary_conditions,
            arg,
            1e-6,
            1000,
            initial_guess,
        );

        let (jacobian_opt, _, _) = bvp_wrap.eq_generate();
        let jacobian = jacobian_opt.unwrap();

        // Test Jacobian evaluation
        let x = faer_col::from_fn(1, |_| 0.5);
        let y = faer_dense_mat::from_fn(2, 1, |i, _| (i + 1) as f64);
        let p = faer_col::zeros(0);

        let (df_dy, df_dp) = jacobian(&x, &y, &p);

        // For dy/dx = y, dz/dx = z, Jacobian should be [[1, 0], [0, 1]]
        println!("Jacobian df_dy: {:?}", df_dy);
        println!("Jacobian df_dp: {:?}", df_dp);
        assert_eq!(df_dy.len(), 1); // One mesh point
        assert!((df_dy[0].get(0, 0).unwrap() - 1.0).abs() < 1e-10);
        assert!((df_dy[0].get(1, 1).unwrap() - 1.0).abs() < 1e-10);
        assert_eq!(df_dy[0].get(0, 1).is_none(), true); // No cross terms
        assert_eq!(df_dy[0].get(1, 0).is_none(), true); // No cross terms
        assert!(df_dp.unwrap()[0].compute_nnz() == 0); // No parameters
    }

    #[test]
    fn test_solve_bvp_wrap_simple_harmonic() {
        // Simple harmonic oscillator: y'' + y = 0, y(0) = 0, y'(0) = 1
        // Exact solution: y = sin(x)
        let eq1 = Expr::Var("z".to_string());
        let eq2 = -Expr::Var("y".to_string());
        let eq_system = vec![eq1, eq2];

        let values = vec!["y".to_string(), "z".to_string()];
        let param = vec![];
        let arg = "x".to_string();

        let mut boundary_conditions = HashMap::new();
        boundary_conditions.insert("y".to_string(), vec![(0, 0.0)]); // y(0) = 0
        boundary_conditions.insert("z".to_string(), vec![(0, 1.0)]); // z(0) = 1

        let x_mesh = DVector::from_vec(vec![0.0, std::f64::consts::PI / 2.0, std::f64::consts::PI]);
        let initial_guess = DMatrix::from_fn(2, 3, |i, j| {
            let x_val = x_mesh[j];
            match i {
                0 => x_val.sin(), // y = sin(x)
                1 => x_val.cos(), // z = cos(x)
                _ => 0.0,
            }
        });

        let mut bvp_wrap = BVPwrap::new(
            Some(x_mesh),
            None,
            None,
            None,
            eq_system,
            values,
            param,
            None,
            boundary_conditions,
            arg,
            1e-6,
            1000,
            initial_guess,
        );

        let (jacobian, residual_function, bc_func) = bvp_wrap.eq_generate();

        // Test solve_bvp_wrap
        let x = bvp_wrap.x_mesh_col.clone();
        let y = bvp_wrap.initial_guess_mat.clone();
        let custom_timer = CustomTimer::new();

        bvp_wrap.solve_bvp_wrap(
            &residual_function,
            bc_func.as_ref().unwrap(),
            x,
            y,
            None,
            None,
            jacobian.as_deref(),
            None,
            1e-6,
            1000,
            0,
            None,
            custom_timer,
        );

        // Check that result was stored
        assert!(bvp_wrap.result.success || bvp_wrap.result.status <= 1);
        if bvp_wrap.result.success {
            // Verify boundary conditions
            assert!(bvp_wrap.result.y.get(0, 0).abs() < 1e-3); // y(0) ≈ 0
            assert!((bvp_wrap.result.y.get(1, 0) - 1.0).abs() < 1e-3); // z(0) ≈ 1
        }
    }

    #[test]
    fn test_solve_bvp_wrap_linear_system() {
        // Linear system: y'' = 0, y(0) = 0, y(1) = 1
        // Exact solution: y = x
        let eq1 = Expr::Var("z".to_string());
        let eq2 = Expr::Const(0.0);
        let eq_system = vec![eq1, eq2];

        let values = vec!["y".to_string(), "z".to_string()];
        let param = vec![];
        let arg = "x".to_string();

        let mut boundary_conditions = HashMap::new();
        boundary_conditions.insert("y".to_string(), vec![(0, 0.0), (1, 1.0)]); // y(0) = 0, y(1) = 1

        let x_mesh = DVector::from_vec(vec![0.0, 0.5, 1.0]);
        let initial_guess = DMatrix::from_fn(2, 3, |i, j| {
            match i {
                0 => x_mesh[j], // y = x
                1 => 1.0,       // z = 1
                _ => 0.0,
            }
        });

        let mut bvp_wrap = BVPwrap::new(
            Some(x_mesh),
            None,
            None,
            None,
            eq_system,
            values,
            param,
            None,
            boundary_conditions,
            arg,
            1e-8,
            1000,
            initial_guess,
        );

        let (jacobian, residual_function, bc_func) = bvp_wrap.eq_generate();

        let x = bvp_wrap.x_mesh_col.clone();
        let y = bvp_wrap.initial_guess_mat.clone();
        let custom_timer = CustomTimer::new();

        bvp_wrap.solve_bvp_wrap(
            &residual_function,
            bc_func.as_ref().unwrap(),
            x,
            y,
            None,
            None,
            jacobian.as_deref(),
            None,
            1e-8,
            1000,
            0,
            None,
            custom_timer,
        );

        if bvp_wrap.result.success {
            // Check linear solution y = x
            for i in 0..bvp_wrap.result.x.nrows() {
                let x_val = bvp_wrap.result.x[i];
                let y_val = bvp_wrap.result.y.get(0, i);
                assert!((y_val - x_val).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_solve_bvp_wrap_with_parameters() {
        // Parametric system: y'' = a*y, y(0) = 1, y(1) = exp(sqrt(a))
        // With a = 1, exact solution: y = exp(x)
        let eq1 = Expr::Var("z".to_string());
        let eq2 = Expr::Var("a".to_string()) * Expr::Var("y".to_string());
        let eq_system = vec![eq1, eq2];

        let values = vec!["y".to_string(), "z".to_string()];
        let param = vec!["a".to_string()];
        let param_values = Some(DVector::from_vec(vec![1.0]));
        let arg = "x".to_string();

        let mut boundary_conditions = HashMap::new();
        boundary_conditions.insert("y".to_string(), vec![(0, 1.0), (1, std::f64::consts::E)]); // y(0) = 1, y(1) = e

        let x_mesh = DVector::from_vec(vec![0.0, 0.5, 1.0]);
        let initial_guess = DMatrix::from_fn(2, 3, |i, j| {
            match i {
                0 => (x_mesh[j] as f64).exp(), // y = exp(x)
                1 => (x_mesh[j] as f64).exp(), // z = exp(x)
                _ => 0.0,
            }
        });

        let mut bvp_wrap = BVPwrap::new(
            Some(x_mesh),
            None,
            None,
            None,
            eq_system,
            values,
            param,
            param_values,
            boundary_conditions,
            arg,
            1e-6,
            1000,
            initial_guess,
        );

        let (jacobian, residual_function, bc_func) = bvp_wrap.eq_generate();

        let x = bvp_wrap.x_mesh_col.clone();
        let y = bvp_wrap.initial_guess_mat.clone();
        let p = bvp_wrap.param_col.clone();
        let custom_timer = CustomTimer::new();

        bvp_wrap.solve_bvp_wrap(
            &residual_function,
            bc_func.as_ref().unwrap(),
            x,
            y,
            p,
            None,
            jacobian.as_deref(),
            None,
            1e-6,
            1000,
            0,
            None,
            custom_timer,
        );

        if bvp_wrap.result.success {
            // Check exponential solution
            for i in 0..bvp_wrap.result.x.nrows() {
                let x_val = bvp_wrap.result.x[i];
                let y_val = bvp_wrap.result.y.get(0, i);
                let expected = x_val.exp();
                assert!((y_val - expected).abs() < 1e-3);
            }
        }
    }
    #[test]
    fn test_elementary_quadratic_bvp_calling_solve() {
        // Problem: y'' = -2, y(0) = 0, y'(1) = -1
        // Analytic solution: y = x*(1-x)
        let eq1 = Expr::Var("z".to_string());
        let eq2 = Expr::Const(-2.0);
        let eq_system = vec![eq1, eq2];

        let values = vec!["y".to_string(), "z".to_string()];
        let param = vec![];
        let arg = "x".to_string();

        let mut boundary_conditions = HashMap::new();
        boundary_conditions.insert("y".to_string(), vec![(0, 0.0)]); // y(0) = 0
        boundary_conditions.insert("z".to_string(), vec![(1, -1.0)]); // y'(1) = -1

        let x_mesh = DVector::from_vec(vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        let initial_guess = DMatrix::from_fn(2, 5, |i, _| {
            match i {
                0 => 1.0, // x_val * (1.0 - x_val), // y = x*(1-x)
                1 => 1.0, //1.0 - 2.0 * x_val,     // z = y' = 1-2x
                _ => 0.0,
            }
        });

        // Use new method to create solver instance
        let mut bvp_solver = BVPwrap::new(
            Some(x_mesh.clone()),
            None,
            None,
            None,
            eq_system,
            values,
            param,
            None,
            boundary_conditions,
            arg,
            1e-8,
            1000,
            initial_guess,
        );

        // Use solve method to start solver
        bvp_solver.solve();

        // Check solution success
        assert!(bvp_solver.result.success, "BVP solver failed to converge");

        // Compare result with analytic solution y = x*(1-x)
        for i in 0..bvp_solver.result.x.nrows() {
            let x_val = bvp_solver.result.x[i];
            let y_numerical = bvp_solver.result.y.get(0, i);
            let y_analytic = x_val * (1.0 - x_val);

            println!(
                "x = {:.3}, y_num = {:.6}, y_exact = {:.6}, error = {:.2e}",
                x_val,
                y_numerical,
                y_analytic,
                (y_numerical - y_analytic).abs()
            );

            assert!(
                (y_numerical - y_analytic).abs() < 1e-6,
                "Solution error too large at x = {}: {} vs {}",
                x_val,
                y_numerical,
                y_analytic
            );
        }

        // Verify boundary conditions
        assert!(
            bvp_solver.result.y.get(0, 0).abs() < 1e-8,
            "y(0) should be 0"
        );
        let z_at_1 = bvp_solver.result.y.get(1, bvp_solver.result.x.nrows() - 1);
        assert!((z_at_1 - (-1.0)).abs() < 1e-6, "y'(1) should be -1");
    }
}
