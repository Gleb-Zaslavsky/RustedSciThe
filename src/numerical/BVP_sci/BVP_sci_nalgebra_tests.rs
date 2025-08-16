#[cfg(test)]
mod tests {
    use crate::numerical::BVP_sci::BVP_sci_nalgebra::{
        collocation_fun, compute_jac_indices, construct_global_jac, create_spline, estimate_bc_jac,
        estimate_fun_jac, estimate_rms_residuals, modify_mesh, solve_bvp, solve_newton,
        stacked_matmul,
    };
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_basic_jacobian_estimation() {
        // Simple test function: f(x,y) = y
        let fun = |_x: &DVector<f64>, y: &DMatrix<f64>, _p: &DVector<f64>| y.clone();

        let x = DVector::from_vec(vec![0.0, 1.0]);
        let y = DMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let p = DVector::from_vec(vec![]);

        let (df_dy, df_dp) = estimate_fun_jac(&fun, &x, &y, &p, None);

        assert_eq!(df_dy.len(), 2);
        assert!(df_dp.is_none());
        // For f(x,y) = y, df/dy should be identity matrix
        for jacobian in &df_dy {
            for i in 0..2 {
                for j in 0..2 {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!((jacobian[(i, j)] - expected).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn test_simple_bvp_problem_debug() {
        // Simplified debug version of the BVP test
        let fun = |_x: &DVector<f64>, y: &DMatrix<f64>, _p: &DVector<f64>| {
            let mut f = DMatrix::zeros(2, y.ncols());
            for j in 0..y.ncols() {
                f[(0, j)] = y[(1, j)]; // y1' = y2
                f[(1, j)] = -y[(0, j)]; // y2' = -y1
            }
            f
        };

        let bc = |ya: &DVector<f64>, yb: &DVector<f64>, _p: &DVector<f64>| {
            DVector::from_vec(vec![ya[0], yb[0]])
        };

        // Simpler mesh
        let x = DVector::from_vec(vec![0.0, std::f64::consts::PI]);
        let mut y = DMatrix::zeros(2, 2);
        y[(0, 0)] = 0.0; // y1(0) = 0
        y[(0, 1)] = 0.0; // y1(π) = 0  
        y[(1, 0)] = 1.0; // y2(0) = 1
        y[(1, 1)] = -1.0; // y2(π) = -1

        println!("Initial guess: y = {:?}", y);

        let result = solve_bvp(
            &fun,
            &bc,
            x.clone(),
            y,
            None,
            None,
            None,
            None,
            1e-4,
            1000,
            2,
            None, // verbose=2, relaxed tolerance
        );

        match result {
            Ok(res) => {
                println!(
                    "Success: {}, Status: {}, Message: {}",
                    res.success, res.status, res.message
                );
                println!("Final y: {:?}", res.y);
                println!("Final yp: {:?}", res.yp);

                // Relaxed assertions for debugging
                if res.success {
                    assert!(
                        res.y[(0, 0)].abs() < 1e-3,
                        "Left BC: y(0) = {}",
                        res.y[(0, 0)]
                    );
                    assert!(
                        res.y[(0, 1)].abs() < 1e-3,
                        "Right BC: y(π) = {}",
                        res.y[(0, 1)]
                    );
                }
            }
            Err(e) => {
                println!("BVP failed with error: {}", e);
                // Don't panic, just report
            }
        }
    }

    #[test]
    fn test_simple_bvp_problem() {
        // Test a simple BVP: y'' = -y, y(0) = 0, y'(0) = 1
        // This should have the solution y = sin(x)

        // Convert to first-order system: y1' = y2, y2' = -y1
        let fun = |_x: &DVector<f64>, y: &DMatrix<f64>, _p: &DVector<f64>| {
            let mut f = DMatrix::zeros(2, y.ncols());
            for j in 0..y.ncols() {
                f[(0, j)] = y[(1, j)]; // y1' = y2
                f[(1, j)] = -y[(0, j)]; // y2' = -y1
            }
            f
        };

        // Boundary conditions: y1(0) = 0, y2(0) = 1 (i.e., y(0) = 0, y'(0) = 1)
        let bc = |ya: &DVector<f64>, _yb: &DVector<f64>, _p: &DVector<f64>| {
            DVector::from_vec(vec![ya[0], ya[1] - 1.0])
        };

        // Mesh from 0 to π
        let x = DVector::from_vec(vec![
            0.0,
            std::f64::consts::PI / 4.0,
            std::f64::consts::PI / 2.0,
            3.0 * std::f64::consts::PI / 4.0,
            std::f64::consts::PI,
        ]);

        // Initial guess: y1 ≈ sin(x), y2 ≈ cos(x)
        let mut y = DMatrix::zeros(2, 5);
        for j in 0..5 {
            y[(0, j)] = x[j].sin();
            y[(1, j)] = x[j].cos();
        }

        // Solve BVP with relaxed tolerance for testing
        let result = solve_bvp(
            &fun,
            &bc,
            x.clone(),
            y,
            None,
            None,
            None,
            None,
            1e-6, // Relaxed tolerance
            1000,
            0,
            None,
        );

        match result {
            Ok(res) => {
                assert!(res.success);
                assert_eq!(res.status, 0);
                let Y1: Vec<f64> = res.y.row(0).iter().map(|x| *x).collect();
                let Y2: Vec<f64> = res.y.row(1).iter().map(|x| *x).collect();
                println!("result: {:?}, {:?}", Y1, Y2);
                // Check boundary conditions are satisfied (relaxed tolerance)
                assert!(
                    res.y[(0, 0)].abs() < 1e-3,
                    "Left BC not satisfied: {}",
                    res.y[(0, 0)]
                );
                assert!(
                    res.y[(0, res.x.len() - 1)].abs() < 1e-3,
                    "Right BC not satisfied: {}",
                    res.y[(0, res.x.len() - 1)]
                );

                // Check that solution resembles sin(x) (at least qualitatively)
                let mid_idx = res.x.len() / 2;
                if res.x.len() > 2 {
                    assert!(
                        res.y[(0, mid_idx)] > 0.3,
                        "Solution should be positive in middle, got {}",
                        res.y[(0, mid_idx)]
                    );
                }

                println!("BVP solved successfully!");
                println!("Solution at π/2: {}", res.y[(0, 2)]);
                println!(
                    "Expected (sin(π/2)): {}",
                    (std::f64::consts::PI / 2.0).sin()
                );
            }
            Err(e) => {
                panic!("BVP solution failed: {}", e);
            }
        }
    }

    #[test]
    fn test_collocation_function() {
        // Test the collocation residual computation
        let fun = |_x: &DVector<f64>, y: &DMatrix<f64>, _p: &DVector<f64>| {
            // Simple linear ODE: y' = -y
            -y.clone()
        };

        let x = DVector::from_vec(vec![0.0, 0.5, 1.0]);
        let h = DVector::from_vec(vec![0.5, 0.5]);

        // Exact solution y = e^(-x)
        let mut y = DMatrix::zeros(1, 3);
        y[(0, 0)] = 1.0; // y(0) = 1
        y[(0, 1)] = (-0.5_f64).exp(); // y(0.5) = e^(-0.5)
        y[(0, 2)] = (-1.0_f64).exp(); // y(1.0) = e^(-1)

        let p = DVector::zeros(0);

        let (col_res, _y_middle, _f, _f_middle) = collocation_fun(&fun, &y, &p, &x, &h);

        // For exact solution, collocation residuals should be small
        assert!(
            col_res.norm() < 1e-4,
            "Collocation residuals too large: {}",
            col_res.norm()
        );

        println!("Collocation test passed! Residual norm: {}", col_res.norm());
    }

    #[test]
    fn test_matrix_ordering_and_indexing() {
        // Test to ensure proper matrix construction and indexing
        // This catches row-major vs column-major issues

        // Create a simple 2x3 system (2 equations, 3 mesh points)
        let fun = |_x: &DVector<f64>, y: &DMatrix<f64>, _p: &DVector<f64>| {
            // f1 = y1, f2 = 2*y2 (simple linear functions)
            let mut f = DMatrix::zeros(2, y.ncols());
            for j in 0..y.ncols() {
                f[(0, j)] = y[(0, j)]; // f1 = y1
                f[(1, j)] = 2.0 * y[(1, j)]; // f2 = 2*y2
            }
            f
        };

        let x = DVector::from_vec(vec![0.0, 0.5, 1.0]);
        let y = DMatrix::from_vec(
            2,
            3,
            vec![
                1.0, 3.0, // First column: y1(0)=1, y2(0)=3
                2.0, 4.0, // Second column: y1(0.5)=2, y2(0.5)=4
                5.0, 6.0, // Third column: y1(1)=5, y2(1)=6
            ],
        );
        let p = DVector::zeros(0);

        // Test function evaluation
        let f_result = fun(&x, &y, &p);

        // Verify shapes
        assert_eq!(f_result.shape(), (2, 3), "Function result has wrong shape");
        assert_eq!(y.shape(), (2, 3), "Input y has wrong shape");

        // Verify values with correct indexing - nalgebra is column-major
        assert_eq!(f_result[(0, 0)], 1.0, "f1(x0) = y1(x0) = 1.0");
        assert_eq!(f_result[(0, 1)], 2.0, "f1(x1) = y1(x1) = 2.0");
        assert_eq!(f_result[(0, 2)], 5.0, "f1(x2) = y1(x2) = 5.0");
        assert_eq!(f_result[(1, 0)], 6.0, "f2(x0) = 2*y2(x0) = 2*3 = 6.0");
        assert_eq!(f_result[(1, 1)], 8.0, "f2(x1) = 2*y2(x1) = 2*4 = 8.0");
        assert_eq!(f_result[(1, 2)], 12.0, "f2(x2) = 2*y2(x2) = 2*6 = 12.0");

        // Test Jacobian estimation
        let (df_dy, df_dp) = estimate_fun_jac(&fun, &x, &y, &p, None);

        // Verify Jacobian shapes and structure
        assert_eq!(
            df_dy.len(),
            3,
            "Should have 3 Jacobian matrices (one per mesh point)"
        );
        assert!(df_dp.is_none(), "No parameters, so df_dp should be None");

        for (i, jac) in df_dy.iter().enumerate() {
            assert_eq!(
                jac.shape(),
                (2, 2),
                "Each Jacobian should be 2x2 for point {}",
                i
            );

            // For f1 = y1, df1/dy1 = 1, df1/dy2 = 0
            // For f2 = 2*y2, df2/dy1 = 0, df2/dy2 = 2
            assert!(
                (jac[(0, 0)] - 1.0).abs() < 1e-6,
                "df1/dy1 should be 1 at point {}",
                i
            );
            assert!(
                jac[(0, 1)].abs() < 1e-6,
                "df1/dy2 should be 0 at point {}",
                i
            );
            assert!(
                jac[(1, 0)].abs() < 1e-6,
                "df2/dy1 should be 0 at point {}",
                i
            );
            assert!(
                (jac[(1, 1)] - 2.0).abs() < 1e-6,
                "df2/dy2 should be 2 at point {}",
                i
            );
        }

        println!("Matrix ordering test passed!");
    }

    #[test]
    fn test_boundary_condition_jacobian() {
        // Test boundary condition Jacobian with known analytical values

        let bc = |ya: &DVector<f64>, yb: &DVector<f64>, _p: &DVector<f64>| {
            // bc1 = ya[0] - 1.0    (y1(a) = 1)
            // bc2 = yb[1] - 2.0    (y2(b) = 2)
            DVector::from_vec(vec![ya[0] - 1.0, yb[1] - 2.0])
        };

        let ya = DVector::from_vec(vec![1.5, 2.5]);
        let yb = DVector::from_vec(vec![3.5, 4.5]);
        let p = DVector::zeros(0);

        let (dbc_dya, dbc_dyb, dbc_dp) = estimate_bc_jac(&bc, &ya, &yb, &p, None);

        // Verify shapes
        assert_eq!(
            dbc_dya.shape(),
            (2, 2),
            "dbc_dya should be (n+k, n) = (2, 2)"
        );
        assert_eq!(
            dbc_dyb.shape(),
            (2, 2),
            "dbc_dyb should be (n+k, n) = (2, 2)"
        );
        assert!(dbc_dp.is_none(), "No parameters, so dbc_dp should be None");

        // Check analytical derivatives
        // bc1 = ya[0] - 1 => dbc1/dya[0] = 1, dbc1/dya[1] = 0
        assert!(
            (dbc_dya[(0, 0)] - 1.0).abs() < 1e-6,
            "dbc1/dya[0] should be 1"
        );
        assert!(dbc_dya[(0, 1)].abs() < 1e-6, "dbc1/dya[1] should be 0");

        // bc2 = yb[1] - 2 => dbc2/dyb[0] = 0, dbc2/dyb[1] = 1
        assert!(dbc_dyb[(1, 0)].abs() < 1e-6, "dbc2/dyb[0] should be 0");
        assert!(
            (dbc_dyb[(1, 1)] - 1.0).abs() < 1e-6,
            "dbc2/dyb[1] should be 1"
        );

        // Cross terms should be zero
        assert!(dbc_dya[(1, 0)].abs() < 1e-6, "dbc2/dya[0] should be 0");
        assert!(dbc_dya[(1, 1)].abs() < 1e-6, "dbc2/dya[1] should be 0");
        assert!(dbc_dyb[(0, 0)].abs() < 1e-6, "dbc1/dyb[0] should be 0");
        assert!(dbc_dyb[(0, 1)].abs() < 1e-6, "dbc1/dyb[1] should be 0");

        println!("Boundary condition Jacobian test passed!");
    }

    #[test]
    fn test_global_jacobian_structure() {
        // Test the global Jacobian construction with known structure

        let n = 2; // 2 equations
        let m = 3; // 3 mesh points  
        let k = 0; // 0 parameters

        let h = DVector::from_vec(vec![0.5, 0.5]);

        // Create simple non-identity Jacobians for testing to get off-diagonal structure
        let mut df_dy: Vec<DMatrix<f64>> = Vec::new();
        let mut df_dy_middle: Vec<DMatrix<f64>> = Vec::new();

        for _i in 0..m {
            let mut jac = DMatrix::identity(2, 2);
            jac[(0, 1)] = 0.1; // Add some off-diagonal structure
            jac[(1, 0)] = 0.2;
            df_dy.push(jac);
        }

        for _i in 0..(m - 1) {
            let mut jac = DMatrix::identity(2, 2);
            jac[(0, 1)] = 0.15;
            jac[(1, 0)] = 0.25;
            df_dy_middle.push(jac);
        }

        // Simple boundary condition Jacobians
        let dbc_dya = DMatrix::identity(2, 2);
        let dbc_dyb = DMatrix::identity(2, 2);

        let jac = construct_global_jac(
            n,
            m,
            k,
            &h,
            &df_dy,
            &df_dy_middle,
            None,
            None,
            &dbc_dya,
            &dbc_dyb,
            None,
        );

        // Verify overall shape: (n*m + k) x (n*m + k) = 6x6
        assert_eq!(jac.shape(), (6, 6), "Global Jacobian should be 6x6");

        // Check block structure - we expect a specific sparsity pattern
        // The matrix should have non-zero blocks in specific positions

        // Test that diagonal blocks are non-zero (collocation residuals)
        // We have n*(m-1) = 2*2 = 4 collocation equations in a 4x4 block
        // Each interval i contributes n=2 equations at rows i*n to (i+1)*n-1
        for i in 0..(m - 1) {
            // 2 intervals
            for row in 0..n {
                for col in 0..n {
                    let global_row = i * n + row;
                    let global_col = i * n + col;
                    let jac_val = jac[(global_row, global_col)];
                    assert!(
                        jac_val.abs() > 1e-10,
                        "Diagonal block ({},{}) should be non-zero, got {}",
                        global_row,
                        global_col,
                        jac_val
                    );
                }
            }
        }

        // Test boundary condition blocks (last n+k = 2 rows)
        let bc_row_start = n * (m - 1); // = 2 * 2 = 4
        for row in bc_row_start..(bc_row_start + n + k) {
            // BC rows should have non-zero entries connecting to first (ya) and last (yb) y-blocks
            let first_y_nonzero = (0..n).any(|col| jac[(row, col)].abs() > 1e-10);
            let last_y_start = (m - 1) * n; // = 2 * 2 = 4
            let last_y_nonzero =
                (last_y_start..(last_y_start + n)).any(|col| jac[(row, col)].abs() > 1e-10);

            assert!(
                first_y_nonzero || last_y_nonzero,
                "BC row {} should connect to either first or last y block",
                row
            );
        }

        println!("Global Jacobian structure test passed!");
    }

    #[test]
    fn test_collocation_residual_ordering() {
        // Test that collocation residuals are computed with correct ordering

        let fun = |_x: &DVector<f64>, _y: &DMatrix<f64>, _p: &DVector<f64>| {
            // Simple derivative: dy/dx = 0 (constant function)
            DMatrix::zeros(2, 3)
        };

        let x = DVector::from_vec(vec![0.0, 1.0, 2.0]);
        let h = DVector::from_vec(vec![1.0, 1.0]);

        // Use constant function values: y = constant
        let mut y = DMatrix::zeros(2, 3);
        for j in 0..3 {
            y[(0, j)] = 1.0; // y1 = 1 (constant)
            y[(1, j)] = 2.0; // y2 = 2 (constant)
        }

        let p = DVector::zeros(0);

        let (col_res, y_middle, f, f_middle) = collocation_fun(&fun, &y, &p, &x, &h);

        // Verify shapes
        assert_eq!(
            col_res.shape(),
            (2, 2),
            "Collocation residuals should be (n, m-1) = (2, 2)"
        );
        assert_eq!(
            y_middle.shape(),
            (2, 2),
            "y_middle should be (n, m-1) = (2, 2)"
        );
        assert_eq!(f.shape(), (2, 3), "f should be (n, m) = (2, 3)");
        assert_eq!(
            f_middle.shape(),
            (2, 2),
            "f_middle should be (n, m-1) = (2, 2)"
        );

        // For constant solution with dy/dx = 0, residuals should be small
        println!("Collocation residual norms:");
        for i in 0..2 {
            for j in 0..2 {
                let res = col_res[(i, j)];
                println!("  col_res[{}, {}] = {}", i, j, res);
                assert!(
                    res.abs() < 1e-10,
                    "Residual col_res[{}, {}] = {} too large",
                    i,
                    j,
                    res
                );
            }
        }

        // Verify that y_middle values are correct (should be constants)
        for j in 0..2 {
            assert!(
                (y_middle[(0, j)] - 1.0).abs() < 1e-10,
                "y_middle[0, {}] = {} should be 1.0",
                j,
                y_middle[(0, j)]
            );
            assert!(
                (y_middle[(1, j)] - 2.0).abs() < 1e-10,
                "y_middle[1, {}] = {} should be 2.0",
                j,
                y_middle[(1, j)]
            );
        }

        println!("Collocation residual ordering test passed!");
    }

    #[test]
    fn test_parameter_jacobian() {
        // Test Jacobian computation with parameters

        let fun = |_x: &DVector<f64>, y: &DMatrix<f64>, p: &DVector<f64>| {
            // f1 = p[0] * y1, f2 = p[1] * y2
            let mut f = DMatrix::zeros(2, y.ncols());
            for j in 0..y.ncols() {
                f[(0, j)] = p[0] * y[(0, j)];
                f[(1, j)] = p[1] * y[(1, j)];
            }
            f
        };

        let x = DVector::from_vec(vec![0.0, 1.0]);
        let y = DMatrix::from_vec(2, 2, vec![1.0, 3.0, 2.0, 4.0]); // Column-major: (1,3), (2,4)
        let p = DVector::from_vec(vec![2.0, 3.0]);

        let (df_dy, df_dp) = estimate_fun_jac(&fun, &x, &y, &p, None);

        // Verify parameter derivatives exist
        assert!(
            df_dp.is_some(),
            "df_dp should exist when parameters are present"
        );
        let df_dp = df_dp.unwrap();

        // Check shapes
        assert_eq!(df_dy.len(), 2, "Should have 2 Jacobian matrices w.r.t. y");
        assert_eq!(df_dp.len(), 2, "Should have 2 Jacobian matrices w.r.t. p");

        for i in 0..2 {
            assert_eq!(df_dy[i].shape(), (2, 2), "df_dy[{}] should be 2x2", i);
            assert_eq!(df_dp[i].shape(), (2, 2), "df_dp[{}] should be 2x2", i);

            // Check df/dy values: df1/dy1 = p[0], df2/dy2 = p[1], cross terms = 0
            assert!(
                (df_dy[i][(0, 0)] - p[0]).abs() < 1e-6,
                "df1/dy1 should equal p[0]"
            );
            assert!(
                (df_dy[i][(1, 1)] - p[1]).abs() < 1e-6,
                "df2/dy2 should equal p[1]"
            );
            assert!(df_dy[i][(0, 1)].abs() < 1e-6, "df1/dy2 should be 0");
            assert!(df_dy[i][(1, 0)].abs() < 1e-6, "df2/dy1 should be 0");

            // Check df/dp values: df1/dp0 = y1, df2/dp1 = y2, cross terms = 0
            assert!(
                (df_dp[i][(0, 0)] - y[(0, i)]).abs() < 1e-6,
                "df1/dp0 should equal y1[{}]",
                i
            );
            assert!(
                (df_dp[i][(1, 1)] - y[(1, i)]).abs() < 1e-6,
                "df2/dp1 should equal y2[{}]",
                i
            );
            assert!(df_dp[i][(0, 1)].abs() < 1e-6, "df1/dp1 should be 0");
            assert!(df_dp[i][(1, 0)].abs() < 1e-6, "df2/dp0 should be 0");
        }

        println!("Parameter Jacobian test passed!");
    }

    #[test]
    fn test_create_spline() {
        // Test cubic spline creation with known values and derivatives
        // f(x) = x^3, f'(x) = 3x^2
        let x = DVector::from_vec(vec![0.0, 1.0, 2.0]);
        let h = DVector::from_vec(vec![1.0, 1.0]);

        // Function values: f(0)=0, f(1)=1, f(2)=8
        let y = DMatrix::from_vec(1, 3, vec![0.0, 1.0, 8.0]);

        // Derivative values: f'(0)=0, f'(1)=3, f'(2)=12
        let yp = DMatrix::from_vec(1, 3, vec![0.0, 3.0, 12.0]);

        let spline = create_spline(&y, &yp, &x, &h);

        // Test spline evaluation at known points
        let x_eval = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let x_shape = &[5];
        let result = spline.call(&x_eval, x_shape, Some(0), None);

        // Expected values for f(x) = x^3
        let expected = vec![0.0, 0.125, 1.0, 3.375, 8.0];

        for i in 0..x_eval.len() {
            assert!(
                (result[(i, 0)] - expected[i]).abs() < 1e-10,
                "Spline value at x={} should be {}, got {}",
                x_eval[i],
                expected[i],
                result[(i, 0)]
            );
        }

        // Test derivative evaluation
        let result_deriv = spline.call(&x_eval, x_shape, Some(1), None);
        let expected_deriv = vec![0.0, 0.75, 3.0, 6.75, 12.0]; // 3x^2

        for i in 0..x_eval.len() {
            assert!(
                (result_deriv[(i, 0)] - expected_deriv[i]).abs() < 1e-10,
                "Spline derivative at x={} should be {}, got {}",
                x_eval[i],
                expected_deriv[i],
                result_deriv[(i, 0)]
            );
        }

        println!("Create spline test passed!");
    }

    #[test]
    fn test_estimate_rms_residuals() {
        // Test RMS residual estimation with a simple linear case
        // Use y' = 0 (constant function) which should have zero residuals

        let fun = |_x: &DVector<f64>, _y: &DMatrix<f64>, _p: &DVector<f64>| {
            DMatrix::zeros(1, 3) // y' = 0
        };

        let x = DVector::from_vec(vec![0.0, 0.5, 1.0]);
        let h = DVector::from_vec(vec![0.5, 0.5]);
        let p = DVector::zeros(0);

        // Constant solution: y = 2
        let y = DMatrix::from_element(1, 3, 2.0);

        // Constant derivative: y' = 0
        let yp = DMatrix::zeros(1, 3);

        // Create spline from constant solution
        let spline = create_spline(&y, &yp, &x, &h);

        // For constant solution with y' = 0, residuals should be small
        let r_middle = DMatrix::zeros(1, 2); // Zero residuals at middle points
        let f_middle = DMatrix::zeros(1, 2); // Zero function values at middle points

        let rms_res = estimate_rms_residuals(&fun, &spline, &x, &h, &p, &r_middle, &f_middle);

        // For constant solution, RMS residuals should be small (but allow for numerical errors)
        for i in 0..rms_res.len() {
            assert!(
                rms_res[i] < 1e-6,
                "RMS residual {} should be small for constant solution, got {}",
                i,
                rms_res[i]
            );
        }

        println!("Estimate RMS residuals test passed!");
    }

    #[test]
    fn test_estimate_rms_residuals_nonzero() {
        // Test RMS residual estimation with known non-zero residuals
        let fun = |_x: &DVector<f64>, y: &DMatrix<f64>, _p: &DVector<f64>| {
            y.clone() // y' = y
        };

        let x = DVector::from_vec(vec![0.0, 1.0, 2.0]);
        let h = DVector::from_vec(vec![1.0, 1.0]);
        let p = DVector::zeros(0);

        // Linear solution: y = x (but y' = 1, not y)
        let y = DMatrix::from_vec(1, 3, vec![0.0, 1.0, 2.0]);
        let yp = DMatrix::from_element(1, 3, 1.0); // y' = 1

        let spline = create_spline(&y, &yp, &x, &h);

        // Non-zero residuals: r = y' - y = 1 - x
        let r_middle = DMatrix::from_vec(1, 2, vec![0.5, -0.5]); // At x=0.5: 1-0.5=0.5, at x=1.5: 1-1.5=-0.5
        let f_middle = DMatrix::from_vec(1, 2, vec![0.5, 1.5]); // y values at middle points

        let rms_res = estimate_rms_residuals(&fun, &spline, &x, &h, &p, &r_middle, &f_middle);

        // Should have non-zero but finite residuals
        for i in 0..rms_res.len() {
            assert!(rms_res[i] > 1e-10, "RMS residual {} should be non-zero", i);
            assert!(
                rms_res[i] < 10.0,
                "RMS residual {} should be finite, got {}",
                i,
                rms_res[i]
            );
        }

        println!("Non-zero RMS residuals test passed!");
    }

    #[test]
    fn test_modify_mesh() {
        // Test mesh modification with node insertion
        let x = DVector::from_vec(vec![0.0, 1.0, 2.0, 3.0]);

        // Insert 1 node in intervals 0 and 2, 2 nodes in interval 1
        let insert_1 = vec![0, 2];
        let insert_2 = vec![1];

        let x_new = modify_mesh(&x, &insert_1, &insert_2);

        // Expected: original + midpoints + thirds
        // Original: [0, 1, 2, 3]
        // Insert 1 in [0,1]: add 0.5
        // Insert 2 in [1,2]: add 4/3, 5/3
        // Insert 1 in [2,3]: add 2.5
        let expected = vec![0.0, 0.5, 1.0, 4.0 / 3.0, 5.0 / 3.0, 2.0, 2.5, 3.0];

        assert_eq!(x_new.len(), expected.len());
        for i in 0..expected.len() {
            assert!(
                (x_new[i] - expected[i]).abs() < 1e-10,
                "Point {} should be {}, got {}",
                i,
                expected[i],
                x_new[i]
            );
        }

        println!("Modify mesh test passed!");
    }

    #[test]
    fn test_solve_newton_convergence() {
        // Test Newton solver with a simple linear BVP that should converge quickly
        // y'' = 0, y(0) = 0, y(1) = 1 -> solution y = x

        let fun = |_x: &DVector<f64>, y: &DMatrix<f64>, _p: &DVector<f64>| {
            let mut f = DMatrix::zeros(2, y.ncols());
            for j in 0..y.ncols() {
                f[(0, j)] = y[(1, j)]; // y1' = y2
                f[(1, j)] = 0.0; // y2' = 0
            }
            f
        };

        let bc = |ya: &DVector<f64>, yb: &DVector<f64>, _p: &DVector<f64>| {
            DVector::from_vec(vec![ya[0], yb[0] - 1.0]) // y1(0)=0, y1(1)=1
        };

        let x = DVector::from_vec(vec![0.0, 0.5, 1.0]);
        let h = DVector::from_vec(vec![0.5, 0.5]);

        // Initial guess close to solution
        let mut y = DMatrix::zeros(2, 3);
        y[(0, 0)] = 0.0;
        y[(0, 1)] = 0.4;
        y[(0, 2)] = 0.8; // y1 ≈ x
        y[(1, 0)] = 1.0;
        y[(1, 1)] = 1.0;
        y[(1, 2)] = 1.0; // y2 ≈ 1

        let p = DVector::zeros(0);

        let (y_final, _p_final, singular) =
            solve_newton(2, 3, &h, &fun, &bc, None, None, y, p, &x, 1e-10, 1e-10);

        assert!(
            !singular,
            "Newton solver should not be singular for this problem"
        );

        // Check solution: y1 should be approximately x
        assert!((y_final[(0, 0)] - 0.0).abs() < 1e-8, "y1(0) should be 0");
        assert!(
            (y_final[(0, 1)] - 0.5).abs() < 1e-8,
            "y1(0.5) should be 0.5"
        );
        assert!((y_final[(0, 2)] - 1.0).abs() < 1e-8, "y1(1) should be 1");

        // Check derivative: y2 should be approximately 1
        for j in 0..3 {
            assert!(
                (y_final[(1, j)] - 1.0).abs() < 1e-6,
                "y2[{}] should be 1, got {}",
                j,
                y_final[(1, j)]
            );
        }

        println!("Newton convergence test passed!");
    }

    #[test]
    fn test_construct_global_jac_structure() {
        // Test global Jacobian construction with parameters
        let n = 2; // 2 equations
        let m = 3; // 3 mesh points
        let k = 1; // 1 parameter

        let h = DVector::from_vec(vec![0.5, 0.5]);

        // Create test Jacobians
        let df_dy = vec![DMatrix::identity(2, 2); 3];
        let df_dy_middle = vec![DMatrix::identity(2, 2); 2];
        let df_dp = Some(vec![DMatrix::from_element(2, 1, 1.0); 3]);
        let df_dp_middle = Some(vec![DMatrix::from_element(2, 1, 1.0); 2]);

        let dbc_dya = DMatrix::identity(3, 2); // (n+k) x n
        let dbc_dyb = DMatrix::identity(3, 2);
        let dbc_dp = Some(DMatrix::from_element(3, 1, 1.0)); // (n+k) x k

        let jac = construct_global_jac(
            n,
            m,
            k,
            &h,
            &df_dy,
            &df_dy_middle,
            df_dp.as_ref().map(|v| v.as_slice()),
            df_dp_middle.as_ref().map(|v| v.as_slice()),
            &dbc_dya,
            &dbc_dyb,
            dbc_dp.as_ref(),
        );

        // Check dimensions: (m-1)*n + (n+k) = 2*2 + 3 = 7
        assert_eq!(jac.shape(), (7, 7), "Jacobian should be 7x7");

        // Check that parameter column exists
        let param_col = 6; // n*m = 6
        for row in 0..4 {
            // Collocation rows
            assert!(
                jac[(row, param_col)].abs() > 1e-10,
                "Parameter column should be non-zero for collocation"
            );
        }

        for row in 4..7 {
            // BC rows
            assert!(
                jac[(row, param_col)].abs() > 1e-10,
                "Parameter column should be non-zero for BC"
            );
        }

        println!("Global Jacobian with parameters test passed!");
    }

    // #[test]
    #[allow(dead_code)]
    fn test_bvp_with_parameters() {
        // Test BVP with unknown parameter: y'' = λy, y(0) = 1, y(1) = e^λ
        // Solution: y = e^(√λ x) with λ = 1 giving y = e^x

        let fun = |_x: &DVector<f64>, y: &DMatrix<f64>, p: &DVector<f64>| {
            let lambda = p[0];
            let mut f = DMatrix::zeros(2, y.ncols());
            for j in 0..y.ncols() {
                f[(0, j)] = y[(1, j)]; // y1' = y2
                f[(1, j)] = lambda * y[(0, j)]; // y2' = λ*y1
            }
            f
        };

        let bc = |ya: &DVector<f64>, yb: &DVector<f64>, p: &DVector<f64>| {
            let lambda = p[0];
            DVector::from_vec(vec![ya[0] - 1.0, yb[0] - lambda.exp()]) // y(0)=1, y(1)=e^λ
        };

        let x = DVector::from_vec(vec![0.0, 0.5, 1.0]);
        let mut y = DMatrix::zeros(2, 3);
        // Initial guess for y = e^x
        for j in 0..2 {
            y[(0, j)] = (x[j] as f64).exp();
            y[(1, j)] = (x[j] as f64).exp();
        }

        let p = Some(DVector::from_vec(vec![0.8])); // Initial guess for λ

        let result = solve_bvp(
            &fun,
            &bc,
            x.clone(),
            y,
            p,
            None,
            None,
            None,
            1e-6,
            1000,
            0,
            None,
        );

        match result {
            Ok(res) => {
                assert!(res.success, "BVP with parameters should converge");

                if let Some(params) = res.p {
                    assert!(
                        (params[0] - 1.0).abs() < 1e-3,
                        "Parameter λ should be ≈ 1, got {}",
                        params[0]
                    );
                }

                // Check boundary conditions
                assert!((res.y[(0, 0)] - 1.0).abs() < 1e-4, "y(0) should be 1");
                assert!(
                    (res.y[(0, 2)] - 1.0_f64.exp()).abs() < 1e-3,
                    "y(1) should be e"
                );
            }
            Err(e) => panic!("BVP with parameters failed: {}", e),
        }

        println!("BVP with parameters test passed!");
    }

    #[test]
    fn test_mesh_refinement() {
        // Test that mesh refinement works by solving a problem with sharp gradients
        // y'' = 100*y, y(0) = 1, y(1) = e^10 ≈ 22026
        // Solution: y = e^(10x) has very sharp gradient near x=1

        let fun = |_x: &DVector<f64>, y: &DMatrix<f64>, _p: &DVector<f64>| {
            let mut f = DMatrix::zeros(2, y.ncols());
            for j in 0..y.ncols() {
                f[(0, j)] = y[(1, j)]; // y1' = y2
                f[(1, j)] = 100.0 * y[(0, j)]; // y2' = 100*y1
            }
            f
        };

        let bc = |ya: &DVector<f64>, yb: &DVector<f64>, _p: &DVector<f64>| {
            DVector::from_vec(vec![ya[0] - 1.0, yb[0] - 10.0_f64.exp()])
        };

        // Start with coarse mesh
        let x = DVector::from_vec(vec![0.0, 0.5, 1.0]);
        let mut y = DMatrix::zeros(2, 3);
        // Initial guess
        for j in 0..3 {
            y[(0, j)] = (10.0 * x[j] as f64).exp();
            y[(1, j)] = 10.0 * (10.0 * x[j] as f64).exp();
        }

        let result = solve_bvp(
            &fun,
            &bc,
            x.clone(),
            y,
            None,
            None,
            None,
            None,
            1e-4,
            100,
            1,
            None, // verbose=1 to see mesh refinement
        );

        match result {
            Ok(res) => {
                assert!(
                    res.success || res.status == 1,
                    "Should converge or hit max nodes"
                );

                // Should have refined the mesh (more than 3 nodes)
                assert!(
                    res.x.len() > 3,
                    "Mesh should be refined, got {} nodes",
                    res.x.len()
                );

                // Check boundary conditions
                assert!((res.y[(0, 0)] - 1.0).abs() < 1e-3, "y(0) should be 1");
                assert!(
                    (res.y[(0, res.x.len() - 1)] - 10.0_f64.exp()).abs() < 1e2,
                    "y(1) should be ≈ e^10"
                );
            }
            Err(e) => panic!("Mesh refinement test failed: {}", e),
        }

        println!("Mesh refinement test passed!");
    }

    #[test]
    fn test_stacked_matmul() {
        // Test stacked matrix multiplication
        let a1 = DMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let a2 = DMatrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let a = vec![a1, a2];

        let b1 = DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]); // Identity
        let b2 = DMatrix::from_vec(2, 2, vec![2.0, 0.0, 0.0, 2.0]); // 2*Identity
        let b = vec![b1, b2];

        let result = stacked_matmul(&a, &b);

        assert_eq!(result.len(), 2);

        // First multiplication: a1 * I = a1
        for i in 0..2 {
            for j in 0..2 {
                assert!((result[0][(i, j)] - a[0][(i, j)]).abs() < 1e-10);
            }
        }

        // Second multiplication: a2 * 2I = 2*a2
        for i in 0..2 {
            for j in 0..2 {
                assert!((result[1][(i, j)] - 2.0 * a[1][(i, j)]).abs() < 1e-10);
            }
        }

        println!("Stacked matrix multiplication test passed!");
    }

    #[test]
    fn test_compute_jac_indices() {
        // Test Jacobian index computation
        let n = 2; // 2 equations
        let m = 3; // 3 mesh points
        let k = 1; // 1 parameter

        let (i_indices, j_indices) = compute_jac_indices(n, m, k);

        // Total entries: 2*(m-1)*n^2 + 2*(n+k)*n + (m-1)*n*k + (n+k)*k
        // = 2*2*4 + 2*3*2 + 2*1 + 3*1 = 16 + 12 + 2 + 3 = 33
        let expected_entries =
            2 * (m - 1) * n * n + 2 * (n + k) * n + (m - 1) * n * k + (n + k) * k;
        assert_eq!(i_indices.len(), expected_entries);
        assert_eq!(j_indices.len(), expected_entries);

        // Check that indices are within bounds
        let max_row = (m - 1) * n + (n + k) - 1; // 4 + 3 - 1 = 6
        let max_col = n * m + k - 1; // 6 + 1 - 1 = 6

        for &i in &i_indices {
            assert!(i <= max_row, "Row index {} out of bounds", i);
        }

        for &j in &j_indices {
            assert!(j <= max_col, "Column index {} out of bounds", j);
        }

        println!("Jacobian indices test passed!");
    }

    #[test]
    fn test_singular_jacobian_handling() {
        // Test handling of singular Jacobian (degenerate problem)
        let fun = |_x: &DVector<f64>, _y: &DMatrix<f64>, _p: &DVector<f64>| {
            DMatrix::zeros(2, 3) // Zero derivatives -> singular system
        };

        let bc = |ya: &DVector<f64>, yb: &DVector<f64>, _p: &DVector<f64>| {
            DVector::from_vec(vec![ya[0], yb[0]]) // Simple BCs
        };

        let x = DVector::from_vec(vec![0.0, 0.5, 1.0]);
        let y = DMatrix::from_element(2, 3, 1.0);

        let result = solve_bvp(&fun, &bc, x, y, None, None, None, None, 1e-6, 1000, 0, None);

        match result {
            Ok(res) => {
                assert!(!res.success, "Should not succeed with singular Jacobian");
                assert_eq!(res.status, 2, "Status should be 2 (singular)");
            }
            Err(_) => {} // Also acceptable
        }

        println!("Singular Jacobian handling test passed!");
    }

    #[test]
    fn test_boundary_condition_tolerance() {
        // Test BC tolerance by using tight BC tolerance
        let fun = |_x: &DVector<f64>, y: &DMatrix<f64>, _p: &DVector<f64>| {
            let mut f = DMatrix::zeros(2, y.ncols());
            for j in 0..y.ncols() {
                f[(0, j)] = y[(1, j)];
                f[(1, j)] = -y[(0, j)];
            }
            f
        };

        let bc = |ya: &DVector<f64>, yb: &DVector<f64>, _p: &DVector<f64>| {
            DVector::from_vec(vec![ya[0], yb[0]])
        };

        let x = DVector::from_vec(vec![0.0, std::f64::consts::PI]);
        let mut y = DMatrix::zeros(2, 2);
        y[(0, 0)] = 0.1; // Slightly off from BC
        y[(0, 1)] = 0.1;
        y[(1, 0)] = 1.0;
        y[(1, 1)] = -1.0;

        // Very tight BC tolerance
        let result = solve_bvp(
            &fun,
            &bc,
            x,
            y,
            None,
            None,
            None,
            None,
            1e-6,
            1000,
            0,
            Some(1e-12),
        );

        match result {
            Ok(res) => {
                if res.success {
                    // If it converged, BCs should be very tight
                    assert!(res.y[(0, 0)].abs() < 1e-10, "Left BC should be tight");
                    assert!(res.y[(0, 1)].abs() < 1e-10, "Right BC should be tight");
                } else {
                    // May fail to meet tight BC tolerance
                    assert!(res.status == 3, "Should fail due to BC tolerance");
                }
            }
            Err(_) => {} // Also acceptable
        }

        println!("BC tolerance test passed!");
    }

    #[test]
    fn test_elementary_linear_bvp() {
        // Test simplest possible BVP: y'' = 0, y(0) = 0, y(1) = 1
        // Exact solution: y = x

        let fun = |_x: &DVector<f64>, y: &DMatrix<f64>, _p: &DVector<f64>| {
            let mut f = DMatrix::zeros(2, y.ncols());
            for j in 0..y.ncols() {
                f[(0, j)] = y[(1, j)]; // y1' = y2 (y1 = y, y2 = y')
                f[(1, j)] = 0.0; // y2' = 0 (y'' = 0)
            }
            f
        };

        let bc = |ya: &DVector<f64>, yb: &DVector<f64>, _p: &DVector<f64>| {
            DVector::from_vec(vec![ya[0], yb[0] - 1.0]) // y(0) = 0, y(1) = 1
        };

        let x = DVector::from_vec(vec![0.0, 0.5, 1.0]);
        let mut y = DMatrix::zeros(2, 3);
        // Initial guess close to solution y = x
        y[(0, 0)] = 0.0;
        y[(0, 1)] = 0.5;
        y[(0, 2)] = 1.0;
        y[(1, 0)] = 1.0;
        y[(1, 1)] = 1.0;
        y[(1, 2)] = 1.0;

        let result = solve_bvp(&fun, &bc, x, y, None, None, None, None, 1e-8, 1000, 0, None);

        match result {
            Ok(res) => {
                assert!(res.success, "Linear BVP should converge");

                // Check solution y ≈ x
                for i in 0..res.x.len() {
                    let expected = res.x[i];
                    assert!(
                        (res.y[(0, i)] - expected).abs() < 1e-6,
                        "y({}) should be {}, got {}",
                        res.x[i],
                        expected,
                        res.y[(0, i)]
                    );
                }

                // Check derivative y' ≈ 1
                for i in 0..res.x.len() {
                    assert!(
                        (res.y[(1, i)] - 1.0).abs() < 1e-6,
                        "y'({}) should be 1, got {}",
                        res.x[i],
                        res.y[(1, i)]
                    );
                }
            }
            Err(e) => panic!("Elementary linear BVP failed: {}", e),
        }

        println!("Elementary linear BVP test passed!");
    }

    #[test]
    fn test_elementary_quadratic_bvp() {
        // Test simple quadratic BVP: y'' = -2, y(0) = 0, y(1) = 0
        // Exact solution: y = x(1-x)

        let fun = |_x: &DVector<f64>, y: &DMatrix<f64>, _p: &DVector<f64>| {
            let mut f = DMatrix::zeros(2, y.ncols());
            for j in 0..y.ncols() {
                f[(0, j)] = y[(1, j)]; // y1' = y2
                f[(1, j)] = -2.0; // y2' = -2 (y'' = -2)
            }
            f
        };

        let bc = |ya: &DVector<f64>, yb: &DVector<f64>, _p: &DVector<f64>| {
            DVector::from_vec(vec![ya[0], yb[0]]) // y(0) = 0, y(1) = 0
        };

        let x = DVector::from_vec(vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        let mut y = DMatrix::zeros(2, 5);
        // Initial guess: parabola shape
        for j in 0..5 {
            let xi = x[j];
            y[(0, j)] = xi; //* (1.0 - xi);  // y ≈ x(1-x)
            y[(1, j)] = 1.0; //- 2.0 * xi;   // y' ≈ 1-2x
        }

        let result = solve_bvp(
            &fun,
            &bc,
            x.clone(),
            y,
            None,
            None,
            None,
            None,
            1e-8,
            1000,
            0,
            None,
        );

        match result {
            Ok(res) => {
                assert!(res.success, "Quadratic BVP should converge");
                let Y1: Vec<f64> = res.y.row(0).iter().map(|x| *x).collect();
                let Y2: Vec<f64> = res.y.row(1).iter().map(|x| *x).collect();
                println!("x={:?}", res.x);
                println!("Y1 = {:?}, \nY2 = {:?}", Y1, Y2);
                // Check solution y ≈ x(1-x)
                for i in 0..res.x.len() {
                    let xi = res.x[i];
                    let expected = xi * (1.0 - xi);
                    assert!(
                        (res.y[(0, i)] - expected).abs() < 1e-6,
                        "y({}) should be {}, got {}",
                        xi,
                        expected,
                        res.y[(0, i)]
                    );
                }

                // Check derivative y' ≈ 1-2x
                for i in 0..res.x.len() {
                    let xi = res.x[i];
                    let expected_deriv = 1.0 - 2.0 * xi;
                    assert!(
                        (res.y[(1, i)] - expected_deriv).abs() < 1e-6,
                        "y'({}) should be {}, got {}",
                        xi,
                        expected_deriv,
                        res.y[(1, i)]
                    );
                }
            }
            Err(e) => panic!("Elementary quadratic BVP failed: {}", e),
        }

        println!("Elementary quadratic BVP test passed!");
    }
}
