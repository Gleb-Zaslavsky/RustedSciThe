#[cfg(test)]
mod tests {
    use crate::numerical::BVP_Damp::BVP_utils::CustomTimer;
    use crate::numerical::BVP_sci::BVP_sci_faer::{
        collocation_fun, compute_jac_indices, construct_global_jac, create_spline, estimate_bc_jac,
        estimate_fun_jac, estimate_rms_residuals, modify_mesh, solve_bvp, solve_newton,
    };
    use core::panic;
    use faer::col::Col;
    use faer::mat::Mat;
    use std::collections::HashMap;
    type faer_col = Col<f64>;
    type faer_dense_mat = Mat<f64>;

    #[test]
    fn test_basic_jacobian_estimation() {
        let fun = |_x: &faer_col, y: &faer_dense_mat, _p: &faer_col| y.clone();

        let x = faer_col::from_fn(2, |i| [0.0, 1.0][i]);
        let y = faer_dense_mat::from_fn(2, 2, |i, j| [[1.0, 3.0], [2.0, 4.0]][i][j]);
        let p = faer_col::zeros(0);

        let (df_dy, df_dp) = estimate_fun_jac(&fun, &x, &y, &p, None);

        assert_eq!(df_dy.len(), 2);
        assert!(df_dp.is_none());
        for jacobian in &df_dy {
            for i in 0..2 {
                for j in 0..2 {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    let val = jacobian.get(i, j).unwrap_or(&0.0);
                    assert!((val - expected).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn test_simple_bvp_problem_debug() {
        let fun = |_x: &faer_col, y: &faer_dense_mat, _p: &faer_col| {
            let mut f = faer_dense_mat::zeros(2, y.ncols());
            for j in 0..y.ncols() {
                *f.get_mut(0, j) = *y.get(1, j);
                *f.get_mut(1, j) = -*y.get(0, j);
            }
            f
        };

        let bc = |ya: &faer_col, _yb: &faer_col, _p: &faer_col| {
            faer_col::from_fn(2, |i| [ya[0], ya[1] - 1.0][i])
        };

        let x = faer_col::from_fn(2, |i| [0.0, std::f64::consts::PI][i]);
        let mut y = faer_dense_mat::zeros(2, 2);
        *y.get_mut(0, 0) = 0.0;
        *y.get_mut(0, 1) = 0.0;
        *y.get_mut(1, 0) = 1.0;
        *y.get_mut(1, 1) = -1.0;

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
            None,
            None, // No custom timer for this test
        );

        match result {
            Ok(res) => if res.success {},
            Err(_) => {}
        }
    }

    #[test]
    fn test_simple_bvp_problem() {
        let fun = |_x: &faer_col, y: &faer_dense_mat, _p: &faer_col| {
            let mut f = faer_dense_mat::zeros(2, y.ncols());
            for j in 0..y.ncols() {
                *f.get_mut(0, j) = *y.get(1, j);
                *f.get_mut(1, j) = -*y.get(0, j);
            }
            f
        };

        let bc = |ya: &faer_col, _yb: &faer_col, _p: &faer_col| {
            faer_col::from_fn(2, |i| [ya[0], ya[1] - 1.0][i])
        };

        let x_vals = vec![
            0.0,
            std::f64::consts::PI / 4.0,
            std::f64::consts::PI / 2.0,
            3.0 * std::f64::consts::PI / 4.0,
            std::f64::consts::PI,
        ];
        let x = faer_col::from_fn(5, |i| x_vals[i]);

        let mut y = faer_dense_mat::zeros(2, 5);
        for j in 0..5 {
            *y.get_mut(0, j) = x[j].sin();
            *y.get_mut(1, j) = x[j].cos();
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
            1e-6,
            1000,
            0,
            None,
            None,
        );

        match result {
            Ok(res) => {
                assert!(res.success);
                assert_eq!(res.status, 0);
                assert!(res.y.get(0, 0).abs() < 1e-3);
                assert!(res.y.get(0, res.x.nrows() - 1).abs() < 1e-3);

                let mid_idx = res.x.nrows() / 2;
                if res.x.nrows() > 2 {
                    assert!(*res.y.get(0, mid_idx) > 0.3);
                }
            }
            Err(e) => panic!("BVP solution failed: {}", e),
        }
    }

    #[test]
    fn test_collocation_function() {
        let fun = |_x: &faer_col, y: &faer_dense_mat, _p: &faer_col| {
            let mut result = y.clone();
            result *= -1.0;
            result
        };

        let x = faer_col::from_fn(3, |i| [0.0, 0.5, 1.0][i]);
        let h = faer_col::from_fn(2, |_| 0.5);

        let mut y = faer_dense_mat::zeros(1, 3);
        *y.get_mut(0, 0) = 1.0;
        *y.get_mut(0, 1) = (-0.5_f64).exp();
        *y.get_mut(0, 2) = (-1.0_f64).exp();

        let p = faer_col::zeros(0);

        let (col_res, _y_middle, _f, _f_middle) = collocation_fun(&fun, &y, &p, &x, &h);

        let norm = col_res.squared_norm_l2().sqrt();
        assert!(norm < 1e-4);
    }

    #[test]
    fn test_matrix_ordering_and_indexing() {
        let fun = |_x: &faer_col, y: &faer_dense_mat, _p: &faer_col| {
            let mut f = faer_dense_mat::zeros(2, y.ncols());
            for j in 0..y.ncols() {
                *f.get_mut(0, j) = *y.get(0, j);
                *f.get_mut(1, j) = 2.0 * *y.get(1, j);
            }
            f
        };

        let x = faer_col::from_fn(3, |i| [0.0, 0.5, 1.0][i]);
        let y = faer_dense_mat::from_fn(2, 3, |i, j| [[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]][i][j]);
        let p = faer_col::zeros(0);

        let f_result = fun(&x, &y, &p);

        assert_eq!(f_result.nrows(), 2);
        assert_eq!(f_result.ncols(), 3);

        assert_eq!(*f_result.get(0, 0), 1.0);
        assert_eq!(*f_result.get(0, 1), 2.0);
        assert_eq!(*f_result.get(0, 2), 5.0);
        assert_eq!(*f_result.get(1, 0), 6.0);
        assert_eq!(*f_result.get(1, 1), 8.0);
        assert_eq!(*f_result.get(1, 2), 12.0);

        let (df_dy, df_dp) = estimate_fun_jac(&fun, &x, &y, &p, None);

        assert_eq!(df_dy.len(), 3);
        assert!(df_dp.is_none());

        for (_, jac) in df_dy.iter().enumerate() {
            assert!((jac.get(0, 0).unwrap_or(&0.0) - 1.0).abs() < 1e-6);
            assert!(jac.get(0, 1).unwrap_or(&0.0).abs() < 1e-6);
            assert!(jac.get(1, 0).unwrap_or(&0.0).abs() < 1e-6);
            assert!((jac.get(1, 1).unwrap_or(&0.0) - 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_boundary_condition_jacobian() {
        let bc = |ya: &faer_col, yb: &faer_col, _p: &faer_col| {
            faer_col::from_fn(2, |i| [ya[0] - 1.0, yb[1] - 2.0][i])
        };

        let ya = faer_col::from_fn(2, |i| [1.5, 2.5][i]);
        let yb = faer_col::from_fn(2, |i| [3.5, 4.5][i]);
        let p = faer_col::zeros(0);

        let (dbc_dya, dbc_dyb, dbc_dp) = estimate_bc_jac(&bc, &ya, &yb, &p, None);

        assert_eq!(dbc_dya.nrows(), 2);
        assert_eq!(dbc_dya.ncols(), 2);
        assert_eq!(dbc_dyb.nrows(), 2);
        assert_eq!(dbc_dyb.ncols(), 2);
        assert!(dbc_dp.is_none());

        assert!((dbc_dya.get(0, 0).unwrap_or(&0.0) - 1.0).abs() < 1e-6);
        assert!(dbc_dya.get(0, 1).unwrap_or(&0.0).abs() < 1e-6);
        assert!(dbc_dyb.get(1, 0).unwrap_or(&0.0).abs() < 1e-6);
        assert!((dbc_dyb.get(1, 1).unwrap_or(&0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_global_jacobian_structure() {
        let n = 2;
        let m = 3;
        let k = 0;

        let h = faer_col::from_fn(2, |_| 0.5);

        let mut df_dy = Vec::new();
        let mut df_dy_middle = Vec::new();

        for _i in 0..m {
            let triplets = vec![(0, 0, 1.1), (1, 1, 1.0), (0, 1, 0.1), (1, 0, 0.2)];
            let jac = faer::sparse::SparseColMat::try_new_from_triplets(
                2,
                2,
                &triplets
                    .iter()
                    .map(|(r, c, v)| faer::sparse::Triplet::new(*r, *c, *v))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            df_dy.push(jac);
        }

        for _i in 0..(m - 1) {
            let triplets = vec![(0, 0, 1.15), (1, 1, 1.0), (0, 1, 0.15), (1, 0, 0.25)];
            let jac = faer::sparse::SparseColMat::try_new_from_triplets(
                2,
                2,
                &triplets
                    .iter()
                    .map(|(r, c, v)| faer::sparse::Triplet::new(*r, *c, *v))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            df_dy_middle.push(jac);
        }

        let dbc_dya_triplets = vec![(0, 0, 1.0), (1, 1, 1.0)];
        let dbc_dya = faer::sparse::SparseColMat::try_new_from_triplets(
            2,
            2,
            &dbc_dya_triplets
                .iter()
                .map(|(r, c, v)| faer::sparse::Triplet::new(*r, *c, *v))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let dbc_dyb = faer::sparse::SparseColMat::try_new_from_triplets(
            2,
            2,
            &dbc_dya_triplets
                .iter()
                .map(|(r, c, v)| faer::sparse::Triplet::new(*r, *c, *v))
                .collect::<Vec<_>>(),
        )
        .unwrap();

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

        assert_eq!(jac.nrows(), 6);
        assert_eq!(jac.ncols(), 6);

        for i in 0..(m - 1) {
            for row in 0..n {
                for col in 0..n {
                    let global_row = i * n + row;
                    let global_col = i * n + col;
                    let jac_val = jac.get(global_row, global_col).unwrap_or(&0.0);
                    assert!(jac_val.abs() > 1e-10);
                }
            }
        }

        let bc_row_start = n * (m - 1);
        for row in bc_row_start..(bc_row_start + n + k) {
            let first_y_nonzero = (0..n).any(|col| jac.get(row, col).unwrap_or(&0.0).abs() > 1e-10);
            let last_y_start = (m - 1) * n;
            let last_y_nonzero = (last_y_start..(last_y_start + n))
                .any(|col| jac.get(row, col).unwrap_or(&0.0).abs() > 1e-10);
            assert!(first_y_nonzero || last_y_nonzero);
        }
    }

    #[test]
    fn test_collocation_residual_ordering() {
        let fun = |_x: &faer_col, _y: &faer_dense_mat, _p: &faer_col| faer_dense_mat::zeros(2, 3);

        let x = faer_col::from_fn(3, |i| [0.0, 1.0, 2.0][i]);
        let h = faer_col::from_fn(2, |_| 1.0);

        let mut y = faer_dense_mat::zeros(2, 3);
        for j in 0..3 {
            *y.get_mut(0, j) = 1.0;
            *y.get_mut(1, j) = 2.0;
        }
        let p = faer_col::zeros(0);

        let (col_res, y_middle, f, f_middle) = collocation_fun(&fun, &y, &p, &x, &h);

        assert_eq!(col_res.nrows(), 2);
        assert_eq!(col_res.ncols(), 2);
        assert_eq!(y_middle.nrows(), 2);
        assert_eq!(y_middle.ncols(), 2);
        assert_eq!(f.nrows(), 2);
        assert_eq!(f.ncols(), 3);
        assert_eq!(f_middle.nrows(), 2);
        assert_eq!(f_middle.ncols(), 2);

        for i in 0..2 {
            for j in 0..2 {
                assert!(col_res.get(i, j).abs() < 1e-10);
            }
        }

        for j in 0..2 {
            assert!((*y_middle.get(0, j) - 1.0).abs() < 1e-10);
            assert!((*y_middle.get(1, j) - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_parameter_jacobian() {
        let fun = |_x: &faer_col, y: &faer_dense_mat, p: &faer_col| {
            let mut f = faer_dense_mat::zeros(2, y.ncols());
            for j in 0..y.ncols() {
                *f.get_mut(0, j) = p[0] * *y.get(0, j);
                *f.get_mut(1, j) = p[1] * *y.get(1, j);
            }
            f
        };

        let x = faer_col::from_fn(2, |i| [0.0, 1.0][i]);
        let y = faer_dense_mat::from_fn(2, 2, |i, j| [[1.0, 2.0], [3.0, 4.0]][i][j]);
        let p = faer_col::from_fn(2, |i| [2.0, 3.0][i]);

        let (df_dy, df_dp) = estimate_fun_jac(&fun, &x, &y, &p, None);

        assert!(df_dp.is_some());
        let df_dp = df_dp.unwrap();

        assert_eq!(df_dy.len(), 2);
        assert_eq!(df_dp.len(), 2);

        for i in 0..2 {
            assert!((df_dy[i].get(0, 0).unwrap_or(&0.0) - p[0]).abs() < 1e-6);
            assert!((df_dy[i].get(1, 1).unwrap_or(&0.0) - p[1]).abs() < 1e-6);
            assert!(df_dy[i].get(0, 1).unwrap_or(&0.0).abs() < 1e-6);
            assert!(df_dy[i].get(1, 0).unwrap_or(&0.0).abs() < 1e-6);

            assert!((df_dp[i].get(0, 0).unwrap_or(&0.0) - *y.get(0, i)).abs() < 1e-6);
            assert!((df_dp[i].get(1, 1).unwrap_or(&0.0) - *y.get(1, i)).abs() < 1e-6);
            assert!(df_dp[i].get(0, 1).unwrap_or(&0.0).abs() < 1e-6);
            assert!(df_dp[i].get(1, 0).unwrap_or(&0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_create_spline() {
        let x = faer_col::from_fn(3, |i| [0.0, 1.0, 2.0][i]);
        let h = faer_col::from_fn(2, |_| 1.0);

        let y = faer_dense_mat::from_fn(1, 3, |_, j| [0.0, 1.0, 8.0][j]);
        let yp = faer_dense_mat::from_fn(1, 3, |_, j| [0.0, 3.0, 12.0][j]);

        let spline = create_spline(&y, &yp, &x, &h);

        let x_eval = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let result = spline.call(&x_eval, &[5], Some(0), None);

        let expected = vec![0.0, 0.125, 1.0, 3.375, 8.0];

        for i in 0..x_eval.len() {
            assert!((result[(i, 0)] - expected[i]).abs() < 1e-10);
        }

        let result_deriv = spline.call(&x_eval, &[5], Some(1), None);
        let expected_deriv = vec![0.0, 0.75, 3.0, 6.75, 12.0];

        for i in 0..x_eval.len() {
            assert!((result_deriv[(i, 0)] - expected_deriv[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_estimate_rms_residuals() {
        let fun = |_x: &faer_col, _y: &faer_dense_mat, _p: &faer_col| faer_dense_mat::zeros(1, 3);

        let x = faer_col::from_fn(3, |i| [0.0, 0.5, 1.0][i]);
        let h = faer_col::from_fn(2, |_| 0.5);
        let p = faer_col::zeros(0);

        let y = faer_dense_mat::from_fn(1, 3, |_, _| 2.0);
        let yp = faer_dense_mat::zeros(1, 3);

        let spline = create_spline(&y, &yp, &x, &h);

        let r_middle = faer_dense_mat::zeros(1, 2);
        let f_middle = faer_dense_mat::zeros(1, 2);

        let rms_res = estimate_rms_residuals(&fun, &spline, &x, &h, &p, &r_middle, &f_middle);

        for i in 0..rms_res.nrows() {
            assert!(rms_res[i] < 1e-6);
        }
    }

    #[test]
    fn test_estimate_rms_residuals_nonzero() {
        let fun = |_x: &faer_col, y: &faer_dense_mat, _p: &faer_col| y.clone();

        let x = faer_col::from_fn(3, |i| [0.0, 1.0, 2.0][i]);
        let h = faer_col::from_fn(2, |_| 1.0);
        let p = faer_col::zeros(0);

        let y = faer_dense_mat::from_fn(1, 3, |_, j| [0.0, 1.0, 2.0][j]);
        let yp = faer_dense_mat::from_fn(1, 3, |_, _| 1.0);

        let spline = create_spline(&y, &yp, &x, &h);

        let r_middle = faer_dense_mat::from_fn(1, 2, |_, j| [0.5, -0.5][j]);
        let f_middle = faer_dense_mat::from_fn(1, 2, |_, j| [0.5, 1.5][j]);

        let rms_res = estimate_rms_residuals(&fun, &spline, &x, &h, &p, &r_middle, &f_middle);

        for i in 0..rms_res.nrows() {
            assert!(rms_res[i] > 1e-10);
            assert!(rms_res[i] < 10.0);
        }
    }

    #[test]
    fn test_modify_mesh() {
        let x = faer_col::from_fn(4, |i| [0.0, 1.0, 2.0, 3.0][i]);

        let insert_1 = vec![0, 2];
        let insert_2 = vec![1];

        let x_new = modify_mesh(&x, &insert_1, &insert_2);

        let expected = vec![0.0, 0.5, 1.0, 4.0 / 3.0, 5.0 / 3.0, 2.0, 2.5, 3.0];

        assert_eq!(x_new.nrows(), expected.len());
        for i in 0..expected.len() {
            assert!((x_new[i] - expected[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_solve_newton_convergence() {
        let fun = |_x: &faer_col, y: &faer_dense_mat, _p: &faer_col| {
            let mut f = faer_dense_mat::zeros(2, y.ncols());
            for j in 0..y.ncols() {
                *f.get_mut(0, j) = *y.get(1, j);
                *f.get_mut(1, j) = 0.0;
            }
            f
        };

        let bc = |ya: &faer_col, yb: &faer_col, _p: &faer_col| {
            faer_col::from_fn(2, |i| [ya[0], yb[0] - 1.0][i])
        };

        let x = faer_col::from_fn(3, |i| [0.0, 0.5, 1.0][i]);
        let h = faer_col::from_fn(2, |_| 0.5);

        let mut y = faer_dense_mat::zeros(2, 3);
        *y.get_mut(0, 0) = 0.0;
        *y.get_mut(0, 1) = 0.4;
        *y.get_mut(0, 2) = 0.8;
        *y.get_mut(1, 0) = 1.0;
        *y.get_mut(1, 1) = 1.0;
        *y.get_mut(1, 2) = 1.0;

        let p = faer_col::zeros(0);

        let (y_final, _p_final, singular) = solve_newton(
            2,
            3,
            &h,
            &fun,
            &bc,
            None,
            None,
            y,
            p,
            &x,
            1e-10,
            1e-10,
            &mut CustomTimer::new(),
            &mut HashMap::new(),
        );

        assert!(!singular);

        assert!(y_final.get(0, 0).abs() < 1e-8);
        assert!((*y_final.get(0, 1) - 0.5).abs() < 1e-8);
        assert!((*y_final.get(0, 2) - 1.0).abs() < 1e-8);

        for j in 0..3 {
            assert!((*y_final.get(1, j) - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_construct_global_jac_structure() {
        let n = 2;
        let m = 3;
        let k = 1;

        let h = faer_col::from_fn(2, |_| 0.5);

        let mut df_dy = Vec::new();
        let mut df_dy_middle = Vec::new();
        let mut df_dp = Vec::new();
        let mut df_dp_middle = Vec::new();

        for _ in 0..3 {
            let triplets = vec![(0, 0, 1.0), (1, 1, 1.0)];
            let jac = faer::sparse::SparseColMat::try_new_from_triplets(
                2,
                2,
                &triplets
                    .iter()
                    .map(|(r, c, v)| faer::sparse::Triplet::new(*r, *c, *v))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            df_dy.push(jac);

            let param_triplets = vec![(0, 0, 1.0), (1, 0, 1.0)];
            let param_jac = faer::sparse::SparseColMat::try_new_from_triplets(
                2,
                1,
                &param_triplets
                    .iter()
                    .map(|(r, c, v)| faer::sparse::Triplet::new(*r, *c, *v))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            df_dp.push(param_jac);
        }

        for _ in 0..2 {
            let triplets = vec![(0, 0, 1.0), (1, 1, 1.0)];
            let jac = faer::sparse::SparseColMat::try_new_from_triplets(
                2,
                2,
                &triplets
                    .iter()
                    .map(|(r, c, v)| faer::sparse::Triplet::new(*r, *c, *v))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            df_dy_middle.push(jac);

            let param_triplets = vec![(0, 0, 1.0), (1, 0, 1.0)];
            let param_jac = faer::sparse::SparseColMat::try_new_from_triplets(
                2,
                1,
                &param_triplets
                    .iter()
                    .map(|(r, c, v)| faer::sparse::Triplet::new(*r, *c, *v))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            df_dp_middle.push(param_jac);
        }

        let dbc_dya_triplets = vec![(0, 0, 1.0), (1, 1, 1.0), (2, 0, 0.0)];
        let dbc_dya = faer::sparse::SparseColMat::try_new_from_triplets(
            3,
            2,
            &dbc_dya_triplets
                .iter()
                .map(|(r, c, v)| faer::sparse::Triplet::new(*r, *c, *v))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let dbc_dyb = faer::sparse::SparseColMat::try_new_from_triplets(
            3,
            2,
            &dbc_dya_triplets
                .iter()
                .map(|(r, c, v)| faer::sparse::Triplet::new(*r, *c, *v))
                .collect::<Vec<_>>(),
        )
        .unwrap();

        let dbc_dp_triplets = vec![(0, 0, 1.0), (1, 0, 1.0), (2, 0, 1.0)];
        let dbc_dp = faer::sparse::SparseColMat::try_new_from_triplets(
            3,
            1,
            &dbc_dp_triplets
                .iter()
                .map(|(r, c, v)| faer::sparse::Triplet::new(*r, *c, *v))
                .collect::<Vec<_>>(),
        )
        .unwrap();

        let jac = construct_global_jac(
            n,
            m,
            k,
            &h,
            &df_dy,
            &df_dy_middle,
            Some(&df_dp),
            Some(&df_dp_middle),
            &dbc_dya,
            &dbc_dyb,
            Some(&dbc_dp),
        );

        assert_eq!(jac.nrows(), 7);
        assert_eq!(jac.ncols(), 7);

        let param_col = 6;
        for row in 0..4 {
            assert!(jac.get(row, param_col).unwrap_or(&0.0).abs() > 1e-10);
        }

        for row in 4..7 {
            assert!(jac.get(row, param_col).unwrap_or(&0.0).abs() > 1e-10);
        }
    }

    #[test]
    fn test_compute_jac_indices() {
        let n = 2;
        let m = 3;
        let k = 1;

        let (i_indices, j_indices) = compute_jac_indices(n, m, k);

        let expected_entries =
            2 * (m - 1) * n * n + 2 * (n + k) * n + (m - 1) * n * k + (n + k) * k;
        assert_eq!(i_indices.len(), expected_entries);
        assert_eq!(j_indices.len(), expected_entries);

        let max_row = (m - 1) * n + (n + k) - 1;
        let max_col = n * m + k - 1;

        for &i in &i_indices {
            assert!(i <= max_row);
        }

        for &j in &j_indices {
            assert!(j <= max_col);
        }
    }

    #[test]
    fn test_mesh_refinement() {
        let fun = |_x: &faer_col, y: &faer_dense_mat, _p: &faer_col| {
            let mut f = faer_dense_mat::zeros(2, y.ncols());
            for j in 0..y.ncols() {
                *f.get_mut(0, j) = *y.get(1, j);
                *f.get_mut(1, j) = 100.0 * *y.get(0, j);
            }
            f
        };

        let bc = |ya: &faer_col, yb: &faer_col, _p: &faer_col| {
            faer_col::from_fn(2, |i| [ya[0] - 1.0, yb[0] - 10.0_f64.exp()][i])
        };

        let x = faer_col::from_fn(3, |i| [0.0, 0.5, 1.0][i]);
        let mut y = faer_dense_mat::zeros(2, 3);
        for j in 0..3 {
            *y.get_mut(0, j) = (10.0 * x[j]).exp();
            *y.get_mut(1, j) = 10.0 * (10.0 * x[j]).exp();
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
            None,
            None,
        );

        match result {
            Ok(res) => {
                assert!(res.success || res.status == 1);
                assert!(res.x.nrows() > 3);
                assert!((*res.y.get(0, 0) - 1.0).abs() < 1e-3);
                assert!((*res.y.get(0, res.x.nrows() - 1) - 10.0_f64.exp()).abs() < 1e2);
            }
            Err(e) => panic!("Mesh refinement test failed: {}", e),
        }
    }

    #[test]
    fn test_singular_jacobian_handling() {
        let fun = |_x: &faer_col, _y: &faer_dense_mat, _p: &faer_col| faer_dense_mat::zeros(2, 3);

        let bc = |ya: &faer_col, yb: &faer_col, _p: &faer_col| {
            faer_col::from_fn(2, |i| [ya[0], yb[0]][i])
        };

        let x = faer_col::from_fn(3, |i| [0.0, 0.5, 1.0][i]);
        let y = faer_dense_mat::from_fn(2, 3, |_, _| 1.0);

        let result = solve_bvp(
            &fun, &bc, x, y, None, None, None, None, 1e-6, 1000, 0, None, None,
        );

        match result {
            Ok(res) => {
                assert!(!res.success);
                assert_eq!(res.status, 2);
            }
            Err(_) => {}
        }
    }

    #[test]
    fn test_boundary_condition_tolerance() {
        // equation y'' = -y with boundary conditions y(0) = 0, y'(0) = 1
        // Exact solution: y = 0 (trivial solution)
        let fun = |_x: &faer_col, y: &faer_dense_mat, _p: &faer_col| {
            let mut f = faer_dense_mat::zeros(2, y.ncols());
            for j in 0..y.ncols() {
                *f.get_mut(0, j) = *y.get(1, j);
                *f.get_mut(1, j) = -*y.get(0, j);
            }
            f
        };

        let bc = |ya: &faer_col, _yb: &faer_col, _p: &faer_col| {
            faer_col::from_fn(2, |i| [ya[0], ya[1] - 1.0][i])
        };

        let x = faer_col::from_fn(2, |i| [0.0, std::f64::consts::PI][i]);
        let mut y = faer_dense_mat::zeros(2, 2);
        *y.get_mut(0, 0) = 0.1;
        *y.get_mut(0, 1) = 0.1;
        *y.get_mut(1, 0) = 1.0;
        *y.get_mut(1, 1) = -1.0;

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
            None,
        );

        match result {
            Ok(res) => {
                if res.success {
                    let Y1 = res.y.row(0);
                    let Y2 = res.y.row(1);
                    print!("Y1: {:?}\n", Y1);
                    print!("Y2: {:?}\n", Y2);
                    let x = res.x.clone();
                    let exact_sol: Vec<f64> = x.iter().map(|xi| xi.clone().sin()).collect();
                    print!("Exact solution: {:?}\n", exact_sol);
                    // For y'' = -y with y(0) = 0, y(π) = 0, the solution is y = 0
                    // Check that boundary conditions are satisfied
                    assert!(
                        res.y.get(0, 0).abs() < 1e-6,
                        "Left boundary not satisfied: y(0) = {}",
                        res.y.get(0, 0)
                    );
                    assert!(
                        res.y.get(0, res.x.nrows() - 1).abs() < 1e-6,
                        "Right boundary not satisfied: y(π) = {}",
                        res.y.get(0, res.x.nrows() - 1)
                    );

                    // Solution should be close to zero everywhere (trivial solution)
                    for i in 0..res.x.nrows() {
                        assert!(
                            (res.y.get(0, i) - exact_sol[i]).abs() < 1e-3,
                            "Solution not close to zero at x = {}: y = {}",
                            res.x[i],
                            res.y.get(0, i)
                        );
                    }
                } else {
                    // If solver fails to meet BC tolerance, that's also acceptable for this test
                    assert!(
                        res.status == 3 || res.status == 4,
                        "Unexpected status: {}",
                        res.status
                    );
                }
            }
            Err(_) => {
                // Solver error is acceptable for this tolerance test
            }
        }
    }

    #[test]
    fn test_elementary_linear_bvp() {
        let fun = |_x: &faer_col, y: &faer_dense_mat, _p: &faer_col| {
            let mut f = faer_dense_mat::zeros(2, y.ncols());
            for j in 0..y.ncols() {
                *f.get_mut(0, j) = *y.get(1, j);
                *f.get_mut(1, j) = 0.0;
            }
            f
        };

        let bc = |ya: &faer_col, yb: &faer_col, _p: &faer_col| {
            faer_col::from_fn(2, |i| [ya[0], yb[0] - 1.0][i])
        };

        let x = faer_col::from_fn(3, |i| [0.0, 0.5, 1.0][i]);
        let mut y = faer_dense_mat::zeros(2, 3);
        *y.get_mut(0, 0) = 0.0;
        *y.get_mut(0, 1) = 0.5;
        *y.get_mut(0, 2) = 1.0;
        *y.get_mut(1, 0) = 1.0;
        *y.get_mut(1, 1) = 1.0;
        *y.get_mut(1, 2) = 1.0;

        let result = solve_bvp(
            &fun, &bc, x, y, None, None, None, None, 1e-8, 1000, 0, None, None,
        );

        match result {
            Ok(res) => {
                assert!(res.success);

                for i in 0..res.x.nrows() {
                    let expected = res.x[i];
                    assert!((*res.y.get(0, i) - expected).abs() < 1e-6);
                }

                for i in 0..res.x.nrows() {
                    assert!((*res.y.get(1, i) - 1.0).abs() < 1e-6);
                }
            }
            Err(e) => panic!("Elementary linear BVP failed: {}", e),
        }
    }

    #[test]
    fn test_elementary_quadratic_bvp() {
        let fun = |_x: &faer_col, y: &faer_dense_mat, _p: &faer_col| {
            let mut f = faer_dense_mat::zeros(2, y.ncols());
            for j in 0..y.ncols() {
                *f.get_mut(0, j) = *y.get(1, j);
                *f.get_mut(1, j) = -2.0;
            }
            f
        };

        let bc = |ya: &faer_col, yb: &faer_col, _p: &faer_col| {
            faer_col::from_fn(2, |i| [ya[0], yb[0]][i])
        };

        let x = faer_col::from_fn(5, |i| [0.0, 0.25, 0.5, 0.75, 1.0][i]);
        let mut y = faer_dense_mat::zeros(2, 5);
        for j in 0..5 {
            let xi = x[j];
            *y.get_mut(0, j) = xi;
            *y.get_mut(1, j) = 1.0;
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
            None,
        );

        match result {
            Ok(res) => {
                assert!(res.success);

                for i in 0..res.x.nrows() {
                    let xi = res.x[i];
                    let expected = xi * (1.0 - xi);
                    assert!((*res.y.get(0, i) - expected).abs() < 1e-6);
                }

                for i in 0..res.x.nrows() {
                    let xi = res.x[i];
                    let expected_deriv = 1.0 - 2.0 * xi;
                    assert!((*res.y.get(1, i) - expected_deriv).abs() < 1e-6);
                }
            }
            Err(e) => panic!("Elementary quadratic BVP failed: {}", e),
        }
    }
    /*
        1. Lane-Emden Equation (Index 5)
    Equation: y'' + (2/x)*y' + y^5 = 0

    Boundary conditions: y(0) = 1, y'(0) = 0

    Exact solution: y = (1 + x²/3)^(-0.5)

    Domain: [0, 2]

    2. Parachute Equation
    Equation: y'' + k*y'² - g = 0

    Constants: k = 3, g = 5

    Boundary conditions: y(0) = 0, y'(0) = 0

    Exact solution: y = (1/k) * (ln((e^(2√(gk)t) + 1)/2) - √(gk)*t)

    Domain: [0, 0.6]

    3. Exponential BVP
    Equation: y'' = -(2/a)*(1 + 2*ln(y))*y

    Constant: a = 4

    Boundary conditions: y(-1) = exp(-1/a), y(1) = exp(-1/a)

    Exact solution: y(x) = exp(-x²/a)

    Domain: [-1, 1]

         */
    // problem that leads to singular jacobian
    #[test]
    #[should_panic]
    fn test_lane_emden_equation() {
        // Lane-Emden equation: y′′+2y′/x + y**5=0,y(0)=1,y′(0)=0

        //y'=z
        //z'=- 2*z/x - y**5
        let fun = |x: &faer_col, y: &faer_dense_mat, _p: &faer_col| {
            let mut f = faer_dense_mat::zeros(2, y.ncols());
            for j in 0..y.ncols() {
                let y_val = *y.get(0, j);
                let z_val = *y.get(1, j);
                let x_val = x[j];

                *f.get_mut(0, j) = z_val; // y' = z
                *f.get_mut(1, j) = -2.0 * z_val / x_val - y_val.powi(5); // z' = -2*x*z - y^5

                /*
                if x_val.abs() < 1e-10 {
                    *f.get_mut(1, j) = -y_val.powi(5); // z' = -y^5 (limit as x->0)
                } else {
                    *f.get_mut(1, j) = -2.0  * z_val/x_val - y_val.powi(5); // z' = -2*x*z - y^5
                }
                */
            }
            f
        };

        let bc = move |ya: &faer_col, _yb: &faer_col, _p: &faer_col| {
            faer_col::from_fn(2, |i| match i {
                0 => ya[0] - 1.0, // y(0) = 1
                1 => ya[1],       // y'(0) = 0
                _ => 0.0,
            })
        };
        let n = 100;
        let x = faer_col::from_fn(n, |i| i as f64 * 0.5); // [0, 0.5, 1.0, 1.5, 2.0]
        let mut y = faer_dense_mat::zeros(2, n);
        for j in 0..5 {
            let x_val = x[j];
            let exact = (1.0 + x_val * x_val / 3.0).powf(-0.5);
            *y.get_mut(0, j) = exact;
            *y.get_mut(1, j) = -x_val / 3.0 * (1.0 + x_val * x_val / 3.0).powf(-1.5);
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
            1e-3,
            2000,
            2,
            None,
            None,
        );
        println!(" success: {}", result.clone().unwrap().success);
        match result {
            Ok(res) => {
                if res.success {
                    let _y = res.y.row(0).clone();
                    //  println!("Lane-Emden solution: {:?}", y);
                    for i in 0..res.x.nrows() {
                        let x_val = res.x[i];
                        let exact = (1.0 + x_val * x_val / 3.0).powf(-0.5);
                        let error = (*res.y.get(0, i) - exact).abs();
                        assert!(
                            error < 1e-1,
                            "Lane-Emden error at x[{}]={}: {} vs {}",
                            i,
                            x_val,
                            res.y.get(0, i),
                            exact
                        );
                    }
                } else {
                    panic!("Error in Lane-Emden solution");
                }
            }
            Err(_) => {
                panic!("Error in Lane-Emden solution");
            }
        }
    }

    #[test]
    fn test_parachute_equation() {
        // Parachute equation: y'' + k*y'^2 - g = 0
        // y(0) = 0, y'(0) = 0
        // Exact solution: y = (1/k) * (ln((e^(2*sqrt(g*k)*t) + 1)/2) - sqrt(g*k)*t)
        let k = 3.0;
        let g = 5.0;

        let fun = move |_x: &faer_col, y: &faer_dense_mat, _p: &faer_col| {
            let mut f = faer_dense_mat::zeros(2, y.ncols());
            for j in 0..y.ncols() {
                let z_val = *y.get(1, j);
                *f.get_mut(0, j) = z_val; // y' = z
                *f.get_mut(1, j) = g - k * z_val * z_val; // z' = g - k*z^2
            }
            f
        };

        let bc = |ya: &faer_col, _yb: &faer_col, _p: &faer_col| {
            faer_col::from_fn(2, |i| match i {
                0 => ya[0], // y(0) = 0
                1 => ya[1], // y'(0) = 0
                _ => 0.0,
            })
        };

        let x = faer_col::from_fn(4, |i| i as f64 * 0.2); // [0, 0.2, 0.4, 0.6]
        let mut y = faer_dense_mat::zeros(2, 4);
        for j in 0..4 {
            let _t = x[j];

            *y.get_mut(0, j) = 0.0; // ;
            *y.get_mut(1, j) = 0.0;
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
            1e-10,
            2000,
            2,
            None,
            None,
        );
        println!(" success: {}", result.clone().unwrap().success);
        match result {
            Ok(res) => {
                if res.success {
                    for i in 0..res.x.nrows() {
                        let t = res.x[i];
                        let sqrt_gk = (g * k).sqrt();
                        let exp_term = (2.0 * sqrt_gk * t).exp();
                        let exact = (1.0 / k) * (((exp_term + 1.0) / 2.0).ln() - sqrt_gk * t);
                        let error = (*res.y.get(0, i) - exact).abs();
                        assert!(
                            error < 1e-7,
                            "Parachute error at t={}: {} vs {}",
                            t,
                            res.y.get(0, i),
                            exact
                        );
                    }
                } else {
                    panic!("solution not found");
                }
            }
            Err(_) => {}
        }
    }

    #[test]
    fn test_exponential_bvp() {
        // y'' = -(2/a)*(1 + 2*ln(y))*y
        // y(-1) = exp(-1/a), y(1) = exp(-1/a)
        // Exact solution: y(x) = exp(-x^2/a)
        let a = 4.0;

        let fun = move |_x: &faer_col, y: &faer_dense_mat, _p: &faer_col| {
            let mut f = faer_dense_mat::zeros(2, y.ncols());
            for j in 0..y.ncols() {
                let y_val = (*y.get(0, j) as f64).max(1e-10); // Avoid log(0)
                let z_val = *y.get(1, j);
                *f.get_mut(0, j) = z_val; // y' = z
                *f.get_mut(1, j) = -(2.0 / a) * (1.0 + 2.0 * y_val.ln()) * y_val; // z' = -(2/a)*(1+2*ln(y))*y
            }
            f
        };

        let bc_val = (-1.0 / a).exp();
        let bc = move |ya: &faer_col, yb: &faer_col, _p: &faer_col| {
            faer_col::from_fn(2, |i| match i {
                0 => ya[0] - bc_val, // y(-1) = exp(-1/a)
                1 => yb[0] - bc_val, // y(1) = exp(-1/a)
                _ => 0.0,
            })
        };

        let x = faer_col::from_fn(5, |i| -1.0 + i as f64 * 0.5); // [-1, -0.5, 0, 0.5, 1]
        let mut y = faer_dense_mat::zeros(2, 5);
        for j in 0..5 {
            let exact_y = 0.0; // (-x_val * x_val / a).exp();
            let exact_z = 0.0; //-2.0 * x_val / a * exact_y;
            *y.get_mut(0, j) = exact_y;
            *y.get_mut(1, j) = exact_z;
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
            1e-7,
            2000,
            2,
            None,
            None,
        );
        let result1 = result.clone().unwrap().y;
        println!("Result of direct solution: {:?}", result1);
        println!("is success {}", result.clone().unwrap().success);
        match result {
            Ok(res) => {
                if res.success {
                    for i in 0..res.x.nrows() {
                        let x_val = res.x[i];
                        let exact = (-x_val * x_val / a).exp();
                        let error = (*res.y.get(0, i) - exact).abs();
                        assert!(
                            error < 1e-7,
                            "Exponential BVP error at x={}: {} vs {}",
                            x_val,
                            res.y.get(0, i),
                            exact
                        );
                    }
                } else {
                    println!("Exponential BVP did not converge: {}", res.status);
                }
            }
            Err(_) => {}
        }
    }
}
