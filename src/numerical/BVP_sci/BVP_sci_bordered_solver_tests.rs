#[cfg(test)]
mod tests {
    use crate::numerical::BVP_sci::BVP_sci_bordered_solver::{
        extract_bordered_banded_blocks, factor_bordered_banded_structured,
        solve_bordered_banded_reference, solve_bordered_banded_structured,
        BvpSciBorderedBandedBlocks,
    };
    use crate::numerical::BVP_sci::BVP_sci_faer::{
        construct_global_jac, faer_col, faer_dense_mat, faer_mat,
    };
    use faer::linalg::solvers::Solve;
    use faer::sparse::{SparseColMat, Triplet};

    fn dense_sparse(rows: usize, cols: usize, values: &[(usize, usize, f64)]) -> faer_mat {
        let triplets = values
            .iter()
            .map(|&(row, col, value)| Triplet::new(row, col, value))
            .collect::<Vec<_>>();
        SparseColMat::try_new_from_triplets(rows, cols, &triplets).unwrap()
    }

    fn scalar_sparse(value: f64) -> faer_mat {
        dense_sparse(1, 1, &[(0, 0, value)])
    }

    #[test]
    fn bordered_banded_extraction_reconstructs_parameter_free_global_jacobian() {
        let n = 2;
        let m = 3;
        let k = 0;
        let h = faer_col::from_fn(m - 1, |_| 0.5);
        let df_dy = vec![dense_sparse(2, 2, &[(0, 0, 2.0), (1, 1, -1.0)]); m];
        let df_dy_middle = vec![dense_sparse(2, 2, &[(0, 1, 0.25), (1, 0, -0.5)]); m - 1];
        let dbc_dya = dense_sparse(2, 2, &[(0, 0, 1.0)]);
        let dbc_dyb = dense_sparse(2, 2, &[(1, 0, 1.0)]);

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
        let blocks = extract_bordered_banded_blocks(&jac, n, m, k).unwrap();

        assert_eq!(blocks.diagonal_blocks.len(), m - 1);
        assert_eq!(blocks.offdiag_blocks.len(), m - 1);
        assert_eq!(blocks.boundary_left.nrows(), n);
        assert_eq!(blocks.boundary_left.ncols(), n);
        assert_eq!(blocks.boundary_right.nrows(), n);
        assert_eq!(blocks.boundary_right.ncols(), n);
        assert!(blocks.collocation_parameter_blocks.is_none());
        assert!(blocks.boundary_parameters.is_none());
        assert!(blocks.max_abs_diff_against_sparse(&jac).unwrap() < 1e-14);
    }

    #[test]
    fn bordered_banded_extraction_preserves_parameter_blocks() {
        let n = 1;
        let m = 4;
        let k = 1;
        let h = faer_col::from_fn(m - 1, |_| 0.25);
        let df_dy = vec![scalar_sparse(0.5); m];
        let df_dy_middle = vec![scalar_sparse(0.75); m - 1];
        let df_dp = Some(vec![scalar_sparse(2.0); m]);
        let df_dp_middle = Some(vec![scalar_sparse(3.0); m - 1]);
        let dbc_dya = dense_sparse(2, 1, &[(0, 0, 1.0)]);
        let dbc_dyb = dense_sparse(2, 1, &[(1, 0, -1.0)]);
        let dbc_dp = Some(dense_sparse(2, 1, &[(1, 0, 4.0)]));

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
        let blocks = extract_bordered_banded_blocks(&jac, n, m, k).unwrap();

        let collocation_param_blocks = blocks.collocation_parameter_blocks.as_ref().unwrap();
        assert_eq!(collocation_param_blocks.len(), m - 1);
        assert_eq!(collocation_param_blocks[0].nrows(), n);
        assert_eq!(collocation_param_blocks[0].ncols(), k);
        let boundary_parameters = blocks.boundary_parameters.as_ref().unwrap();
        assert_eq!(boundary_parameters.nrows(), n + k);
        assert_eq!(boundary_parameters.ncols(), k);
        assert!(blocks.max_abs_diff_against_sparse(&jac).unwrap() < 1e-14);
    }

    #[test]
    fn bordered_banded_reference_solve_matches_sparse_lu_parameter_free() {
        let n = 2;
        let m = 3;
        let k = 0;
        let h = faer_col::from_fn(m - 1, |_| 0.5);
        let df_dy = vec![dense_sparse(2, 2, &[(0, 0, 1.0), (1, 1, -0.25)]); m];
        let df_dy_middle = vec![dense_sparse(2, 2, &[(0, 1, 0.1), (1, 0, -0.2)]); m - 1];
        let dbc_dya = dense_sparse(2, 2, &[(0, 0, 1.0), (1, 1, 0.5)]);
        let dbc_dyb = dense_sparse(2, 2, &[(0, 0, -0.3), (1, 1, 1.0)]);

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

        assert_reference_solve_matches_sparse(&jac, n, m, k);
    }

    #[test]
    fn bordered_banded_reference_solve_matches_sparse_lu_with_parameter() {
        let n = 1;
        let m = 4;
        let k = 1;
        let h = faer_col::from_fn(m - 1, |_| 0.25);
        let df_dy = vec![scalar_sparse(0.5); m];
        let df_dy_middle = vec![scalar_sparse(0.75); m - 1];
        let df_dp = Some(vec![scalar_sparse(2.0); m]);
        let df_dp_middle = Some(vec![scalar_sparse(3.0); m - 1]);
        let dbc_dya = dense_sparse(2, 1, &[(0, 0, 1.0)]);
        let dbc_dyb = dense_sparse(2, 1, &[(0, 0, 0.5), (1, 0, -1.0)]);
        let dbc_dp = Some(dense_sparse(2, 1, &[(1, 0, 4.0)]));

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

        assert_reference_solve_matches_sparse(&jac, n, m, k);
    }

    #[test]
    fn bordered_banded_structured_solve_matches_sparse_lu_parameter_free() {
        let n = 2;
        let m = 4;
        let k = 0;
        let h = faer_col::from_fn(m - 1, |_| 0.25);
        let df_dy = vec![dense_sparse(2, 2, &[(0, 0, 0.8), (1, 1, -0.4)]); m];
        let df_dy_middle = vec![dense_sparse(2, 2, &[(0, 1, 0.15), (1, 0, -0.05)]); m - 1];
        let dbc_dya = dense_sparse(2, 2, &[(0, 0, 1.0), (1, 1, 0.25)]);
        let dbc_dyb = dense_sparse(2, 2, &[(0, 0, 0.2), (1, 1, -1.0)]);

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

        assert_structured_solve_matches_sparse(&jac, n, m, k);
    }

    #[test]
    fn bordered_banded_structured_solve_matches_sparse_lu_with_parameter() {
        let n = 1;
        let m = 5;
        let k = 1;
        let h = faer_col::from_fn(m - 1, |_| 0.2);
        let df_dy = vec![scalar_sparse(0.4); m];
        let df_dy_middle = vec![scalar_sparse(0.6); m - 1];
        let df_dp = Some(vec![scalar_sparse(1.5); m]);
        let df_dp_middle = Some(vec![scalar_sparse(2.5); m - 1]);
        let dbc_dya = dense_sparse(2, 1, &[(0, 0, 1.0)]);
        let dbc_dyb = dense_sparse(2, 1, &[(0, 0, 0.2), (1, 0, -1.0)]);
        let dbc_dp = Some(dense_sparse(2, 1, &[(1, 0, 3.0)]));

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

        assert_structured_solve_matches_sparse(&jac, n, m, k);
    }

    #[test]
    fn bordered_banded_structured_factorization_reuses_multiple_rhs() {
        let n = 2;
        let m = 4;
        let k = 0;
        let h = faer_col::from_fn(m - 1, |_| 0.25);
        let df_dy = vec![dense_sparse(2, 2, &[(0, 0, 0.8), (1, 1, -0.4)]); m];
        let df_dy_middle = vec![dense_sparse(2, 2, &[(0, 1, 0.15), (1, 0, -0.05)]); m - 1];
        let dbc_dya = dense_sparse(2, 2, &[(0, 0, 1.0), (1, 1, 0.25)]);
        let dbc_dyb = dense_sparse(2, 2, &[(0, 0, 0.2), (1, 1, -1.0)]);

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
        let blocks = extract_bordered_banded_blocks(&jac, n, m, k).unwrap();
        let factorization = factor_bordered_banded_structured(&blocks).unwrap();
        let sparse_lu = jac.sp_lu().unwrap();

        for rhs_shift in [0.0, 0.25, -0.5] {
            let rhs = faer_col::from_fn(blocks.total_size(), |row| {
                ((row + 5) as f64 + rhs_shift).cos() - 0.05 * (row as f64)
            });
            let cached = factorization.solve(&rhs).unwrap();
            let one_shot = solve_bordered_banded_structured(&blocks, &rhs).unwrap();
            let sparse_solution = sparse_lu.solve(rhs.as_mat());
            let sparse =
                faer_col::from_fn(sparse_solution.nrows(), |row| *sparse_solution.get(row, 0));

            let max_cached_vs_one_shot = (0..cached.nrows())
                .map(|row| (cached[row] - one_shot[row]).abs())
                .fold(0.0_f64, f64::max);
            let max_cached_vs_sparse = (0..cached.nrows())
                .map(|row| (cached[row] - sparse[row]).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                max_cached_vs_one_shot < 1e-12,
                "cached bordered solve must match one-shot structured solve, max_diff={max_cached_vs_one_shot:e}"
            );
            assert!(
                max_cached_vs_sparse < 1e-10,
                "cached bordered solve must match sparse LU, max_diff={max_cached_vs_sparse:e}"
            );
        }
    }

    #[test]
    fn bordered_banded_structured_solve_rejects_wrong_rhs_length() {
        let blocks = simple_valid_blocks_without_parameters();
        let rhs = faer_col::zeros(blocks.total_size() - 1);
        let err = solve_bordered_banded_structured(&blocks, &rhs).unwrap_err();
        assert!(
            err.contains("expected rhs length"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn bordered_banded_structured_solve_rejects_malformed_block_layout() {
        let mut blocks = simple_valid_blocks_without_parameters();
        blocks.offdiag_blocks.pop();
        let rhs = faer_col::zeros(blocks.total_size());
        let err = solve_bordered_banded_structured(&blocks, &rhs).unwrap_err();
        assert!(
            err.contains("expected 2 offdiag blocks"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn bordered_banded_structured_solve_reports_singular_offdiag_block() {
        let mut blocks = simple_valid_blocks_without_parameters();
        blocks.offdiag_blocks[0] = faer_dense_mat::zeros(1, 1);
        let rhs = faer_col::from_fn(blocks.total_size(), |row| row as f64 + 1.0);
        let err = solve_bordered_banded_structured(&blocks, &rhs).unwrap_err();
        assert!(
            err.contains("singular offdiag block at interval 0"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn bordered_banded_structured_solve_reports_singular_border_system() {
        let mut blocks = simple_valid_blocks_without_parameters();
        blocks.boundary_left = faer_dense_mat::zeros(1, 1);
        blocks.boundary_right = faer_dense_mat::zeros(1, 1);
        let rhs = faer_col::from_fn(blocks.total_size(), |row| row as f64 + 1.0);
        let err = solve_bordered_banded_structured(&blocks, &rhs).unwrap_err();
        assert!(
            err.contains("singular border system"),
            "unexpected error: {err}"
        );
    }

    fn assert_reference_solve_matches_sparse(jac: &faer_mat, n: usize, m: usize, k: usize) {
        let blocks = extract_bordered_banded_blocks(jac, n, m, k).unwrap();
        let rhs = faer_col::from_fn(blocks.total_size(), |row| {
            ((row + 3) as f64).sin() + 0.1 * (row as f64)
        });

        let reference = solve_bordered_banded_reference(&blocks, &rhs).unwrap();
        let sparse_solution = jac.sp_lu().unwrap().solve(rhs.as_mat());
        let sparse = faer_col::from_fn(sparse_solution.nrows(), |row| *sparse_solution.get(row, 0));

        let max_diff = (0..reference.nrows())
            .map(|row| (reference[row] - sparse[row]).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 1e-10,
            "bordered-banded reference solve must match sparse LU, max_diff={max_diff:e}"
        );
    }

    fn assert_structured_solve_matches_sparse(jac: &faer_mat, n: usize, m: usize, k: usize) {
        let blocks = extract_bordered_banded_blocks(jac, n, m, k).unwrap();
        let rhs = faer_col::from_fn(blocks.total_size(), |row| {
            ((row + 5) as f64).cos() - 0.05 * (row as f64)
        });

        let structured = solve_bordered_banded_structured(&blocks, &rhs).unwrap();
        let sparse_solution = jac.sp_lu().unwrap().solve(rhs.as_mat());
        let sparse = faer_col::from_fn(sparse_solution.nrows(), |row| *sparse_solution.get(row, 0));

        let max_diff = (0..structured.nrows())
            .map(|row| (structured[row] - sparse[row]).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 1e-10,
            "bordered-banded structured solve must match sparse LU, max_diff={max_diff:e}"
        );
    }

    fn simple_valid_blocks_without_parameters() -> BvpSciBorderedBandedBlocks {
        BvpSciBorderedBandedBlocks {
            variable_count: 1,
            mesh_points: 3,
            parameter_count: 0,
            diagonal_blocks: vec![faer_dense_mat::from_fn(1, 1, |_, _| -1.0); 2],
            offdiag_blocks: vec![faer_dense_mat::from_fn(1, 1, |_, _| 1.0); 2],
            collocation_parameter_blocks: None,
            boundary_left: faer_dense_mat::from_fn(1, 1, |_, _| 1.0),
            boundary_right: faer_dense_mat::from_fn(1, 1, |_, _| 0.5),
            boundary_parameters: None,
        }
    }
}
