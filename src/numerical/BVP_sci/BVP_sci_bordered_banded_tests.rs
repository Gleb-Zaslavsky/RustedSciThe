#[cfg(test)]
mod tests {
    use crate::numerical::BVP_sci::BVP_sci_bordered_banded::{
        BvpSciBandedRoute, BvpSciBandedRoutePolicy, profile_bordered_banded_global_jacobian,
    };
    use crate::numerical::BVP_sci::BVP_sci_faer::{construct_global_jac, faer_col};
    use faer::sparse::{SparseColMat, Triplet};

    fn sparse_from_triplets(
        nrows: usize,
        ncols: usize,
        triplets: Vec<Triplet<usize, usize, f64>>,
    ) -> SparseColMat<usize, f64> {
        SparseColMat::try_new_from_triplets(nrows, ncols, &triplets).unwrap()
    }

    fn dense_local_jacobian(n: usize, diag: f64, off_diag: f64) -> SparseColMat<usize, f64> {
        let mut triplets = Vec::with_capacity(n * n);
        for row in 0..n {
            for col in 0..n {
                let value = if row == col { diag } else { off_diag };
                triplets.push(Triplet::new(row, col, value));
            }
        }
        sparse_from_triplets(n, n, triplets)
    }

    fn endpoint_bc_jacobian(n: usize) -> SparseColMat<usize, f64> {
        let triplets = (0..n)
            .map(|idx| Triplet::new(idx, idx, 1.0))
            .collect::<Vec<_>>();
        sparse_from_triplets(n, n, triplets)
    }

    fn empty_bc_jacobian(n: usize) -> SparseColMat<usize, f64> {
        sparse_from_triplets(n, n, Vec::new())
    }

    fn global_jacobian_for_dense_block_bvp(
        n: usize,
        m: usize,
        left_bc: bool,
        right_bc: bool,
    ) -> SparseColMat<usize, f64> {
        let k = 0;
        let h = faer_col::from_fn(m - 1, |_| 1.0 / (m - 1) as f64);
        let df_dy = (0..m)
            .map(|_| dense_local_jacobian(n, 2.0, 0.05))
            .collect::<Vec<_>>();
        let df_dy_middle = (0..m - 1)
            .map(|_| dense_local_jacobian(n, 2.1, 0.04))
            .collect::<Vec<_>>();
        let dbc_dya = if left_bc {
            endpoint_bc_jacobian(n)
        } else {
            empty_bc_jacobian(n)
        };
        let dbc_dyb = if right_bc {
            endpoint_bc_jacobian(n)
        } else {
            empty_bc_jacobian(n)
        };

        construct_global_jac(
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
        )
    }

    #[test]
    fn compact_full_scalar_matrix_recommends_full_banded() {
        let mat = sparse_from_triplets(
            4,
            4,
            vec![
                Triplet::new(0, 0, 4.0),
                Triplet::new(1, 0, -1.0),
                Triplet::new(0, 1, -1.0),
                Triplet::new(1, 1, 4.0),
                Triplet::new(2, 1, -1.0),
                Triplet::new(1, 2, -1.0),
                Triplet::new(2, 2, 4.0),
                Triplet::new(3, 2, -1.0),
                Triplet::new(2, 3, -1.0),
                Triplet::new(3, 3, 4.0),
            ],
        );

        let profile = profile_bordered_banded_global_jacobian(
            &mat,
            1,
            4,
            0,
            BvpSciBandedRoutePolicy::default(),
        );
        assert_eq!(
            profile.recommended_route,
            BvpSciBandedRoute::FullScalarBanded
        );
        assert!(profile.full_amplification.unwrap() <= 8.0);
    }

    #[test]
    fn endpoint_bc_rows_recommend_bordered_banded() {
        let n = 6;
        let m = 200;
        let mat = global_jacobian_for_dense_block_bvp(n, m, true, true);

        let profile = profile_bordered_banded_global_jacobian(
            &mat,
            n,
            m,
            0,
            BvpSciBandedRoutePolicy::default(),
        );

        assert_eq!(profile.recommended_route, BvpSciBandedRoute::BorderedBanded);
        assert_eq!(profile.full_scalar.kl, (m - 1) * n);
        assert!(profile.collocation_scalar.kl <= n);
        assert!(profile.full_amplification.unwrap() > 50.0);
        assert!(profile.collocation_amplification.unwrap() < 2.0);
    }

    #[test]
    fn shape_mismatch_recommends_sparse_fallback() {
        let mat = sparse_from_triplets(2, 3, vec![Triplet::new(0, 0, 1.0)]);

        let profile = profile_bordered_banded_global_jacobian(
            &mat,
            1,
            2,
            0,
            BvpSciBandedRoutePolicy::default(),
        );

        assert_eq!(profile.recommended_route, BvpSciBandedRoute::SparseFallback);
        assert_eq!(
            profile.reason,
            "matrix shape does not match BVP_sci global layout"
        );
    }

    #[test]
    fn wide_sparse_collocation_recommends_sparse_fallback() {
        let n = 8;
        let m = 20;
        let size = n * m;
        let collocation_rows = n * (m - 1);
        let mut triplets = Vec::with_capacity(collocation_rows * 2);
        for row in 0..collocation_rows {
            triplets.push(Triplet::new(row, row, 2.0));
            triplets.push(Triplet::new(row, size - 1, 0.01));
        }
        let mat = sparse_from_triplets(size, size, triplets);

        let profile = profile_bordered_banded_global_jacobian(
            &mat,
            n,
            m,
            0,
            BvpSciBandedRoutePolicy::default(),
        );

        assert_eq!(profile.recommended_route, BvpSciBandedRoute::SparseFallback);
        assert!(profile.full_amplification.unwrap() > 8.0);
        assert!(profile.collocation_amplification.unwrap() > 4.0);
    }
}
