#[cfg(test)]
mod tests {
    use crate::numerical::BVP_sci::BVP_sci_banded::{
        infer_banded_profile, infer_banded_profile_for_row_range, solve_banded_lapack_faithful,
        sparse_global_jac_to_banded,
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

    #[test]
    fn banded_profile_detects_tridiagonal_bandwidth() {
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

        let profile = infer_banded_profile(&mat);
        assert_eq!(profile.nrows, 4);
        assert_eq!(profile.ncols, 4);
        assert_eq!(profile.kl, 1);
        assert_eq!(profile.ku, 1);
        assert_eq!(profile.nonfinite_values, 0);
        assert_eq!(profile.storage_amplification(), Some(12.0 / 10.0));
    }

    #[test]
    fn sparse_to_banded_preserves_entries() {
        let mat = sparse_from_triplets(
            5,
            5,
            vec![
                Triplet::new(0, 0, 10.0),
                Triplet::new(2, 0, 20.0),
                Triplet::new(1, 2, 30.0),
                Triplet::new(4, 3, 40.0),
                Triplet::new(3, 4, 50.0),
            ],
        );

        let banded = sparse_global_jac_to_banded(&mat).unwrap();
        assert_eq!(banded.n(), 5);
        assert_eq!(banded.kl(), 2);
        assert_eq!(banded.ku(), 1);
        assert_eq!(banded.get(0, 0), Some(&10.0));
        assert_eq!(banded.get(2, 0), Some(&20.0));
        assert_eq!(banded.get(1, 2), Some(&30.0));
        assert_eq!(banded.get(4, 3), Some(&40.0));
        assert_eq!(banded.get(3, 4), Some(&50.0));
        assert!(banded.get(0, 4).is_none());
    }

    #[test]
    fn banded_solve_matches_known_solution_for_sparse_tridiagonal() {
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

        let expected = [1.0, 2.0, 3.0, 4.0];
        let rhs = faer_col::from_fn(4, |i| {
            let mut acc = 4.0 * expected[i];
            if i > 0 {
                acc -= expected[i - 1];
            }
            if i + 1 < expected.len() {
                acc -= expected[i + 1];
            }
            acc
        });

        let x = solve_banded_lapack_faithful(&mat, &rhs).unwrap();
        for i in 0..4 {
            assert!(
                (x[i] - expected[i]).abs() < 1e-11,
                "x[{i}]={}, expected={}",
                x[i],
                expected[i]
            );
        }
    }

    #[test]
    fn non_square_sparse_matrix_is_rejected() {
        let mat = sparse_from_triplets(2, 3, vec![Triplet::new(0, 0, 1.0)]);
        assert!(sparse_global_jac_to_banded(&mat).is_err());
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

    fn global_jacobian_for_dense_block_bvp(n: usize, m: usize) -> SparseColMat<usize, f64> {
        let k = 0;
        let h = faer_col::from_fn(m - 1, |_| 1.0 / (m - 1) as f64);
        let df_dy = (0..m)
            .map(|_| dense_local_jacobian(n, 2.0, 0.05))
            .collect::<Vec<_>>();
        let df_dy_middle = (0..m - 1)
            .map(|_| dense_local_jacobian(n, 2.1, 0.04))
            .collect::<Vec<_>>();
        let dbc_dya = endpoint_bc_jacobian(n);
        let dbc_dyb = endpoint_bc_jacobian(n);

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
    fn bvp_sci_global_jacobian_bandwidth_story_table() {
        let cases = [("small-2x5", 2, 5), ("combustion-shaped-6x200", 6, 200)];

        println!("\n[BVP_sci banded profile] global Jacobian scalar bandwidth diagnostic");
        println!(
            "case | n | m | size | nnz_full | full_kl | full_ku | full_amp | colloc_nnz | colloc_kl | colloc_ku | colloc_amp"
        );
        println!("{}", "-".repeat(146));

        for (label, n, m) in cases {
            let jac = global_jacobian_for_dense_block_bvp(n, m);
            let full = infer_banded_profile(&jac);
            let collocation_rows = (m - 1) * n;
            let colloc = infer_banded_profile_for_row_range(&jac, 0, collocation_rows);

            println!(
                "{label} | {n} | {m} | {} | {} | {} | {} | {:.3} | {} | {} | {} | {:.3}",
                full.nrows,
                full.nnz,
                full.kl,
                full.ku,
                full.storage_amplification().unwrap_or(f64::NAN),
                colloc.nnz,
                colloc.kl,
                colloc.ku,
                colloc.storage_amplification().unwrap_or(f64::NAN),
            );

            assert_eq!(full.nrows, n * m);
            assert_eq!(full.ncols, n * m);
            assert!(colloc.kl < full.kl, "BC rows should widen lower bandwidth");
            assert!(
                colloc.storage_amplification().unwrap() < full.storage_amplification().unwrap(),
                "collocation-only banded storage should be more compact than full scalar banded storage"
            );
        }
    }
}
