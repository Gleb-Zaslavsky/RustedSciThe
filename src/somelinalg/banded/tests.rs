use super::{error::BandedError, storage::Banded};

use super::{
    general_lu::GeneralBandedLuNoPivot,
    ops::{banded_matvec, residual_linf},
};

use rand::{Rng, SeedableRng, rngs::StdRng};

pub fn random_diag_dominant_banded(n: usize, kl: usize, ku: usize, seed: u64) -> Banded<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut a = Banded::<f64>::zeros(n, kl, ku).unwrap();

    for j in 0..n {
        let i0 = j.saturating_sub(ku);
        let i1 = (j + kl + 1).min(n);

        let mut col_abs_sum = 0.0;
        for i in i0..i1 {
            if i == j {
                continue;
            }
            let v = rng.random_range(-1.0..1.0);
            a[(i, j)] = v;
            col_abs_sum += v.abs();
        }

        // Strong diagonal dominance
        let diag = col_abs_sum + rng.random_range(1.0..2.0);
        a[(j, j)] = diag;
    }

    a
}
#[cfg(test)]

mod tests {
    use super::*;
    use crate::somelinalg::banded::general_lu::GeneralBandedLuNoPivot;

    fn dense_matvec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
        let n = a.len();
        let mut y = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                y[i] += a[i][j] * x[j];
            }
        }
        y
    }

    #[test]
    fn factor_and_solve_tridiagonal() {
        // Dense reference matrix:
        // [ 4 1 0 0 ]
        // [ 2 5 1 0 ]
        // [ 0 3 6 1 ]
        // [ 0 0 4 7 ]
        let dense = vec![
            vec![4.0, 1.0, 0.0, 0.0],
            vec![2.0, 5.0, 1.0, 0.0],
            vec![0.0, 3.0, 6.0, 1.0],
            vec![0.0, 0.0, 4.0, 7.0],
        ];

        let mut a = Banded::<f64>::zeros(4, 1, 1).unwrap();
        a[(0, 0)] = 4.0;
        a[(0, 1)] = 1.0;
        a[(1, 0)] = 2.0;
        a[(1, 1)] = 5.0;
        a[(1, 2)] = 1.0;
        a[(2, 1)] = 3.0;
        a[(2, 2)] = 6.0;
        a[(2, 3)] = 1.0;
        a[(3, 2)] = 4.0;
        a[(3, 3)] = 7.0;

        let x_true = vec![1.0, 1.0, 1.0, 1.0];
        let mut rhs = dense_matvec(&dense, &x_true);

        let mut lu = GeneralBandedLuNoPivot::new(4, 1, 1).unwrap();
        lu.factor_from(&a).unwrap();
        lu.solve_in_place(&mut rhs).unwrap();

        for (x, xt) in rhs.iter().zip(x_true.iter()) {
            assert!((x - xt).abs() < 1e-10);
        }
    }

    #[test]
    fn solve_multiple_rhs() {
        let mut a = Banded::<f64>::zeros(3, 1, 1).unwrap();
        a[(0, 0)] = 4.0;
        a[(0, 1)] = 1.0;
        a[(1, 0)] = 1.0;
        a[(1, 1)] = 3.0;
        a[(1, 2)] = 1.0;
        a[(2, 1)] = 1.0;
        a[(2, 2)] = 2.0;

        let mut lu = GeneralBandedLuNoPivot::new(3, 1, 1).unwrap();
        lu.factor_from(&a).unwrap();

        // Column-major RHS with ldb = 3, nrhs = 2
        // First RHS corresponds to x = [1, 0, 1]
        // Second RHS corresponds to x = [0, 1, 1]
        let mut rhs = vec![
            4.0, 2.0, 2.0, // A * [1,0,1]
            1.0, 4.0, 3.0, // A * [0,1,1]
        ];

        lu.solve_multiple_in_place(&mut rhs, 2, 3).unwrap();

        let expected = vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0];

        for (x, xt) in rhs.iter().zip(expected.iter()) {
            assert!((x - xt).abs() < 1e-10);
        }
    }
}

#[cfg(test)]
mod random_tests {
    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    fn random_diag_dominant_banded(n: usize, kl: usize, ku: usize, seed: u64) -> Banded<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut a = Banded::<f64>::zeros(n, kl, ku).unwrap();

        for j in 0..n {
            let i0 = j.saturating_sub(ku);
            let i1 = (j + kl + 1).min(n);

            let mut col_abs_sum = 0.0;
            for i in i0..i1 {
                if i == j {
                    continue;
                }
                let v = rng.random_range(-1.0..1.0);
                a[(i, j)] = v;
                col_abs_sum += v.abs();
            }

            a[(j, j)] = col_abs_sum + rng.random_range(1.0..2.0);
        }

        a
    }

    #[test]
    fn random_diagonally_dominant_banded_systems() {
        let cases = [
            (8, 1, 1, 1_u64),
            (12, 2, 2, 2_u64),
            (20, 3, 1, 3_u64),
            (25, 2, 4, 4_u64),
        ];

        for (n, kl, ku, seed) in cases {
            let a = random_diag_dominant_banded(n, kl, ku, seed);

            let mut rng = StdRng::seed_from_u64(seed + 1000);
            let x_true: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
            let mut rhs = banded_matvec(&a, &x_true).unwrap();

            let mut lu = GeneralBandedLuNoPivot::new(n, kl, ku).unwrap();
            lu.factor_from(&a).unwrap();
            lu.solve_in_place(&mut rhs).unwrap();

            let res = residual_linf(&a, &rhs, &banded_matvec(&a, &x_true).unwrap()).unwrap();
            assert!(res < 1e-8, "residual too large: {res:e}");

            for i in 0..n {
                assert!(
                    (rhs[i] - x_true[i]).abs() < 1e-8,
                    "x mismatch at {i}: got {}, expected {}",
                    rhs[i],
                    x_true[i]
                );
            }
        }
    }
}

#[cfg(test)]
mod lu_reconstruction_tests {
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use crate::somelinalg::banded::{
        banded_to_dense, dense_diff_linf, general_lu::GeneralBandedLuNoPivot, storage::Banded,
    };

    fn random_diag_dominant_banded(n: usize, kl: usize, ku: usize, seed: u64) -> Banded<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut a = Banded::<f64>::zeros(n, kl, ku).unwrap();

        for j in 0..n {
            let i0 = j.saturating_sub(ku);
            let i1 = (j + kl + 1).min(n);

            let mut col_abs_sum = 0.0;
            for i in i0..i1 {
                if i == j {
                    continue;
                }
                let v = rng.random_range(-1.0..1.0);
                a[(i, j)] = v;
                col_abs_sum += v.abs();
            }

            a[(j, j)] = col_abs_sum + rng.random_range(1.0..2.0);
        }

        a
    }

    #[test]
    fn reconstruct_lu_matches_known_tridiagonal_matrix() {
        let mut a = Banded::<f64>::zeros(4, 1, 1).unwrap();
        a[(0, 0)] = 4.0;
        a[(0, 1)] = 1.0;
        a[(1, 0)] = 2.0;
        a[(1, 1)] = 5.0;
        a[(1, 2)] = 1.0;
        a[(2, 1)] = 3.0;
        a[(2, 2)] = 6.0;
        a[(2, 3)] = 1.0;
        a[(3, 2)] = 4.0;
        a[(3, 3)] = 7.0;

        let a_dense = banded_to_dense(&a);

        let mut lu = GeneralBandedLuNoPivot::new(4, 1, 1).unwrap();
        lu.factor_from(&a).unwrap();

        let lu_dense = lu.reconstruct_lu_product_dense().unwrap();
        let err = dense_diff_linf(&a_dense, &lu_dense);

        assert!(err < 1e-12, "A vs LU mismatch too large: {err:e}");
    }

    #[test]
    fn extract_l_and_u_have_expected_structure() {
        let mut a = Banded::<f64>::zeros(4, 1, 1).unwrap();
        a[(0, 0)] = 4.0;
        a[(0, 1)] = 1.0;
        a[(1, 0)] = 2.0;
        a[(1, 1)] = 5.0;
        a[(1, 2)] = 1.0;
        a[(2, 1)] = 3.0;
        a[(2, 2)] = 6.0;
        a[(2, 3)] = 1.0;
        a[(3, 2)] = 4.0;
        a[(3, 3)] = 7.0;

        let mut lu = GeneralBandedLuNoPivot::new(4, 1, 1).unwrap();
        lu.factor_from(&a).unwrap();

        let l = lu.extract_l_dense().unwrap();
        let u = lu.extract_u_dense().unwrap();

        for i in 0..4 {
            assert!((l[i][i] - 1.0).abs() < 1e-14);
            for j in (i + 1)..4 {
                assert!(l[i][j].abs() < 1e-14, "L must be lower triangular");
            }
            for j in 0..i {
                if i > j + 1 {
                    assert!(l[i][j].abs() < 1e-14, "L must respect lower bandwidth");
                }
            }
        }

        for i in 0..4 {
            for j in 0..i {
                assert!(u[i][j].abs() < 1e-14, "U must be upper triangular");
            }
        }
    }

    #[test]
    fn reconstruct_lu_matches_random_diag_dominant_cases() {
        let cases = [
            (8, 1, 1, 111_u64),
            (10, 2, 2, 222_u64),
            (14, 3, 1, 333_u64),
            (16, 2, 4, 444_u64),
            (20, 4, 3, 555_u64),
        ];

        for (n, kl, ku, seed) in cases {
            let a = random_diag_dominant_banded(n, kl, ku, seed);
            let a_dense = banded_to_dense(&a);

            let mut lu = GeneralBandedLuNoPivot::new(n, kl, ku).unwrap();
            lu.factor_from(&a).unwrap();

            let lu_dense = lu.reconstruct_lu_product_dense().unwrap();
            let err = dense_diff_linf(&a_dense, &lu_dense);

            assert!(
                err < 1e-10,
                "A vs LU mismatch too large for n={n}, kl={kl}, ku={ku}, seed={seed}: {err:e}"
            );
        }
    }
}

#[cfg(test)]
mod robustness_tests {
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use crate::somelinalg::banded::BlockTridiagonalLu;
    use crate::somelinalg::banded::block_tridiagonal::BlockTridiagonal;
    use crate::somelinalg::banded::error::BandedError;

    fn block_matvec(a: &BlockTridiagonal, x: &[f64]) -> Vec<f64> {
        let nb = a.n_blocks();
        let bs = a.block_size();
        let n = a.n();
        assert_eq!(x.len(), n);

        let mut y = vec![0.0; n];

        for blk in 0..nb {
            let r0 = blk * bs;

            let d = a.diag_block(blk).unwrap();
            for i in 0..bs {
                for j in 0..bs {
                    y[r0 + i] += d[i * bs + j] * x[r0 + j];
                }
            }

            if blk > 0 {
                let l = a.lower_block(blk - 1).unwrap();
                let c0 = (blk - 1) * bs;
                for i in 0..bs {
                    for j in 0..bs {
                        y[r0 + i] += l[i * bs + j] * x[c0 + j];
                    }
                }
            }

            if blk + 1 < nb {
                let u = a.upper_block(blk).unwrap();
                let c0 = (blk + 1) * bs;
                for i in 0..bs {
                    for j in 0..bs {
                        y[r0 + i] += u[i * bs + j] * x[c0 + j];
                    }
                }
            }
        }

        y
    }

    fn vec_diff_linf(x: &[f64], y: &[f64]) -> f64 {
        let mut m = 0.0;
        for i in 0..x.len() {
            let d = (x[i] - y[i]).abs();
            if d > m {
                m = d;
            }
        }
        m
    }

    fn generate_random_block_tridiagonal(
        n_blocks: usize,
        block_size: usize,
        diag_scale: f64,
        offdiag_scale: f64,
        diagonal_boost: f64,
        seed: u64,
    ) -> BlockTridiagonal {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut a = BlockTridiagonal::zeros(n_blocks, block_size).unwrap();

        for blk in 0..n_blocks {
            for j in 0..block_size {
                let mut abs_sum = 0.0;

                // diagonal block
                for i in 0..block_size {
                    if i == j {
                        continue;
                    }
                    let v = rng.random_range(-diag_scale..diag_scale);
                    a.set_diag(blk, i, j, v).unwrap();
                    abs_sum += v.abs();
                }

                // lower coupling
                if blk > 0 {
                    for i in 0..block_size {
                        let v = rng.random_range(-offdiag_scale..offdiag_scale);
                        a.set_lower(blk - 1, i, j, v).unwrap();
                        abs_sum += v.abs();
                    }
                }

                // upper coupling
                if blk + 1 < n_blocks {
                    for i in 0..block_size {
                        let v = rng.random_range(-offdiag_scale..offdiag_scale);
                        a.set_upper(blk, i, j, v).unwrap();
                        abs_sum += v.abs();
                    }
                }

                // strong/weak diagonal dominance controlled by diagonal_boost
                a.set_diag(
                    blk,
                    j,
                    j,
                    abs_sum + diagonal_boost + rng.random_range(0.0..1.0),
                )
                .unwrap();
            }
        }

        a
    }

    #[test]
    fn random_block_systems_with_known_solution() {
        let cases = [
            (8, 4, 0.2, 0.05, 1.0, 1001_u64),
            (12, 6, 0.2, 0.05, 1.0, 1002_u64),
            (20, 8, 0.25, 0.05, 1.0, 1003_u64),
        ];

        for (n_blocks, block_size, diag_scale, offdiag_scale, diagonal_boost, seed) in cases {
            let a = generate_random_block_tridiagonal(
                n_blocks,
                block_size,
                diag_scale,
                offdiag_scale,
                diagonal_boost,
                seed,
            );

            let n = a.n();
            let mut rng = StdRng::seed_from_u64(seed + 999);
            let x_true: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
            let b = block_matvec(&a, &x_true);

            let mut x = b.clone();
            let mut lu = BlockTridiagonalLu::new(n_blocks, block_size).unwrap();
            lu.factor_from(&a).unwrap();
            lu.solve_in_place(&mut x).unwrap();

            let err = vec_diff_linf(&x, &x_true);
            assert!(err < 1e-8, "solution error too large: {err:e}");
        }
    }

    #[test]
    fn weaker_diagonal_dominance_still_solves() {
        let cases = [
            (10, 6, 0.3, 0.15, 0.25, 2001_u64),
            (12, 8, 0.35, 0.15, 0.10, 2002_u64),
        ];

        for (n_blocks, block_size, diag_scale, offdiag_scale, diagonal_boost, seed) in cases {
            let a = generate_random_block_tridiagonal(
                n_blocks,
                block_size,
                diag_scale,
                offdiag_scale,
                diagonal_boost,
                seed,
            );

            let n = a.n();
            let mut rng = StdRng::seed_from_u64(seed + 111);
            let x_true: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
            let b = block_matvec(&a, &x_true);

            let mut x = b.clone();
            let mut lu = BlockTridiagonalLu::new(n_blocks, block_size).unwrap();
            lu.factor_from(&a).unwrap();
            lu.solve_in_place(&mut x).unwrap();

            let err = vec_diff_linf(&x, &x_true);
            assert!(err < 1e-7, "solution error too large: {err:e}");
        }
    }

    #[test]
    fn near_singular_diagonal_block_reports_failure() {
        let mut a = BlockTridiagonal::zeros(2, 2).unwrap();

        // First diagonal block nearly singular
        a.set_diag(0, 0, 0, 1.0).unwrap();
        a.set_diag(0, 0, 1, 2.0).unwrap();
        a.set_diag(0, 1, 0, 2.0).unwrap();
        a.set_diag(0, 1, 1, 4.0 + 1e-16).unwrap();

        // Coupling
        a.set_upper(0, 0, 0, 0.1).unwrap();
        a.set_upper(0, 1, 1, 0.1).unwrap();
        a.set_lower(0, 0, 0, 0.1).unwrap();
        a.set_lower(0, 1, 1, 0.1).unwrap();

        // Second diagonal block
        a.set_diag(1, 0, 0, 3.0).unwrap();
        a.set_diag(1, 0, 1, 0.5).unwrap();
        a.set_diag(1, 1, 0, 0.5).unwrap();
        a.set_diag(1, 1, 1, 2.5).unwrap();

        let mut lu = BlockTridiagonalLu::new(2, 2).unwrap();
        let err = lu.factor_from(&a).unwrap_err();

        assert!(matches!(err, BandedError::ZeroPivot { .. }));
    }

    #[test]
    fn larger_multiple_rhs_random_case() {
        let n_blocks = 10;
        let block_size = 6;
        let a = generate_random_block_tridiagonal(n_blocks, block_size, 0.2, 0.05, 1.0, 3001);

        let n = a.n();
        let mut rng = StdRng::seed_from_u64(3002);

        let x1: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
        let x2: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
        let x3: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();

        let b1 = block_matvec(&a, &x1);
        let b2 = block_matvec(&a, &x2);
        let b3 = block_matvec(&a, &x3);

        let mut rhs = Vec::with_capacity(3 * n);
        rhs.extend_from_slice(&b1);
        rhs.extend_from_slice(&b2);
        rhs.extend_from_slice(&b3);

        let mut lu = BlockTridiagonalLu::new(n_blocks, block_size).unwrap();
        lu.factor_from(&a).unwrap();
        lu.solve_multiple_in_place(&mut rhs, 3, n).unwrap();

        let got1 = &rhs[0..n];
        let got2 = &rhs[n..2 * n];
        let got3 = &rhs[2 * n..3 * n];

        assert!(vec_diff_linf(got1, &x1) < 1e-8);
        assert!(vec_diff_linf(got2, &x2) < 1e-8);
        assert!(vec_diff_linf(got3, &x3) < 1e-8);
    }
}
