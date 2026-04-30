use super::{error::BandedError, storage::Banded};

/// Compute y = A * x for compact banded matrix A.
pub fn banded_matvec(a: &Banded<f64>, x: &[f64]) -> Result<Vec<f64>, BandedError> {
    let n = a.n();
    if x.len() != n {
        return Err(BandedError::DimensionMismatch);
    }

    let mut y = vec![0.0; n];

    // Column-oriented traversal matches compact storage naturally.
    for j in 0..n {
        let xj = x[j];
        if xj == 0.0 {
            continue;
        }

        let i0 = j.saturating_sub(a.ku());
        let i1 = (j + a.kl() + 1).min(n);

        for i in i0..i1 {
            y[i] += a[(i, j)] * xj;
        }
    }

    Ok(y)
}

/// Infinity norm of residual r = A*x - b.
pub fn residual_linf(a: &Banded<f64>, x: &[f64], b: &[f64]) -> Result<f64, BandedError> {
    let ax = banded_matvec(a, x)?;
    if b.len() != ax.len() {
        return Err(BandedError::DimensionMismatch);
    }

    let mut rmax = 0.0;
    for i in 0..ax.len() {
        let ri = (ax[i] - b[i]).abs();
        if ri > rmax {
            rmax = ri;
        }
    }

    Ok(rmax)
}

/// Euclidean norm of residual r = A*x - b.
pub fn residual_l2(a: &Banded<f64>, x: &[f64], b: &[f64]) -> Result<f64, BandedError> {
    let ax = banded_matvec(a, x)?;
    if b.len() != ax.len() {
        return Err(BandedError::DimensionMismatch);
    }

    let mut s = 0.0;
    for i in 0..ax.len() {
        let ri = ax[i] - b[i];
        s += ri * ri;
    }

    Ok(s.sqrt())
}

/// Convert compact banded matrix to dense representation.
/// Debug/test helper only.
pub fn banded_to_dense(a: &Banded<f64>) -> Vec<Vec<f64>> {
    let n = a.n();
    let mut out = vec![vec![0.0; n]; n];

    for j in 0..n {
        let i0 = j.saturating_sub(a.ku());
        let i1 = (j + a.kl() + 1).min(n);

        for i in i0..i1 {
            out[i][j] = a[(i, j)];
        }
    }

    out
}

/// Dense y = A * x.
/// Debug/test helper only.
pub fn dense_matvec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    let n = a.len();
    let mut y = vec![0.0; n];

    for i in 0..n {
        for j in 0..n {
            y[i] += a[i][j] * x[j];
        }
    }

    y
}

/// Dense infinity norm of difference between two vectors.
pub fn vec_diff_linf(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let mut vmax = 0.0;
    for i in 0..x.len() {
        let d = (x[i] - y[i]).abs();
        if d > vmax {
            vmax = d;
        }
    }
    vmax
}

/// Dense matrix-matrix multiplication C = A * B.
/// Debug/test helper only.
pub fn dense_matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    assert_eq!(b.len(), n);
    for row in a {
        assert_eq!(row.len(), n);
    }
    for row in b {
        assert_eq!(row.len(), n);
    }

    let mut c = vec![vec![0.0; n]; n];

    for i in 0..n {
        for k in 0..n {
            let aik = a[i][k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i][j] += aik * b[k][j];
            }
        }
    }

    c
}

/// Dense infinity norm of matrix difference A - B.
/// Debug/test helper only.
pub fn dense_diff_linf(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let n = a.len();
    assert_eq!(b.len(), n);

    let mut vmax = 0.0;
    for i in 0..n {
        assert_eq!(a[i].len(), n);
        assert_eq!(b[i].len(), n);
        for j in 0..n {
            let d = (a[i][j] - b[i][j]).abs();
            if d > vmax {
                vmax = d;
            }
        }
    }
    vmax
}
//===================================================================================================
#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use super::{banded_matvec, residual_l2, residual_linf, vec_diff_linf};
    use crate::somelinalg::banded::{general_lu::GeneralBandedLuNoPivot, storage::Banded};

    fn random_diag_dominant_banded(n: usize, kl: usize, ku: usize, seed: u64) -> Banded<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut a = Banded::<f64>::zeros(n, kl, ku).unwrap();

        // Build by columns to match storage naturally.
        for j in 0..n {
            let i0 = j.saturating_sub(ku);
            let i1 = (j + kl + 1).min(n);

            let mut col_abs_sum = 0.0;
            for i in i0..i1 {
                if i == j {
                    continue;
                }
                let v = rng.gen_range(-1.0..1.0);
                a[(i, j)] = v;
                col_abs_sum += v.abs();
            }

            // Make diagonal comfortably dominant.
            a[(j, j)] = col_abs_sum + rng.gen_range(1.0..2.0);
        }

        a
    }

    #[test]
    fn banded_matvec_matches_known_tridiagonal() {
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

        let x = vec![1.0, 1.0, 1.0, 1.0];
        let y = banded_matvec(&a, &x).unwrap();

        let expected = vec![5.0, 8.0, 10.0, 11.0];
        assert!(vec_diff_linf(&y, &expected) < 1e-14);
    }

    #[test]
    fn residuals_are_small_after_solve_known_case() {
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
        let b = banded_matvec(&a, &x_true).unwrap();

        let mut x = b.clone();
        let mut lu = GeneralBandedLuNoPivot::new(4, 1, 1).unwrap();
        lu.factor_from(&a).unwrap();
        lu.solve_in_place(&mut x).unwrap();

        let r_inf = residual_linf(&a, &x, &b).unwrap();
        let r_l2 = residual_l2(&a, &x, &b).unwrap();

        assert!(r_inf < 1e-12, "residual inf too large: {r_inf:e}");
        assert!(r_l2 < 1e-12, "residual l2 too large: {r_l2:e}");
        assert!(vec_diff_linf(&x, &x_true) < 1e-12);
    }

    #[test]
    fn random_diagonally_dominant_banded_systems_solve_correctly() {
        let cases = [
            (8, 1, 1, 11_u64),
            (12, 2, 2, 22_u64),
            (20, 3, 1, 33_u64),
            (24, 2, 4, 44_u64),
            (30, 4, 3, 55_u64),
        ];

        for (n, kl, ku, seed) in cases {
            let a = random_diag_dominant_banded(n, kl, ku, seed);

            let mut rng = StdRng::seed_from_u64(seed + 10_000);
            let x_true: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let b = banded_matvec(&a, &x_true).unwrap();

            let mut x = b.clone();
            let mut lu = GeneralBandedLuNoPivot::new(n, kl, ku).unwrap();
            lu.factor_from(&a).unwrap();

            if let Err(err) = lu.solve_in_place(&mut x) {
                panic!("solve failed for n={n}, kl={kl}, ku={ku}, seed={seed}: {err}");
            }

            let r_inf = residual_linf(&a, &x, &b).unwrap();
            let x_err = vec_diff_linf(&x, &x_true);

            assert!(
                r_inf < 1e-8,
                "large residual for n={n}, kl={kl}, ku={ku}, seed={seed}: {r_inf:e}"
            );
            assert!(
                x_err < 1e-8,
                "large solution error for n={n}, kl={kl}, ku={ku}, seed={seed}: {x_err:e}"
            );
        }
    }

    #[test]
    fn debug_dump_dense_returns_nonempty_string() {
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

        let dump = lu.storage().debug_dump_dense();
        assert!(!dump.is_empty());
        assert!(dump.contains("e"));
    }
}
