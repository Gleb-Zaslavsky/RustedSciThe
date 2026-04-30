use super::error::BandedError;

/// C-style row-major indexing helper.
#[inline]
pub fn idx(bs: usize, i: usize, j: usize) -> usize {
    i * bs + j
}

/// Set block to identity.
pub fn block_set_identity(block: &mut [f64], bs: usize) -> Result<(), BandedError> {
    if block.len() != bs * bs {
        return Err(BandedError::DimensionMismatch);
    }

    block.fill(0.0);
    for i in 0..bs {
        block[idx(bs, i, i)] = 1.0;
    }
    Ok(())
}

/// Copy src into dst.
pub fn block_copy(dst: &mut [f64], src: &[f64], bs: usize) -> Result<(), BandedError> {
    if dst.len() != bs * bs || src.len() != bs * bs {
        return Err(BandedError::DimensionMismatch);
    }
    dst.copy_from_slice(src);
    Ok(())
}

/// In-place dense LU without pivoting.
/// Stores L multipliers below diagonal and U on/above diagonal.
pub fn dense_lu_in_place(block: &mut [f64], bs: usize) -> Result<(), BandedError> {
    if block.len() != bs * bs {
        return Err(BandedError::DimensionMismatch);
    }

    let eps = 1e-14;

    for k in 0..bs {
        let pivot = block[idx(bs, k, k)];
        if pivot.abs() <= eps {
            return Err(BandedError::ZeroPivot {
                index: k,
                value: pivot,
            });
        }

        for i in (k + 1)..bs {
            let ik = idx(bs, i, k);
            let mult = block[ik] / pivot;
            block[ik] = mult;

            for j in (k + 1)..bs {
                let ij = idx(bs, i, j);
                let kj = idx(bs, k, j);
                block[ij] -= mult * block[kj];
            }
        }
    }

    Ok(())
}

/// Solve LU x = rhs in place, where LU is output of dense_lu_in_place.
pub fn dense_lu_solve_in_place(lu: &[f64], bs: usize, rhs: &mut [f64]) -> Result<(), BandedError> {
    if lu.len() != bs * bs || rhs.len() != bs {
        return Err(BandedError::DimensionMismatch);
    }

    // Forward solve Ly = rhs, unit diagonal
    for i in 0..bs {
        let mut sum = rhs[i];
        for j in 0..i {
            sum -= lu[idx(bs, i, j)] * rhs[j];
        }
        rhs[i] = sum;
    }

    // Back solve Ux = y
    let eps = 1e-14;
    for i in (0..bs).rev() {
        let mut sum = rhs[i];
        for j in (i + 1)..bs {
            sum -= lu[idx(bs, i, j)] * rhs[j];
        }
        let uii = lu[idx(bs, i, i)];
        if uii.abs() <= eps {
            return Err(BandedError::ZeroPivot {
                index: i,
                value: uii,
            });
        }
        rhs[i] = sum / uii;
    }

    Ok(())
}

/// Solve LU X = RHS in place, multiple RHS stored column-major with leading dimension ldb.
/// RHS has shape (bs, nrhs), column-major.
pub fn dense_lu_solve_multiple_in_place(
    lu: &[f64],
    bs: usize,
    rhs: &mut [f64],
    nrhs: usize,
    ldb: usize,
) -> Result<(), BandedError> {
    if lu.len() != bs * bs || ldb < bs || rhs.len() < nrhs.saturating_mul(ldb) {
        return Err(BandedError::DimensionMismatch);
    }

    for col in 0..nrhs {
        let start = col * ldb;
        let end = start + bs;
        dense_lu_solve_in_place(lu, bs, &mut rhs[start..end])?;
    }

    Ok(())
}

/// dst -= lhs * rhs, all bs x bs, row-major.
pub fn dense_matmul_sub_assign(
    dst: &mut [f64],
    lhs: &[f64],
    rhs: &[f64],
    bs: usize,
) -> Result<(), BandedError> {
    if dst.len() != bs * bs || lhs.len() != bs * bs || rhs.len() != bs * bs {
        return Err(BandedError::DimensionMismatch);
    }

    for i in 0..bs {
        for k in 0..bs {
            let a = lhs[idx(bs, i, k)];
            if a == 0.0 {
                continue;
            }
            for j in 0..bs {
                dst[idx(bs, i, j)] -= a * rhs[idx(bs, k, j)];
            }
        }
    }

    Ok(())
}

/// Solve X * U^{-1} is not what we need.
/// We need X <- A * inv(U) or equivalently row-wise solves only later if needed.
/// For now, keep a basic helper:
/// solve LU X = B, overwriting B, where B is bs x bs row-major columns solved independently.
pub fn dense_lu_solve_block_in_place(
    lu: &[f64],
    bs: usize,
    block_rhs: &mut [f64],
) -> Result<(), BandedError> {
    if lu.len() != bs * bs || block_rhs.len() != bs * bs {
        return Err(BandedError::DimensionMismatch);
    }

    let mut col = vec![0.0; bs];

    for j in 0..bs {
        for i in 0..bs {
            col[i] = block_rhs[idx(bs, i, j)];
        }

        dense_lu_solve_in_place(lu, bs, &mut col)?;

        for i in 0..bs {
            block_rhs[idx(bs, i, j)] = col[i];
        }
    }

    Ok(())
}

/// Transpose src into dst, both bs x bs row-major.
pub fn dense_transpose_into(src: &[f64], dst: &mut [f64], bs: usize) -> Result<(), BandedError> {
    if src.len() != bs * bs || dst.len() != bs * bs {
        return Err(BandedError::DimensionMismatch);
    }

    for i in 0..bs {
        for j in 0..bs {
            dst[idx(bs, j, i)] = src[idx(bs, i, j)];
        }
    }

    Ok(())
}

/// Solve X * A^{-1}, where A is represented by its LU factorization.
/// Both x_block and result are bs x bs row-major.
///
/// Implementation note:
///     X * A^{-1} = (A^{-T} * X^T)^T
///
/// so we solve bs systems with LU(A)^T using a temporary transposed view.
pub fn dense_lu_right_solve_block_in_place(
    lu: &[f64],
    bs: usize,
    x_block: &mut [f64],
) -> Result<(), BandedError> {
    if lu.len() != bs * bs || x_block.len() != bs * bs {
        return Err(BandedError::DimensionMismatch);
    }

    // X * A^{-1} = (A^{-T} * X^T)^T
    let mut tmp_t = vec![0.0; bs * bs];
    dense_transpose_into(x_block, &mut tmp_t, bs)?;

    let mut col = vec![0.0; bs];

    // Solve A^T * y_j = (X^T)_j, one column at a time.
    for j in 0..bs {
        for i in 0..bs {
            col[i] = tmp_t[idx(bs, i, j)];
        }

        dense_lu_solve_transpose_in_place(lu, bs, &mut col)?;

        for i in 0..bs {
            tmp_t[idx(bs, i, j)] = col[i];
        }
    }

    let mut out = vec![0.0; bs * bs];
    dense_transpose_into(&tmp_t, &mut out, bs)?;
    x_block.copy_from_slice(&out);

    Ok(())
}

/// Solve LU(A)^T x = rhs in place, where lu is output of dense_lu_in_place(A).
pub fn dense_lu_solve_transpose_in_place(
    lu: &[f64],
    bs: usize,
    rhs: &mut [f64],
) -> Result<(), BandedError> {
    if lu.len() != bs * bs || rhs.len() != bs {
        return Err(BandedError::DimensionMismatch);
    }

    let eps = 1e-14;

    // Solve U^T y = rhs  (U^T is lower triangular)
    for i in 0..bs {
        let mut sum = rhs[i];
        for j in 0..i {
            sum -= lu[idx(bs, j, i)] * rhs[j];
        }
        let uii = lu[idx(bs, i, i)];
        if uii.abs() <= eps {
            return Err(BandedError::ZeroPivot {
                index: i,
                value: uii,
            });
        }
        rhs[i] = sum / uii;
    }

    // Solve L^T x = y  (L^T is upper triangular, unit diagonal)
    for i in (0..bs).rev() {
        let mut sum = rhs[i];
        for j in (i + 1)..bs {
            sum -= lu[idx(bs, j, i)] * rhs[j];
        }
        rhs[i] = sum;
    }

    Ok(())
}

/// In-place dense LU with partial pivoting.
/// Stores:
/// - strict lower part: L multipliers
/// - diagonal and upper part: U
/// - pivots[k]: row swapped with k at step k
pub fn dense_lu_pivot_in_place(
    block: &mut [f64],
    bs: usize,
    pivots: &mut [usize],
) -> Result<(), BandedError> {
    if block.len() != bs * bs || pivots.len() != bs {
        return Err(BandedError::DimensionMismatch);
    }

    let eps = 1e-14;

    for k in 0..bs {
        let mut p = k;
        let mut max_val = block[idx(bs, k, k)].abs();

        for i in (k + 1)..bs {
            let v = block[idx(bs, i, k)].abs();
            if v > max_val {
                max_val = v;
                p = i;
            }
        }

        if max_val <= eps {
            return Err(BandedError::ZeroPivot {
                index: k,
                value: max_val,
            });
        }

        pivots[k] = p;

        if p != k {
            for j in 0..bs {
                block.swap(idx(bs, k, j), idx(bs, p, j));
            }
        }

        let pivot = block[idx(bs, k, k)];

        for i in (k + 1)..bs {
            let ik = idx(bs, i, k);
            let mult = block[ik] / pivot;
            block[ik] = mult;

            for j in (k + 1)..bs {
                let ij = idx(bs, i, j);
                let kj = idx(bs, k, j);
                block[ij] -= mult * block[kj];
            }
        }
    }

    Ok(())
}

/// Solve A x = rhs in place, where LU and pivots come from dense_lu_pivot_in_place.
/// If P A = L U, then solve proceeds as:
///   L U x = P rhs
pub fn dense_lu_pivot_solve_in_place(
    lu: &[f64],
    bs: usize,
    pivots: &[usize],
    rhs: &mut [f64],
) -> Result<(), BandedError> {
    if lu.len() != bs * bs || pivots.len() != bs || rhs.len() != bs {
        return Err(BandedError::DimensionMismatch);
    }

    // Apply row pivots: rhs <- P rhs
    for k in 0..bs {
        let p = pivots[k];
        if p != k {
            rhs.swap(k, p);
        }
    }

    // Forward solve Ly = P rhs, unit diagonal
    for i in 0..bs {
        let mut sum = rhs[i];
        for j in 0..i {
            sum -= lu[idx(bs, i, j)] * rhs[j];
        }
        rhs[i] = sum;
    }

    // Back solve Ux = y
    let eps = 1e-14;
    for i in (0..bs).rev() {
        let mut sum = rhs[i];
        for j in (i + 1)..bs {
            sum -= lu[idx(bs, i, j)] * rhs[j];
        }

        let uii = lu[idx(bs, i, i)];
        if uii.abs() <= eps {
            return Err(BandedError::ZeroPivot {
                index: i,
                value: uii,
            });
        }

        rhs[i] = sum / uii;
    }

    Ok(())
}

/// Solve A^T x = rhs in place, where LU and pivots come from dense_lu_pivot_in_place.
/// Since P A = L U, we have:
///   A = P^{-1} L U
///   A^T = U^T L^T P^{-T}
///
/// To solve A^T x = rhs:
///   U^T y = rhs
///   L^T z = y
///   x = P^T z
pub fn dense_lu_pivot_solve_transpose_in_place(
    lu: &[f64],
    bs: usize,
    pivots: &[usize],
    rhs: &mut [f64],
) -> Result<(), BandedError> {
    if lu.len() != bs * bs || pivots.len() != bs || rhs.len() != bs {
        return Err(BandedError::DimensionMismatch);
    }

    let eps = 1e-14;

    // Solve U^T y = rhs  (U^T is lower triangular)
    for i in 0..bs {
        let mut sum = rhs[i];
        for j in 0..i {
            sum -= lu[idx(bs, j, i)] * rhs[j];
        }

        let uii = lu[idx(bs, i, i)];
        if uii.abs() <= eps {
            return Err(BandedError::ZeroPivot {
                index: i,
                value: uii,
            });
        }

        rhs[i] = sum / uii;
    }

    // Solve L^T z = y  (L^T is upper triangular, unit diagonal)
    for i in (0..bs).rev() {
        let mut sum = rhs[i];
        for j in (i + 1)..bs {
            sum -= lu[idx(bs, j, i)] * rhs[j];
        }
        rhs[i] = sum;
    }

    // Apply inverse of the earlier row-swaps in reverse order: x <- P^T z
    for k in (0..bs).rev() {
        let p = pivots[k];
        if p != k {
            rhs.swap(k, p);
        }
    }

    Ok(())
}

/// Compute X <- X * A^{-1}, where A is represented by its pivoted LU.
/// Uses:
///   X A^{-1} = (A^{-T} X^T)^T
pub fn dense_lu_pivot_right_solve_block_in_place(
    lu: &[f64],
    bs: usize,
    pivots: &[usize],
    x_block: &mut [f64],
) -> Result<(), BandedError> {
    if lu.len() != bs * bs || pivots.len() != bs || x_block.len() != bs * bs {
        return Err(BandedError::DimensionMismatch);
    }

    // tmp_t = X^T
    let mut tmp_t = vec![0.0; bs * bs];
    dense_transpose_into(x_block, &mut tmp_t, bs)?;

    let mut col = vec![0.0; bs];

    // Solve A^T y_j = (X^T)_j, one column at a time
    for j in 0..bs {
        for i in 0..bs {
            col[i] = tmp_t[idx(bs, i, j)];
        }

        dense_lu_pivot_solve_transpose_in_place(lu, bs, pivots, &mut col)?;

        for i in 0..bs {
            tmp_t[idx(bs, i, j)] = col[i];
        }
    }

    let mut out = vec![0.0; bs * bs];
    dense_transpose_into(&tmp_t, &mut out, bs)?;
    x_block.copy_from_slice(&out);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        block_set_identity, dense_lu_in_place, dense_lu_solve_in_place, dense_matmul_sub_assign,
    };

    fn matvec_row_major(a: &[f64], bs: usize, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; bs];
        for i in 0..bs {
            for j in 0..bs {
                y[i] += a[i * bs + j] * x[j];
            }
        }
        y
    }

    #[test]
    fn identity_block() {
        let mut a = vec![42.0; 9];
        block_set_identity(&mut a, 3).unwrap();

        let expected = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        assert_eq!(a, expected);
    }

    #[test]
    fn dense_lu_solve_small_system() {
        // A = [[4, 1],
        //      [2, 3]]
        let mut a = vec![4.0, 1.0, 2.0, 3.0];

        let x_true = vec![1.0, 2.0];
        let mut rhs = matvec_row_major(&a, 2, &x_true);

        dense_lu_in_place(&mut a, 2).unwrap();
        dense_lu_solve_in_place(&a, 2, &mut rhs).unwrap();

        assert!((rhs[0] - x_true[0]).abs() < 1e-12);
        assert!((rhs[1] - x_true[1]).abs() < 1e-12);
    }

    #[test]
    fn dense_matmul_sub_assign_basic() {
        let mut dst = vec![10.0, 20.0, 30.0, 40.0];

        let lhs = vec![1.0, 2.0, 3.0, 4.0];

        let rhs = vec![5.0, 6.0, 7.0, 8.0];

        dense_matmul_sub_assign(&mut dst, &lhs, &rhs, 2).unwrap();

        // lhs * rhs = [[19, 22], [43, 50]]
        let expected = vec![-9.0, -2.0, -13.0, -10.0];

        for i in 0..4 {
            assert!((dst[i] - expected[i]).abs() < 1e-12);
        }
    }
}

#[cfg(test)]
mod pivot_tests {
    use super::{
        dense_lu_pivot_in_place, dense_lu_pivot_right_solve_block_in_place,
        dense_lu_pivot_solve_in_place, dense_lu_pivot_solve_transpose_in_place, idx,
    };
    use crate::somelinalg::banded::error::BandedError;

    fn matvec_row_major(a: &[f64], bs: usize, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; bs];
        for i in 0..bs {
            for j in 0..bs {
                y[i] += a[idx(bs, i, j)] * x[j];
            }
        }
        y
    }

    fn matvec_transpose_row_major(a: &[f64], bs: usize, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; bs];
        for i in 0..bs {
            for j in 0..bs {
                y[i] += a[idx(bs, j, i)] * x[j];
            }
        }
        y
    }

    fn matmul_row_major(a: &[f64], b: &[f64], bs: usize) -> Vec<f64> {
        let mut c = vec![0.0; bs * bs];
        for i in 0..bs {
            for k in 0..bs {
                let aik = a[idx(bs, i, k)];
                if aik == 0.0 {
                    continue;
                }
                for j in 0..bs {
                    c[idx(bs, i, j)] += aik * b[idx(bs, k, j)];
                }
            }
        }
        c
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

    fn mat_diff_linf(a: &[f64], b: &[f64]) -> f64 {
        let mut m = 0.0;
        for i in 0..a.len() {
            let d = (a[i] - b[i]).abs();
            if d > m {
                m = d;
            }
        }
        m
    }

    fn dense_inverse_2x2(a: &[f64]) -> Vec<f64> {
        let det = a[0] * a[3] - a[1] * a[2];
        vec![a[3] / det, -a[1] / det, -a[2] / det, a[0] / det]
    }

    #[test]
    fn dense_lu_pivot_succeeds_where_no_pivot_would_fail() {
        // [[0, 2],
        //  [1, 3]]
        let mut a = vec![0.0, 2.0, 1.0, 3.0];

        let mut pivots = vec![0usize; 2];
        dense_lu_pivot_in_place(&mut a, 2, &mut pivots).unwrap();

        assert_eq!(pivots[0], 1);
    }

    #[test]
    fn dense_lu_pivot_solve_small_system() {
        // [[0, 2],
        //  [1, 3]]
        let a = vec![0.0, 2.0, 1.0, 3.0];

        let x_true = vec![1.0, 2.0];
        let mut rhs = matvec_row_major(&a, 2, &x_true);

        let mut lu = a.clone();
        let mut pivots = vec![0usize; 2];

        dense_lu_pivot_in_place(&mut lu, 2, &mut pivots).unwrap();
        dense_lu_pivot_solve_in_place(&lu, 2, &pivots, &mut rhs).unwrap();

        assert!(vec_diff_linf(&rhs, &x_true) < 1e-12);
    }

    #[test]
    fn dense_lu_pivot_solve_transpose_small_system() {
        let a = vec![0.0, 2.0, 1.0, 3.0];

        let x_true = vec![1.5, -0.5];
        let mut rhs = matvec_transpose_row_major(&a, 2, &x_true);

        let mut lu = a.clone();
        let mut pivots = vec![0usize; 2];

        dense_lu_pivot_in_place(&mut lu, 2, &mut pivots).unwrap();
        dense_lu_pivot_solve_transpose_in_place(&lu, 2, &pivots, &mut rhs).unwrap();

        assert!(vec_diff_linf(&rhs, &x_true) < 1e-12);
    }

    #[test]
    fn dense_lu_pivot_right_solve_block_matches_2x2_reference() {
        // A = [[0, 2],
        //      [1, 3]]
        let a = vec![0.0, 2.0, 1.0, 3.0];

        // X = [[1, 2],
        //      [3, 4]]
        let mut x = vec![1.0, 2.0, 3.0, 4.0];

        let ainv = dense_inverse_2x2(&a);
        let expected = matmul_row_major(&x, &ainv, 2);

        let mut lu = a.clone();
        let mut pivots = vec![0usize; 2];
        dense_lu_pivot_in_place(&mut lu, 2, &mut pivots).unwrap();

        dense_lu_pivot_right_solve_block_in_place(&lu, 2, &pivots, &mut x).unwrap();

        assert!(mat_diff_linf(&x, &expected) < 1e-10);
    }

    #[test]
    fn dense_lu_pivot_detects_singular() {
        // singular
        let mut a = vec![1.0, 2.0, 2.0, 4.0];
        let mut pivots = vec![0usize; 2];

        let err = dense_lu_pivot_in_place(&mut a, 2, &mut pivots).unwrap_err();
        assert!(matches!(err, BandedError::ZeroPivot { .. }));
    }
}
