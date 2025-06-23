//! Pivoted QR factorization and a specialized LLS solver.
//!
//! The QR factorization is used to implement an efficient solver for the
//! linear least squares problem which is repeatedly required to be
//! solved in the LM algorithm.
#![allow(clippy::excessive_precision)]

use crate::numerical::optimization::utils::{dot, enorm, epsmch};
use nalgebra::{DMatrix, DVector};
use num_traits::Float;
use std::fmt::Display;

/// Pivoted QR decomposition.
///
/// Let `A` be an `m×n` matrix, then this algorithm computes a permutation matrix `P`,
/// a matrix `Q` with orthonormal columns and an upper triangular matrix `R` such that
/// `P^T * A * P = Q * R`.
#[derive(Debug, Clone, PartialEq)]
pub struct PivotedQR {
    /// The column norms of the input matrix `A`
    column_norms: DVector<f64>,
    /// Strictly upper part of `R` and the Householder transformations,
    /// combined in one matrix.
    qr: DMatrix<f64>,
    /// Diagonal entries of R
    r_diag: DVector<f64>,
    /// Permutation matrix. Entry `i` specifies which column of the identity
    /// matrix to use.
    permutation: Vec<usize>,
    work: DVector<f64>,
}
impl Display for PivotedQR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PivotedQR {{")?;
        writeln!(f, "column_norms: {:?}", self.column_norms)?;
        writeln!(f, "qr: {:?}", self.qr)?;
        writeln!(f, "r_diag: {:?}", self.r_diag)?;
        writeln!(f, "permutation: {:?}", self.permutation)?;
        writeln!(f, "work: {:?}", self.work)
    }
}
impl PivotedQR {
    /// Create a pivoted QR decomposition of a matrix `A`.
    pub fn new(mut a: DMatrix<f64>) -> Self {
        let (m, n) = a.shape();
        let column_norms = DVector::from_iterator(n, a.column_iter().map(|c| enorm(&c)));
        //   println!("column_norms: {:?}", column_norms);
        let mut r_diag = column_norms.clone();
        let mut work = column_norms.clone();
        let mut permutation: Vec<usize> = (0..n).collect();

        for j in 0..m.min(n) {
            // pivot
            let kmax = r_diag.view_range(j.., ..).imax() + j;
            if kmax != j {
                a.swap_columns(j, kmax);
                permutation.swap(j, kmax);
                r_diag[kmax] = r_diag[j];
                work[kmax] = work[j];
            }

            // compute Householder reflection vector w_j to
            // reduce the j-th column
            let mut lower = a.rows_range_mut(j..);
            let (left, mut right) = lower.columns_range_pair_mut(j, j + 1..);
            let w_j = {
                let mut axis = left;
                let mut aj_norm = enorm(&axis);
                if aj_norm == 0.0 {
                    r_diag[j] = 0.0;
                    continue;
                }
                if axis[0] < 0.0 {
                    aj_norm = -aj_norm;
                }
                r_diag[j] = -aj_norm;
                axis /= aj_norm;
                axis[0] += 1.0;
                axis
            };

            // apply reflection to remaining rows
            for (k, mut col) in right.column_iter_mut().enumerate() {
                let k = k + j + 1;
                col.axpy(-(dot(&col, &w_j) / w_j[0]), &w_j, 1.0);

                // update partial column norms
                // see "Lapack Working Note 176"
                if r_diag[k] == 0.0 {
                    continue;
                }
                let r_diagk = &mut r_diag[k];
                *r_diagk *= {
                    let temp = (col[0] / *r_diagk).powi(2);
                    (1.0 - temp).max(0.0).sqrt()
                };
                let z05 = 0.05;
                if z05 * (*r_diagk / work[k]).powi(2) <= epsmch::<f64>() {
                    *r_diagk = enorm(&col.view_range(1.., ..));
                    work[k] = *r_diagk;
                }
            }
        }

        Self {
            column_norms,
            qr: a,
            permutation,
            r_diag,
            work,
        }
    }

    /// Consume the QR-decomposition and transform it into
    /// a parametrized least squares problem.
    pub fn into_least_squares_diagonal_problem(
        mut self,
        mut b: DVector<f64>,
    ) -> LinearLeastSquaresDiagonalProblem {
        // compute first n-entries of Q^T * b
        let (m, n) = self.qr.shape();
        let mut qt_b = DVector::zeros(n);

        for j in 0..m.min(n) {
            let axis = self.qr.view_range(j.., j);
            if axis[0] != 0.0 {
                let temp = -dot(&b.rows_range(j..), &axis) / axis[0];
                b.rows_range_mut(j..).axpy(temp, &axis, 1.0);
            }
            if j < qt_b.len() {
                qt_b[j] = b[j];
            }
        }

        // Set diagonal of qr matrix
        for j in 0..m.min(n) {
            if j < self.r_diag.len() {
                self.qr[(j, j)] = self.r_diag[j];
            }
        }

        LinearLeastSquaresDiagonalProblem {
            qt_b,
            column_norms: self.column_norms,
            upper_r: self.qr.resize(m.max(n), n, 0.0),
            l_diag: self.r_diag,
            permutation: self.permutation,
            work: self.work,
            m,
        }
    }
}

/// Parametrized linear least squares problem for the LM algorithm.
///
/// The problem is of the form
///
///   min_{x} (1/2) * ||[A; D] * x - [b; 0]||^2
///
/// for a matrix `A`, diagonal matrix `D` and vector `b`.
/// Everything except the diagonal matrix `D` is considered fixed.
pub struct LinearLeastSquaresDiagonalProblem {
    /// The first `n` entries of `Q^T * b`.
    qt_b: DVector<f64>,
    /// Upper part of `R`, also used to store strictly lower part of `L`.
    upper_r: DMatrix<f64>,
    /// Diagonal entries of `L`.
    l_diag: DVector<f64>,
    /// Permutation matrix. Entry `i` specifies which column of the identity
    /// matrix to use.
    permutation: Vec<usize>,
    pub(crate) column_norms: DVector<f64>,
    work: DVector<f64>,
    m: usize,
}

pub struct CholeskyFactor<'a> {
    pub permutation: &'a Vec<usize>,
    l: &'a DMatrix<f64>,
    work: &'a mut DVector<f64>,
    qt_b: &'a DVector<f64>,
    lower: bool,
    l_diag: &'a DVector<f64>,
}

impl<'a> CholeskyFactor<'a> {
    /// Solve the equation `L * x = P^T * b`.
    pub fn solve(&mut self, mut rhs: DVector<f64>) -> DVector<f64> {
        for i in 0..self.work.nrows() {
            if i < self.permutation.len() && self.permutation[i] < rhs.len() {
                self.work[i] = rhs[self.permutation[i]];
            }
        }

        let n = self.work.nrows();
        let l = self.l.view_range(0..n, 0..n);

        if self.lower {
            for j in 0..n {
                if j < self.l_diag.len() && self.l_diag[j] != 0.0 {
                    let x = self.work[j] / self.l_diag[j];
                    self.work[j] = x;

                    for i in (j + 1)..n {
                        if i < l.nrows() && j < l.ncols() {
                            self.work[i] -= x * l[(i, j)];
                        }
                    }
                }
            }
        } else {
            for j in 0..n {
                let mut sum = 0.0;
                for i in 0..j {
                    if i < l.nrows() && j < l.ncols() {
                        sum += self.work[i] * l[(i, j)];
                    }
                }
                if j < l.nrows() && j < l.ncols() && l[(j, j)] != 0.0 {
                    self.work[j] = (self.work[j] - sum) / l[(j, j)];
                }
            }
        }

        std::mem::swap(self.work, &mut rhs);
        rhs
    }

    /// Computes `L * Q^T * b`.
    pub fn mul_qt_b(&mut self, mut out: DVector<f64>) -> DVector<f64> {
        out.fill(0.0);
        let n = self.work.nrows();
        let l = self.l.view_range(0..n, 0..n);

        if self.lower {
            for i in 0..n {
                if i < self.qt_b.len() && i < self.l_diag.len() {
                    for j in (i + 1)..n {
                        if j < out.len() && i < l.ncols() && j < l.nrows() {
                            out[j] += self.qt_b[i] * l[(j, i)];
                        }
                    }
                    if i < out.len() {
                        out[i] += self.qt_b[i] * self.l_diag[i];
                    }
                }
            }
        } else {
            for i in 0..n {
                if i < out.len() && i < self.qt_b.len() {
                    let mut sum = 0.0;
                    for j in 0..(i + 1) {
                        if j < self.qt_b.len() && j < l.nrows() && i < l.ncols() {
                            sum += self.qt_b[j] * l[(j, i)];
                        }
                    }
                    out[i] = sum;
                }
            }
        }
        out
    }
}
// is_non_singular,  r_rank, rank, solve_after_elimination, eliminate_diag not providied in the rewritten code
impl LinearLeastSquaresDiagonalProblem {
    /// Compute scaled maximum of dot products between `b` and the columns of `A`.
    pub fn max_a_t_b_scaled(&mut self, b_norm: f64) -> Option<f64> {
        let b = &mut self.work;
        b.copy_from(&self.qt_b);
        *b /= b_norm;

        let mut max = 0.0;
        for (j, col) in self.upper_r.column_iter().enumerate() {
            if j < self.permutation.len() && self.permutation[j] < self.column_norms.len() {
                let scale = self.column_norms[self.permutation[j]];
                if scale == 0.0 {
                    continue;
                }
                let sum = dot(&col.rows_range(..j + 1), &b.rows_range(..j + 1));
                let temp = (sum / scale).abs();
                if temp.is_nan() {
                    return None;
                }
                max = max.max(temp);
            }
        }
        Some(max)
    }

    /// Compute `||A * x||`.
    pub fn a_x_norm(&mut self, x: &DVector<f64>) -> f64 {
        self.work.fill(0.0);
        for (i, (col, &idx)) in self
            .upper_r
            .column_iter()
            .zip(self.permutation.iter())
            .enumerate()
        {
            if idx < x.len() {
                self.work
                    .rows_range_mut(..i + 1)
                    .axpy(x[idx], &col.rows_range(..i + 1), 1.0);
            }
        }
        enorm(&self.work)
    }

    /// Solve the linear least squares problem for a diagonal matrix `D` (`diag`).
    pub fn solve_with_diagonal(
        &mut self,
        diag: &DVector<f64>,
        mut out: DVector<f64>,
    ) -> (DVector<f64>, CholeskyFactor) {
        out.copy_from(&self.qt_b);
        let mut rhs = self.eliminate_diag(diag, out);
        //  println!("rhs: {:?}", rhs);
        std::mem::swap(&mut self.work, &mut rhs);
        self.solve_after_elimination(rhs)
    }

    /// Solve the least squares problem with a zero diagonal.
    pub fn solve_with_zero_diagonal(&mut self) -> (DVector<f64>, CholeskyFactor) {
        let n = self.upper_r.ncols();
        let l = self.upper_r.view_range(0..n, 0..n);
        self.work.copy_from(&self.qt_b);
        let rank = self.r_rank();

        for i in rank..self.work.len() {
            self.work[i] = 0.0;
        }

        // Solve upper triangular system
        for i in (0..rank).rev() {
            let mut sum = 0.0;
            for j in (i + 1)..rank {
                if i < l.nrows() && j < l.ncols() {
                    sum += l[(i, j)] * self.work[j];
                }
            }
            if i < l.nrows() && i < l.ncols() && l[(i, i)] != 0.0 {
                self.work[i] = (self.work[i] - sum) / l[(i, i)];
            }
        }

        let mut x = DVector::zeros(n);
        for j in 0..n {
            if j < self.permutation.len() && self.permutation[j] < x.len() {
                x[self.permutation[j]] = self.work[j];
            }
        }

        let chol = CholeskyFactor {
            permutation: &self.permutation,
            l: &self.upper_r,
            work: &mut self.work,
            qt_b: &self.qt_b,
            lower: false,
            l_diag: &self.l_diag,
        };
        (x, chol)
    }
    /// Compute if the matrix A has full rank.
    pub fn is_non_singular(&self) -> bool {
        let n = self.upper_r.ncols();
        let max_rank = self.m.min(n);
        max_rank == n && !(0..n).any(|j| self.upper_r[(j, j)] == 0.0)
    }

    fn r_rank(&self) -> usize {
        let n = self.upper_r.ncols();
        let max_rank = self.m.min(n);
        (0..max_rank)
            .position(|i| self.upper_r[(i, i)] == 0.0)
            .unwrap_or(max_rank)
    }

    fn rank(&self) -> usize {
        self.l_diag
            .iter()
            .position(|&x| x == 0.0)
            .unwrap_or(self.l_diag.nrows())
    }

    fn solve_after_elimination(&mut self, mut x: DVector<f64>) -> (DVector<f64>, CholeskyFactor) {
        let rank = self.rank();
        let rhs = &mut self.work;

        // Fill remaining elements with zero
        for i in rank..rhs.len() {
            rhs[i] = 0.0;
        }

        let n = self.upper_r.ncols();
        let l = self.upper_r.view_range(0..n, 0..n);

        // solve L^T * x = rhs
        for j in (0..rank).rev() {
            let mut dot_product = 0.0;
            for i in (j + 1)..rank {
                if i < l.nrows() && j < l.ncols() {
                    dot_product += l[(i, j)] * rhs[i];
                }
            }
            if j < self.l_diag.len() && self.l_diag[j] != 0.0 {
                rhs[j] = (rhs[j] - dot_product) / self.l_diag[j];
            }
        }

        // Apply permutation
        for j in 0..n {
            if j < self.permutation.len() && self.permutation[j] < x.len() && j < rhs.len() {
                x[self.permutation[j]] = rhs[j];
            }
        }

        let cholesky_factor = CholeskyFactor {
            l: &self.upper_r,
            work: &mut self.work,
            permutation: &self.permutation,
            qt_b: &self.qt_b,
            lower: true,
            l_diag: &self.l_diag,
        };
        (x, cholesky_factor)
    }

    fn eliminate_diag(&mut self, diag: &DVector<f64>, mut rhs: DVector<f64>) -> DVector<f64> {
        let n = self.upper_r.ncols();
        //     println!("self.upper_ r: {}, shape {:?}", self.upper_r, self.upper_r.shape());
        // only lower triangular part of self.upper_r is used in this function
        // we fill it now with R^T which is then iteratively overwritten with L.
        let mut r_and_l = self.upper_r.view_range_mut(0..n, 0..n);
        //    println!("r_and_l: {:?}", r_and_l);
        r_and_l.fill_lower_triangle_with_upper_triangle();

        // save diagonal of R so we can restore it later.
        for j in 0..n {
            if j < self.work.len() {
                self.work[j] = r_and_l[(j, j)];
            }
        }

        // eliminate the diagonal entries from D using Givens rotations
        let p5 = 0.5;
        let p25 = 0.25;

        for j in 0..n {
            let diag_entry = if j < self.permutation.len() && self.permutation[j] < diag.len() {
                diag[self.permutation[j]]
            } else {
                0.0
            };

            if diag_entry != 0.0 {
                self.l_diag[j] = diag_entry;
                for i in (j + 1)..n {
                    if i < self.l_diag.len() {
                        self.l_diag[i] = 0.0;
                    }
                }

                let mut qtbpj = 0.0;
                for k in j..n {
                    if k < self.l_diag.len() && self.l_diag[k] != 0.0 {
                        // determine the Givens rotation
                        let r_kk = r_and_l[(k, k)];
                        let (sin, cos) = if r_kk.abs() < self.l_diag[k].abs() {
                            let cot = r_kk / self.l_diag[k];
                            let sin = p5 / (p25 + p25 * (cot * cot)).sqrt();
                            (sin, sin * cot)
                        } else {
                            let tan = self.l_diag[k] / r_kk;
                            let cos = p5 / (p25 + p25 * (tan * tan)).sqrt();
                            (cos * tan, cos)
                        };

                        // compute the modified diagonal element of R and (Q^T*b,0)
                        r_and_l[(k, k)] = cos * r_kk + sin * self.l_diag[k];
                        let temp = cos * rhs[k] + sin * qtbpj;
                        qtbpj = -sin * rhs[k] + cos * qtbpj;
                        rhs[k] = temp;

                        // accumulate the transformation in the row of L
                        for i in (k + 1)..n {
                            if i < self.l_diag.len() {
                                let r_ik = r_and_l[(i, k)];
                                let temp = cos * r_ik + sin * self.l_diag[i];
                                self.l_diag[i] = -sin * r_ik + cos * self.l_diag[i];
                                r_and_l[(i, k)] = temp;
                            }
                        }
                    }
                }
            }
            self.l_diag[j] = r_and_l[(j, j)];
            r_and_l[(j, j)] = self.work[j];
        }
        rhs
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/// TESTING
/////////////////////////////////////////////////////////////////////////////////////////////////
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    #[test]
    fn test_pivoted_qr() {
        #[rustfmt::skip]
    let a = DMatrix::<f64>::from_row_slice(4, 3, &[
        2.0,  1.,  4.0,
        0.0, 10., -1.0,
        0.0,  4.,  0.5,
        1.0,  0.,   0.,
    ]);
        let qr = PivotedQR::new(a);

        assert_eq!(qr.permutation, vec![1, 2, 0]);

        let column_norms = DVector::from_vec(vec![
            2.23606797749979,
            10.816653826391969,
            4.153311931459037,
        ]);
        assert_relative_eq!(qr.column_norms, column_norms);

        let r_diag = DVector::from_vec(vec![
            -10.816653826391967,
            4.1368161505254095,
            1.0778765953488594,
        ]);
        assert_relative_eq!(qr.r_diag, r_diag);

        #[rustfmt::skip]
    let qr_ref = DMatrix::<f64>::from_row_slice(4, 3, &[
        1.0924500327042048 ,  0.3698001308168193 , -0.18490006540840964,
        0.9245003270420484 ,  1.9843572039046236 ,  1.9503830421256012 ,
        0.3698001308168194 ,  0.17618426468067802,  1.3732023003846023 ,
        0.                 , -0.                 , -0.9277499894840426 ,
    ]);
        assert_relative_eq!(qr.qr, qr_ref, epsilon = 1e-14);
    }

    #[test]
    /// Test that for a wide matrix the QR is identical to the case
    /// where the matrix is extended with zero rows.
    fn test_wide_matrix() {
        #[rustfmt::skip]
    let a1 = DMatrix::from_row_slice(2, 4, &[
        6., 4., 9., 8.,
        4., 0., 8., 7.,
    ]);
        #[rustfmt::skip]
    let a2 = DMatrix::from_row_slice(4, 4, &[
        6., 4., 9., 8.,
        4., 0., 8., 7.,
        0., 0., 0., 0.,
        0., 0., 0., 0.,
    ]);
        let qr1 = PivotedQR::new(a1);
        let qr2 = PivotedQR::new(a2);
        assert_eq!(qr1.permutation, qr2.permutation);
        assert_relative_eq!(qr1.column_norms, qr2.column_norms);
        assert_relative_eq!(qr1.r_diag, qr2.r_diag);
        let qr_ref = qr2.qr.clone().remove_rows(2, 2);
        assert_relative_eq!(qr1.qr.as_slice(), qr_ref.as_slice());

        let lls1 = qr1.into_least_squares_diagonal_problem(DVector::from_vec(vec![7., -5.]));
        let lls2 =
            qr2.into_least_squares_diagonal_problem(DVector::from_vec(vec![7., -5., 0., 0.]));
        assert_relative_eq!(lls1.qt_b, lls2.qt_b);

        #[rustfmt::skip]
    let a1 = DMatrix::from_row_slice(2, 4, &[
        6., 0., 0., 0.,
        4., 0., 0., 0.,
    ]);
        #[rustfmt::skip]
    let a2 = DMatrix::from_row_slice(4, 4, &[
        6., 0., 0., 0.,
        4., 0., 0., 0.,
        0., 0., 0., 0.,
        0., 0., 0., 0.,
    ]);
        let qr1 = PivotedQR::new(a1);
        let qr2 = PivotedQR::new(a2);
        assert_eq!(qr1.permutation, qr2.permutation);
        assert_relative_eq!(qr1.column_norms, qr2.column_norms);
        assert_relative_eq!(qr1.r_diag, qr2.r_diag);
        let qr_ref = qr2.qr.clone().remove_rows(2, 2);
        assert_relative_eq!(qr1.qr.as_slice(), qr_ref.as_slice());

        let lls1 = qr1.into_least_squares_diagonal_problem(DVector::from_vec(vec![7., -5.]));
        let lls2 =
            qr2.into_least_squares_diagonal_problem(DVector::from_vec(vec![7., -5., 0., 0.]));
        assert_relative_eq!(lls1.qt_b, lls2.qt_b);
    }

    #[test] // FAILS
    fn test_pivoted_qr_more_branches() {
        // This test case was crafted to hit all three
        // branches of the partial column norms
        let a = DMatrix::<f64>::from_row_slice(
            4,
            3,
            &[
                30.0, 30.0, 24.0, 43.0, 43.0, 39.0, 34.0, 34.0, -10.0, 26.0, 26.0, -34.0,
            ],
        );
        println!("a: {}, {}", a, a.nrows());
        let qr = PivotedQR::new(a);
        println!("qr: {:?}", qr);
        let r_diag = DVector::from_vec(vec![
            -67.683085036070864,
            -55.250741178610944,
            0.00000000000001,
        ]);
        assert_relative_eq!(qr.r_diag, r_diag, epsilon = 1e-14);
    }

    #[test]
    fn test_pivoted_qr_big_rank1() {
        // This test case was generated directly from MINPACK's QRFAC
        let a = DMatrix::<f64>::from_fn(10, 5, |i, j| ((i + 1) * (j + 1)) as f64);
        let qr = PivotedQR::new(a);
        let r_diag = DVector::<f64>::from_vec(vec![
            -98.107084351742913,
            -3.9720546451956370E-015,
            0.,
            0.,
            0.,
        ]);
        assert_relative_eq!(qr.r_diag, r_diag);
        #[rustfmt::skip]
    let qr_ref = DMatrix::<f64>::from_column_slice(10, 5, &[
        // matrix looks transposed in this form, this is a column slice!
        // column 1
          1.0509647191437625 , 0.10192943828752511, 0.15289415743128767, 0.20385887657505022,  0.25482359571881280,
          0.30578831486257535, 0.35675303400633790, 0.40771775315010045, 0.45868247229386300,  0.50964719143762560,
        // column 2
        -58.864250611045748  , 1.0000000000000000 , 0.0000000000000000 , 0.0000000000000000 , -0.44721359549995793,
          0.0000000000000000 , 0.0000000000000000 , 0.0000000000000000 , 0.0000000000000000 , -0.89442719099991586,
        // column 3
        -39.242833740697165  , 0.0000000000000000 , 0.0000000000000000 , 0.0000000000000000 ,  0.0000000000000000 ,
          0.0000000000000000 , 0.0000000000000000 , 0.0000000000000000 , 0.0000000000000000 ,  0.0000000000000000 ,
        // column 4
        -78.485667481394330  , 0.0000000000000000 , 0.0000000000000000 , 0.0000000000000000 ,  0.0000000000000000 ,
          0.0000000000000000 , 0.0000000000000000 , 0.0000000000000000 , 0.0000000000000000 ,  0.0000000000000000 ,
        // column 5
        -19.621416870348583  , 0.0000000000000000 , 0.0000000000000000 , 0.0000000000000000 ,  0.0000000000000000 ,
          0.0000000000000000 , 0.0000000000000000 , 0.0000000000000000 , 0.0000000000000000 ,  0.0000000000000000 ,
    ]);
        assert_relative_eq!(qr.qr, qr_ref);
    }

    #[cfg(test)]
    fn default_lls(case: usize) -> LinearLeastSquaresDiagonalProblem {
        let a = match case {
            1 => DMatrix::<f64>::from_iterator(4, 3, (0..12).map(|i| i as f64)),
            2 => DMatrix::<f64>::from_row_slice(
                4,
                3,
                &[30., 43., 34., 26., 30., 43., 34., 26., 24., 39., -10., -34.],
            ),
            3 => DMatrix::from_row_slice(4, 3, &[1., 2., -1., 0., 1., 4., 0., 0., 0.5, 0., 0., 0.]),
            _ => unimplemented!(),
        };
        let qr = PivotedQR::new(a);
        qr.into_least_squares_diagonal_problem(DVector::from_vec(vec![1.0, 2.0, 5.0, 4.0]))
    }

    #[test]
    fn test_into_lls() {
        // data was generated with Python implementation "lmmin" and SciPy MINPACK binding
        #[rustfmt::skip]
    let a = DMatrix::<f64>::from_row_slice(4, 3, &[
        2.0,  1.,  4.0,
        0.0, 10., -1.0,
        0.0,  4.,  0.5,
        1.0,  0.,   0.,
    ]);
        let qr = PivotedQR::new(a);
        let lls =
            qr.into_least_squares_diagonal_problem(DVector::from_vec(vec![1.0, 2.0, 5.0, 4.0]));
        let qt_b = DVector::from_vec(vec![
            -3.790451340872398,
            1.4266308163005572,
            2.334839404175348,
        ]);
        assert_relative_eq!(lls.qt_b, qt_b, epsilon = 1e-14);
    }

    #[test]
    fn test_elimate_diag_and_l() {
        let mut lls = default_lls(1);
        let rhs = lls.eliminate_diag(&DVector::from_vec(vec![1.0, 0.5, 0.0]), lls.qt_b.clone());
        let rhs_ref = DVector::from_vec(vec![
            -6.272500481871799,
            1.731584982206922,
            0.612416936078506,
        ]);
        assert_relative_eq!(rhs, rhs_ref);

        // contains L
        let ldiag_ref = DVector::from_vec(vec![
            -19.131126469708992,
            2.120676250530203,
            0.666641352293790,
        ]);
        assert_relative_eq!(lls.l_diag, ldiag_ref);

        let r_ref = DMatrix::from_row_slice(
            3,
            3,
            &[
                -19.131126469708992,
                -3.240791915633763,
                -11.185959192671376,
                -3.240791915633763,
                1.870098328848738,
                0.935049164424371,
                -11.185959192671376,
                0.824564277241393,
                -0.000000000000001,
            ],
        );
        let r = DMatrix::from_iterator(3, 3, lls.upper_r.view_range(..3, ..3).iter().copied());
        assert_relative_eq!(r, r_ref);
    }

    #[test]
    fn test_lls_x_1() {
        let mut lls = default_lls(1);
        let (x_out, _) =
            lls.solve_with_diagonal(&DVector::from_vec(vec![1.0, 0.5, 0.0]), DVector::zeros(3));
        let x_ref = DVector::from_vec(vec![
            0.459330143540669,
            0.918660287081341,
            -0.287081339712919,
        ]);
        assert_relative_eq!(x_out, x_ref, epsilon = 1e-14);
    }

    #[test]
    fn test_lls_x_2() {
        // R is singular but L is not
        let a = DMatrix::from_column_slice(
            4,
            3,
            &[
                14., -12., 20., -11., 19., 38., -4., -11., -14., 12., -20., 11.,
            ],
        );
        let qr = PivotedQR::new(a);
        let mut lls =
            qr.into_least_squares_diagonal_problem(DVector::from_vec(vec![-5., 3., -2., 7.]));

        let rdiag_exp = DVector::from_vec(vec![-44.068129073061407, 29.147349299100057, 0.]);
        let rdiag_out = DVector::from_iterator(
            3,
            lls.upper_r.view_range(..3, ..3).diagonal().iter().copied(),
        );
        assert_relative_eq!(rdiag_out, rdiag_exp);

        let diag = DVector::from_vec(vec![
            2.772724292099739,
            0.536656314599949,
            0.089442719099992,
        ]);
        let (x_out, _) = lls.solve_with_diagonal(&diag, DVector::zeros(3));
        let x_exp = DVector::from_vec(vec![
            -0.000277544878320,
            -0.046225239392197,
            0.266720628065249,
        ]);
        assert_relative_eq!(x_out, x_exp, epsilon = 1e-14);
    }

    #[test]
    fn test_lls_wide_matrix() {
        #[rustfmt::skip]
    let a1 = DMatrix::from_row_slice(2, 4, &[
        6., 4., 9., 8.,
        4., 0., 8., 7.,
    ]);
        println!("a1, {}, nrows: {}, ncols: {}", a1, a1.nrows(), a1.ncols());
        #[rustfmt::skip]
    let a2 = DMatrix::from_row_slice(4, 4, &[
        6., 4., 9., 8.,
        4., 0., 8., 7.,
        0., 0., 0., 0.,
        0., 0., 0., 0.,
    ]);
        println!("a2, {}, nrows: {}, ncols: {}", a2, a2.nrows(), a2.ncols());
        let mut lls1 = PivotedQR::new(a1)
            .into_least_squares_diagonal_problem(DVector::from_vec(vec![23., -1.]));
        let mut lls2 = PivotedQR::new(a2)
            .into_least_squares_diagonal_problem(DVector::from_vec(vec![23., -1., 0., 0.]));
        println!(
            "lls1, {}, nrows: {}, ncols: {}",
            lls1.upper_r,
            lls1.upper_r.nrows(),
            lls1.upper_r.ncols()
        );
        println!(
            "lls2, {}, nrows: {}, ncols: {}",
            lls2.upper_r,
            lls2.upper_r.nrows(),
            lls2.upper_r.ncols()
        );
        let diag = DVector::from_vec(vec![1., 2., 8., 0.5]);
        let b = DVector::from_vec(vec![0.6301, 0.1611, 0.9104, 0.8998]);
        let (x1, mut chol1) = lls1.solve_with_diagonal(&diag, DVector::zeros(4));
        let (x2, mut chol2) = lls2.solve_with_diagonal(&diag, DVector::zeros(4));
        assert_relative_eq!(chol1.solve(b.clone()), chol2.solve(b));
        assert_relative_eq!(
            chol1.mul_qt_b(DVector::zeros(4)),
            chol2.mul_qt_b(DVector::zeros(4))
        );
        assert_relative_eq!(lls1.upper_r, lls2.upper_r);
        assert_relative_eq!(x1, x2);

        let diag = DVector::from_vec(vec![0.1, 20., 8.2, 1.5]);
        let b = DVector::from_vec(vec![0.851, 0.21, 0.629, 0.714]);
        let (x1, mut chol1) = lls1.solve_with_diagonal(&diag, DVector::zeros(4));
        let (x2, mut chol2) = lls2.solve_with_diagonal(&diag, DVector::zeros(4));
        assert_relative_eq!(chol1.solve(b.clone()), chol2.solve(b));
        assert_relative_eq!(
            chol1.mul_qt_b(DVector::zeros(4)),
            chol2.mul_qt_b(DVector::zeros(4))
        );
        assert_relative_eq!(lls1.upper_r, lls2.upper_r);
        assert_relative_eq!(x1, x2);

        let (x1, mut chol1) = lls1.solve_with_zero_diagonal();
        let (x2, mut chol2) = lls2.solve_with_zero_diagonal();
        assert_relative_eq!(
            chol1.mul_qt_b(DVector::zeros(4)),
            chol2.mul_qt_b(DVector::zeros(4))
        );
        assert_relative_eq!(lls1.upper_r, lls2.upper_r);
        assert_relative_eq!(x1, x2);
    }

    #[test]
    fn test_lls_zero_diagonal() {
        let mut lls = default_lls(3);
        assert!(lls.is_non_singular());
        let (x_out, _l) = lls.solve_with_zero_diagonal();
        let x_ref = DVector::from_vec(vec![87., -38., 10.]);
        assert_relative_eq!(x_out, x_ref);
    }

    #[test]
    fn test_cholesky_lower() {
        let l = DMatrix::from_row_slice(3, 3, &[-1.0e10, 100., -1., 1., 1.0e8, 0.5, 1., 0.5, 100.]);
        let mut chol = CholeskyFactor {
            l: &l,
            l_diag: &DVector::from_vec(vec![2., 1.5, 0.1]),
            lower: true,
            work: &mut DVector::zeros(3),
            permutation: &vec![1, 0, 2],
            qt_b: &DVector::from_vec(vec![1.0, 2.0, 0.5]),
        };

        let out_mul = chol.mul_qt_b(DVector::zeros(3));
        let exp_mul = DVector::from_vec(vec![2., 4., 2.05]);
        assert_relative_eq!(out_mul, exp_mul);

        let out_solve = chol.solve(DVector::from_vec(vec![1.0, 2.0, 0.5]));
        let exp_solve = DVector::from_vec(vec![1., 0., -5.]);
        assert_relative_eq!(out_solve, exp_solve);
    }

    #[test]
    fn test_cholesky_upper() {
        let l = DMatrix::from_row_slice(3, 3, &[4., 7., 1., 123., 6., 8., 34., 34455., 9.]);
        let mut chol = CholeskyFactor {
            l: &l,
            l_diag: &DVector::from_vec(vec![1234.0, -1.5, -1e120]),
            lower: false,
            work: &mut DVector::zeros(3),
            permutation: &vec![2, 1, 0],
            qt_b: &DVector::from_vec(vec![1.0, 2.0, 0.5]),
        };

        let out_mul = chol.mul_qt_b(DVector::zeros(3));
        let exp_mul = DVector::from_vec(vec![4., 19., 21.5]);
        assert_relative_eq!(out_mul, exp_mul);

        let out_solve = chol.solve(DVector::from_vec(vec![1.0, 2.0, 0.5]));
        let exp_solve = DVector::from_vec(vec![0.125, 0.1875, -0.06944444444444445]);
        assert_relative_eq!(out_solve, exp_solve);
    }

    #[test]
    fn test_column_max_norm() {
        let a = DMatrix::from_column_slice(
            4,
            3,
            &[
                14., -12., 20., -11., 19., 38., -4., -11., -14., 12., -20., 11.,
            ],
        );
        let qr = PivotedQR::new(a);
        let b = DVector::from_vec(vec![1., 2., 3., 4.]);
        let max_at_b = qr
            .into_least_squares_diagonal_problem(b)
            .max_a_t_b_scaled(1.);
        assert_relative_eq!(max_at_b.unwrap(), 0.88499332, epsilon = 1e-8);

        let a = DMatrix::from_column_slice(
            4,
            3,
            &[
                f64::NAN,
                -12.,
                20.,
                -11.,
                19.,
                38.,
                -4.,
                -11.,
                -14.,
                12.,
                -20.,
                11.,
            ],
        );
        let qr = PivotedQR::new(a);
        let b = DVector::from_vec(vec![1., 2., 3., 4.]);
        let max_at_b = qr
            .into_least_squares_diagonal_problem(b)
            .max_a_t_b_scaled(1.);
        assert_eq!(max_at_b, None);

        let a = DMatrix::zeros(4, 3);
        let qr = PivotedQR::new(a);
        let b = DVector::from_vec(vec![1., 2., 3., 4.]);
        let max_at_b = qr
            .into_least_squares_diagonal_problem(b)
            .max_a_t_b_scaled(1.);
        assert_eq!(max_at_b, Some(0.));
    }

    #[test]
    fn test_a_x_norm() {
        let a = DMatrix::from_row_slice(4, 3, &[3., 6., 2., 7., 4., 3., 2., 0., 4., 5., 1., 6.]);
        let qr = PivotedQR::new(a);
        let mut lls = qr.into_least_squares_diagonal_problem(DVector::zeros(4));
        let result = lls.a_x_norm(&DVector::from_vec(vec![1., 8., 3.]));
        assert_relative_eq!(result, (6710.0_f64).sqrt());
    }
}
