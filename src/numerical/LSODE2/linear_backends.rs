//! Newton linear backend adapters for the LSODE2-on-BDF bridge.
//!
//! These adapters intentionally sit behind `BDF`'s narrow factor/solve trait.
//! They support two modes:
//! - the current bridge mode, where legacy BDF gives them an already-dense
//!   Newton matrix;
//! - the native LSODE2 path, where BDF passes a Jacobian representation and the
//!   backend forms `[I - cJ]` directly in sparse or banded storage.

use crate::numerical::BDF::BDF_solver::{BdfJacobian, BdfLinearBackend, BdfLinearFactorization};
use crate::somelinalg::banded::lapack_style_banded::LapackStyleBandedLuFaithful;
use crate::somelinalg::banded::solver_traits::{DirectLinearSolver, FaerSparseLuSolver};
use crate::somelinalg::banded::storage::Banded;
use faer::sparse::Triplet;
use nalgebra::{DMatrix, DVector, Dyn, LU};

const DEFAULT_DROP_TOL: f64 = 0.0;

/// Dense LU backend based on `nalgebra` LU factorization.
#[derive(Debug, Clone, Default)]
pub struct DenseLuBdfLinearBackend;

impl BdfLinearBackend for DenseLuBdfLinearBackend {
    fn factor(&mut self, matrix: &DMatrix<f64>) -> Option<Box<dyn BdfLinearFactorization>> {
        if matrix.nrows() != matrix.ncols() {
            return None;
        }
        Some(Box::new(DenseLuFactorization {
            lu: LU::new(matrix.clone()),
        }))
    }

    fn factor_shifted_jacobian(
        &mut self,
        c: f64,
        jacobian: &BdfJacobian,
    ) -> Option<Box<dyn BdfLinearFactorization>> {
        let shifted = jacobian.to_shifted_dense(c)?;
        self.factor(&shifted)
    }
}

struct DenseLuFactorization {
    lu: LU<f64, Dyn, Dyn>,
}

impl BdfLinearFactorization for DenseLuFactorization {
    fn solve(&self, rhs: &DVector<f64>) -> Option<DVector<f64>> {
        self.lu.solve(rhs)
    }
}

/// Sparse LU backend based on `faer` sparse factorization.
#[derive(Debug, Clone)]
pub struct FaerSparseBdfLinearBackend {
    drop_tol: f64,
}

impl Default for FaerSparseBdfLinearBackend {
    fn default() -> Self {
        Self {
            drop_tol: DEFAULT_DROP_TOL,
        }
    }
}

impl FaerSparseBdfLinearBackend {
    pub fn with_drop_tol(mut self, drop_tol: f64) -> Self {
        self.drop_tol = drop_tol.max(0.0);
        self
    }
}

impl BdfLinearBackend for FaerSparseBdfLinearBackend {
    fn factor(&mut self, matrix: &DMatrix<f64>) -> Option<Box<dyn BdfLinearFactorization>> {
        let n = square_dimension(matrix)?;
        let triplets = dense_to_triplets(matrix, self.drop_tol);
        let solver = FaerSparseLuSolver::from_triplets(n, &triplets).ok()?;
        Some(Box::new(DirectSolverFactorization { solver }))
    }

    fn factor_shifted_jacobian(
        &mut self,
        c: f64,
        jacobian: &BdfJacobian,
    ) -> Option<Box<dyn BdfLinearFactorization>> {
        let n = jacobian.n();
        let triplets = jacobian.to_shifted_sparse_triplets(c)?;
        let solver = FaerSparseLuSolver::from_triplets(n, &triplets).ok()?;
        Some(Box::new(DirectSolverFactorization { solver }))
    }
}

/// Faithful LAPACK-style banded LU backend.
#[derive(Debug, Clone)]
pub struct FaithfulBandedBdfLinearBackend {
    drop_tol: f64,
}

impl Default for FaithfulBandedBdfLinearBackend {
    fn default() -> Self {
        Self {
            drop_tol: DEFAULT_DROP_TOL,
        }
    }
}

impl FaithfulBandedBdfLinearBackend {
    pub fn with_drop_tol(mut self, drop_tol: f64) -> Self {
        self.drop_tol = drop_tol.max(0.0);
        self
    }
}

impl BdfLinearBackend for FaithfulBandedBdfLinearBackend {
    fn factor(&mut self, matrix: &DMatrix<f64>) -> Option<Box<dyn BdfLinearFactorization>> {
        let banded = dense_to_banded(matrix, self.drop_tol)?;
        let mut solver =
            LapackStyleBandedLuFaithful::new(banded.n(), banded.kl(), banded.ku()).ok()?;
        solver.factor_from(&banded).ok()?;
        Some(Box::new(DirectSolverFactorization { solver }))
    }

    fn factor_shifted_jacobian(
        &mut self,
        c: f64,
        jacobian: &BdfJacobian,
    ) -> Option<Box<dyn BdfLinearFactorization>> {
        let banded = jacobian.to_shifted_banded(c)?;
        let mut solver =
            LapackStyleBandedLuFaithful::new(banded.n(), banded.kl(), banded.ku()).ok()?;
        solver.factor_from(&banded).ok()?;
        Some(Box::new(DirectSolverFactorization { solver }))
    }
}

struct DirectSolverFactorization<S> {
    solver: S,
}

impl<S> BdfLinearFactorization for DirectSolverFactorization<S>
where
    S: DirectLinearSolver,
{
    fn solve(&self, rhs: &DVector<f64>) -> Option<DVector<f64>> {
        let mut out = rhs.as_slice().to_vec();
        self.solver.solve_in_place(&mut out).ok()?;
        Some(DVector::from_vec(out))
    }
}

fn square_dimension(matrix: &DMatrix<f64>) -> Option<usize> {
    (matrix.nrows() == matrix.ncols()).then_some(matrix.nrows())
}

fn dense_to_triplets(matrix: &DMatrix<f64>, drop_tol: f64) -> Vec<Triplet<usize, usize, f64>> {
    let mut triplets = Vec::new();
    for j in 0..matrix.ncols() {
        for i in 0..matrix.nrows() {
            let value = matrix[(i, j)];
            if value.abs() > drop_tol {
                triplets.push(Triplet::new(i, j, value));
            }
        }
    }
    triplets
}

fn dense_to_banded(matrix: &DMatrix<f64>, drop_tol: f64) -> Option<Banded<f64>> {
    let n = square_dimension(matrix)?;
    let mut kl = 0usize;
    let mut ku = 0usize;

    for j in 0..n {
        for i in 0..n {
            let value = matrix[(i, j)];
            if value.abs() > drop_tol {
                kl = kl.max(i.saturating_sub(j));
                ku = ku.max(j.saturating_sub(i));
            }
        }
    }

    let mut banded = Banded::<f64>::zeros(n, kl, ku).ok()?;
    banded.fill_from_dense(|i, j| matrix[(i, j)]);
    Some(banded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::BDF::BDF_solver::BdfLinearBackend;

    fn solve_with_backend(
        mut backend: impl BdfLinearBackend,
        matrix: DMatrix<f64>,
        rhs: DVector<f64>,
    ) -> DVector<f64> {
        backend
            .factor(&matrix)
            .expect("factorization should succeed")
            .solve(&rhs)
            .expect("solve should succeed")
    }

    fn solve_shifted_with_backend(
        mut backend: impl BdfLinearBackend,
        c: f64,
        jacobian: BdfJacobian,
        rhs: DVector<f64>,
    ) -> DVector<f64> {
        backend
            .factor_shifted_jacobian(c, &jacobian)
            .expect("shifted factorization should succeed")
            .solve(&rhs)
            .expect("solve should succeed")
    }

    fn max_abs_diff(a: &DVector<f64>, b: &DVector<f64>) -> f64 {
        assert_eq!(a.len(), b.len(), "vectors must have equal lengths");
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    }

    fn relative_l2_error(actual: &DVector<f64>, reference: &DVector<f64>) -> f64 {
        assert_eq!(
            actual.len(),
            reference.len(),
            "vectors must have equal lengths"
        );
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for (a, r) in actual.iter().zip(reference.iter()) {
            let d = a - r;
            num += d * d;
            den += r * r;
        }
        num.sqrt() / den.sqrt().max(1.0e-30)
    }

    fn make_pivot_scaling_stress_banded_matrix(n: usize) -> DMatrix<f64> {
        assert!(n >= 6, "stress matrix requires n >= 6");
        let mut a = DMatrix::<f64>::zeros(n, n);

        for i in 0..n {
            let d = match i % 4 {
                0 => 1.0e-10,
                1 => 1.0e-7,
                2 => 1.0e-4,
                _ => 1.0,
            };
            a[(i, i)] = d;

            if i + 1 < n {
                // upper bandwidth ku=1
                a[(i, i + 1)] = -0.75 - 0.01 * i as f64;
            }
            if i >= 1 {
                // lower bandwidth kl>=1; values dominate tiny pivots
                a[(i, i - 1)] = 1.25 + 0.02 * i as f64;
            }
            if i >= 2 {
                // second lower diagonal to exercise kl=2 path
                a[(i, i - 2)] = -0.5;
            }
        }

        // Keep invertibility robust while preserving pivot pressure.
        for i in 0..n {
            a[(i, i)] += 0.2;
        }
        a
    }

    fn make_row_scales(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| match i % 7 {
                0 => 1.0e-6,
                1 => 1.0e-4,
                2 => 1.0e-2,
                3 => 1.0,
                4 => 1.0e2,
                5 => 1.0e4,
                _ => 1.0e6,
            })
            .collect()
    }

    fn left_row_scale(matrix: &DMatrix<f64>, scales: &[f64]) -> DMatrix<f64> {
        assert_eq!(
            matrix.nrows(),
            scales.len(),
            "row scales length must match matrix rows"
        );
        let mut out = matrix.clone();
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                out[(i, j)] = scales[i] * matrix[(i, j)];
            }
        }
        out
    }

    fn left_row_scale_vec(rhs: &DVector<f64>, scales: &[f64]) -> DVector<f64> {
        assert_eq!(
            rhs.len(),
            scales.len(),
            "row scales length must match rhs length"
        );
        let mut out = rhs.clone();
        for i in 0..rhs.len() {
            out[i] = scales[i] * rhs[i];
        }
        out
    }

    #[test]
    fn faer_sparse_backend_solves_small_newton_matrix() {
        let matrix = DMatrix::from_row_slice(3, 3, &[4.0, 1.0, 0.0, 1.0, 5.0, 1.0, 0.0, 1.0, 6.0]);
        let rhs = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let x = solve_with_backend(FaerSparseBdfLinearBackend::default(), matrix.clone(), rhs);

        let residual = &matrix * &x - DVector::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(residual.amax() < 1e-12);
    }

    #[test]
    fn faithful_banded_backend_solves_small_newton_matrix() {
        let matrix = DMatrix::from_row_slice(
            4,
            4,
            &[
                4.0, 1.0, 0.0, 0.0, 2.0, 5.0, 1.0, 0.0, 0.0, 3.0, 6.0, 1.0, 0.0, 0.0, 4.0, 7.0,
            ],
        );
        let rhs = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let x = solve_with_backend(
            FaithfulBandedBdfLinearBackend::default(),
            matrix.clone(),
            rhs,
        );

        let residual = &matrix * &x - DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        assert!(residual.amax() < 1e-12);
    }

    #[test]
    fn faer_sparse_backend_factors_shifted_sparse_jacobian_natively() {
        let c = 0.25;
        let triplets = vec![
            Triplet::new(0, 0, -2.0),
            Triplet::new(1, 0, 1.0),
            Triplet::new(0, 1, 3.0),
            Triplet::new(1, 1, -4.0),
        ];
        let jacobian = BdfJacobian::SparseTriplets { n: 2, triplets };
        let rhs = DVector::from_vec(vec![2.0, -1.0]);
        let x = solve_shifted_with_backend(
            FaerSparseBdfLinearBackend::default(),
            c,
            jacobian.clone(),
            rhs.clone(),
        );

        let residual = jacobian.to_shifted_dense(c).unwrap() * &x - rhs;
        assert!(residual.amax() < 1e-12);
    }

    #[test]
    fn faithful_banded_backend_factors_shifted_banded_jacobian_natively() {
        let c = 0.2;
        let mut jacobian = Banded::<f64>::zeros(3, 1, 1).unwrap();
        jacobian.set(0, 0, -2.0).unwrap();
        jacobian.set(1, 0, 1.0).unwrap();
        jacobian.set(0, 1, 3.0).unwrap();
        jacobian.set(1, 1, -4.0).unwrap();
        jacobian.set(2, 1, 2.0).unwrap();
        jacobian.set(1, 2, -1.0).unwrap();
        jacobian.set(2, 2, -5.0).unwrap();
        let jacobian = BdfJacobian::Banded(jacobian);
        let rhs = DVector::from_vec(vec![2.0, -1.0, 4.0]);
        let x = solve_shifted_with_backend(
            FaithfulBandedBdfLinearBackend::default(),
            c,
            jacobian.clone(),
            rhs.clone(),
        );

        let residual = jacobian.to_shifted_dense(c).unwrap() * &x - rhs;
        assert!(residual.amax() < 1e-12);
    }

    #[test]
    fn dense_lu_backend_factors_shifted_dense_jacobian_natively() {
        let c = 0.3;
        let jacobian = BdfJacobian::from_dense(DMatrix::from_row_slice(
            3,
            3,
            &[
                -2.0, 1.0, 0.0, //
                3.0, -4.0, 2.0, //
                0.0, 1.5, -5.0,
            ],
        ));
        let rhs = DVector::from_vec(vec![1.0, -2.0, 0.5]);
        let x =
            solve_shifted_with_backend(DenseLuBdfLinearBackend, c, jacobian.clone(), rhs.clone());

        let residual = jacobian.to_shifted_dense(c).unwrap() * &x - rhs;
        assert!(residual.amax() < 1e-12);
    }

    #[test]
    fn faithful_banded_backend_pivot_scaling_stress_matches_dense_reference() {
        let n = 20;
        let a = make_pivot_scaling_stress_banded_matrix(n);
        let x_true = DVector::from_iterator(
            n,
            (0..n).map(|i| {
                if i % 2 == 0 {
                    5.0e2 / (i + 1) as f64
                } else {
                    -3.0e-6 * (i + 1) as f64
                }
            }),
        );
        let b = &a * &x_true;

        let x_dense = solve_with_backend(DenseLuBdfLinearBackend, a.clone(), b.clone());
        let x_banded = solve_with_backend(
            FaithfulBandedBdfLinearBackend::default(),
            a.clone(),
            b.clone(),
        );

        let rel_dense = relative_l2_error(&x_dense, &x_true);
        let rel_banded = relative_l2_error(&x_banded, &x_true);
        let rel_vs_dense = relative_l2_error(&x_banded, &x_dense);
        assert!(
            rel_dense < 1.0e-9,
            "dense reference should recover stress system accurately, got rel_l2={rel_dense:e}"
        );
        assert!(
            rel_banded < 1.0e-7,
            "faithful banded should remain accurate on pivot/scaling stress, got rel_l2={rel_banded:e}"
        );
        assert!(
            rel_vs_dense < 1.0e-7,
            "faithful banded should stay close to dense reference on stress matrix, got rel_l2={rel_vs_dense:e}"
        );

        let scales = make_row_scales(n);
        let a_scaled = left_row_scale(&a, &scales);
        let b_scaled = left_row_scale_vec(&b, &scales);

        let x_banded_scaled = solve_with_backend(
            FaithfulBandedBdfLinearBackend::default(),
            a_scaled.clone(),
            b_scaled.clone(),
        );
        let x_dense_scaled = solve_with_backend(DenseLuBdfLinearBackend, a_scaled, b_scaled);

        let rel_banded_scaled = relative_l2_error(&x_banded_scaled, &x_true);
        let rel_dense_scaled = relative_l2_error(&x_dense_scaled, &x_true);
        let rel_scaled_vs_unscaled = relative_l2_error(&x_banded_scaled, &x_banded);
        assert!(
            rel_dense_scaled < 1.0e-8,
            "dense reference on row-scaled stress system should remain accurate, got rel_l2={rel_dense_scaled:e}"
        );
        assert!(
            rel_banded_scaled < 1.0e-6,
            "faithful banded should remain stable under row scaling, got rel_l2={rel_banded_scaled:e}"
        );
        assert!(
            rel_scaled_vs_unscaled < 1.0e-6,
            "faithful banded row-scaled solve should match unscaled solve, got rel_l2={rel_scaled_vs_unscaled:e}"
        );
        assert!(
            max_abs_diff(&x_banded_scaled, &x_banded) < 1.0e-3,
            "max_abs diff between scaled and unscaled faithful solutions is unexpectedly large"
        );
    }

    #[test]
    fn sparse_faer_backend_pivot_scaling_stress_matches_dense_reference() {
        let n = 20;
        let a = make_pivot_scaling_stress_banded_matrix(n);
        let x_true = DVector::from_iterator(
            n,
            (0..n).map(|i| {
                if i % 3 == 0 {
                    2.0e3 / (i + 1) as f64
                } else if i % 3 == 1 {
                    -1.0e-4 * (i + 1) as f64
                } else {
                    1.0e-8 * (i + 1) as f64
                }
            }),
        );
        let b = &a * &x_true;

        let x_dense = solve_with_backend(DenseLuBdfLinearBackend, a.clone(), b.clone());
        let x_sparse =
            solve_with_backend(FaerSparseBdfLinearBackend::default(), a.clone(), b.clone());

        let rel_dense = relative_l2_error(&x_dense, &x_true);
        let rel_sparse = relative_l2_error(&x_sparse, &x_true);
        let rel_vs_dense = relative_l2_error(&x_sparse, &x_dense);
        assert!(
            rel_dense < 1.0e-9,
            "dense reference should recover sparse stress system accurately, got rel_l2={rel_dense:e}"
        );
        assert!(
            rel_sparse < 1.0e-7,
            "faer sparse backend should remain accurate on pivot/scaling stress, got rel_l2={rel_sparse:e}"
        );
        assert!(
            rel_vs_dense < 1.0e-7,
            "faer sparse backend should stay close to dense reference on stress matrix, got rel_l2={rel_vs_dense:e}"
        );

        let scales = make_row_scales(n);
        let a_scaled = left_row_scale(&a, &scales);
        let b_scaled = left_row_scale_vec(&b, &scales);

        let x_sparse_scaled = solve_with_backend(
            FaerSparseBdfLinearBackend::default(),
            a_scaled.clone(),
            b_scaled.clone(),
        );
        let x_dense_scaled = solve_with_backend(DenseLuBdfLinearBackend, a_scaled, b_scaled);

        let rel_sparse_scaled = relative_l2_error(&x_sparse_scaled, &x_true);
        let rel_dense_scaled = relative_l2_error(&x_dense_scaled, &x_true);
        let rel_scaled_vs_unscaled = relative_l2_error(&x_sparse_scaled, &x_sparse);
        assert!(
            rel_dense_scaled < 1.0e-8,
            "dense reference on row-scaled sparse stress system should remain accurate, got rel_l2={rel_dense_scaled:e}"
        );
        assert!(
            rel_sparse_scaled < 1.0e-6,
            "faer sparse backend should remain stable under row scaling, got rel_l2={rel_sparse_scaled:e}"
        );
        assert!(
            rel_scaled_vs_unscaled < 1.0e-6,
            "faer sparse row-scaled solve should match unscaled solve, got rel_l2={rel_scaled_vs_unscaled:e}"
        );
    }
}
