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
use nalgebra::{DMatrix, DVector};

const DEFAULT_DROP_TOL: f64 = 0.0;

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
}
