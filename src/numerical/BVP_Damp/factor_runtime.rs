//! Internal factor-owner runtimes for the Frozen and Damped solvers.
//!
//! This module is deliberately not part of the public `MatrixType` contract.
//! The legacy trait remains available to downstream implementations, while
//! built-in Dense/faer paths can own a prepared factor across repeated RHS
//! solves. Native Banded keeps its existing `BandedMatrixType` cache.

// The compatibility `solve` wrapper and inspection helpers are retained for
// the next PreparedPlan migration slice and are not used by every route yet.
#![allow(dead_code)]

use crate::numerical::BVP_Damp::BVP_traits::{MatrixType, VectorType};
use crate::somelinalg::RustedLINPACK::lu_band_nalg::LU_nalgebra;
use faer::col::Col;
use faer::linalg::solvers::Solve;
use faer::mat::MatRef;
use faer::sparse::SparseColMat;
use faer::sparse::linalg::solvers::Lu as FaerSparseLu;
use nalgebra::{DMatrix, Dyn, LU};
use std::borrow::Cow;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::{Duration, Instant};

/// Typed failures from the internal owned-factor runtime.
///
/// The public legacy solver API still exposes its historical panic wrappers,
/// but new runtime code can preserve the failure stage and return a normal
/// `Result` instead of losing the reason inside `expect`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum LinearFactorError {
    DimensionMismatch {
        matrix_rows: usize,
        matrix_columns: usize,
        rhs_len: usize,
    },
    DenseSolveFailed,
    NonFiniteSolution,
}

/// Prepared factor plus the metadata needed to report first-use versus
/// repeated-RHS timings at the solver boundary.
pub(crate) struct OwnedLinearFactorRuntime {
    factor: OwnedLinearFactor,
    factorization_time: Duration,
    rhs_solves: u64,
}

impl OwnedLinearFactorRuntime {
    pub(crate) fn from_prepared(factor: OwnedLinearFactor, factorization_time: Duration) -> Self {
        Self {
            factor,
            factorization_time,
            rhs_solves: 0,
        }
    }

    pub(crate) fn has_solved_rhs(&self) -> bool {
        self.rhs_solves != 0
    }

    pub(crate) fn try_solve(
        &mut self,
        rhs: &dyn VectorType,
    ) -> Result<(Box<dyn VectorType>, Duration, Duration), LinearFactorError> {
        let (solution, rhs_time) = self.factor.try_solve(rhs)?;
        let factorization_time = if self.rhs_solves == 0 {
            self.factorization_time
        } else {
            Duration::ZERO
        };
        self.rhs_solves += 1;
        Ok((solution, factorization_time, rhs_time))
    }
}

/// A factor that owns its numeric decomposition and can solve multiple RHSs.
pub(crate) enum OwnedLinearFactor {
    Dense {
        factor: LU<f64, Dyn, Dyn>,
        rows: usize,
        columns: usize,
    },
    DenseBanded(LU_nalgebra),
    FaerSparse {
        factor: FaerSparseLu<usize, f64>,
        rows: usize,
        columns: usize,
    },
}

impl OwnedLinearFactor {
    /// Builds a Dense factor while preserving the existing bandwidth heuristic.
    pub(crate) fn from_dense(matrix: &DMatrix<f64>, bandwidth: (usize, usize)) -> (Self, Duration) {
        let begin = Instant::now();
        let condition_for_banded_matrix = matrix.nrows() > 10 * (bandwidth.0 + bandwidth.1);
        let factor = if condition_for_banded_matrix {
            let mut factor = LU_nalgebra::new(matrix.clone(), Some(bandwidth));
            factor.LU();
            Self::DenseBanded(factor)
        } else {
            Self::Dense {
                factor: matrix.clone().lu(),
                rows: matrix.nrows(),
                columns: matrix.ncols(),
            }
        };
        (factor, begin.elapsed())
    }

    pub(crate) fn from_faer_sparse(matrix: &SparseColMat<usize, f64>) -> Option<(Self, Duration)> {
        let begin = Instant::now();
        let factor = matrix.sp_lu().ok()?;
        Some((
            Self::FaerSparse {
                factor,
                rows: matrix.nrows(),
                columns: matrix.ncols(),
            },
            begin.elapsed(),
        ))
    }

    pub(crate) fn solve(&mut self, rhs: &dyn VectorType) -> (Box<dyn VectorType>, Duration) {
        self.try_solve(rhs)
            .unwrap_or_else(|error| panic!("owned BVP linear factor solve failed: {error:?}"))
    }

    /// Solves one RHS without converting a backend failure into a panic.
    pub(crate) fn try_solve(
        &mut self,
        rhs: &dyn VectorType,
    ) -> Result<(Box<dyn VectorType>, Duration), LinearFactorError> {
        let begin = Instant::now();
        let result = match self {
            Self::Dense {
                factor,
                rows,
                columns,
            } => {
                let rhs = dense_rhs(rhs);
                if *rows != rhs.len() || *columns != rhs.len() {
                    return Err(LinearFactorError::DimensionMismatch {
                        matrix_rows: *rows,
                        matrix_columns: *columns,
                        rhs_len: rhs.len(),
                    });
                }
                if !factor.is_invertible() {
                    return Err(LinearFactorError::DenseSolveFailed);
                }
                let solution = factor
                    .solve(&rhs)
                    .ok_or(LinearFactorError::DenseSolveFailed)?;
                if solution.iter().all(|value| value.is_finite()) {
                    Box::new(solution) as Box<dyn VectorType>
                } else {
                    return Err(LinearFactorError::NonFiniteSolution);
                }
            }
            Self::DenseBanded(factor) => {
                let rhs = dense_rhs(rhs);
                let (rows, columns) = factor.shape();
                if rows != rhs.len() || columns != rhs.len() {
                    return Err(LinearFactorError::DimensionMismatch {
                        matrix_rows: rows,
                        matrix_columns: columns,
                        rhs_len: rhs.len(),
                    });
                }
                let solution =
                    catch_unwind(AssertUnwindSafe(|| factor.solve_linear_system_easy(&rhs)))
                        .map_err(|_| LinearFactorError::DenseSolveFailed)?;
                if solution.iter().all(|value| value.is_finite()) {
                    Box::new(solution) as Box<dyn VectorType>
                } else {
                    return Err(LinearFactorError::NonFiniteSolution);
                }
            }
            Self::FaerSparse {
                factor,
                rows,
                columns,
            } => {
                let owned_rhs;
                if *rows != rhs.len() || *columns != rhs.len() {
                    return Err(LinearFactorError::DimensionMismatch {
                        matrix_rows: *rows,
                        matrix_columns: *columns,
                        rhs_len: rhs.len(),
                    });
                }
                let rhs = if let Some(rhs) = rhs.as_any().downcast_ref::<Col<f64>>() {
                    rhs
                } else {
                    owned_rhs = Col::from_fn(rhs.len(), |index| rhs.get_val(index));
                    &owned_rhs
                };
                let lhs: MatRef<f64> = rhs.as_mat();
                let solved = factor.solve(lhs).col(0).to_owned();
                if (0..solved.nrows()).all(|index| solved[index].is_finite()) {
                    Box::new(solved) as Box<dyn VectorType>
                } else {
                    return Err(LinearFactorError::NonFiniteSolution);
                }
            }
        };
        Ok((result, begin.elapsed()))
    }
}

fn dense_rhs(rhs: &dyn VectorType) -> Cow<'_, nalgebra::DVector<f64>> {
    if let Some(rhs) = rhs.as_any().downcast_ref::<nalgebra::DVector<f64>>() {
        Cow::Borrowed(rhs)
    } else {
        Cow::Owned(rhs.to_DVectorType())
    }
}

/// Builds an owner only for the built-in direct Dense/faer routes.
///
/// `Some(linear_sys_method)` selects an iterative/compatibility route and is
/// intentionally left on the old API. Banded ownership is already provided by
/// `BandedMatrixType` and must not be duplicated here.
pub(crate) fn prepare_factor_owner(
    matrix: &dyn MatrixType,
    bandwidth: (usize, usize),
    linear_sys_method: Option<&str>,
) -> Option<(OwnedLinearFactor, Duration)> {
    if linear_sys_method.is_some() {
        return None;
    }
    if let Some(matrix) = matrix.as_any().downcast_ref::<DMatrix<f64>>() {
        return Some(OwnedLinearFactor::from_dense(matrix, bandwidth));
    }
    if let Some(matrix) = matrix.as_any().downcast_ref::<SparseColMat<usize, f64>>() {
        return OwnedLinearFactor::from_faer_sparse(matrix);
    }
    None
}

pub(crate) fn prepare_factor_owner_runtime(
    matrix: &dyn MatrixType,
    bandwidth: (usize, usize),
    linear_sys_method: Option<&str>,
) -> Option<OwnedLinearFactorRuntime> {
    let (factor, factorization_time) = prepare_factor_owner(matrix, bandwidth, linear_sys_method)?;
    Some(OwnedLinearFactorRuntime::from_prepared(
        factor,
        factorization_time,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::sparse::{SparseColMat, Triplet};
    use nalgebra::DVector;

    fn relative_backward_error(
        matrix: &DMatrix<f64>,
        solution: &DVector<f64>,
        rhs: &DVector<f64>,
    ) -> f64 {
        let residual = matrix * solution - rhs;
        residual.norm() / (matrix.norm() * solution.norm() + rhs.norm()).max(f64::MIN_POSITIVE)
    }

    #[test]
    fn dense_factor_owner_reuses_one_factor_for_multiple_rhs_solves() {
        let matrix = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 0.0, 3.0]);
        let (mut owner, factor_time) = prepare_factor_owner(&matrix, (0, 0), None)
            .expect("built-in Dense matrix should create an owner");
        let rhs = DVector::from_vec(vec![5.0, 6.0]);
        let (first, first_time) = owner.solve(&rhs);
        let (second, second_time) = owner.solve(&rhs);
        let (legacy, _) = matrix.solve_sys_with_timing(&rhs, None, 1e-12, 10, (0, 0), &rhs);

        assert!(factor_time > Duration::ZERO);
        assert!(first_time > Duration::ZERO);
        assert!(second_time > Duration::ZERO);
        assert!((first.to_DVectorType()[0] - second.to_DVectorType()[0]).abs() < 1e-12);
        assert!((first.to_DVectorType()[1] - second.to_DVectorType()[1]).abs() < 1e-12);
        assert!((first.to_DVectorType() - legacy.to_DVectorType()).norm() < 1e-12);
        assert!(relative_backward_error(&matrix, &first.to_DVectorType(), &rhs) < 1e-14);
    }

    #[test]
    fn faer_factor_owner_reuses_one_factor_for_multiple_rhs_solves() {
        let triplets = [
            Triplet::new(0, 0, 4.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(1, 1, 3.0),
        ];
        let matrix: SparseColMat<usize, f64> =
            SparseColMat::try_new_from_triplets(2, 2, &triplets).expect("valid sparse matrix");
        let (mut owner, factor_time) = prepare_factor_owner(&matrix, (0, 0), None)
            .expect("built-in faer matrix should create an owner");
        let rhs = Col::from_fn(2, |i| [5.0, 6.0][i]);
        let (first, first_time) = owner.solve(&rhs);
        let (second, second_time) = owner.solve(&rhs);
        let (legacy, _) = matrix.solve_sys_with_timing(&rhs, None, 1e-12, 10, (0, 0), &rhs);

        assert!(factor_time > Duration::ZERO);
        assert!(first_time > Duration::ZERO);
        assert!(second_time > Duration::ZERO);
        assert!((first.to_DVectorType()[0] - second.to_DVectorType()[0]).abs() < 1e-12);
        assert!((first.to_DVectorType()[1] - second.to_DVectorType()[1]).abs() < 1e-12);
        assert!((first.to_DVectorType() - legacy.to_DVectorType()).norm() < 1e-12);
        let dense_matrix = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 0.0, 3.0]);
        let dense_rhs = DVector::from_vec(vec![5.0, 6.0]);
        assert!(
            relative_backward_error(&dense_matrix, &first.to_DVectorType(), &dense_rhs) < 1e-14
        );
    }

    #[test]
    fn factor_owner_reports_dimension_mismatch_without_panicking() {
        let matrix = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 0.0, 3.0]);
        let (mut dense_owner, _) = prepare_factor_owner(&matrix, (0, 0), None)
            .expect("built-in Dense matrix should create an owner");
        let wrong_rhs = DVector::from_element(1, 1.0);
        assert!(matches!(
            dense_owner.try_solve(&wrong_rhs),
            Err(LinearFactorError::DimensionMismatch {
                matrix_rows: 2,
                matrix_columns: 2,
                rhs_len: 1,
            })
        ));

        let triplets = [
            Triplet::new(0, 0, 4.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(1, 1, 3.0),
        ];
        let sparse: SparseColMat<usize, f64> =
            SparseColMat::try_new_from_triplets(2, 2, &triplets).expect("valid sparse matrix");
        let (mut sparse_owner, _) = prepare_factor_owner(&sparse, (0, 0), None)
            .expect("built-in faer matrix should create an owner");
        assert!(matches!(
            sparse_owner.try_solve(&wrong_rhs),
            Err(LinearFactorError::DimensionMismatch {
                matrix_rows: 2,
                matrix_columns: 2,
                rhs_len: 1,
            })
        ));
    }

    #[test]
    fn dense_banded_factor_owner_rejects_nonfinite_solution() {
        let matrix = DMatrix::<f64>::zeros(16, 16);
        let (mut owner, _) = prepare_factor_owner(&matrix, (0, 0), None)
            .expect("large Dense matrix should use the DenseBanded owner");
        let rhs = DVector::from_element(16, 1.0);

        assert!(matches!(
            owner.try_solve(&rhs),
            Err(LinearFactorError::DenseSolveFailed | LinearFactorError::NonFiniteSolution)
        ));
    }
}
