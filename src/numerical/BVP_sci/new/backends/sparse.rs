//! Sparse collocation backend with a prepared structural pattern.

use crate::numerical::BVP_sci::new::{
    BvpSciMatrixLayout, error::BvpSciNewError, linear::LinearSystemBackend,
    telemetry::BvpSciTelemetry,
};
use faer::ColMut;
use faer::linalg::solvers::Solve;
use faer::sparse::linalg::solvers::{Lu, SymbolicLu};
use faer::sparse::{SparseColMat, Triplet};

/// Sparse CSC collocation storage with faer LU.
///
/// The structural pattern is represented by the triplet assembly contract and
/// the global dense matrix is never materialized. The first factorization
/// computes faer's symbolic ordering; subsequent refreshes reuse that
/// `SymbolicLu` and perform only numeric factorization for the same pattern.
#[derive(Clone, Debug)]
pub struct SparseBackend {
    dimension: usize,
    pattern_nnz: usize,
    telemetry: BvpSciTelemetry,
    matrix: Option<SparseColMat<usize, f64>>,
    symbolic: Option<SymbolicLu<usize>>,
    factorization: Option<Lu<usize, f64>>,
}

impl SparseBackend {
    pub fn new(dimension: usize, pattern_nnz: usize) -> Self {
        Self::new_with_telemetry(dimension, pattern_nnz, BvpSciTelemetry::disabled())
    }

    pub fn new_with_telemetry(
        dimension: usize,
        pattern_nnz: usize,
        telemetry: BvpSciTelemetry,
    ) -> Self {
        Self {
            dimension,
            pattern_nnz,
            telemetry,
            matrix: None,
            symbolic: None,
            factorization: None,
        }
    }

    pub fn pattern_nnz(&self) -> usize {
        self.pattern_nnz
    }

    /// Return the shared telemetry snapshot used by the production backend
    /// and by the focused linear-system microbench.
    pub fn telemetry_snapshot(&self) -> crate::numerical::BVP_sci::new::BvpSciTelemetrySnapshot {
        self.telemetry.snapshot()
    }

    pub fn assemble(&mut self, entries: &[(usize, usize, f64)]) -> Result<(), BvpSciNewError> {
        // Triplets are a temporary numeric assembly view, not a second global
        // layout. Duplicate coordinates are resolved by the sparse backend.
        let triplets: Vec<_> = entries
            .iter()
            .map(|&(row, column, value)| Triplet::new(row, column, value))
            .collect();
        let matrix = SparseColMat::<usize, f64>::try_new_from_triplets(
            self.dimension,
            self.dimension,
            &triplets,
        )
        .map_err(|error| BvpSciNewError::LinearBackend {
            message: format!("sparse CSC assembly failed: {error:?}"),
        })?;
        self.matrix = Some(matrix);
        self.factorization = None;
        Ok(())
    }

    pub fn factor(&mut self) -> Result<(), BvpSciNewError> {
        // Moving rather than cloning the matrix avoids retaining a second
        // global sparse value array during factorization. The symbolic plan
        // is retained separately because it is valid for every numeric matrix
        // with this unchanged CSC pattern.
        let matrix = self
            .matrix
            .take()
            .ok_or_else(|| BvpSciNewError::LinearBackend {
                message: "sparse backend was factored before assembly".into(),
            })?;
        let symbolic = if let Some(symbolic) = self.symbolic.clone() {
            symbolic
        } else {
            let started = self.telemetry.start_timing();
            let symbolic = SymbolicLu::try_new(matrix.symbolic()).map_err(|error| {
                BvpSciNewError::LinearFactorization {
                    message: format!("sparse symbolic LU analysis failed: {error:?}"),
                }
            })?;
            self.telemetry.record_sparse_symbolic_analysis(started);
            self.symbolic = Some(symbolic.clone());
            symbolic
        };

        let started = self.telemetry.start_timing();
        let factorization =
            Lu::try_new_with_symbolic(symbolic, matrix.as_ref()).map_err(|error| {
                BvpSciNewError::LinearFactorization {
                    message: format!("sparse numeric LU factorization failed: {error:?}"),
                }
            })?;
        self.telemetry.record_sparse_numeric_factorization(started);
        self.factorization = Some(factorization);
        Ok(())
    }

    pub fn solve_in_place(&mut self, rhs: &mut [f64]) -> Result<(), BvpSciNewError> {
        if rhs.len() != self.dimension {
            return Err(BvpSciNewError::LinearBackend {
                message: "sparse solve vector has an invalid length".into(),
            });
        }
        let factorization =
            self.factorization
                .as_ref()
                .ok_or_else(|| BvpSciNewError::LinearBackend {
                    message: "sparse backend was solved before factorization".into(),
                })?;
        // Solve directly in the caller-owned workspace. The previous
        // `Col::from_fn` plus returned solution allocated two temporary
        // vectors on every Newton/backtracking solve and copied the result
        // back into `rhs`; `ColMut` preserves the same faer algorithm while
        // making the backend genuinely in-place at this API boundary.
        factorization.solve_in_place(ColMut::from_slice_mut(rhs));
        Ok(())
    }
}

impl LinearSystemBackend for SparseBackend {
    fn layout(&self) -> BvpSciMatrixLayout {
        BvpSciMatrixLayout::Sparse
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn assemble(&mut self, entries: &[(usize, usize, f64)]) -> Result<(), BvpSciNewError> {
        Self::assemble(self, entries)
    }

    fn factor(&mut self) -> Result<(), BvpSciNewError> {
        Self::factor(self)
    }

    fn solve_in_place(&mut self, rhs: &mut [f64]) -> Result<(), BvpSciNewError> {
        Self::solve_in_place(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuation_reuses_symbolic_lu_analysis() {
        let telemetry = BvpSciTelemetry::timings();
        let mut backend = SparseBackend::new_with_telemetry(3, 5, telemetry.clone());
        let first = [
            (0, 0, 4.0),
            (1, 0, 1.0),
            (1, 1, 5.0),
            (2, 1, 1.0),
            (2, 2, 6.0),
        ];
        let second = [
            (0, 0, 8.0),
            (1, 0, 2.0),
            (1, 1, 10.0),
            (2, 1, 2.0),
            (2, 2, 12.0),
        ];

        backend
            .assemble(&first)
            .expect("first sparse matrix should assemble");
        backend
            .factor()
            .expect("first sparse factorization should succeed");
        let mut rhs = [1.0, 2.0, 3.0];
        backend
            .solve_in_place(&mut rhs)
            .expect("first sparse solve should succeed");
        assert!(rhs.iter().all(|value| value.is_finite()));

        backend
            .assemble(&second)
            .expect("continued sparse matrix should assemble");
        backend
            .factor()
            .expect("continued sparse numeric factorization should succeed");
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.sparse_symbolic_analyses, 1);
        assert_eq!(snapshot.sparse_numeric_factorizations, 2);
        assert!(snapshot.sparse_symbolic_analysis_ms.is_some());
        assert!(snapshot.sparse_numeric_factorization_ms.is_some());
        assert!(snapshot.validate_contract().is_ok());
    }
}
