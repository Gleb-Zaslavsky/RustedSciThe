//! Opt-in adapter for the isolated reusable faer sparse-LU kernel.
//!
//! This backend exists only for BVP-level comparison. It retains the CSC
//! values between refactors so the numeric kernel can consume a view; the
//! standard `SparseBackend` and shared `somelinalg::banded` paths are untouched.
//! The original hypothesis was that reusable faer symbolic/numeric workspace
//! would close the measured BVP gap. The official continuation A/B confirmed a
//! useful partial gain, but SuperLU remained substantially faster; keep this
//! route isolated as experimental rather than implying it is a supported
//! replacement for either StandardFaer or the opt-in SuperLU backend.

use crate::numerical::BVP_sci::new::{
    BvpSciMatrixLayout, error::BvpSciNewError, linear::LinearSystemBackend,
    telemetry::BvpSciTelemetry,
};
use crate::somelinalg::bvp_sparse_lu::{
    ReusableSparseLu, ReusableSparseLuError, ReusableSparseLuStrategy,
};
use faer::sparse::{SparseColMat, Triplet};

/// Experimental Sparse backend retaining low-level numeric LU workspace.
pub struct ReusableSparseExperimentalBackend {
    dimension: usize,
    _pattern_nnz: usize,
    telemetry: BvpSciTelemetry,
    matrix: Option<SparseColMat<usize, f64>>,
    kernel: ReusableSparseLu,
}

impl ReusableSparseExperimentalBackend {
    pub(crate) fn new(
        dimension: usize,
        pattern_nnz: usize,
        telemetry: BvpSciTelemetry,
        strategy: ReusableSparseLuStrategy,
    ) -> Self {
        Self {
            dimension,
            _pattern_nnz: pattern_nnz,
            telemetry,
            matrix: None,
            kernel: ReusableSparseLu::with_strategy(dimension, strategy),
        }
    }

    fn map_kernel_error(error: ReusableSparseLuError) -> BvpSciNewError {
        match error {
            ReusableSparseLuError::Symbolic(source) => BvpSciNewError::LinearFactorization {
                message: format!("experimental sparse symbolic LU failed: {source:?}"),
            },
            ReusableSparseLuError::Numeric(source) => BvpSciNewError::LinearFactorization {
                message: format!("experimental sparse numeric LU failed: {source:?}"),
            },
            other => BvpSciNewError::LinearBackend {
                message: format!("experimental sparse LU failed: {other}"),
            },
        }
    }
}

impl LinearSystemBackend for ReusableSparseExperimentalBackend {
    fn layout(&self) -> BvpSciMatrixLayout {
        BvpSciMatrixLayout::Sparse
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn assemble(&mut self, entries: &[(usize, usize, f64)]) -> Result<(), BvpSciNewError> {
        let triplet_started = self.telemetry.start_timing();
        let triplets = entries
            .iter()
            .map(|&(row, column, value)| Triplet::new(row, column, value))
            .collect::<Vec<_>>();
        self.telemetry
            .record_sparse_triplet_materialization(triplet_started);

        let csc_started = self.telemetry.start_timing();
        let matrix = SparseColMat::<usize, f64>::try_new_from_triplets(
            self.dimension,
            self.dimension,
            &triplets,
        )
        .map_err(|error| BvpSciNewError::LinearBackend {
            message: format!("experimental sparse CSC assembly failed: {error:?}"),
        })?;
        self.telemetry.record_sparse_csc_assembly(csc_started);
        self.matrix = Some(matrix);
        Ok(())
    }

    fn factor(&mut self) -> Result<(), BvpSciNewError> {
        let matrix = self
            .matrix
            .as_ref()
            .ok_or_else(|| BvpSciNewError::LinearBackend {
                message: "experimental sparse backend was factored before assembly".into(),
            })?;

        let symbolic_started = self.telemetry.start_timing();
        let performed_analysis = self
            .kernel
            .prepare_symbolic(matrix.as_ref())
            .map_err(Self::map_kernel_error)?;
        if performed_analysis {
            self.telemetry
                .record_sparse_symbolic_analysis(symbolic_started);
        } else {
            self.telemetry
                .record_sparse_symbolic_reuse(symbolic_started);
        }

        let numeric_started = self.telemetry.start_timing();
        self.kernel
            .factor_numeric(matrix.as_ref())
            .map_err(Self::map_kernel_error)?;
        self.telemetry
            .record_sparse_numeric_factorization(numeric_started);
        Ok(())
    }

    fn solve_in_place(&mut self, rhs: &mut [f64]) -> Result<(), BvpSciNewError> {
        self.kernel
            .solve_in_place(rhs)
            .map_err(Self::map_kernel_error)
    }
}
