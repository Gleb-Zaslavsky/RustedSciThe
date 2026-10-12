//! Reusable sparse LU state for the experimental BVP_sci direct route.
//!
//! This module deliberately does not alter the shared banded solvers or their
//! policies. It wraps faer's low-level API so numeric factors and scratch
//! buffers can be reused when a collocation matrix keeps the same CSC pattern.
//! This was an experimental reuse hypothesis for BVP_sci, not a replacement
//! for other faer sparse solvers; the matched official BVP test showed partial
//! improvement but did not beat the opt-in SuperLU path.

use dyn_stack::{MemBuffer, MemStack};
use faer::sparse::linalg::SupernodalThreshold;
use faer::sparse::linalg::lu::{
    LuRef, LuSymbolicParams, NumericLu, SymbolicLu, factorize_symbolic_lu,
};
use faer::sparse::{FaerError, SparseColMatRef, linalg::LuError};
use faer::{Conj, MatMut, get_global_parallelism};
use thiserror::Error;

/// Errors from the isolated reusable sparse LU kernel.
#[derive(Debug, Error)]
pub(crate) enum ReusableSparseLuError {
    #[error("sparse LU symbolic analysis failed: {0:?}")]
    Symbolic(#[source] FaerError),
    #[error("sparse LU numeric factorization failed: {0:?}")]
    Numeric(#[source] LuError),
    #[error("could not allocate sparse LU scratch workspace")]
    WorkspaceAllocation,
    #[error("sparse LU factorization was requested for an unprepared CSC pattern")]
    PatternMismatch,
    #[error("sparse LU solve was requested before numeric factorization")]
    NotFactorized,
    #[error("sparse LU right-hand side has an invalid length")]
    InvalidRhsLength,
}

/// Symbolic LU kernel choice used only by the isolated BVP experiment.
// The standalone microbenchmark exercises faer's modes directly rather than
// going through this BVP policy enum.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum ReusableSparseLuStrategy {
    #[default]
    Auto,
    Simplicial,
}

/// Numeric sparse LU state whose allocations survive same-pattern refactors.
///
/// Pattern arrays are copied once so a changed CSC structure cannot silently
/// reuse an incompatible symbolic factorization. Matrix values remain owned by
/// the caller and are passed by view for every numeric factorization.
pub(crate) struct ReusableSparseLu {
    dimension: usize,
    strategy: ReusableSparseLuStrategy,
    pattern_col_ptr: Vec<usize>,
    pattern_row_idx: Vec<usize>,
    symbolic: Option<SymbolicLu<usize>>,
    numeric: NumericLu<usize, f64>,
    factor_scratch: Option<MemBuffer>,
    solve_scratch: Option<MemBuffer>,
    factorized: bool,
}

impl ReusableSparseLu {
    // Kept as the default-policy constructor for direct kernel fixtures.
    #[allow(dead_code)]
    pub(crate) fn new(dimension: usize) -> Self {
        Self::with_strategy(dimension, ReusableSparseLuStrategy::Auto)
    }

    pub(crate) fn with_strategy(dimension: usize, strategy: ReusableSparseLuStrategy) -> Self {
        Self {
            dimension,
            strategy,
            pattern_col_ptr: Vec::new(),
            pattern_row_idx: Vec::new(),
            symbolic: None,
            numeric: NumericLu::new(),
            factor_scratch: None,
            solve_scratch: None,
            factorized: false,
        }
    }

    /// Prepare symbolic structure only when the incoming CSC pattern changes.
    /// Returns `true` when a new symbolic analysis was performed.
    pub(crate) fn prepare_symbolic(
        &mut self,
        matrix: SparseColMatRef<'_, usize, f64>,
    ) -> Result<bool, ReusableSparseLuError> {
        if matrix.nrows() != self.dimension || matrix.ncols() != self.dimension {
            return Err(ReusableSparseLuError::PatternMismatch);
        }
        let csc = matrix.symbolic();
        if self.symbolic.is_some()
            && self.pattern_col_ptr == csc.col_ptr()
            && self.pattern_row_idx == csc.row_idx()
        {
            return Ok(false);
        }

        let supernodal_flop_ratio_threshold = match self.strategy {
            ReusableSparseLuStrategy::Auto => SupernodalThreshold::AUTO,
            ReusableSparseLuStrategy::Simplicial => SupernodalThreshold::FORCE_SIMPLICIAL,
        };
        let symbolic = factorize_symbolic_lu(
            csc,
            LuSymbolicParams {
                colamd_params: Default::default(),
                supernodal_flop_ratio_threshold,
                supernodal_params: Default::default(),
            },
        )
        .map_err(ReusableSparseLuError::Symbolic)?;
        let par = get_global_parallelism();
        let factor_req = symbolic.factorize_numeric_lu_scratch::<f64>(par, Default::default());
        let solve_req = symbolic.solve_in_place_scratch::<f64>(1, par);
        let factor_scratch = MemBuffer::try_new(factor_req)
            .map_err(|_| ReusableSparseLuError::WorkspaceAllocation)?;
        let solve_scratch = MemBuffer::try_new(solve_req)
            .map_err(|_| ReusableSparseLuError::WorkspaceAllocation)?;

        self.pattern_col_ptr.clear();
        self.pattern_col_ptr.extend_from_slice(csc.col_ptr());
        self.pattern_row_idx.clear();
        self.pattern_row_idx.extend_from_slice(csc.row_idx());
        self.symbolic = Some(symbolic);
        self.numeric = NumericLu::new();
        self.factor_scratch = Some(factor_scratch);
        self.solve_scratch = Some(solve_scratch);
        self.factorized = false;
        Ok(true)
    }

    /// Refactor numeric values while retaining faer's numeric buffers.
    pub(crate) fn factor_numeric(
        &mut self,
        matrix: SparseColMatRef<'_, usize, f64>,
    ) -> Result<(), ReusableSparseLuError> {
        if !self.matches_pattern(matrix) {
            return Err(ReusableSparseLuError::PatternMismatch);
        }
        self.factorized = false;
        let symbolic = self
            .symbolic
            .as_ref()
            .ok_or(ReusableSparseLuError::PatternMismatch)?;
        let scratch = self
            .factor_scratch
            .as_mut()
            .ok_or(ReusableSparseLuError::PatternMismatch)?;
        let mut stack = MemStack::new(scratch);
        symbolic
            .factorize_numeric_lu(
                &mut self.numeric,
                matrix,
                get_global_parallelism(),
                &mut stack,
                Default::default(),
            )
            .map_err(ReusableSparseLuError::Numeric)?;
        self.factorized = true;
        Ok(())
    }

    /// Solve in the caller's RHS buffer using the retained factor and scratch.
    pub(crate) fn solve_in_place(&mut self, rhs: &mut [f64]) -> Result<(), ReusableSparseLuError> {
        if rhs.len() != self.dimension {
            return Err(ReusableSparseLuError::InvalidRhsLength);
        }
        if !self.factorized {
            return Err(ReusableSparseLuError::NotFactorized);
        }
        let symbolic = self
            .symbolic
            .as_ref()
            .ok_or(ReusableSparseLuError::NotFactorized)?;
        let scratch = self
            .solve_scratch
            .as_mut()
            .ok_or(ReusableSparseLuError::NotFactorized)?;
        let mut stack = MemStack::new(scratch);

        // `numeric` was produced by `factorize_numeric_lu` using this exact
        // symbolic pattern, and `factorized` is cleared on failed refactors.
        let factors = LuRef::new_unchecked(symbolic, &self.numeric);
        let rhs = MatMut::from_column_major_slice_mut(rhs, self.dimension, 1);
        factors.solve_in_place_with_conj(Conj::No, rhs, get_global_parallelism(), &mut stack);
        Ok(())
    }

    fn matches_pattern(&self, matrix: SparseColMatRef<'_, usize, f64>) -> bool {
        let csc = matrix.symbolic();
        matrix.nrows() == self.dimension
            && matrix.ncols() == self.dimension
            && self.pattern_col_ptr == csc.col_ptr()
            && self.pattern_row_idx == csc.row_idx()
    }
}
