//! Prepared Jacobian layout contracts shared by Radau callback frontends.
//!
//! The symbolic frontend owns evaluator preparation. This module only defines
//! how prepared values are exposed to the future numerical linear backends.
//! In particular, it does not depend on BDF's `BdfJacobian` enum.

use super::{
    config::RadauMatrixLayout,
    error::{RadauConfigError, RadauError},
};

#[derive(Debug, Clone, PartialEq, Eq)]
/// Immutable structural Jacobian pattern used by callback projections.
pub(crate) struct RadauJacobianPattern {
    dimension: usize,
    entries: Vec<(usize, usize)>,
}

impl RadauJacobianPattern {
    /// Validate and normalize an arbitrary structural entry list.
    pub(crate) fn from_entries(
        dimension: usize,
        entries: impl IntoIterator<Item = (usize, usize)>,
    ) -> Result<Self, RadauError> {
        if dimension == 0 {
            return Err(RadauConfigError::ZeroDimension.into());
        }
        let mut entries = entries.into_iter().collect::<Vec<_>>();
        if entries
            .iter()
            .any(|&(row, col)| row >= dimension || col >= dimension)
        {
            return Err(RadauConfigError::JacobianPatternOutOfBounds.into());
        }
        entries.sort_unstable();
        entries.dedup();
        Ok(Self { dimension, entries })
    }

    /// Construct a fully dense `dimension x dimension` pattern.
    pub(crate) fn dense(dimension: usize) -> Result<Self, RadauError> {
        let entries = (0..dimension).flat_map(|row| (0..dimension).map(move |col| (row, col)));
        Self::from_entries(dimension, entries)
    }

    /// Construct a compact band pattern with the requested bandwidth.
    pub(crate) fn banded(dimension: usize, lower: usize, upper: usize) -> Result<Self, RadauError> {
        let entries = (0..dimension).flat_map(|col| {
            let first = col.saturating_sub(upper);
            let last = (col + lower + 1).min(dimension);
            (first..last).map(move |row| (row, col))
        });
        Self::from_entries(dimension, entries)
    }

    /// Construct the pattern required by a selected matrix layout.
    pub(crate) fn for_layout(
        dimension: usize,
        layout: RadauMatrixLayout,
        sparse_entries: Option<&[(usize, usize)]>,
    ) -> Result<Self, RadauError> {
        match layout {
            RadauMatrixLayout::Dense => Self::dense(dimension),
            RadauMatrixLayout::Sparse => Self::from_entries(
                dimension,
                sparse_entries
                    .ok_or(RadauConfigError::SparsePatternMissing)?
                    .iter()
                    .copied(),
            ),
            RadauMatrixLayout::Banded { lower, upper } => Self::banded(dimension, lower, upper),
        }
    }

    /// Return the state dimension represented by this pattern.
    pub(crate) fn dimension(&self) -> usize {
        self.dimension
    }

    /// Borrow entries in deterministic callback/output order.
    pub(crate) fn entries(&self) -> &[(usize, usize)] {
        &self.entries
    }

    /// Return the number of values required by a selected layout.
    pub(crate) fn values_len(&self, layout: RadauMatrixLayout) -> usize {
        match layout {
            RadauMatrixLayout::Dense => self.dimension * self.dimension,
            RadauMatrixLayout::Sparse => self.entries.len(),
            RadauMatrixLayout::Banded { lower, upper } => (lower + upper + 1) * self.dimension,
        }
    }

    /// Return the compact band slot for a matrix coordinate when in-band.
    pub(crate) fn compact_slot(
        &self,
        layout: RadauMatrixLayout,
        row: usize,
        col: usize,
    ) -> Option<usize> {
        match layout {
            RadauMatrixLayout::Dense => Some(row * self.dimension + col),
            RadauMatrixLayout::Banded { lower, upper } => (row < self.dimension
                && col < self.dimension
                && row + upper >= col
                && col + lower >= row)
                .then(|| (upper + row - col) * self.dimension + col),
            RadauMatrixLayout::Sparse => self
                .entries
                .iter()
                .position(|&(entry_row, entry_col)| (entry_row, entry_col) == (row, col)),
        }
    }

    /// Check whether a coordinate belongs to the selected storage layout.
    pub(crate) fn is_in_layout(&self, layout: RadauMatrixLayout, row: usize, col: usize) -> bool {
        self.compact_slot(layout, row, col).is_some()
    }
}
