//! Banded linear algebra adapter for the BVP_sci global Jacobian.
//!
//! BVP_sci currently assembles the Newton Jacobian as a `faer` CSC matrix and
//! solves it through sparse LU. This module is the first safe layer for a
//! future Banded backend: profile the actual sparse structure, convert it into
//! the existing crate-wide banded storage, and solve through the shared banded
//! LU implementation.
//!
//! The adapter intentionally does not hide bandwidth. Boundary-condition rows
//! can make a mathematically block-banded BVP system look wide in scalar
//! storage, so callers should inspect `BvpSciBandedProfile` before selecting a
//! Banded backend in production.

use crate::numerical::BVP_sci::BVP_sci_faer::{faer_col, faer_mat};
use crate::somelinalg::banded::{Banded, BandedError, LapackStyleBandedLuFaithful};

/// Structural bandwidth profile of a BVP_sci global sparse Jacobian.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BvpSciBandedProfile {
    pub nrows: usize,
    pub ncols: usize,
    pub nnz: usize,
    /// Lower scalar bandwidth: max(row - col).
    pub kl: usize,
    /// Upper scalar bandwidth: max(col - row).
    pub ku: usize,
    /// Number of explicitly stored non-finite values.
    pub nonfinite_values: usize,
}

impl BvpSciBandedProfile {
    #[inline]
    pub fn is_square(&self) -> bool {
        self.nrows == self.ncols
    }

    #[inline]
    pub fn scalar_band_rows(&self) -> usize {
        self.kl + self.ku + 1
    }

    /// Compact band storage entries divided by sparse entries.
    ///
    /// This is a deliberately simple diagnostic for "is scalar banded storage
    /// sane here?". Large values usually mean boundary rows or parameter
    /// columns made the scalar band too wide for this backend shape.
    pub fn storage_amplification(&self) -> Option<f64> {
        if !self.is_square() || self.nnz == 0 {
            return None;
        }
        Some((self.scalar_band_rows() * self.ncols) as f64 / self.nnz as f64)
    }
}

pub fn infer_banded_profile(mat: &faer_mat) -> BvpSciBandedProfile {
    let (nrows, _) = mat.shape();
    infer_banded_profile_for_row_range(mat, 0, nrows)
}

/// Infer bandwidth using only a row interval.
///
/// This is useful for BVP matrices because the collocation block is usually
/// narrow, while the final boundary-condition rows may couple the first and
/// last mesh nodes and therefore make scalar band storage look much wider.
pub fn infer_banded_profile_for_row_range(
    mat: &faer_mat,
    row_start: usize,
    row_end: usize,
) -> BvpSciBandedProfile {
    let (nrows, ncols) = mat.shape();
    let row_start = row_start.min(nrows);
    let row_end = row_end.min(nrows).max(row_start);
    let mut profile = BvpSciBandedProfile {
        nrows,
        ncols,
        ..BvpSciBandedProfile::default()
    };

    let dyn_mat = mat.as_dyn();
    for col in 0..ncols {
        for row in dyn_mat.row_idx_of_col(col) {
            if row < row_start || row >= row_end {
                continue;
            }
            profile.nnz += 1;
            if row >= col {
                profile.kl = profile.kl.max(row - col);
            } else {
                profile.ku = profile.ku.max(col - row);
            }

            let value = mat[(row, col)];
            if !value.is_finite() {
                profile.nonfinite_values += 1;
            }
        }
    }

    profile
}

pub fn sparse_global_jac_to_banded(mat: &faer_mat) -> Result<Banded<f64>, BandedError> {
    let profile = infer_banded_profile(mat);
    if !profile.is_square() || profile.nonfinite_values > 0 {
        return Err(BandedError::DimensionMismatch);
    }

    let mut banded = Banded::<f64>::zeros(profile.ncols, profile.kl, profile.ku)?;
    let dyn_mat = mat.as_dyn();
    for col in 0..profile.ncols {
        for row in dyn_mat.row_idx_of_col(col) {
            banded.set(row, col, mat[(row, col)])?;
        }
    }

    Ok(banded)
}

/// Solve a BVP_sci sparse global Jacobian using the shared banded LU path.
pub fn solve_banded_lapack_faithful(
    mat: &faer_mat,
    rhs: &faer_col,
) -> Result<faer_col, BandedError> {
    let banded = sparse_global_jac_to_banded(mat)?;
    if rhs.nrows() != banded.n() {
        return Err(BandedError::DimensionMismatch);
    }

    let mut lu = LapackStyleBandedLuFaithful::new(banded.n(), banded.kl(), banded.ku())?;
    lu.factor_from(&banded)?;

    let mut x = (0..rhs.nrows()).map(|i| rhs[i]).collect::<Vec<_>>();
    lu.solve_in_place(&mut x)?;

    Ok(faer_col::from_fn(x.len(), |i| x[i]))
}
