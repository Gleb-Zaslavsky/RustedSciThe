//! Safe, optional Rust entry points for the SuperLU sparse direct solver.
//!
//! This facade hides the C ABI and dynamic-loader details. The bridge is
//! built for the current target and loaded only when the caller requests it.
//! BVP_sci and other solvers can use the same checked CSC/factor/solve API.

use std::path::Path;

use super::sparse_lu_superlu_experimental::SuperLuControl;
pub use super::sparse_lu_superlu_experimental::{
    SuperLuCsc32 as CscMatrix, SuperLuExperimentError as Error, SuperLuFactor as Factorization,
    SuperLuSolve as SolveReport,
};

/// Loaded SuperLU bridge with a version-checked ABI and retained native factors.
pub struct SolverLibrary(SuperLuControl);

impl SolverLibrary {
    /// Load a platform-native bridge and verify its SuperLU ABI version.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        SuperLuControl::load(path).map(Self)
    }

    /// The version tuple implemented by the loaded bridge.
    pub fn version(&self) -> (i32, i32, i32) {
        self.0.version()
    }

    /// Factor one CSC matrix and return an owned factorization for repeated RHS.
    pub fn factorize(&self, matrix: &CscMatrix) -> Result<Factorization, Error> {
        self.0.factorize(matrix)
    }

    /// Convenience one-shot sparse solve. Use `factorize` when reusing LU for
    /// multiple right-hand sides.
    pub fn solve(&self, matrix: &CscMatrix, rhs: &[f64]) -> Result<SolveReport, Error> {
        self.0.factor_and_solve(matrix, rhs)
    }
}

/// Load the bridge from `SUPERLU_LIBRARY` and solve one CSC system.
///
/// This convenience function is intended for simple standalone use. Repeated
/// solves should load [`SolverLibrary`] once and retain its factorization.
pub fn solve_csc(
    dimension: usize,
    column_pointers: &[usize],
    row_indices: &[usize],
    values: &[f64],
    rhs: &[f64],
) -> Result<SolveReport, Error> {
    let path = std::env::var_os("SUPERLU_LIBRARY").ok_or_else(|| {
        Error::MissingLibraryPath("set SUPERLU_LIBRARY to the bridge for this platform")
    })?;
    let matrix = CscMatrix::try_from_usize(dimension, column_pointers, row_indices, values)?;
    SolverLibrary::load(path)?.solve(&matrix, rhs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csc_matrix_type_is_shared_through_safe_facade() {
        let matrix = CscMatrix::try_from_usize(2, &[0, 2, 4], &[0, 1, 0, 1], &[2.0, 1.0, 1.0, 3.0])
            .expect("valid CSC matrix");
        assert_eq!(matrix.dimension(), 2);
        assert_eq!(matrix.nnz(), 4);
    }
}
