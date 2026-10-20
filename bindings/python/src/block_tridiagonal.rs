//! Python binding for the validated block-tridiagonal LU path.
//!
//! The public representation is three compact NumPy arrays with shapes
//! `(n_blocks - 1, block_size, block_size)`, `(n_blocks, block_size,
//! block_size)` and `(n_blocks - 1, block_size, block_size)`.  The wrapper
//! copies those blocks once and never materializes the global dense matrix.
//! This route intentionally exposes the consistent Rust implementation only;
//! the older block LU and bordered solver remain native/internal routes.

use numpy::{
    PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadwriteArray1,
};
use pyo3::prelude::*;

use rustedscithe_core::somelinalg::banded::{
    BandedError, BlockTridiagonal as NativeBlockTridiagonal,
    BlockTridiagonalLuConsistent as NativeBlockTridiagonalLu,
};

use crate::errors::{
    InvalidConfigurationError, SolverError, annotate, annotate_native, catch_panic, native_variant,
};

fn configuration_error(message: impl Into<String>, operation: &str) -> PyErr {
    annotate(
        InvalidConfigurationError::new_err(message.into()),
        "block_tridiagonal",
        "block_tridiagonal",
        operation,
    )
}

fn map_block_error(error: BandedError, stage: &str, operation: &str) -> PyErr {
    let native_kind = native_variant(&error);
    let message = format!("{operation}: {error}");
    let mapped = match error {
        BandedError::DimensionMismatch
        | BandedError::InvalidRhsLayout { .. }
        | BandedError::OutOfBounds { .. }
        | BandedError::NonFiniteInput { .. } => InvalidConfigurationError::new_err(message),
        BandedError::NotFactorized
        | BandedError::Singular { .. }
        | BandedError::ZeroPivot { .. }
        | BandedError::NonFiniteFactor { .. }
        | BandedError::NonFiniteSolution { .. }
        | BandedError::NonFiniteCallbackValue { .. }
        | BandedError::BackendUnavailable
        | BandedError::LapackArgument { .. } => SolverError::new_err(message),
        BandedError::InvalidBand { .. } => InvalidConfigurationError::new_err(message),
    };
    annotate_native(mapped, stage, "block_tridiagonal", operation, &native_kind)
}

fn flatten_blocks(
    values: &PyReadonlyArray3<'_, f64>,
    expected_blocks: usize,
    block_size: usize,
    name: &str,
) -> PyResult<Vec<f64>> {
    let array = values.as_array();
    let shape = array.shape();
    if shape != [expected_blocks, block_size, block_size] {
        return Err(configuration_error(
            format!(
                "{name} must have shape ({expected_blocks}, {block_size}, {block_size}), got {:?}",
                shape
            ),
            "construct",
        ));
    }
    Ok(array.iter().copied().collect())
}

fn flatten_and_build_matrix(
    lower: &PyReadonlyArray3<'_, f64>,
    diagonal: &PyReadonlyArray3<'_, f64>,
    upper: &PyReadonlyArray3<'_, f64>,
) -> PyResult<NativeBlockTridiagonal> {
    let diagonal_shape = diagonal.as_array().shape().to_vec();
    if diagonal_shape.len() != 3 || diagonal_shape[0] == 0 || diagonal_shape[1] == 0 {
        return Err(configuration_error(
            "diagonal must be a non-empty three-dimensional float64 array",
            "construct",
        ));
    }
    let n_blocks = diagonal_shape[0];
    let block_size = diagonal_shape[1];
    if diagonal_shape[2] != block_size {
        return Err(configuration_error(
            "diagonal blocks must be square",
            "construct",
        ));
    }

    let lower = flatten_blocks(lower, n_blocks.saturating_sub(1), block_size, "lower")?;
    let diagonal = flatten_blocks(diagonal, n_blocks, block_size, "diagonal")?;
    let upper = flatten_blocks(upper, n_blocks.saturating_sub(1), block_size, "upper")?;
    NativeBlockTridiagonal::from_row_major_blocks(n_blocks, block_size, &lower, &diagonal, &upper)
        .map_err(|error| map_block_error(error, "construct", "BlockTridiagonalLuConsistent"))
}

/// Compact reusable LU factorization for a block-tridiagonal float64 matrix.
///
/// This class retains the structured matrix and factors.  It supports direct
/// solves, multiple RHS layouts, factor-quality diagnostics and the native
/// iterative-refinement policy.  It has no native telemetry/logging contract;
/// the Python binding does not invent one.
#[pyclass(name = "BlockTridiagonalLuConsistent", module = "rustedscithe._native")]
pub struct BlockTridiagonalLuConsistent {
    matrix: NativeBlockTridiagonal,
    factorization: NativeBlockTridiagonalLu,
}

#[pymethods]
impl BlockTridiagonalLuConsistent {
    /// Copy compact lower, diagonal and upper blocks and factor them.
    ///
    /// The arrays must have shapes `(n_blocks - 1, bs, bs)`,
    /// `(n_blocks, bs, bs)` and `(n_blocks - 1, bs, bs)`. The global matrix
    /// is never materialized as a dense NumPy array.
    #[new]
    fn new(
        lower: PyReadonlyArray3<'_, f64>,
        diagonal: PyReadonlyArray3<'_, f64>,
        upper: PyReadonlyArray3<'_, f64>,
    ) -> PyResult<Self> {
        let matrix = flatten_and_build_matrix(&lower, &diagonal, &upper)?;
        let n_blocks = matrix.n_blocks();
        let block_size = matrix.block_size();
        let mut factorization = NativeBlockTridiagonalLu::new(n_blocks, block_size)
            .map_err(|error| map_block_error(error, "factor", "construct"))?;
        catch_panic("factor", "block_tridiagonal", "native", || {
            factorization
                .factor_from(&matrix)
                .map_err(|error| map_block_error(error, "factor", "construct"))
        })?;
        Ok(Self {
            matrix,
            factorization,
        })
    }

    /// Number of block rows in the structured system.
    #[getter]
    fn n_blocks(&self) -> usize {
        self.factorization.n_blocks()
    }

    /// Side length of every dense diagonal block.
    #[getter]
    fn block_size(&self) -> usize {
        self.factorization.block_size()
    }

    /// Global scalar dimension `n_blocks * block_size`.
    #[getter]
    fn n(&self) -> usize {
        self.factorization.n()
    }

    /// Whether the constructor completed a usable native factorization.
    #[getter]
    fn is_factorized(&self) -> bool {
        self.factorization.is_factorized()
    }

    /// Approximate bytes retained by the structured factorization workspace.
    #[getter]
    fn storage_bytes(&self) -> usize {
        self.factorization.storage_bytes()
    }

    /// Relative residual of the cached factorization diagnostics.
    #[getter]
    fn factor_residual_relative(&self) -> PyResult<f64> {
        self.factorization
            .cached_factor_residual_relative()
            .ok_or_else(|| configuration_error("factor residual is unavailable", "diagnostics"))
    }

    /// Return `(minimum_abs_diagonal_pivot, maximum_multiplier_norm)`.
    fn condition_diagnostics(&self) -> (f64, f64) {
        self.factorization.condition_diagnostics()
    }

    /// Return `(block_index, swap_count, maximum_swap_distance)` tuples.
    fn pivot_diagnostics(&self) -> Vec<(usize, usize, usize)> {
        self.factorization
            .pivot_diagnostics()
            .into_iter()
            .map(|diagnostic| {
                (
                    diagnostic.block_index,
                    diagnostic.num_nontrivial_swaps,
                    diagnostic.max_swap_distance,
                )
            })
            .collect()
    }

    /// Solve one finite RHS and return a detached `(n,)` NumPy array.
    fn solve<'py>(
        &self,
        py: Python<'py>,
        rhs: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut solution = rhs.as_array().iter().copied().collect::<Vec<_>>();
        if solution.len() != self.n() {
            return Err(configuration_error(
                format!(
                    "rhs must have shape ({},), got ({},)",
                    self.n(),
                    solution.len()
                ),
                "solve",
            ));
        }
        catch_panic("solve", "block_tridiagonal", "native", || {
            self.factorization
                .solve_in_place(&mut solution)
                .map_err(|error| map_block_error(error, "solve", "solve"))
        })?;
        Ok(PyArray1::from_vec(py, solution))
    }

    /// Solve a writable contiguous `(n,)` RHS in place.
    fn solve_in_place(&self, mut rhs: PyReadwriteArray1<'_, f64>) -> PyResult<()> {
        let values = rhs.as_slice_mut().map_err(|_| {
            annotate(
                InvalidConfigurationError::new_err(
                    "rhs must be a one-dimensional contiguous float64 NumPy array",
                ),
                "solve",
                "block_tridiagonal",
                "solve_in_place",
            )
        })?;
        catch_panic("solve", "block_tridiagonal", "native", || {
            self.factorization
                .solve_in_place(values)
                .map_err(|error| map_block_error(error, "solve", "solve_in_place"))
        })
    }

    /// Solve an `(n, nrhs)` NumPy matrix without densifying the coefficients.
    fn solve_multiple<'py>(
        &self,
        py: Python<'py>,
        rhs: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let rhs = rhs.as_array();
        if rhs.nrows() != self.n() || rhs.ncols() == 0 {
            return Err(configuration_error(
                format!(
                    "rhs must have shape (n, nrhs) with n={}, got ({}, {})",
                    self.n(),
                    rhs.nrows(),
                    rhs.ncols()
                ),
                "solve_multiple",
            ));
        }
        let nrhs = rhs.ncols();
        let mut data = vec![
            0.0;
            self.n().checked_mul(nrhs).ok_or_else(|| {
                configuration_error("multiple-RHS layout overflows", "solve_multiple")
            })?
        ];
        for row in 0..self.n() {
            for column in 0..nrhs {
                data[column * self.n() + row] = rhs[(row, column)];
            }
        }
        catch_panic("solve", "block_tridiagonal", "native", || {
            self.factorization
                .solve_multiple_in_place(&mut data, nrhs, self.n())
                .map_err(|error| map_block_error(error, "solve", "solve_multiple"))
        })?;
        let rows = (0..self.n())
            .map(|row| {
                (0..nrhs)
                    .map(|column| data[column * self.n() + row])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        PyArray2::from_vec2(py, &rows).map_err(|error| {
            configuration_error(format!("invalid result shape: {error}"), "solve_multiple")
        })
    }

    /// Solve padded one-dimensional RHS storage in place.
    fn solve_multiple_in_place(
        &self,
        mut rhs: PyReadwriteArray1<'_, f64>,
        nrhs: usize,
        stride: usize,
    ) -> PyResult<()> {
        let values = rhs.as_slice_mut().map_err(|_| {
            annotate(
                InvalidConfigurationError::new_err(
                    "rhs must be a one-dimensional contiguous float64 NumPy array",
                ),
                "solve",
                "block_tridiagonal",
                "solve_multiple_in_place",
            )
        })?;
        catch_panic("solve", "block_tridiagonal", "native", || {
            self.factorization
                .solve_multiple_in_place(values, nrhs, stride)
                .map_err(|error| map_block_error(error, "solve", "solve_multiple_in_place"))
        })
    }

    /// Solve with the native iterative-refinement policy and return its report.
    fn solve_with_refinement(
        &self,
        rhs: PyReadonlyArray1<'_, f64>,
        refinement_steps: usize,
    ) -> PyResult<BlockTridiagonalRefinementResult> {
        let mut solution = rhs.as_array().iter().copied().collect::<Vec<_>>();
        if solution.len() != self.n() {
            return Err(configuration_error(
                format!(
                    "rhs must have shape ({},), got ({},)",
                    self.n(),
                    solution.len()
                ),
                "solve_with_refinement",
            ));
        }
        let report = catch_panic("solve", "block_tridiagonal", "native", || {
            self.factorization
                .solve_in_place_with_iterative_refinement_report(
                    &self.matrix,
                    &mut solution,
                    refinement_steps,
                )
                .map_err(|error| map_block_error(error, "solve", "solve_with_refinement"))
        })?;
        Ok(BlockTridiagonalRefinementResult {
            solution,
            requested_steps: report.requested_steps,
            accepted_steps: report.accepted_steps,
            refinement_attempted: report.refinement_attempted,
            direct_relative_residual: report.direct_relative_residual,
            final_relative_residual: report.final_relative_residual,
        })
    }
}

/// Detached result of a block-tridiagonal iterative-refinement solve.
#[pyclass(
    name = "BlockTridiagonalRefinementResult",
    module = "rustedscithe._native"
)]
pub struct BlockTridiagonalRefinementResult {
    solution: Vec<f64>,
    requested_steps: usize,
    accepted_steps: usize,
    refinement_attempted: bool,
    direct_relative_residual: f64,
    final_relative_residual: f64,
}

#[pymethods]
impl BlockTridiagonalRefinementResult {
    /// Return a detached copy of the refined solution.
    fn solution_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.solution.clone())
    }

    /// Number of refinement iterations requested by the caller.
    #[getter]
    fn requested_steps(&self) -> usize {
        self.requested_steps
    }

    /// Number of refinement iterations accepted by the native policy.
    #[getter]
    fn accepted_steps(&self) -> usize {
        self.accepted_steps
    }

    /// Whether refinement actually evaluated a correction step.
    #[getter]
    fn refinement_attempted(&self) -> bool {
        self.refinement_attempted
    }

    /// Relative residual before refinement.
    #[getter]
    fn direct_relative_residual(&self) -> f64 {
        self.direct_relative_residual
    }

    /// Relative residual after the accepted refinement steps.
    #[getter]
    fn final_relative_residual(&self) -> f64 {
        self.final_relative_residual
    }
}
