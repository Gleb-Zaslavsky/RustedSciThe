//! Python binding for the stable pivoted tridiagonal LU kernel.
//!
//! The wrapper keeps the three diagonal bands as one-dimensional NumPy
//! arrays.  It never constructs a dense coefficient matrix; the native
//! factorization owns its compact copy and can be reused for many RHS values.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::prelude::*;

use rustedscithe_core::somelinalg::banded::{
    DEFAULT_TRIDIAGONAL_PIVOT_EPSILON, TridiagonalLuPp as NativeTridiagonalLu, TridiagonalPpError,
};

use crate::errors::{
    InvalidConfigurationError, SolverError, annotate, annotate_native, catch_panic, native_variant,
};

fn configuration_error(message: impl Into<String>, operation: &str) -> PyErr {
    annotate(
        InvalidConfigurationError::new_err(message.into()),
        "tridiagonal_lu_pp",
        "tridiagonal_lu_pp",
        operation,
    )
}

fn copy_band(
    values: &PyReadonlyArray1<'_, f64>,
    name: &str,
    operation: &str,
) -> PyResult<Vec<f64>> {
    let values = values.as_array().iter().copied().collect::<Vec<_>>();
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(configuration_error(
            format!("{name} must contain only finite values; index={index}, value={value}"),
            operation,
        ));
    }
    Ok(values)
}

fn map_tridiagonal_error(error: TridiagonalPpError, stage: &str, operation: &str) -> PyErr {
    let native_kind = native_variant(&error);
    let message = format!("{operation}: {error}");
    let mapped = match error {
        TridiagonalPpError::DimensionMismatch { .. }
        | TridiagonalPpError::InvalidPivotTolerance { .. }
        | TridiagonalPpError::RhsDimensionMismatch { .. }
        | TridiagonalPpError::InvalidRhsLayout { .. }
        | TridiagonalPpError::NonFiniteInput { .. } => InvalidConfigurationError::new_err(message),
        TridiagonalPpError::ZeroPivot { .. }
        | TridiagonalPpError::NonFiniteFactor { .. }
        | TridiagonalPpError::NonFiniteSolution { .. } => SolverError::new_err(message),
    };
    annotate_native(mapped, stage, "tridiagonal_lu_pp", operation, &native_kind)
}

fn validate_active_rhs(values: &[f64], n: usize, nrhs: usize, stride: usize) -> PyResult<()> {
    if stride < n {
        return Err(configuration_error(
            format!("stride must be at least n={n}; got stride={stride}"),
            "solve_multiple_in_place",
        ));
    }
    let required_len = nrhs.checked_mul(stride).ok_or_else(|| {
        configuration_error(
            format!("multiple-RHS layout overflows: nrhs={nrhs}, stride={stride}"),
            "solve_multiple_in_place",
        )
    })?;
    if values.len() < required_len {
        return Err(configuration_error(
            format!(
                "rhs buffer is too short: required={required_len}, actual={}",
                values.len()
            ),
            "solve_multiple_in_place",
        ));
    }
    for rhs_index in 0..nrhs {
        let start = rhs_index * stride;
        let active = &values[start..start + n];
        if let Some((index, value)) = active
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(configuration_error(
                format!(
                    "rhs contains a non-finite value: rhs={rhs_index}, index={index}, value={value}"
                ),
                "solve_multiple_in_place",
            ));
        }
    }
    Ok(())
}

/// Reusable pivoted LU factorization of a tridiagonal `float64` matrix.
///
/// `lower`, `diagonal` and `upper` have shapes `(n - 1,)`, `(n,)` and
/// `(n - 1,)`.  They are copied during construction; no dense matrix is
/// created, and later mutation of the input arrays cannot affect this object.
#[pyclass(name = "PivotedTridiagonalLu", module = "rustedscithe._native")]
pub struct PivotedTridiagonalLu {
    factorization: NativeTridiagonalLu,
}

#[pymethods]
impl PivotedTridiagonalLu {
    /// Factor three copied `float64` bands with native partial pivoting.
    ///
    /// `lower` and `upper` have shape `(n - 1,)`, while `diagonal` has shape
    /// `(n,)`. The factorization is complete when the constructor returns.
    #[new]
    #[pyo3(signature = (lower, diagonal, upper, pivot_epsilon=DEFAULT_TRIDIAGONAL_PIVOT_EPSILON))]
    fn new(
        lower: PyReadonlyArray1<'_, f64>,
        diagonal: PyReadonlyArray1<'_, f64>,
        upper: PyReadonlyArray1<'_, f64>,
        pivot_epsilon: f64,
    ) -> PyResult<Self> {
        let lower = copy_band(&lower, "lower", "construct")?;
        let diagonal = copy_band(&diagonal, "diagonal", "construct")?;
        let upper = copy_band(&upper, "upper", "construct")?;
        let factorization = catch_panic("factor", "tridiagonal_lu_pp", "native", || {
            NativeTridiagonalLu::factor_with_pivot_epsilon(&lower, &diagonal, &upper, pivot_epsilon)
                .map_err(|error| map_tridiagonal_error(error, "factor", "construct"))
        })?;
        Ok(Self { factorization })
    }

    /// Logical matrix dimension `n`.
    #[getter]
    fn n(&self) -> usize {
        self.factorization.n()
    }

    /// Absolute threshold used to reject unsafe pivots.
    #[getter]
    fn pivot_epsilon(&self) -> f64 {
        self.factorization.pivot_epsilon()
    }

    /// Owned native pivot decisions, one entry per elimination position.
    #[getter]
    fn ipiv(&self) -> Vec<usize> {
        self.factorization.ipiv().to_vec()
    }

    /// Approximate native allocation occupied by compact factors and pivots.
    #[getter]
    fn storage_bytes(&self) -> usize {
        let n = self.n();
        let f64_values = n.saturating_sub(1) * 2 + n.saturating_sub(2) + n;
        f64_values * std::mem::size_of::<f64>() + n * std::mem::size_of::<usize>()
    }

    /// Solve one RHS and return a detached NumPy vector.
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
        validate_active_rhs(&solution, self.n(), 1, self.n())?;
        catch_panic("solve", "tridiagonal_lu_pp", "native", || {
            self.factorization
                .solve_in_place(&mut solution)
                .map_err(|error| map_tridiagonal_error(error, "solve", "solve"))
        })?;
        Ok(PyArray1::from_vec(py, solution))
    }

    /// Solve one RHS in place.  The array must be writable and contiguous.
    fn solve_in_place(&self, mut rhs: PyReadwriteArray1<'_, f64>) -> PyResult<()> {
        let values = rhs.as_slice_mut().map_err(|_| {
            annotate(
                InvalidConfigurationError::new_err(
                    "rhs must be a one-dimensional contiguous float64 NumPy array",
                ),
                "solve",
                "tridiagonal_lu_pp",
                "solve_in_place",
            )
        })?;
        validate_active_rhs(values, self.n(), 1, self.n())?;
        catch_panic("solve", "tridiagonal_lu_pp", "native", || {
            self.factorization
                .solve_in_place(values)
                .map_err(|error| map_tridiagonal_error(error, "solve", "solve_in_place"))
        })
    }

    /// Solve an `(n, nrhs)` NumPy matrix without constructing a coefficient matrix.
    fn solve_multiple<'py>(
        &self,
        py: Python<'py>,
        rhs: numpy::PyReadonlyArray2<'_, f64>,
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
        let mut data = vec![0.0; self.n() * nrhs];
        for row in 0..self.n() {
            for column in 0..nrhs {
                data[column * self.n() + row] = rhs[(row, column)];
            }
        }
        validate_active_rhs(&data, self.n(), nrhs, self.n())?;
        catch_panic("solve", "tridiagonal_lu_pp", "native", || {
            self.factorization
                .solve_multiple_in_place(&mut data, nrhs, self.n())
                .map_err(|error| map_tridiagonal_error(error, "solve", "solve_multiple"))
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
                "tridiagonal_lu_pp",
                "solve_multiple_in_place",
            )
        })?;
        if nrhs == 0 {
            return Ok(());
        }
        validate_active_rhs(values, self.n(), nrhs, stride)?;
        catch_panic("solve", "tridiagonal_lu_pp", "native", || {
            self.factorization
                .solve_multiple_in_place(values, nrhs, stride)
                .map_err(|error| map_tridiagonal_error(error, "solve", "solve_multiple_in_place"))
        })
    }
}
