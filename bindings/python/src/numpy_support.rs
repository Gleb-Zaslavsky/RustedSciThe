//! Small, owned NumPy conversions for detached Python results.
//!
//! The conversion boundary deliberately copies solver-owned data into NumPy
//! arrays. This keeps result arrays valid after a solver is reused or dropped
//! and avoids exposing mutable Rust workspaces through a borrowed view.

use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Copy a Rust slice into an owned one-dimensional NumPy array.
///
/// The returned array is detached from the solver's buffers and remains valid
/// if the solver is subsequently reused or dropped.
pub(crate) fn vector<'py>(py: Python<'py>, values: &[f64]) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, values.to_vec())
}

/// Copy row-major nested vectors into a two-dimensional NumPy array.
///
/// Ragged rows are rejected rather than padded or truncated, preserving the
/// result shape promised by the solver API.
pub(crate) fn matrix<'py>(
    py: Python<'py>,
    rows: &[Vec<f64>],
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    PyArray2::from_vec2(py, rows)
        .map_err(|error| PyValueError::new_err(format!("ragged result matrix: {error}")))
}
