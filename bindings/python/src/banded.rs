//! Python bindings for the stable LAPACK-style compact banded LU path.
//!
//! The public representation is deliberately compact:
//! `values[ku + i - j, j] = A[i, j]`.  Neither construction nor solving
//! materializes an `n x n` dense matrix.  A factorization owns its copied
//! native workspace, so changing the source NumPy array cannot change an
//! already factorized object.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1};
use pyo3::prelude::*;

use rustedscithe_core::somelinalg::banded::{
    Banded as NativeBanded, BandedError, LapackStyleBandedLuFaithful as NativeBandedLu,
    residual_linf,
};

use crate::errors::{
    InvalidConfigurationError, SolverError, annotate, annotate_native, catch_panic, native_variant,
};

fn validate_band_shape(n: usize, kl: usize, ku: usize) -> PyResult<()> {
    if n == 0 {
        return Err(InvalidConfigurationError::new_err("n must be positive"));
    }
    if kl >= n || ku >= n {
        return Err(InvalidConfigurationError::new_err(format!(
            "kl and ku must be smaller than n; got n={n}, kl={kl}, ku={ku}"
        )));
    }
    Ok(())
}

fn finite_values(values: &[f64], name: &str) -> PyResult<()> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(InvalidConfigurationError::new_err(format!(
            "{name} must contain only finite values; index={index}, value={value}"
        )));
    }
    Ok(())
}

fn compact_from_array(
    values: &PyReadonlyArray2<'_, f64>,
    n: usize,
    kl: usize,
    ku: usize,
) -> PyResult<Vec<f64>> {
    let values = values.as_array();
    let expected_rows = kl + ku + 1;
    if values.nrows() != expected_rows || values.ncols() != n {
        return Err(InvalidConfigurationError::new_err(format!(
            "compact banded values must have shape ({expected_rows}, {n}), got ({}, {})",
            values.nrows(),
            values.ncols()
        )));
    }
    let owned = values.iter().copied().collect::<Vec<_>>();
    finite_values(&owned, "compact banded values")?;
    Ok(owned)
}

fn map_banded_error(error: BandedError, stage: &str, operation: &str) -> PyErr {
    let native_kind = native_variant(&error);
    let message = format!("{operation}: {error}");
    let mapped = match error {
        BandedError::DimensionMismatch
        | BandedError::InvalidBand { .. }
        | BandedError::OutOfBounds { .. }
        | BandedError::InvalidRhsLayout { .. }
        | BandedError::NonFiniteInput { .. } => InvalidConfigurationError::new_err(message),
        BandedError::NotFactorized
        | BandedError::Singular { .. }
        | BandedError::ZeroPivot { .. } => SolverError::new_err(message),
        BandedError::NonFiniteCallbackValue { .. }
        | BandedError::NonFiniteFactor { .. }
        | BandedError::NonFiniteSolution { .. }
        | BandedError::BackendUnavailable
        | BandedError::LapackArgument { .. } => SolverError::new_err(message),
    };
    annotate_native(mapped, stage, "lapack_banded", operation, &native_kind)
}

fn map_array_layout_error(name: &str) -> PyErr {
    InvalidConfigurationError::new_err(format!(
        "{name} must be a one-dimensional contiguous float64 NumPy array"
    ))
}

fn matrix_from_bands(
    diagonal: &PyReadonlyArray1<'_, f64>,
    lower: Option<&PyReadonlyArray2<'_, f64>>,
    upper: Option<&PyReadonlyArray2<'_, f64>>,
) -> PyResult<NativeBanded<f64>> {
    let diagonal = diagonal.as_array();
    let n = diagonal.len();
    let kl = lower.map(|values| values.as_array().nrows()).unwrap_or(0);
    let ku = upper.map(|values| values.as_array().nrows()).unwrap_or(0);
    validate_band_shape(n, kl, ku)?;

    let diagonal_values = diagonal.iter().copied().collect::<Vec<_>>();
    finite_values(&diagonal_values, "diagonal")?;
    let mut matrix = NativeBanded::zeros(n, kl, ku)
        .map_err(|error| map_banded_error(error, "construct", "BandedMatrix.from_bands"))?;
    for (index, value) in diagonal_values.into_iter().enumerate() {
        matrix
            .set(index, index, value)
            .map_err(|error| map_banded_error(error, "construct", "BandedMatrix.from_bands"))?;
    }

    if let Some(lower) = lower {
        let values = lower.as_array();
        if values.ncols() != n {
            return Err(InvalidConfigurationError::new_err(format!(
                "lower bands must have shape ({}, {n}), got ({}, {})",
                values.nrows(),
                values.nrows(),
                values.ncols()
            )));
        }
        let owned = values.iter().copied().collect::<Vec<_>>();
        finite_values(&owned, "lower bands")?;
        for row in 0..kl {
            for column in 0..n.saturating_sub(row + 1) {
                matrix
                    .set(column + row + 1, column, values[(row, column)])
                    .map_err(|error| {
                        map_banded_error(error, "construct", "BandedMatrix.from_bands")
                    })?;
            }
        }
    }

    if let Some(upper) = upper {
        let values = upper.as_array();
        if values.ncols() != n {
            return Err(InvalidConfigurationError::new_err(format!(
                "upper bands must have shape ({}, {n}), got ({}, {})",
                values.nrows(),
                values.nrows(),
                values.ncols()
            )));
        }
        let owned = values.iter().copied().collect::<Vec<_>>();
        finite_values(&owned, "upper bands")?;
        for row in 0..ku {
            for column in (row + 1)..n {
                matrix
                    .set(column - row - 1, column, values[(row, column)])
                    .map_err(|error| {
                        map_banded_error(error, "construct", "BandedMatrix.from_bands")
                    })?;
            }
        }
    }

    Ok(matrix)
}

/// Owned compact banded matrix.
///
/// `values` has shape `(kl + ku + 1, n)` and uses
/// `values[ku + i - j, j] = A[i, j]`.  The array is copied once at the
/// Python boundary; later mutation of the caller's NumPy array is harmless.
#[pyclass(name = "BandedMatrix", module = "rustedscithe._native")]
pub struct BandedMatrix {
    matrix: NativeBanded<f64>,
}

#[pymethods]
impl BandedMatrix {
    /// Create an owned compact banded matrix from LAPACK-style storage.
    ///
    /// `values` must have shape `(kl + ku + 1, n)` and is copied before the
    /// native matrix is published. The constructor never densifies the input.
    #[new]
    #[pyo3(signature = (n, kl, ku, values))]
    fn new(n: usize, kl: usize, ku: usize, values: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        validate_band_shape(n, kl, ku)?;
        let data = compact_from_array(&values, n, kl, ku)?;
        let matrix = NativeBanded::from_vec(n, kl, ku, data)
            .map_err(|error| map_banded_error(error, "construct", "BandedMatrix"))?;
        Ok(Self { matrix })
    }

    /// Construct from diagonal, lower and upper bands.
    ///
    /// `lower[r, j]` means `A[j + r + 1, j]`; `upper[r, j]` means
    /// `A[j - r - 1, j]`. Entries outside the matrix edge are ignored.
    #[staticmethod]
    #[pyo3(signature = (diagonal, lower=None, upper=None))]
    fn from_bands(
        diagonal: PyReadonlyArray1<'_, f64>,
        lower: Option<PyReadonlyArray2<'_, f64>>,
        upper: Option<PyReadonlyArray2<'_, f64>>,
    ) -> PyResult<Self> {
        let matrix = matrix_from_bands(&diagonal, lower.as_ref(), upper.as_ref())?;
        Ok(Self { matrix })
    }

    /// Number of rows and columns in the logical square matrix.
    #[getter]
    fn n(&self) -> usize {
        self.matrix.n()
    }

    /// Number of stored lower diagonals.
    #[getter]
    fn kl(&self) -> usize {
        self.matrix.kl()
    }

    /// Number of stored upper diagonals.
    #[getter]
    fn ku(&self) -> usize {
        self.matrix.ku()
    }

    /// Shape of the compact storage, `(kl + ku + 1, n)`.
    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.matrix.rows(), self.matrix.n())
    }

    /// Approximate bytes owned by the compact coefficient storage.
    #[getter]
    fn storage_bytes(&self) -> usize {
        self.matrix.as_slice().len() * std::mem::size_of::<f64>()
    }

    /// Return a detached compact NumPy copy with shape `(kl + ku + 1, n)`.
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let rows = (0..self.matrix.rows())
            .map(|row| {
                self.matrix.as_slice()[row * self.matrix.n()..(row + 1) * self.matrix.n()].to_vec()
            })
            .collect::<Vec<_>>();
        PyArray2::from_vec2(py, &rows).map_err(|error| {
            InvalidConfigurationError::new_err(format!("invalid matrix shape: {error}"))
        })
    }

    /// Read a logical matrix entry, returning zero outside the stored bands.
    fn get(&self, row: usize, column: usize) -> PyResult<f64> {
        if row >= self.n() || column >= self.n() {
            return Err(InvalidConfigurationError::new_err(format!(
                "matrix index ({row}, {column}) is outside n={}",
                self.n()
            )));
        }
        Ok(self.matrix.get(row, column).copied().unwrap_or(0.0))
    }

    /// Set one logical entry without changing the matrix bandwidth.
    fn set(&mut self, row: usize, column: usize, value: f64) -> PyResult<()> {
        if !value.is_finite() {
            return Err(InvalidConfigurationError::new_err(
                "banded matrix values must be finite",
            ));
        }
        self.matrix
            .set(row, column, value)
            .map_err(|error| map_banded_error(error, "assembly", "BandedMatrix.set"))
    }
}

/// Reusable faithful LAPACK-style banded LU factorization.
///
/// The factorization owns native compact workspace and can solve many RHS
/// vectors without repeating the factorization. This route has no native
/// telemetry or logging contract; Python therefore does not invent one.
#[pyclass(name = "LapackStyleBandedLuFaithful", module = "rustedscithe._native")]
pub struct LapackStyleBandedLuFaithful {
    factorization: NativeBandedLu,
    pivot_epsilon: f64,
}

#[pymethods]
impl LapackStyleBandedLuFaithful {
    /// Allocate an unfactorized reusable banded LU workspace.
    ///
    /// `pivot_epsilon` controls the native zero-pivot safety threshold. Call
    /// `factor` or `factor_compact` before calling any solve method.
    #[new]
    #[pyo3(signature = (n, kl, ku, pivot_epsilon=1e-14))]
    fn new(n: usize, kl: usize, ku: usize, pivot_epsilon: f64) -> PyResult<Self> {
        validate_band_shape(n, kl, ku)?;
        if !pivot_epsilon.is_finite() || pivot_epsilon < 0.0 {
            return Err(InvalidConfigurationError::new_err(
                "pivot_epsilon must be finite and nonnegative",
            ));
        }
        let mut factorization = NativeBandedLu::new(n, kl, ku)
            .map_err(|error| map_banded_error(error, "construct", "LapackStyleBandedLuFaithful"))?;
        factorization.set_pivot_epsilon(pivot_epsilon);
        Ok(Self {
            factorization,
            pivot_epsilon,
        })
    }

    /// Number of rows and columns in the factorized matrix.
    #[getter]
    fn n(&self) -> usize {
        self.factorization.n()
    }

    /// Number of lower diagonals retained by the factorization.
    #[getter]
    fn kl(&self) -> usize {
        self.factorization.kl()
    }

    /// Number of upper diagonals retained by the factorization.
    #[getter]
    fn ku(&self) -> usize {
        self.factorization.ku()
    }

    #[getter]
    fn kv(&self) -> usize {
        self.factorization.kv()
    }

    #[getter]
    fn ldab(&self) -> usize {
        self.factorization.ldab()
    }

    #[getter]
    fn pivot_epsilon(&self) -> f64 {
        self.pivot_epsilon
    }

    /// Whether a successful factorization is currently stored.
    #[getter]
    fn is_factorized(&self) -> bool {
        self.factorization.is_factorized()
    }

    /// Return the native pivot indices as an owned Python list.
    #[getter]
    fn ipiv(&self) -> Vec<usize> {
        self.factorization.ipiv().to_vec()
    }

    /// Change the pivot threshold before the next factorization.
    fn set_pivot_epsilon(&mut self, value: f64) -> PyResult<()> {
        if !value.is_finite() || value < 0.0 {
            return Err(InvalidConfigurationError::new_err(
                "pivot_epsilon must be finite and nonnegative",
            ));
        }
        self.factorization.set_pivot_epsilon(value);
        self.pivot_epsilon = value;
        Ok(())
    }

    /// Factor an owned compact `BandedMatrix` and retain the native workspace.
    fn factor(&mut self, matrix: &BandedMatrix) -> PyResult<()> {
        catch_panic("factor", "lapack_banded", "native", || {
            self.factorization
                .factor_from(&matrix.matrix)
                .map_err(|error| map_banded_error(error, "factor", "factor"))
        })
    }

    /// Factor a compact NumPy matrix without constructing a temporary
    /// `BandedMatrix` object.
    fn factor_compact(&mut self, values: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let data = compact_from_array(&values, self.n(), self.kl(), self.ku())?;
        catch_panic("factor", "lapack_banded", "native", || {
            self.factorization
                .factor_from_compact(&data)
                .map_err(|error| map_banded_error(error, "factor", "factor_compact"))
        })
    }

    /// Check the relative factorization residual against the original matrix.
    fn factor_residual_relative(&self, matrix: &BandedMatrix) -> PyResult<f64> {
        catch_panic("diagnostics", "lapack_banded", "native", || {
            self.factorization
                .factor_residual_relative_true(&matrix.matrix)
                .map_err(|error| map_banded_error(error, "diagnostics", "factor_residual_relative"))
        })
    }

    /// Solve one finite RHS and return a detached `(n,)` NumPy array.
    fn solve<'py>(
        &self,
        py: Python<'py>,
        rhs: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut solution = rhs.as_array().iter().copied().collect::<Vec<_>>();
        finite_values(&solution, "rhs")?;
        catch_panic("solve", "lapack_banded", "native", || {
            self.factorization
                .solve_in_place(&mut solution)
                .map_err(|error| map_banded_error(error, "solve", "solve"))
        })?;
        finite_values(&solution, "solution")?;
        Ok(PyArray1::from_vec(py, solution))
    }

    /// Solve in place only for a one-dimensional contiguous NumPy array.
    fn solve_in_place(&self, mut rhs: PyReadwriteArray1<'_, f64>) -> PyResult<()> {
        let values = rhs.as_slice_mut().map_err(|_| {
            annotate(
                map_array_layout_error("rhs"),
                "solve",
                "lapack_banded",
                "native",
            )
        })?;
        finite_values(values, "rhs")?;
        catch_panic("solve", "lapack_banded", "native", || {
            self.factorization
                .solve_in_place(values)
                .map_err(|error| map_banded_error(error, "solve", "solve_in_place"))
        })?;
        finite_values(values, "solution")
    }

    /// Solve a Python `(n, nrhs)` matrix. The native workspace is column-major;
    /// conversion is explicit and does not involve an `n x n` coefficient matrix.
    fn solve_multiple<'py>(
        &self,
        py: Python<'py>,
        rhs: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let rhs = rhs.as_array();
        if rhs.nrows() != self.n() || rhs.ncols() == 0 {
            return Err(InvalidConfigurationError::new_err(format!(
                "rhs must have shape (n, nrhs) with n={}, got ({}, {})",
                self.n(),
                rhs.nrows(),
                rhs.ncols()
            )));
        }
        let nrhs = rhs.ncols();
        let mut data = vec![0.0; self.n() * nrhs];
        for row in 0..self.n() {
            for column in 0..nrhs {
                data[column * self.n() + row] = rhs[(row, column)];
            }
        }
        finite_values(&data, "rhs")?;
        catch_panic("solve", "lapack_banded", "native", || {
            self.factorization
                .solve_multiple_in_place(&mut data, nrhs, self.n())
                .map_err(|error| map_banded_error(error, "solve", "solve_multiple"))
        })?;
        finite_values(&data, "solution")?;
        let rows = (0..self.n())
            .map(|row| {
                (0..nrhs)
                    .map(|column| data[column * self.n() + row])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        PyArray2::from_vec2(py, &rows).map_err(|error| {
            InvalidConfigurationError::new_err(format!("invalid result shape: {error}"))
        })
    }

    /// Solve with explicit iterative refinement and return residual metadata.
    fn solve_with_refinement(
        &self,
        matrix: &BandedMatrix,
        rhs: PyReadonlyArray1<'_, f64>,
        max_iters: usize,
        tolerance: f64,
    ) -> PyResult<BandedRefinementResult> {
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(InvalidConfigurationError::new_err(
                "tolerance must be finite and positive",
            ));
        }
        let input = rhs.as_array().iter().copied().collect::<Vec<_>>();
        finite_values(&input, "rhs")?;
        let mut solution = input.clone();
        let iterations = catch_panic("solve", "lapack_banded", "native", || {
            self.factorization
                .solve_banded_in_place_with_refinement(
                    &matrix.matrix,
                    &mut solution,
                    max_iters,
                    tolerance,
                )
                .map_err(|error| map_banded_error(error, "solve", "solve_with_refinement"))
        })?;
        finite_values(&solution, "solution")?;
        let residual = residual_linf(&matrix.matrix, &solution, &input).map_err(|error| {
            map_banded_error(error, "diagnostics", "solve_with_refinement.residual")
        })?;
        let rhs_norm = input
            .iter()
            .map(|value| value.abs())
            .fold(0.0, f64::max)
            .max(1.0);
        let relative_residual = residual / rhs_norm;
        Ok(BandedRefinementResult {
            solution,
            iterations,
            residual_inf: residual,
            relative_residual_inf: relative_residual,
            tolerance,
            converged: relative_residual < tolerance,
        })
    }
}

/// Detached result of an explicit iterative-refinement solve.
#[pyclass(name = "BandedRefinementResult", module = "rustedscithe._native")]
pub struct BandedRefinementResult {
    solution: Vec<f64>,
    iterations: usize,
    residual_inf: f64,
    relative_residual_inf: f64,
    tolerance: f64,
    converged: bool,
}

#[pymethods]
impl BandedRefinementResult {
    /// Return a detached copy of the refined solution.
    fn solution_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.solution.clone())
    }

    /// Number of accepted refinement iterations.
    #[getter]
    fn iterations(&self) -> usize {
        self.iterations
    }

    /// Absolute infinity-norm residual after refinement.
    #[getter]
    fn residual_inf(&self) -> f64 {
        self.residual_inf
    }

    /// RHS-scaled infinity-norm residual after refinement.
    #[getter]
    fn relative_residual_inf(&self) -> f64 {
        self.relative_residual_inf
    }

    /// Relative residual threshold requested by the caller.
    #[getter]
    fn tolerance(&self) -> f64 {
        self.tolerance
    }

    /// Whether the final relative residual met `tolerance`.
    #[getter]
    fn converged(&self) -> bool {
        self.converged
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::PyArrayMethods;

    fn tridiagonal_values<'py>(py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray2::from_vec2(
            py,
            &[
                vec![0.0, 1.0, 1.0, 1.0],
                vec![4.0, 4.0, 4.0, 4.0],
                vec![2.0, 2.0, 2.0, 0.0],
            ],
        )
        .unwrap()
    }

    #[test]
    fn compact_matrix_and_reusable_factorization_match_known_solution() {
        Python::initialize();
        Python::attach(|py| {
            let values = tridiagonal_values(py);
            let matrix = BandedMatrix::new(4, 1, 1, values.readonly()).unwrap();
            assert_eq!(matrix.shape(), (3, 4));
            assert_eq!(matrix.storage_bytes(), 3 * 4 * std::mem::size_of::<f64>());
            assert_eq!(matrix.get(0, 0).unwrap(), 4.0);
            assert_eq!(matrix.get(0, 3).unwrap(), 0.0);

            let mut lu = LapackStyleBandedLuFaithful::new(4, 1, 1, 1e-14).unwrap();
            lu.factor(&matrix).unwrap();
            assert!(lu.is_factorized());
            assert_eq!(lu.ldab(), 4);

            let rhs = PyArray1::from_vec(py, vec![6.0, 13.0, 20.0, 22.0]);
            let solution = lu.solve(py, rhs.readonly()).unwrap();
            let solution_view = solution.readonly();
            let solution = solution_view.as_slice().unwrap();
            assert_eq!(solution, &[1.0, 2.0, 3.0, 4.0]);

            let second_rhs = PyArray1::from_vec(py, vec![9.0, 8.0, 2.0, 0.0]);
            let second = lu.solve(py, second_rhs.readonly()).unwrap();
            assert_eq!(second.readonly().as_slice().unwrap(), &[2.0, 1.0, 0.0, 0.0]);
        });
    }

    #[test]
    fn separate_bands_and_multiple_rhs_preserve_structured_storage() {
        Python::initialize();
        Python::attach(|py| {
            let diagonal = PyArray1::from_vec(py, vec![4.0, 4.0, 4.0, 4.0]);
            let lower = PyArray2::from_vec2(py, &[vec![2.0, 2.0, 2.0, 0.0]]).unwrap();
            let upper = PyArray2::from_vec2(py, &[vec![0.0, 1.0, 1.0, 1.0]]).unwrap();
            let matrix = BandedMatrix::from_bands(
                diagonal.readonly(),
                Some(lower.readonly()),
                Some(upper.readonly()),
            )
            .unwrap();
            let compact_values = matrix.values(py).unwrap();
            assert_eq!(compact_values.readonly().as_array()[[0, 1]], 1.0);

            let mut lu = LapackStyleBandedLuFaithful::new(4, 1, 1, 1e-14).unwrap();
            lu.factor(&matrix).unwrap();
            let rhs = PyArray2::from_vec2(
                py,
                &[
                    vec![6.0, 9.0],
                    vec![13.0, 8.0],
                    vec![20.0, 2.0],
                    vec![22.0, 0.0],
                ],
            )
            .unwrap();
            let result = lu.solve_multiple(py, rhs.readonly()).unwrap();
            let result_view = result.readonly();
            let result = result_view.as_array();
            assert_eq!(result[[0, 0]], 1.0);
            assert_eq!(result[[3, 0]], 4.0);
            assert_eq!(result[[0, 1]], 2.0);
            assert_eq!(result[[1, 1]], 1.0);
        });
    }

    #[test]
    fn in_place_and_refinement_routes_report_owned_results() {
        Python::initialize();
        Python::attach(|py| {
            let values = tridiagonal_values(py);
            let matrix = BandedMatrix::new(4, 1, 1, values.readonly()).unwrap();
            let mut lu = LapackStyleBandedLuFaithful::new(4, 1, 1, 1e-14).unwrap();
            lu.factor(&matrix).unwrap();

            let rhs = PyArray1::from_vec(py, vec![6.0, 13.0, 20.0, 22.0]);
            lu.solve_in_place(rhs.readwrite()).unwrap();
            assert_eq!(rhs.readonly().as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);

            let rhs = PyArray1::from_vec(py, vec![6.0, 13.0, 20.0, 22.0]);
            let report = lu
                .solve_with_refinement(&matrix, rhs.readonly(), 2, 1e-12)
                .unwrap();
            assert!(report.converged());
            assert!(report.relative_residual_inf() < 1e-12);
            let report_solution = report.solution_array(py);
            assert_eq!(report_solution.readonly().as_slice().unwrap().len(), 4);
        });
    }

    #[test]
    fn diagonal_and_wide_band_routes_keep_compact_semantics() {
        Python::initialize();
        Python::attach(|py| {
            let compact = PyArray2::from_vec2(py, &[vec![2.0, 3.0, 4.0]]).unwrap();
            let mut diagonal_lu = LapackStyleBandedLuFaithful::new(3, 0, 0, 1e-14).unwrap();
            diagonal_lu.factor_compact(compact.readonly()).unwrap();
            let diagonal_rhs = PyArray1::from_vec(py, vec![2.0, 6.0, 12.0]);
            assert_eq!(
                diagonal_lu
                    .solve(py, diagonal_rhs.readonly())
                    .unwrap()
                    .readonly()
                    .as_slice()
                    .unwrap(),
                &[1.0, 2.0, 3.0]
            );

            let diagonal = PyArray1::from_vec(py, vec![5.0; 5]);
            let lower = PyArray2::from_vec2(
                py,
                &[
                    vec![1.0, 1.0, 1.0, 1.0, 0.0],
                    vec![0.25, 0.25, 0.25, 0.0, 0.0],
                ],
            )
            .unwrap();
            let upper = PyArray2::from_vec2(py, &[vec![0.0, 1.0, 1.0, 1.0, 1.0]]).unwrap();
            let matrix = BandedMatrix::from_bands(
                diagonal.readonly(),
                Some(lower.readonly()),
                Some(upper.readonly()),
            )
            .unwrap();
            let mut wide_lu = LapackStyleBandedLuFaithful::new(5, 2, 1, 1e-14).unwrap();
            wide_lu.factor(&matrix).unwrap();
            let rhs = PyArray1::from_vec(py, vec![7.0, 14.0, 21.25, 28.5, 29.75]);
            let solution = wide_lu.solve(py, rhs.readonly()).unwrap();
            let solution_view = solution.readonly();
            let solution = solution_view.as_slice().unwrap();
            for (actual, expected) in solution.iter().zip([1.0, 2.0, 3.0, 4.0, 5.0]) {
                assert!((actual - expected).abs() < 1e-10);
            }
        });
    }

    #[test]
    fn invalid_compact_shape_and_nonfinite_values_are_rejected() {
        Python::initialize();
        Python::attach(|py| {
            let wrong_shape = PyArray2::from_vec2(py, &[vec![1.0, 2.0]]).unwrap();
            assert!(BandedMatrix::new(3, 1, 1, wrong_shape.readonly()).is_err());

            let nonfinite = PyArray2::from_vec2(
                py,
                &[
                    vec![1.0, f64::NAN, 0.0],
                    vec![2.0, 2.0, 2.0],
                    vec![0.0, 0.0, 0.0],
                ],
            )
            .unwrap();
            assert!(BandedMatrix::new(3, 1, 1, nonfinite.readonly()).is_err());
        });
    }
}
