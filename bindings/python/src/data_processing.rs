//! Thin Python bindings for stable one-dimensional data-processing helpers.
//!
//! All numerical work remains in `RustedSciThe::numerical::data_processing`.
//! Inputs are copied from NumPy views before crossing into Rust algorithms and
//! outputs are owned NumPy arrays, so strided inputs and solver-buffer lifetime
//! cannot leak into the public contract.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use numpy::ndarray::Array1;
use rustedscithe_core::numerical::data_processing::{
    LSQSplines::{BSpline as NativeBSpline, SolverKind, check_schoenberg_whitney, make_lsq_spline},
    bsplines::BSplineInterpolator as NativeBSplineInterpolator,
    lowess_wrapper::{
        Backend as LowessBackend, LowessConfig, lowess_smooth_values, lowess_smoothing,
    },
    savgol::{SG_filter, SGInput},
    splines::{SplineKind, mae, r_squared, rmse, spline_resample},
};

use crate::errors::{InvalidConfigurationError, SolverError, annotate, catch_panic};

fn owned_series(input: &PyReadonlyArray1<'_, f64>, name: &str) -> PyResult<Vec<f64>> {
    let values: Vec<f64> = input.as_array().iter().copied().collect();
    if values.is_empty() {
        return Err(InvalidConfigurationError::new_err(format!(
            "{name} must contain at least one value"
        )));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(InvalidConfigurationError::new_err(format!(
            "{name} must contain only finite values"
        )));
    }
    Ok(values)
}

fn parse_lsq_solver(solver: &str) -> PyResult<SolverKind> {
    match solver.trim().to_ascii_lowercase().as_str() {
        "dense" | "denseqr" | "dense_qr" => Ok(SolverKind::DenseQR),
        "banded" => Ok(SolverKind::Banded),
        other => Err(InvalidConfigurationError::new_err(format!(
            "unsupported LSQ spline solver '{other}'; expected 'dense' or 'banded'"
        ))),
    }
}

fn make_lsq_spline_native(
    x: &[f64],
    y: &[f64],
    knots: &[f64],
    degree: usize,
    weights: Option<&[f64]>,
    solver: SolverKind,
) -> PyResult<NativeBSpline> {
    if x.len() != y.len() {
        return Err(InvalidConfigurationError::new_err(
            "x and y must have the same length",
        ));
    }
    if degree >= knots.len() - 1 {
        return Err(InvalidConfigurationError::new_err(
            "knots must contain more entries than degree + 1",
        ));
    }
    if knots.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(InvalidConfigurationError::new_err(
            "knots must be nondecreasing",
        ));
    }
    let max_multiplicity = degree + 1;
    let mut run_start = 0;
    while run_start < knots.len() {
        let mut run_end = run_start + 1;
        while run_end < knots.len() && knots[run_end] == knots[run_start] {
            run_end += 1;
        }
        if run_end - run_start > max_multiplicity {
            return Err(InvalidConfigurationError::new_err(
                "a knot value cannot occur more than degree + 1 times",
            ));
        }
        run_start = run_end;
    }
    if let Some(weights) = weights {
        if weights.len() != x.len() {
            return Err(InvalidConfigurationError::new_err(
                "weights must have the same length as x and y",
            ));
        }
        if weights.iter().any(|weight| *weight < 0.0) {
            return Err(InvalidConfigurationError::new_err(
                "weights must be nonnegative",
            ));
        }
        if weights.iter().all(|weight| *weight == 0.0) {
            return Err(InvalidConfigurationError::new_err(
                "at least one weight must be positive",
            ));
        }
    }
    let coefficient_count = knots.len() - degree - 1;
    if x.len() < coefficient_count {
        return Err(InvalidConfigurationError::new_err(
            "the number of observations must be at least the number of spline coefficients",
        ));
    }
    if x.iter()
        .any(|value| *value < knots[degree] || *value > knots[coefficient_count])
    {
        return Err(InvalidConfigurationError::new_err(
            "all x values must lie inside the spline base interval",
        ));
    }

    let x_array = Array1::from_vec(x.to_vec());
    let knot_array = Array1::from_vec(knots.to_vec());
    catch_panic("least_squares_bspline", "data_processing", "native", || {
        check_schoenberg_whitney(&x_array, &knot_array, degree)
            .map_err(|message| InvalidConfigurationError::new_err(message))
    })
    .map_err(|error| annotate(error, "least_squares_bspline", "data_processing", "native"))?;

    let y_array = Array1::from_vec(y.to_vec());
    let weight_array = weights.map(|values| Array1::from_vec(values.to_vec()));
    catch_panic("least_squares_bspline", "data_processing", "native", || {
        make_lsq_spline(
            &x_array,
            &y_array,
            &knot_array,
            degree,
            weight_array.as_ref(),
            solver,
        )
        .map_err(|error| SolverError::new_err(error.to_string()))
    })
    .map_err(|error| annotate(error, "least_squares_bspline", "data_processing", "native"))
}

fn checked_bspline_interpolator(
    degree: usize,
    control_points: Vec<f64>,
    knots: Vec<f64>,
) -> PyResult<NativeBSplineInterpolator<f64>> {
    if control_points.len() <= degree {
        return Err(InvalidConfigurationError::new_err(
            "control_points must contain at least degree + 1 values",
        ));
    }
    if knots.len() != control_points.len() + degree + 1 {
        return Err(InvalidConfigurationError::new_err(
            "knots length must equal len(control_points) + degree + 1",
        ));
    }
    if knots.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(InvalidConfigurationError::new_err(
            "knots must be nondecreasing",
        ));
    }
    let max_multiplicity = degree + 1;
    let mut start = 0;
    while start < knots.len() {
        let mut end = start + 1;
        while end < knots.len() && knots[end] == knots[start] {
            end += 1;
        }
        if end - start > max_multiplicity {
            return Err(InvalidConfigurationError::new_err(
                "a knot value cannot occur more than degree + 1 times",
            ));
        }
        start = end;
    }
    let domain_start = knots[degree];
    let domain_end = knots[control_points.len()];
    if domain_start >= domain_end {
        return Err(InvalidConfigurationError::new_err(
            "B-spline knot domain must have positive width",
        ));
    }
    catch_panic(
        "construct",
        "data_processing",
        "bspline_interpolator",
        || {
            Ok(NativeBSplineInterpolator::new(
                degree,
                control_points,
                knots,
            ))
        },
    )
}

/// Reusable scalar B-spline interpolator backed by the native `bsplines` module.
///
/// The native constructor panics for malformed knot vectors, so the binding
/// validates the shape and knot domain first and keeps a panic boundary as a
/// final ABI safeguard.
#[pyclass(name = "BSplineInterpolator", module = "rustedscithe._native")]
struct PyBSplineInterpolator {
    spline: NativeBSplineInterpolator<f64>,
}

#[pymethods]
impl PyBSplineInterpolator {
    #[new]
    fn new(
        degree: usize,
        control_points: PyReadonlyArray1<'_, f64>,
        knots: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        Ok(Self {
            spline: checked_bspline_interpolator(
                degree,
                owned_series(&control_points, "control_points")?,
                owned_series(&knots, "knots")?,
            )?,
        })
    }

    #[staticmethod]
    fn with_uniform_knots(
        degree: usize,
        control_points: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        let values = owned_series(&control_points, "control_points")?;
        if values.len() <= degree {
            return Err(InvalidConfigurationError::new_err(
                "control_points must contain at least degree + 1 values",
            ));
        }
        let spline = catch_panic(
            "construct",
            "data_processing",
            "bspline_interpolator",
            || {
                Ok(NativeBSplineInterpolator::with_uniform_knots(
                    degree, values,
                ))
            },
        )?;
        Ok(Self { spline })
    }

    #[getter]
    fn degree(&self) -> usize {
        self.spline.degree()
    }

    #[getter]
    fn knot_domain(&self) -> (f64, f64) {
        self.spline.knot_domain()
    }

    #[getter]
    fn knots<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.spline.knots().to_vec())
    }

    #[getter]
    fn control_points<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.spline.control_points().to_vec())
    }

    fn interpolate(&self, parameter: f64) -> PyResult<f64> {
        validate_bspline_parameter(parameter, self.spline.knot_domain())?;
        let value = catch_panic(
            "interpolate",
            "data_processing",
            "bspline_interpolator",
            || Ok(self.spline.interpolate(parameter)),
        )?;
        if !value.is_finite() {
            return Err(SolverError::new_err(
                "B-spline interpolation produced a non-finite value",
            ));
        }
        Ok(value)
    }

    fn interpolate_batch<'py>(
        &self,
        py: Python<'py>,
        parameters: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let parameters = owned_series(&parameters, "parameters")?;
        let domain = self.spline.knot_domain();
        for &parameter in &parameters {
            validate_bspline_parameter(parameter, domain)?;
        }
        let values = catch_panic(
            "interpolate",
            "data_processing",
            "bspline_interpolator",
            || Ok(self.spline.interpolate_at(&parameters)),
        )?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(SolverError::new_err(
                "B-spline interpolation produced a non-finite value",
            ));
        }
        Ok(PyArray1::from_vec(py, values))
    }
}

fn validate_bspline_parameter(parameter: f64, domain: (f64, f64)) -> PyResult<()> {
    if !parameter.is_finite() || parameter < domain.0 || parameter > domain.1 {
        return Err(InvalidConfigurationError::new_err(format!(
            "parameter must be finite and inside the B-spline knot domain [{}, {}]",
            domain.0, domain.1
        )));
    }
    Ok(())
}

/// Complete, detached LOWESS result with input coordinates and smoothed values.
///
/// The result retains the native x ordering and offers owned NumPy copies so
/// callers can align or plot the output without relying on input-buffer lifetime.
#[pyclass(name = "LowessResult", module = "rustedscithe._native")]
struct PyLowessResult {
    x: Vec<f64>,
    y: Vec<f64>,
    iterations_used: Option<usize>,
    fraction_used: f64,
}

#[pymethods]
impl PyLowessResult {
    #[getter]
    fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.x.clone())
    }

    #[getter]
    fn y<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.y.clone())
    }

    #[getter]
    fn iterations_used(&self) -> Option<usize> {
        self.iterations_used
    }

    #[getter]
    fn fraction_used(&self) -> f64 {
        self.fraction_used
    }
}

/// A fitted univariate least-squares B-spline owned by Rust.
///
/// Construct instances with [`least_squares_bspline`]. The selected `dense`
/// (QR) or `banded` route is used directly; no backend conversion or fallback
/// occurs. Evaluation follows the native de Boor implementation, including its
/// endpoint-clamping behavior outside the base interval.
#[pyclass(name = "LeastSquaresBSpline", module = "rustedscithe._native")]
struct PyLeastSquaresBSpline {
    spline: NativeBSpline,
}

#[pymethods]
impl PyLeastSquaresBSpline {
    #[getter]
    fn knots<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.spline.knots.to_vec())
    }

    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.spline.coeffs.to_vec())
    }

    #[getter]
    fn degree(&self) -> usize {
        self.spline.degree
    }

    /// Evaluate the fitted spline at one finite coordinate.
    fn evaluate(&self, x: f64) -> PyResult<f64> {
        if !x.is_finite() {
            return Err(InvalidConfigurationError::new_err("x must be finite"));
        }
        let value = self.spline.evaluate(x);
        if !value.is_finite() {
            return Err(SolverError::new_err(
                "B-spline evaluation produced a non-finite value",
            ));
        }
        Ok(value)
    }

    /// Evaluate the spline at a one-dimensional NumPy array of coordinates.
    fn evaluate_batch<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x = owned_series(&x, "x")?;
        let output = self.spline.evaluate_batch(&x);
        if output.iter().any(|value| !value.is_finite()) {
            return Err(SolverError::new_err(
                "B-spline batch evaluation produced a non-finite value",
            ));
        }
        Ok(PyArray1::from_vec(py, output))
    }
}

/// Fit a univariate least-squares B-spline with an explicit linear solver.
///
/// `knots` is the full nondecreasing knot vector, including repeated boundary
/// knots. `weights` are nonnegative least-squares weights (not their square
/// roots). `solver` selects `dense`/`denseqr` or `banded` and is never silently
/// changed. The returned spline owns its coefficients and knot vector.
#[pyfunction(name = "least_squares_bspline")]
#[pyo3(signature=(x, y, knots, degree, weights=None, solver="banded"))]
fn least_squares_bspline_py(
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    knots: PyReadonlyArray1<'_, f64>,
    degree: usize,
    weights: Option<PyReadonlyArray1<'_, f64>>,
    solver: &str,
) -> PyResult<PyLeastSquaresBSpline> {
    let x = owned_series(&x, "x")?;
    let y = owned_series(&y, "y")?;
    let knots = owned_series(&knots, "knots")?;
    let weights = weights
        .as_ref()
        .map(|values| owned_series(values, "weights"))
        .transpose()?;
    let solver = parse_lsq_solver(solver)?;
    let spline = make_lsq_spline_native(&x, &y, &knots, degree, weights.as_deref(), solver)?;
    if spline.coeffs.iter().any(|value| !value.is_finite()) {
        return Err(SolverError::new_err(
            "least-squares B-spline produced non-finite coefficients",
        ));
    }
    Ok(PyLeastSquaresBSpline { spline })
}

fn parse_spline_kind(kind: &str, tension: f64) -> PyResult<SplineKind> {
    match kind.trim().to_ascii_lowercase().as_str() {
        "linear" => Ok(SplineKind::Linear),
        "cosine" => Ok(SplineKind::Cosine),
        "bezier" if tension.is_finite() => Ok(SplineKind::Bezier(tension)),
        "bezier" => Err(InvalidConfigurationError::new_err(
            "Bezier tension must be finite",
        )),
        other => Err(InvalidConfigurationError::new_err(format!(
            "unsupported spline kind '{other}'; expected linear, cosine or bezier"
        ))),
    }
}

/// Resample paired one-dimensional data onto an evenly spaced spline grid.
///
/// `kind` is `linear`, `cosine`, or `bezier`; `tension` is used only by the
/// Bezier interpolation. Both returned arrays own their storage.
#[pyfunction(name = "spline_resample")]
#[pyo3(signature=(x, y, n_points, kind="linear", tension=0.0))]
fn spline_resample_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    n_points: usize,
    kind: &str,
    tension: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let x = owned_series(&x, "x")?;
    let y = owned_series(&y, "y")?;
    if x.len() != y.len() {
        return Err(InvalidConfigurationError::new_err(
            "x and y must have the same length",
        ));
    }
    if x.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(InvalidConfigurationError::new_err(
            "x must be strictly increasing",
        ));
    }
    let kind = parse_spline_kind(kind, tension)?;
    let (x_new, y_new) = catch_panic("spline_resample", "data_processing", "native", || {
        spline_resample(&x, &y, n_points, kind)
            .map_err(|error| InvalidConfigurationError::new_err(error))
    })
    .map_err(|error| annotate(error, "spline_resample", "data_processing", "native"))?;
    if x_new.iter().chain(&y_new).any(|value| !value.is_finite()) {
        return Err(SolverError::new_err(
            "spline resampling produced a non-finite value",
        ));
    }
    Ok((PyArray1::from_vec(py, x_new), PyArray1::from_vec(py, y_new)))
}

/// Apply RustedSciThe's Savitzky-Golay filter to a one-dimensional signal.
///
/// The native implementation requires an odd window no larger than the
/// signal. Derivative spacing is one sample unit; this API intentionally does
/// not claim SciPy's configurable `delta` semantics.
#[pyfunction(name = "savgol_filter")]
#[pyo3(signature=(data, window_length, polyorder, derivative=0))]
fn savgol_filter_py<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    window_length: usize,
    polyorder: usize,
    derivative: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data = owned_series(&data, "data")?;
    if window_length == 0 || window_length.is_multiple_of(2) {
        return Err(InvalidConfigurationError::new_err(
            "window_length must be a positive odd integer",
        ));
    }
    if window_length > data.len() {
        return Err(InvalidConfigurationError::new_err(
            "window_length must not exceed the data length",
        ));
    }
    let filtered = catch_panic("savgol_filter", "data_processing", "native", || {
        SG_filter(&SGInput {
            data: &data,
            window: window_length,
            poly_order: polyorder,
            derivative,
        })
        .map_err(|error| InvalidConfigurationError::new_err(error))
    })
    .map_err(|error| annotate(error, "savgol_filter", "data_processing", "native"))?;
    if filtered.iter().any(|value| !value.is_finite()) {
        return Err(SolverError::new_err(
            "Savitzky-Golay produced a non-finite value",
        ));
    }
    Ok(PyArray1::from_vec(py, filtered))
}

/// Smooth a one-dimensional signal with the native CPU LOWESS implementation.
///
/// The return value is a newly allocated NumPy vector. The native wrapper's
/// optional GPU request is intentionally not exposed because its backend is
/// unavailable through the stable Rust batch API.
#[pyfunction(name = "lowess_smooth")]
#[pyo3(signature=(x, y, fraction=0.5, iterations=3, delta=0.01, parallel=true))]
fn lowess_smooth_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    fraction: f64,
    iterations: usize,
    delta: f64,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x = owned_series(&x, "x")?;
    let y = owned_series(&y, "y")?;
    if x.len() != y.len() {
        return Err(InvalidConfigurationError::new_err(
            "x and y must have the same length",
        ));
    }
    if x.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(InvalidConfigurationError::new_err(
            "x must be strictly increasing",
        ));
    }
    let config = LowessConfig {
        fraction,
        iterations,
        delta,
        parallel,
        backend: LowessBackend::CPUBack,
    };
    if !fraction.is_finite() || !delta.is_finite() {
        return Err(InvalidConfigurationError::new_err(
            "LOWESS fraction and delta must be finite",
        ));
    }
    config
        .validate()
        .map_err(InvalidConfigurationError::new_err)?;
    let smoothed = catch_panic("lowess", "data_processing", "native", || {
        lowess_smooth_values(&x, &y, &config)
            .map_err(|error| SolverError::new_err(error.to_string()))
    })
    .map_err(|error| annotate(error, "lowess", "data_processing", "native"))?;
    if smoothed.iter().any(|value| !value.is_finite()) {
        return Err(SolverError::new_err("LOWESS produced a non-finite value"));
    }
    Ok(PyArray1::from_vec(py, smoothed))
}

/// Fit LOWESS and return both sorted x coordinates and smoothed y values.
///
/// Use this result-oriented variant when downstream code needs the exact
/// coordinate order reported by the native implementation. `lowess_smooth`
/// remains the convenience API for callers that only need the y vector.
#[pyfunction(name = "lowess_fit")]
#[pyo3(signature=(x, y, fraction=0.5, iterations=3, delta=0.01, parallel=true))]
fn lowess_fit_py(
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    fraction: f64,
    iterations: usize,
    delta: f64,
    parallel: bool,
) -> PyResult<PyLowessResult> {
    let x = owned_series(&x, "x")?;
    let y = owned_series(&y, "y")?;
    if x.len() != y.len() {
        return Err(InvalidConfigurationError::new_err(
            "x and y must have the same length",
        ));
    }
    if x.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(InvalidConfigurationError::new_err(
            "x must be strictly increasing",
        ));
    }
    if !fraction.is_finite() || !delta.is_finite() {
        return Err(InvalidConfigurationError::new_err(
            "LOWESS fraction and delta must be finite",
        ));
    }
    let config = LowessConfig {
        fraction,
        iterations,
        delta,
        parallel,
        backend: LowessBackend::CPUBack,
    };
    config
        .validate()
        .map_err(InvalidConfigurationError::new_err)?;
    let result = catch_panic("lowess_fit", "data_processing", "native", || {
        lowess_smoothing(&x, &y, &config).map_err(|error| SolverError::new_err(error.to_string()))
    })
    .map_err(|error| annotate(error, "lowess_fit", "data_processing", "native"))?;
    if result
        .x
        .iter()
        .chain(&result.y)
        .any(|value| !value.is_finite())
    {
        return Err(SolverError::new_err(
            "LOWESS produced non-finite coordinates or values",
        ));
    }
    Ok(PyLowessResult {
        x: result.x,
        y: result.y,
        iterations_used: result.iterations_used,
        fraction_used: result.fraction_used,
    })
}

/// Return `(RMSE, MAE, R-squared)` for two equal-length finite vectors.
///
/// R-squared is `None` when the reference values have zero variance, rather
/// than exposing the native formula's non-finite division result.
#[pyfunction(name = "fit_quality")]
fn fit_quality_py(
    y_true: PyReadonlyArray1<'_, f64>,
    y_pred: PyReadonlyArray1<'_, f64>,
) -> PyResult<(f64, f64, Option<f64>)> {
    let y_true = owned_series(&y_true, "y_true")?;
    let y_pred = owned_series(&y_pred, "y_pred")?;
    if y_true.len() != y_pred.len() {
        return Err(InvalidConfigurationError::new_err(
            "y_true and y_pred must have the same length",
        ));
    }
    let rmse_value = rmse(&y_true, &y_pred);
    let mae_value = mae(&y_true, &y_pred);
    let sum = y_true.iter().sum::<f64>();
    let mean = sum / y_true.len() as f64;
    let total_variance = y_true
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>();
    if !rmse_value.is_finite()
        || !mae_value.is_finite()
        || !sum.is_finite()
        || !total_variance.is_finite()
    {
        return Err(SolverError::new_err(
            "quality metric computation overflowed to a non-finite value",
        ));
    }
    let score = r_squared(&y_true, &y_pred);
    Ok((rmse_value, mae_value, score.is_finite().then_some(score)))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyLeastSquaresBSpline>()?;
    module.add_class::<PyBSplineInterpolator>()?;
    module.add_class::<PyLowessResult>()?;
    module.add_function(wrap_pyfunction!(least_squares_bspline_py, module)?)?;
    module.add_function(wrap_pyfunction!(spline_resample_py, module)?)?;
    module.add_function(wrap_pyfunction!(savgol_filter_py, module)?)?;
    module.add_function(wrap_pyfunction!(lowess_smooth_py, module)?)?;
    module.add_function(wrap_pyfunction!(lowess_fit_py, module)?)?;
    module.add_function(wrap_pyfunction!(fit_quality_py, module)?)?;
    Ok(())
}
