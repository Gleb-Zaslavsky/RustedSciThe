//! Detached Python projections of the native CorrectionNewton continuation report.
//! Recovery and checkpoint selection belong exclusively to the Rust solver.

use super::{PyBvpApiSolution, map_api_error};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use rustedscithe_core::numerical::BVP_CorrectionNewton::{
    CorrectionNewtonParameterLoopOutcome, CorrectionNewtonParameterLoopReport, CorrectionNewtonTelemetrySnapshot,
};
use rustedscithe_core::numerical::BVP_api::BvpApiError;
use std::collections::HashMap;

#[pyclass(name = "BvpCorrectionNewtonParameterLoopPoint", module = "rustedscithe", unsendable)]
pub struct BvpCorrectionNewtonParameterLoopPoint {
    #[pyo3(get)]
    index: usize,
    #[pyo3(get)]
    parameters: Vec<f64>,
    /// None means the configured initial guess, not a failed point.
    #[pyo3(get)]
    guess_source_index: Option<usize>,
    #[pyo3(get)]
    status: String,
    #[pyo3(get)]
    telemetry_mode: String,
    #[pyo3(get)]
    iterations: Option<usize>,
    #[pyo3(get)]
    correction_norm: Option<f64>,
    #[pyo3(get)]
    residual_norm: Option<f64>,
    solution: Option<PyBvpApiSolution>,
    error: Option<Py<PyAny>>,
    counters: HashMap<String, usize>,
    timings_ms: HashMap<String, f64>,
}

#[pymethods]
impl BvpCorrectionNewtonParameterLoopPoint {
    /// Only a converged point has a publishable solution.
    #[getter]
    fn solution(&self) -> Option<PyBvpApiSolution> {
        self.solution.clone()
    }

    #[getter]
    fn error<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyAny>> {
        self.error.as_ref().map(|error| error.bind(py).clone())
    }

    /// Empty for Off or when no native attempt report exists.
    fn telemetry_counters(&self) -> HashMap<String, usize> {
        self.counters.clone()
    }

    /// Empty for Off/Counters; a missing measurement is not reported as zero.
    fn telemetry_timings_ms(&self) -> HashMap<String, f64> {
        self.timings_ms.clone()
    }
}

#[pyclass(name = "BvpCorrectionNewtonParameterLoopReport", module = "rustedscithe", unsendable)]
pub struct BvpCorrectionNewtonParameterLoopReport {
    points: Vec<Py<BvpCorrectionNewtonParameterLoopPoint>>,
    stop_error: Option<Py<PyAny>>,
}

#[pymethods]
impl BvpCorrectionNewtonParameterLoopReport {
    #[getter]
    fn points(&self, py: Python<'_>) -> Vec<Py<BvpCorrectionNewtonParameterLoopPoint>> {
        self.points.iter().map(|point| point.clone_ref(py)).collect()
    }

    #[getter]
    fn stop_error<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyAny>> {
        self.stop_error.as_ref().map(|error| error.bind(py).clone())
    }

    fn completed_count(&self) -> usize {
        self.points.len()
    }
}

pub(super) fn telemetry_maps(
    snapshot: Option<&CorrectionNewtonTelemetrySnapshot>,
    mode: &str,
) -> (HashMap<String, usize>, HashMap<String, f64>) {
    let mut counters = HashMap::new();
    let mut timings = HashMap::new();
    if let Some(snapshot) = snapshot {
        if mode != "off" {
            for (name, value) in [
                ("residual_evaluations", snapshot.residual_evaluations),
                ("jacobian_evaluations", snapshot.jacobian_evaluations),
                ("linear_solves", snapshot.linear_solves),
                ("factorizations", snapshot.factorizations),
                ("damping_trials", snapshot.damping_trials),
                ("damping_rejections", snapshot.damping_rejections),
                ("accepted_steps", snapshot.accepted_steps),
                ("bound_limited_steps", snapshot.bound_limited_steps),
            ] {
                counters.insert(name.to_owned(), value);
            }
        }
        if mode == "timings" {
            for (name, value) in [
                ("callback", snapshot.callback_time),
                ("residual_callback", snapshot.residual_callback_time),
                ("jacobian_callback", snapshot.jacobian_callback_time),
                ("jacobian_assembly", snapshot.jacobian_assembly_time),
                ("factorization", snapshot.factorization_time),
                ("linear_solve", snapshot.linear_solve_time),
            ] {
                timings.insert(name.to_owned(), value.as_secs_f64() * 1000.0);
            }
        }
    }
    (counters, timings)
}

pub(super) fn from_native(
    py: Python<'_>,
    report: CorrectionNewtonParameterLoopReport,
    mode: &str,
) -> PyResult<BvpCorrectionNewtonParameterLoopReport> {
    let mut points = Vec::with_capacity(report.points.len());
    for point in report.points {
        let (solution, error, diagnostic) = match point.outcome {
            CorrectionNewtonParameterLoopOutcome::Converged { mesh, values, report } => (
                Some(PyBvpApiSolution {
                    route: "correction_newton".into(),
                    mesh,
                    values: (0..values.nrows())
                        .map(|node| (0..values.ncols()).map(|state| values[(node, state)]).collect())
                        .collect(),
                    sci_details: None,
                }),
                None,
                Some(report),
            ),
            CorrectionNewtonParameterLoopOutcome::Failed { error } => {
                let diagnostic = error.partial_report().cloned();
                let exception = map_api_error(BvpApiError::CorrectionNewton(error));
                exception.value(py).setattr("operation", "parameter_loop")?;
                (None, Some(exception.value(py).clone().unbind().into_any()), diagnostic)
            }
        };
        let (counters, timings_ms) = telemetry_maps(
            diagnostic.as_ref().map(|report| &report.telemetry), mode,
        );
        points.push(Py::new(py, BvpCorrectionNewtonParameterLoopPoint {
            index: point.index,
            parameters: point.parameters,
            guess_source_index: point.guess_source_index,
            status: if solution.is_some() { "converged" } else { "failed" }.into(),
            telemetry_mode: mode.into(),
            iterations: diagnostic.as_ref().map(|report| report.iterations),
            correction_norm: diagnostic.as_ref().and_then(|report| report.correction_norm),
            residual_norm: diagnostic.as_ref().and_then(|report| report.residual_norm),
            solution, error, counters, timings_ms,
        })?);
    }
    let stop_error = report.stop_error.map(|error| {
        map_api_error(BvpApiError::CorrectionNewton(error)).value(py).clone().unbind().into_any()
    });
    Ok(BvpCorrectionNewtonParameterLoopReport { points, stop_error })
}
