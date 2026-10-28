//! Python result container for independent IVP parameter sweeps.

use pyo3::prelude::*;

/// A single parameter-loop attempt. Success retains the solver's ordinary
/// detached result object; failure retains the original typed Python
/// exception, including its stage/solver/route/native_kind metadata.
#[pyclass(name = "IvpParameterLoopPoint", module = "rustedscithe", unsendable)]
pub struct PyIvpParameterLoopPoint {
    index: usize,
    parameter_values: Vec<f64>,
    solution: Option<Py<PyAny>>,
    error: Option<Py<PyAny>>,
}

#[pymethods]
impl PyIvpParameterLoopPoint {
    #[getter]
    fn index(&self) -> usize {
        self.index
    }

    #[getter]
    fn parameter_values(&self) -> Vec<f64> {
        self.parameter_values.clone()
    }

    #[getter]
    fn succeeded(&self) -> bool {
        self.solution.is_some()
    }

    /// The normal detached solver result, including trajectory and telemetry.
    #[getter]
    fn solution(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.solution.as_ref().map(|solution| solution.clone_ref(py))
    }

    /// The original typed Python exception for a failed point, if any.
    #[getter]
    fn error(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.error.as_ref().map(|error| error.clone_ref(py))
    }

    fn __repr__(&self) -> String {
        let outcome = if self.solution.is_some() { "success" } else { "failed" };
        format!(
            "IvpParameterLoopPoint(index={}, outcome='{}', parameter_values={:?})",
            self.index, outcome, self.parameter_values
        )
    }
}

pub(crate) fn run_parameter_loop(
    py: Python<'_>,
    solver: &str,
    route: &str,
    parameter_vectors: Vec<Vec<f64>>,
    mut solve_point: impl FnMut(&[f64]) -> PyResult<Py<PyAny>>,
) -> PyResult<Vec<Py<PyIvpParameterLoopPoint>>> {
    let mut points = Vec::with_capacity(parameter_vectors.len());
    for (index, parameter_values) in parameter_vectors.into_iter().enumerate() {
        let (solution, error) = match solve_point(&parameter_values) {
            Ok(solution) => (Some(solution), None),
            Err(error) => {
                let error = crate::errors::annotate(error, "solve", solver, route);
                let exception = error.value(py).clone().unbind().into_any();
                (None, Some(exception))
            }
        };
        points.push(Py::new(
            py,
            PyIvpParameterLoopPoint {
                index,
                parameter_values,
                solution,
                error,
            },
        )?);
    }
    Ok(points)
}
