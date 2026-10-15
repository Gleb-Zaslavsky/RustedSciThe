//! NumPy-to-Bevy adapters for interactive trajectory playback.
//!
//! Inputs use sample-major `(samples, coordinates)` arrays at the Python
//! boundary. They are copied and transposed once into the core's
//! coordinate-major nalgebra representation before the blocking event loop.

use nalgebra::{DMatrix, DVector};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use rustedscithe_core::Utils::{animation_2d, animation_3d};

/// Bevy's desktop event loop must be created on the process main thread.
/// Releasing the GIL below does not move execution to another OS thread.
fn require_main_thread(py: Python<'_>) -> PyResult<()> {
    let threading = PyModule::import(py, "threading")?;
    let current = threading.call_method0("current_thread")?;
    let main = threading.call_method0("main_thread")?;
    if !current.is(&main) {
        return Err(PyRuntimeError::new_err(
            "trajectory animation must be started from Python's main thread",
        ));
    }
    Ok(())
}

fn trajectory<const D: usize>(
    points: PyReadonlyArray2<'_, f64>,
    times: PyReadonlyArray1<'_, f64>,
) -> PyResult<(DMatrix<f64>, DVector<f64>)> {
    let points = points.as_array();
    let times = times.as_array();
    if points.ncols() != D {
        return Err(PyValueError::new_err(format!(
            "points must have shape (samples, {D}), got ({}, {})",
            points.nrows(),
            points.ncols()
        )));
    }
    if points.nrows() != times.len() {
        return Err(PyValueError::new_err(format!(
            "points has {} samples but times has {} entries",
            points.nrows(),
            times.len()
        )));
    }

    // The animation helpers consume one coordinate per matrix row. Normalize
    // only once here rather than asking Python callers to transpose arrays.
    let matrix = DMatrix::from_fn(D, points.nrows(), |row, sample| points[(sample, row)]);
    let times = DVector::from_iterator(times.len(), times.iter().copied());
    Ok((matrix, times))
}

/// Open a blocking Bevy window and animate a 2D NumPy trajectory.
///
/// `points` must have shape `(samples, 2)` and `times` shape `(samples,)`.
/// Samples are visited in order; strictly increasing and decreasing time
/// vectors are supported. The function releases the Python GIL while the
/// desktop window is open and returns when that window closes.
#[pyfunction]
#[pyo3(signature = (points, times, axis_names=None, speed=1.0))]
pub fn animate_trajectory_2d(
    py: Python<'_>,
    points: PyReadonlyArray2<'_, f64>,
    times: PyReadonlyArray1<'_, f64>,
    axis_names: Option<(String, String)>,
    speed: f32,
) -> PyResult<()> {
    require_main_thread(py)?;
    let (points, times) = trajectory::<2>(points, times)?;
    py.detach(move || animation_2d::try_create_2d_animation(points, times, axis_names, Some(speed)))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Open a blocking Bevy window and animate a 3D NumPy trajectory.
///
/// `points` must have shape `(samples, 3)` and `times` shape `(samples,)`.
/// Samples are visited in order; strictly increasing and decreasing time
/// vectors are supported. The function releases the Python GIL while the
/// desktop window is open and returns when that window closes.
#[pyfunction]
#[pyo3(signature = (points, times, axis_names=None, speed=1.0))]
pub fn animate_trajectory_3d(
    py: Python<'_>,
    points: PyReadonlyArray2<'_, f64>,
    times: PyReadonlyArray1<'_, f64>,
    axis_names: Option<(String, String, String)>,
    speed: f32,
) -> PyResult<()> {
    require_main_thread(py)?;
    let (points, times) = trajectory::<3>(points, times)?;
    py.detach(move || animation_3d::try_create_3d_animation(points, times, axis_names, Some(speed)))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::PyArrayMethods;

    #[test]
    fn trajectory_adapter_transposes_sample_major_input_once() {
        Python::initialize();
        Python::attach(|py| {
            let points =
                numpy::PyArray2::from_vec2(py, &[vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]])
                    .unwrap();
            let times = numpy::PyArray1::from_vec(py, vec![0.0, 0.5, 1.0]);
            let (matrix, time_vector) =
                trajectory::<2>(points.readonly(), times.readonly()).unwrap();
            assert_eq!(matrix.shape(), (2, 3));
            assert_eq!(matrix[(0, 2)], 5.0);
            assert_eq!(matrix[(1, 1)], 4.0);
            assert_eq!(time_vector.as_slice(), &[0.0, 0.5, 1.0]);
        });
    }
}
