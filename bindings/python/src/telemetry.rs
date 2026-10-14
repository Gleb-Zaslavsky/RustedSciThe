//! Small, solver-independent helpers for compact Python telemetry reports.

use std::fs;
use std::path::Path;

use pyo3::prelude::*;

use crate::errors::TelemetryError;

/// Write exactly one already-rendered telemetry report without capturing
/// stdout/stderr or compiler progress.
pub fn write_report(path: &str, report: &str) -> PyResult<()> {
    if path.trim().is_empty() {
        return Err(TelemetryError::new_err(
            "telemetry report path must not be empty",
        ));
    }
    if report.trim().is_empty() {
        return Err(TelemetryError::new_err(
            "telemetry report must not be empty",
        ));
    }
    let path_ref = Path::new(path);
    fs::write(path_ref, report).map_err(|error| {
        TelemetryError::new_err(format!(
            "failed to write telemetry report `{}`: {error}",
            path_ref.display()
        ))
    })
}
