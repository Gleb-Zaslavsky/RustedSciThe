//! Small, solver-independent helpers for compact Python telemetry reports.

use std::fs;
use std::path::Path;

use pyo3::prelude::*;

use crate::errors::TelemetryError;

/// Apply the shared Python snapshot contract without changing native timing
/// scopes. `timings` exposes durations only; `detailed` preserves both maps.
pub(crate) fn apply_collection_mode(
    mode: &str,
    counters: &mut std::collections::HashMap<String, u64>,
    timings_ms: &mut std::collections::HashMap<String, f64>,
) {
    match mode {
        "off" => {
            counters.clear();
            timings_ms.clear();
        }
        "counters" => timings_ms.clear(),
        "timings" => counters.clear(),
        "detailed" => {}
        _ => unreachable!("solver adapters must normalize telemetry mode names"),
    }
}

#[cfg(test)]
mod tests {
    use super::apply_collection_mode;
    use std::collections::HashMap;

    #[test]
    fn collection_modes_expose_only_the_requested_channels() {
        for (mode, expect_counters, expect_timings) in [
            ("off", false, false),
            ("counters", true, false),
            ("timings", false, true),
            ("detailed", true, true),
        ] {
            let mut counters = HashMap::from([("measured_zero".to_owned(), 0)]);
            let mut timings = HashMap::from([("stage_ms".to_owned(), 0.0)]);
            apply_collection_mode(mode, &mut counters, &mut timings);
            assert_eq!(!counters.is_empty(), expect_counters, "mode={mode}");
            assert_eq!(!timings.is_empty(), expect_timings, "mode={mode}");
            if expect_counters {
                assert_eq!(counters.get("measured_zero"), Some(&0));
            }
            if expect_timings {
                assert_eq!(timings.get("stage_ms"), Some(&0.0));
            }
        }
    }
}

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
