use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use std::panic::{self, AssertUnwindSafe};

/// Attach machine-readable context without changing the exception class or
/// human-readable message. Native error enums are richer than a flat `PyErr`;
/// these attributes preserve the useful classification at the ABI boundary.
pub(crate) fn annotate(
    error: PyErr,
    stage: &'static str,
    solver: &'static str,
    route: &'static str,
) -> PyErr {
    Python::attach(|py| {
        let value = error.value(py);
        let _ = value.setattr("stage", stage);
        let _ = value.setattr("solver", solver);
        let _ = value.setattr("route", route);
        let category = match value
            .get_type()
            .name()
            .ok()
            .map(|name| name.to_string())
            .as_deref()
        {
            Some("InvalidExpressionError") => "expression",
            Some("UnsupportedOptionError") => "unsupported_option",
            Some("InvalidConfigurationError") => "configuration",
            Some("PreparationError") => "preparation",
            Some("AotArtifactError") => "aot_artifact",
            Some("AotCompilerError") => "aot_compiler",
            Some("AotLinkError") => "aot_link",
            Some("AotTimeoutError") => "aot_timeout",
            Some("AotLifecycleError") => "aot_lifecycle",
            Some("CallbackError") => "callback",
            Some("ExhaustionError") => "exhaustion",
            Some("TelemetryError") => "telemetry",
            Some("TaskDocumentError") => "task_document",
            Some("BvpError") => "bvp",
            Some("PanicError") => "panic",
            Some("SolverError") => "solver",
            _ => "rustedscithe",
        };
        let _ = value.setattr("category", category);
    });
    error
}

/// Convert a panic from native numerical code into a typed Python exception.
///
/// PyO3 protects the generated method trampoline too, but its fallback is the
/// generic `PanicException`. This boundary keeps the crate's error taxonomy
/// and attaches the same stage/solver/route context as ordinary failures.
pub(crate) fn catch_panic<T, F>(
    stage: &'static str,
    solver: &'static str,
    route: &'static str,
    operation: F,
) -> PyResult<T>
where
    F: FnOnce() -> PyResult<T>,
{
    match panic::catch_unwind(AssertUnwindSafe(operation)) {
        Ok(result) => result,
        Err(payload) => {
            let message = if let Some(message) = payload.downcast_ref::<String>() {
                message.clone()
            } else if let Some(message) = payload.downcast_ref::<&str>() {
                (*message).to_owned()
            } else {
                "panic from native Rust code".to_owned()
            };
            Err(annotate(
                PanicError::new_err(format!(
                    "native panic during {stage} for {solver}: {message}"
                )),
                stage,
                solver,
                route,
            ))
        }
    }
}

// Root for errors raised by the RustedSciThe Python extension.
create_exception!(
    rustedscithe,
    RustedSciTheError,
    PyException,
    "Base exception for RustedSciThe errors."
);

// Raised when a symbolic expression cannot be parsed or evaluated safely.
create_exception!(
    rustedscithe,
    InvalidExpressionError,
    RustedSciTheError,
    "The symbolic expression is invalid."
);

// Raised when a solver option is unknown or unsupported by a solver.
create_exception!(
    rustedscithe,
    UnsupportedOptionError,
    RustedSciTheError,
    "The requested option is not supported."
);

// Raised when a solver or symbolic model configuration is inconsistent.
create_exception!(
    rustedscithe,
    InvalidConfigurationError,
    RustedSciTheError,
    "The configuration is invalid."
);

// Raised when a configured solver cannot prepare or complete an integration.
// Configuration errors remain separate so callers can distinguish an invalid
// request from a numerical failure without matching message text.
create_exception!(
    rustedscithe,
    SolverError,
    RustedSciTheError,
    "The numerical solver could not complete the requested operation."
);

// Raised when symbolic/backend preparation fails after configuration has been
// accepted. Keeping this separate lets callers distinguish setup failures from
// a failed Newton/integration step without parsing diagnostic text.
create_exception!(
    rustedscithe,
    PreparationError,
    RustedSciTheError,
    "The solver could not prepare its symbolic or numerical backend."
);

// Raised when an AOT artifact is missing, stale, partial, or has not been
// materialized for a strict RequirePrebuilt request.
create_exception!(
    rustedscithe,
    AotArtifactError,
    PreparationError,
    "The requested AOT artifact is unavailable."
);

// Raised when an external AOT compiler cannot produce the requested artifact.
create_exception!(
    rustedscithe,
    AotCompilerError,
    PreparationError,
    "The AOT compiler failed."
);

// Raised when an AOT runtime library cannot be linked or loaded.
create_exception!(
    rustedscithe,
    AotLinkError,
    PreparationError,
    "The AOT runtime library could not be linked or loaded."
);

// Raised when an AOT compiler or linker exceeds its configured deadline.
create_exception!(
    rustedscithe,
    AotTimeoutError,
    PreparationError,
    "The AOT toolchain exceeded its configured timeout."
);

// Raised for an AOT lifecycle class that has no more specific Python type.
create_exception!(
    rustedscithe,
    AotLifecycleError,
    PreparationError,
    "The AOT artifact lifecycle failed."
);

// Raised when a Python-owned residual/Jacobian callback raises or returns a
// value with the wrong shape or non-finite entries.
create_exception!(
    rustedscithe,
    CallbackError,
    RustedSciTheError,
    "A Python solver callback failed or returned invalid data."
);

// Raised when a task document cannot be loaded or validated by the shared
// Rust task parser.  Keeping this separate from solver errors lets Python
// callers distinguish an input-document problem before any solver exists.
create_exception!(
    rustedscithe,
    TaskDocumentError,
    RustedSciTheError,
    "The task document is invalid or could not be loaded."
);

// Raised when a parsed BVP task reaches a native solver but cannot produce a
// valid detached result.  Parser/configuration failures remain TaskDocumentError;
// this class is reserved for the execution boundary.
create_exception!(
    rustedscithe,
    BvpError,
    RustedSciTheError,
    "The BVP task could not be solved."
);

// Raised when a compact telemetry report cannot be written without mixing in
// compiler or process noise.
create_exception!(
    rustedscithe,
    TelemetryError,
    RustedSciTheError,
    "The telemetry report could not be written."
);

// Raised when a solver reaches an explicit work budget or continuation
// exhaustion condition.
create_exception!(
    rustedscithe,
    ExhaustionError,
    RustedSciTheError,
    "The solver exhausted a configured numerical budget."
);

// A typed replacement for an otherwise unclassified panic in native code.
create_exception!(
    rustedscithe,
    PanicError,
    RustedSciTheError,
    "Native Rust code panicked during a guarded operation."
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panic_boundary_preserves_typed_context() {
        Python::initialize();
        Python::attach(|py| {
            let error = catch_panic::<(), _>("solve", "test_solver", "test_route", || {
                panic!("guard fixture")
            })
            .expect_err("the fixture must panic");
            assert!(error.is_instance_of::<PanicError>(py));
            let value = error.value(py);
            assert_eq!(
                value.getattr("stage").unwrap().extract::<String>().unwrap(),
                "solve"
            );
            assert_eq!(
                value
                    .getattr("solver")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "test_solver"
            );
            assert_eq!(
                value.getattr("route").unwrap().extract::<String>().unwrap(),
                "test_route"
            );
            assert!(error.to_string().contains("guard fixture"));
        });
    }
}
