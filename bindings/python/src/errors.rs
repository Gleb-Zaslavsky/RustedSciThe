use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use std::panic::{self, AssertUnwindSafe};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

/// Attach machine-readable context without changing the exception class or
/// human-readable message. Native error enums are richer than a flat `PyErr`;
/// these attributes preserve the useful classification at the ABI boundary.
pub(crate) fn annotate(error: PyErr, stage: &str, solver: &str, route: &str) -> PyErr {
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
            Some("UnsupportedSymbolicOperationError") => "unsupported_symbolic_operation",
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
            Some("PoisonedSolverError") => "poisoned_solver",
            Some("SolverError") => "solver",
            _ => "rustedscithe",
        };
        let _ = value.setattr("category", category);
    });
    error
}

/// Attach the Rust enum variant that produced a mapped Python exception.
///
/// Python callers should branch on the typed exception class first. This
/// supplementary field is for diagnostics and telemetry when several native
/// variants intentionally share one Python exception class.
pub(crate) fn annotate_native(
    error: PyErr,
    stage: &str,
    solver: &str,
    route: &str,
    native_kind: &str,
) -> PyErr {
    let error = annotate(error, stage, solver, route);
    Python::attach(|py| {
        let _ = error.value(py).setattr("native_kind", native_kind);
    });
    error
}

/// Read the enum variant identifier from Rust's derived `Debug` form.
///
/// All current native error enums derive `Debug`; payloads are deliberately
/// excluded so this value stays useful as a compact discriminator.
pub(crate) fn native_variant(error: &impl std::fmt::Debug) -> String {
    format!("{error:?}")
        .split(['{', '(', ' '])
        .next()
        .unwrap_or("Unknown")
        .to_owned()
}

/// Guard a public Python/native operation and attach its authoritative context.
///
/// PyO3 protects the generated method trampoline too, but its fallback is the
/// generic `PanicException`. This boundary converts Rust panics into the
/// crate's typed `PanicError` and also overwrites stale mapper context on
/// ordinary errors with the exact public stage, solver and route.
pub(crate) fn catch_panic<T, F>(stage: &str, solver: &str, route: &str, operation: F) -> PyResult<T>
where
    F: FnOnce() -> PyResult<T>,
{
    match panic::catch_unwind(AssertUnwindSafe(operation)) {
        // Native error mappers classify the exception, but the public wrapper
        // knows which API operation actually failed. Re-annotate here so a
        // mapper reused by `prepare`, `restart`, or `continuation` cannot leave
        // stale `stage="solve"` metadata on an otherwise correctly typed error.
        Ok(Ok(value)) => Ok(value),
        Ok(Err(error)) => Err(annotate(error, stage, solver, route)),
        Err(payload) => {
            let message = if let Some(message) = payload.downcast_ref::<String>() {
                message.clone()
            } else if let Some(message) = payload.downcast_ref::<&str>() {
                (*message).to_owned()
            } else {
                "panic from native Rust code".to_owned()
            };
            let error = annotate(
                PanicError::new_err(format!(
                    "native panic during {stage} for {solver}: {message}"
                )),
                stage,
                solver,
                route,
            );
            Python::attach(|py| {
                let value = error.value(py);
                let state_attribute = if solver.ends_with("_task") {
                    "execution_state"
                } else {
                    "solver_state"
                };
                let _ = value.setattr(state_attribute, "discard");
            });
            Err(error)
        }
    }
}

/// Tracks whether a mutable native solver must no longer be used after panic.
///
/// This is shared independently from the solver borrow so a guarded closure
/// can mutate the native solver while the state guard remains available to
/// poison the wrapper on unwind.
#[derive(Clone, Default)]
pub(crate) struct SolverPanicState(Arc<AtomicBool>);

impl SolverPanicState {
    pub(crate) fn run<T, F>(
        &self,
        stage: &str,
        solver: &str,
        route: &str,
        operation: F,
    ) -> PyResult<T>
    where
        F: FnOnce() -> PyResult<T>,
    {
        if self.0.load(Ordering::Acquire) {
            return Err(annotate(
                PoisonedSolverError::new_err(format!(
                    "{solver} solver is unusable after a native panic; construct a new solver"
                )),
                stage,
                solver,
                route,
            ));
        }

        let result = catch_panic(stage, solver, route, operation);
        if result
            .as_ref()
            .is_err_and(|error| Python::attach(|py| error.is_instance_of::<PanicError>(py)))
        {
            self.0.store(true, Ordering::Release);
        }
        result
    }
}

/// Retains the first failure from a Python callback crossing a Rust callback
/// ABI that cannot return `PyResult`. The original exception is restored as
/// `__cause__` when the numerical operation returns to Python.
#[derive(Clone, Default)]
pub(crate) struct CallbackFailureState(Arc<Mutex<Option<CallbackFailureRecord>>>);

struct CallbackFailureRecord {
    message: String,
    cause: Option<PyErr>,
}

impl CallbackFailureState {
    pub(crate) fn record(&self, message: String, cause: Option<PyErr>) {
        if let Ok(mut slot) = self.0.lock() {
            if slot.is_none() {
                *slot = Some(CallbackFailureRecord { message, cause });
            }
        }
    }

    pub(crate) fn clear(&self) {
        if let Ok(mut slot) = self.0.lock() {
            slot.take();
        }
    }

    pub(crate) fn take_pyerr(&self, stage: &str, solver: &str, route: &str) -> Option<PyErr> {
        let record = self.0.lock().ok().and_then(|mut slot| slot.take())?;
        let error = annotate(CallbackError::new_err(record.message), stage, solver, route);
        if let Some(cause) = record.cause {
            Python::attach(|py| error.set_cause(py, Some(cause)));
        }
        Some(error)
    }

    pub(crate) fn attach_cause(&self, error: PyErr) -> PyErr {
        let cause = self
            .0
            .lock()
            .ok()
            .and_then(|mut slot| slot.take())
            .and_then(|record| record.cause);
        if let Some(cause) = cause {
            Python::attach(|py| error.set_cause(py, Some(cause)));
        }
        error
    }
}

/// Debug integration probe for the Python panic/discard contract.
///
/// This class is exported only with the non-default `test-panic-probe` Cargo
/// feature. It deliberately has no facade export and must not be enabled in a
/// published wheel.
#[cfg(feature = "test-panic-probe")]
#[pyclass(name = "_PanicProbeSolver")]
pub(crate) struct PanicProbeSolver {
    state: SolverPanicState,
}

#[cfg(feature = "test-panic-probe")]
#[pymethods]
impl PanicProbeSolver {
    #[new]
    fn new() -> Self {
        Self {
            state: SolverPanicState::default(),
        }
    }

    fn trigger_native_panic(&self) -> PyResult<()> {
        self.state.run("solve", "panic_probe", "test_only", || {
            panic!("intentional test-only native panic")
        })
    }

    fn retry(&self) -> PyResult<()> {
        self.state
            .run("solve", "panic_probe", "test_only", || Ok(()))
    }

    fn fail_numerically(&self) -> PyResult<()> {
        self.state.run("solve", "panic_probe", "test_only", || {
            Err(SolverError::new_err(
                "intentional retryable numerical failure",
            ))
        })
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

// Raised when a requested symbolic transformation is valid as an operation
// but is outside the native engine's supported subset.
create_exception!(
    rustedscithe,
    UnsupportedSymbolicOperationError,
    RustedSciTheError,
    "The symbolic operation is not supported for this expression."
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

// Raised when a solver object is called after a caught native panic. The
// partially-mutated numerical state is intentionally not reused.
create_exception!(
    rustedscithe,
    PoisonedSolverError,
    SolverError,
    "The solver instance is unusable after a native panic; construct a new solver."
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
            assert_eq!(
                value
                    .getattr("solver_state")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "discard"
            );
            assert!(error.to_string().contains("guard fixture"));
        });
    }

    #[test]
    fn solver_panic_state_blocks_reuse_after_unwind() {
        Python::initialize();
        Python::attach(|py| {
            let state = SolverPanicState::default();
            let panic = state
                .run::<(), _>("solve", "test_solver", "test_route", || {
                    panic!("poison fixture")
                })
                .expect_err("the first operation must return the caught panic");
            assert!(panic.is_instance_of::<PanicError>(py));

            let reused = state
                .run("restart", "test_solver", "test_route", || Ok(()))
                .expect_err("a poisoned solver must reject future operations");
            assert!(reused.is_instance_of::<PoisonedSolverError>(py));
            let value = reused.value(py);
            assert_eq!(
                value.getattr("stage").unwrap().extract::<String>().unwrap(),
                "restart"
            );
            assert_eq!(
                value
                    .getattr("solver_state")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "discard"
            );
        });
    }

    #[test]
    fn operation_boundary_overrides_stale_stage_without_losing_native_kind() {
        Python::initialize();
        Python::attach(|py| {
            let mapped = annotate_native(
                SolverError::new_err("backend setup failed"),
                "solve",
                "test_solver",
                "native",
                "BackendSetup",
            );
            let error =
                catch_panic::<(), _>("prepare", "test_solver", "atom_native", || Err(mapped))
                    .expect_err("the mapped failure must remain an error");

            assert!(error.is_instance_of::<SolverError>(py));
            let value = error.value(py);
            assert_eq!(
                value.getattr("stage").unwrap().extract::<String>().unwrap(),
                "prepare"
            );
            assert_eq!(
                value.getattr("route").unwrap().extract::<String>().unwrap(),
                "atom_native"
            );
            assert_eq!(
                value
                    .getattr("native_kind")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "BackendSetup"
            );
        });
    }
}
