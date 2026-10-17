//! Thin Python adapter for the first concrete IVP solver vertical slice.
//!
//! The adapter deliberately exposes only the route that is already stable in
//! the native API: ExprLegacy or AtomNative symbolic preparation, in-process
//! Lambdify, dense Newton storage and sequential callback execution. Rejecting
//! other combinations here is safer than silently translating them to a
//! different backend.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use nalgebra::{DMatrix, DVector};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::aot::{generated_config, map_backend_error, provenance_fields, symbolic_backend};
use crate::contract::SolverOptions;
use crate::errors::{
    CallbackFailureState, ExhaustionError, InvalidConfigurationError, SolverError,
    SolverPanicState, UnsupportedOptionError, catch_panic,
};
use crate::expr::PyExpr;
use crate::telemetry::write_report;
use rustedscithe_core::numerical::BDF::BDF_api::{
    BdfSolveError, BdfSolverOptions, BdfTelemetryMode, ODEsolver,
};
use rustedscithe_core::symbolic::ivp_telemetry::IvpLambdifyExecutionPolicy;

/// Adapt Python callbacks to BDF's native callback ABI.
///
/// The numerical core uses infallible `Send + Sync` callbacks, while Python
/// callbacks can raise exceptions or return malformed data. The adapter keeps
/// the first callback failure out-of-band, returns a non-finite vector/matrix
/// to stop the numerical path, and translates the retained cause to the typed
/// Python `CallbackError` at the public solve boundary.
fn build_native_callbacks(
    residual: Py<PyAny>,
    jacobian: Option<Py<PyAny>>,
    dimension: usize,
) -> PyResult<(
    Arc<dyn Fn(f64, &DVector<f64>) -> DVector<f64> + Send + Sync>,
    Option<Arc<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64> + Send + Sync>>,
    CallbackFailureState,
)> {
    let failure = CallbackFailureState::default();
    Python::attach(|py| {
        if !residual.bind(py).is_callable() {
            return Err(InvalidConfigurationError::new_err(
                "residual_callback must be callable",
            ));
        }
        if let Some(jacobian) = &jacobian {
            if !jacobian.bind(py).is_callable() {
                return Err(InvalidConfigurationError::new_err(
                    "jacobian_callback must be callable when provided",
                ));
            }
        }

        let residual = residual.clone_ref(py);
        let residual_failure = failure.clone();
        let residual_callback: Arc<dyn Fn(f64, &DVector<f64>) -> DVector<f64> + Send + Sync> =
            Arc::new(move |time, state| {
                let result = Python::attach(|py| -> PyResult<Vec<f64>> {
                    residual
                        .bind(py)
                        .call1((time, state.as_slice().to_vec()))?
                        .extract()
                });
                match result {
                    Ok(values)
                        if values.len() == dimension
                            && values.iter().all(|value| value.is_finite()) =>
                    {
                        DVector::from_vec(values)
                    }
                    Ok(values) => {
                        residual_failure.record(
                            format!(
                                "residual_callback must return {dimension} finite values, got {}",
                                values.len()
                            ),
                            None,
                        );
                        DVector::from_element(dimension, f64::NAN)
                    }
                    Err(error) => {
                        residual_failure
                            .record(format!("residual_callback raised: {error}"), Some(error));
                        DVector::from_element(dimension, f64::NAN)
                    }
                }
            });

        let jacobian_callback = jacobian.map(|jacobian| {
            let jacobian = jacobian.clone_ref(py);
            let jacobian_failure = failure.clone();
            let callback: Arc<
                dyn Fn(f64, &DVector<f64>) -> DMatrix<f64> + Send + Sync,
            > = Arc::new(move |time, state| {
                let result = Python::attach(|py| -> PyResult<Vec<Vec<f64>>> {
                    jacobian
                        .bind(py)
                        .call1((time, state.as_slice().to_vec()))?
                        .extract()
                });
                match result {
                    Ok(rows)
                        if rows.len() == dimension
                            && rows.iter().all(|row| {
                                row.len() == dimension && row.iter().all(|value| value.is_finite())
                            }) =>
                    {
                        let flat = rows.into_iter().flatten().collect::<Vec<_>>();
                        DMatrix::from_row_slice(dimension, dimension, &flat)
                    }
                    Ok(rows) => {
                        jacobian_failure.record(
                            format!(
                                "jacobian_callback must return a {dimension}x{dimension} finite matrix, got {} rows",
                                rows.len()
                            ),
                            None,
                        );
                        DMatrix::from_element(dimension, dimension, f64::NAN)
                    }
                    Err(error) => {
                        jacobian_failure.record(
                            format!("jacobian_callback raised: {error}"),
                            Some(error),
                        );
                        DMatrix::from_element(dimension, dimension, f64::NAN)
                    }
                }
            });
            callback
        });

        Ok((residual_callback, jacobian_callback, failure))
    })
}

fn finite_vector(values: &[f64], label: &str) -> PyResult<()> {
    if values.is_empty() {
        return Err(InvalidConfigurationError::new_err(format!(
            "{label} must not be empty"
        )));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(InvalidConfigurationError::new_err(format!(
            "{label} must contain only finite values"
        )));
    }
    Ok(())
}

fn validate_names(names: &[String], label: &str) -> PyResult<()> {
    if names.is_empty() {
        return Err(InvalidConfigurationError::new_err(format!(
            "{label} must not be empty"
        )));
    }
    if names.iter().any(|name| name.trim().is_empty()) {
        return Err(InvalidConfigurationError::new_err(format!(
            "{label} must not contain empty names"
        )));
    }
    let mut seen = HashSet::with_capacity(names.len());
    if names.iter().any(|name| !seen.insert(name.as_str())) {
        return Err(InvalidConfigurationError::new_err(format!(
            "{label} must contain unique names"
        )));
    }
    Ok(())
}

fn validate_scalar(value: f64, label: &str) -> PyResult<()> {
    if !value.is_finite() {
        return Err(InvalidConfigurationError::new_err(format!(
            "{label} must be finite"
        )));
    }
    Ok(())
}

fn validate_route(
    options: &SolverOptions,
) -> PyResult<(BdfTelemetryMode, IvpLambdifyExecutionPolicy)> {
    options.validate_aot_options()?;
    crate::contract::validate_backend_options_for(options, "bdf")?;
    if !matches!(options.frontend(), "expr_legacy" | "atom_native") {
        return Err(UnsupportedOptionError::new_err(
            "BDF supports frontend='expr_legacy' or 'atom_native'",
        ));
    }
    if !matches!(options.execution(), "lambdify" | "aot") {
        return Err(UnsupportedOptionError::new_err(
            "BDF execution must be 'lambdify' or 'aot'",
        ));
    }
    if options.matrix_layout() != "dense" {
        return Err(UnsupportedOptionError::new_err(
            "the first BDF Python adapter supports only matrix_layout='dense'",
        ));
    }
    let policy = match options.execution_policy() {
        "sequential" => IvpLambdifyExecutionPolicy::Sequential,
        "parallel" => IvpLambdifyExecutionPolicy::Parallel { min_work: 1 },
        "auto" => IvpLambdifyExecutionPolicy::Auto { min_work: 1 },
        _ => {
            return Err(InvalidConfigurationError::new_err(
                "unknown execution policy in SolverOptions",
            ));
        }
    };
    let telemetry = match options.telemetry() {
        "off" => BdfTelemetryMode::Off,
        "counters" => BdfTelemetryMode::Counters,
        "timings" => BdfTelemetryMode::Timings,
        "detailed" => BdfTelemetryMode::Detailed,
        _ => Err(InvalidConfigurationError::new_err(
            "unknown telemetry mode in SolverOptions",
        ))?,
    };
    Ok((telemetry, policy))
}

fn map_bdf_error(error: BdfSolveError) -> PyErr {
    let native_kind = crate::errors::native_variant(&error);
    crate::errors::annotate_native(
        match error {
            BdfSolveError::Backend(error) => map_backend_error(error),
            BdfSolveError::Configuration(error) => {
                InvalidConfigurationError::new_err(error.to_string())
            }
            error @ (BdfSolveError::InvalidMaxSteps
            | BdfSolveError::InvalidContinuationBound
            | BdfSolveError::ContinuationRequiresPreparedBackend) => {
                InvalidConfigurationError::new_err(error.to_string())
            }
            BdfSolveError::MaxStepsExceeded { .. } => ExhaustionError::new_err(error.to_string()),
            BdfSolveError::Step(error) => SolverError::new_err(error.to_string()),
        },
        "solve",
        "bdf",
        "native",
        &native_kind,
    )
}

/// Detached result of one BDF integration.
///
/// The time/state vectors are copied at the solver boundary. A later
/// continuation or restart therefore cannot mutate a result already returned
/// to Python. `times()` and `states()` return Python lists, while
/// `times_array()` and `states_array()` return owned NumPy arrays. The status,
/// route provenance, counters and stage timings describe the completed solve.
#[pyclass(name = "BdfSolution", module = "rustedscithe")]
#[derive(Clone)]
pub struct BdfSolution {
    times: Vec<f64>,
    states: Vec<Vec<f64>>,
    status: String,
    telemetry_mode: String,
    counters: HashMap<String, u64>,
    timings_ms: HashMap<String, f64>,
    aot_artifact_keys: Vec<String>,
    aot_metadata: HashMap<String, String>,
    aot_counters: HashMap<String, u64>,
    aot_timings_ms: HashMap<String, f64>,
}

impl BdfSolution {
    fn from_solver(solver: &ODEsolver) -> Self {
        let (times, values) = solver.get_result_ref();
        let statistics = solver.get_statistics();
        let preparation = solver.preparation_timings();
        let telemetry_mode = match solver.telemetry_mode() {
            BdfTelemetryMode::Off => "off",
            BdfTelemetryMode::Counters => "counters",
            BdfTelemetryMode::Timings => "timings",
            BdfTelemetryMode::Detailed => "detailed",
        };
        let (aot_metadata, aot_artifact_keys, aot_counters, aot_timings_ms) =
            crate::bvp::aot_telemetry_maps(solver.preparation_backend_telemetry());
        let states = values
            .row_iter()
            .map(|row| row.iter().copied().collect())
            .collect();
        let mut counters = HashMap::from([
            (
                "backend_prepare_calls".to_owned(),
                statistics.backend_prepare_calls as u64,
            ),
            ("solve_calls".to_owned(), statistics.solve_calls as u64),
            ("step_calls".to_owned(), statistics.step_calls as u64),
            (
                "accepted_steps".to_owned(),
                statistics.accepted_steps_total as u64,
            ),
            (
                "candidate_step_attempts".to_owned(),
                statistics.candidate_step_attempts_total as u64,
            ),
            (
                "rejected_step_attempts".to_owned(),
                statistics.rejected_step_attempts_total as u64,
            ),
            (
                "order_changes".to_owned(),
                statistics.bdf_order_changes_total as u64,
            ),
            (
                "linear_solve_attempts".to_owned(),
                statistics.linear_solve_attempts_total as u64,
            ),
            (
                "nonlinear_solves".to_owned(),
                statistics.nonlinear_solve_calls as u64,
            ),
            (
                "nonlinear_iterations".to_owned(),
                statistics.nonlinear_iterations_total as u64,
            ),
            (
                "residual_calls".to_owned(),
                statistics.residual_calls as u64,
            ),
            (
                "jacobian_calls".to_owned(),
                statistics.jacobian_calls as u64,
            ),
            ("bdf_nfev".to_owned(), statistics.bdf_nfev_total as u64),
            ("bdf_njev".to_owned(), statistics.bdf_njev_total as u64),
            ("bdf_nlu".to_owned(), statistics.bdf_nlu_total as u64),
        ]);
        if let Some(telemetry) = solver.preparation_backend_telemetry() {
            for (key, value) in [
                ("aot_resolution_hits", telemetry.aot_resolution_hits),
                ("aot_resolution_misses", telemetry.aot_resolution_misses),
                ("aot_reconnects", telemetry.aot_reconnects),
                ("aot_build_attempts", telemetry.aot_build_attempts),
                ("aot_build_successes", telemetry.aot_build_successes),
                ("aot_link_attempts", telemetry.aot_link_attempts),
                ("aot_link_successes", telemetry.aot_link_successes),
                ("aot_runtime_ready", telemetry.aot_runtime_ready),
            ] {
                counters.insert(key.to_owned(), value);
            }
        }
        let mut timings_ms = HashMap::from([
            (
                "preparation_configuration_ms".to_owned(),
                preparation.configuration_ms,
            ),
            (
                "preparation_symbolic_backend_ms".to_owned(),
                preparation.symbolic_backend_ms,
            ),
            (
                "preparation_callback_wiring_ms".to_owned(),
                preparation.callback_wiring_ms,
            ),
            (
                "preparation_runtime_initialization_ms".to_owned(),
                preparation.runtime_initialization_ms,
            ),
            (
                "preparation_linear_backend_setup_ms".to_owned(),
                preparation.linear_backend_setup_ms,
            ),
            (
                "preparation_runtime_publication_ms".to_owned(),
                preparation.runtime_publication_ms,
            ),
            ("preparation_ms".to_owned(), preparation.total_ms),
            ("solve_ms".to_owned(), statistics.solve_ms_total),
            (
                "integration_loop_ms".to_owned(),
                statistics.integration_loop_ms_total,
            ),
            ("bdf_step_ms".to_owned(), statistics.bdf_step_ms_total),
            (
                "output_collection_ms".to_owned(),
                statistics.output_collection_ms_total,
            ),
            (
                "result_assembly_ms".to_owned(),
                statistics.result_assembly_ms_total,
            ),
            (
                "linear_factorization_ms".to_owned(),
                statistics.linear_factorization_ms_total,
            ),
            (
                "linear_matrix_assembly_ms".to_owned(),
                statistics.linear_matrix_assembly_ms_total,
            ),
            (
                "linear_solve_ms".to_owned(),
                statistics.linear_solve_ms_total,
            ),
            (
                "bdf_step_snapshot_ms".to_owned(),
                statistics.bdf_step_snapshot_ms_total,
            ),
            (
                "bdf_predictor_setup_ms".to_owned(),
                statistics.bdf_step_predictor_setup_ms_total,
            ),
            (
                "bdf_newton_rhs_assembly_ms".to_owned(),
                statistics.bdf_newton_rhs_assembly_ms_total,
            ),
            (
                "bdf_newton_correction_norm_ms".to_owned(),
                statistics.bdf_newton_correction_norm_ms_total,
            ),
            (
                "bdf_newton_state_update_ms".to_owned(),
                statistics.bdf_newton_state_update_ms_total,
            ),
            (
                "bdf_step_error_estimate_ms".to_owned(),
                statistics.bdf_step_error_estimate_ms_total,
            ),
            (
                "bdf_step_nordsieck_update_ms".to_owned(),
                statistics.bdf_step_nordsieck_update_ms_total,
            ),
            ("residual_ms".to_owned(), statistics.residual_ms_total),
            ("jacobian_ms".to_owned(), statistics.jacobian_ms_total),
        ]);
        crate::telemetry::apply_collection_mode(telemetry_mode, &mut counters, &mut timings_ms);
        Self {
            times: times.iter().copied().collect(),
            states,
            status: solver.get_status().to_owned(),
            telemetry_mode: telemetry_mode.to_owned(),
            counters,
            timings_ms,
            aot_artifact_keys,
            aot_metadata,
            aot_counters,
            aot_timings_ms,
        }
    }
}

#[pymethods]
impl BdfSolution {
    /// Accepted output times, including the initial point.
    pub fn times(&self) -> Vec<f64> {
        self.times.clone()
    }

    /// State rows corresponding to [`Self::times`].
    pub fn states(&self) -> Vec<Vec<f64>> {
        self.states.clone()
    }

    /// Owned NumPy array with shape `(accepted_times, state_dimension)`.
    pub fn states_array<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        crate::numpy_support::matrix(py, &self.states)
    }

    /// Owned NumPy array containing the accepted output times.
    pub fn times_array<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f64>> {
        crate::numpy_support::vector(py, &self.times)
    }

    /// Stable native solver status at the end of the operation.
    #[getter]
    pub fn status(&self) -> &str {
        &self.status
    }

    #[getter]
    /// Telemetry collection level used for this solution snapshot.
    pub fn telemetry_mode(&self) -> &str {
        &self.telemetry_mode
    }

    /// Detached operation counters; empty in off or timings-only mode.
    pub fn telemetry_counters(&self) -> HashMap<String, u64> {
        self.counters.clone()
    }

    /// Measured preparation and numerical-stage timings in milliseconds.
    /// Empty when telemetry is off or counters-only; nested scopes retain
    /// their native names and are not additive.
    pub fn telemetry_timings_ms(&self) -> HashMap<String, f64> {
        self.timings_ms.clone()
    }

    /// Artifact identities observed by the native AOT lifecycle.
    pub fn aot_artifact_keys(&self) -> Vec<String> {
        self.aot_artifact_keys.clone()
    }

    /// Return the selected execution route and native snapshot metadata.
    pub fn telemetry_aot_metadata(&self) -> HashMap<String, String> {
        self.aot_metadata.clone()
    }

    /// Return generated-backend lifecycle counters collected in the selected mode.
    pub fn telemetry_aot_counters(&self) -> HashMap<String, u64> {
        self.aot_counters.clone()
    }

    /// Return cold/warm preparation and generated-route stage timings in milliseconds.
    pub fn telemetry_aot_timings_ms(&self) -> HashMap<String, f64> {
        self.aot_timings_ms.clone()
    }

    /// Return the small solver-independent result snapshot used by batch
    /// runners. Detailed counters and timings remain available separately.
    pub fn diagnostics(&self) -> HashMap<String, String> {
        HashMap::from([
            ("solver".to_owned(), "bdf".to_owned()),
            ("status".to_owned(), self.status.clone()),
            ("telemetry".to_owned(), self.telemetry_mode.clone()),
            ("sample_count".to_owned(), self.times.len().to_string()),
            (
                "state_dimension".to_owned(),
                self.states.first().map_or(0, Vec::len).to_string(),
            ),
            (
                "final_time".to_owned(),
                self.times
                    .last()
                    .map_or_else(|| "none".to_owned(), |value| value.to_string()),
            ),
        ])
    }

    /// Render a compact, deterministic report suitable for batch logs.
    pub fn telemetry_report(&self) -> String {
        let mut counters = self.counters.iter().collect::<Vec<_>>();
        counters.sort_by(|left, right| left.0.cmp(right.0));
        let mut timings = self.timings_ms.iter().collect::<Vec<_>>();
        timings.sort_by(|left, right| left.0.cmp(right.0));
        let counters = counters
            .into_iter()
            .map(|(key, value)| format!("{key}={value}"))
            .collect::<Vec<_>>()
            .join(" ");
        let timings = timings
            .into_iter()
            .map(|(key, value)| format!("{key}={value:.6}"))
            .collect::<Vec<_>>()
            .join(" ");
        format!(
            "telemetry={}; counters: {}; timings_ms: {}",
            self.telemetry_mode, counters, timings
        )
    }

    /// Write only the compact report, without capturing process output.
    pub fn write_telemetry_report(&self, path: &str) -> PyResult<()> {
        write_report(path, &self.telemetry_report())
    }

    fn __repr__(&self) -> String {
        format!(
            "BdfSolution(points={}, state_dimension={}, status='{}')",
            self.times.len(),
            self.states.first().map_or(0, Vec::len),
            self.status
        )
    }
}

/// Prepared symbolic BDF solver for the first supported Python route.
///
/// Construction validates the public contract but defers symbolic preparation
/// until [`Self::prepare`] or [`Self::solve`]. This mirrors the native BDF
/// lifecycle and makes cold preparation distinguishable from warm solving.
/// Use [`SolverOptions`] to select frontend, Lambdify/AOT execution, matrix
/// layout, callback execution policy, and telemetry. Declared parameter values
/// can be rebound with `continue_with_parameters`; `restart` starts a new
/// interval and initial state while retaining prepared symbolic callbacks.
#[pyclass(name = "BdfSolver", module = "rustedscithe", unsendable)]
pub struct BdfSolver {
    solver: ODEsolver,
    parameter_count: usize,
    first_step: Option<f64>,
    max_bdf_order: usize,
    max_steps: usize,
    telemetry_mode: BdfTelemetryMode,
    frontend: String,
    execution: String,
    execution_policy: String,
    aot_metadata: HashMap<String, String>,
    panic_state: SolverPanicState,
}

#[pymethods]
impl BdfSolver {
    /// Configure a symbolic BDF initial-value problem.
    ///
    /// `expressions` contains one right-hand side per state variable. Optional
    /// parameter symbols must have matching `parameter_values`. `y0`, `t0`,
    /// and `t_end` define the initial condition and interval; `max_step`,
    /// `rtol`, `atol`, `first_step`, `max_bdf_order`, and `max_steps` control
    /// integration. `options` selects the symbolic and linear routes.
    #[new]
    #[pyo3(signature=(expressions, state_variables, time_variable, y0, t0, t_end, max_step, rtol=1e-6, atol=1e-8, options=None, parameters=None, parameter_values=None, first_step=None, max_bdf_order=5, max_steps=1_000_000))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        expressions: Vec<PyExpr>,
        state_variables: Vec<String>,
        time_variable: String,
        y0: Vec<f64>,
        t0: f64,
        t_end: f64,
        max_step: f64,
        rtol: f64,
        atol: f64,
        options: Option<PyRef<'_, SolverOptions>>,
        parameters: Option<Vec<String>>,
        parameter_values: Option<Vec<f64>>,
        first_step: Option<f64>,
        max_bdf_order: usize,
        max_steps: usize,
    ) -> PyResult<Self> {
        validate_names(&state_variables, "state_variables")?;
        if expressions.len() != state_variables.len() {
            return Err(InvalidConfigurationError::new_err(format!(
                "expected one RHS expression per state variable, got {} expressions for {} variables",
                expressions.len(),
                state_variables.len()
            )));
        }
        if time_variable.trim().is_empty() {
            return Err(InvalidConfigurationError::new_err(
                "time_variable must not be empty",
            ));
        }
        if state_variables.iter().any(|name| name == &time_variable) {
            return Err(InvalidConfigurationError::new_err(
                "time_variable must differ from state_variables",
            ));
        }
        let parameters = parameters.unwrap_or_default();
        if !parameters.is_empty() {
            validate_names(&parameters, "parameters")?;
        }
        if parameters
            .iter()
            .any(|name| name == &time_variable || state_variables.iter().any(|state| state == name))
        {
            return Err(InvalidConfigurationError::new_err(
                "parameters must not collide with time_variable or state_variables",
            ));
        }
        if let Some(values) = &parameter_values {
            if values.len() != parameters.len() {
                return Err(InvalidConfigurationError::new_err(format!(
                    "expected {} parameter values, got {}",
                    parameters.len(),
                    values.len()
                )));
            }
            if values.iter().any(|value| !value.is_finite()) {
                return Err(InvalidConfigurationError::new_err(
                    "parameter_values must contain only finite values",
                ));
            }
        } else if !parameters.is_empty() {
            return Err(InvalidConfigurationError::new_err(
                "parameter_values are required when parameters are declared",
            ));
        }
        let allowed_symbols: HashSet<&str> = state_variables
            .iter()
            .map(String::as_str)
            .chain(std::iter::once(time_variable.as_str()))
            .chain(parameters.iter().map(String::as_str))
            .collect();
        for (index, expression) in expressions.iter().enumerate() {
            if let Some(unknown) = expression
                .inner
                .extract_variables()
                .1
                .iter()
                .find(|name| !allowed_symbols.contains(name.as_str()))
            {
                return Err(InvalidConfigurationError::new_err(format!(
                    "RHS expression {index} references undeclared symbol '{unknown}'"
                )));
            }
        }
        finite_vector(&y0, "y0")?;
        validate_scalar(t0, "t0")?;
        validate_scalar(t_end, "t_end")?;
        if t0 == t_end {
            return Err(InvalidConfigurationError::new_err(
                "t0 and t_end must differ",
            ));
        }
        validate_scalar(max_step, "max_step")?;
        if max_step <= 0.0 {
            return Err(InvalidConfigurationError::new_err(
                "max_step must be greater than zero",
            ));
        }
        validate_scalar(rtol, "rtol")?;
        validate_scalar(atol, "atol")?;
        if rtol <= 0.0 || atol <= 0.0 {
            return Err(InvalidConfigurationError::new_err(
                "rtol and atol must be greater than zero",
            ));
        }
        if let Some(first_step) = first_step {
            validate_scalar(first_step, "first_step")?;
            if first_step <= 0.0 {
                return Err(InvalidConfigurationError::new_err(
                    "first_step must be greater than zero",
                ));
            }
        }
        if !(1..=5).contains(&max_bdf_order) {
            return Err(InvalidConfigurationError::new_err(
                "max_bdf_order must be between 1 and 5",
            ));
        }
        if max_steps == 0 {
            return Err(InvalidConfigurationError::new_err(
                "max_steps must be greater than zero",
            ));
        }

        let options = match options {
            Some(options) => options.clone(),
            None => SolverOptions::new(None, None, None, None, None, None, None, None, None)?,
        };
        let (telemetry_mode, execution_policy) = validate_route(&options)?;
        let logging_level = crate::contract::native_logging_level(&options, "bdf")?;
        let native_expressions = expressions
            .into_iter()
            .map(|expression| expression.inner)
            .collect();
        let parameter_count = parameters.len();
        let mut native_options = BdfSolverOptions::for_bdf(
            native_expressions,
            state_variables,
            time_variable,
            t0,
            DVector::from_vec(y0),
            t_end,
            max_step,
            rtol,
            atol,
            None,
            false,
            first_step,
        )
        .with_max_bdf_order(max_bdf_order)
        .with_max_steps(max_steps)
        .with_symbolic_assembly_backend(symbolic_backend(&options))
        .with_lambdify_execution_policy(execution_policy)
        .with_telemetry_mode(telemetry_mode)
        .with_logging_level(logging_level);
        if options.execution() == "aot" {
            native_options =
                native_options.with_generated_backend_config(generated_config(&options)?);
        }
        if !parameters.is_empty() {
            native_options = native_options.with_equation_parameters(parameters);
            native_options = native_options.with_equation_parameter_values(DVector::from_vec(
                parameter_values.ok_or_else(|| {
                    InvalidConfigurationError::new_err("parameter values were not validated")
                })?,
            ));
        }

        let route = format!(
            "{}:{}:{}",
            options.frontend(),
            options.execution(),
            options.execution_policy()
        );
        let solver = catch_panic("construct", "bdf", &route, || {
            Ok(ODEsolver::new_with_options(native_options))
        })?;
        Ok(Self {
            solver,
            parameter_count,
            first_step,
            max_bdf_order,
            max_steps,
            telemetry_mode,
            frontend: options.frontend().to_owned(),
            execution: options.execution().to_owned(),
            execution_policy: options.execution_policy().to_owned(),
            aot_metadata: provenance_fields(&options),
            panic_state: SolverPanicState::default(),
        })
    }

    /// Prepare symbolic callbacks and backend state without integrating.
    ///
    /// Preparation is idempotent. Call this explicitly to separate setup
    /// cost from `solve()`, or let `solve()` prepare lazily.
    pub fn prepare(&mut self) -> PyResult<()> {
        let route = format!(
            "{}:{}:{}",
            self.frontend, self.execution, self.execution_policy
        );
        let panic_state = self.panic_state.clone();
        panic_state.run("prepare", "bdf", &route, || {
            self.solver.try_generate().map_err(map_backend_error)
        })
    }

    /// Whether the symbolic callbacks and numerical runtime are prepared.
    pub fn is_prepared(&self) -> bool {
        self.solver.is_backend_prepared()
    }

    /// Integrate the configured initial state and return a detached result.
    ///
    /// The result contains the accepted trajectory and telemetry snapshot.
    /// Callback or numerical failures are raised as typed Python exceptions.
    pub fn solve(&mut self) -> PyResult<BdfSolution> {
        let route = format!(
            "{}:{}:{}",
            self.frontend, self.execution, self.execution_policy
        );
        let panic_state = self.panic_state.clone();
        panic_state.run("solve", "bdf", &route, || self.solve_inner())
    }

    /// Restart a prepared model with a new initial state and interval.
    ///
    /// The native symbolic callbacks remain prepared; only the numerical
    /// history and runtime factorization are rebuilt for the new segment.
    pub fn restart(&mut self, t0: f64, y0: Vec<f64>, t_end: f64) -> PyResult<BdfSolution> {
        finite_vector(&y0, "y0")?;
        validate_scalar(t0, "t0")?;
        validate_scalar(t_end, "t_end")?;
        let route = format!(
            "{}:{}:{}",
            self.frontend, self.execution, self.execution_policy
        );
        let panic_state = self.panic_state.clone();
        panic_state.run("restart", "bdf", &route, || {
            self.solver
                .try_restart_with_initial_state(t0, DVector::from_vec(y0), t_end)
                .map_err(map_bdf_error)?;
            self.solve_inner()
        })
    }

    /// Continue from the latest accepted state with new parameter values.
    ///
    /// The native BDF runtime rebuilds numerical history and factorization but
    /// retains the prepared symbolic callbacks and their parameter slots.
    #[pyo3(signature=(parameter_values, t_end))]
    pub fn continue_with_parameters(
        &mut self,
        parameter_values: Vec<f64>,
        t_end: f64,
    ) -> PyResult<BdfSolution> {
        if parameter_values.len() != self.parameter_count {
            return Err(InvalidConfigurationError::new_err(format!(
                "expected {} parameter values, got {}",
                self.parameter_count,
                parameter_values.len()
            )));
        }
        if parameter_values.iter().any(|value| !value.is_finite()) {
            return Err(InvalidConfigurationError::new_err(
                "parameter_values must contain only finite values",
            ));
        }
        validate_scalar(t_end, "t_end")?;
        let route = format!(
            "{}:{}:{}",
            self.frontend, self.execution, self.execution_policy
        );
        let panic_state = self.panic_state.clone();
        panic_state.run("continuation", "bdf", &route, || {
            self.solver
                .try_continue_with_parameter_values(DVector::from_vec(parameter_values), t_end)
                .map_err(map_bdf_error)?;
            self.solve_inner()
        })
    }

    /// Current native status without exposing mutable solver internals.
    pub fn status(&self) -> String {
        self.solver.get_status().to_owned()
    }

    /// Latest native BDF telemetry in its compact diagnostic representation.
    /// Use `telemetry_counters()` and `telemetry_timings_ms()` for structured
    /// data; this string remains convenient for logs and interactive reports.
    pub fn telemetry_report(&self) -> String {
        self.solver.statistics_report()
    }

    /// Stable route metadata needed to reproduce this Python-side solve.
    pub fn provenance(&self) -> HashMap<String, String> {
        let mut metadata = HashMap::from([
            ("solver".to_owned(), "bdf".to_owned()),
            ("frontend".to_owned(), self.frontend.clone()),
            ("execution".to_owned(), self.execution.clone()),
            ("matrix_layout".to_owned(), "dense".to_owned()),
            ("execution_policy".to_owned(), self.execution_policy.clone()),
            ("telemetry".to_owned(), self.telemetry_mode().to_owned()),
            (
                "first_step".to_owned(),
                self.first_step
                    .map_or_else(|| "auto".to_owned(), |value| value.to_string()),
            ),
            ("max_bdf_order".to_owned(), self.max_bdf_order.to_string()),
            ("max_steps".to_owned(), self.max_steps.to_string()),
        ]);
        metadata.extend(self.aot_metadata.clone());
        metadata
    }

    /// Selected telemetry mode, useful when reports are generated later.
    #[getter]
    /// Telemetry collection level configured on this solver.
    pub fn telemetry_mode(&self) -> &'static str {
        match self.telemetry_mode {
            BdfTelemetryMode::Off => "off",
            BdfTelemetryMode::Counters => "counters",
            BdfTelemetryMode::Timings => "timings",
            BdfTelemetryMode::Detailed => "detailed",
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "BdfSolver(prepared={}, telemetry='{}')",
            self.is_prepared(),
            self.telemetry_mode()
        )
    }
}

impl BdfSolver {
    fn solve_inner(&mut self) -> PyResult<BdfSolution> {
        self.solver.try_solve().map_err(map_bdf_error)?;
        Ok(BdfSolution::from_solver(&self.solver))
    }
}

/// Dense native-callback BDF route.
///
/// This is intentionally separate from [`BdfSolver`]. Symbolic BDF owns an
/// expression-preparation lifecycle, while this class accepts an already
/// prepared Python residual and optionally an analytic Jacobian. Omitting the
/// Jacobian delegates finite-difference construction to the native BDF core.
/// The route is sequential and Dense-only: Python callbacks must not be
/// silently converted into a structured or parallel execution path.
#[pyclass(name = "BdfCallbackSolver", module = "rustedscithe", unsendable)]
pub struct BdfCallbackSolver {
    solver: ODEsolver,
    failure: CallbackFailureState,
    dimension: usize,
    telemetry_mode: String,
    jacobian_source: String,
    first_step: Option<f64>,
    max_bdf_order: usize,
    max_steps: usize,
    panic_state: SolverPanicState,
}

#[pymethods]
impl BdfCallbackSolver {
    /// Configure a Dense BDF solve around Python numerical callbacks.
    ///
    /// `residual_callback(t, y)` must return a finite vector with
    /// `state_dimension` entries. An optional `jacobian_callback(t, y)` must
    /// return a finite square matrix in row-major nested-list form. When it is
    /// omitted, the native BDF implementation constructs a finite-difference
    /// Jacobian. Python callback routes are synchronous, sequential, Dense and
    /// Lambdify-only; symbolic frontend or AOT settings are not applicable.
    /// `y0`, interval and tolerance controls have the same meaning as in
    /// [`BdfSolver`].
    #[new]
    #[pyo3(signature=(state_dimension, t0, t_end, max_step, y0, residual_callback, jacobian_callback=None, rtol=1e-6, atol=1e-8, options=None, first_step=None, max_bdf_order=5, max_steps=1_000_000))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        state_dimension: usize,
        t0: f64,
        t_end: f64,
        max_step: f64,
        y0: Vec<f64>,
        residual_callback: Py<PyAny>,
        jacobian_callback: Option<Py<PyAny>>,
        rtol: f64,
        atol: f64,
        options: Option<PyRef<'_, SolverOptions>>,
        first_step: Option<f64>,
        max_bdf_order: usize,
        max_steps: usize,
    ) -> PyResult<Self> {
        if state_dimension == 0 {
            return Err(InvalidConfigurationError::new_err(
                "state_dimension must be greater than zero",
            ));
        }
        if y0.len() != state_dimension {
            return Err(InvalidConfigurationError::new_err(format!(
                "expected y0 with {state_dimension} values, got {}",
                y0.len()
            )));
        }
        finite_vector(&y0, "y0")?;
        validate_scalar(t0, "t0")?;
        validate_scalar(t_end, "t_end")?;
        if t0 == t_end {
            return Err(InvalidConfigurationError::new_err(
                "t0 and t_end must differ",
            ));
        }
        validate_scalar(max_step, "max_step")?;
        if max_step <= 0.0 {
            return Err(InvalidConfigurationError::new_err(
                "max_step must be greater than zero",
            ));
        }
        validate_scalar(rtol, "rtol")?;
        validate_scalar(atol, "atol")?;
        if rtol <= 0.0 || atol <= 0.0 {
            return Err(InvalidConfigurationError::new_err(
                "rtol and atol must be greater than zero",
            ));
        }
        if let Some(first_step) = first_step {
            validate_scalar(first_step, "first_step")?;
            if first_step <= 0.0 || first_step > (t_end - t0).abs() {
                return Err(InvalidConfigurationError::new_err(
                    "first_step must be positive and no larger than the integration interval",
                ));
            }
        }
        if !(1..=5).contains(&max_bdf_order) {
            return Err(InvalidConfigurationError::new_err(
                "max_bdf_order must be between 1 and 5",
            ));
        }
        if max_steps == 0 {
            return Err(InvalidConfigurationError::new_err(
                "max_steps must be greater than zero",
            ));
        }

        let options = match options {
            Some(options) => options.clone(),
            None => SolverOptions::new(None, None, None, None, None, None, None, None, None)?,
        };
        let (telemetry_mode, execution_policy) = validate_route(&options)?;
        if options.execution() != "lambdify" {
            return Err(UnsupportedOptionError::new_err(
                "BdfCallbackSolver requires execution='lambdify'; Python callbacks cannot be embedded into AOT artifacts",
            ));
        }
        if options.frontend() != "expr_legacy" {
            return Err(UnsupportedOptionError::new_err(
                "BdfCallbackSolver does not use a symbolic frontend; use frontend='expr_legacy' or omit frontend",
            ));
        }
        if options.matrix_layout() != "dense" {
            return Err(UnsupportedOptionError::new_err(
                "BdfCallbackSolver currently supports only matrix_layout='dense'",
            ));
        }
        if options.execution_policy() != "sequential" {
            return Err(UnsupportedOptionError::new_err(
                "BdfCallbackSolver currently supports only execution_policy='sequential'",
            ));
        }

        let (residual, jacobian, failure) =
            build_native_callbacks(residual_callback, jacobian_callback, state_dimension)?;
        let state_variables = (0..state_dimension)
            .map(|index| format!("y{index}"))
            .collect::<Vec<_>>();
        let native_options = BdfSolverOptions::for_bdf(
            Vec::new(),
            state_variables,
            "t".to_owned(),
            t0,
            DVector::from_vec(y0),
            t_end,
            max_step,
            rtol,
            atol,
            None,
            false,
            first_step,
        )
        .with_max_bdf_order(max_bdf_order)
        .with_max_steps(max_steps)
        .with_lambdify_execution_policy(execution_policy)
        .with_telemetry_mode(telemetry_mode)
        .with_logging_level(crate::contract::native_logging_level(&options, "bdf")?);
        let mut solver = ODEsolver::new_with_options(native_options);
        let jacobian_source = if let Some(jacobian) = jacobian {
            let residual_callback = move |time: f64, state: &DVector<f64>| residual(time, state);
            let jacobian_callback = move |time: f64, state: &DVector<f64>| jacobian(time, state);
            solver.set_native_ode_callbacks(residual_callback, Some(jacobian_callback));
            "analytic"
        } else {
            let residual_callback = move |time: f64, state: &DVector<f64>| residual(time, state);
            let no_jacobian: Option<fn(f64, &DVector<f64>) -> DMatrix<f64>> = None;
            solver.set_native_ode_callbacks(residual_callback, no_jacobian);
            "finite_difference"
        };

        Ok(Self {
            solver,
            failure,
            dimension: state_dimension,
            telemetry_mode: options.telemetry().to_owned(),
            jacobian_source: jacobian_source.to_owned(),
            first_step,
            max_bdf_order,
            max_steps,
            panic_state: SolverPanicState::default(),
        })
    }

    /// Prepare the native BDF runtime without taking an integration step.
    pub fn prepare(&mut self) -> PyResult<()> {
        let panic_state = self.panic_state.clone();
        panic_state.run("prepare", "bdf", "callback", || {
            self.solver.try_generate().map_err(map_backend_error)
        })
    }

    /// Whether the native numerical backend has been initialized.
    pub fn is_prepared(&self) -> bool {
        self.solver.is_backend_prepared()
    }

    /// Integrate the configured interval and return a detached trajectory.
    pub fn solve(&mut self) -> PyResult<BdfSolution> {
        let panic_state = self.panic_state.clone();
        panic_state.run("solve", "bdf", "callback", || {
            let result = self.solver.try_solve();
            if let Some(error) = self.failure.take_pyerr("solve", "bdf", "callback") {
                return Err(error);
            }
            result.map_err(map_bdf_error)?;
            Ok(BdfSolution::from_solver(&self.solver))
        })
    }

    /// Restart from a new state and interval, retaining the Python callbacks.
    pub fn restart(&mut self, t0: f64, y0: Vec<f64>, t_end: f64) -> PyResult<BdfSolution> {
        if y0.len() != self.dimension {
            return Err(InvalidConfigurationError::new_err(format!(
                "expected y0 with {} values, got {}",
                self.dimension,
                y0.len()
            )));
        }
        finite_vector(&y0, "y0")?;
        validate_scalar(t0, "t0")?;
        validate_scalar(t_end, "t_end")?;
        if t0 == t_end {
            return Err(InvalidConfigurationError::new_err(
                "t0 and t_end must differ",
            ));
        }
        let panic_state = self.panic_state.clone();
        panic_state.run("restart", "bdf", "callback", || {
            self.solver
                .try_restart_with_initial_state(t0, DVector::from_vec(y0), t_end)
                .map_err(map_bdf_error)?;
            self.solve()
        })
    }

    /// Return discrete native BDF counters for the latest lifecycle operation.
    pub fn telemetry_counters(&self) -> HashMap<String, u64> {
        BdfSolution::from_solver(&self.solver).telemetry_counters()
    }

    /// Return native timing totals in milliseconds for the latest operation.
    pub fn telemetry_timings_ms(&self) -> HashMap<String, f64> {
        BdfSolution::from_solver(&self.solver).telemetry_timings_ms()
    }

    /// Render a compact human-readable telemetry snapshot.
    pub fn telemetry_report(&self) -> String {
        self.solver.statistics_report()
    }

    /// Describe the fixed callback route and selected Jacobian source.
    pub fn provenance(&self) -> HashMap<String, String> {
        HashMap::from([
            ("solver".to_owned(), "bdf".to_owned()),
            ("frontend".to_owned(), "python_callback".to_owned()),
            ("execution".to_owned(), "native_callback".to_owned()),
            ("matrix_layout".to_owned(), "dense".to_owned()),
            ("execution_policy".to_owned(), "sequential".to_owned()),
            ("telemetry".to_owned(), self.telemetry_mode.clone()),
            ("jacobian_source".to_owned(), self.jacobian_source.clone()),
            (
                "first_step".to_owned(),
                self.first_step
                    .map_or_else(|| "auto".to_owned(), |value| value.to_string()),
            ),
            ("max_bdf_order".to_owned(), self.max_bdf_order.to_string()),
            ("max_steps".to_owned(), self.max_steps.to_string()),
        ])
    }

    #[getter]
    /// Telemetry collection level configured for callback evaluation.
    pub fn telemetry_mode(&self) -> &str {
        &self.telemetry_mode
    }

    fn __repr__(&self) -> String {
        format!(
            "BdfCallbackSolver(prepared={}, state_dimension={}, jacobian='{}')",
            self.is_prepared(),
            self.dimension,
            self.jacobian_source
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bdf_route_accepts_dense_aot() {
        let options = SolverOptions::new(
            Some("expr_legacy"),
            Some("aot"),
            Some("dense"),
            Some("sequential"),
            Some("off"),
            None,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(validate_route(&options).is_ok());
    }
}
