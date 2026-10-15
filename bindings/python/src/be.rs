//! Python adapter for the first Backward Euler route.
//!
//! Backward Euler has a slightly different native lifecycle from BDF and
//! Radau: symbolic callback preparation is performed lazily by the first
//! integration call. The adapter therefore exposes `validate()` rather than
//! pretending that a separate Python `prepare()` has already materialized the
//! native callback set.

use std::collections::{HashMap, HashSet};

use nalgebra::DVector;
use pyo3::prelude::*;

use rustedscithe_core::numerical::BE::{
    BE, BeError, BeSolverOptions, BeSymbolicAssemblyBackend, BeTelemetryMode,
};
use rustedscithe_core::symbolic::symbolic_ivp::IvpBackendError;

use crate::aot::{generated_config, map_backend_error, provenance_fields};
use crate::contract::SolverOptions;
use crate::errors::{
    ExhaustionError, InvalidConfigurationError, SolverError, SolverPanicState,
    UnsupportedOptionError, catch_panic,
};
use crate::expr::PyExpr;
use crate::telemetry::write_report;

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
) -> PyResult<(BeSymbolicAssemblyBackend, BeTelemetryMode)> {
    options.validate_aot_options()?;
    let frontend = match options.frontend() {
        "expr_legacy" => BeSymbolicAssemblyBackend::ExprLegacy,
        "atom_native" => BeSymbolicAssemblyBackend::AtomViewNative,
        _ => {
            return Err(UnsupportedOptionError::new_err(
                "Backward Euler supports frontend='expr_legacy' or 'atom_native'",
            ));
        }
    };
    if !matches!(options.execution(), "lambdify" | "aot") {
        return Err(UnsupportedOptionError::new_err(
            "Backward Euler execution must be 'lambdify' or 'aot'",
        ));
    }
    if options.matrix_layout() != "dense" {
        return Err(UnsupportedOptionError::new_err(
            "Backward Euler supports only matrix_layout='dense'",
        ));
    }
    if options.execution_policy() != "sequential" {
        return Err(UnsupportedOptionError::new_err(
            "the first Backward Euler Python adapter supports only execution_policy='sequential'",
        ));
    }
    let telemetry = match options.telemetry() {
        "off" => BeTelemetryMode::Off,
        "counters" => BeTelemetryMode::Counters,
        "timings" => BeTelemetryMode::Timings,
        _ => {
            return Err(InvalidConfigurationError::new_err(
                "unknown telemetry mode in SolverOptions",
            ));
        }
    };
    Ok((frontend, telemetry))
}

fn validate_expression_symbols(
    expressions: &[PyExpr],
    state_variables: &[String],
    time_variable: &str,
    parameters: &[String],
) -> PyResult<()> {
    let allowed: HashSet<&str> = state_variables
        .iter()
        .map(String::as_str)
        .chain(std::iter::once(time_variable))
        .chain(parameters.iter().map(String::as_str))
        .collect();
    for (index, expression) in expressions.iter().enumerate() {
        if let Some(unknown) = expression
            .inner
            .extract_variables()
            .1
            .iter()
            .find(|name| !allowed.contains(name.as_str()))
        {
            return Err(InvalidConfigurationError::new_err(format!(
                "RHS expression {index} references undeclared symbol '{unknown}'"
            )));
        }
    }
    Ok(())
}

/// Detached Backward Euler trajectory and the telemetry snapshot collected
/// while producing it.
///
/// Trajectory data is copied out of the mutable solver. Use the list accessors
/// for ordinary Python code or the `*_array()` accessors for owned NumPy
/// arrays. Telemetry methods expose counters and stage timings for this result.
#[pyclass(name = "BackwardEulerSolution", module = "rustedscithe")]
#[derive(Clone)]
pub struct BackwardEulerSolution {
    times: Vec<f64>,
    states: Vec<Vec<f64>>,
    status: String,
    telemetry_mode: String,
    counters: std::collections::HashMap<String, u64>,
    timings_ms: std::collections::HashMap<String, f64>,
    aot_artifact_keys: Vec<String>,
}

impl BackwardEulerSolution {
    fn from_solver(solver: &BE) -> Self {
        let (times, states) = solver.trajectory();
        let detailed = solver.detailed_statistics();
        let common = solver.get_statistics();
        let aot_artifact_keys = solver
            .symbolic_ivp_telemetry_snapshot()
            .map(|telemetry| telemetry.aot_artifact_keys.clone())
            .unwrap_or_default();
        let mut counters = std::collections::HashMap::from([
            ("accepted_steps".to_owned(), detailed.accepted_steps as u64),
            ("failed_steps".to_owned(), detailed.failed_steps as u64),
            (
                "factorization_calls".to_owned(),
                detailed.factorization_calls as u64,
            ),
            (
                "linear_solve_calls".to_owned(),
                detailed.linear_solve_calls as u64,
            ),
            ("residual_calls".to_owned(), common.residual_calls as u64),
            ("jacobian_calls".to_owned(), common.jacobian_calls as u64),
            (
                "parameter_bind_attempts".to_owned(),
                detailed.parameter_bind_attempts as u64,
            ),
        ]);
        if let Some(telemetry) = solver.symbolic_ivp_telemetry_snapshot() {
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
        let timings_ms = std::collections::HashMap::from([
            (
                "factorization_ms".to_owned(),
                detailed.factorization_ms_total,
            ),
            ("linear_solve_ms".to_owned(), detailed.linear_solve_ms_total),
            (
                "output_assembly_ms".to_owned(),
                detailed.output_assembly_ms_total,
            ),
            (
                "parameter_bind_ms".to_owned(),
                detailed.parameter_bind_ms_total,
            ),
            ("residual_ms".to_owned(), common.residual_ms_total),
            ("jacobian_ms".to_owned(), common.jacobian_ms_total),
        ]);
        Self {
            times: times.iter().copied().collect(),
            states: states
                .row_iter()
                .map(|row| row.iter().copied().collect())
                .collect(),
            status: solver.get_status(),
            telemetry_mode: match solver.telemetry_mode() {
                BeTelemetryMode::Off => "off",
                BeTelemetryMode::Counters => "counters",
                BeTelemetryMode::Timings => "timings",
            }
            .to_owned(),
            counters,
            timings_ms,
            aot_artifact_keys,
        }
    }
}

#[pymethods]
impl BackwardEulerSolution {
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

    #[getter]
    /// Native integration status for the returned trajectory.
    pub fn status(&self) -> &str {
        &self.status
    }

    #[getter]
    /// Telemetry collection level used for this solve.
    pub fn telemetry_mode(&self) -> &str {
        &self.telemetry_mode
    }

    /// Return discrete solver counters; the map is empty when collection is off.
    pub fn telemetry_counters(&self) -> std::collections::HashMap<String, u64> {
        self.counters.clone()
    }

    /// Return measured stage durations in milliseconds, if timing telemetry was enabled.
    pub fn telemetry_timings_ms(&self) -> std::collections::HashMap<String, f64> {
        self.timings_ms.clone()
    }

    /// Artifact identities observed by the native AOT lifecycle.
    pub fn aot_artifact_keys(&self) -> Vec<String> {
        self.aot_artifact_keys.clone()
    }

    /// Return the common solver-independent result snapshot. Detailed BE
    /// counters and stage timings remain available through telemetry methods.
    pub fn diagnostics(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::from([
            ("solver".to_owned(), "backward_euler".to_owned()),
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
            "BackwardEulerSolution(points={}, state_dimension={}, status='{}')",
            self.times.len(),
            self.states.first().map_or(0, Vec::len),
            self.status
        )
    }
}

/// Symbolic Backward Euler solver.
///
/// The current native Backward Euler implementation supports the Dense matrix
/// layout. `SolverOptions` selects the symbolic frontend and Lambdify/AOT
/// execution route; unsupported combinations are rejected. Construction
/// validates the problem, while symbolic callback generation remains lazy
/// until `solve()` or `continue_to()`.
#[pyclass(name = "BackwardEulerSolver", module = "rustedscithe", unsendable)]
pub struct BackwardEulerSolver {
    solver: BE,
    parameter_count: usize,
    telemetry_mode: BeTelemetryMode,
    frontend: String,
    execution: String,
    aot_metadata: HashMap<String, String>,
    panic_state: SolverPanicState,
}

#[pymethods]
impl BackwardEulerSolver {
    /// Configure a symbolic Backward Euler initial-value problem.
    ///
    /// Supply one right-hand-side expression per state variable. Optional
    /// `parameters` declare value-only symbols and `parameter_values` bind
    /// their initial values. `h` selects a fixed step when provided;
    /// `tolerance` and `max_iterations` control the implicit step solve.
    /// `options` selects the frontend, execution route and telemetry.
    #[new]
    #[pyo3(signature=(expressions, state_variables, time_variable, y0, t0, t_end, h=None, tolerance=1e-6, max_iterations=100, options=None, parameters=None, parameter_values=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        expressions: Vec<PyExpr>,
        state_variables: Vec<String>,
        time_variable: String,
        y0: Vec<f64>,
        t0: f64,
        t_end: f64,
        h: Option<f64>,
        tolerance: f64,
        max_iterations: usize,
        options: Option<PyRef<'_, SolverOptions>>,
        parameters: Option<Vec<String>>,
        parameter_values: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        validate_names(&state_variables, "state_variables")?;
        finite_vector(&y0, "y0")?;
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
        validate_expression_symbols(&expressions, &state_variables, &time_variable, &parameters)?;
        validate_scalar(t0, "t0")?;
        validate_scalar(t_end, "t_end")?;
        if t0 == t_end {
            return Err(InvalidConfigurationError::new_err(
                "t0 and t_end must differ",
            ));
        }
        if let Some(h) = h {
            validate_scalar(h, "h")?;
            if h <= 0.0 {
                return Err(InvalidConfigurationError::new_err(
                    "h must be greater than zero when provided",
                ));
            }
        }
        validate_scalar(tolerance, "tolerance")?;
        if tolerance <= 0.0 {
            return Err(InvalidConfigurationError::new_err(
                "tolerance must be greater than zero",
            ));
        }
        if max_iterations == 0 {
            return Err(InvalidConfigurationError::new_err(
                "max_iterations must be greater than zero",
            ));
        }
        if let Some(values) = &parameter_values {
            finite_vector(values, "parameter_values")?;
            if values.len() != parameters.len() {
                return Err(InvalidConfigurationError::new_err(format!(
                    "expected {} parameter values, got {}",
                    parameters.len(),
                    values.len()
                )));
            }
        }

        let options = match options {
            Some(options) => options.clone(),
            None => SolverOptions::new(None, None, None, None, None, None, None, None, None)?,
        };
        let (frontend, telemetry_mode) = validate_route(&options)?;
        let mut native_options = BeSolverOptions::new(
            expressions
                .into_iter()
                .map(|expression| expression.inner)
                .collect(),
            state_variables,
            time_variable,
            tolerance,
            max_iterations,
            h,
            t0,
            t_end,
            DVector::from_vec(y0),
        )
        .with_symbolic_assembly_backend(frontend)
        .with_telemetry_mode(telemetry_mode);
        if options.execution() == "aot" {
            native_options =
                native_options.with_generated_backend_config(generated_config(&options)?);
        }
        let route = format!("{}:{}", options.frontend(), options.execution());
        let solver = catch_panic("construct", "backward_euler", &route, || {
            let mut solver = BE::try_new_with_options(native_options).map_err(map_be_error)?;
            if !parameters.is_empty() {
                let parameter_refs = parameters.iter().map(String::as_str).collect::<Vec<_>>();
                solver
                    .try_set_equation_parameters(Some(&parameter_refs))
                    .map_err(map_be_error)?;
            }
            if let Some(values) = parameter_values {
                solver
                    .try_set_parameter_values(DVector::from_vec(values))
                    .map_err(map_be_error)?;
            }
            Ok(solver)
        })?;
        Ok(Self {
            solver,
            parameter_count: parameters.len(),
            telemetry_mode,
            frontend: options.frontend().to_owned(),
            execution: options.execution().to_owned(),
            aot_metadata: provenance_fields(&options),
            panic_state: SolverPanicState::default(),
        })
    }

    /// Validate the native configuration without starting integration.
    ///
    /// Symbolic callback generation remains lazy and is performed by the
    /// first `solve()`/`continue_to()` call in the current BE core.
    pub fn validate(&self) -> PyResult<()> {
        let route = format!("{}:{}", self.frontend, self.execution);
        let panic_state = self.panic_state.clone();
        panic_state.run("validate", "backward_euler", &route, || {
            self.solver.try_check().map_err(map_be_error)
        })
    }

    /// Integrate the configured interval and return a detached trajectory.
    ///
    /// Symbolic callbacks are prepared lazily on the first solve. Callback or
    /// numerical failures are reported as typed Python exceptions.
    pub fn solve(&mut self) -> PyResult<BackwardEulerSolution> {
        let route = format!("{}:{}", self.frontend, self.execution);
        let panic_state = self.panic_state.clone();
        panic_state.run("solve", "backward_euler", &route, || {
            self.solver.try_solve().map_err(map_be_error)?;
            Ok(BackwardEulerSolution::from_solver(&self.solver))
        })
    }

    /// Continue from the latest accepted state, optionally rebinding values.
    ///
    /// This is a warm segment continuation, not a new initial-value problem.
    /// Use `restart()` when both the initial state and interval should change.
    #[pyo3(signature=(t_end, parameter_values=None))]
    pub fn continue_to(
        &mut self,
        t_end: f64,
        parameter_values: Option<Vec<f64>>,
    ) -> PyResult<BackwardEulerSolution> {
        validate_scalar(t_end, "t_end")?;
        if let Some(values) = parameter_values.as_deref() {
            finite_vector(values, "parameter_values")?;
            if values.len() != self.parameter_count {
                return Err(InvalidConfigurationError::new_err(format!(
                    "expected {} parameter values, got {}",
                    self.parameter_count,
                    values.len()
                )));
            }
        }
        let route = format!("{}:{}", self.frontend, self.execution);
        let panic_state = self.panic_state.clone();
        panic_state.run("continuation", "backward_euler", &route, || {
            if let Some(values) = parameter_values {
                self.solver
                    .try_set_parameter_values(DVector::from_vec(values))
                    .map_err(map_be_error)?;
            }
            self.solver.try_continue_to(t_end).map_err(map_be_error)?;
            Ok(BackwardEulerSolution::from_solver(&self.solver))
        })
    }

    /// Restart the numerical state while retaining prepared symbolic data.
    ///
    /// The new interval endpoints must differ and `y0` must match the state
    /// dimension. The returned result describes the restarted segment.
    pub fn restart(
        &mut self,
        t0: f64,
        y0: Vec<f64>,
        t_end: f64,
    ) -> PyResult<BackwardEulerSolution> {
        finite_vector(&y0, "y0")?;
        validate_scalar(t0, "t0")?;
        validate_scalar(t_end, "t_end")?;
        let route = format!("{}:{}", self.frontend, self.execution);
        let panic_state = self.panic_state.clone();
        panic_state.run("restart", "backward_euler", &route, || {
            self.solver
                .try_restart_with_initial_state(t0, DVector::from_vec(y0), t_end)
                .map_err(map_be_error)?;
            self.solver.try_solve().map_err(map_be_error)?;
            Ok(BackwardEulerSolution::from_solver(&self.solver))
        })
    }

    /// Return the native integration status for the current solver state.
    pub fn status(&self) -> String {
        self.solver.get_status()
    }

    /// Return the native compact statistics report for the latest integration segment.
    pub fn telemetry_report(&self) -> String {
        self.solver.statistics_report()
    }

    /// Stable route metadata needed to reproduce this Python-side solve.
    pub fn provenance(&self) -> HashMap<String, String> {
        let mut metadata = HashMap::from([
            ("solver".to_owned(), "backward_euler".to_owned()),
            ("frontend".to_owned(), self.frontend.clone()),
            ("execution".to_owned(), self.execution.clone()),
            ("matrix_layout".to_owned(), "dense".to_owned()),
            ("execution_policy".to_owned(), "sequential".to_owned()),
            ("telemetry".to_owned(), self.telemetry_mode().to_owned()),
        ]);
        metadata.extend(self.aot_metadata.clone());
        metadata
    }

    #[getter]
    /// Telemetry collection level configured on this solver.
    pub fn telemetry_mode(&self) -> &'static str {
        match self.telemetry_mode {
            BeTelemetryMode::Off => "off",
            BeTelemetryMode::Counters => "counters",
            BeTelemetryMode::Timings => "timings",
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "BackwardEulerSolver(status='{}', telemetry='{}')",
            self.status(),
            self.telemetry_mode()
        )
    }
}

fn map_be_error(error: BeError) -> PyErr {
    let native_kind = crate::errors::native_variant(&error);
    let message = error.to_string();
    crate::errors::annotate_native(
        match error {
            BeError::InvalidConfiguration(_) => InvalidConfigurationError::new_err(message),
            BeError::Backend(error) => map_backend_error(error),
            BeError::GeneratedBackend(error) => {
                map_backend_error(IvpBackendError::from_generated_error(error))
            }
            BeError::StepLimit { .. } => ExhaustionError::new_err(message),
            BeError::Newton(_) | BeError::StepUnderflow { .. } => SolverError::new_err(message),
        },
        "solve",
        "backward_euler",
        "native",
        &native_kind,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backward_euler_route_accepts_dense_aot() {
        let options = SolverOptions::new(
            Some("atom_native"),
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
