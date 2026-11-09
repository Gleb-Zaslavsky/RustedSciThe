//! Thin Python adapter for the nonstiff `ODE_api2::UniversalODESolver` route.
//!
//! Nonstiff adapter for the fixed-step legacy routes and adaptive DOP853.

use std::collections::{HashMap, HashSet};

use nalgebra::{DMatrix, DVector};
use pyo3::prelude::*;

use crate::contract::SolverOptions;
use crate::errors::{
    ExhaustionError, InvalidConfigurationError, SolverError, SolverPanicState,
    UnsupportedOptionError, annotate_native,
};
use crate::expr::PyExpr;
use crate::numpy_support;
use rustedscithe_core::numerical::ODE_api2::{
    NonStiffMethod, SolverType, UniversalIvpStatistics, UniversalODESolver,
};
use rustedscithe_core::symbolic::ivp_telemetry::{IvpTelemetry, IvpTelemetryMode, IvpWarmStage};

#[pyclass(name = "UniversalNonStiffSolution", module = "rustedscithe")]
/// Detached trajectory produced by one of the ODE_api2 nonstiff methods.
pub struct UniversalNonStiffSolution {
    method: String,
    status: String,
    times: Vec<f64>,
    states: Vec<Vec<f64>>,
    telemetry_mode: String,
    statistics: Option<UniversalIvpStatistics>,
    callback_evaluations: Option<u64>,
    callback_time_ms: Option<f64>,
}

#[pymethods]
impl UniversalNonStiffSolution {
    #[getter]
    /// Canonical method name selected in `ODE_api2`.
    pub fn method(&self) -> &str {
        &self.method
    }

    #[getter]
    /// Native termination status, including `stopped_by_condition`.
    pub fn status(&self) -> &str {
        &self.status
    }

    /// Detached time samples collected by the native nonstiff solver.
    pub fn times(&self) -> Vec<f64> {
        self.times.clone()
    }

    /// Detached row-major trajectory: one row per returned time sample.
    pub fn states(&self) -> Vec<Vec<f64>> {
        self.states.clone()
    }

    /// Owned NumPy view of `times()`.
    pub fn times_array<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f64>> {
        numpy_support::vector(py, &self.times)
    }

    /// Owned NumPy view of `states()` with shape `(samples, state_dimension)`.
    pub fn states_array<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        numpy_support::matrix(py, &self.states)
    }

    /// Number of trajectory samples returned by the native solver.
    #[getter]
    pub fn sample_count(&self) -> usize {
        self.times.len()
    }

    /// Stable metadata without fabricated timing/counter values.
    pub fn diagnostics(&self) -> HashMap<String, String> {
        let mut diagnostics = HashMap::from([
            ("solver".to_owned(), "ode_api2_nonstiff".to_owned()),
            ("method".to_owned(), self.method.clone()),
            ("status".to_owned(), self.status.clone()),
            ("samples".to_owned(), self.times.len().to_string()),
            ("telemetry".to_owned(), self.telemetry_mode.clone()),
        ]);
        if let Some(stats) = &self.statistics {
            diagnostics.insert(
                "accepted_steps".to_owned(),
                stats.accepted_steps_total.to_string(),
            );
            diagnostics.insert(
                "rejected_steps".to_owned(),
                stats.rejected_step_attempts_total.to_string(),
            );
            diagnostics.insert(
                "rhs_evaluations".to_owned(),
                stats.residual_calls.to_string(),
            );
            diagnostics.insert("solve_ms".to_owned(), stats.solve_ms_total.to_string());
        }
        if let Some(evaluations) = self.callback_evaluations {
            diagnostics.insert("callback_evaluations".to_owned(), evaluations.to_string());
        }
        if let Some(elapsed_ms) = self.callback_time_ms {
            diagnostics.insert("callback_time_ms".to_owned(), elapsed_ms.to_string());
        }
        diagnostics
    }
}

/// Python entry point for the nonstiff method family in `ODE_api2`.
///
/// This is not a replacement for direct BDF/Radau/Backward Euler/LSODE2
/// bindings. It only accepts the three methods implemented by the native
/// nonstiff adapter: RK45, DOPRI and AB4.
#[pyclass(name = "UniversalNonStiffSolver", module = "rustedscithe", unsendable)]
pub struct UniversalNonStiffSolver {
    solver: UniversalODESolver,
    method: String,
    route: String,
    telemetry_mode: String,
    panic_state: SolverPanicState,
}

#[pymethods]
impl UniversalNonStiffSolver {
    /// Construct a nonstiff solver. DOP853 interprets `step_size` as `max_step`
    /// and uses `rtol`, `atol`, and `first_step` for adaptive control.
    ///
    #[new]
    #[pyo3(signature=(method, expressions, state_variables, time_variable, y0, t0, t_end, step_size, *, options=None, stop_condition=None, neighborhood_check=1e-3, rtol=1e-8, atol=1e-10, first_step=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        method: &str,
        expressions: Vec<PyExpr>,
        state_variables: Vec<String>,
        time_variable: &str,
        y0: Vec<f64>,
        t0: f64,
        t_end: f64,
        step_size: f64,
        options: Option<PyRef<'_, SolverOptions>>,
        stop_condition: Option<HashMap<String, f64>>,
        neighborhood_check: f64,
        rtol: f64,
        atol: f64,
        first_step: Option<f64>,
    ) -> PyResult<Self> {
        let (native_method, canonical_method) = match method.trim().to_ascii_lowercase().as_str() {
            "rk45" => (NonStiffMethod::Rk45, "rk45"),
            "dopri" => (NonStiffMethod::Dopri, "dopri"),
            "ab4" => (NonStiffMethod::Ab4, "ab4"),
            "dop853" => (NonStiffMethod::Dop853, "dop853"),
            other => {
                return Err(UnsupportedOptionError::new_err(format!(
                    "ODE_api2 nonstiff method '{other}' is unsupported; expected rk45, dopri, ab4 or dop853"
                )));
            }
        };
        if native_method != NonStiffMethod::Dop853
            && (rtol != 1e-8 || atol != 1e-10 || first_step.is_some())
        {
            return Err(UnsupportedOptionError::new_err(
                "rtol, atol and first_step are only supported by DOP853",
            ));
        }
        validate_problem(
            &expressions,
            &state_variables,
            time_variable,
            &y0,
            t0,
            t_end,
            step_size,
            neighborhood_check,
            rtol,
            atol,
            first_step,
            stop_condition.as_ref(),
        )?;
        if let Some(options) = options.as_ref() {
            validate_options(options, native_method == NonStiffMethod::Dop853)?;
        }

        let mut solver = UniversalODESolver::new(
            expressions.into_iter().map(|expr| expr.inner).collect(),
            state_variables,
            time_variable.to_owned(),
            SolverType::NonStiff(native_method),
            t0,
            DVector::from_vec(y0),
            t_end,
        );
        solver.set_step_size(step_size);
        solver.set_rtol(rtol);
        solver.set_atol(atol);
        solver.set_max_step(step_size);
        solver.set_first_step(first_step);
        let telemetry_mode = options
            .as_ref()
            .map_or("off", |options| options.telemetry())
            .to_owned();
        solver = solver.with_nonstiff_symbolic_telemetry(telemetry_from_name(&telemetry_mode)?);
        if let Some(conditions) = stop_condition {
            solver.set_stop_condition(conditions);
        }
        solver.set_neighborhood_check(neighborhood_check);

        let route = format!("{canonical_method}:expr_legacy:lambdify:dense:sequential");
        Ok(Self {
            solver,
            method: canonical_method.to_owned(),
            route,
            telemetry_mode,
            panic_state: SolverPanicState::default(),
        })
    }

    /// Integrate and return a detached trajectory result.
    ///
    /// The native nonstiff loop stops after the first step that reaches or
    /// crosses `t_end`; the final sample may therefore overshoot the bound by
    /// less than one integration step.
    pub fn solve(&mut self) -> PyResult<UniversalNonStiffSolution> {
        let panic_state = self.panic_state.clone();
        let route = self.route.clone();
        let method = self.method.clone();
        panic_state.run("solve", "ode_api2_nonstiff", &route, || {
            self.solver.try_solve().map_err(|error| {
                annotate_native(
                    SolverError::new_err(error.to_string()),
                    "solve",
                    "ode_api2_nonstiff",
                    &route,
                    &format!("{error:?}")
                        .split(['{', '(', ' '])
                        .next()
                        .unwrap_or("Unknown"),
                )
            })?;
            let status = self
                .solver
                .get_status()
                .unwrap_or_else(|| "failed".to_owned());
            if status == "failed" {
                return Err(annotate_native(
                    ExhaustionError::new_err(format!(
                        "ODE_api2 {method} stopped before reaching the integration bound"
                    )),
                    "solve",
                    "ode_api2_nonstiff",
                    &route,
                    "NonStiffStepFailure",
                ));
            }
            let (times, states) = self.solver.get_result();
            let times = times.ok_or_else(|| {
                SolverError::new_err("ODE_api2 nonstiff solve returned no time samples")
            })?;
            let states = states.ok_or_else(|| {
                SolverError::new_err("ODE_api2 nonstiff solve returned no state samples")
            })?;
            let callback_snapshot = self.solver.nonstiff_symbolic_telemetry();
            Ok(UniversalNonStiffSolution {
                method,
                status,
                times: times.iter().copied().collect(),
                states: matrix_rows(&states),
                telemetry_mode: self.telemetry_mode.clone(),
                statistics: self.solver.get_statistics(),
                callback_evaluations: callback_snapshot
                    .as_ref()
                    .filter(|snapshot| snapshot.mode.collects_counters())
                    .map(|snapshot| snapshot.residual_evaluations),
                callback_time_ms: callback_snapshot
                    .as_ref()
                    .filter(|snapshot| snapshot.mode.collects_timings())
                    .map(|snapshot| {
                        snapshot.warm[IvpWarmStage::ResidualEvaluation as usize]
                            .elapsed
                            .as_secs_f64()
                            * 1e3
                    }),
            })
        })
    }

    /// Canonical native nonstiff method name.
    #[getter]
    pub fn method(&self) -> &str {
        &self.method
    }

    /// Describe the deliberately narrow native route selected at construction.
    pub fn provenance(&self) -> HashMap<String, String> {
        HashMap::from([
            ("solver".to_owned(), "ode_api2_nonstiff".to_owned()),
            ("method".to_owned(), self.method.clone()),
            ("frontend".to_owned(), "expr_legacy".to_owned()),
            ("execution".to_owned(), "lambdify".to_owned()),
            ("matrix_layout".to_owned(), "dense".to_owned()),
            ("execution_policy".to_owned(), "sequential".to_owned()),
            ("telemetry".to_owned(), self.telemetry_mode.clone()),
        ])
    }
}

fn validate_options(options: &SolverOptions, adaptive_dop853: bool) -> PyResult<()> {
    crate::contract::validate_backend_options_for(options, "ode_api2_nonstiff")?;
    if options.frontend() != "expr_legacy"
        || options.execution() != "lambdify"
        || options.matrix_layout() != "dense"
        || options.execution_policy() != "sequential"
    {
        return Err(UnsupportedOptionError::new_err(
            "ODE_api2 nonstiff requires ExprLegacy, Lambdify, Dense and Sequential",
        ));
    }
    if !adaptive_dop853 && options.telemetry() != "off" {
        return Err(UnsupportedOptionError::new_err(
            "ODE_api2 nonstiff does not provide native telemetry; set telemetry='off'",
        ));
    }
    options.validate_aot_options()?;
    let _ = crate::contract::native_logging_level(options, "ode_api2_nonstiff")?;
    Ok(())
}

fn telemetry_from_name(name: &str) -> PyResult<IvpTelemetry> {
    let mode = match name {
        "off" => IvpTelemetryMode::Off,
        "counters" => IvpTelemetryMode::Counters,
        "timings" => IvpTelemetryMode::Timings,
        "detailed" => IvpTelemetryMode::Detailed,
        _ => return Err(InvalidConfigurationError::new_err("unknown telemetry mode")),
    };
    Ok(IvpTelemetry::with_mode(mode))
}

#[allow(clippy::too_many_arguments)]
fn validate_problem(
    expressions: &[PyExpr],
    states: &[String],
    time_variable: &str,
    y0: &[f64],
    t0: f64,
    t_end: f64,
    step_size: f64,
    neighborhood_check: f64,
    rtol: f64,
    atol: f64,
    first_step: Option<f64>,
    stop_condition: Option<&HashMap<String, f64>>,
) -> PyResult<()> {
    if states.is_empty() || expressions.len() != states.len() || y0.len() != states.len() {
        return Err(InvalidConfigurationError::new_err(
            "state_variables, expressions and y0 must have the same non-zero length",
        ));
    }
    let mut declared = HashSet::with_capacity(states.len() + 1);
    for name in states.iter().map(String::as_str).chain([time_variable]) {
        if name.trim().is_empty() || !declared.insert(name) {
            return Err(InvalidConfigurationError::new_err(
                "state and time variable names must be non-empty and unique",
            ));
        }
    }
    if !t0.is_finite() || !t_end.is_finite() || t_end < t0 {
        return Err(InvalidConfigurationError::new_err(
            "t0 and t_end must be finite and t_end must be >= t0; nonstiff ODE_api2 is forward-only",
        ));
    }
    if !step_size.is_finite() || step_size <= 0.0 {
        return Err(InvalidConfigurationError::new_err(
            "step_size must be finite and positive",
        ));
    }
    if !rtol.is_finite() || rtol <= 0.0 || !atol.is_finite() || atol <= 0.0 {
        return Err(InvalidConfigurationError::new_err(
            "rtol and atol must be finite and positive",
        ));
    }
    if first_step.is_some_and(|step| !step.is_finite() || step <= 0.0 || step > t_end - t0) {
        return Err(InvalidConfigurationError::new_err(
            "first_step must be positive, finite and no larger than the interval",
        ));
    }
    if y0.iter().any(|value| !value.is_finite()) {
        return Err(InvalidConfigurationError::new_err(
            "y0 must contain only finite values",
        ));
    }
    if !neighborhood_check.is_finite() || neighborhood_check <= 0.0 {
        return Err(InvalidConfigurationError::new_err(
            "neighborhood_check must be finite and positive",
        ));
    }
    if let Some(conditions) = stop_condition {
        for (name, value) in conditions {
            if !states.contains(name) {
                return Err(InvalidConfigurationError::new_err(format!(
                    "stop_condition references undeclared state '{name}'"
                )));
            }
            if !value.is_finite() {
                return Err(InvalidConfigurationError::new_err(
                    "stop_condition targets must be finite",
                ));
            }
        }
    }
    for (index, expression) in expressions.iter().enumerate() {
        if let Some(unknown) = expression
            .inner
            .extract_variables()
            .1
            .iter()
            .find(|name| !declared.contains(name.as_str()))
        {
            return Err(InvalidConfigurationError::new_err(format!(
                "RHS expression {index} references undeclared symbol '{unknown}'"
            )));
        }
    }
    Ok(())
}

fn matrix_rows(matrix: &DMatrix<f64>) -> Vec<Vec<f64>> {
    matrix
        .row_iter()
        .map(|row| row.iter().copied().collect())
        .collect()
}
