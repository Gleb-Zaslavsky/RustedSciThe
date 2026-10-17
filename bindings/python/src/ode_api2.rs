//! Thin Python adapter for the nonstiff `ODE_api2::UniversalODESolver` route.
//!
//! The native nonstiff solver currently owns ExprLegacy/Lambdify execution,
//! fixed-step initialization, optional state stop conditions and no telemetry
//! snapshot. Keep this wrapper deliberately narrower than the stiff solver
//! adapters and reject common options that the native route would ignore.

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
use rustedscithe_core::numerical::ODE_api2::{NonStiffMethod, SolverType, UniversalODESolver};

#[pyclass(name = "UniversalNonStiffSolution", module = "rustedscithe")]
/// Detached trajectory produced by one of the ODE_api2 nonstiff methods.
pub struct UniversalNonStiffSolution {
    method: String,
    status: String,
    times: Vec<f64>,
    states: Vec<Vec<f64>>,
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
        HashMap::from([
            ("solver".to_owned(), "ode_api2_nonstiff".to_owned()),
            ("method".to_owned(), self.method.clone()),
            ("status".to_owned(), self.status.clone()),
            ("samples".to_owned(), self.times.len().to_string()),
            (
                "telemetry".to_owned(),
                "not_provided_by_native_route".to_owned(),
            ),
        ])
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
    panic_state: SolverPanicState,
}

#[pymethods]
impl UniversalNonStiffSolver {
    /// Construct an `ODE_api2` nonstiff solver with its native fixed step.
    ///
    /// `options`, when supplied, must describe the only native route currently
    /// supported here: ExprLegacy + Lambdify + Dense + Sequential, with
    /// telemetry disabled. Use the dedicated stiff solver classes for their
    /// richer frontend/backend, AOT, callback and telemetry controls.
    #[new]
    #[pyo3(signature=(method, expressions, state_variables, time_variable, y0, t0, t_end, step_size, *, options=None, stop_condition=None, neighborhood_check=1e-3))]
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
    ) -> PyResult<Self> {
        let (native_method, canonical_method) = match method.trim().to_ascii_lowercase().as_str() {
            "rk45" => (NonStiffMethod::Rk45, "rk45"),
            "dopri" => (NonStiffMethod::Dopri, "dopri"),
            "ab4" => (NonStiffMethod::Ab4, "ab4"),
            other => {
                return Err(UnsupportedOptionError::new_err(format!(
                    "ODE_api2 nonstiff method '{other}' is unsupported; expected rk45, dopri or ab4"
                )));
            }
        };
        validate_problem(
            &expressions,
            &state_variables,
            time_variable,
            &y0,
            t0,
            t_end,
            step_size,
            neighborhood_check,
            stop_condition.as_ref(),
        )?;
        if let Some(options) = options.as_ref() {
            validate_options(options)?;
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
        if let Some(conditions) = stop_condition {
            solver.set_stop_condition(conditions);
        }
        solver.set_neighborhood_check(neighborhood_check);

        let route = format!("{canonical_method}:expr_legacy:lambdify:dense:sequential");
        Ok(Self {
            solver,
            method: canonical_method.to_owned(),
            route,
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
            Ok(UniversalNonStiffSolution {
                method,
                status,
                times: times.iter().copied().collect(),
                states: matrix_rows(&states),
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
            (
                "telemetry".to_owned(),
                "not_provided_by_native_route".to_owned(),
            ),
        ])
    }
}

fn validate_options(options: &SolverOptions) -> PyResult<()> {
    if options.frontend() != "expr_legacy"
        || options.execution() != "lambdify"
        || options.matrix_layout() != "dense"
        || options.execution_policy() != "sequential"
    {
        return Err(UnsupportedOptionError::new_err(
            "ODE_api2 nonstiff requires ExprLegacy, Lambdify, Dense and Sequential",
        ));
    }
    if options.telemetry() != "off" {
        return Err(UnsupportedOptionError::new_err(
            "ODE_api2 nonstiff does not provide native telemetry; set telemetry='off'",
        ));
    }
    options.validate_aot_options()?;
    let _ = crate::contract::native_logging_level(options, "ode_api2_nonstiff")?;
    Ok(())
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
