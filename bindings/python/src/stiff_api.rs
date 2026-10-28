//! PyO3 surface for the native StiffAPI policy and preset catalog.
//!
//! Preset composition and applicability are delegated to the Rust core. This
//! module intentionally does not implement solver selection in Python.

use std::collections::HashMap;
use std::path::PathBuf;

use crate::errors::{
    CallbackFailureState, InvalidConfigurationError, SolverPanicState, UnsupportedOptionError,
};
use nalgebra::DVector;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use rustedscithe_core::Utils::postprocessing::{
    PostprocessAction, PostprocessPlan, PostprocessReport, PostprocessStatus,
};
use rustedscithe_core::numerical::BDF::BDF_api::BdfSolverOptions;
use rustedscithe_core::numerical::BE::BeSolverOptions;
use rustedscithe_core::numerical::LSODE2::{
    Lsode2Method, Lsode2NativeExecutionConfig, Lsode2ProblemConfig, Lsode2ResidualJacobianSource,
    Lsode2SymbolicExecutionMode,
};
use rustedscithe_core::numerical::Radau::api::{
    RadauAotConfig, RadauConfig, RadauExecution, RadauJacobianSource, RadauOutputPolicy,
    RadauProblem,
};
use rustedscithe_core::numerical::SolverTelemetryMode;
use rustedscithe_core::numerical::StiffAPI::{
    PreparedStiffSolver, StiffCallbackPolicy, StiffFrontend, StiffMatrixLayout,
    StiffPostprocessError, StiffPreset, StiffPresetApplicability, StiffRestartError,
    StiffSolveError, StiffSolveOutcome, StiffSolverKind, StiffSolverRequest,
};
use rustedscithe_core::numerical::logging::SolverLogLevel;
use rustedscithe_core::symbolic::symbolic_engine::Expr;

use crate::aot::{generated_config, symbolic_backend};
use crate::bdf::{BdfSolution, map_bdf_error, validate_route};
use crate::be::{BackwardEulerSolution, map_be_error};
use crate::contract::SolverOptions;
use crate::expr::PyExpr;
use crate::lsode2::{Lsode2Solution, map_lsode2_error};
use crate::radau::{RadauSolution, map_radau_error};
use crate::parameter_loop::{PyIvpParameterLoopPoint, run_parameter_loop};

/// A composable native profile for common stiff-solver choices.
///
/// Profiles only override the axes they name. Solver-specific controls such
/// as step sizes, LSODE2 controller selection, and Radau output remain owned
/// by the selected solver's normal configuration.
#[pyclass(name = "StiffPreset", module = "rustedscithe")]
#[derive(Clone)]
pub struct PyStiffPreset {
    inner: StiffPreset,
}

/// One-shot native StiffAPI request. Request construction is explicit and
/// does not add config copies to ordinary direct-solver wrappers.
#[pyclass(name = "StiffSolverRequest", module = "rustedscithe", unsendable)]
pub struct PyStiffSolverRequest {
    inner: Option<StiffSolverRequest>,
    callback_failure: Option<CallbackFailureState>,
    callback_route: Option<String>,
}

/// A native StiffAPI solver prepared from a typed request and optional profile.
/// `solve()` returns the selected solver's ordinary detached result class.
#[pyclass(name = "PreparedStiffSolver", module = "rustedscithe", unsendable)]
pub struct PyPreparedStiffSolver {
    inner: PreparedStiffSolver,
    callback_failure: Option<CallbackFailureState>,
    callback_route: Option<String>,
    panic_state: SolverPanicState,
}

/// Explicit Rust-core output actions shared by prepared stiff solver results.
#[pyclass(name = "StiffPostprocessPlan", module = "rustedscithe")]
#[derive(Clone, Default)]
pub struct PyStiffPostprocessPlan {
    inner: PostprocessPlan,
}

#[pymethods]
impl PyStiffPostprocessPlan {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// Add CSV output. Parent directories are created by the native plan.
    fn save_csv(&self, path: String) -> Self {
        Self {
            inner: self.inner.clone().save_csv(PathBuf::from(path)),
        }
    }

    /// Add the native text-table output action.
    fn save_txt(&self, path: String) -> Self {
        Self {
            inner: self.inner.clone().save_txt(PathBuf::from(path)),
        }
    }

    /// Add a compact Markdown report describing the dataset.
    fn write_report(&self, path: String) -> Self {
        Self {
            inner: self.inner.clone().write_report(PathBuf::from(path)),
        }
    }

    /// Add Plotters PNG output, one image per state variable.
    fn plotters_png(&self, output_dir: String) -> Self {
        Self {
            inner: self.inner.clone().plotters_png(PathBuf::from(output_dir)),
        }
    }

    /// Add optional Gnuplot PNG output; unavailable Gnuplot is reported skipped.
    fn gnuplot_png(&self, output_dir: String) -> Self {
        Self {
            inner: self.inner.clone().gnuplot_png(PathBuf::from(output_dir)),
        }
    }

    /// Add terminal plotting. This action writes to the current terminal.
    fn terminal_plot(&self) -> Self {
        Self {
            inner: self.inner.clone().terminal_plot(),
        }
    }

    fn __repr__(&self) -> String {
        format!("StiffPostprocessPlan(actions={})", self.inner.actions.len())
    }
}

#[pymethods]
impl PyStiffSolverRequest {
    /// Build a native BDF request. Its full route and numerical settings are
    /// represented by `SolverOptions` plus the BDF-specific arguments here.
    #[staticmethod]
    #[pyo3(signature=(expressions, state_variables, time_variable, y0, t0, t_end, max_step, rtol=1e-6, atol=1e-8, options=None, parameters=None, parameter_values=None, first_step=None, max_bdf_order=5, max_steps=1_000_000))]
    #[allow(clippy::too_many_arguments)]
    fn bdf(
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
        if expressions.is_empty() || expressions.len() != state_variables.len() {
            return Err(InvalidConfigurationError::new_err(
                "BDF requires one RHS expression for each state variable",
            ));
        }
        if y0.len() != state_variables.len() || y0.iter().any(|v| !v.is_finite()) {
            return Err(InvalidConfigurationError::new_err(
                "y0 must contain one finite value per state variable",
            ));
        }
        if !t0.is_finite()
            || !t_end.is_finite()
            || t0 == t_end
            || !max_step.is_finite()
            || max_step <= 0.0
            || !rtol.is_finite()
            || rtol <= 0.0
            || !atol.is_finite()
            || atol <= 0.0
        {
            return Err(InvalidConfigurationError::new_err(
                "BDF interval, max_step and tolerances must be finite and valid",
            ));
        }
        if !(1..=5).contains(&max_bdf_order) || max_steps == 0 {
            return Err(InvalidConfigurationError::new_err(
                "max_bdf_order must be between 1 and 5 and max_steps must be positive",
            ));
        }
        let parameters = parameters.unwrap_or_default();
        let values = parameter_values.unwrap_or_default();
        if parameters.len() != values.len() || values.iter().any(|v| !v.is_finite()) {
            return Err(InvalidConfigurationError::new_err(
                "parameters and finite parameter_values must have matching lengths",
            ));
        }
        let options = match options {
            Some(options) => options.clone(),
            None => SolverOptions::new(None, None, None, None, None, None, None, None, None)?,
        };
        let (telemetry, execution_policy) = validate_route(&options)?;
        let logging = crate::contract::native_logging_level(&options, "bdf")?;
        let native_expressions: Vec<Expr> =
            expressions.into_iter().map(|expr| expr.inner).collect();
        let mut config = BdfSolverOptions::for_bdf(
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
        .with_telemetry_mode(telemetry)
        .with_logging_level(logging);
        if options.execution() == "aot" {
            config = config.with_generated_backend_config(generated_config(&options)?);
        }
        if !parameters.is_empty() {
            config = config
                .with_equation_parameters(parameters)
                .with_equation_parameter_values(DVector::from_vec(values));
        }
        Ok(Self {
            inner: Some(StiffSolverRequest::Bdf { options: config }),
            callback_failure: None,
            callback_route: None,
        })
    }

    /// Build a native Backward Euler request while retaining its fixed-step
    /// semantics and separately bound symbolic parameters.
    #[staticmethod]
    #[pyo3(signature=(expressions, state_variables, time_variable, y0, t0, t_end, h=None, tolerance=1e-6, max_iterations=100, options=None, parameters=None, parameter_values=None, max_steps=1_000_000))]
    #[allow(clippy::too_many_arguments)]
    fn backward_euler(
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
        max_steps: usize,
    ) -> PyResult<Self> {
        crate::be::validate_names(&state_variables, "state_variables")?;
        crate::be::finite_vector(&y0, "y0")?;
        if expressions.len() != state_variables.len() {
            return Err(InvalidConfigurationError::new_err(format!(
                "expected one RHS expression per state variable, got {} expressions for {} variables",
                expressions.len(),
                state_variables.len()
            )));
        }
        if time_variable.trim().is_empty()
            || state_variables.iter().any(|name| name == &time_variable)
        {
            return Err(InvalidConfigurationError::new_err(
                "time_variable must be non-empty and differ from state_variables",
            ));
        }
        let parameters = parameters.unwrap_or_default();
        if !parameters.is_empty() {
            crate::be::validate_names(&parameters, "parameters")?;
        }
        if parameters
            .iter()
            .any(|name| name == &time_variable || state_variables.iter().any(|state| state == name))
        {
            return Err(InvalidConfigurationError::new_err(
                "parameters must not collide with time_variable or state_variables",
            ));
        }
        crate::be::validate_expression_symbols(
            &expressions,
            &state_variables,
            &time_variable,
            &parameters,
        )?;
        crate::be::validate_scalar(t0, "t0")?;
        crate::be::validate_scalar(t_end, "t_end")?;
        if t0 == t_end {
            return Err(InvalidConfigurationError::new_err(
                "t0 and t_end must differ",
            ));
        }
        if let Some(h) = h {
            crate::be::validate_scalar(h, "h")?;
            if h <= 0.0 {
                return Err(InvalidConfigurationError::new_err(
                    "h must be greater than zero when provided",
                ));
            }
        }
        crate::be::validate_scalar(tolerance, "tolerance")?;
        if tolerance <= 0.0 || max_iterations == 0 || max_steps == 0 {
            return Err(InvalidConfigurationError::new_err(
                "tolerance, max_iterations and max_steps must be positive",
            ));
        }
        let parameter_values = match parameter_values {
            Some(values) => {
                crate::be::finite_vector(&values, "parameter_values")?;
                if values.len() != parameters.len() {
                    return Err(InvalidConfigurationError::new_err(format!(
                        "expected {} parameter values, got {}",
                        parameters.len(),
                        values.len()
                    )));
                }
                values
            }
            None => Vec::new(),
        };
        let options = match options {
            Some(options) => options.clone(),
            None => SolverOptions::new(None, None, None, None, None, None, None, None, None)?,
        };
        let (frontend, telemetry) = crate::be::validate_route(&options)?;
        let logging = crate::contract::native_logging_level(&options, "backward_euler")?;
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
        .with_telemetry_mode(telemetry)
        .with_logging_level(logging)
        .with_max_steps(max_steps);
        if options.execution() == "aot" {
            native_options =
                native_options.with_generated_backend_config(generated_config(&options)?);
        }
        Ok(Self {
            inner: Some(StiffSolverRequest::BackwardEuler {
                options: native_options,
                equation_parameters: parameters,
                parameter_values,
            }),
            callback_failure: None,
            callback_route: None,
        })
    }

    /// Build a native Radau request, including the initial state required by
    /// its solve lifecycle and optional dense-output retention.
    #[staticmethod]
    #[pyo3(signature=(expressions, state_variables, time_variable, y0, t0, t_end, max_step, rtol=1e-3, atol=1e-6, options=None, parameters=None, jacobian=None, output="final", sample_times=None, first_step=None, max_steps=1_000_000, max_newton_iterations=6, max_retries=12, parameter_values=None))]
    #[allow(clippy::too_many_arguments)]
    fn radau(
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
        jacobian: Option<Vec<PyExpr>>,
        output: &str,
        sample_times: Option<Vec<f64>>,
        first_step: Option<f64>,
        max_steps: usize,
        max_newton_iterations: usize,
        max_retries: usize,
        parameter_values: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        crate::radau::validate_names(&state_variables, "state_variables", false)?;
        if expressions.len() != state_variables.len() {
            return Err(InvalidConfigurationError::new_err(format!(
                "expected one residual expression per state variable, got {} expressions for {} variables",
                expressions.len(),
                state_variables.len()
            )));
        }
        if y0.len() != state_variables.len() || y0.iter().any(|value| !value.is_finite()) {
            return Err(InvalidConfigurationError::new_err(
                "y0 must contain one finite value per state variable",
            ));
        }
        if time_variable.trim().is_empty()
            || state_variables.iter().any(|name| name == &time_variable)
        {
            return Err(InvalidConfigurationError::new_err(
                "time_variable must be non-empty and differ from state_variables",
            ));
        }
        let parameters = parameters.unwrap_or_default();
        crate::radau::validate_names(&parameters, "parameters", true)?;
        if parameters
            .iter()
            .any(|name| name == &time_variable || state_variables.iter().any(|state| state == name))
        {
            return Err(InvalidConfigurationError::new_err(
                "parameters must not collide with time_variable or state_variables",
            ));
        }
        crate::radau::validate_expression_symbols(
            &expressions,
            &state_variables,
            &time_variable,
            &parameters,
            "residual",
        )?;
        if let Some(jacobian) = &jacobian {
            if jacobian.len() != state_variables.len() * state_variables.len() {
                return Err(InvalidConfigurationError::new_err(
                    "jacobian must contain state_dimension squared expressions",
                ));
            }
            crate::radau::validate_expression_symbols(
                jacobian,
                &state_variables,
                &time_variable,
                &parameters,
                "jacobian",
            )?;
        }
        let parameter_values = match parameter_values {
            Some(values) => {
                if values.len() != parameters.len() {
                    return Err(InvalidConfigurationError::new_err(format!(
                        "expected {} parameter values, got {}",
                        parameters.len(),
                        values.len()
                    )));
                }
                crate::radau::finite_vector(&values, "parameter_values")?;
                values
            }
            None => Vec::new(),
        };
        crate::radau::validate_scalar(t0, "t0")?;
        crate::radau::validate_scalar(t_end, "t_end")?;
        if t0 == t_end {
            return Err(InvalidConfigurationError::new_err(
                "t0 and t_end must differ",
            ));
        }
        crate::radau::validate_scalar(max_step, "max_step")?;
        if max_step <= 0.0 {
            return Err(InvalidConfigurationError::new_err(
                "max_step must be greater than zero",
            ));
        }
        crate::radau::validate_scalar(rtol, "rtol")?;
        crate::radau::validate_scalar(atol, "atol")?;
        if rtol <= 0.0 || atol <= 0.0 {
            return Err(InvalidConfigurationError::new_err(
                "rtol and atol must be greater than zero",
            ));
        }
        if let Some(first_step) = first_step {
            crate::radau::validate_scalar(first_step, "first_step")?;
            if first_step <= 0.0 || first_step > (t_end - t0).abs() {
                return Err(InvalidConfigurationError::new_err(
                    "first_step must be positive and no larger than the integration interval",
                ));
            }
        }
        if max_steps == 0 || max_newton_iterations == 0 || max_retries == 0 {
            return Err(InvalidConfigurationError::new_err(
                "max_steps, max_newton_iterations and max_retries must be greater than zero",
            ));
        }
        let options = match options {
            Some(options) => options.clone(),
            None => SolverOptions::new(None, None, None, None, None, None, None, None, None)?,
        };
        let logging = crate::contract::native_logging_level(&options, "radau5")?;
        if logging != SolverLogLevel::Off {
            return Err(UnsupportedOptionError::new_err(
                "Radau does not expose a logging-level control",
            ));
        }
        let (frontend, telemetry, execution_policy, matrix_layout) =
            crate::radau::validate_route(&options)?;
        let output = match (output.trim().to_ascii_lowercase().as_str(), sample_times) {
            ("final" | "final_only", None) => RadauOutputPolicy::FinalOnly,
            ("dense", None) => RadauOutputPolicy::Dense,
            ("sampled", Some(times)) => {
                if times.is_empty() {
                    return Err(InvalidConfigurationError::new_err(
                        "sample_times must not be empty for output='sampled'",
                    ));
                }
                crate::radau::finite_vector(&times, "sample_times")?;
                RadauOutputPolicy::Sampled(times)
            }
            ("sampled", None) => {
                return Err(InvalidConfigurationError::new_err(
                    "sample_times are required when output='sampled'",
                ));
            }
            ("final" | "final_only" | "dense", Some(_)) => {
                return Err(InvalidConfigurationError::new_err(
                    "sample_times are only valid when output='sampled'",
                ));
            }
            _ => {
                return Err(UnsupportedOptionError::new_err(
                    "Radau output must be 'final', 'dense' or 'sampled'",
                ));
            }
        };
        let execution = if options.execution() == "aot" {
            RadauExecution::Aot
        } else {
            RadauExecution::Lambdify
        };
        let aot = if options.execution() == "aot" {
            Some(RadauAotConfig {
                generated: generated_config(&options)?,
            })
        } else {
            None
        };
        let config = RadauConfig {
            t0,
            t_bound: t_end,
            rtol,
            atol,
            first_step,
            max_step,
            max_steps,
            max_newton_iterations,
            max_retries,
            execution,
            frontend,
            matrix_layout,
            jacobian_source: if jacobian.is_some() {
                RadauJacobianSource::Analytic
            } else {
                RadauJacobianSource::FiniteDifference
            },
            telemetry,
            execution_policy,
            output,
            aot,
            ..RadauConfig::default()
        };
        let mut problem = RadauProblem::new(
            expressions
                .into_iter()
                .map(|expression| expression.inner)
                .collect(),
            state_variables,
            time_variable,
        )
        .with_parameter_names(parameters);
        if let Some(jacobian) = jacobian {
            problem = problem.with_jacobian(
                jacobian
                    .into_iter()
                    .map(|expression| expression.inner)
                    .collect(),
            );
        }
        Ok(Self {
            inner: Some(StiffSolverRequest::Radau {
                problem,
                config,
                initial_state: y0,
                parameters: parameter_values,
            }),
            callback_failure: None,
            callback_route: None,
        })
    }

    /// Build an LSODE2 request with independently selectable controller and
    /// native/Bridge execution policies.
    #[staticmethod]
    #[pyo3(signature=(expressions, state_variables, time_variable, y0, t0, t_end, max_step, rtol=1e-6, atol=1e-8, options=None, method="bdf", parameters=None, parameter_values=None, native_execution="native", max_step_attempts=200_000, max_accepted_steps=200_000, first_step=None, residual_callback=None, jacobian_callback=None))]
    #[allow(clippy::too_many_arguments)]
    fn lsode2(
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
        method: &str,
        parameters: Option<Vec<String>>,
        parameter_values: Option<Vec<f64>>,
        native_execution: &str,
        max_step_attempts: usize,
        max_accepted_steps: usize,
        first_step: Option<f64>,
        residual_callback: Option<Py<PyAny>>,
        jacobian_callback: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        crate::lsode2::validate_names(&state_variables, "state_variables")?;
        crate::lsode2::finite_vector(&y0, "y0")?;
        if expressions.len() != state_variables.len() {
            return Err(InvalidConfigurationError::new_err(format!(
                "expected one RHS expression per state variable, got {} expressions for {} variables",
                expressions.len(),
                state_variables.len()
            )));
        }
        if time_variable.trim().is_empty()
            || state_variables.iter().any(|name| name == &time_variable)
        {
            return Err(InvalidConfigurationError::new_err(
                "time_variable must be non-empty and differ from state_variables",
            ));
        }
        let parameters = parameters.unwrap_or_default();
        if !parameters.is_empty() {
            crate::lsode2::validate_names(&parameters, "parameters")?;
        }
        if parameters
            .iter()
            .any(|name| name == &time_variable || state_variables.iter().any(|state| state == name))
        {
            return Err(InvalidConfigurationError::new_err(
                "parameters must not collide with time_variable or state_variables",
            ));
        }
        let parameter_values = match parameter_values {
            Some(values) => {
                crate::lsode2::finite_vector(&values, "parameter_values")?;
                if values.len() != parameters.len() {
                    return Err(InvalidConfigurationError::new_err(format!(
                        "expected {} parameter values, got {}",
                        parameters.len(),
                        values.len()
                    )));
                }
                Some(values)
            }
            None if parameters.is_empty() => None,
            None => {
                return Err(InvalidConfigurationError::new_err(
                    "parameter_values are required when parameters are declared",
                ));
            }
        };
        crate::lsode2::validate_expression_symbols(
            &expressions,
            &state_variables,
            &time_variable,
            &parameters,
        )?;
        crate::lsode2::validate_scalar(t0, "t0")?;
        crate::lsode2::validate_scalar(t_end, "t_end")?;
        if t0 == t_end {
            return Err(InvalidConfigurationError::new_err(
                "t0 and t_end must differ",
            ));
        }
        crate::lsode2::validate_scalar(max_step, "max_step")?;
        crate::lsode2::validate_scalar(rtol, "rtol")?;
        crate::lsode2::validate_scalar(atol, "atol")?;
        if max_step <= 0.0 || rtol <= 0.0 || atol <= 0.0 {
            return Err(InvalidConfigurationError::new_err(
                "max_step, rtol and atol must be positive",
            ));
        }
        if let Some(first_step) = first_step {
            crate::lsode2::validate_scalar(first_step, "first_step")?;
            if first_step <= 0.0 || first_step > (t_end - t0).abs() {
                return Err(InvalidConfigurationError::new_err(
                    "first_step must be positive and no larger than the integration interval",
                ));
            }
        }
        let normalized_method = method.trim().to_ascii_lowercase().replace(['-', ' '], "_");
        let requested_method = match normalized_method.as_str() {
            "bdf" => Lsode2Method::Bdf,
            "adams" => Lsode2Method::Adams,
            "auto" | "automatic" | "lsoda" => Lsode2Method::Auto,
            _ => {
                return Err(UnsupportedOptionError::new_err(
                    "LSODE2 method must be 'bdf', 'adams' or 'auto' (alias: 'lsoda')",
                ));
            }
        };
        let native_execution = match native_execution.trim().to_ascii_lowercase().as_str() {
            "native" | "native_solve" => {
                Lsode2NativeExecutionConfig::native_solve(max_step_attempts, max_accepted_steps)
            }
            "bridge" | "bridge_solve" => Lsode2NativeExecutionConfig::bridge_solve(),
            "probe_before_bridge" => Lsode2NativeExecutionConfig::probe_before_bridge(
                max_step_attempts,
                max_accepted_steps,
            ),
            _ => {
                return Err(UnsupportedOptionError::new_err(
                    "native_execution must be 'native', 'bridge' or 'probe_before_bridge'",
                ));
            }
        };
        let options = match options {
            Some(options) => options.clone(),
            None => SolverOptions::new(None, None, None, None, None, None, None, None, None)?,
        };
        let band_lower = crate::lsode2::merge_band_width(None, options.band_lower(), "band_lower")?;
        let band_upper = crate::lsode2::merge_band_width(None, options.band_upper(), "band_upper")?;
        let (frontend, structure, execution_policy, telemetry, execution) =
            crate::lsode2::validate_route(&options, band_lower, band_upper)?;
        let callback_pair = match (residual_callback, jacobian_callback) {
            (None, None) => None,
            (Some(_), None) | (None, Some(_)) => {
                return Err(InvalidConfigurationError::new_err(
                    "residual_callback and jacobian_callback must be provided together",
                ));
            }
            (Some(residual), Some(jacobian)) => {
                if structure
                    == rustedscithe_core::numerical::LSODE2::Lsode2LinearSystemStructure::BandedAuto
                {
                    return Err(UnsupportedOptionError::new_err(
                        "automatic banded layout requires symbolic equations; analytical Python callbacks need an explicit band width",
                    ));
                }
                if execution != Lsode2SymbolicExecutionMode::LambdifyExpr {
                    return Err(UnsupportedOptionError::new_err(
                        "Python analytical callbacks require execution='lambdify'; AOT callbacks must be native",
                    ));
                }
                Some(crate::lsode2::build_python_callbacks(
                    residual,
                    jacobian,
                    expressions.len(),
                )?)
            }
        };
        let logging = crate::contract::native_logging_level(&options, "lsode2")?;
        let mut config = Lsode2ProblemConfig::new(
            expressions
                .into_iter()
                .map(|expression| expression.inner)
                .collect(),
            state_variables,
            time_variable,
            t0,
            DVector::from_vec(y0),
            t_end,
            max_step,
            rtol,
            atol,
        )
        .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
            assembly: frontend,
            execution,
        })
        .with_linear_system_structure(structure)
        .with_lambdify_execution_policy(execution_policy)
        .with_telemetry(telemetry)
        .with_native_execution(native_execution)
        .with_first_step(first_step);
        config.method = requested_method;
        config = match requested_method {
            Lsode2Method::Bdf => config.with_bdf_only_controller(),
            Lsode2Method::Adams => config.with_adams_only_controller(),
            Lsode2Method::Auto => config.with_automatic_adams_bdf_controller(),
        };
        if options.execution() == "aot" {
            config.backend.generated_backend = crate::lsode2::aot_backend_config(&options)?;
        }
        if !parameters.is_empty() {
            let values = parameter_values.ok_or_else(|| {
                InvalidConfigurationError::new_err(
                    "parameter_values are required when parameters are declared",
                )
            })?;
            config = config.with_equation_parameters(parameters);
            config = config.with_equation_parameter_values(DVector::from_vec(values));
        }
        if let Some((callbacks, _)) = &callback_pair {
            config = config.with_analytical_callback_bundle(callbacks.clone());
        }
        let callback_route = callback_pair.as_ref().map(|_| {
            format!(
                "{}:{}:{}",
                options.frontend(),
                options.execution(),
                options.matrix_layout()
            )
        });
        Ok(Self {
            inner: Some(StiffSolverRequest::Lsode2 {
                config,
                logging_level: Some(logging),
            }),
            callback_failure: callback_pair.map(|(_, failure)| failure),
            callback_route,
        })
    }

    /// Apply a native profile and prepare the configured solver.
    #[pyo3(signature=(preset=None))]
    fn prepare(
        &mut self,
        preset: Option<PyRef<'_, PyStiffPreset>>,
    ) -> PyResult<PyPreparedStiffSolver> {
        let request = self.inner.take().ok_or_else(|| {
            InvalidConfigurationError::new_err("StiffSolverRequest has already been consumed")
        })?;
        let prepared = request
            .prepare(preset.as_deref().map(|profile| &profile.inner))
            .map_err(map_prepare_error)?;
        Ok(PyPreparedStiffSolver {
            inner: prepared,
            callback_failure: self.callback_failure.take(),
            callback_route: self.callback_route.take(),
            panic_state: SolverPanicState::default(),
        })
    }
}

#[pymethods]
impl PyPreparedStiffSolver {
    /// Native solver family selected by the request.
    #[getter]
    fn solver(&self) -> &'static str {
        solver_name(self.inner.kind())
    }

    /// Native telemetry collection level currently configured.
    #[getter]
    fn telemetry_mode(&self) -> &'static str {
        match self.inner.telemetry_mode() {
            SolverTelemetryMode::Off => "off",
            SolverTelemetryMode::Counters => "counters",
            SolverTelemetryMode::Timings => "timings",
            SolverTelemetryMode::Detailed => "detailed",
        }
    }

    /// Native lifecycle text where the selected solver exposes a status.
    /// Radau reports completion and telemetry on its returned solution instead.
    #[getter]
    fn status(&self) -> Option<String> {
        match &self.inner {
            PreparedStiffSolver::Bdf(solver) => Some(solver.get_status().to_owned()),
            PreparedStiffSolver::Radau(_) => None,
            PreparedStiffSolver::BackwardEuler(solver) => Some(solver.get_status()),
            PreparedStiffSolver::Lsode2(solver) => Some(solver.status().to_owned()),
        }
    }

    /// Copy the retained native trajectory into owned NumPy arrays.
    ///
    /// This is available for BDF, Backward Euler and LSODE2 after a solve.
    /// Radau output belongs to its returned `RadauSolution` and is queried
    /// through that result's sampling API.
    fn trajectory_arrays<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, numpy::PyArray1<f64>>,
        Bound<'py, numpy::PyArray2<f64>>,
    )> {
        let view = self.inner.trajectory_view().map_err(map_trajectory_error)?;
        let rows = view
            .values
            .row_iter()
            .map(|row| row.iter().copied().collect::<Vec<_>>())
            .collect::<Vec<_>>();
        Ok((
            crate::numpy_support::vector(py, view.axis.as_slice()),
            crate::numpy_support::matrix(py, &rows)?,
        ))
    }

    /// Run the Rust-core output plan against a retained BDF, BE or LSODE2 trajectory.
    fn postprocess<'py>(
        &self,
        py: Python<'py>,
        plan: PyRef<'_, PyStiffPostprocessPlan>,
        axis_name: String,
        variable_names: Vec<String>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let solver = solver_name(self.inner.kind());
        let route = self
            .callback_route
            .as_deref()
            .unwrap_or("stiff_api")
            .to_owned();
        let panic_state = self.panic_state.clone();
        panic_state.run("postprocess", solver, &route, || {
            let report = self
                .inner
                .trajectory_view()
                .and_then(|view| view.execute(axis_name, variable_names, &plan.inner))
                .map_err(map_trajectory_error)?;
            postprocess_report_dict(py, report)
        })
    }

    /// Return telemetry in a solver-tagged structure using that solver's
    /// established native metric names and report format.
    fn telemetry_snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        match &self.inner {
            PreparedStiffSolver::Bdf(solver) => {
                let result = BdfSolution::from_solver(solver);
                telemetry_snapshot_dict(
                    py,
                    "bdf",
                    result.telemetry_mode(),
                    result.telemetry_counters(),
                    result.telemetry_timings_ms(),
                    result.telemetry_report(),
                    result.telemetry_aot_metadata(),
                    result.telemetry_aot_counters(),
                    result.telemetry_aot_timings_ms(),
                    result.aot_artifact_keys(),
                )
            }
            PreparedStiffSolver::BackwardEuler(solver) => {
                let result = BackwardEulerSolution::from_solver(solver);
                telemetry_snapshot_dict(
                    py,
                    "backward_euler",
                    result.telemetry_mode(),
                    result.telemetry_counters(),
                    result.telemetry_timings_ms(),
                    result.telemetry_report(),
                    result.telemetry_aot_metadata(),
                    result.telemetry_aot_counters(),
                    result.telemetry_aot_timings_ms(),
                    result.aot_artifact_keys(),
                )
            }
            PreparedStiffSolver::Lsode2(solver) => {
                let result = Lsode2Solution::from_solver(solver);
                telemetry_snapshot_dict(
                    py,
                    "lsode2",
                    result.telemetry_mode(),
                    result.telemetry_counters(),
                    result.telemetry_timings_ms(),
                    result.telemetry_report(),
                    result.telemetry_aot_metadata(),
                    result.telemetry_aot_counters(),
                    result.telemetry_aot_timings_ms(),
                    result.aot_artifact_keys(),
                )
            }
            PreparedStiffSolver::Radau(_) => Err(crate::errors::annotate_native(
                UnsupportedOptionError::new_err(
                    "Radau telemetry is available from the RadauSolution returned by solve",
                ),
                "telemetry",
                "radau",
                "stiff_api:telemetry_snapshot",
                "RadauSolutionRequired",
            )),
        }
    }

    /// Start a fresh solve on the request's configured interval.
    ///
    /// New `y0` and parameter values are bound, but symbolic callbacks and AOT
    /// artifacts remain prepared. This is not continuation from a prior result.
    fn solve_with_parameter_values(
        &mut self,
        py: Python<'_>,
        y0: Vec<f64>,
        parameter_values: Vec<f64>,
    ) -> PyResult<Py<PyAny>> {
        solve_with_parameter_values(self, py, None, y0, parameter_values)
    }

    /// Run independent parameter solves through the selected native stiff
    /// solver. Each point starts from the same `y0` and configured interval;
    /// typed failures are retained without stopping later vectors.
    fn solve_parameter_loop(
        &mut self,
        py: Python<'_>,
        y0: Vec<f64>,
        parameter_vectors: Vec<Vec<f64>>,
    ) -> PyResult<Vec<Py<PyIvpParameterLoopPoint>>> {
        crate::radau::finite_vector(&y0, "y0")?;
        let solver = solver_name(self.inner.kind());
        let route = self.callback_route.as_deref().unwrap_or("stiff_api").to_owned();
        run_parameter_loop(py, solver, &route, parameter_vectors, |parameters| {
            solve_with_parameter_values(self, py, None, y0.clone(), parameters.to_vec())
        })
    }

    /// Start a fresh solve on a supplied interval using the prepared route.
    fn solve_with_parameter_values_in_interval(
        &mut self,
        py: Python<'_>,
        t0: f64,
        y0: Vec<f64>,
        t_end: f64,
        parameter_values: Vec<f64>,
    ) -> PyResult<Py<PyAny>> {
        solve_with_parameter_values(self, py, Some((t0, t_end)), y0, parameter_values)
    }

    /// Restart the prepared native route from a new state and interval.
    /// Symbolic preparation, generated artifacts and solver policy are retained.
    fn restart(
        &mut self,
        py: Python<'_>,
        t0: f64,
        y0: Vec<f64>,
        t_end: f64,
    ) -> PyResult<Py<PyAny>> {
        let solver = solver_name(self.inner.kind());
        let route = self
            .callback_route
            .as_deref()
            .unwrap_or("stiff_api")
            .to_owned();
        let panic_state = self.panic_state.clone();
        panic_state.run("restart", solver, &route, || {
            if let Some(failure) = &self.callback_failure {
                failure.clear();
            }
            self.inner
                .try_restart_with_initial_state(t0, DVector::from_vec(y0), t_end)
                .map_err(map_restart_error)?;
            self.solve(py)
        })
    }

    /// Continue a BDF parameter segment from its last accepted state.
    fn continue_bdf_with_parameters(
        &mut self,
        py: Python<'_>,
        parameter_values: Vec<f64>,
        t_end: f64,
    ) -> PyResult<Py<PyAny>> {
        if self.inner.as_bdf().is_none() {
            return Err(wrong_solver("BDF"));
        }
        let route = self
            .callback_route
            .as_deref()
            .unwrap_or("stiff_api")
            .to_owned();
        let panic_state = self.panic_state.clone();
        panic_state.run("continuation", "bdf", &route, || {
            let solver = self.inner.as_bdf_mut().ok_or_else(|| wrong_solver("BDF"))?;
            solver
                .try_continue_in_time_with_parameter_values(
                    DVector::from_vec(parameter_values),
                    t_end,
                )
                .map_err(map_bdf_error)?;
            solver.try_solve().map_err(map_bdf_error)?;
            Py::new(py, BdfSolution::from_solver(solver)).map(Py::into_any)
        })
    }

    /// Extend a Backward Euler trajectory, optionally rebinding equation parameters.
    #[pyo3(signature=(t_end, parameter_values=None))]
    fn continue_backward_euler(
        &mut self,
        py: Python<'_>,
        t_end: f64,
        parameter_values: Option<Vec<f64>>,
    ) -> PyResult<Py<PyAny>> {
        if self.inner.as_backward_euler().is_none() {
            return Err(wrong_solver("Backward Euler"));
        }
        let panic_state = self.panic_state.clone();
        panic_state.run("continuation", "backward_euler", "stiff_api", || {
            let solver = self
                .inner
                .as_backward_euler_mut()
                .ok_or_else(|| wrong_solver("Backward Euler"))?;
            if let Some(values) = parameter_values {
                solver
                    .try_set_parameter_values(DVector::from_vec(values))
                    .map_err(map_be_error)?;
            }
            solver.try_continue_to(t_end).map_err(map_be_error)?;
            Py::new(py, BackwardEulerSolution::from_solver(solver)).map(Py::into_any)
        })
    }

    /// Continue LSODE2 from its last successful endpoint, retaining prepared callbacks.
    fn continue_lsode2_with_parameters(
        &mut self,
        py: Python<'_>,
        parameter_values: Vec<f64>,
        t_end: f64,
    ) -> PyResult<Py<PyAny>> {
        if self.inner.as_lsode2().is_none() {
            return Err(wrong_solver("LSODE2"));
        }
        let panic_state = self.panic_state.clone();
        panic_state.run("continuation", "lsode2", "stiff_api", || {
            let (last_time, last_state) = {
                let solver = self
                    .inner
                    .as_lsode2()
                    .ok_or_else(|| wrong_solver("LSODE2"))?;
                if solver.status() != "finished_native_faithful" {
                    return Err(crate::errors::annotate_native(
                        InvalidConfigurationError::new_err(
                            "LSODE2 continuation requires a previous successful solve",
                        ),
                        "continuation",
                        "lsode2",
                        "stiff_api:continuation",
                        "PreviousSolveRequired",
                    ));
                }
                let (times, values) = solver.get_result_ref();
                let Some(time) = times.get(times.len().saturating_sub(1)).copied() else {
                    return Err(InvalidConfigurationError::new_err(
                        "LSODE2 continuation has no retained endpoint",
                    ));
                };
                if values.nrows() == 0 || values.ncols() == 0 {
                    return Err(InvalidConfigurationError::new_err(
                        "LSODE2 continuation has no retained endpoint state",
                    ));
                }
                let state = values
                    .row(values.nrows() - 1)
                    .iter()
                    .copied()
                    .collect::<Vec<_>>();
                (time, state)
            };
            let solver = self
                .inner
                .as_lsode2_mut()
                .ok_or_else(|| wrong_solver("LSODE2"))?;
            solver
                .set_parameter_values(DVector::from_vec(parameter_values))
                .map_err(map_lsode2_error)?;
            solver
                .try_restart_with_initial_state(last_time, DVector::from_vec(last_state), t_end)
                .map_err(map_lsode2_error)?;
            self.solve(py)
        })
    }

    /// Dispatch solve in Rust and retain each solver's established result type.
    fn solve(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let solver = solver_name(self.inner.kind());
        let route = self
            .callback_route
            .as_deref()
            .unwrap_or("stiff_api")
            .to_owned();
        let panic_state = self.panic_state.clone();
        panic_state.run("solve", solver, &route, || {
            let callback_failure = self.callback_failure.clone();
            let callback_route = self.callback_route.as_deref().unwrap_or("stiff_api");
            if let Some(failure) = &callback_failure {
                failure.clear();
            }
            let outcome = match self.inner.try_solve() {
                Ok(outcome) => outcome,
                Err(error) => {
                    if let Some(callback_error) = callback_failure
                        .as_ref()
                        .and_then(|failure| failure.take_pyerr("solve", "lsode2", callback_route))
                    {
                        return Err(callback_error);
                    }
                    return Err(map_solve_error(error));
                }
            };
            if let Some(callback_error) = callback_failure
                .as_ref()
                .and_then(|failure| failure.take_pyerr("solve", "lsode2", callback_route))
            {
                return Err(callback_error);
            }
            match outcome {
                StiffSolveOutcome::Bdf => {
                    let solver = self.inner.as_bdf().expect("BDF outcome retains solver");
                    Py::new(py, BdfSolution::from_solver(solver)).map(Py::into_any)
                }
                StiffSolveOutcome::Radau(solution) => {
                    Py::new(py, RadauSolution::from_native(solution)).map(Py::into_any)
                }
                StiffSolveOutcome::BackwardEuler => {
                    let solver = self
                        .inner
                        .as_backward_euler()
                        .expect("BE outcome retains solver");
                    Py::new(py, BackwardEulerSolution::from_solver(solver)).map(Py::into_any)
                }
                StiffSolveOutcome::Lsode2(_) => {
                    let solver = self
                        .inner
                        .as_lsode2()
                        .expect("LSODE2 outcome retains solver");
                    Py::new(py, Lsode2Solution::from_solver(solver)).map(Py::into_any)
                }
            }
        })
    }
}

fn solve_with_parameter_values(
    prepared: &mut PyPreparedStiffSolver,
    py: Python<'_>,
    interval: Option<(f64, f64)>,
    y0: Vec<f64>,
    parameter_values: Vec<f64>,
) -> PyResult<Py<PyAny>> {
    crate::radau::finite_vector(&y0, "y0")?;
    crate::radau::finite_vector(&parameter_values, "parameter_values")?;
    if let Some((t0, t_end)) = interval {
        crate::radau::validate_scalar(t0, "t0")?;
        crate::radau::validate_scalar(t_end, "t_end")?;
        if t0 == t_end {
            return Err(InvalidConfigurationError::new_err(
                "t0 and t_end must differ",
            ));
        }
    }
    let solver = solver_name(prepared.inner.kind());
    let route = prepared
        .callback_route
        .as_deref()
        .unwrap_or("stiff_api")
        .to_owned();
    let panic_state = prepared.panic_state.clone();
    panic_state.run("solve", solver, &route, || {
        if let Some(failure) = &prepared.callback_failure {
            failure.clear();
        }
        let result = match interval {
            Some((t0, t_end)) => prepared.inner.try_solve_with_parameter_values_in_interval(
                t0,
                DVector::from_vec(y0),
                t_end,
                DVector::from_vec(parameter_values),
            ),
            None => prepared.inner.try_solve_with_parameter_values(
                DVector::from_vec(y0),
                DVector::from_vec(parameter_values),
            ),
        };
        let outcome = match result {
            Ok(outcome) => outcome,
            Err(error) => {
                if let Some(callback_error) = prepared
                    .callback_failure
                    .as_ref()
                    .and_then(|failure| failure.take_pyerr("solve", "lsode2", &route))
                {
                    return Err(callback_error);
                }
                return Err(map_solve_error(error));
            }
        };
        if let Some(callback_error) = prepared
            .callback_failure
            .as_ref()
            .and_then(|failure| failure.take_pyerr("solve", "lsode2", &route))
        {
            return Err(callback_error);
        }
        match outcome {
            StiffSolveOutcome::Bdf => {
                let solver = prepared
                    .inner
                    .as_bdf()
                    .expect("BDF outcome retains its prepared solver");
                Py::new(py, BdfSolution::from_solver(solver)).map(Py::into_any)
            }
            StiffSolveOutcome::Radau(solution) => {
                Py::new(py, RadauSolution::from_native(solution)).map(Py::into_any)
            }
            StiffSolveOutcome::BackwardEuler => {
                let solver = prepared
                    .inner
                    .as_backward_euler()
                    .expect("Backward Euler outcome retains its prepared solver");
                Py::new(py, BackwardEulerSolution::from_solver(solver)).map(Py::into_any)
            }
            StiffSolveOutcome::Lsode2(_) => {
                let solver = prepared
                    .inner
                    .as_lsode2()
                    .expect("LSODE2 outcome retains its prepared solver");
                Py::new(py, Lsode2Solution::from_solver(solver)).map(Py::into_any)
            }
        }
    })
}

fn wrong_solver(expected: &str) -> PyErr {
    crate::errors::annotate_native(
        UnsupportedOptionError::new_err(format!(
            "continuation method is only available for {expected} requests"
        )),
        "continuation",
        "stiff_api",
        "stiff_api:continuation",
        "WrongSolverRoute",
    )
}

/// Execute a shared plan against Radau's explicit sampled solution output.
pub(crate) fn postprocess_radau_solution(
    py: Python<'_>,
    solution: &crate::radau::RadauSolution,
    plan: &PyStiffPostprocessPlan,
    sample_times: &[f64],
    axis_name: String,
    variable_names: Vec<String>,
) -> PyResult<Py<PyDict>> {
    let dataset = rustedscithe_core::numerical::StiffAPI::radau_postprocess_dataset(
        &solution.native,
        sample_times,
        axis_name,
        variable_names,
    )
    .map_err(map_trajectory_error)?;
    let report = plan.inner.execute(&dataset).map_err(|source| {
        map_trajectory_error(StiffPostprocessError::PlanExecution {
            solver: StiffSolverKind::Radau,
            source,
        })
    })?;
    postprocess_report_dict(py, report).map(Bound::unbind)
}

fn postprocess_report_dict<'py>(
    py: Python<'py>,
    report: PostprocessReport,
) -> PyResult<Bound<'py, PyDict>> {
    let root = PyDict::new(py);
    root.set_item("all_done", report.all_done())?;
    let entries = report
        .entries
        .into_iter()
        .map(|entry| {
            let action = match entry.action {
                PostprocessAction::SaveTxt { .. } => "save_txt",
                PostprocessAction::SaveCsv { .. } => "save_csv",
                PostprocessAction::PlottersPng { .. } => "plotters_png",
                PostprocessAction::GnuplotPng { .. } => "gnuplot_png",
                PostprocessAction::TerminalPlot => "terminal_plot",
                PostprocessAction::WriteReport { .. } => "write_report",
            };
            let status = match entry.status {
                PostprocessStatus::Done => "done",
                PostprocessStatus::Skipped => "skipped",
                PostprocessStatus::Failed => "failed",
            };
            HashMap::from([
                ("action".to_owned(), action.to_owned()),
                ("status".to_owned(), status.to_owned()),
                (
                    "path".to_owned(),
                    entry
                        .path
                        .map(|path| path.display().to_string())
                        .unwrap_or_default(),
                ),
                ("message".to_owned(), entry.message.unwrap_or_default()),
            ])
        })
        .collect::<Vec<_>>();
    root.set_item("entries", entries)?;
    Ok(root)
}

#[allow(clippy::too_many_arguments)]
fn telemetry_snapshot_dict<'py>(
    py: Python<'py>,
    solver: &str,
    mode: &str,
    counters: HashMap<String, u64>,
    timings_ms: HashMap<String, f64>,
    report: String,
    aot_metadata: HashMap<String, String>,
    aot_counters: HashMap<String, u64>,
    aot_timings_ms: HashMap<String, f64>,
    artifact_keys: Vec<String>,
) -> PyResult<Bound<'py, PyDict>> {
    let root = PyDict::new(py);
    let native = PyDict::new(py);
    let aot = PyDict::new(py);
    root.set_item("solver", solver)?;
    root.set_item("mode", mode)?;
    native.set_item("counters", counters)?;
    native.set_item("timings_ms", timings_ms)?;
    native.set_item("report", report)?;
    aot.set_item("metadata", aot_metadata)?;
    aot.set_item("counters", aot_counters)?;
    aot.set_item("timings_ms", aot_timings_ms)?;
    aot.set_item("artifact_keys", artifact_keys)?;
    native.set_item("aot", aot)?;
    root.set_item(solver, native)?;
    Ok(root)
}

fn map_prepare_error(error: rustedscithe_core::numerical::StiffAPI::StiffApiError) -> PyErr {
    use rustedscithe_core::numerical::StiffAPI::StiffApiError;
    match error {
        StiffApiError::Bdf(error) => {
            crate::errors::annotate(map_bdf_error(error), "prepare", "bdf", "stiff_api:prepare")
        }
        StiffApiError::Radau(error) => crate::errors::annotate(
            map_radau_error(error),
            "prepare",
            "radau",
            "stiff_api:prepare",
        ),
        StiffApiError::BackwardEuler(error) => crate::errors::annotate(
            map_be_error(error),
            "prepare",
            "backward_euler",
            "stiff_api:prepare",
        ),
        StiffApiError::Lsode2(error) => crate::errors::annotate(
            map_lsode2_error(error),
            "prepare",
            "lsode2",
            "stiff_api:prepare",
        ),
        StiffApiError::ConflictingPresetAxis(axis) => crate::errors::annotate_native(
            InvalidConfigurationError::new_err(format!(
                "multiple StiffAPI presets configure the {axis} axis"
            )),
            "configuration",
            "stiff_api",
            "preset",
            "ConflictingPresetAxis",
        ),
        StiffApiError::UnsupportedCapability { solver, capability } => {
            crate::errors::annotate_native(
                UnsupportedOptionError::new_err(format!(
                    "{solver:?} does not support this StiffAPI request: {capability}"
                )),
                "configuration",
                solver_name(solver),
                "stiff_api:preset",
                "UnsupportedCapability",
            )
        }
        StiffApiError::DimensionMismatch {
            solver,
            expected,
            actual,
        } => crate::errors::annotate_native(
            InvalidConfigurationError::new_err(format!(
                "{solver:?} request has state dimension {actual}; preset requires {expected}"
            )),
            "configuration",
            solver_name(solver),
            "stiff_api:preset",
            "DimensionMismatch",
        ),
    }
}

fn map_solve_error(error: StiffSolveError) -> PyErr {
    match error {
        StiffSolveError::Bdf(error) => map_bdf_error(error),
        StiffSolveError::Radau(error) => map_radau_error(error),
        StiffSolveError::BackwardEuler(error) => map_be_error(error),
        StiffSolveError::Lsode2(error) => map_lsode2_error(error),
    }
}

fn map_restart_error(error: StiffRestartError) -> PyErr {
    match error {
        StiffRestartError::InvalidInterval { solver } => crate::errors::annotate_native(
            InvalidConfigurationError::new_err(format!(
                "{solver:?} restart interval must be finite and nonzero"
            )),
            "restart",
            solver_name(solver),
            "stiff_api:restart",
            "InvalidInterval",
        ),
        StiffRestartError::InvalidInitialState { solver } => crate::errors::annotate_native(
            InvalidConfigurationError::new_err(format!(
                "{solver:?} restart state must be nonempty and finite"
            )),
            "restart",
            solver_name(solver),
            "stiff_api:restart",
            "InvalidInitialState",
        ),
        StiffRestartError::InitialStateDimension {
            solver,
            expected,
            actual,
        } => crate::errors::annotate_native(
            InvalidConfigurationError::new_err(format!(
                "{solver:?} restart state has dimension {actual}; expected {expected}"
            )),
            "restart",
            solver_name(solver),
            "stiff_api:restart",
            "InitialStateDimension",
        ),
        StiffRestartError::Bdf(error) => map_bdf_error(error),
        StiffRestartError::Radau(error) => map_radau_error(error),
        StiffRestartError::BackwardEuler(error) => map_be_error(error),
        StiffRestartError::Lsode2(error) => map_lsode2_error(error),
    }
}

fn map_trajectory_error(error: StiffPostprocessError) -> PyErr {
    let (error, solver, route, native_kind) = match error {
        StiffPostprocessError::ResultUnavailable { solver } => (
            InvalidConfigurationError::new_err(format!(
                "{solver:?} trajectory is not available before a completed solve"
            )),
            solver_name(solver),
            "stiff_api:trajectory",
            "ResultUnavailable",
        ),
        StiffPostprocessError::RadauSolutionRequired => (
            InvalidConfigurationError::new_err(
                "Radau trajectory data belongs to the RadauSolution returned by solve",
            ),
            "radau",
            "stiff_api:trajectory",
            "RadauSolutionRequired",
        ),
        StiffPostprocessError::RadauSampling(error) => (
            crate::errors::SolverError::new_err(format!("Radau output sampling failed: {error}")),
            "radau",
            "stiff_api:radau_sampling",
            "RadauSampling",
        ),
        StiffPostprocessError::InvalidDataset(error) => (
            InvalidConfigurationError::new_err(format!(
                "invalid native trajectory dataset: {error}"
            )),
            "stiff_api",
            "stiff_api:dataset",
            "InvalidDataset",
        ),
        StiffPostprocessError::PlanExecution { solver, source } => (
            crate::errors::SolverError::new_err(format!(
                "{solver:?} postprocessing plan failed: {source}"
            )),
            solver_name(solver),
            "stiff_api:postprocess_plan",
            "PostprocessPlanExecution",
        ),
    };
    crate::errors::annotate_native(error, "postprocess", solver, route, native_kind)
}

#[pymethods]
impl PyStiffPreset {
    /// Return the native catalog of named profiles and their documented scope.
    #[staticmethod]
    fn catalog() -> Vec<HashMap<String, String>> {
        StiffPreset::catalog()
            .iter()
            .map(|entry| {
                HashMap::from([
                    ("name".to_owned(), entry.name.to_owned()),
                    ("purpose".to_owned(), entry.purpose.to_owned()),
                    ("fields_changed".to_owned(), entry.fields_changed.join(",")),
                    (
                        "tools_if_aot_build_needed".to_owned(),
                        entry.tools_if_aot_build_needed.join(","),
                    ),
                    (
                        "requires_output_directory".to_owned(),
                        entry.requires_output_directory.to_string(),
                    ),
                ])
            })
            .collect()
    }

    /// ExprLegacy + Dense profile; all solver-specific defaults are retained.
    #[staticmethod]
    fn expr_dense() -> Self {
        Self::new(StiffPreset::expr_dense())
    }

    /// AtomViewNative + Dense profile; all solver-specific defaults are retained.
    #[staticmethod]
    fn atom_dense() -> Self {
        Self::new(StiffPreset::atom_dense())
    }

    /// ExprLegacy + Sparse where supported; unsupported routes fail natively.
    #[staticmethod]
    fn expr_sparse() -> Self {
        Self::new(StiffPreset::expr_sparse())
    }

    /// AtomViewNative + Sparse where supported; unsupported routes fail natively.
    #[staticmethod]
    fn atom_sparse() -> Self {
        Self::new(StiffPreset::atom_sparse())
    }

    /// ExprLegacy + symbolically inferred Banded layout where supported.
    #[staticmethod]
    fn expr_banded_auto() -> Self {
        Self::new(StiffPreset::expr_banded_auto())
    }

    /// AtomViewNative + symbolically inferred Banded layout where supported.
    #[staticmethod]
    fn atom_banded_auto() -> Self {
        Self::new(StiffPreset::atom_banded_auto())
    }

    /// Require a one-state problem and use ExprLegacy + Dense storage.
    #[staticmethod]
    fn solve_1d() -> Self {
        Self::new(StiffPreset::solve_1d())
    }

    /// Require native Lambdify execution without changing other route axes.
    #[staticmethod]
    fn lambdify() -> Self {
        Self::new(StiffPreset::lambdify())
    }

    /// Configure a release C+tcc AOT build in the supplied artifact directory.
    #[staticmethod]
    fn aot_tcc(output_parent_dir: String) -> Self {
        Self::new(StiffPreset::aot_tcc(output_parent_dir))
    }

    /// Configure a release C+GCC AOT build in the supplied artifact directory.
    #[staticmethod]
    fn aot_gcc(output_parent_dir: String) -> Self {
        Self::new(StiffPreset::aot_gcc(output_parent_dir))
    }

    /// Configure a release Zig AOT build in the supplied artifact directory.
    #[staticmethod]
    fn aot_zig(output_parent_dir: String) -> Self {
        Self::new(StiffPreset::aot_zig(output_parent_dir))
    }

    /// Configure a release Rust-codegen AOT build in the supplied directory.
    #[staticmethod]
    fn aot_rust(output_parent_dir: String) -> Self {
        Self::new(StiffPreset::aot_rust(output_parent_dir))
    }

    /// Add a frontend override, using the native StiffAPI's accepted names.
    fn with_frontend(&self, frontend: &str) -> PyResult<Self> {
        let frontend = match normalize(frontend).as_str() {
            "expr" | "expr_legacy" => StiffFrontend::ExprLegacy,
            "atom" | "atom_view" | "atom_native" => StiffFrontend::AtomViewNative,
            _ => {
                return Err(UnsupportedOptionError::new_err(
                    "frontend must be 'expr_legacy' or 'atom_native'",
                ));
            }
        };
        Ok(Self::new(self.inner.clone().with_frontend(frontend)))
    }

    /// Add a linear-storage override; banded bandwidth remains auto-inferred.
    fn with_matrix_layout(&self, layout: &str) -> PyResult<Self> {
        let layout = match normalize(layout).as_str() {
            "dense" => StiffMatrixLayout::Dense,
            "sparse" => StiffMatrixLayout::Sparse,
            "banded" | "banded_auto" => StiffMatrixLayout::BandedAuto,
            _ => {
                return Err(UnsupportedOptionError::new_err(
                    "matrix layout must be 'dense', 'sparse' or 'banded_auto'",
                ));
            }
        };
        Ok(Self::new(self.inner.clone().with_matrix_layout(layout)))
    }

    /// Add a supported callback scheduling policy.
    #[pyo3(signature=(policy, min_work=0))]
    fn with_callback_policy(&self, policy: &str, min_work: usize) -> PyResult<Self> {
        let policy = match normalize(policy).as_str() {
            "sequential" => StiffCallbackPolicy::Sequential,
            "parallel" => StiffCallbackPolicy::Parallel { min_work },
            "auto" => StiffCallbackPolicy::Auto { min_work },
            _ => {
                return Err(UnsupportedOptionError::new_err(
                    "callback policy must be 'sequential', 'parallel' or 'auto'",
                ));
            }
        };
        Ok(Self::new(self.inner.clone().with_callback_policy(policy)))
    }

    /// Add the native telemetry mode (`off`, `counters`, `timings`, `detailed`).
    fn with_telemetry(&self, mode: &str) -> PyResult<Self> {
        let mode = match normalize(mode).as_str() {
            "off" => SolverTelemetryMode::Off,
            "counters" => SolverTelemetryMode::Counters,
            "timings" => SolverTelemetryMode::Timings,
            "detailed" => SolverTelemetryMode::Detailed,
            _ => {
                return Err(UnsupportedOptionError::new_err(
                    "telemetry must be 'off', 'counters', 'timings' or 'detailed'",
                ));
            }
        };
        Ok(Self::new(self.inner.clone().with_telemetry(mode)))
    }

    /// Add a Rust `log` level; non-off levels use the native file logger.
    fn with_logging(&self, level: &str) -> PyResult<Self> {
        let level = match normalize(level).as_str() {
            "off" => SolverLogLevel::Off,
            "error" => SolverLogLevel::Error,
            "warn" | "warning" => SolverLogLevel::Warn,
            "info" => SolverLogLevel::Info,
            "debug" => SolverLogLevel::Debug,
            _ => {
                return Err(UnsupportedOptionError::new_err(
                    "logging must be 'off', 'error', 'warn', 'info' or 'debug'",
                ));
            }
        };
        Ok(Self::new(self.inner.clone().with_logging(level)))
    }

    /// Combine profiles only when they configure different policy axes.
    fn compose(&self, other: &PyStiffPreset) -> PyResult<Self> {
        self.inner
            .clone()
            .compose(other.inner.clone())
            .map(Self::new)
            .map_err(|error| InvalidConfigurationError::new_err(error.to_string()))
    }

    /// Describe the native overrides and solver-family applicability.
    fn describe(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let description = self.inner.describe();
        let result = PyDict::new(py);
        result.set_item("fields_changed", description.fields_changed)?;
        result.set_item(
            "tools_if_aot_build_needed",
            description.tools_if_aot_build_needed,
        )?;
        result.set_item(
            "required_state_dimension",
            description.required_state_dimension,
        )?;
        let solvers = description
            .solvers
            .into_iter()
            .map(|entry| {
                let applicability = match entry.applicability {
                    StiffPresetApplicability::Supported => "supported",
                    StiffPresetApplicability::Conditional => "conditional",
                    StiffPresetApplicability::Unsupported => "unsupported",
                };
                HashMap::from([
                    ("solver".to_owned(), solver_name(entry.solver).to_owned()),
                    ("applicability".to_owned(), applicability.to_owned()),
                    (
                        "constraint".to_owned(),
                        entry.constraint.unwrap_or_default().to_owned(),
                    ),
                ])
            })
            .collect::<Vec<_>>();
        result.set_item("solvers", solvers)?;
        Ok(result.into())
    }

    fn __repr__(&self) -> String {
        let fields = self.inner.describe().fields_changed.join(", ");
        format!("StiffPreset(fields=[{fields}])")
    }
}

impl PyStiffPreset {
    fn new(inner: StiffPreset) -> Self {
        Self { inner }
    }
}

fn normalize(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace(['-', ' '], "_")
}

fn solver_name(solver: StiffSolverKind) -> &'static str {
    match solver {
        StiffSolverKind::Bdf => "bdf",
        StiffSolverKind::Radau => "radau",
        StiffSolverKind::BackwardEuler => "backward_euler",
        StiffSolverKind::Lsode2 => "lsode2",
    }
}
