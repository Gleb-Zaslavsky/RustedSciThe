//! Python adapter for the native Gavin symbolic fitting controller.
//!
//! Gavin's native builder prepares and solves in one owned operation. The
//! Python class keeps that lifecycle explicit instead of pretending that the
//! current core builder supports prepared rebinds. Result statistics are
//! detached before the native fit object is released.

use std::collections::HashMap;

use nalgebra::DMatrix;
use pyo3::prelude::*;

use rustedscithe_core::numerical::Nonlinear_systems::symbolic::{
    LambdifyExecutionPolicy, PreparationTelemetryMode, SymbolicLambdifyFrontend,
};
use rustedscithe_core::numerical::optimization::gavin_fitting::GavinSymbolicFitting;
use rustedscithe_core::numerical::optimization::lm_gavin::{
    GavinLmError, GavinLmResult, GavinLmTelemetry, GavinLmUpdatePolicy,
};

use crate::errors::{
    InvalidConfigurationError, InvalidExpressionError, PreparationError, SolverError,
    annotate_native, catch_panic, native_variant,
};
use crate::expr::PyExpr;

fn parse_frontend(value: &str) -> PyResult<SymbolicLambdifyFrontend> {
    match value
        .trim()
        .to_ascii_lowercase()
        .replace(['-', ' '], "_")
        .as_str()
    {
        "expr" | "expr_legacy" | "legacy" => Ok(SymbolicLambdifyFrontend::ExprLegacy),
        "atom" | "atom_native" | "atom_view" => Ok(SymbolicLambdifyFrontend::AtomViewNative),
        other => Err(InvalidConfigurationError::new_err(format!(
            "unsupported Gavin frontend '{other}'; expected expr_legacy or atom_native"
        ))),
    }
}

fn parse_execution_policy(value: &str, min_work: usize) -> PyResult<LambdifyExecutionPolicy> {
    match value.trim().to_ascii_lowercase().as_str() {
        "sequential" | "seq" => Ok(LambdifyExecutionPolicy::Sequential),
        "parallel" => {
            if min_work == 0 {
                return Err(InvalidConfigurationError::new_err(
                    "min_work must be greater than zero for parallel execution",
                ));
            }
            Ok(LambdifyExecutionPolicy::Parallel { min_work })
        }
        "auto" => Err(InvalidConfigurationError::new_err(
            "Gavin supports sequential or parallel execution; auto is not a native Gavin policy",
        )),
        other => Err(InvalidConfigurationError::new_err(format!(
            "unsupported Gavin execution_policy '{other}'"
        ))),
    }
}

fn parse_update_policy(value: &str) -> PyResult<GavinLmUpdatePolicy> {
    match value
        .trim()
        .to_ascii_lowercase()
        .replace(['-', ' '], "_")
        .as_str()
    {
        "marquardt" => Ok(GavinLmUpdatePolicy::Marquardt),
        "quadratic" => Ok(GavinLmUpdatePolicy::Quadratic),
        "nielsen" => Ok(GavinLmUpdatePolicy::Nielsen),
        other => Err(InvalidConfigurationError::new_err(format!(
            "unsupported Gavin update_policy '{other}'"
        ))),
    }
}

fn map_error(error: GavinLmError) -> PyErr {
    let native_kind = native_variant(&error);
    let message = error.to_string();
    let mapped = if matches!(
        &error,
        GavinLmError::SymbolicPreparation(_) | GavinLmError::ObjectiveCallback(_)
    ) {
        PreparationError::new_err(message)
    } else if matches!(&error, GavinLmError::SymbolicParse { .. }) {
        InvalidExpressionError::new_err(message)
    } else if matches!(
        &error,
        GavinLmError::InvalidFitConfiguration { .. }
            | GavinLmError::NoParameters
            | GavinLmError::ParameterLengthMismatch { .. }
            | GavinLmError::EmptyObservations
            | GavinLmError::ModelTargetLengthMismatch { .. }
            | GavinLmError::SymbolicDataLengthMismatch { .. }
            | GavinLmError::InvalidOption { .. }
            | GavinLmError::ZeroIterationBudget
            | GavinLmError::InvalidWeight { .. }
            | GavinLmError::WeightLengthMismatch { .. }
            | GavinLmError::AllWeightsZero
            | GavinLmError::NonFiniteInput { .. }
    ) {
        InvalidConfigurationError::new_err(message)
    } else {
        SolverError::new_err(message)
    };
    annotate_native(mapped, "solve", "gavin", "lambdify", &native_kind)
}

fn matrix_rows(matrix: &DMatrix<f64>) -> Vec<Vec<f64>> {
    matrix
        .row_iter()
        .map(|row| row.iter().copied().collect())
        .collect()
}

fn telemetry_counters(telemetry: &GavinLmTelemetry) -> HashMap<String, u64> {
    HashMap::from([
        ("iterations".to_owned(), telemetry.iterations as u64),
        (
            "objective_evaluations".to_owned(),
            telemetry.objective_evaluations as u64,
        ),
        (
            "finite_difference_jacobians".to_owned(),
            telemetry.finite_difference_jacobians as u64,
        ),
        (
            "analytic_jacobian_evaluations".to_owned(),
            telemetry.analytic_jacobian_evaluations as u64,
        ),
        (
            "broyden_updates".to_owned(),
            telemetry.broyden_updates as u64,
        ),
        ("accepted_steps".to_owned(), telemetry.accepted_steps as u64),
        ("rejected_steps".to_owned(), telemetry.rejected_steps as u64),
        ("linear_solves".to_owned(), telemetry.linear_solves as u64),
    ])
}

fn duration_ms(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn telemetry_timings(telemetry: &GavinLmTelemetry) -> HashMap<String, f64> {
    HashMap::from([
        (
            "objective_ms".to_owned(),
            duration_ms(telemetry.objective_time),
        ),
        (
            "linearization_ms".to_owned(),
            duration_ms(telemetry.linearization_time),
        ),
        (
            "analytic_jacobian_ms".to_owned(),
            duration_ms(telemetry.analytic_jacobian_time),
        ),
        (
            "linear_solve_ms".to_owned(),
            duration_ms(telemetry.linear_solve_time),
        ),
        ("total_ms".to_owned(), duration_ms(telemetry.total_time)),
    ])
}

/// Detached result and diagnostics from one Gavin symbolic fit.
///
/// Parameter vectors and standard errors follow the solver's declared parameter
/// order. Covariance-derived uncertainty fields are copied into Python-owned lists.
/// Solver telemetry is empty when collection was disabled; preparation diagnostics
/// are reported separately because symbolic setup and numerical fitting are distinct
/// lifecycle stages.
#[pyclass(name = "GavinFittingSolution", module = "rustedscithe")]
pub struct GavinFittingSolution {
    parameters: Vec<f64>,
    reduced_chi_square: f64,
    parameter_standard_errors: Vec<f64>,
    fitted_value_standard_errors: Vec<f64>,
    parameter_correlation: Vec<Vec<f64>>,
    r_squared: Option<f64>,
    sum_squared_residuals: f64,
    termination: String,
    preparation: String,
    telemetry: Option<GavinLmTelemetry>,
}

impl GavinFittingSolution {
    fn from_native(
        result: rustedscithe_core::numerical::optimization::gavin_fitting::GavinSymbolicFitResult,
    ) -> Self {
        let fit: GavinLmResult = result.fit;
        Self {
            parameters: fit.parameters.iter().copied().collect(),
            reduced_chi_square: fit.reduced_chi_square,
            parameter_standard_errors: fit.parameter_standard_errors.iter().copied().collect(),
            fitted_value_standard_errors: fit
                .fitted_value_standard_errors
                .iter()
                .copied()
                .collect(),
            parameter_correlation: matrix_rows(&fit.parameter_correlation),
            r_squared: fit.r_squared,
            sum_squared_residuals: fit.sum_squared_residuals,
            termination: format!("{:?}", fit.termination),
            preparation: format!("{:?}", result.preparation),
            telemetry: result.telemetry,
        }
    }
}

#[pymethods]
impl GavinFittingSolution {
    /// Return fitted parameter values in declared unknown order.
    pub fn parameters(&self) -> Vec<f64> {
        self.parameters.clone()
    }
    /// Return estimated standard errors for fitted parameters.
    pub fn parameter_standard_errors(&self) -> Vec<f64> {
        self.parameter_standard_errors.clone()
    }
    /// Return propagated standard errors for fitted model values at the input samples.
    pub fn fitted_value_standard_errors(&self) -> Vec<f64> {
        self.fitted_value_standard_errors.clone()
    }
    /// Return the parameter correlation matrix as Python-owned nested lists.
    pub fn parameter_correlation(&self) -> Vec<Vec<f64>> {
        self.parameter_correlation.clone()
    }
    #[getter]
    /// Reduced chi-square statistic for the weighted fit.
    pub fn reduced_chi_square(&self) -> f64 {
        self.reduced_chi_square
    }
    #[getter]
    /// Coefficient of determination, or `None` when it is undefined for the data.
    pub fn r_squared(&self) -> Option<f64> {
        self.r_squared
    }
    #[getter]
    /// Sum of squared residuals from the fitted model.
    pub fn sum_squared_residuals(&self) -> f64 {
        self.sum_squared_residuals
    }
    #[getter]
    /// Native optimizer termination label.
    pub fn termination(&self) -> &str {
        &self.termination
    }
    /// Return a description of symbolic preparation performed for this fit.
    pub fn preparation_report(&self) -> &str {
        &self.preparation
    }
    /// Return discrete numerical-solver counters, or an empty map when disabled.
    pub fn telemetry_counters(&self) -> HashMap<String, u64> {
        self.telemetry
            .as_ref()
            .map(telemetry_counters)
            .unwrap_or_default()
    }
    /// Return numerical stage timings in milliseconds, or an empty map when disabled.
    pub fn telemetry_timings_ms(&self) -> HashMap<String, f64> {
        self.telemetry
            .as_ref()
            .map(telemetry_timings)
            .unwrap_or_default()
    }
    /// Return a compact summary of termination, residual quality and telemetry mode.
    pub fn diagnostics(&self) -> HashMap<String, String> {
        HashMap::from([
            ("termination".to_owned(), self.termination.clone()),
            (
                "sum_squared_residuals".to_owned(),
                self.sum_squared_residuals.to_string(),
            ),
            (
                "telemetry".to_owned(),
                if self.telemetry.is_some() {
                    "enabled"
                } else {
                    "off"
                }
                .to_owned(),
            ),
        ])
    }
}

/// One-shot adapter for the native Gavin symbolic-fitting builder.
///
/// Supply the model residual equations, sample values, observations, sample-variable
/// name, fitted unknowns and initial guess. Optional weights and positivity constraints
/// are applied by the native fit. The default frontend is ExprLegacy; AtomView and
/// sequential/parallel/automatic execution are selectable independently.
///
/// Calling `solve()` consumes the builder exactly once. Use the reusable
/// `FittingSolver` instead when the same prepared model must be solved repeatedly for
/// changed fixed parameters. Logging and numerical/preparation telemetry are opt-in.
#[pyclass(name = "GavinFittingSolver", module = "rustedscithe")]
pub struct GavinFittingSolver {
    builder: Option<GavinSymbolicFitting>,
}

#[pymethods]
impl GavinFittingSolver {
    #[new]
    #[pyo3(signature=(equations, x_data, observations, sample_parameter, unknowns, initial_guess, weights=None, frontend="expr_legacy", execution_policy="sequential", min_work=1, positive_parameters=None, max_iterations=1000, gradient_tolerance=1e-3, parameter_step_tolerance=1e-3, gain_ratio_threshold=1e-1, update_policy="marquardt", telemetry=false, logging=false, preparation_telemetry="off"))]
    /// Configure one Gavin symbolic fit.
    ///
    /// `equations` define the model residual expressions; `x_data` and `observations`
    /// provide matched sample vectors, and `sample_parameter` names the sample variable
    /// in those expressions. `unknowns` and `initial_guess` have matching order.
    /// Optional `weights` and `positive_parameters` constrain the numerical fit.
    /// `frontend` selects `expr_legacy` (default) or `atom_native`; `execution_policy`
    /// selects sequential, parallel, or automatic callback evaluation. Numerical
    /// controls configure iteration, gradient, parameter-step and gain-ratio stopping.
    /// `update_policy` accepts `marquardt`, `quadratic`, or `nielsen`. Numerical
    /// `telemetry` is a boolean: when enabled, Gavin collects its complete native
    /// snapshot (counters and timings together). Logging and preparation telemetry
    /// are separate opt-in controls and disabled by default.
    pub fn new(
        equations: Vec<PyExpr>,
        x_data: Vec<f64>,
        observations: Vec<f64>,
        sample_parameter: String,
        unknowns: Vec<String>,
        initial_guess: Vec<f64>,
        weights: Option<Vec<f64>>,
        frontend: &str,
        execution_policy: &str,
        min_work: usize,
        positive_parameters: Option<Vec<String>>,
        max_iterations: usize,
        gradient_tolerance: f64,
        parameter_step_tolerance: f64,
        gain_ratio_threshold: f64,
        update_policy: &str,
        telemetry: bool,
        logging: bool,
        preparation_telemetry: &str,
    ) -> PyResult<Self> {
        let frontend = parse_frontend(frontend)?;
        let execution_policy = parse_execution_policy(execution_policy, min_work)?;
        let update_policy = parse_update_policy(update_policy)?;
        let preparation_telemetry = match preparation_telemetry.trim().to_ascii_lowercase().as_str()
        {
            "detailed" | "timings" | "on" => PreparationTelemetryMode::Collect,
            "off" | "none" => PreparationTelemetryMode::Disabled,
            other => {
                return Err(InvalidConfigurationError::new_err(format!(
                    "unsupported Gavin preparation_telemetry '{other}'"
                )));
            }
        };
        let mut builder = GavinSymbolicFitting::new()
            .with_equations(equations.into_iter().map(|expr| expr.inner).collect())
            .with_data(x_data, observations)
            .with_sample_parameter(sample_parameter)
            .with_unknowns(unknowns)
            .with_initial_guess(initial_guess)
            .with_frontend(frontend)
            .with_lambdify_execution_policy(execution_policy)
            .with_solver_controls(
                max_iterations,
                gradient_tolerance,
                parameter_step_tolerance,
                gain_ratio_threshold,
            )
            .with_update_policy(update_policy)
            .with_telemetry(if telemetry {
                rustedscithe_core::numerical::SolverTelemetryMode::Detailed
            } else {
                rustedscithe_core::numerical::SolverTelemetryMode::Off
            })
            .with_logging(logging)
            .with_preparation_telemetry(preparation_telemetry);
        if let Some(weights) = weights {
            builder = builder.with_weights(weights);
        }
        if let Some(names) = positive_parameters {
            builder = builder.with_positive_parameters(names);
        }
        Ok(Self {
            builder: Some(builder),
        })
    }

    /// Run the fit and consume this one-shot builder.
    pub fn solve(&mut self) -> PyResult<GavinFittingSolution> {
        catch_panic("solve", "gavin", "lambdify", || {
            let builder = self.builder.take().ok_or_else(|| {
                SolverError::new_err(
                    "GavinFittingSolver is one-shot; construct a new solver for another fit",
                )
            })?;
            builder
                .try_solve()
                .map(GavinFittingSolution::from_native)
                .map_err(map_error)
        })
    }

    /// Return whether `solve()` has consumed the builder.
    pub fn is_solved(&self) -> bool {
        self.builder.is_none()
    }
}
