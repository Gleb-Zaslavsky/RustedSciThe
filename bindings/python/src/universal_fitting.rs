//! Python binding for the native method-selecting `UniversalFitting` facade.
//!
//! This wrapper intentionally delegates configuration and dispatch to the core
//! Rust facade. It does not reproduce method-selection rules in Python.

use std::collections::HashMap;

use pyo3::prelude::*;

use rustedscithe_core::numerical::Nonlinear_systems::least_squares::LeastSquaresTelemetryMode;
use rustedscithe_core::numerical::Nonlinear_systems::symbolic::{
    LambdifyExecutionPolicy, SymbolicLambdifyFrontend,
};
use rustedscithe_core::numerical::optimization::lm_gavin::GavinLmUpdatePolicy;
use rustedscithe_core::numerical::optimization::universal_fitting::{
    Method, UniversalFitting, UniversalFittingError, UniversalFittingResult,
};
use rustedscithe_core::numerical::optimization::varpro::symbolic::SymbolicVarProError;

use crate::errors::{
    ExhaustionError, InvalidConfigurationError, InvalidExpressionError, SolverError,
    SolverPanicState, UnsupportedOptionError, annotate_native, native_variant,
};
use crate::expr::PyExpr;

fn parse_method(value: &str) -> PyResult<Method> {
    match value
        .trim()
        .to_ascii_lowercase()
        .replace(['-', ' '], "_")
        .as_str()
    {
        "lm" | "levenberg_marquardt" => Ok(Method::LM),
        "varpro" | "variable_projection" => Ok(Method::VARPRO),
        "gavin" | "gavin_lm" => Ok(Method::GAVIN),
        other => Err(InvalidConfigurationError::new_err(format!(
            "unsupported UniversalFitting method '{other}'; expected lm, varpro or gavin"
        ))),
    }
}

fn parse_telemetry(value: &str) -> PyResult<LeastSquaresTelemetryMode> {
    match value.trim().to_ascii_lowercase().as_str() {
        "off" | "none" => Ok(LeastSquaresTelemetryMode::Off),
        "counters" => Ok(LeastSquaresTelemetryMode::Counters),
        "detailed" | "timings" => Ok(LeastSquaresTelemetryMode::Detailed),
        other => Err(InvalidConfigurationError::new_err(format!(
            "unsupported UniversalFitting telemetry '{other}'"
        ))),
    }
}

fn parse_update_policy(value: &str) -> PyResult<GavinLmUpdatePolicy> {
    match value.trim().to_ascii_lowercase().as_str() {
        "marquardt" => Ok(GavinLmUpdatePolicy::Marquardt),
        "quadratic" => Ok(GavinLmUpdatePolicy::Quadratic),
        "nielsen" => Ok(GavinLmUpdatePolicy::Nielsen),
        other => Err(InvalidConfigurationError::new_err(format!(
            "unsupported Gavin update policy '{other}'"
        ))),
    }
}

fn map_error(error: UniversalFittingError) -> PyErr {
    match &error {
        UniversalFittingError::VarPro(SymbolicVarProError::SolverFailed {
            source: Some(source),
            ..
        }) => {
            let native_kind = native_variant(source);
            return annotate_native(
                crate::nonlinear::map_least_squares_error(source.clone()),
                "build",
                "universal_fitting",
                "native",
                &native_kind,
            );
        }
        UniversalFittingError::VarPro(SymbolicVarProError::SolverFailed {
            source: None,
            termination,
            exhausted: true,
        }) => {
            return annotate_native(
                ExhaustionError::new_err(format!(
                    "UniversalFitting VarPro exhausted: {termination}"
                )),
                "build",
                "universal_fitting",
                "native",
                "VarPro",
            );
        }
        _ => {}
    }
    let native_kind = native_variant(&error);
    let mapped = match error {
        UniversalFittingError::UnsupportedOption { .. } => {
            UnsupportedOptionError::new_err(error.to_string())
        }
        UniversalFittingError::SolverConfiguration(_) => {
            InvalidConfigurationError::new_err(error.to_string())
        }
        UniversalFittingError::VarPro(_) | UniversalFittingError::Gavin(_) => {
            SolverError::new_err(error.to_string())
        }
    };
    annotate_native(mapped, "build", "universal_fitting", "native", &native_kind)
}

/// Detached common result from native `UniversalFitting` dispatch.
///
/// The universal Rust result guarantees a parameter map and R-squared value
/// when available. Route-specific covariance or detailed solver telemetry
/// remain on `GavinFittingSolution`, `VarProSolution`, or `FittingSolution`.
#[pyclass(name = "UniversalFittingSolution", module = "rustedscithe")]
pub struct PyUniversalFittingSolution {
    method: String,
    solution: Option<HashMap<String, f64>>,
    r_squared: Option<f64>,
}

impl PyUniversalFittingSolution {
    fn from_native(method: Method, result: UniversalFittingResult) -> Self {
        let method = match method {
            Method::LM => "lm",
            Method::VARPRO => "varpro",
            Method::GAVIN => "gavin",
        }
        .to_owned();
        Self {
            method,
            solution: result.solution_map(),
            r_squared: result.r_squared(),
        }
    }
}

#[pymethods]
impl PyUniversalFittingSolution {
    #[getter]
    /// Method selected by the builder (`lm`, `varpro`, or `gavin`).
    pub fn method(&self) -> &str {
        &self.method
    }

    /// Return the fitted named parameters, or `None` when the native result
    /// cannot provide a named solution map.
    pub fn solution_map(&self) -> Option<HashMap<String, f64>> {
        self.solution.clone()
    }

    #[getter]
    /// Coefficient of determination when defined by the supplied observations.
    pub fn r_squared(&self) -> Option<f64> {
        self.r_squared
    }

    /// Return stable common diagnostics without flattening route-specific data.
    pub fn diagnostics(&self) -> HashMap<String, String> {
        HashMap::from([
            ("method".to_owned(), self.method.clone()),
            (
                "solution_available".to_owned(),
                self.solution.is_some().to_string(),
            ),
            (
                "r_squared".to_owned(),
                self.r_squared
                    .map_or_else(|| "undefined".to_owned(), |value| value.to_string()),
            ),
        ])
    }
}

/// Fluent Python adapter over the native Rust `UniversalFitting` builder.
///
/// Shared inputs are forwarded to the selected native method. Use
/// `with_equations` for LM/Gavin residual models; VarPro instead uses
/// `with_basis` and optional invariant equations. Method-specific controls are
/// intentionally explicit. Unsupported combinations are rejected by the
/// native universal facade at `build()`.
#[pyclass(name = "UniversalFitting", module = "rustedscithe")]
pub struct PyUniversalFitting {
    builder: Option<UniversalFitting>,
    method: Method,
    panic_state: SolverPanicState,
}

impl PyUniversalFitting {
    fn update<F>(&mut self, update: F) -> PyResult<()>
    where
        F: FnOnce(UniversalFitting) -> UniversalFitting,
    {
        let builder = self.builder.take().ok_or_else(|| {
            SolverError::new_err("UniversalFitting is consumed; construct a new builder")
        })?;
        self.builder = Some(update(builder));
        Ok(())
    }
}

#[pymethods]
impl PyUniversalFitting {
    #[new]
    #[pyo3(signature=(method="lm"))]
    /// Create a method-selecting fitting builder.
    pub fn new(method: &str) -> PyResult<Self> {
        let method = parse_method(method)?;
        Ok(Self {
            builder: Some(UniversalFitting::new().with_method(method)),
            method,
            panic_state: SolverPanicState::default(),
        })
    }

    #[getter]
    /// Return the selected native fitting route.
    pub fn method(&self) -> &'static str {
        match self.method {
            Method::LM => "lm",
            Method::VARPRO => "varpro",
            Method::GAVIN => "gavin",
        }
    }

    /// Set paired sample locations and observations.
    pub fn with_data(&mut self, x_data: Vec<f64>, y_data: Vec<f64>) -> PyResult<()> {
        self.update(|builder| builder.with_data(x_data, y_data))
    }

    /// Declare fitted nonlinear parameter names in their stable order.
    pub fn with_unknowns(&mut self, names: Vec<String>) -> PyResult<()> {
        self.update(|builder| builder.with_unknowns(names))
    }

    /// Set the initial values corresponding to `with_unknowns`.
    pub fn with_initial_guess(&mut self, values: Vec<f64>) -> PyResult<()> {
        self.update(|builder| builder.with_initial_guess(values))
    }

    /// Set the independent/sample variable name.
    pub fn with_arg(&mut self, name: String) -> PyResult<()> {
        self.update(|builder| builder.with_arg(name))
    }

    /// Set residual equations for LM/Gavin or invariant equations for VarPro.
    pub fn with_equations(&mut self, expressions: Vec<PyExpr>) -> PyResult<()> {
        let expressions = expressions.into_iter().map(|expr| expr.inner).collect();
        self.update(|builder| builder.with_equations(expressions))
    }

    /// Set one symbolic model expression from a string.
    pub fn with_equation_str(&mut self, expression: &str) -> PyResult<()> {
        let expression =
            rustedscithe_core::symbolic::symbolic_engine::Expr::try_parse_expression(expression)
                .map_err(InvalidExpressionError::new_err)?;
        self.update(|builder| builder.with_equation(expression))
    }

    /// Set a polynomial model and select the native LM method.
    pub fn with_polynomial(&mut self, degree: usize, argument: String) -> PyResult<()> {
        self.update(|builder| builder.with_polynomial(degree, argument))
    }

    /// Parse and set equations using the native expression parser.
    pub fn with_equations_str(&mut self, expressions: Vec<String>) -> PyResult<()> {
        let parsed = expressions
            .iter()
            .map(|text| {
                rustedscithe_core::symbolic::symbolic_engine::Expr::try_parse_expression(text)
                    .map_err(InvalidExpressionError::new_err)
            })
            .collect::<PyResult<Vec<_>>>()?;
        self.update(|builder| builder.with_equations(parsed))
    }

    /// Add a VarPro basis expression for one named nonlinear parameter.
    pub fn with_basis(&mut self, name: String, expression: PyExpr) -> PyResult<()> {
        self.update(|builder| builder.with_basis_expr(name, expression.inner))
    }

    /// Parse and add a VarPro basis expression.
    pub fn with_basis_str(&mut self, name: String, expression: &str) -> PyResult<()> {
        let expression =
            rustedscithe_core::symbolic::symbolic_engine::Expr::try_parse_expression(expression)
                .map_err(InvalidExpressionError::new_err)?;
        self.update(|builder| builder.with_basis_expr(name, expression))
    }

    /// Add a VarPro invariant/background expression.
    pub fn with_invariant(&mut self, expression: PyExpr) -> PyResult<()> {
        self.update(|builder| builder.with_equation(expression.inner))
    }

    /// Add a VarPro invariant/background expression from a string.
    pub fn with_invariant_str(&mut self, expression: &str) -> PyResult<()> {
        let expression =
            rustedscithe_core::symbolic::symbolic_engine::Expr::try_parse_expression(expression)
                .map_err(InvalidExpressionError::new_err)?;
        self.update(|builder| builder.with_equation(expression))
    }

    /// Select ExprLegacy or AtomView symbolic preparation.
    pub fn with_frontend(&mut self, frontend: &str) -> PyResult<()> {
        let frontend = match frontend
            .trim()
            .to_ascii_lowercase()
            .replace(['-', ' '], "_")
            .as_str()
        {
            "expr" | "expr_legacy" | "legacy" => SymbolicLambdifyFrontend::ExprLegacy,
            "atom" | "atom_native" | "atom_view" => SymbolicLambdifyFrontend::AtomViewNative,
            other => {
                return Err(InvalidConfigurationError::new_err(format!(
                    "unsupported frontend '{other}'"
                )));
            }
        };
        self.update(|builder| builder.with_frontend(frontend))
    }

    /// Select sequential or thresholded parallel callback evaluation.
    pub fn with_execution_policy(&mut self, policy: &str, min_work: usize) -> PyResult<()> {
        let policy = match policy.trim().to_ascii_lowercase().as_str() {
            "sequential" | "seq" => LambdifyExecutionPolicy::Sequential,
            "parallel" | "auto" if min_work > 0 => LambdifyExecutionPolicy::Parallel { min_work },
            "parallel" | "auto" => {
                return Err(InvalidConfigurationError::new_err(
                    "min_work must be positive",
                ));
            }
            other => {
                return Err(InvalidConfigurationError::new_err(format!(
                    "unsupported execution policy '{other}'"
                )));
            }
        };
        self.update(|builder| builder.with_lambdify_execution_policy(policy))
    }

    /// Configure the native least-squares telemetry level.
    pub fn with_telemetry(&mut self, mode: &str) -> PyResult<()> {
        let mode = parse_telemetry(mode)?;
        if self.method == Method::GAVIN && mode != LeastSquaresTelemetryMode::Off {
            return Err(InvalidConfigurationError::new_err(
                "Configure Gavin telemetry with with_gavin_telemetry(enabled)",
            ));
        }
        self.update(|builder| builder.with_telemetry(mode))
    }

    /// Enable or disable Gavin's complete native telemetry snapshot.
    pub fn with_gavin_telemetry(&mut self, enabled: bool) -> PyResult<()> {
        self.update(|builder| {
            builder.with_gavin_telemetry(if enabled {
                rustedscithe_core::numerical::SolverTelemetryMode::Detailed
            } else {
                rustedscithe_core::numerical::SolverTelemetryMode::Off
            })
        })
    }

    /// Enable or disable the native application's configured logger.
    pub fn with_logging(&mut self, enabled: bool) -> PyResult<()> {
        self.update(|builder| builder.with_logging(enabled))
    }

    /// Configure LM/VarPro parameter-step tolerance.
    pub fn with_tolerance(&mut self, value: f64) -> PyResult<()> {
        self.update(|builder| builder.with_tolerance(value))
    }

    /// Configure LM/VarPro objective tolerance.
    pub fn with_f_tolerance(&mut self, value: f64) -> PyResult<()> {
        self.update(|builder| builder.with_f_tolerance(value))
    }

    /// Configure LM/VarPro gradient tolerance.
    pub fn with_g_tolerance(&mut self, value: f64) -> PyResult<()> {
        self.update(|builder| builder.with_g_tolerance(value))
    }

    /// Set the selected route's iteration budget.
    pub fn with_max_iterations(&mut self, value: usize) -> PyResult<()> {
        self.update(|builder| builder.with_max_iterations(value))
    }

    /// Configure LM/VarPro diagonal scaling.
    pub fn with_scale_diag(&mut self, value: bool) -> PyResult<()> {
        self.update(|builder| builder.with_scale_diag(value))
    }

    /// Select Gavin damping policy (`marquardt`, `quadratic`, or `nielsen`).
    pub fn with_gavin_update_policy(&mut self, value: &str) -> PyResult<()> {
        let policy = parse_update_policy(value)?;
        self.update(|builder| builder.with_gavin_update_policy(policy))
    }

    /// Set Gavin's iteration and convergence controls.
    pub fn with_gavin_solver_controls(
        &mut self,
        max_iterations: usize,
        gradient_tolerance: f64,
        parameter_step_tolerance: f64,
        gain_ratio_threshold: f64,
    ) -> PyResult<()> {
        self.update(|builder| {
            builder.with_gavin_solver_controls(
                max_iterations,
                gradient_tolerance,
                parameter_step_tolerance,
                gain_ratio_threshold,
            )
        })
    }

    /// Set Gavin's per-observation nonnegative weights.
    pub fn with_gavin_weights(&mut self, weights: Vec<f64>) -> PyResult<()> {
        self.update(|builder| builder.with_gavin_weights(weights))
    }

    /// Require selected Gavin parameters to remain strictly positive.
    pub fn with_gavin_positive_parameters(&mut self, names: Vec<String>) -> PyResult<()> {
        self.update(|builder| builder.with_gavin_positive_parameters(names))
    }

    /// Build and solve using the actual Rust `UniversalFitting` dispatcher.
    pub fn build(&mut self) -> PyResult<PyUniversalFittingSolution> {
        let method = self.method;
        let route = self.method();
        let panic_state = self.panic_state.clone();
        panic_state.run("build", "universal_fitting", route, || {
            let builder = self.builder.take().ok_or_else(|| {
                SolverError::new_err("UniversalFitting is one-shot; construct a new builder")
            })?;
            builder
                .build()
                .map(|result| PyUniversalFittingSolution::from_native(method, result))
                .map_err(map_error)
        })
    }

    /// Whether `build()` has consumed this one-shot facade.
    pub fn is_built(&self) -> bool {
        self.builder.is_none()
    }
}
