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
use rustedscithe_core::numerical::SolverTelemetryMode;
use rustedscithe_core::numerical::optimization::lm_gavin::GavinLmUpdatePolicy;
use rustedscithe_core::numerical::optimization::sym_fitting::FittingObservationPolicy;
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
        "timings" => Ok(LeastSquaresTelemetryMode::Timings),
        "detailed" => Ok(LeastSquaresTelemetryMode::Detailed),
        other => Err(InvalidConfigurationError::new_err(format!(
            "unsupported UniversalFitting telemetry '{other}'"
        ))),
    }
}

fn to_solver_telemetry_mode(mode: LeastSquaresTelemetryMode) -> SolverTelemetryMode {
    match mode {
        LeastSquaresTelemetryMode::Off => SolverTelemetryMode::Off,
        LeastSquaresTelemetryMode::Counters => SolverTelemetryMode::Counters,
        LeastSquaresTelemetryMode::Timings => SolverTelemetryMode::Timings,
        LeastSquaresTelemetryMode::Detailed => SolverTelemetryMode::Detailed,
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
        UniversalFittingError::Fitting(source) => {
            let (mapped, native_kind) = match source {
                rustedscithe_core::numerical::optimization::sym_fitting::FittingError::Preparation(error)
                | rustedscithe_core::numerical::optimization::sym_fitting::FittingError::Solver(error) => (
                    crate::nonlinear::map_least_squares_error(error.clone()),
                    native_variant(error),
                ),
                rustedscithe_core::numerical::optimization::sym_fitting::FittingError::DidNotConverge { .. } => (
                    SolverError::new_err(source.to_string()),
                    "DidNotConverge".to_owned(),
                ),
                rustedscithe_core::numerical::optimization::sym_fitting::FittingError::InvalidInput { .. } => (
                    InvalidConfigurationError::new_err(source.to_string()),
                    "InvalidInput".to_owned(),
                ),
                rustedscithe_core::numerical::optimization::sym_fitting::FittingError::EquationParse(_) => (
                    InvalidExpressionError::new_err(source.to_string()),
                    "EquationParse".to_owned(),
                ),
                rustedscithe_core::numerical::optimization::sym_fitting::FittingError::DimensionMismatch { .. } => (
                    InvalidConfigurationError::new_err(source.to_string()),
                    "DimensionMismatch".to_owned(),
                ),
                rustedscithe_core::numerical::optimization::sym_fitting::FittingError::NonFiniteInput { .. } => (
                    InvalidConfigurationError::new_err(source.to_string()),
                    "NonFiniteInput".to_owned(),
                ),
                rustedscithe_core::numerical::optimization::sym_fitting::FittingError::NoUnknowns => (
                    InvalidConfigurationError::new_err(source.to_string()),
                    "NoUnknowns".to_owned(),
                ),
            };
            return annotate_native(mapped, "build", "universal_fitting", "native", &native_kind);
        }
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
        UniversalFittingError::Fitting(_) => unreachable!("handled before common mapping"),
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
/// when available. Native counters and timings are detached with the result;
/// route-specific covariance and fit diagnostics remain on the standalone
/// `GavinFittingSolution`, `VarProSolution`, or `FittingSolution` classes.
#[pyclass(name = "UniversalFittingSolution", module = "rustedscithe")]
pub struct PyUniversalFittingSolution {
    method: String,
    solution: Option<HashMap<String, f64>>,
    r_squared: Option<f64>,
    telemetry_mode: SolverTelemetryMode,
    counters: HashMap<String, u64>,
    timings_ms: HashMap<String, f64>,
    attempt_counters: Vec<HashMap<String, u64>>,
    attempt_timings_ms: Vec<HashMap<String, f64>>,
}

impl PyUniversalFittingSolution {
    fn from_native(
        method: Method,
        telemetry_mode: SolverTelemetryMode,
        result: UniversalFittingResult,
    ) -> Self {
        let (counters, timings_ms, attempt_counters, attempt_timings_ms) = match &result {
            UniversalFittingResult::LM(fitting) => fitting.last_statistics().map_or_else(
                || (HashMap::new(), HashMap::new(), Vec::new(), Vec::new()),
                |stats| {
                    (
                        crate::nonlinear::statistics_counters(stats),
                        crate::nonlinear::statistics_timings(stats),
                        crate::nonlinear::statistics_attempt_counters(stats),
                        crate::nonlinear::statistics_attempt_timings_ms(stats),
                    )
                },
            ),
            UniversalFittingResult::VARPRO(fit) => (
                crate::nonlinear::statistics_counters(&fit.minimization_report.statistics),
                crate::nonlinear::statistics_timings(&fit.minimization_report.statistics),
                crate::nonlinear::statistics_attempt_counters(&fit.minimization_report.statistics),
                crate::nonlinear::statistics_attempt_timings_ms(
                    &fit.minimization_report.statistics,
                ),
            ),
            UniversalFittingResult::GAVIN(fit) => fit.telemetry.as_ref().map_or_else(
                || (HashMap::new(), HashMap::new(), Vec::new(), Vec::new()),
                |telemetry| {
                    (
                        if telemetry_mode.collects_counters() {
                            crate::gavin::telemetry_counters(telemetry)
                        } else {
                            HashMap::new()
                        },
                        if telemetry_mode.collects_timings() {
                            crate::gavin::telemetry_timings(telemetry)
                        } else {
                            HashMap::new()
                        },
                        Vec::new(),
                        Vec::new(),
                    )
                },
            ),
        };
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
            telemetry_mode,
            counters,
            timings_ms,
            attempt_counters,
            attempt_timings_ms,
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

    #[getter]
    /// Selected native collection policy: `off`, `counters`, `timings`, or `detailed`.
    pub fn telemetry_mode(&self) -> &'static str {
        match self.telemetry_mode {
            SolverTelemetryMode::Off => "off",
            SolverTelemetryMode::Counters => "counters",
            SolverTelemetryMode::Timings => "timings",
            SolverTelemetryMode::Detailed => "detailed",
        }
    }

    /// Return collected discrete solver counts; empty means counters were disabled.
    pub fn telemetry_counters(&self) -> HashMap<String, u64> {
        self.counters.clone()
    }

    /// Return collected solver stage durations in milliseconds.
    pub fn telemetry_timings_ms(&self) -> HashMap<String, f64> {
        self.timings_ms.clone()
    }

    /// Return per-iteration counters when the selected native route provides them.
    pub fn telemetry_attempt_counters(&self) -> Vec<HashMap<String, u64>> {
        self.attempt_counters.clone()
    }

    /// Return per-iteration timings; empty for routes without native attempt traces.
    pub fn telemetry_attempt_timings_ms(&self) -> Vec<HashMap<String, f64>> {
        self.attempt_timings_ms.clone()
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
    telemetry_mode: LeastSquaresTelemetryMode,
    gavin_telemetry_mode: SolverTelemetryMode,
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
            telemetry_mode: LeastSquaresTelemetryMode::Off,
            gavin_telemetry_mode: SolverTelemetryMode::Off,
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

    /// Select observation-level evaluation for classic LM.
    ///
    /// `auto` uses the measured large-data default. `sequential` avoids the
    /// additional sample-major staging buffers. `parallel` enables explicit
    /// blocks once `min_observations` is reached and requires sequential
    /// callbacks. Gavin and VarPro reject this option through the native
    /// typed unsupported-option error.
    #[pyo3(signature=(policy, min_observations=1))]
    pub fn with_observation_policy(
        &mut self,
        policy: &str,
        min_observations: usize,
    ) -> PyResult<()> {
        let policy = match policy.trim().to_ascii_lowercase().as_str() {
            "auto" => FittingObservationPolicy::Auto,
            "sequential" | "seq" => FittingObservationPolicy::Sequential,
            "parallel" if min_observations > 0 => {
                FittingObservationPolicy::Parallel { min_observations }
            }
            "parallel" => {
                return Err(InvalidConfigurationError::new_err(
                    "min_observations must be positive",
                ));
            }
            other => {
                return Err(InvalidConfigurationError::new_err(format!(
                    "unsupported observation policy '{other}'"
                )));
            }
        };
        self.update(|builder| builder.with_observation_policy(policy))
    }

    /// Configure the native least-squares telemetry level.
    pub fn with_telemetry(&mut self, mode: &str) -> PyResult<()> {
        let mode = parse_telemetry(mode)?;
        if self.method == Method::GAVIN {
            self.gavin_telemetry_mode = to_solver_telemetry_mode(mode);
            let gavin_mode = self.gavin_telemetry_mode;
            self.update(|builder| builder.with_gavin_telemetry(gavin_mode))
        } else {
            self.telemetry_mode = mode;
            self.update(|builder| builder.with_telemetry(mode))
        }
    }

    /// Enable or disable Gavin's complete native telemetry snapshot.
    pub fn with_gavin_telemetry(&mut self, enabled: bool) -> PyResult<()> {
        self.gavin_telemetry_mode = if enabled {
            SolverTelemetryMode::Detailed
        } else {
            SolverTelemetryMode::Off
        };
        let mode = self.gavin_telemetry_mode;
        self.update(|builder| builder.with_gavin_telemetry(mode))
    }

    /// Select one of the shared telemetry channels for the Gavin route.
    pub fn with_gavin_telemetry_mode(&mut self, mode: &str) -> PyResult<()> {
        let mode = to_solver_telemetry_mode(parse_telemetry(mode)?);
        self.gavin_telemetry_mode = mode;
        self.update(|builder| builder.with_gavin_telemetry(mode))
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
        let telemetry_mode = if method == Method::GAVIN {
            self.gavin_telemetry_mode
        } else {
            to_solver_telemetry_mode(self.telemetry_mode)
        };
        let route = self.method();
        let panic_state = self.panic_state.clone();
        panic_state.run("build", "universal_fitting", route, || {
            let builder = self.builder.take().ok_or_else(|| {
                SolverError::new_err("UniversalFitting is one-shot; construct a new builder")
            })?;
            builder
                .build()
                .map(|result| {
                    PyUniversalFittingSolution::from_native(method, telemetry_mode, result)
                })
                .map_err(map_error)
        })
    }

    /// Whether `build()` has consumed this one-shot facade.
    pub fn is_built(&self) -> bool {
        self.builder.is_none()
    }
}
