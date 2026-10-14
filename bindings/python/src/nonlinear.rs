//! Python bindings for the canonical nonlinear-system and least-squares APIs.
//!
//! The wrappers deliberately keep symbolic preparation in the core crate.  A
//! Python object owns a prepared Rust problem and each solve only creates a
//! short-lived parameter-bound view.  This makes repeated solves honest:
//! differentiation and Lambdify preparation are not repeated implicitly.

use std::collections::HashMap;

use nalgebra::DVector;
use pyo3::prelude::*;

use rustedscithe_core::numerical::Nonlinear_systems::engine::{
    DiagnosticsOptions, EngineLogLevel, SolveOptions, SolveStatistics,
};
use rustedscithe_core::numerical::Nonlinear_systems::error::SolveError;
use rustedscithe_core::numerical::Nonlinear_systems::least_squares::{
    LeastSquaresError, LeastSquaresTelemetryMode, SymbolicLeastSquaresSolver,
};
use rustedscithe_core::numerical::Nonlinear_systems::prelude::{
    Bounds, DampedNewtonMethod, DampedNewtonMethodAdvanced, LevenbergMarquardtMethod,
    LevenbergMarquardtMinpack, NielsenLevenbergMarquardtMethod,
    NielsenLevenbergMarquardtMethodAdvanced, NonlinearSolverMethod, PowellDoglegMethod,
    TrustRegionLMMethod, TrustRegionMethod,
};
use rustedscithe_core::numerical::Nonlinear_systems::symbolic::{
    LambdifyExecutionPolicy, PreparationTelemetryMode, PreparedSymbolicNonlinearProblem,
    SymbolicLambdifyFrontend, SymbolicProblemOptions,
};

use crate::errors::{
    InvalidConfigurationError, PreparationError, SolverError, UnsupportedOptionError, annotate,
    catch_panic,
};
use crate::expr::PyExpr;
use crate::system::SymbolicSystem;

fn map_solve_error(error: SolveError) -> PyErr {
    annotate(
        match error {
            SolveError::DimensionMismatch { .. }
            | SolveError::InvalidConfig(_)
            | SolveError::InvalidParameterSchema(_)
            | SolveError::ParameterSchemaMismatch(_)
            | SolveError::InvalidVariableSchema(_)
            | SolveError::UndeclaredSymbol { .. }
            | SolveError::NonFiniteParameterValue { .. }
            | SolveError::InfeasibleInitialGuess { .. }
            | SolveError::NonFiniteInitialGuess { .. } => {
                InvalidConfigurationError::new_err(error.to_string())
            }
            SolveError::CompiledAotRuntimeUnavailable(_)
            | SolveError::CompiledAotArtifactMissing(_)
            | SolveError::CompiledAotArtifactNotBuilt(_)
            | SolveError::AotBuildOutputDirMissing
            | SolveError::AotBuildFailed(_) => PreparationError::new_err(error.to_string()),
            SolveError::ResidualEvaluation(_)
            | SolveError::JacobianEvaluation(_)
            | SolveError::LinearSolveFailure(_)
            | SolveError::SingularJacobian
            | SolveError::NumericalBreakdown(_) => SolverError::new_err(error.to_string()),
        },
        "solve",
        "nonlinear_system",
        "lambdify",
    )
}

fn map_least_squares_error(error: LeastSquaresError) -> PyErr {
    annotate(
        match error {
            LeastSquaresError::InvalidConfiguration { .. }
            | LeastSquaresError::InvalidLogLevel(_)
            | LeastSquaresError::WrongSolverKind
            | LeastSquaresError::EmptyProblem { .. }
            | LeastSquaresError::InvalidProblemShape { .. }
            | LeastSquaresError::DimensionMismatch { .. }
            | LeastSquaresError::NonFiniteValue { .. } => {
                InvalidConfigurationError::new_err(error.to_string())
            }
            LeastSquaresError::Problem(source) => map_solve_error(source),
            LeastSquaresError::CallbackFailed { .. }
            | LeastSquaresError::DomainViolation
            | LeastSquaresError::NumericalBreakdown { .. }
            | LeastSquaresError::MaxEvaluationsReached { .. }
            | LeastSquaresError::MaxIterationsReached { .. } => {
                SolverError::new_err(error.to_string())
            }
        },
        "solve",
        "least_squares",
        "lambdify",
    )
}

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
            "unsupported nonlinear frontend '{other}'; expected expr_legacy or atom_native"
        ))),
    }
}

fn parse_execution_policy(value: &str, min_work: usize) -> PyResult<LambdifyExecutionPolicy> {
    match value.trim().to_ascii_lowercase().as_str() {
        "sequential" | "seq" => Ok(LambdifyExecutionPolicy::Sequential),
        "parallel" | "auto" => {
            if min_work == 0 {
                return Err(InvalidConfigurationError::new_err(
                    "min_work must be greater than zero for parallel or auto execution",
                ));
            }
            Ok(LambdifyExecutionPolicy::Parallel { min_work })
        }
        other => Err(InvalidConfigurationError::new_err(format!(
            "unsupported nonlinear execution_policy '{other}'"
        ))),
    }
}

fn parse_preparation_telemetry(value: &str) -> PreparationTelemetryMode {
    match value.trim().to_ascii_lowercase().as_str() {
        "detailed" | "timings" | "on" => PreparationTelemetryMode::Collect,
        _ => PreparationTelemetryMode::Disabled,
    }
}

fn parse_log_level(value: &str) -> PyResult<Option<EngineLogLevel>> {
    match value.trim().to_ascii_lowercase().as_str() {
        "off" | "none" => Ok(None),
        "debug" => Ok(Some(EngineLogLevel::Debug)),
        "info" => Ok(Some(EngineLogLevel::Info)),
        "warn" | "warning" => Ok(Some(EngineLogLevel::Warn)),
        other => Err(InvalidConfigurationError::new_err(format!(
            "unsupported nonlinear log level '{other}'"
        ))),
    }
}

fn normalized_method_options(
    options: Option<HashMap<String, f64>>,
    flags: Option<HashMap<String, bool>>,
) -> PyResult<(HashMap<String, f64>, HashMap<String, bool>)> {
    let normalize = |key: String| key.trim().to_ascii_lowercase().replace(['-', ' '], "_");
    let mut numeric = HashMap::new();
    for (key, value) in options.unwrap_or_default() {
        if !value.is_finite() {
            return Err(InvalidConfigurationError::new_err(format!(
                "method option '{key}' must be finite"
            )));
        }
        numeric.insert(normalize(key), value);
    }
    let mut boolean = HashMap::new();
    for (key, value) in flags.unwrap_or_default() {
        boolean.insert(normalize(key), value);
    }
    Ok((numeric, boolean))
}

fn take_numeric(options: &mut HashMap<String, f64>, key: &str, default: f64) -> PyResult<f64> {
    let value = options.remove(key).unwrap_or(default);
    if !value.is_finite() {
        return Err(InvalidConfigurationError::new_err(format!(
            "method option '{key}' must be finite"
        )));
    }
    Ok(value)
}

fn take_positive_numeric(
    options: &mut HashMap<String, f64>,
    method: &str,
    key: &str,
    default: f64,
) -> PyResult<f64> {
    let value = take_numeric(options, key, default)?;
    if value <= 0.0 {
        return Err(InvalidConfigurationError::new_err(format!(
            "method option '{key}' for '{method}' must be positive"
        )));
    }
    Ok(value)
}

fn take_usize(
    options: &mut HashMap<String, f64>,
    method: &str,
    key: &str,
    default: usize,
) -> PyResult<usize> {
    let value = take_numeric(options, key, default as f64)?;
    if value < 0.0 || value.fract() != 0.0 || value > usize::MAX as f64 {
        return Err(InvalidConfigurationError::new_err(format!(
            "method option '{key}' for '{method}' must be a non-negative integer"
        )));
    }
    Ok(value as usize)
}

fn take_flag(flags: &mut HashMap<String, bool>, key: &str, default: bool) -> bool {
    flags.remove(key).unwrap_or(default)
}

fn take_optional_numeric(
    options: &mut HashMap<String, f64>,
    key: &str,
    default: Option<f64>,
) -> PyResult<Option<f64>> {
    match options.remove(key) {
        Some(value) if value.is_finite() => Ok(Some(value)),
        Some(_) => Err(InvalidConfigurationError::new_err(format!(
            "method option '{key}' must be finite"
        ))),
        None => Ok(default),
    }
}

fn reject_remaining_method_options(
    method: &str,
    options: HashMap<String, f64>,
    flags: HashMap<String, bool>,
) -> PyResult<()> {
    if let Some(key) = options.keys().next().or_else(|| flags.keys().next()) {
        return Err(UnsupportedOptionError::new_err(format!(
            "method '{method}' does not support option '{key}'"
        )));
    }
    Ok(())
}

fn method_from_name(
    value: &str,
    numeric_options: &HashMap<String, f64>,
    flag_options: &HashMap<String, bool>,
) -> PyResult<NonlinearSolverMethod> {
    let method_name = value.trim().to_ascii_lowercase().replace(['-', ' '], "_");
    let mut options = numeric_options.clone();
    let mut flags = flag_options.clone();
    match method_name.as_str() {
        "newton" => {
            reject_remaining_method_options(&method_name, options, flags)?;
            Ok(NonlinearSolverMethod::Newton(
                rustedscithe_core::numerical::Nonlinear_systems::prelude::NewtonMethod,
            ))
        }
        "damped_newton" | "damped" => {
            let defaults = DampedNewtonMethod::default();
            let method = DampedNewtonMethod {
                max_line_search_steps: take_usize(
                    &mut options,
                    &method_name,
                    "max_line_search_steps",
                    defaults.max_line_search_steps,
                )?,
                sufficient_decrease: take_positive_numeric(
                    &mut options,
                    &method_name,
                    "sufficient_decrease",
                    defaults.sufficient_decrease,
                )?,
                shrink_factor: take_positive_numeric(
                    &mut options,
                    &method_name,
                    "shrink_factor",
                    defaults.shrink_factor,
                )?,
            };
            reject_remaining_method_options(&method_name, options, flags)?;
            Ok(NonlinearSolverMethod::DampedNewton(method))
        }
        "damped_newton_advanced" => {
            let defaults = DampedNewtonMethodAdvanced::default();
            let method = DampedNewtonMethodAdvanced {
                max_damping_iterations: take_usize(
                    &mut options,
                    &method_name,
                    "max_damping_iterations",
                    defaults.max_damping_iterations,
                )?,
                sufficient_decrease: take_positive_numeric(
                    &mut options,
                    &method_name,
                    "sufficient_decrease",
                    defaults.sufficient_decrease,
                )?,
                shrink_factor: take_positive_numeric(
                    &mut options,
                    &method_name,
                    "shrink_factor",
                    defaults.shrink_factor,
                )?,
            };
            reject_remaining_method_options(&method_name, options, flags)?;
            Ok(NonlinearSolverMethod::DampedNewtonAdvanced(method))
        }
        "levenberg_marquardt" | "lm" => {
            let defaults = LevenbergMarquardtMethod::default();
            let method = LevenbergMarquardtMethod {
                lambda_init: take_positive_numeric(
                    &mut options,
                    &method_name,
                    "lambda_init",
                    defaults.lambda_init,
                )?,
                diag_scaling: take_flag(&mut flags, "diag_scaling", defaults.diag_scaling),
                increase_factor: take_positive_numeric(
                    &mut options,
                    &method_name,
                    "increase_factor",
                    defaults.increase_factor,
                )?,
                decrease_factor: take_positive_numeric(
                    &mut options,
                    &method_name,
                    "decrease_factor",
                    defaults.decrease_factor,
                )?,
                min_lambda: take_positive_numeric(
                    &mut options,
                    &method_name,
                    "min_lambda",
                    defaults.min_lambda,
                )?,
                max_lambda: take_positive_numeric(
                    &mut options,
                    &method_name,
                    "max_lambda",
                    defaults.max_lambda,
                )?,
            };
            reject_remaining_method_options(&method_name, options, flags)?;
            Ok(NonlinearSolverMethod::LevenbergMarquardt(method))
        }
        "backtracking_levenberg_marquardt" | "backtracking_lm" => {
            let defaults = rustedscithe_core::numerical::Nonlinear_systems::prelude::BacktrackingLevenbergMarquardtMethod::default();
            let method = rustedscithe_core::numerical::Nonlinear_systems::prelude::BacktrackingLevenbergMarquardtMethod {
                lambda_init: take_positive_numeric(&mut options, &method_name, "lambda_init", defaults.lambda_init)?,
                lambda_decrease: take_positive_numeric(&mut options, &method_name, "lambda_decrease", defaults.lambda_decrease)?,
                lambda_increase: take_positive_numeric(&mut options, &method_name, "lambda_increase", defaults.lambda_increase)?,
                alpha_min: take_positive_numeric(&mut options, &method_name, "alpha_min", defaults.alpha_min)?,
                max_lambda: take_positive_numeric(&mut options, &method_name, "max_lambda", defaults.max_lambda)?,
            };
            reject_remaining_method_options(&method_name, options, flags)?;
            Ok(NonlinearSolverMethod::BacktrackingLevenbergMarquardt(
                method,
            ))
        }
        "trust_region" => {
            let defaults = TrustRegionMethod::default();
            let method = TrustRegionMethod {
                delta_init: take_positive_numeric(
                    &mut options,
                    &method_name,
                    "delta_init",
                    defaults.delta_init,
                )?,
                delta_max: take_positive_numeric(
                    &mut options,
                    &method_name,
                    "delta_max",
                    defaults.delta_max,
                )?,
                eta: take_numeric(&mut options, "eta", defaults.eta)?,
                shrink_factor: take_positive_numeric(
                    &mut options,
                    &method_name,
                    "shrink_factor",
                    defaults.shrink_factor,
                )?,
                expand_factor: take_positive_numeric(
                    &mut options,
                    &method_name,
                    "expand_factor",
                    defaults.expand_factor,
                )?,
            };
            reject_remaining_method_options(&method_name, options, flags)?;
            Ok(NonlinearSolverMethod::TrustRegion(method))
        }
        "powell_dogleg" | "dogleg" => {
            let defaults = PowellDoglegMethod::default();
            let method = PowellDoglegMethod {
                delta_init: take_positive_numeric(
                    &mut options,
                    &method_name,
                    "delta_init",
                    defaults.delta_init,
                )?,
                delta_max: take_positive_numeric(
                    &mut options,
                    &method_name,
                    "delta_max",
                    defaults.delta_max,
                )?,
                eta: take_numeric(&mut options, "eta", defaults.eta)?,
                shrink_factor: take_positive_numeric(
                    &mut options,
                    &method_name,
                    "shrink_factor",
                    defaults.shrink_factor,
                )?,
                expand_factor: take_positive_numeric(
                    &mut options,
                    &method_name,
                    "expand_factor",
                    defaults.expand_factor,
                )?,
                use_double_dogleg: take_flag(
                    &mut flags,
                    "use_double_dogleg",
                    defaults.use_double_dogleg,
                ),
                use_column_scaling: take_flag(
                    &mut flags,
                    "use_column_scaling",
                    defaults.use_column_scaling,
                ),
            };
            reject_remaining_method_options(&method_name, options, flags)?;
            Ok(NonlinearSolverMethod::PowellDogleg(method))
        }
        "trust_region_lm" => {
            let defaults = TrustRegionLMMethod::default();
            let method = TrustRegionLMMethod {
                step_bound: take_positive_numeric(
                    &mut options,
                    &method_name,
                    "step_bound",
                    defaults.step_bound,
                )?,
                lambda_init: take_numeric(&mut options, "lambda_init", defaults.lambda_init)?,
                scale_diag: take_flag(&mut flags, "scale_diag", defaults.scale_diag),
                acceptance_threshold: take_numeric(
                    &mut options,
                    "acceptance_threshold",
                    defaults.acceptance_threshold,
                )?,
            };
            reject_remaining_method_options(&method_name, options, flags)?;
            Ok(NonlinearSolverMethod::TrustRegionLM(method))
        }
        "levenberg_marquardt_minpack" | "lm_minpack" => {
            let defaults = LevenbergMarquardtMinpack::default();
            let method = LevenbergMarquardtMinpack {
                ftol: take_positive_numeric(&mut options, &method_name, "ftol", defaults.ftol)?,
                xtol: take_positive_numeric(&mut options, &method_name, "xtol", defaults.xtol)?,
                gtol: take_positive_numeric(&mut options, &method_name, "gtol", defaults.gtol)?,
                maxfev: take_usize(&mut options, &method_name, "maxfev", defaults.maxfev)?,
                mode: take_usize(&mut options, &method_name, "mode", defaults.mode as usize)?
                    as i32,
                factor: take_positive_numeric(
                    &mut options,
                    &method_name,
                    "factor",
                    defaults.factor,
                )?,
                diag: defaults.diag,
            };
            reject_remaining_method_options(&method_name, options, flags)?;
            Ok(NonlinearSolverMethod::LevenbergMarquardtMinpack(method))
        }
        "nielsen_levenberg_marquardt" | "nielsen_lm" => {
            let defaults = NielsenLevenbergMarquardtMethod::default();
            let method = NielsenLevenbergMarquardtMethod {
                tau: take_positive_numeric(&mut options, &method_name, "tau", defaults.tau)?,
                nu_init: take_positive_numeric(
                    &mut options,
                    &method_name,
                    "nu_init",
                    defaults.nu_init,
                )?,
                factor_up: take_positive_numeric(
                    &mut options,
                    &method_name,
                    "factor_up",
                    defaults.factor_up,
                )?,
                factor_down: take_positive_numeric(
                    &mut options,
                    &method_name,
                    "factor_down",
                    defaults.factor_down,
                )?,
                rho_threshold: take_numeric(&mut options, "rho_threshold", defaults.rho_threshold)?,
                f_tolerance: take_optional_numeric(
                    &mut options,
                    "f_tolerance",
                    defaults.f_tolerance,
                )?,
                g_tolerance: take_optional_numeric(
                    &mut options,
                    "g_tolerance",
                    defaults.g_tolerance,
                )?,
                use_column_scaling: take_flag(
                    &mut flags,
                    "use_column_scaling",
                    defaults.use_column_scaling,
                ),
                max_rejections: take_usize(
                    &mut options,
                    &method_name,
                    "max_rejections",
                    defaults.max_rejections,
                )?,
            };
            reject_remaining_method_options(&method_name, options, flags)?;
            Ok(NonlinearSolverMethod::NielsenLevenbergMarquardt(method))
        }
        "nielsen_levenberg_marquardt_advanced" | "nielsen_lm_advanced" => {
            let defaults = NielsenLevenbergMarquardtMethodAdvanced::default();
            let method = NielsenLevenbergMarquardtMethodAdvanced {
                tau: take_positive_numeric(&mut options, &method_name, "tau", defaults.tau)?,
                nu_init: take_positive_numeric(
                    &mut options,
                    &method_name,
                    "nu_init",
                    defaults.nu_init,
                )?,
                factor_up: take_positive_numeric(
                    &mut options,
                    &method_name,
                    "factor_up",
                    defaults.factor_up,
                )?,
                factor_down: take_positive_numeric(
                    &mut options,
                    &method_name,
                    "factor_down",
                    defaults.factor_down,
                )?,
                rho_threshold: take_numeric(&mut options, "rho_threshold", defaults.rho_threshold)?,
                f_tolerance: take_optional_numeric(
                    &mut options,
                    "f_tolerance",
                    defaults.f_tolerance,
                )?,
                g_tolerance: take_optional_numeric(
                    &mut options,
                    "g_tolerance",
                    defaults.g_tolerance,
                )?,
                scaling_method: defaults.scaling_method,
                reduction_ratio_method: defaults.reduction_ratio_method,
                max_rejections: take_usize(
                    &mut options,
                    &method_name,
                    "max_rejections",
                    defaults.max_rejections,
                )?,
            };
            reject_remaining_method_options(&method_name, options, flags)?;
            Ok(NonlinearSolverMethod::NielsenLevenbergMarquardtAdvanced(
                method,
            ))
        }
        other => Err(InvalidConfigurationError::new_err(format!(
            "unsupported nonlinear method '{other}'"
        ))),
    }
}

fn statistics_counters(statistics: &SolveStatistics) -> HashMap<String, u64> {
    HashMap::from([
        ("iterations".into(), statistics.iterations as u64),
        (
            "residual_evaluations".into(),
            statistics.residual_evaluations as u64,
        ),
        (
            "jacobian_evaluations".into(),
            statistics.jacobian_evaluations as u64,
        ),
        ("linear_solves".into(), statistics.linear_solves as u64),
        (
            "linear_factorizations".into(),
            statistics.linear_factorizations as u64,
        ),
        ("accepted_steps".into(), statistics.accepted_steps as u64),
        ("rejected_steps".into(), statistics.rejected_steps as u64),
        (
            "domain_rejections".into(),
            statistics.domain_rejections as u64,
        ),
    ])
}

fn statistics_timings(statistics: &SolveStatistics) -> HashMap<String, f64> {
    let ms = |duration: std::time::Duration| duration.as_secs_f64() * 1e3;
    HashMap::from([
        ("total_ms".into(), ms(statistics.total_duration)),
        ("residual_ms".into(), ms(statistics.residual_duration)),
        ("jacobian_ms".into(), ms(statistics.jacobian_duration)),
        (
            "linear_solve_ms".into(),
            ms(statistics.linear_solve_duration),
        ),
        (
            "linear_factorization_ms".into(),
            ms(statistics.linear_factorization_duration),
        ),
    ])
}

/// Detached result of one square nonlinear-system solve.
#[pyclass(name = "NonlinearSolution", module = "rustedscithe")]
pub struct NonlinearSolution {
    solution: Vec<f64>,
    residual: Vec<f64>,
    residual_norm: f64,
    iterations: usize,
    termination: String,
    statistics: SolveStatistics,
}

#[pymethods]
impl NonlinearSolution {
    pub fn solution(&self) -> Vec<f64> {
        self.solution.clone()
    }
    pub fn residual(&self) -> Vec<f64> {
        self.residual.clone()
    }

    /// Owned NumPy array containing the solution vector.
    pub fn solution_array<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f64>> {
        crate::numpy_support::vector(py, &self.solution)
    }

    /// Owned NumPy array containing the residual vector.
    pub fn residual_array<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f64>> {
        crate::numpy_support::vector(py, &self.residual)
    }
    #[getter]
    pub fn residual_norm(&self) -> f64 {
        self.residual_norm
    }
    #[getter]
    pub fn iterations(&self) -> usize {
        self.iterations
    }
    #[getter]
    pub fn termination(&self) -> &str {
        &self.termination
    }
    pub fn telemetry_counters(&self) -> HashMap<String, u64> {
        statistics_counters(&self.statistics)
    }
    pub fn telemetry_timings_ms(&self) -> HashMap<String, f64> {
        statistics_timings(&self.statistics)
    }
    pub fn diagnostics(&self) -> HashMap<String, String> {
        HashMap::from([
            ("termination".into(), self.termination.clone()),
            ("residual_norm".into(), self.residual_norm.to_string()),
            ("iterations".into(), self.iterations.to_string()),
        ])
    }
}

/// Prepared symbolic square nonlinear-system solver.
#[pyclass(name = "NonlinearSolver", module = "rustedscithe")]
pub struct NonlinearSolver {
    /// Original expressions retained so preparation can be requested explicitly
    /// and is never repeated during a prepared parameter rebind.
    equations: Vec<rustedscithe_core::symbolic::symbolic_engine::Expr>,
    variables: Vec<String>,
    parameter_names: Option<Vec<String>>,
    prepared: Option<PreparedSymbolicNonlinearProblem>,
    frontend: SymbolicLambdifyFrontend,
    execution_policy: LambdifyExecutionPolicy,
    preparation_telemetry: PreparationTelemetryMode,
    method: String,
    method_options: HashMap<String, f64>,
    method_flags: HashMap<String, bool>,
    tolerance: f64,
    max_iterations: usize,
    parameters: Option<Vec<f64>>,
    bounds: Option<Bounds>,
    telemetry: bool,
    logging: Option<EngineLogLevel>,
}

#[pymethods]
impl NonlinearSolver {
    #[new]
    #[pyo3(signature=(system, *, method="newton", frontend="expr_legacy", execution_policy="sequential", min_work=1, telemetry="off", parameters=None, parameter_values=None, method_options=None, method_flags=None, tolerance=1e-6, max_iterations=100, bounds=None, loglevel="off"))]
    pub fn new(
        system: SymbolicSystem,
        method: &str,
        frontend: &str,
        execution_policy: &str,
        min_work: usize,
        telemetry: &str,
        parameters: Option<Vec<String>>,
        parameter_values: Option<Vec<f64>>,
        method_options: Option<HashMap<String, f64>>,
        method_flags: Option<HashMap<String, bool>>,
        tolerance: f64,
        max_iterations: usize,
        bounds: Option<Vec<(f64, f64)>>,
        loglevel: &str,
    ) -> PyResult<Self> {
        if !tolerance.is_finite() || tolerance <= 0.0 || max_iterations == 0 {
            return Err(InvalidConfigurationError::new_err(
                "tolerance must be finite and positive and max_iterations must be greater than zero",
            ));
        }
        let frontend = parse_frontend(frontend)?;
        let execution_policy = parse_execution_policy(execution_policy, min_work)?;
        let preparation_telemetry = parse_preparation_telemetry(telemetry);
        let (method_options, method_flags) =
            normalized_method_options(method_options, method_flags)?;
        // Validate the method name and its option schema at construction time;
        // numerical execution must not be the first place an option typo is
        // discovered.
        let _ = method_from_name(method, &method_options, &method_flags)?;
        let parameter_values = match (parameters.as_ref(), parameter_values) {
            (Some(names), Some(values)) => {
                if names.len() != values.len() || values.iter().any(|v| !v.is_finite()) {
                    return Err(InvalidConfigurationError::new_err(
                        "parameter names and finite parameter_values must have equal length",
                    ));
                }
                Some(values)
            }
            (Some(_), None) => None,
            (None, Some(_)) => {
                return Err(InvalidConfigurationError::new_err(
                    "parameter_values require parameter names",
                ));
            }
            (None, None) => None,
        };
        let variables = system.variables();
        let equations = system
            .expressions()
            .into_iter()
            .map(|expr| expr.inner)
            .collect();
        let parameter_names = parameters;
        let bounds = bounds
            .map(Bounds::new)
            .transpose()
            .map_err(map_solve_error)?;
        Ok(Self {
            equations,
            variables,
            parameter_names,
            prepared: None,
            frontend,
            execution_policy,
            preparation_telemetry,
            method: method.trim().to_owned(),
            method_options,
            method_flags,
            tolerance,
            max_iterations,
            parameters: parameter_values,
            bounds,
            telemetry: !matches!(
                telemetry.trim().to_ascii_lowercase().as_str(),
                "off" | "none"
            ),
            logging: parse_log_level(loglevel)?,
        })
    }

    pub fn variables(&self) -> Vec<String> {
        self.variables.clone()
    }
    pub fn backend(&self) -> String {
        "lambdify".to_owned()
    }
    pub fn frontend(&self) -> String {
        self.frontend.as_str().to_owned()
    }
    pub fn is_prepared(&self) -> bool {
        self.prepared.is_some()
    }

    /// Build the symbolic backend once. Repeated calls are intentionally cheap
    /// and do not repeat differentiation or lambdification.
    pub fn prepare(&mut self) -> PyResult<()> {
        if self.prepared.is_some() {
            return Ok(());
        }
        let mut options = SymbolicProblemOptions::new()
            .with_variables(self.variables.clone())
            .with_lambdify_backend()
            .with_lambdify_frontend(self.frontend)
            .with_lambdify_execution_policy(self.execution_policy)
            .with_preparation_telemetry(self.preparation_telemetry);
        options = match (self.parameter_names.clone(), self.parameters.clone()) {
            (Some(names), Some(values)) => options
                .with_equation_parameters(names)
                .with_equation_parameter_values(DVector::from_vec(values)),
            (Some(names), None) => options.with_equation_parameters(names),
            _ => options,
        };
        self.prepared = Some(
            PreparedSymbolicNonlinearProblem::from_expressions(self.equations.clone(), options)
                .map_err(map_solve_error)?,
        );
        Ok(())
    }

    pub fn prepare_report(&self) -> PyResult<String> {
        self.prepared
            .as_ref()
            .map(|prepared| format!("{:?}", prepared.preparation_report()))
            .ok_or_else(|| PreparationError::new_err("nonlinear solver has not been prepared"))
    }

    #[pyo3(signature=(initial_guess, parameters=None))]
    pub fn solve(
        &mut self,
        initial_guess: Vec<f64>,
        parameters: Option<Vec<f64>>,
    ) -> PyResult<NonlinearSolution> {
        catch_panic("solve", "nonlinear", "lambdify", || {
            self.solve_inner(initial_guess, parameters)
        })
    }

    fn solve_inner(
        &mut self,
        initial_guess: Vec<f64>,
        parameters: Option<Vec<f64>>,
    ) -> PyResult<NonlinearSolution> {
        self.prepare()?;
        let prepared = self
            .prepared
            .as_ref()
            .ok_or_else(|| PreparationError::new_err("nonlinear solver is not prepared"))?;
        let bound = match parameters.or_else(|| self.parameters.clone()) {
            Some(values) => prepared.bind_values(DVector::from_vec(values)),
            None => prepared.bind_without_parameters(),
        }
        .map_err(map_solve_error)?;
        let method = method_from_name(&self.method, &self.method_options, &self.method_flags)?;
        let mut options = SolveOptions {
            tolerance: self.tolerance,
            max_iterations: self.max_iterations,
            bounds: self.bounds.clone(),
            ..SolveOptions::default()
        };
        options.diagnostics = DiagnosticsOptions {
            collect_history: self.telemetry,
            collect_statistics: self.telemetry,
            enable_logging: self.logging.is_some(),
            log_level: self.logging.unwrap_or(EngineLogLevel::Info),
            enable_memory_diagnostics: false,
        };
        let result = method
            .solve(&bound, DVector::from_vec(initial_guess), options)
            .map_err(map_solve_error)?;
        Ok(NonlinearSolution {
            solution: result.x.iter().copied().collect(),
            residual: result.residual.iter().copied().collect(),
            residual_norm: result.residual_norm,
            iterations: result.iterations,
            termination: format!("{:?}", result.termination),
            statistics: result.statistics,
        })
    }
}

/// Detached result of one canonical rectangular least-squares solve.
#[pyclass(name = "FittingSolution", module = "rustedscithe")]
pub struct FittingSolution {
    solution: Vec<f64>,
    report: String,
    objective_function: f64,
    termination: String,
    statistics: SolveStatistics,
}

#[pymethods]
impl FittingSolution {
    pub fn solution(&self) -> Vec<f64> {
        self.solution.clone()
    }

    /// Owned NumPy array containing the fitted parameter vector.
    pub fn solution_array<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f64>> {
        crate::numpy_support::vector(py, &self.solution)
    }
    pub fn report(&self) -> &str {
        &self.report
    }
    #[getter]
    pub fn objective_function(&self) -> f64 {
        self.objective_function
    }
    #[getter]
    pub fn termination(&self) -> &str {
        &self.termination
    }
    pub fn diagnostics(&self) -> HashMap<String, String> {
        HashMap::from([
            ("termination".into(), self.termination.clone()),
            (
                "objective_function".into(),
                self.objective_function.to_string(),
            ),
            ("iterations".into(), self.statistics.iterations.to_string()),
            (
                "residual_evaluations".into(),
                self.statistics.residual_evaluations.to_string(),
            ),
        ])
    }
    pub fn telemetry_counters(&self) -> HashMap<String, u64> {
        statistics_counters(&self.statistics)
    }
    pub fn telemetry_timings_ms(&self) -> HashMap<String, f64> {
        statistics_timings(&self.statistics)
    }
}

/// Builder-style symbolic fitting wrapper over `Nonlinear_systems::least_squares`.
#[pyclass(name = "FittingSolver", module = "rustedscithe")]
pub struct FittingSolver {
    inner: SymbolicLeastSquaresSolver,
}

#[pymethods]
impl FittingSolver {
    #[new]
    #[pyo3(signature=(equations, initial_guess, variables=None, parameters=None, frontend="expr_legacy", execution_policy="sequential", min_work=1, telemetry="off", tolerance=None, f_tolerance=None, g_tolerance=None, max_iterations=None, scale_diag=None, loglevel="off", positive_variables=None))]
    pub fn new(
        equations: Vec<PyExpr>,
        initial_guess: Vec<f64>,
        variables: Option<Vec<String>>,
        parameters: Option<Vec<String>>,
        frontend: &str,
        execution_policy: &str,
        min_work: usize,
        telemetry: &str,
        tolerance: Option<f64>,
        f_tolerance: Option<f64>,
        g_tolerance: Option<f64>,
        max_iterations: Option<usize>,
        scale_diag: Option<bool>,
        loglevel: &str,
        positive_variables: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let mode = match telemetry.trim().to_ascii_lowercase().as_str() {
            "off" | "none" => LeastSquaresTelemetryMode::Off,
            "counters" => LeastSquaresTelemetryMode::Counters,
            "detailed" | "timings" => LeastSquaresTelemetryMode::Detailed,
            other => {
                return Err(InvalidConfigurationError::new_err(format!(
                    "unsupported fitting telemetry '{other}'"
                )));
            }
        };
        let frontend = parse_frontend(frontend)?;
        let execution_policy = parse_execution_policy(execution_policy, min_work)?;
        let mut inner = SymbolicLeastSquaresSolver::new()
            .with_equations(equations.into_iter().map(|expr| expr.inner).collect())
            .with_initial_guess(initial_guess)
            .with_lambdify_frontend(frontend)
            .with_lambdify_execution_policy(execution_policy)
            .with_telemetry(mode)
            .try_with_loglevel(loglevel)
            .map_err(map_least_squares_error)?;
        if let Some(values) = variables {
            inner = inner.with_unknowns(values);
        }
        if let Some(values) = parameters {
            inner = inner.with_parameters(values);
        }
        if let Some(value) = tolerance {
            inner = inner.with_tolerance(value);
        }
        if let Some(value) = f_tolerance {
            inner = inner.with_f_tolerance(value);
        }
        if let Some(value) = g_tolerance {
            inner = inner.with_g_tolerance(value);
        }
        if let Some(value) = max_iterations {
            inner = inner.with_max_iterations(value);
        }
        if let Some(value) = scale_diag {
            inner = inner.with_scale_diag(value);
        }
        if let Some(names) = positive_variables {
            inner
                .set_positive_variables(&names)
                .map_err(map_solve_error)?;
        }
        inner = inner.try_build().map_err(map_least_squares_error)?;
        Ok(Self { inner })
    }

    pub fn variables(&self) -> Vec<String> {
        self.inner.values.clone()
    }
    pub fn solve(&mut self) -> PyResult<FittingSolution> {
        let report = self.inner.try_solve().map_err(map_least_squares_error)?;
        let solution = self
            .inner
            .result
            .as_ref()
            .map(|value| value.iter().copied().collect())
            .unwrap_or_default();
        Ok(FittingSolution {
            solution,
            report: format!("{:?}", report.termination),
            objective_function: report.objective_function,
            termination: format!("{:?}", report.termination),
            statistics: report.statistics,
        })
    }

    pub fn solve_with_parameters(&mut self, parameters: Vec<f64>) -> PyResult<FittingSolution> {
        let report = self
            .inner
            .try_solve_with_params(parameters)
            .map_err(map_least_squares_error)?;
        let solution = self
            .inner
            .result
            .as_ref()
            .map(|value| value.iter().copied().collect())
            .unwrap_or_default();
        Ok(FittingSolution {
            solution,
            report: format!("{:?}", report.termination),
            objective_function: report.objective_function,
            termination: format!("{:?}", report.termination),
            statistics: report.statistics,
        })
    }
}
