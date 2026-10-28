//! Python request/prepared-solver adapter for the native typed `BVP_api`.
//!
//! The adapter translates Python-friendly node-major arrays and common solver
//! options; numerical orchestration, result routing and lifecycle stay in Rust.

use std::collections::HashMap;
use std::path::PathBuf;

use nalgebra::DMatrix;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use rustedscithe_core::numerical::BVP_Damp::telemetry::BvpLoggingMode;
use rustedscithe_core::numerical::BVP_Damp::NR_Damp_solver_frozen::FrozenSolverOptions;
use rustedscithe_core::numerical::BVP_api::{
    BvpApiError, BvpBoundaryCondition, BvpBoundarySide, BvpParameterLoopReport, BvpProblem,
    BvpSciFreeParameters, BvpSciSparseFactorization, BvpSolution, BvpSolver, BvpSolverConfig,
    BvpSolverTelemetrySnapshot,
};
use rustedscithe_core::numerical::BVP_sci::new::{BvpSciAssembly, BvpSciExecution, BvpSciOptions};
use rustedscithe_core::Utils::postprocessing::{PostprocessPlan, PostprocessStatus};

use crate::contract::SolverOptions;
use crate::errors::{
    annotate_native, catch_panic, native_variant, CallbackError, CallbackFailureState,
    ExhaustionError, InvalidConfigurationError, PreparationError, SolverError, SolverPanicState,
    UnsupportedOptionError,
};
use crate::expr::PyExpr;

/// Turn a row-major `[node][state]` Python guess into the native state-major
/// matrix. Route factories validate the route-specific number of nodes first.
fn guess_matrix(rows: Vec<Vec<f64>>, states: usize, nodes: usize) -> PyResult<DMatrix<f64>> {
    if rows.len() != nodes || rows.iter().any(|row| row.len() != states) {
        return Err(InvalidConfigurationError::new_err(format!(
            "initial_guess must have shape ({nodes}, {states}) in [node][state] order"
        )));
    }
    if rows.iter().flatten().any(|value| !value.is_finite()) {
        return Err(InvalidConfigurationError::new_err(
            "initial_guess values must be finite",
        ));
    }
    Ok(DMatrix::from_fn(states, nodes, |state, node| {
        rows[node][state]
    }))
}

fn problem(
    expressions: Vec<PyExpr>,
    state_variables: Vec<String>,
    independent_variable: String,
    mesh: Vec<f64>,
    boundary_conditions: Vec<(usize, String, f64)>,
) -> PyResult<BvpProblem> {
    let conditions = boundary_conditions
        .into_iter()
        .map(|(state, side, value)| {
            let side = match side.trim().to_ascii_lowercase().as_str() {
                "left" => BvpBoundarySide::Left,
                "right" => BvpBoundarySide::Right,
                _ => {
                    return Err(InvalidConfigurationError::new_err(
                        "boundary side must be 'left' or 'right'",
                    ));
                }
            };
            Ok(BvpBoundaryCondition { state, side, value })
        })
        .collect::<PyResult<Vec<_>>>()?;
    BvpProblem::new(
        expressions.into_iter().map(|expr| expr.inner).collect(),
        state_variables,
        independent_variable,
        mesh,
        conditions,
    )
    .map_err(map_api_error)
}

fn shared_options(options: Option<PyRef<'_, SolverOptions>>) -> PyResult<SolverOptions> {
    match options {
        Some(options) => Ok(options.clone()),
        None => SolverOptions::new(None, None, None, None, None, None, None, None, None),
    }
}

fn parse_bvp_logging_mode(value: &str) -> PyResult<BvpLoggingMode> {
    match value {
        "off" => Ok(BvpLoggingMode::Off),
        "warnings" => Ok(BvpLoggingMode::Warnings),
        "detailed" => Ok(BvpLoggingMode::Detailed),
        other => Err(InvalidConfigurationError::new_err(format!(
            "logging_mode must be off, warnings or detailed; got '{other}'"
        ))),
    }
}

fn validate_symbolic_damp_route(options: &SolverOptions) -> PyResult<()> {
    if options.execution() == "numerical" {
        return Err(UnsupportedOptionError::new_err(
            "BVP_Damp symbolic facade routes do not accept numerical callbacks; use the direct numerical API where available",
        ));
    }
    Ok(())
}

fn map_api_error(error: BvpApiError) -> PyErr {
    let (mapped, kind, solver, route) = match error {
        BvpApiError::InvalidProblem(message) => (
            InvalidConfigurationError::new_err(message),
            "InvalidProblem".to_owned(),
            "bvp_api",
            "configuration",
        ),
        BvpApiError::UnsupportedRoute { solver, reason } => (
            UnsupportedOptionError::new_err(format!("{solver}: {reason}")),
            "UnsupportedRoute".to_owned(),
            "bvp_api",
            solver,
        ),
        BvpApiError::Damped(error) | BvpApiError::Frozen(error) => {
            return crate::bvp_damp::map_native_error(error);
        }
        BvpApiError::Sci(error) => {
            let kind = native_variant(&error);
            let message = error.to_string();
            let mapped = match error {
                rustedscithe_core::numerical::BVP_sci::new::BvpSciNewError::InvalidConfiguration(_)
                | rustedscithe_core::numerical::BVP_sci::new::BvpSciNewError::ParameterBindingMismatch { .. } => {
                    InvalidConfigurationError::new_err(message)
                }
                rustedscithe_core::numerical::BVP_sci::new::BvpSciNewError::UnsupportedRoute(_) => {
                    UnsupportedOptionError::new_err(message)
                }
                rustedscithe_core::numerical::BVP_sci::new::BvpSciNewError::SymbolicPreparation(_)
                | rustedscithe_core::numerical::BVP_sci::new::BvpSciNewError::AotPreparation { .. }
                | rustedscithe_core::numerical::BVP_sci::new::BvpSciNewError::AotHandoff { .. } => {
                    PreparationError::new_err(message)
                }
                rustedscithe_core::numerical::BVP_sci::new::BvpSciNewError::Callback { .. }
                | rustedscithe_core::numerical::BVP_sci::new::BvpSciNewError::NonFinite { .. }
                | rustedscithe_core::numerical::BVP_sci::new::BvpSciNewError::AotCallback { .. } => {
                    CallbackError::new_err(message)
                }
                rustedscithe_core::numerical::BVP_sci::new::BvpSciNewError::MaxNodesExceeded { .. }
                | rustedscithe_core::numerical::BVP_sci::new::BvpSciNewError::MeshRefinementLimit { .. } => {
                    ExhaustionError::new_err(message)
                }
                _ => SolverError::new_err(message),
            };
            (mapped, kind, "bvp_sci", "native")
        }
        BvpApiError::Postprocessing { solver, source } => (
            SolverError::new_err(source.to_string()),
            "Postprocessing".to_owned(),
            "bvp_api",
            solver,
        ),
        BvpApiError::MissingResult => (
            SolverError::new_err("native solver completed without a result"),
            "MissingResult".to_owned(),
            "bvp_api",
            "result",
        ),
    };
    annotate_native(mapped, "solve", solver, route, &kind)
}

/// One explicit, tagged native BVP_api request. Use a route factory rather
/// than combining unrelated controls from different solver families.
#[pyclass(name = "BvpSolverRequest", module = "rustedscithe", unsendable)]
pub struct PyBvpSolverRequest {
    problem: Option<BvpProblem>,
    config: Option<BvpSolverConfig>,
    route: &'static str,
    parameters: Option<(Vec<String>, Vec<f64>)>,
    callback_failure: CallbackFailureState,
}

/// Prepared typed BVP_api solver. Its `solve` result keeps the route tag and
/// exposes detached NumPy-friendly node-major values.
#[pyclass(name = "PreparedBvpSolver", module = "rustedscithe", unsendable)]
pub struct PyPreparedBvpSolver {
    inner: BvpSolver,
    route: &'static str,
    panic_state: SolverPanicState,
    callback_failure: CallbackFailureState,
}

/// Route-tagged common BVP solution returned by the generic facade.
#[pyclass(name = "BvpApiSolution", module = "rustedscithe")]
#[derive(Clone)]
pub struct PyBvpApiSolution {
    route: String,
    mesh: Vec<f64>,
    values: Vec<Vec<f64>>,
    sci_details: Option<crate::bvp::BvpSciSolution>,
}

/// Native export/plot plan for datasets produced by the generic BVP facade.
#[pyclass(name = "BvpPostprocessPlan", module = "rustedscithe")]
#[derive(Clone, Default)]
pub struct PyBvpPostprocessPlan {
    inner: PostprocessPlan,
}

#[pymethods]
impl PyBvpPostprocessPlan {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// Save the retained mesh solution as CSV.
    pub fn save_csv(&self, path: String) -> Self {
        Self {
            inner: self.inner.clone().save_csv(PathBuf::from(path)),
        }
    }
    /// Save the retained mesh solution as a native text table.
    pub fn save_txt(&self, path: String) -> Self {
        Self {
            inner: self.inner.clone().save_txt(PathBuf::from(path)),
        }
    }
    /// Write a compact Markdown dataset report.
    pub fn write_report(&self, path: String) -> Self {
        Self {
            inner: self.inner.clone().write_report(PathBuf::from(path)),
        }
    }
    /// Render one Plotters PNG per state variable.
    pub fn plotters_png(&self, output_dir: String) -> Self {
        Self {
            inner: self.inner.clone().plotters_png(PathBuf::from(output_dir)),
        }
    }
    /// Request optional Gnuplot PNG output.
    pub fn gnuplot_png(&self, output_dir: String) -> Self {
        Self {
            inner: self.inner.clone().gnuplot_png(PathBuf::from(output_dir)),
        }
    }
    /// Render a compact plot in the current terminal.
    pub fn terminal_plot(&self) -> Self {
        Self {
            inner: self.inner.clone().terminal_plot(),
        }
    }
}

#[pymethods]
impl PyBvpSolverRequest {
    /// Configure the adaptive Damped Newton route. Its guess has one row per
    /// interval; the solver manages any later mesh refinement itself.
    #[staticmethod]
    #[pyo3(signature=(expressions, state_variables, independent_variable, mesh, boundary_conditions, initial_guess, options=None, parameters=None, parameter_values=None, scheme="forward", strategy="Damped", tolerance=1e-6, max_iterations=25, bounds=None, loglevel=None, linear_sys_method=None, max_jacobian_refreshes=None, max_damping_iterations=None, damping_factor=None, max_refinements=None, grid_method=None, grid_parameters=None, logging_mode="off"))]
    #[allow(clippy::too_many_arguments)]
    fn damped(
        expressions: Vec<PyExpr>,
        state_variables: Vec<String>,
        independent_variable: String,
        mesh: Vec<f64>,
        boundary_conditions: Vec<(usize, String, f64)>,
        initial_guess: Vec<Vec<f64>>,
        options: Option<PyRef<'_, SolverOptions>>,
        parameters: Option<Vec<String>>,
        parameter_values: Option<Vec<f64>>,
        scheme: &str,
        strategy: &str,
        tolerance: f64,
        max_iterations: usize,
        bounds: Option<HashMap<String, (f64, f64)>>,
        loglevel: Option<String>,
        linear_sys_method: Option<String>,
        max_jacobian_refreshes: Option<usize>,
        max_damping_iterations: Option<usize>,
        damping_factor: Option<f64>,
        max_refinements: Option<usize>,
        grid_method: Option<String>,
        grid_parameters: Option<Vec<f64>>,
        logging_mode: &str,
    ) -> PyResult<Self> {
        let options = shared_options(options)?;
        validate_symbolic_damp_route(&options)?;
        if strategy != "Damped" {
            return Err(UnsupportedOptionError::new_err(
                "the Damped route supports only strategy='Damped'",
            ));
        }
        let loglevel = loglevel.or_else(|| {
            (options.logging_level() != "off").then(|| options.logging_level().to_owned())
        });
        let problem = problem(
            expressions,
            state_variables.clone(),
            independent_variable,
            mesh.clone(),
            boundary_conditions,
        )?;
        let guess = guess_matrix(
            initial_guess,
            state_variables.len(),
            mesh.len().saturating_sub(1),
        )?;
        let strategy_params = crate::bvp_damp::strategy_params(
            max_jacobian_refreshes,
            max_damping_iterations,
            damping_factor,
            max_refinements,
            grid_method.as_deref(),
            grid_parameters,
        )?;
        let damped_options = crate::bvp_damp::options_for_solver(
            &options,
            scheme,
            tolerance,
            max_iterations,
            bounds.unwrap_or_default(),
            &state_variables,
            strategy_params,
            linear_sys_method.as_deref(),
            loglevel.as_deref(),
            parse_bvp_logging_mode(logging_mode)?,
            256,
            None,
            None,
            0,
        )?;
        let mut damped_options = damped_options;
        damped_options.strategy = strategy.to_owned();
        let config = BvpSolverConfig::Damped {
            options: damped_options,
            initial_guess: guess,
        };
        Ok(Self {
            problem: Some(problem),
            config: Some(config),
            route: "damped",
            parameters: parameter_pair(parameters, parameter_values)?,
            callback_failure: CallbackFailureState::default(),
        })
    }

    /// Configure the fixed-mesh Frozen route. Its guess has one row per mesh
    /// node. Algorithm controls remain distinct from Damped's adaptive options.
    #[staticmethod]
    #[pyo3(signature=(expressions, state_variables, independent_variable, mesh, boundary_conditions, initial_guess, options=None, parameters=None, parameter_values=None, scheme="forward", strategy="Frozen", tolerance=1e-6, max_iterations=25, strategy_params=None, linear_sys_method=None, logging_mode="off"))]
    #[allow(clippy::too_many_arguments)]
    fn frozen(
        expressions: Vec<PyExpr>,
        state_variables: Vec<String>,
        independent_variable: String,
        mesh: Vec<f64>,
        boundary_conditions: Vec<(usize, String, f64)>,
        initial_guess: Vec<Vec<f64>>,
        options: Option<PyRef<'_, SolverOptions>>,
        parameters: Option<Vec<String>>,
        parameter_values: Option<Vec<f64>>,
        scheme: &str,
        strategy: &str,
        tolerance: f64,
        max_iterations: usize,
        strategy_params: Option<HashMap<String, Option<Vec<f64>>>>,
        linear_sys_method: Option<String>,
        logging_mode: &str,
    ) -> PyResult<Self> {
        let options = shared_options(options)?;
        validate_symbolic_damp_route(&options)?;
        if options.logging_level() != "off" {
            return Err(UnsupportedOptionError::new_err(
                "BVP_Damp Frozen/Naive does not expose the shared Rust log-level control; use logging_mode for its native decision trace",
            ));
        }
        let problem = problem(
            expressions,
            state_variables.clone(),
            independent_variable,
            mesh.clone(),
            boundary_conditions,
        )?;
        let guess = guess_matrix(initial_guess, state_variables.len(), mesh.len())?;
        let config = crate::bvp_damp::backend_config(
            &options,
            parse_bvp_logging_mode(logging_mode)?,
            256,
            None,
            None,
            0,
        )?;
        let strategy_params = match strategy {
            "Frozen" => strategy_params,
            "Naive" if strategy_params.is_none() => None,
            "Naive" => {
                return Err(InvalidConfigurationError::new_err(
                    "Naive Newton does not accept Frozen strategy_params",
                ));
            }
            other => {
                return Err(UnsupportedOptionError::new_err(format!(
                    "Frozen route strategy must be 'Frozen' or 'Naive', got '{other}'"
                )));
            }
        };
        let mut frozen = match options.matrix_layout() {
            "dense" => FrozenSolverOptions::dense_frozen(),
            "sparse" => FrozenSolverOptions::sparse_frozen(),
            "banded" => FrozenSolverOptions::banded_frozen(),
            _ => {
                return Err(UnsupportedOptionError::new_err(
                    "unsupported Frozen matrix layout",
                ));
            }
        };
        frozen.scheme = scheme.to_owned();
        frozen.strategy = strategy.to_owned();
        if strategy_params.is_some() || strategy == "Naive" {
            frozen.strategy_params = strategy_params;
        }
        frozen.linear_sys_method = linear_sys_method;
        frozen.tolerance = tolerance;
        frozen.max_iterations = max_iterations;
        frozen.generated_backend_config = config;
        Ok(Self {
            problem: Some(problem),
            config: Some(BvpSolverConfig::Frozen {
                options: frozen,
                initial_guess: guess,
            }),
            route: if strategy == "Naive" {
                "naive"
            } else {
                "frozen"
            },
            parameters: parameter_pair(parameters, parameter_values)?,
            callback_failure: CallbackFailureState::default(),
        })
    }

    /// Configure symbolic BVP_sci collocation. Numerical Python callbacks are
    /// not part of this symbolic facade and remain on `BvpSciSolver` directly.
    #[staticmethod]
    #[pyo3(signature=(expressions, state_variables, independent_variable, mesh, boundary_conditions, initial_guess, options=None, tolerance=1e-3, boundary_tolerance=None, max_nodes=1000, max_newton_iterations=10, max_jacobian_refreshes=4, max_mesh_refinements=10, max_backtracking_steps=8, output_policy="mesh", free_parameter_names=None, free_parameter_guess=None, boundary_residual=None, boundary_jacobian=None, external_parameters=None, external_parameter_values=None))]
    #[allow(clippy::too_many_arguments)]
    fn sci(
        expressions: Vec<PyExpr>,
        state_variables: Vec<String>,
        independent_variable: String,
        mesh: Vec<f64>,
        boundary_conditions: Vec<(usize, String, f64)>,
        initial_guess: Vec<Vec<f64>>,
        options: Option<PyRef<'_, SolverOptions>>,
        tolerance: f64,
        boundary_tolerance: Option<f64>,
        max_nodes: usize,
        max_newton_iterations: usize,
        max_jacobian_refreshes: usize,
        max_mesh_refinements: usize,
        max_backtracking_steps: usize,
        output_policy: &str,
        free_parameter_names: Option<Vec<String>>,
        free_parameter_guess: Option<Vec<f64>>,
        boundary_residual: Option<Py<PyAny>>,
        boundary_jacobian: Option<Py<PyAny>>,
        external_parameters: Option<Vec<String>>,
        external_parameter_values: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let options = shared_options(options)?;
        if options.execution() == "numerical" {
            return Err(UnsupportedOptionError::new_err(
                "BvpSolverRequest.sci accepts symbolic Lambdify/AOT routes; use BvpSciSolver for numerical callbacks",
            ));
        }
        let mut problem = problem(
            expressions,
            state_variables.clone(),
            independent_variable,
            mesh.clone(),
            boundary_conditions,
        )?;
        if let Some((names, values)) =
            external_parameter_pair(external_parameters, external_parameter_values)?
        {
            problem = problem
                .with_external_parameters(names, values)
                .map_err(map_api_error)?;
        }
        let guess = guess_matrix(initial_guess, state_variables.len(), mesh.len())?;
        let assembly = match options.frontend() {
            "expr_legacy" => BvpSciAssembly::ExprLegacy,
            "atom_native" => BvpSciAssembly::AtomViewNative,
            other => {
                return Err(UnsupportedOptionError::new_err(format!(
                    "unsupported BVP_sci symbolic frontend '{other}'"
                )));
            }
        };
        let execution = match options.execution() {
            "lambdify" => BvpSciExecution::Lambdify,
            "aot" => BvpSciExecution::Aot,
            _ => unreachable!("numerical was rejected above"),
        };
        let mut sci_options = BvpSciOptions::default()
            .with_execution(execution)
            .with_assembly(assembly)
            .with_matrix_layout(crate::bvp::map_layout(&options)?)
            .with_execution_policy(crate::bvp::map_policy(options.execution_policy())?)
            .with_telemetry(crate::bvp::map_telemetry(options.telemetry())?)
            .with_output_policy(crate::bvp::map_output(output_policy)?)
            .with_tolerance(tolerance)
            .with_limits(max_nodes, max_newton_iterations, max_mesh_refinements)
            .with_backtracking_limit(max_backtracking_steps);
        sci_options.max_jacobian_refreshes = max_jacobian_refreshes;
        sci_options = sci_options
            .with_logging_level(crate::contract::native_logging_level(&options, "bvp_sci")?);
        if let Some(boundary_tolerance) = boundary_tolerance {
            sci_options = sci_options.with_boundary_tolerance(boundary_tolerance);
        }
        let sparse_factorization = match options.sparse_factorization() {
            "standard_faer" => BvpSciSparseFactorization::StandardFaer,
            "superlu" => BvpSciSparseFactorization::SuperLu,
            other => {
                return Err(UnsupportedOptionError::new_err(format!(
                    "BVP_api supports standard_faer or superlu; got '{other}'"
                )));
            }
        };
        let aot_config = if execution == BvpSciExecution::Aot {
            Some(crate::aot::generated_config(&options)?)
        } else {
            None
        };
        let callback_failure = CallbackFailureState::default();
        let config = match (
            free_parameter_names,
            free_parameter_guess,
            boundary_residual,
        ) {
            (None, None, None) if boundary_jacobian.is_none() => BvpSolverConfig::Sci {
                options: sci_options,
                sparse_factorization,
                aot_config,
                initial_guess: guess,
            },
            (Some(names), Some(initial_guess), Some(residual_callback)) => {
                let parameter_count = initial_guess.len();
                let residual_failure = callback_failure.clone();
                let callback_state = residual_failure.clone();
                let residual = move |ya: &[f64],
                                     yb: &[f64],
                                     parameters: &[f64],
                                     out: &mut [f64]| {
                    let result = Python::attach(|py| -> PyResult<Vec<f64>> {
                        residual_callback
                            .call1(py, (ya, yb, parameters))?
                            .extract::<Vec<f64>>(py)
                    });
                    let values = match result {
                        Ok(values) => values,
                        Err(error) => {
                            let message = format!("boundary_residual callback failed: {error}");
                            callback_state.record(message.clone(), Some(error));
                            return Err(message);
                        }
                    };
                    if values.len() != out.len() || values.iter().any(|value| !value.is_finite()) {
                        return Err(format!(
                            "boundary residual must return {} finite values",
                            out.len()
                        ));
                    }
                    out.copy_from_slice(&values);
                    Ok(())
                };
                let mut free = BvpSciFreeParameters::new(names, initial_guess, residual);
                if let Some(jacobian_callback) = boundary_jacobian {
                    let state_count = state_variables.len();
                    let callback_failure = residual_failure.clone();
                    let jacobian = move |ya: &[f64],
                                         yb: &[f64],
                                         parameters: &[f64],
                                         out_ya: &mut [f64],
                                         out_yb: &mut [f64],
                                         out_p: &mut [f64]| {
                        let result = Python::attach(|py| -> PyResult<_> {
                            jacobian_callback
                                .call1(py, (ya, yb, parameters))?
                                .extract::<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>)>(py)
                        });
                        let (ya_rows, yb_rows, p_rows) = match result {
                            Ok(rows) => rows,
                            Err(error) => {
                                let message = format!("boundary_jacobian callback failed: {error}");
                                callback_failure.record(message.clone(), Some(error));
                                return Err(message);
                            }
                        };
                        let ya_flat = checked_callback_matrix(
                            ya_rows,
                            parameter_count,
                            state_count,
                            "dR/dya",
                        )?;
                        let yb_flat = checked_callback_matrix(
                            yb_rows,
                            parameter_count,
                            state_count,
                            "dR/dyb",
                        )?;
                        let p_flat = checked_callback_matrix(
                            p_rows,
                            parameter_count,
                            parameter_count,
                            "dR/dp",
                        )?;
                        out_ya.copy_from_slice(&ya_flat);
                        out_yb.copy_from_slice(&yb_flat);
                        out_p.copy_from_slice(&p_flat);
                        Ok(())
                    };
                    free = free.with_jacobian(jacobian);
                }
                BvpSolverConfig::SciWithFreeParameters {
                    options: sci_options,
                    sparse_factorization,
                    aot_config,
                    initial_guess: guess,
                    free_parameters: free,
                }
            }
            _ => {
                return Err(InvalidConfigurationError::new_err(
                    "free_parameter_names, free_parameter_guess and boundary_residual must be supplied together; boundary_jacobian is optional",
                ));
            }
        };
        Ok(Self {
            problem: Some(problem),
            config: Some(config),
            route: "sci",
            parameters: None,
            callback_failure,
        })
    }

    /// Construct the native facade solver. Damped fixed evaluator inputs are
    /// bound before the first solve; BVP_sci parameter unknowns are separate.
    pub fn prepare(&mut self) -> PyResult<PyPreparedBvpSolver> {
        let problem = self.problem.take().ok_or_else(|| {
            InvalidConfigurationError::new_err("request has already been prepared")
        })?;
        let config = self.config.take().ok_or_else(|| {
            InvalidConfigurationError::new_err("request has already been prepared")
        })?;
        let parameters = self.parameters.take();
        let route = self.route;
        let solver = catch_panic("construct", "bvp_api", route, || {
            let mut solver = BvpSolver::new(problem, config).map_err(map_api_error)?;
            if let Some((names, values)) = parameters {
                solver
                    .bind_parameters(names, values)
                    .map_err(map_api_error)?;
            }
            Ok(solver)
        })?;
        Ok(PyPreparedBvpSolver {
            inner: solver,
            route,
            panic_state: SolverPanicState::default(),
            callback_failure: self.callback_failure.clone(),
        })
    }
}

#[pymethods]
impl PyPreparedBvpSolver {
    /// Execute one solve and return route-tagged, detached node-major arrays.
    pub fn solve(&mut self) -> PyResult<PyBvpApiSolution> {
        let panic_state = self.panic_state.clone();
        let route = self.route;
        let callback_failure = self.callback_failure.clone();
        let solution = panic_state.run("solve", "bvp_api", route, || {
            callback_failure.clear();
            self.inner
                .solve()
                .map(Clone::clone)
                .map_err(|error| callback_failure.attach_cause(map_api_error(error)))
        })?;
        let mesh = self
            .inner
            .solution_view()
            .expect("successful solve stores a result")
            .mesh()
            .to_vec();
        let view = self
            .inner
            .solution_view()
            .expect("successful solve stores a result");
        let values = (0..mesh.len())
            .map(|node| {
                (0..view.state_count())
                    .map(|state| view.value(node, state).expect("view bounds are valid"))
                    .collect()
            })
            .collect();
        let route = match &solution {
            BvpSolution::Damped { .. } => "damped",
            BvpSolution::Frozen { .. } => "frozen",
            BvpSolution::Sci(_) => "sci",
        };
        let sci_details = match (solution, self.inner.telemetry_snapshot()) {
            (BvpSolution::Sci(native), BvpSolverTelemetrySnapshot::Sci(snapshot)) => {
                Some(crate::bvp::BvpSciSolution::from_native(native, snapshot))
            }
            _ => None,
        };
        Ok(PyBvpApiSolution {
            route: route.to_owned(),
            mesh,
            values,
            sci_details,
        })
    }

    /// Rebind fixed model parameters for Damped/Frozen, then solve again.
    pub fn solve_with_parameter_values(&mut self, values: Vec<f64>) -> PyResult<PyBvpApiSolution> {
        let panic_state = self.panic_state.clone();
        let route = self.route;
        let solution = panic_state.run("parameter_rebind", "bvp_api", route, || {
            self.inner
                .solve_with_parameter_values(values)
                .map(Clone::clone)
                .map_err(map_api_error)
        })?;
        let view = self
            .inner
            .solution_view()
            .expect("successful solve stores a result");
        let mesh = view.mesh().to_vec();
        let values = (0..mesh.len())
            .map(|node| {
                (0..view.state_count())
                    .map(|state| view.value(node, state).expect("view bounds are valid"))
                    .collect()
            })
            .collect();
        let route = match &solution {
            BvpSolution::Damped { .. } => "damped",
            BvpSolution::Frozen { .. } => "frozen",
            BvpSolution::Sci(_) => "sci",
        };
        let sci_details = match (solution, self.inner.telemetry_snapshot()) {
            (BvpSolution::Sci(native), BvpSolverTelemetrySnapshot::Sci(snapshot)) => {
                Some(crate::bvp::BvpSciSolution::from_native(native, snapshot))
            }
            _ => None,
        };
        Ok(PyBvpApiSolution {
            route: route.to_owned(),
            mesh,
            values,
            sci_details,
        })
    }

    /// Run an ordered parameter continuation loop and return the selected
    /// solver family's native Python report class. Point failures remain in
    /// the report; only preflight or unrecoverable stop errors raise here.
    pub fn solve_parameter_loop(
        &mut self,
        parameter_vectors: Vec<Vec<f64>>,
    ) -> PyResult<Py<PyAny>> {
        for (index, values) in parameter_vectors.iter().enumerate() {
            if values.iter().any(|value| !value.is_finite()) {
                return Err(InvalidConfigurationError::new_err(format!(
                    "parameter_vectors[{index}] must contain only finite values"
                )));
            }
        }
        let route = self.route;
        let report = self
            .panic_state
            .clone()
            .run("parameter_loop", "bvp_api", route, || {
                self.inner
                    .solve_parameter_loop(&parameter_vectors)
                    .map_err(map_api_error)
            })?;

        Python::attach(|py| match report {
            BvpParameterLoopReport::Damped(report) => {
                let report = crate::bvp_damp::damped_parameter_loop_report(py, route, report)?;
                Ok(Py::new(py, report)?.into_any())
            }
            BvpParameterLoopReport::Frozen(report) => {
                let (solver_name, native_route) = match route {
                    "naive" => ("bvp_damp_naive", "naive"),
                    _ => ("bvp_damp_frozen", "frozen"),
                };
                let report =
                    crate::bvp_damp::parameter_loop_report(py, solver_name, native_route, report)?;
                Ok(Py::new(py, report)?.into_any())
            }
            BvpParameterLoopReport::Sci(report) => {
                let report = crate::bvp::parameter_loop_report_from_native(
                    report,
                    self.callback_failure.clone(),
                );
                Ok(Py::new(py, report)?.into_any())
            }
        })
    }

    /// Replace BVP_sci free-parameter Newton guesses without changing their schema.
    pub fn set_parameter_guess(&mut self, values: Vec<f64>) -> PyResult<()> {
        let panic_state = self.panic_state.clone();
        let route = self.route;
        panic_state.run("parameter_guess", "bvp_api", route, || {
            self.inner
                .set_parameter_guess(values)
                .map_err(map_api_error)
        })
    }

    /// Set fixed evaluator values on an already declared Damped/Frozen schema.
    pub fn set_parameter_values(&mut self, values: Vec<f64>) -> PyResult<()> {
        let panic_state = self.panic_state.clone();
        let route = self.route;
        panic_state.run("parameter_rebind", "bvp_api", route, || {
            self.inner
                .set_parameter_values(values)
                .map_err(map_api_error)
        })
    }

    /// Current BVP_sci free-parameter unknowns; unsupported on Damped/Frozen.
    pub fn parameter_guess(&self) -> PyResult<Vec<f64>> {
        self.inner.parameter_guess().map_err(map_api_error)
    }

    /// Route tag selected by the request factory.
    #[getter]
    pub fn route(&self) -> &'static str {
        self.route
    }

    /// Stable native telemetry family tag. Route-native details remain
    /// represented by their own native counter/timing keys; absent measurements
    /// stay absent from those maps rather than being fabricated as zero.
    #[getter]
    pub fn telemetry_route(&self) -> &'static str {
        self.route
    }

    /// Copy route-native telemetry into a Python dict with `mode`, `counters`
    /// and `timings_ms`. The native scopes are not merged or summed.
    pub fn telemetry_snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let output = PyDict::new(py);
        output.set_item("route", self.route)?;
        let snapshot = self.inner.telemetry_snapshot();
        match snapshot {
            BvpSolverTelemetrySnapshot::Damped(snapshot)
            | BvpSolverTelemetrySnapshot::Frozen(snapshot) => {
                output.set_item(
                    "mode",
                    format!("{:?}", snapshot.telemetry_mode).to_ascii_lowercase(),
                )?;
                let counters = PyDict::new(py);
                if snapshot.telemetry_mode.collects_counters() {
                    let values = [
                        ("iterations", snapshot.counters.iterations),
                        ("scalar_evaluations", snapshot.counters.scalar_evaluations),
                        ("residual_calls", snapshot.counters.residual_calls),
                        ("residual_requests", snapshot.counters.residual_requests),
                        ("jacobian_requests", snapshot.counters.jacobian_requests),
                        (
                            "jacobian_recalculations",
                            snapshot.counters.jacobian_recalculations,
                        ),
                        ("residual_chunks", snapshot.counters.residual_chunks),
                        ("jacobian_chunks", snapshot.counters.jacobian_chunks),
                        ("conversions", snapshot.counters.conversions),
                        ("copies", snapshot.counters.copies),
                        ("damping_trials", snapshot.counters.damping_trials),
                        ("damping_rejections", snapshot.counters.damping_rejections),
                        ("linear_solves", snapshot.counters.linear_solves),
                        ("factorizations", snapshot.counters.factorizations),
                        (
                            "factorization_cache_hits",
                            snapshot.counters.factorization_cache_hits,
                        ),
                        (
                            "factorization_invalidations",
                            snapshot.counters.factorization_invalidations,
                        ),
                        ("rhs_solves", snapshot.counters.rhs_solves),
                        ("grid_refinements", snapshot.counters.grid_refinements),
                    ];
                    for (name, value) in values {
                        counters.set_item(name, value)?;
                    }
                }
                output.set_item("counters", counters)?;
                let timings = PyDict::new(py);
                let timing_fields = [
                    ("total", snapshot.timings.total),
                    ("residual", snapshot.timings.residual),
                    ("jacobian", snapshot.timings.jacobian),
                    ("linear_system", snapshot.timings.linear_system),
                    ("factorization", snapshot.timings.factorization),
                    ("rhs_solve", snapshot.timings.rhs_solve),
                    ("symbolic_operations", snapshot.timings.symbolic_operations),
                    ("grid_refinement", snapshot.timings.grid_refinement),
                ];
                for (name, value) in timing_fields {
                    if snapshot.telemetry_mode.collects_timings() {
                        timings.set_item(name, value.as_secs_f64() * 1_000.0)?;
                    }
                }
                if snapshot.telemetry_mode.collects_timings() {
                    for stage in &snapshot.timings.callback_stage_timings {
                        timings.set_item(
                            format!("callback.{:?}", stage.stage).to_ascii_lowercase(),
                            stage.elapsed.as_secs_f64() * 1_000.0,
                        )?;
                    }
                }
                output.set_item("timings_ms", timings)?;
                let scopes = PyDict::new(py);
                for (name, scope) in [
                    ("solve", &snapshot.scopes.solve),
                    ("mesh_revision", &snapshot.scopes.mesh_revision),
                    ("iteration", &snapshot.scopes.iteration),
                    ("damping_trial", &snapshot.scopes.damping_trial),
                ] {
                    let item = PyDict::new(py);
                    if snapshot.telemetry_mode.collects_counters() {
                        item.set_item("events", scope.events)?;
                    }
                    if snapshot.telemetry_mode.collects_timings() {
                        item.set_item("elapsed_ms", scope.elapsed.as_secs_f64() * 1_000.0)?;
                    }
                    scopes.set_item(name, item)?;
                }
                output.set_item("scopes", scopes)?;
                output.set_item("solve_id", snapshot.solve_id)?;
                output.set_item("log_events_dropped", snapshot.log_events_dropped)?;
            }
            BvpSolverTelemetrySnapshot::Sci(snapshot) => {
                let (mode, counters, timings) = crate::bvp::snapshot_maps(&snapshot);
                output.set_item("mode", mode)?;
                output.set_item("counters", counters)?;
                output.set_item("timings_ms", timings)?;
                output.set_item("preparation_scope", snapshot.report_preparation_scope())?;
                let scopes = snapshot
                    .timing_scopes
                    .iter()
                    .map(|scope| {
                        let item = PyDict::new(py);
                        item.set_item("stage", format!("{:?}", scope.stage))?;
                        item.set_item("kind", format!("{:?}", scope.kind))?;
                        item.set_item(
                            "parent",
                            scope.parent.map(|parent| format!("{:?}", parent)),
                        )?;
                        item.set_item("calls", scope.calls)?;
                        item.set_item("elapsed_ms", scope.elapsed_ms)?;
                        Ok(item)
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                output.set_item("scopes", scopes)?;
            }
        }
        Ok(output)
    }

    /// Execute native output actions on the last successful mesh solution.
    /// The report preserves per-action `done`/`skipped`/`failed` status.
    pub fn execute_postprocessing<'py>(
        &self,
        py: Python<'py>,
        plan: PyRef<'_, PyBvpPostprocessPlan>,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let report = self
            .inner
            .execute_postprocessing(&plan.inner)
            .map_err(map_api_error)?;
        report
            .entries
            .into_iter()
            .map(|entry| {
                let row = PyDict::new(py);
                row.set_item("action", format!("{:?}", entry.action))?;
                row.set_item(
                    "status",
                    match entry.status {
                        PostprocessStatus::Done => "done",
                        PostprocessStatus::Skipped => "skipped",
                        PostprocessStatus::Failed => "failed",
                    },
                )?;
                row.set_item("path", entry.path.map(|path| path.display().to_string()))?;
                row.set_item("message", entry.message)?;
                Ok(row)
            })
            .collect()
    }
}

#[pymethods]
impl PyBvpApiSolution {
    /// Solver family that produced this mesh/value projection.
    #[getter]
    pub fn route(&self) -> &str {
        &self.route
    }

    /// Return a detached copy of the accepted mesh, including refined nodes.
    pub fn mesh(&self) -> Vec<f64> {
        self.mesh.clone()
    }

    /// Return detached node-major state rows in the original variable order.
    pub fn values(&self) -> Vec<Vec<f64>> {
        self.values.clone()
    }
    /// Detailed BVP_sci-native result, including status and optional dense output.
    /// Other routes return `None` and retain their common mesh/value projection.
    #[getter]
    pub fn sci_details(&self) -> Option<crate::bvp::BvpSciSolution> {
        self.sci_details.clone()
    }

    /// Copy the mesh into an owned one-dimensional NumPy array.
    pub fn mesh_array<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f64>> {
        crate::numpy_support::vector(py, &self.mesh)
    }

    /// Copy state values into an owned NumPy array with shape `(nodes, states)`.
    pub fn values_array<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        crate::numpy_support::matrix(py, &self.values)
    }
}

fn parameter_pair(
    names: Option<Vec<String>>,
    values: Option<Vec<f64>>,
) -> PyResult<Option<(Vec<String>, Vec<f64>)>> {
    match (names, values) {
        (None, None) => Ok(None),
        (Some(names), Some(values))
            if names.len() == values.len() && values.iter().all(|value| value.is_finite()) =>
        {
            Ok(Some((names, values)))
        }
        (Some(_), Some(_)) => Err(InvalidConfigurationError::new_err(
            "parameters and finite parameter_values must have matching lengths",
        )),
        _ => Err(InvalidConfigurationError::new_err(
            "parameters and parameter_values must be supplied together",
        )),
    }
}

fn external_parameter_pair(
    names: Option<Vec<String>>,
    values: Option<Vec<f64>>,
) -> PyResult<Option<(Vec<String>, Vec<f64>)>> {
    match (names, values) {
        (None, None) => Ok(None),
        (Some(names), values) if !names.is_empty() => {
            let values = values.unwrap_or_else(|| vec![0.0; names.len()]);
            if names.len() != values.len() || values.iter().any(|value| !value.is_finite()) {
                return Err(InvalidConfigurationError::new_err(
                    "external_parameters and finite external_parameter_values must have matching lengths",
                ));
            }
            Ok(Some((names, values)))
        }
        (Some(_), _) => Err(InvalidConfigurationError::new_err(
            "external_parameters must contain at least one name",
        )),
        (None, Some(_)) => Err(InvalidConfigurationError::new_err(
            "external_parameter_values require external_parameters",
        )),
    }
}

fn checked_callback_matrix(
    rows: Vec<Vec<f64>>,
    expected_rows: usize,
    expected_columns: usize,
    label: &str,
) -> Result<Vec<f64>, String> {
    if rows.len() != expected_rows || rows.iter().any(|row| row.len() != expected_columns) {
        return Err(format!(
            "{label} must have shape ({expected_rows}, {expected_columns})"
        ));
    }
    let flat = rows.into_iter().flatten().collect::<Vec<_>>();
    if flat.iter().any(|value| !value.is_finite()) {
        return Err(format!("{label} must contain only finite values"));
    }
    Ok(flat)
}
