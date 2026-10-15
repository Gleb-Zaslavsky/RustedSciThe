//! Direct Python adapter for the new symbolic `BVP_sci` Lambdify route.
//!
//! This module deliberately wraps the existing Rust builder instead of
//! reimplementing collocation, mesh refinement or linear algebra in Python.
//! Boundary expressions use the documented `ya_<state>` / `yb_<state>` names;
//! their derivatives are prepared once and the numerical core receives native
//! closures. AOT uses the shared generated-backend lifecycle, while Python
//! boundary callbacks are an explicit numerical boundary route with typed
//! callback failures and native finite-difference fallback.
//! The optional SciPy singular term is exposed as a validated
//! `(singular_term_a, singular_term_matrix)` pair.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::aot::generated_config;
use rustedscithe_core::numerical::BVP_sci::new::{
    BvpSciAssembly, BvpSciExecutionPolicy, BvpSciMatrixLayout, BvpSciNewError, BvpSciOutputPolicy,
    BvpSciSingularTerm, BvpSciSolution as CoreSolution, BvpSciSolver as CoreSolver,
    BvpSciSparseFactorization, BvpSciTelemetryMode,
};

use crate::contract::SolverOptions;
use crate::errors::{
    CallbackError, CallbackFailureState, ExhaustionError, InvalidConfigurationError,
    PreparationError, SolverError, SolverPanicState, UnsupportedOptionError, annotate_native,
    catch_panic, native_variant,
};
use crate::expr::PyExpr;
use crate::lambdify::LambdifiedExpr;
use crate::telemetry::write_report;

fn finite_values(values: &[f64], label: &str) -> PyResult<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(InvalidConfigurationError::new_err(format!(
            "{label} must contain only finite values"
        )));
    }
    Ok(())
}

fn validate_names(names: &[String], label: &str, allow_empty: bool) -> PyResult<()> {
    if !allow_empty && names.is_empty() {
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

fn validate_expression_symbols(
    expressions: &[PyExpr],
    allowed: &[String],
    label: &str,
) -> PyResult<()> {
    let allowed: HashSet<&str> = allowed.iter().map(String::as_str).collect();
    for (index, expression) in expressions.iter().enumerate() {
        if let Some(unknown) = expression
            .inner
            .extract_variables()
            .1
            .iter()
            .find(|name| !allowed.contains(name.as_str()))
        {
            return Err(InvalidConfigurationError::new_err(format!(
                "{label} expression {index} references undeclared symbol '{unknown}'"
            )));
        }
    }
    Ok(())
}

fn boundary_variables(state_variables: &[String], parameters: &[String]) -> Vec<String> {
    state_variables
        .iter()
        .map(|name| format!("ya_{name}"))
        .chain(state_variables.iter().map(|name| format!("yb_{name}")))
        .chain(parameters.iter().cloned())
        .collect()
}

/// Bind boundary expressions in the declared schema order: left endpoint,
/// right endpoint, then parameters. This order must match `boundary_variables`.
fn boundary_values(ya: &[f64], yb: &[f64], parameters: &[f64]) -> Vec<f64> {
    ya.iter().chain(yb).chain(parameters).copied().collect()
}

fn finite_callback_values(values: &[f64], label: &str) -> Result<(), String> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{label} must return only finite values"));
    }
    Ok(())
}

/// Validate and flatten a Python callback matrix in row-major order, which is
/// the layout expected by nalgebra's `from_row_slice` at the callback boundary.
fn flatten_callback_matrix(
    rows: Vec<Vec<f64>>,
    expected_rows: usize,
    expected_columns: usize,
    label: &str,
) -> Result<Vec<f64>, String> {
    if rows.len() != expected_rows || rows.iter().any(|row| row.len() != expected_columns) {
        return Err(format!(
            "{label} must return a {expected_rows}x{expected_columns} matrix"
        ));
    }
    let values = rows.into_iter().flatten().collect::<Vec<_>>();
    finite_callback_values(&values, label)?;
    Ok(values)
}

fn validate_python_callback(callback: &Py<PyAny>, label: &str) -> PyResult<()> {
    Python::attach(|py| {
        if callback.bind(py).is_callable() {
            Ok(())
        } else {
            Err(InvalidConfigurationError::new_err(format!(
                "{label} must be callable"
            )))
        }
    })
}

fn compile_expressions(
    expressions: &[PyExpr],
    variables: &[String],
) -> PyResult<Vec<LambdifiedExpr>> {
    expressions
        .iter()
        .map(|expression| LambdifiedExpr::compile(expression, Some(variables.to_vec())))
        .collect()
}

fn map_policy(value: &str) -> PyResult<BvpSciExecutionPolicy> {
    match value {
        "sequential" => Ok(BvpSciExecutionPolicy::Sequential),
        "parallel" => Ok(BvpSciExecutionPolicy::Parallel { min_work: 1 }),
        "auto" => Ok(BvpSciExecutionPolicy::Auto { min_work: 1 }),
        other => Err(InvalidConfigurationError::new_err(format!(
            "unknown BVP execution policy '{other}'"
        ))),
    }
}

fn map_telemetry(value: &str) -> PyResult<BvpSciTelemetryMode> {
    match value {
        "off" => Ok(BvpSciTelemetryMode::Off),
        "counters" => Ok(BvpSciTelemetryMode::Counters),
        "timings" => Ok(BvpSciTelemetryMode::Timings),
        other => Err(InvalidConfigurationError::new_err(format!(
            "unknown BVP telemetry mode '{other}'"
        ))),
    }
}

fn map_output(value: &str) -> PyResult<BvpSciOutputPolicy> {
    match value {
        "final" | "final_only" => Ok(BvpSciOutputPolicy::FinalOnly),
        "mesh" | "mesh_and_values" => Ok(BvpSciOutputPolicy::MeshAndValues),
        "dense" | "dense_output" => Ok(BvpSciOutputPolicy::DenseOutput),
        other => Err(UnsupportedOptionError::new_err(format!(
            "unknown BVP output policy '{other}'; expected final, mesh or dense"
        ))),
    }
}

fn map_layout(options: &SolverOptions) -> PyResult<BvpSciMatrixLayout> {
    match options.matrix_layout() {
        "dense" => Ok(BvpSciMatrixLayout::Dense),
        "sparse" => Ok(BvpSciMatrixLayout::Sparse),
        "banded" => Ok(BvpSciMatrixLayout::Banded {
            lower: options.band_lower().ok_or_else(|| {
                InvalidConfigurationError::new_err(
                    "Banded BVP_sci requires options.set_banded_width(lower, upper)",
                )
            })?,
            upper: options.band_upper().ok_or_else(|| {
                InvalidConfigurationError::new_err(
                    "Banded BVP_sci requires options.set_banded_width(lower, upper)",
                )
            })?,
        }),
        other => Err(UnsupportedOptionError::new_err(format!(
            "unsupported BVP matrix layout '{other}'"
        ))),
    }
}

fn map_sparse_factorization(options: &SolverOptions) -> PyResult<BvpSciSparseFactorization> {
    match options.sparse_factorization() {
        "standard_faer" => Ok(BvpSciSparseFactorization::StandardFaer),
        "reusable_faer_experimental" => Ok(BvpSciSparseFactorization::ReusableFaerExperimental),
        "reusable_faer_simplicial_experimental" => {
            Ok(BvpSciSparseFactorization::ReusableFaerSimplicialExperimental)
        }
        "superlu" => Ok(BvpSciSparseFactorization::SuperLu),
        other => Err(UnsupportedOptionError::new_err(format!(
            "unsupported BVP sparse factorization '{other}'"
        ))),
    }
}

fn map_error(error: BvpSciNewError) -> PyErr {
    let native_kind = native_variant(&error);
    let message = error.to_string();
    annotate_native(
        match error {
            BvpSciNewError::InvalidConfiguration(_) | BvpSciNewError::ContinuationMismatch => {
                InvalidConfigurationError::new_err(message)
            }
            BvpSciNewError::UnsupportedRoute(_) => UnsupportedOptionError::new_err(message),
            BvpSciNewError::SymbolicPreparation(_)
            | BvpSciNewError::AotPreparation { .. }
            | BvpSciNewError::AotHandoff { .. } => PreparationError::new_err(message),
            BvpSciNewError::Callback { .. }
            | BvpSciNewError::NonFinite { .. }
            | BvpSciNewError::AotCallback { .. } => CallbackError::new_err(message),
            BvpSciNewError::MaxNodesExceeded { .. }
            | BvpSciNewError::MeshRefinementLimit { .. } => ExhaustionError::new_err(message),
            BvpSciNewError::ShapeMismatch { .. }
            | BvpSciNewError::NewtonFailure { .. }
            | BvpSciNewError::SingularJacobian
            | BvpSciNewError::BoundaryToleranceNotMet { .. }
            | BvpSciNewError::LinearBackend { .. }
            | BvpSciNewError::LinearFactorization { .. }
            | BvpSciNewError::LinearSolve { .. }
            | BvpSciNewError::AotRuntime { .. }
            | BvpSciNewError::OutputOutOfDomain
            | BvpSciNewError::OutputUnavailable => SolverError::new_err(message),
        },
        "solve",
        "bvp_sci",
        "native",
        &native_kind,
    )
}

fn default_options() -> PyResult<SolverOptions> {
    SolverOptions::new(None, None, None, None, None, None, None, None, None)
}

fn snapshot_maps(
    snapshot: &rustedscithe_core::numerical::BVP_sci::new::BvpSciTelemetrySnapshot,
) -> (String, HashMap<String, u64>, HashMap<String, f64>) {
    let mode = format!("{:?}", snapshot.mode).to_ascii_lowercase();
    let counters = [
        ("rhs_calls", snapshot.rhs_calls),
        ("boundary_calls", snapshot.boundary_calls),
        ("jacobian_calls", snapshot.jacobian_calls),
        ("boundary_jacobian_calls", snapshot.boundary_jacobian_calls),
        (
            "finite_difference_probes",
            snapshot.finite_difference_probes,
        ),
        ("newton_iterations", snapshot.newton_iterations),
        ("mesh_refinements", snapshot.mesh_refinements),
        ("linear_assemblies", snapshot.linear_assemblies),
        ("factorizations", snapshot.factorizations),
        ("linear_solves", snapshot.linear_solves),
        ("parameter_rebinds", snapshot.parameter_rebinds),
        ("continuation_solves", snapshot.continuation_solves),
        ("restarts", snapshot.restarts),
        ("parallel_dispatches", snapshot.parallel_dispatches),
        ("sequential_dispatches", snapshot.sequential_dispatches),
        (
            "singular_term_applications",
            snapshot.singular_term_applications,
        ),
    ]
    .into_iter()
    .filter(|(_, value)| *value != 0)
    .map(|(name, value)| (name.to_owned(), value))
    .collect();
    let timings = [
        ("preparation_ms", snapshot.preparation_ms),
        ("expr_to_atom_ms", snapshot.expr_to_atom_ms),
        ("symbolic_jacobian_ms", snapshot.symbolic_jacobian_ms),
        ("lowering_ms", snapshot.lowering_ms),
        (
            "evaluator_compilation_ms",
            snapshot.evaluator_compilation_ms,
        ),
        ("rhs_callback_ms", snapshot.rhs_callback_ms),
        ("boundary_callback_ms", snapshot.boundary_callback_ms),
        ("jacobian_callback_ms", snapshot.jacobian_callback_ms),
        ("residual_evaluation_ms", snapshot.residual_evaluation_ms),
        ("jacobian_evaluation_ms", snapshot.jacobian_evaluation_ms),
        ("collocation_ms", snapshot.collocation_ms),
        ("mesh_refinement_ms", snapshot.mesh_refinement_ms),
        ("newton_ms", snapshot.newton_ms),
        ("linear_assembly_ms", snapshot.linear_assembly_ms),
        ("factorization_ms", snapshot.factorization_ms),
        ("linear_solve_ms", snapshot.linear_solve_ms),
        ("full_solve_ms", snapshot.full_solve_ms),
        ("full_solve_total_ms", snapshot.full_solve_total_ms),
        ("sparse_csc_assembly_ms", snapshot.sparse_csc_assembly_ms),
    ]
    .into_iter()
    .filter_map(|(name, value)| value.map(|value| (name.to_owned(), value)))
    .collect();
    (mode, counters, timings)
}

/// Detached solution returned by the direct BVP_sci Python API.
///
/// Values use node-major orientation `(mesh_nodes, state_dimension)` for both
/// nested lists and NumPy arrays. The result owns the refined mesh, endpoint
/// residual diagnostics, optional dense-output data, and telemetry snapshot.
#[pyclass(name = "BvpSciSolution", module = "rustedscithe")]
#[derive(Clone)]
pub struct BvpSciSolution {
    mesh: Vec<f64>,
    values: Vec<f64>,
    parameters: Vec<f64>,
    residual_norm: f64,
    interval_residuals: Vec<f64>,
    status: String,
    status_code: u8,
    message: String,
    iterations: usize,
    dense_output: Option<rustedscithe_core::numerical::BVP_sci::new::BvpSciDenseOutput>,
    telemetry_mode: String,
    counters: HashMap<String, u64>,
    timings_ms: HashMap<String, f64>,
}

impl BvpSciSolution {
    fn from_native(
        native: CoreSolution,
        snapshot: rustedscithe_core::numerical::BVP_sci::new::BvpSciTelemetrySnapshot,
    ) -> Self {
        let (telemetry_mode, counters, timings_ms) = snapshot_maps(&snapshot);
        Self {
            mesh: native.x,
            values: native.y,
            parameters: native.parameters,
            residual_norm: native.residual_norm,
            interval_residuals: native.interval_residuals,
            status: format!("{:?}", native.status).to_ascii_lowercase(),
            status_code: native.status.code(),
            message: native.message,
            iterations: native.iterations,
            dense_output: native.dense_output,
            telemetry_mode,
            counters,
            timings_ms,
        }
    }
}

#[pymethods]
impl BvpSciSolution {
    /// Accepted mesh, detached from the solver workspace.
    pub fn mesh(&self) -> Vec<f64> {
        self.mesh.clone()
    }

    /// Node-major state values: `values[node * state_dimension + component]`.
    pub fn values_flat(&self) -> Vec<f64> {
        self.values.clone()
    }

    /// Return values as rows of mesh nodes and columns of state components.
    pub fn values(&self) -> Vec<Vec<f64>> {
        let dimension = self.state_dimension();
        self.values
            .chunks(dimension)
            .map(|row| row.to_vec())
            .collect()
    }

    /// Owned NumPy array with shape `(mesh_nodes, state_dimension)`.
    pub fn values_array<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        crate::numpy_support::matrix(py, &self.values())
    }

    /// Owned NumPy array containing the accepted mesh.
    pub fn mesh_array<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f64>> {
        crate::numpy_support::vector(py, &self.mesh)
    }

    #[getter]
    /// Bound parameter values used for this detached BVP solution.
    pub fn parameters(&self) -> Vec<f64> {
        self.parameters.clone()
    }

    #[getter]
    /// Infinity norm of the final boundary residual.
    pub fn residual_norm(&self) -> f64 {
        self.residual_norm
    }

    /// Return the collocation residual norm associated with each mesh interval.
    pub fn interval_residuals(&self) -> Vec<f64> {
        self.interval_residuals.clone()
    }

    #[getter]
    /// Human-readable solver status for the completed BVP solve.
    pub fn status(&self) -> &str {
        &self.status
    }

    #[getter]
    /// Stable numeric status code returned by the native BVP solver.
    pub fn status_code(&self) -> u8 {
        self.status_code
    }

    #[getter]
    /// Human-readable status explanation from the native BVP result.
    pub fn message(&self) -> &str {
        &self.message
    }

    #[getter]
    /// Number of nonlinear iterations reported by the native solver.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    #[getter]
    /// Telemetry collection level used for this solution snapshot.
    pub fn telemetry_mode(&self) -> &str {
        &self.telemetry_mode
    }

    /// Return discrete stage counters from the completed solve.
    pub fn telemetry_counters(&self) -> HashMap<String, u64> {
        self.counters.clone()
    }

    /// Return stage timings in milliseconds; scopes may overlap.
    pub fn telemetry_timings_ms(&self) -> HashMap<String, f64> {
        self.timings_ms.clone()
    }

    /// Render the telemetry snapshot as a compact deterministic text report.
    pub fn telemetry_report(&self) -> String {
        let mut counters = self.counters.iter().collect::<Vec<_>>();
        counters.sort_by_key(|(name, _)| *name);
        let mut timings = self.timings_ms.iter().collect::<Vec<_>>();
        timings.sort_by_key(|(name, _)| *name);
        format!(
            "BVP_sci telemetry={}; counters={}; timings_ms={}",
            self.telemetry_mode,
            counters
                .into_iter()
                .map(|(name, value)| format!("{name}={value}"))
                .collect::<Vec<_>>()
                .join(","),
            timings
                .into_iter()
                .map(|(name, value)| format!("{name}={value:.6}"))
                .collect::<Vec<_>>()
                .join(","),
        )
    }

    /// Write only the compact telemetry report to `path`.
    pub fn write_telemetry_report(&self, path: &str) -> PyResult<()> {
        write_report(path, &self.telemetry_report())
    }

    /// Number of state components stored at each mesh node.
    pub fn state_dimension(&self) -> usize {
        self.values.len() / self.mesh.len().max(1)
    }

    /// Evaluate the retained cubic output without exposing Rust state.
    pub fn sample(&self, x: f64) -> PyResult<Vec<f64>> {
        catch_panic("postprocess", "bvp_sci", "dense_output", || {
            self.dense_output
                .as_ref()
                .ok_or_else(|| SolverError::new_err("dense output was not requested"))?
                .evaluate(x)
                .map_err(map_error)
        })
    }
}

/// Prepared symbolic BVP_sci solver.
///
/// The constructor prepares RHS and boundary expressions once. Use
/// `set_parameters` followed by `solve` for continuation, or `restart` to
/// supply a new mesh and initial state without rebuilding the symbolic plan.
#[pyclass(name = "BvpSciSolver", module = "rustedscithe", unsendable)]
pub struct BvpSciSolver {
    solver: CoreSolver,
    panic_state: SolverPanicState,
    callback_failure: CallbackFailureState,
}

#[pymethods]
impl BvpSciSolver {
    /// Construct and prepare a symbolic or callback BVP_sci problem.
    ///
    /// `expressions` contains one ODE RHS per state variable and
    /// `boundary_expressions` contains the endpoint residual equations.
    /// `mesh` must be strictly increasing; `initial_state` is node-major with
    /// shape `(mesh_nodes, state_dimension)`. Optional parameter symbols and
    /// values enable continuation. `options` selects symbolic frontend,
    /// Lambdify/AOT execution, matrix layout, factorization and telemetry.
    /// Numeric controls configure residual tolerances, mesh/Newton budgets and
    /// output retention. Boundary callbacks may replace the symbolic boundary
    /// route; singular-term inputs must be supplied as a compatible pair.
    #[new]
    #[pyo3(signature=(expressions, state_variables, boundary_expressions, mesh, initial_state, *, parameters=None, parameter_values=None, independent_variable="x", options=None, tolerance=1e-3, boundary_tolerance=None, max_nodes=1000, max_newton_iterations=8, max_mesh_refinements=10, max_backtracking_steps=4, output="mesh", boundary_jacobian="symbolic", singular_term_a=None, singular_term_matrix=None, boundary_callback=None, boundary_jacobian_callback=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        expressions: Vec<PyExpr>,
        state_variables: Vec<String>,
        boundary_expressions: Vec<PyExpr>,
        mesh: Vec<f64>,
        initial_state: Vec<f64>,
        parameters: Option<Vec<String>>,
        parameter_values: Option<Vec<f64>>,
        independent_variable: &str,
        options: Option<PyRef<'_, SolverOptions>>,
        tolerance: f64,
        boundary_tolerance: Option<f64>,
        max_nodes: usize,
        max_newton_iterations: usize,
        max_mesh_refinements: usize,
        max_backtracking_steps: usize,
        output: &str,
        boundary_jacobian: &str,
        singular_term_a: Option<f64>,
        singular_term_matrix: Option<Vec<f64>>,
        boundary_callback: Option<Py<PyAny>>,
        boundary_jacobian_callback: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        validate_names(&state_variables, "state_variables", false)?;
        if independent_variable.trim().is_empty() {
            return Err(InvalidConfigurationError::new_err(
                "independent_variable must not be empty",
            ));
        }
        let parameters = parameters.unwrap_or_default();
        validate_names(&parameters, "parameters", true)?;
        if state_variables
            .iter()
            .any(|name| name == independent_variable)
            || parameters.iter().any(|name| name == independent_variable)
        {
            return Err(InvalidConfigurationError::new_err(
                "independent_variable must not collide with state_variables or parameters",
            ));
        }
        if expressions.len() != state_variables.len() {
            return Err(InvalidConfigurationError::new_err(format!(
                "expected one RHS expression per state variable, got {} expressions for {} variables",
                expressions.len(),
                state_variables.len()
            )));
        }
        let rhs_allowed = std::iter::once(independent_variable.to_owned())
            .chain(state_variables.iter().cloned())
            .chain(parameters.iter().cloned())
            .collect::<Vec<_>>();
        validate_expression_symbols(&expressions, &rhs_allowed, "RHS")?;

        let state_dimension = state_variables.len();
        let parameter_dimension = parameters.len();
        let boundary_dimension = state_dimension + parameter_dimension;
        let callback_boundary = boundary_callback.is_some();
        if callback_boundary && !boundary_expressions.is_empty() {
            return Err(InvalidConfigurationError::new_err(
                "boundary_expressions must be empty when boundary_callback is provided",
            ));
        }
        if !callback_boundary && boundary_expressions.len() != boundary_dimension {
            return Err(InvalidConfigurationError::new_err(format!(
                "expected {} boundary expressions, got {}",
                boundary_dimension,
                boundary_expressions.len()
            )));
        }
        let boundary_variables = boundary_variables(&state_variables, &parameters);
        if !callback_boundary {
            validate_expression_symbols(&boundary_expressions, &boundary_variables, "boundary")?;
        }
        if let Some(callback) = &boundary_callback {
            validate_python_callback(callback, "boundary_callback")?;
        }
        if let Some(callback) = &boundary_jacobian_callback {
            validate_python_callback(callback, "boundary_jacobian_callback")?;
            if !callback_boundary {
                return Err(InvalidConfigurationError::new_err(
                    "boundary_jacobian_callback requires boundary_callback",
                ));
            }
            if boundary_jacobian == "finite_difference" {
                return Err(InvalidConfigurationError::new_err(
                    "boundary_jacobian_callback cannot be combined with boundary_jacobian='finite_difference'",
                ));
            }
        }
        if mesh.len() < 2 || !mesh.windows(2).all(|pair| pair[1] > pair[0]) {
            return Err(InvalidConfigurationError::new_err(
                "mesh must contain at least two strictly increasing nodes",
            ));
        }
        finite_values(&mesh, "mesh")?;
        finite_values(&initial_state, "initial_state")?;
        let singular_term = match (singular_term_a, singular_term_matrix) {
            (None, None) => None,
            (Some(endpoint), Some(values)) => {
                if !endpoint.is_finite() {
                    return Err(InvalidConfigurationError::new_err(
                        "singular_term_a must be finite",
                    ));
                }
                if endpoint != mesh[0] {
                    return Err(InvalidConfigurationError::new_err(format!(
                        "singular_term_a ({endpoint}) must equal the left mesh endpoint ({})",
                        mesh[0]
                    )));
                }
                finite_values(&values, "singular_term_matrix")?;
                Some(BvpSciSingularTerm::new(state_variables.len(), values).map_err(map_error)?)
            }
            (Some(_), None) => {
                return Err(InvalidConfigurationError::new_err(
                    "singular_term_a and singular_term_matrix must be provided together",
                ));
            }
            (None, Some(_)) => {
                return Err(InvalidConfigurationError::new_err(
                    "singular_term_a and singular_term_matrix must be provided together",
                ));
            }
        };
        let expected_state = mesh
            .len()
            .checked_mul(state_variables.len())
            .ok_or_else(|| {
                InvalidConfigurationError::new_err("mesh/state dimensions overflow usize")
            })?;
        if initial_state.len() != expected_state {
            return Err(InvalidConfigurationError::new_err(format!(
                "initial_state must contain {expected_state} node-major values, got {}",
                initial_state.len()
            )));
        }
        let parameter_values = parameter_values.unwrap_or_else(|| vec![0.0; parameters.len()]);
        if parameter_values.len() != parameters.len() {
            return Err(InvalidConfigurationError::new_err(format!(
                "parameter_values must contain {} values, got {}",
                parameters.len(),
                parameter_values.len()
            )));
        }
        finite_values(&parameter_values, "parameter_values")?;
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(InvalidConfigurationError::new_err(
                "tolerance must be finite and greater than zero",
            ));
        }
        if let Some(value) = boundary_tolerance {
            if !value.is_finite() || value <= 0.0 {
                return Err(InvalidConfigurationError::new_err(
                    "boundary_tolerance must be finite and greater than zero",
                ));
            }
        }
        if max_nodes < mesh.len() || max_newton_iterations == 0 || max_mesh_refinements == 0 {
            return Err(InvalidConfigurationError::new_err(
                "BVP iteration budgets must be positive and max_nodes must cover the initial mesh",
            ));
        }

        let options = options
            .map(|value| value.clone())
            .unwrap_or(default_options()?);
        options.validate_for("bvp_sci")?;
        let assembly = match options.frontend() {
            "expr_legacy" => BvpSciAssembly::ExprLegacy,
            "atom_native" => BvpSciAssembly::AtomViewNative,
            other => {
                return Err(UnsupportedOptionError::new_err(format!(
                    "unsupported BVP frontend '{other}'"
                )));
            }
        };
        let layout = map_layout(&options)?;
        let sparse_factorization = map_sparse_factorization(&options)?;
        if !matches!(layout, BvpSciMatrixLayout::Sparse)
            && !matches!(
                sparse_factorization,
                BvpSciSparseFactorization::StandardFaer
            )
        {
            return Err(UnsupportedOptionError::new_err(
                "non-standard sparse factorization requires matrix_layout='sparse'",
            ));
        }
        let policy = map_policy(options.execution_policy())?;
        let telemetry_mode = map_telemetry(options.telemetry())?;
        let output_policy = map_output(output)?;
        if boundary_jacobian != "symbolic" && boundary_jacobian != "finite_difference" {
            return Err(UnsupportedOptionError::new_err(
                "boundary_jacobian must be 'symbolic' or 'finite_difference'",
            ));
        }

        let rhs = expressions
            .iter()
            .map(|expression| expression.inner.clone())
            .collect();
        let mut builder = CoreSolver::builder(rhs, state_variables.clone())
            .with_frontend(assembly)
            .with_independent_variable(independent_variable)
            .with_parameter_names(parameters.clone())
            .with_parameters(parameter_values)
            .with_mesh_and_initial_state(mesh, initial_state)
            .with_matrix_layout(layout)
            .with_tolerance(tolerance)
            .with_limits(max_nodes, max_newton_iterations, max_mesh_refinements)
            .with_backtracking_limit(max_backtracking_steps)
            .with_execution_policy(policy)
            .with_telemetry(telemetry_mode)
            .with_output_policy(output_policy);
        if options.execution() == "aot" {
            builder = builder.with_aot_generated_backend(generated_config(&options)?);
        }
        builder = builder.with_sparse_factorization(sparse_factorization);
        if let Some(value) = boundary_tolerance {
            builder = builder.with_boundary_tolerance(value);
        }
        if let Some(singular_term) = singular_term {
            builder = builder.with_singular_term(singular_term);
        }

        let callback_failure = CallbackFailureState::default();
        if let Some(callback) = boundary_callback {
            let boundary_failure = callback_failure.clone();
            builder = builder.with_boundary_callback(move |ya, yb, parameters, output| {
                let result = Python::attach(|py| -> PyResult<Vec<f64>> {
                    callback
                        .bind(py)
                        .call1((ya.to_vec(), yb.to_vec(), parameters.to_vec()))?
                        .extract()
                });
                match result {
                    Ok(values) if values.len() == output.len() => {
                        finite_callback_values(&values, "boundary_callback")?;
                        output.copy_from_slice(&values);
                        Ok(())
                    }
                    Ok(values) => Err(format!(
                        "boundary_callback must return {} values, got {}",
                        output.len(),
                        values.len()
                    )),
                    Err(error) => {
                        let message = format!("boundary_callback raised: {error}");
                        boundary_failure.record(message.clone(), Some(error));
                        Err(message)
                    }
                }
            });
            if let Some(callback) = boundary_jacobian_callback {
                let jacobian_failure = callback_failure.clone();
                builder = builder.with_boundary_jacobian(
                    move |ya, yb, parameters, out_ya, out_yb, out_p| {
                        let result = Python::attach(
                            |py| -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>)> {
                                callback
                                    .bind(py)
                                    .call1((ya.to_vec(), yb.to_vec(), parameters.to_vec()))?
                                    .extract()
                            },
                        );
                        let (dya, dyb, dp) = match result {
                            Ok(values) => values,
                            Err(error) => {
                                let message = format!("boundary_jacobian_callback raised: {error}");
                                jacobian_failure.record(message.clone(), Some(error));
                                return Err(message);
                            }
                        };
                        let dya = flatten_callback_matrix(
                            dya,
                            boundary_dimension,
                            state_dimension,
                            "boundary_jacobian_callback dya",
                        )?;
                        let dyb = flatten_callback_matrix(
                            dyb,
                            boundary_dimension,
                            state_dimension,
                            "boundary_jacobian_callback dyb",
                        )?;
                        let dp = flatten_callback_matrix(
                            dp,
                            boundary_dimension,
                            parameter_dimension,
                            "boundary_jacobian_callback dp",
                        )?;
                        out_ya.copy_from_slice(&dya);
                        out_yb.copy_from_slice(&dyb);
                        out_p.copy_from_slice(&dp);
                        Ok(())
                    },
                );
            }
        } else {
            let boundary_compiled = Arc::new(compile_expressions(
                &boundary_expressions,
                &boundary_variables,
            )?);
            let boundary_values_variables = boundary_variables.clone();
            let residuals = Arc::clone(&boundary_compiled);
            builder = builder.with_boundary_callback(move |ya, yb, parameters, output| {
                let values = boundary_values(ya, yb, parameters);
                if values.len() != boundary_values_variables.len() {
                    return Err(
                        "boundary input dimension does not match the prepared schema".to_owned(),
                    );
                }
                for (slot, expression) in output.iter_mut().zip(residuals.iter()) {
                    *slot = expression.evaluate_native(&values)?;
                }
                Ok(())
            });
        }

        if !callback_boundary && boundary_jacobian == "symbolic" {
            let mut dya = Vec::with_capacity(boundary_dimension * state_variables.len());
            let mut dyb = Vec::with_capacity(boundary_dimension * state_variables.len());
            let mut dp = Vec::with_capacity(boundary_dimension * parameters.len());
            for expression in &boundary_expressions {
                for state in &state_variables {
                    dya.push(PyExpr::from_inner(
                        expression.inner.diff(&format!("ya_{state}")),
                    ));
                }
                for state in &state_variables {
                    dyb.push(PyExpr::from_inner(
                        expression.inner.diff(&format!("yb_{state}")),
                    ));
                }
                for parameter in &parameters {
                    dp.push(PyExpr::from_inner(expression.inner.diff(parameter)));
                }
            }
            let dya = Arc::new(compile_expressions(&dya, &boundary_variables)?);
            let dyb = Arc::new(compile_expressions(&dyb, &boundary_variables)?);
            let dp = Arc::new(compile_expressions(&dp, &boundary_variables)?);
            builder =
                builder.with_boundary_jacobian(move |ya, yb, parameters, out_ya, out_yb, out_p| {
                    let values = boundary_values(ya, yb, parameters);
                    for (slot, expression) in out_ya.iter_mut().zip(dya.iter()) {
                        *slot = expression.evaluate_native(&values)?;
                    }
                    for (slot, expression) in out_yb.iter_mut().zip(dyb.iter()) {
                        *slot = expression.evaluate_native(&values)?;
                    }
                    for (slot, expression) in out_p.iter_mut().zip(dp.iter()) {
                        *slot = expression.evaluate_native(&values)?;
                    }
                    Ok(())
                });
        }

        let route = format!(
            "{}:{}:{}",
            options.frontend(),
            options.execution(),
            options.matrix_layout()
        );
        catch_panic("construct", "bvp_sci", &route, || {
            builder
                .build()
                .map(|solver| Self {
                    solver,
                    panic_state: SolverPanicState::default(),
                    callback_failure,
                })
                .map_err(map_error)
        })
    }

    /// Rebind numeric parameters while retaining the prepared symbolic plan.
    /// Call `solve()` afterward to solve the updated BVP.
    pub fn set_parameters(&mut self, values: Vec<f64>) -> PyResult<()> {
        finite_values(&values, "parameter values")?;
        let route = self.frontend();
        let panic_state = self.panic_state.clone();
        panic_state.run("continuation", "bvp_sci", &route, || {
            self.solver.set_parameters(values).map_err(map_error)
        })
    }

    /// Replace the mesh and node-major initial guess without rebuilding the
    /// symbolic model. The new mesh must be strictly increasing.
    pub fn restart(&mut self, mesh: Vec<f64>, initial_state: Vec<f64>) -> PyResult<()> {
        if mesh.len() < 2 || !mesh.windows(2).all(|pair| pair[1] > pair[0]) {
            return Err(InvalidConfigurationError::new_err(
                "restart mesh must contain at least two strictly increasing nodes",
            ));
        }
        finite_values(&mesh, "restart mesh")?;
        finite_values(&initial_state, "restart initial_state")?;
        let route = self.frontend();
        let panic_state = self.panic_state.clone();
        panic_state.run("restart", "bvp_sci", &route, || {
            self.solver.restart(mesh, initial_state).map_err(map_error)
        })
    }

    /// Solve the configured BVP and return refined values and diagnostics.
    pub fn solve(&mut self) -> PyResult<BvpSciSolution> {
        let route = self.frontend();
        let panic_state = self.panic_state.clone();
        let callback_failure = self.callback_failure.clone();
        panic_state.run("solve", "bvp_sci", &route, || {
            callback_failure.clear();
            let native = self
                .solver
                .solve()
                .map_err(|error| callback_failure.attach_cause(map_error(error)))?;
            let snapshot = self.solver.plan().telemetry_snapshot();
            Ok(BvpSciSolution::from_native(native, snapshot))
        })
    }

    /// Return the current mesh, including any adaptive refinement.
    pub fn mesh(&self) -> Vec<f64> {
        self.solver.state().0.to_vec()
    }

    /// Return the currently bound parameter values.
    pub fn parameters(&self) -> Vec<f64> {
        self.solver.state().2.to_vec()
    }

    #[getter]
    /// Symbolic frontend used by the prepared BVP solver.
    pub fn frontend(&self) -> String {
        match self.solver.plan().assembly() {
            BvpSciAssembly::ExprLegacy => "expr_legacy".to_owned(),
            BvpSciAssembly::AtomViewNative => "atom_native".to_owned(),
            BvpSciAssembly::Numerical => "numerical".to_owned(),
        }
    }

    #[getter]
    /// Linear-system layout used by the prepared BVP solver.
    pub fn matrix_layout(&self) -> String {
        match self.solver.options().matrix_layout {
            BvpSciMatrixLayout::Dense => "dense".to_owned(),
            BvpSciMatrixLayout::Sparse => "sparse".to_owned(),
            BvpSciMatrixLayout::Banded { .. } => "banded".to_owned(),
        }
    }

    #[getter]
    /// Sparse factorization implementation selected by the solver.
    #[allow(deprecated)]
    pub fn sparse_factorization(&self) -> String {
        match self.solver.sparse_factorization() {
            BvpSciSparseFactorization::StandardFaer => "standard_faer",
            BvpSciSparseFactorization::ReusableFaerExperimental => "reusable_faer_experimental",
            BvpSciSparseFactorization::ReusableFaerSimplicialExperimental => {
                "reusable_faer_simplicial_experimental"
            }
            BvpSciSparseFactorization::SuperLu | BvpSciSparseFactorization::SuperLuExperimental => {
                "superlu"
            }
        }
        .to_owned()
    }
}
