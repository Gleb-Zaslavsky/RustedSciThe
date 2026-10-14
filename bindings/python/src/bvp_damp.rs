//! Direct Python adapter for the established damped Newton BVP solver.
//!
//! The adapter intentionally owns only Python-facing validation, option
//! translation and detached result data.  Discretization, symbolic lowering,
//! generated-backend selection and the nonlinear iteration remain in the
//! native `NRBVP` implementation.

use std::collections::HashMap;

use nalgebra::DMatrix;
use pyo3::prelude::*;

use rustedscithe_core::numerical::BVP_Damp::BvpLambdifyTelemetryMode;
use rustedscithe_core::numerical::BVP_Damp::NR_Damp_solver_damped::{
    AdaptiveGridConfig, DampedBvpStatistics, DampedSolverOptions, NRBVP, SolverParams,
};
use rustedscithe_core::numerical::BVP_Damp::generated_solver_handoff::{
    AotBuildPolicy, AotBuildProfile, AotExecutionPolicy, GeneratedBackendConfig,
};
use rustedscithe_core::numerical::BVP_Damp::grid_api::GridRefinementMethod;
use rustedscithe_core::numerical::BVP_Damp::telemetry::{
    BvpLoggingConfig, BvpLoggingMode, BvpTelemetryMode,
};
use rustedscithe_core::somelinalg::banded::{
    FallbackPolicy, LinearSolverConfig, LinearSolverPolicy,
};
use rustedscithe_core::symbolic::bvp::legacy::BvpBackendIntegrationError;
use rustedscithe_core::symbolic::bvp::legacy::BvpSymbolicAssemblyBackend;
use rustedscithe_core::symbolic::bvp::telemetry::BvpLambdifyExecutionPolicy;
use rustedscithe_core::symbolic::codegen::codegen_aot_driver::AotCodegenBackend;
use rustedscithe_core::symbolic::codegen::codegen_aot_resolution::AotResolver;
use rustedscithe_core::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
use rustedscithe_core::symbolic::codegen::codegen_orchestrator::ParallelExecutorConfig;
use rustedscithe_core::symbolic::codegen::codegen_provider_api::MatrixBackend;

use crate::contract::SolverOptions;
use crate::errors::{
    InvalidConfigurationError, PreparationError, SolverError, UnsupportedOptionError, annotate,
    catch_panic,
};
use crate::expr::PyExpr;
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
    if (!allow_empty && names.is_empty()) || names.iter().any(|name| name.trim().is_empty()) {
        return Err(InvalidConfigurationError::new_err(format!(
            "{label} must contain at least one non-empty name"
        )));
    }
    let mut seen = std::collections::HashSet::with_capacity(names.len());
    if names.iter().any(|name| !seen.insert(name.as_str())) {
        return Err(InvalidConfigurationError::new_err(format!(
            "{label} must contain unique names"
        )));
    }
    Ok(())
}

fn map_native_error(error: BvpBackendIntegrationError) -> PyErr {
    let message = error.to_string();
    annotate(
        match error {
            BvpBackendIntegrationError::InvalidProblem { .. }
            | BvpBackendIntegrationError::InvalidSolverConfiguration { .. }
            | BvpBackendIntegrationError::InvalidLogLevel { .. }
            | BvpBackendIntegrationError::LogFileCreationFailed { .. } => {
                InvalidConfigurationError::new_err(message)
            }
            BvpBackendIntegrationError::CompiledAotRequiredButUnavailable { .. }
            | BvpBackendIntegrationError::AutomaticAotBuildRequested { .. }
            | BvpBackendIntegrationError::AutomaticAotRebuildRequested { .. }
            | BvpBackendIntegrationError::AutomaticAotBuildFailed { .. }
            | BvpBackendIntegrationError::CompiledAotRuntimeUnavailable { .. }
            | BvpBackendIntegrationError::AtomAotPlanPreparationFailed { .. }
            | BvpBackendIntegrationError::AotLifecycleFailure { .. } => {
                PreparationError::new_err(message)
            }
            _ => SolverError::new_err(message),
        },
        "solve",
        "bvp_damp",
        "native",
    )
}

fn map_telemetry(value: &str) -> PyResult<(BvpTelemetryMode, BvpLambdifyTelemetryMode)> {
    match value {
        "off" => Ok((BvpTelemetryMode::Off, BvpLambdifyTelemetryMode::Off)),
        "counters" => Ok((
            BvpTelemetryMode::Counters,
            BvpLambdifyTelemetryMode::Counters,
        )),
        "timings" => Ok((
            BvpTelemetryMode::Detailed,
            BvpLambdifyTelemetryMode::Detailed,
        )),
        other => Err(InvalidConfigurationError::new_err(format!(
            "unknown BVP_Damp telemetry mode '{other}'"
        ))),
    }
}

fn matrix_backend(value: &str) -> PyResult<(String, MatrixBackend)> {
    match value {
        "dense" => Ok(("Dense".to_owned(), MatrixBackend::Dense)),
        "sparse" => Ok(("Sparse".to_owned(), MatrixBackend::SparseCol)),
        "banded" => Ok(("Banded".to_owned(), MatrixBackend::Banded)),
        other => Err(UnsupportedOptionError::new_err(format!(
            "unsupported BVP_Damp matrix layout '{other}'"
        ))),
    }
}

fn frontend(value: &str) -> PyResult<BvpSymbolicAssemblyBackend> {
    match value {
        "expr_legacy" => Ok(BvpSymbolicAssemblyBackend::ExprLegacy),
        "atom_native" => Ok(BvpSymbolicAssemblyBackend::AtomView),
        other => Err(UnsupportedOptionError::new_err(format!(
            "unsupported BVP_Damp frontend '{other}'"
        ))),
    }
}

fn execution_policy(value: &str) -> PyResult<BvpLambdifyExecutionPolicy> {
    match value {
        "sequential" => Ok(BvpLambdifyExecutionPolicy::Sequential),
        "parallel" => Ok(BvpLambdifyExecutionPolicy::Parallel { min_work: 0 }),
        "auto" => Ok(BvpLambdifyExecutionPolicy::Auto { min_work: 0 }),
        other => Err(InvalidConfigurationError::new_err(format!(
            "unknown BVP_Damp execution policy '{other}'"
        ))),
    }
}

fn build_policy(options: &SolverOptions) -> AotBuildPolicy {
    let profile = if options.aot_profile() == "debug" {
        AotBuildProfile::Debug
    } else {
        AotBuildProfile::Release
    };
    match options.aot_policy() {
        "require_prebuilt" => AotBuildPolicy::RequirePrebuilt,
        "rebuild_always" => AotBuildPolicy::RebuildAlways { profile },
        _ => AotBuildPolicy::BuildIfMissing { profile },
    }
}

fn grid_method(value: &str, parameters: &[f64]) -> PyResult<GridRefinementMethod> {
    finite_values(parameters, "grid_parameters")?;
    let normalized = value.trim().to_ascii_lowercase().replace(['-', ' '], "_");
    let wrong_count = |expected: usize| {
        InvalidConfigurationError::new_err(format!(
            "grid_method '{normalized}' expects {expected} grid_parameters, got {}",
            parameters.len()
        ))
    };
    match normalized.as_str() {
        "double_points" | "double" => {
            if !parameters.is_empty() {
                return Err(wrong_count(0));
            }
            Ok(GridRefinementMethod::DoublePoints)
        }
        "easy" => {
            if parameters.len() != 1 {
                return Err(wrong_count(1));
            }
            Ok(GridRefinementMethod::Easy(parameters[0]))
        }
        "pearson" => {
            if parameters.len() != 2 {
                return Err(wrong_count(2));
            }
            Ok(GridRefinementMethod::Pearson(parameters[0], parameters[1]))
        }
        "grcar_smooke" | "grcar" => {
            if parameters.len() != 3 {
                return Err(wrong_count(3));
            }
            Ok(GridRefinementMethod::GrcarSmooke(
                parameters[0],
                parameters[1],
                parameters[2],
            ))
        }
        "sci" | "scipy" => {
            if !parameters.is_empty() {
                return Err(wrong_count(0));
            }
            Ok(GridRefinementMethod::Sci())
        }
        "two_point" | "twopoint" => {
            if parameters.len() != 3 {
                return Err(wrong_count(3));
            }
            Ok(GridRefinementMethod::TwoPoint(
                parameters[0],
                parameters[1],
                parameters[2],
            ))
        }
        _ => Err(UnsupportedOptionError::new_err(format!(
            "unsupported BVP_Damp grid_method '{normalized}'; expected double_points, easy, pearson, grcar_smooke, sci or two_point"
        ))),
    }
}

fn strategy_params(
    max_jacobian_refreshes: Option<usize>,
    max_damping_iterations: Option<usize>,
    damping_factor: Option<f64>,
    max_refinements: Option<usize>,
    grid_method_name: Option<&str>,
    grid_parameters: Option<Vec<f64>>,
) -> PyResult<SolverParams> {
    let mut params = SolverParams::default();
    if let Some(value) = max_jacobian_refreshes {
        if value == 0 {
            return Err(InvalidConfigurationError::new_err(
                "max_jacobian_refreshes must be non-zero",
            ));
        }
        params.max_jac = Some(value);
    }
    if let Some(value) = max_damping_iterations {
        if value == 0 {
            return Err(InvalidConfigurationError::new_err(
                "max_damping_iterations must be non-zero",
            ));
        }
        params.max_damp_iter = Some(value);
    }
    if let Some(value) = damping_factor {
        if !value.is_finite() || !(0.0 < value && value < 1.0) {
            return Err(InvalidConfigurationError::new_err(
                "damping_factor must be finite and strictly between 0 and 1",
            ));
        }
        params.damp_factor = Some(value);
    }
    let grid_parameters = grid_parameters.unwrap_or_default();
    if let Some(name) = grid_method_name {
        let max_refinements = max_refinements.ok_or_else(|| {
            InvalidConfigurationError::new_err(
                "max_refinements is required when grid_method is selected",
            )
        })?;
        let method = grid_method(name, &grid_parameters)?;
        params.adaptive = Some(AdaptiveGridConfig {
            version: 1,
            max_refinements,
            grid_method: method,
        });
    } else if !grid_parameters.is_empty() || max_refinements.is_some() {
        return Err(InvalidConfigurationError::new_err(
            "grid_parameters and max_refinements require grid_method",
        ));
    }
    Ok(params)
}

fn aot_execution_policy(options: &SolverOptions) -> AotExecutionPolicy {
    match options.execution_policy() {
        "sequential" => AotExecutionPolicy::SequentialOnly,
        "parallel" => AotExecutionPolicy::Parallel(ParallelExecutorConfig::default()),
        _ => AotExecutionPolicy::Auto,
    }
}

fn linear_solver_config(
    options: &SolverOptions,
    policy: Option<&str>,
    fallback: Option<&str>,
    iterative_refinement_steps: usize,
) -> PyResult<Option<LinearSolverConfig>> {
    if options.matrix_layout() != "banded" {
        if policy.is_some() || fallback.is_some() || iterative_refinement_steps != 0 {
            return Err(UnsupportedOptionError::new_err(
                "banded linear solver settings require matrix_layout='banded'",
            ));
        }
        return Ok(None);
    }
    let policy = match policy.unwrap_or("auto") {
        "auto" => LinearSolverPolicy::Auto,
        "block_tridiagonal" => LinearSolverPolicy::ForceBlockTridiagonal,
        "block_tridiagonal_consistent" => LinearSolverPolicy::ForceBlockTridiagonalConsistent,
        "banded" => LinearSolverPolicy::ForceBanded,
        "general_banded_partial_pivot" => LinearSolverPolicy::ForceGeneralBandedPartialPivot,
        "faer_sparse" => LinearSolverPolicy::ForceFaerSparse,
        other => {
            return Err(UnsupportedOptionError::new_err(format!(
                "unsupported banded_linear_solver '{other}'"
            )));
        }
    };
    let fallback = match fallback.unwrap_or("to_faer_sparse") {
        "never" => FallbackPolicy::Never,
        "to_faer_sparse" => FallbackPolicy::ToFaerSparse,
        other => {
            return Err(UnsupportedOptionError::new_err(format!(
                "unsupported banded_fallback '{other}'"
            )));
        }
    };
    Ok(Some(LinearSolverConfig {
        policy,
        fallback,
        iterative_refinement_steps,
    }))
}

fn backend_config(
    options: &SolverOptions,
    logging_mode: BvpLoggingMode,
    logging_max_events: usize,
    banded_linear_solver: Option<&str>,
    banded_fallback: Option<&str>,
    iterative_refinement_steps: usize,
) -> PyResult<GeneratedBackendConfig> {
    let assembly = frontend(options.frontend())?;
    let (_, matrix) = matrix_backend(options.matrix_layout())?;
    let policy = if options.execution() == "aot" {
        BackendSelectionPolicy::AotOnly
    } else {
        BackendSelectionPolicy::LambdifyOnly
    };
    let (bvp_telemetry, lambdify_telemetry) = map_telemetry(options.telemetry())?;
    if options.aot_output_dir().is_some() {
        return Err(UnsupportedOptionError::new_err(
            "BVP_Damp does not currently expose a native AOT output-directory override; use the native registry/resolver configuration",
        ));
    }
    let mut config = GeneratedBackendConfig::new()
        .with_symbolic_assembly_backend(assembly)
        .with_matrix_backend_override(matrix)
        .with_backend_policy_override(Some(policy))
        .with_aot_build_policy(build_policy(options))
        .with_aot_execution_policy(aot_execution_policy(options))
        .with_lambdify_execution_policy(execution_policy(options.execution_policy())?)
        .with_lambdify_telemetry_mode(lambdify_telemetry)
        .with_bvp_telemetry_mode(bvp_telemetry)
        .with_bvp_logging_config(
            BvpLoggingConfig::new(logging_mode).with_max_events(logging_max_events),
        );
    config = match options.aot_toolchain() {
        "gcc" | "tcc" => config
            .with_aot_codegen_backend(AotCodegenBackend::C)
            .with_aot_c_compiler(options.aot_toolchain()),
        "zig" => config.with_aot_codegen_backend(AotCodegenBackend::Zig),
        "rust" => config.with_aot_codegen_backend(AotCodegenBackend::Rust),
        _ => unreachable!("SolverOptions validates the AOT toolchain"),
    };
    if let Some(path) = options.aot_handoff_path() {
        let resolver = AotResolver::read_handoff(path).map_err(|error| {
            PreparationError::new_err(format!("could not read AOT handoff '{path}': {error}"))
        })?;
        config = config.with_resolver(Some(resolver));
    }
    if options.sparse_factorization() != "standard_faer" {
        return Err(UnsupportedOptionError::new_err(
            "BVP_Damp direct API supports only its native standard linear policy; sparse_factorization is a BVP_sci option",
        ));
    }
    if let Some(linear_config) = linear_solver_config(
        options,
        banded_linear_solver,
        banded_fallback,
        iterative_refinement_steps,
    )? {
        config = config.with_banded_linear_solver_config(linear_config);
    }
    Ok(config)
}

fn options_for_solver(
    options: &SolverOptions,
    scheme: &str,
    tolerance: f64,
    max_iterations: usize,
    bounds: HashMap<String, (f64, f64)>,
    state_variables: &[String],
    strategy_params: SolverParams,
    linear_sys_method: Option<&str>,
    loglevel: Option<&str>,
    logging_mode: BvpLoggingMode,
    logging_max_events: usize,
    banded_linear_solver: Option<&str>,
    banded_fallback: Option<&str>,
    iterative_refinement_steps: usize,
) -> PyResult<DampedSolverOptions> {
    let (method, _) = matrix_backend(options.matrix_layout())?;
    let mut solver_options = match options.matrix_layout() {
        "dense" => DampedSolverOptions::dense_damped(),
        "sparse" => DampedSolverOptions::sparse_damped(),
        "banded" => DampedSolverOptions::banded_damped(),
        _ => unreachable!("matrix_backend validates layout"),
    };
    solver_options.method = method;
    solver_options = solver_options
        .with_scheme_name(scheme)
        .with_abs_tolerance(tolerance)
        // BVP_Damp requires a per-state relative tolerance in addition to the
        // scalar absolute tolerance.  The Python surface deliberately keeps a
        // compact scalar tolerance, so apply it consistently to every state.
        .with_rel_tolerance(
            state_variables
                .iter()
                .cloned()
                .map(|name| (name, tolerance))
                .collect(),
        )
        .with_max_iterations(max_iterations)
        .with_bounds(bounds)
        .with_strategy_params(Some(strategy_params))
        .with_loglevel(loglevel.map(str::to_owned))
        .with_generated_backend_config(backend_config(
            options,
            logging_mode,
            logging_max_events,
            banded_linear_solver,
            banded_fallback,
            iterative_refinement_steps,
        )?);
    solver_options.linear_sys_method = linear_sys_method.map(str::to_owned);
    Ok(solver_options)
}

fn flatten_node_major(
    initial_state: Vec<f64>,
    nodes: usize,
    state_dimension: usize,
) -> PyResult<DMatrix<f64>> {
    let expected = nodes.checked_mul(state_dimension).ok_or_else(|| {
        InvalidConfigurationError::new_err("mesh/state dimensions overflow usize")
    })?;
    if initial_state.len() != expected {
        return Err(InvalidConfigurationError::new_err(format!(
            "initial_state must contain {expected} node-major values, got {}",
            initial_state.len()
        )));
    }
    finite_values(&initial_state, "initial_state")?;
    let mut matrix = DMatrix::zeros(state_dimension, nodes);
    for node in 0..nodes {
        for state in 0..state_dimension {
            matrix[(state, node)] = initial_state[node * state_dimension + state];
        }
    }
    Ok(matrix)
}

fn detached_values(matrix: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..matrix.ncols())
        .map(|node| {
            (0..matrix.nrows())
                .map(|state| matrix[(state, node)])
                .collect()
        })
        .collect()
}

fn snapshot_maps(stats: &DampedBvpStatistics) -> (HashMap<String, u64>, HashMap<String, f64>) {
    let counters = [
        ("iterations", stats.telemetry.counters.iterations),
        ("residual_calls", stats.telemetry.counters.residual_calls),
        (
            "jacobian_requests",
            stats.telemetry.counters.jacobian_requests,
        ),
        (
            "jacobian_recalculations",
            stats.telemetry.counters.jacobian_recalculations,
        ),
        ("damping_trials", stats.telemetry.counters.damping_trials),
        (
            "damping_rejections",
            stats.telemetry.counters.damping_rejections,
        ),
        ("linear_solves", stats.telemetry.counters.linear_solves),
        ("factorizations", stats.telemetry.counters.factorizations),
        (
            "factorization_cache_hits",
            stats.telemetry.counters.factorization_cache_hits,
        ),
        ("rhs_solves", stats.telemetry.counters.rhs_solves),
        (
            "grid_refinements",
            stats.telemetry.counters.grid_refinements,
        ),
    ]
    .into_iter()
    .map(|(name, value)| (name.to_owned(), value))
    .collect();
    let timings = [
        ("total_ms", stats.telemetry.timings.total),
        ("residual_ms", stats.telemetry.timings.residual),
        ("jacobian_ms", stats.telemetry.timings.jacobian),
        ("linear_system_ms", stats.telemetry.timings.linear_system),
        ("factorization_ms", stats.telemetry.timings.factorization),
        ("rhs_solve_ms", stats.telemetry.timings.rhs_solve),
        (
            "symbolic_operations_ms",
            stats.telemetry.timings.symbolic_operations,
        ),
        (
            "grid_refinement_ms",
            stats.telemetry.timings.grid_refinement,
        ),
    ]
    .into_iter()
    .map(|(name, duration)| (name.to_owned(), duration.as_secs_f64() * 1000.0))
    .collect();
    (counters, timings)
}

/// Detached result for one BVP_Damp solve.
#[pyclass(name = "BvpDampSolution", module = "rustedscithe")]
#[derive(Clone)]
pub struct BvpDampSolution {
    mesh: Vec<f64>,
    values: Vec<Vec<f64>>,
    parameters: Vec<f64>,
    status: String,
    counters: HashMap<String, u64>,
    timings_ms: HashMap<String, f64>,
}

#[pymethods]
impl BvpDampSolution {
    pub fn mesh(&self) -> Vec<f64> {
        self.mesh.clone()
    }

    pub fn values(&self) -> Vec<Vec<f64>> {
        self.values.clone()
    }

    /// Owned NumPy array with shape `(state_dimension, mesh_nodes)`.
    pub fn values_array<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        crate::numpy_support::matrix(py, &self.values)
    }

    /// Owned NumPy array containing the accepted mesh.
    pub fn mesh_array<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f64>> {
        crate::numpy_support::vector(py, &self.mesh)
    }

    #[getter]
    pub fn parameters(&self) -> Vec<f64> {
        self.parameters.clone()
    }

    #[getter]
    pub fn status(&self) -> &str {
        &self.status
    }

    pub fn telemetry_counters(&self) -> HashMap<String, u64> {
        self.counters.clone()
    }

    pub fn telemetry_timings_ms(&self) -> HashMap<String, f64> {
        self.timings_ms.clone()
    }

    /// Return a small stable result descriptor for notebooks and batch tools.
    ///
    /// Mesh and values remain available through their typed accessors; this
    /// method is intentionally metadata-only and never serializes the state
    /// matrix into a second representation.
    pub fn diagnostics(&self) -> HashMap<String, String> {
        HashMap::from([
            ("solver".to_owned(), "bvp_damp".to_owned()),
            ("status".to_owned(), self.status.clone()),
            ("nodes".to_owned(), self.mesh.len().to_string()),
            ("state_dimension".to_owned(), self.values.len().to_string()),
            (
                "parameter_count".to_owned(),
                self.parameters.len().to_string(),
            ),
        ])
    }

    pub fn telemetry_report(&self) -> String {
        let mut counters = self.counters.iter().collect::<Vec<_>>();
        counters.sort_by_key(|(name, _)| *name);
        let mut timings = self.timings_ms.iter().collect::<Vec<_>>();
        timings.sort_by_key(|(name, _)| *name);
        format!(
            "BVP_Damp status={}; counters={}; timings_ms={}",
            self.status,
            counters
                .iter()
                .map(|(name, value)| format!("{name}={value}"))
                .collect::<Vec<_>>()
                .join(","),
            timings
                .iter()
                .map(|(name, value)| format!("{name}={value:.6}"))
                .collect::<Vec<_>>()
                .join(","),
        )
    }

    pub fn write_telemetry_report(&self, path: &str) -> PyResult<()> {
        write_report(path, &self.telemetry_report())
    }

    /// Write one compact CSV row per mesh node without capturing solver logs.
    pub fn write_solution_csv(&self, path: &str) -> PyResult<()> {
        let state_dimension = self.values.len();
        let aligned = self.values.iter().all(|row| row.len() == self.mesh.len());
        let mut report = if aligned {
            String::from("mesh")
        } else {
            String::from("node_index")
        };
        for state in 0..state_dimension {
            report.push_str(&format!(",y{state}"));
        }
        report.push('\n');
        let row_count = if aligned {
            self.mesh.len()
        } else {
            self.values.iter().map(Vec::len).max().unwrap_or(0)
        };
        for node in 0..row_count {
            if aligned {
                report.push_str(&format!("{:.17e}", self.mesh[node]));
            } else {
                report.push_str(&node.to_string());
            }
            for state in 0..state_dimension {
                if let Some(value) = self.values[state].get(node) {
                    report.push_str(&format!(",{value:.17e}"));
                } else {
                    report.push(',');
                }
            }
            report.push('\n');
        }
        write_report(path, &report)
    }

    /// Write a human-readable tabular state snapshot for simple postprocessing.
    pub fn write_solution_txt(&self, path: &str) -> PyResult<()> {
        let state_dimension = self.values.len();
        let aligned = self.values.iter().all(|row| row.len() == self.mesh.len());
        let mut report = if aligned {
            String::from("BVP_Damp solution\nmesh")
        } else {
            String::from("BVP_Damp solution\nnode_index")
        };
        for state in 0..state_dimension {
            report.push_str(&format!("\ty{state}"));
        }
        report.push('\n');
        let row_count = if aligned {
            self.mesh.len()
        } else {
            self.values.iter().map(Vec::len).max().unwrap_or(0)
        };
        for node in 0..row_count {
            if aligned {
                report.push_str(&format!("{:.17e}", self.mesh[node]));
            } else {
                report.push_str(&node.to_string());
            }
            for state in 0..state_dimension {
                if let Some(value) = self.values[state].get(node) {
                    report.push_str(&format!("\t{value:.17e}"));
                } else {
                    report.push('\t');
                }
            }
            report.push('\n');
        }
        write_report(path, &report)
    }
}

/// Direct Python adapter for the native damped Newton BVP solver.
#[pyclass(name = "BvpDampSolver", module = "rustedscithe", unsendable)]
pub struct BvpDampSolver {
    solver: NRBVP,
    frontend: String,
    execution: String,
    matrix_layout: String,
    bandwidth: Option<(usize, usize)>,
    prepared: bool,
}

#[pymethods]
impl BvpDampSolver {
    #[new]
    #[pyo3(signature=(expressions, state_variables, boundary_conditions, mesh, initial_state, *, parameters=None, parameter_values=None, options=None, independent_variable="x", scheme="forward", strategy="damped", tolerance=1e-6, max_iterations=25, max_jacobian_refreshes=None, max_damping_iterations=None, damping_factor=None, max_refinements=None, grid_method=None, grid_parameters=None, linear_sys_method=None, loglevel=None, logging_mode="off", logging_max_events=256, banded_linear_solver=None, banded_fallback=None, iterative_refinement_steps=0, bounds=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        expressions: Vec<PyExpr>,
        state_variables: Vec<String>,
        boundary_conditions: HashMap<String, Vec<(usize, f64)>>,
        mesh: Vec<f64>,
        initial_state: Vec<f64>,
        parameters: Option<Vec<String>>,
        parameter_values: Option<Vec<f64>>,
        options: Option<PyRef<'_, SolverOptions>>,
        independent_variable: &str,
        scheme: &str,
        strategy: &str,
        tolerance: f64,
        max_iterations: usize,
        max_jacobian_refreshes: Option<usize>,
        max_damping_iterations: Option<usize>,
        damping_factor: Option<f64>,
        max_refinements: Option<usize>,
        grid_method: Option<&str>,
        grid_parameters: Option<Vec<f64>>,
        linear_sys_method: Option<&str>,
        loglevel: Option<&str>,
        logging_mode: &str,
        logging_max_events: usize,
        banded_linear_solver: Option<&str>,
        banded_fallback: Option<&str>,
        iterative_refinement_steps: usize,
        bounds: Option<HashMap<String, (f64, f64)>>,
    ) -> PyResult<Self> {
        validate_names(&state_variables, "state_variables", false)?;
        let parameters = parameters.unwrap_or_default();
        validate_names(&parameters, "parameters", true)?;
        if independent_variable.trim().is_empty() {
            return Err(InvalidConfigurationError::new_err(
                "independent_variable must not be empty",
            ));
        }
        if expressions.len() != state_variables.len() {
            return Err(InvalidConfigurationError::new_err(format!(
                "expected one RHS expression per state variable, got {} expressions for {} variables",
                expressions.len(),
                state_variables.len()
            )));
        }
        if state_variables
            .iter()
            .chain(parameters.iter())
            .any(|name| name == independent_variable)
        {
            return Err(InvalidConfigurationError::new_err(
                "independent_variable must not collide with state_variables or parameters",
            ));
        }
        if !matches!(scheme, "forward" | "trapezoid") {
            return Err(UnsupportedOptionError::new_err(
                "BVP_Damp scheme must be 'forward' or 'trapezoid'",
            ));
        }
        if strategy != "damped" {
            return Err(UnsupportedOptionError::new_err(
                "BVP_Damp direct Python API currently supports strategy='damped' only; strategy='frozen' remains an explicit stub",
            ));
        }
        if let Some(method) = linear_sys_method {
            if method.trim().is_empty() {
                return Err(InvalidConfigurationError::new_err(
                    "linear_sys_method must not be empty",
                ));
            }
        }
        let normalized_loglevel = match loglevel {
            None | Some("off") | Some("none") => None,
            Some("debug") => Some("debug"),
            Some("info") => Some("info"),
            Some("warn") => Some("warn"),
            Some("error") => Some("error"),
            Some(value) => {
                return Err(UnsupportedOptionError::new_err(format!(
                    "unsupported BVP_Damp loglevel '{value}'; expected off, debug, info, warn or error"
                )));
            }
        };
        let logging_mode = match logging_mode.trim().to_ascii_lowercase().as_str() {
            "off" => BvpLoggingMode::Off,
            "warnings" | "warn" => BvpLoggingMode::Warnings,
            "detailed" | "debug" => BvpLoggingMode::Detailed,
            other => {
                return Err(UnsupportedOptionError::new_err(format!(
                    "unsupported BVP_Damp logging_mode '{other}'; expected off, warnings or detailed"
                )));
            }
        };
        if !tolerance.is_finite() || tolerance <= 0.0 || max_iterations == 0 {
            return Err(InvalidConfigurationError::new_err(
                "tolerance must be finite and positive and max_iterations must be non-zero",
            ));
        }
        if mesh.len() < 2 || !mesh.windows(2).all(|pair| pair[1] > pair[0]) {
            return Err(InvalidConfigurationError::new_err(
                "mesh must contain at least two strictly increasing nodes",
            ));
        }
        finite_values(&mesh, "mesh")?;
        let parameter_values = parameter_values.unwrap_or_else(|| vec![0.0; parameters.len()]);
        if parameter_values.len() != parameters.len() {
            return Err(InvalidConfigurationError::new_err(format!(
                "parameter_values must contain {} values, got {}",
                parameters.len(),
                parameter_values.len()
            )));
        }
        finite_values(&parameter_values, "parameter_values")?;
        let expected_symbols = std::iter::once(independent_variable.to_owned())
            .chain(state_variables.iter().cloned())
            .chain(parameters.iter().cloned())
            .collect::<std::collections::HashSet<_>>();
        for (index, expression) in expressions.iter().enumerate() {
            if let Some(unknown) = expression
                .inner
                .extract_variables()
                .1
                .iter()
                .find(|name| !expected_symbols.contains(*name))
            {
                return Err(InvalidConfigurationError::new_err(format!(
                    "RHS expression {index} references undeclared symbol '{unknown}'"
                )));
            }
        }
        finite_boundary_conditions(&boundary_conditions)?;
        let bounds = bounds.ok_or_else(|| {
            InvalidConfigurationError::new_err(
                "BVP_Damp requires explicit bounds for every state variable",
            )
        })?;
        validate_bounds(&bounds, &state_variables)?;
        let options = match options {
            Some(value) => value.clone(),
            None => SolverOptions::new(None, None, None, None, None, None, None, None, None)?,
        };
        options.validate_for("bvp_damp")?;
        let _ = matrix_backend(options.matrix_layout())?;
        let bandwidth = match (options.band_lower(), options.band_upper()) {
            (Some(lower), Some(upper)) if options.matrix_layout() == "banded" => {
                Some((lower, upper))
            }
            (None, None) if options.matrix_layout() != "banded" => None,
            (None, None) => {
                return Err(InvalidConfigurationError::new_err(
                    "Banded BVP_Damp requires options.set_banded_width(lower, upper)",
                ));
            }
            _ if options.matrix_layout() != "banded" => {
                return Err(UnsupportedOptionError::new_err(
                    "banded widths are only valid with matrix_layout='banded'",
                ));
            }
            _ => {
                return Err(InvalidConfigurationError::new_err(
                    "band_lower and band_upper must be set together",
                ));
            }
        };
        let strategy_params = strategy_params(
            max_jacobian_refreshes,
            max_damping_iterations,
            damping_factor,
            max_refinements,
            grid_method,
            grid_parameters,
        )?;
        let damp_options = options_for_solver(
            &options,
            scheme,
            tolerance,
            max_iterations,
            bounds,
            &state_variables,
            strategy_params,
            linear_sys_method,
            normalized_loglevel,
            logging_mode,
            logging_max_events,
            banded_linear_solver,
            banded_fallback,
            iterative_refinement_steps,
        )?;
        let initial_guess = flatten_node_major(initial_state, mesh.len(), state_variables.len())?;
        let mut solver = NRBVP::new_with_options(
            expressions
                .into_iter()
                .map(|expression| expression.inner)
                .collect(),
            initial_guess,
            state_variables,
            independent_variable.to_owned(),
            boundary_conditions,
            mesh[0],
            mesh.last().copied().ok_or_else(|| {
                InvalidConfigurationError::new_err("mesh must contain at least two nodes")
            })?,
            mesh.len(),
            damp_options,
        );
        if !parameters.is_empty() {
            let parameter_refs = parameters.iter().map(String::as_str).collect::<Vec<_>>();
            solver
                .try_set_params(Some(parameter_refs.as_slice()))
                .map_err(map_native_error)?;
            solver
                .try_set_param_values(Some(parameter_values))
                .map_err(map_native_error)?;
        }
        Ok(Self {
            solver,
            frontend: options.frontend().to_owned(),
            execution: options.execution().to_owned(),
            matrix_layout: options.matrix_layout().to_owned(),
            bandwidth,
            prepared: false,
        })
    }

    pub fn solve(&mut self) -> PyResult<BvpDampSolution> {
        catch_panic("solve", "bvp_damp", "lambdify", || self.solve_inner())
    }

    fn solve_inner(&mut self) -> PyResult<BvpDampSolution> {
        if self.bandwidth.is_some() {
            self.solver
                .try_eq_generate(None, self.bandwidth)
                .map_err(map_native_error)?;
            self.solver
                .try_solver_prepared()
                .map_err(map_native_error)?;
        } else {
            self.solver.try_solve().map_err(map_native_error)?;
        }
        self.prepared = true;
        self.detached_solution()
    }

    /// Prepare symbolic callbacks and the selected runtime without solving.
    ///
    /// This is the cold-to-prepared lifecycle boundary used by continuation:
    /// subsequent `set_parameters` calls can update numeric bindings without
    /// rebuilding the symbolic graph.
    pub fn prepare(&mut self) -> PyResult<()> {
        self.solver
            .try_eq_generate(None, self.bandwidth)
            .map_err(map_native_error)?;
        self.prepared = true;
        Ok(())
    }

    /// Solve using callbacks prepared by [`Self::prepare`].
    pub fn solve_prepared(&mut self) -> PyResult<BvpDampSolution> {
        self.solver
            .try_solver_prepared()
            .map_err(map_native_error)?;
        self.prepared = true;
        self.detached_solution()
    }

    /// Replace numeric parameter values while retaining the prepared graph.
    pub fn set_parameters(&mut self, values: Vec<f64>) -> PyResult<()> {
        finite_values(&values, "parameter_values")?;
        if values.len() != self.solver.param_names.len() {
            return Err(InvalidConfigurationError::new_err(format!(
                "parameter_values must contain {} values, got {}",
                self.solver.param_names.len(),
                values.len()
            )));
        }
        self.solver
            .try_set_param_values(Some(values))
            .map_err(map_native_error)
    }

    /// Reset the Newton initial guess while retaining prepared callbacks.
    /// Numeric Jacobian and factor state are invalidated by the native solver.
    pub fn restart(&mut self, initial_state: Vec<f64>) -> PyResult<()> {
        let initial_guess =
            flatten_node_major(initial_state, self.solver.n_steps, self.solver.values.len())?;
        self.solver
            .try_set_initial_guess(initial_guess)
            .map_err(map_native_error)
    }

    /// Install a warm-start iterate without invalidating prepared callbacks.
    pub fn set_prepared_iterate(&mut self, initial_state: Vec<f64>) -> PyResult<()> {
        let iterate =
            flatten_node_major(initial_state, self.solver.n_steps, self.solver.values.len())?;
        self.solver
            .try_set_prepared_iterate(iterate)
            .map_err(map_native_error)
    }

    #[getter]
    pub fn parameter_names(&self) -> Vec<String> {
        self.solver.param_names.clone()
    }

    #[getter]
    pub fn parameter_values(&self) -> Vec<f64> {
        self.solver.param_values.clone().unwrap_or_default()
    }

    #[getter]
    pub fn frontend(&self) -> &str {
        &self.frontend
    }

    #[getter]
    pub fn execution(&self) -> &str {
        &self.execution
    }

    #[getter]
    pub fn matrix_layout(&self) -> &str {
        &self.matrix_layout
    }

    /// Return the effective route selected at construction time.
    pub fn provenance(&self) -> HashMap<String, String> {
        let mut values = HashMap::from([
            ("solver".to_owned(), "bvp_damp".to_owned()),
            ("strategy".to_owned(), "damped".to_owned()),
            ("frontend".to_owned(), self.frontend.clone()),
            ("execution".to_owned(), self.execution.clone()),
            ("matrix_layout".to_owned(), self.matrix_layout.clone()),
            (
                "prepared_bandwidth".to_owned(),
                self.bandwidth.map_or_else(
                    || "none".to_owned(),
                    |(lower, upper)| format!("{lower},{upper}"),
                ),
            ),
        ]);
        values.insert("prepared".to_owned(), self.prepared.to_string());
        values
    }

    fn detached_solution(&self) -> PyResult<BvpDampSolution> {
        let result = self.solver.get_result().ok_or_else(|| {
            SolverError::new_err("BVP_Damp completed without a detached solution")
        })?;
        let stats = self.solver.get_statistics();
        let (counters, timings_ms) = snapshot_maps(&stats);
        Ok(BvpDampSolution {
            mesh: self.solver.x_mesh.iter().copied().collect(),
            values: detached_values(&result),
            parameters: self.solver.param_values.clone().unwrap_or_default(),
            status: "finished".to_owned(),
            counters,
            timings_ms,
        })
    }
}

fn finite_boundary_conditions(conditions: &HashMap<String, Vec<(usize, f64)>>) -> PyResult<()> {
    for (name, entries) in conditions {
        if name.trim().is_empty() {
            return Err(InvalidConfigurationError::new_err(
                "boundary condition variable names must not be empty",
            ));
        }
        for (position, value) in entries {
            if *position > 1 || !value.is_finite() {
                return Err(InvalidConfigurationError::new_err(
                    "boundary conditions require position 0 or 1 and finite values",
                ));
            }
        }
    }
    Ok(())
}

fn validate_bounds(
    bounds: &HashMap<String, (f64, f64)>,
    state_variables: &[String],
) -> PyResult<()> {
    for state in state_variables {
        let Some((lower, upper)) = bounds.get(state) else {
            return Err(InvalidConfigurationError::new_err(format!(
                "bounds must contain state variable '{state}'"
            )));
        };
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(InvalidConfigurationError::new_err(format!(
                "bounds for '{state}' must be finite and ordered as lower <= upper"
            )));
        }
    }
    Ok(())
}
