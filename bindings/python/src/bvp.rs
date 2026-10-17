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
use pyo3::types::{PyAny, PyDict};

use crate::aot::generated_config;
use rustedscithe_core::numerical::BVP_sci::new::{
    BvpSciAssembly, BvpSciExecutionPolicy, BvpSciMatrixLayout, BvpSciNewError, BvpSciNumericalPlan,
    BvpSciOutputPolicy, BvpSciSingularTerm, BvpSciSolution as CoreSolution,
    BvpSciSolver as CoreSolver, BvpSciSparseFactorization, BvpSciTelemetry, BvpSciTelemetryMode,
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
        "detailed" => Ok(BvpSciTelemetryMode::Detailed),
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
    let mut counters: HashMap<String, u64> = [
        ("rhs_calls", snapshot.rhs_calls),
        ("boundary_calls", snapshot.boundary_calls),
        ("jacobian_calls", snapshot.jacobian_calls),
        (
            "parameter_jacobian_calls",
            snapshot.parameter_jacobian_calls,
        ),
        ("boundary_jacobian_calls", snapshot.boundary_jacobian_calls),
        (
            "finite_difference_probes",
            snapshot.finite_difference_probes,
        ),
        ("newton_iterations", snapshot.newton_iterations),
        (
            "newton_jacobian_refreshes",
            snapshot.newton_jacobian_refreshes,
        ),
        (
            "newton_backtracking_trials",
            snapshot.newton_backtracking_trials,
        ),
        ("newton_accepted_steps", snapshot.newton_accepted_steps),
        ("newton_rejected_steps", snapshot.newton_rejected_steps),
        ("collocation_evaluations", snapshot.collocation_evaluations),
        ("mesh_defect_probes", snapshot.mesh_defect_probes),
        ("mesh_refinements", snapshot.mesh_refinements),
        ("mesh_points_added", snapshot.mesh_points_added),
        ("linear_assemblies", snapshot.linear_assemblies),
        ("factorizations", snapshot.factorizations),
        ("superlu_fallbacks", snapshot.superlu_fallbacks),
        (
            "sparse_symbolic_analyses",
            snapshot.sparse_symbolic_analyses,
        ),
        ("sparse_symbolic_reuses", snapshot.sparse_symbolic_reuses),
        (
            "sparse_numeric_factorizations",
            snapshot.sparse_numeric_factorizations,
        ),
        (
            "sparse_triplet_materializations",
            snapshot.sparse_triplet_materializations,
        ),
        ("sparse_csc_assemblies", snapshot.sparse_csc_assemblies),
        (
            "banded_core_factorizations",
            snapshot.banded_core_factorizations,
        ),
        (
            "banded_core_to_border_solves",
            snapshot.banded_core_to_border_solves,
        ),
        ("banded_schur_assemblies", snapshot.banded_schur_assemblies),
        (
            "banded_border_factorizations",
            snapshot.banded_border_factorizations,
        ),
        ("linear_solves", snapshot.linear_solves),
        (
            "banded_structured_factorizations",
            snapshot.banded_structured_factorizations,
        ),
        (
            "banded_scalar_fallback_factorizations",
            snapshot.banded_scalar_fallback_factorizations,
        ),
        (
            "banded_scalar_fallback_assemblies",
            snapshot.banded_scalar_fallback_assemblies,
        ),
        (
            "banded_structured_solves",
            snapshot.banded_structured_solves,
        ),
        (
            "banded_scalar_fallback_solves",
            snapshot.banded_scalar_fallback_solves,
        ),
        (
            "banded_sparse_fallback_factorizations",
            snapshot.banded_sparse_fallback_factorizations,
        ),
        (
            "banded_sparse_fallback_solves",
            snapshot.banded_sparse_fallback_solves,
        ),
        (
            "banded_sparse_fallback_assemblies",
            snapshot.banded_sparse_fallback_assemblies,
        ),
        ("banded_residual_checks", snapshot.banded_residual_checks),
        (
            "banded_fallback_switches",
            snapshot.banded_fallback_switches,
        ),
        ("banded_rhs_permutations", snapshot.banded_rhs_permutations),
        (
            "banded_rhs_permutation_entries",
            snapshot.banded_rhs_permutation_entries,
        ),
        ("parameter_rebinds", snapshot.parameter_rebinds),
        ("continuation_solves", snapshot.continuation_solves),
        ("restarts", snapshot.restarts),
        ("copies", snapshot.copies),
        ("workspace_resizes", snapshot.workspace_resizes),
        ("output_writes", snapshot.output_writes),
        ("argument_bindings", snapshot.argument_bindings),
        ("residual_evaluations", snapshot.residual_evaluations),
        ("jacobian_evaluations", snapshot.jacobian_evaluations),
        (
            "jacobian_output_assemblies",
            snapshot.jacobian_output_assemblies,
        ),
        ("allocations", snapshot.allocations),
        ("atom_conversions", snapshot.atom_conversions),
        (
            "symbolic_jacobian_derivations",
            snapshot.symbolic_jacobian_derivations,
        ),
        ("pattern_entries", snapshot.pattern_entries),
        ("lowering_calls", snapshot.lowering_calls),
        ("evaluator_compilations", snapshot.evaluator_compilations),
        (
            "residual_evaluator_compilations",
            snapshot.residual_evaluator_compilations,
        ),
        (
            "jacobian_evaluator_compilations",
            snapshot.jacobian_evaluator_compilations,
        ),
        (
            "singular_term_applications",
            snapshot.singular_term_applications,
        ),
        (
            "dense_output_constructions",
            snapshot.dense_output_constructions,
        ),
        ("preparation_calls", snapshot.preparation_calls),
        ("newton_scope_calls", snapshot.newton_scope_calls),
        (
            "mesh_defect_estimation_calls",
            snapshot.mesh_defect_estimation_calls,
        ),
        (
            "output_construction_calls",
            snapshot.output_construction_calls,
        ),
        ("full_solve_calls", snapshot.full_solve_calls),
        ("parallel_dispatches", snapshot.parallel_dispatches),
        ("sequential_dispatches", snapshot.sequential_dispatches),
        ("max_worker_threads", snapshot.max_worker_threads),
    ]
    .into_iter()
    .map(|(name, value)| (name.to_owned(), value))
    .collect();
    let mut timings: HashMap<String, f64> = [
        ("preparation_ms", snapshot.preparation_ms),
        ("expr_to_atom_ms", snapshot.expr_to_atom_ms),
        ("symbolic_jacobian_ms", snapshot.symbolic_jacobian_ms),
        ("pattern_ms", snapshot.pattern_ms),
        ("lowering_ms", snapshot.lowering_ms),
        (
            "evaluator_compilation_ms",
            snapshot.evaluator_compilation_ms,
        ),
        (
            "residual_evaluator_compilation_ms",
            snapshot.residual_evaluator_compilation_ms,
        ),
        (
            "jacobian_evaluator_compilation_ms",
            snapshot.jacobian_evaluator_compilation_ms,
        ),
        ("binding_ms", snapshot.binding_ms),
        ("rhs_callback_ms", snapshot.rhs_callback_ms),
        ("boundary_callback_ms", snapshot.boundary_callback_ms),
        ("jacobian_callback_ms", snapshot.jacobian_callback_ms),
        (
            "parameter_jacobian_callback_ms",
            snapshot.parameter_jacobian_callback_ms,
        ),
        (
            "boundary_jacobian_callback_ms",
            snapshot.boundary_jacobian_callback_ms,
        ),
        ("residual_evaluation_ms", snapshot.residual_evaluation_ms),
        ("jacobian_evaluation_ms", snapshot.jacobian_evaluation_ms),
        (
            "jacobian_output_assembly_ms",
            snapshot.jacobian_output_assembly_ms,
        ),
        ("callback_ms", snapshot.callback_ms),
        ("collocation_ms", snapshot.collocation_ms),
        ("mesh_refinement_ms", snapshot.mesh_refinement_ms),
        ("newton_ms", snapshot.newton_ms),
        (
            "mesh_defect_estimation_ms",
            snapshot.mesh_defect_estimation_ms,
        ),
        ("output_construction_ms", snapshot.output_construction_ms),
        ("linear_assembly_ms", snapshot.linear_assembly_ms),
        ("factorization_ms", snapshot.factorization_ms),
        (
            "sparse_symbolic_analysis_ms",
            snapshot.sparse_symbolic_analysis_ms,
        ),
        (
            "sparse_symbolic_reuse_ms",
            snapshot.sparse_symbolic_reuse_ms,
        ),
        (
            "sparse_numeric_factorization_ms",
            snapshot.sparse_numeric_factorization_ms,
        ),
        (
            "sparse_triplet_materialization_ms",
            snapshot.sparse_triplet_materialization_ms,
        ),
        ("linear_solve_ms", snapshot.linear_solve_ms),
        ("full_solve_ms", snapshot.full_solve_ms),
        ("full_solve_total_ms", snapshot.full_solve_total_ms),
        ("solve_ms", snapshot.solve_ms),
        ("sparse_csc_assembly_ms", snapshot.sparse_csc_assembly_ms),
        (
            "superlu_abi_conversion_ms",
            snapshot.superlu.abi_conversion_ms,
        ),
        ("superlu_factor_ffi_ms", snapshot.superlu.factor_ffi_ms),
        (
            "superlu_native_factorization_ms",
            snapshot.superlu.native_factorization_ms,
        ),
        ("superlu_solve_ffi_ms", snapshot.superlu.solve_ffi_ms),
        ("superlu_native_solve_ms", snapshot.superlu.native_solve_ms),
        (
            "banded_core_factorization_ms",
            snapshot.banded_core_factorization_ms,
        ),
        (
            "banded_core_to_border_solve_ms",
            snapshot.banded_core_to_border_solve_ms,
        ),
        (
            "banded_schur_assembly_ms",
            snapshot.banded_schur_assembly_ms,
        ),
        (
            "banded_border_factorization_ms",
            snapshot.banded_border_factorization_ms,
        ),
        (
            "banded_structured_factorization_ms",
            snapshot.banded_structured_factorization_ms,
        ),
        (
            "banded_scalar_fallback_factorization_ms",
            snapshot.banded_scalar_fallback_factorization_ms,
        ),
        (
            "banded_scalar_fallback_assembly_ms",
            snapshot.banded_scalar_fallback_assembly_ms,
        ),
        (
            "banded_structured_solve_ms",
            snapshot.banded_structured_solve_ms,
        ),
        (
            "banded_scalar_fallback_solve_ms",
            snapshot.banded_scalar_fallback_solve_ms,
        ),
        (
            "banded_sparse_fallback_factorization_ms",
            snapshot.banded_sparse_fallback_factorization_ms,
        ),
        (
            "banded_sparse_fallback_solve_ms",
            snapshot.banded_sparse_fallback_solve_ms,
        ),
        (
            "banded_sparse_fallback_assembly_ms",
            snapshot.banded_sparse_fallback_assembly_ms,
        ),
        (
            "banded_residual_guard_ms",
            snapshot.banded_residual_guard_ms,
        ),
        (
            "banded_rhs_permutation_ms",
            snapshot.banded_rhs_permutation_ms,
        ),
    ]
    .into_iter()
    .filter_map(|(name, value)| value.map(|value| (name.to_owned(), value)))
    .collect();
    if let Some(value) = snapshot.report_preparation_ms() {
        timings.insert("reported_preparation_ms".to_owned(), value);
    }
    match mode.as_str() {
        "off" => {
            counters.clear();
            timings.clear();
        }
        "counters" => timings.clear(),
        "timings" => counters.clear(),
        "detailed" => {}
        _ => unreachable!("native BVP telemetry enum must map to a known mode"),
    }
    (mode, counters, timings)
}

fn snapshot_metrics(
    snapshot: &rustedscithe_core::numerical::BVP_sci::new::BvpSciTelemetrySnapshot,
) -> HashMap<String, f64> {
    [
        (
            "banded_core_factor_residual_relative",
            snapshot.banded_core_factor_residual_relative,
        ),
        ("banded_min_abs_pivot", snapshot.banded_min_abs_pivot),
        (
            "banded_max_multiplier_norm",
            snapshot.banded_max_multiplier_norm,
        ),
        (
            "banded_border_min_abs_pivot",
            snapshot.banded_border_min_abs_pivot,
        ),
        (
            "banded_border_max_multiplier_norm",
            snapshot.banded_border_max_multiplier_norm,
        ),
        (
            "banded_last_residual_inf",
            snapshot.banded_last_residual_inf,
        ),
        (
            "banded_last_solution_inf",
            snapshot.banded_last_solution_inf,
        ),
    ]
    .into_iter()
    .filter_map(|(name, value)| value.map(|value| (name.to_owned(), value)))
    .collect()
}

const AOT_COLD_STAGES: &[&str] = &[
    "validation",
    "parameter_binding",
    "expr_to_atom",
    "symbolic_jacobian",
    "symbolic_differentiation",
    "symbolic_simplification",
    "atom_to_expr",
    "sparse_pattern",
    "layout_planning",
    "residual_compilation",
    "residual_lambdification",
    "jacobian_compilation",
    "jacobian_lambdification",
    "backend_binding",
    "aot_materialization",
    "aot_build",
    "aot_link",
    "atom_preparation",
    "aot_cache_lookup",
    "aot_lowering",
    "aot_source_generation",
    "aot_publication",
    "solver_preparation",
    "bridge_preparation",
    "native_callback_preparation",
    "atom_residual_preparation",
    "atom_jacobian_preparation",
    "parallel_calibration",
    "atom_dependency_analysis",
    "native_jacobian_evaluator_preparation",
    "aot_input_abi_preparation",
    "aot_problem_key_construction",
    "aot_atom_plan_preparation",
    "shared_symbolic_problem_preparation",
];

const AOT_WARM_STAGES: &[&str] = &[
    "argument_binding",
    "residual_evaluation",
    "residual_output_assembly",
    "jacobian_evaluation",
    "jacobian_output_assembly",
    "factorization",
    "rhs_solve",
    "controller",
    "controller_step_setup",
    "controller_predictor",
    "controller_iteration_inclusive",
    "controller_outcome",
    "controller_stop_condition",
    "controller_method_policy",
    "controller_method_switch",
    "residual_callback_inclusive",
    "jacobian_callback_inclusive",
    "aot_chunk_dispatch",
    "aot_worker_execution",
    "aot_argument_copy",
    "aot_output_write",
    "native_engine_setup",
    "native_result_assembly",
    "solve",
    "summary",
];

pub(crate) fn aot_telemetry_maps(
    snapshot: Option<&rustedscithe_core::symbolic::ivp_telemetry::IvpTelemetrySnapshot>,
) -> (
    HashMap<String, String>,
    Vec<String>,
    HashMap<String, u64>,
    HashMap<String, f64>,
) {
    use rustedscithe_core::symbolic::ivp_telemetry::IvpTelemetryMode;

    let Some(snapshot) = snapshot else {
        return (HashMap::new(), Vec::new(), HashMap::new(), HashMap::new());
    };
    if snapshot.mode == IvpTelemetryMode::Off {
        return (HashMap::new(), Vec::new(), HashMap::new(), HashMap::new());
    }
    let metadata = HashMap::from([
        ("mode".to_owned(), snapshot.mode.label().to_owned()),
        ("route".to_owned(), snapshot.route.label().to_owned()),
        (
            "execution".to_owned(),
            snapshot.execution.label().to_owned(),
        ),
        (
            "matrix_backend".to_owned(),
            snapshot.matrix_backend.label().to_owned(),
        ),
        (
            "execution_policy".to_owned(),
            snapshot.lambdify_execution_policy.label().to_owned(),
        ),
        (
            "state_dimension".to_owned(),
            snapshot.state_dimension.to_string(),
        ),
        (
            "residual_dimension".to_owned(),
            snapshot.residual_dimension.to_string(),
        ),
        (
            "parameter_count".to_owned(),
            snapshot.parameter_count.to_string(),
        ),
        (
            "worker_count".to_owned(),
            snapshot.lambdify_worker_count.to_string(),
        ),
        (
            "auto_min_work_per_job".to_owned(),
            snapshot.lambdify_auto_min_work_per_job.to_string(),
        ),
    ]);
    let mut counters = HashMap::new();
    if matches!(
        snapshot.mode,
        IvpTelemetryMode::Counters | IvpTelemetryMode::Detailed
    ) {
        for (name, value) in [
            ("residual_requests", snapshot.residual_requests),
            ("residual_evaluations", snapshot.residual_evaluations),
            (
                "residual_auxiliary_evaluations",
                snapshot.residual_auxiliary_evaluations,
            ),
            (
                "residual_preparation_evaluations",
                snapshot.residual_preparation_evaluations,
            ),
            ("jacobian_requests", snapshot.jacobian_requests),
            ("jacobian_evaluations", snapshot.jacobian_evaluations),
            (
                "jacobian_auxiliary_evaluations",
                snapshot.jacobian_auxiliary_evaluations,
            ),
            (
                "symbolic_jacobian_builds",
                snapshot.symbolic_jacobian_builds,
            ),
            ("jacobian_rebuilds", snapshot.jacobian_rebuilds),
            ("factorization_requests", snapshot.factorization_requests),
            (
                "steps_using_current_jacobian",
                snapshot.steps_using_current_jacobian,
            ),
            ("linear_solve_requests", snapshot.linear_solve_requests),
            ("accepted_steps", snapshot.accepted_steps),
            ("rejected_steps", snapshot.rejected_steps),
            ("parameter_binds", snapshot.parameter_binds),
            ("method_switches", snapshot.method_switches),
            ("scalar_evaluations", snapshot.scalar_evaluations),
            ("conversions", snapshot.conversions),
            ("copies", snapshot.copies),
            ("copied_bytes", snapshot.copied_bytes),
            ("allocated_bytes", snapshot.allocated_bytes),
            ("errors", snapshot.errors),
            ("parallel_dispatches", snapshot.parallel_dispatches),
            ("sequential_dispatches", snapshot.sequential_dispatches),
            ("aot_resolution_hits", snapshot.aot_resolution_hits),
            ("aot_resolution_misses", snapshot.aot_resolution_misses),
            ("aot_reconnects", snapshot.aot_reconnects),
            ("aot_build_attempts", snapshot.aot_build_attempts),
            ("aot_build_retries", snapshot.aot_build_retries),
            ("aot_build_successes", snapshot.aot_build_successes),
            ("aot_build_failures", snapshot.aot_build_failures),
            ("aot_link_attempts", snapshot.aot_link_attempts),
            ("aot_link_successes", snapshot.aot_link_successes),
            ("aot_link_failures", snapshot.aot_link_failures),
            ("aot_runtime_ready", snapshot.aot_runtime_ready),
            ("aot_chunk_dispatches", snapshot.aot_chunk_dispatches),
            ("aot_dispatch_requests", snapshot.aot_dispatch_requests),
            ("aot_eligible_dispatches", snapshot.aot_eligible_dispatches),
            ("aot_parallel_dispatches", snapshot.aot_parallel_dispatches),
            (
                "aot_sequential_fallbacks",
                snapshot.aot_sequential_fallbacks,
            ),
            (
                "aot_completed_dispatches",
                snapshot.aot_completed_dispatches,
            ),
            ("aot_failed_dispatches", snapshot.aot_failed_dispatches),
            ("aot_chunks", snapshot.aot_chunks),
            ("aot_worker_callbacks", snapshot.aot_worker_callbacks),
        ] {
            counters.insert(name.to_owned(), value);
        }
    }
    let mut timings = HashMap::new();
    if matches!(
        snapshot.mode,
        IvpTelemetryMode::Timings | IvpTelemetryMode::Detailed
    ) {
        for (name, stage) in AOT_COLD_STAGES.iter().zip(snapshot.cold.iter()) {
            if stage.recorded {
                timings.insert(
                    format!("cold.{name}.ms"),
                    stage.elapsed.as_secs_f64() * 1_000.0,
                );
            }
        }
        for (name, stage) in AOT_WARM_STAGES.iter().zip(snapshot.warm.iter()) {
            if stage.recorded {
                timings.insert(
                    format!("warm.{name}.ms"),
                    stage.elapsed.as_secs_f64() * 1_000.0,
                );
            }
        }
    }
    (
        metadata,
        snapshot.aot_artifact_keys.clone(),
        counters,
        timings,
    )
}

fn mesh_pass_rows<'py>(
    py: Python<'py>,
    passes: &[rustedscithe_core::numerical::BVP_sci::new::BvpSciMeshPassSnapshot],
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let mut rows = Vec::with_capacity(passes.len());
    for pass in passes {
        let row = PyDict::new(py);
        macro_rules! add {
            ($($field:ident),+ $(,)?) => {$ (
                row.set_item(stringify!($field), pass.$field)?;
            )+ };
        }
        add!(
            mesh_pass,
            mesh_nodes,
            linear_system_dimension,
            newton_passes,
            newton_iterations,
            jacobian_builds,
            backtracking_trials,
            accepted_steps,
            rejected_steps,
            rhs_calls,
            boundary_calls,
            rhs_jacobian_calls,
            parameter_jacobian_calls,
            boundary_jacobian_calls,
            collocation_evaluations,
            defect_probes,
            linear_assemblies,
            factorizations,
            linear_solves,
            mesh_refinements,
            mesh_points_added,
            sparse_symbolic_analyses,
            sparse_symbolic_reuses,
            sparse_numeric_factorizations,
            sparse_triplet_materializations,
            sparse_csc_assemblies,
            banded_core_factorizations,
            banded_core_to_border_solves,
            banded_schur_assemblies,
            banded_border_factorizations,
            banded_structured_factorizations,
            banded_scalar_fallback_factorizations,
            banded_sparse_fallback_factorizations,
            banded_structured_solves,
            banded_scalar_fallback_solves,
            banded_sparse_fallback_solves,
            elapsed_ms,
            newton_ms,
            rhs_callback_ms,
            jacobian_callback_ms,
            parameter_jacobian_callback_ms,
            boundary_callback_ms,
            linear_assembly_ms,
            factorization_ms,
            linear_solve_ms,
            sparse_symbolic_analysis_ms,
            sparse_symbolic_reuse_ms,
            sparse_numeric_factorization_ms,
            sparse_triplet_materialization_ms,
            sparse_csc_assembly_ms,
            banded_core_factorization_ms,
            banded_core_to_border_solve_ms,
            banded_schur_assembly_ms,
            banded_border_factorization_ms,
            banded_structured_factorization_ms,
            banded_scalar_fallback_factorization_ms,
            banded_sparse_fallback_factorization_ms,
            banded_structured_solve_ms,
            banded_scalar_fallback_solve_ms,
            banded_sparse_fallback_solve_ms,
            banded_residual_guard_ms,
            banded_rhs_permutation_ms,
            collocation_ms,
            defect_estimation_ms,
            mesh_refinement_ms,
        );
        rows.push(row);
    }
    Ok(rows)
}

fn newton_trace_rows<'py>(
    py: Python<'py>,
    trace: &[rustedscithe_core::numerical::BVP_sci::new::BvpSciNewtonTraceEntry],
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let mut rows = Vec::with_capacity(trace.len());
    for entry in trace {
        let row = PyDict::new(py);
        row.set_item("mesh_pass", entry.mesh_pass)?;
        row.set_item("mesh_nodes", entry.mesh_nodes)?;
        row.set_item("linear_system_dimension", entry.linear_system_dimension)?;
        row.set_item("iteration", entry.iteration)?;
        row.set_item("residual_before", entry.residual_before)?;
        row.set_item("residual_after", entry.residual_after)?;
        row.set_item("step_inf_norm", entry.step_inf_norm)?;
        row.set_item("affine_cost_before", entry.affine_cost_before)?;
        row.set_item("affine_cost_after", entry.affine_cost_after)?;
        row.set_item("armijo_alpha", entry.armijo_alpha)?;
        row.set_item("backtracking_trials", entry.backtracking_trials)?;
        row.set_item("jacobian_refreshed", entry.jacobian_refreshed)?;
        row.set_item("accepted", entry.accepted)?;
        row.set_item(
            "max_normalized_collocation_residual",
            entry.max_normalized_collocation_residual,
        )?;
        row.set_item(
            "normalized_boundary_residual",
            entry.normalized_boundary_residual,
        )?;
        rows.push(row);
    }
    Ok(rows)
}

fn timing_scope_rows<'py>(
    py: Python<'py>,
    scopes: &[rustedscithe_core::numerical::BVP_sci::new::BvpSciTelemetryScope],
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let mut rows = Vec::with_capacity(scopes.len());
    for scope in scopes {
        let row = PyDict::new(py);
        row.set_item("stage", format!("{:?}", scope.stage).to_ascii_lowercase())?;
        row.set_item("kind", format!("{:?}", scope.kind).to_ascii_lowercase())?;
        row.set_item(
            "parent",
            scope
                .parent
                .map(|parent| format!("{:?}", parent).to_ascii_lowercase()),
        )?;
        row.set_item("calls", scope.calls)?;
        row.set_item("elapsed_ms", scope.elapsed_ms)?;
        rows.push(row);
    }
    Ok(rows)
}

fn aot_stage_rows<'py>(
    py: Python<'py>,
    snapshot: Option<&rustedscithe_core::symbolic::ivp_telemetry::IvpTelemetrySnapshot>,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let Some(snapshot) = snapshot else {
        return Ok(Vec::new());
    };
    let mut rows = Vec::with_capacity(snapshot.cold.len() + snapshot.warm.len());
    for (family, names, stages) in [
        ("cold", AOT_COLD_STAGES, snapshot.cold.as_slice()),
        ("warm", AOT_WARM_STAGES, snapshot.warm.as_slice()),
    ] {
        for (name, stage) in names.iter().zip(stages) {
            if !stage.recorded {
                continue;
            }
            let row = PyDict::new(py);
            row.set_item("family", family)?;
            row.set_item("stage", name)?;
            row.set_item("calls", stage.calls)?;
            row.set_item("elapsed_ms", stage.elapsed.as_secs_f64() * 1_000.0)?;
            rows.push(row);
        }
    }
    Ok(rows)
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
    metrics: HashMap<String, f64>,
    preparation_scope: String,
    aot_metadata: HashMap<String, String>,
    aot_artifact_keys: Vec<String>,
    aot_counters: HashMap<String, u64>,
    aot_timings_ms: HashMap<String, f64>,
    aot_snapshot: Option<rustedscithe_core::symbolic::ivp_telemetry::IvpTelemetrySnapshot>,
    mesh_passes: Vec<rustedscithe_core::numerical::BVP_sci::new::BvpSciMeshPassSnapshot>,
    newton_trace: Vec<rustedscithe_core::numerical::BVP_sci::new::BvpSciNewtonTraceEntry>,
    timing_scopes: Vec<rustedscithe_core::numerical::BVP_sci::new::BvpSciTelemetryScope>,
}

impl BvpSciSolution {
    fn from_native(
        native: CoreSolution,
        snapshot: rustedscithe_core::numerical::BVP_sci::new::BvpSciTelemetrySnapshot,
    ) -> Self {
        let (telemetry_mode, counters, timings_ms) = snapshot_maps(&snapshot);
        let preparation_scope = snapshot.report_preparation_scope().to_owned();
        let metrics = if snapshot.mode.collects_timings() {
            snapshot_metrics(&snapshot)
        } else {
            HashMap::new()
        };
        let (aot_metadata, aot_artifact_keys, aot_counters, aot_timings_ms) =
            aot_telemetry_maps(snapshot.aot.as_ref());
        let aot_snapshot = snapshot.aot;
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
            metrics,
            preparation_scope,
            aot_metadata,
            aot_artifact_keys,
            aot_counters,
            aot_timings_ms,
            aot_snapshot,
            mesh_passes: snapshot.mesh_passes,
            newton_trace: snapshot.newton_residual_history,
            timing_scopes: snapshot.timing_scopes,
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

    /// Return measured discrete stage counters; empty when telemetry is off.
    pub fn telemetry_counters(&self) -> HashMap<String, u64> {
        self.counters.clone()
    }

    /// Return Banded conditioning/residual metrics when the native backend measured them.
    pub fn telemetry_metrics(&self) -> HashMap<String, f64> {
        self.metrics.clone()
    }

    /// Identify whether reported preparation time came from BVP, AOT, or was not measured.
    pub fn telemetry_preparation_scope(&self) -> &str {
        &self.preparation_scope
    }

    /// Return measured stage timings in milliseconds. Unmeasured/inapplicable
    /// scopes are absent, and scopes may overlap. Empty in off/counters modes.
    pub fn telemetry_timings_ms(&self) -> HashMap<String, f64> {
        self.timings_ms.clone()
    }

    /// Return native timing scopes with parent/kind metadata; scopes can overlap.
    pub fn telemetry_timing_scopes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        timing_scope_rows(py, &self.timing_scopes)
    }

    /// Return per-mesh-pass counters and timings; these diagnostic records are
    /// collected only in Detailed mode.
    pub fn telemetry_mesh_passes<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        mesh_pass_rows(py, &self.mesh_passes)
    }

    /// Return the modified-Newton trace collected only in Detailed mode.
    pub fn telemetry_newton_trace<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        newton_trace_rows(py, &self.newton_trace)
    }

    /// Return execution-route and shared native snapshot metadata.
    pub fn telemetry_aot_metadata(&self) -> HashMap<String, String> {
        self.aot_metadata.clone()
    }

    /// Return AOT artifact keys from the lifecycle snapshot without flattening them.
    pub fn telemetry_aot_artifact_keys(&self) -> Vec<String> {
        self.aot_artifact_keys.clone()
    }

    /// Return nested execution/lifecycle counters from the shared native snapshot.
    pub fn telemetry_aot_counters(&self) -> HashMap<String, u64> {
        self.aot_counters.clone()
    }

    /// Return cold/warm preparation and execution stage timings in milliseconds.
    pub fn telemetry_aot_timings_ms(&self) -> HashMap<String, f64> {
        self.aot_timings_ms.clone()
    }

    /// Return measured AOT stages with integer call counts and millisecond durations.
    pub fn telemetry_aot_stages<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        aot_stage_rows(py, self.aot_snapshot.as_ref())
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
    /// Construct a symbolic or numerical-callback BVP_sci problem.
    ///
    /// `expressions` contains one ODE RHS per state variable and
    /// `boundary_expressions` contains the endpoint residual equations.
    /// `mesh` must be strictly increasing; `initial_state` is a flat vector in
    /// node-major order, representing shape `(mesh_nodes, state_dimension)`.
    /// Optional parameter symbols and
    /// values enable continuation. Symbolic RHS expressions use ExprLegacy or
    /// AtomView with Lambdify/AOT. For an already-numerical RHS, pass an empty
    /// `expressions` list, select `frontend="numerical"` and
    /// `execution="numerical"`, then provide `rhs_callback`. That callback
    /// route is Dense-only; state/parameter Jacobians are optional and fall
    /// back to native finite differences when omitted. `options` also selects
    /// matrix layout, factorization and telemetry.
    /// Numeric controls configure residual tolerances, mesh/Newton budgets and
    /// output retention. Boundary callbacks may replace the symbolic boundary
    /// route; singular-term inputs must be supplied as a compatible pair.
    #[new]
    #[pyo3(signature=(expressions, state_variables, boundary_expressions, mesh, initial_state, *, parameters=None, parameter_values=None, independent_variable="x", options=None, tolerance=1e-3, boundary_tolerance=None, max_nodes=1000, max_newton_iterations=8, max_mesh_refinements=10, max_backtracking_steps=4, output="mesh", boundary_jacobian="symbolic", singular_term_a=None, singular_term_matrix=None, rhs_callback=None, rhs_jacobian_callback=None, rhs_parameter_jacobian_callback=None, boundary_callback=None, boundary_jacobian_callback=None))]
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
        rhs_callback: Option<Py<PyAny>>,
        rhs_jacobian_callback: Option<Py<PyAny>>,
        rhs_parameter_jacobian_callback: Option<Py<PyAny>>,
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
        let numerical_rhs = rhs_callback.is_some();
        if numerical_rhs && !expressions.is_empty() {
            return Err(InvalidConfigurationError::new_err(
                "expressions must be empty when rhs_callback is provided",
            ));
        }
        if !numerical_rhs && expressions.len() != state_variables.len() {
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
        if !numerical_rhs {
            validate_expression_symbols(&expressions, &rhs_allowed, "RHS")?;
        }
        for (callback, label) in [
            (&rhs_callback, "rhs_callback"),
            (&rhs_jacobian_callback, "rhs_jacobian_callback"),
            (
                &rhs_parameter_jacobian_callback,
                "rhs_parameter_jacobian_callback",
            ),
        ] {
            if let Some(callback) = callback {
                validate_python_callback(callback, label)?;
                if !numerical_rhs {
                    return Err(InvalidConfigurationError::new_err(format!(
                        "{label} requires rhs_callback"
                    )));
                }
            }
        }

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
        let logging_level = crate::contract::native_logging_level(&options, "bvp_sci")?;
        let assembly = match options.frontend() {
            "expr_legacy" => BvpSciAssembly::ExprLegacy,
            "atom_native" => BvpSciAssembly::AtomViewNative,
            "numerical" => BvpSciAssembly::Numerical,
            other => {
                return Err(UnsupportedOptionError::new_err(format!(
                    "unsupported BVP frontend '{other}'"
                )));
            }
        };
        let numerical_execution = options.execution() == "numerical";
        if numerical_execution != numerical_rhs
            || (assembly == BvpSciAssembly::Numerical) != numerical_rhs
        {
            return Err(InvalidConfigurationError::new_err(
                "BVP_sci frontend='numerical' and execution='numerical' require rhs_callback; symbolic and callback RHS routes cannot be mixed",
            ));
        }
        if numerical_rhs && options.matrix_layout() != "dense" {
            return Err(UnsupportedOptionError::new_err(
                "BVP_sci numerical callbacks currently support only matrix_layout='dense'",
            ));
        }
        if numerical_rhs
            && (options.aot_output_dir().is_some() || options.aot_handoff_path().is_some())
        {
            return Err(UnsupportedOptionError::new_err(
                "AOT artifact settings are not valid for the numerical callback route",
            ));
        }
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
        builder = builder.with_logging_level(logging_level);
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
        if let Some(callback) = rhs_callback {
            let rhs_failure = callback_failure.clone();
            // Python callbacks run under the GIL and receive owned Python
            // sequences. These copies are confined to the explicitly selected
            // Numerical callback route; symbolic routes stay entirely native.
            let mut numerical = BvpSciNumericalPlan::new(
                state_dimension,
                parameter_dimension,
                move |x, state, parameters, output| {
                    let result = Python::attach(|py| -> PyResult<Vec<f64>> {
                        callback
                            .bind(py)
                            .call1((x, state.to_vec(), parameters.to_vec()))?
                            .extract()
                    });
                    let values = match result {
                        Ok(values) => values,
                        Err(error) => {
                            let message = format!("rhs_callback raised: {error}");
                            rhs_failure.record(message.clone(), Some(error));
                            return Err(message);
                        }
                    };
                    if values.len() != output.len() {
                        return Err(format!(
                            "rhs_callback must return {} values, got {}",
                            output.len(),
                            values.len()
                        ));
                    }
                    finite_callback_values(&values, "rhs_callback")?;
                    output.copy_from_slice(&values);
                    Ok(())
                },
                BvpSciTelemetry::disabled(),
            )
            .map_err(map_error)?;
            if let Some(callback) = rhs_jacobian_callback {
                let failure = callback_failure.clone();
                numerical = numerical.with_rhs_jacobian(move |x, state, parameters, output| {
                    let result = Python::attach(|py| -> PyResult<Vec<Vec<f64>>> {
                        callback
                            .bind(py)
                            .call1((x, state.to_vec(), parameters.to_vec()))?
                            .extract()
                    });
                    let rows = match result {
                        Ok(rows) => rows,
                        Err(error) => {
                            let message = format!("rhs_jacobian_callback raised: {error}");
                            failure.record(message.clone(), Some(error));
                            return Err(message);
                        }
                    };
                    let flat = flatten_callback_matrix(
                        rows,
                        state_dimension,
                        state_dimension,
                        "rhs_jacobian_callback",
                    )?;
                    output.copy_from_slice(&flat);
                    Ok(())
                });
            }
            if let Some(callback) = rhs_parameter_jacobian_callback {
                let failure = callback_failure.clone();
                numerical =
                    numerical.with_rhs_parameter_jacobian(move |x, state, parameters, output| {
                        let result = Python::attach(|py| -> PyResult<Vec<Vec<f64>>> {
                            callback
                                .bind(py)
                                .call1((x, state.to_vec(), parameters.to_vec()))?
                                .extract()
                        });
                        let rows = match result {
                            Ok(rows) => rows,
                            Err(error) => {
                                let message =
                                    format!("rhs_parameter_jacobian_callback raised: {error}");
                                failure.record(message.clone(), Some(error));
                                return Err(message);
                            }
                        };
                        let flat = flatten_callback_matrix(
                            rows,
                            state_dimension,
                            parameter_dimension,
                            "rhs_parameter_jacobian_callback",
                        )?;
                        output.copy_from_slice(&flat);
                        Ok(())
                    });
            }
            builder = builder.with_numerical_plan(numerical);
        }
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

    /// Replace the mesh and flat node-major initial guess without rebuilding
    /// the symbolic model. The new mesh must be strictly increasing.
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
    /// RHS frontend used by the prepared BVP solver (`expr_legacy`,
    /// `atom_native` or `numerical`).
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
