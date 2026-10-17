//! Python access to the shared task-document parser.
//!
//! The parser and its semantic rules remain in the core crate.  This module
//! owns only the Python lifetime and exposes the already-rendered preview and
//! validation reports; it deliberately does not invent a second DSL parser.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use nalgebra::{DMatrix, DVector};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use rustedscithe_core::command_interpreter::task_runner::{
    ParsedTaskSpec, TaskDocumentKind, TaskRunResult, TaskRunnerError, parse_task_spec_from_str,
    render_task_check, render_task_preview, run_task_from_str,
};

use rustedscithe_core::command_interpreter::task_parser_bvp::{
    BvpExecutionSpec, BvpFrontendSpec, BvpLinearBackendSpec, BvpOutputPolicy, BvpSolverFamilySpec,
    BvpTaskError, BvpTaskRunResult,
};
use rustedscithe_core::command_interpreter::task_parser_common::{
    ContinuationMode, ContinuationMonotonicity, ContinuationRestartPolicy, ContinuationSpec,
};
use rustedscithe_core::command_interpreter::task_parser_ivp::{
    IvpMethodSpec, IvpRunStatus, IvpTaskError, IvpTaskRunResult,
};

use crate::errors::{BvpError, TaskDocumentError, annotate, catch_panic};

fn task_location(error: &TaskRunnerError) -> (Option<usize>, Option<usize>) {
    match error {
        TaskRunnerError::Ivp(IvpTaskError::Document(parse)) => (parse.line, parse.column),
        TaskRunnerError::Ivp(IvpTaskError::SymbolDiagnostic { line, column, .. }) => {
            (*line, *column)
        }
        TaskRunnerError::Bvp(BvpTaskError::Document(parse)) => (parse.line, parse.column),
        TaskRunnerError::Bvp(BvpTaskError::SymbolDiagnostic { line, column, .. }) => {
            (*line, *column)
        }
        _ => (None, None),
    }
}

fn annotate_task_error(error: PyErr, source: &TaskRunnerError) -> PyErr {
    let (line, column) = task_location(source);
    Python::attach(|py| {
        let value = error.value(py);
        let _ = value.setattr("category", source.category());
        if let Some(line) = line {
            let _ = value.setattr("line", line);
        }
        if let Some(column) = column {
            let _ = value.setattr("column", column);
        }
    });
    error
}

fn map_task_error(error: TaskRunnerError) -> PyErr {
    let mapped = annotate(
        TaskDocumentError::new_err(format!("{}: {error}", error.category())),
        "task",
        "task_parser",
        "native",
    );
    annotate_task_error(mapped, &error)
}

fn map_bvp_task_error(error: TaskRunnerError) -> PyErr {
    match error {
        TaskRunnerError::Bvp(error) => {
            let category = error.category();
            let mapped = annotate(
                BvpError::new_err(format!("{}: {error}", category)),
                "solve",
                "bvp_task",
                "native",
            );
            let wrapped = TaskRunnerError::Bvp(error);
            annotate_task_error(mapped, &wrapped)
        }
        other => map_task_error(other),
    }
}

fn matrix_rows(matrix: Option<&DMatrix<f64>>) -> Vec<Vec<f64>> {
    matrix
        .map(|matrix| {
            matrix
                .row_iter()
                .map(|row| row.iter().copied().collect())
                .collect()
        })
        .unwrap_or_default()
}

fn vector_values(vector: Option<&DVector<f64>>) -> Vec<f64> {
    vector
        .map(|vector| vector.iter().copied().collect())
        .unwrap_or_default()
}

fn status_label(status: Option<&IvpRunStatus>) -> Option<String> {
    status.map(|status| match status {
        IvpRunStatus::Prepared => "prepared".to_owned(),
        IvpRunStatus::Running => "running".to_owned(),
        IvpRunStatus::Finished => "finished".to_owned(),
        IvpRunStatus::Failed => "failed".to_owned(),
        IvpRunStatus::StoppedByCondition => "stopped_by_condition".to_owned(),
        IvpRunStatus::Unknown(value) => value.clone(),
    })
}

/// Detached Python result for one IVP task execution.
///
/// All trajectory data is copied before the core result is released. This
/// keeps a later native continuation/restart independent from Python-visible
/// data and mirrors the ownership policy of the direct solver wrappers.
#[pyclass(name = "IvpTaskResult", module = "rustedscithe")]
pub struct IvpTaskResult {
    times: Vec<f64>,
    states: Vec<Vec<f64>>,
    status: Option<String>,
    status_code: Option<String>,
    continuation: Option<ContinuationSnapshot>,
}

struct ContinuationSnapshot {
    parameter: String,
    parameters: Vec<String>,
    mode: String,
    restart_each: bool,
    fresh_preparations: usize,
    prepared_reuses: usize,
    segments: Vec<ContinuationSegmentSnapshot>,
}

struct ContinuationSegmentSnapshot {
    parameter_values: Vec<f64>,
    times: Vec<f64>,
    states: Vec<Vec<f64>>,
    prepare_ms: f64,
    solve_ms: f64,
}

impl IvpTaskResult {
    fn from_native(result: IvpTaskRunResult) -> Self {
        let continuation = result
            .continuation
            .map(|continuation| ContinuationSnapshot {
                parameter: continuation.parameter,
                parameters: continuation.parameters,
                mode: format!("{:?}", continuation.mode).to_ascii_lowercase(),
                restart_each: continuation.restart_each,
                fresh_preparations: continuation.fresh_preparations,
                prepared_reuses: continuation.prepared_reuses,
                segments: continuation
                    .segments
                    .into_iter()
                    .map(|segment| ContinuationSegmentSnapshot {
                        parameter_values: segment.parameter_values,
                        times: vector_values(segment.t_result.as_ref()),
                        states: matrix_rows(segment.y_result.as_ref()),
                        prepare_ms: segment.prepare_ms,
                        solve_ms: segment.solve_ms,
                    })
                    .collect(),
            });
        Self {
            times: vector_values(result.t_result.as_ref()),
            states: matrix_rows(result.y_result.as_ref()),
            status: result.status,
            status_code: status_label(result.status_code.as_ref()),
            continuation,
        }
    }
}

struct BvpContinuationSnapshot {
    segments: usize,
    fresh_preparations: usize,
    prepared_reuses: usize,
    mesh_restarts: usize,
    initial_guess_restarts: usize,
    peak_result_elements: usize,
    segment_data: Vec<(Vec<f64>, Vec<f64>, Vec<Vec<f64>>)>,
}

struct BvpTelemetrySnapshot {
    mode: String,
    sparse_backend: String,
    superlu_fallbacks: Option<u64>,
    superlu_factorizations: Option<u64>,
    linear_factorizations: Option<u64>,
    linear_solves: Option<u64>,
    timings_ms: HashMap<String, f64>,
}

fn insert_timing(target: &mut HashMap<String, f64>, name: &str, value: Option<f64>) {
    if let Some(value) = value {
        target.insert(name.to_owned(), value);
    }
}

/// Detached result for a BVP task-document execution.
///
/// The task runner owns adaptive meshes, continuation state and native
/// telemetry. This object copies only the final data and compact counters so a
/// Python caller cannot keep a mutable borrow into a Rust solver workspace.
#[pyclass(name = "BvpTaskResult", module = "rustedscithe")]
pub struct BvpTaskResult {
    mesh: Vec<f64>,
    values: Vec<Vec<f64>>,
    solver: String,
    frontend: String,
    execution: String,
    matrix_layout: String,
    status: String,
    status_code: String,
    continuation: BvpContinuationSnapshot,
    telemetry: BvpTelemetrySnapshot,
}

fn bvp_family_name(family: BvpSolverFamilySpec) -> &'static str {
    match family {
        BvpSolverFamilySpec::Damp => "bvp_damp",
        BvpSolverFamilySpec::Sci => "bvp_sci",
    }
}

fn bvp_frontend_name(frontend: Option<BvpFrontendSpec>) -> &'static str {
    match frontend {
        Some(BvpFrontendSpec::AtomViewNative) => "atom_native",
        Some(BvpFrontendSpec::ExprLegacy) | None => "expr_legacy",
    }
}

fn bvp_execution_name(execution: Option<BvpExecutionSpec>) -> &'static str {
    match execution {
        Some(BvpExecutionSpec::Aot) => "aot",
        Some(BvpExecutionSpec::Lambdify) | None => "lambdify",
    }
}

fn bvp_layout_name(layout: &BvpLinearBackendSpec) -> &'static str {
    match layout {
        BvpLinearBackendSpec::Dense => "dense",
        BvpLinearBackendSpec::Sparse => "sparse",
        BvpLinearBackendSpec::Banded => "banded",
    }
}

impl BvpTaskResult {
    fn from_native(result: BvpTaskRunResult) -> Self {
        let specification = result.specification;
        let values = matrix_rows(result.result.as_ref());
        let mesh = result.mesh.unwrap_or_else(|| {
            let n = specification.mesh.n_steps.max(2);
            (0..n)
                .map(|index| {
                    specification.mesh.t0
                        + (specification.mesh.t_end - specification.mesh.t0) * index as f64
                            / (n - 1) as f64
                })
                .collect()
        });
        let continuation = BvpContinuationSnapshot {
            segments: result.continuation_telemetry.segments,
            fresh_preparations: result.continuation_telemetry.fresh_preparations,
            prepared_reuses: result.continuation_telemetry.prepared_reuses,
            mesh_restarts: result.continuation_telemetry.mesh_restarts,
            initial_guess_restarts: result.continuation_telemetry.initial_guess_restarts,
            peak_result_elements: result.continuation_telemetry.peak_result_elements,
            segment_data: result
                .continuation_segments
                .into_iter()
                .map(|segment| {
                    (
                        segment.parameters,
                        segment.mesh,
                        matrix_rows(Some(&segment.result)),
                    )
                })
                .collect(),
        };
        let telemetry = result
            .runtime_telemetry
            .map(|telemetry| {
                let mut timings_ms = HashMap::new();
                if let Some(timings) = telemetry.timings {
                    insert_timing(&mut timings_ms, "preparation_ms", timings.preparation_ms);
                    insert_timing(&mut timings_ms, "rhs_callback_ms", timings.rhs_callback_ms);
                    insert_timing(
                        &mut timings_ms,
                        "jacobian_callback_ms",
                        timings.jacobian_callback_ms,
                    );
                    insert_timing(
                        &mut timings_ms,
                        "residual_evaluation_ms",
                        timings.residual_evaluation_ms,
                    );
                    insert_timing(
                        &mut timings_ms,
                        "jacobian_evaluation_ms",
                        timings.jacobian_evaluation_ms,
                    );
                    insert_timing(
                        &mut timings_ms,
                        "linear_assembly_ms",
                        timings.linear_assembly_ms,
                    );
                    insert_timing(
                        &mut timings_ms,
                        "factorization_ms",
                        timings.factorization_ms,
                    );
                    insert_timing(&mut timings_ms, "linear_solve_ms", timings.linear_solve_ms);
                    insert_timing(&mut timings_ms, "newton_ms", timings.newton_ms);
                    insert_timing(
                        &mut timings_ms,
                        "mesh_refinement_ms",
                        timings.mesh_refinement_ms,
                    );
                    insert_timing(&mut timings_ms, "full_solve_ms", timings.full_solve_ms);
                    insert_timing(
                        &mut timings_ms,
                        "full_solve_total_ms",
                        timings.full_solve_total_ms,
                    );
                    insert_timing(
                        &mut timings_ms,
                        "sparse_csc_assembly_ms",
                        timings.sparse_csc_assembly_ms,
                    );
                }
                BvpTelemetrySnapshot {
                    mode: format!("{:?}", telemetry.mode).to_ascii_lowercase(),
                    sparse_backend: format!("{:?}", telemetry.sparse_backend_outcome)
                        .to_ascii_lowercase(),
                    superlu_fallbacks: telemetry.superlu_fallbacks,
                    superlu_factorizations: telemetry.superlu_factorizations,
                    linear_factorizations: telemetry.linear_factorizations,
                    linear_solves: telemetry.linear_solves,
                    timings_ms,
                }
            })
            .unwrap_or(BvpTelemetrySnapshot {
                mode: "off".to_owned(),
                sparse_backend: "not_observed".to_owned(),
                superlu_fallbacks: None,
                superlu_factorizations: None,
                linear_factorizations: None,
                linear_solves: None,
                timings_ms: HashMap::new(),
            });
        Self {
            mesh,
            values,
            solver: bvp_family_name(specification.solver.family).to_owned(),
            frontend: bvp_frontend_name(specification.solver.frontend).to_owned(),
            execution: bvp_execution_name(specification.solver.execution).to_owned(),
            matrix_layout: bvp_layout_name(&specification.solver.backend).to_owned(),
            status: "finished".to_owned(),
            status_code: "ok".to_owned(),
            continuation,
            telemetry,
        }
    }
}

#[pymethods]
impl BvpTaskResult {
    /// Final adaptive mesh, detached from the native solver workspace.
    pub fn mesh(&self) -> Vec<f64> {
        self.mesh.clone()
    }

    /// State matrix in `state x node` orientation.
    pub fn values(&self) -> Vec<Vec<f64>> {
        self.values.clone()
    }

    #[getter]
    /// Selected solver family, such as `bvp_sci` or `bvp_damp`.
    pub fn solver(&self) -> &str {
        &self.solver
    }

    #[getter]
    /// Symbolic frontend used by the task route.
    pub fn frontend(&self) -> &str {
        &self.frontend
    }

    #[getter]
    /// Runtime execution route, such as Lambdify or AOT.
    pub fn execution(&self) -> &str {
        &self.execution
    }

    #[getter]
    /// Linear-system layout used by the selected BVP backend.
    pub fn matrix_layout(&self) -> &str {
        &self.matrix_layout
    }

    #[getter]
    /// Human-readable task execution status.
    pub fn status(&self) -> &str {
        &self.status
    }

    #[getter]
    /// Stable machine-readable task status code.
    pub fn status_code(&self) -> &str {
        &self.status_code
    }

    /// Return a compact summary of the chosen route, status and final mesh shape.
    pub fn diagnostics(&self) -> HashMap<String, String> {
        HashMap::from([
            ("solver".to_owned(), self.solver.clone()),
            ("frontend".to_owned(), self.frontend.clone()),
            ("execution".to_owned(), self.execution.clone()),
            ("matrix_layout".to_owned(), self.matrix_layout.clone()),
            ("status".to_owned(), self.status.clone()),
            ("status_code".to_owned(), self.status_code.clone()),
            ("mesh_nodes".to_owned(), self.mesh.len().to_string()),
            ("state_dimension".to_owned(), self.values.len().to_string()),
            ("telemetry_mode".to_owned(), self.telemetry.mode.clone()),
        ])
    }

    /// Compact lifecycle metadata: segments, fresh preparations, prepared
    /// reuses, mesh restarts and initial-guess restarts.
    pub fn continuation_info(&self) -> (usize, usize, usize, usize, usize) {
        (
            self.continuation.segments,
            self.continuation.fresh_preparations,
            self.continuation.prepared_reuses,
            self.continuation.mesh_restarts,
            self.continuation.initial_guess_restarts,
        )
    }

    /// Return named continuation lifecycle counters, including the largest
    /// detached segment retained by the native task runner.
    pub fn continuation_telemetry(&self) -> HashMap<String, u64> {
        HashMap::from([
            ("segments".to_owned(), self.continuation.segments as u64),
            (
                "fresh_preparations".to_owned(),
                self.continuation.fresh_preparations as u64,
            ),
            (
                "prepared_reuses".to_owned(),
                self.continuation.prepared_reuses as u64,
            ),
            (
                "mesh_restarts".to_owned(),
                self.continuation.mesh_restarts as u64,
            ),
            (
                "initial_guess_restarts".to_owned(),
                self.continuation.initial_guess_restarts as u64,
            ),
            (
                "peak_result_elements".to_owned(),
                self.continuation.peak_result_elements as u64,
            ),
        ])
    }

    /// Return `(parameter_values, mesh, values)` for each continuation segment.
    pub fn continuation_segments(&self) -> Vec<(Vec<f64>, Vec<f64>, Vec<Vec<f64>>)> {
        self.continuation.segment_data.clone()
    }

    /// Return available linear-backend counters; unavailable counters are omitted.
    pub fn telemetry_counters(&self) -> HashMap<String, u64> {
        let mut counters = HashMap::new();
        for (name, value) in [
            ("superlu_fallbacks", self.telemetry.superlu_fallbacks),
            (
                "superlu_factorizations",
                self.telemetry.superlu_factorizations,
            ),
            (
                "linear_factorizations",
                self.telemetry.linear_factorizations,
            ),
            ("linear_solves", self.telemetry.linear_solves),
        ] {
            if let Some(value) = value {
                counters.insert(name.to_owned(), value);
            }
        }
        counters
    }

    /// Return captured BVP stage timings in milliseconds.
    pub fn telemetry_timings_ms(&self) -> HashMap<String, f64> {
        self.telemetry.timings_ms.clone()
    }

    /// Render a compact report of selected route, backend counters and telemetry scope.
    pub fn telemetry_report(&self) -> String {
        format!(
            "BVP task telemetry\nsolver={}; frontend={}; execution={}; layout={}; mode={}; sparse_backend={}; superlu_fallbacks={}; superlu_factorizations={}; linear_factorizations={}; linear_solves={}",
            self.solver,
            self.frontend,
            self.execution,
            self.matrix_layout,
            self.telemetry.mode,
            self.telemetry.sparse_backend,
            self.telemetry
                .superlu_fallbacks
                .map_or_else(|| "n/a".to_owned(), |v| v.to_string()),
            self.telemetry
                .superlu_factorizations
                .map_or_else(|| "n/a".to_owned(), |v| v.to_string()),
            self.telemetry
                .linear_factorizations
                .map_or_else(|| "n/a".to_owned(), |v| v.to_string()),
            self.telemetry
                .linear_solves
                .map_or_else(|| "n/a".to_owned(), |v| v.to_string()),
        )
    }

    /// Write only the compact telemetry report to a UTF-8 text file.
    pub fn write_telemetry_report(&self, path: &str) -> PyResult<()> {
        crate::telemetry::write_report(path, &self.telemetry_report())
    }
}

#[pymethods]
impl IvpTaskResult {
    /// Accepted output times, including the initial point when available.
    pub fn times(&self) -> Vec<f64> {
        self.times.clone()
    }

    /// State rows corresponding to [`Self::times`].
    pub fn states(&self) -> Vec<Vec<f64>> {
        self.states.clone()
    }

    #[getter]
    /// Human-readable solver status, or `None` when the task result has no status.
    pub fn status(&self) -> Option<String> {
        self.status.clone()
    }

    #[getter]
    /// Stable machine-readable status code, or `None` when unavailable.
    pub fn status_code(&self) -> Option<String> {
        self.status_code.clone()
    }

    /// Return the common solver-independent result snapshot used by batch
    /// clients. Detailed continuation data remains available separately.
    pub fn diagnostics(&self) -> HashMap<String, String> {
        HashMap::from([
            ("solver".to_owned(), "ivp_task".to_owned()),
            (
                "status".to_owned(),
                self.status.clone().unwrap_or_else(|| "unknown".to_owned()),
            ),
            (
                "status_code".to_owned(),
                self.status_code
                    .clone()
                    .unwrap_or_else(|| "unknown".to_owned()),
            ),
            ("telemetry".to_owned(), "native_task_runner".to_owned()),
            ("sample_count".to_owned(), self.times.len().to_string()),
            (
                "state_dimension".to_owned(),
                self.states.first().map_or(0, Vec::len).to_string(),
            ),
            (
                "final_time".to_owned(),
                self.times
                    .last()
                    .map_or_else(|| "none".to_owned(), |value| value.to_string()),
            ),
        ])
    }

    #[getter]
    /// Whether the parsed task executed a continuation series.
    pub fn has_continuation(&self) -> bool {
        self.continuation.is_some()
    }

    /// Stable continuation metadata without exposing Rust enums to Python.
    pub fn continuation_info(&self) -> Option<(String, Vec<String>, String, bool, usize, usize)> {
        self.continuation.as_ref().map(|continuation| {
            (
                continuation.parameter.clone(),
                continuation.parameters.clone(),
                continuation.mode.clone(),
                continuation.restart_each,
                continuation.fresh_preparations,
                continuation.prepared_reuses,
            )
        })
    }

    /// Return `(parameter_values, times, states, prepare_ms, solve_ms)` rows.
    pub fn continuation_segments(&self) -> Vec<(Vec<f64>, Vec<f64>, Vec<Vec<f64>>, f64, f64)> {
        self.continuation
            .as_ref()
            .map(|continuation| {
                continuation
                    .segments
                    .iter()
                    .map(|segment| {
                        (
                            segment.parameter_values.clone(),
                            segment.times.clone(),
                            segment.states.clone(),
                            segment.prepare_ms,
                            segment.solve_ms,
                        )
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// Parsed IVP/BVP task document backed by the core Rust task parser.
///
/// `TaskDocument` delegates parsing and execution to the shared core runner.
/// IVP execution returns a detached Python result; BVP execution remains
/// explicit until its mesh-result adapter has the same ownership contract.
#[pyclass(name = "TaskDocument", module = "rustedscithe", unsendable)]
pub struct TaskDocument {
    source: String,
    spec: ParsedTaskSpec,
}

impl TaskDocument {
    fn from_source(source: String) -> PyResult<Self> {
        if source.trim().is_empty() {
            return Err(TaskDocumentError::new_err(
                "configuration: task document must not be empty",
            ));
        }
        let spec = parse_task_spec_from_str(&source).map_err(map_task_error)?;
        Ok(Self { source, spec })
    }
}

fn ivp_method_label(method: &IvpMethodSpec) -> String {
    match method {
        IvpMethodSpec::NonStiff(name) => name.to_ascii_lowercase(),
        IvpMethodSpec::Radau5 => "radau5".to_owned(),
        IvpMethodSpec::Bdf => "bdf".to_owned(),
        IvpMethodSpec::BackwardEuler => "backward_euler".to_owned(),
        IvpMethodSpec::Lsode => "lsode".to_owned(),
        IvpMethodSpec::Lsoda => "lsoda".to_owned(),
        IvpMethodSpec::Lsode2 => "lsode2".to_owned(),
    }
}

fn bvp_family_label(family: BvpSolverFamilySpec) -> &'static str {
    match family {
        BvpSolverFamilySpec::Damp => "bvp_damp",
        BvpSolverFamilySpec::Sci => "bvp_sci",
    }
}

fn bvp_frontend_label(frontend: Option<BvpFrontendSpec>) -> &'static str {
    match frontend {
        Some(BvpFrontendSpec::ExprLegacy) => "expr_legacy",
        Some(BvpFrontendSpec::AtomViewNative) => "atom_native",
        None => "default",
    }
}

fn bvp_execution_label(execution: Option<BvpExecutionSpec>) -> &'static str {
    match execution {
        Some(BvpExecutionSpec::Lambdify) => "lambdify",
        Some(BvpExecutionSpec::Aot) => "aot",
        None => "default",
    }
}

fn bvp_layout_label(layout: BvpLinearBackendSpec) -> &'static str {
    match layout {
        BvpLinearBackendSpec::Dense => "dense",
        BvpLinearBackendSpec::Sparse => "sparse",
        BvpLinearBackendSpec::Banded => "banded",
    }
}

fn insert_debug_option(
    options: &mut HashMap<String, String>,
    name: &str,
    value: impl std::fmt::Debug,
) {
    options.insert(name.to_owned(), format!("{value:?}"));
}

fn insert_optional_debug<T: std::fmt::Debug>(
    options: &mut HashMap<String, String>,
    name: &str,
    value: Option<T>,
) {
    if let Some(value) = value {
        insert_debug_option(options, name, value);
    }
}

fn continuation_mode_name(mode: ContinuationMode) -> &'static str {
    match mode {
        ContinuationMode::Fresh => "fresh",
        ContinuationMode::Warm => "warm",
        ContinuationMode::Prepared => "prepared",
    }
}

fn continuation_restart_name(policy: ContinuationRestartPolicy) -> &'static str {
    match policy {
        ContinuationRestartPolicy::Continue => "continue",
        ContinuationRestartPolicy::RestartEach => "restart_each",
        ContinuationRestartPolicy::RestartWithState => "restart_with_state",
    }
}

fn continuation_monotonicity_name(value: ContinuationMonotonicity) -> &'static str {
    match value {
        ContinuationMonotonicity::Allow => "allow",
        ContinuationMonotonicity::Increasing => "increasing",
        ContinuationMonotonicity::Decreasing => "decreasing",
    }
}

fn continuation_to_dict<'py>(
    py: Python<'py>,
    continuation: &ContinuationSpec,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("parameters", &continuation.parameters)?;
    result.set_item("value_grid", &continuation.value_grid)?;
    result.set_item("mode", continuation_mode_name(continuation.mode))?;
    result.set_item("restart_each", continuation.restart_each)?;
    result.set_item(
        "restart_policy",
        continuation_restart_name(continuation.restart_policy),
    )?;
    result.set_item(
        "monotonicity",
        continuation_monotonicity_name(continuation.monotonicity),
    )?;
    result.set_item(
        "symbolic_equations",
        continuation
            .symbolic_rhs
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
    )?;
    if let Some(values) = &continuation.y0_values {
        result.set_item("y0_values", values)?;
    }
    if let Some(values) = &continuation.t0_values {
        result.set_item("t0_values", values)?;
    }
    if let Some(values) = &continuation.t_end_values {
        result.set_item("t_end_values", values)?;
    }
    Ok(result)
}

fn postprocessing_to_dict<'py>(
    py: Python<'py>,
    save_csv: bool,
    csv_path: Option<&str>,
    save_txt: bool,
    txt_path: Option<&str>,
    write_report: bool,
    report_path: Option<&str>,
    plotters_png: bool,
    plotters_dir: Option<&str>,
    gnuplot_png: bool,
    gnuplot_dir: Option<&str>,
    terminal_plot: bool,
    plot: bool,
    output_policy: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("save_csv", save_csv)?;
    result.set_item("csv_path", csv_path)?;
    result.set_item("save_txt", save_txt)?;
    result.set_item("txt_path", txt_path)?;
    result.set_item("write_report", write_report)?;
    result.set_item("report_path", report_path)?;
    result.set_item("plotters_png", plotters_png)?;
    result.set_item("plotters_dir", plotters_dir)?;
    result.set_item("gnuplot_png", gnuplot_png)?;
    result.set_item("gnuplot_dir", gnuplot_dir)?;
    result.set_item("terminal_plot", terminal_plot)?;
    result.set_item("plot", plot)?;
    if let Some(policy) = output_policy {
        result.set_item("output_policy", policy)?;
    }
    Ok(result)
}

fn set_python_option(target: &Bound<'_, PyDict>, name: &str, value: &str) -> PyResult<()> {
    match value {
        "true" => target.set_item(name, true),
        "false" => target.set_item(name, false),
        _ => {
            if let Ok(integer) = value.parse::<u64>() {
                return target.set_item(name, integer);
            }
            if let Ok(number) = value.parse::<f64>()
                && number.is_finite()
            {
                return target.set_item(name, number);
            }
            if value.starts_with('[') && value.ends_with(']') {
                let body = &value[1..value.len() - 1];
                if body.trim().is_empty() {
                    return target.set_item(name, Vec::<f64>::new());
                }
                if let Ok(numbers) = body
                    .split(',')
                    .map(|part| part.trim().parse::<f64>())
                    .collect::<Result<Vec<_>, _>>()
                    && numbers.iter().all(|number| number.is_finite())
                {
                    return target.set_item(name, numbers);
                }
            }
            target.set_item(name, value)
        }
    }
}

fn string_map_to_dict<'py>(
    py: Python<'py>,
    values: HashMap<String, String>,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    let mut entries = values.into_iter().collect::<Vec<_>>();
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    for (key, value) in entries {
        result.set_item(key, value)?;
    }
    Ok(result)
}

fn float_map_to_dict<'py>(
    py: Python<'py>,
    values: &HashMap<String, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    let mut entries = values.iter().collect::<Vec<_>>();
    entries.sort_by(|left, right| left.0.cmp(right.0));
    for (key, value) in entries {
        result.set_item(key, value)?;
    }
    Ok(result)
}

#[pymethods]
impl TaskDocument {
    /// Parse an IVP/BVP task document without executing a solver.
    #[staticmethod]
    pub fn parse(source: &str) -> PyResult<Self> {
        Self::from_source(source.to_owned())
    }

    /// Load and parse a task document from a UTF-8 text file.
    #[staticmethod]
    pub fn from_file(path: &str) -> PyResult<Self> {
        let path_ref = Path::new(path);
        let source = fs::read_to_string(path_ref).map_err(|source| {
            TaskDocumentError::new_err(format!(
                "io: failed to read task document `{}`: {source}",
                path_ref.display()
            ))
        })?;
        Self::from_source(source)
    }

    /// Lowercase document family: `ivp` or `bvp`.
    #[getter]
    pub fn kind(&self) -> &'static str {
        match self.spec.kind() {
            TaskDocumentKind::Ivp => "ivp",
            TaskDocumentKind::Bvp => "bvp",
        }
    }

    /// Return the original task source as an owned Python string.
    #[getter]
    pub fn source(&self) -> String {
        self.source.clone()
    }

    /// Render the compact core preview used by the executable task shell.
    pub fn preview(&self) -> String {
        render_task_preview(&self.spec)
    }

    /// Render the core semantic check report used by the executable task shell.
    pub fn check(&self) -> String {
        render_task_check(&self.spec)
    }

    /// Return the normalized execution route before any numerical work starts.
    ///
    /// The map is intentionally small and stable: it describes dispatch and
    /// the explicitly selected symbolic/runtime axes, while solver-specific
    /// numeric options remain owned by the native task parser. In particular,
    /// `dispatch` makes the stiff/nonstiff policy visible instead of leaving a
    /// Python caller to infer it from a method string.
    pub fn route(&self) -> HashMap<String, String> {
        let mut route = HashMap::new();
        match &self.spec {
            ParsedTaskSpec::Ivp(spec) => {
                let method = ivp_method_label(&spec.solver.method);
                let dispatch = if matches!(spec.solver.method, IvpMethodSpec::NonStiff(_)) {
                    "universal_nonstiff"
                } else {
                    "native_stiff"
                };
                route.insert("family".to_owned(), "ivp".to_owned());
                route.insert("method".to_owned(), method);
                route.insert("dispatch".to_owned(), dispatch.to_owned());
                route.insert(
                    "frontend".to_owned(),
                    spec.solver_options
                        .symbolic_assembly
                        .map(|value| value.label().to_owned())
                        .unwrap_or_else(|| "default".to_owned()),
                );
                route.insert(
                    "execution".to_owned(),
                    spec.solver_options
                        .symbolic_execution
                        .as_ref()
                        .map(|value| value.label())
                        .unwrap_or_else(|| "default".to_owned()),
                );
                route.insert(
                    "matrix_layout".to_owned(),
                    spec.solver_options
                        .matrix_layout
                        .map(|value| value.label().to_owned())
                        .unwrap_or_else(|| "default".to_owned()),
                );
                route.insert(
                    "continuation".to_owned(),
                    if spec.continuation.is_some() {
                        "true"
                    } else {
                        "false"
                    }
                    .to_owned(),
                );
            }
            ParsedTaskSpec::Bvp(spec) => {
                route.insert("family".to_owned(), "bvp".to_owned());
                route.insert(
                    "solver".to_owned(),
                    bvp_family_label(spec.solver.family).to_owned(),
                );
                route.insert(
                    "frontend".to_owned(),
                    bvp_frontend_label(spec.solver.frontend).to_owned(),
                );
                route.insert(
                    "execution".to_owned(),
                    bvp_execution_label(spec.solver.execution).to_owned(),
                );
                route.insert(
                    "matrix_layout".to_owned(),
                    bvp_layout_label(spec.solver.backend.clone()).to_owned(),
                );
                route.insert("strategy".to_owned(), format!("{:?}", spec.solver.strategy));
                route.insert("scheme".to_owned(), spec.solver.scheme.clone());
                route.insert(
                    "continuation".to_owned(),
                    if spec.continuation.is_some() {
                        "true"
                    } else {
                        "false"
                    }
                    .to_owned(),
                );
            }
        }
        route
    }

    /// Return the explicitly supplied, normalized solver options.
    ///
    /// This is an inspection surface, not a second configuration API: parsing,
    /// validation and execution still belong to the shared Rust task runner.
    /// Omitted options are absent rather than represented by guessed defaults,
    /// so callers can distinguish an explicit task choice from native defaults.
    pub fn options(&self) -> HashMap<String, String> {
        let mut options = HashMap::new();
        match &self.spec {
            ParsedTaskSpec::Ivp(spec) => {
                let values = &spec.solver_options;
                insert_optional_debug(&mut options, "step_size", values.step_size);
                insert_optional_debug(&mut options, "tolerance", values.tolerance);
                insert_optional_debug(&mut options, "max_iterations", values.max_iterations);
                insert_optional_debug(&mut options, "rtol", values.rtol);
                insert_optional_debug(&mut options, "atol", values.atol);
                insert_optional_debug(&mut options, "max_step", values.max_step);
                insert_optional_debug(&mut options, "first_step", values.first_step);
                insert_optional_debug(&mut options, "vectorized", values.vectorized);
                insert_optional_debug(&mut options, "parallel", values.parallel);
                insert_optional_debug(
                    &mut options,
                    "neighborhood_check",
                    values.neighborhood_check,
                );
                insert_optional_debug(&mut options, "symbolic_assembly", values.symbolic_assembly);
                if let Some(execution) = &values.symbolic_execution {
                    options.insert("symbolic_execution".to_owned(), execution.label());
                }
                insert_optional_debug(&mut options, "matrix_layout", values.matrix_layout);
                insert_optional_debug(&mut options, "execution_policy", values.execution_policy);
                insert_optional_debug(&mut options, "telemetry", values.telemetry);
                if let Some(lsode2) = &values.lsode2 {
                    options.insert("lsode2".to_owned(), format!("{lsode2:?}"));
                }
            }
            ParsedTaskSpec::Bvp(spec) => {
                let values = &spec.solver_options;
                insert_optional_debug(&mut options, "tolerance", values.tolerance);
                insert_optional_debug(
                    &mut options,
                    "boundary_tolerance",
                    values.boundary_tolerance,
                );
                insert_optional_debug(&mut options, "max_iterations", values.max_iterations);
                insert_optional_debug(&mut options, "max_nodes", values.max_nodes);
                insert_optional_debug(
                    &mut options,
                    "max_newton_iterations",
                    values.max_newton_iterations,
                );
                insert_optional_debug(
                    &mut options,
                    "max_jacobian_refreshes",
                    values.max_jacobian_refreshes,
                );
                insert_optional_debug(
                    &mut options,
                    "max_mesh_refinements",
                    values.max_mesh_refinements,
                );
                insert_optional_debug(
                    &mut options,
                    "max_backtracking_steps",
                    values.max_backtracking_steps,
                );
                insert_optional_debug(&mut options, "banded_lower", values.banded_lower);
                insert_optional_debug(&mut options, "banded_upper", values.banded_upper);
                insert_optional_debug(
                    &mut options,
                    "sparse_factorization",
                    values.sparse_factorization,
                );
                insert_optional_debug(&mut options, "execution_policy", values.execution_policy);
                insert_optional_debug(
                    &mut options,
                    "execution_min_work",
                    values.execution_min_work,
                );
                insert_optional_debug(&mut options, "telemetry", values.telemetry);
                insert_optional_debug(&mut options, "output_policy", values.output_policy);
                insert_optional_debug(&mut options, "singular_term_a", values.singular_term_a);
                insert_optional_debug(
                    &mut options,
                    "singular_term_matrix",
                    values.singular_term_matrix.clone(),
                );
                if let Some(value) = &values.linear_sys_method {
                    options.insert("linear_sys_method".to_owned(), value.clone());
                }
                if let Some(value) = &values.loglevel {
                    options.insert("loglevel".to_owned(), value.clone());
                }
                let generated = &values.generated_backend;
                for (name, value) in [
                    ("generated_backend.preset", generated.preset.as_ref()),
                    (
                        "generated_backend.matrix_backend",
                        generated.matrix_backend.as_ref(),
                    ),
                    (
                        "generated_backend.backend_policy",
                        generated.backend_policy.as_ref(),
                    ),
                    (
                        "generated_backend.symbolic_backend",
                        generated.symbolic_backend.as_ref(),
                    ),
                    (
                        "generated_backend.aot_codegen_backend",
                        generated.aot_codegen_backend.as_ref(),
                    ),
                    (
                        "generated_backend.aot_c_compiler",
                        generated.aot_c_compiler.as_ref(),
                    ),
                    (
                        "generated_backend.aot_build_policy",
                        generated.aot_build_policy.as_ref(),
                    ),
                    (
                        "generated_backend.aot_build_profile",
                        generated.aot_build_profile.as_ref(),
                    ),
                    (
                        "generated_backend.aot_compile_preset",
                        generated.aot_compile_preset.as_ref(),
                    ),
                    (
                        "generated_backend.aot_execution_policy",
                        generated.aot_execution_policy.as_ref(),
                    ),
                    (
                        "generated_backend.aot_output_dir",
                        generated.aot_output_dir.as_ref(),
                    ),
                    (
                        "generated_backend.aot_handoff_path",
                        generated.aot_handoff_path.as_ref(),
                    ),
                    (
                        "generated_backend.banded_linear_solver",
                        generated.banded_linear_solver.as_ref(),
                    ),
                ] {
                    if let Some(value) = value {
                        options.insert(name.to_owned(), value.clone());
                    }
                }
                insert_optional_debug(
                    &mut options,
                    "generated_backend.refinement_steps",
                    generated.refinement_steps,
                );
            }
        }
        options
    }

    /// Return a detached version-1 dictionary built from the parsed native
    /// specification. The top-level fields include `kind`, `source`, `solver`,
    /// `problem`, `continuation` and `postprocessing`, plus native
    /// `preview`/`check` text. Solver options use Python booleans, integers,
    /// finite floats and numeric lists where applicable; enum/compiler choices
    /// are strings. Equations are canonical Rust expression strings, and the
    /// original source is retained for exact replay. This method never runs
    /// the task.
    pub fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let result = PyDict::new(py);
        result.set_item("dict_schema_version", 1)?;
        result.set_item("kind", self.kind())?;
        result.set_item("source", &self.source)?;
        result.set_item("preview", self.preview())?;
        result.set_item("check", self.check())?;

        let solver = PyDict::new(py);
        solver.set_item("selection", string_map_to_dict(py, self.route())?)?;
        let options = PyDict::new(py);
        let generated_backend = PyDict::new(py);
        let mut option_entries = self.options().into_iter().collect::<Vec<_>>();
        option_entries.sort_by(|left, right| left.0.cmp(&right.0));
        for (name, value) in option_entries {
            if let Some(field) = name.strip_prefix("generated_backend.") {
                set_python_option(&generated_backend, field, &value)?;
            } else {
                set_python_option(&options, &name, &value)?;
            }
        }
        if !generated_backend.is_empty() {
            options.set_item("generated_backend", generated_backend)?;
        }
        solver.set_item("options", options)?;
        result.set_item("solver", solver)?;

        let problem = PyDict::new(py);
        let postprocessing = match &self.spec {
            ParsedTaskSpec::Ivp(spec) => {
                result.set_item("task_schema_version", spec.schema_version)?;
                problem.set_item("independent_variable", &spec.equations.arg)?;
                problem.set_item("unknowns", &spec.equations.unknowns)?;
                problem.set_item(
                    "equations",
                    spec.equations
                        .rhs
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>(),
                )?;
                problem.set_item(
                    "symbolic_equations",
                    spec.equations
                        .symbolic_rhs
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>(),
                )?;
                problem.set_item("parameters", &spec.equations.parameter_names)?;
                problem.set_item(
                    "parameter_values",
                    float_map_to_dict(py, &spec.equations.parameter_values)?,
                )?;
                let initial_conditions = PyDict::new(py);
                initial_conditions.set_item("t0", spec.initial_conditions.t0)?;
                initial_conditions.set_item("t_end", spec.initial_conditions.t_end)?;
                initial_conditions.set_item("y0", &spec.initial_conditions.y0)?;
                problem.set_item("initial_conditions", initial_conditions)?;
                postprocessing_to_dict(
                    py,
                    spec.postprocessing.save_csv,
                    spec.postprocessing.csv_path.as_deref(),
                    spec.postprocessing.save_txt,
                    spec.postprocessing.txt_path.as_deref(),
                    spec.postprocessing.write_report,
                    spec.postprocessing.report_path.as_deref(),
                    spec.postprocessing.plotters_png,
                    spec.postprocessing.plotters_dir.as_deref(),
                    spec.postprocessing.gnuplot_png,
                    spec.postprocessing.gnuplot_dir.as_deref(),
                    spec.postprocessing.terminal_plot,
                    spec.postprocessing.plot,
                    None,
                )?
            }
            ParsedTaskSpec::Bvp(spec) => {
                problem.set_item("independent_variable", &spec.equations.arg)?;
                problem.set_item("unknowns", &spec.equations.unknowns)?;
                problem.set_item(
                    "equations",
                    spec.equations
                        .rhs
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>(),
                )?;
                problem.set_item("parameters", &spec.equations.parameter_names)?;
                problem.set_item(
                    "parameter_values",
                    float_map_to_dict(py, &spec.equations.parameter_values)?,
                )?;

                let boundary_conditions = PyDict::new(py);
                let mut names = spec
                    .boundary_conditions
                    .conditions
                    .keys()
                    .cloned()
                    .collect::<Vec<_>>();
                names.sort();
                for name in names {
                    let mut conditions = spec.boundary_conditions.conditions[&name].clone();
                    conditions.sort_by_key(|(derivative_order, _)| *derivative_order);
                    let entries = PyList::empty(py);
                    for (derivative_order, value) in conditions {
                        let entry = PyDict::new(py);
                        entry.set_item("derivative_order", derivative_order)?;
                        entry.set_item("value", value)?;
                        entries.append(entry)?;
                    }
                    boundary_conditions.set_item(name, entries)?;
                }
                problem.set_item("boundary_conditions", boundary_conditions)?;

                let mesh = PyDict::new(py);
                mesh.set_item("t0", spec.mesh.t0)?;
                mesh.set_item("t_end", spec.mesh.t_end)?;
                mesh.set_item("n_steps", spec.mesh.n_steps)?;
                problem.set_item("mesh", mesh)?;
                problem.set_item("initial_guess", &spec.initial_guess.values)?;

                let policy = match spec.postprocessing.output_policy {
                    BvpOutputPolicy::None => "none",
                    BvpOutputPolicy::Plotters => "plotters",
                    BvpOutputPolicy::Gnuplot => "gnuplot",
                    BvpOutputPolicy::Terminal => "terminal",
                };
                postprocessing_to_dict(
                    py,
                    spec.postprocessing.save_csv,
                    spec.postprocessing.csv_path.as_deref(),
                    spec.postprocessing.save_txt,
                    spec.postprocessing.txt_path.as_deref(),
                    spec.postprocessing.write_report,
                    spec.postprocessing.report_path.as_deref(),
                    spec.postprocessing.plotters_png,
                    spec.postprocessing.plotters_dir.as_deref(),
                    spec.postprocessing.gnuplot_png,
                    spec.postprocessing.gnuplot_dir.as_deref(),
                    spec.postprocessing.terminal_plot,
                    spec.postprocessing.plot,
                    Some(policy),
                )?
            }
        };
        result.set_item("problem", problem)?;
        result.set_item("postprocessing", postprocessing)?;
        let continuation = match &self.spec {
            ParsedTaskSpec::Ivp(spec) => spec.continuation.as_ref(),
            ParsedTaskSpec::Bvp(spec) => spec.continuation.as_ref(),
        };
        if let Some(continuation) = continuation {
            result.set_item("continuation", continuation_to_dict(py, continuation)?)?;
        } else {
            result.set_item("continuation", py.None())?;
        }
        Ok(result.unbind())
    }

    /// Execute an IVP document through the shared native task runner.
    ///
    /// BVP documents use the explicit [`Self::run_bvp`] method so Python code
    /// cannot accidentally assume that an IVP trajectory has been returned.
    pub fn run(&self) -> PyResult<IvpTaskResult> {
        catch_panic(
            "task_run",
            "ivp_task",
            "native",
            || match run_task_from_str(&self.source).map_err(map_task_error)? {
                TaskRunResult::Ivp(result) => Ok(IvpTaskResult::from_native(result)),
                TaskRunResult::Bvp(_) => Err(TaskDocumentError::new_err(
                    "unsupported_route: use TaskDocument.run_bvp() for BVP documents",
                )),
            },
        )
    }

    /// Execute a BVP document through the same native runner used by the
    /// command-line task shell.  The result contains the actual adaptive mesh,
    /// detached state matrix, continuation lifecycle and compact telemetry.
    pub fn run_bvp(&self) -> PyResult<BvpTaskResult> {
        catch_panic(
            "task_run",
            "bvp_task",
            "native",
            || match run_task_from_str(&self.source).map_err(map_bvp_task_error)? {
                TaskRunResult::Bvp(result) => Ok(BvpTaskResult::from_native(result)),
                TaskRunResult::Ivp(_) => Err(TaskDocumentError::new_err(
                    "unsupported_route: use TaskDocument.run() for IVP documents",
                )),
            },
        )
    }

    fn __repr__(&self) -> String {
        format!("TaskDocument(kind='{}')", self.kind())
    }
}
