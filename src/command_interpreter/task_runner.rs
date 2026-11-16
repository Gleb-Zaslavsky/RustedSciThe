//! Unified task-document runner for IVP, BVP, nonlinear, and fitting tasks.
//!
//! This module sits one level above the family-specific task parsers:
//! - detects the task family from document text or a normalized `DocumentMap`
//! - routes parsing/execution to the corresponding parser
//! - provides a file-based API that is convenient for CLI tools and embedding
//!   into other Rust programs

use crate::command_interpreter::task_json::{
    TaskJsonError, document_map_from_json, document_map_from_value, json_value_from_json,
};
use crate::command_interpreter::task_parser::{DocumentMap, Value};
use crate::command_interpreter::task_parser_bvp::{
    BvpTaskError, BvpTaskRunResult, BvpTaskSpec, parse_bvp_task_from_document,
    parse_bvp_task_from_str, run_bvp_task,
};
use crate::command_interpreter::task_parser_ivp::{
    IvpMethodSpec, IvpTaskError, IvpTaskRunResult, IvpTaskSpec, IvpTelemetryModeSpec,
    Lsode2TaskExecutionSpec, parse_ivp_task_from_document, parse_ivp_task_from_str, run_ivp_task,
    run_ivp_task_with_cancellation_token,
};
use crate::command_interpreter::task_parser_nonlinear::{
    NonlinearTaskError, NonlinearTaskRunError, NonlinearTaskRunResult, NonlinearTaskSpec,
    parse_nonlinear_task_from_document, parse_nonlinear_task_from_text, run_nonlinear_task,
    run_nonlinear_task_with_cancellation_token,
};
use crate::command_interpreter::task_parser_optimization::{
    OptimizationMethodSpec, OptimizationTaskError, OptimizationTaskRunError, OptimizationTaskSpec,
    parse_optimization_task_from_document, parse_optimization_task_from_text,
    run_optimization_task, run_optimization_task_with_cancellation_token,
};
use crate::numerical::cancellation::CancellationToken;
use crate::numerical::optimization::universal_fitting::UniversalFittingResult;
use std::fs;
use std::path::{Path, PathBuf};
use tabled::{Table, Tabled};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskDocumentKind {
    Ivp,
    Bvp,
    Nonlinear,
    Optimization,
}

#[derive(Debug)]
pub enum ParsedTaskSpec {
    Ivp(IvpTaskSpec),
    Bvp(BvpTaskSpec),
    Nonlinear(NonlinearTaskSpec),
    Optimization(OptimizationTaskSpec),
}

/// Whether the parsed task route has an implemented cooperative cancellation
/// path all the way to its native solver safe point.
pub fn task_supports_cooperative_cancellation(spec: &ParsedTaskSpec) -> bool {
    match spec {
        ParsedTaskSpec::Ivp(spec) => {
            crate::command_interpreter::task_parser_ivp::ivp_task_cancellation_supported(spec)
        }
        ParsedTaskSpec::Bvp(_) => true,
        ParsedTaskSpec::Nonlinear(_) => true,
        ParsedTaskSpec::Optimization(spec) => {
            matches!(
                spec.method,
                OptimizationMethodSpec::LevenbergMarquardt
                    | OptimizationMethodSpec::VarPro
                    | OptimizationMethodSpec::Gavin
            )
        }
    }
}

/// Report whether a validated task selects or configures an AOT route.
///
/// Worker policy must inspect normalized task fields, not recursively search
/// user expressions/metadata for key names that happen to contain "aot".
pub fn task_requests_aot(spec: &ParsedTaskSpec) -> bool {
    match spec {
        ParsedTaskSpec::Ivp(spec) => {
            let is_aot = |execution: Option<&Lsode2TaskExecutionSpec>| {
                matches!(execution, Some(Lsode2TaskExecutionSpec::Aot { .. }))
            };
            is_aot(spec.solver_options.symbolic_execution.as_ref())
                || spec
                    .solver_options
                    .lsode2
                    .as_ref()
                    .is_some_and(|options| is_aot(options.symbolic_execution.as_ref()))
        }
        ParsedTaskSpec::Bvp(spec) => {
            let backend = &spec.solver_options.generated_backend;
            matches!(
                spec.solver.execution,
                Some(crate::command_interpreter::task_parser_bvp::BvpExecutionSpec::Aot)
            ) || backend
                .preset
                .as_deref()
                .is_some_and(bvp_preset_requests_aot)
                || backend
                    .backend_policy
                    .as_deref()
                    .is_some_and(|value| value.to_ascii_lowercase().contains("aot"))
                || backend.aot_codegen_backend.is_some()
                || backend.aot_c_compiler.is_some()
                || backend.aot_build_policy.is_some()
                || backend.aot_build_profile.is_some()
                || backend.aot_compile_preset.is_some()
                || backend.aot_execution_policy.is_some()
                || backend.aot_output_dir.is_some()
                || backend.aot_handoff_path.is_some()
        }
        ParsedTaskSpec::Nonlinear(_) | ParsedTaskSpec::Optimization(_) => false,
    }
}

/// AOT compiler identity selected by a validated task route.
///
/// This is intentionally a small policy-facing enum: task documents may pick
/// among worker-approved toolchains, but cannot provide arbitrary executables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TaskAotToolchain {
    Rust,
    Zig,
    Gcc,
    Tcc,
}

/// Resolve the selected AOT toolchain for worker allowlist enforcement.
///
/// `None` means the task does not request AOT. Unsafe path/compiler overrides
/// are rejected separately by the worker before this value is trusted.
pub fn task_aot_toolchain(spec: &ParsedTaskSpec) -> Option<TaskAotToolchain> {
    use crate::command_interpreter::task_parser_bvp::BvpExecutionSpec;
    use crate::numerical::LSODE2::Lsode2AotToolchain;

    match spec {
        ParsedTaskSpec::Ivp(spec) => {
            let execution = if matches!(spec.solver.method, IvpMethodSpec::Lsode2) {
                spec.solver_options
                    .lsode2
                    .as_ref()
                    .and_then(|options| options.symbolic_execution.as_ref())
                    .or(spec.solver_options.symbolic_execution.as_ref())
            } else {
                spec.solver_options.symbolic_execution.as_ref()
            }?;
            match execution {
                Lsode2TaskExecutionSpec::LambdifyExpr => None,
                Lsode2TaskExecutionSpec::Aot { toolchain, .. } => Some(match toolchain {
                    Lsode2AotToolchain::Rust => TaskAotToolchain::Rust,
                    Lsode2AotToolchain::Zig => TaskAotToolchain::Zig,
                    Lsode2AotToolchain::CGcc => TaskAotToolchain::Gcc,
                    Lsode2AotToolchain::CTcc => TaskAotToolchain::Tcc,
                }),
            }
        }
        ParsedTaskSpec::Bvp(spec) => {
            let backend = &spec.solver_options.generated_backend;
            let preset = backend
                .preset
                .as_deref()
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase()
                .replace(['-', ' '], "_");
            let requests_aot = matches!(spec.solver.execution, Some(BvpExecutionSpec::Aot))
                || bvp_preset_requests_aot(&preset)
                || backend
                    .backend_policy
                    .as_deref()
                    .is_some_and(|value| value.to_ascii_lowercase().contains("aot"))
                || backend.aot_codegen_backend.is_some()
                || backend.aot_c_compiler.is_some()
                || backend.aot_build_policy.is_some()
                || backend.aot_build_profile.is_some()
                || backend.aot_compile_preset.is_some()
                || backend.aot_execution_policy.is_some()
                || backend.aot_output_dir.is_some()
                || backend.aot_handoff_path.is_some();
            if !requests_aot {
                return None;
            }

            if let Some(compiler) = backend.aot_c_compiler.as_deref() {
                return match compiler.trim().to_ascii_lowercase().as_str() {
                    "gcc" => Some(TaskAotToolchain::Gcc),
                    "tcc" => Some(TaskAotToolchain::Tcc),
                    _ => None,
                };
            }
            if preset.ends_with("_gcc") {
                return Some(TaskAotToolchain::Gcc);
            }
            if preset.ends_with("_tcc")
                || preset == "sparse_repeated"
                || preset == "banded_repeated"
            {
                return Some(TaskAotToolchain::Tcc);
            }
            if preset.ends_with("_zig") {
                return Some(TaskAotToolchain::Zig);
            }
            match backend
                .aot_codegen_backend
                .as_deref()
                .map(str::to_ascii_lowercase)
            {
                Some(value) if value == "rust" => Some(TaskAotToolchain::Rust),
                Some(value) if value == "zig" => Some(TaskAotToolchain::Zig),
                Some(value) if value == "c" => Some(TaskAotToolchain::Tcc),
                Some(_) => None,
                None => Some(TaskAotToolchain::Tcc),
            }
        }
        ParsedTaskSpec::Nonlinear(_) | ParsedTaskSpec::Optimization(_) => None,
    }
}

fn bvp_preset_requests_aot(preset: &str) -> bool {
    let normalized = preset.trim().to_ascii_lowercase().replace(['-', ' '], "_");
    matches!(
        normalized.as_str(),
        "sparse_aot"
            | "sparse_build_if_missing"
            | "sparse_aot_gcc"
            | "sparse_atomview_gcc"
            | "sparse_aot_tcc"
            | "sparse_atomview_tcc"
            | "sparse_repeated"
            | "sparse_aot_zig"
            | "sparse_atomview_zig"
            | "banded_aot"
            | "banded_build_if_missing"
            | "banded_aot_gcc"
            | "banded_atomview_gcc"
            | "banded_aot_tcc"
            | "banded_atomview_tcc"
            | "banded_repeated"
            | "banded_aot_zig"
            | "banded_atomview_zig"
    )
}

pub enum TaskRunResult {
    Ivp(IvpTaskRunResult),
    Bvp(BvpTaskRunResult),
    Nonlinear(NonlinearTaskRunResult),
    Optimization(UniversalFittingResult),
}

#[derive(Debug)]
pub enum TaskRunnerError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    MissingSolverField,
    UnsupportedSolver {
        value: String,
    },
    CancellationUnsupported {
        family: &'static str,
    },
    Json(TaskJsonError),
    Ivp(IvpTaskError),
    Bvp(BvpTaskError),
    NonlinearParse(NonlinearTaskError),
    NonlinearRun(NonlinearTaskRunError),
    NonlinearRunTelemetryUnavailable {
        source: NonlinearTaskRunError,
        requested_mode: crate::command_interpreter::task_parser::TaskTelemetryMode,
    },
    OptimizationParse(OptimizationTaskError),
    OptimizationRun(OptimizationTaskRunError),
    OptimizationRunTelemetryUnavailable {
        source: OptimizationTaskRunError,
        requested_mode: crate::command_interpreter::task_parser::TaskTelemetryMode,
    },
}

impl std::fmt::Display for TaskRunnerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(
                    f,
                    "failed to read task document `{}`: {source}",
                    path.display()
                )
            }
            Self::MissingSolverField => write!(
                f,
                "missing task family: expected `task.solver` for IVP/BVP, `problem` and `solver` for nonlinear, or `data` and `model` for optimization"
            ),
            Self::UnsupportedSolver { value } => {
                write!(f, "unsupported solver `{value}` in task document")
            }
            Self::CancellationUnsupported { family } => {
                write!(
                    f,
                    "cooperative cancellation is unsupported for {family} tasks"
                )
            }
            Self::Json(error) => write!(f, "{error}"),
            Self::Ivp(err) => write!(f, "{err}"),
            Self::Bvp(err) => write!(f, "{err}"),
            Self::NonlinearParse(err) => write!(f, "{err}"),
            Self::NonlinearRun(err) => write!(f, "{err}"),
            Self::NonlinearRunTelemetryUnavailable { source, .. } => write!(f, "{source}"),
            Self::OptimizationParse(err) => write!(f, "{err}"),
            Self::OptimizationRun(err) => write!(f, "{err}"),
            Self::OptimizationRunTelemetryUnavailable { source, .. } => write!(f, "{source}"),
        }
    }
}

impl std::error::Error for TaskRunnerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Json(error) => Some(error),
            Self::Ivp(err) => Some(err),
            Self::Bvp(err) => Some(err),
            Self::NonlinearParse(err) => Some(err),
            Self::NonlinearRun(err) => Some(err),
            Self::NonlinearRunTelemetryUnavailable { source, .. } => Some(source),
            Self::OptimizationParse(err) => Some(err),
            Self::OptimizationRun(err) => Some(err),
            Self::OptimizationRunTelemetryUnavailable { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl TaskRunnerError {
    /// Stable compact category used by batch reports and machine-readable
    /// wrappers. The full typed error remains available through `Display`.
    pub fn category(&self) -> &'static str {
        match self {
            Self::Io { .. } => "io",
            Self::MissingSolverField | Self::UnsupportedSolver { .. } | Self::Json(_) => "parse",
            Self::CancellationUnsupported { .. } => "unsupported_route",
            Self::Ivp(error) => error.category(),
            Self::Bvp(error) => error.category(),
            Self::NonlinearParse(_) | Self::OptimizationParse(_) => "configuration",
            Self::NonlinearRun(
                crate::command_interpreter::task_parser_nonlinear::NonlinearTaskRunError::Root(
                    crate::numerical::Nonlinear_systems::error::SolveError::Cancelled { .. },
                ),
            ) => "cancelled",
            Self::NonlinearRun(
                crate::command_interpreter::task_parser_nonlinear::NonlinearTaskRunError::LeastSquares(
                    crate::numerical::Nonlinear_systems::least_squares::LeastSquaresError::Cancelled { .. },
                ),
            ) => "cancelled",
            Self::OptimizationRun(
                crate::command_interpreter::task_parser_optimization::OptimizationTaskRunError(
                    crate::numerical::optimization::universal_fitting::UniversalFittingError::Fitting(
                        crate::numerical::optimization::sym_fitting::FittingError::Solver(
                            crate::numerical::Nonlinear_systems::least_squares::LeastSquaresError::Cancelled { .. },
                        ),
                    ),
                ),
            )
            | Self::OptimizationRun(
                crate::command_interpreter::task_parser_optimization::OptimizationTaskRunError(
                    crate::numerical::optimization::universal_fitting::UniversalFittingError::SolverConfiguration(
                        crate::numerical::Nonlinear_systems::least_squares::LeastSquaresError::Cancelled { .. },
                    ),
                ),
            ) => "cancelled",
            Self::NonlinearRunTelemetryUnavailable { .. }
            | Self::OptimizationRunTelemetryUnavailable { .. }
            | Self::NonlinearRun(_)
            | Self::OptimizationRun(_) => "solver",
        }
    }

    /// Stable process exit code for the executable task shell.
    ///
    /// Batch execution has its own aggregate code in the CLI; this mapping is
    /// for one task and keeps parse, solver, and output failures distinguishable
    /// to scripts without exposing solver-specific error types there.
    pub fn exit_code(&self) -> i32 {
        let category = self.category();
        match category {
            "io" | "postprocess" => 5,
            "solver" | "cancelled" | "solver_result" | "solver_exhaustion" | "solver_singular"
            | "solver_tolerance" | "solver_linear" | "callback" | "aot_compiler" | "aot_link"
            | "aot_timeout" | "aot_artifact" | "aot_publication" | "aot_runtime"
            | "aot_handoff" => 6,
            _ => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchTaskStatus {
    Passed,
    Failed,
}

impl BatchTaskStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Passed => "passed",
            Self::Failed => "failed",
        }
    }
}

/// One compact result row from a batch task run.
#[derive(Debug)]
pub struct BatchTaskSummary {
    pub path: PathBuf,
    pub status: BatchTaskStatus,
    pub kind: Option<TaskDocumentKind>,
    pub error_category: Option<&'static str>,
    pub error: Option<String>,
}

#[derive(Debug, Tabled)]
struct BatchTaskSummaryRow {
    path: String,
    status: String,
    kind: String,
    error_category: String,
    error: String,
}

#[derive(Debug, Default)]
pub struct BatchTaskReport {
    pub entries: Vec<BatchTaskSummary>,
}

impl BatchTaskReport {
    pub fn passed_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.status == BatchTaskStatus::Passed)
            .count()
    }

    pub fn failed_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.status == BatchTaskStatus::Failed)
            .count()
    }

    /// Renders only the stable result table; cargo/compiler diagnostics are
    /// deliberately outside this API and belong in a technical log.
    pub fn render_table(&self) -> String {
        let rows = self
            .entries
            .iter()
            .map(|entry| BatchTaskSummaryRow {
                path: entry.path.display().to_string(),
                status: entry.status.as_str().to_string(),
                kind: entry
                    .kind
                    .map(task_kind_label)
                    .unwrap_or("unknown")
                    .to_string(),
                error_category: entry.error_category.unwrap_or("-").to_string(),
                error: entry.error.as_deref().unwrap_or("-").to_string(),
            })
            .collect::<Vec<_>>();
        Table::new(rows).to_string()
    }

    /// Renders the same stable batch facts as a small machine-readable JSON
    /// document. This intentionally avoids compiler and solver trace output.
    pub fn render_json(&self) -> String {
        let entries = self
            .entries
            .iter()
            .map(|entry| {
                let kind = entry
                    .kind
                    .map(task_kind_label)
                    .map(json_string)
                    .unwrap_or_else(|| "null".to_string());
                let error_category = entry
                    .error_category
                    .map(json_string)
                    .unwrap_or_else(|| "null".to_string());
                let error = entry
                    .error
                    .as_deref()
                    .map(json_string)
                    .unwrap_or_else(|| "null".to_string());
                format!(
                    "    {{\"path\":{},\"status\":{},\"kind\":{},\"error_category\":{},\"error\":{}}}",
                    json_string(&entry.path.display().to_string()),
                    json_string(entry.status.as_str()),
                    kind,
                    error_category,
                    error
                )
            })
            .collect::<Vec<_>>()
            .join(",\n");
        format!(
            "{{\n  \"passed\": {},\n  \"failed\": {},\n  \"entries\": [\n{}\n  ]\n}}\n",
            self.passed_count(),
            self.failed_count(),
            entries
        )
    }

    pub fn write_table(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        self.write_rendered(path, &self.render_table())
    }

    pub fn write_json(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        self.write_rendered(path, &self.render_json())
    }

    fn write_rendered(&self, path: impl AsRef<Path>, contents: &str) -> std::io::Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, contents)
    }
}

fn json_string(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len() + 2);
    escaped.push('"');
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            character if character.is_control() => {
                escaped.push_str(&format!("\\u{:04x}", character as u32))
            }
            character => escaped.push(character),
        }
    }
    escaped.push('"');
    escaped
}

/// Runs every task independently and continues after parse, I/O, or solver
/// failures. This is the library counterpart of a night-runner script.
pub fn run_task_batch<I, P>(paths: I) -> BatchTaskReport
where
    I: IntoIterator<Item = P>,
    P: Into<PathBuf>,
{
    let entries = paths
        .into_iter()
        .map(Into::into)
        .map(|path| {
            let path = resolve_task_path(path);
            // Detect the family before execution so a semantic or solver
            // failure still reports `ivp`/`bvp` instead of losing context.
            let source = match fs::read_to_string(&path) {
                Ok(source) => source,
                Err(source) => {
                    let error = TaskRunnerError::Io {
                        path: path.clone(),
                        source,
                    };
                    return BatchTaskSummary {
                        path,
                        status: BatchTaskStatus::Failed,
                        kind: None,
                        error_category: Some(error.category()),
                        error: Some(error.to_string()),
                    };
                }
            };
            let json_input = path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("json"));
            let detected_kind = if json_input {
                detect_task_kind_from_json(&source).ok()
            } else {
                detect_task_kind_from_str(&source).ok()
            };
            let parsed = if json_input {
                parse_task_spec_from_json(&source)
            } else {
                parse_task_spec_from_str(&source)
            };
            match parsed.map(|mut spec| {
                anchor_task_paths(&mut spec, &path);
                spec
            }) {
                Ok(spec) => match run_task_from_spec(spec) {
                    Ok(result) if result.is_successful() => BatchTaskSummary {
                        path,
                        status: BatchTaskStatus::Passed,
                        kind: detected_kind,
                        error_category: None,
                        error: None,
                    },
                    Ok(result) => BatchTaskSummary {
                        path,
                        status: BatchTaskStatus::Failed,
                        kind: detected_kind,
                        error_category: Some("solver_result"),
                        error: result.unsuccessful_reason(),
                    },
                    Err(error) => BatchTaskSummary {
                        path,
                        status: BatchTaskStatus::Failed,
                        kind: detected_kind,
                        error_category: Some(error.category()),
                        error: Some(error.to_string()),
                    },
                },
                Err(error) => BatchTaskSummary {
                    path,
                    status: BatchTaskStatus::Failed,
                    kind: detected_kind,
                    error_category: Some(error.category()),
                    error: Some(error.to_string()),
                },
            }
        })
        .collect();
    BatchTaskReport { entries }
}

/// Resolves task paths for stable batch reports while retaining non-existing
/// paths as absolute diagnostics instead of hiding them behind the current
/// working directory.
fn resolve_task_path(path: PathBuf) -> PathBuf {
    let absolute = if path.is_absolute() {
        path
    } else {
        std::env::current_dir()
            .map(|directory| directory.join(&path))
            .unwrap_or_else(|_| PathBuf::from(".").join(path))
    };
    fs::canonicalize(&absolute).unwrap_or(absolute)
}

/// Anchors task-relative output paths to the task document directory.
///
/// Task documents are often launched from a project root, a report directory,
/// or a batch runner. Resolving their output paths once at the runner boundary
/// prevents those launch locations from changing where artifacts are written.
pub fn anchor_task_paths(spec: &mut ParsedTaskSpec, task_path: impl AsRef<Path>) {
    let task_path = task_path.as_ref();
    let absolute_task_path = if task_path.is_absolute() {
        task_path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|directory| directory.join(task_path))
            .unwrap_or_else(|_| task_path.to_path_buf())
    };
    let base = absolute_task_path
        .parent()
        .unwrap_or_else(|| Path::new("."));
    match spec {
        ParsedTaskSpec::Ivp(spec) => {
            anchor_enabled_default(
                &mut spec.postprocessing.csv_path,
                spec.postprocessing.save_csv,
                "ivp_result.csv",
                base,
            );
            anchor_enabled_default(
                &mut spec.postprocessing.txt_path,
                spec.postprocessing.save_txt,
                "ivp_result.txt",
                base,
            );
            anchor_enabled_default(
                &mut spec.postprocessing.report_path,
                spec.postprocessing.write_report,
                "ivp_report.md",
                base,
            );
            anchor_enabled_default(
                &mut spec.postprocessing.plotters_dir,
                spec.postprocessing.plotters_png,
                "ivp_plotters",
                base,
            );
            anchor_enabled_default(
                &mut spec.postprocessing.gnuplot_dir,
                spec.postprocessing.gnuplot_png,
                "ivp_gnuplot",
                base,
            );
        }
        ParsedTaskSpec::Bvp(spec) => {
            anchor_enabled_default(
                &mut spec.postprocessing.csv_path,
                spec.postprocessing.save_csv,
                "bvp_result.csv",
                base,
            );
            anchor_enabled_default(
                &mut spec.postprocessing.txt_path,
                spec.postprocessing.save_txt,
                "bvp_result.txt",
                base,
            );
            anchor_enabled_default(
                &mut spec.postprocessing.report_path,
                spec.postprocessing.write_report,
                "bvp_report.md",
                base,
            );
            anchor_enabled_default(
                &mut spec.postprocessing.plotters_dir,
                spec.postprocessing.plotters_png,
                "bvp_plotters",
                base,
            );
            anchor_enabled_default(
                &mut spec.postprocessing.gnuplot_dir,
                spec.postprocessing.gnuplot_png,
                "bvp_gnuplot",
                base,
            );
            anchor_optional_string(
                &mut spec.solver_options.generated_backend.aot_output_dir,
                base,
            );
            anchor_optional_string(
                &mut spec.solver_options.generated_backend.aot_handoff_path,
                base,
            );
        }
        ParsedTaskSpec::Nonlinear(_) | ParsedTaskSpec::Optimization(_) => {}
    }
}

fn anchor_optional_string(value: &mut Option<String>, base: &Path) {
    let Some(value_string) = value.as_mut() else {
        return;
    };
    let path = Path::new(value_string);
    if path.is_relative() {
        *value_string = base.join(path).to_string_lossy().into_owned();
    }
}

fn anchor_enabled_default(value: &mut Option<String>, enabled: bool, default: &str, base: &Path) {
    if enabled && value.is_none() {
        *value = Some(base.join(default).to_string_lossy().into_owned());
    }
    anchor_optional_string(value, base);
}

/// Returns user-visible filesystem outputs, excluding terminal rendering and
/// reusable AOT cache directories. The CLI uses this list to prevent an
/// accidental single-run overwrite unless the user explicitly opts in.
pub fn requested_output_paths(spec: &ParsedTaskSpec) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    match spec {
        ParsedTaskSpec::Ivp(spec) => {
            if spec.postprocessing.save_csv {
                paths.push(path_from_option_or_default(
                    &spec.postprocessing.csv_path,
                    "ivp_result.csv",
                ));
            }
            if spec.postprocessing.save_txt {
                paths.push(path_from_option_or_default(
                    &spec.postprocessing.txt_path,
                    "ivp_result.txt",
                ));
            }
            if spec.postprocessing.write_report {
                paths.push(path_from_option_or_default(
                    &spec.postprocessing.report_path,
                    "ivp_report.md",
                ));
            }
            if spec.postprocessing.plotters_png {
                paths.push(path_from_option_or_default(
                    &spec.postprocessing.plotters_dir,
                    "ivp_plotters",
                ));
            }
            if spec.postprocessing.gnuplot_png {
                paths.push(path_from_option_or_default(
                    &spec.postprocessing.gnuplot_dir,
                    "ivp_gnuplot",
                ));
            }
        }
        ParsedTaskSpec::Bvp(spec) => {
            if spec.postprocessing.save_csv {
                paths.push(path_from_option_or_default(
                    &spec.postprocessing.csv_path,
                    "bvp_result.csv",
                ));
            }
            if spec.postprocessing.save_txt {
                paths.push(path_from_option_or_default(
                    &spec.postprocessing.txt_path,
                    "bvp_result.txt",
                ));
            }
            if spec.postprocessing.write_report {
                paths.push(path_from_option_or_default(
                    &spec.postprocessing.report_path,
                    "bvp_report.md",
                ));
            }
            if spec.postprocessing.plotters_png
                || matches!(
                    spec.postprocessing.output_policy,
                    crate::command_interpreter::task_parser_bvp::BvpOutputPolicy::Plotters
                )
            {
                paths.push(path_from_option_or_default(
                    &spec.postprocessing.plotters_dir,
                    "bvp_plotters",
                ));
            }
            if spec.postprocessing.gnuplot_png
                || matches!(
                    spec.postprocessing.output_policy,
                    crate::command_interpreter::task_parser_bvp::BvpOutputPolicy::Gnuplot
                )
            {
                paths.push(path_from_option_or_default(
                    &spec.postprocessing.gnuplot_dir,
                    "bvp_gnuplot",
                ));
            }
        }
        ParsedTaskSpec::Nonlinear(_) | ParsedTaskSpec::Optimization(_) => {}
    }
    paths
}

fn path_from_option_or_default(value: &Option<String>, default: &str) -> PathBuf {
    value
        .as_deref()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default))
}

fn task_kind_label(kind: TaskDocumentKind) -> &'static str {
    match kind {
        TaskDocumentKind::Ivp => "ivp",
        TaskDocumentKind::Bvp => "bvp",
        TaskDocumentKind::Nonlinear => "nonlinear",
        TaskDocumentKind::Optimization => "optimization",
    }
}

impl TaskRunResult {
    pub fn kind(&self) -> TaskDocumentKind {
        match self {
            Self::Ivp(_) => TaskDocumentKind::Ivp,
            Self::Bvp(_) => TaskDocumentKind::Bvp,
            Self::Nonlinear(_) => TaskDocumentKind::Nonlinear,
            Self::Optimization(_) => TaskDocumentKind::Optimization,
        }
    }

    /// Whether the selected native solver reports a successful numerical
    /// termination. A completed execution is not necessarily convergence.
    pub fn is_successful(&self) -> bool {
        match self {
            Self::Ivp(result) => matches!(
                result.status_code.as_ref(),
                Some(
                    crate::command_interpreter::task_parser_ivp::IvpRunStatus::Finished
                        | crate::command_interpreter::task_parser_ivp::IvpRunStatus::StoppedByCondition
                )
            ),
            Self::Bvp(result) => result.result.is_some(),
            Self::Nonlinear(result) => match result {
                NonlinearTaskRunResult::Root(result) => matches!(
                    &result.termination,
                    crate::numerical::Nonlinear_systems::error::TerminationReason::Converged
                ),
                NonlinearTaskRunResult::LeastSquares { report, .. } => {
                    report.termination.was_successful()
                }
            },
            Self::Optimization(_) => true,
        }
    }

    /// Compact explanation when execution completed without convergence.
    pub fn unsuccessful_reason(&self) -> Option<String> {
        if self.is_successful() {
            return None;
        }
        Some(match self {
            Self::Ivp(result) => format!(
                "IVP solver terminated with status `{}`",
                result.status.as_deref().unwrap_or("unknown")
            ),
            Self::Bvp(_) => "BVP solver returned no solution".to_string(),
            Self::Nonlinear(NonlinearTaskRunResult::Root(result)) => {
                format!("nonlinear root solver terminated: {:?}", result.termination)
            }
            Self::Nonlinear(NonlinearTaskRunResult::LeastSquares { report, .. }) => {
                format!(
                    "nonlinear least-squares terminated: {:?}",
                    report.termination
                )
            }
            Self::Optimization(_) => "optimization did not produce a successful fit".to_string(),
        })
    }
}

impl std::fmt::Debug for TaskRunResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ivp(result) => f.debug_tuple("Ivp").field(result).finish(),
            Self::Bvp(result) => f.debug_tuple("Bvp").field(result).finish(),
            Self::Nonlinear(result) => f.debug_tuple("Nonlinear").field(result).finish(),
            Self::Optimization(result) => f
                .debug_struct("Optimization")
                .field("solution_map", &result.solution_map())
                .field("r_squared", &result.r_squared())
                .finish(),
        }
    }
}

impl ParsedTaskSpec {
    pub fn kind(&self) -> TaskDocumentKind {
        match self {
            Self::Ivp(_) => TaskDocumentKind::Ivp,
            Self::Bvp(_) => TaskDocumentKind::Bvp,
            Self::Nonlinear(_) => TaskDocumentKind::Nonlinear,
            Self::Optimization(_) => TaskDocumentKind::Optimization,
        }
    }
}

pub fn detect_task_kind_from_str(input: &str) -> Result<TaskDocumentKind, TaskRunnerError> {
    let headings = input
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#') && !line.contains(':'))
        .map(str::to_ascii_lowercase)
        .collect::<std::collections::HashSet<_>>();
    if headings.contains("data") && headings.contains("model") {
        return Ok(TaskDocumentKind::Optimization);
    }
    if headings.contains("problem") && headings.contains("solver") {
        return Ok(TaskDocumentKind::Nonlinear);
    }
    for line in input.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some((key, value)) = trimmed.split_once(':') else {
            continue;
        };
        if !key.trim().eq_ignore_ascii_case("solver") {
            continue;
        }
        let solver = value.trim().to_ascii_lowercase();
        return match solver.as_str() {
            "ivp" => Ok(TaskDocumentKind::Ivp),
            "bvp"
            | "bvp_damp"
            | "bvp_damped"
            | "bvp_sci"
            | "bvpsci"
            | "bvp_scipy"
            | "scipy_bvp"
            | "bvp_correctionnewton"
            | "bvp_correction_newton"
            | "correction_newton" => Ok(TaskDocumentKind::Bvp),
            _ => Err(TaskRunnerError::UnsupportedSolver { value: solver }),
        };
    }
    Err(TaskRunnerError::MissingSolverField)
}

pub fn parse_task_spec_from_str(input: &str) -> Result<ParsedTaskSpec, TaskRunnerError> {
    match detect_task_kind_from_str(input)? {
        TaskDocumentKind::Ivp => parse_ivp_task_from_str(input)
            .map(ParsedTaskSpec::Ivp)
            .map_err(TaskRunnerError::Ivp),
        TaskDocumentKind::Bvp => parse_bvp_task_from_str(input)
            .map(ParsedTaskSpec::Bvp)
            .map_err(TaskRunnerError::Bvp),
        TaskDocumentKind::Nonlinear => parse_nonlinear_task_from_text(input)
            .map(ParsedTaskSpec::Nonlinear)
            .map_err(TaskRunnerError::NonlinearParse),
        TaskDocumentKind::Optimization => parse_optimization_task_from_text(input)
            .map(ParsedTaskSpec::Optimization)
            .map_err(TaskRunnerError::OptimizationParse),
    }
}

/// Dispatch an already normalized document map to the corresponding native
/// task parser. JSON adapters should convert wire values into [`DocumentMap`]
/// and call this function rather than routing through Python or serializing
/// the map back into the human-readable DSL.
pub fn parse_task_spec_from_document(
    document: &DocumentMap,
) -> Result<ParsedTaskSpec, TaskRunnerError> {
    match detect_task_kind_from_document(document)? {
        TaskDocumentKind::Ivp => parse_ivp_task_from_document(document)
            .map(ParsedTaskSpec::Ivp)
            .map_err(TaskRunnerError::Ivp),
        TaskDocumentKind::Bvp => parse_bvp_task_from_document(document)
            .map(ParsedTaskSpec::Bvp)
            .map_err(TaskRunnerError::Bvp),
        TaskDocumentKind::Nonlinear => parse_nonlinear_task_from_document(document)
            .map(ParsedTaskSpec::Nonlinear)
            .map_err(TaskRunnerError::NonlinearParse),
        TaskDocumentKind::Optimization => parse_optimization_task_from_document(document)
            .map(ParsedTaskSpec::Optimization)
            .map_err(TaskRunnerError::OptimizationParse),
    }
}

/// Parse a JSON task document through the native family-specific parsers.
pub fn parse_task_spec_from_json(input: &str) -> Result<ParsedTaskSpec, TaskRunnerError> {
    let mut json = json_value_from_json(input).map_err(TaskRunnerError::Json)?;
    let document = document_map_from_value(&json).map_err(TaskRunnerError::Json)?;
    let kind = detect_task_kind_from_document(&document)?;
    validate_json_field_shapes(kind, &json).map_err(TaskRunnerError::Json)?;
    normalize_json_numeric_fields(kind, &mut json);
    normalize_integral_json_fields(kind, &mut json);
    let document = document_map_from_value(&json).map_err(TaskRunnerError::Json)?;
    validate_json_document_fields(kind, &document).map_err(TaskRunnerError::Json)?;
    parse_task_spec_from_document(&document)
}

#[derive(Clone, Copy)]
enum JsonFieldShape {
    String,
    Number,
    Integer,
    Boolean,
    NullableNumber,
    StringOrArray,
    NumberOrArray,
}

fn validate_json_field_shapes(
    kind: TaskDocumentKind,
    root: &serde_json::Value,
) -> Result<(), TaskJsonError> {
    use JsonFieldShape::*;

    fn shape_matches(value: &serde_json::Value, shape: JsonFieldShape) -> bool {
        match shape {
            JsonFieldShape::String => value.is_string(),
            JsonFieldShape::Number => value.is_number(),
            JsonFieldShape::Integer => {
                value.as_i64().is_some()
                    || value.as_u64().is_some()
                    || value
                        .as_f64()
                        .is_some_and(|number| number.is_finite() && number.fract() == 0.0)
            }
            JsonFieldShape::Boolean => value.is_boolean(),
            JsonFieldShape::NullableNumber => value.is_null() || value.is_number(),
            JsonFieldShape::StringOrArray => {
                value.is_string()
                    || value
                        .as_array()
                        .is_some_and(|values| values.iter().all(serde_json::Value::is_string))
            }
            JsonFieldShape::NumberOrArray => {
                value.is_number()
                    || value
                        .as_array()
                        .is_some_and(|values| values.iter().all(serde_json::Value::is_number))
            }
        }
    }

    fn fields(
        root: &serde_json::Value,
        section: &str,
        declarations: &[(&str, JsonFieldShape)],
    ) -> Result<(), TaskJsonError> {
        let Some(object) = root.get(section).and_then(serde_json::Value::as_object) else {
            return Ok(());
        };
        for (name, shape) in declarations {
            if let Some(value) = object.get(*name) {
                if !shape_matches(value, *shape) {
                    return Err(TaskJsonError::InvalidShape {
                        path: format!("$.{section}.{name}"),
                        expected: match shape {
                            JsonFieldShape::String => "a string",
                            JsonFieldShape::Number => "a number",
                            JsonFieldShape::Integer => "an integer",
                            JsonFieldShape::Boolean => "a boolean",
                            JsonFieldShape::NullableNumber => "a number or null",
                            JsonFieldShape::StringOrArray => "a string or an array of strings",
                            JsonFieldShape::NumberOrArray => "a number or an array of numbers",
                        },
                    });
                }
            }
        }
        Ok(())
    }

    fn named_values(
        root: &serde_json::Value,
        section: &str,
        shape: JsonFieldShape,
    ) -> Result<(), TaskJsonError> {
        let Some(object) = root.get(section).and_then(serde_json::Value::as_object) else {
            return Ok(());
        };
        for (name, value) in object {
            if !shape_matches(value, shape) {
                return Err(TaskJsonError::InvalidShape {
                    path: format!("$.{section}.{name}"),
                    expected: match shape {
                        JsonFieldShape::String => "a string",
                        _ => "a number",
                    },
                });
            }
        }
        Ok(())
    }

    fields(
        root,
        "equations",
        &[
            ("arg", String),
            ("unknowns", StringOrArray),
            ("rhs", StringOrArray),
            ("parameters", StringOrArray),
            ("parameter_values", NumberOrArray),
        ],
    )?;
    named_values(root, "parameters", Number)?;
    named_values(root, "where", String)?;
    named_values(root, "substitute", String)?;
    validate_pair_equation_shapes(root)?;

    match kind {
        TaskDocumentKind::Ivp => {
            fields(
                root,
                "task",
                &[
                    ("solver", String),
                    ("method", String),
                    ("schema_version", Integer),
                ],
            )?;
            fields(
                root,
                "initial_conditions",
                &[("t0", Number), ("t_end", Number), ("y0", NumberOrArray)],
            )?;
            fields(
                root,
                "solver_options",
                &[
                    ("step_size", Number),
                    ("tolerance", Number),
                    ("max_iterations", Integer),
                    ("rtol", Number),
                    ("atol", Number),
                    ("max_step", Number),
                    ("first_step", NullableNumber),
                    ("vectorized", Boolean),
                    ("parallel", Boolean),
                    ("neighborhood_check", Number),
                    ("symbolic_frontend", String),
                    ("symbolic_assembly", String),
                    ("symbolic_execution", String),
                    ("aot_toolchain", String),
                    ("aot_profile", String),
                    ("aot_output_dir", String),
                    ("matrix_layout", String),
                    ("linear_structure", String),
                    ("banded_lower", Integer),
                    ("banded_kl", Integer),
                    ("banded_upper", Integer),
                    ("banded_ku", Integer),
                    ("execution_policy", String),
                    ("parallel_min_work", Integer),
                    ("telemetry", String),
                    ("logging", String),
                    ("log_level", String),
                    ("lsode2_method_family", String),
                    ("lsode2_controller", String),
                    ("lsode2_symbolic_assembly", String),
                    ("lsode2_symbolic_execution", String),
                    ("lsode2_aot_toolchain", String),
                    ("lsode2_aot_profile", String),
                    ("lsode2_aot_output_dir", String),
                    ("lsode2_linear_structure", String),
                    ("lsode2_banded_kl", Integer),
                    ("lsode2_banded_ku", Integer),
                    ("lsode2_linear_solver_policy", String),
                    ("lsode2_native_execution", String),
                    ("lsode2_native_max_step_attempts", Integer),
                    ("lsode2_native_max_accepted_steps", Integer),
                    ("lsode2_stop_variable", String),
                    ("lsode2_stop_comparator", String),
                    ("lsode2_stop_target", Number),
                    ("lsode2_stop_tolerance", Number),
                ],
            )?;
            fields(
                root,
                "continuation",
                &[
                    ("parameters", StringOrArray),
                    ("parameter", String),
                    ("values", NumberOrArray),
                    ("mode", String),
                    ("restart_each", Boolean),
                    ("restart_policy", String),
                    ("monotonic", String),
                    ("y0_values", NumberOrArray),
                    ("t0_values", NumberOrArray),
                    ("t_end_values", NumberOrArray),
                ],
            )?;
            validate_dynamic_continuation_shapes(root)?;
            validate_postprocessing_shapes(root, false)?;
        }
        TaskDocumentKind::Bvp => {
            fields(
                root,
                "task",
                &[
                    ("solver", String),
                    ("strategy", String),
                    ("scheme", String),
                    ("method", String),
                    ("backend", String),
                    ("frontend", String),
                    ("execution", String),
                ],
            )?;
            fields(
                root,
                "mesh",
                &[("t0", Number), ("t_end", Number), ("n_steps", Integer)],
            )?;
            fields(
                root,
                "solver_options",
                &[
                    ("tolerance", Number),
                    ("boundary_tolerance", Number),
                    ("max_iterations", Integer),
                    ("max_nodes", Integer),
                    ("max_newton_iterations", Integer),
                    ("max_jacobian_refreshes", Integer),
                    ("max_mesh_refinements", Integer),
                    ("max_backtracking_steps", Integer),
                    ("banded_lower", Integer),
                    ("banded_upper", Integer),
                    ("sparse_factorization", String),
                    ("execution_policy", String),
                    ("execution_min_work", Integer),
                    ("telemetry", String),
                    ("output_policy", String),
                    ("singular_term_a", Number),
                    ("singular_term_matrix", NumberOrArray),
                    ("linear_sys_method", String),
                    ("loglevel", String),
                    ("generated_backend", String),
                    ("matrix_backend", String),
                    ("backend_policy", String),
                    ("symbolic_backend", String),
                    ("aot_codegen_backend", String),
                    ("aot_c_compiler", String),
                    ("aot_build_policy", String),
                    ("aot_build_profile", String),
                    ("aot_compile_preset", String),
                    ("aot_execution_policy", String),
                    ("aot_output_dir", String),
                    ("aot_handoff_path", String),
                    ("banded_linear_solver", String),
                    ("refinement_steps", Integer),
                ],
            )?;
            fields(
                root,
                "continuation",
                &[
                    ("parameters", StringOrArray),
                    ("parameter", String),
                    ("values", NumberOrArray),
                    ("mode", String),
                    ("restart_each", Boolean),
                    ("restart_policy", String),
                    ("monotonic", String),
                    ("y0_values", NumberOrArray),
                    ("t0_values", NumberOrArray),
                    ("t_end_values", NumberOrArray),
                ],
            )?;
            validate_dynamic_continuation_shapes(root)?;
            validate_postprocessing_shapes(root, true)?;
            validate_dynamic_bvp_shapes(root)?;
        }
        TaskDocumentKind::Nonlinear => {
            fields(
                root,
                "problem",
                &[
                    ("formulation", String),
                    ("variables", StringOrArray),
                    ("equations", StringOrArray),
                    ("initial_guess", NumberOrArray),
                    ("parameters", StringOrArray),
                    ("parameter_values", NumberOrArray),
                ],
            )?;
            fields(
                root,
                "solver",
                &[
                    ("method", String),
                    ("tolerance", Number),
                    ("residual_tolerance", Number),
                    ("gradient_tolerance", Number),
                    ("max_iterations", Integer),
                ],
            )?;
        }
        TaskDocumentKind::Optimization => {
            fields(
                root,
                "data",
                &[("arg", String), ("x", NumberOrArray), ("y", NumberOrArray)],
            )?;
            fields(
                root,
                "model",
                &[
                    ("method", String),
                    ("parameters", StringOrArray),
                    ("initial_guess", NumberOrArray),
                    ("equations", StringOrArray),
                    ("basis_parameters", StringOrArray),
                    ("basis_equations", StringOrArray),
                    ("invariants", StringOrArray),
                ],
            )?;
            fields(
                root,
                "solver",
                &[
                    ("tolerance", Number),
                    ("objective_tolerance", Number),
                    ("gradient_tolerance", Number),
                    ("max_iterations", Integer),
                    ("gavin_weights", NumberOrArray),
                    ("gavin_positive_parameters", StringOrArray),
                ],
            )?;
        }
    }
    Ok(())
}

fn validate_pair_equation_shapes(root: &serde_json::Value) -> Result<(), TaskJsonError> {
    let Some(equations) = root.get("equations").and_then(serde_json::Value::as_object) else {
        return Ok(());
    };
    if equations.contains_key("unknowns") || equations.contains_key("rhs") {
        return Ok(());
    }
    const RESERVED: &[&str] = &[
        "arg",
        "parameters",
        "parameter_values",
        "params",
        "parameter_names",
        "params_values",
        "param_values",
    ];
    for (name, value) in equations {
        if !RESERVED.contains(&name.as_str()) && !value.is_string() {
            return Err(TaskJsonError::InvalidShape {
                path: format!("$.equations.{name}"),
                expected: "a symbolic expression string",
            });
        }
    }
    Ok(())
}

fn validate_dynamic_continuation_shapes(root: &serde_json::Value) -> Result<(), TaskJsonError> {
    let Some(fields) = root
        .get("continuation")
        .and_then(serde_json::Value::as_object)
    else {
        return Ok(());
    };
    const FIXED: &[&str] = &[
        "parameters",
        "parameter",
        "values",
        "mode",
        "restart_each",
        "restart_policy",
        "monotonic",
        "y0_values",
        "t0_values",
        "t_end_values",
    ];
    for (name, value) in fields {
        if !FIXED.contains(&name.as_str())
            && name.ends_with("_values")
            && !(value.is_number()
                || value
                    .as_array()
                    .is_some_and(|items| items.iter().all(serde_json::Value::is_number)))
        {
            return Err(TaskJsonError::InvalidShape {
                path: format!("$.continuation.{name}"),
                expected: "a number or an array of numbers",
            });
        }
    }
    Ok(())
}

fn normalize_integral_json_fields(kind: TaskDocumentKind, root: &mut serde_json::Value) {
    fn normalize(section: &mut serde_json::Value, fields: &[&str]) {
        let Some(object) = section.as_object_mut() else {
            return;
        };
        for name in fields {
            let Some(value) = object.get_mut(*name) else {
                continue;
            };
            let Some(number) = value.as_number() else {
                continue;
            };
            if number.as_i64().is_some() || number.as_u64().is_some() {
                continue;
            }
            let Some(value) = number
                .as_f64()
                .filter(|value| value.is_finite() && value.fract() == 0.0)
            else {
                continue;
            };
            let integer = if value >= 0.0 && value < 18_446_744_073_709_551_616.0 {
                Some(serde_json::Number::from(value as u64))
            } else if value >= -9_223_372_036_854_775_808.0 && value < 9_223_372_036_854_775_808.0 {
                Some(serde_json::Number::from(value as i64))
            } else {
                None
            };
            if let Some(integer) = integer {
                *object.get_mut(*name).expect("field was found") =
                    serde_json::Value::Number(integer);
            }
        }
    }

    match kind {
        TaskDocumentKind::Ivp => {
            if let Some(section) = root.get_mut("task") {
                normalize(section, &["schema_version"]);
            }
            if let Some(section) = root.get_mut("solver_options") {
                normalize(
                    section,
                    &[
                        "max_iterations",
                        "banded_lower",
                        "banded_kl",
                        "banded_upper",
                        "banded_ku",
                        "parallel_min_work",
                        "lsode2_banded_kl",
                        "lsode2_banded_ku",
                        "lsode2_native_max_step_attempts",
                        "lsode2_native_max_accepted_steps",
                    ],
                );
            }
        }
        TaskDocumentKind::Bvp => {
            if let Some(section) = root.get_mut("mesh") {
                normalize(section, &["n_steps"]);
            }
            if let Some(section) = root.get_mut("solver_options") {
                normalize(
                    section,
                    &[
                        "max_iterations",
                        "max_nodes",
                        "max_newton_iterations",
                        "max_jacobian_refreshes",
                        "max_mesh_refinements",
                        "max_backtracking_steps",
                        "banded_lower",
                        "banded_upper",
                        "execution_min_work",
                        "refinement_steps",
                    ],
                );
            }
        }
        TaskDocumentKind::Nonlinear | TaskDocumentKind::Optimization => {
            if let Some(section) = root.get_mut("solver") {
                normalize(section, &["max_iterations"]);
            }
        }
    }
}

fn normalize_json_numeric_fields(kind: TaskDocumentKind, root: &mut serde_json::Value) {
    fn scalar_as_float(value: &mut serde_json::Value) {
        let Some(number) = value.as_number() else {
            return;
        };
        if number.as_i64().is_none() && number.as_u64().is_none() {
            return;
        }
        if let Some(number) = number
            .as_f64()
            .filter(|number| number.is_finite())
            .and_then(serde_json::Number::from_f64)
        {
            *value = serde_json::Value::Number(number);
        }
    }

    fn selected_fields(section: &mut serde_json::Value, names: &[&str]) {
        let Some(object) = section.as_object_mut() else {
            return;
        };
        for name in names {
            if let Some(value) = object.get_mut(*name) {
                scalar_as_float(value);
            }
        }
    }

    fn all_values(section: &mut serde_json::Value) {
        if let Some(object) = section.as_object_mut() {
            for value in object.values_mut() {
                scalar_as_float(value);
            }
        }
    }

    match kind {
        TaskDocumentKind::Ivp => {
            if let Some(section) = root.get_mut("equations") {
                selected_fields(section, &["parameter_values"]);
            }
            if let Some(section) = root.get_mut("parameters") {
                all_values(section);
            }
            if let Some(section) = root.get_mut("initial_conditions") {
                selected_fields(section, &["t0", "t_end", "y0"]);
            }
            if let Some(section) = root.get_mut("solver_options") {
                selected_fields(
                    section,
                    &[
                        "step_size",
                        "tolerance",
                        "rtol",
                        "atol",
                        "max_step",
                        "neighborhood_check",
                        "lsode2_stop_target",
                        "lsode2_stop_tolerance",
                    ],
                );
            }
            if let Some(section) = root.get_mut("continuation") {
                selected_fields(
                    section,
                    &["values", "y0_values", "t0_values", "t_end_values"],
                );
            }
        }
        TaskDocumentKind::Bvp => {
            if let Some(section) = root.get_mut("equations") {
                selected_fields(section, &["parameter_values"]);
            }
            if let Some(section) = root.get_mut("parameters") {
                all_values(section);
            }
            if let Some(section) = root.get_mut("boundary_conditions") {
                all_values(section);
            }
            if let Some(section) = root.get_mut("initial_guess") {
                selected_fields(section, &["guess"]);
                for (name, value) in section
                    .as_object_mut()
                    .into_iter()
                    .flat_map(|map| map.iter_mut())
                {
                    if name != "guess" {
                        scalar_as_float(value);
                    }
                }
            }
            if let Some(section) = root.get_mut("mesh") {
                selected_fields(section, &["t0", "t_end"]);
            }
            if let Some(section) = root.get_mut("solver_options") {
                selected_fields(
                    section,
                    &["tolerance", "boundary_tolerance", "singular_term_a"],
                );
            }
            if let Some(section) = root.get_mut("continuation") {
                selected_fields(
                    section,
                    &["values", "y0_values", "t0_values", "t_end_values"],
                );
            }
        }
        TaskDocumentKind::Nonlinear => {
            if let Some(section) = root.get_mut("problem") {
                selected_fields(section, &["initial_guess", "parameter_values"]);
            }
            if let Some(section) = root.get_mut("solver") {
                selected_fields(
                    section,
                    &["tolerance", "residual_tolerance", "gradient_tolerance"],
                );
            }
        }
        TaskDocumentKind::Optimization => {
            if let Some(section) = root.get_mut("data") {
                selected_fields(section, &["x", "y"]);
            }
            if let Some(section) = root.get_mut("model") {
                selected_fields(section, &["initial_guess"]);
            }
            if let Some(section) = root.get_mut("solver") {
                selected_fields(
                    section,
                    &[
                        "tolerance",
                        "objective_tolerance",
                        "gradient_tolerance",
                        "gavin_weights",
                    ],
                );
            }
        }
    }
}

fn validate_postprocessing_shapes(
    root: &serde_json::Value,
    bvp: bool,
) -> Result<(), TaskJsonError> {
    use JsonFieldShape::*;
    let mut fields = vec![
        ("save_csv", Boolean),
        ("csv_path", String),
        ("save_txt", Boolean),
        ("txt_path", String),
        ("write_report", Boolean),
        ("report_path", String),
        ("plotters_png", Boolean),
        ("plotters_dir", String),
        ("gnuplot_png", Boolean),
        ("gnuplot_dir", String),
        ("terminal_plot", Boolean),
        ("plot", Boolean),
    ];
    if bvp {
        fields.push(("output_policy", String));
    }
    // `fields` is checked against the raw JSON so scalar-vs-array distinctions
    // remain visible before conversion into the legacy DocumentMap.
    let Some(postprocessing) = root
        .get("postprocessing")
        .and_then(serde_json::Value::as_object)
    else {
        return Ok(());
    };
    for (field, shape) in fields {
        if let Some(value) = postprocessing.get(field) {
            let valid = match shape {
                Boolean => value.is_boolean(),
                String => value.is_string(),
                _ => true,
            };
            if !valid {
                return Err(TaskJsonError::InvalidShape {
                    path: format!("$.postprocessing.{field}"),
                    expected: if matches!(shape, Boolean) {
                        "a boolean"
                    } else {
                        "a string"
                    },
                });
            }
        }
    }
    Ok(())
}

fn validate_dynamic_bvp_shapes(root: &serde_json::Value) -> Result<(), TaskJsonError> {
    for section in ["boundary_conditions", "initial_guess", "bounds"] {
        let Some(fields) = root.get(section).and_then(serde_json::Value::as_object) else {
            continue;
        };
        for (name, value) in fields {
            let valid = match section {
                "initial_guess" if name == "guess" => {
                    value.is_number()
                        || value
                            .as_array()
                            .is_some_and(|items| items.iter().all(serde_json::Value::is_number))
                }
                "initial_guess" | "boundary_conditions" => value.is_number(),
                "bounds" => value.as_array().is_some_and(|items| {
                    items.len() == 2 && items.iter().all(serde_json::Value::is_number)
                }),
                _ => true,
            };
            if !valid {
                return Err(TaskJsonError::InvalidShape {
                    path: format!("$.{section}.{name}"),
                    expected: match section {
                        "bounds" => "an array of exactly two numbers",
                        "initial_guess" if name == "guess" => "a number or an array of numbers",
                        _ => "a number",
                    },
                });
            }
        }
    }
    Ok(())
}

fn validate_json_document_fields(
    kind: TaskDocumentKind,
    document: &DocumentMap,
) -> Result<(), TaskJsonError> {
    match kind {
        // These native parsers already reject unknown sections and fields.
        TaskDocumentKind::Nonlinear | TaskDocumentKind::Optimization => return Ok(()),
        TaskDocumentKind::Ivp => validate_ivp_json_fields(document),
        TaskDocumentKind::Bvp => validate_bvp_json_fields(document),
    }
}

fn validate_ivp_json_fields(document: &DocumentMap) -> Result<(), TaskJsonError> {
    const SECTIONS: &[&str] = &[
        "task",
        "equations",
        "parameters",
        "where",
        "substitute",
        "initial_conditions",
        "solver_options",
        "postprocessing",
        "continuation",
    ];
    reject_unknown_json_sections(document, SECTIONS)?;
    reject_unknown_json_fields(document, "task", &["solver", "method", "schema_version"])?;
    validate_ivp_json_schema_version(document)?;
    reject_unknown_json_fields(document, "initial_conditions", &["t0", "t_end", "y0"])?;
    reject_unknown_json_fields(document, "solver_options", IVP_SOLVER_FIELDS)?;
    reject_unknown_json_fields(document, "postprocessing", IVP_POSTPROCESSING_FIELDS)?;
    reject_unknown_continuation_fields(document)?;
    validate_equation_declaration_fields(document)?;
    Ok(())
}

fn validate_ivp_json_schema_version(document: &DocumentMap) -> Result<(), TaskJsonError> {
    let Some(values) = document
        .get("task")
        .and_then(|task| task.get("schema_version"))
    else {
        return Ok(());
    };
    if values
        .as_ref()
        .is_some_and(|values| matches!(values.as_slice(), [Value::Integer(1) | Value::Usize(1)]))
    {
        return Ok(());
    }
    Err(TaskJsonError::InvalidShape {
        path: "$.task.schema_version".to_string(),
        expected: "the integer 1",
    })
}

fn validate_bvp_json_fields(document: &DocumentMap) -> Result<(), TaskJsonError> {
    const SECTIONS: &[&str] = &[
        "task",
        "equations",
        "parameters",
        "where",
        "substitute",
        "boundary_conditions",
        "mesh",
        "initial_guess",
        "bounds",
        "solver_options",
        "postprocessing",
        "continuation",
    ];
    reject_unknown_json_sections(document, SECTIONS)?;
    reject_unknown_json_fields(
        document,
        "task",
        &[
            "solver",
            "strategy",
            "scheme",
            "method",
            "backend",
            "frontend",
            "execution",
        ],
    )?;
    reject_unknown_json_fields(document, "mesh", &["t0", "t_end", "n_steps"])?;
    reject_unknown_json_fields(document, "solver_options", BVP_SOLVER_FIELDS)?;
    reject_unknown_json_fields(document, "postprocessing", POSTPROCESSING_FIELDS)?;
    reject_unknown_continuation_fields(document)?;
    validate_equation_declaration_fields(document)?;
    validate_bvp_dynamic_fields(document)?;
    Ok(())
}

const POSTPROCESSING_FIELDS: &[&str] = &[
    "save_csv",
    "csv_path",
    "save_txt",
    "txt_path",
    "write_report",
    "report_path",
    "plotters_png",
    "plotters_dir",
    "gnuplot_png",
    "gnuplot_dir",
    "terminal_plot",
    "output_policy",
    "plot",
];

const IVP_POSTPROCESSING_FIELDS: &[&str] = &[
    "save_csv",
    "csv_path",
    "save_txt",
    "txt_path",
    "write_report",
    "report_path",
    "plotters_png",
    "plotters_dir",
    "gnuplot_png",
    "gnuplot_dir",
    "terminal_plot",
    "plot",
];

const BVP_SOLVER_FIELDS: &[&str] = &[
    "tolerance",
    "boundary_tolerance",
    "max_iterations",
    "max_nodes",
    "max_newton_iterations",
    "max_jacobian_refreshes",
    "max_mesh_refinements",
    "max_backtracking_steps",
    "banded_lower",
    "banded_upper",
    "sparse_factorization",
    "execution_policy",
    "execution_min_work",
    "telemetry",
    "output_policy",
    "singular_term_a",
    "singular_term_matrix",
    "linear_sys_method",
    "loglevel",
    "generated_backend",
    "matrix_backend",
    "backend_policy",
    "symbolic_backend",
    "aot_codegen_backend",
    "aot_c_compiler",
    "aot_build_policy",
    "aot_build_profile",
    "aot_compile_preset",
    "aot_execution_policy",
    "aot_output_dir",
    "aot_handoff_path",
    "banded_linear_solver",
    "refinement_steps",
];

const IVP_SOLVER_FIELDS: &[&str] = &[
    "step_size",
    "tolerance",
    "max_iterations",
    "rtol",
    "atol",
    "max_step",
    "first_step",
    "vectorized",
    "parallel",
    "neighborhood_check",
    "symbolic_frontend",
    "symbolic_assembly",
    "symbolic_execution",
    "aot_toolchain",
    "aot_profile",
    "aot_output_dir",
    "matrix_layout",
    "linear_structure",
    "banded_lower",
    "banded_kl",
    "banded_upper",
    "banded_ku",
    "execution_policy",
    "parallel_min_work",
    "telemetry",
    "logging",
    "log_level",
    "lsode2_method_family",
    "lsode2_controller",
    "lsode2_symbolic_assembly",
    "lsode2_symbolic_execution",
    "lsode2_aot_toolchain",
    "lsode2_aot_profile",
    "lsode2_aot_output_dir",
    "lsode2_linear_structure",
    "lsode2_banded_kl",
    "lsode2_banded_ku",
    "lsode2_linear_solver_policy",
    "lsode2_native_execution",
    "lsode2_native_max_step_attempts",
    "lsode2_native_max_accepted_steps",
    "lsode2_stop_variable",
    "lsode2_stop_comparator",
    "lsode2_stop_target",
    "lsode2_stop_tolerance",
];

fn reject_unknown_json_sections(
    document: &DocumentMap,
    allowed: &[&str],
) -> Result<(), TaskJsonError> {
    let mut sections = document
        .keys()
        .filter(|section| !allowed.contains(&section.as_str()))
        .collect::<Vec<_>>();
    sections.sort_unstable();
    if let Some(section) = sections.first() {
        return Err(TaskJsonError::UnknownField {
            path: format!("$.{section}"),
        });
    }
    Ok(())
}

fn reject_unknown_json_fields(
    document: &DocumentMap,
    section: &str,
    allowed: &[&str],
) -> Result<(), TaskJsonError> {
    let Some(fields) = document.get(section) else {
        return Ok(());
    };
    let mut unknown = fields
        .keys()
        .filter(|field| !allowed.contains(&field.as_str()))
        .collect::<Vec<_>>();
    unknown.sort_unstable();
    if let Some(field) = unknown.first() {
        return Err(TaskJsonError::UnknownField {
            path: format!("$.{section}.{field}"),
        });
    }
    Ok(())
}

fn reject_unknown_continuation_fields(document: &DocumentMap) -> Result<(), TaskJsonError> {
    let Some(fields) = document.get("continuation") else {
        return Ok(());
    };
    const FIXED: &[&str] = &[
        "parameters",
        "parameter",
        "values",
        "mode",
        "restart_each",
        "restart_policy",
        "monotonic",
        "y0_values",
        "t0_values",
        "t_end_values",
    ];
    if let Some(field) = fields
        .keys()
        .find(|field| !FIXED.contains(&field.as_str()) && !field.ends_with("_values"))
    {
        return Err(TaskJsonError::UnknownField {
            path: format!("$.continuation.{field}"),
        });
    }
    Ok(())
}

fn validate_equation_declaration_fields(document: &DocumentMap) -> Result<(), TaskJsonError> {
    let Some(equations) = document.get("equations") else {
        return Ok(());
    };
    if !(equations.contains_key("unknowns") || equations.contains_key("rhs")) {
        // Pair-style equations use each remaining field name as an unknown.
        const DSL_ONLY_ALIASES: &[&str] =
            &["params", "parameter_names", "params_values", "param_values"];
        if let Some(field) = equations
            .keys()
            .find(|field| DSL_ONLY_ALIASES.contains(&field.as_str()))
        {
            return Err(TaskJsonError::UnknownField {
                path: format!("$.equations.{field}"),
            });
        }
        return Ok(());
    }
    const DECLARATION_FIELDS: &[&str] =
        &["arg", "unknowns", "rhs", "parameters", "parameter_values"];
    if let Some(field) = equations
        .keys()
        .find(|field| !DECLARATION_FIELDS.contains(&field.as_str()))
    {
        return Err(TaskJsonError::UnknownField {
            path: format!("$.equations.{field}"),
        });
    }
    Ok(())
}

fn validate_bvp_dynamic_fields(document: &DocumentMap) -> Result<(), TaskJsonError> {
    let Some(equations) = document.get("equations") else {
        return Ok(());
    };
    let unknowns = if let Some(declared) = equations.get("unknowns").and_then(Option::as_ref) {
        declared
            .iter()
            .filter_map(Value::as_string)
            .cloned()
            .collect::<Vec<_>>()
    } else {
        const RESERVED: &[&str] = &["arg", "parameters", "parameter_values", "unknowns", "rhs"];
        equations
            .keys()
            .filter(|field| !RESERVED.contains(&field.as_str()))
            .cloned()
            .collect::<Vec<_>>()
    };
    if unknowns.is_empty() {
        // A malformed declaration is reported by the family parser.
        return Ok(());
    }
    let valid_names = unknowns
        .iter()
        .map(String::as_str)
        .collect::<std::collections::HashSet<_>>();
    for section_name in ["boundary_conditions", "initial_guess", "bounds"] {
        let Some(fields) = document.get(section_name) else {
            continue;
        };
        for field in fields.keys() {
            let valid = if section_name == "initial_guess" && fields.contains_key("guess") {
                field == "guess"
            } else if section_name == "boundary_conditions" {
                ["_left", "_right"].iter().any(|suffix| {
                    field
                        .strip_suffix(suffix)
                        .is_some_and(|name| valid_names.contains(name))
                })
            } else {
                valid_names.contains(field.as_str())
            };
            if !valid {
                return Err(TaskJsonError::UnknownField {
                    path: format!("$.{section_name}.{field}"),
                });
            }
        }
    }
    Ok(())
}

pub fn detect_task_kind_from_document(
    document: &DocumentMap,
) -> Result<TaskDocumentKind, TaskRunnerError> {
    if document.contains_key("data") && document.contains_key("model") {
        return Ok(TaskDocumentKind::Optimization);
    }
    if document.contains_key("problem") && document.contains_key("solver") {
        return Ok(TaskDocumentKind::Nonlinear);
    }
    let solver = document
        .get("task")
        .and_then(|section| section.get("solver"))
        .and_then(Option::as_ref)
        .and_then(|values| values.first())
        .and_then(Value::as_string)
        .map(|value| value.trim().to_ascii_lowercase());
    match solver.as_deref() {
        Some("ivp") => Ok(TaskDocumentKind::Ivp),
        Some(
            "bvp"
            | "bvp_damp"
            | "bvp_damped"
            | "bvp_sci"
            | "bvpsci"
            | "bvp_scipy"
            | "scipy_bvp"
            | "bvp_correctionnewton"
            | "bvp_correction_newton"
            | "correction_newton",
        ) => Ok(TaskDocumentKind::Bvp),
        Some(other) => Err(TaskRunnerError::UnsupportedSolver {
            value: other.to_string(),
        }),
        None => Err(TaskRunnerError::MissingSolverField),
    }
}

pub fn detect_task_kind_from_json(input: &str) -> Result<TaskDocumentKind, TaskRunnerError> {
    let document = document_map_from_json(input).map_err(TaskRunnerError::Json)?;
    detect_task_kind_from_document(&document)
}

pub fn run_task_from_str(input: &str) -> Result<TaskRunResult, TaskRunnerError> {
    let spec = parse_task_spec_from_str(input)?;
    run_task_from_spec(spec)
}

/// Parse and execute a JSON task document in the Rust core.
pub fn run_task_from_json(input: &str) -> Result<TaskRunResult, TaskRunnerError> {
    let spec = parse_task_spec_from_json(input)?;
    run_task_from_spec(spec)
}

/// Parse and execute a JSON task with cooperative cancellation on supported
/// task routes. Single-solve IVP and supported Fresh/explicit-restart IVP
/// continuation routes forward cancellation to native safe points. Stateful
/// IVP `Continue` remains explicitly unsupported.
pub fn run_task_from_json_with_cancellation_token(
    input: &str,
    token: &CancellationToken,
) -> Result<TaskRunResult, TaskRunnerError> {
    let spec = parse_task_spec_from_json(input)?;
    run_task_from_spec_with_cancellation_token(spec, token)
}

pub fn run_task_from_spec_with_cancellation_token(
    spec: ParsedTaskSpec,
    token: &CancellationToken,
) -> Result<TaskRunResult, TaskRunnerError> {
    match spec {
        ParsedTaskSpec::Bvp(spec) => {
            crate::command_interpreter::task_parser_bvp::run_bvp_task_with_cancellation_token(
                spec, token,
            )
            .map(TaskRunResult::Bvp)
            .map_err(TaskRunnerError::Bvp)
        }
        ParsedTaskSpec::Ivp(spec) => run_ivp_task_with_cancellation_token(spec, token)
            .map(TaskRunResult::Ivp)
            .map_err(TaskRunnerError::Ivp),
        ParsedTaskSpec::Nonlinear(spec) => {
            let requested_mode = spec.solver_options.telemetry.unwrap_or_default();
            run_nonlinear_task_with_cancellation_token(spec, token)
                .map(TaskRunResult::Nonlinear)
                .map_err(|source| {
                    if requested_mode
                        != crate::command_interpreter::task_parser::TaskTelemetryMode::Off
                        && !nonlinear_error_is_cancellation(&source)
                    {
                        TaskRunnerError::NonlinearRunTelemetryUnavailable {
                            source,
                            requested_mode,
                        }
                    } else {
                        TaskRunnerError::NonlinearRun(source)
                    }
                })
        }
        ParsedTaskSpec::Optimization(spec)
            if matches!(
                spec.method,
                OptimizationMethodSpec::LevenbergMarquardt
                    | OptimizationMethodSpec::VarPro
                    | OptimizationMethodSpec::Gavin
            ) =>
        {
            let requested_mode = spec.solver_options.telemetry.unwrap_or_default();
            run_optimization_task_with_cancellation_token(spec, token)
                .map(TaskRunResult::Optimization)
                .map_err(|source| {
                    if requested_mode
                        != crate::command_interpreter::task_parser::TaskTelemetryMode::Off
                        && !optimization_error_is_cancellation(&source)
                    {
                        TaskRunnerError::OptimizationRunTelemetryUnavailable {
                            source,
                            requested_mode,
                        }
                    } else {
                        TaskRunnerError::OptimizationRun(source)
                    }
                })
        }
        ParsedTaskSpec::Optimization(spec) => Err(TaskRunnerError::CancellationUnsupported {
            family: match spec.method {
                OptimizationMethodSpec::LevenbergMarquardt => "optimization LM",
                OptimizationMethodSpec::Gavin => "optimization Gavin",
                OptimizationMethodSpec::VarPro => "optimization VarPro",
            },
        }),
    }
}

pub fn run_task_from_spec(spec: ParsedTaskSpec) -> Result<TaskRunResult, TaskRunnerError> {
    match spec {
        ParsedTaskSpec::Ivp(spec) => run_ivp_task(spec)
            .map(TaskRunResult::Ivp)
            .map_err(TaskRunnerError::Ivp),
        ParsedTaskSpec::Bvp(spec) => run_bvp_task(spec)
            .map(TaskRunResult::Bvp)
            .map_err(TaskRunnerError::Bvp),
        ParsedTaskSpec::Nonlinear(spec) => {
            let requested_mode = spec.solver_options.telemetry.unwrap_or_default();
            run_nonlinear_task(spec)
                .map(TaskRunResult::Nonlinear)
                .map_err(|source| {
                    if requested_mode
                        != crate::command_interpreter::task_parser::TaskTelemetryMode::Off
                        && !nonlinear_error_is_cancellation(&source)
                    {
                        TaskRunnerError::NonlinearRunTelemetryUnavailable {
                            source,
                            requested_mode,
                        }
                    } else {
                        TaskRunnerError::NonlinearRun(source)
                    }
                })
        }
        ParsedTaskSpec::Optimization(spec) => {
            let requested_mode = spec.solver_options.telemetry.unwrap_or_default();
            run_optimization_task(spec)
                .map(TaskRunResult::Optimization)
                .map_err(|source| {
                    if requested_mode
                        != crate::command_interpreter::task_parser::TaskTelemetryMode::Off
                        && !optimization_error_is_cancellation(&source)
                    {
                        TaskRunnerError::OptimizationRunTelemetryUnavailable {
                            source,
                            requested_mode,
                        }
                    } else {
                        TaskRunnerError::OptimizationRun(source)
                    }
                })
        }
    }
}

fn nonlinear_error_is_cancellation(error: &NonlinearTaskRunError) -> bool {
    matches!(
        error,
        NonlinearTaskRunError::Root(
            crate::numerical::Nonlinear_systems::error::SolveError::Cancelled { .. }
        ) | NonlinearTaskRunError::LeastSquares(
            crate::numerical::Nonlinear_systems::least_squares::LeastSquaresError::Cancelled { .. }
        )
    )
}

fn optimization_error_is_cancellation(error: &OptimizationTaskRunError) -> bool {
    use crate::numerical::Nonlinear_systems::least_squares::LeastSquaresError;
    use crate::numerical::optimization::lm_gavin::GavinLmError;
    use crate::numerical::optimization::sym_fitting::FittingError;
    use crate::numerical::optimization::universal_fitting::UniversalFittingError;
    use crate::numerical::optimization::varpro::symbolic::SymbolicVarProError;

    match &error.0 {
        UniversalFittingError::Fitting(FittingError::Solver(LeastSquaresError::Cancelled {
            ..
        }))
        | UniversalFittingError::Fitting(FittingError::Preparation(
            LeastSquaresError::Cancelled { .. },
        ))
        | UniversalFittingError::SolverConfiguration(LeastSquaresError::Cancelled { .. })
        | UniversalFittingError::Gavin(GavinLmError::Cancelled { .. }) => true,
        UniversalFittingError::VarPro(SymbolicVarProError::SolverFailed {
            source: Some(LeastSquaresError::Cancelled { .. }),
            ..
        }) => true,
        _ => false,
    }
}

pub fn parse_task_spec_from_file(
    path: impl AsRef<Path>,
) -> Result<ParsedTaskSpec, TaskRunnerError> {
    let path_ref = path.as_ref();
    let text = fs::read_to_string(path_ref).map_err(|source| TaskRunnerError::Io {
        path: path_ref.to_path_buf(),
        source,
    })?;
    if is_json_path(path_ref) {
        parse_task_spec_from_json(&text)
    } else {
        parse_task_spec_from_str(&text)
    }
}

pub fn run_task_from_file(path: impl AsRef<Path>) -> Result<TaskRunResult, TaskRunnerError> {
    let path_ref = path.as_ref();
    let text = fs::read_to_string(path_ref).map_err(|source| TaskRunnerError::Io {
        path: path_ref.to_path_buf(),
        source,
    })?;
    if is_json_path(path_ref) {
        run_task_from_json(&text)
    } else {
        run_task_from_str(&text)
    }
}

fn is_json_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("json"))
}

pub fn render_task_preview(spec: &ParsedTaskSpec) -> String {
    match spec {
        ParsedTaskSpec::Ivp(ivp) => render_ivp_preview(ivp),
        ParsedTaskSpec::Bvp(bvp) => render_bvp_preview(bvp),
        ParsedTaskSpec::Nonlinear(spec) => render_nonlinear_preview(spec),
        ParsedTaskSpec::Optimization(spec) => render_optimization_preview(spec),
    }
}

pub fn render_task_check(spec: &ParsedTaskSpec) -> String {
    match spec {
        ParsedTaskSpec::Ivp(ivp) => render_ivp_check(ivp),
        ParsedTaskSpec::Bvp(bvp) => render_bvp_check(bvp),
        ParsedTaskSpec::Nonlinear(spec) => render_nonlinear_check(spec),
        ParsedTaskSpec::Optimization(spec) => render_optimization_check(spec),
    }
}

fn render_nonlinear_preview(spec: &NonlinearTaskSpec) -> String {
    let mut out = String::from("[Task preview] Nonlinear\n");
    out.push_str(&format!(
        "formulation={:?}, method={:?}, variables={}, residuals={}\n",
        spec.problem.formulation,
        spec.method,
        spec.problem.variables.join(","),
        spec.problem.equations.len(),
    ));
    for (index, equation) in spec.problem.equations.iter().enumerate() {
        out.push_str(&format!("{index}: {equation}\n"));
    }
    out
}

fn render_nonlinear_check(spec: &NonlinearTaskSpec) -> String {
    format!(
        "[Task check] Nonlinear\nformulation={:?}\nmethod={:?}\nvariables={}\ninitial_guess_len={}\nparameters={}\nmax_iterations={:?}\ntolerance={:?}\npreflight=valid; solve has not been run\n",
        spec.problem.formulation,
        spec.method,
        spec.problem.variables.join(","),
        spec.problem.initial_guess.len(),
        spec.problem.parameter_names.join(","),
        spec.solver_options.max_iterations,
        spec.solver_options.tolerance,
    )
}

fn render_optimization_preview(spec: &OptimizationTaskSpec) -> String {
    let mut out = format!(
        "[Task preview] Optimization\nmethod={:?}, arg={}, observations={}\n",
        spec.method,
        spec.data.arg,
        spec.data.x.len(),
    );
    match &spec.model {
        crate::command_interpreter::task_parser_optimization::OptimizationModelSpec::EquationModel {
            equations,
            parameter_names,
            ..
        } => {
            out.push_str(&format!("parameters={}\n", parameter_names.join(",")));
            for (index, equation) in equations.iter().enumerate() {
                out.push_str(&format!("model[{index}]={equation}\n"));
            }
        }
        crate::command_interpreter::task_parser_optimization::OptimizationModelSpec::VarProModel {
            parameter_names,
            basis,
            invariants,
            ..
        } => {
            out.push_str(&format!("parameters={}\n", parameter_names.join(",")));
            for (name, expression) in basis {
                out.push_str(&format!("basis[{name}]={expression}\n"));
            }
            out.push_str(&format!("invariants={}\n", invariants.len()));
        }
    }
    out
}

fn render_optimization_check(spec: &OptimizationTaskSpec) -> String {
    let (parameter_count, model_count) = match &spec.model {
        crate::command_interpreter::task_parser_optimization::OptimizationModelSpec::EquationModel {
            equations,
            parameter_names,
            ..
        } => (parameter_names.len(), equations.len()),
        crate::command_interpreter::task_parser_optimization::OptimizationModelSpec::VarProModel {
            parameter_names,
            basis,
            ..
        } => (parameter_names.len(), basis.len()),
    };
    format!(
        "[Task check] Optimization\nmethod={:?}\narg={}\nobservations={}\nparameters={}\nmodel_terms={}\nmax_iterations={:?}\ntolerance={:?}\npreflight=valid; fit has not been run\n",
        spec.method,
        spec.data.arg,
        spec.data.x.len(),
        parameter_count,
        model_count,
        spec.solver_options.max_iterations,
        spec.solver_options.tolerance,
    )
}

fn render_ivp_preview(spec: &IvpTaskSpec) -> String {
    let method = ivp_method_label(&spec.solver.method);
    let mut out = String::new();
    out.push_str("[Task preview] IVP\n");
    out.push_str(&format!(
        "solver=IVP, method={}, arg={}, t0={}, t_end={}\n",
        method, spec.equations.arg, spec.initial_conditions.t0, spec.initial_conditions.t_end
    ));
    out.push_str("equations:\n");
    out.push_str("idx | unknown | rhs\n");
    out.push_str("-------------------\n");
    for (idx, (name, rhs)) in spec
        .equations
        .unknowns
        .iter()
        .zip(spec.equations.rhs.iter())
        .enumerate()
    {
        out.push_str(&format!("{idx:>3} | {name} | {rhs}\n"));
    }
    out
}

fn render_bvp_preview(spec: &BvpTaskSpec) -> String {
    let mut out = String::new();
    out.push_str("[Task preview] BVP\n");
    let family = match spec.solver.family {
        crate::command_interpreter::task_parser_bvp::BvpSolverFamilySpec::Damp => "BVP_Damp",
        crate::command_interpreter::task_parser_bvp::BvpSolverFamilySpec::Sci => "BVP_sci",
        crate::command_interpreter::task_parser_bvp::BvpSolverFamilySpec::CorrectionNewton => {
            "BVP_CorrectionNewton"
        }
    };
    out.push_str(&format!(
        "solver={family}, layout={:?}, arg={}, t0={}, t_end={}, n_steps={}\n",
        spec.solver.backend, spec.equations.arg, spec.mesh.t0, spec.mesh.t_end, spec.mesh.n_steps
    ));
    if spec.solver.family == crate::command_interpreter::task_parser_bvp::BvpSolverFamilySpec::Damp
    {
        out.push_str(&format!(
            "strategy={:?}, scheme={}\n",
            spec.solver.strategy, spec.solver.scheme
        ));
    } else if spec.solver.family
        == crate::command_interpreter::task_parser_bvp::BvpSolverFamilySpec::CorrectionNewton
    {
        out.push_str(&render_correction_newton_settings(spec));
    } else {
        out.push_str(&format!(
            "frontend={:?}, execution={:?}, callback_policy={:?}, tolerance={}, max_nodes={}\n",
            spec.solver.frontend.unwrap_or(
                crate::command_interpreter::task_parser_bvp::BvpFrontendSpec::ExprLegacy
            ),
            spec.solver
                .execution
                .unwrap_or(crate::command_interpreter::task_parser_bvp::BvpExecutionSpec::Lambdify),
            spec.solver_options
                .execution_policy
                .unwrap_or(crate::numerical::BVP_sci::new::BvpSciExecutionPolicy::Sequential),
            spec.solver_options.tolerance.unwrap_or(1e-3),
            spec.solver_options.max_nodes.unwrap_or_else(|| spec
                .mesh
                .n_steps
                .saturating_mul(16)
                .max(128)),
        ));
    }
    out.push_str("equations:\n");
    out.push_str("idx | unknown | rhs\n");
    out.push_str("-------------------\n");
    for (idx, (name, rhs)) in spec
        .equations
        .unknowns
        .iter()
        .zip(spec.equations.rhs.iter())
        .enumerate()
    {
        out.push_str(&format!("{idx:>3} | {name} | {rhs}\n"));
    }
    out
}

fn render_ivp_check(spec: &IvpTaskSpec) -> String {
    let mut out = String::new();
    out.push_str("[Task check] IVP\n");
    out.push_str(&format!(
        "solver=IVP, method={}, arg={}, t0={}, t_end={}\n",
        ivp_method_label(&spec.solver.method),
        spec.equations.arg,
        spec.initial_conditions.t0,
        spec.initial_conditions.t_end
    ));
    let options = &spec.solver_options;
    out.push_str(&format!(
        "options: tolerance={:?}, rtol={:?}, atol={:?}, step_size={:?}, max_step={:?}, first_step={:?}, max_iterations={:?}, vectorized={:?}, parallel={:?}, neighborhood_check={:?}\n",
        options.tolerance,
        options.rtol,
        options.atol,
        options.step_size,
        options.max_step,
        options.first_step,
        options.max_iterations,
        options.vectorized,
        options.parallel,
        options.neighborhood_check,
    ));
    if options.symbolic_assembly.is_some()
        || options.symbolic_execution.is_some()
        || options.matrix_layout.is_some()
        || options.execution_policy.is_some()
        || options.telemetry.is_some()
    {
        out.push_str(&format!(
            "native_options: symbolic_frontend={}, symbolic_execution={}, matrix_layout={}, execution_policy={}, telemetry={}\n",
            options
                .symbolic_assembly
                .map(|value| value.label())
                .unwrap_or("default"),
            options
                .symbolic_execution
                .as_ref()
                .map(Lsode2TaskExecutionSpec::label)
                .unwrap_or_else(|| "default".to_string()),
            options
                .matrix_layout
                .map(|value| value.label())
                .unwrap_or("default"),
            options
                .execution_policy
                .map(|value| value.label())
                .unwrap_or("default"),
            ivp_telemetry_label(options.telemetry),
        ));
    }
    if let Some(lsode2) = &options.lsode2 {
        let controller = lsode2
            .controller
            .map(|value| value.mode.label())
            .unwrap_or("bdf_only");
        let symbolic_assembly = lsode2
            .symbolic_assembly
            .map(|value| value.label())
            .unwrap_or("default");
        let symbolic_execution = lsode2
            .symbolic_execution
            .as_ref()
            .map(Lsode2TaskExecutionSpec::label)
            .unwrap_or_else(|| "default".to_string());
        let linear_structure = lsode2
            .linear_system_structure
            .map(|value| value.label())
            .unwrap_or("default");
        let linear_policy = lsode2
            .linear_solver_policy
            .map(|value| value.label())
            .unwrap_or_else(|| "default".to_string());
        let native_execution = lsode2
            .native_execution
            .map(|value| value.label())
            .unwrap_or_else(|| "default".to_string());
        out.push_str(&format!(
            "lsode2_options: controller={controller}, symbolic_assembly={symbolic_assembly}, symbolic_execution={symbolic_execution}, linear_structure={linear_structure}, linear_policy={linear_policy}, native_execution={native_execution}, stop_conditions={}\n",
            lsode2.stop_conditions.len(),
        ));
    }
    if let Some(continuation) = &spec.continuation {
        out.push_str(&format!(
            "continuation: parameters={}, segments={}, mode={:?}, restart_each={}, monotonicity={:?}\n",
            continuation.parameters.join(","),
            continuation.value_grid.len(),
            continuation.mode,
            continuation.restart_each,
            continuation.monotonicity,
        ));
    } else {
        out.push_str("continuation: none\n");
    }
    out.push_str(&format!(
        "outputs: {}\n",
        render_ivp_output_targets(&spec.postprocessing)
    ));
    out.push_str("fully substituted equations:\n");
    out.push_str("idx | unknown | rhs\n");
    out.push_str("-------------------\n");
    for (idx, (name, rhs)) in spec
        .equations
        .unknowns
        .iter()
        .zip(spec.equations.rhs.iter())
        .enumerate()
    {
        out.push_str(&format!("{idx:>3} | {name} | {rhs}\n"));
    }
    out.push_str("preflight=valid; integration has not been run\n");
    out
}

fn ivp_telemetry_label(mode: Option<IvpTelemetryModeSpec>) -> &'static str {
    match mode {
        Some(IvpTelemetryModeSpec::Off) => "off",
        Some(IvpTelemetryModeSpec::Counters) => "counters",
        Some(IvpTelemetryModeSpec::Timings) => "timings",
        Some(IvpTelemetryModeSpec::Detailed) => "detailed",
        None => "default",
    }
}

fn render_ivp_output_targets(
    output: &crate::command_interpreter::task_parser_ivp::PostprocessingSpec,
) -> String {
    let mut targets = Vec::new();
    if output.save_csv {
        targets.push(format!(
            "csv:{}",
            output.csv_path.as_deref().unwrap_or("ivp_result.csv")
        ));
    }
    if output.save_txt {
        targets.push(format!(
            "text:{}",
            output.txt_path.as_deref().unwrap_or("ivp_result.txt")
        ));
    }
    if output.write_report {
        targets.push(format!(
            "report:{}",
            output.report_path.as_deref().unwrap_or("ivp_report.md")
        ));
    }
    if output.plotters_png {
        targets.push(format!(
            "plotters:{}",
            output.plotters_dir.as_deref().unwrap_or("ivp_plotters")
        ));
    }
    if output.gnuplot_png {
        targets.push(format!(
            "gnuplot:{}",
            output.gnuplot_dir.as_deref().unwrap_or("ivp_gnuplot")
        ));
    }
    if output.terminal_plot {
        targets.push("terminal".to_string());
    }
    if targets.is_empty() {
        "none".to_string()
    } else {
        targets.join(",")
    }
}

fn render_bvp_check(spec: &BvpTaskSpec) -> String {
    let mut out = String::new();
    out.push_str("[Task check] BVP\n");
    let family = match spec.solver.family {
        crate::command_interpreter::task_parser_bvp::BvpSolverFamilySpec::Damp => "BVP_Damp",
        crate::command_interpreter::task_parser_bvp::BvpSolverFamilySpec::Sci => "BVP_sci",
        crate::command_interpreter::task_parser_bvp::BvpSolverFamilySpec::CorrectionNewton => {
            "BVP_CorrectionNewton"
        }
    };
    let capabilities =
        crate::command_interpreter::task_parser_bvp::bvp_solver_capabilities(spec.solver.family);
    out.push_str(&format!("solver_family={family}\n"));
    out.push_str(&format!(
        "capabilities: frontends={}, executions={}, layouts={}, continuation={}, singular_term={}\n",
        capabilities.frontends.join("/"),
        capabilities.executions.join("/"),
        capabilities.layouts.join("/"),
        capabilities.supports_continuation,
        capabilities.supports_singular_term,
    ));
    out.push_str(&format!("matrix_layout={:?}\n", spec.solver.backend));
    out.push_str(&format!(
        "domain={}..{}, initial_nodes={}\n",
        spec.mesh.t0, spec.mesh.t_end, spec.mesh.n_steps
    ));
    if let Some(continuation) = &spec.continuation {
        out.push_str(&format!(
            "continuation: parameters={}, segments={}, mode={:?}, restart_each={}, restart_policy={:?}, monotonicity={:?}\n",
            continuation.parameters.join(","),
            continuation.value_grid.len(),
            continuation.mode,
            continuation.restart_each,
            continuation.restart_policy,
            continuation.monotonicity,
        ));
    } else {
        out.push_str("continuation: none\n");
    }
    match spec.solver.family {
        crate::command_interpreter::task_parser_bvp::BvpSolverFamilySpec::CorrectionNewton => {
            out.push_str(&render_correction_newton_settings(spec));
        }
        crate::command_interpreter::task_parser_bvp::BvpSolverFamilySpec::Damp => {
            out.push_str(&format!(
                "strategy={:?}, scheme={}\n",
                spec.solver.strategy, spec.solver.scheme
            ));
            out.push_str(&format!(
                "symbolic_frontend={:?}, execution={:?}\n",
                spec.solver.frontend, spec.solver.execution
            ));
        }
        crate::command_interpreter::task_parser_bvp::BvpSolverFamilySpec::Sci => {
            let execution_policy = spec
                .solver_options
                .execution_policy
                .unwrap_or(crate::numerical::BVP_sci::new::BvpSciExecutionPolicy::Sequential);
            let sparse_factorization = spec
                .solver_options
                .sparse_factorization
                .unwrap_or(crate::numerical::BVP_sci::new::BvpSciSparseFactorization::StandardFaer);
            out.push_str(&format!(
                "symbolic_frontend={:?}, execution={:?}, callback_policy={:?}\n",
                spec.solver.frontend.unwrap_or(
                    crate::command_interpreter::task_parser_bvp::BvpFrontendSpec::ExprLegacy
                ),
                spec.solver.execution.unwrap_or(
                    crate::command_interpreter::task_parser_bvp::BvpExecutionSpec::Lambdify
                ),
                execution_policy
            ));
            out.push_str(&format!(
                "tolerance={}, boundary_tolerance={}, max_nodes={}, newton_iterations={}, jacobian_refreshes={}, mesh_refinements={}, backtracking_trials={}\n",
                spec.solver_options.tolerance.unwrap_or(1e-3),
                spec.solver_options.boundary_tolerance.unwrap_or(spec.solver_options.tolerance.unwrap_or(1e-3)),
                spec.solver_options.max_nodes.unwrap_or_else(|| spec.mesh.n_steps.saturating_mul(16).max(128)),
                spec.solver_options.max_newton_iterations.or(spec.solver_options.max_iterations).unwrap_or(8),
                spec.solver_options.max_jacobian_refreshes.unwrap_or(4),
                spec.solver_options.max_mesh_refinements.unwrap_or(10),
                spec.solver_options.max_backtracking_steps.unwrap_or(4),
            ));
            out.push_str(&format!(
                "telemetry={}, output_policy={}, sparse_factorization={:?}\n",
                spec.solver_options
                    .telemetry
                    .map(|value| format!("{value:?}"))
                    .unwrap_or_else(|| "Off".to_string()),
                spec.solver_options
                    .output_policy
                    .map(|value| format!("{value:?}"))
                    .unwrap_or_else(|| "MeshAndValues".to_string()),
                sparse_factorization,
            ));
            if sparse_factorization
                == crate::numerical::BVP_sci::new::BvpSciSparseFactorization::SuperLu
            {
                out.push_str("SuperLU requirement: Cargo feature `superlu` plus SUPERLU_LIBRARY; unavailable bridge falls back to StandardFaer at runtime\n");
            }
            if matches!(
                spec.solver.execution,
                Some(crate::command_interpreter::task_parser_bvp::BvpExecutionSpec::Aot)
            ) {
                out.push_str(&format!(
                    "AOT preflight: lifecycle={}, output_dir={}, codegen={}, compiler={}\n",
                    spec.solver_options
                        .generated_backend
                        .aot_build_policy
                        .as_deref()
                        .unwrap_or("build_if_missing"),
                    spec.solver_options
                        .generated_backend
                        .aot_output_dir
                        .as_deref()
                        .unwrap_or("<required>"),
                    spec.solver_options
                        .generated_backend
                        .aot_codegen_backend
                        .as_deref()
                        .unwrap_or("default"),
                    spec.solver_options
                        .generated_backend
                        .aot_c_compiler
                        .as_deref()
                        .unwrap_or("toolchain default"),
                ));
            }
        }
    }
    out.push_str(&format!(
        "outputs: {}\n",
        render_bvp_output_targets(&spec.postprocessing)
    ));
    out.push_str("preflight=valid; numerical convergence has not been evaluated\n");
    out.push_str("fully substituted equations:\n");
    out.push_str("idx | unknown | rhs\n");
    out.push_str("-------------------\n");
    for (idx, (name, rhs)) in spec
        .equations
        .unknowns
        .iter()
        .zip(spec.equations.rhs.iter())
        .enumerate()
    {
        out.push_str(&format!("{idx:>3} | {name} | {rhs}\n"));
    }
    out
}

fn render_correction_newton_settings(spec: &BvpTaskSpec) -> String {
    match spec
        .solver_options
        .correction_newton
        .as_ref()
        .map(|options| options.build_options(&spec.solver))
    {
        Some(Ok(options)) => format!(
            "symbolic_frontend={:?}, execution={:?}, fixed_mesh_intervals={}, absolute_tolerance={}, default_relative_tolerance={}, max_iterations={}, max_jacobian_age={}, max_damping_trials={}, telemetry={:?}\n",
            options.symbolic_frontend,
            options.symbolic_execution,
            spec.mesh.n_steps,
            options.absolute_tolerance,
            options.default_relative_tolerance,
            options.max_iterations,
            options.max_jacobian_age,
            options.max_damping_trials,
            options.telemetry_mode,
        ),
        _ => "CorrectionNewton settings are invalid\n".into(),
    }
}

fn render_bvp_output_targets(
    output: &crate::command_interpreter::task_parser_bvp::BvpPostprocessingSpec,
) -> String {
    let mut targets = Vec::new();
    if output.save_csv {
        targets.push(format!(
            "csv:{}",
            output.csv_path.as_deref().unwrap_or("bvp_result.csv")
        ));
    }
    if output.save_txt {
        targets.push(format!(
            "text:{}",
            output.txt_path.as_deref().unwrap_or("bvp_result.txt")
        ));
    }
    if output.write_report {
        targets.push(format!(
            "report:{}",
            output.report_path.as_deref().unwrap_or("bvp_report.md")
        ));
    }
    if output.plotters_png {
        targets.push(format!(
            "plotters:{}",
            output.plotters_dir.as_deref().unwrap_or("bvp_plotters")
        ));
    }
    if output.gnuplot_png {
        targets.push(format!(
            "gnuplot:{}",
            output.gnuplot_dir.as_deref().unwrap_or("bvp_gnuplot")
        ));
    }
    if output.terminal_plot {
        targets.push("terminal".to_string());
    }
    if targets.is_empty() {
        "none".to_string()
    } else {
        targets.join(",")
    }
}

fn ivp_method_label(method: &IvpMethodSpec) -> String {
    match method {
        IvpMethodSpec::NonStiff(name) => name.clone(),
        IvpMethodSpec::Radau5 => "Radau5".to_string(),
        IvpMethodSpec::Bdf => "BDF".to_string(),
        IvpMethodSpec::BackwardEuler => "BackwardEuler".to_string(),
        IvpMethodSpec::Lsode => "LSODE".to_string(),
        IvpMethodSpec::Lsoda => "LSODA".to_string(),
        IvpMethodSpec::Lsode2 => "LSODE2".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn detect_task_kind_accepts_ivp_and_bvp_solver_lines() {
        let ivp = "task\nsolver: IVP\nmethod: RK45\n";
        let bvp = "task\nsolver: BVP\nstrategy: Damped\n";
        let correction_newton = "task\nsolver: BVP_CorrectionNewton\n";
        assert_eq!(
            detect_task_kind_from_str(ivp).unwrap(),
            TaskDocumentKind::Ivp
        );
        assert_eq!(
            detect_task_kind_from_str(bvp).unwrap(),
            TaskDocumentKind::Bvp
        );
        assert_eq!(
            detect_task_kind_from_str(correction_newton).unwrap(),
            TaskDocumentKind::Bvp
        );
        assert_eq!(
            detect_task_kind_from_json(
                r#"{"format_version":1,"task":{"solver":"BVP_CorrectionNewton"}}"#
            )
            .unwrap(),
            TaskDocumentKind::Bvp
        );
    }

    #[test]
    fn detects_and_runs_nonlinear_and_optimization_documents() {
        let nonlinear = "problem\nformulation: root\nvariables: x\nequations: x - 2\ninitial_guess: 0.0\nsolver\nmethod: newton\n";
        let fitting = "data\nx: 0.0, 1.0, 2.0\ny: 1.0, 3.0, 5.0\nmodel\nmethod: lm\nparameters: a, b\ninitial_guess: 0.0, 0.0\nequations: a*x + b\n";

        assert_eq!(
            detect_task_kind_from_str(nonlinear).unwrap(),
            TaskDocumentKind::Nonlinear
        );
        assert_eq!(
            detect_task_kind_from_str(fitting).unwrap(),
            TaskDocumentKind::Optimization
        );

        let nonlinear_result = run_task_from_str(nonlinear).expect("nonlinear task runs");
        assert_eq!(nonlinear_result.kind(), TaskDocumentKind::Nonlinear);
        let fitting_result = run_task_from_str(fitting).expect("optimization task runs");
        assert_eq!(fitting_result.kind(), TaskDocumentKind::Optimization);

        let nonlinear_spec = parse_task_spec_from_str(nonlinear).unwrap();
        assert!(render_task_preview(&nonlinear_spec).contains("Nonlinear"));
        assert!(render_task_check(&nonlinear_spec).contains("preflight=valid"));
        let fitting_spec = parse_task_spec_from_str(fitting).unwrap();
        assert!(render_task_preview(&fitting_spec).contains("Optimization"));
        assert!(render_task_check(&fitting_spec).contains("preflight=valid"));
    }

    #[test]
    fn requested_telemetry_is_explicit_when_native_nonlinear_error_has_no_snapshot() {
        let document = "problem\nformulation: root\nvariables: x\nequations: x^2 + 1\ninitial_guess: 0.0\nsolver\nmethod: newton\ntelemetry: detailed\n";
        let error = run_task_from_str(document).expect_err("the scalar Jacobian is singular");
        assert!(matches!(
            error,
            TaskRunnerError::NonlinearRunTelemetryUnavailable {
                requested_mode:
                    crate::command_interpreter::task_parser::TaskTelemetryMode::Detailed,
                ..
            }
        ));
        assert_eq!(error.category(), "solver");
        assert!(std::error::Error::source(&error).is_some());
    }

    #[test]
    fn json_and_dsl_share_the_same_native_specs_for_all_task_families() {
        let cases = [
            (
                "task\nsolver: IVP\nmethod: RK45\nequations\narg: t\ny: -y\ninitial_conditions\nt0: 0.0\nt_end: 0.1\ny0: 1.0\n",
                r#"{"format_version":1,"task":{"solver":"IVP","method":"RK45"},"equations":{"arg":"t","y":"-y"},"initial_conditions":{"t0":0.0,"t_end":0.1,"y0":1.0}}"#,
            ),
            (
                "task\nsolver: BVP\nstrategy: Damped\nscheme: forward\nmethod: Sparse\nequations\narg: x\nunknowns: z, y\nrhs: y-z, -z^3\nboundary_conditions\nz_left: 1.0\ny_right: 1.0\nmesh\nt0: 0.0\nt_end: 1.0\nn_steps: 20\ninitial_guess\nz: 0.0\ny: 0.0\n",
                r#"{"format_version":1,"task":{"solver":"BVP","strategy":"Damped","scheme":"forward","method":"Sparse"},"equations":{"arg":"x","unknowns":["z","y"],"rhs":["y-z","-z^3"]},"boundary_conditions":{"z_left":1.0,"y_right":1.0},"mesh":{"t0":0.0,"t_end":1.0,"n_steps":20},"initial_guess":{"z":0.0,"y":0.0}}"#,
            ),
            (
                "problem\nformulation: root\nvariables: x\nequations: x - 2\ninitial_guess: 0.0\nsolver\nmethod: newton\n",
                r#"{"format_version":1,"problem":{"formulation":"root","variables":"x","equations":"x - 2","initial_guess":0.0},"solver":{"method":"newton"}}"#,
            ),
            (
                "data\nx: 0.0, 1.0, 2.0\ny: 1.0, 3.0, 5.0\nmodel\nmethod: lm\nparameters: a, b\ninitial_guess: 0.0, 0.0\nequations: a*x + b\n",
                r#"{"format_version":1,"data":{"x":[0.0,1.0,2.0],"y":[1.0,3.0,5.0]},"model":{"method":"lm","parameters":["a","b"],"initial_guess":[0.0,0.0],"equations":"a*x + b"}}"#,
            ),
        ];

        for (dsl, json) in cases {
            let from_dsl = parse_task_spec_from_str(dsl).expect("DSL task parses");
            let from_json = parse_task_spec_from_json(json).expect("JSON task parses");
            assert_eq!(from_json.kind(), from_dsl.kind());
            assert_eq!(
                render_task_preview(&from_json),
                render_task_preview(&from_dsl)
            );
            assert_eq!(render_task_check(&from_json), render_task_check(&from_dsl));
        }
    }

    #[test]
    fn json_task_runner_executes_nonlinear_and_optimization_documents() {
        let nonlinear = r#"{"format_version":1,"problem":{"formulation":"root","variables":"x","equations":"x - 2","initial_guess":0.0},"solver":{"method":"newton"}}"#;
        let fitting = r#"{"format_version":1,"data":{"x":[0.0,1.0,2.0],"y":[1.0,3.0,5.0]},"model":{"method":"lm","parameters":["a","b"],"initial_guess":[0.0,0.0],"equations":"a*x + b"}}"#;

        let root = run_task_from_json(nonlinear).expect("JSON root task runs");
        assert!(root.is_successful());
        assert_eq!(root.kind(), TaskDocumentKind::Nonlinear);

        let fit = run_task_from_json(fitting).expect("JSON optimization task runs");
        assert!(fit.is_successful());
        assert_eq!(fit.kind(), TaskDocumentKind::Optimization);
    }

    #[test]
    fn cancellation_capabilities_match_all_fitting_native_routes() {
        let varpro = r#"{"format_version":1,"data":{"x":[0.0,1.0,2.0,3.0],"y":[2.0,1.5,1.0,0.8]},"model":{"method":"varpro","parameters":["tau"],"initial_guess":[1.2],"basis_parameters":["tau"],"basis_equations":["exp(-x/tau)"],"invariants":["1"]}}"#;
        let gavin = r#"{"format_version":1,"data":{"x":[0.0,1.0],"y":[1.0,3.0]},"model":{"method":"gavin","parameters":["a","b"],"initial_guess":[1.0,1.0],"equations":"a*x+b"}}"#;

        let varpro = parse_task_spec_from_json(varpro).expect("VarPro task parses");
        let gavin = parse_task_spec_from_json(gavin).expect("Gavin task parses");
        assert!(task_supports_cooperative_cancellation(&varpro));
        assert!(task_supports_cooperative_cancellation(&gavin));
    }

    #[test]
    fn ivp_continuation_cancellation_capability_tracks_lifecycle_policy() {
        let base = r#"{"format_version":1,"task":{"solver":"IVP","method":"BDF"},"equations":{"unknowns":"y","rhs":"-k*y","parameters":"k","parameter_values":1.0},"initial_conditions":{"t0":0.0,"t_end":1.0,"y0":1.0}"#;
        let fresh = format!(
            "{base},\"continuation\":{{\"parameter\":\"k\",\"values\":[1.0,2.0],\"mode\":\"fresh\"}}}}"
        );
        let explicit_restart = format!(
            "{base},\"continuation\":{{\"parameter\":\"k\",\"values\":[1.0,2.0],\"restart_policy\":\"restart_each\"}}}}"
        );
        let stateful_continue = format!(
            "{base},\"continuation\":{{\"parameter\":\"k\",\"values\":[1.0,2.0],\"restart_policy\":\"continue\"}}}}"
        );

        for input in [fresh, explicit_restart] {
            let task = parse_task_spec_from_json(&input).expect("supported loop parses");
            assert!(task_supports_cooperative_cancellation(&task));
        }
        let task = parse_task_spec_from_json(&stateful_continue)
            .expect("stateful loop parses but has a narrower capability");
        assert!(!task_supports_cooperative_cancellation(&task));

        for method in ["LSODE", "LSODA"] {
            let input = format!(
                r#"{{"format_version":1,"task":{{"solver":"IVP","method":"{method}"}},"equations":{{"arg":"t","y":"-y"}},"initial_conditions":{{"t0":0.0,"t_end":1.0,"y0":1.0}}}}"#
            );
            let task = parse_task_spec_from_json(&input).expect("legacy route parses");
            assert!(!task_supports_cooperative_cancellation(&task));
        }
    }

    #[test]
    fn json_task_runner_executes_ivp_and_bvp_documents() {
        let ivp = r#"{"format_version":1,"task":{"solver":"IVP","method":"RK45"},"equations":{"arg":"t","y":"-y"},"initial_conditions":{"t0":0.0,"t_end":0.1,"y0":1.0}}"#;
        let bvp = r#"{"format_version":1,"task":{"solver":"BVP","strategy":"Damped","method":"Dense"},"equations":{"arg":"x","unknowns":"y","rhs":"0"},"boundary_conditions":{"y_left":1.0},"mesh":{"t0":0.0,"t_end":1.0,"n_steps":4},"initial_guess":{"y":1.0}}"#;

        let ivp_result = run_task_from_json(ivp).expect("JSON IVP task runs");
        assert!(ivp_result.is_successful());
        assert_eq!(ivp_result.kind(), TaskDocumentKind::Ivp);

        let bvp_result = run_task_from_json(bvp).expect("JSON BVP task runs");
        assert!(bvp_result.is_successful());
        assert_eq!(bvp_result.kind(), TaskDocumentKind::Bvp);
    }

    #[test]
    fn json_file_api_and_batch_runner_dispatch_by_extension() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("root_task.json");
        fs::write(
            &path,
            r#"{"format_version":1,"problem":{"formulation":"root","variables":"x","equations":"x - 2","initial_guess":0.0},"solver":{"method":"newton"}}"#,
        )
        .unwrap();

        let spec = parse_task_spec_from_file(&path).expect("JSON file parses");
        assert_eq!(spec.kind(), TaskDocumentKind::Nonlinear);
        let report = run_task_batch([path]);
        assert_eq!(report.failed_count(), 0);
        assert_eq!(report.passed_count(), 1);
        assert_eq!(report.entries[0].kind, Some(TaskDocumentKind::Nonlinear));
    }

    #[test]
    fn json_errors_are_typed_as_parse_failures_and_keep_field_paths() {
        let malformed = parse_task_spec_from_json("{not json}").unwrap_err();
        assert!(matches!(malformed, TaskRunnerError::Json(_)));
        assert_eq!(malformed.category(), "parse");

        let nested =
            parse_task_spec_from_json(r#"{"format_version":1,"problem":{"equations":{"x":1}}}"#)
                .unwrap_err();
        assert!(nested.to_string().contains("$.problem.equations"));
    }

    #[test]
    fn json_rejects_unknown_sections_and_fixed_fields_with_paths() {
        let cases = [
            (
                r#"{"format_version":1,"task":{"solver":"IVP","method":"RK45"},"equations":{"y":"-y"},"initial_conditions":{"t0":0.0,"t_end":1.0,"y0":1.0},"solver_options":{"rtoll":1e-6}}"#,
                "$.solver_options.rtoll",
            ),
            (
                r#"{"format_version":1,"task":{"solver":"IVP","method":"RK45"},"equations":{"y":"-y"},"initial_conditions":{"t0":0.0,"t_end":1.0,"y0":1.0},"extra":{}}"#,
                "$.extra",
            ),
            (
                r#"{"format_version":1,"task":{"solver":"BVP","method":"Sparse"},"equations":{"unknowns":["y"],"rhs":["0"]},"boundary_conditions":{"y_left":1.0},"mesh":{"t0":0.0,"t_end":1.0,"n_steps":4},"initial_guess":{"y":1.0},"postprocessing":{"plot_pgn":true}}"#,
                "$.postprocessing.plot_pgn",
            ),
        ];

        for (input, expected_path) in cases {
            let error = parse_task_spec_from_json(input).unwrap_err();
            assert!(matches!(
                error,
                TaskRunnerError::Json(TaskJsonError::UnknownField { .. })
            ));
            assert!(error.to_string().contains(expected_path), "{error}");
            assert_eq!(error.category(), "parse");
        }
    }

    #[test]
    fn json_requires_numeric_ivp_schema_version() {
        let input = r#"{"format_version":1,"task":{"solver":"IVP","method":"RK45","schema_version":"1"},"equations":{"y":"-y"},"initial_conditions":{"t0":0.0,"t_end":1.0,"y0":1.0}}"#;
        let error = parse_task_spec_from_json(input).unwrap_err();
        assert!(matches!(
            error,
            TaskRunnerError::Json(TaskJsonError::InvalidShape { .. })
        ));
        assert!(error.to_string().contains("$.task.schema_version"));

        let integral_float = r#"{"format_version":1.0,"task":{"solver":"IVP","method":"RK45","schema_version":1.0},"equations":{"y":"-y"},"initial_conditions":{"t0":0.0,"t_end":1.0,"y0":1.0}}"#;
        parse_task_spec_from_json(integral_float)
            .expect("JSON Schema integer-valued numbers remain valid integer fields");
    }

    #[test]
    fn json_schema_preserves_valid_dynamic_bvp_state_fields() {
        let bvp = r#"{"format_version":1,"task":{"solver":"BVP","strategy":"Damped","method":"Sparse"},"equations":{"arg":"x","unknowns":["temperature","fuel"],"rhs":["fuel","-fuel"]},"boundary_conditions":{"temperature_left":300.0,"fuel_right":0.0},"mesh":{"t0":0.0,"t_end":1.0,"n_steps":8},"initial_guess":{"temperature":300.0,"fuel":0.1},"bounds":{"temperature":[200.0,2000.0],"fuel":[0.0,1.0]}}"#;
        parse_task_spec_from_json(bvp).expect("dynamic fields matching declared states are valid");
    }

    #[test]
    fn published_json_schema_tracks_fixed_solver_option_allowlists() {
        let schema: serde_json::Value =
            serde_json::from_str(crate::command_interpreter::task_json::TASK_DOCUMENT_JSON_SCHEMA)
                .expect("published task JSON schema is valid JSON");
        assert_eq!(
            schema["$schema"],
            "https://json-schema.org/draft/2020-12/schema"
        );
        assert_eq!(schema["oneOf"].as_array().unwrap().len(), 4);
        assert_schema_refs_resolve(&schema, &schema);

        let assert_properties = |definition: &str, expected: &[&str]| {
            let properties = schema["$defs"][definition]["properties"]
                .as_object()
                .expect("schema definition has a properties object");
            let mut actual = properties.keys().map(String::as_str).collect::<Vec<_>>();
            let mut expected = expected.to_vec();
            actual.sort_unstable();
            expected.sort_unstable();
            assert_eq!(
                actual, expected,
                "schema properties drifted for {definition}"
            );
        };

        assert_properties("ivpSolverOptions", IVP_SOLVER_FIELDS);
        assert_properties("bvpSolverOptions", BVP_SOLVER_FIELDS);
        assert_properties("postprocessing", POSTPROCESSING_FIELDS);
    }

    #[test]
    fn published_schema_and_runtime_accept_the_same_family_contract_fixtures() {
        let schema: serde_json::Value =
            serde_json::from_str(crate::command_interpreter::task_json::TASK_DOCUMENT_JSON_SCHEMA)
                .expect("published task JSON schema is valid JSON");
        let validator =
            jsonschema::validator_for(&schema).expect("published schema compiles as Draft 2020-12");
        let valid_documents = [
            // IVP: pair-style and declared symbolic equations, including inline parameters.
            r#"{"format_version":1,"task":{"solver":"IVP","method":"RK45","schema_version":1},"equations":{"arg":"t","y":"-k*y","parameters":["k"],"parameter_values":[2]},"initial_conditions":{"t0":0,"t_end":1,"y0":[1]}}"#,
            r#"{"format_version":1,"task":{"solver":"IVP","method":"RK45"},"equations":{"unknowns":"y","rhs":"-k*y","parameters":"k","parameter_values":1.0},"initial_conditions":{"t0":0.0,"t_end":1.0,"y0":1.0},"continuation":{"parameter":"k","values":[1.0,2.0]}}"#,
            // BVP: both equation styles and documented dynamic state-name sections.
            r#"{"format_version":1,"task":{"solver":"BVP","strategy":"Damped","method":"Sparse"},"equations":{"arg":"x","unknowns":["y","z"],"rhs":["z","-y"]},"boundary_conditions":{"y_left":1.0,"z_right":0.0},"mesh":{"t0":0.0,"t_end":1.0,"n_steps":8},"initial_guess":{"y":0.5,"z":0.0},"bounds":{"y":[0.0,2.0]}}"#,
            r#"{"format_version":1,"task":{"solver":"BVP","strategy":"Frozen","method":"Sparse"},"equations":{"y":"0"},"boundary_conditions":{"y_left":1.0},"mesh":{"t0":0.0,"t_end":1.0,"n_steps":4},"initial_guess":{"guess":[1.0]}}"#,
            // Nonlinear: root and least-squares-specific options/method combinations.
            r#"{"format_version":1,"problem":{"formulation":"root_finding","variables":["x"],"equations":["x-2"],"initial_guess":0.0,"parameters":["p"],"parameter_values":[1.0]},"solver":{"method":"damped_newton","tolerance":1e-8,"max_iterations":20,"telemetry":"counters"}}"#,
            r#"{"format_version":1,"problem":{"formulation":"least-squares","variables":["x"],"equations":["x-2"],"initial_guess":[0.0]},"solver":{"method":"levenberg-marquardt","residual_tolerance":1e-8,"gradient_tolerance":1e-8,"telemetry":"detailed"}}"#,
            // Optimization: each model family, plus method-scoped settings.
            r#"{"format_version":1,"data":{"x":[0.0,1.0],"y":[1.0,3.0]},"model":{"method":"lm","parameters":["a","b"],"initial_guess":[1.0,1.0],"equations":"a*x+b"},"solver":{"tolerance":1e-8,"telemetry":"timings"}}"#,
            r#"{"format_version":1,"data":{"x":[0.0,1.0],"y":[1.0,3.0]},"model":{"method":"gavin-lm","parameters":["a","b"],"initial_guess":[1.0,1.0],"equations":"a*x+b"},"solver":{"gavin_weights":[1.0,1.0],"gavin_positive_parameters":["a"],"telemetry":"counters"}}"#,
            r#"{"format_version":1,"data":{"x":[0.0,1.0],"y":[1.0,3.0]},"model":{"method":"variable_projection","parameters":["a","b"],"initial_guess":[1.0,1.0],"basis_parameters":["a","b"],"basis_equations":["x","1"],"invariants":["x"]}}"#,
        ];

        for (index, input) in valid_documents.iter().enumerate() {
            let value: serde_json::Value = serde_json::from_str(input).unwrap();
            assert!(
                validator.is_valid(&value),
                "schema rejected fixture {index}"
            );
            parse_task_spec_from_json(input)
                .unwrap_or_else(|error| panic!("runtime rejected fixture {index}: {error}"));
        }

        let invalid_documents = [
            // Wrong type, unknown fields, unsupported method-specific fields, and DSL-only aliases.
            r#"{"format_version":1,"task":{"solver":"IVP","method":"RK45"},"equations":{"y":"-y"},"initial_conditions":{"t0":"0","t_end":1.0,"y0":1.0}}"#,
            r#"{"format_version":1,"task":{"solver":"BVP"},"equations":{"unknowns":"y","rhs":"0"},"boundary_conditions":{"y_left":1.0},"mesh":{"t0":0.0,"t_end":1.0,"n_steps":4},"initial_guess":{"y":1.0},"bounds":{"y":[0.0,1.0,2.0]}}"#,
            r#"{"format_version":1,"problem":{"formulation":"root","variables":"x","equations":"x","initial_guess":0.0},"solver":{"method":"newton","residual_tolerance":1e-6}}"#,
            r#"{"format_version":1,"problem":{"formulation":"root","variables":"x","equations":"x","initial_guess":0.0},"solver":{"method":"newton","telemetry":"verbose"}}"#,
            r#"{"format_version":1,"data":{"x":[0.0,1.0],"y":[1.0,3.0]},"model":{"method":"lm","parameters":"a","initial_guess":1.0,"basis_parameters":"a","basis_equations":"x"}}"#,
            r#"{"format_version":1,"task":{"solver":"IVP","method":"RK45"},"equations":{"y":"-y","params":["k"]},"initial_conditions":{"t0":0.0,"t_end":1.0,"y0":1.0}}"#,
        ];

        for (index, input) in invalid_documents.iter().enumerate() {
            let value: serde_json::Value = serde_json::from_str(input).unwrap();
            assert!(
                !validator.is_valid(&value),
                "schema accepted invalid fixture {index}"
            );
            assert!(
                parse_task_spec_from_json(input).is_err(),
                "runtime accepted invalid fixture {index}"
            );
        }
    }

    #[test]
    fn schema_declared_field_types_are_rejected_by_runtime_when_mismatched() {
        let schema: serde_json::Value =
            serde_json::from_str(crate::command_interpreter::task_json::TASK_DOCUMENT_JSON_SCHEMA)
                .unwrap();
        let validator = jsonschema::validator_for(&schema).unwrap();
        let cases = [
            (
                "ivpTask",
                "solver_options",
                r#"{"format_version":1,"task":{"solver":"IVP","method":"RK45"},"equations":{"y":"-y"},"initial_conditions":{"t0":0.0,"t_end":1.0,"y0":1.0},"solver_options":{}}"#,
            ),
            (
                "bvpTask",
                "solver_options",
                r#"{"format_version":1,"task":{"solver":"BVP","strategy":"Damped","method":"Dense"},"equations":{"y":"0"},"boundary_conditions":{"y_left":1.0},"mesh":{"t0":0.0,"t_end":1.0,"n_steps":4},"initial_guess":{"y":1.0},"solver_options":{}}"#,
            ),
            (
                "ivpTask",
                "postprocessing",
                r#"{"format_version":1,"task":{"solver":"IVP","method":"RK45"},"equations":{"y":"-y"},"initial_conditions":{"t0":0.0,"t_end":1.0,"y0":1.0},"postprocessing":{}}"#,
            ),
            (
                "bvpTask",
                "postprocessing",
                r#"{"format_version":1,"task":{"solver":"BVP","strategy":"Damped","method":"Dense"},"equations":{"y":"0"},"boundary_conditions":{"y_left":1.0},"mesh":{"t0":0.0,"t_end":1.0,"n_steps":4},"initial_guess":{"y":1.0},"postprocessing":{}}"#,
            ),
            (
                "nonlinearTask",
                "problem",
                r#"{"format_version":1,"problem":{"formulation":"root","variables":"x","equations":"x-2","initial_guess":0.0},"solver":{"method":"newton"}}"#,
            ),
            (
                "nonlinearTask",
                "solver",
                r#"{"format_version":1,"problem":{"formulation":"root","variables":"x","equations":"x-2","initial_guess":0.0},"solver":{"method":"newton"}}"#,
            ),
            (
                "optimizationTask",
                "data",
                r#"{"format_version":1,"data":{"x":[0.0,1.0],"y":[1.0,3.0]},"model":{"method":"lm","parameters":"a","initial_guess":1.0,"equations":"a*x"}}"#,
            ),
            (
                "optimizationTask",
                "model",
                r#"{"format_version":1,"data":{"x":[0.0,1.0],"y":[1.0,3.0]},"model":{"method":"lm","parameters":"a","initial_guess":1.0,"equations":"a*x"}}"#,
            ),
            (
                "optimizationTask",
                "solver",
                r#"{"format_version":1,"data":{"x":[0.0,1.0],"y":[1.0,3.0]},"model":{"method":"lm","parameters":"a","initial_guess":1.0,"equations":"a*x"},"solver":{}}"#,
            ),
        ];

        fn resolve_local_refs<'a>(
            mut node: &'a serde_json::Value,
            root: &'a serde_json::Value,
        ) -> &'a serde_json::Value {
            while let Some(reference) = node.get("$ref").and_then(serde_json::Value::as_str) {
                let name = reference
                    .strip_prefix("#/$defs/")
                    .expect("test schema refs should be local definitions");
                node = &root["$defs"][name];
            }
            node
        }

        for (task_def, section, source) in cases {
            let section_schema =
                resolve_local_refs(&schema["$defs"][task_def]["properties"][section], &schema);
            let properties = section_schema["properties"]
                .as_object()
                .expect("tested section schema has explicit properties");
            for (field, field_schema) in properties {
                let wrong_type = match field_schema["type"].as_str() {
                    Some("string") => serde_json::json!(17),
                    Some("number") => serde_json::json!("not-a-number"),
                    Some("integer") => serde_json::json!(1.5),
                    Some("boolean") => serde_json::json!("not-a-boolean"),
                    Some(_) => serde_json::json!({"not": "a scalar or supported array"}),
                    None => serde_json::json!({"not": "a supported scalar or array"}),
                };
                let mut value: serde_json::Value = serde_json::from_str(source).unwrap();
                value[section][field] = wrong_type;
                assert!(
                    !validator.is_valid(&value),
                    "schema accepted invalid type for {task_def}.{section}.{field}"
                );
                let input = serde_json::to_string(&value).unwrap();
                assert!(
                    parse_task_spec_from_json(&input).is_err(),
                    "runtime accepted invalid type for {task_def}.{section}.{field}"
                );
            }
        }
    }

    fn assert_schema_refs_resolve(node: &serde_json::Value, root: &serde_json::Value) {
        match node {
            serde_json::Value::Object(object) => {
                if let Some(reference) = object.get("$ref").and_then(serde_json::Value::as_str) {
                    let definition = reference
                        .strip_prefix("#/$defs/")
                        .expect("schema references should target local definitions");
                    assert!(
                        root["$defs"].get(definition).is_some(),
                        "unresolved schema reference {reference}"
                    );
                }
                for child in object.values() {
                    assert_schema_refs_resolve(child, root);
                }
            }
            serde_json::Value::Array(values) => {
                for child in values {
                    assert_schema_refs_resolve(child, root);
                }
            }
            _ => {}
        }
    }

    #[test]
    fn normalized_document_map_dispatches_without_text_or_python_roundtrip() {
        let input = "problem\nformulation: root\nvariables: x\nequations: x - 2\ninitial_guess: 0.0\nsolver\nmethod: newton\n";
        let mut parser = crate::command_interpreter::task_parser::DocumentParser::new(input.into());
        parser.parse_document_typed().unwrap();
        parser.try_keys_to_lower_case_typed(None).unwrap();
        let document = parser.get_result().unwrap();

        assert_eq!(
            detect_task_kind_from_document(document).unwrap(),
            TaskDocumentKind::Nonlinear
        );
        let spec = parse_task_spec_from_document(document).unwrap();
        assert_eq!(spec.kind(), TaskDocumentKind::Nonlinear);
        let result = run_task_from_spec(spec).unwrap();
        assert!(result.is_successful());
    }

    #[test]
    fn batch_keeps_going_after_a_new_family_parse_failure() {
        let directory = tempdir().unwrap();
        let bad = directory.path().join("bad_optimization.task");
        let good = directory.path().join("good_nonlinear.task");
        fs::write(
            &bad,
            "data\nx: 0.0, 1.0\ny: 1.0, 2.0\nmodel\nmethod: not-a-method\n",
        )
        .unwrap();
        fs::write(
            &good,
            "problem\nformulation: root\nvariables: x\nequations: x - 2\ninitial_guess: 0.0\nsolver\nmethod: newton\n",
        )
        .unwrap();

        let report = run_task_batch([bad, good]);
        assert_eq!(report.failed_count(), 1);
        assert_eq!(report.passed_count(), 1);
        assert_eq!(report.entries[0].kind, Some(TaskDocumentKind::Optimization));
        assert_eq!(report.entries[1].kind, Some(TaskDocumentKind::Nonlinear));
        assert_eq!(report.entries[0].error_category, Some("configuration"));
    }

    #[test]
    fn batch_marks_nonlinear_nonconvergence_as_solver_failure() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("nonconverged_root.task");
        fs::write(
            &path,
            "problem\nformulation: root\nvariables: x\nequations: x^2 - 2\ninitial_guess: 10.0\nsolver\nmethod: newton\nmax_iterations: 1\n",
        )
        .unwrap();

        let report = run_task_batch([path]);
        assert_eq!(report.failed_count(), 1);
        assert_eq!(report.passed_count(), 0);
        assert_eq!(report.entries[0].kind, Some(TaskDocumentKind::Nonlinear));
        assert_eq!(report.entries[0].error_category, Some("solver_result"));
        assert!(
            report.entries[0]
                .error
                .as_deref()
                .unwrap_or_default()
                .contains("terminated")
        );
    }

    #[test]
    fn run_task_from_str_routes_ivp_document() {
        let ivp_doc = r#"
task
solver: IVP
method: RK45

equations
arg: t
y: -y

initial_conditions
t0: 0.0
t_end: 0.1
y0: 1.0

solver_options
step_size: 1e-3
"#;
        let result = run_task_from_str(ivp_doc).expect("IVP document should run");
        assert_eq!(result.kind(), TaskDocumentKind::Ivp);
    }

    #[test]
    fn render_task_preview_includes_solver_kind_and_equations() {
        let ivp_doc = r#"
task
solver: IVP
method: LSODE2

equations
arg: t
y: -2.0*y

initial_conditions
t0: 0.0
t_end: 0.1
y0: 1.0

solver_options
max_step: 1e-2
"#;
        let spec = parse_task_spec_from_str(ivp_doc).expect("IVP document should parse");
        let preview = render_task_preview(&spec);
        assert!(preview.contains("[Task preview] IVP"));
        assert!(preview.contains("method=LSODE2"));
        assert!(preview.contains("y |"));
        assert!(preview.contains("2"));
    }

    #[test]
    fn render_task_check_exposes_normalized_lsode2_route_without_debug_dump() {
        let ivp_doc = r#"
task
solver: IVP
method: LSODE2

equations
arg: t
unknowns: y
rhs: -y

initial_conditions
t0: 0.0
t_end: 0.1
y0: 1.0

solver_options
symbolic_frontend: AtomView
symbolic_execution: Lambdify
matrix_layout: sparse
execution_policy: auto
telemetry: timings
"#;
        let spec = parse_task_spec_from_str(ivp_doc).expect("IVP document should parse");
        let check = render_task_check(&spec);

        assert!(check.contains("lsode2_options: controller=bdf_only"));
        assert!(check.contains("symbolic_assembly=atom_view"));
        assert!(check.contains("symbolic_execution=lambdify"));
        assert!(check.contains("linear_structure=sparse"));
        assert!(check.contains("linear_policy=default"));
        assert!(check.contains("native_execution=default"));
        assert!(check.contains(
            "native_options: symbolic_frontend=atom_view, symbolic_execution=lambdify, matrix_layout=sparse, execution_policy=auto, telemetry=timings"
        ));
        assert!(!check.contains("Lsode2TaskOptionsSpec {"));
    }

    #[test]
    fn render_task_check_uses_one_native_options_contract_for_stiff_ivp_routes() {
        for method in ["BDF", "Radau5", "BackwardEuler"] {
            let ivp_doc = format!(
                "task\nsolver: IVP\nmethod: {method}\n\n\
                 equations\narg: t\nunknowns: y\nrhs: -y\n\n\
                 initial_conditions\nt0: 0.0\nt_end: 0.1\ny0: 1.0\n\n\
                 solver_options\n\
                 symbolic_frontend: ExprLegacy\n\
                 symbolic_execution: Lambdify\n\
                 matrix_layout: dense\n\
                 execution_policy: sequential\n\
                 telemetry: counters\n"
            );
            let spec = parse_task_spec_from_str(&ivp_doc)
                .unwrap_or_else(|error| panic!("{method} document should parse: {error}"));
            let check = render_task_check(&spec);
            assert!(check.contains(
                "native_options: symbolic_frontend=expr_legacy, symbolic_execution=lambdify, matrix_layout=dense, execution_policy=sequential, telemetry=counters"
            ));
            assert!(!check.contains("Some(ExprLegacy)"));
            assert!(!check.contains("Some(Lambdify"));
        }
    }

    #[test]
    fn render_task_check_shows_fully_substituted_equations() {
        let bvp_doc = r#"
task
solver: BVP
strategy: Damped
backend: Sparse

equations
arg: x
parameters: a
parameter_values: 4.0
unknowns: y
rhs: gain * y

where
base: 2 * x
gain: base + a

boundary_conditions
y_left: 1.0
y_right: 0.0

mesh
t0: 0.0
t_end: 1.0
n_steps: 10

initial_guess
y: 0.0
"#;
        let spec = parse_task_spec_from_str(bvp_doc).expect("BVP document should parse");
        let check = render_task_check(&spec);
        assert!(check.contains("[Task check] BVP"));
        assert!(check.contains("fully substituted equations"));
        assert!(check.contains('4'));
        assert!(!check.contains("gain:"));
    }

    #[test]
    fn bvp_preflight_reports_resolved_bvp_sci_route_and_limits() {
        let task = r#"
task
solver: BVP_sci
frontend: AtomViewNative
execution: Lambdify
method: Sparse

equations
arg: x
unknowns: y
rhs: -y

boundary_conditions
y_left: 1.0

mesh
t0: 0.0
t_end: 1.0
n_steps: 8

initial_guess
y: 0.0

solver_options
max_nodes: 200
max_newton_iterations: 6
"#;
        let parsed = parse_task_spec_from_str(task).expect("task should parse");
        let check = render_task_check(&parsed);
        assert!(check.contains("solver_family=BVP_sci"));
        assert!(check.contains("symbolic_frontend=AtomViewNative"));
        assert!(check.contains("max_nodes=200"));
        assert!(check.contains("preflight=valid; numerical convergence has not been evaluated"));
    }

    #[test]
    fn parse_new_ivp_task_doc_with_where_section() {
        let path = format!(
            "{}/examples/task_docs/ivp_solid_combustion_with_sublimation_task.txt",
            env!("CARGO_MANIFEST_DIR")
        );
        let text = fs::read_to_string(path).expect("IVP task doc should be readable");
        let spec = parse_task_spec_from_str(&text).expect("IVP task doc should parse");
        assert_eq!(spec.kind(), TaskDocumentKind::Ivp);
    }

    #[test]
    fn parse_new_bvp_task_doc_combustion_like() {
        let path = format!(
            "{}/examples/task_docs/bvp_combustion_1000_task.txt",
            env!("CARGO_MANIFEST_DIR")
        );
        let text = fs::read_to_string(path).expect("BVP task doc should be readable");
        let spec = parse_task_spec_from_str(&text).expect("BVP task doc should parse");
        assert_eq!(spec.kind(), TaskDocumentKind::Bvp);
    }

    #[test]
    fn batch_runner_continues_after_failure_and_writes_compact_typed_table() {
        let directory = tempdir().expect("temporary task directory");
        let valid_before = directory.path().join("valid_before.txt");
        let invalid = directory.path().join("invalid.txt");
        let valid_after = directory.path().join("valid_after.txt");
        let valid = r#"
task
solver: IVP
method: RK45

equations
arg: t
y: -y

initial_conditions
t0: 0.0
t_end: 0.01
y0: 1.0

solver_options
step_size: 1e-3
"#;
        fs::write(&valid_before, valid).expect("write first task");
        fs::write(&invalid, "task\nsolver: IVP\nmethod: UnknownMethod\n")
            .expect("write invalid task");
        fs::write(&valid_after, valid).expect("write second task");

        let report = run_task_batch([valid_before, invalid, valid_after]);
        assert_eq!(report.entries.len(), 3);
        assert_eq!(report.passed_count(), 2);
        assert_eq!(report.failed_count(), 1);
        assert_eq!(report.entries[1].kind, Some(TaskDocumentKind::Ivp));
        assert_eq!(report.entries[1].error_category, Some("configuration"));
        let table = report.render_table();
        assert!(table.contains("passed"));
        assert!(table.contains("failed"));
        assert!(table.contains("configuration"));

        let table_path = directory.path().join("summary.md");
        report
            .write_table(&table_path)
            .expect("write batch summary");
        assert_eq!(fs::read_to_string(table_path).unwrap(), table);
    }
}
