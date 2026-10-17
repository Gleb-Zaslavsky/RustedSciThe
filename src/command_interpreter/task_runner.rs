//! Unified task-document runner for IVP/BVP shells.
//!
//! This module sits one level above `task_parser_ivp` and `task_parser_bvp`:
//! - detects task kind (`solver: IVP|BVP`) from document text
//! - routes parsing/execution to the corresponding parser
//! - provides a file-based API that is convenient for CLI tools and embedding
//!   into other Rust programs

use crate::command_interpreter::task_parser_bvp::{
    BvpTaskError, BvpTaskRunResult, BvpTaskSpec, parse_bvp_task_from_str, run_bvp_task,
};
use crate::command_interpreter::task_parser_ivp::{
    IvpMethodSpec, IvpTaskError, IvpTaskRunResult, IvpTaskSpec, IvpTelemetryModeSpec,
    Lsode2TaskExecutionSpec, parse_ivp_task_from_str, run_ivp_task,
};
use std::fs;
use std::path::{Path, PathBuf};
use tabled::{Table, Tabled};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskDocumentKind {
    Ivp,
    Bvp,
}

#[derive(Debug)]
pub enum ParsedTaskSpec {
    Ivp(IvpTaskSpec),
    Bvp(BvpTaskSpec),
}

#[derive(Debug)]
pub enum TaskRunResult {
    Ivp(IvpTaskRunResult),
    Bvp(BvpTaskRunResult),
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
    Ivp(IvpTaskError),
    Bvp(BvpTaskError),
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
                "task document must contain a `solver: IVP|BVP` field in the `task` section"
            ),
            Self::UnsupportedSolver { value } => {
                write!(
                    f,
                    "unsupported solver `{value}` in task document (expected IVP or BVP)"
                )
            }
            Self::Ivp(err) => write!(f, "{err}"),
            Self::Bvp(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for TaskRunnerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Ivp(err) => Some(err),
            Self::Bvp(err) => Some(err),
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
            Self::MissingSolverField | Self::UnsupportedSolver { .. } => "parse",
            Self::Ivp(error) => error.category(),
            Self::Bvp(error) => error.category(),
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
            "solver" | "solver_result" | "solver_exhaustion" | "solver_singular"
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
            let detected_kind = detect_task_kind_from_str(&source).ok();
            match parse_task_spec_from_str(&source).map(|mut spec| {
                anchor_task_paths(&mut spec, &path);
                spec
            }) {
                Ok(spec) => match run_task_from_spec(spec) {
                    Ok(_) => BatchTaskSummary {
                        path,
                        status: BatchTaskStatus::Passed,
                        kind: detected_kind,
                        error_category: None,
                        error: None,
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
    }
}

impl TaskRunResult {
    pub fn kind(&self) -> TaskDocumentKind {
        match self {
            Self::Ivp(_) => TaskDocumentKind::Ivp,
            Self::Bvp(_) => TaskDocumentKind::Bvp,
        }
    }
}

impl ParsedTaskSpec {
    pub fn kind(&self) -> TaskDocumentKind {
        match self {
            Self::Ivp(_) => TaskDocumentKind::Ivp,
            Self::Bvp(_) => TaskDocumentKind::Bvp,
        }
    }
}

pub fn detect_task_kind_from_str(input: &str) -> Result<TaskDocumentKind, TaskRunnerError> {
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
            "bvp" | "bvp_damp" | "bvp_damped" | "bvp_sci" | "bvpsci" | "bvp_scipy"
            | "scipy_bvp" => Ok(TaskDocumentKind::Bvp),
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
    }
}

pub fn run_task_from_str(input: &str) -> Result<TaskRunResult, TaskRunnerError> {
    let spec = parse_task_spec_from_str(input)?;
    run_task_from_spec(spec)
}

pub fn run_task_from_spec(spec: ParsedTaskSpec) -> Result<TaskRunResult, TaskRunnerError> {
    match spec {
        ParsedTaskSpec::Ivp(spec) => run_ivp_task(spec)
            .map(TaskRunResult::Ivp)
            .map_err(TaskRunnerError::Ivp),
        ParsedTaskSpec::Bvp(spec) => run_bvp_task(spec)
            .map(TaskRunResult::Bvp)
            .map_err(TaskRunnerError::Bvp),
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
    parse_task_spec_from_str(&text)
}

pub fn run_task_from_file(path: impl AsRef<Path>) -> Result<TaskRunResult, TaskRunnerError> {
    let path_ref = path.as_ref();
    let text = fs::read_to_string(path_ref).map_err(|source| TaskRunnerError::Io {
        path: path_ref.to_path_buf(),
        source,
    })?;
    run_task_from_str(&text)
}

pub fn render_task_preview(spec: &ParsedTaskSpec) -> String {
    match spec {
        ParsedTaskSpec::Ivp(ivp) => render_ivp_preview(ivp),
        ParsedTaskSpec::Bvp(bvp) => render_bvp_preview(bvp),
    }
}

pub fn render_task_check(spec: &ParsedTaskSpec) -> String {
    match spec {
        ParsedTaskSpec::Ivp(ivp) => render_ivp_check(ivp),
        ParsedTaskSpec::Bvp(bvp) => render_bvp_check(bvp),
    }
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
        assert_eq!(
            detect_task_kind_from_str(ivp).unwrap(),
            TaskDocumentKind::Ivp
        );
        assert_eq!(
            detect_task_kind_from_str(bvp).unwrap(),
            TaskDocumentKind::Bvp
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
