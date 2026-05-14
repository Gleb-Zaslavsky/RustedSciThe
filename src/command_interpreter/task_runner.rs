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
    IvpMethodSpec, IvpTaskError, IvpTaskRunResult, IvpTaskSpec, parse_ivp_task_from_str,
    run_ivp_task,
};
use std::fs;
use std::path::{Path, PathBuf};

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
            "bvp" => Ok(TaskDocumentKind::Bvp),
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

fn render_ivp_preview(spec: &IvpTaskSpec) -> String {
    let method = ivp_method_name(&spec.solver.method);
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
    out.push_str(&format!(
        "solver=BVP, strategy={:?}, backend={:?}, arg={}, t0={}, t_end={}, n_steps={}\n",
        spec.solver.strategy,
        spec.solver.backend,
        spec.equations.arg,
        spec.mesh.t0,
        spec.mesh.t_end,
        spec.mesh.n_steps
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

fn ivp_method_name(method: &IvpMethodSpec) -> &'static str {
    match method {
        IvpMethodSpec::NonStiff(_) => "NonStiff",
        IvpMethodSpec::Radau3 => "Radau3",
        IvpMethodSpec::Radau5 => "Radau5",
        IvpMethodSpec::Bdf => "BDF",
        IvpMethodSpec::BackwardEuler => "BackwardEuler",
        IvpMethodSpec::Lsode2 => "LSODE2",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

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
}
