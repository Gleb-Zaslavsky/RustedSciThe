//! BVP task-shell built on top of the generic [`DocumentParser`].
//!
//! This module mirrors the IVP task shell, but targets the historical
//! [`crate::numerical::BVP_api::BVP`] facade.
//!
//! The parser supports two usage modes:
//! - full end-to-end task documents, where equations, boundary conditions,
//!   mesh, initial guess, solver selection, solver options, and postprocessing
//!   live in one DSL document
//! - split mode, where the document contributes only solver settings and the
//!   BVP problem itself is assembled from plain Rust data before the solver is
//!   built
//!
//! The goal is intentionally modest: provide a small, typed, text-facing entry
//! point that is easy to validate and easy to extend later.
//!
//! The shell keeps the problem description compact while exposing the generated
//! BVP backend knobs needed by production runs:
//! - symbolic RHS is accepted as text and parsed into [`Expr`]
//! - numeric parameters are substituted before solver construction
//! - boundary conditions are expressed as `<unknown>_left` / `<unknown>_right`
//! - initial guess is described as constant profiles, not arbitrary matrices
//! - `solver_options` can select Lambdify/AOT, Sparse/Banded matrix assembly,
//!   AOT compiler/build policy, symbolic assembly backend, and native banded
//!   linear solver policy
//! - postprocessing is routed through the unified postprocessing facade, while
//!   the historical plot flag remains available for legacy plotters output

use crate::command_interpreter::task_parser::{DocumentMap, DocumentParser, ParseError, Value};
use crate::command_interpreter::task_parser_common::{
    parse_continuation_spec, parse_symbolic_equation_system, ContinuationMode, ContinuationSpec,
    SharedEquationParseError,
};
use crate::numerical::BVP_Damp::generated_solver_handoff::{
    AotBuildPolicy, AotBuildProfile, AotExecutionPolicy, GeneratedBackendConfig,
};
use crate::numerical::BVP_api::BVP;
use crate::numerical::BVP_sci::new::{
    BvpSciAssembly, BvpSciExecution, BvpSciMatrixLayout, BvpSciNewError, BvpSciOptions,
    BvpSciSolver,
};
use crate::somelinalg::banded::{LinearSolverConfig, LinearSolverPolicy};
use crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend;
use crate::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
use crate::symbolic::codegen::codegen_provider_api::MatrixBackend;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions_BVP::BvpBackendIntegrationError;
use crate::symbolic::symbolic_functions_BVP::BvpSymbolicAssemblyBackend;
use crate::symbolic::symbolic_ivp_generated::{
    SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
};
use crate::Utils::postprocessing::{PostprocessDataset, PostprocessError, PostprocessPlan};
use nalgebra::{DMatrix, DVector};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

type GenericSectionMap = HashMap<String, Option<Vec<Value>>>;

/// High-level task kind supported by the BVP task shell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BvpTaskKindSpec {
    Bvp,
}

/// Concrete BVP implementation selected by a task document.
///
/// `BVP` remains the compatibility spelling for the mature damped solver;
/// `BVP_sci` is an explicit route to the new SciPy-like collocation solver.
/// Keeping this distinction in the parsed contract prevents the runner from
/// silently routing one algorithm through another solver's legacy facade.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BvpSolverFamilySpec {
    Damp,
    Sci,
}

/// Symbolic representation used to build residual/Jacobian callbacks.
///
/// This is independent from [`BvpExecutionSpec`]: both representations can
/// be lowered to Lambdify or AOT when that execution route is available.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BvpFrontendSpec {
    ExprLegacy,
    AtomViewNative,
}

impl BvpFrontendSpec {
    fn from_str(raw: &str) -> Result<Self, BvpTaskError> {
        match normalize_token(raw).as_str() {
            "exprlegacy" | "expr_legacy" | "expr" | "legacy" => Ok(Self::ExprLegacy),
            "atomview" | "atomviewnative" | "atom_view" | "atomnative" | "atom_native" | "atom" => {
                Ok(Self::AtomViewNative)
            }
            other => Err(BvpTaskError::UnsupportedRoute(format!(
                "unsupported BVP frontend `{other}`"
            ))),
        }
    }
}

/// Runtime lowering/execution route for symbolic BVP callbacks.
///
/// This is deliberately separate from [`BvpFrontendSpec`]. `ExprLegacy` and
/// `AtomViewNative` describe the symbolic graph representation; `Lambdify` and
/// `Aot` describe how the prepared callbacks execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BvpExecutionSpec {
    Lambdify,
    Aot,
}

impl BvpExecutionSpec {
    fn from_str(raw: &str) -> Result<Self, BvpTaskError> {
        match normalize_token(raw).as_str() {
            "lambdify" | "lambdify_expr" | "lambdifyexpr" => Ok(Self::Lambdify),
            "aot" | "generated" => Ok(Self::Aot),
            "numerical" | "callbacks" => Err(BvpTaskError::UnsupportedRoute(
                "BVP task documents contain symbolic equations; numerical callback execution belongs to the native Rust API".to_string(),
            )),
            other => Err(BvpTaskError::UnsupportedRoute(format!(
                "unsupported BVP execution route `{other}` (use Lambdify or AOT)"
            ))),
        }
    }
}

/// Damped/frozen/naive solver family chosen by the task document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BvpStrategySpec {
    Damped,
    Frozen,
    Naive,
}

impl BvpStrategySpec {
    fn from_str(raw: &str) -> Result<Self, BvpTaskError> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "damped" => Ok(Self::Damped),
            "frozen" => Ok(Self::Frozen),
            "naive" => Ok(Self::Naive),
            other => Err(BvpTaskError::UnknownStrategy(other.to_string())),
        }
    }

    fn as_solver_string(&self) -> String {
        match self {
            Self::Damped => "Damped",
            Self::Frozen => "Frozen",
            Self::Naive => "Naive",
        }
        .to_string()
    }
}

/// Linear matrix structure requested by the task document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BvpLinearBackendSpec {
    Dense,
    Sparse,
    Banded,
}

impl BvpLinearBackendSpec {
    fn from_str(raw: &str) -> Result<Self, BvpTaskError> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "dense" => Ok(Self::Dense),
            "sparse" => Ok(Self::Sparse),
            "banded" => Ok(Self::Banded),
            other => Err(BvpTaskError::UnknownBackend(other.to_string())),
        }
    }

    fn as_solver_string(&self) -> String {
        match self {
            Self::Dense => "Dense",
            Self::Sparse => "Sparse",
            Self::Banded => "Banded",
        }
        .to_string()
    }
}

/// Top-level solver selection extracted from the DSL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BvpSolverSelectionSpec {
    pub task_kind: BvpTaskKindSpec,
    pub family: BvpSolverFamilySpec,
    /// Explicit symbolic representation. `None` preserves the selected
    /// solver/backend preset's frontend default.
    pub frontend: Option<BvpFrontendSpec>,
    /// Explicit execution route, if the task overrides generated-backend
    /// policy. `None` preserves the selected backend preset's policy.
    pub execution: Option<BvpExecutionSpec>,
    pub strategy: BvpStrategySpec,
    pub scheme: String,
    pub backend: BvpLinearBackendSpec,
}

/// Symbolic equation system and parameter declarations parsed from the DSL.
#[derive(Debug, Clone, PartialEq)]
pub struct BvpEquationSpec {
    pub arg: String,
    pub unknowns: Vec<String>,
    pub rhs: Vec<Expr>,
    pub parameter_names: Vec<String>,
    pub parameter_values: HashMap<String, f64>,
}

/// Boundary condition map keyed by unknown name.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryConditionSpec {
    pub conditions: HashMap<String, Vec<(usize, f64)>>,
}

/// Solver mesh description parsed from the task document.
#[derive(Debug, Clone, PartialEq)]
pub struct BvpMeshSpec {
    pub t0: f64,
    pub t_end: f64,
    pub n_steps: usize,
}

/// Constant initial guess values for each unknown.
#[derive(Debug, Clone, PartialEq)]
pub struct BvpInitialGuessSpec {
    pub values: Vec<f64>,
}

/// Scalar solver options and generated-backend knobs.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BvpSolverOptionsSpec {
    pub tolerance: Option<f64>,
    pub max_iterations: Option<usize>,
    pub linear_sys_method: Option<String>,
    pub loglevel: Option<String>,
    pub generated_backend: BvpGeneratedBackendSpec,
}

/// AOT / backend configuration nested inside solver options.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BvpGeneratedBackendSpec {
    pub preset: Option<String>,
    pub matrix_backend: Option<String>,
    pub backend_policy: Option<String>,
    pub symbolic_backend: Option<String>,
    pub aot_codegen_backend: Option<String>,
    pub aot_c_compiler: Option<String>,
    pub aot_build_policy: Option<String>,
    pub aot_build_profile: Option<String>,
    pub aot_compile_preset: Option<String>,
    pub aot_execution_policy: Option<String>,
    pub aot_output_dir: Option<String>,
    pub aot_handoff_path: Option<String>,
    pub banded_linear_solver: Option<String>,
    pub refinement_steps: Option<usize>,
}

/// Optional postprocessing plan collected from the task document.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BvpPostprocessingSpec {
    pub save_csv: bool,
    pub csv_path: Option<String>,
    pub save_txt: bool,
    pub txt_path: Option<String>,
    pub write_report: bool,
    pub report_path: Option<String>,
    pub plotters_png: bool,
    pub plotters_dir: Option<String>,
    pub gnuplot_png: bool,
    pub gnuplot_dir: Option<String>,
    pub terminal_plot: bool,
    /// Explicit replacement for the historical `plot` boolean. The boolean
    /// remains accepted as a compatibility alias for legacy task files.
    pub output_policy: BvpOutputPolicy,
    pub plot: bool,
}

/// Typed plotting/output policy for task documents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BvpOutputPolicy {
    #[default]
    None,
    Plotters,
    Gnuplot,
    Terminal,
}

impl BvpOutputPolicy {
    fn from_str(raw: &str) -> Result<Self, BvpTaskError> {
        match normalize_token(raw).as_str() {
            "none" | "off" | "disabled" => Ok(Self::None),
            "plotters" | "plotter" | "plotterspng" | "plotters_png" => Ok(Self::Plotters),
            "gnuplot" | "gnuplotpng" | "gnuplot_png" => Ok(Self::Gnuplot),
            "terminal" | "terminalplot" | "terminal_plot" => Ok(Self::Terminal),
            other => Err(BvpTaskError::UnsupportedRoute(format!(
                "unsupported BVP output policy `{other}`"
            ))),
        }
    }
}

impl BvpPostprocessingSpec {
    fn to_plan(&self, default_csv_path: &str) -> PostprocessPlan {
        let mut plan = PostprocessPlan::new();
        if self.save_csv {
            plan = plan.save_csv(
                self.csv_path
                    .clone()
                    .unwrap_or_else(|| default_csv_path.to_string()),
            );
        }
        if self.save_txt {
            plan = plan.save_txt(
                self.txt_path
                    .clone()
                    .unwrap_or_else(|| "bvp_result.txt".to_string()),
            );
        }
        if self.write_report {
            plan = plan.write_report(
                self.report_path
                    .clone()
                    .unwrap_or_else(|| "bvp_report.md".to_string()),
            );
        }
        if self.plotters_png {
            plan = plan.plotters_png(
                self.plotters_dir
                    .clone()
                    .unwrap_or_else(|| "bvp_plotters".to_string()),
            );
        }
        if self.gnuplot_png {
            plan = plan.gnuplot_png(
                self.gnuplot_dir
                    .clone()
                    .unwrap_or_else(|| "bvp_gnuplot".to_string()),
            );
        }
        if self.terminal_plot {
            plan = plan.terminal_plot();
        }
        match self.output_policy {
            BvpOutputPolicy::None => {}
            BvpOutputPolicy::Plotters => {
                plan = plan.plotters_png("bvp_plotters");
            }
            BvpOutputPolicy::Gnuplot => {
                plan = plan.gnuplot_png("bvp_gnuplot");
            }
            BvpOutputPolicy::Terminal => {
                plan = plan.terminal_plot();
            }
        }
        plan
    }
}

/// Fully parsed task document: solver selection, problem, options, and output plan.
#[derive(Debug, Clone, PartialEq)]
pub struct BvpTaskSpec {
    pub solver: BvpSolverSelectionSpec,
    pub equations: BvpEquationSpec,
    pub boundary_conditions: BoundaryConditionSpec,
    pub mesh: BvpMeshSpec,
    pub initial_guess: BvpInitialGuessSpec,
    pub solver_options: BvpSolverOptionsSpec,
    pub postprocessing: BvpPostprocessingSpec,
    /// Optional repeated-parameter plan retaining symbolic RHS expressions.
    pub continuation: Option<ContinuationSpec>,
}

/// Problem-only subset of the BVP task DSL.
#[derive(Debug, Clone, PartialEq)]
pub struct BvpProblemSpec {
    pub equations: BvpEquationSpec,
    pub boundary_conditions: BoundaryConditionSpec,
    pub mesh: BvpMeshSpec,
    pub initial_guess: BvpInitialGuessSpec,
}

/// Solver-settings-only subset of the BVP task DSL.
#[derive(Debug, Clone, PartialEq)]
pub struct BvpSolverSettingsSpec {
    pub solver: BvpSolverSelectionSpec,
    pub solver_options: BvpSolverOptionsSpec,
}

impl BvpTaskSpec {
    /// Extract only the mathematical problem data from a full task document.
    pub fn problem_spec(&self) -> BvpProblemSpec {
        BvpProblemSpec {
            equations: self.equations.clone(),
            boundary_conditions: self.boundary_conditions.clone(),
            mesh: self.mesh.clone(),
            initial_guess: self.initial_guess.clone(),
        }
    }

    /// Extract only the solver-selection and solver-option settings from a full task document.
    pub fn solver_settings_spec(&self) -> BvpSolverSettingsSpec {
        BvpSolverSettingsSpec {
            solver: self.solver.clone(),
            solver_options: self.solver_options.clone(),
        }
    }

    /// Extract only the postprocessing subset from a full task document.
    pub fn postprocessing_spec(&self) -> BvpPostprocessingSpec {
        self.postprocessing.clone()
    }
}

/// Parsed task plus the optional solver output matrix.
#[derive(Debug)]
pub struct BvpTaskRunResult {
    pub specification: BvpTaskSpec,
    pub result: Option<DMatrix<f64>>,
    /// One final matrix per executed continuation segment. The last entry is
    /// also exposed through `result` for compatibility with existing callers.
    pub continuation_segments: Vec<BvpTaskSegmentResult>,
    /// Execution-level facts needed to distinguish fresh and prepared task
    /// runs without scraping solver logs.
    pub continuation_telemetry: BvpTaskContinuationTelemetry,
}

/// Compact continuation lifecycle telemetry owned by the task runner.
#[derive(Debug, Clone, Default)]
pub struct BvpTaskContinuationTelemetry {
    pub segments: usize,
    pub fresh_preparations: usize,
    pub prepared_reuses: usize,
    pub mesh_restarts: usize,
    pub initial_guess_restarts: usize,
    pub peak_result_elements: usize,
}

/// Compact result of one task-document continuation segment.
#[derive(Debug)]
pub struct BvpTaskSegmentResult {
    pub index: usize,
    pub parameters: Vec<f64>,
    pub result: DMatrix<f64>,
}

/// Parser/build error for BVP task documents.
#[derive(Debug)]
pub enum BvpTaskError {
    Parser(String),
    Io {
        path: PathBuf,
        source: io::Error,
    },
    Document(ParseError),
    MissingSection(&'static str),
    MissingField {
        section: String,
        field: String,
    },
    InvalidField {
        section: String,
        field: String,
        message: String,
    },
    InconsistentEquationCounts {
        unknowns: usize,
        rhs: usize,
    },
    UnknownStrategy(String),
    UnknownBackend(String),
    UnsupportedRoute(String),
    UnsupportedContinuation(String),
    BvpDamp(BvpBackendIntegrationError),
    BvpSci(BvpSciNewError),
    InvalidConfiguration {
        field: String,
        message: String,
    },
    /// Semantic validation discovered after the generic document map was
    /// built, with a best-effort source location recovered from the task text.
    SymbolDiagnostic {
        message: String,
        token: String,
        line: Option<usize>,
        column: Option<usize>,
    },
    Postprocess(PostprocessError),
    Semantic(String),
    Solver(String),
}

impl BvpTaskError {
    /// Stable category used by batch reports and task-runner summaries.
    pub fn category(&self) -> &'static str {
        match self {
            Self::Parser(_) | Self::Document(_) => "parse",
            Self::Io { .. } => "io",
            Self::MissingSection(_)
            | Self::MissingField { .. }
            | Self::InvalidField { .. }
            | Self::InconsistentEquationCounts { .. }
            | Self::UnknownStrategy(_)
            | Self::UnknownBackend(_)
            | Self::InvalidConfiguration { .. }
            | Self::SymbolDiagnostic { .. }
            | Self::Semantic(_) => "configuration",
            Self::UnsupportedRoute(_) => "unsupported_route",
            Self::UnsupportedContinuation(_) => "continuation",
            Self::BvpSci(_) | Self::BvpDamp(_) | Self::Solver(_) => "solver",
            Self::Postprocess(_) => "postprocess",
        }
    }
}

impl std::fmt::Display for BvpTaskError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parser(msg) => write!(f, "parser error: {msg}"),
            Self::Io { path, source } => {
                write!(f, "failed to read BVP task `{}`: {source}", path.display())
            }
            Self::Document(error) => write!(f, "document parser error: {error}"),
            Self::MissingSection(section) => write!(f, "missing section `{section}`"),
            Self::MissingField { section, field } => {
                write!(f, "missing field `{field}` in section `{section}`")
            }
            Self::InvalidField {
                section,
                field,
                message,
            } => write!(
                f,
                "invalid field `{field}` in section `{section}`: {message}"
            ),
            Self::InconsistentEquationCounts { unknowns, rhs } => write!(
                f,
                "number of unknowns ({unknowns}) does not match number of rhs expressions ({rhs})"
            ),
            Self::UnknownStrategy(strategy) => write!(f, "unknown BVP strategy `{strategy}`"),
            Self::UnknownBackend(method) => write!(f, "unknown BVP backend `{method}`"),
            Self::UnsupportedRoute(message) => write!(f, "unsupported BVP route: {message}"),
            Self::UnsupportedContinuation(message) => {
                write!(f, "unsupported BVP continuation: {message}")
            }
            Self::BvpSci(error) => write!(f, "BVP_sci failed: {error}"),
            Self::BvpDamp(error) => write!(f, "BVP_Damp failed: {error}"),
            Self::InvalidConfiguration { field, message } => {
                write!(f, "invalid BVP configuration `{field}`: {message}")
            }
            Self::SymbolDiagnostic {
                message,
                token,
                line,
                column,
            } => {
                write!(f, "BVP diagnostic for `{token}`: {message}")?;
                if let (Some(line), Some(column)) = (line, column) {
                    write!(f, " at line {line}, column {column}")?;
                }
                Ok(())
            }
            Self::Postprocess(error) => write!(f, "postprocessing error: {error}"),
            Self::Semantic(message) => write!(f, "{message}"),
            Self::Solver(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for BvpTaskError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Document(error) => Some(error),
            Self::BvpSci(error) => Some(error),
            Self::BvpDamp(error) => Some(error),
            Self::Postprocess(error) => Some(error),
            _ => None,
        }
    }
}

/// Parse a full BVP task document from DSL text.
pub fn parse_bvp_task_from_str(input: &str) -> Result<BvpTaskSpec, BvpTaskError> {
    let mut parser = DocumentParser::new(input.to_string());
    let pseudonyms = default_bvp_pseudonyms();
    parser
        .try_with_pseudonims_typed(Some(pseudonyms.0), Some(pseudonyms.1))
        .map_err(BvpTaskError::Document)?;
    parser
        .parse_document_typed()
        .map_err(BvpTaskError::Document)?;
    parser
        .try_keys_to_lower_case_typed(Some(vec![
            "equations".to_string(),
            "parameters".to_string(),
            "boundary_conditions".to_string(),
            "initial_guess".to_string(),
            "where".to_string(),
            "substitute".to_string(),
        ]))
        .map_err(BvpTaskError::Document)?;
    let document = parser
        .get_result()
        .ok_or_else(|| BvpTaskError::Parser("document parser returned no result".to_string()))?;
    parse_bvp_task_from_document(document).map_err(|error| attach_symbol_position(error, input))
}

fn attach_symbol_position(error: BvpTaskError, input: &str) -> BvpTaskError {
    let (message, token) = match error {
        BvpTaskError::Semantic(message) => {
            let token = diagnostic_token(&message);
            (message, token)
        }
        BvpTaskError::InvalidField {
            section,
            field,
            message,
        } => {
            let token = if field == "*" {
                diagnostic_token(&message)
            } else {
                field.clone()
            };
            (
                format!("invalid field `{field}` in section `{section}`: {message}"),
                token,
            )
        }
        other => return other,
    };
    let (line, column) = source_position(input, &token)
        .map(|(line, column)| (Some(line), Some(column)))
        .unwrap_or((None, None));
    BvpTaskError::SymbolDiagnostic {
        message,
        token,
        line,
        column,
    }
}

fn diagnostic_token(message: &str) -> String {
    if let Some((_, remainder)) = message.split_once("symbol(s):") {
        return remainder
            .split(|ch: char| ch == ',' || ch.is_whitespace())
            .find(|token| !token.is_empty())
            .unwrap_or("<unknown>")
            .trim_matches('`')
            .to_string();
    }
    message
        .split('`')
        .nth(1)
        .unwrap_or("equations")
        .to_string()
}

fn source_position(input: &str, token: &str) -> Option<(usize, usize)> {
    let offset = input.find(token)?;
    let before = &input[..offset];
    let line = before.lines().count().max(1);
    let column = before
        .rsplit('\n')
        .next()
        .map(|line| line.chars().count() + 1)
        .unwrap_or(1);
    Some((line, column))
}

/// Parse a full BVP task document from an on-disk file.
pub fn parse_bvp_task_from_file(path: Option<PathBuf>) -> Result<BvpTaskSpec, BvpTaskError> {
    let path = match path {
        Some(path) => path,
        None => find_default_bvp_task_file()?,
    };
    let input = fs::read_to_string(&path).map_err(|source| BvpTaskError::Io {
        path: path.clone(),
        source,
    })?;
    parse_bvp_task_from_str(&input)
}

fn find_default_bvp_task_file() -> Result<PathBuf, BvpTaskError> {
    let current_dir = std::env::current_dir().map_err(|source| BvpTaskError::Io {
        path: PathBuf::from("."),
        source,
    })?;
    let entries = fs::read_dir(&current_dir).map_err(|source| BvpTaskError::Io {
        path: current_dir.clone(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| BvpTaskError::Io {
            path: current_dir.clone(),
            source,
        })?;
        let name = entry.file_name();
        if name.to_string_lossy().starts_with("problem") && name.to_string_lossy().ends_with(".txt")
        {
            return Ok(entry.path());
        }
    }
    Err(BvpTaskError::Io {
        path: current_dir,
        source: io::Error::new(
            io::ErrorKind::NotFound,
            "no task file starting with `problem` and ending with `.txt` was found",
        ),
    })
}

/// Fallible alias for [`parse_bvp_problem_from_document`].
pub fn try_parse_bvp_problem_from_document(
    document: &DocumentMap,
) -> Result<BvpProblemSpec, BvpTaskError> {
    parse_bvp_problem_from_document(document)
}

/// Fallible alias for [`parse_bvp_solver_settings_from_document`].
pub fn try_parse_bvp_solver_settings_from_document(
    document: &DocumentMap,
) -> Result<BvpSolverSettingsSpec, BvpTaskError> {
    parse_bvp_solver_settings_from_document(document)
}

/// Fallible alias for [`parse_bvp_task_from_document`].
pub fn try_parse_bvp_task_from_document(
    document: &DocumentMap,
) -> Result<BvpTaskSpec, BvpTaskError> {
    parse_bvp_task_from_document(document)
}

/// Fallible alias for [`parse_bvp_task_from_str`].
pub fn try_parse_bvp_task_from_str(input: &str) -> Result<BvpTaskSpec, BvpTaskError> {
    parse_bvp_task_from_str(input)
}

/// Fallible alias for [`parse_bvp_task_from_file`].
pub fn try_parse_bvp_task_from_file(path: Option<PathBuf>) -> Result<BvpTaskSpec, BvpTaskError> {
    parse_bvp_task_from_file(path)
}

/// Parse only the equation/BC/mesh/initial-guess part of the DSL.
pub fn parse_bvp_problem_from_document(
    document: &DocumentMap,
) -> Result<BvpProblemSpec, BvpTaskError> {
    let equations = parse_bvp_equations(document)?;
    let boundary_conditions = parse_boundary_conditions(document, &equations.unknowns)?;
    let mesh = parse_bvp_mesh(document)?;
    let initial_guess = parse_bvp_initial_guess(document, &equations.unknowns)?;

    Ok(BvpProblemSpec {
        equations,
        boundary_conditions,
        mesh,
        initial_guess,
    })
}

/// Parse only the solver-selection and solver-option part of the DSL.
pub fn parse_bvp_solver_settings_from_document(
    document: &DocumentMap,
) -> Result<BvpSolverSettingsSpec, BvpTaskError> {
    Ok(BvpSolverSettingsSpec {
        solver: parse_bvp_solver_selection(document)?,
        solver_options: parse_bvp_solver_options(document)?,
    })
}

/// Parse a full BVP task from an already parsed document map.
pub fn parse_bvp_task_from_document(document: &DocumentMap) -> Result<BvpTaskSpec, BvpTaskError> {
    let problem = parse_bvp_problem_from_document(document)?;
    let solver_settings = parse_bvp_solver_settings_from_document(document)?;
    let postprocessing = parse_bvp_postprocessing(document)?;
    let continuation = if document.contains_key("continuation") {
        let parsed = parse_symbolic_equation_system(document, "x").map_err(map_equation_error)?;
        parse_continuation_spec(document, parsed.symbolic_rhs, &parsed.parameter_names)
            .map_err(map_equation_error)?
    } else {
        None
    };

    let spec = BvpTaskSpec {
        solver: solver_settings.solver,
        equations: problem.equations,
        boundary_conditions: problem.boundary_conditions,
        mesh: problem.mesh,
        initial_guess: problem.initial_guess,
        solver_options: solver_settings.solver_options,
        postprocessing,
        continuation,
    };
    validate_bvp_task_spec(&spec)?;
    Ok(spec)
}

/// Build a [`BVP`] solver from a full task specification.
pub fn build_bvp_solver_from_spec(spec: &BvpTaskSpec) -> Result<BVP, BvpTaskError> {
    validate_bvp_task_spec(spec)?;
    if spec.solver.family != BvpSolverFamilySpec::Damp {
        return Err(BvpTaskError::UnsupportedRoute(
            "build_bvp_solver_from_spec is only for BVP_Damp; use run_bvp_task for BVP_sci"
                .to_string(),
        ));
    }
    build_bvp_solver_from_problem_and_settings(&spec.problem_spec(), &spec.solver_settings_spec())
}

/// Build a [`BVP`] solver from the Rust-side problem spec plus task-doc solver settings.
pub fn build_bvp_solver_from_problem_and_settings(
    problem: &BvpProblemSpec,
    settings: &BvpSolverSettingsSpec,
) -> Result<BVP, BvpTaskError> {
    validate_bvp_problem_spec(problem)?;
    validate_bvp_solver_options(&settings.solver_options)?;
    if settings.solver.family != BvpSolverFamilySpec::Damp {
        return Err(BvpTaskError::UnsupportedRoute(
            "the legacy BVP facade cannot build BVP_sci settings".to_string(),
        ));
    }
    build_bvp_solver_core(
        &problem.equations,
        &problem.boundary_conditions,
        &problem.mesh,
        &problem.initial_guess,
        &settings.solver,
        &settings.solver_options,
    )
}

fn build_bvp_solver_core(
    equations: &BvpEquationSpec,
    boundary_conditions: &BoundaryConditionSpec,
    mesh: &BvpMeshSpec,
    initial_guess_spec: &BvpInitialGuessSpec,
    solver_selection: &BvpSolverSelectionSpec,
    solver_options: &BvpSolverOptionsSpec,
) -> Result<BVP, BvpTaskError> {
    let dimension = equations.unknowns.len();
    if initial_guess_spec.values.len() != dimension {
        return Err(BvpTaskError::Semantic(format!(
            "initial guess dimension {} does not match number of unknowns {}",
            initial_guess_spec.values.len(),
            dimension
        )));
    }

    let initial_guess = DMatrix::from_fn(dimension, mesh.n_steps, |row, _col| {
        initial_guess_spec.values[row]
    });

    let tolerance = solver_options.tolerance.unwrap_or(1e-5);
    let max_iterations = solver_options.max_iterations.unwrap_or(50);
    let (strategy_params, rel_tolerance, bounds) = match solver_selection.strategy {
        BvpStrategySpec::Damped => {
            let rel_tolerance = Some(HashMap::from_iter(
                equations
                    .unknowns
                    .iter()
                    .cloned()
                    .map(|name| (name, 1e-4_f64)),
            ));
            let bounds = Some(HashMap::from_iter(
                equations
                    .unknowns
                    .iter()
                    .cloned()
                    .map(|name| (name, (-1.0e6_f64, 1.0e6_f64))),
            ));
            (
                Some(HashMap::from([
                    ("max_jac".to_string(), None),
                    ("maxDampIter".to_string(), None),
                    ("DampFacor".to_string(), None),
                    ("adaptive".to_string(), None),
                ])),
                rel_tolerance,
                bounds,
            )
        }
        BvpStrategySpec::Frozen | BvpStrategySpec::Naive => (None, None, None),
    };

    let mut bvp = BVP::new(
        equations.rhs.clone(),
        initial_guess,
        equations.unknowns.clone(),
        equations.arg.clone(),
        boundary_conditions.conditions.clone(),
        mesh.t0,
        mesh.t_end,
        mesh.n_steps,
        solver_selection.scheme.clone(),
        solver_selection.strategy.as_solver_string(),
        strategy_params,
        solver_options.linear_sys_method.clone(),
        solver_selection.backend.as_solver_string(),
        tolerance,
        max_iterations,
        rel_tolerance,
        bounds,
        solver_options.loglevel.clone(),
    );

    if !equations.parameter_names.is_empty() {
        let parameter_values = equations
            .parameter_names
            .iter()
            .map(|name| {
                equations
                    .parameter_values
                    .get(name)
                    .copied()
                    .ok_or_else(|| BvpTaskError::InvalidConfiguration {
                        field: format!("parameters.{name}"),
                        message: "missing initial numeric value".to_string(),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        bvp.try_set_damped_parameter_binding(&equations.parameter_names, parameter_values)
            .map_err(BvpTaskError::BvpDamp)?;
    }

    let mut generated_config = generated_backend_config_from_spec(
        &solver_options.generated_backend,
        &solver_selection.backend,
    )?;
    if let Some(frontend) = solver_selection.frontend {
        let symbolic_backend = match frontend {
            BvpFrontendSpec::ExprLegacy => BvpSymbolicAssemblyBackend::ExprLegacy,
            BvpFrontendSpec::AtomViewNative => BvpSymbolicAssemblyBackend::AtomView,
        };
        generated_config = generated_config.with_symbolic_assembly_backend(symbolic_backend);
    }
    if let Some(execution) = solver_selection.execution {
        let policy = match execution {
            BvpExecutionSpec::Lambdify => BackendSelectionPolicy::LambdifyOnly,
            BvpExecutionSpec::Aot => BackendSelectionPolicy::AotOnly,
        };
        generated_config = generated_config.with_backend_policy_override(Some(policy));
    }
    if matches!(
        generated_config.backend_policy_override,
        Some(
            BackendSelectionPolicy::NumericOnly
                | BackendSelectionPolicy::PreferAotThenNumeric
                | BackendSelectionPolicy::PreferLambdifyThenNumeric
        )
    ) {
        return Err(BvpTaskError::Semantic(
            "BVP task documents are symbolic inputs and cannot provide Rust numeric_rhs closures; \
             use lambdify/AOT backend policies in task docs, or the damped NRBVP Rust API with \
             set_numeric_rhs/with_numeric_rhs for pure numeric BVP solving"
                .to_string(),
        ));
    }
    if let Some(structure_damp) = &mut bvp.structure_damp {
        structure_damp.set_generated_backend_config(generated_config.clone());
    }
    if let Some(structure) = &mut bvp.structure {
        structure.set_generated_backend_config(generated_config);
    }

    Ok(bvp)
}

/// Parse and run a full BVP task document in one step.
pub fn run_bvp_task_from_str(input: &str) -> Result<BvpTaskRunResult, BvpTaskError> {
    let spec = parse_bvp_task_from_str(input)?;
    run_bvp_task(spec)
}

/// Execute a fully parsed BVP task and apply any requested postprocessing.
pub fn run_bvp_task(spec: BvpTaskSpec) -> Result<BvpTaskRunResult, BvpTaskError> {
    validate_bvp_task_spec(&spec)?;
    match spec.solver.family {
        BvpSolverFamilySpec::Damp => run_bvp_damp_task(spec),
        BvpSolverFamilySpec::Sci => run_bvp_sci_task(spec),
    }
}

fn run_bvp_damp_task(spec: BvpTaskSpec) -> Result<BvpTaskRunResult, BvpTaskError> {
    if matches!(
        spec.continuation
            .as_ref()
            .map(|continuation| continuation.mode),
        Some(ContinuationMode::Warm | ContinuationMode::Prepared)
    ) {
        return run_bvp_damp_reused_continuation(spec);
    }
    let segments = continuation_rows(&spec)?;
    let mut executed = Vec::with_capacity(segments.len());
    let mut continuation_telemetry = BvpTaskContinuationTelemetry::default();
    for (index, values) in segments.iter().enumerate() {
        let mut segment_spec = bind_continuation_segment(&spec, values)?;
        // Task execution is batch-friendly by default. The historical BVP
        // facade otherwise installs a terminal logger whenever `loglevel` is
        // absent, flooding callers with solver internals. An explicit task
        // `loglevel` still overrides this quiet default.
        if segment_spec.solver_options.loglevel.is_none() {
            segment_spec.solver_options.loglevel = Some("off".to_string());
        }
        let mut solver = build_bvp_solver_from_spec(&segment_spec)?;
        solver.solve();
        let plan = segment_spec.postprocessing.to_plan("bvp_result.csv");
        if !plan.actions.is_empty() {
            solver
                .execute_postprocessing(&plan)
                .map_err(BvpTaskError::Postprocess)?;
        }
        if segment_spec.postprocessing.plot {
            solver.plot_result();
        }
        let result = solver
            .get_result()
            .ok_or_else(|| BvpTaskError::Solver("BVP_Damp produced no result".to_string()))?;
        continuation_telemetry.segments += 1;
        continuation_telemetry.fresh_preparations += 1;
        continuation_telemetry.peak_result_elements = continuation_telemetry
            .peak_result_elements
            .max(result.len());
        executed.push(BvpTaskSegmentResult {
            index,
            parameters: values.clone(),
            result,
        });
    }
    let result = executed.last().map(|segment| segment.result.clone());
    Ok(BvpTaskRunResult {
        specification: spec,
        result,
        continuation_segments: executed,
        continuation_telemetry,
    })
}

/// Execute a BVP_Damp continuation while retaining its prepared symbolic and
/// generated callback bundle. Numeric parameter rebinding invalidates only
/// numeric Jacobian/factor state; a mesh change explicitly re-prepares layout.
fn run_bvp_damp_reused_continuation(spec: BvpTaskSpec) -> Result<BvpTaskRunResult, BvpTaskError> {
    let continuation =
        spec.continuation
            .as_ref()
            .ok_or_else(|| BvpTaskError::InvalidConfiguration {
                field: "continuation.mode".to_string(),
                message: "warm/prepared mode requires a continuation section".to_string(),
            })?;
    let rows = continuation.value_grid.clone();
    if rows.is_empty() {
        return Err(BvpTaskError::InvalidConfiguration {
            field: "continuation.value_grid".to_string(),
            message: "must contain at least one row".to_string(),
        });
    }

    // Keep the symbolic parameter schema intact. Unlike fresh mode this is
    // intentionally not substituted into new expressions per segment.
    let mut prepared_spec = spec.clone();
    prepared_spec.continuation = None;
    let mut solver = build_bvp_solver_from_spec(&prepared_spec)?;
    let mut executed = Vec::with_capacity(rows.len());
    let mut telemetry = BvpTaskContinuationTelemetry::default();
    let mut prepared = false;
    let mut previous_result: Option<DMatrix<f64>> = None;

    for (index, row) in rows.iter().enumerate() {
        let parameters = bvp_parameter_values(&spec, continuation, row)?;
        solver
            .try_set_damped_parameter_binding(&spec.equations.parameter_names, parameters)
            .map_err(BvpTaskError::BvpDamp)?;

        let restart = continuation_needs_restart(continuation, index);
        let mut mesh_changed = false;
        if restart {
            let t0 = continuation
                .t0_values
                .as_ref()
                .and_then(|values| values.get(if values.len() == 1 { 0 } else { index }))
                .copied()
                .unwrap_or(spec.mesh.t0);
            let t_end = continuation
                .t_end_values
                .as_ref()
                .and_then(|values| values.get(if values.len() == 1 { 0 } else { index }))
                .copied()
                .unwrap_or(spec.mesh.t_end);
            mesh_changed = t0 != spec.mesh.t0
                || t_end != spec.mesh.t_end
                || continuation.t0_values.is_some()
                || continuation.t_end_values.is_some();
            if mesh_changed {
                solver
                    .try_set_damped_mesh(t0, t_end, spec.mesh.n_steps)
                    .map_err(BvpTaskError::BvpDamp)?;
                telemetry.mesh_restarts += 1;
            }

            let initial_guess = if let Some(values) = continuation
                .y0_values
                .as_ref()
                .and_then(|values| values.get(if values.len() == 1 { 0 } else { index }))
            {
                DMatrix::from_fn(
                    spec.equations.unknowns.len(),
                    spec.mesh.n_steps,
                    |row, _| values[row],
                )
            } else if let Some(previous) = previous_result.as_ref() {
                damped_initial_guess_from_result(
                    previous,
                    spec.equations.unknowns.len(),
                    spec.mesh.n_steps,
                )
            } else {
                DMatrix::from_fn(
                    spec.equations.unknowns.len(),
                    spec.mesh.n_steps,
                    |row, _| spec.initial_guess.values[row],
                )
            };
            if mesh_changed {
                solver
                    .try_set_damped_initial_guess(initial_guess)
                    .map_err(BvpTaskError::BvpDamp)?;
            } else {
                solver
                    .try_set_damped_prepared_iterate(initial_guess)
                    .map_err(BvpTaskError::BvpDamp)?;
            }
            // The first segment consumes the document's initial guess; it is
            // preparation, not a continuation restart. Count only later
            // segment-level resets here.
            if index > 0 {
                telemetry.initial_guess_restarts += 1;
            }
        } else if let Some(previous) = previous_result.as_ref() {
            solver
                .try_set_damped_prepared_iterate(damped_initial_guess_from_result(
                    previous,
                    spec.equations.unknowns.len(),
                    spec.mesh.n_steps,
                ))
                .map_err(BvpTaskError::BvpDamp)?;
            telemetry.initial_guess_restarts += 1;
        }

        if mesh_changed {
            solver.try_prepare_damped().map_err(BvpTaskError::BvpDamp)?;
            prepared = true;
        }
        solver
            .try_solve_damped(prepared)
            .map_err(BvpTaskError::BvpDamp)?;
        prepared = true;
        telemetry.segments += 1;
        if index > 0 {
            telemetry.prepared_reuses += 1;
        } else {
            telemetry.fresh_preparations += 1;
        }

        let result = solver
            .get_result()
            .ok_or_else(|| BvpTaskError::Solver("BVP_Damp produced no result".to_string()))?;
        telemetry.peak_result_elements = telemetry.peak_result_elements.max(result.len());
        let plan = spec.postprocessing.to_plan("bvp_result.csv");
        if !plan.actions.is_empty() {
            solver
                .execute_postprocessing(&plan)
                .map_err(BvpTaskError::Postprocess)?;
        }
        if spec.postprocessing.plot {
            solver.plot_result();
        }
        previous_result = Some(result.clone());
        executed.push(BvpTaskSegmentResult {
            index,
            parameters: row.clone(),
            result,
        });
    }

    let result = executed.last().map(|segment| segment.result.clone());
    Ok(BvpTaskRunResult {
        specification: spec,
        result,
        continuation_segments: executed,
        continuation_telemetry: telemetry,
    })
}

fn validate_bvp_task_spec(spec: &BvpTaskSpec) -> Result<(), BvpTaskError> {
    validate_bvp_problem_spec(&spec.problem_spec())?;
    validate_bvp_solver_options(&spec.solver_options)?;

    for (name, value) in &spec.equations.parameter_values {
        if !value.is_finite() {
            return Err(BvpTaskError::InvalidConfiguration {
                field: format!("parameters.{name}"),
                message: "must be finite".to_string(),
            });
        }
    }
    for name in &spec.equations.parameter_names {
        if !spec.equations.parameter_values.contains_key(name) {
            return Err(BvpTaskError::InvalidConfiguration {
                field: format!("parameters.{name}"),
                message: "declared BVP parameter requires an initial numeric value".to_string(),
            });
        }
    }

    if let Some(continuation) = &spec.continuation {
        if continuation.parameters.is_empty() || continuation.value_grid.is_empty() {
            return Err(BvpTaskError::InvalidConfiguration {
                field: "continuation.value_grid".to_string(),
                message: "must contain at least one parameter row".to_string(),
            });
        }
        for parameter in &continuation.parameters {
            if !spec
                .equations
                .parameter_names
                .iter()
                .any(|declared| declared == parameter)
            {
                return Err(BvpTaskError::InvalidConfiguration {
                    field: format!("continuation.parameters.{parameter}"),
                    message: "parameter is not declared in equations".to_string(),
                });
            }
        }
        for (row, values) in continuation.value_grid.iter().enumerate() {
            if values.len() != continuation.parameters.len() {
                return Err(BvpTaskError::InvalidConfiguration {
                    field: format!("continuation.value_grid[{row}]"),
                    message: format!(
                        "expected {} values, got {}",
                        continuation.parameters.len(),
                        values.len()
                    ),
                });
            }
            if values.iter().any(|value| !value.is_finite()) {
                return Err(BvpTaskError::InvalidConfiguration {
                    field: format!("continuation.value_grid[{row}]"),
                    message: "all values must be finite".to_string(),
                });
            }
        }
        if let Some(y0_values) = &continuation.y0_values {
            for (row, values) in y0_values.iter().enumerate() {
                if values.len() != spec.equations.unknowns.len() {
                    return Err(BvpTaskError::InvalidConfiguration {
                        field: format!("continuation.y0_values[{row}]"),
                        message: format!(
                            "expected {} values, got {}",
                            spec.equations.unknowns.len(),
                            values.len()
                        ),
                    });
                }
                if values.iter().any(|value| !value.is_finite()) {
                    return Err(BvpTaskError::InvalidConfiguration {
                        field: format!("continuation.y0_values[{row}]"),
                        message: "all values must be finite".to_string(),
                    });
                }
            }
        }
        for (field, values) in [
            ("continuation.t0_values", continuation.t0_values.as_ref()),
            (
                "continuation.t_end_values",
                continuation.t_end_values.as_ref(),
            ),
        ] {
            if let Some(values) = values {
                if values.len() != continuation.value_grid.len() {
                    return Err(BvpTaskError::InvalidConfiguration {
                        field: field.to_string(),
                        message: format!(
                            "expected {} values, got {}",
                            continuation.value_grid.len(),
                            values.len()
                        ),
                    });
                }
                if values.iter().any(|value| !value.is_finite()) {
                    return Err(BvpTaskError::InvalidConfiguration {
                        field: field.to_string(),
                        message: "all values must be finite".to_string(),
                    });
                }
            }
        }
    }
    Ok(())
}

fn validate_bvp_problem_spec(problem: &BvpProblemSpec) -> Result<(), BvpTaskError> {
    let mesh = &problem.mesh;
    if !mesh.t0.is_finite() {
        return Err(BvpTaskError::InvalidConfiguration {
            field: "mesh.t0".to_string(),
            message: "must be finite".to_string(),
        });
    }
    if !mesh.t_end.is_finite() {
        return Err(BvpTaskError::InvalidConfiguration {
            field: "mesh.t_end".to_string(),
            message: "must be finite".to_string(),
        });
    }
    if mesh.t_end <= mesh.t0 {
        return Err(BvpTaskError::InvalidConfiguration {
            field: "mesh.t_end".to_string(),
            message: "must be greater than mesh.t0".to_string(),
        });
    }
    if mesh.n_steps < 2 {
        return Err(BvpTaskError::InvalidConfiguration {
            field: "mesh.n_steps".to_string(),
            message: "must be at least 2".to_string(),
        });
    }
    if problem.initial_guess.values.len() != problem.equations.unknowns.len() {
        return Err(BvpTaskError::InvalidConfiguration {
            field: "initial_guess".to_string(),
            message: format!(
                "expected {} values, got {}",
                problem.equations.unknowns.len(),
                problem.initial_guess.values.len()
            ),
        });
    }
    if problem
        .initial_guess
        .values
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(BvpTaskError::InvalidConfiguration {
            field: "initial_guess".to_string(),
            message: "all values must be finite".to_string(),
        });
    }
    for (unknown, conditions) in &problem.boundary_conditions.conditions {
        for (_, value) in conditions {
            if !value.is_finite() {
                return Err(BvpTaskError::InvalidConfiguration {
                    field: format!("boundary_conditions.{unknown}"),
                    message: "all values must be finite".to_string(),
                });
            }
        }
    }
    Ok(())
}

fn validate_bvp_solver_options(options: &BvpSolverOptionsSpec) -> Result<(), BvpTaskError> {
    if let Some(tolerance) = options.tolerance {
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(BvpTaskError::InvalidConfiguration {
                field: "solver_options.tolerance".to_string(),
                message: "must be finite and greater than zero".to_string(),
            });
        }
    }
    if matches!(options.max_iterations, Some(0)) {
        return Err(BvpTaskError::InvalidConfiguration {
            field: "solver_options.max_iterations".to_string(),
            message: "must be greater than zero".to_string(),
        });
    }
    Ok(())
}

fn run_bvp_sci_task(spec: BvpTaskSpec) -> Result<BvpTaskRunResult, BvpTaskError> {
    if spec.postprocessing.plot {
        return Err(BvpTaskError::UnsupportedRoute(
            "BVP_sci legacy `plot` is not connected; use the typed postprocessing actions"
                .to_string(),
        ));
    }
    if matches!(
        spec.continuation
            .as_ref()
            .map(|continuation| continuation.mode),
        Some(ContinuationMode::Warm | ContinuationMode::Prepared)
    ) {
        return run_bvp_sci_reused_continuation(spec);
    }
    let segments = continuation_rows(&spec)?;
    let mut executed = Vec::with_capacity(segments.len());
    for (index, values) in segments.iter().enumerate() {
        let segment_spec = bind_continuation_segment(&spec, values)?;
        let mut solver = build_bvp_sci_solver_from_spec(&segment_spec)?;
        let solution = solver.solve().map_err(BvpTaskError::BvpSci)?;
        let plan = segment_spec.postprocessing.to_plan("bvp_result.csv");
        if !plan.actions.is_empty() {
            let dataset = sci_solution_dataset(&solution, &segment_spec.equations.unknowns)?;
            plan.execute(&dataset).map_err(BvpTaskError::Postprocess)?;
        }
        let result = sci_solution_matrix(&solution)?;
        executed.push(BvpTaskSegmentResult {
            index,
            parameters: values.clone(),
            result,
        });
    }
    let result = executed.last().map(|segment| segment.result.clone());
    let continuation_telemetry = BvpTaskContinuationTelemetry {
        segments: executed.len(),
        fresh_preparations: executed.len(),
        peak_result_elements: result.as_ref().map(DMatrix::len).unwrap_or(0),
        ..Default::default()
    };
    Ok(BvpTaskRunResult {
        specification: spec,
        result,
        continuation_segments: executed,
        continuation_telemetry,
    })
}

fn run_bvp_sci_reused_continuation(spec: BvpTaskSpec) -> Result<BvpTaskRunResult, BvpTaskError> {
    let continuation =
        spec.continuation
            .as_ref()
            .ok_or_else(|| BvpTaskError::InvalidConfiguration {
                field: "continuation.mode".to_string(),
                message: "warm/prepared mode requires a continuation section".to_string(),
            })?;
    if spec.equations.parameter_names.is_empty() {
        return Err(BvpTaskError::UnsupportedContinuation(
            "BVP_sci warm/prepared continuation requires declared symbolic parameters".to_string(),
        ));
    }
    let rows = continuation.value_grid.clone();
    if rows.is_empty() {
        return Err(BvpTaskError::InvalidConfiguration {
            field: "continuation.value_grid".to_string(),
            message: "must contain at least one row".to_string(),
        });
    }

    // Retain the symbolic model for the complete series. Only the numeric
    // parameter vector is rebound between solves; no expression parsing or
    // frontend preparation is repeated.
    let mut prepared_spec = spec.clone();
    prepared_spec.equations.rhs = continuation.symbolic_rhs.clone();
    prepared_spec.continuation = None;
    let initial_parameters = bvp_parameter_values(&spec, continuation, &rows[0])?;
    let parameter_targets = Arc::new(Mutex::new(initial_parameters.clone()));
    let mut solver = build_bvp_sci_solver_with_parameter_targets(
        &prepared_spec,
        initial_parameters,
        Arc::clone(&parameter_targets),
    )?;

    let mut executed = Vec::with_capacity(rows.len());
    for (index, values) in rows.iter().enumerate() {
        let parameters = bvp_parameter_values(&spec, continuation, values)?;
        if index > 0 {
            parameter_targets
                .lock()
                .map_err(|_| {
                    BvpTaskError::UnsupportedContinuation(
                        "BVP_sci continuation parameter target lock was poisoned".to_string(),
                    )
                })?
                .clone_from(&parameters);
            solver
                .set_parameters(parameters)
                .map_err(BvpTaskError::BvpSci)?;
        }

        if continuation_needs_restart(continuation, index) {
            let mesh = bvp_segment_mesh(&spec, continuation, index)?;
            let initial_state = bvp_segment_initial_state(&spec, continuation, index, mesh.len());
            solver
                .restart(mesh, initial_state)
                .map_err(BvpTaskError::BvpSci)?;
        }

        let solution = solver.solve().map_err(BvpTaskError::BvpSci)?;
        let plan = spec.postprocessing.to_plan("bvp_result.csv");
        if !plan.actions.is_empty() {
            let dataset = sci_solution_dataset(&solution, &spec.equations.unknowns)?;
            plan.execute(&dataset).map_err(BvpTaskError::Postprocess)?;
        }
        executed.push(BvpTaskSegmentResult {
            index,
            parameters: values.clone(),
            result: sci_solution_matrix(&solution)?,
        });
    }

    let result = executed.last().map(|segment| segment.result.clone());
    let continuation_telemetry = BvpTaskContinuationTelemetry {
        segments: executed.len(),
        fresh_preparations: 1,
        prepared_reuses: executed.len().saturating_sub(1),
        peak_result_elements: result.as_ref().map(DMatrix::len).unwrap_or(0),
        ..Default::default()
    };
    Ok(BvpTaskRunResult {
        specification: spec,
        result,
        continuation_segments: executed,
        continuation_telemetry,
    })
}

fn bvp_parameter_values(
    spec: &BvpTaskSpec,
    continuation: &ContinuationSpec,
    row: &[f64],
) -> Result<Vec<f64>, BvpTaskError> {
    let mut values = spec
        .equations
        .parameter_names
        .iter()
        .map(|name| {
            spec.equations
                .parameter_values
                .get(name)
                .copied()
                .ok_or_else(|| BvpTaskError::InvalidConfiguration {
                    field: format!("parameters.{name}"),
                    message: "missing initial value".to_string(),
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (name, value) in continuation.parameters.iter().zip(row.iter().copied()) {
        let index = spec
            .equations
            .parameter_names
            .iter()
            .position(|candidate| candidate == name)
            .ok_or_else(|| BvpTaskError::InvalidConfiguration {
                field: format!("continuation.parameters.{name}"),
                message: "parameter is not declared in equations".to_string(),
            })?;
        values[index] = value;
    }
    Ok(values)
}

/// Convert BVP_Damp's node-major full result (`nodes x states`) into the
/// state-major Newton initial-guess layout (`states x n_steps`). The full
/// result includes one reconstructed boundary node, while the internal guess
/// intentionally stores only the solver's `n_steps` columns.
fn damped_initial_guess_from_result(
    result: &DMatrix<f64>,
    state_count: usize,
    n_steps: usize,
) -> DMatrix<f64> {
    DMatrix::from_fn(state_count, n_steps, |state, node| {
        result
            .get((node.min(result.nrows().saturating_sub(1)), state))
            .copied()
            .unwrap_or(0.0)
    })
}

fn continuation_needs_restart(continuation: &ContinuationSpec, index: usize) -> bool {
    index == 0
        || !matches!(
            continuation.restart_policy,
            crate::command_interpreter::task_parser_common::ContinuationRestartPolicy::Continue
        )
        || continuation.y0_values.is_some()
        || continuation.t0_values.is_some()
        || continuation.t_end_values.is_some()
}

fn bvp_segment_mesh(
    spec: &BvpTaskSpec,
    continuation: &ContinuationSpec,
    index: usize,
) -> Result<Vec<f64>, BvpTaskError> {
    let t0 = continuation
        .t0_values
        .as_ref()
        .and_then(|values| values.get(if values.len() == 1 { 0 } else { index }))
        .copied()
        .unwrap_or(spec.mesh.t0);
    let t_end = continuation
        .t_end_values
        .as_ref()
        .and_then(|values| values.get(if values.len() == 1 { 0 } else { index }))
        .copied()
        .unwrap_or(spec.mesh.t_end);
    if !t0.is_finite() || !t_end.is_finite() || t_end <= t0 {
        return Err(BvpTaskError::InvalidConfiguration {
            field: format!("continuation.segment[{index}].interval"),
            message: "t_end must be finite and greater than t0".to_string(),
        });
    }
    Ok((0..spec.mesh.n_steps)
        .map(|node| t0 + (t_end - t0) * node as f64 / (spec.mesh.n_steps - 1) as f64)
        .collect())
}

fn bvp_segment_initial_state(
    spec: &BvpTaskSpec,
    continuation: &ContinuationSpec,
    index: usize,
    node_count: usize,
) -> Vec<f64> {
    let values = continuation
        .y0_values
        .as_ref()
        .and_then(|rows| rows.get(if rows.len() == 1 { 0 } else { index }))
        .cloned()
        .unwrap_or_else(|| spec.initial_guess.values.clone());
    (0..node_count)
        .flat_map(|_| values.iter().copied())
        .collect()
}

fn continuation_rows(spec: &BvpTaskSpec) -> Result<Vec<Vec<f64>>, BvpTaskError> {
    let Some(continuation) = &spec.continuation else {
        return Ok(vec![Vec::new()]);
    };
    if !matches!(continuation.mode, ContinuationMode::Fresh) {
        return Err(BvpTaskError::UnsupportedContinuation(
            "BVP task execution currently supports only `mode: fresh`; warm/prepared reuse will be added after native rebind contracts are exposed".to_string(),
        ));
    }
    if continuation.value_grid.is_empty() {
        return Err(BvpTaskError::UnsupportedContinuation(
            "continuation contains no value rows".to_string(),
        ));
    }
    Ok(continuation.value_grid.clone())
}

fn bind_continuation_segment(
    spec: &BvpTaskSpec,
    values: &[f64],
) -> Result<BvpTaskSpec, BvpTaskError> {
    let Some(continuation) = &spec.continuation else {
        return Ok(spec.clone());
    };
    let mut bound = spec.clone();
    bound.equations.rhs = continuation.rhs_for_values(values);
    bound.equations.parameter_names.clear();
    bound.equations.parameter_values.clear();
    bound.continuation = None;
    Ok(bound)
}

fn build_bvp_sci_solver_from_spec(spec: &BvpTaskSpec) -> Result<BvpSciSolver, BvpTaskError> {
    if !spec.equations.parameter_names.is_empty() {
        return Err(BvpTaskError::UnsupportedRoute(
            "BVP_sci task parameters must be expressed through fresh continuation rows; free BVP parameters need explicit parameter boundary equations".to_string(),
        ));
    }
    build_bvp_sci_solver_with_parameter_targets(spec, Vec::new(), Arc::new(Mutex::new(Vec::new())))
}

fn build_bvp_sci_solver_with_parameter_targets(
    spec: &BvpTaskSpec,
    initial_parameters: Vec<f64>,
    parameter_targets: Arc<Mutex<Vec<f64>>>,
) -> Result<BvpSciSolver, BvpTaskError> {
    let dimension = spec.equations.unknowns.len();
    let initial_state = (0..spec.mesh.n_steps)
        .flat_map(|_| spec.initial_guess.values.iter().copied())
        .collect::<Vec<_>>();
    let boundary_conditions =
        sci_boundary_conditions(&spec.boundary_conditions, &spec.equations.unknowns)?;
    let parameter_names = spec.equations.parameter_names.clone();
    if initial_parameters.len() != parameter_names.len() {
        return Err(BvpTaskError::InvalidConfiguration {
            field: "parameters".to_string(),
            message: format!(
                "expected {} initial values, got {}",
                parameter_names.len(),
                initial_parameters.len()
            ),
        });
    }
    let boundary_dimension = dimension + parameter_names.len();
    let execution = match spec.solver.execution.unwrap_or(BvpExecutionSpec::Lambdify) {
        BvpExecutionSpec::Lambdify => BvpSciExecution::Lambdify,
        BvpExecutionSpec::Aot => BvpSciExecution::Aot,
    };
    let options = BvpSciOptions::default()
        .with_execution(execution)
        .with_assembly(
            match spec.solver.frontend.unwrap_or(BvpFrontendSpec::ExprLegacy) {
                BvpFrontendSpec::ExprLegacy => BvpSciAssembly::ExprLegacy,
                BvpFrontendSpec::AtomViewNative => BvpSciAssembly::AtomViewNative,
            },
        )
        .with_matrix_layout(match spec.solver.backend {
            BvpLinearBackendSpec::Dense => BvpSciMatrixLayout::Dense,
            BvpLinearBackendSpec::Sparse => BvpSciMatrixLayout::Sparse,
            BvpLinearBackendSpec::Banded => BvpSciMatrixLayout::Banded { lower: 1, upper: 1 },
        })
        .with_tolerance(spec.solver_options.tolerance.unwrap_or(1e-3))
        .with_limits(
            spec.mesh.n_steps.saturating_mul(16).max(128),
            spec.solver_options.max_iterations.unwrap_or(10),
            10,
        );
    let mut builder =
        BvpSciSolver::builder(spec.equations.rhs.clone(), spec.equations.unknowns.clone())
            .with_independent_variable(spec.equations.arg.clone())
            .with_mesh_and_initial_state(
                (0..spec.mesh.n_steps)
                    .map(|index| {
                        spec.mesh.t0
                            + (spec.mesh.t_end - spec.mesh.t0) * index as f64
                                / (spec.mesh.n_steps - 1) as f64
                    })
                    .collect(),
                initial_state,
            )
            .with_matrix_layout(options.matrix_layout)
            .with_tolerance(options.tolerance)
            .with_limits(
                options.max_nodes,
                options.max_newton_iterations,
                options.max_mesh_refinements,
            )
            .with_parameter_names(parameter_names.clone())
            .with_parameters(initial_parameters);
    builder = match spec.solver.frontend.unwrap_or(BvpFrontendSpec::ExprLegacy) {
        BvpFrontendSpec::ExprLegacy => builder.with_expr_legacy(),
        BvpFrontendSpec::AtomViewNative => builder.with_atom_native(),
    };
    let builder = if execution == BvpSciExecution::Aot {
        builder.with_aot_generated_backend(symbolic_aot_config_from_spec(
            &spec.solver_options.generated_backend,
        )?)
    } else {
        builder
    };
    builder
        .with_boundary_callback(move |ya, yb, parameters, output| {
            if output.len() != boundary_dimension {
                return Err("BVP boundary output has unexpected dimension".to_string());
            }
            for (row, side, value) in &boundary_conditions {
                output[*row] = if *side == 0 {
                    ya[*row] - *value
                } else {
                    yb[*row] - *value
                };
            }
            if !parameter_names.is_empty() {
                let targets = parameter_targets
                    .lock()
                    .map_err(|_| "BVP parameter target lock was poisoned".to_string())?;
                if targets.len() != parameter_names.len()
                    || parameters.len() != parameter_names.len()
                {
                    return Err("BVP parameter target dimension mismatch".to_string());
                }
                for (index, target) in targets.iter().copied().enumerate() {
                    output[dimension + index] = parameters[index] - target;
                }
            }
            Ok(())
        })
        .build()
        .map_err(BvpTaskError::BvpSci)
}

fn sci_boundary_conditions(
    conditions: &BoundaryConditionSpec,
    unknowns: &[String],
) -> Result<Vec<(usize, usize, f64)>, BvpTaskError> {
    unknowns
        .iter()
        .enumerate()
        .map(|(row, name)| {
            let entries = conditions.conditions.get(name).ok_or_else(|| {
                BvpTaskError::Semantic(format!("missing boundary condition for `{name}`"))
            })?;
            if entries.len() != 1 {
                return Err(BvpTaskError::UnsupportedRoute(format!(
                    "BVP_sci task route requires exactly one scalar boundary condition per state; `{name}` has {}",
                    entries.len()
                )));
            }
            let (side, value) = entries[0];
            Ok((row, side, value))
        })
        .collect()
}

fn sci_solution_matrix(
    solution: &crate::numerical::BVP_sci::new::BvpSciSolution,
) -> Result<DMatrix<f64>, BvpTaskError> {
    if solution.x.is_empty() || solution.y.len() % solution.x.len() != 0 {
        return Err(BvpTaskError::Solver(
            "BVP_sci returned an invalid node-major solution".to_string(),
        ));
    }
    let dimension = solution.y.len() / solution.x.len();
    Ok(DMatrix::from_fn(
        dimension,
        solution.x.len(),
        |row, node| solution.y[node * dimension + row],
    ))
}

fn sci_solution_dataset(
    solution: &crate::numerical::BVP_sci::new::BvpSciSolution,
    variable_names: &[String],
) -> Result<PostprocessDataset, BvpTaskError> {
    let matrix = sci_solution_matrix(solution)?;
    let dimension = matrix.nrows();
    if variable_names.len() != dimension {
        return Err(BvpTaskError::Solver(format!(
            "BVP_sci returned {dimension} state components but the task declares {}",
            variable_names.len()
        )));
    }
    let values = DMatrix::from_fn(solution.x.len(), dimension, |node, component| {
        matrix[(component, node)]
    });
    PostprocessDataset::new(
        "x",
        variable_names.to_vec(),
        DVector::from_vec(solution.x.clone()),
        values,
    )
    .map_err(BvpTaskError::Postprocess)
}

/// Write a starter BVP task document template to disk or to the current folder.
pub fn create_bvp_template_file(path: Option<PathBuf>) {
    use std::env;
    use std::fs::File;
    use std::io::Write;

    let template = r#"
task
solver: BVP
strategy: Damped
scheme: forward
method: Sparse

equations
arg: x
unknowns: z, y
rhs: y-z, -z^3

boundary_conditions
z_left: 1.0
y_right: 1.0

mesh
t0: 0.0
t_end: 1.0
n_steps: 20

initial_guess
z: 0.0
y: 0.0

solver_options
tolerance: 1e-5
max_iterations: 20
loglevel: warn

postprocessing
save_csv: false
csv_path: bvp_result.csv
save_txt: false
txt_path: bvp_result.txt
write_report: false
report_path: bvp_report.md
plotters_png: false
plotters_dir: bvp_plotters
gnuplot_png: false
gnuplot_dir: bvp_gnuplot
terminal_plot: false
plot: false
"#;

    let file_path = path.unwrap_or_else(|| {
        let mut default_path = env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
        default_path.push("bvp_task_template.txt");
        default_path
    });

    match File::create(&file_path) {
        Ok(mut file) => {
            if let Err(err) = file.write_all(template.as_bytes()) {
                eprintln!("Failed to write BVP template file: {err}");
            }
        }
        Err(err) => eprintln!("Failed to create BVP template file: {err}"),
    }
}

fn parse_bvp_solver_selection(
    document: &DocumentMap,
) -> Result<BvpSolverSelectionSpec, BvpTaskError> {
    let task_section = get_required_section(document, "task")?;
    let solver_name = get_required_string(task_section, "task", "solver")?;
    let family = match normalize_token(&solver_name).as_str() {
        "bvp" | "bvp_damp" | "bvp_damped" => BvpSolverFamilySpec::Damp,
        "bvp_sci" | "bvpsci" | "bvp_scipy" | "scipy_bvp" => BvpSolverFamilySpec::Sci,
        _ => {
            return Err(BvpTaskError::InvalidField {
                section: "task".to_string(),
                field: "solver".to_string(),
                message: format!("expected `BVP` or `BVP_sci`, got `{solver_name}`"),
            })
        }
    };

    let strategy = BvpStrategySpec::from_str(
        &get_optional_string(task_section, "strategy", "task")?.unwrap_or_else(|| "Damped".into()),
    )?;
    let scheme = get_optional_string(task_section, "scheme", "task")?
        .unwrap_or_else(|| "forward".to_string());
    let backend = BvpLinearBackendSpec::from_str(
        &get_optional_string(task_section, "method", "task")?.unwrap_or_else(|| "Sparse".into()),
    )?;
    let frontend = get_optional_string(task_section, "frontend", "task")?
        .map(|raw| BvpFrontendSpec::from_str(&raw))
        .transpose()?;
    let execution = get_optional_string(task_section, "execution", "task")?
        .map(|raw| BvpExecutionSpec::from_str(&raw))
        .transpose()?;

    Ok(BvpSolverSelectionSpec {
        task_kind: BvpTaskKindSpec::Bvp,
        family,
        frontend,
        execution,
        strategy,
        scheme,
        backend,
    })
}

fn parse_bvp_equations(document: &DocumentMap) -> Result<BvpEquationSpec, BvpTaskError> {
    let parsed = parse_symbolic_equation_system(document, "x").map_err(map_equation_error)?;
    Ok(BvpEquationSpec {
        arg: parsed.arg,
        unknowns: parsed.unknowns,
        rhs: parsed.rhs,
        parameter_names: parsed.parameter_names,
        parameter_values: parsed.parameter_values,
    })
}

fn map_equation_error(error: SharedEquationParseError) -> BvpTaskError {
    match error {
        SharedEquationParseError::MissingSection(section) => BvpTaskError::MissingSection(section),
        SharedEquationParseError::MissingField { section, field } => {
            BvpTaskError::MissingField { section, field }
        }
        SharedEquationParseError::InvalidField {
            section,
            field,
            message,
        } => BvpTaskError::InvalidField {
            section,
            field,
            message,
        },
        SharedEquationParseError::InconsistentEquationCounts { unknowns, rhs } => {
            BvpTaskError::InconsistentEquationCounts { unknowns, rhs }
        }
        SharedEquationParseError::Semantic(message) => BvpTaskError::Semantic(message),
    }
}

fn parse_boundary_conditions(
    document: &DocumentMap,
    unknowns: &[String],
) -> Result<BoundaryConditionSpec, BvpTaskError> {
    let section = get_required_section(document, "boundary_conditions")?;
    let unknown_set: HashSet<&str> = unknowns.iter().map(String::as_str).collect();
    let mut conditions: HashMap<String, Vec<(usize, f64)>> = HashMap::new();

    for key in section.keys() {
        let (name, side_index) = if let Some(name) = key.strip_suffix("_left") {
            (name.to_string(), 0usize)
        } else if let Some(name) = key.strip_suffix("_right") {
            (name.to_string(), 1usize)
        } else {
            continue;
        };

        if !unknown_set.contains(name.as_str()) {
            return Err(BvpTaskError::InvalidField {
                section: "boundary_conditions".to_string(),
                field: key.clone(),
                message: format!("`{name}` is not listed in `unknowns`"),
            });
        }

        let value = get_required_float(section, "boundary_conditions", key)?;
        conditions
            .entry(name)
            .or_default()
            .push((side_index, value));
    }

    if conditions.is_empty() {
        return Err(BvpTaskError::Semantic(
            "boundary_conditions must contain keys like `y_left` or `y_right`".to_string(),
        ));
    }

    Ok(BoundaryConditionSpec { conditions })
}

fn parse_bvp_mesh(document: &DocumentMap) -> Result<BvpMeshSpec, BvpTaskError> {
    let section = get_required_section(document, "mesh")?;
    Ok(BvpMeshSpec {
        t0: get_required_float(section, "mesh", "t0")?,
        t_end: get_required_float(section, "mesh", "t_end")?,
        n_steps: get_required_usize(section, "mesh", "n_steps")?,
    })
}

fn parse_bvp_initial_guess(
    document: &DocumentMap,
    unknowns: &[String],
) -> Result<BvpInitialGuessSpec, BvpTaskError> {
    let section = get_required_section(document, "initial_guess")?;
    if section.contains_key("guess") {
        let values = get_required_float_list(section, "initial_guess", "guess")?;
        if values.len() != unknowns.len() {
            return Err(BvpTaskError::InvalidField {
                section: "initial_guess".to_string(),
                field: "guess".to_string(),
                message: format!("expected {} values, got {}", unknowns.len(), values.len()),
            });
        }
        return Ok(BvpInitialGuessSpec { values });
    }

    let mut values = Vec::with_capacity(unknowns.len());
    for unknown in unknowns {
        values.push(get_required_float(section, "initial_guess", unknown)?);
    }
    Ok(BvpInitialGuessSpec { values })
}

fn parse_bvp_solver_options(document: &DocumentMap) -> Result<BvpSolverOptionsSpec, BvpTaskError> {
    let section = match document.get("solver_options") {
        Some(section) => section,
        None => return Ok(BvpSolverOptionsSpec::default()),
    };

    Ok(BvpSolverOptionsSpec {
        tolerance: get_optional_float(section, "tolerance", "solver_options")?,
        max_iterations: get_optional_usize(section, "max_iterations", "solver_options")?,
        linear_sys_method: get_optional_string(section, "linear_sys_method", "solver_options")?,
        loglevel: get_optional_string(section, "loglevel", "solver_options")?,
        generated_backend: parse_generated_backend_options(section)?,
    })
}

fn parse_generated_backend_options(
    section: &GenericSectionMap,
) -> Result<BvpGeneratedBackendSpec, BvpTaskError> {
    Ok(BvpGeneratedBackendSpec {
        preset: get_optional_string(section, "generated_backend", "solver_options")?,
        matrix_backend: get_optional_string(section, "matrix_backend", "solver_options")?,
        backend_policy: get_optional_string(section, "backend_policy", "solver_options")?,
        symbolic_backend: get_optional_string(section, "symbolic_backend", "solver_options")?,
        aot_codegen_backend: get_optional_string(section, "aot_codegen_backend", "solver_options")?,
        aot_c_compiler: get_optional_string(section, "aot_c_compiler", "solver_options")?,
        aot_build_policy: get_optional_string(section, "aot_build_policy", "solver_options")?,
        aot_build_profile: get_optional_string(section, "aot_build_profile", "solver_options")?,
        aot_compile_preset: get_optional_string(section, "aot_compile_preset", "solver_options")?,
        aot_execution_policy: get_optional_string(
            section,
            "aot_execution_policy",
            "solver_options",
        )?,
        aot_output_dir: get_optional_string(section, "aot_output_dir", "solver_options")?,
        aot_handoff_path: get_optional_string(section, "aot_handoff_path", "solver_options")?,
        banded_linear_solver: get_optional_string(
            section,
            "banded_linear_solver",
            "solver_options",
        )?,
        refinement_steps: get_optional_usize(section, "refinement_steps", "solver_options")?,
    })
}

fn generated_backend_config_from_spec(
    spec: &BvpGeneratedBackendSpec,
    task_backend: &BvpLinearBackendSpec,
) -> Result<GeneratedBackendConfig, BvpTaskError> {
    let mut config = match normalized_option(spec.preset.as_deref()).as_deref() {
        None | Some("default") | Some("defaults") => match task_backend {
            BvpLinearBackendSpec::Banded => GeneratedBackendConfig::banded_defaults(),
            BvpLinearBackendSpec::Dense | BvpLinearBackendSpec::Sparse => {
                GeneratedBackendConfig::sparse_defaults()
            }
        },
        Some("sparse") | Some("sparse_default") | Some("sparse_defaults") => {
            GeneratedBackendConfig::sparse_defaults()
        }
        Some("sparse_lambdify") | Some("lambdify_sparse") => {
            GeneratedBackendConfig::sparse_defaults()
                .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
        }
        Some("sparse_aot") | Some("sparse_build_if_missing") => {
            GeneratedBackendConfig::sparse_build_if_missing_release()
        }
        Some("sparse_aot_gcc") | Some("sparse_atomview_gcc") => {
            GeneratedBackendConfig::sparse_atomview_build_if_missing_release_gcc()
        }
        Some("sparse_aot_tcc") | Some("sparse_atomview_tcc") | Some("sparse_repeated") => {
            GeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc()
        }
        Some("sparse_aot_zig") | Some("sparse_atomview_zig") => {
            GeneratedBackendConfig::sparse_atomview_build_if_missing_release_zig()
        }
        Some("banded") | Some("banded_default") | Some("banded_defaults") => {
            GeneratedBackendConfig::banded_defaults()
        }
        Some("banded_lambdify") | Some("lambdify_banded") => {
            GeneratedBackendConfig::banded_lambdify_defaults()
        }
        Some("banded_aot") | Some("banded_build_if_missing") => {
            GeneratedBackendConfig::banded_build_if_missing_release()
        }
        Some("banded_aot_gcc") | Some("banded_atomview_gcc") => {
            GeneratedBackendConfig::banded_atomview_build_if_missing_release_gcc()
        }
        Some("banded_aot_tcc") | Some("banded_atomview_tcc") | Some("banded_repeated") => {
            GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc()
        }
        Some("banded_aot_zig") | Some("banded_atomview_zig") => {
            GeneratedBackendConfig::banded_atomview_build_if_missing_release_zig()
        }
        Some(other) => {
            return Err(invalid_solver_option(
                "generated_backend",
                format!("unknown generated backend preset `{other}`"),
            ));
        }
    };

    if let Some(matrix_backend) = spec.matrix_backend.as_deref() {
        config = apply_matrix_backend_override(config, matrix_backend)?;
    }
    if let Some(policy) = spec.backend_policy.as_deref() {
        config = config.with_backend_policy_override(Some(parse_backend_policy(policy)?));
    }
    if let Some(symbolic_backend) = spec.symbolic_backend.as_deref() {
        config = config.with_symbolic_assembly_backend(parse_symbolic_backend(symbolic_backend)?);
    }
    if let Some(codegen_backend) = spec.aot_codegen_backend.as_deref() {
        config = config.with_aot_codegen_backend(parse_aot_codegen_backend(codegen_backend)?);
    }
    if let Some(compiler) = spec.aot_c_compiler.as_deref() {
        config = config.with_aot_c_compiler(compiler);
    }
    if spec.aot_build_policy.is_some() || spec.aot_build_profile.is_some() {
        config = config.with_aot_build_policy(parse_aot_build_policy(
            spec.aot_build_policy.as_deref(),
            spec.aot_build_profile.as_deref(),
        )?);
    }
    if let Some(compile_preset) = spec.aot_compile_preset.as_deref() {
        config = apply_aot_compile_preset(config, compile_preset)?;
    }
    if let Some(execution_policy) = spec.aot_execution_policy.as_deref() {
        config = config.with_aot_execution_policy(parse_aot_execution_policy(execution_policy)?);
    }
    if spec.banded_linear_solver.is_some() || spec.refinement_steps.is_some() {
        config = config.with_banded_linear_solver_config(parse_banded_linear_solver_config(
            spec.banded_linear_solver.as_deref(),
            spec.refinement_steps,
        )?);
    }

    Ok(config)
}

/// Map task-document AOT knobs to the shared generated-IVP lifecycle config.
///
/// BVP_Damp has a richer compatibility facade, so it keeps its own handoff
/// config. BVP_sci consumes the shared symbolic lifecycle directly; keeping
/// this adapter here avoids making either solver know about the other's config
/// type while preserving one compiler/cache implementation.
fn symbolic_aot_config_from_spec(
    spec: &BvpGeneratedBackendSpec,
) -> Result<SymbolicIvpGeneratedBackendConfig, BvpTaskError> {
    let mut config = SymbolicIvpGeneratedBackendConfig::new();
    if let Some(output_dir) = spec.aot_output_dir.as_deref() {
        config = config.with_output_parent_dir(Some(PathBuf::from(output_dir)));
    }
    if let Some(handoff_path) = spec.aot_handoff_path.as_deref() {
        config = config.with_handoff_path(Some(PathBuf::from(handoff_path)));
    }
    if let Some(codegen_backend) = spec.aot_codegen_backend.as_deref() {
        config = config.with_aot_codegen_backend(parse_aot_codegen_backend(codegen_backend)?);
    }
    if let Some(compiler) = spec.aot_c_compiler.as_deref() {
        config = config.with_aot_c_compiler(compiler);
    }

    let profile =
        parse_symbolic_aot_build_profile(spec.aot_build_profile.as_deref().unwrap_or("release"))?;
    let policy = match normalized_option(spec.aot_build_policy.as_deref()).as_deref() {
        None | Some("build_if_missing") | Some("build_if_missing_release") => {
            SymbolicIvpAotBuildPolicy::BuildIfMissing { profile }
        }
        Some("require_prebuilt") | Some("require") => SymbolicIvpAotBuildPolicy::RequirePrebuilt,
        Some("rebuild_always") | Some("rebuild") => {
            SymbolicIvpAotBuildPolicy::RebuildAlways { profile }
        }
        Some(other) => {
            return Err(invalid_solver_option(
                "aot_build_policy",
                format!("unknown BVP_sci AOT build policy `{other}`"),
            ));
        }
    };
    Ok(config.with_build_policy(policy))
}

fn parse_symbolic_aot_build_profile(
    raw: &str,
) -> Result<crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile, BvpTaskError>
{
    match normalize_token(raw).as_str() {
        "debug" => {
            Ok(crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile::Debug)
        }
        "release" => {
            Ok(crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile::Release)
        }
        other => Err(invalid_solver_option(
            "aot_build_profile",
            format!("unknown AOT build profile `{other}`"),
        )),
    }
}

fn normalized_option(raw: Option<&str>) -> Option<String> {
    raw.map(normalize_token).filter(|value| !value.is_empty())
}

fn normalize_token(raw: &str) -> String {
    raw.trim().to_ascii_lowercase().replace(['-', ' '], "_")
}

fn invalid_solver_option(field: &str, message: String) -> BvpTaskError {
    BvpTaskError::InvalidField {
        section: "solver_options".to_string(),
        field: field.to_string(),
        message,
    }
}

fn apply_matrix_backend_override(
    config: GeneratedBackendConfig,
    raw: &str,
) -> Result<GeneratedBackendConfig, BvpTaskError> {
    match normalize_token(raw).as_str() {
        "default" | "none" => Ok(config),
        "sparse" | "sparse_col" | "sparsecol" => {
            Ok(config.with_matrix_backend_override(MatrixBackend::SparseCol))
        }
        "banded" => Ok(config.with_matrix_backend_override(MatrixBackend::Banded)),
        "dense" => Ok(config.with_matrix_backend_override(MatrixBackend::Dense)),
        other => Err(invalid_solver_option(
            "matrix_backend",
            format!("unknown matrix backend `{other}`"),
        )),
    }
}

fn parse_backend_policy(raw: &str) -> Result<BackendSelectionPolicy, BvpTaskError> {
    match normalize_token(raw).as_str() {
        "numeric" | "numeric_only" => Ok(BackendSelectionPolicy::NumericOnly),
        "lambdify" | "lambdify_only" => Ok(BackendSelectionPolicy::LambdifyOnly),
        "aot" | "aot_only" => Ok(BackendSelectionPolicy::AotOnly),
        "prefer_aot" | "prefer_aot_then_lambdify" | "aot_then_lambdify" => {
            Ok(BackendSelectionPolicy::PreferAotThenLambdify)
        }
        "prefer_aot_then_numeric" | "aot_then_numeric" => {
            Ok(BackendSelectionPolicy::PreferAotThenNumeric)
        }
        "prefer_lambdify_then_numeric" | "lambdify_then_numeric" => {
            Ok(BackendSelectionPolicy::PreferLambdifyThenNumeric)
        }
        other => Err(invalid_solver_option(
            "backend_policy",
            format!("unknown backend policy `{other}`"),
        )),
    }
}

fn parse_symbolic_backend(raw: &str) -> Result<BvpSymbolicAssemblyBackend, BvpTaskError> {
    match normalize_token(raw).as_str() {
        "exprlegacy" | "expr_legacy" | "legacy" => Ok(BvpSymbolicAssemblyBackend::ExprLegacy),
        "atomview" | "atomviewnative" | "atom_view" | "atomnative" | "atom_native" | "atom" => {
            Ok(BvpSymbolicAssemblyBackend::AtomView)
        }
        other => Err(invalid_solver_option(
            "symbolic_backend",
            format!("unknown symbolic backend `{other}`"),
        )),
    }
}

fn parse_aot_codegen_backend(raw: &str) -> Result<AotCodegenBackend, BvpTaskError> {
    match normalize_token(raw).as_str() {
        "rust" | "rs" => Ok(AotCodegenBackend::Rust),
        "c" => Ok(AotCodegenBackend::C),
        "zig" => Ok(AotCodegenBackend::Zig),
        other => Err(invalid_solver_option(
            "aot_codegen_backend",
            format!("unknown AOT codegen backend `{other}`"),
        )),
    }
}

fn parse_aot_build_policy(
    raw_policy: Option<&str>,
    raw_profile: Option<&str>,
) -> Result<AotBuildPolicy, BvpTaskError> {
    let profile = parse_aot_build_profile(raw_profile.unwrap_or("release"))?;
    match normalized_option(raw_policy).as_deref() {
        None | Some("use_if_available") | Some("use") | Some("auto") => {
            Ok(AotBuildPolicy::UseIfAvailable)
        }
        Some("build_if_missing") | Some("build") => Ok(AotBuildPolicy::BuildIfMissing { profile }),
        Some("require_prebuilt") | Some("require") | Some("prebuilt") => {
            Ok(AotBuildPolicy::RequirePrebuilt)
        }
        Some("rebuild_always") | Some("rebuild") => Ok(AotBuildPolicy::RebuildAlways { profile }),
        Some(other) => Err(invalid_solver_option(
            "aot_build_policy",
            format!("unknown AOT build policy `{other}`"),
        )),
    }
}

fn parse_aot_build_profile(raw: &str) -> Result<AotBuildProfile, BvpTaskError> {
    match normalize_token(raw).as_str() {
        "release" => Ok(AotBuildProfile::Release),
        "debug" => Ok(AotBuildProfile::Debug),
        other => Err(invalid_solver_option(
            "aot_build_profile",
            format!("unknown AOT build profile `{other}`"),
        )),
    }
}

fn apply_aot_compile_preset(
    config: GeneratedBackendConfig,
    raw: &str,
) -> Result<GeneratedBackendConfig, BvpTaskError> {
    match normalize_token(raw).as_str() {
        "production" | "prod" | "release" => Ok(config.with_aot_compile_production()),
        "fast_build" | "fastbuild" => Ok(config.with_aot_compile_fast_build()),
        "dev_fastest" | "devfastest" | "debug_fast" => Ok(config.with_aot_compile_dev_fastest()),
        other => Err(invalid_solver_option(
            "aot_compile_preset",
            format!("unknown AOT compile preset `{other}`"),
        )),
    }
}

fn parse_aot_execution_policy(raw: &str) -> Result<AotExecutionPolicy, BvpTaskError> {
    match normalize_token(raw).as_str() {
        "auto" => Ok(AotExecutionPolicy::Auto),
        "sequential" | "sequential_only" | "seq" => Ok(AotExecutionPolicy::SequentialOnly),
        "parallel" => Err(invalid_solver_option(
            "aot_execution_policy",
            "`parallel` requires a ParallelExecutorConfig and is not yet exposed in task files"
                .to_string(),
        )),
        other => Err(invalid_solver_option(
            "aot_execution_policy",
            format!("unknown AOT execution policy `{other}`"),
        )),
    }
}

fn parse_banded_linear_solver_config(
    raw_solver: Option<&str>,
    refinement_steps: Option<usize>,
) -> Result<LinearSolverConfig, BvpTaskError> {
    let refinement_steps = refinement_steps.unwrap_or(0);
    let config = match normalized_option(raw_solver).as_deref() {
        None | Some("default") | Some("auto") => {
            LinearSolverConfig::auto().with_iterative_refinement_steps(refinement_steps)
        }
        Some("faithful") | Some("lapack") | Some("lapack_style_banded_lu") => {
            LinearSolverConfig::faithful_banded_with_refinement(refinement_steps)
        }
        Some("block_tridiagonal") | Some("block_tridiagonal_lu") => LinearSolverConfig {
            policy: LinearSolverPolicy::ForceBlockTridiagonal,
            iterative_refinement_steps: refinement_steps,
            ..LinearSolverConfig::default()
        },
        Some("block_tridiagonal_consistent")
        | Some("block_tridiagonal_lu_consistent")
        | Some("consistent") => LinearSolverConfig {
            policy: LinearSolverPolicy::ForceBlockTridiagonalConsistent,
            iterative_refinement_steps: refinement_steps,
            ..LinearSolverConfig::default()
        },
        Some("faer_sparse") | Some("faer_sparse_lu") => LinearSolverConfig {
            policy: LinearSolverPolicy::ForceFaerSparse,
            iterative_refinement_steps: refinement_steps,
            ..LinearSolverConfig::default()
        },
        Some("general_partial_pivot") | Some("dense_general_pivot") => LinearSolverConfig {
            policy: LinearSolverPolicy::ForceGeneralBandedPartialPivot,
            iterative_refinement_steps: refinement_steps,
            ..LinearSolverConfig::default()
        },
        Some(other) => {
            return Err(invalid_solver_option(
                "banded_linear_solver",
                format!("unknown banded linear solver `{other}`"),
            ));
        }
    };
    Ok(config)
}

fn parse_bvp_postprocessing(document: &DocumentMap) -> Result<BvpPostprocessingSpec, BvpTaskError> {
    let section = match document.get("postprocessing") {
        Some(section) => section,
        None => return Ok(BvpPostprocessingSpec::default()),
    };

    Ok(BvpPostprocessingSpec {
        save_csv: get_optional_bool(section, "save_csv", "postprocessing")?.unwrap_or(false),
        csv_path: get_optional_string(section, "csv_path", "postprocessing")?,
        save_txt: get_optional_bool(section, "save_txt", "postprocessing")?.unwrap_or(false),
        txt_path: get_optional_string(section, "txt_path", "postprocessing")?,
        write_report: get_optional_bool(section, "write_report", "postprocessing")?
            .unwrap_or(false),
        report_path: get_optional_string(section, "report_path", "postprocessing")?,
        plotters_png: get_optional_bool(section, "plotters_png", "postprocessing")?
            .unwrap_or(false),
        plotters_dir: get_optional_string(section, "plotters_dir", "postprocessing")?,
        gnuplot_png: get_optional_bool(section, "gnuplot_png", "postprocessing")?.unwrap_or(false),
        gnuplot_dir: get_optional_string(section, "gnuplot_dir", "postprocessing")?,
        terminal_plot: get_optional_bool(section, "terminal_plot", "postprocessing")?
            .unwrap_or(false),
        output_policy: get_optional_string(section, "output_policy", "postprocessing")?
            .map(|value| BvpOutputPolicy::from_str(&value))
            .transpose()?
            .unwrap_or_default(),
        plot: get_optional_bool(section, "plot", "postprocessing")?.unwrap_or(false),
    })
}

fn default_bvp_pseudonyms() -> (HashMap<String, Vec<String>>, HashMap<String, Vec<String>>) {
    let headers = HashMap::from([
        (
            "task".to_string(),
            vec!["problem".to_string(), "solver_selection".to_string()],
        ),
        (
            "equations".to_string(),
            vec!["system".to_string(), "bvp_system".to_string()],
        ),
        (
            "where".to_string(),
            vec!["substitute".to_string(), "aliases".to_string()],
        ),
        (
            "boundary_conditions".to_string(),
            vec!["boundary".to_string(), "bc".to_string()],
        ),
        (
            "mesh".to_string(),
            vec!["grid".to_string(), "domain".to_string()],
        ),
        (
            "initial_guess".to_string(),
            vec!["guess".to_string(), "initial".to_string()],
        ),
        (
            "solver_options".to_string(),
            vec!["solver_settings".to_string(), "options".to_string()],
        ),
        (
            "postprocessing".to_string(),
            vec!["output".to_string(), "postprocess".to_string()],
        ),
    ]);
    let fields = HashMap::from([
        (
            "parameters".to_string(),
            vec!["params".to_string(), "parameter_names".to_string()],
        ),
        (
            "parameter_values".to_string(),
            vec!["params_values".to_string(), "param_values".to_string()],
        ),
        (
            "t_end".to_string(),
            vec!["x_end".to_string(), "tbound".to_string()],
        ),
    ]);
    (headers, fields)
}

fn get_required_section<'a>(
    document: &'a DocumentMap,
    section: &'static str,
) -> Result<&'a GenericSectionMap, BvpTaskError> {
    document
        .get(section)
        .ok_or(BvpTaskError::MissingSection(section))
}

fn get_required_values<'a>(
    section: &'a GenericSectionMap,
    section_name: &str,
    field: &str,
) -> Result<&'a Vec<Value>, BvpTaskError> {
    section
        .get(field)
        .ok_or_else(|| BvpTaskError::MissingField {
            section: section_name.to_string(),
            field: field.to_string(),
        })?
        .as_ref()
        .ok_or_else(|| BvpTaskError::MissingField {
            section: section_name.to_string(),
            field: field.to_string(),
        })
}

fn get_required_string(
    section: &GenericSectionMap,
    section_name: &str,
    field: &str,
) -> Result<String, BvpTaskError> {
    let values = get_required_values(section, section_name, field)?;
    if values.len() != 1 {
        return Err(BvpTaskError::InvalidField {
            section: section_name.to_string(),
            field: field.to_string(),
            message: "expected a single string value".to_string(),
        });
    }
    value_to_string(&values[0], section_name, field)
}

fn get_optional_string(
    section: &GenericSectionMap,
    field: &str,
    section_name: &str,
) -> Result<Option<String>, BvpTaskError> {
    match section.get(field) {
        Some(Some(values)) if !values.is_empty() => {
            if values.len() != 1 {
                return Err(BvpTaskError::InvalidField {
                    section: section_name.to_string(),
                    field: field.to_string(),
                    message: "expected a single string value".to_string(),
                });
            }
            Ok(Some(value_to_string(&values[0], section_name, field)?))
        }
        _ => Ok(None),
    }
}

fn get_required_float(
    section: &GenericSectionMap,
    section_name: &str,
    field: &str,
) -> Result<f64, BvpTaskError> {
    let values = get_required_values(section, section_name, field)?;
    if values.len() != 1 {
        return Err(BvpTaskError::InvalidField {
            section: section_name.to_string(),
            field: field.to_string(),
            message: "expected a single numeric value".to_string(),
        });
    }
    value_to_float(&values[0], section_name, field)
}

fn get_optional_float(
    section: &GenericSectionMap,
    field: &str,
    section_name: &str,
) -> Result<Option<f64>, BvpTaskError> {
    match section.get(field) {
        Some(Some(values)) if !values.is_empty() => {
            if values.len() != 1 {
                return Err(BvpTaskError::InvalidField {
                    section: section_name.to_string(),
                    field: field.to_string(),
                    message: "expected a single numeric value".to_string(),
                });
            }
            Ok(Some(value_to_float(&values[0], section_name, field)?))
        }
        _ => Ok(None),
    }
}

fn get_required_usize(
    section: &GenericSectionMap,
    section_name: &str,
    field: &str,
) -> Result<usize, BvpTaskError> {
    let values = get_required_values(section, section_name, field)?;
    if values.len() != 1 {
        return Err(BvpTaskError::InvalidField {
            section: section_name.to_string(),
            field: field.to_string(),
            message: "expected a single integer value".to_string(),
        });
    }
    values[0]
        .as_usize()
        .ok_or_else(|| BvpTaskError::InvalidField {
            section: section_name.to_string(),
            field: field.to_string(),
            message: "expected usize".to_string(),
        })
}

fn get_optional_usize(
    section: &GenericSectionMap,
    field: &str,
    section_name: &str,
) -> Result<Option<usize>, BvpTaskError> {
    match section.get(field) {
        Some(Some(values)) if !values.is_empty() => {
            if values.len() != 1 {
                return Err(BvpTaskError::InvalidField {
                    section: section_name.to_string(),
                    field: field.to_string(),
                    message: "expected a single integer value".to_string(),
                });
            }
            values[0]
                .as_usize()
                .ok_or_else(|| BvpTaskError::InvalidField {
                    section: section_name.to_string(),
                    field: field.to_string(),
                    message: "expected usize".to_string(),
                })
                .map(Some)
        }
        _ => Ok(None),
    }
}

fn get_optional_bool(
    section: &GenericSectionMap,
    field: &str,
    section_name: &str,
) -> Result<Option<bool>, BvpTaskError> {
    match section.get(field) {
        Some(Some(values)) if !values.is_empty() => {
            if values.len() != 1 {
                return Err(BvpTaskError::InvalidField {
                    section: section_name.to_string(),
                    field: field.to_string(),
                    message: "expected a single boolean value".to_string(),
                });
            }
            values[0]
                .as_boolean()
                .ok_or_else(|| BvpTaskError::InvalidField {
                    section: section_name.to_string(),
                    field: field.to_string(),
                    message: "expected bool".to_string(),
                })
                .map(Some)
        }
        _ => Ok(None),
    }
}

fn get_required_float_list(
    section: &GenericSectionMap,
    section_name: &str,
    field: &str,
) -> Result<Vec<f64>, BvpTaskError> {
    let values = get_required_values(section, section_name, field)?;
    values_to_float_list(values, section_name, field)
}

fn values_to_float_list(
    values: &[Value],
    section_name: &str,
    field: &str,
) -> Result<Vec<f64>, BvpTaskError> {
    if values.len() == 1 {
        if let Some(vector) = values[0].as_vector() {
            return Ok(vector.clone());
        }
    }
    values
        .iter()
        .map(|value| value_to_float(value, section_name, field))
        .collect()
}

fn value_to_string(value: &Value, section_name: &str, field: &str) -> Result<String, BvpTaskError> {
    if let Some(text) = value.as_string() {
        Ok(text.clone())
    } else {
        Err(BvpTaskError::InvalidField {
            section: section_name.to_string(),
            field: field.to_string(),
            message: "expected string".to_string(),
        })
    }
}

fn value_to_float(value: &Value, section_name: &str, field: &str) -> Result<f64, BvpTaskError> {
    if let Some(number) = value.as_float() {
        Ok(number)
    } else if let Some(integer) = value.as_usize() {
        Ok(integer as f64)
    } else {
        Err(BvpTaskError::InvalidField {
            section: section_name.to_string(),
            field: field.to_string(),
            message: "expected numeric value".to_string(),
        })
    }
}

#[cfg(test)]
#[path = "task_parser_bvp_tests.rs"]
mod task_parser_bvp_tests;
