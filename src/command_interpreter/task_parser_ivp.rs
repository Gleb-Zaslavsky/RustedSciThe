//! IVP task-shell built on top of the generic [`DocumentParser`].
//!
//! This module turns a structured text document into a typed IVP specification
//! and then wires it into [`UniversalODESolver`]. It is intentionally narrower
//! than the historical damped-BVP task parser: the parser stage only validates
//! and normalizes user input, while solver execution and postprocessing remain
//! explicit follow-up steps.

use crate::command_interpreter::task_parser::{DocumentMap, DocumentParser, Value};
use crate::numerical::ODE_api2::{SolverType, UniversalODESolver};
use crate::numerical::Radau::Radau_main::RadauOrder;
use crate::symbolic::symbolic_engine::Expr;
use csv::Writer;
use nalgebra::{DMatrix, DVector};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

/// Top-level family selector for text-driven task shells.
///
/// For now we only support IVP tasks, but keeping this enum explicit makes it
/// easier to expand the textual interface later without redesigning the whole
/// normalization layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskKindSpec {
    Ivp,
}

/// Method selection normalized out of the user-facing string document.
///
/// We keep a small typed enum instead of storing raw strings all the way down
/// so that validation happens once, early, and the adapter to
/// [`UniversalODESolver`] stays straightforward.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IvpMethodSpec {
    NonStiff(String),
    Radau3,
    Radau5,
    Bdf,
    BackwardEuler,
}

impl IvpMethodSpec {
    fn to_solver_type(&self) -> SolverType {
        match self {
            Self::NonStiff(name) => SolverType::NonStiff(name.clone()),
            Self::Radau3 => SolverType::Radau(RadauOrder::Order3),
            Self::Radau5 => SolverType::Radau(RadauOrder::Order5),
            Self::Bdf => SolverType::BDF,
            Self::BackwardEuler => SolverType::BackwardEuler,
        }
    }

    fn from_str(raw: &str) -> Result<Self, IvpTaskError> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "rk45" => Ok(Self::NonStiff("RK45".to_string())),
            "rk4" => Ok(Self::NonStiff("RK4".to_string())),
            "euler" => Ok(Self::NonStiff("euler".to_string())),
            "ab4" => Ok(Self::NonStiff("AB4".to_string())),
            "radau3" | "radau-3" | "radau_iia_3" => Ok(Self::Radau3),
            "radau5" | "radau-5" | "radau_iia_5" => Ok(Self::Radau5),
            "bdf" => Ok(Self::Bdf),
            "backwardeuler" | "backward_euler" | "implicit_euler" => Ok(Self::BackwardEuler),
            other => Err(IvpTaskError::UnknownMethod(other.to_string())),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolverSelectionSpec {
    pub task_kind: TaskKindSpec,
    pub method: IvpMethodSpec,
}

/// Symbolic IVP system after document normalization.
///
/// At this stage the independent variable, unknown names, symbolic RHS and
/// numeric parameter substitutions are already separated and validated.
#[derive(Debug, Clone, PartialEq)]
pub struct EquationSpec {
    pub arg: String,
    pub unknowns: Vec<String>,
    pub rhs: Vec<Expr>,
    pub parameter_names: Vec<String>,
    pub parameter_values: HashMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InitialConditionSpec {
    pub t0: f64,
    pub t_end: f64,
    pub y0: Vec<f64>,
}

/// Solver knobs that can be mapped directly onto `UniversalODESolver` setters.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct IvpSolverOptionsSpec {
    pub step_size: Option<f64>,
    pub tolerance: Option<f64>,
    pub max_iterations: Option<usize>,
    pub rtol: Option<f64>,
    pub atol: Option<f64>,
    pub max_step: Option<f64>,
    pub first_step: Option<f64>,
    pub vectorized: Option<bool>,
    pub parallel: Option<bool>,
    pub neighborhood_check: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct PostprocessingSpec {
    pub save_csv: bool,
    pub csv_path: Option<String>,
    pub plot: bool,
}

/// Fully normalized textual IVP task ready to be turned into a solver.
#[derive(Debug, Clone, PartialEq)]
pub struct IvpTaskSpec {
    pub solver: SolverSelectionSpec,
    pub equations: EquationSpec,
    pub initial_conditions: InitialConditionSpec,
    pub solver_options: IvpSolverOptionsSpec,
    pub postprocessing: PostprocessingSpec,
}

#[derive(Debug)]
pub struct IvpTaskRunResult {
    pub specification: IvpTaskSpec,
    pub t_result: Option<DVector<f64>>,
    pub y_result: Option<DMatrix<f64>>,
    pub status: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum IvpTaskError {
    Parser(String),
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
    UnknownMethod(String),
    Semantic(String),
    Solver(String),
}

impl std::fmt::Display for IvpTaskError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parser(msg) => write!(f, "parser error: {msg}"),
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
            Self::UnknownMethod(method) => write!(f, "unknown IVP method `{method}`"),
            Self::Semantic(message) => write!(f, "{message}"),
            Self::Solver(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for IvpTaskError {}

type GenericSectionMap = HashMap<String, Option<Vec<Value>>>;

/// Parse a user-facing IVP document into a typed specification.
///
/// The generic `DocumentParser` handles syntax. This function is responsible
/// for IVP-specific normalization:
/// - resolve section/field pseudonyms
/// - preserve pair-style variable names in `equations`
/// - validate counts and types
/// - substitute numeric parameters into symbolic RHS expressions
pub fn parse_ivp_task_from_str(input: &str) -> Result<IvpTaskSpec, IvpTaskError> {
    let mut parser = DocumentParser::new(input.to_string());
    let pseudonyms = default_ivp_pseudonyms();
    parser.with_pseudonims(Some(pseudonyms.0), Some(pseudonyms.1));
    parser.parse_document().map_err(IvpTaskError::Parser)?;
    parser.keys_to_lower_case(Some(vec!["equations".to_string()]));
    let document = parser
        .get_result()
        .ok_or_else(|| IvpTaskError::Parser("document parser returned no result".to_string()))?;
    parse_ivp_task_from_document(document)
}

pub fn parse_ivp_task_from_document(document: &DocumentMap) -> Result<IvpTaskSpec, IvpTaskError> {
    let solver = parse_solver_selection(document)?;
    let equations = parse_equations(document)?;
    let initial_conditions = parse_initial_conditions(document, equations.unknowns.len())?;
    let solver_options = parse_solver_options(document)?;
    let postprocessing = parse_postprocessing(document)?;

    Ok(IvpTaskSpec {
        solver,
        equations,
        initial_conditions,
        solver_options,
        postprocessing,
    })
}

/// Convert a typed IVP spec into the user-facing ODE facade.
///
/// This adapter intentionally stays shallow: all semantic validation should
/// already be done in the parser/normalizer layer, so here we just move data
/// into `UniversalODESolver` and forward supported options through setters.
pub fn build_ivp_solver_from_spec(spec: &IvpTaskSpec) -> Result<UniversalODESolver, IvpTaskError> {
    if spec.equations.unknowns.len() != spec.initial_conditions.y0.len() {
        return Err(IvpTaskError::Semantic(format!(
            "initial condition vector length {} does not match number of unknowns {}",
            spec.initial_conditions.y0.len(),
            spec.equations.unknowns.len()
        )));
    }

    let mut solver = UniversalODESolver::new(
        spec.equations.rhs.clone(),
        spec.equations.unknowns.clone(),
        spec.equations.arg.clone(),
        spec.solver.method.to_solver_type(),
        spec.initial_conditions.t0,
        DVector::from_vec(spec.initial_conditions.y0.clone()),
        spec.initial_conditions.t_end,
    );

    if let Some(value) = spec.solver_options.step_size {
        solver.set_step_size(value);
    }
    if let Some(value) = spec.solver_options.tolerance {
        solver.set_tolerance(value);
    }
    if let Some(value) = spec.solver_options.max_iterations {
        solver.set_max_iterations(value);
    }
    if let Some(value) = spec.solver_options.rtol {
        solver.set_rtol(value);
    }
    if let Some(value) = spec.solver_options.atol {
        solver.set_atol(value);
    }
    if let Some(value) = spec.solver_options.max_step {
        solver.set_max_step(value);
    }
    solver.set_first_step(spec.solver_options.first_step);
    if let Some(value) = spec.solver_options.vectorized {
        solver.set_vectorized(value);
    }
    if let Some(value) = spec.solver_options.parallel {
        solver.set_parallel(value);
    }
    if let Some(value) = spec.solver_options.neighborhood_check {
        solver.set_neighborhood_check(value);
    }

    Ok(solver)
}

pub fn run_ivp_task_from_str(input: &str) -> Result<IvpTaskRunResult, IvpTaskError> {
    let spec = parse_ivp_task_from_str(input)?;
    run_ivp_task(spec)
}

/// Solve a normalized IVP task and optionally execute lightweight postprocessing.
///
/// At the moment postprocessing is intentionally conservative:
/// - CSV export is supported
/// - plot requests are parsed and preserved in the spec, but plotting is left
///   to higher-level wrappers instead of being triggered implicitly here
pub fn run_ivp_task(spec: IvpTaskSpec) -> Result<IvpTaskRunResult, IvpTaskError> {
    let mut solver = build_ivp_solver_from_spec(&spec)?;
    solver
        .try_solve()
        .map_err(|err| IvpTaskError::Solver(err.to_string()))?;
    let (t_result, y_result) = solver.get_result();
    let status = solver.get_status();
    if spec.postprocessing.save_csv {
        let csv_path = spec
            .postprocessing
            .csv_path
            .clone()
            .unwrap_or_else(|| "ivp_result.csv".to_string());
        write_ivp_result_to_csv(
            &csv_path,
            &spec.equations.arg,
            &spec.equations.unknowns,
            t_result.as_ref(),
            y_result.as_ref(),
        )?;
    }
    Ok(IvpTaskRunResult {
        specification: spec,
        t_result,
        y_result,
        status,
    })
}

pub fn create_ivp_template_file(path: Option<PathBuf>) {
    use std::env;
    use std::fs::File;
    use std::io::Write;

    let template = r#"
task
solver: IVP
method: BDF

equations
arg: t
parameters: a
parameter_values: 1.0
y: -a*y

initial_conditions
t0: 0.0
t_end: 1.0
y0: 1.0

solver_options
rtol: 1e-6
atol: 1e-8
max_step: 0.1
first_step: Some(1e-4)
parallel: false

postprocessing
save_csv: false
csv_path: ivp_result.csv
plot: false
"#;

    let file_path = path.unwrap_or_else(|| {
        let mut default_path = env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
        default_path.push("ivp_task_template.txt");
        default_path
    });

    match File::create(&file_path) {
        Ok(mut file) => {
            if let Err(err) = file.write_all(template.as_bytes()) {
                eprintln!("Failed to write IVP template file: {err}");
            }
        }
        Err(err) => eprintln!("Failed to create IVP template file: {err}"),
    }
}

/// Parse the `task` section and select the IVP method family.
fn parse_solver_selection(document: &DocumentMap) -> Result<SolverSelectionSpec, IvpTaskError> {
    let task_section = get_required_section(document, "task")?;
    let solver_name = get_required_string(task_section, "task", "solver")?;
    if !solver_name.eq_ignore_ascii_case("ivp") {
        return Err(IvpTaskError::InvalidField {
            section: "task".to_string(),
            field: "solver".to_string(),
            message: format!("expected `IVP`, got `{solver_name}`"),
        });
    }
    let method = IvpMethodSpec::from_str(&get_required_string(task_section, "task", "method")?)?;
    Ok(SolverSelectionSpec {
        task_kind: TaskKindSpec::Ivp,
        method,
    })
}

/// Parse the symbolic ODE system and substitute constant parameters.
///
/// We support two equivalent user-facing forms:
/// - `unknowns` + `rhs`
/// - pair-style `y: -a*y`
///
/// The second form is often nicer for humans, so we preserve variable names in
/// the `equations` section when lowercasing other configuration keys.
fn parse_equations(document: &DocumentMap) -> Result<EquationSpec, IvpTaskError> {
    let section = get_required_section(document, "equations")?;
    let arg = get_optional_string(section, "arg", "equations")?.unwrap_or_else(|| "t".to_string());

    let parameter_names =
        get_optional_string_list(section, "parameters", "equations")?.unwrap_or_default();
    let parameter_values_vec =
        get_optional_float_list(section, "parameter_values", "equations")?.unwrap_or_default();
    if parameter_names.len() != parameter_values_vec.len() {
        return Err(IvpTaskError::InvalidField {
            section: "equations".to_string(),
            field: "parameter_values".to_string(),
            message: format!(
                "expected {} parameter values, got {}",
                parameter_names.len(),
                parameter_values_vec.len()
            ),
        });
    }
    let parameter_values: HashMap<String, f64> = parameter_names
        .iter()
        .cloned()
        .zip(parameter_values_vec)
        .collect();

    validate_symbol_names(&arg, &parameter_names)?;

    let (unknowns, rhs_raw) = if section.contains_key("unknowns") || section.contains_key("rhs") {
        let unknowns = get_required_string_list(section, "equations", "unknowns")?;
        let rhs = get_required_string_list(section, "equations", "rhs")?;
        (unknowns, rhs)
    } else {
        parse_pair_style_equations(section)?
    };

    if unknowns.len() != rhs_raw.len() {
        return Err(IvpTaskError::InconsistentEquationCounts {
            unknowns: unknowns.len(),
            rhs: rhs_raw.len(),
        });
    }

    let unknown_set: HashSet<&str> = unknowns.iter().map(String::as_str).collect();
    if unknown_set.contains(arg.as_str()) {
        return Err(IvpTaskError::Semantic(format!(
            "argument `{arg}` cannot also be listed as an unknown"
        )));
    }
    for parameter in &parameter_names {
        if unknown_set.contains(parameter.as_str()) {
            return Err(IvpTaskError::Semantic(format!(
                "parameter `{parameter}` cannot also be listed as an unknown"
            )));
        }
        if parameter == &arg {
            return Err(IvpTaskError::Semantic(format!(
                "parameter `{parameter}` cannot also be the independent argument"
            )));
        }
    }

    let rhs = rhs_raw
        .iter()
        .map(|expr| Expr::parse_expression(expr).set_variable_from_map(&parameter_values))
        .collect();

    Ok(EquationSpec {
        arg,
        unknowns,
        rhs,
        parameter_names,
        parameter_values,
    })
}

/// Extract pair-style equations where each field name is itself an unknown.
fn parse_pair_style_equations(
    section: &GenericSectionMap,
) -> Result<(Vec<String>, Vec<String>), IvpTaskError> {
    let reserved = ["arg", "parameters", "parameter_values", "unknowns", "rhs"];
    let mut unknowns = Vec::new();
    let mut rhs = Vec::new();

    for (key, _) in section {
        if reserved.contains(&key.as_str()) {
            continue;
        }
        let expr = get_required_string(section, "equations", key)?;
        unknowns.push(key.clone());
        rhs.push(expr);
    }

    if unknowns.is_empty() {
        return Err(IvpTaskError::Semantic(
            "equations section must contain either `unknowns`/`rhs` lists or variable-to-rhs pairs"
                .to_string(),
        ));
    }

    Ok((unknowns, rhs))
}

/// Parse `t0`, `t_end`, and the initial state vector.
fn parse_initial_conditions(
    document: &DocumentMap,
    expected_dimension: usize,
) -> Result<InitialConditionSpec, IvpTaskError> {
    let section = get_required_section(document, "initial_conditions")?;
    let t0 = get_required_float(section, "initial_conditions", "t0")?;
    let t_end = get_required_float(section, "initial_conditions", "t_end")?;
    let y0 = get_required_float_list(section, "initial_conditions", "y0")?;
    if y0.len() != expected_dimension {
        return Err(IvpTaskError::InvalidField {
            section: "initial_conditions".to_string(),
            field: "y0".to_string(),
            message: format!(
                "expected {expected_dimension} initial values, got {}",
                y0.len()
            ),
        });
    }
    Ok(InitialConditionSpec { t0, t_end, y0 })
}

/// Parse optional solver controls that map onto `ODE_api2`.
fn parse_solver_options(document: &DocumentMap) -> Result<IvpSolverOptionsSpec, IvpTaskError> {
    let section = match document.get("solver_options") {
        Some(section) => section,
        None => return Ok(IvpSolverOptionsSpec::default()),
    };
    Ok(IvpSolverOptionsSpec {
        step_size: get_optional_float(section, "step_size")?,
        tolerance: get_optional_float(section, "tolerance")?,
        max_iterations: get_optional_usize(section, "max_iterations")?,
        rtol: get_optional_float(section, "rtol")?,
        atol: get_optional_float(section, "atol")?,
        max_step: get_optional_float(section, "max_step")?,
        first_step: get_optional_float_or_option(section, "first_step")?,
        vectorized: get_optional_bool(section, "vectorized")?,
        parallel: get_optional_bool(section, "parallel")?,
        neighborhood_check: get_optional_float(section, "neighborhood_check")?,
    })
}

/// Parse lightweight output controls.
fn parse_postprocessing(document: &DocumentMap) -> Result<PostprocessingSpec, IvpTaskError> {
    let section = match document.get("postprocessing") {
        Some(section) => section,
        None => return Ok(PostprocessingSpec::default()),
    };
    Ok(PostprocessingSpec {
        save_csv: get_optional_bool(section, "save_csv")?.unwrap_or(false),
        csv_path: get_optional_string(section, "csv_path", "postprocessing")?,
        plot: get_optional_bool(section, "plot")?.unwrap_or(false),
    })
}

/// Provide a small set of forgiving aliases for the text format.
fn default_ivp_pseudonyms() -> (HashMap<String, Vec<String>>, HashMap<String, Vec<String>>) {
    let headers = HashMap::from([
        (
            "task".to_string(),
            vec!["problem".to_string(), "solver_selection".to_string()],
        ),
        (
            "equations".to_string(),
            vec!["system".to_string(), "ode_system".to_string()],
        ),
        (
            "initial_conditions".to_string(),
            vec!["initial".to_string(), "iv".to_string()],
        ),
        (
            "solver_options".to_string(),
            vec!["solver_settings".to_string(), "options".to_string()],
        ),
    ]);
    let fields = HashMap::from([
        (
            "method".to_string(),
            vec!["ivp_method".to_string(), "solver_method".to_string()],
        ),
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
            vec!["tbound".to_string(), "t_bound".to_string()],
        ),
    ]);
    (headers, fields)
}

/// Save a solved IVP trajectory as a simple CSV table.
///
/// The first column is the independent variable, followed by one column per
/// unknown in solver order.
fn write_ivp_result_to_csv(
    path: &str,
    arg_name: &str,
    unknowns: &[String],
    t_result: Option<&DVector<f64>>,
    y_result: Option<&DMatrix<f64>>,
) -> Result<(), IvpTaskError> {
    let t_result = t_result.ok_or_else(|| {
        IvpTaskError::Solver("cannot write CSV because t_result is missing".to_string())
    })?;
    let y_result = y_result.ok_or_else(|| {
        IvpTaskError::Solver("cannot write CSV because y_result is missing".to_string())
    })?;

    if y_result.nrows() != t_result.len() {
        return Err(IvpTaskError::Solver(format!(
            "cannot write CSV because time grid has {} points but solution matrix has {} rows",
            t_result.len(),
            y_result.nrows()
        )));
    }
    if y_result.ncols() != unknowns.len() {
        return Err(IvpTaskError::Solver(format!(
            "cannot write CSV because solution matrix has {} columns but there are {} unknowns",
            y_result.ncols(),
            unknowns.len()
        )));
    }

    let mut writer = Writer::from_path(path)
        .map_err(|err| IvpTaskError::Solver(format!("failed to create CSV `{path}`: {err}")))?;

    let mut header = Vec::with_capacity(unknowns.len() + 1);
    header.push(arg_name.to_string());
    header.extend(unknowns.iter().cloned());
    writer
        .write_record(&header)
        .map_err(|err| IvpTaskError::Solver(format!("failed to write CSV header: {err}")))?;

    for row in 0..t_result.len() {
        let mut record = Vec::with_capacity(unknowns.len() + 1);
        record.push(t_result[row].to_string());
        for col in 0..y_result.ncols() {
            record.push(y_result[(row, col)].to_string());
        }
        writer
            .write_record(&record)
            .map_err(|err| IvpTaskError::Solver(format!("failed to write CSV row: {err}")))?;
    }

    writer
        .flush()
        .map_err(|err| IvpTaskError::Solver(format!("failed to flush CSV `{path}`: {err}")))?;
    Ok(())
}

fn get_required_section<'a>(
    document: &'a DocumentMap,
    section: &'static str,
) -> Result<&'a GenericSectionMap, IvpTaskError> {
    document
        .get(section)
        .ok_or(IvpTaskError::MissingSection(section))
}

fn get_required_values<'a>(
    section: &'a GenericSectionMap,
    section_name: &str,
    field: &str,
) -> Result<&'a Vec<Value>, IvpTaskError> {
    section
        .get(field)
        .ok_or_else(|| IvpTaskError::MissingField {
            section: section_name.to_string(),
            field: field.to_string(),
        })?
        .as_ref()
        .ok_or_else(|| IvpTaskError::MissingField {
            section: section_name.to_string(),
            field: field.to_string(),
        })
}

fn get_required_string(
    section: &GenericSectionMap,
    section_name: &str,
    field: &str,
) -> Result<String, IvpTaskError> {
    let values = get_required_values(section, section_name, field)?;
    if values.len() != 1 {
        return Err(IvpTaskError::InvalidField {
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
) -> Result<Option<String>, IvpTaskError> {
    match section.get(field) {
        Some(Some(values)) if !values.is_empty() => {
            if values.len() != 1 {
                return Err(IvpTaskError::InvalidField {
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

fn get_required_string_list(
    section: &GenericSectionMap,
    section_name: &str,
    field: &str,
) -> Result<Vec<String>, IvpTaskError> {
    let values = get_required_values(section, section_name, field)?;
    values
        .iter()
        .map(|value| value_to_string(value, section_name, field))
        .collect()
}

fn get_optional_string_list(
    section: &GenericSectionMap,
    field: &str,
    section_name: &str,
) -> Result<Option<Vec<String>>, IvpTaskError> {
    match section.get(field) {
        Some(Some(values)) => values
            .iter()
            .map(|value| value_to_string(value, section_name, field))
            .collect::<Result<Vec<_>, _>>()
            .map(Some),
        _ => Ok(None),
    }
}

fn get_required_float(
    section: &GenericSectionMap,
    section_name: &str,
    field: &str,
) -> Result<f64, IvpTaskError> {
    let values = get_required_values(section, section_name, field)?;
    if values.len() != 1 {
        return Err(IvpTaskError::InvalidField {
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
) -> Result<Option<f64>, IvpTaskError> {
    match section.get(field) {
        Some(Some(values)) if !values.is_empty() => {
            if values.len() != 1 {
                return Err(IvpTaskError::InvalidField {
                    section: "solver_options".to_string(),
                    field: field.to_string(),
                    message: "expected a single numeric value".to_string(),
                });
            }
            Ok(Some(value_to_float(&values[0], "solver_options", field)?))
        }
        _ => Ok(None),
    }
}

fn get_optional_float_or_option(
    section: &GenericSectionMap,
    field: &str,
) -> Result<Option<f64>, IvpTaskError> {
    match section.get(field) {
        Some(Some(values)) if !values.is_empty() => {
            if values.len() != 1 {
                return Err(IvpTaskError::InvalidField {
                    section: "solver_options".to_string(),
                    field: field.to_string(),
                    message: "expected a single numeric or optional numeric value".to_string(),
                });
            }
            match &values[0] {
                Value::Optional(None) => Ok(None),
                Value::Optional(Some(_)) => Ok(values[0].as_option_float()),
                _ => Ok(Some(value_to_float(&values[0], "solver_options", field)?)),
            }
        }
        _ => Ok(None),
    }
}

fn get_optional_usize(
    section: &GenericSectionMap,
    field: &str,
) -> Result<Option<usize>, IvpTaskError> {
    match section.get(field) {
        Some(Some(values)) if !values.is_empty() => {
            if values.len() != 1 {
                return Err(IvpTaskError::InvalidField {
                    section: "solver_options".to_string(),
                    field: field.to_string(),
                    message: "expected a single integer value".to_string(),
                });
            }
            values[0]
                .as_usize()
                .ok_or_else(|| IvpTaskError::InvalidField {
                    section: "solver_options".to_string(),
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
) -> Result<Option<bool>, IvpTaskError> {
    match section.get(field) {
        Some(Some(values)) if !values.is_empty() => {
            if values.len() != 1 {
                return Err(IvpTaskError::InvalidField {
                    section: "solver_options".to_string(),
                    field: field.to_string(),
                    message: "expected a single boolean value".to_string(),
                });
            }
            values[0]
                .as_boolean()
                .ok_or_else(|| IvpTaskError::InvalidField {
                    section: "solver_options".to_string(),
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
) -> Result<Vec<f64>, IvpTaskError> {
    let values = get_required_values(section, section_name, field)?;
    values_to_float_list(values, section_name, field)
}

fn get_optional_float_list(
    section: &GenericSectionMap,
    field: &str,
    section_name: &str,
) -> Result<Option<Vec<f64>>, IvpTaskError> {
    match section.get(field) {
        Some(Some(values)) => values_to_float_list(values, section_name, field).map(Some),
        _ => Ok(None),
    }
}

fn values_to_float_list(
    values: &[Value],
    section_name: &str,
    field: &str,
) -> Result<Vec<f64>, IvpTaskError> {
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

fn value_to_string(value: &Value, section_name: &str, field: &str) -> Result<String, IvpTaskError> {
    if let Some(text) = value.as_string() {
        Ok(text.clone())
    } else {
        Err(IvpTaskError::InvalidField {
            section: section_name.to_string(),
            field: field.to_string(),
            message: "expected string".to_string(),
        })
    }
}

fn value_to_float(value: &Value, section_name: &str, field: &str) -> Result<f64, IvpTaskError> {
    if let Some(number) = value.as_float() {
        Ok(number)
    } else if let Some(integer) = value.as_usize() {
        Ok(integer as f64)
    } else {
        Err(IvpTaskError::InvalidField {
            section: section_name.to_string(),
            field: field.to_string(),
            message: "expected numeric value".to_string(),
        })
    }
}

fn validate_symbol_names(arg: &str, parameter_names: &[String]) -> Result<(), IvpTaskError> {
    if arg.trim().is_empty() {
        return Err(IvpTaskError::Semantic(
            "independent argument name cannot be empty".to_string(),
        ));
    }
    let mut seen = HashSet::new();
    for name in parameter_names {
        if !seen.insert(name) {
            return Err(IvpTaskError::Semantic(format!(
                "duplicate parameter name `{name}`"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn ivp_task_parser_supports_pair_style_equations() {
        let input = r#"
task
solver: IVP
method: RK45

equations
arg: t
parameters: a
parameter_values: 2.0
y: -a*y

initial_conditions
t0: 0.0
t_end: 1.0
y0: 1.0

solver_options
step_size: 1e-3
"#;

        let spec = parse_ivp_task_from_str(input).expect("pair-style IVP task should parse");
        assert_eq!(spec.equations.unknowns, vec!["y".to_string()]);
        assert_eq!(spec.equations.parameter_values["a"], 2.0);
        assert_eq!(spec.initial_conditions.y0, vec![1.0]);
        assert_eq!(
            spec.solver.method,
            IvpMethodSpec::NonStiff("RK45".to_string())
        );
    }

    #[test]
    fn ivp_task_runner_solves_simple_decay() {
        let input = r#"
task
solver: IVP
method: RK45

equations
arg: t
unknowns: y
rhs: -y

initial_conditions
t0: 0.0
t_end: 0.2
y0: 1.0

solver_options
step_size: 1e-3
"#;

        let result = run_ivp_task_from_str(input).expect("simple IVP task should solve");
        assert_eq!(result.status.as_deref(), Some("finished"));
        let y = result.y_result.expect("solver should produce y_result");
        let final_y = y[(y.nrows() - 1, 0)];
        let expected = (-0.2_f64).exp();
        assert!((final_y - expected).abs() < 1e-2);
    }

    #[test]
    fn ivp_task_runner_can_save_csv() {
        let dir = tempdir().expect("tempdir should be created");
        let csv_path = dir.path().join("ivp_task_output.csv");
        let input = format!(
            r#"
task
solver: IVP
method: RK45

equations
arg: t
y: -y

initial_conditions
t0: 0.0
t_end: 0.05
y0: 1.0

solver_options
step_size: 1e-3

postprocessing
save_csv: true
csv_path: {}
"#,
            csv_path.display()
        );

        let result = run_ivp_task_from_str(&input).expect("IVP task should solve and save CSV");
        assert_eq!(result.status.as_deref(), Some("finished"));
        let contents =
            std::fs::read_to_string(&csv_path).expect("CSV file should be readable after solve");
        assert!(contents.contains("t,y"));
    }

    #[test]
    fn ivp_task_runner_supports_backward_euler() {
        let input = r#"
task
solver: IVP
method: BackwardEuler

equations
arg: t
y: -10.0*y

initial_conditions
t0: 0.0
t_end: 0.1
y0: 1.0

solver_options
step_size: 1e-3
tolerance: 1e-8
max_iterations: 100
"#;

        let result = run_ivp_task_from_str(input).expect("Backward Euler IVP task should solve");
        assert_eq!(result.status.as_deref(), Some("finished"));
        assert!(result.y_result.is_some());
    }

    #[test]
    fn ivp_task_runner_supports_bdf() {
        let input = r#"
task
solver: IVP
method: BDF

equations
arg: t
y: -20.0*y

initial_conditions
t0: 0.0
t_end: 0.1
y0: 1.0

solver_options
first_step: Some(1e-3)
rtol: 1e-6
atol: 1e-8
max_step: 0.05
"#;

        let result = run_ivp_task_from_str(input).expect("BDF IVP task should solve");
        assert_eq!(result.status.as_deref(), Some("finished"));
        assert!(result.y_result.is_some());
    }

    #[test]
    fn ivp_task_runner_supports_radau5() {
        let input = r#"
task
solver: IVP
method: Radau5

equations
arg: t
y: -15.0*y

initial_conditions
t0: 0.0
t_end: 0.1
y0: 1.0

solver_options
first_step: Some(1e-3)
rtol: 1e-6
atol: 1e-8
max_step: 0.05
"#;

        let result = run_ivp_task_from_str(input).expect("Radau5 IVP task should solve");
        assert_eq!(result.status.as_deref(), Some("finished"));
        assert!(result.y_result.is_some());
    }
}
