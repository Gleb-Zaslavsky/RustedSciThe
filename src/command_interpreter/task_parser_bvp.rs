//! BVP task-shell built on top of the generic [`DocumentParser`].
//!
//! This module mirrors the IVP task shell, but targets the historical
//! [`crate::numerical::BVP_api::BVP`] facade. The goal is intentionally modest:
//! provide a small, typed, text-facing entry point that is easy to validate and
//! easy to extend later.
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
//! - postprocessing currently supports CSV saving and optional plot flag parsing

use crate::command_interpreter::task_parser::{DocumentMap, DocumentParser, Value};
use crate::numerical::BVP_Damp::generated_solver_handoff::{
    AotBuildPolicy, AotBuildProfile, AotExecutionPolicy, GeneratedBackendConfig,
};
use crate::numerical::BVP_api::BVP;
use crate::somelinalg::banded::{LinearSolverConfig, LinearSolverPolicy};
use crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend;
use crate::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
use crate::symbolic::codegen::codegen_provider_api::MatrixBackend;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions_BVP::BvpSymbolicAssemblyBackend;
use nalgebra::DMatrix;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

type GenericSectionMap = HashMap<String, Option<Vec<Value>>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BvpTaskKindSpec {
    Bvp,
}

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BvpSolverSelectionSpec {
    pub task_kind: BvpTaskKindSpec,
    pub strategy: BvpStrategySpec,
    pub scheme: String,
    pub backend: BvpLinearBackendSpec,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BvpEquationSpec {
    pub arg: String,
    pub unknowns: Vec<String>,
    pub rhs: Vec<Expr>,
    pub parameter_names: Vec<String>,
    pub parameter_values: HashMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryConditionSpec {
    pub conditions: HashMap<String, Vec<(usize, f64)>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BvpMeshSpec {
    pub t0: f64,
    pub t_end: f64,
    pub n_steps: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BvpInitialGuessSpec {
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct BvpSolverOptionsSpec {
    pub tolerance: Option<f64>,
    pub max_iterations: Option<usize>,
    pub linear_sys_method: Option<String>,
    pub loglevel: Option<String>,
    pub generated_backend: BvpGeneratedBackendSpec,
}

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
    pub banded_linear_solver: Option<String>,
    pub refinement_steps: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct BvpPostprocessingSpec {
    pub save_csv: bool,
    pub csv_path: Option<String>,
    pub plot: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BvpTaskSpec {
    pub solver: BvpSolverSelectionSpec,
    pub equations: BvpEquationSpec,
    pub boundary_conditions: BoundaryConditionSpec,
    pub mesh: BvpMeshSpec,
    pub initial_guess: BvpInitialGuessSpec,
    pub solver_options: BvpSolverOptionsSpec,
    pub postprocessing: BvpPostprocessingSpec,
}

#[derive(Debug)]
pub struct BvpTaskRunResult {
    pub specification: BvpTaskSpec,
    pub result: Option<DMatrix<f64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BvpTaskError {
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
    UnknownStrategy(String),
    UnknownBackend(String),
    Semantic(String),
    Solver(String),
}

impl std::fmt::Display for BvpTaskError {
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
            Self::UnknownStrategy(strategy) => write!(f, "unknown BVP strategy `{strategy}`"),
            Self::UnknownBackend(method) => write!(f, "unknown BVP backend `{method}`"),
            Self::Semantic(message) => write!(f, "{message}"),
            Self::Solver(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for BvpTaskError {}

pub fn parse_bvp_task_from_str(input: &str) -> Result<BvpTaskSpec, BvpTaskError> {
    let mut parser = DocumentParser::new(input.to_string());
    let pseudonyms = default_bvp_pseudonyms();
    parser.with_pseudonims(Some(pseudonyms.0), Some(pseudonyms.1));
    parser.parse_document().map_err(BvpTaskError::Parser)?;
    parser.keys_to_lower_case(Some(vec![
        "equations".to_string(),
        "boundary_conditions".to_string(),
        "initial_guess".to_string(),
    ]));
    let document = parser
        .get_result()
        .ok_or_else(|| BvpTaskError::Parser("document parser returned no result".to_string()))?;
    parse_bvp_task_from_document(document)
}

pub fn parse_bvp_task_from_document(document: &DocumentMap) -> Result<BvpTaskSpec, BvpTaskError> {
    let solver = parse_bvp_solver_selection(document)?;
    let equations = parse_bvp_equations(document)?;
    let boundary_conditions = parse_boundary_conditions(document, &equations.unknowns)?;
    let mesh = parse_bvp_mesh(document)?;
    let initial_guess = parse_bvp_initial_guess(document, &equations.unknowns)?;
    let solver_options = parse_bvp_solver_options(document)?;
    let postprocessing = parse_bvp_postprocessing(document)?;

    Ok(BvpTaskSpec {
        solver,
        equations,
        boundary_conditions,
        mesh,
        initial_guess,
        solver_options,
        postprocessing,
    })
}

pub fn build_bvp_solver_from_spec(spec: &BvpTaskSpec) -> Result<BVP, BvpTaskError> {
    let dimension = spec.equations.unknowns.len();
    if spec.initial_guess.values.len() != dimension {
        return Err(BvpTaskError::Semantic(format!(
            "initial guess dimension {} does not match number of unknowns {}",
            spec.initial_guess.values.len(),
            dimension
        )));
    }

    let initial_guess = DMatrix::from_fn(dimension, spec.mesh.n_steps, |row, _col| {
        spec.initial_guess.values[row]
    });

    let tolerance = spec.solver_options.tolerance.unwrap_or(1e-5);
    let max_iterations = spec.solver_options.max_iterations.unwrap_or(50);
    let (strategy_params, rel_tolerance, bounds) = match spec.solver.strategy {
        BvpStrategySpec::Damped => {
            let rel_tolerance = Some(HashMap::from_iter(
                spec.equations
                    .unknowns
                    .iter()
                    .cloned()
                    .map(|name| (name, 1e-4_f64)),
            ));
            let bounds = Some(HashMap::from_iter(
                spec.equations
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

    let mut solver = BVP::new(
        spec.equations.rhs.clone(),
        initial_guess,
        spec.equations.unknowns.clone(),
        spec.equations.arg.clone(),
        spec.boundary_conditions.conditions.clone(),
        spec.mesh.t0,
        spec.mesh.t_end,
        spec.mesh.n_steps,
        spec.solver.scheme.clone(),
        spec.solver.strategy.as_solver_string(),
        strategy_params,
        spec.solver_options.linear_sys_method.clone(),
        spec.solver.backend.as_solver_string(),
        tolerance,
        max_iterations,
        rel_tolerance,
        bounds,
        spec.solver_options.loglevel.clone(),
    );

    let generated_config = generated_backend_config_from_spec(
        &spec.solver_options.generated_backend,
        &spec.solver.backend,
    )?;
    if let Some(structure_damp) = &mut solver.structure_damp {
        structure_damp.set_generated_backend_config(generated_config.clone());
    }
    if let Some(structure) = &mut solver.structure {
        structure.set_generated_backend_config(generated_config);
    }

    Ok(solver)
}

pub fn run_bvp_task_from_str(input: &str) -> Result<BvpTaskRunResult, BvpTaskError> {
    let spec = parse_bvp_task_from_str(input)?;
    run_bvp_task(spec)
}

pub fn run_bvp_task(spec: BvpTaskSpec) -> Result<BvpTaskRunResult, BvpTaskError> {
    let mut solver = build_bvp_solver_from_spec(&spec)?;
    solver.solve();
    if spec.postprocessing.save_csv {
        solver.save_to_csv(spec.postprocessing.csv_path.clone());
    }
    if spec.postprocessing.plot {
        solver.plot_result();
    }
    let result = solver.get_result();
    Ok(BvpTaskRunResult {
        specification: spec,
        result,
    })
}

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
    if !solver_name.eq_ignore_ascii_case("bvp") {
        return Err(BvpTaskError::InvalidField {
            section: "task".to_string(),
            field: "solver".to_string(),
            message: format!("expected `BVP`, got `{solver_name}`"),
        });
    }

    let strategy = BvpStrategySpec::from_str(
        &get_optional_string(task_section, "strategy", "task")?.unwrap_or_else(|| "Damped".into()),
    )?;
    let scheme = get_optional_string(task_section, "scheme", "task")?
        .unwrap_or_else(|| "forward".to_string());
    let backend = BvpLinearBackendSpec::from_str(
        &get_optional_string(task_section, "method", "task")?.unwrap_or_else(|| "Sparse".into()),
    )?;

    Ok(BvpSolverSelectionSpec {
        task_kind: BvpTaskKindSpec::Bvp,
        strategy,
        scheme,
        backend,
    })
}

fn parse_bvp_equations(document: &DocumentMap) -> Result<BvpEquationSpec, BvpTaskError> {
    let section = get_required_section(document, "equations")?;
    let arg = get_optional_string(section, "arg", "equations")?.unwrap_or_else(|| "x".to_string());

    let parameter_names =
        get_optional_string_list(section, "parameters", "equations")?.unwrap_or_default();
    let parameter_values_vec =
        get_optional_float_list(section, "parameter_values", "equations")?.unwrap_or_default();
    if parameter_names.len() != parameter_values_vec.len() {
        return Err(BvpTaskError::InvalidField {
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
        return Err(BvpTaskError::InconsistentEquationCounts {
            unknowns: unknowns.len(),
            rhs: rhs_raw.len(),
        });
    }

    let unknown_set: HashSet<&str> = unknowns.iter().map(String::as_str).collect();
    if unknown_set.contains(arg.as_str()) {
        return Err(BvpTaskError::Semantic(format!(
            "argument `{arg}` cannot also be listed as an unknown"
        )));
    }
    for parameter in &parameter_names {
        if unknown_set.contains(parameter.as_str()) {
            return Err(BvpTaskError::Semantic(format!(
                "parameter `{parameter}` cannot also be listed as an unknown"
            )));
        }
        if parameter == &arg {
            return Err(BvpTaskError::Semantic(format!(
                "parameter `{parameter}` cannot also be the independent argument"
            )));
        }
    }

    let rhs = rhs_raw
        .iter()
        .map(|expr| Expr::parse_expression(expr).set_variable_from_map(&parameter_values))
        .collect();

    Ok(BvpEquationSpec {
        arg,
        unknowns,
        rhs,
        parameter_names,
        parameter_values,
    })
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
        "atomview" | "atom_view" | "atom" => Ok(BvpSymbolicAssemblyBackend::AtomView),
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
        plot: get_optional_bool(section, "plot", "postprocessing")?.unwrap_or(false),
    })
}

fn parse_pair_style_equations(
    section: &GenericSectionMap,
) -> Result<(Vec<String>, Vec<String>), BvpTaskError> {
    // `DocumentParser` currently stores section entries in a `HashMap`, so
    // pair-style equations are best suited to scalar problems or cases where
    // the exact ordering is otherwise irrelevant. For multi-equation BVPs the
    // recommended text form is explicit `unknowns` + `rhs`.
    let reserved = ["arg", "parameters", "parameter_values", "unknowns", "rhs"];
    let mut unknowns = Vec::new();
    let mut rhs = Vec::new();

    for key in section.keys() {
        if reserved.contains(&key.as_str()) {
            continue;
        }
        let expr = get_required_string(section, "equations", key)?;
        unknowns.push(key.clone());
        rhs.push(expr);
    }

    if unknowns.is_empty() {
        return Err(BvpTaskError::Semantic(
            "equations section must contain either `unknowns`/`rhs` lists or variable-to-rhs pairs"
                .to_string(),
        ));
    }

    Ok((unknowns, rhs))
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

fn get_required_string_list(
    section: &GenericSectionMap,
    section_name: &str,
    field: &str,
) -> Result<Vec<String>, BvpTaskError> {
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
) -> Result<Option<Vec<String>>, BvpTaskError> {
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

fn get_optional_float_list(
    section: &GenericSectionMap,
    field: &str,
    section_name: &str,
) -> Result<Option<Vec<f64>>, BvpTaskError> {
    match section.get(field) {
        Some(Some(values)) => values_to_float_list(values, section_name, field).map(Some),
        _ => Ok(None),
    }
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

fn validate_symbol_names(arg: &str, parameter_names: &[String]) -> Result<(), BvpTaskError> {
    if arg.trim().is_empty() {
        return Err(BvpTaskError::Semantic(
            "independent argument name cannot be empty".to_string(),
        ));
    }
    let mut seen = HashSet::new();
    for name in parameter_names {
        if !seen.insert(name) {
            return Err(BvpTaskError::Semantic(format!(
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
    fn bvp_task_parser_supports_pair_style_equations() {
        let input = r#"
task
solver: BVP
strategy: Damped
scheme: forward
method: Sparse

equations
arg: x
y: -2.0*y

boundary_conditions
y_left: 1.0

mesh
t0: 0.0
t_end: 1.0
n_steps: 20

initial_guess
y: 0.0
"#;

        let spec = parse_bvp_task_from_str(input).expect("pair-style BVP task should parse");
        assert_eq!(spec.equations.unknowns, vec!["y".to_string()]);
        assert_eq!(spec.mesh.n_steps, 20);
        assert_eq!(
            spec.boundary_conditions.conditions["y"],
            vec![(0usize, 1.0f64)]
        );
    }

    #[test]
    fn bvp_task_parser_maps_banded_lambdify_generated_backend_config() {
        let input = r#"
task
solver: BVP
strategy: Damped
scheme: forward
method: Banded

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
generated_backend: banded_lambdify
banded_linear_solver: lapack
refinement_steps: 0
"#;

        let spec = parse_bvp_task_from_str(input).expect("banded BVP task should parse");
        assert_eq!(spec.solver.backend, BvpLinearBackendSpec::Banded);
        assert_eq!(
            spec.solver_options.generated_backend.preset.as_deref(),
            Some("banded_lambdify")
        );

        let config = generated_backend_config_from_spec(
            &spec.solver_options.generated_backend,
            &spec.solver.backend,
        )
        .expect("banded lambdify generated backend config should be valid");

        assert_eq!(config.matrix_backend_override, Some(MatrixBackend::Banded));
        assert_eq!(
            config.backend_policy_override,
            Some(BackendSelectionPolicy::LambdifyOnly)
        );
        assert_eq!(
            config.symbolic_assembly_backend,
            BvpSymbolicAssemblyBackend::ExprLegacy
        );
        assert_eq!(
            config.banded_linear_solver_config.policy,
            LinearSolverPolicy::ForceBanded
        );
        assert_eq!(
            config
                .banded_linear_solver_config
                .iterative_refinement_steps,
            0
        );
    }

    #[test]
    fn bvp_task_parser_maps_banded_aot_tcc_generated_backend_config() {
        let input = r#"
task
solver: BVP
strategy: Frozen
scheme: forward
method: Banded

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
generated_backend: banded_aot_tcc
aot_build_policy: rebuild
aot_build_profile: release
aot_compile_preset: dev_fastest
aot_execution_policy: sequential
symbolic_backend: atomview
banded_linear_solver: faithful
refinement_steps: 1
"#;

        let spec = parse_bvp_task_from_str(input).expect("banded AOT BVP task should parse");
        let config = generated_backend_config_from_spec(
            &spec.solver_options.generated_backend,
            &spec.solver.backend,
        )
        .expect("banded AOT/tcc generated backend config should be valid");

        assert_eq!(config.matrix_backend_override, Some(MatrixBackend::Banded));
        assert_eq!(
            config.backend_policy_override,
            Some(BackendSelectionPolicy::PreferAotThenLambdify)
        );
        assert_eq!(
            config.symbolic_assembly_backend,
            BvpSymbolicAssemblyBackend::AtomView
        );
        assert_eq!(config.aot_codegen_backend, AotCodegenBackend::C);
        assert_eq!(config.aot_c_compiler.as_deref(), Some("tcc"));
        assert_eq!(
            config.aot_execution_policy,
            AotExecutionPolicy::SequentialOnly
        );
        assert!(matches!(
            config.aot_build_policy,
            AotBuildPolicy::RebuildAlways {
                profile: AotBuildProfile::Release
            }
        ));
        assert_eq!(
            config.banded_linear_solver_config.policy,
            LinearSolverPolicy::ForceBanded
        );
        assert_eq!(
            config
                .banded_linear_solver_config
                .iterative_refinement_steps,
            1
        );
    }

    #[test]
    fn bvp_task_runner_solves_reference_problem() {
        let input = r#"
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
guess: 0.0, 0.0

solver_options
tolerance: 1e-5
max_iterations: 20
loglevel: off
"#;

        let result = run_bvp_task_from_str(input).expect("BVP task should solve");
        let matrix = result
            .result
            .expect("solver should produce a result matrix");
        assert!(matrix.nrows() > 0);
        assert!(matrix.ncols() > 0);
    }

    #[test]
    fn bvp_task_runner_can_save_csv() {
        let dir = tempdir().expect("tempdir should be created");
        let csv_path = dir.path().join("bvp_task_output.csv");
        let input = format!(
            r#"
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
loglevel: off

postprocessing
save_csv: true
csv_path: {}
plot: false
"#,
            csv_path.display()
        );

        let result = run_bvp_task_from_str(&input).expect("BVP task should solve and save CSV");
        assert!(result.result.is_some());
        assert!(csv_path.exists());
    }
}
