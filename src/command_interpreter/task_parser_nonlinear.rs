//! Task-document parsing for symbolic nonlinear systems.
//!
//! This module only normalizes and validates user input. It deliberately does
//! not prepare callbacks or execute a solver; consumers can map the resulting
//! [`NonlinearTaskSpec`] onto the native nonlinear APIs.

use crate::command_interpreter::task_parser::{
    DocumentMap, DocumentParser, ParseError, TaskTelemetryMode, Value,
};
use crate::numerical::CancellationToken;
use crate::numerical::Nonlinear_systems::engine::{SolveOptions, SolveResult};
use crate::numerical::Nonlinear_systems::least_squares::{
    LeastSquaresError, MinimizationReport, SymbolicLeastSquaresSolver,
};
use crate::numerical::Nonlinear_systems::prelude::{
    BacktrackingLevenbergMarquardtMethod, DampedNewtonMethod, DampedNewtonMethodAdvanced,
    LevenbergMarquardtMethod, LevenbergMarquardtMinpack, NewtonMethod,
    NielsenLevenbergMarquardtMethod, NielsenLevenbergMarquardtMethodAdvanced,
    NonlinearSolverMethod, PowellDoglegMethod, TrustRegionLMMethod, TrustRegionMethod,
};
use crate::numerical::Nonlinear_systems::solver::NonlinearSolver;
use crate::numerical::Nonlinear_systems::symbolic::SymbolicNonlinearProblem;
use crate::numerical::Nonlinear_systems::{
    error::SolveError, least_squares::LeastSquaresTelemetryMode,
};
use crate::symbolic::substitutions::apply_symbolic_substitutions;
use crate::symbolic::symbolic_engine::Expr;
use nalgebra::DVector;
use std::collections::HashSet;
use thiserror::Error;

#[cfg(test)]
#[path = "task_parser_nonlinear_tests.rs"]
mod tests;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonlinearFormulation {
    Root,
    LeastSquares,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonlinearMethodSpec {
    Newton,
    DampedNewton,
    DampedNewtonAdvanced,
    LevenbergMarquardt,
    BacktrackingLevenbergMarquardt,
    LevenbergMarquardtMinpack,
    NielsenLevenbergMarquardt,
    NielsenLevenbergMarquardtAdvanced,
    TrustRegion,
    PowellDogleg,
    TrustRegionLm,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NonlinearProblemSpec {
    pub formulation: NonlinearFormulation,
    pub variables: Vec<String>,
    pub equations: Vec<Expr>,
    pub initial_guess: Vec<f64>,
    pub parameter_names: Vec<String>,
    pub parameter_values: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonlinearSolverOptionsSpec {
    pub tolerance: Option<f64>,
    pub residual_tolerance: Option<f64>,
    pub gradient_tolerance: Option<f64>,
    pub max_iterations: Option<usize>,
    pub telemetry: Option<TaskTelemetryMode>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NonlinearTaskSpec {
    pub problem: NonlinearProblemSpec,
    pub method: NonlinearMethodSpec,
    pub solver_options: NonlinearSolverOptionsSpec,
}

#[derive(Debug)]
pub enum NonlinearTaskRunResult {
    Root {
        result: SolveResult,
        variables: Vec<String>,
    },
    LeastSquares {
        report: MinimizationReport,
        solution: Option<DVector<f64>>,
        variables: Vec<String>,
    },
}

#[derive(Debug, Error)]
pub enum NonlinearTaskRunError {
    #[error("nonlinear symbolic root route failed: {0}")]
    Root(#[from] SolveError),
    #[error("nonlinear least-squares route failed: {0}")]
    LeastSquares(#[from] LeastSquaresError),
}

#[derive(Debug, Error)]
pub enum NonlinearTaskError {
    #[error("task document parse failed: {0}")]
    Document(#[from] ParseError),
    #[error("missing section `{0}`")]
    MissingSection(&'static str),
    #[error("missing field `{field}` in section `{section}`")]
    MissingField {
        section: &'static str,
        field: &'static str,
    },
    #[error("invalid field `{field}` in section `{section}`: {message}")]
    InvalidField {
        section: &'static str,
        field: &'static str,
        message: String,
    },
    #[error("unknown nonlinear formulation `{0}`")]
    UnknownFormulation(String),
    #[error("unknown nonlinear method `{0}`")]
    UnknownMethod(String),
    #[error(
        "root formulation requires equal equation and variable counts; got {equations} equations and {variables} variables"
    )]
    NonSquareRoot { equations: usize, variables: usize },
}

pub fn parse_nonlinear_task_from_text(
    input: &str,
) -> Result<NonlinearTaskSpec, NonlinearTaskError> {
    let mut parser = DocumentParser::new(input.to_string());
    parser.parse_document_typed()?;
    parser.try_keys_to_lower_case_typed(None)?;
    let document = parser
        .get_result()
        .ok_or_else(|| NonlinearTaskError::InvalidField {
            section: "document",
            field: "*",
            message: "parser returned no document".into(),
        })?;
    parse_nonlinear_task_from_document(document)
}

pub fn parse_nonlinear_task_from_document(
    document: &DocumentMap,
) -> Result<NonlinearTaskSpec, NonlinearTaskError> {
    reject_unknown_sections(document, &["problem", "solver"])?;
    let problem = section(document, "problem")?;
    reject_unknown_fields(
        problem,
        "problem",
        &[
            "formulation",
            "variables",
            "equations",
            "initial_guess",
            "parameters",
            "parameter_values",
            "substitutions",
        ],
    )?;
    let formulation_raw = required_string(problem, "problem", "formulation")?;
    let formulation = match formulation_raw.trim().to_ascii_lowercase().as_str() {
        "root" | "root_finding" => NonlinearFormulation::Root,
        "least_squares" | "least-squares" => NonlinearFormulation::LeastSquares,
        other => return Err(NonlinearTaskError::UnknownFormulation(other.to_string())),
    };
    let variables = required_strings(problem, "problem", "variables")?;
    let equation_strings = required_strings(problem, "problem", "equations")?;
    if equation_strings.is_empty() {
        return invalid(
            "problem",
            "equations",
            "at least one equation is required".into(),
        );
    }
    let equations = parse_expressions(&equation_strings, "problem", "equations")?;
    let initial_guess = required_floats(problem, "problem", "initial_guess")?;
    let parameter_names = optional_strings(problem, "problem", "parameters")?.unwrap_or_default();
    let parameter_values =
        optional_floats(problem, "problem", "parameter_values")?.unwrap_or_default();
    let substitutions = parse_substitutions(problem)?;

    validate_names(&variables, "variables")?;
    validate_optional_names(&parameter_names, "parameters")?;
    if variables.iter().any(|name| parameter_names.contains(name)) {
        return invalid(
            "problem",
            "parameters",
            "parameter names must not overlap solved variables".into(),
        );
    }
    if variables.len() != initial_guess.len() {
        return invalid(
            "problem",
            "initial_guess",
            format!(
                "expected {} values, got {}",
                variables.len(),
                initial_guess.len()
            ),
        );
    }
    if parameter_names.len() != parameter_values.len() {
        return invalid(
            "problem",
            "parameter_values",
            format!(
                "expected {} values for declared parameters, got {}",
                parameter_names.len(),
                parameter_values.len()
            ),
        );
    }
    if !initial_guess
        .iter()
        .chain(&parameter_values)
        .all(|v| v.is_finite())
    {
        return invalid(
            "problem",
            "initial_guess",
            "all numeric values must be finite".into(),
        );
    }
    if formulation == NonlinearFormulation::Root && equations.len() != variables.len() {
        return Err(NonlinearTaskError::NonSquareRoot {
            equations: equations.len(),
            variables: variables.len(),
        });
    }
    let allowed = variables
        .iter()
        .chain(&parameter_names)
        .map(String::as_str)
        .collect::<HashSet<_>>();
    let mut expanded_equations = equations;
    apply_symbolic_substitutions(
        &mut expanded_equations,
        substitutions,
        &variables
            .iter()
            .chain(&parameter_names)
            .cloned()
            .collect::<Vec<_>>(),
    )
    .map_err(|error| NonlinearTaskError::InvalidField {
        section: "problem",
        field: "substitutions",
        message: error.to_string(),
    })?;
    for (index, equation) in expanded_equations.iter().enumerate() {
        let unknown = equation
            .all_arguments_are_variables()
            .into_iter()
            .filter(|name| !allowed.contains(name.as_str()))
            .collect::<Vec<_>>();
        if !unknown.is_empty() {
            return invalid(
                "problem",
                "equations",
                format!(
                    "equation {index} references undeclared symbol(s): {}",
                    unknown.join(", ")
                ),
            );
        }
    }

    let solver = section(document, "solver")?;
    reject_unknown_fields(
        solver,
        "solver",
        &[
            "method",
            "tolerance",
            "residual_tolerance",
            "gradient_tolerance",
            "max_iterations",
            "telemetry",
        ],
    )?;
    let method = parse_method(&required_string(solver, "solver", "method")?)?;
    if formulation == NonlinearFormulation::LeastSquares
        && method != NonlinearMethodSpec::LevenbergMarquardt
    {
        return invalid(
            "solver",
            "method",
            "selected method is a root solver; choose a least-squares method".into(),
        );
    }
    let options = NonlinearSolverOptionsSpec {
        tolerance: optional_positive_float(solver, "solver", "tolerance")?,
        residual_tolerance: optional_positive_float(solver, "solver", "residual_tolerance")?,
        gradient_tolerance: optional_positive_float(solver, "solver", "gradient_tolerance")?,
        max_iterations: optional_positive_usize(solver, "solver", "max_iterations")?,
        telemetry: optional_telemetry(solver)?,
    };
    if formulation == NonlinearFormulation::Root
        && (options.residual_tolerance.is_some() || options.gradient_tolerance.is_some())
    {
        return invalid(
            "solver",
            "residual_tolerance",
            "residual_tolerance and gradient_tolerance are currently supported only by least_squares"
                .into(),
        );
    }
    Ok(NonlinearTaskSpec {
        problem: NonlinearProblemSpec {
            formulation,
            variables,
            equations: expanded_equations,
            initial_guess,
            parameter_names,
            parameter_values,
        },
        method,
        solver_options: options,
    })
}

/// Build and execute one already validated nonlinear task through the native
/// root or rectangular least-squares API. No Python or generic task runner is
/// involved in this family-specific dispatch.
pub fn run_nonlinear_task(
    spec: NonlinearTaskSpec,
) -> Result<NonlinearTaskRunResult, NonlinearTaskRunError> {
    run_nonlinear_task_inner(spec, None)
}

/// Execute one nonlinear task with cooperative cancellation at the native
/// solver's safe iteration/trial boundaries.
pub fn run_nonlinear_task_with_cancellation_token(
    spec: NonlinearTaskSpec,
    cancellation: &CancellationToken,
) -> Result<NonlinearTaskRunResult, NonlinearTaskRunError> {
    run_nonlinear_task_inner(spec, Some(cancellation))
}

fn run_nonlinear_task_inner(
    spec: NonlinearTaskSpec,
    cancellation: Option<&CancellationToken>,
) -> Result<NonlinearTaskRunResult, NonlinearTaskRunError> {
    let variables = spec.problem.variables.clone();
    match spec.problem.formulation {
        NonlinearFormulation::Root => {
            let parameters = (!spec.problem.parameter_names.is_empty())
                .then(|| DVector::from_vec(spec.problem.parameter_values.clone()));
            let problem = SymbolicNonlinearProblem::from_expressions(
                spec.problem.equations,
                Some(spec.problem.variables.clone()),
                (!spec.problem.parameter_names.is_empty()).then_some(spec.problem.parameter_names),
                parameters,
            )?;
            let method = native_root_method(spec.method);
            let mut options = SolveOptions {
                tolerance: spec.solver_options.tolerance.unwrap_or(1e-6),
                max_iterations: spec.solver_options.max_iterations.unwrap_or(100),
                ..SolveOptions::default()
            };
            options.diagnostics.telemetry_mode = map_root_telemetry(spec.solver_options.telemetry);
            let solver = NonlinearSolver::Root(method);
            let initial_guess = DVector::from_vec(spec.problem.initial_guess);
            let result = match cancellation {
                Some(token) => solver.solve_root_with_cancellation_token(
                    &problem,
                    initial_guess,
                    options,
                    token.clone(),
                )?,
                None => solver.solve_root(&problem, initial_guess, options)?,
            };
            Ok(NonlinearTaskRunResult::Root { result, variables })
        }
        NonlinearFormulation::LeastSquares => {
            let mut solver = SymbolicLeastSquaresSolver::new()
                .with_equations(spec.problem.equations)
                .with_unknowns(spec.problem.variables)
                .with_initial_guess(spec.problem.initial_guess)
                .with_telemetry(map_least_squares_telemetry(spec.solver_options.telemetry));
            if !spec.problem.parameter_names.is_empty() {
                solver = solver.with_parameter_names(spec.problem.parameter_names);
            }
            if let Some(value) = spec.solver_options.tolerance {
                solver = solver.with_tolerance(value);
            }
            if let Some(value) = spec.solver_options.residual_tolerance {
                solver = solver.with_f_tolerance(value);
            }
            if let Some(value) = spec.solver_options.gradient_tolerance {
                solver = solver.with_g_tolerance(value);
            }
            if let Some(value) = spec.solver_options.max_iterations {
                solver = solver.with_max_iterations(value);
            }
            let mut solver = solver.try_build()?;
            let report = match (spec.problem.parameter_values.is_empty(), cancellation) {
                (true, Some(token)) => solver.try_solve_with_cancellation_token(token.clone())?,
                (false, Some(token)) => solver
                    .try_solve_with_parameter_values_and_cancellation_token(
                        spec.problem.parameter_values,
                        token.clone(),
                    )?,
                (true, None) => solver.try_solve()?,
                (false, None) => {
                    solver.try_solve_with_parameter_values(spec.problem.parameter_values)?
                }
            };
            let solution = solver.result.clone();
            Ok(NonlinearTaskRunResult::LeastSquares {
                report,
                solution,
                variables,
            })
        }
    }
}

fn optional_telemetry(
    solver: &std::collections::HashMap<String, Option<Vec<Value>>>,
) -> Result<Option<TaskTelemetryMode>, NonlinearTaskError> {
    solver
        .get("telemetry")
        .and_then(Option::as_deref)
        .map(|values| match values {
            [Value::String(raw)] => Ok(raw.clone()),
            _ => invalid("solver", "telemetry", "expected a string".into()),
        })
        .transpose()?
        .map(|raw| match raw.trim().to_ascii_lowercase().as_str() {
            "off" | "disabled" => Ok(TaskTelemetryMode::Off),
            "counters" | "counter" => Ok(TaskTelemetryMode::Counters),
            "timings" | "timing" => Ok(TaskTelemetryMode::Timings),
            "detailed" | "full" => Ok(TaskTelemetryMode::Detailed),
            _ => invalid(
                "solver",
                "telemetry",
                format!("unknown mode `{raw}`; use off, counters, timings or detailed"),
            ),
        })
        .transpose()
}

fn map_root_telemetry(mode: Option<TaskTelemetryMode>) -> crate::numerical::SolverTelemetryMode {
    use crate::numerical::SolverTelemetryMode as Native;
    match mode.unwrap_or(TaskTelemetryMode::Off) {
        TaskTelemetryMode::Off => Native::Off,
        TaskTelemetryMode::Counters => Native::Counters,
        TaskTelemetryMode::Timings => Native::Timings,
        TaskTelemetryMode::Detailed => Native::Detailed,
    }
}

fn map_least_squares_telemetry(mode: Option<TaskTelemetryMode>) -> LeastSquaresTelemetryMode {
    match mode.unwrap_or(TaskTelemetryMode::Off) {
        TaskTelemetryMode::Off => LeastSquaresTelemetryMode::Off,
        TaskTelemetryMode::Counters => LeastSquaresTelemetryMode::Counters,
        TaskTelemetryMode::Timings => LeastSquaresTelemetryMode::Timings,
        TaskTelemetryMode::Detailed => LeastSquaresTelemetryMode::Detailed,
    }
}

fn native_root_method(method: NonlinearMethodSpec) -> NonlinearSolverMethod {
    match method {
        NonlinearMethodSpec::Newton => NonlinearSolverMethod::Newton(NewtonMethod),
        NonlinearMethodSpec::DampedNewton => {
            NonlinearSolverMethod::DampedNewton(DampedNewtonMethod::default())
        }
        NonlinearMethodSpec::DampedNewtonAdvanced => {
            NonlinearSolverMethod::DampedNewtonAdvanced(DampedNewtonMethodAdvanced::default())
        }
        NonlinearMethodSpec::LevenbergMarquardt => {
            NonlinearSolverMethod::LevenbergMarquardt(LevenbergMarquardtMethod::default())
        }
        NonlinearMethodSpec::BacktrackingLevenbergMarquardt => {
            NonlinearSolverMethod::BacktrackingLevenbergMarquardt(
                BacktrackingLevenbergMarquardtMethod::default(),
            )
        }
        NonlinearMethodSpec::LevenbergMarquardtMinpack => {
            NonlinearSolverMethod::LevenbergMarquardtMinpack(LevenbergMarquardtMinpack::default())
        }
        NonlinearMethodSpec::NielsenLevenbergMarquardt => {
            NonlinearSolverMethod::NielsenLevenbergMarquardt(
                NielsenLevenbergMarquardtMethod::default(),
            )
        }
        NonlinearMethodSpec::NielsenLevenbergMarquardtAdvanced => {
            NonlinearSolverMethod::NielsenLevenbergMarquardtAdvanced(
                NielsenLevenbergMarquardtMethodAdvanced::default(),
            )
        }
        NonlinearMethodSpec::TrustRegion => {
            NonlinearSolverMethod::TrustRegion(TrustRegionMethod::default())
        }
        NonlinearMethodSpec::PowellDogleg => {
            NonlinearSolverMethod::PowellDogleg(PowellDoglegMethod::default())
        }
        NonlinearMethodSpec::TrustRegionLm => {
            NonlinearSolverMethod::TrustRegionLM(TrustRegionLMMethod::default())
        }
    }
}

fn parse_method(value: &str) -> Result<NonlinearMethodSpec, NonlinearTaskError> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "newton" => Ok(NonlinearMethodSpec::Newton),
        "damped_newton" => Ok(NonlinearMethodSpec::DampedNewton),
        "damped_newton_advanced" => Ok(NonlinearMethodSpec::DampedNewtonAdvanced),
        "lm" | "levenberg_marquardt" => Ok(NonlinearMethodSpec::LevenbergMarquardt),
        "backtracking_lm" | "backtracking_levenberg_marquardt" => {
            Ok(NonlinearMethodSpec::BacktrackingLevenbergMarquardt)
        }
        "lm_minpack" | "levenberg_marquardt_minpack" => {
            Ok(NonlinearMethodSpec::LevenbergMarquardtMinpack)
        }
        "nielsen_lm" | "nielsen_levenberg_marquardt" => {
            Ok(NonlinearMethodSpec::NielsenLevenbergMarquardt)
        }
        "nielsen_lm_advanced" | "nielsen_levenberg_marquardt_advanced" => {
            Ok(NonlinearMethodSpec::NielsenLevenbergMarquardtAdvanced)
        }
        "trust_region" => Ok(NonlinearMethodSpec::TrustRegion),
        "powell_dogleg" => Ok(NonlinearMethodSpec::PowellDogleg),
        "trust_region_lm" => Ok(NonlinearMethodSpec::TrustRegionLm),
        other => Err(NonlinearTaskError::UnknownMethod(other.to_string())),
    }
}

fn section<'a>(
    document: &'a DocumentMap,
    name: &'static str,
) -> Result<&'a std::collections::HashMap<String, Option<Vec<Value>>>, NonlinearTaskError> {
    document
        .get(name)
        .ok_or(NonlinearTaskError::MissingSection(name))
}
fn field<'a>(
    section: &'a std::collections::HashMap<String, Option<Vec<Value>>>,
    name: &'static str,
    section_name: &'static str,
) -> Result<&'a [Value], NonlinearTaskError> {
    section
        .get(name)
        .and_then(Option::as_deref)
        .ok_or(NonlinearTaskError::MissingField {
            section: section_name,
            field: name,
        })
}
fn required_string(
    section: &std::collections::HashMap<String, Option<Vec<Value>>>,
    section_name: &'static str,
    name: &'static str,
) -> Result<String, NonlinearTaskError> {
    match field(section, name, section_name)? {
        [Value::String(v)] => Ok(v.clone()),
        _ => invalid(section_name, name, "expected a string".into()),
    }
}
fn required_strings(
    section: &std::collections::HashMap<String, Option<Vec<Value>>>,
    section_name: &'static str,
    name: &'static str,
) -> Result<Vec<String>, NonlinearTaskError> {
    field(section, name, section_name)?
        .iter()
        .map(|v| match v {
            Value::String(s) => Ok(s.clone()),
            _ => invalid(section_name, name, "expected a list of strings".into()),
        })
        .collect()
}
fn optional_strings(
    section: &std::collections::HashMap<String, Option<Vec<Value>>>,
    section_name: &'static str,
    name: &'static str,
) -> Result<Option<Vec<String>>, NonlinearTaskError> {
    section
        .get(name)
        .and_then(Option::as_deref)
        .map(|values| {
            values
                .iter()
                .map(|v| match v {
                    Value::String(s) => Ok(s.clone()),
                    _ => invalid(section_name, name, "expected a list of strings".into()),
                })
                .collect()
        })
        .transpose()
}
fn required_floats(
    section: &std::collections::HashMap<String, Option<Vec<Value>>>,
    section_name: &'static str,
    name: &'static str,
) -> Result<Vec<f64>, NonlinearTaskError> {
    let values = field(section, name, section_name)?;
    if values.len() == 1 {
        if let Value::Vector(v) = &values[0] {
            return Ok(v.clone());
        }
    }
    values
        .iter()
        .map(|v| match v {
            Value::Float(x) => Ok(*x),
            Value::Integer(x) => Ok(*x as f64),
            Value::Usize(x) => Ok(*x as f64),
            _ => invalid(section_name, name, "expected numeric values".into()),
        })
        .collect()
}
fn optional_floats(
    section: &std::collections::HashMap<String, Option<Vec<Value>>>,
    section_name: &'static str,
    name: &'static str,
) -> Result<Option<Vec<f64>>, NonlinearTaskError> {
    section
        .get(name)
        .and_then(Option::as_deref)
        .map(|_| required_floats(section, section_name, name))
        .transpose()
}
fn optional_positive_float(
    section: &std::collections::HashMap<String, Option<Vec<Value>>>,
    section_name: &'static str,
    name: &'static str,
) -> Result<Option<f64>, NonlinearTaskError> {
    let Some(values) = section.get(name).and_then(Option::as_deref) else {
        return Ok(None);
    };
    let value = match values {
        [Value::Float(x)] => *x,
        [Value::Integer(x)] => *x as f64,
        _ => return invalid(section_name, name, "expected one positive number".into()),
    };
    if !value.is_finite() || value <= 0.0 {
        return invalid(
            section_name,
            name,
            "value must be finite and positive".into(),
        );
    }
    Ok(Some(value))
}
fn optional_positive_usize(
    section: &std::collections::HashMap<String, Option<Vec<Value>>>,
    section_name: &'static str,
    name: &'static str,
) -> Result<Option<usize>, NonlinearTaskError> {
    let Some(values) = section.get(name).and_then(Option::as_deref) else {
        return Ok(None);
    };
    let value = match values {
        [Value::Usize(x)] => *x,
        [Value::Integer(x)] if *x > 0 => *x as usize,
        _ => return invalid(section_name, name, "expected a positive integer".into()),
    };
    if value == 0 {
        return invalid(section_name, name, "value must be positive".into());
    }
    Ok(Some(value))
}
fn parse_expressions(
    values: &[String],
    section: &'static str,
    field_name: &'static str,
) -> Result<Vec<Expr>, NonlinearTaskError> {
    values
        .iter()
        .enumerate()
        .map(|(i, text)| {
            Expr::try_parse_expression(text).map_err(|message| NonlinearTaskError::InvalidField {
                section,
                field: field_name,
                message: format!("expression {i}: {message}"),
            })
        })
        .collect()
}
fn parse_substitutions(
    section: &std::collections::HashMap<String, Option<Vec<Value>>>,
) -> Result<Vec<(String, Expr)>, NonlinearTaskError> {
    let entries = optional_strings(section, "problem", "substitutions")?.unwrap_or_default();
    let mut definitions = Vec::with_capacity(entries.len());
    for (index, entry) in entries.into_iter().enumerate() {
        let Some((name, expression)) = entry.split_once('=') else {
            return invalid(
                "problem",
                "substitutions",
                format!("entry {index} must use `name=expression` form"),
            );
        };
        let name = name.trim().to_string();
        let expression = Expr::try_parse_expression(expression.trim()).map_err(|error| {
            NonlinearTaskError::InvalidField {
                section: "problem",
                field: "substitutions",
                message: format!("entry {index} expression: {error}"),
            }
        })?;
        definitions.push((name, expression));
    }
    let names = definitions
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    let mut unique = HashSet::with_capacity(names.len());
    if names
        .iter()
        .any(|name| name.trim().is_empty() || !unique.insert(name))
    {
        return invalid(
            "problem",
            "substitutions",
            "alias names must be non-empty and unique".into(),
        );
    }
    Ok(definitions)
}
fn validate_names(names: &[String], field_name: &'static str) -> Result<(), NonlinearTaskError> {
    if names.is_empty() {
        return invalid(
            "problem",
            field_name,
            "at least one name is required".into(),
        );
    }
    let mut unique = HashSet::with_capacity(names.len());
    if names
        .iter()
        .any(|name| name.trim().is_empty() || !unique.insert(name))
    {
        return invalid(
            "problem",
            field_name,
            "names must be non-empty and unique".into(),
        );
    }
    Ok(())
}
fn validate_optional_names(
    names: &[String],
    field_name: &'static str,
) -> Result<(), NonlinearTaskError> {
    if names.is_empty() {
        return Ok(());
    }
    validate_names(names, field_name)
}
fn reject_unknown_fields(
    section: &std::collections::HashMap<String, Option<Vec<Value>>>,
    section_name: &'static str,
    allowed: &[&'static str],
) -> Result<(), NonlinearTaskError> {
    if let Some(name) = section
        .keys()
        .find(|name| !allowed.contains(&name.as_str()))
    {
        return invalid(section_name, "*", format!("unknown field `{name}`"));
    }
    Ok(())
}
fn reject_unknown_sections(
    document: &DocumentMap,
    allowed: &[&'static str],
) -> Result<(), NonlinearTaskError> {
    if let Some(name) = document
        .keys()
        .find(|name| !allowed.contains(&name.as_str()))
    {
        return invalid("document", "*", format!("unknown section `{name}`"));
    }
    Ok(())
}
fn invalid<T>(
    section: &'static str,
    field: &'static str,
    message: String,
) -> Result<T, NonlinearTaskError> {
    Err(NonlinearTaskError::InvalidField {
        section,
        field,
        message,
    })
}
