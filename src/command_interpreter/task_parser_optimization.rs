//! Task-document parsing for symbolic data-fitting and optimization jobs.
//!
//! The parser produces a validated description only. It does not trigger
//! symbolic preparation, compilation, or a numerical fit.

use crate::command_interpreter::task_parser::{
    DocumentMap, DocumentParser, ParseError, TaskTelemetryMode, Value,
};
use crate::numerical::CancellationToken;
use crate::numerical::Nonlinear_systems::least_squares::LeastSquaresTelemetryMode;
use crate::numerical::optimization::universal_fitting::{
    Method as FittingMethod, UniversalFitting, UniversalFittingError, UniversalFittingResult,
};
use crate::symbolic::symbolic_engine::Expr;
use std::collections::HashSet;
use thiserror::Error;

#[cfg(test)]
#[path = "task_parser_optimization_tests.rs"]
mod tests;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationMethodSpec {
    LevenbergMarquardt,
    Gavin,
    VarPro,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationModelSpec {
    EquationModel {
        equations: Vec<Expr>,
        parameter_names: Vec<String>,
        initial_guess: Vec<f64>,
    },
    VarProModel {
        parameter_names: Vec<String>,
        initial_guess: Vec<f64>,
        basis: Vec<(String, Expr)>,
        invariants: Vec<Expr>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct OptimizationDataSpec {
    pub arg: String,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptimizationSolverOptionsSpec {
    pub tolerance: Option<f64>,
    pub objective_tolerance: Option<f64>,
    pub gradient_tolerance: Option<f64>,
    pub max_iterations: Option<usize>,
    pub telemetry: Option<TaskTelemetryMode>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OptimizationTaskSpec {
    pub method: OptimizationMethodSpec,
    pub data: OptimizationDataSpec,
    pub model: OptimizationModelSpec,
    pub solver_options: OptimizationSolverOptionsSpec,
    pub gavin_weights: Option<Vec<f64>>,
    pub gavin_positive_parameters: Vec<String>,
}

/// Execute one validated task through the canonical fitting facade.
pub fn run_optimization_task(
    spec: OptimizationTaskSpec,
) -> Result<UniversalFittingResult, OptimizationTaskRunError> {
    run_optimization_task_inner(spec, None)
}

/// Execute an LM or VarPro fitting task with cooperative cancellation.
/// Gavin retains its explicit native unsupported-option error.
pub fn run_optimization_task_with_cancellation_token(
    spec: OptimizationTaskSpec,
    cancellation: &CancellationToken,
) -> Result<UniversalFittingResult, OptimizationTaskRunError> {
    run_optimization_task_inner(spec, Some(cancellation))
}

fn run_optimization_task_inner(
    spec: OptimizationTaskSpec,
    cancellation: Option<&CancellationToken>,
) -> Result<UniversalFittingResult, OptimizationTaskRunError> {
    let method = match spec.method {
        OptimizationMethodSpec::LevenbergMarquardt => FittingMethod::LM,
        OptimizationMethodSpec::Gavin => FittingMethod::GAVIN,
        OptimizationMethodSpec::VarPro => FittingMethod::VARPRO,
    };
    let mut fitting = UniversalFitting::new()
        .with_method(method)
        .with_data(spec.data.x, spec.data.y)
        .with_arg(spec.data.arg);
    if let Some(value) = spec.solver_options.tolerance {
        fitting = fitting.with_tolerance(value);
    }
    if let Some(value) = spec.solver_options.objective_tolerance {
        fitting = fitting.with_f_tolerance(value);
    }
    if let Some(value) = spec.solver_options.gradient_tolerance {
        fitting = fitting.with_g_tolerance(value);
    }
    if let Some(value) = spec.solver_options.max_iterations {
        fitting = fitting.with_max_iterations(value);
    }
    match spec
        .solver_options
        .telemetry
        .unwrap_or(TaskTelemetryMode::Off)
    {
        TaskTelemetryMode::Off => {}
        TaskTelemetryMode::Counters => {
            if spec.method == OptimizationMethodSpec::Gavin {
                fitting =
                    fitting.with_gavin_telemetry(crate::numerical::SolverTelemetryMode::Counters);
            } else {
                fitting = fitting.with_telemetry(LeastSquaresTelemetryMode::Counters);
            }
        }
        TaskTelemetryMode::Timings => {
            if spec.method == OptimizationMethodSpec::Gavin {
                fitting =
                    fitting.with_gavin_telemetry(crate::numerical::SolverTelemetryMode::Timings);
            } else {
                fitting = fitting.with_telemetry(LeastSquaresTelemetryMode::Timings);
            }
        }
        TaskTelemetryMode::Detailed => {
            if spec.method == OptimizationMethodSpec::Gavin {
                fitting =
                    fitting.with_gavin_telemetry(crate::numerical::SolverTelemetryMode::Detailed);
            } else {
                fitting = fitting.with_telemetry(LeastSquaresTelemetryMode::Detailed);
            }
        }
    }
    match spec.model {
        OptimizationModelSpec::EquationModel {
            equations,
            parameter_names,
            initial_guess,
        } => {
            fitting = fitting
                .with_unknowns(parameter_names)
                .with_initial_guess(initial_guess)
                .with_equations(equations);
        }
        OptimizationModelSpec::VarProModel {
            parameter_names,
            initial_guess,
            basis,
            invariants,
        } => {
            fitting = fitting
                .with_parameters(parameter_names)
                .with_initial_guess(initial_guess)
                .with_equations(invariants);
            for (name, expression) in basis {
                fitting = fitting.with_basis_expr(name, expression);
            }
        }
    }
    if let Some(weights) = spec.gavin_weights {
        fitting = fitting.with_gavin_weights(weights);
    }
    if !spec.gavin_positive_parameters.is_empty() {
        fitting = fitting.with_gavin_positive_parameters(spec.gavin_positive_parameters);
    }
    match cancellation {
        Some(token) => fitting.build_with_cancellation_token(token.clone()),
        None => fitting.build(),
    }
    .map_err(OptimizationTaskRunError::from)
}

#[derive(Debug, Error)]
pub enum OptimizationTaskError {
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
    #[error("unknown optimization method `{0}`")]
    UnknownMethod(String),
    #[error("method `{method}` does not accept field `{field}`")]
    UnsupportedField {
        method: &'static str,
        field: &'static str,
    },
}

#[derive(Debug, Error)]
#[error("optimization task execution failed: {0}")]
pub struct OptimizationTaskRunError(#[from] pub UniversalFittingError);

pub fn parse_optimization_task_from_text(
    input: &str,
) -> Result<OptimizationTaskSpec, OptimizationTaskError> {
    let mut parser = DocumentParser::new(input.to_string());
    parser.parse_document_typed()?;
    parser.try_keys_to_lower_case_typed(None)?;
    let document = parser
        .get_result()
        .ok_or_else(|| OptimizationTaskError::InvalidField {
            section: "document",
            field: "*",
            message: "parser returned no document".into(),
        })?;
    parse_optimization_task_from_document(document)
}

pub fn parse_optimization_task_from_document(
    document: &DocumentMap,
) -> Result<OptimizationTaskSpec, OptimizationTaskError> {
    reject_unknown_sections(document, &["data", "model", "solver"])?;
    let model_section = section(document, "model")?;
    reject_unknown_fields(
        model_section,
        "model",
        &[
            "method",
            "parameters",
            "initial_guess",
            "equations",
            "basis_parameters",
            "basis_equations",
            "invariants",
        ],
    )?;
    let method_raw = required_string(model_section, "model", "method")?;
    let method = match method_raw
        .trim()
        .to_ascii_lowercase()
        .replace('-', "_")
        .as_str()
    {
        "lm" | "levenberg_marquardt" => OptimizationMethodSpec::LevenbergMarquardt,
        "gavin" | "gavin_lm" => OptimizationMethodSpec::Gavin,
        "varpro" | "variable_projection" => OptimizationMethodSpec::VarPro,
        other => return Err(OptimizationTaskError::UnknownMethod(other.to_string())),
    };
    let data_section = section(document, "data")?;
    reject_unknown_fields(data_section, "data", &["arg", "x", "y"])?;
    let arg = optional_string(data_section, "data", "arg")?.unwrap_or_else(|| "x".to_string());
    if arg.trim().is_empty() {
        return invalid("data", "arg", "must not be empty".into());
    }
    let x = required_floats(data_section, "data", "x")?;
    let y = required_floats(data_section, "data", "y")?;
    if x.is_empty() || x.len() != y.len() {
        return invalid(
            "data",
            "y",
            format!(
                "x and y must have the same non-zero length ({} vs {})",
                x.len(),
                y.len()
            ),
        );
    }
    if !x.iter().chain(&y).all(|v| v.is_finite()) {
        return invalid("data", "x", "all sample values must be finite".into());
    }

    let (model, symbols) = match method {
        OptimizationMethodSpec::LevenbergMarquardt | OptimizationMethodSpec::Gavin => {
            reject_fields(
                model_section,
                method,
                &["basis_parameters", "basis_equations", "invariants"],
            )?;
            let parameter_names = required_strings(model_section, "model", "parameters")?;
            let initial_guess = required_floats(model_section, "model", "initial_guess")?;
            validate_names(&parameter_names, "model", "parameters", false)?;
            if parameter_names.len() != initial_guess.len() {
                return invalid(
                    "model",
                    "initial_guess",
                    format!(
                        "expected {} values, got {}",
                        parameter_names.len(),
                        initial_guess.len()
                    ),
                );
            }
            if !initial_guess.iter().all(|v| v.is_finite()) {
                return invalid("model", "initial_guess", "all values must be finite".into());
            }
            let equation_strings = required_strings(model_section, "model", "equations")?;
            if equation_strings.is_empty() {
                return invalid(
                    "model",
                    "equations",
                    "at least one model equation is required".into(),
                );
            }
            let equations = parse_expressions(&equation_strings, "model", "equations")?;
            let allowed = parameter_names
                .iter()
                .map(String::as_str)
                .chain(std::iter::once(arg.as_str()))
                .collect::<HashSet<_>>();
            validate_symbols(&equations, &allowed, "equations")?;
            let names = parameter_names.clone();
            (
                OptimizationModelSpec::EquationModel {
                    equations,
                    parameter_names,
                    initial_guess,
                },
                names,
            )
        }
        OptimizationMethodSpec::VarPro => {
            reject_fields(model_section, method, &["equations"])?;
            let parameter_names = required_strings(model_section, "model", "parameters")?;
            let initial_guess = required_floats(model_section, "model", "initial_guess")?;
            validate_names(&parameter_names, "model", "parameters", false)?;
            if parameter_names.len() != initial_guess.len() {
                return invalid(
                    "model",
                    "initial_guess",
                    format!(
                        "expected {} values, got {}",
                        parameter_names.len(),
                        initial_guess.len()
                    ),
                );
            }
            if !initial_guess.iter().all(|v| v.is_finite()) {
                return invalid("model", "initial_guess", "all values must be finite".into());
            }
            let basis_names = required_strings(model_section, "model", "basis_parameters")?;
            let basis_text = required_strings(model_section, "model", "basis_equations")?;
            if basis_names.len() != basis_text.len() || basis_names.is_empty() {
                return invalid(
                    "model",
                    "basis_equations",
                    "basis_parameters and basis_equations must have equal non-zero lengths".into(),
                );
            }
            validate_names(&basis_names, "model", "basis_parameters", false)?;
            if basis_names
                .iter()
                .any(|name| !parameter_names.contains(name))
            {
                return invalid(
                    "model",
                    "basis_parameters",
                    "each basis parameter must be declared in parameters".into(),
                );
            }
            let basis_exprs = parse_expressions(&basis_text, "model", "basis_equations")?;
            let invariant_text =
                optional_strings(model_section, "model", "invariants")?.unwrap_or_default();
            let invariants = parse_expressions(&invariant_text, "model", "invariants")?;
            let allowed = parameter_names
                .iter()
                .map(String::as_str)
                .chain(std::iter::once(arg.as_str()))
                .collect::<HashSet<_>>();
            validate_symbols(&basis_exprs, &allowed, "basis_equations")?;
            validate_symbols(&invariants, &HashSet::from([arg.as_str()]), "invariants")?;
            let basis = basis_names.into_iter().zip(basis_exprs).collect();
            let names = parameter_names.clone();
            (
                OptimizationModelSpec::VarProModel {
                    parameter_names,
                    initial_guess,
                    basis,
                    invariants,
                },
                names,
            )
        }
    };

    let solver = document.get("solver");
    let empty = std::collections::HashMap::new();
    let solver = solver.unwrap_or(&empty);
    reject_unknown_fields(
        solver,
        "solver",
        &[
            "tolerance",
            "objective_tolerance",
            "gradient_tolerance",
            "max_iterations",
            "gavin_weights",
            "gavin_positive_parameters",
            "telemetry",
        ],
    )?;
    let solver_options = OptimizationSolverOptionsSpec {
        tolerance: optional_positive_float(solver, "solver", "tolerance")?,
        objective_tolerance: optional_positive_float(solver, "solver", "objective_tolerance")?,
        gradient_tolerance: optional_positive_float(solver, "solver", "gradient_tolerance")?,
        max_iterations: optional_positive_usize(solver, "solver", "max_iterations")?,
        telemetry: optional_telemetry(solver)?,
    };
    let gavin_weights = optional_floats(solver, "solver", "gavin_weights")?;
    if let Some(weights) = &gavin_weights {
        if method != OptimizationMethodSpec::Gavin {
            return Err(OptimizationTaskError::UnsupportedField {
                method: method_label(method),
                field: "gavin_weights",
            });
        }
        if weights.len() != x.len() || !weights.iter().all(|w| w.is_finite() && *w >= 0.0) {
            return invalid(
                "solver",
                "gavin_weights",
                "weights must be finite, non-negative, and match the observation count".into(),
            );
        }
    }
    let gavin_positive_parameters =
        optional_strings(solver, "solver", "gavin_positive_parameters")?.unwrap_or_default();
    if !gavin_positive_parameters.is_empty() {
        if method != OptimizationMethodSpec::Gavin {
            return Err(OptimizationTaskError::UnsupportedField {
                method: method_label(method),
                field: "gavin_positive_parameters",
            });
        }
        if gavin_positive_parameters
            .iter()
            .any(|name| !symbols.contains(name))
        {
            return invalid(
                "solver",
                "gavin_positive_parameters",
                "each name must refer to a declared model parameter".into(),
            );
        }
    }
    Ok(OptimizationTaskSpec {
        method,
        data: OptimizationDataSpec { arg, x, y },
        model,
        solver_options,
        gavin_weights,
        gavin_positive_parameters,
    })
}

fn optional_telemetry(
    solver: &std::collections::HashMap<String, Option<Vec<Value>>>,
) -> Result<Option<TaskTelemetryMode>, OptimizationTaskError> {
    optional_string(solver, "solver", "telemetry")?
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

fn method_label(method: OptimizationMethodSpec) -> &'static str {
    match method {
        OptimizationMethodSpec::LevenbergMarquardt => "lm",
        OptimizationMethodSpec::Gavin => "gavin",
        OptimizationMethodSpec::VarPro => "varpro",
    }
}
fn reject_fields(
    section: &std::collections::HashMap<String, Option<Vec<Value>>>,
    method: OptimizationMethodSpec,
    names: &[&'static str],
) -> Result<(), OptimizationTaskError> {
    if let Some(name) = names.iter().find(|name| section.contains_key(**name)) {
        return Err(OptimizationTaskError::UnsupportedField {
            method: method_label(method),
            field: name,
        });
    }
    Ok(())
}
fn reject_unknown_fields(
    section: &std::collections::HashMap<String, Option<Vec<Value>>>,
    section_name: &'static str,
    allowed: &[&'static str],
) -> Result<(), OptimizationTaskError> {
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
) -> Result<(), OptimizationTaskError> {
    if let Some(name) = document
        .keys()
        .find(|name| !allowed.contains(&name.as_str()))
    {
        return invalid("document", "*", format!("unknown section `{name}`"));
    }
    Ok(())
}
fn section<'a>(
    document: &'a DocumentMap,
    name: &'static str,
) -> Result<&'a std::collections::HashMap<String, Option<Vec<Value>>>, OptimizationTaskError> {
    document
        .get(name)
        .ok_or(OptimizationTaskError::MissingSection(name))
}
fn field<'a>(
    section: &'a std::collections::HashMap<String, Option<Vec<Value>>>,
    name: &'static str,
    section_name: &'static str,
) -> Result<&'a [Value], OptimizationTaskError> {
    section
        .get(name)
        .and_then(Option::as_deref)
        .ok_or(OptimizationTaskError::MissingField {
            section: section_name,
            field: name,
        })
}
fn required_string(
    section: &std::collections::HashMap<String, Option<Vec<Value>>>,
    section_name: &'static str,
    name: &'static str,
) -> Result<String, OptimizationTaskError> {
    match field(section, name, section_name)? {
        [Value::String(v)] => Ok(v.clone()),
        _ => invalid(section_name, name, "expected a string".into()),
    }
}
fn optional_string(
    section: &std::collections::HashMap<String, Option<Vec<Value>>>,
    section_name: &'static str,
    name: &'static str,
) -> Result<Option<String>, OptimizationTaskError> {
    section
        .get(name)
        .and_then(Option::as_deref)
        .map(|v| match v {
            [Value::String(s)] => Ok(s.clone()),
            _ => invalid(section_name, name, "expected a string".into()),
        })
        .transpose()
}
fn required_strings(
    section: &std::collections::HashMap<String, Option<Vec<Value>>>,
    section_name: &'static str,
    name: &'static str,
) -> Result<Vec<String>, OptimizationTaskError> {
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
) -> Result<Option<Vec<String>>, OptimizationTaskError> {
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
) -> Result<Vec<f64>, OptimizationTaskError> {
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
) -> Result<Option<Vec<f64>>, OptimizationTaskError> {
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
) -> Result<Option<f64>, OptimizationTaskError> {
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
) -> Result<Option<usize>, OptimizationTaskError> {
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
    field: &'static str,
) -> Result<Vec<Expr>, OptimizationTaskError> {
    values
        .iter()
        .enumerate()
        .map(|(i, text)| {
            Expr::try_parse_expression(text).map_err(|message| {
                OptimizationTaskError::InvalidField {
                    section,
                    field,
                    message: format!("expression {i}: {message}"),
                }
            })
        })
        .collect()
}
fn validate_symbols(
    expressions: &[Expr],
    allowed: &HashSet<&str>,
    field: &'static str,
) -> Result<(), OptimizationTaskError> {
    for (index, expression) in expressions.iter().enumerate() {
        let unknown = expression
            .all_arguments_are_variables()
            .into_iter()
            .filter(|name| !allowed.contains(name.as_str()))
            .collect::<Vec<_>>();
        if !unknown.is_empty() {
            return invalid(
                "model",
                field,
                format!(
                    "expression {index} references undeclared symbol(s): {}",
                    unknown.join(", ")
                ),
            );
        }
    }
    Ok(())
}
fn validate_names(
    names: &[String],
    section: &'static str,
    field: &'static str,
    allow_empty: bool,
) -> Result<(), OptimizationTaskError> {
    if names.is_empty() && !allow_empty {
        return invalid(section, field, "at least one name is required".into());
    }
    let mut unique = HashSet::with_capacity(names.len());
    if names
        .iter()
        .any(|name| name.trim().is_empty() || !unique.insert(name))
    {
        return invalid(section, field, "names must be non-empty and unique".into());
    }
    Ok(())
}
fn invalid<T>(
    section: &'static str,
    field: &'static str,
    message: String,
) -> Result<T, OptimizationTaskError> {
    Err(OptimizationTaskError::InvalidField {
        section,
        field,
        message,
    })
}
