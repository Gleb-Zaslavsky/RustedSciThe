//! Shared helpers for text task parsers (IVP/BVP).
//!
//! This module keeps parser-agnostic utilities in one place to avoid
//! duplicating the same symbolic pre-processing logic across task shells.

use crate::command_interpreter::task_parser::{DocumentMap, Value};
use crate::symbolic::parse_expr::parse_expression_func;
use crate::symbolic::symbolic_engine::Expr;
use std::collections::{HashMap, HashSet};

type GenericSectionMap = HashMap<String, Option<Vec<Value>>>;

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedEquationSystem {
    pub arg: String,
    pub unknowns: Vec<String>,
    pub rhs: Vec<Expr>,
    pub parameter_names: Vec<String>,
    pub parameter_values: HashMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SharedEquationParseError {
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
    Semantic(String),
}

impl std::fmt::Display for SharedEquationParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
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
            Self::Semantic(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for SharedEquationParseError {}

pub fn parse_symbolic_equation_system(
    document: &DocumentMap,
    default_arg: &str,
) -> Result<ParsedEquationSystem, SharedEquationParseError> {
    let section = get_required_section(document, "equations")?;
    let arg = get_optional_string(section, "arg", "equations")?
        .unwrap_or_else(|| default_arg.to_string());

    let parameter_names =
        get_optional_string_list(section, "parameters", "equations")?.unwrap_or_default();
    let parameter_values_vec =
        get_optional_float_list(section, "parameter_values", "equations")?.unwrap_or_default();
    if parameter_names.len() != parameter_values_vec.len() {
        return Err(SharedEquationParseError::InvalidField {
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

    validate_symbol_names(&arg, &parameter_names).map_err(SharedEquationParseError::Semantic)?;

    let (unknowns, rhs_raw) = if section.contains_key("unknowns") || section.contains_key("rhs") {
        let unknowns = get_required_string_list(section, "equations", "unknowns")?;
        let rhs = get_required_string_list(section, "equations", "rhs")?;
        (unknowns, rhs)
    } else {
        parse_pair_style_equations(section)?
    };

    if unknowns.len() != rhs_raw.len() {
        return Err(SharedEquationParseError::InconsistentEquationCounts {
            unknowns: unknowns.len(),
            rhs: rhs_raw.len(),
        });
    }

    let unknown_set: HashSet<&str> = unknowns.iter().map(String::as_str).collect();
    if unknown_set.contains(arg.as_str()) {
        return Err(SharedEquationParseError::Semantic(format!(
            "argument `{arg}` cannot also be listed as an unknown"
        )));
    }
    for parameter in &parameter_names {
        if unknown_set.contains(parameter.as_str()) {
            return Err(SharedEquationParseError::Semantic(format!(
                "parameter `{parameter}` cannot also be listed as an unknown"
            )));
        }
        if parameter == &arg {
            return Err(SharedEquationParseError::Semantic(format!(
                "parameter `{parameter}` cannot also be the independent argument"
            )));
        }
    }

    let substitutions = parse_symbolic_substitutions(document).map_err(|message| {
        SharedEquationParseError::InvalidField {
            section: "where/substitute".to_string(),
            field: "*".to_string(),
            message,
        }
    })?;
    let substitutions = apply_parameter_values_to_substitutions(substitutions, &parameter_values);
    let rhs = rhs_raw
        .iter()
        .map(|expr| parse_expr_safe(expr, "equations", "rhs"))
        .collect::<Result<Vec<_>, _>>()?;
    let rhs = apply_symbolic_substitutions_to_vec(rhs, &substitutions)
        .into_iter()
        .map(|expr| expr.set_variable_from_map(&parameter_values))
        .collect();

    Ok(ParsedEquationSystem {
        arg,
        unknowns,
        rhs,
        parameter_names,
        parameter_values,
    })
}

pub fn parse_symbolic_substitutions(
    document: &DocumentMap,
) -> Result<HashMap<String, Expr>, String> {
    let mut raw: HashMap<String, Expr> = HashMap::new();
    for section_name in ["where", "substitute"] {
        if let Some(section) = document.get(section_name) {
            parse_substitution_section(section_name, section, &mut raw)?;
        }
    }
    if raw.is_empty() {
        return Ok(HashMap::new());
    }
    resolve_substitution_aliases(&raw)
}

pub fn apply_symbolic_substitutions(expr: Expr, substitutions: &HashMap<String, Expr>) -> Expr {
    let mut expanded = expr;
    for (alias, replacement) in substitutions {
        expanded = expanded.substitute_variable(alias, replacement);
    }
    expanded
}

pub fn apply_symbolic_substitutions_to_vec(
    exprs: Vec<Expr>,
    substitutions: &HashMap<String, Expr>,
) -> Vec<Expr> {
    exprs
        .into_iter()
        .map(|expr| apply_symbolic_substitutions(expr, substitutions))
        .collect()
}

pub fn apply_parameter_values_to_substitutions(
    substitutions: HashMap<String, Expr>,
    parameter_values: &HashMap<String, f64>,
) -> HashMap<String, Expr> {
    substitutions
        .into_iter()
        .map(|(name, expr)| (name, expr.set_variable_from_map(parameter_values)))
        .collect()
}

pub fn validate_symbol_names(arg: &str, parameter_names: &[String]) -> Result<(), String> {
    if arg.trim().is_empty() {
        return Err("independent argument name cannot be empty".to_string());
    }
    let mut seen = HashSet::new();
    for name in parameter_names {
        if !seen.insert(name) {
            return Err(format!("duplicate parameter name `{name}`"));
        }
    }
    Ok(())
}

fn parse_substitution_section(
    section_name: &str,
    section: &GenericSectionMap,
    raw: &mut HashMap<String, Expr>,
) -> Result<(), String> {
    for (alias, values_opt) in section {
        let Some(values) = values_opt else {
            return Err(format!(
                "section `{section_name}` key `{alias}` must contain one expression string"
            ));
        };
        if values.len() != 1 {
            return Err(format!(
                "section `{section_name}` key `{alias}` must contain exactly one expression"
            ));
        }
        let value = &values[0];
        let Some(expr_text) = value.as_string() else {
            return Err(format!(
                "section `{section_name}` key `{alias}` must be a string expression"
            ));
        };
        raw.insert(
            alias.clone(),
            parse_substitution_expr_safe(expr_text, section_name, alias)?,
        );
    }
    Ok(())
}

fn parse_substitution_expr_safe(
    expr_text: &str,
    section: &str,
    field: &str,
) -> Result<Expr, String> {
    parse_expression_func(0, expr_text).map_err(|message| {
        format!(
            "failed to parse symbolic expression `{expr_text}` in section `{section}` field `{field}`: {message}"
        )
    })
}

fn parse_expr_safe(
    expr_text: &str,
    section: &str,
    field: &str,
) -> Result<Expr, SharedEquationParseError> {
    parse_expression_func(0, expr_text).map_err(|message| SharedEquationParseError::InvalidField {
        section: section.to_string(),
        field: field.to_string(),
        message: format!("failed to parse symbolic expression `{expr_text}`: {message}"),
    })
}

fn parse_pair_style_equations(
    section: &GenericSectionMap,
) -> Result<(Vec<String>, Vec<String>), SharedEquationParseError> {
    let reserved = [
        "arg",
        "parameters",
        "params",
        "parameter_names",
        "parameter_values",
        "params_values",
        "param_values",
        "unknowns",
        "rhs",
    ];
    let mut unknowns = Vec::new();
    let mut rhs = Vec::new();

    for key in section.keys() {
        if reserved.contains(&key.as_str()) {
            continue;
        }
        let expr = get_required_symbolic_expr(section, "equations", key)?;
        unknowns.push(key.clone());
        rhs.push(expr);
    }

    if unknowns.is_empty() {
        return Err(SharedEquationParseError::Semantic(
            "equations section must contain either `unknowns`/`rhs` lists or variable-to-rhs pairs"
                .to_string(),
        ));
    }

    Ok((unknowns, rhs))
}

fn get_required_symbolic_expr(
    section: &GenericSectionMap,
    section_name: &str,
    field: &str,
) -> Result<String, SharedEquationParseError> {
    let values = get_required_values(section, section_name, field)?;
    if values.is_empty() {
        return Err(SharedEquationParseError::MissingField {
            section: section_name.to_string(),
            field: field.to_string(),
        });
    }
    Ok(values
        .iter()
        .map(Value::to_string_value)
        .collect::<Vec<_>>()
        .join(" "))
}

fn get_required_section<'a>(
    document: &'a DocumentMap,
    section: &'static str,
) -> Result<&'a GenericSectionMap, SharedEquationParseError> {
    document
        .get(section)
        .ok_or(SharedEquationParseError::MissingSection(section))
}

fn get_required_values<'a>(
    section: &'a GenericSectionMap,
    section_name: &str,
    field: &str,
) -> Result<&'a Vec<Value>, SharedEquationParseError> {
    section
        .get(field)
        .ok_or_else(|| SharedEquationParseError::MissingField {
            section: section_name.to_string(),
            field: field.to_string(),
        })?
        .as_ref()
        .ok_or_else(|| SharedEquationParseError::MissingField {
            section: section_name.to_string(),
            field: field.to_string(),
        })
}

fn get_optional_string(
    section: &GenericSectionMap,
    field: &str,
    section_name: &str,
) -> Result<Option<String>, SharedEquationParseError> {
    match section.get(field) {
        Some(Some(values)) if !values.is_empty() => {
            if values.len() != 1 {
                return Err(SharedEquationParseError::InvalidField {
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
) -> Result<Vec<String>, SharedEquationParseError> {
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
) -> Result<Option<Vec<String>>, SharedEquationParseError> {
    match section.get(field) {
        Some(Some(values)) => values
            .iter()
            .map(|value| value_to_string(value, section_name, field))
            .collect::<Result<Vec<_>, _>>()
            .map(Some),
        _ => Ok(None),
    }
}

fn get_optional_float_list(
    section: &GenericSectionMap,
    field: &str,
    section_name: &str,
) -> Result<Option<Vec<f64>>, SharedEquationParseError> {
    match section.get(field) {
        Some(Some(values)) => values_to_float_list(values, section_name, field).map(Some),
        _ => Ok(None),
    }
}

fn values_to_float_list(
    values: &[Value],
    section_name: &str,
    field: &str,
) -> Result<Vec<f64>, SharedEquationParseError> {
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

fn value_to_string(
    value: &Value,
    section_name: &str,
    field: &str,
) -> Result<String, SharedEquationParseError> {
    if let Some(text) = value.as_string() {
        Ok(text.clone())
    } else {
        Err(SharedEquationParseError::InvalidField {
            section: section_name.to_string(),
            field: field.to_string(),
            message: "expected string".to_string(),
        })
    }
}

fn value_to_float(
    value: &Value,
    section_name: &str,
    field: &str,
) -> Result<f64, SharedEquationParseError> {
    if let Some(number) = value.as_float() {
        Ok(number)
    } else if let Some(integer) = value.as_usize() {
        Ok(integer as f64)
    } else {
        Err(SharedEquationParseError::InvalidField {
            section: section_name.to_string(),
            field: field.to_string(),
            message: "expected numeric value".to_string(),
        })
    }
}

fn resolve_substitution_aliases(
    raw: &HashMap<String, Expr>,
) -> Result<HashMap<String, Expr>, String> {
    fn resolve_one(
        name: &str,
        raw: &HashMap<String, Expr>,
        resolved: &mut HashMap<String, Expr>,
        visiting: &mut HashSet<String>,
    ) -> Result<Expr, String> {
        if let Some(expr) = resolved.get(name) {
            return Ok(expr.clone());
        }
        if !raw.contains_key(name) {
            return Err(format!(
                "internal substitution error: alias `{name}` was requested but not found"
            ));
        }
        if !visiting.insert(name.to_string()) {
            return Err(format!(
                "cyclic symbolic substitution detected around `{name}`"
            ));
        }

        let mut expr = raw[name].clone();
        let deps: Vec<String> = raw
            .keys()
            .filter(|dep| dep.as_str() != name && expr.contains_variable(dep.as_str()))
            .cloned()
            .collect();
        for dep in deps {
            let replacement = resolve_one(&dep, raw, resolved, visiting)?;
            expr = expr.substitute_variable(&dep, &replacement);
        }

        visiting.remove(name);
        resolved.insert(name.to_string(), expr.clone());
        Ok(expr)
    }

    let mut resolved: HashMap<String, Expr> = HashMap::new();
    for alias in raw.keys() {
        let mut visiting = HashSet::new();
        let _ = resolve_one(alias, raw, &mut resolved, &mut visiting)?;
    }
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command_interpreter::task_parser::DocumentParser;

    #[test]
    fn where_section_supports_nested_aliases() {
        let doc = r#"
where
a: 2*t
b: a + y
"#;
        let mut parser = DocumentParser::new(doc.to_string());
        parser.parse_document().expect("parse where section");
        let map = parser.get_result().expect("document map should exist");
        let substitutions =
            parse_symbolic_substitutions(map).expect("substitution parsing should succeed");
        assert!(substitutions.contains_key("a"));
        assert!(substitutions.contains_key("b"));
        let expr = Expr::parse_expression("b - y");
        let expanded = apply_symbolic_substitutions(expr, &substitutions);
        let f = expanded.lambdify_borrowed_thread_safe(&["t", "y"]);
        let value = f(&[3.0, 10.0]);
        assert!((value - 6.0).abs() < 1e-12);
    }

    #[test]
    fn where_section_reports_bad_expression_without_panic() {
        let doc = r#"
where
bad_alias: sin(
"#;
        let mut parser = DocumentParser::new(doc.to_string());
        parser.parse_document().expect("parse where section");
        let map = parser.get_result().expect("document map should exist");
        let error = parse_symbolic_substitutions(map)
            .expect_err("bad symbolic alias should return a parser error");
        assert!(error.contains("failed to parse symbolic expression"));
        assert!(error.contains("bad_alias"));
    }

    #[test]
    fn symbolic_substitutions_receive_numeric_parameters_before_rhs_expansion() {
        let doc = r#"
equations
arg: t
parameters: a
parameter_values: 3.0
unknowns: y
rhs: gain * y

where
base: 2 * t
gain: base + a
"#;
        let mut parser = DocumentParser::new(doc.to_string());
        parser.parse_document().expect("parse equations section");
        let map = parser.get_result().expect("document map should exist");
        let parsed =
            parse_symbolic_equation_system(map, "t").expect("equation parsing should succeed");
        let rendered = parsed.rhs[0].to_string();
        assert!(rendered.contains('3'));
        assert!(rendered.contains('2'));
        assert!(!rendered.contains('a'));
    }
}
