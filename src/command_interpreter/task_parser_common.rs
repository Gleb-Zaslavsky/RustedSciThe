//! Shared helpers for text task parsers (IVP/BVP).
//!
//! This module keeps parser-agnostic utilities in one place to avoid
//! duplicating the same symbolic pre-processing logic across task shells.

use crate::command_interpreter::task_parser::{DocumentMap, Value};
use crate::symbolic::parse_expr::parse_expression_func;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symexpr_graphs::{Definition, SymbolicSystem};
use std::collections::{BTreeSet, HashMap, HashSet};

type GenericSectionMap = HashMap<String, Option<Vec<Value>>>;

/// Parsed symbolic IVP/BVP equations after task-document parameters and named
/// aliases have been resolved.
///
/// `rhs` is solver-ready: it may reference only [`Self::arg`] and the declared
/// [`Self::unknowns`].
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedEquationSystem {
    pub arg: String,
    pub unknowns: Vec<String>,
    pub rhs: Vec<Expr>,
    pub parameter_names: Vec<String>,
    pub parameter_values: HashMap<String, f64>,
}

/// Typed validation error shared by IVP and BVP task-document parsers.
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

/// Parses the solver-facing `equations` section of an IVP or BVP task document.
///
/// The pipeline validates declaration names, expands `where`/`substitute`
/// aliases through [`SymbolicSystem`], substitutes numeric parameters, and
/// rejects unresolved RHS symbols before any numerical solver is constructed.
pub fn parse_symbolic_equation_system(
    document: &DocumentMap,
    default_arg: &str,
) -> Result<ParsedEquationSystem, SharedEquationParseError> {
    let section = get_required_section(document, "equations")?;
    let arg = get_optional_string(section, "arg", "equations")?
        .unwrap_or_else(|| default_arg.to_string());

    let (parameter_names, parameter_values) =
        parse_numeric_parameter_declarations(document, section)?;

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

    validate_unknown_names(&unknowns).map_err(SharedEquationParseError::Semantic)?;
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

    let substitutions = parse_equation_substitutions(document, &parameter_values, &arg, &unknowns)
        .map_err(|message| SharedEquationParseError::InvalidField {
            section: "where/substitute".to_string(),
            field: "*".to_string(),
            message,
        })?;
    let rhs = rhs_raw
        .iter()
        .map(|expr| parse_expr_safe(expr, "equations", "rhs"))
        .collect::<Result<Vec<_>, _>>()?;
    let rhs = apply_symbolic_substitutions_to_vec(rhs, &substitutions)
        .into_iter()
        .map(|expr| expr.set_variable_from_map(&parameter_values))
        .collect::<Vec<_>>();
    validate_equation_rhs_variables(&rhs, &arg, &unknowns)?;

    Ok(ParsedEquationSystem {
        arg,
        unknowns,
        rhs,
        parameter_names,
        parameter_values,
    })
}

/// Parses numeric task parameters declared in either supported document form.
///
/// The compact legacy form stays inside `equations`:
///
/// ```text
/// parameters: rate, offset
/// parameter_values: 2.0, 0.5
/// ```
///
/// The readable form is a dedicated section and is preferable for larger
/// models:
///
/// ```text
/// parameters
/// rate: 2.0
/// offset: 0.5
/// ```
///
/// Both forms are intentionally mutually exclusive. Silently merging them
/// would make a duplicated parameter name depend on parser implementation
/// details instead of the task document's explicit intent.
fn parse_numeric_parameter_declarations(
    document: &DocumentMap,
    equations: &GenericSectionMap,
) -> Result<(Vec<String>, HashMap<String, f64>), SharedEquationParseError> {
    let uses_inline_form =
        equations.contains_key("parameters") || equations.contains_key("parameter_values");
    let section_parameters = document.get("parameters");

    if uses_inline_form && section_parameters.is_some() {
        return Err(SharedEquationParseError::InvalidField {
            section: "parameters".to_string(),
            field: "*".to_string(),
            message: "use either `equations.parameters` with `parameter_values` or the dedicated `parameters` section, not both".to_string(),
        });
    }

    if let Some(parameters) = section_parameters {
        let mut parameter_names = parameters.keys().cloned().collect::<Vec<_>>();
        parameter_names.sort();
        let mut parameter_values = HashMap::with_capacity(parameter_names.len());

        for name in &parameter_names {
            let values = get_required_values(parameters, "parameters", name)?;
            if values.len() != 1 {
                return Err(SharedEquationParseError::InvalidField {
                    section: "parameters".to_string(),
                    field: name.clone(),
                    message: "expected exactly one numeric value".to_string(),
                });
            }
            parameter_values.insert(
                name.clone(),
                value_to_float(&values[0], "parameters", name)?,
            );
        }

        return Ok((parameter_names, parameter_values));
    }

    let parameter_names =
        get_optional_string_list(equations, "parameters", "equations")?.unwrap_or_default();
    let parameter_values_vec =
        get_optional_float_list(equations, "parameter_values", "equations")?.unwrap_or_default();
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
    let parameter_values = parameter_names
        .iter()
        .cloned()
        .zip(parameter_values_vec)
        .collect();

    Ok((parameter_names, parameter_values))
}

/// Parses `where` and `substitute` definitions and fully resolves aliases.
///
/// This compatibility helper intentionally returns ordinary [`Expr`] trees.
/// IVP/BVP task documents are normally small, so this keeps their existing
/// solver interfaces simple while [`SymbolicSystem`] provides common cycle and
/// duplicate-definition validation.
pub fn parse_symbolic_substitutions(
    document: &DocumentMap,
) -> Result<HashMap<String, Expr>, String> {
    parse_symbolic_substitutions_with_parameter_values(document, &HashMap::new())
}

/// Parses symbolic aliases after substituting numeric task-document parameters
/// into every definition.
///
/// Parameters are applied before dependency resolution so every expanded alias
/// and final equation observes the same numeric values.
pub fn parse_symbolic_substitutions_with_parameter_values(
    document: &DocumentMap,
    parameter_values: &HashMap<String, f64>,
) -> Result<HashMap<String, Expr>, String> {
    parse_symbolic_substitutions_with_context(document, parameter_values, &HashMap::new())
}

/// Resolves task-document aliases while reserving names owned by the equation
/// system. This prevents aliases from silently replacing `arg`, unknowns, or
/// numeric parameters before the solver sees the final RHS expressions.
fn parse_equation_substitutions(
    document: &DocumentMap,
    parameter_values: &HashMap<String, f64>,
    arg: &str,
    unknowns: &[String],
) -> Result<HashMap<String, Expr>, String> {
    let mut reserved_names = HashMap::new();
    reserved_names.insert(arg.to_string(), "independent argument");
    for unknown in unknowns {
        reserved_names.insert(unknown.clone(), "unknown");
    }
    for parameter in parameter_values.keys() {
        reserved_names.insert(parameter.clone(), "parameter");
    }

    parse_symbolic_substitutions_with_context(document, parameter_values, &reserved_names)
}

fn parse_symbolic_substitutions_with_context(
    document: &DocumentMap,
    parameter_values: &HashMap<String, f64>,
    reserved_names: &HashMap<String, &'static str>,
) -> Result<HashMap<String, Expr>, String> {
    let mut definitions = Vec::new();
    for section_name in ["where", "substitute"] {
        if let Some(section) = document.get(section_name) {
            parse_substitution_section(section_name, section, &mut definitions)?;
        }
    }
    if definitions.is_empty() {
        return Ok(HashMap::new());
    }

    for definition in &definitions {
        if let Some(owner) = reserved_names.get(&definition.name) {
            return Err(format!(
                "symbolic alias `{}` conflicts with reserved {owner} name",
                definition.name
            ));
        }
    }

    let definitions = definitions.into_iter().map(|definition| {
        Definition::new(
            definition.name,
            definition
                .expression
                .set_variable_from_map(parameter_values),
        )
    });

    resolve_substitution_definitions(definitions)
}

/// Validates the final solver-facing RHS expressions after parameters and
/// symbolic aliases have been applied. A generic [`SymbolicSystem`] may keep
/// free variables, but an IVP/BVP callback can only evaluate its independent
/// argument and declared state variables.
fn validate_equation_rhs_variables(
    rhs: &[Expr],
    arg: &str,
    unknowns: &[String],
) -> Result<(), SharedEquationParseError> {
    let mut allowed = HashSet::with_capacity(unknowns.len() + 1);
    allowed.insert(arg);
    allowed.extend(unknowns.iter().map(String::as_str));

    for (index, expression) in rhs.iter().enumerate() {
        let unresolved = expression
            .all_arguments_are_variables()
            .into_iter()
            .filter(|name| !allowed.contains(name.as_str()))
            .collect::<BTreeSet<_>>();

        if !unresolved.is_empty() {
            return Err(SharedEquationParseError::Semantic(format!(
                "rhs expression {index} contains undeclared symbol(s): {}",
                unresolved.into_iter().collect::<Vec<_>>().join(", ")
            )));
        }
    }

    Ok(())
}

/// Applies already resolved named substitutions to one expression.
pub fn apply_symbolic_substitutions(expr: Expr, substitutions: &HashMap<String, Expr>) -> Expr {
    let mut expanded = expr;
    for (alias, replacement) in substitutions {
        expanded = expanded.substitute_variable(alias, replacement);
    }
    expanded
}

/// Applies already resolved named substitutions to each expression in order.
pub fn apply_symbolic_substitutions_to_vec(
    exprs: Vec<Expr>,
    substitutions: &HashMap<String, Expr>,
) -> Vec<Expr> {
    exprs
        .into_iter()
        .map(|expr| apply_symbolic_substitutions(expr, substitutions))
        .collect()
}

/// Applies numeric parameter values to an externally supplied substitution map.
///
/// This compatibility helper is intentionally separate from the canonical task
/// document path, which substitutes parameters before graph expansion.
pub fn apply_parameter_values_to_substitutions(
    substitutions: HashMap<String, Expr>,
    parameter_values: &HashMap<String, f64>,
) -> HashMap<String, Expr> {
    substitutions
        .into_iter()
        .map(|(name, expr)| (name, expr.set_variable_from_map(parameter_values)))
        .collect()
}

/// Validates the independent argument and numeric parameter declarations.
pub fn validate_symbol_names(arg: &str, parameter_names: &[String]) -> Result<(), String> {
    if arg.trim().is_empty() {
        return Err("independent argument name cannot be empty".to_string());
    }
    let mut seen = HashSet::new();
    for name in parameter_names {
        if name.trim().is_empty() {
            return Err("parameter name cannot be empty".to_string());
        }
        if !seen.insert(name) {
            return Err(format!("duplicate parameter name `{name}`"));
        }
    }
    Ok(())
}

/// Validates state-variable names before symbolic substitutions or solver
/// construction. An explicit duplicate would otherwise make RHS-to-state
/// alignment ambiguous even when the expression text itself is valid.
fn validate_unknown_names(unknowns: &[String]) -> Result<(), String> {
    let mut seen = HashSet::new();
    for name in unknowns {
        if name.trim().is_empty() {
            return Err("unknown name cannot be empty".to_string());
        }
        if !seen.insert(name) {
            return Err(format!("duplicate unknown name `{name}`"));
        }
    }
    Ok(())
}

fn parse_substitution_section(
    section_name: &str,
    section: &GenericSectionMap,
    definitions: &mut Vec<Definition>,
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
        // Preserve every definition until SymbolicSystem validates duplicates.
        // A HashMap here would silently let `substitute` overwrite `where`.
        definitions.push(Definition::new(
            alias.clone(),
            parse_substitution_expr_safe(expr_text, section_name, alias)?,
        ));
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

/// Resolves definitions through the shared graph implementation used by both
/// IVP and BVP parsers.
fn resolve_substitution_definitions(
    definitions: impl IntoIterator<Item = Definition>,
) -> Result<HashMap<String, Expr>, String> {
    let system = SymbolicSystem::new(definitions)
        .map_err(|error| format!("invalid symbolic substitutions: {error}"))?;

    Ok(system
        .expand_all()
        .map_err(|error| format!("failed to expand symbolic substitutions: {error}"))?
        .into_iter()
        .collect())
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

    #[test]
    fn equation_system_accepts_readable_parameter_section() {
        let doc = r#"
equations
arg: t
unknowns: y
rhs: source - y

parameters
R: 2.0
offset: 0.5

where
source: R * t + offset
"#;
        let mut parser = DocumentParser::new(doc.to_string());
        parser.parse_document().expect("parse equation document");
        let map = parser.get_result().expect("document map should exist");

        let parsed = parse_symbolic_equation_system(map, "t")
            .expect("dedicated parameters section should parse");

        assert_eq!(parsed.parameter_names, vec!["R", "offset"]);
        assert_eq!(parsed.parameter_values["R"], 2.0);
        assert_eq!(parsed.parameter_values["offset"], 0.5);
        let rhs = parsed.rhs[0].lambdify_borrowed_thread_safe(&["t", "y"]);
        assert!((rhs(&[3.0, 1.0]) - 5.5).abs() < 1e-12);
    }

    #[test]
    fn equation_system_rejects_mixed_parameter_declaration_styles() {
        let doc = r#"
equations
arg: t
parameters: rate
parameter_values: 2.0
unknowns: y
rhs: -rate * y

parameters
rate: 3.0
"#;
        let mut parser = DocumentParser::new(doc.to_string());
        parser.parse_document().expect("parse equation document");
        let map = parser.get_result().expect("document map should exist");

        let error = parse_symbolic_equation_system(map, "t")
            .expect_err("mixed parameter declaration styles must be rejected");
        assert!(matches!(
            error,
            SharedEquationParseError::InvalidField { .. }
        ));
        assert!(error.to_string().contains("either"));
    }

    #[test]
    fn symbolic_substitutions_reject_duplicate_aliases_across_sections() {
        let doc = r#"
where
gain: 2 * t

substitute
gain: 3 * t
"#;
        let mut parser = DocumentParser::new(doc.to_string());
        parser
            .parse_document()
            .expect("parse substitution sections");
        let map = parser.get_result().expect("document map should exist");

        let error = parse_symbolic_substitutions(map)
            .expect_err("duplicate aliases must not silently overwrite one another");

        assert!(error.contains("invalid symbolic substitutions"));
        assert!(error.contains("определена более одного раза"));
        assert!(error.contains("gain"));
    }

    #[test]
    fn symbolic_substitutions_report_indirect_cycles() {
        let doc = r#"
where
a: b + 1
b: c + 1
c: a + 1
"#;
        let mut parser = DocumentParser::new(doc.to_string());
        parser
            .parse_document()
            .expect("parse cyclic substitution section");
        let map = parser.get_result().expect("document map should exist");

        let error =
            parse_symbolic_substitutions(map).expect_err("indirect alias cycle must be reported");

        assert!(error.contains("циклическая зависимость"));
        assert!(error.contains("a"));
        assert!(error.contains("b"));
        assert!(error.contains("c"));
    }

    #[test]
    fn equation_alias_cannot_shadow_an_unknown() {
        let doc = r#"
equations
arg: t
unknowns: y
rhs: y

where
y: 2 * t
"#;
        let mut parser = DocumentParser::new(doc.to_string());
        parser.parse_document().expect("parse equation document");
        let map = parser.get_result().expect("document map should exist");

        let error = parse_symbolic_equation_system(map, "t")
            .expect_err("an alias must not replace a state variable");

        assert!(matches!(
            &error,
            SharedEquationParseError::InvalidField { .. }
        ));
        assert!(error.to_string().contains("alias `y`"));
        assert!(error.to_string().contains("unknown"));
    }

    #[test]
    fn equation_alias_cannot_shadow_argument_or_parameter() {
        let cases = [
            (
                r#"
equations
arg: t
unknowns: y
rhs: y

where
t: 2 * y
"#,
                "independent argument",
            ),
            (
                r#"
equations
arg: t
parameters: gain
parameter_values: 2.0
unknowns: y
rhs: gain * y

where
gain: t + y
"#,
                "parameter",
            ),
        ];

        for (document, reserved_owner) in cases {
            let mut parser = DocumentParser::new(document.to_string());
            parser.parse_document().expect("parse equation document");
            let map = parser.get_result().expect("document map should exist");

            let error = parse_symbolic_equation_system(map, "t")
                .expect_err("an alias must not replace a solver-owned name");

            assert!(matches!(
                &error,
                SharedEquationParseError::InvalidField { .. }
            ));
            assert!(error.to_string().contains(reserved_owner));
        }
    }

    #[test]
    fn equation_rhs_rejects_undeclared_symbols_after_alias_expansion() {
        let doc = r#"
equations
arg: t
unknowns: y
rhs: source - y

where
source: misspelled_gain * t
"#;
        let mut parser = DocumentParser::new(doc.to_string());
        parser.parse_document().expect("parse equation document");
        let map = parser.get_result().expect("document map should exist");

        let error = parse_symbolic_equation_system(map, "t")
            .expect_err("undeclared symbolic inputs must be reported before solving");

        assert!(matches!(&error, SharedEquationParseError::Semantic(_)));
        assert!(error.to_string().contains("misspelled_gain"));
    }

    #[test]
    fn equation_system_rejects_duplicate_unknown_names() {
        let doc = r#"
equations
arg: t
unknowns: y, y
rhs: -y, -y
"#;
        let mut parser = DocumentParser::new(doc.to_string());
        parser.parse_document().expect("parse equation document");
        let map = parser.get_result().expect("document map should exist");

        let error = parse_symbolic_equation_system(map, "t")
            .expect_err("duplicate state variables must not reach solver construction");

        assert!(matches!(&error, SharedEquationParseError::Semantic(_)));
        assert!(error.to_string().contains("duplicate unknown name `y`"));
    }
}
