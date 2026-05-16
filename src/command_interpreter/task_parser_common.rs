//! Shared helpers for text task parsers (IVP/BVP).
//!
//! This module keeps parser-agnostic utilities in one place to avoid
//! duplicating the same symbolic pre-processing logic across task shells.

use crate::command_interpreter::task_parser::{DocumentMap, Value};
use crate::symbolic::symbolic_engine::Expr;
use std::collections::{HashMap, HashSet};

type GenericSectionMap = HashMap<String, Option<Vec<Value>>>;

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
        raw.insert(alias.clone(), Expr::parse_expression(expr_text));
    }
    Ok(())
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
}
