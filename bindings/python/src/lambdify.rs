use std::collections::HashSet;

use pyo3::prelude::*;

use crate::errors::InvalidExpressionError;
use crate::expr::PyExpr;

/// A reusable scalar callable compiled by RustedSciThe's Lambdify engine.
///
/// Compilation happens once in `Expr.lambdify`. The callable owns the native
/// closure and can therefore be evaluated repeatedly without reparsing or
/// differentiating the expression. The input order is explicit and exposed to
/// Python through `variables()`.
#[pyclass(name = "LambdifiedExpr", module = "rustedscithe")]
pub struct LambdifiedExpr {
    expression: String,
    variables: Vec<String>,
    evaluator: Box<dyn Fn(&[f64]) -> f64 + Send + Sync>,
}

impl LambdifiedExpr {
    pub(crate) fn compile(expression: &PyExpr, variables: Option<Vec<String>>) -> PyResult<Self> {
        let required = expression.inner.extract_variables().1;
        let variables = variables.unwrap_or(required.clone());
        validate_variable_list(&required, &variables)?;

        let variable_refs: Vec<&str> = variables.iter().map(String::as_str).collect();
        let evaluator = expression
            .inner
            .lambdify_borrowed_thread_safe(variable_refs.as_slice());

        Ok(Self {
            expression: expression.inner.to_string(),
            variables,
            evaluator,
        })
    }

    pub(crate) fn evaluate_slice(&self, values: &[f64]) -> PyResult<f64> {
        self.evaluate_native(values)
            .map_err(InvalidExpressionError::new_err)
    }

    /// Evaluate without constructing a Python exception on the native hot
    /// path. Solver adapters use this while holding no GIL; the public Python
    /// method above translates the same validation failure into `PyErr`.
    pub(crate) fn evaluate_native(&self, values: &[f64]) -> Result<f64, String> {
        if values.len() != self.variables.len() {
            return Err(format!(
                "expected {} values for variables {:?}, got {}",
                self.variables.len(),
                self.variables,
                values.len()
            ));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err("lambdify evaluation values must be finite".to_owned());
        }

        let value = (self.evaluator)(values);
        if value.is_finite() {
            Ok(value)
        } else {
            Err("lambdify evaluation produced a non-finite value".to_owned())
        }
    }
}

fn validate_variable_list(required: &[String], variables: &[String]) -> PyResult<()> {
    if variables.iter().any(|name| name.trim().is_empty()) {
        return Err(InvalidExpressionError::new_err(
            "lambdify variable names must not be empty",
        ));
    }

    let mut seen = HashSet::with_capacity(variables.len());
    if variables.iter().any(|name| !seen.insert(name.as_str())) {
        return Err(InvalidExpressionError::new_err(
            "lambdify variable names must be unique",
        ));
    }

    if let Some(missing) = required.iter().find(|name| !seen.contains(name.as_str())) {
        return Err(InvalidExpressionError::new_err(format!(
            "lambdify variable list is missing required symbol '{missing}'"
        )));
    }
    Ok(())
}

#[pymethods]
impl LambdifiedExpr {
    /// Evaluate the compiled scalar expression using the documented order.
    pub fn evaluate(&self, values: Vec<f64>) -> PyResult<f64> {
        self.evaluate_slice(&values)
    }

    /// Return the exact input order expected by `evaluate`.
    pub fn variables(&self) -> Vec<String> {
        self.variables.clone()
    }

    /// Return the expression captured before native compilation.
    #[getter]
    pub fn expression(&self) -> &str {
        &self.expression
    }

    fn __repr__(&self) -> String {
        format!(
            "LambdifiedExpr(expression={:?}, variables={:?})",
            self.expression, self.variables
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_required_and_duplicate_variables_before_compilation() {
        Python::initialize();
        let expression = PyExpr::parse("x + y").unwrap();

        assert!(LambdifiedExpr::compile(&expression, Some(vec!["x".into()])).is_err());
        assert!(
            LambdifiedExpr::compile(&expression, Some(vec!["x".into(), "x".into(), "y".into()]))
                .is_err()
        );
    }
}
