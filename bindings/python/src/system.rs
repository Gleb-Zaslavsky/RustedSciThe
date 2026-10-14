use std::collections::HashSet;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::errors::InvalidExpressionError;
use crate::expr::PyExpr;
use crate::lambdify::LambdifiedExpr;

/// A vector-valued symbolic residual system.
///
/// The system keeps the native expression trees and computes its Jacobian by
/// native recursive differentiation. Python receives ordinary lists only at
/// the API boundary; no second symbolic representation is created.
#[pyclass(name = "SymbolicSystem", module = "rustedscithe")]
#[derive(Clone)]
pub struct SymbolicSystem {
    expressions: Vec<PyExpr>,
    variables: Vec<String>,
}

impl SymbolicSystem {
    pub(crate) fn build(
        expressions: Vec<PyExpr>,
        variables: Option<Vec<String>>,
    ) -> PyResult<Self> {
        if expressions.is_empty() {
            return Err(InvalidExpressionError::new_err(
                "a symbolic system must contain at least one expression",
            ));
        }
        let required = required_variables(&expressions);
        let variables = variables.unwrap_or(required.clone());
        validate_variables(&required, &variables)?;
        Ok(Self {
            expressions,
            variables,
        })
    }
}

fn required_variables(expressions: &[PyExpr]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut required = Vec::new();
    for expression in expressions {
        for variable in expression.inner.extract_variables().1 {
            if seen.insert(variable.clone()) {
                required.push(variable);
            }
        }
    }
    required
}

fn validate_variables(required: &[String], variables: &[String]) -> PyResult<()> {
    if variables.iter().any(|name| name.trim().is_empty()) {
        return Err(InvalidExpressionError::new_err(
            "symbolic system variable names must not be empty",
        ));
    }
    let mut seen = HashSet::with_capacity(variables.len());
    if variables.iter().any(|name| !seen.insert(name.as_str())) {
        return Err(InvalidExpressionError::new_err(
            "symbolic system variable names must be unique",
        ));
    }
    if let Some(missing) = required.iter().find(|name| !seen.contains(name.as_str())) {
        return Err(InvalidExpressionError::new_err(format!(
            "symbolic system variable list is missing required symbol '{missing}'"
        )));
    }
    Ok(())
}

fn checked_values(values: &[f64], variables: &[String], label: &str) -> PyResult<()> {
    if values.len() != variables.len() {
        return Err(InvalidExpressionError::new_err(format!(
            "expected {} values for variables {:?}, got {}",
            variables.len(),
            variables,
            values.len()
        )));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(InvalidExpressionError::new_err(format!(
            "{label} values must be finite"
        )));
    }
    Ok(())
}

#[pymethods]
impl SymbolicSystem {
    #[new]
    #[pyo3(signature=(expressions, variables=None))]
    pub fn new(expressions: Vec<PyExpr>, variables: Option<Vec<String>>) -> PyResult<Self> {
        Self::build(expressions, variables)
    }

    /// Return the residual expressions in their original order.
    pub fn expressions(&self) -> Vec<PyExpr> {
        self.expressions.clone()
    }

    /// Return the input order used by evaluation and Lambdify.
    pub fn variables(&self) -> Vec<String> {
        self.variables.clone()
    }

    /// Evaluate every residual at one finite input point.
    pub fn evaluate(&self, values: Vec<f64>) -> PyResult<Vec<f64>> {
        checked_values(&values, &self.variables, "system evaluation")?;
        let variable_refs: Vec<&str> = self.variables.iter().map(String::as_str).collect();
        self.expressions
            .iter()
            .map(|expression| {
                let value = expression.inner.eval_expression(&variable_refs, &values);
                if value.is_finite() {
                    Ok(value)
                } else {
                    Err(InvalidExpressionError::new_err(
                        "system evaluation produced a non-finite value",
                    ))
                }
            })
            .collect()
    }

    /// Build the symbolic residual Jacobian without leaving the native AST.
    pub fn jacobian(&self) -> Vec<Vec<PyExpr>> {
        self.expressions
            .iter()
            .map(|expression| {
                self.variables
                    .iter()
                    .map(|variable| PyExpr::from_inner(expression.inner.diff(variable)))
                    .collect()
            })
            .collect()
    }

    /// Return a new system after recursively expanding named substitutions.
    ///
    /// Variables are inferred again after expansion: aliases replaced by
    /// constants disappear from the numerical input schema, while variables
    /// introduced by an alias are added in first-use order.
    pub fn substitute_many(&self, replacements: Vec<(String, PyExpr)>) -> PyResult<Self> {
        let expressions = self
            .expressions
            .iter()
            .map(|expression| PyExpr::apply_substitutions(&expression.inner, replacements.clone()))
            .collect::<PyResult<Vec<_>>>()?;
        Self::build(expressions, None)
    }

    /// Apply task-document-style `where` definitions to every residual.
    pub fn where_substitute(&self, replacements: &Bound<'_, PyDict>) -> PyResult<Self> {
        let entries = replacements
            .iter()
            .map(|(name, expression)| {
                Ok((name.extract::<String>()?, expression.extract::<PyExpr>()?))
            })
            .collect::<PyResult<Vec<_>>>()?;
        self.substitute_many(entries)
    }

    /// Compile all residuals once into reusable native scalar callables.
    pub fn lambdify(&self) -> PyResult<LambdifiedSystem> {
        let compiled = self
            .expressions
            .iter()
            .map(|expression| LambdifiedExpr::compile(expression, Some(self.variables.clone())))
            .collect::<PyResult<Vec<_>>>()?;
        let jacobian = self
            .jacobian()
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|expression| {
                        LambdifiedExpr::compile(&expression, Some(self.variables.clone()))
                    })
                    .collect::<PyResult<Vec<_>>>()
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(LambdifiedSystem {
            variables: self.variables.clone(),
            residuals: compiled,
            jacobian,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "SymbolicSystem(size={}, variables={:?})",
            self.expressions.len(),
            self.variables
        )
    }
}

/// Reusable vector residual callable produced by `SymbolicSystem.lambdify`.
#[pyclass(name = "LambdifiedSystem", module = "rustedscithe")]
pub struct LambdifiedSystem {
    variables: Vec<String>,
    residuals: Vec<LambdifiedExpr>,
    jacobian: Vec<Vec<LambdifiedExpr>>,
}

#[pymethods]
impl LambdifiedSystem {
    /// Evaluate all residuals using one shared input vector.
    pub fn evaluate(&self, values: Vec<f64>) -> PyResult<Vec<f64>> {
        checked_values(&values, &self.variables, "system Lambdify evaluation")?;
        self.residuals
            .iter()
            .map(|residual| residual.evaluate_slice(&values))
            .collect()
    }

    /// Evaluate the precompiled residual Jacobian without rebuilding it.
    pub fn evaluate_jacobian(&self, values: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
        checked_values(&values, &self.variables, "system Jacobian evaluation")?;
        self.jacobian
            .iter()
            .map(|row| {
                row.iter()
                    .map(|entry| entry.evaluate_slice(&values))
                    .collect()
            })
            .collect()
    }

    /// Return the input order expected by `evaluate`.
    pub fn variables(&self) -> Vec<String> {
        self.variables.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "LambdifiedSystem(size={}, variables={:?})",
            self.residuals.len(),
            self.variables
        )
    }
}
