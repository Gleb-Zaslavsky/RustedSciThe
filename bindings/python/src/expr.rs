use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;
use pyo3::types::PyDict;

use rustedscithe_core::symbolic::symbolic_engine::Expr;

use crate::errors::InvalidExpressionError;
use crate::lambdify::LambdifiedExpr;

/// Python-owned immutable handle to RustedSciThe's symbolic expression tree.
///
/// Expressions are values: every transformation returns a new expression.
/// This makes it safe to retain an expression while a prepared Rust model is
/// using it and avoids exposing Rust ownership details to Python callers.
#[pyclass(name = "Expr", module = "rustedscithe")]
#[derive(Clone)]
pub struct PyExpr {
    pub(crate) inner: Expr,
}

impl PyExpr {
    pub(crate) fn from_inner(inner: Expr) -> Self {
        Self { inner }
    }

    fn checked_name(value: &str, role: &str) -> PyResult<()> {
        if value.trim().is_empty() {
            return Err(InvalidExpressionError::new_err(format!(
                "{role} must not be empty"
            )));
        }
        Ok(())
    }

    fn checked_finite(value: f64, role: &str) -> PyResult<()> {
        if !value.is_finite() {
            return Err(InvalidExpressionError::new_err(format!(
                "{role} must be finite"
            )));
        }
        Ok(())
    }

    pub(crate) fn resolve_substitutions(
        replacements: Vec<(String, PyExpr)>,
    ) -> PyResult<HashMap<String, Expr>> {
        let mut definitions = HashMap::with_capacity(replacements.len());
        for (name, expression) in replacements {
            Self::checked_name(&name, "substitution name")?;
            if definitions.insert(name.clone(), expression.inner).is_some() {
                return Err(InvalidExpressionError::new_err(format!(
                    "substitution name '{name}' was declared more than once"
                )));
            }
        }

        let mut resolved = HashMap::with_capacity(definitions.len());
        let mut visiting = HashSet::with_capacity(definitions.len());
        for name in definitions.keys() {
            resolve_one_substitution(name, &definitions, &mut resolved, &mut visiting)?;
        }
        Ok(resolved)
    }

    pub(crate) fn apply_substitutions(
        expression: &Expr,
        replacements: Vec<(String, PyExpr)>,
    ) -> PyResult<Self> {
        let resolved = Self::resolve_substitutions(replacements)?;
        let mut expanded = expression.clone();
        for (name, replacement) in resolved {
            expanded = expanded.substitute_variable(&name, &replacement);
        }
        Ok(Self::from_inner(expanded))
    }
}

fn resolve_one_substitution(
    name: &str,
    definitions: &HashMap<String, Expr>,
    resolved: &mut HashMap<String, Expr>,
    visiting: &mut HashSet<String>,
) -> PyResult<Expr> {
    if let Some(expression) = resolved.get(name) {
        return Ok(expression.clone());
    }
    if !visiting.insert(name.to_owned()) {
        return Err(InvalidExpressionError::new_err(format!(
            "cyclic symbolic substitution involving '{name}'"
        )));
    }

    let definition = definitions.get(name).ok_or_else(|| {
        InvalidExpressionError::new_err(format!(
            "symbolic substitution dependency '{name}' is not defined"
        ))
    })?;
    let mut expanded = definition.clone();
    for dependency in definition.extract_variables().1 {
        if definitions.contains_key(&dependency) {
            let replacement =
                resolve_one_substitution(&dependency, definitions, resolved, visiting)?;
            expanded = expanded.substitute_variable(&dependency, &replacement);
        }
    }

    visiting.remove(name);
    resolved.insert(name.to_owned(), expanded.clone());
    Ok(expanded)
}

#[pymethods]
impl PyExpr {
    /// Create a symbolic variable such as `Expr.variable("x")`.
    #[staticmethod]
    pub fn variable(name: &str) -> PyResult<Self> {
        Self::checked_name(name, "symbol name")?;
        Ok(Self::from_inner(Expr::Var(name.to_owned())))
    }

    /// Create a finite symbolic constant.
    #[staticmethod]
    pub fn constant(value: f64) -> PyResult<Self> {
        Self::checked_finite(value, "symbolic constants")?;
        Ok(Self::from_inner(Expr::Const(value)))
    }

    /// Parse an expression using RustedSciThe's native expression grammar.
    #[staticmethod]
    pub fn parse(text: &str) -> PyResult<Self> {
        Expr::try_parse_expression(text)
            .map(Self::from_inner)
            .map_err(InvalidExpressionError::new_err)
    }

    /// Return the symbolic derivative with respect to `variable`.
    pub fn diff(&self, variable: &str) -> PyResult<Self> {
        Self::checked_name(variable, "differentiation variable")?;
        Ok(Self::from_inner(self.inner.diff(variable)))
    }

    /// Substitute a finite scalar for one variable.
    pub fn substitute(&self, variable: &str, value: f64) -> PyResult<Self> {
        Self::checked_name(variable, "substitution variable")?;
        Self::checked_finite(value, "substitution values")?;
        Ok(Self::from_inner(self.inner.set_variable(variable, value)))
    }

    /// Substitute another expression for one variable.
    pub fn substitute_expr(&self, variable: &str, value: &PyExpr) -> PyResult<Self> {
        Self::checked_name(variable, "substitution variable")?;
        Ok(Self::from_inner(
            self.inner.substitute_variable(variable, &value.inner),
        ))
    }

    /// Recursively expand a set of named symbolic substitutions.
    ///
    /// The mapping is resolved as a dependency graph, so definitions may refer
    /// to definitions declared later in the Python dictionary. Cycles and
    /// duplicate names are rejected before the target expression is changed.
    pub fn substitute_many(&self, replacements: Vec<(String, PyExpr)>) -> PyResult<Self> {
        Self::apply_substitutions(&self.inner, replacements)
    }

    /// Apply task-document-style `where` definitions from a Python mapping.
    ///
    /// This is an ergonomic alias for `substitute_many`: the operation is
    /// simultaneous after recursive expansion, not dependent on dictionary
    /// iteration order.
    pub fn where_substitute(&self, replacements: &Bound<'_, PyDict>) -> PyResult<Self> {
        let entries = replacements
            .iter()
            .map(|(name, expression)| {
                Ok((name.extract::<String>()?, expression.extract::<PyExpr>()?))
            })
            .collect::<PyResult<Vec<_>>>()?;
        self.substitute_many(entries)
    }

    /// Compile this expression once into a reusable native scalar callable.
    ///
    /// If `variables` is omitted, the native expression variable order is
    /// used. Supplying an explicit order is recommended for solver adapters.
    #[pyo3(signature=(variables=None))]
    pub fn lambdify(&self, variables: Option<Vec<String>>) -> PyResult<LambdifiedExpr> {
        LambdifiedExpr::compile(self, variables)
    }

    /// Return variable names in the order used by `evaluate`.
    pub fn variables(&self) -> Vec<String> {
        self.inner.extract_variables().1
    }

    /// Return a numerically evaluated scalar using `variables()` order.
    pub fn evaluate(&self, values: Vec<f64>) -> PyResult<f64> {
        let variables = self.variables();
        if variables.len() != values.len() {
            return Err(InvalidExpressionError::new_err(format!(
                "expected {} values for variables {:?}, got {}",
                variables.len(),
                variables,
                values.len()
            )));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(InvalidExpressionError::new_err(
                "evaluation values must be finite",
            ));
        }
        let variable_refs: Vec<&str> = variables.iter().map(String::as_str).collect();
        let value = self.inner.eval_expression(&variable_refs, &values);
        if value.is_finite() {
            Ok(value)
        } else {
            Err(InvalidExpressionError::new_err(
                "expression evaluation produced a non-finite value",
            ))
        }
    }

    /// Apply RustedSciThe's symbolic simplification pass.
    pub fn simplify(&self) -> Self {
        Self::from_inner(self.inner.simplify())
    }

    /// Build `exp(self)`.
    pub fn exp(&self) -> Self {
        Self::from_inner(self.inner.clone().exp())
    }

    /// Build `ln(self)`.
    pub fn ln(&self) -> Self {
        Self::from_inner(self.inner.clone().ln())
    }

    /// Build `sin(self)`.
    pub fn sin(&self) -> Self {
        Self::from_inner(Expr::sin(Box::new(self.inner.clone())))
    }

    /// Build `cos(self)`.
    pub fn cos(&self) -> Self {
        Self::from_inner(Expr::cos(Box::new(self.inner.clone())))
    }

    /// Build `tan(self)` using RustedSciThe's `tg` representation.
    pub fn tan(&self) -> Self {
        Self::from_inner(Expr::tg(Box::new(self.inner.clone())))
    }

    /// Build the inverse sine of `self`.
    pub fn asin(&self) -> Self {
        Self::from_inner(Expr::arcsin(Box::new(self.inner.clone())))
    }

    /// Build the inverse cosine of `self`.
    pub fn acos(&self) -> Self {
        Self::from_inner(Expr::arccos(Box::new(self.inner.clone())))
    }

    /// Build the inverse tangent of `self`.
    pub fn atan(&self) -> Self {
        Self::from_inner(Expr::arctg(Box::new(self.inner.clone())))
    }

    /// Build the inverse cotangent of `self`.
    pub fn acot(&self) -> Self {
        Self::from_inner(Expr::arcctg(Box::new(self.inner.clone())))
    }

    /// Native spelling for [`Self::asin`].
    pub fn arcsin(&self) -> Self {
        self.asin()
    }

    /// Native spelling for [`Self::acos`].
    pub fn arccos(&self) -> Self {
        self.acos()
    }

    /// Native spelling for [`Self::atan`].
    pub fn arctg(&self) -> Self {
        self.atan()
    }

    /// Native spelling for [`Self::acot`].
    pub fn arcctg(&self) -> Self {
        self.acot()
    }

    /// Build `self ** exponent`.
    pub fn pow(&self, exponent: &PyExpr) -> Self {
        Self::from_inner(self.inner.clone().pow(exponent.inner.clone()))
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("Expr({:?})", self.inner.to_string())
    }

    fn __add__(&self, rhs: &PyExpr) -> Self {
        Self::from_inner(self.inner.clone() + rhs.inner.clone())
    }

    fn __sub__(&self, rhs: &PyExpr) -> Self {
        Self::from_inner(self.inner.clone() - rhs.inner.clone())
    }

    fn __mul__(&self, rhs: &PyExpr) -> Self {
        Self::from_inner(self.inner.clone() * rhs.inner.clone())
    }

    fn __truediv__(&self, rhs: &PyExpr) -> PyResult<Self> {
        if matches!(&rhs.inner, Expr::Const(value) if *value == 0.0) {
            return Err(InvalidExpressionError::new_err("division by zero"));
        }
        Ok(Self::from_inner(self.inner.clone() / rhs.inner.clone()))
    }

    fn __pow__(&self, rhs: &PyExpr, _modulo: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        Ok(self.pow(rhs))
    }

    fn __neg__(&self) -> Self {
        Self::from_inner(-self.inner.clone())
    }
}

#[pyfunction]
pub fn parse_expr(text: &str) -> PyResult<PyExpr> {
    PyExpr::parse(text)
}

/// Create a symbolic variable without importing the `Expr` class.
#[pyfunction]
pub fn symbol(name: &str) -> PyResult<PyExpr> {
    PyExpr::variable(name)
}

/// Create a symbolic constant without importing the `Expr` class.
#[pyfunction]
pub fn constant(value: f64) -> PyResult<PyExpr> {
    PyExpr::constant(value)
}
