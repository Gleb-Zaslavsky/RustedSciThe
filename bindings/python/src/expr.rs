use std::collections::{HashMap, HashSet};

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict};

use rustedscithe_core::symbolic::symbolic_engine::Expr;

use crate::errors::{
    InvalidConfigurationError, InvalidExpressionError, UnsupportedSymbolicOperationError, annotate,
};
use crate::lambdify::LambdifiedExpr;
use rustedscithe_core::symbolic::symbolic_integration::QuadMethod;

const MAX_TAYLOR_ORDER: usize = 32;
const MAX_GAUSS_QUADRATURE_DEGREE: usize = 4096;

fn checked_gaussian_quadrature(
    expression: &Expr,
    method: QuadMethod,
    method_name: &str,
    degree: usize,
    lower: f64,
    upper: f64,
    alpha: Option<f64>,
) -> PyResult<f64> {
    if degree == 0 || degree > MAX_GAUSS_QUADRATURE_DEGREE {
        return Err(InvalidConfigurationError::new_err(format!(
            "{method_name} degree must be between 1 and {MAX_GAUSS_QUADRATURE_DEGREE}"
        )));
    }
    let variables = expression.extract_variables().1;
    if variables.len() > 1 {
        return Err(InvalidConfigurationError::new_err(format!(
            "one-dimensional {method_name} requires at most one free variable, found {}",
            variables.len()
        )));
    }
    let result = expression
        .quad(method, degree, lower, upper, alpha)
        .map_err(InvalidConfigurationError::new_err)?;
    if !result.is_finite() {
        return Err(InvalidExpressionError::new_err(format!(
            "{method_name} produced a non-finite result; inspect the integrand domain and singularities"
        )));
    }
    Ok(result)
}

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

    fn operand(value: &Bound<'_, PyAny>) -> PyResult<Expr> {
        if let Ok(expression) = value.extract::<PyRef<'_, PyExpr>>() {
            return Ok(expression.inner.clone());
        }
        if value.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(
                "boolean values are not valid symbolic scalar operands",
            ));
        }
        let scalar = value.extract::<f64>().map_err(|_| {
            PyTypeError::new_err("symbolic operands must be Expr instances or finite numbers")
        })?;
        Self::checked_finite(scalar, "symbolic scalar operands")?;
        Ok(Expr::Const(scalar))
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

    pub(crate) fn apply_resolved_substitutions(
        expression: &Expr,
        replacements: &HashMap<String, Expr>,
    ) -> Self {
        let mut expanded = expression.clone();
        for (name, replacement) in replacements {
            expanded = expanded.substitute_variable(name, replacement);
        }
        Self::from_inner(expanded)
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

    /// Construct multiple variables in the supplied order.
    ///
    /// Names must be non-empty and unique. Unlike `variables()`, this helper
    /// preserves caller order because it returns a sequence of independent
    /// symbols rather than discovering free variables in an expression.
    #[staticmethod]
    pub fn symbols(names: Vec<String>) -> PyResult<Vec<Self>> {
        if names.is_empty() {
            return Err(InvalidExpressionError::new_err(
                "symbol batch must contain at least one name",
            ));
        }
        let mut seen = HashSet::with_capacity(names.len());
        names
            .into_iter()
            .map(|name| {
                Self::checked_name(&name, "symbol name")?;
                if !seen.insert(name.clone()) {
                    return Err(InvalidExpressionError::new_err(format!(
                        "symbol name '{name}' was declared more than once"
                    )));
                }
                Ok(Self::from_inner(Expr::Var(name)))
            })
            .collect()
    }

    /// Construct `count` variables named `prefix0`, `prefix1`, ... .
    #[staticmethod]
    pub fn indexed(prefix: &str, count: usize) -> PyResult<Vec<Self>> {
        Self::checked_name(prefix, "indexed symbol prefix")?;
        if count == 0 {
            return Err(InvalidExpressionError::new_err(
                "indexed symbol count must be positive",
            ));
        }
        (0..count)
            .map(|index| Self::variable(&format!("{prefix}{index}")))
            .collect()
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

    /// Return the `order`-th symbolic derivative, simplifying after each pass.
    pub fn diff_n(&self, variable: &str, order: usize) -> PyResult<Self> {
        Self::checked_name(variable, "differentiation variable")?;
        Ok(Self::from_inner(
            self.inner.n_th_derivative1D(variable, order),
        ))
    }

    /// Return partial derivatives in the expression's sorted free-variable order.
    pub fn gradient(&self) -> Vec<Self> {
        self.variables()
            .iter()
            .map(|variable| Self::from_inner(self.inner.diff(variable)))
            .collect()
    }

    /// Build a formal Taylor polynomial through `order` around `center`.
    ///
    /// Other free variables remain symbolic. This constructs a formal
    /// polynomial; it does not assert convergence or a radius of convergence.
    /// A conservative order limit keeps accidental high-order requests from
    /// causing unbounded tree growth.
    pub fn taylor_series(&self, variable: &str, center: f64, order: usize) -> PyResult<Self> {
        Self::checked_name(variable, "Taylor variable")?;
        Self::checked_finite(center, "Taylor expansion point")?;
        if order > MAX_TAYLOR_ORDER {
            return Err(InvalidExpressionError::new_err(format!(
                "Taylor order must not exceed {MAX_TAYLOR_ORDER}"
            )));
        }

        let delta = Expr::Var(variable.to_owned()) - Expr::Const(center);
        let mut derivative = self.inner.clone();
        let mut factorial = 1.0;
        let mut polynomial = Expr::Const(0.0);
        for degree in 0..=order {
            if degree > 0 {
                derivative = derivative.diff(variable).simplify();
                factorial *= degree as f64;
            }
            let coefficient = derivative.set_variable(variable, center) / Expr::Const(factorial);
            let term = coefficient * delta.clone().pow(Expr::Const(degree as f64));
            polynomial = polynomial + term;
        }
        Ok(Self::from_inner(polynomial.simplify()))
    }

    /// Substitute a finite scalar for one variable.
    pub fn substitute(&self, variable: &str, value: f64) -> PyResult<Self> {
        Self::checked_name(variable, "substitution variable")?;
        Self::checked_finite(value, "substitution values")?;
        Ok(Self::from_inner(self.inner.set_variable(variable, value)))
    }

    /// Substitute several variables with finite numeric constants in one tree
    /// traversal. Variables omitted from the mapping remain symbolic; entries
    /// that do not occur in the expression are ignored.
    pub fn substitute_values(&self, replacements: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut values = HashMap::with_capacity(replacements.len());
        for (name, value) in replacements.iter() {
            let name = name
                .extract::<String>()
                .map_err(|_| PyTypeError::new_err("substitution mapping keys must be strings"))?;
            Self::checked_name(&name, "substitution variable")?;
            if value.is_instance_of::<PyBool>() {
                return Err(PyTypeError::new_err(
                    "boolean values are not valid symbolic substitution values",
                ));
            }
            let scalar = value
                .extract::<f64>()
                .map_err(|_| PyTypeError::new_err("substitution values must be finite numbers"))?;
            Self::checked_finite(scalar, "substitution values")?;
            values.insert(name, scalar);
        }

        Ok(Self::from_inner(self.inner.set_variable_from_map(&values)))
    }

    /// Substitute another expression for one variable.
    pub fn substitute_expr(&self, variable: &str, value: &PyExpr) -> PyResult<Self> {
        Self::checked_name(variable, "substitution variable")?;
        Ok(Self::from_inner(
            self.inner.substitute_variable(variable, &value.inner),
        ))
    }

    /// Rename one variable throughout a new copy of the expression.
    pub fn rename_variable(&self, old_name: &str, new_name: &str) -> PyResult<Self> {
        Self::checked_name(old_name, "old variable name")?;
        Self::checked_name(new_name, "new variable name")?;
        Ok(Self::from_inner(
            self.inner.rename_variable(old_name, new_name),
        ))
    }

    /// Check whether the expression contains a named free variable.
    pub fn contains_variable(&self, name: &str) -> PyResult<bool> {
        Self::checked_name(name, "variable name")?;
        Ok(self.inner.contains_variable(name))
    }

    /// Recursively expand a set of named symbolic substitutions.
    ///
    /// The mapping is resolved as a dependency graph, so definitions may refer
    /// to definitions declared later in the Python dictionary. Cycles and
    /// duplicate names are rejected before the target expression is changed.
    /// Names absent from the mapping remain free variables; extra mapping
    /// entries that are not referenced by the expression are harmless.
    pub fn substitute_many(&self, replacements: Vec<(String, PyExpr)>) -> PyResult<Self> {
        Self::apply_substitutions(&self.inner, replacements)
    }

    /// Apply task-document-style `where` definitions from a Python mapping.
    ///
    /// This is an ergonomic alias for `substitute_many`: the operation is
    /// simultaneous after recursive expansion, not dependent on dictionary
    /// iteration order. Unmapped symbols remain free, and unused definitions
    /// do not alter the expression.
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
    /// If `variables` is omitted, sorted free-variable order is used. Choose
    /// `SymbolicFrontend.AtomNative` to convert once and then evaluate directly
    /// from the packed Atom route. Diagnostics are disabled by default.
    #[pyo3(signature=(variables=None, frontend=None, diagnostics=false))]
    pub fn lambdify(
        &self,
        variables: Option<Vec<String>>,
        frontend: Option<crate::contract::SymbolicFrontend>,
        diagnostics: bool,
    ) -> PyResult<LambdifiedExpr> {
        LambdifiedExpr::compile_with_options(
            self,
            variables,
            frontend.unwrap_or(crate::contract::SymbolicFrontend::ExprLegacy),
            diagnostics,
        )
    }

    /// Return sorted, unique free-variable names.
    ///
    /// `evaluate()` and default `lambdify()` use this exact order; an explicit
    /// Lambdify argument list may choose a different order.
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

    /// Apply RustedSciThe's algebraic simplification pass.
    ///
    /// This pass is not domain-preserving: for example, `x / x` may simplify
    /// to `1`, even though the original expression is undefined at `x = 0`.
    /// Preserve the unsimplified expression or enforce its domain separately
    /// when singular points are relevant to the model.
    pub fn simplify(&self) -> Self {
        Self::from_inner(self.inner.simplify())
    }

    /// Return an indefinite primitive with respect to `variable`.
    ///
    /// The native integrator supports a subset of elementary expressions and
    /// reports unsupported forms as `UnsupportedSymbolicOperationError`.
    /// The returned primitive has no integration constant. In particular, the
    /// native rule for `1/x` is `ln(x)`; callers integrating across negative
    /// values or zero must account for the real-domain branch themselves.
    pub fn integrate(&self, variable: &str) -> PyResult<Self> {
        Self::checked_name(variable, "integration variable")?;
        self.inner
            .integrate(variable)
            .map(Self::from_inner)
            .map_err(|message| {
                let error = annotate(
                    UnsupportedSymbolicOperationError::new_err(message),
                    "symbolic_integration",
                    "Expr.integrate",
                    "native_symbolic",
                );
                Python::attach(|py| {
                    let _ = error.value(py).setattr("operation", "integrate");
                });
                error
            })
    }

    /// Numerically integrate a one-variable expression with composite Simpson's rule.
    ///
    /// `subdivisions` must be positive and even. The interval must be finite;
    /// this method does not detect interior singularities or certify convergence.
    pub fn integrate_simpson(&self, lower: f64, upper: f64, subdivisions: usize) -> PyResult<f64> {
        if !lower.is_finite() || !upper.is_finite() {
            return Err(InvalidConfigurationError::new_err(
                "Simpson integration bounds must be finite",
            ));
        }
        if subdivisions == 0 || subdivisions % 2 != 0 {
            return Err(InvalidConfigurationError::new_err(
                "Simpson subdivisions must be a positive even integer",
            ));
        }
        let variables = self.inner.extract_variables().1;
        if variables.len() > 1 {
            return Err(InvalidConfigurationError::new_err(format!(
                "one-dimensional Simpson integration requires at most one free variable, found {}",
                variables.len()
            )));
        }

        let result = self.inner.numerical_integrate(lower, upper, subdivisions);
        if !result.is_finite() {
            return Err(InvalidExpressionError::new_err(
                "Simpson integration produced a non-finite result; inspect the integrand domain and singularities",
            ));
        }
        Ok(result)
    }

    /// Numerically integrate an unweighted one-variable expression by Gauss-Legendre quadrature.
    ///
    /// The interval must be finite and `degree` must be positive. This method
    /// does not detect interior singularities or certify convergence.
    pub fn integrate_gauss_legendre(&self, lower: f64, upper: f64, degree: usize) -> PyResult<f64> {
        if !lower.is_finite() || !upper.is_finite() {
            return Err(InvalidConfigurationError::new_err(
                "Gauss-Legendre integration bounds must be finite",
            ));
        }
        checked_gaussian_quadrature(
            &self.inner,
            QuadMethod::GaussLegendre,
            "Gauss-Legendre",
            degree,
            lower,
            upper,
            None,
        )
    }

    /// Integrate `exp(-x^2) * self(x)` over the real line by Gauss-Hermite quadrature.
    pub fn integrate_gauss_hermite(&self, degree: usize) -> PyResult<f64> {
        checked_gaussian_quadrature(
            &self.inner,
            QuadMethod::GaussHermite,
            "Gauss-Hermite",
            degree,
            f64::NEG_INFINITY,
            f64::INFINITY,
            None,
        )
    }

    /// Integrate `x^alpha * exp(-x) * self(x)` over `[0, +inf)` by Gauss-Laguerre.
    #[pyo3(signature = (degree, alpha=0.5))]
    pub fn integrate_gauss_laguerre(&self, degree: usize, alpha: f64) -> PyResult<f64> {
        if !alpha.is_finite() || alpha <= -1.0 {
            return Err(InvalidConfigurationError::new_err(
                "Gauss-Laguerre alpha must be finite and greater than -1",
            ));
        }
        checked_gaussian_quadrature(
            &self.inner,
            QuadMethod::GaussLaguerre,
            "Gauss-Laguerre",
            degree,
            0.0,
            f64::INFINITY,
            Some(alpha),
        )
    }

    /// Build `exp(self)`.
    pub fn exp(&self) -> Self {
        Self::from_inner(self.inner.clone().exp())
    }

    /// Build `ln(self)`.
    pub fn ln(&self) -> Self {
        Self::from_inner(self.inner.clone().ln())
    }

    /// Build `log10(self)` using the core's natural-log identity.
    pub fn log10(&self) -> Self {
        Self::from_inner(self.inner.clone().log10())
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

    /// Build cotangent as `cos(self) / sin(self)`.
    ///
    /// The core's legacy `ctg` node currently has inconsistent scalar
    /// evaluation semantics, so this binding uses the equivalent elementary
    /// expression rather than exposing a numerically incorrect node.
    pub fn cot(&self) -> Self {
        Self::from_inner(
            Expr::cos(Box::new(self.inner.clone())) / Expr::sin(Box::new(self.inner.clone())),
        )
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

    /// Render a two-dimensional mathematical presentation for display only.
    /// The result is not a parser or serialization format.
    pub fn pretty_print(&self) -> String {
        self.inner.pretty_print()
    }

    fn __add__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self::from_inner(self.inner.clone() + Self::operand(rhs)?))
    }

    fn __radd__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self::from_inner(Self::operand(lhs)? + self.inner.clone()))
    }

    fn __sub__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self::from_inner(self.inner.clone() - Self::operand(rhs)?))
    }

    fn __rsub__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self::from_inner(Self::operand(lhs)? - self.inner.clone()))
    }

    fn __mul__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self::from_inner(self.inner.clone() * Self::operand(rhs)?))
    }

    fn __rmul__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self::from_inner(Self::operand(lhs)? * self.inner.clone()))
    }

    fn __truediv__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = Self::operand(rhs)?;
        if matches!(&rhs, Expr::Const(value) if *value == 0.0) {
            return Err(InvalidExpressionError::new_err("division by zero"));
        }
        Ok(Self::from_inner(self.inner.clone() / rhs))
    }

    fn __rtruediv__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let lhs = Self::operand(lhs)?;
        if matches!(&self.inner, Expr::Const(value) if *value == 0.0) {
            return Err(InvalidExpressionError::new_err("division by zero"));
        }
        Ok(Self::from_inner(lhs / self.inner.clone()))
    }

    fn __pow__(
        &self,
        rhs: &Bound<'_, PyAny>,
        _modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(self.pow(&Self::from_inner(Self::operand(rhs)?)))
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

/// Construct a caller-ordered batch of symbolic variables.
#[pyfunction]
pub fn symbols(names: Vec<String>) -> PyResult<Vec<PyExpr>> {
    PyExpr::symbols(names)
}

/// Construct an indexed variable batch named `prefix0` through `prefixN-1`.
#[pyfunction]
pub fn indexed_symbols(prefix: &str, count: usize) -> PyResult<Vec<PyExpr>> {
    PyExpr::indexed(prefix, count)
}
