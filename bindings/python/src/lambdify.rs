use std::collections::HashSet;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustedscithe_core::symbolic::View::conversions::expr_to_atom;

use crate::contract::SymbolicFrontend;
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
    frontend: SymbolicFrontend,
    diagnostics_enabled: bool,
    preparation_ms: Option<f64>,
    evaluation_count: AtomicU64,
    callback_time_ns: AtomicU64,
}

impl LambdifiedExpr {
    pub(crate) fn compile(expression: &PyExpr, variables: Option<Vec<String>>) -> PyResult<Self> {
        Self::compile_with_options(expression, variables, SymbolicFrontend::ExprLegacy, false)
    }

    pub(crate) fn compile_with_options(
        expression: &PyExpr,
        variables: Option<Vec<String>>,
        frontend: SymbolicFrontend,
        diagnostics_enabled: bool,
    ) -> PyResult<Self> {
        let started = diagnostics_enabled.then(Instant::now);
        let required = expression.inner.extract_variables().1;
        let variables = variables.unwrap_or(required.clone());
        validate_variable_list(&required, &variables)?;

        let variable_refs: Vec<&str> = variables.iter().map(String::as_str).collect();
        let evaluator = match frontend {
            SymbolicFrontend::ExprLegacy => expression
                .inner
                .lambdify_borrowed_thread_safe(variable_refs.as_slice()),
            SymbolicFrontend::AtomNative => {
                // Convert once at the frontend boundary. The prepared callback
                // then owns and evaluates the packed Atom without returning to Expr.
                let atom = expr_to_atom(&expression.inner);
                std::panic::catch_unwind(AssertUnwindSafe(|| {
                    atom.lambdify_borrowed_thread_safe(variable_refs.as_slice())
                }))
                .map_err(|payload| {
                    let message = payload
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                        .unwrap_or_else(|| "unknown AtomNative preparation panic".to_owned());
                    InvalidExpressionError::new_err(format!(
                        "AtomNative symbolic preparation failed: {message}"
                    ))
                })?
            }
        };

        Ok(Self {
            expression: expression.inner.to_string(),
            variables,
            evaluator,
            frontend,
            diagnostics_enabled,
            preparation_ms: started.map(|start| start.elapsed().as_secs_f64() * 1_000.0),
            evaluation_count: AtomicU64::new(0),
            callback_time_ns: AtomicU64::new(0),
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

        let started = self.diagnostics_enabled.then(Instant::now);
        let value = (self.evaluator)(values);
        if let Some(started) = started {
            self.callback_time_ns.fetch_add(
                started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                Ordering::Relaxed,
            );
            self.evaluation_count.fetch_add(1, Ordering::Relaxed);
        }
        if value.is_finite() {
            Ok(value)
        } else {
            Err("lambdify evaluation produced a non-finite value".to_owned())
        }
    }

    pub(crate) fn diagnostic_counts(&self) -> (u64, u64) {
        (
            self.evaluation_count.load(Ordering::Relaxed),
            self.callback_time_ns.load(Ordering::Relaxed),
        )
    }

    pub(crate) fn preparation_time_ms(&self) -> Option<f64> {
        self.preparation_ms
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

    /// Evaluate each row of a NumPy matrix using the already-compiled closure.
    ///
    /// The input convention is `(samples, variables)`, and the returned array
    /// has shape `(samples,)`. No Python callback or symbolic preparation is
    /// performed inside the sample loop.
    pub fn evaluate_batch<'py>(
        &self,
        py: Python<'py>,
        values: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let input = values.as_array();
        if input.ncols() != self.variables.len() {
            return Err(InvalidExpressionError::new_err(format!(
                "expected a NumPy matrix with {} columns for variables {:?}, got {}",
                self.variables.len(),
                self.variables,
                input.ncols()
            )));
        }

        let mut output = Vec::with_capacity(input.nrows());
        for row in input.rows() {
            let value = if let Some(contiguous) = row.as_slice() {
                self.evaluate_slice(contiguous)?
            } else {
                // Preserve support for transposed/strided NumPy views while
                // keeping the common C-contiguous row path allocation-free.
                let owned: Vec<f64> = row.iter().copied().collect();
                self.evaluate_slice(&owned)?
            };
            output.push(value);
        }
        Ok(PyArray1::from_vec(py, output))
    }

    /// Return the exact input order expected by `evaluate`.
    pub fn variables(&self) -> Vec<String> {
        self.variables.clone()
    }

    /// Return the selected scalar symbolic frontend.
    #[getter]
    pub fn frontend(&self) -> &'static str {
        match self.frontend {
            SymbolicFrontend::ExprLegacy => "expr_legacy",
            SymbolicFrontend::AtomNative => "atom_native",
        }
    }

    /// Return whether per-call diagnostics were enabled before compilation.
    #[getter]
    pub fn diagnostics_enabled(&self) -> bool {
        self.diagnostics_enabled
    }

    /// Return a compact diagnostics snapshot, excluding Python/NumPy conversion.
    pub fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let report = PyDict::new(py);
        report.set_item("enabled", self.diagnostics_enabled)?;
        report.set_item("evaluations", self.evaluation_count.load(Ordering::Relaxed))?;
        report.set_item(
            "callback_time_ms",
            self.callback_time_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0,
        )?;
        Ok(report)
    }

    /// Return cold symbolic-to-callable preparation time in milliseconds.
    #[getter]
    pub fn preparation_ms(&self) -> Option<f64> {
        self.preparation_ms
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
