//! Thin, checked Python wrappers around the core symbolic vector/matrix types.
//!
//! The core array methods are optimized for trusted Rust callers and some use
//! assertions for dimensional invariants. This module validates those
//! invariants at the Python boundary before delegating the actual algebra.

use std::collections::BTreeSet;

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use rustedscithe_core::symbolic::symbolic_vectors::{ExprMatrix, ExprVector};

use crate::errors::{InvalidConfigurationError, InvalidExpressionError};
use crate::expr::PyExpr;
use crate::numpy_support;

fn ordered_variables(
    expressions: impl IntoIterator<Item = rustedscithe_core::symbolic::symbolic_engine::Expr>,
) -> Vec<String> {
    let mut names = BTreeSet::new();
    for expression in expressions {
        names.extend(expression.extract_variables().1);
    }
    names.into_iter().collect()
}

fn checked_values<'a>(
    values: &[f64],
    expected: usize,
    variables: &'a [String],
) -> PyResult<Vec<&'a str>> {
    if values.len() != expected {
        return Err(InvalidExpressionError::new_err(format!(
            "expected {expected} values for variables {variables:?}, got {}",
            values.len()
        )));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(InvalidExpressionError::new_err(
            "symbolic array evaluation values must be finite",
        ));
    }
    Ok(variables.iter().map(String::as_str).collect())
}

/// Immutable symbolic vector backed by the native `ExprVector` type.
#[pyclass(name = "ExprVector", module = "rustedscithe")]
#[derive(Clone)]
pub struct PyExprVector {
    inner: ExprVector,
}

impl PyExprVector {
    fn from_inner(inner: ExprVector) -> Self {
        Self { inner }
    }

    fn variable_names(&self) -> Vec<String> {
        ordered_variables(self.inner.data.iter().cloned())
    }
}

#[pymethods]
impl PyExprVector {
    /// Build a symbolic vector from an ordered sequence of scalar expressions.
    #[new]
    pub fn new(expressions: Vec<PyExpr>) -> Self {
        Self::from_inner(ExprVector::new(
            expressions
                .into_iter()
                .map(|expression| expression.inner)
                .collect(),
        ))
    }

    /// Construct a zero vector of the requested length.
    #[staticmethod]
    pub fn zeros(size: usize) -> Self {
        Self::from_inner(ExprVector::zeros(size))
    }

    /// Number of scalar expressions in the vector.
    pub fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Return one scalar expression, raising `IndexError` for an invalid index.
    pub fn __getitem__(&self, index: usize) -> PyResult<PyExpr> {
        self.inner
            .get(index)
            .cloned()
            .map(PyExpr::from_inner)
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("vector index out of range"))
    }

    /// Return independent scalar-expression handles in vector order.
    pub fn elements(&self) -> Vec<PyExpr> {
        self.inner
            .data
            .iter()
            .cloned()
            .map(PyExpr::from_inner)
            .collect()
    }

    /// Return sorted free-variable names shared by `evaluate`'s input order.
    pub fn variables(&self) -> Vec<String> {
        self.variable_names()
    }

    /// Return an immutable vector with each entry differentiated by `variable`.
    pub fn diff(&self, variable: &str) -> PyResult<Self> {
        if variable.trim().is_empty() {
            return Err(InvalidExpressionError::new_err(
                "differentiation variable must not be empty",
            ));
        }
        Ok(Self::from_inner(self.inner.diff(variable)))
    }

    /// Simplify each entry using the core symbolic simplifier.
    pub fn simplify(&self) -> Self {
        Self::from_inner(self.inner.simplify())
    }

    /// Substitute one variable with an expression in each entry.
    pub fn substitute_expr(&self, variable: &str, value: &PyExpr) -> PyResult<Self> {
        if variable.trim().is_empty() {
            return Err(InvalidExpressionError::new_err(
                "substitution variable must not be empty",
            ));
        }
        Ok(Self::from_inner(
            self.inner.substitute(variable, &value.inner),
        ))
    }

    /// Compute a symbolic dot product, rejecting mismatched lengths before core assertions.
    pub fn dot(&self, other: &PyExprVector) -> PyResult<PyExpr> {
        if self.inner.len() != other.inner.len() {
            return Err(InvalidConfigurationError::new_err(format!(
                "dot-product lengths differ: {} and {}",
                self.inner.len(),
                other.inner.len()
            )));
        }
        Ok(PyExpr::from_inner(self.inner.dot(&other.inner)))
    }

    /// Evaluate to a new owned NumPy vector using sorted free-variable order.
    pub fn evaluate<'py>(
        &self,
        py: Python<'py>,
        values: Vec<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let variables = self.variable_names();
        let variable_refs = checked_values(&values, variables.len(), &variables)?;
        let result = self.inner.evaluate(&variable_refs, &values);
        let output: Vec<f64> = result.iter().copied().collect();
        if output.iter().any(|value| !value.is_finite()) {
            return Err(InvalidExpressionError::new_err(
                "symbolic vector evaluation produced a non-finite value",
            ));
        }
        Ok(numpy_support::vector(py, &output))
    }

    fn __repr__(&self) -> String {
        format!("ExprVector({:?})", self.inner.to_strings())
    }
}

/// Immutable rectangular symbolic matrix backed by the native `ExprMatrix` type.
#[pyclass(name = "ExprMatrix", module = "rustedscithe")]
#[derive(Clone)]
pub struct PyExprMatrix {
    inner: ExprMatrix,
}

impl PyExprMatrix {
    fn from_inner(inner: ExprMatrix) -> Self {
        Self { inner }
    }

    fn variable_names(&self) -> Vec<String> {
        ordered_variables(self.inner.data.iter().flatten().cloned())
    }
}

#[pymethods]
impl PyExprMatrix {
    /// Build a rectangular symbolic matrix from rows of scalar expressions.
    ///
    /// Ragged rows are rejected here instead of reaching the core constructor's
    /// assertion. An empty row sequence creates a `0 x 0` matrix.
    #[new]
    pub fn new(rows: Vec<Vec<PyExpr>>) -> PyResult<Self> {
        if let Some(first) = rows.first() {
            let expected = first.len();
            if rows.iter().any(|row| row.len() != expected) {
                return Err(InvalidConfigurationError::new_err(
                    "symbolic matrix rows must all have the same length",
                ));
            }
        }
        let rows = rows
            .into_iter()
            .map(|row| row.into_iter().map(|expression| expression.inner).collect())
            .collect();
        Ok(Self::from_inner(ExprMatrix::new(rows)))
    }

    /// Construct a symbolic zero matrix with the requested shape.
    #[staticmethod]
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self::from_inner(ExprMatrix::zeros(nrows, ncols))
    }

    /// Construct a square symbolic identity matrix.
    #[staticmethod]
    pub fn identity(size: usize) -> Self {
        Self::from_inner(ExprMatrix::identity(size))
    }

    /// Return `(rows, columns)`.
    pub fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    /// Return sorted free-variable names shared by `evaluate`'s input order.
    pub fn variables(&self) -> Vec<String> {
        self.variable_names()
    }

    /// Return an entry, raising `IndexError` when either index is out of range.
    pub fn get(&self, row: usize, column: usize) -> PyResult<PyExpr> {
        self.inner
            .get(row, column)
            .cloned()
            .map(PyExpr::from_inner)
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("matrix index out of range"))
    }

    /// Return one row as a new symbolic vector.
    pub fn row(&self, index: usize) -> PyResult<PyExprVector> {
        if index >= self.inner.nrows() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "matrix row index out of range",
            ));
        }
        Ok(PyExprVector::from_inner(self.inner.row(index)))
    }

    /// Return one column as a new symbolic vector.
    pub fn column(&self, index: usize) -> PyResult<PyExprVector> {
        if index >= self.inner.ncols() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "matrix column index out of range",
            ));
        }
        Ok(PyExprVector::from_inner(self.inner.column(index)))
    }

    /// Return a new matrix with rows and columns exchanged.
    pub fn transpose(&self) -> Self {
        Self::from_inner(self.inner.transpose())
    }

    /// Differentiate every entry by one variable.
    pub fn diff(&self, variable: &str) -> PyResult<Self> {
        if variable.trim().is_empty() {
            return Err(InvalidExpressionError::new_err(
                "differentiation variable must not be empty",
            ));
        }
        Ok(Self::from_inner(self.inner.diff(variable)))
    }

    /// Simplify every entry independently.
    pub fn simplify(&self) -> Self {
        Self::from_inner(self.inner.simplify())
    }

    /// Substitute one variable with an expression in every matrix entry.
    pub fn substitute_expr(&self, variable: &str, value: &PyExpr) -> PyResult<Self> {
        if variable.trim().is_empty() {
            return Err(InvalidExpressionError::new_err(
                "substitution variable must not be empty",
            ));
        }
        Ok(Self::from_inner(
            self.inner.substitute(variable, &value.inner),
        ))
    }

    /// Multiply by a symbolic vector after checking the inner dimensions.
    pub fn matvec(&self, vector: &PyExprVector) -> PyResult<PyExprVector> {
        if self.inner.ncols() != vector.inner.len() {
            return Err(InvalidConfigurationError::new_err(format!(
                "matrix-vector dimensions do not match: matrix is {:?}, vector length is {}",
                self.inner.shape(),
                vector.inner.len()
            )));
        }
        Ok(PyExprVector::from_inner(
            self.inner.mul_vector(&vector.inner),
        ))
    }

    /// Return the symbolic trace of a square matrix.
    pub fn trace(&self) -> PyResult<PyExpr> {
        if !self.inner.is_square() {
            return Err(InvalidConfigurationError::new_err(format!(
                "trace requires a square matrix, got {:?}",
                self.inner.shape()
            )));
        }
        Ok(PyExpr::from_inner(self.inner.trace()))
    }

    /// Return a symbolic determinant for square matrices of size one through three.
    pub fn determinant(&self) -> PyResult<PyExpr> {
        let (rows, columns) = self.inner.shape();
        if rows != columns || !(1..=3).contains(&rows) {
            return Err(InvalidConfigurationError::new_err(format!(
                "symbolic determinant supports square matrices of size 1 through 3, got {:?}",
                self.inner.shape()
            )));
        }
        Ok(PyExpr::from_inner(self.inner.determinant()))
    }

    /// Evaluate to a new owned NumPy matrix using sorted free-variable order.
    pub fn evaluate<'py>(
        &self,
        py: Python<'py>,
        values: Vec<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let variables = self.variable_names();
        let variable_refs = checked_values(&values, variables.len(), &variables)?;
        let result = self.inner.evaluate(&variable_refs, &values);
        if result.iter().any(|value| !value.is_finite()) {
            return Err(InvalidExpressionError::new_err(
                "symbolic matrix evaluation produced a non-finite value",
            ));
        }
        // Allocate with the exact native shape, including valid empty shapes
        // such as `(0, n)` that cannot be inferred from nested empty rows.
        let output = PyArray2::zeros(py, [result.nrows(), result.ncols()], false);
        if result.nrows() > 0 && result.ncols() > 0 {
            let mut writable = output.readwrite();
            let mut array = writable.as_array_mut();
            for row in 0..result.nrows() {
                for column in 0..result.ncols() {
                    array[(row, column)] = result[(row, column)];
                }
            }
        }
        Ok(output)
    }

    fn __repr__(&self) -> String {
        let rows: Vec<Vec<String>> = self
            .inner
            .data
            .iter()
            .map(|row| row.iter().map(ToString::to_string).collect())
            .collect();
        format!("ExprMatrix({rows:?})")
    }
}
