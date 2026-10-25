use std::collections::HashSet;
use std::time::Instant;

use nalgebra::{DMatrix, DVector};
use numpy::{
    PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rustedscithe_core::numerical::Nonlinear_systems::least_squares::PreparedSymbolicLeastSquaresProblem;
use rustedscithe_core::numerical::Nonlinear_systems::symbolic::{
    PreparationStage, PreparationTelemetryMode, SymbolicLambdifyFrontend, SymbolicProblemOptions,
};

use crate::contract::SymbolicFrontend;
use crate::errors::{InvalidConfigurationError, InvalidExpressionError};
use crate::expr::PyExpr;
use crate::nonlinear::map_solve_error_context;

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
        // Resolve the dependency graph once, not once per residual row. This
        // keeps preparation proportional to the shared substitution set plus
        // the actual expression transformations.
        let resolved = PyExpr::resolve_substitutions(replacements)?;
        let expressions = self
            .expressions
            .iter()
            .map(|expression| PyExpr::apply_resolved_substitutions(&expression.inner, &resolved))
            .collect();
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

    /// Compile residuals and their Jacobian once using the selected frontend.
    /// Diagnostics default to off so repeated native callbacks are not timed.
    #[pyo3(signature=(frontend=None, diagnostics=false))]
    pub fn lambdify(
        &self,
        frontend: Option<SymbolicFrontend>,
        diagnostics: bool,
    ) -> PyResult<LambdifiedSystem> {
        let frontend = frontend.unwrap_or(SymbolicFrontend::ExprLegacy);
        let native_frontend = match frontend {
            SymbolicFrontend::ExprLegacy => SymbolicLambdifyFrontend::ExprLegacy,
            SymbolicFrontend::AtomNative => SymbolicLambdifyFrontend::AtomViewNative,
        };
        let options = SymbolicProblemOptions::new()
            .with_variables(self.variables.clone())
            .with_lambdify_frontend(native_frontend)
            .with_preparation_telemetry(if diagnostics {
                PreparationTelemetryMode::Collect
            } else {
                PreparationTelemetryMode::Disabled
            });
        let prepared = PreparedSymbolicLeastSquaresProblem::from_expressions(
            self.expressions
                .iter()
                .map(|expression| expression.inner.clone())
                .collect(),
            options,
        )
        .map_err(|error| {
            map_solve_error_context(error, "prepare", "symbolic_system", frontend_name(frontend))
        })?;
        let residual_count = prepared.residual_count();
        let variable_count = prepared.parameter_count();
        let preparation_ms = diagnostics.then(|| {
            prepared
                .preparation_report()
                .preparation_duration
                .as_secs_f64()
                * 1_000.0
        });
        Ok(LambdifiedSystem {
            variables: self.variables.clone(),
            prepared,
            input_workspace: DVector::zeros(variable_count),
            residual_workspace: DVector::zeros(residual_count),
            jacobian_workspace: DMatrix::zeros(residual_count, variable_count),
            batch_workspace: Vec::new(),
            batch_rows: 0,
            frontend,
            diagnostics_enabled: diagnostics,
            preparation_ms,
            evaluation_count: 0,
            callback_time_ns: 0,
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
    prepared: PreparedSymbolicLeastSquaresProblem,
    input_workspace: DVector<f64>,
    residual_workspace: DVector<f64>,
    jacobian_workspace: DMatrix<f64>,
    batch_workspace: Vec<f64>,
    batch_rows: usize,
    frontend: SymbolicFrontend,
    diagnostics_enabled: bool,
    preparation_ms: Option<f64>,
    evaluation_count: u64,
    callback_time_ns: u64,
}

fn frontend_name(frontend: SymbolicFrontend) -> &'static str {
    match frontend {
        SymbolicFrontend::ExprLegacy => "expr_legacy",
        SymbolicFrontend::AtomNative => "atom_native",
    }
}

fn preparation_stage_name(stage: PreparationStage) -> &'static str {
    match stage {
        PreparationStage::InputValidation => "input_validation",
        PreparationStage::ExpressionParsing => "expression_parsing",
        PreparationStage::ExpressionGraphMaterialization => "expression_graph_materialization",
        PreparationStage::AtomConversion => "atom_conversion",
        PreparationStage::AtomDependencyAnalysis => "atom_dependency_analysis",
        PreparationStage::AtomDifferentiation => "atom_differentiation",
        PreparationStage::JacobianDifferentiation => "jacobian_differentiation",
        PreparationStage::ResidualCallbackPreparation => "residual_callback_preparation",
        PreparationStage::JacobianCallbackPreparation => "jacobian_callback_preparation",
        PreparationStage::ParameterBinding => "parameter_binding",
        PreparationStage::PreparedProblemAssembly => "prepared_problem_assembly",
        PreparationStage::AotMaterialization => "aot_materialization",
        PreparationStage::FinalValidation => "final_validation",
    }
}

fn validate_batch(values: &PyReadonlyArray2<'_, f64>, columns: usize, label: &str) -> PyResult<()> {
    let values = values.as_array();
    if values.ncols() != columns {
        return Err(InvalidConfigurationError::new_err(format!(
            "{label} expects shape (n_samples, {columns}), got {} columns",
            values.ncols()
        )));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(InvalidConfigurationError::new_err(format!(
            "{label} values must be finite"
        )));
    }
    Ok(())
}

fn validate_point_array(
    values: &PyReadonlyArray1<'_, f64>,
    size: usize,
    label: &str,
) -> PyResult<()> {
    let values = values.as_array();
    if values.len() != size {
        return Err(InvalidConfigurationError::new_err(format!(
            "{label} expects shape ({size},), got ({},)",
            values.len()
        )));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(InvalidConfigurationError::new_err(format!(
            "{label} values must be finite"
        )));
    }
    Ok(())
}

fn extract_f64_array1<'py>(
    value: Bound<'py, PyAny>,
    label: &str,
) -> PyResult<PyReadonlyArray1<'py, f64>> {
    let array = value.cast::<PyArray1<f64>>().map_err(|_| {
        InvalidConfigurationError::new_err(format!(
            "{label} must be a one-dimensional float64 NumPy array"
        ))
    })?;
    array.try_readonly().map_err(|error| {
        InvalidConfigurationError::new_err(format!(
            "{label} cannot be borrowed for reading: {error}"
        ))
    })
}

fn extract_f64_array2<'py>(
    value: Bound<'py, PyAny>,
    label: &str,
) -> PyResult<PyReadonlyArray2<'py, f64>> {
    let array = value.cast::<PyArray2<f64>>().map_err(|_| {
        InvalidConfigurationError::new_err(format!(
            "{label} must be a two-dimensional float64 NumPy array"
        ))
    })?;
    array.try_readonly().map_err(|error| {
        InvalidConfigurationError::new_err(format!(
            "{label} cannot be borrowed for reading: {error}"
        ))
    })
}

fn extract_f64_output1<'py>(value: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    value
        .cast::<PyArray1<f64>>()
        .map(Clone::clone)
        .map_err(|_| {
            InvalidConfigurationError::new_err("out must be a one-dimensional float64 NumPy array")
        })
}

fn extract_f64_output2<'py>(value: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    value
        .cast::<PyArray2<f64>>()
        .map(Clone::clone)
        .map_err(|_| {
            InvalidConfigurationError::new_err("out must be a two-dimensional float64 NumPy array")
        })
}

fn extract_f64_output3<'py>(value: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyArray3<f64>>> {
    value
        .cast::<PyArray3<f64>>()
        .map(Clone::clone)
        .map_err(|_| {
            InvalidConfigurationError::new_err(
                "out must be a three-dimensional float64 NumPy array",
            )
        })
}

impl LambdifiedSystem {
    fn evaluate_batch_values(&mut self, values: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let input = values.as_array();
        validate_batch(&values, self.variables.len(), "system batch evaluation")?;
        self.batch_workspace.clear();
        self.batch_rows = input.nrows();
        self.batch_workspace
            .reserve(input.nrows().saturating_mul(self.residual_workspace.len()));
        let started = self.diagnostics_enabled.then(Instant::now);
        for row in input.rows() {
            self.input_workspace
                .iter_mut()
                .zip(row.iter())
                .for_each(|(slot, value)| *slot = *value);
            self.prepared
                .evaluate_residual_into(&self.input_workspace, &mut self.residual_workspace)
                .map_err(|error| {
                    map_solve_error_context(
                        error,
                        "evaluate_batch",
                        "symbolic_system",
                        frontend_name(self.frontend),
                    )
                })?;
            self.batch_workspace
                .extend(self.residual_workspace.iter().copied());
        }
        self.record_diagnostic(
            started,
            (input.nrows() * self.residual_workspace.len()) as u64,
        );
        Ok(())
    }

    fn evaluate_jacobian_batch_values(
        &mut self,
        values: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<()> {
        let input = values.as_array();
        validate_batch(
            &values,
            self.variables.len(),
            "system Jacobian batch evaluation",
        )?;
        let entries = self
            .residual_workspace
            .len()
            .saturating_mul(self.variables.len());
        self.batch_workspace.clear();
        self.batch_rows = input.nrows();
        self.batch_workspace
            .reserve(input.nrows().saturating_mul(entries));
        let started = self.diagnostics_enabled.then(Instant::now);
        for row in input.rows() {
            self.input_workspace
                .iter_mut()
                .zip(row.iter())
                .for_each(|(slot, value)| *slot = *value);
            self.prepared
                .evaluate_jacobian_into(&self.input_workspace, &mut self.jacobian_workspace)
                .map_err(|error| {
                    map_solve_error_context(
                        error,
                        "evaluate_jacobian_batch",
                        "symbolic_system",
                        frontend_name(self.frontend),
                    )
                })?;
            for residual in 0..self.jacobian_workspace.nrows() {
                for variable in 0..self.jacobian_workspace.ncols() {
                    self.batch_workspace
                        .push(self.jacobian_workspace[(residual, variable)]);
                }
            }
        }
        self.record_diagnostic(started, (input.nrows() * entries) as u64);
        Ok(())
    }

    fn record_diagnostic(&mut self, started: Option<Instant>, evaluations: u64) {
        if let Some(started) = started {
            self.evaluation_count = self.evaluation_count.saturating_add(evaluations);
            self.callback_time_ns = self
                .callback_time_ns
                .saturating_add(started.elapsed().as_nanos().min(u64::MAX as u128) as u64);
        }
    }
}

#[pymethods]
impl LambdifiedSystem {
    /// Evaluate all residuals using one shared input vector.
    pub fn evaluate(&mut self, values: Vec<f64>) -> PyResult<Vec<f64>> {
        checked_values(&values, &self.variables, "system Lambdify evaluation")?;
        self.input_workspace.copy_from_slice(&values);
        let started = self.diagnostics_enabled.then(Instant::now);
        self.prepared
            .evaluate_residual_into(&self.input_workspace, &mut self.residual_workspace)
            .map_err(|error| {
                map_solve_error_context(
                    error,
                    "evaluate",
                    "symbolic_system",
                    frontend_name(self.frontend),
                )
            })?;
        self.record_diagnostic(started, self.residual_workspace.len() as u64);
        Ok(self.residual_workspace.iter().copied().collect())
    }

    /// Evaluate the precompiled residual Jacobian without rebuilding it.
    pub fn evaluate_jacobian(&mut self, values: Vec<f64>) -> PyResult<Vec<Vec<f64>>> {
        checked_values(&values, &self.variables, "system Jacobian evaluation")?;
        self.input_workspace.copy_from_slice(&values);
        let started = self.diagnostics_enabled.then(Instant::now);
        self.prepared
            .evaluate_jacobian_into(&self.input_workspace, &mut self.jacobian_workspace)
            .map_err(|error| {
                map_solve_error_context(
                    error,
                    "evaluate_jacobian",
                    "symbolic_system",
                    frontend_name(self.frontend),
                )
            })?;
        self.record_diagnostic(
            started,
            (self.jacobian_workspace.nrows() * self.jacobian_workspace.ncols()) as u64,
        );
        Ok((0..self.jacobian_workspace.nrows())
            .map(|row| {
                (0..self.jacobian_workspace.ncols())
                    .map(|column| self.jacobian_workspace[(row, column)])
                    .collect()
            })
            .collect())
    }

    /// Evaluate one NumPy point and return residuals with shape `(n_residuals,)`.
    pub fn evaluate_array<'py>(
        &mut self,
        py: Python<'py>,
        values: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = extract_f64_array1(values, "values")?;
        validate_point_array(&values, self.variables.len(), "system evaluation")?;
        self.input_workspace
            .iter_mut()
            .zip(values.as_array().iter())
            .for_each(|(slot, value)| *slot = *value);
        let started = self.diagnostics_enabled.then(Instant::now);
        self.prepared
            .evaluate_residual_into(&self.input_workspace, &mut self.residual_workspace)
            .map_err(|error| {
                map_solve_error_context(
                    error,
                    "evaluate",
                    "symbolic_system",
                    frontend_name(self.frontend),
                )
            })?;
        self.record_diagnostic(started, self.residual_workspace.len() as u64);
        Ok(PyArray1::from_vec(
            py,
            self.residual_workspace.iter().copied().collect(),
        ))
    }

    /// Fill and return a caller-owned residual vector of exact shape `(n_residuals,)`.
    pub fn evaluate_array_into<'py>(
        &mut self,
        values: Bound<'py, PyAny>,
        out: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = extract_f64_array1(values, "values")?;
        let out = extract_f64_output1(out)?;
        validate_point_array(&values, self.variables.len(), "system evaluation")?;
        if out.shape() != [self.residual_workspace.len()] {
            return Err(InvalidConfigurationError::new_err(format!(
                "out must have shape ({},)",
                self.residual_workspace.len()
            )));
        }
        self.input_workspace
            .iter_mut()
            .zip(values.as_array().iter())
            .for_each(|(slot, value)| *slot = *value);
        drop(values);
        let started = self.diagnostics_enabled.then(Instant::now);
        self.prepared
            .evaluate_residual_into(&self.input_workspace, &mut self.residual_workspace)
            .map_err(|error| {
                map_solve_error_context(
                    error,
                    "evaluate",
                    "symbolic_system",
                    frontend_name(self.frontend),
                )
            })?;
        self.record_diagnostic(started, self.residual_workspace.len() as u64);
        let mut target = out.try_readwrite().map_err(|error| {
            InvalidConfigurationError::new_err(format!(
                "out must be writable and unborrowed: {error}"
            ))
        })?;
        for (slot, value) in target
            .as_array_mut()
            .iter_mut()
            .zip(self.residual_workspace.iter())
        {
            *slot = *value;
        }
        drop(target);
        Ok(out)
    }

    /// Evaluate a NumPy point and return a dense Jacobian `(n_residuals, n_variables)`.
    pub fn evaluate_jacobian_array<'py>(
        &mut self,
        py: Python<'py>,
        values: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let values = extract_f64_array1(values, "values")?;
        validate_point_array(&values, self.variables.len(), "system Jacobian evaluation")?;
        self.input_workspace
            .iter_mut()
            .zip(values.as_array().iter())
            .for_each(|(slot, value)| *slot = *value);
        let started = self.diagnostics_enabled.then(Instant::now);
        self.prepared
            .evaluate_jacobian_into(&self.input_workspace, &mut self.jacobian_workspace)
            .map_err(|error| {
                map_solve_error_context(
                    error,
                    "evaluate_jacobian",
                    "symbolic_system",
                    frontend_name(self.frontend),
                )
            })?;
        self.record_diagnostic(
            started,
            (self.jacobian_workspace.nrows() * self.jacobian_workspace.ncols()) as u64,
        );
        let output = PyArray2::zeros(
            py,
            [
                self.jacobian_workspace.nrows(),
                self.jacobian_workspace.ncols(),
            ],
            false,
        );
        let mut target = output.readwrite();
        let mut target = target.as_array_mut();
        for row in 0..self.jacobian_workspace.nrows() {
            for column in 0..self.jacobian_workspace.ncols() {
                target[(row, column)] = self.jacobian_workspace[(row, column)];
            }
        }
        drop(target);
        Ok(output)
    }

    /// Fill and return a caller-owned dense Jacobian array of exact shape `(n_residuals, n_variables)`.
    pub fn evaluate_jacobian_array_into<'py>(
        &mut self,
        values: Bound<'py, PyAny>,
        out: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let values = extract_f64_array1(values, "values")?;
        let out = extract_f64_output2(out)?;
        validate_point_array(&values, self.variables.len(), "system Jacobian evaluation")?;
        let expected = [
            self.jacobian_workspace.nrows(),
            self.jacobian_workspace.ncols(),
        ];
        if out.shape() != expected {
            return Err(InvalidConfigurationError::new_err(format!(
                "out must have shape {expected:?}"
            )));
        }
        self.input_workspace
            .iter_mut()
            .zip(values.as_array().iter())
            .for_each(|(slot, value)| *slot = *value);
        drop(values);
        let started = self.diagnostics_enabled.then(Instant::now);
        self.prepared
            .evaluate_jacobian_into(&self.input_workspace, &mut self.jacobian_workspace)
            .map_err(|error| {
                map_solve_error_context(
                    error,
                    "evaluate_jacobian",
                    "symbolic_system",
                    frontend_name(self.frontend),
                )
            })?;
        self.record_diagnostic(
            started,
            (self.jacobian_workspace.nrows() * self.jacobian_workspace.ncols()) as u64,
        );
        let mut target = out.try_readwrite().map_err(|error| {
            InvalidConfigurationError::new_err(format!(
                "out must be writable and unborrowed: {error}"
            ))
        })?;
        for ((row, column), slot) in target.as_array_mut().indexed_iter_mut() {
            *slot = self.jacobian_workspace[(row, column)];
        }
        drop(target);
        Ok(out)
    }

    /// Evaluate a batch with shape `(n_samples, n_variables)`.
    ///
    /// Strided NumPy inputs are read directly; the persistent native input,
    /// residual/Jacobian and batch scratch buffers are reused between calls.
    pub fn evaluate_batch<'py>(
        &mut self,
        py: Python<'py>,
        values: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let values = extract_f64_array2(values, "values")?;
        self.evaluate_batch_values(values)?;
        let samples = self.batch_rows;
        let output = PyArray2::zeros(py, [samples, self.residual_workspace.len()], false);
        let mut target = output.readwrite();
        let mut target = target.as_array_mut();
        for (index, value) in self.batch_workspace.iter().copied().enumerate() {
            target[(
                index / self.residual_workspace.len(),
                index % self.residual_workspace.len(),
            )] = value;
        }
        drop(target);
        Ok(output)
    }

    /// Fill and return a caller-owned `(n_samples, n_residuals)` float64 array.
    pub fn evaluate_batch_into<'py>(
        &mut self,
        values: Bound<'py, PyAny>,
        out: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let values = extract_f64_array2(values, "values")?;
        let out = extract_f64_output2(out)?;
        let expected = [values.shape()[0], self.residual_workspace.len()];
        if out.shape() != expected {
            return Err(InvalidConfigurationError::new_err(format!(
                "out must have shape {expected:?}",
            )));
        }
        self.evaluate_batch_values(values)?;
        let expected = (self.batch_rows, self.residual_workspace.len());
        let mut writable = out.try_readwrite().map_err(|error| {
            InvalidConfigurationError::new_err(format!(
                "out must be writable and unborrowed: {error}"
            ))
        })?;
        let mut target = writable.as_array_mut();
        for (index, value) in self.batch_workspace.iter().copied().enumerate() {
            target[(index / expected.1, index % expected.1)] = value;
        }
        drop(writable);
        Ok(out)
    }

    /// Evaluate dense Jacobians with output shape `(n_samples, n_residuals, n_variables)`.
    pub fn evaluate_jacobian_batch<'py>(
        &mut self,
        py: Python<'py>,
        values: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let values = extract_f64_array2(values, "values")?;
        self.evaluate_jacobian_batch_values(values)?;
        let samples = self.batch_rows;
        let output = PyArray3::zeros(
            py,
            [samples, self.residual_workspace.len(), self.variables.len()],
            false,
        );
        let mut target = output.readwrite();
        let mut target = target.as_array_mut();
        for (index, value) in self.batch_workspace.iter().copied().enumerate() {
            let column = index % self.variables.len();
            let row = (index / self.variables.len()) % self.residual_workspace.len();
            let sample = index / (self.residual_workspace.len() * self.variables.len());
            target[(sample, row, column)] = value;
        }
        drop(target);
        Ok(output)
    }

    /// Fill and return a caller-owned dense Jacobian batch array.
    pub fn evaluate_jacobian_batch_into<'py>(
        &mut self,
        values: Bound<'py, PyAny>,
        out: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let values = extract_f64_array2(values, "values")?;
        let out = extract_f64_output3(out)?;
        let expected = [
            values.shape()[0],
            self.residual_workspace.len(),
            self.variables.len(),
        ];
        if out.shape() != expected {
            return Err(InvalidConfigurationError::new_err(format!(
                "out must have shape {expected:?}"
            )));
        }
        self.evaluate_jacobian_batch_values(values)?;
        let mut writable = out.try_readwrite().map_err(|error| {
            InvalidConfigurationError::new_err(format!(
                "out must be writable and unborrowed: {error}"
            ))
        })?;
        let mut target = writable.as_array_mut();
        for (index, value) in self.batch_workspace.iter().copied().enumerate() {
            let column = index % expected[2];
            let row = (index / expected[2]) % expected[1];
            let sample = index / (expected[1] * expected[2]);
            target[(sample, row, column)] = value;
        }
        drop(writable);
        Ok(out)
    }

    /// Return the input order expected by `evaluate`.
    pub fn variables(&self) -> Vec<String> {
        self.variables.clone()
    }

    /// Return the selected symbolic frontend for this prepared system.
    #[getter]
    pub fn frontend(&self) -> &'static str {
        frontend_name(self.frontend)
    }

    /// Return the number of equations prepared in the native residual provider.
    #[getter]
    pub fn residual_count(&self) -> usize {
        self.residual_workspace.len()
    }

    /// Return `(n_residuals, n_variables)` for the dense Jacobian route.
    #[getter]
    pub fn jacobian_shape(&self) -> (usize, usize) {
        (self.residual_workspace.len(), self.variables.len())
    }

    /// Return explicit native preparation provenance; this route is dense Lambdify.
    pub fn provenance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let report = PyDict::new(py);
        report.set_item("requested_frontend", frontend_name(self.frontend))?;
        report.set_item(
            "effective_frontend",
            self.prepared.lambdify_frontend().as_str().replace('-', "_"),
        )?;
        report.set_item("backend", self.prepared.backend_kind().as_str())?;
        report.set_item("residual_count", self.residual_workspace.len())?;
        report.set_item("variable_count", self.variables.len())?;
        report.set_item(
            "jacobian_shape",
            (self.residual_workspace.len(), self.variables.len()),
        )?;
        report.set_item("jacobian_structure", "dense")?;
        report.set_item("structural_nnz", None::<usize>)?;
        report.set_item("preparation_count", 1_u64)?;
        Ok(report)
    }

    /// Return whether preparation and callback timing/counters were enabled.
    #[getter]
    pub fn diagnostics_enabled(&self) -> bool {
        self.diagnostics_enabled
    }

    /// Return native evaluation counters and the optional symbolic preparation timeline.
    pub fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let report = PyDict::new(py);
        report.set_item("enabled", self.diagnostics_enabled)?;
        report.set_item("evaluations", self.evaluation_count)?;
        report.set_item(
            "callback_time_ms",
            self.callback_time_ns as f64 / 1_000_000.0,
        )?;
        report.set_item("preparation_ms", self.preparation_ms)?;
        if let Some(telemetry) = &self.prepared.preparation_report().detailed {
            let stages = PyList::empty(py);
            for stage in &telemetry.stages {
                let item = PyDict::new(py);
                item.set_item("name", preparation_stage_name(stage.stage))?;
                item.set_item(
                    "wall_time_ms",
                    stage
                        .wall_time
                        .map(|duration| duration.as_secs_f64() * 1_000.0),
                )?;
                item.set_item("calls", stage.calls)?;
                item.set_item("items", stage.items)?;
                stages.append(item)?;
            }
            report.set_item("preparation_stages", stages)?;
            report.set_item(
                "preparation_unattributed_ms",
                telemetry.unattributed_wall_time.as_secs_f64() * 1_000.0,
            )?;
            report.set_item("preparation_reused", telemetry.reused_prepared_problem)?;
        } else {
            report.set_item("preparation_stages", None::<()>)?;
            report.set_item("preparation_unattributed_ms", None::<f64>)?;
            report.set_item("preparation_reused", None::<bool>)?;
        }
        Ok(report)
    }

    /// Return native preparation time when diagnostics were enabled.
    #[getter]
    pub fn preparation_ms(&self) -> Option<f64> {
        self.preparation_ms
    }

    fn __repr__(&self) -> String {
        format!(
            "LambdifiedSystem(size={}, variables={:?})",
            self.residual_workspace.len(),
            self.variables
        )
    }
}
