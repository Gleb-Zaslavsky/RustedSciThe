//! Native packed-Atom frontend for dense nonlinear systems.
//!
//! The public nonlinear API accepts the crate's legacy [`Expr`] input because
//! that is the stable user-facing representation. This module is the native
//! execution boundary after that input has been accepted: each equation is
//! converted to [`Atom`] once, residual and Jacobian expressions stay packed,
//! and all numeric callbacks use [`PreparedEvaluator`] directly. In
//! particular, this route does not differentiate an `Expr` Jacobian and does
//! not convert the Atom Jacobian back to `Expr`.

use super::symbolic::{
    LambdifyExecutionPolicy, PreparationStage, PreparationTelemetryRecorder, SymbolicBackendKind,
    SymbolicEvaluationBackend,
};
use crate::numerical::Nonlinear_systems::error::SolveError;
use crate::symbolic::View::conversions::expr_to_atom;
use crate::symbolic::View::evaluate::PreparedVariableContext;
use crate::symbolic::View::{Atom, AtomView, FunctionMap, PreparedEvaluator, Symbol};
use crate::symbolic::symbolic_engine::Expr;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::time::Instant;

thread_local! {
    /// Reused parameter-plus-state ABI for one worker thread.
    static ATOM_INPUT_WORKSPACE: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

fn collect_variable_dependencies(
    view: AtomView<'_>,
    dependencies: &mut [bool],
    variable_indices: &HashMap<Symbol, usize>,
) {
    match view {
        AtomView::Num(_) => {}
        AtomView::Var(variable) => {
            if let Some(&index) = variable_indices.get(&variable.get_symbol()) {
                dependencies[index] = true;
            }
        }
        AtomView::Fun(function) => {
            for argument in function.iter() {
                collect_variable_dependencies(argument, dependencies, variable_indices);
            }
        }
        AtomView::Pow(power) => {
            collect_variable_dependencies(power.get_base(), dependencies, variable_indices);
            collect_variable_dependencies(power.get_exp(), dependencies, variable_indices);
        }
        AtomView::Mul(product) => {
            for factor in product.iter() {
                collect_variable_dependencies(factor, dependencies, variable_indices);
            }
        }
        AtomView::Add(sum) => {
            for term in sum.iter() {
                collect_variable_dependencies(term, dependencies, variable_indices);
            }
        }
    }
}

struct AtomJacobianEntry {
    row: usize,
    column: usize,
    evaluator: PreparedEvaluator,
}

/// Dense in-process AtomView backend for nonlinear residuals and Jacobians.
pub(crate) struct AtomNativeSymbolicBackend {
    /// Prepared residual evaluators in equation order.
    residual_evaluators: Vec<PreparedEvaluator>,
    /// Prepared non-zero Jacobian evaluators in structural metadata order.
    ///
    /// The row and column are stored beside the evaluator so callback code
    /// does not walk an `Option` matrix or perform a second row/column lookup.
    jacobian_entries: Vec<AtomJacobianEntry>,
    /// Entry indices grouped by nalgebra column for disjoint parallel writes.
    jacobian_entry_indices_by_column: Vec<Vec<usize>>,
    variable_names: Vec<String>,
    parameter_names: Vec<String>,
    execution_policy: LambdifyExecutionPolicy,
    nonzero_jacobian_entries: usize,
}

impl AtomNativeSymbolicBackend {
    /// Converts, differentiates, and compiles the complete Atom graph once.
    pub(crate) fn from_expressions(
        equations: &[Expr],
        variables: &[String],
        equation_parameters: Option<&[String]>,
        execution_policy: LambdifyExecutionPolicy,
        mut preparation_recorder: Option<&mut PreparationTelemetryRecorder>,
    ) -> Result<Self, SolveError> {
        let parameter_names = equation_parameters.unwrap_or(&[]).to_vec();
        let parameter_symbols = parameter_names
            .iter()
            .map(|name| Symbol::new(crate::wrap_symbol!(name.as_str())))
            .collect::<Vec<_>>();
        let variable_symbols = variables
            .iter()
            .map(|name| Symbol::new(crate::wrap_symbol!(name.as_str())))
            .collect::<Vec<_>>();
        let mut input_symbols = parameter_symbols;
        input_symbols.extend(variable_symbols.iter().copied());
        let context = PreparedVariableContext::new(&input_symbols);
        let function_map = FunctionMap::new();

        let conversion_started = Instant::now();
        let atoms = equations.iter().map(expr_to_atom).collect::<Vec<_>>();
        if let Some(recorder) = preparation_recorder.as_mut() {
            recorder.record(
                PreparationStage::AtomConversion,
                conversion_started,
                Some(equations.len() as u64),
                Some(equations.len()),
            );
        }

        let variable_indices = variable_symbols
            .iter()
            .copied()
            .enumerate()
            .map(|(index, symbol)| (symbol, index))
            .collect::<HashMap<_, _>>();
        let dependency_started = Instant::now();
        let mut dependency_offsets = Vec::with_capacity(atoms.len() + 1);
        let mut dependency_columns = Vec::new();
        let mut dependency_flags = vec![false; variables.len()];
        dependency_offsets.push(0);
        for equation in &atoms {
            dependency_flags.fill(false);
            collect_variable_dependencies(
                equation.as_view(),
                &mut dependency_flags,
                &variable_indices,
            );
            dependency_columns.extend(
                dependency_flags
                    .iter()
                    .enumerate()
                    .filter_map(|(column, present)| present.then_some(column)),
            );
            dependency_offsets.push(dependency_columns.len());
        }
        if let Some(recorder) = preparation_recorder.as_mut() {
            recorder.record(
                PreparationStage::AtomDependencyAnalysis,
                dependency_started,
                Some(dependency_columns.len() as u64),
                Some(equations.len() * variables.len()),
            );
        }

        let differentiation_started = Instant::now();
        let mut jacobian_atoms = Vec::<(usize, usize, Atom)>::new();
        let mut derivative_calls = 0usize;
        for (row_index, equation) in atoms.iter().enumerate() {
            let start = dependency_offsets[row_index];
            let end = dependency_offsets[row_index + 1];
            for &column_index in &dependency_columns[start..end] {
                let variable = variable_symbols[column_index];
                derivative_calls += 1;
                let derivative = equation.try_derivative(variable).map_err(|error| {
                    SolveError::InvalidConfig(format!(
                        "AtomView Jacobian differentiation failed at ({row_index}, {column_index}): {error}"
                    ))
                })?;
                if !derivative.is_zero() {
                    jacobian_atoms.push((row_index, column_index, derivative));
                }
            }
        }
        if let Some(recorder) = preparation_recorder.as_mut() {
            recorder.record(
                PreparationStage::AtomDifferentiation,
                differentiation_started,
                Some(derivative_calls as u64),
                Some(equations.len() * variables.len()),
            );
        }

        let residual_started = Instant::now();
        let residual_evaluators = atoms
            .iter()
            .map(|atom| {
                PreparedEvaluator::new_with_context(atom, &context, &function_map).map_err(
                    |error| {
                        SolveError::InvalidConfig(format!(
                            "AtomView residual evaluator preparation failed: {error}"
                        ))
                    },
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(recorder) = preparation_recorder.as_mut() {
            recorder.record(
                PreparationStage::ResidualCallbackPreparation,
                residual_started,
                Some(residual_evaluators.len() as u64),
                Some(residual_evaluators.len()),
            );
        }

        let jacobian_started = Instant::now();
        let mut jacobian_entries = Vec::with_capacity(jacobian_atoms.len());
        let mut jacobian_entry_indices_by_column = vec![Vec::new(); variables.len()];
        for (row, column, atom) in jacobian_atoms {
            let evaluator = PreparedEvaluator::new_with_context(&atom, &context, &function_map)
                .map_err(|error| {
                    SolveError::InvalidConfig(format!(
                        "AtomView Jacobian evaluator preparation failed: {error}"
                    ))
                })?;
            let entry_index = jacobian_entries.len();
            jacobian_entries.push(AtomJacobianEntry {
                row,
                column,
                evaluator,
            });
            jacobian_entry_indices_by_column[column].push(entry_index);
        }
        let nonzero_entries = jacobian_entries.len();
        if let Some(recorder) = preparation_recorder.as_mut() {
            recorder.record(
                PreparationStage::JacobianCallbackPreparation,
                jacobian_started,
                Some(nonzero_entries as u64),
                Some(nonzero_entries),
            );
        }

        Ok(Self {
            residual_evaluators,
            jacobian_entries,
            jacobian_entry_indices_by_column,
            variable_names: variables.to_vec(),
            parameter_names,
            execution_policy,
            nonzero_jacobian_entries: nonzero_entries,
        })
    }

    fn should_parallelize(&self, work: usize) -> bool {
        match self.execution_policy {
            LambdifyExecutionPolicy::Sequential => false,
            LambdifyExecutionPolicy::Parallel { min_work } => work >= min_work.max(1),
        }
    }

    fn with_input<T>(
        &self,
        x: &DVector<f64>,
        equation_parameters: Option<&[String]>,
        equation_parameter_values: Option<&DVector<f64>>,
        variables: &[String],
        context: &'static str,
        callback: impl FnOnce(&[f64]) -> Result<T, SolveError>,
    ) -> Result<T, SolveError> {
        if x.len() != self.variable_names.len() || variables != self.variable_names.as_slice() {
            return Err(SolveError::DimensionMismatch {
                expected: self.variable_names.len(),
                actual: x.len(),
                context,
            });
        }
        let supplied = equation_parameters.unwrap_or(&[]);
        if supplied != self.parameter_names.as_slice() {
            return Err(SolveError::InvalidParameterSchema(
                "AtomView evaluator received a different parameter schema".to_string(),
            ));
        }
        if self.parameter_names.is_empty() {
            if equation_parameter_values.is_some() {
                return Err(SolveError::InvalidParameterSchema(
                    "AtomView evaluator received parameter values without a schema".to_string(),
                ));
            }
            return callback(x.as_slice());
        }
        let values = equation_parameter_values.ok_or_else(|| {
            SolveError::InvalidParameterSchema(
                "AtomView evaluator requires values for the declared parameter schema".to_string(),
            )
        })?;
        if values.len() != self.parameter_names.len() {
            return Err(SolveError::DimensionMismatch {
                expected: self.parameter_names.len(),
                actual: values.len(),
                context: "AtomView parameter values",
            });
        }
        if let Some((index, value)) = values
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(SolveError::NonFiniteParameterValue { index, value });
        }
        ATOM_INPUT_WORKSPACE.with(|workspace| {
            let mut input = workspace.borrow_mut();
            input.clear();
            input.extend(values.iter().copied());
            input.extend(x.iter().copied());
            callback(input.as_slice())
        })
    }
}

impl SymbolicEvaluationBackend for AtomNativeSymbolicBackend {
    fn kind(&self) -> SymbolicBackendKind {
        SymbolicBackendKind::Lambdify
    }

    fn lambdify_execution_policy(&self) -> Option<LambdifyExecutionPolicy> {
        Some(self.execution_policy)
    }

    fn supports_residual_into(&self) -> bool {
        true
    }

    fn supports_jacobian_into(&self) -> bool {
        true
    }

    fn residual_into(
        &self,
        x: &DVector<f64>,
        equation_parameters: Option<&[String]>,
        equation_parameter_values: Option<&DVector<f64>>,
        variables: &[String],
        out: &mut DVector<f64>,
    ) -> Result<(), SolveError> {
        if out.len() != self.residual_evaluators.len() {
            return Err(SolveError::DimensionMismatch {
                expected: self.residual_evaluators.len(),
                actual: out.len(),
                context: "AtomView residual output",
            });
        }
        self.with_input(
            x,
            equation_parameters,
            equation_parameter_values,
            variables,
            "AtomView residual input",
            |input| {
                if self.should_parallelize(self.residual_evaluators.len()) {
                    out.as_mut_slice()
                        .par_iter_mut()
                        .zip(self.residual_evaluators.par_iter())
                        .try_for_each(|(slot, evaluator)| {
                            *slot = evaluator.evaluate_thread_local(input).map_err(|error| {
                                SolveError::ResidualEvaluation(format!(
                                    "AtomView residual evaluation failed: {error}"
                                ))
                            })?;
                            Ok::<(), SolveError>(())
                        })?;
                } else {
                    for (slot, evaluator) in out.iter_mut().zip(&self.residual_evaluators) {
                        *slot = evaluator.evaluate_thread_local(input).map_err(|error| {
                            SolveError::ResidualEvaluation(format!(
                                "AtomView residual evaluation failed: {error}"
                            ))
                        })?;
                    }
                }
                if out.iter().any(|value| !value.is_finite()) {
                    return Err(SolveError::ResidualEvaluation(
                        "AtomView residual returned NaN or Inf".to_string(),
                    ));
                }
                Ok(())
            },
        )
    }

    fn jacobian_into(
        &self,
        x: &DVector<f64>,
        equation_parameters: Option<&[String]>,
        equation_parameter_values: Option<&DVector<f64>>,
        variables: &[String],
        out: &mut DMatrix<f64>,
    ) -> Result<(), SolveError> {
        let rows = self.residual_evaluators.len();
        let cols = self.variable_names.len();
        if out.nrows() != rows || out.ncols() != cols {
            return Err(SolveError::InvalidConfig(format!(
                "AtomView Jacobian output is {}x{}, expected {}x{}",
                out.nrows(),
                out.ncols(),
                rows,
                cols
            )));
        }
        out.fill(0.0);
        self.with_input(
            x,
            equation_parameters,
            equation_parameter_values,
            variables,
            "AtomView Jacobian input",
            |input| {
                let evaluate_column = |column_index: usize, column: &mut [f64]| {
                    for &entry_index in &self.jacobian_entry_indices_by_column[column_index] {
                        let entry = &self.jacobian_entries[entry_index];
                        debug_assert_eq!(entry.column, column_index);
                        column[entry.row] =
                            entry
                                .evaluator
                                .evaluate_thread_local(input)
                                .map_err(|error| {
                                    SolveError::JacobianEvaluation(format!(
                                        "AtomView Jacobian evaluation failed: {error}"
                                    ))
                                })?;
                    }
                    Ok::<(), SolveError>(())
                };
                if self.should_parallelize(self.nonzero_jacobian_entries) {
                    out.as_mut_slice()
                        .par_chunks_mut(rows)
                        .enumerate()
                        .try_for_each(|(column_index, column)| {
                            evaluate_column(column_index, column)
                        })?;
                } else {
                    for column_index in 0..cols {
                        let column =
                            &mut out.as_mut_slice()[column_index * rows..(column_index + 1) * rows];
                        evaluate_column(column_index, column)?;
                    }
                }
                if out.iter().any(|value| !value.is_finite()) {
                    return Err(SolveError::JacobianEvaluation(
                        "AtomView Jacobian returned NaN or Inf".to_string(),
                    ));
                }
                Ok(())
            },
        )
    }

    fn residual(
        &self,
        x: &DVector<f64>,
        equation_parameters: Option<&[String]>,
        equation_parameter_values: Option<&DVector<f64>>,
        variables: &[String],
    ) -> Result<DVector<f64>, SolveError> {
        let mut out = DVector::zeros(self.residual_evaluators.len());
        self.residual_into(
            x,
            equation_parameters,
            equation_parameter_values,
            variables,
            &mut out,
        )?;
        Ok(out)
    }

    fn jacobian(
        &self,
        x: &DVector<f64>,
        equation_parameters: Option<&[String]>,
        equation_parameter_values: Option<&DVector<f64>>,
        variables: &[String],
    ) -> Result<DMatrix<f64>, SolveError> {
        let mut out = DMatrix::zeros(self.residual_evaluators.len(), self.variable_names.len());
        self.jacobian_into(
            x,
            equation_parameters,
            equation_parameter_values,
            variables,
            &mut out,
        )?;
        Ok(out)
    }
}
