//! Legacy Lambdify backend for nonlinear systems.
//!
//! This module intentionally preserves the historical Jacobian construction
//! and compatibility callback semantics. The prepared symbolic orchestration
//! lives in the sibling symbolic module; this file owns only the legacy
//! Lambdify evaluator implementation and its optional execution policy.

use super::symbolic::{LambdifyExecutionPolicy, SymbolicBackendKind, SymbolicEvaluationBackend};
use crate::global::THRESHOLD as SYMBOLIC_ZERO_THRESHOLD;
use crate::numerical::Nonlinear_systems::error::SolveError;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions::Jacobian;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use std::cell::RefCell;

type ScalarEvaluator = Box<dyn Fn(&[f64]) -> f64 + Send + Sync>;

thread_local! {
    /// Reused input workspace for parameterized Lambdify callbacks.
    ///
    /// The scalar expression ABI accepts one contiguous slice. Keeping this
    /// scratch per worker thread avoids allocating parameters + variables
    /// for every residual/Jacobian callback while retaining a Sync backend.
    static LAMBDIFY_INPUT_WORKSPACE: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

pub(crate) fn lambdify_input_workspace_capacity() -> usize {
    LAMBDIFY_INPUT_WORKSPACE.with(|workspace| workspace.borrow().capacity())
}

/// Concrete backend implementation that uses the existing `Jacobian` lambdify path.
pub(crate) struct LegacyLambdifySymbolicBackend {
    /// Symbolic Jacobian retained as preparation metadata and diagnostics.
    symbolic_jacobian: Vec<Vec<Expr>>,
    /// Scalar residual evaluators compiled once during preparation.
    residual_evaluators: Vec<ScalarEvaluator>,
    /// Non-zero symbolic Jacobian entries compiled once during preparation.
    jacobian_evaluators: Vec<Vec<Option<ScalarEvaluator>>>,
    /// Number of variables in the fixed evaluation schema.
    variable_count: usize,
    /// Number of parameters in the fixed evaluation schema.
    parameter_count: usize,
    /// Runtime policy used by residual/Jacobian callbacks.
    execution_policy: LambdifyExecutionPolicy,
    /// Number of non-zero Jacobian evaluators known during preparation.
    nonzero_jacobian_entries: usize,
    /// Non-zero Jacobian row positions grouped by column.
    ///
    /// `DMatrix` is column-major, so each group can be evaluated through a
    /// disjoint mutable column slice without a mutex or temporary matrix.
    jacobian_nonzero_rows_by_column: Vec<Vec<usize>>,
}

impl LegacyLambdifySymbolicBackend {
    /// Builds the lambdify backend from symbolic equations.
    ///
    /// This is the legacy-backed execution branch: symbolic equations are
    /// differentiated through [`Jacobian`] and then lambdified into callable
    /// dense residual/Jacobian closures.
    pub(crate) fn from_expressions(
        equations: &[Expr],
        variables: &[String],
        equation_parameters: Option<&[String]>,
        execution_policy: LambdifyExecutionPolicy,
    ) -> Result<Self, SolveError> {
        let mut jacobian = Jacobian::new();
        let variable_refs = variables
            .iter()
            .map(|value| value.as_str())
            .collect::<Vec<_>>();
        jacobian.set_vector_of_functions(equations.to_vec());
        jacobian.set_variables(variable_refs);

        if let Some(parameters) = equation_parameters {
            jacobian.set_params(parameters.to_vec());
            jacobian.calc_jacobian();
        } else {
            jacobian.calc_jacobian();
        }

        let mut input_names = equation_parameters
            .unwrap_or(&[])
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        input_names.extend(variables.iter().map(String::as_str));

        // The generic symbolic_functions2 closures return freshly allocated
        // DVector/DMatrix values and use a mutex for every parallel write.
        // Nonlinear solves repeatedly evaluate one point, so keep the same
        // compiled scalar expressions but fill solver-owned output buffers
        // directly on this production Lambdify path.
        let residual_evaluators = equations
            .iter()
            .map(|equation| Expr::lambdify_borrowed_thread_safe(equation, input_names.as_slice()))
            .collect();
        let symbolic_jacobian = jacobian.symbolic_jacobian;
        let mut jacobian_nonzero_rows_by_column = vec![Vec::new(); variables.len()];
        for (row_index, row) in symbolic_jacobian.iter().enumerate() {
            for (column_index, entry) in row.iter().enumerate() {
                if !entry.is_zero() {
                    jacobian_nonzero_rows_by_column[column_index].push(row_index);
                }
            }
        }
        let nonzero_jacobian_entries = jacobian_nonzero_rows_by_column.iter().map(Vec::len).sum();
        let jacobian_evaluators = symbolic_jacobian
            .iter()
            .map(|row| {
                row.iter()
                    .map(|entry| {
                        (!entry.is_zero()).then(|| {
                            Expr::lambdify_borrowed_thread_safe(entry, input_names.as_slice())
                        })
                    })
                    .collect()
            })
            .collect();

        Ok(Self {
            symbolic_jacobian,
            residual_evaluators,
            jacobian_evaluators,
            variable_count: variables.len(),
            parameter_count: equation_parameters.map_or(0, <[String]>::len),
            execution_policy,
            nonzero_jacobian_entries,
            jacobian_nonzero_rows_by_column,
        })
    }
}

impl SymbolicEvaluationBackend for LegacyLambdifySymbolicBackend {
    fn kind(&self) -> SymbolicBackendKind {
        SymbolicBackendKind::Lambdify
    }

    fn symbolic_jacobian(&self) -> Option<&[Vec<Expr>]> {
        Some(&self.symbolic_jacobian)
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
                context: "symbolic residual output",
            });
        }
        self.with_evaluation_input(
            x,
            equation_parameters,
            equation_parameter_values,
            variables,
            "symbolic residual input",
            |input| {
                if self.should_parallelize(self.residual_evaluators.len()) {
                    out.as_mut_slice()
                        .par_iter_mut()
                        .zip(self.residual_evaluators.par_iter())
                        .for_each(|(slot, evaluator)| *slot = evaluator(input));
                } else {
                    for (slot, evaluator) in out.iter_mut().zip(&self.residual_evaluators) {
                        *slot = evaluator(input);
                    }
                }
                if out.iter().any(|value| !value.is_finite()) {
                    return Err(SolveError::ResidualEvaluation(
                        "symbolic residual returned NaN or Inf".to_string(),
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
        if out.nrows() != self.jacobian_evaluators.len() || out.ncols() != self.variable_count {
            return Err(SolveError::InvalidConfig(format!(
                "symbolic Jacobian output is {}x{}, expected {}x{}",
                out.nrows(),
                out.ncols(),
                self.jacobian_evaluators.len(),
                self.variable_count
            )));
        }
        out.fill(0.0);
        self.with_evaluation_input(
            x,
            equation_parameters,
            equation_parameter_values,
            variables,
            "symbolic jacobian input",
            |input| {
                if self.should_parallelize(self.nonzero_jacobian_entries) {
                    let row_count = self.jacobian_evaluators.len();
                    out.as_mut_slice()
                        .par_chunks_mut(row_count)
                        .enumerate()
                        .for_each(|(column_index, column)| {
                            for &row_index in &self.jacobian_nonzero_rows_by_column[column_index] {
                                if let Some(evaluator) =
                                    &self.jacobian_evaluators[row_index][column_index]
                                {
                                    let value = evaluator(input);
                                    if !value.is_finite() || value.abs() > SYMBOLIC_ZERO_THRESHOLD {
                                        column[row_index] = value;
                                    }
                                }
                            }
                        });
                } else {
                    for (row_index, row) in self.jacobian_evaluators.iter().enumerate() {
                        for (column_index, evaluator) in row.iter().enumerate() {
                            if let Some(evaluator) = evaluator {
                                let value = evaluator(input);
                                if !value.is_finite() {
                                    return Err(SolveError::JacobianEvaluation(
                                        "symbolic jacobian returned NaN or Inf".to_string(),
                                    ));
                                }
                                if value.abs() > SYMBOLIC_ZERO_THRESHOLD {
                                    out[(row_index, column_index)] = value;
                                }
                            }
                        }
                    }
                }
                if out.iter().any(|value| !value.is_finite()) {
                    return Err(SolveError::JacobianEvaluation(
                        "symbolic jacobian returned NaN or Inf".to_string(),
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
        let mut result = DVector::zeros(self.residual_evaluators.len());
        self.residual_into(
            x,
            equation_parameters,
            equation_parameter_values,
            variables,
            &mut result,
        )?;
        Ok(result)
    }

    fn jacobian(
        &self,
        x: &DVector<f64>,
        equation_parameters: Option<&[String]>,
        equation_parameter_values: Option<&DVector<f64>>,
        variables: &[String],
    ) -> Result<DMatrix<f64>, SolveError> {
        let mut result = DMatrix::zeros(self.jacobian_evaluators.len(), self.variable_count);
        self.jacobian_into(
            x,
            equation_parameters,
            equation_parameter_values,
            variables,
            &mut result,
        )?;
        Ok(result)
    }
}

impl LegacyLambdifySymbolicBackend {
    fn should_parallelize(&self, work: usize) -> bool {
        match self.execution_policy {
            LambdifyExecutionPolicy::Sequential => false,
            LambdifyExecutionPolicy::Parallel { min_work } => work >= min_work.max(1),
        }
    }

    /// Validates the fixed symbolic schema and runs one evaluator call.
    fn with_evaluation_input(
        &self,
        x: &DVector<f64>,
        equation_parameters: Option<&[String]>,
        equation_parameter_values: Option<&DVector<f64>>,
        variables: &[String],
        context: &'static str,
        evaluate: impl FnOnce(&[f64]) -> Result<(), SolveError>,
    ) -> Result<(), SolveError> {
        if x.len() != self.variable_count || variables.len() != self.variable_count {
            return Err(SolveError::DimensionMismatch {
                expected: self.variable_count,
                actual: x.len(),
                context,
            });
        }
        let supplied_parameter_count = equation_parameters.map_or(0, <[String]>::len);
        if supplied_parameter_count != self.parameter_count {
            return Err(SolveError::DimensionMismatch {
                expected: self.parameter_count,
                actual: supplied_parameter_count,
                context: "symbolic parameter schema",
            });
        }
        if self.parameter_count == 0 {
            if equation_parameter_values.is_some() {
                return Err(SolveError::InvalidParameterSchema(
                    "symbolic evaluator received parameter values without a parameter schema"
                        .to_string(),
                ));
            }
            return evaluate(x.as_slice());
        }
        let values = equation_parameter_values.ok_or_else(|| {
            SolveError::InvalidParameterSchema(
                "symbolic evaluator requires values for the declared parameter schema".to_string(),
            )
        })?;
        if values.len() != self.parameter_count {
            return Err(SolveError::DimensionMismatch {
                expected: self.parameter_count,
                actual: values.len(),
                context: "symbolic parameter values",
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
        LAMBDIFY_INPUT_WORKSPACE.with(|workspace| {
            let mut input = workspace.borrow_mut();
            input.clear();
            input.extend(values.iter().copied());
            input.extend(x.iter().copied());
            evaluate(input.as_slice())
        })
    }
}
