//! Reusable prepared residual/Jacobian runtime for packed Atom expressions.
//!
//! This module is deliberately solver-independent.  A caller prepares an
//! immutable residual and symbolic Jacobian plan once, then reuses caller-owned
//! buffers for repeated evaluations.  Dense, sparse, Banded, Python and
//! solver-specific adapters can consume the same ordered Jacobian values
//! without knowing the packed Atom layout.

use std::{collections::HashSet, sync::Arc};

use thiserror::Error;

use super::{
    Atom, FunctionMap, PreparedEvaluationError, PreparedEvaluationScratch, PreparedEvaluator,
    PreparedSparseAtomSystem, SparseAtomJacobianEntry, Symbol,
    conversions::{AtomConversionError, try_expr_to_atom},
};
use crate::symbolic::symbolic_engine::Expr;

/// A stable row/column location in the prepared Jacobian value stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedJacobianEntry {
    /// Residual row containing this derivative.
    pub row: usize,
    /// Input column differentiated against.
    pub col: usize,
}

/// Caller-owned reusable workspace for [`PreparedAtomSystem`].
///
/// One scratch value must not be shared by concurrent evaluations.  Sharing
/// the immutable prepared system is cheap; each worker should own one scratch
/// value so the hot path never needs a global mutex.
#[derive(Default)]
pub struct PreparedAtomSystemScratch {
    evaluator: PreparedEvaluationScratch,
}

impl PreparedAtomSystemScratch {
    /// Return the current reusable scalar workspace capacity.
    ///
    /// This is intended for diagnostics and retention tests, not for sizing
    /// application buffers manually.
    pub fn capacity(&self) -> usize {
        self.evaluator.capacity()
    }
}

/// Typed failure from prepared residual/Jacobian preparation or execution.
#[derive(Debug, Error, PartialEq)]
pub enum PreparedAtomSystemError {
    /// One input vector was not in the immutable prepared schema.
    #[error("prepared system expected {expected} input value(s), got {actual}")]
    InputShape { expected: usize, actual: usize },
    /// The immutable input ABI cannot contain duplicate symbols.
    #[error("prepared system input symbol {symbol} is declared more than once")]
    DuplicateInputSymbol { symbol: String },
    /// Empty input names are rejected before symbol slots are compiled.
    #[error("prepared system input name at index {index} is empty")]
    EmptyInputName { index: usize },
    /// The residual destination had the wrong number of rows.
    #[error("prepared residual output expected {expected} value(s), got {actual}")]
    ResidualOutputShape { expected: usize, actual: usize },
    /// The Jacobian destination had the wrong number of stored values.
    #[error("prepared Jacobian output expected {expected} value(s), got {actual}")]
    JacobianOutputShape { expected: usize, actual: usize },
    /// A residual evaluator failed while evaluating one row.
    #[error("prepared residual row {row} failed: {source}")]
    ResidualEvaluation {
        row: usize,
        #[source]
        source: PreparedEvaluationError,
    },
    /// A residual evaluator could not be compiled during preparation.
    #[error("prepared residual row {row} could not be compiled: {source}")]
    ResidualPreparation {
        row: usize,
        #[source]
        source: PreparedEvaluationError,
    },
    /// A derivative evaluator failed while evaluating one entry.
    #[error("prepared Jacobian entry ({row}, {col}) failed: {source}")]
    JacobianEvaluation {
        row: usize,
        col: usize,
        #[source]
        source: PreparedEvaluationError,
    },
    /// Symbolic conversion failed before the runtime became usable.
    #[error("prepared system conversion failed: {source}")]
    Conversion {
        #[source]
        source: AtomConversionError,
    },
    /// Symbolic differentiation failed while constructing the Jacobian plan.
    #[error("prepared Jacobian differentiation failed at ({row}, {col}): {source}")]
    Differentiation {
        row: usize,
        col: usize,
        #[source]
        source: super::DerivativeError,
    },
    /// A derivative plan could not be compiled into the prepared evaluator.
    #[error("prepared Jacobian evaluator at ({row}, {col}) failed: {source}")]
    JacobianPreparation {
        row: usize,
        col: usize,
        #[source]
        source: PreparedEvaluationError,
    },
}

/// Immutable prepared residual and Jacobian execution plan.
///
/// The input order is exactly the order supplied to the constructor.  The
/// Jacobian value stream is deterministic row-major order and contains only
/// nonzero symbolic entries after optional bandwidth filtering.  Consumers
/// that need a dense matrix can scatter these values into their own storage;
/// this runtime never allocates a dense matrix on their behalf.
pub struct PreparedAtomSystem {
    input_symbols: Arc<[Symbol]>,
    residuals: Arc<[PreparedEvaluator]>,
    jacobian: Arc<[PreparedJacobianPlan]>,
}

struct PreparedJacobianPlan {
    metadata: PreparedJacobianEntry,
    evaluator: PreparedEvaluator,
}

impl PreparedAtomSystem {
    /// Prepare a numeric residual/Jacobian runtime from legacy expressions.
    ///
    /// `bandwidth` uses `(lower, upper)` and filters the symbolic Jacobian
    /// before evaluator compilation.  Passing `None` keeps all dependencies.
    pub fn try_from_exprs(
        residuals: &[Expr],
        input_names: &[String],
        bandwidth: Option<(usize, usize)>,
    ) -> Result<Self, PreparedAtomSystemError> {
        validate_input_names(input_names)?;
        let atoms = residuals
            .iter()
            .map(try_expr_to_atom)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|source| PreparedAtomSystemError::Conversion { source })?;
        Self::try_from_atoms(&atoms, input_names, bandwidth, FunctionMap::default())
    }

    /// Prepare a runtime from already packed residual atoms.
    pub fn try_from_atoms(
        residuals: &[Atom],
        input_names: &[String],
        bandwidth: Option<(usize, usize)>,
        function_map: FunctionMap,
    ) -> Result<Self, PreparedAtomSystemError> {
        validate_input_names(input_names)?;
        let sparse =
            PreparedSparseAtomSystem::from_atoms_discovering_dependencies(residuals, input_names);
        let input_symbols: Arc<[Symbol]> = sparse.variable_symbols().to_vec().into();
        let context = super::evaluate::PreparedVariableContext::new(&input_symbols);

        let residuals = residuals
            .iter()
            .enumerate()
            .map(|(row, atom)| {
                PreparedEvaluator::new_with_context_typed(atom, &context, &function_map)
                    .map_err(|source| PreparedAtomSystemError::ResidualPreparation { row, source })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let entries = sparse
            .try_calc_sparse_jacobian_with_bandwidth(bandwidth)
            .map_err(|error| match error.source {
                source => PreparedAtomSystemError::Differentiation {
                    row: error.row,
                    col: error.col,
                    source,
                },
            })?;

        let jacobian = entries
            .into_iter()
            .map(|entry: SparseAtomJacobianEntry| {
                let metadata = PreparedJacobianEntry {
                    row: entry.row,
                    col: entry.col,
                };
                PreparedEvaluator::new_with_context_typed(&entry.value, &context, &function_map)
                    .map(|evaluator| PreparedJacobianPlan {
                        metadata,
                        evaluator,
                    })
                    .map_err(|source| PreparedAtomSystemError::JacobianPreparation {
                        row: metadata.row,
                        col: metadata.col,
                        source,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            input_symbols,
            residuals: residuals.into(),
            jacobian: jacobian.into(),
        })
    }

    /// Number of input values required by every prepared evaluator.
    pub fn input_len(&self) -> usize {
        self.input_symbols.len()
    }

    /// Ordered input symbols used by the packed evaluation ABI.
    pub fn input_symbols(&self) -> &[Symbol] {
        &self.input_symbols
    }

    /// Number of residual equations.
    pub fn residual_len(&self) -> usize {
        self.residuals.len()
    }

    /// Number of stored Jacobian values after sparsity/bandwidth filtering.
    pub fn jacobian_len(&self) -> usize {
        self.jacobian.len()
    }

    /// Deterministic row-major Jacobian metadata.
    pub fn jacobian_entries(&self) -> impl ExactSizeIterator<Item = PreparedJacobianEntry> + '_ {
        self.jacobian.iter().map(|entry| entry.metadata)
    }

    /// Evaluate all residuals into caller-owned storage.
    pub fn residual_into(
        &self,
        input: &[f64],
        output: &mut [f64],
        scratch: &mut PreparedAtomSystemScratch,
    ) -> Result<(), PreparedAtomSystemError> {
        self.validate_input_and_output(input, output.len(), self.residuals.len(), false)?;
        for (row, evaluator) in self.residuals.iter().enumerate() {
            output[row] = evaluator
                .evaluate_with_scratch_typed(input, &mut scratch.evaluator)
                .map_err(|source| PreparedAtomSystemError::ResidualEvaluation { row, source })?;
        }
        Ok(())
    }

    /// Evaluate the stored Jacobian values into caller-owned storage.
    ///
    /// Values correspond one-to-one with [`Self::jacobian_entries`].  This
    /// avoids imposing a dense representation on sparse or banded consumers.
    pub fn jacobian_values_into(
        &self,
        input: &[f64],
        output: &mut [f64],
        scratch: &mut PreparedAtomSystemScratch,
    ) -> Result<(), PreparedAtomSystemError> {
        self.validate_input_and_output(input, output.len(), self.jacobian.len(), true)?;
        for (index, plan) in self.jacobian.iter().enumerate() {
            output[index] = plan
                .evaluator
                .evaluate_with_scratch_typed(input, &mut scratch.evaluator)
                .map_err(|source| PreparedAtomSystemError::JacobianEvaluation {
                    row: plan.metadata.row,
                    col: plan.metadata.col,
                    source,
                })?;
        }
        Ok(())
    }

    /// Evaluate residuals and Jacobian values using the same input snapshot.
    pub fn residual_and_jacobian_into(
        &self,
        input: &[f64],
        residual_output: &mut [f64],
        jacobian_output: &mut [f64],
        scratch: &mut PreparedAtomSystemScratch,
    ) -> Result<(), PreparedAtomSystemError> {
        self.residual_into(input, residual_output, scratch)?;
        self.jacobian_values_into(input, jacobian_output, scratch)
    }

    fn validate_input_and_output(
        &self,
        input: &[f64],
        actual_output: usize,
        expected_output: usize,
        jacobian: bool,
    ) -> Result<(), PreparedAtomSystemError> {
        if input.len() != self.input_symbols.len() {
            return Err(PreparedAtomSystemError::InputShape {
                expected: self.input_symbols.len(),
                actual: input.len(),
            });
        }
        if actual_output != expected_output {
            return Err(if jacobian {
                PreparedAtomSystemError::JacobianOutputShape {
                    expected: expected_output,
                    actual: actual_output,
                }
            } else {
                PreparedAtomSystemError::ResidualOutputShape {
                    expected: expected_output,
                    actual: actual_output,
                }
            });
        }
        Ok(())
    }
}

fn validate_input_names(input_names: &[String]) -> Result<(), PreparedAtomSystemError> {
    let mut seen = HashSet::with_capacity(input_names.len());
    for (index, name) in input_names.iter().enumerate() {
        if name.trim().is_empty() {
            return Err(PreparedAtomSystemError::EmptyInputName { index });
        }
        if !seen.insert(name.as_str()) {
            return Err(PreparedAtomSystemError::DuplicateInputSymbol {
                symbol: name.clone(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn prepared_runtime_evaluates_residual_and_sparse_jacobian_without_dense_storage() {
        let prepared = PreparedAtomSystem::try_from_exprs(
            &[
                Expr::parse_expression("x + y^2"),
                Expr::parse_expression("x*y"),
            ],
            &names(&["x", "y"]),
            None,
        )
        .unwrap();
        let entries = prepared.jacobian_entries().collect::<Vec<_>>();
        assert_eq!(entries.len(), 4);
        assert_eq!(entries[0], PreparedJacobianEntry { row: 0, col: 0 });
        assert_eq!(entries[1], PreparedJacobianEntry { row: 0, col: 1 });

        let mut scratch = PreparedAtomSystemScratch::default();
        let mut residual = [0.0; 2];
        let mut jacobian = [0.0; 4];
        prepared
            .residual_and_jacobian_into(&[2.0, 3.0], &mut residual, &mut jacobian, &mut scratch)
            .unwrap();
        assert_eq!(residual, [11.0, 6.0]);
        assert_eq!(jacobian, [1.0, 6.0, 3.0, 2.0]);
    }

    #[test]
    fn prepared_runtime_reports_shapes_and_reuses_scratch() {
        let prepared = PreparedAtomSystem::try_from_exprs(
            &[Expr::parse_expression("log(x)")],
            &names(&["x"]),
            None,
        )
        .unwrap();
        let mut scratch = PreparedAtomSystemScratch::default();
        let mut residual = [0.0; 1];
        prepared
            .residual_into(&[2.0], &mut residual, &mut scratch)
            .unwrap();
        let capacity = scratch.capacity();
        prepared
            .residual_into(&[3.0], &mut residual, &mut scratch)
            .unwrap();
        assert_eq!(scratch.capacity(), capacity);

        assert!(matches!(
            prepared.residual_into(&[], &mut residual, &mut scratch),
            Err(PreparedAtomSystemError::InputShape { .. })
        ));
        assert!(matches!(
            prepared.residual_into(&[1.0], &mut [], &mut scratch),
            Err(PreparedAtomSystemError::ResidualOutputShape { .. })
        ));
        let error = prepared.residual_into(&[-1.0], &mut residual, &mut scratch);
        assert!(matches!(
            error,
            Err(PreparedAtomSystemError::ResidualEvaluation { row: 0, .. })
        ));
    }

    #[test]
    fn prepared_runtime_rejects_ambiguous_input_schema_before_compilation() {
        let residuals = [Expr::parse_expression("x")];
        let duplicate = PreparedAtomSystem::try_from_exprs(&residuals, &names(&["x", "x"]), None);
        assert!(matches!(
            duplicate,
            Err(PreparedAtomSystemError::DuplicateInputSymbol { .. })
        ));

        let empty = PreparedAtomSystem::try_from_exprs(&residuals, &names(&[" "]), None);
        assert!(matches!(
            empty,
            Err(PreparedAtomSystemError::EmptyInputName { index: 0 })
        ));
    }
}
