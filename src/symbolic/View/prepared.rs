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
    /// Return the evaluator's coarse workspace-capacity proxy.
    ///
    /// This is a maximum element/bucket capacity for selected internal
    /// buffers, not a byte count, total retained-memory estimate, or
    /// allocation counter. Use it for diagnostics and retention tests only,
    /// not for sizing application buffers manually.
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
    use crate::symbolic::View::conversions::{atom_to_expr, try_expr_to_atom};

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
    fn prepared_runtime_parameterized_abi_matches_analytic_and_finite_difference_jacobian() {
        let residuals = [
            Expr::parse_expression("p*x^2 + sin(y) - q"),
            Expr::parse_expression("x*y + exp(p) + q^2"),
            Expr::parse_expression("x + 1"),
        ];
        let input_names = names(&["y", "q", "x", "p"]);
        let prepared = PreparedAtomSystem::try_from_exprs(&residuals, &input_names, None).unwrap();
        let entries = prepared.jacobian_entries().collect::<Vec<_>>();
        assert_eq!(
            entries,
            vec![
                PreparedJacobianEntry { row: 0, col: 0 },
                PreparedJacobianEntry { row: 0, col: 1 },
                PreparedJacobianEntry { row: 0, col: 2 },
                PreparedJacobianEntry { row: 0, col: 3 },
                PreparedJacobianEntry { row: 1, col: 0 },
                PreparedJacobianEntry { row: 1, col: 1 },
                PreparedJacobianEntry { row: 1, col: 2 },
                PreparedJacobianEntry { row: 1, col: 3 },
                PreparedJacobianEntry { row: 2, col: 2 },
            ]
        );

        let states = [
            [0.2, -0.3, 1.4, 0.1],
            [1.0, 0.5, -0.7, -0.2],
            [-1.1, 1.3, 2.2, 0.4],
        ];
        let mut scratch = PreparedAtomSystemScratch::default();
        let mut values = [0.0; 3];
        let mut jacobian = vec![0.0; prepared.jacobian_len()];
        let mut plus_values = [0.0; 3];
        let mut minus_values = [0.0; 3];
        let mut perturbed = [0.0; 4];

        for (state_index, state) in states.iter().enumerate() {
            prepared
                .residual_and_jacobian_into(state, &mut values, &mut jacobian, &mut scratch)
                .unwrap();

            let [y, q, x, p] = *state;
            let expected_residuals = [p * x * x + y.sin() - q, x * y + p.exp() + q * q, x + 1.0];
            for (row, (actual, expected)) in values.iter().zip(expected_residuals).enumerate() {
                assert!(
                    (actual - expected).abs() <= 1.0e-13,
                    "state {state_index}, residual {row}: {actual} != {expected}"
                );
            }

            let expected_jacobian = [
                y.cos(),
                -1.0,
                2.0 * p * x,
                x * x,
                x,
                2.0 * q,
                y,
                p.exp(),
                1.0,
            ];
            for (index, (actual, expected)) in jacobian.iter().zip(expected_jacobian).enumerate() {
                assert!(
                    (actual - expected).abs() <= 1.0e-13,
                    "state {state_index}, Jacobian entry {:?}: {actual} != {expected}",
                    entries[index]
                );
            }

            for input_column in 0..state.len() {
                perturbed.copy_from_slice(state);
                let step = 1.0e-6 * state[input_column].abs().max(1.0);
                perturbed[input_column] += step;
                prepared
                    .residual_into(&perturbed, &mut plus_values, &mut scratch)
                    .unwrap();
                perturbed[input_column] -= 2.0 * step;
                prepared
                    .residual_into(&perturbed, &mut minus_values, &mut scratch)
                    .unwrap();

                for row in 0..prepared.residual_len() {
                    let finite_difference = (plus_values[row] - minus_values[row]) / (2.0 * step);
                    if let Some(index) = entries
                        .iter()
                        .position(|entry| entry.row == row && entry.col == input_column)
                    {
                        assert!(
                            (jacobian[index] - finite_difference).abs() <= 2.0e-9,
                            "state {state_index}, Jacobian ({row}, {input_column}): {} != FD {finite_difference}",
                            jacobian[index]
                        );
                    } else {
                        assert!(
                            finite_difference.abs() <= 2.0e-9,
                            "state {state_index}, omitted Jacobian ({row}, {input_column}) has FD {finite_difference}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn prepared_runtime_preserves_sparse_structure_across_zero_and_empty_rows() {
        let residuals = [
            Expr::parse_expression("p*x + y"),
            Expr::parse_expression("0"),
        ];
        let input_names = names(&["x", "y", "p", "unused"]);
        let prepared = PreparedAtomSystem::try_from_exprs(&residuals, &input_names, None).unwrap();
        let expected_entries = [
            PreparedJacobianEntry { row: 0, col: 0 },
            PreparedJacobianEntry { row: 0, col: 1 },
            PreparedJacobianEntry { row: 0, col: 2 },
        ];
        assert_eq!(prepared.input_len(), input_names.len());
        assert_eq!(prepared.residual_len(), 2);
        assert_eq!(
            prepared.jacobian_entries().collect::<Vec<_>>(),
            expected_entries
        );

        let mut scratch = PreparedAtomSystemScratch::default();
        let mut residual_values = [f64::NAN; 2];
        let mut jacobian_values = [f64::NAN; 3];
        let mut previous_entries = None;
        for p in [1.0, 0.0, -1.0] {
            prepared
                .residual_and_jacobian_into(
                    &[2.0, 3.0, p, 99.0],
                    &mut residual_values,
                    &mut jacobian_values,
                    &mut scratch,
                )
                .unwrap();

            assert_eq!(residual_values, [p * 2.0 + 3.0, 0.0]);
            assert_eq!(jacobian_values, [p, 1.0, 2.0]);
            assert_eq!(
                prepared.jacobian_entries().collect::<Vec<_>>(),
                expected_entries,
                "numeric zero crossing must not mutate symbolic sparsity"
            );
            if let Some(previous) = &previous_entries {
                assert_eq!(previous, &expected_entries);
            }
            previous_entries = Some(prepared.jacobian_entries().collect::<Vec<_>>());
        }
    }

    #[test]
    fn prepared_runtime_asymmetric_bandwidth_keeps_expected_row_major_entries() {
        let diagonal_residuals = [
            Expr::parse_expression("x"),
            Expr::parse_expression("y"),
            Expr::parse_expression("z"),
        ];
        let input_names = names(&["x", "y", "z"]);
        let diagonal =
            PreparedAtomSystem::try_from_exprs(&diagonal_residuals, &input_names, None).unwrap();
        assert_eq!(
            diagonal.jacobian_entries().collect::<Vec<_>>(),
            [
                PreparedJacobianEntry { row: 0, col: 0 },
                PreparedJacobianEntry { row: 1, col: 1 },
                PreparedJacobianEntry { row: 2, col: 2 },
            ]
        );

        let residuals = [
            Expr::parse_expression("x + y + z"),
            Expr::parse_expression("x + y + z"),
            Expr::parse_expression("x + y + z"),
        ];
        let prepared =
            PreparedAtomSystem::try_from_exprs(&residuals, &input_names, Some((0, 1))).unwrap();
        let expected_entries = [
            PreparedJacobianEntry { row: 0, col: 0 },
            PreparedJacobianEntry { row: 0, col: 1 },
            PreparedJacobianEntry { row: 1, col: 0 },
            PreparedJacobianEntry { row: 1, col: 1 },
            PreparedJacobianEntry { row: 1, col: 2 },
            PreparedJacobianEntry { row: 2, col: 1 },
            PreparedJacobianEntry { row: 2, col: 2 },
        ];
        assert_eq!(
            prepared.jacobian_entries().collect::<Vec<_>>(),
            expected_entries
        );

        let mut scratch = PreparedAtomSystemScratch::default();
        let mut residual_values = [0.0; 3];
        let mut jacobian_values = [0.0; 7];
        prepared
            .residual_and_jacobian_into(
                &[2.0, 3.0, 5.0],
                &mut residual_values,
                &mut jacobian_values,
                &mut scratch,
            )
            .unwrap();
        assert_eq!(residual_values, [10.0; 3]);
        assert_eq!(jacobian_values, [1.0; 7]);
    }

    #[test]
    fn expr_atom_derivative_normalization_and_prepared_jacobian_are_componentwise_equal() {
        let names = ["x", "y", "p"];
        let input_names = names
            .iter()
            .map(|name| (*name).to_string())
            .collect::<Vec<_>>();
        let residuals = [
            Expr::parse_expression("x-y+p"),
            Expr::parse_expression("(1+x)^-2 + y^2/3 + p"),
            Expr::parse_expression("exp(0.2*x)+sin(y)+log(p)"),
            Expr::parse_expression("(1+x)^(y/2) + p*x*y"),
            Expr::parse_expression("1e-12*x+1e12*y-p"),
        ];
        let prepared = PreparedAtomSystem::try_from_exprs(&residuals, &input_names, None).unwrap();
        let entries = prepared.jacobian_entries().collect::<Vec<_>>();
        let states = [[1.3, 0.7, 2.0], [0.8, 1.2, 0.5], [2.1, 0.4, 3.0]];
        let mut scratch = PreparedAtomSystemScratch::default();
        let mut residual_values = vec![0.0; residuals.len()];
        let mut prepared_jacobian = vec![0.0; prepared.jacobian_len()];

        for (state_index, state) in states.iter().enumerate() {
            prepared
                .residual_and_jacobian_into(
                    state,
                    &mut residual_values,
                    &mut prepared_jacobian,
                    &mut scratch,
                )
                .unwrap();

            for (row, residual) in residuals.iter().enumerate() {
                let atom = try_expr_to_atom(residual).unwrap();
                for (column, name) in names.iter().enumerate() {
                    let expr_derivative = residual.diff(name);
                    let expr_eval = expr_derivative.lambdify_borrowed_thread_safe(&names);
                    let expected = expr_eval(state);

                    let symbol = Symbol::new(crate::wrap_symbol!(name));
                    let atom_derivative = atom.try_derivative(symbol).unwrap();
                    let atom_eval = atom_derivative.lambdify_borrowed_thread_safe(&names);
                    let atom_value = atom_eval(state);
                    let roundtrip_expr = atom_to_expr(&atom_derivative);
                    let roundtrip_eval = roundtrip_expr.lambdify_borrowed_thread_safe(&names);
                    let roundtrip_value = roundtrip_eval(state);

                    for (stage, actual) in [
                        ("AtomNative", atom_value),
                        ("Atom-to-Expr", roundtrip_value),
                    ] {
                        let tolerance = 2.0e-10 * expected.abs().max(1.0);
                        assert!(
                            (actual - expected).abs() <= tolerance,
                            "state {state_index}, row {row}, column {column}, {stage}: {actual} != Expr derivative {expected}"
                        );
                    }

                    let prepared_value = entries
                        .iter()
                        .position(|entry| entry.row == row && entry.col == column)
                        .map(|index| prepared_jacobian[index])
                        .unwrap_or(0.0);
                    let tolerance = 2.0e-10 * expected.abs().max(1.0);
                    assert!(
                        (prepared_value - expected).abs() <= tolerance,
                        "state {state_index}, row {row}, column {column}, PreparedAtomSystem: {prepared_value} != Expr derivative {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn prepared_runtime_large_sparse_rebind_series_preserves_values_and_capacity() {
        const N: usize = 128;
        const REBINDS: usize = 256;

        let input_names = (0..N).map(|index| format!("y{index}")).collect::<Vec<_>>();
        let residuals = (0..N)
            .map(|row| {
                if row == 0 {
                    Expr::parse_expression("y0^2")
                } else {
                    Expr::parse_expression(&format!("y{row}^2 + 0.125*y{}", row - 1))
                }
            })
            .collect::<Vec<_>>();
        let prepared = PreparedAtomSystem::try_from_exprs(&residuals, &input_names, None).unwrap();
        assert_eq!(prepared.residual_len(), N);
        assert_eq!(prepared.jacobian_len(), 2 * N - 1);

        let mut scratch = PreparedAtomSystemScratch::default();
        let mut residual_values = vec![0.0; N];
        let mut jacobian_values = vec![0.0; prepared.jacobian_len()];
        let mut input = (0..N)
            .map(|index| 0.5 + index as f64 * 1.0e-3)
            .collect::<Vec<_>>();

        prepared
            .residual_and_jacobian_into(
                &input,
                &mut residual_values,
                &mut jacobian_values,
                &mut scratch,
            )
            .unwrap();
        let warmed_capacity = scratch.capacity();

        for rebind in 0..REBINDS {
            let shift = rebind as f64 * 1.0e-6;
            for (index, value) in input.iter_mut().enumerate() {
                *value = 0.5 + index as f64 * 1.0e-3 + shift;
            }

            prepared
                .residual_and_jacobian_into(
                    &input,
                    &mut residual_values,
                    &mut jacobian_values,
                    &mut scratch,
                )
                .unwrap();

            for (row, actual) in residual_values.iter().enumerate() {
                let previous = if row == 0 {
                    0.0
                } else {
                    0.125 * input[row - 1]
                };
                let expected = input[row] * input[row] + previous;
                assert!(
                    (actual - expected).abs() <= 1.0e-13,
                    "residual row {row} drifted on rebind {rebind}: {actual} != {expected}"
                );
            }

            for (entry, actual) in prepared.jacobian_entries().zip(&jacobian_values) {
                let expected = if entry.col == entry.row {
                    2.0 * input[entry.row]
                } else {
                    assert_eq!(entry.col + 1, entry.row);
                    0.125
                };
                assert!(
                    (actual - expected).abs() <= 1.0e-13,
                    "Jacobian entry ({}, {}) drifted on rebind {rebind}: {actual} != {expected}",
                    entry.row,
                    entry.col
                );
            }

            assert_eq!(
                scratch.capacity(),
                warmed_capacity,
                "prepared scratch capacity grew after warmup on rebind {rebind}"
            );
        }
    }

    #[test]
    fn prepared_runtime_shared_plan_uses_worker_owned_scratch_concurrently() {
        const N: usize = 32;
        const WORKERS: usize = 4;
        const REBINDS: usize = 64;

        let input_names = (0..N).map(|index| format!("y{index}")).collect::<Vec<_>>();
        let residuals = (0..N)
            .map(|row| {
                if row == 0 {
                    Expr::parse_expression("y0^2")
                } else {
                    Expr::parse_expression(&format!("y{row}^2 + 0.125*y{}", row - 1))
                }
            })
            .collect::<Vec<_>>();
        let prepared = PreparedAtomSystem::try_from_exprs(&residuals, &input_names, None).unwrap();

        // Build a sequential reference for the exact inputs used by each
        // worker, so this gate checks execution-policy parity as well as the
        // independent analytic formulas below.
        let sequential = (0..WORKERS)
            .map(|worker| {
                let mut scratch = PreparedAtomSystemScratch::default();
                let mut residual_values = vec![0.0; N];
                let mut jacobian_values = vec![0.0; prepared.jacobian_len()];
                let mut input = vec![0.0; N];
                let mut outputs = Vec::with_capacity(REBINDS);

                for rebind in 0..REBINDS {
                    let shift = worker as f64 * 0.01 + rebind as f64 * 1.0e-5;
                    for (index, value) in input.iter_mut().enumerate() {
                        *value = 0.5 + index as f64 * 1.0e-3 + shift;
                    }

                    prepared
                        .residual_and_jacobian_into(
                            &input,
                            &mut residual_values,
                            &mut jacobian_values,
                            &mut scratch,
                        )
                        .unwrap();
                    outputs.push((residual_values.clone(), jacobian_values.clone()));
                }

                outputs
            })
            .collect::<Vec<_>>();

        let worker_capacities = std::thread::scope(|scope| {
            let handles = (0..WORKERS)
                .map(|worker| {
                let prepared = &prepared;
                let sequential_worker = &sequential[worker];
                scope.spawn(move || {
                    let mut scratch = PreparedAtomSystemScratch::default();
                    let mut residual_values = vec![0.0; N];
                    let mut jacobian_values = vec![0.0; prepared.jacobian_len()];
                    let mut input = vec![0.0; N];
                    let mut warmed_capacity = None;

                    for rebind in 0..REBINDS {
                        let shift = worker as f64 * 0.01 + rebind as f64 * 1.0e-5;
                        for (index, value) in input.iter_mut().enumerate() {
                            *value = 0.5 + index as f64 * 1.0e-3 + shift;
                        }

                        prepared
                            .residual_and_jacobian_into(
                                &input,
                                &mut residual_values,
                                &mut jacobian_values,
                                &mut scratch,
                            )
                            .unwrap();

                        let current_capacity = scratch.capacity();
                        if let Some(expected_capacity) = warmed_capacity {
                            assert_eq!(
                                current_capacity, expected_capacity,
                                "worker {worker}, rebind {rebind}: scratch high-water capacity grew"
                            );
                        } else {
                            warmed_capacity = Some(current_capacity);
                        }

                        let (expected_residual, expected_jacobian) = &sequential_worker[rebind];
                        assert_eq!(
                            &residual_values, expected_residual,
                            "worker {worker}, rebind {rebind}: residual differs from sequential execution"
                        );
                        assert_eq!(
                            &jacobian_values, expected_jacobian,
                            "worker {worker}, rebind {rebind}: Jacobian differs from sequential execution"
                        );

                        for (row, actual) in residual_values.iter().enumerate() {
                            let previous = if row == 0 {
                                0.0
                            } else {
                                0.125 * input[row - 1]
                            };
                            let expected = input[row] * input[row] + previous;
                            assert!(
                                (actual - expected).abs() <= 1.0e-13,
                                "worker {worker}, rebind {rebind}, residual row {row}: {actual} != {expected}"
                            );
                        }

                        for (entry, actual) in
                            prepared.jacobian_entries().zip(&jacobian_values)
                        {
                            let expected = if entry.col == entry.row {
                                2.0 * input[entry.row]
                            } else {
                                assert_eq!(entry.col + 1, entry.row);
                                0.125
                            };
                            assert!(
                                (actual - expected).abs() <= 1.0e-13,
                                "worker {worker}, rebind {rebind}, Jacobian ({}, {}): {actual} != {expected}",
                                entry.row,
                                entry.col
                            );
                        }
                    }
                    warmed_capacity.expect("each worker must evaluate at least once")
                })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("worker parity thread must succeed"))
                .collect::<Vec<_>>()
        });
        assert!(
            worker_capacities
                .iter()
                .all(|capacity| *capacity == worker_capacities[0]),
            "identical worker workloads should retain the same scratch high-water capacity: {worker_capacities:?}"
        );
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
