//! AtomView-native Lambdify callback builders for BVP systems.
//!
//! This module intentionally does not belong to `legacy_lambdify`: the
//! symbolic input is already a packed [`Atom`] graph and must not be converted
//! to `Expr` merely to reach the runtime callback. The output trait objects
//! remain compatible with the historical solver boundary, but preparation and
//! evaluation stay on the AtomView side.

use super::telemetry::{BvpLambdifyExecutionPolicy, BvpLambdifyTelemetry, flatten_lambdify_args};
use crate::global::THRESHOLD as T;
use crate::numerical::BVP_Damp::BVP_traits::{Fun, FunEnum, Jac, JacEnum};
use crate::symbolic::View::atom::Atom;
use crate::symbolic::View::evaluate::{FunctionMap, PreparedVariableContext};
use crate::symbolic::View::jacobian::SparseAtomJacobianEntry;
use crate::symbolic::View::lambdify::lambdify_with_context;
use crate::wrap_symbol;
use faer::col::{Col, ColRef};
use faer::sparse::{SparseColMat, SymbolicSparseColMat, Triplet};
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use std::borrow::Cow;

use super::legacy::BvpBackendIntegrationError;
use super::parameter_binding::BvpParameterBindingHandle;

fn validate_parameter_binding(
    parameter_count: usize,
    parameter_values: Option<&[f64]>,
) -> Result<(), BvpBackendIntegrationError> {
    match (parameter_count, parameter_values) {
        (0, None) | (0, Some([])) => Ok(()),
        (expected, Some(values)) if values.len() == expected => Ok(()),
        (expected, Some(values)) => Err(BvpBackendIntegrationError::InvalidSolverConfiguration {
            field: "parameter_values".to_string(),
            value: format!("{} values", values.len()),
            message: format!("expected exactly {expected} bound parameter values"),
        }),
        (expected, None) => Err(BvpBackendIntegrationError::InvalidSolverConfiguration {
            field: "parameter_values".to_string(),
            value: "missing".to_string(),
            message: format!("the prepared AtomView callback requires {expected} parameter values"),
        }),
    }
}

fn validate_sparse_coordinates(
    entries: &[SparseAtomJacobianEntry],
    rows: usize,
    cols: usize,
) -> Result<(), BvpBackendIntegrationError> {
    if rows == 0 || cols == 0 {
        return Err(BvpBackendIntegrationError::InvalidProblem {
            field: "atom_jacobian_shape".to_string(),
            message: format!("AtomView Jacobian shape must be non-empty, got {rows}x{cols}"),
        });
    }
    if let Some(entry) = entries
        .iter()
        .find(|entry| entry.row >= rows || entry.col >= cols)
    {
        return Err(BvpBackendIntegrationError::InvalidProblem {
            field: "atom_jacobian_sparse_coordinates".to_string(),
            message: format!(
                "entry ({}, {}) is outside Jacobian shape {rows}x{cols}",
                entry.row, entry.col
            ),
        });
    }
    Ok(())
}

/// Validates the AtomView callback inputs before an infallible compatibility
/// trait object is installed.
pub(crate) fn validate_jacobian_inputs(
    entries: &[SparseAtomJacobianEntry],
    rows: usize,
    cols: usize,
    parameter_values: Option<&[f64]>,
    parameter_count: usize,
) -> Result<(), BvpBackendIntegrationError> {
    validate_parameter_binding(parameter_count, parameter_values)?;
    validate_sparse_coordinates(entries, rows, cols)
}

/// Validates the AtomView residual binding before callback installation.
pub(crate) fn validate_residual_inputs(
    functions: &[Atom],
    parameter_values: Option<&[f64]>,
    parameter_count: usize,
) -> Result<(), BvpBackendIntegrationError> {
    if functions.is_empty() {
        return Err(BvpBackendIntegrationError::InvalidProblem {
            field: "atom_residuals".to_string(),
            message: "AtomView residual system must contain at least one equation".to_string(),
        });
    }
    validate_parameter_binding(parameter_count, parameter_values)
}

fn variable_context(argument_names: &[String]) -> std::sync::Arc<PreparedVariableContext> {
    let symbols = argument_names
        .iter()
        .map(|name| crate::symbolic::View::state::Symbol::new(wrap_symbol!(name.as_str())))
        .collect::<Vec<_>>();
    std::sync::Arc::new(PreparedVariableContext::new(symbols.as_slice()))
}

#[inline]
fn callback_arguments<'a>(
    parameter_values: Option<&[f64]>,
    unknowns: &'a [f64],
    parameter_count: usize,
) -> Cow<'a, [f64]> {
    if parameter_count == 0 {
        Cow::Borrowed(unknowns)
    } else {
        Cow::Owned(flatten_lambdify_args(parameter_values, unknowns))
    }
}

/// Installs a dense callback from already differentiated packed Atom entries.
///
/// The entry list is sparse by design; the callback only scatters evaluated
/// nonzeros into the dense matrix required by the compatibility trait.
pub(crate) fn dense_jacobian(
    entries: Vec<SparseAtomJacobianEntry>,
    rows: usize,
    cols: usize,
    argument_names: Vec<String>,
    parameter_values: Option<Vec<f64>>,
    parameter_count: usize,
    telemetry: BvpLambdifyTelemetry,
) -> Box<dyn Jac> {
    let binding = BvpParameterBindingHandle::new(parameter_values);
    dense_jacobian_with_binding(
        entries,
        rows,
        cols,
        argument_names,
        binding,
        parameter_count,
        telemetry,
    )
}

/// Dense AtomView Jacobian callback using a reusable numeric parameter bind.
pub(crate) fn dense_jacobian_with_binding(
    entries: Vec<SparseAtomJacobianEntry>,
    rows: usize,
    cols: usize,
    argument_names: Vec<String>,
    binding: BvpParameterBindingHandle,
    parameter_count: usize,
    telemetry: BvpLambdifyTelemetry,
) -> Box<dyn Jac> {
    dense_jacobian_with_binding_and_policy(
        entries,
        rows,
        cols,
        argument_names,
        binding,
        parameter_count,
        telemetry,
        BvpLambdifyExecutionPolicy::default(),
    )
}

/// Dense AtomView Jacobian callback with an explicit runtime execution policy.
pub(crate) fn dense_jacobian_with_binding_and_policy(
    entries: Vec<SparseAtomJacobianEntry>,
    rows: usize,
    cols: usize,
    argument_names: Vec<String>,
    binding: BvpParameterBindingHandle,
    parameter_count: usize,
    telemetry: BvpLambdifyTelemetry,
    execution_policy: BvpLambdifyExecutionPolicy,
) -> Box<dyn Jac> {
    validate_jacobian_inputs(
        entries.as_slice(),
        rows,
        cols,
        binding.snapshot().as_deref(),
        parameter_count,
    )
    .unwrap_or_else(|error| panic!("invalid AtomView dense Jacobian inputs: {error:?}"));
    let context = variable_context(argument_names.as_slice());
    let function_map = FunctionMap::new();
    let positions: Vec<(usize, usize, Box<dyn Fn(&[f64]) -> f64 + Send + Sync>)> = entries
        .into_par_iter()
        .map(|entry| {
            let compiled = lambdify_with_context(&entry.value, context.as_ref(), &function_map);
            (entry.row, entry.col, compiled)
        })
        .collect();
    let callback = Box::new(move |_x: f64, values: &DVector<f64>| -> DMatrix<f64> {
        let started = telemetry.start_timing();
        let parameter_values = binding.snapshot();
        assert!(
            parameter_count == 0 || parameter_values.is_some(),
            "parameter values must be provided when parameters are configured"
        );
        let args = callback_arguments(
            parameter_values.as_deref(),
            values.as_slice(),
            parameter_count,
        );
        let mut matrix = DMatrix::zeros(rows, cols);
        telemetry.record_dispatch(execution_policy, positions.len());
        if execution_policy.should_parallel(positions.len()) {
            let values: Vec<(usize, usize, f64)> = positions
                .par_iter()
                .map(|(row, col, function)| (*row, *col, function(args.as_ref())))
                .collect();
            for (row, col, value) in values {
                if value.abs() > T {
                    matrix[(row, col)] = value;
                }
            }
        } else {
            for (row, col, function) in &positions {
                let value = function(args.as_ref());
                if value.abs() > T {
                    matrix[(*row, *col)] = value;
                }
            }
        }
        telemetry.record_jacobian_sample(started);
        matrix
    });
    Box::new(JacEnum::Dense(callback))
}

/// Installs a faer sparse callback from already differentiated Atom entries.
pub(crate) fn sparse_jacobian(
    entries: Vec<SparseAtomJacobianEntry>,
    rows: usize,
    cols: usize,
    argument_names: Vec<String>,
    parameter_values: Option<Vec<f64>>,
    parameter_count: usize,
    telemetry: BvpLambdifyTelemetry,
) -> Box<dyn Jac> {
    let binding = BvpParameterBindingHandle::new(parameter_values);
    sparse_jacobian_with_binding(
        entries,
        rows,
        cols,
        argument_names,
        binding,
        parameter_count,
        telemetry,
    )
}

/// Sparse AtomView Jacobian callback using a reusable numeric parameter bind.
pub(crate) fn sparse_jacobian_with_binding(
    entries: Vec<SparseAtomJacobianEntry>,
    rows: usize,
    cols: usize,
    argument_names: Vec<String>,
    binding: BvpParameterBindingHandle,
    parameter_count: usize,
    telemetry: BvpLambdifyTelemetry,
) -> Box<dyn Jac> {
    sparse_jacobian_with_binding_and_policy(
        entries,
        rows,
        cols,
        argument_names,
        binding,
        parameter_count,
        telemetry,
        BvpLambdifyExecutionPolicy::default(),
    )
}

/// Sparse AtomView Jacobian callback with an explicit runtime execution policy.
pub(crate) fn sparse_jacobian_with_binding_and_policy(
    entries: Vec<SparseAtomJacobianEntry>,
    rows: usize,
    cols: usize,
    argument_names: Vec<String>,
    binding: BvpParameterBindingHandle,
    parameter_count: usize,
    telemetry: BvpLambdifyTelemetry,
    execution_policy: BvpLambdifyExecutionPolicy,
) -> Box<dyn Jac> {
    validate_jacobian_inputs(
        entries.as_slice(),
        rows,
        cols,
        binding.snapshot().as_deref(),
        parameter_count,
    )
    .unwrap_or_else(|error| panic!("invalid AtomView sparse Jacobian inputs: {error:?}"));
    let context = variable_context(argument_names.as_slice());
    let function_map = FunctionMap::new();
    let mut positions: Vec<(usize, usize, Box<dyn Fn(&[f64]) -> f64 + Send + Sync>)> = entries
        .into_par_iter()
        .map(|entry| {
            let compiled = lambdify_with_context(&entry.value, context.as_ref(), &function_map);
            (entry.row, entry.col, compiled)
        })
        .collect();
    // A fixed CSC pattern avoids rebuilding triplets and re-running sparse
    // coordinate analysis for every Newton callback. Duplicate coordinates
    // retain the historical triplet path because faer sums those entries.
    positions.sort_unstable_by_key(|(row, col, _)| (*col, *row));
    let has_duplicate_coordinates = positions
        .windows(2)
        .any(|pair| (pair[0].0, pair[0].1) == (pair[1].0, pair[1].1));
    let fixed_symbolic = if has_duplicate_coordinates {
        None
    } else {
        let mut col_ptr = vec![0usize; cols + 1];
        for (_, col, _) in &positions {
            col_ptr[*col + 1] += 1;
        }
        for col in 1..=cols {
            col_ptr[col] += col_ptr[col - 1];
        }
        let row_indices = positions.iter().map(|(row, _, _)| *row).collect();
        Some(SymbolicSparseColMat::new_checked(
            rows,
            cols,
            col_ptr,
            None,
            row_indices,
        ))
    };
    let callback = Box::new(
        move |_x: f64, values: &Col<f64>| -> SparseColMat<usize, f64> {
            let started = telemetry.start_timing();
            let parameter_values = binding.snapshot();
            assert!(
                parameter_count == 0 || parameter_values.is_some(),
                "parameter values must be provided when parameters are configured"
            );
            let args = callback_arguments(
                parameter_values.as_deref(),
                values
                    .try_as_col_major()
                    .expect("faer callback state must be column-major")
                    .as_slice(),
                parameter_count,
            );
            telemetry.record_dispatch(execution_policy, positions.len());
            let mut numeric_values: Vec<f64> = if execution_policy.should_parallel(positions.len())
            {
                positions
                    .par_iter()
                    .map(|(_, _, function)| function(args.as_ref()))
                    .collect()
            } else {
                positions
                    .iter()
                    .map(|(_, _, function)| function(args.as_ref()))
                    .collect()
            };
            let matrix = if let Some(symbolic) = fixed_symbolic.as_ref() {
                for value in &mut numeric_values {
                    if value.abs() <= T {
                        *value = 0.0;
                    }
                }
                SparseColMat::new(symbolic.clone(), numeric_values)
            } else {
                let triplets: Vec<Triplet<usize, usize, f64>> = positions
                    .iter()
                    .zip(numeric_values)
                    .filter_map(|((row, col, _), value)| {
                        (value.abs() > T).then(|| Triplet::new(*row, *col, value))
                    })
                    .collect();
                SparseColMat::try_new_from_triplets(rows, cols, triplets.as_slice())
                    .unwrap_or_else(|error| {
                        panic!(
                            "AtomView sparse Jacobian triplets have invalid coordinates: rows={rows}, cols={cols}, triplets={}, first={:?}, error={error:?}",
                            triplets.len(),
                            triplets.first().map(|triplet| format!("{triplet:?}"))
                        )
                    })
            };
            telemetry.record_jacobian_sample(started);
            matrix
        },
    );
    Box::new(JacEnum::Sparse_3(callback))
}

/// Installs a dense residual callback directly from packed Atom residuals.
pub(crate) fn dense_residual(
    functions: Vec<Atom>,
    argument_names: Vec<String>,
    parameter_values: Option<Vec<f64>>,
    parameter_count: usize,
    telemetry: BvpLambdifyTelemetry,
) -> Box<dyn Fun> {
    let binding = BvpParameterBindingHandle::new(parameter_values);
    dense_residual_with_binding(
        functions,
        argument_names,
        binding,
        parameter_count,
        telemetry,
    )
}

/// Dense AtomView residual callback using a reusable numeric parameter bind.
pub(crate) fn dense_residual_with_binding(
    functions: Vec<Atom>,
    argument_names: Vec<String>,
    binding: BvpParameterBindingHandle,
    parameter_count: usize,
    telemetry: BvpLambdifyTelemetry,
) -> Box<dyn Fun> {
    dense_residual_with_binding_and_policy(
        functions,
        argument_names,
        binding,
        parameter_count,
        telemetry,
        BvpLambdifyExecutionPolicy::default(),
    )
}

/// Dense AtomView residual callback with an explicit runtime execution policy.
pub(crate) fn dense_residual_with_binding_and_policy(
    functions: Vec<Atom>,
    argument_names: Vec<String>,
    binding: BvpParameterBindingHandle,
    parameter_count: usize,
    telemetry: BvpLambdifyTelemetry,
    execution_policy: BvpLambdifyExecutionPolicy,
) -> Box<dyn Fun> {
    validate_residual_inputs(
        functions.as_slice(),
        binding.snapshot().as_deref(),
        parameter_count,
    )
    .unwrap_or_else(|error| panic!("invalid AtomView dense residual inputs: {error:?}"));
    let context = variable_context(argument_names.as_slice());
    let function_map = FunctionMap::new();
    let compiled_functions: Vec<Box<dyn Fn(&[f64]) -> f64 + Send + Sync>> = functions
        .into_par_iter()
        .map(|function| lambdify_with_context(&function, context.as_ref(), &function_map))
        .collect();
    let callback = Box::new(move |_x: f64, values: &DVector<f64>| -> DVector<f64> {
        let started = telemetry.start_timing();
        let parameter_values = binding.snapshot();
        assert!(
            parameter_count == 0 || parameter_values.is_some(),
            "parameter values must be provided when parameters are configured"
        );
        let args = callback_arguments(
            parameter_values.as_deref(),
            values.as_slice(),
            parameter_count,
        );
        telemetry.record_dispatch(execution_policy, compiled_functions.len());
        let result: Vec<f64> = if execution_policy.should_parallel(compiled_functions.len()) {
            compiled_functions
                .par_iter()
                .map(|function| function(args.as_ref()))
                .collect()
        } else {
            compiled_functions
                .iter()
                .map(|function| function(args.as_ref()))
                .collect()
        };
        telemetry.record_residual_sample(started);
        DVector::from_vec(result)
    });
    Box::new(FunEnum::Dense(callback))
}

/// Installs a faer residual callback directly from packed Atom residuals.
pub(crate) fn sparse_residual(
    functions: Vec<Atom>,
    argument_names: Vec<String>,
    parameter_values: Option<Vec<f64>>,
    parameter_count: usize,
    telemetry: BvpLambdifyTelemetry,
) -> Box<dyn Fun> {
    let binding = BvpParameterBindingHandle::new(parameter_values);
    sparse_residual_with_binding(
        functions,
        argument_names,
        binding,
        parameter_count,
        telemetry,
    )
}

/// Sparse AtomView residual callback using a reusable numeric parameter bind.
pub(crate) fn sparse_residual_with_binding(
    functions: Vec<Atom>,
    argument_names: Vec<String>,
    binding: BvpParameterBindingHandle,
    parameter_count: usize,
    telemetry: BvpLambdifyTelemetry,
) -> Box<dyn Fun> {
    sparse_residual_with_binding_and_policy(
        functions,
        argument_names,
        binding,
        parameter_count,
        telemetry,
        BvpLambdifyExecutionPolicy::default(),
    )
}

/// Sparse AtomView residual callback with an explicit runtime execution policy.
pub(crate) fn sparse_residual_with_binding_and_policy(
    functions: Vec<Atom>,
    argument_names: Vec<String>,
    binding: BvpParameterBindingHandle,
    parameter_count: usize,
    telemetry: BvpLambdifyTelemetry,
    execution_policy: BvpLambdifyExecutionPolicy,
) -> Box<dyn Fun> {
    validate_residual_inputs(
        functions.as_slice(),
        binding.snapshot().as_deref(),
        parameter_count,
    )
    .unwrap_or_else(|error| panic!("invalid AtomView sparse residual inputs: {error:?}"));
    let context = variable_context(argument_names.as_slice());
    let function_map = FunctionMap::new();
    let compiled_functions: Vec<Box<dyn Fn(&[f64]) -> f64 + Send + Sync>> = functions
        .into_par_iter()
        .map(|function| lambdify_with_context(&function, context.as_ref(), &function_map))
        .collect();
    let callback = Box::new(move |_x: f64, values: &Col<f64>| -> Col<f64> {
        let started = telemetry.start_timing();
        let parameter_values = binding.snapshot();
        assert!(
            parameter_count == 0 || parameter_values.is_some(),
            "parameter values must be provided when parameters are configured"
        );
        let args = callback_arguments(
            parameter_values.as_deref(),
            values
                .try_as_col_major()
                .expect("faer callback state must be column-major")
                .as_slice(),
            parameter_count,
        );
        telemetry.record_dispatch(execution_policy, compiled_functions.len());
        let result: Vec<f64> = if execution_policy.should_parallel(compiled_functions.len()) {
            compiled_functions
                .par_iter()
                .map(|function| function(args.as_ref()))
                .collect()
        } else {
            compiled_functions
                .iter()
                .map(|function| function(args.as_ref()))
                .collect()
        };
        telemetry.record_residual_sample(started);
        ColRef::from_slice(result.as_slice()).to_owned()
    });
    Box::new(FunEnum::Sparse_3(callback))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::BVP_Damp::BVP_traits::{Fun, Jac, MatrixType, VectorType};
    use crate::symbolic::View::jacobian::SparseAtomJacobianEntry;
    use crate::symbolic::View::parser;
    use nalgebra::DVector;

    fn entry(row: usize, col: usize) -> SparseAtomJacobianEntry {
        SparseAtomJacobianEntry {
            row,
            col,
            value: parser::parse("x").expect("test Atom expression must parse"),
        }
    }

    #[test]
    fn atom_lambdify_preflight_rejects_missing_parameter_binding() {
        let error = validate_jacobian_inputs(&[entry(0, 0)], 1, 1, None, 1)
            .expect_err("missing parameter values must be typed error");
        assert!(matches!(
            error,
            BvpBackendIntegrationError::InvalidSolverConfiguration { field, .. }
                if field == "parameter_values"
        ));
    }

    #[test]
    fn atom_lambdify_preflight_rejects_out_of_bounds_sparse_coordinate() {
        let error = validate_jacobian_inputs(&[entry(1, 0)], 1, 1, None, 0)
            .expect_err("invalid sparse coordinate must be typed error");
        assert!(matches!(
            error,
            BvpBackendIntegrationError::InvalidProblem { field, .. }
                if field == "atom_jacobian_sparse_coordinates"
        ));
    }

    #[test]
    fn atom_lambdify_preflight_accepts_bound_valid_sparse_system() {
        validate_jacobian_inputs(&[entry(0, 0)], 1, 1, Some(&[2.0]), 1)
            .expect("valid AtomView binding must pass preflight");
    }

    #[test]
    fn atomview_callback_reuses_compiled_functions_after_numeric_rebind() {
        let binding = BvpParameterBindingHandle::new(Some(vec![1.0]));
        let mut callback = dense_residual_with_binding(
            vec![parser::parse("p + x").expect("test Atom expression must parse")],
            vec!["p".to_string(), "x".to_string()],
            binding.clone(),
            1,
            BvpLambdifyTelemetry::new(),
        );
        let first = callback
            .try_call(0.0, &DVector::from_vec(vec![2.0]))
            .expect("initial parameter binding should evaluate")
            .to_DVectorType();
        binding.replace(Some(vec![4.0]));
        let second = callback
            .try_call(0.0, &DVector::from_vec(vec![2.0]))
            .expect("rebound parameter should evaluate")
            .to_DVectorType();
        assert_eq!(first[0], 3.0);
        assert_eq!(second[0], 6.0);
    }

    #[test]
    fn atomview_sparse_callbacks_rebind_numeric_parameters() {
        let binding = BvpParameterBindingHandle::new(Some(vec![2.0]));
        let arguments = vec!["p".to_string(), "x".to_string()];
        let entries = vec![SparseAtomJacobianEntry {
            row: 0,
            col: 0,
            value: parser::parse("p").expect("test Atom Jacobian must parse"),
        }];
        let mut jacobian = sparse_jacobian_with_binding(
            entries,
            1,
            1,
            arguments.clone(),
            binding.clone(),
            1,
            BvpLambdifyTelemetry::new(),
        );
        let residual = sparse_residual_with_binding(
            vec![parser::parse("p*x").expect("test Atom residual must parse")],
            arguments,
            binding.clone(),
            1,
            BvpLambdifyTelemetry::new(),
        );
        let x = Col::from_fn(1, |_| 3.0);

        let first_jacobian = jacobian
            .try_call(0.0, &x)
            .expect("sparse Atom Jacobian should evaluate")
            .to_DMatrixType();
        let first_residual = residual
            .try_call(0.0, &x)
            .expect("sparse Atom residual should evaluate")
            .to_DVectorType();
        binding.replace(Some(vec![5.0]));
        let second_jacobian = jacobian
            .try_call(0.0, &x)
            .expect("rebound sparse Atom Jacobian should evaluate")
            .to_DMatrixType();
        let second_residual = residual
            .try_call(0.0, &x)
            .expect("rebound sparse Atom residual should evaluate")
            .to_DVectorType();

        assert_eq!(first_jacobian[(0, 0)], 2.0);
        assert_eq!(first_residual[0], 6.0);
        assert_eq!(second_jacobian[(0, 0)], 5.0);
        assert_eq!(second_residual[0], 15.0);
    }

    #[test]
    fn atomview_sequential_and_parallel_callbacks_are_identical() {
        let arguments = vec!["x".to_string(), "y".to_string()];
        let binding = BvpParameterBindingHandle::new(None);
        let residuals = vec![
            parser::parse("x + y").expect("test residual must parse"),
            parser::parse("x*y").expect("test residual must parse"),
            parser::parse("x^2-y").expect("test residual must parse"),
        ];
        let entries = vec![
            SparseAtomJacobianEntry {
                row: 0,
                col: 0,
                value: parser::parse("1").expect("test Jacobian must parse"),
            },
            SparseAtomJacobianEntry {
                row: 0,
                col: 1,
                value: parser::parse("1").expect("test Jacobian must parse"),
            },
            SparseAtomJacobianEntry {
                row: 1,
                col: 0,
                value: parser::parse("y").expect("test Jacobian must parse"),
            },
            SparseAtomJacobianEntry {
                row: 1,
                col: 1,
                value: parser::parse("x").expect("test Jacobian must parse"),
            },
            SparseAtomJacobianEntry {
                row: 2,
                col: 0,
                value: parser::parse("2*x").expect("test Jacobian must parse"),
            },
            SparseAtomJacobianEntry {
                row: 2,
                col: 1,
                value: parser::parse("-1").expect("test Jacobian must parse"),
            },
        ];
        let sequential = BvpLambdifyExecutionPolicy::Sequential;
        let parallel = BvpLambdifyExecutionPolicy::Parallel { min_work: 0 };
        let mut seq_residual = dense_residual_with_binding_and_policy(
            residuals.clone(),
            arguments.clone(),
            binding.clone(),
            0,
            BvpLambdifyTelemetry::disabled(),
            sequential,
        );
        let mut par_residual = dense_residual_with_binding_and_policy(
            residuals,
            arguments.clone(),
            binding.clone(),
            0,
            BvpLambdifyTelemetry::disabled(),
            parallel,
        );
        let mut seq_jacobian = dense_jacobian_with_binding_and_policy(
            entries.clone(),
            3,
            2,
            arguments.clone(),
            binding.clone(),
            0,
            BvpLambdifyTelemetry::disabled(),
            sequential,
        );
        let mut par_jacobian = dense_jacobian_with_binding_and_policy(
            entries,
            3,
            2,
            arguments,
            binding,
            0,
            BvpLambdifyTelemetry::disabled(),
            parallel,
        );
        let values = DVector::from_vec(vec![2.0, 3.0]);

        let seq_r = seq_residual
            .try_call(0.0, &values)
            .expect("sequential residual should evaluate")
            .to_DVectorType();
        let par_r = par_residual
            .try_call(0.0, &values)
            .expect("parallel residual should evaluate")
            .to_DVectorType();
        let seq_j = seq_jacobian
            .try_call(0.0, &values)
            .expect("sequential Jacobian should evaluate")
            .to_DMatrixType();
        let par_j = par_jacobian
            .try_call(0.0, &values)
            .expect("parallel Jacobian should evaluate")
            .to_DMatrixType();

        assert_eq!(seq_r, par_r);
        assert_eq!(seq_j, par_j);
        assert_eq!(seq_r, DVector::from_vec(vec![5.0, 6.0, 1.0]));
        assert_eq!(
            seq_j,
            DMatrix::from_row_slice(3, 2, &[1.0, 1.0, 3.0, 2.0, 4.0, -1.0])
        );
    }

    #[test]
    fn atomview_sparse_sequential_and_parallel_callbacks_are_identical() {
        let arguments = vec!["x".to_string(), "y".to_string()];
        let entries = vec![
            SparseAtomJacobianEntry {
                row: 0,
                col: 0,
                value: parser::parse("x").expect("test Jacobian must parse"),
            },
            SparseAtomJacobianEntry {
                row: 0,
                col: 1,
                value: parser::parse("y").expect("test Jacobian must parse"),
            },
            SparseAtomJacobianEntry {
                row: 1,
                col: 0,
                value: parser::parse("x+y").expect("test Jacobian must parse"),
            },
        ];
        let mut sequential = sparse_jacobian_with_binding_and_policy(
            entries.clone(),
            2,
            2,
            arguments.clone(),
            BvpParameterBindingHandle::new(None),
            0,
            BvpLambdifyTelemetry::disabled(),
            BvpLambdifyExecutionPolicy::Sequential,
        );
        let mut parallel = sparse_jacobian_with_binding_and_policy(
            entries,
            2,
            2,
            arguments,
            BvpParameterBindingHandle::new(None),
            0,
            BvpLambdifyTelemetry::disabled(),
            BvpLambdifyExecutionPolicy::Parallel { min_work: 0 },
        );
        let values = Col::from_fn(2, |index| [2.0, 3.0][index]);
        let seq = sequential
            .try_call(0.0, &values)
            .expect("sequential sparse Jacobian should evaluate")
            .to_DMatrixType();
        let par = parallel
            .try_call(0.0, &values)
            .expect("parallel sparse Jacobian should evaluate")
            .to_DMatrixType();
        assert_eq!(seq, par);
        assert_eq!(seq, DMatrix::from_row_slice(2, 2, &[2.0, 3.0, 5.0, 0.0]));
    }

    #[test]
    fn atomview_sparse_fixed_csc_pattern_survives_numeric_zero_crossing() {
        let entries = vec![
            SparseAtomJacobianEntry {
                row: 0,
                col: 0,
                value: parser::parse("x").expect("test Jacobian must parse"),
            },
            SparseAtomJacobianEntry {
                row: 1,
                col: 0,
                value: parser::parse("y").expect("test Jacobian must parse"),
            },
            SparseAtomJacobianEntry {
                row: 1,
                col: 1,
                value: parser::parse("x+y").expect("test Jacobian must parse"),
            },
        ];
        let mut callback = sparse_jacobian_with_binding_and_policy(
            entries,
            2,
            2,
            vec!["x".to_string(), "y".to_string()],
            BvpParameterBindingHandle::new(None),
            0,
            BvpLambdifyTelemetry::disabled(),
            BvpLambdifyExecutionPolicy::Sequential,
        );

        let zero = Col::from_fn(2, |_| 0.0);
        let zero_matrix = callback
            .try_call(0.0, &zero)
            .expect("zero-valued fixed pattern should evaluate");
        let zero_matrix = zero_matrix
            .as_any()
            .downcast_ref::<SparseColMat<usize, f64>>()
            .expect("sparse callback must return faer CSC storage");
        assert_eq!(zero_matrix.symbolic().col_ptr(), &[0, 2, 3]);
        assert_eq!(zero_matrix.symbolic().row_idx(), &[0, 1, 1]);
        assert_eq!(zero_matrix.val(), &[0.0, 0.0, 0.0]);

        let nonzero = Col::from_fn(2, |index| [2.0, 3.0][index]);
        let nonzero_matrix = callback
            .try_call(0.0, &nonzero)
            .expect("nonzero fixed pattern should evaluate");
        let nonzero_matrix = nonzero_matrix
            .as_any()
            .downcast_ref::<SparseColMat<usize, f64>>()
            .expect("sparse callback must return faer CSC storage");
        assert_eq!(nonzero_matrix.symbolic().col_ptr(), &[0, 2, 3]);
        assert_eq!(nonzero_matrix.symbolic().row_idx(), &[0, 1, 1]);
        assert_eq!(nonzero_matrix.val(), &[2.0, 3.0, 5.0]);
    }
}
