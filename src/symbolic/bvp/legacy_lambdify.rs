#![allow(dead_code)]
//! Legacy BVP Lambdify runtime support.
//!
//! This module deliberately contains only compatibility instrumentation and
//! small shared helpers for the historical Expr-based callbacks.  It is kept
//! separate from the direct no-Mutex runtime so benchmark results can identify
//! which implementation produced them.

use super::parameter_binding::BvpParameterBindingHandle;
use super::telemetry::{BvpLambdifyExecutionPolicy, flatten_lambdify_args};
use crate::global::THRESHOLD as T;
use crate::numerical::BVP_Damp::BVP_traits::{Fun, FunEnum, Jac, JacEnum};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_lambdify::ExprLambdifyContext;
use faer::col::{Col, ColRef};
use faer::sparse::{SparseColMat, Triplet};
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

/// Compatibility names retained for callers of the old legacy module.
pub use super::telemetry::{
    BvpLambdifyTelemetry as LegacyLambdifyTelemetry,
    BvpLambdifyTelemetrySnapshot as LegacyLambdifyTelemetrySnapshot,
};

/// Installs the historical Expr-based dense Jacobian callback.
///
/// The public `Jacobian::lambdify_jacobian_DMatrix_par` method delegates here
/// so the legacy callback implementation and its telemetry are physically
/// separate from symbolic preparation and from the direct AtomView runtime.
pub(crate) fn dense_jacobian(
    jac: Vec<Vec<Expr>>,
    rows: usize,
    cols: usize,
    bandwidth: Option<(usize, usize)>,
    argument_names: Vec<String>,
    parameter_values: Option<Vec<f64>>,
    parameter_count: usize,
    telemetry: LegacyLambdifyTelemetry,
) -> Box<dyn Jac> {
    let binding = BvpParameterBindingHandle::new(parameter_values);
    dense_jacobian_with_binding(
        jac,
        rows,
        cols,
        bandwidth,
        argument_names,
        binding,
        parameter_count,
        telemetry,
    )
}

/// Dense ExprLegacy Jacobian callback using a reusable numeric parameter bind.
pub(crate) fn dense_jacobian_with_binding(
    jac: Vec<Vec<Expr>>,
    rows: usize,
    cols: usize,
    bandwidth: Option<(usize, usize)>,
    argument_names: Vec<String>,
    binding: BvpParameterBindingHandle,
    parameter_count: usize,
    telemetry: LegacyLambdifyTelemetry,
) -> Box<dyn Jac> {
    dense_jacobian_with_binding_and_policy(
        jac,
        rows,
        cols,
        bandwidth,
        argument_names,
        binding,
        parameter_count,
        telemetry,
        BvpLambdifyExecutionPolicy::default(),
    )
}

/// Dense ExprLegacy Jacobian callback with an explicit runtime policy.
pub(crate) fn dense_jacobian_with_binding_and_policy(
    jac: Vec<Vec<Expr>>,
    rows: usize,
    cols: usize,
    bandwidth: Option<(usize, usize)>,
    argument_names: Vec<String>,
    binding: BvpParameterBindingHandle,
    parameter_count: usize,
    telemetry: LegacyLambdifyTelemetry,
    execution_policy: BvpLambdifyExecutionPolicy,
) -> Box<dyn Jac> {
    let jacobian_positions: Vec<(usize, usize, Box<dyn Fn(&[f64]) -> f64 + Send + Sync>)> = (0
        ..rows)
        .into_par_iter()
        .flat_map(|i| {
            let (right_border, left_border) = if let Some((kl, ku)) = bandwidth {
                let right_border = std::cmp::min(i + ku + 1, cols);
                let left_border = i.saturating_sub(kl + 1);
                (right_border, left_border)
            } else {
                (cols, 0)
            };
            (left_border..right_border)
                .filter_map(|j| {
                    let derivative = &jac[i][j];
                    (!derivative.is_zero()).then(|| {
                        let names = argument_names
                            .iter()
                            .map(String::as_str)
                            .collect::<Vec<_>>();
                        let compiled =
                            Expr::lambdify_borrowed_thread_safe(derivative, names.as_slice());
                        (i, j, compiled)
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let callback = Box::new(move |_x: f64, values: &DVector<f64>| -> DMatrix<f64> {
        let started = telemetry.start_timing();
        let parameter_values = binding.snapshot();
        let args = flatten_lambdify_args(parameter_values.as_deref(), values.as_slice());
        assert!(
            parameter_count == 0 || parameter_values.is_some(),
            "parameter values must be provided when parameters are configured"
        );
        let mut matrix = DMatrix::zeros(rows, cols);
        telemetry.record_dispatch(execution_policy, jacobian_positions.len());
        if execution_policy.should_parallel(jacobian_positions.len()) {
            let values: Vec<(usize, usize, f64)> = jacobian_positions
                .par_iter()
                .map(|(row, col, compiled)| (*row, *col, compiled(args.as_slice())))
                .collect();
            for (row, col, value) in values {
                if value.abs() > T {
                    matrix[(row, col)] = value;
                }
            }
        } else {
            for (row, col, compiled) in &jacobian_positions {
                let value = compiled(args.as_slice());
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

/// Installs the historical Expr-based dense residual callback.
pub(crate) fn dense_residual(
    functions: Vec<Expr>,
    argument_names: Vec<String>,
    parameter_values: Option<Vec<f64>>,
    parameter_count: usize,
    telemetry: LegacyLambdifyTelemetry,
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

/// Dense ExprLegacy residual callback using a reusable numeric parameter bind.
pub(crate) fn dense_residual_with_binding(
    functions: Vec<Expr>,
    argument_names: Vec<String>,
    binding: BvpParameterBindingHandle,
    parameter_count: usize,
    telemetry: LegacyLambdifyTelemetry,
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

/// Dense ExprLegacy residual callback with an explicit runtime policy.
pub(crate) fn dense_residual_with_binding_and_policy(
    functions: Vec<Expr>,
    argument_names: Vec<String>,
    binding: BvpParameterBindingHandle,
    parameter_count: usize,
    telemetry: LegacyLambdifyTelemetry,
    execution_policy: BvpLambdifyExecutionPolicy,
) -> Box<dyn Fun> {
    let compiled_functions: Vec<Box<dyn Fn(&[f64]) -> f64 + Send + Sync>> = functions
        .par_iter()
        .map(|function| {
            let names = argument_names
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>();
            Expr::lambdify_borrowed_thread_safe(function, names.as_slice())
        })
        .collect();
    let callback = Box::new(move |_x: f64, values: &DVector<f64>| -> DVector<f64> {
        let started = telemetry.start_timing();
        let parameter_values = binding.snapshot();
        let args = flatten_lambdify_args(parameter_values.as_deref(), values.as_slice());
        assert!(
            parameter_count == 0 || parameter_values.is_some(),
            "parameter values must be provided when parameters are configured"
        );
        telemetry.record_dispatch(execution_policy, compiled_functions.len());
        let result: Vec<f64> = if execution_policy.should_parallel(compiled_functions.len()) {
            compiled_functions
                .par_iter()
                .map(|function| function(args.as_slice()))
                .collect()
        } else {
            compiled_functions
                .iter()
                .map(|function| function(args.as_slice()))
                .collect()
        };
        telemetry.record_residual_sample(started);
        DVector::from_vec(result)
    });
    Box::new(FunEnum::Dense(callback))
}

/// Installs the historical parallel faer sparse Jacobian callback.
pub(crate) fn faer_sparse_jacobian(
    entries: Vec<(usize, usize, Expr)>,
    rows: usize,
    cols: usize,
    argument_names: Vec<String>,
    parameter_values: Option<Vec<f64>>,
    parameter_count: usize,
    telemetry: LegacyLambdifyTelemetry,
) -> Box<dyn Jac> {
    let binding = BvpParameterBindingHandle::new(parameter_values);
    faer_sparse_jacobian_with_binding(
        entries,
        rows,
        cols,
        argument_names,
        binding,
        parameter_count,
        telemetry,
    )
}

/// Sparse ExprLegacy Jacobian callback using a reusable numeric parameter bind.
pub(crate) fn faer_sparse_jacobian_with_binding(
    entries: Vec<(usize, usize, Expr)>,
    rows: usize,
    cols: usize,
    argument_names: Vec<String>,
    binding: BvpParameterBindingHandle,
    parameter_count: usize,
    telemetry: LegacyLambdifyTelemetry,
) -> Box<dyn Jac> {
    faer_sparse_jacobian_with_binding_and_policy(
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

/// Sparse ExprLegacy Jacobian callback with an explicit runtime policy.
pub(crate) fn faer_sparse_jacobian_with_binding_and_policy(
    entries: Vec<(usize, usize, Expr)>,
    rows: usize,
    cols: usize,
    argument_names: Vec<String>,
    binding: BvpParameterBindingHandle,
    parameter_count: usize,
    telemetry: LegacyLambdifyTelemetry,
    execution_policy: BvpLambdifyExecutionPolicy,
) -> Box<dyn Jac> {
    faer_sparse_jacobian_with_lookup_mode(
        entries,
        rows,
        cols,
        argument_names,
        binding,
        parameter_count,
        telemetry,
        execution_policy,
        false,
    )
}

/// Experimental sparse ExprLegacy Jacobian callback with a shared argument
/// index map for compiling the complete derivative set.
pub(crate) fn faer_sparse_jacobian_with_indexed_lookup(
    entries: Vec<(usize, usize, Expr)>,
    rows: usize,
    cols: usize,
    argument_names: Vec<String>,
    binding: BvpParameterBindingHandle,
    parameter_count: usize,
    telemetry: LegacyLambdifyTelemetry,
    execution_policy: BvpLambdifyExecutionPolicy,
) -> Box<dyn Jac> {
    faer_sparse_jacobian_with_lookup_mode(
        entries,
        rows,
        cols,
        argument_names,
        binding,
        parameter_count,
        telemetry,
        execution_policy,
        true,
    )
}

fn faer_sparse_jacobian_with_lookup_mode(
    entries: Vec<(usize, usize, Expr)>,
    rows: usize,
    cols: usize,
    argument_names: Vec<String>,
    binding: BvpParameterBindingHandle,
    parameter_count: usize,
    telemetry: LegacyLambdifyTelemetry,
    execution_policy: BvpLambdifyExecutionPolicy,
    indexed_lookup: bool,
) -> Box<dyn Jac> {
    let indexed_context = indexed_lookup.then(|| {
        let names = argument_names
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        ExprLambdifyContext::new(names.as_slice())
    });
    let positions: Vec<(usize, usize, Box<dyn Fn(&[f64]) -> f64 + Send + Sync>)> = entries
        .into_par_iter()
        .map(|(row, col, derivative)| {
            let compiled = if let Some(context) = &indexed_context {
                context.compile(&derivative)
            } else {
                let names = argument_names
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>();
                Expr::lambdify_borrowed_thread_safe(&derivative, names.as_slice())
            };
            (row, col, compiled)
        })
        .collect();
    let callback = Box::new(
        move |_x: f64, values: &Col<f64>| -> SparseColMat<usize, f64> {
            let started = telemetry.start_timing();
            let dense_values: Vec<f64> = values.iter().copied().collect();
            let parameter_values = binding.snapshot();
            assert!(
                parameter_count == 0 || parameter_values.is_some(),
                "parameter values must be provided when parameters are configured"
            );
            let args = flatten_lambdify_args(parameter_values.as_deref(), dense_values.as_slice());
            telemetry.record_dispatch(execution_policy, positions.len());
            let triplets: Vec<Triplet<usize, usize, f64>> =
                if execution_policy.should_parallel(positions.len()) {
                    positions
                        .par_iter()
                        .filter_map(|(row, col, function)| {
                            let value = function(args.as_slice());
                            (value.abs() > T).then(|| Triplet::new(*row, *col, value))
                        })
                        .collect()
                } else {
                    positions
                        .iter()
                        .filter_map(|(row, col, function)| {
                            let value = function(args.as_slice());
                            (value.abs() > T).then(|| Triplet::new(*row, *col, value))
                        })
                        .collect()
                };
            let matrix = SparseColMat::try_new_from_triplets(rows, cols, triplets.as_slice())
                .expect("legacy sparse Jacobian triplets must have valid coordinates");
            telemetry.record_jacobian_sample(started);
            matrix
        },
    );
    Box::new(JacEnum::Sparse_3(callback))
}

/// Installs the historical parallel faer sparse residual callback.
pub(crate) fn faer_sparse_residual(
    functions: Vec<Expr>,
    argument_names: Vec<String>,
    parameter_values: Option<Vec<f64>>,
    parameter_count: usize,
    telemetry: LegacyLambdifyTelemetry,
) -> Box<dyn Fun> {
    let binding = BvpParameterBindingHandle::new(parameter_values);
    faer_sparse_residual_with_binding(
        functions,
        argument_names,
        binding,
        parameter_count,
        telemetry,
    )
}

/// Sparse ExprLegacy residual callback using a reusable numeric parameter bind.
pub(crate) fn faer_sparse_residual_with_binding(
    functions: Vec<Expr>,
    argument_names: Vec<String>,
    binding: BvpParameterBindingHandle,
    parameter_count: usize,
    telemetry: LegacyLambdifyTelemetry,
) -> Box<dyn Fun> {
    faer_sparse_residual_with_binding_and_policy(
        functions,
        argument_names,
        binding,
        parameter_count,
        telemetry,
        BvpLambdifyExecutionPolicy::default(),
    )
}

/// Sparse ExprLegacy residual callback with an explicit runtime policy.
pub(crate) fn faer_sparse_residual_with_binding_and_policy(
    functions: Vec<Expr>,
    argument_names: Vec<String>,
    binding: BvpParameterBindingHandle,
    parameter_count: usize,
    telemetry: LegacyLambdifyTelemetry,
    execution_policy: BvpLambdifyExecutionPolicy,
) -> Box<dyn Fun> {
    faer_sparse_residual_with_lookup_mode(
        functions,
        argument_names,
        binding,
        parameter_count,
        telemetry,
        execution_policy,
        false,
    )
}

/// Experimental sparse ExprLegacy residual callback with a shared argument
/// index map for compiling the residual evaluator set.
pub(crate) fn faer_sparse_residual_with_indexed_lookup(
    functions: Vec<Expr>,
    argument_names: Vec<String>,
    binding: BvpParameterBindingHandle,
    parameter_count: usize,
    telemetry: LegacyLambdifyTelemetry,
    execution_policy: BvpLambdifyExecutionPolicy,
) -> Box<dyn Fun> {
    faer_sparse_residual_with_lookup_mode(
        functions,
        argument_names,
        binding,
        parameter_count,
        telemetry,
        execution_policy,
        true,
    )
}

fn faer_sparse_residual_with_lookup_mode(
    functions: Vec<Expr>,
    argument_names: Vec<String>,
    binding: BvpParameterBindingHandle,
    parameter_count: usize,
    telemetry: LegacyLambdifyTelemetry,
    execution_policy: BvpLambdifyExecutionPolicy,
    indexed_lookup: bool,
) -> Box<dyn Fun> {
    let indexed_context = indexed_lookup.then(|| {
        let names = argument_names
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        ExprLambdifyContext::new(names.as_slice())
    });
    let compiled_functions: Vec<Box<dyn Fn(&[f64]) -> f64 + Send + Sync>> = functions
        .par_iter()
        .map(|function| {
            if let Some(context) = &indexed_context {
                context.compile(function)
            } else {
                let names = argument_names
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>();
                Expr::lambdify_borrowed_thread_safe(function, names.as_slice())
            }
        })
        .collect();
    let callback = Box::new(move |_x: f64, values: &Col<f64>| -> Col<f64> {
        let started = telemetry.start_timing();
        let dense_values: Vec<f64> = values.iter().copied().collect();
        let parameter_values = binding.snapshot();
        assert!(
            parameter_count == 0 || parameter_values.is_some(),
            "parameter values must be provided when parameters are configured"
        );
        let args = flatten_lambdify_args(parameter_values.as_deref(), dense_values.as_slice());
        telemetry.record_dispatch(execution_policy, compiled_functions.len());
        let result: Vec<f64> = if execution_policy.should_parallel(compiled_functions.len()) {
            compiled_functions
                .par_iter()
                .map(|function| function(args.as_slice()))
                .collect()
        } else {
            compiled_functions
                .iter()
                .map(|function| function(args.as_slice()))
                .collect()
        };
        telemetry.record_residual_sample(started);
        ColRef::from_slice(result.as_slice()).to_owned()
    });
    Box::new(FunEnum::Sparse_3(callback))
}

#[cfg(test)]
mod tests {
    use super::{
        LegacyLambdifyTelemetry, dense_residual_with_binding, faer_sparse_jacobian_with_binding,
        faer_sparse_jacobian_with_indexed_lookup, faer_sparse_residual_with_binding,
        faer_sparse_residual_with_indexed_lookup,
    };

    use crate::symbolic::bvp::legacy::Jacobian;
    use crate::symbolic::bvp::parameter_binding::BvpParameterBindingHandle;
    use crate::symbolic::symbolic_engine::Expr;
    use faer::col::Col;
    use nalgebra::DVector;
    use std::time::Duration;

    #[test]
    fn legacy_lambdify_telemetry_accumulates_typed_callback_costs() {
        let telemetry = LegacyLambdifyTelemetry::new();
        telemetry.record_residual(Duration::from_nanos(3));
        telemetry.record_residual(Duration::from_nanos(5));
        telemetry.record_jacobian(Duration::from_nanos(7));

        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.residual_calls, 2);
        assert_eq!(snapshot.residual_elapsed, Duration::from_nanos(8));
        assert_eq!(snapshot.jacobian_calls, 1);
        assert_eq!(snapshot.jacobian_elapsed, Duration::from_nanos(7));
    }

    #[test]
    fn exprlegacy_callback_reuses_compiled_functions_after_numeric_rebind() {
        let binding = BvpParameterBindingHandle::new(Some(vec![1.0]));
        let callback = dense_residual_with_binding(
            vec![Expr::parse_expression("p + x")],
            vec!["p".to_string(), "x".to_string()],
            binding.clone(),
            1,
            LegacyLambdifyTelemetry::new(),
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
    fn exprlegacy_sparse_callbacks_rebind_numeric_parameters() {
        let binding = BvpParameterBindingHandle::new(Some(vec![2.0]));
        let arguments = vec!["p".to_string(), "x".to_string()];
        let mut jacobian = faer_sparse_jacobian_with_binding(
            vec![(0, 0, Expr::parse_expression("p"))],
            1,
            1,
            arguments.clone(),
            binding.clone(),
            1,
            LegacyLambdifyTelemetry::new(),
        );
        let residual = faer_sparse_residual_with_binding(
            vec![Expr::parse_expression("p*x")],
            arguments,
            binding.clone(),
            1,
            LegacyLambdifyTelemetry::new(),
        );
        let x = Col::from_fn(1, |_| 3.0);

        let first_jacobian = jacobian
            .try_call(0.0, &x)
            .expect("sparse Jacobian should evaluate")
            .to_DMatrixType();
        let first_residual = residual
            .try_call(0.0, &x)
            .expect("sparse residual should evaluate")
            .to_DVectorType();
        binding.replace(Some(vec![5.0]));
        let second_jacobian = jacobian
            .try_call(0.0, &x)
            .expect("rebound sparse Jacobian should evaluate")
            .to_DMatrixType();
        let second_residual = residual
            .try_call(0.0, &x)
            .expect("rebound sparse residual should evaluate")
            .to_DVectorType();

        assert_eq!(first_jacobian[(0, 0)], 2.0);
        assert_eq!(first_residual[0], 6.0);
        assert_eq!(second_jacobian[(0, 0)], 5.0);
        assert_eq!(second_residual[0], 15.0);
    }

    #[test]
    fn indexed_sparse_expr_callbacks_match_legacy_values_and_parameter_rebinds() {
        let binding = BvpParameterBindingHandle::new(Some(vec![1.25]));
        let arguments = vec!["p".to_string(), "x".to_string(), "y".to_string()];
        let expressions = vec![
            Expr::parse_expression("p*x + y^2"),
            Expr::parse_expression("sin(p) + x*y"),
        ];
        let entries = vec![
            (0, 0, Expr::parse_expression("p")),
            (0, 1, Expr::parse_expression("2*y")),
            (1, 0, Expr::parse_expression("y")),
            (1, 1, Expr::parse_expression("x")),
        ];
        let mut legacy_jacobian = faer_sparse_jacobian_with_binding(
            entries.clone(),
            2,
            2,
            arguments.clone(),
            binding.clone(),
            1,
            LegacyLambdifyTelemetry::new(),
        );
        let mut indexed_jacobian = faer_sparse_jacobian_with_indexed_lookup(
            entries,
            2,
            2,
            arguments.clone(),
            binding.clone(),
            1,
            LegacyLambdifyTelemetry::new(),
            super::super::telemetry::BvpLambdifyExecutionPolicy::default(),
        );
        let legacy_residual = faer_sparse_residual_with_binding(
            expressions.clone(),
            arguments.clone(),
            binding.clone(),
            1,
            LegacyLambdifyTelemetry::new(),
        );
        let indexed_residual = faer_sparse_residual_with_indexed_lookup(
            expressions,
            arguments,
            binding.clone(),
            1,
            LegacyLambdifyTelemetry::new(),
            super::super::telemetry::BvpLambdifyExecutionPolicy::default(),
        );
        let x = Col::from_fn(2, |index| [0.75, -0.5][index]);

        for parameter in [1.25, -0.3] {
            binding.replace(Some(vec![parameter]));
            let legacy_jacobian_value = legacy_jacobian
                .try_call(0.0, &x)
                .expect("legacy Jacobian should evaluate")
                .to_DMatrixType();
            let indexed_jacobian_value = indexed_jacobian
                .try_call(0.0, &x)
                .expect("indexed Jacobian should evaluate")
                .to_DMatrixType();
            let legacy_residual_value = legacy_residual
                .try_call(0.0, &x)
                .expect("legacy residual should evaluate")
                .to_DVectorType();
            let indexed_residual_value = indexed_residual
                .try_call(0.0, &x)
                .expect("indexed residual should evaluate")
                .to_DVectorType();

            assert_eq!(indexed_jacobian_value, legacy_jacobian_value);
            assert_eq!(indexed_residual_value, legacy_residual_value);
        }
    }

    #[test]
    fn jacobian_indexed_compiler_switch_matches_legacy_sparse_callbacks() {
        fn prepare(indexed: bool) -> Jacobian {
            let mut system = Jacobian::new();
            system.from_vectors(
                vec![
                    Expr::parse_expression("x + y^2"),
                    Expr::parse_expression("sin(x) - y"),
                ],
                vec!["x".to_owned(), "y".to_owned()],
            );
            system.variables_for_all_disrete =
                crate::symbolic::bvp::legacy_symbolic::variables_for_functions(
                    &system.vector_of_functions,
                );
            system.set_expr_indexed_lambdify_enabled(indexed);
            system.calc_jacobian_parallel_smart_optimized();
            system.lambdify_jacobian_SparseColMat_parallel2("", vec!["x", "y"]);
            system.lambdify_residual_Col_parallel2("", vec!["x", "y"]);
            system
        }

        let mut legacy = prepare(false);
        let mut indexed = prepare(true);
        assert!(!legacy.expr_indexed_lambdify_enabled());
        assert!(indexed.expr_indexed_lambdify_enabled());
        let point = Col::from_fn(2, |index| [0.75, -0.5][index]);

        let legacy_residual = legacy
            .residiual_function
            .try_call(0.0, &point)
            .expect("legacy residual should be installed");
        let indexed_residual = indexed
            .residiual_function
            .try_call(0.0, &point)
            .expect("indexed residual should be installed");
        assert_eq!(
            legacy_residual.to_DVectorType(),
            indexed_residual.to_DVectorType()
        );

        let legacy_jacobian = legacy
            .jac_function
            .as_mut()
            .expect("legacy Jacobian should be installed")
            .try_call(0.0, &point)
            .expect("legacy Jacobian should be installed");
        let indexed_jacobian = indexed
            .jac_function
            .as_mut()
            .expect("indexed Jacobian should be installed")
            .try_call(0.0, &point)
            .expect("indexed Jacobian should be installed");
        assert_eq!(
            legacy_jacobian.to_DMatrixType(),
            indexed_jacobian.to_DMatrixType()
        );
    }
}
