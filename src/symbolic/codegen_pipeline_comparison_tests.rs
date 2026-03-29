#![cfg(test)]

//! Comparison tests between the existing symbolic/lambdify pipeline and the
//! emerging AOT-oriented planning pipeline.
//!
//! These tests intentionally focus on *semantic equivalence*:
//! - the same symbolic residuals should evaluate to the same numeric values,
//! - the same symbolic Jacobian should yield the same sparse non-zero values
//!   in the same explicit entry order.
//!
//! Full emitted-AOT execution and microbenchmarking will come later. For now
//! this module acts as a one-to-one correctness scaffold for the new planning
//! layers.

use crate::symbolic::codegen_runtime_api::{
    RuntimeArguments, extract_sparse_entries_from_dense_jacobian,
    recommended_row_chunking_for_parallelism,
};
use crate::symbolic::codegen_tasks::{
    ResidualTask, SparseChunkingStrategy, SparseExprEntry, SparseJacobianTask,
};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions_BVP::Jacobian;
use crate::symbolic::test_codegen_generated_bvp_fixtures::generated_bvp_fixture;
use std::collections::HashMap;

fn eval_exprs_with_lambdify(exprs: &[Expr], vars: &[&str], values: &[f64]) -> Vec<f64> {
    exprs
        .iter()
        .map(|expr| expr.lambdify_borrowed_thread_safe(vars)(values))
        .collect()
}

fn build_real_bvp_damp1_case(n_steps: usize) -> Jacobian {
    let eq1 = Expr::parse_expression("y-z");
    let eq2 = Expr::parse_expression("-z^3");
    let eq_system = vec![eq1, eq2];
    let values = vec!["z".to_string(), "y".to_string()];

    let mut border_conditions = HashMap::new();
    border_conditions.insert("z".to_string(), vec![(0usize, 1.0f64)]);
    border_conditions.insert("y".to_string(), vec![(1usize, 1.0f64)]);

    let mut jac = Jacobian::new();
    jac.discretization_system_BVP_par(
        eq_system,
        values,
        "x".to_string(),
        0.0,
        Some(n_steps),
        None,
        None,
        border_conditions,
        None,
        None,
        "forward".to_string(),
    );
    jac.calc_jacobian_parallel_smart_optimized();
    jac
}

fn borrowed_sparse_entries(owned_entries: &[(usize, usize, Expr)]) -> Vec<SparseExprEntry<'_>> {
    owned_entries
        .iter()
        .map(|(row, col, expr)| SparseExprEntry {
            row: *row,
            col: *col,
            expr,
        })
        .collect()
}

fn eval_generated_bvp_residual_fixture(args: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; 16];
    generated_bvp_fixture::fixture_bvp_residual_chunk_0(args, &mut out[0..8]);
    generated_bvp_fixture::fixture_bvp_residual_chunk_1(args, &mut out[8..16]);
    out
}

fn eval_generated_bvp_sparse_fixture(args: &[f64]) -> Vec<f64> {
    let mut outputs = Vec::new();
    let chunk_fns: [fn(&[f64], &mut [f64]); 16] = [
        generated_bvp_fixture::fixture_bvp_sparse_values_chunk_0,
        generated_bvp_fixture::fixture_bvp_sparse_values_chunk_1,
        generated_bvp_fixture::fixture_bvp_sparse_values_chunk_2,
        generated_bvp_fixture::fixture_bvp_sparse_values_chunk_3,
        generated_bvp_fixture::fixture_bvp_sparse_values_chunk_4,
        generated_bvp_fixture::fixture_bvp_sparse_values_chunk_5,
        generated_bvp_fixture::fixture_bvp_sparse_values_chunk_6,
        generated_bvp_fixture::fixture_bvp_sparse_values_chunk_7,
        generated_bvp_fixture::fixture_bvp_sparse_values_chunk_8,
        generated_bvp_fixture::fixture_bvp_sparse_values_chunk_9,
        generated_bvp_fixture::fixture_bvp_sparse_values_chunk_10,
        generated_bvp_fixture::fixture_bvp_sparse_values_chunk_11,
        generated_bvp_fixture::fixture_bvp_sparse_values_chunk_12,
        generated_bvp_fixture::fixture_bvp_sparse_values_chunk_13,
        generated_bvp_fixture::fixture_bvp_sparse_values_chunk_14,
        generated_bvp_fixture::fixture_bvp_sparse_values_chunk_15,
    ];
    let chunk_sizes = [2usize, 2, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 2];

    for (chunk_fn, chunk_size) in chunk_fns.into_iter().zip(chunk_sizes) {
        let mut chunk_out = vec![0.0; chunk_size];
        chunk_fn(args, &mut chunk_out);
        outputs.extend(chunk_out);
    }

    outputs
}

#[test]
fn residual_task_runtime_plan_matches_existing_lambdify_values() {
    let residuals = vec![
        Expr::parse_expression("alpha*y0 + y1^2"),
        Expr::parse_expression("sin(y0) - beta*y1"),
    ];
    let params = ["alpha", "beta"];
    let vars = ["y0", "y1"];
    let task = ResidualTask {
        fn_name: "eval_residual",
        residuals: &residuals,
        variables: &vars,
        params: Some(&params),
    };

    let runtime_plan = task.runtime_plan(
        crate::symbolic::codegen_runtime_api::ResidualChunkingStrategy::ByOutputCount {
            max_outputs_per_chunk: 1,
        },
    );
    let grouped_args = RuntimeArguments::new(&[2.0, -3.0], Some(&[1.5, 0.25]));
    let flat_args = grouped_args.flatten();
    let expected = eval_exprs_with_lambdify(&residuals, &["alpha", "beta", "y0", "y1"], &flat_args);

    let actual: Vec<f64> = runtime_plan
        .chunks
        .iter()
        .flat_map(|chunk| {
            chunk
                .residuals
                .iter()
                .map(|expr| expr.lambdify_borrowed_thread_safe(&chunk.plan.input_names)(&flat_args))
                .collect::<Vec<_>>()
        })
        .collect();

    assert_eq!(actual.len(), expected.len());
    for (lhs, rhs) in actual.iter().zip(expected.iter()) {
        assert!(
            (lhs - rhs).abs() < 1e-12,
            "residual mismatch: {lhs} vs {rhs}"
        );
    }
}

#[test]
fn sparse_jacobian_runtime_plan_matches_existing_lambdify_values() {
    let jacobian = vec![
        vec![
            Expr::parse_expression("2*y0"),
            Expr::Const(0.0),
            Expr::parse_expression("alpha"),
        ],
        vec![
            Expr::Const(0.0),
            Expr::parse_expression("cos(y1)"),
            Expr::Const(0.0),
        ],
        vec![
            Expr::parse_expression("beta*y2"),
            Expr::Const(0.0),
            Expr::Const(0.0),
        ],
    ];
    let params = ["alpha", "beta"];
    let vars = ["y0", "y1", "y2"];
    let entries = extract_sparse_entries_from_dense_jacobian(&jacobian);
    let task = SparseJacobianTask {
        fn_name: "eval_sparse_values",
        shape: (3, 3),
        entries: &entries,
        variables: &vars,
        params: Some(&params),
    };

    let strategy = match recommended_row_chunking_for_parallelism(3, 2) {
        SparseChunkingStrategy::ByRowCount { rows_per_chunk } => {
            SparseChunkingStrategy::ByRowCount { rows_per_chunk }
        }
        other => other,
    };

    let runtime_plan = task.runtime_plan(strategy);
    let grouped_args = RuntimeArguments::new(&[2.0, 0.25, -4.0], Some(&[1.5, 0.5]));
    let flat_args = grouped_args.flatten();

    let expected: Vec<f64> = entries
        .iter()
        .map(|entry| {
            entry
                .expr
                .lambdify_borrowed_thread_safe(&["alpha", "beta", "y0", "y1", "y2"])(
                &flat_args
            )
        })
        .collect();

    let mut actual = vec![0.0; runtime_plan.nnz()];
    for chunk in &runtime_plan.chunks {
        let values: Vec<f64> = chunk
            .entries
            .iter()
            .map(|entry| {
                entry
                    .expr
                    .lambdify_borrowed_thread_safe(&chunk.plan.input_names)(
                    &flat_args
                )
            })
            .collect();
        actual[chunk.value_range()].copy_from_slice(&values);
    }

    assert_eq!(actual.len(), expected.len());
    for (lhs, rhs) in actual.iter().zip(expected.iter()) {
        assert!(
            (lhs - rhs).abs() < 1e-12,
            "jacobian value mismatch: {lhs} vs {rhs}"
        );
    }

    let sparse_matrix = runtime_plan.assemble_sparse_col_mat(&actual);
    assert_eq!(sparse_matrix.nrows(), 3);
    assert_eq!(sparse_matrix.ncols(), 3);
    assert_eq!(sparse_matrix.compute_nnz(), expected.len());
}

#[test]
fn real_bvp_residual_runtime_plan_matches_existing_lambdify_values() {
    let jac = build_real_bvp_damp1_case(24);
    let variable_names: Vec<&str> = jac
        .variable_string
        .iter()
        .map(|name| name.as_str())
        .collect();
    let task = ResidualTask {
        fn_name: "eval_bvp_residual",
        residuals: &jac.vector_of_functions,
        variables: &variable_names,
        params: None,
    };
    let runtime_plan = task.runtime_plan(
        crate::symbolic::codegen_runtime_api::ResidualChunkingStrategy::ByOutputCount {
            max_outputs_per_chunk: 32,
        },
    );

    let flat_args: Vec<f64> = (0..jac.variable_string.len())
        .map(|index| 0.2 + index as f64 * 0.01)
        .collect();
    let expected = eval_exprs_with_lambdify(&jac.vector_of_functions, &variable_names, &flat_args);

    let actual: Vec<f64> = runtime_plan
        .chunks
        .iter()
        .flat_map(|chunk| {
            chunk
                .residuals
                .iter()
                .map(|expr| expr.lambdify_borrowed_thread_safe(&chunk.plan.input_names)(&flat_args))
                .collect::<Vec<_>>()
        })
        .collect();

    assert_eq!(actual.len(), expected.len());
    for (lhs, rhs) in actual.iter().zip(expected.iter()) {
        assert!(
            (lhs - rhs).abs() < 1e-11,
            "real BVP residual mismatch: {lhs} vs {rhs}"
        );
    }
}

#[test]
fn real_bvp_sparse_jacobian_runtime_plan_matches_existing_lambdify_values() {
    let jac = build_real_bvp_damp1_case(24);
    let variable_names: Vec<&str> = jac
        .variable_string
        .iter()
        .map(|name| name.as_str())
        .collect();
    let entries_owned = jac.symbolic_jacobian_sparse_entries_owned();
    let entries = borrowed_sparse_entries(&entries_owned);
    let task = SparseJacobianTask {
        fn_name: "eval_bvp_sparse_values",
        shape: (jac.symbolic_jacobian.len(), jac.symbolic_jacobian.len()),
        entries: &entries,
        variables: &variable_names,
        params: None,
    };
    let runtime_plan = task.runtime_plan(recommended_row_chunking_for_parallelism(
        jac.symbolic_jacobian.len(),
        4,
    ));

    let flat_args: Vec<f64> = (0..jac.variable_string.len())
        .map(|index| 0.2 + index as f64 * 0.01)
        .collect();
    let expected: Vec<f64> = entries
        .iter()
        .map(|entry| entry.expr.lambdify_borrowed_thread_safe(&variable_names)(&flat_args))
        .collect();

    let mut actual = vec![0.0; runtime_plan.nnz()];
    for chunk in &runtime_plan.chunks {
        let values: Vec<f64> = chunk
            .entries
            .iter()
            .map(|entry| {
                entry
                    .expr
                    .lambdify_borrowed_thread_safe(&chunk.plan.input_names)(
                    &flat_args
                )
            })
            .collect();
        actual[chunk.value_range()].copy_from_slice(&values);
    }

    assert_eq!(actual.len(), expected.len());
    for (lhs, rhs) in actual.iter().zip(expected.iter()) {
        assert!(
            (lhs - rhs).abs() < 1e-11,
            "real BVP jacobian mismatch: {lhs} vs {rhs}"
        );
    }

    let sparse_matrix = runtime_plan.assemble_sparse_col_mat(&actual);
    assert_eq!(sparse_matrix.nrows(), jac.symbolic_jacobian.len());
    assert_eq!(sparse_matrix.ncols(), jac.symbolic_jacobian.len());
    assert_eq!(sparse_matrix.compute_nnz(), expected.len());
}

#[test]
fn compiled_generated_bvp_residual_fixture_matches_existing_lambdify_values() {
    let jac = build_real_bvp_damp1_case(8);
    let variable_names: Vec<&str> = jac
        .variable_string
        .iter()
        .map(|name| name.as_str())
        .collect();
    let flat_args: Vec<f64> = (0..jac.variable_string.len())
        .map(|index| 0.2 + index as f64 * 0.01)
        .collect();

    let expected = eval_exprs_with_lambdify(&jac.vector_of_functions, &variable_names, &flat_args);
    let actual = eval_generated_bvp_residual_fixture(&flat_args);

    assert_eq!(actual.len(), expected.len());
    for (lhs, rhs) in actual.iter().zip(expected.iter()) {
        assert!(
            (lhs - rhs).abs() < 1e-11,
            "compiled residual fixture mismatch: {lhs} vs {rhs}"
        );
    }
}

#[test]
fn compiled_generated_bvp_sparse_fixture_matches_existing_lambdify_values() {
    let jac = build_real_bvp_damp1_case(8);
    let variable_names: Vec<&str> = jac
        .variable_string
        .iter()
        .map(|name| name.as_str())
        .collect();
    let entries_owned = jac.symbolic_jacobian_sparse_entries_owned();
    let entries = borrowed_sparse_entries(&entries_owned);
    let flat_args: Vec<f64> = (0..jac.variable_string.len())
        .map(|index| 0.2 + index as f64 * 0.01)
        .collect();

    let expected: Vec<f64> = entries
        .iter()
        .map(|entry| entry.expr.lambdify_borrowed_thread_safe(&variable_names)(&flat_args))
        .collect();
    let actual = eval_generated_bvp_sparse_fixture(&flat_args);

    assert_eq!(actual.len(), expected.len());
    for (lhs, rhs) in actual.iter().zip(expected.iter()) {
        assert!(
            (lhs - rhs).abs() < 1e-11,
            "compiled sparse fixture mismatch: {lhs} vs {rhs}"
        );
    }
}
