#![cfg(test)]

//! Dense algebraic codegen checks.
//!
//! This module is the first dense vertical slice built on top of the shared
//! AOT layers:
//! - `codegen_tasks`
//! - `codegen_runtime_api`
//! - `codegen_adapters`
//! - `CodegenIR`
//!
//! Scope of this first step:
//! - correctness only,
//! - IR-level AOT evaluation compared directly against the existing
//!   symbolic/lambdify dense path,
//! - compiled generated dense fixtures through sequential and parallel
//!   dense Jacobian executors.
//!
//! Current example scale:
//! - 3 residual outputs,
//! - dense 3x2 Jacobian,
//! - optional scalar parameter group present.
//!
//! Baselines and runtime paths:
//! - legacy dense residual via `Jacobian::lambdify_funcvector`,
//! - legacy dense Jacobian via `Jacobian::jacobian_generate`,
//! - sequential AOT executors on checked-in generated fixtures,
//! - parallel AOT executors on the same checked-in fixtures.
//!
//! The goal is to keep the extension small and safe while proving that dense
//! residuals and dense Jacobians can already travel through the same staged
//! pipeline that BVP sparse tasks use.

use crate::symbolic::CodegenIR::LinearBlock;
use crate::symbolic::codegen_adapters::{dense_ir_blocks, dense_runtime_plan, residual_ir_blocks};
use crate::symbolic::codegen_orchestrator::{
    DenseJacobianChunkBinding, ParallelDenseJacobianExecutor, ParallelExecutorConfig,
    ParallelFallbackPolicy, ParallelResidualExecutor, ResidualChunkBinding,
    SequentialDenseJacobianExecutor, SequentialResidualExecutor,
};
use crate::symbolic::codegen_runtime_api::{
    DenseJacobianChunkingStrategy, ResidualChunkingStrategy,
};
use crate::symbolic::codegen_tasks::{JacobianTask, ResidualTask};
use crate::symbolic::codegen_test_support::{
    benchmark_parallel_config, median_duration, per_iter_ns,
};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions::Jacobian;
use crate::symbolic::test_codegen_generated_dense_fixtures::generated_dense_fixture;
use nalgebra::{DMatrix, DVector};
use std::time::Instant;

fn build_dense_algebraic_case() -> Jacobian {
    let mut jacobian = Jacobian::new();
    jacobian.set_funcvecor_from_str(vec![
        "x^2 + y + a".to_string(),
        "x - y^2 + b".to_string(),
        "x*y + a*b".to_string(),
    ]);
    jacobian.set_varvecor_from_str("x, y");
    jacobian.parameters_string = vec!["a".to_string(), "b".to_string()];
    jacobian.calc_jacobian();
    jacobian
}

fn dense_variables_and_params<'a>(jacobian: &'a Jacobian) -> (Vec<&'a str>, Option<Vec<&'a str>>) {
    let variables = jacobian
        .variable_string
        .iter()
        .map(|name| name.as_str())
        .collect::<Vec<_>>();
    let params = if jacobian.parameters_string.is_empty() {
        None
    } else {
        Some(
            jacobian
                .parameters_string
                .iter()
                .map(|name| name.as_str())
                .collect::<Vec<_>>(),
        )
    };
    (variables, params)
}

fn eval_ir_blocks_sequential(
    args: &[f64],
    blocks: &[(usize, LinearBlock)],
    output_len: usize,
) -> Vec<f64> {
    let mut out = vec![0.0; output_len];
    for (offset, block) in blocks {
        let len = block.outputs.len();
        block.eval_into(args, &mut out[*offset..(*offset + len)]);
    }
    out
}

fn bind_dense_residual_fixture_chunks<'a>(
    plan: &'a crate::symbolic::codegen_runtime_api::ResidualRuntimePlan<'a>,
) -> Vec<ResidualChunkBinding<'a>> {
    let chunk_fns = [
        generated_dense_fixture::fixture_dense_residual_chunk_0
            as crate::symbolic::codegen_orchestrator::GeneratedChunkFn,
        generated_dense_fixture::fixture_dense_residual_chunk_1
            as crate::symbolic::codegen_orchestrator::GeneratedChunkFn,
    ];

    plan.chunks
        .iter()
        .zip(chunk_fns)
        .map(|(chunk, eval)| ResidualChunkBinding {
            fn_name: chunk.fn_name.as_str(),
            output_offset: chunk.output_offset,
            output_len: chunk.residuals.len(),
            eval,
        })
        .collect()
}

fn bind_dense_jacobian_fixture_chunks<'a>(
    plan: &'a crate::symbolic::codegen_runtime_api::DenseJacobianRuntimePlan<'a>,
) -> Vec<DenseJacobianChunkBinding<'a>> {
    let chunk_fns = [
        generated_dense_fixture::fixture_dense_jacobian_chunk_0
            as crate::symbolic::codegen_orchestrator::GeneratedChunkFn,
        generated_dense_fixture::fixture_dense_jacobian_chunk_1
            as crate::symbolic::codegen_orchestrator::GeneratedChunkFn,
    ];

    plan.chunks
        .iter()
        .zip(chunk_fns)
        .map(|(chunk, eval)| DenseJacobianChunkBinding {
            fn_name: chunk.fn_name.as_str(),
            value_offset: chunk.value_offset,
            value_len: chunk.value_range().len(),
            eval,
        })
        .collect()
}

#[test]
fn dense_residual_aot_ir_matches_existing_lambdify() {
    let mut jacobian = build_dense_algebraic_case();
    jacobian.lambdify_funcvector(vec!["a", "b", "x", "y"]);

    let (variables, params) = dense_variables_and_params(&jacobian);
    let param_refs = params.as_ref().map(|params| params.as_slice());
    let task = ResidualTask {
        fn_name: "dense_residual_eval",
        residuals: &jacobian.vector_of_functions,
        variables: variables.as_slice(),
        params: param_refs,
    };
    let runtime_plan = task.runtime_plan(ResidualChunkingStrategy::ByOutputCount {
        max_outputs_per_chunk: 2,
    });
    let blocks = residual_ir_blocks(&runtime_plan);

    let args = vec![1.5, -2.5, 0.75, -1.25];
    let expected = jacobian.evaluate_funvector_lambdified_DVector_unmut(args.clone());
    let actual = DVector::from_vec(eval_ir_blocks_sequential(
        args.as_slice(),
        blocks.as_slice(),
        runtime_plan.output_len,
    ));

    assert_eq!(actual, expected);
}

#[test]
fn dense_jacobian_aot_ir_matches_existing_lambdify() {
    let mut jacobian = build_dense_algebraic_case();
    jacobian.jacobian_generate(vec!["a", "b", "x", "y"]);

    let (variables, params) = dense_variables_and_params(&jacobian);
    let param_refs = params.as_ref().map(|params| params.as_slice());
    let task = JacobianTask {
        fn_name: "dense_jacobian_eval",
        jacobian: &jacobian.symbolic_jacobian,
        variables: variables.as_slice(),
        params: param_refs,
    };
    let runtime_plan = dense_runtime_plan(
        &task,
        DenseJacobianChunkingStrategy::ByRowCount { rows_per_chunk: 2 },
    );
    let blocks = dense_ir_blocks(&runtime_plan);

    let args = vec![1.5, -2.5, 0.75, -1.25];
    let expected = jacobian.evaluate_func_jacobian_DMatrix_unmut(args.clone());
    let actual = runtime_plan.assemble_dense_matrix(
        eval_ir_blocks_sequential(args.as_slice(), blocks.as_slice(), runtime_plan.len())
            .as_slice(),
    );

    assert_eq!(actual, expected);
}

#[test]
fn dense_jacobian_runtime_plan_preserves_row_major_layout() {
    let jacobian = vec![
        vec![Expr::parse_expression("x"), Expr::parse_expression("y")],
        vec![Expr::parse_expression("x + y"), Expr::parse_expression("2")],
        vec![Expr::parse_expression("x*y"), Expr::parse_expression("3")],
    ];
    let task = JacobianTask {
        fn_name: "dense_jacobian_eval",
        jacobian: &jacobian,
        variables: &["x", "y"],
        params: None,
    };

    let runtime_plan = dense_runtime_plan(
        &task,
        DenseJacobianChunkingStrategy::ByRowCount { rows_per_chunk: 2 },
    );
    let blocks = dense_ir_blocks(&runtime_plan);
    let values = eval_ir_blocks_sequential(&[2.0, 5.0], blocks.as_slice(), runtime_plan.len());
    let matrix = DMatrix::from_row_slice(runtime_plan.rows, runtime_plan.cols, values.as_slice());

    assert_eq!(matrix[(0, 0)], 2.0);
    assert_eq!(matrix[(0, 1)], 5.0);
    assert_eq!(matrix[(1, 0)], 7.0);
    assert_eq!(matrix[(1, 1)], 2.0);
    assert_eq!(matrix[(2, 0)], 10.0);
    assert_eq!(matrix[(2, 1)], 3.0);
}

#[test]
fn compiled_generated_dense_residual_fixture_matches_existing_lambdify() {
    let mut jacobian = build_dense_algebraic_case();
    jacobian.lambdify_funcvector(vec!["a", "b", "x", "y"]);

    let (variables, params) = dense_variables_and_params(&jacobian);
    let param_refs = params.as_ref().map(|params| params.as_slice());
    let task = ResidualTask {
        fn_name: "fixture_dense_residual",
        residuals: &jacobian.vector_of_functions,
        variables: variables.as_slice(),
        params: param_refs,
    };
    let runtime_plan = task.runtime_plan(ResidualChunkingStrategy::ByOutputCount {
        max_outputs_per_chunk: 2,
    });
    let executor = SequentialResidualExecutor::new(
        &runtime_plan,
        bind_dense_residual_fixture_chunks(&runtime_plan),
    );

    let args = vec![1.5, -2.5, 0.75, -1.25];
    let expected = jacobian.evaluate_funvector_lambdified_DVector_unmut(args.clone());
    let actual = DVector::from_vec(executor.eval(args.as_slice()));

    assert_eq!(actual, expected);
}

#[test]
fn compiled_generated_dense_jacobian_fixture_matches_existing_lambdify() {
    let mut jacobian = build_dense_algebraic_case();
    jacobian.jacobian_generate(vec!["a", "b", "x", "y"]);

    let (variables, params) = dense_variables_and_params(&jacobian);
    let param_refs = params.as_ref().map(|params| params.as_slice());
    let task = JacobianTask {
        fn_name: "fixture_dense_jacobian",
        jacobian: &jacobian.symbolic_jacobian,
        variables: variables.as_slice(),
        params: param_refs,
    };
    let runtime_plan = dense_runtime_plan(
        &task,
        DenseJacobianChunkingStrategy::ByRowCount { rows_per_chunk: 2 },
    );
    let sequential = SequentialDenseJacobianExecutor::new(
        &runtime_plan,
        bind_dense_jacobian_fixture_chunks(&runtime_plan),
    );
    let parallel = ParallelDenseJacobianExecutor::with_config(
        &runtime_plan,
        bind_dense_jacobian_fixture_chunks(&runtime_plan),
        ParallelExecutorConfig {
            jobs_per_worker: 1,
            max_residual_jobs: None,
            max_sparse_jobs: Some(2),
            fallback_policy: ParallelFallbackPolicy::Never,
        },
    );

    let args = vec![1.5, -2.5, 0.75, -1.25];
    let expected = jacobian.evaluate_func_jacobian_DMatrix_unmut(args.clone());
    let actual_sequential = sequential.eval_dense_matrix(args.as_slice());
    let actual_parallel = parallel.eval_dense_matrix(args.as_slice());

    assert_eq!(actual_sequential, expected);
    assert_eq!(actual_parallel, expected);
}

#[test]
fn benchmark_compiled_dense_aot_runtime_paths() {
    let mut jacobian = build_dense_algebraic_case();
    jacobian.lambdify_funcvector(vec!["a", "b", "x", "y"]);
    jacobian.jacobian_generate(vec!["a", "b", "x", "y"]);

    let (variables, params) = dense_variables_and_params(&jacobian);
    let param_refs = params.as_ref().map(|params| params.as_slice());

    let residual_task = ResidualTask {
        fn_name: "fixture_dense_residual",
        residuals: &jacobian.vector_of_functions,
        variables: variables.as_slice(),
        params: param_refs,
    };
    let residual_plan = residual_task.runtime_plan(ResidualChunkingStrategy::ByOutputCount {
        max_outputs_per_chunk: 2,
    });
    let residual_bindings = bind_dense_residual_fixture_chunks(&residual_plan);
    let residual_seq = SequentialResidualExecutor::new(&residual_plan, residual_bindings.clone());
    let residual_par = ParallelResidualExecutor::with_config(
        &residual_plan,
        residual_bindings,
        benchmark_parallel_config(),
    );

    let jacobian_task = JacobianTask {
        fn_name: "fixture_dense_jacobian",
        jacobian: &jacobian.symbolic_jacobian,
        variables: variables.as_slice(),
        params: param_refs,
    };
    let jacobian_plan = dense_runtime_plan(
        &jacobian_task,
        DenseJacobianChunkingStrategy::ByRowCount { rows_per_chunk: 2 },
    );
    let jacobian_bindings = bind_dense_jacobian_fixture_chunks(&jacobian_plan);
    let jacobian_seq =
        SequentialDenseJacobianExecutor::new(&jacobian_plan, jacobian_bindings.clone());
    let jacobian_par = ParallelDenseJacobianExecutor::with_config(
        &jacobian_plan,
        jacobian_bindings,
        ParallelExecutorConfig {
            jobs_per_worker: 1,
            max_residual_jobs: None,
            max_sparse_jobs: Some(2),
            fallback_policy: ParallelFallbackPolicy::Never,
        },
    );

    let args = vec![1.5, -2.5, 0.75, -1.25];
    let expected_residual = jacobian.evaluate_funvector_lambdified_DVector_unmut(args.clone());
    let expected_jacobian = jacobian.evaluate_func_jacobian_DMatrix_unmut(args.clone());
    assert_eq!(
        DVector::from_vec(residual_par.eval(args.as_slice())),
        expected_residual
    );
    assert_eq!(
        jacobian_par.eval_dense_matrix(args.as_slice()),
        expected_jacobian
    );

    const SAMPLES: usize = 7;
    const RESIDUAL_ITERS: usize = 2000;
    const JACOBIAN_ITERS: usize = 2000;

    let mut residual_seq_samples = Vec::with_capacity(SAMPLES);
    let mut residual_par_samples = Vec::with_capacity(SAMPLES);
    let mut jacobian_seq_samples = Vec::with_capacity(SAMPLES);
    let mut jacobian_par_samples = Vec::with_capacity(SAMPLES);

    for _ in 0..SAMPLES {
        let start = Instant::now();
        for _ in 0..RESIDUAL_ITERS {
            let _ = residual_seq.eval(args.as_slice());
        }
        residual_seq_samples.push(start.elapsed());

        let start = Instant::now();
        for _ in 0..RESIDUAL_ITERS {
            let _ = residual_par.eval(args.as_slice());
        }
        residual_par_samples.push(start.elapsed());

        let start = Instant::now();
        for _ in 0..JACOBIAN_ITERS {
            let _ = jacobian_seq.eval_dense_matrix(args.as_slice());
        }
        jacobian_seq_samples.push(start.elapsed());

        let start = Instant::now();
        for _ in 0..JACOBIAN_ITERS {
            let _ = jacobian_par.eval_dense_matrix(args.as_slice());
        }
        jacobian_par_samples.push(start.elapsed());
    }

    println!("=== Compiled dense AOT runtime ===");
    println!(
        "sequential AOT residual: {:.2} ns/call",
        per_iter_ns(median_duration(residual_seq_samples), RESIDUAL_ITERS)
    );
    println!(
        "parallel AOT residual: {:.2} ns/call",
        per_iter_ns(median_duration(residual_par_samples), RESIDUAL_ITERS)
    );
    println!(
        "sequential AOT dense Jacobian: {:.2} ns/call",
        per_iter_ns(median_duration(jacobian_seq_samples), JACOBIAN_ITERS)
    );
    println!(
        "parallel AOT dense Jacobian: {:.2} ns/call",
        per_iter_ns(median_duration(jacobian_par_samples), JACOBIAN_ITERS)
    );
    println!(
        "dense plans: residual_chunks={}, jacobian_chunks={}, jobs_residual={}, jobs_dense={}",
        residual_plan.chunks.len(),
        jacobian_plan.chunks.len(),
        residual_par.job_count(),
        jacobian_par.job_count()
    );
}
