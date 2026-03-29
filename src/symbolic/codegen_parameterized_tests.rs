#![cfg(test)]

//! Parameterized AOT correctness checks.
//!
//! This module verifies that the AOT pipeline preserves the semantic split
//! between:
//! - differentiable variables,
//! - non-differentiated evaluation parameters.
//!
//! Scope:
//! - dense algebraic residuals with params,
//! - dense algebraic Jacobians with params,
//! - IR-level AOT checks,
//! - compiled generated-fixture checks.
//!
//! Current example scale:
//! - 2 residual outputs,
//! - dense 2x2 Jacobian,
//! - 3 scalar parameters,
//! - 2 differentiation variables.
//!
//! Baselines and runtime paths:
//! - legacy parameterized residual via
//!   `Jacobian::lambdify_vector_funvector_DVector_with_parameters_parallel`,
//! - legacy parameterized Jacobian via
//!   `Jacobian::lambdify_jacobian_DMatrix_with_parameters_parallel`,
//! - sequential AOT executors on checked-in generated fixtures,
//! - parallel AOT executors on the same checked-in fixtures.

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
use crate::symbolic::test_codegen_generated_parameterized_dense_fixtures::generated_parameterized_dense_fixture;
use nalgebra::DVector;
use std::time::Instant;

fn build_parameterized_dense_case() -> Jacobian {
    let mut jacobian = Jacobian::new();
    let eq1 = Expr::parse_expression("a*x + b*y");
    let eq2 = Expr::parse_expression("c*x*y");
    jacobian.set_vector_of_functions(vec![eq1, eq2]);
    jacobian.variable_string = vec!["x".to_string(), "y".to_string()];
    jacobian.parameters_string = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    jacobian.set_variables(vec!["x", "y"]);
    jacobian.calc_jacobian();
    jacobian
}

fn parameterized_names<'a>(jacobian: &'a Jacobian) -> (Vec<&'a str>, Vec<&'a str>) {
    let variables = jacobian
        .variable_string
        .iter()
        .map(|name| name.as_str())
        .collect::<Vec<_>>();
    let params = jacobian
        .parameters_string
        .iter()
        .map(|name| name.as_str())
        .collect::<Vec<_>>();
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

fn bind_parameterized_residual_fixture_chunks<'a>(
    plan: &'a crate::symbolic::codegen_runtime_api::ResidualRuntimePlan<'a>,
) -> Vec<ResidualChunkBinding<'a>> {
    let chunk_fns: Vec<crate::symbolic::codegen_orchestrator::GeneratedChunkFn> =
        if plan.chunks.len() == 1 {
            vec![generated_parameterized_dense_fixture::fixture_parameterized_residual_chunk_0]
        } else {
            vec![
                generated_parameterized_dense_fixture::fixture_parameterized_residual_chunk_split_0,
                generated_parameterized_dense_fixture::fixture_parameterized_residual_chunk_split_1,
            ]
        };

    plan.chunks
        .iter()
        .zip(chunk_fns.into_iter())
        .map(|(chunk, eval)| ResidualChunkBinding {
            fn_name: chunk.fn_name.as_str(),
            output_offset: chunk.output_offset,
            output_len: chunk.residuals.len(),
            eval,
        })
        .collect()
}

fn bind_parameterized_jacobian_fixture_chunks<'a>(
    plan: &'a crate::symbolic::codegen_runtime_api::DenseJacobianRuntimePlan<'a>,
) -> Vec<DenseJacobianChunkBinding<'a>> {
    let chunk_fns: Vec<crate::symbolic::codegen_orchestrator::GeneratedChunkFn> =
        if plan.chunks.len() == 1 {
            vec![generated_parameterized_dense_fixture::fixture_parameterized_jacobian_chunk_0]
        } else {
            vec![
                generated_parameterized_dense_fixture::fixture_parameterized_jacobian_chunk_split_0,
                generated_parameterized_dense_fixture::fixture_parameterized_jacobian_chunk_split_1,
            ]
        };

    plan.chunks
        .iter()
        .zip(chunk_fns.into_iter())
        .map(|(chunk, eval)| DenseJacobianChunkBinding {
            fn_name: chunk.fn_name.as_str(),
            value_offset: chunk.value_offset,
            value_len: chunk.value_range().len(),
            eval,
        })
        .collect()
}

#[test]
fn parameterized_residual_aot_ir_matches_existing_lambdify() {
    let mut jacobian = build_parameterized_dense_case();
    jacobian.lambdify_vector_funvector_DVector_with_parameters_parallel();

    let (variables, params) = parameterized_names(&jacobian);
    let task = ResidualTask {
        fn_name: "parameterized_residual_eval",
        residuals: &jacobian.vector_of_functions,
        variables: variables.as_slice(),
        params: Some(params.as_slice()),
    };
    let runtime_plan = task.runtime_plan(ResidualChunkingStrategy::Whole);
    let blocks = residual_ir_blocks(&runtime_plan);

    let params_values = DVector::from_vec(vec![2.0, 3.0, 4.0]);
    let var_values = DVector::from_vec(vec![1.0, 2.0]);
    let flat_args = vec![2.0, 3.0, 4.0, 1.0, 2.0];
    let expected = (jacobian.lambdified_function_with_params)(&params_values, &var_values);
    let actual = DVector::from_vec(eval_ir_blocks_sequential(
        flat_args.as_slice(),
        blocks.as_slice(),
        runtime_plan.output_len,
    ));

    assert_eq!(actual, expected);
}

#[test]
fn parameterized_jacobian_aot_ir_matches_existing_lambdify() {
    let mut jacobian = build_parameterized_dense_case();
    jacobian.lambdify_jacobian_DMatrix_with_parameters_parallel();

    let (variables, params) = parameterized_names(&jacobian);
    let task = JacobianTask {
        fn_name: "parameterized_jacobian_eval",
        jacobian: &jacobian.symbolic_jacobian,
        variables: variables.as_slice(),
        params: Some(params.as_slice()),
    };
    let runtime_plan = dense_runtime_plan(&task, DenseJacobianChunkingStrategy::Whole);
    let blocks = dense_ir_blocks(&runtime_plan);

    let params_values = DVector::from_vec(vec![2.0, 3.0, 4.0]);
    let var_values = DVector::from_vec(vec![1.0, 2.0]);
    let flat_args = vec![2.0, 3.0, 4.0, 1.0, 2.0];
    let expected = (jacobian.lambdified_jacobian_DMatrix_with_params)(&params_values, &var_values);
    let actual = runtime_plan.assemble_dense_matrix(
        eval_ir_blocks_sequential(flat_args.as_slice(), blocks.as_slice(), runtime_plan.len())
            .as_slice(),
    );

    assert_eq!(actual, expected);
}

#[test]
fn compiled_generated_parameterized_dense_fixture_matches_existing_lambdify() {
    let mut jacobian = build_parameterized_dense_case();
    jacobian.lambdify_vector_funvector_DVector_with_parameters_parallel();
    jacobian.lambdify_jacobian_DMatrix_with_parameters_parallel();

    let (variables, params) = parameterized_names(&jacobian);
    let residual_task = ResidualTask {
        fn_name: "fixture_parameterized_residual",
        residuals: &jacobian.vector_of_functions,
        variables: variables.as_slice(),
        params: Some(params.as_slice()),
    };
    let residual_plan = residual_task.runtime_plan(ResidualChunkingStrategy::Whole);

    let jacobian_task = JacobianTask {
        fn_name: "fixture_parameterized_jacobian",
        jacobian: &jacobian.symbolic_jacobian,
        variables: variables.as_slice(),
        params: Some(params.as_slice()),
    };
    let jacobian_plan = dense_runtime_plan(&jacobian_task, DenseJacobianChunkingStrategy::Whole);

    let residual_executor = SequentialResidualExecutor::new(
        &residual_plan,
        bind_parameterized_residual_fixture_chunks(&residual_plan),
    );
    let dense_sequential = SequentialDenseJacobianExecutor::new(
        &jacobian_plan,
        bind_parameterized_jacobian_fixture_chunks(&jacobian_plan),
    );
    let dense_parallel = ParallelDenseJacobianExecutor::with_config(
        &jacobian_plan,
        bind_parameterized_jacobian_fixture_chunks(&jacobian_plan),
        ParallelExecutorConfig {
            jobs_per_worker: 1,
            max_residual_jobs: None,
            max_sparse_jobs: Some(2),
            fallback_policy: ParallelFallbackPolicy::Never,
        },
    );

    let params_values = DVector::from_vec(vec![2.0, 3.0, 4.0]);
    let var_values = DVector::from_vec(vec![1.0, 2.0]);
    let flat_args = vec![2.0, 3.0, 4.0, 1.0, 2.0];

    let expected_residual = (jacobian.lambdified_function_with_params)(&params_values, &var_values);
    let expected_jacobian =
        (jacobian.lambdified_jacobian_DMatrix_with_params)(&params_values, &var_values);

    assert_eq!(
        DVector::from_vec(residual_executor.eval(flat_args.as_slice())),
        expected_residual
    );
    assert_eq!(
        dense_sequential.eval_dense_matrix(flat_args.as_slice()),
        expected_jacobian
    );
    assert_eq!(
        dense_parallel.eval_dense_matrix(flat_args.as_slice()),
        expected_jacobian
    );
}

#[test]
fn benchmark_compiled_parameterized_dense_aot_runtime_paths() {
    let mut jacobian = build_parameterized_dense_case();
    jacobian.lambdify_vector_funvector_DVector_with_parameters_parallel();
    jacobian.lambdify_jacobian_DMatrix_with_parameters_parallel();

    let (variables, params) = parameterized_names(&jacobian);
    let residual_task = ResidualTask {
        fn_name: "fixture_parameterized_residual",
        residuals: &jacobian.vector_of_functions,
        variables: variables.as_slice(),
        params: Some(params.as_slice()),
    };
    let residual_plan = residual_task.runtime_plan(ResidualChunkingStrategy::ByOutputCount {
        max_outputs_per_chunk: 1,
    });
    let residual_bindings = bind_parameterized_residual_fixture_chunks(&residual_plan);
    let residual_seq = SequentialResidualExecutor::new(&residual_plan, residual_bindings.clone());
    let residual_par = ParallelResidualExecutor::with_config(
        &residual_plan,
        residual_bindings,
        benchmark_parallel_config(),
    );

    let jacobian_task = JacobianTask {
        fn_name: "fixture_parameterized_jacobian",
        jacobian: &jacobian.symbolic_jacobian,
        variables: variables.as_slice(),
        params: Some(params.as_slice()),
    };
    let jacobian_plan = dense_runtime_plan(
        &jacobian_task,
        DenseJacobianChunkingStrategy::ByRowCount { rows_per_chunk: 1 },
    );
    let jacobian_bindings = bind_parameterized_jacobian_fixture_chunks(&jacobian_plan);
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

    let params_values = DVector::from_vec(vec![2.0, 3.0, 4.0]);
    let var_values = DVector::from_vec(vec![1.0, 2.0]);
    let flat_args = vec![2.0, 3.0, 4.0, 1.0, 2.0];

    let expected_residual = (jacobian.lambdified_function_with_params)(&params_values, &var_values);
    let expected_jacobian =
        (jacobian.lambdified_jacobian_DMatrix_with_params)(&params_values, &var_values);
    assert_eq!(
        DVector::from_vec(residual_par.eval(flat_args.as_slice())),
        expected_residual
    );
    assert_eq!(
        jacobian_par.eval_dense_matrix(flat_args.as_slice()),
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
            let _ = residual_seq.eval(flat_args.as_slice());
        }
        residual_seq_samples.push(start.elapsed());

        let start = Instant::now();
        for _ in 0..RESIDUAL_ITERS {
            let _ = residual_par.eval(flat_args.as_slice());
        }
        residual_par_samples.push(start.elapsed());

        let start = Instant::now();
        for _ in 0..JACOBIAN_ITERS {
            let _ = jacobian_seq.eval_dense_matrix(flat_args.as_slice());
        }
        jacobian_seq_samples.push(start.elapsed());

        let start = Instant::now();
        for _ in 0..JACOBIAN_ITERS {
            let _ = jacobian_par.eval_dense_matrix(flat_args.as_slice());
        }
        jacobian_par_samples.push(start.elapsed());
    }

    println!("=== Compiled parameterized dense AOT runtime ===");
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
        "parameterized dense plans: residual_chunks={}, jacobian_chunks={}, jobs_residual={}, jobs_dense={}",
        residual_plan.chunks.len(),
        jacobian_plan.chunks.len(),
        residual_par.job_count(),
        jacobian_par.job_count()
    );
}
