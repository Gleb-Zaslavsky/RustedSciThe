#![cfg(test)]

//! IVP AOT correctness checks.
//!
//! This is the first dense IVP vertical slice. It mirrors the early dense
//! algebraic slice:
//! - small and safe,
//! - correctness-focused,
//! - IR-level comparison against the legacy IVP path,
//! - compiled generated-fixture comparison through the orchestrator layer.
//!
//! Baselines used here:
//! - legacy IVP residual closure stored in `Jacobian::lambdified_functions_IVP_DVector`,
//! - legacy IVP dense Jacobian closure stored in
//!   `Jacobian::function_jacobian_IVP_DMatrix`.
//!
//! The current IVP fixture is intentionally tiny:
//! - 2 residual outputs,
//! - dense 2x2 Jacobian,
//! - 1 distinguished time argument,
//! - no extra params yet.
//!
//! Runtime paths compared:
//! - IR-level AOT lowering/evaluation,
//! - checked-in compiled residual fixture through sequential/parallel residual
//!   executors,
//! - checked-in compiled dense Jacobian fixture through sequential/parallel
//!   dense Jacobian executors.
//!
//! That makes it a good correctness slice and a lightweight hot-path baseline,
//! but not yet a realistic crossover-scale perf case like the larger BVP
//! stress fixtures.

use crate::symbolic::CodegenIR::LinearBlock;
use crate::symbolic::codegen_adapters::{
    ivp_dense_ir_blocks, ivp_dense_runtime_plan, ivp_residual_ir_blocks, ivp_residual_runtime_plan,
};
use crate::symbolic::codegen_orchestrator::ParallelResidualExecutor;
use crate::symbolic::codegen_orchestrator::{
    DenseJacobianChunkBinding, ParallelDenseJacobianExecutor, ParallelExecutorConfig,
    ParallelFallbackPolicy, ResidualChunkBinding, SequentialDenseJacobianExecutor,
    SequentialResidualExecutor,
};
use crate::symbolic::codegen_runtime_api::{
    DenseJacobianChunkingStrategy, ResidualChunkingStrategy,
};
use crate::symbolic::codegen_tasks::{IvpJacobianTask, IvpResidualTask};
use crate::symbolic::codegen_test_support::{
    benchmark_parallel_config, median_duration, per_iter_ns,
};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions::Jacobian;
use crate::symbolic::test_codegen_generated_ivp_fixtures::generated_ivp_fixture;
use nalgebra::DVector;
use std::time::Instant;

fn build_ivp_case() -> Jacobian {
    let mut jacobian = Jacobian::new();
    let eq_system = vec![
        Expr::parse_expression("t + y"),
        Expr::parse_expression("t*y - z"),
    ];
    let variables = vec!["y".to_string(), "z".to_string()];
    let arg = "t".to_string();

    jacobian.generate_IVP_ODEsolver(eq_system, variables, arg);
    jacobian
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

fn bind_ivp_residual_fixture_chunks<'a>(
    plan: &'a crate::symbolic::codegen_runtime_api::ResidualRuntimePlan<'a>,
) -> Vec<ResidualChunkBinding<'a>> {
    let chunk_fns: Vec<crate::symbolic::codegen_orchestrator::GeneratedChunkFn> =
        if plan.chunks.len() == 1 {
            vec![generated_ivp_fixture::fixture_ivp_residual_chunk_0]
        } else {
            vec![
                generated_ivp_fixture::fixture_ivp_residual_chunk_split_0,
                generated_ivp_fixture::fixture_ivp_residual_chunk_split_1,
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

fn bind_ivp_jacobian_fixture_chunks<'a>(
    plan: &'a crate::symbolic::codegen_runtime_api::DenseJacobianRuntimePlan<'a>,
) -> Vec<DenseJacobianChunkBinding<'a>> {
    plan.chunks
        .iter()
        .map(|chunk| DenseJacobianChunkBinding {
            fn_name: chunk.fn_name.as_str(),
            value_offset: chunk.value_offset,
            value_len: chunk.value_range().len(),
            eval: generated_ivp_fixture::fixture_ivp_jacobian_chunk_0,
        })
        .collect()
}

#[test]
fn ivp_residual_aot_ir_matches_existing_lambdify() {
    let jacobian = build_ivp_case();
    let task = IvpResidualTask {
        fn_name: "ivp_residual_eval",
        time_arg: "t",
        residuals: &jacobian.vector_of_functions,
        variables: &["y", "z"],
        params: None,
    };
    let runtime_plan = ivp_residual_runtime_plan(&task, ResidualChunkingStrategy::Whole);
    let blocks = ivp_residual_ir_blocks(&runtime_plan);

    let time = 1.5;
    let state = DVector::from_vec(vec![2.0, 4.0]);
    let mut args = vec![time];
    args.extend(state.iter().copied());

    let expected = (jacobian.lambdified_functions_IVP_DVector)(time, &state);
    let actual = DVector::from_vec(eval_ir_blocks_sequential(
        args.as_slice(),
        blocks.as_slice(),
        runtime_plan.output_len,
    ));

    assert_eq!(actual, expected);
}

#[test]
fn ivp_jacobian_aot_ir_matches_existing_lambdify() {
    let jacobian = build_ivp_case();
    let task = IvpJacobianTask {
        fn_name: "ivp_jacobian_eval",
        time_arg: "t",
        jacobian: &jacobian.symbolic_jacobian,
        variables: &["y", "z"],
        params: None,
    };
    let runtime_plan = ivp_dense_runtime_plan(&task, DenseJacobianChunkingStrategy::Whole);
    let blocks = ivp_dense_ir_blocks(&runtime_plan);

    let time = 1.5;
    let state = DVector::from_vec(vec![2.0, 4.0]);
    let mut args = vec![time];
    args.extend(state.iter().copied());

    let expected = (jacobian.function_jacobian_IVP_DMatrix)(time, &state);
    let actual = runtime_plan.assemble_dense_matrix(
        eval_ir_blocks_sequential(args.as_slice(), blocks.as_slice(), runtime_plan.len())
            .as_slice(),
    );

    assert_eq!(actual, expected);
}

#[test]
fn compiled_generated_ivp_fixture_matches_existing_lambdify() {
    let jacobian = build_ivp_case();

    let residual_task = IvpResidualTask {
        fn_name: "fixture_ivp_residual",
        time_arg: "t",
        residuals: &jacobian.vector_of_functions,
        variables: &["y", "z"],
        params: None,
    };
    let residual_plan = ivp_residual_runtime_plan(&residual_task, ResidualChunkingStrategy::Whole);
    let residual_executor = SequentialResidualExecutor::new(
        &residual_plan,
        bind_ivp_residual_fixture_chunks(&residual_plan),
    );

    let jacobian_task = IvpJacobianTask {
        fn_name: "fixture_ivp_jacobian",
        time_arg: "t",
        jacobian: &jacobian.symbolic_jacobian,
        variables: &["y", "z"],
        params: None,
    };
    let jacobian_plan =
        ivp_dense_runtime_plan(&jacobian_task, DenseJacobianChunkingStrategy::Whole);
    let dense_bindings = bind_ivp_jacobian_fixture_chunks(&jacobian_plan);
    let sequential_dense =
        SequentialDenseJacobianExecutor::new(&jacobian_plan, dense_bindings.clone());
    let parallel_dense = ParallelDenseJacobianExecutor::with_config(
        &jacobian_plan,
        dense_bindings,
        ParallelExecutorConfig {
            jobs_per_worker: 1,
            max_residual_jobs: None,
            max_sparse_jobs: Some(2),
            fallback_policy: ParallelFallbackPolicy::Never,
        },
    );

    let time = 1.5;
    let state = DVector::from_vec(vec![2.0, 4.0]);
    let mut args = vec![time];
    args.extend(state.iter().copied());

    let expected_residual = (jacobian.lambdified_functions_IVP_DVector)(time, &state);
    let actual_residual = DVector::from_vec(residual_executor.eval(args.as_slice()));
    assert_eq!(actual_residual, expected_residual);

    let expected_jacobian = (jacobian.function_jacobian_IVP_DMatrix)(time, &state);
    let actual_jacobian_seq = sequential_dense.eval_dense_matrix(args.as_slice());
    let actual_jacobian_par = parallel_dense.eval_dense_matrix(args.as_slice());
    assert_eq!(actual_jacobian_seq, expected_jacobian);
    assert_eq!(actual_jacobian_par, expected_jacobian);
}

#[test]
fn ivp_runtime_plan_preserves_time_first_argument_order() {
    let task = IvpResidualTask {
        fn_name: "ivp_residual_eval",
        time_arg: "t",
        residuals: &[
            Expr::parse_expression("t + y"),
            Expr::parse_expression("t*y - z"),
        ],
        variables: &["y", "z"],
        params: None,
    };

    let runtime_plan = ivp_residual_runtime_plan(&task, ResidualChunkingStrategy::Whole);

    assert_eq!(runtime_plan.input_names, vec!["t", "y", "z"]);
}

#[test]
fn benchmark_compiled_ivp_aot_runtime_paths() {
    let jacobian = build_ivp_case();

    let residual_task = IvpResidualTask {
        fn_name: "fixture_ivp_residual",
        time_arg: "t",
        residuals: &jacobian.vector_of_functions,
        variables: &["y", "z"],
        params: None,
    };
    let residual_plan = ivp_residual_runtime_plan(
        &residual_task,
        ResidualChunkingStrategy::ByOutputCount {
            max_outputs_per_chunk: 1,
        },
    );
    let residual_bindings = bind_ivp_residual_fixture_chunks(&residual_plan);
    let residual_seq = SequentialResidualExecutor::new(&residual_plan, residual_bindings.clone());
    let residual_par = ParallelResidualExecutor::with_config(
        &residual_plan,
        residual_bindings,
        benchmark_parallel_config(),
    );

    let jacobian_task = IvpJacobianTask {
        fn_name: "fixture_ivp_jacobian",
        time_arg: "t",
        jacobian: &jacobian.symbolic_jacobian,
        variables: &["y", "z"],
        params: None,
    };
    let jacobian_plan =
        ivp_dense_runtime_plan(&jacobian_task, DenseJacobianChunkingStrategy::Whole);
    let dense_bindings = bind_ivp_jacobian_fixture_chunks(&jacobian_plan);
    let dense_seq = SequentialDenseJacobianExecutor::new(&jacobian_plan, dense_bindings.clone());
    let dense_par = ParallelDenseJacobianExecutor::with_config(
        &jacobian_plan,
        dense_bindings,
        ParallelExecutorConfig {
            jobs_per_worker: 1,
            max_residual_jobs: None,
            max_sparse_jobs: Some(2),
            fallback_policy: ParallelFallbackPolicy::Never,
        },
    );

    let time = 1.5;
    let state = DVector::from_vec(vec![2.0, 4.0]);
    let mut args = vec![time];
    args.extend(state.iter().copied());

    let expected_residual = (jacobian.lambdified_functions_IVP_DVector)(time, &state);
    let expected_jacobian = (jacobian.function_jacobian_IVP_DMatrix)(time, &state);
    assert_eq!(
        DVector::from_vec(residual_seq.eval(args.as_slice())),
        expected_residual
    );
    assert_eq!(
        dense_seq.eval_dense_matrix(args.as_slice()),
        expected_jacobian
    );
    assert_eq!(
        DVector::from_vec(residual_par.eval(args.as_slice())),
        expected_residual
    );
    assert_eq!(
        dense_par.eval_dense_matrix(args.as_slice()),
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
            let _ = dense_seq.eval_dense_matrix(args.as_slice());
        }
        jacobian_seq_samples.push(start.elapsed());

        let start = Instant::now();
        for _ in 0..JACOBIAN_ITERS {
            let _ = dense_par.eval_dense_matrix(args.as_slice());
        }
        jacobian_par_samples.push(start.elapsed());
    }

    let residual_seq_ns = per_iter_ns(median_duration(residual_seq_samples), RESIDUAL_ITERS);
    let residual_par_ns = per_iter_ns(median_duration(residual_par_samples), RESIDUAL_ITERS);
    let jacobian_seq_ns = per_iter_ns(median_duration(jacobian_seq_samples), JACOBIAN_ITERS);
    let jacobian_par_ns = per_iter_ns(median_duration(jacobian_par_samples), JACOBIAN_ITERS);

    println!("=== Compiled IVP AOT runtime ===");
    println!("sequential AOT residual: {:.2} ns/call", residual_seq_ns);
    println!("parallel AOT residual: {:.2} ns/call", residual_par_ns);
    println!(
        "sequential AOT dense Jacobian: {:.2} ns/call",
        jacobian_seq_ns
    );
    println!(
        "parallel AOT dense Jacobian: {:.2} ns/call",
        jacobian_par_ns
    );
    println!(
        "ivp plans: residual_chunks={}, jacobian_chunks={}, jobs_residual={}, jobs_dense={}",
        residual_plan.chunks.len(),
        jacobian_plan.chunks.len(),
        residual_par.job_count(),
        dense_par.job_count()
    );
}
