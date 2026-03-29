#![cfg(test)]

//! Parameterized IVP AOT checks.
//!
//! This slice verifies the case `f(t, y, p)` where:
//! - `t` is the distinguished IVP time argument,
//! - `y` are differentiation variables,
//! - `p` are evaluation-only parameters.
//!
//! Current example scale:
//! - 2 residual outputs,
//! - dense 2x2 Jacobian,
//! - 3 scalar parameters,
//! - 2 state variables.
//!
//! Runtime paths compared:
//! - expression-level baseline using the same flattened argument order as AOT,
//! - IR-level AOT lowering/evaluation,
//! - checked-in compiled fixtures through sequential/parallel residual and
//!   dense Jacobian executors.
//!
//! The legacy symbolic layer does not yet expose one clean parameterized IVP
//! wrapper analogous to the parameterized dense algebraic path, so the baseline
//! here is expression-level compilation with the same flattened input order the
//! AOT pipeline uses:
//! - `t`
//! - `params`
//! - `state variables`

use crate::symbolic::CodegenIR::LinearBlock;
use crate::symbolic::codegen_adapters::{
    ivp_dense_ir_blocks, ivp_dense_runtime_plan, ivp_residual_ir_blocks, ivp_residual_runtime_plan,
};
use crate::symbolic::codegen_orchestrator::{
    DenseJacobianChunkBinding, ParallelDenseJacobianExecutor, ParallelExecutorConfig,
    ParallelFallbackPolicy, ParallelResidualExecutor, ResidualChunkBinding,
    SequentialDenseJacobianExecutor, SequentialResidualExecutor,
};
use crate::symbolic::codegen_runtime_api::{
    DenseJacobianChunkingStrategy, ResidualChunkingStrategy,
};
use crate::symbolic::codegen_tasks::{IvpJacobianTask, IvpResidualTask};
use crate::symbolic::codegen_test_support::{
    benchmark_parallel_config, median_duration, per_iter_ns,
};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::test_codegen_generated_parameterized_ivp_fixtures::generated_parameterized_ivp_fixture;
use nalgebra::{DMatrix, DVector};
use std::time::Instant;

fn build_parameterized_ivp_case() -> (Vec<Expr>, Vec<Vec<Expr>>) {
    let residuals = vec![
        Expr::parse_expression("a*t + y + b*z"),
        Expr::parse_expression("c*y - z + b*t"),
    ];
    let jacobian = residuals
        .iter()
        .map(|expr| vec![expr.diff("y").simplify(), expr.diff("z").simplify()])
        .collect::<Vec<_>>();
    (residuals, jacobian)
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

fn bind_parameterized_ivp_residual_fixture_chunks<'a>(
    plan: &'a crate::symbolic::codegen_runtime_api::ResidualRuntimePlan<'a>,
) -> Vec<ResidualChunkBinding<'a>> {
    let chunk_fns: Vec<crate::symbolic::codegen_orchestrator::GeneratedChunkFn> =
        if plan.chunks.len() == 1 {
            vec![generated_parameterized_ivp_fixture::fixture_parameterized_ivp_residual_chunk_0]
        } else {
            vec![
            generated_parameterized_ivp_fixture::fixture_parameterized_ivp_residual_chunk_split_0,
            generated_parameterized_ivp_fixture::fixture_parameterized_ivp_residual_chunk_split_1,
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

fn bind_parameterized_ivp_jacobian_fixture_chunks<'a>(
    plan: &'a crate::symbolic::codegen_runtime_api::DenseJacobianRuntimePlan<'a>,
) -> Vec<DenseJacobianChunkBinding<'a>> {
    plan.chunks
        .iter()
        .map(|chunk| DenseJacobianChunkBinding {
            fn_name: chunk.fn_name.as_str(),
            value_offset: chunk.value_offset,
            value_len: chunk.value_range().len(),
            eval: generated_parameterized_ivp_fixture::fixture_parameterized_ivp_jacobian_chunk_0,
        })
        .collect()
}

fn baseline_parameterized_ivp_residual(args: &[f64]) -> DVector<f64> {
    let exprs = vec![
        Expr::parse_expression("a*t + y + b*z"),
        Expr::parse_expression("c*y - z + b*t"),
    ];
    let names = ["t", "a", "b", "c", "y", "z"];
    let values = exprs
        .iter()
        .map(|expr| expr.lambdify_borrowed_thread_safe(&names)(args))
        .collect::<Vec<_>>();
    DVector::from_vec(values)
}

fn baseline_parameterized_ivp_jacobian(args: &[f64]) -> DMatrix<f64> {
    let residuals = vec![
        Expr::parse_expression("a*t + y + b*z"),
        Expr::parse_expression("c*y - z + b*t"),
    ];
    let names = ["t", "a", "b", "c", "y", "z"];
    let values = residuals
        .iter()
        .flat_map(|expr| {
            [
                expr.diff("y")
                    .simplify()
                    .lambdify_borrowed_thread_safe(&names)(args),
                expr.diff("z")
                    .simplify()
                    .lambdify_borrowed_thread_safe(&names)(args),
            ]
        })
        .collect::<Vec<_>>();
    DMatrix::from_row_slice(2, 2, values.as_slice())
}

#[test]
fn parameterized_ivp_residual_aot_ir_matches_expression_level_baseline() {
    let (residuals, _) = build_parameterized_ivp_case();
    let task = IvpResidualTask {
        fn_name: "param_ivp_residual_eval",
        time_arg: "t",
        residuals: &residuals,
        variables: &["y", "z"],
        params: Some(&["a", "b", "c"]),
    };
    let runtime_plan = ivp_residual_runtime_plan(&task, ResidualChunkingStrategy::Whole);
    let blocks = ivp_residual_ir_blocks(&runtime_plan);

    let args = vec![1.5, 2.0, -0.5, 3.0, 0.75, -1.25];
    let expected = baseline_parameterized_ivp_residual(args.as_slice());
    let actual = DVector::from_vec(eval_ir_blocks_sequential(
        args.as_slice(),
        blocks.as_slice(),
        runtime_plan.output_len,
    ));

    assert_eq!(actual, expected);
}

#[test]
fn parameterized_ivp_jacobian_aot_ir_matches_expression_level_baseline() {
    let (_, jacobian) = build_parameterized_ivp_case();
    let task = IvpJacobianTask {
        fn_name: "param_ivp_jacobian_eval",
        time_arg: "t",
        jacobian: &jacobian,
        variables: &["y", "z"],
        params: Some(&["a", "b", "c"]),
    };
    let runtime_plan = ivp_dense_runtime_plan(&task, DenseJacobianChunkingStrategy::Whole);
    let blocks = ivp_dense_ir_blocks(&runtime_plan);

    let args = vec![1.5, 2.0, -0.5, 3.0, 0.75, -1.25];
    let expected = baseline_parameterized_ivp_jacobian(args.as_slice());
    let actual = runtime_plan.assemble_dense_matrix(
        eval_ir_blocks_sequential(args.as_slice(), blocks.as_slice(), runtime_plan.len())
            .as_slice(),
    );

    assert_eq!(actual, expected);
}

#[test]
fn compiled_generated_parameterized_ivp_fixture_matches_expression_level_baseline() {
    let (residuals, jacobian) = build_parameterized_ivp_case();

    let residual_task = IvpResidualTask {
        fn_name: "fixture_param_ivp_residual",
        time_arg: "t",
        residuals: &residuals,
        variables: &["y", "z"],
        params: Some(&["a", "b", "c"]),
    };
    let residual_plan = ivp_residual_runtime_plan(&residual_task, ResidualChunkingStrategy::Whole);
    let residual_seq = SequentialResidualExecutor::new(
        &residual_plan,
        bind_parameterized_ivp_residual_fixture_chunks(&residual_plan),
    );

    let jacobian_task = IvpJacobianTask {
        fn_name: "fixture_param_ivp_jacobian",
        time_arg: "t",
        jacobian: &jacobian,
        variables: &["y", "z"],
        params: Some(&["a", "b", "c"]),
    };
    let jacobian_plan =
        ivp_dense_runtime_plan(&jacobian_task, DenseJacobianChunkingStrategy::Whole);
    let dense_bindings = bind_parameterized_ivp_jacobian_fixture_chunks(&jacobian_plan);
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

    let args = vec![1.5, 2.0, -0.5, 3.0, 0.75, -1.25];
    let expected_residual = baseline_parameterized_ivp_residual(args.as_slice());
    let expected_jacobian = baseline_parameterized_ivp_jacobian(args.as_slice());

    assert_eq!(
        DVector::from_vec(residual_seq.eval(args.as_slice())),
        expected_residual
    );
    assert_eq!(
        dense_seq.eval_dense_matrix(args.as_slice()),
        expected_jacobian
    );
    assert_eq!(
        dense_par.eval_dense_matrix(args.as_slice()),
        expected_jacobian
    );
}

#[test]
fn parameterized_ivp_runtime_plan_keeps_time_then_params_then_variables() {
    let (residuals, _) = build_parameterized_ivp_case();
    let task = IvpResidualTask {
        fn_name: "param_ivp_residual_eval",
        time_arg: "t",
        residuals: &residuals,
        variables: &["y", "z"],
        params: Some(&["a", "b", "c"]),
    };

    let runtime_plan = ivp_residual_runtime_plan(&task, ResidualChunkingStrategy::Whole);

    assert_eq!(runtime_plan.input_names, vec!["t", "a", "b", "c", "y", "z"]);
}

#[test]
fn benchmark_compiled_parameterized_ivp_aot_runtime_paths() {
    let (residuals, jacobian) = build_parameterized_ivp_case();

    let residual_task = IvpResidualTask {
        fn_name: "fixture_param_ivp_residual",
        time_arg: "t",
        residuals: &residuals,
        variables: &["y", "z"],
        params: Some(&["a", "b", "c"]),
    };
    let residual_plan = ivp_residual_runtime_plan(
        &residual_task,
        ResidualChunkingStrategy::ByOutputCount {
            max_outputs_per_chunk: 1,
        },
    );
    let residual_bindings = bind_parameterized_ivp_residual_fixture_chunks(&residual_plan);
    let residual_seq = SequentialResidualExecutor::new(&residual_plan, residual_bindings.clone());
    let residual_par = ParallelResidualExecutor::with_config(
        &residual_plan,
        residual_bindings,
        benchmark_parallel_config(),
    );

    let jacobian_task = IvpJacobianTask {
        fn_name: "fixture_param_ivp_jacobian",
        time_arg: "t",
        jacobian: &jacobian,
        variables: &["y", "z"],
        params: Some(&["a", "b", "c"]),
    };
    let jacobian_plan =
        ivp_dense_runtime_plan(&jacobian_task, DenseJacobianChunkingStrategy::Whole);
    let dense_bindings = bind_parameterized_ivp_jacobian_fixture_chunks(&jacobian_plan);
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

    let args = vec![1.5, 2.0, -0.5, 3.0, 0.75, -1.25];
    let expected_residual = baseline_parameterized_ivp_residual(args.as_slice());
    let expected_jacobian = baseline_parameterized_ivp_jacobian(args.as_slice());
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

    println!("=== Compiled parameterized IVP AOT runtime ===");
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
        "param ivp plans: residual_chunks={}, jacobian_chunks={}, jobs_residual={}, jobs_dense={}",
        residual_plan.chunks.len(),
        jacobian_plan.chunks.len(),
        residual_par.job_count(),
        dense_par.job_count()
    );
}
