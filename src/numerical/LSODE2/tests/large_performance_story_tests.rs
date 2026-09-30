//! Release-oriented large-system Lambdify performance stories.
//!
//! These tests are ignored by default because they intentionally run large
//! Sparse/Banded systems repeatedly. They report cold preparation, warm solver
//! stages, total wall-clock and integer trajectory counters. Dense is excluded
//! by construction: it is a correctness control, not a production large-system
//! route.

use super::native_jacobian::{NativeJacobianStorage, try_prepare_native_atomview_jacobian_runtime};
use super::story_support::{
    ChainMatrixRoute, chain_equations, chain_solver_config, chain_state, max_vector_diff,
    prepare_chain_residual,
};
use super::{IvpColdStage, IvpLambdifyExecutionPolicy, IvpTelemetry, IvpWarmStage, Lsode2Solver};
use crate::symbolic::codegen::codegen_orchestrator::{
    machine_min_work_per_parallel_job, rayon_overhead_baseline,
};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp::{
    IvpSymbolicAssemblyBackend, PreparedNativeJacobianMetrics, build_symbolic_jacobian,
};
use nalgebra::DVector;
use std::hint::black_box;
use std::process::{Command, Stdio};
use std::sync::{Arc, RwLock};
use std::time::Instant;

macro_rules! reportln {
    ($($arg:tt)*) => {
        crate::Utils::test_reporting::capture_test_line(format_args!($($arg)*));
    };
}

fn dimensions_from_env(name: &str, default: &[usize]) -> Vec<usize> {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|part| part.trim().parse::<usize>().ok())
                .filter(|dimension| *dimension >= 2)
                .collect::<Vec<_>>()
        })
        .filter(|dimensions| !dimensions.is_empty())
        .unwrap_or_else(|| default.to_vec())
}

fn usize_from_env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn cold_ms(snapshot: &super::IvpTelemetrySnapshot, stage: IvpColdStage) -> f64 {
    snapshot.cold_stage(stage).elapsed.as_secs_f64() * 1.0e3
}

fn warm_ms(snapshot: &super::IvpTelemetrySnapshot, stage: IvpWarmStage) -> f64 {
    snapshot.warm_stage(stage).elapsed.as_secs_f64() * 1.0e3
}

#[derive(Clone, Copy, Debug, Default)]
struct ExprJacobianShape {
    entries: usize,
    total_nodes: usize,
    max_nodes: usize,
    add_nodes: usize,
    mul_nodes: usize,
    div_nodes: usize,
    pow_nodes: usize,
    builtin_nodes: usize,
}

fn inspect_expr_jacobian_shape(jacobian: &[Vec<Expr>]) -> ExprJacobianShape {
    fn visit(expr: &Expr, shape: &mut ExprJacobianShape) -> usize {
        shape.total_nodes += 1;
        let (children, operation) = match expr {
            Expr::Var(_) | Expr::Const(_) => (&[][..], 0usize),
            Expr::Add(left, right) => {
                shape.add_nodes += 1;
                (&[left.as_ref(), right.as_ref()][..], 0)
            }
            Expr::Sub(left, right) => {
                shape.add_nodes += 1;
                (&[left.as_ref(), right.as_ref()][..], 0)
            }
            Expr::Mul(left, right) => {
                shape.mul_nodes += 1;
                (&[left.as_ref(), right.as_ref()][..], 0)
            }
            Expr::Div(left, right) => {
                shape.div_nodes += 1;
                (&[left.as_ref(), right.as_ref()][..], 0)
            }
            Expr::Pow(base, exponent) => {
                shape.pow_nodes += 1;
                (&[base.as_ref(), exponent.as_ref()][..], 0)
            }
            Expr::Exp(arg)
            | Expr::Ln(arg)
            | Expr::sin(arg)
            | Expr::cos(arg)
            | Expr::tg(arg)
            | Expr::ctg(arg)
            | Expr::arcsin(arg)
            | Expr::arccos(arg)
            | Expr::arctg(arg)
            | Expr::arcctg(arg) => {
                shape.builtin_nodes += 1;
                (&[arg.as_ref()][..], 0)
            }
        };
        let child_nodes = children
            .iter()
            .map(|child| visit(child, shape))
            .sum::<usize>();
        let node_count = 1 + child_nodes;
        let _ = operation;
        node_count
    }

    let mut shape = ExprJacobianShape::default();
    for row in jacobian {
        for expr in row {
            if !expr.is_zero() {
                shape.entries += 1;
                let before = shape.total_nodes;
                visit(expr, &mut shape);
                shape.max_nodes = shape.max_nodes.max(shape.total_nodes - before);
            }
        }
    }
    shape
}

fn inspect_native_shape(metrics: PreparedNativeJacobianMetrics) -> ExprJacobianShape {
    ExprJacobianShape {
        entries: metrics.entries,
        total_nodes: metrics.total_nodes,
        max_nodes: metrics.max_nodes,
        add_nodes: metrics.add_nodes,
        mul_nodes: metrics.mul_nodes,
        pow_nodes: metrics.powi_nodes + metrics.pow_nodes,
        ..ExprJacobianShape::default()
    }
}

#[derive(Clone, Debug)]
struct FullSolveRun {
    prepare_ms: f64,
    solve_ms: f64,
    total_ms: f64,
    solver_preparation_ms: f64,
    bridge_preparation_ms: f64,
    native_callback_preparation_ms: f64,
    solve_scope_ms: f64,
    summary_scope_ms: f64,
    final_state: DVector<f64>,
    residual_ms: f64,
    jacobian_ms: f64,
    linear_ms: f64,
    expr_to_atom_ms: f64,
    differentiation_ms: f64,
    simplify_ms: f64,
    sparse_pattern_ms: f64,
    layout_ms: f64,
    residual_lambdify_ms: f64,
    jacobian_lambdify_ms: f64,
    residual_calls: usize,
    jacobian_calls: usize,
    linear_solves: usize,
    accepted: usize,
    rejected: usize,
}

fn run_full_case(
    dimension: usize,
    frontend: IvpSymbolicAssemblyBackend,
    matrix: ChainMatrixRoute,
    policy: IvpLambdifyExecutionPolicy,
) -> FullSolveRun {
    let telemetry = IvpTelemetry::detailed();
    let total_started = Instant::now();
    let mut solver = Lsode2Solver::new(chain_solver_config(
        dimension,
        frontend,
        matrix,
        policy,
        telemetry.clone(),
    ))
    .expect("large performance solver should construct");
    let prepare_started = Instant::now();
    solver
        .prepare()
        .expect("large performance preparation should succeed");
    let prepare_ms = prepare_started.elapsed().as_secs_f64() * 1.0e3;
    let solve_started = Instant::now();
    let summary = solver
        .solve_with_summary()
        .expect("large performance solve should succeed");
    let solve_ms = solve_started.elapsed().as_secs_f64() * 1.0e3;
    let snapshot = telemetry.snapshot();
    let final_state = summary
        .final_y
        .expect("large performance solve should expose final state");
    let evaluations = summary.evaluation_telemetry;
    FullSolveRun {
        prepare_ms,
        solve_ms,
        total_ms: total_started.elapsed().as_secs_f64() * 1.0e3,
        solver_preparation_ms: cold_ms(&snapshot, IvpColdStage::SolverPreparation),
        bridge_preparation_ms: cold_ms(&snapshot, IvpColdStage::BridgePreparation),
        native_callback_preparation_ms: cold_ms(&snapshot, IvpColdStage::NativeCallbackPreparation),
        solve_scope_ms: warm_ms(&snapshot, IvpWarmStage::Solve),
        summary_scope_ms: warm_ms(&snapshot, IvpWarmStage::Summary),
        final_state,
        residual_ms: evaluations.residual_ms_total,
        jacobian_ms: evaluations.jacobian_ms_total,
        linear_ms: evaluations.linear_ms_total,
        expr_to_atom_ms: cold_ms(&snapshot, IvpColdStage::ExprToAtom),
        differentiation_ms: cold_ms(&snapshot, IvpColdStage::SymbolicDifferentiation),
        simplify_ms: cold_ms(&snapshot, IvpColdStage::Simplification),
        sparse_pattern_ms: cold_ms(&snapshot, IvpColdStage::SparsePattern),
        layout_ms: cold_ms(&snapshot, IvpColdStage::LayoutPlanning),
        residual_lambdify_ms: cold_ms(&snapshot, IvpColdStage::ResidualLambdification),
        jacobian_lambdify_ms: cold_ms(&snapshot, IvpColdStage::JacobianLambdification),
        residual_calls: evaluations.residual_evaluations,
        jacobian_calls: evaluations.jacobian_evaluations,
        linear_solves: evaluations.linear_solves,
        accepted: evaluations.accepted_steps,
        rejected: evaluations.rejected_steps,
    }
}

#[test]
#[ignore = "release Jacobian shape diagnostic; run explicitly while investigating native evaluator overhead"]
fn lsode2_large_jacobian_shape_diagnostic_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::large_performance_story_tests::lsode2_large_jacobian_shape_diagnostic_story",
    );
    let dimensions = dimensions_from_env("LSODE2_JACOBIAN_SHAPE_DIMENSIONS", &[128, 512]);
    reportln!(
        "[LSODE2 Jacobian shape diagnostic] dimensions={dimensions:?}; ExprLegacy is diff+simplify; AtomViewNative is direct Atom derivative; callback timing excluded"
    );
    reportln!(
        "frontend | dimension | entries | total_nodes | avg_nodes | max_nodes | add | mul | div | pow | builtin | interpretation"
    );

    for dimension in dimensions {
        let equations = chain_equations(dimension);
        let variables = (0..dimension)
            .map(|index| format!("y{index}"))
            .collect::<Vec<_>>();
        let expr_jacobian = build_symbolic_jacobian(
            &equations,
            &variables,
            IvpSymbolicAssemblyBackend::ExprLegacy,
            &IvpTelemetry::disabled(),
        );
        let expr = inspect_expr_jacobian_shape(&expr_jacobian);

        let native = try_prepare_native_atomview_jacobian_runtime(
            &equations,
            &variables,
            "t",
            Some(&[
                "k".to_string(),
                "d".to_string(),
                "q".to_string(),
                "nl".to_string(),
            ]),
            Some(Arc::new(RwLock::new(DVector::from_vec(vec![
                20.0, 4.0, 0.20, 0.010,
            ])))),
            NativeJacobianStorage::SparseTriplets,
            IvpTelemetry::disabled(),
            IvpLambdifyExecutionPolicy::Sequential,
        )
        .expect("native Jacobian shape diagnostic should prepare");
        let native = inspect_native_shape(native.plan_metrics());

        assert_eq!(expr.entries, native.entries);
        reportln!(
            "ExprLegacy | {dimension} | {} | {} | {:.2} | {} | {} | {} | {} | {} | {} | simplified Expr closure shape",
            expr.entries,
            expr.total_nodes,
            expr.total_nodes as f64 / expr.entries as f64,
            expr.max_nodes,
            expr.add_nodes,
            expr.mul_nodes,
            expr.div_nodes,
            expr.pow_nodes,
            expr.builtin_nodes,
        );
        reportln!(
            "AtomViewNative | {dimension} | {} | {} | {:.2} | {} | {} | {} | {} | {} | {} | direct prepared Atom evaluator shape",
            native.entries,
            native.total_nodes,
            native.total_nodes as f64 / native.entries as f64,
            native.max_nodes,
            native.add_nodes,
            native.mul_nodes,
            native.div_nodes,
            native.pow_nodes,
            native.builtin_nodes,
        );
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn stddev(values: &[f64], average: f64) -> f64 {
    let variance = values
        .iter()
        .map(|value| (value - average).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt()
}

#[test]
#[ignore = "release large-system stage baseline; run explicitly with --ignored"]
fn lsode2_large_system_sparse_banded_total_and_stage_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::large_performance_story_tests::lsode2_large_system_sparse_banded_total_and_stage_story",
    );
    let dimensions = dimensions_from_env("LSODE2_LARGE_STAGE_DIMENSIONS", &[128, 256, 512]);
    let repetitions = usize_from_env("LSODE2_LARGE_STAGE_REPETITIONS", 3);
    reportln!(
        "[LSODE2 large stage baseline] dimensions={dimensions:?}; repetitions={repetitions}; Dense excluded; BDF controller fixed; release run required"
    );
    reportln!(
        "[LSODE2 lifecycle scopes] preparation and solve scopes are inclusive; do not sum them with nested symbolic/callback stages"
    );
    reportln!(
        "frontend | matrix | dimension | prepare_ms mean+/-std | solve_ms mean+/-std | total_ms mean+/-std | solver_preparation_ms | bridge_preparation_ms | native_callback_preparation_ms | solve_scope_ms | summary_scope_ms | cold_expr_to_atom_ms | cold_diff_ms | cold_simplify_ms | cold_pattern_ms | cold_layout_ms | cold_res_lambdify_ms | cold_jac_lambdify_ms | warm_residual_ms | warm_jacobian_ms | warm_linear_ms | residual_calls | jacobian_calls | linear_solves | accepted | rejected | max_diff_vs_expr"
    );

    for dimension in dimensions {
        for matrix in [ChainMatrixRoute::Sparse, ChainMatrixRoute::Banded] {
            let mut reference_state = None;
            for frontend in [
                IvpSymbolicAssemblyBackend::ExprLegacy,
                IvpSymbolicAssemblyBackend::AtomView,
            ] {
                let runs = (0..repetitions)
                    .map(|_| {
                        run_full_case(
                            dimension,
                            frontend,
                            matrix,
                            IvpLambdifyExecutionPolicy::Sequential,
                        )
                    })
                    .collect::<Vec<_>>();
                let first = &runs[0];
                let max_diff = reference_state
                    .as_ref()
                    .map(|reference: &DVector<f64>| max_vector_diff(reference, &first.final_state))
                    .unwrap_or(0.0);
                if reference_state.is_none() {
                    reference_state = Some(first.final_state.clone());
                }
                assert!(
                    max_diff <= 1.0e-8,
                    "frontend drift at dimension {dimension}"
                );
                for run in &runs[1..] {
                    assert!(
                        max_vector_diff(&first.final_state, &run.final_state) <= 1.0e-8,
                        "repeat trajectory drift at dimension {dimension}"
                    );
                    assert_eq!(run.residual_calls, first.residual_calls);
                    assert_eq!(run.jacobian_calls, first.jacobian_calls);
                    assert_eq!(run.linear_solves, first.linear_solves);
                    assert_eq!(run.accepted, first.accepted);
                    assert_eq!(run.rejected, first.rejected);
                }
                let prepare = runs.iter().map(|run| run.prepare_ms).collect::<Vec<_>>();
                let solve = runs.iter().map(|run| run.solve_ms).collect::<Vec<_>>();
                let total = runs.iter().map(|run| run.total_ms).collect::<Vec<_>>();
                let run = first;
                reportln!(
                    "{} | {} | {} | {:.3}+/-{:.3} | {:.3}+/-{:.3} | {:.3}+/-{:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {} | {} | {} | {} | {:.3e}",
                    if matches!(frontend, IvpSymbolicAssemblyBackend::ExprLegacy) {
                        "ExprLegacy"
                    } else {
                        "AtomViewNative"
                    },
                    matrix.label(),
                    dimension,
                    mean(&prepare),
                    stddev(&prepare, mean(&prepare)),
                    mean(&solve),
                    stddev(&solve, mean(&solve)),
                    mean(&total),
                    stddev(&total, mean(&total)),
                    run.solver_preparation_ms,
                    run.bridge_preparation_ms,
                    run.native_callback_preparation_ms,
                    run.solve_scope_ms,
                    run.summary_scope_ms,
                    run.expr_to_atom_ms,
                    run.differentiation_ms,
                    run.simplify_ms,
                    run.sparse_pattern_ms,
                    run.layout_ms,
                    run.residual_lambdify_ms,
                    run.jacobian_lambdify_ms,
                    run.residual_ms,
                    run.jacobian_ms,
                    run.linear_ms,
                    run.residual_calls,
                    run.jacobian_calls,
                    run.linear_solves,
                    run.accepted,
                    run.rejected,
                    max_diff,
                );
            }
        }
    }
}

#[derive(Debug)]
struct CallbackPolicyRun {
    policy: IvpLambdifyExecutionPolicy,
    residual_ms: Vec<f64>,
    jacobian_ms: Vec<f64>,
    residual_reference: Vec<f64>,
    jacobian_reference: Vec<f64>,
    parallel_dispatches: u64,
    sequential_dispatches: u64,
    worker_count: usize,
    residual_calls: u64,
    jacobian_calls: u64,
    auto_min_work_per_job: usize,
    rayon_join2_ns: f64,
    rayon_join4_ns: f64,
    calibrated_min_work_per_job: usize,
}

fn run_callback_policy_case(
    dimension: usize,
    matrix: ChainMatrixRoute,
    policy: IvpLambdifyExecutionPolicy,
    checkpoints: &[usize],
) -> CallbackPolicyRun {
    let telemetry = IvpTelemetry::detailed();
    let problem = prepare_chain_residual(
        dimension,
        IvpSymbolicAssemblyBackend::AtomView,
        policy,
        telemetry.clone(),
    );
    let variables = (0..dimension)
        .map(|index| format!("y{index}"))
        .collect::<Vec<_>>();
    let parameter_handle = Arc::new(RwLock::new(DVector::from_vec(vec![20.0, 4.0, 0.20, 0.010])));
    let mut jacobian = try_prepare_native_atomview_jacobian_runtime(
        &chain_equations(dimension),
        &variables,
        "t",
        Some(&[
            "k".to_string(),
            "d".to_string(),
            "q".to_string(),
            "nl".to_string(),
        ]),
        Some(parameter_handle),
        match matrix {
            ChainMatrixRoute::Sparse => NativeJacobianStorage::SparseTriplets,
            ChainMatrixRoute::Banded => NativeJacobianStorage::Banded { bandwidth: None },
        },
        telemetry.clone(),
        policy,
    )
    .expect("large callback Jacobian runtime should prepare");
    let state = chain_state(dimension);
    let max_checkpoint = *checkpoints.last().expect("at least one checkpoint");
    let mut residual_ms = Vec::with_capacity(checkpoints.len());
    let mut jacobian_ms = Vec::with_capacity(checkpoints.len());
    let mut residual_reference = Vec::new();
    let mut jacobian_reference = Vec::new();
    let mut sparse_values = vec![0.0; jacobian.sparse_pattern().len()];
    let (_, _, banded_len) = jacobian.banded_layout().unwrap_or((0, 0, 0));
    let mut banded_values = vec![0.0; banded_len];
    let started = Instant::now();
    let mut next_checkpoint = 0;
    for repetition in 1..=max_checkpoint {
        let residual = problem
            .try_evaluate_residual(0.5 + repetition as f64 * 1.0e-3, &state)
            .expect("large callback residual should evaluate");
        if repetition == 1 {
            residual_reference = residual.iter().copied().collect();
        }
        let residual_elapsed = started.elapsed().as_secs_f64() * 1.0e3;
        if matrix_matches_checkpoint(repetition, checkpoints, &mut next_checkpoint) {
            residual_ms.push(residual_elapsed);
        }
    }
    let residual_total = started.elapsed().as_secs_f64() * 1.0e3;
    let jacobian_started = Instant::now();
    next_checkpoint = 0;
    for repetition in 1..=max_checkpoint {
        let time = 0.5 + repetition as f64 * 1.0e-3;
        match matrix {
            ChainMatrixRoute::Sparse => jacobian
                .try_evaluate_sparse_values_into(time, &state, &mut sparse_values)
                .expect("large sparse callback should evaluate"),
            ChainMatrixRoute::Banded => jacobian
                .try_evaluate_banded_values_into(time, &state, &mut banded_values)
                .expect("large banded callback should evaluate"),
        }
        if repetition == 1 {
            jacobian_reference = match matrix {
                ChainMatrixRoute::Sparse => sparse_values.clone(),
                ChainMatrixRoute::Banded => banded_values.clone(),
            };
        }
        let elapsed = jacobian_started.elapsed().as_secs_f64() * 1.0e3;
        if matrix_matches_checkpoint(repetition, checkpoints, &mut next_checkpoint) {
            jacobian_ms.push(elapsed);
        }
    }
    black_box((residual_total, &sparse_values, &banded_values));
    let snapshot = telemetry.snapshot();
    let overhead = rayon_overhead_baseline();
    CallbackPolicyRun {
        policy,
        residual_ms,
        jacobian_ms,
        residual_reference,
        jacobian_reference,
        parallel_dispatches: snapshot.parallel_dispatches,
        sequential_dispatches: snapshot.sequential_dispatches,
        worker_count: snapshot.lambdify_worker_count,
        residual_calls: snapshot.residual_evaluations,
        jacobian_calls: snapshot.jacobian_evaluations,
        auto_min_work_per_job: snapshot.lambdify_auto_min_work_per_job,
        rayon_join2_ns: overhead.join2_ns,
        rayon_join4_ns: overhead.join4_ns,
        calibrated_min_work_per_job: machine_min_work_per_parallel_job(),
    }
}

fn matrix_matches_checkpoint(
    repetition: usize,
    checkpoints: &[usize],
    next_checkpoint: &mut usize,
) -> bool {
    if *next_checkpoint < checkpoints.len() && repetition == checkpoints[*next_checkpoint] {
        *next_checkpoint += 1;
        true
    } else {
        false
    }
}

fn max_slice_diff(left: &[f64], right: &[f64]) -> f64 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max)
}

#[test]
#[ignore = "release Auto break-even matrix; run explicitly with --ignored"]
fn lsode2_large_auto_break_even_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::large_performance_story_tests::lsode2_large_auto_break_even_story",
    );
    let dimensions = dimensions_from_env("LSODE2_AUTO_DIMENSIONS", &[128, 256, 512, 1024]);
    let checkpoints = [1usize, 4, 16, 64];
    let min_work = usize_from_env("LSODE2_AUTO_MIN_WORK", 64);
    let overhead = rayon_overhead_baseline();
    reportln!(
        "[LSODE2 large Auto break-even] dimensions={dimensions:?}; checkpoints={checkpoints:?}; min_work={min_work}; AtomViewNative; Sparse/Banded; preparation excluded; calibration_workers={}; rayon_join2_ns={:.3}; rayon_join4_ns={:.3}; calibrated_min_work_per_job={}",
        overhead.workers,
        overhead.join2_ns,
        overhead.join4_ns,
        machine_min_work_per_parallel_job(),
    );
    reportln!(
        "matrix | dimension | policy | residual_ms@1 | residual_ms@4 | residual_ms@16 | residual_ms@64 | jacobian_ms@1 | jacobian_ms@4 | jacobian_ms@16 | jacobian_ms@64 | parallel_dispatches | sequential_dispatches | worker_count | auto_min_work_per_job | calibrated_min_work_per_job | rayon_join2_ns | rayon_join4_ns | residual_calls | jacobian_calls | residual_diff | jacobian_diff"
    );
    reportln!(
        "--- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:"
    );

    for dimension in dimensions {
        for matrix in [ChainMatrixRoute::Sparse, ChainMatrixRoute::Banded] {
            let runs = [
                run_callback_policy_case(
                    dimension,
                    matrix,
                    IvpLambdifyExecutionPolicy::Sequential,
                    &checkpoints,
                ),
                run_callback_policy_case(
                    dimension,
                    matrix,
                    IvpLambdifyExecutionPolicy::Parallel { min_work: 1 },
                    &checkpoints,
                ),
                run_callback_policy_case(
                    dimension,
                    matrix,
                    IvpLambdifyExecutionPolicy::Auto { min_work },
                    &checkpoints,
                ),
            ];
            let sequential = &runs[0];
            let auto = &runs[2];
            for run in &runs[1..] {
                assert!(
                    max_slice_diff(&sequential.residual_reference, &run.residual_reference)
                        <= 1.0e-10
                );
                assert!(
                    max_slice_diff(&sequential.jacobian_reference, &run.jacobian_reference)
                        <= 1.0e-10
                );
            }
            let auto_break_even = auto_crossover_label(sequential, auto, &checkpoints);
            let portable_break_even =
                portable_auto_crossover_label(sequential, auto, &checkpoints, dimension);
            reportln!(
                "[LSODE2 Auto crossover] matrix={} dimension={} raw_first_both_stage_crossover={} portable_stable_crossover={}",
                matrix.label(),
                dimension,
                auto_break_even,
                portable_break_even
            );
            for run in &runs {
                let residual_diff =
                    max_slice_diff(&sequential.residual_reference, &run.residual_reference);
                let jacobian_diff =
                    max_slice_diff(&sequential.jacobian_reference, &run.jacobian_reference);
                reportln!(
                    "{} | {} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {} | {} | {} | {} | {:.3} | {:.3} | {} | {} | {:.3e} | {:.3e}",
                    matrix.label(),
                    dimension,
                    run.policy.label(),
                    run.residual_ms[0],
                    run.residual_ms[1],
                    run.residual_ms[2],
                    run.residual_ms[3],
                    run.jacobian_ms[0],
                    run.jacobian_ms[1],
                    run.jacobian_ms[2],
                    run.jacobian_ms[3],
                    run.parallel_dispatches,
                    run.sequential_dispatches,
                    run.worker_count,
                    run.auto_min_work_per_job,
                    run.calibrated_min_work_per_job,
                    run.rayon_join2_ns,
                    run.rayon_join4_ns,
                    run.residual_calls,
                    run.jacobian_calls,
                    residual_diff,
                    jacobian_diff,
                );
                assert_eq!(run.residual_calls, *checkpoints.last().unwrap() as u64);
                assert_eq!(run.jacobian_calls, *checkpoints.last().unwrap() as u64);
            }
        }
    }
}

const AUTO_WORKER_CHILD_TEST: &str =
    "numerical::LSODE2::large_performance_story_tests::lsode2_large_auto_break_even_worker_child";

fn auto_crossover_label(
    sequential: &CallbackPolicyRun,
    auto: &CallbackPolicyRun,
    checkpoints: &[usize],
) -> String {
    if auto.parallel_dispatches == 0 {
        return "none (Auto remained sequential)".to_string();
    }
    checkpoints
        .iter()
        .enumerate()
        .find(|(index, _)| {
            auto.residual_ms[*index] <= sequential.residual_ms[*index]
                && auto.jacobian_ms[*index] <= sequential.jacobian_ms[*index]
        })
        .map(|(_, checkpoint)| checkpoint.to_string())
        .unwrap_or_else(|| "none".to_string())
}

fn portable_auto_crossover_label(
    sequential: &CallbackPolicyRun,
    auto: &CallbackPolicyRun,
    checkpoints: &[usize],
    dimension: usize,
) -> String {
    if auto.parallel_dispatches == 0 {
        return "none (Auto remained sequential)".to_string();
    }
    let minimum_work = auto
        .calibrated_min_work_per_job
        .saturating_mul(auto.worker_count.max(2));
    checkpoints
        .iter()
        .enumerate()
        .find(|(index, checkpoint)| {
            let work = dimension.saturating_mul(**checkpoint);
            work >= minimum_work
                && auto.residual_ms[*index] <= sequential.residual_ms[*index]
                && auto.jacobian_ms[*index] <= sequential.jacobian_ms[*index]
                && (0..checkpoints.len()).skip(*index).all(|later| {
                    auto.residual_ms[later] <= sequential.residual_ms[later]
                        && auto.jacobian_ms[later] <= sequential.jacobian_ms[later]
                })
        })
        .map(|(_, checkpoint)| checkpoint.to_string())
        .unwrap_or_else(|| "none".to_string())
}

fn worker_counts_from_env() -> Vec<usize> {
    std::env::var("LSODE2_AUTO_WORKER_COUNTS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|part| part.trim().parse::<usize>().ok())
                .filter(|count| *count > 0)
                .collect::<Vec<_>>()
        })
        .filter(|counts| !counts.is_empty())
        .unwrap_or_else(|| vec![1, 2, 4])
}

fn run_auto_worker_child(
    worker_count: usize,
    dimensions: &[usize],
    checkpoints: &[usize],
    min_work: usize,
) {
    for &dimension in dimensions {
        for matrix in [ChainMatrixRoute::Sparse, ChainMatrixRoute::Banded] {
            let runs = [
                run_callback_policy_case(
                    dimension,
                    matrix,
                    IvpLambdifyExecutionPolicy::Sequential,
                    checkpoints,
                ),
                run_callback_policy_case(
                    dimension,
                    matrix,
                    IvpLambdifyExecutionPolicy::Parallel { min_work: 1 },
                    checkpoints,
                ),
                run_callback_policy_case(
                    dimension,
                    matrix,
                    IvpLambdifyExecutionPolicy::Auto { min_work },
                    checkpoints,
                ),
            ];
            let sequential = &runs[0];
            let auto = &runs[2];
            assert_eq!(
                auto.worker_count, worker_count,
                "telemetry must report the worker count selected for the child process"
            );
            for run in &runs[1..] {
                assert!(
                    max_slice_diff(&sequential.residual_reference, &run.residual_reference)
                        <= 1.0e-10
                );
                assert!(
                    max_slice_diff(&sequential.jacobian_reference, &run.jacobian_reference)
                        <= 1.0e-10
                );
                assert_eq!(
                    run.residual_calls,
                    checkpoints.last().copied().unwrap() as u64
                );
                assert_eq!(
                    run.jacobian_calls,
                    checkpoints.last().copied().unwrap() as u64
                );
            }
            let residual_diff =
                max_slice_diff(&sequential.residual_reference, &auto.residual_reference);
            let jacobian_diff =
                max_slice_diff(&sequential.jacobian_reference, &auto.jacobian_reference);
            println!(
                "RST_AUTO_WORKER|workers={worker_count}|matrix={}|dimension={dimension}|auto_crossover={}|portable_crossover={}|sequential_residual_ms={:.6},{:.6},{:.6},{:.6}|auto_residual_ms={:.6},{:.6},{:.6},{:.6}|sequential_jacobian_ms={:.6},{:.6},{:.6},{:.6}|auto_jacobian_ms={:.6},{:.6},{:.6},{:.6}|parallel_dispatches={}|sequential_dispatches={}|worker_count_observed={}|auto_min_work_per_job={}|calibrated_min_work_per_job={}|rayon_join2_ns={:.3}|rayon_join4_ns={:.3}|residual_calls={}|jacobian_calls={}|residual_diff={:.3e}|jacobian_diff={:.3e}",
                matrix.label(),
                auto_crossover_label(sequential, auto, checkpoints),
                portable_auto_crossover_label(sequential, auto, checkpoints, dimension),
                sequential.residual_ms[0],
                sequential.residual_ms[1],
                sequential.residual_ms[2],
                sequential.residual_ms[3],
                auto.residual_ms[0],
                auto.residual_ms[1],
                auto.residual_ms[2],
                auto.residual_ms[3],
                sequential.jacobian_ms[0],
                sequential.jacobian_ms[1],
                sequential.jacobian_ms[2],
                sequential.jacobian_ms[3],
                auto.jacobian_ms[0],
                auto.jacobian_ms[1],
                auto.jacobian_ms[2],
                auto.jacobian_ms[3],
                auto.parallel_dispatches,
                auto.sequential_dispatches,
                auto.worker_count,
                auto.auto_min_work_per_job,
                auto.calibrated_min_work_per_job,
                auto.rayon_join2_ns,
                auto.rayon_join4_ns,
                auto.residual_calls,
                auto.jacobian_calls,
                residual_diff,
                jacobian_diff,
            );
        }
    }
}

#[test]
#[ignore = "child process for the multi-worker Auto break-even story"]
fn lsode2_large_auto_break_even_worker_child() {
    if std::env::var_os("LSODE2_AUTO_WORKER_CHILD").is_none() {
        return;
    }
    let worker_count = usize_from_env("LSODE2_AUTO_WORKER_COUNT", 1);
    rayon::ThreadPoolBuilder::new()
        .num_threads(worker_count)
        .build_global()
        .expect("worker child must initialize its requested Rayon pool before callbacks");
    let dimensions = dimensions_from_env("LSODE2_AUTO_DIMENSIONS", &[256, 512]);
    let checkpoints = [1usize, 4, 16, 64];
    let min_work = usize_from_env("LSODE2_AUTO_MIN_WORK", 64);
    run_auto_worker_child(worker_count, &dimensions, &checkpoints, min_work);
}

#[test]
#[ignore = "release multi-worker Auto/Parallel break-even matrix; run explicitly with --ignored"]
fn lsode2_large_auto_break_even_multi_worker_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::large_performance_story_tests::lsode2_large_auto_break_even_multi_worker_story",
    );
    let workers = worker_counts_from_env();
    let dimensions = dimensions_from_env("LSODE2_AUTO_DIMENSIONS", &[256, 512]);
    let min_work = usize_from_env("LSODE2_AUTO_MIN_WORK", 64);
    reportln!(
        "[LSODE2 multi-worker Auto break-even] workers={workers:?}; dimensions={dimensions:?}; checkpoints=[1,4,16,64]; min_work={min_work}; each worker count is a fresh child process"
    );
    reportln!(
        "workers | matrix | dimension | raw_auto_crossover | portable_auto_crossover | sequential_residual_ms@1,@4,@16,@64 | auto_residual_ms@1,@4,@16,@64 | sequential_jacobian_ms@1,@4,@16,@64 | auto_jacobian_ms@1,@4,@16,@64 | parallel_dispatches | sequential_dispatches | observed_workers | auto_min_work | calibrated_min_work | rayon_join2_ns | rayon_join4_ns | residual_calls | jacobian_calls | residual_diff | jacobian_diff"
    );

    for worker_count in workers {
        let executable = std::env::current_exe().expect("test executable should be available");
        let output = Command::new(executable)
            .arg("--exact")
            .arg(AUTO_WORKER_CHILD_TEST)
            .arg("--ignored")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .env("LSODE2_AUTO_WORKER_CHILD", "1")
            .env("LSODE2_AUTO_WORKER_COUNT", worker_count.to_string())
            .env(
                "LSODE2_AUTO_DIMENSIONS",
                dimensions
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            )
            .env("LSODE2_AUTO_MIN_WORK", min_work.to_string())
            .env("RAYON_NUM_THREADS", worker_count.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .expect("multi-worker Auto child should start");
        assert!(
            output.status.success(),
            "multi-worker Auto child failed for workers={worker_count}: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let mut rows = 0usize;
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            if let Some(index) = line.find("RST_AUTO_WORKER|") {
                let row = &line[index + "RST_AUTO_WORKER|".len()..];
                let fields = row
                    .split('|')
                    .filter_map(|field| field.split_once('='))
                    .collect::<std::collections::BTreeMap<_, _>>();
                let field = |name: &str| fields.get(name).copied().unwrap_or("missing");
                assert_eq!(field("workers").parse::<usize>().ok(), Some(worker_count));
                reportln!(
                    "{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}",
                    field("workers"),
                    field("matrix"),
                    field("dimension"),
                    field("auto_crossover"),
                    field("portable_crossover"),
                    field("sequential_residual_ms"),
                    field("auto_residual_ms"),
                    field("sequential_jacobian_ms"),
                    field("auto_jacobian_ms"),
                    field("parallel_dispatches"),
                    field("sequential_dispatches"),
                    field("worker_count_observed"),
                    field("auto_min_work_per_job"),
                    field("calibrated_min_work_per_job"),
                    field("rayon_join2_ns"),
                    field("rayon_join4_ns"),
                    field("residual_calls"),
                    field("jacobian_calls"),
                    field("residual_diff"),
                    field("jacobian_diff"),
                );
                rows += 1;
            }
        }
        assert_eq!(
            rows,
            dimensions.len() * 2,
            "worker child must return one row per dimension and matrix"
        );
    }
}
