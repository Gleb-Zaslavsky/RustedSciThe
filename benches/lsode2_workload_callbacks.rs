//! Cross-workload LSODE2 callback benchmark.
//!
//! This is intentionally callback-only. Solver policy and AOT process
//! lifecycle have separate benches; keeping this corpus focused makes a
//! regression in symbolic preparation or callback evaluation easy to localize.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nalgebra::DVector;
use std::hint::black_box;

#[path = "support/lsode2_bench_support.rs"]
mod lsode2_bench_support;

use RustedSciThe::numerical::LSODE2::workload_fixtures::{
    SymbolicWorkload, WorkloadKind, build_workload,
};
use RustedSciThe::symbolic::symbolic_ivp::{
    IvpSymbolicAssemblyBackend, PreparedSymbolicIvpProblem, SymbolicIvpProblemOptions,
    prepare_symbolic_ivp_problem,
};

const DEFAULT_DIFFUSION_DIMENSIONS: &[usize] = &[64, 256, 512];

fn dimensions(kind: WorkloadKind) -> Vec<usize> {
    match kind {
        WorkloadKind::DiffusionChain => lsode2_bench_support::dimensions_from_env(
            "LSODE2_BENCH_CALLBACK_DIFFUSION_DIMENSIONS",
            DEFAULT_DIFFUSION_DIMENSIONS,
        ),
        WorkloadKind::CombustionLike | WorkloadKind::Robertson => vec![3],
        WorkloadKind::StiffScalar => vec![1],
        WorkloadKind::ThreeBody => vec![12],
    }
}

fn prepare(
    workload: SymbolicWorkload,
    backend: IvpSymbolicAssemblyBackend,
) -> PreparedSymbolicIvpProblem {
    let mut options = SymbolicIvpProblemOptions::new().with_symbolic_assembly_backend(backend);
    if workload.is_parameterized() {
        options = options
            .with_equation_parameters(workload.parameter_names.clone())
            .with_equation_parameter_values(workload.parameter_values.clone());
    }
    prepare_symbolic_ivp_problem(
        workload.equations,
        workload.variables,
        workload.time_variable,
        options,
    )
    .expect("shared workload preparation should succeed")
}

fn assert_callback_parity(
    expr: &PreparedSymbolicIvpProblem,
    atom: &PreparedSymbolicIvpProblem,
    state: &DVector<f64>,
) {
    let mut expr_residual = DVector::zeros(state.len());
    let mut atom_residual = DVector::zeros(state.len());
    expr.try_evaluate_residual_into(0.25, state, &mut expr_residual)
        .expect("ExprLegacy parity residual should succeed");
    atom.try_evaluate_residual_into(0.25, state, &mut atom_residual)
        .expect("AtomViewNative parity residual should succeed");
    let residual_diff = expr_residual
        .iter()
        .zip(atom_residual.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max);
    assert!(
        residual_diff <= 1.0e-10,
        "workload residual parity failed: diff={residual_diff:e}"
    );

    let expr_jacobian = expr
        .try_evaluate_jacobian(0.25, state)
        .expect("ExprLegacy parity Jacobian should succeed");
    let atom_jacobian = atom
        .try_evaluate_jacobian(0.25, state)
        .expect("AtomViewNative parity Jacobian should succeed");
    assert_eq!(expr_jacobian.shape(), atom_jacobian.shape());
    let jacobian_diff = expr_jacobian
        .iter()
        .zip(atom_jacobian.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max);
    assert!(
        jacobian_diff <= 1.0e-10,
        "workload Jacobian parity failed: diff={jacobian_diff:e}"
    );
}

fn benchmark_callback_matrix(c: &mut Criterion) {
    let workloads = lsode2_bench_support::workloads_from_env(
        "LSODE2_BENCH_CALLBACK_WORKLOADS",
        &WorkloadKind::ALL,
    );
    let metadata = format!(
        "routes=ExprLegacy,AtomViewNative; groups=callback-matrix,parameter-continuation; workloads={workloads:?}"
    );
    lsode2_bench_support::print_metadata(
        "lsode2_workload_callbacks",
        &lsode2_bench_support::dimensions_from_env(
            "LSODE2_BENCH_CALLBACK_DIFFUSION_DIMENSIONS",
            DEFAULT_DIFFUSION_DIMENSIONS,
        ),
        &metadata,
    );
    let mut group = c.benchmark_group("lsode2_workload_callbacks");
    group.sample_size(lsode2_bench_support::sample_size());
    if let Some(measurement_time) = lsode2_bench_support::measurement_time() {
        group.measurement_time(measurement_time);
    }

    for kind in workloads.iter().copied() {
        for dimension in dimensions(kind) {
            let expr_workload = build_workload(kind, dimension);
            let atom_workload = expr_workload.clone();
            let expr = prepare(expr_workload, IvpSymbolicAssemblyBackend::ExprLegacy);
            let atom = prepare(atom_workload, IvpSymbolicAssemblyBackend::AtomView);
            let expr_state = build_workload(kind, dimension).initial_state;
            let atom_state = expr_state.clone();
            assert_callback_parity(&expr, &atom, &expr_state);
            let mut expr_residual = DVector::zeros(expr_state.len());
            let mut atom_residual = DVector::zeros(atom_state.len());

            group.bench_with_input(
                BenchmarkId::new(format!("{}/expr-legacy/residual", kind.label()), dimension),
                &dimension,
                |bencher, _| {
                    bencher.iter(|| {
                        expr.try_evaluate_residual_into(
                            black_box(0.25),
                            black_box(&expr_state),
                            &mut expr_residual,
                        )
                        .expect("ExprLegacy residual callback should succeed");
                        black_box(expr_residual.as_slice());
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("{}/atom-native/residual", kind.label()), dimension),
                &dimension,
                |bencher, _| {
                    bencher.iter(|| {
                        atom.try_evaluate_residual_into(
                            black_box(0.25),
                            black_box(&atom_state),
                            &mut atom_residual,
                        )
                        .expect("AtomViewNative residual callback should succeed");
                        black_box(atom_residual.as_slice());
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("{}/expr-legacy/jacobian", kind.label()), dimension),
                &dimension,
                |bencher, _| {
                    bencher.iter(|| {
                        let jacobian = expr
                            .try_evaluate_jacobian(black_box(0.25), black_box(&expr_state))
                            .expect("ExprLegacy Jacobian callback should succeed");
                        black_box(jacobian.as_slice());
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("{}/atom-native/jacobian", kind.label()), dimension),
                &dimension,
                |bencher, _| {
                    bencher.iter(|| {
                        let jacobian = atom
                            .try_evaluate_jacobian(black_box(0.25), black_box(&atom_state))
                            .expect("AtomViewNative Jacobian callback should succeed");
                        black_box(jacobian.as_slice());
                    });
                },
            );
        }
    }

    group.finish();
}

fn benchmark_parameter_continuation(c: &mut Criterion) {
    let workloads = lsode2_bench_support::workloads_from_env(
        "LSODE2_BENCH_CALLBACK_WORKLOADS",
        &[
            WorkloadKind::DiffusionChain,
            WorkloadKind::CombustionLike,
            WorkloadKind::ThreeBody,
        ],
    );
    let mut group = c.benchmark_group("lsode2_workload_parameter_continuation");
    group.sample_size(lsode2_bench_support::sample_size());
    if let Some(measurement_time) = lsode2_bench_support::measurement_time() {
        group.measurement_time(measurement_time);
    }
    for kind in workloads {
        let dimension = match kind {
            WorkloadKind::DiffusionChain => 256,
            WorkloadKind::CombustionLike | WorkloadKind::Robertson => 3,
            WorkloadKind::StiffScalar => 1,
            WorkloadKind::ThreeBody => 12,
        };
        let workload = build_workload(kind, dimension);
        let state = workload.initial_state.clone();
        let next_parameters = workload
            .parameter_values
            .iter()
            .map(|value| value * 1.01)
            .collect::<Vec<_>>();
        let next_parameters = DVector::from_vec(next_parameters);
        let expr = prepare(workload.clone(), IvpSymbolicAssemblyBackend::ExprLegacy);
        let atom = prepare(workload, IvpSymbolicAssemblyBackend::AtomView);
        let mut expr_output = DVector::zeros(state.len());
        let mut atom_output = DVector::zeros(state.len());

        group.bench_function(
            BenchmarkId::new(format!("{}/expr-legacy", kind.label()), dimension),
            |bencher| {
                bencher.iter(|| {
                    expr.set_parameter_values(next_parameters.clone())
                        .expect("ExprLegacy parameter rebind should succeed");
                    expr.try_evaluate_residual_into(
                        black_box(0.25),
                        black_box(&state),
                        &mut expr_output,
                    )
                    .expect("ExprLegacy continuation callback should succeed");
                    black_box(expr_output.as_slice());
                });
            },
        );
        group.bench_function(
            BenchmarkId::new(format!("{}/atom-native", kind.label()), dimension),
            |bencher| {
                bencher.iter(|| {
                    atom.set_parameter_values(next_parameters.clone())
                        .expect("AtomViewNative parameter rebind should succeed");
                    atom.try_evaluate_residual_into(
                        black_box(0.25),
                        black_box(&state),
                        &mut atom_output,
                    )
                    .expect("AtomViewNative continuation callback should succeed");
                    black_box(atom_output.as_slice());
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_callback_matrix,
    benchmark_parameter_continuation
);
criterion_main!(benches);
