//! Cold, warm, and full-solve AOT measurements over the shared LSODE2 corpus.
//!
//! Callback-only throughput lives in `lsode2_workload_callbacks`. This bench
//! deliberately keeps solver preparation and integration separate so an AOT
//! cache or controller regression is visible without reusing test fixtures.

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;
use tempfile::tempdir;

#[path = "support/lsode2_bench_support.rs"]
mod lsode2_bench_support;

use RustedSciThe::numerical::LSODE2::workload_fixtures::{
    SymbolicWorkload, WorkloadKind, build_workload,
};
use RustedSciThe::numerical::LSODE2::{
    IvpLambdifyExecutionPolicy, IvpTelemetry, Lsode2AotProfile, Lsode2AotToolchain,
    Lsode2ControllerConfig, Lsode2ProblemConfig, Lsode2ResidualJacobianSource, Lsode2Solver,
    Lsode2SymbolicAssemblyBackend, Lsode2SymbolicExecutionMode,
};
use RustedSciThe::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy;
use RustedSciThe::symbolic::codegen::codegen_tasks::SparseChunkingStrategy;
use RustedSciThe::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use RustedSciThe::symbolic::symbolic_ivp_generated::{
    SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
};

const DEFAULT_DIFFUSION_DIMENSIONS: &[usize] = &[64, 128];

fn diffusion_dimensions() -> Vec<usize> {
    lsode2_bench_support::dimensions_from_env(
        "LSODE2_BENCH_AOT_DIFFUSION_DIMENSIONS",
        DEFAULT_DIFFUSION_DIMENSIONS,
    )
}

#[derive(Clone, Copy, Debug)]
enum MatrixRoute {
    Sparse,
    Banded,
}

impl MatrixRoute {
    const ALL: [Self; 2] = [Self::Sparse, Self::Banded];

    const fn label(self) -> &'static str {
        match self {
            Self::Sparse => "sparse",
            Self::Banded => "banded",
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum FrontendRoute {
    ExprLegacy,
    AtomViewNative,
}

impl FrontendRoute {
    const ALL: [Self; 2] = [Self::ExprLegacy, Self::AtomViewNative];

    const fn label(self) -> &'static str {
        match self {
            Self::ExprLegacy => "expr-legacy",
            Self::AtomViewNative => "atom-native",
        }
    }

    const fn assembly(self) -> Lsode2SymbolicAssemblyBackend {
        match self {
            Self::ExprLegacy => Lsode2SymbolicAssemblyBackend::ExprLegacy,
            Self::AtomViewNative => Lsode2SymbolicAssemblyBackend::AtomView,
        }
    }
}

fn dimensions(kind: WorkloadKind) -> Vec<usize> {
    match kind {
        WorkloadKind::DiffusionChain => diffusion_dimensions(),
        WorkloadKind::CombustionLike | WorkloadKind::Robertson => vec![3],
        WorkloadKind::StiffScalar => vec![1],
        WorkloadKind::ThreeBody => vec![12],
    }
}

fn aot_workloads() -> Vec<WorkloadKind> {
    lsode2_bench_support::workloads_from_env("LSODE2_BENCH_AOT_WORKLOADS", &WorkloadKind::ALL)
}

fn generated_backend(output_parent: &Path, cold: bool) -> SymbolicIvpGeneratedBackendConfig {
    let policy = if cold {
        SymbolicIvpAotBuildPolicy::RebuildAlways {
            profile: AotBuildProfile::Release,
        }
    } else {
        SymbolicIvpAotBuildPolicy::BuildIfMissing {
            profile: AotBuildProfile::Release,
        }
    };

    SymbolicIvpGeneratedBackendConfig::new()
        .with_output_parent_dir(Some(output_parent.to_path_buf()))
        .with_build_policy(policy)
        .with_c_tcc()
        .with_residual_chunking_strategy(ResidualChunkingStrategy::Whole)
        .with_sparse_jacobian_chunking_strategy(SparseChunkingStrategy::Whole)
}

fn solver_config(
    workload: SymbolicWorkload,
    matrix: MatrixRoute,
    frontend: FrontendRoute,
    execution: Lsode2SymbolicExecutionMode,
    output_parent: &Path,
    cold: bool,
) -> Lsode2ProblemConfig {
    let assembly = frontend.assembly();
    let parameter_names = workload.parameter_names.clone();
    let parameter_values = workload.parameter_values.clone();
    let mut config = Lsode2ProblemConfig::new(
        workload.equations,
        workload.variables,
        workload.time_variable,
        0.0,
        workload.initial_state,
        0.25,
        0.02,
        1.0e-7,
        1.0e-9,
    )
    .with_controller(Lsode2ControllerConfig::bdf_only())
    .with_faithful_bdf_solve(200_000, 200_000)
    .with_lambdify_execution_policy(IvpLambdifyExecutionPolicy::Sequential)
    .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
        assembly,
        execution,
    })
    .with_telemetry(IvpTelemetry::detailed());

    if !parameter_names.is_empty() {
        config = config
            .with_equation_parameters(parameter_names)
            .with_equation_parameter_values(parameter_values);
    }

    config = match matrix {
        MatrixRoute::Sparse => config.with_native_sparse_faer_backend(),
        MatrixRoute::Banded => config.with_native_banded_faithful_backend(),
    };

    if matches!(execution, Lsode2SymbolicExecutionMode::Aot { .. }) {
        let generated = generated_backend(output_parent, cold);
        config = match matrix {
            MatrixRoute::Sparse => config.with_native_sparse_faer_generated_backend(generated),
            MatrixRoute::Banded => config.with_native_banded_faithful_generated_backend(generated),
        };
    }
    config
}

fn build_solver_config(
    kind: WorkloadKind,
    dimension: usize,
    matrix: MatrixRoute,
    frontend: FrontendRoute,
    execution: Lsode2SymbolicExecutionMode,
    output_parent: &Path,
    cold: bool,
) -> Lsode2ProblemConfig {
    solver_config(
        build_workload(kind, dimension),
        matrix,
        frontend,
        execution,
        output_parent,
        cold,
    )
}

fn execution(aot: bool) -> Lsode2SymbolicExecutionMode {
    if aot {
        Lsode2SymbolicExecutionMode::Aot {
            toolchain: Lsode2AotToolchain::CTcc,
            profile: Lsode2AotProfile::Release,
        }
    } else {
        Lsode2SymbolicExecutionMode::LambdifyExpr
    }
}

fn benchmark_cold_preparation(c: &mut Criterion) {
    let workloads = aot_workloads();
    lsode2_bench_support::print_metadata(
        "lsode2_workload_aot",
        &diffusion_dimensions(),
        &format!(
            "compiler=tcc; workloads={:?}; routes=ExprLegacy,AtomViewNative; cold=RebuildAlways+fresh-output-per-iteration; groups=cold-preparation,warm-full-solve,cold-full-solve",
            workloads
                .iter()
                .map(|workload| workload.label())
                .collect::<Vec<_>>()
        ),
    );
    let mut group = c.benchmark_group("lsode2_workload_aot_cold_preparation");
    group.sample_size(lsode2_bench_support::sample_size());
    if let Some(measurement_time) = lsode2_bench_support::measurement_time() {
        group.measurement_time(measurement_time);
    }

    for kind in &workloads {
        for dimension in dimensions(*kind) {
            for matrix in MatrixRoute::ALL {
                for frontend in FrontendRoute::ALL {
                    let id = BenchmarkId::new(
                        format!(
                            "{}/{}/{}/cold-prepare",
                            kind.label(),
                            matrix.label(),
                            frontend.label()
                        ),
                        dimension,
                    );
                    group.bench_function(id, |bencher| {
                        bencher.iter_custom(|iterations| {
                            let started = Instant::now();
                            for _ in 0..iterations {
                                let output = tempdir()
                                    .expect("AOT cold benchmark output directory should exist");
                                let config = build_solver_config(
                                    *kind,
                                    dimension,
                                    matrix,
                                    frontend,
                                    execution(true),
                                    output.path(),
                                    true,
                                );
                                let mut solver = Lsode2Solver::new(config)
                                    .expect("AOT cold solver construction should succeed");
                                solver
                                    .prepare()
                                    .expect("AOT cold solver preparation should succeed");
                                black_box(solver.is_prepared());
                            }
                            started.elapsed()
                        });
                    });
                }
            }
        }
    }
    group.finish();
}

fn benchmark_warm_full_solve(c: &mut Criterion) {
    let workloads = aot_workloads();
    let mut group = c.benchmark_group("lsode2_workload_aot_warm_full_solve");
    group.sample_size(lsode2_bench_support::sample_size());
    if let Some(measurement_time) = lsode2_bench_support::measurement_time() {
        group.measurement_time(measurement_time);
    }
    let cache_root = tempdir().expect("AOT warm benchmark cache directory should exist");

    for kind in &workloads {
        for dimension in dimensions(*kind) {
            for matrix in MatrixRoute::ALL {
                for frontend in FrontendRoute::ALL {
                    for aot in [false, true] {
                        let route = if aot { "aot" } else { "lambdify" };
                        let id = BenchmarkId::new(
                            format!(
                                "{}/{}/{}/{}/warm-solve",
                                kind.label(),
                                matrix.label(),
                                frontend.label(),
                                route
                            ),
                            dimension,
                        );
                        group.bench_function(id, |bencher| {
                            bencher.iter_batched(
                                || {
                                    let config = build_solver_config(
                                        *kind,
                                        dimension,
                                        matrix,
                                        frontend,
                                        execution(aot),
                                        cache_root.path(),
                                        false,
                                    );
                                    let mut solver = Lsode2Solver::new(config)
                                        .expect("warm solver construction should succeed");
                                    solver
                                        .prepare()
                                        .expect("warm solver preparation should succeed");
                                    solver
                                },
                                |mut solver| {
                                    let summary = solver
                                        .solve_with_summary()
                                        .expect("warm full solve should succeed");
                                    assert!(summary.final_y.as_ref().is_some_and(|state| {
                                        state.iter().all(|value| value.is_finite())
                                    }));
                                    black_box(summary);
                                },
                                BatchSize::SmallInput,
                            );
                        });
                    }
                }
            }
        }
    }
    group.finish();
}

fn benchmark_cold_full_solve(c: &mut Criterion) {
    let workloads = aot_workloads();
    let mut group = c.benchmark_group("lsode2_workload_aot_cold_full_solve");
    group.sample_size(lsode2_bench_support::sample_size());
    if let Some(measurement_time) = lsode2_bench_support::measurement_time() {
        group.measurement_time(measurement_time);
    }

    for kind in &workloads {
        for dimension in dimensions(*kind) {
            for matrix in MatrixRoute::ALL {
                for frontend in FrontendRoute::ALL {
                    let id = BenchmarkId::new(
                        format!(
                            "{}/{}/{}/cold-e2e",
                            kind.label(),
                            matrix.label(),
                            frontend.label()
                        ),
                        dimension,
                    );
                    group.bench_function(id, |bencher| {
                        bencher.iter_custom(|iterations| {
                            let started = Instant::now();
                            for _ in 0..iterations {
                                let output =
                                    tempdir().expect("AOT cold E2E output directory should exist");
                                let config = build_solver_config(
                                    *kind,
                                    dimension,
                                    matrix,
                                    frontend,
                                    execution(true),
                                    output.path(),
                                    true,
                                );
                                let mut solver = Lsode2Solver::new(config)
                                    .expect("AOT cold E2E solver construction should succeed");
                                solver
                                    .prepare()
                                    .expect("AOT cold E2E preparation should succeed");
                                let summary = solver
                                    .solve_with_summary()
                                    .expect("AOT cold E2E solve should succeed");
                                assert!(summary.final_y.as_ref().is_some_and(|state| {
                                    state.iter().all(|value| value.is_finite())
                                }));
                                black_box(summary);
                            }
                            started.elapsed()
                        });
                    });
                }
            }
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_cold_preparation,
    benchmark_warm_full_solve,
    benchmark_cold_full_solve
);
criterion_main!(benches);
