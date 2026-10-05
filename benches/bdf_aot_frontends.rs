//! Solver-facing BDF AOT measurements with isolated cold artifact directories.

use RustedSciThe::numerical::BDF::BDF_api::{BdfTelemetryMode, ODEsolver};
use RustedSciThe::numerical::ivp_workloads::WorkloadKind;
use RustedSciThe::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use RustedSciThe::symbolic::symbolic_ivp::IvpSymbolicAssemblyBackend;
use RustedSciThe::symbolic::symbolic_ivp_generated::{
    SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::Duration;
use tempfile::TempDir;

#[path = "support/bdf_bench_support.rs"]
mod bdf_bench_support;

struct IsolatedAotCase {
    solver: ODEsolver,
    _artifact_dir: TempDir,
}

fn aot_routes() -> [(IvpSymbolicAssemblyBackend, &'static str); 2] {
    [
        (IvpSymbolicAssemblyBackend::ExprLegacy, "expr-legacy"),
        (IvpSymbolicAssemblyBackend::AtomView, "atom-native"),
    ]
}

fn selected_aot_workloads() -> Vec<WorkloadKind> {
    bdf_bench_support::workloads_from_env("BDF_BENCH_AOT_WORKLOADS", "stiff-scalar,robertson")
}

fn make_isolated_aot_case(
    workload: WorkloadKind,
    dimension: usize,
    assembly: IvpSymbolicAssemblyBackend,
    build_policy: SymbolicIvpAotBuildPolicy,
    telemetry_mode: BdfTelemetryMode,
) -> IsolatedAotCase {
    let artifact_dir = tempfile::tempdir().expect("isolated BDF AOT artifact directory");
    let config = SymbolicIvpGeneratedBackendConfig::new()
        .with_output_parent_dir(Some(artifact_dir.path().to_path_buf()))
        .with_build_policy(build_policy)
        .with_c_tcc();
    let solver = bdf_bench_support::make_solver_with_telemetry(
        workload,
        dimension,
        assembly,
        telemetry_mode,
    )
    .with_generated_backend_config(config);
    IsolatedAotCase {
        solver,
        _artifact_dir: artifact_dir,
    }
}

fn dimension_label(workload: WorkloadKind, dimension: usize) -> String {
    if matches!(workload, WorkloadKind::DiffusionChain) {
        dimension.to_string()
    } else {
        "fixed".to_string()
    }
}

fn benchmark_aot_frontends(c: &mut Criterion) {
    let mut group = c.benchmark_group("bdf_aot_frontends_tcc");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));

    for workload in selected_aot_workloads() {
        for dimension in bdf_bench_support::dimensions(workload) {
            let mut reference = bdf_bench_support::make_solver(
                workload,
                dimension,
                IvpSymbolicAssemblyBackend::ExprLegacy,
            );
            reference.solve();
            assert_eq!(reference.get_status(), "finished");
            let reference_state = bdf_bench_support::final_state(&reference);

            for (assembly, route) in aot_routes() {
                let mut check = make_isolated_aot_case(
                    workload,
                    dimension,
                    assembly,
                    SymbolicIvpAotBuildPolicy::RebuildAlways {
                        profile: AotBuildProfile::Release,
                    },
                    BdfTelemetryMode::Timings,
                );
                check.solver.try_generate().unwrap_or_else(|error| {
                    panic!("{} {route} AOT preflight: {error}", workload.label())
                });
                check.solver.solve();
                assert_eq!(check.solver.get_status(), "finished");
                let final_state = bdf_bench_support::final_state(&check.solver);
                let max_diff = reference_state
                    .iter()
                    .zip(final_state.iter())
                    .map(|(left, right)| (left - right).abs())
                    .fold(0.0_f64, f64::max);
                assert!(
                    max_diff <= 2e-5,
                    "{} {route} AOT parity preflight diff={max_diff:e}",
                    workload.label()
                );
                if let Some(provenance) = check.solver.aot_provenance() {
                    let snapshot = &provenance.telemetry;
                    println!(
                        "[BDF AOT provenance] workload={} dimension={} assembly={} policy={:?} codegen={} compiler={:?} execution={} route={:?} matrix={:?} key={:?} hits={} misses={} builds={}/{} links={}/{} runtime_ready={}",
                        workload.label(),
                        dimension_label(workload, dimension),
                        route,
                        provenance.build_policy,
                        provenance.backend_label(),
                        provenance.c_compiler,
                        provenance.execution_label(),
                        snapshot.route,
                        snapshot.matrix_backend,
                        snapshot.aot_artifact_keys,
                        snapshot.aot_resolution_hits,
                        snapshot.aot_resolution_misses,
                        snapshot.aot_build_successes,
                        snapshot.aot_build_attempts,
                        snapshot.aot_link_successes,
                        snapshot.aot_link_attempts,
                        snapshot.aot_runtime_ready,
                    );
                }

                let input = dimension_label(workload, dimension);
                group.bench_function(
                    BenchmarkId::new(format!("cold-prepare/{}/{route}", workload.label()), &input),
                    |bencher| {
                        bencher.iter_batched(
                            || {
                                make_isolated_aot_case(
                                    workload,
                                    dimension,
                                    assembly,
                                    SymbolicIvpAotBuildPolicy::RebuildAlways {
                                        profile: AotBuildProfile::Release,
                                    },
                                    BdfTelemetryMode::Off,
                                )
                            },
                            |mut case| {
                                case.solver
                                    .try_generate()
                                    .expect("cold BDF AOT preparation");
                                black_box(case.solver.get_statistics());
                            },
                            BatchSize::SmallInput,
                        );
                    },
                );
                group.bench_function(
                    BenchmarkId::new(format!("warm-solve/{}/{route}", workload.label()), &input),
                    |bencher| {
                        bencher.iter_batched(
                            || {
                                let mut case = make_isolated_aot_case(
                                    workload,
                                    dimension,
                                    assembly,
                                    SymbolicIvpAotBuildPolicy::BuildIfMissing {
                                        profile: AotBuildProfile::Release,
                                    },
                                    BdfTelemetryMode::Off,
                                );
                                case.solver.try_generate().expect("prepare BDF AOT solver");
                                case
                            },
                            |mut case| {
                                case.solver.solve();
                                black_box(bdf_bench_support::final_state(&case.solver));
                            },
                            BatchSize::SmallInput,
                        );
                    },
                );
                group.bench_function(
                    BenchmarkId::new(format!("cold-e2e/{}/{route}", workload.label()), &input),
                    |bencher| {
                        bencher.iter_batched(
                            || {
                                make_isolated_aot_case(
                                    workload,
                                    dimension,
                                    assembly,
                                    SymbolicIvpAotBuildPolicy::RebuildAlways {
                                        profile: AotBuildProfile::Release,
                                    },
                                    BdfTelemetryMode::Off,
                                )
                            },
                            |mut case| {
                                case.solver
                                    .try_generate()
                                    .expect("cold BDF AOT preparation");
                                case.solver.solve();
                                black_box(bdf_bench_support::final_state(&case.solver));
                            },
                            BatchSize::SmallInput,
                        );
                    },
                );
            }
        }
    }
    group.finish();
}

criterion_group!(benches, benchmark_aot_frontends);
criterion_main!(benches);
