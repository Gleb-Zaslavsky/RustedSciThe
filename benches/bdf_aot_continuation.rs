//! AOT parameter-continuation cost: reuse prepared runtime vs fresh solver
//! reconnects against the same prebuilt artifact. Artifact production is
//! intentionally outside the timed region; cold AOT cost is measured separately.

use RustedSciThe::numerical::BDF::BDF_api::{BdfSolverOptions, ODEsolver};
use RustedSciThe::numerical::ivp_workloads::{
    SymbolicWorkload, WorkloadKind, build_workload, parameter_continuation_target,
};
use RustedSciThe::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use RustedSciThe::symbolic::symbolic_ivp::IvpSymbolicAssemblyBackend;
use RustedSciThe::symbolic::symbolic_ivp_generated::{
    SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use nalgebra::DVector;
use std::{hint::black_box, time::Duration};
use tempfile::TempDir;

#[path = "support/bdf_bench_support.rs"]
#[allow(dead_code)]
mod bdf_bench_support;

struct AotArtifact {
    config: SymbolicIvpGeneratedBackendConfig,
    _directory: TempDir,
}

fn selected_workloads() -> Vec<WorkloadKind> {
    bdf_bench_support::workloads_from_env(
        "BDF_BENCH_AOT_CONTINUATION_WORKLOADS",
        "combustion-like,three-body",
    )
}

fn dimensions(kind: WorkloadKind) -> Vec<usize> {
    if kind != WorkloadKind::DiffusionChain {
        return vec![0];
    }
    std::env::var("BDF_BENCH_AOT_CONTINUATION_DIFFUSION_DIMENSIONS")
        .unwrap_or_else(|_| "8,16".to_string())
        .split(',')
        .map(|value| {
            value
                .trim()
                .parse::<usize>()
                .expect("diffusion dimensions must be positive integers")
        })
        .inspect(|dimension| assert!(*dimension > 0, "diffusion dimension must be positive"))
        .collect()
}

fn segment_counts() -> Vec<usize> {
    std::env::var("BDF_BENCH_AOT_CONTINUATION_COUNTS")
        .unwrap_or_else(|_| "1,4,16".to_string())
        .split(',')
        .map(|value| {
            value
                .trim()
                .parse::<usize>()
                .expect("continuation counts must be positive integers")
        })
        .inspect(|count| assert!(*count > 0, "continuation count must be positive"))
        .collect()
}

fn segment_width(kind: WorkloadKind) -> f64 {
    match kind {
        WorkloadKind::CombustionLike => 2.0e-4,
        WorkloadKind::ThreeBody | WorkloadKind::DiffusionChain => 5.0e-4,
        _ => panic!(
            "{} is not a parameterized continuation workload",
            kind.label()
        ),
    }
}

fn assembly_routes() -> [(IvpSymbolicAssemblyBackend, &'static str); 2] {
    [
        (IvpSymbolicAssemblyBackend::ExprLegacy, "expr-legacy"),
        (IvpSymbolicAssemblyBackend::AtomView, "atom-native"),
    ]
}

fn make_aot_artifact(
    workload: &SymbolicWorkload,
    dimension: usize,
    assembly: IvpSymbolicAssemblyBackend,
) -> AotArtifact {
    let directory = tempfile::tempdir().expect("isolated AOT continuation artifact directory");
    let config = SymbolicIvpGeneratedBackendConfig::new()
        .with_output_parent_dir(Some(directory.path().to_path_buf()))
        .with_build_policy(SymbolicIvpAotBuildPolicy::RebuildAlways {
            profile: AotBuildProfile::Release,
        })
        .with_c_tcc();
    let mut producer = make_solver(
        workload,
        dimension,
        assembly,
        0.0,
        workload.initial_state.clone(),
        segment_width(workload.kind),
        workload.parameter_values.clone(),
        config,
    );
    producer.try_generate().unwrap_or_else(|error| {
        panic!(
            "{} {assembly:?} AOT artifact build: {error}",
            workload.kind.label()
        )
    });
    let config = producer
        .generated_backend_config()
        .clone()
        .with_build_policy(SymbolicIvpAotBuildPolicy::RequirePrebuilt);
    AotArtifact {
        config,
        _directory: directory,
    }
}

fn make_solver(
    workload: &SymbolicWorkload,
    _dimension: usize,
    assembly: IvpSymbolicAssemblyBackend,
    t0: f64,
    y0: DVector<f64>,
    t_bound: f64,
    parameters: DVector<f64>,
    generated: SymbolicIvpGeneratedBackendConfig,
) -> ODEsolver {
    let max_step = match workload.kind {
        WorkloadKind::CombustionLike => 1.0e-4,
        WorkloadKind::ThreeBody | WorkloadKind::DiffusionChain => 2.5e-4,
        _ => panic!(
            "{} is not a parameterized continuation workload",
            workload.kind.label()
        ),
    };
    let options = BdfSolverOptions::for_bdf(
        workload.equations.clone(),
        workload.variables.clone(),
        workload.time_variable.clone(),
        t0,
        y0,
        t_bound,
        max_step,
        1.0e-7,
        1.0e-10,
        None,
        false,
        Some(max_step),
    )
    .with_equation_parameters(workload.parameter_names.clone())
    .with_equation_parameter_values(parameters)
    .with_symbolic_assembly_backend(assembly);
    ODEsolver::new_with_options(options).with_generated_backend_config(generated)
}

fn seed_solver(
    workload: &SymbolicWorkload,
    dimension: usize,
    assembly: IvpSymbolicAssemblyBackend,
    artifact: &AotArtifact,
) -> ODEsolver {
    let mut solver = make_solver(
        workload,
        dimension,
        assembly,
        0.0,
        workload.initial_state.clone(),
        segment_width(workload.kind),
        workload.parameter_values.clone(),
        artifact.config.clone(),
    );
    solver.try_solve().unwrap_or_else(|error| {
        panic!(
            "{} {assembly:?} AOT continuation seed: {error}",
            workload.kind.label()
        )
    });
    solver
}

fn benchmark_aot_parameter_continuation(c: &mut Criterion) {
    let workloads = selected_workloads();
    let counts = segment_counts();
    let mut group = c.benchmark_group("bdf_aot_parameter_continuation");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));

    for kind in workloads {
        assert!(
            matches!(
                kind,
                WorkloadKind::CombustionLike
                    | WorkloadKind::ThreeBody
                    | WorkloadKind::DiffusionChain
            ),
            "{} has no parameterized AOT continuation fixture",
            kind.label()
        );
        for dimension in dimensions(kind) {
            let workload = build_workload(kind, dimension.max(1));
            let width = segment_width(kind);
            for (assembly, assembly_name) in assembly_routes() {
                let artifact = make_aot_artifact(&workload, dimension, assembly);
                for segments in counts.iter().copied() {
                    let mut continued = seed_solver(&workload, dimension, assembly, &artifact);
                    let mut fresh = seed_solver(&workload, dimension, assembly, &artifact);
                    for index in 1..=segments {
                        let parameters =
                            parameter_continuation_target(&workload.parameter_values, index);
                        let (times, states) = fresh.get_result_ref();
                        let t0 = *times.as_slice().last().expect("seed final time");
                        let y0 = states.row(states.nrows() - 1).transpose();
                        let target = t0 + width;

                        continued
                            .try_continue_with_parameter_values(parameters.clone(), target)
                            .expect("continuation restart");
                        continued.try_solve().expect("continued AOT solve");
                        fresh = make_solver(
                            &workload,
                            dimension,
                            assembly,
                            t0,
                            y0,
                            target,
                            parameters,
                            artifact.config.clone(),
                        );
                        fresh.try_solve().expect("fresh AOT segment solve");
                    }
                    let drift = bdf_bench_support::final_state(&continued)
                        .iter()
                        .zip(bdf_bench_support::final_state(&fresh).iter())
                        .map(|(left, right)| (left - right).abs())
                        .fold(0.0_f64, f64::max);
                    assert!(
                        drift <= 2.0e-8,
                        "{} {assembly_name} segments={segments} continuation drift={drift:e}",
                        kind.label()
                    );
                    eprintln!(
                        "[BDF AOT continuation preflight] workload={} dimension={} assembly={} segments={} final_diff={drift:.3e} cache_policy=require-prebuilt status=ok",
                        kind.label(),
                        if kind == WorkloadKind::DiffusionChain {
                            dimension.to_string()
                        } else {
                            "fixed".to_string()
                        },
                        assembly_name,
                        segments,
                    );

                    let id = format!(
                        "{}/{}/{}",
                        kind.label(),
                        if kind == WorkloadKind::DiffusionChain {
                            dimension.to_string()
                        } else {
                            "fixed".to_string()
                        },
                        assembly_name,
                    );
                    group.bench_function(
                        BenchmarkId::new(format!("warm-reuse/{id}"), segments),
                        |bencher| {
                            bencher.iter_batched(
                                || seed_solver(&workload, dimension, assembly, &artifact),
                                |mut solver| {
                                    for index in 1..=segments {
                                        let target = width * (index + 1) as f64;
                                        solver
                                            .try_continue_with_parameter_values(
                                                parameter_continuation_target(
                                                    &workload.parameter_values,
                                                    index,
                                                ),
                                                target,
                                            )
                                            .expect("warm continuation restart");
                                        solver.try_solve().expect("warm continuation solve");
                                    }
                                    black_box(bdf_bench_support::final_state(&solver));
                                },
                                BatchSize::SmallInput,
                            );
                        },
                    );
                    group.bench_function(
                        BenchmarkId::new(format!("fresh-reconnect/{id}"), segments),
                        |bencher| {
                            bencher.iter_batched(
                                || seed_solver(&workload, dimension, assembly, &artifact),
                                |mut solver| {
                                    for index in 1..=segments {
                                        let (times, states) = solver.get_result_ref();
                                        let t0 =
                                            *times.as_slice().last().expect("fresh final time");
                                        let y0 = states.row(states.nrows() - 1).transpose();
                                        let target = t0 + width;
                                        solver = make_solver(
                                            &workload,
                                            dimension,
                                            assembly,
                                            t0,
                                            y0,
                                            target,
                                            parameter_continuation_target(
                                                &workload.parameter_values,
                                                index,
                                            ),
                                            artifact.config.clone(),
                                        );
                                        solver.try_solve().expect("fresh AOT continuation segment");
                                    }
                                    black_box(bdf_bench_support::final_state(&solver));
                                },
                                BatchSize::SmallInput,
                            );
                        },
                    );
                }
            }
        }
    }
    group.finish();
}

criterion_group!(benches, benchmark_aot_parameter_continuation);
criterion_main!(benches);
