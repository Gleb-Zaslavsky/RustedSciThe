//! Apples-to-apples BDF execution matrix for Lambdify and dense AOT.
//!
//! Each execution mode uses the same workload builder, assembly frontend,
//! tolerances, interval, initial state, and preflight reference.

use RustedSciThe::numerical::BDF::BDF_api::{BdfTelemetryMode, ODEsolver};
use RustedSciThe::numerical::BDF::BDF_solver::BdfJacobian;
use RustedSciThe::numerical::ivp_workloads::{WorkloadKind, build_workload};
use RustedSciThe::symbolic::codegen::codegen_aot_runtime_link::{
    LinkedDenseAotBackend, resolve_linked_dense_backend,
};
use RustedSciThe::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use RustedSciThe::symbolic::ivp_telemetry::IvpTelemetry;
use RustedSciThe::symbolic::symbolic_ivp::{IvpSymbolicAssemblyBackend, SymbolicIvpProblemOptions};
use RustedSciThe::symbolic::symbolic_ivp_generated::{
    PreparedGeneratedSymbolicIvpProblem, SymbolicIvpAotBuildPolicy,
    SymbolicIvpGeneratedBackendConfig, prepare_generated_symbolic_ivp_problem,
};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use faer::prelude::Solve as _;
use nalgebra::{DMatrix, DVector, LU};
use std::{hint::black_box, time::Duration};
use tempfile::TempDir;

#[path = "support/bdf_bench_support.rs"]
#[allow(dead_code)]
mod bdf_bench_support;

#[derive(Clone, Copy)]
enum Execution {
    Lambdify,
    Aot,
}

impl Execution {
    fn label(self) -> &'static str {
        match self {
            Self::Lambdify => "lambdify",
            Self::Aot => "aot-tcc",
        }
    }
}

struct AotContext {
    config: SymbolicIvpGeneratedBackendConfig,
    _artifact_dir: TempDir,
}

struct PreparedCallbacks {
    prepared: PreparedGeneratedSymbolicIvpProblem,
    state: DVector<f64>,
    residual_out: DVector<f64>,
    args: Vec<f64>,
    aot_jacobian: Option<LinkedDenseAotBackend>,
    jacobian_args: Vec<f64>,
    jacobian_out: Vec<f64>,
}

fn workloads() -> Vec<WorkloadKind> {
    bdf_bench_support::workloads_from_env(
        "BDF_BENCH_APPLE_WORKLOADS",
        "stiff-scalar,robertson,combustion-like",
    )
}

fn env_enabled(name: &str, default: bool) -> bool {
    std::env::var(name)
        .map(|value| {
            !matches!(
                value.to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
        })
        .unwrap_or(default)
}

fn dense_linear_dimensions() -> Vec<usize> {
    std::env::var("BDF_BENCH_LINEAR_DIMENSIONS")
        .unwrap_or_else(|_| "32,64,100,512,1024".to_string())
        .split(',')
        .map(|value| {
            value
                .trim()
                .parse::<usize>()
                .expect("BDF_BENCH_LINEAR_DIMENSIONS must contain positive integers")
        })
        .collect()
}

fn dense_linear_fixture(dimension: usize) -> DMatrix<f64> {
    DMatrix::from_fn(dimension, dimension, |row, col| {
        if row == col {
            4.0 + dimension as f64 * 1e-3
        } else {
            (((row * 17 + col * 31) % 11) as f64 - 5.0) * 1e-4
        }
    })
}

fn make_aot_config(
    output: &std::path::Path,
    policy: SymbolicIvpAotBuildPolicy,
) -> SymbolicIvpGeneratedBackendConfig {
    SymbolicIvpGeneratedBackendConfig::new()
        .with_output_parent_dir(Some(output.to_path_buf()))
        .with_build_policy(policy)
        .with_c_tcc()
}

fn prepare_aot_context(
    workload: WorkloadKind,
    dimension: usize,
    assembly: IvpSymbolicAssemblyBackend,
) -> AotContext {
    let artifact_dir = tempfile::tempdir().expect("BDF comparison AOT directory");
    let config = make_aot_config(
        artifact_dir.path(),
        SymbolicIvpAotBuildPolicy::RebuildAlways {
            profile: AotBuildProfile::Release,
        },
    );
    let mut solver = bdf_bench_support::make_solver_with_telemetry(
        workload,
        dimension,
        assembly,
        BdfTelemetryMode::Timings,
    )
    .with_generated_backend_config(config);
    solver.try_generate().unwrap_or_else(|error| {
        panic!(
            "{} {assembly:?} AOT matrix build: {error}",
            workload.label()
        )
    });
    if let Some(provenance) = solver.aot_provenance() {
        let snapshot = &provenance.telemetry;
        println!(
            "[BDF matched AOT provenance] workload={} dimension={} assembly={assembly:?} producer=RebuildAlways consumer=RequirePrebuilt policy={:?} codegen={} compiler={:?} execution={} route={:?} matrix={:?} key={:?} hits={} misses={} builds={}/{} links={}/{} runtime_ready={}",
            workload.label(),
            dimension,
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
    let config = solver
        .generated_backend_config()
        .clone()
        .with_build_policy(SymbolicIvpAotBuildPolicy::RequirePrebuilt);
    AotContext {
        config,
        _artifact_dir: artifact_dir,
    }
}

fn prepare_dense_aot_context(dimension: usize, assembly: IvpSymbolicAssemblyBackend) -> AotContext {
    let artifact_dir = tempfile::tempdir().expect("dense comparison AOT directory");
    let config = make_aot_config(
        artifact_dir.path(),
        SymbolicIvpAotBuildPolicy::RebuildAlways {
            profile: AotBuildProfile::Release,
        },
    );
    let mut solver = bdf_bench_support::make_dense_coupled_solver(
        dimension,
        assembly,
        BdfTelemetryMode::Timings,
    )
    .with_generated_backend_config(config);
    solver
        .try_generate()
        .unwrap_or_else(|error| panic!("dense n={dimension} {assembly:?} AOT build: {error}"));
    if let Some(provenance) = solver.aot_provenance() {
        let snapshot = &provenance.telemetry;
        println!(
            "[BDF matched dense AOT provenance] dimension={} assembly={assembly:?} producer=RebuildAlways consumer=RequirePrebuilt policy={:?} codegen={} compiler={:?} execution={} route={:?} matrix={:?} key={:?} hits={} misses={} builds={}/{} links={}/{} runtime_ready={}",
            dimension,
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
    let config = solver
        .generated_backend_config()
        .clone()
        .with_build_policy(SymbolicIvpAotBuildPolicy::RequirePrebuilt);
    AotContext {
        config,
        _artifact_dir: artifact_dir,
    }
}

fn solver_for(
    workload: WorkloadKind,
    dimension: usize,
    assembly: IvpSymbolicAssemblyBackend,
    execution: Execution,
    aot: Option<&AotContext>,
) -> ODEsolver {
    let solver = bdf_bench_support::make_solver(workload, dimension, assembly);
    match execution {
        Execution::Lambdify => solver,
        Execution::Aot => {
            solver.with_generated_backend_config(aot.expect("AOT context").config.clone())
        }
    }
}

fn prepare_callbacks(
    workload_kind: WorkloadKind,
    dimension: usize,
    assembly: IvpSymbolicAssemblyBackend,
    execution: Execution,
    aot: Option<&AotContext>,
) -> PreparedCallbacks {
    let workload = build_workload(workload_kind, dimension.max(1));
    let state = workload.initial_state.clone();
    let residual_len = workload.equations.len();
    let args_len = workload.variables.len() + workload.parameter_names.len() + 1;
    let mut jacobian_args = Vec::with_capacity(args_len);
    jacobian_args.push(0.01);
    jacobian_args.extend(workload.parameter_values.iter().copied());
    jacobian_args.extend(state.iter().copied());
    let mut options = SymbolicIvpProblemOptions::new()
        .with_symbolic_assembly_backend(assembly)
        .with_telemetry(IvpTelemetry::disabled());
    if !workload.parameter_names.is_empty() {
        options = options
            .with_equation_parameters(workload.parameter_names)
            .with_equation_parameter_values(workload.parameter_values);
    }
    let config = match execution {
        Execution::Lambdify => SymbolicIvpGeneratedBackendConfig::new(),
        Execution::Aot => aot.expect("AOT context").config.clone(),
    };
    let prepared = prepare_generated_symbolic_ivp_problem(
        workload.equations,
        workload.variables,
        workload.time_variable,
        options,
        config,
    )
    .expect("prepare callback-only matrix route");
    let aot_jacobian = prepared.aot_runtime().map(|runtime| {
        resolve_linked_dense_backend(runtime.problem_key())
            .expect("prepared AOT dense callback is registered")
    });
    let jacobian_out = aot_jacobian
        .as_ref()
        .map(|backend| vec![0.0; backend.shape.0 * backend.shape.1])
        .unwrap_or_default();
    PreparedCallbacks {
        prepared,
        state,
        residual_out: DVector::zeros(residual_len),
        args: Vec::with_capacity(args_len),
        aot_jacobian,
        jacobian_args,
        jacobian_out,
    }
}

fn benchmark_callback_stages(c: &mut Criterion) {
    if !env_enabled("BDF_BENCH_RUN_CALLBACK_MATRIX", true) {
        eprintln!("[BDF callback matrix] skipped by BDF_BENCH_RUN_CALLBACK_MATRIX");
        return;
    }
    let include_aot_jacobian_boundaries =
        env_enabled("BDF_BENCH_RUN_AOT_JACOBIAN_BOUNDARIES", false);
    if include_aot_jacobian_boundaries {
        eprintln!(
            "[BDF AOT Jacobian boundaries] enabled: raw callback, checked ABI, flattened-argument copy, and row-major dense assembly"
        );
    }
    let mut group = c.benchmark_group("bdf_backend_callback_only");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(2));
    let routes = [
        (IvpSymbolicAssemblyBackend::ExprLegacy, "expr-legacy"),
        (IvpSymbolicAssemblyBackend::AtomView, "atom-native"),
    ];

    for workload in workloads() {
        for dimension in bdf_bench_support::dimensions(workload) {
            for (assembly, assembly_name) in routes {
                for execution in [Execution::Lambdify, Execution::Aot] {
                    let aot_context = matches!(execution, Execution::Aot)
                        .then(|| prepare_aot_context(workload, dimension, assembly));
                    let mut callbacks = prepare_callbacks(
                        workload,
                        dimension,
                        assembly,
                        execution,
                        aot_context.as_ref(),
                    );
                    callbacks
                        .prepared
                        .problem
                        .try_evaluate_residual_into_with_workspace(
                            0.01,
                            &callbacks.state,
                            &mut callbacks.residual_out,
                            &mut callbacks.args,
                        )
                        .expect("callback residual preflight");
                    callbacks
                        .prepared
                        .problem
                        .try_evaluate_jacobian(0.01, &callbacks.state)
                        .expect("callback Jacobian preflight");

                    if include_aot_jacobian_boundaries {
                        if let Some(linked) = callbacks.aot_jacobian.as_ref() {
                            let mut raw_output = callbacks.jacobian_out.clone();
                            (&*linked.jacobian_eval)(&callbacks.jacobian_args, &mut raw_output);
                            let mut checked_output = callbacks.jacobian_out.clone();
                            linked
                                .try_jacobian_eval(&callbacks.jacobian_args, &mut checked_output)
                                .expect("AOT checked Jacobian preflight");
                            assert_eq!(raw_output, checked_output, "raw/checked AOT mismatch");
                            let typed = callbacks
                                .prepared
                                .problem
                                .try_evaluate_jacobian(0.01, &callbacks.state)
                                .expect("AOT typed Jacobian preflight");
                            let (rows, cols) = linked.shape;
                            for row in 0..rows {
                                for col in 0..cols {
                                    let index = row * cols + col;
                                    assert_eq!(raw_output[index], typed[(row, col)]);
                                }
                            }
                            callbacks.jacobian_out.copy_from_slice(&raw_output);
                            eprintln!(
                                "[BDF AOT Jacobian boundary shape] workload={} n={} assembly={} input_values={} input_bytes={} dense_output_values={} dense_output_bytes={} (buffer sizes, not allocator measurements)",
                                workload.label(),
                                dimension,
                                assembly_name,
                                callbacks.jacobian_args.len(),
                                callbacks.jacobian_args.len() * std::mem::size_of::<f64>(),
                                raw_output.len(),
                                raw_output.len() * std::mem::size_of::<f64>(),
                            );
                        }
                    }

                    let prefix = format!(
                        "{}/{}/{}/{}",
                        workload.label(),
                        dimension,
                        assembly_name,
                        execution.label()
                    );
                    group.bench_function(
                        BenchmarkId::new(format!("residual/{prefix}"), "typed-into"),
                        |bencher| {
                            bencher.iter(|| {
                                callbacks
                                    .prepared
                                    .problem
                                    .try_evaluate_residual_into_with_workspace(
                                        black_box(0.01),
                                        black_box(&callbacks.state),
                                        &mut callbacks.residual_out,
                                        &mut callbacks.args,
                                    )
                                    .expect("prepared residual callback");
                                black_box(&callbacks.residual_out);
                            });
                        },
                    );
                    group.bench_function(
                        BenchmarkId::new(format!("jacobian/{prefix}"), "typed-owned"),
                        |bencher| {
                            bencher.iter(|| {
                                let jacobian = callbacks
                                    .prepared
                                    .problem
                                    .try_evaluate_jacobian(
                                        black_box(0.01),
                                        black_box(&callbacks.state),
                                    )
                                    .expect("prepared Jacobian callback");
                                black_box(jacobian);
                            });
                        },
                    );

                    if include_aot_jacobian_boundaries {
                        if let Some(linked) = callbacks.aot_jacobian.as_ref() {
                            let (rows, cols) = linked.shape;
                            let mut assembled = DMatrix::zeros(rows, cols);
                            group.bench_function(
                                BenchmarkId::new(format!("jacobian/{prefix}"), "raw-buffer"),
                                |bencher| {
                                    bencher.iter(|| {
                                        (&*linked.jacobian_eval)(
                                            black_box(&callbacks.jacobian_args),
                                            &mut callbacks.jacobian_out,
                                        );
                                        black_box(&callbacks.jacobian_out);
                                    });
                                },
                            );
                            group.bench_function(
                                BenchmarkId::new(
                                    format!("jacobian/{prefix}"),
                                    "abi-checked-buffer",
                                ),
                                |bencher| {
                                    bencher.iter(|| {
                                        linked
                                            .try_jacobian_eval(
                                                black_box(&callbacks.jacobian_args),
                                                &mut callbacks.jacobian_out,
                                            )
                                            .expect("checked AOT Jacobian callback");
                                        black_box(&callbacks.jacobian_out);
                                    });
                                },
                            );
                            group.bench_function(
                                BenchmarkId::new(format!("jacobian/{prefix}"), "flat-args-copy"),
                                |bencher| {
                                    let mut packed =
                                        Vec::with_capacity(callbacks.jacobian_args.len());
                                    bencher.iter(|| {
                                        packed.clear();
                                        packed
                                            .extend_from_slice(black_box(&callbacks.jacobian_args));
                                        black_box(&packed);
                                    });
                                },
                            );
                            group.bench_function(
                                BenchmarkId::new(
                                    format!("jacobian/{prefix}"),
                                    "row-major-to-dmatrix",
                                ),
                                |bencher| {
                                    bencher.iter(|| {
                                        let input = black_box(&callbacks.jacobian_out);
                                        for row in 0..rows {
                                            for col in 0..cols {
                                                assembled[(row, col)] = input[row * cols + col];
                                            }
                                        }
                                        black_box(&assembled);
                                    });
                                },
                            );
                        }
                    }
                }
            }
        }
    }
    group.finish();
}

fn benchmark_full_execution_matrix(c: &mut Criterion) {
    if !env_enabled("BDF_BENCH_RUN_SOLVER_MATRIX", true) {
        eprintln!("[BDF solver matrix] skipped by BDF_BENCH_RUN_SOLVER_MATRIX");
        return;
    }
    let mut group = c.benchmark_group("bdf_backend_apple_to_apple");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));
    let routes = [
        (IvpSymbolicAssemblyBackend::ExprLegacy, "expr-legacy"),
        (IvpSymbolicAssemblyBackend::AtomView, "atom-native"),
    ];

    for workload in workloads() {
        for dimension in bdf_bench_support::dimensions(workload) {
            let reference = {
                let mut solver = bdf_bench_support::make_solver(
                    workload,
                    dimension,
                    IvpSymbolicAssemblyBackend::ExprLegacy,
                );
                solver.try_solve().expect("Lambdify reference solve");
                bdf_bench_support::final_state(&solver)
            };

            for (assembly, assembly_name) in routes {
                let aot_context = prepare_aot_context(workload, dimension, assembly);
                for execution in [Execution::Lambdify, Execution::Aot] {
                    let input = format!(
                        "{}/{}/{}/{}",
                        workload.label(),
                        dimension,
                        assembly_name,
                        execution.label()
                    );
                    let aot = matches!(execution, Execution::Aot).then_some(&aot_context);

                    let mut preflight = solver_for(workload, dimension, assembly, execution, aot);
                    preflight
                        .try_solve()
                        .unwrap_or_else(|error| panic!("{input} preflight solve failed: {error}"));
                    let state = bdf_bench_support::final_state(&preflight);
                    let max_diff = state
                        .iter()
                        .zip(reference.iter())
                        .map(|(left, right)| (left - right).abs())
                        .fold(0.0_f64, f64::max);
                    assert!(max_diff <= 2e-5, "{input} parity drift={max_diff:e}");
                    eprintln!(
                        "[BDF backend matrix preflight] workload={} dimension={} assembly={} execution={} final_diff={max_diff:.3e} status=ok",
                        workload.label(),
                        dimension,
                        assembly_name,
                        execution.label()
                    );

                    group.bench_function(
                        BenchmarkId::new(format!("prepare/{input}"), "cold"),
                        |bencher| {
                            // Borrowed batches keep solver and artifact cleanup outside timing.
                            bencher.iter_batched_ref(
                                || -> (ODEsolver, Option<TempDir>) {
                                    match execution {
                                        Execution::Lambdify => (
                                            solver_for(
                                                workload, dimension, assembly, execution, None,
                                            ),
                                            None,
                                        ),
                                        Execution::Aot => {
                                            let artifact_dir = tempfile::tempdir()
                                                .expect("isolated cold AOT sample");
                                            let config = make_aot_config(
                                                artifact_dir.path(),
                                                SymbolicIvpAotBuildPolicy::RebuildAlways {
                                                    profile: AotBuildProfile::Release,
                                                },
                                            );
                                            let solver = bdf_bench_support::make_solver(
                                                workload, dimension, assembly,
                                            )
                                            .with_generated_backend_config(config);
                                            (solver, Some(artifact_dir))
                                        }
                                    }
                                },
                                |(solver, _artifact_dir)| {
                                    solver.try_generate().expect("cold preparation");
                                    black_box(solver.get_statistics());
                                },
                                BatchSize::SmallInput,
                            );
                        },
                    );

                    group.bench_function(
                        BenchmarkId::new(format!("prepared-solve/{input}"), "warm"),
                        |bencher| {
                            bencher.iter_batched_ref(
                                || {
                                    let mut solver =
                                        solver_for(workload, dimension, assembly, execution, aot);
                                    solver.try_generate().expect("prepare warm solve");
                                    solver
                                },
                                |solver| {
                                    solver.try_solve().expect("prepared solve");
                                    black_box(bdf_bench_support::final_state(&solver));
                                },
                                BatchSize::SmallInput,
                            );
                        },
                    );

                    group.bench_function(
                        BenchmarkId::new(format!("fresh-e2e/{input}"), "cold"),
                        |bencher| {
                            bencher.iter_batched_ref(
                                || -> (ODEsolver, Option<TempDir>) {
                                    match execution {
                                        Execution::Lambdify => (
                                            solver_for(
                                                workload, dimension, assembly, execution, None,
                                            ),
                                            None,
                                        ),
                                        Execution::Aot => {
                                            let isolated = tempfile::tempdir()
                                                .expect("isolated cold E2E AOT sample");
                                            let config = make_aot_config(
                                                isolated.path(),
                                                SymbolicIvpAotBuildPolicy::RebuildAlways {
                                                    profile: AotBuildProfile::Release,
                                                },
                                            );
                                            let solver = bdf_bench_support::make_solver(
                                                workload, dimension, assembly,
                                            )
                                            .with_generated_backend_config(config);
                                            (solver, Some(isolated))
                                        }
                                    }
                                },
                                |(solver, _artifact_dir)| {
                                    solver.try_solve().expect("fresh E2E solve");
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

fn benchmark_dense_full_execution_matrix(c: &mut Criterion) {
    if !matches!(
        std::env::var("BDF_BENCH_DENSE_BACKEND_MATRIX")
            .unwrap_or_else(|_| "off".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "on"
    ) {
        eprintln!(
            "[BDF dense backend matrix] skipped; set BDF_BENCH_DENSE_BACKEND_MATRIX=1 to opt in"
        );
        return;
    }

    let mut group = c.benchmark_group("bdf_dense_backend_apple_to_apple");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));
    let routes = [
        (IvpSymbolicAssemblyBackend::ExprLegacy, "expr-legacy"),
        (IvpSymbolicAssemblyBackend::AtomView, "atom-native"),
    ];

    for dimension in bdf_bench_support::dense_dimensions() {
        let mut reference = bdf_bench_support::make_dense_coupled_solver(
            dimension,
            IvpSymbolicAssemblyBackend::ExprLegacy,
            BdfTelemetryMode::Off,
        );
        reference.try_solve().expect("dense Lambdify reference");
        let reference_state = bdf_bench_support::final_state(&reference);

        for (assembly, assembly_name) in routes {
            let aot_context = prepare_dense_aot_context(dimension, assembly);
            for execution in [Execution::Lambdify, Execution::Aot] {
                let input = format!("n={dimension}/{assembly_name}/{}", execution.label());
                let aot = matches!(execution, Execution::Aot).then_some(&aot_context);
                let mut preflight = match execution {
                    Execution::Lambdify => bdf_bench_support::make_dense_coupled_solver(
                        dimension,
                        assembly,
                        BdfTelemetryMode::Off,
                    ),
                    Execution::Aot => bdf_bench_support::make_dense_coupled_solver(
                        dimension,
                        assembly,
                        BdfTelemetryMode::Off,
                    )
                    .with_generated_backend_config(aot.unwrap().config.clone()),
                };
                preflight
                    .try_solve()
                    .unwrap_or_else(|error| panic!("dense {input} preflight: {error}"));
                let parity = bdf_bench_support::final_state(&preflight)
                    .iter()
                    .zip(reference_state.iter())
                    .map(|(left, right)| (left - right).abs())
                    .fold(0.0_f64, f64::max);
                assert!(parity <= 2e-5, "dense {input} parity={parity:e}");
                eprintln!(
                    "[BDF dense backend preflight] dimension={dimension} assembly={assembly_name} execution={} final_diff={parity:.3e} status=ok",
                    execution.label()
                );

                group.bench_function(
                    BenchmarkId::new(format!("prepare/{input}"), "cold"),
                    |bencher| {
                        // Borrowed batches keep solver and artifact cleanup outside timing.
                        bencher.iter_batched_ref(
                            || {
                                let mut solver = bdf_bench_support::make_dense_coupled_solver(
                                    dimension,
                                    assembly,
                                    BdfTelemetryMode::Off,
                                );
                                let directory = if matches!(execution, Execution::Aot) {
                                    let directory = tempfile::tempdir()
                                        .expect("dense cold AOT sample directory");
                                    solver.set_generated_backend_config(make_aot_config(
                                        directory.path(),
                                        SymbolicIvpAotBuildPolicy::RebuildAlways {
                                            profile: AotBuildProfile::Release,
                                        },
                                    ));
                                    Some(directory)
                                } else {
                                    None
                                };
                                (solver, directory)
                            },
                            |(solver, _directory)| {
                                solver.try_generate().expect("dense cold preparation");
                                black_box(solver.get_statistics());
                            },
                            BatchSize::SmallInput,
                        );
                    },
                );
                group.bench_function(
                    BenchmarkId::new(format!("prepared-solve/{input}"), "warm"),
                    |bencher| {
                        bencher.iter_batched_ref(
                            || {
                                let mut solver = match execution {
                                    Execution::Lambdify => {
                                        bdf_bench_support::make_dense_coupled_solver(
                                            dimension,
                                            assembly,
                                            BdfTelemetryMode::Off,
                                        )
                                    }
                                    Execution::Aot => bdf_bench_support::make_dense_coupled_solver(
                                        dimension,
                                        assembly,
                                        BdfTelemetryMode::Off,
                                    )
                                    .with_generated_backend_config(aot.unwrap().config.clone()),
                                };
                                solver.try_generate().expect("dense warm preparation");
                                solver
                            },
                            |solver| {
                                solver.try_solve().expect("dense prepared solve");
                                black_box(bdf_bench_support::final_state(&solver));
                            },
                            BatchSize::SmallInput,
                        );
                    },
                );
                group.bench_function(
                    BenchmarkId::new(format!("fresh-e2e/{input}"), "cold"),
                    |bencher| {
                        bencher.iter_batched_ref(
                            || {
                                let mut solver = bdf_bench_support::make_dense_coupled_solver(
                                    dimension,
                                    assembly,
                                    BdfTelemetryMode::Off,
                                );
                                let directory = if matches!(execution, Execution::Aot) {
                                    let directory =
                                        tempfile::tempdir().expect("dense fresh AOT directory");
                                    solver.set_generated_backend_config(make_aot_config(
                                        directory.path(),
                                        SymbolicIvpAotBuildPolicy::RebuildAlways {
                                            profile: AotBuildProfile::Release,
                                        },
                                    ));
                                    Some(directory)
                                } else {
                                    None
                                };
                                (solver, directory)
                            },
                            |(solver, _directory)| {
                                solver.try_solve().expect("dense fresh E2E solve");
                                black_box(bdf_bench_support::final_state(&solver));
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

fn benchmark_dense_linear_kernel(c: &mut Criterion) {
    if !env_enabled("BDF_BENCH_RUN_LINEAR_KERNEL", false) {
        eprintln!("[BDF dense linear kernel] skipped; set BDF_BENCH_RUN_LINEAR_KERNEL=1 to opt in");
        return;
    }

    let mut group = c.benchmark_group("bdf_dense_linear_kernel");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(2));

    for dimension in dense_linear_dimensions() {
        let jacobian_matrix = dense_linear_fixture(dimension);
        let jacobian = BdfJacobian::from_dense(jacobian_matrix.clone());
        let shifted = jacobian
            .to_shifted_dense(0.125)
            .expect("dense shifted matrix assembly");
        let faer_shifted =
            faer::Mat::<f64>::from_fn(dimension, dimension, |row, col| shifted[(row, col)]);
        let nalgebra_factor = LU::new(shifted.clone());
        let nalgebra_rhs = DVector::from_fn(dimension, |row, _| 1.0 + row as f64 * 1e-3);
        let faer_factor = faer_shifted.as_ref().partial_piv_lu();
        let faer_rhs = faer::Col::<f64>::from_fn(dimension, |row| 1.0 + row as f64 * 1e-3);

        group.bench_function(BenchmarkId::new("matrix-clone", dimension), |bencher| {
            bencher.iter(|| black_box(black_box(&shifted).clone()));
        });
        group.bench_function(
            BenchmarkId::new("shifted-matrix-assembly", dimension),
            |bencher| {
                bencher.iter(|| {
                    black_box(
                        black_box(&jacobian)
                            .to_shifted_dense(0.125)
                            .expect("dense shifted matrix assembly"),
                    )
                });
            },
        );
        group.bench_function(
            BenchmarkId::new("nalgebra-lu-owned", dimension),
            |bencher| {
                bencher.iter_batched(
                    || shifted.clone(),
                    |matrix| black_box(LU::new(matrix)),
                    BatchSize::SmallInput,
                );
            },
        );
        group.bench_function(
            BenchmarkId::new("nalgebra-lu-clone-plus-factor", dimension),
            |bencher| {
                bencher.iter(|| black_box(LU::new(black_box(&shifted).clone())));
            },
        );
        group.bench_function(
            BenchmarkId::new("faer-partial-piv-lu", dimension),
            |bencher| {
                bencher.iter(|| black_box(black_box(&faer_shifted).as_ref().partial_piv_lu()));
            },
        );
        group.bench_function(BenchmarkId::new("nalgebra-solve", dimension), |bencher| {
            bencher.iter(|| {
                black_box(
                    nalgebra_factor
                        .solve(black_box(&nalgebra_rhs))
                        .expect("nonsingular nalgebra fixture"),
                )
            });
        });
        group.bench_function(BenchmarkId::new("faer-solve", dimension), |bencher| {
            bencher.iter(|| black_box(faer_factor.solve(black_box(faer_rhs.as_ref()))));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_callback_stages,
    benchmark_full_execution_matrix,
    benchmark_dense_full_execution_matrix,
    benchmark_dense_linear_kernel
);
criterion_main!(benches);
