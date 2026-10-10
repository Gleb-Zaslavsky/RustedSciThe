//! Compact BDF workload matrix and bounded Criterion smoke benchmark.
//!
//! Set `BDF_BENCH_COMPACT_REPORT=1` to write one reviewable table through the
//! shared report helper. Without compact mode this target provides a small
//! repeated Criterion view; the historical BDF targets remain available for
//! detailed stage-specific experiments.

use std::hint::black_box;
use std::time::{Duration, Instant};

use RustedSciThe::numerical::BDF::BDF_api::{BdfTelemetryMode, ODEsolver};
use RustedSciThe::numerical::ivp_workloads::{
    WorkloadKind, build_workload, parameter_continuation_target,
};
use RustedSciThe::symbolic::codegen::codegen_runtime_api::{
    recommended_dense_jacobian_chunking_for_parallelism,
    recommended_residual_chunking_for_parallelism,
};
use RustedSciThe::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use RustedSciThe::symbolic::ivp_telemetry::IvpLambdifyExecutionPolicy;
use RustedSciThe::symbolic::symbolic_ivp::IvpSymbolicAssemblyBackend;
use RustedSciThe::symbolic::symbolic_ivp::SymbolicIvpAotOptions;
use RustedSciThe::symbolic::symbolic_ivp_generated::{
    SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use tabled::{Table, Tabled};
use tempfile::TempDir;

#[path = "support/bdf_bench_support.rs"]
mod bdf_bench_support;

#[derive(Debug, Tabled)]
struct BdfCompactRow {
    workload: String,
    dimension: String,
    frontend: String,
    prepare_ms: String,
    parallel_calibration_ms: String,
    prepare_without_calibration_ms: String,
    solve_ms: String,
    continuation_ms: String,
    residual_calls: usize,
    jacobian_calls: usize,
    factorizations: usize,
    accepted_steps: usize,
    max_final_abs: String,
    parity_diff: String,
    status: String,
}

fn frontend_label(assembly: IvpSymbolicAssemblyBackend) -> &'static str {
    match assembly {
        IvpSymbolicAssemblyBackend::ExprLegacy => "expr-legacy",
        IvpSymbolicAssemblyBackend::AtomView => "atom-native",
        IvpSymbolicAssemblyBackend::AtomViewExprCompat => "atom-expr-compat",
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CompactExecution {
    Lambdify(IvpLambdifyExecutionPolicy),
    AotWhole,
    AotParallel2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CompactRoute {
    assembly: IvpSymbolicAssemblyBackend,
    execution: CompactExecution,
}

fn compact_routes() -> Vec<CompactRoute> {
    std::env::var("BDF_BENCH_COMPACT_ROUTES")
        .unwrap_or_else(|_| "lambdify".to_owned())
        .split(',')
        .filter_map(|item| match item.trim().to_ascii_lowercase().as_str() {
            "lambdify" | "lambdify-sequential" => {
                Some(CompactExecution::Lambdify(IvpLambdifyExecutionPolicy::Sequential))
            }
            "lambdify-parallel" => Some(CompactExecution::Lambdify(
                IvpLambdifyExecutionPolicy::Parallel { min_work: 1 },
            )),
            "lambdify-auto" => Some(CompactExecution::Lambdify(
                IvpLambdifyExecutionPolicy::Auto { min_work: 1 },
            )),
            "aot" | "aot-whole" => Some(CompactExecution::AotWhole),
            "aot-parallel2" => Some(CompactExecution::AotParallel2),
            "" => None,
            other => panic!("unknown BDF_BENCH_COMPACT_ROUTES item {other:?}; expected lambdify-sequential, lambdify-parallel, lambdify-auto, aot-whole or aot-parallel2"),
        })
        .flat_map(|execution| {
            [
                CompactRoute {
                    assembly: IvpSymbolicAssemblyBackend::ExprLegacy,
                    execution,
                },
                CompactRoute {
                    assembly: IvpSymbolicAssemblyBackend::AtomView,
                    execution,
                },
            ]
        })
        .collect()
}

fn route_label(route: CompactRoute) -> String {
    let frontend = frontend_label(route.assembly);
    match route.execution {
        CompactExecution::Lambdify(policy) => format!("{frontend}-{}", policy.label()),
        CompactExecution::AotWhole => format!("{frontend}-aot-whole"),
        CompactExecution::AotParallel2 => format!("{frontend}-aot-parallel2"),
    }
}

fn route_is_aot(route: CompactRoute) -> bool {
    matches!(
        route.execution,
        CompactExecution::AotWhole | CompactExecution::AotParallel2
    )
}

fn route_lambdify_policy(route: CompactRoute) -> IvpLambdifyExecutionPolicy {
    match route.execution {
        CompactExecution::Lambdify(policy) => policy,
        CompactExecution::AotWhole | CompactExecution::AotParallel2 => {
            IvpLambdifyExecutionPolicy::Sequential
        }
    }
}

fn route_aot_options(route: CompactRoute, variable_count: usize) -> SymbolicIvpAotOptions {
    match route.execution {
        CompactExecution::AotWhole => SymbolicIvpAotOptions::default(),
        CompactExecution::AotParallel2 => SymbolicIvpAotOptions {
            residual_strategy: recommended_residual_chunking_for_parallelism(variable_count, 2),
            jacobian_strategy: recommended_dense_jacobian_chunking_for_parallelism(
                variable_count,
                2,
            ),
        },
        CompactExecution::Lambdify(_) => SymbolicIvpAotOptions::default(),
    }
}

fn aot_config(
    artifact_dir: &TempDir,
    require_prebuilt: bool,
    route: CompactRoute,
    variable_count: usize,
) -> SymbolicIvpGeneratedBackendConfig {
    let policy = if require_prebuilt {
        SymbolicIvpAotBuildPolicy::RequirePrebuilt
    } else {
        SymbolicIvpAotBuildPolicy::RebuildAlways {
            profile: AotBuildProfile::Release,
        }
    };
    SymbolicIvpGeneratedBackendConfig::new()
        .with_output_parent_dir(Some(artifact_dir.path().to_path_buf()))
        .with_build_policy(policy)
        .with_aot_options(route_aot_options(route, variable_count))
        .with_c_tcc()
}

fn attach_route(
    solver: ODEsolver,
    route: CompactRoute,
    artifact_dir: Option<&TempDir>,
    require_prebuilt: bool,
    variable_count: usize,
) -> ODEsolver {
    match (route_is_aot(route), artifact_dir) {
        (true, Some(directory)) => solver.with_generated_backend_config(aot_config(
            directory,
            require_prebuilt,
            route,
            variable_count,
        )),
        (true, None) => panic!("AOT compact route requires an isolated artifact directory"),
        (false, _) => solver,
    }
}

fn continuation_counts() -> Vec<usize> {
    std::env::var("BDF_COMPACT_CONTINUATION_COUNTS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|item| item.trim().parse::<usize>().ok())
                .filter(|count| *count > 0)
                .collect::<Vec<_>>()
        })
        .filter(|counts| !counts.is_empty())
        .unwrap_or_default()
}

fn continuation_width(kind: WorkloadKind) -> f64 {
    match kind {
        WorkloadKind::CombustionLike => 0.01,
        WorkloadKind::DiffusionChain => 0.01,
        WorkloadKind::ThreeBody => 0.01,
        WorkloadKind::StiffScalar => 0.05,
        WorkloadKind::Robertson => 0.1,
    }
}

fn compact_dimension(kind: WorkloadKind, dimension: usize) -> String {
    if matches!(kind, WorkloadKind::DiffusionChain) {
        dimension.to_string()
    } else {
        "fixed".to_owned()
    }
}

fn max_abs_state(solver: &ODEsolver) -> f64 {
    let (_, trajectory) = solver.get_result_ref();
    trajectory
        .row(trajectory.nrows() - 1)
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max)
}

fn final_state(solver: &ODEsolver) -> Vec<f64> {
    let (_, trajectory) = solver.get_result_ref();
    trajectory
        .row(trajectory.nrows() - 1)
        .iter()
        .copied()
        .collect()
}

fn parallel_calibration_ms(solver: &ODEsolver) -> Option<f64> {
    solver.preparation_backend_telemetry().map(|snapshot| {
        snapshot
            .cold_stage(RustedSciThe::symbolic::ivp_telemetry::IvpColdStage::ParallelCalibration)
            .elapsed
            .as_secs_f64()
            * 1_000.0
    })
}

fn prepare_without_calibration_ms(prepare_ms: f64, calibration_ms: Option<f64>) -> f64 {
    calibration_ms
        .map(|calibration| (prepare_ms - calibration).max(0.0))
        .unwrap_or(prepare_ms)
}

fn run_compact_report() {
    let workloads = bdf_bench_support::workloads_from_env(
        "BDF_BENCH_COMPACT_WORKLOADS",
        "stiff-scalar,robertson,combustion-like,diffusion-chain",
    );
    let mut rows = Vec::new();

    for workload in workloads {
        for dimension in bdf_bench_support::dimensions(workload) {
            let mut route_states = Vec::new();
            let mut route_rows = Vec::new();

            for route in compact_routes() {
                let artifact_dir = route_is_aot(route)
                    .then(|| tempfile::tempdir().expect("isolated BDF compact AOT directory"));
                let mut solver = bdf_bench_support::make_solver_with_telemetry_and_policy(
                    workload,
                    dimension,
                    route.assembly,
                    BdfTelemetryMode::Timings,
                    route_lambdify_policy(route),
                );
                solver = attach_route(
                    solver,
                    route,
                    artifact_dir.as_ref(),
                    false,
                    dimension.max(1),
                );
                let prepare_started = Instant::now();
                let preparation = solver.try_generate();
                let prepare_ms = prepare_started.elapsed().as_secs_f64() * 1_000.0;
                if let Err(error) = preparation {
                    route_rows.push((
                        route,
                        BdfCompactRow {
                            workload: workload.label().to_owned(),
                            dimension: compact_dimension(workload, dimension),
                            frontend: route_label(route),
                            prepare_ms: format!("{prepare_ms:.3}"),
                            parallel_calibration_ms: "-".to_owned(),
                            prepare_without_calibration_ms: "-".to_owned(),
                            solve_ms: "-".to_owned(),
                            continuation_ms: "-".to_owned(),
                            residual_calls: 0,
                            jacobian_calls: 0,
                            factorizations: 0,
                            accepted_steps: 0,
                            max_final_abs: "-".to_owned(),
                            parity_diff: "-".to_owned(),
                            status: format!("prepare-error: {error}"),
                        },
                    ));
                    continue;
                }

                let solve_started = Instant::now();
                let solve = solver.try_solve();
                let solve_ms = solve_started.elapsed().as_secs_f64() * 1_000.0;
                if let Err(error) = solve {
                    route_rows.push((
                        route,
                        BdfCompactRow {
                            workload: workload.label().to_owned(),
                            dimension: compact_dimension(workload, dimension),
                            frontend: route_label(route),
                            prepare_ms: format!("{prepare_ms:.3}"),
                            parallel_calibration_ms: "-".to_owned(),
                            prepare_without_calibration_ms: "-".to_owned(),
                            solve_ms: format!("{solve_ms:.3}"),
                            continuation_ms: "-".to_owned(),
                            residual_calls: 0,
                            jacobian_calls: 0,
                            factorizations: 0,
                            accepted_steps: 0,
                            max_final_abs: "-".to_owned(),
                            parity_diff: "-".to_owned(),
                            status: format!("solve-error: {error}"),
                        },
                    ));
                    continue;
                }

                let initial_final_state = final_state(&solver);
                route_states.push((route, initial_final_state));
                let stats = solver.get_statistics();
                let calibration_ms = parallel_calibration_ms(&solver);
                let prepare_without_calibration =
                    prepare_without_calibration_ms(prepare_ms, calibration_ms);
                let mut continuation_ms = "-".to_owned();
                let mut status = "ok".to_owned();

                let parameter_values = build_workload(workload, dimension.max(1)).parameter_values;
                if !parameter_values.is_empty() {
                    let mut measurements = Vec::new();
                    for count in continuation_counts() {
                        let mut continuation_solver =
                            bdf_bench_support::make_solver_with_telemetry_and_policy(
                                workload,
                                dimension,
                                route.assembly,
                                BdfTelemetryMode::Timings,
                                route_lambdify_policy(route),
                            );
                        continuation_solver = attach_route(
                            continuation_solver,
                            route,
                            artifact_dir.as_ref(),
                            route_is_aot(route),
                            dimension.max(1),
                        );
                        if let Err(error) = continuation_solver.try_generate() {
                            status = format!("continuation-seed-prepare-error: {error}");
                            break;
                        }
                        if let Err(error) = continuation_solver.try_solve() {
                            status = format!("continuation-seed-solve-error: {error}");
                            break;
                        }
                        let started = Instant::now();
                        let width = continuation_width(workload);
                        for index in 1..=count {
                            let (times, _) = continuation_solver.get_result_ref();
                            let t0 = *times.as_slice().last().expect("BDF segment time");
                            let parameters =
                                parameter_continuation_target(&parameter_values, index);
                            if let Err(error) = continuation_solver
                                .try_continue_with_parameter_values(parameters, t0 + width)
                                .and_then(|_| continuation_solver.try_solve())
                            {
                                status = format!("continuation-error: {error}");
                                break;
                            }
                        }
                        measurements.push(format!(
                            "{count}:{:.3}",
                            started.elapsed().as_secs_f64() * 1_000.0
                        ));
                        if status != "ok" {
                            break;
                        }
                    }
                    if !measurements.is_empty() {
                        continuation_ms = measurements.join(";");
                    }
                }

                route_rows.push((
                    route,
                    BdfCompactRow {
                        workload: workload.label().to_owned(),
                        dimension: compact_dimension(workload, dimension),
                        frontend: route_label(route),
                        prepare_ms: format!("{prepare_ms:.3}"),
                        parallel_calibration_ms: calibration_ms
                            .map(|value| format!("{value:.3}"))
                            .unwrap_or_else(|| "-".to_owned()),
                        prepare_without_calibration_ms: format!("{prepare_without_calibration:.3}"),
                        solve_ms: format!("{solve_ms:.3}"),
                        continuation_ms,
                        residual_calls: stats.residual_calls,
                        jacobian_calls: stats.jacobian_calls,
                        factorizations: stats.bdf_nlu_total,
                        accepted_steps: stats.accepted_steps_total,
                        max_final_abs: format!("{:.6e}", max_abs_state(&solver)),
                        parity_diff: "pending".to_owned(),
                        status,
                    },
                ));
            }

            for (route, row) in &mut route_rows {
                let reference = route_states
                    .iter()
                    .find(|(candidate, _)| {
                        candidate.execution == route.execution
                            && candidate.assembly != route.assembly
                    })
                    .map(|(_, state)| state);
                let candidate = route_states
                    .iter()
                    .find(|(candidate, _)| candidate == route)
                    .map(|(_, state)| state);
                row.parity_diff = match (reference, candidate) {
                    (Some(reference), Some(candidate)) => {
                        let diff = reference
                            .iter()
                            .zip(candidate.iter())
                            .map(|(left, right)| (left - right).abs())
                            .fold(0.0, f64::max);
                        format!("{diff:.3e}")
                    }
                    _ => "-".to_owned(),
                };
            }
            rows.extend(route_rows.into_iter().map(|(_, row)| row));
        }
    }

    let metadata = format!(
        "profile=bench; compact=true; routes={:?}; workloads={:?}; continuation_counts={:?}; telemetry=timings; scopes=diagnostic/non-additive",
        std::env::var("BDF_BENCH_COMPACT_ROUTES").unwrap_or_else(|_| "lambdify".to_owned()),
        std::env::var("BDF_BENCH_COMPACT_WORKLOADS").unwrap_or_else(|_| "default".to_owned()),
        continuation_counts()
    );
    let body = format!("- {metadata}\n\n{}", Table::new(rows));
    let name = std::env::var("BDF_COMPACT_REPORT_NAME")
        .unwrap_or_else(|_| "bdf_workload_matrix".to_owned());
    let path = RustedSciThe::Utils::test_reporting::write_test_report("BDF_Bench", &name, &body)
        .expect("write BDF compact report");
    println!("[BDF compact benchmark] report={}", path.display());
}

fn criterion_sample_size() -> usize {
    std::env::var("BDF_BENCH_SAMPLE_SIZE")
        .ok()
        .and_then(|value| value.parse().ok())
        .map(|value: usize| value.max(10))
        .unwrap_or(10)
}

fn criterion_measurement_time() -> Duration {
    Duration::from_secs(
        std::env::var("BDF_BENCH_MEASUREMENT_SECONDS")
            .ok()
            .and_then(|value| value.parse().ok())
            .map(|value: u64| value.max(1))
            .unwrap_or(1),
    )
}

fn benchmark_bdf_workloads(c: &mut Criterion) {
    if std::env::var("BDF_BENCH_COMPACT_REPORT").as_deref() == Ok("1") {
        run_compact_report();
        return;
    }

    let mut group = c.benchmark_group("bdf_workloads");
    group.sample_size(criterion_sample_size());
    group.measurement_time(criterion_measurement_time());
    for workload in bdf_bench_support::workloads_from_env(
        "BDF_BENCH_WORKLOADS",
        "stiff-scalar,robertson,combustion-like",
    ) {
        for dimension in bdf_bench_support::dimensions(workload) {
            for assembly in [
                IvpSymbolicAssemblyBackend::ExprLegacy,
                IvpSymbolicAssemblyBackend::AtomView,
            ] {
                let label = format!(
                    "{}/{}/{}/warm-solve",
                    workload.label(),
                    compact_dimension(workload, dimension),
                    frontend_label(assembly)
                );
                group.bench_function(BenchmarkId::new("solve", label), |bencher| {
                    bencher.iter_batched(
                        || {
                            let mut solver =
                                bdf_bench_support::make_solver(workload, dimension, assembly);
                            solver.try_generate().expect("BDF workload preparation");
                            solver
                        },
                        |mut solver| black_box(solver.try_solve().expect("BDF workload solve")),
                        criterion::BatchSize::SmallInput,
                    );
                });
            }
        }
    }
    group.finish();
}

criterion_group!(benches, benchmark_bdf_workloads);
criterion_main!(benches);
