//! Bounded Radau workload matrix.
//!
//! The benchmark uses the same shared IVP fixtures as LSODE2 and BDF. It is
//! deliberately small enough for debug smoke checks; the release campaign can
//! widen dimensions through a later profile-aware harness.

use std::hint::black_box;
use std::time::Duration;
use std::time::Instant;

use RustedSciThe::numerical::Radau::benchmark::{
    BenchmarkAssembly, BenchmarkExecutionPolicy, BenchmarkLayout, PreparedAotCallbacks,
    PreparedBenchmark,
};
use RustedSciThe::numerical::Radau::{
    RadauAotConfig, RadauConfig, RadauExecution, RadauExecutionPolicy, RadauFrontend,
    RadauMatrixLayout, RadauProblem, RadauSolver, RadauTelemetryMode,
};
use RustedSciThe::numerical::ivp_workloads::{WorkloadKind, parameter_continuation_target};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use tabled::{Table, Tabled};

fn canonical_cases() -> Vec<(WorkloadKind, usize)> {
    let mut cases = vec![
        (WorkloadKind::StiffScalar, 1),
        (WorkloadKind::Robertson, 3),
        (WorkloadKind::CombustionLike, 3),
        (WorkloadKind::ThreeBody, 12),
    ];
    cases.extend(
        dimensions_from_env("RADAU_BENCH_DIFFUSION_DIMENSIONS", &[8, 16])
            .into_iter()
            .map(|dimension| (WorkloadKind::DiffusionChain, dimension)),
    );
    cases
}

fn dimensions_from_env(name: &str, default: &[usize]) -> Vec<usize> {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|item| item.trim().parse::<usize>().ok())
                .filter(|dimension| *dimension > 0)
                .collect::<Vec<_>>()
        })
        .filter(|dimensions| !dimensions.is_empty())
        .unwrap_or_else(|| default.to_vec())
}

fn continuation_counts() -> Vec<usize> {
    std::env::var("RADAU_BENCH_CONTINUATION_COUNTS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|item| item.trim().parse::<usize>().ok())
                .filter(|count| *count > 0)
                .collect::<Vec<_>>()
        })
        .filter(|counts| !counts.is_empty())
        .unwrap_or_else(|| vec![4])
}

/// Number of Criterion samples for the non-compact statistical profile.
///
/// Compact release reports intentionally execute one bounded measurement per
/// row. Criterion uses this value only when the compact early-return path is
/// disabled, so an overnight statistical run can increase confidence without
/// changing the release evidence format.
fn criterion_sample_size() -> usize {
    std::env::var("RADAU_BENCH_SAMPLE_SIZE")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .map(|value| value.max(10))
        .unwrap_or(10)
}

/// Wall-clock measurement window for one Criterion benchmark group.
fn criterion_measurement_time() -> Duration {
    Duration::from_secs(
        std::env::var("RADAU_BENCH_MEASUREMENT_SECONDS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .map(|value| value.max(1))
            .unwrap_or(1),
    )
}

fn policy_diffusion_dimensions() -> Vec<usize> {
    dimensions_from_env("RADAU_BENCH_POLICY_DIFFUSION_DIMENSIONS", &[16])
}

fn worker_label() -> String {
    std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".to_owned())
}

fn effective_worker_label() -> String {
    rayon::current_num_threads().to_string()
}

fn aot_cases() -> Vec<(WorkloadKind, usize)> {
    let mut cases = vec![
        (WorkloadKind::StiffScalar, 1),
        (WorkloadKind::Robertson, 3),
        (WorkloadKind::CombustionLike, 3),
        (WorkloadKind::ThreeBody, 12),
    ];
    cases.extend(
        dimensions_from_env("RADAU_BENCH_AOT_DIFFUSION_DIMENSIONS", &[16])
            .into_iter()
            .map(|dimension| (WorkloadKind::DiffusionChain, dimension)),
    );
    cases
}

fn route_label(assembly: BenchmarkAssembly, layout: BenchmarkLayout) -> String {
    let frontend = match assembly {
        BenchmarkAssembly::ExprLegacy => "expr-legacy",
        BenchmarkAssembly::AtomViewNative => "atom-native",
    };
    let matrix = match layout {
        BenchmarkLayout::Dense => "dense".to_string(),
        BenchmarkLayout::Sparse => "sparse".to_string(),
        BenchmarkLayout::Banded { lower, upper } => format!("banded-{lower}-{upper}"),
    };
    format!("{frontend}/{matrix}")
}

fn benchmark_radau_workloads(c: &mut Criterion) {
    if std::env::var("RADAU_BENCH_COMPACT_REPORT").as_deref() == Ok("1") {
        run_compact_report();
        return;
    }
    let mut group = c.benchmark_group("radau_workloads");
    group.sample_size(criterion_sample_size());
    group.measurement_time(criterion_measurement_time());

    for (workload, dimension) in canonical_cases() {
        let layouts = if workload == WorkloadKind::DiffusionChain {
            vec![
                BenchmarkLayout::Dense,
                BenchmarkLayout::Sparse,
                BenchmarkLayout::Banded { lower: 1, upper: 1 },
                BenchmarkLayout::Banded { lower: 2, upper: 1 },
            ]
        } else {
            vec![BenchmarkLayout::Dense]
        };
        for assembly in [
            BenchmarkAssembly::ExprLegacy,
            BenchmarkAssembly::AtomViewNative,
        ] {
            for layout in layouts.iter().copied() {
                let route = route_label(assembly, layout);
                let id = format!("{}/{}/{}", workload.label(), dimension, route);

                group.bench_with_input(
                    BenchmarkId::new("cold-prepare", &id),
                    &(workload, dimension, assembly, layout),
                    |bencher, &(workload, dimension, assembly, layout)| {
                        bencher.iter_batched(
                            || (workload, dimension, assembly, layout),
                            |(workload, dimension, assembly, layout)| {
                                let prepared = PreparedBenchmark::prepare(
                                    workload, dimension, assembly, layout,
                                )
                                .expect("Radau diagnostic preparation");
                                black_box(prepared);
                            },
                            BatchSize::SmallInput,
                        );
                    },
                );

                let prepared = PreparedBenchmark::prepare(workload, dimension, assembly, layout)
                    .expect("Radau diagnostic preparation");
                group.bench_with_input(
                    BenchmarkId::new("warm-callbacks", &id),
                    &id,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(
                                prepared
                                    .warm_callbacks(64)
                                    .expect("Radau warm callback evaluation"),
                            );
                        });
                    },
                );
                group.bench_with_input(BenchmarkId::new("full-solve", &id), &id, |bencher, _| {
                    bencher.iter(|| {
                        black_box(prepared.solve().expect("Radau diagnostic solve"));
                    });
                });
                group.bench_with_input(
                    BenchmarkId::new("linear-kernel", &id),
                    &id,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(
                                prepared
                                    .linear_kernel(8)
                                    .expect("Radau linear backend diagnostic"),
                            );
                        });
                    },
                );
                if workload != WorkloadKind::StiffScalar {
                    for count in continuation_counts() {
                        group.bench_with_input(
                            BenchmarkId::new(format!("continuation-{count}"), &id),
                            &id,
                            |bencher, _| {
                                bencher.iter(|| {
                                    black_box(
                                        prepared
                                            .continuation(count)
                                            .expect("Radau continuation diagnostic"),
                                    );
                                });
                            },
                        );
                    }
                }
            }
        }
    }
    group.finish();

    if std::env::var("RADAU_BENCH_POLICY_MATRIX").as_deref() == Ok("1") {
        benchmark_radau_policy_matrix(c);
    }

    if std::env::var("RADAU_BENCH_AOT").as_deref() == Ok("1") {
        benchmark_radau_aot(c);
    }
}

#[derive(Debug, Tabled)]
struct CompactBenchmarkRow {
    route: String,
    workload: String,
    dimension: usize,
    layout: String,
    policy: String,
    cold_prepare_ms: String,
    callback_ms: String,
    residual_ms: String,
    jacobian_ms: String,
    full_solve_ms: String,
    continuation_ms: String,
    checksum: String,
    status: String,
}

fn compact_dimensions() -> Vec<usize> {
    dimensions_from_env("RADAU_COMPACT_DIMENSIONS", &[8, 16, 32])
}

fn compact_workloads() -> Vec<WorkloadKind> {
    let requested = match std::env::var("RADAU_COMPACT_WORKLOADS") {
        Ok(value) => value,
        Err(_) => return WorkloadKind::ALL.to_vec(),
    };
    let selected = requested
        .split(',')
        .filter_map(|item| match item.trim() {
            "diffusion-chain" => Some(WorkloadKind::DiffusionChain),
            "combustion-like" => Some(WorkloadKind::CombustionLike),
            "stiff-scalar" => Some(WorkloadKind::StiffScalar),
            "robertson" => Some(WorkloadKind::Robertson),
            "three-body" => Some(WorkloadKind::ThreeBody),
            _ => None,
        })
        .collect::<Vec<_>>();
    if selected.is_empty() {
        WorkloadKind::ALL.to_vec()
    } else {
        selected
    }
}

fn compact_policies() -> Vec<BenchmarkExecutionPolicy> {
    let requested =
        std::env::var("RADAU_COMPACT_POLICIES").unwrap_or_else(|_| "sequential".to_owned());
    let mut policies = Vec::new();
    for item in requested.split(',').map(str::trim) {
        match item {
            "all" => {
                policies.extend([
                    BenchmarkExecutionPolicy::Sequential,
                    BenchmarkExecutionPolicy::Parallel { min_work: 1 },
                    BenchmarkExecutionPolicy::Auto { min_work: 1 },
                ]);
            }
            "sequential" => policies.push(BenchmarkExecutionPolicy::Sequential),
            "parallel" => policies.push(BenchmarkExecutionPolicy::Parallel { min_work: 1 }),
            "auto" => policies.push(BenchmarkExecutionPolicy::Auto { min_work: 1 }),
            _ => {}
        }
    }
    if policies.is_empty() {
        policies.push(BenchmarkExecutionPolicy::Sequential);
    }
    policies
}

fn compact_layouts(workload: WorkloadKind) -> Vec<BenchmarkLayout> {
    if workload == WorkloadKind::DiffusionChain {
        vec![
            BenchmarkLayout::Dense,
            BenchmarkLayout::Sparse,
            BenchmarkLayout::Banded { lower: 1, upper: 1 },
        ]
    } else {
        vec![BenchmarkLayout::Dense]
    }
}

fn compact_layout_label(layout: BenchmarkLayout) -> String {
    match layout {
        BenchmarkLayout::Dense => "dense".to_owned(),
        BenchmarkLayout::Sparse => "sparse".to_owned(),
        BenchmarkLayout::Banded { lower, upper } => format!("banded-{lower}-{upper}"),
    }
}

fn compact_ms(started: Instant) -> String {
    format!("{:.3}", started.elapsed().as_secs_f64() * 1_000.0)
}

fn compact_report_name() -> String {
    std::env::var("RADAU_COMPACT_REPORT_NAME").unwrap_or_else(|_| "radau_compact_matrix".to_owned())
}

fn compact_report_status(rows: &[CompactBenchmarkRow], complete: bool) -> &'static str {
    if rows.is_empty() {
        "empty"
    } else if rows.iter().all(|row| row.status == "ok") {
        if complete { "completed" } else { "in_progress" }
    } else if complete {
        "completed_with_errors"
    } else {
        "in_progress_with_errors"
    }
}

fn compact_report_body(
    rows: &[CompactBenchmarkRow],
    _continuation_count: usize,
    complete: bool,
) -> String {
    let table = Table::new(rows).to_string();
    let continuation_counts = continuation_counts()
        .into_iter()
        .map(|count| count.to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "report_version: 3\nstatus: {}\nprofile: {}\nhost_os: {}\nhost_arch: {}\ncompiler: {}\nrow_count: {}\nconfigured_worker_count: {}\neffective_worker_count: {}\ncontinuation_count: {}\n\n{}",
        compact_report_status(rows, complete),
        std::env::var("RST_TEST_REPORT_PROFILE").unwrap_or_else(|_| "bench".to_owned()),
        std::env::consts::OS,
        std::env::consts::ARCH,
        if std::env::var("RADAU_COMPACT_AOT").as_deref() == Ok("1") {
            "tcc-or-configured-aot"
        } else {
            "not-applicable"
        },
        rows.len(),
        worker_label(),
        effective_worker_label(),
        continuation_counts,
        table
    )
}

fn write_compact_progress(rows: &[CompactBenchmarkRow], continuation_count: usize) {
    let body = compact_report_body(rows, continuation_count, false);
    if let Err(error) = RustedSciThe::Utils::test_reporting::write_test_report_snapshot(
        "Radau_Bench",
        &compact_report_name(),
        &body,
    ) {
        eprintln!("[Radau compact progress] report_write_error={error}");
    }
}

fn record_compact_row(
    rows: &mut Vec<CompactBenchmarkRow>,
    row: CompactBenchmarkRow,
    continuation_count: usize,
    report_started: Instant,
) {
    rows.push(row);
    let row = rows.last().expect("just-recorded compact row");
    println!(
        "[Radau compact progress] rows={} route={} workload={} dimension={} layout={} policy={} status={} elapsed_ms={:.0}",
        rows.len(),
        row.route,
        row.workload,
        row.dimension,
        row.layout,
        row.policy,
        row.status,
        report_started.elapsed().as_secs_f64() * 1_000.0,
    );
    write_compact_progress(rows, continuation_count);
}

/// Produce one compact, profile-aware report without Criterion's warm-up
/// chatter. This is intended for overnight matched matrices; statistical
/// claims still belong to the ordinary Criterion groups.
fn run_compact_report() {
    let report_started = Instant::now();
    let mut cases = Vec::new();
    for workload in compact_workloads() {
        match workload {
            WorkloadKind::DiffusionChain => cases.extend(
                compact_dimensions()
                    .into_iter()
                    .map(|dimension| (workload, dimension)),
            ),
            WorkloadKind::Robertson | WorkloadKind::CombustionLike => cases.push((workload, 3)),
            WorkloadKind::ThreeBody => cases.push((workload, 12)),
            WorkloadKind::StiffScalar => cases.push((workload, 1)),
        }
    }
    let policies = compact_policies();
    let continuation_count = continuation_counts().into_iter().max().unwrap_or(4);
    let mut rows = Vec::new();

    for (workload, dimension) in cases {
        for layout in compact_layouts(workload) {
            for policy in policies.iter().copied() {
                let layout_label = compact_layout_label(layout);
                let policy_label = policy.label().to_owned();

                for assembly in [
                    BenchmarkAssembly::ExprLegacy,
                    BenchmarkAssembly::AtomViewNative,
                ] {
                    let route = format!("lambdify/{}", route_label(assembly, layout));
                    let started = Instant::now();
                    let prepared = match PreparedBenchmark::prepare_with_policy(
                        workload, dimension, assembly, layout, policy,
                    ) {
                        Ok(prepared) => prepared,
                        Err(error) => {
                            record_compact_row(
                                &mut rows,
                                CompactBenchmarkRow {
                                    route,
                                    workload: workload.label().to_owned(),
                                    dimension,
                                    layout: layout_label.clone(),
                                    policy: policy_label.clone(),
                                    cold_prepare_ms: compact_ms(started),
                                    callback_ms: "-".to_owned(),
                                    residual_ms: "-".to_owned(),
                                    jacobian_ms: "-".to_owned(),
                                    full_solve_ms: "-".to_owned(),
                                    continuation_ms: "-".to_owned(),
                                    checksum: "-".to_owned(),
                                    status: format!("error: {error}"),
                                },
                                continuation_count,
                                report_started,
                            );
                            continue;
                        }
                    };
                    let cold_prepare_ms = compact_ms(started);
                    let started = Instant::now();
                    let callback = prepared.warm_callback_breakdown(32);
                    let callback_ms = callback
                        .as_ref()
                        .map(|value| format!("{:.3}", value.callback_ms))
                        .unwrap_or_else(|_| compact_ms(started));
                    let residual_ms = callback
                        .as_ref()
                        .map(|value| format!("{:.3}", value.residual_ms))
                        .unwrap_or_else(|_| "-".to_owned());
                    let jacobian_ms = callback
                        .as_ref()
                        .map(|value| format!("{:.3}", value.jacobian_ms))
                        .unwrap_or_else(|_| "-".to_owned());
                    let started = Instant::now();
                    let solve = prepared.solve();
                    let full_solve_ms = compact_ms(started);
                    let mut continuation_results = Vec::new();
                    let mut continuation_checksum = 0.0;
                    let mut continuation_ok = true;
                    for count in continuation_counts() {
                        let started = Instant::now();
                        match prepared.continuation(count) {
                            Ok(value) => {
                                continuation_checksum += value;
                                continuation_results.push(format!(
                                    "{count}:{:.3}",
                                    started.elapsed().as_secs_f64() * 1_000.0
                                ));
                            }
                            Err(_) => {
                                continuation_ok = false;
                                continuation_results.push(format!("{count}:error"));
                            }
                        }
                    }
                    let continuation_ms = continuation_results.join(";");
                    let status_ok = callback.is_ok() && solve.is_ok() && continuation_ok;
                    let checksum = callback
                        .as_ref()
                        .ok()
                        .map(|value| value.checksum)
                        .unwrap_or_default()
                        + solve.as_ref().ok().copied().unwrap_or_default()
                        + continuation_checksum;
                    record_compact_row(
                        &mut rows,
                        CompactBenchmarkRow {
                            route,
                            workload: workload.label().to_owned(),
                            dimension,
                            layout: layout_label.clone(),
                            policy: policy_label.clone(),
                            cold_prepare_ms,
                            callback_ms,
                            residual_ms,
                            jacobian_ms,
                            full_solve_ms,
                            continuation_ms,
                            checksum: format!("{checksum:.6e}"),
                            status: if status_ok {
                                "ok".to_owned()
                            } else {
                                "error".to_owned()
                            },
                        },
                        continuation_count,
                        report_started,
                    );
                }

                if std::env::var("RADAU_COMPACT_AOT").as_deref() == Ok("1") {
                    for frontend in [RadauFrontend::ExprLegacy, RadauFrontend::AtomViewNative] {
                        let route = format!(
                            "aot/{}/{}/{}/workers-{}",
                            match frontend {
                                RadauFrontend::ExprLegacy => "expr-legacy",
                                RadauFrontend::AtomViewNative => "atom-native",
                            },
                            layout_label,
                            policy_label,
                            worker_label()
                        );
                        let output_dir = tempfile::tempdir().expect("Radau compact AOT directory");
                        let started = Instant::now();
                        let prepared = prepare_aot_benchmark(
                            workload,
                            dimension,
                            frontend,
                            match layout {
                                BenchmarkLayout::Dense => RadauMatrixLayout::Dense,
                                BenchmarkLayout::Sparse => RadauMatrixLayout::Sparse,
                                BenchmarkLayout::Banded { lower, upper } => {
                                    RadauMatrixLayout::Banded { lower, upper }
                                }
                            },
                            match policy {
                                BenchmarkExecutionPolicy::Sequential => {
                                    RadauExecutionPolicy::Sequential
                                }
                                BenchmarkExecutionPolicy::Parallel { min_work } => {
                                    RadauExecutionPolicy::Parallel { min_work }
                                }
                                BenchmarkExecutionPolicy::Auto { min_work } => {
                                    RadauExecutionPolicy::Auto { min_work }
                                }
                            },
                            output_dir,
                        );
                        let cold_prepare_ms = compact_ms(started);
                        match prepared {
                            Ok(mut prepared) => {
                                let started = Instant::now();
                                let solve = prepared.warm_solve();
                                let full_solve_ms = compact_ms(started);
                                let mut continuation_results = Vec::new();
                                let mut continuation_checksum = 0.0;
                                for count in continuation_counts() {
                                    let started = Instant::now();
                                    let value = prepared.continuation(count);
                                    continuation_checksum += value;
                                    continuation_results.push(format!(
                                        "{count}:{:.3}",
                                        started.elapsed().as_secs_f64() * 1_000.0
                                    ));
                                }
                                let continuation_ms = continuation_results.join(";");
                                record_compact_row(
                                    &mut rows,
                                    CompactBenchmarkRow {
                                        route: route.clone(),
                                        workload: workload.label().to_owned(),
                                        dimension,
                                        layout: layout_label.clone(),
                                        policy: policy_label.clone(),
                                        cold_prepare_ms,
                                        // Callback-only work is emitted as a
                                        // separate row below. Keeping it out of
                                        // this row prevents adding unlike scopes.
                                        callback_ms: "-".to_owned(),
                                        residual_ms: "-".to_owned(),
                                        jacobian_ms: "-".to_owned(),
                                        full_solve_ms,
                                        continuation_ms,
                                        checksum: format!("{:.6e}", solve + continuation_checksum),
                                        status: "ok".to_owned(),
                                    },
                                    continuation_count,
                                    report_started,
                                );

                                // Keep callback preparation and callback
                                // execution explicit. This is a separate AOT
                                // owner, so its cold preparation must not be
                                // presented as part of the solver lifecycle.
                                let callback_directory = tempfile::tempdir()
                                    .expect("Radau compact AOT callback directory");
                                let callback_started = Instant::now();
                                let callback_prepared = PreparedAotCallbacks::prepare(
                                    workload,
                                    dimension,
                                    match frontend {
                                        RadauFrontend::ExprLegacy => BenchmarkAssembly::ExprLegacy,
                                        RadauFrontend::AtomViewNative => {
                                            BenchmarkAssembly::AtomViewNative
                                        }
                                    },
                                    layout,
                                    policy,
                                    RadauAotConfig::rebuild_always_release(
                                        callback_directory.path().to_path_buf(),
                                    )
                                    .with_c_compiler("tcc")
                                    .generated,
                                );
                                let callback_prepare_ms = compact_ms(callback_started);
                                match callback_prepared {
                                    Ok(callback_owner) => {
                                        let callback_started = Instant::now();
                                        let callback_result =
                                            callback_owner.warm_callback_breakdown(32);
                                        let callback_ms = callback_result
                                            .as_ref()
                                            .map(|value| format!("{:.3}", value.callback_ms))
                                            .unwrap_or_else(|_| compact_ms(callback_started));
                                        let residual_ms = callback_result
                                            .as_ref()
                                            .map(|value| format!("{:.3}", value.residual_ms))
                                            .unwrap_or_else(|_| "-".to_owned());
                                        let jacobian_ms = callback_result
                                            .as_ref()
                                            .map(|value| format!("{:.3}", value.jacobian_ms))
                                            .unwrap_or_else(|_| "-".to_owned());
                                        record_compact_row(
                                            &mut rows,
                                            CompactBenchmarkRow {
                                                route: format!("aot-callback/{}", route),
                                                workload: workload.label().to_owned(),
                                                dimension,
                                                layout: layout_label.clone(),
                                                policy: policy_label.clone(),
                                                cold_prepare_ms: callback_prepare_ms,
                                                callback_ms,
                                                residual_ms,
                                                jacobian_ms,
                                                full_solve_ms: "-".to_owned(),
                                                continuation_ms: "-".to_owned(),
                                                checksum: callback_result
                                                    .as_ref()
                                                    .map(|value| format!("{:.6e}", value.checksum))
                                                    .unwrap_or_else(|_| "-".to_owned()),
                                                status: if callback_result.is_ok() {
                                                    "ok".to_owned()
                                                } else {
                                                    "error".to_owned()
                                                },
                                            },
                                            continuation_count,
                                            report_started,
                                        );
                                    }
                                    Err(error) => record_compact_row(
                                        &mut rows,
                                        CompactBenchmarkRow {
                                            route: format!("aot-callback/{}", route),
                                            workload: workload.label().to_owned(),
                                            dimension,
                                            layout: layout_label.clone(),
                                            policy: policy_label.clone(),
                                            cold_prepare_ms: callback_prepare_ms,
                                            callback_ms: "-".to_owned(),
                                            residual_ms: "-".to_owned(),
                                            jacobian_ms: "-".to_owned(),
                                            full_solve_ms: "-".to_owned(),
                                            continuation_ms: "-".to_owned(),
                                            checksum: "-".to_owned(),
                                            status: format!("error: {error}"),
                                        },
                                        continuation_count,
                                        report_started,
                                    ),
                                }
                            }
                            Err(error) => record_compact_row(
                                &mut rows,
                                CompactBenchmarkRow {
                                    route,
                                    workload: workload.label().to_owned(),
                                    dimension,
                                    layout: layout_label.clone(),
                                    policy: policy_label.clone(),
                                    cold_prepare_ms,
                                    callback_ms: "-".to_owned(),
                                    residual_ms: "-".to_owned(),
                                    jacobian_ms: "-".to_owned(),
                                    full_solve_ms: "-".to_owned(),
                                    continuation_ms: "-".to_owned(),
                                    checksum: "-".to_owned(),
                                    status: format!("error: {error}"),
                                },
                                continuation_count,
                                report_started,
                            ),
                        }
                    }
                }
            }
        }
    }

    let body = compact_report_body(&rows, continuation_count, true);
    let path = RustedSciThe::Utils::test_reporting::write_test_report(
        "Radau_Bench",
        &compact_report_name(),
        &body,
    )
    .expect("write Radau compact benchmark report");
    println!(
        "[Radau compact benchmark] rows={} report={}",
        rows.len(),
        path.display()
    );
}

/// Matched warm/full-solve/continuation policy matrix. This is opt-in because
/// every row deliberately repeats the callback and solver work for three
/// scheduling policies; the default workload group remains a compact smoke
/// baseline.
fn benchmark_radau_policy_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("radau_execution_policy_matrix");
    group.sample_size(criterion_sample_size());
    group.measurement_time(criterion_measurement_time());
    let mut cases = Vec::new();
    for dimension in policy_diffusion_dimensions() {
        cases.push((
            WorkloadKind::DiffusionChain,
            dimension,
            BenchmarkLayout::Dense,
        ));
        cases.push((
            WorkloadKind::DiffusionChain,
            dimension,
            BenchmarkLayout::Banded { lower: 1, upper: 1 },
        ));
    }
    cases.push((WorkloadKind::ThreeBody, 12, BenchmarkLayout::Dense));
    let policies = [
        BenchmarkExecutionPolicy::Sequential,
        BenchmarkExecutionPolicy::Parallel { min_work: 1 },
        BenchmarkExecutionPolicy::Auto { min_work: 1 },
    ];
    for (workload, dimension, layout) in cases {
        for assembly in [
            BenchmarkAssembly::ExprLegacy,
            BenchmarkAssembly::AtomViewNative,
        ] {
            for policy in policies {
                let prepared = PreparedBenchmark::prepare_with_policy(
                    workload, dimension, assembly, layout, policy,
                )
                .expect("Radau policy benchmark preparation");
                let id = format!(
                    "{}/{}/{}/{}/{}/{}",
                    workload.label(),
                    dimension,
                    route_label(assembly, layout),
                    policy.label(),
                    worker_label(),
                    "fixed"
                );
                group.bench_function(BenchmarkId::new("warm-callbacks", &id), |bencher| {
                    bencher.iter(|| black_box(prepared.warm_callbacks(32).expect("callbacks")));
                });
                group.bench_function(BenchmarkId::new("full-solve", &id), |bencher| {
                    bencher.iter(|| black_box(prepared.solve().expect("solve")));
                });
                for count in continuation_counts() {
                    group.bench_function(
                        BenchmarkId::new(format!("continuation-{count}"), &id),
                        |bencher| {
                            bencher.iter(|| {
                                black_box(prepared.continuation(count).expect("continuation"))
                            });
                        },
                    );
                }
            }
        }
    }
    group.finish();
}

struct PreparedAotBenchmark {
    // The directory owns the published artifact for the lifetime of the
    // prepared solver. Dropping it while Criterion still calls warm stages
    // would turn a valid benchmark into a cache-lifetime test.
    _output_dir: tempfile::TempDir,
    solver: RadauSolver,
    initial_state: Vec<f64>,
    parameters: Vec<f64>,
}

impl PreparedAotBenchmark {
    fn warm_solve(&mut self) -> f64 {
        let solution = if self.parameters.is_empty() {
            self.solver
                .solve(&self.initial_state)
                .expect("Radau AOT warm solve")
        } else {
            self.solver
                .solve_with_parameters(&self.initial_state, &self.parameters)
                .expect("Radau AOT warm solve")
        };
        solution.y.iter().map(|value| value.abs()).sum()
    }

    fn continuation(&mut self, count: usize) -> f64 {
        let mut checksum = 0.0;
        for index in 0..count {
            let parameters = if self.parameters.is_empty() {
                Vec::new()
            } else {
                parameter_continuation_target(
                    &nalgebra::DVector::from_vec(self.parameters.clone()),
                    index,
                )
                .as_slice()
                .to_vec()
            };
            let solution = if parameters.is_empty() {
                self.solver
                    .solve(&self.initial_state)
                    .expect("Radau AOT continuation")
            } else {
                self.solver
                    .continue_with_parameters(&self.initial_state, &parameters)
                    .expect("Radau AOT continuation")
            };
            checksum += solution.y.iter().map(|value| value.abs()).sum::<f64>();
        }
        checksum
    }
}

fn aot_workload_problem(
    workload: WorkloadKind,
    dimension: usize,
) -> (RadauProblem, Vec<f64>, Vec<f64>) {
    let data = RustedSciThe::numerical::ivp_workloads::build_workload(workload, dimension);
    let problem = RadauProblem::new(data.equations, data.variables, data.time_variable)
        .with_parameters(data.parameter_names);
    (
        problem,
        data.initial_state.as_slice().to_vec(),
        data.parameter_values.as_slice().to_vec(),
    )
}

fn prepare_aot_benchmark(
    workload: WorkloadKind,
    dimension: usize,
    frontend: RadauFrontend,
    layout: RadauMatrixLayout,
    execution_policy: RadauExecutionPolicy,
    output_dir: tempfile::TempDir,
) -> Result<PreparedAotBenchmark, String> {
    let (problem, initial_state, parameters) = aot_workload_problem(workload, dimension);
    let config = RadauConfig {
        t_bound: match workload {
            WorkloadKind::DiffusionChain => 0.01,
            WorkloadKind::CombustionLike | WorkloadKind::Robertson | WorkloadKind::ThreeBody => {
                0.002
            }
            WorkloadKind::StiffScalar => 0.01,
        },
        first_step: Some(0.0005),
        max_step: 0.002,
        rtol: 1.0e-7,
        atol: 1.0e-10,
        execution: RadauExecution::Aot,
        frontend,
        matrix_layout: layout,
        telemetry: RadauTelemetryMode::Counters,
        execution_policy,
        aot: Some(
            if std::env::var("RADAU_BENCH_COMPACT_REPORT").as_deref() == Ok("1") {
                RadauAotConfig::rebuild_always_release(output_dir.path().to_path_buf())
                    .with_c_compiler("tcc")
            } else {
                RadauAotConfig::build_if_missing_release(output_dir.path().to_path_buf())
                    .with_c_compiler("tcc")
            },
        ),
        ..RadauConfig::default()
    };
    let solver = RadauSolver::prepare(problem, config).map_err(|error| error.to_string())?;
    Ok(PreparedAotBenchmark {
        _output_dir: output_dir,
        solver,
        initial_state,
        parameters,
    })
}

/// Optional toolchain-backed AOT matrix matched to the canonical Lambdify
/// workloads above. It is opt-in because even a cached AOT preparation may
/// invoke filesystem and compiler work during a benchmark.
fn benchmark_radau_aot(c: &mut Criterion) {
    let mut group = c.benchmark_group("radau_aot_vs_lambdify");
    group.sample_size(criterion_sample_size());
    group.measurement_time(criterion_measurement_time());
    let workloads = aot_cases();

    let policies = if std::env::var("RADAU_BENCH_POLICY_MATRIX").as_deref() == Ok("1") {
        vec![
            RadauExecutionPolicy::Sequential,
            RadauExecutionPolicy::Parallel { min_work: 1 },
            RadauExecutionPolicy::Auto { min_work: 1 },
        ]
    } else {
        vec![RadauExecutionPolicy::Sequential]
    };

    for (workload, dimension) in workloads {
        let layouts = if workload == WorkloadKind::DiffusionChain {
            vec![
                RadauMatrixLayout::Dense,
                RadauMatrixLayout::Sparse,
                RadauMatrixLayout::Banded { lower: 1, upper: 1 },
            ]
        } else {
            vec![RadauMatrixLayout::Dense]
        };
        for frontend in [RadauFrontend::ExprLegacy, RadauFrontend::AtomViewNative] {
            for layout in layouts.iter().copied() {
                for execution_policy in policies.iter().copied() {
                    let frontend_label = match frontend {
                        RadauFrontend::ExprLegacy => "expr-legacy",
                        RadauFrontend::AtomViewNative => "atom-native",
                    };
                    let layout_label = match layout {
                        RadauMatrixLayout::Dense => "dense".to_owned(),
                        RadauMatrixLayout::Sparse => "sparse".to_owned(),
                        RadauMatrixLayout::Banded { lower, upper } => {
                            format!("banded-{lower}-{upper}")
                        }
                    };
                    let policy_label = match execution_policy {
                        RadauExecutionPolicy::Sequential => "sequential",
                        RadauExecutionPolicy::Parallel { .. } => "parallel",
                        RadauExecutionPolicy::Auto { .. } => "auto",
                    };
                    let id = format!(
                        "{}/{dimension}/{frontend_label}/{layout_label}/{policy_label}",
                        workload.label()
                    );
                    let id = format!("{id}/workers-{}", worker_label());

                    group.bench_function(BenchmarkId::new("cold-prepare", &id), |bencher| {
                        bencher.iter_batched(
                            || tempfile::tempdir().expect("Radau AOT benchmark directory"),
                            |directory| {
                                black_box(
                                    prepare_aot_benchmark(
                                        workload,
                                        dimension,
                                        frontend,
                                        layout,
                                        execution_policy,
                                        directory,
                                    )
                                    .expect("Radau AOT prepare"),
                                );
                            },
                            BatchSize::SmallInput,
                        );
                    });

                    let directory =
                        tempfile::tempdir().expect("Radau AOT warm benchmark directory");
                    let mut prepared = prepare_aot_benchmark(
                        workload,
                        dimension,
                        frontend,
                        layout,
                        execution_policy,
                        directory,
                    )
                    .expect("Radau AOT warm prepare");
                    group.bench_function(BenchmarkId::new("warm-solve", &id), |bencher| {
                        bencher.iter(|| black_box(prepared.warm_solve()));
                    });

                    // Keep callback-only timing separate from full solves. The
                    // callback owner uses the same generated AOT plan and
                    // caller-owned buffers, so this does not include Radau's
                    // controller or linear algebra stages.
                    let callback_directory =
                        tempfile::tempdir().expect("Radau AOT callback benchmark directory");
                    let prepared_callbacks = PreparedAotCallbacks::prepare(
                        workload,
                        dimension,
                        match frontend {
                            RadauFrontend::ExprLegacy => BenchmarkAssembly::ExprLegacy,
                            RadauFrontend::AtomViewNative => BenchmarkAssembly::AtomViewNative,
                        },
                        match layout {
                            RadauMatrixLayout::Dense => BenchmarkLayout::Dense,
                            RadauMatrixLayout::Sparse => BenchmarkLayout::Sparse,
                            RadauMatrixLayout::Banded { lower, upper } => {
                                BenchmarkLayout::Banded { lower, upper }
                            }
                        },
                        match execution_policy {
                            RadauExecutionPolicy::Sequential => {
                                BenchmarkExecutionPolicy::Sequential
                            }
                            RadauExecutionPolicy::Parallel { min_work } => {
                                BenchmarkExecutionPolicy::Parallel { min_work }
                            }
                            RadauExecutionPolicy::Auto { min_work } => {
                                BenchmarkExecutionPolicy::Auto { min_work }
                            }
                        },
                        RadauAotConfig::build_if_missing_release(
                            callback_directory.path().to_path_buf(),
                        )
                        .with_c_compiler("tcc")
                        .generated,
                    )
                    .expect("Radau AOT callback preparation");
                    group.bench_function(BenchmarkId::new("warm-callbacks", &id), |bencher| {
                        bencher.iter(|| {
                            black_box(
                                prepared_callbacks
                                    .warm_callbacks(64)
                                    .expect("Radau AOT callback evaluation"),
                            )
                        });
                    });
                    for count in continuation_counts() {
                        group.bench_function(
                            BenchmarkId::new(format!("continuation-{count}"), &id),
                            |bencher| {
                                bencher.iter(|| black_box(prepared.continuation(count)));
                            },
                        );
                    }
                }
            }
        }
    }
    group.finish();
}

criterion_group!(benches, benchmark_radau_workloads);
criterion_main!(benches);
