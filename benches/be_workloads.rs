//! Compact release dashboard for the Backward Euler workload corpus.
//!
//! This is intentionally a plain bench target rather than a Criterion target:
//! one run produces one compact table and does not emit warm-up/sample noise.
//! Detailed microbenchmarks remain in the existing BE Criterion targets.

use RustedSciThe::Utils::test_reporting::write_test_report;
use RustedSciThe::numerical::BE::{
    BE, BeSolverOptions, BeSymbolicAssemblyBackend, BeTelemetryMode,
};
use RustedSciThe::numerical::ivp_workloads::{
    SymbolicWorkload, WorkloadKind, build_workload, parameter_continuation_target,
};
use std::fmt::Write as _;
use std::hint::black_box;
use std::process::Command;
use std::time::Instant;
use tabled::{Table, Tabled};
use tempfile::TempDir;

#[path = "support/be_bench_support.rs"]
#[allow(dead_code)]
mod be_bench_support;

#[derive(Clone, Copy, Debug)]
enum Route {
    NativeAnalytic,
    NativeFiniteDifference,
    LambdifyExprLegacy,
    LambdifyAtomView,
    AotExprLegacy,
    AotAtomView,
}

impl Route {
    fn label(self) -> &'static str {
        match self {
            Self::NativeAnalytic => "native-analytic",
            Self::NativeFiniteDifference => "native-fd",
            Self::LambdifyExprLegacy => "lambdify-expr-legacy",
            Self::LambdifyAtomView => "lambdify-atom-native",
            Self::AotExprLegacy => "aot-expr-legacy-tcc",
            Self::AotAtomView => "aot-atom-native-tcc",
        }
    }

    fn is_aot(self) -> bool {
        matches!(self, Self::AotExprLegacy | Self::AotAtomView)
    }

    fn assembly(self) -> Option<BeSymbolicAssemblyBackend> {
        match self {
            Self::LambdifyExprLegacy | Self::AotExprLegacy => {
                Some(BeSymbolicAssemblyBackend::ExprLegacy)
            }
            Self::LambdifyAtomView | Self::AotAtomView => {
                Some(BeSymbolicAssemblyBackend::AtomViewNative)
            }
            Self::NativeAnalytic | Self::NativeFiniteDifference => None,
        }
    }
}

#[derive(Tabled)]
struct DashboardRow {
    workload: String,
    dimension: String,
    route: String,
    phase: String,
    prepare_ms: String,
    solve_ms: String,
    continuation_ms: String,
    residual_calls: String,
    jacobian_calls: String,
    factorization_calls: String,
    linear_solve_calls: String,
    accepted_steps: String,
    max_abs: String,
    aot_builds: String,
    aot_links: String,
    status: String,
}

struct PreparedCase {
    solver: BE,
    // The generated backend stores paths, so the directory must outlive the
    // solver even though all formatting happens after the timed work.
    _artifact_dir: Option<TempDir>,
}

fn env_list(name: &str, default: &str) -> Vec<String> {
    std::env::var(name)
        .unwrap_or_else(|_| default.to_owned())
        .split(',')
        .map(|item| item.trim().to_ascii_lowercase())
        .filter(|item| !item.is_empty())
        .collect()
}

fn env_usizes(name: &str, default: &str) -> Vec<usize> {
    env_list(name, default)
        .into_iter()
        .filter_map(|item| item.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .collect()
}

fn selected_workloads() -> Vec<WorkloadKind> {
    env_list(
        "BE_BENCH_WORKLOADS",
        "diffusion-chain,combustion-like,robertson,three-body",
    )
    .into_iter()
    .filter_map(|label| {
        WorkloadKind::ALL
            .into_iter()
            .find(|kind| kind.label() == label)
    })
    .collect()
}

fn selected_routes(include_aot: bool) -> Vec<Route> {
    let default = if include_aot {
        "native-analytic,native-fd,lambdify-expr-legacy,lambdify-atom-native,aot-expr-legacy-tcc,aot-atom-native-tcc"
    } else {
        "native-analytic,native-fd,lambdify-expr-legacy,lambdify-atom-native"
    };
    env_list("BE_BENCH_ROUTES", default)
        .into_iter()
        .filter_map(|label| match label.as_str() {
            "native-analytic" => Some(Route::NativeAnalytic),
            "native-fd" => Some(Route::NativeFiniteDifference),
            "lambdify-expr-legacy" => Some(Route::LambdifyExprLegacy),
            "lambdify-atom-native" => Some(Route::LambdifyAtomView),
            "aot-expr-legacy-tcc" => Some(Route::AotExprLegacy),
            "aot-atom-native-tcc" => Some(Route::AotAtomView),
            _ => None,
        })
        .filter(|route| include_aot || !route.is_aot())
        .collect()
}

fn tcc_available() -> bool {
    let locator = if cfg!(windows) { "where" } else { "which" };
    Command::new(locator)
        .arg("tcc")
        .output()
        .is_ok_and(|output| output.status.success())
}

fn make_symbolic_case(workload: SymbolicWorkload, route: Route) -> Result<PreparedCase, String> {
    let parameter_names: Vec<&str> = workload
        .parameter_names
        .iter()
        .map(String::as_str)
        .collect();
    let mut options = BeSolverOptions::new(
        workload.equations,
        workload.variables,
        workload.time_variable,
        1e-10,
        20,
        Some(0.002),
        0.0,
        0.02,
        workload.initial_state,
    )
    .with_symbolic_assembly_backend(route.assembly().expect("symbolic route"))
    .with_telemetry_mode(BeTelemetryMode::Timings);

    let artifact_dir = if route.is_aot() {
        let directory = tempfile::tempdir().map_err(|error| error.to_string())?;
        options = options.with_dense_generated_backend_c_tcc(directory.path().to_path_buf());
        Some(directory)
    } else {
        None
    };

    let mut solver = BE::try_new_with_options(options).map_err(|error| format!("{error:?}"))?;
    if !parameter_names.is_empty() {
        solver
            .try_set_equation_parameters(Some(&parameter_names))
            .map_err(|error| format!("{error:?}"))?;
        solver
            .set_parameter_values(workload.parameter_values)
            .map_err(|error| format!("{error:?}"))?;
    }
    Ok(PreparedCase {
        solver,
        _artifact_dir: artifact_dir,
    })
}

fn make_native_case(
    workload: WorkloadKind,
    dimension: usize,
    route: Route,
) -> Result<PreparedCase, String> {
    let solver = match (workload, route) {
        (WorkloadKind::DiffusionChain, Route::NativeAnalytic) => {
            be_bench_support::make_diffusion_solver(
                dimension,
                true,
                BeTelemetryMode::Timings,
                0.002,
                0.02,
            )
        }
        (WorkloadKind::DiffusionChain, Route::NativeFiniteDifference) => {
            be_bench_support::make_diffusion_solver(
                dimension,
                false,
                BeTelemetryMode::Timings,
                0.002,
                0.02,
            )
        }
        (WorkloadKind::CombustionLike, Route::NativeAnalytic) => {
            be_bench_support::make_combustion_solver(BeTelemetryMode::Timings)
        }
        _ => return Err("native route is not defined for this workload".to_owned()),
    };
    Ok(PreparedCase {
        solver,
        _artifact_dir: None,
    })
}

fn fmt_ms(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "n/a".to_owned())
}

fn max_abs(solver: &BE) -> f64 {
    solver
        .trajectory()
        .1
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max)
}

fn run_case(
    workload: WorkloadKind,
    dimension: usize,
    route: Route,
    continuation_count: usize,
) -> Result<DashboardRow, String> {
    if route.is_aot() && !tcc_available() {
        return Ok(DashboardRow {
            workload: workload.label().to_owned(),
            dimension: dimension.to_string(),
            route: route.label().to_owned(),
            phase: "all".to_owned(),
            prepare_ms: "n/a".to_owned(),
            solve_ms: "n/a".to_owned(),
            continuation_ms: "n/a".to_owned(),
            residual_calls: "n/a".to_owned(),
            jacobian_calls: "n/a".to_owned(),
            factorization_calls: "n/a".to_owned(),
            linear_solve_calls: "n/a".to_owned(),
            accepted_steps: "n/a".to_owned(),
            max_abs: "n/a".to_owned(),
            aot_builds: "n/a".to_owned(),
            aot_links: "n/a".to_owned(),
            status: "skipped:tcc-unavailable".to_owned(),
        });
    }

    let preparation_started = Instant::now();
    let mut prepared = if route.assembly().is_some() {
        make_symbolic_case(build_workload(workload, dimension), route)?
    } else {
        make_native_case(workload, dimension, route)?
    };
    let prepare_ms = preparation_started.elapsed().as_secs_f64() * 1_000.0;

    let solve_started = Instant::now();
    prepared
        .solver
        .try_solve()
        .map_err(|error| format!("{error:?}"))?;
    let solve_wall_ms = solve_started.elapsed().as_secs_f64() * 1_000.0;

    let continuation_ms = if route.assembly().is_some()
        && build_workload(workload, dimension).is_parameterized()
        && continuation_count > 0
    {
        let base = build_workload(workload, dimension).parameter_values;
        let started = Instant::now();
        for index in 0..continuation_count {
            let values = parameter_continuation_target(&base, index + 1);
            prepared
                .solver
                .set_parameter_values(values)
                .map_err(|error| format!("{error:?}"))?;
            prepared
                .solver
                .try_solve()
                .map_err(|error| format!("{error:?}"))?;
        }
        Some(started.elapsed().as_secs_f64() * 1_000.0)
    } else {
        None
    };

    let stats = prepared.solver.get_statistics();
    let detailed = prepared.solver.detailed_statistics();
    let snapshot = prepared.solver.symbolic_ivp_telemetry_snapshot();
    let (aot_builds, aot_links) = if route.is_aot() {
        snapshot
            .map(|snapshot| {
                (
                    snapshot.aot_build_attempts.to_string(),
                    snapshot.aot_link_attempts.to_string(),
                )
            })
            .unwrap_or_else(|| ("n/a".to_owned(), "n/a".to_owned()))
    } else {
        ("n/a".to_owned(), "n/a".to_owned())
    };
    let phase = if continuation_ms.is_some() {
        format!("solve+continuation-{continuation_count}")
    } else {
        "single-solve".to_owned()
    };
    let row = DashboardRow {
        workload: workload.label().to_owned(),
        dimension: dimension.to_string(),
        route: route.label().to_owned(),
        phase,
        prepare_ms: fmt_ms(Some(prepare_ms)),
        solve_ms: fmt_ms(Some(solve_wall_ms)),
        continuation_ms: fmt_ms(continuation_ms),
        residual_calls: stats.residual_calls.to_string(),
        jacobian_calls: stats.jacobian_calls.to_string(),
        factorization_calls: detailed.factorization_calls.to_string(),
        linear_solve_calls: detailed.linear_solve_calls.to_string(),
        accepted_steps: detailed.accepted_steps.to_string(),
        max_abs: format!("{:.6e}", max_abs(&prepared.solver)),
        aot_builds,
        aot_links,
        status: "ok".to_owned(),
    };
    black_box(prepared.solver.trajectory());
    Ok(row)
}

fn main() {
    let include_aot = env_list("BE_BENCH_ROUTES", "")
        .iter()
        .any(|route| route.starts_with("aot-"));
    let workloads = selected_workloads();
    let dimensions = env_usizes("BE_BENCH_DIMENSIONS", "8,32");
    let routes = selected_routes(include_aot);
    let continuation_count = std::env::var("BE_BENCH_CONTINUATION")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4);
    let mut rows = Vec::new();

    for workload in workloads {
        let workload_dimensions = if workload == WorkloadKind::DiffusionChain {
            dimensions.clone()
        } else {
            vec![0]
        };
        for dimension in workload_dimensions {
            for route in &routes {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run_case(workload, dimension, *route, continuation_count)
                }));
                match result {
                    Ok(Ok(row)) => rows.push(row),
                    Ok(Err(error)) => rows.push(error_row(workload, dimension, *route, error)),
                    Err(_) => rows.push(error_row(
                        workload,
                        dimension,
                        *route,
                        "panic during dashboard case".to_owned(),
                    )),
                }
            }
        }
    }

    let table = Table::new(&rows).to_string();
    let mut report = String::new();
    writeln!(report, "# BE compact release dashboard").unwrap();
    writeln!(report).unwrap();
    writeln!(
        report,
        "- workloads: {:?}",
        env_list("BE_BENCH_WORKLOADS", "default")
    )
    .unwrap();
    writeln!(report, "- dimensions: {:?}", dimensions).unwrap();
    writeln!(report, "- continuation_targets: {continuation_count}").unwrap();
    writeln!(report, "- telemetry: Timings").unwrap();
    writeln!(report).unwrap();
    writeln!(report, "{table}").unwrap();
    writeln!(report).unwrap();
    writeln!(report, "Preparation includes solver construction, symbolic binding and AOT build/link when enabled.").unwrap();
    writeln!(report, "Continuation is warm parameter rebind after the first solve; telemetry scopes are diagnostic and not additive.").unwrap();

    println!("{report}");
    if let Err(error) = write_test_report("BE", "be_workloads_dashboard", &report) {
        eprintln!("[BE dashboard] unable to write compact report: {error}");
    }
}

fn error_row(
    workload: WorkloadKind,
    dimension: usize,
    route: Route,
    error: String,
) -> DashboardRow {
    DashboardRow {
        workload: workload.label().to_owned(),
        dimension: dimension.to_string(),
        route: route.label().to_owned(),
        phase: "all".to_owned(),
        prepare_ms: "n/a".to_owned(),
        solve_ms: "n/a".to_owned(),
        continuation_ms: "n/a".to_owned(),
        residual_calls: "n/a".to_owned(),
        jacobian_calls: "n/a".to_owned(),
        factorization_calls: "n/a".to_owned(),
        linear_solve_calls: "n/a".to_owned(),
        accepted_steps: "n/a".to_owned(),
        max_abs: "n/a".to_owned(),
        aot_builds: "n/a".to_owned(),
        aot_links: "n/a".to_owned(),
        status: format!("error:{}", error.replace('|', "/")),
    }
}
