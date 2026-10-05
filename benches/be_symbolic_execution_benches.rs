//! Bounded BE comparison of symbolic Lambdify and dense AtomView AOT routes.
//!
//! AOT artifacts are prepared before timed iterations. These groups measure
//! warm solves and fresh solver preparation with an already-built artifact;
//! the ignored release lifecycle story measures the cold build/link path.

use RustedSciThe::numerical::BE::{
    BE, BeSolverOptions, BeSymbolicAssemblyBackend, BeTelemetryMode,
};
use RustedSciThe::numerical::ivp_workloads;
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use nalgebra::DVector;
use std::hint::black_box;
use std::process::Command;
use std::time::Duration;

const DIMENSION: usize = 8;
const STEP: f64 = 0.002;
const FINAL_TIME: f64 = 0.02;
const AOT_DIR: &str = "target/be-symbolic-execution-aot";

fn workload_parameters(rate_scale: f64) -> DVector<f64> {
    let mut parameters = ivp_workloads::diffusion_chain(DIMENSION).parameter_values;
    parameters[0] *= rate_scale;
    parameters
}

#[derive(Clone, Copy)]
struct Route {
    label: &'static str,
    assembly: BeSymbolicAssemblyBackend,
    aot: bool,
}

const ROUTES: [Route; 3] = [
    Route {
        label: "lambdify-expr-legacy",
        assembly: BeSymbolicAssemblyBackend::ExprLegacy,
        aot: false,
    },
    Route {
        label: "lambdify-atom-native",
        assembly: BeSymbolicAssemblyBackend::AtomViewNative,
        aot: false,
    },
    Route {
        label: "aot-atom-native-tcc",
        assembly: BeSymbolicAssemblyBackend::AtomViewNative,
        aot: true,
    },
];

fn tcc_available() -> bool {
    let locator = if cfg!(windows) { "where" } else { "which" };
    Command::new(locator)
        .arg("tcc")
        .output()
        .is_ok_and(|output| output.status.success())
}

fn make_solver(route: Route, rate_scale: f64, telemetry: BeTelemetryMode) -> BE {
    let workload = ivp_workloads::diffusion_chain(DIMENSION);
    let parameter_names: Vec<&str> = workload
        .parameter_names
        .iter()
        .map(String::as_str)
        .collect();
    let parameter_values = workload_parameters(rate_scale);
    let options = BeSolverOptions::new(
        workload.equations,
        workload.variables,
        workload.time_variable,
        1e-10,
        20,
        Some(STEP),
        0.0,
        FINAL_TIME,
        workload.initial_state,
    )
    .with_symbolic_assembly_backend(route.assembly)
    .with_telemetry_mode(telemetry);
    let mut solver = BE::try_new_with_options(options).expect("symbolic BE setup should succeed");
    solver
        .try_set_equation_parameters(Some(&parameter_names))
        .expect("shared workload parameter schema should be valid");
    solver
        .set_parameter_values(parameter_values)
        .expect("shared workload parameters should bind");
    if route.aot {
        solver.set_dense_generated_backend_c_tcc(AOT_DIR);
    }
    solver
}

fn ensure_compiled_aot() {
    if !tcc_available() {
        println!("[BE symbolic execution benches] tcc unavailable; AOT cases skipped");
        return;
    }

    let route = ROUTES[2];
    let mut solver = make_solver(route, 0.8, BeTelemetryMode::Timings);
    solver
        .try_solve()
        .expect("AOT warm-up build should succeed");
    let telemetry = solver
        .symbolic_ivp_telemetry_snapshot()
        .expect("AOT preparation should emit symbolic telemetry");
    assert_eq!(telemetry.execution.label(), "aot");
    assert!(telemetry.aot_runtime_ready > 0);
    println!(
        "[BE symbolic execution benches] prebuilt AOT ready; build_attempts={} link_attempts={} artifact_keys={:?}; cold build excluded from Criterion timings",
        telemetry.aot_build_attempts, telemetry.aot_link_attempts, telemetry.aot_artifact_keys
    );
}

fn benchmark_symbolic_execution(c: &mut Criterion) {
    ensure_compiled_aot();
    let has_aot = tcc_available();
    let routes = ROUTES.into_iter().filter(|route| !route.aot || has_aot);
    let mut group = c.benchmark_group("be_symbolic_execution");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));
    println!(
        "[BE symbolic execution] fixture=shared-diffusion-chain-{DIMENSION}; steps={}; Lambdify full solve versus AtomView AOT; telemetry=Off; AOT build/link done before timing",
        (FINAL_TIME / STEP) as usize
    );

    let mut reference: Option<Vec<f64>> = None;
    for route in routes {
        let mut warm = make_solver(route, 0.8, BeTelemetryMode::Off);
        warm.try_solve()
            .expect("route warm-up solve should succeed");
        let trajectory = warm.trajectory().1;
        if let Some(expected) = &reference {
            assert_eq!(expected.len(), trajectory.len());
            assert!(
                expected
                    .iter()
                    .zip(trajectory.iter())
                    .all(|(left, right)| (left - right).abs() <= 1e-10)
            );
        } else {
            reference = Some(trajectory.iter().copied().collect());
        }

        group.bench_function(
            BenchmarkId::new("warm-full-solve", route.label),
            |bencher| {
                bencher.iter(|| {
                    warm.try_solve().expect("warm BE solve should succeed");
                    black_box(warm.trajectory());
                });
            },
        );
        group.bench_function(
            BenchmarkId::new("fresh-solver-setup-solve", route.label),
            |bencher| {
                bencher.iter_batched(
                    || make_solver(route, 0.8, BeTelemetryMode::Off),
                    |mut solver| {
                        solver
                            .try_solve()
                            .expect("fresh configured BE solve should succeed");
                        black_box(solver.trajectory());
                    },
                    BatchSize::SmallInput,
                );
            },
        );

        if route.assembly == BeSymbolicAssemblyBackend::AtomViewNative {
            for target_count in [1, 4] {
                let rates: Vec<f64> = (0..target_count)
                    .map(|index| 0.6 + index as f64 * 0.2)
                    .collect();
                let mut prepared = make_solver(route, rates[0], BeTelemetryMode::Off);
                prepared
                    .try_solve()
                    .expect("parameter-series warmup should succeed");

                for rate in &rates {
                    prepared
                        .set_parameter_values(workload_parameters(*rate))
                        .expect("parameter rebind should succeed");
                    prepared.try_solve().expect("rebound solve should succeed");
                    let prepared_values: Vec<f64> =
                        prepared.trajectory().1.iter().copied().collect();
                    let mut fresh = make_solver(route, *rate, BeTelemetryMode::Off);
                    fresh.try_solve().expect("fresh solve should succeed");
                    assert!(
                        prepared_values
                            .iter()
                            .zip(fresh.trajectory().1.iter())
                            .all(|(left, right)| (left - right).abs() <= 1e-10)
                    );
                }

                group.bench_with_input(
                    BenchmarkId::new(
                        "warm-parameter-rebind-series",
                        format!("{}/{target_count}", route.label),
                    ),
                    &rates,
                    |bencher, rates| {
                        bencher.iter(|| {
                            for rate in rates {
                                prepared
                                    .set_parameter_values(workload_parameters(*rate))
                                    .expect("parameter rebind should succeed");
                                prepared.try_solve().expect("rebound solve should succeed");
                            }
                            black_box(prepared.trajectory());
                        });
                    },
                );
            }
        }
    }
    group.finish();
}

criterion_group!(benches, benchmark_symbolic_execution);
criterion_main!(benches);
