//! Process-level allocation and peak-live-heap audit for fitting observation
//! batching. Timings are diagnostic only because the counting allocator adds
//! overhead; do not compare them with ordinary release benchmarks.
//!
//! Inputs and symbolic expressions are constructed before each measurement.
//! The measured region covers model preparation and one complete fit. Peak
//! bytes are allocator-observed live heap above the pre-solve process baseline,
//! not process RSS and not a per-thread memory limit.
//!
//! Run from the repository root:
//!
//! ```text
//! cargo bench --bench fitting_observation_batch_memory_audit
//! ```

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use RustedSciThe::numerical::optimization::sym_fitting::{Fitting, FittingObservationPolicy};
use RustedSciThe::numerical::Nonlinear_systems::least_squares::LeastSquaresTelemetryMode;
use RustedSciThe::numerical::Nonlinear_systems::symbolic::{
    LambdifyExecutionPolicy, SymbolicLambdifyFrontend,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;

struct AuditAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);

#[global_allocator]
static GLOBAL: AuditAllocator = AuditAllocator;

fn update_peak(candidate: usize) {
    let mut current = PEAK_LIVE_BYTES.load(Ordering::Relaxed);
    while candidate > current {
        match PEAK_LIVE_BYTES.compare_exchange_weak(
            current,
            candidate,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(observed) => current = observed,
        }
    }
}

// SAFETY: all operations forward the original layouts and pointers to System.
unsafe impl GlobalAlloc for AuditAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller provides a valid layout as required by GlobalAlloc.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
            let live = LIVE_BYTES.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            update_peak(live);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: pointer/layout are forwarded unchanged to the backing allocator.
        unsafe { System.dealloc(pointer, layout) };
        DEALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        DEALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the original allocation contract is forwarded unchanged.
        let new_pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !new_pointer.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(new_size, Ordering::Relaxed);
            DEALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            DEALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
            let previous = LIVE_BYTES
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |live| {
                    live.checked_sub(layout.size())?.checked_add(new_size)
                })
                .expect("tracked live bytes should include the resized allocation");
            update_peak(previous - layout.size() + new_size);
        }
        new_pointer
    }
}

#[derive(Clone, Copy, Debug)]
enum Workload {
    SingleResponse,
    TwoResponse,
}

struct Fixture {
    x: Vec<f64>,
    y: Vec<f64>,
    equations: Vec<Expr>,
    unknowns: Vec<String>,
    initial_guess: Vec<f64>,
}

impl Fixture {
    fn new(workload: Workload, samples: usize) -> Self {
        let end = match workload {
            Workload::SingleResponse => 20.0,
            Workload::TwoResponse => 4.0,
        };
        let x = (0..samples)
            .map(|index| end * index as f64 / (samples - 1) as f64)
            .collect::<Vec<_>>();

        let (equations, unknowns, initial_guess, y) = match workload {
            Workload::SingleResponse => {
                let y = x
                    .iter()
                    .map(|&x| 2.0 * (-x / 0.5_f64).exp() + (x / 2.0_f64).sin())
                    .collect();
                (
                    vec![Expr::parse_expression("a*exp(-x/tau)+c*sin(x/omega)")],
                    ["a", "tau", "c", "omega"].map(str::to_owned).to_vec(),
                    vec![1.9, 0.55, 0.9, 1.9],
                    y,
                )
            }
            Workload::TwoResponse => {
                let mut y = Vec::with_capacity(2 * samples);
                y.extend(x.iter().map(|&x| 1.7 * (-0.35 * x).exp() + 0.2));
                y.extend(x.iter().map(|&x| 0.8 * (1.2 * x).sin() - 0.1));
                (
                    vec![
                        Expr::parse_expression("a*exp(b*x)+c"),
                        Expr::parse_expression("d*sin(e*x)+f"),
                    ],
                    ["a", "b", "c", "d", "e", "f"].map(str::to_owned).to_vec(),
                    vec![1.6, -0.32, 0.15, 0.75, 1.15, -0.08],
                    y,
                )
            }
        };

        Self {
            x,
            y,
            equations,
            unknowns,
            initial_guess,
        }
    }

    fn build(
        &self,
        frontend: SymbolicLambdifyFrontend,
        policy: FittingObservationPolicy,
    ) -> Fitting {
        Fitting::new()
            .with_data(self.x.clone(), self.y.clone())
            .with_equations(self.equations.clone())
            .with_arg("x".to_owned())
            .with_unknowns(self.unknowns.clone())
            .with_initial_guess(self.initial_guess.clone())
            .with_symbolic_frontend(frontend)
            .with_lambdify_execution_policy(LambdifyExecutionPolicy::Sequential)
            .with_observation_policy(policy)
            .with_telemetry(LeastSquaresTelemetryMode::Counters)
    }
}

#[derive(Clone, Copy, Default)]
struct Snapshot {
    allocations: usize,
    allocated_bytes: usize,
    deallocations: usize,
    deallocated_bytes: usize,
    peak_extra_live_bytes: usize,
    solve_ms: f64,
    iterations: usize,
    residual_evaluations: usize,
    jacobian_evaluations: usize,
}

fn reset_counters() {
    for counter in [
        &ALLOCATIONS,
        &DEALLOCATIONS,
        &ALLOCATED_BYTES,
        &DEALLOCATED_BYTES,
    ] {
        counter.store(0, Ordering::Relaxed);
    }
    PEAK_LIVE_BYTES.store(LIVE_BYTES.load(Ordering::Relaxed), Ordering::Relaxed);
}

fn measure(
    fixture: &Fixture,
    frontend: SymbolicLambdifyFrontend,
    policy: FittingObservationPolicy,
) -> Snapshot {
    let fit = fixture.build(frontend, policy);
    reset_counters();
    let live_before = LIVE_BYTES.load(Ordering::Relaxed);
    let started = Instant::now();
    let fit = fit.try_build().expect("fitting fixture should converge");
    let solve_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let stats = fit.last_statistics().expect("counter telemetry enabled");
    let snapshot = Snapshot {
        allocations: ALLOCATIONS.load(Ordering::Relaxed),
        allocated_bytes: ALLOCATED_BYTES.load(Ordering::Relaxed),
        deallocations: DEALLOCATIONS.load(Ordering::Relaxed),
        deallocated_bytes: DEALLOCATED_BYTES.load(Ordering::Relaxed),
        peak_extra_live_bytes: PEAK_LIVE_BYTES
            .load(Ordering::Relaxed)
            .saturating_sub(live_before),
        solve_ms,
        iterations: stats.iterations,
        residual_evaluations: stats.residual_evaluations,
        jacobian_evaluations: stats.jacobian_evaluations,
    };
    assert!(fit.r_squared().is_some_and(|r_squared| r_squared > 0.999));
    black_box(fit.result.as_ref());
    snapshot
}

fn median(mut values: Vec<usize>) -> usize {
    values.sort_unstable();
    values[values.len() / 2]
}

fn main() {
    let frontends = [
        SymbolicLambdifyFrontend::ExprLegacy,
        SymbolicLambdifyFrontend::AtomViewNative,
    ];
    let parallel = FittingObservationPolicy::Parallel {
        min_observations: 256,
    };

    // Warm Rayon workers and symbolic preparation before collecting samples.
    for frontend in frontends {
        let warmup = Fixture::new(Workload::TwoResponse, 1024);
        warmup
            .build(frontend, parallel)
            .try_build()
            .expect("warm-up fit should converge");
    }

    println!(
        "Fitting observation-batch allocator audit; 5 samples/row; peak metric is additional live heap, not RSS; allocator-instrumented timings are diagnostic only"
    );
    println!("workload | N | equations | params | frontend | policy | allocs median | allocated MiB median | peak extra live MiB median | deallocs median | deallocated MiB median | solve ms median* | iterations / residual / jacobian");

    for (workload, samples) in [
        (Workload::SingleResponse, 100_001),
        (Workload::TwoResponse, 100_001),
        (Workload::TwoResponse, 200_001),
    ] {
        let fixture = Fixture::new(workload, samples);
        let equation_count = fixture.equations.len();
        let parameter_count = fixture.unknowns.len();
        for frontend in frontends {
            for policy in [FittingObservationPolicy::Sequential, parallel] {
                let mut rows = Vec::with_capacity(5);
                for _ in 0..5 {
                    let row = measure(&fixture, frontend, policy);
                    rows.push(row);
                }

                let reference = rows[0];
                assert!(rows.iter().all(|row| {
                    row.iterations == reference.iterations
                        && row.residual_evaluations == reference.residual_evaluations
                        && row.jacobian_evaluations == reference.jacobian_evaluations
                }));
                println!(
                    "{workload:?} | {samples} | {equation_count} | {parameter_count} | {frontend:?} | {policy:?} | {} | {:.3} | {:.3} | {} | {:.3} | {:.3} | {}/{}/{}",
                    median(rows.iter().map(|row| row.allocations).collect()),
                    median(rows.iter().map(|row| row.allocated_bytes).collect()) as f64
                        / (1024.0 * 1024.0),
                    median(rows.iter().map(|row| row.peak_extra_live_bytes).collect()) as f64
                        / (1024.0 * 1024.0),
                    median(rows.iter().map(|row| row.deallocations).collect()),
                    median(rows.iter().map(|row| row.deallocated_bytes).collect()) as f64
                        / (1024.0 * 1024.0),
                    {
                        let mut times = rows.iter().map(|row| row.solve_ms).collect::<Vec<_>>();
                        times.sort_by(f64::total_cmp);
                        times[times.len() / 2]
                    },
                    reference.iterations,
                    reference.residual_evaluations,
                    reference.jacobian_evaluations,
                );
            }
        }
    }
}
