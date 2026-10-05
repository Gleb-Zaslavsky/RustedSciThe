//! Process-level allocation audit for nonlinear solver hot paths.
//!
//! This is intentionally a separate benchmark target from the Criterion timing
//! benchmark. The counting allocator adds instrumentation overhead, so its
//! numbers must not be compared with ordinary wall-clock benchmarks.
//!
//! The default audit covers dimensions 32, 128, and 512. It is intended to
//! answer whether the dimension-32 ownership conclusions still hold for a
//! dense production-shaped workload. Allocation bytes include the complete
//! solve result lifetime; this is not a peak-RSS measurement.
//!
//! Run with:
//!
//! ```text
//! cargo bench --bench nonlinear_systems_allocation_audit
//! NONLINEAR_TRUST_REGION_OWNERSHIP_ONLY=1 cargo bench --bench nonlinear_systems_allocation_audit -- --noplot
//! ```

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use nalgebra::{DMatrix, DVector};

use RustedSciThe::numerical::Nonlinear_systems::error::SolveError;
use RustedSciThe::numerical::Nonlinear_systems::prelude::{
    DampedNewtonMethod, DampedNewtonMethodAdvanced, DiagnosticsOptions, JacobianProvider,
    LevenbergMarquardtMethod, LevenbergMarquardtMinpack, NewtonMethod,
    NielsenLevenbergMarquardtMethod, NielsenLevenbergMarquardtMethodAdvanced, NonlinearProblem,
    NonlinearSolverMethod, PowellDoglegMethod, SolveOptions, TrustRegionLMMethod,
    TrustRegionMethod,
};

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarding the allocator contract to the platform allocator.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the pointer/layout pair is supplied by Rust's allocation API.
        unsafe { System.dealloc(pointer, layout) };
        DEALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        DEALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: forwarding the allocator contract to the platform allocator.
        let new_pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !new_pointer.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(new_size, Ordering::Relaxed);
            DEALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            DEALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        new_pointer
    }
}

#[derive(Clone, Copy, Default)]
struct AllocationSample {
    allocations: usize,
    deallocations: usize,
    allocated_bytes: usize,
    deallocated_bytes: usize,
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
}

fn counters() -> AllocationSample {
    AllocationSample {
        allocations: ALLOCATIONS.load(Ordering::Relaxed),
        deallocations: DEALLOCATIONS.load(Ordering::Relaxed),
        allocated_bytes: ALLOCATED_BYTES.load(Ordering::Relaxed),
        deallocated_bytes: DEALLOCATED_BYTES.load(Ordering::Relaxed),
    }
}

struct DenseQuadraticProblem {
    dimension: usize,
}

/// Same problem with caller-owned callback output support enabled.
struct ReusableDenseQuadraticProblem {
    dimension: usize,
}

impl NonlinearProblem for DenseQuadraticProblem {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn residual(&self, x: &DVector<f64>) -> Result<DVector<f64>, SolveError> {
        Ok(DVector::from_iterator(
            self.dimension,
            x.iter().map(|value| value * value - 1.0),
        ))
    }
}

impl JacobianProvider for DenseQuadraticProblem {
    fn jacobian(&self, x: &DVector<f64>) -> Result<DMatrix<f64>, SolveError> {
        Ok(DMatrix::from_fn(
            self.dimension,
            self.dimension,
            |row, column| {
                if row == column { 2.0 * x[row] } else { 0.0 }
            },
        ))
    }
}

impl NonlinearProblem for ReusableDenseQuadraticProblem {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn supports_residual_into(&self) -> bool {
        true
    }

    fn residual(&self, x: &DVector<f64>) -> Result<DVector<f64>, SolveError> {
        Ok(DVector::from_iterator(
            self.dimension,
            x.iter().map(|value| value * value - 1.0),
        ))
    }

    fn residual_into(&self, x: &DVector<f64>, out: &mut DVector<f64>) -> Result<(), SolveError> {
        if out.len() != self.dimension {
            return Err(SolveError::DimensionMismatch {
                expected: self.dimension,
                actual: out.len(),
                context: "reusable residual output",
            });
        }
        for (slot, value) in out.iter_mut().zip(x.iter()) {
            *slot = value * value - 1.0;
        }
        Ok(())
    }
}

impl JacobianProvider for ReusableDenseQuadraticProblem {
    fn supports_jacobian_into(&self) -> bool {
        true
    }

    fn jacobian(&self, x: &DVector<f64>) -> Result<DMatrix<f64>, SolveError> {
        Ok(DMatrix::from_fn(
            self.dimension,
            self.dimension,
            |row, column| {
                if row == column { 2.0 * x[row] } else { 0.0 }
            },
        ))
    }

    fn jacobian_into(&self, x: &DVector<f64>, out: &mut DMatrix<f64>) -> Result<(), SolveError> {
        if out.shape() != (self.dimension, self.dimension) {
            return Err(SolveError::InvalidConfig(
                "reusable Jacobian output has an unexpected shape".to_string(),
            ));
        }
        for row in 0..self.dimension {
            for column in 0..self.dimension {
                out[(row, column)] = if row == column { 2.0 * x[row] } else { 0.0 };
            }
        }
        Ok(())
    }
}

struct RosenbrockProblem;

impl NonlinearProblem for RosenbrockProblem {
    fn dimension(&self) -> usize {
        2
    }

    fn supports_residual_into(&self) -> bool {
        true
    }

    fn residual(&self, x: &DVector<f64>) -> Result<DVector<f64>, SolveError> {
        Ok(DVector::from_vec(vec![
            10.0 * (x[1] - x[0] * x[0]),
            1.0 - x[0],
        ]))
    }

    fn residual_into(&self, x: &DVector<f64>, out: &mut DVector<f64>) -> Result<(), SolveError> {
        if out.len() != 2 {
            return Err(SolveError::DimensionMismatch {
                expected: 2,
                actual: out.len(),
                context: "reusable Rosenbrock residual output",
            });
        }
        out[0] = 10.0 * (x[1] - x[0] * x[0]);
        out[1] = 1.0 - x[0];
        Ok(())
    }
}

impl JacobianProvider for RosenbrockProblem {
    fn jacobian(&self, x: &DVector<f64>) -> Result<DMatrix<f64>, SolveError> {
        Ok(DMatrix::from_row_slice(
            2,
            2,
            &[-20.0 * x[0], 10.0, -1.0, 0.0],
        ))
    }
}

fn methods() -> Vec<NonlinearSolverMethod> {
    vec![
        NonlinearSolverMethod::Newton(NewtonMethod),
        NonlinearSolverMethod::DampedNewton(DampedNewtonMethod::default()),
        NonlinearSolverMethod::DampedNewtonAdvanced(DampedNewtonMethodAdvanced::default()),
        NonlinearSolverMethod::LevenbergMarquardt(LevenbergMarquardtMethod::default()),
        NonlinearSolverMethod::LevenbergMarquardtMinpack(LevenbergMarquardtMinpack::default()),
        NonlinearSolverMethod::NielsenLevenbergMarquardt(NielsenLevenbergMarquardtMethod::default()),
        NonlinearSolverMethod::NielsenLevenbergMarquardtAdvanced(
            NielsenLevenbergMarquardtMethodAdvanced::default(),
        ),
        NonlinearSolverMethod::TrustRegion(TrustRegionMethod::default()),
        NonlinearSolverMethod::PowellDogleg(PowellDoglegMethod::default()),
        NonlinearSolverMethod::TrustRegionLM(TrustRegionLMMethod::default()),
    ]
}

fn options(collect_history: bool) -> SolveOptions {
    SolveOptions {
        tolerance: 1e-8,
        max_iterations: 64,
        diagnostics: DiagnosticsOptions {
            collect_statistics: false,
            collect_history,
            enable_logging: false,
            ..DiagnosticsOptions::default()
        },
        ..SolveOptions::default()
    }
}

fn audit_one(
    method_template: &NonlinearSolverMethod,
    problem: &impl JacobianProvider,
    initial: &DVector<f64>,
    solve_options: SolveOptions,
) -> (AllocationSample, f64) {
    // Prepare all caller-owned values before resetting counters. The measured
    // region therefore focuses on the solver engine and method hot path.
    let method = method_template.clone();
    let initial = initial.clone();
    reset_counters();
    let started = Instant::now();
    let result = method.solve(problem, initial, solve_options);
    let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
    black_box(&result);
    drop(result);
    (counters(), elapsed_ms)
}

fn average(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn print_audit(
    scenario: &str,
    method: &NonlinearSolverMethod,
    problem: &impl JacobianProvider,
    initial: &DVector<f64>,
    solve_options: SolveOptions,
    rejected_label: &str,
    dimension: usize,
) {
    const RUNS: usize = 5;
    let mut samples = [AllocationSample::default(); RUNS];
    let mut elapsed = [0.0; RUNS];
    for run in 0..RUNS {
        (samples[run], elapsed[run]) = audit_one(method, problem, initial, solve_options.clone());
    }
    let allocs = samples
        .iter()
        .map(|sample| sample.allocations as f64)
        .collect::<Vec<_>>();
    let deallocs = samples
        .iter()
        .map(|sample| sample.deallocations as f64)
        .collect::<Vec<_>>();
    let allocated_bytes = samples
        .iter()
        .map(|sample| sample.allocated_bytes as f64)
        .collect::<Vec<_>>();
    let deallocated_bytes = samples
        .iter()
        .map(|sample| sample.deallocated_bytes as f64)
        .collect::<Vec<_>>();
    println!(
        "{dimension:9} | {scenario:25} | {:28} | runs={RUNS} | allocs={:.1} | alloc_bytes={:.1} | deallocs={:.1} | dealloc_bytes={:.1} | elapsed_ms={:.3} | rejected_preflight={rejected_label}",
        method.name(),
        average(&allocs),
        average(&allocated_bytes),
        average(&deallocs),
        average(&deallocated_bytes),
        average(&elapsed),
    );
}

fn print_trust_region_ownership_audit() {
    const DIMENSIONS: &[usize] = &[128, 512];
    const RUNS: usize = 5;
    let method = NonlinearSolverMethod::TrustRegionLM(TrustRegionLMMethod {
        step_bound: 100.0,
        ..TrustRegionLMMethod::default()
    });

    println!(
        "[Nonlinear TrustRegionLM ownership audit] runs={RUNS}; dimensions=128,512; allocation counters include the complete solve result lifetime"
    );
    println!(
        "dimension | scenario                  | rejected | iterations | residuals | jacobians | linear | allocs mean | alloc_bytes mean | elapsed_ms mean | status"
    );
    for &dimension in DIMENSIONS {
        let problem = DenseQuadraticProblem { dimension };
        for (scenario, initial_value) in [("accepted", 0.9), ("rejection-heavy", 0.25)] {
            let initial = DVector::from_element(dimension, initial_value);
            let preflight = method
                .clone()
                .solve(
                    &problem,
                    initial.clone(),
                    SolveOptions {
                        max_iterations: 128,
                        diagnostics: DiagnosticsOptions {
                            collect_statistics: true,
                            ..DiagnosticsOptions::default()
                        },
                        ..options(false)
                    },
                )
                .expect("TrustRegionLM ownership preflight should finish");
            let rejected_steps = preflight.statistics.rejected_steps;
            let iterations = preflight.statistics.iterations;
            let residual_calls = preflight.statistics.residual_evaluations;
            let jacobian_calls = preflight.statistics.jacobian_evaluations;
            let linear_solves = preflight.statistics.linear_solves;

            let mut samples = [AllocationSample::default(); RUNS];
            let mut elapsed = [0.0; RUNS];
            for run in 0..RUNS {
                (samples[run], elapsed[run]) =
                    audit_one(&method, &problem, &initial, options(false));
            }
            let allocs = samples
                .iter()
                .map(|sample| sample.allocations as f64)
                .collect::<Vec<_>>();
            let allocated_bytes = samples
                .iter()
                .map(|sample| sample.allocated_bytes as f64)
                .collect::<Vec<_>>();
            println!(
                "{dimension:9} | {scenario:25} | {rejected_steps:8} | {iterations:10} | {residual_calls:9} | {jacobian_calls:9} | {linear_solves:6} | {:11.1} | {:16.1} | {:16.3} | ok",
                average(&allocs),
                average(&allocated_bytes),
                average(&elapsed),
            );
        }
    }
}

fn main() {
    if std::env::var_os("NONLINEAR_TRUST_REGION_OWNERSHIP_ONLY").is_some() {
        print_trust_region_ownership_audit();
        return;
    }

    const DIMENSIONS: &[usize] = &[32, 128, 512];
    let rejected_problem = RosenbrockProblem;
    let rejected_initial = DVector::from_vec(vec![-1.2, 1.0]);
    let all_methods = methods();

    let preflight = NonlinearSolverMethod::DampedNewton(DampedNewtonMethod::default())
        .solve(
            &rejected_problem,
            rejected_initial.clone(),
            SolveOptions {
                diagnostics: DiagnosticsOptions {
                    collect_statistics: true,
                    ..DiagnosticsOptions::default()
                },
                ..options(false)
            },
        )
        .expect("rejected-step preflight should finish");
    assert!(
        preflight.statistics.rejected_steps > 0,
        "rejected-step workload disappeared"
    );
    let rejected_preflight = preflight.statistics.rejected_steps.to_string();

    println!(
        "[Nonlinear allocation audit] runs=5; dimensions=32,128,512; allocation counters include the complete solve result lifetime"
    );
    println!(
        "dimension | scenario                  | method | allocs | alloc_bytes | deallocs | dealloc_bytes | elapsed_ms | rejected_preflight"
    );
    for &dimension in DIMENSIONS {
        let accepted_problem = DenseQuadraticProblem { dimension };
        let accepted_initial = DVector::from_element(dimension, 0.25);
        for method in &all_methods {
            print_audit(
                "accepted/history-off",
                method,
                &accepted_problem,
                &accepted_initial,
                options(false),
                "not-measured",
                dimension,
            );
        }
        // History is an explicit opt-in, so compare it at the smallest and
        // largest audit dimensions without doubling every intermediate row.
        if dimension == DIMENSIONS[0] || dimension == DIMENSIONS[DIMENSIONS.len() - 1] {
            for method in &all_methods {
                print_audit(
                    "accepted/history-on",
                    method,
                    &accepted_problem,
                    &accepted_initial,
                    options(true),
                    "not-measured",
                    dimension,
                );
            }
        }
    }
    for method in &all_methods {
        print_audit(
            "rejected/history-off",
            method,
            &rejected_problem,
            &rejected_initial,
            options(false),
            &rejected_preflight,
            rejected_problem.dimension(),
        );
    }

    println!(
        "[Nonlinear allocation audit] reusable callback output comparison; same solve and options"
    );
    println!(
        "dimension | scenario                  | method | allocs | alloc_bytes | deallocs | dealloc_bytes | elapsed_ms | rejected_preflight"
    );
    for &dimension in DIMENSIONS {
        let reusable_problem = ReusableDenseQuadraticProblem { dimension };
        let accepted_initial = DVector::from_element(dimension, 0.25);
        for method in [
            NonlinearSolverMethod::Newton(NewtonMethod),
            NonlinearSolverMethod::DampedNewton(DampedNewtonMethod::default()),
        ] {
            print_audit(
                "reusable-callback/history-off",
                &method,
                &reusable_problem,
                &accepted_initial,
                options(false),
                "not-measured",
                dimension,
            );
        }
    }
}
