//! Focused benchmark for the pure-numerical BVP Jacobian assembly boundary.
//!
//! The benchmark deliberately stops before factorization and Newton iteration.
//! Dense is the compatibility reference with row-major `N^2` staging; Sparse
//! and Banded exercise the direct structural-triplet path. Allocation numbers
//! include the complete lifetime of each returned matrix and therefore are not
//! peak RSS measurements.
//!
//! Run with:
//!
//! ```text
//! cargo bench --bench bvp_numeric_assembly -- --noplot
//! cargo bench --bench bvp_numeric_assembly -- bvp_numeric_jacobian --noplot --sample-size 10
//! ```

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nalgebra::{DMatrix, DVector};
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use RustedSciThe::numerical::BVP_Damp::BVP_traits::{Jac, VectorType, Vectors_type_casting};
use RustedSciThe::numerical::BVP_Damp::numeric_discretization::{
    NumericBvpJacobian, NumericBvpRhs, build_numeric_generated_solver_state,
};

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: delegate to the platform allocator.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: delegate the original allocation contract unchanged.
        unsafe { System.dealloc(pointer, layout) };
        DEALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        DEALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: delegate to the platform allocator.
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
    allocated_bytes: usize,
    deallocations: usize,
    deallocated_bytes: usize,
}

fn reset_allocations() {
    for counter in [
        &ALLOCATIONS,
        &ALLOCATED_BYTES,
        &DEALLOCATIONS,
        &DEALLOCATED_BYTES,
    ] {
        counter.store(0, Ordering::Relaxed);
    }
}

fn allocations() -> AllocationSample {
    AllocationSample {
        allocations: ALLOCATIONS.load(Ordering::Relaxed),
        allocated_bytes: ALLOCATED_BYTES.load(Ordering::Relaxed),
        deallocations: DEALLOCATIONS.load(Ordering::Relaxed),
        deallocated_bytes: DEALLOCATED_BYTES.load(Ordering::Relaxed),
    }
}

fn prepared_jacobian(method: &str, n_steps: usize) -> (Box<dyn Jac>, Box<dyn VectorType>, usize) {
    let values = vec!["y".to_string(), "z".to_string()];
    let mesh = (0..=n_steps)
        .map(|index| index as f64 / n_steps as f64)
        .collect::<Vec<_>>();
    let border_conditions = HashMap::from([
        ("y".to_string(), vec![(0usize, 0.0)]),
        ("z".to_string(), vec![(1usize, 1.0)]),
    ]);
    let bounds = HashMap::from([
        ("y".to_string(), (-2.0, 2.0)),
        ("z".to_string(), (-2.0, 2.0)),
    ]);
    let rel_tolerance = HashMap::from([("y".to_string(), 1e-8), ("z".to_string(), 1e-8)]);

    let rhs: NumericBvpRhs = Arc::new(|_, state, _| DVector::from_vec(vec![state[1], -state[0]]));
    let jacobian: NumericBvpJacobian =
        Arc::new(|_, _, _| DMatrix::from_row_slice(2, 2, &[0.0, 1.0, -1.0, 0.0]));

    let state = build_numeric_generated_solver_state(
        rhs,
        Some(jacobian),
        method,
        "forward",
        &values,
        &border_conditions,
        &bounds,
        &rel_tolerance,
        n_steps,
        &mesh,
        None,
        None,
    )
    .expect("numeric benchmark state should build");

    let n_unknowns = values.len() * n_steps;
    let initial = DVector::from_element(n_unknowns, 0.25);
    let vector = Vectors_type_casting(&initial, method.to_string());
    (
        state.jac.expect("numeric benchmark must expose Jacobian"),
        vector,
        n_unknowns,
    )
}

fn report_allocations() {
    const RUNS: usize = 7;
    println!("[BVP numeric assembly] runs={RUNS}; callback output lifetime included");
    println!(
        "method | n_steps | unknowns | callback_ms mean | allocs/call | alloc_bytes/call | deallocs/call | dealloc_bytes/call"
    );
    println!(
        "----------------------------------------------------------------------------------------------------------------"
    );

    for n_steps in [64usize, 256, 512] {
        for method in ["Dense", "Sparse", "Banded"] {
            let (mut jacobian, vector, n_unknowns) = prepared_jacobian(method, n_steps);
            reset_allocations();
            let started = Instant::now();
            for _ in 0..RUNS {
                black_box(jacobian.call(0.5, vector.as_ref()));
            }
            let elapsed_ms = started.elapsed().as_secs_f64() * 1e3 / RUNS as f64;
            let sample = allocations();
            println!(
                "{method:6} | {n_steps:7} | {n_unknowns:8} | {elapsed_ms:16.3} | {:12.1} | {:16.1} | {:14.1} | {:17.1}",
                sample.allocations as f64 / RUNS as f64,
                sample.allocated_bytes as f64 / RUNS as f64,
                sample.deallocations as f64 / RUNS as f64,
                sample.deallocated_bytes as f64 / RUNS as f64,
            );
        }
    }
}

fn benchmark_numeric_jacobian(c: &mut Criterion) {
    report_allocations();
    let mut group = c.benchmark_group("bvp_numeric_jacobian");
    group.sample_size(10);

    for n_steps in [64usize, 256, 512] {
        for method in ["Dense", "Sparse", "Banded"] {
            let (mut jacobian, vector, _) = prepared_jacobian(method, n_steps);
            group.bench_function(BenchmarkId::new(method, n_steps), |bencher| {
                bencher.iter(|| black_box(jacobian.call(black_box(0.5), vector.as_ref())));
            });
        }
    }
    group.finish();
}

criterion_group!(benches, benchmark_numeric_jacobian);
criterion_main!(benches);
