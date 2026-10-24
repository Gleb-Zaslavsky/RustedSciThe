//! Allocation audit for the warmed, caller-owned `PreparedAtomSystem` path.
//!
//! This is intentionally separate from Criterion timing: the global counting
//! allocator changes allocation cost and its numbers must not be compared to
//! wall-clock benchmarks. The fixture contains only built-in numeric
//! expressions and caller-owned buffers, so no custom callback allocations are
//! included. Scratch capacity is reported as a separate retained-workspace
//! proxy; neither metric is a process RSS measurement.
//!
//! Run with:
//!
//! ```text
//! cargo bench --bench symbolic_prepared_memory_audit
//! SYMBOLIC_PREPARED_MEMORY_REPEATS=5000 cargo bench --bench symbolic_prepared_memory_audit
//! ```

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use RustedSciThe::symbolic::{
    View::{PreparedAtomSystem, PreparedAtomSystemScratch},
    symbolic_engine::Expr,
};

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

// SAFETY: every operation forwards the exact allocation contract to System.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: `layout` is supplied by the GlobalAlloc caller.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: pointer/layout are forwarded unchanged to the backing allocator.
        unsafe { System.dealloc(pointer, layout) };
        DEALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        DEALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: pointer/layout/new_size are forwarded unchanged to System.
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

fn reset_allocations() {
    for counter in [
        &ALLOCATIONS,
        &DEALLOCATIONS,
        &ALLOCATED_BYTES,
        &DEALLOCATED_BYTES,
    ] {
        counter.store(0, Ordering::Relaxed);
    }
}

fn main() {
    let repeats = std::env::var("SYMBOLIC_PREPARED_MEMORY_REPEATS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(1_000);

    println!(
        "PreparedAtomSystem warmed allocation audit; repeats={repeats}; custom callbacks=none; outputs and scratch preallocated"
    );
    println!(
        "dimension | jacobian_nnz | allocations/eval | allocated_bytes/eval | deallocations/eval | deallocated_bytes/eval | scratch_capacity | elapsed_ms"
    );

    for dimension in [128, 512] {
        let input_names = (0..dimension)
            .map(|index| format!("y{index}"))
            .collect::<Vec<_>>();
        let residuals = (0..dimension)
            .map(|row| {
                if row == 0 {
                    Expr::parse_expression("y0^2")
                } else {
                    Expr::parse_expression(&format!("y{row}^2 + 0.125*y{}", row - 1))
                }
            })
            .collect::<Vec<_>>();
        let prepared = PreparedAtomSystem::try_from_exprs(&residuals, &input_names, None)
            .expect("sparse-chain prepared system should compile");
        let mut input = (0..dimension)
            .map(|index| 0.5 + index as f64 * 1.0e-3)
            .collect::<Vec<_>>();
        let mut residual_values = vec![0.0; prepared.residual_len()];
        let mut jacobian_values = vec![0.0; prepared.jacobian_len()];
        let mut scratch = PreparedAtomSystemScratch::default();

        prepared
            .residual_and_jacobian_into(
                &input,
                &mut residual_values,
                &mut jacobian_values,
                &mut scratch,
            )
            .expect("warmup evaluation should succeed");
        let scratch_capacity = scratch.capacity();

        reset_allocations();
        let started = Instant::now();
        for iteration in 0..repeats {
            let shift = iteration as f64 * 1.0e-9;
            for (index, value) in input.iter_mut().enumerate() {
                *value = 0.5 + index as f64 * 1.0e-3 + shift;
            }
            prepared
                .residual_and_jacobian_into(
                    black_box(&input),
                    &mut residual_values,
                    &mut jacobian_values,
                    &mut scratch,
                )
                .expect("warmed finite callback should succeed");
            black_box((&residual_values, &jacobian_values));
        }
        let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;

        let allocation_count = ALLOCATIONS.load(Ordering::Relaxed);
        let deallocation_count = DEALLOCATIONS.load(Ordering::Relaxed);
        let allocated_bytes = ALLOCATED_BYTES.load(Ordering::Relaxed);
        let deallocated_bytes = DEALLOCATED_BYTES.load(Ordering::Relaxed);
        println!(
            "{dimension:>9} | {:>12} | {:>16.4} | {:>20.2} | {:>18.4} | {:>22.2} | {:>15} | {elapsed_ms:.3}",
            prepared.jacobian_len(),
            allocation_count as f64 / repeats as f64,
            allocated_bytes as f64 / repeats as f64,
            deallocation_count as f64 / repeats as f64,
            deallocated_bytes as f64 / repeats as f64,
            scratch_capacity,
        );
        assert_eq!(
            scratch.capacity(),
            scratch_capacity,
            "warmed scratch capacity changed during the audit"
        );
    }
}
