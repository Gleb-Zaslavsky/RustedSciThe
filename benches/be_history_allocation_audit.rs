//! Allocation-only audit for BE trajectory history assembly.
//!
//! Run separately from Criterion. The counting allocator perturbs latency, so
//! this target reports allocation counts/bytes only and makes no timing claim.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};

use nalgebra::{DMatrix, DVector};

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwards the allocator contract to the system allocator.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: forwards the allocator contract to the system allocator.
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: forwards the allocator contract to the system allocator.
        let new_pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !new_pointer.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(new_size, Ordering::Relaxed);
        }
        new_pointer
    }
}

#[derive(Clone, Copy)]
struct AllocationCounts {
    allocations: usize,
    bytes: usize,
}

fn reset_counts() {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
}

fn counts() -> AllocationCounts {
    AllocationCounts {
        allocations: ALLOCATIONS.load(Ordering::Relaxed),
        bytes: ALLOCATED_BYTES.load(Ordering::Relaxed),
    }
}

fn legacy_clone_flatten(input: &[f64], rows: usize, columns: usize) -> DMatrix<f64> {
    let states: Vec<DVector<f64>> = input
        .chunks_exact(columns)
        .map(|row| DVector::from_row_slice(row))
        .collect();
    let mut flattened = Vec::with_capacity(input.len());
    for state in states {
        flattened.extend(state.iter().copied());
    }
    DMatrix::from_vec(columns, rows, flattened).transpose()
}

fn streamed_row_slice(input: &[f64], rows: usize, columns: usize) -> DMatrix<f64> {
    DMatrix::from_row_slice(rows, columns, input)
}

fn streamed_owned_transpose(input: Vec<f64>, rows: usize, columns: usize) -> DMatrix<f64> {
    DMatrix::from_vec(columns, rows, input).transpose()
}

fn measure(operation: impl FnOnce() -> DMatrix<f64>) -> AllocationCounts {
    reset_counts();
    black_box(operation());
    counts()
}

fn measure_owned_transpose(input: &[f64], rows: usize, columns: usize) -> AllocationCounts {
    let owned_input = input.to_vec();
    measure(|| streamed_owned_transpose(owned_input, rows, columns))
}

fn main() {
    println!(
        "[BE history allocation audit] instrumented process; counts/bytes only; setup excluded"
    );
    println!(
        "rows | states | legacy_allocations | legacy_bytes | row_slice_allocations | row_slice_bytes | owned_transpose_allocations | owned_transpose_bytes | parity"
    );
    for (rows, columns) in [
        (32, 3),
        (128, 12),
        (64, 64),
        (128, 64),
        (512, 32),
        (1024, 64),
    ] {
        let input = vec![0.75; rows * columns];
        let legacy = measure(|| legacy_clone_flatten(&input, rows, columns));
        let row_slice = measure(|| streamed_row_slice(&input, rows, columns));
        let owned = measure_owned_transpose(&input, rows, columns);
        let expected = legacy_clone_flatten(&input, rows, columns);
        let parity = expected == streamed_row_slice(&input, rows, columns)
            && expected == streamed_owned_transpose(input.clone(), rows, columns);
        println!(
            "{rows} | {columns} | {} | {} | {} | {} | {} | {} | {parity}",
            legacy.allocations,
            legacy.bytes,
            row_slice.allocations,
            row_slice.bytes,
            owned.allocations,
            owned.bytes
        );
        assert!(parity, "history assembly outputs must match");
    }
}
