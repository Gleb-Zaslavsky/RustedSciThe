#![cfg(test)]

use super::codegen_bvp_performance_tests::{
    run_diagnose_chunk_granularity_and_fallback,
    run_diagnose_combustion_chunk_ir_amplification,
    run_diagnose_combustion_banded_native_zero_pivot,
    run_diagnose_combustion_superblock_groupings,
    run_diagnose_problem_size_crossover,
    run_diagnose_rayon_overhead_baseline,
};

/// Structural combustion diagnostic kept outside the main perf module so the
/// benchmark file can stay focused on comparable timing stories.
#[test]
#[ignore = "diagnostic for combustion native block-tridiagonal zero-pivot"]
fn diagnose_combustion_banded_native_zero_pivot() {
    run_diagnose_combustion_banded_native_zero_pivot();
}

/// Superblock grouping lab notebook for the combustion banded path.
#[test]
#[ignore = "diagnostic for combustion superblock grouping experiments"]
fn diagnose_combustion_superblock_groupings() {
    run_diagnose_combustion_superblock_groupings();
}

/// Measures raw rayon scheduling overhead without mixing it into perf stories.
#[test]
fn diagnose_rayon_overhead_baseline() {
    run_diagnose_rayon_overhead_baseline();
}

/// Prints chunk-granularity / fallback behavior for the compiled stress path.
#[test]
fn diagnose_chunk_granularity_and_fallback() {
    run_diagnose_chunk_granularity_and_fallback();
}

/// Prints whole-vs-chunk IR/source amplification for combustion BVP callbacks.
#[test]
#[ignore = "diagnostic for chunked IR/source amplification; use BVP_CHUNK_IR_STEPS to scale"]
fn diagnose_combustion_chunk_ir_amplification() {
    run_diagnose_combustion_chunk_ir_amplification();
}

/// IR-level crossover sweep for when parallel sparse Jacobian execution wins.
#[test]
fn diagnose_problem_size_crossover() {
    run_diagnose_problem_size_crossover();
}
