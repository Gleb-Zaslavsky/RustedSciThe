#![cfg(test)]

use super::codegen_bvp_performance_tests::{
    run_benchmark_compiled_above_threshold_stress_multifield_aot_runtime,
    run_benchmark_compiled_large_stress_multifield_aot_runtime,
    run_benchmark_compiled_stress_multifield_aot_runtime,
    run_benchmark_compiled_stress_sparse_parallel_job_counts,
    run_benchmark_compiled_xlarge_stress_seq_vs_par,
    run_benchmark_stress_multifield_bvp_pipeline,
    run_benchmark_stress_multifield_parallel_crossover,
    run_benchmark_stress_multifield_scaling_sweep,
};

/// Large multifield stress diagnostic kept outside the core perf module so the
/// benchmark file stays focused on user-facing timing tables.
#[test]
#[ignore = "stress-scale diagnostic; run explicitly with optional RST_STRESS_FIELDS / RST_STRESS_STEPS"]
fn benchmark_stress_multifield_bvp_pipeline() {
    run_benchmark_stress_multifield_bvp_pipeline();
}

/// IR-level scaling sweep for larger multifield systems.
#[test]
#[ignore = "stress-scale field-count sweep for larger multifield systems"]
fn benchmark_stress_multifield_scaling_sweep() {
    run_benchmark_stress_multifield_scaling_sweep();
}

/// Searches for the crossover point where parallel AOT IR starts to win.
#[test]
#[ignore = "stress-scale crossover search for sequential vs parallel AOT IR"]
fn benchmark_stress_multifield_parallel_crossover() {
    run_benchmark_stress_multifield_parallel_crossover();
}

/// Compiled stress AOT runtime on the checked-in baseline fixture.
#[test]
#[ignore = "stress-scale compiled AOT runtime diagnostic"]
fn benchmark_compiled_stress_multifield_aot_runtime() {
    run_benchmark_compiled_stress_multifield_aot_runtime();
}

/// Larger compiled stress AOT runtime fixture.
#[test]
fn benchmark_compiled_large_stress_multifield_aot_runtime() {
    run_benchmark_compiled_large_stress_multifield_aot_runtime();
}

/// Above-threshold compiled stress AOT runtime fixture.
#[test]
#[ignore = "compiled above-threshold stress AOT runtime diagnostic"]
fn benchmark_compiled_above_threshold_stress_multifield_aot_runtime() {
    run_benchmark_compiled_above_threshold_stress_multifield_aot_runtime();
}

/// Sparse parallel job-count sweep for the compiled stress fixture.
#[test]
#[ignore = "stress-scale sparse parallel job sweep"]
fn benchmark_compiled_stress_sparse_parallel_job_counts() {
    run_benchmark_compiled_stress_sparse_parallel_job_counts();
}

/// Sequential vs parallel comparison on the xlarge compiled fixture.
#[test]
fn benchmark_compiled_xlarge_stress_seq_vs_par() {
    run_benchmark_compiled_xlarge_stress_seq_vs_par();
}
