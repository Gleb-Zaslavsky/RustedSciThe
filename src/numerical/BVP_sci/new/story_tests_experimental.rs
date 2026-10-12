//! Ignored, opt-in experiments for unpromoted BVP_sci sparse factorization
//! strategies. These are intentionally separate from stable story gates.

#[test]
#[ignore = "experimental release A/B: reusable faer and optional SuperLU on a large nonlinear workload"]
fn sparse_reusable_kernel_full_solve_continuation_ab_is_compact() {
    super::story_tests::run_sparse_reusable_kernel_full_solve_continuation_ab_is_compact();
}

#[test]
#[ignore = "experimental release A/B: compare sparse factorization policies across real BVP workloads"]
fn sparse_kernel_real_workload_matrix_ab_is_compact() {
    super::story_tests::run_sparse_kernel_real_workload_matrix_ab_is_compact();
}

#[test]
#[ignore = "experimental release A/B: official 60-state continuation policy comparison"]
fn sparse_kernel_official_60_state_continuation_ab_is_compact() {
    super::story_tests::run_sparse_kernel_official_60_state_continuation_ab_is_compact();
}
