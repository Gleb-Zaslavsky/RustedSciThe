#![cfg(test)]

//! Shared helpers for codegen benchmarks and regression tests.
//!
//! This module is intentionally small at first. It hosts generic timing,
//! environment-driven benchmark settings, and reusable stress-problem builders
//! so that BVP, dense, and IVP codegen tests do not each grow their own copy
//! of the same support code.

use crate::symbolic::codegen_orchestrator::{ParallelExecutorConfig, ParallelFallbackPolicy};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions_BVP::Jacobian;
use std::collections::HashMap;
use std::env;
use std::time::Duration;

pub(crate) fn benchmark_parallel_config() -> ParallelExecutorConfig {
    ParallelExecutorConfig {
        jobs_per_worker: 1,
        max_residual_jobs: None,
        max_sparse_jobs: None,
        fallback_policy: ParallelFallbackPolicy::Never,
    }
}

pub(crate) fn benchmark_sparse_parallel_config_with_jobs(
    max_sparse_jobs: usize,
) -> ParallelExecutorConfig {
    ParallelExecutorConfig {
        jobs_per_worker: 1,
        max_residual_jobs: None,
        max_sparse_jobs: Some(max_sparse_jobs),
        fallback_policy: ParallelFallbackPolicy::Never,
    }
}

pub(crate) fn env_usize(key: &str, default_value: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default_value)
}

pub(crate) fn stress_samples(default_value: usize) -> usize {
    env_usize("RST_STRESS_SAMPLES", default_value)
}

pub(crate) fn stress_residual_iters(default_value: usize) -> usize {
    env_usize("RST_STRESS_RESIDUAL_ITERS", default_value)
}

pub(crate) fn stress_jacobian_iters(default_value: usize) -> usize {
    env_usize("RST_STRESS_JACOBIAN_ITERS", default_value)
}

pub(crate) fn stress_env_list_usize(key: &str, default_values: &[usize]) -> Vec<usize> {
    env::var(key)
        .ok()
        .map(|raw| {
            raw.split(',')
                .filter_map(|part| part.trim().parse::<usize>().ok())
                .collect::<Vec<_>>()
        })
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| default_values.to_vec())
}

pub(crate) fn median_duration(mut samples: Vec<Duration>) -> Duration {
    samples.sort_by_key(|duration| duration.as_nanos());
    samples[samples.len() / 2]
}

pub(crate) fn per_iter_ns(duration: Duration, iterations: usize) -> f64 {
    duration.as_secs_f64() * 1e9 / iterations as f64
}

pub(crate) fn build_multifield_stress_bvp_case(field_count: usize, n_steps: usize) -> Jacobian {
    assert!(field_count >= 2, "field_count must be at least 2");

    let values: Vec<String> = (0..field_count).map(|i| format!("u{i}")).collect();
    let mut eq_system = Vec::with_capacity(field_count);

    for i in 0..field_count {
        let self_var = Expr::Var(format!("u{i}"));
        let next_var = Expr::Var(format!("u{}", (i + 1) % field_count));
        let prev_var = Expr::Var(format!("u{}", (i + field_count - 1) % field_count));

        let linear_damping = Expr::Const(-(0.35 + 0.01 * i as f64)) * self_var.clone();
        let forward_coupling =
            Expr::Const(0.20 + 0.002 * i as f64) * Expr::sin(Box::new(next_var.clone()));
        let backward_coupling = Expr::Const(0.05) * (prev_var.clone() - self_var.clone());
        let cubic_local = Expr::Const(-0.015) * self_var.clone().pow(Expr::Const(3.0));
        let weak_quadratic = Expr::Const(0.01) * next_var.clone().pow(Expr::Const(2.0));

        eq_system.push(
            linear_damping + forward_coupling + backward_coupling + cubic_local + weak_quadratic,
        );
    }

    let mut border_conditions = HashMap::new();
    for (i, name) in values.iter().enumerate() {
        let side = i % 2;
        let value = 0.1 + i as f64 * 0.01;
        border_conditions.insert(name.clone(), vec![(side, value)]);
    }

    let mut jac = Jacobian::new();
    jac.discretization_system_BVP_par(
        eq_system,
        values,
        "x".to_string(),
        0.0,
        Some(n_steps),
        None,
        None,
        border_conditions,
        None,
        None,
        "forward".to_string(),
    );
    jac.calc_jacobian_parallel_smart_optimized();
    jac
}
