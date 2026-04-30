#![cfg(test)]

//! Shared helpers for codegen benchmarks and regression tests.
//!
//! This module is intentionally small at first. It hosts generic timing,
//! environment-driven benchmark settings, and reusable stress-problem builders
//! so that BVP, dense, and IVP codegen tests do not each grow their own copy
//! of the same support code.

use crate::symbolic::codegen::codegen_orchestrator::{
    ParallelExecutorConfig, ParallelFallbackPolicy,
};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions_BVP::{BvpSymbolicAssemblyBackend, Jacobian};
use std::collections::HashMap;
use std::env;
use std::process::Command;
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

pub(crate) fn build_real_bvp_damp1_case(n_steps: usize) -> Jacobian {
    build_real_bvp_damp1_case_with_backend(n_steps, BvpSymbolicAssemblyBackend::ExprLegacy)
}

pub(crate) fn build_real_bvp_damp1_case_with_backend(
    n_steps: usize,
    symbolic_backend: BvpSymbolicAssemblyBackend,
) -> Jacobian {
    let eq1 = Expr::parse_expression("y-z");
    let eq2 = Expr::parse_expression("-z^3");
    let eq_system = vec![eq1, eq2];
    let values = vec!["z".to_string(), "y".to_string()];

    let mut border_conditions = HashMap::new();
    border_conditions.insert("z".to_string(), vec![(0usize, 1.0f64)]);
    border_conditions.insert("y".to_string(), vec![(1usize, 1.0f64)]);

    let mut jac = Jacobian::new();
    jac.set_symbolic_assembly_backend(symbolic_backend);
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

pub(crate) fn combustion_bvp_fixture(
    n_steps: usize,
) -> (
    Vec<Expr>,
    Vec<String>,
    HashMap<String, Vec<(usize, f64)>>,
) {
    let unknowns_str: Vec<&str> = vec!["Teta", "q", "C0", "J0", "C1", "J1"];
    let unknowns: Vec<Expr> = Expr::parse_vector_expression(unknowns_str.clone());
    let teta = unknowns[0].clone();
    let q = unknowns[1].clone();
    let c0 = unknowns[2].clone();
    let j0 = unknowns[3].clone();
    let j1 = unknowns[5].clone();

    let q_heat = 3000.0 * 1e3 * 0.034;
    let dt = 600.0;
    let t_scale = 600.0;
    let l: f64 = 3e-4;
    let m0 = 34.2 / 1000.0;
    let lambda = 0.07;
    let p = 2e6;
    let tm = 1500.0;
    let c1_0 = 1.0;
    let t_initial = 1000.0;
    let pe_q = 0.0090168;
    let d_ro = 2.88e-4;
    let pe_d = 1.50e-3;
    let ro_m_ = m0 * p / (8.314 * tm);

    let dt_sym = Expr::Const(dt);
    let t_scale_sym = Expr::Const(t_scale);
    let lambda_sym = Expr::Const(lambda);
    let q_heat = Expr::Const(q_heat);
    let a = Expr::Const(1.3e5);
    let e = Expr::Const(5000.0 * 4.184);
    let m = Expr::Const(m0);
    let r_g = Expr::Const(8.314);
    let ro_m = Expr::Const(ro_m_);
    let qm = Expr::Const(l.powf(2.0) / t_scale);
    let qs = Expr::Const(l.powf(2.0));
    let pe_q_sym = Expr::Const(pe_q);
    let ro_d = vec![Expr::Const(d_ro), Expr::Const(d_ro)];
    let pe_d = vec![Expr::Const(pe_d), Expr::Const(pe_d)];
    let minus = Expr::Const(-1.0);
    let m_reag = Expr::Const(0.342);

    let rate = a
        * Expr::exp(-e / (r_g * (teta.clone() * t_scale_sym + dt_sym)))
        * c0.clone()
        * (ro_m.clone() / m_reag.clone());
    let eq_t = q.clone() / lambda_sym;
    let eq_q = q * pe_q_sym - q_heat * rate.clone() * qm;
    let eq_c0 = j0.clone() / ro_d[0].clone();
    let eq_j0 = j0 * pe_d[0].clone()
        - (m.clone() * minus * rate.clone() * ro_m.clone() / m.clone()) * qs.clone();
    let eq_c1 = j1.clone() / ro_d[1].clone();
    let eq_j1 = j1 * pe_d[1].clone() - (m.clone() * rate * ro_m / m) * qs;
    let eqs = vec![eq_t, eq_q, eq_c0, eq_j0, eq_c1, eq_j1];

    let boundary_conditions = HashMap::from([
        ("Teta".to_string(), vec![(0, (t_initial - dt) / t_scale)]),
        ("q".to_string(), vec![(1, 1e-10)]),
        ("C0".to_string(), vec![(0, c1_0)]),
        ("J0".to_string(), vec![(1, 1e-7)]),
        ("C1".to_string(), vec![(0, 1e-3)]),
        ("J1".to_string(), vec![(1, 1e-7)]),
    ]);

    let values = unknowns_str
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>();
    let _ = n_steps;
    (eqs, values, boundary_conditions)
}

pub(crate) fn build_combustion_bvp_case(n_steps: usize) -> Jacobian {
    build_combustion_bvp_case_with_backend(n_steps, BvpSymbolicAssemblyBackend::ExprLegacy)
}

pub(crate) fn build_combustion_bvp_case_with_backend(
    n_steps: usize,
    symbolic_backend: BvpSymbolicAssemblyBackend,
) -> Jacobian {
    let (eqs, values, boundary_conditions) = combustion_bvp_fixture(n_steps);
    let mut jac = Jacobian::new();
    jac.set_symbolic_assembly_backend(symbolic_backend);
    jac.discretization_system_BVP_par(
        eqs,
        values,
        "x".to_string(),
        0.0,
        Some(n_steps),
        None,
        None,
        boundary_conditions,
        None,
        None,
        "trapezoid".to_string(),
    );
    jac.calc_jacobian_parallel_smart_optimized();
    jac
}

pub(crate) fn command_exists(program: &str) -> bool {
    let resolved = match program.to_ascii_lowercase().as_str() {
        "tcc" => env::var("RUSTEDSCITHE_TCC")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .or_else(|| {
                env::var("RUSTEDSCITHE_C_COMPILER")
                    .ok()
                    .filter(|value| !value.trim().is_empty())
            })
            .unwrap_or_else(|| program.to_string()),
        "gcc" | "clang" | "cl" | "cc" => env::var(format!(
            "RUSTEDSCITHE_{}",
            program.to_ascii_uppercase()
        ))
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            env::var("RUSTEDSCITHE_C_COMPILER")
                .ok()
                .filter(|value| !value.trim().is_empty())
        })
        .unwrap_or_else(|| program.to_string()),
        _ => program.to_string(),
    };

    let candidate = std::path::PathBuf::from(&resolved);
    if candidate.components().count() > 1 && candidate.exists() {
        return true;
    }

    #[cfg(target_os = "windows")]
    let locator = "where";
    #[cfg(not(target_os = "windows"))]
    let locator = "which";

    if Command::new(locator)
        .arg(&resolved)
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
    {
        return true;
    }

    for probe_arg in ["--version", "-v", "version"] {
        if Command::new(&resolved).arg(probe_arg).output().is_ok() {
            return true;
        }
    }
    Command::new(&resolved).output().is_ok()
}
