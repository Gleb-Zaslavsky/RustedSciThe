//! Shared fixtures for ignored Radau story tests.
//!
//! The stories deliberately use the crate-level IVP workloads so correctness
//! and performance diagnostics remain comparable with LSODE2 and BDF.

use std::path::PathBuf;

use super::super::api::{
    RadauAotConfig, RadauConfig, RadauExecution, RadauExecutionPolicy, RadauFrontend,
    RadauMatrixLayout, RadauOutputPolicy, RadauProblem, RadauTelemetryMode,
};
use crate::numerical::ivp_workloads::{WorkloadKind, build_workload};

/// Build one shared workload as a public Radau problem and value buffers.
pub fn workload_problem(
    kind: WorkloadKind,
    dimension: usize,
    frontend: RadauFrontend,
) -> (RadauProblem, Vec<f64>, Vec<f64>) {
    let workload = build_workload(kind, dimension);
    let variables = workload.variables.clone();
    let variable_refs: Vec<&str> = variables.iter().map(String::as_str).collect();
    let jacobian = match frontend {
        RadauFrontend::ExprLegacy => Some(
            workload
                .equations
                .iter()
                .flat_map(|equation| {
                    variable_refs
                        .iter()
                        .map(move |variable| equation.diff(variable))
                })
                .collect(),
        ),
        RadauFrontend::AtomViewNative => None,
    };
    let mut problem = RadauProblem::new(workload.equations, variables, workload.time_variable)
        .with_parameters(workload.parameter_names);
    if let Some(jacobian) = jacobian {
        problem = problem.with_jacobian(jacobian);
    }
    (
        problem,
        workload.initial_state.as_slice().to_vec(),
        workload.parameter_values.as_slice().to_vec(),
    )
}

/// Keep story horizons short and deterministic while retaining workload shape.
pub fn workload_config(
    kind: WorkloadKind,
    frontend: RadauFrontend,
    execution: RadauExecution,
    layout: RadauMatrixLayout,
    policy: RadauExecutionPolicy,
    telemetry: RadauTelemetryMode,
) -> RadauConfig {
    let t_bound = match kind {
        WorkloadKind::CombustionLike | WorkloadKind::Robertson | WorkloadKind::ThreeBody => 0.002,
        WorkloadKind::DiffusionChain | WorkloadKind::StiffScalar => 0.01,
    };
    RadauConfig {
        t_bound,
        first_step: Some(t_bound * 0.1),
        max_step: t_bound * 0.25,
        rtol: 1.0e-7,
        atol: 1.0e-10,
        max_steps: 2_000,
        max_newton_iterations: 8,
        max_retries: 24,
        execution,
        frontend,
        matrix_layout: layout,
        telemetry,
        execution_policy: policy,
        output: RadauOutputPolicy::FinalOnly,
        ..RadauConfig::default()
    }
}

/// Use the same toolchain policy as the release AOT stories.
pub fn rebuild_aot_config(output_dir: impl Into<PathBuf>) -> RadauAotConfig {
    RadauAotConfig::rebuild_always_release(output_dir).with_c_compiler("tcc")
}

/// Reuse an already published artifact without invoking a compiler.
pub fn require_or_build_aot_config(output_dir: impl Into<PathBuf>) -> RadauAotConfig {
    RadauAotConfig::build_if_missing_release(output_dir).with_c_compiler("tcc")
}

pub fn max_diff(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max)
}

pub fn policy_label(policy: RadauExecutionPolicy) -> &'static str {
    match policy {
        RadauExecutionPolicy::Sequential => "sequential",
        RadauExecutionPolicy::Parallel { .. } => "parallel",
        RadauExecutionPolicy::Auto { .. } => "auto",
    }
}
