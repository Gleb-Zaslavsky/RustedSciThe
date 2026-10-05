//! Execution-policy stories.
//!
//! These tests are diagnostic rather than threshold gates. They establish
//! parity and show callback/full-solve attribution for each policy without
//! turning one noisy wall-clock sample into a portability claim.

use std::time::Instant;

use super::super::api::{
    RadauExecution, RadauExecutionPolicy, RadauFrontend, RadauMatrixLayout, RadauSolver,
    RadauTelemetryMode,
};
use super::story_support::{max_diff, policy_label, workload_config, workload_problem};
use crate::numerical::ivp_workloads::WorkloadKind;
use tabled::Tabled;

#[derive(Debug, Tabled)]
struct PolicyRow {
    execution: String,
    workload: String,
    dimension: usize,
    frontend: String,
    policy: String,
    repetitions: usize,
    prepare_ms: String,
    callback_ms: String,
    full_solve_ms: String,
    linear_ms: String,
    parallel_dispatches: u64,
    sequential_dispatches: u64,
    max_diff: String,
    status: &'static str,
}

fn timing(solution: &super::super::api::RadauSolution, key: &'static str) -> f64 {
    solution
        .telemetry()
        .timings_ms
        .get(key)
        .copied()
        .unwrap_or_default()
}

fn policies() -> [RadauExecutionPolicy; 3] {
    [
        RadauExecutionPolicy::Sequential,
        RadauExecutionPolicy::Parallel { min_work: 1 },
        RadauExecutionPolicy::Auto { min_work: 1 },
    ]
}

#[test]
#[ignore = "diagnostic Lambdify execution-policy/full-solve matrix"]
fn lambdify_policy_full_solve_and_callback_break_even_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "Radau_Policy",
        "numerical::Radau::tests::policy_stories::lambdify_policy_full_solve_and_callback_break_even_story",
    );

    let repetitions = 3;
    let mut rows = Vec::new();
    for (kind, dimension) in [
        (WorkloadKind::CombustionLike, 3),
        (WorkloadKind::DiffusionChain, 16),
    ] {
        for frontend in [RadauFrontend::ExprLegacy, RadauFrontend::AtomViewNative] {
            let mut reference = None;
            for policy in policies() {
                let (problem, initial_state, parameters) =
                    workload_problem(kind, dimension, frontend);
                let config = workload_config(
                    kind,
                    frontend,
                    RadauExecution::Lambdify,
                    RadauMatrixLayout::Dense,
                    policy,
                    RadauTelemetryMode::Timings,
                );
                let mut solver = RadauSolver::prepare(problem, config)
                    .unwrap_or_else(|error| panic!("Lambdify prepare failed: {error}"));
                let start = Instant::now();
                let mut last = None;
                for _ in 0..repetitions {
                    last = Some(
                        solver
                            .solve_with_parameters(&initial_state, &parameters)
                            .unwrap_or_else(|error| panic!("Lambdify solve failed: {error}")),
                    );
                }
                let elapsed_ms = start.elapsed().as_secs_f64() * 1.0e3 / repetitions as f64;
                let solution = last.expect("at least one policy repetition");
                let drift = reference
                    .as_ref()
                    .map(|expected: &Vec<f64>| max_diff(expected, &solution.y))
                    .unwrap_or(0.0);
                if reference.is_none() {
                    reference = Some(solution.y.clone());
                }
                assert!(
                    drift < 2.0e-5,
                    "{kind:?}/{frontend:?} policy drift={drift:.3e}"
                );

                let counters = &solution.telemetry().counters;
                assert!(
                    counters["parallel_dispatches"] + counters["sequential_dispatches"] > 0,
                    "policy produced no callback dispatches"
                );
                rows.push(PolicyRow {
                    execution: "lambdify".to_owned(),
                    workload: kind.label().to_owned(),
                    dimension,
                    frontend: format!("{frontend:?}"),
                    policy: policy_label(policy).to_owned(),
                    repetitions,
                    prepare_ms: format!("{:.3}", timing(&solution, "preparation_ms")),
                    callback_ms: format!("{:.3}", timing(&solution, "callback_ms")),
                    full_solve_ms: format!("{elapsed_ms:.3}"),
                    linear_ms: format!("{:.3}", timing(&solution, "linear_ms")),
                    parallel_dispatches: counters["parallel_dispatches"],
                    sequential_dispatches: counters["sequential_dispatches"],
                    max_diff: format!("{drift:.3e}"),
                    status: "ok",
                });
            }
        }
    }
    crate::Utils::test_reporting::capture_test_table(
        "[Radau Lambdify policy/full-solve matrix]",
        &rows,
    );
}

#[test]
#[ignore = "diagnostic AOT execution-policy/full-solve matrix; requires tcc"]
fn aot_policy_full_solve_and_callback_break_even_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "Radau_Policy",
        "numerical::Radau::tests::policy_stories::aot_policy_full_solve_and_callback_break_even_story",
    );

    let workload = WorkloadKind::DiffusionChain;
    let dimension = 8;
    let mut rows = Vec::new();
    for frontend in [RadauFrontend::ExprLegacy, RadauFrontend::AtomViewNative] {
        for policy in policies() {
            let output_dir = tempfile::tempdir().expect("AOT policy artifact directory");
            let (problem, initial_state, parameters) =
                workload_problem(workload, dimension, frontend);
            let mut config = workload_config(
                workload,
                frontend,
                RadauExecution::Aot,
                RadauMatrixLayout::Dense,
                policy,
                RadauTelemetryMode::Timings,
            );
            config.aot = Some(super::story_support::rebuild_aot_config(output_dir.path()));
            let mut solver = RadauSolver::prepare(problem, config)
                .unwrap_or_else(|error| panic!("AOT policy prepare failed: {error}"));
            let start = Instant::now();
            let solution = solver
                .solve_with_parameters(&initial_state, &parameters)
                .unwrap_or_else(|error| panic!("AOT policy solve failed: {error}"));
            let elapsed_ms = start.elapsed().as_secs_f64() * 1.0e3;
            let counters = &solution.telemetry().counters;
            assert!(solution.y.iter().all(|value| value.is_finite()));
            assert_eq!(counters["aot_build_successes"], 1);
            assert_eq!(counters["aot_link_successes"], 1);
            rows.push(PolicyRow {
                execution: "aot".to_owned(),
                workload: workload.label().to_owned(),
                dimension,
                frontend: format!("{frontend:?}"),
                policy: policy_label(policy).to_owned(),
                repetitions: 1,
                prepare_ms: format!("{:.3}", timing(&solution, "preparation_ms")),
                callback_ms: format!("{:.3}", timing(&solution, "callback_ms")),
                full_solve_ms: format!("{elapsed_ms:.3}"),
                linear_ms: format!("{:.3}", timing(&solution, "linear_ms")),
                parallel_dispatches: counters["parallel_dispatches"],
                sequential_dispatches: counters["sequential_dispatches"],
                max_diff: "0.000e0".to_owned(),
                status: "ok",
            });
        }
    }
    crate::Utils::test_reporting::capture_test_table("[Radau AOT policy/full-solve matrix]", &rows);
}
