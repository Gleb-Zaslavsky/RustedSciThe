//! Public telemetry contract stories.
//!
//! These checks make report interpretation executable: counters distinguish
//! applicability from dispatch, and timing metadata prevents child scopes
//! from being added to their inclusive parent a second time.

use super::super::api::{
    RadauExecution, RadauExecutionPolicy, RadauFrontend, RadauMatrixLayout, RadauSolver,
    RadauTelemetryMode, RadauTelemetryScopeKind,
};
use super::story_support::{rebuild_aot_config, workload_config, workload_problem};
use crate::numerical::ivp_workloads::WorkloadKind;
use tabled::Tabled;

#[derive(Debug, Tabled)]
struct TelemetryRow {
    frontend: String,
    policy: String,
    residual_calls: u64,
    jacobian_calls: u64,
    residual_evaluations: u64,
    jacobian_evaluations: u64,
    jacobian_output_assemblies: u64,
    parallel_applicable: u64,
    parallel_dispatches: u64,
    sequential_dispatches: u64,
    worker_count: u64,
    configured_worker_count: u64,
    aot_chunking_applicable: u64,
    allocations: u64,
    copies: u64,
    status: &'static str,
}

#[derive(Debug, Tabled)]
struct LifecycleScopeRow {
    frontend: String,
    preparation_ms: String,
    materialize_ms: String,
    build_ms: String,
    link_ms: String,
    publication_ms: String,
    build_attempts: u64,
    link_attempts: u64,
    runtime_ready: u64,
    status: &'static str,
}

fn policy_label(policy: RadauExecutionPolicy) -> &'static str {
    match policy {
        RadauExecutionPolicy::Sequential => "sequential",
        RadauExecutionPolicy::Parallel { .. } => "parallel",
        RadauExecutionPolicy::Auto { .. } => "auto",
    }
}

#[test]
#[ignore = "diagnostic public telemetry scope and counter contract"]
fn lambdify_telemetry_scope_and_applicability_contract_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "Radau_Telemetry",
        "numerical::Radau::tests::telemetry_stories::lambdify_telemetry_scope_and_applicability_contract_story",
    );
    let mut rows = Vec::new();

    for frontend in [RadauFrontend::ExprLegacy, RadauFrontend::AtomViewNative] {
        for policy in [
            RadauExecutionPolicy::Sequential,
            RadauExecutionPolicy::Parallel { min_work: 1 },
            RadauExecutionPolicy::Auto { min_work: 1 },
        ] {
            let (problem, initial_state, parameters) =
                workload_problem(WorkloadKind::CombustionLike, 3, frontend);
            let config = workload_config(
                WorkloadKind::CombustionLike,
                frontend,
                RadauExecution::Lambdify,
                RadauMatrixLayout::Dense,
                policy,
                RadauTelemetryMode::Timings,
            );
            let mut solver = RadauSolver::prepare(problem, config)
                .unwrap_or_else(|error| panic!("telemetry prepare failed: {error}"));
            let solution = solver
                .solve_with_parameters(&initial_state, &parameters)
                .unwrap_or_else(|error| panic!("telemetry solve failed: {error}"));
            let report = solution.telemetry();
            let scopes = &report.timing_scopes;

            assert_eq!(
                scopes["preparation_ms"].kind,
                RadauTelemetryScopeKind::Inclusive
            );
            assert_eq!(
                scopes["expr_legacy_prepare_ms"].kind,
                RadauTelemetryScopeKind::Child
            );
            assert_eq!(
                scopes["expr_legacy_prepare_ms"].parent,
                Some("preparation_ms")
            );
            assert_eq!(
                scopes["callback_ms"].kind,
                RadauTelemetryScopeKind::Inclusive
            );
            assert_eq!(scopes["residual_evaluation_ms"].parent, Some("callback_ms"));
            assert_eq!(scopes["linear_ms"].kind, RadauTelemetryScopeKind::Inclusive);
            assert_eq!(scopes["factorization_ms"].parent, Some("linear_ms"));
            assert!(report.timings_ms.values().all(|value| value.is_finite()));

            let counters = &report.counters;
            assert!(counters["residual_calls"] > 0);
            assert!(counters["jacobian_calls"] > 0);
            assert!(counters["residual_evaluations"] > 0);
            assert!(counters["jacobian_evaluations"] > 0);
            assert!(counters["parallel_dispatch_applicable"] > 0);
            assert!(counters["worker_count"] >= 1);
            assert!(counters["parallel_dispatches"] + counters["sequential_dispatches"] > 0);
            assert_eq!(counters["aot_chunking_applicable"], 0);
            if matches!(policy, RadauExecutionPolicy::Sequential) {
                assert_eq!(counters["parallel_dispatches"], 0);
            }
            if matches!(policy, RadauExecutionPolicy::Parallel { .. }) {
                assert!(counters["parallel_dispatches"] > 0);
            }

            rows.push(TelemetryRow {
                frontend: format!("{frontend:?}"),
                policy: policy_label(policy).to_owned(),
                residual_calls: counters["residual_calls"],
                jacobian_calls: counters["jacobian_calls"],
                residual_evaluations: counters["residual_evaluations"],
                jacobian_evaluations: counters["jacobian_evaluations"],
                jacobian_output_assemblies: counters["jacobian_output_assemblies"],
                parallel_applicable: counters["parallel_dispatch_applicable"],
                parallel_dispatches: counters["parallel_dispatches"],
                sequential_dispatches: counters["sequential_dispatches"],
                worker_count: counters["worker_count"],
                configured_worker_count: counters["configured_worker_count"],
                aot_chunking_applicable: counters["aot_chunking_applicable"],
                allocations: counters["allocations"],
                copies: counters["copies"],
                status: "ok",
            });
        }
    }
    crate::Utils::test_reporting::capture_test_table(
        "[Radau telemetry scope/applicability contract]",
        &rows,
    );
}

#[test]
#[ignore = "diagnostic AOT lifecycle scope contract; requires tcc"]
fn aot_lifecycle_scope_relationships_are_consistent_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "Radau_Telemetry",
        "numerical::Radau::tests::telemetry_stories::aot_lifecycle_scope_relationships_are_consistent_story",
    );
    let mut rows = Vec::new();

    for frontend in [RadauFrontend::ExprLegacy, RadauFrontend::AtomViewNative] {
        let (problem, initial_state, parameters) =
            workload_problem(WorkloadKind::StiffScalar, 1, frontend);
        let artifact_dir = tempfile::tempdir().expect("AOT lifecycle artifact directory");
        let mut config = workload_config(
            WorkloadKind::StiffScalar,
            frontend,
            RadauExecution::Aot,
            RadauMatrixLayout::Dense,
            RadauExecutionPolicy::Sequential,
            RadauTelemetryMode::Timings,
        );
        config.aot = Some(rebuild_aot_config(artifact_dir.path()));
        let mut solver = RadauSolver::prepare(problem, config)
            .unwrap_or_else(|error| panic!("AOT lifecycle prepare failed: {error}"));
        let solution = solver
            .solve_with_parameters(&initial_state, &parameters)
            .unwrap_or_else(|error| panic!("AOT lifecycle solve failed: {error}"));
        let report = solution.telemetry();

        for child in [
            "aot_cache_lookup_ms",
            "aot_lowering_ms",
            "aot_source_generation_ms",
            "aot_materialize_ms",
            "aot_build_ms",
            "aot_link_ms",
            "aot_publication_ms",
        ] {
            let metadata = report.timing_scopes[child];
            assert_eq!(metadata.kind, RadauTelemetryScopeKind::Child);
            assert_eq!(metadata.parent, Some("preparation_ms"));
            let child_ms = report.timings_ms[child];
            let parent_ms = report.timings_ms["preparation_ms"];
            assert!(child_ms.is_finite() && child_ms >= 0.0);
            assert!(
                child_ms <= parent_ms + 1.0e-6,
                "lifecycle child {child}={child_ms} exceeds preparation={parent_ms}"
            );
        }
        assert!(report.timings_ms.values().all(|value| value.is_finite()));
        assert_eq!(report.counters["aot_build_attempts"], 1);
        assert_eq!(report.counters["aot_link_attempts"], 1);
        assert_eq!(report.counters["aot_build_failures"], 0);
        assert_eq!(report.counters["aot_link_failures"], 0);
        assert!(report.counters["aot_runtime_ready"] > 0);
        assert_eq!(report.aot_artifact_keys.len(), 1);

        rows.push(LifecycleScopeRow {
            frontend: format!("{frontend:?}"),
            preparation_ms: format!("{:.3}", report.timings_ms["preparation_ms"]),
            materialize_ms: format!("{:.3}", report.timings_ms["aot_materialize_ms"]),
            build_ms: format!("{:.3}", report.timings_ms["aot_build_ms"]),
            link_ms: format!("{:.3}", report.timings_ms["aot_link_ms"]),
            publication_ms: format!("{:.3}", report.timings_ms["aot_publication_ms"]),
            build_attempts: report.counters["aot_build_attempts"],
            link_attempts: report.counters["aot_link_attempts"],
            runtime_ready: report.counters["aot_runtime_ready"],
            status: "ok",
        });
    }

    crate::Utils::test_reporting::capture_test_table("[Radau AOT lifecycle scope contract]", &rows);
}
