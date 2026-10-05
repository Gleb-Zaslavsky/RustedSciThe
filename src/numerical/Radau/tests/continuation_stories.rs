//! AOT continuation and amortization stories.
//!
//! The stories compare a prepared value-only series with cached fresh
//! preparation on the same published artifact. Cold preparation is reported
//! separately so the result cannot mix `RebuildAlways` and `BuildIfMissing`
//! scopes into one misleading ratio.

use std::time::Instant;

use super::super::api::{
    RadauExecution, RadauExecutionPolicy, RadauFrontend, RadauMatrixLayout, RadauSolver,
    RadauTelemetryMode,
};
use super::story_support::{
    max_diff, rebuild_aot_config, require_or_build_aot_config, workload_config, workload_problem,
};
use crate::numerical::ivp_workloads::{WorkloadKind, parameter_continuation_target};
use tabled::Tabled;

#[derive(Debug, Tabled)]
struct ContinuationRow {
    workload: String,
    dimension: usize,
    frontend: String,
    layout: String,
    targets: usize,
    cold_prepare_ms: String,
    prepared_series_ms: String,
    prepared_total_ms: String,
    fresh_cached_series_ms: String,
    fresh_over_prepared_total: String,
    fresh_over_prepared_series: String,
    final_diff: String,
    build_attempts: u64,
    link_attempts: u64,
    artifact_keys: usize,
    status: &'static str,
}

fn targets(base: &[f64], count: usize) -> Vec<Vec<f64>> {
    (0..count)
        .map(|index| {
            parameter_continuation_target(&nalgebra::DVector::from_vec(base.to_vec()), index + 1)
                .as_slice()
                .to_vec()
        })
        .collect()
}

#[test]
#[ignore = "diagnostic AOT continuation amortization matrix; requires tcc"]
fn aot_continuation_break_even_matches_fresh_reprepare() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "Radau_Continuation",
        "numerical::Radau::tests::continuation_stories::aot_continuation_break_even_matches_fresh_reprepare",
    );

    let workload = WorkloadKind::DiffusionChain;
    let dimension = 8;
    let counts = [1usize, 4, 16];
    let layouts = [RadauMatrixLayout::Dense, RadauMatrixLayout::Sparse];
    let mut rows = Vec::new();

    for frontend in [RadauFrontend::ExprLegacy, RadauFrontend::AtomViewNative] {
        for layout in layouts {
            for count in counts {
                let artifact_dir =
                    tempfile::tempdir().expect("AOT continuation artifact directory");
                let (problem, initial_state, base_parameters) =
                    workload_problem(workload, dimension, frontend);
                let parameter_targets = targets(&base_parameters, count);

                let mut prepared_config = workload_config(
                    workload,
                    frontend,
                    RadauExecution::Aot,
                    layout,
                    RadauExecutionPolicy::Sequential,
                    RadauTelemetryMode::Timings,
                );
                prepared_config.aot = Some(rebuild_aot_config(artifact_dir.path()));
                let prepare_start = Instant::now();
                let mut prepared = RadauSolver::prepare(problem.clone(), prepared_config)
                    .unwrap_or_else(|error| panic!("AOT continuation prepare failed: {error}"));
                let cold_prepare_ms = prepare_start.elapsed().as_secs_f64() * 1.0e3;

                let prepared_series_start = Instant::now();
                let mut continued_final = None;
                for parameters in &parameter_targets {
                    continued_final = Some(
                        prepared
                            .continue_with_parameters(&initial_state, parameters)
                            .unwrap_or_else(|error| {
                                panic!(
                                    "AOT continuation failed for {frontend:?}/{layout:?}: {error}"
                                )
                            }),
                    );
                }
                let prepared_series_ms = prepared_series_start.elapsed().as_secs_f64() * 1.0e3;
                let prepared_total_ms = cold_prepare_ms + prepared_series_ms;
                let continued = continued_final.expect("at least one continuation target");

                let fresh_start = Instant::now();
                let mut fresh_final = None;
                for parameters in &parameter_targets {
                    let (fresh_problem, _, _) = workload_problem(workload, dimension, frontend);
                    let mut fresh_config = workload_config(
                        workload,
                        frontend,
                        RadauExecution::Aot,
                        layout,
                        RadauExecutionPolicy::Sequential,
                        RadauTelemetryMode::Counters,
                    );
                    fresh_config.aot = Some(require_or_build_aot_config(artifact_dir.path()));
                    let mut fresh = RadauSolver::prepare(fresh_problem, fresh_config)
                        .unwrap_or_else(|error| panic!("AOT fresh prepare failed: {error}"));
                    fresh_final = Some(
                        fresh
                            .solve_with_parameters(&initial_state, parameters)
                            .unwrap_or_else(|error| panic!("AOT fresh solve failed: {error}")),
                    );
                }
                let fresh_cached_series_ms = fresh_start.elapsed().as_secs_f64() * 1.0e3;
                let fresh = fresh_final.expect("at least one fresh target");
                let drift = max_diff(&continued.y, &fresh.y);
                let counters = &continued.telemetry().counters;
                assert!(continued.y.iter().all(|value| value.is_finite()));
                assert!(fresh.y.iter().all(|value| value.is_finite()));
                assert!(drift < 2.0e-5, "continuation drift={drift:.3e}");
                assert_eq!(counters["aot_build_attempts"], 1);
                assert_eq!(counters["aot_link_attempts"], 1);

                rows.push(ContinuationRow {
                    workload: workload.label().to_owned(),
                    dimension,
                    frontend: format!("{frontend:?}"),
                    layout: format!("{layout:?}"),
                    targets: count,
                    cold_prepare_ms: format!("{cold_prepare_ms:.3}"),
                    prepared_series_ms: format!("{prepared_series_ms:.3}"),
                    prepared_total_ms: format!("{prepared_total_ms:.3}"),
                    fresh_cached_series_ms: format!("{fresh_cached_series_ms:.3}"),
                    fresh_over_prepared_total: format!(
                        "{:.3}",
                        fresh_cached_series_ms / prepared_total_ms.max(f64::EPSILON)
                    ),
                    fresh_over_prepared_series: format!(
                        "{:.3}",
                        fresh_cached_series_ms / prepared_series_ms.max(f64::EPSILON)
                    ),
                    final_diff: format!("{drift:.3e}"),
                    build_attempts: counters["aot_build_attempts"],
                    link_attempts: counters["aot_link_attempts"],
                    artifact_keys: continued.telemetry().aot_artifact_keys.len(),
                    status: "ok",
                });
            }
        }
    }
    crate::Utils::test_reporting::capture_test_table(
        "[Radau AOT continuation amortization]",
        &rows,
    );
}

#[test]
#[ignore = "diagnostic AOT continuation retention story; requires tcc"]
fn aot_continuation_long_series_keeps_artifact_and_build_counts_stable() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "Radau_Continuation",
        "numerical::Radau::tests::continuation_stories::aot_continuation_long_series_keeps_artifact_and_build_counts_stable",
    );
    let workload = WorkloadKind::CombustionLike;
    let dimension = 3;
    let mut rows = Vec::new();

    for frontend in [RadauFrontend::ExprLegacy, RadauFrontend::AtomViewNative] {
        let artifact_dir = tempfile::tempdir().expect("AOT retention artifact directory");
        let (problem, initial_state, base_parameters) =
            workload_problem(workload, dimension, frontend);
        let mut config = workload_config(
            workload,
            frontend,
            RadauExecution::Aot,
            RadauMatrixLayout::Dense,
            RadauExecutionPolicy::Sequential,
            RadauTelemetryMode::Counters,
        );
        config.aot = Some(rebuild_aot_config(artifact_dir.path()));
        let mut solver = RadauSolver::prepare(problem, config)
            .unwrap_or_else(|error| panic!("AOT retention prepare failed: {error}"));
        let mut final_solution = None;
        for index in 0..64 {
            let parameters = parameter_continuation_target(
                &nalgebra::DVector::from_vec(base_parameters.clone()),
                index + 1,
            );
            final_solution = Some(
                solver
                    .continue_with_parameters(&initial_state, parameters.as_slice())
                    .unwrap_or_else(|error| panic!("AOT retention continuation failed: {error}")),
            );
        }
        let solution = final_solution.expect("retention series must solve");
        let counters = &solution.telemetry().counters;
        assert!(solution.y.iter().all(|value| value.is_finite()));
        assert_eq!(counters["aot_build_attempts"], 1);
        assert_eq!(counters["aot_link_attempts"], 1);
        assert_eq!(counters["aot_build_failures"], 0);
        assert_eq!(counters["aot_link_failures"], 0);
        assert!(!solution.telemetry().aot_artifact_keys.is_empty());
        rows.push(ContinuationRow {
            workload: workload.label().to_owned(),
            dimension,
            frontend: format!("{frontend:?}"),
            layout: "Dense".to_owned(),
            targets: 64,
            cold_prepare_ms: "n/a".to_owned(),
            prepared_series_ms: "n/a".to_owned(),
            prepared_total_ms: "n/a".to_owned(),
            fresh_cached_series_ms: "n/a".to_owned(),
            fresh_over_prepared_total: "n/a".to_owned(),
            fresh_over_prepared_series: "n/a".to_owned(),
            final_diff: "n/a".to_owned(),
            build_attempts: counters["aot_build_attempts"],
            link_attempts: counters["aot_link_attempts"],
            artifact_keys: solution.telemetry().aot_artifact_keys.len(),
            status: "ok",
        });
    }
    crate::Utils::test_reporting::capture_test_table("[Radau AOT continuation retention]", &rows);
}
