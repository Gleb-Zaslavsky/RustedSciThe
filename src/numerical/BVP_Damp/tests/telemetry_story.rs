//! Release-oriented telemetry cost and scope story.
//!
//! ```text
//! cargo test --release --lib --no-default-features telemetry_off_vs_detailed_adaptive_story -- --ignored --nocapture --test-threads=1
//! ```
//!
//! The story deliberately uses a small initial guess for a nonlinear Bratu
//! problem and uniform `DoublePoints` refinement. This makes mesh revision,
//! factor invalidation, nonlinear iterations and damping trials observable in
//! one report without involving AOT build time. The price comparison alternates
//! route order and reports medians, so a first-run cold-start cannot masquerade
//! as telemetry overhead.

#[cfg(test)]
mod tests {
    use crate::Utils::test_reporting::write_test_report;
    use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{
        AdaptiveGridConfig, DampedSolverOptions, NRBVP, SolverParams,
    };
    use crate::numerical::BVP_Damp::grid_api::GridRefinementMethod;
    use crate::numerical::BVP_Damp::telemetry::{
        BvpDampedDecisionEvent, BvpLoggingConfig, BvpLoggingMode, BvpTelemetryMode,
    };
    use nalgebra::{DMatrix, DVector};
    use std::collections::HashMap;
    use std::time::Instant;

    type DampedStats = crate::numerical::BVP_Damp::NR_Damp_solver_damped::DampedBvpStatistics;

    struct TimedRun {
        wall_ms: f64,
        stats: DampedStats,
        result: DMatrix<f64>,
        decision_trace: crate::numerical::BVP_Damp::telemetry::BvpDampedDecisionTrace,
    }

    fn bratu_solver(telemetry_mode: BvpTelemetryMode, logging_mode: BvpLoggingMode) -> NRBVP {
        let n_steps = 12usize;
        let values = vec!["y".to_string(), "z".to_string()];
        let initial_guess = DMatrix::from_element(values.len(), n_steps, 0.0);
        let boundary_conditions =
            HashMap::from([("y".to_string(), vec![(0usize, 0.0f64), (1usize, 0.0f64)])]);
        let bounds = HashMap::from([
            ("y".to_string(), (-20.0, 20.0)),
            ("z".to_string(), (-200.0, 200.0)),
        ]);
        let rel_tolerance = HashMap::from([("y".to_string(), 1e-6), ("z".to_string(), 1e-6)]);
        let strategy = SolverParams {
            max_jac: Some(2),
            max_damp_iter: Some(8),
            damp_factor: Some(0.5),
            adaptive: Some(AdaptiveGridConfig {
                version: 1,
                max_refinements: 1,
                grid_method: GridRefinementMethod::DoublePoints,
            }),
        };
        let options = DampedSolverOptions::sparse_damped()
            .with_strategy_params(Some(strategy))
            .with_abs_tolerance(1e-7)
            .with_max_iterations(40)
            .with_bounds(bounds)
            .with_rel_tolerance(rel_tolerance)
            .with_bvp_telemetry_mode(telemetry_mode)
            .with_bvp_logging_config(BvpLoggingConfig::new(logging_mode).with_max_events(512));

        // y' = z, z' = -lambda*exp(y), lambda=2. The zero initial guess is
        // intentionally poor but remains inside the configured safety bounds.
        NRBVP::new_numeric_with_jacobian_options(
            initial_guess,
            values,
            "x".to_string(),
            boundary_conditions,
            0.0,
            1.0,
            n_steps,
            options,
            |_x, state: &DVector<f64>, _params| {
                DVector::from_vec(vec![state[1], -2.0 * state[0].exp()])
            },
            |_x, state: &DVector<f64>, _params| {
                DMatrix::from_row_slice(2, 2, &[0.0, 1.0, -2.0 * state[0].exp(), 0.0])
            },
        )
    }

    fn small_linear_bvp_solver() -> NRBVP {
        let n_steps = 8;
        let values = vec!["y".to_string(), "z".to_string()];
        let initial_guess = DMatrix::zeros(values.len(), n_steps);
        let boundary_conditions =
            HashMap::from([("y".to_string(), vec![(0usize, 0.0), (1usize, 1.0)])]);
        let bounds = HashMap::from([
            ("y".to_string(), (-2.0, 2.0)),
            ("z".to_string(), (-2.0, 2.0)),
        ]);
        let rel_tolerance = HashMap::from([("y".to_string(), 1e-8), ("z".to_string(), 1e-8)]);
        let options = DampedSolverOptions::sparse_damped()
            .with_strategy_params(Some(SolverParams::default()))
            .with_abs_tolerance(1e-9)
            .with_max_iterations(20)
            .with_bounds(bounds)
            .with_rel_tolerance(rel_tolerance)
            .with_bvp_telemetry_mode(BvpTelemetryMode::Off)
            .with_bvp_logging_config(
                BvpLoggingConfig::new(BvpLoggingMode::Detailed).with_max_events(64),
            );

        // y' = z, z' = 0 with y(0)=0 and y(1)=1 has the exact solution y=x, z=1.
        NRBVP::new_numeric_with_jacobian_options(
            initial_guess,
            values,
            "x".to_string(),
            boundary_conditions,
            0.0,
            1.0,
            n_steps,
            options,
            |_x, state: &DVector<f64>, _params| DVector::from_vec(vec![state[1], 0.0]),
            |_x, _state: &DVector<f64>, _params| {
                DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0])
            },
        )
    }

    fn run_story_case(telemetry_mode: BvpTelemetryMode, logging_mode: BvpLoggingMode) -> TimedRun {
        let mut solver = bratu_solver(telemetry_mode, logging_mode);
        let started = Instant::now();
        solver
            .try_solve()
            .expect("telemetry story Bratu problem should solve");
        let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let result = solver
            .get_result()
            .expect("telemetry story should publish a result");
        assert!(result.iter().all(|value| value.is_finite()));
        TimedRun {
            wall_ms: elapsed_ms,
            stats: solver.get_statistics(),
            result,
            decision_trace: solver.damped_decision_trace_snapshot(),
        }
    }

    #[test]
    fn damped_decision_trace_is_opt_in_and_does_not_change_solver_work() {
        let off = run_story_case(BvpTelemetryMode::Off, BvpLoggingMode::Off);
        let logging_only = run_story_case(BvpTelemetryMode::Off, BvpLoggingMode::Detailed);
        let counters_only = run_story_case(BvpTelemetryMode::Counters, BvpLoggingMode::Off);
        let detailed_trace = run_story_case(BvpTelemetryMode::Counters, BvpLoggingMode::Detailed);

        assert_eq!(off.result, detailed_trace.result);
        assert_eq!(off.result, logging_only.result);
        assert!(off.decision_trace.events.is_empty());
        assert!(counters_only.decision_trace.events.is_empty());
        assert!(!logging_only.decision_trace.events.is_empty());
        assert_eq!(logging_only.stats.telemetry.counters, Default::default());
        assert!(
            !logging_only
                .stats
                .telemetry
                .log_events
                .iter()
                .any(|event| event.kind
                    == crate::numerical::BVP_Damp::telemetry::BvpLogEventKind::DampingTrial)
        );
        assert!(!detailed_trace.decision_trace.events.is_empty());
        assert_eq!(detailed_trace.decision_trace.dropped_events, 0);

        let off_counters = &counters_only.stats.telemetry.counters;
        let detailed_counters = &detailed_trace.stats.telemetry.counters;
        assert_eq!(off_counters, detailed_counters);
        assert_eq!(
            off_counters.residual_calls,
            detailed_counters.residual_calls
        );
        assert_eq!(
            off_counters.jacobian_recalculations,
            detailed_counters.jacobian_recalculations
        );
        assert_eq!(
            off_counters.damping_trials,
            detailed_counters.damping_trials
        );
        assert_eq!(
            off_counters.damping_rejections,
            detailed_counters.damping_rejections
        );
        assert_eq!(off_counters.linear_solves, detailed_counters.linear_solves);
        assert_eq!(
            off_counters.factorizations,
            detailed_counters.factorizations
        );

        let trials = detailed_trace
            .decision_trace
            .events
            .iter()
            .filter_map(|event| match event {
                BvpDampedDecisionEvent::DampingTrialDecision { .. } => Some(event),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(trials.len() as u64, detailed_counters.damping_trials);
        assert!(trials.iter().any(|event| matches!(
            event,
            BvpDampedDecisionEvent::DampingTrialDecision {
                accepted: true,
                accepted_by_threshold: Some(_),
                accepted_by_decrease: Some(_),
                comparison_norm,
                convergence_threshold,
                ..
            } if comparison_norm.is_finite() && *convergence_threshold == 1.0
        )));
        for event in &trials {
            if let BvpDampedDecisionEvent::DampingTrialDecision {
                attempted_alpha,
                next_alpha,
                current_norm: Some(s1),
                comparison_norm: s0,
                convergence_threshold,
                accepted_by_threshold: Some(by_threshold),
                accepted_by_decrease: Some(by_decrease),
                accepted,
                ..
            } = event
            {
                assert_eq!(*by_threshold, *s1 < *convergence_threshold);
                assert_eq!(*by_decrease, *s1 < *s0);
                assert_eq!(*accepted, *by_threshold || *by_decrease);
                if *accepted {
                    assert!(next_alpha.is_none());
                } else {
                    let next_alpha = next_alpha.expect("rejected trial records next alpha");
                    assert!(next_alpha < *attempted_alpha);
                }
            }
        }

        let legacy_trial_alphas = detailed_trace
            .stats
            .telemetry
            .log_events
            .iter()
            .filter(|event| {
                event.kind == crate::numerical::BVP_Damp::telemetry::BvpLogEventKind::DampingTrial
            })
            .map(|event| (event.value_a, event.value_b))
            .collect::<Vec<_>>();
        assert_eq!(legacy_trial_alphas.len(), trials.len());
        for (typed, (legacy_alpha, legacy_value_b)) in trials.iter().zip(legacy_trial_alphas) {
            let BvpDampedDecisionEvent::DampingTrialDecision {
                attempted_alpha, ..
            } = typed
            else {
                unreachable!()
            };
            assert_eq!(*attempted_alpha, legacy_alpha);
            assert!(legacy_value_b.is_nan());
        }
        let legacy_rejected = detailed_trace
            .stats
            .telemetry
            .log_events
            .iter()
            .filter(|event| {
                event.kind
                    == crate::numerical::BVP_Damp::telemetry::BvpLogEventKind::DampingRejected
            })
            .collect::<Vec<_>>();
        let typed_rejected = trials
            .iter()
            .filter_map(|event| match event {
                BvpDampedDecisionEvent::DampingTrialDecision {
                    accepted: false,
                    next_alpha: Some(next_alpha),
                    current_norm: Some(current_norm),
                    ..
                } => Some((*next_alpha, *current_norm)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(legacy_rejected.len(), typed_rejected.len());
        for (legacy, (next_alpha, rejected_norm)) in legacy_rejected.iter().zip(typed_rejected) {
            assert_eq!(legacy.value_a, next_alpha);
            assert_eq!(legacy.value_b, rejected_norm);
        }

        let legacy_trial = detailed_trace
            .stats
            .telemetry
            .log_events
            .iter()
            .find(|event| {
                event.kind == crate::numerical::BVP_Damp::telemetry::BvpLogEventKind::DampingTrial
            })
            .expect("legacy decision event remains available");
        assert!(legacy_trial.value_a.is_finite());
        assert!(legacy_trial.value_b.is_nan());
        assert!(
            detailed_trace
                .decision_trace
                .events
                .iter()
                .any(|event| matches!(
                    event,
                    BvpDampedDecisionEvent::IterationDecision {
                        status: crate::numerical::BVP_Damp::telemetry::BvpDampedIterationStatus::Converged,
                        accepted_alpha: Some(_),
                        accepted_correction_norm: Some(_),
                        stopping_threshold: Some(_),
                        ..
                    }
                ))
        );
        assert!(detailed_trace.decision_trace.events.iter().any(|event| matches!(
            event,
            BvpDampedDecisionEvent::SolveTermination {
                reason: crate::numerical::BVP_Damp::telemetry::BvpDampedSolveTerminationReason::Converged,
                status: crate::numerical::BVP_Damp::telemetry::BvpSolveStatus::Converged,
                iteration_index: Some(_),
                stopping_metric: Some(_),
                stopping_threshold: Some(_),
                residual_norm: Some(_),
                ..
            }
        )));
    }

    #[test]
    fn small_linear_reference_bvp_emits_complete_decision_and_termination() {
        let mut solver = small_linear_bvp_solver();
        let result = solver
            .try_solve()
            .expect("small linear reference BVP should solve")
            .expect("small linear reference BVP should converge");
        assert!(result.iter().all(|value| value.is_finite()));

        let profile = solver
            .get_result()
            .expect("converged linear reference profile should be published");
        assert_eq!(profile.ncols(), 2);
        assert!((profile[(0, 0)] - 0.0).abs() <= 1e-8);
        assert!((profile[(profile.nrows() - 1, 0)] - 1.0).abs() <= 1e-8);
        assert!((profile[(0, 1)] - 1.0).abs() <= 1e-8);

        let trace = solver.damped_decision_trace_snapshot();
        println!("[BVP Damp typed trace example] {:#?}", trace.events);
        let iteration = trace.events.iter().find_map(|event| match event {
            BvpDampedDecisionEvent::IterationDecision {
                correction_metric: Some(metric),
                base_correction_norm: Some(base_norm),
                base_residual_norm: Some(base_residual),
                bound_limited_alpha: Some(alpha),
                jacobian_age,
                jacobian_refreshed,
                accepted_alpha: Some(accepted_alpha),
                accepted_correction_norm: Some(accepted_norm),
                residual_norm: Some(residual_norm),
                stopping_threshold: Some(threshold),
                status: crate::numerical::BVP_Damp::telemetry::BvpDampedIterationStatus::Converged,
                ..
            } => Some((
                *metric,
                *base_norm,
                *base_residual,
                *alpha,
                *jacobian_age,
                *jacobian_refreshed,
                *accepted_alpha,
                *accepted_norm,
                *residual_norm,
                *threshold,
            )),
            _ => None,
        });
        let (
            metric,
            base_norm,
            base_residual,
            alpha,
            jacobian_age,
            jacobian_refreshed,
            accepted_alpha,
            accepted_norm,
            residual_norm,
            threshold,
        ) = iteration.expect("trace records completed Newton iteration");
        assert_eq!(
            metric,
            crate::numerical::BVP_Damp::telemetry::BvpDampedCorrectionMetric::LegacyEuclidean
        );
        assert!(base_norm.is_finite() && base_residual.is_finite());
        assert!(alpha.is_finite() && accepted_alpha.is_finite());
        assert!(jacobian_age > 0);
        assert!(jacobian_refreshed);
        assert!(accepted_norm <= threshold);
        assert!(residual_norm.is_finite());

        assert!(trace.events.iter().any(|event| matches!(
            event,
            BvpDampedDecisionEvent::SolveTermination {
                reason: crate::numerical::BVP_Damp::telemetry::BvpDampedSolveTerminationReason::Converged,
                status: crate::numerical::BVP_Damp::telemetry::BvpSolveStatus::Converged,
                iteration_index: Some(_),
                stopping_metric: Some(_),
                stopping_threshold: Some(_),
                residual_norm: Some(_),
                ..
            }
        )));
    }

    #[test]
    fn bounded_newton_budget_reports_no_result_termination() {
        let mut solver = bratu_solver(BvpTelemetryMode::Off, BvpLoggingMode::Detailed);
        solver.max_iterations = 2;
        if let Some(strategy) = solver.strategy_params.as_mut() {
            strategy.max_damp_iter = Some(1);
            strategy.adaptive = None;
        }
        let result = solver
            .try_solve()
            .expect("bounded Newton budget should return a classified solve outcome");
        assert!(result.is_none(), "tiny Newton budget should not converge");

        let trace = solver.damped_decision_trace_snapshot();
        let termination = trace.events.iter().find_map(|event| match event {
            BvpDampedDecisionEvent::SolveTermination {
                reason,
                status,
                iteration_index,
                stopping_metric,
                stopping_threshold,
                ..
            } => Some((
                *reason,
                *status,
                *iteration_index,
                *stopping_metric,
                *stopping_threshold,
            )),
            _ => None,
        });
        let (reason, status, iteration, metric, threshold) =
            termination.expect("no-result solve should retain its typed termination");
        assert_eq!(
            status,
            crate::numerical::BVP_Damp::telemetry::BvpSolveStatus::NoResult
        );
        assert!(matches!(
            reason,
            crate::numerical::BVP_Damp::telemetry::BvpDampedSolveTerminationReason::IterationLimit
                | crate::numerical::BVP_Damp::telemetry::BvpDampedSolveTerminationReason::DampingTrialsExhausted
        ));
        assert!(iteration.is_some());
        assert!(metric.is_some());
        assert!(threshold.is_some());
    }

    fn median(values: &[f64]) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).expect("finite timing"));
        sorted[sorted.len() / 2]
    }

    fn min_max(values: &[f64]) -> (f64, f64) {
        values
            .iter()
            .copied()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), value| {
                (min.min(value), max.max(value))
            })
    }

    #[test]
    #[ignore = "release-only telemetry price and adaptive-scope story"]
    fn telemetry_off_vs_detailed_adaptive_story() {
        const SAMPLES: usize = 5;
        let mut off_runs = Vec::with_capacity(SAMPLES);
        let mut telemetry_runs = Vec::with_capacity(SAMPLES);

        // Alternate the order so process warm-up and global cache effects are
        // not systematically charged to one telemetry mode.
        for sample in 0..SAMPLES {
            if sample % 2 == 0 {
                off_runs.push(run_story_case(BvpTelemetryMode::Off, BvpLoggingMode::Off));
                telemetry_runs.push(run_story_case(
                    BvpTelemetryMode::Detailed,
                    BvpLoggingMode::Off,
                ));
            } else {
                telemetry_runs.push(run_story_case(
                    BvpTelemetryMode::Detailed,
                    BvpLoggingMode::Off,
                ));
                off_runs.push(run_story_case(BvpTelemetryMode::Off, BvpLoggingMode::Off));
            }
        }

        // This third route is deliberately outside the price samples: it
        // validates the full diagnostic report, including structured events.
        let detailed_logging_run =
            run_story_case(BvpTelemetryMode::Detailed, BvpLoggingMode::Detailed);

        let off = &off_runs[0].stats;
        let telemetry = &telemetry_runs[0].stats;
        let detailed_logging = &detailed_logging_run.stats;

        let off_snapshot = &off.telemetry;
        let telemetry_snapshot = &telemetry.telemetry;
        let detailed_snapshot = &detailed_logging.telemetry;
        assert_eq!(off_snapshot.telemetry_mode, BvpTelemetryMode::Off);
        assert_eq!(off_snapshot.counters, Default::default());
        assert_eq!(off_snapshot.log_events.len(), 0);

        assert_eq!(
            telemetry_snapshot.telemetry_mode,
            BvpTelemetryMode::Detailed
        );
        assert!(telemetry_snapshot.counters.iterations > 0);
        assert!(telemetry_snapshot.counters.grid_refinements > 0);
        assert!(telemetry_snapshot.counters.factorization_invalidations > 0);
        assert!(telemetry_snapshot.scopes.iteration.elapsed > std::time::Duration::ZERO);
        assert!(telemetry_snapshot.scopes.damping_trial.elapsed > std::time::Duration::ZERO);
        assert_eq!(telemetry_snapshot.log_events.len(), 0);

        assert_eq!(detailed_snapshot.telemetry_mode, BvpTelemetryMode::Detailed);
        assert!(detailed_snapshot.counters.iterations > 0);
        assert!(detailed_snapshot.counters.grid_refinements > 0);
        assert!(detailed_snapshot.counters.factorization_invalidations > 0);
        assert!(detailed_snapshot.scopes.iteration.elapsed > std::time::Duration::ZERO);
        assert!(detailed_snapshot.scopes.damping_trial.elapsed > std::time::Duration::ZERO);
        assert!(
            detailed_snapshot.log_events.iter().any(
                |event| event.kind == crate::numerical::BVP_Damp::BvpLogEventKind::MeshRevision
            )
        );
        assert!(detailed_snapshot.log_events.iter().any(|event| {
            event.kind == crate::numerical::BVP_Damp::BvpLogEventKind::FactorizationInvalidated
        }));

        let off_times: Vec<_> = off_runs.iter().map(|run| run.wall_ms).collect();
        let telemetry_times: Vec<_> = telemetry_runs.iter().map(|run| run.wall_ms).collect();
        let paired_deltas: Vec<_> = telemetry_times
            .iter()
            .zip(off_times.iter())
            .map(|(telemetry, off)| telemetry - off)
            .collect();
        let (off_min, off_max) = min_max(&off_times);
        let (telemetry_min, telemetry_max) = min_max(&telemetry_times);
        let (delta_min, delta_max) = min_max(&paired_deltas);
        let off_median = median(&off_times);
        let telemetry_median = median(&telemetry_times);
        let delta_median = median(&paired_deltas);
        println!(
            "[BVP Damp telemetry price] samples={SAMPLES}; route | median_ms | min_ms | max_ms"
        );
        println!(
            "Off                  | {:10.3} | {:7.3} | {:7.3}",
            off_median, off_min, off_max,
        );
        println!(
            "Detailed telemetry   | {:10.3} | {:7.3} | {:7.3}",
            telemetry_median, telemetry_min, telemetry_max,
        );
        println!(
            "Paired delta (D-Off) | {:10.3} | {:7.3} | {:7.3} | relative_to_off={:.2}%",
            delta_median,
            delta_min,
            delta_max,
            100.0 * delta_median / off_median,
        );
        println!(
            "[BVP Damp telemetry scopes] iterations={} trials={} refinements={} invalidations={} iteration_ms={:.3} trial_ms={:.3}",
            telemetry_snapshot.counters.iterations,
            telemetry_snapshot.counters.damping_trials,
            telemetry_snapshot.counters.grid_refinements,
            telemetry_snapshot.counters.factorization_invalidations,
            telemetry_snapshot.scopes.iteration.elapsed.as_secs_f64() * 1_000.0,
            telemetry_snapshot
                .scopes
                .damping_trial
                .elapsed
                .as_secs_f64()
                * 1_000.0,
        );
        println!(
            "[BVP Damp logging report] wall_ms={:.3} log_events={} counters={:?}",
            detailed_logging_run.wall_ms,
            detailed_snapshot.log_events.len(),
            detailed_snapshot.counters,
        );

        let report = format!(
            "status: passed\n\nsamples: {SAMPLES}\n\noff_median_ms: {off_median:.3}\noff_min_ms: {off_min:.3}\noff_max_ms: {off_max:.3}\ndetailed_median_ms: {telemetry_median:.3}\ndetailed_min_ms: {telemetry_min:.3}\ndetailed_max_ms: {telemetry_max:.3}\npaired_delta_median_ms: {delta_median:.3}\npaired_delta_min_ms: {delta_min:.3}\npaired_delta_max_ms: {delta_max:.3}\nrelative_to_off_percent: {:.2}\n\ntelemetry_scopes: iterations={} damping_trials={} refinements={} invalidations={} iteration_ms={:.3} trial_ms={:.3}\nlogging_wall_ms: {:.3}\nlogging_event_count: {}\nlogging_counters: {:?}\n",
            100.0 * delta_median / off_median,
            telemetry_snapshot.counters.iterations,
            telemetry_snapshot.counters.damping_trials,
            telemetry_snapshot.counters.grid_refinements,
            telemetry_snapshot.counters.factorization_invalidations,
            telemetry_snapshot.scopes.iteration.elapsed.as_secs_f64() * 1_000.0,
            telemetry_snapshot
                .scopes
                .damping_trial
                .elapsed
                .as_secs_f64()
                * 1_000.0,
            detailed_logging_run.wall_ms,
            detailed_snapshot.log_events.len(),
            detailed_snapshot.counters,
        );
        if let Err(error) = write_test_report(
            "BVP_Damp",
            "telemetry_off_vs_detailed_adaptive_story",
            &report,
        ) {
            eprintln!("[BVP test report] unable to write telemetry story report: {error}");
        }
    }
}
