use super::{
    Lsode2BackendConfig, Lsode2ControllerConfig, Lsode2DstodaState, Lsode2IterationMode,
    Lsode2JacobianBackend, Lsode2LinearSolverBackend, Lsode2MethodFamily,
    Lsode2NativeExecutionConfig, Lsode2NativeIntegrationLimits, Lsode2NativeStatistics,
    Lsode2ProblemConfig, Lsode2Solver, Lsode2SwitchReason, Lsode2SwitchTelemetry,
};
use crate::symbolic::codegen::codegen_aot_runtime_link::{
    LinkedResidualAotBackend, register_linked_residual_backend, unregister_linked_residual_backend,
};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp::{
    SymbolicIvpProblemOptions, prepare_symbolic_ivp_residual_problem,
};
use crate::symbolic::symbolic_ivp_generated::SymbolicIvpGeneratedBackendConfig;
use nalgebra::DVector;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

fn exponential_decay_config() -> Lsode2ProblemConfig {
    Lsode2ProblemConfig::new(
        vec![Expr::parse_expression("-y")],
        vec!["y".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0]),
        1.0,
        0.02,
        1e-6,
        1e-8,
    )
}

fn parameterized_decay_config() -> Lsode2ProblemConfig {
    Lsode2ProblemConfig::new(
        vec![Expr::parse_expression("a*y")],
        vec!["y".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0]),
        1.0,
        0.02,
        1e-6,
        1e-8,
    )
    .with_equation_parameters(vec!["a".to_string()])
    .with_equation_parameter_values(DVector::from_vec(vec![-1.0]))
}

fn stiff_relaxation_config() -> Lsode2ProblemConfig {
    Lsode2ProblemConfig::new(
        vec![Expr::parse_expression("-10000*(y-cos(t))-sin(t)")],
        vec!["y".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0]),
        1.0,
        0.25,
        1e-6,
        1e-8,
    )
    .with_first_step(Some(0.25))
}

fn assert_exponential_decay_solve(config: Lsode2ProblemConfig) {
    let mut solver = Lsode2Solver::new(config).expect("dense symbolic LSODE2 config should build");
    solver
        .solve()
        .expect("LSODE2 dense symbolic solve should finish");

    let (t, y) = solver.get_result();
    assert_eq!(solver.status(), "finished");
    assert!(!t.is_empty());
    assert_eq!(y.ncols(), 1);

    let y_final = y[(y.nrows() - 1, 0)];
    let expected = (-1.0_f64).exp();
    assert!(
        (y_final - expected).abs() < 1e-4,
        "exp decay mismatch: got={y_final:e}, expected={expected:e}"
    );
}

fn solve_exponential_decay(config: Lsode2ProblemConfig) -> Lsode2Solver {
    let mut solver = Lsode2Solver::new(config).expect("LSODE2 config should build");
    solver.solve().expect("LSODE2 solve should finish");
    solver
}

#[test]
fn lsode2_dense_symbolic_bdf_solves_exponential_decay() {
    assert_exponential_decay_solve(exponential_decay_config());
}

#[test]
fn lsode2_sparse_faer_bdf_solves_exponential_decay() {
    assert_exponential_decay_solve(exponential_decay_config().with_native_sparse_faer_backend());
}

#[test]
fn lsode2_banded_faithful_bdf_solves_exponential_decay() {
    assert_exponential_decay_solve(
        exponential_decay_config().with_native_banded_faithful_backend(),
    );
}

#[test]
fn lsode2_dense_aot_config_surface_builds_solver_without_native_backend() {
    let config = exponential_decay_config().with_dense_aot_c_tcc("target/lsode2-tests");
    let solver = Lsode2Solver::new(config).expect("dense AOT LSODE2 config should build");

    assert_eq!(
        solver.config().backend.linear_solver_backend,
        Lsode2LinearSolverBackend::Dense
    );
    assert_eq!(
        solver
            .config()
            .backend
            .generated_backend
            .aot_c_compiler
            .as_deref(),
        Some("tcc")
    );
}

#[test]
fn lsode2_sparse_native_path_records_residual_jacobian_and_lu_stats() {
    let solver = solve_exponential_decay(
        exponential_decay_config().with_backend(
            Lsode2BackendConfig::default()
                .with_linear_solver_backend(Lsode2LinearSolverBackend::SparseFaer),
        ),
    );
    let stats = solver.statistics();

    assert!(
        stats.residual_calls > 0,
        "native sparse path should call residual"
    );
    assert!(
        stats.jacobian_calls > 0,
        "native sparse path should record native Jacobian evaluations"
    );
    assert!(
        stats.jacobian_ms_total.is_finite(),
        "native sparse Jacobian timing should be finite"
    );
    assert!(
        stats.bdf_nlu_total > 0,
        "native sparse path should factor Newton systems"
    );
}

#[test]
fn lsode2_banded_native_path_records_residual_jacobian_and_lu_stats() {
    let solver = solve_exponential_decay(
        exponential_decay_config().with_backend(
            Lsode2BackendConfig::default()
                .with_linear_solver_backend(Lsode2LinearSolverBackend::BandedFaithful),
        ),
    );
    let stats = solver.statistics();

    assert!(
        stats.residual_calls > 0,
        "native banded path should call residual"
    );
    assert!(
        stats.jacobian_calls > 0,
        "native banded path should record native Jacobian evaluations"
    );
    assert!(
        stats.jacobian_ms_total.is_finite(),
        "native banded Jacobian timing should be finite"
    );
    assert!(
        stats.bdf_nlu_total > 0,
        "native banded path should factor Newton systems"
    );
}

#[test]
fn lsode2_prepare_is_idempotent_and_separate_from_solve() {
    let mut solver =
        Lsode2Solver::new(exponential_decay_config().with_native_banded_faithful_backend())
            .expect("LSODE2 native banded config should build");

    assert!(!solver.is_prepared());
    solver.prepare().expect("LSODE2 prepare should succeed");
    assert!(solver.is_prepared());

    let stats_after_first_prepare = solver.statistics();
    assert_eq!(stats_after_first_prepare.backend_prepare_calls, 1);
    assert_eq!(stats_after_first_prepare.solve_calls, 0);
    assert_eq!(stats_after_first_prepare.bdf_nlu_total, 0);

    solver
        .prepare()
        .expect("second LSODE2 prepare should be a no-op");
    assert_eq!(solver.statistics().backend_prepare_calls, 1);

    solver.solve().expect("prepared LSODE2 solve should finish");
    assert_eq!(solver.status(), "finished");
    assert_eq!(solver.statistics().backend_prepare_calls, 1);
}

#[test]
fn lsode2_native_statistics_track_prepare_solve_and_controller_decision() {
    let mut solver =
        Lsode2Solver::new(exponential_decay_config().with_native_banded_faithful_backend())
            .expect("LSODE2 native banded config should build");

    solver.prepare().expect("LSODE2 prepare should succeed");
    solver.solve().expect("LSODE2 solve should finish");

    let native = solver.native_statistics();
    assert_eq!(native.backend_prepare_calls, 1);
    assert_eq!(native.solve_calls, 1);
    assert_eq!(native.algorithm_decision_calls, 1);
    assert_eq!(native.preferred_bdf_count, 1);
    assert_eq!(native.executed_bdf_count, 1);
    assert_eq!(native.bridge_prepare_calls, 1);
    assert!(native.bridge_step_calls > 0);
    assert!(native.bridge_bdf_nlu_total > 0);
    assert!(native.native_step_attempts > 0);
    assert!(native.native_residual_calls > 0);
    assert!(native.native_jacobian_calls > 0);
    assert!(native.native_linear_solve_calls > 0);
    let probe = solver
        .native_step_probe()
        .expect("native banded backend should expose a native step probe");
    assert!(probe.iterations > 0);
    assert!(probe.attempted_steps > 0);
    assert!(probe.attempted_steps >= probe.accepted_steps);
    assert_eq!(
        probe.attempted_steps,
        probe.accepted_steps + probe.rejected_steps
    );
    assert!(
        probe.accepted_steps > 0 || probe.rejected_steps > 0,
        "native preflight should end in either accepted or rejected step attempts"
    );
    assert!(probe.h_trial > 0.0);
    assert!(probe.t_trial > solver.config().t0);
    assert!(probe.final_t >= solver.config().t0);
}

#[test]
fn lsode2_solve_with_summary_reports_final_state_and_statistics() {
    let mut solver =
        Lsode2Solver::new(exponential_decay_config().with_native_sparse_faer_backend())
            .expect("LSODE2 native sparse config should build");

    let summary = solver
        .solve_with_summary()
        .expect("LSODE2 solve with summary should finish");

    assert_eq!(summary.method, "bdf");
    assert_eq!(summary.jacobian_backend, "symbolic_generated");
    assert_eq!(summary.linear_solver_backend, "faer_sparse_lu");
    assert_eq!(summary.status, "finished");
    assert!(summary.time_points > 0);
    assert_eq!(summary.variable_count, 1);
    let final_t = summary.final_t.expect("final t should be available");
    assert!(final_t.is_finite(), "final t should be finite");
    assert!(
        (final_t - 1.0).abs() <= 0.02,
        "final t should land within one configured max_step of t_bound, got={final_t:e}"
    );

    let final_y = summary.final_y.expect("final y should be available");
    let expected = (-1.0_f64).exp();
    assert!(
        (final_y[0] - expected).abs() < 1e-4,
        "summary final y mismatch: got={:e}, expected={:e}",
        final_y[0],
        expected
    );
    assert!(summary.max_abs_solution >= final_y[0].abs());
    assert_eq!(summary.algorithm.controller_mode, "bdf_only");
    assert_eq!(summary.algorithm.active_family, "bdf");
    assert_eq!(summary.algorithm.preferred_family, "bdf");
    assert_eq!(summary.algorithm.executed_family, Some("bdf"));
    assert_eq!(summary.algorithm.switch_reason, "fixed_controller");
    assert!(!summary.algorithm.switch_uses_fallback);
    assert!(!summary.algorithm.method_switching_enabled);
    assert!(summary.algorithm.bdf_current_order.is_some());
    assert_eq!(summary.algorithm.bdf_max_order_cap, Some(5));
    assert!(summary.algorithm.bdf_equal_step_count.is_some());
    assert_eq!(summary.statistics.backend_prepare_calls, 1);
    assert_eq!(summary.statistics.solve_calls, 1);
    assert!(summary.statistics.step_calls > 0);
    assert_eq!(summary.native_statistics.backend_prepare_calls, 1);
    assert_eq!(summary.native_statistics.solve_calls, 1);
    assert_eq!(summary.native_statistics.preferred_bdf_count, 1);
    assert_eq!(summary.native_statistics.executed_bdf_count, 1);
    assert_eq!(summary.native_statistics.bridge_prepare_calls, 1);
    assert!(summary.native_statistics.bridge_bdf_nlu_total > 0);
    let probe = summary
        .native_step_probe
        .as_ref()
        .expect("native sparse summary should carry a step probe");
    assert!(probe.iterations > 0);
    assert!(probe.attempted_steps > 0);
    assert!(probe.attempted_steps >= probe.accepted_steps);
    assert_eq!(
        probe.attempted_steps,
        probe.accepted_steps + probe.rejected_steps
    );
    assert!(
        probe.accepted_steps > 0 || probe.rejected_steps > 0,
        "native preflight should end in either accepted or rejected step attempts"
    );
    assert!(probe.h_trial > 0.0);
    assert!(probe.t_trial > 0.0);
    assert!(probe.final_t >= 0.0);
    assert!(summary.native_integration_preview.is_none());
    assert!(summary.native_integration_solve.is_none());
}

#[test]
fn lsode2_dense_backend_skips_native_step_probe() {
    let mut solver =
        Lsode2Solver::new(exponential_decay_config()).expect("LSODE2 dense config should build");

    solver.solve().expect("LSODE2 dense solve should finish");

    assert!(
        solver.native_step_probe().is_none(),
        "dense bridge path should not claim a native sparse/banded probe"
    );
    let native = solver.native_statistics();
    assert_eq!(native.native_step_attempts, 0);
    assert_eq!(native.native_residual_calls, 0);
    assert_eq!(native.native_jacobian_calls, 0);
    assert_eq!(native.native_linear_solve_calls, 0);
}

#[test]
fn lsode2_sparse_native_step_probe_records_solver_level_probe() {
    let mut solver =
        Lsode2Solver::new(exponential_decay_config().with_native_sparse_faer_backend())
            .expect("LSODE2 native sparse config should build");

    solver.solve().expect("LSODE2 sparse solve should finish");

    let probe = solver
        .native_step_probe()
        .expect("native sparse backend should expose a step probe");
    assert!(probe.iterations > 0);
    assert!(probe.attempted_steps > 0);
    assert!(probe.attempted_steps >= probe.accepted_steps);
    assert_eq!(
        probe.attempted_steps,
        probe.accepted_steps + probe.rejected_steps
    );
    assert!(
        probe.accepted_steps > 0 || probe.rejected_steps > 0,
        "native preflight should end in either accepted or rejected step attempts"
    );
    assert!(probe.h_trial > 0.0);
    assert!(probe.t_trial > solver.config().t0);
    assert!(probe.final_t >= solver.config().t0);

    let native = solver.native_statistics();
    assert!(native.native_step_attempts > 0);
    assert!(native.native_residual_calls > 0);
    assert!(native.native_jacobian_calls > 0);
    assert!(native.native_linear_solve_calls > 0);
}

#[test]
fn lsode2_dense_native_integration_preview_skips_cleanly() {
    let mut solver =
        Lsode2Solver::new(exponential_decay_config()).expect("LSODE2 dense config should build");

    let summary = solver
        .run_native_integration_preview(Lsode2NativeIntegrationLimits::new(4, 2))
        .expect("dense native preview should not error");

    assert!(summary.is_none());
    let native = solver.native_statistics();
    assert_eq!(native.algorithm_decision_calls, 1);
    assert_eq!(native.native_step_attempts, 0);
    assert_eq!(native.native_residual_calls, 0);
    assert_eq!(native.native_jacobian_calls, 0);
    assert_eq!(native.native_linear_solve_calls, 0);
}

#[test]
fn lsode2_sparse_native_integration_preview_returns_solver_level_summary() {
    let mut solver =
        Lsode2Solver::new(exponential_decay_config().with_native_sparse_faer_backend())
            .expect("LSODE2 native sparse config should build");

    let summary = solver
        .run_native_integration_preview(Lsode2NativeIntegrationLimits::new(4, 2))
        .expect("native sparse preview should run")
        .expect("native sparse preview should return a summary");

    assert!(summary.attempted_steps > 0);
    assert_eq!(
        summary.attempted_steps,
        summary.accepted_steps + summary.rejected_steps
    );
    assert!(summary.total_iterations > 0);
    assert!(summary.first_report.predicted.t_trial > solver.config().t0);
    assert!(summary.final_t >= solver.config().t0);
    assert_eq!(summary.final_y.len(), 1);

    let native = solver.native_statistics();
    assert_eq!(native.algorithm_decision_calls, 1);
    assert!(native.native_step_attempts > 0);
    assert!(native.native_residual_calls > 0);
    assert!(native.native_jacobian_calls > 0);
    assert!(native.native_linear_solve_calls > 0);
}

#[test]
fn lsode2_sparse_native_integration_preview_can_run_explicit_adams_order1_family() {
    let mut solver =
        Lsode2Solver::new(exponential_decay_config().with_native_sparse_faer_backend())
            .expect("LSODE2 native sparse config should build");

    let summary = solver
        .run_native_integration_preview_for_family(
            Lsode2NativeIntegrationLimits::new(5, 3),
            Lsode2MethodFamily::Adams,
        )
        .expect("native sparse Adams-order1 preview should run")
        .expect("native sparse Adams-order1 preview should return a summary");

    assert!(summary.attempted_steps > 0);
    assert!(summary.accepted_steps > 0);
    assert_eq!(summary.first_report.predicted.order, 1);
    assert!(summary.last_report.predicted.order >= 1);
}

#[test]
fn lsode2_native_preview_auto_method_can_switch_to_adams_order1_after_probe_warmup() {
    let mut solver = Lsode2Solver::new(
        exponential_decay_config()
            .with_native_sparse_faer_backend()
            .with_controller(
                Lsode2ControllerConfig::automatic_adams_bdf().with_method_switch_probe_steps(1),
            ),
    )
    .expect("LSODE2 automatic-controller sparse config should build");

    let first = solver
        .run_native_integration_preview(Lsode2NativeIntegrationLimits::new(32, 16))
        .expect("first native preview should run")
        .expect("first native preview should return a summary");
    assert!(
        first.accepted_steps >= 2,
        "probe warmup for method_switch_probe_steps=1 needs at least two accepted steps"
    );

    let second = solver
        .run_native_integration_preview(Lsode2NativeIntegrationLimits::new(6, 3))
        .expect("second native preview should run")
        .expect("second native preview should return a summary");
    assert_eq!(second.first_report.predicted.order, 1);
    assert!(second.last_report.predicted.order >= 1);
}

#[test]
fn lsode2_solve_can_use_configured_native_preview_before_bridge() {
    let mut solver = Lsode2Solver::new(
        exponential_decay_config()
            .with_native_sparse_faer_backend()
            .with_native_execution(Lsode2NativeExecutionConfig::preview_before_bridge(4, 2)),
    )
    .expect("LSODE2 native-preview config should build");

    let summary = solver
        .solve_with_summary()
        .expect("LSODE2 solve with configured native preview should finish");

    assert_eq!(summary.status, "finished");
    let preview = summary
        .native_integration_preview
        .as_ref()
        .expect("configured solve should expose native integration preview");
    assert!(preview.attempted_steps > 0);
    assert_eq!(
        preview.attempted_steps,
        preview.accepted_steps + preview.rejected_steps
    );
    assert!(preview.total_iterations > 0);
    assert!(preview.final_t >= solver.config().t0);

    let probe = summary
        .native_step_probe
        .as_ref()
        .expect("configured solve should also expose legacy-compatible probe view");
    assert_eq!(probe.attempted_steps, preview.attempted_steps);
    assert_eq!(probe.accepted_steps, preview.accepted_steps);
    assert_eq!(probe.rejected_steps, preview.rejected_steps);

    assert_eq!(summary.native_statistics.algorithm_decision_calls, 1);
    assert!(summary.native_statistics.native_step_attempts > 0);
    assert!(summary.native_statistics.bridge_bdf_nlu_total > 0);
    assert!(summary.native_integration_solve.is_none());
}

#[test]
fn lsode2_solve_can_use_configured_experimental_native_solve() {
    let mut solver = Lsode2Solver::new(
        exponential_decay_config()
            .with_native_sparse_faer_backend()
            .with_experimental_native_solve(128, 128),
    )
    .expect("LSODE2 experimental native-solve config should build");

    let summary = solver
        .solve_with_summary()
        .expect("LSODE2 experimental native solve should finish");

    assert!(
        summary.status == "finished_native_experimental"
            || summary.status == "finished_native_experimental_partial"
    );
    assert!(summary.native_integration_preview.is_none());
    let native_solve = summary
        .native_integration_solve
        .as_ref()
        .expect("experimental native solve should expose native integration summary");
    assert!(native_solve.attempted_steps > 0);
    assert!(
        native_solve.attempted_steps <= native_solve.accepted_steps + native_solve.rejected_steps
    );
    assert!(native_solve.accepted_steps > 0);
    assert_eq!(
        native_solve.accepted_t_history.len(),
        native_solve.accepted_y_history.len()
    );

    let final_t = summary
        .final_t
        .expect("native solve should provide final t");
    assert!(
        final_t >= solver.config().t0 && final_t <= solver.config().t_bound,
        "native solve final t should stay inside the integration interval, got={final_t:e}"
    );
    let final_y = summary
        .final_y
        .expect("native solve should provide final y");
    assert!(
        final_y[0].is_finite(),
        "native experimental solve should keep its state finite"
    );

    let (t, y) = solver.get_result();
    assert_eq!(t.len(), native_solve.accepted_t_history.len());
    assert_eq!(y.nrows(), native_solve.accepted_y_history.len());
    assert_eq!(solver.status(), summary.status);

    assert_eq!(summary.native_statistics.algorithm_decision_calls, 1);
    assert!(summary.native_statistics.native_step_attempts > 0);
    assert_eq!(summary.native_statistics.bridge_bdf_nlu_total, 0);
    assert_eq!(summary.statistics.backend_prepare_calls, 0);
    assert_eq!(summary.statistics.solve_calls, 0);
}

#[test]
fn lsode2_experimental_native_solve_rejects_dense_backend() {
    let mut solver =
        Lsode2Solver::new(exponential_decay_config().with_experimental_native_solve(8, 8))
            .expect("dense LSODE2 config should build");

    let err = solver
        .solve_with_summary()
        .expect_err("dense backend should reject experimental native solve");
    assert!(err.to_string().contains("sparse or banded"));
}

#[test]
fn lsode2_algorithm_controller_snapshot_reports_auto_mode_bridge_fallback_honestly() {
    let mut solver = Lsode2Solver::new(
        exponential_decay_config()
            .with_controller(Lsode2ControllerConfig::automatic_adams_bdf())
            .with_native_banded_faithful_backend(),
    )
    .expect("LSODE2 automatic-controller config should build");

    let summary = solver
        .solve_with_summary()
        .expect("LSODE2 solve should finish through current BDF engine");

    assert_eq!(summary.status, "finished");
    assert_eq!(summary.algorithm.controller_mode, "automatic_adams_bdf");
    assert_eq!(summary.algorithm.active_family, "bdf");
    assert_eq!(summary.algorithm.executed_family, Some("bdf"));
    assert!(summary.algorithm.method_switching_enabled);
    assert!(
        summary.algorithm.preferred_family == "bdf"
            || summary.algorithm.preferred_family == "adams",
        "automatic controller should prefer either bdf (warmup) or adams (probe-ready), got={}",
        summary.algorithm.preferred_family
    );
    if summary.algorithm.preferred_family == "bdf" {
        assert_eq!(summary.algorithm.switch_reason, "switch_probe_warmup");
        assert!(!summary.algorithm.switch_uses_fallback);
        assert!(summary.algorithm.note.contains("warmup"));
    } else {
        assert_eq!(summary.algorithm.preferred_family, "adams");
        assert_eq!(summary.algorithm.switch_reason, "adams_engine_unavailable");
        assert!(summary.algorithm.switch_uses_fallback);
        assert!(
            summary
                .algorithm
                .note
                .contains("native Adams execution is unavailable")
        );
    }

    let stiff_decision = solver.algorithm_switch_decision_with_telemetry(
        Lsode2SwitchTelemetry::default()
            .with_stiffness_ratio(summary.algorithm.stiffness_ratio_threshold),
    );
    assert_eq!(stiff_decision.preferred_family, Lsode2MethodFamily::Bdf);
    assert_eq!(
        stiff_decision.reason,
        Lsode2SwitchReason::StiffnessSuspected
    );
    assert!(!stiff_decision.uses_fallback);
}

#[test]
fn lsode2_rejects_adams_only_on_bridge_path() {
    let config = exponential_decay_config().with_adams_only_controller();
    let plan = config.controller.execution_plan();
    assert!(!plan.is_executable_now());
    assert!(plan.requires_adams_engine);

    let err = match Lsode2Solver::new(config) {
        Ok(_) => panic!("Adams-only mode should not silently execute through BDF"),
        Err(err) => err,
    };

    let message = err.to_string();
    assert!(message.contains("Adams-only controller"));
    assert!(message.contains("requires native Adams execution support"));
}

#[test]
fn lsode2_allows_adams_only_with_experimental_native_solve() {
    let mut solver = Lsode2Solver::new(
        exponential_decay_config()
            .with_adams_only_controller()
            .with_native_sparse_faer_backend()
            .with_experimental_native_solve(512, 512),
    )
    .expect("adams-only should be allowed on native experimental execution path");

    let summary = solver
        .solve_with_summary()
        .expect("adams-only native experimental solve should complete");

    assert!(
        summary.status == "finished_native_experimental"
            || summary.status == "finished_native_experimental_partial"
    );
    assert_eq!(summary.algorithm.controller_mode, "adams_only");
    assert_eq!(summary.algorithm.preferred_family, "adams");
    assert_eq!(summary.algorithm.switch_reason, "fixed_controller");
    assert!(summary.native_integration_solve.is_some());
}

#[test]
fn lsode2_automatic_native_nonstiff_uses_cost_aware_family_selection_after_probe_warmup() {
    let mut solver = Lsode2Solver::new(
        exponential_decay_config()
            .with_automatic_adams_bdf_controller()
            .with_native_sparse_faer_backend()
            .with_experimental_native_solve(256, 128),
    )
    .expect("automatic non-stiff native config should build");

    let first = solver
        .solve_with_summary()
        .expect("first non-stiff native solve should finish");
    assert!(
        first.status == "finished_native_experimental"
            || first.status == "finished_native_experimental_partial"
    );
    assert_eq!(
        first.algorithm.preferred_family, "bdf",
        "first non-stiff automatic run should stay in BDF warmup; got reason={}",
        first.algorithm.switch_reason
    );
    assert!(first.native_statistics.native_step_accepts > 0);
    assert!(
        solver
            .run_native_integration_preview_for_family(
                Lsode2NativeIntegrationLimits::new(64, 32),
                Lsode2MethodFamily::Adams,
            )
            .expect("non-stiff adams probe should run")
            .is_some(),
        "non-stiff adams probe should produce a native summary"
    );

    let second = solver
        .solve_with_summary()
        .expect("second non-stiff native solve should finish");
    assert!(
        second.status == "finished_native_experimental"
            || second.status == "finished_native_experimental_partial"
    );
    assert_eq!(second.algorithm.controller_mode, "automatic_adams_bdf");
    assert!(
        second.algorithm.switch_reason == "cost_preference_adams"
            || second.algorithm.switch_reason == "cost_preference_bdf"
            || second.algorithm.switch_reason == "convergence_trouble"
            || second.algorithm.switch_reason == "stiffness_suspected",
        "automatic non-stiff path should switch by ODEPACK-style signals (cost/stiffness/convergence), not heuristic fallback; got={}",
        second.algorithm.switch_reason
    );
    assert_eq!(
        second.algorithm.executed_family,
        Some(second.algorithm.preferred_family),
        "native automatic path should execute the same family it prefers"
    );
    assert!(
        second.native_statistics.native_adams_cost_samples > 0
            && second.native_statistics.native_bdf_cost_samples > 0,
        "non-stiff automatic switch should be backed by recorded native cost evidence for both families"
    );
}

#[test]
fn lsode2_automatic_native_stiff_keeps_bdf_family() {
    let mut solver = Lsode2Solver::new(
        stiff_relaxation_config()
            .with_automatic_adams_bdf_controller()
            .with_native_sparse_faer_backend()
            .with_experimental_native_solve(512, 256),
    )
    .expect("automatic stiff native config should build");

    let first = solver
        .solve_with_summary()
        .expect("first stiff native solve should finish");
    assert!(
        first.status == "finished_native_experimental"
            || first.status == "finished_native_experimental_partial"
    );
    assert!(
        solver
            .run_native_integration_preview_for_family(
                Lsode2NativeIntegrationLimits::new(96, 48),
                Lsode2MethodFamily::Bdf,
            )
            .expect("stiff bdf probe should run")
            .is_some(),
        "stiff bdf probe should produce a native summary"
    );

    let second = solver
        .solve_with_summary()
        .expect("second stiff native solve should finish");
    assert!(
        second.status == "finished_native_experimental"
            || second.status == "finished_native_experimental_partial"
    );
    assert_eq!(second.algorithm.controller_mode, "automatic_adams_bdf");
    assert!(
        second.native_statistics.native_bdf_cost_samples > 0,
        "stiff path expected at least one native BDF cost sample before automatic decision"
    );
    assert_eq!(second.algorithm.preferred_family, "bdf");
    assert_eq!(second.algorithm.executed_family, Some("bdf"));
    assert!(
        second.algorithm.switch_reason == "convergence_trouble"
            || second.algorithm.switch_reason == "stiffness_suspected"
            || second.algorithm.switch_reason == "cost_preference_bdf"
            || second.algorithm.switch_reason == "switch_probe_warmup",
        "unexpected auto-switch reason on stiff path: {}",
        second.algorithm.switch_reason
    );
    assert!(
        second.native_statistics.native_bdf_cost_samples > 0,
        "stiff automatic switch should be backed by recorded native BDF cost evidence"
    );
}

#[test]
fn lsode2_rejects_invalid_controller_order_caps() {
    let err = match Lsode2Solver::new(
        exponential_decay_config()
            .with_controller(Lsode2ControllerConfig::bdf_only().with_max_bdf_order(0)),
    ) {
        Ok(_) => panic!("invalid BDF max order should be rejected"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("BDF max order"));

    let err = match Lsode2Solver::new(
        exponential_decay_config()
            .with_controller(Lsode2ControllerConfig::bdf_only().with_max_adams_order(13)),
    ) {
        Ok(_) => panic!("invalid Adams max order should be rejected"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("Adams max order"));
}

#[test]
fn lsode2_controller_bdf_order_cap_reaches_low_level_bdf_engine() {
    let mut solver = Lsode2Solver::new(
        exponential_decay_config()
            .with_controller(Lsode2ControllerConfig::bdf_only().with_max_bdf_order(2))
            .with_native_sparse_faer_backend(),
    )
    .expect("LSODE2 capped-order config should build");

    solver
        .prepare()
        .expect("LSODE2 prepare should install capped BDF engine");

    let snapshot = solver.algorithm_snapshot();
    assert_eq!(snapshot.max_bdf_order, 2);
    assert_eq!(snapshot.bdf_current_order, Some(1));
    assert_eq!(solver.bdf_max_order_cap(), 2);
}

#[test]
fn lsode2_sparse_native_path_solves_parameterized_decay() {
    let solver = solve_exponential_decay(
        parameterized_decay_config().with_backend(
            Lsode2BackendConfig::default()
                .with_linear_solver_backend(Lsode2LinearSolverBackend::SparseFaer),
        ),
    );

    let (_, y) = solver.get_result();
    let y_final = y[(y.nrows() - 1, 0)];
    let expected = (-1.0_f64).exp();
    assert!(
        (y_final - expected).abs() < 1e-4,
        "parameterized sparse decay mismatch: got={y_final:e}, expected={expected:e}"
    );
}

#[test]
fn lsode2_banded_native_path_solves_parameterized_decay() {
    let solver = solve_exponential_decay(
        parameterized_decay_config().with_backend(
            Lsode2BackendConfig::default()
                .with_linear_solver_backend(Lsode2LinearSolverBackend::BandedFaithful),
        ),
    );

    let (_, y) = solver.get_result();
    let y_final = y[(y.nrows() - 1, 0)];
    let expected = (-1.0_f64).exp();
    assert!(
        (y_final - expected).abs() < 1e-4,
        "parameterized banded decay mismatch: got={y_final:e}, expected={expected:e}"
    );
}

#[test]
fn lsode2_rejects_parameter_value_count_mismatch() {
    let config = parameterized_decay_config()
        .with_equation_parameter_values(DVector::from_vec(vec![-1.0, -2.0]));

    let err = match Lsode2Solver::new(config) {
        Ok(_) => panic!("parameter count mismatch should be rejected"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("expected 1 parameter values"));
}

#[test]
fn lsode2_sparse_native_path_uses_updated_parameter_values_before_solve() {
    let config = parameterized_decay_config()
        .with_equation_parameter_values(DVector::from_vec(vec![-2.0]))
        .with_backend(
            Lsode2BackendConfig::default()
                .with_linear_solver_backend(Lsode2LinearSolverBackend::SparseFaer),
        );
    let mut solver = Lsode2Solver::new(config).expect("LSODE2 config should build");
    solver
        .set_parameter_values(DVector::from_vec(vec![-1.0]))
        .expect("parameter update should succeed before solve");
    solver.solve().expect("LSODE2 solve should finish");

    let (_, y) = solver.get_result();
    let y_final = y[(y.nrows() - 1, 0)];
    let expected = (-1.0_f64).exp();
    assert!(
        (y_final - expected).abs() < 1e-4,
        "updated sparse parameter mismatch: got={y_final:e}, expected={expected:e}"
    );
}

#[test]
fn lsode2_native_path_uses_updated_parameter_values_after_prepare() {
    let config = parameterized_decay_config()
        .with_equation_parameter_values(DVector::from_vec(vec![-2.0]))
        .with_native_sparse_faer_backend();
    let mut solver = Lsode2Solver::new(config).expect("LSODE2 config should build");
    solver.prepare().expect("LSODE2 prepare should succeed");
    solver
        .set_parameter_values(DVector::from_vec(vec![-1.0]))
        .expect("parameter update should succeed after prepare");
    assert!(
        !solver.is_prepared(),
        "parameter updates should invalidate cached prepared BDF state"
    );
    solver.solve().expect("LSODE2 solve should finish");

    let (_, y) = solver.get_result();
    let y_final = y[(y.nrows() - 1, 0)];
    let expected = (-1.0_f64).exp();
    assert!(
        (y_final - expected).abs() < 1e-4,
        "post-prepare parameter update mismatch: got={y_final:e}, expected={expected:e}"
    );
    assert_eq!(solver.statistics().backend_prepare_calls, 2);
}

#[test]
fn lsode2_rejects_planned_but_unwired_fd_jacobian_backend() {
    let config = Lsode2ProblemConfig::new(
        vec![Expr::parse_expression("-y")],
        vec!["y".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0]),
        1.0,
        0.05,
        1e-6,
        1e-8,
    )
    .with_backend(
        Lsode2BackendConfig::default()
            .with_jacobian_backend(Lsode2JacobianBackend::FiniteDifference),
    );

    let err = match Lsode2Solver::new(config) {
        Ok(_) => panic!("FD Jacobian is a planned backend"),
        Err(err) => err,
    };
    assert!(
        err.to_string()
            .contains("finite-difference Jacobians are planned")
    );
}

#[test]
fn lsode2_native_linear_backend_accepts_residual_generated_aot_config() {
    let config = exponential_decay_config().with_backend(
        Lsode2BackendConfig::native_banded_faithful().with_generated_backend(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release("target/lsode2-tests"),
        ),
    );

    let solver = Lsode2Solver::new(config)
        .expect("native banded path should accept residual-only generated AOT config");
    assert_eq!(
        solver.config().backend.linear_solver_backend,
        Lsode2LinearSolverBackend::BandedFaithful
    );
}

#[test]
fn lsode2_native_residual_aot_presets_select_expected_backends() {
    let sparse_tcc =
        exponential_decay_config().with_native_sparse_faer_aot_c_tcc("target/lsode2-tests");
    assert_eq!(
        sparse_tcc.backend.linear_solver_backend,
        Lsode2LinearSolverBackend::SparseFaer
    );
    assert_eq!(
        sparse_tcc
            .backend
            .generated_backend
            .aot_c_compiler
            .as_deref(),
        Some("tcc")
    );

    let banded_gcc =
        exponential_decay_config().with_native_banded_faithful_aot_c_gcc("target/lsode2-tests");
    assert_eq!(
        banded_gcc.backend.linear_solver_backend,
        Lsode2LinearSolverBackend::BandedFaithful
    );
    assert_eq!(
        banded_gcc
            .backend
            .generated_backend
            .aot_c_compiler
            .as_deref(),
        Some("gcc")
    );

    let banded_zig =
        exponential_decay_config().with_native_banded_faithful_aot_zig("target/lsode2-tests");
    assert_eq!(
        banded_zig.backend.linear_solver_backend,
        Lsode2LinearSolverBackend::BandedFaithful
    );
    assert!(
        banded_zig
            .backend
            .generated_backend
            .aot_c_compiler
            .is_none(),
        "Zig preset should not carry a C compiler override"
    );
}

#[test]
fn lsode2_native_banded_path_can_use_prelinked_residual_aot_backend() {
    let generated_backend = SymbolicIvpGeneratedBackendConfig::require_prebuilt()
        .with_crate_name_override(Some("generated_lsode2_prelinked_banded_test".to_string()))
        .with_module_name_override(Some("generated_lsode2_prelinked_banded_test".to_string()));
    let config = exponential_decay_config()
        .with_native_banded_faithful_generated_backend(generated_backend.clone());
    let residual_problem = prepare_symbolic_ivp_residual_problem(
        config.eq_system.clone(),
        config.values.clone(),
        config.arg.clone(),
        SymbolicIvpProblemOptions::new().with_aot_options(generated_backend.aot_options),
    )
    .expect("residual problem should prepare");
    let problem_key = residual_problem
        .prepare_residual_aot_problem(generated_backend.aot_options)
        .problem_key();
    let residual_calls = Arc::new(AtomicUsize::new(0));
    let residual_calls_for_backend = Arc::clone(&residual_calls);

    register_linked_residual_backend(LinkedResidualAotBackend::new(
        problem_key.clone(),
        1,
        Arc::new(move |args: &[f64], out: &mut [f64]| {
            residual_calls_for_backend.fetch_add(1, Ordering::Relaxed);
            assert_eq!(out.len(), 1);
            assert!(args.len() >= 2, "IVP residual AOT args should be [t, y...]");
            out[0] = -args[1];
        }),
    ));

    let result = (|| {
        let mut solver = Lsode2Solver::new(config).expect("LSODE2 config should build");
        solver.solve().expect("LSODE2 solve should finish");
        let (_, y) = solver.get_result();
        y[(y.nrows() - 1, 0)]
    })();

    unregister_linked_residual_backend(problem_key.as_str());

    let expected = (-1.0_f64).exp();
    assert!(
        (result - expected).abs() < 1e-4,
        "prelinked residual AOT banded solve mismatch: got={result:e}, expected={expected:e}"
    );
    assert!(
        residual_calls.load(Ordering::Relaxed) > 0,
        "prelinked residual backend should be called during solve"
    );
}

#[test]
fn lsode2_fortran_labels_replay_quality_gate() {
    let mut dstoda = Lsode2DstodaState::default();
    let mut stats = Lsode2NativeStatistics::default();

    let mut record = |state: &Lsode2DstodaState| {
        stats.record_dstoda_flags(
            state.jacobian_currency(),
            state.ipup(),
            state.ipup_trigger(),
            state.kflag(),
            state.icf(),
            state.iret(),
            state.redo_stage(),
        );
    };

    // 1) baseline accepted-like control state.
    dstoda.mark_jacobian_current(0);
    record(&dstoda);

    // 2) error-test failure path.
    dstoda.record_error_test_failure();
    record(&dstoda);

    // 3) repeated-error reset choreography.
    dstoda.record_repeated_error_test_reset();
    record(&dstoda);

    // 4) stale-J one-shot same-step refresh branch.
    dstoda.mark_jacobian_stale();
    let _ = dstoda.decide_after_corrector_failure(Lsode2IterationMode::JacobianBased);
    record(&dstoda);

    // 5) post-refresh no-recover branch.
    dstoda.mark_jacobian_current(0);
    let _ = dstoda.decide_after_corrector_failure(Lsode2IterationMode::JacobianBased);
    record(&dstoda);

    // 6) terminal repeated convergence branch.
    dstoda.record_repeated_convergence_failure();
    record(&dstoda);

    // 7) predictor-driven IPUP reason replay.
    dstoda.record_step_accepted();
    dstoda.mark_jacobian_current(0);
    dstoda.set_coefficient_ratio(1.31);
    dstoda.maybe_request_jacobian_update_before_predict(1, Lsode2IterationMode::JacobianBased);
    record(&dstoda);

    // 8) forced restart-with-derivative-refresh branch.
    dstoda.mark_jacobian_current(0);
    dstoda.record_repeated_error_test_failure();
    record(&dstoda);

    assert_eq!(stats.native_kflag_ok_count, 2);
    assert_eq!(stats.native_kflag_error_test_failure_count, 2);
    assert_eq!(stats.native_kflag_repeated_convergence_failure_count, 1);
    assert_eq!(stats.native_icf_refresh_requested_count, 1);
    assert_eq!(stats.native_icf_refresh_did_not_recover_count, 2);
    assert_eq!(stats.native_iret_retry_after_error_test_failure_count, 1);
    assert_eq!(stats.native_iret_restart_with_derivative_refresh_count, 1);
    assert_eq!(stats.native_redo_corrector_refresh_same_step_count, 1);
    assert_eq!(stats.native_redo_corrector_failure_retry_count, 2);
    assert_eq!(stats.native_redo_error_test_retry_count, 1);
    assert_eq!(stats.native_redo_repeated_error_reset_count, 1);
    assert_eq!(stats.native_ipup_trigger_predictor_rc_ccmax_count, 1);

    // terminal KFLAG=-2 class must be represented.
    assert!(stats.native_kflag_repeated_convergence_failure_count > 0);
}
