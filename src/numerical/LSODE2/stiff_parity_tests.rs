use super::{
    Lsode2ControllerConfig, Lsode2NativeExecutionConfig, Lsode2ProblemConfig,
    Lsode2ResidualJacobianSource, Lsode2Solver,
};
use crate::symbolic::symbolic_engine::Expr;
use nalgebra::{DMatrix, DVector};

fn stiff_relaxation_exact_sparse_config() -> Lsode2ProblemConfig {
    // Exact solution: y(t) = cos(t)
    // y' = -lambda*(y - cos(t)) - sin(t)
    let lambda = 1.0e5_f64;
    Lsode2ProblemConfig::new(
        vec![Expr::parse_expression("-100000.0*(y-cos(t))-sin(t)")],
        vec!["y".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0]),
        1.0,
        1.0,
        1.0e-8,
        1.0e-10,
    )
    .with_first_step(Some(0.5))
    .with_native_sparse_faer_backend()
    .with_controller(Lsode2ControllerConfig::bdf_only())
    .with_native_execution(Lsode2NativeExecutionConfig::native_solve(300_000, 300_000))
    .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Analytical)
    .with_analytical_callbacks(
        move |t, y: &DVector<f64>| {
            let val = -lambda * (y[0] - t.cos()) - t.sin();
            DVector::from_vec(vec![val])
        },
        move |_t, _y: &DVector<f64>| DMatrix::from_row_slice(1, 1, &[-lambda]),
    )
}

fn two_scale_linear_stiff_exact_sparse_config() -> Lsode2ProblemConfig {
    // y2' = -y2, y2(0)=1 => y2 = exp(-t)
    // y1' = -1000*y1 + 1000*y2, y1(0)=0
    // y1(t) = 1000/999 * (exp(-t) - exp(-1000*t))
    Lsode2ProblemConfig::new(
        vec![
            Expr::parse_expression("-1000.0*y1 + 1000.0*y2"),
            Expr::parse_expression("-y2"),
        ],
        vec!["y1".to_string(), "y2".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![0.0, 1.0]),
        10.0,
        1.0,
        1.0e-8,
        1.0e-10,
    )
    .with_first_step(Some(0.25))
    .with_native_sparse_faer_backend()
    .with_controller(Lsode2ControllerConfig::bdf_only())
    .with_native_execution(Lsode2NativeExecutionConfig::native_solve(300_000, 300_000))
    .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Analytical)
    .with_analytical_callbacks(
        |_t, y: &DVector<f64>| DVector::from_vec(vec![-1000.0 * y[0] + 1000.0 * y[1], -y[1]]),
        |_t, _y: &DVector<f64>| DMatrix::from_row_slice(2, 2, &[-1000.0, 1000.0, 0.0, -1.0]),
    )
}

fn two_scale_linear_stiff_exact_banded_config() -> Lsode2ProblemConfig {
    two_scale_linear_stiff_exact_sparse_config()
        .with_native_banded_faithful_backend()
        .with_controller(Lsode2ControllerConfig::bdf_only())
}

fn max_predicted_order(summary: &super::Lsode2NativeIntegrationSummary) -> usize {
    summary
        .attempt_reports
        .iter()
        .map(|r| r.predicted.order)
        .max()
        .unwrap_or(0)
}

#[test]
fn lsode2_bdf_stiff_exact_scalar_sparse_survives_aggressive_first_step_and_matches_exact() {
    let mut solver = Lsode2Solver::new(stiff_relaxation_exact_sparse_config())
        .expect("stiff exact scalar sparse config should build");
    let summary = solver
        .solve_with_summary()
        .expect("stiff exact scalar sparse solve should finish");

    assert!(
        summary.status == "finished_native_faithful"
            || summary.status == "finished_native_faithful_partial",
        "unexpected status for stiff exact scalar sparse run: {}",
        summary.status
    );
    let final_t = summary.final_t.expect("final t should be present");
    assert!(
        final_t >= 0.999,
        "stiff exact scalar sparse run should reach near t_bound, got t={final_t:e}"
    );
    let y_final = summary.final_y.expect("final y should be present")[0];
    let exact = final_t.cos();
    assert!(
        (y_final - exact).abs() < 3.0e-4,
        "stiff exact scalar sparse mismatch: got={y_final:e}, exact={exact:e}"
    );
    let stats = &summary.native_statistics;
    assert!(
        stats.native_step_rejects_error_test + stats.native_step_rejects_nonlinear > 0,
        "aggressive first step should trigger at least one native rejection in stiff scalar case"
    );
    assert!(
        stats.native_linear_solve_calls > 0,
        "stiff scalar BDF run should perform native linear solves"
    );
}

#[test]
fn lsode2_bdf_stiff_exact_two_scale_sparse_matches_closed_form_and_exercises_adaptation() {
    let mut solver = Lsode2Solver::new(two_scale_linear_stiff_exact_sparse_config())
        .expect("two-scale stiff sparse config should build");
    let summary = solver
        .solve_with_summary()
        .expect("two-scale stiff sparse solve should finish");

    assert!(
        summary.status == "finished_native_faithful"
            || summary.status == "finished_native_faithful_partial",
        "unexpected status for two-scale stiff sparse run: {}",
        summary.status
    );
    let final_t = summary.final_t.expect("final t should be present");
    assert!(
        final_t >= 9.99,
        "two-scale stiff sparse run should reach near t_bound, got t={final_t:e}"
    );
    let y = summary.final_y.expect("final y should be present");
    let y2_exact = (-final_t).exp();
    let y1_exact = (1000.0 / 999.0) * ((-final_t).exp() - (-1000.0 * final_t).exp());
    assert!(
        (y[0] - y1_exact).abs() < 5.0e-6,
        "two-scale stiff sparse y1 mismatch: got={:e}, exact={:e}",
        y[0],
        y1_exact
    );
    assert!(
        (y[1] - y2_exact).abs() < 5.0e-6,
        "two-scale stiff sparse y2 mismatch: got={:e}, exact={:e}",
        y[1],
        y2_exact
    );
    let stats = &summary.native_statistics;
    assert!(
        stats.native_ialth_positive_count > 0 || stats.native_ialth_sum > 0,
        "two-scale stiff sparse run should exercise step adaptation counters"
    );
    assert!(
        stats.native_linear_solve_calls > 0,
        "two-scale stiff sparse run should perform native linear solves"
    );
}

#[test]
fn lsode2_bdf_stiff_exact_two_scale_banded_matches_closed_form() {
    let mut solver = Lsode2Solver::new(two_scale_linear_stiff_exact_banded_config())
        .expect("two-scale stiff banded config should build");
    let summary = solver
        .solve_with_summary()
        .expect("two-scale stiff banded solve should finish");

    assert!(
        summary.status == "finished_native_faithful"
            || summary.status == "finished_native_faithful_partial",
        "unexpected status for two-scale stiff banded run: {}",
        summary.status
    );
    let final_t = summary.final_t.expect("final t should be present");
    assert!(
        final_t >= 9.99,
        "two-scale stiff banded run should reach near t_bound, got t={final_t:e}"
    );
    let y = summary.final_y.expect("final y should be present");
    let y2_exact = (-final_t).exp();
    let y1_exact = (1000.0 / 999.0) * ((-final_t).exp() - (-1000.0 * final_t).exp());
    assert!(
        (y[0] - y1_exact).abs() < 5.0e-6,
        "two-scale stiff banded y1 mismatch: got={:e}, exact={:e}",
        y[0],
        y1_exact
    );
    assert!(
        (y[1] - y2_exact).abs() < 5.0e-6,
        "two-scale stiff banded y2 mismatch: got={:e}, exact={:e}",
        y[1],
        y2_exact
    );
    assert!(
        summary.native_statistics.native_linear_solve_calls > 0,
        "two-scale stiff banded run should perform native linear solves"
    );
}

#[test]
fn lsode2_bdf_stiff_exact_sparse_order_trace_respects_cap_and_reaches_higher_order() {
    let config = two_scale_linear_stiff_exact_sparse_config()
        .with_controller(Lsode2ControllerConfig::bdf_only().with_max_bdf_order(2));
    let mut solver =
        Lsode2Solver::new(config).expect("capped two-scale sparse config should build");
    let summary = solver
        .solve_with_summary()
        .expect("capped two-scale sparse solve should finish");

    let native = summary
        .native_integration_solve
        .as_ref()
        .expect("faithful native solve should include integration summary");
    assert!(
        native.accepted_steps > 0,
        "capped two-scale sparse run should accept at least one step"
    );

    let mut max_order_seen = 0usize;
    for report in &native.attempt_reports {
        max_order_seen = max_order_seen.max(report.predicted.order);
        assert!(
            report.predicted.order <= 2,
            "predicted order must respect max_bdf_order=2, got {}",
            report.predicted.order
        );
    }
    assert!(
        max_order_seen >= 2,
        "expected BDF order growth up to the configured cap in smooth stiff tail; max seen={max_order_seen}"
    );
}

#[test]
fn lsode2_bdf_stiff_exact_scalar_trace_contains_retry_and_history_rescale_signals() {
    let mut solver = Lsode2Solver::new(stiff_relaxation_exact_sparse_config())
        .expect("stiff exact scalar sparse config should build");
    let summary = solver
        .solve_with_summary()
        .expect("stiff exact scalar sparse solve should finish");

    let native = summary
        .native_integration_solve
        .as_ref()
        .expect("faithful native solve should include integration summary");
    assert!(
        native.attempted_steps > 0,
        "stiff scalar run should contain native attempts"
    );

    let saw_retry = native.attempt_reports.iter().any(|r| r.retry_count > 0);
    let saw_kflag_failure = native
        .attempt_reports
        .iter()
        .any(|r| r.kflag_code < 0 || !r.accepted());
    assert!(
        saw_retry || saw_kflag_failure,
        "expected retry/failure signals in stiff scalar trace"
    );

    let stats = &summary.native_statistics;
    assert!(
        stats.native_iret_rescale_history_count > 0
            || stats.native_redo_history_or_step_size_changed_count > 0,
        "expected history-rescale choreography signals (IRET/REDO) in stiff scalar trace"
    );
}

#[test]
fn lsode2_bdf_stiff_exact_sparse_order_cap_sensitivity_is_visible_in_trace() {
    let mut solver_cap1 = Lsode2Solver::new(
        two_scale_linear_stiff_exact_sparse_config()
            .with_controller(Lsode2ControllerConfig::bdf_only().with_max_bdf_order(1)),
    )
    .expect("cap1 config should build");
    let summary_cap1 = solver_cap1
        .solve_with_summary()
        .expect("cap1 solve should finish");
    let native_cap1 = summary_cap1
        .native_integration_solve
        .as_ref()
        .expect("cap1 faithful native solve should include integration summary");

    let mut solver_cap5 = Lsode2Solver::new(
        two_scale_linear_stiff_exact_sparse_config()
            .with_controller(Lsode2ControllerConfig::bdf_only().with_max_bdf_order(5)),
    )
    .expect("cap5 config should build");
    let summary_cap5 = solver_cap5
        .solve_with_summary()
        .expect("cap5 solve should finish");
    let native_cap5 = summary_cap5
        .native_integration_solve
        .as_ref()
        .expect("cap5 faithful native solve should include integration summary");

    assert_eq!(
        max_predicted_order(native_cap1),
        1,
        "order cap=1 must keep all predicted orders at 1"
    );
    assert!(
        max_predicted_order(native_cap5) >= 2,
        "order cap=5 run should explore higher BDF order on smooth stiff tail"
    );

    let t1 = summary_cap1
        .final_t
        .expect("cap1 final t should be present");
    let y1 = summary_cap1
        .final_y
        .as_ref()
        .expect("cap1 final y should be present");
    let y2_exact_1 = (-t1).exp();
    let y1_exact_1 = (1000.0 / 999.0) * ((-t1).exp() - (-1000.0 * t1).exp());
    assert!((y1[0] - y1_exact_1).abs() < 2.0e-5);
    assert!((y1[1] - y2_exact_1).abs() < 2.0e-5);

    let t5 = summary_cap5
        .final_t
        .expect("cap5 final t should be present");
    let y5 = summary_cap5
        .final_y
        .as_ref()
        .expect("cap5 final y should be present");
    let y2_exact_5 = (-t5).exp();
    let y1_exact_5 = (1000.0 / 999.0) * ((-t5).exp() - (-1000.0 * t5).exp());
    assert!((y5[0] - y1_exact_5).abs() < 2.0e-5);
    assert!((y5[1] - y2_exact_5).abs() < 2.0e-5);

    // We avoid strict performance assertions, but with cap=1 the solver should
    // not require fewer accepted steps than cap=5 for this smooth tail problem.
    assert!(
        native_cap1.accepted_steps >= native_cap5.accepted_steps,
        "cap=1 unexpectedly accepted fewer steps than cap=5: cap1={}, cap5={}",
        native_cap1.accepted_steps,
        native_cap5.accepted_steps
    );
}

#[test]
fn lsode2_bdf_stiff_sparse_lmax2_retry_choreography_trace_is_consistent() {
    let config = two_scale_linear_stiff_exact_sparse_config()
        .with_first_step(Some(1.0))
        .with_controller(Lsode2ControllerConfig::bdf_only().with_max_bdf_order(2));
    let mut solver = Lsode2Solver::new(config).expect("stiff lmax2 config should build");
    let summary = solver
        .solve_with_summary()
        .expect("stiff lmax2 solve should finish");

    assert!(
        summary.status == "finished_native_faithful"
            || summary.status == "finished_native_faithful_partial",
        "unexpected status for stiff lmax2 run: {}",
        summary.status
    );
    let final_t = summary.final_t.expect("final t should be present");
    assert!(
        final_t >= 9.99,
        "stiff lmax2 run should reach near t_bound, got t={final_t:e}"
    );
    let y = summary.final_y.expect("final y should be present");
    let y2_exact = (-final_t).exp();
    let y1_exact = (1000.0 / 999.0) * ((-final_t).exp() - (-1000.0 * final_t).exp());
    assert!(
        (y[0] - y1_exact).abs() < 2.0e-5,
        "stiff lmax2 y1 mismatch: got={:e}, exact={:e}",
        y[0],
        y1_exact
    );
    assert!(
        (y[1] - y2_exact).abs() < 2.0e-5,
        "stiff lmax2 y2 mismatch: got={:e}, exact={:e}",
        y[1],
        y2_exact
    );

    let native = summary
        .native_integration_solve
        .as_ref()
        .expect("faithful native solve should include integration summary");
    assert!(
        native.attempted_steps > 0 && !native.attempt_reports.is_empty(),
        "stiff lmax2 run should contain native attempt reports"
    );
    for report in &native.attempt_reports {
        assert!(
            report.predicted.order <= 2,
            "predicted order must respect max_bdf_order=2, got {}",
            report.predicted.order
        );
    }
    assert!(
        max_predicted_order(native) >= 2,
        "stiff lmax2 run should still reach order 2 on smooth stiff tail"
    );

    let saw_retry = native.attempt_reports.iter().any(|r| r.retry_count > 0);
    let saw_negative_kflag = native.attempt_reports.iter().any(|r| r.kflag_code < 0);
    assert!(
        saw_retry || saw_negative_kflag,
        "stiff lmax2 trace should include retry/failure choreography signals"
    );

    let stats = &summary.native_statistics;
    if stats.native_iret_restart_with_derivative_refresh_count > 0 {
        assert!(
            stats.native_redo_repeated_error_reset_count > 0,
            "IRET restart-with-derivative-refresh should co-occur with REDO repeated-error-reset"
        );
    }
}

#[test]
fn lsode2_bdf_stiff_banded_lmax2_retry_choreography_trace_is_consistent() {
    let config = two_scale_linear_stiff_exact_banded_config()
        .with_first_step(Some(1.0))
        .with_controller(Lsode2ControllerConfig::bdf_only().with_max_bdf_order(2));
    let mut solver = Lsode2Solver::new(config).expect("stiff banded lmax2 config should build");
    let summary = solver
        .solve_with_summary()
        .expect("stiff banded lmax2 solve should finish");

    assert!(
        summary.status == "finished_native_faithful"
            || summary.status == "finished_native_faithful_partial",
        "unexpected status for stiff banded lmax2 run: {}",
        summary.status
    );
    let final_t = summary.final_t.expect("final t should be present");
    assert!(
        final_t >= 9.99,
        "stiff banded lmax2 run should reach near t_bound, got t={final_t:e}"
    );
    let y = summary.final_y.expect("final y should be present");
    let y2_exact = (-final_t).exp();
    let y1_exact = (1000.0 / 999.0) * ((-final_t).exp() - (-1000.0 * final_t).exp());
    assert!(
        (y[0] - y1_exact).abs() < 2.0e-5,
        "stiff banded lmax2 y1 mismatch: got={:e}, exact={:e}",
        y[0],
        y1_exact
    );
    assert!(
        (y[1] - y2_exact).abs() < 2.0e-5,
        "stiff banded lmax2 y2 mismatch: got={:e}, exact={:e}",
        y[1],
        y2_exact
    );

    let native = summary
        .native_integration_solve
        .as_ref()
        .expect("faithful native solve should include integration summary");
    assert!(
        native.attempted_steps > 0 && !native.attempt_reports.is_empty(),
        "stiff banded lmax2 run should contain native attempt reports"
    );
    for report in &native.attempt_reports {
        assert!(
            report.predicted.order <= 2,
            "predicted order must respect max_bdf_order=2, got {}",
            report.predicted.order
        );
    }
    assert!(
        max_predicted_order(native) >= 2,
        "stiff banded lmax2 run should still reach order 2 on smooth stiff tail"
    );

    let saw_retry = native.attempt_reports.iter().any(|r| r.retry_count > 0);
    let saw_negative_kflag = native.attempt_reports.iter().any(|r| r.kflag_code < 0);
    assert!(
        saw_retry || saw_negative_kflag,
        "stiff banded lmax2 trace should include retry/failure choreography signals"
    );

    let stats = &summary.native_statistics;
    assert!(
        stats.native_linear_solve_calls > 0,
        "stiff banded lmax2 run should perform native linear solves"
    );
}
