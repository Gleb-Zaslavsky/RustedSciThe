use crate::numerical::BDF::BDF_api::{
    BdfSolveError, BdfSolverOptions, BdfTelemetryMode, ODEsolver,
};
use crate::numerical::BDF::BDF_solver::{BdfLinearBackend, BdfLinearFactorization};
use crate::symbolic::symbolic_engine::Expr;
use nalgebra::{DMatrix, DVector};

fn native_solver(mode: BdfTelemetryMode) -> ODEsolver {
    let options = BdfSolverOptions::new(
        vec![Expr::parse_expression("-y")],
        vec!["y".to_string()],
        "t".to_string(),
        "BDF".to_string(),
        0.0,
        DVector::from_vec(vec![1.0]),
        0.03,
        0.01,
        1e-6,
        1e-8,
        None,
        false,
        Some(0.01),
    )
    .with_telemetry_mode(mode);
    let mut solver = ODEsolver::new_with_options(options);
    solver.set_native_ode_callbacks(
        |_: f64, y: &DVector<f64>| DVector::from_element(y.len(), -y[0]),
        Some(|_: f64, _: &DVector<f64>| DMatrix::from_element(1, 1, -1.0)),
    );
    solver
}

#[test]
fn telemetry_is_off_by_default_and_collects_nothing() {
    let mut solver = native_solver(BdfTelemetryMode::Off);
    assert_eq!(solver.telemetry_mode(), BdfTelemetryMode::Off);
    solver.solve();

    let stats = solver.get_statistics();
    assert_eq!(stats.backend_prepare_calls, 0);
    assert_eq!(stats.solve_calls, 0);
    assert_eq!(stats.step_calls, 0);
    assert_eq!(stats.accepted_steps_total, 0);
    assert_eq!(stats.candidate_step_attempts_total, 0);
    assert_eq!(stats.rejected_step_attempts_total, 0);
    assert_eq!(stats.linear_solve_attempts_total, 0);
    assert_eq!(stats.residual_calls, 0);
    assert_eq!(stats.jacobian_calls, 0);
    assert_eq!(stats.solve_ms_total, 0.0);
    assert_eq!(stats.integration_loop_ms_total, 0.0);
    assert_eq!(stats.bdf_step_ms_total, 0.0);
    assert_eq!(stats.output_collection_ms_total, 0.0);
    assert_eq!(stats.result_assembly_ms_total, 0.0);
    assert_eq!(stats.linear_factorization_ms_total, 0.0);
    assert_eq!(stats.linear_solve_ms_total, 0.0);
}

#[test]
fn changing_telemetry_mode_rebuilds_callbacks_and_resets_statistics() {
    let mut solver = native_solver(BdfTelemetryMode::Off);
    solver.solve();
    assert_eq!(solver.get_statistics().solve_calls, 0);

    solver.set_telemetry_mode(BdfTelemetryMode::Timings);
    solver.solve();

    let timed_stats = solver.get_statistics();
    assert_eq!(timed_stats.backend_prepare_calls, 1);
    assert_eq!(timed_stats.solve_calls, 1);
    assert!(timed_stats.solve_ms_total > 0.0);
    assert!(timed_stats.residual_calls > 0);

    solver.set_telemetry_mode(BdfTelemetryMode::Off);
    solver.solve();
    let off_stats = solver.get_statistics();
    assert_eq!(off_stats.solve_calls, 0);
    assert_eq!(off_stats.backend_prepare_calls, 0);
    assert_eq!(off_stats.residual_calls, 0);
}

#[test]
fn counters_mode_counts_without_reading_timers() {
    let mut solver = native_solver(BdfTelemetryMode::Counters);
    solver.solve();

    let stats = solver.get_statistics();
    assert_eq!(stats.backend_prepare_calls, 1);
    assert_eq!(stats.solve_calls, 1);
    assert!(stats.step_calls > 0);
    assert!(stats.residual_calls > 0);
    assert!(stats.jacobian_calls > 0);
    assert!(stats.bdf_nlu_total > 0);
    assert!(stats.linear_solve_attempts_total > 0);
    assert!(stats.accepted_steps_total > 0);
    assert_eq!(stats.accepted_steps_total, stats.step_calls);
    assert_eq!(
        stats.candidate_step_attempts_total,
        stats.accepted_steps_total + stats.rejected_step_attempts_total
    );
    assert_eq!(stats.linear_solve_ms_total, 0.0);
    assert_eq!(stats.bdf_nfev_total, stats.residual_calls);
    assert!(stats.nonlinear_solve_calls > 0);
    assert!(stats.nonlinear_iterations_total >= stats.nonlinear_solve_calls);
    assert_eq!(stats.solve_ms_total, 0.0);
    assert_eq!(stats.integration_loop_ms_total, 0.0);
    assert_eq!(stats.bdf_step_ms_total, 0.0);
    assert_eq!(stats.output_collection_ms_total, 0.0);
    assert_eq!(stats.residual_ms_total, 0.0);
    assert_eq!(stats.jacobian_ms_total, 0.0);
    assert_eq!(stats.result_assembly_ms_total, 0.0);
    assert_eq!(stats.linear_factorization_ms_total, 0.0);
}

#[test]
fn timings_mode_collects_calls_and_durations() {
    let mut solver = native_solver(BdfTelemetryMode::Timings);
    solver.solve();

    let stats = solver.get_statistics();
    assert_eq!(stats.backend_prepare_calls, 1);
    assert_eq!(stats.solve_calls, 1);
    assert!(stats.step_calls > 0);
    assert!(stats.residual_calls > 0);
    assert!(stats.jacobian_calls > 0);
    assert_eq!(stats.bdf_nfev_total, stats.residual_calls);
    assert!(stats.nonlinear_solve_calls > 0);
    assert!(stats.nonlinear_iterations_total >= stats.nonlinear_solve_calls);
    assert!(stats.solve_ms_total > 0.0);
    assert!(stats.integration_loop_ms_total > 0.0);
    assert!(stats.bdf_step_ms_total > 0.0);
    assert!(stats.output_collection_ms_total > 0.0);
    assert!(stats.residual_ms_total > 0.0);
    assert!(stats.jacobian_ms_total > 0.0);
    assert!(stats.result_assembly_ms_total > 0.0);
    assert!(stats.linear_factorization_ms_total > 0.0);
    assert!(stats.linear_solve_ms_total > 0.0);
    assert!(
        stats.integration_loop_ms_total + stats.result_assembly_ms_total
            <= stats.solve_ms_total + 1e-9,
        "sequential child scopes must fit inside the solve scope"
    );
    assert!(
        stats.bdf_step_ms_total + stats.output_collection_ms_total
            <= stats.integration_loop_ms_total + 1e-9,
        "step and output-copy scopes must fit inside the integration loop"
    );
    assert!(stats.accepted_steps_total > 0);
    assert_eq!(stats.accepted_steps_total, stats.step_calls);
    assert_eq!(
        stats.candidate_step_attempts_total,
        stats.accepted_steps_total + stats.rejected_step_attempts_total
    );
    let report = stats.table_report();
    assert!(report.contains("result_assembly_ms="));
    assert!(report.contains("integration_loop_ms="));
    assert!(report.contains("bdf_step_ms="));
    assert!(report.contains("output_collection_ms="));
    assert!(report.contains("linear_factorization_ms="));
    assert!(report.contains("linear_solve_ms="));
    assert!(report.contains("(nested)"));
}

#[test]
fn counters_include_finite_difference_rhs_probes() {
    let mut solver = native_solver(BdfTelemetryMode::Counters);
    let no_jacobian: Option<fn(f64, &DVector<f64>) -> DMatrix<f64>> = None;
    solver.set_native_ode_callbacks(
        |_: f64, y: &DVector<f64>| DVector::from_element(y.len(), -y[0]),
        no_jacobian,
    );
    solver.solve();

    let stats = solver.get_statistics();
    assert!(stats.bdf_nfev_total > 0);
    assert!(stats.bdf_njev_total > 0);
    assert_eq!(stats.bdf_nfev_total, stats.residual_calls);
    assert_eq!(stats.solve_ms_total, 0.0);
}

#[test]
fn telemetry_counts_rejected_candidate_steps() {
    let options = BdfSolverOptions::new(
        vec![Expr::parse_expression("-1000*y")],
        vec!["y".to_string()],
        "t".to_string(),
        "BDF".to_string(),
        0.0,
        DVector::from_vec(vec![1.0]),
        0.03,
        0.03,
        1e-6,
        1e-8,
        None,
        false,
        Some(0.03),
    )
    .with_telemetry_mode(BdfTelemetryMode::Counters);
    let mut solver = ODEsolver::new_with_options(options);
    solver.set_native_ode_callbacks(
        |_: f64, y: &DVector<f64>| DVector::from_element(y.len(), -1000.0 * y[0]),
        Some(|_: f64, _: &DVector<f64>| DMatrix::from_element(1, 1, -1000.0)),
    );
    solver.solve();

    let stats = solver.get_statistics();
    assert!(stats.accepted_steps_total > 0);
    assert!(stats.rejected_step_attempts_total > 0);
    assert_eq!(
        stats.candidate_step_attempts_total,
        stats.accepted_steps_total + stats.rejected_step_attempts_total
    );
}

#[test]
fn finite_difference_jacobian_is_refreshed_after_accepted_nonlinear_steps() {
    let options = BdfSolverOptions::for_bdf(
        vec![Expr::parse_expression("-5*y^2")],
        vec!["y".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_element(1, 1.0),
        0.4,
        0.05,
        1e-8,
        1e-11,
        None,
        false,
        Some(0.002),
    )
    .with_telemetry_mode(BdfTelemetryMode::Counters);
    let mut solver = ODEsolver::new_with_options(options);
    let no_jacobian: Option<fn(f64, &DVector<f64>) -> DMatrix<f64>> = None;
    solver.set_native_ode_callbacks(
        |_: f64, y: &DVector<f64>| DVector::from_element(1, -5.0 * y[0] * y[0]),
        no_jacobian,
    );
    solver.solve();

    let stats = solver.get_statistics();
    let (_, trajectory) = solver.get_result_ref();
    let final_y = trajectory[(trajectory.nrows() - 1, 0)];
    let exact_y = 1.0 / (1.0 + 5.0 * 0.4);
    assert_eq!(solver.get_status(), "finished");
    assert!((final_y - exact_y).abs() < 2e-7, "FD Jacobian solution error={:e}", (final_y - exact_y).abs());
    assert!(
        stats.bdf_njev_total > stats.accepted_steps_total,
        "finite-difference J must be refreshed as y changes; njev={} accepted={}",
        stats.bdf_njev_total,
        stats.accepted_steps_total
    );
}

#[test]
fn fallible_solve_enforces_max_steps_and_keeps_partial_trajectory() {
    let options = BdfSolverOptions::for_bdf(
        vec![Expr::parse_expression("-y")],
        vec!["y".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_element(1, 1.0),
        0.5,
        0.01,
        1e-6,
        1e-8,
        None,
        false,
        Some(0.01),
    )
    .with_max_steps(1);
    let mut solver = ODEsolver::new_with_options(options);
    solver.set_native_ode_callbacks(
        |_: f64, y: &DVector<f64>| DVector::from_element(1, -y[0]),
        Some(|_: f64, _: &DVector<f64>| DMatrix::from_element(1, 1, -1.0)),
    );

    assert!(matches!(
        solver.try_solve(),
        Err(BdfSolveError::MaxStepsExceeded { max_steps: 1 })
    ));
    assert_eq!(solver.get_status(), "failed");
    let (times, states) = solver.get_result_ref();
    assert_eq!(times.len(), states.nrows());
    assert!(times.len() >= 2, "initial and accepted state are retained");
    assert_eq!(times[0], 0.0);
    assert!(times[1] > times[0]);
}

struct SingularBackend;

impl BdfLinearBackend for SingularBackend {
    fn factor(&mut self, _: &DMatrix<f64>) -> Option<Box<dyn BdfLinearFactorization>> {
        None
    }
}

#[test]
fn fallible_solve_returns_typed_step_error_without_empty_result_panic() {
    let mut solver = native_solver(BdfTelemetryMode::Counters);
    solver.set_bdf_linear_backend_factory(|| Box::new(SingularBackend));

    let result = solver.try_solve();
    assert!(matches!(
        result,
        Err(BdfSolveError::Step(_))
    ));
    assert_eq!(solver.get_status(), "failed");
    let (times, states) = solver.get_result_ref();
    assert_eq!(times.len(), 1);
    assert_eq!(states.nrows(), 1);
    assert_eq!(times[0], 0.0);
    assert_eq!(states[(0, 0)], 1.0);
}

#[test]
fn zero_step_budget_is_a_typed_configuration_error() {
    let options = BdfSolverOptions::for_bdf(
        vec![Expr::parse_expression("-y")],
        vec!["y".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_element(1, 1.0),
        0.1,
        0.01,
        1e-6,
        1e-8,
        None,
        false,
        None,
    )
    .with_max_steps(0);
    let mut solver = ODEsolver::new_with_options(options);
    assert!(matches!(solver.try_solve(), Err(BdfSolveError::InvalidMaxSteps)));
}
