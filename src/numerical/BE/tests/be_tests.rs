use super::*;

fn one_state_be(t_bound: f64, h: Option<f64>) -> BE {
    let mut solver = BE::new();
    solver
        .try_set_initial(
            vec![Expr::parse_expression("-y")],
            vec!["y".to_string()],
            "t".to_string(),
            1e-10,
            20,
            h,
            0.0,
            t_bound,
            DVector::from_vec(vec![1.0]),
        )
        .unwrap();
    solver
}

fn decay_rhs(_: f64, y: &DVector<f64>) -> DVector<f64> {
    -y
}

fn decay_jac(_: f64, _: &DVector<f64>) -> DMatrix<f64> {
    DMatrix::from_element(1, 1, -1.0)
}

fn rhs_fails_after_three_quarters(t: f64, y: &DVector<f64>) -> DVector<f64> {
    if t > 0.75 {
        DVector::from_element(y.len(), f64::NAN)
    } else {
        -y
    }
}

fn constant_rhs(_: f64, _: &DVector<f64>) -> DVector<f64> {
    DVector::from_vec(vec![1.0])
}

fn nonautonomous_rhs(t: f64, _: &DVector<f64>) -> DVector<f64> {
    DVector::from_vec(vec![t])
}

fn zero_scalar_jac(_: f64, _: &DVector<f64>) -> DMatrix<f64> {
    DMatrix::zeros(1, 1)
}

fn singular_newton_jac(_: f64, _: &DVector<f64>) -> DMatrix<f64> {
    DMatrix::from_element(1, 1, 10.0)
}

#[test]
fn be_fixed_steps_clip_to_final_time_and_keep_initial_sample() {
    let mut solver = one_state_be(1.0, Some(0.3));
    solver.set_native_ode_callbacks(decay_rhs, Some(decay_jac));

    solver.try_solve().unwrap();
    let (times, states) = solver.get_result();
    let times = times.unwrap();
    let states = states.unwrap();
    assert_eq!(solver.get_status(), "finished");
    assert_eq!(times[0], 0.0);
    assert_eq!(times[times.len() - 1], 1.0);
    assert_eq!(states.nrows(), times.len());
    assert_eq!(times.len(), 5);
}

#[test]
fn be_legacy_step_heuristic_uses_remaining_interval_once() {
    let mut solver = one_state_be(1.0, None);
    solver.newton.tolerance = 0.5;
    solver.set_native_ode_callbacks(decay_rhs, Some(decay_jac));

    solver.try_solve().unwrap();

    let (times, states) = solver.get_result();
    assert_eq!(times.unwrap().as_slice(), &[0.0, 1.0]);
    assert_eq!(states.unwrap(), DMatrix::from_row_slice(2, 1, &[1.0, 0.5]));
    let stats = solver.get_statistics();
    assert_eq!(stats.step_calls, 1);
    assert_eq!(stats.residual_calls, 3);
    assert_eq!(stats.jacobian_calls, 3);
}

#[test]
fn be_parameter_rebind_reuses_prepared_symbolic_callbacks() {
    let make_parameterized = |rate: f64| {
        let mut solver = BE::new();
        solver
            .try_set_initial(
                vec![Expr::parse_expression("-rate*y")],
                vec!["y".to_string()],
                "t".to_string(),
                1e-12,
                20,
                Some(0.1),
                0.0,
                0.5,
                DVector::from_vec(vec![1.0]),
            )
            .unwrap();
        solver.try_set_equation_parameters(Some(&["rate"])).unwrap();
        solver
            .set_parameter_values(DVector::from_vec(vec![rate]))
            .unwrap();
        solver
    };

    let mut rebound = make_parameterized(1.0);
    rebound.try_solve().unwrap();
    let first_solution = rebound.y.clone();
    assert_eq!(rebound.get_statistics().backend_prepare_calls, 1);

    rebound
        .set_parameter_values(DVector::from_vec(vec![2.0]))
        .unwrap();
    assert!(rebound
        .set_parameter_values(DVector::from_vec(vec![f64::NAN]))
        .is_err());
    rebound.try_solve().unwrap();
    let rebound_solution = rebound.y.clone();
    assert_eq!(rebound.get_statistics().backend_prepare_calls, 1);
    assert_ne!(first_solution, rebound_solution);

    let mut fresh = make_parameterized(2.0);
    fresh.try_solve().unwrap();
    assert!((rebound_solution[0] - fresh.y[0]).abs() < 1e-12);
    assert_eq!(fresh.get_statistics().backend_prepare_calls, 1);

    assert!(rebound.try_set_equation_parameters(Some(&["y"])).is_err());
    assert_eq!(
        rebound.newton.equation_parameters.as_deref(),
        Some(&["rate".to_string()][..])
    );
    assert!(rebound.newton.jac.is_some());
    rebound.try_solve().unwrap();
    assert_eq!(rebound.get_statistics().backend_prepare_calls, 1);

    rebound
        .try_set_equation_parameters(Some(&["rate", "unused_scale"]))
        .unwrap();
    assert!(rebound.newton.jac.is_none());
    rebound
        .set_parameter_values(DVector::from_vec(vec![2.0, 7.0]))
        .unwrap();
    rebound.try_solve().unwrap();
    assert_eq!(rebound.get_statistics().backend_prepare_calls, 2);

    let mut fresh_schema = make_parameterized(2.0);
    fresh_schema
        .try_set_equation_parameters(Some(&["rate", "unused_scale"]))
        .unwrap();
    fresh_schema
        .set_parameter_values(DVector::from_vec(vec![2.0, 7.0]))
        .unwrap();
    fresh_schema.try_solve().unwrap();
    assert!((rebound.y[0] - fresh_schema.y[0]).abs() < 1e-12);
}

#[test]
fn be_parameter_binding_telemetry_respects_selected_mode() {
    let mut solver = one_state_be(0.2, Some(0.1));
    solver.try_set_equation_parameters(Some(&["rate"])).unwrap();
    solver
        .try_set_telemetry_mode(BeTelemetryMode::Counters)
        .unwrap();

    solver
        .set_parameter_values(DVector::from_vec(vec![2.0]))
        .unwrap();
    assert!(solver
        .set_parameter_values(DVector::from_vec(vec![f64::NAN]))
        .is_err());
    assert!(matches!(
        solver.set_parameter_values(DVector::from_vec(vec![2.0, 3.0])),
        Err(BeError::Backend(IvpBackendError::ParameterCountMismatch {
            expected: 1,
            actual: 2
        }))
    ));

    let counters = solver.detailed_statistics();
    assert_eq!(counters.parameter_bind_attempts, 3);
    assert_eq!(counters.parameter_bind_successes, 1);
    assert_eq!(counters.parameter_bind_failures, 2);
    assert_eq!(counters.parameter_bind_ms_total, 0.0);

    let mut disabled = one_state_be(0.2, Some(0.1));
    disabled
        .try_set_equation_parameters(Some(&["rate"]))
        .unwrap();
    disabled
        .try_set_telemetry_mode(BeTelemetryMode::Off)
        .unwrap();
    disabled
        .set_parameter_values(DVector::from_vec(vec![2.0]))
        .unwrap();
    assert_eq!(
        disabled.detailed_statistics(),
        BeDetailedStatistics::default()
    );

    let mut timed = one_state_be(0.2, Some(0.1));
    timed.try_set_equation_parameters(Some(&["rate"])).unwrap();
    timed
        .set_parameter_values(DVector::from_vec(vec![2.0]))
        .unwrap();
    let timings = timed.detailed_statistics();
    assert_eq!(timings.parameter_bind_attempts, 1);
    assert_eq!(timings.parameter_bind_successes, 1);
    assert_eq!(timings.parameter_bind_failures, 0);
    assert!(timings.parameter_bind_ms_total >= 0.0);
}

#[test]
fn be_accepted_state_continuation_appends_history_and_reuses_backend() {
    let make_parameterized = |t_bound| {
        let mut solver = BE::new();
        solver
            .try_set_initial(
                vec![Expr::parse_expression("-rate*y")],
                vec!["y".to_string()],
                "t".to_string(),
                1e-12,
                20,
                Some(0.125),
                0.0,
                t_bound,
                DVector::from_vec(vec![1.0]),
            )
            .unwrap();
        solver.try_set_equation_parameters(Some(&["rate"])).unwrap();
        solver
            .set_parameter_values(DVector::from_vec(vec![1.0]))
            .unwrap();
        solver
    };

    let mut continued = make_parameterized(0.5);
    continued.try_solve().unwrap();
    assert_eq!(continued.get_statistics().backend_prepare_calls, 1);
    let prefix_len = continued.get_result().0.unwrap().len();

    continued.try_continue_to(1.0).unwrap();
    let (times, states) = continued.get_result();
    let times = times.unwrap();
    let states = states.unwrap();
    assert_eq!(times.len(), 9);
    assert_eq!(states.shape(), (times.len(), 1));
    assert_eq!(times[0], 0.0);
    assert_eq!(times[prefix_len - 1], 0.5);
    assert_eq!(times[times.len() - 1], 1.0);
    assert_eq!(continued.get_statistics().backend_prepare_calls, 1);
    assert_eq!(continued.get_statistics().step_calls, 8);
    assert_eq!(continued.continuation_statistics().attempts, 1);
    assert_eq!(continued.continuation_statistics().completed, 1);
    assert_eq!(continued.continuation_statistics().failures, 0);
    assert!(continued
        .statistics_report()
        .contains("continuation: attempts=1"));

    let mut fresh = make_parameterized(1.0);
    fresh.try_solve().unwrap();
    let fresh_states = fresh.get_result().1.unwrap();
    assert_eq!(states, fresh_states);

    // The compatibility solve API remains an explicit restart from (t0, y0).
    continued.try_solve().unwrap();
    assert_eq!(continued.get_result().0.unwrap().len(), 9);
    assert_eq!(continued.get_statistics().backend_prepare_calls, 1);
}

#[test]
fn be_continuation_rejects_invalid_bounds_without_mutating_solution() {
    let mut solver = one_state_be(0.5, Some(0.1));
    solver.set_native_ode_callbacks(decay_rhs, Some(decay_jac));
    solver.try_solve().unwrap();
    let previous_times = solver.get_result().0.unwrap();
    let previous_states = solver.get_result().1.unwrap();

    for invalid_bound in [0.5, 0.4, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            solver.try_continue_to(invalid_bound),
            Err(BeError::InvalidConfiguration(_))
        ));
        assert_eq!(solver.get_result().0.unwrap(), previous_times);
        assert_eq!(solver.get_result().1.unwrap(), previous_states);
    }
    assert_eq!(solver.detailed_statistics().configuration_failures, 4);

    let telemetry = solver.continuation_statistics();
    assert_eq!(telemetry.attempts, 4);
    assert_eq!(telemetry.completed, 0);
    assert_eq!(telemetry.failures, 4);
}

#[test]
fn be_failed_continuation_keeps_all_accepted_samples_and_reports_failure() {
    let mut solver = one_state_be(0.5, Some(0.25));
    solver.set_native_ode_callbacks(rhs_fails_after_three_quarters, Some(decay_jac));
    solver.try_solve().unwrap();

    assert!(matches!(
        solver.try_continue_to(1.0),
        Err(BeError::Newton(NreError::NonFiniteCallback { .. }))
    ));
    let (times, states) = solver.get_result();
    let times = times.unwrap();
    let states = states.unwrap();
    assert_eq!(times.as_slice(), &[0.0, 0.25, 0.5, 0.75]);
    assert_eq!(states.shape(), (4, 1));
    assert_eq!(solver.t, 0.75);
    assert_eq!(solver.get_status(), "failed");

    let telemetry = solver.continuation_statistics();
    assert_eq!(telemetry.attempts, 1);
    assert_eq!(telemetry.completed, 0);
    assert_eq!(telemetry.failures, 1);
}

#[test]
fn be_initial_result_uses_time_rows_before_first_solve() {
    let mut solver = BE::new();
    solver
        .try_set_initial(
            vec![Expr::parse_expression("0"), Expr::parse_expression("0")],
            vec!["x".to_string(), "y".to_string()],
            "t".to_string(),
            1e-10,
            10,
            Some(0.1),
            0.0,
            0.0,
            DVector::from_vec(vec![2.0, 3.0]),
        )
        .unwrap();
    let (times, states) = solver.get_result();
    assert_eq!(times.unwrap().as_slice(), &[0.0]);
    assert_eq!(states.unwrap(), DMatrix::from_row_slice(1, 2, &[2.0, 3.0]));
}

#[test]
fn be_telemetry_can_be_disabled_before_callbacks_are_installed() {
    let mut solver = one_state_be(0.25, Some(0.125));
    solver
        .try_set_telemetry_mode(BeTelemetryMode::Disabled)
        .unwrap();
    solver.try_solve().unwrap();
    solver.try_continue_to(0.5).unwrap();

    assert_eq!(solver.telemetry_mode(), BeTelemetryMode::Disabled);
    let statistics = solver.get_statistics();
    assert_eq!(statistics.solve_calls, 0);
    assert_eq!(statistics.step_calls, 0);
    assert_eq!(statistics.residual_calls, 0);
    assert_eq!(statistics.jacobian_calls, 0);
    assert_eq!(
        solver.continuation_statistics(),
        &BeContinuationStatistics::default()
    );
    assert!(solver
        .statistics_report()
        .contains("attempts=0 completed=0 failures=0"));
    assert_eq!(
        solver.detailed_statistics(),
        BeDetailedStatistics::default()
    );
    assert!(matches!(
        solver.try_set_telemetry_mode(BeTelemetryMode::Enabled),
        Err(BeError::InvalidConfiguration(_))
    ));

    let mut native_solver = one_state_be(0.25, Some(0.125));
    native_solver
        .try_set_telemetry_mode(BeTelemetryMode::Disabled)
        .unwrap();
    native_solver.set_native_ode_callbacks(decay_rhs, Some(decay_jac));
    native_solver.try_solve().unwrap();
    assert_eq!(native_solver.get_statistics().residual_calls, 0);
    assert_eq!(native_solver.get_statistics().jacobian_calls, 0);
}

#[test]
fn be_counters_telemetry_skips_all_timing_fields() {
    let mut solver = one_state_be(0.25, Some(0.125));
    solver
        .try_set_telemetry_mode(BeTelemetryMode::Counters)
        .unwrap();
    solver.try_solve().unwrap();
    solver.try_continue_to(0.5).unwrap();

    let stats = solver.get_statistics();
    assert_eq!(stats.solve_calls, 2);
    assert!(stats.step_calls > 0);
    assert_eq!(stats.backend_prepare_calls, 1);
    assert!(stats.residual_calls > 0);
    assert!(stats.jacobian_calls > 0);
    assert_eq!(stats.solve_ms_total, 0.0);
    assert_eq!(stats.backend_prepare_ms_total, 0.0);
    assert_eq!(stats.residual_ms_total, 0.0);
    assert_eq!(stats.jacobian_ms_total, 0.0);
    assert_eq!(solver.continuation_statistics().attempts, 1);
    assert_eq!(solver.continuation_statistics().completed, 1);
    assert_eq!(solver.continuation_statistics().elapsed_ms_total, 0.0);
    let detailed = solver.detailed_statistics();
    assert_eq!(detailed.accepted_steps, 4);
    assert_eq!(detailed.failed_steps, 0);
    assert!(detailed.factorization_calls > 0);
    assert_eq!(detailed.linear_solve_calls, detailed.factorization_calls);
    assert_eq!(detailed.output_assembly_calls, 3);
    assert_eq!(detailed.factorization_ms_total, 0.0);
    assert_eq!(detailed.linear_solve_ms_total, 0.0);
    assert_eq!(detailed.output_assembly_ms_total, 0.0);
    assert!(solver.statistics_report().contains("telemetry=counters"));

    let mut fd_solver = one_state_be(0.125, Some(0.125));
    fd_solver
        .try_set_telemetry_mode(BeTelemetryMode::Counters)
        .unwrap();
    fd_solver.set_native_ode_callbacks(decay_rhs, None::<fn(f64, &DVector<f64>) -> DMatrix<f64>>);
    fd_solver.try_solve().unwrap();
    assert!(fd_solver.detailed_statistics().fd_rhs_evaluations > 0);
}

#[test]
fn be_pure_native_problem_needs_no_symbolic_placeholder_or_preparation() {
    let mut solver = BE::new();
    solver
        .try_set_telemetry_mode(BeTelemetryMode::Counters)
        .unwrap();
    solver
        .try_set_native_initial(
            vec!["y".to_string()],
            "t".to_string(),
            1e-12,
            20,
            Some(0.125),
            0.0,
            0.25,
            DVector::from_vec(vec![1.0]),
            decay_rhs,
            None::<fn(f64, &DVector<f64>) -> DMatrix<f64>>,
        )
        .unwrap();

    assert!(solver.newton.eq_system.is_empty());
    assert!(matches!(
        solver.try_set_equation_parameters(Some(&["rate"])),
        Err(BeError::InvalidConfiguration(_))
    ));
    assert!(matches!(
        solver.set_parameter_values(DVector::from_vec(vec![2.0])),
        Err(BeError::InvalidConfiguration(_))
    ));
    let bind_stats = solver.detailed_statistics();
    assert_eq!(bind_stats.parameter_bind_attempts, 1);
    assert_eq!(bind_stats.parameter_bind_successes, 0);
    assert_eq!(bind_stats.parameter_bind_failures, 1);
    assert_eq!(bind_stats.parameter_bind_ms_total, 0.0);

    solver.try_solve().unwrap();

    assert!((solver.y[0] - 1.0 / 1.125_f64.powi(2)).abs() < 1e-12);
    assert_eq!(solver.get_statistics().backend_prepare_calls, 0);
    assert!(solver.detailed_statistics().fd_rhs_evaluations > 0);
}

#[test]
fn be_max_steps_is_configurable_and_reports_typed_limit() {
    let mut solver = one_state_be(1.0, Some(0.1));
    solver
        .try_set_telemetry_mode(BeTelemetryMode::Counters)
        .unwrap();
    assert_eq!(solver.max_steps(), DEFAULT_BE_MAX_STEPS);
    assert!(solver.try_set_max_steps(0).is_err());
    solver.try_set_max_steps(2).unwrap();
    solver.set_native_ode_callbacks(decay_rhs, Some(decay_jac));

    assert!(matches!(
        solver.try_solve(),
        Err(BeError::StepLimit { max_steps: 2 })
    ));
    assert_eq!(solver.status(), BeStatus::Failed);
    assert_eq!(solver.detailed_statistics().step_limit_failures, 1);
    assert_eq!(solver.get_status(), "failed");
    assert_eq!(solver.get_result().0.unwrap().as_slice(), &[0.0, 0.1, 0.2]);
}

#[test]
fn be_solver_options_expose_step_limit_and_telemetry_mode() {
    let options = BeSolverOptions::new(
        vec![Expr::parse_expression("-y")],
        vec!["y".to_string()],
        "t".to_string(),
        1e-10,
        10,
        Some(0.1),
        0.0,
        1.0,
        DVector::from_vec(vec![1.0]),
    )
    .with_max_steps(7)
    .with_telemetry_mode(BeTelemetryMode::Disabled);

    let solver = BE::try_new_with_options(options).unwrap();
    assert_eq!(solver.max_steps(), 7);
    assert_eq!(solver.telemetry_mode(), BeTelemetryMode::Disabled);
}

#[test]
fn be_required_prebuilt_failure_is_typed_and_keeps_initial_sample() {
    let mut solver = BE::new();
    solver
        .try_set_initial(
            vec![Expr::parse_expression("-7.123456789*y")],
            vec!["y".to_string()],
            "t".to_string(),
            1e-10,
            20,
            Some(0.1),
            0.0,
            1.0,
            DVector::from_vec(vec![1.0]),
        )
        .unwrap();
    solver.try_solve().unwrap();
    assert_eq!(solver.get_status(), "finished");

    let output_parent = tempfile::tempdir().unwrap();
    let config = SymbolicIvpGeneratedBackendConfig::require_prebuilt()
        .with_output_parent_dir(Some(output_parent.path().to_path_buf()));
    solver.set_generated_backend_config(config);

    assert!(matches!(
        solver.try_solve(),
        Err(BeError::GeneratedBackend(_))
    ));
    assert_eq!(solver.detailed_statistics().generated_backend_failures, 1);
    assert_eq!(solver.get_status(), "failed");
    let (times, states) = solver.get_result();
    assert_eq!(times.unwrap().as_slice(), &[0.0]);
    assert_eq!(states.unwrap(), DMatrix::from_element(1, 1, 1.0));
}

#[test]
fn be_aot_compiler_failure_is_typed_after_preparation_invalidation() {
    let mut solver = BE::new();
    solver
        .try_set_initial(
            vec![Expr::parse_expression(
                "-3.141592653589793*y+0.2718281828459045",
            )],
            vec!["y".to_string()],
            "t".to_string(),
            1e-10,
            20,
            Some(0.1),
            0.0,
            1.0,
            DVector::from_vec(vec![1.0]),
        )
        .unwrap();
    solver.try_solve().unwrap();
    assert_eq!(solver.get_status(), "finished");

    let output_parent = tempfile::tempdir().unwrap();
    let missing_compiler = output_parent.path().join("missing-c-compiler.exe");
    let config = SymbolicIvpGeneratedBackendConfig::new()
        .with_output_parent_dir(Some(output_parent.path().to_path_buf()))
        .with_build_policy(
            crate::symbolic::symbolic_ivp_generated::SymbolicIvpAotBuildPolicy::BuildIfMissing {
                profile: crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile::Debug,
            },
        )
        .with_c_tcc()
        .with_aot_c_compiler(missing_compiler.to_string_lossy().into_owned());
    solver.set_generated_backend_config(config);

    assert!(matches!(
        solver.try_solve(),
        Err(BeError::GeneratedBackend(_))
    ));
    assert_eq!(solver.get_status(), "failed");
    let (times, states) = solver.get_result();
    assert_eq!(times.unwrap().as_slice(), &[0.0]);
    assert_eq!(states.unwrap(), DMatrix::from_element(1, 1, 1.0));
}

#[test]
fn be_newton_failure_is_typed_and_preserves_initial_trajectory() {
    let mut solver = one_state_be(1.0, Some(0.1));
    solver
        .try_set_telemetry_mode(BeTelemetryMode::Counters)
        .unwrap();
    solver.set_native_ode_callbacks(constant_rhs, Some(singular_newton_jac));

    let error = solver.try_solve().unwrap_err();
    assert!(matches!(
        error,
        BeError::Newton(NreError::SingularNewtonMatrix)
    ));
    assert_eq!(solver.get_status(), "failed");
    let (times, states) = solver.get_result();
    assert_eq!(times.unwrap().as_slice(), &[0.0]);
    assert_eq!(states.unwrap(), DMatrix::from_element(1, 1, 1.0));
    let detailed = solver.detailed_statistics();
    assert_eq!(detailed.accepted_steps, 0);
    assert_eq!(detailed.failed_steps, 1);
    assert_eq!(detailed.factorization_calls, 1);
    assert_eq!(detailed.linear_solve_calls, 1);
    assert_eq!(detailed.newton_failures, 1);
    assert_eq!(detailed.step_limit_failures, 0);
}

#[test]
fn be_failure_after_an_accepted_step_preserves_last_accepted_sample() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let mut solver = one_state_be(0.3, Some(0.1));
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_rhs = Arc::clone(&calls);
    solver.set_native_ode_callbacks(
        move |_, y| {
            if calls_for_rhs.fetch_add(1, Ordering::Relaxed) >= 2 {
                DVector::from_element(y.len(), f64::NAN)
            } else {
                DVector::from_element(y.len(), 1.0)
            }
        },
        Some(zero_scalar_jac),
    );

    assert!(matches!(
        solver.try_solve(),
        Err(BeError::Newton(NreError::NonFiniteCallback {
            stage: "residual"
        }))
    ));
    assert_eq!(solver.get_status(), "failed");
    assert_eq!(solver.t, 0.1);
    assert_eq!(solver.y.as_slice(), &[1.1]);
    let (times, states) = solver.get_result();
    assert_eq!(times.unwrap().as_slice(), &[0.0, 0.1]);
    assert_eq!(states.unwrap(), DMatrix::from_row_slice(2, 1, &[1.0, 1.1]));
}

#[test]
fn be_lambdify_non_finite_callback_preserves_last_accepted_sample() {
    let output_parent = tempfile::tempdir().unwrap();
    let mut solver = BE::new();
    solver
        .try_set_initial(
            vec![Expr::parse_expression("1 / (0.2 - t)")],
            vec!["y".to_string()],
            "t".to_string(),
            1e-12,
            20,
            Some(0.1),
            0.0,
            0.3,
            DVector::from_vec(vec![1.0]),
        )
        .unwrap();
    solver.set_generated_backend_config(
        SymbolicIvpGeneratedBackendConfig::new()
            .with_resolver(Some(
                crate::symbolic::codegen::codegen_aot_resolution::AotResolver::new(
                    crate::symbolic::codegen::codegen_aot_registry::AotRegistry::new(),
                ),
            ))
            .with_output_parent_dir(Some(output_parent.path().to_path_buf())),
    );
    solver
        .try_set_telemetry_mode(BeTelemetryMode::Counters)
        .unwrap();

    assert!(matches!(
        solver.try_solve(),
        Err(BeError::Newton(NreError::NonFiniteCallback {
            stage: "residual"
        }))
    ));
    assert_eq!(solver.status(), BeStatus::Failed);
    assert_eq!(solver.t, 0.1);
    assert_eq!(solver.get_result().0.unwrap().as_slice(), &[0.0, 0.1]);
    assert_eq!(solver.detailed_statistics().newton_failures, 1);
    assert_eq!(solver.detailed_statistics().failed_steps, 1);
}

#[test]
fn be_rust_aot_non_finite_callback_preserves_last_accepted_sample() {
    let output_parent = tempfile::tempdir().unwrap();
    let mut solver = BE::new();
    solver
        .try_set_initial(
            vec![Expr::parse_expression("1 / (0.2 - t)")],
            vec!["y".to_string()],
            "t".to_string(),
            1e-12,
            20,
            Some(0.1),
            0.0,
            0.3,
            DVector::from_vec(vec![1.0]),
        )
        .unwrap();
    let config = SymbolicIvpGeneratedBackendConfig::new()
        .with_resolver(Some(crate::symbolic::codegen::codegen_aot_resolution::AotResolver::new(
            crate::symbolic::codegen::codegen_aot_registry::AotRegistry::new(),
        )))
        .with_output_parent_dir(Some(output_parent.path().to_path_buf()))
        .with_rust()
        .with_build_policy(
            crate::symbolic::symbolic_ivp_generated::SymbolicIvpAotBuildPolicy::BuildIfMissing {
                profile: crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile::Debug,
            },
        );
    solver.set_generated_backend_config(config);
    solver
        .try_set_telemetry_mode(BeTelemetryMode::Counters)
        .unwrap();

    assert!(matches!(
        solver.try_solve(),
        Err(BeError::Newton(NreError::NonFiniteCallback {
            stage: "residual"
        }))
    ));
    assert_eq!(solver.status(), BeStatus::Failed);
    assert_eq!(solver.t, 0.1);
    assert_eq!(solver.get_result().0.unwrap().as_slice(), &[0.0, 0.1]);
    assert_eq!(solver.detailed_statistics().newton_failures, 1);
    assert_eq!(solver.detailed_statistics().failed_steps, 1);

    let resolver = solver
        .generated_backend_config()
        .resolver
        .as_ref()
        .expect("successful AOT build must publish an updated resolver");
    let keys = resolver.registry().problem_keys();
    assert_eq!(keys.len(), 1);
    let artifact = resolver
        .registry()
        .get_by_problem_key(&keys[0])
        .expect("built AOT problem should be registered");
    assert!(artifact.compiled_artifact_exists());
}

#[test]
fn be_fixed_step_has_first_order_decay_convergence() {
    let exact = (-1.0_f64).exp();
    let final_value = |step| {
        let mut solver = one_state_be(1.0, Some(step));
        solver.set_native_ode_callbacks(decay_rhs, Some(decay_jac));
        solver.try_solve().unwrap();
        solver.y[0]
    };

    let coarse_error = (final_value(0.2) - exact).abs();
    let fine_error = (final_value(0.1) - exact).abs();
    let finer_error = (final_value(0.05) - exact).abs();
    assert!(coarse_error > fine_error && fine_error > finer_error);
    assert!((coarse_error / fine_error - 2.0).abs() < 0.2);
    assert!((fine_error / finer_error - 2.0).abs() < 0.2);
}

#[test]
fn be_implicit_step_evaluates_nonautonomous_rhs_at_new_time() {
    let mut solver = one_state_be(1.0, Some(0.25));
    solver.set_native_ode_callbacks(nonautonomous_rhs, Some(zero_scalar_jac));
    solver.try_solve().unwrap();

    assert!(
        (solver.y[0] - 1.625).abs() < 1e-12,
        "implicit nonautonomous solution was {} instead of the right-endpoint sum",
        solver.y[0]
    );
}

#[test]
fn be_automatic_step_heuristic_and_newton_use_distinct_times() {
    use std::sync::Mutex;

    let rhs_times = Arc::new(Mutex::new(Vec::new()));
    let jacobian_times = Arc::new(Mutex::new(Vec::new()));
    let rhs_times_callback = Arc::clone(&rhs_times);
    let jacobian_times_callback = Arc::clone(&jacobian_times);
    let mut solver = one_state_be(1.0, None);
    solver.set_native_ode_callbacks(
        move |t: f64, _: &DVector<f64>| {
            rhs_times_callback.lock().unwrap().push(t);
            DVector::from_element(1, t)
        },
        Some(move |t: f64, _: &DVector<f64>| {
            jacobian_times_callback.lock().unwrap().push(t);
            DMatrix::zeros(1, 1)
        }),
    );

    solver.try_solve().unwrap();

    assert_eq!(rhs_times.lock().unwrap().as_slice(), &[0.0, 1.0, 1.0]);
    assert_eq!(jacobian_times.lock().unwrap().as_slice(), &[0.0, 1.0, 1.0]);
    assert_eq!(solver.get_statistics().residual_calls, 3);
    assert_eq!(solver.get_statistics().jacobian_calls, 3);
    assert!((solver.y[0] - 2.0).abs() < 1e-12);
}

#[test]
fn be_fallible_step_and_loop_preserve_typed_callback_errors() {
    let make_solver = || {
        let mut solver = one_state_be(0.2, Some(0.1));
        solver.set_native_ode_callbacks(
            |_: f64, y: &DVector<f64>| DVector::from_element(y.len(), f64::NAN),
            Some(|_: f64, _: &DVector<f64>| DMatrix::from_element(1, 1, -1.0)),
        );
        solver
    };

    let mut one_step = make_solver();
    assert!(matches!(
        one_step.try_step(),
        Err(BeError::Newton(NreError::NonFiniteCallback {
            stage: "residual"
        }))
    ));
    assert_eq!(one_step.t, 0.0);
    assert_eq!(one_step.status, BeStatus::Failed);
    assert_eq!(one_step.detailed_statistics().newton_failures, 1);

    let mut loop_solver = make_solver();
    assert!(matches!(
        loop_solver.try_main_loop(),
        Err(BeError::Newton(NreError::NonFiniteCallback {
            stage: "residual"
        }))
    ));
    assert_eq!(loop_solver.t, 0.0);
    assert_eq!(loop_solver.status, BeStatus::Failed);
    assert_eq!(loop_solver.detailed_statistics().newton_failures, 1);
}

#[test]
fn be_rejects_invalid_configuration_without_panicking() {
    assert!(matches!(
        BE::new().try_check(),
        Err(BeError::InvalidConfiguration(_))
    ));

    let result = BE::new().try_set_initial(
        vec![Expr::parse_expression("-y")],
        vec!["y".to_string()],
        "t".to_string(),
        f64::NAN,
        10,
        Some(0.1),
        0.0,
        1.0,
        DVector::from_vec(vec![1.0]),
    );
    assert!(matches!(result, Err(BeError::InvalidConfiguration(_))));

    let valid = one_state_be(0.5, Some(0.1));
    assert!(valid.try_check().is_ok());
}

#[test]
fn be_zero_length_interval_returns_only_the_initial_sample() {
    let mut solver = one_state_be(0.0, Some(0.1));
    solver.set_native_ode_callbacks(decay_rhs, Some(decay_jac));
    solver.try_solve().unwrap();

    assert_eq!(solver.get_status(), "finished");
    let (times, states) = solver.get_result();
    assert_eq!(times.unwrap().as_slice(), &[0.0]);
    assert_eq!(states.unwrap(), DMatrix::from_element(1, 1, 1.0));
    assert_eq!(solver.statistics.step_calls, 0);
}

#[test]
fn be_rejects_backward_time_interval() {
    let result = BE::new().try_set_initial(
        vec![Expr::parse_expression("-y")],
        vec!["y".to_string()],
        "t".to_string(),
        1e-8,
        10,
        Some(0.1),
        1.0,
        0.0,
        DVector::from_vec(vec![1.0]),
    );
    assert!(matches!(result, Err(BeError::InvalidConfiguration(_))));
}

#[test]
fn be_reports_positive_step_that_cannot_advance_floating_time() {
    let t0: f64 = 1.0e300;
    let t_bound = f64::from_bits(t0.to_bits() + 1);
    let mut solver = BE::new();
    solver
        .try_set_initial(
            vec![Expr::parse_expression("-y")],
            vec!["y".to_string()],
            "t".to_string(),
            1e-8,
            10,
            Some(f64::MIN_POSITIVE),
            t0,
            t_bound,
            DVector::from_vec(vec![1.0]),
        )
        .unwrap();
    solver.set_native_ode_callbacks(decay_rhs, Some(decay_jac));

    assert!(matches!(
        solver.try_solve(),
        Err(BeError::StepUnderflow { time, step })
            if time == t0 && step == f64::MIN_POSITIVE
    ));
    assert_eq!(solver.t, t0);
    assert_eq!(solver.y.as_slice(), &[1.0]);
}

#[test]
fn be_stop_conditions_are_validated_and_reset_on_reinitialization() {
    let mut solver = one_state_be(1.0, Some(0.1));
    let unknown = HashMap::from([("missing".to_string(), 0.0)]);
    assert!(matches!(
        solver.try_set_stop_condition(unknown),
        Err(BeError::InvalidConfiguration(_))
    ));
    let non_finite = HashMap::from([("y".to_string(), f64::NAN)]);
    assert!(matches!(
        solver.try_set_stop_condition(non_finite),
        Err(BeError::InvalidConfiguration(_))
    ));

    solver
        .try_set_stop_condition(HashMap::from([("y".to_string(), 0.5)]))
        .unwrap();
    assert!(solver
        .try_set_stop_condition(HashMap::from([("missing".to_string(), 1.0)]))
        .is_err());
    assert_eq!(solver.stop_conditions, vec![(0, 0.5)]);
    solver
        .try_set_initial(
            vec![Expr::parse_expression("-y")],
            vec!["y".to_string()],
            "t".to_string(),
            1e-10,
            20,
            Some(0.1),
            0.0,
            1.0,
            DVector::from_vec(vec![1.0]),
        )
        .unwrap();
    assert!(solver.stop_conditions.is_empty());
}

#[test]
fn be_reinitialization_is_transactional_and_resets_runtime_state() {
    let mut solver = one_state_be(0.2, Some(0.1));
    solver.try_set_max_steps(17).unwrap();
    solver
        .try_set_telemetry_mode(BeTelemetryMode::Counters)
        .unwrap();
    solver
        .try_set_stop_condition(HashMap::from([("y".to_string(), 100.0)]))
        .unwrap();
    solver.set_native_ode_callbacks(constant_rhs, Some(zero_scalar_jac));
    solver.try_solve().unwrap();

    let accepted_times = solver.get_result().0.unwrap();
    let accepted_states = solver.get_result().1.unwrap();
    let accepted_statistics = solver.get_statistics();
    assert_eq!(solver.status(), BeStatus::Finished);
    assert!(accepted_statistics.step_calls > 0);

    let invalid = solver.try_set_initial(
        vec![Expr::parse_expression("-y")],
        vec!["y".to_string()],
        "t".to_string(),
        1e-10,
        20,
        Some(0.0),
        0.0,
        1.0,
        DVector::from_vec(vec![9.0]),
    );
    assert!(matches!(invalid, Err(BeError::InvalidConfiguration(_))));
    assert_eq!(solver.get_result().0.unwrap(), accepted_times);
    assert_eq!(solver.get_result().1.unwrap(), accepted_states);
    let stats_after_rejected_reconfigure = solver.get_statistics();
    assert_eq!(
        stats_after_rejected_reconfigure.step_calls,
        accepted_statistics.step_calls
    );
    assert_eq!(
        stats_after_rejected_reconfigure.residual_calls,
        accepted_statistics.residual_calls
    );
    assert!(solver.native_rhs.is_some());
    assert_eq!(solver.stop_conditions, vec![(0, 100.0)]);

    solver
        .try_set_initial(
            vec![Expr::parse_expression("-2*y")],
            vec!["y".to_string()],
            "t".to_string(),
            1e-10,
            20,
            Some(0.1),
            5.0,
            5.2,
            DVector::from_vec(vec![2.0]),
        )
        .unwrap();

    assert_eq!(solver.status(), BeStatus::Running);
    assert_eq!(solver.t, 5.0);
    assert_eq!(solver.y.as_slice(), &[2.0]);
    assert_eq!(solver.get_result().0.unwrap().as_slice(), &[5.0]);
    assert_eq!(
        solver.get_result().1.unwrap(),
        DMatrix::from_element(1, 1, 2.0)
    );
    assert!(solver.native_rhs.is_none());
    assert!(solver.newton.jac.is_none());
    assert!(solver.stop_conditions.is_empty());
    let reset_statistics = solver.get_statistics();
    assert_eq!(reset_statistics.step_calls, 0);
    assert_eq!(reset_statistics.residual_calls, 0);
    assert_eq!(reset_statistics.jacobian_calls, 0);
    assert_eq!(
        solver.detailed_statistics(),
        BeDetailedStatistics::default()
    );
    assert_eq!(
        solver.continuation_statistics(),
        &BeContinuationStatistics::default()
    );
    assert_eq!(solver.telemetry_mode(), BeTelemetryMode::Counters);
    assert_eq!(solver.max_steps(), 17);

    solver.try_solve().unwrap();
    assert!((solver.y[0] - 2.0 / 1.44).abs() < 1e-12);
}

#[test]
fn be_initially_satisfied_stop_condition_returns_initial_sample() {
    let mut solver = one_state_be(1.0, Some(0.1));
    solver
        .try_set_stop_condition(HashMap::from([("y".to_string(), 1.0)]))
        .unwrap();
    solver.try_solve().unwrap();

    assert_eq!(solver.get_status(), "stopped_by_condition");
    let (times, states) = solver.get_result();
    assert_eq!(times.unwrap().as_slice(), &[0.0]);
    assert_eq!(states.unwrap(), DMatrix::from_element(1, 1, 1.0));
    assert_eq!(solver.statistics.step_calls, 0);
}

#[test]
fn be_stop_condition_uses_accepted_samples_without_event_localization() {
    let mut solver = one_state_be(1.0, Some(0.3));
    solver.set_native_ode_callbacks(constant_rhs, Some(zero_scalar_jac));
    solver
        .try_set_stop_condition(HashMap::from([("y".to_string(), 1.5)]))
        .unwrap();
    solver.try_set_neighborhood_check(0.11).unwrap();

    solver.try_solve().unwrap();

    assert_eq!(solver.get_status(), "stopped_by_condition");
    let (times, states) = solver.get_result();
    assert_eq!(times.unwrap().as_slice(), &[0.0, 0.3, 0.6]);
    assert_eq!(
        states.unwrap(),
        DMatrix::from_row_slice(3, 1, &[1.0, 1.3, 1.6])
    );
}

#[test]
fn be_neighborhood_tolerance_rejects_invalid_values() {
    let mut solver = one_state_be(1.0, Some(0.1));
    assert!(solver.try_set_neighborhood_check(f64::NAN).is_err());
    assert!(solver.try_set_neighborhood_check(0.0).is_err());
    assert_eq!(solver.neighborhood_check, 1e-6);
}

#[test]
fn be_parameter_schema_rejects_state_and_time_collisions() {
    let mut solver = one_state_be(1.0, Some(0.1));
    solver.try_set_equation_parameters(Some(&["rate"])).unwrap();
    assert!(solver.try_set_equation_parameters(Some(&["y"])).is_err());
    assert!(solver.try_set_equation_parameters(Some(&["t"])).is_err());
    assert_eq!(
        solver.newton.equation_parameters.as_deref(),
        Some(&["rate".to_string()][..])
    );
}

#[test]
fn be_new_with_options_installs_generated_backend_mode() {
    use crate::symbolic::symbolic_ivp_generated::SymbolicIvpAotBuildPolicy;

    let solver = BE::new_with_options(
        BeSolverOptions::new(
            vec![Expr::parse_expression("y")],
            vec!["y".to_string()],
            "t".to_string(),
            1e-6,
            20,
            Some(0.1),
            0.0,
            1.0,
            DVector::from_vec(vec![1.0]),
        )
        .with_dense_generated_backend_mode(DenseIvpGeneratedBackendMode::RequirePrebuilt),
    );

    assert_eq!(
        solver.generated_backend_config().build_policy,
        SymbolicIvpAotBuildPolicy::RequirePrebuilt
    );
    assert_eq!(solver.newton.values, vec!["y".to_string()]);
}

#[test]
fn generated_backend_surface_mode_updates_be_config() {
    use crate::symbolic::symbolic_ivp_generated::SymbolicIvpAotBuildPolicy;

    let solver = BE::new()
        .with_dense_generated_backend_mode(DenseIvpGeneratedBackendMode::BuildIfMissingRelease);

    assert_eq!(
        solver.generated_backend_config().build_policy,
        SymbolicIvpAotBuildPolicy::BuildIfMissing {
            profile:
                crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile::Release
        }
    );
}

#[test]
fn be_generated_backend_surface_keeps_selected_c_backend() {
    let solver = BE::new()
        .with_dense_generated_backend_c_tcc("target/generated-ivp-tests")
        .with_dense_generated_backend_mode(DenseIvpGeneratedBackendMode::BuildIfMissingRelease);

    assert_eq!(
        solver.generated_backend_config().aot_codegen_backend,
        crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend::C
    );
    assert_eq!(
        solver.generated_backend_config().aot_c_compiler.as_deref(),
        Some("tcc")
    );
}

#[test]
fn be_generated_backend_repeated_solves_alias_prefers_c_gcc() {
    let solver =
        BE::new().with_dense_generated_backend_for_repeated_solves("target/generated-ivp-tests");

    assert_eq!(
        solver.generated_backend_config().aot_codegen_backend,
        crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend::C
    );
    assert_eq!(
        solver.generated_backend_config().aot_c_compiler.as_deref(),
        Some("gcc")
    );
}

#[test]
fn test_newton_raphson_solver_for_Euler_1() {
    let eq1 = Expr::parse_expression("z+y-10.0*x");
    let eq2 = Expr::parse_expression("z*y-4.0*x");
    let eq_system = vec![eq1, eq2];
    info!("eq_system = {:?}", eq_system);
    let y0 = DVector::from_vec(vec![1.0, 1.0]);
    let values = vec!["z".to_string(), "y".to_string()];
    let arg = "x".to_string();
    let tolerance = 1e-2;
    let max_iterations = 50;

    let h = Some(1e-2);
    let t0 = 0.0;
    let t_bound = 1.0;

    let mut solver = BE::new();

    solver.set_initial(
        eq_system,
        values,
        arg,
        tolerance,
        max_iterations,
        h,
        t0,
        t_bound,
        y0,
    );
    info!(
        "y = {:?}, initial_guess = {:?}",
        solver.newton.y, solver.newton.initial_guess
    );
    solver.newton.eq_generate();
    let (success, message) = solver._step_impl();
    assert_eq!(solver.y.len(), 2);
    assert_eq!(success, true, "success = {} must be true", success);
    assert_eq!(message, None, "message = {:?} must be None", message);
}

#[test]

fn test_newton_raphson_solver_for_Euler_2() {
    let eq1 = Expr::parse_expression("z+y-10.0*x");
    let eq2 = Expr::parse_expression("z*y-4.0*x");
    let eq_system = vec![eq1, eq2];
    info!("eq_system = {:?}", eq_system);
    let y0 = DVector::from_vec(vec![1.0, 1.0]);
    let values = vec!["z".to_string(), "y".to_string()];
    let arg = "x".to_string();
    let tolerance = 1e-2;
    let max_iterations = 50;

    let h = Some(1e-2);
    let t0 = 0.0;
    let t_bound = 1.0;

    let mut solver = BE::new();

    solver.set_initial(
        eq_system,
        values,
        arg,
        tolerance,
        max_iterations,
        h,
        t0,
        t_bound,
        y0,
    );
    info!(
        "y = {:?}, initial_guess = {:?}",
        solver.newton.y, solver.newton.initial_guess
    );
    solver.newton.eq_generate();
    solver.step();
    assert_eq!(solver.status, BeStatus::Running);
}

#[test]

fn test_newton_raphson_solver_for_Euler_3() {
    let eq1 = Expr::parse_expression("z+y-10.0*x");
    let eq2 = Expr::parse_expression("z*y-4.0*x");
    let eq_system = vec![eq1, eq2];
    info!("eq_system = {:?}", eq_system);
    let y0 = DVector::from_vec(vec![1.0, 1.0]);
    let values = vec!["z".to_string(), "y".to_string()];
    let arg = "x".to_string();
    let tolerance = 1e-2;
    let max_iterations = 50;

    let h = Some(1e-2);
    let t0 = 0.0;
    let t_bound = 1.0;

    let mut solver = BE::new();

    solver.set_initial(
        eq_system,
        values,
        arg,
        tolerance,
        max_iterations,
        h,
        t0,
        t_bound,
        y0,
    );
    info!(
        "y = {:?}, initial_guess = {:?}",
        solver.newton.y, solver.newton.initial_guess
    );
    solver.newton.eq_generate();
    solver.solve();
    assert_eq!(solver.status, BeStatus::Finished);
}

#[test]
fn test_newton_raphson_solver_for_Euler_4() {
    let eq1 = Expr::parse_expression("z+y-10.0*x");
    let eq2 = Expr::parse_expression("z*y-4.0*x");
    let eq_system = vec![eq1, eq2];
    info!("eq_system = {:?}", eq_system);
    let y0 = DVector::from_vec(vec![1.0, 1.0]);
    let values = vec!["z".to_string(), "y".to_string()];
    let arg = "x".to_string();
    let tolerance = 1e-2;
    let max_iterations = 50;

    let h = None;
    let t0 = 0.0;
    let t_bound = 1.0;

    let mut solver = BE::new();

    solver.set_initial(
        eq_system,
        values,
        arg,
        tolerance,
        max_iterations,
        h,
        t0,
        t_bound,
        y0,
    );
    info!(
        "y = {:?}, initial_guess = {:?}",
        solver.newton.y, solver.newton.initial_guess
    );
    solver.newton.eq_generate();
    solver.solve();
    let res = solver.get_result();
    let _result = res.1.unwrap();
    // assert_eq!(result.shape(), (2, 1)) ;
    assert_eq!(solver.status, BeStatus::Finished);
}

#[test]
fn test_be_stop_condition_single_variable() {
    // Test: y' = y, y(0) = 1, stop when y reaches 2.0
    let eq1 = Expr::parse_expression("-z+2.0*x"); // z' = -z + 2*x, solution grows
    let eq_system = vec![eq1];
    let y0 = DVector::from_vec(vec![1.0]);
    let values = vec!["z".to_string()];
    let arg = "x".to_string();
    let tolerance = 1e-6;
    let max_iterations = 50;
    let h = Some(0.01);
    let t0 = 0.0;
    let t_bound = 10.0; // Large bound to ensure stop condition triggers first

    let mut solver = BE::new();
    solver.set_initial(
        eq_system,
        values,
        arg,
        tolerance,
        max_iterations,
        h,
        t0,
        t_bound,
        y0,
    );

    let mut stop_condition = HashMap::new();
    stop_condition.insert("z".to_string(), 1.5);
    solver.set_stop_condition(stop_condition);
    solver.set_neighborhood_check(1e-2);

    solver.solve();

    assert_eq!(solver.get_status(), "stopped_by_condition");
    let (_, y_result) = solver.get_result();
    let y_res = y_result.unwrap();
    let final_y = y_res[(y_res.nrows() - 1, 0)];
    assert!((final_y - 1.5).abs() <= 1e-2);
}

#[test]
fn test_be_stop_condition_multiple_variables() {
    // Test system with multiple variables
    let eq1 = Expr::parse_expression("z+y-2.0*x");
    let eq2 = Expr::parse_expression("-z*y+3.0*x");
    let eq_system = vec![eq1, eq2];
    let y0 = DVector::from_vec(vec![1.0, 1.0]);
    let values = vec!["z".to_string(), "y".to_string()];
    let arg = "x".to_string();
    let tolerance = 1e-6;
    let max_iterations = 50;
    let h = Some(0.01);
    let t0 = 0.0;
    let t_bound = 10.0;

    let mut solver = BE::new();
    solver.set_initial(
        eq_system,
        values,
        arg,
        tolerance,
        max_iterations,
        h,
        t0,
        t_bound,
        y0,
    );

    let mut stop_condition = HashMap::new();
    stop_condition.insert("z".to_string(), 1.2);
    solver.set_stop_condition(stop_condition);
    solver.set_neighborhood_check(1e-2);

    solver.solve();

    assert_eq!(solver.get_status(), "stopped_by_condition");
    let (_, y_result) = solver.get_result();
    let y_res = y_result.unwrap();
    let final_z = y_res[(y_res.nrows() - 1, 0)];
    assert!((final_z - 1.2).abs() <= 1e-2);
}

#[test]
fn test_be_no_stop_condition() {
    // Test without stop condition - should run to t_bound
    let eq1 = Expr::parse_expression("z+y-10.0*x");
    let eq2 = Expr::parse_expression("z*y-4.0*x");
    let eq_system = vec![eq1, eq2];
    let y0 = DVector::from_vec(vec![1.0, 1.0]);
    let values = vec!["z".to_string(), "y".to_string()];
    let arg = "x".to_string();
    let tolerance = 1e-2;
    let max_iterations = 50;
    let h = Some(1e-2);
    let t0 = 0.0;
    let t_bound = 0.1;

    let mut solver = BE::new();
    solver.set_initial(
        eq_system,
        values,
        arg,
        tolerance,
        max_iterations,
        h,
        t0,
        t_bound,
        y0,
    );

    solver.solve();

    assert_eq!(solver.get_status(), "finished");
    let (t_result, _) = solver.get_result();
    let t_res = t_result.unwrap();
    let final_t = t_res[t_res.len() - 1];
    assert!((final_t - t_bound).abs() <= h.unwrap());
}
