use super::*;

#[test]
fn callback_failure_retains_partial_report() {
    let mut config = CanteraNewtonConfig::new(1, 1e-8, 0.0);
    config.telemetry_mode = CanteraTelemetryMode::Counters;
    let mut solver = CanteraNewtonSolver::new(
        DVector::from_element(1, 0.0),
        |_| Err("fixture callback failure".into()),
        |_| Ok(DMatrix::identity(1, 1)),
        config,
    )
    .unwrap();
    match solver.solve() {
        Err(CanteraSolverError::ResidualCallback {
            message, report, ..
        }) => {
            assert_eq!(message, "fixture callback failure");
            assert_eq!(report.telemetry.jacobian_evaluations, 1);
            assert_eq!(report.telemetry.factorizations, 1);
            assert_eq!(report.telemetry.residual_evaluations, 1);
        }
        other => panic!("expected residual callback error, got {other:?}"),
    }
}

#[test]
fn symbolic_callback_failure_preserves_typed_source_and_partial_report() {
    let source = crate::numerical::BVP_sci::new::BvpSciNewError::Callback {
        stage: crate::numerical::BVP_sci::new::BvpSciStage::ResidualCallback,
        message: "symbolic fixture failure".into(),
    };
    let pattern = CanteraJacobianPattern::dense(1).unwrap();
    let mut config = CanteraNewtonConfig::new(1, 1e-8, 0.0);
    config.telemetry_mode = CanteraTelemetryMode::Counters;
    let mut solver = CanteraNewtonSolver::new_with_buffered_callbacks(
        DVector::from_element(1, 0.0),
        {
            let source = source.clone();
            move |_, _| Err(CanteraCallbackFailure::Symbolic(source.clone()))
        },
        pattern,
        |_, output| {
            output[0] = 1.0;
            Ok(())
        },
        config,
    )
    .unwrap();

    match solver.solve() {
        Err(CanteraSolverError::ResidualCallback {
            source: Some(actual),
            report,
            ..
        }) => {
            assert_eq!(actual, source);
            assert_eq!(report.telemetry.jacobian_evaluations, 1);
            assert_eq!(report.telemetry.residual_evaluations, 1);
        }
        other => panic!("expected symbolic typed callback failure, got {other:?}"),
    }
}

#[test]
fn symbolic_jacobian_callback_failure_preserves_typed_source_across_backends() {
    let source = BvpSciNewError::Callback {
        stage: crate::numerical::BVP_sci::new::BvpSciStage::JacobianCallback,
        message: "symbolic Jacobian fixture failure".into(),
    };
    let pattern = CanteraJacobianPattern::dense(1).unwrap();

    for layout in [
        CanteraMatrixLayout::Dense,
        CanteraMatrixLayout::Sparse,
        CanteraMatrixLayout::BandedAuto,
    ] {
        let mut config = CanteraNewtonConfig::new(1, 1e-8, 0.0);
        config.matrix_layout = layout;
        config.telemetry_mode = CanteraTelemetryMode::Counters;
        let callback_source = source.clone();
        let mut solver = CanteraNewtonSolver::new_with_buffered_callbacks(
            DVector::from_element(1, 0.0),
            |state, output| {
                output[0] = state[0] - 1.0;
                Ok(())
            },
            pattern.clone(),
            move |_, _| Err(CanteraCallbackFailure::Symbolic(callback_source.clone())),
            config,
        )
        .expect("all supported backends should accept the coordinate callback");

        match solver.solve() {
            Err(CanteraSolverError::JacobianCallback {
                source: Some(actual),
                report,
                ..
            }) => {
                assert_eq!(actual, source, "layout={layout:?}");
                assert_eq!(
                    report.telemetry.residual_evaluations, 0,
                    "layout={layout:?}"
                );
                assert_eq!(
                    report.telemetry.jacobian_evaluations, 1,
                    "layout={layout:?}"
                );
                assert_eq!(report.telemetry.factorizations, 0, "layout={layout:?}");
                assert_eq!(solver.state(), &report.state, "layout={layout:?}");
            }
            other => {
                panic!("expected typed Jacobian callback failure for {layout:?}, got {other:?}")
            }
        }
    }
}

#[test]
fn off_telemetry_does_not_collect_operation_counters() {
    let mut solver = scalar_solver(0.25, 0.5, 3.0);
    assert_eq!(solver.config.logging_level, SolverLogLevel::Off);
    solver.config.telemetry_mode = CanteraTelemetryMode::Off;
    let report = solver.solve().unwrap();
    assert_eq!(report.telemetry, CanteraTelemetrySnapshot::default());
}

#[test]
fn telemetry_modes_preserve_solution_and_counter_semantics() {
    let mut off = scalar_solver(0.25, 0.5, 3.0);
    off.config.telemetry_mode = CanteraTelemetryMode::Off;
    let off_report = off.solve().unwrap();
    assert_eq!(off_report.telemetry, CanteraTelemetrySnapshot::default());

    let mut counters = scalar_solver(0.25, 0.5, 3.0);
    counters.config.telemetry_mode = CanteraTelemetryMode::Counters;
    let counter_report = counters.solve().unwrap();

    let mut timings = scalar_solver(0.25, 0.5, 3.0);
    timings.config.telemetry_mode = CanteraTelemetryMode::Timings;
    let timing_report = timings.solve().unwrap();

    assert_eq!(off_report.state, counter_report.state);
    assert_eq!(counter_report.state, timing_report.state);
    assert_eq!(off_report.iterations, counter_report.iterations);
    assert_eq!(counter_report.iterations, timing_report.iterations);
    let counters_snapshot = &counter_report.telemetry;
    assert_eq!(counters_snapshot.residual_evaluations, 2);
    assert_eq!(counters_snapshot.jacobian_evaluations, 1);
    assert_eq!(counters_snapshot.factorizations, 1);
    assert_eq!(counters_snapshot.linear_solves, 2);
    assert_eq!(counters_snapshot.accepted_steps, 1);
    assert_eq!(
        counters_snapshot.damping_trials,
        timing_report.telemetry.damping_trials
    );
    assert_eq!(
        counters_snapshot.damping_rejections,
        timing_report.telemetry.damping_rejections
    );
    assert_eq!(counters_snapshot.callback_time, Duration::ZERO);
    assert_eq!(counters_snapshot.factorization_time, Duration::ZERO);
    assert_eq!(counters_snapshot.linear_solve_time, Duration::ZERO);
    assert_eq!(
        counters_snapshot.residual_evaluations,
        timing_report.telemetry.residual_evaluations
    );
    assert_eq!(
        counters_snapshot.jacobian_evaluations,
        timing_report.telemetry.jacobian_evaluations
    );
}

#[test]
fn telemetry_modes_preserve_rejected_trial_counts_on_partial_failure() {
    let run = |telemetry_mode| {
        let mut config = CanteraNewtonConfig::new(1, 1e-8, 0.0);
        config.max_iterations = 1;
        config.telemetry_mode = telemetry_mode;
        let mut solver = CanteraNewtonSolver::new(
            DVector::from_element(1, 10.0),
            |state| Ok(DVector::from_element(1, state[0] * state[0] - 2.0)),
            |_| Ok(DMatrix::from_element(1, 1, 1.0)),
            config,
        )
        .unwrap();
        match solver.solve() {
            Err(CanteraSolverError::NonConvergence { report }) => report,
            other => panic!("expected partial nonconvergence report, got {other:?}"),
        }
    };

    let off = run(CanteraTelemetryMode::Off);
    let counters = run(CanteraTelemetryMode::Counters);
    let timings = run(CanteraTelemetryMode::Timings);
    assert_eq!(off.state, counters.state);
    assert_eq!(counters.state, timings.state);
    assert_eq!(off.iterations, counters.iterations);
    assert_eq!(counters.iterations, timings.iterations);
    assert_eq!(off.telemetry, CanteraTelemetrySnapshot::default());

    let counts = &counters.telemetry;
    assert_eq!(counts.jacobian_evaluations, 1);
    assert_eq!(counts.factorizations, 1);
    assert!(counts.damping_trials > 1);
    assert!(counts.damping_rejections > 0);
    assert!(counts.damping_rejections < counts.damping_trials);
    assert_eq!(counts.accepted_steps, 1);
    assert_eq!(counts.residual_evaluations, 1 + counts.damping_trials);
    assert_eq!(counts.linear_solves, 1 + counts.damping_trials);
    assert_eq!(
        counts.damping_rejections,
        timings.telemetry.damping_rejections
    );
    assert_eq!(
        counts.residual_evaluations,
        timings.telemetry.residual_evaluations
    );
    assert_eq!(counts.linear_solves, timings.telemetry.linear_solves);
}

#[test]
fn exprlegacy_lambdify_dense_adapter_runs_the_standalone_core() {
    let problem = CanteraExprBvpProblem {
        equations: vec![Expr::parse_expression("0")],
        initial_guess: DMatrix::from_element(1, 3, 1.0),
        state_names: vec!["y".into()],
        independent_variable: "x".into(),
        boundary_conditions: std::collections::HashMap::from([("y".into(), vec![(0, 1.0)])]),
        interval: (0.0, 1.0),
        steps: 3,
    };
    let options = CanteraExprBvpOptions {
        absolute_tolerance: 1e-8,
        relative_tolerances: std::collections::HashMap::from([("y".into(), 1e-8)]),
        bounds: std::collections::HashMap::from([("y".into(), (0.0, 2.0))]),
        telemetry_mode: CanteraTelemetryMode::Counters,
        ..CanteraExprBvpOptions::default()
    };
    let mut solver = CanteraNewtonSolver::from_expr_legacy_dense(problem, options)
        .expect("the selected ExprLegacy/Dense frontend should prepare");
    let report = solver
        .solve()
        .expect("the independent Cantera core should solve the constant profile");
    assert!(report
        .state
        .iter()
        .all(|value| (*value - 1.0).abs() < 1e-12));
    assert_eq!(report.telemetry.jacobian_evaluations, 1);
    assert_eq!(report.telemetry.factorizations, 1);
}
