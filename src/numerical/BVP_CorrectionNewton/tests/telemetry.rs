use super::*;

#[test]
fn callback_failure_retains_partial_report() {
    let mut config = CorrectionNewtonConfig::new(1, 1e-8, 0.0);
    config.telemetry_mode = CorrectionNewtonTelemetryMode::Counters;
    let mut solver = CorrectionNewtonSolver::new(
        DVector::from_element(1, 0.0),
        |_| {
            Err(CorrectionNewtonCallbackError::external(
                std::io::Error::other("fixture callback failure"),
            ))
        },
        |_| Ok(DMatrix::identity(1, 1)),
        config,
    )
    .unwrap();
    match solver.solve() {
        Err(CorrectionNewtonSolverError::ResidualCallback { source, report, .. }) => {
            assert!(source.to_string().contains("fixture callback failure"));
            assert_eq!(report.telemetry.jacobian_evaluations, 1);
            assert_eq!(report.telemetry.factorizations, 1);
            assert_eq!(report.telemetry.residual_evaluations, 1);
        }
        other => panic!("expected residual callback error, got {other:?}"),
    }
}

#[test]
fn typed_domain_failure_at_accepted_state_is_not_retried() {
    let mut config = CorrectionNewtonConfig::new(1, 1e-8, 0.0);
    config.telemetry_mode = CorrectionNewtonTelemetryMode::Counters;
    let mut solver = CorrectionNewtonSolver::new(
        DVector::from_element(1, 0.0),
        |_| {
            Err(CorrectionNewtonCallbackError::domain(
                std::io::Error::other("temperature outside model domain"),
            ))
        },
        |_| Ok(DMatrix::identity(1, 1)),
        config,
    )
    .unwrap();

    match solver.solve() {
        Err(CorrectionNewtonSolverError::ResidualCallback {
            kind,
            source,
            report,
            ..
        }) => {
            assert_eq!(kind, CorrectionNewtonCallbackErrorKind::Domain);
            assert!(
                source
                    .to_string()
                    .contains("temperature outside model domain")
            );
            assert_eq!(report.telemetry.jacobian_evaluations, 1);
            assert_eq!(report.telemetry.residual_evaluations, 1);
            assert_eq!(report.iterations, 1);
        }
        other => panic!("expected typed domain failure, got {other:?}"),
    }
}

#[test]
fn damping_rejects_domain_trials_and_retries_with_smaller_alpha() {
    let run = |telemetry_mode| {
        let mut config = CorrectionNewtonConfig::new(1, 0.2, 0.0);
        config.telemetry_mode = telemetry_mode;
        let mut solver = CorrectionNewtonSolver::new(
            DVector::from_element(1, 0.0),
            |state| {
                if state[0] > 0.06 {
                    Err(CorrectionNewtonCallbackError::domain(
                        std::io::Error::other("trial temperature exceeds model domain"),
                    ))
                } else {
                    Ok(DVector::from_element(1, state[0] - 0.1))
                }
            },
            |_| Ok(DMatrix::identity(1, 1)),
            config,
        )
        .unwrap();
        let report = solver
            .solve()
            .expect("smaller damping trials should return to the model domain");
        (report.state, report.telemetry)
    };

    let (off_state, off_telemetry) = run(CorrectionNewtonTelemetryMode::Off);
    let (state, telemetry) = run(CorrectionNewtonTelemetryMode::Counters);
    assert_eq!(state, off_state);
    assert_eq!(off_telemetry, CorrectionNewtonTelemetrySnapshot::default());
    assert!((state[0] - 0.05).abs() < 1e-12);
    assert_eq!(telemetry.damping_trials, 3);
    assert_eq!(telemetry.damping_rejections, 2);
    assert_eq!(telemetry.domain_rejections, 2);
    assert_eq!(telemetry.residual_evaluations, 4);
    assert_eq!(telemetry.linear_solves, 2);
    assert_eq!(telemetry.accepted_steps, 1);
}

#[test]
fn exhausting_only_domain_invalid_trials_returns_typed_partial_failure() {
    let mut config = CorrectionNewtonConfig::new(1, 0.2, 0.0);
    config.max_damping_trials = 3;
    config.telemetry_mode = CorrectionNewtonTelemetryMode::Counters;
    let mut solver = CorrectionNewtonSolver::new(
        DVector::from_element(1, 0.0),
        |state| {
            if state[0] > 0.0 {
                Err(CorrectionNewtonCallbackError::domain(
                    std::io::Error::other("positive states are outside this fixture domain"),
                ))
            } else {
                Ok(DVector::from_element(1, state[0] - 0.1))
            }
        },
        |_| Ok(DMatrix::identity(1, 1)),
        config,
    )
    .unwrap();

    let error = solver
        .solve()
        .expect_err("all configured damping trials should be outside the domain");
    let report = error
        .partial_report()
        .expect("domain exhaustion should retain the attempted solve report");
    assert!(matches!(
        error,
        CorrectionNewtonSolverError::DampingDomainExhausted {
            source: CorrectionNewtonCallbackError::Domain(_),
            ..
        }
    ));
    assert_eq!(report.state[0], 0.0);
    assert_eq!(report.telemetry.damping_trials, 3);
    assert_eq!(report.telemetry.damping_rejections, 3);
    assert_eq!(report.telemetry.domain_rejections, 3);
    assert_eq!(report.telemetry.residual_evaluations, 4);
    assert_eq!(report.telemetry.linear_solves, 1);
    assert_eq!(report.telemetry.jacobian_evaluations, 1);
    assert!(matches!(
        &error,
        CorrectionNewtonSolverError::DampingDomainExhausted {
            source: CorrectionNewtonCallbackError::Domain(source),
            ..
        } if source.to_string().contains("positive states are outside")
    ));
    assert_eq!(solver.state()[0], 0.0);
}

#[test]
fn external_callback_failure_at_a_damping_trial_is_not_recovered() {
    let mut config = CorrectionNewtonConfig::new(1, 0.2, 0.0);
    config.telemetry_mode = CorrectionNewtonTelemetryMode::Counters;
    let mut solver = CorrectionNewtonSolver::new(
        DVector::from_element(1, 0.0),
        |state| {
            if state[0] > 0.0 {
                Err(CorrectionNewtonCallbackError::external(
                    std::io::Error::other("external callback failure"),
                ))
            } else {
                Ok(DVector::from_element(1, state[0] - 0.1))
            }
        },
        |_| Ok(DMatrix::identity(1, 1)),
        config,
    )
    .unwrap();

    match solver.solve() {
        Err(CorrectionNewtonSolverError::ResidualCallback {
            kind: CorrectionNewtonCallbackErrorKind::External,
            report,
            ..
        }) => {
            assert_eq!(report.telemetry.damping_trials, 1);
            assert_eq!(report.telemetry.domain_rejections, 0);
            assert_eq!(report.telemetry.damping_rejections, 0);
        }
        other => panic!("external trial failure must remain terminal, got {other:?}"),
    }
}

#[test]
fn structural_trial_failures_are_terminal_and_never_count_as_domain_rejections() {
    #[derive(Clone, Copy)]
    enum Failure {
        External,
        Shape,
        NonFinite,
        Symbolic,
    }

    for (failure, expected_kind) in [
        (
            Failure::External,
            CorrectionNewtonCallbackErrorKind::External,
        ),
        (
            Failure::Shape,
            CorrectionNewtonCallbackErrorKind::ShapeMismatch,
        ),
        (
            Failure::NonFinite,
            CorrectionNewtonCallbackErrorKind::NonFinite,
        ),
        (
            Failure::Symbolic,
            CorrectionNewtonCallbackErrorKind::Symbolic,
        ),
    ] {
        let mut config = CorrectionNewtonConfig::new(1, 0.2, 0.0);
        config.telemetry_mode = CorrectionNewtonTelemetryMode::Counters;
        let mut solver = CorrectionNewtonSolver::new(
            DVector::from_element(1, 0.0),
            move |state| {
                if state[0] <= 0.0 {
                    return Ok(DVector::from_element(1, -0.1));
                }
                match failure {
                    Failure::External => Err(CorrectionNewtonCallbackError::external(
                        std::io::Error::other("external trial failure"),
                    )),
                    Failure::Shape => Ok(DVector::zeros(0)),
                    Failure::NonFinite => Ok(DVector::from_element(1, f64::NAN)),
                    Failure::Symbolic => Err(CorrectionNewtonCallbackFailure::Symbolic(
                        crate::numerical::BVP_sci::new::BvpSciNewError::Callback {
                            stage: crate::numerical::BVP_sci::new::BvpSciStage::ResidualCallback,
                            message: "symbolic trial failure".into(),
                        },
                    )),
                }
            },
            |_| Ok(DMatrix::identity(1, 1)),
            config,
        )
        .unwrap();

        match solver.solve() {
            Err(CorrectionNewtonSolverError::ResidualCallback { kind, report, .. }) => {
                assert_eq!(kind, expected_kind);
                assert_eq!(report.telemetry.damping_trials, 1);
                assert_eq!(report.telemetry.domain_rejections, 0);
                assert_eq!(report.telemetry.damping_rejections, 0);
                assert_eq!(report.state[0], 0.0);
            }
            other => panic!("{expected_kind:?} trial failure must remain terminal: {other:?}"),
        }
        assert_eq!(solver.state()[0], 0.0);
    }
}

#[test]
fn callback_output_shape_and_non_finite_failures_have_distinct_kinds() {
    let mut config = CorrectionNewtonConfig::new(1, 1e-8, 0.0);
    config.telemetry_mode = CorrectionNewtonTelemetryMode::Counters;
    let mut bad_jacobian = CorrectionNewtonSolver::new(
        DVector::from_element(1, 0.0),
        |_| Ok(DVector::from_element(1, -1.0)),
        |_| Ok(DMatrix::zeros(1, 2)),
        config.clone(),
    )
    .unwrap();
    assert!(matches!(
        bad_jacobian.solve(),
        Err(CorrectionNewtonSolverError::JacobianCallback {
            kind: CorrectionNewtonCallbackErrorKind::ShapeMismatch,
            ..
        })
    ));

    let mut non_finite_residual = CorrectionNewtonSolver::new(
        DVector::from_element(1, 0.0),
        |_| Ok(DVector::from_element(1, f64::NAN)),
        |_| Ok(DMatrix::identity(1, 1)),
        config,
    )
    .unwrap();
    assert!(matches!(
        non_finite_residual.solve(),
        Err(CorrectionNewtonSolverError::ResidualCallback {
            kind: CorrectionNewtonCallbackErrorKind::NonFinite,
            ..
        })
    ));
}

#[test]
fn symbolic_callback_failure_preserves_typed_source_and_partial_report() {
    let source = crate::numerical::BVP_sci::new::BvpSciNewError::Callback {
        stage: crate::numerical::BVP_sci::new::BvpSciStage::ResidualCallback,
        message: "symbolic fixture failure".into(),
    };
    let pattern = CorrectionNewtonJacobianPattern::dense(1).unwrap();
    let mut config = CorrectionNewtonConfig::new(1, 1e-8, 0.0);
    config.telemetry_mode = CorrectionNewtonTelemetryMode::Counters;
    let mut solver = CorrectionNewtonSolver::new_with_buffered_callbacks(
        DVector::from_element(1, 0.0),
        {
            let source = source.clone();
            move |_, _| Err(CorrectionNewtonCallbackFailure::Symbolic(source.clone()))
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
        Err(CorrectionNewtonSolverError::ResidualCallback {
            source: CorrectionNewtonCallbackError::Symbolic(actual),
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
    let pattern = CorrectionNewtonJacobianPattern::dense(1).unwrap();

    for layout in [
        CorrectionNewtonMatrixLayout::Dense,
        CorrectionNewtonMatrixLayout::Sparse,
        CorrectionNewtonMatrixLayout::BandedAuto,
    ] {
        let mut config = CorrectionNewtonConfig::new(1, 1e-8, 0.0);
        config.matrix_layout = layout;
        config.telemetry_mode = CorrectionNewtonTelemetryMode::Counters;
        let callback_source = source.clone();
        let mut solver = CorrectionNewtonSolver::new_with_buffered_callbacks(
            DVector::from_element(1, 0.0),
            |state, output| {
                output[0] = state[0] - 1.0;
                Ok(())
            },
            pattern.clone(),
            move |_, _| {
                Err(CorrectionNewtonCallbackFailure::Symbolic(
                    callback_source.clone(),
                ))
            },
            config,
        )
        .expect("all supported backends should accept the coordinate callback");

        match solver.solve() {
            Err(CorrectionNewtonSolverError::JacobianCallback {
                source: CorrectionNewtonCallbackError::Symbolic(actual),
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
    solver.config.telemetry_mode = CorrectionNewtonTelemetryMode::Off;
    let report = solver.solve().unwrap();
    assert_eq!(
        report.telemetry,
        CorrectionNewtonTelemetrySnapshot::default()
    );
}

#[test]
fn telemetry_modes_preserve_solution_and_counter_semantics() {
    let mut off = scalar_solver(0.25, 0.5, 3.0);
    off.config.telemetry_mode = CorrectionNewtonTelemetryMode::Off;
    let off_report = off.solve().unwrap();
    assert_eq!(
        off_report.telemetry,
        CorrectionNewtonTelemetrySnapshot::default()
    );

    let mut counters = scalar_solver(0.25, 0.5, 3.0);
    counters.config.telemetry_mode = CorrectionNewtonTelemetryMode::Counters;
    let counter_report = counters.solve().unwrap();

    let mut timings = scalar_solver(0.25, 0.5, 3.0);
    timings.config.telemetry_mode = CorrectionNewtonTelemetryMode::Timings;
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
    assert_eq!(counters_snapshot.residual_callback_time, Duration::ZERO);
    assert_eq!(counters_snapshot.jacobian_callback_time, Duration::ZERO);
    assert_eq!(counters_snapshot.jacobian_assembly_time, Duration::ZERO);
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
        let mut config = CorrectionNewtonConfig::new(1, 1e-8, 0.0);
        config.max_iterations = 1;
        config.telemetry_mode = telemetry_mode;
        let mut solver = CorrectionNewtonSolver::new(
            DVector::from_element(1, 10.0),
            |state| Ok(DVector::from_element(1, state[0] * state[0] - 2.0)),
            |_| Ok(DMatrix::from_element(1, 1, 1.0)),
            config,
        )
        .unwrap();
        match solver.solve() {
            Err(CorrectionNewtonSolverError::NonConvergence { report }) => report,
            other => panic!("expected partial nonconvergence report, got {other:?}"),
        }
    };

    let off = run(CorrectionNewtonTelemetryMode::Off);
    let counters = run(CorrectionNewtonTelemetryMode::Counters);
    let timings = run(CorrectionNewtonTelemetryMode::Timings);
    assert_eq!(off.state, counters.state);
    assert_eq!(counters.state, timings.state);
    assert_eq!(off.iterations, counters.iterations);
    assert_eq!(counters.iterations, timings.iterations);
    assert_eq!(off.telemetry, CorrectionNewtonTelemetrySnapshot::default());

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
    let problem = CorrectionNewtonProblem {
        equations: vec![Expr::parse_expression("0")],
        initial_guess: DMatrix::from_element(1, 3, 1.0),
        state_names: vec!["y".into()],
        independent_variable: "x".into(),
        boundary_conditions: std::collections::HashMap::from([("y".into(), vec![(0, 1.0)])]),
        interval: (0.0, 1.0),
        steps: 3,
    };
    let options = CorrectionNewtonOptions {
        absolute_tolerance: 1e-8,
        relative_tolerances: std::collections::HashMap::from([("y".into(), 1e-8)]),
        bounds: std::collections::HashMap::from([("y".into(), (0.0, 2.0))]),
        telemetry_mode: CorrectionNewtonTelemetryMode::Counters,
        ..CorrectionNewtonOptions::default()
    };
    let mut solver = CorrectionNewtonSolver::from_expr_legacy_dense(problem, options)
        .expect("the selected ExprLegacy/Dense frontend should prepare");
    let report = solver
        .solve()
        .expect("the independent CorrectionNewton core should solve the constant profile");
    assert!(
        report
            .state
            .iter()
            .all(|value| (*value - 1.0).abs() < 1e-12)
    );
    assert_eq!(report.telemetry.jacobian_evaluations, 1);
    assert_eq!(report.telemetry.factorizations, 1);
}
