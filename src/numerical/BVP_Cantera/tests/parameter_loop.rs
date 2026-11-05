use super::*;
use std::{cell::RefCell, rc::Rc};

type Starts = Rc<RefCell<Vec<(f64, DVector<f64>)>>>;

fn combustion_problem() -> CanteraExprBvpProblem {
    let mut problem = super::large_workloads::combustion_six_state_problem(32);
    problem.equations[1] = Expr::parse_expression("-0.5*q + rate*exp(Teta)*C0");
    problem
}

fn combustion_solver(
    frontend: CanteraSymbolicFrontend,
    layout: CanteraMatrixLayout,
    mode: CanteraTelemetryMode,
) -> CanteraNewtonSolver {
    let mut solver = CanteraNewtonSolver::from_symbolic(
        combustion_problem(),
        CanteraExprBvpOptions {
            symbolic_frontend: frontend,
            matrix_layout: layout,
            absolute_tolerance: 1e-10,
            default_relative_tolerance: 1e-9,
            max_iterations: 100,
            telemetry_mode: mode,
            ..CanteraExprBvpOptions::default()
        },
    )
    .unwrap();
    solver
        .bind_parameters(vec!["rate".into()], vec![0.1])
        .unwrap();
    solver
}

// Preserve the prepared callbacks but inject typed failures at controlled
// residual calls. -1 fails immediately; -2 evaluates rate=0.4 and fails after
// the first trial, exercising rollback of an already accepted Newton step.
fn failing_solver(mode: CanteraTelemetryMode) -> (CanteraNewtonSolver, Starts) {
    let mut solver = combustion_solver(
        CanteraSymbolicFrontend::ExprLegacy,
        CanteraMatrixLayout::BandedAuto,
        mode,
    );
    let starts = Rc::new(RefCell::new(Vec::new()));
    let recorded = starts.clone();
    let ResidualSource::Buffered(mut residual) = solver.residual else {
        panic!("symbolic fixture must use buffered residuals")
    };
    let mut previous = None;
    let mut calls = 0;
    solver.residual = ResidualSource::Buffered(Box::new(move |state, parameters, output| {
        let rate = parameters[0];
        if previous != Some(rate) {
            calls = 0;
            previous = Some(rate);
        }
        if calls == 0 {
            recorded.borrow_mut().push((rate, state.clone()));
        }
        calls += 1;
        if rate == -1.0 || (rate == -2.0 && calls >= 3) {
            return Err(CanteraCallbackFailure::External(
                "injected point failure".into(),
            ));
        }
        residual(state, &[if rate == -2.0 { 0.4 } else { rate }], output)
    }));
    let JacobianSource::SymbolicCoordinates { callback, .. } = &mut solver.jacobian else {
        panic!("symbolic fixture must use coordinate Jacobians")
    };
    let mut jacobian = std::mem::replace(callback, Box::new(|_, _, _| unreachable!()));
    *callback = Box::new(move |state, parameters, output| {
        jacobian(
            state,
            &[if parameters[0] == -2.0 {
                0.4
            } else {
                parameters[0]
            }],
            output,
        )
    });
    (solver, starts)
}

fn success(point: &CanteraParameterLoopPoint) -> (&[f64], &DMatrix<f64>, &CanteraSolveReport) {
    match &point.outcome {
        CanteraParameterLoopOutcome::Converged {
            mesh,
            values,
            report,
        } => (mesh, values, report),
        other => panic!("expected convergence: {other:?}"),
    }
}

fn failure(point: &CanteraParameterLoopPoint) -> &CanteraSolveReport {
    match &point.outcome {
        CanteraParameterLoopOutcome::Failed { error } => {
            assert!(matches!(error, CanteraSolverError::ResidualCallback { .. }));
            error.partial_report().unwrap()
        }
        other => panic!("expected typed failure: {other:?}"),
    }
}

#[test]
fn loop_recovers_failures_before_and_after_success_and_keeps_native_reports() {
    let (mut solver, starts) = failing_solver(CanteraTelemetryMode::Counters);
    // A previous success must not become the first loop guess.
    solver.continue_with_parameter_values(vec![0.15]).unwrap();
    starts.borrow_mut().clear();
    let initial = solver.initial_state.clone();
    let result = solver
        .solve_parameter_loop(&[vec![-1.0], vec![0.1], vec![-2.0], vec![0.2], vec![-1.0]])
        .unwrap();
    assert!(result.stop_error.is_none());
    assert_eq!(result.telemetry_mode, CanteraTelemetryMode::Counters);
    assert_eq!(result.points.len(), 5);
    assert_eq!(
        result
            .points
            .iter()
            .map(|point| point.guess_source_index)
            .collect::<Vec<_>>(),
        [None, None, Some(1), Some(1), Some(3)]
    );
    for (index, point) in result.points.iter().enumerate() {
        assert_eq!(point.index, index);
        assert_eq!(point.parameters, [starts.borrow()[index].0]);
    }
    assert_eq!(failure(&result.points[0]).state, initial);
    let (_, _, first) = success(&result.points[1]);
    let advanced = failure(&result.points[2]);
    assert!(advanced.telemetry.accepted_steps > 0);
    assert_ne!(advanced.state, first.state);
    assert!(advanced.telemetry.residual_evaluations >= 3);
    let (mesh, values, last) = success(&result.points[3]);
    assert_eq!(mesh.len(), 33);
    assert_eq!(values.shape(), (33, 6));
    assert_eq!((mesh[0], mesh[32]), (0.0, 1.0));
    assert_eq!(values, &solver.full_grid_values(&last.state).unwrap());
    assert_eq!(solver.state(), &last.state);
    assert_eq!(solver.external_parameter_values(), [0.2]);
    let starts = starts.borrow();
    assert_eq!(starts[0].1, initial);
    assert_eq!(starts[1].1, initial);
    assert_eq!(starts[2].1, first.state);
    assert_eq!(starts[3].1, first.state);
    assert_eq!(starts[4].1, last.state);
}

#[test]
fn continuation_recovers_raw_failure_and_restores_own_failed_attempt() {
    let (mut solver, starts) = failing_solver(CanteraTelemetryMode::Counters);
    let initial = solver.state().clone();
    let error = solver
        .continue_with_parameter_values(vec![-2.0])
        .unwrap_err();
    assert!(error.partial_report().unwrap().telemetry.accepted_steps > 0);
    assert_eq!(solver.state(), &initial);
    assert_eq!(solver.external_parameter_values(), [0.1]);
    let first = solver.continue_with_parameter_values(vec![0.15]).unwrap();
    solver.set_parameter_values(vec![-2.0]).unwrap();
    let error = solver.solve().unwrap_err();
    assert_ne!(solver.state(), &first.state);
    assert_eq!(solver.state(), &error.partial_report().unwrap().state);
    let error = solver
        .continue_with_parameter_values(vec![-1.0])
        .unwrap_err();
    assert_eq!(error.partial_report().unwrap().state, first.state);
    assert_eq!(solver.state(), &first.state);
    assert_eq!(solver.external_parameter_values(), [0.15]);
    solver.continue_with_parameter_values(vec![0.25]).unwrap();
    assert_eq!(starts.borrow().last().unwrap().1, first.state);
}

#[test]
fn all_failed_loop_restores_exact_entry_and_preserves_prior_success_for_continuation() {
    for prior_success in [false, true] {
        let (mut solver, starts) = failing_solver(CanteraTelemetryMode::Counters);
        let previous =
            prior_success.then(|| solver.continue_with_parameter_values(vec![0.15]).unwrap());
        if prior_success {
            solver.set_parameter_values(vec![-2.0]).unwrap();
            solver.solve().unwrap_err();
        }
        let entry_state = solver.state().clone();
        let entry_parameters = solver.external_parameter_values().to_vec();
        let result = solver
            .try_solve_parameter_loop(&[vec![-1.0], vec![-2.0]])
            .unwrap();
        assert!(result.stop_error.is_none());
        assert_eq!(result.points.len(), 2);
        for point in &result.points {
            failure(point);
            assert_eq!(point.guess_source_index, None);
        }
        assert_eq!(solver.state(), &entry_state);
        assert_eq!(solver.external_parameter_values(), entry_parameters);
        solver.continue_with_parameter_values(vec![0.25]).unwrap();
        let expected = previous
            .map(|report| report.state)
            .unwrap_or_else(|| solver.initial_state.clone());
        assert_eq!(starts.borrow().last().unwrap().1, expected);
    }
}

#[test]
fn entire_loop_is_preflighted_before_mutation_or_callback_work() {
    let (mut solver, starts) = failing_solver(CanteraTelemetryMode::Counters);
    let first = solver.continue_with_parameter_values(vec![0.15]).unwrap();
    starts.borrow_mut().clear();
    for rows in [
        vec![],
        vec![vec![0.2], vec![]],
        vec![vec![0.2], vec![0.3, 0.4]],
        vec![vec![0.2], vec![f64::NAN]],
        vec![vec![0.2], vec![f64::INFINITY]],
        vec![vec![0.2], vec![f64::NEG_INFINITY]],
    ] {
        assert!(matches!(
            solver.solve_parameter_loop(&rows),
            Err(CanteraSolverError::InvalidConfiguration { .. })
        ));
        assert_eq!(solver.state(), &first.state);
        assert_eq!(solver.external_parameter_values(), [0.15]);
        assert!(starts.borrow().is_empty());
    }
    assert!(solver
        .continue_with_parameter_values(vec![f64::NAN])
        .is_err());
    assert!(solver.continue_with_parameter_values(vec![]).is_err());
    assert_eq!(solver.state(), &first.state);
    assert_eq!(solver.external_parameter_values(), [0.15]);
    assert!(starts.borrow().is_empty());
    solver.continue_with_parameter_values(vec![0.25]).unwrap();
    assert_eq!(starts.borrow()[0].1, first.state);
}

#[test]
fn loop_rejects_missing_schema_binding_or_physical_mesh() {
    let mut no_schema = scalar_solver(0.0, 1.0, 1.0);
    assert!(matches!(
        no_schema.solve_parameter_loop(&[vec![]]),
        Err(CanteraSolverError::InvalidConfiguration { .. })
    ));
    let mut unbound =
        CanteraNewtonSolver::from_symbolic(combustion_problem(), CanteraExprBvpOptions::default())
            .unwrap();
    assert!(matches!(
        unbound.solve_parameter_loop(&[vec![0.2]]),
        Err(CanteraSolverError::MissingParameterValues { .. })
    ));
    let mut no_mesh = scalar_solver(0.0, 1.0, 1.0)
        .with_parameter_schema(vec!["rate".into()])
        .unwrap();
    no_mesh
        .bind_parameters(vec!["rate".into()], vec![0.1])
        .unwrap();
    assert!(
        matches!(no_mesh.solve_parameter_loop(&[vec![0.2]]), Err(CanteraSolverError::InvalidConfiguration { message }) if message.contains("physical mesh"))
    );
    assert_eq!(no_mesh.state()[0], 0.0);
    assert_eq!(no_mesh.external_parameter_values(), [0.1]);
    // Single-point continuation does not require a BVP mesh.
    assert_eq!(
        no_mesh
            .continue_with_parameter_values(vec![0.2])
            .unwrap()
            .state[0],
        1.0
    );
}

#[test]
fn off_mode_retains_failure_states_but_not_measurements() {
    let (mut solver, _) = failing_solver(CanteraTelemetryMode::Off);
    let result = solver
        .solve_parameter_loop(&[vec![0.1], vec![-2.0], vec![0.2]])
        .unwrap();
    assert!(result.stop_error.is_none());
    assert_eq!(result.telemetry_mode, CanteraTelemetryMode::Off);
    let first = success(&result.points[0]).2;
    let failed = failure(&result.points[1]);
    let last = success(&result.points[2]).2;
    assert_ne!(failed.state, first.state);
    for report in [first, failed, last] {
        assert_eq!(report.telemetry, CanteraTelemetrySnapshot::default());
        assert!(report.iterations > 0);
    }
    assert_eq!(solver.state(), &last.state);
}

#[test]
fn cyclic_three_parameter_binding_order_survives_set_continue_and_loop() {
    let make = || {
        CanteraNewtonSolver::from_symbolic(
            CanteraExprBvpProblem {
                equations: vec![Expr::parse_expression("-(a + 2*b + 4*c)*y")],
                initial_guess: DMatrix::from_element(1, 16, 1.0),
                state_names: vec!["y".into()],
                independent_variable: "x".into(),
                boundary_conditions: HashMap::from([("y".into(), vec![(0, 1.0)])]),
                interval: (2.0, 3.0),
                steps: 16,
            },
            CanteraExprBvpOptions::default(),
        )
        .unwrap()
    };
    let mut solver = make();
    solver
        .bind_parameters(
            vec!["c".into(), "a".into(), "b".into()],
            vec![0.1, 0.2, 0.3],
        )
        .unwrap();
    assert_eq!(solver.external_parameter_values(), [0.2, 0.3, 0.1]);
    solver.set_parameter_values(vec![0.4, 0.5, 0.6]).unwrap();
    assert_eq!(solver.external_parameter_values(), [0.5, 0.6, 0.4]);
    solver
        .continue_with_parameter_values(vec![0.2, 0.3, 0.4])
        .unwrap();
    assert_eq!(solver.external_parameter_values(), [0.3, 0.4, 0.2]);
    let rows = [vec![0.1, 0.2, 0.3], vec![0.3, 0.4, 0.5]];
    let result = solver.solve_parameter_loop(&rows).unwrap();
    assert!(result.stop_error.is_none());
    for (point, parameters) in result.points.iter().zip(rows) {
        assert_eq!(point.parameters, parameters);
        let mut fresh = make();
        fresh
            .bind_parameters(
                vec!["a".into(), "b".into(), "c".into()],
                vec![parameters[1], parameters[2], parameters[0]],
            )
            .unwrap();
        let reference = fresh.solve().unwrap();
        let (mesh, values, report) = success(point);
        assert_eq!((mesh[0], mesh[16]), (2.0, 3.0));
        assert_eq!(values[(0, 0)], 1.0);
        assert!((&report.state - &reference.state).amax() < 1e-10);
    }
    assert_eq!(solver.external_parameter_values(), [0.4, 0.5, 0.3]);
    // Changing caller order must not change how canonical checkpoint values restore.
    solver
        .bind_parameters(
            vec!["b".into(), "c".into(), "a".into()],
            vec![0.8, 0.7, 0.6],
        )
        .unwrap();
    solver
        .continue_with_parameter_values(vec![0.2, 0.1, 0.3])
        .unwrap();
    assert_eq!(solver.external_parameter_values(), [0.3, 0.2, 0.1]);
}

#[test]
fn nonlinear_combustion_continuation_and_loop_match_fresh_solves() {
    for frontend in [
        CanteraSymbolicFrontend::ExprLegacy,
        CanteraSymbolicFrontend::AtomView,
    ] {
        for layout in [
            CanteraMatrixLayout::Dense,
            CanteraMatrixLayout::Sparse,
            CanteraMatrixLayout::BandedAuto,
        ] {
            let make = || combustion_solver(frontend, layout, CanteraTelemetryMode::Counters);
            let mut solver = make();
            let rows = [vec![0.1], vec![0.2], vec![0.3], vec![0.15]];
            let result = solver.solve_parameter_loop(&rows).unwrap();
            assert!(result.stop_error.is_none());
            let mut continued = make();
            for (point, parameters) in result.points.iter().zip(rows) {
                let mut fresh = make();
                fresh.set_parameter_values(parameters.clone()).unwrap();
                let reference = fresh.solve().unwrap();
                let continuation = continued
                    .continue_with_parameter_values(parameters)
                    .unwrap();
                let (_, values, report) = success(point);
                assert!(report.telemetry.jacobian_evaluations > 0);
                assert!(
                    (&report.state - &reference.state).amax() < 1e-7,
                    "{frontend:?}/{layout:?}"
                );
                assert!((&continuation.state - &reference.state).amax() < 1e-7);
                assert!((values - fresh.full_grid_values(&reference.state).unwrap()).amax() < 1e-7);
                assert_eq!(values[(0, 0)], 0.0);
                assert_eq!(values[(0, 2)], 1.0);
                assert_eq!(values[(32, 1)], 0.0);
                assert!(values.column(0).amax() > 1e-3);
                assert!(report.residual_norm.unwrap() < 1e-8);
            }
        }
    }
}
