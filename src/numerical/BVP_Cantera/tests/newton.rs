use super::*;

#[derive(Debug)]
struct ScalarCanteraTrace {
    residual_states: Vec<f64>,
    jacobian_states: Vec<f64>,
    accepted_states: Vec<f64>,
    accepted_correction_norms: Vec<f64>,
    base_correction_norms: Vec<f64>,
    trial_correction_norms: Vec<f64>,
    jacobian_ages: Vec<usize>,
    damping_factors: Vec<f64>,
    refresh_retries: usize,
    iterations: usize,
    correction_norm: f64,
    converged: bool,
}

fn scalar_cantera_reference<F>(
    initial: f64,
    absolute_tolerance: f64,
    max_jacobian_age: usize,
    max_damping_trials: usize,
    max_iterations: usize,
    residual: F,
) -> ScalarCanteraTrace
where
    F: Fn(f64) -> f64,
{
    let mut trace = ScalarCanteraTrace {
        residual_states: Vec::new(),
        jacobian_states: Vec::new(),
        accepted_states: Vec::new(),
        accepted_correction_norms: Vec::new(),
        base_correction_norms: Vec::new(),
        trial_correction_norms: Vec::new(),
        jacobian_ages: Vec::new(),
        damping_factors: Vec::new(),
        refresh_retries: 0,
        iterations: 0,
        correction_norm: f64::NAN,
        converged: false,
    };
    let mut state = initial;
    let mut cached_jacobian = None;
    let mut jacobian_age = 0;
    let mut force_jacobian = true;
    let mut refresh_retries = 0;

    for iteration in 0..max_iterations {
        trace.iterations = iteration + 1;
        if force_jacobian || jacobian_age > max_jacobian_age {
            trace.jacobian_states.push(state);
            cached_jacobian = Some(1.0);
            jacobian_age = 0;
            force_jacobian = false;
        }
        let jacobian = cached_jacobian.expect("reference Jacobian initialized");
        trace.residual_states.push(state);
        let s0 = (-residual(state) / jacobian).abs() / absolute_tolerance;
        trace.correction_norm = s0;
        trace.base_correction_norms.push(s0);
        jacobian_age += 1;
        trace.jacobian_ages.push(jacobian_age);

        let mut alpha = 1.0;
        let mut accepted = None;
        for _ in 0..max_damping_trials {
            let trial = state + alpha * (-residual(state) / jacobian);
            trace.damping_factors.push(alpha);
            trace.residual_states.push(trial);
            let s1 = (-residual(trial) / jacobian).abs() / absolute_tolerance;
            trace.trial_correction_norms.push(s1);
            if s1 < 1.0 || s1 < s0 {
                accepted = Some((trial, s1));
                break;
            }
            alpha /= 2.0_f64.sqrt();
        }

        if let Some((accepted_state, correction_norm)) = accepted {
            state = accepted_state;
            trace.accepted_states.push(state);
            trace.accepted_correction_norms.push(correction_norm);
            trace.correction_norm = correction_norm;
            if correction_norm <= 1.0 {
                trace.converged = true;
                break;
            }
        } else if jacobian_age > 1 && refresh_retries <= 3 {
            force_jacobian = true;
            refresh_retries += 1;
            trace.refresh_retries += 1;
        } else {
            break;
        }
    }
    trace
}

#[test]
fn fixed_dense_newton_solves_scalar_linear_system_and_reports_work() {
    let mut solver = scalar_solver(0.25, 0.5, 3.0);
    let report = solver.solve().unwrap();
    assert!((report.state[0] - 0.5).abs() < 1e-14);
    assert_eq!(report.iterations, 1);
    assert_eq!(report.telemetry.jacobian_evaluations, 1);
    assert_eq!(report.telemetry.factorizations, 1);
    assert_eq!(report.telemetry.linear_solves, 2);
    assert_eq!(report.telemetry.residual_evaluations, 2);
    assert_eq!(report.telemetry.accepted_steps, 1);
    assert_eq!(report.correction_norm, Some(0.0));
    assert_eq!(report.residual_norm, Some(0.0));
}

#[test]
fn coordinate_jacobian_callback_solves_without_allocating_pattern_per_call() {
    let pattern = CanteraJacobianPattern::try_new(2, vec![(0, 0), (1, 1)])
        .expect("diagonal Jacobian pattern should be valid");
    let mut config = CanteraNewtonConfig::new(2, 1e-8, 0.0);
    config.telemetry_mode = CanteraTelemetryMode::Counters;
    let mut solver = CanteraNewtonSolver::new_with_coordinate_jacobian(
        DVector::from_vec(vec![0.25, -0.5]),
        |state| {
            Ok(DVector::from_vec(vec![
                2.0 * (state[0] - 1.0),
                4.0 * (state[1] + 2.0),
            ]))
        },
        pattern,
        |_, values| {
            values.copy_from_slice(&[2.0, 4.0]);
            Ok(())
        },
        config,
    )
    .expect("coordinate callback should build the core");

    let report = solver
        .solve()
        .expect("coordinate Jacobian solve should converge");
    assert!((report.state[0] - 1.0).abs() < 1e-8);
    assert!((report.state[1] + 2.0).abs() < 1e-8);
    assert_eq!(report.telemetry.jacobian_evaluations, 1);
    assert_eq!(report.telemetry.factorizations, 1);
    assert_eq!(report.telemetry.linear_solves, 2);
    assert_eq!(report.linear_backend, CanteraLinearBackend::Dense);
}

#[test]
fn coordinate_callback_matches_dense_sparse_and_automatic_banded_backends() {
    let coordinates = vec![(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)];
    let matrix_values = [4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
    let expected = [1.0, 2.0, 3.0];
    for (layout, expected_backend) in [
        (CanteraMatrixLayout::Dense, CanteraLinearBackend::Dense),
        (CanteraMatrixLayout::Sparse, CanteraLinearBackend::Sparse),
        (
            CanteraMatrixLayout::BandedAuto,
            CanteraLinearBackend::Banded { lower: 1, upper: 1 },
        ),
    ] {
        let pattern = CanteraJacobianPattern::try_new(3, coordinates.clone())
            .expect("tridiagonal pattern should be valid");
        let mut config = CanteraNewtonConfig::new(3, 1e-8, 0.0);
        config.matrix_layout = layout;
        config.telemetry_mode = CanteraTelemetryMode::Timings;
        let mut solver = CanteraNewtonSolver::new_with_coordinate_jacobian(
            DVector::zeros(3),
            |state| {
                let e0 = state[0] - 1.0;
                let e1 = state[1] - 2.0;
                let e2 = state[2] - 3.0;
                Ok(DVector::from_vec(vec![
                    4.0 * e0 + e1,
                    e0 + 4.0 * e1 + e2,
                    e1 + 4.0 * e2,
                ]))
            },
            pattern,
            move |_, values| {
                values.copy_from_slice(&matrix_values);
                Ok(())
            },
            config,
        )
        .unwrap_or_else(|error| panic!("{layout:?} workspace should prepare: {error}"));

        let report = solver
            .solve()
            .unwrap_or_else(|error| panic!("{layout:?} solve should converge: {error}"));
        for (actual, expected) in report.state.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-8, "layout={layout:?}");
        }
        assert_eq!(report.linear_backend, expected_backend);
        assert_eq!(report.telemetry.jacobian_evaluations, 1);
        assert_eq!(report.telemetry.factorizations, 1);
        assert_eq!(report.telemetry.linear_solves, 2);
        assert!(report.telemetry.jacobian_assembly_time > std::time::Duration::ZERO);
        assert!(report.telemetry.factorization_time > std::time::Duration::ZERO);
    }
}

#[test]
fn expr_legacy_global_coordinate_jacobian_matches_finite_differences() {
    let problem = CanteraExprBvpProblem {
        equations: vec![
            Expr::parse_expression("y + 2*z"),
            Expr::parse_expression("-3*y + z"),
        ],
        initial_guess: DMatrix::from_fn(2, 3, |row, column| {
            0.25 + row as f64 * 0.1 + column as f64 * 0.05
        }),
        state_names: vec!["y".into(), "z".into()],
        independent_variable: "x".into(),
        boundary_conditions: std::collections::HashMap::from([
            ("y".into(), vec![(0, 1.0)]),
            ("z".into(), vec![(1, 0.5)]),
        ]),
        interval: (0.0, 1.0),
        steps: 3,
    };
    for frontend in [
        CanteraSymbolicFrontend::ExprLegacy,
        CanteraSymbolicFrontend::AtomView,
    ] {
        let mut solver = CanteraNewtonSolver::from_symbolic(
            problem.clone(),
            CanteraExprBvpOptions {
                symbolic_frontend: frontend,
                matrix_layout: CanteraMatrixLayout::Sparse,
                ..CanteraExprBvpOptions::default()
            },
        )
        .unwrap_or_else(|error| panic!("{frontend:?} mixed-endpoint preparation failed: {error}"));
        let state = solver.state().clone();
        let mut report = solver.empty_report();
        CanteraNewtonSolver::eval_residual(
            &mut solver.residual,
            &mut solver.residual_scratch,
            &state,
            &[],
            &mut report,
            solver.config.telemetry_mode,
        )
        .expect("base residual should evaluate");
        let baseline = solver.residual_scratch.clone();
        let pattern = solver.jacobian.pattern().unwrap().clone();
        let values = match solver
            .jacobian
            .evaluate(&state, &[])
            .expect("coordinate Jacobian should evaluate")
        {
            EvaluatedJacobian::Coordinates { values } => values.to_vec(),
            EvaluatedJacobian::Dense(_) => panic!("symbolic route must remain coordinate-native"),
        };
        let mut assembled = DMatrix::zeros(state.len(), state.len());
        for (&(row, column), value) in pattern.coordinates().iter().zip(values) {
            assembled[(row, column)] = value;
        }

        let epsilon = 1e-6;
        for column in 0..state.len() {
            let mut perturbed = state.clone();
            perturbed[column] += epsilon;
            CanteraNewtonSolver::eval_residual(
                &mut solver.residual,
                &mut solver.residual_scratch,
                &perturbed,
                &[],
                &mut report,
                solver.config.telemetry_mode,
            )
            .expect("perturbed residual should evaluate");
            for row in 0..state.len() {
                let finite_difference = (solver.residual_scratch[row] - baseline[row]) / epsilon;
                assert!(
                    (assembled[(row, column)] - finite_difference).abs() < 1e-8,
                    "global Jacobian mismatch at ({row}, {column}): analytic={}, finite-difference={finite_difference}",
                    assembled[(row, column)]
                );
            }
        }
    }
}

#[test]
fn dense_jacobian_callback_rejects_structural_backend_selection() {
    let mut config = CanteraNewtonConfig::new(1, 1e-8, 0.0);
    config.matrix_layout = CanteraMatrixLayout::Sparse;
    let result = CanteraNewtonSolver::new(
        DVector::from_element(1, 0.0),
        |_| Ok(DVector::from_element(1, 0.0)),
        |_| Ok(DMatrix::from_element(1, 1, 1.0)),
        config,
    );
    assert!(matches!(
        result,
        Err(CanteraSolverError::InvalidConfiguration { .. })
    ));
}

#[test]
fn coordinate_jacobian_assembly_failure_is_typed_and_keeps_partial_report() {
    let pattern = CanteraJacobianPattern::try_new(1, vec![(0, 0)])
        .expect("scalar Jacobian pattern should be valid");
    let mut config = CanteraNewtonConfig::new(1, 1e-8, 0.0);
    config.telemetry_mode = CanteraTelemetryMode::Counters;
    let mut solver = CanteraNewtonSolver::new_with_coordinate_jacobian(
        DVector::from_element(1, 0.0),
        |_| Ok(DVector::from_element(1, 1.0)),
        pattern,
        |_, values| {
            values[0] = f64::NAN;
            Ok(())
        },
        config,
    )
    .expect("coordinate callback should build the core");

    let error = solver
        .solve()
        .expect_err("non-finite Jacobian value must fail");
    match error {
        CanteraSolverError::JacobianAssembly { source, report } => {
            assert_eq!(
                source,
                CanteraJacobianError::NonFiniteValue { row: 0, column: 0 }
            );
            assert_eq!(report.telemetry.jacobian_evaluations, 1);
            assert_eq!(report.telemetry.factorizations, 0);
        }
        other => panic!("expected typed coordinate assembly error, got {other:?}"),
    }
}

#[test]
fn coordinate_jacobian_pattern_must_match_the_state_dimension() {
    let pattern = CanteraJacobianPattern::try_new(2, vec![(0, 0), (1, 1)])
        .expect("pattern is internally valid");
    let result = CanteraNewtonSolver::new_with_coordinate_jacobian(
        DVector::from_element(1, 0.0),
        |_| Ok(DVector::from_element(1, 0.0)),
        pattern,
        |_, _| Ok(()),
        CanteraNewtonConfig::new(1, 1e-8, 0.0),
    );
    let error = match result {
        Ok(_) => panic!("pattern dimension mismatch must be rejected before solving"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        CanteraSolverError::InvalidConfiguration { .. }
    ));
}

#[test]
fn bound_limited_update_never_leaves_the_configured_interval() {
    let mut config = CanteraNewtonConfig::new(1, 1e-8, 0.0);
    config.bounds[0] = (0.0, 0.5);
    config.telemetry_mode = CanteraTelemetryMode::Counters;
    let mut solver = CanteraNewtonSolver::new(
        DVector::from_element(1, 0.25),
        |state| Ok(DVector::from_element(1, state[0] - 2.0)),
        |_| Ok(DMatrix::from_element(1, 1, 1.0)),
        config,
    )
    .unwrap();
    let _ = solver.solve();
    assert!((solver.state()[0] - 0.5).abs() < 1e-14);
}

#[test]
fn nonconvergence_returns_typed_error_and_partial_telemetry_without_publishing_trial() {
    let mut config = CanteraNewtonConfig::new(1, 1e-12, 0.0);
    config.max_iterations = 1;
    config.max_damping_trials = 1;
    config.telemetry_mode = CanteraTelemetryMode::Counters;
    let mut solver = CanteraNewtonSolver::new(
        DVector::from_element(1, 0.1),
        |_| Ok(DVector::from_element(1, 1.0)),
        |_| Ok(DMatrix::from_element(1, 1, 1.0)),
        config,
    )
    .unwrap();
    let original = solver.state().clone();
    match solver.solve() {
        Err(CanteraSolverError::NonConvergence { report }) => {
            assert_eq!(report.state, original);
            assert_eq!(report.telemetry.residual_evaluations, 2);
            assert_eq!(report.telemetry.linear_solves, 2);
            assert_eq!(report.telemetry.damping_trials, 1);
        }
        other => panic!("expected typed nonconvergence, got {other:?}"),
    }
    assert_eq!(solver.state(), &original);
}

#[test]
fn sparse_inaccurate_singular_solve_is_a_typed_failure_with_partial_report() {
    let pattern = CanteraJacobianPattern::try_new(2, vec![(0, 0), (0, 1), (1, 0), (1, 1)]).unwrap();
    let mut config = CanteraNewtonConfig::new(2, 1e-8, 0.0);
    config.matrix_layout = CanteraMatrixLayout::Sparse;
    config.telemetry_mode = CanteraTelemetryMode::Counters;
    let mut solver = CanteraNewtonSolver::new_with_coordinate_jacobian(
        DVector::zeros(2),
        |_| Ok(DVector::from_vec(vec![1.0, 0.0])),
        pattern,
        |_, values| {
            values.copy_from_slice(&[1.0, 2.0, 2.0, 4.0]);
            Ok(())
        },
        config,
    )
    .unwrap();

    let error = solver
        .solve()
        .expect_err("Sparse must reject a solution that fails the linear residual check");
    match error {
        CanteraSolverError::NonFiniteLinearSolution { backend, report } => {
            assert_eq!(backend, "Sparse");
            assert_eq!(report.telemetry.jacobian_evaluations, 1);
            assert_eq!(report.telemetry.factorizations, 1);
            assert_eq!(report.telemetry.linear_solves, 1);
            assert_eq!(report.iterations, 1);
        }
        CanteraSolverError::LinearResidualCheckFailed {
            backend,
            residual_norm,
            tolerance,
            report,
        } => {
            assert_eq!(backend, "Sparse");
            assert!(residual_norm > tolerance);
            assert_eq!(report.telemetry.jacobian_evaluations, 1);
            assert_eq!(report.telemetry.factorizations, 1);
            assert_eq!(report.telemetry.linear_solves, 1);
            assert_eq!(report.iterations, 0);
        }
        other => panic!("expected typed Sparse residual-check failure, got {other:?}"),
    }
}

#[test]
fn backend_failure_after_accepted_step_preserves_state_and_partial_telemetry() {
    use std::cell::Cell;

    let pattern = CanteraJacobianPattern::dense(1).unwrap();
    for layout in [
        CanteraMatrixLayout::Dense,
        CanteraMatrixLayout::Sparse,
        CanteraMatrixLayout::BandedAuto,
    ] {
        let jacobian_calls = Cell::new(0);
        let mut config = CanteraNewtonConfig::new(1, 1e-8, 0.0);
        config.matrix_layout = layout;
        config.max_jacobian_age = 0;
        config.telemetry_mode = CanteraTelemetryMode::Counters;
        let mut solver = CanteraNewtonSolver::new_with_coordinate_jacobian(
            DVector::from_element(1, 10.0),
            |state| Ok(DVector::from_element(1, state[0] * state[0] - 2.0)),
            pattern.clone(),
            move |_, values| {
                let call = jacobian_calls.get();
                jacobian_calls.set(call + 1);
                values[0] = if call == 0 { 20.0 } else { 0.0 };
                Ok(())
            },
            config,
        )
        .unwrap();

        let error = solver
            .solve()
            .expect_err("the refreshed singular Jacobian must fail after one accepted step");
        let report = error
            .partial_report()
            .expect("runtime backend error must retain its partial report");
        assert_eq!(report.iterations, 2, "layout={layout:?}");
        assert_eq!(report.telemetry.accepted_steps, 1, "layout={layout:?}");
        assert_eq!(
            report.telemetry.jacobian_evaluations, 2,
            "layout={layout:?}"
        );
        let factorization_failed = report.telemetry.factorizations == 1;
        assert_eq!(
            report.telemetry.factorizations,
            if factorization_failed { 1 } else { 2 },
            "layout={layout:?}"
        );
        assert_eq!(
            report.telemetry.linear_solves,
            if factorization_failed { 2 } else { 3 },
            "layout={layout:?}"
        );
        assert!((report.state[0] - 5.1).abs() < 1e-12, "layout={layout:?}");
        assert_eq!(solver.state(), &report.state, "layout={layout:?}");
        assert!(
            matches!(
                error,
                CanteraSolverError::SingularJacobian { .. }
                    | CanteraSolverError::LinearResidualCheckFailed { .. }
                    | CanteraSolverError::NonFiniteLinearSolution { .. }
                    | CanteraSolverError::LinearBackend { .. }
            ),
            "layout={layout:?}: {error:?}"
        );
    }
}

#[test]
fn weighted_rms_uses_component_mean_magnitude_over_the_mesh() {
    let mut config = CanteraNewtonConfig::new(4, 0.1, 0.1);
    config.component_count = 2;
    config.absolute_tolerances = DVector::from_vec(vec![0.1, 0.2]);
    config.relative_tolerances = DVector::from_vec(vec![0.01, 0.1]);
    let solver = CanteraNewtonSolver::new(
        DVector::from_vec(vec![100.0, 1.0, 200.0, 3.0]),
        |_| Ok(DVector::zeros(4)),
        |_| Ok(DMatrix::identity(4, 4)),
        config,
    )
    .unwrap();
    let state = DVector::from_vec(vec![100.0, 1.0, 200.0, 3.0]);
    let correction = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let mut component_sums = vec![0.0; 2];
    let expected = (((1.0_f64 / 1.6).powi(2)
        + (3.0_f64 / 1.6).powi(2)
        + (2.0_f64 / 0.4).powi(2)
        + (4.0_f64 / 0.4).powi(2))
        / 4.0)
        .sqrt();
    let actual =
        CanteraNewtonSolver::weighted_rms(&solver.config, &state, &correction, &mut component_sums)
            .unwrap();
    assert!((actual - expected).abs() < 1e-12);
}

#[test]
fn reduced_bvp_guess_and_component_mapping_respect_mixed_endpoint_conditions() {
    let problem = CanteraExprBvpProblem {
        equations: vec![Expr::parse_expression("0"), Expr::parse_expression("0")],
        initial_guess: DMatrix::from_row_slice(2, 2, &[10.0, 11.0, 20.0, 21.0]),
        state_names: vec!["y".into(), "z".into()],
        independent_variable: "x".into(),
        boundary_conditions: std::collections::HashMap::from([
            ("y".into(), vec![(0, 1.0)]),
            ("z".into(), vec![(1, 2.0)]),
        ]),
        interval: (0.0, 1.0),
        steps: 2,
    };

    let packed = pack_bvp_initial_state(&problem).unwrap();
    assert_eq!(packed.values.as_slice(), &[20.0, 10.0, 21.0, 11.0]);
    assert_eq!(packed.full_state_positions, [1, 2, 3, 4]);
    assert_eq!(packed.fixed_state_values, [(0, 1.0), (5, 2.0)]);

    let mut config = CanteraNewtonConfig::new(4, 0.1, 0.1);
    config.component_count = 2;
    config.absolute_tolerances = DVector::from_vec(vec![0.1, 0.2]);
    config.relative_tolerances = DVector::from_vec(vec![0.01, 0.1]);
    config.full_state_positions = packed.full_state_positions;
    config.fixed_state_values = packed.fixed_state_values;
    let state = DVector::from_vec(vec![20.0, 10.0, 21.0, 11.0]);
    let correction = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let solver = CanteraNewtonSolver::new(
        state.clone(),
        |_| Ok(DVector::zeros(4)),
        |_| Ok(DMatrix::identity(4, 4)),
        config,
    )
    .unwrap();
    let mut component_sums = vec![0.0; 2];
    let actual =
        CanteraNewtonSolver::weighted_rms(&solver.config, &state, &correction, &mut component_sums)
            .unwrap();
    let y_scale: f64 = 0.1 + 0.01 * (1.0 + 10.0 + 11.0) / 3.0;
    let z_scale: f64 = 0.2 + 0.1 * (20.0 + 21.0 + 2.0) / 3.0;
    let expected = (((1.0 / z_scale).powi(2)
        + (2.0 / y_scale).powi(2)
        + (3.0 / z_scale).powi(2)
        + (4.0 / y_scale).powi(2))
        / 6.0)
        .sqrt();
    assert!((actual - expected).abs() < 1e-12);
}

#[test]
fn expr_adapter_solves_with_mixed_left_and_right_endpoint_conditions() {
    let problem = CanteraExprBvpProblem {
        equations: vec![Expr::parse_expression("1"), Expr::parse_expression("2")],
        initial_guess: DMatrix::from_row_slice(2, 3, &[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        state_names: vec!["y".into(), "z".into()],
        independent_variable: "x".into(),
        boundary_conditions: std::collections::HashMap::from([
            ("y".into(), vec![(0, 1.0)]),
            ("z".into(), vec![(1, 2.0)]),
        ]),
        interval: (0.0, 1.0),
        steps: 3,
    };
    let mut solver = CanteraNewtonSolver::from_expr_legacy_dense(
        problem,
        CanteraExprBvpOptions {
            absolute_tolerance: 1e-10,
            ..CanteraExprBvpOptions::default()
        },
    )
    .expect("mixed endpoint symbolic callbacks should prepare");

    let report = solver
        .solve()
        .expect("mixed left/right endpoint problem should converge");
    let expected = [0.0, 4.0 / 3.0, 2.0 / 3.0, 5.0 / 3.0, 4.0 / 3.0, 2.0];
    for (actual, expected) in report.state.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-10);
    }
}

#[test]
fn bounds_blocked_step_is_typed_and_does_not_evaluate_damping_trials() {
    let mut config = CanteraNewtonConfig::new(1, 1e-8, 0.0);
    config.bounds[0] = (0.0, 0.5);
    config.telemetry_mode = CanteraTelemetryMode::Counters;
    let initial = 0.5 - 1e-12;
    let mut solver = CanteraNewtonSolver::new(
        DVector::from_element(1, initial),
        |_| Ok(DVector::from_element(1, -1.0)),
        |_| Ok(DMatrix::identity(1, 1)),
        config,
    )
    .unwrap();
    match solver.solve() {
        Err(CanteraSolverError::BoundsBlocked { report }) => {
            assert_eq!(report.state[0], initial);
            assert!(report.last_bound_factor.unwrap() < 1e-10);
            assert_eq!(report.telemetry.residual_evaluations, 1);
            assert_eq!(report.telemetry.linear_solves, 1);
            assert_eq!(report.telemetry.damping_trials, 0);
        }
        other => panic!("expected typed bounds-blocked error, got {other:?}"),
    }
    assert_eq!(solver.state()[0], initial);
}

#[test]
fn bounds_blocked_stale_jacobian_is_refreshed_before_typed_failure() {
    let mut config = CanteraNewtonConfig::new(1, 0.01, 0.0);
    config.bounds[0] = (0.0, 1.0);
    config.telemetry_mode = CanteraTelemetryMode::Counters;
    let mut solver = CanteraNewtonSolver::new(
        DVector::from_element(1, 0.5),
        |state| Ok(DVector::from_element(1, state[0] - 2.0)),
        |_| Ok(DMatrix::identity(1, 1)),
        config,
    )
    .unwrap();

    match solver.solve() {
        Err(CanteraSolverError::BoundsBlocked { report }) => {
            assert_eq!(report.state[0], 1.0);
            assert_eq!(report.telemetry.jacobian_evaluations, 2);
            assert_eq!(report.telemetry.factorizations, 2);
            assert_eq!(report.telemetry.accepted_steps, 1);
            assert_eq!(report.telemetry.bound_limited_steps, 3);
            assert_eq!(report.telemetry.damping_trials, 1);
            assert_eq!(report.telemetry.residual_evaluations, 4);
        }
        other => panic!("expected bounds-blocked after Jacobian refresh, got {other:?}"),
    }
    assert_eq!(solver.state()[0], 1.0);
}

#[test]
fn accepted_unit_correction_satisfies_cantera_convergence_boundary() {
    let mut config = CanteraNewtonConfig::new(1, 1.0, 0.0);
    config.telemetry_mode = CanteraTelemetryMode::Counters;
    let mut solver = CanteraNewtonSolver::new(
        DVector::from_element(1, 0.0),
        |state| {
            Ok(DVector::from_element(
                1,
                if state[0] == 0.0 { -2.0 } else { -1.0 },
            ))
        },
        |_| Ok(DMatrix::identity(1, 1)),
        config,
    )
    .unwrap();

    let report = solver.solve().unwrap();
    assert_eq!(report.state[0], 2.0);
    assert_eq!(report.correction_norm, Some(1.0));
    assert_eq!(report.iterations, 1);
}

#[test]
fn iteration_limit_returns_last_accepted_state_and_partial_report() {
    let mut config = CanteraNewtonConfig::new(1, 0.01, 0.0);
    config.max_iterations = 3;
    config.telemetry_mode = CanteraTelemetryMode::Counters;
    let mut solver = CanteraNewtonSolver::new(
        DVector::from_element(1, 0.0),
        |state| Ok(DVector::from_element(1, state[0] - 1.0)),
        |_| Ok(DMatrix::from_element(1, 1, 2.0)),
        config,
    )
    .unwrap();

    match solver.solve() {
        Err(CanteraSolverError::NonConvergence { report }) => {
            assert_eq!(report.iterations, 3);
            assert_eq!(report.state[0], 0.875);
            assert_eq!(report.correction_norm, Some(6.25));
            assert_eq!(report.telemetry.accepted_steps, 3);
            assert_eq!(report.telemetry.jacobian_evaluations, 1);
            assert_eq!(report.telemetry.residual_evaluations, 6);
        }
        other => panic!("expected typed iteration-limit error, got {other:?}"),
    }
    assert_eq!(solver.state()[0], 0.875);
}

#[test]
fn timing_telemetry_separates_residual_and_jacobian_callbacks() {
    let mut solver = scalar_solver(0.25, 0.5, 3.0);
    solver.config.telemetry_mode = CanteraTelemetryMode::Timings;
    let report = solver.solve().unwrap();
    assert_eq!(report.telemetry.residual_evaluations, 2);
    assert_eq!(report.telemetry.jacobian_evaluations, 1);
    assert_eq!(
        report.telemetry.callback_time,
        report.telemetry.residual_callback_time + report.telemetry.jacobian_callback_time
    );
}

#[test]
fn jacobian_age_refreshes_only_after_the_configured_age_is_exceeded() {
    let mut config = CanteraNewtonConfig::new(1, 1e-8, 1e-8);
    config.max_jacobian_age = 1;
    config.telemetry_mode = CanteraTelemetryMode::Counters;
    let mut solver = CanteraNewtonSolver::new(
        DVector::from_element(1, 1.0),
        |state| Ok(DVector::from_element(1, state[0] * state[0] - 2.0)),
        |state| Ok(DMatrix::from_element(1, 1, 2.0 * state[0])),
        config,
    )
    .unwrap();
    let report = solver.solve().unwrap();
    assert!((report.state[0] - 2.0_f64.sqrt()).abs() < 1e-7);
    assert_eq!(
        report.telemetry.jacobian_evaluations,
        (report.iterations + 1) / 2,
        "age=1 permits two Newton steps before the next Jacobian refresh"
    );
}

#[test]
fn failed_damping_retries_with_a_fresh_jacobian_only_after_age_exceeds_one() {
    let mut config = CanteraNewtonConfig::new(1, 1e-8, 0.0);
    config.max_damping_trials = 7;
    config.max_jacobian_age = usize::MAX;
    config.telemetry_mode = CanteraTelemetryMode::Counters;
    let mut solver = CanteraNewtonSolver::new(
        DVector::from_element(1, 0.0),
        |state| {
            let value = if state[0] < 0.9 { -1.0 } else { -0.5 };
            Ok(DVector::from_element(1, value))
        },
        |_| Ok(DMatrix::identity(1, 1)),
        config,
    )
    .unwrap();
    match solver.solve() {
        Err(CanteraSolverError::NonConvergence { report }) => {
            assert_eq!(report.iterations, 3);
            assert_eq!(report.telemetry.accepted_steps, 1);
            assert_eq!(report.telemetry.jacobian_evaluations, 2);
            assert_eq!(report.telemetry.damping_trials, 15);
            assert_eq!(report.telemetry.damping_rejections, 14);
            assert_eq!(report.telemetry.residual_evaluations, 18);
            assert_eq!(report.telemetry.linear_solves, 18);
            assert_eq!(report.state[0], 1.0);
        }
        other => panic!("expected typed nonconvergence after refresh retry, got {other:?}"),
    }
    assert_eq!(solver.state()[0], 1.0);
}

#[test]
fn scalar_transition_trace_matches_independent_cantera_contract_model() {
    let residual_trace = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let jacobian_trace = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let residual_capture = residual_trace.clone();
    let jacobian_capture = jacobian_trace.clone();
    let mut config = CanteraNewtonConfig::new(1, 1e-8, 0.0);
    config.max_damping_trials = 7;
    config.max_jacobian_age = usize::MAX;
    config.telemetry_mode = CanteraTelemetryMode::Counters;
    let mut solver = CanteraNewtonSolver::new(
        DVector::from_element(1, 0.0),
        move |state| {
            residual_capture.borrow_mut().push(state[0]);
            Ok(DVector::from_element(
                1,
                if state[0] < 0.9 { -1.0 } else { -0.5 },
            ))
        },
        move |state| {
            jacobian_capture.borrow_mut().push(state[0]);
            Ok(DMatrix::identity(1, 1))
        },
        config,
    )
    .unwrap();

    let reference = scalar_cantera_reference(0.0, 1e-8, usize::MAX, 7, 10, |state| {
        if state < 0.9 {
            -1.0
        } else {
            -0.5
        }
    });
    assert!(!reference.converged);
    assert_eq!(reference.iterations, 3);
    assert_eq!(reference.accepted_states, [1.0]);
    assert_eq!(reference.accepted_correction_norms, [5e7]);
    assert_eq!(reference.jacobian_states, [0.0, 1.0]);
    assert_eq!(reference.damping_factors.len(), 15);
    assert_eq!(reference.base_correction_norms, [1e8, 5e7, 5e7]);
    assert_eq!(reference.trial_correction_norms, vec![5e7; 15]);
    assert_eq!(reference.jacobian_ages, [1, 2, 1]);
    assert_eq!(reference.refresh_retries, 1);

    let error = solver
        .solve()
        .expect_err("reference trace ends in line-search failure");
    let report = match error {
        CanteraSolverError::NonConvergence { report } => report,
        other => panic!("expected Cantera terminal nonconvergence, got {other:?}"),
    };
    let assert_trace_close = |actual: &[f64], expected: &[f64], label: &str| {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-13,
                "{label}[{index}] differs: actual={actual:.16e}, expected={expected:.16e}"
            );
        }
    };
    assert_trace_close(
        &residual_trace.borrow(),
        &reference.residual_states,
        "residual state trace",
    );
    assert_trace_close(
        &jacobian_trace.borrow(),
        &reference.jacobian_states,
        "Jacobian refresh trace",
    );
    assert_trace_close(
        &report.state.as_slice()[..],
        &reference.accepted_states[0..1],
        "terminal accepted state",
    );
    assert_trace_close(
        &reference.damping_factors,
        &[
            1.0,
            1.0,
            2.0_f64.sqrt().recip(),
            0.5,
            0.5 / 2.0_f64.sqrt(),
            0.25,
            0.25 / 2.0_f64.sqrt(),
            0.125,
            1.0,
            2.0_f64.sqrt().recip(),
            0.5,
            0.5 / 2.0_f64.sqrt(),
            0.25,
            0.25 / 2.0_f64.sqrt(),
            0.125,
        ],
        "reference damping sequence",
    );
    assert_eq!(report.iterations, reference.iterations);
    assert_eq!(report.correction_norm, Some(reference.correction_norm));
    assert_eq!(
        report.telemetry.jacobian_evaluations,
        reference.jacobian_states.len()
    );
    assert_eq!(
        report.telemetry.accepted_steps,
        reference.accepted_states.len()
    );
    assert_eq!(
        report.telemetry.damping_trials,
        reference.damping_factors.len()
    );
    assert_eq!(report.telemetry.damping_rejections, 14);
    assert_eq!(
        report.telemetry.residual_evaluations,
        reference.residual_states.len()
    );
}

#[test]
fn failed_damping_allows_exactly_four_jacobian_refresh_retries() {
    let mut config = CanteraNewtonConfig::new(1, 1e-8, 0.0);
    config.max_damping_trials = 7;
    config.max_jacobian_age = usize::MAX;
    config.telemetry_mode = CanteraTelemetryMode::Counters;
    let mut jacobian_evaluations = 0_i32;
    let mut solver = CanteraNewtonSolver::new(
        DVector::from_element(1, 0.0),
        |state| {
            let magnitude = 0.5_f64.powi(state[0].floor() as i32);
            Ok(DVector::from_element(1, -magnitude))
        },
        move |_| {
            let value = 0.5_f64.powi(jacobian_evaluations);
            jacobian_evaluations += 1;
            Ok(DMatrix::from_element(1, 1, value))
        },
        config,
    )
    .unwrap();

    match solver.solve() {
        Err(CanteraSolverError::NonConvergence { report }) => {
            assert_eq!(report.iterations, 10);
            assert_eq!(report.state[0], 5.0);
            assert_eq!(report.telemetry.accepted_steps, 5);
            assert_eq!(report.telemetry.jacobian_evaluations, 5);
            assert_eq!(report.telemetry.damping_trials, 40);
            assert_eq!(report.telemetry.damping_rejections, 35);
            assert_eq!(report.telemetry.residual_evaluations, 50);
            assert_eq!(report.telemetry.linear_solves, 50);
        }
        other => panic!("expected typed failure after four Jacobian retries, got {other:?}"),
    }
    assert_eq!(solver.state()[0], 5.0);
}
