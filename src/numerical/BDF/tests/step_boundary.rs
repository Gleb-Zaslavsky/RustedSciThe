use super::*;
use crate::numerical::BDF::common::{
    ArgumentValidationError, check_arguments, scale_func, select_initial_step,
};

#[test]
fn common_argument_validation_returns_typed_errors() {
    let rhs = |_, y: &DVector<f64>| y.clone();
    assert_eq!(
        check_arguments(rhs, &[]).err().unwrap(),
        ArgumentValidationError::EmptyInitialState
    );

    let rhs = |_, y: &DVector<f64>| y.clone();
    assert_eq!(
        check_arguments(rhs, &[f64::NAN]).err().unwrap(),
        ArgumentValidationError::NonFiniteInitialState
    );
}

#[test]
fn in_place_dense_jacobian_matches_owned_callback_and_reuses_contract() {
    let rate = 25.0;
    let make_solver = |in_place: bool| {
        let mut solver = BDF::new();
        solver.set_operation_counters_enabled(true);
        let rhs = move |_, y: &DVector<f64>| DVector::from_element(1, -rate * y[0]);
        let source = if in_place {
            BdfJacobianSource::StateDependentDenseInto(Box::new(move |_, _, out| {
                assert_eq!(out.shape(), (1, 1));
                out[(0, 0)] = -rate;
                Ok(())
            }))
        } else {
            BdfJacobianSource::StateDependent(Box::new(move |_, _| {
                BdfJacobian::from_dense(DMatrix::from_element(1, 1, -rate))
            }))
        };
        solver
            .try_set_initial_with_jacobian_source(
                Box::new(rhs),
                0.0,
                DVector::from_element(1, 1.0),
                0.2,
                0.05,
                NumberOrVec::Number(1e-8),
                NumberOrVec::Number(1e-10),
                source,
                None,
                false,
                Some(1e-4),
            )
            .unwrap();
        solver
    };

    let mut owned = make_solver(false);
    let mut in_place = make_solver(true);
    while owned.t < owned.t_bound {
        let (owned_ok, owned_error) = owned._step_impl();
        let (in_place_ok, in_place_error) = in_place._step_impl();
        assert!(owned_ok, "owned Jacobian step failed: {owned_error:?}");
        assert!(
            in_place_ok,
            "in-place Jacobian step failed: {in_place_error:?}"
        );
        assert!((owned.t - in_place.t).abs() < 1e-14);
        assert!((owned.y[0] - in_place.y[0]).abs() < 1e-12);
    }
    let counters = in_place.operation_counters();
    assert!(counters.jacobian_evaluations >= 1);
    assert!(counters.factorization_attempts >= 1);
}

#[test]
fn in_place_dense_jacobian_callback_failure_is_typed() {
    let mut solver = BDF::new();
    let error = solver.try_set_initial_with_jacobian_source(
        Box::new(|_, y| DVector::zeros(y.len())),
        0.0,
        DVector::from_element(1, 1.0),
        1.0,
        0.1,
        NumberOrVec::Number(1e-6),
        NumberOrVec::Number(1e-8),
        BdfJacobianSource::StateDependentDenseInto(Box::new(|_, _, _| {
            Err(BdfJacobianCallbackError::EvaluationFailed)
        })),
        None,
        false,
        Some(1e-3),
    );
    assert_eq!(error, Err(BdfConfigurationError::JacobianCallback));
}

#[test]
fn in_place_tolerance_scale_matches_public_scale_contract() {
    let state = DVector::from_vec(vec![-3.0, 0.5, 8.0]);
    for (rtol, atol) in [
        (NumberOrVec::Number(1e-3), NumberOrVec::Number(1e-6)),
        (
            NumberOrVec::Vec(vec![1e-3, 2e-3, 3e-3]),
            NumberOrVec::Number(1e-6),
        ),
        (
            NumberOrVec::Number(1e-3),
            NumberOrVec::Vec(vec![1e-6, 2e-6, 3e-6]),
        ),
        (
            NumberOrVec::Vec(vec![1e-3, 2e-3, 3e-3]),
            NumberOrVec::Vec(vec![1e-6, 2e-6, 3e-6]),
        ),
    ] {
        let expected = DVector::from_vec(scale_func(rtol.clone(), atol.clone(), &state));
        let mut actual = DVector::zeros(state.len());
        fill_tolerance_scale(&mut actual, &rtol, &atol, &state);
        assert_eq!(actual, expected);
    }
}

#[test]
fn adjacent_order_estimates_use_accepted_state_scale() {
    let order = 2;
    let max_order = 5;
    let mut differences = DMatrix::zeros(max_order + 3, 1);
    differences[(order, 0)] = 1.0;
    differences[(order + 2, 0)] = 1.0;
    let error_const = DVector::from_element(max_order + 1, 1.0);
    let accepted_state = DVector::from_element(1, 1_000.0);
    let predictor_state = DVector::from_element(1, 1.0);
    let accepted_scale = DVector::from_vec(scale_func(
        NumberOrVec::Number(0.01),
        NumberOrVec::Number(0.0),
        &accepted_state,
    ));
    let predictor_scale = DVector::from_vec(scale_func(
        NumberOrVec::Number(0.01),
        NumberOrVec::Number(0.0),
        &predictor_state,
    ));

    let (accepted_order, _) = select_order_and_step_factor(
        order,
        max_order,
        &differences,
        &error_const,
        0.1,
        &accepted_scale,
        0.9,
    );
    let (predictor_scaled_order, _) = select_order_and_step_factor(
        order,
        max_order,
        &differences,
        &error_const,
        0.1,
        &predictor_scale,
        0.9,
    );

    assert_eq!(accepted_order, 1);
    assert_eq!(predictor_scaled_order, 2);
}

#[test]
fn first_step_is_clipped_to_final_time_from_nonzero_start() {
    let mut solver = BDF::new();
    solver.set_initial(
        Box::new(|_, y| DVector::zeros(y.len())),
        0.9,
        DVector::from_vec(vec![1.0]),
        1.0,
        0.5,
        NumberOrVec::Number(1e-6),
        NumberOrVec::Number(1e-8),
        Some(Box::new(|_, _| DMatrix::zeros(1, 1))),
        None,
        false,
        Some(0.05),
    );

    // Force the first proposal past the endpoint. Starting away from zero
    // catches the old comparison against the preinitialized t_new = 0.
    solver.max_step = 0.5;
    solver.h_abs = 0.2;

    let (accepted, error) = solver._step_impl();

    assert!(accepted, "unexpected step failure: {error:?}");
    assert_eq!(solver.t, 1.0);
}

#[test]
fn automatic_initial_step_is_not_artificially_capped_at_one() {
    let fun: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>> =
        Box::new(|_, _| DVector::from_element(1, -1e-3));
    let y0 = DVector::from_element(1, 1_000.0);
    let f0 = fun(0.0, &y0);

    let h = select_initial_step(
        &fun,
        0.0,
        &y0,
        100.0,
        100.0,
        &f0,
        1.0,
        1.0,
        NumberOrVec::Number(1e-3),
        NumberOrVec::Number(1.0),
    );

    assert!(h > 1.0, "the selector unexpectedly capped h at one: {h}");
    assert!(
        (h - 20.0_f64.sqrt()).abs() < 1e-12,
        "unexpected initial h: {h}"
    );
}

#[test]
fn backward_integration_respects_order_cap_and_matches_exponential_reference() {
    let mut solver = BDF::new();
    solver.set_initial(
        Box::new(|_, y| 2.0 * y),
        1.0,
        DVector::from_element(1, 2.0_f64.exp()),
        0.0,
        0.05,
        NumberOrVec::Number(1e-7),
        NumberOrVec::Number(1e-10),
        Some(Box::new(|_, _| DMatrix::from_element(1, 1, 2.0))),
        None,
        false,
        Some(0.01),
    );
    solver.set_max_order_cap(2);

    let mut accepted_steps = 0;
    let mut max_order_seen = solver.current_order();
    while solver.t > solver.t_bound {
        let previous_t = solver.t;
        let (accepted, error) = solver._step_impl();
        assert!(
            accepted,
            "backward step failed at t={previous_t}: {error:?}"
        );
        assert!(
            solver.t < previous_t,
            "backward integration did not advance"
        );
        assert!(solver.current_order() <= 2, "order cap was exceeded");
        max_order_seen = max_order_seen.max(solver.current_order());
        accepted_steps += 1;
        assert!(accepted_steps < 20_000, "unexpectedly excessive step count");
    }

    let final_error = (solver.y[0] - 1.0).abs();
    assert_eq!(solver.t, 0.0);
    assert_eq!(
        max_order_seen, 2,
        "the solver did not exercise order growth"
    );
    assert!(
        final_error < 2e-5,
        "backward analytic error={final_error:e}"
    );
    println!(
        "[BDF backward order-cap] accepted_steps={accepted_steps} final_abs_error={final_error:.3e} max_order_seen={max_order_seen}"
    );
}

#[test]
fn stiff_nonlinear_logistic_matches_closed_form_and_reports_work() {
    let rate = 400.0;
    let y0 = 1e-3;
    let t_bound = 0.05;
    let exact = |t: f64| 1.0 / (1.0 + ((1.0 - y0) / y0) * (-rate * t).exp());
    let mut solver = BDF::new();
    solver.set_operation_counters_enabled(true);
    solver
        .try_set_initial_with_jacobian_source(
            Box::new(move |_, y| DVector::from_element(1, rate * y[0] * (1.0 - y[0]))),
            0.0,
            DVector::from_element(1, y0),
            t_bound,
            0.01,
            NumberOrVec::Number(1e-8),
            NumberOrVec::Number(1e-12),
            BdfJacobianSource::StateDependent(Box::new(move |_, y| {
                BdfJacobian::from_dense(DMatrix::from_element(1, 1, rate * (1.0 - 2.0 * y[0])))
            })),
            None,
            false,
            Some(1e-5),
        )
        .unwrap();

    let mut max_order_seen = solver.current_order();
    while solver.t < solver.t_bound {
        let previous_t = solver.t;
        let (accepted, error) = solver._step_impl();
        assert!(accepted, "stiff step failed at t={previous_t}: {error:?}");
        assert!(solver.t > previous_t, "accepted step did not advance time");
        max_order_seen = max_order_seen.max(solver.current_order());
        assert!(solver.operation_counters().accepted_steps < 20_000);
    }

    let counters = solver.operation_counters();
    let abs_error = (solver.y[0] - exact(t_bound)).abs();
    assert_eq!(solver.t, t_bound);
    assert!(abs_error < 2e-7, "stiff logistic error={abs_error:e}");
    assert!(max_order_seen >= 2, "controller did not increase order");
    assert!(counters.rhs_evaluations > 0);
    assert!(counters.jacobian_evaluations > 0);
    assert!(counters.factorization_attempts > 0);
    assert!(counters.linear_solve_attempts > 0);
    assert!(counters.accepted_steps > 0);
    assert!(counters.candidate_step_attempts >= counters.accepted_steps);
    println!(
        "[BDF stiff logistic] accepted={} rejected={} rhs={} jac={} factorizations={} linear_solves={} max_order={} abs_error={abs_error:.3e}",
        counters.accepted_steps,
        counters.rejected_step_attempts,
        counters.rhs_evaluations,
        counters.jacobian_evaluations,
        counters.factorization_attempts,
        counters.linear_solve_attempts,
        max_order_seen,
    );
}

#[test]
fn stiff_decay_matches_exponential_reference_and_reports_work() {
    let rate = 1_000.0;
    let t_bound = 0.02;
    let mut solver = BDF::new();
    solver.set_operation_counters_enabled(true);
    solver
        .try_set_initial_with_jacobian_source(
            Box::new(move |_, y| DVector::from_element(1, -rate * y[0])),
            0.0,
            DVector::from_element(1, 1.0),
            t_bound,
            0.002,
            NumberOrVec::Number(1e-8),
            NumberOrVec::Number(1e-12),
            BdfJacobianSource::Constant(BdfJacobian::from_dense(DMatrix::from_element(
                1, 1, -rate,
            ))),
            None,
            false,
            Some(1e-5),
        )
        .unwrap();

    let mut max_order_seen = solver.current_order();
    while solver.t < solver.t_bound {
        let previous_t = solver.t;
        let (accepted, error) = solver._step_impl();
        assert!(
            accepted,
            "stiff decay step failed at t={previous_t}: {error:?}"
        );
        assert!(solver.t > previous_t, "accepted step did not advance time");
        max_order_seen = max_order_seen.max(solver.current_order());
        assert!(solver.operation_counters().accepted_steps < 20_000);
    }

    let counters = solver.operation_counters();
    let exact = (-rate * t_bound).exp();
    let abs_error = (solver.y[0] - exact).abs();
    assert_eq!(solver.t, t_bound);
    assert!(abs_error < 2e-11, "stiff decay error={abs_error:e}");
    assert!(max_order_seen >= 2, "controller did not increase order");
    assert_eq!(counters.jacobian_evaluations, 1);
    assert!(counters.rhs_evaluations > 0);
    assert!(counters.factorization_attempts > 0);
    assert!(counters.linear_solve_attempts > 0);
    assert!(counters.candidate_step_attempts >= counters.accepted_steps);
    println!(
        "[BDF stiff decay] accepted={} rejected={} rhs={} jac={} factorizations={} linear_solves={} max_order={} abs_error={abs_error:.3e}",
        counters.accepted_steps,
        counters.rejected_step_attempts,
        counters.rhs_evaluations,
        counters.jacobian_evaluations,
        counters.factorization_attempts,
        counters.linear_solve_attempts,
        max_order_seen,
    );
}

struct NeverFactorizes;

impl BdfLinearBackend for NeverFactorizes {
    fn factor(&mut self, _matrix: &DMatrix<f64>) -> Option<Box<dyn BdfLinearFactorization>> {
        None
    }
}

struct OwnedFactorizationProbe {
    owned_calls: std::rc::Rc<std::cell::Cell<usize>>,
}

impl BdfLinearBackend for OwnedFactorizationProbe {
    fn factor(&mut self, _matrix: &DMatrix<f64>) -> Option<Box<dyn BdfLinearFactorization>> {
        panic!("shifted Jacobian unexpectedly used the borrowed factorization path");
    }

    fn factor_owned(&mut self, matrix: DMatrix<f64>) -> Option<Box<dyn BdfLinearFactorization>> {
        self.owned_calls.set(self.owned_calls.get() + 1);
        Some(Box::new(FailFirstSolveFactorization {
            matrix,
            fail: false,
        }))
    }
}

#[test]
fn shifted_dense_jacobian_is_moved_into_factorization_backend() {
    let owned_calls = std::rc::Rc::new(std::cell::Cell::new(0));
    let mut backend = OwnedFactorizationProbe {
        owned_calls: std::rc::Rc::clone(&owned_calls),
    };
    let jacobian = BdfJacobian::from_dense(DMatrix::from_diagonal_element(2, 2, -2.0));

    let factorization = backend
        .factor_shifted_jacobian(0.25, &jacobian)
        .expect("shifted matrix should factorize");
    let solution = factorization
        .solve(&DVector::from_element(2, 1.0))
        .expect("factorized shifted matrix should solve");

    assert_eq!(owned_calls.get(), 1);
    assert_eq!(solution, DVector::from_element(2, 2.0 / 3.0));
}

struct FixedCorrection;

impl BdfLinearFactorization for FixedCorrection {
    fn solve(&self, rhs: &DVector<f64>) -> Option<DVector<f64>> {
        Some(DVector::from_element(rhs.len(), 1.0))
    }
}

#[test]
fn newton_iteration_count_includes_rate_divergence_check() {
    let mut workspace = NewtonWorkspace::new(1);
    let result = solve_bdf_system(
        |_, _| DVector::from_element(1, 1.0),
        0.0,
        &DVector::zeros(1),
        1.0,
        &DVector::zeros(1),
        &FixedCorrection,
        &DVector::from_element(1, 1.0),
        1e-8,
        false,
        &mut Default::default(),
        &mut workspace,
    )
    .unwrap();

    assert!(!result.converged);
    assert_eq!(result.iterations, 2);
}

struct FailFirstSolveBackend {
    factor_calls: usize,
}

struct FailFirstSolveFactorization {
    matrix: DMatrix<f64>,
    fail: bool,
}

impl BdfLinearFactorization for FailFirstSolveFactorization {
    fn solve(&self, rhs: &DVector<f64>) -> Option<DVector<f64>> {
        if self.fail {
            Some(DVector::from_element(rhs.len(), 1e6))
        } else {
            self.matrix.clone().lu().solve(rhs)
        }
    }
}

impl BdfLinearBackend for FailFirstSolveBackend {
    fn factor(&mut self, matrix: &DMatrix<f64>) -> Option<Box<dyn BdfLinearFactorization>> {
        let fail = self.factor_calls == 0;
        self.factor_calls += 1;
        Some(Box::new(FailFirstSolveFactorization {
            matrix: matrix.clone(),
            fail,
        }))
    }
}

#[test]
fn failed_newton_attempt_refreshes_finite_difference_jacobian_before_retry() {
    let mut solver = BDF::new();
    solver.set_operation_counters_enabled(true);
    solver.set_initial(
        Box::new(|_, y| -y),
        0.0,
        DVector::from_vec(vec![1.0]),
        0.1,
        0.1,
        NumberOrVec::Number(1e-6),
        NumberOrVec::Number(1e-8),
        None,
        None,
        false,
        Some(0.01),
    );
    solver.set_linear_backend(Box::new(FailFirstSolveBackend { factor_calls: 0 }));

    let (accepted, error) = solver._step_impl();

    assert!(
        accepted,
        "the retry with a refreshed Jacobian should succeed: {error:?}"
    );
    assert_eq!(solver.operation_counters().jacobian_evaluations, 2);
    assert_eq!(solver.operation_counters().factorization_attempts, 2);
    assert_eq!(solver.accepted_steps, 1);
}

#[test]
fn invalid_jacobian_on_newton_refresh_returns_typed_error_without_advancing() {
    for (invalid_jacobian, expected_error) in [
        (
            BdfJacobian::from_dense(DMatrix::zeros(2, 2)),
            BdfStepError::InvalidJacobianDimension,
        ),
        (
            BdfJacobian::from_dense(DMatrix::from_element(1, 1, f64::NAN)),
            BdfStepError::NonFiniteJacobian,
        ),
    ] {
        let calls = std::rc::Rc::new(std::cell::Cell::new(0));
        let callback_calls = std::rc::Rc::clone(&calls);
        let mut solver = BDF::new();
        solver.set_operation_counters_enabled(true);
        solver
            .try_set_initial_with_jacobian_source(
                Box::new(|_, y| -y),
                0.0,
                DVector::from_vec(vec![1.0]),
                0.1,
                0.1,
                NumberOrVec::Number(1e-6),
                NumberOrVec::Number(1e-8),
                BdfJacobianSource::StateDependent(Box::new(move |_, _| {
                    let call = callback_calls.get();
                    callback_calls.set(call + 1);
                    if call == 0 {
                        BdfJacobian::from_dense(DMatrix::from_element(1, 1, -1.0))
                    } else {
                        invalid_jacobian.clone()
                    }
                })),
                None,
                false,
                Some(0.01),
            )
            .unwrap();
        solver.set_linear_backend(Box::new(FailFirstSolveBackend { factor_calls: 0 }));

        let (accepted, error) = solver._step_impl();

        assert!(!accepted);
        assert_eq!(error, Some(expected_error));
        assert_eq!(solver.t, 0.0, "invalid refresh must not advance time");
        assert_eq!(calls.get(), 2, "test must exercise the refresh callback");
        assert_eq!(solver.operation_counters().jacobian_evaluations, 1);
    }
}

#[test]
fn repeated_newton_factorization_failure_terminates_with_typed_underflow() {
    let mut solver = BDF::new();
    solver.set_initial(
        Box::new(|_, y| DVector::zeros(y.len())),
        0.9,
        DVector::from_vec(vec![1.0]),
        1.0,
        0.5,
        NumberOrVec::Number(1e-6),
        NumberOrVec::Number(1e-8),
        Some(Box::new(|_, _| DMatrix::zeros(1, 1))),
        None,
        false,
        Some(0.05),
    );
    solver.max_step = 0.5;
    solver.h_abs = 0.05;
    solver.set_linear_backend(Box::new(NeverFactorizes));

    let (accepted, error) = solver._step_impl();

    assert!(!accepted);
    assert_eq!(error, Some(BdfStepError::NewtonNonConvergence));
    assert_eq!(
        solver.t, 0.9,
        "a failed retry sequence must not advance time"
    );
}

#[test]
fn non_finite_step_size_is_rejected_before_step_mutation() {
    let mut solver = BDF::new();
    solver.set_initial(
        Box::new(|_, y| DVector::zeros(y.len())),
        0.0,
        DVector::from_vec(vec![1.0]),
        1.0,
        0.5,
        NumberOrVec::Number(1e-6),
        NumberOrVec::Number(1e-8),
        Some(Box::new(|_, _| DMatrix::zeros(1, 1))),
        None,
        false,
        Some(0.05),
    );
    solver.h_abs = f64::INFINITY;

    let (accepted, error) = solver._step_impl();

    assert!(!accepted);
    assert_eq!(error, Some(BdfStepError::InvalidStepSize));
    assert_eq!(solver.t, 0.0);
}

#[test]
fn proposed_time_reports_no_progress_when_addition_rounds_back_to_t() {
    let t = 2.0_f64.powi(53);
    let result = proposed_step_time(t, 0.25, 1.0, t + 4.0);

    assert_eq!(result, Err(BdfStepError::NoProgress));
}

#[test]
fn configured_max_step_caps_the_first_accepted_step() {
    let mut solver = BDF::new();
    solver.set_initial(
        Box::new(|_, y| DVector::zeros(y.len())),
        0.0,
        DVector::from_vec(vec![1.0]),
        1.0,
        0.025,
        NumberOrVec::Number(1e-6),
        NumberOrVec::Number(1e-8),
        Some(Box::new(|_, _| DMatrix::zeros(1, 1))),
        None,
        false,
        Some(0.2),
    );

    assert_eq!(solver.max_step, 0.025);
    let (accepted, error) = solver._step_impl();

    assert!(accepted, "unexpected step failure: {error:?}");
    assert_eq!(solver.t, 0.025);
}

#[test]
fn invalid_tolerance_dimensions_are_reported_without_mutating_solver() {
    let mut solver = BDF::new();
    let initial_y = solver.y.clone();
    let error = solver.try_set_initial(
        Box::new(|_, y| DVector::zeros(y.len())),
        2.0,
        DVector::from_vec(vec![1.0, 2.0]),
        3.0,
        0.1,
        NumberOrVec::Vec(vec![1e-6]),
        NumberOrVec::Number(1e-8),
        None,
        None,
        false,
        None,
    );

    assert_eq!(error, Err(BdfConfigurationError::ToleranceDimension));
    assert_eq!(solver.t, 0.0);
    assert_eq!(solver.y, initial_y);
}

#[test]
fn non_finite_tolerance_is_a_typed_configuration_error() {
    let mut solver = BDF::new();
    let initial_y = solver.y.clone();
    let error = solver.try_set_initial(
        Box::new(|_, y| DVector::zeros(y.len())),
        0.0,
        DVector::from_vec(vec![1.0]),
        1.0,
        0.1,
        NumberOrVec::Number(f64::NAN),
        NumberOrVec::Number(1e-8),
        None,
        None,
        false,
        None,
    );

    assert_eq!(error, Err(BdfConfigurationError::InvalidRelativeTolerance));
    assert_eq!(solver.t, 0.0);
    assert_eq!(solver.y, initial_y);
}

#[test]
fn invalid_initial_rhs_outputs_are_typed_and_transactional() {
    let invalid_rhs = [
        (
            Box::new(|_, _: &DVector<f64>| DVector::zeros(2))
                as Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
            BdfConfigurationError::InitialRhsDimension,
        ),
        (
            Box::new(|_, _: &DVector<f64>| DVector::from_element(1, f64::NAN)),
            BdfConfigurationError::NonFiniteInitialRhs,
        ),
    ];

    for (rhs, expected_error) in invalid_rhs {
        let mut solver = BDF::new();
        let original_y = solver.y.clone();
        let error = solver.try_set_initial(
            rhs,
            0.0,
            DVector::from_vec(vec![1.0]),
            1.0,
            0.1,
            NumberOrVec::Number(1e-6),
            NumberOrVec::Number(1e-8),
            None,
            None,
            false,
            Some(0.01),
        );

        assert_eq!(error, Err(expected_error));
        assert_eq!(solver.t, 0.0);
        assert_eq!(solver.y, original_y);
    }
}

#[test]
fn invalid_automatic_initial_step_rhs_is_typed_and_transactional() {
    for (return_non_finite, expected_error) in [
        (false, BdfConfigurationError::InitialStepRhsDimension),
        (true, BdfConfigurationError::NonFiniteInitialStepRhs),
    ] {
        let calls = std::rc::Rc::new(std::cell::Cell::new(0));
        let callback_calls = std::rc::Rc::clone(&calls);
        let mut solver = BDF::new();
        let original_y = solver.y.clone();
        let error = solver.try_set_initial(
            Box::new(move |_, y| {
                let call = callback_calls.get();
                callback_calls.set(call + 1);
                if call == 0 {
                    DVector::zeros(y.len())
                } else if return_non_finite {
                    DVector::from_element(y.len(), f64::NAN)
                } else {
                    DVector::zeros(y.len() + 1)
                }
            }),
            0.0,
            DVector::from_vec(vec![1.0]),
            1.0,
            0.1,
            NumberOrVec::Number(1e-6),
            NumberOrVec::Number(1e-8),
            None,
            None,
            false,
            None,
        );

        assert_eq!(error, Err(expected_error));
        assert_eq!(
            calls.get(),
            2,
            "initial value and selector probe are evaluated once"
        );
        assert_eq!(solver.t, 0.0);
        assert_eq!(solver.y, original_y);
    }
}

#[test]
fn initialization_rhs_counter_includes_automatic_step_probe() {
    let mut solver = BDF::new();
    solver.set_operation_counters_enabled(true);
    solver
        .try_set_initial(
            Box::new(|_, y| -y),
            0.0,
            DVector::from_vec(vec![1.0]),
            1.0,
            0.1,
            NumberOrVec::Number(1e-6),
            NumberOrVec::Number(1e-8),
            Some(Box::new(|_, _| DMatrix::from_element(1, 1, -1.0))),
            None,
            false,
            None,
        )
        .unwrap();

    assert_eq!(solver.operation_counters().rhs_evaluations, 2);
}

#[test]
fn malformed_initial_finite_difference_probes_return_typed_errors() {
    for (return_non_finite, expected_error) in [
        (false, BdfConfigurationError::InitialJacobianRhsDimension),
        (true, BdfConfigurationError::NonFiniteInitialJacobianRhs),
    ] {
        let calls = std::rc::Rc::new(std::cell::Cell::new(0));
        let callback_calls = std::rc::Rc::clone(&calls);
        let mut solver = BDF::new();
        let original_y = solver.y.clone();
        let error = solver.try_set_initial(
            Box::new(move |_, y| {
                let call = callback_calls.get();
                callback_calls.set(call + 1);
                if call == 0 {
                    -y
                } else if return_non_finite {
                    DVector::from_element(y.len(), f64::NAN)
                } else {
                    DVector::zeros(y.len() + 1)
                }
            }),
            0.0,
            DVector::from_vec(vec![1.0]),
            1.0,
            0.1,
            NumberOrVec::Number(1e-6),
            NumberOrVec::Number(1e-8),
            None,
            None,
            false,
            Some(0.01),
        );

        assert_eq!(error, Err(expected_error));
        assert_eq!(calls.get(), 2);
        assert_eq!(solver.t, 0.0);
        assert_eq!(solver.y, original_y);
    }
}

#[test]
fn runtime_rhs_dimension_error_is_typed_and_does_not_advance_time() {
    let calls = std::rc::Rc::new(std::cell::Cell::new(0));
    let callback_calls = std::rc::Rc::clone(&calls);
    let mut solver = BDF::new();
    solver
        .try_set_initial_with_jacobian_source(
            Box::new(move |_, y| {
                let call = callback_calls.get();
                callback_calls.set(call + 1);
                if call == 0 {
                    -y
                } else {
                    DVector::zeros(y.len() + 1)
                }
            }),
            0.0,
            DVector::from_vec(vec![1.0]),
            0.1,
            0.1,
            NumberOrVec::Number(1e-6),
            NumberOrVec::Number(1e-8),
            BdfJacobianSource::Constant(BdfJacobian::from_dense(DMatrix::from_element(1, 1, -1.0))),
            None,
            false,
            Some(0.01),
        )
        .unwrap();

    let (accepted, error) = solver._step_impl();

    assert!(!accepted);
    assert_eq!(error, Some(BdfStepError::RhsDimension));
    assert_eq!(solver.t, 0.0);
}

#[test]
fn non_finite_newton_rhs_keeps_the_step_shrink_retry_path() {
    let calls = std::rc::Rc::new(std::cell::Cell::new(0));
    let callback_calls = std::rc::Rc::clone(&calls);
    let mut solver = BDF::new();
    solver
        .try_set_initial_with_jacobian_source(
            Box::new(move |_, y| {
                let call = callback_calls.get();
                callback_calls.set(call + 1);
                if call == 1 {
                    DVector::from_element(y.len(), f64::NAN)
                } else {
                    -y
                }
            }),
            0.0,
            DVector::from_vec(vec![1.0]),
            0.1,
            0.1,
            NumberOrVec::Number(1e-6),
            NumberOrVec::Number(1e-8),
            BdfJacobianSource::Constant(BdfJacobian::from_dense(DMatrix::from_element(1, 1, -1.0))),
            None,
            false,
            Some(0.01),
        )
        .unwrap();

    let (accepted, error) = solver._step_impl();

    assert!(accepted, "finite retry should recover: {error:?}");
    assert!(solver.t > 0.0);
    assert!(calls.get() >= 3, "the failed callback should be retried");
}

#[test]
fn persistent_non_finite_newton_rhs_reports_typed_error_after_retries() {
    let calls = std::rc::Rc::new(std::cell::Cell::new(0));
    let callback_calls = std::rc::Rc::clone(&calls);
    let mut solver = BDF::new();
    solver
        .try_set_initial_with_jacobian_source(
            Box::new(move |_, y| {
                let call = callback_calls.get();
                callback_calls.set(call + 1);
                if call == 0 {
                    -y
                } else {
                    DVector::from_element(y.len(), f64::NAN)
                }
            }),
            0.0,
            DVector::from_vec(vec![1.0]),
            0.1,
            0.1,
            NumberOrVec::Number(1e-6),
            NumberOrVec::Number(1e-8),
            BdfJacobianSource::Constant(BdfJacobian::from_dense(DMatrix::from_element(1, 1, -1.0))),
            None,
            false,
            Some(0.01),
        )
        .unwrap();

    let (accepted, error) = solver._step_impl();

    assert!(!accepted);
    assert_eq!(error, Some(BdfStepError::NonFiniteRhs));
    assert_eq!(solver.t, 0.0);
    assert!(calls.get() > 2, "the step should retry before failing");
}

#[test]
fn malformed_runtime_finite_difference_probes_return_typed_step_errors() {
    for (return_non_finite, expected_error) in [
        (false, BdfStepError::RhsDimension),
        (true, BdfStepError::NonFiniteRhs),
    ] {
        let calls = std::rc::Rc::new(std::cell::Cell::new(0));
        let callback_calls = std::rc::Rc::clone(&calls);
        let mut solver = BDF::new();
        solver
            .try_set_initial(
                Box::new(move |_, y| {
                    let call = callback_calls.get();
                    callback_calls.set(call + 1);
                    if call >= 5 {
                        if return_non_finite {
                            DVector::from_element(y.len(), f64::NAN)
                        } else {
                            DVector::zeros(y.len() + 1)
                        }
                    } else {
                        -y
                    }
                }),
                0.0,
                DVector::from_vec(vec![1.0]),
                0.1,
                0.1,
                NumberOrVec::Number(1e-6),
                NumberOrVec::Number(1e-8),
                None,
                None,
                false,
                Some(0.01),
            )
            .unwrap();
        solver.set_linear_backend(Box::new(FailFirstSolveBackend { factor_calls: 0 }));

        let (accepted, error) = solver._step_impl();

        assert!(!accepted);
        assert_eq!(error, Some(expected_error));
        assert_eq!(solver.t, 0.0);
        assert_eq!(
            calls.get(),
            6,
            "the invalid output must come from the FD probe"
        );
    }
}

#[test]
fn unsupported_jacobian_sparsity_is_rejected_without_mutating_solver() {
    let mut solver = BDF::new();
    let initial_y = solver.y.clone();
    let error = solver.try_set_initial(
        Box::new(|_, y| DVector::zeros(y.len())),
        0.0,
        DVector::from_vec(vec![1.0]),
        1.0,
        0.1,
        NumberOrVec::Number(1e-6),
        NumberOrVec::Number(1e-8),
        None,
        Some(DMatrix::from_element(1, 1, 1.0)),
        false,
        None,
    );

    assert_eq!(
        error,
        Err(BdfConfigurationError::UnsupportedJacobianSparsity)
    );
    assert_eq!(solver.t, 0.0);
    assert_eq!(solver.y, initial_y);
}

#[test]
fn vectorized_rhs_is_rejected_without_mutating_solver() {
    let mut solver = BDF::new();
    let initial_y = solver.y.clone();
    let error = solver.try_set_initial(
        Box::new(|_, y| DVector::zeros(y.len())),
        0.0,
        DVector::from_vec(vec![1.0]),
        1.0,
        0.1,
        NumberOrVec::Number(1e-6),
        NumberOrVec::Number(1e-8),
        None,
        None,
        true,
        None,
    );

    assert_eq!(error, Err(BdfConfigurationError::UnsupportedVectorizedRhs));
    assert_eq!(solver.t, 0.0);
    assert_eq!(solver.y, initial_y);
}

#[test]
fn explicit_constant_jacobian_is_reused_and_counted_once() {
    let mut solver = BDF::new();
    solver.set_operation_counters_enabled(true);
    solver
        .try_set_initial_with_jacobian_source(
            Box::new(|_, y| -y),
            0.0,
            DVector::from_vec(vec![1.0]),
            0.2,
            0.05,
            NumberOrVec::Number(1e-7),
            NumberOrVec::Number(1e-10),
            BdfJacobianSource::Constant(BdfJacobian::from_dense(DMatrix::from_element(1, 1, -1.0))),
            None,
            false,
            Some(0.01),
        )
        .unwrap();
    solver.set_linear_backend(Box::new(FailFirstSolveBackend { factor_calls: 0 }));

    let (accepted, error) = solver._step_impl();
    assert!(accepted, "constant-J retry failed: {error:?}");
    assert_eq!(solver.operation_counters().factorization_attempts, 2);

    while solver.t < solver.t_bound {
        let (accepted, error) = solver._step_impl();
        assert!(accepted, "unexpected step failure: {error:?}");
    }

    assert_eq!(
        solver.jacobian_refresh_policy,
        JacobianRefreshPolicy::Constant
    );
    assert_eq!(solver.operation_counters().jacobian_evaluations, 1);
    assert!((solver.y[0] - (-0.2_f64).exp()).abs() < 2e-6);
}

#[test]
fn invalid_constant_jacobian_is_rejected_before_solver_mutation() {
    let mut solver = BDF::new();
    let original_y = solver.y.clone();
    let error = solver.try_set_initial_with_jacobian_source(
        Box::new(|_, y| DVector::zeros(y.len())),
        0.0,
        DVector::from_vec(vec![1.0]),
        1.0,
        0.1,
        NumberOrVec::Number(1e-6),
        NumberOrVec::Number(1e-8),
        BdfJacobianSource::Constant(BdfJacobian::from_dense(DMatrix::zeros(2, 2))),
        None,
        false,
        None,
    );

    assert_eq!(error, Err(BdfConfigurationError::JacobianDimension));
    assert_eq!(solver.t, 0.0);
    assert_eq!(solver.y, original_y);
}

#[test]
fn invalid_state_dependent_jacobian_is_a_transactional_typed_error() {
    let mut solver = BDF::new();
    let original_y = solver.y.clone();
    let error = solver.try_set_initial_with_jacobian_source(
        Box::new(|_, y| DVector::zeros(y.len())),
        0.0,
        DVector::from_vec(vec![1.0]),
        1.0,
        0.1,
        NumberOrVec::Number(1e-6),
        NumberOrVec::Number(1e-8),
        BdfJacobianSource::StateDependent(Box::new(|_, _| {
            BdfJacobian::from_dense(DMatrix::zeros(2, 2))
        })),
        None,
        false,
        None,
    );

    assert_eq!(error, Err(BdfConfigurationError::JacobianDimension));
    assert_eq!(solver.t, 0.0);
    assert_eq!(solver.y, original_y);
}

#[test]
fn native_jacobian_replacement_reports_typed_errors_transactionally() {
    for (invalid_jacobian, expected_error) in [
        (
            BdfJacobian::from_dense(DMatrix::zeros(2, 2)),
            BdfConfigurationError::JacobianDimension,
        ),
        (
            BdfJacobian::from_dense(DMatrix::from_element(1, 1, f64::INFINITY)),
            BdfConfigurationError::NonFiniteJacobian,
        ),
    ] {
        let mut solver = BDF::new();
        solver
            .try_set_initial(
                Box::new(|_, y| -y),
                0.0,
                DVector::from_vec(vec![1.0]),
                1.0,
                0.1,
                NumberOrVec::Number(1e-6),
                NumberOrVec::Number(1e-8),
                None,
                None,
                false,
                Some(0.01),
            )
            .unwrap();
        let original_jacobian = match &solver.J {
            BdfJacobian::Dense(matrix) => matrix.clone(),
            _ => panic!("finite-difference initialization should use dense storage"),
        };

        let error = solver.try_set_native_jacobian(Box::new(move |_, _| invalid_jacobian.clone()));

        assert_eq!(error, Err(expected_error));
        assert_eq!(
            solver.jacobian_refresh_policy,
            JacobianRefreshPolicy::FiniteDifference
        );
        assert_eq!(solver.J.shape(), original_jacobian.shape());
        match &solver.J {
            BdfJacobian::Dense(matrix) => assert_eq!(matrix, &original_jacobian),
            _ => unreachable!("failed replacement must preserve the original Jacobian"),
        }
    }
}

#[test]
fn finite_difference_jacobian_scales_steps_per_component() {
    let y = DVector::from_vec(vec![1e-12, 1e6]);
    let fun = |_, y: &DVector<f64>| DVector::from_vec(vec![-2.0 * y[0], 3.0 * y[1]]);
    let f0 = fun(0.0, &y);
    let result = finite_difference_jacobian_rhs(
        &fun,
        0.0,
        &y,
        &f0,
        &NumberOrVec::Vec(vec![1e-14, 1e-8]),
        None,
    )
    .unwrap();

    assert!(
        (result.matrix[(0, 0)] + 2.0).abs() < 1e-8,
        "J00={}",
        result.matrix[(0, 0)]
    );
    assert!(
        (result.matrix[(1, 1)] - 3.0).abs() < 1e-8,
        "J11={}",
        result.matrix[(1, 1)]
    );
    assert!(result.matrix[(0, 1)].abs() < 1e-12);
    assert!(result.matrix[(1, 0)].abs() < 1e-12);
    assert!(
        result
            .factor
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
    );
}

#[test]
fn finite_difference_jacobian_increases_factor_when_signal_is_near_roundoff() {
    let y = DVector::from_element(1, 1.0);
    let fun = |_, y: &DVector<f64>| DVector::from_element(1, 1e8 + y[0]);
    let f0 = fun(0.0, &y);
    let result =
        finite_difference_jacobian_rhs(&fun, 0.0, &y, &f0, &NumberOrVec::Number(1e-8), None)
            .unwrap();

    assert!(
        (result.matrix[(0, 0)] - 1.0).abs() < 1e-3,
        "J00={}",
        result.matrix[(0, 0)]
    );
    assert!(result.factor[0] > f64::EPSILON.sqrt());
}
