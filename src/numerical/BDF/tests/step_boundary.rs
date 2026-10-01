use super::*;

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

struct NeverFactorizes;

impl BdfLinearBackend for NeverFactorizes {
    fn factor(&mut self, _matrix: &DMatrix<f64>) -> Option<Box<dyn BdfLinearFactorization>> {
        None
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
