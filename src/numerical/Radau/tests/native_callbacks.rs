//! Correctness and contract stories for direct native Radau callbacks.

use nalgebra::{DMatrix, DVector};

use super::super::api::{
    RadauConfig, RadauErrorKind, RadauExecution, RadauJacobianSource, RadauNativeSolver,
    RadauTelemetryMode,
};

fn config() -> RadauConfig {
    RadauConfig {
        t_bound: 0.5,
        first_step: Some(0.05),
        max_step: 0.1,
        rtol: 1.0e-9,
        atol: 1.0e-11,
        max_newton_iterations: 8,
        telemetry: RadauTelemetryMode::Counters,
        ..RadauConfig::default()
    }
}

#[test]
fn native_analytic_jacobian_callback_reaches_new_core() {
    let mut solver = RadauNativeSolver::prepare(
        config(),
        |_t, y: &DVector<f64>| DVector::from_vec(vec![-y[0]]),
        Some(|_t: f64, _y: &DVector<f64>| DMatrix::from_vec(1, 1, vec![-1.0])),
    )
    .unwrap();

    let solution = solver.solve(&[1.0]).unwrap();
    assert_eq!(solver.config().execution, RadauExecution::NativeCallbacks);
    assert_eq!(
        solver.config().jacobian_source,
        RadauJacobianSource::Analytic
    );
    assert_eq!(solution.t, 0.5);
    assert!((solution.y[0] - (-0.5f64).exp()).abs() < 2.0e-8);
    assert!(
        solution.telemetry().counters["jacobian_calls"] > 0,
        "counters={:?}",
        solution.telemetry().counters
    );
    assert!(solution.telemetry().counters["workspace_resizes"] > 0);
}

#[test]
fn native_residual_only_uses_finite_difference_jacobian() {
    let mut solver = RadauNativeSolver::prepare(
        config(),
        |_t, y: &DVector<f64>| DVector::from_vec(vec![-y[0]]),
        Option::<fn(f64, &DVector<f64>) -> DMatrix<f64>>::None,
    )
    .unwrap();

    let solution = solver.solve(&[1.0]).unwrap();
    assert!((solution.y[0] - (-0.5f64).exp()).abs() < 2.0e-7);
    assert!(
        solution.telemetry().counters["jacobian_calls"] > 0,
        "counters={:?}",
        solution.telemetry().counters
    );
    assert!(solution.telemetry().counters["finite_difference_probes"] > 0);
}

#[test]
fn native_callback_shapes_and_non_finite_values_are_typed_errors() {
    let mut bad_shape = RadauNativeSolver::prepare(
        config(),
        |_t, _y: &DVector<f64>| DVector::from_vec(vec![1.0, 2.0]),
        Option::<fn(f64, &DVector<f64>) -> DMatrix<f64>>::None,
    )
    .unwrap();
    let shape = bad_shape.solve(&[1.0]).unwrap_err();
    assert_eq!(shape.kind(), RadauErrorKind::Shape);

    let mut bad_jacobian = RadauNativeSolver::prepare(
        config(),
        |_t, y: &DVector<f64>| DVector::from_vec(vec![-y[0]]),
        Some(|_t: f64, _y: &DVector<f64>| DMatrix::from_vec(1, 1, vec![f64::NAN])),
    )
    .unwrap();
    let non_finite = bad_jacobian.solve(&[1.0]).unwrap_err();
    assert_eq!(non_finite.kind(), RadauErrorKind::NonFiniteCallback);
}
