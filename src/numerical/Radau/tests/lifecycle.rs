//! Preparation, continuation, restart, cache provenance, and typed-error
//! lifecycle stories.

use super::super::new::linear::RadauLinearWorkspace;
use super::super::new::workspace::RadauWorkspace;
use super::super::new::{
    config::RadauMatrixLayout,
    continuation::RadauContinuation,
    linear::{RadauDenseComplexShiftedMatrix, RadauDenseShiftedMatrix, RadauFactorKey, RadauShift},
    prepared::RadauPreparedModel,
    session::RadauSession,
};
use crate::symbolic::symbolic_engine::Expr;

#[test]
fn workspace_reuses_capacity_for_same_dimension() {
    let mut workspace = RadauWorkspace::default();
    workspace.resize_for_dimension(8).unwrap();
    let capacities = [
        workspace.stage_states.capacity(),
        workspace.jacobian.capacity(),
        match &workspace.linear {
            RadauLinearWorkspace::Dense(linear) => linear.complex_real.capacity(),
            _ => panic!("default workspace must use dense linear storage"),
        },
    ];
    workspace.resize_for_dimension(8).unwrap();
    assert_eq!(capacities[0], workspace.stage_states.capacity());
    assert_eq!(capacities[1], workspace.jacobian.capacity());
    assert_eq!(
        capacities[2],
        match &workspace.linear {
            RadauLinearWorkspace::Dense(linear) => linear.complex_real.capacity(),
            _ => 0,
        }
    );
}

#[test]
fn factor_key_invalidates_on_jacobian_step_or_shift_change() {
    let key = RadauFactorKey::new(
        2,
        7,
        RadauShift::Complex {
            real: 1.0,
            imag: 0.5,
        },
        RadauMatrixLayout::Dense,
    );
    assert!(key.remains_valid_for(key));
    assert!(!key.remains_valid_for(RadauFactorKey::new(
        3,
        7,
        RadauShift::Complex {
            real: 1.0,
            imag: 0.5,
        },
        RadauMatrixLayout::Dense,
    )));
    assert!(!key.remains_valid_for(RadauFactorKey::new(
        2,
        7,
        RadauShift::Real { mu_over_h: 1.0 },
        RadauMatrixLayout::Dense,
    )));
}

#[test]
fn parameter_rebind_reuses_session_storage_and_invalidates_numeric_generation() {
    let config = super::super::new::config::RadauConfig::default();
    let prepared = RadauPreparedModel::prepare_with_parameter_count(4, 2, &config).unwrap();
    let mut session = RadauSession::new(prepared).unwrap();
    let parameter_ptr = session.parameter_values().as_ptr();
    let generation = session.numeric_generation();

    session.rebind_parameters(&[2.0, 3.0]).unwrap();

    assert_eq!(session.parameter_values(), &[2.0, 3.0]);
    assert_eq!(session.parameter_values().as_ptr(), parameter_ptr);
    assert!(session.numeric_generation() > generation);
    assert!(session.rebind_parameters(&[2.0]).is_err());
}

#[test]
fn dense_shifted_storage_is_filled_in_place_and_can_transfer_ownership() {
    let jacobian = [1.0, 2.0, 3.0, 4.0];
    let mut shifted = RadauDenseShiftedMatrix::new(2).unwrap();
    let storage_ptr = shifted.values().as_ptr();

    shifted.load_real_shifted(&jacobian, 10.0).unwrap();

    assert_eq!(shifted.dimension(), 2);
    assert_eq!(shifted.values(), &[9.0, -2.0, -3.0, 6.0]);
    shifted.load_real_shifted(&jacobian, 11.0).unwrap();
    assert_eq!(shifted.values().as_ptr(), storage_ptr);

    let owned = shifted.into_values();
    assert_eq!(owned, vec![10.0, -2.0, -3.0, 7.0]);
}

#[test]
fn complex_shifted_storage_keeps_real_and_imaginary_parts_separate() {
    let jacobian = [1.0, 2.0, 3.0, 4.0];
    let mut shifted = RadauDenseComplexShiftedMatrix::new(2).unwrap();
    shifted.load_shifted(&jacobian, 10.0, 0.5).unwrap();

    assert_eq!(shifted.real(), &[9.0, -2.0, -3.0, 6.0]);
    assert_eq!(shifted.imag(), &[0.5, 0.0, 0.0, 0.5]);
}

#[test]
fn aot_route_requires_an_explicit_lifecycle_configuration() {
    let problem = super::super::api::RadauProblem::new(
        vec![Expr::parse_expression("-y")],
        vec!["y".to_owned()],
        "t",
    );
    let error = match super::super::api::RadauSolver::prepare(
        problem,
        super::super::api::RadauConfig {
            execution: super::super::api::RadauExecution::Aot,
            ..super::super::api::RadauConfig::default()
        },
    ) {
        Err(error) => error,
        Ok(_) => panic!("AOT preparation unexpectedly accepted missing configuration"),
    };
    assert_eq!(error.kind(), super::super::api::RadauErrorKind::Unsupported);
}

#[test]
fn session_dense_step_reuses_prepared_workspace_across_steps() {
    let config = super::super::new::config::RadauConfig {
        matrix_layout: RadauMatrixLayout::Dense,
        ..super::super::new::config::RadauConfig::default()
    };
    let prepared = RadauPreparedModel::prepare(1, &config).unwrap();
    let mut session = RadauSession::new(prepared).unwrap();
    let initial_capacity = session.workspace().stage_states.capacity();
    let mut residual = |_t: f64, state: &[f64], output: &mut [f64]| {
        output[0] = -state[0];
        Ok::<(), super::super::new::error::RadauError>(())
    };
    let mut jacobian = |_t: f64, _state: &[f64], output: &mut [f64]| {
        output[0] = -1.0;
        Ok::<(), super::super::new::error::RadauError>(())
    };
    let mut next = [0.0];
    session
        .try_dense_step(
            &config,
            &mut residual,
            &mut jacobian,
            0.0,
            0.1,
            &[1.0],
            &mut next,
        )
        .unwrap();
    let after_first = session.workspace().stage_states.capacity();
    let mut next_again = [0.0];
    session
        .try_dense_step(
            &config,
            &mut residual,
            &mut jacobian,
            0.1,
            0.1,
            &next,
            &mut next_again,
        )
        .unwrap();
    assert_eq!(initial_capacity, after_first);
    assert_eq!(after_first, session.workspace().stage_states.capacity());
    assert!((next_again[0] - (-0.2_f64).exp()).abs() < 1.0e-9);
}

#[test]
fn session_dense_step_rejects_structured_layout_without_fallback() {
    let dense_config = super::super::new::config::RadauConfig::default();
    let prepared = RadauPreparedModel::prepare(1, &dense_config).unwrap();
    let mut session = RadauSession::new(prepared).unwrap();
    let structured_config = super::super::new::config::RadauConfig {
        matrix_layout: RadauMatrixLayout::Banded { lower: 1, upper: 1 },
        ..dense_config
    };
    let mut residual = |_t: f64, state: &[f64], output: &mut [f64]| {
        output[0] = -state[0];
        Ok::<(), super::super::new::error::RadauError>(())
    };
    let mut jacobian = |_t: f64, _state: &[f64], output: &mut [f64]| {
        output[0] = -1.0;
        Ok::<(), super::super::new::error::RadauError>(())
    };
    let error = session
        .try_dense_step(
            &structured_config,
            &mut residual,
            &mut jacobian,
            0.0,
            0.1,
            &[1.0],
            &mut [0.0],
        )
        .unwrap_err();
    assert!(matches!(
        error,
        super::super::new::error::RadauError::UnsupportedRoute(
            super::super::new::error::RadauUnsupportedRoute::DenseConfigurationRequired,
        )
    ));
}

#[test]
fn continuation_rebind_and_restart_preserve_preparation_and_capacity() {
    let config = super::super::new::config::RadauConfig::default();
    let mut continuation =
        RadauContinuation::prepare(2, 1, &[1.0, 2.0], 0.0, 1.0, &config).unwrap();
    let preparation_generation = continuation.prepared().preparation_generation();
    let state_ptr = continuation.initial_state().as_ptr();
    let state_capacity = continuation.initial_state_capacity();
    let numeric_generation = continuation.session().numeric_generation();

    continuation.rebind_parameters(&[3.0]).unwrap();
    continuation.restart(&[4.0, 5.0], 2.0, 3.0).unwrap();

    assert_eq!(
        continuation.prepared().preparation_generation(),
        preparation_generation
    );
    assert_eq!(continuation.initial_state(), &[4.0, 5.0]);
    assert_eq!(continuation.initial_state().as_ptr(), state_ptr);
    assert_eq!(continuation.initial_state_capacity(), state_capacity);
    assert_eq!(continuation.interval(), (2.0, 3.0));
    assert!(continuation.session().numeric_generation() > numeric_generation);
    assert_eq!(continuation.session().parameter_values(), &[3.0]);
}

#[test]
fn continuation_restart_validation_is_transactional() {
    let config = super::super::new::config::RadauConfig::default();
    let mut continuation =
        RadauContinuation::prepare(2, 0, &[1.0, 2.0], 0.0, 1.0, &config).unwrap();
    let generation = continuation.session().numeric_generation();
    let error = continuation
        .restart(&[f64::NAN, 9.0], 4.0, 4.0)
        .unwrap_err();

    assert!(matches!(
        error,
        super::super::new::error::RadauError::InvalidConfiguration(
            super::super::new::error::RadauConfigError::InvalidTimeBounds,
        )
    ));
    assert_eq!(continuation.initial_state(), &[1.0, 2.0]);
    assert_eq!(continuation.interval(), (0.0, 1.0));
    assert_eq!(continuation.session().numeric_generation(), generation);
}

#[test]
fn continuation_long_series_keeps_prepared_storage_bounded() {
    let config = super::super::new::config::RadauConfig::default();
    let mut continuation =
        RadauContinuation::prepare(2, 1, &[1.0, 2.0], 0.0, 1.0, &config).unwrap();
    let preparation_generation = continuation.prepared().preparation_generation();
    let initial_state_ptr = continuation.initial_state().as_ptr();
    let initial_state_capacity = continuation.initial_state_capacity();
    let initial_workspace_capacities = {
        let session = continuation.session();
        let workspace = session.workspace();
        (
            workspace.stage_states.capacity(),
            workspace.jacobian.capacity(),
            workspace.real_rhs.capacity(),
            workspace.complex_rhs_real.capacity(),
            workspace.complex_rhs_imag.capacity(),
        )
    };

    for index in 0..64 {
        let parameter = 1.0 + index as f64 * 0.01;
        continuation.rebind_parameters(&[parameter]).unwrap();
        continuation
            .restart(
                &[1.0 + parameter, 2.0 - parameter * 0.1],
                index as f64,
                index as f64 + 1.0,
            )
            .unwrap();

        let workspace_capacities = {
            let session = continuation.session();
            let workspace = session.workspace();
            (
                workspace.stage_states.capacity(),
                workspace.jacobian.capacity(),
                workspace.real_rhs.capacity(),
                workspace.complex_rhs_real.capacity(),
                workspace.complex_rhs_imag.capacity(),
            )
        };
        assert_eq!(
            continuation.initial_state_capacity(),
            initial_state_capacity
        );
        assert_eq!(continuation.initial_state().as_ptr(), initial_state_ptr);
        assert_eq!(workspace_capacities, initial_workspace_capacities);
    }

    assert_eq!(
        continuation.prepared().preparation_generation(),
        preparation_generation
    );
    assert_eq!(continuation.session().numeric_generation(), 1 + 64 * 2);
}

#[test]
fn prepared_symbolic_sessions_keep_parameter_bindings_isolated() {
    let callbacks = super::super::new::callbacks::PreparedSymbolicCallbacks::prepare(
        super::super::new::config::RadauAssembly::ExprLegacy,
        vec![Expr::Mul(
            Box::new(Expr::Var("p".to_string())),
            Box::new(Expr::Var("y1".to_string())),
        )],
        Some(vec![Expr::Var("p".to_string())]),
        "t",
        &["y1"],
        &["p"],
    )
    .unwrap();
    let mut first = callbacks.session();
    let mut second = callbacks.session();
    first.rebind_parameters(&[2.0]).unwrap();
    second.rebind_parameters(&[3.0]).unwrap();
    let mut first_output = [0.0];
    let mut second_output = [0.0];

    first
        .evaluate_residual(0.0, &[4.0], &mut first_output)
        .unwrap();
    second
        .evaluate_residual(0.0, &[4.0], &mut second_output)
        .unwrap();

    assert_eq!(first_output, [8.0]);
    assert_eq!(second_output, [12.0]);
    assert_eq!(first.parameters(), &[2.0]);
    assert_eq!(second.parameters(), &[3.0]);
    assert_ne!(first.parameters().as_ptr(), second.parameters().as_ptr());
}
