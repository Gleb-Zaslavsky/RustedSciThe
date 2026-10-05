//! Typed-error stories for the new Radau numerical core.
//!
//! These tests deliberately exercise invalid inputs through `try_*` paths.
//! A malformed callback or configuration must become a classified error, not
//! a panic or an implicit dense fallback.

use super::super::new::callbacks::validate_callback_output;
use super::super::new::config::{RadauConfig, RadauExecution, RadauMatrixLayout};
use super::super::new::error::{RadauConfigError, RadauError, RadauStage};
use super::super::new::solver::try_solve_dense_with_callbacks;

#[test]
fn invalid_configuration_is_classified_before_numerical_work() {
    let cases = [
        (
            RadauConfig {
                t_bound: 0.0,
                ..RadauConfig::default()
            },
            RadauConfigError::InvalidTimeBounds,
        ),
        (
            RadauConfig {
                rtol: 0.0,
                ..RadauConfig::default()
            },
            RadauConfigError::InvalidRelativeTolerance,
        ),
        (
            RadauConfig {
                atol: f64::NAN,
                ..RadauConfig::default()
            },
            RadauConfigError::InvalidAbsoluteTolerance,
        ),
        (
            RadauConfig {
                first_step: Some(0.0),
                ..RadauConfig::default()
            },
            RadauConfigError::InvalidFirstStep,
        ),
        (
            RadauConfig {
                max_steps: 0,
                ..RadauConfig::default()
            },
            RadauConfigError::EmptyIterationBudget,
        ),
        (
            RadauConfig {
                matrix_layout: RadauMatrixLayout::Banded { lower: 0, upper: 0 },
                ..RadauConfig::default()
            },
            RadauConfigError::EmptyBandwidth,
        ),
        (
            RadauConfig {
                execution: RadauExecution::Native,
                assembly: Some(super::super::new::config::RadauAssembly::ExprLegacy),
                ..RadauConfig::default()
            },
            RadauConfigError::NativeWithSymbolicAssembly,
        ),
    ];

    for (config, expected) in cases {
        assert!(
            matches!(config.validate(), Err(RadauError::InvalidConfiguration(error)) if error == expected)
        );
    }
}

#[test]
fn callback_shape_and_non_finite_outputs_are_typed_errors() {
    let config = RadauConfig {
        t_bound: 0.1,
        first_step: Some(0.05),
        ..RadauConfig::default()
    };
    let shape_error = validate_callback_output(RadauStage::Residual, 1, &[]).unwrap_err();
    assert!(matches!(
        shape_error,
        RadauError::ShapeMismatch {
            stage: RadauStage::Residual,
            expected: 1,
            actual: 0
        }
    ));

    let mut non_finite = |_t: f64, _state: &[f64], output: &mut [f64]| {
        output.fill(f64::NAN);
        Ok::<(), RadauError>(())
    };
    let mut jacobian = |_t: f64, _state: &[f64], output: &mut [f64]| {
        output.fill(0.0);
        Ok::<(), RadauError>(())
    };
    let non_finite_error =
        try_solve_dense_with_callbacks(&config, &mut non_finite, &mut jacobian, &[1.0])
            .unwrap_err();
    assert!(matches!(
        non_finite_error,
        RadauError::NonFiniteCallback {
            stage: RadauStage::Residual
        }
    ));
}

#[test]
fn callback_failure_preserves_stage_and_message() {
    let config = RadauConfig {
        t_bound: 0.1,
        first_step: Some(0.05),
        ..RadauConfig::default()
    };
    let mut residual = |_t: f64, _state: &[f64], _output: &mut [f64]| {
        Err::<(), _>(RadauError::Callback {
            stage: RadauStage::Residual,
            message: "fixture failure".to_string(),
        })
    };
    let mut jacobian = |_t: f64, _state: &[f64], output: &mut [f64]| {
        output.fill(0.0);
        Ok::<(), RadauError>(())
    };
    let error =
        try_solve_dense_with_callbacks(&config, &mut residual, &mut jacobian, &[1.0]).unwrap_err();
    assert!(matches!(
        error,
        RadauError::Callback {
            stage: RadauStage::Residual,
            message
        } if message == "fixture failure"
    ));
}

#[test]
fn invalid_initial_state_is_rejected_without_a_panic_path() {
    let config = RadauConfig {
        t_bound: 0.1,
        first_step: Some(0.05),
        ..RadauConfig::default()
    };
    let mut residual = |_t: f64, state: &[f64], output: &mut [f64]| {
        output[0] = -state[0];
        Ok::<(), RadauError>(())
    };
    let mut jacobian = |_t: f64, _state: &[f64], output: &mut [f64]| {
        output[0] = -1.0;
        Ok::<(), RadauError>(())
    };
    let error =
        try_solve_dense_with_callbacks(&config, &mut residual, &mut jacobian, &[]).unwrap_err();
    assert!(matches!(
        error,
        RadauError::ShapeMismatch {
            stage: RadauStage::Preparation,
            expected: 1,
            actual: 0
        }
    ));
}
