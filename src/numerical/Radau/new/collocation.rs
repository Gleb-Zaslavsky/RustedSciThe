//! Radau collocation coefficients and stage equations.
//!
//! Numerical constants and the nonlinear stage residual belong here, separate
//! from lifecycle, backend, and telemetry concerns.

use super::callbacks::validate_callback_output;
use super::{callbacks::ResidualCallback, coefficients::RadauIia5, error::RadauError};

/// Initialize the three collocation stage states from the current state.
pub(crate) fn initialize_stage_states(
    coefficients: &RadauIia5,
    t: f64,
    h: f64,
    y: &[f64],
    base_rhs: &[f64],
    stage_states: &mut [f64],
) {
    let dimension = y.len();
    for stage in 0..3 {
        let offset = stage * dimension;
        for index in 0..dimension {
            stage_states[offset + index] = y[index] + coefficients.c[stage] * h * base_rhs[index];
        }
    }
    let _ = t;
}

/// Evaluate all collocation residuals into preallocated stage storage.
pub(crate) fn evaluate_stage_residuals<C: ResidualCallback>(
    coefficients: &RadauIia5,
    callback: &mut C,
    t: f64,
    h: f64,
    y: &[f64],
    stage_states: &[f64],
    stage_rhs: &mut [f64],
    residual: &mut [f64],
) -> Result<(), RadauError> {
    let dimension = y.len();
    for stage in 0..3 {
        let offset = stage * dimension;
        callback.eval(
            t + coefficients.c[stage] * h,
            &stage_states[offset..offset + dimension],
            &mut stage_rhs[offset..offset + dimension],
        )?;
        validate_callback_output(
            super::error::RadauStage::Residual,
            dimension,
            &stage_rhs[offset..offset + dimension],
        )?;
    }
    for stage in 0..3 {
        let offset = stage * dimension;
        for index in 0..dimension {
            let mut value = stage_states[offset + index] - y[index];
            for other_stage in 0..3 {
                value -= h
                    * coefficients.a[stage][other_stage]
                    * stage_rhs[other_stage * dimension + index];
            }
            residual[offset + index] = value;
        }
    }
    Ok(())
}
