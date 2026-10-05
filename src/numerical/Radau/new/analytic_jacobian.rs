//! Analytic dense-Jacobian adapter for the native Radau API.
//!
//! Keeping this boundary separate from finite differences makes the
//! production contract explicit: an analytic callback is called once per
//! requested Jacobian and its nalgebra column-major result is copied into the
//! row-major buffer owned by the numerical core.

use nalgebra::DVector;

use super::callbacks::DenseJacobianCallback;
use super::error::{RadauError, RadauStage};
use super::native_callbacks::NativeJacobianFn;

/// Adapter for one user-supplied analytic dense Jacobian closure.
pub(crate) struct AnalyticJacobianCallback {
    callback: NativeJacobianFn,
}

impl AnalyticJacobianCallback {
    pub(crate) fn new(callback: NativeJacobianFn) -> Self {
        Self { callback }
    }
}

impl DenseJacobianCallback for AnalyticJacobianCallback {
    fn eval_into(&mut self, t: f64, y: &[f64], out: &mut [f64]) -> Result<(), RadauError> {
        let value = (self.callback)(t, &DVector::from_column_slice(y));
        let expected = y
            .len()
            .checked_mul(y.len())
            .ok_or(RadauError::WorkspaceSizeOverflow { dimension: y.len() })?;
        if value.nrows() != y.len() || value.ncols() != y.len() {
            return Err(RadauError::ShapeMismatch {
                stage: RadauStage::Jacobian,
                expected,
                actual: value.nrows().saturating_mul(value.ncols()),
            });
        }
        if value.iter().any(|entry| !entry.is_finite()) {
            return Err(RadauError::NonFiniteCallback {
                stage: RadauStage::Jacobian,
            });
        }
        if out.len() != expected {
            return Err(RadauError::ShapeMismatch {
                stage: RadauStage::Jacobian,
                expected,
                actual: out.len(),
            });
        }
        for row in 0..y.len() {
            for column in 0..y.len() {
                out[row * y.len() + column] = value[(row, column)];
            }
        }
        Ok(())
    }
}
