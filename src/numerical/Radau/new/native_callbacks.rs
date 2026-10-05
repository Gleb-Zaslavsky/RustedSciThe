//! Adapters for the public native Radau callback API.
//!
//! The numerical core deliberately consumes caller-owned slices.  The public
//! compatibility API uses nalgebra vectors and matrices, so this module is the
//! only boundary where those values are converted into the core's row-major
//! buffers.  Analytic and finite-difference Jacobians are separate types to
//! keep their contracts and costs explicit.

use std::sync::Arc;

use nalgebra::{DMatrix, DVector};

use super::callbacks::{validate_callback_output, ResidualCallback};
use super::error::{RadauError, RadauStage};

pub(crate) type NativeResidualFn = Arc<dyn Fn(f64, &DVector<f64>) -> DVector<f64> + Send + Sync>;
pub(crate) type NativeJacobianFn = Arc<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64> + Send + Sync>;

/// Adapter for one user-supplied native residual closure.
pub(crate) struct NativeResidualCallback {
    callback: NativeResidualFn,
}

impl NativeResidualCallback {
    pub(crate) fn new(callback: NativeResidualFn) -> Self {
        Self { callback }
    }
}

impl ResidualCallback for NativeResidualCallback {
    fn eval(&mut self, t: f64, y: &[f64], out: &mut [f64]) -> Result<(), RadauError> {
        let value = (self.callback)(t, &DVector::from_column_slice(y));
        validate_callback_output(RadauStage::Residual, y.len(), value.as_slice())?;
        if out.len() != value.len() {
            return Err(RadauError::ShapeMismatch {
                stage: RadauStage::Residual,
                expected: out.len(),
                actual: value.len(),
            });
        }
        out.copy_from_slice(value.as_slice());
        Ok(())
    }
}
