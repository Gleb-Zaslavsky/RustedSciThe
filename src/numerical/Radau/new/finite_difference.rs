//! Forward-difference Jacobian fallback for native Radau callbacks.
//!
//! The adapter is deliberately stateful: its perturbation vector is reused
//! across columns and repeated solves. The callback ABI still returns
//! nalgebra vectors, so only the public-boundary conversion remains per probe.

use nalgebra::DVector;

use super::callbacks::{DenseJacobianCallback, validate_callback_output};
use super::error::{RadauError, RadauStage};
use super::native_callbacks::NativeResidualFn;

/// Component-wise forward-difference Jacobian fallback.
pub(crate) struct FiniteDifferenceJacobianCallback {
    residual: NativeResidualFn,
    atol: f64,
    state: Vec<f64>,
    probes: Option<u64>,
}

impl FiniteDifferenceJacobianCallback {
    pub(crate) fn new(residual: NativeResidualFn, atol: f64, track_probes: bool) -> Self {
        Self {
            residual,
            atol,
            state: Vec::new(),
            probes: track_probes.then_some(0),
        }
    }

    /// Number of residual evaluations performed internally by FD Jacobians.
    pub(crate) fn probes(&self) -> u64 {
        self.probes.unwrap_or(0)
    }
}

impl DenseJacobianCallback for FiniteDifferenceJacobianCallback {
    fn eval_into(&mut self, t: f64, y: &[f64], out: &mut [f64]) -> Result<(), RadauError> {
        let dimension = y.len();
        let expected = dimension
            .checked_mul(dimension)
            .ok_or(RadauError::WorkspaceSizeOverflow { dimension })?;
        if out.len() != expected {
            return Err(RadauError::ShapeMismatch {
                stage: RadauStage::Jacobian,
                expected,
                actual: out.len(),
            });
        }
        self.state.resize(y.len(), 0.0);
        self.state.copy_from_slice(y);
        let base = (self.residual)(t, &DVector::from_column_slice(y));
        if let Some(probes) = &mut self.probes {
            *probes = probes.saturating_add(1);
        }
        validate_callback_output(RadauStage::Residual, dimension, base.as_slice())?;
        let eps = f64::EPSILON.sqrt();
        for column in 0..dimension {
            let scale = y[column].abs().max(self.atol).max(1.0);
            let mut h = eps * scale;
            if h == 0.0 || !h.is_finite() {
                h = eps;
            }
            self.state[column] = y[column] + h;
            let perturbed = (self.residual)(t, &DVector::from_column_slice(&self.state));
            if let Some(probes) = &mut self.probes {
                *probes = probes.saturating_add(1);
            }
            self.state[column] = y[column];
            validate_callback_output(RadauStage::Residual, dimension, perturbed.as_slice())?;
            for row in 0..dimension {
                out[row * dimension + column] = (perturbed[row] - base[row]) / h;
            }
        }
        validate_callback_output(RadauStage::Jacobian, expected, out)
    }
}
