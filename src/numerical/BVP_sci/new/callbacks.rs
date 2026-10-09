//! Callback adapters shared by numerical and Lambdify frontends.
//!
//! Boundary callbacks are deliberately separate from the symbolic RHS plan:
//! boundary conditions are not part of the ODE expression and may be supplied
//! by a user closure.  The adapter owns output-buffer and finite-value checks,
//! so the numerical core does not need a second error taxonomy.

use super::error::{BvpSciNewError, BvpSciStage};
use super::telemetry::BvpSciTelemetry;
use std::sync::Arc;

/// Boundary-condition callback used by the new collocation solver.
pub trait BvpSciBoundary: Send + Sync {
    /// Evaluate `bc(ya, yb, p)` into the caller-owned output buffer.
    fn evaluate(
        &self,
        ya: &[f64],
        yb: &[f64],
        parameters: &[f64],
        output: &mut [f64],
    ) -> Result<(), BvpSciNewError>;

    /// Number of residuals returned by the boundary callback.
    fn residual_dimension(&self) -> usize;

    /// Evaluate analytical boundary Jacobians when the caller supplied them.
    ///
    /// The three outputs are row-major `(dbc/dya, dbc/dyb, dbc/dp)` blocks.
    /// Returning `false` selects the numerical core's finite-difference
    /// fallback without making symbolic frontends implement a dummy path.
    fn evaluate_jacobian(
        &self,
        _ya: &[f64],
        _yb: &[f64],
        _parameters: &[f64],
        _dya: &mut [f64],
        _dyb: &mut [f64],
        _dp: &mut [f64],
    ) -> Result<bool, BvpSciNewError> {
        Ok(false)
    }

    /// Return the callback-owned telemetry handle when one exists.
    ///
    /// The numerical core uses this identity to avoid counting a
    /// `BvpSciBoundaryCallbacks` invocation twice when the caller deliberately
    /// shares the solver telemetry handle with the boundary adapter.
    fn telemetry(&self) -> Option<&BvpSciTelemetry> {
        None
    }
}

type BoundaryFn = dyn Fn(&[f64], &[f64], &[f64], &mut [f64]) -> Result<(), String> + Send + Sync;
type BoundaryJacobianFn = dyn Fn(
        &[f64],
        &[f64],
        &[f64],
        &mut [f64],
        &mut [f64],
        &mut [f64],
    ) -> Result<(), String>
    + Send
    + Sync;

/// Closure-backed boundary callback with reusable output storage.
#[derive(Clone)]
pub struct BvpSciBoundaryCallbacks {
    dimension: usize,
    callback: Arc<BoundaryFn>,
    jacobian: Option<Arc<BoundaryJacobianFn>>,
    telemetry: BvpSciTelemetry,
}

impl BvpSciBoundaryCallbacks {
    /// Build a callback.  The closure must write exactly `dimension` values.
    pub fn new<F>(dimension: usize, callback: F, telemetry: BvpSciTelemetry) -> Self
    where
        F: Fn(&[f64], &[f64], &[f64], &mut [f64]) -> Result<(), String> + Send + Sync + 'static,
    {
        Self {
            dimension,
            callback: Arc::new(callback),
            jacobian: None,
            telemetry,
        }
    }

    /// Build a boundary callback with analytical endpoint Jacobians.
    pub fn new_with_jacobian<F, J>(
        dimension: usize,
        callback: F,
        jacobian: J,
        telemetry: BvpSciTelemetry,
    ) -> Self
    where
        F: Fn(&[f64], &[f64], &[f64], &mut [f64]) -> Result<(), String> + Send + Sync + 'static,
        J: Fn(
                &[f64],
                &[f64],
                &[f64],
                &mut [f64],
                &mut [f64],
                &mut [f64],
            ) -> Result<(), String>
            + Send
            + Sync
            + 'static,
    {
        Self {
            dimension,
            callback: Arc::new(callback),
            jacobian: Some(Arc::new(jacobian)),
            telemetry,
        }
    }

    pub fn telemetry(&self) -> &BvpSciTelemetry {
        &self.telemetry
    }
}

/// Evaluate one boundary callback and attribute it to the solver lifecycle.
///
/// Custom boundary implementations normally return `None` from
/// `BvpSciBoundary::telemetry`, so their calls are recorded here.  The built-in
/// closure adapter reports through its own handle; when that is the same
/// handle as the plan's, this helper avoids a duplicate count.
pub(crate) fn evaluate_boundary(
    boundary: &dyn BvpSciBoundary,
    solver_telemetry: &BvpSciTelemetry,
    ya: &[f64],
    yb: &[f64],
    parameters: &[f64],
    output: &mut [f64],
) -> Result<(), BvpSciNewError> {
    let started = solver_telemetry.start_timing();
    let result = boundary.evaluate(ya, yb, parameters, output);
    let already_recorded = boundary
        .telemetry()
        .is_some_and(|telemetry| telemetry.shares_storage(solver_telemetry));
    if !already_recorded {
        solver_telemetry.record_boundary(started);
    }
    result
}

impl BvpSciBoundary for BvpSciBoundaryCallbacks {
    fn evaluate(
        &self,
        ya: &[f64],
        yb: &[f64],
        parameters: &[f64],
        output: &mut [f64],
    ) -> Result<(), BvpSciNewError> {
        if output.len() != self.dimension {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::BoundaryCallback,
                expected: self.dimension,
                actual: output.len(),
            });
        }
        if ya
            .iter()
            .chain(yb)
            .chain(parameters)
            .any(|value| !value.is_finite())
        {
            return Err(BvpSciNewError::NonFinite {
                stage: BvpSciStage::BoundaryCallback,
            });
        }
        let started = self.telemetry.start_timing();
        let callback_result = (self.callback)(ya, yb, parameters, output).map_err(|message| {
            BvpSciNewError::Callback {
                stage: BvpSciStage::BoundaryCallback,
                message,
            }
        });
        let result = callback_result.and_then(|()| {
            if output.iter().any(|value| !value.is_finite()) {
                return Err(BvpSciNewError::NonFinite {
                    stage: BvpSciStage::BoundaryCallback,
                });
            }
            self.telemetry.record_output_writes(output.len());
            Ok(())
        });
        self.telemetry.record_boundary(started);
        result
    }

    fn residual_dimension(&self) -> usize {
        self.dimension
    }

    fn evaluate_jacobian(
        &self,
        ya: &[f64],
        yb: &[f64],
        parameters: &[f64],
        dya: &mut [f64],
        dyb: &mut [f64],
        dp: &mut [f64],
    ) -> Result<bool, BvpSciNewError> {
        let Some(jacobian) = &self.jacobian else {
            return Ok(false);
        };
        let expected_state = self.dimension.checked_mul(ya.len()).ok_or_else(|| {
            BvpSciNewError::InvalidConfiguration("boundary Jacobian size overflow".into())
        })?;
        let expected_parameter = self.dimension.checked_mul(parameters.len()).ok_or_else(|| {
            BvpSciNewError::InvalidConfiguration("boundary parameter Jacobian size overflow".into())
        })?;
        if dya.len() != expected_state || dyb.len() != expected_state || dp.len() != expected_parameter {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::JacobianCallback,
                expected: expected_state,
                actual: dya.len().max(dyb.len()).max(dp.len()),
            });
        }
        let started = self.telemetry.start_timing();
        jacobian(ya, yb, parameters, dya, dyb, dp).map_err(|message| {
            BvpSciNewError::Callback {
                stage: BvpSciStage::JacobianCallback,
                message,
            }
        })?;
        if dya
            .iter()
            .chain(dyb.iter())
            .chain(dp.iter())
            .any(|value| !value.is_finite())
        {
            return Err(BvpSciNewError::NonFinite {
                stage: BvpSciStage::JacobianCallback,
            });
        }
        self.telemetry.record_jacobian_evaluation(started);
        self.telemetry.record_output_writes(dya.len() + dyb.len() + dp.len());
        Ok(true)
    }

    fn telemetry(&self) -> Option<&BvpSciTelemetry> {
        Some(&self.telemetry)
    }
}

#[cfg(test)]
mod tests {
    use super::{BvpSciBoundary, BvpSciBoundaryCallbacks};
    use crate::numerical::BVP_sci::new::{BvpSciNewError, BvpSciStage, BvpSciTelemetry};

    #[test]
    fn boundary_callback_rejects_non_finite_residual_output_with_stage() {
        let boundary = BvpSciBoundaryCallbacks::new(
            1,
            |_ya, _yb, _parameters, output| {
                output[0] = f64::NAN;
                Ok(())
            },
            BvpSciTelemetry::disabled(),
        );
        let error = boundary
            .evaluate(&[0.0], &[0.0], &[], &mut [0.0])
            .expect_err("non-finite boundary output must be rejected");

        assert!(matches!(
            error,
            BvpSciNewError::NonFinite {
                stage: BvpSciStage::BoundaryCallback
            }
        ));
    }

    #[test]
    fn analytic_boundary_jacobian_rejects_non_finite_output_with_stage() {
        let boundary = BvpSciBoundaryCallbacks::new_with_jacobian(
            1,
            |_ya, _yb, _parameters, output| {
                output[0] = 0.0;
                Ok(())
            },
            |_ya, _yb, _parameters, dya, _dyb, _dp| {
                dya[0] = f64::INFINITY;
                Ok(())
            },
            BvpSciTelemetry::disabled(),
        );
        let error = boundary
            .evaluate_jacobian(
                &[0.0],
                &[0.0],
                &[],
                &mut [0.0],
                &mut [0.0],
                &mut [],
            )
            .expect_err("non-finite analytical boundary Jacobian must be rejected");

        assert!(matches!(
            error,
            BvpSciNewError::NonFinite {
                stage: BvpSciStage::JacobianCallback
            }
        ));
    }
}
