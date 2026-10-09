//! User-supplied numerical callbacks for the new `BVP_sci` solver.
//!
//! This frontend deliberately has no symbolic preparation. The user provides
//! the ODE residual, and may optionally provide its dense pointwise Jacobian.
//! When the Jacobian is absent, the plan computes a forward finite difference
//! into caller-owned workspace. This keeps the numerical route independent of
//! ExprLegacy, AtomView and AOT while sharing the same collocation core.

use std::sync::Arc;

use super::{
    config::BvpSciExecutionPolicy,
    error::{BvpSciNewError, BvpSciStage},
    telemetry::{BvpSciTelemetry, BvpSciTelemetrySnapshot},
};

/// Pointwise ODE residual callback `f(x, y, p)`.
pub type NumericalRhsCallback =
    dyn Fn(f64, &[f64], &[f64], &mut [f64]) -> Result<(), String> + Send + Sync;

/// Pointwise dense state Jacobian callback `df/dy` in row-major order.
pub type NumericalJacobianCallback =
    dyn Fn(f64, &[f64], &[f64], &mut [f64]) -> Result<(), String> + Send + Sync;

/// Optional pointwise parameter Jacobian callback `df/dp` in row-major order.
pub type NumericalParameterJacobianCallback =
    dyn Fn(f64, &[f64], &[f64], &mut [f64]) -> Result<(), String> + Send + Sync;

/// Prepared numerical frontend used by the Dense BVP route.
#[derive(Clone)]
pub struct BvpSciNumericalPlan {
    dimension: usize,
    parameter_dimension: usize,
    rhs: Arc<NumericalRhsCallback>,
    rhs_jacobian: Option<Arc<NumericalJacobianCallback>>,
    rhs_parameter_jacobian: Option<Arc<NumericalParameterJacobianCallback>>,
    telemetry: BvpSciTelemetry,
    execution_policy: BvpSciExecutionPolicy,
}

impl BvpSciNumericalPlan {
    /// Create a residual-only plan. State Jacobians use finite differences.
    pub fn new<F>(
        dimension: usize,
        parameter_dimension: usize,
        rhs: F,
        telemetry: BvpSciTelemetry,
    ) -> Result<Self, BvpSciNewError>
    where
        F: Fn(f64, &[f64], &[f64], &mut [f64]) -> Result<(), String> + Send + Sync + 'static,
    {
        if dimension == 0 {
            return Err(BvpSciNewError::InvalidConfiguration(
                "numerical BVP dimension must be positive".into(),
            ));
        }
        Ok(Self {
            dimension,
            parameter_dimension,
            rhs: Arc::new(rhs),
            rhs_jacobian: None,
            rhs_parameter_jacobian: None,
            telemetry,
            execution_policy: BvpSciExecutionPolicy::Sequential,
        })
    }

    /// Attach an analytical dense `df/dy` callback without rebuilding the plan.
    pub fn with_rhs_jacobian<F>(mut self, jacobian: F) -> Self
    where
        F: Fn(f64, &[f64], &[f64], &mut [f64]) -> Result<(), String> + Send + Sync + 'static,
    {
        self.rhs_jacobian = Some(Arc::new(jacobian));
        self
    }

    /// Attach an analytical dense `df/dp` callback for parameter continuation.
    pub fn with_rhs_parameter_jacobian<F>(mut self, jacobian: F) -> Self
    where
        F: Fn(f64, &[f64], &[f64], &mut [f64]) -> Result<(), String> + Send + Sync + 'static,
    {
        self.rhs_parameter_jacobian = Some(Arc::new(jacobian));
        self
    }

    pub fn with_execution_policy(mut self, policy: BvpSciExecutionPolicy) -> Self {
        self.execution_policy = policy;
        self
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn parameter_dimension(&self) -> usize {
        self.parameter_dimension
    }

    pub fn has_rhs_jacobian(&self) -> bool {
        self.rhs_jacobian.is_some()
    }

    pub fn has_rhs_parameter_jacobian(&self) -> bool {
        self.rhs_parameter_jacobian.is_some()
    }

    pub fn jacobian_nnz(&self) -> usize {
        self.dimension * self.dimension
    }

    pub fn jacobian_pattern(&self) -> Vec<(usize, usize)> {
        (0..self.dimension)
            .flat_map(|row| (0..self.dimension).map(move |column| (row, column)))
            .collect()
    }

    pub fn telemetry(&self) -> &BvpSciTelemetry {
        &self.telemetry
    }

    pub fn telemetry_snapshot(&self) -> BvpSciTelemetrySnapshot {
        self.telemetry.snapshot()
    }

    /// Evaluate the residual using only caller-owned argument/output buffers.
    pub fn evaluate_rhs(
        &self,
        x: f64,
        state: &[f64],
        parameters: &[f64],
        arguments: &mut [f64],
        output: &mut [f64],
    ) -> Result<(), BvpSciNewError> {
        self.fill_arguments(x, state, parameters, arguments)?;
        self.evaluate_raw(
            arguments[0],
            &arguments[1..1 + self.dimension],
            &arguments[1 + self.dimension..],
            output,
        )
    }

    /// Evaluate an analytical Jacobian, or finite-difference it in place.
    ///
    /// `scratch` must contain one slot for a scalar problem and `2 *
    /// dimension` slots for larger problems. For dimensions two and above
    /// the existing `n*n` Jacobian scratch is reused for both the
    /// baseline and trial RHS values; the scalar case uses the output slot
    /// directly and therefore does not allocate a special-case buffer.
    pub fn evaluate_jacobian_dense_with_scratch(
        &self,
        x: f64,
        state: &[f64],
        parameters: &[f64],
        arguments: &mut [f64],
        output: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), BvpSciNewError> {
        let expected = self
            .dimension
            .checked_mul(self.dimension)
            .ok_or_else(|| BvpSciNewError::InvalidConfiguration("Jacobian size overflow".into()))?;
        if output.len() != expected {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::JacobianCallback,
                expected,
                actual: output.len(),
            });
        }
        let scratch_required = if self.dimension == 1 {
            1
        } else {
            self.dimension.checked_mul(2).ok_or_else(|| {
                BvpSciNewError::InvalidConfiguration("Jacobian scratch size overflow".into())
            })?
        };
        if scratch.len() < scratch_required {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::JacobianCallback,
                expected: scratch_required,
                actual: scratch.len(),
            });
        }
        self.fill_arguments(x, state, parameters, arguments)?;
        let started = self.telemetry.start_timing();

        if let Some(jacobian) = &self.rhs_jacobian {
            let result = jacobian(x, state, parameters, output).map_err(|message| {
                BvpSciNewError::Callback {
                    stage: BvpSciStage::JacobianCallback,
                    message,
                }
            });
            result?;
            self.validate_finite(output, BvpSciStage::JacobianCallback)?;
            self.telemetry.record_jacobian_evaluation(started);
            self.telemetry.record_output_writes(output.len());
            self.telemetry.record_jacobian(started);
            return Ok(());
        }

        let n = self.dimension;
        self.evaluate_raw(x, state, parameters, &mut scratch[..n])?;
        for column in 0..n {
            let nominal_delta = f64::EPSILON.sqrt() * (1.0 + state[column].abs());
            let base = state[column];
            arguments[1 + column] = base + nominal_delta;
            let delta = arguments[1 + column] - base;
            if delta == 0.0 || !delta.is_finite() {
                return Err(BvpSciNewError::InvalidConfiguration(
                    "state finite-difference step is not representable".into(),
                ));
            }
            let trial_state = &arguments[1..1 + n];
            if n == 1 {
                self.evaluate_raw(x, trial_state, parameters, &mut output[..n])?;
                output[0] = (output[0] - scratch[0]) / delta;
            } else {
                self.evaluate_raw(x, trial_state, parameters, &mut scratch[n..2 * n])?;
                for row in 0..n {
                    output[row * n + column] = (scratch[n + row] - scratch[row]) / delta;
                }
            }
            // Restore the exact caller value instead of subtracting a rounded
            // increment from the perturbed representation.
            arguments[1 + column] = base;
            self.telemetry.record_finite_difference_probe();
        }
        self.validate_finite(output, BvpSciStage::JacobianCallback)?;
        self.telemetry.record_jacobian(started);
        self.telemetry.record_output_writes(output.len());
        Ok(())
    }

    /// Evaluate `df/dp`; return `false` when the caller must use FD fallback.
    pub fn evaluate_parameter_jacobian(
        &self,
        x: f64,
        state: &[f64],
        parameters: &[f64],
        arguments: &mut [f64],
        output: &mut [f64],
    ) -> Result<bool, BvpSciNewError> {
        let expected = self
            .dimension
            .checked_mul(self.parameter_dimension)
            .ok_or_else(|| {
                BvpSciNewError::InvalidConfiguration("parameter Jacobian size overflow".into())
            })?;
        if output.len() != expected {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::JacobianCallback,
                expected,
                actual: output.len(),
            });
        }
        let Some(jacobian) = &self.rhs_parameter_jacobian else {
            return Ok(false);
        };
        self.fill_arguments(x, state, parameters, arguments)?;
        let started = self.telemetry.start_timing();
        jacobian(x, state, parameters, output).map_err(|message| BvpSciNewError::Callback {
            stage: BvpSciStage::JacobianCallback,
            message,
        })?;
        self.validate_finite(output, BvpSciStage::JacobianCallback)?;
        self.telemetry.record_jacobian_evaluation(started);
        self.telemetry.record_output_writes(output.len());
        Ok(true)
    }

    fn evaluate_raw(
        &self,
        x: f64,
        state: &[f64],
        parameters: &[f64],
        output: &mut [f64],
    ) -> Result<(), BvpSciNewError> {
        if state.len() != self.dimension {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::ResidualCallback,
                expected: self.dimension,
                actual: state.len(),
            });
        }
        if parameters.len() != self.parameter_dimension || output.len() != self.dimension {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::ResidualCallback,
                expected: self.dimension,
                actual: output.len(),
            });
        }
        if !x.is_finite()
            || state
                .iter()
                .chain(parameters)
                .any(|value| !value.is_finite())
        {
            return Err(BvpSciNewError::NonFinite {
                stage: BvpSciStage::CallbackInput,
            });
        }
        let started = self.telemetry.start_timing();
        let result =
            (self.rhs)(x, state, parameters, output).map_err(|message| BvpSciNewError::Callback {
                stage: BvpSciStage::ResidualCallback,
                message,
            });
        result?;
        self.validate_finite(output, BvpSciStage::ResidualCallback)?;
        self.telemetry
            .record_dispatch(BvpSciExecutionPolicy::Sequential, 1);
        self.telemetry.record_residual_evaluation(started);
        self.telemetry.record_rhs(started);
        self.telemetry.record_output_writes(output.len());
        Ok(())
    }

    fn fill_arguments(
        &self,
        x: f64,
        state: &[f64],
        parameters: &[f64],
        arguments: &mut [f64],
    ) -> Result<(), BvpSciNewError> {
        let expected = 1 + self.dimension + self.parameter_dimension;
        if arguments.len() != expected {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::ArgumentBuffer,
                expected,
                actual: arguments.len(),
            });
        }
        if state.len() != self.dimension {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::CallbackInput,
                expected: self.dimension,
                actual: state.len(),
            });
        }
        if parameters.len() != self.parameter_dimension {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::CallbackInput,
                expected: self.parameter_dimension,
                actual: parameters.len(),
            });
        }
        arguments[0] = x;
        arguments[1..1 + self.dimension].copy_from_slice(state);
        arguments[1 + self.dimension..].copy_from_slice(parameters);
        Ok(())
    }

    fn validate_finite(&self, values: &[f64], stage: BvpSciStage) -> Result<(), BvpSciNewError> {
        if values.iter().all(|value| value.is_finite()) {
            Ok(())
        } else {
            Err(BvpSciNewError::NonFinite { stage })
        }
    }
}
