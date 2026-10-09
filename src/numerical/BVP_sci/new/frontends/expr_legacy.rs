//! ExprLegacy Lambdify frontend.
//!
//! Expr construction, differentiation and evaluator materialization belong
//! here.  The numerical core must receive prepared callbacks only.

use crate::numerical::BVP_sci::new::config::BvpSciExecutionPolicy;
use crate::numerical::BVP_sci::new::error::{BvpSciNewError, BvpSciStage};
use crate::numerical::BVP_sci::new::telemetry::{BvpSciTelemetry, BvpSciTelemetrySnapshot};
use crate::symbolic::symbolic_engine::Expr;
use std::collections::HashSet;
use std::sync::Arc;

type ScalarEvaluator = Arc<dyn Fn(&[f64]) -> f64 + Send + Sync>;

/// Prepared ExprLegacy residual and pointwise Jacobian evaluators.
///
/// Preparation owns symbolic differentiation and closure compilation. Runtime
/// methods require caller-owned argument and output buffers, so continuation
/// and Newton loops do not allocate a temporary `Vec` for every scalar call.
#[derive(Clone)]
pub struct ExprLegacyLambdifyPlan {
    independent_name: String,
    state_names: Vec<String>,
    parameter_names: Vec<String>,
    residuals: Vec<ScalarEvaluator>,
    jacobian: Vec<(usize, usize, ScalarEvaluator)>,
    jacobian_nnz: usize,
    telemetry: BvpSciTelemetry,
    execution_policy: BvpSciExecutionPolicy,
}

impl ExprLegacyLambdifyPlan {
    /// Differentiate and compile one symbolic BVP RHS once.
    pub fn prepare(
        equations: &[Expr],
        state_names: &[String],
        parameter_names: &[String],
        independent_name: impl Into<String>,
        telemetry: BvpSciTelemetry,
    ) -> Result<Self, BvpSciNewError> {
        let independent_name = independent_name.into();
        validate_names(&independent_name, state_names, parameter_names)?;
        if equations.is_empty() || equations.len() != state_names.len() {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::SymbolicPreparation,
                expected: state_names.len(),
                actual: equations.len(),
            });
        }

        let started = telemetry.start_timing();
        let mut argument_names = Vec::with_capacity(1 + state_names.len() + parameter_names.len());
        argument_names.push(independent_name.clone());
        argument_names.extend(state_names.iter().cloned());
        argument_names.extend(parameter_names.iter().cloned());
        let argument_refs: Vec<&str> = argument_names.iter().map(String::as_str).collect();

        // Residual closures are materialized before any solver is created.
        // This keeps symbolic work out of Newton and makes preparation timing
        // comparable with the AtomView route.
        let evaluator_started = telemetry.start_timing();
        let residuals = equations
            .iter()
            .map(|equation| {
                Arc::from(Expr::lambdify_borrowed_thread_safe(
                    equation,
                    argument_refs.as_slice(),
                ))
            })
            .collect::<Vec<ScalarEvaluator>>();
        telemetry.record_residual_evaluator_compilation(evaluator_started, residuals.len() as u64);

        // Differentiate in ExprLegacy exactly once. The resulting expressions
        // are then simplified and compiled into the fixed structural plan;
        // parameter changes only alter callback arguments later.
        let symbolic_started = telemetry.start_timing();
        let derivatives = equations
            .iter()
            .map(|equation| {
                state_names
                    .iter()
                    .map(|state_name| equation.diff(state_name).simplify())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<Vec<_>>>();
        telemetry.record_symbolic_jacobian(
            symbolic_started,
            (equations.len() * state_names.len()) as u64,
        );

        // Compile only nonzero state derivatives. Sparse/Banded consume this
        // stable list directly, while dense callbacks scatter it into a small
        // pointwise block without changing symbolic representation.
        let evaluator_started = telemetry.start_timing();
        let mut jacobian = Vec::with_capacity(equations.len() * state_names.len());
        let mut jacobian_nnz = 0;
        for (row, derivatives) in derivatives.iter().enumerate() {
            for (column, derivative) in derivatives.iter().enumerate() {
                if !derivative.is_zero() {
                    jacobian_nnz += 1;
                    jacobian.push((
                        row,
                        column,
                        Arc::from(Expr::lambdify_borrowed_thread_safe(
                            derivative,
                            argument_refs.as_slice(),
                        )),
                    ));
                }
            }
        }
        telemetry.record_jacobian_evaluator_compilation(evaluator_started, jacobian_nnz as u64);
        telemetry.record_preparation(started);

        Ok(Self {
            independent_name,
            state_names: state_names.to_vec(),
            parameter_names: parameter_names.to_vec(),
            residuals,
            jacobian,
            jacobian_nnz,
            telemetry,
            execution_policy: BvpSciExecutionPolicy::Sequential,
        })
    }

    /// Change only warm callback dispatch; symbolic preparation is unchanged.
    pub fn with_execution_policy(mut self, policy: BvpSciExecutionPolicy) -> Self {
        self.execution_policy = policy;
        self
    }

    /// Return the number of state equations in the prepared RHS.
    pub fn dimension(&self) -> usize {
        self.state_names.len()
    }

    /// Return the number of runtime parameters accepted by each evaluator.
    pub fn parameter_dimension(&self) -> usize {
        self.parameter_names.len()
    }

    pub fn independent_name(&self) -> &str {
        &self.independent_name
    }

    pub fn jacobian_nnz(&self) -> usize {
        self.jacobian_nnz
    }

    /// Fixed structural order used by sparse and banded native storage.
    pub fn jacobian_pattern(&self) -> impl ExactSizeIterator<Item = (usize, usize)> + '_ {
        self.jacobian.iter().map(|(row, column, _)| (*row, *column))
    }

    pub fn telemetry(&self) -> &BvpSciTelemetry {
        &self.telemetry
    }

    pub fn telemetry_snapshot(&self) -> BvpSciTelemetrySnapshot {
        self.telemetry.snapshot()
    }

    /// Evaluate all RHS equations into a caller-owned output buffer.
    pub fn evaluate_rhs(
        &self,
        x: f64,
        state: &[f64],
        parameters: &[f64],
        arguments: &mut [f64],
        output: &mut [f64],
    ) -> Result<(), BvpSciNewError> {
        self.fill_arguments(x, state, parameters, arguments)?;
        if output.len() != self.dimension() {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::ResidualCallback,
                expected: self.dimension(),
                actual: output.len(),
            });
        }
        let started = self.telemetry.start_timing();
        let evaluation_started = self.telemetry.start_timing();
        self.telemetry
            .record_dispatch(self.execution_policy, self.residuals.len());
        if self.residuals.len() > 1
            && self
            .execution_policy
            .should_parallel_with_tasks(self.residuals.len(), self.residuals.len())
        {
            use rayon::prelude::*;
            output
                .par_iter_mut()
                .zip(self.residuals.par_iter())
                .for_each(|(slot, evaluator)| *slot = evaluator(arguments));
        } else {
            for (slot, evaluator) in output.iter_mut().zip(&self.residuals) {
                *slot = evaluator(arguments);
            }
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(BvpSciNewError::NonFinite {
                stage: BvpSciStage::ResidualCallback,
            });
        }
        self.telemetry
            .record_residual_evaluation(evaluation_started);
        self.telemetry.record_output_writes(output.len());
        self.telemetry.record_rhs(started);
        Ok(())
    }

    /// Evaluate a dense pointwise Jacobian directly into caller-owned storage.
    pub fn evaluate_jacobian_dense(
        &self,
        x: f64,
        state: &[f64],
        parameters: &[f64],
        arguments: &mut [f64],
        output: &mut [f64],
    ) -> Result<(), BvpSciNewError> {
        self.fill_arguments(x, state, parameters, arguments)?;
        let dimension = self.dimension();
        let expected = dimension
            .checked_mul(dimension)
            .ok_or_else(|| BvpSciNewError::InvalidConfiguration("Jacobian size overflow".into()))?;
        if output.len() != expected {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::JacobianCallback,
                expected,
                actual: output.len(),
            });
        }
        output.fill(0.0);
        let started = self.telemetry.start_timing();
        let evaluation_started = self.telemetry.start_timing();
        self.telemetry
            .record_dispatch(self.execution_policy, self.jacobian.len());
        if self.jacobian.len() > 1
            && self
            .execution_policy
            .should_parallel_with_tasks(self.jacobian.len(), self.jacobian.len())
        {
            use rayon::prelude::*;
            let values = self
                .jacobian
                .par_iter()
                .map(|&(row, column, ref evaluator)| {
                    (row * dimension + column, evaluator(arguments))
                })
                .collect::<Vec<_>>();
            for (index, value) in values {
                if !value.is_finite() {
                    return Err(BvpSciNewError::NonFinite {
                        stage: BvpSciStage::JacobianCallback,
                    });
                }
                output[index] = value;
            }
        } else {
            for &(row, column, ref evaluator) in &self.jacobian {
                let value = evaluator(arguments);
                if !value.is_finite() {
                    return Err(BvpSciNewError::NonFinite {
                        stage: BvpSciStage::JacobianCallback,
                    });
                }
                output[row * dimension + column] = value;
            }
        }
        self.telemetry
            .record_jacobian_evaluation(evaluation_started);
        self.telemetry.record_output_writes(output.len());
        self.telemetry.record_jacobian(started);
        Ok(())
    }

    /// Evaluate only structural Jacobian values in the prepared pattern order.
    ///
    /// Sparse and banded backends can copy these values directly into their
    /// native workspaces without first materializing a dense matrix.
    pub fn evaluate_jacobian_values(
        &self,
        x: f64,
        state: &[f64],
        parameters: &[f64],
        arguments: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), BvpSciNewError> {
        self.fill_arguments(x, state, parameters, arguments)?;
        if values.len() != self.jacobian_nnz {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::JacobianCallback,
                expected: self.jacobian_nnz,
                actual: values.len(),
            });
        }
        let started = self.telemetry.start_timing();
        let evaluation_started = self.telemetry.start_timing();
        self.telemetry
            .record_dispatch(self.execution_policy, self.jacobian.len());
        if self.jacobian.len() > 1
            && self
            .execution_policy
            .should_parallel_with_tasks(self.jacobian.len(), self.jacobian.len())
        {
            use rayon::prelude::*;
            values
                .par_iter_mut()
                .zip(self.jacobian.par_iter())
                .for_each(|(slot, (_, _, evaluator))| *slot = evaluator(arguments));
        } else {
            for (slot, (_, _, evaluator)) in values.iter_mut().zip(&self.jacobian) {
                *slot = evaluator(arguments);
            }
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(BvpSciNewError::NonFinite {
                stage: BvpSciStage::JacobianCallback,
            });
        }
        self.telemetry
            .record_jacobian_evaluation(evaluation_started);
        self.telemetry.record_output_writes(values.len());
        self.telemetry.record_jacobian(started);
        Ok(())
    }

    fn fill_arguments(
        &self,
        x: f64,
        state: &[f64],
        parameters: &[f64],
        arguments: &mut [f64],
    ) -> Result<(), BvpSciNewError> {
        let binding_started = self.telemetry.start_timing();
        if !x.is_finite() {
            return Err(BvpSciNewError::NonFinite {
                stage: BvpSciStage::CallbackInput,
            });
        }
        if state.len() != self.dimension() || parameters.len() != self.parameter_dimension() {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::CallbackInput,
                expected: 1 + self.dimension() + self.parameter_dimension(),
                actual: 1 + state.len() + parameters.len(),
            });
        }
        let expected = 1 + state.len() + parameters.len();
        if arguments.len() != expected {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::ArgumentBuffer,
                expected,
                actual: arguments.len(),
            });
        }
        arguments[0] = x;
        if !state.is_empty() {
            self.telemetry.record_copy();
        }
        arguments[1..1 + state.len()].copy_from_slice(state);
        if !parameters.is_empty() {
            self.telemetry.record_copy();
        }
        arguments[1 + state.len()..].copy_from_slice(parameters);
        self.telemetry.record_argument_binding(binding_started);
        Ok(())
    }
}

fn validate_names(
    independent_name: &str,
    state_names: &[String],
    parameter_names: &[String],
) -> Result<(), BvpSciNewError> {
    let mut seen = HashSet::new();
    if independent_name.is_empty() {
        return Err(BvpSciNewError::InvalidConfiguration(
            "independent variable name must not be empty".into(),
        ));
    }
    seen.insert(independent_name);
    for name in state_names.iter().chain(parameter_names) {
        if name.is_empty() || !seen.insert(name.as_str()) {
            return Err(BvpSciNewError::InvalidConfiguration(format!(
                "duplicate or empty symbolic name: {name:?}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::ExprLegacyLambdifyPlan;
    use crate::numerical::BVP_sci::new::error::{BvpSciNewError, BvpSciStage};
    use crate::numerical::BVP_sci::new::{BvpSciTelemetry, BvpSciTelemetryMode};
    use crate::symbolic::symbolic_engine::Expr;

    #[test]
    fn exprlegacy_prepares_reusable_rhs_and_jacobian_buffers() {
        let plan = ExprLegacyLambdifyPlan::prepare(
            &[Expr::parse_expression("p*y + x")],
            &["y".into()],
            &["p".into()],
            "x",
            BvpSciTelemetry::timings(),
        )
        .expect("ExprLegacy preparation should succeed");
        let mut arguments = [0.0; 3];
        let mut rhs = [0.0; 1];
        let mut jacobian = [0.0; 1];

        plan.evaluate_rhs(2.0, &[3.0], &[4.0], &mut arguments, &mut rhs)
            .expect("RHS callback should succeed");
        plan.evaluate_jacobian_dense(2.0, &[3.0], &[4.0], &mut arguments, &mut jacobian)
            .expect("Jacobian callback should succeed");

        assert_eq!(rhs, [14.0]);
        assert_eq!(jacobian, [4.0]);
        assert_eq!(plan.jacobian_nnz(), 1);
        let snapshot = plan.telemetry_snapshot();
        assert_eq!(snapshot.mode, BvpSciTelemetryMode::Timings);
        assert_eq!(snapshot.rhs_calls, 1);
        assert_eq!(snapshot.jacobian_calls, 1);
        assert_eq!(snapshot.argument_bindings, 2);
        assert_eq!(snapshot.residual_evaluations, 1);
        assert_eq!(snapshot.jacobian_evaluations, 1);
        assert_eq!(snapshot.copies, 4);
        assert_eq!(snapshot.output_writes, 2);
        assert!(snapshot.binding_ms.is_some());
        assert!(snapshot.residual_evaluation_ms.is_some());
        assert!(snapshot.jacobian_evaluation_ms.is_some());
        assert!(snapshot.callback_ms.is_some());
    }

    #[test]
    fn callback_validation_returns_typed_stage_errors() {
        let plan = ExprLegacyLambdifyPlan::prepare(
            &[Expr::parse_expression("y")],
            &["y".into()],
            &[],
            "x",
            BvpSciTelemetry::disabled(),
        )
        .expect("ExprLegacy preparation should succeed");
        let mut arguments = [0.0; 2];
        let mut output = [0.0; 1];
        let error = plan
            .evaluate_rhs(0.0, &[], &[], &mut arguments, &mut output)
            .expect_err("wrong state shape should fail");
        assert!(matches!(
            error,
            BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::CallbackInput,
                ..
            }
        ));

        let error = plan
            .evaluate_rhs(f64::NAN, &[1.0], &[], &mut arguments, &mut output)
            .expect_err("non-finite input should fail");
        assert!(matches!(
            error,
            BvpSciNewError::NonFinite {
                stage: BvpSciStage::CallbackInput
            }
        ));
    }

    #[test]
    fn disabled_telemetry_has_no_runtime_measurements() {
        let plan = ExprLegacyLambdifyPlan::prepare(
            &[Expr::parse_expression("y")],
            &["y".into()],
            &[],
            "x",
            BvpSciTelemetry::disabled(),
        )
        .expect("ExprLegacy preparation should succeed");
        let mut arguments = [0.0; 2];
        let mut rhs = [0.0; 1];
        plan.evaluate_rhs(0.0, &[2.0], &[], &mut arguments, &mut rhs)
            .expect("RHS callback should succeed");
        let snapshot = plan.telemetry_snapshot();
        assert_eq!(snapshot.mode, BvpSciTelemetryMode::Off);
        assert_eq!(snapshot.rhs_calls, 0);
        assert_eq!(snapshot.callback_ms, None);
    }
}
