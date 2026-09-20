use std::borrow::Cow;

use nalgebra::{DMatrix, DVector};

use crate::numerical::Nonlinear_systems::engine::{
    IterationState, MethodWorkspace, NonlinearMethod, RuntimeDiagnostics, SolveOptions,
    StepOutcome, eval_residual_with_runtime, measure_linear_system_operation_owned,
};
use crate::numerical::Nonlinear_systems::error::{SolveError, TerminationReason};
use crate::numerical::Nonlinear_systems::problem::JacobianProvider;

/// Identity-damped Levenberg-Marquardt with feasible residual-decrease
/// backtracking.
///
/// This is intentionally a separate method from the canonical MINPACK and
/// Nielsen implementations. It uses identity damping, strict
/// residual-decrease backtracking, `lambda *= 0.3` after acceptance, and
/// `lambda *= 10` after a failed line search. The generic engine still owns
/// the root convergence contract and returns `Converged` only after the
/// accepted state is checked.
#[derive(Debug, Clone, Copy)]
pub struct BacktrackingLevenbergMarquardtMethod {
    /// Initial identity-damping parameter.
    pub lambda_init: f64,
    /// Damping multiplier after an accepted step.
    pub lambda_decrease: f64,
    /// Damping multiplier after a failed line search.
    pub lambda_increase: f64,
    /// Smallest backtracking factor considered for a trial point.
    pub alpha_min: f64,
    /// Safety cap for the damping parameter.
    pub max_lambda: f64,
}

impl Default for BacktrackingLevenbergMarquardtMethod {
    fn default() -> Self {
        Self {
            lambda_init: 1.0e-3,
            lambda_decrease: 0.3,
            lambda_increase: 10.0,
            alpha_min: 1.0e-6,
            max_lambda: 1.0e15,
        }
    }
}

/// Mutable damping state for [`BacktrackingLevenbergMarquardtMethod`].
#[derive(Debug, Clone, Copy)]
pub struct BacktrackingLevenbergMarquardtState {
    /// Current identity-damping parameter.
    pub lambda: f64,
}

impl BacktrackingLevenbergMarquardtMethod {
    fn validate(&self) -> Result<(), SolveError> {
        if !self.lambda_init.is_finite()
            || self.lambda_init <= 0.0
            || !self.lambda_decrease.is_finite()
            || !(0.0..1.0).contains(&self.lambda_decrease)
            || !self.lambda_increase.is_finite()
            || self.lambda_increase <= 1.0
            || !self.alpha_min.is_finite()
            || !(0.0..1.0).contains(&self.alpha_min)
            || !self.max_lambda.is_finite()
            || self.max_lambda < self.lambda_init
        {
            return Err(SolveError::InvalidConfig(
                "Backtracking LM requires 0 < lambda_decrease < 1, lambda_increase > 1, 0 < alpha_min < 1, and finite lambda bounds".to_string(),
            ));
        }
        Ok(())
    }

    fn is_feasible(options: &SolveOptions, candidate: &DVector<f64>) -> bool {
        candidate.iter().all(|value| value.is_finite())
            && options
                .bounds
                .as_ref()
                .map(|bounds| bounds.validate(candidate).is_ok())
                .unwrap_or(true)
    }
}

impl NonlinearMethod for BacktrackingLevenbergMarquardtMethod {
    type MethodState = BacktrackingLevenbergMarquardtState;

    fn init<P: JacobianProvider>(
        &self,
        _problem: &P,
        _x0: &DVector<f64>,
        _options: &SolveOptions,
        _residual: &DVector<f64>,
        _jacobian: &DMatrix<f64>,
    ) -> Result<Self::MethodState, SolveError> {
        self.validate()?;
        Ok(BacktrackingLevenbergMarquardtState {
            lambda: self.lambda_init,
        })
    }

    fn step<P: JacobianProvider>(
        &self,
        problem: &P,
        state: &IterationState,
        method_state: &mut Self::MethodState,
        options: &SolveOptions,
        runtime: &mut RuntimeDiagnostics,
    ) -> Result<StepOutcome, SolveError> {
        self.step_impl(problem, state, method_state, options, runtime, None)
    }

    fn supports_step_workspace(&self) -> bool {
        true
    }

    fn step_with_workspace<P: JacobianProvider>(
        &self,
        problem: &P,
        state: &IterationState,
        method_state: &mut Self::MethodState,
        options: &SolveOptions,
        runtime: &mut RuntimeDiagnostics,
        workspace: Option<&mut MethodWorkspace>,
    ) -> Result<StepOutcome, SolveError> {
        self.step_impl(problem, state, method_state, options, runtime, workspace)
    }
}

impl BacktrackingLevenbergMarquardtMethod {
    fn step_impl<P: JacobianProvider>(
        &self,
        problem: &P,
        state: &IterationState,
        method_state: &mut BacktrackingLevenbergMarquardtState,
        options: &SolveOptions,
        runtime: &mut RuntimeDiagnostics,
        mut workspace: Option<&mut MethodWorkspace>,
    ) -> Result<StepOutcome, SolveError> {
        let mut regularized = state.jacobian.transpose() * &state.jacobian;
        for diagonal in 0..regularized.nrows() {
            regularized[(diagonal, diagonal)] += method_state.lambda;
        }
        let rhs = -(state.jacobian.transpose() * &state.residual);

        runtime.linear_solves += 1;
        let delta = measure_linear_system_operation_owned(
            options.linear_solver,
            regularized,
            &rhs,
            runtime,
            options.diagnostics.collect_statistics,
        )?;
        if !delta.iter().all(|value| value.is_finite()) || delta.norm() <= f64::EPSILON {
            return Ok(StepOutcome::Terminated(TerminationReason::StepTooSmall));
        }

        let current_norm = state.residual_norm;
        let mut alpha = 1.0;
        while alpha >= self.alpha_min {
            let candidate = if let Some(workspace) = workspace.as_deref_mut() {
                workspace.set_affine_trial(&state.x, alpha, &delta)?;
                Cow::Borrowed(workspace.trial_x())
            } else {
                Cow::Owned(&state.x + alpha * &delta)
            };

            if Self::is_feasible(options, &candidate) {
                let trial_residual = eval_residual_with_runtime(
                    problem,
                    &candidate,
                    runtime,
                    options.diagnostics.collect_statistics,
                )?;
                if trial_residual.iter().all(|value| value.is_finite())
                    && trial_residual.norm() < current_norm
                {
                    method_state.lambda =
                        (method_state.lambda * self.lambda_decrease).max(f64::MIN_POSITIVE);
                    runtime.accepted_steps += 1;
                    return Ok(StepOutcome::Continue {
                        next_x: candidate.into_owned(),
                        accepted: true,
                    });
                }
            }

            runtime.rejected_steps += 1;
            alpha *= 0.5;
        }

        method_state.lambda = (method_state.lambda * self.lambda_increase).min(self.max_lambda);
        Ok(StepOutcome::Continue {
            next_x: state.x.clone(),
            accepted: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::Nonlinear_systems::engine::SolverEngine;
    use crate::numerical::Nonlinear_systems::problem::NonlinearProblem;
    use approx::assert_relative_eq;

    struct ScalarQuadratic;

    impl NonlinearProblem for ScalarQuadratic {
        fn dimension(&self) -> usize {
            1
        }

        fn residual(&self, x: &DVector<f64>) -> Result<DVector<f64>, SolveError> {
            Ok(DVector::from_element(1, x[0] * x[0] - 2.0))
        }
    }

    impl JacobianProvider for ScalarQuadratic {
        fn jacobian(&self, x: &DVector<f64>) -> Result<DMatrix<f64>, SolveError> {
            Ok(DMatrix::from_element(1, 1, 2.0 * x[0]))
        }
    }

    #[test]
    fn backtracking_lm_solves_scalar_problem_with_identity_damping() {
        let result = SolverEngine::new(
            BacktrackingLevenbergMarquardtMethod::default(),
            SolveOptions {
                tolerance: 1.0e-10,
                max_iterations: 100,
                ..SolveOptions::default()
            },
        )
        .solve(&ScalarQuadratic, DVector::from_element(1, 1.5))
        .expect("backtracking LM should solve the scalar problem");

        assert_eq!(result.termination, TerminationReason::Converged);
        assert_relative_eq!(result.x[0], 2.0_f64.sqrt(), epsilon = 1.0e-8);
    }

    #[test]
    fn backtracking_lm_rejects_infeasible_trials_without_clipping_them() {
        let result = SolverEngine::new(
            BacktrackingLevenbergMarquardtMethod::default(),
            SolveOptions {
                tolerance: 1.0e-10,
                max_iterations: 100,
                bounds: Some(
                    crate::numerical::Nonlinear_systems::problem::Bounds::new(vec![(0.0, 2.0)])
                        .expect("bounds"),
                ),
                ..SolveOptions::default()
            },
        )
        .solve(&ScalarQuadratic, DVector::from_element(1, 0.25))
        .expect("bounded backtracking LM should remain typed and finite");

        assert_eq!(result.termination, TerminationReason::Converged);
        assert!(result.statistics.rejected_steps > 0);
        assert_relative_eq!(result.x[0], 2.0_f64.sqrt(), epsilon = 1.0e-8);
    }
}
