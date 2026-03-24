//! MINPACK-style trust-region subproblem solver for nonlinear systems.
//!
//! The low-level helper in this module mirrors the `LMPAR` logic used by
//! MINPACK. It solves the linearized least-squares subproblem
//! `min ||J p - r||` with the trust-region constraint `||D p|| <= delta`.
//!
//! The nonlinear-system method built on top of it converts the returned
//! parameter update into the usual Newton-like step `x_{k+1} = x_k - p`.

use crate::numerical::Nonlinear_systems::engine::{
    IterationState, NonlinearMethod, RuntimeDiagnostics, SolveOptions, StepOutcome, scaled_norm,
    scaling_vector,
};
use crate::numerical::Nonlinear_systems::error::{SolveError, TerminationReason};
use crate::numerical::Nonlinear_systems::problem::JacobianProvider;
use crate::numerical::optimization::qr_LM::PivotedQR;
use crate::numerical::optimization::trust_region_LM::determine_lambda_and_parameter_update;
use nalgebra::{DMatrix, DVector};

#[cfg(test)]
use approx::assert_relative_eq;

/// Output of the MINPACK-style trust-region subproblem.
#[derive(Debug, Clone)]
pub struct TrustRegionResult {
    /// Parameter update returned by the trust-region subproblem.
    ///
    /// In MINPACK notation, the nonlinear iterate is updated as `x_new = x - step`.
    pub step: DVector<f64>,
    /// Levenberg-Marquardt parameter chosen by the subproblem solve.
    pub lambda: f64,
    /// Norm of the scaled parameter update `||D step||`.
    pub scaled_step_norm: f64,
    /// `true` when the step lies on the trust-region boundary.
    pub on_boundary: bool,
}

/// Solves the MINPACK trust-region subproblem for one linearized least-squares model.
///
/// This helper uses pivoted QR and a MINPACK-style `LMPAR` iteration. The returned
/// `step` follows the same sign convention as MINPACK, so a nonlinear solver usually
/// applies it as `x_new = x - step`.
pub fn solve_trust_region_subproblem(
    jacobian: &DMatrix<f64>,
    residuals: &DVector<f64>,
    diag: &DVector<f64>,
    delta: f64,
    lambda_prev: f64,
) -> Result<TrustRegionResult, &'static str> {
    if residuals.len() != jacobian.nrows() {
        return Err("Residuals dimension mismatch");
    }
    if diag.len() != jacobian.ncols() {
        return Err("Diagonal vector dimension mismatch");
    }
    if delta <= 0.0 {
        return Err("Trust region radius must be positive");
    }
    if diag.iter().any(|value| *value <= 0.0 || !value.is_finite()) {
        return Err("Diagonal scaling entries must be positive and finite");
    }
    if !lambda_prev.is_finite() || lambda_prev < 0.0 {
        return Err("Previous lambda must be finite and non-negative");
    }

    let qr = PivotedQR::new(jacobian.clone());
    let mut lls = qr.into_least_squares_diagonal_problem(residuals.clone());
    let parameter = determine_lambda_and_parameter_update(&mut lls, diag, delta, lambda_prev);

    Ok(TrustRegionResult {
        on_boundary: parameter.lambda > 0.0,
        step: parameter.step,
        lambda: parameter.lambda,
        scaled_step_norm: parameter.dp_norm,
    })
}

/// MINPACK-style trust-region Levenberg-Marquardt method for nonlinear systems.
#[derive(Debug, Clone, Copy)]
pub struct TrustRegionLMMethod {
    /// Initial trust-region radius multiplier.
    pub step_bound: f64,
    /// Initial damping parameter.
    pub lambda_init: f64,
    /// Reuse column norms as diagonal scaling.
    pub scale_diag: bool,
    /// Minimum ratio required to accept a trial step.
    pub acceptance_threshold: f64,
}

impl Default for TrustRegionLMMethod {
    fn default() -> Self {
        Self {
            step_bound: 100.0,
            lambda_init: 0.0,
            scale_diag: true,
            acceptance_threshold: 1.0e-4,
        }
    }
}

/// Mutable state carried between trust-region LM iterations.
#[derive(Debug, Clone)]
pub struct TrustRegionLMState {
    /// Current damping parameter.
    lambda: f64,
    /// Current trust-region radius.
    delta: f64,
    /// Current diagonal scaling.
    diag: DVector<f64>,
    /// Special first-step rule from MINPACK.
    first_trust_region_iteration: bool,
}

impl NonlinearMethod for TrustRegionLMMethod {
    type MethodState = TrustRegionLMState;

    fn init<P: JacobianProvider>(
        &self,
        _problem: &P,
        x0: &DVector<f64>,
        _options: &SolveOptions,
        _residual: &DVector<f64>,
        jacobian: &DMatrix<f64>,
    ) -> Result<Self::MethodState, SolveError> {
        if self.step_bound <= 0.0 {
            return Err(SolveError::InvalidConfig(
                "step_bound must be positive".to_string(),
            ));
        }
        if self.lambda_init < 0.0 || !self.lambda_init.is_finite() {
            return Err(SolveError::InvalidConfig(
                "lambda_init must be finite and non-negative".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.acceptance_threshold) {
            return Err(SolveError::InvalidConfig(
                "acceptance_threshold must belong to [0, 1]".to_string(),
            ));
        }

        let diag = scaling_vector(jacobian, self.scale_diag);
        let xnorm = scaled_norm(&diag, x0);
        let delta = if xnorm == 0.0 {
            self.step_bound
        } else {
            self.step_bound * xnorm
        };

        Ok(TrustRegionLMState {
            lambda: self.lambda_init,
            delta,
            diag,
            first_trust_region_iteration: true,
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
        const P1: f64 = 0.1;
        const P25: f64 = 0.25;
        const P75: f64 = 0.75;
        const HALF: f64 = 0.5;

        if self.scale_diag {
            let fresh_diag = scaling_vector(&state.jacobian, true);
            for index in 0..method_state.diag.len() {
                method_state.diag[index] = method_state.diag[index].max(fresh_diag[index]);
            }
        }

        runtime.linear_solves += 1;
        let parameter = solve_trust_region_subproblem(
            &state.jacobian,
            &state.residual,
            &method_state.diag,
            method_state.delta,
            method_state.lambda,
        )
        .map_err(|message| SolveError::LinearSolveFailure(message.to_string()))?;

        method_state.lambda = parameter.lambda;

        if method_state.first_trust_region_iteration
            && parameter.scaled_step_norm < method_state.delta
        {
            method_state.delta = parameter.scaled_step_norm;
        }
        method_state.first_trust_region_iteration = false;

        let nonlinear_step = -parameter.step.clone();
        if nonlinear_step.norm() < options.tolerance {
            if state.residual_norm <= 10.0 * options.tolerance {
                return Ok(StepOutcome::Converged);
            }
            return Ok(StepOutcome::Terminated(TerminationReason::StepTooSmall));
        }

        let trial_x = if let Some(bounds) = &options.bounds {
            bounds.project(&(&state.x + &nonlinear_step))
        } else {
            &state.x + &nonlinear_step
        };
        let effective_step = &trial_x - &state.x;
        if effective_step.norm() < options.tolerance {
            if state.residual_norm <= 10.0 * options.tolerance {
                return Ok(StepOutcome::Converged);
            }
            return Ok(StepOutcome::Terminated(TerminationReason::StepTooSmall));
        }

        let trial_residual = problem.residual(&trial_x)?;
        let trial_norm = trial_residual.norm();

        let actual_reduction = if trial_norm * P1 < state.residual_norm {
            1.0 - (trial_norm / state.residual_norm).powi(2)
        } else {
            -1.0
        };

        let predicted_reduction = {
            let residual_norm = state.residual_norm.max(f64::EPSILON);
            let temp1 = (&state.jacobian * &parameter.step).norm() / residual_norm;
            let temp2 = (parameter.lambda.sqrt() * parameter.scaled_step_norm) / residual_norm;
            temp1.powi(2) + temp2.powi(2) / HALF
        };
        let dir_derivative = -predicted_reduction * HALF;

        let ratio = if predicted_reduction.abs() > f64::EPSILON {
            actual_reduction / predicted_reduction
        } else {
            0.0
        };

        if ratio <= P25 {
            let mut temp = if actual_reduction >= 0.0 {
                HALF
            } else {
                HALF * dir_derivative / (dir_derivative + HALF * actual_reduction)
            };
            if trial_norm * P1 >= state.residual_norm || temp < P1 {
                temp = P1;
            }
            method_state.delta = temp * method_state.delta.min(parameter.scaled_step_norm * 10.0);
            method_state.lambda /= temp;
        } else if parameter.lambda == 0.0 || ratio >= P75 {
            method_state.delta = parameter.scaled_step_norm / HALF;
            method_state.lambda *= HALF;
        }

        if !method_state.delta.is_finite() || !method_state.lambda.is_finite() {
            return Err(SolveError::NumericalBreakdown(
                "trust-region LM produced NaN or Inf".to_string(),
            ));
        }

        if ratio >= self.acceptance_threshold {
            runtime.accepted_steps += 1;
            Ok(StepOutcome::Continue {
                next_x: trial_x,
                accepted: true,
            })
        } else {
            runtime.rejected_steps += 1;
            if method_state.delta <= options.tolerance.max(1.0e-14) {
                return Ok(StepOutcome::Terminated(TerminationReason::Stagnation));
            }
            Ok(StepOutcome::Continue {
                next_x: state.x.clone(),
                accepted: false,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::Nonlinear_systems::engine::{SolveOptions, SolverEngine};
    use crate::numerical::Nonlinear_systems::problem::{JacobianProvider, NonlinearProblem};
    use crate::numerical::Nonlinear_systems::symbolic::SymbolicNonlinearProblem;
    use nalgebra::{dmatrix, dvector};

    struct CoupledPlainProblem;

    impl NonlinearProblem for CoupledPlainProblem {
        fn dimension(&self) -> usize {
            2
        }

        fn residual(&self, x: &DVector<f64>) -> Result<DVector<f64>, SolveError> {
            Ok(dvector![
                x[0] * x[0] + x[1] * x[1] - 10.0,
                x[0] - x[1] - 4.0
            ])
        }
    }

    impl JacobianProvider for CoupledPlainProblem {
        fn jacobian(&self, x: &DVector<f64>) -> Result<DMatrix<f64>, SolveError> {
            Ok(dmatrix![2.0 * x[0], 2.0 * x[1]; 1.0, -1.0])
        }
    }

    #[test]
    fn gauss_newton_step_within_trust_region() {
        let jacobian = dmatrix![1.0, 0.0; 0.0, 1.0];
        let residuals = dvector![0.1, 0.2];
        let diag = dvector![1.0, 1.0];

        let result = solve_trust_region_subproblem(&jacobian, &residuals, &diag, 1.0, 0.0).unwrap();

        assert_relative_eq!(result.lambda, 0.0, epsilon = 1e-12);
        assert!(!result.on_boundary);
        assert_relative_eq!(result.step[0], 0.1, epsilon = 1e-12);
        assert_relative_eq!(result.step[1], 0.2, epsilon = 1e-12);
    }

    #[test]
    fn constrained_step_on_boundary() {
        let jacobian = dmatrix![1.0, 0.0; 0.0, 1.0];
        let residuals = dvector![1.0, 1.0];
        let diag = dvector![1.0, 1.0];

        let result = solve_trust_region_subproblem(&jacobian, &residuals, &diag, 0.5, 0.0).unwrap();

        assert!(result.lambda > 0.0);
        assert!(result.on_boundary);
        assert_relative_eq!(result.scaled_step_norm, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn minpack_reference_case1_matches() {
        let j = DMatrix::from_column_slice(
            4,
            3,
            &[
                33., -40., 44., -43., -37., -1., -40., 48., 43., -11., -40., 43.,
            ],
        );
        let residual = dvector![7., -1., 0., -1.];
        let diag = dvector![18.2, 18.2, 3.2];

        let result = solve_trust_region_subproblem(&j, &residual, &diag, 0.5, 0.2).unwrap();

        assert_relative_eq!(result.lambda, 34.628643558156341, epsilon = 1e-12);
        assert_relative_eq!(
            result.step,
            dvector![0.017591648698939, -0.020395135814051, 0.059285196018896],
            epsilon = 1e-14
        );
    }

    #[test]
    fn minpack_reference_case2_matches() {
        let j = DMatrix::from_column_slice(
            4,
            3,
            &[
                -7., 28., -40., 29., 7., -49., -39., 43., -25., -47., -11., 34.,
            ],
        );
        let residual = dvector![-7., -8., -8., -10.];
        let diag = dvector![10.2, 13.2, 1.2];

        let result = solve_trust_region_subproblem(&j, &residual, &diag, 0.5, 0.2).unwrap();

        assert_relative_eq!(result.lambda, 0.0, epsilon = 1e-12);
        assert_relative_eq!(
            result.step,
            dvector![-0.048474221517806, -0.007207732068190, 0.083138659283539],
            epsilon = 1e-14
        );
    }

    #[test]
    fn dimension_and_radius_validation_work() {
        let j = dmatrix![1.0, 0.0; 0.0, 1.0];
        let residual = dvector![1.0];
        let diag = dvector![1.0, 1.0];

        assert_eq!(
            solve_trust_region_subproblem(&j, &residual, &diag, 1.0, 0.0).unwrap_err(),
            "Residuals dimension mismatch"
        );
        assert_eq!(
            solve_trust_region_subproblem(&j, &dvector![1.0, 1.0], &dvector![1.0], 1.0, 0.0)
                .unwrap_err(),
            "Diagonal vector dimension mismatch"
        );
        assert_eq!(
            solve_trust_region_subproblem(&j, &dvector![1.0, 1.0], &diag, 0.0, 0.0).unwrap_err(),
            "Trust region radius must be positive"
        );
    }

    #[test]
    fn trust_region_lm_method_solves_plain_problem() {
        let options = SolveOptions {
            tolerance: 1.0e-8,
            max_iterations: 50,
            ..SolveOptions::default()
        };

        let result = SolverEngine::new(TrustRegionLMMethod::default(), options)
            .solve(&CoupledPlainProblem, dvector![1.0, 1.0])
            .unwrap();

        assert_eq!(result.termination, TerminationReason::Converged);
        assert!(result.residual_norm < 1.0e-8);
        assert_relative_eq!(result.x[0], 3.0, epsilon = 1.0e-6);
        assert_relative_eq!(result.x[1], -1.0, epsilon = 1.0e-6);
    }

    #[test]
    fn trust_region_lm_method_solves_symbolic_problem() {
        let problem = SymbolicNonlinearProblem::from_strings(
            vec!["x^2+y^2-10".to_string(), "x-y-4".to_string()],
            Some(vec!["x".to_string(), "y".to_string()]),
            None,
            None,
        )
        .unwrap();
        let options = SolveOptions {
            tolerance: 1.0e-8,
            max_iterations: 50,
            ..SolveOptions::default()
        };

        let result = SolverEngine::new(TrustRegionLMMethod::default(), options)
            .solve(&problem, dvector![1.0, 1.0])
            .unwrap();

        assert_eq!(result.termination, TerminationReason::Converged);
        assert!(result.residual_norm < 1.0e-8);
        assert_relative_eq!(result.x[0], 3.0, epsilon = 1.0e-6);
        assert_relative_eq!(result.x[1], -1.0, epsilon = 1.0e-6);
    }
}
