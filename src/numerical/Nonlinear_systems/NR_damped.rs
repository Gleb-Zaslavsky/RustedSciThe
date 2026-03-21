use log::{error, info, warn};
use nalgebra::{DMatrix, DVector};

use crate::numerical::Nonlinear_systems::engine::{
    IterationState, NonlinearMethod, RuntimeDiagnostics, SolveOptions, StepOutcome,
    solve_linear_system,
};
use crate::numerical::Nonlinear_systems::error::{SolveError, TerminationReason};
use crate::numerical::Nonlinear_systems::problem::JacobianProvider;

/// Damped Newton-Raphson with Clipping
/// A constrained version of the Newton-Raphson (NR) method that prevents the solution vector from violating bounds (e.g.,
// xi∈[ai,bi] xi ∈[ai​,bi]) by combining damping (line search) and clipping (projection).
/// 1.) Key Concepts
///  (a) Standard Newton-Raphson Method
/// The NR method iteratively solves
///   F(x)=0 via:x(k+1)=x(k)−J−1(x(k))⋅
/// where:F(x)=0: System of nonlinear equations.J(x): Jacobian matrix (Jij=∂Fi/∂xj=∂Fj/∂xi
/// Problem: Unconstrained NR may violate bounds (e.g., negative concentrations, unphysical values).
///
///(b) Damping (Line Search)
/// Introduces a step size
/// α(k)∈(0,1] to control the update:x(k+1)=x(k)−α(k)J^−1(x(k))F(x(k))
/// α(k)∈(0,1] is chosen to:
/// Ensure x(k+1) stays within bounds.
/// Guarantees progress (e.g., reduce ∥F(x)∥).
///
/// (c) Clipping (Projection)
/// After each NR step, enforce bounds explicitly:
/// x(k+1)=clip(  x(k)−α(k)J^−1F(x(k)) )
/// where:
/// clip(x,a,b)={ai
/// if xi<ai,
/// xi if ai≤xi≤bi,
/// bi if xi>bi.
/// 2.) Algorithm Steps
/// Initialize:
/// x(0) (must be feasible, i.e., a≤x(0)≤b).
/// Iterate until convergence:
/// k=0,1,2,…
/// Compute Newton step: Δx=−J−1(x(k))F(x(k)).
/// Choose α(k) (e.g., via backtracking line search).
/// Update: x(k+1)=x(k)+α(k)Δx.
/// Enforce bounds: x(k+1)=clip(x(k),a,b).
/// k=k+1.
///
/// 3.) Advantages
///     1. Prevents NR from violating bounds.
///     2. Guarantees progress (e.g., reduce ∥F(x)∥).
/// 4.) Limitations
/// 1. Requires Jacobian matrix.
/// 2. Requires backtracking line search.
/// 3. May require multiple iterations.
/// 5. Applications
///     1. Chemical kinetics.
///     2. Fluid mechanics.
///
/// Damped Newton method with backtracking.
#[derive(Debug, Clone, Copy)]
pub struct DampedNewtonMethod {
    /// Maximum number of backtracking reductions.
    pub max_line_search_steps: usize,
    /// Armijo-like decrease factor.
    pub sufficient_decrease: f64,
    /// Step shrink factor.
    pub shrink_factor: f64,
}

impl Default for DampedNewtonMethod {
    fn default() -> Self {
        Self {
            max_line_search_steps: 50,
            sufficient_decrease: 1e-4,
            shrink_factor: 0.5,
        }
    }
}

//======================================================================================
//  Damped Newton method with backtracking.
//==================================================================================
impl NonlinearMethod for DampedNewtonMethod {
    type MethodState = ();

    fn init<P: JacobianProvider>(
        &self,
        _problem: &P,
        _x0: &DVector<f64>,
        _options: &SolveOptions,
        _residual: &DVector<f64>,
        _jacobian: &DMatrix<f64>,
    ) -> Result<Self::MethodState, SolveError> {
        if self.max_line_search_steps == 0 {
            return Err(SolveError::InvalidConfig(
                "max_line_search_steps must be greater than zero".to_string(),
            ));
        }
        if !(0.0 < self.shrink_factor && self.shrink_factor < 1.0) {
            return Err(SolveError::InvalidConfig(
                "shrink_factor must belong to (0, 1)".to_string(),
            ));
        }
        if self.sufficient_decrease <= 0.0 {
            return Err(SolveError::InvalidConfig(
                "sufficient_decrease must be positive".to_string(),
            ));
        }
        Ok(())
    }

    fn step<P: JacobianProvider>(
        &self,
        problem: &P,
        state: &IterationState,
        _method_state: &mut Self::MethodState,
        options: &SolveOptions,
        runtime: &mut RuntimeDiagnostics,
    ) -> Result<StepOutcome, SolveError> {
        runtime.linear_solves += 1;
        // Compute Newton step:
        let newton_step =
            solve_linear_system(options.linear_solver, &state.jacobian, &state.residual)?;
        if newton_step.norm() < options.tolerance {
            return Ok(StepOutcome::Terminated(TerminationReason::StepTooSmall));
        }

        let mut alpha: f64 = 1.0;
        if let Some(bounds) = &options.bounds {
            alpha = alpha.min(bounds.max_step_scale(&state.x, &(-&newton_step)));
        }
        let current_norm = state.residual_norm;

        for _ in 0..self.max_line_search_steps {
            let trial_x = &state.x - alpha * &newton_step;
            let trial_x = if let Some(bounds) = &options.bounds {
                bounds.project(&trial_x)
            } else {
                trial_x
            };
            let trial_norm = problem.residual(&trial_x)?.norm();
            if trial_norm < current_norm * (1.0 - self.sufficient_decrease * alpha)
                || trial_norm < options.tolerance
            {
                runtime.accepted_steps += 1;
                return Ok(StepOutcome::Continue {
                    next_x: trial_x,
                    accepted: true,
                });
            }
            alpha *= self.shrink_factor;
            runtime.rejected_steps += 1;
            if alpha < options.tolerance {
                return Ok(StepOutcome::Terminated(TerminationReason::Stagnation));
            }
        }

        Ok(StepOutcome::Terminated(
            TerminationReason::RejectedStepLimit,
        ))
    }
}

/// Advanced damped Newton method with clipping and bound-aware damping.
/// Combines bound_step calculation with clipping for constrained problems.
#[derive(Debug, Clone, Copy)]
pub struct DampedNewtonMethodAdvanced {
    /// Maximum number of damping iterations.
    pub max_damping_iterations: usize,
    /// Sufficient decrease factor (Armijo-like).
    pub sufficient_decrease: f64,
    /// Damping shrink factor.
    pub shrink_factor: f64,
}

impl Default for DampedNewtonMethodAdvanced {
    fn default() -> Self {
        Self {
            max_damping_iterations: 50,
            sufficient_decrease: 1e-4,
            shrink_factor: 0.5,
        }
    }
}

impl NonlinearMethod for DampedNewtonMethodAdvanced {
    type MethodState = ();

    fn init<P: JacobianProvider>(
        &self,
        _problem: &P,
        _x0: &DVector<f64>,
        options: &SolveOptions,
        _residual: &DVector<f64>,
        _jacobian: &DMatrix<f64>,
    ) -> Result<Self::MethodState, SolveError> {
        if self.max_damping_iterations == 0 {
            return Err(SolveError::InvalidConfig(
                "max_damping_iterations must be greater than zero".to_string(),
            ));
        }
        if !(0.0 < self.shrink_factor && self.shrink_factor < 1.0) {
            return Err(SolveError::InvalidConfig(
                "shrink_factor must belong to (0, 1)".to_string(),
            ));
        }
        if self.sufficient_decrease <= 0.0 {
            return Err(SolveError::InvalidConfig(
                "sufficient_decrease must be positive".to_string(),
            ));
        }
        if options.bounds.is_none() {
            warn!("DampedNewtonMethodAdvanced works best with bounds specified");
        }
        Ok(())
    }

    fn step<P: JacobianProvider>(
        &self,
        problem: &P,
        state: &IterationState,
        _method_state: &mut Self::MethodState,
        options: &SolveOptions,
        runtime: &mut RuntimeDiagnostics,
    ) -> Result<StepOutcome, SolveError> {
        runtime.linear_solves += 1;
        // Compute Newton step:
        let newton_step =
            solve_linear_system(options.linear_solver, &state.jacobian, &state.residual)?;

        if newton_step.norm() < options.tolerance {
            return Ok(StepOutcome::Terminated(TerminationReason::StepTooSmall));
        }
        let current_norm = state.residual_norm;
        let mut lambda = 1.0;

        if let Some(bounds) = &options.bounds {
            let fbound = bound_step(&state.x, &newton_step, bounds);

            if fbound.is_nan() || fbound.is_infinite() {
                return Err(SolveError::NumericalBreakdown(
                    "bound_step returned invalid value".to_string(),
                ));
            }

            if fbound < 1e-10 {
                warn!(
                    "Initial step violates bounds severely, fbound = {:.6e}",
                    fbound
                );
            }
            lambda = fbound.max(1e-10);
        }

        for k in 0..self.max_damping_iterations {
            info!("Damping iteration {}, lambda = {:.6e}", k, lambda);

            let damped_step = lambda * &newton_step;
            let trial_x = &state.x - &damped_step;

            let trial_x = if let Some(bounds) = &options.bounds {
                bounds.project(&trial_x)
            } else {
                trial_x
            };

            let trial_norm = problem.residual(&trial_x)?.norm();

            info!(
                "Trial norm = {:.6e}, current norm = {:.6e}",
                trial_norm, current_norm
            );
            // If the norm decreases, then accept this
            // damping coefficient. Also accept it if this step would result in a
            // converged solution. Otherwise, decrease the damping coefficient and
            // try again.
            // The  criterion for accepting is that the undamped steps decrease in
            // magnitude, This prevents the iteration from stepping away from the region where there is good reason to believe a solution lies
            if trial_norm < current_norm * (1.0 - lambda * self.sufficient_decrease)
                || trial_norm < options.tolerance
            {
                info!("Damping coefficient accepted");
                runtime.accepted_steps += 1;
                return Ok(StepOutcome::Continue {
                    next_x: trial_x,
                    accepted: true,
                });
            }
            // if fail this criterion we must reject it and retries the step with a reduced (often halved) damping parameter trying again until
            // criterion is met  or max damping iterations is reached
            lambda *= self.shrink_factor;
            runtime.rejected_steps += 1;
        }

        warn!("Max damping iterations reached");
        Ok(StepOutcome::Terminated(
            TerminationReason::RejectedStepLimit,
        ))
    }
}

/// Calculates the minimum damping factor to keep the solution within bounds after a Newton step.
pub fn bound_step(
    y: &DVector<f64>,
    step: &DVector<f64>,
    bounds: &crate::numerical::Nonlinear_systems::problem::Bounds,
) -> f64 {
    let mut fbound = 1.0;
    let limits = bounds.as_slice();

    for (i, &y_i) in y.iter().enumerate() {
        if i >= limits.len() {
            break;
        }
        let (below, above) = limits[i];

        let s_i = step[i];
        if y_i <= below + 1e-14 {
            warn!(
                "Solution y[{}] = {:.6e} is at lower bound {:.6e}",
                i, y_i, below
            );
        }
        if y_i >= above - 1e-14 {
            warn!(
                "Solution y[{}] = {:.6e} is at upper bound {:.6e}",
                i, y_i, above
            );
        }

        if s_i > f64::max(y_i - below, 0.0) {
            let temp = (y_i - below) / s_i;
            if temp < fbound {
                fbound = temp;
            }
        } else if s_i < f64::min(y_i - above, 0.0) {
            let temp = (y_i - above) / s_i;
            if temp < fbound {
                fbound = temp;
            }
        }
    }
    fbound
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::Nonlinear_systems::engine::{SolveOptions, SolverEngine};
    use crate::numerical::Nonlinear_systems::error::TerminationReason;
    use crate::numerical::Nonlinear_systems::problem::{
        Bounds, JacobianProvider, NonlinearProblem,
    };
    use crate::numerical::Nonlinear_systems::symbolic::SymbolicNonlinearProblem;
    use approx::assert_relative_eq;

    struct BoundedQuadraticProblem;

    impl NonlinearProblem for BoundedQuadraticProblem {
        fn dimension(&self) -> usize {
            2
        }
        fn residual(&self, x: &DVector<f64>) -> Result<DVector<f64>, SolveError> {
            Ok(DVector::from_vec(vec![
                x[0] * x[0] + x[1] * x[1] - 10.0,
                x[0] - x[1] - 4.0,
            ]))
        }
    }

    impl JacobianProvider for BoundedQuadraticProblem {
        fn jacobian(&self, x: &DVector<f64>) -> Result<DMatrix<f64>, SolveError> {
            Ok(DMatrix::from_row_slice(
                2,
                2,
                &[2.0 * x[0], 2.0 * x[1], 1.0, -1.0],
            ))
        }
    }

    #[test]
    fn damped_newton_advanced_converges_with_bounds() {
        let bounds = Bounds::new(vec![(-5.0, 5.0), (-5.0, 5.0)]).expect("bounds");
        let options = SolveOptions {
            bounds: Some(bounds),
            tolerance: 1e-6,
            max_iterations: 100,
            ..SolveOptions::default()
        };
        let result = SolverEngine::new(DampedNewtonMethodAdvanced::default(), options)
            .solve(&BoundedQuadraticProblem, DVector::from_vec(vec![1.0, 1.0]))
            .expect("solve");

        assert_eq!(result.termination, TerminationReason::Converged);
        assert_relative_eq!(result.x[0], 3.0, epsilon = 1e-5);
        assert_relative_eq!(result.x[1], -1.0, epsilon = 1e-5);
    }

    #[test]
    fn damped_newton_advanced_respects_tight_bounds() {
        let bounds = Bounds::new(vec![(0.0, 2.0), (0.0, 2.0)]).expect("bounds");
        let options = SolveOptions {
            bounds: Some(bounds),
            tolerance: 1e-6,
            max_iterations: 100,
            ..SolveOptions::default()
        };
        let result = SolverEngine::new(DampedNewtonMethodAdvanced::default(), options)
            .solve(&BoundedQuadraticProblem, DVector::from_vec(vec![1.5, 1.5]))
            .expect("solve");

        assert!(result.x[0] >= 0.0 && result.x[0] <= 2.0);
        assert!(result.x[1] >= 0.0 && result.x[1] <= 2.0);
    }

    #[test]
    fn damped_newton_advanced_with_symbolic_problem() {
        let problem = SymbolicNonlinearProblem::from_strings(
            vec!["x^2+y^2-10".to_string(), "x-y-4".to_string()],
            Some(vec!["x".to_string(), "y".to_string()]),
            None,
            None,
        )
        .expect("symbolic problem");

        let bounds = Bounds::new(vec![(-10.0, 10.0), (-10.0, 10.0)]).expect("bounds");
        let options = SolveOptions {
            bounds: Some(bounds),
            tolerance: 1e-6,
            max_iterations: 100,
            ..SolveOptions::default()
        };

        let result = SolverEngine::new(DampedNewtonMethodAdvanced::default(), options)
            .solve(&problem, DVector::from_vec(vec![1.0, 1.0]))
            .expect("solve");

        assert_eq!(result.termination, TerminationReason::Converged);
        assert_relative_eq!(result.x[0], 3.0, epsilon = 1e-5);
        assert_relative_eq!(result.x[1], -1.0, epsilon = 1e-5);
    }

    #[test]
    fn bound_step_calculates_correct_damping_factor() {
        let y = DVector::from_vec(vec![1.0, 2.0]);
        let step = DVector::from_vec(vec![2.0, 3.0]);
        let bounds = Bounds::new(vec![(0.0, 5.0), (0.0, 6.0)]).expect("bounds");

        let fbound = bound_step(&y, &step, &bounds);

        assert_relative_eq!(fbound, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn bound_step_handles_no_violation() {
        let y = DVector::from_vec(vec![2.0, 3.0]);
        let step = DVector::from_vec(vec![0.5, 0.5]);
        let bounds = Bounds::new(vec![(0.0, 10.0), (0.0, 10.0)]).expect("bounds");

        let fbound = bound_step(&y, &step, &bounds);

        assert_relative_eq!(fbound, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn damped_newton_advanced_works_without_bounds() {
        let options = SolveOptions {
            bounds: None,
            tolerance: 1e-6,
            max_iterations: 100,
            ..SolveOptions::default()
        };

        let result = SolverEngine::new(DampedNewtonMethodAdvanced::default(), options)
            .solve(&BoundedQuadraticProblem, DVector::from_vec(vec![1.0, 1.0]))
            .expect("solve");

        assert_eq!(result.termination, TerminationReason::Converged);
        assert_relative_eq!(result.x[0], 3.0, epsilon = 1e-5);
        assert_relative_eq!(result.x[1], -1.0, epsilon = 1e-5);
    }
}
