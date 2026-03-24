use nalgebra::{DMatrix, DVector};

use crate::numerical::Nonlinear_systems::engine::{
    IterationState, LinearSolverKind, NonlinearMethod, RuntimeDiagnostics, SolveOptions,
    StepOutcome, scaled_norm, scaling_vector, solve_linear_system,
};
use crate::numerical::Nonlinear_systems::error::{SolveError, TerminationReason};
use crate::numerical::Nonlinear_systems::problem::JacobianProvider;
use crate::numerical::Nonlinear_systems::trust_region_LM::solve_trust_region_subproblem;

/// CLASSIC LEVENBERG-MARQUARDT ALGORITHM

/// " It can be seen that simple gradient descent and Gauss-Newton iteration are complementary in
/// Levenberg proposed an algorithm based on this observation, whose
/// update rule is a blend of the above mentioned algorithms and is given as
/// xk+1 = xk − (Hk + λI)−1∇f(xk)
/// where H is the Hessian matrix evaluated at xi. This update rule is used as follows. If the error
/// goes down following an update, it implies that our quadratic assumption on f  xЎ is working and
/// we reduce λ (usually by a factor of 10) to reduce the inﬂuence of gradient descent. On the other
/// hand, if the error goes up, we would like to follow the gradient more and so λ is increased by the
/// same factor. The Levenberg algorithm is thus -
/// 1. Do an update as directed by the rule above.
/// 2. Evaluate the error at the new parameter vector.
/// 3. If the error has increased as a result the update, then retract the step (i.e. reset the weights to
/// their previous values) and increase λ by a factor of 10 or some such signiﬁcant factor. Then
/// go to (1) and try an update again.
/// 4. If the error has decreased as a result of the update, then accept the step (i.e. keep the weights
/// at their new values) and decrease λ by a factor of 10 or so. "The Levenberg-Marquardt Algorithm" by Ananth Ranganathan

///

/// Classical Levenberg-Marquardt method.
#[derive(Debug, Clone, Copy)]
pub struct LevenbergMarquardtMethod {
    /// Initial damping.
    pub lambda_init: f64,
    /// Use `diag(J^T J)` instead of identity.
    pub diag_scaling: bool,
    /// Damping growth after rejection.
    pub increase_factor: f64,
    /// Damping decay after acceptance.
    pub decrease_factor: f64,
    /// Lower damping bound.
    pub min_lambda: f64,
    /// Upper damping bound.
    pub max_lambda: f64,
}

impl Default for LevenbergMarquardtMethod {
    fn default() -> Self {
        Self {
            lambda_init: 1e-3,
            diag_scaling: true,
            increase_factor: 3.0,
            decrease_factor: 10.0,
            min_lambda: 1e-6,
            max_lambda: 1e3,
        }
    }
}

//====================================================================================
// LEVENBERG-MARQUARDT METHOD
//====================================================================================
/// Internal state of classical LM.
#[derive(Debug, Clone)]
pub struct LevenbergMarquardtState {
    lambda: f64,
}

impl NonlinearMethod for LevenbergMarquardtMethod {
    type MethodState = LevenbergMarquardtState;

    fn init<P: JacobianProvider>(
        &self,
        _problem: &P,
        _x0: &DVector<f64>,
        _options: &SolveOptions,
        _residual: &DVector<f64>,
        _jacobian: &DMatrix<f64>,
    ) -> Result<Self::MethodState, SolveError> {
        if self.lambda_init <= 0.0 {
            return Err(SolveError::InvalidConfig(
                "lambda_init must be positive".to_string(),
            ));
        }
        if self.increase_factor <= 1.0 || self.decrease_factor <= 1.0 {
            return Err(SolveError::InvalidConfig(
                "increase_factor and decrease_factor must be greater than 1".to_string(),
            ));
        }
        if self.min_lambda <= 0.0 || self.max_lambda < self.min_lambda {
            return Err(SolveError::InvalidConfig(
                "invalid lambda bounds".to_string(),
            ));
        }
        Ok(LevenbergMarquardtState {
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
        let jtj = state.jacobian.transpose() * &state.jacobian;
        let scaling = if self.diag_scaling {
            DMatrix::from_diagonal(&jtj.diagonal())
        } else {
            DMatrix::identity(jtj.nrows(), jtj.ncols())
        };
        runtime.linear_solves += 1;
        let step = solve_linear_system(
            options.linear_solver,
            &(&jtj + method_state.lambda * scaling),
            &(-state.jacobian.transpose() * &state.residual),
        )?;
        if step.norm() < options.tolerance {
            return Ok(StepOutcome::Terminated(TerminationReason::StepTooSmall));
        }

        let trial_x = if let Some(bounds) = &options.bounds {
            bounds.project(&(&state.x + &step))
        } else {
            &state.x + &step
        };
        let trial_residual = problem.residual(&trial_x)?;
        let actual = state.residual.norm_squared() - trial_residual.norm_squared();
        let predicted = state.residual.norm_squared()
            - (&state.residual + &state.jacobian * &step).norm_squared();
        let rho = if predicted.abs() > 1e-12 {
            actual / predicted
        } else {
            0.0
        };

        if rho > 0.0 {
            method_state.lambda = (method_state.lambda / self.decrease_factor).max(self.min_lambda);
            runtime.accepted_steps += 1;
            Ok(StepOutcome::Continue {
                next_x: trial_x,
                accepted: true,
            })
        } else {
            method_state.lambda = (method_state.lambda * self.increase_factor).min(self.max_lambda);
            runtime.rejected_steps += 1;
            if method_state.lambda >= self.max_lambda {
                return Ok(StepOutcome::Terminated(TerminationReason::Stagnation));
            }
            Ok(StepOutcome::Continue {
                next_x: state.x.clone(),
                accepted: false,
            })
        }
    }
}

/// MODULE IS UNDER CONSTRUCTION
///
/// ALGORITHM: Levenberg-Marquardt Method (LMDER)

/// INPUT:
///  - fcn: user function that computes F(x) and Jacobian J(x)
///  - m: number of functions
///  - n: number of variables (n ≤ m)
///  - x: initial guess (length n)
///  - ftol, xtol, gtol: convergence tolerances
///  - maxfev: maximum function evaluations
///  - mode: scaling mode (1=auto, 2=user-provided)
///  - factor: initial step bound factor
///  - diag: scaling factors (if mode=2)
/*
OUTPUT:
  - x: solution vector
  - info: termination status
  - nfev, njev: function/jacobian evaluation counts

CONSTANTS:
  p1 = 0.1, p5 = 0.5, p25 = 0.25, p75 = 0.75, p0001 = 0.0001

INITIALIZATION:
  1. Validate input parameters
  2. Evaluate F(x₀) and compute fnorm = ||F(x₀)||
  3. Set par = 0 (Levenberg-Marquardt parameter)
  4. Set iter = 1

MAIN OUTER LOOP:
  REPEAT:

    // Compute Jacobian matrix
    5. Call fcn to compute J(x) at current x

    // QR factorization of Jacobian
    6. Compute QR factorization: J*P = Q*R
       Store permutation in ipvt[]

    // First iteration scaling
    7. IF iter == 1:
         IF mode == 1:
           Set diag[j] = column norms of J (or 1 if zero)
         Compute xnorm = ||diag ⊙ x||
         Set delta = factor * xnorm (or factor if xnorm=0)

    // Form Q^T * F(x)
    8. Compute qtf = first n components of Q^T * F(x)

    // Compute gradient norm
    9. Compute gnorm = ||J^T * F(x)|| (scaled)

    // Test gradient convergence
    10. IF gnorm ≤ gtol:
          Set info = 4 and TERMINATE

    // Update scaling
    11. IF mode == 1:
          Update diag[j] = max(diag[j], column_norm[j])

    INNER LOOP:
      REPEAT:

        // Solve trust region subproblem
        12. Call LMPAR to solve:
            (J^T*J + par*D²)*p = -J^T*F(x)
            subject to ||D*p|| ≤ delta
            Returns step p in wa1[]

        // Prepare trial point
        13. Set p = -wa1 (negate step)
            Set x_trial = x + p
            Compute pnorm = ||diag ⊙ p||

        // Adjust step bound on first iteration
        14. IF iter == 1:
              delta = min(delta, pnorm)

        // Evaluate at trial point
        15. Compute F(x_trial) and fnorm1 = ||F(x_trial)||

        // Compute actual reduction
        16. IF 0.1*fnorm1 < fnorm:
              actred = 1 - (fnorm1/fnorm)²
            ELSE:
              actred = -1

        // Compute predicted reduction
        17. Compute temp1 = ||J*p||/fnorm
            Compute temp2 = sqrt(par)*pnorm/fnorm
            prered = temp1² + temp2²/0.5
            dirder = -(temp1² + temp2²)

        // Compute reduction ratio
        18. IF prered ≠ 0:
              ratio = actred/prered
            ELSE:
              ratio = 0

        // Update trust region radius
        19. IF ratio > 0.25:
              IF par == 0 OR ratio < 0.75:
                delta = pnorm/0.5
                par = 0.5*par
            ELSE:
              IF actred ≥ 0: temp = 0.5
              IF actred < 0: temp = 0.5*dirder/(dirder + 0.5*actred)
              IF 0.1*fnorm1 ≥ fnorm OR temp < 0.1: temp = 0.1
              delta = temp * min(delta, pnorm/0.1)
              par = par/temp

        // Test for successful iteration
        20. IF ratio < 0.0001:
              CONTINUE inner loop (unsuccessful step)

        // Accept step
        21. x = x_trial
            F(x) = F(x_trial)
            fnorm = fnorm1
            iter = iter + 1

        // Test convergence
        22. IF |actred| ≤ ftol AND prered ≤ ftol AND 0.5*ratio ≤ 1:
              info = 1
        23. IF delta ≤ xtol*xnorm:
              info = 2
        24. IF conditions 22 AND 23:
              info = 3
        25. IF info ≠ 0: TERMINATE

        // Test for failure
        26. IF nfev ≥ maxfev: info = 5, TERMINATE
        27. IF |actred| ≤ machine_precision AND prered ≤ machine_precision AND 0.5*ratio ≤ 1:
              info = 6, TERMINATE
        28. IF delta ≤ machine_precision*xnorm: info = 7, TERMINATE
        29. IF gnorm ≤ machine_precision: info = 8, TERMINATE

        BREAK inner loop (successful step)

      END INNER LOOP

  END OUTER LOOP


*/

/// MINPACK-style Levenberg-Marquardt variant (approximation of `lmder` behavior).
#[derive(Debug, Clone)]
pub struct LevenbergMarquardtMinpack {
    /// tolerance on reduction in the sum of squares (ftol)
    pub ftol: f64,
    /// tolerance on solution step (xtol)
    pub xtol: f64,
    /// tolerance on scaled gradient (gtol)
    pub gtol: f64,
    /// maximum allowed function evaluations per outer call
    pub maxfev: usize,
    /// scaling mode: 1 for automatic, 2 for user-provided diag
    pub mode: i32,
    /// initial step bound factor
    pub factor: f64,
    /// optional user-provided diagonal scaling (used when mode == 2)
    pub diag: Option<DVector<f64>>,
}

impl Default for LevenbergMarquardtMinpack {
    fn default() -> Self {
        Self {
            ftol: 1e-8,
            xtol: 1e-8,
            gtol: 1e-8,
            maxfev: 1000,
            mode: 1,
            factor: 100.0,
            diag: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LMMinpackState {
    /// LM parameter par
    par: f64,
    /// trust-region bound delta
    delta: f64,
    /// scaling diagonal (D vector)
    diag: DVector<f64>,
    /// function and jacobian eval counters (method-local)
    nfev: usize,
    njev: usize,
}

impl LevenbergMarquardtMinpack {
    /// Small helper: compute (J^T J + par * D^2)
    fn build_augmented(&self, jtj: &DMatrix<f64>, par: f64, diag: &DVector<f64>) -> DMatrix<f64> {
        let n = jtj.nrows();
        let mut mat = jtj.clone();
        for i in 0..n {
            mat[(i, i)] += par * diag[i] * diag[i];
        }
        mat
    }

    /// lmpar-like simple bracket-and-bisect solver for par.
    /// Solves (JtJ + par D^2) p = -g  and aims to enforce ||D p|| ≤ delta.
    /// Returns (p, par).
    fn find_par(
        &self,
        solver: LinearSolverKind,
        jtj: &DMatrix<f64>,
        g: &DVector<f64>,
        diag: &DVector<f64>,
        delta: f64,
    ) -> Result<(DVector<f64>, f64), SolveError> {
        // try par = 0 first
        let mut par = 0.0;
        let mut p = solve_linear_system(solver, &self.build_augmented(jtj, par, diag), &(-g))?;
        let mut pnorm = scaled_norm(diag, &p);
        if pnorm <= delta {
            return Ok((p, par));
        }

        // find an upper bound for par by increasing until pnorm <= delta
        let mut par_lo = 0.0;
        let mut par_hi: f64 = 1.0;
        for _ in 0..60 {
            let mat = self.build_augmented(jtj, par_hi, diag);
            match solve_linear_system(solver, &mat, &(-g)) {
                Ok(p_try) => {
                    let pn = scaled_norm(diag, &p_try);
                    if pn <= delta {
                        p = p_try;
                        pnorm = pn;
                        par = par_hi;
                        break;
                    } else {
                        par_hi *= 10.0;
                    }
                }
                Err(_) => {
                    // If solve fails, grow par_hi and continue
                    par_hi *= 10.0;
                }
            }
        }

        if pnorm > delta {
            // perform bisection search between par_lo and par_hi
            for _ in 0..80 {
                let par_mid = 0.5 * (par_lo + par_hi);
                let mat = self.build_augmented(jtj, par_mid, diag);
                let p_try = solve_linear_system(solver, &mat, &(-g))?;
                let pn = scaled_norm(diag, &p_try);
                if pn <= delta {
                    par_hi = par_mid;
                    p = p_try;
                    pnorm = pn;
                } else {
                    par_lo = par_mid;
                }
                if (par_hi - par_lo).abs() < 1e-18 {
                    break;
                }
            }
            par = par_hi;
        }

        Ok((p, par))
    }
}

impl NonlinearMethod for LevenbergMarquardtMinpack {
    type MethodState = LMMinpackState;

    fn init<P: JacobianProvider>(
        &self,
        _problem: &P,
        x0: &DVector<f64>,
        _options: &SolveOptions,
        _residual: &DVector<f64>,
        jacobian: &DMatrix<f64>,
    ) -> Result<Self::MethodState, SolveError> {
        let n = jacobian.ncols();
        // build initial diag
        let diag = if self.mode == 2 {
            if let Some(d) = &self.diag {
                if d.len() != n {
                    return Err(SolveError::DimensionMismatch {
                        expected: n,
                        actual: d.len(),
                        context: "lm diag",
                    });
                }
                d.clone()
            } else {
                return Err(SolveError::InvalidConfig(
                    "mode=2 requires diag to be set".to_string(),
                ));
            }
        } else {
            scaling_vector(jacobian, true)
        };

        // compute initial delta = factor * ||diag .* x0||, or factor if zero
        let xscaled = diag.component_mul(x0);
        let xnorm = xscaled.norm();
        let delta = self.factor * if xnorm > 0.0 { xnorm } else { 1.0 };

        Ok(LMMinpackState {
            par: 0.0,
            delta,
            diag,
            nfev: 0,
            njev: 0,
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
        // MINPACK constants (used by rules below)
        let p1 = 0.1;
        let p5 = 0.5;
        let p25 = 0.25;
        let p75 = 0.75;
        let p0001 = 1e-4;
        let epsmch = f64::EPSILON;

        let n = state.jacobian.ncols();

        // Norm of current residual
        let fnorm = state.residual.norm();

        // Form J^T*J and gradient g = J^T * f
        let j = &state.jacobian;
        let jtj = j.transpose() * j;
        let g = j.transpose() * &state.residual;

        // Approximate scaled gradient norm (simple test)
        let mut gnorm: f64 = 0.0;
        if fnorm > 0.0 {
            for jcol in 0..n {
                let colnorm = j.column(jcol).norm();
                if colnorm > 0.0 {
                    // compute dot of column with qtf-like vector approximated by g/fnorm
                    let mut sum = 0.0;
                    for i in 0..n {
                        sum += jtj[(i, jcol)] * (g[i] / fnorm);
                    }
                    let denom = if self.mode == 2 {
                        method_state.diag[jcol]
                    } else {
                        colnorm
                    };
                    if denom != 0.0 {
                        gnorm = gnorm.max((sum.abs()) / denom);
                    }
                }
            }
        }
        if gnorm <= self.gtol {
            return Ok(StepOutcome::Terminated(TerminationReason::Converged));
        }

        // if mode==1, update diag to be max(diag, column_norm)
        if self.mode != 2 {
            for jcol in 0..n {
                let colnorm = j.column(jcol).norm();
                method_state.diag[jcol] = method_state.diag[jcol].max(colnorm);
                if method_state.diag[jcol] == 0.0 {
                    method_state.diag[jcol] = 1.0;
                }
            }
        }

        // Determine par and step with the shared MINPACK-style trust-region solver.
        let _solver_kind = options.linear_solver;
        runtime.linear_solves += 1;
        let subproblem = solve_trust_region_subproblem(
            &state.jacobian,
            &state.residual,
            &method_state.diag,
            method_state.delta,
            method_state.par,
        )
        .map_err(|message| SolveError::LinearSolveFailure(message.to_string()))?;
        let pvec = subproblem.step;
        let par = subproblem.lambda;
        method_state.par = par;

        // Compute pnorm and xnorm
        let pnorm = scaled_norm(&method_state.diag, &pvec);
        let xscaled = method_state.diag.component_mul(&state.x);
        let xnorm = xscaled.norm();

        // On first iteration adjust delta
        if state.iteration == 0 {
            method_state.delta = method_state.delta.min(pnorm);
        }

        // MINPACK computes a parameter update `p` and applies it as `x_new = x - p`.
        let mut trial_x = &state.x - &pvec;
        if let Some(bounds) = &options.bounds {
            trial_x = bounds.project(&trial_x);
        }

        // Evaluate residual at trial_x
        let trial_residual = problem
            .residual(&trial_x)
            .map_err(|e| SolveError::ResidualEvaluation(format!("{:?}", e)))?;
        method_state.nfev += 1;
        let fnorm1 = trial_residual.norm();

        // Compute actual reduction
        let actred = if p1 * fnorm1 < fnorm {
            1.0 - (fnorm1 / fnorm).powi(2)
        } else {
            -1.0
        };

        // Compute predicted reduction
        let j_p = &state.jacobian * &pvec;
        let temp1 = j_p.norm() / fnorm.max(1e-300);
        let temp2 = (par.sqrt() * pnorm) / fnorm.max(1e-300);
        let prered = temp1 * temp1 + (temp2 * temp2) / p5;
        let dirder = -(temp1 * temp1 + temp2 * temp2);

        let ratio = if prered != 0.0 { actred / prered } else { 0.0 };

        // Update delta and par following MINPACK-like rules (simplified)
        if ratio > p25 {
            if par == 0.0 || ratio < p75 {
                method_state.delta = pnorm / p5;
                method_state.par = 0.5 * par;
            } else {
                method_state.delta = pnorm / p5;
            }
        } else {
            let mut temp = p5;
            if actred < 0.0 {
                temp = p5 * dirder / (dirder + p5 * actred);
            }
            if p1 * fnorm1 >= fnorm || temp < p1 {
                temp = p1;
            }
            method_state.delta = temp * method_state.delta.min(pnorm / p1);
            if par != 0.0 {
                method_state.par = par / temp;
            }
        }

        // Decide acceptance
        if ratio > p0001 {
            runtime.accepted_steps += 1;
            return Ok(StepOutcome::Continue {
                next_x: trial_x,
                accepted: true,
            });
        } else {
            runtime.rejected_steps += 1;
        }

        // Termination (mapped from MINPACK conditions)
        if method_state.nfev >= self.maxfev {
            return Ok(StepOutcome::Terminated(TerminationReason::MaxIterations));
        }
        if actred.abs() <= epsmch && prered <= epsmch && p5 * ratio <= 1.0 {
            return Ok(StepOutcome::Terminated(TerminationReason::Stagnation));
        }
        if method_state.delta <= epsmch * xnorm {
            return Ok(StepOutcome::Terminated(TerminationReason::Stagnation));
        }

        // Not accepted, continue without update
        Ok(StepOutcome::Continue {
            next_x: state.x.clone(),
            accepted: false,
        })
    }
}
#[cfg(test)]
mod lm_minpack_tests {
    use super::*;
    use crate::numerical::Nonlinear_systems::engine::{SolveOptions, SolverEngine};
    use crate::numerical::Nonlinear_systems::error::TerminationReason;
    use approx::assert_relative_eq;
    use nalgebra::{DMatrix, DVector};

    // Minimal NonlinearProblem trait alias in your codebase is JacobianProvider.
    // We'll implement JacobianProvider for simple problems here.

    struct ScalarQuadratic; // f(x) = x^2 - 2 -> root sqrt(2)
    impl crate::numerical::Nonlinear_systems::problem::NonlinearProblem for ScalarQuadratic {
        fn dimension(&self) -> usize {
            1
        }
        fn residual(&self, x: &DVector<f64>) -> Result<DVector<f64>, SolveError> {
            Ok(DVector::from_vec(vec![x[0] * x[0] - 2.0]))
        }
    }
    impl crate::numerical::Nonlinear_systems::problem::JacobianProvider for ScalarQuadratic {
        fn jacobian(&self, x: &DVector<f64>) -> Result<DMatrix<f64>, SolveError> {
            Ok(DMatrix::from_row_slice(1, 1, &[2.0 * x[0]]))
        }
    }

    #[test]
    fn lm_minpack_scalar_quadratic_converges() {
        let method = LevenbergMarquardtMinpack {
            ftol: 1e-10,
            xtol: 1e-10,
            gtol: 1e-10,
            maxfev: 200,
            mode: 1,
            factor: 10.0,
            diag: None,
        };
        let options = SolveOptions {
            tolerance: 1e-8,
            max_iterations: 50,
            ..Default::default()
        };
        let engine = SolverEngine::new(method, options);
        let x0 = DVector::from_vec(vec![1.5]);
        let res = engine.solve(&ScalarQuadratic, x0).expect("solve failed");
        assert_eq!(res.termination, TerminationReason::Converged);
        let root = res.x[0];
        assert!(
            (root - 2f64.sqrt()).abs() < 1e-6,
            "root ~= sqrt(2): got {}",
            root
        );
    }

    // 2D system: x^2 + y^2 = 10, x - y = 4 => solutions around (3, -1)
    struct TwoEq;
    impl crate::numerical::Nonlinear_systems::problem::NonlinearProblem for TwoEq {
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
    impl crate::numerical::Nonlinear_systems::problem::JacobianProvider for TwoEq {
        fn jacobian(&self, x: &DVector<f64>) -> Result<DMatrix<f64>, SolveError> {
            Ok(DMatrix::from_row_slice(
                2,
                2,
                &[2.0 * x[0], 2.0 * x[1], 1.0, -1.0],
            ))
        }
    }

    #[test]
    fn lm_minpack_two_eq_converges() {
        let method = LevenbergMarquardtMinpack::default();
        let options = SolveOptions {
            tolerance: 1e-8,
            max_iterations: 100,
            ..Default::default()
        };
        let engine = SolverEngine::new(method, options);
        let x0 = DVector::from_vec(vec![1.0, 1.0]);
        let res = engine.solve(&TwoEq, x0).expect("solve failed");
        assert_eq!(res.termination, TerminationReason::Converged);
        let x = res.x;
        assert!((x[0] - 3.0).abs() < 1e-4, "x[0] close to 3: got {}", x[0]);
        assert!((x[1] + 1.0).abs() < 1e-4, "x[1] close to -1: got {}", x[1]);
    }

    // Coupled nonlinear example: system with known root
    // f1 = x^2 + y - 37 = 0
    // f2 = x - y^2 - 5 = 0
    // (one solution near x=6, y=1)
    struct Coupled;
    impl crate::numerical::Nonlinear_systems::problem::NonlinearProblem for Coupled {
        fn dimension(&self) -> usize {
            2
        }
        fn residual(&self, x: &DVector<f64>) -> Result<DVector<f64>, SolveError> {
            Ok(DVector::from_vec(vec![
                x[0] * x[0] + x[1] - 37.0,
                x[0] - x[1] * x[1] - 5.0,
            ]))
        }
    }
    impl crate::numerical::Nonlinear_systems::problem::JacobianProvider for Coupled {
        fn jacobian(&self, x: &DVector<f64>) -> Result<DMatrix<f64>, SolveError> {
            Ok(DMatrix::from_row_slice(
                2,
                2,
                &[2.0 * x[0], 1.0, 1.0, -2.0 * x[1]],
            ))
        }
    }

    #[test]
    fn lm_minpack_coupled_converges() {
        let method = LevenbergMarquardtMinpack {
            ftol: 1e-10,
            xtol: 1e-10,
            gtol: 1e-10,
            maxfev: 500,
            mode: 1,
            factor: 10.0,
            diag: None,
        };
        let options = SolveOptions {
            tolerance: 1e-8,
            max_iterations: 200,
            ..Default::default()
        };
        let engine = SolverEngine::new(method, options);
        let x0 = DVector::from_vec(vec![6.0, 1.0]);
        let res = engine.solve(&Coupled, x0).expect("solve failed");
        assert_eq!(res.termination, TerminationReason::Converged);
        let x = res.x;
        println!("x = {:?}", x);
        assert_relative_eq!(x[0], 6.0, epsilon = 1e-6);
        assert_relative_eq!(x[1], 1.0, epsilon = 1e-6);
        assert!((x[0] - 6.0).abs() < 1e-4, "x[0] close to 6: got {}", x[0]);
        // check equations nearly zero
        //   let r = Coupled.residual(&x).expect("residual");
        //   assert!(r[0].abs() < 1e-6 && r[1].abs() < 1e-6, "residuals not small: {:?}", r);
    }
}
