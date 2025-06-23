use crate::numerical::optimization::problem_LM::LeastSquaresProblem;
use crate::numerical::optimization::qr_LM::{LinearLeastSquaresDiagonalProblem, PivotedQR};
use crate::numerical::optimization::trust_region_LM::{
    LMParameter, determine_lambda_and_parameter_update,
};
use crate::numerical::optimization::utils::{enorm, epsmch};
use nalgebra::{DMatrix, DVector};
use num_traits::Float;
// Global boolean flag to control MINPACK compatibility
const MINPACK_COMPAT: bool = false; // Set to true for MINPACK compatibility, false for modern behavior
/*
#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::excessive_precision,
    clippy::redundant_clone
)]
pub(crate) mod test_examples;
#[cfg(test)]
mod test_helpers;
#[cfg(test)]
mod test_init_step;
#[cfg(test)]
#[allow(clippy::float_cmp, clippy::clone_on_copy, clippy::redundant_clone)]
mod test_update_diag;
*/
/// Reasons for terminating the minimization.
#[derive(PartialEq, Eq, Debug)]
pub enum TerminationReason {
    /// The residual or Jacobian computation was not successful, it returned `None`.
    User(&'static str),
    /// Encountered `NaN` or inf`.
    Numerical(&'static str),
    /// The residuals are literally zero.
    ResidualsZero,
    /// The residuals vector and the Jacobian columns are almost orthogonal.
    ///
    /// This is the `gtol` termination criterion.
    Orthogonal,
    /// The `ftol` or `xtol` criterion was fulfilled.
    Converged { ftol: bool, xtol: bool },
    /// The bound for `ftol`, `xtol` or `gtol` was set so low that the
    /// test passed with the machine epsilon but not with the actual
    /// bound. This means you must increase the bound.
    NoImprovementPossible(&'static str),
    /// Maximum number of function evaluations was hit.
    LostPatience,
    /// The number of parameters n is zero.
    NoParameters,
    /// The number of residuals m is zero.
    NoResiduals,
    /// The dimensions of the problem are wrong.
    WrongDimensions(&'static str),
}

impl TerminationReason {
    /// Compute whether the outcome is considered successful.
    ///
    /// This does not necessarily mean we have a minimizer.
    /// Some termination criteria are approximations for necessary
    /// optimality conditions or check limitations due to
    /// floating point arithmetic.
    pub fn was_successful(&self) -> bool {
        matches!(
            self,
            TerminationReason::ResidualsZero
                | TerminationReason::Orthogonal
                | TerminationReason::Converged { .. }
        )
    }
    /// A fundamental assumptions was not met.
    ///
    /// For example if the number of residuals changed.
    pub fn was_usage_issue(&self) -> bool {
        matches!(
            self,
            TerminationReason::NoParameters
                | TerminationReason::NoResiduals
                | TerminationReason::NoImprovementPossible(_)
                | TerminationReason::WrongDimensions(_)
        )
    }
}
/// Information about the minimization.
///
/// Use this to inspect the minimization process. Most importantly
/// you may want to check if there was a failure.
#[derive(Debug)]
pub struct MinimizationReport {
    pub termination: TerminationReason,
    /// Number of residuals which were computed.
    pub number_of_evaluations: usize,
    /// Contains the value of x
    pub objective_function: f64,
}
/// Levenberg-Marquardt optimization algorithm.
///
/// See the [module documentation](index.html) for a usage example.
///
/// The runtime and termination behavior can be controlled by various hyperparameters.

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LevenbergMarquardt {
    ftol: f64,
    xtol: f64,
    gtol: f64,
    stepbound: f64,
    patience: usize,
    scale_diag: bool,
}

impl Default for LevenbergMarquardt {
    fn default() -> Self {
        Self::new()
    }
}

impl LevenbergMarquardt {
    pub fn new() -> Self {
        if MINPACK_COMPAT {
            let user_tol = 1.49012e-08;
            Self {
                ftol: user_tol,
                xtol: user_tol,
                gtol: 0.0,
                stepbound: 100.0,
                patience: 100,
                scale_diag: true,
            }
        } else {
            let user_tol = f64::EPSILON * 30.0;
            Self {
                ftol: user_tol,
                xtol: user_tol,
                gtol: user_tol,
                stepbound: 100.0,
                patience: 100,
                scale_diag: true,
            }
        }
    }

    /// Set the relative error desired in the objective function f.
    ///
    /// Termination occurs when both the actual and
    /// predicted relative reductions for f are at most `ftol`.
    ///
    /// # Panics
    ///
    /// Panics if ftol is negative.

    #[must_use]
    pub fn with_ftol(self, ftol: f64) -> Self {
        assert!(!ftol.is_sign_negative(), "ftol must be >= 0");
        Self { ftol, ..self }
    }

    /// Set relative error between last two approximations.
    ///
    /// Termination occurs when the relative error between
    /// two consecutive iterates is at most `xtol`.
    ///
    /// # Panics
    ///
    /// Panics if xtol is negative.
    ///
    #[must_use]
    pub fn with_xtol(self, xtol: f64) -> Self {
        assert!(!xtol.is_sign_negative(), "xtol must be >= 0");
        Self { xtol, ..self }
    }
    /// Set orthogonality desired between the residual vector and its derivative.
    ///
    /// Termination occurs when the cosine of the angle
    /// between the residual vector `$\vec{r}$` and any column of the Jacobian `$\mathbf{J}$` is at
    /// most `gtol` in absolute value.
    ///
    /// With other words, the algorithm will terminate if
    /// ```math
    ///   \cos\bigl(\sphericalangle (\mathbf{J}\vec{e}_i, \vec{r})\bigr) =
    ///   \frac{|(\mathbf{J}^\top \vec{r})_i|}{\|\mathbf{J}\vec{e}_i\|\|\vec{r}\|} \leq \texttt{gtol}
    ///   \quad\text{for all }i=1,\ldots,n.
    /// ```
    ///
    /// This is based on the fact that those vectors are orthogonal near the optimum (gradient is zero).
    /// The angle check is scale invariant, whereas checking that condition
    /// "nabla f(x) approximates zero" is not.
    ///
    /// # Panics
    ///
    /// Panics if `$\mathtt{gtol} < 0$`.
    #[must_use]
    pub fn with_gtol(self, gtol: f64) -> Self {
        assert!(!gtol.is_sign_negative(), "gtol must be >= 0");
        Self { gtol, ..self }
    }
    /// Shortcut to set `tol` as in MINPACK `LMDER1`.
    ///
    /// Sets `ftol = xtol = tol` and `gtol = 0`.
    ///
    /// # Panics
    ///
    /// Panics if `tol<=0`.
    #[must_use]
    pub fn with_tol(self, tol: f64) -> Self {
        assert!(tol.is_sign_positive(), "tol must > 0");
        Self {
            ftol: tol,
            xtol: tol,
            gtol: 0.0,
            ..self
        }
    }
    /// Set factor for the initial step bound.
    ///
    /// This bound is set to stepbound*||D*x||` where `D` is the diagonal
    /// if nonzero, or else to `stepbound` itself. In most cases `stepbound` should lie
    /// in the interval `$[0.1,100]$`.
    ///
    /// # Panics
    ///
    /// Panics if `stepbound <= 0`.
    #[must_use]
    pub fn with_stepbound(self, stepbound: f64) -> Self {
        assert!(stepbound.is_sign_positive(), "stepbound must be > 0");
        Self { stepbound, ..self }
    }
    /// Set factor for the maximal number of function evaluations.
    ///
    /// The maximal number of function evaluations is set to
    /// patince*(n+1)
    ///
    /// # Panics
    ///
    /// Panics if `patience <= 0`.
    #[must_use]
    pub fn with_patience(self, patience: usize) -> Self {
        assert!(patience > 0, "patience must be > 0");
        Self { patience, ..self }
    }
    /// Enable or disable whether the variables will be rescaled internally.
    #[must_use]
    pub fn with_scale_diag(self, scale_diag: bool) -> Self {
        Self { scale_diag, ..self }
    }
    /// Try to solve the given least squares problem.
    ///
    /// The parameters of the problem which are set when this function is called
    /// are used as the initial guess for `x (vector)`.
    pub fn minimize<O>(&self, target: O) -> (O, MinimizationReport)
    where
        O: LeastSquaresProblem,
    {
        let (mut lm, mut residuals) = match LM::new(self, target) {
            Err(report) => return report,
            Ok(res) => res,
        };
        let n = lm.x.nrows();
        loop {
            // Build linear least squaress problem used for the trust-region subproblem
            let mut lls = {
                let jacobian = match lm.jacobian() {
                    Err(reason) => return lm.into_report(reason),
                    Ok(jacobian) => jacobian,
                };
                if jacobian.ncols() != n || jacobian.nrows() != lm.m {
                    return lm.into_report(TerminationReason::WrongDimensions("jacobian"));
                }

                let qr = PivotedQR::new(jacobian);
                qr.into_least_squares_diagonal_problem(residuals)
            };
            // Update the diagonal, initialize "delta" in first call
            if let Err(reason) = lm.update_diag(&mut lls) {
                return lm.into_report(reason);
            };

            residuals = loop {
                let param =
                    determine_lambda_and_parameter_update(&mut lls, &lm.diag, lm.delta, lm.lambda);
                let tr_iteration = lm.trust_region_iteration(&mut lls, param);
                match tr_iteration {
                    // successful parameter update, break and recompute Jacobian
                    Ok(Some(residuals)) => break residuals,
                    // terminate (either success or failure)
                    Err(reason) => return lm.into_report(reason),
                    // need another iteration
                    Ok(None) => (),
                }
            };
        }
    }
}
/// Struct which holds the state of the LM algorithm and which implements its individual steps.
struct LM<'a, O>
where
    O: LeastSquaresProblem,
{
    config: &'a LevenbergMarquardt,
    /// Current parameters x (vector)
    x: DVector<f64>,
    tmp: DVector<f64>,
    /// The implementation of `LeastSquaresProblem`
    target: O,
    /// Statistics and termination reasons, used for return value
    report: MinimizationReport,
    /// The delta from the trust-region algorithm
    delta: f64,
    lambda: f64,
    /// ||DX||
    xnorm: f64,
    gnorm: f64,
    residuals_norm: f64,
    /// The diagonal of D
    diag: DVector<f64>,
    /// Flag to check if it is the first trust region iteration
    first_trust_region_iteration: bool,
    /// Flag to check if it is the first diagonal update
    first_update: bool,
    max_fev: usize,
    m: usize,
}

impl<'a, O> LM<'a, O>
where
    O: LeastSquaresProblem,
{
    fn new(
        config: &'a LevenbergMarquardt,
        target: O,
    ) -> Result<(Self, DVector<f64>), (O, MinimizationReport)> {
        let mut report = MinimizationReport {
            termination: TerminationReason::ResidualsZero,
            number_of_evaluations: 1,
            objective_function: f64::NAN,
        };
        // Evaluate at start point
        let x = target.params();
        let (residuals, residuals_norm) = if let Some(residuals) = target.residuals() {
            let norm = enorm(&residuals);
            report.objective_function = norm * norm * 0.5;
            (residuals, norm)
        } else {
            return Err((
                target,
                MinimizationReport {
                    termination: TerminationReason::User("residuals"),
                    ..report
                },
            ));
        };
        // Initialize diagonal
        let n = x.nrows();
        // Check n > 0
        let diag = DVector::from_element(n, 1.0);
        if diag.nrows() == 0 {
            return Err((
                target,
                MinimizationReport {
                    termination: TerminationReason::NoParameters,
                    ..report
                },
            ));
        }

        let m: usize = residuals.nrows();
        if m == 0 {
            return Err((
                target,
                MinimizationReport {
                    termination: TerminationReason::NoResiduals,
                    ..report
                },
            ));
        }

        if !residuals_norm.is_finite() && !MINPACK_COMPAT {
            return Err((
                target,
                MinimizationReport {
                    termination: TerminationReason::Numerical("residuals norm"),
                    ..report
                },
            ));
        }

        if residuals_norm <= f64::MIN_POSITIVE && !MINPACK_COMPAT {
            return Err((target, report));
        }

        Ok((
            Self {
                config,
                target,
                report,
                tmp: x.clone(),
                x,
                diag,
                delta: 0.0,
                lambda: 0.0,
                xnorm: 0.0,
                gnorm: 0.0,
                residuals_norm,
                first_trust_region_iteration: true,
                first_update: true,
                max_fev: config.patience * (n + 1),
                m,
            },
            residuals,
        ))
    }

    fn into_report(self, termination: TerminationReason) -> (O, MinimizationReport) {
        (
            self.target,
            MinimizationReport {
                termination,
                ..self.report
            },
        )
    }

    fn jacobian(&self) -> Result<DMatrix<f64>, TerminationReason> {
        match self.target.jacobian() {
            Some(jacobian) => Ok(jacobian),
            None => Err(TerminationReason::User("jacobian")),
        }
    }
    // Compute norm of scaled gradient and detect degeneracy
    fn update_diag(
        &mut self,
        lls: &mut LinearLeastSquaresDiagonalProblem,
    ) -> Result<(), TerminationReason> {
        self.gnorm = match lls.max_a_t_b_scaled(self.residuals_norm) {
            Some(max_at_b) => max_at_b,
            None if !MINPACK_COMPAT => return Err(TerminationReason::Numerical("jacobian")),
            None => 0.0,
        };
        if self.gnorm <= self.config.gtol {
            return Err(TerminationReason::Orthogonal);
        }

        if self.first_update {
            // Initialize diag and xnorm
            self.xnorm = if self.config.scale_diag {
                for (d, col_norm) in self.diag.iter_mut().zip(lls.column_norms.iter()) {
                    *d = if *col_norm == 0.0 { 1.0 } else { *col_norm };
                }
                self.tmp.copy_from(&self.x);
                self.tmp.component_mul_assign(&self.diag);
                enorm(&self.tmp)
            } else {
                enorm(&self.x)
            };
            if !self.xnorm.is_finite() && !MINPACK_COMPAT {
                return Err(TerminationReason::Numerical("subproblem x"));
            }
            // Initialize delta
            self.delta = if self.xnorm == 0.0 {
                self.config.stepbound
            } else {
                self.config.stepbound * self.xnorm
            };
            self.first_update = false;
        } else if self.config.scale_diag {
            // Update diag
            for (d, norm) in self.diag.iter_mut().zip(lls.column_norms.iter()) {
                *d = Float::max(*norm, *d);
            }
        }
        Ok(())
    }

    fn trust_region_iteration(
        &mut self,
        lls: &mut LinearLeastSquaresDiagonalProblem,
        param: LMParameter,
    ) -> Result<Option<DVector<f64>>, TerminationReason> {
        const P1: f64 = 0.1;
        const P0001: f64 = 1.0e-4;

        self.lambda = param.lambda;
        let pnorm = param.dp_norm;
        if !pnorm.is_finite() && !MINPACK_COMPAT {
            return Err(TerminationReason::Numerical("subproblem ||Dp||"));
        }

        let predicted_reduction;
        let dir_der;
        {
            let temp1 = (lls.a_x_norm(&param.step) / self.residuals_norm).powi(2);
            if !temp1.is_finite() && !MINPACK_COMPAT {
                return Err(TerminationReason::Numerical("trust-region reduction"));
            }
            let temp2 = ((self.lambda.sqrt() * pnorm) / self.residuals_norm).powi(2);
            if !temp2.is_finite() && !MINPACK_COMPAT {
                return Err(TerminationReason::Numerical("trust-region reduction"));
            }
            predicted_reduction = temp1 + temp2 / 0.5;
            dir_der = -(temp1 + temp2);
        }

        if self.first_trust_region_iteration && pnorm < self.delta {
            self.delta = pnorm;
        }
        self.first_trust_region_iteration = false;
        // Compute new parameters: x - p
        self.tmp.copy_from(&self.x);
        self.tmp -= &param.step;
        // Evaluate
        self.target.set_params(&self.tmp);
        self.report.number_of_evaluations += 1;
        let new_objective_function;
        let (residuals, new_residuals_norm) = if let Some(residuals) = self.target.residuals() {
            if residuals.nrows() != self.m {
                return Err(TerminationReason::WrongDimensions("residuals"));
            }
            let norm = enorm(&residuals);
            new_objective_function = norm * norm * 0.5;
            (residuals, norm)
        } else {
            return Err(TerminationReason::User("residuals"));
        };
        // Compute predicted and actual reduction
        let actual_reduction = if new_residuals_norm * P1 < self.residuals_norm {
            1.0 - (new_residuals_norm / self.residuals_norm).powi(2)
        } else {
            -1.0
        };

        let ratio = if predicted_reduction == 0.0 {
            0.0
        } else {
            actual_reduction / predicted_reduction
        };
        let half = 0.5;
        if ratio <= 0.25 {
            let mut temp = if !actual_reduction.is_sign_negative() {
                half
            } else {
                half * dir_der / (dir_der + half * actual_reduction)
            };
            if new_residuals_norm * P1 >= self.residuals_norm || temp < P1 {
                temp = P1;
            };
            self.delta = temp * f64::min(self.delta, pnorm * 10.0);
            self.lambda /= temp;
        } else if self.lambda == 0.0 || ratio >= 0.75 {
            self.delta = pnorm / 0.5;
            self.lambda *= half;
        }

        let update_considered_good = ratio >= P0001;
        if update_considered_good {
            // update x, residuals and their norms
            core::mem::swap(&mut self.x, &mut self.tmp);
            self.xnorm = if self.config.scale_diag {
                self.tmp.copy_from(&self.x);
                self.tmp.component_mul_assign(&self.diag);
                enorm(&self.tmp)
            } else {
                enorm(&self.x)
            };
            if !self.xnorm.is_finite() && !MINPACK_COMPAT {
                return Err(TerminationReason::Numerical("new x"));
            }
            self.residuals_norm = new_residuals_norm;
            self.report.objective_function = new_objective_function;
        }

        if !MINPACK_COMPAT && self.residuals_norm <= f64::MIN_POSITIVE {
            self.reset_params_if(!update_considered_good);
            return Err(TerminationReason::ResidualsZero);
        }
        let ftol_check = f64::abs(actual_reduction) <= self.config.ftol
            && predicted_reduction <= self.config.ftol
            && ratio * 0.5 <= 1.0;
        let xtol_check = self.delta <= self.config.xtol * self.xnorm;
        if ftol_check || xtol_check {
            self.reset_params_if(!update_considered_good);
            return Err(TerminationReason::Converged {
                ftol: ftol_check,
                xtol: xtol_check,
            });
        }
        // termination tests
        if self.report.number_of_evaluations >= self.max_fev {
            self.reset_params_if(!update_considered_good);
            return Err(TerminationReason::LostPatience);
        }

        // We now check if one of the ftol, xtol or gtol criteria
        // is fulfilld with the machine epsilon.
        if f64::abs(actual_reduction) <= epsmch()
            && predicted_reduction <= epsmch()
            && ratio * 0.5 <= 1.0
        {
            self.reset_params_if(!update_considered_good);
            return Err(TerminationReason::NoImprovementPossible("ftol"));
        }
        if self.delta <= epsmch::<f64>() * self.xnorm {
            self.reset_params_if(!update_considered_good);
            return Err(TerminationReason::NoImprovementPossible("xtol"));
        }
        if self.gnorm <= epsmch() {
            self.reset_params_if(!update_considered_good);
            return Err(TerminationReason::NoImprovementPossible("gtol"));
        }

        if update_considered_good {
            Ok(Some(residuals))
        } else {
            Ok(None)
        }
    }

    #[inline]
    fn reset_params_if(&mut self, reset: bool) {
        if reset {
            self.target.set_params(&self.x);
        }
    }
}
