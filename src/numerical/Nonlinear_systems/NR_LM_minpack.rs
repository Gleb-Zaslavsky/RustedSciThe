use super::NR::{NR, solve_linear_system};
use crate::numerical::BVP_Damp::BVP_utils::elapsed_time;
use log::{info, warn};
use nalgebra::{DMatrix, DVector};
use std::time::Instant;
/// DONT USE IN PRODUCTION CODE
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

TERMINATION CODES:
  info = 1: Both actual and predicted reductions ≤ ftol
  info = 2: Relative error between iterates ≤ xtol
  info = 3: Both conditions 1 and 2 hold
  info = 4: Gradient orthogonality ≤ gtol
  info = 5: Maximum function evaluations reached
  info = 6: ftol too small (no further reduction possible)
  info = 7: xtol too small (no further improvement possible)
  info = 8: gtol too small (gradient orthogonal to machine precision)
*/
impl NR {
    pub fn step_lm_minpack(&mut self) -> (i32, Option<DVector<f64>>) {
        let now = Instant::now();
        let bounds_vec = self.bounds_vec.clone();
        let method = self.linear_sys_method.clone().unwrap();
        let parameters = self.parameters.clone().unwrap();
        let diag_flag: bool = if parameters.get("diag").unwrap() == &1.0 {
            true
        } else {
            false
        };
        let decrease_factor = parameters.get("decrease_factor").unwrap();
        let increase_factor = parameters.get("increase_factor").unwrap();
        let max_lambda = parameters.get("max_lambda").unwrap();
        let min_lambda = parameters.get("min_lambda").unwrap();
        let mut lambda = self.dumping_factor.clone();
        info!("lambda = {}", lambda);
        let y = self.y.clone();
        let Fy = self.evaluate_function(y.clone());
        let Jy = self.evaluate_jacobian(y.clone());
        // Check convergence
        let old_norm = Fy.norm_squared();
        info!("norm = {}", old_norm);
        if old_norm < self.tolerance {
            let y_result = y.clone();
            info!("Solution found!");
            return (1, Some(y_result));
        }
        // Solve (J^T J + λI)p = -J^T F
        let JtJ = Jy.transpose() * &Jy;
        let D: DMatrix<f64> = if diag_flag {
            DMatrix::from_diagonal(&JtJ.diagonal())
        } else {
            DMatrix::identity(y.len(), y.len())
        };
        let A = JtJ + lambda * D * DMatrix::identity(y.len(), y.len());
        let b = -Jy.transpose() * &Fy;
        let step = solve_linear_system(method, &A, &b).unwrap();

        let mut y_new = y.clone() + &step;

        // Project onto bounds (if they exist)
        y_new = self.clip(&y_new, &bounds_vec);

        let Fy_new = self.evaluate_function(y_new.clone());
        // let rho = (old_norm - Fy_new.norm_squared())/old_norm;

        let actual_reduction = Fy.norm_squared() - Fy_new.norm_squared();
        let predicted_reduction = Fy.norm_squared() - (Fy.clone() + Jy * &step).norm_squared();
        let rho = if predicted_reduction.abs() > 1e-12 {
            actual_reduction / (predicted_reduction + 1e-12)
        } else {
            0.0
        };

        info!("rho is: {}", rho);
        let elapsed = now.elapsed();
        elapsed_time(elapsed);
        if rho > 0.0 {
            lambda = (lambda / decrease_factor).max(*min_lambda);
            if lambda == *min_lambda {
                warn!("lambda reached minimum!");
            }
            self.dumping_factor = lambda;
            info!("lambda decreased: {}", lambda);

            return (0, Some(y_new));
        } else {
            lambda = (lambda * increase_factor).min(*max_lambda);
            if lambda == *max_lambda {
                warn!("lambda reached maximum!");
            }
            self.dumping_factor = lambda;
            info!("lambda increased: {}", lambda);
            return (0, Some(y.clone()));
        }
    }
}
