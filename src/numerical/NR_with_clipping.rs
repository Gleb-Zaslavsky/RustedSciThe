use super::NR::NR;
use crate::numerical::BVP_Damp::BVP_utils::elapsed_time;
use log::{info, warn};
use nalgebra::DVector;
use std::time::Instant;
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

impl NR {
    pub fn step_with_clipping(&mut self) -> (i32, Option<DVector<f64>>) {
        // DAMPED NEWTON STEPS
        // BOUNDS ARE SET
        let maxDampIter =  self.parameters.clone().unwrap()["maxDampIter"] as usize;
        // let safeguard_factor: f64 = 1e-1;
        // let DampFacor: f64 = 0.5;
        let now = Instant::now();
        let y_k_minus_1 = self.y.clone();

        // Compute Newton step:
        let undamped_step_k_minus_1 = self.step(y_k_minus_1.clone());
        let (undamped_step_k_minus_1, F_k_minus_1) = undamped_step_k_minus_1.clone();
        let S_k_minus_1 = F_k_minus_1.norm();
        info!("\n \n L2 norm of damped step = {}", S_k_minus_1);
        self.custom_timer.linear_system_tac();
        *self
            .calc_statistics
            .entry("number of solving linear systems".to_string())
            .or_insert(0) += 1;

        // preallocate variables
        let mut k_: usize = 0;
        let mut S_k_plus_1: Option<f64> = None;
        let mut damped_step_result: Option<DVector<f64>> = None;
        let mut conv: f64 = 0.0;
        let mut lambda = 1.0;
        let mut k = 0;
        while k < maxDampIter {
            info!(
                "\n____________________________________damping iteration = {}_______________________________________",
                k
            );
            if k > 1 {
                info!("\n \n damped_step number {} ", k);
            }
            info!("\n \n Damping coefficient = {}", lambda);
            // Compute damped step:

            let damped_step_k_minus_1 = lambda * undamped_step_k_minus_1.clone();
            // compute next iteration guess with current damping coefficient
            let y_k: DVector<f64> = y_k_minus_1.clone() - damped_step_k_minus_1;
            //dbg!(&y_k);
            // clip the result vector i.e.
            let y_k_clipped: DVector<f64> = self.clip(&y_k, &self.bounds_vec);

            //
            let F_k = self.evaluate_function(y_k_clipped.clone()); //???????????????

            let error = F_k.norm();
            self.max_error = error;
            info!("\n \n L2 norm of undamped step = {}", error);
            let convergence_cond_for_step = self.tolerance;
            // compute the norm of the undamped step
            let S_k_temp = error;
            // If the norm of S_k_plus_1 is less than the norm of S_k, then accept this
            // damping coefficient. Also accept it if this step would result in a
            // converged solution. Otherwise, decrease the damping coefficient and
            // try again.
            let elapsed = now.elapsed();
            elapsed_time(elapsed);
            if (S_k_temp < S_k_minus_1) || (S_k_temp < convergence_cond_for_step) {
                // The  criterion for accepting is that the undamped steps decrease in
                // magnitude, This prevents the iteration from stepping away from the region where there is good reason to believe a solution lies

                k_ = k;
                S_k_plus_1 = Some(S_k_temp);
                // update the solution vector
                damped_step_result = Some(y_k_clipped.clone());
                conv = convergence_cond_for_step;
                info!(
                    "\n \n  Damping coefficient accepted S_k_plus_1_temp < S_k {}, {},",
                    S_k_temp, S_k_minus_1,
                );
                info!(
                    "norm of undamped step = {} vs convergence criterion = {}",
                    S_k_temp, convergence_cond_for_step
                );
                info!("damped step result = {:?}", &damped_step_result);
                break;
            }
            // if fail this criterion we must reject it and retries the step with a reduced (often halved) damping parameter trying again until
            // criterion is met  or max damping iterations is reached

            // lambda = lambda / (2.0f64.powf(k as f64 + DampFacor));
            lambda *= 0.5; // Simpler and more controlled decay
            k_ = k;
            S_k_plus_1 = Some(S_k_temp);

            k = k + 1;
        }

        if k_ < maxDampIter {
            // if there is a damping coefficient found (so max damp steps not exceeded)
            if S_k_plus_1.unwrap() > conv {
                //found damping coefficient but not converged yet
                info!("\n \n  Damping coefficient found (solution has not converged yet)");
                info!(
                    "\n \n  step norm =  {}, weight norm = {}, damped_step_result = {}",
                    self.max_error,
                    S_k_plus_1.unwrap(),
                    conv
                );
                (0, damped_step_result)
            } else {
                info!("\n \n  Damping coefficient found (solution has converged)");
                info!(
                    "\n \n step norm =  {}, weight norm = {}, convergence condition = {}",
                    self.max_error,
                    S_k_plus_1.unwrap(),
                    conv
                );
                (1, damped_step_result)
            }
        } else {
            //  if we have reached max damping iterations without finding a damping coefficient we must reject the step
            warn!("\n \n  No damping coefficient found (max damping iterations reached)");
            return (-2, None);
        }
    }

    pub fn clip(&self, y: &DVector<f64>, vec_of_bounds: &Vec<(f64, f64)>) -> DVector<f64> {
        let mut clipped_y = y.clone();
        for (i, y_i) in y.iter().enumerate() {
            let (lower_bound, upper_bound) = vec_of_bounds[i];
            if *y_i < lower_bound {
                clipped_y[i] = lower_bound;
            } else if *y_i > upper_bound {
                clipped_y[i] = upper_bound;
            }
        }
        clipped_y
    }
}
