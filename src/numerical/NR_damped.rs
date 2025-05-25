use super::NR::{NR, bound_step};
use crate::numerical::BVP_Damp::BVP_utils::elapsed_time;
use log::{error, info, warn};
use nalgebra::DVector;
use std::time::Instant;
impl NR {
    pub fn step_damped(&mut self) -> (i32, Option<DVector<f64>>) {
        // DAMPED NEWTON STEPS
        // BOUNDS ARE SET
        let maxDampIter = 10;
        let DampFacor: f64 = 0.5;
        let now = Instant::now();
        let y_k_minus_1 = self.y.clone();
        let undamped_step_k_minus_1 = self.step(y_k_minus_1.clone()).0;
        let lambda = self.dumping_factor;
        let y_k: DVector<f64> = y_k_minus_1.clone() - lambda * undamped_step_k_minus_1.clone();
        let fbound = bound_step(&y_k_minus_1, &undamped_step_k_minus_1, &self.bounds_vec).abs(); // absolute value of the step

        if fbound.is_nan() {
            error!("\n \n fbound is NaN \n \n");
            panic!()
        }
        if fbound.is_infinite() {
            error!("\n \n fbound is infinite \n \n");
            panic!()
        }
        // let fbound =1.0;
        info!("\n \n fboundary  = {}", fbound);
        let mut lambda = 1.0 * fbound;
        // if fbound is very small, then x0 is already close to the boundary and
        // step0 points out of the allowed domain. In this case, the Newton
        // algorithm fails, so return an error condition.
        if fbound.abs() < 1e-10 {
            log::warn!(
                "\n  No damped step can be taken without violating solution component bounds."
            );
            return (-3, None);
        }
        let mut k_: usize = 0;
        let mut S_k_plus_1: Option<f64> = None;
        let mut damped_step_result: Option<DVector<f64>> = None;
        let mut conv: f64 = 0.0;
        //   let fun = &self.fun;
        // calculate damped step
        let undamped_step_k = self.step(y_k.clone());

        self.custom_timer.linear_system_tac();
        *self
            .calc_statistics
            .entry("number of solving linear systems".to_string())
            .or_insert(0) += 1;
        //
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
            let damped_step_k = undamped_step_k.clone().0 * lambda;
            let S_k = &damped_step_k.norm();
            info!("\n \n L2 norm of damped step = {}", S_k);
            let y_k_plus_1: DVector<f64> = y_k.clone() - damped_step_k;

            // / compute the next undamped step that would result if x1 is accepted
            // J(x_k)^-1 F(x_k+1)
            let undamped_step_k_plus_1 = self.step(y_k_plus_1.clone()); //???????????????
            self.custom_timer.linear_system_tac();
            *self
                .calc_statistics
                .entry("number of solving linear systems".to_string())
                .or_insert(0) += 1;

            let error = &undamped_step_k_plus_1.0.norm();
            self.max_error = *error;
            info!("\n \n L2 norm of undamped step = {}", error);
            let convergence_cond_for_step = self.tolerance;
            // compute the norm of the undamped step
            let S_k_plus_1_temp = &undamped_step_k_plus_1.0.norm();
            // If the norm of S_k_plus_1 is less than the norm of S_k, then accept this
            // damping coefficient. Also accept it if this step would result in a
            // converged solution. Otherwise, decrease the damping coefficient and
            // try again.
            let elapsed = now.elapsed();
            elapsed_time(elapsed);
            if ((S_k_plus_1_temp + self.tolerance) < *S_k)
                || (*S_k_plus_1_temp < convergence_cond_for_step)
            {
                // The  criterion for accepting is that the undamped steps decrease in
                // magnitude, This prevents the iteration from stepping away from the region where there is good reason to believe a solution lies

                k_ = k;
                S_k_plus_1 = Some(*S_k_plus_1_temp);
                damped_step_result = Some(y_k_plus_1.clone());
                conv = convergence_cond_for_step;
                info!(
                    "\n \n  Damping coefficient accepted S_k_plus_1_temp < S_k {}, {},",
                    S_k_plus_1_temp, S_k,
                );
                info!(
                    "norm of undamped step = {} vs convergence criterion = {}",
                    *S_k_plus_1_temp, convergence_cond_for_step
                );
                info!("damped step result = {:?}", &damped_step_result);
                break;
            }
            // if fail this criterion we must reject it and retries the step with a reduced (often halved) damping parameter trying again until
            // criterion is met  or max damping iterations is reached

            lambda = lambda / (2.0f64.powf(k as f64 + DampFacor));
            k_ = k;
            S_k_plus_1 = Some(*S_k_plus_1_temp);

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
}
