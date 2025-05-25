use super::NR::NR;
use crate::numerical::BVP_Damp::BVP_utils::elapsed_time;
use log::info;
use nalgebra::{DMatrix, DVector};
use std::time::Instant;
///
/// Trust region method for solving nonlinear systems of equations.
/// https://en.wikipedia.org/wiki/Trust_region
///

impl NR {
    pub fn step_trust_region(&mut self) -> (i32, Option<DVector<f64>>) {
        let now = Instant::now();
        let bounds_vec = self.bounds_vec.clone();
        let parameters = self.parameters.clone().unwrap();
        let eta = parameters.get("eta").unwrap();

        let mut parameters_mut = self.parameters.clone().unwrap();
        let delta = parameters_mut.get_mut("delta").unwrap();
        info!("delta = {}", delta);

        let y = self.y.clone();
        let Fy = self.evaluate_function(y.clone());
        let Jy = self.evaluate_jacobian(y.clone());
        let norm = Fy.norm_squared();
        info!("norm = {}", norm);
        if norm < self.tolerance {
            info!("Solution found!");
            return (1, Some(y));
        }
        let delta_ = delta.clone();
        let p = self.solve_trust_region_subproblem(&Jy, &Fy, delta_.clone());
        let mut y_new = y.clone() + &p;
   
        // Project onto bounds (if they exist)
        y_new = self.clip(&y_new, &bounds_vec);

        let Fy_new = self.evaluate_function(y_new.clone());
        let actual_reduction = Fy.norm_squared() - Fy_new.norm_squared();

        let predicted_reduction = Fy.norm_squared() - (Fy.clone() + Jy * &p).norm_squared();
        let rho = if predicted_reduction.abs() > 1e-12 {
            actual_reduction / predicted_reduction
        } else {
            0.0
        };
        info!("rho is: {}", rho);
        let elapsed = now.elapsed();
        elapsed_time(elapsed);
        if rho > *eta {
            let y_result = y_new; // Accept step
            return (0, Some(y_result));
        }
        if rho <= 0.25 {
            *delta *= 0.5;
            info!("delta changed to {}", delta);
            self.parameters = Some(parameters_mut);
            return (0, Some(y));
        } else if rho > 0.75 && (p.norm() - *delta).abs() < 1e-6 {
            *delta = (2.0 * *delta).min(1e6);
            info!("delta changed to {}", delta);
            self.parameters = Some(parameters_mut);
            return (0, Some(y));
        } else {
            println!("Trust region step failed.");
            // unreachable!();
            return (-2, None);
        }
    }

    /// Solve min ||Jp + F||² s.t. ||p|| ≤ Δ (dogleg method)
    fn solve_trust_region_subproblem(
        &self,
        J: &DMatrix<f64>,
        F: &DVector<f64>,
        delta: f64,
    ) -> DVector<f64> {
        // Compute Newton step (unconstrained)
        let p_newton = -J.clone().svd(true, true).solve(F, 0.0).unwrap();

        if p_newton.norm() <= delta {
            return p_newton; // Newton step is inside trust region
        }

        // Compute Cauchy point (steepest descent)
        let JtF = J.transpose() * F;
        let p_cauchy = -JtF.dot(F) / JtF.norm_squared() * JtF;

        if p_cauchy.norm() >= delta {
            return delta * p_cauchy.normalize(); // Return scaled steepest descent
        }

        // Find intersection with trust-region boundary (dogleg path)
        let p_diff = p_newton - p_cauchy.clone();
        let a = p_diff.norm_squared();
        let b = 2.0 * p_cauchy.dot(&p_diff);
        let c = p_cauchy.norm_squared() - delta.powi(2);
        let tau = (-b + (b.powi(2) - 4.0 * a * c).sqrt()) / (2.0 * a);

        p_cauchy + tau * p_diff
    }
}
