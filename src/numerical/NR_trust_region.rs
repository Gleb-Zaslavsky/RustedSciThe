use super::NR::{NR, solve_linear_system};
use crate::numerical::BVP_Damp::BVP_utils::elapsed_time;
use log::info;
use nalgebra::{DMatrix, DVector};

use std::time::Instant;
/// DONT USE IN PRODUCTION CODE
/// MODULE IS UNDER CONSTRUCTION

impl NR {
    pub fn step_trust_region(&mut self) -> (i32, Option<DVector<f64>>) {
        let now = Instant::now();
        let bounds_vec = self.bounds_vec.clone();
        let method = self.linear_sys_method.clone().unwrap();
        let parameters = self.parameters.clone().unwrap();
        let eta_min = parameters.get("eta_min").unwrap();
        let eta_max = parameters.get("eta_max").unwrap();
        let ro_threshold0 = parameters.get("ro_threshold0").unwrap();
        let ro_threshold1 = parameters.get("ro_threshold1").unwrap();
        let C0 = parameters.get("C0").unwrap();
        let M = parameters.get("M").unwrap();
        let d = parameters.get("d").unwrap();
        let m = parameters.get("m").unwrap();
        let mut parameters_mut = self.parameters.clone().unwrap();
        //  let delta = parameters_mut.get_mut("delta").unwrap();
        let mu = parameters_mut.get_mut("mu").unwrap();

        let y = self.y.clone();
        let Fy = self.evaluate_function(y.clone());
        let Jy = self.evaluate_jacobian(y.clone());

        // Check convergence
        let old_norm = Fy.norm_squared();
        let delta = mu.clone() * old_norm.powf(*d);
        info!("delta = {}", delta);
        info!("norm = {}", old_norm);
        if old_norm < self.tolerance {
            let y_result = y.clone();
            info!("Solution found!");
            return (1, Some(y_result));
        }

        let b = -Jy.transpose() * &Fy;
        let step = self
            .solve_trust_region_subproblem(method, &Jy, &b, delta, *m)
            .unwrap();

        let mut y_new = y.clone() + &step;

        // Project onto bounds (if they exist)
        y_new = self.clip(&y_new, &bounds_vec);

        let Fy_new = self.evaluate_function(y_new.clone());
        // let rho = (old_norm - Fy_new.norm_squared())/old_norm;

        let actual_reduction = Fy.norm_squared() - Fy_new.norm_squared();
        let predicted_reduction = Fy.norm_squared() - (Fy.clone() + Jy * &step).norm_squared();
        let rho = actual_reduction / (predicted_reduction + 1e-12);

        info!("rho is: {}", rho);
        let elapsed = now.elapsed();
        elapsed_time(elapsed);
        let y_result = if rho > *C0 { y_new.clone() } else { y.clone() };

        if rho < *ro_threshold0 {
            *mu = *eta_min * *mu;
            info!("mu decreased: {}", mu);
            self.parameters = Some(parameters_mut);
            return (0, Some(y_result));
        } else if (rho >= *ro_threshold0) && (rho <= *ro_threshold1) {
            *mu = *mu;
            self.parameters = Some(parameters_mut);
            return (0, Some(y_result));
        } else {
            *mu = M.min(*eta_max * *mu);
            self.parameters = Some(parameters_mut);
            return (0, Some(y_result));
        }
    }

    fn solve_trust_region_subproblem_dogleg(
        &self,
        J: &DMatrix<f64>,
        F: &DVector<f64>,
        B: &DMatrix<f64>,
        H: &DMatrix<f64>,
        delta: f64,
    ) -> DVector<f64> {
        let pU: DVector<f64> = DVector::zeros(0);
        pU
        /*

        // Compute the Newton point.
        // This is the optimum for the quadratic model function.
        //If it is inside the trust radius then return this point.
        let pB: DVector<f64> = - H.cross(J);
        let norm_pB = pB.abs();

        // Test if the full step is within the trust region.
        if norm_pB <= delta:
            return pB

        //Compute the Cauchy point.
        // This is the predicted optimum along the direction of steepest descent.
        pU = -  J.dot(J) /  J.dot( );
        dot_pU = np.dot(pU, pU)
        norm_pU = sqrt(dot_pU)

        # If the Cauchy point is outside the trust region,
        # then return the point where the path intersects the boundary.
        if norm_pU >= trust_radius:
            return trust_radius * pU / norm_pU

        # Find the solution to the scalar quadratic equation.
        # Compute the intersection of the trust region boundary
        # and the line segment connecting the Cauchy and Newton points.
        # This requires solving a quadratic equation.
        # ||p_u + tau*(p_b - p_u)||**2 == trust_radius**2
        # Solve this for positive time t using the quadratic formula.
        pB_pU = pB - pU
        dot_pB_pU = np.dot(pB_pU, pB_pU)
        dot_pU_pB_pU = np.dot(pU, pB_pU)
        fact = dot_pU_pB_pU**2 - dot_pB_pU * (dot_pU - trust_radius**2)
        tau = (-dot_pU_pB_pU + sqrt(fact)) / dot_pB_pU

        # Decide on which part of the trajectory to take.
        return pU + tau * pB_pU
        */
    }
    /*
    def dogleg(H, g, B, trust_radius):
        pb = -H@g                    # full newton step
        norm_pb = np.linalg.norm(pb)

        # full newton step lies inside the trust region
        if np.linalg.norm(pb) <= trust_radius:
            return pb
        # step along the steepest descent direction lies outside the
        # trust region       pu = - (np.dot(g, g) / np.dot(g, B@g)) * g
        dot_pu = np.dot(pu, pu)
        norm_pu = np.sqrt(dot_pu)
        if norm_pu >= trust_radius:
            return trust_radius * pu / norm_pu
        # solve ||pu**2 +(tau-1)*(pb-pu)**2|| = trust_radius**2
        pb_pu = pb - pu
        pb_pu_sq = np.dot(pb_pu, pb_pu)
        pu_pb_pu_sq = np.dot(pu, pb_pu)
        d = pu_pb_pu_sq ** 2 - pb_pu_sq * (dot_pu - trust_radius ** 2)
        tau = (-pu_pb_pu_sq + np.sqrt(d)) / pb_pu_sq+1    # 0<tau<1
        if tau < 1:
            return pu*tau
        # 1<tau<2
        return pu + (tau-1) * pb_pu
    */

    /// Solve min ||Jp + F||² s.t. ||p|| ≤ Δ (dogleg method)
    fn solve_trust_region_subproblem(
        &self,
        method: String,
        Jy: &DMatrix<f64>,
        Ft: &DVector<f64>,
        delta: f64,
        m: f64,
    ) -> Result<DVector<f64>, String> {
        let B = Jy.clone().transpose() * Jy.clone();
        /*
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
        */
        Ok(DVector::zeros(Jy.nrows()))
    }
}

/*
   pub fn step_trust_region(&mut self) -> (i32, Option<DVector<f64>>) {
       let now = Instant::now();
       let method = self.linear_sys_method.clone().unwrap();
       let bounds_vec = self.bounds_vec.clone();
       let parameters = self.parameters.clone().unwrap();
       // Parameters
       let eta_1 = parameters.get("eta_1").unwrap();
       let eta_2 = parameters.get("eta_2").unwrap();
       let gamma_1 = parameters.get("gamma_1").unwrap();
       let gamma_2 = parameters.get("gamma_2").unwrap();
       let delta_max = 1e6;
       let mut parameters_mut = self.parameters.clone().unwrap();
       let delta = parameters_mut.get_mut("delta").unwrap();
       info!("delta = {}", delta);

       let y = self.y.clone();
       let Fy = self.evaluate_function(y.clone());
       let Jy = self.evaluate_jacobian(y.clone());
       let B: DMatrix<f64>  = Jy.transpose() * &Jy;
       // Check convergence
       let old_norm = Fy.norm_squared();
       info!("norm = {}", old_norm);
       if old_norm < self.tolerance {
           let y_result = y.clone();
           info!("Solution found!");
           return (1, Some(y_result));
       }
       let step: DVector<f64>  = solve_subproblem(method, &Jy, &Fy, &B, &delta).unwrap();
       let mut y_new = y.clone() + &step;
       // Project onto bounds (if they exist)
       y_new = self.clip(&y_new, &bounds_vec);
       let m_step: DVector<f64> = Fy + Jy * &step + 0.5 * &step *  B*&step;
       let Fy_new = self.evaluate_function(y_new.clone());
       let actual_reduction = Fy.norm_squared() - Fy_new.norm_squared();
       let predicted_reduction = Fy.norm_squared() -  m_step.norm_squared();
       let rho = if predicted_reduction.abs() > 1e-12 {
           actual_reduction / predicted_reduction
       } else {
           0.0
       };
       info!("rho is: {}", rho);
       let elapsed = now.elapsed();
       elapsed_time(elapsed);
       // renew trust region
       if rho >= eta_2.clone() {
           *delta = *delta * gamma_2;
           info!("delta changed to {}", delta);
           self.parameters = Some(parameters_mut);
           let y_result = y_new + &step; // Accept step
           return (0, Some(y_result));
       } else if rho < eta_2.clone() && rho >= eta_1.clone() {
            *delta = *delta;
            info!("delta unchanged");
            self.parameters = Some(parameters_mut);
            let y_result = y_new + &step; // Accept step
            return (0, Some(y_result));
       } else if rho < eta_1.clone() {
           *delta = *delta * gamma_1;
           info!("delta changed to {}", delta);
           self.parameters = Some(parameters_mut);
           let y_result = y_new; // Accept step
           return (0, Some(y_result));

       } else {
           unreachable!();
       }

   }
*/
