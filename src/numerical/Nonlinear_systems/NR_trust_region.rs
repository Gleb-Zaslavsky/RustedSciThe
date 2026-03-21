use super::NR::{NR, solve_linear_system};
use crate::numerical::BVP_Damp::BVP_utils::elapsed_time;

use nalgebra::{DMatrix, DVector};
use log::{error, info};

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



/// Errors that can occur during nonlinear solving
#[derive(Debug, Clone)]
pub enum SolveError {
    /// Solver did not converge within max_iter
    MaxIterations,
    /// Linear system (LM normal equations) could not be solved
    SingularMatrix,
    /// Residual or Jacobian evaluation failed during solve
    EvalError(String),
}

/// Comprehensive error types for chemical equilibrium calculations
#[derive(Debug)]
pub enum ReactionExtentError {

    /// Error originating from the underlying nonlinear solver
    SolveError(SolveError),
    /// SVD failed when computing reaction basis
    SVDError(String),
    /// Dimension mismatch between vectors/matrices
    DimensionMismatch(String),
    /// Duplicate species assigned to multiple phases
    DuplicateSpecies(usize),
    /// A species is not assigned to any phase
    SpeciesNotAssigned,
    /// Initial residual evaluation produced NaN/Inf
    InvalidInitialResiduals(Vec<f64>),
    /// Residual evaluation failed (generic)
    ResidualEvaluation(String),
    /// Jacobian evaluation failed (generic)
    JacobianEvaluation(String),
    /// Invalid species mole numbers (negative or zero)
    InvalidSpeciesAmount { index: usize, value: f64 },
    /// Invalid per-phase totals
    InvalidNPhase { index: usize, value: f64 },
    /// Invalid phi value
    InvalidPhi { index: usize, value: f64 },
    /// Invalid DG0 or temperature leading to bad ln_k
    InvalidDG0 { dg0: f64, temperature: f64 },
    /// Other error
    Other(String),
}
pub struct LMSolver<F, J, C>
where
    F: Fn(&[f64]) -> Result<Vec<f64>, ReactionExtentError>,
    J: Fn(&[f64]) -> Result<DMatrix<f64>, ReactionExtentError>,
    C: Fn(&[f64]) -> bool,
{
    /// Residual function f(x) = 0
    pub f: F,
    /// Jacobian function J(x) = df/dx
    pub jacobian: J,
    /// Feasibility constraint checker
    pub feasible: C,
    /// Damping parameter (increased when steps rejected)
    pub lambda: f64,
    /// Convergence tolerance for ||f(x)||
    pub tol: f64,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Minimum step size before giving up
    pub alpha_min: f64,
}

impl<F, J, C> LMSolver<F, J, C>
where
    F: Fn(&[f64]) -> Result<Vec<f64>, ReactionExtentError>,
    J: Fn(&[f64]) -> Result<DMatrix<f64>, ReactionExtentError>,
    C: Fn(&[f64]) -> bool,
{
    /// Solves nonlinear system f(x) = 0 using Levenberg-Marquardt algorithm
    ///
    /// Uses adaptive damping and line search to ensure robust convergence.
    /// Respects feasibility constraints throughout the solution process.
    pub fn solve(&mut self, mut x: Vec<f64>) -> Result<Vec<f64>, SolveError> {
        let n = x.len();

        let mut lambda = self.lambda;

        for _iter in 0..self.max_iter {
            let f_val = (self.f)(&x)
                .map_err(|e| SolveError::EvalError(format!("Residual eval error: {:?}", e)))?;
            info!("Iteration {}, x = {:?}, f = {:?}", _iter, x, f_val);
            let f_norm = l2_norm(&f_val);
            info!("  ||f|| = {}", f_norm);
            if f_norm < self.tol {
                return Ok(x);
            }
            if f_norm.is_nan() || f_norm.is_infinite() {
                return Err(SolveError::SingularMatrix);
            }

            let j = (self.jacobian)(&x)
                .map_err(|e| SolveError::EvalError(format!("Jacobian eval error: {:?}", e)))?;
            info!("  Jacobian:\n{}", j);
            let jt = j.transpose();

            let jtj = &jt * &j;
            let mut lhs = jtj.clone();
            info!("  Jᵀ·J:\n{}", jtj);
            for i in 0..n {
                lhs[(i, i)] += lambda;
            }

            let rhs = -(&jt * DVector::from_vec(f_val.clone()));

            let delta = lhs.lu().solve(&rhs).ok_or(SolveError::SingularMatrix)?;
            info!("  Step delta: {:?}", delta);

            let delta = delta.data.as_vec().clone();

            let mut alpha = 1.0;
            let mut accepted = false;

            while alpha >= self.alpha_min {
                let x_trial: Vec<f64> = x
                    .iter()
                    .zip(delta.iter())
                    .map(|(xi, dxi)| xi + alpha * dxi)
                    .collect();
                info!("    Trial x (alpha={}): {:?}", alpha, x_trial);
                if !(self.feasible)(&x_trial) {
                    alpha *= 0.5;
                    continue;
                }

                let f_trial = (self.f)(&x_trial)
                    .map_err(|e| SolveError::EvalError(format!("Residual eval error: {:?}", e)))?;
                info!("    Trial f: {:?}", f_trial);
                let f_trial_norm = l2_norm(&f_trial);
                info!("    ||f_trial|| = {}", f_trial_norm);
                if f_trial_norm < f_norm {
                    x = x_trial;
                    lambda *= 0.3;
                    accepted = true;
                    break;
                } else {
                    alpha *= 0.5;
                }
            }

            if !accepted {
                lambda *= 10.0;
                info!(
                    "  No acceptable step found; increasing lambda to {}",
                    lambda
                );
            }
        }

        Err(SolveError::MaxIterations)
    }
}
//////////////////////////////////////NR SOLVER//////////////////////////////////////////////////////////////

/*
pub fn tolerance_calc(f_val: &Vec<f64>, x: &Vec<f64>){
    let mut complex = 0.0;
    let x_sum: f64 = x.iter().sum();
    for (i, f_val_i) in f_val.iter().enumerate(){
        let rel
       let s =  f_val_i/x[i].abs()
    }
}
*/

/// Newton-Raphson solver with line search and feasibility constraints
///
/// Fast quadratic convergence near solution, with backtracking line search
/// and bound constraints to handle chemical equilibrium problems.
pub struct NRSolver<F, J, C>
where
    F: Fn(&[f64]) -> Result<Vec<f64>, ReactionExtentError>,
    J: Fn(&[f64]) -> Result<DMatrix<f64>, ReactionExtentError>,
    C: Fn(&[f64]) -> bool,
{
    /// Residual function f(x) = 0
    pub f: F,
    /// Jacobian function J(x) = df/dx
    pub jacobian: J,
    /// Feasibility constraint checker
    pub feasible: C,
    /// Initial mole numbers (for bound checking)
    pub n0: Vec<f64>,
    /// Reaction stoichiometry matrix
    pub reactions: DMatrix<f64>,
    /// Convergence tolerance for ||f(x)||
    pub tol: f64,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Minimum step size before giving up
    pub alpha_min: f64,
}

impl<F, J, C> NRSolver<F, J, C>
where
    F: Fn(&[f64]) -> Result<Vec<f64>, ReactionExtentError>,
    J: Fn(&[f64]) -> Result<DMatrix<f64>, ReactionExtentError>,
    C: Fn(&[f64]) -> bool,
{
    /// Solves nonlinear system using Newton-Raphson with line search
    ///
    /// Performs backtracking line search with feasibility checking.
    /// Uses bound-aware step limiting to prevent negative mole numbers.
    pub fn solve(&mut self, mut x: Vec<f64>) -> Result<Vec<f64>, SolveError> {
        let n = x.len();

        for iter in 0..self.max_iter {
            let f_val = (self.f)(&x).map_err(|e| SolveError::EvalError(format!("{:?}", e)))?;
            info!("NR Iteration {}, x = {:?}, f = {:?}", iter, x, f_val);
            let f_norm = l2_norm(&f_val);
            info!(" ||f|| = {}", f_norm);
            if f_norm < self.tol {
                return Ok(x);
            }
            if f_norm.is_nan() || f_norm.is_infinite() {
                return Err(SolveError::SingularMatrix);
            }
            let j = (self.jacobian)(&x).map_err(|e| SolveError::EvalError(format!("{:?}", e)))?;

            info!(" Jacobian:\n{}", j);
            if j.nrows() != n || j.ncols() != n {
                error!(
                    "Jacobian nrows = {} ncols ={}, initial guess length {}",
                    j.nrows(),
                    j.ncols(),
                    n
                );
                return Err(SolveError::EvalError(
                    "Jacobian dimension mismatch".to_string(),
                ));
            }
            // Solve J * delta = -f
            let rhs = -DVector::from_vec(f_val);
            let delta_vec = j.lu().solve(&rhs).ok_or(SolveError::SingularMatrix)?;
            let delta = delta_vec.data.as_vec();
            info!(" Step delta: {:?}", delta_vec);
            // --- NEW: bound-aware step size ---
            // let alpha_species =  max_step_moles_nonnegative(&x, &delta, 0.95);

            // let mut alpha = alpha_species;
            let mut alpha = 1.0;
            info!("bounded step {}", alpha);

            let mut accepted = false;
            while alpha >= self.alpha_min {
                let x_trial: Vec<f64> = x
                    .iter()
                    .zip(delta.iter())
                    .map(|(xi, dxi)| xi + alpha * dxi)
                    .collect();
                info!(" Trial x (alpha={}): {:?}", alpha, x_trial);
                if !(self.feasible)(&x_trial) {
                    alpha *= 0.5;
                    continue;
                }

                let f_trial =
                    (self.f)(&x_trial).map_err(|e| SolveError::EvalError(format!("{:?}", e)))?;

                info!(" Trial f: {:?}", f_trial);

                let f_trial_norm = l2_norm(&f_trial);
                info!(" ||f_trial|| = {}", f_trial_norm);
                if f_trial_norm < f_norm {
                    x = x_trial;
                    accepted = true;
                    break;
                }

                alpha *= 0.5;
                if !accepted {
                    info!(" No acceptable step found; continuing to next iteration");
                }
            }

            if !accepted {
                return Err(SolveError::MaxIterations);
            }
        }

        Err(SolveError::MaxIterations)
    }
}

/// Computes L2 norm of a vector
///
/// Helper function for convergence checking in solvers.
fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}
///////////////////////TRUST REGION///////////////////////////////////////////////////

pub struct TrustRegionSolver<F, J, C>
where
    F: Fn(&[f64]) -> Result<Vec<f64>, ReactionExtentError>,
    J: Fn(&[f64]) -> Result<DMatrix<f64>, ReactionExtentError>,
    C: Fn(&[f64]) -> bool,
{
    pub f: F,
    pub jacobian: J,
    pub feasible: C,

    pub tol: f64,
    pub max_iter: usize,

    // Trust region parameters
    pub delta_init: f64,
    pub delta_max: f64,
    pub eta: f64, // acceptance threshold (e.g. 0.1)
}

impl<F, J, C> TrustRegionSolver<F, J, C>
where
    F: Fn(&[f64]) -> Result<Vec<f64>, ReactionExtentError>,
    J: Fn(&[f64]) -> Result<DMatrix<f64>, ReactionExtentError>,
    C: Fn(&[f64]) -> bool,
{
    pub fn solve(&self, mut x: Vec<f64>) -> Result<Vec<f64>, SolveError> {
        let n = x.len();
        let mut delta = self.delta_init;

        for iter in 0..self.max_iter {
            let f_val = (self.f)(&x).map_err(|e| SolveError::EvalError(format!("{:?}", e)))?;

            let f_norm = l2_norm(&f_val);
            if f_norm < self.tol {
                return Ok(x);
            }

            let J = (self.jacobian)(&x).map_err(|e| SolveError::EvalError(format!("{:?}", e)))?;

            if J.nrows() != n || J.ncols() != n {
                return Err(SolveError::EvalError("Jacobian not square".into()));
            }

            let fvec = DVector::from_vec(f_val.clone());

            // --- Newton step ---
            let newton_step = J
                .clone()
                .lu()
                .solve(&(-&fvec))
                .unwrap_or_else(|| DVector::zeros(n));

            // --- Cauchy step ---
            let g = &J.transpose() * &fvec;
            let g_norm_sq = g.dot(&g);
            let jg = &J * &g;
            let alpha = g_norm_sq / jg.dot(&jg).max(1e-16);
            let cauchy_step = -alpha * g;

            // --- Dogleg step ---
            let p = if newton_step.norm() <= delta {
                newton_step
            } else if cauchy_step.norm() >= delta {
                delta / cauchy_step.norm() * cauchy_step
            } else {
                let p_u = cauchy_step;
                let p_b = newton_step;
                let d = &p_b - &p_u;

                let a = d.dot(&d);
                let b = 2.0 * p_u.dot(&d);
                let c = p_u.dot(&p_u) - delta * delta;

                let tau = (-b + (b * b - 4.0 * a * c).sqrt()) / (2.0 * a);
                p_u + tau * d
            };

            let p_vec = p.data.as_vec().clone();
            let x_trial: Vec<f64> = x.iter().zip(p_vec.iter()).map(|(xi, pi)| xi + pi).collect();

            if !(self.feasible)(&x_trial) {
                delta *= 0.25;
                continue;
            }

            let f_trial =
                (self.f)(&x_trial).map_err(|e| SolveError::EvalError(format!("{:?}", e)))?;

            let actual_reduction = f_norm.powi(2) - l2_norm(&f_trial).powi(2);

            let model_reduction = -2.0 * fvec.dot(&p) - p.dot(&(J.transpose() * &J * &p));

            let rho = actual_reduction / model_reduction.max(1e-16);

            // --- Trust region update ---
            if rho < 0.25 {
                delta *= 0.25;
            } else if rho > 0.75 && (p.norm() - delta).abs() < 1e-12 {
                delta = (2.0 * delta).min(self.delta_max);
            }

            // --- Accept / reject ---
            if rho > self.eta {
                x = x_trial;
            }

            if delta < 1e-14 {
                return Err(SolveError::SingularMatrix);
            }
        }

        Err(SolveError::MaxIterations)
    }
}


///////////////////////////////TESTS////////////////////////////////////
#[cfg(test)]
mod tests {
    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }
    use super::*;
    #[test]
    fn lm_solves_scalar_quadratic() {
        let f = |x: &[f64]| Ok(vec![x[0] * x[0] - 2.0]) as Result<Vec<f64>, ReactionExtentError>;

        let j = |x: &[f64]| {
            Ok(nalgebra::DMatrix::from_row_slice(1, 1, &[2.0 * x[0]]))
                as Result<DMatrix<f64>, ReactionExtentError>
        };

        let feasible = |_x: &[f64]| true;

        let mut solver = LMSolver {
            f,
            jacobian: j,
            feasible,
            lambda: 1e-3,
            tol: 1e-12,
            max_iter: 50,
            alpha_min: 1e-6,
        };

        let x0 = vec![1.0];
        let sol = solver.solve(x0).unwrap();

        assert!(approx_eq(sol[0], 2.0_f64.sqrt(), 1e-8));
    }
    #[test]
    fn lm_solves_2d_nonlinear_system() {
        let f = |x: &[f64]| {
            Ok(vec![x[0] * x[0] + x[1] * x[1] - 1.0, x[0] - x[1]])
                as Result<Vec<f64>, ReactionExtentError>
        };

        let j = |x: &[f64]| {
            Ok(nalgebra::DMatrix::from_row_slice(
                2,
                2,
                &[2.0 * x[0], 2.0 * x[1], 1.0, -1.0],
            )) as Result<DMatrix<f64>, ReactionExtentError>
        };

        let feasible = |_x: &[f64]| true;

        let mut solver = LMSolver {
            f,
            jacobian: j,
            feasible,
            lambda: 1e-3,
            tol: 1e-12,
            max_iter: 50,
            alpha_min: 1e-6,
        };

        let x0 = vec![0.8, 0.3];
        let sol = solver.solve(x0).unwrap();

        let expected = 1.0 / 2.0_f64.sqrt();
        assert!(approx_eq(sol[0], expected, 1e-8));
        assert!(approx_eq(sol[1], expected, 1e-8));
    }

    #[test]
    fn lm_respects_feasibility_constraint() {
        let f = |x: &[f64]| Ok(vec![x[0] * x[0] - 1.0]) as Result<Vec<f64>, ReactionExtentError>;

        let j = |x: &[f64]| {
            Ok(nalgebra::DMatrix::from_row_slice(1, 1, &[2.0 * x[0]]))
                as Result<DMatrix<f64>, ReactionExtentError>
        };

        let feasible = |x: &[f64]| x[0] >= 0.0;

        let mut solver = LMSolver {
            f,
            jacobian: j,
            feasible,
            lambda: 1e-3,
            tol: 1e-12,
            max_iter: 50,
            alpha_min: 1e-6,
        };

        let x0 = vec![0.1];
        let sol = solver.solve(x0).unwrap();

        assert!(approx_eq(sol[0], 1.0, 1e-8));
    }

    #[test]
    fn lm_handles_flat_jacobian() {
        let f = |x: &[f64]| Ok(vec![x[0].powi(3)]) as Result<Vec<f64>, ReactionExtentError>;

        let j = |x: &[f64]| {
            Ok(nalgebra::DMatrix::from_row_slice(
                1,
                1,
                &[3.0 * x[0] * x[0]],
            )) as Result<DMatrix<f64>, ReactionExtentError>
        };

        let feasible = |_x: &[f64]| true;

        let mut solver = LMSolver {
            f,
            jacobian: j,
            feasible,
            lambda: 1e-4,
            tol: 1e-15,
            max_iter: 1000,
            alpha_min: 1e-8,
        };

        let x0 = vec![0.5];
        let sol = solver.solve(x0).unwrap();
        info!("{:?}", &sol);
        assert!(sol[0].abs() < 1e-5);
    }
    #[test]
    fn lm_reports_max_iterations() {
        let f = |_x: &[f64]| Ok(vec![1.0]); // no root
        let j = |_x: &[f64]| Ok(nalgebra::DMatrix::identity(1, 1));
        let feasible = |_x: &[f64]| true;

        let mut solver = LMSolver {
            f,
            jacobian: j,
            feasible,
            lambda: 1e-3,
            tol: 1e-12,
            max_iter: 5,
            alpha_min: 1e-6,
        };

        let res = solver.solve(vec![0.0]);
        assert!(matches!(res, Err(SolveError::MaxIterations)));
    }
}


#[cfg(test)]
mod nr_tests {
    use super::*;
    use nalgebra::DMatrix;



    #[test]
    fn nr_solver_respects_bounds_and_converges() {
        // Solve simple scalar problem f(x) = x (root at 0). Start at small positive x0.
        let f = |x: &[f64]| Ok(vec![x[0]]) as Result<Vec<f64>, ReactionExtentError>;
        let j = |_: &[f64]| {
            Ok(DMatrix::from_row_slice(1, 1, &[1.0])) as Result<DMatrix<f64>, ReactionExtentError>
        };

        let feasible = |x: &[f64]| x[0] >= 0.0;

        let mut solver = NRSolver {
            f,
            jacobian: j,
            feasible,
            n0: vec![0.01],
            reactions: DMatrix::zeros(1, 0),
            tol: 1e-12,
            max_iter: 100,
            alpha_min: 1e-12,
        };

        let sol = solver.solve(vec![0.01]).unwrap();

        // Solution should remain non-negative and be close to zero
        assert!(sol[0] >= 0.0);
        assert!(sol[0].abs() < 1e-8);
    }
}

#[cfg(test)]
mod trust_region_tests {
    use super::*;
    use approx::assert_relative_eq;
    fn vec_norm(v: &[f64]) -> f64 {
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    fn approx_eq(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.iter().zip(b.iter()).all(|(x, y)| (*x - *y).abs() < tol)
    }

    #[test]
    fn trust_region_linear_system() {
        let f = |x: &[f64]| -> Result<Vec<f64>, ReactionExtentError> {
            Ok(vec![3.0 * x[0] + 1.0 * x[1] - 1.0, 1.0 * x[0] + 2.0 * x[1]])
        };

        let j = |_x: &[f64]| -> Result<DMatrix<f64>, ReactionExtentError> {
            Ok(DMatrix::from_row_slice(2, 2, &[3.0, 1.0, 1.0, 2.0]))
        };

        let feasible = |_x: &[f64]| true;

        let solver = TrustRegionSolver {
            f,
            jacobian: j,
            feasible,
            tol: 1e-12,
            max_iter: 10,
            delta_init: 1.0,
            delta_max: 10.0,
            eta: 0.1,
        };

        let x0 = vec![0.0, 0.0];
        let sol = solver.solve(x0).unwrap();

        assert!(approx_eq(&sol, &[0.4, -0.2], 1e-10));
    }

    #[test]
    fn trust_region_scalar_nonlinear() {
        let f =
            |x: &[f64]| -> Result<Vec<f64>, ReactionExtentError> { Ok(vec![x[0] * x[0] - 2.0]) };

        let j = |x: &[f64]| -> Result<DMatrix<f64>, ReactionExtentError> {
            Ok(DMatrix::from_row_slice(1, 1, &[2.0 * x[0]]))
        };

        let feasible = |_x: &[f64]| true;

        let solver = TrustRegionSolver {
            f,
            jacobian: j,
            feasible,
            tol: 1e-12,
            max_iter: 50,
            delta_init: 0.1, // intentionally small
            delta_max: 10.0,
            eta: 0.1,
        };

        let sol = solver.solve(vec![0.1]).unwrap();
        assert!((sol[0] - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn trust_region_ill_conditioned() {
        let f = |x: &[f64]| -> Result<Vec<f64>, ReactionExtentError> {
            Ok(vec![1e6 * x[0] - 1.0, x[1] - 1.0])
        };

        let j = |_x: &[f64]| -> Result<DMatrix<f64>, ReactionExtentError> {
            Ok(DMatrix::from_row_slice(2, 2, &[1e6, 0.0, 0.0, 1.0]))
        };

        let feasible = |_x: &[f64]| true;

        let solver = TrustRegionSolver {
            f,
            jacobian: j,
            feasible,
            tol: 1e-12,
            max_iter: 50,
            delta_init: 0.01,
            delta_max: 1.0,
            eta: 0.1,
        };

        let sol = solver.solve(vec![0.0, 0.0]).unwrap();

        assert!((sol[0] - 1e-6).abs() < 1e-10);
        assert!((sol[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn trust_region_feasibility_constraint() {
        let f = |x: &[f64]| -> Result<Vec<f64>, ReactionExtentError> { Ok(vec![x[0] - 1.0]) };

        let j = |_x: &[f64]| -> Result<DMatrix<f64>, ReactionExtentError> {
            Ok(DMatrix::from_row_slice(1, 1, &[1.0]))
        };

        let feasible = |x: &[f64]| x[0] >= 0.0;

        let solver = TrustRegionSolver {
            f,
            jacobian: j,
            feasible,
            tol: 1e-12,
            max_iter: 50,
            delta_init: 10.0,
            delta_max: 10.0,
            eta: 0.1,
        };

        let sol = solver.solve(vec![-10.0]).unwrap();
        assert!((sol[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn trust_region_singular_jacobian() {
        let f = |_x: &[f64]| -> Result<Vec<f64>, ReactionExtentError> { Ok(vec![1.0]) };

        let j =
            |_x: &[f64]| -> Result<DMatrix<f64>, ReactionExtentError> { Ok(DMatrix::zeros(1, 1)) };

        let feasible = |_x: &[f64]| true;

        let solver = TrustRegionSolver {
            f,
            jacobian: j,
            feasible,
            tol: 1e-12,
            max_iter: 5,
            delta_init: 1.0,
            delta_max: 1.0,
            eta: 0.1,
        };

        let res = solver.solve(vec![0.0]);
        assert!(res.is_err());
    }
    ///////////////////////
    #[test]
    fn tr_solves_scalar_quadratic() {
        let f = |x: &[f64]| Ok(vec![x[0] * x[0] - 2.0]) as Result<Vec<f64>, ReactionExtentError>;

        let j = |x: &[f64]| {
            Ok(nalgebra::DMatrix::from_row_slice(1, 1, &[2.0 * x[0]]))
                as Result<DMatrix<f64>, ReactionExtentError>
        };

        let feasible = |_x: &[f64]| true;

        let solver = TrustRegionSolver {
            f,
            jacobian: j,
            feasible,
            tol: 1e-12,
            max_iter: 50,
            delta_init: 0.1,
            delta_max: 10.0,
            eta: 0.0,
        };

        let x0 = vec![0.1];
        let sol = solver.solve(x0).unwrap();

        assert_relative_eq!(sol[0], 2.0_f64.sqrt(), epsilon = 1e-8);
    }

    #[test]
    fn tr_solves_2d_nonlinear_system() {
        let f = |x: &[f64]| {
            Ok(vec![x[0] * x[0] + x[1] * x[1] - 1.0, x[0] - x[1]])
                as Result<Vec<f64>, ReactionExtentError>
        };

        let j = |x: &[f64]| {
            Ok(nalgebra::DMatrix::from_row_slice(
                2,
                2,
                &[2.0 * x[0], 2.0 * x[1], 1.0, -1.0],
            )) as Result<DMatrix<f64>, ReactionExtentError>
        };

        let feasible = |_x: &[f64]| true;

        let solver = TrustRegionSolver {
            f,
            jacobian: j,
            feasible,
            tol: 1e-12,
            max_iter: 1000,
            delta_init: 0.1,
            delta_max: 10.0,
            eta: 0.0,
        };

        let x0 = vec![0.5, 0.5];
        let sol = solver.solve(x0).unwrap();

        let expected = 1.0 / 2.0_f64.sqrt();
        assert_relative_eq!(sol[0], expected, epsilon = 1e-8);
        assert_relative_eq!(sol[1], expected, epsilon = 1e-8);
    }

    #[test]
    fn tr_respects_feasibility_constraint() {
        let f = |x: &[f64]| Ok(vec![x[0] * x[0] - 1.0]) as Result<Vec<f64>, ReactionExtentError>;

        let j = |x: &[f64]| {
            Ok(nalgebra::DMatrix::from_row_slice(1, 1, &[2.0 * x[0]]))
                as Result<DMatrix<f64>, ReactionExtentError>
        };

        let feasible = |x: &[f64]| true;

        let solver = TrustRegionSolver {
            f,
            jacobian: j,
            feasible,
            tol: 1e-12,
            max_iter: 50,
            delta_init: 1.0,
            delta_max: 10.0,
            eta: 0.1,
        };

        let x0 = vec![0.1];
        let sol = solver.solve(x0).unwrap();

        assert_relative_eq!(sol[0], 1.0, epsilon = 1e-8);
    }

    #[test]
    fn tr_reports_max_iterations() {
        let f = |_x: &[f64]| Ok(vec![1.0]); // no root
        let j = |_x: &[f64]| Ok(nalgebra::DMatrix::identity(1, 1));
        let feasible = |_x: &[f64]| true;

        let solver = TrustRegionSolver {
            f,
            jacobian: j,
            feasible,
            tol: 1e-12,
            max_iter: 5,
            delta_init: 1.0,
            delta_max: 10.0,
            eta: 0.1,
        };

        let res = solver.solve(vec![0.0]);
        assert!(matches!(res, Err(SolveError::MaxIterations)));
    }
}
