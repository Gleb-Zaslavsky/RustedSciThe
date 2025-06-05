use super::NR::{NR, solve_linear_system};
use crate::numerical::BVP_Damp::BVP_utils::elapsed_time;
use log::{info, warn};
use nalgebra::{DMatrix, DVector};
use std::time::Instant;

impl NR {
    pub fn step_chi(&mut self) -> (i32, Option<DVector<f64>>) {
        let now = Instant::now();
        let bounds_vec = self.bounds_vec.clone();
        let method = self.linear_sys_method.clone().unwrap();
        let parameters = self.parameters.clone().unwrap();
        let diag_flag: bool = if parameters.get("diag").unwrap() == &1.0 {
            true
        } else {
            false
        };
        let nu = parameters.get("nu").unwrap();
        let max_lambda = parameters.get("max_lambda").unwrap();
        let mut lambda = self.dumping_factor.clone();
        info!("lambda = {}", lambda);
        if self.i ==0 {
            
        }
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
        let A_lu = A.lu();
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
