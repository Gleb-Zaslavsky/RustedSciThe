use super::NR::{NR, solve_linear_system};
use crate::numerical::BVP_Damp::BVP_utils::elapsed_time;
use log::{info, warn};
use nalgebra::{DMatrix, DVector};
use std::time::Instant;
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

impl NR {
    pub fn step_lm(&mut self) -> (i32, Option<DVector<f64>>) {
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
