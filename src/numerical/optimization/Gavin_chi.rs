use nalgebra::{DMatrix, DVector};
use std::f64;
// Code adapted from Gavin, H.P. (2020) The Levenberg-Marquardt method for
// nonlinear least squares curve-fitting problems.
static mut FUNC_CALLS: usize = 0;
static mut ITERATION: usize = 0;

pub trait LeastSquaresProblemGavin {
    fn lm_func(&self, t: &DVector<f64>, p: &DVector<f64>) -> DVector<f64>;
}

fn lm_func(t: &DVector<f64>, p: &DVector<f64>) -> DVector<f64> {
    t.map(|ti| p[0] * (-ti / p[1]).exp() + p[2] * (ti / p[3]).sin())
}

fn lm_fd_j(
    t: &DVector<f64>,
    p: &DVector<f64>,
    y: &DVector<f64>,
    dp: &DVector<f64>,
) -> DMatrix<f64> {
    unsafe {
        let m = y.len();
        let n = p.len();

        let mut ps = p.clone();
        let mut j = DMatrix::zeros(m, n);
        let mut del = DVector::zeros(n);
        // START --- loop over all parameters
        for idx in 0..n {
            // parameter perturbation
            del[idx] = dp[idx] * (1.0 + p[idx].abs());
            // perturb parameter p(j)
            ps[idx] = p[idx] + del[idx];

            if del[idx] != 0.0 {
                let y1 = lm_func(t, &ps);
                FUNC_CALLS += 1;

                if dp[idx] < 0.0 {
                    for i in 0..m {
                        // backwards difference
                        j[(i, idx)] = (y1[i] - y[i]) / del[idx];
                    }
                } else {
                    // central difference, additional func call
                    ps[idx] = p[idx] - del[idx];
                    let y2 = lm_func(t, &ps);
                    FUNC_CALLS += 1;
                    for i in 0..m {
                        j[(i, idx)] = (y1[i] - y2[i]) / (2.0 * del[idx]);
                    }
                }
            }
            //restore p(j)
            ps[idx] = p[idx];
        }

        j
    }
}
/// Broyden's method
///    Carry out a rank-1 update to the Jacobian matrix using Broyden's equation.
///
///    Parameters
///    ----------
///    p_old :     previous set of parameters (n x 1)
///    y_old :     model evaluation at previous set of parameters, y_hat(t,p_old) (m x 1)
///    J     :     current version of the Jacobian matrix (m x n)
///    p     :     current set of parameters (n x 1)
///    y     :     model evaluation at current  set of parameters, y_hat(t,p) (m x 1)
///
///    Returns
///    -------
///    J     :     rank-1 update to Jacobian Matrix J(i,j)=dy(i)/dp(j) (m x n)
fn lm_broyden_j(
    p_old: &DVector<f64>,
    y_old: &DVector<f64>,
    j: &DMatrix<f64>,
    p: &DVector<f64>,
    y: &DVector<f64>,
) -> DMatrix<f64> {
    let h = p - p_old;
    let dy = y - y_old;

    let a: DMatrix<f64> = (dy - j * &h) * h.transpose();
    let b = h.dot(&h);
    // Broyden rank-1 update eq'n
    j + a / b
}
///    Evaluate the linearized fitting matrix, JtWJ, and vector JtWdy, and
///    calculate the Chi-squared error function, Chi_sq used by Levenberg-Marquardt
///    algorithm (lm).
///    
///    Parameters
///    ----------
///    t      :     independent variables used as arg to lm_func (m x 1)
///    p_old  :     previous parameter values (n x 1)
///    y_old  :     previous model ... y_old = y_hat(t,p_old) (m x 1)
///    dX2    :     previous change in Chi-squared criteria (1 x 1)
///    J      :     Jacobian of model, y_hat, with respect to parameters, p (m x n)
///    p      :     current parameter values (n x 1)
///    y_dat  :     data to be fit by func(t,p,c) (m x 1)
///    weight :     the weighting vector for least squares fit inverse of
///                 the squared standard measurement errors
///    dp     :     fractional increment of 'p' for numerical derivatives
///                  - dp(j)>0 central differences calculated
///                  - dp(j)<0 one sided differences calculated
///                  - dp(j)=0 sets corresponding partials to zero; i.e. holds p(j) fixed
///
///     Returns
///     -------
///     JtWJ   :     linearized Hessian matrix (inverse of covariance matrix) (n x n)
///     JtWdy  :     linearized fitting vector (n x m)
///     Chi_sq :     Chi-squared criteria: weighted sum of the squared residuals WSSR
///     y_hat  :     model evaluated with parameters 'p' (m x 1)
///     J :          Jacobian of model, y_hat, with respect to parameters, p (m x n)

fn lm_matx(
    t: &DVector<f64>,
    p_old: &DVector<f64>,
    y_old: &DVector<f64>,
    dx2: f64,
    j: &DMatrix<f64>,
    p: &DVector<f64>,
    y_dat: &DVector<f64>,
    weight: &DVector<f64>,
    dp: &DVector<f64>,
) -> (DMatrix<f64>, DVector<f64>, f64, DVector<f64>, DMatrix<f64>) {
    unsafe {
        let npar = p.len();
        // evaluate model using parameters 'p'
        let y_hat = lm_func(t, p);
        FUNC_CALLS += 1;

        let j_new = if ITERATION % (2 * npar) == 0 || dx2 > 0.0 {
            // finite difference
            lm_fd_j(t, p, &y_hat, dp)
        } else {
            // rank-1 update
            lm_broyden_j(p_old, y_old, j, p, &y_hat)
        };
        // residual error between model and data
        let delta_y = y_dat - &y_hat;
        // residual error between model and data
        //  Chi-squared error criteria
        let chi_sq = delta_y.dot(&delta_y.component_mul(weight));

        let jt = j_new.transpose();
        let w_matrix = DMatrix::from_diagonal(weight);
        let jtw = &jt * &w_matrix;
        let jtwj = &jtw * &j_new;
        let jtwdy = &jtw * &delta_y;

        (jtwj, jtwdy, chi_sq, y_hat, j_new)
    }
}
//  Levenberg Marquardt curve-fitting: minimize sum of weighted squared residuals

//     Parameters
//     ----------
//     p : initial guess of parameter values (n x 1)
//     t : independent variables (used as arg to lm_func) (m x 1)
//     y_dat : data to be fit by func(t,p) (m x 1)
//
//     Returns
//     -------
//     p       : least-squares optimal estimate of the parameter values
//     redX2   : reduced Chi squared error criteria - should be close to 1
//     sigma_p : asymptotic standard error of the parameters
//     sigma_y : asymptotic standard error of the curve-fit
//     corr_p  : correlation matrix of the parameters
//     R_sq    : R-squared cofficient of multiple determination
//     cvg_hst : convergence history (col 1: function calls, col 2: reduced chi-sq,
//               col 3 through n: parameter values). Row number corresponds to
//               iteration number.
pub fn lm(
    p_init: &DVector<f64>,
    t: &DVector<f64>,
    y_dat: &DVector<f64>,
) -> (
    DVector<f64>,
    f64,
    DVector<f64>,
    DVector<f64>,
    DMatrix<f64>,
    f64,
    DMatrix<f64>,
) {
    unsafe {
        ITERATION = 0;
        FUNC_CALLS = 0;

        let eps = f64::EPSILON;
        let npar = p_init.len();
        let npnt = y_dat.len();

        let mut p = p_init.clone();
        // previous set of parameters
        let mut p_old = DVector::zeros(npar);
        // previous model, y_old = y_hat(t,p_old)
        let mut y_old = DVector::zeros(npnt);
        let mut x2 = 1e-3 / eps;
        let mut x2_old = 1e-3 / eps;
        let mut j = DMatrix::zeros(npnt, npar);
        let dof = (npnt - npar + 1) as f64;

        if t.len() != y_dat.len() {
            panic!("The length of t must equal the length of y_dat!");
        }
        // weights or a scalar weight value ( weight >= 0 )
        let weight = DVector::from_element(npnt, 1.0 / y_dat.dot(y_dat));
        // fractional increment of 'p' for numerical derivatives
        let dp = DVector::from_element(npar, -0.001);
        //lower bounds for parameter values
        let p_min = p_init.map(|x| -100.0 * x.abs());
        //upper bounds for parameter values
        let p_max = p_init.map(|x| 100.0 * x.abs());

        let max_iter = 1000; // maximum number of iterations
        let epsilon_1 = 1e-3; //convergence tolerance for gradient
        let epsilon_2 = 1e-3; //convergence tolerance for parameters
        let epsilon_4 = 1e-1; // determines acceptance of a L-M step
        let lambda_0 = 1e-2; // initial value of damping paramter, lambda
        let lambda_up_fac = 11.0; // factor for increasing lambda
        let lambda_dn_fac = 9.0; //factor for decreasing lambda
        let update_type = 1; // 1: Levenberg-Marquardt lambda update, 2: Quadratic update, 3: Nielsen's lambda update equations 

        let mut stop = false;
        //  initialize Jacobian with finite difference calculation
        let (mut jtwj, mut jtwdy, x2_new, mut y_hat, j_new) =
            lm_matx(t, &p_old, &y_old, 1.0, &j, &p, y_dat, &weight, &dp);
        x2 = x2_new;
        j = j_new;

        if jtwdy.amax() < epsilon_1 {
            println!("*** Your Initial Guess is Extremely Close to Optimal ***");
        }

        let mut lambda = if update_type == 1 {
            lambda_0
        } else {
            lambda_0 * jtwj.diagonal().max()
        };
        // previous value of X2
        x2_old = x2;
        //  initialize convergence history
        let mut cvg_hst = DMatrix::zeros(max_iter, npar + 2);
        //    # -------- Start Main Loop ----------- #
        while !stop && ITERATION <= max_iter {
            ITERATION += 1;

            let h = if update_type == 1 {
                let diag_jtwj = DMatrix::from_diagonal(&jtwj.diagonal());
                (jtwj.clone() + lambda * diag_jtwj)
                    .lu()
                    .solve(&jtwdy)
                    .unwrap()
            } else {
                let eye = DMatrix::identity(npar, npar);
                (jtwj.clone() + lambda * eye).lu().solve(&jtwdy).unwrap()
            };
            //  apply constraints
            let mut p_try = &p + &h;
            for i in 0..npar {
                p_try[i] = p_try[i].max(p_min[i]).min(p_max[i]);
            }
            //   residual error using p_try
            let delta_y = y_dat - &lm_func(t, &p_try);

            if !delta_y.iter().all(|&x| x.is_finite()) {
                stop = true;
                break;
            }

            FUNC_CALLS += 1;
            let x2_try = delta_y.dot(&delta_y.component_mul(&weight));

            let rho_num = h.dot(&(lambda * &h + &jtwdy));
            //residual error using p_try
            let rho_den = x2 - x2_try;
            let rho = if rho_den != 0.0 {
                rho_num / rho_den
            } else {
                0.0
            };
            // it IS significantly better
            if rho > epsilon_4 {
                let dx2 = x2 - x2_old;
                x2_old = x2;
                p_old = p.clone();
                y_old = y_hat.clone();
                p = p_try;

                let result = lm_matx(t, &p_old, &y_old, dx2, &j, &p, y_dat, &weight, &dp);
                jtwj = result.0;
                jtwdy = result.1;
                x2 = result.2;
                y_hat = result.3;
                j = result.4;

                lambda = if update_type == 1 {
                    (lambda / lambda_dn_fac).max(1e-7)
                } else {
                    lambda * f64::max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0).powi(3))
                };
            } else {
                //% do not accept p_try
                x2 = x2_old;

                if ITERATION % (2 * npar) == 0 {
                    let result = lm_matx(t, &p_old, &y_old, -1.0, &j, &p, y_dat, &weight, &dp);
                    jtwj = result.0;
                    jtwdy = result.1;
                    j = result.4;
                }
                // increase lambda  ==> gradient descent method
                // Levenberg
                lambda = if update_type == 1 {
                    (lambda * lambda_up_fac).min(1e7)
                } else {
                    lambda * 2.0
                };
            }

            cvg_hst[(ITERATION - 1, 0)] = FUNC_CALLS as f64;
            cvg_hst[(ITERATION - 1, 1)] = x2 / dof;

            for i in 0..npar {
                cvg_hst[(ITERATION - 1, i + 2)] = p[i];
            }

            if jtwdy.amax() < epsilon_1 && ITERATION > 2 {
                println!("**** Convergence in r.h.s. (\"JtWdy\")  ****");
                stop = true;
            }

            let h_rel_max = h
                .iter()
                .zip(p.iter())
                .map(|(&hi, &pi)| hi.abs() / (pi.abs() + 1e-12))
                .fold(0.0, f64::max);

            if h_rel_max < epsilon_2 && ITERATION > 2 {
                println!("**** Convergence in Parameters ****");
                stop = true;
            }

            if ITERATION == max_iter {
                println!("!! Maximum Number of Iterations Reached Without Convergence !!");
                stop = true;
            }
        }
        // --- End of Main Loop --- #
        //--- convergence achieved, find covariance and confidence intervals

        // ---- Error Analysis ----
        //  recompute equal weights for paramter error analysis
        let weight_uniform =
            DVector::from_element(npnt, dof / (y_dat - &y_hat).dot(&(y_dat - &y_hat)));
        //  reduced Chi-square
        let red_x2 = x2 / dof;

        let result = lm_matx(t, &p_old, &y_old, -1.0, &j, &p, y_dat, &weight_uniform, &dp);
        jtwj = result.0;
        j = result.4;
        // standard error of parameters
        let covar_p = jtwj.try_inverse().unwrap();
        let sigma_p = covar_p.diagonal().map(|x| x.sqrt());
        // standard error of the fit
        let mut sigma_y = DVector::zeros(npnt);
        for i in 0..npnt {
            let j_row = j.row(i);
            sigma_y[i] = (j_row * &covar_p * j_row.transpose())[(0, 0)].sqrt();
        }

        let sigma_p_outer = &sigma_p * sigma_p.transpose();
        //parameter correlation matrix
        let corr_p = covar_p.component_div(&sigma_p_outer);

        let r_sq = 0.0;

        let cvg_hst_final = cvg_hst.rows(0, ITERATION).into_owned();

        println!("\nLM fitting results:");
        for i in 0..npar {
            println!("-----------------------------");
            println!("parameter      = p{}", i + 1);
            println!("fitted value   = {:.4}", p[i]);
            println!("standard error = {:.2} %", 100.0 * sigma_p[i] / p[i].abs());
        }

        (p, red_x2, sigma_p, sigma_y, corr_p, r_sq, cvg_hst_final)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_Gavin() {
        fn lm_func_test(t: &DVector<f64>, p: &DVector<f64>) -> DVector<f64> {
            t.map(|ti| p[0] * (-ti / p[1]).exp() + p[2] * (ti / p[3]).sin())
        }
        let t = DVector::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let y_dat = lm_func_test(&t, &DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]));
        let p_init = DVector::from_vec(vec![0.0, 0.0, 3.0, 4.0]);
        let params = lm(&p_init, &t, &y_dat);
        println!("\n \n Parameters: {:?}", params);
    }
}
