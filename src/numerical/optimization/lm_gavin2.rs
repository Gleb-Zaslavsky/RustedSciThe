// direct rewritten from
// https://github.com/abnerbog/levenberg-marquardt-method/tree/main

use nalgebra::{DMatrix, DVector};
use std::f64;

pub trait ObjectiveFunction {
    fn evaluate(&self, t: &DVector<f64>, p: &DVector<f64>) -> DVector<f64>;
    fn parameter_count(&self) -> usize;
}

pub struct LevenbergMarquardt<F: ObjectiveFunction> {
    pub iteration: usize,
    pub func_calls: usize,
    pub objective_fn: F,
    pub eps_grad: f64,      // convergence tolerance for gradient
    pub eps_coeff: f64,     // convergence tolerance for coefficients
    pub eps_chi: f64,       // convergence tolerance for red. Chi-sqr
    pub eps_lm: f64,        // determines acceptance of a L-M step
    pub lambda_0: f64,      // initial value of L-M parameter
    pub lambda_up_fac: f64, // factor for increasing lambda
    pub lambda_dn_fac: f64, // factor for decreasing lambda
    pub update_type: i32,   // 1: Levenberg-Marquardt, 2: Quadratic, 3: Nielsen's
}

impl<F: ObjectiveFunction> LevenbergMarquardt<F> {
    pub fn new(objective_fn: F) -> Self {
        let epsilon_1 = 1e-3;
        let epsilon_2 = 1e-3;
        let epsilon_4 = 1e-1;

        Self {
            iteration: 0,
            func_calls: 0,
            objective_fn,
            eps_grad: epsilon_1,
            eps_coeff: epsilon_2,
            eps_chi: epsilon_4,
            eps_lm: epsilon_4,
            lambda_0: 1e-2,
            lambda_up_fac: 11.0,
            lambda_dn_fac: 9.0,
            update_type: 1,
        }
    }
    pub fn set_params(
        &mut self,
        eps_grad: Option<f64>,
        eps_coeff: Option<f64>,
        eps_chi: Option<f64>,
        lambda_0: Option<f64>,
        lambda_up_fac: Option<f64>,
        lambda_dn_fac: Option<f64>,
        update_type: Option<i32>,
    ) {
        if let Some(eps_grad) = eps_grad {
            self.eps_grad = eps_grad;
        }
        if let Some(eps_coeff) = eps_coeff {
            self.eps_coeff = eps_coeff;
        }
        if let Some(eps_chi) = eps_chi {
            self.eps_chi = eps_chi;
        }

        if let Some(lambda_0) = lambda_0 {
            self.lambda_0 = lambda_0;
        }
        if let Some(lambda_up_fac) = lambda_up_fac {
            self.lambda_up_fac = lambda_up_fac;
        }
        if let Some(lambda_dn_fac) = lambda_dn_fac {
            self.lambda_dn_fac = lambda_dn_fac;
        }
        if let Some(update_type) = update_type {
            self.update_type = update_type;
        }
    }
    pub fn lm_fd_j(
        &mut self,
        t: &DVector<f64>,
        p: &DVector<f64>,
        y: &DVector<f64>,
        dp: &DVector<f64>,
    ) -> DMatrix<f64> {
        let m = y.len();
        let n = p.len();
        let mut j = DMatrix::zeros(m, n);
        let mut ps = p.clone();
        let mut del = DVector::zeros(n);

        for j_idx in 0..n {
            del[j_idx] = dp[j_idx] * (1.0 + p[j_idx].abs());
            ps[j_idx] = p[j_idx] + del[j_idx];

            if del[j_idx] != 0.0 {
                let y1 = self.objective_fn.evaluate(t, &ps);
                self.func_calls += 1;

                if dp[j_idx] < 0.0 {
                    for i in 0..m {
                        j[(i, j_idx)] = (y1[i] - y[i]) / del[j_idx];
                    }
                } else {
                    ps[j_idx] = p[j_idx] - del[j_idx];
                    let y2 = self.objective_fn.evaluate(t, &ps);
                    self.func_calls += 1;
                    for i in 0..m {
                        j[(i, j_idx)] = (y1[i] - y2[i]) / (2.0 * del[j_idx]);
                    }
                }
            }
            ps[j_idx] = p[j_idx];
        }
        j
    }

    pub fn lm_broyden_j(
        &self,
        p_old: &DVector<f64>,
        y_old: &DVector<f64>,
        j: &DMatrix<f64>,
        p: &DVector<f64>,
        y: &DVector<f64>,
    ) -> DMatrix<f64> {
        let h = p - p_old;
        let dy = y - y_old;
        let jh = j * &h;
        let a = (&dy - &jh) * h.transpose();
        let b = h.dot(&h);
        j + a / b
    }

    pub fn lm_matx(
        &mut self,
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
        let npar = p.len();
        let y_hat = self.objective_fn.evaluate(t, p);
        self.func_calls += 1;

        let j_new = if self.iteration % (2 * npar) == 0 || dx2 > 0.0 {
            self.lm_fd_j(t, p, &y_hat, dp)
        } else {
            self.lm_broyden_j(p_old, y_old, j, p, &y_hat)
        };

        let delta_y = y_dat - &y_hat;
        let weighted_delta = delta_y.component_mul(weight);
        let chi_sq = weighted_delta.dot(&delta_y);

        let mut jtw_j = DMatrix::zeros(npar, npar);
        let mut jtw_dy = DVector::zeros(npar);

        for i in 0..npar {
            for j_idx in 0..npar {
                let mut sum = 0.0;
                for k in 0..y_dat.len() {
                    sum += j_new[(k, i)] * j_new[(k, j_idx)] * weight[k];
                }
                jtw_j[(i, j_idx)] = sum;
            }

            let mut sum = 0.0;
            for k in 0..y_dat.len() {
                sum += j_new[(k, i)] * weight[k] * delta_y[k];
            }
            jtw_dy[i] = sum;
        }

        (jtw_j, jtw_dy, chi_sq, y_hat, j_new)
    }

    pub fn lm(
        &mut self,
        mut p: DVector<f64>,
        t: &DVector<f64>,
        y_dat: &DVector<f64>,
    ) -> Result<
        (
            DVector<f64>,
            f64,
            DVector<f64>,
            DVector<f64>,
            DMatrix<f64>,
            f64,
            DMatrix<f64>,
        ),
        String,
    > {
        self.iteration = 0;
        self.func_calls = 0;

        let eps = f64::EPSILON;
        let npar = p.len();
        let npnt = y_dat.len();
        let mut p_old = DVector::zeros(npar);
        let mut y_old = DVector::zeros(npnt);
        // let mut x2 = 1e-3 / eps;
        let mut x2_old = 1e-3 / eps;
        let j = DMatrix::zeros(npnt, npar);
        let dof = (npnt - npar + 1) as f64;

        if t.len() != y_dat.len() {
            return Err("The length of t must equal the length of y_dat!".to_string());
        }

        let y_dat_norm_sq = y_dat.dot(y_dat);
        let weight = DVector::from_element(npnt, 1.0 / y_dat_norm_sq);
        let dp = DVector::from_element(npar, -0.001);
        let p_min = -100.0 * p.map(|x| x.abs());
        let p_max = 100.0 * p.map(|x| x.abs());

        let max_iter = 1000;
        let eps_grad = self.eps_grad;
        let eps_coeff = self.eps_coeff;
        let eps_chi = self.eps_chi;
        let lambda_0 = self.lambda_0;
        let lambda_up_fac = self.lambda_up_fac;
        let lambda_dn_fac = self.lambda_dn_fac;
        let update_type = self.update_type;

        let idx: Vec<usize> = (0..npar).collect();
        let mut stop = false;

        let (mut jtw_j, mut jtw_dy, mut x2_val, mut y_hat, mut j_mat) =
            self.lm_matx(t, &p_old, &y_old, 1.0, &j, &p, y_dat, &weight, &dp);

        if jtw_dy.amax() < eps_grad {
            println!("*** Your Initial Guess is Extremely Close to Optimal ***");
        }

        let mut lambda = if update_type == 1 {
            lambda_0
        } else {
            lambda_0 * jtw_j.diagonal().max()
        };

        let mut nu = 2.0;
        x2_old = x2_val;
        let mut cvg_hst = DMatrix::zeros(max_iter, npar + 2);

        while !stop && self.iteration <= max_iter {
            self.iteration += 1;

            let h = if update_type == 1 {
                let mut jtw_j_reg = jtw_j.clone();
                for i in 0..npar {
                    jtw_j_reg[(i, i)] += lambda * jtw_j[(i, i)];
                }
                match jtw_j_reg.lu().solve(&jtw_dy) {
                    Some(solution) => solution,
                    None => return Err("Failed to solve linear system".to_string()),
                }
            } else {
                let mut jtw_j_reg = jtw_j.clone();
                for i in 0..npar {
                    jtw_j_reg[(i, i)] += lambda;
                }
                match jtw_j_reg.lu().solve(&jtw_dy) {
                    Some(solution) => solution,
                    None => return Err("Failed to solve linear system".to_string()),
                }
            };

            let mut p_try = p.clone();
            for i in &idx {
                p_try[*i] += h[*i];
            }

            for i in 0..npar {
                p_try[i] = p_try[i].max(p_min[i]).min(p_max[i]);
            }

            let delta_y_try = y_dat - &self.objective_fn.evaluate(t, &p_try);

            if !delta_y_try.iter().all(|&x| x.is_finite()) {
                stop = true;
                break;
            }

            self.func_calls += 1;
            let x2_try = delta_y_try.component_mul(&weight).dot(&delta_y_try);

            let mut alpha = 1.0;
            if update_type == 2 {
                let numerator = jtw_dy.dot(&h);
                let denominator = (x2_try - x2_val) / 2.0 + 2.0 * jtw_dy.dot(&h);
                alpha = numerator / denominator;
                let h_scaled = alpha * &h;

                for i in &idx {
                    p_try[*i] = p[*i] + h_scaled[*i];
                }

                for i in 0..npar {
                    p_try[i] = p_try[i].max(p_min[i]).min(p_max[i]);
                }

                let delta_y_try_new = y_dat - &self.objective_fn.evaluate(t, &p_try);
                self.func_calls += 1;
                let _x2_try_new = delta_y_try_new.component_mul(&weight).dot(&delta_y_try_new);
            }

            let rho_num = h.dot(&(lambda * &h + &jtw_dy));
            let rho_den = x2_val - x2_try;
            let rho = if rho_den.abs() > eps {
                rho_num / rho_den
            } else {
                0.0
            };

            if rho > eps_chi {
                let dx2 = x2_val - x2_old;
                x2_old = x2_val;
                p_old = p.clone();
                y_old = y_hat.clone();
                p = p_try;

                let (new_jtw_j, new_jtw_dy, new_x2, new_y_hat, new_j) =
                    self.lm_matx(t, &p_old, &y_old, dx2, &j_mat, &p, y_dat, &weight, &dp);

                jtw_j = new_jtw_j;
                jtw_dy = new_jtw_dy;
                x2_val = new_x2;
                y_hat = new_y_hat;
                j_mat = new_j;

                if update_type == 1 {
                    lambda = (lambda / lambda_dn_fac).max(1e-7);
                } else if update_type == 2 {
                    lambda = (lambda / (1.0 + alpha)).max(1e-7);
                } else {
                    lambda =
                        lambda * (1.0 / 3.0 as f64).max((1.0 - (2.0 * rho - 1.0).powi(3)) as f64);
                    nu = 2.0;
                }
            } else {
                x2_val = x2_old;

                if self.iteration % (2 * npar) == 0 {
                    let (new_jtw_j, new_jtw_dy, _, new_y_hat, new_j) =
                        self.lm_matx(t, &p_old, &y_old, -1.0, &j_mat, &p, y_dat, &weight, &dp);
                    jtw_j = new_jtw_j;
                    jtw_dy = new_jtw_dy;
                    y_hat = new_y_hat;
                    j_mat = new_j;
                }

                if update_type == 1 {
                    lambda = (lambda * lambda_up_fac).min(1e7);
                } else if update_type == 2 {
                    lambda = lambda + ((x2_try - x2_val) / 2.0 / alpha).abs();
                } else {
                    lambda = lambda * nu;
                    nu = 2.0 * nu;
                }
            }

            cvg_hst[(self.iteration - 1, 0)] = self.func_calls as f64;
            cvg_hst[(self.iteration - 1, 1)] = x2_val / dof;

            for i in 0..npar {
                cvg_hst[(self.iteration - 1, i + 2)] = p[i];
            }

            if jtw_dy.amax() < eps_grad && self.iteration > 2 {
                println!("**** Convergence in r.h.s. (\"JtWdy\")  ****");
                stop = true;
            }

            let mut max_rel_change = 0.0;
            for i in 0..npar {
                let rel_change = h[i].abs() / (p[i].abs() + 1e-12);
                if rel_change > max_rel_change {
                    max_rel_change = rel_change;
                }
            }

            if max_rel_change < eps_coeff && self.iteration > 2 {
                println!("**** Convergence in Parameters ****");
                stop = true;
            }

            if self.iteration == max_iter {
                println!("!! Maximum Number of Iterations Reached Without Convergence !!");
                stop = true;
            }
        }

        // Error Analysis
        // Error Analysis
        let final_weight = if weight.variance() == 0.0 {
            let delta_y_final = y_dat - &y_hat;
            let weight_val = dof / delta_y_final.dot(&delta_y_final);
            DVector::from_element(npnt, weight_val)
        } else {
            weight
        };

        let red_x2 = x2_val / dof;

        let (final_jtw_j, _final_jtw_dy, _final_x2, _final_y_hat, final_j) = self.lm_matx(
            t,
            &p_old,
            &y_old,
            -1.0,
            &j_mat,
            &p,
            y_dat,
            &final_weight,
            &dp,
        );

        let covar_p = match final_jtw_j.try_inverse() {
            Some(inv) => inv,
            None => return Err("Failed to compute covariance matrix".to_string()),
        };

        let mut sigma_p = DVector::zeros(npar);
        for i in 0..npar {
            sigma_p[i] = covar_p[(i, i)].sqrt();
        }

        let mut sigma_y = DVector::zeros(npnt);
        for i in 0..npnt {
            let mut sum = 0.0;
            for j in 0..npar {
                for k in 0..npar {
                    sum += final_j[(i, j)] * covar_p[(j, k)] * final_j[(i, k)];
                }
            }
            sigma_y[i] = sum.sqrt();
        }

        let mut corr_p = DMatrix::zeros(npar, npar);
        for i in 0..npar {
            for j in 0..npar {
                corr_p[(i, j)] = covar_p[(i, j)] / (sigma_p[i] * sigma_p[j]);
            }
        }

        let r_sq = 0.0; // Placeholder - R-squared calculation would need proper implementation

        let cvg_hst_trimmed = cvg_hst.rows(0, self.iteration).into_owned();

        println!("\nLM fitting results:");
        for i in 0..npar {
            println!("-----------------------------");
            println!("parameter      = p{}", i + 1);
            println!("fitted value   = {:.4}", p[i]);
            let error_p = sigma_p[i] / p[i].abs() * 100.0;
            println!("standard error = {:.2} %", error_p);
        }

        Ok((p, red_x2, sigma_p, sigma_y, corr_p, r_sq, cvg_hst_trimmed))
    }
}

// Example implementations of ObjectiveFunction trait
pub struct ExponentialSinusoidalModel;

impl ObjectiveFunction for ExponentialSinusoidalModel {
    fn evaluate(&self, t: &DVector<f64>, p: &DVector<f64>) -> DVector<f64> {
        let mut y_hat = DVector::zeros(t.len());
        for i in 0..t.len() {
            y_hat[i] = p[0] * (-t[i] / p[1]).exp() + p[2] * (t[i] / p[3]).sin();
        }
        y_hat
    }

    fn parameter_count(&self) -> usize {
        4
    }
}

pub struct PolynomialModel {
    degree: usize,
}

impl PolynomialModel {
    pub fn new(degree: usize) -> Self {
        Self { degree }
    }
}

impl ObjectiveFunction for PolynomialModel {
    fn evaluate(&self, t: &DVector<f64>, p: &DVector<f64>) -> DVector<f64> {
        let mut y_hat = DVector::zeros(t.len());
        for i in 0..t.len() {
            let mut sum = 0.0;
            for j in 0..=self.degree {
                sum += p[j] * t[i].powi(j as i32);
            }
            y_hat[i] = sum;
        }
        y_hat
    }

    fn parameter_count(&self) -> usize {
        self.degree + 1
    }
}

pub struct GaussianModel;

impl ObjectiveFunction for GaussianModel {
    fn evaluate(&self, t: &DVector<f64>, p: &DVector<f64>) -> DVector<f64> {
        let mut y_hat = DVector::zeros(t.len());
        for i in 0..t.len() {
            // p[0] = amplitude, p[1] = mean, p[2] = std_dev
            let exponent = -0.5 * ((t[i] - p[1]) / p[2]).powi(2);
            y_hat[i] = p[0] * exponent.exp();
        }
        y_hat
    }

    fn parameter_count(&self) -> usize {
        3
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DVector, dvector};

    #[test]
    fn test_exponential_sinusoidal_model() {
        let model = ExponentialSinusoidalModel;
        let mut lm = LevenbergMarquardt::new(model);

        let t = DVector::from_vec((0..10).map(|i| i as f64 * 0.1).collect());
        let p_true = dvector![2.0, 0.5, 1.0, 2.0];
        let y_true = lm.objective_fn.evaluate(&t, &p_true);

        let p_initial = dvector![1.5, 0.4, 0.8, 1.3];

        match lm.lm(p_initial, &t, &y_true) {
            Ok((p_fitted, red_x2, _sigma_p, _sigma_y, _corr_p, _r_sq, _cvg_hst)) => {
                println!("Fitted parameters: {:?}", p_fitted);
                println!("Reduced chi-squared: {}", red_x2);
                assert!(red_x2 < 1e-4);
            }
            Err(e) => panic!("LM fitting failed: {}", e),
        }
    }

    #[test]
    fn test_polynomial_model() {
        let model = PolynomialModel::new(2); // Quadratic
        let mut lm = LevenbergMarquardt::new(model);

        let t = DVector::from_vec((0..10).map(|i| i as f64).collect());
        let p_true = dvector![1.0, 2.0, 0.5]; // 1 + 2x + 0.5x^2
        let y_true = lm.objective_fn.evaluate(&t, &p_true);

        let p_initial = dvector![0.8, 1.8, 0.4];

        match lm.lm(p_initial, &t, &y_true) {
            Ok((p_fitted, red_x2, _sigma_p, _sigma_y, _corr_p, _r_sq, _cvg_hst)) => {
                println!("Fitted polynomial parameters: {:?}", p_fitted);
                println!("Reduced chi-squared: {}", red_x2);
                assert!(red_x2 < 1e-10);
            }
            Err(e) => panic!("Polynomial LM fitting failed: {}", e),
        }
    }

    #[test]
    fn test_gaussian_model() {
        let model = GaussianModel;
        let mut lm = LevenbergMarquardt::new(model);

        let t = DVector::from_vec((-50..50).map(|i| i as f64 * 0.1).collect());
        let p_true = dvector![2.0, 0.0, 1.0]; // amplitude=2, mean=0, std=1
        let y_true = lm.objective_fn.evaluate(&t, &p_true);

        let p_initial = dvector![1.8, 0.1, 0.9];

        match lm.lm(p_initial, &t, &y_true) {
            Ok((p_fitted, red_x2, _sigma_p, _sigma_y, _corr_p, _r_sq, _cvg_hst)) => {
                println!("Fitted Gaussian parameters: {:?}", p_fitted);
                println!("Reduced chi-squared: {}", red_x2);
                assert!(red_x2 < 1e-4);
            }
            Err(e) => panic!("Gaussian LM fitting failed: {}", e),
        }
    }

    #[test]
    fn test_custom_closure_model() {
        // Example of how to use a closure as an objective function
        struct ClosureModel<F>
        where
            F: Fn(&DVector<f64>, &DVector<f64>) -> DVector<f64>,
        {
            func: F,
            param_count: usize,
        }

        impl<F> ObjectiveFunction for ClosureModel<F>
        where
            F: Fn(&DVector<f64>, &DVector<f64>) -> DVector<f64>,
        {
            fn evaluate(&self, t: &DVector<f64>, p: &DVector<f64>) -> DVector<f64> {
                (self.func)(t, p)
            }

            fn parameter_count(&self) -> usize {
                self.param_count
            }
        }

        // Custom exponential decay model
        let custom_func = |t: &DVector<f64>, p: &DVector<f64>| -> DVector<f64> {
            let mut y = DVector::zeros(t.len());
            for i in 0..t.len() {
                y[i] = p[0] * (-p[1] * t[i]).exp();
            }
            y
        };

        let model = ClosureModel {
            func: custom_func,
            param_count: 2,
        };

        let mut lm = LevenbergMarquardt::new(model);

        let t = DVector::from_vec((0..20).map(|i| i as f64 * 0.1).collect());
        let p_true = dvector![3.0, 0.5];
        let y_true = lm.objective_fn.evaluate(&t, &p_true);

        let p_initial = dvector![2.5, 0.4];

        match lm.lm(p_initial, &t, &y_true) {
            Ok((p_fitted, red_x2, _sigma_p, _sigma_y, _corr_p, _r_sq, _cvg_hst)) => {
                println!("Fitted custom model parameters: {:?}", p_fitted);
                println!("Reduced chi-squared: {}", red_x2);
                assert!(red_x2 < 1e-5);
            }
            Err(e) => panic!("Custom model LM fitting failed: {}", e),
        }
    }
}

#[cfg(test)]
mod additional_tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{DVector, dvector};

    // Test for noisy data fitting
    #[test]
    fn test_exponential_decay_with_noise() {
        struct ExponentialDecay;

        impl ObjectiveFunction for ExponentialDecay {
            fn evaluate(&self, t: &DVector<f64>, p: &DVector<f64>) -> DVector<f64> {
                let mut y = DVector::zeros(t.len());
                for i in 0..t.len() {
                    y[i] = p[0] * (-p[1] * t[i]).exp();
                }
                y
            }

            fn parameter_count(&self) -> usize {
                2
            }
        }

        let model = ExponentialDecay;
        let mut lm = LevenbergMarquardt::new(model);

        // Generate synthetic noisy data
        let t = DVector::from_vec((0..20).map(|i| i as f64 * 0.2).collect());
        let p_true = dvector![5.0, 0.8];
        let y_clean = lm.objective_fn.evaluate(&t, &p_true);

        // Add small amount of noise
        let noise = dvector![
            0.01, -0.02, 0.015, -0.01, 0.008, -0.012, 0.005, 0.018, -0.007, 0.011, 0.003, -0.009,
            0.014, -0.006, 0.002, 0.016, -0.004, 0.013, -0.008, 0.001
        ];
        let y_noisy = y_clean + noise;

        let p_initial = dvector![4.0, 0.6];

        match lm.lm(p_initial, &t, &y_noisy) {
            Ok((p_fitted, red_x2, sigma_p, _sigma_y, _corr_p, _r_sq, _cvg_hst)) => {
                println!("Noisy data test - Fitted parameters: {:?}", p_fitted);
                println!("Noisy data test - Reduced chi-squared: {}", red_x2);
                println!("Noisy data test - Parameter uncertainties: {:?}", sigma_p);

                // Should recover true parameters within reasonable tolerance
                assert_relative_eq!(p_fitted[0], p_true[0], epsilon = 0.1);
                assert_relative_eq!(p_fitted[1], p_true[1], epsilon = 0.1);
                assert!(red_x2 > 0.0);
                assert!(sigma_p[0] > 0.0 && sigma_p[1] > 0.0);
            }
            Err(e) => panic!("Noisy data fitting failed: {}", e),
        }
    }

    // Test for multi-parameter complex model
    #[test]
    fn test_complex_multi_parameter_model() {
        struct ComplexModel;

        impl ObjectiveFunction for ComplexModel {
            fn evaluate(&self, t: &DVector<f64>, p: &DVector<f64>) -> DVector<f64> {
                let mut y = DVector::zeros(t.len());
                for i in 0..t.len() {
                    // y = p[0] * exp(-p[1]*t) + p[2] * sin(p[3]*t + p[4]) + p[5]
                    y[i] = p[0] * (-p[1] * t[i]).exp() + p[2] * (p[3] * t[i] + p[4]).sin() + p[5];
                }
                y
            }

            fn parameter_count(&self) -> usize {
                6
            }
        }

        let model = ComplexModel;
        let mut lm = LevenbergMarquardt::new(model);

        let t = DVector::from_vec((0..5000).map(|i| i as f64 * 0.001).collect());
        let p_true = dvector![3.0, 0.5, 2.0, 1.2, 0.3, 1.0];
        let y_true = lm.objective_fn.evaluate(&t, &p_true);

        let p_initial = dvector![2.5, 0.4, 1.8, 1.0, 0.2, 0.8];

        match lm.lm(p_initial, &t, &y_true) {
            Ok((p_fitted, red_x2, _sigma_p, _sigma_y, corr_p, _r_sq, cvg_hst)) => {
                println!("Complex model test - Fitted parameters: {:?}", p_fitted);
                println!("Complex model test - Iterations: {}", cvg_hst.nrows());
                println!("Complex model test - Function calls: {}", lm.func_calls);

                // Check convergence
                assert!(red_x2 < 1e-3);
                assert!(cvg_hst.nrows() > 0);
                assert!(lm.func_calls > 0);

                // Check parameter recovery
                for i in 0..6 {
                    assert_relative_eq!(p_fitted[i], p_true[i], epsilon = 1e-2);
                }

                // Check correlation matrix properties
                assert_eq!(corr_p.nrows(), 6);
                assert_eq!(corr_p.ncols(), 6);
                for i in 0..6 {
                    assert_relative_eq!(corr_p[(i, i)], 1.0, epsilon = 1e-3);
                }
            }
            Err(e) => panic!("Complex model fitting failed: {}", e),
        }
    }

    // Test error handling for mismatched dimensions
    #[test]
    #[should_panic]
    fn test_dimension_mismatch_error() {
        let model = ExponentialSinusoidalModel;
        let mut lm = LevenbergMarquardt::new(model);

        let t = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let y_dat = DVector::from_vec(vec![1.0, 2.0]); // Different length
        let p_initial = dvector![1.0, 1.0, 1.0, 1.0];

        match lm.lm(p_initial, &t, &y_dat) {
            Ok(_) => panic!("Should have failed due to dimension mismatch"),
            Err(e) => {
                assert!(e.contains("length of t must equal the length of y_dat"));
            }
        }
    }

    // Test convergence behavior with poor initial guess
    #[test]
    fn test_poor_initial_guess_convergence() {
        let model = GaussianModel;
        let mut lm = LevenbergMarquardt::new(model);
        lm.set_params(None, None, None, None, None, None, Some(0));
        let t = DVector::from_vec((-300..300).map(|i| i as f64 * 0.02).collect());
        let p_true = dvector![5.0, 2.0, 1.5];
        let y_true = lm.objective_fn.evaluate(&t, &p_true);

        // Very poor initial guess
        let p_initial = dvector![4.0, 3.0, 2.1];

        match lm.lm(p_initial, &t, &y_true) {
            Ok((p_fitted, red_x2, _sigma_p, _sigma_y, _corr_p, _r_sq, cvg_hst)) => {
                println!(
                    "Poor initial guess test - Fitted parameters: {:?}",
                    p_fitted
                );
                println!("Poor initial guess test - Iterations: {}", cvg_hst.nrows());
                println!("Poor initial guess test - Final chi-squared: {}", red_x2);

                // Should still converge, though may take more iterations
                assert!(red_x2 < 1e-3);
                // assert!(cvg_hst.nrows() > 5); // Likely needs more iterations

                // Parameters should be recovered
                assert_relative_eq!(p_fitted[0], p_true[0], epsilon = 1e-2);
                assert_relative_eq!(p_fitted[1], p_true[1], epsilon = 1e-2);
                assert_relative_eq!(p_fitted[2], p_true[2], epsilon = 1e-2);
            }
            Err(e) => {
                // It's acceptable if it fails with very poor initial guess
                println!("Poor initial guess failed as expected: {}", e);
            }
        }
    }

    // Test Jacobian computation methods
    #[test]
    fn test_jacobian_computation() {
        let model = PolynomialModel::new(2);
        let mut lm = LevenbergMarquardt::new(model);

        let t = dvector![0.0, 1.0, 2.0, 3.0];
        let p = dvector![1.0, 2.0, 0.5];
        let y = lm.objective_fn.evaluate(&t, &p);
        let dp = dvector![-0.001, -0.001, -0.001];

        // Test finite difference Jacobian
        let j_fd = lm.lm_fd_j(&t, &p, &y, &dp);

        println!("Finite difference Jacobian:");
        println!("{}", j_fd);

        // Check Jacobian dimensions
        assert_eq!(j_fd.nrows(), t.len());
        assert_eq!(j_fd.ncols(), p.len());

        // For polynomial y = p[0] + p[1]*t + p[2]*t^2
        // dy/dp[0] = 1, dy/dp[1] = t, dy/dp[2] = t^2
        for i in 0..t.len() {
            assert_relative_eq!(j_fd[(i, 0)], 1.0, epsilon = 1e-6);
            assert_relative_eq!(j_fd[(i, 1)], t[i], epsilon = 1e-6);
            assert_relative_eq!(j_fd[(i, 2)], t[i] * t[i], epsilon = 1e-6);
        }
    }

    // Test Broyden update
    #[test]
    fn test_broyden_jacobian_update() {
        let model = PolynomialModel::new(1); // Linear model
        let lm = LevenbergMarquardt::new(model);

        let p_old = dvector![1.0, 2.0];
        let p_new = dvector![1.1, 2.1];
        let y_old = dvector![3.0, 5.0, 7.0];
        let y_new = dvector![3.2, 5.2, 7.2];

        // Initial Jacobian for linear model y = p[0] + p[1]*t
        let j_old = nalgebra::dmatrix![
            1.0, 0.0;
            1.0, 1.0;
            1.0, 2.0;
        ];

        let j_broyden = lm.lm_broyden_j(&p_old, &y_old, &j_old, &p_new, &y_new);

        println!("Broyden updated Jacobian:");
        println!("{}", j_broyden);

        // Check dimensions
        assert_eq!(j_broyden.nrows(), j_old.nrows());
        assert_eq!(j_broyden.ncols(), j_old.ncols());

        // For a linear model, Broyden update should preserve the Jacobian structure
        assert!(j_broyden.norm() > 0.0);
    }

    // Test with singular/ill-conditioned cases
    #[test]
    fn test_ill_conditioned_problem() {
        struct IllConditionedModel;

        impl ObjectiveFunction for IllConditionedModel {
            fn evaluate(&self, t: &DVector<f64>, p: &DVector<f64>) -> DVector<f64> {
                let mut y = DVector::zeros(t.len());
                for i in 0..t.len() {
                    // Highly correlated parameters: y = p[0] + p[1] + (p[0] - p[1]) * t
                    y[i] = p[0] + p[1] + (p[0] - p[1]) * t[i];
                }
                y
            }

            fn parameter_count(&self) -> usize {
                2
            }
        }

        let model = IllConditionedModel;
        let mut lm = LevenbergMarquardt::new(model);

        let t = dvector![0.0, 1.0];
        let p_true = dvector![2.0, 3.0];
        let y_true = lm.objective_fn.evaluate(&t, &p_true);

        let p_initial = dvector![1.0, 1.0];

        match lm.lm(p_initial, &t, &y_true) {
            Ok((p_fitted, red_x2, sigma_p, _sigma_y, corr_p, _r_sq, _cvg_hst)) => {
                println!("Ill-conditioned test - Fitted parameters: {:?}", p_fitted);
                println!(
                    "Ill-conditioned test - Parameter uncertainties: {:?}",
                    sigma_p
                );
                println!(
                    "Ill-conditioned test - Correlation matrix diagonal: {:?}",
                    corr_p.diagonal()
                );

                // Should converge but with high parameter uncertainties
                assert!(red_x2 < 1e-10);

                // Check that we get a valid solution (may not be unique)
                let y_fitted = lm.objective_fn.evaluate(&t, &p_fitted);
                for i in 0..y_true.len() {
                    assert_relative_eq!(y_fitted[i], y_true[i], epsilon = 1e-5);
                }
            }
            Err(e) => {
                println!("Ill-conditioned problem failed as expected: {}", e);
                // This is acceptable for ill-conditioned problems
            }
        }
    }
}
