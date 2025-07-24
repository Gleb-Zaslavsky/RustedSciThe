use nalgebra::{DMatrix, DVector};

/// Levenberg Marquardt curve-fitting: minimize sum of weighted squared residuals
/// Based on H.P. Gavin's implementation
pub struct LevenbergMarquardtGavin {
    pub max_evals: usize,
    pub eps_grad: f64,      // convergence tolerance for gradient
    pub eps_coeff: f64,     // convergence tolerance for coefficients
    pub eps_chi: f64,       // convergence tolerance for red. Chi-sqr
    pub eps_lm: f64,        // determines acceptance of a L-M step
    pub lambda_0: f64,      // initial value of L-M parameter
    pub lambda_up_fac: f64, // factor for increasing lambda
    pub lambda_dn_fac: f64, // factor for decreasing lambda
    pub update_type: i32,   // 1: Levenberg-Marquardt, 2: Quadratic, 3: Nielsen's
    pub print_level: i32,   // >1 intermediate results; >2 plots
}

impl Default for LevenbergMarquardtGavin {
    fn default() -> Self {
        Self {
            max_evals: 0, // Will be set to 10*Ncof^2 in solve
            eps_grad: 1e-3,
            eps_coeff: 1e-3,
            eps_chi: 1e-1,
            eps_lm: 1e-1,
            lambda_0: 1e-2,
            lambda_up_fac: 11.0,
            lambda_dn_fac: 9.0,
            update_type: 1,
            print_level: 3,
        }
    }
}

pub struct LMResult {
    pub a: DVector<f64>,               // optimal coefficients
    pub red_x2: f64,                   // reduced Chi squared
    pub sigma_a: Option<DVector<f64>>, // standard error of coefficients
    pub sigma_y: Option<DVector<f64>>, // standard error of the fit
    pub corr_a: Option<DMatrix<f64>>,  // correlation matrix
    pub r_sq: Option<f64>,             // R-squared coefficient
    pub cvg_hst: DMatrix<f64>,         // convergence history
    pub func_calls: usize,
    pub iteration: usize,
}

impl LevenbergMarquardtGavin {
    pub fn new() -> Self {
        Self::default()
    }

    /// Main Levenberg-Marquardt solver
    /// func: function that evaluates model y_hat = func(t, a, c)
    /// a: initial guess of coefficient values
    /// t: independent variables
    /// y_dat: data to be fit
    /// weight: weights (inverse of standard measurement errors)
    /// a_lb: lower bounds for coefficients
    /// a_ub: upper bounds for coefficients
    /// c: optional model constants
    /// jacobian_func: function that computes Jacobian matrix
    pub fn solve<F, J>(
        &mut self,
        func: F,
        mut a: DVector<f64>,
        t: &DMatrix<f64>,
        y_dat: &DVector<f64>,
        weight: Option<DVector<f64>>,
        a_lb: Option<DVector<f64>>,
        a_ub: Option<DVector<f64>>,
        c: Option<f64>,
        jacobian_func: J,
    ) -> Result<LMResult, String>
    where
        F: Fn(&DMatrix<f64>, &DVector<f64>, Option<f64>) -> DVector<f64>,
        J: Fn(&DMatrix<f64>, &DVector<f64>, &DVector<f64>, Option<f64>) -> DMatrix<f64>,
    {
        let mut iteration = 0;
        let mut func_calls = 0;

        let ncof = a.len();
        let npnt = y_dat.len();
        let mut a_old = DVector::zeros(ncof);
        let mut y_old = DVector::zeros(npnt);
        let mut x2 = 1e-3 / f64::EPSILON;
        let mut x2_old = 1e-3 / f64::EPSILON;
        let mut j = DMatrix::zeros(npnt, ncof);
        let dof = npnt - ncof;

        // Set default max_evals if not set
        if self.max_evals == 0 {
            self.max_evals = 10 * ncof * ncof;
        }

        // Check dimensions
        if t.nrows() != npnt {
            return Err(format!(
                "Number of rows of t ({}) must equal length of y_dat ({})",
                t.nrows(),
                npnt
            ));
        }

        // Set default weight
        let weight = weight.unwrap_or_else(|| {
            let y_norm = y_dat.dot(y_dat);
            DVector::from_element(npnt, 1.0 / y_norm)
        });

        // Set default bounds
        let a_lb = a_lb.unwrap_or_else(|| a.map(|x| -100.0 * x.abs()));
        let a_ub = a_ub.unwrap_or_else(|| a.map(|x| 100.0 * x.abs()));

        let c_val = c.unwrap_or(1.0);

        // Initialize
        let y_init = func(t, &a, Some(c_val));
        func_calls += 1;

        // Check if weights are uniform
        let weight = if weight.iter().all(|&w| (w - weight[0]).abs() < f64::EPSILON) {
            if self.print_level > 0 {
                println!("using uniform weights for error analysis");
            }
            DVector::from_element(npnt, weight[0].abs())
        } else {
            weight.map(|w| w.abs())
        };

        // Initialize Jacobian and matrices
        let (mut jtw_j, mut jtw_dy, x2_new, y_hat, j_new) = self.lm_matx(
            &func,
            &jacobian_func,
            t,
            &a_old,
            &y_old,
            1.0,
            &j,
            &a,
            y_dat,
            &weight,
            Some(c_val),
        )?;

        x2 = x2_new;
        j = j_new;
        func_calls += 1;

        // Check initial gradient convergence
        if jtw_dy.amax() < self.eps_grad {
            if self.print_level > 0 {
                println!(" *** Your initial guess meets gradient convergence criteria ***");
                println!(" *** To converge further, reduce epsilon_1 and restart ***");
                println!(" *** epsilon_1 = {:.6e}", self.eps_grad);
            }
            return Ok(LMResult {
                a,
                red_x2: x2 / dof as f64,
                sigma_a: None,
                sigma_y: None,
                corr_a: None,
                r_sq: None,
                cvg_hst: DMatrix::zeros(1, ncof + 3),
                func_calls,
                iteration,
            });
        }

        // Initialize lambda
        let mut lambda = match self.update_type {
            1 => self.lambda_0,                          // Marquardt
            _ => self.lambda_0 * jtw_j.diagonal().max(), // Quadratic and Nielsen
        };

        let mut nu = 2.0; // For Nielsen update

        x2_old = x2;
        let mut cvg_hst = DMatrix::zeros(self.max_evals, ncof + 3);
        let mut stop = false;

        // Main iteration loop
        while !stop && func_calls <= self.max_evals {
            iteration += 1;

            // Compute incremental change in coefficients
            let x_matrix = match self.update_type {
                1 => {
                    // Marquardt
                    let mut x = jtw_j.clone();
                    for i in 0..ncof {
                        x[(i, i)] += lambda * jtw_j[(i, i)];
                    }
                    x
                }
                _ => {
                    // Quadratic and Nielsen
                    let mut x = jtw_j.clone();
                    for i in 0..ncof {
                        x[(i, i)] += lambda;
                    }
                    x
                }
            };

            // Ensure matrix is well-conditioned
            let mut x_reg = x_matrix;
            while self.rcond(&x_reg) < 1e-15 {
                let trace_avg = x_reg.trace() / ncof as f64;
                for i in 0..ncof {
                    x_reg[(i, i)] += 1e-6 * trace_avg;
                }
            }

            // Solve for step h
            let h = match x_reg.lu().solve(&jtw_dy) {
                Some(solution) => solution,
                None => return Err("Failed to solve linear system".to_string()),
            };

            // Apply step with bounds
            let mut a_try = a.clone();
            for i in 0..ncof {
                a_try[i] = (a[i] + h[i]).max(a_lb[i]).min(a_ub[i]);
            }

            // Evaluate function at trial point
            let delta_y = y_dat - func(t, &a_try, Some(c_val));
            if !delta_y.iter().all(|&x| x.is_finite()) {
                stop = true;
                break;
            }
            func_calls += 1;

            let mut x2_try = delta_y.component_mul(&weight).dot(&delta_y);

            // Quadratic line search
            if self.update_type == 2 {
                let alpha = jtw_dy.dot(&h) / ((x2_try - x2) / 2.0 + 2.0 * jtw_dy.dot(&h));
                let h_scaled = h.clone() * alpha;

                a_try = a.clone();
                for i in 0..ncof {
                    a_try[i] = (a[i] + h_scaled[i]).max(a_lb[i]).min(a_ub[i]);
                }

                let delta_y_new = y_dat - func(t, &a_try, Some(c_val));
                func_calls += 1;
                x2_try = delta_y_new.component_mul(&weight).dot(&delta_y_new);
            }

            // Compute rho for step acceptance
            let rho = match self.update_type {
                1 => {
                    let lambda_diag = DVector::from_fn(ncof, |i, _| lambda * jtw_j[(i, i)]);
                    let denominator =
                        h.clone().component_mul(&lambda_diag).dot(&h) + jtw_dy.dot(&h);
                    (x2 - x2_try) / denominator.abs()
                }
                _ => {
                    let denominator = lambda * h.dot(&h) + jtw_dy.dot(&h);
                    (x2 - x2_try) / denominator.abs()
                }
            };

            // Accept or reject step
            if rho > self.eps_chi {
                // Accept step
                let dx2 = x2 - x2_old;
                x2_old = x2;
                a_old = a.clone();
                y_old = func(t, &a, Some(c_val));
                a = a_try;

                // Recompute matrices
                let (jtw_j_new, jtw_dy_new, x2_new, _y_hat, j_new) = self.lm_matx(
                    &func,
                    &jacobian_func,
                    t,
                    &a_old,
                    &y_old,
                    dx2,
                    &j,
                    &a,
                    y_dat,
                    &weight,
                    Some(c_val),
                )?;

                jtw_j = jtw_j_new;
                jtw_dy = jtw_dy_new;
                x2 = x2_new;
                j = j_new;

                // Decrease lambda
                match self.update_type {
                    1 => lambda = (lambda / self.lambda_dn_fac).max(1e-7),
                    2 => {
                        // Note: alpha would need to be computed from quadratic update
                        lambda = (lambda / (1.0 + 1.0)).max(1e-7); // Simplified
                    }
                    3 => {
                        lambda = lambda
                            * ((1.0 / 3.0) as f64).max((1.0 - (2.0 * rho - 1.0).powi(3)) as f64);
                        nu = 2.0;
                    }
                    _ => {}
                }
            } else {
                // Reject step
                x2 = x2_old;

                // Recompute Jacobian periodically
                if iteration % (2 * ncof) == 0 {
                    let (jtw_j_new, jtw_dy_new, _dx2, _y_hat, j_new) = self.lm_matx(
                        &func,
                        &jacobian_func,
                        t,
                        &a_old,
                        &y_old,
                        -1.0,
                        &j,
                        &a,
                        y_dat,
                        &weight,
                        Some(c_val),
                    )?;
                    jtw_j = jtw_j_new;
                    jtw_dy = jtw_dy_new;
                    j = j_new;
                }

                // Increase lambda
                match self.update_type {
                    1 => lambda = (lambda * self.lambda_up_fac).min(1e7),
                    2 => lambda = lambda + ((x2_try - x2) / 2.0 / 1.0).abs(), // Simplified
                    3 => {
                        lambda = lambda * nu;
                        nu = 2.0 * nu;
                    }
                    _ => {}
                }
            }

            // Print progress
            if self.print_level > 1 {
                println!(
                    ">{:3}:{:3} | chi_sq={:10.3e} | lambda={:8.1e}",
                    iteration,
                    func_calls,
                    x2 / dof as f64,
                    lambda
                );
                print!("      a  :  ");
                for i in 0..ncof {
                    print!(" {:10.3e}", a[i]);
                }
                println!();
                print!("    da/a :  ");
                for i in 0..ncof {
                    print!(" {:10.3e}", h[i] / a[i]);
                }
                println!();
            }

            // Update convergence history
            let mut row = DVector::zeros(ncof + 3);
            row[0] = func_calls as f64;
            for i in 0..ncof {
                row[i + 1] = a[i];
            }
            row[ncof + 1] = x2 / dof as f64;
            row[ncof + 2] = lambda;
            cvg_hst.set_row(iteration - 1, &row.transpose());

            // Check convergence criteria
            if jtw_dy.amax() < self.eps_grad && iteration > 2 {
                if self.print_level > 0 {
                    println!(" **** Convergence in r.h.s. (\"JtWdy\")  ****");
                    println!(" **** epsilon_1 = {:.6e}", self.eps_grad);
                }
                stop = true;
            }

            let max_rel_change = h
                .iter()
                .zip(a.iter())
                .map(|(h_i, a_i)| (h_i / (a_i.abs() + 1e-12)).abs())
                .fold(0.0, f64::max);

            if max_rel_change < self.eps_coeff && iteration > 2 {
                if self.print_level > 0 {
                    println!(" **** Convergence in Parameters ****");
                    println!(" **** epsilon_2 = {:.6e}", self.eps_coeff);
                }
                stop = true;
            }

            if (x2 / (dof as f64) < self.eps_chi) && (iteration > 2) {
                if self.print_level > 0 {
                    println!(" **** Convergence in reduced Chi-square  ****");
                    println!(" **** epsilon_3 = {:.6e}", self.eps_chi);
                }
                stop = true;
            }

            if func_calls >= self.max_evals {
                println!(" !! Maximum Number of Function Calls Reached Without Convergence !!");
                stop = true;
            }
        } // End of main loop

        // --- Error Analysis ---

        // Recompute equal weights for parameter error analysis if needed
        let final_weight = if weight.iter().all(|&w| (w - weight[0]).abs() < f64::EPSILON) {
            let delta_y = y_dat - func(t, &a, Some(c_val));
            let weight_val = dof as f64 / delta_y.dot(&delta_y);
            DVector::from_element(npnt, weight_val)
        } else {
            weight
        };

        // Recompute final matrices
        let (final_jtw_j, _jtw_dy, final_x2, y_hat, _j) = self.lm_matx(
            &func,
            &jacobian_func,
            t,
            &a_old,
            &y_old,
            -1.0,
            &j,
            &a,
            y_dat,
            &final_weight,
            Some(c_val),
        )?;

        let red_x2 = final_x2 / dof as f64;

        // Compute covariance matrix and standard errors
        let (sigma_a, sigma_y, corr_a) = if self.rcond(&final_jtw_j) > 1e-15 {
            let covar_a = match final_jtw_j.clone().try_inverse() {
                Some(inv) => inv,
                None => {
                    let mut regularized = final_jtw_j.clone();
                    let trace_avg = regularized.trace() / ncof as f64;
                    for i in 0..ncof {
                        regularized[(i, i)] += 1e-6 * trace_avg;
                    }
                    regularized
                        .try_inverse()
                        .unwrap_or_else(|| DMatrix::identity(ncof, ncof))
                }
            };

            let sigma_a = DVector::from_fn(ncof, |i, _| covar_a[(i, i)].sqrt());

            // Compute sigma_y
            let mut sigma_y = DVector::zeros(npnt);
            for i in 0..npnt {
                let j_row = j.row(i);
                sigma_y[i] = (j_row * &covar_a * j_row.transpose())[(0, 0)].sqrt();
            }

            // Compute correlation matrix
            let mut corr_a = DMatrix::zeros(ncof, ncof);
            for i in 0..ncof {
                for j in 0..ncof {
                    corr_a[(i, j)] = covar_a[(i, j)] / (sigma_a[i] * sigma_a[j]);
                }
            }

            (Some(sigma_a), Some(sigma_y), Some(corr_a))
        } else {
            (None, None, None)
        };

        // Compute R-squared
        let r_sq = if y_hat.len() == y_dat.len() {
            let y_mean = y_dat.mean();
            let ss_tot: f64 = y_dat.iter().map(|&y| (y - y_mean).powi(2)).sum();
            let ss_res: f64 = y_dat
                .iter()
                .zip(y_hat.iter())
                .map(|(&y, &yh)| (y - yh).powi(2))
                .sum();
            Some(1.0 - ss_res / ss_tot)
        } else {
            None
        };

        // Trim convergence history
        let cvg_hst_trimmed = cvg_hst.rows(0, iteration).into_owned();

        Ok(LMResult {
            a,
            red_x2,
            sigma_a,
            sigma_y,
            corr_a,
            r_sq,
            cvg_hst: cvg_hst_trimmed,
            func_calls,
            iteration,
        })
    }

    /// Helper function to compute matrices (simplified version without Jacobian calculation)
    fn lm_matx<F, J>(
        &self,
        func: &F,
        jacobian_func: &J,
        t: &DMatrix<f64>,
        _a_old: &DVector<f64>,
        _y_old: &DVector<f64>,
        _dx2: f64,
        _j_old: &DMatrix<f64>,
        a: &DVector<f64>,
        y_dat: &DVector<f64>,
        weight: &DVector<f64>,
        c: Option<f64>,
    ) -> Result<(DMatrix<f64>, DVector<f64>, f64, DVector<f64>, DMatrix<f64>), String>
    where
        F: Fn(&DMatrix<f64>, &DVector<f64>, Option<f64>) -> DVector<f64>,
        J: Fn(&DMatrix<f64>, &DVector<f64>, &DVector<f64>, Option<f64>) -> DMatrix<f64>,
    {
        let npnt = y_dat.len();
        let ncof = a.len();

        // Evaluate model
        let y_hat = func(t, a, c);

        // Compute Jacobian using provided function
        let J = jacobian_func(t, a, &y_hat, c);

        // Compute residuals
        let delta_y = y_dat - &y_hat;

        // Compute Chi-squared
        let chi_sq = delta_y.component_mul(weight).dot(&delta_y);

        // Compute JtWJ
        let mut jtw_j = DMatrix::zeros(ncof, ncof);
        for i in 0..ncof {
            for k in 0..ncof {
                let mut sum = 0.0;
                for j in 0..npnt {
                    sum += J[(j, i)] * weight[j] * J[(j, k)];
                }
                jtw_j[(i, k)] = sum;
            }
        }

        // Compute JtWdy
        let mut jtw_dy = DVector::zeros(ncof);
        for i in 0..ncof {
            let mut sum = 0.0;
            for k in 0..npnt {
                sum += J[(k, i)] * weight[k] * delta_y[k];
            }
            jtw_dy[i] = sum;
        }

        Ok((jtw_j, jtw_dy, chi_sq, y_hat, J))
    }

    /// Helper function to estimate condition number (simplified)
    fn rcond(&self, matrix: &DMatrix<f64>) -> f64 {
        /*
        match matrix.svd(true, true) {
            Ok(svd) => {
                let singular_values = svd.singular_values;
                let max_sv = singular_values.max();
                let min_sv = singular_values.min();
                if min_sv > 0.0 {
                    min_sv / max_sv
                } else {
                    0.0
                }
            }
            Err(_) => 0.0
        }
         */
        0.0
    }
}
