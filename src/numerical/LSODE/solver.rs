use super::core::*;
use super::dense::DenseBackend;
use super::jacobian::{DenseAnalyticJac, DenseAotJac, DenseFdJacobian, JacobianEvaluator};
/// =========================
/// BDF solver
/// =========================

pub struct BdfSolver<F, J, L>
where
    F: OdeRhs,
    L: LinearBackend,
    J: JacobianEvaluator<F, L::Storage>,
{
    rhs: F,
    jac: J,
    lin: L,
    pub core: BdfCore,
    opts: BdfOptions,
    tol: TolMode,
}

impl<F, J, L> BdfSolver<F, J, L>
where
    F: OdeRhs,
    L: LinearBackend,
    J: JacobianEvaluator<F, L::Storage>,
{
    pub fn new(
        mut rhs: F,
        jac: J,
        lin: L,
        t0: f64,
        y0: Vec<f64>,
        tol: TolMode,
        opts: BdfOptions,
    ) -> Result<Self, OdeError> {
        let n = y0.len();
        if n == 0 {
            return Err(OdeError::IllegalInput("state dimension must be > 0"));
        }
        if opts.max_order == 0 || opts.max_order > 5 {
            return Err(OdeError::IllegalInput("BDF max_order must be in 1..=5"));
        }

        let h0 = opts.h0.unwrap_or(1.0e-6).clamp(opts.h_min, opts.h_max);
        let mut core = BdfCore::new(n, t0, y0, h0, opts.max_order);

        rhs.eval(t0, &core.y, &mut core.f);
        core.stats.n_f_evals += 1;

        let mut solver = Self {
            rhs,
            jac,
            lin,
            core,
            opts,
            tol,
        };

        solver.compute_ewt()?;
        solver.initialize_history();
        solver.core.order = 1;
        solver.core.order_cap = 1;
        Ok(solver)
    }

    #[inline(always)]
    pub fn t(&self) -> f64 {
        self.core.t
    }

    #[inline(always)]
    pub fn y(&self) -> &[f64] {
        &self.core.y
    }

    #[inline(always)]
    pub fn stats(&self) -> &SolverStats {
        &self.core.stats
    }

    /// Integrate until tout.
    pub fn integrate_to(&mut self, tout: f64) -> Result<(), OdeError> {
        let forward = tout >= self.core.t;

        for _ in 0..self.opts.max_steps {
            if (self.core.t - tout).abs() == 0.0 {
                return Ok(());
            }

            if forward {
                if self.core.t + self.core.h > tout {
                    self.core.h = tout - self.core.t;
                }
            } else if self.core.t + self.core.h < tout {
                self.core.h = tout - self.core.t;
            }

            match self.try_step() {
                Ok(()) => {
                    if (self.core.t - tout).abs() == 0.0 {
                        return Ok(());
                    }
                }
                Err(OdeError::StepRejected) => {
                    continue;
                }
                Err(e) => return Err(e),
            }
        }

        Err(OdeError::TooManySteps)
    }
    fn update_order_cap_from_history(&mut self) {
        self.core.order_cap = (self.core.n_accept_total + 1)
            .min(self.core.max_order)
            .max(1);
        if self.core.order > self.core.order_cap {
            self.core.order = self.core.order_cap;
        }
    }
    /// One outer BDF step attempt with internal Newton solve.
    pub fn try_step(&mut self) -> Result<(), OdeError> {
        self.core.stats.n_steps += 1;

        if self.core.h.abs() < self.opts.h_min {
            return Err(OdeError::StepSizeUnderflow);
        }

        // Force current stabilized path to be pure BDF1 / backward Euler.
        self.core.order = 1;
        self.core.order_cap = 1;

        self.predict();
        self.compute_ewt()?;
        self.update_bdf_coeffs();

        let converged = self.correct()?;
        if !converged {
            self.reject_step_after_nonlinear_failure()?;
            return Err(OdeError::StepRejected);
        }

        // Stricter acceptance now that solver is stable.
        let err = self.error_norm(&self.core.acor)?;
        if err <= 0.8 {
            self.accept_step(err)?;
            Ok(())
        } else {
            self.reject_step_after_error_test(err)?;
            Err(OdeError::StepRejected)
        }
    }

    /// -------------------------
    /// Initialization
    /// -------------------------

    fn initialize_history(&mut self) {
        self.core.zn.col_mut(0).copy_from_slice(&self.core.y);

        if self.core.max_order >= 1 {
            let h = self.core.h;
            let z1 = self.core.zn.col_mut(1);
            for i in 0..self.core.n {
                z1[i] = h * self.core.f[i];
            }
        }

        if self.core.max_order >= 2 {
            self.core.zn.fill_zero_from(2);
        }

        self.core.yhist.block_mut(0).copy_from_slice(&self.core.y);
        for k in 1..=self.core.max_order {
            self.core.yhist.block_mut(k).fill(0.0);
        }
    }
    /// -------------------------
    /// Tolerances / norms
    /// -------------------------

    fn compute_ewt(&mut self) -> Result<(), OdeError> {
        match &self.tol {
            TolMode::Scalar { rtol, atol } => {
                if *rtol < 0.0 || *atol < 0.0 {
                    return Err(OdeError::IllegalInput("negative tolerance"));
                }
                for i in 0..self.core.n {
                    let yscale = self.core.y_new[i].abs();
                    self.core.ewt[i] = atol + rtol * yscale;
                    if self.core.ewt[i] <= 0.0 {
                        return Err(OdeError::IllegalInput("non-positive error weight"));
                    }
                }
            }
            TolMode::Vector { rtol, atol } => {
                if rtol.len() != self.core.n || atol.len() != self.core.n {
                    return Err(OdeError::IllegalInput("vector tolerance length mismatch"));
                }
                for i in 0..self.core.n {
                    let yscale = self.core.y_new[i].abs();
                    let ewt = atol[i] + rtol[i] * yscale;
                    if ewt <= 0.0 {
                        return Err(OdeError::IllegalInput("non-positive error weight"));
                    }
                    self.core.ewt[i] = ewt;
                }
            }
        }
        Ok(())
    }

    #[inline]
    fn error_norm(&self, v: &[f64]) -> Result<f64, OdeError> {
        if v.len() != self.core.n {
            return Err(OdeError::IllegalInput("error_norm length mismatch"));
        }
        let mut s = 0.0;
        for i in 0..self.core.n {
            let z = v[i] / self.core.ewt[i];
            s += z * z;
        }
        Ok((s / self.core.n as f64).sqrt())
    }

    /// -------------------------
    /// Predictor / corrector
    /// -------------------------
    fn predict(&mut self) {
        if self.core.order == 1 {
            for i in 0..self.core.n {
                self.core.y_pred[i] = self.core.y[i] + self.core.h * self.core.f[i];
            }
            self.core.y_new.copy_from_slice(&self.core.y_pred);
            self.core.acor.fill(0.0);
            self.core.delta.fill(0.0);
            return;
        }

        predict_nordsieck(&self.core.zn, &mut self.core.zpred, self.core.order);
        self.core.y_pred.copy_from_slice(self.core.zpred.col(0));
        self.core.y_new.copy_from_slice(&self.core.y_pred);
        self.core.acor.fill(0.0);
        self.core.delta.fill(0.0);
    }

    fn update_bdf_coeffs(&mut self) {
        let q = self.core.order;
        self.core.gamma = self.core.h * BDF_FORMULAS[q].beta;
    }

    fn build_residual(&mut self) {
        let q = self.core.order;

        build_affine_part(&self.core.yhist, q, &mut self.core.affine);

        build_bdf_residual(
            &self.core.y_new,
            &self.core.f_new,
            &self.core.affine,
            self.core.h,
            q,
            &mut self.core.residual,
        );

        for i in 0..self.core.n {
            self.core.delta[i] = -self.core.residual[i];
        }
    }

    fn need_jacobian_refresh(&self, newton_iter: usize) -> bool {
        if !self.core.jac_current {
            return true;
        }
        if newton_iter == 0 && self.core.steps_since_jac >= self.opts.jacobian_reuse_steps {
            return true;
        }
        false
    }

    fn correct(&mut self) -> Result<bool, OdeError> {
        let t_new = self.core.t + self.core.h;

        for newton_iter in 0..self.opts.max_newton_iters {
            self.rhs.eval(t_new, &self.core.y_new, &mut self.core.f_new);
            self.core.stats.n_f_evals += 1;
            self.core.stats.n_newton_iters += 1;

            self.build_residual();

            if self.need_jacobian_refresh(newton_iter) {
                self.jac.eval(
                    &mut self.rhs,
                    t_new,
                    &self.core.y_new,
                    self.lin.storage_mut(),
                );
                self.core.stats.n_j_evals += 1;

                self.lin.assemble_system_matrix(self.core.gamma);
                self.lin.factorize()?;
                self.core.stats.n_lu_factorizations += 1;

                self.core.jac_current = true;
                self.core.steps_since_jac = 0;
            }

            self.lin.solve_in_place(&mut self.core.delta)?;
            self.core.stats.n_linear_solves += 1;

            let del_norm = self.error_norm(&self.core.delta)?;
            self.core.last_correction_norm = del_norm;

            for i in 0..self.core.n {
                self.core.y_new[i] += self.core.delta[i];
                self.core.acor[i] += self.core.delta[i];
            }

            if del_norm <= self.opts.newton_tol {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// -------------------------
    /// Step acceptance/rejection
    /// -------------------------

    fn accept_step(&mut self, err: f64) -> Result<(), OdeError> {
        self.core.t += self.core.h;
        self.core.y.copy_from_slice(&self.core.y_new);
        self.core.f.copy_from_slice(&self.core.f_new);

        self.update_history_after_accept();

        self.core.stats.n_accepted += 1;
        self.core.steps_since_jac += 1;

        self.choose_next_step(err)?;
        self.maybe_change_order_after_accept(err);

        Ok(())
    }
    fn reject_step_after_error_test(&mut self, err: f64) -> Result<(), OdeError> {
        self.core.stats.n_rejected += 1;
        self.core.n_reject_consecutive += 1;
        self.core.n_accept_since_order_change = 0;

        let factor = (0.9 / err.powf(1.0 / ((self.core.order + 1) as f64)))
            .clamp(self.opts.step_shrink_min, 0.5);

        self.core.h *= factor;
        if self.core.h.abs() < self.opts.h_min {
            return Err(OdeError::StepSizeUnderflow);
        }

        self.maybe_lower_order_after_reject();
        self.core.jac_current = false;
        Ok(())
    }

    fn reject_step_after_nonlinear_failure(&mut self) -> Result<(), OdeError> {
        self.core.stats.n_rejected += 1;
        self.core.n_reject_consecutive += 1;
        self.core.n_accept_since_order_change = 0;

        self.core.h *= 0.5;

        if self.core.h.abs() < self.opts.h_min {
            return Err(OdeError::StepSizeUnderflow);
        }

        self.maybe_lower_order_after_reject();
        self.core.jac_current = false;
        Ok(())
    }

    fn choose_next_step(&mut self, err: f64) -> Result<(), OdeError> {
        let mut factor = if err == 0.0 {
            self.opts.step_growth_max
        } else {
            (0.9 / err.powf(1.0 / ((self.core.order + 1) as f64)))
                .clamp(1.0, self.opts.step_growth_max)
        };

        // During startup / shortly after order changes, avoid aggressive growth.
        if self.core.n_accept_since_order_change < 2
            || self.core.n_accept_total < self.core.order + 2
        {
            factor = factor.min(1.5);
        }

        self.core.h = (self.core.h * factor).clamp(-self.opts.h_max, self.opts.h_max);

        if self.core.h.abs() < self.opts.h_min {
            return Err(OdeError::StepSizeUnderflow);
        }
        Ok(())
    }

    fn maybe_change_order_after_accept(&mut self, _err: f64) {
        self.core.n_accept_total += 1;
        self.core.n_accept_since_order_change += 1;
        self.core.n_reject_consecutive = 0;

        self.core.order_cap = 1;
        self.core.order = 1;
    }
    /*
    fn update_history_after_accept(&mut self) {
        let n = self.core.n;
        let q = self.core.order;

        // 1. Update raw y-history first: [y_n, y_{n-1}, ...]
        self.core.yhist.push_front(&self.core.y);

        // 2. Build backward differences from updated y-history into diff_work.
        // Reuse tmp layout as output prefix; diff_work is the recurrence workspace.
        //
        // Store diffs compactly into zpred.data prefix temporarily, then convert.
        build_backward_differences_in_place(
            &self.core.yhist,
            q,
            &mut self.core.diff_work,
            &mut self.core.zpred.data,
        );

        // 3. Convert backward differences -> accepted Nordsieck history.
        backward_differences_to_nordsieck(q, n, &self.core.zpred.data, &mut self.core.zn.data);

        // 4. Zero inactive higher columns beyond current order.
        if q < self.core.max_order {
            self.core.zn.fill_zero_from(q + 1);
        }

        // 5. Force consistency of first derivative with accepted RHS.
        // This is important for stiff problems.
        if q >= 1 {
            self.reconcile_first_nordsieck_derivative();
        }
    }*/
    fn update_history_after_accept(&mut self) {
        self.core.yhist.push_front(&self.core.y);

        if self.core.order == 1 {
            // Keep accepted history consistent with variable-step backward Euler.
            self.core.zn.col_mut(0).copy_from_slice(&self.core.y);

            let h = self.core.h;
            let z1 = self.core.zn.col_mut(1);
            for i in 0..self.core.n {
                z1[i] = h * self.core.f[i];
            }

            if self.core.max_order >= 2 {
                self.core.zn.fill_zero_from(2);
            }

            return;
        }

        let n = self.core.n;
        let q = self.core.order;

        build_backward_differences_in_place(
            &self.core.yhist,
            q,
            &mut self.core.diff_work,
            &mut self.core.zpred.data,
        );

        backward_differences_to_nordsieck(q, n, &self.core.zpred.data, &mut self.core.zn.data);

        if q < self.core.max_order {
            self.core.zn.fill_zero_from(q + 1);
        }

        if q >= 1 {
            self.reconcile_first_nordsieck_derivative();
        }
    }

    pub fn reconcile_first_nordsieck_derivative(&mut self) {
        let n = self.core.n;
        let h = self.core.h;
        let z1 = self.core.zn.col_mut(1);

        for i in 0..n {
            z1[i] = h * self.core.f[i];
        }
    }
    fn should_raise_order_after_accept(&self, err: f64) -> bool {
        // During very early startup stay at BDF1
        /*
        if self.core.n_accept_total < 2 {
            return false;
        }

        if self.core.order >= self.core.order_cap {
            return false;
        }

        if self.core.order >= self.core.max_order {
            return false;
        }

        // Need a couple of stable accepted steps before changing order
        if self.core.n_accept_since_order_change < 2 {
            return false;
        }

        err < 0.5
        */
        false
    }
    fn maybe_lower_order_after_reject(&mut self) {
        if self.core.order > 1 && self.core.n_reject_consecutive >= 2 {
            self.core.order -= 1;
            self.core.n_accept_since_order_change = 0;
        }
    }
}

/// =========================
/// Convenience constructors
/// =========================

pub type BdfDenseAnalyticSolver<F, Cb> = BdfSolver<F, DenseAnalyticJac<Cb>, DenseBackend>;
pub type BdfDenseAotSolver<F> = BdfSolver<F, DenseAotJac, DenseBackend>;
pub type BdfDenseFdSolver<F> = BdfSolver<F, DenseFdJacobian, DenseBackend>;

pub fn new_bdf_dense_analytic<F, Cb>(
    rhs: F,
    jac: Cb,
    t0: f64,
    y0: Vec<f64>,
    tol: TolMode,
    opts: BdfOptions,
) -> Result<BdfDenseAnalyticSolver<F, Cb>, OdeError>
where
    F: OdeRhs,
    Cb: FnMut(f64, &[f64], &mut [f64], usize),
{
    let n = y0.len();
    let lin = DenseBackend::new(n);
    let jac = DenseAnalyticJac::new(jac);
    BdfSolver::new(rhs, jac, lin, t0, y0, tol, opts)
}

pub fn new_bdf_dense_aot<F>(
    rhs: F,
    jac: fn(f64, &[f64], &mut [f64], usize),
    t0: f64,
    y0: Vec<f64>,
    tol: TolMode,
    opts: BdfOptions,
) -> Result<BdfDenseAotSolver<F>, OdeError>
where
    F: OdeRhs,
{
    let n = y0.len();
    let lin = DenseBackend::new(n);
    let jac = DenseAotJac::new(jac);
    BdfSolver::new(rhs, jac, lin, t0, y0, tol, opts)
}

pub fn new_bdf_dense_fd<F>(
    rhs: F,
    t0: f64,
    y0: Vec<f64>,
    tol: TolMode,
    opts: BdfOptions,
    fd: FdOptions,
) -> Result<BdfDenseFdSolver<F>, OdeError>
where
    F: OdeRhs,
{
    let n = y0.len();
    let lin = DenseBackend::new(n);
    let jac = DenseFdJacobian::new(n, fd);
    BdfSolver::new(rhs, jac, lin, t0, y0, tol, opts)
}

//====================================================================================

use super::banded::BandedBackend;
use super::jacobian::{BandedAnalyticJac, BandedFdJacobian};

pub type BdfBandedAnalyticSolver<F, Cb> = BdfSolver<F, BandedAnalyticJac<Cb>, BandedBackend>;

pub type BdfBandedFdSolver<F> = BdfSolver<F, BandedFdJacobian, BandedBackend>;

pub fn new_bdf_banded_analytic<F, Cb>(
    rhs: F,
    jac: Cb,
    ml: usize,
    mu: usize,
    t0: f64,
    y0: Vec<f64>,
    tol: TolMode,
    opts: BdfOptions,
) -> Result<BdfBandedAnalyticSolver<F, Cb>, OdeError>
where
    F: OdeRhs,
    Cb: FnMut(f64, &[f64], &mut [f64], usize, usize, usize, usize),
{
    let n = y0.len();
    let lin = BandedBackend::new(n, ml, mu);
    let jac = BandedAnalyticJac::new(jac);
    BdfSolver::new(rhs, jac, lin, t0, y0, tol, opts)
}

pub fn new_bdf_banded_fd<F>(
    rhs: F,
    ml: usize,
    mu: usize,
    t0: f64,
    y0: Vec<f64>,
    tol: TolMode,
    opts: BdfOptions,
    fd: FdOptions,
) -> Result<BdfBandedFdSolver<F>, OdeError>
where
    F: OdeRhs,
{
    let n = y0.len();
    let lin = BandedBackend::new(n, ml, mu);
    let jac = BandedFdJacobian::new(n, fd);
    BdfSolver::new(rhs, jac, lin, t0, y0, tol, opts)
}
//====================================================================================
use super::jacobian::SparseAnalyticJac;
use super::sparse::{SparseBackend, SparseMatrixStorage};

pub type BdfSparseAnalyticSolver<F, Cb> = BdfSolver<F, SparseAnalyticJac<Cb>, SparseBackend>;

pub fn new_bdf_sparse_analytic<F, Cb>(
    rhs: F,
    jac: Cb,
    storage: SparseMatrixStorage,
    t0: f64,
    y0: Vec<f64>,
    tol: TolMode,
    opts: BdfOptions,
) -> Result<BdfSparseAnalyticSolver<F, Cb>, OdeError>
where
    F: OdeRhs,
    Cb: FnMut(f64, &[f64], &mut [f64]),
{
    let jac = SparseAnalyticJac::new(jac);
    let lin = SparseBackend::new(storage);
    BdfSolver::new(rhs, jac, lin, t0, y0, tol, opts)
}
