use super::core::{FdOptions, OdeRhs};
use super::dense::DenseMatrixStorage;

use super::banded::BandedMatrixStorage;
/// =========================
/// Jacobian evaluator
/// F = RHS type
/// S = storage type
/// =========================

pub trait JacobianEvaluator<F, S>
where
    F: OdeRhs,
{
    fn eval(&mut self, rhs: &mut F, t: f64, y: &[f64], storage: &mut S);
}

/// =========================
/// Dense analytic Jacobian
/// callback writes full dense Jacobian
/// in column-major layout
/// =========================

pub struct DenseAnalyticJac<Cb> {
    callback: Cb,
}

impl<Cb> DenseAnalyticJac<Cb> {
    pub fn new(callback: Cb) -> Self {
        Self { callback }
    }
}

impl<F, Cb> JacobianEvaluator<F, DenseMatrixStorage> for DenseAnalyticJac<Cb>
where
    F: OdeRhs,
    Cb: FnMut(f64, &[f64], &mut [f64], usize),
{
    #[inline]
    fn eval(&mut self, _rhs: &mut F, t: f64, y: &[f64], storage: &mut DenseMatrixStorage) {
        (self.callback)(t, y, &mut storage.jac, storage.n);
    }
}

/// =========================
/// Dense AOT Jacobian
/// same shape as analytic callback,
/// but intended for generated extern or fn ptr
/// =========================

pub struct DenseAotJac {
    callback: fn(f64, &[f64], &mut [f64], usize),
}

impl DenseAotJac {
    pub fn new(callback: fn(f64, &[f64], &mut [f64], usize)) -> Self {
        Self { callback }
    }
}

impl<F> JacobianEvaluator<F, DenseMatrixStorage> for DenseAotJac
where
    F: OdeRhs,
{
    #[inline]
    fn eval(&mut self, _rhs: &mut F, t: f64, y: &[f64], storage: &mut DenseMatrixStorage) {
        (self.callback)(t, y, &mut storage.jac, storage.n);
    }
}

/// =========================
/// Dense FD Jacobian
/// =========================

pub struct DenseFdJacobian {
    fd: FdOptions,
    f_base: Vec<f64>,
    f_work: Vec<f64>,
    y_work: Vec<f64>,
}

impl DenseFdJacobian {
    pub fn new(n: usize, fd: FdOptions) -> Self {
        Self {
            fd,
            f_base: vec![0.0; n],
            f_work: vec![0.0; n],
            y_work: vec![0.0; n],
        }
    }

    #[inline(always)]
    fn perturbation(&self, yj: f64) -> f64 {
        let scale = yj.abs().max(1.0);
        let mut h = self.fd.rel_step * scale;
        if h.abs() < self.fd.abs_step {
            h = self.fd.abs_step;
        }
        if yj < 0.0 { -h } else { h }
    }
}

impl<F> JacobianEvaluator<F, DenseMatrixStorage> for DenseFdJacobian
where
    F: OdeRhs,
{
    fn eval(&mut self, rhs: &mut F, t: f64, y: &[f64], storage: &mut DenseMatrixStorage) {
        let n = storage.n;
        debug_assert_eq!(y.len(), n);

        rhs.eval(t, y, &mut self.f_base);
        self.y_work.copy_from_slice(y);
        storage.zero_jac();

        for j in 0..n {
            let yj = y[j];
            let delta = self.perturbation(yj);
            self.y_work[j] = yj + delta;

            rhs.eval(t, &self.y_work, &mut self.f_work);

            let inv_delta = 1.0 / delta;
            let col = storage.jac_col_mut(j);
            for i in 0..n {
                col[i] = (self.f_work[i] - self.f_base[i]) * inv_delta;
            }

            self.y_work[j] = yj;
        }
    }
}

// =========================
// Banded FD Jacobian
// =========================

/// =========================
/// Banded analytic Jacobian
///
/// Callback signature:
///   (t, y, jac_band, n, ml, mu, ldab)
/// =========================

pub struct BandedAnalyticJac<Cb> {
    callback: Cb,
}

impl<Cb> BandedAnalyticJac<Cb> {
    pub fn new(callback: Cb) -> Self {
        Self { callback }
    }
}

impl<F, Cb> JacobianEvaluator<F, BandedMatrixStorage> for BandedAnalyticJac<Cb>
where
    F: OdeRhs,
    Cb: FnMut(f64, &[f64], &mut [f64], usize, usize, usize, usize),
{
    #[inline]
    fn eval(&mut self, _rhs: &mut F, t: f64, y: &[f64], storage: &mut BandedMatrixStorage) {
        storage.zero_jac();
        (self.callback)(
            t,
            y,
            &mut storage.jac,
            storage.n,
            storage.ml,
            storage.mu,
            storage.ldab,
        );
    }
}

/// =========================
/// Banded FD Jacobian
///
/// Writes only entries inside the band.
/// =========================

pub struct BandedFdJacobian {
    fd: FdOptions,
    f_base: Vec<f64>,
    f_work: Vec<f64>,
    y_work: Vec<f64>,
}

impl BandedFdJacobian {
    pub fn new(n: usize, fd: FdOptions) -> Self {
        Self {
            fd,
            f_base: vec![0.0; n],
            f_work: vec![0.0; n],
            y_work: vec![0.0; n],
        }
    }

    #[inline(always)]
    fn perturbation(&self, yj: f64) -> f64 {
        let scale = yj.abs().max(1.0);
        let mut h = self.fd.rel_step * scale;
        if h.abs() < self.fd.abs_step {
            h = self.fd.abs_step;
        }
        if yj < 0.0 { -h } else { h }
    }
}

impl<F> JacobianEvaluator<F, BandedMatrixStorage> for BandedFdJacobian
where
    F: OdeRhs,
{
    fn eval(&mut self, rhs: &mut F, t: f64, y: &[f64], storage: &mut BandedMatrixStorage) {
        let n = storage.n;
        debug_assert_eq!(y.len(), n);

        rhs.eval(t, y, &mut self.f_base);
        self.y_work.copy_from_slice(y);
        storage.zero_jac();

        for j in 0..n {
            let yj = y[j];
            let delta = self.perturbation(yj);
            self.y_work[j] = yj + delta;

            rhs.eval(t, &self.y_work, &mut self.f_work);

            let i_min = j.saturating_sub(storage.mu);
            let i_max = (j + storage.ml + 1).min(n);

            let inv_delta = 1.0 / delta;
            for i in i_min..i_max {
                let val = (self.f_work[i] - self.f_base[i]) * inv_delta;
                storage.set_jac(i, j, val);
            }

            self.y_work[j] = yj;
        }
    }
}
//====================================================================================

use super::sparse::SparseMatrixStorage;

/// =========================
/// Sparse analytic Jacobian
///
/// Callback updates only the values array, in the fixed CSC nonzero order.
/// Pattern lives in SparseMatrixStorage.
/// =========================

pub struct SparseAnalyticJac<Cb> {
    callback: Cb,
}

impl<Cb> SparseAnalyticJac<Cb> {
    pub fn new(callback: Cb) -> Self {
        Self { callback }
    }
}

impl<F, Cb> JacobianEvaluator<F, SparseMatrixStorage> for SparseAnalyticJac<Cb>
where
    F: OdeRhs,
    Cb: FnMut(f64, &[f64], &mut [f64]),
{
    #[inline]
    fn eval(&mut self, _rhs: &mut F, t: f64, y: &[f64], storage: &mut SparseMatrixStorage) {
        storage.zero_jac();
        (self.callback)(t, y, &mut storage.jac_values);
    }
}
