#![allow(dead_code)]

use super::solver::new_bdf_dense_analytic;
use std::fmt;
/// =========================
/// Errors / stats / options
/// =========================

#[derive(Debug, Clone)]
pub enum OdeError {
    IllegalInput(&'static str),
    StepSizeUnderflow,
    TooManySteps,
    NewtonDiverged,
    LinearSolveFailed,
    SingularJacobian,
    StepRejected,
}

impl fmt::Display for OdeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OdeError::IllegalInput(msg) => write!(f, "illegal input: {msg}"),
            OdeError::StepSizeUnderflow => write!(f, "step size underflow"),
            OdeError::TooManySteps => write!(f, "too many steps"),
            OdeError::NewtonDiverged => write!(f, "newton iteration diverged"),
            OdeError::LinearSolveFailed => write!(f, "linear solve failed"),
            OdeError::SingularJacobian => write!(f, "singular jacobian"),
            OdeError::StepRejected => write!(f, "step rejected"),
        }
    }
}

impl std::error::Error for OdeError {}

#[derive(Debug, Clone, Default)]
pub struct SolverStats {
    pub n_steps: usize,
    pub n_accepted: usize,
    pub n_rejected: usize,
    pub n_f_evals: usize,
    pub n_j_evals: usize,
    pub n_lu_factorizations: usize,
    pub n_linear_solves: usize,
    pub n_newton_iters: usize,
}

#[derive(Debug, Clone)]
pub struct BdfOptions {
    pub max_order: usize,
    pub max_steps: usize,
    pub max_newton_iters: usize,
    pub h_min: f64,
    pub h_max: f64,
    pub h0: Option<f64>,
    pub newton_tol: f64,
    pub jacobian_reuse_steps: usize,
    pub step_growth_max: f64,
    pub step_shrink_min: f64,
}

impl Default for BdfOptions {
    fn default() -> Self {
        Self {
            max_order: 2,
            max_steps: 100_000,
            max_newton_iters: 6,
            h_min: 1.0e-14,
            h_max: 1.0e6,
            h0: None,
            newton_tol: 1.0e-1,
            jacobian_reuse_steps: 20,
            step_growth_max: 5.0,
            step_shrink_min: 0.5,
        }
    }
}

#[derive(Debug, Clone)]
pub enum TolMode {
    Scalar { rtol: f64, atol: f64 },
    Vector { rtol: Vec<f64>, atol: Vec<f64> },
}

#[derive(Debug, Clone)]
pub struct FdOptions {
    pub rel_step: f64,
    pub abs_step: f64,
}

impl Default for FdOptions {
    fn default() -> Self {
        Self {
            rel_step: f64::EPSILON.sqrt(),
            abs_step: 1.0e-8,
        }
    }
}
//=========================================================================

const MAX_BDF_ORDER: usize = 5;

/// Pascal matrix for Nordsieck predictor:
/// z_j^{pred} = sum_{k=j}^q C(k,j) z_k
const PASCAL: [[f64; MAX_BDF_ORDER + 1]; MAX_BDF_ORDER + 1] = [
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 2.0, 1.0, 0.0, 0.0, 0.0],
    [1.0, 3.0, 3.0, 1.0, 0.0, 0.0],
    [1.0, 4.0, 6.0, 4.0, 1.0, 0.0],
    [1.0, 5.0, 10.0, 10.0, 5.0, 1.0],
];

#[derive(Clone, Copy, Debug)]
pub struct BdfFormula {
    pub q: usize,
    /// beta in y - h*beta*f(t_{n+1}, y) - a = 0
    pub beta: f64,
    /// affine[j] multiplies y_{n+1-j}, j=1..q
    pub affine: [f64; MAX_BDF_ORDER + 1],
}

pub const BDF_FORMULAS: [BdfFormula; MAX_BDF_ORDER + 1] = [
    BdfFormula {
        q: 0,
        beta: 0.0,
        affine: [0.0; MAX_BDF_ORDER + 1],
    },
    BdfFormula {
        q: 1,
        beta: 1.0,
        affine: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    },
    BdfFormula {
        q: 2,
        beta: 2.0 / 3.0,
        affine: [0.0, 4.0 / 3.0, -1.0 / 3.0, 0.0, 0.0, 0.0],
    },
    BdfFormula {
        q: 3,
        beta: 6.0 / 11.0,
        affine: [0.0, 18.0 / 11.0, -9.0 / 11.0, 2.0 / 11.0, 0.0, 0.0],
    },
    BdfFormula {
        q: 4,
        beta: 12.0 / 25.0,
        affine: [
            0.0,
            48.0 / 25.0,
            -36.0 / 25.0,
            16.0 / 25.0,
            -3.0 / 25.0,
            0.0,
        ],
    },
    BdfFormula {
        q: 5,
        beta: 60.0 / 137.0,
        affine: [
            0.0,
            300.0 / 137.0,
            -300.0 / 137.0,
            200.0 / 137.0,
            -75.0 / 137.0,
            12.0 / 137.0,
        ],
    },
];

/// Backward-difference to Nordsieck conversion matrices for orders 1..=5.
///
/// Suppose after accept we have backward differences at t_n:
///   d0 = y_n
///   d1 = ∇y_n
///   d2 = ∇²y_n
///   ...
///   dq = ∇^q y_n
///
/// We want Nordsieck-like columns z_j such that predictor via Pascal works well.
/// For low orders and equal-step startup regime, the following triangular transforms
/// are a practical and stable choice.
///
/// Row j, column k means contribution of d_k to z_j, with j,k starting from 0.
/// Only k >= j is used. Entries beyond current q are ignored.
///
/// z0 = d0
/// z1..zq are approximate scaled derivatives reconstructed from backward diffs.
///
/// This is the A2 history engine: cheap, stable, and good enough before a full
/// LSODE-style coefficient engine.
const DIFF_TO_NORD_1: [[f64; 2]; 2] = [[1.0, 0.0], [0.0, 1.0]];

const DIFF_TO_NORD_2: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, -0.5], [0.0, 0.0, 0.5]];

const DIFF_TO_NORD_3: [[f64; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, -0.5, 1.0 / 3.0],
    [0.0, 0.0, 0.5, -0.5],
    [0.0, 0.0, 0.0, 1.0 / 6.0],
];

const DIFF_TO_NORD_4: [[f64; 5]; 5] = [
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, -0.5, 1.0 / 3.0, -0.25],
    [0.0, 0.0, 0.5, -0.5, 11.0 / 24.0],
    [0.0, 0.0, 0.0, 1.0 / 6.0, -0.25],
    [0.0, 0.0, 0.0, 0.0, 1.0 / 24.0],
];

const DIFF_TO_NORD_5: [[f64; 6]; 6] = [
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, -0.5, 1.0 / 3.0, -0.25, 0.2],
    [0.0, 0.0, 0.5, -0.5, 11.0 / 24.0, -5.0 / 12.0],
    [0.0, 0.0, 0.0, 1.0 / 6.0, -0.25, 7.0 / 24.0],
    [0.0, 0.0, 0.0, 0.0, 1.0 / 24.0, -1.0 / 12.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 120.0],
];
#[derive(Debug, Clone)]
pub struct NordsieckHistory {
    pub n: usize,
    pub qmax: usize,
    /// columns z_0..z_qmax, column-major by block:
    /// data[j * n + i]
    pub data: Vec<f64>,
}

impl NordsieckHistory {
    pub fn new(n: usize, qmax: usize) -> Self {
        Self {
            n,
            qmax,
            data: vec![0.0; (qmax + 1) * n],
        }
    }

    #[inline(always)]
    pub fn col(&self, j: usize) -> &[f64] {
        let start = j * self.n;
        &self.data[start..start + self.n]
    }

    #[inline(always)]
    pub fn col_mut(&mut self, j: usize) -> &mut [f64] {
        let start = j * self.n;
        &mut self.data[start..start + self.n]
    }

    #[inline(always)]
    pub fn fill_zero_from(&mut self, from_col: usize) {
        for j in from_col..=self.qmax {
            self.col_mut(j).fill(0.0);
        }
    }
}

#[derive(Debug, Clone)]
pub struct YHistory {
    pub n: usize,
    pub qmax: usize,
    /// blocks:
    /// 0 -> y_n
    /// 1 -> y_{n-1}
    /// ...
    /// qmax -> y_{n-qmax}
    pub data: Vec<f64>,
}

impl YHistory {
    pub fn new(n: usize, qmax: usize, y0: &[f64]) -> Self {
        let mut data = vec![0.0; (qmax + 1) * n];
        data[..n].copy_from_slice(y0);
        Self { n, qmax, data }
    }

    #[inline(always)]
    pub fn block(&self, k: usize) -> &[f64] {
        let start = k * self.n;
        &self.data[start..start + self.n]
    }

    #[inline(always)]
    pub fn block_mut(&mut self, k: usize) -> &mut [f64] {
        let start = k * self.n;
        &mut self.data[start..start + self.n]
    }

    pub fn push_front(&mut self, y_new: &[f64]) {
        let n = self.n;
        for k in (1..=self.qmax).rev() {
            let dst = k * n;
            let src = (k - 1) * n;
            self.data.copy_within(src..src + n, dst);
        }
        self.data[..n].copy_from_slice(y_new);
    }
}

#[inline]
pub fn predict_nordsieck(zn: &NordsieckHistory, zpred: &mut NordsieckHistory, q: usize) {
    let n = zn.n;
    zpred.data.fill(0.0);

    for j in 0..=q {
        for k in j..=q {
            let coeff = PASCAL[k][j];
            let zk = zn.col(k);
            let out = zpred.col_mut(j);
            for i in 0..n {
                out[i] += coeff * zk[i];
            }
        }
    }
}

#[inline]
pub fn build_affine_part(yh: &YHistory, q: usize, out: &mut [f64]) {
    let n = yh.n;
    let formula = &BDF_FORMULAS[q];
    out.fill(0.0);

    for j in 1..=q {
        let coeff = formula.affine[j];
        let yj = yh.block(j - 1);
        for i in 0..n {
            out[i] += coeff * yj[i];
        }
    }
}

#[inline]
pub fn build_bdf_residual(
    y_trial: &[f64],
    f_trial: &[f64],
    affine: &[f64],
    h: f64,
    q: usize,
    residual: &mut [f64],
) {
    let beta = BDF_FORMULAS[q].beta;
    let hb = h * beta;
    for i in 0..y_trial.len() {
        residual[i] = y_trial[i] - hb * f_trial[i] - affine[i];
    }
}

#[inline]
pub fn build_backward_differences(yh: &YHistory, q: usize, out: &mut [f64]) {
    let n = yh.n;
    debug_assert_eq!(out.len(), (q + 1) * n);

    // d0 = y_n
    out[..n].copy_from_slice(yh.block(0));

    if q == 0 {
        return;
    }

    // workspace recurrence:
    // level 1 starts as y_n, y_{n-1}, ..., y_{n-q}
    // then we overwrite progressively into out blocks.
    let mut work = vec![0.0; (q + 1) * n];
    for k in 0..=q {
        let dst = k * n;
        work[dst..dst + n].copy_from_slice(yh.block(k));
    }

    // Build ∇^j y_n into out[j]
    for j in 1..=q {
        for k in 0..=(q - j) {
            let a_start = k * n;
            let b_start = (k + 1) * n;
            for i in 0..n {
                work[a_start + i] -= work[b_start + i];
            }
        }

        let dst = j * n;
        out[dst..dst + n].copy_from_slice(&work[..n]);
    }
}

#[inline]
pub fn build_backward_differences_in_place(
    yh: &YHistory,
    q: usize,
    work: &mut [f64],
    out: &mut [f64],
) {
    let n = yh.n;
    debug_assert_eq!(work.len(), (yh.qmax + 1) * n);
    debug_assert_eq!(out.len(), (yh.qmax + 1) * n);

    // Initialize work with y_n, y_{n-1}, ..., y_{n-q}
    for k in 0..=q {
        let start = k * n;
        work[start..start + n].copy_from_slice(yh.block(k));
    }

    // Zero out result area first
    out[..(q + 1) * n].fill(0.0);

    // d0 = y_n
    out[..n].copy_from_slice(yh.block(0));

    // Higher backward differences
    for j in 1..=q {
        for k in 0..=(q - j) {
            let a_start = k * n;
            let b_start = (k + 1) * n;
            for i in 0..n {
                work[a_start + i] -= work[b_start + i];
            }
        }

        let dst = j * n;
        out[dst..dst + n].copy_from_slice(&work[..n]);
    }
}

#[inline]
pub fn backward_differences_to_nordsieck(q: usize, n: usize, diffs: &[f64], zn_data: &mut [f64]) {
    debug_assert!(q <= MAX_BDF_ORDER);
    debug_assert!(diffs.len() >= (q + 1) * n);
    debug_assert!(zn_data.len() >= (q + 1) * n);

    // Zero only the active prefix
    zn_data[..(q + 1) * n].fill(0.0);

    match q {
        0 => {
            zn_data[..n].copy_from_slice(&diffs[..n]);
        }
        1 => {
            for j in 0..=1 {
                for k in j..=1 {
                    let coeff = DIFF_TO_NORD_1[j][k];
                    let dj = &diffs[k * n..(k + 1) * n];
                    let zj = &mut zn_data[j * n..(j + 1) * n];
                    for i in 0..n {
                        zj[i] += coeff * dj[i];
                    }
                }
            }
        }
        2 => {
            for j in 0..=2 {
                for k in j..=2 {
                    let coeff = DIFF_TO_NORD_2[j][k];
                    let dk = &diffs[k * n..(k + 1) * n];
                    let zj = &mut zn_data[j * n..(j + 1) * n];
                    for i in 0..n {
                        zj[i] += coeff * dk[i];
                    }
                }
            }
        }
        3 => {
            for j in 0..=3 {
                for k in j..=3 {
                    let coeff = DIFF_TO_NORD_3[j][k];
                    let dk = &diffs[k * n..(k + 1) * n];
                    let zj = &mut zn_data[j * n..(j + 1) * n];
                    for i in 0..n {
                        zj[i] += coeff * dk[i];
                    }
                }
            }
        }
        4 => {
            for j in 0..=4 {
                for k in j..=4 {
                    let coeff = DIFF_TO_NORD_4[j][k];
                    let dk = &diffs[k * n..(k + 1) * n];
                    let zj = &mut zn_data[j * n..(j + 1) * n];
                    for i in 0..n {
                        zj[i] += coeff * dk[i];
                    }
                }
            }
        }
        5 => {
            for j in 0..=5 {
                for k in j..=5 {
                    let coeff = DIFF_TO_NORD_5[j][k];
                    let dk = &diffs[k * n..(k + 1) * n];
                    let zj = &mut zn_data[j * n..(j + 1) * n];
                    for i in 0..n {
                        zj[i] += coeff * dk[i];
                    }
                }
            }
        }
        _ => unreachable!("q > MAX_BDF_ORDER"),
    }
}
/// =========================
/// RHS callback
/// =========================

pub trait OdeRhs {
    fn eval(&mut self, t: f64, y: &[f64], fy: &mut [f64]);
}

/// =========================
/// Linear backend
/// =========================

pub trait LinearBackend {
    type Storage;

    fn storage(&self) -> &Self::Storage;
    fn storage_mut(&mut self) -> &mut Self::Storage;

    /// Build system matrix P = I - gamma * J
    fn assemble_system_matrix(&mut self, gamma: f64);

    /// Factorize system matrix in-place
    fn factorize(&mut self) -> Result<(), OdeError>;

    /// Solve P x = rhs, overwrite rhs with x
    fn solve_in_place(&mut self, rhs: &mut [f64]) -> Result<(), OdeError>;
}

/// =========================
/// BDF core state
/// phi layout: column-major by history block
/// phi[k * n + i]
/// =========================
pub struct BdfCore {
    pub n: usize,
    pub t: f64,
    pub h: f64,
    pub order: usize,
    pub max_order: usize,

    pub y: Vec<f64>,
    pub y_pred: Vec<f64>,
    pub y_new: Vec<f64>,

    pub f: Vec<f64>,
    pub f_new: Vec<f64>,

    pub ewt: Vec<f64>,
    pub acor: Vec<f64>,
    pub delta: Vec<f64>,
    pub residual: Vec<f64>,
    pub affine: Vec<f64>,
    pub tmp: Vec<f64>,

    pub diff_work: Vec<f64>,
    /// Current accepted Nordsieck history at t_n
    pub zn: NordsieckHistory,
    /// Predicted Nordsieck history at t_{n+1}
    pub zpred: NordsieckHistory,
    /// Raw y-history for BDF affine combination:
    /// [y_n, y_{n-1}, ..., y_{n-qmax}]
    pub yhist: YHistory,

    /// gamma = h * beta_q, used in P = I - gamma * J
    pub gamma: f64,

    pub jac_current: bool,
    pub steps_since_jac: usize,
    /// Number of accepted steps after initialization.
    pub n_accept_total: usize,

    /// Maximum order currently allowed by available history.
    pub order_cap: usize,

    /// Consecutive accepted steps since the last order change.
    pub n_accept_since_order_change: usize,
    pub last_correction_norm: f64,
    /// Consecutive rejected steps.
    pub n_reject_consecutive: usize,
    pub stats: SolverStats,
}

impl BdfCore {
    pub fn new(n: usize, t0: f64, y0: Vec<f64>, h0: f64, max_order: usize) -> Self {
        let zn = NordsieckHistory::new(n, max_order);
        let zpred = NordsieckHistory::new(n, max_order);
        let yhist = YHistory::new(n, max_order, &y0);

        Self {
            n,
            t: t0,
            h: h0,
            order: 1,
            max_order,

            y: y0.clone(),
            y_pred: y0.clone(),
            y_new: y0.clone(),

            f: vec![0.0; n],
            f_new: vec![0.0; n],

            ewt: vec![1.0; n],
            acor: vec![0.0; n],
            delta: vec![0.0; n],
            residual: vec![0.0; n],
            affine: vec![0.0; n],
            tmp: vec![0.0; n],

            zn,
            zpred,
            yhist,

            gamma: h0,
            diff_work: vec![0.0; (max_order + 1) * n],
            jac_current: false,
            steps_since_jac: 0,
            n_accept_total: 0,
            order_cap: 1,
            n_accept_since_order_change: 0,
            last_correction_norm: 0.0,
            n_reject_consecutive: 0,
            stats: SolverStats::default(),
        }
    }
}
