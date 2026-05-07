// src/ode/Rodas4_faer.rs
// src/ode/rodas4_faer.rs

use faer::Mat;
use faer::prelude::*;
use faer::sparse::{SparseColMat, Triplet};

const S: usize = 6;

pub type DfdtFn = fn(f64, &[f64], &mut [f64]);

#[derive(Clone)]
struct Rodas4Table {
    gamma: f64,
    gamma_i: [f64; S],
    alpha: [f64; S],
    new_f: [bool; S],
    a: [[f64; S]; S],
    c: [[f64; S]; S],
    m: [f64; S],
    e: [f64; S],
}

impl Rodas4Table {
    fn new() -> Self {
        let mut a = [[0.0; S]; S];
        let mut c = [[0.0; S]; S];

        a[1][0] = 1.5440000000000000;

        a[2][0] = 0.9466785280815826;
        a[2][1] = 0.2557011698983284;

        a[3][0] = 3.314825187068521;
        a[3][1] = 2.896124015972201;
        a[3][2] = 0.9986419139977817;

        a[4][0] = 1.221224509226641;
        a[4][1] = 6.019134481288629;
        a[4][2] = 12.53708332932087;
        a[4][3] = -0.6878860361058950;

        a[5][0] = a[4][0];
        a[5][1] = a[4][1];
        a[5][2] = a[4][2];
        a[5][3] = a[4][3];
        a[5][4] = 1.0;

        c[1][0] = -5.668800000000000;

        c[2][0] = -2.430093356833875;
        c[2][1] = -0.2063599157091915;

        c[3][0] = -0.1073529058151375;
        c[3][1] = -9.594562251023355;
        c[3][2] = -20.47028614809616;

        c[4][0] = 7.496443313967647;
        c[4][1] = -10.24680431464352;
        c[4][2] = -33.99990352819905;
        c[4][3] = 11.70890893206160;

        c[5][0] = 8.083246795921522;
        c[5][1] = -7.981132988064893;
        c[5][2] = -31.52159432874371;
        c[5][3] = 16.31930543123136;
        c[5][4] = -6.058818238834054;

        Self {
            gamma: 0.25,
            gamma_i: [
                0.25,
                -0.1043000000000000,
                0.1035000000000000,
                -0.03620000000000023,
                0.0,
                0.0,
            ],
            alpha: [0.0, 0.386, 0.210, 0.630, 1.0, 1.0],
            new_f: [true, true, true, true, true, true],
            a,
            c,
            m: [
                1.221224509226641,
                6.019134481288629,
                12.53708332932087,
                -0.6878860361058950,
                1.0,
                1.0,
            ],
            e: [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }
    }
}

#[derive(Debug, Clone)]
pub struct Rodas4Config {
    pub rtol: f64,
    pub atol: f64,
    pub h_min: f64,
    pub h_max: f64,
    pub safety: f64,
    pub max_steps: usize,
    pub max_rejects: usize,
}

impl Default for Rodas4Config {
    fn default() -> Self {
        Self {
            rtol: 1e-6,
            atol: 1e-9,
            h_min: 1e-14,
            h_max: 1.0,
            safety: 0.9,
            max_steps: 100_000,
            max_rejects: 100,
        }
    }
}

struct LinState {
    // Jacobian в виде triplets (переиспользуется)
    jac_triplets: Vec<Triplet<usize, usize, f64>>,

    // triplets матрицы A = I - hγJ (фиксированный pattern)
    triplets: Vec<Triplet<usize, usize, f64>>,

    // позиции диагонали в triplets
    diag_pos: Vec<usize>,

    // LU факторизация
    lu: Option<faer::sparse::linalg::solvers::Lu<usize, f64>>,

    // служебное
    h_last: f64,
    steps_since_rebuild: usize,
    need_rebuild: bool,
    pattern_initialized: bool,
}

#[derive(Debug, Clone)]
pub struct Rodas4Stats {
    pub accepted_steps: usize,
    pub rejected_steps: usize,
    pub rhs_evals: usize,
    pub jacobian_evals: usize,
    pub dfdt_evals: usize,
    pub lu_factorizations: usize,
    pub linear_solves: usize,
}

impl Rodas4Stats {
    fn new() -> Self {
        Self {
            accepted_steps: 0,
            rejected_steps: 0,
            rhs_evals: 0,
            jacobian_evals: 0,
            dfdt_evals: 0,
            lu_factorizations: 0,
            linear_solves: 0,
        }
    }
}

#[derive(Debug)]
pub enum Rodas4Error {
    DimensionMismatch,
    StepUnderflow,
    TooManySteps,
    TooManyRejects,
    SingularMatrix,
    InvalidNumber,
}

pub struct Rodas4Solver<F, J, T = DfdtFn>
where
    F: Fn(f64, &[f64], &mut [f64]),
    J: Fn(f64, &[f64], &mut Vec<Triplet<usize, usize, f64>>),
    T: Fn(f64, &[f64], &mut [f64]),
{
    f: F,
    jac: J,
    dfdt: T,
    has_dfdt: bool,
    cfg: Rodas4Config,
    stats: Rodas4Stats,
    lin: LinState,
}

impl<F, J> Rodas4Solver<F, J, DfdtFn>
where
    F: Fn(f64, &[f64], &mut [f64]),
    J: Fn(f64, &[f64], &mut Vec<Triplet<usize, usize, f64>>),
{
    pub fn new(f: F, jac: J, cfg: Rodas4Config) -> Self {
        fn zero_dfdt(_: f64, _: &[f64], out: &mut [f64]) {
            out.fill(0.0);
        }

        Self {
            f,
            jac,
            dfdt: zero_dfdt,
            has_dfdt: false,
            cfg,
            stats: Rodas4Stats::new(),
            lin: LinState {
                jac_triplets: Vec::new(),
                triplets: Vec::new(),
                diag_pos: Vec::new(),
                lu: None,
                h_last: 0.0,
                steps_since_rebuild: 0,
                need_rebuild: true,
                pattern_initialized: false,
            },
        }
    }
}

impl<F, J, T> Rodas4Solver<F, J, T>
where
    F: Fn(f64, &[f64], &mut [f64]),
    J: Fn(f64, &[f64], &mut Vec<Triplet<usize, usize, f64>>),
    T: Fn(f64, &[f64], &mut [f64]),
{
    pub fn new_non_autonomous(f: F, jac: J, dfdt: T, cfg: Rodas4Config) -> Self {
        Self {
            f,
            jac,
            dfdt,
            has_dfdt: true,
            cfg,
            stats: Rodas4Stats::new(),
            lin: LinState {
                jac_triplets: Vec::new(),
                triplets: Vec::new(),
                diag_pos: Vec::new(),
                lu: None,
                h_last: 0.0,
                steps_since_rebuild: 0,
                need_rebuild: true,
                pattern_initialized: false,
            },
        }
    }

    pub fn stats(&self) -> &Rodas4Stats {
        &self.stats
    }

    fn should_rebuild(&self, h: f64) -> bool {
        if self.lin.need_rebuild {
            return true;
        }

        // изменение шага
        if self.lin.h_last != 0.0 {
            let ratio = (h / self.lin.h_last).abs();
            if (ratio - 1.0).abs() > 0.3 {
                return true;
            }
        }

        // периодическая пересборка
        if self.lin.steps_since_rebuild >= 8 {
            return true;
        }

        false
    }
    fn init_pattern(&mut self, n: usize) {
        self.lin.triplets.clear();
        self.lin.diag_pos = vec![0; n];

        // сначала Jacobian pattern
        for tr in &self.lin.jac_triplets {
            self.lin.triplets.push(Triplet::new(tr.row, tr.col, 0.0));
        }

        // затем диагональ
        for i in 0..n {
            self.lin.diag_pos[i] = self.lin.triplets.len();
            self.lin.triplets.push(Triplet::new(i, i, 0.0));
        }

        self.lin.pattern_initialized = true;
    }

    fn update_matrix_values(&mut self, h: f64, gamma: f64) {
        // Jacobian часть
        for (k, tr) in self.lin.jac_triplets.iter().enumerate() {
            self.lin.triplets[k].val = -h * gamma * tr.val;
        }

        // диагональ = 1
        for &pos in &self.lin.diag_pos {
            self.lin.triplets[pos].val = 1.0;
        }
    }

    pub fn integrate(
        &mut self,
        t0: f64,
        y0: &[f64],
        t_end: f64,
        h0: f64,
    ) -> Result<(f64, Vec<f64>), Rodas4Error> {
        let n = y0.len();
        if n == 0 {
            return Err(Rodas4Error::DimensionMismatch);
        }

        let direction = if t_end >= t0 { 1.0 } else { -1.0 };
        let mut t = t0;
        let mut y = y0.to_vec();

        let mut h = h0.abs().min(self.cfg.h_max).max(self.cfg.h_min) * direction;
        let mut rejects_in_row = 0usize;

        for _ in 0..self.cfg.max_steps {
            if direction * (t_end - t) <= 0.0 {
                return Ok((t, y));
            }

            if direction * (t + h - t_end) > 0.0 {
                h = t_end - t;
            }

            let step = self.try_step(t, &y, h)?;

            if step.err_norm <= 1.0 {
                t += h;
                y = step.y_new;

                self.stats.accepted_steps += 1;
                rejects_in_row = 0;

                let factor = step_size_factor(step.err_norm, 3.0, self.cfg.safety);
                h *= factor;
                h = clamp_abs(h, self.cfg.h_min, self.cfg.h_max);
            } else {
                self.stats.rejected_steps += 1;
                rejects_in_row += 1;
                self.lin.need_rebuild = true;

                if rejects_in_row > self.cfg.max_rejects {
                    return Err(Rodas4Error::TooManyRejects);
                }

                let factor = step_size_factor(step.err_norm, 3.0, self.cfg.safety)
                    .min(0.5)
                    .max(0.1);

                h *= factor;

                if h.abs() < self.cfg.h_min {
                    return Err(Rodas4Error::StepUnderflow);
                }
            }
        }

        Err(Rodas4Error::TooManySteps)
    }

    fn try_step(&mut self, t: f64, y: &[f64], h: f64) -> Result<Rodas4Step, Rodas4Error> {
        let n = y.len();
        let tab = Rodas4Table::new();

        let mut f0 = vec![0.0; n];
        (self.f)(t, y, &mut f0);
        self.stats.rhs_evals += 1;
        check_finite_slice(&f0)?;

        let mut dfdt0: Option<Vec<f64>> = None;

        let need_dfdt = self.has_dfdt && tab.gamma_i.iter().any(|&g| g != 0.0);

        if need_dfdt {
            let mut tmp = vec![0.0; n];

            (self.dfdt)(t, y, &mut tmp);
            self.stats.dfdt_evals += 1;
            check_finite_slice(&tmp)?;

            dfdt0 = Some(tmp);
        }

        let rebuild = self.should_rebuild(h);

        if rebuild {
            // --- Jacobian ---
            self.lin.jac_triplets.clear();
            (self.jac)(t, y, &mut self.lin.jac_triplets);
            self.stats.jacobian_evals += 1;

            // --- init pattern (1 раз) ---
            if !self.lin.pattern_initialized
                || self.lin.triplets.len() != self.lin.jac_triplets.len() + n
            {
                self.init_pattern(n);
            }

            // --- обновление значений ---
            self.update_matrix_values(h, tab.gamma);

            // --- сборка матрицы ---
            let mat = SparseColMat::try_new_from_triplets(n, n, &self.lin.triplets)
                .map_err(|_| Rodas4Error::SingularMatrix)?;

            // --- LU ---
            let lu = mat.sp_lu().map_err(|_| Rodas4Error::SingularMatrix)?;
            self.stats.lu_factorizations += 1;

            self.lin.lu = Some(lu);
            self.lin.h_last = h;
            self.lin.steps_since_rebuild = 0;
            self.lin.need_rebuild = false;
        } else {
            self.lin.steps_since_rebuild += 1;
        }

        let mut k: Vec<Vec<f64>> = vec![vec![0.0; n]; S];
        let mut f_stage = f0.clone();
        for istage in 0..S {
            if istage == 0 {
                f_stage.copy_from_slice(&f0);
            } else if tab.new_f[istage] {
                let mut y_stage = y.to_vec();

                for j in 0..istage {
                    let aij = tab.a[istage][j];

                    if aij != 0.0 {
                        for i in 0..n {
                            y_stage[i] += aij * k[j][i];
                        }
                    }
                }

                let tau = t + tab.alpha[istage] * h;

                (self.f)(tau, &y_stage, &mut f_stage);
                self.stats.rhs_evals += 1;
                check_finite_slice(&f_stage)?;
            }

            let mut rhs = vec![0.0; n];

            for i in 0..n {
                rhs[i] = h * tab.gamma * f_stage[i];

                if tab.gamma_i[istage] != 0.0 {
                    if let Some(dfdt0) = &dfdt0 {
                        rhs[i] += h * h * tab.gamma * tab.gamma_i[istage] * dfdt0[i];
                    }
                }
            }

            for j in 0..istage {
                let cij = tab.c[istage][j];

                if cij != 0.0 {
                    for i in 0..n {
                        rhs[i] += tab.gamma * cij * k[j][i];
                    }
                }
            }

            let rhs_mat = col_from_slice(&rhs);
            let lu = self.lin.lu.as_ref().ok_or(Rodas4Error::SingularMatrix)?;
            let ki_mat = lu.solve(&rhs_mat);
            self.stats.linear_solves += 1;

            k[istage] = mat_col_to_vec(&ki_mat);
            check_finite_slice(&k[istage])?;
        }
        let mut y_new = y.to_vec();
        let mut y_err = vec![0.0; n];

        for s in 0..S {
            for i in 0..n {
                y_new[i] += tab.m[s] * k[s][i];
                y_err[i] += tab.e[s] * k[s][i];
            }
        }

        let err_norm =
            weighted_error_norm_from_error_vector(y, &y_new, &y_err, self.cfg.atol, self.cfg.rtol)?;

        Ok(Rodas4Step { y_new, err_norm })
    }
}

struct Rodas4Step {
    y_new: Vec<f64>,
    err_norm: f64,
}

fn build_ros_matrix(
    n: usize,
    gamma: f64,
    h: f64,
    jac_triplets: &[Triplet<usize, usize, f64>],
) -> Result<SparseColMat<usize, f64>, Rodas4Error> {
    let mut triplets = Vec::with_capacity(jac_triplets.len() + n);

    for tr in jac_triplets {
        triplets.push(Triplet::new(tr.row, tr.col, -h * gamma * tr.val));
    }

    for i in 0..n {
        triplets.push(Triplet::new(i, i, 1.0));
    }

    SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets)
        .map_err(|_| Rodas4Error::DimensionMismatch)
}

fn col_from_slice(x: &[f64]) -> Mat<f64> {
    Mat::from_fn(x.len(), 1, |i, _| x[i])
}

fn mat_col_to_vec(x: &Mat<f64>) -> Vec<f64> {
    let n = x.nrows();
    let mut out = vec![0.0; n];

    for i in 0..n {
        out[i] = x[(i, 0)];
    }

    out
}

fn weighted_error_norm_from_error_vector(
    y_old: &[f64],
    y_new: &[f64],
    y_err: &[f64],
    atol: f64,
    rtol: f64,
) -> Result<f64, Rodas4Error> {
    let n = y_old.len();
    let mut sum = 0.0;

    for i in 0..n {
        let scale = atol + rtol * y_old[i].abs().max(y_new[i].abs());
        let e = y_err[i] / scale;
        sum += e * e;
    }

    let norm = (sum / n as f64).sqrt();

    if norm.is_finite() {
        Ok(norm)
    } else {
        Err(Rodas4Error::InvalidNumber)
    }
}

fn step_size_factor(err_norm: f64, order: f64, safety: f64) -> f64 {
    if err_norm == 0.0 {
        return 5.0;
    }

    let exponent = -1.0 / (order + 1.0);
    (safety * err_norm.powf(exponent)).clamp(0.1, 5.0)
}

fn clamp_abs(h: f64, h_min: f64, h_max: f64) -> f64 {
    let sign = h.signum();
    sign * h.abs().clamp(h_min, h_max)
}

fn check_finite_slice(x: &[f64]) -> Result<(), Rodas4Error> {
    if x.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(Rodas4Error::InvalidNumber)
    }
}
fn jacobian_times_vec(
    n: usize,
    jac_triplets: &[Triplet<usize, usize, f64>],
    x: &[f64],
) -> Result<Vec<f64>, Rodas4Error> {
    if x.len() != n {
        return Err(Rodas4Error::DimensionMismatch);
    }

    let mut out = vec![0.0; n];

    for tr in jac_triplets {
        if tr.row >= n || tr.col >= n {
            return Err(Rodas4Error::DimensionMismatch);
        }

        out[tr.row] += tr.val * x[tr.col];
    }

    Ok(out)
}
#[cfg(test)]
mod tests2 {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        let err = (actual - expected).abs();

        assert!(
            err <= tol,
            "actual = {actual:e}, expected = {expected:e}, err = {err:e}, tol = {tol:e}"
        );
    }

    fn assert_vec_close(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(actual.len(), expected.len());

        for i in 0..actual.len() {
            assert_close(actual[i], expected[i], tol);
        }
    }

    fn integrate_fixed<F, J, T>(
        solver: &mut Rodas4Solver<F, J, T>,
        t0: f64,
        y0: &[f64],
        t_end: f64,
        h: f64,
    ) -> Vec<f64>
    where
        F: Fn(f64, &[f64], &mut [f64]),
        J: Fn(f64, &[f64], &mut Vec<Triplet<usize, usize, f64>>),
        T: Fn(f64, &[f64], &mut [f64]),
    {
        let mut t = t0;
        let mut y = y0.to_vec();

        while t < t_end - 1e-15 {
            let h_step = (t_end - t).min(h);
            let step = solver.try_step(t, &y, h_step).unwrap();

            y = step.y_new;
            t += h_step;
        }

        y
    }

    #[test]
    fn scalar_stiff_decay_matches_exact_solution() {
        let lambda = 1000.0;

        let f = move |_: f64, y: &[f64], out: &mut [f64]| {
            out[0] = -lambda * y[0];
        };

        let jac = move |_: f64, _: &[f64], triplets: &mut Vec<Triplet<usize, usize, f64>>| {
            triplets.clear();
            triplets.push(Triplet::new(0, 0, -lambda));
        };

        let cfg = Rodas4Config {
            rtol: 1e-8,
            atol: 1e-12,
            h_min: 1e-16,
            h_max: 1e-2,
            max_steps: 1_000_000,
            ..Default::default()
        };

        let mut solver = Rodas4Solver::new(f, jac, cfg);

        let y0 = vec![1.0];
        let t_end = 0.01;

        let (_, y) = solver.integrate(0.0, &y0, t_end, 1e-6).unwrap();

        let exact = (-lambda * t_end).exp();

        assert_close(y[0], exact, 1e-8);
    }

    #[test]
    fn diagonal_stiff_linear_system_matches_exact_solution() {
        let lambdas = [1.0, 100.0, 1000.0];

        let f = move |_: f64, y: &[f64], out: &mut [f64]| {
            for i in 0..3 {
                out[i] = -lambdas[i] * y[i];
            }
        };

        let jac = move |_: f64, _: &[f64], triplets: &mut Vec<Triplet<usize, usize, f64>>| {
            triplets.clear();

            for i in 0..3 {
                triplets.push(Triplet::new(i, i, -lambdas[i]));
            }
        };

        let cfg = Rodas4Config {
            rtol: 1e-9,
            atol: 1e-13,
            h_min: 1e-16,
            h_max: 1e-2,
            max_steps: 1_000_000,
            ..Default::default()
        };

        let mut solver = Rodas4Solver::new(f, jac, cfg);

        let y0 = vec![1.0, 1.0, 1.0];
        let t_end = 0.01;

        let (_, y) = solver.integrate(0.0, &y0, t_end, 1e-6).unwrap();

        let exact = vec![
            (-lambdas[0] * t_end).exp(),
            (-lambdas[1] * t_end).exp(),
            (-lambdas[2] * t_end).exp(),
        ];

        assert_vec_close(&y, &exact, 1e-8);
    }

    #[test]
    fn robertson_matches_reference_at_t_1_tighter() {
        let f = |_: f64, y: &[f64], out: &mut [f64]| {
            out[0] = -0.04 * y[0] + 1.0e4 * y[1] * y[2];
            out[1] = 0.04 * y[0] - 1.0e4 * y[1] * y[2] - 3.0e7 * y[1] * y[1];
            out[2] = 3.0e7 * y[1] * y[1];
        };

        let jac = |_: f64, y: &[f64], triplets: &mut Vec<Triplet<usize, usize, f64>>| {
            triplets.clear();

            triplets.push(Triplet::new(0, 0, -0.04));
            triplets.push(Triplet::new(0, 1, 1.0e4 * y[2]));
            triplets.push(Triplet::new(0, 2, 1.0e4 * y[1]));

            triplets.push(Triplet::new(1, 0, 0.04));
            triplets.push(Triplet::new(1, 1, -1.0e4 * y[2] - 6.0e7 * y[1]));
            triplets.push(Triplet::new(1, 2, -1.0e4 * y[1]));

            triplets.push(Triplet::new(2, 1, 6.0e7 * y[1]));
        };

        let cfg = Rodas4Config {
            rtol: 1e-10,
            atol: 1e-14,
            h_min: 1e-16,
            h_max: 1e-2,
            max_steps: 1_000_000,
            ..Default::default()
        };

        let mut solver = Rodas4Solver::new(f, jac, cfg);

        let y0 = vec![1.0, 0.0, 0.0];
        let (_, y) = solver.integrate(0.0, &y0, 1.0, 1e-7).unwrap();

        let reference = vec![9.66459737e-1, 3.07462658e-5, 3.35095160e-2];

        assert_vec_close(&y, &reference, 1e-7);

        let mass = y.iter().sum::<f64>();
        assert_close(mass, 1.0, 1e-10);

        for (i, yi) in y.iter().enumerate() {
            assert!(*yi >= -1e-13, "negative concentration at index {i}: {yi:e}");
        }
    }

    #[test]
    fn fixed_step_observed_order_is_close_to_four_on_autonomous_problem() {
        let make_solver = || {
            let f = |_: f64, y: &[f64], out: &mut [f64]| {
                out[0] = -2.0 * y[0];
            };

            let jac = |_: f64, _: &[f64], triplets: &mut Vec<Triplet<usize, usize, f64>>| {
                triplets.clear();
                triplets.push(Triplet::new(0, 0, -2.0));
            };

            let cfg = Rodas4Config {
                rtol: 1e-13,
                atol: 1e-15,
                h_min: 1e-16,
                h_max: 1.0,
                max_steps: 1_000_000,
                ..Default::default()
            };

            Rodas4Solver::new(f, jac, cfg)
        };

        let y0 = vec![1.0];
        let t_end = 1.0;
        let exact = (-2.0_f64 * t_end).exp();

        let mut solver_h = make_solver();
        let y_h = integrate_fixed(&mut solver_h, 0.0, &y0, t_end, 1.0 / 5.0);
        let err_h = (y_h[0] - exact).abs();

        let mut solver_h2 = make_solver();
        let y_h2 = integrate_fixed(&mut solver_h2, 0.0, &y0, t_end, 1.0 / 10.0);
        let err_h2 = (y_h2[0] - exact).abs();

        let mut solver_h4 = make_solver();
        let y_h4 = integrate_fixed(&mut solver_h4, 0.0, &y0, t_end, 1.0 / 20.0);
        let err_h4 = (y_h4[0] - exact).abs();

        let order_1 = (err_h / err_h2).log2();
        let order_2 = (err_h2 / err_h4).log2();

        println!("autonomous err_h  = {err_h:e}");
        println!("autonomous err_h2 = {err_h2:e}");
        println!("autonomous err_h4 = {err_h4:e}");
        println!("autonomous observed order: {order_1:.4}, {order_2:.4}");

        assert!(
            order_1 > 3.65 && order_2 > 3.65,
            "observed autonomous order too low: {order_1:.4}, {order_2:.4}"
        );
    }

    #[test]
    fn non_autonomous_linear_problem_matches_exact_solution() {
        let lambda = 50.0;

        let exact = move |t: f64, y0: f64| {
            let particular_0 = -1.0 / (lambda * lambda + 1.0);
            let c = y0 - particular_0;

            c * (-lambda * t).exp() + (lambda * t.sin() - t.cos()) / (lambda * lambda + 1.0)
        };

        let f = move |t: f64, y: &[f64], out: &mut [f64]| {
            out[0] = -lambda * y[0] + t.sin();
        };

        let jac = move |_: f64, _: &[f64], triplets: &mut Vec<Triplet<usize, usize, f64>>| {
            triplets.clear();
            triplets.push(Triplet::new(0, 0, -lambda));
        };

        let dfdt = move |t: f64, _: &[f64], out: &mut [f64]| {
            out[0] = t.cos();
        };

        let cfg = Rodas4Config {
            rtol: 1e-10,
            atol: 1e-13,
            h_min: 1e-16,
            h_max: 1e-2,
            max_steps: 1_000_000,
            ..Default::default()
        };

        let mut solver = Rodas4Solver::new_non_autonomous(f, jac, dfdt, cfg);

        let y0 = vec![0.3];
        let t_end = 1.0;

        let (_, y) = solver.integrate(0.0, &y0, t_end, 1e-5).unwrap();
        let expected = exact(t_end, y0[0]);

        assert_close(y[0], expected, 1e-9);
        assert!(
            solver.stats().dfdt_evals >= solver.stats().jacobian_evals,
            "dfdt should be evaluated at least as often as jacobian"
        );
    }

    #[test]
    fn tighter_tolerance_reduces_error_on_non_autonomous_problem() {
        let run = |rtol: f64| {
            let lambda = 10.0;

            let exact = move |t: f64, y0: f64| {
                let particular_0 = -1.0 / (lambda * lambda + 1.0);
                let c = y0 - particular_0;

                c * (-lambda * t).exp() + (lambda * t.sin() - t.cos()) / (lambda * lambda + 1.0)
            };

            let f = move |t: f64, y: &[f64], out: &mut [f64]| {
                out[0] = -lambda * y[0] + t.sin();
            };

            let jac = move |_: f64, _: &[f64], triplets: &mut Vec<Triplet<usize, usize, f64>>| {
                triplets.clear();
                triplets.push(Triplet::new(0, 0, -lambda));
            };

            let dfdt = move |t: f64, _: &[f64], out: &mut [f64]| {
                out[0] = t.cos();
            };

            let cfg = Rodas4Config {
                rtol,
                atol: 1e-13,
                h_min: 1e-16,
                h_max: 0.1,
                max_steps: 1_000_000,
                ..Default::default()
            };

            let mut solver = Rodas4Solver::new_non_autonomous(f, jac, dfdt, cfg);

            let y0 = vec![0.3];
            let t_end = 1.0;

            let (_, y) = solver.integrate(0.0, &y0, t_end, 1e-5).unwrap();
            let expected = exact(t_end, y0[0]);

            (y[0] - expected).abs()
        };

        let err_loose = run(1e-5);
        let err_tight = run(1e-9);

        println!("err_loose = {err_loose:e}");
        println!("err_tight = {err_tight:e}");

        assert!(
            err_tight < err_loose,
            "tighter tolerance did not reduce error: loose={err_loose:e}, tight={err_tight:e}"
        );
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn main_test() {
        // Robertson kinetics:
        //
        // y1' = -0.04 y1 + 1e4 y2 y3
        // y2' =  0.04 y1 - 1e4 y2 y3 - 3e7 y2^2
        // y3' =  3e7 y2^2

        let f = |_: f64, y: &[f64], out: &mut [f64]| {
            out[0] = -0.04 * y[0] + 1.0e4 * y[1] * y[2];
            out[1] = 0.04 * y[0] - 1.0e4 * y[1] * y[2] - 3.0e7 * y[1] * y[1];
            out[2] = 3.0e7 * y[1] * y[1];
        };

        let jac = |_: f64, y: &[f64], triplets: &mut Vec<Triplet<usize, usize, f64>>| {
            triplets.clear();

            triplets.push(Triplet::new(0, 0, -0.04));
            triplets.push(Triplet::new(0, 1, 1.0e4 * y[2]));
            triplets.push(Triplet::new(0, 2, 1.0e4 * y[1]));

            triplets.push(Triplet::new(1, 0, 0.04));
            triplets.push(Triplet::new(1, 1, -1.0e4 * y[2] - 6.0e7 * y[1]));
            triplets.push(Triplet::new(1, 2, -1.0e4 * y[1]));

            triplets.push(Triplet::new(2, 1, 6.0e7 * y[1]));
        };

        let cfg = Rodas4Config {
            rtol: 1e-6,
            atol: 1e-12,
            h_min: 1e-16,
            h_max: 1.0,
            ..Default::default()
        };

        let mut solver = Rodas4Solver::new(f, jac, cfg);

        let y0 = vec![1.0, 0.0, 0.0];
        let (_, y) = solver.integrate(0.0, &y0, 1.0, 1e-6).unwrap();

        println!("y = {y:?}");
        println!("stats = {:?}", solver.stats());
    }
}

#[cfg(test)]
mod correctness_tests {
    use super::*;
    use faer::sparse::Triplet;

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        let err = (actual - expected).abs();
        assert!(
            err <= tol,
            "actual = {actual:e}, expected = {expected:e}, err = {err:e}, tol = {tol:e}"
        );
    }

    fn assert_vec_close(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(actual.len(), expected.len());

        for i in 0..actual.len() {
            assert_close(actual[i], expected[i], tol);
        }
    }

    fn integrate_fixed<F, J>(
        solver: &mut Rodas4Solver<F, J>,
        t0: f64,
        y0: &[f64],
        t_end: f64,
        h: f64,
    ) -> Vec<f64>
    where
        F: Fn(f64, &[f64], &mut [f64]),
        J: Fn(f64, &[f64], &mut Vec<Triplet<usize, usize, f64>>),
    {
        let mut t = t0;
        let mut y = y0.to_vec();

        while t < t_end - 1e-15 {
            let h_step = (t_end - t).min(h);
            let step = solver.try_step(t, &y, h_step).unwrap();

            y = step.y_new;
            t += h_step;
        }

        y
    }

    #[test]
    fn scalar_relaxation_to_equilibrium_matches_exact_solution() {
        // y' = -k (y - y_eq)
        // y(t) = y_eq + (y0 - y_eq) exp(-k t)

        let k = 25.0;
        let y_eq = 2.5;

        let f = move |_: f64, y: &[f64], out: &mut [f64]| {
            out[0] = -k * (y[0] - y_eq);
        };

        let jac = move |_: f64, _: &[f64], triplets: &mut Vec<Triplet<usize, usize, f64>>| {
            triplets.clear();
            triplets.push(Triplet::new(0, 0, -k));
        };

        let cfg = Rodas4Config {
            rtol: 1e-8,
            atol: 1e-12,
            h_min: 1e-16,
            h_max: 1e-2,
            ..Default::default()
        };

        let mut solver = Rodas4Solver::new(f, jac, cfg);

        let y0 = vec![10.0];
        let t_end = 0.5;

        let (_, y) = solver.integrate(0.0, &y0, t_end, 1e-4).unwrap();

        let exact = y_eq + (y0[0] - y_eq) * (-k * t_end).exp();

        assert_close(y[0], exact, 1e-7);
    }

    #[test]
    fn robertson_preserves_mass_and_positivity() {
        let f = |_: f64, y: &[f64], out: &mut [f64]| {
            out[0] = -0.04 * y[0] + 1.0e4 * y[1] * y[2];
            out[1] = 0.04 * y[0] - 1.0e4 * y[1] * y[2] - 3.0e7 * y[1] * y[1];
            out[2] = 3.0e7 * y[1] * y[1];
        };

        let jac = |_: f64, y: &[f64], triplets: &mut Vec<Triplet<usize, usize, f64>>| {
            triplets.clear();

            triplets.push(Triplet::new(0, 0, -0.04));
            triplets.push(Triplet::new(0, 1, 1.0e4 * y[2]));
            triplets.push(Triplet::new(0, 2, 1.0e4 * y[1]));

            triplets.push(Triplet::new(1, 0, 0.04));
            triplets.push(Triplet::new(1, 1, -1.0e4 * y[2] - 6.0e7 * y[1]));
            triplets.push(Triplet::new(1, 2, -1.0e4 * y[1]));

            triplets.push(Triplet::new(2, 1, 6.0e7 * y[1]));
        };

        let cfg = Rodas4Config {
            rtol: 1e-7,
            atol: 1e-12,
            h_min: 1e-16,
            h_max: 1e-2,
            ..Default::default()
        };

        let mut solver = Rodas4Solver::new(f, jac, cfg);

        let y0 = vec![1.0, 0.0, 0.0];
        let (_, y) = solver.integrate(0.0, &y0, 1.0, 1e-6).unwrap();

        let mass = y.iter().sum::<f64>();

        assert_close(mass, 1.0, 1e-8);

        for (i, yi) in y.iter().enumerate() {
            assert!(*yi >= -1e-12, "negative concentration at index {i}: {yi:e}");
        }
    }

    #[test]
    fn robertson_matches_reference_at_t_1() {
        let f = |_: f64, y: &[f64], out: &mut [f64]| {
            out[0] = -0.04 * y[0] + 1.0e4 * y[1] * y[2];
            out[1] = 0.04 * y[0] - 1.0e4 * y[1] * y[2] - 3.0e7 * y[1] * y[1];
            out[2] = 3.0e7 * y[1] * y[1];
        };

        let jac = |_: f64, y: &[f64], triplets: &mut Vec<Triplet<usize, usize, f64>>| {
            triplets.clear();

            triplets.push(Triplet::new(0, 0, -0.04));
            triplets.push(Triplet::new(0, 1, 1.0e4 * y[2]));
            triplets.push(Triplet::new(0, 2, 1.0e4 * y[1]));

            triplets.push(Triplet::new(1, 0, 0.04));
            triplets.push(Triplet::new(1, 1, -1.0e4 * y[2] - 6.0e7 * y[1]));
            triplets.push(Triplet::new(1, 2, -1.0e4 * y[1]));

            triplets.push(Triplet::new(2, 1, 6.0e7 * y[1]));
        };

        let cfg = Rodas4Config {
            rtol: 1e-8,
            atol: 1e-12,
            h_min: 1e-16,
            h_max: 1e-3,
            ..Default::default()
        };

        let mut solver = Rodas4Solver::new(f, jac, cfg);

        let y0 = vec![1.0, 0.0, 0.0];
        let (_, y) = solver.integrate(0.0, &y0, 1.0, 1e-6).unwrap();

        // Reference value for Robertson problem near t = 1.
        // The tolerance is intentionally not microscopic: this is a solver-level
        // correctness test, not a bitwise regression test.
        let reference = vec![9.66459737e-1, 3.07462658e-5, 3.35095160e-2];

        assert_vec_close(&y, &reference, 5e-5);
    }

    #[test]
    fn fixed_step_observed_order_is_close_to_two_on_autonomous_problem() {
        // This test answers the important question:
        // does the current implementation behave as a second-order method?
        //
        // We use fixed step sizes and compare against the exact solution.
        // For a second-order method, halving h should reduce the error
        // by approximately 4.

        let make_solver = || {
            let f = |_: f64, y: &[f64], out: &mut [f64]| {
                out[0] = -2.0 * y[0];
            };

            let jac = |_: f64, _: &[f64], triplets: &mut Vec<Triplet<usize, usize, f64>>| {
                triplets.clear();
                triplets.push(Triplet::new(0, 0, -2.0));
            };

            let cfg = Rodas4Config {
                rtol: 1e-12,
                atol: 1e-14,
                h_min: 1e-16,
                h_max: 1.0,
                ..Default::default()
            };

            Rodas4Solver::new(f, jac, cfg)
        };

        let y0 = vec![1.0];
        let t_end = 1.0;
        let exact = (-2.0_f64 * t_end).exp();

        let mut solver_h = make_solver();
        let y_h = integrate_fixed(&mut solver_h, 0.0, &y0, t_end, 1.0 / 20.0);
        let err_h = (y_h[0] - exact).abs();

        let mut solver_h2 = make_solver();
        let y_h2 = integrate_fixed(&mut solver_h2, 0.0, &y0, t_end, 1.0 / 40.0);
        let err_h2 = (y_h2[0] - exact).abs();

        let mut solver_h4 = make_solver();
        let y_h4 = integrate_fixed(&mut solver_h4, 0.0, &y0, t_end, 1.0 / 80.0);
        let err_h4 = (y_h4[0] - exact).abs();

        let order_1 = (err_h / err_h2).log2();
        let order_2 = (err_h2 / err_h4).log2();

        println!("err_h  = {err_h:e}");
        println!("err_h2 = {err_h2:e}");
        println!("err_h4 = {err_h4:e}");
        println!("observed order: {order_1:.4}, {order_2:.4}");

        assert!(
            order_1 > 1.75 && order_2 > 1.75,
            "observed order too low: {order_1:.4}, {order_2:.4}"
        );
    }

    #[test]
    fn tighter_tolerance_reduces_error_on_scalar_problem() {
        let run = |rtol: f64| {
            let f = |_: f64, y: &[f64], out: &mut [f64]| {
                out[0] = -10.0 * y[0];
            };

            let jac = |_: f64, _: &[f64], triplets: &mut Vec<Triplet<usize, usize, f64>>| {
                triplets.clear();
                triplets.push(Triplet::new(0, 0, -10.0));
            };

            let cfg = Rodas4Config {
                rtol,
                atol: 1e-12,
                h_min: 1e-16,
                h_max: 0.1,
                ..Default::default()
            };

            let mut solver = Rodas4Solver::new(f, jac, cfg);

            let y0 = vec![1.0];
            let t_end = 1.0;

            let (_, y) = solver.integrate(0.0, &y0, t_end, 1e-4).unwrap();

            let exact = (-10.0_f64 * t_end).exp();

            (y[0] - exact).abs()
        };

        let err_loose = run(1e-4);
        let err_tight = run(1e-7);

        println!("err_loose = {err_loose:e}");
        println!("err_tight = {err_tight:e}");

        assert!(
            err_tight < err_loose,
            "tighter tolerance did not reduce error: loose={err_loose:e}, tight={err_tight:e}"
        );
    }

    #[test]
    fn fixed_step_observed_order_is_close_to_three_on_autonomous_problem() {
        let make_solver = || {
            let f = |_: f64, y: &[f64], out: &mut [f64]| {
                out[0] = -2.0 * y[0];
            };

            let jac = |_: f64, _: &[f64], triplets: &mut Vec<Triplet<usize, usize, f64>>| {
                triplets.clear();
                triplets.push(Triplet::new(0, 0, -2.0));
            };

            let cfg = Rodas4Config {
                rtol: 1e-13,
                atol: 1e-15,
                h_min: 1e-16,
                h_max: 1.0,
                max_steps: 1_000_000,
                ..Default::default()
            };

            Rodas4Solver::new(f, jac, cfg)
        };

        let y0 = vec![1.0];
        let t_end = 1.0;
        let exact = (-2.0_f64 * t_end).exp();

        let mut solver_h = make_solver();
        let y_h = integrate_fixed(&mut solver_h, 0.0, &y0, t_end, 1.0 / 10.0);
        let err_h = (y_h[0] - exact).abs();

        let mut solver_h2 = make_solver();
        let y_h2 = integrate_fixed(&mut solver_h2, 0.0, &y0, t_end, 1.0 / 20.0);
        let err_h2 = (y_h2[0] - exact).abs();

        let mut solver_h4 = make_solver();
        let y_h4 = integrate_fixed(&mut solver_h4, 0.0, &y0, t_end, 1.0 / 40.0);
        let err_h4 = (y_h4[0] - exact).abs();

        let order_1 = (err_h / err_h2).log2();
        let order_2 = (err_h2 / err_h4).log2();

        println!("err_h  = {err_h:e}");
        println!("err_h2 = {err_h2:e}");
        println!("err_h4 = {err_h4:e}");
        println!("observed order: {order_1:.4}, {order_2:.4}");

        assert!(
            order_1 > 2.75 && order_2 > 2.75,
            "observed order too low: {order_1:.4}, {order_2:.4}"
        );
    }

    #[test]
    fn robertson_tight_tolerance_uses_reasonable_number_of_steps() {
        let f = |_: f64, y: &[f64], out: &mut [f64]| {
            out[0] = -0.04 * y[0] + 1.0e4 * y[1] * y[2];
            out[1] = 0.04 * y[0] - 1.0e4 * y[1] * y[2] - 3.0e7 * y[1] * y[1];
            out[2] = 3.0e7 * y[1] * y[1];
        };

        let jac = |_: f64, y: &[f64], triplets: &mut Vec<Triplet<usize, usize, f64>>| {
            triplets.clear();

            triplets.push(Triplet::new(0, 0, -0.04));
            triplets.push(Triplet::new(0, 1, 1.0e4 * y[2]));
            triplets.push(Triplet::new(0, 2, 1.0e4 * y[1]));

            triplets.push(Triplet::new(1, 0, 0.04));
            triplets.push(Triplet::new(1, 1, -1.0e4 * y[2] - 6.0e7 * y[1]));
            triplets.push(Triplet::new(1, 2, -1.0e4 * y[1]));

            triplets.push(Triplet::new(2, 1, 6.0e7 * y[1]));
        };

        let cfg = Rodas4Config {
            rtol: 1e-8,
            atol: 1e-12,
            h_min: 1e-16,
            h_max: 1e-2,
            max_steps: 1_000_000,
            ..Default::default()
        };

        let mut solver = Rodas4Solver::new(f, jac, cfg);

        let y0 = vec![1.0, 0.0, 0.0];
        let (_, y) = solver.integrate(0.0, &y0, 1.0, 1e-6).unwrap();

        let reference = vec![9.66459737e-1, 3.07462658e-5, 3.35095160e-2];

        assert_vec_close(&y, &reference, 1e-6);

        let stats = solver.stats();
        println!("Rodas4 stats = {stats:?}");

        assert!(
            stats.accepted_steps < 20_000,
            "too many accepted steps: {}",
            stats.accepted_steps
        );

        assert!(
            stats.rejected_steps < stats.accepted_steps / 2 + 10,
            "too many rejected steps: accepted={}, rejected={}",
            stats.accepted_steps,
            stats.rejected_steps
        );
    }

    #[test]
    fn dfdt_not_called_for_autonomous_problem() {
        let f = |_: f64, y: &[f64], out: &mut [f64]| {
            out[0] = -y[0];
        };

        let jac = |_: f64, _: &[f64], triplets: &mut Vec<Triplet<usize, usize, f64>>| {
            triplets.clear();
            triplets.push(Triplet::new(0, 0, -1.0));
        };

        let cfg = Rodas4Config::default();

        let mut solver = Rodas4Solver::new(f, jac, cfg);

        let y0 = vec![1.0];
        let _ = solver.integrate(0.0, &y0, 1.0, 1e-3).unwrap();

        assert_eq!(solver.stats().dfdt_evals, 0);
    }
}
