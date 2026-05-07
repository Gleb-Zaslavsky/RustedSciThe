// src/ode/Rodas3_faer.rs

use faer::Mat;
use faer::prelude::*;
use faer::sparse::{SparseColMat, Triplet};

#[derive(Debug, Clone)]
pub struct Rodas3Config {
    pub rtol: f64,
    pub atol: f64,
    pub h_min: f64,
    pub h_max: f64,
    pub safety: f64,
    pub max_steps: usize,
    pub max_rejects: usize,
}

impl Default for Rodas3Config {
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

#[derive(Debug, Clone)]
pub struct Rodas3Stats {
    pub accepted_steps: usize,
    pub rejected_steps: usize,
    pub rhs_evals: usize,
    pub jacobian_evals: usize,
    pub lu_factorizations: usize,
    pub linear_solves: usize,
}

impl Rodas3Stats {
    fn new() -> Self {
        Self {
            accepted_steps: 0,
            rejected_steps: 0,
            rhs_evals: 0,
            jacobian_evals: 0,
            lu_factorizations: 0,
            linear_solves: 0,
        }
    }
}

#[derive(Debug)]
pub enum Rodas3Error {
    DimensionMismatch,
    StepUnderflow,
    TooManySteps,
    TooManyRejects,
    SingularMatrix,
    InvalidNumber,
}

pub struct Rodas3Solver<F, J>
where
    F: Fn(f64, &[f64], &mut [f64]),
    J: Fn(f64, &[f64], &mut Vec<Triplet<usize, usize, f64>>),
{
    f: F,
    jac: J,
    cfg: Rodas3Config,
    stats: Rodas3Stats,
}

impl<F, J> Rodas3Solver<F, J>
where
    F: Fn(f64, &[f64], &mut [f64]),
    J: Fn(f64, &[f64], &mut Vec<Triplet<usize, usize, f64>>),
{
    pub fn new(f: F, jac: J, cfg: Rodas3Config) -> Self {
        Self {
            f,
            jac,
            cfg,
            stats: Rodas3Stats::new(),
        }
    }

    pub fn stats(&self) -> &Rodas3Stats {
        &self.stats
    }

    pub fn integrate(
        &mut self,
        t0: f64,
        y0: &[f64],
        t_end: f64,
        h0: f64,
    ) -> Result<(f64, Vec<f64>), Rodas3Error> {
        let n = y0.len();
        if n == 0 {
            return Err(Rodas3Error::DimensionMismatch);
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

                let factor = step_size_factor(step.err_norm, 2.0, self.cfg.safety);
                h *= factor;
                h = clamp_abs(h, self.cfg.h_min, self.cfg.h_max);
            } else {
                self.stats.rejected_steps += 1;
                rejects_in_row += 1;

                if rejects_in_row > self.cfg.max_rejects {
                    return Err(Rodas3Error::TooManyRejects);
                }

                let factor = step_size_factor(step.err_norm, 2.0, self.cfg.safety)
                    .min(0.5)
                    .max(0.1);

                h *= factor;

                if h.abs() < self.cfg.h_min {
                    return Err(Rodas3Error::StepUnderflow);
                }
            }
        }

        Err(Rodas3Error::TooManySteps)
    }
    fn try_step(&mut self, t: f64, y: &[f64], h: f64) -> Result<Rodas3Step, Rodas3Error> {
        let n = y.len();

        // KPP RODAS-3, autonomous version.
        //
        // KPP form:
        //
        //     G = 1 / (h gamma) I - J
        //     G k_i = f(T_i, Y_i) + sum_j c_ij / h * k_j
        //
        // Scaled form used here:
        //
        //     A = I - h gamma J
        //     A k_i = h gamma f(T_i, Y_i)
        //             + gamma * sum_j c_ij k_j
        //
        // For RODAS3:
        //     gamma = gamma_1 = 0.5

        let gamma = 0.5;

        let a31 = 2.0;
        let a41 = 2.0;
        let a43 = 1.0;

        let c21 = 4.0;
        let c31 = 1.0;
        let c32 = -1.0;
        let c41 = 1.0;
        let c42 = -1.0;
        let c43 = -8.0 / 3.0;

        let m1 = 2.0;
        let m2 = 0.0;
        let m3 = 1.0;
        let m4 = 1.0;

        // KPP error coefficients for RODAS3:
        // e1 = e2 = e3 = 0, e4 = 1
        //
        // Thus the last stage itself is the embedded error vector.
        let e4 = 1.0;

        let mut f0 = vec![0.0; n];
        (self.f)(t, y, &mut f0);
        self.stats.rhs_evals += 1;
        check_finite_slice(&f0)?;

        let mut jac_triplets = Vec::new();
        (self.jac)(t, y, &mut jac_triplets);
        self.stats.jacobian_evals += 1;

        let a = build_ros_matrix(n, gamma, h, &jac_triplets)?;
        let lu = a.sp_lu().map_err(|_| Rodas3Error::SingularMatrix)?;
        self.stats.lu_factorizations += 1;

        // Stage 1:
        //
        // A k1 = h gamma f(t, y)
        let rhs1 = col_from_slice_scaled(&f0, h * gamma);
        let k1_mat = lu.solve(&rhs1);
        self.stats.linear_solves += 1;

        let k1 = mat_col_to_vec(&k1_mat);
        check_finite_slice(&k1)?;

        // Stage 2:
        //
        // alpha2 = 0, a21 = 0, so f(T2, Y2) = f(t, y).
        //
        // A k2 = h gamma f0 + gamma c21 k1
        let mut rhs2_vec = vec![0.0; n];
        for i in 0..n {
            rhs2_vec[i] = h * gamma * f0[i] + gamma * c21 * k1[i];
        }

        let rhs2 = col_from_slice(&rhs2_vec);
        let k2_mat = lu.solve(&rhs2);
        self.stats.linear_solves += 1;

        let k2 = mat_col_to_vec(&k2_mat);
        check_finite_slice(&k2)?;

        // Stage 3:
        //
        // alpha3 = 1
        // Y3 = y + 2 k1
        //
        // A k3 = h gamma f(t + h, y + 2 k1)
        //        + gamma (c31 k1 + c32 k2)
        let mut y3 = vec![0.0; n];
        for i in 0..n {
            y3[i] = y[i] + a31 * k1[i];
        }

        let mut f3 = vec![0.0; n];
        (self.f)(t + h, &y3, &mut f3);
        self.stats.rhs_evals += 1;
        check_finite_slice(&f3)?;

        let mut rhs3_vec = vec![0.0; n];
        for i in 0..n {
            rhs3_vec[i] = h * gamma * f3[i] + gamma * (c31 * k1[i] + c32 * k2[i]);
        }

        let rhs3 = col_from_slice(&rhs3_vec);
        let k3_mat = lu.solve(&rhs3);
        self.stats.linear_solves += 1;

        let k3 = mat_col_to_vec(&k3_mat);
        check_finite_slice(&k3)?;

        // Stage 4:
        //
        // alpha4 = 1
        // Y4 = y + 2 k1 + k3
        //
        // A k4 = h gamma f(t + h, y + 2 k1 + k3)
        //        + gamma (c41 k1 + c42 k2 + c43 k3)
        let mut y4 = vec![0.0; n];
        for i in 0..n {
            y4[i] = y[i] + a41 * k1[i] + a43 * k3[i];
        }

        let mut f4 = vec![0.0; n];
        (self.f)(t + h, &y4, &mut f4);
        self.stats.rhs_evals += 1;
        check_finite_slice(&f4)?;

        let mut rhs4_vec = vec![0.0; n];
        for i in 0..n {
            rhs4_vec[i] = h * gamma * f4[i] + gamma * (c41 * k1[i] + c42 * k2[i] + c43 * k3[i]);
        }

        let rhs4 = col_from_slice(&rhs4_vec);
        let k4_mat = lu.solve(&rhs4);
        self.stats.linear_solves += 1;

        let k4 = mat_col_to_vec(&k4_mat);
        check_finite_slice(&k4)?;

        // Main solution:
        //
        // y_{n+1} = y + 2 k1 + k3 + k4
        //
        // Embedded error:
        //
        // err = k4
        let mut y_new = vec![0.0; n];
        let mut y_err = vec![0.0; n];

        for i in 0..n {
            y_new[i] = y[i] + m1 * k1[i] + m2 * k2[i] + m3 * k3[i] + m4 * k4[i];
            y_err[i] = e4 * k4[i];
        }

        let err_norm =
            weighted_error_norm_from_error_vector(y, &y_new, &y_err, self.cfg.atol, self.cfg.rtol)?;

        Ok(Rodas3Step { y_new, err_norm })
    }
}

struct Rodas3Step {
    y_new: Vec<f64>,
    err_norm: f64,
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

fn check_finite_slice(x: &[f64]) -> Result<(), Rodas3Error> {
    if x.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(Rodas3Error::InvalidNumber)
    }
}
fn mat_col_to_vec(x: &Mat<f64>) -> Vec<f64> {
    let n = x.nrows();
    let mut out = vec![0.0; n];

    for i in 0..n {
        out[i] = x[(i, 0)];
    }

    out
}
fn build_ros_matrix(
    n: usize,
    gamma: f64,
    h: f64,
    jac_triplets: &[Triplet<usize, usize, f64>],
) -> Result<SparseColMat<usize, f64>, Rodas3Error> {
    let mut triplets = Vec::with_capacity(jac_triplets.len() + n);

    for tr in jac_triplets {
        triplets.push(Triplet::new(tr.row, tr.col, -h * gamma * tr.val));
    }

    for i in 0..n {
        triplets.push(Triplet::new(i, i, 1.0));
    }

    SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets)
        .map_err(|_| Rodas3Error::DimensionMismatch)
}

fn col_from_slice(x: &[f64]) -> Mat<f64> {
    Mat::from_fn(x.len(), 1, |i, _| x[i])
}

fn col_from_slice_scaled(x: &[f64], scale: f64) -> Mat<f64> {
    Mat::from_fn(x.len(), 1, |i, _| scale * x[i])
}

fn weighted_error_norm_from_error_vector(
    y_old: &[f64],
    y_new: &[f64],
    y_err: &[f64],
    atol: f64,
    rtol: f64,
) -> Result<f64, Rodas3Error> {
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
        Err(Rodas3Error::InvalidNumber)
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

        let cfg = Rodas3Config {
            rtol: 1e-6,
            atol: 1e-12,
            h_min: 1e-16,
            h_max: 1.0,
            ..Default::default()
        };

        let mut solver = Rodas3Solver::new(f, jac, cfg);

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
        solver: &mut Rodas3Solver<F, J>,
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
    fn scalar_stiff_decay_matches_exact_solution_Rodas3_reasonable_tol() {
        let lambda = 1000.0;

        let f = move |_: f64, y: &[f64], out: &mut [f64]| {
            out[0] = -lambda * y[0];
        };

        let jac = move |_: f64, _: &[f64], triplets: &mut Vec<Triplet<usize, usize, f64>>| {
            triplets.clear();
            triplets.push(Triplet::new(0, 0, -lambda));
        };

        let cfg = Rodas3Config {
            rtol: 1e-6,
            atol: 1e-10,
            h_min: 1e-16,
            h_max: 1e-3,
            max_steps: 1_000_0,
            ..Default::default()
        };

        let mut solver = Rodas3Solver::new(f, jac, cfg);

        let y0 = vec![1.0];
        let (_, y) = solver.integrate(0.0, &y0, 0.01, 1e-6).unwrap();

        let exact = (-lambda * 0.01_f64).exp();

        assert_close(y[0], exact, 1e-6);
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

        let cfg = Rodas3Config {
            rtol: 1e-8,
            atol: 1e-12,
            h_min: 1e-16,
            h_max: 1e-2,
            ..Default::default()
        };

        let mut solver = Rodas3Solver::new(f, jac, cfg);

        let y0 = vec![10.0];
        let t_end = 0.5;

        let (_, y) = solver.integrate(0.0, &y0, t_end, 1e-4).unwrap();

        let exact = y_eq + (y0[0] - y_eq) * (-k * t_end).exp();

        assert_close(y[0], exact, 1e-7);
    }

    #[test]
    fn diagonal_stiff_linear_system_matches_exact_solution() {
        // Independent modes with very different time scales.

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

        let cfg = Rodas3Config {
            rtol: 1e-8,
            atol: 1e-12,
            h_min: 1e-16,
            h_max: 1e-3,
            ..Default::default()
        };

        let mut solver = Rodas3Solver::new(f, jac, cfg);

        let y0 = vec![1.0, 1.0, 1.0];
        let t_end = 0.01;

        let (_, y) = solver.integrate(0.0, &y0, t_end, 1e-6).unwrap();

        let exact = vec![
            (-lambdas[0] * t_end).exp(),
            (-lambdas[1] * t_end).exp(),
            (-lambdas[2] * t_end).exp(),
        ];

        assert_vec_close(&y, &exact, 1e-7);
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

        let cfg = Rodas3Config {
            rtol: 1e-7,
            atol: 1e-12,
            h_min: 1e-16,
            h_max: 1e-2,
            ..Default::default()
        };

        let mut solver = Rodas3Solver::new(f, jac, cfg);

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

        let cfg = Rodas3Config {
            rtol: 1e-8,
            atol: 1e-12,
            h_min: 1e-16,
            h_max: 1e-3,
            ..Default::default()
        };

        let mut solver = Rodas3Solver::new(f, jac, cfg);

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

            let cfg = Rodas3Config {
                rtol: 1e-12,
                atol: 1e-14,
                h_min: 1e-16,
                h_max: 1.0,
                ..Default::default()
            };

            Rodas3Solver::new(f, jac, cfg)
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

            let cfg = Rodas3Config {
                rtol,
                atol: 1e-12,
                h_min: 1e-16,
                h_max: 0.1,
                ..Default::default()
            };

            let mut solver = Rodas3Solver::new(f, jac, cfg);

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

            let cfg = Rodas3Config {
                rtol: 1e-13,
                atol: 1e-15,
                h_min: 1e-16,
                h_max: 1.0,
                max_steps: 1_000_000,
                ..Default::default()
            };

            Rodas3Solver::new(f, jac, cfg)
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

        let cfg = Rodas3Config {
            rtol: 1e-10,
            atol: 1e-14,
            h_min: 1e-16,
            h_max: 1e-3,
            max_steps: 1_000_000,
            ..Default::default()
        };

        let mut solver = Rodas3Solver::new(f, jac, cfg);

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

        let cfg = Rodas3Config {
            rtol: 1e-8,
            atol: 1e-12,
            h_min: 1e-16,
            h_max: 1e-2,
            max_steps: 1_000_000,
            ..Default::default()
        };

        let mut solver = Rodas3Solver::new(f, jac, cfg);

        let y0 = vec![1.0, 0.0, 0.0];
        let (_, y) = solver.integrate(0.0, &y0, 1.0, 1e-6).unwrap();

        let reference = vec![9.66459737e-1, 3.07462658e-5, 3.35095160e-2];

        assert_vec_close(&y, &reference, 1e-6);

        let stats = solver.stats();
        println!("RODAS3 stats = {stats:?}");

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
}
