//! Pure numerical `BVP_sci` guide.
//!
//! This example is the user-facing entry point for the no-symbolic BVP story:
//! - `FiniteDifference` is the easiest mode: implement only `rhs()` and
//!   `boundary_residual()`.
//! - `AnalyticalPointwise` is the performance-oriented mode: implement those
//!   callbacks plus pointwise Jacobians.
//!
//! Practical rule of thumb from the current comparison suite:
//! - start with `FiniteDifference` for small systems or quick prototyping;
//! - move to `AnalyticalPointwise` once the problem becomes medium/large or the
//!   same model needs to be solved repeatedly with good throughput.

use std::time::Instant;

use RustedSciThe::numerical::BVP_Damp::BVP_utils::CustomTimer;
use RustedSciThe::numerical::BVP_sci::BVP_sci_faer::{faer_col, faer_dense_mat, faer_mat};
use RustedSciThe::numerical::BVP_sci::BVP_sci_numerical::{
    NumericalBvpProblem, NumericalBvpSolveOptions, NumericalJacobianMode, solve_numerical_bvp,
};
use faer::sparse::Triplet;

struct HarmonicProblemFd;

impl NumericalBvpProblem for HarmonicProblemFd {
    fn dimension(&self) -> usize {
        2
    }

    fn rhs(&self, _x: f64, y: &[f64], _p: &[f64], out: &mut [f64]) {
        out[0] = y[1];
        out[1] = -y[0];
    }

    fn boundary_residual(&self, ya: &[f64], _yb: &[f64], _p: &[f64], out: &mut [f64]) {
        out[0] = ya[0];
        out[1] = ya[1] - 1.0;
    }
}

struct HarmonicProblemAnalytical;

impl NumericalBvpProblem for HarmonicProblemAnalytical {
    fn dimension(&self) -> usize {
        2
    }

    fn jacobian_mode(&self) -> NumericalJacobianMode {
        NumericalJacobianMode::AnalyticalPointwise
    }

    fn rhs(&self, _x: f64, y: &[f64], _p: &[f64], out: &mut [f64]) {
        out[0] = y[1];
        out[1] = -y[0];
    }

    fn boundary_residual(&self, ya: &[f64], _yb: &[f64], _p: &[f64], out: &mut [f64]) {
        out[0] = ya[0];
        out[1] = ya[1] - 1.0;
    }

    fn rhs_jacobian(&self, _x: f64, _y: &[f64], _p: &[f64]) -> Option<faer_mat> {
        Some(
            faer_mat::try_new_from_triplets(
                2,
                2,
                &[
                    Triplet::new(0usize, 1usize, 1.0),
                    Triplet::new(1usize, 0usize, -1.0),
                ],
            )
            .expect("harmonic rhs Jacobian should be constructible"),
        )
    }

    fn boundary_jacobian(
        &self,
        _ya: &[f64],
        _yb: &[f64],
        _p: &[f64],
    ) -> Option<(faer_mat, faer_mat, Option<faer_mat>)> {
        let dya = faer_mat::try_new_from_triplets(
            2,
            2,
            &[
                Triplet::new(0usize, 0usize, 1.0),
                Triplet::new(1usize, 1usize, 1.0),
            ],
        )
        .expect("harmonic left BC Jacobian should be constructible");
        let dyb = faer_mat::try_new_from_triplets(2, 2, &[])
            .expect("harmonic right BC Jacobian should be constructible");
        Some((dya, dyb, None))
    }
}

fn harmonic_mesh(n_steps: usize) -> faer_col {
    faer_col::from_fn(n_steps, |i| {
        std::f64::consts::PI * i as f64 / (n_steps.saturating_sub(1)) as f64
    })
}

fn harmonic_initial_guess(mesh: &faer_col) -> faer_dense_mat {
    faer_dense_mat::from_fn(2, mesh.nrows(), |i, j| match i {
        0 => mesh[j].sin(),
        1 => mesh[j].cos(),
        _ => 0.0,
    })
}

fn run_case<P: NumericalBvpProblem + 'static>(label: &str, problem: P, n_steps: usize) {
    let mesh = harmonic_mesh(n_steps);
    let guess = harmonic_initial_guess(&mesh);
    let custom_timer = CustomTimer::new();
    let started = Instant::now();
    let result = solve_numerical_bvp(
        problem,
        NumericalBvpSolveOptions::new(mesh, guess, 1e-6, 512)
            .with_verbose(0)
            .with_custom_timer(Some(custom_timer)),
    )
    .unwrap_or_else(|err| panic!("{label} failed: {err}"));
    let total_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let (nrows, ncols) = result.y.shape();
    let mut max_abs_solution = 0.0_f64;
    for i in 0..nrows {
        for j in 0..ncols {
            max_abs_solution = max_abs_solution.max(result.y[(i, j)].abs());
        }
    }
    let timer = &result.custom_timer;

    println!(
        "{label:28} total_ms = {total_ms:9.3}, residual_ms = {:8.3}, jacobian_ms = {:8.3}, linear_ms = {:8.3}, max_abs_solution = {:.6e}",
        timer.fun.as_secs_f64() * 1_000.0,
        timer.jac.as_secs_f64() * 1_000.0,
        timer.linear_system.as_secs_f64() * 1_000.0,
        max_abs_solution
    );
}

fn print_guide() {
    println!("BVP_sci pure numerical guide");
    println!("============================");
    println!("Two public modes:");
    println!("  - FiniteDifference: easiest API, only rhs() and boundary_residual()");
    println!("  - AnalyticalPointwise: add rhs_jacobian() and boundary_jacobian()");
    println!();
    println!("When to choose which:");
    println!("  - Use FiniteDifference for quick prototypes and small systems.");
    println!("  - Use AnalyticalPointwise when Jacobian cost starts to dominate");
    println!("    or when the same numerical model must solve repeatedly.");
    println!();
    println!("Demo problem: y'' + y = 0 with y(0) = 0 and y'(0) = 1.");
    println!("Exact solution: y(x) = sin(x), y'(x) = cos(x).");
    println!();
}

fn main() {
    let n_steps = 64;
    print_guide();
    run_case("FiniteDifference", HarmonicProblemFd, n_steps);
    run_case("AnalyticalPointwise", HarmonicProblemAnalytical, n_steps);
}
