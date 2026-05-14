//! Parametric pure numerical `BVP_sci` guide.
//!
//! This example shows the next step after the basic no-symbolic guide:
//! the solver can recover unknown scalar parameters together with the state.
//!
//! Demo problem:
//! - y'(x) = p
//! - y(0) = 0
//! - p = 1
//!
//! Exact solution:
//! - p = 1
//! - y(x) = x

use std::time::Instant;

use RustedSciThe::numerical::BVP_sci::BVP_sci_faer::{faer_col, faer_dense_mat};
use RustedSciThe::numerical::BVP_sci::BVP_sci_numerical::{
    NumericalBvpProblem, NumericalBvpSolveOptions, solve_numerical_bvp,
};

struct ParametricLinearProblem;

impl NumericalBvpProblem for ParametricLinearProblem {
    fn dimension(&self) -> usize {
        1
    }

    fn parameter_dimension(&self) -> usize {
        1
    }

    fn rhs(&self, _x: f64, _y: &[f64], p: &[f64], out: &mut [f64]) {
        out[0] = p[0];
    }

    fn boundary_residual(&self, ya: &[f64], _yb: &[f64], p: &[f64], out: &mut [f64]) {
        out[0] = ya[0];
        out[1] = p[0] - 1.0;
    }
}

fn print_guide() {
    println!("BVP_sci pure numerical parameters guide");
    println!("=======================================");
    println!("This example shows how to solve for state and parameters together.");
    println!();
    println!("Key API pieces:");
    println!("  - implement parameter_dimension()");
    println!("  - read p inside rhs() and boundary_residual()");
    println!("  - pass an initial parameter guess with with_parameters(...)");
    println!();
}

fn main() {
    print_guide();

    let mesh = faer_col::from_fn(12, |i| i as f64 / 11.0);
    let initial_guess = faer_dense_mat::from_fn(1, mesh.nrows(), |_, j| mesh[j] * 0.5);
    let initial_parameter_guess = faer_col::from_fn(1, |_| 0.25);

    let started = Instant::now();
    let result = solve_numerical_bvp(
        ParametricLinearProblem,
        NumericalBvpSolveOptions::new(mesh, initial_guess, 1e-7, 256)
            .with_parameters(Some(initial_parameter_guess))
            .with_verbose(0),
    )
    .expect("parametric pure numerical BVP should solve");
    let total_ms = started.elapsed().as_secs_f64() * 1_000.0;

    let solved_parameter = result
        .p
        .as_ref()
        .expect("solver should return the recovered parameter");
    let y_left = result.y[(0, 0)];
    let y_right = result.y[(0, result.y.ncols() - 1)];

    println!("Solved in {total_ms:.3} ms");
    println!("Recovered parameter p = {:.6}", solved_parameter[0]);
    println!(
        "Solution endpoints: y(0) = {:.6}, y(1) = {:.6}",
        y_left, y_right
    );
    println!("Expected reference: p = 1.0, y(0) = 0.0, y(1) = 1.0");
}
