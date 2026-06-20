//! Параметрический чисто числовой гайд по `BVP_sci`.
//!
//! Этот пример показывает следующий шаг после базового BVP без символики:
//! солвер может восстанавливать неизвестные скалярные параметры вместе с состоянием.
//!
//! Демо-задача:
//! - y'(x) = p
//! - y(0) = 0
//! - p = 1
//!
//! Точное решение:
//! - p = 1
//! - y(x) = x
//! 
//! запуск: cargo run --example bvp_sci_numerical_parameters_guide

use std::time::Instant;

use RustedSciThe::numerical::BVP_sci::BVP_sci_faer::{faer_col, faer_dense_mat};
use RustedSciThe::numerical::BVP_sci::BVP_sci_numerical::{
    NumericalBvpClosureProblem, NumericalBvpSolveOptions, solve_numerical_bvp,
};

fn print_guide() {
    println!("Параметрический чисто числовой гайд по BVP_sci");
    println!("=======================================");
    println!("Этот пример показывает, как решать одновременно по состоянию и параметрам.");
    println!();
    println!("Ключевые элементы API:");
    println!("  - реализуйте parameter_dimension()");
    println!("  - читайте p внутри rhs() и boundary_residual()");
    println!("  - передайте начальное приближение параметра через with_parameters(...)");
    println!();
}

fn main() {
    print_guide();

    let mesh = faer_col::from_fn(12, |i| i as f64 / 11.0);
    let initial_guess = faer_dense_mat::from_fn(1, mesh.nrows(), |_, j| mesh[j] * 0.5);
    let initial_parameter_guess = faer_col::from_fn(1, |_| 0.25);
    let problem = NumericalBvpClosureProblem::new_fd(
        1,
        1,
        |_x, _y, p, out| {
            out[0] = p[0];
        },
        |ya, _yb, p, out| {
            out[0] = ya[0];
            out[1] = p[0] - 1.0;
        },
    );

    let started = Instant::now();
    let result = solve_numerical_bvp(
        problem,
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
