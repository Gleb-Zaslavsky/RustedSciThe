//! BVP_sci Lambdify guide.
//!
//! This example shows the symbolic baseline route for `BVP_sci`:
//! `ExprLegacy + Lambdify + sparse runtime planning`.
//!
//! It is the simplest symbolic entry point when you already have equations as
//! `Expr` objects and want a fast correctness-first solve without compiling an
//! external artifact.
//!
//! Run with:
//! `cargo run --example bvp_sci_lambdify_guide`

use std::collections::HashMap;

use nalgebra::DMatrix;

use RustedSciThe::numerical::BVP_sci::BVP_sci_symb::{BVPwrap, BvpSciSolverOptions};
use RustedSciThe::symbolic::symbolic_engine::Expr;

const N_STEPS: usize = 64;

fn initial_guess() -> DMatrix<f64> {
    DMatrix::from_element(2, N_STEPS, 0.5)
}

fn solver_options() -> BvpSciSolverOptions {
    let equations = vec![Expr::parse_expression("z"), Expr::parse_expression("0.0")];
    let boundary_conditions = HashMap::from([
        ("y".to_string(), vec![(0usize, 0.0f64)]),
        ("z".to_string(), vec![(1usize, 1.0f64)]),
    ]);

    BvpSciSolverOptions::new(
        None,
        Some(0.0),
        Some(1.0),
        Some(N_STEPS),
        equations,
        vec!["y".to_string(), "z".to_string()],
        vec![],
        None,
        boundary_conditions,
        "x".to_string(),
        1e-8,
        256,
        initial_guess(),
    )
    .with_loglevel(Some("none".to_string()))
    .with_expr_legacy_smart_sparse()
}

fn main() {
    println!("BVP_sci Lambdify guide");
    println!("======================");
    println!("This is the symbolic baseline route.");
    println!("Use it when you want the cleanest one-shot symbolic workflow.");
    println!();

    let mut solver = BVPwrap::new_with_options(solver_options());
    solver
        .try_solve()
        .expect("BVP_sci Lambdify guide should solve successfully");
    let result = solver.get_result().expect("solver should store a result");
    let stats = solver.get_statistics();
    let max_error = (0..result.ncols())
        .map(|i| (result[(0, i)] - i as f64 / (N_STEPS - 1) as f64).abs())
        .fold(0.0_f64, f64::max);

    println!("rows          = {}", result.nrows());
    println!("cols          = {}", result.ncols());
    println!("max |y - x|   = {max_error:.3e}");
    println!("iterations    = {}", stats.number_of_iterations);
    println!("residual_ms   = {:.3}", stats.residual_ms_total);
    println!("jacobian_ms   = {:.3}", stats.jacobian_ms_total);
    println!("linear_ms     = {:.3}", stats.linear_system_ms_total);
    println!("symbolic_ms   = {:.3}", stats.symbolic_prepare_ms_total);
}
