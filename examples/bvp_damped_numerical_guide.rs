//! BVP Damped Numerical Guide
//!
//! Goal:
//! show the pure numerical route, where ordinary Rust closures are the source
//! of truth and no symbolic equation placeholder is passed to the solver.
//!
//! We run the same BVP in two modes:
//! 1. residual closure + finite-difference Jacobian
//! 2. residual closure + analytical continuous Jacobian `df/dy`
//!
//! Problem:
//!
//! ```text
//! y' = z, z' = 0, y(0) = 0, z(1) = 1,
//! ```
//!
//! with exact solution `y(x) = x`, `z(x) = 1`.
//!
//! Why this guide matters:
//! - this route is appropriate when the model already exists as Rust code;
//! - the numerical constructors avoid meaningless symbolic placeholders;
//! - Banded linear algebra is still available for the discretized BVP matrix.

use std::collections::HashMap;

use RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_damped::{
    DampedSolverOptions, NRBVP, SolverParams,
};
use nalgebra::{DMatrix, DVector};

const N_STEPS: usize = 80;

fn options() -> DampedSolverOptions {
    DampedSolverOptions::sparse_damped()
        .with_strategy_params(Some(SolverParams::default()))
        .with_abs_tolerance(1e-10)
        .with_rel_tolerance(HashMap::from([
            ("y".to_string(), 1e-6),
            ("z".to_string(), 1e-6),
        ]))
        .with_bounds(HashMap::from([
            ("y".to_string(), (-0.2, 1.2)),
            ("z".to_string(), (-2.0, 2.0)),
        ]))
        .with_banded_generated_backend_defaults()
        .with_max_iterations(40)
        .with_loglevel(Some("none".to_string()))
}

fn initial_guess() -> DMatrix<f64> {
    let h = 1.0 / N_STEPS as f64;
    let mut guess = vec![0.0; 2 * N_STEPS];
    for i in 0..N_STEPS {
        guess[2 * i] = i as f64 * h;
        guess[2 * i + 1] = 1.0;
    }
    DMatrix::from_column_slice(2, N_STEPS, &guess)
}

fn boundary_conditions() -> HashMap<String, Vec<(usize, f64)>> {
    HashMap::from([
        ("y".to_string(), vec![(0, 0.0)]),
        ("z".to_string(), vec![(1, 1.0)]),
    ])
}

fn report(label: &str, solver: &NRBVP) {
    let result = solver.get_result().expect("solver must store a result");
    let h = 1.0 / N_STEPS as f64;
    let max_error = (0..result.nrows())
        .map(|i| (result[(i, 0)] - i as f64 * h).abs())
        .fold(0.0_f64, f64::max);
    println!(
        "{label}: rows={}, max |y-x|={max_error:.3e}",
        result.nrows()
    );
    let stats = solver.get_statistics();
    println!(
        "  backend_policy = {}",
        stats
            .diagnostics
            .get("generated.backend_policy")
            .map(String::as_str)
            .unwrap_or("unknown")
    );
    println!(
        "  iterations     = {}",
        stats
            .counters
            .get("number of iterations")
            .copied()
            .unwrap_or(0)
    );
    println!(
        "  linear_solves  = {}",
        stats
            .counters
            .get("number of solving linear systems")
            .copied()
            .unwrap_or(0)
    );
    assert!(max_error < 5e-5, "{label}: max error is {max_error:e}");
}

fn main() {
    let values = vec!["y".to_string(), "z".to_string()];

    // Route 1: provide only the continuous RHS. The large Newton Jacobian of
    // the discretized BVP is obtained by finite differences.
    let mut finite_difference = NRBVP::new_numeric_fd_with_options(
        initial_guess(),
        values.clone(),
        "x".to_string(),
        boundary_conditions(),
        0.0,
        1.0,
        N_STEPS,
        options(),
        |_x, y, _params| DVector::from_vec(vec![y[1], 0.0]),
    );
    finite_difference.dont_save_log(true);
    finite_difference
        .try_solve()
        .expect("finite-difference numerical BVP must solve");
    report(
        "Damped numerical route with FD Jacobian",
        &finite_difference,
    );

    // Route 2: provide the small per-node Jacobian. RustedSciThe assembles
    // its discretized global banded counterpart internally.
    let mut analytical_jacobian = NRBVP::new_numeric_with_jacobian_options(
        initial_guess(),
        values,
        "x".to_string(),
        boundary_conditions(),
        0.0,
        1.0,
        N_STEPS,
        options(),
        |_x, y, _params| DVector::from_vec(vec![y[1], 0.0]),
        |_x, _y, _params| DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]),
    );
    analytical_jacobian.dont_save_log(true);
    analytical_jacobian
        .try_solve()
        .expect("analytical-Jacobian numerical BVP must solve");
    report(
        "Damped numerical route with user Jacobian",
        &analytical_jacobian,
    );
}
