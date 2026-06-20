//! BVP Frozen Numerical Route Note
//!
//! Goal:
//! make the Numerical/Frozen boundary explicit in a runnable guide.
//!
//! `NR_Damp_solver_frozen` intentionally supports symbolic Lambdify and AOT
//! routes, not `NumericOnly`. A closure-defined BVP should be solved with
//! `NR_Damp_solver_damped`, whose numerical discretization accepts a Rust
//! residual and optionally a user Jacobian. This is an API boundary, not a
//! missing placeholder to emulate with an empty symbolic system.
//!
//! The runnable code below is the supported replacement for a hypothetical
//! Frozen numerical call: it solves the same linear BVP through Damped with a
//! residual closure and an analytical continuous Jacobian.
//! 
//! run cargo run --example bvp_frozen_numerical_route_guide

use std::collections::HashMap;

use RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_damped::{
    DampedSolverOptions, NRBVP, SolverParams,
};
use nalgebra::{DMatrix, DVector};

const N_STEPS: usize = 80;

fn main() {
    println!("Frozen has no pure Numerical route by design; using Damped Numerical.");

    let h = 1.0 / N_STEPS as f64;
    let mut guess = vec![0.0; 2 * N_STEPS];
    for i in 0..N_STEPS {
        guess[2 * i] = i as f64 * h;
        guess[2 * i + 1] = 1.0;
    }
    let initial_guess = DMatrix::from_column_slice(2, N_STEPS, &guess);
    let options = DampedSolverOptions::sparse_damped()
        .with_strategy_params(Some(SolverParams::default()))
        .with_abs_tolerance(1e-8)
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
        .with_loglevel(Some("none".to_string()));

    let mut solver = NRBVP::new_numeric_with_jacobian_options(
        initial_guess,
        vec!["y".to_string(), "z".to_string()],
        "x".to_string(),
        HashMap::from([
            ("y".to_string(), vec![(0, 0.0)]),
            ("z".to_string(), vec![(1, 1.0)]),
        ]),
        0.0,
        1.0,
        N_STEPS,
        options,
        |_x, y, _params| DVector::from_vec(vec![y[1], 0.0]),
        |_x, _y, _params| DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]),
    );
    solver.dont_save_log(true);
    solver
        .try_solve()
        .expect("Damped numerical replacement must solve");

    let result = solver.get_result().expect("solver must store a result");
    let max_error = (0..result.nrows())
        .map(|i| (result[(i, 0)] - i as f64 * h).abs())
        .fold(0.0_f64, f64::max);
    println!("Damped Numerical replacement: max |y-x|={max_error:.3e}");
    let stats = solver.get_statistics();
    println!(
        "  backend_policy = {}",
        stats
            .diagnostics
            .get("generated.backend_policy")
            .map(String::as_str)
            .unwrap_or("unknown")
    );
    assert!(max_error < 1e-7, "max error is {max_error:e}");
}
