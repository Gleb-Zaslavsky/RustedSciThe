//! BVP Frozen Lambdify Guide
//!
//! Goal:
//! solve a symbolic BVP with the Frozen Newton strategy through Lambdify.
//!
//! Frozen Newton keeps a Jacobian fixed across nonlinear updates. It can be
//! useful when Jacobian rebuilds dominate a problem, but unlike Damped it does
//! not expose a pure numerical closure route. Frozen receives symbolic
//! equations and executes them through Lambdify or AOT.
//!
//! This first example uses Banded + AtomView + Lambdify and therefore has no
//! external compiler requirement.

use std::collections::HashMap;

use RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_frozen::{FrozenSolverOptions, NRBVP};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DMatrix;

const N_STEPS: usize = 40;

fn initial_guess() -> DMatrix<f64> {
    let h = 1.0 / N_STEPS as f64;
    let mut guess = vec![0.0; 2 * N_STEPS];
    for i in 0..N_STEPS {
        guess[2 * i] = i as f64 * h;
        guess[2 * i + 1] = 1.0;
    }
    DMatrix::from_column_slice(2, N_STEPS, &guess)
}

fn main() {
    // Step 1: Banded linear algebra and Lambdify callback execution.
    let options = FrozenSolverOptions::banded_frozen()
        .with_banded_lambdify()
        .with_tolerance(1e-8)
        .with_max_iterations(40);
    // Step 2: the symbolic first-order BVP has exact solution y=x, z=1.
    let mut solver = NRBVP::new_with_options(
        vec![Expr::parse_expression("z"), Expr::parse_expression("0.0")],
        initial_guess(),
        vec!["y".to_string(), "z".to_string()],
        "x".to_string(),
        HashMap::from([
            ("y".to_string(), vec![(0, 0.0)]),
            ("z".to_string(), vec![(0, 1.0)]),
        ]),
        0.0,
        1.0,
        N_STEPS,
        options,
    );
    solver.dont_save_log(true);
    solver.try_solve().expect("Frozen Lambdify BVP must solve");

    let result = solver.get_result().expect("solver must store a result");
    let h = 1.0 / N_STEPS as f64;
    let max_error = (0..result.nrows())
        .map(|i| (result[(i, 0)] - (i + 1) as f64 * h).abs())
        .fold(0.0_f64, f64::max);
    println!(
        "Frozen Banded Lambdify/AtomView: rows={}, max |y-x|={max_error:.3e}",
        result.nrows()
    );
    let stats = solver.get_statistics();
    println!(
        "  selected_backend = {}",
        stats
            .diagnostics
            .get("generated.selected_backend")
            .map(String::as_str)
            .unwrap_or("unknown")
    );
    println!(
        "  iterations       = {}",
        stats
            .counters
            .get("number of iterations")
            .copied()
            .unwrap_or(0)
    );
    assert!(max_error < 1e-7, "max error is {max_error:e}");
}
