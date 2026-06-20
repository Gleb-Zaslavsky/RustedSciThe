//! LSODE2 Numerical Guide
//!
//! Goal:
//! show the "pure numerical" route where user code provides callbacks directly,
//! without symbolic equations/Lambdify/AOT.
//!
//! We run the same IVP in two modes:
//! 1. analytical residual + analytical Jacobian
//! 2. analytical residual + finite-difference Jacobian
//!
//! Problem:
//! y' = -2y, y(0)=1, t in [0,1], exact y(1)=exp(-2) ~= 0.13533528
//!
//! Why this guide matters:
//! - this is the minimal integration path for users who already have formulas
//!   in Rust code;
//! - the public API does not ask for symbolic `Expr` placeholders on this route;
//! - it highlights what each LSODE2 config knob does;
//! - it gives a baseline before moving to symbolic Lambdify/AOT routes.
//! 
//! run cargo run --example lsode2_numerical_guide
//!

use RustedSciThe::numerical::LSODE2::{
    Lsode2LinearSystemStructure, Lsode2NumericProblemOptions, Lsode2ProblemConfig,
};
use RustedSciThe::numerical::ODE_api2::UniversalODESolver;
use nalgebra::{DMatrix, DVector};

fn numeric_options() -> Lsode2NumericProblemOptions {
    // Numerical options describe only IVP geometry and solver policy. No
    // symbolic equations are required on this route.
    //
    // For the shortest scalar examples you can use
    // `Lsode2ProblemConfig::new_numeric_fd(...)` or
    // `Lsode2ProblemConfig::new_numeric_with_jacobian(...)` directly. Here we
    // use the `_with_options` forms because they make backend policy visible.
    Lsode2NumericProblemOptions::new(
        vec!["y".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0]),
        1.0,
        0.05,
        1e-6,
        1e-8,
    )
    // We explicitly pick sparse structure to demonstrate that numerical mode
    // can use native sparse linear algebra, same as symbolic routes.
    .with_linear_system_structure(Lsode2LinearSystemStructure::Sparse)
    // "faithful_bdf_solve" selects the LSODE2 faithful native BDF execution.
    .with_faithful_bdf_solve(100_000, 100_000)
}

fn run_and_report(title: &str, config: Lsode2ProblemConfig) {
    let mut solver = UniversalODESolver::lsode2_with_problem_config(config);
    solver.solve();
    let (t, y) = solver.get_result();
    let final_t = t.as_ref().map(|mesh| mesh[mesh.len() - 1]).unwrap_or(0.0);
    let final_y = y
        .as_ref()
        .map(|sol| sol[(sol.nrows() - 1, 0)])
        .unwrap_or(f64::NAN);
    println!("{title}");
    println!("  status  = {}", solver.get_status().unwrap_or_default());
    println!("  final_t = {final_t:.6}");
    println!("  final_y = {final_y:.8e}");
    println!("  exact   = {:.8e}", (-2.0_f64).exp());
    println!("  |err|   = {:.3e}", (final_y - (-2.0_f64).exp()).abs());
    println!(
        "  note    = numerical route used Rust closures only; symbolic Expr pipeline was not user-facing."
    );
    if let Some(stats) = solver.get_statistics() {
        println!("  stats   = {}", stats.table_report());
    }
    println!();
}

fn main() {
    // Route 1: full analytical callbacks (residual + Jacobian).
    // Residual callback signature: f(t, y) -> dy/dt vector.
    // Jacobian callback signature: J(t, y) = df/dy matrix.
    let analytical = Lsode2ProblemConfig::new_numeric_with_jacobian_options(
        numeric_options(),
        |_t, y: &DVector<f64>| DVector::from_vec(vec![-2.0 * y[0]]),
        |_t, _y: &DVector<f64>| DMatrix::from_row_slice(1, 1, &[-2.0]),
    );

    // Route 2: analytical residual + finite-difference Jacobian.
    //
    // We provide only the residual callback. LSODE2 computes the Jacobian by
    // finite differences internally, so there is no dummy Jacobian closure.
    let finite_difference =
        Lsode2ProblemConfig::new_numeric_fd_with_options(numeric_options(), |_t, y| {
            DVector::from_vec(vec![-2.0 * y[0]])
        });

    run_and_report("LSODE2 numerical: analytical Jacobian", analytical);
    run_and_report(
        "LSODE2 numerical: finite-difference Jacobian",
        finite_difference,
    );
    println!("Takeaway:");
    println!("  - use `new_numeric_with_jacobian_options` when df/dy is cheap to maintain;");
    println!(
        "  - use `new_numeric_fd_with_options` when RHS is the only reliable source of truth;"
    );
    println!(
        "  - move to Lambdify/AOT when the model is naturally symbolic or repeated generated callbacks pay off."
    );
}
