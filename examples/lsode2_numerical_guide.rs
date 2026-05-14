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
//! - it highlights what each LSODE2 config knob does;
//! - it gives a baseline before moving to symbolic Lambdify/AOT routes.

use RustedSciThe::numerical::LSODE2::{
    Lsode2BackendConfig, Lsode2JacobianBackend, Lsode2LinearSolverBackend,
    Lsode2LinearSolverPolicy, Lsode2LinearSystemStructure, Lsode2ProblemConfig,
    Lsode2ResidualJacobianSource,
};
use RustedSciThe::numerical::ODE_api2::UniversalODESolver;
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::{DMatrix, DVector};

fn base_problem() -> Lsode2ProblemConfig {
    // `Lsode2ProblemConfig::new(...)` still takes symbolic placeholders
    // (equations/variable names) because this config type is shared with
    // symbolic routes. In analytical mode below, solver callbacks are taken
    // from `.with_analytical_callbacks(...)`.
    Lsode2ProblemConfig::new(
        vec![Expr::parse_expression("-2.0*y")],
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
    .with_linear_solver_policy(Lsode2LinearSolverPolicy::Auto)
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
    if let Some(stats) = solver.get_statistics() {
        println!("  stats   = {}", stats.table_report());
    }
}

fn main() {
    // Route 1: full analytical callbacks (residual + Jacobian).
    // Residual callback signature: f(t, y) -> dy/dt vector.
    // Jacobian callback signature: J(t, y) = df/dy matrix.
    let analytical = base_problem()
        .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Analytical)
        .with_analytical_callbacks(
            |_t, y: &DVector<f64>| DVector::from_vec(vec![-2.0 * y[0]]),
            |_t, _y: &DVector<f64>| DMatrix::from_row_slice(1, 1, &[-2.0]),
        );

    // Route 2: analytical residual + finite-difference Jacobian.
    //
    // We keep residual callback exact, but request FD Jacobian backend via
    // `with_backend(...FiniteDifference...)`.
    // The Jacobian closure is still required by the current API; its value is
    // not used when `FiniteDifference` backend is selected.
    let finite_difference = base_problem()
        .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Analytical)
        .with_analytical_callbacks(
            |_t, y: &DVector<f64>| DVector::from_vec(vec![-2.0 * y[0]]),
            |_t, _y: &DVector<f64>| DMatrix::from_row_slice(1, 1, &[0.0]),
        )
        .with_backend(
            Lsode2BackendConfig::default()
                .with_jacobian_backend(Lsode2JacobianBackend::FiniteDifference)
                .with_linear_solver_backend(Lsode2LinearSolverBackend::SparseFaer),
        );

    run_and_report("LSODE2 numerical: analytical Jacobian", analytical);
    run_and_report(
        "LSODE2 numerical: finite-difference Jacobian",
        finite_difference,
    );
}
