//! LSODE2 Manual Method Guide (Adams-only)
//!
//! Why:
//! for clearly non-stiff IVPs, LSODE-style users may want to pin Adams family
//! manually instead of relying on automatic switching.

use RustedSciThe::numerical::LSODE2::{
    Lsode2LinearSolverPolicy, Lsode2LinearSystemStructure, Lsode2ProblemConfig,
    Lsode2ResidualJacobianSource, Lsode2SymbolicAssemblyBackend, Lsode2SymbolicExecutionMode,
};
use RustedSciThe::numerical::ODE_api2::UniversalODESolver;
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DVector;

fn main() {
    // Non-stiff scalar decay, convenient Adams-only demo.
    let config = Lsode2ProblemConfig::new(
        vec![Expr::parse_expression("-1.2*y")],
        vec!["y".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0]),
        1.0,
        0.02,
        1e-7,
        1e-9,
    )
    .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
        assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
        execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
    })
    .with_linear_system_structure(Lsode2LinearSystemStructure::Dense)
    .with_linear_solver_policy(Lsode2LinearSolverPolicy::Auto)
    // Explicit manual LSODE-style method choice:
    .with_adams_only_controller()
    // Native solve path (same "faithful" execution entrypoint).
    .with_faithful_bdf_solve(100_000, 100_000);

    let mut solver = UniversalODESolver::lsode2_with_problem_config(config);
    solver.solve();

    let (t, y) = solver.get_result();
    let final_t = t.as_ref().map(|mesh| mesh[mesh.len() - 1]).unwrap_or(0.0);
    let final_y = y
        .as_ref()
        .map(|sol| sol[(sol.nrows() - 1, 0)])
        .unwrap_or(f64::NAN);

    println!("LSODE2 manual Adams-only guide");
    println!("status  = {}", solver.get_status().unwrap_or_default());
    println!("final_t = {final_t:.6}");
    println!("final_y = {final_y:.8e}");
    if let Some(stats) = solver.get_statistics() {
        println!("stats   = {}", stats.table_report());
    }
}
