//! LSODE2 Manual Method Guide (BDF-only)
//!
//! Why:
//! LSODE-style workflows often need explicit method control.
//! This guide shows how to force BDF family explicitly.
//! 
//! 
//! run cargo run --example lsode2_manual_bdf_guide
//!

use RustedSciThe::numerical::LSODE2::{
    Lsode2ControllerConfig, Lsode2LinearSolverPolicy, Lsode2LinearSystemStructure,
    Lsode2ProblemConfig, Lsode2ResidualJacobianSource, Lsode2SymbolicAssemblyBackend,
    Lsode2SymbolicExecutionMode,
};
use RustedSciThe::numerical::ODE_api2::UniversalODESolver;
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DVector;

fn main() {
    // Mildly stiff 2x2 system, convenient for short demo.
    let config = Lsode2ProblemConfig::new(
        vec![
            Expr::parse_expression("-25.0*y1 + 24.0*y2"),
            Expr::parse_expression("y1 - y2"),
        ],
        vec!["y1".to_string(), "y2".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0, 0.0]),
        0.5,
        0.01,
        1e-7,
        1e-9,
    )
    .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
        assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
        execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
    })
    .with_linear_system_structure(Lsode2LinearSystemStructure::Sparse)
    .with_linear_solver_policy(Lsode2LinearSolverPolicy::Auto)
    // Explicit manual LSODE-style method choice:
    .with_bdf_only_controller()
    // or .with_controller(Lsode2ControllerConfig::bdf_only())
    // Native faithful BDF stepping path.
    .with_faithful_bdf_solve(100_000, 100_000);

    let mut solver = UniversalODESolver::lsode2_with_problem_config(config);
    solver.solve();

    let (t, y) = solver.get_result();
    let final_t = t.as_ref().map(|mesh| mesh[mesh.len() - 1]).unwrap_or(0.0);
    let final_y1 = y
        .as_ref()
        .map(|sol| sol[(sol.nrows() - 1, 0)])
        .unwrap_or(f64::NAN);
    let final_y2 = y
        .as_ref()
        .map(|sol| sol[(sol.nrows() - 1, 1)])
        .unwrap_or(f64::NAN);

    println!("LSODE2 manual BDF-only guide");
    println!("status  = {}", solver.get_status().unwrap_or_default());
    println!("final_t = {final_t:.6}");
    println!("final_y = [{final_y1:.8e}, {final_y2:.8e}]");
    if let Some(stats) = solver.get_statistics() {
        println!("stats   = {}", stats.table_report());
    }
}
