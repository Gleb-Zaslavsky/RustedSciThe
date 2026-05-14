//! LSODE2 Lambdify Guide
//!
//! Goal:
//! solve a symbolic IVP using Lambdify (no ahead-of-time compilation).
//!
//! This is usually the best first symbolic workflow:
//! 1. write equations as strings (`Expr::parse_expression`)
//! 2. pick Jacobian structure (dense/sparse/banded)
//! 3. let LSODE2 build residual/Jacobian evaluators through Lambdify
//! 4. run through `UniversalODESolver` (same facade as other ODE solvers)
//!
//! Problem:
//! y1' = -10 y1 + 9 y2
//! y2' =  y1 - y2
//! with y(0)=[1,0], t in [0,1]
//!
//! Why sparse here:
//! this small system can be dense too, but sparse setup demonstrates the
//! production-oriented route used on larger Jacobians.

use RustedSciThe::numerical::LSODE2::{
    Lsode2BackendConfig, Lsode2LinearSolverPolicy, Lsode2LinearSystemStructure,
    Lsode2ProblemConfig, Lsode2ResidualJacobianSource, Lsode2SymbolicAssemblyBackend,
    Lsode2SymbolicExecutionMode,
};
use RustedSciThe::numerical::ODE_api2::UniversalODESolver;
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DVector;

fn main() {
    // Step 1: define symbolic system, variables, and integration interval.
    let config = Lsode2ProblemConfig::new(
        vec![
            Expr::parse_expression("-10.0*y1 + 9.0*y2"),
            Expr::parse_expression("y1 - y2"),
        ],
        vec!["y1".to_string(), "y2".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0, 0.0]),
        1.0,
        0.02,
        1e-6,
        1e-8,
    )
    // Step 2: select symbolic source + execution mode.
    //
    // `ExprLegacy`:
    // stable symbolic assembly backend.
    // `LambdifyExpr`:
    // runtime-generated Rust evaluator closures (no external compiler).
    .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
        assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
        execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
    })
    // Optional:
    // even on Lambdify route we can shape generated runtime chunking.
    // This is useful when evaluating heavy symbolic kernels on many cores.
    .with_backend(
        Lsode2BackendConfig::native_sparse_faer().with_generated_backend_target_chunks(4, 4),
    )
    // Step 3: choose linear algebra structure/policy.
    //
    // `Sparse + Auto` resolves to sparse LU backend in LSODE2 resolved plan.
    .with_linear_system_structure(Lsode2LinearSystemStructure::Sparse)
    .with_linear_solver_policy(Lsode2LinearSolverPolicy::Auto)
    // Step 4: pick faithful native BDF stepping path.
    .with_faithful_bdf_solve(100_000, 100_000);

    // Step 5: run through universal ODE facade.
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

    println!("LSODE2 Lambdify guide");
    println!("status  = {}", solver.get_status().unwrap_or_default());
    println!("final_t = {final_t:.6}");
    println!("final_y = [{final_y1:.8e}, {final_y2:.8e}]");
    println!(
        "note    = Lambdify route keeps setup simple; move to AOT when Jacobian evaluation becomes expensive."
    );
    println!(
        "note    = Dense/Banded routes use the same pattern: change linear structure + policy."
    );
    if let Some(stats) = solver.get_statistics() {
        println!("stats   = {}", stats.table_report());
    }
}
