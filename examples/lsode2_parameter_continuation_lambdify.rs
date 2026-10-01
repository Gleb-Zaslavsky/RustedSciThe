//! Reuse a prepared Lambdify solver while sweeping a numeric model parameter.
//!
//! Run with `cargo run --no-default-features --example lsode2_parameter_continuation_lambdify`.

use RustedSciThe::numerical::LSODE2::{
    Lsode2LinearSolverPolicy, Lsode2LinearSystemStructure, Lsode2ProblemConfig,
    Lsode2ResidualJacobianSource, Lsode2Solver, Lsode2SymbolicAssemblyBackend,
    Lsode2SymbolicExecutionMode,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DVector;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Lsode2ProblemConfig::new(
        vec![Expr::parse_expression("-k*y")],
        vec!["y".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0]),
        1.0,
        0.05,
        1e-8,
        1e-10,
    )
    .with_equation_parameters(vec!["k".to_string()])
    .with_equation_parameter_values(DVector::from_vec(vec![1.0]))
    .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
        assembly: Lsode2SymbolicAssemblyBackend::AtomView,
        execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
    })
    .with_linear_system_structure(Lsode2LinearSystemStructure::Sparse)
    .with_linear_solver_policy(Lsode2LinearSolverPolicy::Auto)
    .with_faithful_bdf_solve(10_000, 10_000);

    let mut solver = Lsode2Solver::new(config)?;
    for k in [1.0, 2.0, 4.0] {
        solver.set_parameter_values(DVector::from_vec(vec![k]))?;
        solver.solve()?;
        let (_, states) = solver.get_result();
        let final_y = states[(states.nrows() - 1, 0)];
        let exact = (-k).exp();
        assert!((final_y - exact).abs() < 1e-7, "parameter k={k} drifted");
        println!("k={k:.1}, y(1)={final_y:.8e}, exact={exact:.8e}");
    }

    Ok(())
}
