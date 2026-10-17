//! A small, typed ShootingBVP example for a scalar second-order equation.
//!
//! The shooting method is intended here for a small system with one unknown
//! initial slope. It is not a general replacement for collocation on large
//! coupled systems.

use RustedSciThe::numerical::Nonlinear_systems::symbolic::SymbolicLambdifyFrontend;
use RustedSciThe::numerical::ODE_api2::{NonStiffMethod, SolverType};
use RustedSciThe::numerical::ShootingBVP::Shooting_simple::{
    BoundaryCondition, BoundaryConditionType,
};
use RustedSciThe::numerical::ShootingBVP::{ShootingSymbolicOptions, shooting_bvp};
use RustedSciThe::numerical::SolverTelemetryMode;
use RustedSciThe::symbolic::symbolic_engine::Expr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // y'' = y becomes y' = dy and dy' = y. The exact solution satisfying
    // y(0)=0 and y(1)=sinh(1) is y(x)=sinh(x).
    let problem = shooting_bvp(
        vec![Expr::parse_expression("dy"), Expr::parse_expression("y")],
        vec!["y".into(), "dy".into()],
        "x",
    )
    .interval(0.0, 1.0)
    .boundary_conditions(
        BoundaryCondition {
            value: 0.0,
            bc_type: BoundaryConditionType::Dirichlet,
        },
        BoundaryCondition {
            value: 1.0_f64.sinh(),
            bc_type: BoundaryConditionType::Dirichlet,
        },
    )
    .initial_guess(0.8)
    .tolerance(1e-8)
    .max_iterations(30)
    .step_size(0.002)
    .ivp_solver(SolverType::NonStiff(NonStiffMethod::Rk45))
    .symbolic_options(ShootingSymbolicOptions {
        // ExprLegacy is the default; AtomViewNative is also supported.
        frontend: SymbolicLambdifyFrontend::ExprLegacy,
        ..ShootingSymbolicOptions::default()
    })
    .build()?;

    let report = problem.solve(SolverTelemetryMode::Detailed)?;
    let solution = &report.solve.solution;
    println!("inferred initial slope: {:.10}", solution.s);
    println!("right endpoint: {:?}", solution.bound_values);
    println!("frontend: {:?}", report.frontend);
    println!("symbolic preparation: {:?}", report.preparation);
    println!("solve telemetry: {:?}", report.solve.telemetry);
    Ok(())
}
