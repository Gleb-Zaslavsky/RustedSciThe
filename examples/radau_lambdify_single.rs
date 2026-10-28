//! Minimal public Radau Lambdify solve.

use RustedSciThe::numerical::Radau::{
    RadauConfig, RadauFrontend, RadauMatrixLayout, RadauProblem, RadauSolver, RadauTelemetryMode,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let problem = RadauProblem::new(
        vec![Expr::parse_expression("-k*y")],
        vec!["y".to_owned()],
        "t",
    )
    .with_parameter_names(vec!["k".to_owned()])
    .with_jacobian(vec![Expr::parse_expression("-k")]);
    let config = RadauConfig {
        frontend: RadauFrontend::ExprLegacy,
        matrix_layout: RadauMatrixLayout::Dense,
        telemetry: RadauTelemetryMode::Timings,
        t_bound: 1.0,
        ..RadauConfig::default()
    };
    let mut solver = RadauSolver::prepare(problem, config)?;
    let solution = solver.try_solve_with_parameter_values(&[1.0], &[2.0])?;
    println!(
        "Lambdify single solve: t={:.6} y={:?}",
        solution.t, solution.y
    );
    println!("timings_ms={:?}", solution.telemetry().timings_ms);
    Ok(())
}
