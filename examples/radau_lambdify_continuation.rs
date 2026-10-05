//! Reuse one prepared Lambdify model for a parameter series.

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
    .with_parameters(vec!["k".to_owned()])
    .with_jacobian(vec![Expr::parse_expression("-k")]);
    let config = RadauConfig {
        frontend: RadauFrontend::ExprLegacy,
        matrix_layout: RadauMatrixLayout::Dense,
        telemetry: RadauTelemetryMode::Counters,
        t_bound: 1.0,
        ..RadauConfig::default()
    };
    let mut solver = RadauSolver::prepare(problem, config)?;
    for k in [0.5, 1.0, 2.0, 4.0] {
        let solution = solver.continue_with_parameters(&[1.0], &[k])?;
        println!("k={k:.1} y(1)={:.9e}", solution.y[0]);
    }
    Ok(())
}
