//! Public Radau API: selectable Lambdify frontend, layout, continuation,
//! telemetry, and dense output.

use std::error::Error;

use RustedSciThe::numerical::Radau::{
    RadauConfig, RadauFrontend, RadauMatrixLayout, RadauOutputPolicy, RadauProblem, RadauSolver,
    RadauTelemetryMode,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;

fn main() -> Result<(), Box<dyn Error>> {
    let problem = RadauProblem::new(
        vec![Expr::parse_expression("-a*y")],
        vec!["y".to_owned()],
        "t",
    )
    .with_parameters(vec!["a".to_owned()])
    .with_jacobian(vec![Expr::parse_expression("-a")]);

    let config = RadauConfig {
        frontend: RadauFrontend::AtomViewNative,
        matrix_layout: RadauMatrixLayout::Dense,
        telemetry: RadauTelemetryMode::Counters,
        output: RadauOutputPolicy::Dense,
        t_bound: 1.0,
        ..RadauConfig::default()
    };
    let mut solver = RadauSolver::prepare(problem, config)?;

    let first = solver.solve_with_parameters(&[1.0], &[1.0])?;
    let continued = solver.continue_with_parameters(&[1.0], &[2.0])?;
    let midpoint = continued.sample(0.5)?;

    println!("first final y = {:?}", first.y);
    println!("continued final y = {:?}", continued.y);
    println!("continued y(0.5) = {:?}", midpoint);
    println!("telemetry counters = {:?}", continued.telemetry().counters);
    Ok(())
}
