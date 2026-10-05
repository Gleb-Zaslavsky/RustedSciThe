//! Compare public Radau configuration choices without using the archived API.

use RustedSciThe::numerical::Radau::{
    RadauConfig, RadauFrontend, RadauMatrixLayout, RadauProblem, RadauSolver, RadauTelemetryMode,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let problem = RadauProblem::new(
        vec![Expr::parse_expression("-2.0*y")],
        vec!["y".to_owned()],
        "t",
    )
    .with_jacobian(vec![Expr::parse_expression("-2.0")]);

    for (label, frontend, layout) in [
        (
            "ExprLegacy/Dense",
            RadauFrontend::ExprLegacy,
            RadauMatrixLayout::Dense,
        ),
        (
            "AtomViewNative/Dense",
            RadauFrontend::AtomViewNative,
            RadauMatrixLayout::Dense,
        ),
        (
            "AtomViewNative/Sparse",
            RadauFrontend::AtomViewNative,
            RadauMatrixLayout::Sparse,
        ),
        (
            "AtomViewNative/Banded",
            RadauFrontend::AtomViewNative,
            RadauMatrixLayout::Banded { lower: 1, upper: 1 },
        ),
    ] {
        let config = RadauConfig {
            frontend,
            matrix_layout: layout,
            telemetry: RadauTelemetryMode::Counters,
            t_bound: 0.1,
            ..RadauConfig::default()
        };
        let mut solver = RadauSolver::prepare(problem.clone(), config)?;
        let solution = solver.solve(&[1.0])?;
        println!("{label}: t={:.6}, y={:?}", solution.t, solution.y);
        println!("  counters={:?}", solution.telemetry().counters);
    }

    println!("No layout is universally fastest: measure the workload-matched route.");
    Ok(())
}
