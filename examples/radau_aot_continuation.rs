//! Reuse one published Radau AOT artifact for a parameter series.

use RustedSciThe::numerical::Radau::{
    RadauAotConfig, RadauConfig, RadauExecution, RadauFrontend, RadauMatrixLayout, RadauProblem,
    RadauSolver, RadauTelemetryMode,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !Command::new(if cfg!(windows) { "where" } else { "which" })
        .arg("tcc")
        .output()?
        .status
        .success()
    {
        println!("Skipping Radau AOT continuation example: tcc is not on PATH.");
        return Ok(());
    }
    let problem = RadauProblem::new(
        vec![Expr::parse_expression("-k*y")],
        vec!["y".to_owned()],
        "t",
    )
    .with_parameters(vec!["k".to_owned()])
    .with_jacobian(vec![Expr::parse_expression("-k")]);
    let config = RadauConfig {
        execution: RadauExecution::Aot,
        frontend: RadauFrontend::AtomViewNative,
        matrix_layout: RadauMatrixLayout::Dense,
        telemetry: RadauTelemetryMode::Counters,
        aot: Some(RadauAotConfig::build_if_missing_release(
            "target/radau-aot-continuation",
        )),
        t_bound: 1.0,
        ..RadauConfig::default()
    };
    let mut solver = RadauSolver::prepare(problem, config)?;
    for k in [0.5, 1.0, 2.0, 4.0] {
        let solution = solver.continue_with_parameters(&[1.0], &[k])?;
        let telemetry = solution.telemetry();
        println!(
            "k={k:.1} y(1)={:.9e} builds={} links={} rebinds={}",
            solution.y[0],
            telemetry.counters["aot_build_attempts"],
            telemetry.counters["aot_link_attempts"],
            telemetry.counters["parameter_rebinds"],
        );
    }
    Ok(())
}
