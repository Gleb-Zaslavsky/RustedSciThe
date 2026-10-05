//! Minimal public Radau AOT solve.
//!
//! The example exits cleanly when `tcc` is unavailable. AOT is explicit: it
//! never silently falls back to Lambdify.

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
        println!("Skipping Radau AOT example: tcc is not on PATH.");
        return Ok(());
    }
    let problem = RadauProblem::new(
        vec![Expr::parse_expression("-2.0*y")],
        vec!["y".to_owned()],
        "t",
    )
    .with_jacobian(vec![Expr::parse_expression("-2.0")]);
    let config = RadauConfig {
        execution: RadauExecution::Aot,
        frontend: RadauFrontend::AtomViewNative,
        matrix_layout: RadauMatrixLayout::Dense,
        telemetry: RadauTelemetryMode::Timings,
        aot: Some(RadauAotConfig::build_if_missing_release(
            "target/radau-aot-single",
        )),
        t_bound: 0.5,
        ..RadauConfig::default()
    };
    let mut solver = RadauSolver::prepare(problem, config)?;
    let solution = solver.solve(&[1.0])?;
    println!("AOT single solve: t={:.6} y={:?}", solution.t, solution.y);
    println!("AOT timings_ms={:?}", solution.telemetry().timings_ms);
    Ok(())
}
