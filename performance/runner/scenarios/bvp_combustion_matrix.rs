use crate::error::RunnerError;
use crate::registry::ScenarioOutput;

#[path = "../../../benches/bvp_combustion_solver_matrix.rs"]
mod benchmark_matrix;

pub fn run() -> Result<ScenarioOutput, RunnerError> {
    let metrics = benchmark_matrix::perf_gate_metrics()
        .map_err(|error| RunnerError::Scenario(format!("combustion BVP matrix: {error}")))?;
    Ok(ScenarioOutput {
        metrics,
        ..ScenarioOutput::default()
    })
}
