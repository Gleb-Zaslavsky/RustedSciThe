use crate::error::RunnerError;
use crate::registry::ScenarioOutput;
use RustedSciThe::numerical::BDF::BDF_api::{BdfSolverOptions, BdfTelemetryMode, ODEsolver};
use RustedSciThe::numerical::BE::{BE, BeSolverOptions, BeTelemetryMode};
use RustedSciThe::numerical::LSODE2::{
    IvpTelemetry, Lsode2ProblemConfig, Lsode2ResidualJacobianSource, Lsode2Solver,
    Lsode2SymbolicAssemblyBackend, Lsode2SymbolicExecutionMode,
};
use RustedSciThe::numerical::Radau::api::{RadauConfig, RadauProblem, RadauSolver};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DVector;
use std::hint::black_box;
use std::time::Instant;

const POINTS: usize = 512;
const T_BOUND: f64 = 0.5;

fn parameter_vectors() -> Vec<Vec<f64>> {
    (0..POINTS)
        .map(|index| vec![0.5 + index as f64 * 0.01])
        .collect()
}

fn finish(
    started: Instant,
    solver: &'static str,
    final_states: impl IntoIterator<Item = f64>,
) -> Result<ScenarioOutput, RunnerError> {
    let final_states = final_states.into_iter().collect::<Vec<_>>();
    if final_states.len() != POINTS || final_states.iter().any(|value| !value.is_finite()) {
        return Err(RunnerError::Scenario(format!(
            "{solver} parameter loop returned missing or non-finite solutions"
        )));
    }
    let checksum = final_states.iter().sum::<f64>();
    black_box(&final_states);
    Ok(ScenarioOutput {
        metrics: vec![(
            "scenario_wall_ms".into(),
            "ms/off".into(),
            started.elapsed().as_secs_f64() * 1000.0,
        )],
        counters: vec![("parameter_points".into(), POINTS as u64)],
        diagnostics: vec![("final_state_checksum".into(), "state".into(), checksum)],
        ..ScenarioOutput::default()
    })
}

pub fn bdf() -> Result<ScenarioOutput, RunnerError> {
    let options = BdfSolverOptions::for_bdf(
        vec![Expr::parse_expression("-rate*y")],
        vec!["y".into()],
        "t".into(),
        0.0,
        DVector::from_element(1, 1.0),
        T_BOUND,
        0.01,
        1.0e-8,
        1.0e-10,
        None,
        false,
        Some(0.005),
    )
    .with_equation_parameters(vec!["rate".into()])
    .with_equation_parameter_values(DVector::from_element(1, 1.0))
    .with_telemetry_mode(BdfTelemetryMode::Off);
    let mut solver = ODEsolver::new_with_options(options);
    solver
        .try_generate()
        .map_err(|error| RunnerError::Scenario(format!("BDF preparation failed: {error}")))?;
    let values = parameter_vectors();
    let started = Instant::now();
    let report = solver.try_solve_parameter_loop(DVector::from_element(1, 1.0), &values);
    let finals = report
        .points
        .into_iter()
        .map(|point| {
            let rate = point.parameter_values[0];
            point
                .result
                .map(|solution| {
                    let final_value = solution.values[(solution.values.nrows() - 1, 0)];
                    check_exact_final("BDF", rate, final_value, 2.0e-4)?;
                    Ok(final_value)
                })
                .map_err(|error| RunnerError::Scenario(format!("BDF point failed: {error}")))
                .and_then(|value| value)
        })
        .collect::<Result<Vec<_>, _>>()?;
    finish(started, "BDF", finals)
}

pub fn backward_euler() -> Result<ScenarioOutput, RunnerError> {
    let options = BeSolverOptions::new(
        vec![Expr::parse_expression("-rate*y")],
        vec!["y".into()],
        "t".into(),
        1.0e-8,
        100,
        Some(0.0025),
        0.0,
        T_BOUND,
        DVector::from_element(1, 1.0),
    )
    .with_telemetry_mode(BeTelemetryMode::Off);
    let mut solver = BE::try_new_with_options(options)
        .map_err(|error| RunnerError::Scenario(format!("BE preparation failed: {error}")))?;
    solver
        .try_set_equation_parameters(Some(&["rate"]))
        .map_err(|error| RunnerError::Scenario(format!("BE parameter setup failed: {error}")))?;
    let values = parameter_vectors();
    let started = Instant::now();
    let report = solver.try_solve_parameter_loop(DVector::from_element(1, 1.0), &values);
    let finals = report.points.into_iter().map(|point| {
        let rate = point.parameter_values[0];
        point
            .result
            .map(|solution| {
                let final_value = solution.values[(solution.values.nrows() - 1, 0)];
                let expected = (1.0 + 0.0025 * rate).powi(-200);
                if (final_value - expected).abs() > 1.0e-10 {
                    return Err(RunnerError::Scenario(format!(
                        "BE loop final value differs from fixed-step reference: rate={rate}, got={final_value}, expected={expected}"
                    )));
                }
                Ok(final_value)
            })
            .map_err(|error| RunnerError::Scenario(format!("BE point failed: {error}")))
            .and_then(|value| value)
    }).collect::<Result<Vec<_>, _>>()?;
    finish(started, "BE", finals)
}

pub fn lsode2() -> Result<ScenarioOutput, RunnerError> {
    let config = Lsode2ProblemConfig::new(
        vec![Expr::parse_expression("-rate*y")],
        vec!["y".into()],
        "t".into(),
        0.0,
        DVector::from_element(1, 1.0),
        T_BOUND,
        0.01,
        1.0e-8,
        1.0e-10,
    )
    .with_equation_parameters(vec!["rate".into()])
    .with_equation_parameter_values(DVector::from_element(1, 1.0))
    .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
        assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
        execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
    })
    .with_bridge_solve()
    .with_telemetry(IvpTelemetry::disabled());
    let mut solver = Lsode2Solver::new(config)
        .map_err(|error| RunnerError::Scenario(format!("LSODE2 preparation failed: {error}")))?;
    let values = parameter_vectors();
    let started = Instant::now();
    let report = solver.try_solve_parameter_loop(DVector::from_element(1, 1.0), &values);
    let finals = report
        .points
        .into_iter()
        .map(|point| {
            let rate = point.parameter_values[0];
            point
                .result
                .map(|solution| {
                    let final_value = solution.values[(solution.values.nrows() - 1, 0)];
                    check_exact_final("LSODE2", rate, final_value, 2.0e-3)?;
                    Ok(final_value)
                })
                .map_err(|error| RunnerError::Scenario(format!("LSODE2 point failed: {error}")))
                .and_then(|value| value)
        })
        .collect::<Result<Vec<_>, _>>()?;
    finish(started, "LSODE2", finals)
}

pub fn radau() -> Result<ScenarioOutput, RunnerError> {
    let problem = RadauProblem::new(
        vec![Expr::parse_expression("-rate*y")],
        vec!["y".into()],
        "t",
    )
    .with_parameter_names(vec!["rate".into()]);
    let config = RadauConfig {
        t_bound: T_BOUND,
        max_step: 0.01,
        ..RadauConfig::default()
    };
    let mut solver = RadauSolver::prepare(problem, config)
        .map_err(|error| RunnerError::Scenario(format!("Radau preparation failed: {error}")))?;
    let values = parameter_vectors();
    let started = Instant::now();
    let report = solver.try_solve_parameter_loop(&[1.0], &values);
    let finals = report
        .points
        .into_iter()
        .map(|point| {
            let rate = point.parameter_values[0];
            point
                .result
                .map(|solution| {
                    let final_value = solution.values[(solution.values.nrows() - 1, 0)];
                    check_exact_final("Radau", rate, final_value, 2.0e-4)?;
                    Ok(final_value)
                })
                .map_err(|error| RunnerError::Scenario(format!("Radau point failed: {error}")))
                .and_then(|value| value)
        })
        .collect::<Result<Vec<_>, _>>()?;
    finish(started, "Radau", finals)
}

fn check_exact_final(
    solver: &str,
    rate: f64,
    actual: f64,
    tolerance: f64,
) -> Result<(), RunnerError> {
    let expected = (-rate * T_BOUND).exp();
    if (actual - expected).abs() > tolerance {
        return Err(RunnerError::Scenario(format!(
            "{solver} parameter loop failed exact-solution check: rate={rate}, got={actual}, expected={expected}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_solver_loop_matches_its_independent_scalar_reference() {
        for (solver, run) in [
            ("BDF", bdf as fn() -> Result<ScenarioOutput, RunnerError>),
            ("BE", backward_euler),
            ("LSODE2", lsode2),
            ("Radau", radau),
        ] {
            let output = run().unwrap_or_else(|error| panic!("{solver}: {error}"));
            assert_eq!(
                output.counters,
                [("parameter_points".into(), POINTS as u64)]
            );
            assert_eq!(output.metrics.len(), 1);
            assert!(output.metrics[0].2.is_finite() && output.metrics[0].2 > 0.0);
        }
    }
}
