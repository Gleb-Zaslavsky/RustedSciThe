//! Release-oriented preparation/reuse telemetry tables for prepared nonlinear systems.
//!
//! This target is intentionally a small reporting executable rather than a
//! Criterion benchmark. It keeps cold preparation, prepared reuse, and solver
//! attempts in separate tables. Exact numbers are machine-dependent evidence,
//! not correctness thresholds.
//!
//! Run with:
//!
//! ```text
//! cargo bench --bench nonlinear_preparation_telemetry -- --noplot
//! ```

use std::hint::black_box;
use std::time::{Duration, Instant};

use nalgebra::DVector;

use RustedSciThe::numerical::Nonlinear_systems::engine::{
    DiagnosticsOptions, NewtonMethod, SolveOptions, SolverEngine,
};
use RustedSciThe::numerical::Nonlinear_systems::prelude::{
    JacobianProvider, NonlinearProblem, PreparationStage, PreparationTelemetryMode,
    PreparedSymbolicNonlinearProblem, SymbolicProblemOptions,
};

const RUNS: usize = 5;
const COLD_DIMENSIONS: &[usize] = &[3, 15, 40, 128, 256, 512];
const SOLVE_DIMENSIONS: &[usize] = &[3, 15, 40, 128];
const COLD_STAGES: &[PreparationStage] = &[
    PreparationStage::InputValidation,
    PreparationStage::ExpressionParsing,
    PreparationStage::JacobianDifferentiation,
    PreparationStage::ResidualCallbackPreparation,
    PreparationStage::JacobianCallbackPreparation,
    PreparationStage::ParameterBinding,
    PreparationStage::PreparedProblemAssembly,
];

fn equations(dimension: usize) -> (Vec<String>, Vec<String>, DVector<f64>) {
    let variables = (0..dimension)
        .map(|index| format!("x{index}"))
        .collect::<Vec<_>>();
    let target = DVector::from_iterator(
        dimension,
        (0..dimension).map(|index| 1.0 + index as f64 * 0.001),
    );
    let equations = variables
        .iter()
        .enumerate()
        .map(|(index, variable)| {
            format!(
                "{variable} + 0.1*p0 + 0.01*p1 + 0.001*p2 - {}",
                target[index]
            )
        })
        .collect::<Vec<_>>();
    (equations, variables, target)
}

fn options(variables: &[String], mode: PreparationTelemetryMode) -> SymbolicProblemOptions {
    SymbolicProblemOptions::new()
        .with_variables(variables.to_vec())
        .with_equation_parameters(vec!["p0".into(), "p1".into(), "p2".into()])
        .with_preparation_telemetry(mode)
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn duration_summary(samples: &[Duration]) -> String {
    if samples.is_empty() {
        return "n/a".to_string();
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    format!(
        "{:.3}[{:.3},{:.3}]",
        ms(sorted[sorted.len() / 2]),
        ms(sorted[0]),
        ms(sorted[sorted.len() - 1]),
    )
}

fn stage_duration(
    prepared: &PreparedSymbolicNonlinearProblem,
    stage: PreparationStage,
) -> Option<Duration> {
    prepared
        .preparation_report()
        .detailed
        .as_ref()?
        .stage(stage)?
        .wall_time
}

fn main() {
    println!("[Nonlinear preparation telemetry] runs={RUNS}");
    println!("Table A: cold preparation; times are median[min,max] milliseconds");
    println!(
        "dimension | validation median[min,max] | parse median[min,max] | jacobian_diff median[min,max] | residual_cb median[min,max] | jacobian_cb median[min,max] | binding median[min,max] | assembly median[min,max] | unattributed median[min,max] | total median[min,max]"
    );

    for &dimension in COLD_DIMENSIONS {
        let mut samples = [Duration::ZERO; RUNS];
        let mut stage_samples: [Vec<Duration>; COLD_STAGES.len()] =
            std::array::from_fn(|_| Vec::with_capacity(RUNS));
        let mut unattributed_samples = Vec::with_capacity(RUNS);
        for sample in &mut samples {
            let (equations, variables, _) = equations(dimension);
            let started = Instant::now();
            let prepared = PreparedSymbolicNonlinearProblem::from_strings(
                equations,
                options(&variables, PreparationTelemetryMode::Collect),
            )
            .expect("telemetry benchmark preparation should succeed");
            *sample = started.elapsed();
            for (index, stage) in COLD_STAGES.iter().copied().enumerate() {
                if let Some(duration) = stage_duration(&prepared, stage) {
                    stage_samples[index].push(duration);
                }
            }
            unattributed_samples.push(
                prepared
                    .preparation_report()
                    .detailed
                    .as_ref()
                    .expect("detailed preparation report")
                    .unattributed_wall_time,
            );
        }
        println!(
            "{dimension:>9} | {} | {} | {} | {} | {} | {} | {} | {} | {}",
            duration_summary(&stage_samples[0]),
            duration_summary(&stage_samples[1]),
            duration_summary(&stage_samples[2]),
            duration_summary(&stage_samples[3]),
            duration_summary(&stage_samples[4]),
            duration_summary(&stage_samples[5]),
            duration_summary(&stage_samples[6]),
            duration_summary(&unattributed_samples),
            duration_summary(&samples),
        );
    }

    println!("Table B: prepared reuse; cold preparation excluded");
    println!("dimension | rebind_ms | residual_ms | jacobian_ms | repeated_total_ms");
    for &dimension in COLD_DIMENSIONS {
        let (equations, variables, target) = equations(dimension);
        let prepared = PreparedSymbolicNonlinearProblem::from_strings(
            equations,
            options(&variables, PreparationTelemetryMode::Collect),
        )
        .expect("reuse benchmark preparation should succeed");
        let x = DVector::from_element(dimension, 0.0);
        let mut rebind_samples = [Duration::ZERO; RUNS];
        let mut residual_samples = [Duration::ZERO; RUNS];
        let mut jacobian_samples = [Duration::ZERO; RUNS];
        let mut total_samples = [Duration::ZERO; RUNS];
        for run in 0..RUNS {
            let bind_started = Instant::now();
            let bound = prepared
                .bind_values(DVector::from_vec(vec![0.5, 0.25, 0.125]))
                .expect("rebind should succeed");
            rebind_samples[run] = bind_started.elapsed();
            let residual_started = Instant::now();
            black_box(bound.residual(&x).expect("residual should evaluate"));
            residual_samples[run] = residual_started.elapsed();
            let jacobian_started = Instant::now();
            black_box(bound.jacobian(&x).expect("Jacobian should evaluate"));
            jacobian_samples[run] = jacobian_started.elapsed();
            let total_started = Instant::now();
            let bound = prepared
                .bind_values(DVector::from_vec(vec![target[0], target[0], target[0]]))
                .expect("second rebind should succeed");
            black_box(
                bound
                    .residual(&x)
                    .expect("repeated residual should evaluate"),
            );
            black_box(
                bound
                    .jacobian(&x)
                    .expect("repeated Jacobian should evaluate"),
            );
            total_samples[run] = total_started.elapsed();
        }
        println!(
            "{dimension:>9} | {} | {} | {} | {}",
            duration_summary(&rebind_samples),
            duration_summary(&residual_samples),
            duration_summary(&jacobian_samples),
            duration_summary(&total_samples),
        );
    }

    println!("Table C: prepared Newton attempts; preparation and binding excluded");
    println!(
        "dimension | solve_ms | iterations | residual_calls | jacobian_calls | linear_solves | termination"
    );
    for &dimension in SOLVE_DIMENSIONS {
        let (equations, variables, target) = equations(dimension);
        let prepared = PreparedSymbolicNonlinearProblem::from_strings(
            equations,
            options(&variables, PreparationTelemetryMode::Collect),
        )
        .expect("solve benchmark preparation should succeed");
        let bound = prepared
            .bind_values(DVector::from_vec(vec![0.5, 0.25, 0.125]))
            .expect("solve binding should succeed");
        let mut solve_samples = [Duration::ZERO; RUNS];
        let mut last_result = None;
        for sample in &mut solve_samples {
            let started = Instant::now();
            let result = SolverEngine::new(
                NewtonMethod,
                SolveOptions {
                    tolerance: 1e-10,
                    max_iterations: 8,
                    diagnostics: DiagnosticsOptions {
                        collect_history: false,
                        collect_statistics: true,
                        ..DiagnosticsOptions::default()
                    },
                    ..SolveOptions::default()
                },
            )
            .solve(&bound, DVector::from_element(dimension, 0.0));
            *sample = started.elapsed();
            last_result = Some(result.expect("Newton benchmark solve should succeed"));
        }
        let result = last_result.expect("one solve result");
        let statistics = &result.statistics;
        black_box(target);
        println!(
            "{dimension:>9} | {} | {:>10} | {:>14} | {:>14} | {:>13} | {:?}",
            duration_summary(&solve_samples),
            statistics.iterations,
            statistics.residual_evaluations,
            statistics.jacobian_evaluations,
            statistics.linear_solves,
            result.termination,
        );
    }

    println!("Table D: detailed telemetry overhead; cold preparation only");
    println!("dimension | disabled median[min,max] | collect median[min,max]");
    for &dimension in &[40, 128, 512] {
        let mut disabled_samples = [Duration::ZERO; RUNS];
        let mut collect_samples = [Duration::ZERO; RUNS];
        for run in 0..RUNS {
            let (disabled_equations, disabled_variables, _) = equations(dimension);
            let started = Instant::now();
            black_box(
                PreparedSymbolicNonlinearProblem::from_strings(
                    disabled_equations,
                    options(&disabled_variables, PreparationTelemetryMode::Disabled),
                )
                .expect("disabled telemetry preparation should succeed"),
            );
            disabled_samples[run] = started.elapsed();

            let (collect_equations, collect_variables, _) = equations(dimension);
            let started = Instant::now();
            black_box(
                PreparedSymbolicNonlinearProblem::from_strings(
                    collect_equations,
                    options(&collect_variables, PreparationTelemetryMode::Collect),
                )
                .expect("enabled telemetry preparation should succeed"),
            );
            collect_samples[run] = started.elapsed();
        }
        println!(
            "{dimension:>9} | {} | {}",
            duration_summary(&disabled_samples),
            duration_summary(&collect_samples),
        );
    }
}
