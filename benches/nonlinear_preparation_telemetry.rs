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

use std::fmt::Write as FmtWrite;
use std::hint::black_box;
use std::time::{Duration, Instant};

use nalgebra::DVector;
use tabled::{Table, Tabled};

use RustedSciThe::numerical::Nonlinear_systems::engine::{
    DiagnosticsOptions, NewtonMethod, SolveOptions, SolverEngine,
};
use RustedSciThe::numerical::Nonlinear_systems::prelude::{
    JacobianProvider, NonlinearProblem, PreparationStage, PreparationTelemetryMode,
    PreparedSymbolicNonlinearProblem, SymbolicLambdifyFrontend, SymbolicProblemOptions,
};

const RUNS: usize = 5;
const COLD_DIMENSIONS: &[usize] = &[3, 15, 40, 128, 256, 512];
const SOLVE_DIMENSIONS: &[usize] = &[3, 15, 40, 128];
const COLD_STAGES: &[PreparationStage] = &[
    PreparationStage::InputValidation,
    PreparationStage::ExpressionParsing,
    PreparationStage::AtomConversion,
    PreparationStage::AtomDependencyAnalysis,
    PreparationStage::AtomDifferentiation,
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

fn options(
    variables: &[String],
    mode: PreparationTelemetryMode,
    frontend: SymbolicLambdifyFrontend,
) -> SymbolicProblemOptions {
    SymbolicProblemOptions::new()
        .with_variables(variables.to_vec())
        .with_equation_parameters(vec!["p0".into(), "p1".into(), "p2".into()])
        .with_preparation_telemetry(mode)
        .with_lambdify_frontend(frontend)
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

#[derive(Tabled)]
struct ColdPreparationRow {
    frontend: String,
    dimension: usize,
    jacobian_diff_work: String,
    validation: String,
    parse: String,
    atom_conversion: String,
    atom_dependency: String,
    atom_differentiation: String,
    jacobian_diff: String,
    residual_callback: String,
    jacobian_callback: String,
    binding: String,
    assembly: String,
    unattributed: String,
    total: String,
}

#[derive(Tabled)]
struct ReuseRow {
    frontend: String,
    dimension: usize,
    rebind: String,
    residual: String,
    jacobian: String,
    repeated_total: String,
}

#[derive(Tabled)]
struct SolveRow {
    frontend: String,
    dimension: usize,
    solve: String,
    iterations: usize,
    residual_calls: usize,
    jacobian_calls: usize,
    linear_solves: usize,
    termination: String,
}

#[derive(Tabled)]
struct OverheadRow {
    frontend: String,
    dimension: usize,
    disabled: String,
    collect: String,
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

fn stage_work(prepared: &PreparedSymbolicNonlinearProblem, stage: PreparationStage) -> String {
    prepared
        .preparation_report()
        .detailed
        .as_ref()
        .and_then(|detailed| detailed.stage(stage))
        .map(|timing| {
            format!(
                "{}/{}",
                timing
                    .calls
                    .map_or_else(|| "n/a".to_owned(), |value| value.to_string()),
                timing
                    .items
                    .map_or_else(|| "n/a".to_owned(), |value| value.to_string())
            )
        })
        .unwrap_or_else(|| "n/a".to_owned())
}

fn main() {
    let mut cold_rows = Vec::new();

    for frontend in [
        SymbolicLambdifyFrontend::ExprLegacy,
        SymbolicLambdifyFrontend::AtomViewNative,
    ] {
        for &dimension in COLD_DIMENSIONS {
            let mut samples = [Duration::ZERO; RUNS];
            let mut stage_samples: [Vec<Duration>; COLD_STAGES.len()] =
                std::array::from_fn(|_| Vec::with_capacity(RUNS));
            let mut unattributed_samples = Vec::with_capacity(RUNS);
            let mut jacobian_diff_work = "n/a".to_owned();
            for sample in &mut samples {
                let (equations, variables, _) = equations(dimension);
                let started = Instant::now();
                let prepared = PreparedSymbolicNonlinearProblem::from_strings(
                    equations,
                    options(&variables, PreparationTelemetryMode::Collect, frontend),
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
                let work_stage = match frontend {
                    SymbolicLambdifyFrontend::ExprLegacy => {
                        PreparationStage::JacobianDifferentiation
                    }
                    SymbolicLambdifyFrontend::AtomViewNative => {
                        PreparationStage::AtomDifferentiation
                    }
                };
                jacobian_diff_work = stage_work(&prepared, work_stage);
            }
            cold_rows.push(ColdPreparationRow {
                frontend: frontend.as_str().to_owned(),
                dimension,
                jacobian_diff_work,
                validation: duration_summary(&stage_samples[0]),
                parse: duration_summary(&stage_samples[1]),
                atom_conversion: duration_summary(&stage_samples[2]),
                atom_dependency: duration_summary(&stage_samples[3]),
                atom_differentiation: duration_summary(&stage_samples[4]),
                jacobian_diff: duration_summary(&stage_samples[5]),
                residual_callback: duration_summary(&stage_samples[6]),
                jacobian_callback: duration_summary(&stage_samples[7]),
                binding: duration_summary(&stage_samples[8]),
                assembly: duration_summary(&stage_samples[9]),
                unattributed: duration_summary(&unattributed_samples),
                total: duration_summary(&samples),
            });
        }
    }

    let mut reuse_rows = Vec::new();
    for frontend in [
        SymbolicLambdifyFrontend::ExprLegacy,
        SymbolicLambdifyFrontend::AtomViewNative,
    ] {
        for &dimension in COLD_DIMENSIONS {
            let (equations, variables, target) = equations(dimension);
            let prepared = PreparedSymbolicNonlinearProblem::from_strings(
                equations,
                options(&variables, PreparationTelemetryMode::Collect, frontend),
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
            reuse_rows.push(ReuseRow {
                frontend: frontend.as_str().to_owned(),
                dimension,
                rebind: duration_summary(&rebind_samples),
                residual: duration_summary(&residual_samples),
                jacobian: duration_summary(&jacobian_samples),
                repeated_total: duration_summary(&total_samples),
            });
        }
    }

    let mut solve_rows = Vec::new();
    for frontend in [
        SymbolicLambdifyFrontend::ExprLegacy,
        SymbolicLambdifyFrontend::AtomViewNative,
    ] {
        for &dimension in SOLVE_DIMENSIONS {
            let (equations, variables, target) = equations(dimension);
            let prepared = PreparedSymbolicNonlinearProblem::from_strings(
                equations,
                options(&variables, PreparationTelemetryMode::Collect, frontend),
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
            solve_rows.push(SolveRow {
                frontend: frontend.as_str().to_owned(),
                dimension,
                solve: duration_summary(&solve_samples),
                iterations: statistics.iterations,
                residual_calls: statistics.residual_evaluations,
                jacobian_calls: statistics.jacobian_evaluations,
                linear_solves: statistics.linear_solves,
                termination: format!("{:?}", result.termination),
            });
        }
    }

    let mut overhead_rows = Vec::new();
    for frontend in [
        SymbolicLambdifyFrontend::ExprLegacy,
        SymbolicLambdifyFrontend::AtomViewNative,
    ] {
        for &dimension in &[40, 128, 512] {
            let mut disabled_samples = [Duration::ZERO; RUNS];
            let mut collect_samples = [Duration::ZERO; RUNS];
            for run in 0..RUNS {
                let (disabled_equations, disabled_variables, _) = equations(dimension);
                let started = Instant::now();
                black_box(
                    PreparedSymbolicNonlinearProblem::from_strings(
                        disabled_equations,
                        options(
                            &disabled_variables,
                            PreparationTelemetryMode::Disabled,
                            frontend,
                        ),
                    )
                    .expect("disabled telemetry preparation should succeed"),
                );
                disabled_samples[run] = started.elapsed();

                let (collect_equations, collect_variables, _) = equations(dimension);
                let started = Instant::now();
                black_box(
                    PreparedSymbolicNonlinearProblem::from_strings(
                        collect_equations,
                        options(
                            &collect_variables,
                            PreparationTelemetryMode::Collect,
                            frontend,
                        ),
                    )
                    .expect("enabled telemetry preparation should succeed"),
                );
                collect_samples[run] = started.elapsed();
            }
            overhead_rows.push(OverheadRow {
                frontend: frontend.as_str().to_owned(),
                dimension,
                disabled: duration_summary(&disabled_samples),
                collect: duration_summary(&collect_samples),
            });
        }
    }

    let mut report = String::new();
    writeln!(report, "# Nonlinear preparation telemetry").unwrap();
    writeln!(report, "\n- runs: {RUNS}").unwrap();
    writeln!(
        report,
        "- timing: median[min,max] milliseconds; tables are emitted after measurement"
    )
    .unwrap();
    writeln!(report, "\n## Table A: cold preparation").unwrap();
    report.push_str(&Table::new(&cold_rows).to_string());
    writeln!(report, "\n\n## Table B: prepared reuse").unwrap();
    writeln!(report, "Cold preparation is excluded.").unwrap();
    report.push_str(&Table::new(&reuse_rows).to_string());
    writeln!(report, "\n\n## Table C: prepared Newton attempts").unwrap();
    writeln!(report, "Preparation and binding are excluded.").unwrap();
    report.push_str(&Table::new(&solve_rows).to_string());
    writeln!(report, "\n\n## Table D: telemetry overhead").unwrap();
    writeln!(report, "Cold preparation only.").unwrap();
    report.push_str(&Table::new(&overhead_rows).to_string());

    match RustedSciThe::Utils::test_reporting::write_test_report(
        "Nonlinear_systems",
        "nonlinear_preparation_telemetry",
        &report,
    ) {
        Ok(path) => eprintln!(
            "[Nonlinear preparation telemetry] report={}",
            path.display()
        ),
        Err(error) => {
            eprintln!("[Nonlinear preparation telemetry] report write failed: {error}");
            std::process::exit(1);
        }
    }
}
