//! Compact dense frontend matrix for nonlinear systems.
//!
//! This complements `nonlinear_workloads`: it uses the richer nonlinear
//! corpus (quadratic chain, nonlinear Poisson, and five-point band) and keeps
//! preparation, callback, solve, and continuation measurements in one table.
//! AOT is opt-in because each row may materialize and compile a dense crate.

use std::fmt::Write as _;
use std::hint::black_box;
use std::time::Instant;

use nalgebra::DVector;
use tabled::{Table, Tabled};
use tempfile::TempDir;

use RustedSciThe::Utils::test_reporting::write_test_report;
use RustedSciThe::numerical::Nonlinear_systems::prelude::{
    DiagnosticsOptions, LambdifyExecutionPolicy, NewtonMethod, NonlinearSolverMethod,
    PreparationStage, PreparationTelemetryMode, PreparedSymbolicNonlinearProblem, SolveOptions,
    SymbolicAotBuildPolicy, SymbolicGeneratedBackendConfig, SymbolicLambdifyFrontend,
    SymbolicNonlinearProblem, SymbolicProblemOptions,
};
use RustedSciThe::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;

#[derive(Clone, Copy)]
enum Route {
    Lambdify(SymbolicLambdifyFrontend),
    Aot(SymbolicLambdifyFrontend),
}

impl Route {
    fn label(self) -> &'static str {
        match self {
            Self::Lambdify(SymbolicLambdifyFrontend::ExprLegacy) => "lambdify/expr-legacy",
            Self::Lambdify(SymbolicLambdifyFrontend::AtomViewNative) => "lambdify/atom-native",
            Self::Aot(SymbolicLambdifyFrontend::ExprLegacy) => "aot/expr-legacy",
            Self::Aot(SymbolicLambdifyFrontend::AtomViewNative) => "aot/atom-native",
        }
    }
}

#[derive(Clone, Copy)]
enum Policy {
    Sequential,
    Parallel,
}

impl Policy {
    fn label(self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
            Self::Parallel => "parallel",
        }
    }

    fn execution(self) -> LambdifyExecutionPolicy {
        match self {
            Self::Sequential => LambdifyExecutionPolicy::Sequential,
            Self::Parallel => LambdifyExecutionPolicy::Parallel { min_work: 1 },
        }
    }
}

#[derive(Clone, Copy)]
enum Workload {
    QuadraticChain,
    NonlinearPoisson,
    BandFive,
}

impl Workload {
    fn label(self) -> &'static str {
        match self {
            Self::QuadraticChain => "quadratic-chain",
            Self::NonlinearPoisson => "nonlinear-poisson",
            Self::BandFive => "band-five",
        }
    }

    fn target(self, index: usize) -> f64 {
        match self {
            Self::QuadraticChain => 1.0 + index as f64 * 0.01,
            Self::NonlinearPoisson => 0.2 + index as f64 * 0.001,
            Self::BandFive => 0.5 + index as f64 * 0.003,
        }
    }

    fn equations(self, dimension: usize) -> (Vec<String>, Vec<String>, DVector<f64>, DVector<f64>) {
        let variables = (0..dimension)
            .map(|index| format!("x{index}"))
            .collect::<Vec<_>>();
        let target =
            DVector::from_iterator(dimension, (0..dimension).map(|index| self.target(index)));
        let equations = variables
            .iter()
            .enumerate()
            .map(|(index, variable)| {
                let mut equation = match self {
                    Self::QuadraticChain => {
                        format!("p0*({variable}^2-{:.17})", target[index].powi(2))
                    }
                    Self::NonlinearPoisson => {
                        format!("p0*({variable}^2-{:.17})", target[index].powi(2))
                    }
                    Self::BandFive => {
                        format!("p0*({variable}^3-{:.17})", target[index].powi(3))
                    }
                };
                let radius = match self {
                    Self::QuadraticChain => 1,
                    Self::NonlinearPoisson => 2,
                    Self::BandFive => 2,
                };
                for neighbor in 1..=radius {
                    if index >= neighbor {
                        equation.push_str(&format!(
                            "+0.02*(x{}-{:.17})",
                            index - neighbor,
                            target[index - neighbor]
                        ));
                    }
                    if index + neighbor < dimension {
                        equation.push_str(&format!(
                            "+0.02*(x{}-{:.17})",
                            index + neighbor,
                            target[index + neighbor]
                        ));
                    }
                }
                equation
            })
            .collect::<Vec<_>>();
        (
            equations,
            variables,
            target.clone(),
            target.map(|value| value * 0.9),
        )
    }
}

#[derive(Tabled)]
struct Row {
    workload: String,
    route: String,
    lifecycle: String,
    policy: String,
    dimension: usize,
    continuation_count: usize,
    prepare_ms: String,
    parse_ms: String,
    diff_ms: String,
    atom_ms: String,
    callback_ms: String,
    aot_ms: String,
    residual_ms: String,
    jacobian_ms: String,
    solve_ms: String,
    continuation_ms: String,
    residual_calls: usize,
    jacobian_calls: usize,
    status: String,
}

struct PreparedCase {
    problem: PreparedSymbolicNonlinearProblem,
    artifact_dir: Option<TempDir>,
    prepare_ms: f64,
    lifecycle: String,
}

fn dimensions() -> Vec<usize> {
    std::env::var("NONLINEAR_FRONTEND_DIMENSIONS")
        .unwrap_or_else(|_| "16,64,128".to_owned())
        .split(',')
        .filter_map(|value| value.trim().parse().ok())
        .filter(|dimension: &usize| *dimension > 0)
        .collect()
}

fn continuation_counts() -> Vec<usize> {
    let counts = std::env::var("NONLINEAR_FRONTEND_CONTINUATION")
        .unwrap_or_else(|_| "4".to_owned())
        .split(',')
        .filter_map(|value| value.trim().parse().ok())
        .filter(|count: &usize| *count > 0)
        .collect::<Vec<_>>()
        .into_iter()
        .fold(Vec::new(), |mut counts, count| {
            if !counts.contains(&count) {
                counts.push(count);
            }
            counts
        });
    if counts.is_empty() { vec![4] } else { counts }
}

fn aot_build_policy() -> (SymbolicAotBuildPolicy, &'static str) {
    match std::env::var("NONLINEAR_FRONTEND_AOT_LIFECYCLE")
        .unwrap_or_else(|_| "build-if-missing".to_owned())
        .to_ascii_lowercase()
        .as_str()
    {
        "rebuild-always" | "rebuild_always" | "cold" => (
            SymbolicAotBuildPolicy::RebuildAlways {
                profile: AotBuildProfile::Release,
            },
            "rebuild-always",
        ),
        _ => (
            SymbolicAotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Release,
            },
            "build-if-missing",
        ),
    }
}

fn include_aot() -> bool {
    std::env::var("NONLINEAR_FRONTEND_INCLUDE_AOT").as_deref() == Ok("1")
}

fn options(
    variables: Vec<String>,
    frontend: SymbolicLambdifyFrontend,
    policy: Policy,
) -> SymbolicProblemOptions {
    SymbolicProblemOptions::new()
        .with_variables(variables)
        .with_equation_parameters(vec!["p0".to_owned()])
        .with_equation_parameter_values(DVector::from_vec(vec![1.0]))
        .with_lambdify_frontend(frontend)
        .with_lambdify_execution_policy(policy.execution())
        .with_preparation_telemetry(PreparationTelemetryMode::Collect)
}

fn prepare(
    workload: Workload,
    dimension: usize,
    route: Route,
    policy: Policy,
) -> Result<PreparedCase, String> {
    let (equations, variables, _, _) = workload.equations(dimension);
    let started = Instant::now();
    match route {
        Route::Lambdify(frontend) => PreparedSymbolicNonlinearProblem::from_strings(
            equations,
            options(variables, frontend, policy),
        )
        .map(|problem| PreparedCase {
            problem,
            artifact_dir: None,
            prepare_ms: started.elapsed().as_secs_f64() * 1e3,
            lifecycle: "not-applicable".to_owned(),
        })
        .map_err(|error| error.to_string()),
        Route::Aot(frontend) => {
            let artifact_dir = tempfile::tempdir().map_err(|error| error.to_string())?;
            let (build_policy, policy_label) = aot_build_policy();
            let generated = SymbolicNonlinearProblem::from_strings_with_generated_backend(
                equations,
                options(variables, frontend, policy),
                SymbolicGeneratedBackendConfig::defaults()
                    .with_build_policy(build_policy)
                    .with_output_parent_dir(Some(artifact_dir.path().to_path_buf())),
            )
            .map_err(|error| error.to_string())?;
            Ok(PreparedCase {
                problem: generated.into_prepared(),
                artifact_dir: Some(artifact_dir),
                prepare_ms: started.elapsed().as_secs_f64() * 1e3,
                lifecycle: policy_label.to_owned(),
            })
        }
    }
}

fn stage_ms(problem: &PreparedSymbolicNonlinearProblem, stage: PreparationStage) -> String {
    problem
        .preparation_report()
        .detailed
        .as_ref()
        .and_then(|report| report.stage(stage))
        .and_then(|timing| timing.wall_time)
        .map(|duration| format!("{:.3}", duration.as_secs_f64() * 1e3))
        .unwrap_or_else(|| "n/a".to_owned())
}

fn solve_options() -> SolveOptions {
    SolveOptions {
        tolerance: 1e-10,
        max_iterations: 80,
        diagnostics: DiagnosticsOptions {
            collect_statistics: true,
            ..DiagnosticsOptions::default()
        },
        ..SolveOptions::default()
    }
}

fn main() {
    let dimensions = dimensions();
    let continuation_counts = continuation_counts();
    let routes = [
        Route::Lambdify(SymbolicLambdifyFrontend::ExprLegacy),
        Route::Lambdify(SymbolicLambdifyFrontend::AtomViewNative),
    ];
    let mut routes = routes.to_vec();
    if include_aot() {
        routes.extend([
            Route::Aot(SymbolicLambdifyFrontend::ExprLegacy),
            Route::Aot(SymbolicLambdifyFrontend::AtomViewNative),
        ]);
    }
    let mut rows = Vec::new();
    let mut failures = Vec::new();

    for workload in [
        Workload::QuadraticChain,
        Workload::NonlinearPoisson,
        Workload::BandFive,
    ] {
        for &dimension in &dimensions {
            for &route in &routes {
                for &continuation in &continuation_counts {
                    for policy in [Policy::Sequential, Policy::Parallel] {
                        let (_, _, target, initial) = workload.equations(dimension);
                        let case = match prepare(workload, dimension, route, policy) {
                            Ok(case) => case,
                            Err(error) => {
                                failures.push(format!(
                                    "prepare/{}/{}/{}/{}: {error}",
                                    workload.label(),
                                    route.label(),
                                    dimension,
                                    continuation
                                ));
                                continue;
                            }
                        };
                        let bound = match case.problem.bind_values(DVector::from_vec(vec![1.0])) {
                            Ok(bound) => bound,
                            Err(error) => {
                                failures.push(format!(
                                    "bind/{}/{}/{}/{}: {error}",
                                    workload.label(),
                                    route.label(),
                                    dimension,
                                    continuation
                                ));
                                continue;
                            }
                        };
                        let solve_started = Instant::now();
                        let result = match NonlinearSolverMethod::Newton(NewtonMethod).solve(
                            &bound,
                            initial.clone(),
                            solve_options(),
                        ) {
                            Ok(result) => result,
                            Err(error) => {
                                failures.push(format!(
                                    "solve/{}/{}/{}/{}: {error}",
                                    workload.label(),
                                    route.label(),
                                    dimension,
                                    continuation
                                ));
                                continue;
                            }
                        };
                        black_box(&result.x);
                        let initial_solve_elapsed = solve_started.elapsed();
                        let mut continuation_state = result.x.clone();
                        let continuation_started = Instant::now();
                        for step in 0..continuation.saturating_sub(1) {
                            let bound = case
                                .problem
                                .bind_values(DVector::from_vec(vec![0.75 + step as f64 * 0.25]))
                                .expect("continuation parameter should bind");
                            continuation_state = NonlinearSolverMethod::Newton(NewtonMethod)
                                .solve(&bound, continuation_state, solve_options())
                                .expect("continuation solve should succeed")
                                .x;
                        }
                        let statistics = result.statistics;
                        let lifecycle = if matches!(route, Route::Aot(_)) {
                            match case.lifecycle.as_str() {
                                "rebuild-always" => "cold-rebuild".to_owned(),
                                "build-if-missing"
                                    if case
                                        .problem
                                        .preparation_report()
                                        .build_duration
                                        .is_some() =>
                                {
                                    "cold-build-if-missing".to_owned()
                                }
                                "build-if-missing" => "warm-cache-hit".to_owned(),
                                other => other.to_owned(),
                            }
                        } else {
                            case.lifecycle.clone()
                        };
                        rows.push(Row {
                            workload: workload.label().to_owned(),
                            route: route.label().to_owned(),
                            lifecycle,
                            policy: policy.label().to_owned(),
                            dimension,
                            continuation_count: continuation,
                            prepare_ms: format!("{:.3}", case.prepare_ms),
                            parse_ms: stage_ms(&case.problem, PreparationStage::ExpressionParsing),
                            diff_ms: stage_ms(
                                &case.problem,
                                PreparationStage::JacobianDifferentiation,
                            ),
                            atom_ms: stage_ms(&case.problem, PreparationStage::AtomDifferentiation),
                            callback_ms: stage_ms(
                                &case.problem,
                                PreparationStage::JacobianCallbackPreparation,
                            ),
                            aot_ms: case
                                .problem
                                .preparation_report()
                                .build_duration
                                .map(|duration| format!("{:.3}", duration.as_secs_f64() * 1e3))
                                .unwrap_or_else(|| "n/a".to_owned()),
                            residual_ms: format!(
                                "{:.3}",
                                statistics.residual_duration.as_secs_f64() * 1e3
                            ),
                            jacobian_ms: format!(
                                "{:.3}",
                                statistics.jacobian_duration.as_secs_f64() * 1e3
                            ),
                            solve_ms: format!("{:.3}", initial_solve_elapsed.as_secs_f64() * 1e3),
                            continuation_ms: format!(
                                "{:.3}",
                                continuation_started.elapsed().as_secs_f64() * 1e3
                            ),
                            residual_calls: statistics.residual_evaluations,
                            jacobian_calls: statistics.jacobian_evaluations,
                            status: if (continuation_state - target).norm() < 1e-7 {
                                "ok"
                            } else {
                                "bad"
                            }
                            .to_owned(),
                        });
                        if let Some(artifact_dir) = case.artifact_dir {
                            drop(artifact_dir);
                        }
                    }
                }
            }
        }
    }

    let mut report = String::new();
    writeln!(report, "# Nonlinear frontend matrix").unwrap();
    writeln!(
        report,
        "\n- dimensions: {:?}\n- continuation_counts: {:?}\n- include_aot: {}\n- aot_lifecycle_policy: {}",
        dimensions,
        continuation_counts,
        include_aot(),
        aot_build_policy().1
    )
    .unwrap();
    writeln!(report, "\nTiming note: preparation stages are diagnostic and may overlap; solve columns exclude preparation; continuation is warm parameter rebinding plus subsequent solves. AOT lifecycle is explicit: cold-rebuild always rebuilds, cold-build-if-missing is the first build, and warm-cache-hit reuses the prepared artifact. Do not compare rows with different lifecycle labels as a cold/warm speedup.\n").unwrap();
    report.push_str(&Table::new(&rows).to_string());
    writeln!(
        report,
        "\n\n## Status\n\nstatus | value\n--- | ---\nstatus | {}",
        if failures.is_empty() {
            "ok"
        } else {
            "completed_with_failures"
        }
    )
    .unwrap();
    for failure in failures {
        writeln!(report, "failure | `{failure}`").unwrap();
    }
    println!("{report}");
    if let Err(error) = write_test_report("Nonlinear_systems", "nonlinear_frontend_matrix", &report)
    {
        eprintln!("[nonlinear frontend matrix] report write failed: {error}");
    }
}
