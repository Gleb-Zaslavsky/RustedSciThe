//! Compact workload dashboard for the nonlinear-system solver family.
//!
//! This target deliberately uses one table per concern instead of Criterion's
//! warm-up/sample transcript.  It covers prepared Lambdify execution policies,
//! optional dense generated AOT, solver stages, and parameter continuation.
//! Existing Criterion targets remain the source for statistical microbenchmarks.

use std::fmt::Write as _;
use std::hint::black_box;
use std::time::{Duration, Instant};

use nalgebra::DVector;
use tabled::{Table, Tabled};
use tempfile::TempDir;

use RustedSciThe::Utils::test_reporting::write_test_report;
use RustedSciThe::numerical::Nonlinear_systems::prelude::{
    DampedNewtonMethod, DiagnosticsOptions, LambdifyExecutionPolicy, LevenbergMarquardtMinpack,
    NewtonMethod, NonlinearSolverMethod, PreparationStage, PreparationTelemetryMode,
    PreparedSymbolicNonlinearProblem, SolveOptions, SymbolicAotBuildPolicy,
    SymbolicGeneratedBackendConfig, SymbolicLambdifyFrontend, SymbolicNonlinearProblem,
    SymbolicProblemOptions,
};
use RustedSciThe::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Route {
    Lambdify(SymbolicLambdifyFrontend),
    AotDense(SymbolicLambdifyFrontend),
}

impl Route {
    fn label(self) -> &'static str {
        match self {
            Self::Lambdify(SymbolicLambdifyFrontend::ExprLegacy) => "lambdify/expr-legacy",
            Self::Lambdify(SymbolicLambdifyFrontend::AtomViewNative) => "lambdify/atom-native",
            Self::AotDense(SymbolicLambdifyFrontend::ExprLegacy) => "aot/expr-legacy",
            Self::AotDense(SymbolicLambdifyFrontend::AtomViewNative) => "aot/atom-native",
        }
    }

    fn frontend(self) -> SymbolicLambdifyFrontend {
        match self {
            Self::Lambdify(frontend) => frontend,
            Self::AotDense(frontend) => frontend,
        }
    }

    fn is_aot(self) -> bool {
        matches!(self, Self::AotDense(_))
    }
}

#[derive(Clone, Copy, Debug)]
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

#[derive(Clone, Copy, Debug)]
enum MethodKind {
    Newton,
    DampedNewton,
    LevenbergMarquardtMinpack,
}

impl MethodKind {
    fn label(self) -> &'static str {
        match self {
            Self::Newton => "newton",
            Self::DampedNewton => "damped-newton",
            Self::LevenbergMarquardtMinpack => "lm-minpack",
        }
    }

    fn method(self) -> NonlinearSolverMethod {
        match self {
            Self::Newton => NonlinearSolverMethod::Newton(NewtonMethod),
            Self::DampedNewton => {
                NonlinearSolverMethod::DampedNewton(DampedNewtonMethod::default())
            }
            Self::LevenbergMarquardtMinpack => NonlinearSolverMethod::LevenbergMarquardtMinpack(
                LevenbergMarquardtMinpack::default(),
            ),
        }
    }
}

#[derive(Tabled)]
struct PreparationRow {
    route: String,
    frontend: String,
    policy: String,
    dimension: usize,
    prepare_ms: String,
    parse_ms: String,
    symbolic_jacobian_ms: String,
    atom_conversion_ms: String,
    atom_jacobian_ms: String,
    residual_callback_ms: String,
    jacobian_callback_ms: String,
    aot_ms: String,
    backend: String,
    status: String,
}

#[derive(Tabled)]
struct SolveRow {
    route: String,
    frontend: String,
    policy: String,
    method: String,
    dimension: usize,
    phase: String,
    solve_ms: String,
    iterations: String,
    residual_calls: String,
    jacobian_calls: String,
    factorizations: String,
    linear_solves: String,
    residual_ms: String,
    jacobian_ms: String,
    linear_ms: String,
    final_residual: String,
    status: String,
}

struct PreparedCase {
    prepared: PreparedSymbolicNonlinearProblem,
    _artifact_dir: Option<TempDir>,
    preparation_ms: f64,
}

#[derive(Default)]
struct SolveSummary {
    elapsed: Duration,
    iterations: usize,
    residual_calls: usize,
    jacobian_calls: usize,
    factorizations: usize,
    linear_solves: usize,
    residual: Duration,
    jacobian: Duration,
    linear: Duration,
    final_residual: f64,
    status: String,
}

fn env_list(name: &str, default: &str) -> Vec<String> {
    std::env::var(name)
        .unwrap_or_else(|_| default.to_owned())
        .split(',')
        .map(|item| item.trim().to_ascii_lowercase())
        .filter(|item| !item.is_empty())
        .collect()
}

fn dimensions() -> Vec<usize> {
    env_list("NONLINEAR_BENCH_DIMENSIONS", "3,16,64")
        .into_iter()
        .filter_map(|item| item.parse::<usize>().ok())
        .filter(|dimension| *dimension > 0)
        .collect()
}

fn continuation_count() -> usize {
    std::env::var("NONLINEAR_BENCH_CONTINUATION")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|count: &usize| *count > 0)
        .unwrap_or(4)
}

fn selected_routes() -> Vec<Route> {
    env_list(
        "NONLINEAR_BENCH_ROUTES",
        "lambdify-expr-legacy,lambdify-atom-native",
    )
    .into_iter()
    .flat_map(|label| match label.as_str() {
        "lambdify" | "lambdify-expr-legacy" | "expr-legacy" => {
            vec![Route::Lambdify(SymbolicLambdifyFrontend::ExprLegacy)]
        }
        "lambdify-atom-native" | "atom-native" => {
            vec![Route::Lambdify(SymbolicLambdifyFrontend::AtomViewNative)]
        }
        "aot-dense" | "aot" => vec![
            Route::AotDense(SymbolicLambdifyFrontend::ExprLegacy),
            Route::AotDense(SymbolicLambdifyFrontend::AtomViewNative),
        ],
        "aot-expr-legacy" => vec![Route::AotDense(SymbolicLambdifyFrontend::ExprLegacy)],
        "aot-atom-native" => vec![Route::AotDense(SymbolicLambdifyFrontend::AtomViewNative)],
        _ => Vec::new(),
    })
    .filter(|route| {
        !route.is_aot() || std::env::var("NONLINEAR_BENCH_INCLUDE_AOT").as_deref() == Ok("1")
    })
    .collect()
}

fn selected_policies() -> Vec<Policy> {
    env_list("NONLINEAR_BENCH_POLICIES", "sequential,parallel")
        .into_iter()
        .filter_map(|label| match label.as_str() {
            "sequential" | "seq" => Some(Policy::Sequential),
            "parallel" | "par" => Some(Policy::Parallel),
            _ => None,
        })
        .collect()
}

fn selected_methods() -> Vec<MethodKind> {
    env_list("NONLINEAR_BENCH_METHODS", "newton,damped-newton,lm-minpack")
        .into_iter()
        .filter_map(|label| match label.as_str() {
            "newton" => Some(MethodKind::Newton),
            "damped-newton" | "damped_newton" => Some(MethodKind::DampedNewton),
            "lm-minpack" | "levenberg-marquardt-minpack" => {
                Some(MethodKind::LevenbergMarquardtMinpack)
            }
            _ => None,
        })
        .collect()
}

fn equations(dimension: usize) -> (Vec<String>, Vec<String>, DVector<f64>) {
    let variables = (0..dimension)
        .map(|index| format!("x{index}"))
        .collect::<Vec<_>>();
    let equations = variables
        .iter()
        .map(|variable| format!("p0*{variable}^2 - 1.0"))
        .collect::<Vec<_>>();
    (equations, variables, DVector::from_element(dimension, 0.8))
}

fn options(
    variables: &[String],
    policy: Policy,
    frontend: SymbolicLambdifyFrontend,
) -> SymbolicProblemOptions {
    SymbolicProblemOptions::new()
        .with_variables(variables.to_vec())
        .with_equation_parameters(vec!["p0".to_owned()])
        .with_equation_parameter_values(DVector::from_vec(vec![1.0]))
        .with_lambdify_execution_policy(policy.execution())
        .with_preparation_telemetry(PreparationTelemetryMode::Collect)
        .with_lambdify_frontend(frontend)
        .with_lambdify_backend()
}

fn prepare(dimension: usize, route: Route, policy: Policy) -> Result<PreparedCase, String> {
    let (equations, variables, _) = equations(dimension);
    let started = Instant::now();
    match route {
        Route::Lambdify(frontend) => {
            let prepared = PreparedSymbolicNonlinearProblem::from_strings(
                equations,
                options(&variables, policy, frontend),
            )
            .map_err(|error| format!("{error}"))?;
            Ok(PreparedCase {
                preparation_ms: started.elapsed().as_secs_f64() * 1e3,
                prepared,
                _artifact_dir: None,
            })
        }
        Route::AotDense(frontend) => {
            let artifact_dir = tempfile::tempdir().map_err(|error| error.to_string())?;
            let generated = SymbolicNonlinearProblem::from_strings_with_generated_backend(
                equations,
                options(&variables, policy, frontend),
                SymbolicGeneratedBackendConfig::defaults()
                    .with_build_policy(SymbolicAotBuildPolicy::BuildIfMissing {
                        profile: AotBuildProfile::Release,
                    })
                    .with_output_parent_dir(Some(artifact_dir.path().to_path_buf())),
            )
            .map_err(|error| format!("{error}"))?;
            Ok(PreparedCase {
                preparation_ms: started.elapsed().as_secs_f64() * 1e3,
                prepared: generated.into_prepared(),
                _artifact_dir: Some(artifact_dir),
            })
        }
    }
}

fn fmt_ms(value: Option<Duration>) -> String {
    value
        .map(|duration| format!("{:.3}", duration.as_secs_f64() * 1e3))
        .unwrap_or_else(|| "n/a".to_owned())
}

fn fmt_duration_ms(duration: Duration) -> String {
    format!("{:.3}", duration.as_secs_f64() * 1e3)
}

fn stage_ms(prepared: &PreparedSymbolicNonlinearProblem, stage: PreparationStage) -> String {
    prepared
        .preparation_report()
        .detailed
        .as_ref()
        .and_then(|report| report.stage(stage))
        .and_then(|timing| timing.wall_time)
        .map(|duration| fmt_duration_ms(duration))
        .unwrap_or_else(|| "n/a".to_owned())
}

fn preparation_row(
    dimension: usize,
    route: Route,
    policy: Policy,
    case: &PreparedCase,
) -> PreparationRow {
    let report = case.prepared.preparation_report();
    PreparationRow {
        route: route.label().to_owned(),
        frontend: route.frontend().as_str().to_owned(),
        policy: policy.label().to_owned(),
        dimension,
        prepare_ms: format!("{:.3}", case.preparation_ms),
        parse_ms: stage_ms(&case.prepared, PreparationStage::ExpressionParsing),
        symbolic_jacobian_ms: stage_ms(&case.prepared, PreparationStage::JacobianDifferentiation),
        atom_conversion_ms: stage_ms(&case.prepared, PreparationStage::AtomConversion),
        atom_jacobian_ms: stage_ms(&case.prepared, PreparationStage::AtomDifferentiation),
        residual_callback_ms: stage_ms(
            &case.prepared,
            PreparationStage::ResidualCallbackPreparation,
        ),
        jacobian_callback_ms: stage_ms(
            &case.prepared,
            PreparationStage::JacobianCallbackPreparation,
        ),
        aot_ms: fmt_ms(report.build_duration),
        backend: report.effective_backend.as_str().to_owned(),
        status: "prepared".to_owned(),
    }
}

fn solve_options() -> SolveOptions {
    SolveOptions {
        tolerance: 1e-10,
        max_iterations: 30,
        diagnostics: DiagnosticsOptions {
            collect_history: false,
            collect_statistics: true,
            ..DiagnosticsOptions::default()
        },
        ..SolveOptions::default()
    }
}

fn solve_row(
    route: Route,
    policy: Policy,
    method: MethodKind,
    dimension: usize,
    phase: &str,
    problem: &PreparedSymbolicNonlinearProblem,
    parameter: f64,
    initial: DVector<f64>,
) -> Result<(SolveRow, DVector<f64>, SolveSummary), String> {
    let bound = problem
        .bind_values(DVector::from_vec(vec![parameter]))
        .map_err(|error| format!("binding failed: {error}"))?;
    let started = Instant::now();
    let result = method
        .method()
        .solve(&bound, initial, solve_options())
        .map_err(|error| format!("solve failed: {error}"))?;
    black_box(&result.x);
    let elapsed = started.elapsed();
    let statistics = result.statistics;
    let status = format!("{:?}", result.termination);
    let summary = SolveSummary {
        elapsed,
        iterations: statistics.iterations,
        residual_calls: statistics.residual_evaluations,
        jacobian_calls: statistics.jacobian_evaluations,
        factorizations: statistics.linear_factorizations,
        linear_solves: statistics.linear_solves,
        residual: statistics.residual_duration,
        jacobian: statistics.jacobian_duration,
        linear: statistics.linear_solve_duration,
        final_residual: result.residual_norm,
        status: status.clone(),
    };
    let row = SolveRow {
        route: route.label().to_owned(),
        frontend: route.frontend().as_str().to_owned(),
        policy: policy.label().to_owned(),
        method: method.label().to_owned(),
        dimension,
        phase: phase.to_owned(),
        solve_ms: fmt_duration_ms(elapsed),
        iterations: statistics.iterations.to_string(),
        residual_calls: statistics.residual_evaluations.to_string(),
        jacobian_calls: statistics.jacobian_evaluations.to_string(),
        factorizations: statistics.linear_factorizations.to_string(),
        linear_solves: statistics.linear_solves.to_string(),
        residual_ms: fmt_duration_ms(statistics.residual_duration),
        jacobian_ms: fmt_duration_ms(statistics.jacobian_duration),
        linear_ms: fmt_duration_ms(statistics.linear_solve_duration),
        final_residual: format!("{:.3e}", result.residual_norm),
        status,
    };
    Ok((row, result.x, summary))
}

fn aggregate_row(
    route: Route,
    policy: Policy,
    method: MethodKind,
    dimension: usize,
    summaries: &[SolveSummary],
) -> SolveRow {
    let mut total = SolveSummary::default();
    for summary in summaries {
        total.elapsed += summary.elapsed;
        total.iterations += summary.iterations;
        total.residual_calls += summary.residual_calls;
        total.jacobian_calls += summary.jacobian_calls;
        total.factorizations += summary.factorizations;
        total.linear_solves += summary.linear_solves;
        total.residual += summary.residual;
        total.jacobian += summary.jacobian;
        total.linear += summary.linear;
        total.final_residual = summary.final_residual;
        total.status = summary.status.clone();
    }
    SolveRow {
        route: route.label().to_owned(),
        frontend: route.frontend().as_str().to_owned(),
        policy: policy.label().to_owned(),
        method: method.label().to_owned(),
        dimension,
        phase: "continuation".to_owned(),
        solve_ms: fmt_duration_ms(total.elapsed),
        iterations: total.iterations.to_string(),
        residual_calls: total.residual_calls.to_string(),
        jacobian_calls: total.jacobian_calls.to_string(),
        factorizations: total.factorizations.to_string(),
        linear_solves: total.linear_solves.to_string(),
        residual_ms: fmt_duration_ms(total.residual),
        jacobian_ms: fmt_duration_ms(total.jacobian),
        linear_ms: fmt_duration_ms(total.linear),
        final_residual: format!("{:.3e}", total.final_residual),
        status: total.status,
    }
}

fn main() {
    let routes = selected_routes();
    let policies = selected_policies();
    let methods = selected_methods();
    let dimensions = dimensions();
    let continuation = continuation_count();
    let mut preparation_rows = Vec::new();
    let mut solve_rows = Vec::new();
    let mut failures = Vec::new();

    for &dimension in &dimensions {
        for &route in &routes {
            for &policy in &policies {
                let case = match prepare(dimension, route, policy) {
                    Ok(case) => case,
                    Err(error) => {
                        failures.push(format!(
                            "prepare/{}/{dimension}/{}: {error}",
                            route.label(),
                            policy.label()
                        ));
                        preparation_rows.push(PreparationRow {
                            route: route.label().to_owned(),
                            frontend: route.frontend().as_str().to_owned(),
                            policy: policy.label().to_owned(),
                            dimension,
                            prepare_ms: "n/a".to_owned(),
                            parse_ms: "n/a".to_owned(),
                            symbolic_jacobian_ms: "n/a".to_owned(),
                            atom_conversion_ms: "n/a".to_owned(),
                            atom_jacobian_ms: "n/a".to_owned(),
                            residual_callback_ms: "n/a".to_owned(),
                            jacobian_callback_ms: "n/a".to_owned(),
                            aot_ms: "n/a".to_owned(),
                            backend: "n/a".to_owned(),
                            status: "error".to_owned(),
                        });
                        continue;
                    }
                };
                preparation_rows.push(preparation_row(dimension, route, policy, &case));

                for &method in &methods {
                    let (_, _, initial) = equations(dimension);
                    match solve_row(
                        route,
                        policy,
                        method,
                        dimension,
                        "single",
                        &case.prepared,
                        1.0,
                        initial.clone(),
                    ) {
                        Ok((row, previous, _single_summary)) => {
                            solve_rows.push(row);
                            let mut current = previous;
                            let mut continuation_summaries = Vec::new();
                            for step in 1..continuation {
                                match solve_row(
                                    route,
                                    policy,
                                    method,
                                    dimension,
                                    "continuation",
                                    &case.prepared,
                                    1.0 + step as f64 * 0.05,
                                    current,
                                ) {
                                    Ok((_row, next, summary)) => {
                                        current = next;
                                        continuation_summaries.push(summary);
                                    }
                                    Err(error) => {
                                        failures.push(format!(
                                            "continuation/{}/{dimension}/{}/{}: {error}",
                                            route.label(),
                                            policy.label(),
                                            method.label()
                                        ));
                                        break;
                                    }
                                }
                            }
                            if !continuation_summaries.is_empty() {
                                solve_rows.push(aggregate_row(
                                    route,
                                    policy,
                                    method,
                                    dimension,
                                    &continuation_summaries,
                                ));
                            }
                        }
                        Err(error) => failures.push(format!(
                            "single/{}/{dimension}/{}/{}: {error}",
                            route.label(),
                            policy.label(),
                            method.label()
                        )),
                    }
                }
            }
        }
    }

    let mut report = String::new();
    writeln!(
        report,
        "# Nonlinear systems workload dashboard\n\n- dimensions: {:?}\n- routes: {:?}\n- policies: {:?}\n- methods: {:?}\n- continuation_count: {}\n- timing note: preparation stages are exclusive where available; solve durations include numerical solver overhead; continuation row is the aggregate of the remaining parameter binds\n",
        dimensions,
        routes.iter().map(|route| route.label()).collect::<Vec<_>>(),
        policies.iter().map(|policy| policy.label()).collect::<Vec<_>>(),
        methods.iter().map(|method| method.label()).collect::<Vec<_>>(),
        continuation
    )
    .expect("writing dashboard heading");
    report.push_str("## Preparation\n\n");
    report.push_str(&Table::new(&preparation_rows).to_string());
    report.push_str("\n\n## Solves and continuation\n\n");
    report.push_str(&Table::new(&solve_rows).to_string());
    report.push_str("\n\n## Status\n\n");
    if failures.is_empty() {
        report.push_str("status | ok\n--- | ---\n");
    } else {
        report.push_str("status | completed_with_failures\n--- | ---\n");
        for failure in &failures {
            writeln!(report, "failure | `{failure}`").expect("writing failure");
        }
    }

    println!("{report}");
    if let Err(error) = write_test_report(
        "Nonlinear_systems",
        "nonlinear_workloads_dashboard",
        &report,
    ) {
        eprintln!("[nonlinear dashboard] report write failed: {error}");
    }
}
