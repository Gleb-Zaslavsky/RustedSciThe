//! Bounded large-workload evidence before the release campaign.
//!
//! These stories mirror the LSODE2 separation between callback-only and
//! full-solve measurements. They are ignored by the normal debug suite because
//! their purpose is release evidence, not a fast correctness gate.

use std::time::Instant;

use super::super::api::{
    RadauConfig, RadauFrontend, RadauMatrixLayout, RadauOutputPolicy, RadauProblem, RadauSolver,
    RadauTelemetryMode,
};
use super::super::new::aot::AotPlan;
use super::super::new::callbacks::PreparedSymbolicCallbacks;
use super::super::new::config::{RadauAssembly, RadauMatrixLayout as InternalMatrixLayout};
use super::super::new::telemetry::{RadauTelemetry, RadauTelemetryMode as InternalTelemetryMode};
use super::story_support::rebuild_aot_config;
use crate::numerical::ivp_workloads::{
    WorkloadKind, build_workload, parameter_continuation_target,
};
use tabled::Tabled;

#[derive(Tabled)]
struct TrajectoryRow {
    workload: String,
    dimension: usize,
    frontend: String,
    layout: String,
    baseline_newton_ms: String,
    baseline_linear_ms: String,
    continued_newton_ms: String,
    continued_linear_ms: String,
    residual_evaluations: u64,
    jacobian_evaluations: u64,
    drift: String,
    status: &'static str,
}

#[derive(Tabled)]
struct StageRow {
    workload: String,
    dimension: usize,
    frontend: String,
    layout: String,
    repetitions: usize,
    callback_wall_ms: String,
    callback_telemetry_ms: String,
    preparation_ms: String,
    solve_ms: String,
    residual_evaluations: u64,
    jacobian_evaluations: u64,
    factorizations: u64,
    allocations: u64,
    status: &'static str,
}

#[derive(Tabled)]
struct DenseAotDiagnosticRow {
    frontend: String,
    dimension: usize,
    initial_h_abs: String,
    prepare_ms: String,
    full_solve_ms: String,
    callback_ms: String,
    newton_ms: String,
    linear_ms: String,
    jacobian_assembly_ms: String,
    factorization_ms: String,
    real_solve_ms: String,
    complex_solve_ms: String,
    residual_calls: u64,
    jacobian_calls: u64,
    factorizations: u64,
    accepted_steps: u64,
    rejected_steps: u64,
    adaptive_attempts: usize,
    accepted_h_preview: String,
    error_norm_preview: String,
    factor_invalidations: usize,
    jacobian_refreshes: usize,
    parity_drift: String,
    final_norm: String,
    status: &'static str,
}

fn adaptive_trace_columns(
    report: &crate::numerical::Radau::api::RadauTelemetryReport,
) -> (String, String, usize, usize) {
    let preview = |values: Vec<String>| {
        if values.len() <= 8 {
            values.join(",")
        } else {
            format!(
                "{},...,{}",
                values[..4].join(","),
                values[values.len() - 4..].join(",")
            )
        }
    };
    let accepted_h = preview(
        report
            .adaptive_steps
            .iter()
            .filter(|step| step.accepted)
            .map(|step| format!("{:.3e}", step.h_abs))
            .collect(),
    );
    let errors = preview(
        report
            .adaptive_steps
            .iter()
            .map(|step| format!("{:.3e}", step.error_norm))
            .collect(),
    );
    let factor_invalidations = report
        .adaptive_steps
        .iter()
        .filter(|step| step.factor_invalidated)
        .count();
    let jacobian_refreshes = report
        .adaptive_steps
        .iter()
        .filter(|step| step.jacobian_refreshed)
        .count();
    (accepted_h, errors, factor_invalidations, jacobian_refreshes)
}

#[derive(Tabled)]
struct DenseAotCallbackParityRow {
    frontend: String,
    dimension: usize,
    residual_diff: String,
    jacobian_diff: String,
    lambdify_residual_norm: String,
    aot_residual_norm: String,
    lambdify_jacobian_norm: String,
    aot_jacobian_norm: String,
    status: &'static str,
}

fn dimensions_from_env(name: &str, default: &[usize]) -> Vec<usize> {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|item| item.trim().parse::<usize>().ok())
                .filter(|dimension| *dimension > 0)
                .collect::<Vec<_>>()
        })
        .filter(|dimensions| !dimensions.is_empty())
        .unwrap_or_else(|| default.to_vec())
}

fn t_bound(kind: WorkloadKind) -> f64 {
    match kind {
        WorkloadKind::DiffusionChain => 0.02,
        WorkloadKind::CombustionLike | WorkloadKind::Robertson | WorkloadKind::ThreeBody => 0.002,
        WorkloadKind::StiffScalar => 0.01,
    }
}

fn problem(
    kind: WorkloadKind,
    dimension: usize,
    frontend: RadauFrontend,
) -> (RadauProblem, Vec<f64>, Vec<f64>) {
    let workload = build_workload(kind, dimension);
    let jacobian = match frontend {
        RadauFrontend::ExprLegacy => {
            let variables: Vec<&str> = workload.variables.iter().map(String::as_str).collect();
            Some(
                workload
                    .equations
                    .iter()
                    .flat_map(|equation| {
                        variables
                            .iter()
                            .map(move |variable| equation.diff(variable))
                    })
                    .collect(),
            )
        }
        RadauFrontend::AtomViewNative => None,
    };
    let mut problem = RadauProblem::new(
        workload.equations,
        workload.variables,
        workload.time_variable,
    )
    .with_parameters(workload.parameter_names);
    if let Some(jacobian) = jacobian {
        problem = problem.with_jacobian(jacobian);
    }
    (
        problem,
        workload.initial_state.as_slice().to_vec(),
        workload.parameter_values.as_slice().to_vec(),
    )
}

fn config(frontend: RadauFrontend, layout: RadauMatrixLayout, kind: WorkloadKind) -> RadauConfig {
    let bound = t_bound(kind);
    RadauConfig {
        t_bound: bound,
        first_step: Some(bound * 0.1),
        max_step: bound * 0.25,
        rtol: 1.0e-7,
        atol: 1.0e-10,
        max_steps: 20_000,
        max_newton_iterations: 8,
        max_retries: 24,
        frontend,
        matrix_layout: layout,
        telemetry: RadauTelemetryMode::Timings,
        output: RadauOutputPolicy::Dense,
        ..RadauConfig::default()
    }
}

fn max_diff(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

fn layouts(kind: WorkloadKind) -> Vec<RadauMatrixLayout> {
    if kind == WorkloadKind::DiffusionChain {
        vec![
            RadauMatrixLayout::Dense,
            RadauMatrixLayout::Sparse,
            RadauMatrixLayout::Banded { lower: 1, upper: 1 },
            RadauMatrixLayout::Banded { lower: 2, upper: 1 },
        ]
    } else {
        vec![RadauMatrixLayout::Dense]
    }
}

#[test]
#[ignore = "release workload trajectory and continuation matrix"]
fn large_workload_trajectory_and_continuation_match_fresh_reference() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "Radau_Large",
        "numerical::Radau::tests::large_performance::large_workload_trajectory_and_continuation_match_fresh_reference",
    );
    let dimensions = dimensions_from_env("RADAU_STORY_DIFFUSION_DIMENSIONS", &[32, 64, 128]);
    let sample_fractions = [0.0, 0.25, 0.5, 0.75, 1.0];
    let mut rows = Vec::new();

    for dimension in dimensions {
        run_trajectory_case(
            WorkloadKind::DiffusionChain,
            dimension,
            &sample_fractions,
            &mut rows,
        );
    }

    // These small nonlinear fixtures complement the large diffusion sweep:
    // they exercise trajectory drift and continuation on the same stiff
    // workload families used by the LSODE2 release corpus.
    for (kind, dimension) in [
        (WorkloadKind::CombustionLike, 3),
        (WorkloadKind::ThreeBody, 12),
    ] {
        run_trajectory_case(kind, dimension, &sample_fractions, &mut rows);
    }

    crate::Utils::test_reporting::capture_test_table("[Radau trajectory summary]", &rows);
}

fn run_trajectory_case(
    kind: WorkloadKind,
    dimension: usize,
    sample_fractions: &[f64],
    rows: &mut Vec<TrajectoryRow>,
) {
    let bound = t_bound(kind);
    let sample_times: Vec<f64> = sample_fractions
        .iter()
        .map(|fraction| bound * fraction)
        .collect();
    let target = parameter_continuation_target(
        &nalgebra::DVector::from_vec(problem(kind, dimension, RadauFrontend::ExprLegacy).2),
        8,
    );
    let mut route_reference: Option<Vec<f64>> = None;

    for frontend in [RadauFrontend::ExprLegacy, RadauFrontend::AtomViewNative] {
        for layout in layouts(kind) {
            let (route_problem, initial_state, parameters) = problem(kind, dimension, frontend);
            let route_config = config(frontend, layout, kind);
            let mut solver = RadauSolver::prepare(route_problem, route_config.clone())
                .unwrap_or_else(|error| {
                    panic!("prepare {kind:?}/{frontend:?}/{layout:?}: {error}")
                });
            let baseline = solver
                .solve_with_parameters(&initial_state, &parameters)
                .unwrap_or_else(|error| panic!("solve {kind:?}/{frontend:?}/{layout:?}: {error}"));
            let baseline_samples = baseline
                .sample_many(&sample_times)
                .unwrap_or_else(|error| panic!("sample {kind:?}/{frontend:?}/{layout:?}: {error}"));

            if let Some(reference) = &route_reference {
                assert!(
                    max_diff(reference, &baseline_samples) < 2.0e-5,
                    "trajectory drift workload={kind:?} dimension={dimension} frontend={frontend:?} layout={layout:?} drift={:.3e}",
                    max_diff(reference, &baseline_samples)
                );
            } else {
                route_reference = Some(baseline_samples.clone());
            }

            let continued = solver
                .continue_with_parameters(&initial_state, target.as_slice())
                .unwrap_or_else(|error| {
                    panic!("continuation {kind:?}/{frontend:?}/{layout:?}: {error}")
                });
            let (fresh_problem, fresh_state, _) = problem(kind, dimension, frontend);
            let mut fresh = RadauSolver::prepare(fresh_problem, config(frontend, layout, kind))
                .unwrap_or_else(|error| {
                    panic!("fresh prepare {kind:?}/{frontend:?}/{layout:?}: {error}")
                });
            let fresh_solution = fresh
                .solve_with_parameters(&fresh_state, target.as_slice())
                .unwrap_or_else(|error| {
                    panic!("fresh solve {kind:?}/{frontend:?}/{layout:?}: {error}")
                });
            let continuation_samples =
                continued
                    .sample_many(&sample_times)
                    .unwrap_or_else(|error| {
                        panic!("continued sample {kind:?}/{frontend:?}/{layout:?}: {error}")
                    });
            let fresh_samples = fresh_solution
                .sample_many(&sample_times)
                .unwrap_or_else(|error| {
                    panic!("fresh sample {kind:?}/{frontend:?}/{layout:?}: {error}")
                });
            let drift = max_diff(&continuation_samples, &fresh_samples);

            assert!(
                drift < 2.0e-5,
                "continuation drift workload={kind:?} dimension={dimension} frontend={frontend:?} layout={layout:?} drift={drift:.3e}"
            );
            rows.push(TrajectoryRow {
                workload: kind.label().to_owned(),
                dimension,
                frontend: format!("{frontend:?}"),
                layout: format!("{layout:?}"),
                baseline_newton_ms: format!("{:.3}", baseline.telemetry().timings_ms["newton_ms"]),
                baseline_linear_ms: format!("{:.3}", baseline.telemetry().timings_ms["linear_ms"]),
                continued_newton_ms: format!(
                    "{:.3}",
                    continued.telemetry().timings_ms["newton_ms"]
                ),
                continued_linear_ms: format!(
                    "{:.3}",
                    continued.telemetry().timings_ms["linear_ms"]
                ),
                residual_evaluations: continued.telemetry().counters["residual_evaluations"],
                jacobian_evaluations: continued.telemetry().counters["jacobian_evaluations"],
                drift: format!("{drift:.3e}"),
                status: "ok",
            });
        }
    }
}

#[test]
#[ignore = "release callback/full-solve stage matrix"]
fn large_workload_callback_and_full_solve_stage_breakdown() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "Radau_Large",
        "numerical::Radau::tests::large_performance::large_workload_callback_and_full_solve_stage_breakdown",
    );
    let dimensions = dimensions_from_env("RADAU_STORY_STAGE_DIMENSIONS", &[32, 64, 128]);
    let repetitions = std::env::var("RADAU_STORY_CALLBACK_REPETITIONS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(32);
    let mut rows = Vec::new();

    for (kind, dimension) in dimensions
        .into_iter()
        .map(|dimension| (WorkloadKind::DiffusionChain, dimension))
        .chain(std::iter::once((WorkloadKind::ThreeBody, 12)))
    {
        for frontend in [RadauAssembly::ExprLegacy, RadauAssembly::AtomViewNative] {
            for layout in layouts(kind).into_iter().map(|layout| match layout {
                RadauMatrixLayout::Dense => InternalMatrixLayout::Dense,
                RadauMatrixLayout::Sparse => InternalMatrixLayout::Sparse,
                RadauMatrixLayout::Banded { lower, upper } => {
                    InternalMatrixLayout::Banded { lower, upper }
                }
            }) {
                let workload = build_workload(kind, dimension);
                let variables: Vec<&str> = workload.variables.iter().map(String::as_str).collect();
                let jacobian = match frontend {
                    RadauAssembly::ExprLegacy => Some(
                        workload
                            .equations
                            .iter()
                            .flat_map(|equation| {
                                variables
                                    .iter()
                                    .map(move |variable| equation.diff(variable))
                            })
                            .collect(),
                    ),
                    RadauAssembly::AtomViewNative => None,
                };
                let mut preparation = RadauTelemetry::new(InternalTelemetryMode::Timings);
                let callbacks = PreparedSymbolicCallbacks::prepare_with_telemetry(
                    frontend,
                    workload.equations.clone(),
                    jacobian,
                    &workload.time_variable,
                    &variables,
                    &workload
                        .parameter_names
                        .iter()
                        .map(String::as_str)
                        .collect::<Vec<_>>(),
                    &mut preparation,
                )
                .unwrap_or_else(|error| {
                    panic!("callback prepare {frontend:?}/{layout:?}: {error}")
                });
                let mut session = callbacks.session_with_telemetry(InternalTelemetryMode::Timings);
                session
                    .rebind_parameters(workload.parameter_values.as_slice())
                    .unwrap();
                let mut residual = vec![0.0; dimension];
                let jacobian_len = match layout {
                    InternalMatrixLayout::Dense => dimension * dimension,
                    InternalMatrixLayout::Sparse => callbacks.jacobian_pattern().len(),
                    InternalMatrixLayout::Banded { lower, upper } => {
                        (lower + upper + 1) * dimension
                    }
                };
                let mut jacobian_output = vec![0.0; jacobian_len];
                let callback_start = Instant::now();
                for _ in 0..repetitions {
                    session
                        .evaluate_residual(0.0, workload.initial_state.as_slice(), &mut residual)
                        .unwrap();
                    session
                        .evaluate_jacobian_layout(
                            0.0,
                            workload.initial_state.as_slice(),
                            layout,
                            &mut jacobian_output,
                        )
                        .unwrap();
                }
                let callback_wall_ms = callback_start.elapsed().as_secs_f64() * 1_000.0;

                let public_layout = match layout {
                    InternalMatrixLayout::Dense => RadauMatrixLayout::Dense,
                    InternalMatrixLayout::Sparse => RadauMatrixLayout::Sparse,
                    InternalMatrixLayout::Banded { lower, upper } => {
                        RadauMatrixLayout::Banded { lower, upper }
                    }
                };
                let public_frontend = match frontend {
                    RadauAssembly::ExprLegacy => RadauFrontend::ExprLegacy,
                    RadauAssembly::AtomViewNative => RadauFrontend::AtomViewNative,
                };
                let (problem, _, _) = problem(kind, dimension, public_frontend);
                let mut solver =
                    RadauSolver::prepare(problem, config(public_frontend, public_layout, kind))
                        .unwrap();
                let solve = solver
                    .solve_with_parameters(
                        workload.initial_state.as_slice(),
                        workload.parameter_values.as_slice(),
                    )
                    .unwrap();
                rows.push(StageRow {
                    workload: kind.label().to_owned(),
                    dimension,
                    frontend: format!("{frontend:?}"),
                    layout: format!("{layout:?}"),
                    repetitions,
                    callback_wall_ms: format!("{callback_wall_ms:.3}"),
                    callback_telemetry_ms: format!(
                        "{:.3}",
                        session.telemetry().timings.callback_ms
                    ),
                    preparation_ms: format!("{:.3}", preparation.timings.preparation_ms),
                    solve_ms: format!(
                        "{:.3}",
                        solve.telemetry().timings_ms["newton_ms"]
                            + solve.telemetry().timings_ms["linear_ms"]
                    ),
                    residual_evaluations: solve.telemetry().counters["residual_evaluations"],
                    jacobian_evaluations: solve.telemetry().counters["jacobian_evaluations"],
                    factorizations: solve.telemetry().counters["factorizations"],
                    allocations: solve.telemetry().counters["allocations"],
                    status: "ok",
                });
            }
        }
    }

    crate::Utils::test_reporting::capture_test_table("[Radau stage summary]", &rows);
}

#[test]
#[ignore = "targeted Dense AOT full-solve attribution; requires tcc"]
fn dense_aot_full_solve_attribution_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "Radau_Diagnostics",
        "numerical::Radau::tests::large_performance::dense_aot_full_solve_attribution_story",
    );
    let dimensions = dimensions_from_env("RADAU_STORY_DENSE_AOT_DIAGNOSTIC_DIMENSIONS", &[512]);
    let short_horizon = std::env::var("RADAU_STORY_DENSE_AOT_DIAGNOSTIC_T_BOUND")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0);
    let fixed_step = std::env::var("RADAU_STORY_DENSE_AOT_DIAGNOSTIC_FIXED_STEP")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0);
    let mut rows = Vec::new();

    for dimension in dimensions {
        let mut reference: Option<Vec<f64>> = None;
        for frontend in [RadauFrontend::ExprLegacy, RadauFrontend::AtomViewNative] {
            let (problem, initial_state, parameters) =
                problem(WorkloadKind::DiffusionChain, dimension, frontend);
            let mut config = config(
                frontend,
                RadauMatrixLayout::Dense,
                WorkloadKind::DiffusionChain,
            );
            config.t_bound = 0.01;
            config.first_step = Some(0.001);
            config.max_step = 0.0025;
            if let Some(t_bound) = short_horizon {
                config.t_bound = t_bound;
                let step = fixed_step.unwrap_or(t_bound * 0.5);
                config.first_step = Some(step);
                config.max_step = step;
            }
            config.telemetry = RadauTelemetryMode::Timings;
            let route_name = format!("lambdify/{frontend:?}");
            let started = Instant::now();
            let mut solver = RadauSolver::prepare(problem, config.clone())
                .unwrap_or_else(|error| panic!("{route_name} prepare failed: {error}"));
            let prepare_ms = started.elapsed().as_secs_f64() * 1_000.0;
            let started = Instant::now();
            let solution = solver
                .solve_with_parameters(&initial_state, &parameters)
                .unwrap_or_else(|error| panic!("{route_name} solve failed: {error}"));
            let full_solve_ms = started.elapsed().as_secs_f64() * 1_000.0;
            let report = solution.telemetry();
            let drift = reference
                .as_ref()
                .map(|reference| max_diff(reference, &solution.y));
            if reference.is_none() {
                reference = Some(solution.y.clone());
            }
            let timing = |key: &'static str| report.timings_ms[key];
            let (accepted_h_preview, error_norm_preview, factor_invalidations, jacobian_refreshes) =
                adaptive_trace_columns(report);
            rows.push(DenseAotDiagnosticRow {
                frontend: route_name,
                dimension,
                initial_h_abs: report
                    .initial_h_abs
                    .map(|value| format!("{value:.3e}"))
                    .unwrap_or_else(|| "missing".to_owned()),
                prepare_ms: format!("{prepare_ms:.3}"),
                full_solve_ms: format!("{full_solve_ms:.3}"),
                callback_ms: format!("{:.3}", timing("callback_ms")),
                newton_ms: format!("{:.3}", timing("newton_ms")),
                linear_ms: format!("{:.3}", timing("linear_ms")),
                jacobian_assembly_ms: format!("{:.3}", timing("jacobian_assembly_ms")),
                factorization_ms: format!("{:.3}", timing("factorization_ms")),
                real_solve_ms: format!("{:.3}", timing("real_solve_ms")),
                complex_solve_ms: format!("{:.3}", timing("complex_solve_ms")),
                residual_calls: report.counters["residual_calls"],
                jacobian_calls: report.counters["jacobian_calls"],
                factorizations: report.counters["factorizations"],
                accepted_steps: report.counters["accepted_steps"],
                rejected_steps: report.counters["rejected_steps"],
                adaptive_attempts: report.adaptive_steps.len(),
                accepted_h_preview,
                error_norm_preview,
                factor_invalidations,
                jacobian_refreshes,
                parity_drift: drift
                    .map(|value| format!("{value:.3e}"))
                    .unwrap_or_else(|| "reference".to_owned()),
                final_norm: format!(
                    "{:.6e}",
                    solution.y.iter().map(|value| value.abs()).sum::<f64>()
                ),
                status: "ok",
            });
        }

        // Repeat the same dense solve through AOT. The temporary artifact
        // directory stays alive until the solver and solution are finished.
        for frontend in [RadauFrontend::ExprLegacy, RadauFrontend::AtomViewNative] {
            let (problem, initial_state, parameters) =
                problem(WorkloadKind::DiffusionChain, dimension, frontend);
            let mut config = config(
                frontend,
                RadauMatrixLayout::Dense,
                WorkloadKind::DiffusionChain,
            );
            config.execution = super::super::api::RadauExecution::Aot;
            // Keep this route apple-to-apple with the Lambdify diagnostic. If
            // these values are omitted, the adaptive controller follows a
            // different trajectory and the difference is misattributed to AOT.
            config.t_bound = 0.01;
            config.first_step = Some(0.001);
            config.max_step = 0.0025;
            if let Some(t_bound) = short_horizon {
                config.t_bound = t_bound;
                let step = fixed_step.unwrap_or(t_bound * 0.5);
                config.first_step = Some(step);
                config.max_step = step;
            }
            config.telemetry = RadauTelemetryMode::Timings;
            let artifact_directory = tempfile::tempdir().expect("diagnostic AOT directory");
            config.aot = Some(rebuild_aot_config(artifact_directory.path()));
            let route_name = format!("aot/{frontend:?}");
            let started = Instant::now();
            let mut solver = RadauSolver::prepare(problem, config)
                .unwrap_or_else(|error| panic!("{route_name} prepare failed: {error}"));
            let prepare_ms = started.elapsed().as_secs_f64() * 1_000.0;
            let started = Instant::now();
            let solution = solver
                .solve_with_parameters(&initial_state, &parameters)
                .unwrap_or_else(|error| panic!("{route_name} solve failed: {error}"));
            let full_solve_ms = started.elapsed().as_secs_f64() * 1_000.0;
            let report = solution.telemetry();
            let drift = reference
                .as_ref()
                .map(|reference| max_diff(reference, &solution.y));
            let timing = |key: &'static str| report.timings_ms[key];
            let (accepted_h_preview, error_norm_preview, factor_invalidations, jacobian_refreshes) =
                adaptive_trace_columns(report);
            rows.push(DenseAotDiagnosticRow {
                frontend: route_name,
                dimension,
                initial_h_abs: report
                    .initial_h_abs
                    .map(|value| format!("{value:.3e}"))
                    .unwrap_or_else(|| "missing".to_owned()),
                prepare_ms: format!("{prepare_ms:.3}"),
                full_solve_ms: format!("{full_solve_ms:.3}"),
                callback_ms: format!("{:.3}", timing("callback_ms")),
                newton_ms: format!("{:.3}", timing("newton_ms")),
                linear_ms: format!("{:.3}", timing("linear_ms")),
                jacobian_assembly_ms: format!("{:.3}", timing("jacobian_assembly_ms")),
                factorization_ms: format!("{:.3}", timing("factorization_ms")),
                real_solve_ms: format!("{:.3}", timing("real_solve_ms")),
                complex_solve_ms: format!("{:.3}", timing("complex_solve_ms")),
                residual_calls: report.counters["residual_calls"],
                jacobian_calls: report.counters["jacobian_calls"],
                factorizations: report.counters["factorizations"],
                accepted_steps: report.counters["accepted_steps"],
                rejected_steps: report.counters["rejected_steps"],
                adaptive_attempts: report.adaptive_steps.len(),
                accepted_h_preview,
                error_norm_preview,
                factor_invalidations,
                jacobian_refreshes,
                parity_drift: drift
                    .map(|value| format!("{value:.3e}"))
                    .unwrap_or_else(|| "missing".to_owned()),
                final_norm: format!(
                    "{:.6e}",
                    solution.y.iter().map(|value| value.abs()).sum::<f64>()
                ),
                status: if drift.is_some_and(|value| value < 2.0e-5) {
                    "ok"
                } else {
                    "parity-drift"
                },
            });
            drop(artifact_directory);
        }
    }

    crate::Utils::test_reporting::capture_test_table(
        "[Radau Dense AOT full-solve attribution]",
        &rows,
    );
}

#[test]
#[ignore = "targeted Dense AOT callback parity; requires tcc"]
fn dense_aot_callback_parity_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "Radau_Diagnostics",
        "numerical::Radau::tests::large_performance::dense_aot_callback_parity_story",
    );
    let dimensions = dimensions_from_env("RADAU_STORY_DENSE_AOT_DIAGNOSTIC_DIMENSIONS", &[128]);
    let mut rows = Vec::new();

    for dimension in dimensions {
        let workload = build_workload(WorkloadKind::DiffusionChain, dimension);
        let variables: Vec<&str> = workload.variables.iter().map(String::as_str).collect();
        let parameters: Vec<&str> = workload
            .parameter_names
            .iter()
            .map(String::as_str)
            .collect();
        let explicit_jacobian = |frontend| match frontend {
            RadauAssembly::ExprLegacy => Some(
                workload
                    .equations
                    .iter()
                    .flat_map(|equation| {
                        variables
                            .iter()
                            .map(move |variable| equation.diff(variable))
                    })
                    .collect(),
            ),
            RadauAssembly::AtomViewNative => None,
        };

        for assembly in [RadauAssembly::ExprLegacy, RadauAssembly::AtomViewNative] {
            let mut preparation = RadauTelemetry::new(InternalTelemetryMode::Timings);
            let lambdify = PreparedSymbolicCallbacks::prepare_with_telemetry(
                assembly,
                workload.equations.clone(),
                explicit_jacobian(assembly),
                &workload.time_variable,
                &variables,
                &parameters,
                &mut preparation,
            )
            .unwrap_or_else(|error| panic!("Lambdify callback preparation failed: {error}"));
            let mut lambdify = lambdify.session_with_telemetry(InternalTelemetryMode::Timings);
            lambdify
                .rebind_parameters(workload.parameter_values.as_slice())
                .expect("Lambdify parameter binding");

            let artifact_directory = tempfile::tempdir().expect("Dense AOT callback directory");
            let mut preparation = RadauTelemetry::new(InternalTelemetryMode::Timings);
            let prepared = AotPlan::prepare(
                assembly,
                InternalMatrixLayout::Dense,
                workload.equations.clone(),
                explicit_jacobian(assembly),
                workload.time_variable.clone(),
                workload.variables.clone(),
                workload.parameter_names.clone(),
                rebuild_aot_config(artifact_directory.path()).generated,
                &mut preparation,
            )
            .unwrap_or_else(|error| panic!("AOT callback preparation failed: {error}"));
            let callbacks = PreparedSymbolicCallbacks::Aot(prepared.plan);
            let mut aot = callbacks.session_with_telemetry(InternalTelemetryMode::Timings);
            aot.rebind_parameters(workload.parameter_values.as_slice())
                .expect("AOT parameter binding");
            let perturbed_state = workload
                .initial_state
                .iter()
                .enumerate()
                .map(|(index, value)| value * 1.037 + 0.0001 * index as f64)
                .collect::<Vec<_>>();
            let probes = [
                (0.0, workload.initial_state.as_slice()),
                (0.007, perturbed_state.as_slice()),
            ];
            let mut lambdify_residual = vec![0.0; dimension];
            let mut aot_residual = vec![0.0; dimension];
            let mut lambdify_jacobian = vec![0.0; dimension * dimension];
            // A non-zero sentinel catches a generated dense callback that
            // skips zero-valued entries instead of overwriting the caller's
            // reusable Jacobian workspace.
            let mut aot_jacobian = vec![7.0; dimension * dimension];
            let mut residual_diff: f64 = 0.0;
            let mut jacobian_diff: f64 = 0.0;
            for (time, probe_state) in probes {
                lambdify
                    .evaluate_residual(time, probe_state, &mut lambdify_residual)
                    .expect("Lambdify residual callback");
                aot.evaluate_residual(time, probe_state, &mut aot_residual)
                    .expect("AOT residual callback");
                lambdify
                    .evaluate_jacobian_layout(
                        time,
                        probe_state,
                        InternalMatrixLayout::Dense,
                        &mut lambdify_jacobian,
                    )
                    .expect("Lambdify Jacobian callback");
                aot.evaluate_jacobian_layout(
                    time,
                    probe_state,
                    InternalMatrixLayout::Dense,
                    &mut aot_jacobian,
                )
                .expect("AOT Jacobian callback");
                residual_diff = residual_diff.max(max_diff(&lambdify_residual, &aot_residual));
                jacobian_diff = jacobian_diff.max(max_diff(&lambdify_jacobian, &aot_jacobian));
            }
            rows.push(DenseAotCallbackParityRow {
                frontend: format!("{assembly:?}"),
                dimension,
                residual_diff: format!("{residual_diff:.3e}"),
                jacobian_diff: format!("{jacobian_diff:.3e}"),
                lambdify_residual_norm: format!(
                    "{:.6e}",
                    lambdify_residual
                        .iter()
                        .map(|value| value.abs())
                        .sum::<f64>()
                ),
                aot_residual_norm: format!(
                    "{:.6e}",
                    aot_residual.iter().map(|value| value.abs()).sum::<f64>()
                ),
                lambdify_jacobian_norm: format!(
                    "{:.6e}",
                    lambdify_jacobian
                        .iter()
                        .map(|value| value.abs())
                        .sum::<f64>()
                ),
                aot_jacobian_norm: format!(
                    "{:.6e}",
                    aot_jacobian.iter().map(|value| value.abs()).sum::<f64>()
                ),
                status: if residual_diff < 1.0e-10 && jacobian_diff < 1.0e-10 {
                    "ok"
                } else {
                    "callback-drift"
                },
            });
            drop(artifact_directory);
        }
    }

    crate::Utils::test_reporting::capture_test_table("[Radau Dense AOT callback parity]", &rows);
}
