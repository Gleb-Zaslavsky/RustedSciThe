//! Large warm-stage comparison for the prepared Lambdify and generated AOT
//! nonlinear routes.
//!
//! This is intentionally an ignored release story. It performs real generated
//! Rust AOT builds, so ordinary correctness tests remain fast. The measured
//! solve loop starts only after preparation/build and compares the same dense
//! problem, initial point, method, and solver options on both routes.

#![cfg(test)]

use crate::numerical::Nonlinear_systems::engine::{
    DiagnosticsOptions, NewtonMethod, SolveOptions, SolverEngine,
};
use crate::numerical::Nonlinear_systems::error::TerminationReason;
use crate::numerical::Nonlinear_systems::symbolic::{
    PreparedSymbolicNonlinearProblem, SymbolicDenseAotOptions, SymbolicNonlinearProblem,
    SymbolicProblemOptions,
};
use crate::numerical::Nonlinear_systems::symbolic_aot_test_support::aot_solver_test_guard;
use crate::numerical::Nonlinear_systems::symbolic_backend::SelectedSymbolicNonlinearBackendKind;
use crate::numerical::Nonlinear_systems::symbolic_generated::{
    SymbolicAotBuildPolicy, SymbolicGeneratedBackendConfig,
};
use crate::symbolic::codegen::codegen_aot_runtime_link::unregister_linked_dense_backend;
use crate::symbolic::codegen::codegen_runtime_api::DenseJacobianChunkingStrategy;
use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use crate::symbolic::symbolic_engine::Expr;
use nalgebra::DVector;
use std::time::Duration;

const DEFAULT_RUNS: usize = 5;

#[derive(Debug, Clone)]
struct WarmSample {
    x: DVector<f64>,
    total_ms: f64,
    residual_ms: f64,
    jacobian_ms: f64,
    linear_ms: f64,
    counters: (usize, usize, usize, usize),
}

fn dimensions() -> Vec<usize> {
    std::env::var("NONLINEAR_AOT_LARGE_DIMENSIONS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|part| part.trim().parse::<usize>().ok())
                .filter(|dimension| *dimension >= 8)
                .collect::<Vec<_>>()
        })
        .filter(|dimensions| !dimensions.is_empty())
        .unwrap_or_else(|| vec![128])
}

fn runs() -> usize {
    std::env::var("NONLINEAR_AOT_LARGE_RUNS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|runs| *runs >= 3)
        .unwrap_or(DEFAULT_RUNS)
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

fn dense_diagonal_problem(dimension: usize) -> (Vec<Expr>, Vec<String>, DVector<f64>) {
    let variables = (0..dimension)
        .map(|index| format!("x{index}"))
        .collect::<Vec<_>>();
    let equations = variables
        .iter()
        .map(|variable| Expr::parse_expression(&format!("{variable}^2-1.0")))
        .collect::<Vec<_>>();
    let initial = DVector::from_element(dimension, 0.9);
    (equations, variables, initial)
}

fn problem_options(variables: Vec<String>) -> SymbolicProblemOptions {
    SymbolicProblemOptions::new().with_variables(variables)
}

fn solve_options() -> SolveOptions {
    SolveOptions {
        tolerance: 1e-11,
        max_iterations: 30,
        diagnostics: DiagnosticsOptions {
            collect_history: false,
            collect_statistics: true,
            ..DiagnosticsOptions::default()
        },
        ..SolveOptions::default()
    }
}

fn collect_samples<P: crate::numerical::Nonlinear_systems::problem::JacobianProvider>(
    problem: &P,
    initial: &DVector<f64>,
    options: &SolveOptions,
    repetitions: usize,
) -> Vec<WarmSample> {
    (0..repetitions)
        .map(|_| {
            let result = SolverEngine::new(NewtonMethod, options.clone())
                .solve(problem, initial.clone())
                .expect("large AOT/Lambdify solve should succeed");
            assert_eq!(
                result.termination,
                TerminationReason::Converged,
                "large comparison solve must converge"
            );
            assert!(result.residual_norm < options.tolerance);
            let statistics = result.statistics;
            WarmSample {
                x: result.x,
                total_ms: duration_ms(statistics.total_duration),
                residual_ms: duration_ms(statistics.residual_duration),
                jacobian_ms: duration_ms(statistics.jacobian_duration),
                linear_ms: duration_ms(statistics.linear_solve_duration),
                counters: (
                    statistics.residual_evaluations,
                    statistics.jacobian_evaluations,
                    statistics.linear_solves,
                    statistics.iterations,
                ),
            }
        })
        .collect()
}

fn mean_std_min_max(values: &[f64]) -> String {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    format!("{mean:.3}+/-{:.3}[{min:.3},{max:.3}]", variance.sqrt())
}

fn stage_values(samples: &[WarmSample], select: impl Fn(&WarmSample) -> f64) -> String {
    let values = samples.iter().map(select).collect::<Vec<_>>();
    mean_std_min_max(&values)
}

fn mean_counters(samples: &[WarmSample]) -> (f64, f64, f64, f64) {
    let sums = samples.iter().fold([0.0; 4], |mut sums, sample| {
        let (residuals, jacobians, linear_solves, iterations) = sample.counters;
        for (slot, value) in sums
            .iter_mut()
            .zip([residuals, jacobians, linear_solves, iterations])
        {
            *slot += value as f64;
        }
        sums
    });
    (
        sums[0] / samples.len() as f64,
        sums[1] / samples.len() as f64,
        sums[2] / samples.len() as f64,
        sums[3] / samples.len() as f64,
    )
}

fn assert_same_counters(left: &[WarmSample], right: &[WarmSample]) {
    assert!(!left.is_empty() && !right.is_empty());
    assert_eq!(left[0].counters, right[0].counters);
    assert!(
        left.iter()
            .all(|sample| sample.counters == left[0].counters)
    );
    assert!(
        right
            .iter()
            .all(|sample| sample.counters == right[0].counters)
    );
}

fn assert_solution_agreement(left: &[WarmSample], right: &[WarmSample]) {
    let max_diff = left
        .iter()
        .flat_map(|left_sample| {
            right.iter().map(move |right_sample| {
                left_sample
                    .x
                    .iter()
                    .zip(right_sample.x.iter())
                    .map(|(left, right)| (left - right).abs())
                    .fold(0.0, f64::max)
            })
        })
        .fold(0.0, f64::max);
    assert!(max_diff < 1e-10, "Lambdify/AOT max_diff={max_diff:.3e}");
}

fn print_route(
    dimension: usize,
    route: &str,
    samples: &[WarmSample],
    preparation_ms: f64,
    build_ms: Option<f64>,
) {
    let counters = mean_counters(samples);
    println!(
        "{dimension:>9} | {route:<19} | {preparation_ms:>14.3} | {:>12} | {:>19} | {:>19} | {:>19} | {:>19} | {:>7.1}/{:>7.1}/{:>7.1}/{:>7.1}",
        build_ms
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "-".to_string()),
        stage_values(samples, |sample| sample.total_ms),
        stage_values(samples, |sample| sample.residual_ms),
        stage_values(samples, |sample| sample.jacobian_ms),
        stage_values(samples, |sample| sample.linear_ms),
        counters.0,
        counters.1,
        counters.2,
        counters.3,
    );
}

#[test]
#[ignore = "release-oriented large warm Lambdify versus generated AOT story"]
fn large_dense_aot_vs_lambdify_warm_stage_story() {
    let _guard = aot_solver_test_guard();
    let repetitions = runs();
    println!(
        "[Nonlinear AOT large warm story] dimensions={:?}; runs={repetitions}; build/preparation excluded from warm stages",
        dimensions()
    );
    println!(
        "dimension | route               | preparation_ms | build_ms     | total_ms mean+/-std[min,max] | residual_ms mean+/-std[min,max] | jacobian_ms mean+/-std[min,max] | linear_ms mean+/-std[min,max] | mean R/J/L/I"
    );

    for dimension in dimensions() {
        let (equations, variables, initial) = dense_diagonal_problem(dimension);
        let options = solve_options();
        // Keep generated functions comfortably small without creating one
        // dynamic-dispatch call per diagonal entry. Structural-zero elision
        // makes a 32-row block safe for this corpus; the benchmark should
        // measure the AOT runtime, not an artificial 256-chunk dispatch tax.
        let jacobian_rows_per_chunk = dimension.min(32);
        let aot_options = SymbolicDenseAotOptions {
            jacobian_strategy: DenseJacobianChunkingStrategy::ByRowCount {
                rows_per_chunk: jacobian_rows_per_chunk,
            },
            ..SymbolicDenseAotOptions::default()
        };
        println!(
            "[Nonlinear AOT large warm story] dimension={dimension}; jacobian_rows_per_chunk={jacobian_rows_per_chunk}; expected_jacobian_chunks={}",
            dimension.div_ceil(jacobian_rows_per_chunk)
        );

        let lambdify_started = std::time::Instant::now();
        let lambdify_prepared = PreparedSymbolicNonlinearProblem::from_expressions(
            equations.clone(),
            problem_options(variables.clone()),
        )
        .expect("large Lambdify preparation should succeed");
        let lambdify = lambdify_prepared
            .bind_without_parameters()
            .expect("large Lambdify binding should succeed");
        let lambdify_preparation_ms = duration_ms(lambdify_started.elapsed());
        let lambdify_samples = collect_samples(&lambdify, &initial, &options, repetitions);

        let output_dir = tempfile::tempdir().expect("AOT output directory should exist");
        let aot_started = std::time::Instant::now();
        let built = SymbolicNonlinearProblem::from_expressions_with_generated_backend(
            equations.clone(),
            problem_options(variables),
            SymbolicGeneratedBackendConfig::new()
                .with_backend_policy_override(Some(
                    crate::numerical::Nonlinear_systems::symbolic_backend::
                        SymbolicBackendSelectionPolicy::AotOnly,
                ))
                .with_build_policy(SymbolicAotBuildPolicy::BuildIfMissing {
                    profile: AotBuildProfile::Release,
                })
                .with_aot_options(aot_options)
                .with_output_parent_dir(Some(output_dir.path().to_path_buf())),
        )
        .expect("large AOT build should succeed");
        assert_eq!(
            built.selected_backend,
            SelectedSymbolicNonlinearBackendKind::AotCompiled
        );
        let aot_preparation_ms = duration_ms(aot_started.elapsed());
        let aot_build_ms = built.preparation_report.build_duration.map(duration_ms);
        let resolver = built
            .updated_resolver
            .clone()
            .expect("large AOT build should publish a resolver");
        let problem_key = built
            .preparation_report
            .artifact_key
            .clone()
            .expect("large AOT build should publish an artifact key");
        let aot_prepared = built.into_prepared();
        let aot = aot_prepared
            .bind_without_parameters()
            .expect("large AOT binding should succeed");
        let aot_samples = collect_samples(&aot, &initial, &options, repetitions);

        let strict = SymbolicNonlinearProblem::from_expressions_with_generated_backend(
            equations,
            problem_options((0..dimension).map(|index| format!("x{index}")).collect()),
            SymbolicGeneratedBackendConfig::require_prebuilt()
                .with_aot_options(aot_options)
                .with_resolver(Some(resolver)),
        )
        .expect("large AOT strict reuse should succeed");
        assert!(strict.build_result.is_none());
        assert!(strict.preparation_report.build_duration.is_none());
        assert_eq!(
            strict.selected_backend,
            SelectedSymbolicNonlinearBackendKind::AotCompiled
        );

        assert_same_counters(&lambdify_samples, &aot_samples);
        assert_solution_agreement(&lambdify_samples, &aot_samples);
        print_route(
            dimension,
            "Lambdify warm",
            &lambdify_samples,
            lambdify_preparation_ms,
            None,
        );
        print_route(
            dimension,
            "AOT warm",
            &aot_samples,
            aot_preparation_ms,
            aot_build_ms,
        );
        println!(
            "[Nonlinear AOT large warm story] dimension={dimension}; strict RequirePrebuilt preparation_ms={:.3}; strict_build=none; artifact_key={problem_key}",
            duration_ms(strict.preparation_report.preparation_duration)
        );
        assert!(unregister_linked_dense_backend(&problem_key).is_some());
    }
}
