//! Provisional release-oriented Frozen runtime story.
//!
//! The story intentionally stays separate from legacy API tests. It records
//! the current Dense/faer/Banded stage split while the internal factor-owner
//! runtime is still under validation, so every result is dated and explicitly
//! non-production.

use crate::Utils::test_reporting::write_test_report;
use crate::numerical::BVP_Damp::NR_Damp_solver_frozen::{
    FrozenBvpStatistics, FrozenSolverOptions, NRBVP,
};
use crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig;
use crate::symbolic::bvp::telemetry::BvpGenerationTelemetrySnapshot;
use crate::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
use crate::symbolic::symbolic_engine::Expr;
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::time::{Duration, Instant};

const RECORDED_ON: &str = "2026-09-19";
const RUNS: usize = 5;

#[derive(Clone, Copy)]
enum Route {
    Dense,
    SparseFaer,
    Banded,
}

impl Route {
    fn label(self) -> &'static str {
        match self {
            Self::Dense => "Dense",
            Self::SparseFaer => "faer-Sparse",
            Self::Banded => "Banded",
        }
    }

    fn options(self) -> FrozenSolverOptions {
        match self {
            Self::Dense => FrozenSolverOptions::dense_frozen(),
            Self::SparseFaer => FrozenSolverOptions::sparse_frozen().with_generated_backend_config(
                GeneratedBackendConfig::sparse_defaults()
                    .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly)),
            ),
            Self::Banded => FrozenSolverOptions::banded_frozen().with_banded_lambdify(),
        }
        .with_tolerance(1e-8)
        .with_max_iterations(12)
    }
}

#[derive(Clone, Default)]
struct Sample {
    wall_ms: f64,
    total_ms: f64,
    symbolic_ms: f64,
    reported_prepare_ms: f64,
    preparation_gap_ms: f64,
    handoff_total_ms: f64,
    initial_generate_ms: f64,
    execution_bind_ms: f64,
    residual_ms: f64,
    jacobian_ms: f64,
    linear_ms: f64,
    factor_ms: f64,
    rhs_ms: f64,
    iterations: u64,
    factorizations: u64,
    cache_hits: u64,
    max_error: f64,
    generation: Option<BvpGenerationTelemetrySnapshot>,
}

fn generation_stage_rows(
    generation: BvpGenerationTelemetrySnapshot,
) -> [(&'static str, Duration); 8] {
    [
        ("total", generation.total),
        ("discretization", generation.discretization),
        ("symbolic_jacobian", generation.symbolic_jacobian),
        ("find_bandwidth", generation.find_bandwidth),
        ("backend_selection", generation.backend_selection),
        ("runtime_binding", generation.runtime_binding),
        (
            "lambdify_jacobian_compile",
            generation.lambdify_jacobian_compile,
        ),
        (
            "lambdify_residual_compile",
            generation.lambdify_residual_compile,
        ),
    ]
}

fn diagnostic_ms(diagnostics: &HashMap<String, String>, key: &str) -> f64 {
    diagnostics
        .get(key)
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.0)
}

fn frozen_linear_solver(n_steps: usize, route: Route) -> NRBVP {
    let values = vec!["y".to_string(), "z".to_string()];
    let mut guess = vec![0.0; values.len() * n_steps];
    for i in 0..n_steps {
        guess[2 * i] = 0.25;
        guess[2 * i + 1] = 0.75;
    }

    let mut solver = NRBVP::new_with_options(
        vec![Expr::parse_expression("z"), Expr::parse_expression("0.0")],
        DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(guess).as_slice()),
        values,
        "x".to_string(),
        HashMap::from([
            ("y".to_string(), vec![(0usize, 0.0f64)]),
            ("z".to_string(), vec![(0usize, 1.0f64)]),
        ]),
        0.0,
        1.0,
        n_steps,
        route.options(),
    );
    solver.dont_save_log(true);
    solver
}

fn timed_run(n_steps: usize, route: Route) -> Sample {
    let mut solver = frozen_linear_solver(n_steps, route);
    let begin = Instant::now();
    let result = solver
        .try_solve()
        .unwrap_or_else(|err| panic!("{} Frozen solve failed: {err:?}", route.label()))
        .unwrap_or_else(|| panic!("{} Frozen solve did not converge", route.label()));
    let wall_ms = begin.elapsed().as_secs_f64() * 1e3;
    assert!(result.iter().all(|value| value.is_finite()));

    let stats: FrozenBvpStatistics = solver.get_statistics();
    let telemetry = stats.telemetry;
    let generation = telemetry.generation;
    let timings = telemetry.timings;
    let reported_prepare_ms = generation.map_or(0.0, |value| value.total.as_secs_f64() * 1e3);
    let symbolic_ms = timings.symbolic_operations.as_secs_f64() * 1e3;
    Sample {
        wall_ms,
        total_ms: timings.total.as_secs_f64() * 1e3,
        symbolic_ms,
        reported_prepare_ms,
        preparation_gap_ms: symbolic_ms - reported_prepare_ms,
        handoff_total_ms: diagnostic_ms(&stats.diagnostics, "generated.handoff.total_wall_ms"),
        initial_generate_ms: diagnostic_ms(
            &stats.diagnostics,
            "generated.handoff.initial_generate_wall_ms",
        ),
        execution_bind_ms: diagnostic_ms(
            &stats.diagnostics,
            "generated.handoff.execution_bind_wall_ms",
        ),
        residual_ms: timings.residual.as_secs_f64() * 1e3,
        jacobian_ms: timings.jacobian.as_secs_f64() * 1e3,
        linear_ms: timings.linear_system.as_secs_f64() * 1e3,
        factor_ms: timings.factorization.as_secs_f64() * 1e3,
        rhs_ms: timings.rhs_solve.as_secs_f64() * 1e3,
        iterations: telemetry.counters.iterations,
        factorizations: telemetry.counters.factorizations,
        cache_hits: telemetry.counters.factorization_cache_hits,
        max_error: solver.max_error,
        generation,
    }
}

fn mean(samples: &[Sample], value: impl Fn(&Sample) -> f64) -> f64 {
    samples.iter().map(value).sum::<f64>() / samples.len() as f64
}

#[test]
#[ignore = "release provisional Frozen Dense/faer/Banded runtime story; rerun after factor-owner validation"]
fn frozen_dense_faer_banded_runtime_story() {
    let n_steps = 128;
    let mut report = format!(
        "status: passed\n\nrecorded_on: {RECORDED_ON}\nruns: {RUNS}\nn_steps: {n_steps}\n\n\
         | route | wall_ms mean | solver_total_ms | residual_ms | jacobian_ms | linear_ms | factor_ms | rhs_ms | iterations | factorizations | cache_hits | max_error |\n\
         |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    );
    println!(
        "[BVP Frozen runtime provisional] recorded_on={RECORDED_ON}; runs={RUNS}; n_steps={n_steps}; factor-owner runtime is prototype under validation"
    );
    println!(
        "route | wall_ms mean | solver_total_ms | residual_ms | jacobian_ms | linear_ms | factor_ms | rhs_ms | iterations | factorizations | cache_hits | max_error"
    );
    println!("{}", "-".repeat(150));

    for route in [Route::Dense, Route::SparseFaer, Route::Banded] {
        let samples: Vec<_> = (0..RUNS).map(|_| timed_run(n_steps, route)).collect();
        let first = &samples[0];
        assert!(samples.iter().all(|sample| sample.max_error <= 1e-7));
        println!(
            "[BVP Frozen preparation] route={} symbolic_ms={:.3} reported_prepare_ms={:.3} preparation_gap_ms={:.3}",
            route.label(),
            mean(&samples, |s| s.symbolic_ms),
            mean(&samples, |s| s.reported_prepare_ms),
            mean(&samples, |s| s.preparation_gap_ms),
        );
        println!(
            "[BVP Frozen handoff] route={} total_ms={:.3} initial_generate_ms={:.3} execution_bind_ms={:.3}",
            route.label(),
            mean(&samples, |s| s.handoff_total_ms),
            mean(&samples, |s| s.initial_generate_ms),
            mean(&samples, |s| s.execution_bind_ms),
        );
        if let Some(generation) = first.generation {
            println!("[BVP Frozen preparation] route={} stages:", route.label());
            for (key, value) in generation_stage_rows(generation) {
                println!("  {key}_ms={:.3}", value.as_secs_f64() * 1e3);
            }
        }
        match route {
            Route::Dense | Route::SparseFaer => {
                assert_eq!(first.factorizations, 1);
                assert!(first.cache_hits > 0);
            }
            Route::Banded => {
                assert!(first.factorizations > 0);
                assert!(first.cache_hits > 0);
            }
        }
        let wall_ms = mean(&samples, |s| s.wall_ms);
        let total_ms = mean(&samples, |s| s.total_ms);
        let residual_ms = mean(&samples, |s| s.residual_ms);
        let jacobian_ms = mean(&samples, |s| s.jacobian_ms);
        let linear_ms = mean(&samples, |s| s.linear_ms);
        let factor_ms = mean(&samples, |s| s.factor_ms);
        let rhs_ms = mean(&samples, |s| s.rhs_ms);
        let max_error = mean(&samples, |s| s.max_error);
        println!(
            "{:<12} | {:>13.3} | {:>15.3} | {:>11.3} | {:>11.3} | {:>9.3} | {:>9.3} | {:>6.3} | {:>10} | {:>14} | {:>10} | {:.3e}",
            route.label(),
            wall_ms,
            total_ms,
            residual_ms,
            jacobian_ms,
            linear_ms,
            factor_ms,
            rhs_ms,
            first.iterations,
            first.factorizations,
            first.cache_hits,
            max_error,
        );
        writeln!(
            report,
            "| {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {} | {} | {:.3e} |",
            route.label(),
            wall_ms,
            total_ms,
            residual_ms,
            jacobian_ms,
            linear_ms,
            factor_ms,
            rhs_ms,
            first.iterations,
            first.factorizations,
            first.cache_hits,
            max_error,
        )
        .expect("writing an in-memory test report cannot fail");
        if let Some(generation) = first.generation {
            writeln!(report, "\n### {} preparation stages", route.label())
                .expect("writing an in-memory test report cannot fail");
            for (key, value) in generation_stage_rows(generation) {
                writeln!(report, "- {key}_ms: {:.3}", value.as_secs_f64() * 1e3)
                    .expect("writing an in-memory test report cannot fail");
            }
        }
    }

    if let Err(error) = write_test_report(
        "bvp_damp",
        "frozen_dense_faer_banded_runtime_story",
        &report,
    ) {
        eprintln!("[BVP test report] unable to write Frozen story report: {error}");
    }
}
