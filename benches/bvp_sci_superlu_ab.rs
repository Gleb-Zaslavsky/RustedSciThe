//! Experimental matched full-solve A/B for opt-in SuperLU and default faer.
//!
//! Symbolic preparation and runtime construction are reported separately from
//! `solve()`. Both routes receive the same linear two-state BVP, mesh, initial
//! guess, tolerances and telemetry mode; only the sparse LU policy changes.

use std::time::{Duration, Instant};

use RustedSciThe::Utils::test_reporting::write_test_report;
use RustedSciThe::numerical::BVP_sci::new::{
    BvpSciAssembly, BvpSciBoundaryCallbacks, BvpSciExecution, BvpSciLambdifyPlan,
    BvpSciMatrixLayout, BvpSciOptions, BvpSciSolver, BvpSciSparseFactorization, BvpSciStatus,
    BvpSciTelemetry, BvpSciTelemetryMode, BvpSciTelemetrySnapshot,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use tabled::{Table, Tabled};

#[derive(Clone, Copy)]
enum Route {
    StandardFaer,
    SuperLu,
}

impl Route {
    fn label(self) -> &'static str {
        match self {
            Self::StandardFaer => "StandardFaer",
            Self::SuperLu => "SuperLU-opt-in",
        }
    }

    fn policy(self) -> BvpSciSparseFactorization {
        match self {
            Self::StandardFaer => BvpSciSparseFactorization::StandardFaer,
            Self::SuperLu => BvpSciSparseFactorization::SuperLu,
        }
    }
}

struct Sample {
    prepare: Duration,
    runtime_setup: Duration,
    solve: Duration,
    snapshot: BvpSciTelemetrySnapshot,
    solution: Vec<f64>,
    mesh_nodes: usize,
    status: BvpSciStatus,
    residual_norm: f64,
    iterations: usize,
}

#[derive(Debug, Tabled)]
struct Row {
    route: &'static str,
    effective_backend: &'static str,
    superlu_fallbacks: u64,
    requested_nodes: usize,
    final_nodes: usize,
    system_dimension: usize,
    repeats: usize,
    prepare_median_ms: String,
    runtime_setup_median_ms: String,
    solve_wall_median_ms: String,
    solve_telemetry_median_ms: String,
    assembly_median_ms: String,
    factorization_median_ms: String,
    linear_solve_median_ms: String,
    factorizations_median: String,
    iterations_median: String,
    residual_norm_max: String,
    paired_solution_diff: String,
    status: &'static str,
}

fn env_usize(name: &str, default: &str) -> Vec<usize> {
    std::env::var(name)
        .unwrap_or_else(|_| default.to_owned())
        .split(',')
        .filter_map(|value| value.trim().parse().ok())
        .filter(|value: &usize| *value >= 8)
        .collect()
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some(values[values.len() / 2])
}

fn median_ms(values: impl IntoIterator<Item = Option<f64>>) -> String {
    median(values.into_iter().flatten().collect())
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "n/a".to_owned())
}

fn median_duration_ms(values: impl IntoIterator<Item = Duration>) -> String {
    median(
        values
            .into_iter()
            .map(|duration| duration.as_secs_f64() * 1_000.0)
            .collect(),
    )
    .map(|value| format!("{value:.3}"))
    .unwrap_or_else(|| "n/a".to_owned())
}

fn median_u64(values: impl IntoIterator<Item = u64>) -> String {
    let mut values: Vec<_> = values.into_iter().collect();
    values.sort_unstable();
    values
        .get(values.len() / 2)
        .map(u64::to_string)
        .unwrap_or_else(|| "n/a".to_owned())
}

fn build_and_solve(route: Route, nodes: usize) -> Result<Sample, String> {
    let telemetry = BvpSciTelemetry::detailed();
    let equations = [Expr::parse_expression("z"), Expr::parse_expression("0")];
    let state_names = ["y".to_owned(), "z".to_owned()];

    let prepare_started = Instant::now();
    let plan = BvpSciLambdifyPlan::prepare(
        BvpSciAssembly::ExprLegacy,
        &equations,
        &state_names,
        &[],
        "x",
        telemetry.clone(),
    )
    .map_err(|error| format!("prepare failed: {error}"))?;
    let prepare = prepare_started.elapsed();

    let boundary = BvpSciBoundaryCallbacks::new(
        2,
        |ya, yb, _, output| {
            output[0] = ya[0];
            output[1] = yb[0] - 1.0;
            Ok(())
        },
        telemetry.clone(),
    );
    let x: Vec<_> = (0..nodes)
        .map(|index| index as f64 / (nodes - 1) as f64)
        .collect();
    let y: Vec<_> = x.iter().flat_map(|&x| [0.9 * x, 0.9]).collect();
    let options = BvpSciOptions::default()
        .with_execution(BvpSciExecution::Lambdify)
        .with_assembly(BvpSciAssembly::ExprLegacy)
        .with_matrix_layout(BvpSciMatrixLayout::Sparse)
        .with_tolerance(1e-8)
        .with_limits(nodes.saturating_mul(2), 12, 0)
        .with_telemetry(BvpSciTelemetryMode::Detailed);

    let setup_started = Instant::now();
    let mut solver = BvpSciSolver::new_with_sparse_factorization(
        plan,
        boundary,
        x,
        y,
        vec![],
        options,
        route.policy(),
    )
    .map_err(|error| format!("solver construction failed: {error}"))?;
    let runtime_setup = setup_started.elapsed();

    let solve_started = Instant::now();
    let solution = solver
        .solve()
        .map_err(|error| format!("solve failed: {error}"))?;
    let solve = solve_started.elapsed();
    let snapshot = telemetry.snapshot();

    if solution.status != BvpSciStatus::Success {
        return Err(format!(
            "{} returned {:?}: {}",
            route.label(),
            solution.status,
            solution.message
        ));
    }

    Ok(Sample {
        prepare,
        runtime_setup,
        solve,
        snapshot,
        solution: solution.y,
        mesh_nodes: solution.x.len(),
        status: solution.status,
        residual_norm: solution.residual_norm,
        iterations: solution.iterations,
    })
}

fn paired_diff(faer: &[Sample], superlu: &[Sample]) -> f64 {
    faer.iter()
        .zip(superlu)
        .map(|(left, right)| {
            if left.solution.len() != right.solution.len() {
                return f64::INFINITY;
            }
            left.solution
                .iter()
                .zip(&right.solution)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max)
        })
        .fold(0.0, f64::max)
}

fn row(route: Route, requested_nodes: usize, samples: &[Sample], diff: f64) -> Row {
    let first = samples.first().expect("at least one sample");
    let fallback_count: u64 = samples
        .iter()
        .map(|sample| sample.snapshot.superlu_fallbacks)
        .sum();
    let effective_backend = match route {
        Route::StandardFaer => "StandardFaer",
        Route::SuperLu if fallback_count > 0 => "StandardFaer-fallback",
        Route::SuperLu => "SuperLU",
    };
    let telemetry = |select: fn(&BvpSciTelemetrySnapshot) -> Option<f64>| {
        median_ms(samples.iter().map(|sample| select(&sample.snapshot)))
    };
    Row {
        route: route.label(),
        effective_backend,
        superlu_fallbacks: fallback_count,
        requested_nodes,
        final_nodes: first.mesh_nodes,
        system_dimension: first.mesh_nodes * 2,
        repeats: samples.len(),
        prepare_median_ms: median_duration_ms(samples.iter().map(|sample| sample.prepare)),
        runtime_setup_median_ms: median_duration_ms(
            samples.iter().map(|sample| sample.runtime_setup),
        ),
        solve_wall_median_ms: median_duration_ms(samples.iter().map(|sample| sample.solve)),
        solve_telemetry_median_ms: telemetry(|snapshot| snapshot.full_solve_ms),
        assembly_median_ms: telemetry(|snapshot| snapshot.linear_assembly_ms),
        factorization_median_ms: telemetry(|snapshot| snapshot.factorization_ms),
        linear_solve_median_ms: telemetry(|snapshot| snapshot.linear_solve_ms),
        factorizations_median: median_u64(
            samples.iter().map(|sample| sample.snapshot.factorizations),
        ),
        iterations_median: median_u64(samples.iter().map(|sample| sample.iterations as u64)),
        residual_norm_max: format!(
            "{:.3e}",
            samples
                .iter()
                .map(|sample| sample.residual_norm)
                .fold(0.0, f64::max)
        ),
        paired_solution_diff: format!("{diff:.3e}"),
        status: if matches!(route, Route::SuperLu) && fallback_count > 0 {
            "fallback-not-superlu"
        } else if samples
            .iter()
            .all(|sample| sample.status == BvpSciStatus::Success)
            && diff <= 1e-8
        {
            "ok"
        } else {
            "mismatch"
        },
    }
}

fn main() {
    let nodes = env_usize("BVP_SCI_SUPERLU_AB_NODES", "128,512,1024");
    let repeats = std::env::var("BVP_SCI_SUPERLU_AB_REPEATS")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value: &usize| *value >= 3)
        .unwrap_or(5);
    assert!(!nodes.is_empty(), "no valid mesh sizes selected");

    let mut rows = Vec::new();
    for mesh_nodes in nodes {
        let mut faer_samples = Vec::with_capacity(repeats);
        let mut superlu_samples = Vec::with_capacity(repeats);
        for repeat in 0..repeats {
            // Alternate execution order to reduce systematic thermal/order bias.
            let pair = if repeat % 2 == 0 {
                [Route::StandardFaer, Route::SuperLu]
            } else {
                [Route::SuperLu, Route::StandardFaer]
            };
            for route in pair {
                let sample = build_and_solve(route, mesh_nodes)
                    .unwrap_or_else(|error| panic!("{} n={mesh_nodes}: {error}", route.label()));
                match route {
                    Route::StandardFaer => faer_samples.push(sample),
                    Route::SuperLu => superlu_samples.push(sample),
                }
            }
        }

        let diff = paired_diff(&faer_samples, &superlu_samples);
        rows.push(row(Route::StandardFaer, mesh_nodes, &faer_samples, diff));
        rows.push(row(Route::SuperLu, mesh_nodes, &superlu_samples, diff));
    }

    let table = Table::new(&rows).to_string();
    println!(
        "[BVP_sci SuperLU A/B] same ExprLegacy workload; preparation, runtime setup and solve are separate"
    );
    println!("[BVP_sci SuperLU A/B] fallback rows are not valid SuperLU performance evidence");
    println!("{table}");
    let body = format!(
        "# BVP_sci SuperLU versus StandardFaer full-solve A/B\n\n\
         - Workload: linear two-state BVP (`y'=z`, `z'=0`) with identical mesh and initial state.\n\
         - Only the sparse factorization policy changes; symbolic frontend is ExprLegacy for both.\n\
         - telemetry: `Detailed` collects counters, timings and detailed traces; solve durations include instrumentation and are diagnostic, not an uninstrumented speed baseline.\n\
         - `prepare`, runtime construction and `solve` are timed separately. Full-solve telemetry is inclusive and not additive with child stages.\n\
         - Each row is the median of {repeats} independent fresh solver instances; execution order alternates.\n\
         - Paired solution max-difference threshold: 1e-8.\n\
         - If the bridge cannot load, the requested route falls back to StandardFaer and is marked `fallback-not-superlu`; this is not SuperLU performance evidence.\n\n{table}\n"
    );
    let report = write_test_report("BVP_sci_SuperLU_Bench", "superlu_full_solve_ab", &body)
        .expect("write compact SuperLU A/B report");
    println!("report={}", report.display());
    assert!(
        rows.iter().all(|row| row.status == "ok"),
        "A/B gate failed: parity mismatch or requested SuperLU route fell back; inspect the compact report"
    );
}
