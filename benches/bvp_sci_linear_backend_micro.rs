//! Small direct-backend benchmark for the BVP_sci linear bottleneck.
//!
//! This is intentionally not a solver benchmark. It keeps the matrix entries
//! fixed and reports assembly, factorization and in-place solve separately so
//! a structured Banded regression cannot hide behind nonlinear/controller
//! work. The default slice is Sparse versus Banded; add `dense` explicitly for
//! compact small-system comparisons.

use std::time::Instant;

use RustedSciThe::Utils::test_reporting::write_test_report;
use RustedSciThe::numerical::BVP_sci::new::backends::{BandedBackend, DenseBackend, SparseBackend};
use RustedSciThe::numerical::BVP_sci::new::{BvpSciTelemetry, BvpSciTelemetrySnapshot};
use tabled::{Table, Tabled};

#[derive(Clone, Copy)]
enum Route {
    Dense,
    Sparse,
    Banded,
}

impl Route {
    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "dense" => Some(Self::Dense),
            "sparse" => Some(Self::Sparse),
            "banded" => Some(Self::Banded),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Dense => "Dense",
            Self::Sparse => "Sparse/faer",
            Self::Banded => "Banded/structured",
        }
    }
}

enum Backend {
    Dense(DenseBackend),
    Sparse(SparseBackend),
    Banded(BandedBackend),
}

impl Backend {
    fn new(route: Route, dimension: usize) -> Self {
        match route {
            Route::Dense => Self::Dense(DenseBackend::new(dimension)),
            Route::Sparse => Self::Sparse(SparseBackend::new_with_telemetry(
                dimension,
                dimension * 5,
                BvpSciTelemetry::timings(),
            )),
            Route::Banded => Self::Banded(
                BandedBackend::new_for_collocation(
                    dimension,
                    1,
                    1,
                    1,
                    dimension,
                    0,
                    BvpSciTelemetry::timings(),
                )
                .expect("microbench Banded backend should construct"),
            ),
        }
    }

    fn assemble(&mut self, entries: &[(usize, usize, f64)]) {
        match self {
            Self::Dense(backend) => backend.assemble(entries),
            Self::Sparse(backend) => backend.assemble(entries),
            Self::Banded(backend) => backend.assemble(entries),
        }
        .expect("microbench matrix should assemble");
    }

    fn factor(&mut self) {
        match self {
            Self::Dense(backend) => backend.factor(),
            Self::Sparse(backend) => backend.factor(),
            Self::Banded(backend) => backend.factor(),
        }
        .expect("microbench matrix should factor");
    }

    fn solve(&mut self, rhs: &mut [f64]) {
        match self {
            Self::Dense(backend) => backend.solve_in_place(rhs),
            Self::Sparse(backend) => backend.solve_in_place(rhs),
            Self::Banded(backend) => backend.solve_in_place(rhs),
        }
        .expect("microbench factorization should solve");
    }

    fn telemetry_snapshot(&self) -> Option<BvpSciTelemetrySnapshot> {
        match self {
            Self::Sparse(backend) => Some(backend.telemetry_snapshot()),
            Self::Banded(backend) => backend.telemetry_snapshot(),
            Self::Dense(_) => None,
        }
    }
}

#[derive(Debug, Tabled)]
struct Row {
    route: String,
    dimension: usize,
    entries: usize,
    repeats: usize,
    assembly_ms: String,
    factorization_ms: String,
    sparse_symbolic_analysis_ms: String,
    sparse_numeric_factorization_ms: String,
    sparse_symbolic_analyses: String,
    sparse_numeric_factorizations: String,
    solve_us: String,
    structured_factor_ms: String,
    residual_guard_ms: String,
    rhs_permutation_ms: String,
    structured_solve_us: String,
    fallback_switches: String,
    status: &'static str,
}

fn parse_usizes(name: &str, default: &str) -> Vec<usize> {
    std::env::var(name)
        .unwrap_or_else(|_| default.into())
        .split(',')
        .filter_map(|value| value.trim().parse().ok())
        .filter(|value: &usize| *value >= 4)
        .collect()
}

fn selected_routes() -> Vec<Route> {
    std::env::var("BVP_SCI_LINEAR_MICRO_ROUTES")
        .unwrap_or_else(|_| "sparse,banded".into())
        .split(',')
        .filter_map(Route::parse)
        .collect()
}

fn matrix_entries(dimension: usize) -> Vec<(usize, usize, f64)> {
    let mut entries = Vec::with_capacity(dimension * 5 + 4);
    // A stiff tridiagonal core models the local collocation coupling. The two
    // endpoint rows also receive distant border entries, which is the part
    // that makes a scalar global band a poor representation of BVP Jacobians.
    entries.push((0, 0, 2.0));
    entries.push((0, dimension - 1, 0.1));
    entries.push((dimension - 1, 0, 0.1));
    entries.push((dimension - 1, dimension - 1, 2.0));
    for row in 1..dimension - 1 {
        entries.push((row, row - 1, -1.0));
        entries.push((row, row, 24.0));
        entries.push((row, row + 1, -1.0));
        // Keep a weak second-neighbour coupling so sparse assembly/factor fill
        // is visible without changing the structured Banded core contract.
        if row + 2 < dimension {
            entries.push((row, row + 2, -0.05));
        }
    }
    entries
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn stage_delta(
    before: &Option<BvpSciTelemetrySnapshot>,
    after: &Option<BvpSciTelemetrySnapshot>,
    select: impl Fn(&BvpSciTelemetrySnapshot) -> Option<f64>,
) -> f64 {
    let before = before.as_ref().and_then(&select).unwrap_or(0.0);
    let after = after.as_ref().and_then(select).unwrap_or(0.0);
    (after - before).max(0.0)
}

fn count_delta(
    before: &Option<BvpSciTelemetrySnapshot>,
    after: &Option<BvpSciTelemetrySnapshot>,
    select: impl Fn(&BvpSciTelemetrySnapshot) -> u64,
) -> u64 {
    let before = before.as_ref().map(&select).unwrap_or(0);
    let after = after.as_ref().map(select).unwrap_or(0);
    after.saturating_sub(before)
}

fn measure(route: Route, dimension: usize, entries: &[(usize, usize, f64)]) -> Row {
    let repeats = if dimension >= 1024 { 3 } else { 7 };
    let rhs = vec![1.0; dimension];
    let mut assembly = Vec::with_capacity(repeats);
    let mut factorization = Vec::with_capacity(repeats);
    let mut symbolic_analysis_samples = Vec::with_capacity(repeats);
    let mut numeric_factorization_samples = Vec::with_capacity(repeats);
    let mut symbolic_analysis_calls = Vec::with_capacity(repeats);
    let mut numeric_factorization_calls = Vec::with_capacity(repeats);
    let mut solve = Vec::with_capacity(repeats);
    let mut structured_factor = Vec::with_capacity(repeats);
    let mut residual_guard = Vec::with_capacity(repeats);
    let mut rhs_permutation = Vec::with_capacity(repeats);
    let mut structured_solve = Vec::with_capacity(repeats);
    let mut fallback_switches = Vec::with_capacity(repeats);
    let mut backend = Backend::new(route, dimension);

    for _ in 0..repeats {
        let started = Instant::now();
        backend.assemble(entries);
        assembly.push(started.elapsed().as_secs_f64() * 1e3);

        let started = Instant::now();
        let before_factor = backend.telemetry_snapshot();
        backend.factor();
        factorization.push(started.elapsed().as_secs_f64() * 1e3);
        let after_factor = backend.telemetry_snapshot();
        let symbolic_analysis_ms = stage_delta(&before_factor, &after_factor, |snapshot| {
            snapshot.sparse_symbolic_analysis_ms
        });
        let numeric_factorization_ms = stage_delta(&before_factor, &after_factor, |snapshot| {
            snapshot.sparse_numeric_factorization_ms
        });
        symbolic_analysis_samples.push(symbolic_analysis_ms);
        numeric_factorization_samples.push(numeric_factorization_ms);
        symbolic_analysis_calls.push(count_delta(&before_factor, &after_factor, |snapshot| {
            snapshot.sparse_symbolic_analyses
        }) as f64);
        numeric_factorization_calls.push(count_delta(&before_factor, &after_factor, |snapshot| {
            snapshot.sparse_numeric_factorizations
        }) as f64);
        structured_factor.push(stage_delta(&before_factor, &after_factor, |snapshot| {
            snapshot.banded_structured_factorization_ms
        }));

        let mut trial_rhs = rhs.clone();
        let started = Instant::now();
        let before_solve = backend.telemetry_snapshot();
        backend.solve(&mut trial_rhs);
        solve.push(started.elapsed().as_secs_f64() * 1e6);
        let after_solve = backend.telemetry_snapshot();
        residual_guard.push(stage_delta(&before_solve, &after_solve, |snapshot| {
            snapshot.banded_residual_guard_ms
        }));
        rhs_permutation.push(stage_delta(&before_solve, &after_solve, |snapshot| {
            snapshot.banded_rhs_permutation_ms
        }));
        structured_solve.push(
            stage_delta(&before_solve, &after_solve, |snapshot| {
                snapshot.banded_structured_solve_ms
            }) * 1e3,
        );
        let before_switches = before_solve
            .as_ref()
            .map(|snapshot| snapshot.banded_fallback_switches)
            .unwrap_or(0);
        let after_switches = after_solve
            .as_ref()
            .map(|snapshot| snapshot.banded_fallback_switches)
            .unwrap_or(0);
        fallback_switches.push(after_switches.saturating_sub(before_switches) as f64);
        assert!(trial_rhs.iter().all(|value| value.is_finite()));
    }

    let symbolic_analysis_total_ms: f64 = symbolic_analysis_samples.iter().sum();
    let numeric_factorization_median_ms = median(&mut numeric_factorization_samples);
    let symbolic_analysis_count_total: f64 = symbolic_analysis_calls.iter().sum();
    let numeric_factorization_count_total: f64 = numeric_factorization_calls.iter().sum();

    Row {
        route: route.label().into(),
        dimension,
        entries: entries.len(),
        repeats,
        assembly_ms: format!("{:.3}", median(&mut assembly)),
        factorization_ms: format!("{:.3}", median(&mut factorization)),
        sparse_symbolic_analysis_ms: if symbolic_analysis_count_total > 0.0 {
            format!("{symbolic_analysis_total_ms:.3}")
        } else {
            "-".into()
        },
        sparse_numeric_factorization_ms: if numeric_factorization_count_total > 0.0 {
            format!("{numeric_factorization_median_ms:.3}")
        } else {
            "-".into()
        },
        sparse_symbolic_analyses: if symbolic_analysis_count_total > 0.0 {
            format!("{symbolic_analysis_count_total:.0}")
        } else {
            "-".into()
        },
        sparse_numeric_factorizations: if numeric_factorization_count_total > 0.0 {
            format!("{numeric_factorization_count_total:.0}")
        } else {
            "-".into()
        },
        solve_us: format!("{:.3}", median(&mut solve)),
        structured_factor_ms: if structured_factor.is_empty() {
            "-".into()
        } else {
            format!("{:.3}", median(&mut structured_factor))
        },
        residual_guard_ms: if residual_guard.is_empty() {
            "-".into()
        } else {
            format!("{:.3}", median(&mut residual_guard))
        },
        rhs_permutation_ms: if rhs_permutation.is_empty() {
            "-".into()
        } else {
            format!("{:.3}", median(&mut rhs_permutation))
        },
        structured_solve_us: if structured_solve.is_empty() {
            "-".into()
        } else {
            format!("{:.3}", median(&mut structured_solve))
        },
        fallback_switches: if fallback_switches.is_empty() {
            "-".into()
        } else {
            format!("{:.0}", median(&mut fallback_switches))
        },
        status: "ok",
    }
}

fn main() {
    let dimensions = parse_usizes("BVP_SCI_LINEAR_MICRO_DIMENSIONS", "128,256,512,1024");
    let routes = selected_routes();
    assert!(
        !dimensions.is_empty(),
        "microbench requires at least one dimension"
    );
    assert!(!routes.is_empty(), "microbench requires at least one route");

    let mut rows = Vec::new();
    for dimension in dimensions {
        let entries = matrix_entries(dimension);
        for route in routes.iter().copied() {
            rows.push(measure(route, dimension, &entries));
        }
    }
    let mut measured_dimensions = rows.iter().map(|row| row.dimension).collect::<Vec<_>>();
    measured_dimensions.sort_unstable();
    measured_dimensions.dedup();
    let table = Table::new(&rows).to_string();
    let body = format!(
        "# BVP_sci linear backend microbench\n\n- routes: {:?}\n- dimensions: {:?}\n- matrix: stiff tridiagonal core with endpoint border coupling\n- factorization and solve are measured after a separate assembly phase\n- `sparse_symbolic_analysis_ms` is cumulative cold-pattern work; `sparse_numeric_factorization_ms` is the warm median per numeric refresh\n- sparse counters are cumulative over the repeated factorization series\n- timings are local medians/summaries, not release thresholds\n\n{}",
        routes.iter().map(|route| route.label()).collect::<Vec<_>>(),
        measured_dimensions,
        table
    );
    let report_name = std::env::var("BVP_SCI_LINEAR_MICRO_REPORT")
        .unwrap_or_else(|_| "linear_backend_microbench".into());
    let path = write_test_report("BVP_sci_Linear_Microbench", &report_name, &body)
        .expect("write BVP_sci linear microbench report");
    println!("report={}", path.display());
}
