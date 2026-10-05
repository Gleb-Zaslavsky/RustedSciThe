//! Provisional release benchmark for repeated Frozen BVP solves.
//!
//! This benchmark deliberately measures the current provisional runtime,
//! including the internal Dense/faer factor-owner prototype. It must be rerun
//! after factor-owner validation; its numbers are not a production baseline.
//!
//! ```text
//! cargo bench --release --bench bvp_frozen_runtime_benches -- --noplot
//! ```

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;
use std::hint::black_box;

use RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_frozen::{FrozenSolverOptions, NRBVP};
use RustedSciThe::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig;
use RustedSciThe::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;

const RUNS: usize = 5;

#[derive(Clone, Copy)]
enum Route {
    Dense,
    SparseFaer,
    Banded,
}

impl Route {
    fn name(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::SparseFaer => "faer-sparse",
            Self::Banded => "banded",
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

fn frozen_linear_solver(n_steps: usize, route: Route) -> NRBVP {
    let values = vec!["y".to_string(), "z".to_string()];
    let mut guess = vec![0.0; values.len() * n_steps];
    for i in 0..n_steps {
        guess[2 * i] = 0.25;
        guess[2 * i + 1] = 0.75;
    }

    let mut solver = NRBVP::new_with_options(
        vec![
            RustedSciThe::symbolic::symbolic_engine::Expr::parse_expression("z"),
            RustedSciThe::symbolic::symbolic_engine::Expr::parse_expression("0.0"),
        ],
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

fn benchmark_repeated_frozen_solves(c: &mut Criterion) {
    let mut group = c.benchmark_group("bvp_frozen_repeated_solves_provisional");
    for n_steps in [32usize, 128usize] {
        for route in [Route::Dense, Route::SparseFaer, Route::Banded] {
            group.bench_with_input(
                BenchmarkId::new(route.name(), n_steps),
                &n_steps,
                |bencher, &n_steps| {
                    bencher.iter(|| {
                        for _ in 0..RUNS {
                            let mut solver = frozen_linear_solver(n_steps, route);
                            let result = solver
                                .try_solve()
                                .expect("provisional Frozen benchmark should solve")
                                .expect("provisional Frozen benchmark should converge");
                            black_box(result);
                        }
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, benchmark_repeated_frozen_solves);
criterion_main!(benches);
