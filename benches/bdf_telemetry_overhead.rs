use RustedSciThe::numerical::BDF::BDF_api::{BdfSolverOptions, BdfTelemetryMode, ODEsolver};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use nalgebra::{DMatrix, DVector};
use std::{hint::black_box, time::Duration};

fn make_solver(dimension: usize, mode: BdfTelemetryMode) -> ODEsolver {
    let values: Vec<_> = (0..dimension).map(|index| format!("y{index}")).collect();
    let equations = (0..dimension)
        .map(|index| Expr::parse_expression(&format!("-{}*y{index}", 0.1 + index as f64 * 0.01)))
        .collect();
    let options = BdfSolverOptions::new(
        equations,
        values,
        "t".to_string(),
        "BDF".to_string(),
        0.0,
        DVector::from_element(dimension, 1.0),
        0.03,
        0.001,
        1e-6,
        1e-9,
        None,
        false,
        Some(0.001),
    )
    .with_telemetry_mode(mode);
    let mut solver = ODEsolver::new_with_options(options);
    solver.set_native_ode_callbacks(
        |_: f64, y: &DVector<f64>| {
            DVector::from_fn(y.len(), |index, _| -(0.1 + index as f64 * 0.01) * y[index])
        },
        Some(|_: f64, y: &DVector<f64>| {
            DMatrix::from_fn(y.len(), y.len(), |row, column| {
                if row == column {
                    -(0.1 + row as f64 * 0.01)
                } else {
                    0.0
                }
            })
        }),
    );
    solver
}

fn final_state(solver: &ODEsolver) -> DVector<f64> {
    let (_, trajectory) = solver.get_result();
    trajectory.row(trajectory.nrows() - 1).transpose()
}

fn benchmark_telemetry_modes(c: &mut Criterion) {
    let modes = [
        ("off", BdfTelemetryMode::Off),
        ("counters", BdfTelemetryMode::Counters),
        ("timings", BdfTelemetryMode::Timings),
    ];
    let mut group = c.benchmark_group("bdf_telemetry_full_solve");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));

    for dimension in [1, 3, 16, 64] {
        let mut reference_solver = make_solver(dimension, BdfTelemetryMode::Off);
        reference_solver.solve();
        let reference = final_state(&reference_solver);

        for (label, mode) in modes {
            let mut check_solver = make_solver(dimension, mode);
            check_solver.solve();
            let state = final_state(&check_solver);
            let max_diff = state
                .iter()
                .zip(reference.iter())
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                max_diff <= 1e-12,
                "telemetry changed BDF trajectory: {mode:?}"
            );

            group.bench_with_input(
                BenchmarkId::new(label, dimension),
                &dimension,
                |bencher, &dimension| {
                    bencher.iter_batched(
                        || make_solver(dimension, mode),
                        |mut solver| {
                            solver.solve();
                            black_box(solver.get_result_ref());
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, benchmark_telemetry_modes);
criterion_main!(benches);
