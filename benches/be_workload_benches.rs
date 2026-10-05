//! Criterion coverage for Backward Euler warm solves and output-history costs.
//!
//! The history group compares the current streamed flat-sample layout with the
//! previous clone-per-state plus flatten layout. The solver group is separate
//! so output-assembly microbenchmarks are not mistaken for full-solve results.

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use nalgebra::{DMatrix, DVector};
use std::hint::black_box;
use std::time::Duration;

#[path = "support/be_bench_support.rs"]
mod be_bench_support;

use RustedSciThe::numerical::BE::BeSymbolicAssemblyBackend;
use RustedSciThe::numerical::BE::BeTelemetryMode;
use be_bench_support::{
    make_combustion_solver, make_diffusion_solver, make_parameterized_combustion_solver,
    make_parameterized_decay_solver,
};

fn legacy_clone_then_flatten(samples: &[f64], rows: usize, columns: usize) -> DMatrix<f64> {
    let vectors: Vec<DVector<f64>> = samples
        .chunks_exact(columns)
        .map(DVector::from_row_slice)
        .collect();
    let mut flattened = Vec::with_capacity(samples.len());
    for vector in &vectors {
        flattened.extend(vector.iter().copied());
    }
    DMatrix::from_vec(columns, rows, flattened).transpose()
}

fn streamed_row_slice(samples: &[f64], rows: usize, columns: usize) -> DMatrix<f64> {
    DMatrix::from_row_slice(rows, columns, samples)
}

fn streamed_owned_transpose(samples: Vec<f64>, rows: usize, columns: usize) -> DMatrix<f64> {
    DMatrix::from_vec(columns, rows, samples).transpose()
}

fn stitch_scalar_segments(
    prefix_times: &DVector<f64>,
    prefix_states: &DMatrix<f64>,
    segment_times: &DVector<f64>,
    segment_states: &DMatrix<f64>,
) -> (Vec<f64>, DMatrix<f64>) {
    let times = prefix_times
        .iter()
        .copied()
        .chain(segment_times.iter().skip(1).copied())
        .collect();
    let states: Vec<f64> = prefix_states
        .iter()
        .copied()
        .chain(segment_states.iter().skip(1).copied())
        .collect();
    let trajectory = DMatrix::from_column_slice(states.len(), 1, &states);
    (times, trajectory)
}

fn benchmark_history_assembly(c: &mut Criterion) {
    let mut group = c.benchmark_group("be_history_assembly");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));

    for (rows, columns) in [
        (32, 3),
        (128, 12),
        (64, 64),
        (128, 64),
        (256, 64),
        (512, 32),
    ] {
        let samples = vec![0.75; rows * columns];
        let legacy = legacy_clone_then_flatten(&samples, rows, columns);
        let streamed = streamed_row_slice(&samples, rows, columns);
        let owned_transpose = streamed_owned_transpose(samples.clone(), rows, columns);
        assert_eq!(legacy, streamed, "row-slice history parity failed");
        assert_eq!(
            legacy, owned_transpose,
            "owned-transpose history parity failed"
        );

        group.bench_with_input(
            BenchmarkId::new("legacy-clone-flatten", format!("{rows}x{columns}")),
            &samples,
            |bencher, values| {
                bencher.iter(|| {
                    black_box(legacy_clone_then_flatten(values, rows, columns));
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("streamed-row-slice", format!("{rows}x{columns}")),
            &samples,
            |bencher, values| {
                bencher.iter(|| {
                    black_box(streamed_row_slice(values, rows, columns));
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("owned-buffer-transpose", format!("{rows}x{columns}")),
            &samples,
            |bencher, values| {
                bencher.iter_batched(
                    || values.to_vec(),
                    |owned| black_box(streamed_owned_transpose(owned, rows, columns)),
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

fn benchmark_native_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("be_native_solve");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));

    for (dimension, steps) in [(8, 16), (32, 16), (64, 16), (8, 128)] {
        for analytic_jacobian in [true, false] {
            let route = if analytic_jacobian {
                "analytic-jacobian"
            } else {
                "fd-jacobian"
            };
            let final_time = steps as f64 * 0.001;
            let mut warm = make_diffusion_solver(
                dimension,
                analytic_jacobian,
                BeTelemetryMode::Off,
                0.001,
                final_time,
            );
            warm.try_solve().expect("warm-up solve should succeed");
            group.bench_with_input(
                BenchmarkId::new(
                    format!("warm-repeated/{route}"),
                    format!("n{dimension}-steps{steps}"),
                ),
                &dimension,
                |bencher, _| {
                    bencher.iter(|| {
                        warm.try_solve()
                            .expect("repeated native solve should succeed");
                        black_box(warm.trajectory().1);
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new(
                    format!("native-e2e/{route}"),
                    format!("n{dimension}-steps{steps}"),
                ),
                &dimension,
                |bencher, _| {
                    bencher.iter(|| {
                        let mut solver = make_diffusion_solver(
                            dimension,
                            analytic_jacobian,
                            BeTelemetryMode::Off,
                            0.001,
                            final_time,
                        );
                        solver
                            .try_solve()
                            .expect("native end-to-end solve should succeed");
                        black_box(solver.trajectory().1);
                    });
                },
            );
        }
    }

    let mut combustion = make_combustion_solver(BeTelemetryMode::Off);
    combustion
        .try_solve()
        .expect("combustion warm-up should succeed");
    group.bench_function(
        "warm-repeated/combustion-like/analytic-jacobian",
        |bencher| {
            bencher.iter(|| {
                combustion
                    .try_solve()
                    .expect("repeated combustion-like solve should succeed");
                black_box(combustion.trajectory().1);
            });
        },
    );
    group.bench_function("native-e2e/combustion-like/analytic-jacobian", |bencher| {
        bencher.iter(|| {
            let mut solver = make_combustion_solver(BeTelemetryMode::Off);
            solver
                .try_solve()
                .expect("combustion-like end-to-end solve should succeed");
            black_box(solver.trajectory().1);
        });
    });
    group.finish();
}

fn benchmark_parameter_rebind(c: &mut Criterion) {
    let mut group = c.benchmark_group("be_parameter_continuation");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));

    let mut rebound = make_parameterized_decay_solver(1.0, 1.0, 0.0, 0.5, BeTelemetryMode::Off);
    rebound
        .try_solve()
        .expect("initial symbolic solve should succeed");
    rebound
        .set_parameter_values(DVector::from_vec(vec![2.0]))
        .expect("rate rebind should succeed");
    rebound
        .try_solve()
        .expect("rebound symbolic solve should succeed");
    let rebound_result = rebound.get_result().1.unwrap();

    let mut fresh = make_parameterized_decay_solver(2.0, 1.0, 0.0, 0.5, BeTelemetryMode::Off);
    fresh
        .try_solve()
        .expect("fresh rate-2 solve should succeed");
    let fresh_result = fresh.get_result().1.unwrap();
    assert_eq!(rebound_result.shape(), fresh_result.shape());
    assert!(
        rebound_result
            .iter()
            .zip(fresh_result.iter())
            .all(|(left, right)| (left - right).abs() < 1e-12)
    );

    group.bench_function("warm-rebind/rate-1-to-2", |bencher| {
        bencher.iter(|| {
            rebound
                .set_parameter_values(DVector::from_vec(vec![1.0]))
                .expect("rate-1 rebind should succeed");
            rebound.try_solve().expect("rate-1 solve should succeed");
            rebound
                .set_parameter_values(DVector::from_vec(vec![2.0]))
                .expect("rate-2 rebind should succeed");
            rebound.try_solve().expect("rate-2 solve should succeed");
            black_box(rebound.trajectory().1);
        });
    });

    group.bench_function("fresh-instance/rate-2-setup-and-solve", |bencher| {
        bencher.iter(|| {
            let mut solver =
                make_parameterized_decay_solver(2.0, 1.0, 0.0, 0.5, BeTelemetryMode::Off);
            solver
                .try_solve()
                .expect("fresh symbolic solve should succeed");
            black_box(solver.trajectory().1);
        });
    });

    for target_count in [1, 4, 16] {
        let rates: Vec<f64> = (0..target_count)
            .map(|index| 0.5 + (index % 8) as f64 * 0.5)
            .collect();
        let mut prepared =
            make_parameterized_decay_solver(rates[0], 1.0, 0.0, 0.5, BeTelemetryMode::Off);
        prepared
            .try_solve()
            .expect("parameter series warm-up should succeed");
        for rate in &rates {
            prepared
                .set_parameter_values(DVector::from_vec(vec![*rate]))
                .expect("series parameter bind should succeed");
            prepared
                .try_solve()
                .expect("prepared parameter series solve should succeed");

            let mut fresh =
                make_parameterized_decay_solver(*rate, 1.0, 0.0, 0.5, BeTelemetryMode::Off);
            fresh
                .try_solve()
                .expect("fresh parameter series solve should succeed");
            assert_eq!(
                prepared.get_result().0.unwrap(),
                fresh.get_result().0.unwrap()
            );
            assert!(
                prepared
                    .get_result()
                    .1
                    .unwrap()
                    .iter()
                    .zip(fresh.get_result().1.unwrap().iter())
                    .all(|(left, right)| (left - right).abs() < 1e-12)
            );
        }

        group.bench_with_input(
            BenchmarkId::new("warm-rebind-series", format!("targets-{target_count}")),
            &rates,
            |bencher, rates| {
                bencher.iter(|| {
                    for rate in rates {
                        prepared
                            .set_parameter_values(DVector::from_vec(vec![*rate]))
                            .expect("series parameter bind should succeed");
                        prepared
                            .try_solve()
                            .expect("prepared series solve should succeed");
                    }
                    black_box(prepared.trajectory().1);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("fresh-instance-series", format!("targets-{target_count}")),
            &rates,
            |bencher, rates| {
                bencher.iter(|| {
                    for rate in rates {
                        let mut fresh = make_parameterized_decay_solver(
                            *rate,
                            1.0,
                            0.0,
                            0.5,
                            BeTelemetryMode::Off,
                        );
                        fresh
                            .try_solve()
                            .expect("fresh series solve should succeed");
                        black_box(fresh.trajectory().1);
                    }
                });
            },
        );
    }

    let mut continued = make_parameterized_decay_solver(1.0, 1.0, 0.0, 0.5, BeTelemetryMode::Off);
    continued
        .try_solve()
        .expect("first continuation segment should solve");
    continued
        .set_parameter_values(DVector::from_vec(vec![2.0]))
        .expect("continuation parameter update should succeed");
    continued
        .try_continue_to(1.0)
        .expect("accepted-state continuation should succeed");
    let continued_result = continued.get_result();

    let mut first_segment =
        make_parameterized_decay_solver(1.0, 1.0, 0.0, 0.5, BeTelemetryMode::Off);
    first_segment
        .try_solve()
        .expect("reference first segment should solve");
    let segment_times = first_segment.get_result().0.unwrap().clone();
    let segment_states = first_segment.get_result().1.unwrap().clone();
    let mut second_segment = make_parameterized_decay_solver(
        2.0,
        segment_states[(segment_states.nrows() - 1, 0)],
        0.5,
        1.0,
        BeTelemetryMode::Off,
    );
    second_segment
        .try_solve()
        .expect("reference second segment should solve");
    let expected_times = [
        segment_times.as_slice(),
        &second_segment.get_result().0.unwrap().as_slice()[1..],
    ]
    .concat();
    let expected_states = [
        segment_states.as_slice(),
        &second_segment.get_result().1.unwrap().as_slice()[1..],
    ]
    .concat();
    assert_eq!(continued_result.0.unwrap().as_slice(), expected_times);
    assert!(
        continued_result
            .1
            .unwrap()
            .iter()
            .zip(expected_states.iter())
            .all(|(left, right)| (left - right).abs() < 1e-12)
    );

    group.bench_function("warm-accepted-continuation/second-segment", |bencher| {
        bencher.iter_batched(
            || {
                let mut solver =
                    make_parameterized_decay_solver(1.0, 1.0, 0.0, 0.5, BeTelemetryMode::Off);
                solver
                    .try_solve()
                    .expect("first continuation segment should solve");
                solver
            },
            |mut solver| {
                solver
                    .set_parameter_values(DVector::from_vec(vec![2.0]))
                    .expect("continuation parameter update should succeed");
                solver
                    .try_continue_to(1.0)
                    .expect("accepted-state continuation should succeed");
                black_box(solver.trajectory());
            },
            BatchSize::SmallInput,
        );
    });
    group.bench_function("fresh-instance/second-segment-and-stitch", |bencher| {
        bencher.iter_batched(
            || {
                let mut first =
                    make_parameterized_decay_solver(1.0, 1.0, 0.0, 0.5, BeTelemetryMode::Off);
                first
                    .try_solve()
                    .expect("common first segment should solve");
                let (times, states) = first.get_result();
                (times.unwrap(), states.unwrap())
            },
            |(prefix_times, prefix_states)| {
                let midpoint = prefix_states[(prefix_states.nrows() - 1, 0)];
                let mut second =
                    make_parameterized_decay_solver(2.0, midpoint, 0.5, 1.0, BeTelemetryMode::Off);
                second
                    .try_solve()
                    .expect("fresh second segment should solve");
                let (segment_times, segment_states) = second.get_result();
                black_box(stitch_scalar_segments(
                    &prefix_times,
                    &prefix_states,
                    &segment_times.unwrap(),
                    &segment_states.unwrap(),
                ));
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn benchmark_combustion_parameter_rebind(c: &mut Criterion) {
    let mut group = c.benchmark_group("be_combustion_parameter_continuation");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));

    for target_count in [1, 4] {
        let rates: Vec<f64> = (0..target_count)
            .map(|index| 0.8 + index as f64 * 0.2)
            .collect();
        let mut reused = make_parameterized_combustion_solver(
            rates[0],
            BeTelemetryMode::Off,
            BeSymbolicAssemblyBackend::AtomViewNative,
        );
        reused
            .try_solve()
            .expect("combustion continuation warmup should succeed");

        for rate in &rates {
            reused
                .set_parameter_values(DVector::from_vec(vec![*rate]))
                .expect("combustion rate rebind should succeed");
            reused
                .try_solve()
                .expect("reused combustion solve should succeed");
            let reused_states = reused.trajectory().1;

            let mut fresh = make_parameterized_combustion_solver(
                *rate,
                BeTelemetryMode::Off,
                BeSymbolicAssemblyBackend::AtomViewNative,
            );
            fresh
                .try_solve()
                .expect("fresh combustion solve should succeed");
            let fresh_states = fresh.trajectory().1;
            assert_eq!(reused_states.shape(), fresh_states.shape());
            assert!(
                reused_states
                    .iter()
                    .zip(fresh_states.iter())
                    .all(|(left, right)| (left - right).abs() <= 1e-12)
            );
        }

        group.bench_with_input(
            BenchmarkId::new("warm-rebind-series", format!("targets-{target_count}")),
            &rates,
            |bencher, rates| {
                bencher.iter(|| {
                    for rate in rates {
                        reused
                            .set_parameter_values(DVector::from_vec(vec![*rate]))
                            .expect("combustion rate rebind should succeed");
                        reused
                            .try_solve()
                            .expect("reused combustion solve should succeed");
                    }
                    black_box(reused.trajectory());
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new(
                "fresh-prepare-solve-series",
                format!("targets-{target_count}"),
            ),
            &rates,
            |bencher, rates| {
                bencher.iter(|| {
                    for rate in rates {
                        let mut fresh = make_parameterized_combustion_solver(
                            *rate,
                            BeTelemetryMode::Off,
                            BeSymbolicAssemblyBackend::AtomViewNative,
                        );
                        fresh
                            .try_solve()
                            .expect("fresh combustion solve should succeed");
                        black_box(fresh.trajectory());
                    }
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_history_assembly,
    benchmark_native_solve,
    benchmark_parameter_rebind,
    benchmark_combustion_parameter_rebind
);
criterion_main!(benches);
