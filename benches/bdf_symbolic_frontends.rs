//! BDF symbolic frontend measurements with preparation and solve scopes split.
//!
//! Set `BDF_BENCH_WORKLOADS` to select workloads. Diffusion dimensions are
//! selected with `BDF_BENCH_DIFFUSION_DIMENSIONS`; default sizes remain bounded.

use RustedSciThe::numerical::BDF::BDF_api::{BdfSolverOptions, BdfTelemetryMode, ODEsolver};
use RustedSciThe::numerical::ivp_workloads::{
    SymbolicWorkload, WorkloadKind, build_workload, dense_coupled, parameter_continuation_target,
};
use RustedSciThe::symbolic::View::conversions::expr_to_atom;
use RustedSciThe::symbolic::ivp_telemetry::{IvpColdStage, IvpTelemetrySnapshot};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use RustedSciThe::symbolic::symbolic_ivp::IvpSymbolicAssemblyBackend;
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use nalgebra::DVector;
use std::{hint::black_box, time::Duration};

#[path = "support/bdf_bench_support.rs"]
mod bdf_bench_support;

fn assembly_routes() -> [(IvpSymbolicAssemblyBackend, &'static str); 2] {
    [
        (IvpSymbolicAssemblyBackend::ExprLegacy, "expr-legacy"),
        (IvpSymbolicAssemblyBackend::AtomView, "atom-native"),
    ]
}

fn dimensions_label(kind: WorkloadKind, dimension: usize) -> String {
    if matches!(kind, WorkloadKind::DiffusionChain) {
        dimension.to_string()
    } else {
        "fixed".to_string()
    }
}

fn report_stage_breakdown(
    workload: WorkloadKind,
    dimension: usize,
    assembly: IvpSymbolicAssemblyBackend,
    route: &str,
) {
    let mut solver = bdf_bench_support::make_solver_with_telemetry(
        workload,
        dimension,
        assembly,
        BdfTelemetryMode::Timings,
    );
    solver.try_generate().unwrap_or_else(|error| {
        panic!(
            "{} {route} diagnostic prepare failed: {error}",
            workload.label()
        )
    });
    solver.try_solve().unwrap_or_else(|error| {
        panic!(
            "{} {route} diagnostic solve failed: {error}",
            workload.label()
        )
    });
    let stats = solver.get_statistics();
    eprintln!(
        "[BDF stage diagnostic] workload={} dimension={} route={} scopes=nested/non-additive {}",
        workload.label(),
        dimensions_label(workload, dimension),
        route,
        stats.table_report(),
    );
}

fn benchmark_symbolic_frontends(c: &mut Criterion) {
    let workloads = bdf_bench_support::workloads_from_env(
        "BDF_BENCH_WORKLOADS",
        "stiff-scalar,robertson,combustion-like",
    );
    let mut group = c.benchmark_group("bdf_symbolic_frontends");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));

    for workload in workloads {
        for dimension in bdf_bench_support::dimensions(workload) {
            let mut route_states = Vec::new();
            for (assembly, route) in assembly_routes() {
                let mut solver = bdf_bench_support::make_solver(workload, dimension, assembly);
                solver.try_generate().unwrap_or_else(|error| {
                    panic!("{} {route} prepare failed: {error}", workload.label())
                });
                solver.solve();
                assert_eq!(
                    solver.get_status(),
                    "finished",
                    "{} {route}",
                    workload.label()
                );
                route_states.push((route, bdf_bench_support::final_state(&solver)));
            }
            let max_diff = route_states[0]
                .1
                .iter()
                .zip(route_states[1].1.iter())
                .map(|(left, right)| (left - right).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                max_diff <= 2e-5,
                "{} frontend parity preflight failed: diff={max_diff:e}",
                workload.label()
            );
            eprintln!(
                "[BDF benchmark preflight] workload={} dimension={} ExprLegacy-vs-AtomView final_diff={max_diff:.3e} status=ok",
                workload.label(),
                dimensions_label(workload, dimension)
            );

            for (assembly, route) in assembly_routes() {
                report_stage_breakdown(workload, dimension, assembly, route);
            }

            for (assembly, route) in assembly_routes() {
                let dimension_label = dimensions_label(workload, dimension);
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("prepare/{}/{route}", workload.label()),
                        &dimension_label,
                    ),
                    &dimension,
                    |bencher, &dimension| {
                        bencher.iter_batched(
                            || bdf_bench_support::make_solver(workload, dimension, assembly),
                            |mut solver| {
                                solver.try_generate().expect("BDF symbolic preparation");
                                black_box(solver.get_statistics());
                            },
                            BatchSize::SmallInput,
                        );
                    },
                );

                group.bench_with_input(
                    BenchmarkId::new(
                        format!("prepared-solve/{}/{route}", workload.label()),
                        &dimension_label,
                    ),
                    &dimension,
                    |bencher, &dimension| {
                        bencher.iter_batched(
                            || {
                                let mut solver =
                                    bdf_bench_support::make_solver(workload, dimension, assembly);
                                solver.try_generate().expect("BDF symbolic preparation");
                                solver
                            },
                            |mut solver| {
                                solver.solve();
                                black_box(bdf_bench_support::final_state(&solver));
                            },
                            BatchSize::SmallInput,
                        );
                    },
                );

                group.bench_with_input(
                    BenchmarkId::new(
                        format!("fresh-e2e/{}/{route}", workload.label()),
                        &dimension_label,
                    ),
                    &dimension,
                    |bencher, &dimension| {
                        bencher.iter_batched(
                            || bdf_bench_support::make_solver(workload, dimension, assembly),
                            |mut solver| {
                                solver.solve();
                                black_box(bdf_bench_support::final_state(&solver));
                            },
                            BatchSize::SmallInput,
                        );
                    },
                );
            }
        }
    }
    group.finish();
}

fn cold_stage_ms(snapshot: &IvpTelemetrySnapshot, stage: IvpColdStage) -> f64 {
    snapshot.cold_stage(stage).elapsed.as_secs_f64() * 1_000.0
}

fn benchmark_dense_symbolic_preparation(c: &mut Criterion) {
    let dimensions = bdf_bench_support::dense_dimensions();
    let mut group = c.benchmark_group("bdf_dense_symbolic_preparation_only");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(2));

    eprintln!(
        "[BDF frontend preparation-only] dimensions={dimensions:?}; fixture construction and solving excluded; diagnostic stages may overlap and are not additive"
    );
    eprintln!(
        "dimension | route | prepare_ms | expr_to_atom_ms | residual_evaluator_ms | atom_jacobian_ms | atom_dependency_ms | differentiation_ms | native_jac_evaluator_ms | symbolic_jacobian_ms | simplify_ms"
    );

    for dimension in dimensions {
        for (assembly, route) in assembly_routes() {
            let (prepare_ms, snapshot) =
                bdf_bench_support::prepare_symbolic_frontend(dimension, assembly);
            assert_eq!(snapshot.state_dimension, dimension);
            eprintln!(
                "{dimension} | {route} | {prepare_ms:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3}",
                cold_stage_ms(&snapshot, IvpColdStage::ExprToAtom),
                cold_stage_ms(&snapshot, IvpColdStage::ResidualLambdification),
                cold_stage_ms(&snapshot, IvpColdStage::AtomJacobianPreparation),
                cold_stage_ms(&snapshot, IvpColdStage::AtomDependencyAnalysis),
                cold_stage_ms(&snapshot, IvpColdStage::SymbolicDifferentiation),
                cold_stage_ms(&snapshot, IvpColdStage::NativeJacobianEvaluatorPreparation),
                cold_stage_ms(&snapshot, IvpColdStage::SymbolicJacobian),
                cold_stage_ms(&snapshot, IvpColdStage::Simplification),
            );

            group.bench_with_input(
                BenchmarkId::new(route, dimension),
                &dimension,
                |bencher, &dimension| {
                    bencher.iter_batched(
                        || bdf_bench_support::symbolic_frontend_inputs(dimension, assembly, false),
                        |(equations, variables, time_variable, options)| {
                            let result = bdf_bench_support::prepare_symbolic_inputs(
                                equations,
                                variables,
                                time_variable,
                                options,
                            );
                            black_box(result);
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }
    group.finish();
}

fn benchmark_dense_expr_to_atom_conversion(c: &mut Criterion) {
    let dimensions = [32, 64, 100];
    let mut group = c.benchmark_group("bdf_dense_expr_to_atom_conversion");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(2));

    for dimension in dimensions {
        let workload = dense_coupled(dimension);
        group.bench_with_input(
            BenchmarkId::from_parameter(dimension),
            &workload.equations,
            |bencher, equations| {
                bencher.iter(|| {
                    let atoms = equations.iter().map(expr_to_atom).collect::<Vec<_>>();
                    black_box(atoms)
                });
            },
        );
    }
    group.finish();
}

fn benchmark_dense_coupled_solver(c: &mut Criterion) {
    let dimensions = bdf_bench_support::dense_dimensions();
    let mut group = c.benchmark_group("bdf_dense_coupled_frontends");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));

    for dimension in dimensions {
        let mut route_states = Vec::new();
        for (assembly, route) in assembly_routes() {
            let mut solver = bdf_bench_support::make_dense_coupled_solver(
                dimension,
                assembly,
                BdfTelemetryMode::Off,
            );
            solver.try_solve().unwrap_or_else(|error| {
                panic!("dense-coupled n={dimension} {route} preflight failed: {error}")
            });
            assert_eq!(solver.get_status(), "finished");
            route_states.push((route, bdf_bench_support::final_state(&solver)));
        }
        let max_diff = route_states[0]
            .1
            .iter()
            .zip(route_states[1].1.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff <= 2e-5,
            "dense-coupled n={dimension} frontend parity drift: {max_diff:e}"
        );
        eprintln!(
            "[BDF dense-coupled benchmark preflight] dimension={dimension} ExprLegacy-vs-AtomView final_diff={max_diff:.3e} status=ok"
        );

        for (assembly, route) in assembly_routes() {
            group.bench_with_input(
                BenchmarkId::new(format!("prepare/{route}"), dimension),
                &dimension,
                |bencher, &dimension| {
                    bencher.iter_batched(
                        || {
                            bdf_bench_support::make_dense_coupled_solver(
                                dimension,
                                assembly,
                                BdfTelemetryMode::Off,
                            )
                        },
                        |mut solver| {
                            solver.try_generate().expect("dense coupled preparation");
                            black_box(solver.get_statistics());
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("prepared-solve/{route}"), dimension),
                &dimension,
                |bencher, &dimension| {
                    bencher.iter_batched(
                        || {
                            let mut solver = bdf_bench_support::make_dense_coupled_solver(
                                dimension,
                                assembly,
                                BdfTelemetryMode::Off,
                            );
                            solver.try_generate().expect("dense coupled preparation");
                            solver
                        },
                        |mut solver| {
                            solver.try_solve().expect("dense coupled solve");
                            black_box(bdf_bench_support::final_state(&solver));
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("fresh-e2e/{route}"), dimension),
                &dimension,
                |bencher, &dimension| {
                    bencher.iter_batched(
                        || {
                            bdf_bench_support::make_dense_coupled_solver(
                                dimension,
                                assembly,
                                BdfTelemetryMode::Off,
                            )
                        },
                        |mut solver| {
                            solver.try_solve().expect("dense coupled fresh solve");
                            black_box(bdf_bench_support::final_state(&solver));
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }
    group.finish();
}

fn parameterized_scalar_solver(
    assembly: IvpSymbolicAssemblyBackend,
    t0: f64,
    y0: f64,
    t_bound: f64,
    rate: f64,
) -> ODEsolver {
    let options = BdfSolverOptions::for_bdf(
        vec![Expr::parse_expression("-rate*y")],
        vec!["y".to_string()],
        "t".to_string(),
        t0,
        DVector::from_element(1, y0),
        t_bound,
        0.01,
        1e-8,
        1e-11,
        None,
        false,
        Some(0.0025),
    )
    .with_equation_parameters(vec!["rate".to_string()])
    .with_equation_parameter_values(DVector::from_element(1, rate))
    .with_symbolic_assembly_backend(assembly);
    ODEsolver::new_with_options(options)
}

fn seed_parameterized_scalar(assembly: IvpSymbolicAssemblyBackend) -> ODEsolver {
    let mut solver = parameterized_scalar_solver(assembly, 0.0, 1.0, 0.05, 2.5);
    solver
        .try_solve()
        .expect("BDF continuation benchmark warm-up segment");
    solver
}

fn parameterized_workload_solver(
    workload: &SymbolicWorkload,
    assembly: IvpSymbolicAssemblyBackend,
    t0: f64,
    y0: DVector<f64>,
    t_bound: f64,
    parameters: DVector<f64>,
) -> ODEsolver {
    let max_step = match workload.kind {
        WorkloadKind::CombustionLike => 1.0e-4,
        WorkloadKind::DiffusionChain => 2.5e-4,
        WorkloadKind::ThreeBody => 2.5e-4,
        _ => panic!(
            "{} is not a parameterized continuation workload",
            workload.kind.label()
        ),
    };
    let options = BdfSolverOptions::for_bdf(
        workload.equations.clone(),
        workload.variables.clone(),
        workload.time_variable.clone(),
        t0,
        y0,
        t_bound,
        max_step,
        1.0e-7,
        1.0e-10,
        None,
        false,
        Some(max_step),
    )
    .with_equation_parameters(workload.parameter_names.clone())
    .with_equation_parameter_values(parameters)
    .with_symbolic_assembly_backend(assembly);
    ODEsolver::new_with_options(options)
}

fn parameterized_workload_segment_width(kind: WorkloadKind) -> f64 {
    match kind {
        WorkloadKind::CombustionLike => 2.0e-4,
        WorkloadKind::DiffusionChain => 5.0e-4,
        WorkloadKind::ThreeBody => 5.0e-4,
        _ => panic!("{} has no continuation parameters", kind.label()),
    }
}

fn parameterized_workload_dimensions(kind: WorkloadKind) -> Vec<usize> {
    if kind != WorkloadKind::DiffusionChain {
        return vec![0];
    }
    std::env::var("BDF_BENCH_CONTINUATION_DIFFUSION_DIMENSIONS")
        .unwrap_or_else(|_| "8,16".to_string())
        .split(',')
        .map(|value| {
            value.trim().parse::<usize>().unwrap_or_else(|_| {
                panic!("BDF_BENCH_CONTINUATION_DIFFUSION_DIMENSIONS must contain positive integers")
            })
        })
        .inspect(|dimension| assert!(*dimension > 0, "diffusion dimensions must be positive"))
        .collect()
}

fn continuation_segment_counts() -> Vec<usize> {
    std::env::var("BDF_BENCH_CONTINUATION_COUNTS")
        .unwrap_or_else(|_| "1,4,16".to_string())
        .split(',')
        .map(|value| {
            value.trim().parse::<usize>().unwrap_or_else(|_| {
                panic!("BDF_BENCH_CONTINUATION_COUNTS must contain positive integers")
            })
        })
        .inspect(|count| assert!(*count > 0, "continuation segment counts must be positive"))
        .collect()
}

fn seed_parameterized_workload(
    workload: &SymbolicWorkload,
    dimension: usize,
    assembly: IvpSymbolicAssemblyBackend,
) -> ODEsolver {
    let width = parameterized_workload_segment_width(workload.kind);
    let mut solver = parameterized_workload_solver(
        workload,
        assembly,
        0.0,
        workload.initial_state.clone(),
        width,
        workload.parameter_values.clone(),
    );
    solver.try_solve().unwrap_or_else(|error| {
        panic!(
            "{} dimension={dimension} continuation seed failed: {error}",
            workload.kind.label()
        )
    });
    solver
}

fn benchmark_parameterized_workload_continuation(c: &mut Criterion) {
    let workloads = bdf_bench_support::workloads_from_env(
        "BDF_BENCH_CONTINUATION_WORKLOADS",
        "combustion-like,diffusion-chain",
    );
    let segment_counts = continuation_segment_counts();
    let mut group = c.benchmark_group("bdf_parameterized_workload_continuation");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));

    for kind in workloads {
        assert!(
            matches!(
                kind,
                WorkloadKind::CombustionLike
                    | WorkloadKind::DiffusionChain
                    | WorkloadKind::ThreeBody
            ),
            "continuation matrix workload {} must be parameterized",
            kind.label()
        );
        for dimension in parameterized_workload_dimensions(kind) {
            let workload = build_workload(kind, dimension.max(1));
            let width = parameterized_workload_segment_width(kind);
            for (assembly, route) in assembly_routes() {
                for segments in segment_counts.iter().copied() {
                    let mut warm = seed_parameterized_workload(&workload, dimension, assembly);
                    let mut fresh = seed_parameterized_workload(&workload, dimension, assembly);
                    for index in 1..=segments {
                        let (_, states) = fresh.get_result_ref();
                        let (times, _) = fresh.get_result_ref();
                        let t0 = *times
                            .as_slice()
                            .last()
                            .expect("fresh route has a segment time");
                        let y0 = states.row(states.nrows() - 1).transpose();
                        let t_bound = t0 + width;
                        let parameters =
                            parameter_continuation_target(&workload.parameter_values, index);

                        warm.try_continue_with_parameter_values(parameters.clone(), t_bound)
                            .expect("warm workload continuation restart");
                        warm.try_solve().expect("warm workload continuation solve");

                        fresh = parameterized_workload_solver(
                            &workload, assembly, t0, y0, t_bound, parameters,
                        );
                        fresh
                            .try_solve()
                            .expect("fresh workload continuation segment");
                    }
                    let parity = bdf_bench_support::final_state(&warm)
                        .iter()
                        .zip(bdf_bench_support::final_state(&fresh).iter())
                        .map(|(left, right)| (left - right).abs())
                        .fold(0.0_f64, f64::max);
                    assert!(
                        parity <= 2.0e-8,
                        "{} {route} continuation parity drift={parity:e} for {segments} segments",
                        kind.label()
                    );
                    eprintln!(
                        "[BDF continuation benchmark preflight] workload={} dimension={} route={} segments={} final_diff={parity:.3e} status=ok",
                        kind.label(),
                        dimensions_label(kind, dimension),
                        route,
                        segments
                    );

                    let id = format!(
                        "{}/{}/{route}",
                        kind.label(),
                        dimensions_label(kind, dimension)
                    );
                    group.bench_function(
                        BenchmarkId::new(format!("warm-reuse/{id}"), segments),
                        |bencher| {
                            bencher.iter_batched(
                                || seed_parameterized_workload(&workload, dimension, assembly),
                                |mut solver| {
                                    for index in 1..=segments {
                                        let t_bound = width * (index + 1) as f64;
                                        let parameters = parameter_continuation_target(
                                            &workload.parameter_values,
                                            index,
                                        );
                                        solver
                                            .try_continue_with_parameter_values(parameters, t_bound)
                                            .expect("warm workload continuation restart");
                                        solver
                                            .try_solve()
                                            .expect("warm workload continuation solve");
                                    }
                                    black_box(bdf_bench_support::final_state(&solver));
                                },
                                BatchSize::SmallInput,
                            );
                        },
                    );
                    group.bench_function(
                        BenchmarkId::new(format!("fresh-reprepare/{id}"), segments),
                        |bencher| {
                            bencher.iter_batched(
                                || seed_parameterized_workload(&workload, dimension, assembly),
                                |mut solver| {
                                    for index in 1..=segments {
                                        let (times, states) = solver.get_result_ref();
                                        let t0 = *times
                                            .as_slice()
                                            .last()
                                            .expect("fresh route segment time");
                                        let y0 = states.row(states.nrows() - 1).transpose();
                                        let t_bound = t0 + width;
                                        solver = parameterized_workload_solver(
                                            &workload,
                                            assembly,
                                            t0,
                                            y0,
                                            t_bound,
                                            parameter_continuation_target(
                                                &workload.parameter_values,
                                                index,
                                            ),
                                        );
                                        solver.try_solve().expect("fresh workload segment solve");
                                    }
                                    black_box(bdf_bench_support::final_state(&solver));
                                },
                                BatchSize::SmallInput,
                            );
                        },
                    );
                }
            }
        }
    }
    group.finish();
}

fn benchmark_parameter_continuation(c: &mut Criterion) {
    let mut group = c.benchmark_group("bdf_parameter_continuation");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));

    for (assembly, route) in assembly_routes() {
        for segments in [1usize, 4, 16] {
            let mut warm = seed_parameterized_scalar(assembly);
            let mut fresh = seed_parameterized_scalar(assembly);
            for index in 0..segments {
                let target = 2.5 + (index + 1) as f64 * 0.25;
                let next_bound = 0.05 * (index + 2) as f64;
                warm.try_continue_with_parameter_values(
                    DVector::from_element(1, target),
                    next_bound,
                )
                .expect("warm continuation preflight restart");
                warm.try_solve().expect("warm continuation preflight solve");

                let (_, y) = fresh.get_result_ref();
                let mut fresh_segment = parameterized_scalar_solver(
                    assembly,
                    next_bound - 0.05,
                    y[(y.nrows() - 1, 0)],
                    next_bound,
                    target,
                );
                fresh_segment
                    .try_solve()
                    .expect("fresh continuation preflight solve");
                fresh = fresh_segment;
            }
            let warm_final = bdf_bench_support::final_state(&warm)[0];
            let fresh_final = bdf_bench_support::final_state(&fresh)[0];
            assert!(
                (warm_final - fresh_final).abs() <= 2e-10,
                "{route} continuation parity drift={:e} for {segments} segments",
                (warm_final - fresh_final).abs()
            );

            group.bench_function(
                BenchmarkId::new(format!("warm-reuse/{route}"), segments),
                |bencher| {
                    bencher.iter_batched(
                        || seed_parameterized_scalar(assembly),
                        |mut solver| {
                            for index in 0..segments {
                                solver
                                    .try_continue_with_parameter_values(
                                        DVector::from_element(1, 2.5 + (index + 1) as f64 * 0.25),
                                        0.05 * (index + 2) as f64,
                                    )
                                    .expect("warm continuation restart");
                                solver.try_solve().expect("warm continuation solve");
                            }
                            black_box(bdf_bench_support::final_state(&solver));
                        },
                        BatchSize::SmallInput,
                    );
                },
            );

            group.bench_function(
                BenchmarkId::new(format!("fresh-reprepare/{route}"), segments),
                |bencher| {
                    bencher.iter_batched(
                        || seed_parameterized_scalar(assembly),
                        |mut solver| {
                            for index in 0..segments {
                                let (times, states) = solver.get_result_ref();
                                let t0 = times[times.len() - 1];
                                let y0 = states[(states.nrows() - 1, 0)];
                                let t_bound = t0 + 0.05;
                                let mut next = parameterized_scalar_solver(
                                    assembly,
                                    t0,
                                    y0,
                                    t_bound,
                                    2.5 + (index + 1) as f64 * 0.25,
                                );
                                next.try_solve().expect("fresh continuation solve");
                                solver = next;
                            }
                            black_box(bdf_bench_support::final_state(&solver));
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_symbolic_frontends,
    benchmark_dense_symbolic_preparation,
    benchmark_dense_expr_to_atom_conversion,
    benchmark_dense_coupled_solver,
    benchmark_parameter_continuation,
    benchmark_parameterized_workload_continuation
);
criterion_main!(benches);
