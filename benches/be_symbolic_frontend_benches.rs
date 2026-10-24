use RustedSciThe::numerical::BDF::BDF_solver::BdfJacobian;
use RustedSciThe::numerical::LSODE2::native_jacobian::{
    NativeJacobianStorage, compile_native_symbolic_jacobian,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use RustedSciThe::symbolic::symbolic_ivp::{
    IvpSymbolicAssemblyBackend, SymbolicIvpProblemOptions, prepare_symbolic_ivp_problem,
};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use faer::sparse::Triplet;
use nalgebra::DVector;
use std::hint::black_box;
use std::time::Duration;

fn dense_coupled_fixture(dimension: usize) -> (Vec<Expr>, Vec<String>) {
    let variables: Vec<String> = (0..dimension).map(|i| format!("y{i}")).collect();
    let total_state = variables.join("+");
    let equations = variables
        .iter()
        .enumerate()
        .map(|(i, variable)| {
            let next = &variables[(i + 1) % dimension];
            Expr::parse_expression(&format!(
                "-0.1*{variable}+0.00001*({total_state})*({total_state})+0.02*{variable}*{next}"
            ))
        })
        .collect();
    (equations, variables)
}

fn sparse_chain_fixture(dimension: usize) -> (Vec<Expr>, Vec<String>) {
    let variables: Vec<String> = (0..dimension).map(|i| format!("y{i}")).collect();
    let equations = variables
        .iter()
        .enumerate()
        .map(|(i, variable)| {
            let previous = &variables[(i + dimension - 1) % dimension];
            let next = &variables[(i + 1) % dimension];
            Expr::parse_expression(&format!("-0.1*{variable}+0.02*{previous}+0.03*{next}"))
        })
        .collect();
    (equations, variables)
}

fn banded_chain_fixture(dimension: usize) -> (Vec<Expr>, Vec<String>) {
    let variables: Vec<String> = (0..dimension).map(|i| format!("y{i}")).collect();
    let equations = variables
        .iter()
        .enumerate()
        .map(|(row, variable)| {
            let mut terms = vec![format!("-0.1*{variable}")];
            if row > 0 {
                terms.push(format!("0.02*{}", variables[row - 1]));
            }
            if row + 1 < dimension {
                terms.push(format!("0.03*{}", variables[row + 1]));
            }
            Expr::parse_expression(&terms.join("+"))
        })
        .collect();
    (equations, variables)
}

fn full_jacobian_reference(equations: &[Expr], variables: &[String]) -> Vec<Vec<Expr>> {
    equations
        .iter()
        .map(|equation| {
            variables
                .iter()
                .map(|variable| {
                    let derivative = equation.diff(variable);
                    match derivative {
                        Expr::Const(_) => derivative,
                        _ => derivative.simplify(),
                    }
                })
                .collect()
        })
        .collect()
}

fn dependency_aware_jacobian(equations: &[Expr], variables: &[String]) -> Vec<Vec<Expr>> {
    equations
        .iter()
        .map(|equation| {
            let dependencies = equation.all_arguments_are_variables();
            variables
                .iter()
                .map(|variable| {
                    if dependencies
                        .binary_search_by(|name| name.as_str().cmp(variable.as_str()))
                        .is_ok()
                    {
                        let derivative = equation.diff(variable);
                        match derivative {
                            Expr::Const(_) => derivative,
                            _ => derivative.simplify(),
                        }
                    } else {
                        Expr::Const(0.0)
                    }
                })
                .collect()
        })
        .collect()
}

fn sparse_jacobian_plan(equations: &[Expr], variables: &[String]) -> Vec<(usize, usize, Expr)> {
    let mut entries = Vec::new();
    for (row, equation) in equations.iter().enumerate() {
        let dependencies = equation.all_arguments_are_variables();
        for (col, variable) in variables.iter().enumerate() {
            if dependencies
                .binary_search_by(|name| name.as_str().cmp(variable.as_str()))
                .is_err()
            {
                continue;
            }
            let derivative = equation.diff(variable);
            let derivative = match derivative {
                Expr::Const(_) => derivative,
                _ => derivative.simplify(),
            };
            if !derivative.is_zero() {
                entries.push((row, col, derivative));
            }
        }
    }
    entries
}

type ScalarEvaluator = Box<dyn Fn(&[f64]) -> f64 + Send + Sync>;

fn full_dense_callback_reference(
    equations: &[Expr],
    variables: &[String],
) -> Vec<Vec<ScalarEvaluator>> {
    let jacobian = full_jacobian_reference(equations, variables);
    let names = std::iter::once("t".to_string())
        .chain(variables.iter().cloned())
        .collect::<Vec<_>>();
    let name_refs = names.iter().map(String::as_str).collect::<Vec<_>>();
    jacobian
        .iter()
        .map(|row| {
            row.iter()
                .map(|expression| Expr::lambdify_borrowed_thread_safe(expression, &name_refs))
                .collect()
        })
        .collect()
}

fn evaluate_full_dense_reference(
    evaluators: &[Vec<ScalarEvaluator>],
    time: f64,
    state: &DVector<f64>,
) -> nalgebra::DMatrix<f64> {
    let mut args = Vec::with_capacity(state.len() + 1);
    args.push(time);
    args.extend(state.iter().copied());
    let values = evaluators
        .iter()
        .flat_map(|row| row.iter().map(|evaluate| evaluate(&args)))
        .collect::<Vec<_>>();
    nalgebra::DMatrix::from_row_slice(evaluators.len(), evaluators[0].len(), &values)
}

fn dense_scan_sparse_callback_reference(
    equations: &[Expr],
    variables: &[String],
) -> Vec<(usize, usize, ScalarEvaluator)> {
    let jacobian = full_jacobian_reference(equations, variables);
    let names = std::iter::once("t".to_string())
        .chain(variables.iter().cloned())
        .collect::<Vec<_>>();
    let name_refs = names.iter().map(String::as_str).collect::<Vec<_>>();
    jacobian
        .iter()
        .enumerate()
        .flat_map(|(row, expressions)| {
            expressions
                .iter()
                .enumerate()
                .filter(|(_, expression)| !expression.is_zero())
                .map(|(col, expression)| {
                    (
                        row,
                        col,
                        Expr::lambdify_borrowed_thread_safe(expression, &name_refs),
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn evaluate_sparse_triplets_reference(
    evaluators: &[(usize, usize, ScalarEvaluator)],
    time: f64,
    state: &DVector<f64>,
) -> BdfJacobian {
    let mut args = Vec::with_capacity(state.len() + 1);
    args.push(time);
    args.extend(state.iter().copied());
    let triplets = evaluators
        .iter()
        .map(|(row, col, evaluate)| Triplet::new(*row, *col, evaluate(&args)))
        .collect();
    BdfJacobian::SparseTriplets {
        n: state.len(),
        triplets,
    }
}

fn prepare(
    equations: Vec<Expr>,
    variables: Vec<String>,
    assembly: IvpSymbolicAssemblyBackend,
) -> RustedSciThe::symbolic::symbolic_ivp::PreparedSymbolicIvpProblem {
    prepare_symbolic_ivp_problem(
        equations,
        variables,
        "t".to_string(),
        SymbolicIvpProblemOptions::new().with_symbolic_assembly_backend(assembly),
    )
    .expect("symbolic frontend preparation should succeed")
}

fn benchmark_symbolic_frontends(c: &mut Criterion) {
    let mut group = c.benchmark_group("be_symbolic_frontend");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));

    for dimension in [3, 8, 16, 32] {
        let (equations, variables) = dense_coupled_fixture(dimension);
        let state = DVector::from_fn(dimension, |i, _| 0.5 + i as f64 / dimension as f64);
        let legacy = prepare(
            equations.clone(),
            variables.clone(),
            IvpSymbolicAssemblyBackend::ExprLegacy,
        );
        let atom = prepare(
            equations.clone(),
            variables.clone(),
            IvpSymbolicAssemblyBackend::AtomView,
        );

        let legacy_residual = (legacy.residual)(0.25, &state);
        let atom_residual = (atom.residual)(0.25, &state);
        let legacy_jacobian = (legacy.jacobian)(0.25, &state);
        let atom_jacobian = (atom.jacobian)(0.25, &state);
        assert!(
            legacy_residual
                .iter()
                .zip(atom_residual.iter())
                .all(|(a, b)| (a - b).abs() <= 1e-10)
        );
        assert!(
            legacy_jacobian
                .iter()
                .zip(atom_jacobian.iter())
                .all(|(a, b)| (a - b).abs() <= 1e-10)
        );

        for (frontend, assembly) in [
            ("expr-legacy", IvpSymbolicAssemblyBackend::ExprLegacy),
            ("atom-native", IvpSymbolicAssemblyBackend::AtomView),
        ] {
            group.bench_function(
                BenchmarkId::new("prepare", format!("{frontend}/{dimension}")),
                |bencher| {
                    bencher.iter_batched(
                        || (equations.clone(), variables.clone()),
                        |(equations, variables)| {
                            black_box(prepare(equations, variables, assembly));
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }

        group.bench_function(
            BenchmarkId::new("residual", format!("expr-legacy/{dimension}")),
            |bencher| {
                bencher.iter(|| black_box((legacy.residual)(0.25, black_box(&state))));
            },
        );
        group.bench_function(
            BenchmarkId::new("residual", format!("atom-native/{dimension}")),
            |bencher| {
                bencher.iter(|| black_box((atom.residual)(0.25, black_box(&state))));
            },
        );
        group.bench_function(
            BenchmarkId::new("jacobian", format!("expr-legacy/{dimension}")),
            |bencher| {
                bencher.iter(|| black_box((legacy.jacobian)(0.25, black_box(&state))));
            },
        );
        group.bench_function(
            BenchmarkId::new("jacobian", format!("atom-native/{dimension}")),
            |bencher| {
                bencher.iter(|| black_box((atom.jacobian)(0.25, black_box(&state))));
            },
        );
    }

    // This isolates the cold ExprLegacy preparation path changed by the
    // dependency-aware Jacobian builder; the local stencil stays O(1) wide.
    for dimension in [32, 64, 128, 256] {
        let (equations, variables) = sparse_chain_fixture(dimension);
        let reference = full_jacobian_reference(&equations, &variables);
        let dependency_aware = dependency_aware_jacobian(&equations, &variables);
        assert_eq!(reference, dependency_aware, "Jacobian construction parity");

        for (route, build) in [
            (
                "full-reference",
                full_jacobian_reference as fn(&[Expr], &[String]) -> Vec<Vec<Expr>>,
            ),
            ("dependency-aware", dependency_aware_jacobian),
        ] {
            group.bench_function(
                BenchmarkId::new(
                    "jacobian-build-sparse-chain",
                    format!("{route}/{dimension}"),
                ),
                |bencher| bencher.iter(|| black_box(build(&equations, &variables))),
            );
        }

        group.bench_function(
            BenchmarkId::new("prepare-sparse-chain/expr-legacy", dimension),
            |bencher| {
                bencher.iter_batched(
                    || (equations.clone(), variables.clone()),
                    |(equations, variables)| {
                        black_box(prepare(
                            equations,
                            variables,
                            IvpSymbolicAssemblyBackend::ExprLegacy,
                        ));
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();

    // A/B gate for structural-zero elimination: the control differentiates,
    // simplifies and lambdifies the full dense matrix; the candidate prepares
    // the production ExprLegacy callback which retains only structural entries.
    let mut sparse_ab = c.benchmark_group("be_symbolic_ivp_sparse_jacobian_ab");
    sparse_ab.sample_size(50);
    sparse_ab.measurement_time(Duration::from_secs(5));
    for dimension in [128, 256, 512] {
        let (equations, variables) = sparse_chain_fixture(dimension);
        let dense_reference = full_jacobian_reference(&equations, &variables);
        let sparse_plan = sparse_jacobian_plan(&equations, &variables);
        assert_eq!(
            dense_reference
                .iter()
                .flatten()
                .filter(|expression| !expression.is_zero())
                .count(),
            sparse_plan.len(),
            "dense and sparse symbolic patterns must match"
        );

        sparse_ab.bench_function(
            BenchmarkId::new(
                "jacobian-plan-build-sparse-chain",
                format!("full-dense-reference/{dimension}"),
            ),
            |bencher| {
                bencher.iter(|| black_box(full_jacobian_reference(&equations, &variables)));
            },
        );
        sparse_ab.bench_function(
            BenchmarkId::new(
                "jacobian-plan-build-sparse-chain",
                format!("structural-sparse-plan/{dimension}"),
            ),
            |bencher| {
                bencher.iter(|| black_box(sparse_jacobian_plan(&equations, &variables)));
            },
        );

        let candidate = prepare(
            equations.clone(),
            variables.clone(),
            IvpSymbolicAssemblyBackend::ExprLegacy,
        );
        let reference_callback = full_dense_callback_reference(&equations, &variables);
        let state = DVector::from_fn(dimension, |index, _| 0.5 + index as f64 / dimension as f64);
        let expected = evaluate_full_dense_reference(&reference_callback, 0.25, &state);
        let actual = (candidate.jacobian)(0.25, &state);
        assert_eq!(expected.shape(), actual.shape());
        assert!(
            expected
                .iter()
                .zip(actual.iter())
                .all(|(left, right)| (left - right).abs() <= 1.0e-12)
        );

        sparse_ab.bench_function(
            BenchmarkId::new(
                "jacobian-callback-sparse-chain",
                format!("full-dense-reference/{dimension}"),
            ),
            |bencher| {
                bencher.iter(|| {
                    black_box(evaluate_full_dense_reference(
                        &reference_callback,
                        0.25,
                        black_box(&state),
                    ));
                });
            },
        );
        sparse_ab.bench_function(
            BenchmarkId::new(
                "jacobian-callback-sparse-chain",
                format!("expr-legacy-zero-skip/{dimension}"),
            ),
            |bencher| {
                bencher.iter(|| black_box((candidate.jacobian)(0.25, black_box(&state))));
            },
        );

        let mut native_sparse = compile_native_symbolic_jacobian(
            &equations,
            &variables,
            "t",
            NativeJacobianStorage::SparseTriplets,
        );
        let native_reference = dense_scan_sparse_callback_reference(&equations, &variables);
        let actual = native_sparse(0.25, &state);
        let expected = evaluate_sparse_triplets_reference(&native_reference, 0.25, &state);
        let (
            BdfJacobian::SparseTriplets {
                triplets: actual, ..
            },
            BdfJacobian::SparseTriplets {
                triplets: expected, ..
            },
        ) = (actual, expected)
        else {
            panic!("sparse LSODE2 A/B routes must both return triplets");
        };
        assert_eq!(actual.len(), expected.len());
        assert!(
            actual
                .iter()
                .zip(expected.iter())
                .all(|(left, right)| left.row == right.row
                    && left.col == right.col
                    && (left.val - right.val).abs() <= 1.0e-12)
        );

        sparse_ab.bench_function(
            BenchmarkId::new(
                "lsode2-sparse-jacobian-prepare",
                format!("full-dense-scan-reference/{dimension}"),
            ),
            |bencher| {
                bencher.iter_batched(
                    || (equations.clone(), variables.clone()),
                    |(equations, variables)| {
                        black_box(dense_scan_sparse_callback_reference(&equations, &variables));
                    },
                    BatchSize::SmallInput,
                );
            },
        );
        sparse_ab.bench_function(
            BenchmarkId::new(
                "lsode2-sparse-jacobian-prepare",
                format!("expr-legacy-direct-sparse/{dimension}"),
            ),
            |bencher| {
                bencher.iter_batched(
                    || (equations.clone(), variables.clone()),
                    |(equations, variables)| {
                        let _ = black_box(compile_native_symbolic_jacobian(
                            &equations,
                            &variables,
                            "t",
                            NativeJacobianStorage::SparseTriplets,
                        ));
                    },
                    BatchSize::SmallInput,
                );
            },
        );
        sparse_ab.bench_function(
            BenchmarkId::new(
                "lsode2-sparse-jacobian-callback",
                format!("full-dense-scan-reference/{dimension}"),
            ),
            |bencher| {
                bencher.iter(|| {
                    black_box(evaluate_sparse_triplets_reference(
                        &native_reference,
                        0.25,
                        black_box(&state),
                    ));
                });
            },
        );
        sparse_ab.bench_function(
            BenchmarkId::new(
                "lsode2-sparse-jacobian-callback",
                format!("expr-legacy-direct-sparse/{dimension}"),
            ),
            |bencher| {
                bencher.iter(|| black_box(native_sparse(0.25, black_box(&state))));
            },
        );
    }

    // Banded A/B: the control discovers dependencies across the full variable
    // list; the candidate uses the declared (1, 1) layout to restrict symbolic
    // differentiation to local columns. Both constructors return solver-ready
    // native callbacks, so this measures cold preparation rather than a helper.
    for dimension in [256, 1024, 2048] {
        let (equations, variables) = banded_chain_fixture(dimension);
        let serial_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("serial benchmark pool should build");
        let four_worker_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("four-worker benchmark pool should build");
        let state = DVector::from_fn(dimension, |index, _| 0.5 + index as f64 / dimension as f64);
        let mut sparse = compile_native_symbolic_jacobian(
            &equations,
            &variables,
            "t",
            NativeJacobianStorage::SparseTriplets,
        );
        let mut banded = compile_native_symbolic_jacobian(
            &equations,
            &variables,
            "t",
            NativeJacobianStorage::Banded {
                bandwidth: Some((1, 1)),
            },
        );
        let sparse_values = sparse(0.25, &state);
        let banded_values = banded(0.25, &state);
        let (BdfJacobian::SparseTriplets { n, triplets }, BdfJacobian::Banded(matrix)) =
            (sparse_values, banded_values)
        else {
            panic!("Banded A/B routes must produce their requested layouts");
        };
        assert_eq!(n, dimension);
        for triplet in triplets {
            assert_eq!(matrix[(triplet.row, triplet.col)], triplet.val);
        }

        for (route, storage, pool) in [
            (
                "sparse-dependency-global",
                NativeJacobianStorage::SparseTriplets,
                None,
            ),
            (
                "known-band-global-auto",
                NativeJacobianStorage::Banded {
                    bandwidth: Some((1, 1)),
                },
                None,
            ),
            (
                "known-band-one-worker",
                NativeJacobianStorage::Banded {
                    bandwidth: Some((1, 1)),
                },
                Some(&serial_pool),
            ),
            (
                "known-band-four-workers",
                NativeJacobianStorage::Banded {
                    bandwidth: Some((1, 1)),
                },
                Some(&four_worker_pool),
            ),
        ] {
            sparse_ab.bench_function(
                BenchmarkId::new(
                    "lsode2-banded-jacobian-prepare",
                    format!("{route}/{dimension}"),
                ),
                |bencher| {
                    bencher.iter_batched(
                        || (equations.clone(), variables.clone()),
                        |(equations, variables)| {
                            let prepare = || {
                                let _ = black_box(compile_native_symbolic_jacobian(
                                    &equations, &variables, "t", storage,
                                ));
                            };
                            if let Some(pool) = pool {
                                pool.install(prepare);
                            } else {
                                prepare();
                            }
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }
    sparse_ab.finish();

    // Isolate the public row-major Expr callback's reusable scalar scratch.
    // Both routes write the same dense ABI output; only the candidate keeps
    // the structurally nonzero value buffer between calls.
    let mut workspace_ab = c.benchmark_group("be_symbolic_ivp_workspace_reuse");
    workspace_ab.sample_size(30);
    workspace_ab.measurement_time(Duration::from_secs(5));
    for dimension in [128, 256, 512] {
        let (equations, variables) = sparse_chain_fixture(dimension);
        let problem = prepare(equations, variables, IvpSymbolicAssemblyBackend::ExprLegacy);
        let state = DVector::from_fn(dimension, |index, _| 0.5 + index as f64 / dimension as f64);
        let mut output = vec![0.0; dimension * dimension];
        let mut args = Vec::new();
        let mut values = Vec::new();
        problem
            .try_evaluate_jacobian_row_major_into_with_workspace(
                0.25,
                &state,
                &mut output,
                &mut args,
            )
            .expect("compatibility row-major callback should evaluate");
        problem
            .try_evaluate_jacobian_row_major_into_with_workspaces(
                0.25,
                &state,
                &mut output,
                &mut values,
                &mut args,
            )
            .expect("caller-owned row-major callback should evaluate");
        let warmed_values_capacity = values.capacity();
        let values_capacity_bytes = warmed_values_capacity * std::mem::size_of::<f64>();
        let dense_reference_bytes = dimension * dimension * std::mem::size_of::<f64>();
        eprintln!(
            "[IVP jacobian scratch] n={dimension} sparse_value_capacity={} ({values_capacity_bytes} B); prior_dense_scratch={} B",
            warmed_values_capacity, dense_reference_bytes
        );

        workspace_ab.bench_function(
            BenchmarkId::new("row-major-compatibility-temporary", dimension),
            |bencher| {
                bencher.iter(|| {
                    problem
                        .try_evaluate_jacobian_row_major_into_with_workspace(
                            0.25,
                            black_box(&state),
                            black_box(&mut output),
                            black_box(&mut args),
                        )
                        .expect("compatibility callback should evaluate");
                });
            },
        );
        workspace_ab.bench_function(
            BenchmarkId::new("row-major-caller-owned-scratch", dimension),
            |bencher| {
                bencher.iter(|| {
                    problem
                        .try_evaluate_jacobian_row_major_into_with_workspaces(
                            0.25,
                            black_box(&state),
                            black_box(&mut output),
                            black_box(&mut values),
                            black_box(&mut args),
                        )
                        .expect("caller-owned callback should evaluate");
                });
            },
        );
    }
    workspace_ab.finish();
}

criterion_group!(benches, benchmark_symbolic_frontends);
criterion_main!(benches);
