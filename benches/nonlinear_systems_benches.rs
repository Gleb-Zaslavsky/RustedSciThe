//! Focused benchmarks for the generic nonlinear-system path.
//!
//! These benchmarks intentionally keep preparation outside `iter` and measure
//! only one concern at a time:
//! - symbolic residual/Jacobian dispatch,
//! - dense factor-and-solve,
//! - and the complete generic Newton loop.
//!
//! Run with:
//!
//! ```text
//! cargo bench --bench nonlinear_systems_benches
//! cargo bench --bench nonlinear_systems_benches -- nonlinear_lambdify_realistic_end_to_end --noplot --warm-up-time 1 --measurement-time 1 --sample-size 10
//! cargo bench --bench nonlinear_systems_benches -- nonlinear_lambdify_parameterized_dispatch --noplot --warm-up-time 1 --measurement-time 1 --sample-size 10
//! cargo bench --bench nonlinear_systems_benches -- nonlinear_lambdify_large_legacy_vs_prepared --noplot --warm-up-time 1 --measurement-time 1 --sample-size 10
//! cargo bench --bench nonlinear_systems_benches -- nonlinear_lambdify_large_corpus_callbacks --noplot --warm-up-time 1 --measurement-time 1 --sample-size 10
//! cargo bench --bench nonlinear_systems_benches -- nonlinear_lambdify_large_corpus_solves --noplot --warm-up-time 1 --measurement-time 1 --sample-size 10
//! cargo bench --bench nonlinear_systems_benches -- nonlinear_lambdify_parallel_threshold_sweep --noplot --warm-up-time 1 --measurement-time 1 --sample-size 10
//! ```

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nalgebra::{DMatrix, DVector};
use std::hint::black_box;

use RustedSciThe::numerical::Nonlinear_systems::engine::{
    DiagnosticsOptions, IterationState, LinearSolverKind, NewtonMethod, SolveOptions, SolverEngine,
    solve_linear_system,
};
use RustedSciThe::numerical::Nonlinear_systems::prelude::{
    DampedNewtonMethod, DampedNewtonMethodAdvanced, LevenbergMarquardtMethod,
    LevenbergMarquardtMinpack, NielsenLevenbergMarquardtMethod,
    NielsenLevenbergMarquardtMethodAdvanced, NonlinearSolverMethod, PowellDoglegMethod,
    TrustRegionLMMethod, TrustRegionMethod,
};
use RustedSciThe::numerical::Nonlinear_systems::problem::{JacobianProvider, NonlinearProblem};
use RustedSciThe::numerical::Nonlinear_systems::symbolic::{
    LambdifyExecutionPolicy, PreparedSymbolicNonlinearProblem, SymbolicNonlinearProblem,
    SymbolicProblemOptions,
};
use RustedSciThe::symbolic::symbolic_functions::Jacobian;

fn symbolic_problem(dimension: usize) -> SymbolicNonlinearProblem {
    let variables = (0..dimension)
        .map(|index| format!("x{index}"))
        .collect::<Vec<_>>();
    let equations = variables
        .iter()
        .enumerate()
        .map(|(index, variable)| format!("{variable}-{:.1}", index as f64 + 1.0))
        .collect::<Vec<_>>();

    SymbolicNonlinearProblem::from_strings_with_options(
        equations,
        SymbolicProblemOptions::new().with_variables(variables),
    )
    .expect("benchmark symbolic problem should build")
}

fn legacy_lambdify_problem(dimension: usize) -> Jacobian {
    let variables = (0..dimension)
        .map(|index| format!("x{index}"))
        .collect::<Vec<_>>();
    let equations = variables
        .iter()
        .enumerate()
        .map(|(index, variable)| {
            RustedSciThe::symbolic::symbolic_engine::Expr::parse_expression(&format!(
                "{variable}-{:.1}",
                index as f64 + 1.0
            ))
        })
        .collect::<Vec<_>>();

    let mut jacobian = Jacobian::new();
    jacobian.set_vector_of_functions(equations);
    jacobian.set_variables(variables.iter().map(String::as_str).collect());
    jacobian.calc_jacobian();
    jacobian.lambdify_vector_funvector_DVector();
    jacobian.lambdify_jacobian_DMatrix_parallel();
    jacobian
}

fn sparse_chain_data(dimension: usize) -> (Vec<String>, Vec<String>, DVector<f64>) {
    sparse_corpus_data(SparseCorpus::QuadraticChain, dimension)
}

#[derive(Clone, Copy)]
enum SparseCorpus {
    QuadraticChain,
    BroydenTridiagonal,
    NonlinearPoisson,
    BandFive,
}

impl SparseCorpus {
    fn name(self) -> &'static str {
        match self {
            Self::QuadraticChain => "quadratic-chain",
            Self::BroydenTridiagonal => "broyden-tridiagonal",
            Self::NonlinearPoisson => "nonlinear-poisson",
            Self::BandFive => "band-five",
        }
    }
}

fn sparse_corpus_data(
    corpus: SparseCorpus,
    dimension: usize,
) -> (Vec<String>, Vec<String>, DVector<f64>) {
    let variables = (0..dimension)
        .map(|index| format!("x{index}"))
        .collect::<Vec<_>>();
    let target = DVector::from_iterator(
        dimension,
        (0..dimension).map(|index| match corpus {
            SparseCorpus::QuadraticChain | SparseCorpus::BroydenTridiagonal => {
                1.0 + index as f64 * 0.01
            }
            SparseCorpus::NonlinearPoisson => 0.2 + index as f64 * 0.001,
            SparseCorpus::BandFive => 0.5 + index as f64 * 0.003,
        }),
    );
    let equations = (0..dimension)
        .map(|index| {
            let variable = format!("x{index}");
            match corpus {
                SparseCorpus::QuadraticChain => {
                    let mut equation =
                        format!("({variable}^2-{:.17})", target[index] * target[index]);
                    append_shifted_neighbor(&mut equation, index.checked_sub(1), &target, 0.08);
                    append_shifted_neighbor(
                        &mut equation,
                        (index + 1 < dimension).then_some(index + 1),
                        &target,
                        0.08,
                    );
                    equation
                }
                SparseCorpus::BroydenTridiagonal => {
                    let mut equation = format!(
                        "({variable}^2-{:.17})+0.11*({variable}-{:.17})",
                        target[index] * target[index],
                        target[index]
                    );
                    append_shifted_neighbor(&mut equation, index.checked_sub(1), &target, -1.0);
                    append_shifted_neighbor(
                        &mut equation,
                        (index + 1 < dimension).then_some(index + 1),
                        &target,
                        -2.0,
                    );
                    equation
                }
                SparseCorpus::NonlinearPoisson => {
                    let h2 = 1.0 / ((dimension + 1) * (dimension + 1)) as f64;
                    let mut equation = format!(
                        "2*({variable}-{:.17})+{h2:.17}*(exp({variable})-exp({:.17}))",
                        target[index], target[index]
                    );
                    append_shifted_neighbor(&mut equation, index.checked_sub(1), &target, -1.0);
                    append_shifted_neighbor(
                        &mut equation,
                        (index + 1 < dimension).then_some(index + 1),
                        &target,
                        -1.0,
                    );
                    equation
                }
                SparseCorpus::BandFive => {
                    let mut equation =
                        format!("({variable}^2-{:.17})", target[index] * target[index]);
                    for offset in 1..=5 {
                        let coefficient = 0.01 / offset as f64;
                        append_shifted_neighbor(
                            &mut equation,
                            index.checked_sub(offset),
                            &target,
                            coefficient,
                        );
                        append_shifted_neighbor(
                            &mut equation,
                            (index + offset < dimension).then_some(index + offset),
                            &target,
                            coefficient,
                        );
                    }
                    equation
                }
            }
        })
        .collect::<Vec<_>>();
    (equations, variables, target)
}

fn append_shifted_neighbor(
    equation: &mut String,
    neighbor: Option<usize>,
    target: &DVector<f64>,
    coefficient: f64,
) {
    if let Some(neighbor) = neighbor {
        equation.push_str(&format!(
            "{coefficient:+.17}*(x{neighbor}-{:.17})",
            target[neighbor]
        ));
    }
}

fn legacy_sparse_chain_problem(dimension: usize) -> Jacobian {
    legacy_sparse_corpus_problem(SparseCorpus::QuadraticChain, dimension)
}

fn legacy_sparse_corpus_problem(corpus: SparseCorpus, dimension: usize) -> Jacobian {
    let (equations, variables, _target) = sparse_corpus_data(corpus, dimension);
    let expressions = equations
        .iter()
        .map(|equation| RustedSciThe::symbolic::symbolic_engine::Expr::parse_expression(equation))
        .collect::<Vec<_>>();
    let mut jacobian = Jacobian::new();
    jacobian.set_vector_of_functions(expressions);
    jacobian.set_variables(variables.iter().map(String::as_str).collect());
    jacobian.calc_jacobian();
    jacobian.lambdify_vector_funvector_DVector();
    jacobian.lambdify_jacobian_DMatrix_parallel();
    jacobian
}

fn prepared_sparse_chain_problem(
    dimension: usize,
    execution_policy: LambdifyExecutionPolicy,
) -> (PreparedSymbolicNonlinearProblem, DVector<f64>) {
    prepared_sparse_corpus_problem(SparseCorpus::QuadraticChain, dimension, execution_policy)
}

fn prepared_sparse_corpus_problem(
    corpus: SparseCorpus,
    dimension: usize,
    execution_policy: LambdifyExecutionPolicy,
) -> (PreparedSymbolicNonlinearProblem, DVector<f64>) {
    let (equations, variables, target) = sparse_corpus_data(corpus, dimension);
    let prepared = PreparedSymbolicNonlinearProblem::from_strings(
        equations,
        SymbolicProblemOptions::new()
            .with_variables(variables)
            .with_lambdify_execution_policy(execution_policy)
            .with_lambdify_backend(),
    )
    .expect("sparse chain Lambdify problem should prepare");
    (prepared, target)
}

struct LegacyLambdifyProblem<'a> {
    backend: &'a Jacobian,
    dimension: usize,
}

impl NonlinearProblem for LegacyLambdifyProblem<'_> {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn residual(
        &self,
        x: &DVector<f64>,
    ) -> Result<DVector<f64>, RustedSciThe::numerical::Nonlinear_systems::error::SolveError> {
        Ok((self.backend.lambdified_function_DVector)(x))
    }
}

impl JacobianProvider for LegacyLambdifyProblem<'_> {
    fn jacobian(
        &self,
        x: &DVector<f64>,
    ) -> Result<DMatrix<f64>, RustedSciThe::numerical::Nonlinear_systems::error::SolveError> {
        Ok((self.backend.lambdified_jacobian_DMatrix)(x))
    }
}

fn diagonally_dominant_system(dimension: usize) -> (DMatrix<f64>, DVector<f64>) {
    let mut matrix = DMatrix::zeros(dimension, dimension);
    for row in 0..dimension {
        for column in 0..dimension {
            matrix[(row, column)] = if row == column {
                4.0 + row as f64 * 0.01
            } else {
                0.01
            };
        }
    }
    let rhs = DVector::from_iterator(dimension, (0..dimension).map(|index| index as f64 + 1.0));
    (matrix, rhs)
}

fn bench_symbolic_dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("nonlinear_symbolic_dispatch");
    for dimension in [4, 16, 64] {
        let problem = symbolic_problem(dimension);
        let legacy = legacy_lambdify_problem(dimension);
        let point = DVector::from_element(dimension, 0.5);
        let mut residual = DVector::zeros(dimension);
        let mut jacobian = DMatrix::zeros(dimension, dimension);

        group.bench_with_input(
            BenchmarkId::new("prepared_residual_owned", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    let value = problem
                        .residual(black_box(&point))
                        .expect("residual should evaluate");
                    black_box(value);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("prepared_residual_into", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    problem
                        .residual_into(black_box(&point), &mut residual)
                        .expect("residual_into should evaluate");
                    black_box(&residual);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("prepared_jacobian_owned", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    let value = problem
                        .jacobian(black_box(&point))
                        .expect("jacobian should evaluate");
                    black_box(value);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("prepared_jacobian_into", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    problem
                        .jacobian_into(black_box(&point), &mut jacobian)
                        .expect("jacobian_into should evaluate");
                    black_box(&jacobian);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("legacy_parallel_residual", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    let value = (legacy.lambdified_function_DVector)(black_box(&point));
                    black_box(value);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("legacy_parallel_jacobian", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    let value = (legacy.lambdified_jacobian_DMatrix)(black_box(&point));
                    black_box(value);
                });
            },
        );
    }
    group.finish();
}

fn realistic_lambdify_problem(
    dimension: usize,
) -> (PreparedSymbolicNonlinearProblem, DVector<f64>) {
    realistic_lambdify_problem_with_policy(dimension, LambdifyExecutionPolicy::Sequential)
}

fn realistic_lambdify_problem_with_policy(
    dimension: usize,
    execution_policy: LambdifyExecutionPolicy,
) -> (PreparedSymbolicNonlinearProblem, DVector<f64>) {
    let target = DVector::from_iterator(
        dimension,
        (0..dimension).map(|index| 1.0 + index as f64 * 0.015),
    );
    let equations = (0..dimension)
        .map(|index| {
            let variable = format!("x{index}");
            let mut equation = format!("a*({variable}^2-{:.17})", target[index] * target[index]);
            if index > 0 {
                equation.push_str(&format!("+0.04*(x{}-{:.17})", index - 1, target[index - 1]));
            }
            if index + 1 < dimension {
                equation.push_str(&format!("+0.04*(x{}-{:.17})", index + 1, target[index + 1]));
            }
            equation
        })
        .collect::<Vec<_>>();
    let variables = (0..dimension)
        .map(|index| format!("x{index}"))
        .collect::<Vec<_>>();
    let prepared = PreparedSymbolicNonlinearProblem::from_strings(
        equations,
        SymbolicProblemOptions::new()
            .with_variables(variables)
            .with_equation_parameters(vec!["a".to_string()])
            .with_lambdify_execution_policy(execution_policy)
            .with_lambdify_backend(),
    )
    .expect("realistic Lambdify benchmark problem should prepare");
    (prepared, target)
}

fn bench_lambdify_realistic_end_to_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("nonlinear_lambdify_realistic_end_to_end");
    for dimension in [16, 64] {
        let (prepared, _target) = realistic_lambdify_problem(dimension);
        let bound = prepared
            .bind_values(DVector::from_vec(vec![1.0]))
            .expect("realistic Lambdify benchmark binding should succeed");
        let initial = DVector::from_element(dimension, 0.8);
        let options = SolveOptions {
            tolerance: 1e-10,
            max_iterations: 80,
            diagnostics: DiagnosticsOptions {
                collect_history: false,
                collect_statistics: false,
                enable_logging: false,
                ..DiagnosticsOptions::default()
            },
            ..SolveOptions::default()
        };

        for method_template in method_matrix() {
            let method_name = method_template.name();
            group.bench_with_input(
                BenchmarkId::new(method_name, dimension),
                &dimension,
                |b, _| {
                    b.iter(|| {
                        let result = method_template.clone().solve(
                            &bound,
                            black_box(initial.clone()),
                            options.clone(),
                        );
                        let _ = black_box(result);
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_lambdify_parameterized_dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("nonlinear_lambdify_parameterized_dispatch");
    for dimension in [16, 64, 128] {
        let (prepared, _target) = realistic_lambdify_problem(dimension);
        let bound = prepared
            .bind_values(DVector::from_vec(vec![1.0]))
            .expect("parameterized Lambdify benchmark binding should succeed");
        let point = DVector::from_element(dimension, 0.8);
        let mut residual = DVector::zeros(dimension);
        let mut jacobian = DMatrix::zeros(dimension, dimension);

        group.bench_with_input(
            BenchmarkId::new("residual_owned", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    let value = bound
                        .residual(black_box(&point))
                        .expect("parameterized residual should evaluate");
                    black_box(value);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("residual_into", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    bound
                        .residual_into(black_box(&point), &mut residual)
                        .expect("parameterized residual_into should evaluate");
                    black_box(&residual);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("jacobian_owned", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    let value = bound
                        .jacobian(black_box(&point))
                        .expect("parameterized Jacobian should evaluate");
                    black_box(value);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("jacobian_into", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    bound
                        .jacobian_into(black_box(&point), &mut jacobian)
                        .expect("parameterized jacobian_into should evaluate");
                    black_box(&jacobian);
                });
            },
        );
    }
    group.finish();
}

fn bench_lambdify_jacobian_execution_policy(c: &mut Criterion) {
    let mut group = c.benchmark_group("nonlinear_lambdify_jacobian_execution_policy");
    for dimension in [16, 64, 128, 256] {
        let (sequential_prepared, _target) =
            realistic_lambdify_problem_with_policy(dimension, LambdifyExecutionPolicy::Sequential);
        let (parallel_prepared, _target) = realistic_lambdify_problem_with_policy(
            dimension,
            LambdifyExecutionPolicy::Parallel { min_work: 64 },
        );
        let sequential = sequential_prepared
            .bind_values(DVector::from_vec(vec![1.0]))
            .expect("sequential policy binding should succeed");
        let parallel = parallel_prepared
            .bind_values(DVector::from_vec(vec![1.0]))
            .expect("parallel policy binding should succeed");
        let point = DVector::from_element(dimension, 0.8);
        let mut sequential_residual = DVector::zeros(dimension);
        let mut parallel_residual = DVector::zeros(dimension);
        let mut sequential_jacobian = DMatrix::zeros(dimension, dimension);
        let mut parallel_jacobian = DMatrix::zeros(dimension, dimension);

        group.bench_with_input(
            BenchmarkId::new("sequential_residual_into", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    sequential
                        .residual_into(black_box(&point), &mut sequential_residual)
                        .expect("sequential residual should evaluate");
                    black_box(&sequential_residual);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("parallel_residual_into", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    parallel
                        .residual_into(black_box(&point), &mut parallel_residual)
                        .expect("parallel residual should evaluate");
                    black_box(&parallel_residual);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("sequential_jacobian_into", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    sequential
                        .jacobian_into(black_box(&point), &mut sequential_jacobian)
                        .expect("sequential Jacobian should evaluate");
                    black_box(&sequential_jacobian);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("parallel_jacobian_into", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    parallel
                        .jacobian_into(black_box(&point), &mut parallel_jacobian)
                        .expect("parallel Jacobian should evaluate");
                    black_box(&parallel_jacobian);
                });
            },
        );
    }
    group.finish();
}

fn bench_large_lambdify_legacy_vs_prepared(c: &mut Criterion) {
    let mut callback_group = c.benchmark_group("nonlinear_lambdify_large_legacy_vs_prepared");
    for dimension in [128, 512] {
        let legacy_backend = legacy_sparse_chain_problem(dimension);
        let legacy = LegacyLambdifyProblem {
            backend: &legacy_backend,
            dimension,
        };
        let (sequential_prepared, target) =
            prepared_sparse_chain_problem(dimension, LambdifyExecutionPolicy::Sequential);
        let (parallel_prepared, _) = prepared_sparse_chain_problem(
            dimension,
            LambdifyExecutionPolicy::Parallel { min_work: 1 },
        );
        let sequential = sequential_prepared
            .bind_without_parameters()
            .expect("large sequential problem should bind");
        let parallel = parallel_prepared
            .bind_without_parameters()
            .expect("large parallel problem should bind");
        let point = target.map(|value| value * 0.8);
        let mut sequential_residual = DVector::zeros(dimension);
        let mut parallel_residual = DVector::zeros(dimension);
        let mut sequential_jacobian = DMatrix::zeros(dimension, dimension);
        let mut parallel_jacobian = DMatrix::zeros(dimension, dimension);

        callback_group.bench_with_input(
            BenchmarkId::new("legacy_parallel_residual_owned", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    let value = legacy
                        .residual(black_box(&point))
                        .expect("legacy residual should evaluate");
                    black_box(value);
                });
            },
        );
        callback_group.bench_with_input(
            BenchmarkId::new("prepared_sequential_residual_into", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    sequential
                        .residual_into(black_box(&point), &mut sequential_residual)
                        .expect("prepared sequential residual should evaluate");
                    black_box(&sequential_residual);
                });
            },
        );
        callback_group.bench_with_input(
            BenchmarkId::new("prepared_parallel_residual_into", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    parallel
                        .residual_into(black_box(&point), &mut parallel_residual)
                        .expect("prepared parallel residual should evaluate");
                    black_box(&parallel_residual);
                });
            },
        );
        callback_group.bench_with_input(
            BenchmarkId::new("legacy_parallel_jacobian_owned", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    let value = legacy
                        .jacobian(black_box(&point))
                        .expect("legacy Jacobian should evaluate");
                    black_box(value);
                });
            },
        );
        callback_group.bench_with_input(
            BenchmarkId::new("prepared_sequential_jacobian_into", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    sequential
                        .jacobian_into(black_box(&point), &mut sequential_jacobian)
                        .expect("prepared sequential Jacobian should evaluate");
                    black_box(&sequential_jacobian);
                });
            },
        );
        callback_group.bench_with_input(
            BenchmarkId::new("prepared_parallel_jacobian_into", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    parallel
                        .jacobian_into(black_box(&point), &mut parallel_jacobian)
                        .expect("prepared parallel Jacobian should evaluate");
                    black_box(&parallel_jacobian);
                });
            },
        );
    }
    callback_group.finish();

    let mut solve_group = c.benchmark_group("nonlinear_lambdify_large_legacy_vs_prepared_solve");
    for dimension in [128] {
        let legacy_backend = legacy_sparse_chain_problem(dimension);
        let legacy = LegacyLambdifyProblem {
            backend: &legacy_backend,
            dimension,
        };
        let (sequential_prepared, target) =
            prepared_sparse_chain_problem(dimension, LambdifyExecutionPolicy::Sequential);
        let (parallel_prepared, _) = prepared_sparse_chain_problem(
            dimension,
            LambdifyExecutionPolicy::Parallel { min_work: 1 },
        );
        let sequential = sequential_prepared
            .bind_without_parameters()
            .expect("large sequential solve problem should bind");
        let parallel = parallel_prepared
            .bind_without_parameters()
            .expect("large parallel solve problem should bind");
        let initial = target.map(|value| value * 0.8);
        let options = SolveOptions {
            tolerance: 1e-10,
            max_iterations: 20,
            diagnostics: DiagnosticsOptions {
                collect_history: false,
                collect_statistics: false,
                enable_logging: false,
                ..DiagnosticsOptions::default()
            },
            ..SolveOptions::default()
        };

        solve_group.bench_with_input(
            BenchmarkId::new("legacy_newton_solve", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    let result = SolverEngine::new(NewtonMethod, options.clone())
                        .solve(&legacy, black_box(initial.clone()))
                        .expect("legacy large Newton solve should converge");
                    black_box(result.x);
                });
            },
        );
        solve_group.bench_with_input(
            BenchmarkId::new("prepared_sequential_newton_solve", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    let result = SolverEngine::new(NewtonMethod, options.clone())
                        .solve(&sequential, black_box(initial.clone()))
                        .expect("prepared sequential Newton solve should converge");
                    black_box(result.x);
                });
            },
        );
        solve_group.bench_with_input(
            BenchmarkId::new("prepared_parallel_newton_solve", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    let result = SolverEngine::new(NewtonMethod, options.clone())
                        .solve(&parallel, black_box(initial.clone()))
                        .expect("prepared parallel Newton solve should converge");
                    black_box(result.x);
                });
            },
        );
    }
    solve_group.finish();
}

fn sparse_corpus_nnz(corpus: SparseCorpus, dimension: usize) -> usize {
    (0..dimension)
        .map(|index| match corpus {
            SparseCorpus::QuadraticChain | SparseCorpus::BroydenTridiagonal => {
                1 + usize::from(index > 0) + usize::from(index + 1 < dimension)
            }
            SparseCorpus::NonlinearPoisson => {
                1 + usize::from(index > 0) + usize::from(index + 1 < dimension)
            }
            SparseCorpus::BandFive => {
                1 + (1..=5)
                    .map(|offset| {
                        usize::from(index >= offset) + usize::from(index + offset < dimension)
                    })
                    .sum::<usize>()
            }
        })
        .sum()
}

fn bench_large_lambdify_corpus(c: &mut Criterion) {
    let corpus = [
        SparseCorpus::BroydenTridiagonal,
        SparseCorpus::NonlinearPoisson,
        SparseCorpus::BandFive,
    ];
    let mut callback_group = c.benchmark_group("nonlinear_lambdify_large_corpus_callbacks");
    for case in corpus {
        for dimension in [128, 512] {
            let legacy_backend = legacy_sparse_corpus_problem(case, dimension);
            let legacy = LegacyLambdifyProblem {
                backend: &legacy_backend,
                dimension,
            };
            let (sequential_prepared, target) = prepared_sparse_corpus_problem(
                case,
                dimension,
                LambdifyExecutionPolicy::Sequential,
            );
            let (parallel_prepared, _) = prepared_sparse_corpus_problem(
                case,
                dimension,
                LambdifyExecutionPolicy::Parallel { min_work: 1 },
            );
            let sequential = sequential_prepared
                .bind_without_parameters()
                .expect("corpus sequential problem should bind");
            let parallel = parallel_prepared
                .bind_without_parameters()
                .expect("corpus parallel problem should bind");
            let point = target.map(|value| value * 0.8);
            let mut sequential_residual = DVector::zeros(dimension);
            let mut parallel_residual = DVector::zeros(dimension);
            let mut sequential_jacobian = DMatrix::zeros(dimension, dimension);
            let mut parallel_jacobian = DMatrix::zeros(dimension, dimension);
            eprintln!(
                "[Nonlinear large corpus] case={} dimension={} structural_nnz={}",
                case.name(),
                dimension,
                sparse_corpus_nnz(case, dimension)
            );

            let id =
                |metric: &str| BenchmarkId::new(format!("{}/{}", case.name(), metric), dimension);
            callback_group.bench_with_input(id("legacy_residual_owned"), &dimension, |b, _| {
                b.iter(|| {
                    let value = legacy
                        .residual(black_box(&point))
                        .expect("corpus legacy residual should evaluate");
                    black_box(value);
                });
            });
            callback_group.bench_with_input(
                id("prepared_sequential_residual_into"),
                &dimension,
                |b, _| {
                    b.iter(|| {
                        sequential
                            .residual_into(black_box(&point), &mut sequential_residual)
                            .expect("corpus sequential residual should evaluate");
                        black_box(&sequential_residual);
                    });
                },
            );
            callback_group.bench_with_input(
                id("prepared_parallel_residual_into"),
                &dimension,
                |b, _| {
                    b.iter(|| {
                        parallel
                            .residual_into(black_box(&point), &mut parallel_residual)
                            .expect("corpus parallel residual should evaluate");
                        black_box(&parallel_residual);
                    });
                },
            );
            callback_group.bench_with_input(
                id("prepared_sequential_residual_owned"),
                &dimension,
                |b, _| {
                    b.iter(|| {
                        let value = sequential
                            .residual(black_box(&point))
                            .expect("corpus sequential owned residual should evaluate");
                        black_box(value);
                    });
                },
            );
            callback_group.bench_with_input(
                id("prepared_parallel_residual_owned"),
                &dimension,
                |b, _| {
                    b.iter(|| {
                        let value = parallel
                            .residual(black_box(&point))
                            .expect("corpus parallel owned residual should evaluate");
                        black_box(value);
                    });
                },
            );
            callback_group.bench_with_input(id("legacy_jacobian_owned"), &dimension, |b, _| {
                b.iter(|| {
                    let value = legacy
                        .jacobian(black_box(&point))
                        .expect("corpus legacy Jacobian should evaluate");
                    black_box(value);
                });
            });
            callback_group.bench_with_input(
                id("prepared_sequential_jacobian_into"),
                &dimension,
                |b, _| {
                    b.iter(|| {
                        sequential
                            .jacobian_into(black_box(&point), &mut sequential_jacobian)
                            .expect("corpus sequential Jacobian should evaluate");
                        black_box(&sequential_jacobian);
                    });
                },
            );
            callback_group.bench_with_input(
                id("prepared_parallel_jacobian_into"),
                &dimension,
                |b, _| {
                    b.iter(|| {
                        parallel
                            .jacobian_into(black_box(&point), &mut parallel_jacobian)
                            .expect("corpus parallel Jacobian should evaluate");
                        black_box(&parallel_jacobian);
                    });
                },
            );
            callback_group.bench_with_input(
                id("prepared_sequential_jacobian_owned"),
                &dimension,
                |b, _| {
                    b.iter(|| {
                        let value = sequential
                            .jacobian(black_box(&point))
                            .expect("corpus sequential owned Jacobian should evaluate");
                        black_box(value);
                    });
                },
            );
            callback_group.bench_with_input(
                id("prepared_parallel_jacobian_owned"),
                &dimension,
                |b, _| {
                    b.iter(|| {
                        let value = parallel
                            .jacobian(black_box(&point))
                            .expect("corpus parallel owned Jacobian should evaluate");
                        black_box(value);
                    });
                },
            );
        }
    }
    callback_group.finish();

    let mut solve_group = c.benchmark_group("nonlinear_lambdify_large_corpus_solves");
    for case in corpus {
        for dimension in [128, 512] {
            let legacy_backend = legacy_sparse_corpus_problem(case, dimension);
            let legacy = LegacyLambdifyProblem {
                backend: &legacy_backend,
                dimension,
            };
            let (sequential_prepared, target) = prepared_sparse_corpus_problem(
                case,
                dimension,
                LambdifyExecutionPolicy::Sequential,
            );
            let (parallel_prepared, _) = prepared_sparse_corpus_problem(
                case,
                dimension,
                LambdifyExecutionPolicy::Parallel { min_work: 1 },
            );
            let sequential = sequential_prepared
                .bind_without_parameters()
                .expect("corpus sequential solve problem should bind");
            let parallel = parallel_prepared
                .bind_without_parameters()
                .expect("corpus parallel solve problem should bind");
            let initial = target.map(|value| value * 0.8);
            let options = SolveOptions {
                tolerance: 1e-10,
                max_iterations: 20,
                diagnostics: DiagnosticsOptions {
                    collect_history: false,
                    collect_statistics: false,
                    enable_logging: false,
                    ..DiagnosticsOptions::default()
                },
                ..SolveOptions::default()
            };
            let id =
                |metric: &str| BenchmarkId::new(format!("{}/{}", case.name(), metric), dimension);

            solve_group.bench_with_input(id("legacy_newton"), &dimension, |b, _| {
                b.iter(|| {
                    let result = SolverEngine::new(NewtonMethod, options.clone())
                        .solve(&legacy, black_box(initial.clone()))
                        .expect("corpus legacy Newton solve should converge");
                    black_box(result.x);
                });
            });
            solve_group.bench_with_input(id("prepared_sequential_newton"), &dimension, |b, _| {
                b.iter(|| {
                    let result = SolverEngine::new(NewtonMethod, options.clone())
                        .solve(&sequential, black_box(initial.clone()))
                        .expect("corpus sequential Newton solve should converge");
                    black_box(result.x);
                });
            });
            solve_group.bench_with_input(id("prepared_parallel_newton"), &dimension, |b, _| {
                b.iter(|| {
                    let result = SolverEngine::new(NewtonMethod, options.clone())
                        .solve(&parallel, black_box(initial.clone()))
                        .expect("corpus parallel Newton solve should converge");
                    black_box(result.x);
                });
            });
        }
    }
    solve_group.finish();
}

fn bench_lambdify_parallel_threshold_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("nonlinear_lambdify_parallel_threshold_sweep");
    for case in [
        SparseCorpus::BroydenTridiagonal,
        SparseCorpus::NonlinearPoisson,
        SparseCorpus::BandFive,
    ] {
        let dimension = 512;
        let nnz = sparse_corpus_nnz(case, dimension);
        let (sequential_prepared, target) =
            prepared_sparse_corpus_problem(case, dimension, LambdifyExecutionPolicy::Sequential);
        let sequential = sequential_prepared
            .bind_without_parameters()
            .expect("threshold sequential problem should bind");
        let point = target.map(|value| value * 0.8);
        let mut sequential_jacobian = DMatrix::zeros(dimension, dimension);
        let policies = [
            ("sequential", LambdifyExecutionPolicy::Sequential),
            (
                "parallel-1",
                LambdifyExecutionPolicy::Parallel { min_work: 1 },
            ),
            (
                "parallel-at-nnz",
                LambdifyExecutionPolicy::Parallel { min_work: nnz },
            ),
            (
                "parallel-above-nnz",
                LambdifyExecutionPolicy::Parallel { min_work: nnz + 1 },
            ),
        ];
        eprintln!(
            "[Nonlinear threshold sweep] case={} dimension={} structural_nnz={}",
            case.name(),
            dimension,
            nnz
        );

        for (label, policy) in policies {
            let (prepared, _) = prepared_sparse_corpus_problem(case, dimension, policy);
            let bound = prepared
                .bind_without_parameters()
                .expect("threshold policy problem should bind");
            let mut jacobian = DMatrix::zeros(dimension, dimension);
            group.bench_with_input(
                BenchmarkId::new(format!("{}/jacobian_into", case.name()), label),
                &dimension,
                |b, _| {
                    b.iter(|| {
                        bound
                            .jacobian_into(black_box(&point), &mut jacobian)
                            .expect("threshold Jacobian should evaluate");
                        black_box(&jacobian);
                    });
                },
            );
        }

        group.bench_with_input(
            BenchmarkId::new(format!("{}/sequential_reference", case.name()), dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    sequential
                        .jacobian_into(black_box(&point), &mut sequential_jacobian)
                        .expect("sequential threshold reference should evaluate");
                    black_box(&sequential_jacobian);
                });
            },
        );
    }
    group.finish();
}

fn bench_dense_factor_and_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("nonlinear_dense_factor_and_solve");
    for dimension in [8, 32, 128] {
        let (matrix, rhs) = diagonally_dominant_system(dimension);
        group.bench_with_input(BenchmarkId::new("lu", dimension), &dimension, |b, _| {
            b.iter(|| {
                let solution =
                    solve_linear_system(LinearSolverKind::Lu, black_box(&matrix), black_box(&rhs))
                        .expect("benchmark system should be nonsingular");
                black_box(solution);
            });
        });
    }
    group.finish();
}

fn bench_full_newton(c: &mut Criterion) {
    let mut group = c.benchmark_group("nonlinear_full_newton");
    for dimension in [4, 16, 64] {
        let problem = symbolic_problem(dimension);
        let initial = DVector::zeros(dimension);
        let options = SolveOptions {
            max_iterations: 8,
            diagnostics: DiagnosticsOptions {
                collect_history: false,
                collect_statistics: false,
                ..DiagnosticsOptions::default()
            },
            ..SolveOptions::default()
        };

        group.bench_with_input(BenchmarkId::new("newton", dimension), &dimension, |b, _| {
            b.iter(|| {
                let result = SolverEngine::new(NewtonMethod, options.clone())
                    .solve(&problem, black_box(initial.clone()))
                    .expect("benchmark Newton solve should succeed");
                black_box(result.x);
            });
        });
    }
    group.finish();
}

fn bench_telemetry_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("nonlinear_telemetry_overhead");
    let dimension = 64;
    let problem = symbolic_problem(dimension);
    let initial = DVector::zeros(dimension);
    let cases = [
        (
            "enabled",
            DiagnosticsOptions {
                collect_history: true,
                collect_statistics: true,
                enable_logging: false,
                enable_memory_diagnostics: false,
                ..DiagnosticsOptions::default()
            },
        ),
        (
            "disabled",
            DiagnosticsOptions {
                collect_history: false,
                collect_statistics: false,
                enable_logging: false,
                enable_memory_diagnostics: false,
                ..DiagnosticsOptions::default()
            },
        ),
    ];

    for (label, diagnostics) in cases {
        let options = SolveOptions {
            max_iterations: 8,
            diagnostics,
            ..SolveOptions::default()
        };
        group.bench_function(label, |b| {
            b.iter(|| {
                let result = SolverEngine::new(NewtonMethod, options.clone())
                    .solve(&problem, black_box(initial.clone()))
                    .expect("telemetry benchmark solve should succeed");
                black_box((result.x, result.statistics));
            });
        });
    }
    group.finish();
}

struct DenseQuadraticProblem {
    dimension: usize,
}

impl NonlinearProblem for DenseQuadraticProblem {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn residual(
        &self,
        x: &DVector<f64>,
    ) -> Result<DVector<f64>, RustedSciThe::numerical::Nonlinear_systems::error::SolveError> {
        Ok(DVector::from_iterator(
            self.dimension,
            x.iter().map(|value| value * value - 1.0),
        ))
    }
}

impl JacobianProvider for DenseQuadraticProblem {
    fn jacobian(
        &self,
        x: &DVector<f64>,
    ) -> Result<DMatrix<f64>, RustedSciThe::numerical::Nonlinear_systems::error::SolveError> {
        Ok(DMatrix::from_fn(
            self.dimension,
            self.dimension,
            |row, column| {
                if row == column { 2.0 * x[row] } else { 0.0 }
            },
        ))
    }
}

struct RosenbrockProblem;

impl NonlinearProblem for RosenbrockProblem {
    fn dimension(&self) -> usize {
        2
    }

    fn residual(
        &self,
        x: &DVector<f64>,
    ) -> Result<DVector<f64>, RustedSciThe::numerical::Nonlinear_systems::error::SolveError> {
        Ok(DVector::from_vec(vec![
            10.0 * (x[1] - x[0] * x[0]),
            1.0 - x[0],
        ]))
    }
}

impl JacobianProvider for RosenbrockProblem {
    fn jacobian(
        &self,
        x: &DVector<f64>,
    ) -> Result<DMatrix<f64>, RustedSciThe::numerical::Nonlinear_systems::error::SolveError> {
        Ok(DMatrix::from_row_slice(
            2,
            2,
            &[-20.0 * x[0], 10.0, -1.0, 0.0],
        ))
    }
}

fn method_matrix() -> Vec<NonlinearSolverMethod> {
    vec![
        NonlinearSolverMethod::Newton(NewtonMethod),
        NonlinearSolverMethod::DampedNewton(DampedNewtonMethod::default()),
        NonlinearSolverMethod::DampedNewtonAdvanced(DampedNewtonMethodAdvanced::default()),
        NonlinearSolverMethod::LevenbergMarquardt(LevenbergMarquardtMethod::default()),
        NonlinearSolverMethod::LevenbergMarquardtMinpack(LevenbergMarquardtMinpack::default()),
        NonlinearSolverMethod::NielsenLevenbergMarquardt(NielsenLevenbergMarquardtMethod::default()),
        NonlinearSolverMethod::NielsenLevenbergMarquardtAdvanced(
            NielsenLevenbergMarquardtMethodAdvanced::default(),
        ),
        NonlinearSolverMethod::TrustRegion(TrustRegionMethod::default()),
        NonlinearSolverMethod::PowellDogleg(PowellDoglegMethod::default()),
        NonlinearSolverMethod::TrustRegionLM(TrustRegionLMMethod::default()),
    ]
}

fn hot_path_options(collect_statistics: bool, collect_history: bool) -> SolveOptions {
    SolveOptions {
        tolerance: 1e-8,
        max_iterations: 64,
        diagnostics: DiagnosticsOptions {
            collect_statistics,
            collect_history,
            enable_logging: false,
            ..DiagnosticsOptions::default()
        },
        ..SolveOptions::default()
    }
}

fn bench_method_hot_path(c: &mut Criterion) {
    let mut group = c.benchmark_group("nonlinear_method_hot_path");
    for dimension in [8, 32] {
        let problem = DenseQuadraticProblem { dimension };
        let initial = DVector::from_element(dimension, 0.25);
        let options = hot_path_options(false, false);
        for method_template in method_matrix() {
            let method_name = method_template.name();
            group.bench_with_input(
                BenchmarkId::new(method_name, dimension),
                &dimension,
                |b, _| {
                    b.iter(|| {
                        let result = method_template.clone().solve(
                            &problem,
                            black_box(initial.clone()),
                            options.clone(),
                        );
                        let _ = black_box(result);
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_rejected_step_path(c: &mut Criterion) {
    let problem = RosenbrockProblem;
    let initial = DVector::from_vec(vec![-1.2, 1.0]);
    let diagnostic_options = hot_path_options(true, false);
    let preflight = NonlinearSolverMethod::DampedNewton(DampedNewtonMethod::default())
        .solve(&problem, initial.clone(), diagnostic_options)
        .expect("Rosenbrock rejection preflight should finish");
    assert!(
        preflight.statistics.rejected_steps > 0,
        "rejection benchmark lost its rejected-step workload"
    );

    let mut group = c.benchmark_group("nonlinear_rejected_step_path");
    let options = hot_path_options(false, false);
    for method_template in [
        NonlinearSolverMethod::DampedNewton(DampedNewtonMethod::default()),
        NonlinearSolverMethod::DampedNewtonAdvanced(DampedNewtonMethodAdvanced::default()),
        NonlinearSolverMethod::PowellDogleg(PowellDoglegMethod::default()),
    ] {
        let method_name = method_template.name();
        group.bench_function(method_name, |b| {
            b.iter(|| {
                let result = method_template.clone().solve(
                    &problem,
                    black_box(initial.clone()),
                    options.clone(),
                );
                let _ = black_box(result);
            });
        });
    }
    group.finish();
}

fn bench_iteration_state_and_trial_vectors(c: &mut Criterion) {
    let mut group = c.benchmark_group("nonlinear_iteration_state_and_trial_vectors");
    for dimension in [8, 32, 128] {
        let x = DVector::from_element(dimension, 0.25);
        let residual = DVector::from_element(dimension, -0.9375);
        let jacobian = DMatrix::identity(dimension, dimension);
        let step = DVector::from_element(dimension, 0.125);

        group.bench_with_input(
            BenchmarkId::new("owned_iteration_state_snapshot", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    // Mirrors the current public method contract: each step
                    // receives owned vector/matrix fields inside a snapshot.
                    let state = IterationState {
                        iteration: 1,
                        x: black_box(x.clone()),
                        residual: black_box(residual.clone()),
                        jacobian: black_box(jacobian.clone()),
                        residual_norm: 1.0,
                    };
                    black_box(state);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("trial_vector", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    let trial = black_box(&x) - black_box(&step);
                    black_box(trial);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("rejected_current_x_clone", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    // This is the avoidable payload currently used by
                    // rejected trust-region outcomes.
                    black_box(x.clone());
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_symbolic_dispatch,
    bench_lambdify_realistic_end_to_end,
    bench_lambdify_parameterized_dispatch,
    bench_lambdify_jacobian_execution_policy,
    bench_large_lambdify_legacy_vs_prepared,
    bench_large_lambdify_corpus,
    bench_lambdify_parallel_threshold_sweep,
    bench_dense_factor_and_solve,
    bench_full_newton,
    bench_telemetry_overhead,
    bench_method_hot_path,
    bench_rejected_step_path,
    bench_iteration_state_and_trial_vectors,
);
criterion_main!(benches);
