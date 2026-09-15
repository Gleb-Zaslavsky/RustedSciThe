//! Release story for full-solve Lambdify route diagnostics.
//!
//! This module compares the compatibility legacy callback, prepared
//! Sequential, and prepared Parallel routes on the same systems. Preparation
//! is outside the measured solve loop; solver-level timings and counters are
//! collected from `SolveStatistics`.
//!
//! Run the focused story in debug or release with:
//!
//! ```text
//! cargo test --lib numerical::Nonlinear_systems::nonlinear_lambdify_stage_story_tests::tests::large_lambdify_corpus_end_to_end_stage_story -- --nocapture --ignored --test-threads=1
//! cargo test --release --lib numerical::Nonlinear_systems::nonlinear_lambdify_stage_story_tests::tests::large_lambdify_corpus_end_to_end_stage_story -- --nocapture --ignored --test-threads=1
//! ```

#[cfg(test)]
mod tests {
    use nalgebra::{DMatrix, DVector};

    use crate::numerical::Nonlinear_systems::error::{SolveError, TerminationReason};
    use crate::numerical::Nonlinear_systems::prelude::{
        BoundSymbolicNonlinearProblem, DampedNewtonMethod, JacobianProvider,
        LambdifyExecutionPolicy, NonlinearProblem, PreparedSymbolicNonlinearProblem, SolveOptions,
        SolverEngine, SymbolicProblemOptions,
    };
    use crate::symbolic::symbolic_functions::Jacobian;

    const DEFAULT_RUNS: usize = 3;

    #[derive(Clone, Copy)]
    enum Corpus {
        QuadraticChain,
        NonlinearPoisson,
        BandFive,
    }

    impl Corpus {
        fn name(self) -> &'static str {
            match self {
                Self::QuadraticChain => "quadratic-chain",
                Self::NonlinearPoisson => "nonlinear-poisson",
                Self::BandFive => "band-five",
            }
        }
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

    fn corpus_data(corpus: Corpus, dimension: usize) -> (Vec<String>, Vec<String>, DVector<f64>) {
        let variables = (0..dimension)
            .map(|index| format!("x{index}"))
            .collect::<Vec<_>>();
        let target = DVector::from_iterator(
            dimension,
            (0..dimension).map(|index| match corpus {
                Corpus::QuadraticChain => 1.0 + index as f64 * 0.01,
                Corpus::NonlinearPoisson => 0.2 + index as f64 * 0.001,
                Corpus::BandFive => 0.5 + index as f64 * 0.003,
            }),
        );
        let equations = (0..dimension)
            .map(|index| {
                let variable = format!("x{index}");
                match corpus {
                    Corpus::QuadraticChain => {
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
                    Corpus::NonlinearPoisson => {
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
                    Corpus::BandFive => {
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

    fn structural_nnz(corpus: Corpus, dimension: usize) -> usize {
        (0..dimension)
            .map(|index| match corpus {
                Corpus::QuadraticChain | Corpus::NonlinearPoisson => {
                    1 + usize::from(index > 0) + usize::from(index + 1 < dimension)
                }
                Corpus::BandFive => {
                    1 + (1..=5)
                        .map(|offset| {
                            usize::from(index >= offset) + usize::from(index + offset < dimension)
                        })
                        .sum::<usize>()
                }
            })
            .sum()
    }

    struct LegacyProblem<'a> {
        backend: &'a Jacobian,
        dimension: usize,
    }

    impl NonlinearProblem for LegacyProblem<'_> {
        fn dimension(&self) -> usize {
            self.dimension
        }

        fn residual(&self, x: &DVector<f64>) -> Result<DVector<f64>, SolveError> {
            Ok((self.backend.lambdified_function_DVector)(x))
        }
    }

    impl JacobianProvider for LegacyProblem<'_> {
        fn jacobian(&self, x: &DVector<f64>) -> Result<DMatrix<f64>, SolveError> {
            Ok((self.backend.lambdified_jacobian_DMatrix)(x))
        }
    }

    fn legacy_problem(equations: &[String], variables: &[String]) -> Jacobian {
        let expressions = equations
            .iter()
            .map(|equation| crate::symbolic::symbolic_engine::Expr::parse_expression(equation))
            .collect::<Vec<_>>();
        let mut jacobian = Jacobian::new();
        jacobian.set_vector_of_functions(expressions);
        jacobian.set_variables(variables.iter().map(String::as_str).collect());
        jacobian.calc_jacobian();
        jacobian.lambdify_vector_funvector_DVector();
        jacobian.lambdify_jacobian_DMatrix_parallel();
        jacobian
    }

    fn prepared_problem(
        equations: Vec<String>,
        variables: Vec<String>,
        policy: LambdifyExecutionPolicy,
    ) -> PreparedSymbolicNonlinearProblem {
        PreparedSymbolicNonlinearProblem::from_strings(
            equations,
            SymbolicProblemOptions::new()
                .with_variables(variables)
                .with_lambdify_execution_policy(policy)
                .with_lambdify_backend(),
        )
        .expect("prepared Lambdify corpus should build")
    }

    enum Route<'a> {
        Legacy(LegacyProblem<'a>),
        Sequential(BoundSymbolicNonlinearProblem<'a>),
        Parallel(BoundSymbolicNonlinearProblem<'a>),
    }

    impl Route<'_> {
        fn solve(
            &self,
            initial: DVector<f64>,
            options: SolveOptions,
        ) -> Result<crate::numerical::Nonlinear_systems::engine::SolveResult, SolveError> {
            match self {
                Self::Legacy(problem) => SolverEngine::new(DampedNewtonMethod::default(), options)
                    .solve(problem, initial),
                Self::Sequential(problem) => {
                    SolverEngine::new(DampedNewtonMethod::default(), options)
                        .solve(problem, initial)
                }
                Self::Parallel(problem) => {
                    SolverEngine::new(DampedNewtonMethod::default(), options)
                        .solve(problem, initial)
                }
            }
        }
    }

    #[derive(Default)]
    struct Metrics {
        total_ms: Vec<f64>,
        residual_ms: Vec<f64>,
        jacobian_ms: Vec<f64>,
        linear_ms: Vec<f64>,
        residual_calls: Vec<f64>,
        jacobian_calls: Vec<f64>,
        linear_calls: Vec<f64>,
        iterations: Vec<f64>,
        max_solution_diff: f64,
    }

    impl Metrics {
        fn push(&mut self, result: &crate::numerical::Nonlinear_systems::engine::SolveResult) {
            let statistics = &result.statistics;
            self.total_ms
                .push(statistics.total_duration.as_secs_f64() * 1e3);
            self.residual_ms
                .push(statistics.residual_duration.as_secs_f64() * 1e3);
            self.jacobian_ms
                .push(statistics.jacobian_duration.as_secs_f64() * 1e3);
            self.linear_ms
                .push(statistics.linear_solve_duration.as_secs_f64() * 1e3);
            self.residual_calls
                .push(statistics.residual_evaluations as f64);
            self.jacobian_calls
                .push(statistics.jacobian_evaluations as f64);
            self.linear_calls.push(statistics.linear_solves as f64);
            self.iterations.push(statistics.iterations as f64);
        }
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

    fn mean_count(values: &[f64]) -> String {
        format!("{:.1}", values.iter().sum::<f64>() / values.len() as f64)
    }

    fn mean_us_per_call(durations_ms: &[f64], calls: &[f64]) -> String {
        let mean = durations_ms
            .iter()
            .zip(calls.iter())
            .map(|(duration, calls)| duration * 1e3 / calls.max(1.0))
            .sum::<f64>()
            / durations_ms.len() as f64;
        format!("{mean:.3}")
    }

    fn run_count() -> usize {
        std::env::var("NONLINEAR_LAMBDIFY_STAGE_RUNS")
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|runs| *runs >= 2)
            .unwrap_or(DEFAULT_RUNS)
    }

    #[test]
    #[ignore = "release-oriented full-solve Lambdify stage audit"]
    fn large_lambdify_corpus_end_to_end_stage_story() {
        let runs = run_count();
        println!(
            "[Nonlinear Lambdify stage story] method=damped-newton runs={runs}; times are milliseconds mean+/-std[min,max]"
        );
        println!(
            "case | dimension | route | structural_nnz | ok/runs | total_ms | residual_ms | jacobian_ms | linear_ms | residual_calls | jacobian_calls | linear_calls | iterations | res_us/call | jac_us/call | linear_us/call | max_diff"
        );

        for corpus in [
            Corpus::QuadraticChain,
            Corpus::NonlinearPoisson,
            Corpus::BandFive,
        ] {
            for dimension in [128, 512] {
                let (equations, variables, target) = corpus_data(corpus, dimension);
                let legacy_backend = legacy_problem(&equations, &variables);
                let sequential_prepared = prepared_problem(
                    equations.clone(),
                    variables.clone(),
                    LambdifyExecutionPolicy::Sequential,
                );
                let parallel_prepared = prepared_problem(
                    equations,
                    variables,
                    LambdifyExecutionPolicy::Parallel { min_work: 1 },
                );
                let sequential = sequential_prepared
                    .bind_without_parameters()
                    .expect("sequential corpus binding should succeed");
                let parallel = parallel_prepared
                    .bind_without_parameters()
                    .expect("parallel corpus binding should succeed");
                let routes = [
                    (
                        "legacy",
                        Route::Legacy(LegacyProblem {
                            backend: &legacy_backend,
                            dimension,
                        }),
                    ),
                    ("prepared-sequential", Route::Sequential(sequential)),
                    ("prepared-parallel", Route::Parallel(parallel)),
                ];
                // Keep the known root as the controlled reference while
                // avoiding an unrelated globalization failure in this
                // stage-comparison harness. Raw callback benchmarks retain
                // their wider `0.8 * target` point.
                let initial = target.map(|value| value * 0.99);
                let target_residual = (legacy_backend.lambdified_function_DVector)(&target);
                assert!(
                    target_residual.norm() < 1e-10,
                    "corpus target is not a numerical root: case={} dimension={} residual={:.3e}",
                    corpus.name(),
                    dimension,
                    target_residual.norm()
                );
                let options = SolveOptions {
                    tolerance: 1e-10,
                    // The wider corpus needs the same headroom as the
                    // realistic end-to-end Lambdify benchmark.
                    max_iterations: 80,
                    diagnostics: crate::numerical::Nonlinear_systems::engine::DiagnosticsOptions {
                        collect_history: false,
                        collect_statistics: true,
                        enable_logging: false,
                        ..Default::default()
                    },
                    ..Default::default()
                };
                let mut baseline: Option<DVector<f64>> = None;
                let mut expected_counts: Option<(usize, usize, usize, usize)> = None;

                for (route_name, route) in routes {
                    let mut metrics = Metrics::default();
                    let mut ok = 0;
                    for _ in 0..runs {
                        let result = route
                            .solve(initial.clone(), options.clone())
                            .expect("large Lambdify route should solve");
                        assert!(
                            matches!(result.termination, TerminationReason::Converged),
                            "case={} dimension={} route={} terminated as {:?} at residual {:.3e}",
                            corpus.name(),
                            dimension,
                            route_name,
                            result.termination,
                            result.residual_norm
                        );
                        assert!(result.residual_norm.is_finite());
                        assert!(result.x.iter().all(|value| value.is_finite()));
                        assert!(result.statistics.residual_evaluations > 0);
                        assert!(result.statistics.jacobian_evaluations > 0);
                        assert!(result.statistics.linear_solves > 0);
                        let counts = (
                            result.statistics.residual_evaluations,
                            result.statistics.jacobian_evaluations,
                            result.statistics.linear_solves,
                            result.statistics.iterations,
                        );
                        if let Some(expected) = expected_counts {
                            assert_eq!(
                                counts,
                                expected,
                                "solver-level counters differ: case={} dimension={} route={}",
                                corpus.name(),
                                dimension,
                                route_name
                            );
                        } else {
                            expected_counts = Some(counts);
                        }
                        if let Some(reference) = &baseline {
                            let difference = result
                                .x
                                .iter()
                                .zip(reference.iter())
                                .map(|(value, reference)| (value - reference).abs())
                                .fold(0.0, f64::max);
                            metrics.max_solution_diff = metrics.max_solution_diff.max(difference);
                        } else {
                            baseline = Some(result.x.clone());
                        }
                        metrics.push(&result);
                        ok += 1;
                    }
                    assert_eq!(ok, runs);
                    println!(
                        "{:<18} | {:>9} | {:<19} | {:>14} | {:>2}/{:<4} | {:<19} | {:<19} | {:<19} | {:<19} | {:>14} | {:>14} | {:>12} | {:>10} | {:>12} | {:>12} | {:>14} | {:.3e}",
                        corpus.name(),
                        dimension,
                        route_name,
                        structural_nnz(corpus, dimension),
                        ok,
                        runs,
                        mean_std_min_max(&metrics.total_ms),
                        mean_std_min_max(&metrics.residual_ms),
                        mean_std_min_max(&metrics.jacobian_ms),
                        mean_std_min_max(&metrics.linear_ms),
                        mean_count(&metrics.residual_calls),
                        mean_count(&metrics.jacobian_calls),
                        mean_count(&metrics.linear_calls),
                        mean_count(&metrics.iterations),
                        mean_us_per_call(&metrics.residual_ms, &metrics.residual_calls),
                        mean_us_per_call(&metrics.jacobian_ms, &metrics.jacobian_calls),
                        mean_us_per_call(&metrics.linear_ms, &metrics.linear_calls),
                        metrics.max_solution_diff,
                    );
                }
            }
        }
    }
}
