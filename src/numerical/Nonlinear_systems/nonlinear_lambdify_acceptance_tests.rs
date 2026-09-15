//! Acceptance tests for the production-facing prepared Lambdify lifecycle.
//!
//! These tests deliberately separate correctness contracts from performance
//! claims. The ignored story prints preparation, bind, solve, and callback
//! metrics; it must be run in release mode before using the numbers as
//! evidence. The ordinary tests are small regression gates and run normally.
//!
//! Debug correctness tests:
//! ```text
//! cargo test --lib numerical::Nonlinear_systems::nonlinear_lambdify_acceptance_tests -- --nocapture --test-threads=1
//! ```
//!
//! Release acceptance tests and story:
//! ```text
//! cargo test --release --lib numerical::Nonlinear_systems::nonlinear_lambdify_acceptance_tests -- --nocapture --test-threads=1
//! cargo test --release --lib numerical::Nonlinear_systems::nonlinear_lambdify_acceptance_tests -- --nocapture --ignored --test-threads=1
//! ```

#[cfg(test)]
mod tests {
    use super::super::engine::{DiagnosticsOptions, SolveOptions, StatisticsAvailability};
    use super::super::error::TerminationReason;
    use super::super::prelude::{DampedNewtonMethod, NonlinearSolverMethod};
    use super::super::problem::{JacobianProvider, NonlinearProblem};
    use super::super::symbolic::{
        LambdifyExecutionPolicy, PreparedSymbolicNonlinearProblem, SymbolicProblemOptions,
    };
    use nalgebra::{DMatrix, DVector};
    use std::time::Instant;

    fn equations() -> Vec<String> {
        vec!["a*x^2-4".to_string(), "y-x".to_string()]
    }

    fn options(policy: LambdifyExecutionPolicy) -> SymbolicProblemOptions {
        SymbolicProblemOptions::new()
            .with_variables(vec!["x".to_string(), "y".to_string()])
            .with_equation_parameters(vec!["a".to_string()])
            .with_lambdify_execution_policy(policy)
    }

    fn solve_options(collect_statistics: bool) -> SolveOptions {
        SolveOptions {
            tolerance: 1e-11,
            max_iterations: 50,
            diagnostics: DiagnosticsOptions {
                collect_history: false,
                collect_statistics,
                enable_logging: false,
                ..DiagnosticsOptions::default()
            },
            ..SolveOptions::default()
        }
    }

    fn expected_solution(a: f64) -> DVector<f64> {
        let x = 2.0 / a.sqrt();
        DVector::from_vec(vec![x, x])
    }

    fn solve_bound(
        problem: &super::super::symbolic::BoundSymbolicNonlinearProblem<'_>,
        initial: DVector<f64>,
        collect_statistics: bool,
    ) -> super::super::engine::SolveResult {
        NonlinearSolverMethod::DampedNewton(DampedNewtonMethod::default())
            .solve(problem, initial, solve_options(collect_statistics))
            .expect("prepared Lambdify solve should succeed")
    }

    #[test]
    fn prepared_lambdify_sequential_and_parallel_solver_contracts_match() {
        let sequential = PreparedSymbolicNonlinearProblem::from_strings(
            equations(),
            options(LambdifyExecutionPolicy::Sequential),
        )
        .expect("sequential preparation should succeed");
        let parallel = PreparedSymbolicNonlinearProblem::from_strings(
            equations(),
            options(LambdifyExecutionPolicy::Parallel { min_work: 1 }),
        )
        .expect("parallel preparation should succeed");
        let sequential = sequential
            .bind_values(DVector::from_vec(vec![1.0]))
            .expect("sequential binding should succeed");
        let parallel = parallel
            .bind_values(DVector::from_vec(vec![1.0]))
            .expect("parallel binding should succeed");

        let sequential_result = solve_bound(&sequential, DVector::from_vec(vec![1.5, 1.5]), true);
        let parallel_result = solve_bound(&parallel, DVector::from_vec(vec![1.5, 1.5]), true);

        assert!(matches!(
            sequential_result.termination,
            TerminationReason::Converged
        ));
        assert!(matches!(
            parallel_result.termination,
            TerminationReason::Converged
        ));
        assert!((sequential_result.x - expected_solution(1.0)).norm() < 1e-9);
        assert!((parallel_result.x - expected_solution(1.0)).norm() < 1e-9);

        let sequential_stats = sequential_result.statistics;
        let parallel_stats = parallel_result.statistics;
        assert_eq!(
            sequential_stats.availability,
            StatisticsAvailability::Collected
        );
        assert_eq!(
            parallel_stats.availability,
            StatisticsAvailability::Collected
        );
        assert_eq!(
            (
                sequential_stats.iterations,
                sequential_stats.residual_evaluations,
                sequential_stats.jacobian_evaluations,
                sequential_stats.linear_solves,
                sequential_stats.accepted_steps,
                sequential_stats.rejected_steps,
            ),
            (
                parallel_stats.iterations,
                parallel_stats.residual_evaluations,
                parallel_stats.jacobian_evaluations,
                parallel_stats.linear_solves,
                parallel_stats.accepted_steps,
                parallel_stats.rejected_steps,
            ),
            "execution policy must not change solver-level work"
        );
    }

    #[test]
    fn prepared_lambdify_telemetry_off_is_explicit_and_solution_equivalent() {
        let prepared = PreparedSymbolicNonlinearProblem::from_strings(
            equations(),
            options(LambdifyExecutionPolicy::Parallel { min_work: 1 }),
        )
        .expect("preparation should succeed");
        let bound = prepared
            .bind_values(DVector::from_vec(vec![2.0]))
            .expect("binding should succeed");

        let measured = solve_bound(&bound, DVector::from_vec(vec![1.0, 1.0]), true);
        let unmeasured = solve_bound(&bound, DVector::from_vec(vec![1.0, 1.0]), false);

        assert!(matches!(measured.termination, TerminationReason::Converged));
        assert!(matches!(
            unmeasured.termination,
            TerminationReason::Converged
        ));
        assert!((measured.x.clone() - unmeasured.x).norm() < 1e-12);
        assert_eq!(
            measured.statistics.availability,
            StatisticsAvailability::Collected
        );
        assert_eq!(
            unmeasured.statistics.availability,
            StatisticsAvailability::NotCollected
        );
        assert_eq!(unmeasured.statistics.residual_evaluations, 0);
        assert_eq!(unmeasured.statistics.jacobian_evaluations, 0);
        assert_eq!(unmeasured.statistics.linear_solves, 0);
        assert_eq!(
            unmeasured.statistics.residual_duration,
            std::time::Duration::ZERO
        );
        assert_eq!(
            unmeasured.statistics.jacobian_duration,
            std::time::Duration::ZERO
        );
        assert_eq!(
            unmeasured.statistics.linear_solve_duration,
            std::time::Duration::ZERO
        );
        assert_eq!(
            unmeasured.statistics.total_duration,
            std::time::Duration::ZERO
        );
    }

    #[test]
    fn prepared_lambdify_many_parameters_preserves_order_and_sparse_layout() {
        const DIMENSION: usize = 96;
        let variables = (0..DIMENSION)
            .map(|index| format!("x{index}"))
            .collect::<Vec<_>>();
        let parameters = (0..DIMENSION)
            .map(|index| format!("p{index}"))
            .collect::<Vec<_>>();
        let equations = (0..DIMENSION)
            .map(|index| format!("p{index}*x{index}-p{index}"))
            .collect::<Vec<_>>();
        let prepared = PreparedSymbolicNonlinearProblem::from_strings(
            equations,
            SymbolicProblemOptions::new()
                .with_variables(variables)
                .with_equation_parameters(parameters.clone())
                .with_lambdify_execution_policy(LambdifyExecutionPolicy::Parallel { min_work: 1 }),
        )
        .expect("many-parameter Lambdify preparation should succeed");

        let schema = prepared
            .parameter_schema()
            .expect("parameter schema should be present");
        assert_eq!(schema.names(), &parameters);

        let point = DVector::from_iterator(
            DIMENSION,
            (0..DIMENSION).map(|index| 0.25 + index as f64 * 0.001),
        );
        let first_values = DVector::from_iterator(
            DIMENSION,
            (0..DIMENSION).map(|index| 1.0 + index as f64 * 0.01),
        );
        let second_values = DVector::from_iterator(
            DIMENSION,
            (0..DIMENSION).map(|index| 2.0 + index as f64 * 0.02),
        );
        let first = prepared
            .bind_values(first_values.clone())
            .expect("first many-parameter binding should succeed");
        let second = prepared
            .bind_values(second_values.clone())
            .expect("second many-parameter binding should succeed");

        let mut residual = DVector::zeros(DIMENSION);
        let mut jacobian = DMatrix::zeros(DIMENSION, DIMENSION);
        first
            .residual_into(&point, &mut residual)
            .expect("first many-parameter residual should succeed");
        first
            .jacobian_into(&point, &mut jacobian)
            .expect("first many-parameter Jacobian should succeed");
        for index in 0..DIMENSION {
            assert!(
                (residual[index] - first_values[index] * (point[index] - 1.0)).abs() < 1e-13,
                "first residual mismatch at index {index}"
            );
            for column in 0..DIMENSION {
                let expected = if index == column {
                    first_values[index]
                } else {
                    0.0
                };
                assert_eq!(jacobian[(index, column)], expected);
            }
        }

        second
            .residual_into(&point, &mut residual)
            .expect("second many-parameter residual should succeed");
        second
            .jacobian_into(&point, &mut jacobian)
            .expect("second many-parameter Jacobian should succeed");
        for index in 0..DIMENSION {
            assert!(
                (residual[index] - second_values[index] * (point[index] - 1.0)).abs() < 1e-13,
                "second residual mismatch at index {index}"
            );
            assert_eq!(jacobian[(index, index)], second_values[index]);
        }
    }

    #[test]
    #[ignore = "release-oriented high-cardinality Lambdify callback story"]
    fn high_cardinality_parameter_callback_story() {
        let runs = std::env::var("NONLINEAR_LAMBDIFY_PARAMETER_RUNS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|runs| *runs >= 3)
            .unwrap_or(5);

        fn summarize(values: &[f64]) -> String {
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

        println!(
            "[Nonlinear Lambdify parameter hot path] runs={runs}; times are microseconds mean+/-std[min,max]"
        );
        println!(
            "parameter_count | seq_residual_us | seq_jacobian_us | parallel_residual_us | parallel_jacobian_us"
        );

        for dimension in [8, 64, 256] {
            let variables = (0..dimension)
                .map(|index| format!("x{index}"))
                .collect::<Vec<_>>();
            let parameters = (0..dimension)
                .map(|index| format!("p{index}"))
                .collect::<Vec<_>>();
            let equations = (0..dimension)
                .map(|index| format!("p{index}*x{index}-p{index}"))
                .collect::<Vec<_>>();
            let prepare = |policy| {
                PreparedSymbolicNonlinearProblem::from_strings(
                    equations.clone(),
                    SymbolicProblemOptions::new()
                        .with_variables(variables.clone())
                        .with_equation_parameters(parameters.clone())
                        .with_lambdify_execution_policy(policy),
                )
                .expect("parameter hot-path preparation should succeed")
            };
            let prepared_sequential = prepare(LambdifyExecutionPolicy::Sequential);
            let sequential = prepared_sequential
                .bind_values(DVector::from_iterator(
                    dimension,
                    (0..dimension).map(|index| 1.0 + index as f64 * 0.01),
                ))
                .expect("sequential hot-path binding should succeed");
            let prepared_parallel = prepare(LambdifyExecutionPolicy::Parallel { min_work: 1 });
            let parallel = prepared_parallel
                .bind_values(DVector::from_iterator(
                    dimension,
                    (0..dimension).map(|index| 1.0 + index as f64 * 0.01),
                ))
                .expect("parallel hot-path binding should succeed");
            let point = DVector::from_iterator(
                dimension,
                (0..dimension).map(|index| 0.25 + index as f64 * 0.001),
            );
            let mut sequential_residual = DVector::zeros(dimension);
            let mut sequential_jacobian = DMatrix::zeros(dimension, dimension);
            let mut parallel_residual = DVector::zeros(dimension);
            let mut parallel_jacobian = DMatrix::zeros(dimension, dimension);

            sequential
                .residual_into(&point, &mut sequential_residual)
                .expect("sequential warm residual should succeed");
            sequential
                .jacobian_into(&point, &mut sequential_jacobian)
                .expect("sequential warm Jacobian should succeed");
            parallel
                .residual_into(&point, &mut parallel_residual)
                .expect("parallel warm residual should succeed");
            parallel
                .jacobian_into(&point, &mut parallel_jacobian)
                .expect("parallel warm Jacobian should succeed");
            assert_eq!(sequential_residual, parallel_residual);
            assert_eq!(sequential_jacobian, parallel_jacobian);

            let mut seq_residual_us = Vec::with_capacity(runs);
            let mut seq_jacobian_us = Vec::with_capacity(runs);
            let mut parallel_residual_us = Vec::with_capacity(runs);
            let mut parallel_jacobian_us = Vec::with_capacity(runs);
            for _ in 0..runs {
                let started = Instant::now();
                sequential
                    .residual_into(&point, &mut sequential_residual)
                    .expect("sequential residual should succeed");
                seq_residual_us.push(started.elapsed().as_secs_f64() * 1e6);

                let started = Instant::now();
                sequential
                    .jacobian_into(&point, &mut sequential_jacobian)
                    .expect("sequential Jacobian should succeed");
                seq_jacobian_us.push(started.elapsed().as_secs_f64() * 1e6);

                let started = Instant::now();
                parallel
                    .residual_into(&point, &mut parallel_residual)
                    .expect("parallel residual should succeed");
                parallel_residual_us.push(started.elapsed().as_secs_f64() * 1e6);

                let started = Instant::now();
                parallel
                    .jacobian_into(&point, &mut parallel_jacobian)
                    .expect("parallel Jacobian should succeed");
                parallel_jacobian_us.push(started.elapsed().as_secs_f64() * 1e6);
            }
            assert_eq!(sequential_residual, parallel_residual);
            assert_eq!(sequential_jacobian, parallel_jacobian);
            println!(
                "{:>15} | {:<16} | {:<16} | {:<21} | {}",
                dimension,
                summarize(&seq_residual_us),
                summarize(&seq_jacobian_us),
                summarize(&parallel_residual_us),
                summarize(&parallel_jacobian_us),
            );
        }
    }

    #[test]
    #[ignore = "release-oriented prepared-vs-rebuild Lambdify lifecycle story"]
    fn prepared_lambdify_reuse_vs_rebuild_parameter_story() {
        const PARAMETER_VALUES: [f64; 5] = [0.5, 0.75, 1.0, 1.5, 2.0];
        let prepared = PreparedSymbolicNonlinearProblem::from_strings(
            equations(),
            options(LambdifyExecutionPolicy::Sequential),
        )
        .expect("prepared Lambdify problem should succeed");

        let mut reuse_prepare_ms = Vec::new();
        let mut reuse_solve_ms = Vec::new();
        let mut rebuild_prepare_ms = Vec::new();
        let mut rebuild_solve_ms = Vec::new();
        let mut reuse_errors = Vec::new();
        let mut rebuild_errors = Vec::new();

        for a in PARAMETER_VALUES {
            let bind_started = Instant::now();
            let bound = prepared
                .bind_values(DVector::from_vec(vec![a]))
                .expect("reuse binding should succeed");
            reuse_prepare_ms.push(bind_started.elapsed().as_secs_f64() * 1e3);
            let solve_started = Instant::now();
            let reused = solve_bound(&bound, DVector::from_vec(vec![1.0, 1.0]), true);
            reuse_solve_ms.push(solve_started.elapsed().as_secs_f64() * 1e3);
            reuse_errors.push((reused.x - expected_solution(a)).norm());

            let rebuild_started = Instant::now();
            let rebuilt = PreparedSymbolicNonlinearProblem::from_strings(
                equations(),
                options(LambdifyExecutionPolicy::Sequential),
            )
            .expect("rebuild preparation should succeed");
            let rebuilt = rebuilt
                .bind_values(DVector::from_vec(vec![a]))
                .expect("rebuild binding should succeed");
            rebuild_prepare_ms.push(rebuild_started.elapsed().as_secs_f64() * 1e3);
            let solve_started = Instant::now();
            let rebuilt_result = solve_bound(&rebuilt, DVector::from_vec(vec![1.0, 1.0]), true);
            rebuild_solve_ms.push(solve_started.elapsed().as_secs_f64() * 1e3);
            rebuild_errors.push((rebuilt_result.x - expected_solution(a)).norm());
        }

        assert!(reuse_errors.iter().all(|error| *error < 1e-9));
        assert!(rebuild_errors.iter().all(|error| *error < 1e-9));
        assert!(reuse_prepare_ms.iter().all(|value| value.is_finite()));
        assert!(rebuild_prepare_ms.iter().all(|value| value.is_finite()));

        fn mean(values: &[f64]) -> f64 {
            values.iter().sum::<f64>() / values.len() as f64
        }

        println!(
            "[Nonlinear Lambdify lifecycle] parameter_values={PARAMETER_VALUES:?}; prepared object is reused for every bind"
        );
        println!("path | prepare_or_bind_ms_mean | solve_ms_mean | max_solution_error");
        println!(
            "prepared-reuse | {:.3} | {:.3} | {:.3e}",
            mean(&reuse_prepare_ms),
            mean(&reuse_solve_ms),
            reuse_errors.iter().copied().fold(0.0, f64::max),
        );
        println!(
            "rebuild-each-time | {:.3} | {:.3} | {:.3e}",
            mean(&rebuild_prepare_ms),
            mean(&rebuild_solve_ms),
            rebuild_errors.iter().copied().fold(0.0, f64::max),
        );
    }
}
