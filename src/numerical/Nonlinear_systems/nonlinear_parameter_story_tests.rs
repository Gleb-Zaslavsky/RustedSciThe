//! Parameter-sweep story for prepared nonlinear symbolic backends.
//!
//! Preparation and generated-artifact work are intentionally outside the warm
//! parameter loop. The story therefore answers two separate questions:
//! whether parameter-only binding preserves correctness, and whether a prepared
//! Lambdify/AOT backend can be reused without repeating symbolic preparation.

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::time::Instant;

    use approx::assert_relative_eq;
    use nalgebra::DVector;

    use crate::numerical::Nonlinear_systems::engine::{
        DiagnosticsOptions, NewtonMethod, SolveOptions,
    };
    use crate::numerical::Nonlinear_systems::error::SolveError;
    use crate::numerical::Nonlinear_systems::prelude::{
        DampedNewtonMethod, DampedNewtonMethodAdvanced, LevenbergMarquardtMethod,
        LevenbergMarquardtMinpack, NielsenLevenbergMarquardtMethod,
        NielsenLevenbergMarquardtMethodAdvanced, NonlinearSolverMethod, PowellDoglegMethod,
        PreparedSymbolicNonlinearProblem, SymbolicNonlinearProblem, SymbolicProblemOptions,
        TrustRegionLMMethod, TrustRegionMethod,
    };
    use crate::numerical::Nonlinear_systems::symbolic_aot_test_support::{
        aot_solver_test_guard, register_parameterized_dense_backend,
    };
    use crate::numerical::Nonlinear_systems::symbolic_generated::{
        SymbolicAotBuildPolicy, SymbolicGeneratedBackendConfig,
    };
    use crate::symbolic::codegen::codegen_aot_runtime_link::unregister_linked_dense_backend;
    use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;

    const RUNS: usize = 5;

    fn equations() -> Vec<String> {
        vec!["a*x+y-3".to_string(), "x-y".to_string()]
    }

    fn problem_options() -> SymbolicProblemOptions {
        SymbolicProblemOptions::new()
            .with_variables(vec!["x".to_string(), "y".to_string()])
            .with_equation_parameters(vec!["a".to_string()])
    }

    fn solve_options() -> SolveOptions {
        SolveOptions {
            tolerance: 1e-10,
            max_iterations: 32,
            diagnostics: DiagnosticsOptions {
                collect_history: false,
                collect_statistics: true,
                ..DiagnosticsOptions::default()
            },
            ..SolveOptions::default()
        }
    }

    fn parameter_sweep() -> [(f64, f64); RUNS] {
        [
            (1.0, 1.5),
            (2.0, 1.0),
            (3.0, 0.75),
            (4.0, 0.6),
            (8.0, 1.0 / 3.0),
        ]
    }

    #[derive(Default)]
    struct SweepSummary {
        bind_solve_ms: Vec<f64>,
        residual_ms: Vec<f64>,
        jacobian_ms: Vec<f64>,
        linear_ms: Vec<f64>,
        final_errors: Vec<f64>,
    }

    fn run_sweep(prepared: &PreparedSymbolicNonlinearProblem) -> SweepSummary {
        let mut summary = SweepSummary::default();
        for (parameter, expected) in parameter_sweep() {
            let started = Instant::now();
            let bound = prepared
                .bind_values(DVector::from_vec(vec![parameter]))
                .expect("parameter binding should be valid");
            let result = NonlinearSolverMethod::Newton(NewtonMethod)
                .solve(&bound, DVector::from_vec(vec![0.5, 0.5]), solve_options())
                .expect("parameter sweep solve should succeed");
            summary
                .bind_solve_ms
                .push(started.elapsed().as_secs_f64() * 1e3);
            summary
                .residual_ms
                .push(result.statistics.residual_duration.as_secs_f64() * 1e3);
            summary
                .jacobian_ms
                .push(result.statistics.jacobian_duration.as_secs_f64() * 1e3);
            summary
                .linear_ms
                .push(result.statistics.linear_solve_duration.as_secs_f64() * 1e3);
            let expected_error =
                ((result.x[0] - expected).powi(2) + (result.x[1] - expected).powi(2)).sqrt();
            summary.final_errors.push(expected_error);
            assert_relative_eq!(result.x[0], expected, epsilon = 1e-9);
            assert_relative_eq!(result.x[1], expected, epsilon = 1e-9);
        }
        summary
    }

    fn mean(values: &[f64]) -> f64 {
        values.iter().sum::<f64>() / values.len() as f64
    }

    fn format_metric(values: &[f64]) -> String {
        let average = mean(values);
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        format!("{average:.3}[{min:.3},{max:.3}]")
    }

    fn format_metric_with_std(values: &[f64]) -> String {
        if values.is_empty() {
            return "-".to_string();
        }
        let average = mean(values);
        let variance = values
            .iter()
            .map(|value| (value - average).powi(2))
            .sum::<f64>()
            / values.len() as f64;
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        format!("{average:.3}+/-{:.3}[{min:.3},{max:.3}]", variance.sqrt())
    }

    fn invalid_parameter_updates_are_atomic() -> usize {
        let mut problem = SymbolicNonlinearProblem::from_strings_with_options(
            equations(),
            problem_options().with_equation_parameter_values(DVector::from_vec(vec![2.0])),
        )
        .expect("parameterized problem should be valid");
        let key_before = problem
            .prepare_dense_aot_problem(Default::default())
            .problem_key();
        let jacobian_before = problem.symbolic_jacobian().to_vec();

        let invalid_updates = [
            DVector::from_vec(vec![]),
            DVector::from_vec(vec![f64::NAN]),
            DVector::from_vec(vec![f64::INFINITY]),
        ];
        let invalid_update_count = invalid_updates.len();
        for candidate in invalid_updates {
            let error = problem
                .set_parameter_values(candidate)
                .expect_err("invalid parameter update must be rejected");
            assert!(matches!(
                error,
                SolveError::DimensionMismatch {
                    context: "nonlinear parameter values",
                    ..
                } | SolveError::NonFiniteParameterValue { .. }
            ));
            assert_eq!(
                problem.parameter_values().expect("old values").as_slice(),
                &[2.0]
            );
        }

        problem
            .set_parameter_values(DVector::from_vec(vec![4.0]))
            .expect("valid parameter update should succeed");
        assert_eq!(
            problem.parameter_values().expect("new values").as_slice(),
            &[4.0]
        );
        assert_eq!(
            problem
                .prepare_dense_aot_problem(Default::default())
                .problem_key(),
            key_before
        );
        assert_eq!(problem.symbolic_jacobian(), jacobian_before.as_slice());
        invalid_update_count
    }

    #[test]
    fn parameter_sweep_rejects_invalid_updates_without_mutating_prepared_state() {
        assert_eq!(invalid_parameter_updates_are_atomic(), 3);
    }

    #[test]
    fn lambdify_realistic_coupled_corpus_reports_uniform_stage_metrics() {
        const DIMENSION: usize = 16;
        const RUNS: usize = 3;
        let target = (0..DIMENSION)
            .map(|index| 1.0 + index as f64 * 0.015)
            .collect::<Vec<_>>();
        let equations = (0..DIMENSION)
            .map(|index| {
                let variable = format!("x{index}");
                let mut equation =
                    format!("a*({variable}^2-{:.17})", target[index] * target[index]);
                if index > 0 {
                    equation.push_str(&format!("+0.04*(x{}-{:.17})", index - 1, target[index - 1]));
                }
                if index + 1 < DIMENSION {
                    equation.push_str(&format!("+0.04*(x{}-{:.17})", index + 1, target[index + 1]));
                }
                equation
            })
            .collect::<Vec<_>>();
        let variables = (0..DIMENSION)
            .map(|index| format!("x{index}"))
            .collect::<Vec<_>>();
        let prepared = PreparedSymbolicNonlinearProblem::from_strings(
            equations,
            SymbolicProblemOptions::new()
                .with_variables(variables)
                .with_equation_parameters(vec!["a".to_string()])
                .with_lambdify_backend(),
        )
        .expect("realistic Lambdify corpus should prepare");
        let methods = vec![
            NonlinearSolverMethod::Newton(
                crate::numerical::Nonlinear_systems::engine::NewtonMethod,
            ),
            NonlinearSolverMethod::DampedNewton(DampedNewtonMethod::default()),
            NonlinearSolverMethod::DampedNewtonAdvanced(DampedNewtonMethodAdvanced::default()),
            NonlinearSolverMethod::LevenbergMarquardt(LevenbergMarquardtMethod::default()),
            NonlinearSolverMethod::LevenbergMarquardtMinpack(LevenbergMarquardtMinpack::default()),
            NonlinearSolverMethod::NielsenLevenbergMarquardt(
                NielsenLevenbergMarquardtMethod::default(),
            ),
            NonlinearSolverMethod::NielsenLevenbergMarquardtAdvanced(
                NielsenLevenbergMarquardtMethodAdvanced::default(),
            ),
            NonlinearSolverMethod::TrustRegion(TrustRegionMethod::default()),
            NonlinearSolverMethod::PowellDogleg(PowellDoglegMethod::default()),
            NonlinearSolverMethod::TrustRegionLM(TrustRegionLMMethod::default()),
        ];
        let options = SolveOptions {
            tolerance: 1e-10,
            max_iterations: 80,
            bounds: Some(
                crate::numerical::Nonlinear_systems::problem::Bounds::new(vec![
                    (0.0, 2.0);
                    DIMENSION
                ])
                .expect("realistic corpus bounds should validate"),
            ),
            diagnostics: DiagnosticsOptions {
                collect_history: false,
                collect_statistics: true,
                ..DiagnosticsOptions::default()
            },
            ..SolveOptions::default()
        };

        println!(
            "[Nonlinear Lambdify story] dimension={DIMENSION}, parameter_values=0.5/1.0/2.0, runs={RUNS}; timings are ms mean+/-std[min,max]"
        );
        println!(
            "method | cases | converged | nonconverged | typed_errors | panics | total_ms | residual_ms | jacobian_ms | linear_ms | residual_calls | jacobian_calls | linear_calls | rejected"
        );

        for method_template in methods {
            let method_name = method_template.name();
            let mut total_ms = Vec::new();
            let mut residual_ms = Vec::new();
            let mut jacobian_ms = Vec::new();
            let mut linear_ms = Vec::new();
            let mut residual_calls = Vec::new();
            let mut jacobian_calls = Vec::new();
            let mut linear_calls = Vec::new();
            let mut rejected_steps = Vec::new();
            let mut converged = 0;
            let mut nonconverged = 0;
            let mut typed_errors = 0;
            let mut panics = 0;

            for parameter in [0.5, 1.0, 2.0] {
                let bound = prepared
                    .bind_values(DVector::from_vec(vec![parameter]))
                    .expect("realistic parameter binding should validate");
                for _ in 0..RUNS {
                    let initial = DVector::from_element(DIMENSION, 0.8);
                    let outcome = catch_unwind(AssertUnwindSafe(|| {
                        method_template
                            .clone()
                            .solve(&bound, initial, options.clone())
                    }));
                    let result = match outcome {
                        Ok(Ok(result)) => result,
                        Ok(Err(error)) => {
                            typed_errors += 1;
                            println!(
                                "[Nonlinear Lambdify story] method={method_name} typed_error={error}"
                            );
                            continue;
                        }
                        Err(_) => {
                            panics += 1;
                            continue;
                        }
                    };
                    assert!(
                        result.residual_norm.is_finite()
                            && result.x.iter().all(|value| value.is_finite())
                            && result.residual.iter().all(|value| value.is_finite()),
                        "{method_name} produced a non-finite realistic Lambdify result: residual={} termination={:?}",
                        result.residual_norm,
                        result.termination
                    );
                    if matches!(
                        result.termination,
                        crate::numerical::Nonlinear_systems::error::TerminationReason::Converged
                    ) {
                        assert!(
                            result.residual_norm <= 1e-8,
                            "{method_name} reported convergence with an inaccurate solution: residual={}",
                            result.residual_norm
                        );
                        for (value, expected) in result.x.iter().zip(&target) {
                            assert!((value - expected).abs() <= 1e-7);
                        }
                        converged += 1;
                    } else {
                        nonconverged += 1;
                    }
                    let statistics = result.statistics;
                    total_ms.push(statistics.total_duration.as_secs_f64() * 1e3);
                    residual_ms.push(statistics.residual_duration.as_secs_f64() * 1e3);
                    jacobian_ms.push(statistics.jacobian_duration.as_secs_f64() * 1e3);
                    linear_ms.push(statistics.linear_solve_duration.as_secs_f64() * 1e3);
                    residual_calls.push(statistics.residual_evaluations as f64);
                    jacobian_calls.push(statistics.jacobian_evaluations as f64);
                    linear_calls.push(statistics.linear_solves as f64);
                    rejected_steps.push(statistics.rejected_steps as f64);
                }
            }

            assert_eq!(panics, 0, "{method_name} panicked during the story");
            println!(
                "{method_name} | {} | {converged} | {nonconverged} | {typed_errors} | {panics} | {} | {} | {} | {} | {} | {} | {} | {}",
                3 * RUNS,
                format_metric_with_std(&total_ms),
                format_metric_with_std(&residual_ms),
                format_metric_with_std(&jacobian_ms),
                format_metric_with_std(&linear_ms),
                format_metric(&residual_calls),
                format_metric(&jacobian_calls),
                format_metric(&linear_calls),
                format_metric(&rejected_steps),
            );
        }
    }

    #[test]
    #[ignore = "builds a generated nonlinear AOT artifact and measures warm parameter reuse"]
    fn parameter_sweep_lambdify_vs_generated_aot_reuses_prepared_backend() {
        let _guard = aot_solver_test_guard();

        let invalid_updates = invalid_parameter_updates_are_atomic();

        let lambdify_started = Instant::now();
        let lambdify = PreparedSymbolicNonlinearProblem::from_strings(
            equations(),
            problem_options().with_lambdify_backend(),
        )
        .expect("Lambdify preparation should succeed");
        let lambdify_prepare_ms = lambdify_started.elapsed().as_secs_f64() * 1e3;
        let lambdify_summary = run_sweep(&lambdify);

        let output_dir = tempfile::tempdir().expect("AOT output directory should exist");
        let aot_started = Instant::now();
        let built = crate::numerical::Nonlinear_systems::symbolic::SymbolicNonlinearProblem::from_strings_with_generated_backend(
            equations(),
            problem_options().with_lambdify_backend(),
            SymbolicGeneratedBackendConfig::defaults()
                .with_build_policy(SymbolicAotBuildPolicy::BuildIfMissing {
                    profile: AotBuildProfile::Release,
                })
                .with_output_parent_dir(Some(output_dir.path().to_path_buf())),
        )
        .expect("generated AOT build should succeed");
        let aot_build_ms = aot_started.elapsed().as_secs_f64() * 1e3;
        assert!(
            built.build_result.is_some(),
            "cold AOT build should be recorded"
        );
        let resolver = built
            .updated_resolver
            .clone()
            .expect("AOT build should return a resolver");
        let problem_key = resolver
            .registry()
            .problem_keys()
            .into_iter()
            .next()
            .expect("AOT resolver should contain the parameterized problem");
        register_parameterized_dense_backend(&problem_key);

        let reuse_started = Instant::now();
        let aot = crate::numerical::Nonlinear_systems::symbolic::SymbolicNonlinearProblem::from_strings_with_generated_backend(
            equations(),
            problem_options().with_lambdify_backend(),
            SymbolicGeneratedBackendConfig::require_prebuilt().with_resolver(Some(resolver)),
        )
        .expect("RequirePrebuilt should reuse the generated backend");
        let aot_reuse_ms = reuse_started.elapsed().as_secs_f64() * 1e3;
        assert_eq!(
            aot.selected_backend,
            crate::numerical::Nonlinear_systems::symbolic_backend::
                SelectedSymbolicNonlinearBackendKind::AotCompiled
        );
        assert!(
            aot.build_result.is_none(),
            "warm AOT reuse must not rebuild"
        );
        let aot_summary = run_sweep(&aot.into_prepared());

        println!(
            "[Nonlinear parameter sweep] runs={RUNS}; invalid_updates={invalid_updates}; warm metrics are mean[min,max] milliseconds"
        );
        println!(
            "backend | prepare_or_build_ms | require_prebuilt_ms | bind_solve_ms | residual_ms | jacobian_ms | linear_ms | max_solution_error"
        );
        println!(
            "Lambdify | {lambdify_prepare_ms:.3} | - | {} | {} | {} | {} | {:.3e}",
            format_metric(&lambdify_summary.bind_solve_ms),
            format_metric(&lambdify_summary.residual_ms),
            format_metric(&lambdify_summary.jacobian_ms),
            format_metric(&lambdify_summary.linear_ms),
            lambdify_summary
                .final_errors
                .iter()
                .copied()
                .fold(0.0, f64::max),
        );
        println!(
            "AOT | {aot_build_ms:.3} | {aot_reuse_ms:.3} | {} | {} | {} | {} | {:.3e}",
            format_metric(&aot_summary.bind_solve_ms),
            format_metric(&aot_summary.residual_ms),
            format_metric(&aot_summary.jacobian_ms),
            format_metric(&aot_summary.linear_ms),
            aot_summary.final_errors.iter().copied().fold(0.0, f64::max),
        );

        unregister_linked_dense_backend(&problem_key);
    }
}
