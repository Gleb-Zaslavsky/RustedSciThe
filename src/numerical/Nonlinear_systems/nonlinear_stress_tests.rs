//! Stress and failure-safety tests for the public nonlinear solver methods.
//!
//! These tests deliberately include difficult landscapes and scaling. A
//! method-specific non-convergence is useful evidence and is not converted
//! into a passing convergence claim. The invariant for the full method matrix
//! is stronger and more basic: no panic, no non-finite result, and only typed
//! errors when a method cannot make progress.

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use nalgebra::{DMatrix, DVector};

    use crate::numerical::Nonlinear_systems::LM_Nielsen::{
        NielsenLevenbergMarquardtMethod, NielsenLevenbergMarquardtMethodAdvanced,
    };
    use crate::numerical::Nonlinear_systems::LM_vanilla::{
        LevenbergMarquardtMethod, LevenbergMarquardtMinpack,
    };
    use crate::numerical::Nonlinear_systems::NR_damped::{
        DampedNewtonMethod, DampedNewtonMethodAdvanced,
    };
    use crate::numerical::Nonlinear_systems::engine::{
        DiagnosticsOptions, NewtonMethod, SolveOptions,
    };
    use crate::numerical::Nonlinear_systems::error::TerminationReason;
    use crate::numerical::Nonlinear_systems::prelude::NonlinearSolverMethod;
    use crate::numerical::Nonlinear_systems::problem::{
        Bounds, JacobianProvider, NonlinearProblem,
    };
    use crate::numerical::Nonlinear_systems::trust_region::PowellDoglegMethod;
    use crate::numerical::Nonlinear_systems::trust_region::TrustRegionMethod;
    use crate::numerical::Nonlinear_systems::trust_region_LM::TrustRegionLMMethod;

    const NEARLY_SINGULAR_EPS: f64 = 1.0e-10;

    #[derive(Debug, Clone, Copy)]
    enum StressCase {
        Rosenbrock,
        PowellSingular,
        CubicCoupled,
        BadlyScaled,
        NearlySingular,
    }

    impl StressCase {
        fn name(self) -> &'static str {
            match self {
                Self::Rosenbrock => "rosenbrock",
                Self::PowellSingular => "powell_singular",
                Self::CubicCoupled => "cubic_coupled",
                Self::BadlyScaled => "badly_scaled",
                Self::NearlySingular => "nearly_singular",
            }
        }

        fn dimension(self) -> usize {
            match self {
                Self::Rosenbrock | Self::CubicCoupled | Self::BadlyScaled => 2,
                Self::PowellSingular => 4,
                Self::NearlySingular => 2,
            }
        }
    }

    struct StronglyNonlinearProblem {
        case: StressCase,
    }

    impl NonlinearProblem for StronglyNonlinearProblem {
        fn dimension(&self) -> usize {
            self.case.dimension()
        }

        fn residual(
            &self,
            x: &DVector<f64>,
        ) -> Result<DVector<f64>, crate::numerical::Nonlinear_systems::error::SolveError> {
            let residual = match self.case {
                StressCase::Rosenbrock => {
                    DVector::from_vec(vec![10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]])
                }
                StressCase::PowellSingular => DVector::from_vec(vec![
                    x[0] + 10.0 * x[1],
                    5.0_f64.sqrt() * (x[2] - x[3]),
                    (x[1] - 2.0 * x[2]).powi(2),
                    10.0_f64.sqrt() * (x[0] - x[3]).powi(2),
                ]),
                StressCase::CubicCoupled => {
                    DVector::from_vec(vec![x[0].powi(3) + x[1] - 1.0, x[0] + x[1].powi(3) - 1.0])
                }
                StressCase::BadlyScaled => {
                    DVector::from_vec(vec![1.0e6 * (x[0] * x[0] - x[1]), 1.0e-6 * (x[0] - x[1])])
                }
                StressCase::NearlySingular => DVector::from_vec(vec![
                    x[0] + x[1] - 1.0,
                    (1.0 + NEARLY_SINGULAR_EPS) * (x[0] + x[1] - 1.0),
                ]),
            };
            Ok(residual)
        }
    }

    impl JacobianProvider for StronglyNonlinearProblem {
        fn jacobian(
            &self,
            x: &DVector<f64>,
        ) -> Result<DMatrix<f64>, crate::numerical::Nonlinear_systems::error::SolveError> {
            let jacobian = match self.case {
                StressCase::Rosenbrock => {
                    DMatrix::from_row_slice(2, 2, &[-20.0 * x[0], 10.0, -1.0, 0.0])
                }
                StressCase::PowellSingular => DMatrix::from_row_slice(
                    4,
                    4,
                    &[
                        1.0,
                        10.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        5.0_f64.sqrt(),
                        -5.0_f64.sqrt(),
                        0.0,
                        2.0 * (x[1] - 2.0 * x[2]),
                        -4.0 * (x[1] - 2.0 * x[2]),
                        0.0,
                        2.0 * 10.0_f64.sqrt() * (x[0] - x[3]),
                        0.0,
                        0.0,
                        -2.0 * 10.0_f64.sqrt() * (x[0] - x[3]),
                    ],
                ),
                StressCase::CubicCoupled => DMatrix::from_row_slice(
                    2,
                    2,
                    &[3.0 * x[0].powi(2), 1.0, 1.0, 3.0 * x[1].powi(2)],
                ),
                StressCase::BadlyScaled => {
                    DMatrix::from_row_slice(2, 2, &[2.0e6 * x[0], -1.0e6, 1.0e-6, -1.0e-6])
                }
                StressCase::NearlySingular => DMatrix::from_row_slice(
                    2,
                    2,
                    &[
                        1.0,
                        1.0,
                        1.0 + NEARLY_SINGULAR_EPS,
                        1.0 + NEARLY_SINGULAR_EPS,
                    ],
                ),
            };
            Ok(jacobian)
        }
    }

    fn methods() -> Vec<NonlinearSolverMethod> {
        vec![
            NonlinearSolverMethod::Newton(NewtonMethod),
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
        ]
    }

    fn initial_guess(case: StressCase) -> DVector<f64> {
        match case {
            StressCase::Rosenbrock => DVector::from_vec(vec![-1.2, 1.0]),
            StressCase::PowellSingular => DVector::from_vec(vec![3.0, -1.0, 0.0, 1.0]),
            StressCase::CubicCoupled => DVector::from_vec(vec![-1.5, 1.5]),
            StressCase::BadlyScaled => DVector::from_vec(vec![0.1, 10.0]),
            StressCase::NearlySingular => DVector::from_vec(vec![0.0, 0.0]),
        }
    }

    fn stress_options(bounds: Option<Bounds>) -> SolveOptions {
        SolveOptions {
            tolerance: 1e-6,
            max_iterations: 250,
            bounds,
            diagnostics: DiagnosticsOptions {
                collect_history: false,
                collect_statistics: true,
                ..DiagnosticsOptions::default()
            },
            ..SolveOptions::default()
        }
    }

    fn assert_finite_result(
        case: StressCase,
        method: &str,
        result: &crate::numerical::Nonlinear_systems::engine::SolveResult,
    ) {
        assert!(
            result.x.iter().all(|value| value.is_finite()),
            "{case:?}/{method} returned a non-finite solution"
        );
        assert!(
            result.residual.iter().all(|value| value.is_finite())
                && result.residual_norm.is_finite(),
            "{case:?}/{method} returned a non-finite residual"
        );
        assert!(
            result.statistics.iterations <= 250,
            "{case:?}/{method} exceeded the stress iteration limit"
        );
    }

    #[test]
    fn strongly_nonlinear_corpus_is_panic_free_and_finite_across_methods() {
        for case in [
            StressCase::Rosenbrock,
            StressCase::PowellSingular,
            StressCase::CubicCoupled,
            StressCase::BadlyScaled,
            StressCase::NearlySingular,
        ] {
            let problem = StronglyNonlinearProblem { case };
            let mut converged = 0;
            let mut nonconverged = 0;
            let mut typed_errors = 0;
            for method in methods() {
                let method_name = method.name();
                let initial_guess = initial_guess(case);
                let options = stress_options(None);
                let outcome = catch_unwind(AssertUnwindSafe(|| {
                    method.solve(&problem, initial_guess, options)
                }));
                let outcome =
                    outcome.unwrap_or_else(|_| panic!("{}/{} panicked", case.name(), method_name));
                if let Ok(result) = &outcome {
                    assert_finite_result(case, method_name, result);
                    if result.termination == TerminationReason::Converged {
                        assert!(
                            result.residual_norm <= 1.0e-6,
                            "{}/{} reported convergence with residual {}",
                            case.name(),
                            method_name,
                            result.residual_norm
                        );
                        converged += 1;
                    } else {
                        nonconverged += 1;
                    }
                } else {
                    typed_errors += 1;
                }
            }
            println!(
                "[Nonlinear stress] case={} methods={} converged={} nonconverged={} typed_errors={}",
                case.name(),
                converged + nonconverged + typed_errors,
                converged,
                nonconverged,
                typed_errors
            );
        }
    }

    #[test]
    fn remote_rosenbrock_converges_with_newton_and_damped_newton() {
        let problem = StronglyNonlinearProblem {
            case: StressCase::Rosenbrock,
        };
        let options = stress_options(Some(
            Bounds::new(vec![(-3.0, 3.0), (-2.0, 4.0)]).expect("valid stress bounds"),
        ));

        for method in [
            NonlinearSolverMethod::Newton(NewtonMethod),
            NonlinearSolverMethod::DampedNewton(DampedNewtonMethod::default()),
        ] {
            let name = method.name();
            let result = method
                .solve(
                    &problem,
                    initial_guess(StressCase::Rosenbrock),
                    options.clone(),
                )
                .unwrap_or_else(|error| panic!("{name} Rosenbrock solve failed: {error}"));
            assert_finite_result(StressCase::Rosenbrock, name, &result);
            assert_eq!(result.termination, TerminationReason::Converged);
            assert!(result.residual_norm < 1e-6, "{name} residual too large");
            assert!((result.x[0] - 1.0).abs() < 1e-5);
            assert!((result.x[1] - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn bounded_stress_matrix_never_returns_out_of_bounds_solution() {
        let problem = StronglyNonlinearProblem {
            case: StressCase::CubicCoupled,
        };
        let bounds = Bounds::new(vec![(-2.0, 2.0), (-2.0, 2.0)]).expect("valid stress bounds");
        let options = stress_options(Some(bounds.clone()));

        for method in methods() {
            let name = method.name();
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                method.solve(
                    &problem,
                    DVector::from_vec(vec![-1.5, 1.5]),
                    options.clone(),
                )
            }))
            .unwrap_or_else(|_| panic!("bounded cubic/{name} panicked"));
            if let Ok(result) = outcome {
                assert_finite_result(StressCase::CubicCoupled, name, &result);
                for (value, (lower, upper)) in result.x.iter().zip(bounds.as_slice()) {
                    assert!(*value >= *lower && *value <= *upper);
                }
            }
        }
    }

    #[derive(Default)]
    struct RunSamples {
        total_ms: Vec<f64>,
        residual_ms: Vec<f64>,
        jacobian_ms: Vec<f64>,
        linear_ms: Vec<f64>,
        residual_calls: Vec<f64>,
        jacobian_calls: Vec<f64>,
        linear_calls: Vec<f64>,
        iterations: Vec<f64>,
        rejected_steps: Vec<f64>,
    }

    impl RunSamples {
        fn push(&mut self, result: &crate::numerical::Nonlinear_systems::engine::SolveResult) {
            let stats = &result.statistics;
            self.total_ms.push(stats.total_duration.as_secs_f64() * 1e3);
            self.residual_ms
                .push(stats.residual_duration.as_secs_f64() * 1e3);
            self.jacobian_ms
                .push(stats.jacobian_duration.as_secs_f64() * 1e3);
            self.linear_ms
                .push(stats.linear_solve_duration.as_secs_f64() * 1e3);
            self.residual_calls.push(stats.residual_evaluations as f64);
            self.jacobian_calls.push(stats.jacobian_evaluations as f64);
            self.linear_calls.push(stats.linear_solves as f64);
            self.iterations.push(stats.iterations as f64);
            self.rejected_steps.push(stats.rejected_steps as f64);
        }
    }

    fn summary(values: &[f64]) -> (f64, f64, f64, f64) {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / values.len() as f64;
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (mean, variance.sqrt(), min, max)
    }

    fn print_summary(label: &str, values: &[f64]) -> String {
        if values.is_empty() {
            return "-".to_string();
        }
        let (mean, std, min, max) = summary(values);
        format!("{label}={mean:.3}+/-{std:.3}[{min:.3},{max:.3}]")
    }

    fn mean_only(values: &[f64]) -> String {
        if values.is_empty() {
            return "-".to_string();
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        format!("{mean:.3}")
    }

    #[test]
    fn strongly_nonlinear_multi_run_story_reports_stage_metrics() {
        const RUNS: usize = 3;
        let cases = [
            StressCase::Rosenbrock,
            StressCase::PowellSingular,
            StressCase::BadlyScaled,
            StressCase::NearlySingular,
        ];

        println!(
            "[Nonlinear story] runs={RUNS}; timings are ms mean+/-std[min,max]; counters are mean"
        );
        println!(
            "case | method | ok/runs | typed_errors | total_ms | residual_ms | jacobian_ms | linear_ms | residual_calls | jacobian_calls | linear_calls | iterations | rejected"
        );

        for case in cases {
            let problem = StronglyNonlinearProblem { case };
            for method_template in methods() {
                let method_name = method_template.name();
                let mut samples = RunSamples::default();
                let mut converged = 0;
                let mut typed_errors = 0;

                for _ in 0..RUNS {
                    let outcome = catch_unwind(AssertUnwindSafe(|| {
                        method_template.clone().solve(
                            &problem,
                            initial_guess(case),
                            stress_options(None),
                        )
                    }))
                    .unwrap_or_else(|_| panic!("{}/{} panicked", case.name(), method_name));

                    match outcome {
                        Ok(result) => {
                            assert_finite_result(case, method_name, &result);
                            if result.termination == TerminationReason::Converged {
                                assert!(
                                    result.residual_norm <= 1.0e-6,
                                    "{}/{} reported convergence with residual {}",
                                    case.name(),
                                    method_name,
                                    result.residual_norm
                                );
                                converged += 1;
                            }
                            samples.push(&result);
                        }
                        Err(_) => typed_errors += 1,
                    }
                }

                let successful_runs = RUNS - typed_errors;
                println!(
                    "{} | {} | {}/{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}",
                    case.name(),
                    method_name,
                    successful_runs,
                    RUNS,
                    typed_errors,
                    print_summary("total", &samples.total_ms),
                    print_summary("res", &samples.residual_ms),
                    print_summary("jac", &samples.jacobian_ms),
                    print_summary("lin", &samples.linear_ms),
                    mean_only(&samples.residual_calls),
                    mean_only(&samples.jacobian_calls),
                    mean_only(&samples.linear_calls),
                    mean_only(&samples.iterations),
                    mean_only(&samples.rejected_steps),
                );

                assert!(
                    converged <= successful_runs,
                    "{}/{} convergence count exceeded successful runs",
                    case.name(),
                    method_name
                );
            }
        }
    }
}
