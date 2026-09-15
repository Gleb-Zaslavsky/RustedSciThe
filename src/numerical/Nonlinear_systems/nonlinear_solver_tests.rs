#[cfg(test)]
mod tests {

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
        DiagnosticsOptions, NewtonMethod, SolveOptions, SolverEngine,
    };
    use crate::numerical::Nonlinear_systems::prelude::NonlinearSolverMethod;
    use crate::numerical::Nonlinear_systems::problem::Bounds;
    use crate::numerical::Nonlinear_systems::symbolic::{
        PreparedSymbolicNonlinearProblem, SymbolicNonlinearProblem, SymbolicProblemOptions,
    };
    use crate::numerical::Nonlinear_systems::trust_region::PowellDoglegMethod;
    use crate::numerical::Nonlinear_systems::trust_region::TrustRegionMethod;
    use crate::numerical::Nonlinear_systems::trust_region_LM::TrustRegionLMMethod;
    use nalgebra::{DMatrix, DVector};

    use crate::numerical::Nonlinear_systems::problem::{JacobianProvider, NonlinearProblem};
    use approx::assert_relative_eq;

    struct PlainBackendProblem;

    impl NonlinearProblem for PlainBackendProblem {
        fn dimension(&self) -> usize {
            2
        }
        fn residual(
            &self,
            x: &DVector<f64>,
        ) -> Result<DVector<f64>, crate::numerical::Nonlinear_systems::error::SolveError> {
            Ok(DVector::from_vec(vec![
                x[0] * x[0] + x[1] * x[1] - 1.0,
                x[0] - x[1],
            ]))
        }
    }

    impl JacobianProvider for PlainBackendProblem {
        fn jacobian(
            &self,
            x: &DVector<f64>,
        ) -> Result<DMatrix<f64>, crate::numerical::Nonlinear_systems::error::SolveError> {
            Ok(DMatrix::from_row_slice(
                2,
                2,
                &[2.0 * x[0], 2.0 * x[1], 1.0, -1.0],
            ))
        }
    }

    fn symbolic_problem() -> SymbolicNonlinearProblem {
        SymbolicNonlinearProblem::from_strings(
            vec!["x^2+y^2-10".to_string(), "x-y-4".to_string()],
            Some(vec!["x".to_string(), "y".to_string()]),
            None,
            None,
        )
        .expect("symbolic problem")
    }

    #[test]
    fn symbolic_backend_solves_with_engine_and_collects_diagnostics() {
        let options = SolveOptions {
            diagnostics: DiagnosticsOptions {
                enable_memory_diagnostics: true,
                ..DiagnosticsOptions::default()
            },
            ..SolveOptions::default()
        };
        let result = SolverEngine::new(NewtonMethod, options)
            .solve(&symbolic_problem(), DVector::from_vec(vec![1.0, 1.0]))
            .expect("solve");
        assert_eq!(
            result.termination,
            crate::numerical::Nonlinear_systems::error::TerminationReason::Converged
        );
        assert!(result.memory_diagnostics.is_some());
    }

    #[test]
    fn plain_backend_is_supported_by_engine_api() {
        let result = SolverEngine::new(TrustRegionMethod::default(), SolveOptions::default())
            .solve(&PlainBackendProblem, DVector::from_vec(vec![0.8, 0.3]))
            .expect("solve");
        let expected = 1.0 / 2.0_f64.sqrt();
        assert_eq!(
            result.termination,
            crate::numerical::Nonlinear_systems::error::TerminationReason::Converged
        );
        assert_relative_eq!(result.x[0], expected, epsilon = 1e-8);
        assert_relative_eq!(result.x[1], expected, epsilon = 1e-8);
    }

    #[test]
    fn prepared_parameter_updates_support_repeated_solves_for_all_facade_methods() {
        let prepared = PreparedSymbolicNonlinearProblem::from_strings(
            vec!["a*x-2".to_string(), "y+b".to_string()],
            SymbolicProblemOptions::new()
                .with_variables(vec!["x".to_string(), "y".to_string()])
                .with_equation_parameters(vec!["a".to_string(), "b".to_string()]),
        )
        .expect("affine parameterized problem should prepare");
        let first = prepared
            .bind_values(DVector::from_vec(vec![1.0, 1.0]))
            .expect("first parameter binding should validate");
        let second = prepared
            .bind_values(DVector::from_vec(vec![2.0, 2.0]))
            .expect("second parameter binding should validate");

        let methods = vec![
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
        ];
        let options = SolveOptions {
            tolerance: 1e-6,
            max_iterations: 100,
            bounds: Some(Bounds::new(vec![(-10.0, 10.0), (-10.0, 10.0)]).expect("bounds")),
            diagnostics: DiagnosticsOptions {
                collect_history: false,
                ..DiagnosticsOptions::default()
            },
            ..SolveOptions::default()
        };

        for method in methods {
            let name = method.name();
            let first_result = method
                .clone()
                .solve(&first, DVector::from_vec(vec![0.0, 0.0]), options.clone())
                .unwrap_or_else(|error| panic!("{name} first solve failed: {error}"));
            assert!(
                first_result.residual_norm < 1e-6,
                "{name} first residual is too large: {} (termination={:?}, iterations={})",
                first_result.residual_norm,
                first_result.termination,
                first_result.iterations
            );
            assert_relative_eq!(first_result.x[0], 2.0, epsilon = 1e-5);
            assert_relative_eq!(first_result.x[1], -1.0, epsilon = 1e-5);

            let second_result = method
                .solve(&second, DVector::from_vec(vec![0.0, 0.0]), options.clone())
                .unwrap_or_else(|error| panic!("{name} second solve failed: {error}"));
            assert!(
                second_result.residual_norm < 1e-6,
                "{name} second residual is too large: {}",
                second_result.residual_norm
            );
            assert_relative_eq!(second_result.x[0], 1.0, epsilon = 1e-5);
            assert_relative_eq!(second_result.x[1], -2.0, epsilon = 1e-5);
            assert!(
                first_result.statistics.iterations <= options.max_iterations
                    && second_result.statistics.iterations <= options.max_iterations,
                "{name} exceeded the configured iteration limit"
            );
        }
    }

    #[test]
    fn classic_lm_accepts_a_trial_that_meets_strict_residual_tolerance() {
        let prepared = PreparedSymbolicNonlinearProblem::from_strings(
            vec!["a*x-2".to_string(), "y+b".to_string()],
            SymbolicProblemOptions::new()
                .with_variables(vec!["x".to_string(), "y".to_string()])
                .with_equation_parameters(vec!["a".to_string(), "b".to_string()]),
        )
        .expect("strict LM regression problem should prepare");
        let bound = prepared
            .bind_values(DVector::from_vec(vec![1.0, 1.0]))
            .expect("strict LM parameter binding should validate");
        let options = SolveOptions {
            tolerance: 1e-10,
            max_iterations: 100,
            bounds: Some(Bounds::new(vec![(-10.0, 10.0), (-10.0, 10.0)]).expect("bounds")),
            diagnostics: DiagnosticsOptions {
                collect_history: false,
                ..DiagnosticsOptions::default()
            },
            ..SolveOptions::default()
        };

        let result = SolverEngine::new(LevenbergMarquardtMethod::default(), options)
            .solve(&bound, DVector::from_vec(vec![0.0, 0.0]))
            .expect("strict LM solve should succeed");
        assert!(
            result.residual_norm < 1e-10,
            "strict LM residual is too large: {} (termination={:?})",
            result.residual_norm,
            result.termination
        );
    }
}
