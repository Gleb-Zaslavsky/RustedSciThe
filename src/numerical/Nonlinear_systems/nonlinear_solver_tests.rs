#[cfg(test)]
mod tests {

    use crate::numerical::Nonlinear_systems::engine::{
        DiagnosticsOptions, NewtonMethod, SolveOptions, SolverEngine,
    };
    use crate::numerical::Nonlinear_systems::symbolic::SymbolicNonlinearProblem;
    use crate::numerical::Nonlinear_systems::trust_region::TrustRegionMethod;
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
}
