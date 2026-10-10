//! Correctness gates for detailed prepared-problem preparation telemetry.
//!
//! These tests intentionally assert report structure and numerical metadata,
//! not absolute wall-clock values. Performance conclusions belong to release
//! stories and benchmarks.

#[cfg(test)]
mod tests {
    use crate::numerical::Nonlinear_systems::prelude::{
        PreparationExecutionMode, PreparationStage, PreparationTelemetryMode,
        SymbolicGeneratedBackendConfig, SymbolicNonlinearProblem, SymbolicProblemOptions,
    };
    use nalgebra::DVector;

    fn parameterized_options(mode: PreparationTelemetryMode) -> SymbolicProblemOptions {
        SymbolicProblemOptions::new()
            .with_variables(vec!["x".to_string(), "y".to_string()])
            .with_equation_parameters(vec!["a".to_string()])
            .with_equation_parameter_values(DVector::from_vec(vec![2.0]))
            .with_preparation_telemetry(mode)
    }

    #[test]
    fn detailed_preparation_telemetry_is_opt_in() {
        let problem = SymbolicNonlinearProblem::from_strings_with_options(
            vec!["a*x - 1".to_string(), "y - 2".to_string()],
            parameterized_options(PreparationTelemetryMode::Disabled),
        )
        .expect("problem should prepare");

        assert!(problem.preparation_report().detailed.is_none());
    }

    #[test]
    fn detailed_preparation_telemetry_reports_stages_and_unavailable_values() {
        let problem = SymbolicNonlinearProblem::from_strings_with_options(
            vec!["a*x - 1".to_string(), "y - 2".to_string()],
            parameterized_options(PreparationTelemetryMode::Collect),
        )
        .expect("problem should prepare");
        let report = problem
            .preparation_report()
            .detailed
            .as_ref()
            .expect("detailed telemetry should be present");

        assert_eq!(report.stages.len(), PreparationStage::ALL.len());
        assert_eq!(
            report.input_kind,
            crate::numerical::Nonlinear_systems::prelude::PreparationInputKind::Strings
        );
        assert_eq!(report.execution_mode, PreparationExecutionMode::Sequential);
        assert_eq!(report.equation_count, 2);
        assert_eq!(report.variable_count, 2);
        assert_eq!(report.parameter_count, 1);
        assert!(report.total_wall_time > std::time::Duration::ZERO);
        assert!(report.unattributed_wall_time <= report.total_wall_time);

        for stage in [
            PreparationStage::InputValidation,
            PreparationStage::ExpressionParsing,
            PreparationStage::JacobianDifferentiation,
            PreparationStage::ResidualCallbackPreparation,
            PreparationStage::JacobianCallbackPreparation,
            PreparationStage::ParameterBinding,
            PreparationStage::PreparedProblemAssembly,
        ] {
            assert!(
                report
                    .stage(stage)
                    .expect("stage should be listed")
                    .wall_time
                    .is_some(),
                "stage {stage:?} should be measured"
            );
        }

        assert!(report
            .stage(PreparationStage::ExpressionGraphMaterialization)
            .expect("stage should be listed")
            .wall_time
            .is_none());
        assert!(report
            .stage(PreparationStage::AotMaterialization)
            .expect("stage should be listed")
            .wall_time
            .is_none());
        assert!(report
            .stage(PreparationStage::FinalValidation)
            .expect("stage should be listed")
            .wall_time
            .is_none());
    }

    #[test]
    fn preparation_telemetry_survives_prepared_binding_without_rebuild() {
        let prepared = crate::numerical::Nonlinear_systems::prelude::PreparedSymbolicNonlinearProblem::from_strings(
            vec!["a*x - 1".to_string(), "y - 2".to_string()],
            parameterized_options(PreparationTelemetryMode::Collect),
        )
        .expect("prepared problem should prepare");
        let before = prepared
            .preparation_report()
            .detailed
            .clone()
            .expect("detailed telemetry should be present");

        let first = prepared
            .bind_values(DVector::from_vec(vec![2.0]))
            .expect("first binding should succeed");
        let second = prepared
            .bind_values(DVector::from_vec(vec![3.0]))
            .expect("second binding should succeed");

        assert_eq!(
            first.prepared().preparation_report().detailed.as_ref(),
            Some(&before)
        );
        assert_eq!(
            second.prepared().preparation_report().detailed.as_ref(),
            Some(&before)
        );
    }

    #[test]
    fn parallel_preparation_report_uses_the_same_schema() {
        let options = parameterized_options(PreparationTelemetryMode::Collect)
            .with_lambdify_execution_policy(
                crate::numerical::Nonlinear_systems::prelude::LambdifyExecutionPolicy::Parallel {
                    min_work: 1,
                },
            );
        let problem = SymbolicNonlinearProblem::from_strings_with_options(
            vec!["a*x - 1".to_string(), "y - 2".to_string()],
            options,
        )
        .expect("parallel problem should prepare");
        let report = problem
            .preparation_report()
            .detailed
            .as_ref()
            .expect("detailed telemetry should be present");

        assert_eq!(report.execution_mode, PreparationExecutionMode::Parallel);
        assert_eq!(report.stages.len(), PreparationStage::ALL.len());
    }

    #[test]
    fn generated_string_constructor_reports_parsing_and_typed_parse_errors() {
        let prepared = SymbolicNonlinearProblem::from_strings_with_generated_backend(
            vec!["x - 1".to_string(), "y - 2".to_string()],
            SymbolicProblemOptions::new()
                .with_variables(vec!["x".to_string(), "y".to_string()])
                .with_preparation_telemetry(PreparationTelemetryMode::Collect),
            SymbolicGeneratedBackendConfig::defaults(),
        )
        .expect("generated string preparation should succeed");
        assert!(prepared
            .preparation_report
            .detailed
            .as_ref()
            .expect("detailed report should be present")
            .stage(PreparationStage::ExpressionParsing)
            .expect("parsing stage should be listed")
            .wall_time
            .is_some());

        let error = match SymbolicNonlinearProblem::from_strings_with_generated_backend(
            vec!["x -".to_string()],
            SymbolicProblemOptions::new().with_variables(vec!["x".to_string()]),
            SymbolicGeneratedBackendConfig::defaults(),
        ) {
            Ok(_) => panic!("malformed generated input should be typed error"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            crate::numerical::Nonlinear_systems::error::SolveError::InvalidConfig(_)
        ));
    }

    #[test]
    fn detailed_constructor_preserves_partial_telemetry_on_failure() {
        let failure = match SymbolicNonlinearProblem::from_strings_with_options_detailed(
            vec!["x -".to_string()],
            SymbolicProblemOptions::new()
                .with_variables(vec!["x".to_string()])
                .with_preparation_telemetry(PreparationTelemetryMode::Collect),
        ) {
            Ok(_) => panic!("malformed input should fail"),
            Err(failure) => failure,
        };

        assert!(matches!(
            failure.error,
            crate::numerical::Nonlinear_systems::error::SolveError::InvalidConfig(_)
        ));
        let telemetry = failure
            .telemetry
            .expect("opt-in failure should retain partial telemetry");
        assert!(telemetry.total_wall_time > std::time::Duration::ZERO);
        assert!(telemetry
            .stage(PreparationStage::ExpressionParsing)
            .expect("parsing stage should be listed")
            .wall_time
            .is_some());

        let disabled = match SymbolicNonlinearProblem::from_strings_with_options_detailed(
            vec!["x -".to_string()],
            SymbolicProblemOptions::new().with_variables(vec!["x".to_string()]),
        ) {
            Ok(_) => panic!("malformed input should fail"),
            Err(failure) => failure,
        };
        assert!(disabled.telemetry.is_none());
    }
}
