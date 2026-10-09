//! Dense backend parity stories for the nonlinear-system solver family.
//!
//! These are the nonlinear analogue of the LSODE2/Radau frontend and
//! continuation stories.  The matrix is intentionally dense-only: sparse and
//! banded layout axes belong to the ODE solvers, not to this solver family.

#[cfg(test)]
mod tests {
    use crate::numerical::Nonlinear_systems::engine::{
        DiagnosticsOptions, NewtonMethod, SolveOptions,
    };
    use crate::numerical::Nonlinear_systems::prelude::{
        JacobianProvider, LambdifyExecutionPolicy, NonlinearProblem, NonlinearSolverMethod,
        PreparationTelemetryMode, PreparedSymbolicNonlinearProblem, SymbolicGeneratedBackendConfig,
        SymbolicLambdifyFrontend, SymbolicNonlinearProblem, SymbolicProblemOptions,
    };
    use crate::numerical::Nonlinear_systems::symbolic_aot_test_support::aot_solver_test_guard;
    use crate::numerical::Nonlinear_systems::symbolic_generated::SymbolicAotBuildPolicy;
    use crate::symbolic::codegen::codegen_aot_runtime_link::unregister_linked_dense_backend;
    use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
    use approx::assert_relative_eq;
    use nalgebra::DVector;
    use std::time::Instant;
    use tabled::Tabled;
    use tempfile::tempdir;

    #[derive(Clone, Copy)]
    enum Policy {
        Sequential,
        Parallel,
    }

    impl Policy {
        fn execution(self) -> LambdifyExecutionPolicy {
            match self {
                Self::Sequential => LambdifyExecutionPolicy::Sequential,
                Self::Parallel => LambdifyExecutionPolicy::Parallel { min_work: 1 },
            }
        }
    }

    fn fixture(dimension: usize) -> (Vec<String>, Vec<String>, DVector<f64>, DVector<f64>) {
        let variables = (0..dimension)
            .map(|index| format!("x{index}"))
            .collect::<Vec<_>>();
        let target = DVector::from_iterator(
            dimension,
            (0..dimension).map(|index| 1.0 + index as f64 * 0.01),
        );
        let equations = variables
            .iter()
            .enumerate()
            .map(|(index, variable)| {
                let mut equation = format!("p0*({variable}^2-{:.17})", target[index].powi(2));
                if index > 0 {
                    equation.push_str(&format!("+0.04*(x{}-{:.17})", index - 1, target[index - 1]));
                }
                if index + 1 < dimension {
                    equation.push_str(&format!("+0.04*(x{}-{:.17})", index + 1, target[index + 1]));
                }
                equation
            })
            .collect::<Vec<_>>();
        let initial = target.map(|value| value * 0.9);
        (equations, variables, target, initial)
    }

    fn options(frontend: SymbolicLambdifyFrontend, policy: Policy) -> SymbolicProblemOptions {
        SymbolicProblemOptions::new()
            .with_variables((0..16).map(|index| format!("x{index}")).collect())
            .with_equation_parameters(vec!["p0".to_owned()])
            .with_equation_parameter_values(DVector::from_vec(vec![1.0]))
            .with_lambdify_frontend(frontend)
            .with_lambdify_execution_policy(policy.execution())
            .with_preparation_telemetry(PreparationTelemetryMode::Collect)
    }

    fn options_for(
        variables: Vec<String>,
        frontend: SymbolicLambdifyFrontend,
        policy: Policy,
    ) -> SymbolicProblemOptions {
        options(frontend, policy).with_variables(variables)
    }

    fn solve_options() -> SolveOptions {
        SolveOptions {
            tolerance: 1e-10,
            max_iterations: 60,
            diagnostics: DiagnosticsOptions {
                collect_statistics: true,
                ..DiagnosticsOptions::default()
            },
            ..SolveOptions::default()
        }
    }

    #[test]
    fn dense_frontends_and_execution_policies_have_matching_contracts() {
        let (equations, variables, target, initial) = fixture(16);
        let point = initial.clone();
        let mut reference_x = None;
        let mut reference_residual = None;
        let mut reference_jacobian = None;

        for frontend in [
            SymbolicLambdifyFrontend::ExprLegacy,
            SymbolicLambdifyFrontend::AtomViewNative,
        ] {
            for policy in [Policy::Sequential, Policy::Parallel] {
                let prepared = PreparedSymbolicNonlinearProblem::from_strings(
                    equations.clone(),
                    options_for(variables.clone(), frontend, policy),
                )
                .expect("dense frontend should prepare");
                let bound = prepared
                    .bind_values(DVector::from_vec(vec![1.0]))
                    .expect("parameter binding should succeed");
                let residual = bound.residual(&point).expect("residual should evaluate");
                let jacobian = bound.jacobian(&point).expect("Jacobian should evaluate");
                let result = NonlinearSolverMethod::Newton(NewtonMethod)
                    .solve(&bound, initial.clone(), solve_options())
                    .expect("dense frontend should solve");

                assert_relative_eq!(result.x, target, epsilon = 1e-8);
                if let Some(reference) = reference_x.as_ref() {
                    assert_relative_eq!(result.x, *reference, epsilon = 1e-10);
                } else {
                    reference_x = Some(result.x.clone());
                }
                if let Some(reference) = reference_residual.as_ref() {
                    assert_relative_eq!(residual, *reference, epsilon = 1e-12);
                } else {
                    reference_residual = Some(residual);
                }
                if let Some(reference) = reference_jacobian.as_ref() {
                    assert_relative_eq!(jacobian, *reference, epsilon = 1e-12);
                } else {
                    reference_jacobian = Some(jacobian);
                }
                assert!(result.statistics.residual_evaluations > 0);
                assert!(result.statistics.jacobian_evaluations > 0);
                assert!(result.statistics.linear_solves > 0);
            }
        }
    }

    #[test]
    fn dense_parameter_continuation_reuses_prepared_graph_for_both_frontends() {
        let (equations, variables, target, initial) = fixture(16);
        for frontend in [
            SymbolicLambdifyFrontend::ExprLegacy,
            SymbolicLambdifyFrontend::AtomViewNative,
        ] {
            let prepared = PreparedSymbolicNonlinearProblem::from_strings(
                equations.clone(),
                options_for(variables.clone(), frontend, Policy::Sequential),
            )
            .expect("parameterized frontend should prepare");
            let preparation_before = prepared.preparation_report().clone();
            let mut current = initial.clone();
            for parameter in [0.75, 1.0, 1.25, 1.5] {
                let bound = prepared
                    .bind_values(DVector::from_vec(vec![parameter]))
                    .expect("continuation parameter should bind");
                let result = NonlinearSolverMethod::Newton(NewtonMethod)
                    .solve(&bound, current, solve_options())
                    .expect("continuation solve should succeed");
                assert_relative_eq!(result.x, target, epsilon = 1e-8);
                current = result.x;
            }
            assert_eq!(*prepared.preparation_report(), preparation_before);
        }
    }

    #[derive(Tabled)]
    struct AotParityRow {
        frontend: String,
        lambdify_prepare_ms: String,
        aot_prepare_ms: String,
        aot_build_ms: String,
        solve_ms: String,
        continuation_ms: String,
        max_diff: String,
        status: String,
    }

    #[test]
    #[ignore = "release dense AOT versus Lambdify frontend parity and continuation story"]
    fn dense_aot_frontends_match_lambdify_with_continuation_table() {
        let _guard = aot_solver_test_guard();
        let (equations, variables, target, initial) = fixture(16);
        let mut rows = Vec::new();

        for frontend in [
            SymbolicLambdifyFrontend::ExprLegacy,
            SymbolicLambdifyFrontend::AtomViewNative,
        ] {
            let lambdify_started = Instant::now();
            let lambdify = PreparedSymbolicNonlinearProblem::from_strings(
                equations.clone(),
                options_for(variables.clone(), frontend, Policy::Sequential),
            )
            .expect("Lambdify frontend should prepare");
            let lambdify_prepare_ms = lambdify_started.elapsed().as_secs_f64() * 1e3;
            let lambdify_bound = lambdify
                .bind_values(DVector::from_vec(vec![1.0]))
                .expect("Lambdify binding should succeed");
            let lambdify_result = NonlinearSolverMethod::Newton(NewtonMethod)
                .solve(&lambdify_bound, initial.clone(), solve_options())
                .expect("Lambdify solve should succeed");

            let output_dir = tempdir().expect("AOT output directory should exist");
            let aot_started = Instant::now();
            let generated = SymbolicNonlinearProblem::from_strings_with_generated_backend(
                equations.clone(),
                options_for(variables.clone(), frontend, Policy::Sequential),
                SymbolicGeneratedBackendConfig::defaults()
                    .with_build_policy(SymbolicAotBuildPolicy::BuildIfMissing {
                        profile: AotBuildProfile::Release,
                    })
                    .with_output_parent_dir(Some(output_dir.path().to_path_buf())),
            )
            .expect("AOT frontend should build");
            let aot_prepare_ms = aot_started.elapsed().as_secs_f64() * 1e3;
            let aot_build_ms = generated
                .preparation_report
                .build_duration
                .map(|duration| duration.as_secs_f64() * 1e3)
                .unwrap_or(0.0);
            let artifact_key = generated.preparation_report.artifact_key.clone();
            let aot = generated.into_prepared();
            let aot_bound = aot
                .bind_values(DVector::from_vec(vec![1.0]))
                .expect("AOT binding should succeed");
            let solve_started = Instant::now();
            let aot_result = NonlinearSolverMethod::Newton(NewtonMethod)
                .solve(&aot_bound, initial.clone(), solve_options())
                .expect("AOT solve should succeed");
            let solve_ms = solve_started.elapsed().as_secs_f64() * 1e3;
            let mut continuation_state = aot_result.x.clone();
            let continuation_started = Instant::now();
            for parameter in [0.75, 1.0, 1.25, 1.5] {
                let bound = aot
                    .bind_values(DVector::from_vec(vec![parameter]))
                    .expect("AOT continuation binding should succeed");
                continuation_state = NonlinearSolverMethod::Newton(NewtonMethod)
                    .solve(&bound, continuation_state, solve_options())
                    .expect("AOT continuation solve should succeed")
                    .x;
            }
            let continuation_ms = continuation_started.elapsed().as_secs_f64() * 1e3;
            let max_diff = (lambdify_result.x.clone() - aot_result.x.clone())
                .norm()
                .max((aot_result.x - target.clone()).norm());
            assert_relative_eq!(continuation_state, target, epsilon = 1e-8);
            assert!(max_diff < 1e-8);
            rows.push(AotParityRow {
                frontend: frontend.as_str().to_owned(),
                lambdify_prepare_ms: format!("{lambdify_prepare_ms:.3}"),
                aot_prepare_ms: format!("{aot_prepare_ms:.3}"),
                aot_build_ms: format!("{aot_build_ms:.3}"),
                solve_ms: format!("{solve_ms:.3}"),
                continuation_ms: format!("{continuation_ms:.3}"),
                max_diff: format!("{max_diff:.3e}"),
                status: "ok".to_owned(),
            });
            if let Some(key) = artifact_key {
                let _ = unregister_linked_dense_backend(&key);
            }
        }

        crate::Utils::test_reporting::capture_test_table(
            "[Nonlinear dense AOT/Lambdify frontend parity]",
            &rows,
        );
    }
}
