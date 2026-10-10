//! Correctness and compact story coverage for the dense AtomView-native route.
//!
//! These tests deliberately compare the two supported dense Lambdify
//! frontends on the same equations. Dense AOT uses the same ABI for both
//! ExprLegacy and AtomViewNative, while keeping generated artifacts separate.

#[cfg(test)]
mod tests {
    use super::super::engine::{DiagnosticsOptions, SolveOptions};
    use super::super::prelude::{
        JacobianProvider, LambdifyExecutionPolicy, NewtonMethod, NonlinearProblem,
        NonlinearSolverMethod, PreparationStage, PreparationTelemetryMode,
        PreparedSymbolicNonlinearProblem, SolveError, SymbolicBackendSelectionPolicy,
        SymbolicDenseAotOptions, SymbolicGeneratedBackendConfig, SymbolicLambdifyFrontend,
        SymbolicNonlinearProblem, SymbolicProblemOptions,
    };
    use crate::numerical::Nonlinear_systems::symbolic_generated::SymbolicAotBuildPolicy;
    use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
    use nalgebra::{DMatrix, DVector};
    use std::time::{Duration, Instant};
    use tabled::{Table, Tabled};
    use tempfile::tempdir;

    fn equations() -> Vec<String> {
        vec!["a*x^2+y-5".to_string(), "x-y".to_string()]
    }

    fn options(frontend: SymbolicLambdifyFrontend) -> SymbolicProblemOptions {
        SymbolicProblemOptions::new()
            .with_variables(vec!["x".to_string(), "y".to_string()])
            .with_equation_parameters(vec!["a".to_string()])
            .with_equation_parameter_values(DVector::from_vec(vec![2.0]))
            .with_lambdify_execution_policy(LambdifyExecutionPolicy::Sequential)
            .with_lambdify_frontend(frontend)
            .with_preparation_telemetry(PreparationTelemetryMode::Collect)
    }

    fn prepared(frontend: SymbolicLambdifyFrontend) -> PreparedSymbolicNonlinearProblem {
        PreparedSymbolicNonlinearProblem::from_strings(equations(), options(frontend))
            .expect("nonlinear frontend preparation should succeed")
    }

    fn solve_options() -> SolveOptions {
        SolveOptions {
            tolerance: 1e-11,
            max_iterations: 40,
            diagnostics: DiagnosticsOptions {
                collect_statistics: true,
                ..DiagnosticsOptions::default()
            },
            ..SolveOptions::default()
        }
    }

    #[test]
    fn atom_native_matches_expr_legacy_residual_and_jacobian() {
        let legacy_prepared = prepared(SymbolicLambdifyFrontend::ExprLegacy);
        let legacy = legacy_prepared
            .bind_values(DVector::from_vec(vec![2.0]))
            .expect("legacy binding");
        let atom_prepared = prepared(SymbolicLambdifyFrontend::AtomViewNative);
        let atom = atom_prepared
            .bind_values(DVector::from_vec(vec![2.0]))
            .expect("AtomNative binding");
        let point = DVector::from_vec(vec![1.2, 0.7]);
        let legacy_residual = legacy.residual(&point).expect("legacy residual");
        let atom_residual = atom.residual(&point).expect("AtomNative residual");
        let legacy_jacobian = legacy.jacobian(&point).expect("legacy Jacobian");
        let atom_jacobian = atom.jacobian(&point).expect("AtomNative Jacobian");

        assert!((legacy_residual.clone() - atom_residual).norm() < 1e-13);
        assert!((legacy_jacobian.clone() - atom_jacobian).norm() < 1e-13);
        assert_eq!(
            atom.prepared().lambdify_frontend(),
            SymbolicLambdifyFrontend::AtomViewNative
        );
        let detailed = atom
            .prepared()
            .preparation_report()
            .detailed
            .as_ref()
            .expect("AtomNative telemetry should be collected");
        assert!(detailed
            .stage(PreparationStage::AtomConversion)
            .and_then(|stage| stage.wall_time)
            .is_some());
        assert!(detailed
            .stage(PreparationStage::AtomDependencyAnalysis)
            .and_then(|stage| stage.wall_time)
            .is_some());
        assert!(detailed
            .stage(PreparationStage::AtomDifferentiation)
            .and_then(|stage| stage.wall_time)
            .is_some());
        for stage in [
            PreparationStage::ResidualCallbackPreparation,
            PreparationStage::JacobianCallbackPreparation,
            PreparationStage::PreparedProblemAssembly,
        ] {
            assert!(detailed
                .stage(stage)
                .and_then(|record| record.wall_time)
                .is_some());
        }
        assert!(detailed.total_wall_time >= Duration::ZERO);
        assert!(detailed.unattributed_wall_time <= detailed.total_wall_time);
    }

    #[test]
    fn atom_native_and_expr_legacy_solver_contracts_match() {
        let initial = DVector::from_vec(vec![1.0, 1.0]);
        let legacy_prepared = prepared(SymbolicLambdifyFrontend::ExprLegacy);
        let legacy = legacy_prepared
            .bind_values(DVector::from_vec(vec![2.0]))
            .expect("legacy binding");
        let atom_prepared = prepared(SymbolicLambdifyFrontend::AtomViewNative);
        let atom = atom_prepared
            .bind_values(DVector::from_vec(vec![2.0]))
            .expect("AtomNative binding");
        let legacy_result = NonlinearSolverMethod::Newton(NewtonMethod)
            .solve(&legacy, initial.clone(), solve_options())
            .expect("legacy solve");
        let atom_result = NonlinearSolverMethod::Newton(NewtonMethod)
            .solve(&atom, initial, solve_options())
            .expect("AtomNative solve");

        assert!((legacy_result.x.clone() - atom_result.x).norm() < 1e-10);
        assert!((legacy_result.residual_norm - atom_result.residual_norm).abs() < 1e-12);
        assert_eq!(
            (
                legacy_result.statistics.iterations,
                legacy_result.statistics.jacobian_evaluations
            ),
            (
                atom_result.statistics.iterations,
                atom_result.statistics.jacobian_evaluations
            ),
            "frontend selection must not change solver-level work"
        );
    }

    #[test]
    fn atom_native_parameter_rebind_reuses_prepared_graph() {
        let prepared = prepared(SymbolicLambdifyFrontend::AtomViewNative);
        let report_before = prepared.preparation_report().clone();
        let first = prepared
            .bind_values(DVector::from_vec(vec![1.0]))
            .expect("first binding");
        let second = prepared
            .bind_values(DVector::from_vec(vec![2.0]))
            .expect("second binding");
        let point = DVector::from_vec(vec![1.0, 1.0]);
        let first_residual = first.residual(&point).expect("first residual");
        let second_residual = second.residual(&point).expect("second residual");

        assert_ne!(first_residual, second_residual);
        assert_eq!(prepared.preparation_report(), &report_before);
        assert_eq!(
            prepared.lambdify_frontend(),
            SymbolicLambdifyFrontend::AtomViewNative
        );
    }

    #[test]
    fn atom_native_generated_aot_has_typed_missing_artifact_path() {
        let direct = SymbolicNonlinearProblem::from_strings_with_backend_selection(
            equations(),
            options(SymbolicLambdifyFrontend::AtomViewNative),
            SymbolicBackendSelectionPolicy::AotOnly,
            None,
            SymbolicDenseAotOptions::default(),
        );
        let generated = SymbolicNonlinearProblem::from_strings_with_generated_backend(
            equations(),
            options(SymbolicLambdifyFrontend::AtomViewNative),
            SymbolicGeneratedBackendConfig::defaults(),
        );

        match direct {
            Err(SolveError::InvalidConfig(message)) => {
                assert!(message.contains("AOT"));
            }
            _ => panic!("expected typed AOT-only configuration error"),
        }
        match generated {
            Ok(prepared) => {
                assert_eq!(
                    prepared.selected_backend,
                    super::super::symbolic_backend::SelectedSymbolicNonlinearBackendKind::Lambdify
                );
                assert_eq!(
                    prepared.problem.lambdify_frontend(),
                    SymbolicLambdifyFrontend::AtomViewNative
                );
            }
            Err(error) => {
                panic!("default AOT policy should fall back to AtomNative Lambdify: {error}")
            }
        }
    }

    #[derive(Tabled)]
    struct AtomAotStoryRow {
        route: String,
        prepare_ms: String,
        solve_ms: String,
        max_diff: String,
        status: String,
    }

    #[test]
    #[ignore = "builds and links a generated AtomView AOT artifact"]
    fn atom_native_aot_matches_atom_lambdify_dense_story() {
        let baseline_started = Instant::now();
        let baseline_prepared = prepared(SymbolicLambdifyFrontend::AtomViewNative);
        let baseline = baseline_prepared
            .bind_values(DVector::from_vec(vec![2.0]))
            .expect("AtomNative Lambdify binding");
        let baseline_result = NonlinearSolverMethod::Newton(NewtonMethod)
            .solve(
                &baseline,
                DVector::from_vec(vec![1.0, 1.0]),
                solve_options(),
            )
            .expect("AtomNative Lambdify solve");
        let baseline_ms = baseline_started.elapsed().as_secs_f64() * 1e3;

        let output_dir = tempdir().expect("AOT output directory");
        let prepare_started = Instant::now();
        let generated = SymbolicNonlinearProblem::from_strings_with_generated_backend(
            equations(),
            options(SymbolicLambdifyFrontend::AtomViewNative),
            SymbolicGeneratedBackendConfig::defaults()
                .with_build_policy(SymbolicAotBuildPolicy::BuildIfMissing {
                    profile: AotBuildProfile::Release,
                })
                .with_output_parent_dir(Some(output_dir.path().to_path_buf())),
        )
        .expect("AtomView AOT build should succeed");
        assert_eq!(
            generated.selected_backend,
            super::super::symbolic_backend::SelectedSymbolicNonlinearBackendKind::AotCompiled
        );
        let aot_prepared = generated.into_prepared();
        let aot = aot_prepared
            .bind_values(DVector::from_vec(vec![2.0]))
            .expect("AtomView AOT binding");
        let prepare_ms = prepare_started.elapsed().as_secs_f64() * 1e3;
        let solve_started = Instant::now();
        let aot_result = NonlinearSolverMethod::Newton(NewtonMethod)
            .solve(&aot, DVector::from_vec(vec![1.0, 1.0]), solve_options())
            .expect("AtomView AOT solve");
        let solve_ms = solve_started.elapsed().as_secs_f64() * 1e3;
        let max_diff = (baseline_result.x - aot_result.x).norm();
        assert!(max_diff < 1e-10);

        let rows = [AtomAotStoryRow {
            route: "aot/atom-native vs lambdify/atom-native".to_owned(),
            prepare_ms: format!("{prepare_ms:.3}"),
            solve_ms: format!("{solve_ms:.3} (baseline {baseline_ms:.3})"),
            max_diff: format!("{max_diff:.3e}"),
            status: "ok".to_owned(),
        }];
        println!("{}", Table::new(rows));
    }

    #[derive(Tabled)]
    struct FrontendStoryRow {
        frontend: String,
        dimension: usize,
        prepare_ms: String,
        residual_ms: String,
        jacobian_ms: String,
        solve_ms: String,
        max_diff: String,
        status: String,
    }

    #[test]
    #[ignore = "release frontend performance/story evidence"]
    fn atom_native_vs_expr_legacy_dense_story_table() {
        let _report = crate::Utils::test_reporting::TestReportCapture::new(
            "Nonlinear_systems",
            "nonlinear_atom_native_story_tests::atom_native_vs_expr_legacy_dense_story_table",
        );
        let mut rows = Vec::new();
        let mut baselines: Option<(DVector<f64>, DMatrix<f64>)> = None;
        for frontend in [
            SymbolicLambdifyFrontend::ExprLegacy,
            SymbolicLambdifyFrontend::AtomViewNative,
        ] {
            let started = Instant::now();
            let prepared = prepared(frontend);
            let prepare_ms = started.elapsed().as_secs_f64() * 1e3;
            let bound = prepared
                .bind_values(DVector::from_vec(vec![2.0]))
                .expect("binding");
            let point = DVector::from_vec(vec![1.2, 0.7]);
            let mut residual = DVector::zeros(2);
            let mut jacobian = DMatrix::zeros(2, 2);
            let started = Instant::now();
            bound
                .residual_into(&point, &mut residual)
                .expect("residual");
            let residual_ms = started.elapsed().as_secs_f64() * 1e3;
            let started = Instant::now();
            bound
                .jacobian_into(&point, &mut jacobian)
                .expect("Jacobian");
            let jacobian_ms = started.elapsed().as_secs_f64() * 1e3;
            let started = Instant::now();
            let result = NonlinearSolverMethod::Newton(NewtonMethod)
                .solve(&bound, DVector::from_vec(vec![1.0, 1.0]), solve_options())
                .expect("solve");
            let solve_ms = started.elapsed().as_secs_f64() * 1e3;
            let diff = if let Some((reference_residual, reference_jacobian)) = baselines.as_ref() {
                (residual.clone() - reference_residual)
                    .norm()
                    .max((jacobian.clone() - reference_jacobian).norm())
            } else {
                baselines = Some((residual.clone(), jacobian.clone()));
                0.0
            };
            rows.push(FrontendStoryRow {
                frontend: frontend.as_str().to_owned(),
                dimension: 2,
                prepare_ms: format!("{prepare_ms:.3}"),
                residual_ms: format!("{residual_ms:.6}"),
                jacobian_ms: format!("{jacobian_ms:.6}"),
                solve_ms: format!("{solve_ms:.3}"),
                max_diff: format!("{diff:.3e}"),
                status: format!("{:?}", result.termination),
            });
        }
        crate::Utils::test_reporting::capture_test_table(
            "[Nonlinear AtomNative frontend matrix]",
            &rows,
        );
    }
}
