use super::*;
use crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend;
use crate::symbolic::symbolic_engine::Expr;
use nalgebra::DVector;

fn lsode2_config() -> Lsode2ProblemConfig {
    Lsode2ProblemConfig::new(
        vec![Expr::parse_expression("-y")],
        vec!["y".into()],
        "t".into(),
        0.0,
        DVector::from_element(1, 1.0),
        0.1,
        0.025,
        1e-7,
        1e-9,
    )
}

#[test]
fn borrowed_trajectory_materializes_row_oriented_dataset_only_when_requested() {
    let axis = DVector::from_vec(vec![0.0, 0.5]);
    let values = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    let view = StiffTrajectoryView {
        solver: StiffSolverKind::BackwardEuler,
        axis: &axis,
        values: &values,
    };

    let dataset = view
        .to_dataset("time", vec!["a".into(), "b".into()])
        .unwrap();
    assert_eq!(dataset.axis, axis);
    assert_eq!(dataset.variable_names, vec!["a", "b"]);
    assert_eq!(dataset.values.shape(), (2, 2));
    assert_eq!(dataset.values[(0, 0)], 1.0);
    assert_eq!(dataset.values[(0, 1)], 2.0);
    assert_eq!(dataset.values[(1, 0)], 3.0);
    assert_eq!(dataset.values[(1, 1)], 4.0);

    let error = view.to_dataset("time", vec!["only_one_name".into()]);
    assert!(matches!(
        error,
        Err(StiffPostprocessError::InvalidDataset(
            PostprocessError::InvalidDataset(_)
        ))
    ));
}

#[test]
fn stiff_parameter_loop_keeps_each_native_failure_and_continues() {
    let options = BdfSolverOptions::for_bdf(
        vec![Expr::parse_expression("-rate*y")],
        vec!["y".into()],
        "t".into(),
        0.0,
        DVector::from_element(1, 1.0),
        0.2,
        0.02,
        1e-8,
        1e-10,
        None,
        false,
        Some(0.01),
    )
    .with_equation_parameters(vec!["rate".into()])
    .with_equation_parameter_values(DVector::from_element(1, 1.0))
    .with_telemetry_mode(BdfTelemetryMode::Counters);
    let mut prepared = StiffSolverRequest::Bdf { options }
        .prepare(None)
        .expect("valid BDF request should prepare");

    let report = prepared.try_solve_parameter_loop(
        DVector::from_element(1, 1.0),
        &[vec![1.0], vec![], vec![2.0]],
    );

    assert_eq!(report.points.len(), 3);
    assert!(report.points[0].result.is_ok());
    assert!(report.points[1].result.is_err());
    assert!(report.points[2].result.is_ok());
    assert!(report.points.iter().all(|point| point.telemetry.is_some()));
    for index in [0, 2] {
        let solution = report.points[index].result.as_ref().unwrap();
        assert_eq!(solution.values[(0, 0)], 1.0);
        assert_eq!(solution.values.nrows(), solution.x_mesh.len());
    }
}

#[test]
fn prepared_solver_reports_unavailable_or_solver_specific_trajectory_routes() {
    let bdf = prepare_bdf(
        BdfSolverOptions::for_bdf(
            vec![Expr::parse_expression("-y")],
            vec!["y".into()],
            "t".into(),
            0.0,
            DVector::from_element(1, 1.0),
            0.1,
            0.025,
            1e-7,
            1e-9,
            None,
            false,
            None,
        ),
        None,
    )
    .unwrap();
    assert!(matches!(
        PreparedStiffSolver::Bdf(bdf).trajectory_view(),
        Err(StiffPostprocessError::ResultUnavailable {
            solver: StiffSolverKind::Bdf
        })
    ));

    let be = prepare_backward_euler(
        BeSolverOptions::new(
            vec![Expr::parse_expression("-y")],
            vec!["y".into()],
            "t".into(),
            1e-8,
            20,
            Some(0.01),
            0.0,
            0.1,
            DVector::from_element(1, 1.0),
        ),
        None,
    )
    .unwrap();
    let prepared_be = PreparedStiffSolver::BackwardEuler(be);
    let be_view = prepared_be
        .trajectory_view()
        .expect("BE deliberately retains its initial partial trajectory");
    assert_eq!(be_view.axis.as_slice(), &[0.0]);
    assert_eq!(be_view.values.shape(), (1, 1));
    assert_eq!(be_view.values[(0, 0)], 1.0);

    let lsode = prepare_lsode2(lsode2_config(), None).unwrap();
    assert!(matches!(
        PreparedStiffSolver::Lsode2(lsode).trajectory_view(),
        Err(StiffPostprocessError::ResultUnavailable {
            solver: StiffSolverKind::Lsode2
        })
    ));

    let radau = prepare_radau(
        crate::numerical::Radau::api::RadauProblem::new(
            vec![Expr::parse_expression("-y")],
            vec!["y".into()],
            "t",
        ),
        RadauConfig::default(),
        None,
    )
    .unwrap();
    assert!(matches!(
        PreparedStiffSolver::Radau(PreparedRadau {
            solver: radau,
            initial_state: vec![1.0],
            parameters: Vec::new(),
        })
        .trajectory_view(),
        Err(StiffPostprocessError::RadauSolutionRequired)
    ));
}

#[test]
fn radau_dataset_uses_explicit_dense_output_samples_and_preserves_typed_errors() {
    let problem = crate::numerical::Radau::api::RadauProblem::new(
        vec![Expr::parse_expression("-y")],
        vec!["y".into()],
        "t",
    );
    let mut solver = prepare_radau(
        problem,
        RadauConfig {
            t_bound: 1.0,
            output: crate::numerical::Radau::api::RadauOutputPolicy::Dense,
            telemetry: crate::numerical::Radau::api::RadauTelemetryMode::Detailed,
            ..RadauConfig::default()
        },
        None,
    )
    .unwrap();
    let solution = solver.solve(&[1.0]).unwrap();
    assert!(matches!(
        StiffTelemetrySnapshot::from_radau_solution(&solution),
        StiffTelemetrySnapshot::Radau(ref telemetry)
            if telemetry.mode == crate::numerical::Radau::api::RadauTelemetryMode::Detailed
    ));
    let dataset =
        radau_postprocess_dataset(&solution, &[0.0, 0.5, 1.0], "time", vec!["y".into()]).unwrap();
    assert_eq!(dataset.values.shape(), (3, 1));
    assert!((dataset.values[(0, 0)] - 1.0).abs() < 1e-14);
    assert!((dataset.values[(1, 0)] - (-0.5_f64).exp()).abs() < 2e-5);
    assert!((dataset.values[(2, 0)] - (-1.0_f64).exp()).abs() < 2e-5);

    let outside = radau_postprocess_dataset(&solution, &[2.0], "time", vec!["y".into()]);
    assert!(matches!(
        outside,
        Err(StiffPostprocessError::RadauSampling(_))
    ));
}

#[test]
fn postprocessing_plan_failure_keeps_source_trajectory_borrowable() {
    let axis = DVector::from_vec(vec![0.0, 1.0]);
    let values = DMatrix::from_column_slice(2, 1, &[1.0, 0.5]);
    let view = StiffTrajectoryView {
        solver: StiffSolverKind::Lsode2,
        axis: &axis,
        values: &values,
    };
    let target = std::env::temp_dir().join(format!(
        "rst-stiff-postprocess-target-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&target).unwrap();
    let plan = PostprocessPlan::new().write_report(&target);

    let error = view.execute("time", vec!["y".into()], &plan).unwrap_err();
    assert!(matches!(
        error,
        StiffPostprocessError::PlanExecution {
            solver: StiffSolverKind::Lsode2,
            ..
        }
    ));
    assert!(std::error::Error::source(&error).is_some());
    assert_eq!(view.values[(1, 0)], 0.5);

    std::fs::remove_dir_all(target).unwrap();
}

#[test]
fn expr_dense_preset_changes_only_shared_axes() {
    let mut radau = RadauConfig::default();
    radau.rtol = 2.5e-5;
    radau.max_steps = 42;
    StiffPreset::expr_dense().apply_radau(&mut radau).unwrap();
    assert_eq!(radau.frontend, RadauFrontend::ExprLegacy);
    assert_eq!(radau.matrix_layout, RadauMatrixLayout::Dense);
    assert_eq!(radau.rtol, 2.5e-5);
    assert_eq!(radau.max_steps, 42);

    let mut be = BeSolverOptions::new(
        vec![Expr::parse_expression("-y")],
        vec!["y".into()],
        "t".into(),
        1e-8,
        20,
        Some(0.01),
        0.0,
        1.0,
        DVector::from_element(1, 1.0),
    );
    be.max_steps = 123;
    StiffPreset::expr_dense().apply_be(&mut be).unwrap();
    assert_eq!(
        be.symbolic_assembly_backend,
        BeSymbolicAssemblyBackend::ExprLegacy
    );
    assert_eq!(be.max_steps, 123);
}

#[test]
fn backend_rejects_structured_layout_instead_of_densifying() {
    let mut options = BeSolverOptions::new(
        vec![Expr::parse_expression("-y")],
        vec!["y".into()],
        "t".into(),
        1e-8,
        20,
        Some(0.01),
        0.0,
        1.0,
        DVector::from_element(1, 1.0),
    );
    let error = StiffPreset::default()
        .with_matrix_layout(StiffMatrixLayout::Sparse)
        .apply_be(&mut options)
        .unwrap_err();
    assert!(matches!(
        error,
        StiffApiError::UnsupportedCapability {
            solver: StiffSolverKind::BackwardEuler,
            ..
        }
    ));
}

#[test]
fn radau_auto_band_preset_uses_native_symbolic_inference() {
    let problem = crate::numerical::Radau::api::RadauProblem::new(
        vec![
            Expr::parse_expression("-2*y0 + y1"),
            Expr::parse_expression("y0 - 2*y1 + y2"),
            Expr::parse_expression("y1 - 2*y2"),
        ],
        vec!["y0".into(), "y1".into(), "y2".into()],
        "t",
    );
    let preset = StiffPreset::default().with_matrix_layout(StiffMatrixLayout::BandedAuto);
    let solver = prepare_radau(problem, RadauConfig::default(), Some(&preset))
        .expect("StiffAPI should pass automatic bandwidth to native Radau preparation");

    assert_eq!(
        solver.config().matrix_layout,
        RadauMatrixLayout::Banded { lower: 1, upper: 1 }
    );
}

#[test]
fn route_aware_request_dispatches_without_flattening_native_configs() {
    let radau_request = StiffSolverRequest::Radau {
        problem: crate::numerical::Radau::api::RadauProblem::new(
            vec![Expr::parse_expression("-y")],
            vec!["y".into()],
            "t",
        ),
        config: RadauConfig::default(),
        initial_state: vec![1.0],
        parameters: Vec::new(),
    };
    let mut radau = radau_request
        .prepare(Some(&StiffPreset::expr_dense()))
        .expect("Radau request should dispatch to its native preparation");
    assert_eq!(radau.kind(), StiffSolverKind::Radau);
    assert!(radau.as_radau().is_some());
    assert!(radau.as_backward_euler().is_none());
    assert!(matches!(radau.try_solve(), Ok(StiffSolveOutcome::Radau(_))));

    let be_request = StiffSolverRequest::BackwardEuler {
        options: BeSolverOptions::new(
            vec![Expr::parse_expression("-y")],
            vec!["y".into()],
            "t".into(),
            1e-8,
            20,
            Some(0.01),
            0.0,
            0.1,
            DVector::from_element(1, 1.0),
        ),
        equation_parameters: Vec::new(),
        parameter_values: Vec::new(),
    };
    let be = be_request
        .prepare(Some(&StiffPreset::expr_dense()))
        .expect("BE request should dispatch to its native preparation");
    assert_eq!(be.kind(), StiffSolverKind::BackwardEuler);
    assert!(be.as_backward_euler().is_some());
    assert!(be.as_radau().is_none());
}

#[test]
fn dispatched_solve_preserves_native_error_and_keeps_solver_available() {
    let problem = crate::numerical::Radau::api::RadauProblem::new(
        vec![Expr::parse_expression("-rate*y")],
        vec!["y".into()],
        "t",
    )
    .with_parameter_names(vec!["rate".into()]);
    let mut prepared = StiffSolverRequest::Radau {
        problem,
        config: RadauConfig::default(),
        initial_state: vec![1.0],
        parameters: Vec::new(),
    }
    .prepare(None)
    .unwrap();

    let error = prepared.try_solve().unwrap_err();
    assert_eq!(error.solver_kind(), StiffSolverKind::Radau);
    assert!(matches!(error, StiffSolveError::Radau(ref native)
        if native.kind() == crate::numerical::Radau::api::RadauErrorKind::Shape));
    assert!(std::error::Error::source(&error).is_some());
    assert!(prepared.as_radau().is_some());
}

#[test]
fn composition_rejects_duplicate_axes_and_preserves_independent_profiles() {
    let preset = StiffPreset::expr_dense()
        .compose(StiffPreset::default().with_telemetry(SolverTelemetryMode::Timings))
        .unwrap();
    assert!(matches!(
        StiffPreset::lambdify().compose(StiffPreset::lambdify()),
        Err(StiffApiError::ConflictingPresetAxis("execution"))
    ));
    assert!(matches!(
        preset.compose(StiffPreset::default().with_frontend(StiffFrontend::AtomViewNative)),
        Err(StiffApiError::ConflictingPresetAxis("frontend"))
    ));
    assert!(matches!(
        StiffPreset::default()
            .with_callback_policy(StiffCallbackPolicy::Sequential)
            .compose(
                StiffPreset::default()
                    .with_callback_policy(StiffCallbackPolicy::Auto { min_work: 64 })
            ),
        Err(StiffApiError::ConflictingPresetAxis("callback_policy"))
    ));
}

#[test]
fn named_profiles_change_only_their_declared_axes() {
    assert_eq!(
        StiffPreset::expr_dense().frontend,
        Some(StiffFrontend::ExprLegacy)
    );
    assert_eq!(
        StiffPreset::expr_dense().matrix_layout,
        Some(StiffMatrixLayout::Dense)
    );
    assert_eq!(
        StiffPreset::atom_sparse().frontend,
        Some(StiffFrontend::AtomViewNative)
    );
    assert_eq!(
        StiffPreset::atom_sparse().matrix_layout,
        Some(StiffMatrixLayout::Sparse)
    );
    assert_eq!(
        StiffPreset::expr_banded_auto().matrix_layout,
        Some(StiffMatrixLayout::BandedAuto)
    );
    assert_eq!(
        StiffPreset::automatic_callbacks(48).callback_policy,
        Some(StiffCallbackPolicy::Auto { min_work: 48 })
    );
    assert!(StiffPreset::atom_dense().execution.is_none());
    assert!(StiffPreset::expr_sparse().telemetry.is_none());
    assert_eq!(StiffPreset::solve_1d().required_state_dimension, Some(1));
    assert_eq!(
        StiffPreset::solve_1d().frontend,
        Some(StiffFrontend::ExprLegacy)
    );
    assert_eq!(
        StiffPreset::solve_1d().matrix_layout,
        Some(StiffMatrixLayout::Dense)
    );
}

#[test]
fn preset_catalog_and_composed_descriptions_report_fields_tools_and_constraints() {
    let catalog = StiffPreset::catalog();
    let names = catalog
        .iter()
        .map(|entry| entry.name)
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(
        names.len(),
        catalog.len(),
        "preset catalog names must be unique"
    );
    assert!(names.contains("solve_1d"));
    assert!(names.contains("aot_tcc"));
    let aot_template = catalog
        .iter()
        .find(|entry| entry.name == "aot_tcc")
        .unwrap();
    assert_eq!(
        aot_template.tools_if_aot_build_needed,
        &["cargo", "rustc", "tcc"]
    );
    assert!(aot_template.requires_output_directory);

    let prebuilt = StiffPreset::default()
        .with_execution(StiffExecution::Aot(
            SymbolicIvpGeneratedBackendConfig::require_prebuilt().with_c_tcc(),
        ))
        .describe();
    assert!(prebuilt.tools_if_aot_build_needed.is_empty());

    let banded = StiffPreset::atom_banded_auto().describe();
    assert_eq!(
        banded.fields_changed,
        vec!["symbolic_frontend", "linear_system_layout"]
    );
    assert_eq!(
        banded
            .solvers
            .iter()
            .find(|entry| entry.solver == StiffSolverKind::Bdf)
            .unwrap()
            .applicability,
        StiffPresetApplicability::Unsupported
    );
    assert_eq!(
        banded
            .solvers
            .iter()
            .find(|entry| entry.solver == StiffSolverKind::Radau)
            .unwrap()
            .applicability,
        StiffPresetApplicability::Supported
    );
    assert_eq!(
        banded
            .solvers
            .iter()
            .find(|entry| entry.solver == StiffSolverKind::BackwardEuler)
            .unwrap()
            .applicability,
        StiffPresetApplicability::Unsupported
    );
    assert_eq!(
        banded
            .solvers
            .iter()
            .find(|entry| entry.solver == StiffSolverKind::Lsode2)
            .unwrap()
            .applicability,
        StiffPresetApplicability::Conditional
    );
    assert!(
        banded
            .solvers
            .iter()
            .find(|entry| entry.solver == StiffSolverKind::Lsode2)
            .unwrap()
            .constraint
            .unwrap()
            .contains("symbolic")
    );

    let callback_aot = StiffPreset::parallel_callbacks(64)
        .compose(StiffPreset::aot_tcc("unused-catalog-test"))
        .unwrap()
        .describe();
    assert_eq!(
        callback_aot.tools_if_aot_build_needed,
        vec!["cargo", "rustc", "tcc"]
    );
    assert_eq!(
        callback_aot
            .solvers
            .iter()
            .find(|entry| entry.solver == StiffSolverKind::Bdf)
            .unwrap()
            .applicability,
        StiffPresetApplicability::Unsupported
    );
    assert_eq!(
        callback_aot
            .solvers
            .iter()
            .find(|entry| entry.solver == StiffSolverKind::Radau)
            .unwrap()
            .applicability,
        StiffPresetApplicability::Supported
    );
    assert_eq!(
        callback_aot
            .solvers
            .iter()
            .find(|entry| entry.solver == StiffSolverKind::BackwardEuler)
            .unwrap()
            .applicability,
        StiffPresetApplicability::Unsupported
    );
    assert_eq!(
        callback_aot
            .solvers
            .iter()
            .find(|entry| entry.solver == StiffSolverKind::Lsode2)
            .unwrap()
            .applicability,
        StiffPresetApplicability::Unsupported
    );
}

#[test]
fn every_preset_catalog_entry_matches_its_constructor_description() {
    let cases = [
        ("solve_1d", StiffPreset::solve_1d()),
        ("expr_dense", StiffPreset::expr_dense()),
        ("atom_dense", StiffPreset::atom_dense()),
        ("expr_sparse", StiffPreset::expr_sparse()),
        ("atom_sparse", StiffPreset::atom_sparse()),
        ("expr_banded_auto", StiffPreset::expr_banded_auto()),
        ("atom_banded_auto", StiffPreset::atom_banded_auto()),
        ("lambdify", StiffPreset::lambdify()),
        ("sequential_callbacks", StiffPreset::sequential_callbacks()),
        ("parallel_callbacks", StiffPreset::parallel_callbacks(32)),
        ("automatic_callbacks", StiffPreset::automatic_callbacks(32)),
        ("aot_tcc", StiffPreset::aot_tcc("unused-catalog")),
        ("aot_gcc", StiffPreset::aot_gcc("unused-catalog")),
        ("aot_zig", StiffPreset::aot_zig("unused-catalog")),
        ("aot_rust", StiffPreset::aot_rust("unused-catalog")),
    ];

    assert_eq!(cases.len(), StiffPreset::catalog().len());
    for (name, preset) in cases {
        let entry = StiffPreset::catalog()
            .iter()
            .find(|entry| entry.name == name)
            .unwrap_or_else(|| panic!("missing catalog entry for {name}"));
        let description = preset.describe();
        assert_eq!(description.fields_changed, entry.fields_changed, "{name}");
        assert_eq!(
            description.tools_if_aot_build_needed, entry.tools_if_aot_build_needed,
            "{name}"
        );
        assert_eq!(
            entry.requires_output_directory,
            matches!(preset.execution, Some(StiffExecution::Aot(_))),
            "{name}"
        );
    }
}

#[test]
fn named_aot_profiles_retain_exact_codegen_and_build_policy() {
    let cases = [
        (
            StiffPreset::aot_tcc("unused-tcc"),
            AotCodegenBackend::C,
            Some("tcc"),
        ),
        (
            StiffPreset::aot_gcc("unused-gcc"),
            AotCodegenBackend::C,
            Some("gcc"),
        ),
        (
            StiffPreset::aot_zig("unused-zig"),
            AotCodegenBackend::Zig,
            None,
        ),
        (
            StiffPreset::aot_rust("unused-rust"),
            AotCodegenBackend::Rust,
            None,
        ),
    ];

    for (preset, backend, compiler) in cases {
        let Some(StiffExecution::Aot(config)) = preset.execution else {
            panic!("named AOT profile must explicitly select AOT");
        };
        assert_eq!(config.aot_codegen_backend, backend);
        assert_eq!(config.aot_c_compiler.as_deref(), compiler);
        assert!(matches!(
                config.build_policy,
                SymbolicIvpAotBuildPolicy::BuildIfMissing {
                    profile: crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile::Release
                }
            ));
    }
}

#[test]
fn frontend_presets_forward_expr_and_atom_to_all_solver_adapters() {
    for (frontend, ivp_backend, be_backend, radau_frontend, lsode_assembly) in [
        (
            StiffFrontend::ExprLegacy,
            IvpSymbolicAssemblyBackend::ExprLegacy,
            BeSymbolicAssemblyBackend::ExprLegacy,
            RadauFrontend::ExprLegacy,
            Lsode2SymbolicAssemblyBackend::ExprLegacy,
        ),
        (
            StiffFrontend::AtomViewNative,
            IvpSymbolicAssemblyBackend::AtomView,
            BeSymbolicAssemblyBackend::AtomViewNative,
            RadauFrontend::AtomViewNative,
            Lsode2SymbolicAssemblyBackend::AtomView,
        ),
    ] {
        let preset = StiffPreset::default().with_frontend(frontend);
        let bdf = prepare_bdf(
            BdfSolverOptions::for_bdf(
                vec![Expr::parse_expression("-y")],
                vec!["y".into()],
                "t".into(),
                0.0,
                DVector::from_element(1, 1.0),
                0.1,
                0.025,
                1e-7,
                1e-9,
                None,
                false,
                None,
            ),
            Some(&preset),
        )
        .unwrap();
        assert_eq!(bdf.symbolic_assembly_backend(), ivp_backend);

        let be = prepare_backward_euler(
            BeSolverOptions::new(
                vec![Expr::parse_expression("-y")],
                vec!["y".into()],
                "t".into(),
                1e-8,
                20,
                Some(0.01),
                0.0,
                0.1,
                DVector::from_element(1, 1.0),
            ),
            Some(&preset),
        )
        .unwrap();
        assert_eq!(be.symbolic_assembly_backend(), be_backend);

        let radau = prepare_radau(
            crate::numerical::Radau::api::RadauProblem::new(
                vec![Expr::parse_expression("-y")],
                vec!["y".into()],
                "t",
            ),
            RadauConfig::default(),
            Some(&preset),
        )
        .unwrap();
        assert_eq!(radau.config().frontend, radau_frontend);

        let lsode = prepare_lsode2(lsode2_config(), Some(&preset)).unwrap();
        assert!(matches!(
            lsode.config().residual_jacobian_source,
            Lsode2ResidualJacobianSource::Symbolic { assembly, .. }
                if assembly == lsode_assembly
        ));
    }
}

#[test]
fn callback_policy_forwards_to_radau_and_is_rejected_by_backward_euler() {
    let mut config = RadauConfig::default();
    StiffPreset::default()
        .with_callback_policy(StiffCallbackPolicy::Auto { min_work: 128 })
        .apply_radau(&mut config)
        .unwrap();
    assert_eq!(
        config.execution_policy,
        RadauExecutionPolicy::Auto { min_work: 128 }
    );

    let mut be = BeSolverOptions::new(
        vec![Expr::parse_expression("-y")],
        vec!["y".into()],
        "t".into(),
        1e-8,
        20,
        Some(0.01),
        0.0,
        1.0,
        DVector::from_element(1, 1.0),
    );
    assert!(matches!(
        StiffPreset::default()
            .with_callback_policy(StiffCallbackPolicy::Sequential)
            .apply_be(&mut be),
        Err(StiffApiError::UnsupportedCapability {
            solver: StiffSolverKind::BackwardEuler,
            ..
        })
    ));
}

#[test]
fn telemetry_and_logging_preserve_solver_specific_native_capabilities() {
    let telemetry_cases = [
        (
            SolverTelemetryMode::Off,
            crate::numerical::Radau::api::RadauTelemetryMode::Off,
            BeTelemetryMode::Off,
        ),
        (
            SolverTelemetryMode::Counters,
            crate::numerical::Radau::api::RadauTelemetryMode::Counters,
            BeTelemetryMode::Counters,
        ),
        (
            SolverTelemetryMode::Timings,
            crate::numerical::Radau::api::RadauTelemetryMode::Timings,
            BeTelemetryMode::Timings,
        ),
        (
            SolverTelemetryMode::Detailed,
            crate::numerical::Radau::api::RadauTelemetryMode::Detailed,
            BeTelemetryMode::Detailed,
        ),
    ];

    for (requested, radau_expected, be_expected) in telemetry_cases {
        let preset = StiffPreset::default().with_telemetry(requested);
        let mut radau = RadauConfig::default();
        preset.apply_radau(&mut radau).unwrap();
        assert_eq!(radau.telemetry, radau_expected);

        let mut be = BeSolverOptions::new(
            vec![Expr::parse_expression("-y")],
            vec!["y".into()],
            "t".into(),
            1e-8,
            20,
            Some(0.01),
            0.0,
            1.0,
            DVector::from_element(1, 1.0),
        );
        preset.apply_be(&mut be).unwrap();
        assert_eq!(be.telemetry_mode, be_expected);
    }

    let mut be = BeSolverOptions::new(
        vec![Expr::parse_expression("-y")],
        vec!["y".into()],
        "t".into(),
        1e-8,
        20,
        Some(0.01),
        0.0,
        1.0,
        DVector::from_element(1, 1.0),
    );
    StiffPreset::default()
        .with_logging(SolverLogLevel::Warn)
        .apply_be(&mut be)
        .unwrap();
    assert_eq!(be.logging_level, SolverLogLevel::Warn);

    assert!(matches!(
        StiffPreset::default()
            .with_logging(SolverLogLevel::Warn)
            .apply_radau(&mut RadauConfig::default()),
        Err(StiffApiError::UnsupportedCapability {
            solver: StiffSolverKind::Radau,
            ..
        })
    ));
}

#[test]
fn prepared_facade_reports_effective_telemetry_mode_for_each_native_solver() {
    for mode in [
        SolverTelemetryMode::Off,
        SolverTelemetryMode::Counters,
        SolverTelemetryMode::Timings,
        SolverTelemetryMode::Detailed,
    ] {
        let preset = StiffPreset::default().with_telemetry(mode);
        let bdf = prepare_bdf(
            BdfSolverOptions::for_bdf(
                vec![Expr::parse_expression("-y")],
                vec!["y".into()],
                "t".into(),
                0.0,
                DVector::from_element(1, 1.0),
                0.1,
                0.025,
                1e-7,
                1e-9,
                None,
                false,
                None,
            ),
            Some(&preset),
        )
        .unwrap();
        let bdf = PreparedStiffSolver::Bdf(bdf);
        assert_eq!(bdf.telemetry_mode(), mode);
        assert!(matches!(
            bdf.telemetry_snapshot(),
            Ok(StiffTelemetrySnapshot::Bdf { mode: actual, .. }) if actual == mode
        ));

        let radau = prepare_radau(
            crate::numerical::Radau::api::RadauProblem::new(
                vec![Expr::parse_expression("-y")],
                vec!["y".into()],
                "t",
            ),
            RadauConfig::default(),
            Some(&preset),
        )
        .unwrap();
        let radau = PreparedStiffSolver::Radau(PreparedRadau {
            solver: radau,
            initial_state: vec![1.0],
            parameters: Vec::new(),
        });
        assert_eq!(radau.telemetry_mode(), mode);
        assert_eq!(
            radau.telemetry_snapshot().unwrap_err(),
            StiffTelemetryError::RadauSolutionRequired
        );

        let be = prepare_backward_euler(
            BeSolverOptions::new(
                vec![Expr::parse_expression("-y")],
                vec!["y".into()],
                "t".into(),
                1e-8,
                20,
                Some(0.01),
                0.0,
                0.1,
                DVector::from_element(1, 1.0),
            ),
            Some(&preset),
        )
        .unwrap();
        let be = PreparedStiffSolver::BackwardEuler(be);
        assert_eq!(be.telemetry_mode(), mode);
        assert!(matches!(
            be.telemetry_snapshot(),
            Ok(StiffTelemetrySnapshot::BackwardEuler { mode: actual, .. }) if actual == mode
        ));

        let lsode = prepare_lsode2(lsode2_config(), Some(&preset)).unwrap();
        let lsode = PreparedStiffSolver::Lsode2(lsode);
        assert_eq!(lsode.telemetry_mode(), mode);
        assert!(matches!(
            lsode.telemetry_snapshot(),
            Ok(StiffTelemetrySnapshot::Lsode2 { mode: actual, .. }) if actual == mode
        ));
    }
}

#[test]
fn preparation_errors_keep_solver_stage_and_native_error_chain() {
    let mut bdf_options = BdfSolverOptions::for_bdf(
        vec![Expr::parse_expression("-k*y")],
        vec!["y".into()],
        "t".into(),
        0.0,
        DVector::from_element(1, 1.0),
        0.1,
        0.025,
        1e-7,
        1e-9,
        None,
        false,
        None,
    );
    bdf_options.equation_parameters = Some(vec!["k".into()]);
    let bdf_error = match prepare_bdf(bdf_options, None) {
        Ok(_) => panic!("missing BDF parameter values must fail preparation"),
        Err(error) => error,
    };
    assert_eq!(bdf_error.solver_kind(), Some(StiffSolverKind::Bdf));
    assert_eq!(bdf_error.stage(), StiffApiErrorStage::Preparation);
    let StiffApiError::Bdf(BdfSolveError::Backend(backend_error)) = &bdf_error else {
        panic!("BDF preparation error should preserve the typed backend error")
    };
    assert!(matches!(
        backend_error,
        crate::symbolic::symbolic_ivp::IvpBackendError::MissingParameterValues { expected: 1 }
    ));
    let native_error = Error::source(&bdf_error).unwrap();
    assert!(native_error.downcast_ref::<BdfSolveError>().is_some());
    assert!(
        Error::source(native_error)
            .unwrap()
            .downcast_ref::<crate::symbolic::symbolic_ivp::IvpBackendError>()
            .is_some()
    );

    let be_options = BeSolverOptions::new(
        vec![Expr::parse_expression("-y")],
        vec!["y".into()],
        "t".into(),
        1e-8,
        20,
        Some(0.01),
        0.0,
        0.1,
        DVector::zeros(0),
    );
    let be_error = match prepare_backward_euler(be_options, None) {
        Ok(_) => panic!("empty BE initial state must fail preparation"),
        Err(error) => error,
    };
    assert_eq!(be_error.solver_kind(), Some(StiffSolverKind::BackwardEuler));
    assert_eq!(be_error.stage(), StiffApiErrorStage::Preparation);
    assert!(matches!(
        be_error,
        StiffApiError::BackwardEuler(BeError::InvalidConfiguration(_))
    ));

    let config_error = StiffApiError::DimensionMismatch {
        solver: StiffSolverKind::Radau,
        expected: 1,
        actual: 2,
    };
    assert_eq!(config_error.solver_kind(), Some(StiffSolverKind::Radau));
    assert_eq!(config_error.stage(), StiffApiErrorStage::Configuration);
    assert_eq!(
        StiffApiError::ConflictingPresetAxis("frontend").solver_kind(),
        None
    );
}

#[test]
fn preparation_errors_keep_native_radau_and_lsode2_categories() {
    let radau_problem = crate::numerical::Radau::api::RadauProblem::new(
        vec![Expr::parse_expression("-y")],
        vec!["y".into()],
        "t",
    );
    let radau_error = match prepare_radau(
        radau_problem,
        RadauConfig {
            t_bound: 0.0,
            ..RadauConfig::default()
        },
        None,
    ) {
        Ok(_) => panic!("invalid Radau interval must fail preparation"),
        Err(error) => error,
    };
    assert_eq!(radau_error.solver_kind(), Some(StiffSolverKind::Radau));
    assert_eq!(radau_error.stage(), StiffApiErrorStage::Preparation);
    assert!(matches!(radau_error, StiffApiError::Radau(_)));
    assert!(Error::source(&radau_error).is_some());

    let mut config = lsode2_config();
    config.controller.max_bdf_order = 0;
    let lsode_error = match prepare_lsode2(config, None) {
        Ok(_) => panic!("invalid LSODE2 controller must fail preparation"),
        Err(error) => error,
    };
    assert_eq!(lsode_error.solver_kind(), Some(StiffSolverKind::Lsode2));
    assert_eq!(lsode_error.stage(), StiffApiErrorStage::Preparation);
    assert!(matches!(
        lsode_error,
        StiffApiError::Lsode2(Lsode2Error::InvalidConfig(_))
    ));
    assert!(Error::source(&lsode_error).is_some());
}

#[test]
fn bdf_adapter_forwards_native_axes_and_rejects_unsupported_layouts() {
    let mut options = BdfSolverOptions::for_bdf(
        vec![Expr::parse_expression("-y")],
        vec!["y".into()],
        "t".into(),
        0.0,
        DVector::from_element(1, 1.0),
        0.1,
        0.025,
        1e-7,
        1e-9,
        None,
        false,
        None,
    );
    let preset = StiffPreset::default()
        .with_frontend(StiffFrontend::AtomViewNative)
        .with_matrix_layout(StiffMatrixLayout::Dense)
        .with_callback_policy(StiffCallbackPolicy::Auto { min_work: 32 })
        .with_telemetry(SolverTelemetryMode::Counters)
        .with_logging(SolverLogLevel::Warn);
    preset.apply_bdf(&mut options).unwrap();
    assert_eq!(
        options.symbolic_assembly_backend,
        IvpSymbolicAssemblyBackend::AtomView
    );
    assert_eq!(
        options.lambdify_execution_policy,
        crate::symbolic::ivp_telemetry::IvpLambdifyExecutionPolicy::Auto { min_work: 32 }
    );
    assert_eq!(options.telemetry_mode, BdfTelemetryMode::Counters);
    assert_eq!(options.logging_level, SolverLogLevel::Warn);

    options.generated_backend_config.handoff_path = Some(PathBuf::from("unused-test-handoff.json"));
    options.generated_backend_config.build_policy = SymbolicIvpAotBuildPolicy::RequirePrebuilt;
    StiffPreset::lambdify().apply_bdf(&mut options).unwrap();
    assert!(options.generated_backend_config.resolver.is_none());
    assert!(options.generated_backend_config.handoff_path.is_none());
    assert_eq!(
        options.generated_backend_config.build_policy,
        SymbolicIvpAotBuildPolicy::UseIfAvailable
    );

    options.lambdify_execution_policy =
        crate::symbolic::ivp_telemetry::IvpLambdifyExecutionPolicy::Sequential;
    StiffPreset::aot_tcc("unused-test-aot")
        .apply_bdf(&mut options)
        .unwrap();
    assert_eq!(
        options.generated_backend_config.aot_c_compiler.as_deref(),
        Some("tcc")
    );
    assert!(matches!(
        options.generated_backend_config.build_policy,
        SymbolicIvpAotBuildPolicy::BuildIfMissing { .. }
    ));

    assert!(matches!(
        StiffPreset::default()
            .with_matrix_layout(StiffMatrixLayout::BandedAuto)
            .apply_bdf(&mut options),
        Err(StiffApiError::UnsupportedCapability {
            solver: StiffSolverKind::Bdf,
            ..
        })
    ));

    let aot_fallback = SymbolicIvpGeneratedBackendConfig::defaults();
    assert!(matches!(
        StiffPreset::default()
            .with_execution(StiffExecution::Aot(aot_fallback))
            .apply_bdf(&mut options),
        Err(StiffApiError::UnsupportedCapability {
            solver: StiffSolverKind::Bdf,
            ..
        })
    ));
    let StiffExecution::Aot(aot_config) =
        StiffPreset::aot_tcc("unused-test-aot").execution.unwrap()
    else {
        unreachable!("TCC profile always selects AOT")
    };
    assert!(matches!(
        StiffPreset::default()
            .with_callback_policy(StiffCallbackPolicy::Parallel { min_work: 16 })
            .with_execution(StiffExecution::Aot(aot_config))
            .apply_bdf(&mut options),
        Err(StiffApiError::UnsupportedCapability {
            solver: StiffSolverKind::Bdf,
            ..
        })
    ));
}

#[test]
fn lsode2_adapter_preserves_native_controller_and_execution_axes() {
    let mut config = lsode2_config();
    config.controller = crate::numerical::LSODE2::Lsode2ControllerConfig::automatic_adams_bdf();
    config.native_execution =
        crate::numerical::LSODE2::Lsode2NativeExecutionConfig::native_solve(128, 96);
    let solver = prepare_lsode2(config, Some(&StiffPreset::atom_sparse()))
        .expect("shared Atom/Sparse preset should preserve LSODE2-native policies");

    assert_eq!(
        solver.config().controller.mode,
        crate::numerical::LSODE2::Lsode2ControllerMode::AutomaticAdamsBdf
    );
    assert_eq!(
        solver.config().native_execution,
        crate::numerical::LSODE2::Lsode2NativeExecutionConfig::NativeSolve {
            max_step_attempts: 128,
            max_accepted_steps: 96,
        }
    );
    assert_eq!(
        solver.config().residual_jacobian_source,
        Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::AtomView,
            execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
        }
    );
    assert_eq!(
        solver.config().resolve_plan().structure,
        Lsode2LinearSystemStructure::Sparse
    );
    assert!(
        solver.config().backend.jacobian_backend
            != crate::numerical::LSODE2::Lsode2JacobianBackend::FiniteDifference
    );
}

#[test]
fn lsode2_banded_auto_is_explicit_and_symbolic_only_at_stiff_api_boundary() {
    let mut config = lsode2_config();
    StiffPreset::default()
        .with_matrix_layout(StiffMatrixLayout::BandedAuto)
        .apply_lsode2(&mut config)
        .unwrap();
    assert_eq!(
        config.linear_system_structure,
        Lsode2LinearSystemStructure::BandedAuto
    );
    assert_eq!(
        config.resolve_plan().linear_solver,
        crate::numerical::LSODE2::Lsode2LinearSolverChoice::LapackFaithfulBandedLu
    );

    let mut numeric = Lsode2ProblemConfig::new_numeric_fd(
        vec!["y".into()],
        "t".into(),
        0.0,
        DVector::from_element(1, 1.0),
        0.1,
        0.025,
        1e-7,
        1e-9,
        |_t, y| -y,
    );
    assert!(matches!(
        StiffPreset::default()
            .with_matrix_layout(StiffMatrixLayout::BandedAuto)
            .apply_lsode2(&mut numeric),
        Err(StiffApiError::UnsupportedCapability {
            solver: StiffSolverKind::Lsode2,
            ..
        })
    ));

    let mut conflicting = lsode2_config();
    conflicting.linear_solver_policy =
        Lsode2LinearSolverPolicy::Force(Lsode2LinearSolverChoice::DenseLu);
    assert!(matches!(
        StiffPreset::default()
            .with_matrix_layout(StiffMatrixLayout::BandedAuto)
            .apply_lsode2(&mut conflicting),
        Err(StiffApiError::UnsupportedCapability {
            solver: StiffSolverKind::Lsode2,
            ..
        })
    ));
}

#[test]
fn lsode2_adapter_forwards_telemetry_and_rust_logging_controls() {
    let preset = StiffPreset::default()
        .with_telemetry(SolverTelemetryMode::Detailed)
        .with_logging(SolverLogLevel::Warn);
    let solver = prepare_lsode2(lsode2_config(), Some(&preset)).unwrap();
    assert_eq!(
        solver.config().telemetry.mode(),
        crate::numerical::LSODE2::IvpTelemetryMode::Detailed
    );
    assert_eq!(solver.logging_level(), SolverLogLevel::Warn);
}

#[test]
fn lsode2_native_configuration_failure_keeps_its_typed_error_source() {
    let mut config = lsode2_config();
    config.controller.max_bdf_order = 0;
    let error = match prepare_lsode2(config, None) {
        Ok(_) => panic!("invalid LSODE2 controller should be rejected"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        StiffApiError::Lsode2(Lsode2Error::InvalidConfig(_))
    ));
    let native = Error::source(&error).expect("facade preserves the native LSODE2 error");
    assert!(native.downcast_ref::<Lsode2Error>().is_some());
    assert!(Error::source(native).is_none());
}

#[test]
fn lsode2_generated_backend_error_preserves_its_source_chain() {
    let backend = crate::symbolic::symbolic_ivp::IvpBackendError::GeneratedBackendFailure {
        message: "test backend failure".into(),
    };
    let error = Lsode2Error::GeneratedBackend(backend);
    let source = Error::source(&error).expect("backend cause remains discoverable");
    assert!(
        source
            .downcast_ref::<crate::symbolic::symbolic_ivp::IvpBackendError>()
            .is_some()
    );
}

#[test]
fn lsode2_rejects_nonproduction_prepared_atom_and_invalid_aot_fallback() {
    let mut config = lsode2_config();
    config.residual_jacobian_source = Lsode2ResidualJacobianSource::Symbolic {
        assembly: Lsode2SymbolicAssemblyBackend::AtomView,
        execution: Lsode2SymbolicExecutionMode::PreparedAtom,
    };
    assert!(matches!(
        prepare_lsode2(config, None),
        Err(StiffApiError::UnsupportedCapability {
            solver: StiffSolverKind::Lsode2,
            ..
        })
    ));

    let mut config = lsode2_config();
    let error = StiffPreset::aot_tcc("unused")
        .with_execution(StiffExecution::Aot(
            SymbolicIvpGeneratedBackendConfig::defaults(),
        ))
        .apply_lsode2(&mut config)
        .unwrap_err();
    assert!(matches!(
        error,
        StiffApiError::UnsupportedCapability {
            solver: StiffSolverKind::Lsode2,
            ..
        }
    ));
}

#[test]
fn lsode2_stiff_request_matches_direct_native_solution() {
    let mut config = lsode2_config();
    config.telemetry = crate::numerical::LSODE2::IvpTelemetry::counters();
    let facade_config = config.clone();
    let mut direct = Lsode2Solver::new(config.clone()).unwrap();
    direct.prepare().unwrap();
    direct.solve().unwrap();

    let mut through_facade = StiffSolverRequest::Lsode2 {
        config: facade_config,
        logging_level: None,
    }
    .prepare(None)
    .unwrap();
    assert_eq!(through_facade.kind(), StiffSolverKind::Lsode2);
    assert!(matches!(
        through_facade.try_solve(),
        Ok(StiffSolveOutcome::Lsode2(_))
    ));

    let (direct_times, direct_states) = direct.get_result();
    let facade_solver = through_facade.as_lsode2().unwrap();
    let (facade_times, facade_states) = facade_solver.get_result();
    assert_eq!(direct_times, facade_times);
    assert_eq!(direct_states.shape(), facade_states.shape());
    assert!(
        direct_states
            .iter()
            .zip(facade_states.iter())
            .all(|(a, b)| a == b)
    );
    assert_eq!(
        direct.telemetry_snapshot(),
        facade_solver.telemetry_snapshot()
    );
}

#[test]
fn lsode2_bridge_request_preserves_direct_bridge_telemetry_and_restart() {
    let config = lsode2_config()
        .with_bridge_solve()
        .with_telemetry(crate::numerical::LSODE2::IvpTelemetry::detailed());
    let mut direct = Lsode2Solver::new(config.clone()).unwrap();
    let mut dispatched = StiffSolverRequest::Lsode2 {
        config,
        logging_level: None,
    }
    .prepare(None)
    .unwrap();

    direct.prepare().unwrap();
    direct.solve().unwrap();
    assert!(matches!(
        dispatched.try_solve(),
        Ok(StiffSolveOutcome::Lsode2(_))
    ));

    assert_eq!(
        direct.get_result(),
        dispatched.as_lsode2().unwrap().get_result()
    );
    assert_eq!(
        direct.telemetry_snapshot(),
        dispatched.as_lsode2().unwrap().telemetry_snapshot()
    );
    let direct_native = direct.native_statistics();
    let facade_native = dispatched.as_lsode2().unwrap().native_statistics();
    assert_eq!(
        direct_native.bridge_solve_calls,
        facade_native.bridge_solve_calls
    );
    assert_eq!(
        direct_native.bridge_accepted_steps,
        facade_native.bridge_accepted_steps
    );
    assert_eq!(
        direct_native.algorithm_decision_calls,
        facade_native.algorithm_decision_calls
    );
    assert_eq!(
        direct_native.native_step_attempts,
        facade_native.native_step_attempts
    );
    assert_eq!(
        direct_native.native_step_accepts,
        facade_native.native_step_accepts
    );
    assert_eq!(
        direct_native.native_step_rejects_error_test,
        facade_native.native_step_rejects_error_test
    );
    assert_eq!(
        direct_native.native_step_rejects_nonlinear,
        facade_native.native_step_rejects_nonlinear
    );
    assert_eq!(
        direct_native.native_residual_calls,
        facade_native.native_residual_calls
    );
    assert_eq!(
        direct_native.native_jacobian_calls,
        facade_native.native_jacobian_calls
    );
    assert_eq!(
        direct_native.native_linear_solve_calls,
        facade_native.native_linear_solve_calls
    );

    direct
        .try_restart_with_initial_state(0.0, DVector::from_element(1, 2.0), 0.1)
        .unwrap();
    dispatched
        .as_lsode2_mut()
        .unwrap()
        .try_restart_with_initial_state(0.0, DVector::from_element(1, 2.0), 0.1)
        .unwrap();
    direct.solve().unwrap();
    assert!(matches!(
        dispatched.try_solve(),
        Ok(StiffSolveOutcome::Lsode2(_))
    ));
    assert_eq!(
        direct.get_result(),
        dispatched.as_lsode2().unwrap().get_result()
    );
}

#[test]
fn lsode2_native_limit_partial_result_survives_facade_restart() {
    let config = lsode2_config()
        .with_native_sparse_faer_backend()
        .with_faithful_bdf_solve(1, 1);
    let mut direct = Lsode2Solver::new(config.clone()).unwrap();
    let mut dispatched = StiffSolverRequest::Lsode2 {
        config,
        logging_level: None,
    }
    .prepare(None)
    .unwrap();

    direct.solve().unwrap();
    assert!(matches!(
        dispatched.try_solve(),
        Ok(StiffSolveOutcome::Lsode2(_))
    ));
    assert_eq!(direct.status(), "finished_native_faithful_partial");
    assert_eq!(direct.status(), dispatched.as_lsode2().unwrap().status());
    assert_eq!(
        direct.get_result(),
        dispatched.as_lsode2().unwrap().get_result()
    );
    assert!(!direct.get_result().0.is_empty());

    direct
        .try_restart_with_initial_state(0.0, DVector::from_element(1, 2.0), 0.1)
        .unwrap();
    dispatched
        .as_lsode2_mut()
        .unwrap()
        .try_restart_with_initial_state(0.0, DVector::from_element(1, 2.0), 0.1)
        .unwrap();
    direct.solve().unwrap();
    assert!(matches!(
        dispatched.try_solve(),
        Ok(StiffSolveOutcome::Lsode2(_))
    ));
    assert_eq!(
        direct.get_result(),
        dispatched.as_lsode2().unwrap().get_result()
    );
    assert_eq!(direct.get_result().1[(0, 0)], 2.0);
}

#[test]
fn solve_1d_preset_rejects_non_scalar_requests_before_native_preparation() {
    let preset = StiffPreset::solve_1d();
    let two_eq = vec![
        Expr::parse_expression("-y0"),
        Expr::parse_expression("-2*y1"),
    ];
    let options = BdfSolverOptions::for_bdf(
        two_eq.clone(),
        vec!["y0".into(), "y1".into()],
        "t".into(),
        0.0,
        DVector::from_element(2, 1.0),
        0.1,
        0.025,
        1e-7,
        1e-9,
        None,
        false,
        None,
    );
    assert!(matches!(
        prepare_bdf(options, Some(&preset)),
        Err(StiffApiError::DimensionMismatch {
            solver: StiffSolverKind::Bdf,
            expected: 1,
            actual: 2,
        })
    ));

    let mut lsode = lsode2_config();
    lsode.eq_system = two_eq;
    lsode.values = vec!["y0".into(), "y1".into()];
    lsode.y0 = DVector::from_element(2, 1.0);
    assert!(matches!(
        prepare_lsode2(lsode, Some(&preset)),
        Err(StiffApiError::DimensionMismatch {
            solver: StiffSolverKind::Lsode2,
            expected: 1,
            actual: 2,
        })
    ));

    let problem = crate::numerical::Radau::api::RadauProblem::new(
        vec![
            Expr::parse_expression("-y0"),
            Expr::parse_expression("-2*y1"),
        ],
        vec!["y0".into(), "y1".into()],
        "t",
    );
    assert!(matches!(
        prepare_radau(problem, RadauConfig::default(), Some(&preset)),
        Err(StiffApiError::DimensionMismatch {
            solver: StiffSolverKind::Radau,
            expected: 1,
            actual: 2,
        })
    ));

    let inconsistent_radau_problem = crate::numerical::Radau::api::RadauProblem::new(
        vec![Expr::parse_expression("-y0")],
        vec!["y0".into(), "y1".into()],
        "t",
    );
    assert!(matches!(
        prepare_radau(
            inconsistent_radau_problem,
            RadauConfig::default(),
            Some(&preset)
        ),
        Err(StiffApiError::DimensionMismatch {
            solver: StiffSolverKind::Radau,
            expected: 1,
            actual: 2,
        })
    ));

    let be = BeSolverOptions::new(
        vec![
            Expr::parse_expression("-y0"),
            Expr::parse_expression("-2*y1"),
        ],
        vec!["y0".into(), "y1".into()],
        "t".into(),
        1e-8,
        20,
        Some(0.01),
        0.0,
        0.1,
        DVector::from_element(2, 1.0),
    );
    assert!(matches!(
        prepare_backward_euler(be, Some(&preset)),
        Err(StiffApiError::DimensionMismatch {
            solver: StiffSolverKind::BackwardEuler,
            expected: 1,
            actual: 2,
        })
    ));
}

#[test]
fn backward_euler_does_not_claim_forced_lambdify() {
    let mut options = BeSolverOptions::new(
        vec![Expr::parse_expression("-y")],
        vec!["y".into()],
        "t".into(),
        1e-8,
        20,
        Some(0.01),
        0.0,
        1.0,
        DVector::from_element(1, 1.0),
    );
    assert!(matches!(
        StiffPreset::lambdify().apply_be(&mut options),
        Err(StiffApiError::UnsupportedCapability {
            solver: StiffSolverKind::BackwardEuler,
            ..
        })
    ));
}

#[test]
fn aot_tcc_profile_is_explicit_and_applies_to_symbolic_native_routes() {
    let preset = StiffPreset::aot_tcc("target/stiffapi-aot");
    let generated = match preset.execution.as_ref().unwrap() {
        StiffExecution::Aot(generated) => generated,
        StiffExecution::Lambdify => panic!("TCC profile must request AOT"),
    };
    assert_eq!(generated.aot_codegen_backend, AotCodegenBackend::C);
    assert_eq!(generated.aot_c_compiler.as_deref(), Some("tcc"));
    assert!(matches!(
        generated.build_policy,
        SymbolicIvpAotBuildPolicy::BuildIfMissing { .. }
    ));

    let mut radau = RadauConfig::default();
    preset.apply_radau(&mut radau).unwrap();
    assert_eq!(radau.execution, RadauExecution::Aot);
    assert!(radau.aot.is_some());

    let mut be = BeSolverOptions::new(
        vec![Expr::parse_expression("-y")],
        vec!["y".into()],
        "t".into(),
        1e-8,
        20,
        Some(0.01),
        0.0,
        1.0,
        DVector::from_element(1, 1.0),
    );
    preset.apply_be(&mut be).unwrap();
    assert_eq!(
        be.generated_backend_config.aot_c_compiler.as_deref(),
        Some("tcc")
    );

    let mut lsode2 = lsode2_config();
    preset.apply_lsode2(&mut lsode2).unwrap();
    assert_eq!(
        lsode2.residual_jacobian_source,
        Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
            execution: Lsode2SymbolicExecutionMode::Aot {
                toolchain: Lsode2AotToolchain::CTcc,
                profile: Lsode2AotProfile::Release,
            },
        }
    );
    assert_eq!(
        lsode2.backend.generated_backend.aot_c_compiler.as_deref(),
        Some("tcc")
    );
}

#[test]
fn radau_delegate_preserves_the_native_typed_error() {
    let problem = crate::numerical::Radau::api::RadauProblem::new(
        vec![Expr::parse_expression("-y")],
        vec!["y".into()],
        "t",
    );
    let result = prepare_radau(
        problem,
        RadauConfig {
            t_bound: 0.0,
            ..RadauConfig::default()
        },
        None,
    );
    let error = match result {
        Ok(_) => panic!("invalid Radau interval should be rejected"),
        Err(error) => error,
    };
    assert!(matches!(error, StiffApiError::Radau(_)));
    assert!(Error::source(&error).is_some());
}

#[test]
fn backward_euler_delegate_constructs_the_native_solver() {
    let options = BeSolverOptions::new(
        vec![Expr::parse_expression("-y")],
        vec!["y".into()],
        "t".into(),
        1e-8,
        20,
        Some(0.01),
        0.0,
        0.1,
        DVector::from_element(1, 1.0),
    );
    assert!(prepare_backward_euler(options, Some(&StiffPreset::expr_dense())).is_ok());
}

#[test]
fn dispatched_native_solvers_keep_solve_and_continuation_behavior() {
    let problem = crate::numerical::Radau::api::RadauProblem::new(
        vec![Expr::parse_expression("-rate*y")],
        vec!["y".into()],
        "t",
    )
    .with_parameter_names(vec!["rate".into()]);
    let config = RadauConfig {
        t_bound: 0.2,
        first_step: Some(0.02),
        max_step: 0.05,
        telemetry: crate::numerical::Radau::api::RadauTelemetryMode::Counters,
        ..RadauConfig::default()
    };
    let mut direct_radau = RadauSolver::prepare(problem.clone(), config.clone()).unwrap();
    let mut dispatched_radau = StiffSolverRequest::Radau {
        problem,
        config,
        initial_state: vec![1.0],
        parameters: vec![0.7],
    }
    .prepare(Some(&StiffPreset::default()))
    .unwrap();
    let direct_solution = direct_radau
        .try_solve_with_parameter_values(&[1.0], &[0.7])
        .unwrap();
    let facade_solution = match dispatched_radau.try_solve().unwrap() {
        StiffSolveOutcome::Radau(solution) => solution,
        other => panic!("expected Radau outcome, got {other:?}"),
    };
    assert!((direct_solution.y[0] - facade_solution.y[0]).abs() < 1e-13);
    assert_eq!(
        direct_solution.telemetry().counters["accepted_steps"],
        facade_solution.telemetry().counters["accepted_steps"]
    );
    assert_eq!(direct_solution.telemetry(), facade_solution.telemetry());
    let direct_continuation = direct_radau
        .try_solve_with_parameter_values(&[1.0], &[1.1])
        .unwrap();
    let facade_continuation = dispatched_radau
        .as_radau_mut()
        .unwrap()
        .try_solve_with_parameter_values(&[1.0], &[1.1])
        .unwrap();
    assert!((direct_continuation.y[0] - facade_continuation.y[0]).abs() < 1e-13);

    let make_be_options = || {
        BeSolverOptions::new(
            vec![Expr::parse_expression("-0.7*y")],
            vec!["y".into()],
            "t".into(),
            1e-9,
            20,
            Some(0.01),
            0.0,
            0.1,
            DVector::from_element(1, 1.0),
        )
        .with_telemetry_mode(BeTelemetryMode::Counters)
    };
    let mut direct_be = BE::try_new_with_options(make_be_options()).unwrap();
    let mut dispatched_be = StiffSolverRequest::BackwardEuler {
        options: make_be_options(),
        equation_parameters: Vec::new(),
        parameter_values: Vec::new(),
    }
    .prepare(Some(&StiffPreset::default()))
    .unwrap();
    direct_be.try_solve().unwrap();
    assert!(matches!(
        dispatched_be.try_solve(),
        Ok(StiffSolveOutcome::BackwardEuler)
    ));
    let direct_states = direct_be.trajectory().1;
    let direct_final = direct_states[(direct_states.nrows() - 1, 0)];
    let facade_be = dispatched_be.as_backward_euler().unwrap();
    let facade_states = facade_be.trajectory().1;
    let facade_final = facade_states[(facade_states.nrows() - 1, 0)];
    assert!((direct_final - facade_final).abs() < 1e-13);
    assert_eq!(
        direct_be.detailed_statistics().accepted_steps,
        facade_be.detailed_statistics().accepted_steps
    );
    assert_eq!(
        direct_be.detailed_statistics(),
        facade_be.detailed_statistics()
    );
    assert_eq!(
        direct_be.get_statistics().accepted_steps_total,
        facade_be.get_statistics().accepted_steps_total
    );
    assert_eq!(
        direct_be.get_statistics().solve_calls,
        facade_be.get_statistics().solve_calls
    );

    direct_be.try_continue_to(0.2).unwrap();
    dispatched_be
        .as_backward_euler_mut()
        .unwrap()
        .try_continue_to(0.2)
        .unwrap();
    let direct_states = direct_be.trajectory().1;
    let direct_final = direct_states[(direct_states.nrows() - 1, 0)];
    let facade_be = dispatched_be.as_backward_euler().unwrap();
    let facade_states = facade_be.trajectory().1;
    let facade_final = facade_states[(facade_states.nrows() - 1, 0)];
    assert!((direct_final - facade_final).abs() < 1e-13);
}

#[test]
fn fresh_parameterized_solve_shortcuts_reset_state_and_reuse_preparation() {
    let make_bdf = || {
        BdfSolverOptions::for_bdf(
            vec![Expr::parse_expression("-rate*y")],
            vec!["y".into()],
            "t".into(),
            0.0,
            DVector::from_element(1, 1.0),
            0.1,
            0.02,
            1e-8,
            1e-10,
            None,
            false,
            None,
        )
        .with_equation_parameters(vec!["rate".into()])
        .with_equation_parameter_values(DVector::from_element(1, 1.0))
    };
    let make_be = || {
        BeSolverOptions::new(
            vec![Expr::parse_expression("-rate*y")],
            vec!["y".into()],
            "t".into(),
            1e-9,
            20,
            Some(0.001),
            0.0,
            0.1,
            DVector::from_element(1, 1.0),
        )
    };
    let make_lsode2 = || {
        Lsode2ProblemConfig::new(
            vec![Expr::parse_expression("-rate*y")],
            vec!["y".into()],
            "t".into(),
            0.0,
            DVector::from_element(1, 1.0),
            0.1,
            0.01,
            1e-8,
            1e-10,
        )
        .with_equation_parameters(vec!["rate".into()])
        .with_equation_parameter_values(DVector::from_element(1, 1.0))
    };

    let mut bdf = StiffSolverRequest::Bdf {
        options: make_bdf(),
    }
    .prepare(None)
    .unwrap();
    let mut radau = StiffSolverRequest::Radau {
        problem: crate::numerical::Radau::api::RadauProblem::new(
            vec![Expr::parse_expression("-rate*y")],
            vec!["y".into()],
            "t",
        )
        .with_parameter_names(vec!["rate".into()]),
        config: RadauConfig {
            t_bound: 0.1,
            ..RadauConfig::default()
        },
        initial_state: vec![1.0],
        parameters: vec![1.0],
    }
    .prepare(None)
    .unwrap();
    let mut be = StiffSolverRequest::BackwardEuler {
        options: make_be(),
        equation_parameters: vec!["rate".into()],
        parameter_values: vec![1.0],
    }
    .prepare(None)
    .unwrap();
    let mut lsode2 = StiffSolverRequest::Lsode2 {
        config: make_lsode2(),
        logging_level: None,
    }
    .prepare(None)
    .unwrap();

    let first = DVector::from_element(1, 2.0);
    let first_parameter = DVector::from_element(1, 1.5);
    let first_expected = 2.0 * (-1.5_f64 * 0.1).exp();
    for solver in [&mut bdf, &mut radau, &mut be, &mut lsode2] {
        let outcome = solver
            .try_solve_with_parameter_values(first.clone(), first_parameter.clone())
            .unwrap();
        let terminal = match (&*solver, outcome) {
            (PreparedStiffSolver::Bdf(native), StiffSolveOutcome::Bdf) => {
                let states = native.get_result_ref().1;
                states[(states.nrows() - 1, 0)]
            }
            (PreparedStiffSolver::Radau(_), StiffSolveOutcome::Radau(solution)) => solution.y[0],
            (PreparedStiffSolver::BackwardEuler(native), StiffSolveOutcome::BackwardEuler) => {
                let states = native.trajectory().1;
                states[(states.nrows() - 1, 0)]
            }
            (PreparedStiffSolver::Lsode2(native), StiffSolveOutcome::Lsode2(_)) => {
                let states = native.get_result_ref().1;
                states[(states.nrows() - 1, 0)]
            }
            (_, other) => panic!("solver returned a mismatched outcome: {other:?}"),
        };
        assert!((terminal - first_expected).abs() < 2e-3);
    }

    let second = DVector::from_element(1, 3.0);
    let second_parameter = DVector::from_element(1, 0.8);
    for solver in [&mut bdf, &mut radau, &mut be, &mut lsode2] {
        solver
            .try_solve_with_parameter_values_in_interval(
                0.2,
                second.clone(),
                0.4,
                second_parameter.clone(),
            )
            .unwrap();
    }

    let expected = 3.0 * (-0.8_f64 * 0.2).exp();
    let final_value = |solver: &PreparedStiffSolver| match solver {
        PreparedStiffSolver::Bdf(solver) => {
            solver.get_result_ref().1[(solver.get_result_ref().1.nrows() - 1, 0)]
        }
        PreparedStiffSolver::Radau(_) => panic!("Radau returns its solution from the outcome"),
        PreparedStiffSolver::BackwardEuler(solver) => {
            let states = solver.trajectory().1;
            states[(states.nrows() - 1, 0)]
        }
        PreparedStiffSolver::Lsode2(solver) => {
            let states = solver.get_result_ref().1;
            states[(states.nrows() - 1, 0)]
        }
    };
    assert!((final_value(&bdf) - expected).abs() < 2e-4);
    assert!((final_value(&be) - expected).abs() < 2e-3);
    assert!((final_value(&lsode2) - expected).abs() < 2e-4);
    assert_eq!(bdf.as_bdf().unwrap().get_result_ref().0[0], 0.2);
    assert_eq!(be.as_backward_euler().unwrap().trajectory().0[0], 0.2);
    assert_eq!(lsode2.as_lsode2().unwrap().get_result_ref().0[0], 0.2);

    let radau_solution = match radau.try_solve().unwrap() {
        StiffSolveOutcome::Radau(solution) => solution,
        other => panic!("expected Radau outcome, got {other:?}"),
    };
    assert!((radau_solution.y[0] - expected).abs() < 2e-7);
}

#[test]
fn bdf_dispatch_matches_native_solve_and_discrete_statistics() {
    let make_options = || {
        BdfSolverOptions::for_bdf(
            vec![Expr::parse_expression("-0.7*y")],
            vec!["y".into()],
            "t".into(),
            0.0,
            DVector::from_element(1, 1.0),
            0.1,
            0.025,
            1e-7,
            1e-9,
            None,
            false,
            None,
        )
        .with_telemetry_mode(BdfTelemetryMode::Counters)
    };

    let mut direct = BdfSolver::new_with_options(make_options());
    direct.try_generate().unwrap();
    let mut dispatched = StiffSolverRequest::Bdf {
        options: make_options(),
    }
    .prepare(Some(&StiffPreset::default()))
    .unwrap();
    assert_eq!(dispatched.kind(), StiffSolverKind::Bdf);
    assert!(dispatched.as_bdf().unwrap().is_backend_prepared());

    direct.try_solve().unwrap();
    assert!(matches!(dispatched.try_solve(), Ok(StiffSolveOutcome::Bdf)));

    let (_, direct_states) = direct.get_result_ref();
    let (_, facade_states) = dispatched.as_bdf().unwrap().get_result_ref();
    let direct_final = direct_states[(direct_states.nrows() - 1, 0)];
    let facade_final = facade_states[(facade_states.nrows() - 1, 0)];
    assert!((direct_final - facade_final).abs() < 1e-13);
    let direct_stats = direct.get_statistics();
    let facade_stats = dispatched.as_bdf().unwrap().get_statistics();
    assert_eq!(
        direct_stats.accepted_steps_total,
        facade_stats.accepted_steps_total
    );
    assert_eq!(direct_stats.bdf_nfev_total, facade_stats.bdf_nfev_total);
    assert_eq!(direct_stats.bdf_njev_total, facade_stats.bdf_njev_total);
    assert_eq!(direct_stats.bdf_nlu_total, facade_stats.bdf_nlu_total);

    direct
        .try_restart_with_initial_state(0.0, DVector::from_element(1, 2.0), 0.1)
        .unwrap();
    dispatched
        .as_bdf_mut()
        .unwrap()
        .try_restart_with_initial_state(0.0, DVector::from_element(1, 2.0), 0.1)
        .unwrap();
    direct.try_solve().unwrap();
    assert!(matches!(dispatched.try_solve(), Ok(StiffSolveOutcome::Bdf)));
    let (_, direct_states) = direct.get_result_ref();
    let (_, facade_states) = dispatched.as_bdf().unwrap().get_result_ref();
    assert_eq!(direct_states.shape(), facade_states.shape());
    assert!(
        direct_states
            .iter()
            .zip(facade_states.iter())
            .all(|(a, b)| a == b)
    );
}
