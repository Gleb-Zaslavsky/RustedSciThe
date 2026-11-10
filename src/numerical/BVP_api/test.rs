use super::{
    BvpApiError, BvpApiTelemetryMode, BvpBoundaryCondition, BvpBoundarySide, BvpNativeSolver,
    BvpProblem, BvpSciFreeParameters, BvpSciSparseFactorization, BvpSettingsError, BvpSolution,
    BvpSolver, BvpSolverConfig, BvpSolverSettings,
};
use crate::Utils::postprocessing::{PostprocessError, PostprocessPlan};
use crate::numerical::BVP_CorrectionNewton::CorrectionNewtonOptions;
use crate::numerical::BVP_Damp::NR_Damp_solver_damped::DampedSolverOptions;
use crate::numerical::BVP_Damp::NR_Damp_solver_frozen::FrozenSolverOptions;
use crate::numerical::BVP_Damp::telemetry::BvpSolveStatus;
use crate::numerical::BVP_Damp::telemetry::{BvpLoggingConfig, BvpLoggingMode, BvpTelemetryMode};
use crate::numerical::BVP_sci::new::BvpSciSparseFactorization as NativeBvpSciSparseFactorization;
use crate::numerical::BVP_sci::new::{
    BvpSciAssembly, BvpSciExecution, BvpSciMatrixLayout, BvpSciOptions, BvpSciOutputPolicy,
    BvpSciTelemetryMode,
};
use crate::numerical::logging::SolverLogLevel;
use crate::symbolic::bvp::aot_telemetry::BvpAotTelemetryMode;
use crate::symbolic::bvp::legacy::BvpBackendIntegrationError;
use crate::symbolic::bvp::legacy::BvpSymbolicAssemblyBackend;
use crate::symbolic::bvp::telemetry::{BvpLambdifyExecutionPolicy, BvpLambdifyTelemetryMode};
use crate::symbolic::symbolic_engine::Expr;
use nalgebra::DMatrix;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[path = "correction_newton_continuation_tests.rs"]
mod correction_newton_continuation;

fn valid_problem() -> BvpProblem {
    BvpProblem::new(
        vec![Expr::parse_expression("-y")],
        vec!["y".into()],
        "x",
        vec![0.0, 0.5, 1.0],
        vec![BvpBoundaryCondition {
            state: 0,
            side: BvpBoundarySide::Left,
            value: 1.0,
        }],
    )
    .expect("valid BVP should construct")
}

#[test]
fn damped_facade_exposes_opt_in_decision_trace() {
    let mut solver = BvpSolver::new(
        valid_problem(),
        BvpSolverConfig::Damped {
            options: DampedSolverOptions::default()
                .with_bounds(HashMap::from([("y".to_string(), (-10.0, 10.0))]))
                .with_rel_tolerance(HashMap::from([("y".to_string(), 1e-6)]))
                .with_bvp_logging_config(BvpLoggingConfig::new(BvpLoggingMode::Detailed)),
            initial_guess: DMatrix::from_element(1, 2, 1.0),
        },
    )
    .expect("Damped facade should construct");
    solver.solve().expect("simple Damped problem should solve");

    let trace = solver
        .damped_decision_trace_snapshot()
        .expect("Damped facade exposes the native decision trace");
    assert!(!trace.events.is_empty());
    assert_eq!(trace.dropped_events, 0);
}

#[test]
fn correction_newton_facade_routes_solution_profile_postprocessing_and_native_telemetry() {
    assert!(matches!(
        BvpSolverSettings::CorrectionNewton(CorrectionNewtonOptions::default())
            .with_telemetry_mode(BvpApiTelemetryMode::Timings),
        Err(BvpSettingsError::UnsupportedTelemetry {
            solver: "BVP_CorrectionNewton",
            mode: BvpApiTelemetryMode::Timings,
            ..
        })
    ));
    let problem = BvpProblem::new(
        vec![Expr::parse_expression("0")],
        vec!["y".into()],
        "x",
        vec![0.0, 0.5, 1.0],
        vec![BvpBoundaryCondition {
            state: 0,
            side: BvpBoundarySide::Left,
            value: 1.0,
        }],
    )
    .expect("CorrectionNewton fixture should be a valid BVP");
    let settings = BvpSolverSettings::CorrectionNewton(CorrectionNewtonOptions::default())
        .with_telemetry_mode(BvpApiTelemetryMode::Detailed)
        .expect("CorrectionNewton supports detailed native telemetry")
        .with_initial_guess(DMatrix::from_element(1, 2, 1.0))
        .with_logging_level(SolverLogLevel::Off);
    let mut solver = BvpSolver::new(problem, settings)
        .expect("CorrectionNewton facade route should prepare through its native adapter");
    assert!(matches!(
        solver.telemetry_snapshot(),
        super::BvpSolverTelemetrySnapshot::CorrectionNewton(None)
    ));

    solver.solve().expect("constant profile should solve");
    let view = solver
        .solution_view()
        .expect("successful result is published");
    assert_eq!(view.mesh(), &[0.0, 0.5, 1.0]);
    assert_eq!(view.state_count(), 1);
    for node in 0..view.mesh().len() {
        assert!((view.value(node, 0).unwrap() - 1.0).abs() < 1e-12);
    }
    let BvpSolution::CorrectionNewton { report, .. } = solver.solution().unwrap() else {
        panic!("facade must retain the CorrectionNewton-native result report");
    };
    assert_eq!(
        report.linear_backend,
        crate::numerical::BVP_CorrectionNewton::CorrectionNewtonLinearBackend::Dense
    );
    assert!(report.telemetry.residual_evaluations > 0);
    let telemetry = solver.telemetry_snapshot();
    let super::BvpSolverTelemetrySnapshot::CorrectionNewton(Some(telemetry)) = telemetry else {
        panic!("facade must preserve CorrectionNewton telemetry fields");
    };
    assert!(telemetry.residual_evaluations > 0);
    let dataset = solver
        .postprocess_dataset()
        .expect("CorrectionNewton profile should feed the common postprocessing dataset");
    assert_eq!(dataset.axis.len(), 3);
}

#[test]
fn boundary_value_rebind_reuses_correction_newton_symbolic_route() {
    let problem = BvpProblem::new(
        vec![Expr::parse_expression("0")],
        vec!["y".into()],
        "x",
        vec![0.0, 0.5, 1.0],
        vec![BvpBoundaryCondition {
            state: 0,
            side: BvpBoundarySide::Left,
            value: 1.0,
        }],
    )
    .unwrap();
    let config = BvpSolverSettings::CorrectionNewton(CorrectionNewtonOptions::default())
        .with_initial_guess(DMatrix::from_element(1, 2, 1.0));
    let mut solver = BvpSolver::new(problem, config).unwrap();
    solver.solve().unwrap();
    solver
        .solve_with_boundary_values(vec![2.0])
        .expect("same-schema boundary values should rebind");
    let solution = solver.solution_view().unwrap();
    assert!((solution.value(0, 0).unwrap() - 2.0).abs() < 1e-14);
    assert!((solution.value(2, 0).unwrap() - 2.0).abs() < 1e-14);
}

#[test]
fn boundary_value_rebind_reuses_bvp_sci_boundary_callback() {
    let mesh = vec![0.0, 0.5, 1.0];
    let problem = BvpProblem::new(
        vec![Expr::parse_expression("0")],
        vec!["y".into()],
        "x",
        mesh.clone(),
        vec![BvpBoundaryCondition {
            state: 0,
            side: BvpBoundarySide::Left,
            value: 1.0,
        }],
    )
    .unwrap();
    let config = BvpSolverSettings::Sci {
        options: BvpSciOptions::default()
            .with_execution(BvpSciExecution::Lambdify)
            .with_assembly(BvpSciAssembly::ExprLegacy),
        sparse_factorization: BvpSciSparseFactorization::default(),
        aot_config: None,
    }
    .with_initial_guess(DMatrix::from_element(1, mesh.len(), 1.0));
    let mut solver = BvpSolver::new(problem, config).unwrap();
    solver.solve().unwrap();
    solver
        .solve_with_boundary_values(vec![2.0])
        .expect("same-schema boundary values should rebind");
    let solution = solver.solution_view().unwrap();
    for node in 0..mesh.len() {
        assert!((solution.value(node, 0).unwrap() - 2.0).abs() < 1e-8);
    }
}

#[test]
fn boundary_value_rebind_reuses_damped_and_frozen_symbolic_callbacks() {
    for frontend in [
        BvpSymbolicAssemblyBackend::ExprLegacy,
        BvpSymbolicAssemblyBackend::AtomView,
    ] {
        let problem = BvpProblem::new(
            vec![Expr::parse_expression("-a*y")],
            vec!["y".into()],
            "x",
            vec![0.0, 0.5, 1.0],
            vec![BvpBoundaryCondition {
                state: 0,
                side: BvpBoundarySide::Left,
                value: 1.0,
            }],
        )
        .expect("symbolic boundary-rebind fixture should construct");

        let generated = crate::numerical::BVP_Damp::generated_solver_handoff::GeneratedBackendConfig::banded_lambdify_defaults()
            .with_symbolic_assembly_backend(frontend);
        let damped_options = DampedSolverOptions::banded_damped()
            .with_banded_lambdify()
            .with_bounds(HashMap::from([("y".to_string(), (-10.0, 10.0))]))
            .with_rel_tolerance(HashMap::from([("y".to_string(), 1e-6)]))
            .with_generated_backend_config(generated.clone());
        let mut damped = BvpSolver::new(
            problem.clone(),
            BvpSolverConfig::Damped {
                options: damped_options,
                initial_guess: DMatrix::from_element(1, 2, 1.0),
            },
        )
        .expect("Damped symbolic route should prepare");
        damped
            .bind_parameters(vec!["a".into()], vec![1.0])
            .expect("Damped external evaluator parameter should bind");
        damped.solve().expect("initial Damped solve should succeed");
        assert!(damped.set_boundary_values(vec![f64::NAN]).is_err());
        assert!(
            damped.solution().is_some(),
            "invalid Damped rebind must preserve the accepted result"
        );
        damped
            .solve_with_boundary_values(vec![2.0])
            .expect("Damped should rebind the numeric boundary value");
        let damped_result = damped.solution_view().expect("rebound result is published");
        assert!((damped_result.value(0, 0).unwrap() - 2.0).abs() < 1e-10);
        let damped_right_a1 = damped_result.value(2, 0).unwrap();
        damped
            .set_parameter_values(vec![2.0])
            .expect("Damped external parameter should remain independently rebindable");
        damped
            .solve()
            .expect("Damped should solve after both rebinds");
        let damped_right_a2 = damped.solution_view().unwrap().value(2, 0).unwrap();
        assert!((damped_right_a1 - damped_right_a2).abs() > 1e-5);

        let frozen_options = FrozenSolverOptions::default()
            .with_bounds(HashMap::from([("y".to_string(), (-10.0, 10.0))]))
            .with_generated_backend_config(generated);
        let mut frozen = BvpSolver::new(
            problem,
            BvpSolverConfig::Frozen {
                options: frozen_options,
                initial_guess: DMatrix::from_element(1, 2, 1.0),
            },
        )
        .expect("Frozen symbolic route should prepare");
        frozen
            .bind_parameters(vec!["a".into()], vec![1.0])
            .expect("Frozen external evaluator parameter should bind");
        frozen.solve().expect("initial Frozen solve should succeed");
        assert!(frozen.set_boundary_values(vec![f64::NAN]).is_err());
        assert!(
            frozen.solution().is_some(),
            "invalid Frozen rebind must preserve the accepted result"
        );
        frozen
            .solve_with_boundary_values(vec![2.0])
            .expect("Frozen should rebind the numeric boundary value");
        let frozen_result = frozen.solution_view().expect("rebound result is published");
        assert!((frozen_result.value(0, 0).unwrap() - 2.0).abs() < 1e-10);
        let frozen_right_a1 = frozen_result.value(2, 0).unwrap();
        frozen
            .set_parameter_values(vec![2.0])
            .expect("Frozen external parameter should remain independently rebindable");
        frozen
            .solve()
            .expect("Frozen should solve after both rebinds");
        let frozen_right_a2 = frozen.solution_view().unwrap().value(2, 0).unwrap();
        assert!((frozen_right_a1 - frozen_right_a2).abs() > 1e-5);
    }
}

#[test]
fn correction_newton_facade_routes_symbolic_sparse_and_automatic_banded_backends() {
    for frontend in [
        crate::numerical::BVP_CorrectionNewton::CorrectionNewtonSymbolicFrontend::ExprLegacy,
        crate::numerical::BVP_CorrectionNewton::CorrectionNewtonSymbolicFrontend::AtomView,
    ] {
        for (layout, expected) in [
            (
                crate::numerical::BVP_CorrectionNewton::CorrectionNewtonMatrixLayout::Sparse,
                crate::numerical::BVP_CorrectionNewton::CorrectionNewtonLinearBackend::Sparse,
            ),
            (
                crate::numerical::BVP_CorrectionNewton::CorrectionNewtonMatrixLayout::BandedAuto,
                crate::numerical::BVP_CorrectionNewton::CorrectionNewtonLinearBackend::Banded {
                    lower: 1,
                    upper: 0,
                },
            ),
        ] {
            let problem = BvpProblem::new(
                vec![Expr::parse_expression("-y")],
                vec!["y".into()],
                "x",
                vec![0.0, 0.25, 0.5, 0.75, 1.0],
                vec![BvpBoundaryCondition {
                    state: 0,
                    side: BvpBoundarySide::Left,
                    value: 1.0,
                }],
            )
            .expect("symbolic fixed-mesh problem should validate");
            let options = CorrectionNewtonOptions {
                symbolic_frontend: frontend,
                matrix_layout: layout,
                ..CorrectionNewtonOptions::default()
            };
            let settings = BvpSolverSettings::CorrectionNewton(options)
                .with_initial_guess(DMatrix::from_element(1, 4, 1.0));
            let mut solver = BvpSolver::new(problem, settings)
                .expect("CorrectionNewton facade should prepare structural backends");
            solver.solve().expect("linear BVP should converge");
            let BvpSolution::CorrectionNewton { report, .. } = solver.solution().unwrap() else {
                panic!("expected a CorrectionNewton result");
            };
            assert_eq!(report.linear_backend, expected);
            let final_value = solver.solution_view().unwrap().value(4, 0).unwrap();
            let trapezoid_ratio: f64 = (1.0 - 0.5 / 4.0) / (1.0 + 0.5 / 4.0);
            assert!((final_value - trapezoid_ratio.powi(4)).abs() < 1e-12);
        }
    }
}

#[test]
fn correction_newton_facade_rebinds_fixed_inputs_without_rebuilding_symbolic_callbacks() {
    let problem = BvpProblem::new(
        vec![Expr::parse_expression("-(rate + offset)*y")],
        vec!["y".into()],
        "x",
        (0..17).map(|index| index as f64 / 16.0).collect(),
        vec![BvpBoundaryCondition {
            state: 0,
            side: BvpBoundarySide::Left,
            value: 1.0,
        }],
    )
    .unwrap()
    .with_external_parameters(vec!["rate".into(), "offset".into()], vec![0.4, 0.1])
    .unwrap();
    let mut solver = BvpSolver::new(
        problem,
        BvpSolverConfig::CorrectionNewton {
            options: CorrectionNewtonOptions::default(),
            initial_guess: DMatrix::from_element(1, 16, 1.0),
        },
    )
    .expect("CorrectionNewton facade should prepare fixed symbolic inputs");

    let first = solver.solve().unwrap().clone();
    let BvpSolution::CorrectionNewton { values: first, .. } = first else {
        panic!("CorrectionNewton facade should preserve its route result");
    };
    solver
        .set_parameter_values(vec![0.5, 0.1])
        .expect("fixed inputs should be rebound without preparing callbacks");
    let second = solver.solve().unwrap().clone();
    let BvpSolution::CorrectionNewton { values: second, .. } = second else {
        panic!("CorrectionNewton facade should preserve its route result");
    };
    assert!((first[(16, 0)] - (-0.5_f64).exp()).abs() < 2e-3);
    assert!((second[(16, 0)] - (-0.6_f64).exp()).abs() < 2e-3);
    assert!(second[(16, 0)] < first[(16, 0)]);
}

#[test]
fn correction_newton_facade_exposes_partial_telemetry_without_publishing_a_failed_profile() {
    let problem = BvpProblem::new(
        vec![Expr::parse_expression("1")],
        vec!["y".into()],
        "x",
        vec![0.0, 0.5, 1.0],
        vec![BvpBoundaryCondition {
            state: 0,
            side: BvpBoundarySide::Left,
            value: 0.0,
        }],
    )
    .expect("bounded CorrectionNewton fixture should be a valid BVP");
    let mut options = CorrectionNewtonOptions::default();
    options.bounds.insert("y".into(), (0.0, 0.0));
    let settings = BvpSolverSettings::CorrectionNewton(options)
        .with_telemetry_mode(BvpApiTelemetryMode::Counters)
        .expect("counter telemetry is supported")
        .with_initial_guess(DMatrix::zeros(1, 2));
    let mut solver =
        BvpSolver::new(problem, settings).expect("bounded CorrectionNewton route should prepare");

    let error = solver
        .solve()
        .expect_err("the required positive profile is blocked by the zero-width bound");
    match error {
        BvpApiError::CorrectionNewton(
            crate::numerical::BVP_CorrectionNewton::CorrectionNewtonSolverError::BoundsBlocked {
                report,
            },
        ) => {
            assert!(report.telemetry.jacobian_evaluations > 0);
            assert!(report.telemetry.linear_solves > 0);
        }
        other => panic!("expected typed CorrectionNewton bounds failure, got {other:?}"),
    }
    assert!(solver.solution().is_none());
    let super::BvpSolverTelemetrySnapshot::CorrectionNewton(Some(snapshot)) =
        solver.telemetry_snapshot()
    else {
        panic!("facade must retain telemetry from the failed native attempt");
    };
    assert!(snapshot.residual_evaluations > 0);
    assert!(snapshot.jacobian_evaluations > 0);
}

#[test]
fn native_solver_defaults_are_distinct_and_facade_preserves_them() {
    assert_eq!(
        NativeBvpSciSparseFactorization::from(BvpSciSparseFactorization::StandardFaer),
        NativeBvpSciSparseFactorization::StandardFaer
    );
    assert_eq!(
        NativeBvpSciSparseFactorization::from(BvpSciSparseFactorization::SuperLu),
        NativeBvpSciSparseFactorization::SuperLu
    );

    let damped = DampedSolverOptions::default();
    assert_eq!(damped.scheme, "forward");
    assert_eq!(damped.strategy, "Damped");
    assert_eq!(damped.method, "Sparse");
    assert_eq!(damped.abs_tolerance, 1e-6);
    assert_eq!(damped.max_iterations, 25);
    assert!(damped.bounds.is_none());
    assert!(damped.rel_tolerance.is_none());
    let strategy = damped.strategy_params.as_ref().unwrap();
    assert_eq!(strategy.max_jac, Some(3));
    assert_eq!(strategy.max_damp_iter, Some(7));
    assert_eq!(strategy.damp_factor, Some(0.5));

    let damped_generated = &damped.generated_backend_config;
    assert_eq!(
        damped_generated.symbolic_assembly_backend,
        BvpSymbolicAssemblyBackend::AtomView
    );
    assert_eq!(damped_generated.aot_c_compiler.as_deref(), Some("tcc"));
    assert_eq!(
        damped_generated.atom_optimization_profile,
        crate::symbolic::codegen::CodegenIR::AtomOptimizationProfile::Full
    );
    assert_eq!(
        damped_generated.lambdify_telemetry_mode,
        BvpLambdifyTelemetryMode::Off
    );
    assert_eq!(
        damped_generated.aot_telemetry_mode,
        BvpAotTelemetryMode::Off
    );
    assert_eq!(
        damped_generated.bvp_telemetry_mode,
        BvpTelemetryMode::Counters
    );
    assert_eq!(damped_generated.bvp_logging_mode, BvpLoggingMode::Off);
    assert_eq!(
        damped_generated.lambdify_execution_policy,
        BvpLambdifyExecutionPolicy::Parallel { min_work: 0 }
    );

    let frozen = FrozenSolverOptions::default();
    assert_eq!(frozen.scheme, "forward");
    assert_eq!(frozen.strategy, "Frozen");
    assert_eq!(frozen.method, "Sparse");
    assert_eq!(frozen.tolerance, 1e-6);
    assert_eq!(frozen.max_iterations, 25);
    assert!(
        frozen
            .strategy_params
            .as_ref()
            .is_some_and(|params| params.contains_key("Frozen_naive"))
    );
    assert_eq!(
        frozen.generated_backend_config.symbolic_assembly_backend,
        BvpSymbolicAssemblyBackend::AtomView
    );

    let sci = BvpSciOptions::default();
    assert_eq!(sci.execution, BvpSciExecution::Numerical);
    assert_eq!(sci.assembly, None);
    assert_eq!(
        sci.matrix_layout,
        crate::numerical::BVP_sci::new::BvpSciMatrixLayout::Dense
    );
    assert_eq!(sci.tolerance, 1e-3);
    assert_eq!(sci.boundary_tolerance, None);
    assert_eq!(sci.max_nodes, 1_000);
    assert_eq!(sci.max_newton_iterations, 8);
    assert_eq!(sci.max_jacobian_refreshes, 4);
    assert_eq!(sci.max_mesh_refinements, 10);
    assert_eq!(sci.max_backtracking_steps, 4);
    assert_eq!(sci.telemetry, BvpSciTelemetryMode::Off);
    assert_eq!(sci.logging_level, SolverLogLevel::Off);
    assert_eq!(
        sci.execution_policy,
        crate::numerical::BVP_sci::new::BvpSciExecutionPolicy::Sequential
    );
    assert_eq!(sci.output_policy, BvpSciOutputPolicy::MeshAndValues);
    assert!(sci.singular_term.is_none());
}

#[test]
fn common_logging_level_is_forwarded_without_overwriting_typed_trace_settings() {
    use super::BvpSolverConfig;
    use crate::numerical::logging::SolverLogLevel;

    let level = SolverLogLevel::Debug;
    let correction_newton_settings =
        BvpSolverSettings::CorrectionNewton(CorrectionNewtonOptions::default())
            .with_logging_level(level);
    let BvpSolverSettings::CorrectionNewton(options) = correction_newton_settings else {
        unreachable!()
    };
    assert_eq!(options.logging_level, level);

    let correction_newton_config = BvpSolverConfig::CorrectionNewton {
        options: CorrectionNewtonOptions::default(),
        initial_guess: DMatrix::zeros(1, 2),
    }
    .with_logging_level(level);
    let BvpSolverConfig::CorrectionNewton { options, .. } = correction_newton_config else {
        unreachable!()
    };
    assert_eq!(options.logging_level, level);

    let damped = BvpSolverConfig::Damped {
        options: DampedSolverOptions::default(),
        initial_guess: DMatrix::zeros(1, 2),
    }
    .with_logging_level(level);
    let BvpSolverConfig::Damped { options, .. } = damped else {
        unreachable!()
    };
    assert_eq!(options.loglevel.as_deref(), Some("debug"));

    let frozen = BvpSolverConfig::Frozen {
        options: FrozenSolverOptions::default().with_bvp_logging_mode(BvpLoggingMode::Detailed),
        initial_guess: DMatrix::zeros(1, 2),
    }
    .with_logging_level(level);
    let BvpSolverConfig::Frozen { options, .. } = frozen else {
        unreachable!()
    };
    assert_eq!(options.logging_level, Some(level));
    assert_eq!(
        options.generated_backend_config.bvp_logging_mode,
        BvpLoggingMode::Detailed
    );

    let sci = BvpSolverConfig::Sci {
        options: BvpSciOptions::default(),
        sparse_factorization: BvpSciSparseFactorization::default(),
        aot_config: None,
        initial_guess: DMatrix::zeros(1, 2),
    }
    .with_logging_level(level);
    let BvpSolverConfig::Sci { options, .. } = sci else {
        unreachable!()
    };
    assert_eq!(options.logging_level, level);
}

#[test]
fn correction_newton_logging_defaults_off_and_can_be_disabled_before_guess_attachment() {
    let BvpSolverSettings::CorrectionNewton(defaults) =
        BvpSolverSettings::CorrectionNewton(CorrectionNewtonOptions::default())
    else {
        unreachable!()
    };
    assert_eq!(defaults.logging_level, SolverLogLevel::Off);

    let config = BvpSolverSettings::CorrectionNewton(CorrectionNewtonOptions::default())
        .with_logging_level(SolverLogLevel::Off)
        .with_initial_guess(DMatrix::zeros(1, 2));
    let BvpSolverConfig::CorrectionNewton { options, .. } = config else {
        unreachable!()
    };
    assert_eq!(options.logging_level, SolverLogLevel::Off);
}

#[test]
fn solver_settings_attach_caller_guess_without_erasing_native_route_options() {
    let damped = BvpSolverSettings::Damped(DampedSolverOptions::default())
        .with_initial_guess(DMatrix::zeros(1, 2));
    assert!(matches!(damped, BvpSolverConfig::Damped { .. }));

    let frozen = BvpSolverSettings::Frozen(FrozenSolverOptions::default())
        .with_initial_guess(DMatrix::zeros(1, 3));
    assert!(matches!(frozen, BvpSolverConfig::Frozen { .. }));

    let sci = BvpSolverSettings::Sci {
        options: BvpSciOptions::default().with_matrix_layout(BvpSciMatrixLayout::Sparse),
        sparse_factorization: BvpSciSparseFactorization::SuperLu,
        aot_config: None,
    }
    .with_initial_guess(DMatrix::zeros(1, 3));
    let BvpSolverConfig::Sci {
        options,
        sparse_factorization,
        ..
    } = sci
    else {
        unreachable!()
    };
    assert_eq!(options.matrix_layout, BvpSciMatrixLayout::Sparse);
    assert_eq!(sparse_factorization, BvpSciSparseFactorization::SuperLu);
}

#[test]
fn common_telemetry_policy_is_forwarded_or_rejected_without_promotion() {
    let damped = BvpSolverSettings::Damped(DampedSolverOptions::default())
        .with_telemetry_mode(BvpApiTelemetryMode::Detailed)
        .unwrap();
    let BvpSolverSettings::Damped(damped) = damped else {
        unreachable!()
    };
    assert_eq!(
        damped.generated_backend_config.bvp_telemetry_mode,
        BvpTelemetryMode::Detailed
    );
    assert_eq!(
        damped.generated_backend_config.aot_telemetry_mode,
        BvpAotTelemetryMode::Detailed
    );

    let unsupported = match BvpSolverSettings::Damped(DampedSolverOptions::default())
        .with_telemetry_mode(BvpApiTelemetryMode::Timings)
    {
        Ok(_) => panic!("Damped AOT must reject Timings-only collection"),
        Err(error) => error,
    };
    assert!(matches!(
        unsupported,
        BvpSettingsError::UnsupportedTelemetry {
            solver: "BVP_Damp Damped",
            mode: BvpApiTelemetryMode::Timings,
            ..
        }
    ));

    let sci = BvpSolverSettings::Sci {
        options: BvpSciOptions::default(),
        sparse_factorization: BvpSciSparseFactorization::StandardFaer,
        aot_config: None,
    }
    .with_telemetry_mode(BvpApiTelemetryMode::Timings)
    .unwrap();
    let BvpSolverSettings::Sci { options, .. } = sci else {
        unreachable!()
    };
    assert_eq!(options.telemetry, BvpSciTelemetryMode::Timings);
}

#[test]
fn problem_rejects_duplicate_conditions_and_non_finite_data() {
    let duplicate = BvpProblem::new(
        vec![Expr::parse_expression("v"), Expr::parse_expression("-y")],
        vec!["y".into(), "v".into()],
        "x",
        vec![0.0, 1.0],
        vec![
            BvpBoundaryCondition {
                state: 0,
                side: BvpBoundarySide::Left,
                value: 1.0,
            },
            BvpBoundaryCondition {
                state: 0,
                side: BvpBoundarySide::Left,
                value: 0.0,
            },
        ],
    );
    assert!(matches!(duplicate, Err(BvpApiError::InvalidProblem(_))));

    let non_finite = BvpProblem::new(
        vec![Expr::parse_expression("-y")],
        vec!["y".into()],
        "x",
        vec![0.0, f64::NAN],
        vec![BvpBoundaryCondition {
            state: 0,
            side: BvpBoundarySide::Left,
            value: 1.0,
        }],
    );
    assert!(matches!(non_finite, Err(BvpApiError::InvalidProblem(_))));
}

#[test]
fn problem_rejects_duplicate_symbol_names() {
    let error = BvpProblem::new(
        vec![Expr::parse_expression("-y"), Expr::parse_expression("y")],
        vec!["y".into(), "x".into()],
        "x",
        vec![0.0, 1.0],
        vec![
            BvpBoundaryCondition {
                state: 0,
                side: BvpBoundarySide::Left,
                value: 1.0,
            },
            BvpBoundaryCondition {
                state: 1,
                side: BvpBoundarySide::Right,
                value: 0.0,
            },
        ],
    )
    .expect_err("state and independent-variable names must be distinct");
    assert!(matches!(error, BvpApiError::InvalidProblem(_)));
}

#[test]
fn logging_setup_failures_remain_typed_through_facade_error_sources() {
    use crate::numerical::BVP_sci::new::BvpSciNewError;
    use crate::numerical::logging::SolverLoggingError;

    let damped_native = BvpBackendIntegrationError::LogSetupFailed {
        source: SolverLoggingError::FileCreation("permission denied".into()),
    };
    let damped_facade = BvpApiError::Damped(damped_native);
    let damped_source = std::error::Error::source(&damped_facade)
        .expect("facade must preserve the native Damped logging error");
    assert!(
        damped_source
            .to_string()
            .contains("cannot initialize BVP solver logging")
    );
    assert!(std::error::Error::source(damped_source).is_some());

    let frozen_facade = BvpApiError::Frozen(BvpBackendIntegrationError::LogSetupFailed {
        source: SolverLoggingError::LoggerAlreadyInstalled,
    });
    let frozen_source = std::error::Error::source(&frozen_facade)
        .expect("facade must preserve the native Frozen logging error");
    assert!(std::error::Error::source(frozen_source).is_some());

    let sci_facade = BvpApiError::Sci(BvpSciNewError::Logging(
        SolverLoggingError::LoggerAlreadyInstalled,
    ));
    assert!(std::error::Error::source(&sci_facade).is_some());
}

#[test]
fn first_order_system_may_constrain_one_state_at_both_endpoints() {
    let problem = BvpProblem::new(
        vec![Expr::parse_expression("v"), Expr::parse_expression("-y")],
        vec!["y".into(), "v".into()],
        "x",
        vec![0.0, 0.5, 1.0],
        vec![
            BvpBoundaryCondition {
                state: 0,
                side: BvpBoundarySide::Left,
                value: 0.0,
            },
            BvpBoundaryCondition {
                state: 0,
                side: BvpBoundarySide::Right,
                value: 1.0,
            },
        ],
    )
    .expect("two independent endpoint conditions on one state are valid");

    let native = problem.legacy_boundary_conditions();
    assert_eq!(native["y"], vec![(0, 0.0), (1, 1.0)]);
}

#[test]
fn facade_builds_native_solver_variants_and_preserves_route_guess_layouts() {
    let mut damped = BvpSolver::new(
        valid_problem(),
        BvpSolverConfig::Damped {
            options: DampedSolverOptions::default()
                .with_bounds(HashMap::from([("y".to_string(), (-10.0, 10.0))]))
                .with_rel_tolerance(HashMap::from([("y".to_string(), 1e-6)])),
            initial_guess: DMatrix::from_element(1, 2, 1.0),
        },
    )
    .expect("Damped native configuration should build");
    let BvpNativeSolver::Damped(native_damped) = damped.native() else {
        panic!("Damped route must retain its native solver");
    };
    assert_eq!(native_damped.n_steps, 2, "Damped counts intervals");
    assert_eq!(native_damped.initial_guess.shape(), (1, 2));
    assert_eq!(native_damped.x_mesh.len(), 3);
    assert!(matches!(
        damped
            .solve()
            .expect("Damped route should solve the sample BVP"),
        super::BvpSolution::Damped { .. }
    ));
    let damped_view = damped.solution_view().expect("solved result has a view");
    assert_eq!(damped_view.mesh().len(), 3);
    assert_eq!(damped_view.state_count(), 1);
    assert!(damped_view.value(2, 0).unwrap().is_finite());
    assert_eq!(damped_view.value(3, 0), None);
    let super::BvpSolverTelemetrySnapshot::Damped(facade_snapshot) = damped.telemetry_snapshot()
    else {
        panic!("Damped facade must retain its telemetry variant");
    };
    let BvpNativeSolver::Damped(native_damped) = damped.native() else {
        panic!("native route changed unexpectedly");
    };
    assert_eq!(facade_snapshot, native_damped.get_statistics().telemetry);
    assert_eq!(
        damped
            .postprocess_dataset()
            .expect("Damped result should be convertible to a dataset")
            .values
            .shape(),
        (3, 1)
    );
    let damped_export = tempfile::tempdir().expect("Damped export directory");
    let damped_csv = damped_export.path().join("damped.csv");
    assert!(
        damped
            .execute_postprocessing(&PostprocessPlan::new().save_csv(&damped_csv))
            .expect("Damped mesh result should export")
            .all_done()
    );
    assert!(damped_csv.exists());
    let unsupported_restart = damped.restart(vec![0.0, 0.5, 1.0], DMatrix::from_element(1, 3, 1.0));
    assert!(matches!(
        unsupported_restart,
        Err(BvpApiError::UnsupportedRoute { .. })
    ));
    assert!(
        damped.solution().is_some(),
        "BVP_Damp has no explicit facade restart; rejection preserves its result"
    );

    let mut frozen = BvpSolver::new(
        valid_problem(),
        BvpSolverConfig::Frozen {
            options: FrozenSolverOptions::default()
                .with_bounds(HashMap::from([("y".to_string(), (0.0, 2.0))])),
            initial_guess: DMatrix::from_element(1, 2, 1.0),
        },
    )
    .expect("Frozen native configuration should build");
    let BvpNativeSolver::Frozen(native_frozen) = frozen.native() else {
        panic!("Frozen route must retain its native solver");
    };
    assert_eq!(
        native_frozen
            .Bounds
            .as_ref()
            .and_then(|bounds| bounds.get("y")),
        Some(&(0.0, 2.0)),
        "BVP_api must preserve the complete Frozen bounds configuration"
    );
    assert_eq!(native_frozen.n_steps, 2, "Frozen counts mesh intervals");
    assert_eq!(native_frozen.initial_guess.shape(), (1, 2));
    assert_eq!(native_frozen.x_mesh.len(), 3);
    assert!(matches!(
        frozen
            .solve()
            .expect("Frozen route should solve the sample BVP"),
        super::BvpSolution::Frozen { .. }
    ));
    let frozen_view = frozen.solution_view().expect("solved result has a view");
    assert_eq!(frozen_view.mesh().len(), 3);
    assert_eq!(frozen_view.value(0, 0), Some(1.0));
    assert_eq!(frozen_view.state_count(), 1);
    for node in 0..frozen_view.mesh().len() {
        let value = frozen_view.value(node, 0).unwrap();
        assert!(value.is_finite() && (0.0..=2.0).contains(&value));
    }
    let super::BvpSolverTelemetrySnapshot::Frozen(facade_snapshot) = frozen.telemetry_snapshot()
    else {
        panic!("Frozen facade must retain its telemetry variant");
    };
    let BvpNativeSolver::Frozen(native_frozen) = frozen.native() else {
        panic!("native route changed unexpectedly");
    };
    assert_eq!(facade_snapshot, native_frozen.get_statistics().telemetry);
    assert_eq!(
        frozen
            .postprocess_dataset()
            .expect("Frozen result should be convertible to a dataset")
            .values
            .shape(),
        (3, 1)
    );
    let frozen_export = tempfile::tempdir().expect("Frozen export directory");
    let frozen_csv = frozen_export.path().join("frozen.csv");
    assert!(
        frozen
            .execute_postprocessing(&PostprocessPlan::new().save_csv(&frozen_csv))
            .expect("Frozen mesh result should export")
            .all_done()
    );
    assert!(frozen_csv.exists());

    let sci_options = BvpSciOptions::default()
        .with_execution(BvpSciExecution::Lambdify)
        .with_assembly(BvpSciAssembly::ExprLegacy);
    let sci = BvpSolver::new(
        valid_problem(),
        BvpSolverConfig::Sci {
            options: sci_options,
            sparse_factorization: BvpSciSparseFactorization::StandardFaer,
            aot_config: None,
            initial_guess: DMatrix::from_element(1, 3, 1.0),
        },
    )
    .expect("BVP_sci Lambdify route should build");
    assert!(matches!(sci.native(), BvpNativeSolver::Sci(_)));
}

#[test]
fn damped_and_frozen_facade_reports_native_failure_without_publishing_a_result() {
    let modes = [
        BvpTelemetryMode::Off,
        BvpTelemetryMode::Counters,
        BvpTelemetryMode::Timings,
        BvpTelemetryMode::Detailed,
    ];

    for mode in modes {
        for frozen in [false, true] {
            let config = if frozen {
                let mut options = FrozenSolverOptions::default();
                options.generated_backend_config.bvp_telemetry_mode = mode;
                BvpSolverConfig::Frozen {
                    options,
                    initial_guess: DMatrix::zeros(1, 2),
                }
            } else {
                let mut options = DampedSolverOptions::default()
                    .with_bounds(HashMap::from([("y".to_string(), (-10.0, 10.0))]))
                    .with_rel_tolerance(HashMap::from([("y".to_string(), 1e-6)]));
                options.generated_backend_config.bvp_telemetry_mode = mode;
                BvpSolverConfig::Damped {
                    options,
                    initial_guess: DMatrix::zeros(1, 2),
                }
            };
            let mut solver = BvpSolver::new(valid_problem(), config)
                .expect("failure fixture should prepare before callback execution");
            match &mut solver.native {
                BvpNativeSolver::CorrectionNewton(_) => {
                    unreachable!("test only constructs Damped/Frozen")
                }
                BvpNativeSolver::Damped(native) => native.method = "invalid".into(),
                BvpNativeSolver::Frozen(native) => native.method = "invalid".into(),
                BvpNativeSolver::Sci(_) => unreachable!("test only constructs Damped/Frozen"),
            }
            let error = solver
                .solve()
                .expect_err("invalid native route must fail typed");
            assert!(
                matches!(
                    (&error, frozen),
                    (BvpApiError::Damped(_), false) | (BvpApiError::Frozen(_), true)
                ),
                "frozen={frozen}, unexpected facade error: {error:?}"
            );
            assert!(std::error::Error::source(&error).is_some());
            assert!(
                solver.solution().is_none(),
                "failed attempt must not publish a result"
            );

            let snapshot = solver.telemetry_snapshot();
            let native_snapshot = match (snapshot, frozen) {
                (super::BvpSolverTelemetrySnapshot::Damped(snapshot), false)
                | (super::BvpSolverTelemetrySnapshot::Frozen(snapshot), true) => snapshot,
                _ => panic!("facade must preserve the selected native telemetry variant"),
            };
            assert_eq!(native_snapshot.telemetry_mode, mode);
            assert_eq!(native_snapshot.last_solve_status, BvpSolveStatus::Failed);
            assert_eq!(native_snapshot.solve_id, 1);

            match mode {
                BvpTelemetryMode::Off => {
                    assert_eq!(native_snapshot.counters, Default::default());
                    assert_eq!(native_snapshot.timings, Default::default());
                }
                BvpTelemetryMode::Counters => {
                    assert_eq!(native_snapshot.counters, Default::default());
                    assert_eq!(native_snapshot.timings, Default::default());
                }
                BvpTelemetryMode::Timings => {
                    assert_eq!(native_snapshot.counters, Default::default());
                    assert!(native_snapshot.timings.total > std::time::Duration::ZERO);
                }
                BvpTelemetryMode::Detailed => {
                    assert_eq!(native_snapshot.counters, Default::default());
                    assert!(native_snapshot.timings.total > std::time::Duration::ZERO);
                }
            }
        }
    }
}

#[test]
fn damped_and_frozen_failed_retry_preserves_last_accepted_facade_solution() {
    for frozen in [false, true] {
        let config = if frozen {
            BvpSolverConfig::Frozen {
                options: FrozenSolverOptions::default(),
                initial_guess: DMatrix::from_element(1, 2, 1.0),
            }
        } else {
            BvpSolverConfig::Damped {
                options: DampedSolverOptions::default()
                    .with_bounds(HashMap::from([("y".to_string(), (-10.0, 10.0))]))
                    .with_rel_tolerance(HashMap::from([("y".to_string(), 1e-6)])),
                initial_guess: DMatrix::from_element(1, 2, 1.0),
            }
        };
        let mut solver =
            BvpSolver::new(valid_problem(), config).expect("sample solver should construct");
        solver.solve().expect("initial solve should succeed");
        let (accepted_mesh, accepted_values) = match solver.solution().unwrap() {
            BvpSolution::Damped { mesh, values } | BvpSolution::Frozen { mesh, values } => {
                (mesh.as_slice().to_vec(), values.clone())
            }
            BvpSolution::CorrectionNewton { .. } => {
                unreachable!("test only constructs Damped/Frozen")
            }
            BvpSolution::Sci(_) => unreachable!("test only constructs Damped/Frozen"),
        };

        match &mut solver.native {
            BvpNativeSolver::CorrectionNewton(_) => {
                unreachable!("test only constructs Damped/Frozen")
            }
            BvpNativeSolver::Damped(native) => native.method = "invalid".into(),
            BvpNativeSolver::Frozen(native) => native.method = "invalid".into(),
            BvpNativeSolver::Sci(_) => unreachable!("test only constructs Damped/Frozen"),
        }
        let retry = solver.solve();
        assert!(
            matches!(
                (&retry, frozen),
                (Err(BvpApiError::Damped(_)), false) | (Err(BvpApiError::Frozen(_)), true)
            ),
            "frozen={frozen}, unexpected retry result: {retry:?}"
        );

        let (retained_mesh, retained_values) = match solver.solution().unwrap() {
            BvpSolution::Damped { mesh, values } | BvpSolution::Frozen { mesh, values } => {
                (mesh.as_slice(), values)
            }
            BvpSolution::CorrectionNewton { .. } => {
                unreachable!("test only constructs Damped/Frozen")
            }
            BvpSolution::Sci(_) => unreachable!("test only constructs Damped/Frozen"),
        };
        assert_eq!(retained_mesh, accepted_mesh);
        assert_eq!(retained_values, &accepted_values);
        let snapshot = solver.telemetry_snapshot();
        match (snapshot, frozen) {
            (super::BvpSolverTelemetrySnapshot::Damped(snapshot), false)
            | (super::BvpSolverTelemetrySnapshot::Frozen(snapshot), true) => {
                assert_eq!(snapshot.last_solve_status, BvpSolveStatus::Failed);
            }
            _ => panic!("facade must preserve the native telemetry route"),
        }
    }
}

#[test]
fn route_specific_initial_guess_shapes_are_validated_before_native_construction() {
    let error = BvpSolver::new(
        valid_problem(),
        BvpSolverConfig::Damped {
            options: DampedSolverOptions::default(),
            initial_guess: DMatrix::from_element(1, 3, 1.0),
        },
    )
    .err()
    .expect("Damped guess must contain one column per interval");
    assert!(matches!(error, BvpApiError::InvalidProblem(_)));
}

#[test]
fn damped_facade_packs_state_major_guess_into_mixed_boundary_node_order() {
    let problem = BvpProblem::new(
        vec![Expr::parse_expression("1"), Expr::parse_expression("2")],
        vec!["u".into(), "v".into()],
        "x",
        vec![0.0, 0.25, 0.5, 0.75, 1.0],
        vec![
            BvpBoundaryCondition {
                state: 0,
                side: BvpBoundarySide::Left,
                value: 0.0,
            },
            BvpBoundaryCondition {
                state: 1,
                side: BvpBoundarySide::Right,
                value: 5.0,
            },
        ],
    )
    .expect("mixed endpoint problem should validate");
    let state_major = DMatrix::from_row_slice(2, 4, &[0.25, 0.5, 0.75, 1.0, 3.0, 3.5, 4.0, 4.5]);

    let mut solver = BvpSolver::new(
        problem,
        BvpSolverConfig::Damped {
            options: DampedSolverOptions::default()
                .trapezoid_derivative()
                .with_bounds(HashMap::from([
                    ("u".to_string(), (-1.0, 2.0)),
                    ("v".to_string(), (2.0, 6.0)),
                ]))
                .with_rel_tolerance(HashMap::from([
                    ("u".to_string(), 1.0e-8),
                    ("v".to_string(), 1.0e-8),
                ]))
                .with_bvp_logging_config(BvpLoggingConfig::new(BvpLoggingMode::Detailed)),
            initial_guess: state_major.clone(),
        },
    )
    .expect("Damped facade should pack the public state-major profile");
    let BvpNativeSolver::Damped(native) = solver.native() else {
        panic!("expected Damped native route");
    };

    assert_eq!(
        native.initial_guess, state_major,
        "native bounds-validation profile remains state-major"
    );

    solver
        .solve()
        .expect("the supplied linear profile is an exact trapezoid solution");
    let trace = solver
        .damped_decision_trace_snapshot()
        .expect("detailed decision trace should be available through the facade");
    let initial_residual = trace.events.iter().find_map(|event| match event {
        crate::numerical::BVP_Damp::telemetry::BvpDampedDecisionEvent::IterationDecision {
            base_residual_norm,
            ..
        } => *base_residual_norm,
        _ => None,
    });
    assert!(
        initial_residual.expect("trace should include the initial residual") < 1e-12,
        "state-major exact profile should enter Newton with near-zero residual"
    );
}

#[test]
fn damped_facade_rejects_ambiguous_state_major_boundary_topology() {
    let problem = BvpProblem::new(
        vec![Expr::parse_expression("0"), Expr::parse_expression("0")],
        vec!["u".into(), "v".into()],
        "x",
        vec![0.0, 0.5, 1.0],
        vec![
            BvpBoundaryCondition {
                state: 0,
                side: BvpBoundarySide::Left,
                value: 0.0,
            },
            BvpBoundaryCondition {
                state: 0,
                side: BvpBoundarySide::Right,
                value: 1.0,
            },
        ],
    )
    .expect("problem-level boundary count is valid");

    let error = match BvpSolver::new(
        problem,
        BvpSolverConfig::Damped {
            options: DampedSolverOptions::default(),
            initial_guess: DMatrix::zeros(2, 2),
        },
    ) {
        Ok(_) => panic!("state-major rows cannot represent unequal free-node counts"),
        Err(error) => error,
    };
    assert!(matches!(error, BvpApiError::InvalidProblem(_)));
}

#[test]
fn sci_facade_explicitly_rejects_numerical_callback_configuration() {
    let error = BvpSolver::new(
        valid_problem(),
        BvpSolverConfig::Sci {
            options: BvpSciOptions::default(),
            sparse_factorization: BvpSciSparseFactorization::StandardFaer,
            aot_config: None,
            initial_guess: DMatrix::from_element(1, 3, 1.0),
        },
    )
    .err()
    .expect("symbolic facade must reject default numerical options as callbacks");
    assert!(matches!(
        error,
        BvpApiError::UnsupportedRoute {
            solver: "BVP_sci",
            reason: "BVP_api accepts symbolic equations; numerical callback plans must use the native BVP_sci builder",
        }
    ));

    let error = BvpSolver::new(
        valid_problem(),
        BvpSolverConfig::Sci {
            options: BvpSciOptions::default()
                .with_execution(BvpSciExecution::Lambdify)
                .with_assembly(BvpSciAssembly::Numerical),
            sparse_factorization: BvpSciSparseFactorization::StandardFaer,
            aot_config: None,
            initial_guess: DMatrix::from_element(1, 3, 1.0),
        },
    )
    .err()
    .expect("numerical assembly requires a native callback plan");
    assert!(matches!(error, BvpApiError::UnsupportedRoute { .. }));
}

#[test]
fn sci_symbolic_capability_matrix_preserves_frontend_layout_and_execution_policy() {
    let problem = BvpProblem::new(
        vec![Expr::parse_expression("-y")],
        vec!["y".into()],
        "x",
        vec![0.0, 0.25, 0.5, 0.75, 1.0],
        vec![BvpBoundaryCondition {
            state: 0,
            side: BvpBoundarySide::Left,
            value: 1.0,
        }],
    )
    .expect("matrix problem should validate");

    let frontends = [BvpSciAssembly::ExprLegacy, BvpSciAssembly::AtomViewNative];
    let layouts = [
        BvpSciMatrixLayout::Dense,
        BvpSciMatrixLayout::Sparse,
        BvpSciMatrixLayout::BandedAuto,
    ];
    let policies = [
        crate::symbolic::bvp::telemetry::BvpLambdifyExecutionPolicy::Sequential,
        crate::symbolic::bvp::telemetry::BvpLambdifyExecutionPolicy::Parallel { min_work: 0 },
    ];

    for frontend in frontends {
        for layout in layouts {
            for policy in policies {
                let options = BvpSciOptions::default()
                    .with_execution(BvpSciExecution::Lambdify)
                    .with_assembly(frontend)
                    .with_matrix_layout(layout)
                    .with_execution_policy(policy)
                    .with_tolerance(1e-5)
                    .with_limits(64, 8, 3)
                    .with_telemetry(BvpSciTelemetryMode::Counters);
                let mut solver = BvpSolver::new(
                    problem.clone(),
                    BvpSolverConfig::Sci {
                        options,
                        sparse_factorization: BvpSciSparseFactorization::StandardFaer,
                        aot_config: None,
                        initial_guess: DMatrix::from_row_slice(
                            1,
                            5,
                            &[1.0, 0.7788, 0.6065, 0.4724, 0.3679],
                        ),
                    },
                )
                .unwrap_or_else(|error| {
                    panic!("supported route {frontend:?}/{layout:?}/{policy:?} rejected: {error}")
                });

                let BvpNativeSolver::Sci(native) = solver.native() else {
                    panic!("facade must retain BVP_sci native route");
                };
                let effective_layout = match layout {
                    BvpSciMatrixLayout::BandedAuto => {
                        BvpSciMatrixLayout::Banded { lower: 0, upper: 0 }
                    }
                    explicit => explicit,
                };
                assert_eq!(native.options().execution, BvpSciExecution::Lambdify);
                assert_eq!(native.options().assembly, Some(frontend));
                assert_eq!(native.options().matrix_layout, effective_layout);
                assert_eq!(native.options().execution_policy, policy);
                assert_eq!(native.options().telemetry, BvpSciTelemetryMode::Counters);
                assert_eq!(native.options().tolerance, 1e-5);
                assert_eq!(native.options().max_nodes, 64);
                assert_eq!(native.options().max_newton_iterations, 8);
                assert_eq!(native.options().max_mesh_refinements, 3);
                assert_eq!(
                    native.sparse_factorization(),
                    NativeBvpSciSparseFactorization::StandardFaer
                );
                assert_eq!(native.plan().assembly(), frontend);

                let solution = solver.solve().unwrap_or_else(|error| {
                    panic!("route {frontend:?}/{layout:?}/{policy:?} failed: {error}")
                });
                let BvpSolution::Sci(solution) = solution else {
                    panic!("facade must preserve BVP_sci solution variant");
                };
                assert!(solution.residual_norm < 1e-5);
                assert!((solution.y[0] - 1.0).abs() < 1e-7);
                assert!((solution.y[solution.y.len() - 1] - (-1.0_f64).exp()).abs() < 2e-4);
            }
        }
    }
}

#[test]
fn sci_aot_route_without_generated_configuration_fails_without_lambdify_fallback() {
    let result = BvpSolver::new(
        valid_problem(),
        BvpSolverConfig::Sci {
            options: BvpSciOptions::default()
                .with_execution(BvpSciExecution::Aot)
                .with_assembly(BvpSciAssembly::AtomViewNative),
            sparse_factorization: BvpSciSparseFactorization::StandardFaer,
            aot_config: None,
            initial_guess: DMatrix::from_element(1, 3, 1.0),
        },
    );
    let error = match result {
        Ok(_) => panic!("AOT must require explicit generated-backend configuration"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        BvpApiError::Sci(crate::numerical::BVP_sci::new::BvpSciNewError::InvalidConfiguration(_))
    ));
}

#[test]
fn damped_facade_rebinds_fixed_inputs_and_reuses_the_previous_iterate() {
    let problem = BvpProblem::new(
        vec![Expr::parse_expression("rate")],
        vec!["y".into()],
        "x",
        vec![0.0, 0.25, 0.5, 0.75, 1.0],
        vec![BvpBoundaryCondition {
            state: 0,
            side: BvpBoundarySide::Left,
            value: 0.0,
        }],
    )
    .unwrap();
    let mut solver = BvpSolver::new(
        problem,
        BvpSolverConfig::Damped {
            options: DampedSolverOptions::default()
                .with_bounds(HashMap::from([("y".to_string(), (-10.0, 10.0))]))
                .with_rel_tolerance(HashMap::from([("y".to_string(), 1e-6)])),
            initial_guess: DMatrix::zeros(1, 4),
        },
    )
    .unwrap();
    solver
        .bind_parameters(vec!["rate".into()], vec![1.0])
        .expect("fixed input schema should bind before preparation");
    let first = solver.solve().unwrap();
    let super::BvpSolution::Damped { values, .. } = first else {
        panic!("expected Damped result");
    };
    assert!((values[(4, 0)] - 1.0).abs() < 1e-7);

    let invalid_rebind = solver
        .set_parameter_values(vec![])
        .expect_err("native binding must reject the wrong value count");
    assert!(matches!(
        &invalid_rebind,
        BvpApiError::Damped(BvpBackendIntegrationError::InvalidSolverConfiguration {
            field,
            value,
            message,
        }) if field == "param_values"
            && value == "0 values"
            && message.contains("expected exactly 1 values")
    ));
    assert!(
        std::error::Error::source(&invalid_rebind).is_some(),
        "Damped facade must preserve the typed native binding error"
    );
    assert!(
        solver.solution().is_some(),
        "rejected Damped rebind must preserve the previous successful result"
    );

    let second = solver
        .solve_with_parameter_values(vec![2.0])
        .expect("same prepared problem should solve with rebound input");
    let super::BvpSolution::Damped { values, .. } = second else {
        panic!("expected Damped result");
    };
    assert!((values[(4, 0)] - 2.0).abs() < 1e-7);
}

#[test]
fn frozen_facade_rebinds_fixed_inputs_and_warm_starts_on_the_same_mesh() {
    let problem = BvpProblem::new(
        vec![Expr::parse_expression("rate")],
        vec!["y".into()],
        "x",
        vec![0.0, 0.25, 0.5, 0.75, 1.0],
        vec![BvpBoundaryCondition {
            state: 0,
            side: BvpBoundarySide::Left,
            value: 0.0,
        }],
    )
    .unwrap();
    let mut solver = BvpSolver::new(
        problem,
        BvpSolverConfig::Frozen {
            options: FrozenSolverOptions::default(),
            initial_guess: DMatrix::zeros(1, 4),
        },
    )
    .unwrap();
    solver
        .bind_parameters(vec!["rate".into()], vec![1.0])
        .expect("fixed input schema should bind before preparation");
    let first = solver.solve().unwrap();
    let super::BvpSolution::Frozen { values, .. } = first else {
        panic!("expected Frozen result");
    };
    assert!((values[(4, 0)] - 1.0).abs() < 1e-7);

    let invalid_rebind = solver
        .set_parameter_values(vec![])
        .expect_err("native binding must reject the wrong value count");
    assert!(matches!(
        &invalid_rebind,
        BvpApiError::Frozen(BvpBackendIntegrationError::InvalidSolverConfiguration {
            field,
            value,
            message,
        }) if field == "param_values"
            && value == "0 values"
            && message.contains("expected exactly 1 values")
    ));
    assert!(
        std::error::Error::source(&invalid_rebind).is_some(),
        "Frozen facade must preserve the typed native binding error"
    );
    assert!(
        solver.solution().is_some(),
        "rejected Frozen rebind must preserve the previous successful result"
    );

    let second = solver
        .solve_with_parameter_values(vec![2.0])
        .expect("same prepared problem should solve with rebound input");
    let super::BvpSolution::Frozen { values, .. } = second else {
        panic!("expected Frozen result");
    };
    assert!((values[(4, 0)] - 2.0).abs() < 1e-7);
}

#[test]
fn damped_and_frozen_facade_parameter_loops_keep_native_point_contracts() {
    use super::BvpParameterLoopReport;
    use crate::numerical::BVP_Damp::NR_Damp_solver_damped::DampedParameterLoopOutcome;
    use crate::numerical::BVP_Damp::NR_Damp_solver_frozen::FrozenParameterLoopOutcome;

    let make_problem = || {
        BvpProblem::new(
            vec![Expr::parse_expression("1 / rate")],
            vec!["y".into()],
            "x",
            vec![0.0, 0.25, 0.5, 0.75, 1.0],
            vec![BvpBoundaryCondition {
                state: 0,
                side: BvpBoundarySide::Left,
                value: 0.0,
            }],
        )
        .expect("parameterized facade problem should validate")
    };
    let parameter_vectors = [vec![1.0], vec![0.0], vec![4.0]];

    let mut damped = BvpSolver::new(
        make_problem(),
        BvpSolverConfig::Damped {
            options: DampedSolverOptions::default()
                .with_bounds(HashMap::from([("y".to_string(), (-10.0, 10.0))]))
                .with_rel_tolerance(HashMap::from([("y".to_string(), 1e-6)]))
                .with_bvp_telemetry_mode(BvpTelemetryMode::Counters),
            initial_guess: DMatrix::zeros(1, 4),
        },
    )
    .expect("Damped facade route should construct");
    damped
        .bind_parameters(vec!["rate".into()], vec![1.0])
        .expect("fixed input schema should bind before the loop");
    let BvpParameterLoopReport::Damped(damped_report) = damped
        .solve_parameter_loop(&parameter_vectors)
        .expect("Damped facade should return a point report")
    else {
        panic!("facade must preserve the Damped loop report variant");
    };
    assert_eq!(damped_report.points.len(), 3);
    assert!(damped_report.stop_error.is_none());
    assert!(matches!(
        &damped_report.points[0].outcome,
        DampedParameterLoopOutcome::Converged { values, telemetry, .. }
            if (values[(4, 0)] - 1.0).abs() < 1e-7
                && telemetry.telemetry_mode == BvpTelemetryMode::Counters
    ));
    assert!(matches!(
        &damped_report.points[1].outcome,
        DampedParameterLoopOutcome::Failed { error, telemetry, .. }
            if matches!(
                error,
                BvpBackendIntegrationError::CallbackExecutionFailed { .. }
                    | BvpBackendIntegrationError::NonFiniteCallbackValue { .. }
                    | BvpBackendIntegrationError::LinearSolveFailed { .. }
            ) && telemetry.telemetry_mode == BvpTelemetryMode::Counters
    ));
    assert!(matches!(
        &damped_report.points[2].outcome,
        DampedParameterLoopOutcome::Converged { values, telemetry, .. }
            if (values[(4, 0)] - 0.25).abs() < 1e-7
                && telemetry.telemetry_mode == BvpTelemetryMode::Counters
    ));

    let mut frozen = BvpSolver::new(
        make_problem(),
        BvpSolverConfig::Frozen {
            options: FrozenSolverOptions::default()
                .with_bvp_telemetry_mode(BvpTelemetryMode::Counters),
            initial_guess: DMatrix::zeros(1, 4),
        },
    )
    .expect("Frozen facade route should construct");
    frozen
        .bind_parameters(vec!["rate".into()], vec![1.0])
        .expect("fixed input schema should bind before the loop");
    let BvpParameterLoopReport::Frozen(frozen_report) = frozen
        .solve_parameter_loop(&parameter_vectors)
        .expect("Frozen facade should return a point report")
    else {
        panic!("facade must preserve the Frozen loop report variant");
    };
    assert_eq!(frozen_report.points.len(), 3);
    assert!(frozen_report.stop_error.is_none());
    assert!(matches!(
        &frozen_report.points[0].outcome,
        FrozenParameterLoopOutcome::Converged { values, telemetry, .. }
            if (values[(4, 0)] - 1.0).abs() < 1e-7
                && telemetry.telemetry_mode == BvpTelemetryMode::Counters
    ));
    assert!(matches!(
        &frozen_report.points[1].outcome,
        FrozenParameterLoopOutcome::Failed { error, telemetry, .. }
            if matches!(
                error,
                BvpBackendIntegrationError::CallbackExecutionFailed { .. }
                    | BvpBackendIntegrationError::NonFiniteCallbackValue { .. }
                    | BvpBackendIntegrationError::LinearSolveFailed { .. }
            ) && telemetry.telemetry_mode == BvpTelemetryMode::Counters
    ));
    assert!(matches!(
        &frozen_report.points[2].outcome,
        FrozenParameterLoopOutcome::Converged { values, telemetry, .. }
            if (values[(4, 0)] - 0.25).abs() < 1e-7
                && telemetry.telemetry_mode == BvpTelemetryMode::Counters
    ));
}

#[test]
fn facade_does_not_treat_bvp_sci_unknowns_as_fixed_inputs() {
    let problem = valid_problem();
    let mut solver = BvpSolver::new(
        problem,
        BvpSolverConfig::Sci {
            options: BvpSciOptions::default()
                .with_execution(BvpSciExecution::Lambdify)
                .with_assembly(BvpSciAssembly::ExprLegacy),
            sparse_factorization: BvpSciSparseFactorization::StandardFaer,
            aot_config: None,
            initial_guess: DMatrix::from_element(1, 3, 1.0),
        },
    )
    .unwrap();
    let error = solver
        .bind_parameters(vec!["rate".into()], vec![1.0])
        .expect_err("Sci parameters are free unknowns, not fixed inputs");
    assert!(matches!(
        error,
        BvpApiError::UnsupportedRoute {
            solver: "BVP_sci",
            ..
        }
    ));
}

#[test]
fn bvp_sci_facade_parameter_loop_prepares_fixed_inputs_once_and_keeps_route_report() {
    use super::BvpParameterLoopReport;

    let problem = BvpProblem::new(
        vec![Expr::parse_expression("rate * y")],
        vec!["y".into()],
        "x",
        (0..21).map(|index| index as f64 / 20.0).collect(),
        vec![BvpBoundaryCondition {
            state: 0,
            side: BvpBoundarySide::Left,
            value: 1.0,
        }],
    )
    .unwrap()
    .with_external_parameters(vec!["rate".into()], vec![0.5])
    .expect("fixed inputs are declared before symbolic preparation");
    let mut solver = BvpSolver::new(
        problem,
        BvpSolverConfig::Sci {
            options: BvpSciOptions::default()
                .with_execution(BvpSciExecution::Lambdify)
                .with_assembly(BvpSciAssembly::ExprLegacy)
                .with_tolerance(1e-7)
                .with_limits(128, 10, 4)
                .with_telemetry(BvpSciTelemetryMode::Detailed),
            sparse_factorization: BvpSciSparseFactorization::StandardFaer,
            aot_config: None,
            initial_guess: DMatrix::from_element(1, 21, 1.0),
        },
    )
    .expect("BVP_sci facade route should prepare its fixed input schema");

    let report = solver
        .solve_parameter_loop(&[vec![0.5], vec![1.0]])
        .expect("both fixed-input points should be attempted");
    let BvpParameterLoopReport::Sci(report) = report else {
        panic!("facade must preserve the native BVP_sci report variant");
    };
    assert_eq!(report.points.len(), 2);
    assert_eq!(report.points[0].external_parameters, [0.5]);
    assert_eq!(report.points[1].external_parameters, [1.0]);
    let first = report.points[0]
        .result
        .as_ref()
        .expect("first parameter point should converge");
    let second = report.points[1]
        .result
        .as_ref()
        .expect("second parameter point should converge from the prior profile");
    assert!((first.y[first.y.len() - 1] - 0.5_f64.exp()).abs() < 2e-5);
    assert!((second.y[second.y.len() - 1] - 1.0_f64.exp()).abs() < 2e-5);
    assert!(
        report
            .points
            .iter()
            .all(|point| point.telemetry.rhs_calls > 0 && point.telemetry.jacobian_calls > 0)
    );
    assert!(solver.solution().is_none());
}

#[test]
fn bvp_sci_external_parameter_schema_rejects_colliding_names_and_bad_values() {
    let problem = valid_problem();
    assert!(matches!(
        problem
            .clone()
            .with_external_parameters(vec!["y".into()], vec![1.0]),
        Err(BvpApiError::InvalidProblem(_))
    ));
    assert!(matches!(
        problem.with_external_parameters(vec!["rate".into()], vec![f64::NAN]),
        Err(BvpApiError::InvalidProblem(_))
    ));
}

#[test]
fn sci_facade_solves_free_parameter_with_extra_boundary_residual_and_jacobian() {
    let problem = BvpProblem::new(
        vec![Expr::parse_expression("p")],
        vec!["y".into()],
        "x",
        vec![0.0, 0.25, 0.5, 0.75, 1.0],
        vec![BvpBoundaryCondition {
            state: 0,
            side: BvpBoundarySide::Left,
            value: 0.0,
        }],
    )
    .unwrap();
    let free_parameters = BvpSciFreeParameters::new(
        vec!["p".into()],
        vec![0.4],
        |_ya, _yb, parameters, extra_residuals| {
            extra_residuals[0] = parameters[0] - 1.0;
            Ok(())
        },
    )
    .with_jacobian(|_ya, _yb, _parameters, dya, dyb, dp| {
        dya[0] = 0.0;
        dyb[0] = 0.0;
        dp[0] = 1.0;
        Ok(())
    });
    let options = BvpSciOptions::default()
        .with_execution(BvpSciExecution::Lambdify)
        .with_assembly(BvpSciAssembly::ExprLegacy)
        .with_tolerance(1e-7)
        .with_limits(64, 8, 3)
        .with_telemetry(BvpSciTelemetryMode::Detailed);
    let mut solver = BvpSolver::new(
        problem,
        BvpSolverConfig::SciWithFreeParameters {
            options,
            sparse_factorization: BvpSciSparseFactorization::StandardFaer,
            aot_config: None,
            initial_guess: DMatrix::zeros(1, 5),
            free_parameters,
        },
    )
    .unwrap();

    assert_eq!(solver.parameter_guess().unwrap(), vec![0.4]);
    let first = solver.solve().unwrap();
    let BvpSolution::Sci(first) = first else {
        panic!("expected BVP_sci solution");
    };
    assert!((first.parameters[0] - 1.0).abs() < 1e-7);
    assert!((first.y[0] - 0.0).abs() < 1e-8);
    assert!((first.y[first.y.len() - 1] - 1.0).abs() < 1e-6);

    let second_parameter = {
        let second = solver
            .solve_with_parameter_guess(vec![0.8])
            .expect("new guess should re-solve with retained mesh and profile");
        let BvpSolution::Sci(second) = second else {
            panic!("expected BVP_sci solution");
        };
        assert!((second.parameters[0] - 1.0).abs() < 1e-7);
        second.parameters[0]
    };
    assert_eq!(solver.parameter_guess().unwrap(), vec![second_parameter]);
    let super::BvpSolverTelemetrySnapshot::Sci(telemetry) = solver.telemetry_snapshot() else {
        panic!("expected BVP_sci telemetry");
    };
    assert!(telemetry.boundary_jacobian_calls > 0);
}

#[test]
fn sci_callback_failure_preserves_native_stage_and_allows_retry() {
    use crate::numerical::BVP_sci::new::{BvpSciNewError, BvpSciStage};

    let problem = BvpProblem::new(
        vec![Expr::parse_expression("p")],
        vec!["y".into()],
        "x",
        vec![0.0, 0.25, 0.5, 0.75, 1.0],
        vec![BvpBoundaryCondition {
            state: 0,
            side: BvpBoundarySide::Left,
            value: 0.0,
        }],
    )
    .expect("test problem should validate");
    let fail_once = Arc::new(AtomicBool::new(true));
    let callback_failure = Arc::clone(&fail_once);
    let free_parameters = BvpSciFreeParameters::new(
        vec!["p".into()],
        vec![0.4],
        move |_ya, _yb, parameters, extra_residuals| {
            if callback_failure.swap(false, Ordering::SeqCst) {
                return Err("injected one-shot boundary callback failure".into());
            }
            extra_residuals[0] = parameters[0] - 1.0;
            Ok(())
        },
    );
    let mut options = BvpSciOptions::default()
        .with_execution(BvpSciExecution::Lambdify)
        .with_assembly(BvpSciAssembly::ExprLegacy)
        .with_tolerance(1e-7)
        .with_limits(64, 8, 3);
    options.telemetry = crate::numerical::BVP_sci::new::BvpSciTelemetryMode::Detailed;
    let mut solver = BvpSolver::new(
        problem,
        BvpSolverConfig::SciWithFreeParameters {
            options,
            sparse_factorization: BvpSciSparseFactorization::StandardFaer,
            aot_config: None,
            initial_guess: DMatrix::zeros(1, 5),
            free_parameters,
        },
    )
    .expect("BVP_sci facade route should prepare");

    let error = solver
        .solve()
        .expect_err("first boundary callback invocation is injected to fail");
    assert!(matches!(
        &error,
        BvpApiError::Sci(BvpSciNewError::Callback {
            stage: BvpSciStage::BoundaryCallback,
            message,
        }) if message.contains("injected one-shot")
    ));
    assert!(
        std::error::Error::source(&error).is_some(),
        "facade error must retain the native BVP_sci error as its source"
    );
    assert!(
        solver.solution().is_none(),
        "a failed solve must not expose a stale facade result"
    );
    let super::BvpSolverTelemetrySnapshot::Sci(failed_snapshot) = solver.telemetry_snapshot()
    else {
        panic!("failed Sci solve must keep the native telemetry route");
    };
    assert_eq!(
        failed_snapshot.mode,
        crate::numerical::BVP_sci::new::BvpSciTelemetryMode::Detailed
    );
    assert_eq!(failed_snapshot.full_solve_calls, 1);

    let result = solver
        .solve()
        .expect("same prepared solver should recover after a transient callback failure");
    let BvpSolution::Sci(result) = result else {
        panic!("facade must retain the native BVP_sci result");
    };
    assert!((result.parameters[0] - 1.0).abs() < 1e-7);
    assert!(solver.solution().is_some());
}

#[test]
fn legacy_damped_route_rejects_nonuniform_mesh_instead_of_silently_uniformizing_it() {
    let problem = BvpProblem::new(
        vec![Expr::parse_expression("-y")],
        vec!["y".into()],
        "x",
        vec![0.0, 0.2, 1.0],
        vec![BvpBoundaryCondition {
            state: 0,
            side: BvpBoundarySide::Left,
            value: 1.0,
        }],
    )
    .expect("nonuniform BVP problem is valid");
    let error = BvpSolver::new(
        problem,
        BvpSolverConfig::Damped {
            options: DampedSolverOptions::default(),
            initial_guess: DMatrix::from_element(1, 2, 1.0),
        },
    )
    .err()
    .expect("Damped native constructor cannot preserve arbitrary mesh nodes");
    assert!(matches!(error, BvpApiError::UnsupportedRoute { .. }));
}

#[test]
fn facade_sci_route_retains_native_solution_and_builds_postprocessing_data() {
    let problem = BvpProblem::new(
        vec![Expr::parse_expression("-y")],
        vec!["y".into()],
        "x",
        vec![0.0, 0.25, 0.5, 0.75, 1.0],
        vec![BvpBoundaryCondition {
            state: 0,
            side: BvpBoundarySide::Left,
            value: 1.0,
        }],
    )
    .expect("test problem should validate");
    let options = BvpSciOptions::default()
        .with_execution(BvpSciExecution::Lambdify)
        .with_assembly(BvpSciAssembly::ExprLegacy)
        .with_tolerance(1e-5)
        .with_limits(64, 8, 3);
    let mut solver = BvpSolver::new(
        problem,
        BvpSolverConfig::Sci {
            options,
            sparse_factorization: BvpSciSparseFactorization::StandardFaer,
            aot_config: None,
            initial_guess: DMatrix::from_row_slice(1, 5, &[1.0, 0.7788, 0.6065, 0.4724, 0.3679]),
        },
    )
    .expect("BVP_sci route should prepare");

    let node_count = {
        let solution = solver.solve().expect("BVP_sci should converge");
        let super::BvpSolution::Sci(result) = solution else {
            panic!("facade must keep the native Sci result variant");
        };
        assert!((result.y[0] - 1.0).abs() < 1e-8);
        assert!((result.y[result.y.len() - 1] - (-1.0_f64).exp()).abs() < 1e-4);
        assert!(result.iterations > 0);
        result.x.len()
    };
    let sci_view = solver.solution_view().expect("solved result has a view");
    assert_eq!(sci_view.mesh().len(), node_count);
    assert_eq!(sci_view.state_count(), 1);
    assert!((sci_view.value(node_count - 1, 0).unwrap() - (-1.0_f64).exp()).abs() < 1e-4);

    let super::BvpSolverTelemetrySnapshot::Sci(facade_snapshot) = solver.telemetry_snapshot()
    else {
        panic!("BVP_sci facade must retain its telemetry variant");
    };
    let BvpNativeSolver::Sci(native_sci) = solver.native() else {
        panic!("native route changed unexpectedly");
    };
    assert_eq!(facade_snapshot, native_sci.plan().telemetry_snapshot());

    let dataset = solver
        .postprocess_dataset()
        .expect("successful route result should be postprocessable");
    let BvpSolution::Sci(native_result) = solver.solution().expect("result retained") else {
        panic!("expected native BVP_sci result");
    };
    assert_eq!(dataset.axis_name, "x");
    assert_eq!(dataset.variable_names, vec!["y"]);
    assert_eq!(dataset.axis.as_slice(), native_result.x.as_slice());
    assert_eq!(dataset.values.nrows(), node_count);
    assert_eq!(dataset.values.ncols(), 1);
    for node in 0..node_count {
        assert_eq!(dataset.values[(node, 0)], native_result.y[node]);
    }
    let export_dir = tempfile::tempdir().expect("temporary export directory");
    let export_path = export_dir.path().join("solution.csv");
    let report = solver
        .execute_postprocessing(&PostprocessPlan::new().save_csv(&export_path))
        .expect("facade should execute the shared export plan");
    assert!(report.all_done());
    assert!(export_path.exists());
    assert!(
        solver.solution().is_some(),
        "export must retain the solve result"
    );

    let export_error = solver
        .execute_postprocessing(&PostprocessPlan::new().save_csv(export_dir.path()))
        .expect_err("writing a file over the export directory must fail");
    assert!(matches!(
        export_error,
        BvpApiError::Postprocessing {
            solver: "BVP_sci",
            source: PostprocessError::Io(_)
        }
    ));
    assert!(
        solver.solution().is_some(),
        "postprocessing failure must leave the native solve result inspectable"
    );

    let invalid_restart = solver.restart(
        vec![0.0, 0.8, 0.7],
        DMatrix::from_row_slice(1, 3, &[1.0, 0.6065, 0.3679]),
    );
    assert!(matches!(
        invalid_restart,
        Err(BvpApiError::InvalidProblem(_))
    ));
    assert!(
        solver.solution().is_some(),
        "rejected restart validation must preserve the previous successful result"
    );

    solver
        .restart(
            vec![0.0, 0.5, 1.0],
            DMatrix::from_row_slice(1, 3, &[1.0, 0.6065, 0.3679]),
        )
        .expect("BVP_sci should restart with a new mesh and state");
    assert!(
        solver.solution().is_none(),
        "restart invalidates the prior result"
    );
    let restarted = solver.solve().expect("restarted BVP_sci should converge");
    let super::BvpSolution::Sci(restarted) = restarted else {
        panic!("restart must retain the native Sci solution variant");
    };
    assert!(
        restarted.x.len() >= 3,
        "adaptive mesh may add nodes after restart"
    );
    assert_eq!(restarted.x.first(), Some(&0.0));
    assert_eq!(restarted.x.last(), Some(&1.0));
    assert!((restarted.y[restarted.y.len() - 1] - (-1.0_f64).exp()).abs() < 1e-4);
    let restarted_node_count = restarted.x.len();
    assert_eq!(
        solver
            .postprocess_dataset()
            .expect("restarted result should be postprocessable")
            .values
            .nrows(),
        restarted_node_count
    );

    let _ = solver.native_mut();
    assert!(
        solver.solution().is_none(),
        "mutable native access invalidates the facade result"
    );
}
