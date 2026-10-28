use super::{
    BvpApiError, BvpBoundaryCondition, BvpBoundarySide, BvpNativeSolver, BvpProblem,
    BvpSciFreeParameters, BvpSciSparseFactorization, BvpSolution, BvpSolver, BvpSolverConfig,
};
use crate::Utils::postprocessing::{PostprocessError, PostprocessPlan};
use crate::numerical::BVP_Damp::NR_Damp_solver_damped::DampedSolverOptions;
use crate::numerical::BVP_Damp::NR_Damp_solver_frozen::FrozenSolverOptions;
use crate::numerical::BVP_Damp::telemetry::{BvpLoggingMode, BvpTelemetryMode};
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
    assert_eq!(strategy.max_damp_iter, Some(5));
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
            options: FrozenSolverOptions::default(),
            initial_guess: DMatrix::from_element(1, 3, 1.0),
        },
    )
    .expect("Frozen native configuration should build");
    let BvpNativeSolver::Frozen(native_frozen) = frozen.native() else {
        panic!("Frozen route must retain its native solver");
    };
    assert_eq!(native_frozen.n_steps, 3, "Frozen counts mesh points");
    assert_eq!(native_frozen.initial_guess.shape(), (1, 3));
    assert_eq!(native_frozen.x_mesh.len(), 3);
    assert!(matches!(
        frozen
            .solve()
            .expect("Frozen route should solve the sample BVP"),
        super::BvpSolution::Frozen { .. }
    ));
    let frozen_view = frozen.solution_view().expect("solved result has a view");
    assert_eq!(frozen_view.mesh().len(), 3);
    assert_eq!(frozen_view.state_count(), 1);
    assert!(frozen_view.value(2, 0).unwrap().is_finite());
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
            initial_guess: DMatrix::zeros(1, 5),
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
        DampedParameterLoopOutcome::Failed { error, .. }
            if matches!(
                error,
                BvpBackendIntegrationError::CallbackExecutionFailed { .. }
                    | BvpBackendIntegrationError::NonFiniteCallbackValue { .. }
                    | BvpBackendIntegrationError::LinearSolveFailed { .. }
            )
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
            initial_guess: DMatrix::zeros(1, 5),
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
        FrozenParameterLoopOutcome::Failed { error, .. }
            if matches!(
                error,
                BvpBackendIntegrationError::CallbackExecutionFailed { .. }
                    | BvpBackendIntegrationError::NonFiniteCallbackValue { .. }
                    | BvpBackendIntegrationError::LinearSolveFailed { .. }
            )
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
    let mut solver = BvpSolver::new(
        problem,
        BvpSolverConfig::SciWithFreeParameters {
            options: BvpSciOptions::default()
                .with_execution(BvpSciExecution::Lambdify)
                .with_assembly(BvpSciAssembly::ExprLegacy)
                .with_tolerance(1e-7)
                .with_limits(64, 8, 3),
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
