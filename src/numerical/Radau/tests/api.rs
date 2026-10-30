//! Public-facade stories: frontend/layout selection, continuation, output,
//! telemetry, and typed errors.

use super::super::api::{
    RadauConfig, RadauErrorKind, RadauFrontend, RadauJacobianSource, RadauMatrixLayout,
    RadauOutputPolicy, RadauProblem, RadauSolver, RadauTelemetryMode,
};
use crate::numerical::ivp_workloads::{WorkloadKind, build_workload};
use crate::symbolic::symbolic_engine::Expr;

#[test]
fn banded_auto_infers_structural_widths_and_resolves_public_config() {
    let problem = RadauProblem::new(
        vec![
            Expr::parse_expression("-2*y0 + y1"),
            Expr::parse_expression("y0 - 2*y1 + y2"),
            Expr::parse_expression("y1 - 2*y2"),
        ],
        vec!["y0".into(), "y1".into(), "y2".into()],
        "t",
    );
    let mut solver = RadauSolver::prepare(
        problem,
        RadauConfig {
            matrix_layout: RadauMatrixLayout::BandedAuto,
            ..RadauConfig::default()
        },
    )
    .unwrap();
    assert_eq!(
        solver.config().matrix_layout,
        RadauMatrixLayout::Banded { lower: 1, upper: 1 }
    );
    let solution = solver.solve(&[1.0, 0.5, 0.25]).unwrap();
    assert!(solution.y.iter().all(|value| value.is_finite()));
}

#[test]
fn public_solver_exposes_frontend_continuation_output_and_telemetry() {
    for frontend in [RadauFrontend::ExprLegacy, RadauFrontend::AtomViewNative] {
        let problem = RadauProblem::new(
            vec![Expr::parse_expression("-a*y")],
            vec!["y".to_owned()],
            "t",
        )
        .with_parameter_names(vec!["a".to_owned()]);
        let problem = if frontend == RadauFrontend::ExprLegacy {
            problem.with_jacobian(vec![Expr::parse_expression("-a")])
        } else {
            problem
        };
        let config = RadauConfig {
            frontend,
            matrix_layout: RadauMatrixLayout::Dense,
            first_step: Some(0.1),
            max_step: 0.25,
            t_bound: 1.0,
            telemetry: RadauTelemetryMode::Counters,
            output: RadauOutputPolicy::Dense,
            ..RadauConfig::default()
        };
        let mut solver = RadauSolver::prepare(problem, config).unwrap();
        let solution = solver
            .try_solve_with_parameter_values(&[1.0], &[1.0])
            .unwrap();
        assert!((solution.y[0] - (-1.0f64).exp()).abs() < 2.0e-4);
        let midpoint = solution.sample(0.5).unwrap();
        assert!((midpoint[0] - (-0.5f64).exp()).abs() < 2.0e-4);
        assert_eq!(solution.telemetry().mode, RadauTelemetryMode::Counters);
        assert!(solution.telemetry().counters["accepted_steps"] > 0);
        assert_eq!(
            solution.telemetry().counters["frontend_preparations"],
            1,
            "public telemetry must retain the cold preparation event"
        );

        let continued = solver
            .try_solve_with_parameter_values(&[1.0], &[2.0])
            .unwrap();
        assert!((continued.y[0] - (-2.0f64).exp()).abs() < 3.0e-4);
    }
}

#[test]
fn expr_legacy_derives_compact_analytic_jacobian_when_omitted() {
    let workload = build_workload(WorkloadKind::DiffusionChain, 12);
    let variables: Vec<&str> = workload.variables.iter().map(String::as_str).collect();
    let explicit_jacobian = workload
        .equations
        .iter()
        .flat_map(|equation| {
            variables
                .iter()
                .map(move |variable| equation.diff(variable))
        })
        .collect::<Vec<_>>();
    let parameters = workload.parameter_values.as_slice().to_vec();
    let initial = workload.initial_state.as_slice().to_vec();

    for layout in [
        RadauMatrixLayout::Dense,
        RadauMatrixLayout::Sparse,
        RadauMatrixLayout::Banded { lower: 1, upper: 1 },
    ] {
        let make_problem = || {
            RadauProblem::new(
                workload.equations.clone(),
                workload.variables.clone(),
                workload.time_variable.clone(),
            )
            .with_parameter_names(workload.parameter_names.clone())
        };
        let config = RadauConfig {
            t_bound: 0.002,
            first_step: Some(0.0002),
            max_step: 0.0005,
            matrix_layout: layout,
            telemetry: RadauTelemetryMode::Detailed,
            ..RadauConfig::default()
        };
        let mut derived = RadauSolver::prepare(make_problem(), config.clone()).unwrap();
        let mut explicit = RadauSolver::prepare(
            make_problem().with_jacobian(explicit_jacobian.clone()),
            config,
        )
        .unwrap();

        let derived_solution = derived
            .try_solve_with_parameter_values(&initial, &parameters)
            .unwrap();
        let explicit_solution = explicit
            .try_solve_with_parameter_values(&initial, &parameters)
            .unwrap();
        let max_state_diff = derived_solution
            .y
            .iter()
            .zip(&explicit_solution.y)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f64::max);

        assert!(
            max_state_diff <= 1.0e-12,
            "{layout:?}: {max_state_diff:.3e}"
        );
        assert_eq!(
            derived_solution.telemetry().counters["finite_difference_probes"],
            0,
            "omitted Expr Jacobian must use analytic symbolic differentiation"
        );
        assert_eq!(
            derived_solution.telemetry().counters["jacobian_calls"],
            explicit_solution.telemetry().counters["jacobian_calls"]
        );
        assert!(
            derived_solution.telemetry().timings_ms["symbolic_jacobian_ms"] > 0.0,
            "{layout:?}: derived Jacobian stage must be attributed"
        );
    }
}

#[test]
fn expr_finite_difference_remains_explicit_when_selected() {
    let problem = RadauProblem::new(
        vec![Expr::parse_expression("-a*y")],
        vec!["y".to_owned()],
        "t",
    )
    .with_parameter_names(vec!["a".to_owned()]);
    let mut solver = RadauSolver::prepare(
        problem,
        RadauConfig {
            t_bound: 0.1,
            first_step: Some(0.025),
            max_step: 0.05,
            jacobian_source: RadauJacobianSource::FiniteDifference,
            telemetry: RadauTelemetryMode::Counters,
            ..RadauConfig::default()
        },
    )
    .unwrap();
    let solution = solver
        .try_solve_with_parameter_values(&[1.0], &[1.0])
        .unwrap();

    assert!((solution.y[0] - (-0.1_f64).exp()).abs() < 3.0e-5);
    assert!(solution.telemetry().counters["finite_difference_probes"] > 0);
}

#[test]
fn public_solver_telemetry_modes_obey_four_mode_contract() {
    for mode in [
        RadauTelemetryMode::Off,
        RadauTelemetryMode::Counters,
        RadauTelemetryMode::Timings,
        RadauTelemetryMode::Detailed,
    ] {
        let problem = RadauProblem::new(
            vec![Expr::parse_expression("-a*y")],
            vec!["y".to_owned()],
            "t",
        )
        .with_parameter_names(vec!["a".to_owned()])
        .with_jacobian(vec![Expr::parse_expression("-a")]);
        let config = RadauConfig {
            t_bound: 0.1,
            first_step: Some(0.025),
            max_step: 0.05,
            telemetry: mode,
            ..RadauConfig::default()
        };
        let mut solver = RadauSolver::prepare(problem, config).unwrap();
        let solution = solver
            .try_solve_with_parameter_values(&[1.0], &[1.0])
            .unwrap();
        let report = solution.telemetry();

        assert_eq!(report.mode, mode);
        assert!((solution.y[0] - (-0.1_f64).exp()).abs() < 3.0e-5);
        assert!(
            report.counters.values().all(|value| *value == 0)
                == matches!(mode, RadauTelemetryMode::Off | RadauTelemetryMode::Timings)
        );
        assert!(
            report.timings_ms.values().all(|value| *value == 0.0)
                == matches!(mode, RadauTelemetryMode::Off | RadauTelemetryMode::Counters)
        );
        assert!(report.timings_ms.values().all(|value| value.is_finite()));

        if matches!(
            mode,
            RadauTelemetryMode::Counters | RadauTelemetryMode::Detailed
        ) {
            assert_eq!(report.counters["frontend_preparations"], 1);
            assert!(report.counters["residual_calls"] > 0);
            assert!(report.counters["jacobian_calls"] > 0);
            if mode == RadauTelemetryMode::Counters {
                assert_eq!(report.timings_ms["preparation_ms"], 0.0);
            }
        }
        if matches!(
            mode,
            RadauTelemetryMode::Timings | RadauTelemetryMode::Detailed
        ) {
            assert!(report.timings_ms["preparation_ms"] > 0.0);
            assert!(report.timings_ms["newton_ms"] > 0.0);
            if mode == RadauTelemetryMode::Timings {
                assert_eq!(report.counters["residual_calls"], 0);
                assert_eq!(report.counters["jacobian_calls"], 0);
            }
        }
    }
}

#[test]
fn public_aot_route_is_typed_and_does_not_fallback() {
    let problem = RadauProblem::new(
        vec![Expr::parse_expression("-y")],
        vec!["y".to_owned()],
        "t",
    );
    let error = match RadauSolver::prepare(
        problem,
        RadauConfig {
            execution: super::super::api::RadauExecution::Aot,
            ..RadauConfig::default()
        },
    ) {
        Err(error) => error,
        Ok(_) => panic!("AOT preparation unexpectedly fell back to Lambdify"),
    };
    assert_eq!(error.kind(), RadauErrorKind::Unsupported);
}

#[test]
fn public_layout_selection_reaches_native_structured_backends() {
    for layout in [
        RadauMatrixLayout::Sparse,
        RadauMatrixLayout::Banded { lower: 1, upper: 1 },
    ] {
        let problem = RadauProblem::new(
            vec![Expr::parse_expression("-y")],
            vec!["y".to_owned()],
            "t",
        );
        let config = RadauConfig {
            frontend: RadauFrontend::AtomViewNative,
            matrix_layout: layout,
            first_step: Some(0.05),
            t_bound: 0.1,
            ..RadauConfig::default()
        };
        let mut solver = RadauSolver::prepare(problem, config).unwrap();
        let solution = solver.solve(&[1.0]).unwrap();
        assert!((solution.y[0] - (-0.1f64).exp()).abs() < 2.0e-4);
    }
}

#[test]
fn public_sparse_solve_matches_dense_when_jacobian_lacks_two_diagonals() {
    let equations = ["-2*y1 + y2", "y0 - 2*y2", "-2*y2 + y3", "y2 - 2*y3"]
        .into_iter()
        .map(Expr::parse_expression)
        .collect::<Vec<_>>();
    let variables = ["y0", "y1", "y2", "y3"]
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let initial = [1.0, 0.5, -0.25, 0.75];
    let mut reference: Option<Vec<f64>> = None;

    for frontend in [RadauFrontend::ExprLegacy, RadauFrontend::AtomViewNative] {
        for layout in [RadauMatrixLayout::Dense, RadauMatrixLayout::Sparse] {
            let mut problem = RadauProblem::new(equations.clone(), variables.clone(), "t");
            if frontend == RadauFrontend::ExprLegacy {
                let jacobian = equations
                    .iter()
                    .flat_map(|equation| {
                        variables
                            .iter()
                            .map(move |variable| equation.diff(variable))
                    })
                    .collect();
                problem = problem.with_jacobian(jacobian);
            }
            let config = RadauConfig {
                frontend,
                matrix_layout: layout,
                first_step: Some(0.002),
                max_step: 0.01,
                t_bound: 0.05,
                ..RadauConfig::default()
            };
            let mut solver = RadauSolver::prepare(problem, config).unwrap();
            let solution = solver.solve(&initial).unwrap();
            assert!(solution.y.iter().all(|value| value.is_finite()));

            if let Some(expected) = reference.as_ref() {
                for (actual, expected) in solution.y.iter().zip(expected.iter()) {
                    assert!((actual - expected).abs() < 2.0e-8);
                }
            } else {
                reference = Some(solution.y);
            }
        }
    }
}
