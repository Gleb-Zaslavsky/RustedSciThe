use super::*;
use crate::numerical::BVP_CorrectionNewton::{
    CorrectionNewtonMatrixLayout, CorrectionNewtonParameterLoopOutcome,
    CorrectionNewtonSymbolicFrontend, CorrectionNewtonTelemetryMode,
};
use crate::numerical::BVP_api::{BvpParameterLoopReport, BvpSolverTelemetrySnapshot};

fn solver(frontend: CorrectionNewtonSymbolicFrontend) -> BvpSolver {
    let steps = 16;
    let problem = BvpProblem::new(
        vec![Expr::parse_expression("1/rate")],
        vec!["y".into()],
        "x",
        (0..=steps).map(|i| i as f64 / steps as f64).collect(),
        vec![BvpBoundaryCondition {
            state: 0,
            side: BvpBoundarySide::Left,
            value: 1.0,
        }],
    )
    .unwrap()
    .with_external_parameters(vec!["rate".into()], vec![1.0])
    .unwrap();
    BvpSolver::new(
        problem,
        BvpSolverSettings::CorrectionNewton(CorrectionNewtonOptions {
            symbolic_frontend: frontend,
            matrix_layout: CorrectionNewtonMatrixLayout::BandedAuto,
            telemetry_mode: CorrectionNewtonTelemetryMode::Counters,
            ..CorrectionNewtonOptions::default()
        })
        .with_initial_guess(DMatrix::from_element(1, steps, 1.0)),
    )
    .unwrap()
}

#[test]
fn correction_newton_loop_publishes_last_success_preserves_failures_and_postprocessing() {
    for frontend in [
        CorrectionNewtonSymbolicFrontend::ExprLegacy,
        CorrectionNewtonSymbolicFrontend::AtomView,
    ] {
        let mut solver = solver(frontend);
        let BvpParameterLoopReport::CorrectionNewton(report) = solver
            .solve_parameter_loop(&[vec![1.0], vec![0.0], vec![2.0], vec![0.0]])
            .unwrap()
        else {
            panic!("native route lost")
        };
        assert_eq!(report.points.len(), 4);
        assert!(report.stop_error.is_none());
        assert_eq!(
            report
                .points
                .iter()
                .map(|p| p.guess_source_index)
                .collect::<Vec<_>>(),
            vec![None, Some(0), Some(0), Some(2)]
        );
        for index in [1, 3] {
            let CorrectionNewtonParameterLoopOutcome::Failed { error } =
                &report.points[index].outcome
            else {
                panic!("zero rate must fail")
            };
            assert!(
                error
                    .partial_report()
                    .unwrap()
                    .telemetry
                    .residual_evaluations
                    > 0
            );
        }
        let view = solver.solution_view().unwrap();
        assert!((view.value(16, 0).unwrap() - 1.5).abs() < 1e-10);
        let dataset = solver.postprocess_dataset().unwrap();
        assert_eq!(dataset.values.shape(), (17, 1));
        assert!((dataset.values[(16, 0)] - 1.5).abs() < 1e-10);
        let BvpSolverTelemetrySnapshot::CorrectionNewton(Some(telemetry)) =
            solver.telemetry_snapshot()
        else {
            panic!("last attempt telemetry missing")
        };
        assert!(telemetry.residual_evaluations > 0);
        // A failed shortcut must retain the same published solution.
        assert!(solver.solve_with_parameter_values(vec![0.0]).is_err());
        assert!((solver.solution_view().unwrap().value(16, 0).unwrap() - 1.5).abs() < 1e-10);
        solver.solve_with_parameter_values(vec![4.0]).unwrap();
        assert!((solver.solution_view().unwrap().value(16, 0).unwrap() - 1.25).abs() < 1e-10);
    }
}

#[test]
fn correction_newton_loop_validation_and_all_failed_points_preserve_published_profile() {
    let mut solver = solver(CorrectionNewtonSymbolicFrontend::ExprLegacy);
    solver.solve().unwrap();
    let previous = solver.postprocess_dataset().unwrap().values;
    for input in [vec![], vec![vec![1.0], vec![]], vec![vec![f64::NAN]]] {
        assert!(solver.solve_parameter_loop(&input).is_err());
        assert_eq!(solver.postprocess_dataset().unwrap().values, previous);
    }
    let BvpParameterLoopReport::CorrectionNewton(report) = solver
        .solve_parameter_loop(&[vec![0.0], vec![0.0]])
        .unwrap()
    else {
        panic!("wrong route")
    };
    assert!(report.points.iter().all(|point| matches!(
        point.outcome,
        CorrectionNewtonParameterLoopOutcome::Failed { .. }
    )));
    assert_eq!(solver.postprocess_dataset().unwrap().values, previous);
}

#[test]
fn correction_newton_multistate_loop_preserves_frontend_layout_and_facade_telemetry_parity() {
    let steps = 2048;
    let states = (0..6).map(|index| format!("y{index}")).collect::<Vec<_>>();
    for frontend in [
        CorrectionNewtonSymbolicFrontend::ExprLegacy,
        CorrectionNewtonSymbolicFrontend::AtomView,
    ] {
        for layout in [
            CorrectionNewtonMatrixLayout::Sparse,
            CorrectionNewtonMatrixLayout::BandedAuto,
        ] {
            let problem = BvpProblem::new(
                states
                    .iter()
                    .map(|state| Expr::parse_expression(&format!("-rate*{state}")))
                    .collect(),
                states.clone(),
                "x",
                (0..=steps).map(|node| node as f64 / steps as f64).collect(),
                (0..states.len())
                    .map(|state| BvpBoundaryCondition {
                        state,
                        side: BvpBoundarySide::Left,
                        value: 1.0,
                    })
                    .collect(),
            )
            .unwrap()
            .with_external_parameters(vec!["rate".into()], vec![0.2])
            .unwrap();
            let mut solver = BvpSolver::new(
                problem,
                BvpSolverConfig::CorrectionNewton {
                    options: CorrectionNewtonOptions {
                        symbolic_frontend: frontend,
                        matrix_layout: layout,
                        absolute_tolerance: 1e-10,
                        default_relative_tolerance: 1e-10,
                        telemetry_mode: CorrectionNewtonTelemetryMode::Counters,
                        ..CorrectionNewtonOptions::default()
                    },
                    initial_guess: DMatrix::from_element(states.len(), steps, 1.0),
                },
            )
            .unwrap();
            let BvpParameterLoopReport::CorrectionNewton(report) = solver
                .solve_parameter_loop(&[vec![0.2], vec![0.25], vec![0.3]])
                .unwrap()
            else {
                panic!("CorrectionNewton route did not preserve its loop report")
            };
            assert_eq!(report.points.len(), 3);
            assert!(report.stop_error.is_none());
            for (index, point) in report.points.iter().enumerate() {
                let expected_guess_source = if index == 0 { None } else { Some(index - 1) };
                assert_eq!(point.guess_source_index, expected_guess_source);
                let CorrectionNewtonParameterLoopOutcome::Converged {
                    mesh,
                    values,
                    report,
                } = &point.outcome
                else {
                    panic!("multi-state CorrectionNewton point failed")
                };
                assert_eq!(mesh.len(), steps + 1);
                assert_eq!(values.shape(), (steps + 1, states.len()));
                assert_eq!(report.telemetry.jacobian_evaluations, 1);
                let half_step = 0.5 * point.parameters[0] / steps as f64;
                let exact_endpoint = ((1.0 - half_step) / (1.0 + half_step)).powi(steps as i32);
                for state in 0..states.len() {
                    assert!((values[(0, state)] - 1.0).abs() < 1e-14);
                    assert!((values[(steps, state)] - exact_endpoint).abs() < 1e-8);
                }
            }
            let dataset = solver.postprocess_dataset().unwrap();
            assert_eq!(dataset.values.shape(), (steps + 1, states.len()));
            let BvpSolverTelemetrySnapshot::CorrectionNewton(Some(snapshot)) =
                solver.telemetry_snapshot()
            else {
                panic!("facade did not retain the final CorrectionNewton snapshot")
            };
            assert!(snapshot.residual_evaluations > 0);
        }
    }
}
