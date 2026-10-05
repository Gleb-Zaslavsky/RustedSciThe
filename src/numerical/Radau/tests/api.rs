//! Public-facade stories: frontend/layout selection, continuation, output,
//! telemetry, and typed errors.

use super::super::api::{
    RadauConfig, RadauErrorKind, RadauFrontend, RadauMatrixLayout, RadauOutputPolicy, RadauProblem,
    RadauSolver, RadauTelemetryMode,
};
use crate::symbolic::symbolic_engine::Expr;

#[test]
fn public_solver_exposes_frontend_continuation_output_and_telemetry() {
    for frontend in [RadauFrontend::ExprLegacy, RadauFrontend::AtomViewNative] {
        let problem = RadauProblem::new(
            vec![Expr::parse_expression("-a*y")],
            vec!["y".to_owned()],
            "t",
        )
        .with_parameters(vec!["a".to_owned()]);
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
        let solution = solver.solve_with_parameters(&[1.0], &[1.0]).unwrap();
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

        let continued = solver.continue_with_parameters(&[1.0], &[2.0]).unwrap();
        assert!((continued.y[0] - (-2.0f64).exp()).abs() < 3.0e-4);
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
