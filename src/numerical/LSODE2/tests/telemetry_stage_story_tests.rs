//! Stories for typed preparation and callback telemetry reporting.
//!
//! This module is intentionally separate from backend correctness and
//! evaluator-policy stories. Report formatting remains outside solver timing.

use super::{Lsode2ProblemConfig, Lsode2Solver};
use crate::symbolic::ivp_telemetry::{IvpTelemetry, IvpTelemetryExecution};
use crate::symbolic::symbolic_engine::Expr;
use nalgebra::DVector;

fn exponential_decay_config() -> Lsode2ProblemConfig {
    Lsode2ProblemConfig::new(
        vec![Expr::parse_expression("-y")],
        vec!["y".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0]),
        1.0,
        0.02,
        1e-6,
        1e-8,
    )
}

#[test]
fn lsode2_lambdify_telemetry_pretty_report_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::story_tests2::tests::lsode2_lambdify_telemetry_pretty_report_story",
    );
    let telemetry = IvpTelemetry::detailed();
    let config = exponential_decay_config()
        .with_native_banded_faithful_backend()
        .with_faithful_bdf_solve(256, 256)
        .with_telemetry(telemetry.clone());
    let mut solver = Lsode2Solver::new(config).expect("telemetry story config should build");
    solver
        .solve_with_summary()
        .expect("telemetry story solve should finish");

    let snapshot = solver.telemetry_snapshot();
    println!("[LSODE2 Lambdify telemetry] typed report follows");
    println!("{}", snapshot.pretty_report());
    assert_eq!(snapshot.execution, IvpTelemetryExecution::Lambdify);
    assert!(snapshot.residual_requests > 0);
    assert!(snapshot.residual_evaluations >= snapshot.residual_requests);
    assert!(snapshot.jacobian_requests > 0);
    assert!(snapshot.jacobian_evaluations >= snapshot.jacobian_requests);
}

#[test]
fn lsode2_telemetry_schema_contract_story() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_Lambdify",
        "numerical::LSODE2::telemetry_stage_story_tests::lsode2_telemetry_schema_contract_story",
    );
    let report = IvpTelemetry::detailed().snapshot().pretty_report();
    let required_labels = [
        "validation",
        "parameter_binding",
        "expr_to_atom",
        "layout_planning",
        "aot_materialization",
        "aot_build",
        "aot_link",
        "aot_cache_lookup",
        "aot_publication",
        "parallel_calibration",
        "argument_binding",
        "residual_evaluation",
        "jacobian_evaluation",
        "residual_output_assembly",
        "jacobian_output_assembly",
        "aot_chunk_dispatch",
        "aot_worker_execution",
        "aot_argument_copy",
        "aot_output_write",
        "controller",
        "controller_iteration_inclusive",
        "factorization",
        "rhs_solve",
        "copies",
        "copied_bytes",
        "allocated_bytes",
        "parallel_dispatches",
        "aot_resolution_hits",
        "aot_resolution_misses",
        "aot_build_attempts",
        "aot_link_attempts",
        "residual_requests",
        "residual_evaluations",
        "jacobian_requests",
        "jacobian_evaluations",
    ];
    for label in required_labels {
        assert!(
            report.contains(label),
            "telemetry report is missing `{label}`"
        );
    }
    println!("[LSODE2 telemetry schema] required cold/warm stages and counters present");
    println!("{report}");
}
