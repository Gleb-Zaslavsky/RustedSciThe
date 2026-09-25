//! Debug AOT trajectory parity gate.
//!
//! This is intentionally a small, deterministic fixture.  It verifies that
//! the generated callback route does not change the controller trajectory
//! before larger Sparse/Banded release matrices are attempted.

use super::{
    IvpLambdifyExecutionPolicy, IvpTelemetry, Lsode2AotProfile, Lsode2AotToolchain,
    Lsode2LinearSystemStructure, Lsode2ProblemConfig, Lsode2ResidualJacobianSource,
    Lsode2SolveSummary, Lsode2Solver, Lsode2SymbolicAssemblyBackend, Lsode2SymbolicExecutionMode,
};
use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp_generated::{
    SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
};
use nalgebra::DVector;
use tempfile::tempdir;

macro_rules! reportln {
    ($($arg:tt)*) => {
        crate::Utils::test_reporting::capture_test_line(format_args!($($arg)*));
    };
}

pub(super) fn trajectory_config(
    structure: Lsode2LinearSystemStructure,
    assembly: Lsode2SymbolicAssemblyBackend,
    execution: Lsode2SymbolicExecutionMode,
    generated: Option<SymbolicIvpGeneratedBackendConfig>,
    telemetry: IvpTelemetry,
) -> Lsode2ProblemConfig {
    let base = Lsode2ProblemConfig::new(
        vec![Expr::parse_expression("-a*y")],
        vec!["y".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0]),
        0.5,
        0.025,
        1.0e-9,
        1.0e-11,
    )
    .with_equation_parameters(vec!["a".to_string()])
    .with_equation_parameter_values(DVector::from_vec(vec![2.0]))
    .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
        assembly,
        execution,
    })
    .with_faithful_bdf_solve(2_000, 2_000)
    .with_lambdify_execution_policy(IvpLambdifyExecutionPolicy::Sequential)
    .with_telemetry(telemetry);

    match (structure, generated) {
        (Lsode2LinearSystemStructure::Sparse, Some(generated)) => {
            base.with_native_sparse_faer_generated_backend(generated)
        }
        (Lsode2LinearSystemStructure::Sparse, None) => base.with_native_sparse_faer_backend(),
        (Lsode2LinearSystemStructure::Banded { .. }, Some(generated)) => {
            base.with_native_banded_faithful_generated_backend(generated)
        }
        (Lsode2LinearSystemStructure::Banded { .. }, None) => {
            base.with_native_banded_faithful_backend()
        }
        (Lsode2LinearSystemStructure::Dense, _) => {
            panic!("the trajectory parity gate intentionally excludes Dense")
        }
    }
}

#[test]
fn aot_trajectory_parity_matches_exprlegacy_atomview_and_lambdify() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_AOT",
        "numerical::LSODE2::aot_trajectory_parity_story_tests::aot_trajectory_parity_matches_exprlegacy_atomview_and_lambdify",
    );
    let output_parent = tempdir().expect("AOT output directory should exist");
    let generated = || {
        SymbolicIvpGeneratedBackendConfig::defaults()
            .with_build_policy(SymbolicIvpAotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Debug,
            })
            .with_output_parent_dir(Some(output_parent.path().to_path_buf()))
            .with_c_tcc()
    };

    let expr_telemetry = IvpTelemetry::detailed();
    let atom_telemetry = IvpTelemetry::detailed();
    let lambdify_telemetry = IvpTelemetry::detailed();
    let mut expr = Lsode2Solver::new(trajectory_config(
        Lsode2LinearSystemStructure::Banded { kl: 0, ku: 0 },
        Lsode2SymbolicAssemblyBackend::ExprLegacy,
        Lsode2SymbolicExecutionMode::Aot {
            toolchain: Lsode2AotToolchain::CTcc,
            profile: Lsode2AotProfile::Debug,
        },
        Some(generated()),
        expr_telemetry.clone(),
    ))
    .expect("ExprLegacy AOT trajectory fixture should construct");
    let mut atom = Lsode2Solver::new(trajectory_config(
        Lsode2LinearSystemStructure::Banded { kl: 0, ku: 0 },
        Lsode2SymbolicAssemblyBackend::AtomView,
        Lsode2SymbolicExecutionMode::Aot {
            toolchain: Lsode2AotToolchain::CTcc,
            profile: Lsode2AotProfile::Debug,
        },
        Some(generated()),
        atom_telemetry.clone(),
    ))
    .expect("AtomViewNative AOT trajectory fixture should construct");
    let mut lambdify = Lsode2Solver::new(trajectory_config(
        Lsode2LinearSystemStructure::Banded { kl: 0, ku: 0 },
        Lsode2SymbolicAssemblyBackend::AtomView,
        Lsode2SymbolicExecutionMode::LambdifyExpr,
        None,
        lambdify_telemetry.clone(),
    ))
    .expect("AtomViewNative Lambdify trajectory fixture should construct");

    let expr_summary = expr
        .solve_with_summary()
        .expect("ExprLegacy AOT trajectory fixture should solve");
    let atom_summary = atom
        .solve_with_summary()
        .expect("AtomViewNative AOT trajectory fixture should solve");
    let lambdify_summary = lambdify
        .solve_with_summary()
        .expect("AtomViewNative Lambdify trajectory fixture should solve");
    let (expr_times, expr_values) = expr.get_result();
    let (atom_times, atom_values) = atom.get_result();
    let (lambdify_times, lambdify_values) = lambdify.get_result();

    let max_time_diff = expr_times
        .iter()
        .zip(atom_times.iter())
        .map(|(left, right)| (left - right).abs())
        .chain(
            expr_times
                .iter()
                .zip(lambdify_times.iter())
                .map(|(left, right)| (left - right).abs()),
        )
        .fold(0.0, f64::max);
    let max_state_diff = expr_values
        .iter()
        .zip(atom_values.iter())
        .map(|(left, right)| (left - right).abs())
        .chain(
            expr_values
                .iter()
                .zip(lambdify_values.iter())
                .map(|(left, right)| (left - right).abs()),
        )
        .fold(0.0, f64::max);
    let expr_evaluation = expr_summary.evaluation_telemetry;
    let atom_evaluation = atom_summary.evaluation_telemetry;
    let lambdify_evaluation = lambdify_summary.evaluation_telemetry;
    let expr_callback = expr_telemetry.snapshot();
    let atom_callback = atom_telemetry.snapshot();
    let lambdify_callback = lambdify_telemetry.snapshot();

    reportln!(
        "[LSODE2 AOT trajectory parity] fixture=scalar-parameterized; matrix=Banded; max_time_diff={max_time_diff:.3e}; max_state_diff={max_state_diff:.3e}"
    );
    reportln!(
        "route | residual_calls | jacobian_calls | linear_solves | jacobian_rebuilds | accepted | rejected | status"
    );
    for (route, telemetry, rebuilds, status) in [
        (
            "ExprLegacy-AOT",
            expr_evaluation,
            expr_callback.jacobian_rebuilds,
            expr_summary.status.as_str(),
        ),
        (
            "AtomViewNative-AOT",
            atom_evaluation,
            atom_callback.jacobian_rebuilds,
            atom_summary.status.as_str(),
        ),
        (
            "AtomViewNative-Lambdify",
            lambdify_evaluation,
            lambdify_callback.jacobian_rebuilds,
            lambdify_summary.status.as_str(),
        ),
    ] {
        reportln!(
            "{route} | {} | {} | {} | {} | {} | {} | {status}",
            telemetry.residual_evaluations,
            telemetry.jacobian_evaluations,
            telemetry.linear_solves,
            rebuilds,
            telemetry.accepted_steps,
            telemetry.rejected_steps,
        );
    }

    assert!(max_time_diff <= 1.0e-12);
    assert!(max_state_diff <= 1.0e-9);
    assert_eq!(expr_times.len(), atom_times.len());
    assert_eq!(expr_times.len(), lambdify_times.len());
    assert_eq!(expr_values.shape(), atom_values.shape());
    assert_eq!(expr_values.shape(), lambdify_values.shape());
    for (label, candidate) in [
        ("AtomViewNative-AOT", atom_evaluation),
        ("AtomViewNative-Lambdify", lambdify_evaluation),
    ] {
        assert_eq!(
            expr_evaluation.residual_evaluations, candidate.residual_evaluations,
            "{label} residual trajectory count"
        );
        assert_eq!(
            expr_evaluation.jacobian_evaluations, candidate.jacobian_evaluations,
            "{label} Jacobian trajectory count"
        );
        assert_eq!(
            expr_evaluation.linear_solves, candidate.linear_solves,
            "{label} linear trajectory count"
        );
        let candidate_callback = if label == "AtomViewNative-AOT" {
            &atom_callback
        } else {
            &lambdify_callback
        };
        assert_eq!(
            expr_callback.jacobian_rebuilds, candidate_callback.jacobian_rebuilds,
            "{label} Jacobian refresh/reuse trajectory count"
        );
        assert_eq!(
            expr_evaluation.accepted_steps, candidate.accepted_steps,
            "{label} accepted-step trajectory count"
        );
        assert_eq!(
            expr_evaluation.rejected_steps, candidate.rejected_steps,
            "{label} rejected-step trajectory count"
        );
    }
    assert_eq!(expr_summary.algorithm, atom_summary.algorithm);
    assert_eq!(expr_summary.algorithm, lambdify_summary.algorithm);
    assert_eq!(
        retry_trace(&expr_summary),
        retry_trace(&atom_summary),
        "AtomViewNative-AOT retry trajectory"
    );
    assert_eq!(
        retry_trace(&expr_summary),
        retry_trace(&lambdify_summary),
        "AtomViewNative-Lambdify retry trajectory"
    );
}

fn retry_trace(summary: &Lsode2SolveSummary) -> Vec<String> {
    summary
        .native_integration_solve
        .as_ref()
        .expect("faithful trajectory gate should expose native integration reports")
        .attempt_reports
        .iter()
        .map(|report| {
            format!(
                "{}:{}:{}:{}:{:?}:{:?}:{:?}:{}",
                report.outcome_label(),
                report.retry_count,
                report.jacobian_refresh_retry_count,
                report.kflag_code,
                report.icf,
                report.iredo,
                report.redo_stage,
                report.ialth,
            )
        })
        .collect()
}

#[test]
fn aot_trajectory_parity_covers_sparse_and_banded_routes() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_AOT",
        "numerical::LSODE2::aot_trajectory_parity_story_tests::aot_trajectory_parity_covers_sparse_and_banded_routes",
    );
    let output_parent = tempdir().expect("AOT output directory should exist");
    let generated = || {
        SymbolicIvpGeneratedBackendConfig::defaults()
            .with_build_policy(SymbolicIvpAotBuildPolicy::BuildIfMissing {
                profile: AotBuildProfile::Debug,
            })
            .with_output_parent_dir(Some(output_parent.path().to_path_buf()))
            .with_c_tcc()
    };

    for (label, structure) in [
        ("Sparse", Lsode2LinearSystemStructure::Sparse),
        (
            "Banded",
            Lsode2LinearSystemStructure::Banded { kl: 0, ku: 0 },
        ),
    ] {
        let expr_telemetry = IvpTelemetry::detailed();
        let atom_telemetry = IvpTelemetry::detailed();
        let lambdify_telemetry = IvpTelemetry::detailed();
        let mut expr = Lsode2Solver::new(trajectory_config(
            structure,
            Lsode2SymbolicAssemblyBackend::ExprLegacy,
            Lsode2SymbolicExecutionMode::Aot {
                toolchain: Lsode2AotToolchain::CTcc,
                profile: Lsode2AotProfile::Debug,
            },
            Some(generated()),
            expr_telemetry,
        ))
        .expect("ExprLegacy production-structure fixture should construct");
        let mut atom = Lsode2Solver::new(trajectory_config(
            structure,
            Lsode2SymbolicAssemblyBackend::AtomView,
            Lsode2SymbolicExecutionMode::Aot {
                toolchain: Lsode2AotToolchain::CTcc,
                profile: Lsode2AotProfile::Debug,
            },
            Some(generated()),
            atom_telemetry,
        ))
        .expect("AtomViewNative production-structure fixture should construct");
        let mut lambdify = Lsode2Solver::new(trajectory_config(
            structure,
            Lsode2SymbolicAssemblyBackend::AtomView,
            Lsode2SymbolicExecutionMode::LambdifyExpr,
            None,
            lambdify_telemetry,
        ))
        .expect("AtomViewNative Lambdify production-structure fixture should construct");

        let expr_summary = expr
            .solve_with_summary()
            .expect("ExprLegacy production-structure fixture should solve");
        let atom_summary = atom
            .solve_with_summary()
            .expect("AtomViewNative production-structure fixture should solve");
        let lambdify_summary = lambdify
            .solve_with_summary()
            .expect("AtomViewNative Lambdify production-structure fixture should solve");
        let (expr_times, expr_values) = expr.get_result();
        let (atom_times, atom_values) = atom.get_result();
        let (lambdify_times, lambdify_values) = lambdify.get_result();
        let max_time_diff = max_vector_diff(&expr_times, &atom_times)
            .max(max_vector_diff(&expr_times, &lambdify_times));
        let max_state_diff = max_matrix_diff(&expr_values, &atom_values)
            .max(max_matrix_diff(&expr_values, &lambdify_values));

        reportln!(
            "[LSODE2 AOT production trajectory parity] matrix={label}; max_time_diff={max_time_diff:.3e}; max_state_diff={max_state_diff:.3e}; retry_events={}",
            retry_trace(&expr_summary)
                .iter()
                .filter(|event| event.contains("rejected"))
                .count()
        );
        assert!(max_time_diff <= 1.0e-12, "{label} time trajectory drift");
        assert!(max_state_diff <= 1.0e-9, "{label} state trajectory drift");
        assert_eq!(
            retry_trace(&expr_summary),
            retry_trace(&atom_summary),
            "{label} AtomViewNative-AOT retry trajectory"
        );
        assert_eq!(
            retry_trace(&expr_summary),
            retry_trace(&lambdify_summary),
            "{label} AtomViewNative-Lambdify retry trajectory"
        );
    }
}

fn max_vector_diff(left: &nalgebra::DVector<f64>, right: &nalgebra::DVector<f64>) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

fn max_matrix_diff(left: &nalgebra::DMatrix<f64>, right: &nalgebra::DMatrix<f64>) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}
