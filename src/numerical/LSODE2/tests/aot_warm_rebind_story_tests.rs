//! End-to-end AOT parameter-rebind and repeated-warm-solve gates.

use super::aot_trajectory_parity_story_tests::trajectory_config;
use super::solver::Lsode2EvaluationTelemetry;
use super::{
    IvpTelemetry, Lsode2AotProfile, Lsode2AotToolchain, Lsode2LinearSystemStructure, Lsode2Solver,
    Lsode2SymbolicAssemblyBackend, Lsode2SymbolicExecutionMode,
};
use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use crate::symbolic::symbolic_ivp_generated::{
    SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
};
use nalgebra::{DMatrix, DVector};
use tempfile::tempdir;

macro_rules! reportln {
    ($($arg:tt)*) => {
        crate::Utils::test_reporting::capture_test_line(format_args!($($arg)*));
    };
}

fn generated_config(
    output_parent: &std::path::Path,
    policy: SymbolicIvpAotBuildPolicy,
) -> SymbolicIvpGeneratedBackendConfig {
    SymbolicIvpGeneratedBackendConfig::defaults()
        .with_build_policy(policy)
        .with_output_parent_dir(Some(output_parent.to_path_buf()))
        .with_c_tcc()
}

fn parameterized_config(
    structure: Lsode2LinearSystemStructure,
    assembly: Lsode2SymbolicAssemblyBackend,
    execution: Lsode2SymbolicExecutionMode,
    generated: Option<SymbolicIvpGeneratedBackendConfig>,
    telemetry: IvpTelemetry,
    parameter: f64,
) -> super::Lsode2ProblemConfig {
    trajectory_config(structure, assembly, execution, generated, telemetry)
        .with_equation_parameter_values(DVector::from_vec(vec![parameter]))
}

fn max_matrix_diff(left: &DMatrix<f64>, right: &DMatrix<f64>) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

fn telemetry_delta(
    after: Lsode2EvaluationTelemetry,
    before: Lsode2EvaluationTelemetry,
) -> Lsode2EvaluationTelemetry {
    Lsode2EvaluationTelemetry {
        scope: after.scope,
        residual_evaluations: after
            .residual_evaluations
            .saturating_sub(before.residual_evaluations),
        jacobian_evaluations: after
            .jacobian_evaluations
            .saturating_sub(before.jacobian_evaluations),
        linear_solves: after.linear_solves.saturating_sub(before.linear_solves),
        residual_ms_total: (after.residual_ms_total - before.residual_ms_total).max(0.0),
        jacobian_ms_total: (after.jacobian_ms_total - before.jacobian_ms_total).max(0.0),
        linear_ms_total: (after.linear_ms_total - before.linear_ms_total).max(0.0),
        accepted_steps: after.accepted_steps.saturating_sub(before.accepted_steps),
        rejected_steps: after.rejected_steps.saturating_sub(before.rejected_steps),
    }
}

#[test]
fn aot_parameter_rebind_and_repeated_warm_solve_reject_stale_runtime() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_AOT",
        "numerical::LSODE2::aot_warm_rebind_story_tests::aot_parameter_rebind_and_repeated_warm_solve_reject_stale_runtime",
    );
    let output_parent = tempdir().expect("AOT output directory should exist");

    reportln!(
        "[LSODE2 AOT warm rebind] initial=BuildIfMissing(Debug); rebound=parameter 3.0; fresh=RequirePrebuilt; stale callback/factor gate"
    );
    reportln!(
        "matrix | route | rebound_vs_fresh_max_state_diff | rebound_vs_fresh_max_time_diff | rebound_counters | fresh_counters | status"
    );

    for (matrix, structure) in [
        ("Sparse", Lsode2LinearSystemStructure::Sparse),
        (
            "Banded",
            Lsode2LinearSystemStructure::Banded { kl: 0, ku: 0 },
        ),
    ] {
        let routes = [
            (
                "ExprLegacy-AOT",
                Lsode2SymbolicAssemblyBackend::ExprLegacy,
                Lsode2SymbolicExecutionMode::Aot {
                    toolchain: Lsode2AotToolchain::CTcc,
                    profile: Lsode2AotProfile::Debug,
                },
                true,
            ),
            (
                "AtomViewNative-AOT",
                Lsode2SymbolicAssemblyBackend::AtomView,
                Lsode2SymbolicExecutionMode::Aot {
                    toolchain: Lsode2AotToolchain::CTcc,
                    profile: Lsode2AotProfile::Debug,
                },
                true,
            ),
            (
                "AtomViewNative-Lambdify",
                Lsode2SymbolicAssemblyBackend::AtomView,
                Lsode2SymbolicExecutionMode::LambdifyExpr,
                false,
            ),
        ];

        for (route, assembly, execution, is_aot) in routes {
            let telemetry = IvpTelemetry::detailed();
            let initial_generated = is_aot.then(|| {
                generated_config(
                    output_parent.path(),
                    SymbolicIvpAotBuildPolicy::BuildIfMissing {
                        profile: AotBuildProfile::Debug,
                    },
                )
            });
            let mut rebound_solver = Lsode2Solver::new(parameterized_config(
                structure,
                assembly,
                execution,
                initial_generated,
                telemetry.clone(),
                2.0,
            ))
            .expect("initial prepared solver should construct");
            let initial_summary = rebound_solver
                .solve_with_summary()
                .expect("initial parameter solve should succeed");
            assert!(
                rebound_solver
                    .set_parameter_values(DVector::from_vec(vec![3.0, 4.0]))
                    .is_err(),
                "{matrix}/{route} invalid rebind must be rejected"
            );
            rebound_solver
                .set_parameter_values(DVector::from_vec(vec![3.0]))
                .expect("numeric parameter rebind should succeed");
            let rebound_summary = rebound_solver
                .solve_with_summary()
                .expect("rebound warm solve should succeed");
            let (rebound_times, rebound_values) = rebound_solver.get_result();

            let fresh_generated = is_aot.then(|| {
                generated_config(
                    output_parent.path(),
                    SymbolicIvpAotBuildPolicy::RequirePrebuilt,
                )
            });
            let mut fresh_solver = Lsode2Solver::new(parameterized_config(
                structure,
                assembly,
                execution,
                fresh_generated,
                IvpTelemetry::detailed(),
                3.0,
            ))
            .expect("fresh parameterized solver should construct");
            let fresh_summary = fresh_solver
                .solve_with_summary()
                .expect("fresh parameter solve should succeed");
            let (fresh_times, fresh_values) = fresh_solver.get_result();

            let max_time_diff = rebound_times
                .iter()
                .zip(fresh_times.iter())
                .map(|(left, right)| (left - right).abs())
                .fold(0.0, f64::max);
            let max_state_diff = max_matrix_diff(&rebound_values, &fresh_values);
            let rebound_counters = telemetry_delta(
                rebound_summary.evaluation_telemetry,
                initial_summary.evaluation_telemetry,
            );
            let fresh_counters = fresh_summary.evaluation_telemetry;
            reportln!(
                "{matrix} | {route} | {max_state_diff:.3e} | {max_time_diff:.3e} | {}/{}/{} | {}/{}/{} | ok",
                rebound_counters.residual_evaluations,
                rebound_counters.jacobian_evaluations,
                rebound_counters.linear_solves,
                fresh_counters.residual_evaluations,
                fresh_counters.jacobian_evaluations,
                fresh_counters.linear_solves,
            );

            assert!(
                max_state_diff <= 1.0e-9,
                "{matrix}/{route} stale state drift"
            );
            assert!(
                max_time_diff <= 1.0e-12,
                "{matrix}/{route} stale time trajectory"
            );
            assert_eq!(
                rebound_counters.scope, fresh_counters.scope,
                "{matrix}/{route} rebind/fresh telemetry scope"
            );
            assert_eq!(
                (
                    rebound_counters.residual_evaluations,
                    rebound_counters.jacobian_evaluations,
                    rebound_counters.linear_solves,
                    rebound_counters.accepted_steps,
                    rebound_counters.rejected_steps,
                ),
                (
                    fresh_counters.residual_evaluations,
                    fresh_counters.jacobian_evaluations,
                    fresh_counters.linear_solves,
                    fresh_counters.accepted_steps,
                    fresh_counters.rejected_steps,
                ),
                "{matrix}/{route} rebind/fresh integer trajectory counters"
            );
            assert_eq!(
                rebound_summary.algorithm, fresh_summary.algorithm,
                "{matrix}/{route} rebind/fresh algorithm trajectory"
            );
        }
    }
}
