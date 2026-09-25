//! Residual-only AOT toolchain stories.
//!
//! This module intentionally owns the real compiler/build smoke test rather
//! than keeping it in the mixed native/Lambdify story module.  The numerical
//! fixture is deliberately small; the test checks lifecycle and correctness,
//! while large cold-build measurements live in the release story reports.

use super::{Lsode2ProblemConfig, Lsode2Solver};
use crate::symbolic::codegen::codegen_aot_runtime_link::unregister_linked_residual_backend;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp::{
    SymbolicIvpProblemOptions, prepare_symbolic_ivp_residual_problem,
};
use crate::symbolic::symbolic_ivp_generated::SymbolicIvpGeneratedBackendConfig;
use nalgebra::DVector;
use std::time::Instant;
use tempfile::tempdir;

struct ResidualBackendGuard {
    problem_key: String,
}

impl Drop for ResidualBackendGuard {
    fn drop(&mut self) {
        unregister_linked_residual_backend(self.problem_key.as_str());
    }
}

struct AotBuildStoryRow {
    variant: &'static str,
    total_ms: f64,
    prepare_ms: Option<f64>,
    solve_ms: Option<f64>,
    final_t: Option<f64>,
    final_diff: Option<f64>,
    residual_calls: Option<usize>,
    jacobian_calls: Option<usize>,
    nlu: Option<usize>,
    status: String,
}

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

fn compact_status(message: &str) -> String {
    const LIMIT: usize = 140;
    let flat = message.replace(['\r', '\n'], " ");
    if flat.len() <= LIMIT {
        flat
    } else {
        format!("{}...", &flat[..LIMIT])
    }
}

fn is_native_faithful_status(status: &str) -> bool {
    status == "finished_native_faithful" || status == "finished_native_faithful_partial"
}

fn is_finished_status(status: &str) -> bool {
    status == "finished" || is_native_faithful_status(status)
}

fn with_unique_generated_names(
    mut config: Lsode2ProblemConfig,
    crate_suffix: &str,
) -> Lsode2ProblemConfig {
    config.backend.generated_backend = config
        .backend
        .generated_backend
        .with_crate_name_override(Some(format!("generated_lsode2_residual_{crate_suffix}")))
        .with_module_name_override(Some(format!("generated_lsode2_residual_{crate_suffix}")));
    config
}

fn residual_problem_key(
    config: &Lsode2ProblemConfig,
    generated_backend: &SymbolicIvpGeneratedBackendConfig,
) -> String {
    prepare_symbolic_ivp_residual_problem(
        config.eq_system.clone(),
        config.values.clone(),
        config.arg.clone(),
        SymbolicIvpProblemOptions::new().with_aot_options(generated_backend.aot_options),
    )
    .expect("story residual problem should prepare")
    .prepare_residual_aot_problem(generated_backend.aot_options)
    .problem_key()
}

fn solve_real_aot_build_story_row(
    variant: &'static str,
    config: Lsode2ProblemConfig,
) -> AotBuildStoryRow {
    let problem_key = residual_problem_key(&config, &config.backend.generated_backend);
    unregister_linked_residual_backend(problem_key.as_str());
    let _cleanup = ResidualBackendGuard { problem_key };
    let started = Instant::now();
    let result = (|| {
        let mut solver = Lsode2Solver::new(config).expect("LSODE2 real-AOT config should build");
        solver.solve_with_summary()
    })();
    let total_ms = started.elapsed().as_secs_f64() * 1_000.0;

    match result {
        Ok(summary) => {
            let final_t = summary
                .final_t
                .expect("real-AOT solve should report terminal time");
            let final_y = summary
                .final_y
                .as_ref()
                .expect("real-AOT solve should have final y")[0];
            let expected_at_final_t = (-final_t).exp();
            let faithful_status = is_native_faithful_status(&summary.status);
            let (prepare_ms, solve_ms, residual_calls, jacobian_calls, nlu) = if faithful_status {
                (
                    Some(summary.native_statistics.backend_prepare_ms_total),
                    Some(summary.native_statistics.solve_ms_total),
                    Some(summary.native_statistics.native_residual_calls),
                    Some(summary.native_statistics.native_jacobian_calls),
                    Some(summary.native_statistics.native_linear_solve_calls),
                )
            } else {
                (
                    Some(summary.statistics.backend_prepare_ms_total),
                    Some(summary.statistics.solve_ms_total),
                    Some(summary.statistics.residual_calls),
                    Some(summary.statistics.jacobian_calls),
                    Some(summary.statistics.bdf_nlu_total),
                )
            };
            AotBuildStoryRow {
                variant,
                total_ms,
                prepare_ms,
                solve_ms,
                final_t: Some(final_t),
                final_diff: Some((final_y - expected_at_final_t).abs()),
                residual_calls,
                jacobian_calls,
                nlu,
                status: summary.status,
            }
        }
        Err(err) => AotBuildStoryRow {
            variant,
            total_ms,
            prepare_ms: None,
            solve_ms: None,
            final_t: None,
            final_diff: None,
            residual_calls: None,
            jacobian_calls: None,
            nlu: None,
            status: compact_status(&err.to_string()),
        },
    }
}

fn fmt_optional_f64(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_string(), |value| format!("{value:.3}"))
}

fn fmt_optional_sci(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_string(), |value| format!("{value:.3e}"))
}

fn fmt_optional_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "-".to_string(), |value| value.to_string())
}

#[test]
fn lsode2_native_banded_real_residual_aot_build_story_table() {
    let _report = crate::Utils::test_reporting::TestReportCapture::new(
        "LSODE2_AOT",
        "numerical::LSODE2::aot_residual_story_tests::lsode2_native_banded_real_residual_aot_build_story_table",
    );
    let temp = tempdir().expect("temporary generated-AOT directory should exist");
    let rows = vec![
        solve_real_aot_build_story_row(
            "C-tcc",
            with_unique_generated_names(
                exponential_decay_config()
                    .with_native_banded_faithful_aot_c_tcc(temp.path().join("c_tcc")),
                "c_tcc",
            ),
        ),
        solve_real_aot_build_story_row(
            "C-gcc",
            with_unique_generated_names(
                exponential_decay_config()
                    .with_native_banded_faithful_aot_c_gcc(temp.path().join("c_gcc")),
                "c_gcc",
            ),
        ),
        solve_real_aot_build_story_row(
            "Zig",
            with_unique_generated_names(
                exponential_decay_config()
                    .with_native_banded_faithful_aot_zig(temp.path().join("zig")),
                "zig",
            ),
        ),
    ];

    println!(
        "[LSODE2 story] native banded residual-only real AOT build/load table; all time columns are milliseconds"
    );
    println!(
        "variant | total_ms | prepare_ms | solve_ms | final_diff | residual_calls | jacobian_calls | nlu | status"
    );
    println!(
        "----------------------------------------------------------------------------------------------------------------------------"
    );
    for row in &rows {
        println!(
            "{:<7} | {:>8.3} | {:>10} | {:>8} | {:>10} | {:>14} | {:>13} | {:>3} | {}",
            row.variant,
            row.total_ms,
            fmt_optional_f64(row.prepare_ms),
            fmt_optional_f64(row.solve_ms),
            fmt_optional_sci(row.final_diff),
            fmt_optional_usize(row.residual_calls),
            fmt_optional_usize(row.jacobian_calls),
            fmt_optional_usize(row.nlu),
            row.status,
        );
    }

    let successes = rows
        .iter()
        .filter(|row| is_finished_status(&row.status))
        .collect::<Vec<_>>();
    assert!(
        !successes.is_empty(),
        "at least one residual-only AOT compiler backend should build and run"
    );
    for row in successes {
        assert!(row.final_t.is_some());
        assert!(row.final_diff.unwrap_or(f64::INFINITY) < 1.0e-4);
        assert!(row.residual_calls.unwrap_or(0) > 0);
        assert!(row.nlu.unwrap_or(0) > 0 || row.jacobian_calls.unwrap_or(0) > 0);
    }
}
