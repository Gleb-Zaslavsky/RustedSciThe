use super::{
    Lsode2DstodaState, Lsode2Icf, Lsode2Ipup, Lsode2IpupTrigger, Lsode2Iret, Lsode2IterationMode,
    Lsode2JacobianCurrency, Lsode2Kflag, Lsode2NativeStatistics, Lsode2ProblemConfig,
    Lsode2RedoStage, Lsode2SolveSummary, Lsode2Solver,
};
use crate::symbolic::codegen::codegen_aot_runtime_link::{
    LinkedResidualAotBackend, register_linked_residual_backend, unregister_linked_residual_backend,
};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp::{
    SymbolicIvpProblemOptions, prepare_symbolic_ivp_residual_problem,
};
use crate::symbolic::symbolic_ivp_generated::SymbolicIvpGeneratedBackendConfig;
use nalgebra::DVector;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use tempfile::tempdir;

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

struct NativeQualityDashboardRow {
    path: &'static str,
    matrix: &'static str,
    summary: Lsode2SolveSummary,
    total_ms: f64,
    final_diff: f64,
    rel_final_diff: f64,
    reached_t_bound: Option<bool>,
    final_t: f64,
    accepted_steps: Option<usize>,
    rejected_steps: Option<usize>,
    total_iterations: Option<usize>,
    first_jcur: Option<Lsode2JacobianCurrency>,
    first_ipup: Option<Lsode2Ipup>,
    first_predictor_ipup_trigger: Option<Lsode2IpupTrigger>,
    first_ipup_trigger: Option<Lsode2IpupTrigger>,
    first_kflag: Option<Lsode2Kflag>,
    first_kflag_code: Option<i32>,
    first_icf: Option<Lsode2Icf>,
    first_iret: Option<Lsode2Iret>,
    first_redo: Option<Lsode2RedoStage>,
    first_iredo_code: Option<i32>,
    last_jcur: Option<Lsode2JacobianCurrency>,
    last_ipup: Option<Lsode2Ipup>,
    last_predictor_ipup_trigger: Option<Lsode2IpupTrigger>,
    last_ipup_trigger: Option<Lsode2IpupTrigger>,
    last_kflag: Option<Lsode2Kflag>,
    last_kflag_code: Option<i32>,
    last_icf: Option<Lsode2Icf>,
    last_iret: Option<Lsode2Iret>,
    last_redo: Option<Lsode2RedoStage>,
    last_iredo_code: Option<i32>,
    first_ialth: Option<usize>,
    last_ialth: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
struct Aggregate {
    mean: f64,
    stddev: f64,
    min: f64,
    max: f64,
}

impl Aggregate {
    fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self {
                mean: f64::NAN,
                stddev: f64::NAN,
                min: f64::NAN,
                max: f64::NAN,
            };
        }
        let count = values.len() as f64;
        let mean = values.iter().sum::<f64>() / count;
        let variance = values
            .iter()
            .map(|value| {
                let diff = *value - mean;
                diff * diff
            })
            .sum::<f64>()
            / count;
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Self {
            mean,
            stddev: variance.sqrt(),
            min,
            max,
        }
    }
}

#[derive(Debug, Clone)]
struct NativeDashboardAggregateRow {
    path: &'static str,
    matrix: &'static str,
    resolved_source: &'static str,
    resolved_structure: &'static str,
    linear_solver_backend: &'static str,
    linear_solver_reason: &'static str,
    controller_mode: &'static str,
    preferred_family: &'static str,
    switch_reason: &'static str,
    runs: usize,
    ok_runs: usize,
    total_ms: Aggregate,
    final_t: Aggregate,
    final_diff: Aggregate,
    rel_final_diff: Aggregate,
    reached_t_bound_ratio: Option<Aggregate>,
    accepted_steps: Option<Aggregate>,
    rejected_steps: Option<Aggregate>,
    native_step_attempts: Option<Aggregate>,
}

fn solve_real_aot_build_story_row(
    variant: &'static str,
    config: Lsode2ProblemConfig,
    _reference_final_y: f64,
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

fn compact_status(message: &str) -> String {
    const LIMIT: usize = 140;
    let flat = message.replace(['\r', '\n'], " ");
    if flat.len() <= LIMIT {
        flat
    } else {
        format!("{}...", &flat[..LIMIT])
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

fn fmt_optional_ms(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_string(), |value| format!("{value:.3}"))
}

fn is_native_faithful_status(status: &str) -> bool {
    status == "finished_native_faithful" || status == "finished_native_faithful_partial"
}

fn is_finished_status(status: &str) -> bool {
    status == "finished" || is_native_faithful_status(status)
}

fn fmt_agg_ms(value: Aggregate) -> String {
    if value.mean.is_finite() {
        format!(
            "{:.3} +/- {:.3} [{:.3}, {:.3}]",
            value.mean, value.stddev, value.min, value.max
        )
    } else {
        "-".to_string()
    }
}

fn fmt_agg_sci(value: Aggregate) -> String {
    if value.mean.is_finite() {
        format!(
            "{:.3e} +/- {:.1e} [{:.3e}, {:.3e}]",
            value.mean, value.stddev, value.min, value.max
        )
    } else {
        "-".to_string()
    }
}

fn fmt_optional_agg_count(value: Option<Aggregate>) -> String {
    value.map_or_else(
        || "-".to_string(),
        |value| {
            if value.mean.is_finite() {
                format!("{:.2} +/- {:.2}", value.mean, value.stddev)
            } else {
                "-".to_string()
            }
        },
    )
}

fn fmt_optional_ratio_percent(value: Option<Aggregate>) -> String {
    value.map_or_else(
        || "-".to_string(),
        |value| {
            if value.mean.is_finite() {
                format!(
                    "{:.1}% +/- {:.1}%",
                    value.mean * 100.0,
                    value.stddev * 100.0
                )
            } else {
                "-".to_string()
            }
        },
    )
}

fn fmt_jcur(value: Option<Lsode2JacobianCurrency>) -> &'static str {
    match value {
        Some(Lsode2JacobianCurrency::Current) => "current",
        Some(Lsode2JacobianCurrency::Stale) => "stale",
        None => "-",
    }
}

fn fmt_ipup(value: Option<Lsode2Ipup>) -> &'static str {
    match value {
        Some(Lsode2Ipup::UpToDate) => "up_to_date",
        Some(Lsode2Ipup::NeedsJacobianUpdate) => "needs_update",
        None => "-",
    }
}

fn fmt_ipup_trigger(value: Option<Lsode2IpupTrigger>) -> &'static str {
    match value {
        Some(Lsode2IpupTrigger::None) => "none",
        Some(Lsode2IpupTrigger::PredictorRcCcmax) => "rc",
        Some(Lsode2IpupTrigger::PredictorMsbp) => "msbp",
        Some(Lsode2IpupTrigger::PredictorRcCcmaxAndMsbp) => "rc+msbp",
        Some(Lsode2IpupTrigger::FailurePath) => "failure",
        None => "-",
    }
}

fn fmt_kflag(value: Option<Lsode2Kflag>) -> &'static str {
    match value {
        Some(Lsode2Kflag::Ok) => "ok",
        Some(Lsode2Kflag::ErrorTestFailure) => "err_test",
        Some(Lsode2Kflag::ConvergenceFailure) => "conv",
        Some(Lsode2Kflag::RepeatedErrorTestFailure) => "err_rep",
        Some(Lsode2Kflag::RepeatedConvergenceFailure) => "conv_rep",
        None => "-",
    }
}

fn fmt_icf(value: Option<Lsode2Icf>) -> &'static str {
    match value {
        Some(Lsode2Icf::None) => "none",
        Some(Lsode2Icf::RefreshRequested) => "refresh",
        Some(Lsode2Icf::RefreshDidNotRecover) => "no_recover",
        None => "-",
    }
}

fn fmt_iret(value: Option<Lsode2Iret>) -> &'static str {
    match value {
        Some(Lsode2Iret::NormalFlow) => "normal",
        Some(Lsode2Iret::RescaleHistory) => "rescale",
        Some(Lsode2Iret::RetryAfterErrorTestFailure) => "retry_err",
        Some(Lsode2Iret::RestartWithDerivativeRefresh) => "restart",
        None => "-",
    }
}

fn fmt_redo(value: Option<Lsode2RedoStage>) -> &'static str {
    match value {
        Some(Lsode2RedoStage::None) => "none",
        Some(Lsode2RedoStage::ErrorTestRetry) => "err_retry",
        Some(Lsode2RedoStage::RepeatedErrorReset) => "err_reset",
        Some(Lsode2RedoStage::CorrectorFailureRetry) => "corr_retry",
        Some(Lsode2RedoStage::CorrectorRefreshSameStep) => "corr_refresh",
        Some(Lsode2RedoStage::HistoryOrStepSizeChanged) => "history",
        None => "-",
    }
}

fn solve_native_quality_row(
    path: &'static str,
    matrix: &'static str,
    config: Lsode2ProblemConfig,
    _reference_final_y: f64,
) -> NativeQualityDashboardRow {
    let mut solver = Lsode2Solver::new(config).expect("LSODE2 native-quality config should build");
    let started = Instant::now();
    let summary = solver
        .solve_with_summary()
        .expect("LSODE2 native-quality solve should finish");
    let total_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let final_y = summary
        .final_y
        .as_ref()
        .expect("native-quality solve should have final y")[0];

    let (
        reached_t_bound,
        accepted_steps,
        rejected_steps,
        total_iterations,
        first_jcur,
        first_ipup,
        first_predictor_ipup_trigger,
        first_ipup_trigger,
        first_kflag,
        first_kflag_code,
        first_icf,
        first_iret,
        first_redo,
        first_iredo_code,
        last_jcur,
        last_ipup,
        last_predictor_ipup_trigger,
        last_ipup_trigger,
        last_kflag,
        last_kflag_code,
        last_icf,
        last_iret,
        last_redo,
        last_iredo_code,
        first_ialth,
        last_ialth,
    ) = if let Some(native) = summary.native_integration_solve.as_ref() {
        (
            Some(native.reached_t_bound),
            Some(native.accepted_steps),
            Some(native.rejected_steps),
            Some(native.total_iterations),
            Some(native.first_report.jcur),
            Some(native.first_report.ipup),
            Some(native.first_report.predictor_ipup_trigger),
            Some(native.first_report.ipup_trigger),
            Some(native.first_report.kflag),
            Some(native.first_report.kflag_code),
            Some(native.first_report.icf),
            Some(native.first_report.iret),
            Some(native.first_report.redo_stage),
            Some(native.first_report.iredo.code()),
            Some(native.last_report.jcur),
            Some(native.last_report.ipup),
            Some(native.last_report.predictor_ipup_trigger),
            Some(native.last_report.ipup_trigger),
            Some(native.last_report.kflag),
            Some(native.last_report.kflag_code),
            Some(native.last_report.icf),
            Some(native.last_report.iret),
            Some(native.last_report.redo_stage),
            Some(native.last_report.iredo.code()),
            Some(native.first_report.ialth),
            Some(native.last_report.ialth),
        )
    } else {
        (
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None, None,
        )
    };

    let final_t = summary
        .final_t
        .expect("native-quality solve should report final t");
    // This dashboard uses the fixed test fixture y' = -y, y(0)=1.
    // Compare quality at the same terminal time reached by each path.
    let expected_at_final_t = (-final_t).exp();
    let final_diff = (final_y - expected_at_final_t).abs();
    let rel_final_diff = final_diff / expected_at_final_t.abs().max(1.0e-15);

    NativeQualityDashboardRow {
        path,
        matrix,
        summary,
        total_ms,
        final_diff,
        rel_final_diff,
        reached_t_bound,
        final_t,
        accepted_steps,
        rejected_steps,
        total_iterations,
        first_jcur,
        first_ipup,
        first_predictor_ipup_trigger,
        first_ipup_trigger,
        first_kflag,
        first_kflag_code,
        first_icf,
        first_iret,
        first_redo,
        first_iredo_code,
        last_jcur,
        last_ipup,
        last_predictor_ipup_trigger,
        last_ipup_trigger,
        last_kflag,
        last_kflag_code,
        last_icf,
        last_iret,
        last_redo,
        last_iredo_code,
        first_ialth,
        last_ialth,
    }
}

fn build_native_adams_dashboard_case_config_with_limits(
    path: &'static str,
    matrix: &'static str,
    max_step_attempts: usize,
    max_accepted_steps: usize,
) -> Lsode2ProblemConfig {
    let sparse_auto = || {
        exponential_decay_config()
            .with_linear_system_structure(super::Lsode2LinearSystemStructure::Sparse)
            .with_linear_solver_policy(super::Lsode2LinearSolverPolicy::Auto)
    };
    let banded_auto = || {
        exponential_decay_config()
            .with_linear_system_structure(super::Lsode2LinearSystemStructure::Banded {
                kl: 0,
                ku: 0,
            })
            .with_linear_solver_policy(super::Lsode2LinearSolverPolicy::Auto)
    };

    match (path, matrix) {
        ("BridgeBaseline-Bdf", "Sparse") => sparse_auto()
            .with_controller(super::algorithm::Lsode2ControllerConfig::bdf_only())
            .with_native_execution(super::Lsode2NativeExecutionConfig::bridge_solve()),
        ("NativeFaithful-Bdf", "Sparse") => sparse_auto()
            .with_controller(super::algorithm::Lsode2ControllerConfig::bdf_only())
            .with_faithful_bdf_solve(max_step_attempts, max_accepted_steps),
        ("NativeFaithful-Adams", "Sparse") => sparse_auto()
            .with_adams_only_controller()
            .with_faithful_bdf_solve(max_step_attempts, max_accepted_steps),
        ("BridgeBaseline-Bdf", "Banded") => banded_auto()
            .with_controller(super::algorithm::Lsode2ControllerConfig::bdf_only())
            .with_native_execution(super::Lsode2NativeExecutionConfig::bridge_solve()),
        ("NativeFaithful-Bdf", "Banded") => banded_auto()
            .with_controller(super::algorithm::Lsode2ControllerConfig::bdf_only())
            .with_faithful_bdf_solve(max_step_attempts, max_accepted_steps),
        ("NativeFaithful-Adams", "Banded") => banded_auto()
            .with_adams_only_controller()
            .with_faithful_bdf_solve(max_step_attempts, max_accepted_steps),
        _ => panic!("unsupported native Adams dashboard case: path={path}, matrix={matrix}"),
    }
}

fn build_native_adams_dashboard_case_config(
    path: &'static str,
    matrix: &'static str,
) -> Lsode2ProblemConfig {
    build_native_adams_dashboard_case_config_with_limits(path, matrix, 128, 128)
}

fn aggregate_native_dashboard_rows(
    rows: &[NativeQualityDashboardRow],
    path: &'static str,
    matrix: &'static str,
) -> NativeDashboardAggregateRow {
    let case_rows = rows
        .iter()
        .filter(|row| row.path == path && row.matrix == matrix)
        .collect::<Vec<_>>();
    let first = case_rows
        .first()
        .expect("native dashboard case should contain at least one row");
    let runs = case_rows.len();
    let ok_runs = case_rows
        .iter()
        .filter(|row| {
            if path.starts_with("BridgeBaseline") {
                row.summary.status == "finished"
            } else {
                row.summary.status == "finished_native_faithful"
                    || row.summary.status == "finished_native_faithful_partial"
            }
        })
        .count();
    let accepted_values = case_rows
        .iter()
        .filter_map(|row| row.accepted_steps.map(|value| value as f64))
        .collect::<Vec<_>>();
    let rejected_values = case_rows
        .iter()
        .filter_map(|row| row.rejected_steps.map(|value| value as f64))
        .collect::<Vec<_>>();
    let native_attempts_values = case_rows
        .iter()
        .map(|row| row.summary.native_statistics.native_step_attempts as f64)
        .collect::<Vec<_>>();
    let reached_ratio_values = case_rows
        .iter()
        .filter_map(|row| {
            row.reached_t_bound
                .map(|value| if value { 1.0 } else { 0.0 })
        })
        .collect::<Vec<_>>();

    NativeDashboardAggregateRow {
        path,
        matrix,
        resolved_source: first.summary.resolved_source,
        resolved_structure: first.summary.resolved_structure,
        linear_solver_backend: first.summary.linear_solver_backend,
        linear_solver_reason: first.summary.linear_solver_reason,
        controller_mode: first.summary.algorithm.controller_mode,
        preferred_family: first.summary.algorithm.preferred_family,
        switch_reason: first.summary.algorithm.switch_reason,
        runs,
        ok_runs,
        total_ms: Aggregate::from_values(
            &case_rows.iter().map(|row| row.total_ms).collect::<Vec<_>>(),
        ),
        final_t: Aggregate::from_values(
            &case_rows.iter().map(|row| row.final_t).collect::<Vec<_>>(),
        ),
        final_diff: Aggregate::from_values(
            &case_rows
                .iter()
                .map(|row| row.final_diff)
                .collect::<Vec<_>>(),
        ),
        rel_final_diff: Aggregate::from_values(
            &case_rows
                .iter()
                .map(|row| row.rel_final_diff)
                .collect::<Vec<_>>(),
        ),
        reached_t_bound_ratio: (!reached_ratio_values.is_empty())
            .then(|| Aggregate::from_values(&reached_ratio_values)),
        accepted_steps: (!accepted_values.is_empty())
            .then(|| Aggregate::from_values(&accepted_values)),
        rejected_steps: (!rejected_values.is_empty())
            .then(|| Aggregate::from_values(&rejected_values)),
        native_step_attempts: (!native_attempts_values.is_empty())
            .then(|| Aggregate::from_values(&native_attempts_values)),
    }
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

#[test]
//#[ignore = "builds and loads real residual-only AOT artifacts; requires local C/Zig toolchains"]
fn lsode2_native_banded_real_residual_aot_build_story_table() {
    let mut reference_solver = Lsode2Solver::new(exponential_decay_config())
        .expect("reference LSODE2 config should build");
    let reference = reference_solver
        .solve_with_summary()
        .expect("reference LSODE2 solve should finish");
    let reference_final_y = reference
        .final_y
        .as_ref()
        .expect("reference solve should have final y")[0];
    let temp = tempdir().expect("temporary generated-AOT directory should exist");

    let rows = vec![
        solve_real_aot_build_story_row(
            "C-tcc",
            with_unique_generated_names(
                exponential_decay_config()
                    .with_native_banded_faithful_aot_c_tcc(temp.path().join("c_tcc")),
                "c_tcc",
            ),
            reference_final_y,
        ),
        solve_real_aot_build_story_row(
            "C-gcc",
            with_unique_generated_names(
                exponential_decay_config()
                    .with_native_banded_faithful_aot_c_gcc(temp.path().join("c_gcc")),
                "c_gcc",
            ),
            reference_final_y,
        ),
        solve_real_aot_build_story_row(
            "Zig",
            with_unique_generated_names(
                exponential_decay_config()
                    .with_native_banded_faithful_aot_zig(temp.path().join("zig")),
                "zig",
            ),
            reference_final_y,
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
        "at least one residual-only AOT compiler backend should build and run; rows printed above contain toolchain/build errors"
    );
    for row in successes {
        assert!(
            row.final_diff.unwrap_or(f64::INFINITY) < 1.0e-4,
            "{} real residual-AOT solve drifted from analytical y(t) on its own final_t: diff={:?}, final_t={:?}, status={}",
            row.variant,
            row.final_diff,
            row.final_t,
            row.status
        );
        assert!(
            row.residual_calls.unwrap_or(0) > 0,
            "{} real residual-AOT solve should evaluate residuals",
            row.variant
        );
        assert!(
            row.nlu.unwrap_or(0) > 0 || row.jacobian_calls.unwrap_or(0) > 0,
            "{} real residual-AOT solve should execute Jacobian/linear work",
            row.variant
        );
    }
}

#[test]
fn lsode2_native_quality_dashboard_bridge_vs_faithful_native() {
    let mut reference_solver = Lsode2Solver::new(exponential_decay_config())
        .expect("reference LSODE2 config should build");
    let reference = reference_solver
        .solve_with_summary()
        .expect("reference LSODE2 solve should finish");
    let reference_final_y = reference
        .final_y
        .as_ref()
        .expect("reference solve should have final y")[0];

    let rows = vec![
        solve_native_quality_row(
            "Bridge",
            "Sparse",
            exponential_decay_config()
                .with_native_sparse_faer_backend()
                .with_bridge_solve(),
            reference_final_y,
        ),
        solve_native_quality_row(
            "NativeFaithful",
            "Sparse",
            exponential_decay_config()
                .with_native_sparse_faer_backend()
                .with_faithful_bdf_solve(128, 128),
            reference_final_y,
        ),
        solve_native_quality_row(
            "Bridge",
            "Banded",
            exponential_decay_config()
                .with_native_banded_faithful_backend()
                .with_bridge_solve(),
            reference_final_y,
        ),
        solve_native_quality_row(
            "NativeFaithful",
            "Banded",
            exponential_decay_config()
                .with_native_banded_faithful_backend()
                .with_faithful_bdf_solve(128, 128),
            reference_final_y,
        ),
    ];

    println!(
        "[LSODE2 story] native quality dashboard: bridge solve vs faithful native solve; all time columns are milliseconds"
    );
    println!(
        "path               | matrix | resolved_struct | linear_solver                 | linear_reason                  | status                             | total_ms | final_t   | reached | final_diff | rel_final_diff | accepted | rejected | total_iters"
    );
    println!(
        "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for row in &rows {
        println!(
            "{:<18} | {:<6} | {:<15} | {:<29} | {:<30} | {:<34} | {:>8.3} | {:>8.3e} | {:>7} | {:>10.3e} | {:>14.3e} | {:>8} | {:>8} | {:>11}",
            row.path,
            row.matrix,
            row.summary.resolved_structure,
            row.summary.linear_solver_backend,
            row.summary.linear_solver_reason,
            row.summary.status,
            row.total_ms,
            row.final_t,
            row.reached_t_bound
                .map(|value| if value { "yes" } else { "no" })
                .unwrap_or("-"),
            row.final_diff,
            row.rel_final_diff,
            fmt_optional_usize(row.accepted_steps),
            fmt_optional_usize(row.rejected_steps),
            fmt_optional_usize(row.total_iterations),
        );
    }

    println!(
        "[LSODE2 story] native quality dashboard timings: bridge/native counters are milliseconds"
    );
    println!(
        "path               | matrix | native_solve_ms | native_residual_ms | native_jacobian_ms | native_linear_ms | bridge_solve_ms | bridge_nlu"
    );
    println!(
        "------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for row in &rows {
        println!(
            "{:<18} | {:<6} | {:>15.3} | {:>18.3} | {:>18.3} | {:>16.3} | {:>15} | {:>10}",
            row.path,
            row.matrix,
            row.summary.native_statistics.solve_ms_total,
            row.summary.native_statistics.native_residual_ms_total,
            row.summary.native_statistics.native_jacobian_ms_total,
            row.summary.native_statistics.native_linear_solve_ms_total,
            fmt_optional_ms(
                (row.path == "Bridge").then_some(row.summary.statistics.solve_ms_total)
            ),
            if row.path == "Bridge" {
                row.summary.statistics.bdf_nlu_total.to_string()
            } else {
                "-".to_string()
            },
        );
    }

    println!(
        "[LSODE2 story] native quality dashboard ODEPACK-style flags (JCUR/IPUP/IPUP_REASON/KFLAG/ICF/IRET/REDO): first vs last attempt"
    );
    println!(
        "path               | matrix | first_jcur | first_ipup | first_pred_reason | first_ipup_reason | first_kflag | first_kcode | first_icf  | first_iret | first_redo  | first_iredo | first_ialth | last_jcur | last_ipup | last_pred_reason | last_ipup_reason | last_kflag | last_kcode | last_icf   | last_iret | last_redo | last_iredo | last_ialth"
    );
    println!(
        "-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for row in &rows {
        println!(
            "{:<18} | {:<6} | {:<10} | {:<10} | {:<17} | {:<17} | {:<11} | {:>11} | {:<10} | {:<10} | {:<11} | {:>11} | {:>11} | {:<9} | {:<9} | {:<16} | {:<16} | {:<10} | {:>10} | {:<10} | {:<9} | {:<9} | {:>10} | {}",
            row.path,
            row.matrix,
            fmt_jcur(row.first_jcur),
            fmt_ipup(row.first_ipup),
            fmt_ipup_trigger(row.first_predictor_ipup_trigger),
            fmt_ipup_trigger(row.first_ipup_trigger),
            fmt_kflag(row.first_kflag),
            row.first_kflag_code
                .map_or_else(|| "-".to_string(), |v| v.to_string()),
            fmt_icf(row.first_icf),
            fmt_iret(row.first_iret),
            fmt_redo(row.first_redo),
            row.first_iredo_code
                .map_or_else(|| "-".to_string(), |v| v.to_string()),
            fmt_optional_usize(row.first_ialth),
            fmt_jcur(row.last_jcur),
            fmt_ipup(row.last_ipup),
            fmt_ipup_trigger(row.last_predictor_ipup_trigger),
            fmt_ipup_trigger(row.last_ipup_trigger),
            fmt_kflag(row.last_kflag),
            row.last_kflag_code
                .map_or_else(|| "-".to_string(), |v| v.to_string()),
            fmt_icf(row.last_icf),
            fmt_iret(row.last_iret),
            fmt_redo(row.last_redo),
            row.last_iredo_code
                .map_or_else(|| "-".to_string(), |v| v.to_string()),
            fmt_optional_usize(row.last_ialth),
        );
    }

    println!(
        "[LSODE2 story] native quality dashboard ODEPACK-style aggregate counters over step attempts"
    );
    println!(
        "path               | matrix | predict_attempts | reported_attempts | jcur[cur/stale] | ipup[up/need] | pred_reason[none/rc/msbp/rc+msbp/fail] | final_reason[none/rc/msbp/rc+msbp/fail] | kflag[ok/err/err_rep/conv/conv_rep] | icf[none/refresh/no_recover] | iret[normal/rescale/retry/restart] | redo[none/corr_refresh/corr_retry/err_retry/err_reset/history] | ialth[zero/pos/sum]"
    );
    println!(
        "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for row in &rows {
        let s = &row.summary.native_statistics;
        let reported_attempts = s.native_jcur_current_count + s.native_jcur_stale_count;
        println!(
            "{:<18} | {:<6} | {:>16} | {:>17} | {:>7}/{:<7} | {:>6}/{:<6} | {:>4}/{:>2}/{:>4}/{:>7}/{:>4} | {:>4}/{:>2}/{:>4}/{:>7}/{:>4} | {:>3}/{:>3}/{:>7}/{:>4}/{:>8} | {:>4}/{:>7}/{:>10} | {:>6}/{:>7}/{:>5}/{:>7} | {:>4}/{:>11}/{:>10}/{:>9}/{:>8}/{:>7} | {:>4}/{:>3}/{:>3}",
            row.path,
            row.matrix,
            s.native_step_attempts,
            reported_attempts,
            s.native_jcur_current_count,
            s.native_jcur_stale_count,
            s.native_ipup_up_to_date_count,
            s.native_ipup_needs_update_count,
            s.native_predictor_ipup_trigger_none_count,
            s.native_predictor_ipup_trigger_predictor_rc_ccmax_count,
            s.native_predictor_ipup_trigger_predictor_msbp_count,
            s.native_predictor_ipup_trigger_predictor_rc_ccmax_and_msbp_count,
            s.native_predictor_ipup_trigger_failure_path_count,
            s.native_ipup_trigger_none_count,
            s.native_ipup_trigger_predictor_rc_ccmax_count,
            s.native_ipup_trigger_predictor_msbp_count,
            s.native_ipup_trigger_predictor_rc_ccmax_and_msbp_count,
            s.native_ipup_trigger_failure_path_count,
            s.native_kflag_ok_count,
            s.native_kflag_error_test_failure_count,
            s.native_kflag_repeated_error_test_failure_count,
            s.native_kflag_convergence_failure_count,
            s.native_kflag_repeated_convergence_failure_count,
            s.native_icf_none_count,
            s.native_icf_refresh_requested_count,
            s.native_icf_refresh_did_not_recover_count,
            s.native_iret_normal_flow_count,
            s.native_iret_rescale_history_count,
            s.native_iret_retry_after_error_test_failure_count,
            s.native_iret_restart_with_derivative_refresh_count,
            s.native_redo_none_count,
            s.native_redo_corrector_refresh_same_step_count,
            s.native_redo_corrector_failure_retry_count,
            s.native_redo_error_test_retry_count,
            s.native_redo_repeated_error_reset_count,
            s.native_redo_history_or_step_size_changed_count,
            s.native_ialth_zero_count,
            s.native_ialth_positive_count,
            s.native_ialth_sum,
        );
    }

    for row in &rows {
        assert!(
            row.final_diff.is_finite() && row.rel_final_diff.is_finite(),
            "{} {} should keep quality metrics finite",
            row.path,
            row.matrix
        );
        assert!(
            row.total_ms.is_finite(),
            "{} {} should keep total timing finite",
            row.path,
            row.matrix
        );
        if row.path == "Bridge" {
            assert_eq!(row.summary.status, "finished");
            assert!(
                row.final_diff < 1.0e-6,
                "bridge {} should stay near dense reference: {:e}",
                row.matrix,
                row.final_diff
            );
            assert!(row.accepted_steps.is_none());
            assert!(row.summary.statistics.bdf_nlu_total > 0);
        } else {
            assert!(
                row.summary.status == "finished_native_faithful"
                    || row.summary.status == "finished_native_faithful_partial"
            );
            assert!(row.accepted_steps.unwrap_or(0) > 0);
            assert!(row.total_iterations.unwrap_or(0) > 0);
            assert!(row.summary.native_statistics.native_step_attempts > 0);
            assert!(row.summary.native_statistics.native_residual_calls > 0);
            assert!(row.summary.native_statistics.native_jacobian_calls > 0);
            assert!(row.summary.native_statistics.native_linear_solve_calls > 0);
            let s = &row.summary.native_statistics;
            let reported_attempts = s.native_jcur_current_count + s.native_jcur_stale_count;
            assert!(
                reported_attempts <= s.native_step_attempts,
                "reported DSTODA attempts should not exceed native step attempts"
            );
            assert!(
                reported_attempts
                    <= row.accepted_steps.unwrap_or(0) + row.rejected_steps.unwrap_or(0),
                "reported DSTODA attempts should not exceed accepted+rejected aggregates"
            );
            assert_eq!(
                reported_attempts,
                s.native_ipup_up_to_date_count + s.native_ipup_needs_update_count
            );
            let predictor_reason_samples = s.native_predictor_ipup_trigger_none_count
                + s.native_predictor_ipup_trigger_predictor_rc_ccmax_count
                + s.native_predictor_ipup_trigger_predictor_msbp_count
                + s.native_predictor_ipup_trigger_predictor_rc_ccmax_and_msbp_count
                + s.native_predictor_ipup_trigger_failure_path_count;
            assert!(
                predictor_reason_samples == 0 || predictor_reason_samples == s.native_step_attempts,
                "predictor reason counters should be either unavailable (0) or cover all native step attempts"
            );
            assert_eq!(
                reported_attempts,
                s.native_ipup_trigger_none_count
                    + s.native_ipup_trigger_predictor_rc_ccmax_count
                    + s.native_ipup_trigger_predictor_msbp_count
                    + s.native_ipup_trigger_predictor_rc_ccmax_and_msbp_count
                    + s.native_ipup_trigger_failure_path_count
            );
            assert_eq!(
                reported_attempts,
                s.native_kflag_ok_count
                    + s.native_kflag_error_test_failure_count
                    + s.native_kflag_repeated_error_test_failure_count
                    + s.native_kflag_convergence_failure_count
                    + s.native_kflag_repeated_convergence_failure_count
            );
            assert_eq!(
                reported_attempts,
                s.native_icf_none_count
                    + s.native_icf_refresh_requested_count
                    + s.native_icf_refresh_did_not_recover_count
            );
            assert_eq!(
                reported_attempts,
                s.native_iret_normal_flow_count
                    + s.native_iret_rescale_history_count
                    + s.native_iret_retry_after_error_test_failure_count
                    + s.native_iret_restart_with_derivative_refresh_count
            );
            assert_eq!(
                reported_attempts,
                s.native_redo_none_count
                    + s.native_redo_corrector_refresh_same_step_count
                    + s.native_redo_corrector_failure_retry_count
                    + s.native_redo_error_test_retry_count
                    + s.native_redo_repeated_error_reset_count
                    + s.native_redo_history_or_step_size_changed_count
            );
            assert_eq!(
                reported_attempts,
                s.native_ialth_zero_count + s.native_ialth_positive_count
            );
            assert!(
                row.final_t >= 0.0 && row.final_t <= 1.0,
                "faithful native {} should stay within the integration interval",
                row.matrix
            );
        }
    }
}

#[test]
fn lsode2_native_quality_dashboard_adams_faithful_vs_bridge_baseline() {
    let mut reference_solver = Lsode2Solver::new(exponential_decay_config())
        .expect("reference LSODE2 config should build");
    let reference = reference_solver
        .solve_with_summary()
        .expect("reference LSODE2 solve should finish");
    let reference_final_y = reference
        .final_y
        .as_ref()
        .expect("reference solve should have final y")[0];

    let rows = vec![
        solve_native_quality_row(
            "BridgeBaseline-Bdf",
            "Sparse",
            exponential_decay_config()
                .with_controller(super::algorithm::Lsode2ControllerConfig::bdf_only())
                .with_native_sparse_faer_backend()
                .with_bridge_solve(),
            reference_final_y,
        ),
        solve_native_quality_row(
            "NativeFaithful-Bdf",
            "Sparse",
            exponential_decay_config()
                .with_controller(super::algorithm::Lsode2ControllerConfig::bdf_only())
                .with_native_sparse_faer_backend()
                .with_faithful_bdf_solve(128, 128),
            reference_final_y,
        ),
        solve_native_quality_row(
            "NativeFaithful-Adams",
            "Sparse",
            exponential_decay_config()
                .with_adams_only_controller()
                .with_native_sparse_faer_backend()
                .with_faithful_bdf_solve(128, 128),
            reference_final_y,
        ),
        solve_native_quality_row(
            "BridgeBaseline-Bdf",
            "Banded",
            exponential_decay_config()
                .with_controller(super::algorithm::Lsode2ControllerConfig::bdf_only())
                .with_native_banded_faithful_backend()
                .with_bridge_solve(),
            reference_final_y,
        ),
        solve_native_quality_row(
            "NativeFaithful-Bdf",
            "Banded",
            exponential_decay_config()
                .with_controller(super::algorithm::Lsode2ControllerConfig::bdf_only())
                .with_native_banded_faithful_backend()
                .with_faithful_bdf_solve(128, 128),
            reference_final_y,
        ),
        solve_native_quality_row(
            "NativeFaithful-Adams",
            "Banded",
            exponential_decay_config()
                .with_adams_only_controller()
                .with_native_banded_faithful_backend()
                .with_faithful_bdf_solve(128, 128),
            reference_final_y,
        ),
    ];

    println!(
        "[LSODE2 story] native Adams dashboard: faithful native solve vs bridge baseline; all time columns are milliseconds"
    );
    println!(
        "path                    | matrix | resolved_struct | linear_solver                 | linear_reason                  | controller  | preferred | executed | switch_reason      | status                             | total_ms | final_diff | rel_final_diff | accepted | rejected | native_step_attempts"
    );
    println!(
        "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for row in &rows {
        println!(
            "{:<23} | {:<6} | {:<15} | {:<29} | {:<30} | {:<11} | {:<9} | {:<8} | {:<18} | {:<34} | {:>8.3} | {:>10.3e} | {:>14.3e} | {:>8} | {:>8} | {:>20}",
            row.path,
            row.matrix,
            row.summary.resolved_structure,
            row.summary.linear_solver_backend,
            row.summary.linear_solver_reason,
            row.summary.algorithm.controller_mode,
            row.summary.algorithm.preferred_family,
            row.summary.algorithm.executed_family.unwrap_or("-"),
            row.summary.algorithm.switch_reason,
            row.summary.status,
            row.total_ms,
            row.final_diff,
            row.rel_final_diff,
            fmt_optional_usize(row.accepted_steps),
            fmt_optional_usize(row.rejected_steps),
            row.summary.native_statistics.native_step_attempts,
        );
    }

    for row in &rows {
        assert!(
            row.final_diff.is_finite() && row.rel_final_diff.is_finite(),
            "{} {} should keep quality metrics finite",
            row.path,
            row.matrix
        );

        if row.path.starts_with("BridgeBaseline") {
            assert_eq!(row.summary.status, "finished");
            assert!(
                row.final_diff < 1.0e-6,
                "bridge baseline {} should stay near dense reference: {:e}",
                row.matrix,
                row.final_diff
            );
            assert!(row.summary.native_integration_solve.is_none());
            assert!(row.summary.statistics.bdf_nlu_total > 0);
            continue;
        }

        assert!(
            row.summary.status == "finished_native_faithful"
                || row.summary.status == "finished_native_faithful_partial"
        );
        assert!(row.summary.native_integration_solve.is_some());
        assert!(row.summary.native_statistics.native_step_attempts > 0);
        assert!(row.summary.native_statistics.native_residual_calls > 0);
        assert!(row.summary.native_statistics.native_jacobian_calls > 0);
        assert!(row.summary.native_statistics.native_linear_solve_calls > 0);
        assert!(row.accepted_steps.unwrap_or(0) > 0);

        if row.path.ends_with("Adams") {
            assert_eq!(row.summary.algorithm.controller_mode, "adams_only");
            assert_eq!(row.summary.algorithm.preferred_family, "adams");
            assert_eq!(row.summary.algorithm.switch_reason, "fixed_controller");
        } else {
            assert_eq!(row.summary.algorithm.controller_mode, "bdf_only");
            assert_eq!(row.summary.algorithm.preferred_family, "bdf");
        }
    }

    for matrix in ["Sparse", "Banded"] {
        let native_bdf = rows
            .iter()
            .find(|row| row.path == "NativeFaithful-Bdf" && row.matrix == matrix)
            .expect("native BDF row must exist for each matrix");
        let native_adams = rows
            .iter()
            .find(|row| row.path == "NativeFaithful-Adams" && row.matrix == matrix)
            .expect("native Adams row must exist for each matrix");
        let diff_gap = (native_bdf.final_diff - native_adams.final_diff).abs();
        assert!(
            diff_gap < 1.0e-2,
            "native Adams/BDF quality gap should stay bounded for {}: {:e}",
            matrix,
            diff_gap
        );
    }
}

#[test]
fn lsode2_native_quality_dashboard_adams_faithful_vs_bridge_baseline_multi_run() {
    const RUNS: usize = 5;
    const CASES: [(&str, &str); 6] = [
        ("BridgeBaseline-Bdf", "Sparse"),
        ("NativeFaithful-Bdf", "Sparse"),
        ("NativeFaithful-Adams", "Sparse"),
        ("BridgeBaseline-Bdf", "Banded"),
        ("NativeFaithful-Bdf", "Banded"),
        ("NativeFaithful-Adams", "Banded"),
    ];

    let mut reference_solver = Lsode2Solver::new(exponential_decay_config())
        .expect("reference LSODE2 config should build");
    let reference = reference_solver
        .solve_with_summary()
        .expect("reference LSODE2 solve should finish");
    let reference_final_y = reference
        .final_y
        .as_ref()
        .expect("reference solve should have final y")[0];

    let mut rows = Vec::with_capacity(RUNS * CASES.len());
    for _run in 0..RUNS {
        for (path, matrix) in CASES {
            rows.push(solve_native_quality_row(
                path,
                matrix,
                build_native_adams_dashboard_case_config(path, matrix),
                reference_final_y,
            ));
        }
    }

    let mut aggregated = CASES
        .iter()
        .map(|(path, matrix)| aggregate_native_dashboard_rows(&rows, path, matrix))
        .collect::<Vec<_>>();
    aggregated.sort_by(|lhs, rhs| (lhs.matrix, lhs.path).cmp(&(rhs.matrix, rhs.path)));

    println!(
        "[LSODE2 story] native Adams dashboard multi-run summary; all time columns are milliseconds"
    );
    println!(
        "path                    | matrix | resolved_src | resolved_struct | linear_solver                 | linear_reason                  | ctrl       | pref  | reason            | runs | total_ms mean+/-std [min,max]    | final_t mean+/-std [min,max]        | reached_t_bound | final_diff mean+/-std [min,max]     | rel_final_diff mean+/-std [min,max] | accepted | rejected | attempts | status"
    );
    println!(
        "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for row in &aggregated {
        let status = format!("ok {}/{}", row.ok_runs, row.runs);
        println!(
            "{:<23} | {:<6} | {:<12} | {:<15} | {:<29} | {:<30} | {:<10} | {:<5} | {:<17} | {:>4} | {:<33} | {:<33} | {:<15} | {:<34} | {:<35} | {:>8} | {:>8} | {:>8} | {}",
            row.path,
            row.matrix,
            row.resolved_source,
            row.resolved_structure,
            row.linear_solver_backend,
            row.linear_solver_reason,
            row.controller_mode,
            row.preferred_family,
            row.switch_reason,
            row.runs,
            fmt_agg_ms(row.total_ms),
            fmt_agg_sci(row.final_t),
            fmt_optional_ratio_percent(row.reached_t_bound_ratio),
            fmt_agg_sci(row.final_diff),
            fmt_agg_sci(row.rel_final_diff),
            fmt_optional_agg_count(row.accepted_steps),
            fmt_optional_agg_count(row.rejected_steps),
            fmt_optional_agg_count(row.native_step_attempts),
            status,
        );
    }

    let mut per_matrix_gap = BTreeMap::<&'static str, Vec<f64>>::new();
    for run in 0..RUNS {
        let run_rows = &rows[run * CASES.len()..(run + 1) * CASES.len()];
        for matrix in ["Sparse", "Banded"] {
            let native_bdf = run_rows
                .iter()
                .find(|row| row.path == "NativeFaithful-Bdf" && row.matrix == matrix)
                .expect("native BDF row must exist for each matrix/run");
            let native_adams = run_rows
                .iter()
                .find(|row| row.path == "NativeFaithful-Adams" && row.matrix == matrix)
                .expect("native Adams row must exist for each matrix/run");
            per_matrix_gap
                .entry(matrix)
                .or_default()
                .push((native_bdf.final_diff - native_adams.final_diff).abs());
        }
    }

    for row in &aggregated {
        assert_eq!(
            row.ok_runs, row.runs,
            "{} {} should stay stable across all runs",
            row.path, row.matrix
        );
        assert!(
            row.final_t.mean.is_finite()
                && row.final_diff.mean.is_finite()
                && row.rel_final_diff.mean.is_finite(),
            "{} {} aggregated quality metrics should stay finite",
            row.path,
            row.matrix
        );
        if row.path.starts_with("BridgeBaseline") {
            assert!(
                row.final_diff.max < 1.0e-6,
                "bridge baseline {} should remain near dense reference across runs: max diff={:e}",
                row.matrix,
                row.final_diff.max
            );
        } else if row.path.ends_with("Adams") {
            assert_eq!(row.controller_mode, "adams_only");
            assert_eq!(row.preferred_family, "adams");
            assert_eq!(row.switch_reason, "fixed_controller");
            let reached_ratio = row
                .reached_t_bound_ratio
                .expect("native rows should expose reached_t_bound aggregate ratio");
            assert!(
                (reached_ratio.mean - 1.0).abs() <= f64::EPSILON,
                "native Adams {} should reach t_bound in every run: reached_ratio={:e}",
                row.matrix,
                reached_ratio.mean
            );
            assert!(
                row.final_t.min >= 0.999,
                "native Adams {} should finish at t≈1 in every run: min final_t={:e}",
                row.matrix,
                row.final_t.min
            );
        } else {
            assert_eq!(row.controller_mode, "bdf_only");
            assert_eq!(row.preferred_family, "bdf");
            let reached_ratio = row
                .reached_t_bound_ratio
                .expect("native rows should expose reached_t_bound aggregate ratio");
            assert!(
                (reached_ratio.mean - 1.0).abs() <= f64::EPSILON,
                "native BDF {} should reach t_bound in every run: reached_ratio={:e}",
                row.matrix,
                reached_ratio.mean
            );
            assert!(
                row.final_t.min >= 0.999,
                "native BDF {} should finish at t≈1 in every run: min final_t={:e}",
                row.matrix,
                row.final_t.min
            );
        }
    }

    for (matrix, gaps) in per_matrix_gap {
        let gap = Aggregate::from_values(&gaps);
        assert!(
            gap.max < 1.0e-2,
            "native Adams/BDF per-run quality gap should stay bounded for {}: max gap={:e}",
            matrix,
            gap.max
        );
    }
}

#[test]
fn lsode2_native_quality_dashboard_adams_faithful_deep_limit_probe() {
    const SHALLOW_ATTEMPTS: usize = 128;
    const SHALLOW_ACCEPTED: usize = 128;
    const DEEP_ATTEMPTS: usize = 4096;
    const DEEP_ACCEPTED: usize = 2048;

    let mut reference_solver = Lsode2Solver::new(exponential_decay_config())
        .expect("reference LSODE2 config should build");
    let reference = reference_solver
        .solve_with_summary()
        .expect("reference LSODE2 solve should finish");
    let reference_final_y = reference
        .final_y
        .as_ref()
        .expect("reference solve should have final y")[0];

    let mut rows = Vec::new();
    for matrix in ["Sparse", "Banded"] {
        for path in ["NativeFaithful-Bdf", "NativeFaithful-Adams"] {
            rows.push(solve_native_quality_row(
                "ShallowLimit",
                matrix,
                build_native_adams_dashboard_case_config_with_limits(
                    path,
                    matrix,
                    SHALLOW_ATTEMPTS,
                    SHALLOW_ACCEPTED,
                ),
                reference_final_y,
            ));
            rows.push(solve_native_quality_row(
                "DeepLimit",
                matrix,
                build_native_adams_dashboard_case_config_with_limits(
                    path,
                    matrix,
                    DEEP_ATTEMPTS,
                    DEEP_ACCEPTED,
                ),
                reference_final_y,
            ));
        }
    }

    println!(
        "[LSODE2 story] native Adams deep-limit probe: shallow vs deep faithful-native limits; all time columns are milliseconds"
    );
    println!(
        "limit   | matrix | method_pref | status                             | total_ms | final_t | reached_t_bound | final_diff | rel_final_diff | accepted | rejected | attempts | err_rejects | nonlin_rejects | stale_retry | jac_refresh | diverged | iter_limit"
    );
    println!(
        "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for row in &rows {
        let ns = &row.summary.native_statistics;
        println!(
            "{:<7} | {:<6} | {:<11} | {:<34} | {:>8.3} | {:>7.3e} | {:>15} | {:>10.3e} | {:>14.3e} | {:>8} | {:>8} | {:>8} | {:>11} | {:>14} | {:>11} | {:>11} | {:>8} | {:>10}",
            row.path,
            row.matrix,
            row.summary.algorithm.preferred_family,
            row.summary.status,
            row.total_ms,
            row.final_t,
            row.reached_t_bound
                .map(|value| if value { "yes" } else { "no" })
                .unwrap_or("-"),
            row.final_diff,
            row.rel_final_diff,
            fmt_optional_usize(row.accepted_steps),
            fmt_optional_usize(row.rejected_steps),
            ns.native_step_attempts,
            ns.native_step_rejects_error_test,
            ns.native_step_rejects_nonlinear,
            ns.native_stale_jacobian_retry_count,
            ns.native_jacobian_refresh_requests,
            ns.native_nonlinear_diverged_count,
            ns.native_nonlinear_iteration_limit_count,
        );
    }

    for matrix in ["Sparse", "Banded"] {
        for family in ["bdf", "adams"] {
            let shallow = rows
                .iter()
                .find(|row| {
                    row.path == "ShallowLimit"
                        && row.matrix == matrix
                        && row.summary.algorithm.preferred_family == family
                })
                .expect("shallow probe row should exist");
            let deep = rows
                .iter()
                .find(|row| {
                    row.path == "DeepLimit"
                        && row.matrix == matrix
                        && row.summary.algorithm.preferred_family == family
                })
                .expect("deep probe row should exist");
            assert!(
                deep.final_t + 1.0e-12 >= shallow.final_t,
                "deep-limit final_t should not regress vs shallow for matrix={} family={}: deep={:e}, shallow={:e}",
                matrix,
                family,
                deep.final_t,
                shallow.final_t
            );
            assert!(
                deep.summary.native_statistics.native_step_attempts
                    >= shallow.summary.native_statistics.native_step_attempts,
                "deep-limit attempts should be >= shallow for matrix={} family={}",
                matrix,
                family
            );
        }
    }
}

#[test]
fn lsode2_native_attempt_timeline_story_replay() {
    let config = exponential_decay_config()
        .with_controller(super::algorithm::Lsode2ControllerConfig::bdf_only())
        .with_native_banded_faithful_backend()
        .with_faithful_bdf_solve(96, 24);
    let mut solver = Lsode2Solver::new(config).expect("native timeline config should build");
    let summary = solver
        .solve_with_summary()
        .expect("native timeline solve should finish");
    let native = summary
        .native_integration_solve
        .as_ref()
        .expect("native timeline should expose integration summary");

    println!(
        "[LSODE2 story] native attempt timeline replay (faithful native solve; attempt-by-attempt DSTODA flags)"
    );
    println!(
        "idx | outcome              | accepted | pred_jcur | pred_ipup    | pred_reason | final_jcur | final_ipup   | final_reason | kflag      | kcode | icf        | iret      | redo         | iredo"
    );
    println!(
        "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for (idx, report) in native.attempt_reports.iter().enumerate() {
        println!(
            "{:>3} | {:<20} | {:>8} | {:<9} | {:<12} | {:<11} | {:<10} | {:<12} | {:<12} | {:<10} | {:>5} | {:<10} | {:<9} | {:<12} | {:>5}",
            idx,
            report.outcome_label(),
            if report.accepted() { "yes" } else { "no" },
            fmt_jcur(Some(report.predictor_jcur)),
            fmt_ipup(Some(report.predictor_ipup)),
            fmt_ipup_trigger(Some(report.predictor_ipup_trigger)),
            fmt_jcur(Some(report.jcur)),
            fmt_ipup(Some(report.ipup)),
            fmt_ipup_trigger(Some(report.ipup_trigger)),
            fmt_kflag(Some(report.kflag)),
            report.kflag_code,
            fmt_icf(Some(report.icf)),
            fmt_iret(Some(report.iret)),
            fmt_redo(Some(report.redo_stage)),
            report.iredo.code(),
        );
    }

    assert_eq!(native.attempt_reports.len(), native.attempted_steps);
    assert_eq!(
        native.attempt_reports.first().map(|r| r.kflag),
        Some(native.first_report.kflag)
    );
    assert_eq!(
        native.attempt_reports.last().map(|r| r.kflag),
        Some(native.last_report.kflag)
    );

    for report in &native.attempt_reports {
        assert!(
            matches!(report.kflag_code, 0 | -1 | -2),
            "native attempt kflag code must stay in ODEPACK-like classes {{0,-1,-2}}, got {}",
            report.kflag_code
        );
        if report.icf == Lsode2Icf::RefreshRequested {
            assert_eq!(report.redo_stage, Lsode2RedoStage::CorrectorRefreshSameStep);
        }
        if report.kflag == Lsode2Kflag::RepeatedConvergenceFailure {
            assert_eq!(report.kflag_code, -2);
            assert_eq!(report.icf, Lsode2Icf::RefreshDidNotRecover);
            assert_eq!(report.iret, Lsode2Iret::NormalFlow);
            assert_eq!(report.iredo.code(), 1);
        }
    }
}

#[test]
#[ignore = "parity replay gate; covered by parity_micro/tests in default CI"]
fn lsode2_ipup_reason_control_plane_story_replay() {
    let mut dstoda = Lsode2DstodaState::default();
    let mut stats = Lsode2NativeStatistics::default();
    let mut attempts = 0usize;

    let mut record = |state: &Lsode2DstodaState| {
        attempts += 1;
        stats.record_dstoda_flags(
            state.jacobian_currency(),
            state.ipup(),
            state.ipup_trigger(),
            state.kflag(),
            state.icf(),
            state.iret(),
            state.redo_stage(),
        );
    };

    dstoda.mark_jacobian_current(0);
    record(&dstoda);

    dstoda.set_coefficient_ratio(1.31);
    dstoda.maybe_request_jacobian_update_before_predict(1, Lsode2IterationMode::JacobianBased);
    record(&dstoda);

    dstoda.mark_jacobian_current(1);
    dstoda.maybe_request_jacobian_update_before_predict(21, Lsode2IterationMode::JacobianBased);
    record(&dstoda);

    dstoda.mark_jacobian_current(0);
    dstoda.set_coefficient_ratio(1.5);
    dstoda.maybe_request_jacobian_update_before_predict(20, Lsode2IterationMode::JacobianBased);
    record(&dstoda);

    dstoda.record_history_or_step_size_change();
    record(&dstoda);

    println!("[LSODE2 story] IPUP reason control-plane replay (deterministic); attempt counters");
    println!(
        "attempts={} ipup_reason[none/rc/msbp/rc+msbp/fail]={}/{}/{}/{}/{}",
        attempts,
        stats.native_ipup_trigger_none_count,
        stats.native_ipup_trigger_predictor_rc_ccmax_count,
        stats.native_ipup_trigger_predictor_msbp_count,
        stats.native_ipup_trigger_predictor_rc_ccmax_and_msbp_count,
        stats.native_ipup_trigger_failure_path_count
    );

    assert_eq!(attempts, 5);
    assert_eq!(stats.native_ipup_trigger_none_count, 1);
    assert_eq!(stats.native_ipup_trigger_predictor_rc_ccmax_count, 1);
    assert_eq!(stats.native_ipup_trigger_predictor_msbp_count, 1);
    assert_eq!(
        stats.native_ipup_trigger_predictor_rc_ccmax_and_msbp_count,
        1
    );
    assert_eq!(stats.native_ipup_trigger_failure_path_count, 1);
}

#[test]
#[ignore = "parity replay gate; covered by parity_micro/tests in default CI"]
fn lsode2_dstoda_flag_control_plane_story_replay() {
    let mut dstoda = Lsode2DstodaState::default();
    let mut stats = Lsode2NativeStatistics::default();
    let mut attempts = 0usize;

    let mut record = |state: &Lsode2DstodaState| {
        attempts += 1;
        stats.record_dstoda_flags(
            state.jacobian_currency(),
            state.ipup(),
            state.ipup_trigger(),
            state.kflag(),
            state.icf(),
            state.iret(),
            state.redo_stage(),
        );
    };

    dstoda.mark_jacobian_current(0);
    record(&dstoda);

    dstoda.record_error_test_failure();
    record(&dstoda);

    dstoda.record_repeated_error_test_reset();
    record(&dstoda);

    dstoda.mark_jacobian_stale();
    let _ = dstoda.decide_after_corrector_failure(Lsode2IterationMode::JacobianBased);
    record(&dstoda);

    dstoda.mark_jacobian_current(0);
    let _ = dstoda.decide_after_corrector_failure(Lsode2IterationMode::JacobianBased);
    record(&dstoda);

    dstoda.record_repeated_convergence_failure();
    record(&dstoda);

    dstoda.record_repeated_error_test_failure();
    record(&dstoda);

    dstoda.record_step_accepted();
    record(&dstoda);

    println!("[LSODE2 story] DSTODA flag control-plane replay (deterministic); attempt counters");
    println!(
        "attempts={} kflag[ok/err/err_rep/conv/conv_rep]={}/{}/{}/{}/{} icf[none/refresh/no_recover]={}/{}/{} iret[normal/rescale/retry/restart]={}/{}/{}/{} redo[none/corr_refresh/corr_retry/err_retry/err_reset/history]={}/{}/{}/{}/{}/{}",
        attempts,
        stats.native_kflag_ok_count,
        stats.native_kflag_error_test_failure_count,
        stats.native_kflag_repeated_error_test_failure_count,
        stats.native_kflag_convergence_failure_count,
        stats.native_kflag_repeated_convergence_failure_count,
        stats.native_icf_none_count,
        stats.native_icf_refresh_requested_count,
        stats.native_icf_refresh_did_not_recover_count,
        stats.native_iret_normal_flow_count,
        stats.native_iret_rescale_history_count,
        stats.native_iret_retry_after_error_test_failure_count,
        stats.native_iret_restart_with_derivative_refresh_count,
        stats.native_redo_none_count,
        stats.native_redo_corrector_refresh_same_step_count,
        stats.native_redo_corrector_failure_retry_count,
        stats.native_redo_error_test_retry_count,
        stats.native_redo_repeated_error_reset_count,
        stats.native_redo_history_or_step_size_changed_count
    );

    assert_eq!(attempts, 8);

    assert_eq!(stats.native_kflag_ok_count, 2);
    assert_eq!(stats.native_kflag_error_test_failure_count, 2);
    assert_eq!(stats.native_kflag_repeated_error_test_failure_count, 1);
    assert_eq!(stats.native_kflag_convergence_failure_count, 2);
    assert_eq!(stats.native_kflag_repeated_convergence_failure_count, 1);

    assert_eq!(stats.native_icf_none_count, 5);
    assert_eq!(stats.native_icf_refresh_requested_count, 1);
    assert_eq!(stats.native_icf_refresh_did_not_recover_count, 2);

    assert_eq!(stats.native_iret_normal_flow_count, 6);
    assert_eq!(stats.native_iret_rescale_history_count, 0);
    assert_eq!(stats.native_iret_retry_after_error_test_failure_count, 1);
    assert_eq!(stats.native_iret_restart_with_derivative_refresh_count, 1);

    assert_eq!(stats.native_redo_none_count, 3);
    assert_eq!(stats.native_redo_corrector_refresh_same_step_count, 1);
    // Terminal repeated error-test failure now clears sticky REDO marker.
    assert_eq!(stats.native_redo_corrector_failure_retry_count, 2);
    assert_eq!(stats.native_redo_error_test_retry_count, 1);
    assert_eq!(stats.native_redo_repeated_error_reset_count, 1);
    assert_eq!(stats.native_redo_history_or_step_size_changed_count, 0);
}

#[test]
#[ignore = "parity replay gate; covered by parity_micro/tests in default CI"]
fn lsode2_dstoda_terminal_convergence_reason_story_replay() {
    fn make_cycle(config: super::Lsode2StepControlConfig) -> super::Lsode2StepCycle {
        let state = super::Lsode2RuntimeState::new(0.0, &[1.0], 0.1, 2, config)
            .expect("runtime state should initialize for DSTODA terminal reason story");
        let error_control = super::Lsode2ErrorController::new(
            super::Lsode2Tolerance::scalar(1.0e-3, 1.0e-6),
            super::Lsode2ErrorControlConfig::default(),
        )
        .expect("error control should initialize for DSTODA terminal reason story");
        super::Lsode2StepCycle::new(state, error_control)
    }

    struct TerminalRow {
        reason: &'static str,
        action: super::Lsode2RetryAction,
        kflag: super::Lsode2Kflag,
        kflag_code: i32,
        icf: super::Lsode2Icf,
        iret: super::Lsode2Iret,
        redo: super::Lsode2RedoStage,
        iredo_code: i32,
    }

    let mut rows = Vec::new();
    let mut stats = super::Lsode2NativeStatistics::default();

    // Path A: terminal because repeated convergence failures reached MXNCF-like cap.
    let mut cycle_mxncf = make_cycle(super::Lsode2StepControlConfig {
        max_convergence_failures: 2,
        h_min: 1.0e-14,
        ..super::Lsode2StepControlConfig::default()
    });
    // Mirror DSTODA convergence choreography used by the step-cycle parity tests:
    // 1) ICF=1 stale-J one-shot same-step retry (no NCF counting),
    // 2) post-refresh failure transitions to retract path (ICF=2, NCF starts),
    // 3) next retract-path failure reaches MXNCF terminal branch.
    let first = cycle_mxncf
        .retry_after_stale_jacobian_nonlinear_failure()
        .expect("first stale-J nonlinear failure should produce same-step refresh retry");
    assert_eq!(
        first.action,
        super::Lsode2RetryAction::RetryWithJacobianRefresh
    );
    assert_eq!(cycle_mxncf.icf(), super::Lsode2Icf::RefreshRequested);

    cycle_mxncf.mark_jacobian_stale();
    let second = cycle_mxncf
        .retry_after_stale_jacobian_nonlinear_failure()
        .expect("second stale-J nonlinear failure should transition to retract path");
    assert_eq!(
        second.action,
        super::Lsode2RetryAction::RetryWithJacobianRefresh
    );
    assert_eq!(cycle_mxncf.icf(), super::Lsode2Icf::RefreshDidNotRecover);

    cycle_mxncf.mark_jacobian_current();
    let third = cycle_mxncf
        .reject_after_nonlinear_failure()
        .expect("third nonlinear failure should produce terminal MXNCF-like retry");
    rows.push(TerminalRow {
        reason: "mxncf",
        action: third.action,
        kflag: cycle_mxncf.kflag(),
        kflag_code: cycle_mxncf.kflag_code(),
        icf: cycle_mxncf.icf(),
        iret: cycle_mxncf.iret(),
        redo: cycle_mxncf.redo_stage(),
        iredo_code: cycle_mxncf.iredo().code(),
    });
    stats.record_dstoda_flags(
        cycle_mxncf.jacobian_currency(),
        cycle_mxncf.ipup(),
        cycle_mxncf.ipup_trigger(),
        cycle_mxncf.kflag(),
        cycle_mxncf.icf(),
        cycle_mxncf.iret(),
        cycle_mxncf.redo_stage(),
    );

    // Path B: terminal because step-size shrink would cross HMIN.
    let mut cycle_hmin = make_cycle(super::Lsode2StepControlConfig {
        h_min: 0.09,
        max_convergence_failures: 10,
        ..super::Lsode2StepControlConfig::default()
    });
    cycle_hmin.mark_jacobian_current();
    let terminal = cycle_hmin
        .reject_after_nonlinear_failure()
        .expect("HMIN underflow branch should terminate on first nonlinear failure");
    rows.push(TerminalRow {
        reason: "hmin_underflow",
        action: terminal.action,
        kflag: cycle_hmin.kflag(),
        kflag_code: cycle_hmin.kflag_code(),
        icf: cycle_hmin.icf(),
        iret: cycle_hmin.iret(),
        redo: cycle_hmin.redo_stage(),
        iredo_code: cycle_hmin.iredo().code(),
    });
    stats.record_dstoda_flags(
        cycle_hmin.jacobian_currency(),
        cycle_hmin.ipup(),
        cycle_hmin.ipup_trigger(),
        cycle_hmin.kflag(),
        cycle_hmin.icf(),
        cycle_hmin.iret(),
        cycle_hmin.redo_stage(),
    );

    println!("[LSODE2 story] DSTODA terminal convergence reason replay");
    println!(
        "reason           | retry_action                      | kflag                  | kflag_code | icf                   | iret          | redo_stage                | iredo_code"
    );
    println!(
        "----------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    for row in &rows {
        println!(
            "{:<16} | {:<33?} | {:<22?} | {:>10} | {:<21?} | {:<13?} | {:<25?} | {:>10}",
            row.reason,
            row.action,
            row.kflag,
            row.kflag_code,
            row.icf,
            row.iret,
            row.redo,
            row.iredo_code
        );
    }

    assert_eq!(rows.len(), 2);
    assert_eq!(
        rows[0].action,
        super::Lsode2RetryAction::FailRepeatedConvergenceFailures
    );
    assert_eq!(
        rows[1].action,
        super::Lsode2RetryAction::FailStepSizeUnderflow
    );

    // DSTODA parity: both terminal reasons are KFLAG=-2 class.
    assert_eq!(
        rows[0].kflag,
        super::Lsode2Kflag::RepeatedConvergenceFailure
    );
    assert_eq!(
        rows[1].kflag,
        super::Lsode2Kflag::RepeatedConvergenceFailure
    );
    assert_eq!(rows[0].kflag_code, -2);
    assert_eq!(rows[1].kflag_code, -2);

    assert_eq!(rows[0].icf, super::Lsode2Icf::RefreshDidNotRecover);
    assert_eq!(rows[1].icf, super::Lsode2Icf::RefreshDidNotRecover);
    assert_eq!(rows[0].iret, super::Lsode2Iret::NormalFlow);
    assert_eq!(rows[1].iret, super::Lsode2Iret::NormalFlow);
    assert_eq!(rows[0].iredo_code, 1);
    assert_eq!(rows[1].iredo_code, 1);

    assert_eq!(stats.native_kflag_repeated_convergence_failure_count, 2);
    assert_eq!(stats.native_icf_refresh_did_not_recover_count, 2);
    assert_eq!(stats.native_iret_normal_flow_count, 2);
    assert_eq!(stats.native_redo_corrector_failure_retry_count, 2);
}
