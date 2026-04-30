//! Small multi-step native integration runner for LSODE2.
//!
//! This layer keeps `native_step_engine` focused on one honest step attempt and
//! collects a short sequence of such attempts into one reusable report:
//! - exploratory preflight,
//! - future native warm-up,
//! - and eventually a solver-owned native path inside `solve()`.

use super::config::Lsode2ProblemConfig;
use super::native_step_engine::{
    Lsode2NativeStepAttemptReport, Lsode2NativeStepEngine, Lsode2NativeStepMethod,
};
use super::statistics::Lsode2NativeStatistics;
use crate::symbolic::symbolic_ivp::IvpBackendError;

#[derive(Debug, Clone, Copy)]
pub struct Lsode2NativeIntegrationLimits {
    pub max_step_attempts: usize,
    pub max_accepted_steps: usize,
}

impl Lsode2NativeIntegrationLimits {
    pub fn new(max_step_attempts: usize, max_accepted_steps: usize) -> Self {
        Self {
            max_step_attempts: max_step_attempts.max(1),
            max_accepted_steps: max_accepted_steps.max(1),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Lsode2NativeIntegrationSummary {
    pub first_report: Lsode2NativeStepAttemptReport,
    pub last_report: Lsode2NativeStepAttemptReport,
    pub attempt_reports: Vec<Lsode2NativeStepAttemptReport>,
    pub attempted_steps: usize,
    pub accepted_steps: usize,
    pub rejected_steps: usize,
    pub total_iterations: usize,
    pub reached_t_bound: bool,
    pub final_t: f64,
    pub final_h: f64,
    pub final_y: Vec<f64>,
    pub accepted_t_history: Vec<f64>,
    pub accepted_y_history: Vec<Vec<f64>>,
}

#[derive(Debug, Clone)]
pub struct Lsode2NativeIntegrationOutcome {
    pub summary: Option<Lsode2NativeIntegrationSummary>,
    pub statistics: Lsode2NativeStatistics,
}

pub fn run_native_integration(
    config: &Lsode2ProblemConfig,
    limits: Lsode2NativeIntegrationLimits,
) -> Result<Lsode2NativeIntegrationOutcome, IvpBackendError> {
    run_native_integration_for_method(config, limits, Lsode2NativeStepMethod::BdfLike)
}

pub fn run_native_integration_for_method(
    config: &Lsode2ProblemConfig,
    limits: Lsode2NativeIntegrationLimits,
    method: Lsode2NativeStepMethod,
) -> Result<Lsode2NativeIntegrationOutcome, IvpBackendError> {
    let mut engine = match Lsode2NativeStepEngine::from_problem_config_with_method(config, method)?
    {
        Some(engine) => engine,
        None => {
            return Ok(Lsode2NativeIntegrationOutcome {
                summary: None,
                statistics: Lsode2NativeStatistics::default(),
            });
        }
    };

    let mut first_report: Option<Lsode2NativeStepAttemptReport> = None;
    let mut last_report: Option<Lsode2NativeStepAttemptReport> = None;
    let mut attempt_reports: Vec<Lsode2NativeStepAttemptReport> = Vec::new();
    let mut attempted_steps = 0usize;
    let mut accepted_steps = 0usize;
    let mut rejected_steps = 0usize;
    let mut total_iterations = 0usize;
    let initial_state = engine.state_snapshot();
    let mut accepted_t_history = vec![initial_state.t];
    let mut accepted_y_history = vec![engine.current_solution()];

    while attempted_steps < limits.max_step_attempts
        && accepted_steps < limits.max_accepted_steps
        && !reached_t_bound(
            engine.state_snapshot().t,
            config.t_bound,
            engine.state_snapshot().h,
        )
    {
        engine.clamp_step_to_t_bound(config.t_bound)?;
        let report = engine.step_once()?;
        if first_report.is_none() {
            first_report = Some(report.clone());
        }
        attempt_reports.push(report.clone());
        attempted_steps += 1;
        total_iterations += report.iterations;
        rejected_steps += report.retry_count;
        if report.accepted() {
            accepted_steps += 1;
            let state = engine.state_snapshot();
            accepted_t_history.push(state.t);
            accepted_y_history.push(engine.current_solution());
        } else {
            rejected_steps += 1;
        }
        last_report = Some(report);
    }

    let summary = match (first_report, last_report) {
        (Some(first_report), Some(last_report)) => {
            let final_state = engine.state_snapshot();
            Some(Lsode2NativeIntegrationSummary {
                first_report,
                last_report,
                attempt_reports,
                attempted_steps,
                accepted_steps,
                rejected_steps,
                total_iterations,
                reached_t_bound: reached_t_bound(final_state.t, config.t_bound, final_state.h),
                final_t: final_state.t,
                final_h: final_state.h,
                final_y: engine.current_solution(),
                accepted_t_history,
                accepted_y_history,
            })
        }
        (None, None) => None,
        _ => {
            return Err(IvpBackendError::GeneratedBackendFailure {
                message: "native integration loop ended in an inconsistent report state"
                    .to_string(),
            });
        }
    };

    Ok(Lsode2NativeIntegrationOutcome {
        summary,
        statistics: engine.statistics().clone(),
    })
}

fn reached_t_bound(t: f64, t_bound: f64, h: f64) -> bool {
    if h >= 0.0 { t >= t_bound } else { t <= t_bound }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::LSODE2::Lsode2ProblemConfig;
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
    fn native_integration_skips_dense_backend() {
        let outcome = run_native_integration(
            &exponential_decay_config(),
            Lsode2NativeIntegrationLimits::new(4, 2),
        )
        .expect("dense path should not error");
        assert!(outcome.summary.is_none());
        assert_eq!(outcome.statistics.native_step_attempts, 0);
    }

    #[test]
    fn native_integration_sparse_runner_collects_multi_step_summary() {
        let outcome = run_native_integration(
            &exponential_decay_config().with_native_sparse_faer_backend(),
            Lsode2NativeIntegrationLimits::new(4, 2),
        )
        .expect("sparse path should run native integration");
        let summary = outcome
            .summary
            .expect("native sparse path should produce integration summary");

        assert!(summary.attempted_steps > 0);
        assert_eq!(summary.attempt_reports.len(), summary.attempted_steps);
        assert!(summary.accepted_steps > 0);
        assert!(summary.rejected_steps <= summary.total_iterations);
        assert!(summary.total_iterations > 0);
        assert!(summary.final_t >= 0.0);
        assert_eq!(summary.final_y.len(), 1);
        assert!(outcome.statistics.native_residual_calls > 0);
        assert!(outcome.statistics.native_jacobian_calls > 0);
        assert!(outcome.statistics.native_linear_solve_calls > 0);
    }

    #[test]
    fn native_integration_adams_like_mode_respects_controller_adams_order_cap() {
        let config = exponential_decay_config()
            .with_controller(
                super::super::algorithm::Lsode2ControllerConfig::bdf_only().with_max_adams_order(4),
            )
            .with_native_sparse_faer_backend();
        let outcome = run_native_integration_for_method(
            &config,
            Lsode2NativeIntegrationLimits::new(5, 3),
            Lsode2NativeStepMethod::AdamsLike,
        )
        .expect("adams-like preview should run on sparse native path");

        let summary = outcome
            .summary
            .expect("adams-like sparse path should produce integration summary");
        assert_eq!(summary.first_report.predicted.order, 1);
        assert_eq!(summary.attempt_reports.len(), summary.attempted_steps);
        assert!(summary.accepted_steps > 0);
        assert!(summary.final_t > 0.0);
        assert!(summary.last_report.predicted.order <= 4);
    }
}
