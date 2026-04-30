//! Small solver-owned native preflight loop for LSODE2.
//!
//! This layer is intentionally modest: it does not replace the current
//! bridge-to-BDF solve, but it already runs a short real native stepping
//! fragment using native residual/Jacobian/linear-solve machinery.

use super::algorithm::Lsode2SwitchTelemetry;
use super::config::Lsode2ProblemConfig;
use super::native_integration::{Lsode2NativeIntegrationLimits, run_native_integration};
use super::statistics::Lsode2NativeStatistics;
use crate::symbolic::symbolic_ivp::IvpBackendError;

#[derive(Debug, Clone)]
pub struct Lsode2NativeStepProbeSummary {
    pub outcome: &'static str,
    pub accepted: bool,
    pub iterations: usize,
    pub attempted_steps: usize,
    pub accepted_steps: usize,
    pub rejected_steps: usize,
    pub t_trial: f64,
    pub h_trial: f64,
    pub final_t: f64,
    pub final_h: f64,
    pub telemetry: Lsode2SwitchTelemetry,
}

#[derive(Debug, Clone)]
pub struct Lsode2NativePreflightOutcome {
    pub summary: Option<Lsode2NativeStepProbeSummary>,
    pub statistics: Lsode2NativeStatistics,
}

pub fn run_native_step_preflight(
    config: &Lsode2ProblemConfig,
) -> Result<Lsode2NativePreflightOutcome, IvpBackendError> {
    let outcome = run_native_integration(
        config,
        Lsode2NativeIntegrationLimits::new(
            max_native_preflight_step_attempts(),
            max_native_preflight_accepted_steps(),
        ),
    )?;

    Ok(Lsode2NativePreflightOutcome {
        summary: outcome.summary.map(|summary| Lsode2NativeStepProbeSummary {
            outcome: summary.last_report.outcome_label(),
            accepted: summary.last_report.accepted(),
            iterations: summary.total_iterations,
            attempted_steps: summary.attempted_steps,
            accepted_steps: summary.accepted_steps,
            rejected_steps: summary.rejected_steps,
            t_trial: summary.first_report.predicted.t_trial,
            h_trial: summary.first_report.predicted.h_trial,
            final_t: summary.final_t,
            final_h: summary.final_h,
            telemetry: summary.last_report.telemetry,
        }),
        statistics: outcome.statistics,
    })
}

fn max_native_preflight_step_attempts() -> usize {
    4
}

fn max_native_preflight_accepted_steps() -> usize {
    2
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
    fn native_preflight_skips_dense_backend() {
        let outcome = run_native_step_preflight(&exponential_decay_config()).unwrap();
        assert!(outcome.summary.is_none());
        assert_eq!(outcome.statistics.native_step_attempts, 0);
    }

    #[test]
    fn native_preflight_records_sparse_probe_statistics() {
        let outcome = run_native_step_preflight(
            &exponential_decay_config().with_native_sparse_faer_backend(),
        )
        .unwrap();
        let summary = outcome
            .summary
            .as_ref()
            .expect("sparse native preflight should produce a summary");
        assert!(summary.iterations > 0);
        assert!(summary.attempted_steps > 0);
        assert_eq!(
            summary.attempted_steps,
            summary.accepted_steps + summary.rejected_steps
        );
        assert!(outcome.statistics.native_residual_calls > 0);
        assert!(outcome.statistics.native_jacobian_calls > 0);
        assert!(outcome.statistics.native_linear_solve_calls > 0);
    }
}
