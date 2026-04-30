//! Native LSODE2 statistics collector.
//!
//! The bridge-to-BDF phase already exposes useful counters through
//! `IvpBackendStatistics`, but LSODE2 needs its own long-lived telemetry model:
//! method-family choices, native step outcomes, nonlinear retries, explicit
//! Jacobian refresh requests, and stage timings that will continue to make
//! sense after the solver stops delegating its main loop to the legacy BDF
//! engine.

use super::algorithm::{Lsode2MethodFamily, Lsode2SwitchDecision};
use super::correction::{Lsode2CorrectionAssessment, Lsode2CorrectionStatus};
use super::dstoda_state::{
    Lsode2Icf, Lsode2Ipup, Lsode2IpupTrigger, Lsode2Iret, Lsode2JacobianCurrency, Lsode2Kflag,
    Lsode2RedoStage,
};
use super::step_control::Lsode2StepFailure;
use super::step_cycle::Lsode2StepCycleOutcome;
use crate::symbolic::symbolic_ivp_generated::IvpBackendStatistics;
use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Lsode2NativeStatistics {
    pub backend_prepare_calls: usize,
    pub backend_prepare_ms_total: f64,
    pub solve_calls: usize,
    pub solve_ms_total: f64,
    pub bridge_prepare_calls: usize,
    pub bridge_prepare_ms_total: f64,
    pub bridge_solve_calls: usize,
    pub bridge_solve_ms_total: f64,
    pub bridge_residual_calls: usize,
    pub bridge_residual_ms_total: f64,
    pub bridge_jacobian_calls: usize,
    pub bridge_jacobian_ms_total: f64,
    pub bridge_step_calls: usize,
    pub bridge_accepted_steps: usize,
    pub bridge_nonlinear_solve_calls: usize,
    pub bridge_nonlinear_iterations_total: usize,
    pub bridge_bdf_nfev_total: usize,
    pub bridge_bdf_njev_total: usize,
    pub bridge_bdf_nlu_total: usize,
    pub algorithm_decision_calls: usize,
    pub preferred_adams_count: usize,
    pub preferred_bdf_count: usize,
    pub executed_adams_count: usize,
    pub executed_bdf_count: usize,
    pub fallback_decision_count: usize,
    pub native_adams_cost_samples: usize,
    pub native_bdf_cost_samples: usize,
    pub native_adams_step_cost_accum: f64,
    pub native_bdf_step_cost_accum: f64,
    pub native_step_attempts: usize,
    pub native_step_accepts: usize,
    pub native_step_rejects_error_test: usize,
    pub native_step_rejects_nonlinear: usize,
    pub native_nonlinear_continue_count: usize,
    pub native_nonlinear_converged_count: usize,
    pub native_nonlinear_diverged_count: usize,
    pub native_nonlinear_iteration_limit_count: usize,
    pub native_stale_jacobian_retry_count: usize,
    pub native_jacobian_refresh_requests: usize,
    pub native_linear_solve_calls: usize,
    pub native_linear_solve_ms_total: f64,
    pub native_residual_calls: usize,
    pub native_residual_ms_total: f64,
    pub native_jacobian_calls: usize,
    pub native_jacobian_ms_total: f64,
    pub native_jcur_current_count: usize,
    pub native_jcur_stale_count: usize,
    pub native_ipup_up_to_date_count: usize,
    pub native_ipup_needs_update_count: usize,
    pub native_predictor_ipup_trigger_none_count: usize,
    pub native_predictor_ipup_trigger_predictor_rc_ccmax_count: usize,
    pub native_predictor_ipup_trigger_predictor_msbp_count: usize,
    pub native_predictor_ipup_trigger_predictor_rc_ccmax_and_msbp_count: usize,
    pub native_predictor_ipup_trigger_failure_path_count: usize,
    pub native_ipup_trigger_none_count: usize,
    pub native_ipup_trigger_predictor_rc_ccmax_count: usize,
    pub native_ipup_trigger_predictor_msbp_count: usize,
    pub native_ipup_trigger_predictor_rc_ccmax_and_msbp_count: usize,
    pub native_ipup_trigger_failure_path_count: usize,
    pub native_kflag_ok_count: usize,
    pub native_kflag_error_test_failure_count: usize,
    pub native_kflag_repeated_error_test_failure_count: usize,
    pub native_kflag_convergence_failure_count: usize,
    pub native_kflag_repeated_convergence_failure_count: usize,
    pub native_icf_none_count: usize,
    pub native_icf_refresh_requested_count: usize,
    pub native_icf_refresh_did_not_recover_count: usize,
    pub native_iret_normal_flow_count: usize,
    pub native_iret_rescale_history_count: usize,
    pub native_iret_retry_after_error_test_failure_count: usize,
    pub native_iret_restart_with_derivative_refresh_count: usize,
    pub native_redo_none_count: usize,
    pub native_redo_corrector_refresh_same_step_count: usize,
    pub native_redo_corrector_failure_retry_count: usize,
    pub native_redo_error_test_retry_count: usize,
    pub native_redo_repeated_error_reset_count: usize,
    pub native_redo_history_or_step_size_changed_count: usize,
    pub native_ialth_zero_count: usize,
    pub native_ialth_positive_count: usize,
    pub native_ialth_sum: usize,
}

impl Lsode2NativeStatistics {
    pub fn record_backend_prepare_duration(&mut self, duration: Duration) {
        self.backend_prepare_calls += 1;
        self.backend_prepare_ms_total += duration.as_secs_f64() * 1_000.0;
    }

    pub fn record_solve_duration(&mut self, duration: Duration) {
        self.solve_calls += 1;
        self.solve_ms_total += duration.as_secs_f64() * 1_000.0;
    }

    pub fn record_algorithm_decision(&mut self, decision: &Lsode2SwitchDecision) {
        self.algorithm_decision_calls += 1;
        match decision.preferred_family {
            Lsode2MethodFamily::Adams => self.preferred_adams_count += 1,
            Lsode2MethodFamily::Bdf => self.preferred_bdf_count += 1,
        }
        match decision.executed_family() {
            Some(Lsode2MethodFamily::Adams) => self.executed_adams_count += 1,
            Some(Lsode2MethodFamily::Bdf) => self.executed_bdf_count += 1,
            None => {}
        }
        if decision.uses_fallback {
            self.fallback_decision_count += 1;
        }
    }

    pub fn record_native_method_cost_sample(&mut self, family: Lsode2MethodFamily, cost: f64) {
        if !cost.is_finite() || cost <= 0.0 {
            return;
        }
        match family {
            Lsode2MethodFamily::Adams => {
                self.native_adams_cost_samples += 1;
                self.native_adams_step_cost_accum += cost;
            }
            Lsode2MethodFamily::Bdf => {
                self.native_bdf_cost_samples += 1;
                self.native_bdf_step_cost_accum += cost;
            }
        }
    }

    pub fn adams_step_cost_estimate(&self) -> Option<f64> {
        if self.native_adams_cost_samples == 0 {
            None
        } else {
            Some(self.native_adams_step_cost_accum / self.native_adams_cost_samples as f64)
        }
    }

    pub fn bdf_step_cost_estimate(&self) -> Option<f64> {
        if self.native_bdf_cost_samples == 0 {
            None
        } else {
            Some(self.native_bdf_step_cost_accum / self.native_bdf_cost_samples as f64)
        }
    }

    pub fn sync_from_bridge(&mut self, bridge: &IvpBackendStatistics) {
        self.bridge_prepare_calls = bridge.backend_prepare_calls;
        self.bridge_prepare_ms_total = bridge.backend_prepare_ms_total;
        self.bridge_solve_calls = bridge.solve_calls;
        self.bridge_solve_ms_total = bridge.solve_ms_total;
        self.bridge_step_calls = bridge.step_calls;
        self.bridge_nonlinear_solve_calls = bridge.nonlinear_solve_calls;
        self.bridge_nonlinear_iterations_total = bridge.nonlinear_iterations_total;
        self.bridge_residual_calls = bridge.residual_calls;
        self.bridge_residual_ms_total = bridge.residual_ms_total;
        self.bridge_jacobian_calls = bridge.jacobian_calls;
        self.bridge_jacobian_ms_total = bridge.jacobian_ms_total;
        self.bridge_bdf_nfev_total = bridge.bdf_nfev_total;
        self.bridge_bdf_njev_total = bridge.bdf_njev_total;
        self.bridge_bdf_nlu_total = bridge.bdf_nlu_total;
    }

    pub fn record_native_step_attempt(&mut self) {
        self.native_step_attempts += 1;
    }

    pub fn record_prediction(&mut self) {
        self.record_native_step_attempt();
    }

    pub fn record_native_step_accept(&mut self) {
        self.native_step_accepts += 1;
    }

    pub fn record_native_step_reject_error_test(&mut self) {
        self.native_step_rejects_error_test += 1;
    }

    pub fn record_native_step_reject_nonlinear(&mut self) {
        self.native_step_rejects_nonlinear += 1;
    }

    pub fn record_native_nonlinear_continue(&mut self) {
        self.native_nonlinear_continue_count += 1;
    }

    pub fn record_native_nonlinear_converged(&mut self) {
        self.native_nonlinear_converged_count += 1;
    }

    pub fn record_native_nonlinear_diverged(&mut self) {
        self.native_nonlinear_diverged_count += 1;
    }

    pub fn record_native_nonlinear_iteration_limit(&mut self) {
        self.native_nonlinear_iteration_limit_count += 1;
    }

    pub fn record_native_stale_jacobian_retry(&mut self) {
        self.native_stale_jacobian_retry_count += 1;
    }

    pub fn record_jacobian_refresh_request(&mut self) {
        self.native_jacobian_refresh_requests += 1;
    }

    pub fn record_native_linear_solve_duration(&mut self, duration: Duration) {
        self.native_linear_solve_calls += 1;
        self.native_linear_solve_ms_total += duration.as_secs_f64() * 1_000.0;
    }

    pub fn record_native_residual_duration(&mut self, duration: Duration) {
        self.native_residual_calls += 1;
        self.native_residual_ms_total += duration.as_secs_f64() * 1_000.0;
    }

    pub fn record_native_jacobian_duration(&mut self, duration: Duration) {
        self.native_jacobian_calls += 1;
        self.native_jacobian_ms_total += duration.as_secs_f64() * 1_000.0;
    }

    pub fn record_dstoda_flags(
        &mut self,
        jcur: Lsode2JacobianCurrency,
        ipup: Lsode2Ipup,
        ipup_trigger: Lsode2IpupTrigger,
        kflag: Lsode2Kflag,
        icf: Lsode2Icf,
        iret: Lsode2Iret,
        redo_stage: Lsode2RedoStage,
    ) {
        match jcur {
            Lsode2JacobianCurrency::Current => self.native_jcur_current_count += 1,
            Lsode2JacobianCurrency::Stale => self.native_jcur_stale_count += 1,
        }
        match ipup {
            Lsode2Ipup::UpToDate => self.native_ipup_up_to_date_count += 1,
            Lsode2Ipup::NeedsJacobianUpdate => self.native_ipup_needs_update_count += 1,
        }
        match ipup_trigger {
            Lsode2IpupTrigger::None => self.native_ipup_trigger_none_count += 1,
            Lsode2IpupTrigger::PredictorRcCcmax => {
                self.native_ipup_trigger_predictor_rc_ccmax_count += 1
            }
            Lsode2IpupTrigger::PredictorMsbp => self.native_ipup_trigger_predictor_msbp_count += 1,
            Lsode2IpupTrigger::PredictorRcCcmaxAndMsbp => {
                self.native_ipup_trigger_predictor_rc_ccmax_and_msbp_count += 1
            }
            Lsode2IpupTrigger::FailurePath => self.native_ipup_trigger_failure_path_count += 1,
        }
        match kflag {
            Lsode2Kflag::Ok => self.native_kflag_ok_count += 1,
            Lsode2Kflag::ErrorTestFailure => self.native_kflag_error_test_failure_count += 1,
            Lsode2Kflag::RepeatedErrorTestFailure => {
                self.native_kflag_repeated_error_test_failure_count += 1
            }
            Lsode2Kflag::ConvergenceFailure => self.native_kflag_convergence_failure_count += 1,
            Lsode2Kflag::RepeatedConvergenceFailure => {
                self.native_kflag_repeated_convergence_failure_count += 1
            }
        }
        match icf {
            Lsode2Icf::None => self.native_icf_none_count += 1,
            Lsode2Icf::RefreshRequested => self.native_icf_refresh_requested_count += 1,
            Lsode2Icf::RefreshDidNotRecover => self.native_icf_refresh_did_not_recover_count += 1,
        }
        match iret {
            Lsode2Iret::NormalFlow => self.native_iret_normal_flow_count += 1,
            Lsode2Iret::RescaleHistory => self.native_iret_rescale_history_count += 1,
            Lsode2Iret::RetryAfterErrorTestFailure => {
                self.native_iret_retry_after_error_test_failure_count += 1
            }
            Lsode2Iret::RestartWithDerivativeRefresh => {
                self.native_iret_restart_with_derivative_refresh_count += 1
            }
        }
        match redo_stage {
            Lsode2RedoStage::None => self.native_redo_none_count += 1,
            Lsode2RedoStage::CorrectorRefreshSameStep => {
                self.native_redo_corrector_refresh_same_step_count += 1
            }
            Lsode2RedoStage::CorrectorFailureRetry => {
                self.native_redo_corrector_failure_retry_count += 1
            }
            Lsode2RedoStage::ErrorTestRetry => self.native_redo_error_test_retry_count += 1,
            Lsode2RedoStage::RepeatedErrorReset => self.native_redo_repeated_error_reset_count += 1,
            Lsode2RedoStage::HistoryOrStepSizeChanged => {
                self.native_redo_history_or_step_size_changed_count += 1
            }
        }
    }

    pub fn record_predictor_ipup_trigger(&mut self, ipup_trigger: Lsode2IpupTrigger) {
        match ipup_trigger {
            Lsode2IpupTrigger::None => self.native_predictor_ipup_trigger_none_count += 1,
            Lsode2IpupTrigger::PredictorRcCcmax => {
                self.native_predictor_ipup_trigger_predictor_rc_ccmax_count += 1
            }
            Lsode2IpupTrigger::PredictorMsbp => {
                self.native_predictor_ipup_trigger_predictor_msbp_count += 1
            }
            Lsode2IpupTrigger::PredictorRcCcmaxAndMsbp => {
                self.native_predictor_ipup_trigger_predictor_rc_ccmax_and_msbp_count += 1
            }
            Lsode2IpupTrigger::FailurePath => {
                self.native_predictor_ipup_trigger_failure_path_count += 1
            }
        }
    }

    pub fn record_ialth(&mut self, ialth: usize) {
        if ialth == 0 {
            self.native_ialth_zero_count += 1;
        } else {
            self.native_ialth_positive_count += 1;
        }
        self.native_ialth_sum += ialth;
    }

    pub fn record_correction_assessment(&mut self, assessment: &Lsode2CorrectionAssessment) {
        match assessment.status {
            Lsode2CorrectionStatus::Continue => self.record_native_nonlinear_continue(),
            Lsode2CorrectionStatus::Converged => self.record_native_nonlinear_converged(),
            Lsode2CorrectionStatus::Diverged => self.record_native_nonlinear_diverged(),
            Lsode2CorrectionStatus::IterationLimitReached => {
                self.record_native_nonlinear_iteration_limit()
            }
        }
        if assessment.needs_jacobian_refresh {
            self.record_jacobian_refresh_request();
        }
    }

    pub fn record_step_cycle_outcome(&mut self, outcome: &Lsode2StepCycleOutcome) {
        match outcome {
            Lsode2StepCycleOutcome::Accepted { .. } => self.record_native_step_accept(),
            Lsode2StepCycleOutcome::Rejected { .. } => self.record_native_step_reject_error_test(),
            Lsode2StepCycleOutcome::NonlinearContinue { .. } => {}
            Lsode2StepCycleOutcome::NonlinearRejected { retry, .. } => {
                // DSTODA parity:
                // `ICF=1` same-step stale-J retry is not a true step rejection.
                // We classify it via shrink_factor=1 on nonlinear-convergence path.
                let same_step_stale_retry = retry.failure
                    == Lsode2StepFailure::NonlinearConvergence
                    && (retry.shrink_factor - 1.0).abs() <= f64::EPSILON;
                if same_step_stale_retry {
                    self.record_native_stale_jacobian_retry();
                } else {
                    self.record_native_step_reject_nonlinear();
                }
            }
        }
    }

    pub fn table_report(&self) -> String {
        format!(
            "lsode2_prepare_calls={} lsode2_prepare_ms_total={:.3} lsode2_solve_calls={} lsode2_solve_ms_total={:.3} preferred[a/b]={}/{} executed[a/b]={}/{} fallbacks={} native_steps[attempt/accept/reject_err/reject_nonlin]={}/{}/{}/{} native_nonlinear[continue/converged/diverged/iter_limit/stale_jac_retry]={}/{}/{}/{}/{} native_eval[residual_calls/residual_ms/jacobian_calls/jacobian_ms/linear_calls/linear_ms]={}/{:.3}/{}/{:.3}/{}/{:.3} jac_refresh={} dstoda_jcur[cur/stale]={}/{} dstoda_ipup[up/need]={}/{} dstoda_pred_ipup_trigger[none/rc/msbp/rc+msbp/fail]={}/{}/{}/{}/{} dstoda_ipup_trigger[none/rc/msbp/rc+msbp/fail]={}/{}/{}/{}/{} dstoda_kflag[ok/err/err_rep/conv/conv_rep]={}/{}/{}/{}/{} dstoda_icf[none/refresh/no_recover]={}/{}/{} dstoda_iret[normal/rescale/retry/restart]={}/{}/{}/{} dstoda_redo[none/corr_refresh/corr_retry/err_retry/err_reset/history]={}/{}/{}/{}/{}/{} dstoda_ialth[zero/pos/sum]={}/{}/{} bridge_prepare_calls={} bridge_solve_calls={} bridge_steps={} bridge_accepted_steps={} bridge_residual_calls={} bridge_jacobian_calls={} bridge_bdf[nfev/njev/nlu]={}/{}/{}",
            self.backend_prepare_calls,
            self.backend_prepare_ms_total,
            self.solve_calls,
            self.solve_ms_total,
            self.preferred_adams_count,
            self.preferred_bdf_count,
            self.executed_adams_count,
            self.executed_bdf_count,
            self.fallback_decision_count,
            self.native_step_attempts,
            self.native_step_accepts,
            self.native_step_rejects_error_test,
            self.native_step_rejects_nonlinear,
            self.native_nonlinear_continue_count,
            self.native_nonlinear_converged_count,
            self.native_nonlinear_diverged_count,
            self.native_nonlinear_iteration_limit_count,
            self.native_stale_jacobian_retry_count,
            self.native_residual_calls,
            self.native_residual_ms_total,
            self.native_jacobian_calls,
            self.native_jacobian_ms_total,
            self.native_linear_solve_calls,
            self.native_linear_solve_ms_total,
            self.native_jacobian_refresh_requests,
            self.native_jcur_current_count,
            self.native_jcur_stale_count,
            self.native_ipup_up_to_date_count,
            self.native_ipup_needs_update_count,
            self.native_predictor_ipup_trigger_none_count,
            self.native_predictor_ipup_trigger_predictor_rc_ccmax_count,
            self.native_predictor_ipup_trigger_predictor_msbp_count,
            self.native_predictor_ipup_trigger_predictor_rc_ccmax_and_msbp_count,
            self.native_predictor_ipup_trigger_failure_path_count,
            self.native_ipup_trigger_none_count,
            self.native_ipup_trigger_predictor_rc_ccmax_count,
            self.native_ipup_trigger_predictor_msbp_count,
            self.native_ipup_trigger_predictor_rc_ccmax_and_msbp_count,
            self.native_ipup_trigger_failure_path_count,
            self.native_kflag_ok_count,
            self.native_kflag_error_test_failure_count,
            self.native_kflag_repeated_error_test_failure_count,
            self.native_kflag_convergence_failure_count,
            self.native_kflag_repeated_convergence_failure_count,
            self.native_icf_none_count,
            self.native_icf_refresh_requested_count,
            self.native_icf_refresh_did_not_recover_count,
            self.native_iret_normal_flow_count,
            self.native_iret_rescale_history_count,
            self.native_iret_retry_after_error_test_failure_count,
            self.native_iret_restart_with_derivative_refresh_count,
            self.native_redo_none_count,
            self.native_redo_corrector_refresh_same_step_count,
            self.native_redo_corrector_failure_retry_count,
            self.native_redo_error_test_retry_count,
            self.native_redo_repeated_error_reset_count,
            self.native_redo_history_or_step_size_changed_count,
            self.native_ialth_zero_count,
            self.native_ialth_positive_count,
            self.native_ialth_sum,
            self.bridge_prepare_calls,
            self.bridge_solve_calls,
            self.bridge_step_calls,
            self.bridge_accepted_steps,
            self.bridge_residual_calls,
            self.bridge_jacobian_calls,
            self.bridge_bdf_nfev_total,
            self.bridge_bdf_njev_total,
            self.bridge_bdf_nlu_total,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::LSODE2::{
        Lsode2CorrectionControlConfig, Lsode2CorrectionController, Lsode2ErrorControlConfig,
        Lsode2ErrorController, Lsode2Icf, Lsode2Ipup, Lsode2IpupTrigger, Lsode2Iret,
        Lsode2JacobianCurrency, Lsode2Kflag, Lsode2RedoStage, Lsode2StepControlConfig,
        Lsode2StepCycle, Lsode2StepCycleOutcome, Lsode2SwitchReason, Lsode2Tolerance,
    };

    #[test]
    fn native_statistics_records_algorithm_decisions_and_bridge_sync() {
        let mut stats = Lsode2NativeStatistics::default();
        let adams = Lsode2SwitchDecision {
            preferred_family: Lsode2MethodFamily::Adams,
            executable_family: Some(Lsode2MethodFamily::Bdf),
            reason: Lsode2SwitchReason::AdamsEngineUnavailable,
            uses_fallback: true,
            message: "automatic policy prefers Adams, but Adams engine is unavailable",
        };
        stats.record_algorithm_decision(&adams);

        let bdf = Lsode2SwitchDecision {
            preferred_family: Lsode2MethodFamily::Bdf,
            executable_family: Some(Lsode2MethodFamily::Bdf),
            reason: Lsode2SwitchReason::StiffnessSuspected,
            uses_fallback: false,
            message: "automatic policy prefers BDF because stiffness is suspected",
        };
        stats.record_algorithm_decision(&bdf);

        let bridge = IvpBackendStatistics {
            backend_prepare_calls: 1,
            backend_prepare_ms_total: 3.0,
            solve_calls: 2,
            solve_ms_total: 5.0,
            step_calls: 8,
            nonlinear_solve_calls: 4,
            nonlinear_iterations_total: 9,
            residual_calls: 11,
            residual_ms_total: 7.0,
            jacobian_calls: 6,
            jacobian_ms_total: 4.0,
            bdf_nfev_total: 12,
            bdf_njev_total: 13,
            bdf_nlu_total: 14,
        };
        stats.sync_from_bridge(&bridge);

        assert_eq!(stats.algorithm_decision_calls, 2);
        assert_eq!(stats.preferred_adams_count, 1);
        assert_eq!(stats.preferred_bdf_count, 1);
        assert_eq!(stats.executed_bdf_count, 2);
        assert_eq!(stats.fallback_decision_count, 1);
        assert_eq!(stats.bridge_step_calls, 8);
        assert_eq!(stats.bridge_bdf_nlu_total, 14);
    }

    #[test]
    fn native_statistics_records_native_phase_counters() {
        let mut stats = Lsode2NativeStatistics::default();
        stats.record_native_step_attempt();
        stats.record_native_step_accept();
        stats.record_native_step_reject_error_test();
        stats.record_native_step_reject_nonlinear();
        stats.record_native_nonlinear_continue();
        stats.record_native_nonlinear_converged();
        stats.record_native_nonlinear_diverged();
        stats.record_native_nonlinear_iteration_limit();
        stats.record_native_stale_jacobian_retry();
        stats.record_jacobian_refresh_request();
        stats.record_backend_prepare_duration(Duration::from_millis(10));
        stats.record_solve_duration(Duration::from_millis(20));
        stats.record_native_residual_duration(Duration::from_millis(3));
        stats.record_native_jacobian_duration(Duration::from_millis(4));
        stats.record_native_linear_solve_duration(Duration::from_millis(5));
        stats.record_dstoda_flags(
            Lsode2JacobianCurrency::Stale,
            Lsode2Ipup::NeedsJacobianUpdate,
            Lsode2IpupTrigger::FailurePath,
            Lsode2Kflag::ConvergenceFailure,
            Lsode2Icf::RefreshRequested,
            Lsode2Iret::RetryAfterErrorTestFailure,
            Lsode2RedoStage::CorrectorRefreshSameStep,
        );
        stats.record_ialth(0);
        stats.record_ialth(3);

        assert_eq!(stats.native_step_attempts, 1);
        assert_eq!(stats.native_step_accepts, 1);
        assert_eq!(stats.native_step_rejects_error_test, 1);
        assert_eq!(stats.native_step_rejects_nonlinear, 1);
        assert_eq!(stats.native_stale_jacobian_retry_count, 1);
        assert_eq!(stats.native_nonlinear_continue_count, 1);
        assert_eq!(stats.native_nonlinear_converged_count, 1);
        assert_eq!(stats.native_nonlinear_diverged_count, 1);
        assert_eq!(stats.native_nonlinear_iteration_limit_count, 1);
        assert_eq!(stats.native_stale_jacobian_retry_count, 1);
        assert_eq!(stats.native_jacobian_refresh_requests, 1);
        assert_eq!(stats.native_residual_calls, 1);
        assert_eq!(stats.native_jacobian_calls, 1);
        assert_eq!(stats.native_linear_solve_calls, 1);
        assert_eq!(stats.native_jcur_stale_count, 1);
        assert_eq!(stats.native_ipup_needs_update_count, 1);
        assert_eq!(stats.native_ipup_trigger_failure_path_count, 1);
        assert_eq!(stats.native_kflag_convergence_failure_count, 1);
        assert_eq!(stats.native_icf_refresh_requested_count, 1);
        assert_eq!(stats.native_iret_retry_after_error_test_failure_count, 1);
        assert_eq!(stats.native_redo_corrector_refresh_same_step_count, 1);
        assert_eq!(stats.native_ialth_zero_count, 1);
        assert_eq!(stats.native_ialth_positive_count, 1);
        assert_eq!(stats.native_ialth_sum, 3);
        assert!(stats.backend_prepare_ms_total >= 10.0);
        assert!(stats.solve_ms_total >= 20.0);
    }

    #[test]
    fn native_statistics_can_track_real_step_cycle_paths() {
        let mut cycle = Lsode2StepCycle::new(
            super::super::state::Lsode2RuntimeState::new(
                0.0,
                &[1.0],
                0.1,
                3,
                Lsode2StepControlConfig::default(),
            )
            .unwrap(),
            Lsode2ErrorController::new(
                Lsode2Tolerance::scalar(1.0e-3, 1.0e-6),
                Lsode2ErrorControlConfig::default(),
            )
            .unwrap(),
        );
        let correction = Lsode2CorrectionController::new(
            Lsode2Tolerance::scalar(1.0e-3, 1.0e-6),
            Lsode2CorrectionControlConfig::default(),
        )
        .unwrap();
        let mut stats = Lsode2NativeStatistics::default();

        stats.record_prediction();
        let accepted_assessment = correction
            .assess_iteration(1, &[0.905], &[1.0e-5], &[1.0e-5], None, None, 1)
            .unwrap();
        stats.record_correction_assessment(&accepted_assessment);
        let accepted = cycle
            .finish_with_correction(
                0.1,
                &[0.905],
                &[1.0e-5],
                &[1.0e-5],
                &correction,
                None,
                None,
                1,
            )
            .unwrap();
        stats.record_step_cycle_outcome(&accepted);

        stats.record_prediction();
        let rejected_assessment = correction
            .assess_iteration(1, &[0.8], &[1.0e-2], &[1.0e-2], Some(1.0e-4), Some(0.5), 2)
            .unwrap();
        stats.record_correction_assessment(&rejected_assessment);
        let rejected = cycle
            .finish_with_correction(
                0.2,
                &[0.8],
                &[1.0e-2],
                &[1.0e-2],
                &correction,
                Some(1.0e-4),
                Some(0.5),
                2,
            )
            .unwrap();
        stats.record_step_cycle_outcome(&rejected);

        assert_eq!(stats.native_step_attempts, 2);
        assert_eq!(stats.native_step_accepts, 1);
        // DSTODA parity: first convergence-failure branch can be ICF=1
        // same-step refresh (no true rejected step yet).
        assert_eq!(stats.native_step_rejects_nonlinear, 0);
        assert_eq!(stats.native_stale_jacobian_retry_count, 1);
        assert_eq!(stats.native_nonlinear_converged_count, 1);
        assert_eq!(stats.native_nonlinear_diverged_count, 1);
        assert_eq!(stats.native_jacobian_refresh_requests, 1);

        match rejected {
            Lsode2StepCycleOutcome::NonlinearRejected { state, .. } => {
                assert_eq!(state.convergence_failures, 0);
            }
            other => panic!("expected nonlinear rejection, got {other:?}"),
        }
    }

    #[test]
    fn native_statistics_treat_same_step_refresh_as_stale_retry_not_reject() {
        let mut cycle = super::super::step_cycle::Lsode2StepCycle::new(
            super::super::state::Lsode2RuntimeState::new(
                0.0,
                &[1.0],
                0.1,
                3,
                Lsode2StepControlConfig::default(),
            )
            .unwrap(),
            Lsode2ErrorController::new(
                Lsode2Tolerance::scalar(1.0e-3, 1.0e-6),
                Lsode2ErrorControlConfig::default(),
            )
            .unwrap(),
        );
        let correction = Lsode2CorrectionController::new(
            Lsode2Tolerance::scalar(1.0e-3, 1.0e-6),
            Lsode2CorrectionControlConfig::default(),
        )
        .unwrap();
        let mut stats = Lsode2NativeStatistics::default();

        cycle.mark_jacobian_stale();
        let rejected = cycle
            .finish_with_correction(
                0.1,
                &[0.8],
                &[1.0e-2],
                &[1.0e-2],
                &correction,
                Some(1.0e-4),
                Some(0.5),
                2,
            )
            .unwrap();

        stats.record_step_cycle_outcome(&rejected);
        assert_eq!(stats.native_stale_jacobian_retry_count, 1);
        assert_eq!(stats.native_step_rejects_nonlinear, 0);
    }
}
