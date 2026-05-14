//! Native LSODE2 DSTODA step-cycle choreography.
//!
//! This module owns predictor/corrector step control, local error testing,
//! retry/terminal branches, and ODEPACK-style control-plane flags
//! (`KFLAG/ICF/IRET/IREDO/IPUP/JCUR`) for one-step progression.

use super::adams_engine::Lsode2AdamsDcfodeTables;
use super::algorithm::Lsode2SwitchTelemetry;
use super::correction::{
    Lsode2CorrectionAssessment, Lsode2CorrectionController, Lsode2CorrectionError,
    Lsode2CorrectionStatus, Lsode2DstodaCorrectorContext,
};
use super::dcfode::{Lsode2BdfDcfodeTables, Lsode2DcfodeError};
use super::dstoda_state::{
    Lsode2CorrectorFailureDecision, Lsode2DstodaState, Lsode2Icf, Lsode2Ipup, Lsode2IpupTrigger,
    Lsode2Iredo, Lsode2Iret, Lsode2IterationMode, Lsode2JacobianCurrency,
    Lsode2JacobianUpdateRequest, Lsode2Kflag, Lsode2RedoStage,
};
use super::error_control::{
    Lsode2ErrorControlError, Lsode2ErrorController, Lsode2ErrorTestAction, Lsode2ErrorTestResult,
};
use super::history::{Lsode2HistoryError, weighted_rms_norm};
use super::order_selection::{
    Lsode2OrderCandidate, Lsode2OrderSelectionDecision, Lsode2OrderSelectionError,
};
use super::state::{
    Lsode2CorrectionAcceptUpdateMode, Lsode2RuntimeState, Lsode2RuntimeStateError,
    Lsode2RuntimeStateSnapshot,
};
use super::step_control::{Lsode2AcceptDecision, Lsode2RetryAction, Lsode2RetryDecision};

const DSTODA_HMIN_GUARD_FACTOR: f64 = 1.00001;

#[derive(Debug, Clone, PartialEq)]
pub enum Lsode2StepCycleError {
    Correction(Lsode2CorrectionError),
    Dcfode(Lsode2DcfodeError),
    RuntimeState(Lsode2RuntimeStateError),
    ErrorControl(Lsode2ErrorControlError),
    History(Lsode2HistoryError),
    OrderSelection(Lsode2OrderSelectionError),
}

impl std::fmt::Display for Lsode2StepCycleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Correction(err) => write!(f, "{err}"),
            Self::Dcfode(err) => write!(f, "{err}"),
            Self::RuntimeState(err) => write!(f, "{err}"),
            Self::ErrorControl(err) => write!(f, "{err}"),
            Self::History(err) => write!(f, "{err}"),
            Self::OrderSelection(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for Lsode2StepCycleError {}

impl From<Lsode2CorrectionError> for Lsode2StepCycleError {
    fn from(value: Lsode2CorrectionError) -> Self {
        Self::Correction(value)
    }
}

impl From<Lsode2DcfodeError> for Lsode2StepCycleError {
    fn from(value: Lsode2DcfodeError) -> Self {
        Self::Dcfode(value)
    }
}

impl From<Lsode2RuntimeStateError> for Lsode2StepCycleError {
    fn from(value: Lsode2RuntimeStateError) -> Self {
        Self::RuntimeState(value)
    }
}

impl From<Lsode2ErrorControlError> for Lsode2StepCycleError {
    fn from(value: Lsode2ErrorControlError) -> Self {
        Self::ErrorControl(value)
    }
}

impl From<Lsode2HistoryError> for Lsode2StepCycleError {
    fn from(value: Lsode2HistoryError) -> Self {
        Self::History(value)
    }
}

impl From<Lsode2OrderSelectionError> for Lsode2StepCycleError {
    fn from(value: Lsode2OrderSelectionError) -> Self {
        Self::OrderSelection(value)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Lsode2PredictedStep {
    pub t_trial: f64,
    pub h_trial: f64,
    pub order: usize,
    pub y_pred: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Lsode2StepCycleOutcome {
    Accepted {
        t_new: f64,
        y_new: Vec<f64>,
        error_test: Lsode2ErrorTestResult,
        accept: Lsode2AcceptDecision,
        state: Lsode2RuntimeStateSnapshot,
    },
    Rejected {
        error_test: Lsode2ErrorTestResult,
        retry: Lsode2RetryDecision,
        state: Lsode2RuntimeStateSnapshot,
    },
    NonlinearContinue {
        assessment: Lsode2CorrectionAssessment,
    },
    NonlinearRejected {
        assessment: Lsode2CorrectionAssessment,
        retry: Lsode2RetryDecision,
        state: Lsode2RuntimeStateSnapshot,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lsode2StepMethod {
    BdfLike,
    AdamsLike,
}

#[derive(Debug, Clone)]
pub struct Lsode2StepCycle {
    state: Lsode2RuntimeState,
    error_control: Lsode2ErrorController,
    dstoda: Lsode2DstodaState,
    iteration_mode: Lsode2IterationMode,
    method: Lsode2StepMethod,
    adams_pdest: f64,
    adams_pdlast: f64,
    last_stiffness_ratio_estimate: Option<f64>,
    last_adams_step_size_cap_estimate: Option<f64>,
    last_bdf_step_size_cap_estimate: Option<f64>,
}

impl Lsode2StepCycle {
    pub fn new(state: Lsode2RuntimeState, error_control: Lsode2ErrorController) -> Self {
        Self::new_with_method(state, error_control, Lsode2StepMethod::BdfLike)
    }

    pub fn new_with_method(
        state: Lsode2RuntimeState,
        error_control: Lsode2ErrorController,
        method: Lsode2StepMethod,
    ) -> Self {
        Self {
            state,
            error_control,
            dstoda: Lsode2DstodaState::default(),
            iteration_mode: Lsode2IterationMode::JacobianBased,
            method,
            adams_pdest: 0.0,
            adams_pdlast: 0.0,
            last_stiffness_ratio_estimate: None,
            last_adams_step_size_cap_estimate: None,
            last_bdf_step_size_cap_estimate: None,
        }
    }

    pub fn state(&self) -> &Lsode2RuntimeState {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut Lsode2RuntimeState {
        &mut self.state
    }

    pub fn error_control(&self) -> &Lsode2ErrorController {
        &self.error_control
    }

    pub fn dstoda_state(&self) -> Lsode2DstodaState {
        self.dstoda
    }

    pub fn mark_jacobian_current(&mut self) {
        let nst = self.state.step_control_snapshot().accepted_steps;
        self.dstoda.mark_jacobian_current(nst);
    }

    pub fn mark_jacobian_stale(&mut self) {
        self.dstoda.mark_jacobian_stale();
    }

    pub fn iteration_mode(&self) -> Lsode2IterationMode {
        self.iteration_mode
    }

    pub fn method(&self) -> Lsode2StepMethod {
        self.method
    }

    pub fn adams_pdest(&self) -> f64 {
        self.adams_pdest
    }

    pub fn adams_pdlast(&self) -> f64 {
        self.adams_pdlast
    }

    pub fn set_iteration_mode(&mut self, mode: Lsode2IterationMode) {
        self.iteration_mode = mode;
    }

    pub fn predict(&mut self) -> Result<Lsode2PredictedStep, Lsode2StepCycleError> {
        let nst = self.state.step_control_snapshot().accepted_steps;
        self.dstoda
            .maybe_request_jacobian_update_before_predict(nst, self.iteration_mode);
        let t_current = self.state.t();
        let h_current = self.state.h();
        if t_current + h_current == t_current {
            // LSODE/LSODA NHNIL branch parity:
            // when machine precision makes TN + H == TN, Fortran increments NHNIL
            // and only warns up to MXHNIL times. This is controller-plane telemetry;
            // it must not alter step math directly here.
            self.state.record_null_step_event();
        }
        let y_pred = self.state.predict_from_nordsieck()?.to_vec();
        Ok(Lsode2PredictedStep {
            t_trial: t_current + h_current,
            h_trial: h_current,
            order: self.state.order(),
            y_pred,
        })
    }

    pub fn finish_with_local_error(
        &mut self,
        t_new: f64,
        y_candidate: &[f64],
        local_error: &[f64],
    ) -> Result<Lsode2StepCycleOutcome, Lsode2StepCycleError> {
        let h_before_accept = self.state.h();
        let order_before_accept = self.state.order();
        let error_test = self.error_control.evaluate_local_error(
            y_candidate,
            local_error,
            self.state.order(),
        )?;
        match error_test.action {
            Lsode2ErrorTestAction::Accept => {
                let order_decision =
                    self.select_post_accept_order(y_candidate, error_test.error_norm, None)?;
                let suggested_growth = error_test
                    .suggested_growth
                    .min(order_decision.suggested_growth);
                let accept = self.state.accept_step(
                    t_new,
                    y_candidate,
                    suggested_growth,
                    error_test.error_norm,
                    order_decision.order_new,
                )?;
                self.finalize_dstoda_after_accept(h_before_accept, order_before_accept, &accept);
                Ok(Lsode2StepCycleOutcome::Accepted {
                    t_new,
                    y_new: y_candidate.to_vec(),
                    error_test,
                    accept,
                    state: self.state.snapshot(),
                })
            }
            Lsode2ErrorTestAction::Reject => {
                let retry =
                    self.select_and_apply_error_test_retry(y_candidate, error_test.error_norm)?;
                self.record_error_test_retry_kflag(retry.action);
                Ok(Lsode2StepCycleOutcome::Rejected {
                    error_test,
                    retry,
                    state: self.state.snapshot(),
                })
            }
        }
    }

    pub fn finish_with_correction(
        &mut self,
        t_new: f64,
        y_candidate: &[f64],
        correction: &[f64],
        accumulated_correction: &[f64],
        correction_controller: &Lsode2CorrectionController,
        previous_weighted_norm: Option<f64>,
        previous_rate_estimate: Option<f64>,
        iteration: usize,
    ) -> Result<Lsode2StepCycleOutcome, Lsode2StepCycleError> {
        let roundoff_tolerance = correction_controller.dstoda_roundoff_tolerance(y_candidate)?;
        let assessment = correction_controller.assess_iteration_with_dstoda_context(
            self.state.order(),
            y_candidate,
            correction,
            accumulated_correction,
            previous_weighted_norm,
            previous_rate_estimate,
            iteration,
            Some(tesco2_for_method(self.method, self.state.order())),
            Some(Lsode2DstodaCorrectorContext {
                method_is_adams: self.method == Lsode2StepMethod::AdamsLike,
                previous_rate_max: None,
                h_el1_abs: None,
                roundoff_tolerance: Some(roundoff_tolerance),
            }),
        )?;
        match assessment.status {
            Lsode2CorrectionStatus::Converged => self.finish_with_converged_correction(
                t_new,
                y_candidate,
                accumulated_correction,
                &assessment.local_error,
            ),
            Lsode2CorrectionStatus::Continue => {
                Ok(Lsode2StepCycleOutcome::NonlinearContinue { assessment })
            }
            Lsode2CorrectionStatus::Diverged | Lsode2CorrectionStatus::IterationLimitReached => {
                let retry = self.apply_corrector_failure_policy()?;
                Ok(Lsode2StepCycleOutcome::NonlinearRejected {
                    assessment,
                    retry,
                    state: self.state.snapshot(),
                })
            }
        }
    }

    pub fn finish_after_converged_correction(
        &mut self,
        t_new: f64,
        y_candidate: &[f64],
        accumulated_correction: &[f64],
        local_error: &[f64],
    ) -> Result<Lsode2StepCycleOutcome, Lsode2StepCycleError> {
        self.finish_with_converged_correction(
            t_new,
            y_candidate,
            accumulated_correction,
            local_error,
        )
    }

    pub fn reject_after_nonlinear_failure(
        &mut self,
    ) -> Result<Lsode2RetryDecision, Lsode2StepCycleError> {
        self.apply_corrector_failure_policy()
    }

    pub fn retry_after_stale_jacobian_nonlinear_failure(
        &mut self,
    ) -> Result<Lsode2RetryDecision, Lsode2StepCycleError> {
        self.dstoda.mark_jacobian_stale();
        self.apply_corrector_failure_policy()
    }

    pub fn jacobian_currency(&self) -> Lsode2JacobianCurrency {
        self.dstoda.jacobian_currency()
    }

    pub fn jacobian_update_request(&self) -> Lsode2JacobianUpdateRequest {
        self.dstoda.jacobian_update_request()
    }

    pub fn kflag(&self) -> Lsode2Kflag {
        self.dstoda.kflag()
    }

    pub fn kflag_code(&self) -> i32 {
        self.dstoda.kflag().code()
    }

    pub fn ipup(&self) -> Lsode2Ipup {
        self.dstoda.ipup()
    }

    pub fn ipup_trigger(&self) -> Lsode2IpupTrigger {
        self.dstoda.ipup_trigger()
    }

    pub fn icf(&self) -> Lsode2Icf {
        self.dstoda.icf()
    }

    pub fn iret(&self) -> Lsode2Iret {
        self.dstoda.iret()
    }

    pub fn redo_stage(&self) -> Lsode2RedoStage {
        self.dstoda.redo_stage()
    }

    pub fn iredo(&self) -> Lsode2Iredo {
        self.dstoda.iredo()
    }

    #[cfg(test)]
    pub(crate) fn force_dstoda_coefficient_ratio_for_test(&mut self, rc: f64) {
        self.dstoda.set_coefficient_ratio(rc);
    }

    pub fn switch_telemetry(&self, stiffness_ratio: Option<f64>) -> Lsode2SwitchTelemetry {
        let mut telemetry = self
            .state
            .switch_telemetry(stiffness_ratio.or(self.last_stiffness_ratio_estimate));
        telemetry.adams_step_size_cap_estimate = self.last_adams_step_size_cap_estimate;
        telemetry.bdf_step_size_cap_estimate = self.last_bdf_step_size_cap_estimate;
        telemetry
    }

    pub fn record_adams_lipschitz_estimate_from_assessment(
        &mut self,
        assessment: &Lsode2CorrectionAssessment,
    ) {
        if self.method != Lsode2StepMethod::AdamsLike {
            return;
        }
        if let Some(candidate) = assessment.pdest_candidate {
            if candidate.is_finite() && candidate > 0.0 {
                self.adams_pdest = self.adams_pdest.max(candidate);
                self.adams_pdlast = self.adams_pdest;
            }
        }
    }

    fn finish_with_converged_correction(
        &mut self,
        t_new: f64,
        y_candidate: &[f64],
        accumulated_correction: &[f64],
        local_error: &[f64],
    ) -> Result<Lsode2StepCycleOutcome, Lsode2StepCycleError> {
        let h_before_accept = self.state.h();
        let order_before_accept = self.state.order();
        self.dstoda.record_corrector_converged();
        let error_test = self.error_control.evaluate_local_error(
            y_candidate,
            local_error,
            self.state.order(),
        )?;
        match error_test.action {
            Lsode2ErrorTestAction::Accept => {
                let order_decision = self.select_post_accept_order(
                    y_candidate,
                    error_test.error_norm,
                    Some(accumulated_correction),
                )?;
                let suggested_growth = error_test
                    .suggested_growth
                    .min(order_decision.suggested_growth);
                let update_mode = if self.method == Lsode2StepMethod::AdamsLike {
                    Lsode2CorrectionAcceptUpdateMode::AdamsDstodaDirect
                } else {
                    Lsode2CorrectionAcceptUpdateMode::BdfDstodaDirect
                };
                let accept = self.state.accept_step_with_correction_mode(
                    t_new,
                    y_candidate,
                    accumulated_correction,
                    suggested_growth,
                    error_test.error_norm,
                    order_decision.order_new,
                    update_mode,
                )?;
                self.finalize_dstoda_after_accept(h_before_accept, order_before_accept, &accept);
                Ok(Lsode2StepCycleOutcome::Accepted {
                    t_new,
                    y_new: y_candidate.to_vec(),
                    error_test,
                    accept,
                    state: self.state.snapshot(),
                })
            }
            Lsode2ErrorTestAction::Reject => {
                let retry =
                    self.select_and_apply_error_test_retry(y_candidate, error_test.error_norm)?;
                self.record_error_test_retry_kflag(retry.action);
                Ok(Lsode2StepCycleOutcome::Rejected {
                    error_test,
                    retry,
                    state: self.state.snapshot(),
                })
            }
        }
    }

    fn apply_corrector_failure_policy(
        &mut self,
    ) -> Result<Lsode2RetryDecision, Lsode2StepCycleError> {
        let decision = self
            .dstoda
            .decide_after_corrector_failure(self.iteration_mode);
        let retry = match decision {
            Lsode2CorrectorFailureDecision::RefreshJacobianSameStep => {
                let retry = self.state.retry_after_stale_jacobian_nonlinear_failure()?;
                retry
            }
            Lsode2CorrectorFailureDecision::RetractAndShrinkStep => {
                let retry = self.state.reject_after_nonlinear_failure()?;
                // DSTODA label-430 parity:
                // `IPUP = MITER` is set only on the retry path (after passing
                // HMIN/MXNCF terminal guards), not preemptively.
                if retry.action == Lsode2RetryAction::RetryWithJacobianRefresh {
                    self.dstoda
                        .request_jacobian_update(Lsode2IpupTrigger::FailurePath);
                }
                retry
            }
        };

        // ODEPACK parity:
        // DSTODA exits with KFLAG = -2 for both:
        // 1) repeated convergence failures (MXNCF reached), and
        // 2) inability to reduce H further (H <= HMIN).
        // In our split retry API both map to terminal retry actions.
        if matches!(
            retry.action,
            Lsode2RetryAction::FailRepeatedConvergenceFailures
                | Lsode2RetryAction::FailStepSizeUnderflow
        ) {
            self.dstoda.record_repeated_convergence_failure();
        }

        Ok(retry)
    }

    fn record_error_test_retry_kflag(&mut self, action: Lsode2RetryAction) {
        if action == Lsode2RetryAction::FailRepeatedErrorTestFailures {
            self.dstoda.record_repeated_error_test_failure();
        } else if self.dstoda.redo_stage() == Lsode2RedoStage::RepeatedErrorReset {
            self.dstoda.record_repeated_error_test_reset();
        } else {
            self.dstoda.record_error_test_failure();
        }
    }

    pub fn select_post_accept_order(
        &mut self,
        y_candidate: &[f64],
        current_error_norm: f64,
        accumulated_correction: Option<&[f64]>,
    ) -> Result<Lsode2OrderSelectionDecision, Lsode2StepCycleError> {
        let order_current = self.state.order();
        let order_cap = self.state.next_accept_order_cap();
        let weights = self.error_control.weights_for(y_candidate)?;
        let n = y_candidate.len();

        let lower_error_norm = if order_current > 1 {
            let tesco1_current = tesco1_for_method(self.method, order_current);
            if tesco1_current > 0.0 && tesco1_current.is_finite() {
                weighted_rms_norm(self.state.nordsieck().col(order_current)?, &weights)?
                    / tesco1_current
            } else {
                f64::INFINITY
            }
        } else {
            f64::INFINITY
        };
        let tesco3_current = tesco3_for_method(self.method, order_current);
        // ODEPACK/DSTODA parity:
        // RHUP is based on DUP = ||ACOR - SAVF||/TESCO(3,NQ), where SAVF is the
        // staged correction snapshot from the previous eligible step.
        // If this staged snapshot is unavailable, do not synthesize RHUP from
        // backward differences; treat the q+1 candidate as unavailable.
        let higher_error_norm = if let (Some(current), Some(staged_previous)) = (
            accumulated_correction,
            self.state.staged_higher_order_correction(),
        ) {
            if order_current < order_cap && current.len() == n && staged_previous.len() == n {
                let mut diff = vec![0.0; n];
                for i in 0..n {
                    diff[i] = current[i] - staged_previous[i];
                }
                if tesco3_current > 0.0 && tesco3_current.is_finite() {
                    weighted_rms_norm(&diff, &weights)? / tesco3_current
                } else {
                    f64::INFINITY
                }
            } else {
                f64::INFINITY
            }
        } else {
            f64::INFINITY
        };

        let decision = select_dstoda_order_and_growth(
            self.method,
            self.state.h(),
            self.adams_pdlast,
            order_current,
            order_cap,
            current_error_norm,
            lower_error_norm,
            Some(higher_error_norm),
            false,
            true,
        );
        self.update_switch_telemetry_hints_from_order_decision(
            &decision,
            order_current,
            self.state.h(),
        );
        self.reset_adams_pdest_after_rh_selection();
        Ok(decision)
    }

    fn select_and_apply_error_test_retry(
        &mut self,
        y_candidate: &[f64],
        current_error_norm: f64,
    ) -> Result<Lsode2RetryDecision, Lsode2StepCycleError> {
        let snapshot = self.state.step_control_snapshot();
        let failure_count = snapshot.consecutive_error_test_failures + 1;
        let next_kflag = -(failure_count as i32);

        // DSTODA label 500:
        // IF (ABS(H) .LE. HMIN*1.00001D0) GO TO 660
        // We mirror the guard with the same tolerance factor.
        if self.state.h().abs() <= self.state.step_control_config().h_min * DSTODA_HMIN_GUARD_FACTOR
        {
            return self
                .state
                .reject_after_error_test_with_hint(self.state.h(), self.state.order())
                .map_err(Into::into);
        }

        // DSTODA label 640 trigger:
        // IF (KFLAG .LE. -3) GO TO 640
        // We always pass through the label-640 reset choreography first.
        // The terminal repeated-error exit (`KFLAG = -10`) is then emitted
        // by the retry decision produced by that reset path.
        if next_kflag <= -3 {
            let retry = self
                .state
                .reset_after_repeated_error_failures()
                .map_err(Lsode2StepCycleError::from)?;
            if retry.action == Lsode2RetryAction::Retry {
                self.dstoda.record_repeated_error_test_reset();
            }
            return Ok(retry);
        }

        let order_current = self.state.order();
        let weights = self.error_control.weights_for(y_candidate)?;
        let tesco1_current = tesco1_for_method(self.method, order_current);

        let mut lower_error_norm = 0.0;
        if order_current > 1 && tesco1_current > 0.0 && tesco1_current.is_finite() {
            lower_error_norm =
                weighted_rms_norm(self.state.nordsieck().col(order_current)?, &weights)?
                    / tesco1_current;
        }

        let decision = select_dstoda_order_and_growth(
            self.method,
            self.state.h(),
            self.adams_pdlast,
            order_current,
            order_current,
            current_error_norm,
            lower_error_norm,
            None,
            true,
            false,
        );
        self.update_switch_telemetry_hints_from_order_decision(
            &decision,
            order_current,
            self.state.h(),
        );
        self.reset_adams_pdest_after_rh_selection();
        let order_new = decision.order_new;
        let mut rh = decision.suggested_growth.min(1.0);
        // DSTODA label 620:
        // IF (KFLAG .LE. -2) RH = MIN(RH,0.2D0)
        if next_kflag <= -2 {
            rh = rh.min(0.2);
        }

        let h_new = self.state.h() * rh;
        self.state
            .reject_after_error_test_with_hint(h_new, order_new)
            .map_err(Into::into)
    }

    fn finalize_dstoda_after_accept(
        &mut self,
        h_before_accept: f64,
        order_before_accept: usize,
        accept: &Lsode2AcceptDecision,
    ) {
        let rc = coefficient_ratio_h_el1_for_method(
            self.method,
            h_before_accept,
            order_before_accept,
            accept.h_next,
            accept.order_new,
        );
        self.dstoda.set_coefficient_ratio(rc);
        self.dstoda.record_step_accepted();
        if accept.h_next != h_before_accept || accept.order_new != order_before_accept {
            self.dstoda.record_history_or_step_size_change();
        }
    }

    fn reset_adams_pdest_after_rh_selection(&mut self) {
        if self.method == Lsode2StepMethod::AdamsLike {
            // DSTODA parity: after RH candidates are stability-limited using PDLAST,
            // the step-local PDEST accumulator is cleared for the next step.
            self.adams_pdest = 0.0;
        }
    }

    fn update_switch_telemetry_hints_from_order_decision(
        &mut self,
        decision: &Lsode2OrderSelectionDecision,
        order_current: usize,
        h_current: f64,
    ) {
        let cap = sanitize_positive_finite(decision.current_factor);
        match self.method {
            Lsode2StepMethod::BdfLike => {
                self.last_bdf_step_size_cap_estimate = cap;
            }
            Lsode2StepMethod::AdamsLike => {
                self.last_adams_step_size_cap_estimate = cap;
                let pdh = dstoda_adams_pdh(h_current, self.adams_pdlast);
                let sm1 = dstoda_adams_sm1(order_current).max(1.0e-12);
                self.last_stiffness_ratio_estimate = sanitize_positive_finite(pdh / sm1);
            }
        }
    }
}

fn sanitize_positive_finite(value: f64) -> Option<f64> {
    if value.is_finite() && value > 0.0 {
        Some(value)
    } else {
        None
    }
}

fn coefficient_ratio_h_el1_for_method(
    method: Lsode2StepMethod,
    h_old: f64,
    order_old: usize,
    h_new: f64,
    order_new: usize,
) -> f64 {
    let (old_coeff, new_coeff) = match method {
        Lsode2StepMethod::BdfLike => {
            let bdf = Lsode2BdfDcfodeTables::default();
            (
                bdf.order(order_old)
                    .map(|c| h_old * c.el[0])
                    .unwrap_or(h_old),
                bdf.order(order_new)
                    .map(|c| h_new * c.el[0])
                    .unwrap_or(h_new),
            )
        }
        Lsode2StepMethod::AdamsLike => {
            let adams = Lsode2AdamsDcfodeTables::default();
            (
                adams
                    .order(order_old)
                    .map(|c| h_old * c.el[1])
                    .unwrap_or(h_old),
                adams
                    .order(order_new)
                    .map(|c| h_new * c.el[1])
                    .unwrap_or(h_new),
            )
        }
    };

    if old_coeff == 0.0 {
        1.0
    } else {
        new_coeff / old_coeff
    }
}

fn select_dstoda_order_and_growth(
    method: Lsode2StepMethod,
    h: f64,
    adams_pdlast: f64,
    order_current: usize,
    order_cap: usize,
    current_error_norm: f64,
    lower_error_norm: f64,
    higher_error_norm: Option<f64>,
    kflag_negative: bool,
    allow_order_increase: bool,
) -> Lsode2OrderSelectionDecision {
    let l = order_current + 1;
    let mut rhsm = dstoda_rh_sm(current_error_norm, l);
    let mut rhdn = if order_current > 1 {
        dstoda_rh_dn(lower_error_norm, order_current)
    } else {
        0.0
    };
    let mut rhup = if allow_order_increase && order_current < order_cap {
        higher_error_norm
            .map(|dup| dstoda_rh_up(dup, l + 1))
            .unwrap_or(0.0)
    } else {
        0.0
    };
    let mut pdh = 0.0;
    if method == Lsode2StepMethod::AdamsLike {
        pdh = dstoda_adams_pdh(h, adams_pdlast);
        if pdh > 0.0 {
            if allow_order_increase && l < order_cap + 1 {
                rhup = rhup.min(dstoda_adams_sm1(l) / pdh);
            }
            rhsm = rhsm.min(dstoda_adams_sm1(order_current) / pdh);
            if order_current > 1 {
                rhdn = rhdn.min(dstoda_adams_sm1(order_current - 1) / pdh);
            }
        }
    }

    let (candidate, mut rh, mut order_new) = if rhsm >= rhup {
        if rhsm < rhdn {
            (
                Lsode2OrderCandidate::Lower,
                rhdn,
                order_current.saturating_sub(1).max(1),
            )
        } else {
            (Lsode2OrderCandidate::Current, rhsm, order_current)
        }
    } else if rhup > rhdn {
        (
            Lsode2OrderCandidate::Higher,
            rhup,
            (order_current + 1).min(order_cap),
        )
    } else {
        (
            Lsode2OrderCandidate::Lower,
            rhdn,
            order_current.saturating_sub(1).max(1),
        )
    };

    if kflag_negative && candidate == Lsode2OrderCandidate::Lower && rh > 1.0 {
        rh = 1.0;
    }

    if candidate == Lsode2OrderCandidate::Higher && rh < 1.1 {
        order_new = order_current;
        rh = 1.0;
    }

    let stability_limited_bypass = method == Lsode2StepMethod::AdamsLike
        && pdh > 0.0
        && order_new >= 1
        && order_new <= 12
        && (rh * pdh * 1.00001) < dstoda_adams_sm1(order_new);
    if !kflag_negative && rh < 1.1 && !stability_limited_bypass {
        order_new = order_current;
        rh = 1.0;
    }

    if !rh.is_finite() || rh <= 0.0 {
        order_new = order_current;
        rh = 1.0;
    }

    Lsode2OrderSelectionDecision {
        candidate: if order_new == order_current {
            Lsode2OrderCandidate::Current
        } else {
            candidate
        },
        order_new,
        suggested_growth: rh,
        lower_factor: rhdn,
        current_factor: rhsm,
        higher_factor: rhup,
    }
}

fn dstoda_rh_sm(dsm: f64, l: usize) -> f64 {
    if !dsm.is_finite() || dsm <= 0.0 {
        return 0.0;
    }
    let exsm = 1.0 / l as f64;
    1.0 / (1.2 * dsm.powf(exsm) + 1.2e-6)
}

fn dstoda_rh_dn(ddn: f64, nq: usize) -> f64 {
    if !ddn.is_finite() || ddn <= 0.0 || nq == 0 {
        return 0.0;
    }
    let exdn = 1.0 / nq as f64;
    1.0 / (1.3 * ddn.powf(exdn) + 1.3e-6)
}

fn dstoda_rh_up(dup: f64, lp1: usize) -> f64 {
    if !dup.is_finite() || dup <= 0.0 || lp1 == 0 {
        return 0.0;
    }
    let exup = 1.0 / lp1 as f64;
    1.0 / (1.4 * dup.powf(exup) + 1.4e-6)
}

fn dstoda_adams_pdh(h: f64, pdlast: f64) -> f64 {
    if !h.is_finite() || !pdlast.is_finite() {
        return 0.0;
    }
    (h.abs() * pdlast).max(1.0e-6)
}

fn dstoda_adams_sm1(index_1_based: usize) -> f64 {
    const SM1: [f64; 12] = [
        0.5, 0.575, 0.55, 0.45, 0.35, 0.25, 0.2, 0.15, 0.1, 0.075, 0.05, 0.025,
    ];
    if (1..=12).contains(&index_1_based) {
        SM1[index_1_based - 1]
    } else {
        SM1[11]
    }
}

fn tesco1_for_method(method: Lsode2StepMethod, order: usize) -> f64 {
    match method {
        Lsode2StepMethod::BdfLike => Lsode2BdfDcfodeTables::default()
            .tesco1(order)
            .unwrap_or(0.0),
        Lsode2StepMethod::AdamsLike => Lsode2AdamsDcfodeTables::default()
            .order(order)
            .map(|c| c.tesco1)
            .unwrap_or(0.0),
    }
}

fn tesco3_for_method(method: Lsode2StepMethod, order: usize) -> f64 {
    match method {
        Lsode2StepMethod::BdfLike => Lsode2BdfDcfodeTables::default()
            .tesco3(order)
            .unwrap_or(0.0),
        Lsode2StepMethod::AdamsLike => Lsode2AdamsDcfodeTables::default()
            .order(order)
            .map(|c| c.tesco3)
            .unwrap_or(0.0),
    }
}

fn tesco2_for_method(method: Lsode2StepMethod, order: usize) -> f64 {
    match method {
        Lsode2StepMethod::BdfLike => Lsode2BdfDcfodeTables::default()
            .tesco2(order)
            .unwrap_or(1.0),
        Lsode2StepMethod::AdamsLike => Lsode2AdamsDcfodeTables::default()
            .order(order)
            .map(|c| c.tesco2)
            .unwrap_or(1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::LSODE2::{
        Lsode2CorrectionControlConfig, Lsode2ErrorControlConfig, Lsode2StepControlConfig,
        Lsode2Tolerance,
    };

    fn make_cycle() -> Lsode2StepCycle {
        make_cycle_with_step_config(Lsode2StepControlConfig::default())
    }

    fn make_cycle_with_step_config(step_config: Lsode2StepControlConfig) -> Lsode2StepCycle {
        let state = Lsode2RuntimeState::new(0.0, &[1.0], 0.1, 3, step_config).unwrap();
        let error_control = Lsode2ErrorController::new(
            Lsode2Tolerance::scalar(1.0e-3, 1.0e-6),
            Lsode2ErrorControlConfig::default(),
        )
        .unwrap();
        Lsode2StepCycle::new(state, error_control)
    }

    fn make_adams_cycle_with_step_config(step_config: Lsode2StepControlConfig) -> Lsode2StepCycle {
        let state = Lsode2RuntimeState::new(0.0, &[1.0], 0.1, 1, step_config).unwrap();
        let error_control = Lsode2ErrorController::new(
            Lsode2Tolerance::scalar(1.0e-3, 1.0e-6),
            Lsode2ErrorControlConfig::default(),
        )
        .unwrap();
        Lsode2StepCycle::new_with_method(state, error_control, Lsode2StepMethod::AdamsLike)
    }

    fn make_adams_cycle_with_max_order(
        max_order: usize,
        step_config: Lsode2StepControlConfig,
    ) -> Lsode2StepCycle {
        let state = Lsode2RuntimeState::new(0.0, &[1.0], 0.1, max_order, step_config).unwrap();
        let error_control = Lsode2ErrorController::new(
            Lsode2Tolerance::scalar(1.0e-3, 1.0e-6),
            Lsode2ErrorControlConfig::default(),
        )
        .unwrap();
        Lsode2StepCycle::new_with_method(state, error_control, Lsode2StepMethod::AdamsLike)
    }

    fn make_correction_controller() -> Lsode2CorrectionController {
        Lsode2CorrectionController::new(
            Lsode2Tolerance::scalar(1.0e-3, 1.0e-6),
            Lsode2CorrectionControlConfig::default(),
        )
        .unwrap()
    }

    #[test]
    fn step_cycle_prediction_uses_runtime_state_prediction() {
        let mut cycle = make_cycle();
        cycle.state_mut().set_order(2).unwrap();
        cycle
            .state_mut()
            .nordsieck_mut()
            .set_col(1, &[2.0])
            .unwrap();
        cycle
            .state_mut()
            .nordsieck_mut()
            .set_col(2, &[3.0])
            .unwrap();

        let predicted = cycle.predict().unwrap();

        assert_eq!(predicted.t_trial, 0.1);
        assert_eq!(predicted.h_trial, 0.1);
        assert_eq!(predicted.order, 2);
        assert_eq!(predicted.y_pred, vec![6.0]);
    }

    #[test]
    fn step_cycle_accepts_candidate_with_small_error() {
        let mut cycle = make_cycle();

        let outcome = cycle
            .finish_with_local_error(0.1, &[0.905], &[1.0e-5])
            .unwrap();

        match outcome {
            Lsode2StepCycleOutcome::Accepted {
                error_test,
                accept,
                state,
                ..
            } => {
                assert!(error_test.accepted());
                assert!(accept.h_next >= 0.1);
                assert_eq!(state.accepted_steps, 1);
                assert_eq!(state.rejected_steps, 0);
            }
            other => panic!("expected accepted step, got {other:?}"),
        }
        assert_eq!(cycle.state().t(), 0.1);
        assert_eq!(cycle.state().y(), &[0.905]);
    }

    #[test]
    fn step_cycle_rejects_candidate_with_large_error() {
        let mut cycle = make_cycle();
        cycle.state_mut().set_order(2).unwrap();

        let outcome = cycle
            .finish_with_local_error(0.1, &[0.905], &[1.0e-1])
            .unwrap();

        match outcome {
            Lsode2StepCycleOutcome::Rejected {
                error_test,
                retry,
                state,
            } => {
                assert_eq!(error_test.action, Lsode2ErrorTestAction::Reject);
                assert!(retry.h_new <= 0.1);
                assert_eq!(state.accepted_steps, 0);
                assert_eq!(state.rejected_steps, 1);
            }
            other => panic!("expected rejected step, got {other:?}"),
        }
        assert_eq!(cycle.state().t(), 0.0);
        assert_eq!(cycle.state().y(), &[1.0]);
    }

    #[test]
    fn step_cycle_error_test_failure_can_choose_lower_order_retry() {
        let mut cycle = make_cycle();
        cycle.state_mut().set_order(2).unwrap();
        cycle
            .state_mut()
            .nordsieck_mut()
            .set_col(2, &[1.0e-8])
            .unwrap();

        let outcome = cycle
            .finish_with_local_error(0.1, &[0.905], &[1.0e-1])
            .unwrap();

        match outcome {
            Lsode2StepCycleOutcome::Rejected { retry, state, .. } => {
                assert_eq!(retry.order_new, 1);
                assert!(retry.h_new <= 0.1);
                assert_eq!(state.rejected_steps, 1);
                assert_eq!(cycle.kflag(), Lsode2Kflag::ErrorTestFailure);
                assert_eq!(cycle.iret(), Lsode2Iret::RetryAfterErrorTestFailure);
                assert_eq!(cycle.redo_stage(), Lsode2RedoStage::ErrorTestRetry);
            }
            other => panic!("expected rejected step, got {other:?}"),
        }
    }

    #[test]
    fn step_cycle_repeated_error_test_failures_reset_to_first_order() {
        let mut cycle = make_cycle();
        cycle.state_mut().set_step_size(1.0).unwrap();
        cycle.state_mut().set_order(3).unwrap();
        cycle
            .state_mut()
            .nordsieck_mut()
            .set_col(1, &[2.0])
            .unwrap();
        cycle
            .state_mut()
            .nordsieck_mut()
            .set_col(2, &[3.0])
            .unwrap();
        cycle
            .state_mut()
            .nordsieck_mut()
            .set_col(3, &[4.0])
            .unwrap();

        cycle
            .state_mut()
            .reject_after_error_test_with_hint(0.5, 3)
            .unwrap();
        cycle
            .state_mut()
            .reject_after_error_test_with_hint(0.25, 3)
            .unwrap();

        let outcome = cycle
            .finish_with_local_error(0.1, &[0.905], &[1.0e-1])
            .unwrap();

        match outcome {
            Lsode2StepCycleOutcome::Rejected { retry, state, .. } => {
                assert_eq!(retry.order_new, 1);
                assert!((retry.h_new - 0.025).abs() < 1.0e-12);
                assert_eq!(state.order, 1);
                assert_eq!(state.rejected_steps, 3);
                assert_eq!(state.jacobian_refresh_requests, 1);
                assert!(cycle.jacobian_update_request().is_requested());
                assert_eq!(cycle.iret(), Lsode2Iret::RestartWithDerivativeRefresh);
                assert_eq!(cycle.ipup(), Lsode2Ipup::NeedsJacobianUpdate);
                // DSTODA label-640 reset path is an explicit redo-class branch.
                assert_eq!(cycle.redo_stage(), Lsode2RedoStage::RepeatedErrorReset);
                assert_eq!(cycle.iredo().code(), 3);
            }
            other => panic!("expected rejected step, got {other:?}"),
        }
        // With DSTODA-style reject rescaling enabled on each retry, YH(:,2)
        // has already been scaled on the two preloaded reject steps (1.0 -> 0.5 -> 0.25).
        // The label-640 reset applies one more RH=0.1 factor: 2.0 * 0.25 * 0.1 = 0.05.
        assert_eq!(cycle.state().nordsieck().col(1).unwrap(), &[0.05]);
        assert_eq!(cycle.state().nordsieck().col(2).unwrap(), &[0.0]);
        assert_eq!(cycle.state().nordsieck().col(3).unwrap(), &[0.0]);
    }

    #[test]
    fn step_cycle_repeated_error_test_failures_reach_terminal_kflag() {
        let mut cycle = make_cycle_with_step_config(Lsode2StepControlConfig {
            max_error_test_failures: 3,
            ..Lsode2StepControlConfig::default()
        });
        cycle.state_mut().set_step_size(1.0).unwrap();
        cycle.state_mut().set_order(3).unwrap();

        cycle
            .state_mut()
            .reject_after_error_test_with_hint(0.5, 3)
            .unwrap();
        cycle
            .state_mut()
            .reject_after_error_test_with_hint(0.25, 3)
            .unwrap();

        let h_before = cycle.state().h();
        let outcome = cycle
            .finish_with_local_error(0.1, &[0.905], &[1.0e-1])
            .unwrap();

        match outcome {
            Lsode2StepCycleOutcome::Rejected { retry, state, .. } => {
                assert_eq!(
                    retry.action,
                    Lsode2RetryAction::FailRepeatedErrorTestFailures
                );
                let expected_h = h_before
                    * (cycle.state().step_control_config().h_min / h_before.abs()).max(0.1);
                assert_eq!(retry.h_new, expected_h);
                assert_eq!(state.h, expected_h);
                assert_eq!(state.consecutive_error_test_failures, 3);
                assert_eq!(cycle.kflag(), Lsode2Kflag::RepeatedErrorTestFailure);
                assert_eq!(cycle.redo_stage(), Lsode2RedoStage::None);
            }
            other => panic!("expected terminal repeated error-test failure, got {other:?}"),
        }
    }

    #[test]
    fn step_cycle_error_test_reject_hits_dstoda_hmin_guard_before_rh_choreography() {
        let mut cycle = make_cycle_with_step_config(Lsode2StepControlConfig {
            h_min: 1.0,
            ..Lsode2StepControlConfig::default()
        });
        cycle.state_mut().set_step_size(1.000009).unwrap();

        let h_before = cycle.state().h();
        let outcome = cycle
            .finish_with_local_error(0.1, &[0.905], &[1.0e-1])
            .unwrap();

        match outcome {
            Lsode2StepCycleOutcome::Rejected { retry, .. } => {
                assert_eq!(retry.action, Lsode2RetryAction::FailStepSizeUnderflow);
                assert_eq!(retry.h_new, h_before);
                assert_eq!(cycle.kflag(), Lsode2Kflag::ErrorTestFailure);
            }
            other => panic!("expected HMIN-guard rejection, got {other:?}"),
        }
    }

    #[test]
    fn step_cycle_error_test_reject_above_hmin_guard_uses_regular_retry_path() {
        let mut cycle = make_cycle_with_step_config(Lsode2StepControlConfig {
            h_min: 1.0e-6,
            ..Lsode2StepControlConfig::default()
        });
        cycle.state_mut().set_step_size(1.2).unwrap();

        let h_before = cycle.state().h();
        let outcome = cycle
            .finish_with_local_error(0.1, &[0.905], &[1.0e-1])
            .unwrap();

        match outcome {
            Lsode2StepCycleOutcome::Rejected { retry, .. } => {
                assert_eq!(retry.action, Lsode2RetryAction::Retry);
                assert!(retry.h_new < h_before);
                assert_eq!(cycle.kflag(), Lsode2Kflag::ErrorTestFailure);
            }
            other => panic!("expected regular error-test retry, got {other:?}"),
        }
    }

    #[test]
    fn step_cycle_reports_nonlinear_failure_through_runtime_state() {
        let mut cycle = make_cycle();
        let retry = cycle.reject_after_nonlinear_failure().unwrap();
        let telemetry = cycle.switch_telemetry(Some(42.0));

        assert!(retry.h_new < 0.1);
        assert_eq!(cycle.kflag(), Lsode2Kflag::ConvergenceFailure);
        assert_eq!(cycle.jacobian_currency(), Lsode2JacobianCurrency::Stale);
        assert!(cycle.jacobian_update_request().is_requested());
        assert_eq!(cycle.ipup(), Lsode2Ipup::NeedsJacobianUpdate);
        assert_eq!(cycle.icf(), Lsode2Icf::RefreshDidNotRecover);
        assert_eq!(cycle.redo_stage(), Lsode2RedoStage::CorrectorFailureRetry);
        assert_eq!(telemetry.stiffness_ratio, Some(42.0));
        assert_eq!(telemetry.convergence_failures, 1);
        assert_eq!(telemetry.rejected_steps, 1);
    }

    #[test]
    fn step_cycle_can_retry_stale_jacobian_failure_without_step_shrink() {
        let mut cycle = make_cycle();
        cycle.state_mut().set_order(2).unwrap();

        let retry = cycle
            .retry_after_stale_jacobian_nonlinear_failure()
            .unwrap();
        let telemetry = cycle.switch_telemetry(Some(42.0));

        assert_eq!(retry.h_new, 0.1);
        assert_eq!(retry.order_new, 2);
        assert_eq!(retry.shrink_factor, 1.0);
        assert_eq!(cycle.kflag(), Lsode2Kflag::ConvergenceFailure);
        assert_eq!(cycle.jacobian_currency(), Lsode2JacobianCurrency::Stale);
        assert!(cycle.jacobian_update_request().is_requested());
        assert_eq!(cycle.ipup(), Lsode2Ipup::NeedsJacobianUpdate);
        assert_eq!(cycle.icf(), Lsode2Icf::RefreshRequested);
        assert_eq!(
            cycle.redo_stage(),
            Lsode2RedoStage::CorrectorRefreshSameStep
        );
        assert_eq!(cycle.state().snapshot().rejected_steps, 0);
        assert_eq!(telemetry.stiffness_ratio, Some(42.0));
        assert_eq!(telemetry.convergence_failures, 0);
        assert_eq!(telemetry.rejected_steps, 0);
    }

    #[test]
    fn step_cycle_bdf_updates_switch_cap_hint_from_dstoda_order_selection() {
        let mut cycle = make_cycle();
        let _ = cycle.select_post_accept_order(&[0.905], 1.0, None).unwrap();

        let telemetry = cycle.switch_telemetry(None);
        assert!(telemetry.bdf_step_size_cap_estimate.is_some());
        assert!(
            telemetry
                .bdf_step_size_cap_estimate
                .is_some_and(|value| value.is_finite() && value > 0.0)
        );
        assert!(telemetry.adams_step_size_cap_estimate.is_none());
    }

    #[test]
    fn step_cycle_adams_updates_switch_hints_from_dstoda_order_selection() {
        let mut cycle = make_adams_cycle_with_step_config(Lsode2StepControlConfig::default());
        let _ = cycle.select_post_accept_order(&[0.905], 1.0, None).unwrap();

        let telemetry = cycle.switch_telemetry(None);
        assert!(telemetry.adams_step_size_cap_estimate.is_some());
        assert!(
            telemetry
                .adams_step_size_cap_estimate
                .is_some_and(|value| value.is_finite() && value > 0.0)
        );
        assert!(
            telemetry
                .stiffness_ratio
                .is_some_and(|value| value.is_finite() && value > 0.0)
        );
    }

    #[test]
    fn stale_jacobian_refresh_retry_is_one_shot_then_retracts_on_next_failure() {
        let mut cycle = make_cycle();
        cycle.state_mut().set_order(2).unwrap();

        let first = cycle
            .retry_after_stale_jacobian_nonlinear_failure()
            .unwrap();
        assert_eq!(first.h_new, 0.1);
        assert_eq!(first.shrink_factor, 1.0);
        assert_eq!(cycle.icf(), Lsode2Icf::RefreshRequested);
        assert_eq!(
            cycle.redo_stage(),
            Lsode2RedoStage::CorrectorRefreshSameStep
        );

        // DSTODA 410->430 parity: if refresh retry did not lead to success,
        // next convergence failure retracts/shrinks the step.
        cycle.mark_jacobian_stale();
        let second = cycle
            .retry_after_stale_jacobian_nonlinear_failure()
            .unwrap();

        assert!(second.h_new < first.h_new);
        assert_eq!(second.shrink_factor, 0.25);
        assert_eq!(cycle.icf(), Lsode2Icf::RefreshDidNotRecover);
        assert_eq!(cycle.redo_stage(), Lsode2RedoStage::CorrectorFailureRetry);
        assert_eq!(cycle.state().snapshot().convergence_failures, 1);
    }

    #[test]
    fn convergence_failure_counter_ignores_icf1_refresh_retry_and_trips_on_retract_path() {
        let mut cycle = make_cycle_with_step_config(Lsode2StepControlConfig {
            max_convergence_failures: 2,
            ..Lsode2StepControlConfig::default()
        });
        cycle.state_mut().set_order(2).unwrap();
        cycle.set_iteration_mode(Lsode2IterationMode::JacobianBased);

        // First failure: stale-Jacobian one-shot refresh path (ICF=1),
        // should not increment NCF-like counters.
        let first = cycle
            .retry_after_stale_jacobian_nonlinear_failure()
            .unwrap();
        assert_eq!(first.action, Lsode2RetryAction::RetryWithJacobianRefresh);
        assert_eq!(cycle.icf(), Lsode2Icf::RefreshRequested);
        assert_eq!(cycle.state().snapshot().convergence_failures, 0);
        assert_eq!(cycle.state().snapshot().consecutive_convergence_failures, 0);
        assert_eq!(cycle.state().snapshot().rejected_steps, 0);

        // Second failure after ICF=1: DSTODA 410 -> 430 transition
        // enters retract/shrink path and starts NCF counting.
        cycle.mark_jacobian_stale();
        let second = cycle
            .retry_after_stale_jacobian_nonlinear_failure()
            .unwrap();
        assert_eq!(second.action, Lsode2RetryAction::RetryWithJacobianRefresh);
        assert_eq!(cycle.icf(), Lsode2Icf::RefreshDidNotRecover);
        assert_eq!(cycle.redo_stage(), Lsode2RedoStage::CorrectorFailureRetry);
        assert_eq!(cycle.state().snapshot().convergence_failures, 1);
        assert_eq!(cycle.state().snapshot().consecutive_convergence_failures, 1);
        assert_eq!(cycle.state().snapshot().rejected_steps, 1);

        // Next attempt consumes IPUP by rebuilding Jacobian (JCUR <- 1),
        // so the next convergence failure must remain on retract/count path.
        cycle.mark_jacobian_current();

        // Third failure on retract/shrink path reaches MXNCF-like terminal.
        let third = cycle.reject_after_nonlinear_failure().unwrap();
        assert_eq!(
            third.action,
            Lsode2RetryAction::FailRepeatedConvergenceFailures
        );
        assert_eq!(cycle.kflag(), Lsode2Kflag::RepeatedConvergenceFailure);
        assert_eq!(cycle.state().snapshot().convergence_failures, 2);
        assert_eq!(cycle.state().snapshot().consecutive_convergence_failures, 2);
    }

    #[test]
    fn convergence_underflow_path_maps_to_terminal_kflag_minus_two_class() {
        let mut cycle = make_cycle_with_step_config(Lsode2StepControlConfig {
            h_min: 0.09,
            max_convergence_failures: 10,
            ..Lsode2StepControlConfig::default()
        });
        cycle.state_mut().set_order(2).unwrap();
        cycle.mark_jacobian_current();

        let retry = cycle.reject_after_nonlinear_failure().unwrap();

        assert_eq!(retry.action, Lsode2RetryAction::FailStepSizeUnderflow);
        assert_eq!(cycle.kflag(), Lsode2Kflag::RepeatedConvergenceFailure);
        assert_eq!(cycle.kflag_code(), -2);
        // DSTODA label-430/670 parity:
        // terminal HMIN branch exits before `IPUP = MITER`.
        assert_eq!(cycle.ipup(), Lsode2Ipup::UpToDate);
        assert_eq!(cycle.ipup_trigger(), Lsode2IpupTrigger::None);
        assert_eq!(
            cycle.jacobian_update_request(),
            Lsode2JacobianUpdateRequest::None
        );
        assert_eq!(cycle.icf(), Lsode2Icf::RefreshDidNotRecover);
        assert_eq!(cycle.iredo(), Lsode2Iredo::CorrectorFailureRetry);
        assert_eq!(cycle.iredo().code(), 1);
        assert_eq!(cycle.iret(), Lsode2Iret::NormalFlow);
        assert_eq!(cycle.redo_stage(), Lsode2RedoStage::CorrectorFailureRetry);
    }

    #[test]
    fn terminal_convergence_branches_preserve_dstoda_flag_choreography() {
        // Path A (MXNCF-like terminal):
        // 1) one-shot stale-J retry (ICF=1),
        // 2) retract path (ICF=2),
        // 3) repeated convergence terminal (KFLAG=-2 class).
        let mut cycle_mxncf = make_cycle_with_step_config(Lsode2StepControlConfig {
            max_convergence_failures: 2,
            ..Lsode2StepControlConfig::default()
        });
        cycle_mxncf.state_mut().set_order(2).unwrap();
        cycle_mxncf.set_iteration_mode(Lsode2IterationMode::JacobianBased);

        let first = cycle_mxncf
            .retry_after_stale_jacobian_nonlinear_failure()
            .unwrap();
        assert_eq!(first.action, Lsode2RetryAction::RetryWithJacobianRefresh);
        assert_eq!(cycle_mxncf.kflag(), Lsode2Kflag::ConvergenceFailure);
        assert_eq!(cycle_mxncf.kflag_code(), -2);
        assert_eq!(cycle_mxncf.icf(), Lsode2Icf::RefreshRequested);
        assert_eq!(cycle_mxncf.ipup(), Lsode2Ipup::NeedsJacobianUpdate);
        assert_eq!(cycle_mxncf.ipup_trigger(), Lsode2IpupTrigger::FailurePath);
        assert_eq!(
            cycle_mxncf.jacobian_update_request(),
            Lsode2JacobianUpdateRequest::Requested
        );
        assert_eq!(cycle_mxncf.iret(), Lsode2Iret::NormalFlow);
        assert_eq!(
            cycle_mxncf.redo_stage(),
            Lsode2RedoStage::CorrectorRefreshSameStep
        );
        assert_eq!(cycle_mxncf.iredo(), Lsode2Iredo::CorrectorRefreshSameStep);
        assert_eq!(cycle_mxncf.iredo().code(), 1);

        cycle_mxncf.mark_jacobian_stale();
        let second = cycle_mxncf
            .retry_after_stale_jacobian_nonlinear_failure()
            .unwrap();
        assert_eq!(second.action, Lsode2RetryAction::RetryWithJacobianRefresh);
        assert_eq!(cycle_mxncf.kflag(), Lsode2Kflag::ConvergenceFailure);
        assert_eq!(cycle_mxncf.kflag_code(), -2);
        assert_eq!(cycle_mxncf.icf(), Lsode2Icf::RefreshDidNotRecover);
        assert_eq!(cycle_mxncf.ipup(), Lsode2Ipup::NeedsJacobianUpdate);
        assert_eq!(cycle_mxncf.ipup_trigger(), Lsode2IpupTrigger::FailurePath);
        assert_eq!(
            cycle_mxncf.jacobian_update_request(),
            Lsode2JacobianUpdateRequest::Requested
        );
        assert_eq!(cycle_mxncf.iret(), Lsode2Iret::NormalFlow);
        assert_eq!(
            cycle_mxncf.redo_stage(),
            Lsode2RedoStage::CorrectorFailureRetry
        );
        assert_eq!(cycle_mxncf.iredo(), Lsode2Iredo::CorrectorFailureRetry);
        assert_eq!(cycle_mxncf.iredo().code(), 1);

        cycle_mxncf.mark_jacobian_current();
        let third = cycle_mxncf.reject_after_nonlinear_failure().unwrap();
        assert_eq!(
            third.action,
            Lsode2RetryAction::FailRepeatedConvergenceFailures
        );
        assert_eq!(cycle_mxncf.kflag(), Lsode2Kflag::RepeatedConvergenceFailure);
        assert_eq!(cycle_mxncf.kflag_code(), -2);
        // DSTODA label-430 parity: terminal MXNCF branch exits before a new
        // `IPUP = MITER` request is issued.
        assert_eq!(cycle_mxncf.ipup(), Lsode2Ipup::UpToDate);
        assert_eq!(cycle_mxncf.ipup_trigger(), Lsode2IpupTrigger::None);
        assert_eq!(
            cycle_mxncf.jacobian_update_request(),
            Lsode2JacobianUpdateRequest::None
        );
        assert_eq!(cycle_mxncf.icf(), Lsode2Icf::RefreshDidNotRecover);
        assert_eq!(cycle_mxncf.iret(), Lsode2Iret::NormalFlow);
        assert_eq!(
            cycle_mxncf.redo_stage(),
            Lsode2RedoStage::CorrectorFailureRetry
        );
        assert_eq!(cycle_mxncf.iredo(), Lsode2Iredo::CorrectorFailureRetry);
        assert_eq!(cycle_mxncf.iredo().code(), 1);

        // Path B (HMIN guard terminal):
        // terminal reason differs, but KFLAG/ICF/IRET/IREDO class must match
        // DSTODA convergence-terminal group.
        let mut cycle_hmin = make_cycle_with_step_config(Lsode2StepControlConfig {
            h_min: 0.09,
            max_convergence_failures: 10,
            ..Lsode2StepControlConfig::default()
        });
        cycle_hmin.state_mut().set_order(2).unwrap();
        cycle_hmin.mark_jacobian_current();

        let terminal = cycle_hmin.reject_after_nonlinear_failure().unwrap();
        assert_eq!(terminal.action, Lsode2RetryAction::FailStepSizeUnderflow);
        assert_eq!(cycle_hmin.kflag(), Lsode2Kflag::RepeatedConvergenceFailure);
        assert_eq!(cycle_hmin.kflag_code(), -2);
        assert_eq!(cycle_hmin.icf(), Lsode2Icf::RefreshDidNotRecover);
        assert_eq!(cycle_hmin.iret(), Lsode2Iret::NormalFlow);
        assert_eq!(
            cycle_hmin.redo_stage(),
            Lsode2RedoStage::CorrectorFailureRetry
        );
        assert_eq!(cycle_hmin.iredo(), Lsode2Iredo::CorrectorFailureRetry);
        assert_eq!(cycle_hmin.iredo().code(), 1);
    }

    #[test]
    fn functional_iteration_stale_jacobian_failure_retracts_step_like_odepack_miter_zero() {
        let mut cycle = make_cycle();
        cycle.state_mut().set_order(2).unwrap();
        cycle.mark_jacobian_stale();
        cycle.set_iteration_mode(Lsode2IterationMode::Functional);

        let retry = cycle
            .retry_after_stale_jacobian_nonlinear_failure()
            .unwrap();

        assert!(retry.h_new < 0.1);
        assert_eq!(retry.order_new, 1);
        assert_eq!(cycle.kflag(), Lsode2Kflag::ConvergenceFailure);
        assert_eq!(cycle.icf(), Lsode2Icf::RefreshDidNotRecover);
        assert_eq!(cycle.redo_stage(), Lsode2RedoStage::CorrectorFailureRetry);
        assert_eq!(cycle.state().snapshot().rejected_steps, 1);
    }

    #[test]
    fn step_cycle_adams_like_reject_path_stays_first_order() {
        let mut cycle = make_adams_cycle_with_step_config(Lsode2StepControlConfig::default());
        assert_eq!(cycle.state().order(), 1);

        let outcome = cycle
            .finish_with_local_error(0.1, &[0.905], &[1.0e-1])
            .unwrap();

        match outcome {
            Lsode2StepCycleOutcome::Rejected { retry, state, .. } => {
                assert_eq!(retry.order_new, 1);
                assert_eq!(state.order, 1);
                assert!(retry.h_new <= 0.1);
                assert_eq!(cycle.kflag(), Lsode2Kflag::ErrorTestFailure);
            }
            other => panic!("expected rejected Adams-like step, got {other:?}"),
        }
    }

    #[test]
    fn tesco1_is_resolved_by_method_tables() {
        let bdf_q2 = tesco1_for_method(Lsode2StepMethod::BdfLike, 2);
        let adams_q2 = tesco1_for_method(Lsode2StepMethod::AdamsLike, 2);
        assert!(bdf_q2.is_finite());
        assert!(adams_q2.is_finite());
        assert_eq!(adams_q2, 1.0);
    }

    #[test]
    fn tesco2_is_resolved_by_method_tables() {
        let bdf_q3 = tesco2_for_method(Lsode2StepMethod::BdfLike, 3);
        let adams_q3 = tesco2_for_method(Lsode2StepMethod::AdamsLike, 3);
        assert!(bdf_q3.is_finite() && bdf_q3 > 0.0);
        assert!(adams_q3.is_finite() && adams_q3 > 0.0);
        assert!(
            (bdf_q3 - adams_q3).abs() > 1.0e-12,
            "BDF and Adams should not share identical TESCO(2) at q=3"
        );
    }

    #[test]
    fn step_cycle_can_finish_from_converged_correction() {
        let mut cycle = make_cycle();
        let correction = make_correction_controller();

        let outcome = cycle
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

        match outcome {
            Lsode2StepCycleOutcome::Accepted { state, .. } => {
                assert_eq!(state.accepted_steps, 1);
                assert_eq!(state.rejected_steps, 0);
            }
            other => panic!("expected accepted step from converged correction, got {other:?}"),
        }
    }

    #[test]
    fn step_cycle_adams_like_can_finish_from_converged_correction() {
        let mut cycle = make_adams_cycle_with_step_config(Lsode2StepControlConfig::default());
        let correction = make_correction_controller();

        let first = cycle
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
        match first {
            Lsode2StepCycleOutcome::NonlinearContinue { assessment } => {
                assert_eq!(assessment.status, Lsode2CorrectionStatus::Continue);
            }
            other => panic!("expected forced Adams second-iteration continue, got {other:?}"),
        }

        let outcome = cycle
            .finish_with_correction(
                0.1,
                &[0.905],
                &[1.0e-6],
                &[1.1e-5],
                &correction,
                Some(1.0e-5),
                Some(0.7),
                2,
            )
            .unwrap();

        match outcome {
            Lsode2StepCycleOutcome::Accepted { state, accept, .. } => {
                assert_eq!(state.accepted_steps, 1);
                assert_eq!(state.rejected_steps, 0);
                assert_eq!(accept.order_new, 1);
                assert_eq!(state.order, 1);
            }
            other => {
                panic!("expected accepted Adams-like converged-correction step, got {other:?}")
            }
        }
    }

    #[test]
    fn step_cycle_adams_like_roundoff_correction_can_converge_on_first_iteration() {
        let mut cycle = make_adams_cycle_with_step_config(Lsode2StepControlConfig::default());
        let correction = make_correction_controller();

        let outcome = cycle
            .finish_with_correction(
                0.1,
                &[0.905],
                &[1.0e-30],
                &[1.0e-30],
                &correction,
                None,
                None,
                1,
            )
            .unwrap();

        match outcome {
            Lsode2StepCycleOutcome::Accepted { state, .. } => {
                assert_eq!(state.accepted_steps, 1);
            }
            other => panic!(
                "expected immediate Adams acceptance for roundoff-level correction, got {other:?}"
            ),
        }
    }

    #[test]
    fn step_cycle_adams_like_order_selection_without_staged_delta_disables_q_plus_1_candidate() {
        let mut cycle = make_adams_cycle_with_max_order(
            3,
            Lsode2StepControlConfig {
                raise_order_after_accepts: 1,
                ..Lsode2StepControlConfig::default()
            },
        );
        cycle
            .state_mut()
            .accept_step(0.1, &[0.9], 1.0, 0.1, 1)
            .unwrap();
        cycle
            .state_mut()
            .accept_step(0.2, &[0.81], 1.0, 0.1, 1)
            .unwrap();
        cycle
            .state_mut()
            .accept_step(0.3, &[0.729], 1.0, 0.1, 1)
            .unwrap();
        cycle
            .state_mut()
            .accept_step(0.4, &[0.6561], 1.0, 0.1, 1)
            .unwrap();

        let decision = cycle
            .select_post_accept_order(&[0.59049], 1.0, None)
            .unwrap();
        assert_eq!(cycle.state().order(), 1);
        assert_eq!(cycle.state().next_accept_order_cap(), 3);
        assert_eq!(decision.lower_factor, 0.0);
        assert!(decision.current_factor.is_finite());
        assert_eq!(decision.higher_factor, 0.0);
        assert_eq!(decision.candidate, Lsode2OrderCandidate::Current);
        assert_eq!(decision.order_new, 1);
    }

    #[test]
    fn dstoda_adams_stability_cap_can_block_order_increase() {
        let decision = select_dstoda_order_and_growth(
            Lsode2StepMethod::AdamsLike,
            1.0,
            100.0,
            2,
            3,
            1.0e-12,
            1.0e-12,
            Some(1.0e-12),
            false,
            true,
        );

        assert_eq!(decision.order_new, 2);
        assert_eq!(decision.candidate, Lsode2OrderCandidate::Current);
        assert!(decision.suggested_growth <= 1.0);
    }

    #[test]
    fn dstoda_adams_stability_limited_path_can_bypass_ten_percent_gate() {
        let decision = select_dstoda_order_and_growth(
            Lsode2StepMethod::AdamsLike,
            1.0,
            1.0e6,
            2,
            3,
            1.0e-3,
            1.0e-3,
            Some(1.0),
            false,
            true,
        );

        assert!(
            decision.suggested_growth < 1.1,
            "setup should force a sub-10% RH under stability cap"
        );
        assert!(
            decision.order_new == 1 || decision.order_new == 2,
            "stability-limited bypass should allow keeping/lowering order without hard forcing RH=1"
        );
    }

    #[test]
    fn step_cycle_adams_like_order_selection_uses_staged_correction_delta_path() {
        let mut cycle = make_adams_cycle_with_max_order(
            3,
            Lsode2StepControlConfig {
                raise_order_after_accepts: 1,
                ..Lsode2StepControlConfig::default()
            },
        );
        let correction = make_correction_controller();

        let first = cycle
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
        assert!(matches!(
            first,
            Lsode2StepCycleOutcome::NonlinearContinue { .. }
        ));
        let second = cycle
            .finish_with_correction(
                0.1,
                &[0.905],
                &[1.0e-6],
                &[1.1e-5],
                &correction,
                Some(1.0e-5),
                Some(0.7),
                2,
            )
            .unwrap();
        assert!(matches!(second, Lsode2StepCycleOutcome::Accepted { .. }));
        assert_eq!(
            cycle
                .state()
                .staged_higher_order_correction()
                .map(|v| v.len()),
            Some(1)
        );

        let decision = cycle
            .select_post_accept_order(&[0.81], 1.0, Some(&[2.0e-5]))
            .unwrap();
        assert_eq!(cycle.state().order(), 1);
        assert!(cycle.state().next_accept_order_cap() >= 2);
        assert_eq!(decision.lower_factor, 0.0);
        assert!(decision.current_factor.is_finite());
        assert!(decision.higher_factor.is_finite());
        assert!(decision.higher_factor > 0.0);
    }

    #[test]
    fn step_cycle_adams_like_clears_pdest_after_order_selection() {
        let mut cycle = make_adams_cycle_with_step_config(Lsode2StepControlConfig::default());
        cycle.record_adams_lipschitz_estimate_from_assessment(&Lsode2CorrectionAssessment {
            order: 1,
            iteration: 2,
            weighted_norm: 0.0,
            accumulated_weighted_norm: 0.0,
            previous_weighted_norm: None,
            previous_rate_max: None,
            convergence_ratio: None,
            convergence_rate_estimate: None,
            rate_max_estimate: None,
            pdest_candidate: Some(5.0),
            convergence_measure: 0.0,
            status: Lsode2CorrectionStatus::Converged,
            local_error: vec![0.0],
            needs_jacobian_refresh: false,
        });
        assert_eq!(cycle.adams_pdest(), 5.0);
        assert_eq!(cycle.adams_pdlast(), 5.0);

        let _ = cycle.select_post_accept_order(&[0.905], 1.0, None).unwrap();

        assert_eq!(
            cycle.adams_pdest(),
            0.0,
            "DSTODA parity: PDEST is step-local and should be cleared after RH selection"
        );
        assert_eq!(
            cycle.adams_pdlast(),
            5.0,
            "PDLAST should keep the most recent nonzero Lipschitz estimate"
        );
    }

    #[test]
    fn step_cycle_adams_like_pdest_does_not_monotonically_accumulate_across_steps() {
        let mut cycle = make_adams_cycle_with_step_config(Lsode2StepControlConfig::default());

        cycle.record_adams_lipschitz_estimate_from_assessment(&Lsode2CorrectionAssessment {
            order: 1,
            iteration: 2,
            weighted_norm: 0.0,
            accumulated_weighted_norm: 0.0,
            previous_weighted_norm: None,
            previous_rate_max: None,
            convergence_ratio: None,
            convergence_rate_estimate: None,
            rate_max_estimate: None,
            pdest_candidate: Some(10.0),
            convergence_measure: 0.0,
            status: Lsode2CorrectionStatus::Converged,
            local_error: vec![0.0],
            needs_jacobian_refresh: false,
        });
        let _ = cycle.select_post_accept_order(&[0.905], 1.0, None).unwrap();
        assert_eq!(cycle.adams_pdlast(), 10.0);
        assert_eq!(cycle.adams_pdest(), 0.0);

        cycle.record_adams_lipschitz_estimate_from_assessment(&Lsode2CorrectionAssessment {
            order: 1,
            iteration: 2,
            weighted_norm: 0.0,
            accumulated_weighted_norm: 0.0,
            previous_weighted_norm: None,
            previous_rate_max: None,
            convergence_ratio: None,
            convergence_rate_estimate: None,
            rate_max_estimate: None,
            pdest_candidate: Some(2.0),
            convergence_measure: 0.0,
            status: Lsode2CorrectionStatus::Converged,
            local_error: vec![0.0],
            needs_jacobian_refresh: false,
        });
        assert_eq!(
            cycle.adams_pdlast(),
            2.0,
            "PDLAST should follow the latest step-local estimate, not historical max"
        );
    }

    #[test]
    fn step_cycle_accept_with_h_change_sets_history_rescale_flags() {
        let mut cycle = make_cycle();
        let mut saw_rescale = false;

        for _ in 0..6 {
            let predicted = cycle.predict().unwrap();
            let outcome = cycle
                .finish_with_local_error(predicted.t_trial, predicted.y_pred.as_slice(), &[1.0e-8])
                .unwrap();
            assert!(matches!(outcome, Lsode2StepCycleOutcome::Accepted { .. }));
            if cycle.iret() == Lsode2Iret::RescaleHistory {
                saw_rescale = true;
                break;
            }
        }

        assert!(
            saw_rescale,
            "expected at least one accepted step that changes H or NQ and sets RescaleHistory"
        );
        assert_eq!(cycle.iret(), Lsode2Iret::RescaleHistory);
        assert_eq!(
            cycle.redo_stage(),
            Lsode2RedoStage::HistoryOrStepSizeChanged
        );
        assert_eq!(cycle.iredo(), Lsode2Iredo::HistoryOrStepSizeChanged);
        assert_eq!(cycle.ipup(), Lsode2Ipup::NeedsJacobianUpdate);
    }

    #[test]
    fn step_cycle_converged_correction_uses_dstoda_style_nordsieck_update() {
        let mut cycle = make_cycle();
        cycle
            .state_mut()
            .nordsieck_mut()
            .set_col(1, &[1.0e-4])
            .unwrap();
        cycle.predict().unwrap();
        let correction = make_correction_controller();

        let outcome = cycle
            .finish_with_correction(
                0.1,
                &[1.00005],
                &[-5.0e-6],
                &[-5.0e-5],
                &correction,
                None,
                None,
                1,
            )
            .unwrap();

        assert!(matches!(outcome, Lsode2StepCycleOutcome::Accepted { .. }));
        assert!((cycle.state().nordsieck().col(0).unwrap()[0] - 1.00005).abs() < 1.0e-12);
        assert!((cycle.state().nordsieck().col(1).unwrap()[0] - 5.0e-5).abs() < 1.0e-12);
        assert_eq!(
            cycle.state().staged_higher_order_correction(),
            Some(&[-5.0e-5][..])
        );
        assert_eq!(cycle.jacobian_currency(), Lsode2JacobianCurrency::Stale);
        assert_eq!(
            cycle.jacobian_update_request(),
            Lsode2JacobianUpdateRequest::None
        );
    }

    #[test]
    fn step_cycle_can_request_more_nonlinear_iterations() {
        let mut cycle = make_cycle();
        let correction = make_correction_controller();

        let outcome = cycle
            .finish_with_correction(
                0.1,
                &[0.905],
                &[1.0e-3],
                &[1.0e-3],
                &correction,
                Some(2.0),
                Some(0.1),
                1,
            )
            .unwrap();

        match outcome {
            Lsode2StepCycleOutcome::NonlinearContinue { assessment } => {
                assert_eq!(assessment.status, Lsode2CorrectionStatus::Continue);
                assert_eq!(cycle.state().t(), 0.0);
            }
            other => panic!("expected nonlinear continue, got {other:?}"),
        }
    }

    #[test]
    fn step_cycle_rejects_after_diverged_correction() {
        let mut cycle = make_cycle();
        let correction = make_correction_controller();

        let outcome = cycle
            .finish_with_correction(
                0.1,
                &[0.905],
                &[1.0e-2],
                &[1.0e-2],
                &correction,
                Some(1.0e-4),
                Some(0.5),
                2,
            )
            .unwrap();

        match outcome {
            Lsode2StepCycleOutcome::NonlinearRejected {
                assessment,
                retry,
                state,
            } => {
                assert_eq!(assessment.status, Lsode2CorrectionStatus::Diverged);
                assert!(assessment.needs_jacobian_refresh);
                assert!(retry.h_new < 0.1);
                assert_eq!(state.convergence_failures, 1);
                assert_eq!(cycle.kflag(), Lsode2Kflag::ConvergenceFailure);
                assert_eq!(cycle.jacobian_currency(), Lsode2JacobianCurrency::Stale);
                assert_eq!(cycle.redo_stage(), Lsode2RedoStage::CorrectorFailureRetry);
            }
            other => panic!("expected nonlinear rejection, got {other:?}"),
        }
    }

    #[test]
    fn step_cycle_diverged_correction_with_stale_jacobian_retries_same_step() {
        let mut cycle = make_cycle();
        cycle.mark_jacobian_stale();
        let correction = make_correction_controller();

        let outcome = cycle
            .finish_with_correction(
                0.1,
                &[0.905],
                &[1.0e-2],
                &[1.0e-2],
                &correction,
                Some(1.0e-4),
                Some(0.5),
                2,
            )
            .unwrap();

        match outcome {
            Lsode2StepCycleOutcome::NonlinearRejected { retry, state, .. } => {
                assert_eq!(retry.h_new, 0.1);
                assert_eq!(retry.shrink_factor, 1.0);
                assert_eq!(state.rejected_steps, 0);
                assert_eq!(state.convergence_failures, 0);
                assert_eq!(cycle.kflag(), Lsode2Kflag::ConvergenceFailure);
                assert!(cycle.jacobian_update_request().is_requested());
                assert_eq!(
                    cycle.redo_stage(),
                    Lsode2RedoStage::CorrectorRefreshSameStep
                );
            }
            other => panic!("expected nonlinear retry, got {other:?}"),
        }
    }

    #[test]
    fn step_cycle_error_test_reject_after_converged_corrector_keeps_jacobian_stale() {
        let mut cycle = make_cycle();
        cycle.mark_jacobian_current();

        let outcome = cycle
            .finish_after_converged_correction(0.1, &[0.905], &[1.0e-5], &[1.0e-1])
            .unwrap();

        match outcome {
            Lsode2StepCycleOutcome::Rejected { .. } => {
                assert_eq!(cycle.kflag(), Lsode2Kflag::ErrorTestFailure);
                assert_eq!(cycle.jacobian_currency(), Lsode2JacobianCurrency::Stale);
                assert_eq!(
                    cycle.jacobian_update_request(),
                    Lsode2JacobianUpdateRequest::None
                );
            }
            other => {
                panic!("expected error-test rejection after converged correction, got {other:?}")
            }
        }
    }

    #[test]
    fn step_cycle_records_repeated_convergence_failure_terminal_kflag() {
        let mut cycle = make_cycle_with_step_config(Lsode2StepControlConfig {
            max_convergence_failures: 2,
            ..Lsode2StepControlConfig::default()
        });
        cycle.set_iteration_mode(Lsode2IterationMode::Functional);

        let first = cycle.reject_after_nonlinear_failure().unwrap();
        let second = cycle.reject_after_nonlinear_failure().unwrap();

        assert_eq!(first.action, Lsode2RetryAction::RetryWithJacobianRefresh);
        assert_eq!(
            second.action,
            Lsode2RetryAction::FailRepeatedConvergenceFailures
        );
        assert_eq!(cycle.kflag(), Lsode2Kflag::RepeatedConvergenceFailure);
        assert_eq!(cycle.redo_stage(), Lsode2RedoStage::CorrectorFailureRetry);
        assert_eq!(cycle.state().snapshot().consecutive_convergence_failures, 2);
    }
}
