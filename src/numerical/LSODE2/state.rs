//! LSODE2 runtime state shared by future predictor/corrector engines.
//!
//! This module owns the core state that a native LSODE loop needs directly:
//! current time/step/order, Nordsieck prediction buffers, raw `y` history, and
//! LSODE-style accept/reject bookkeeping.

use super::adams_engine::{Lsode2AdamsDcfodeError, Lsode2AdamsDcfodeTables};
use super::algorithm::Lsode2SwitchTelemetry;
use super::dcfode::{Lsode2BdfDcfodeTables, Lsode2DcfodeError};
use super::history::{
    Lsode2HistoryError, Lsode2NordsieckHistory, Lsode2YHistory, backward_differences_to_nordsieck,
    reconcile_first_nordsieck_derivative,
};
use super::step_control::{
    Lsode2AcceptDecision, Lsode2RetryAction, Lsode2RetryDecision, Lsode2StepControlConfig,
    Lsode2StepControlError, Lsode2StepControlSnapshot, Lsode2StepController,
};

#[derive(Debug, Clone, PartialEq)]
pub enum Lsode2RuntimeStateError {
    InvalidStepSize(f64),
    Dcfode(Lsode2DcfodeError),
    AdamsDcfode(Lsode2AdamsDcfodeError),
    History(Lsode2HistoryError),
    StepControl(Lsode2StepControlError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lsode2CorrectionAcceptUpdateMode {
    /// Use BDF/DSTODA-style direct `ELCO * ACOR` Nordsieck update when eligible.
    BdfDstodaDirect,
    /// Use Adams/DSTODA-style direct `ELCO * ACOR` Nordsieck update when eligible.
    AdamsDstodaDirect,
    /// Always rebuild Nordsieck state from accepted `y` history.
    HistoryRebuild,
}

impl std::fmt::Display for Lsode2RuntimeStateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidStepSize(value) => {
                write!(
                    f,
                    "LSODE2 runtime state requires finite non-zero h, got {value}"
                )
            }
            Self::Dcfode(err) => write!(f, "{err}"),
            Self::AdamsDcfode(err) => write!(f, "{err}"),
            Self::History(err) => write!(f, "{err}"),
            Self::StepControl(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for Lsode2RuntimeStateError {}

impl From<Lsode2HistoryError> for Lsode2RuntimeStateError {
    fn from(value: Lsode2HistoryError) -> Self {
        Self::History(value)
    }
}

impl From<Lsode2DcfodeError> for Lsode2RuntimeStateError {
    fn from(value: Lsode2DcfodeError) -> Self {
        Self::Dcfode(value)
    }
}

impl From<Lsode2AdamsDcfodeError> for Lsode2RuntimeStateError {
    fn from(value: Lsode2AdamsDcfodeError) -> Self {
        Self::AdamsDcfode(value)
    }
}

impl From<Lsode2StepControlError> for Lsode2RuntimeStateError {
    fn from(value: Lsode2StepControlError) -> Self {
        Self::StepControl(value)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Lsode2RuntimeStateSnapshot {
    pub t: f64,
    pub h: f64,
    pub order: usize,
    pub order_cap: usize,
    pub max_order: usize,
    pub n: usize,
    pub accepted_steps: usize,
    pub rejected_steps: usize,
    pub error_test_failures: usize,
    pub consecutive_error_test_failures: usize,
    pub convergence_failures: usize,
    pub consecutive_convergence_failures: usize,
    pub consecutive_rejections: usize,
    pub consecutive_accepts: usize,
    pub jacobian_refresh_requests: usize,
    pub first_derivative_refresh_requested: bool,
    pub last_failure: Option<&'static str>,
}

#[derive(Debug, Clone)]
pub struct Lsode2RuntimeState {
    t: f64,
    h: f64,
    order: usize,
    max_order: usize,
    y: Vec<f64>,
    y_pred: Vec<f64>,
    nordsieck: Lsode2NordsieckHistory,
    predicted_nordsieck: Lsode2NordsieckHistory,
    staged_higher_order_correction: Option<Vec<f64>>,
    first_derivative_refresh_requested: bool,
    y_history: Lsode2YHistory,
    step_controller: Lsode2StepController,
}

impl Lsode2RuntimeState {
    pub fn new(
        t0: f64,
        y0: &[f64],
        h0: f64,
        max_order: usize,
        step_control_config: Lsode2StepControlConfig,
    ) -> Result<Self, Lsode2RuntimeStateError> {
        if !h0.is_finite() || h0 == 0.0 {
            return Err(Lsode2RuntimeStateError::InvalidStepSize(h0));
        }

        let mut nordsieck = Lsode2NordsieckHistory::new(y0.len(), max_order)?;
        nordsieck.set_col(0, y0)?;
        Ok(Self {
            t: t0,
            h: h0,
            order: 1.min(max_order),
            max_order,
            y: y0.to_vec(),
            y_pred: y0.to_vec(),
            predicted_nordsieck: Lsode2NordsieckHistory::new(y0.len(), max_order)?,
            staged_higher_order_correction: None,
            // ODEPACK/DSTODA parity:
            // before the very first predictor/corrector step, YH(:,2) must be
            // initialized from H0 * F(t0, y0). Native step engine performs this
            // through `refresh_first_derivative_if_requested()`.
            first_derivative_refresh_requested: true,
            y_history: Lsode2YHistory::new(y0, max_order)?,
            step_controller: Lsode2StepController::new(step_control_config)?,
            nordsieck,
        })
    }

    pub fn t(&self) -> f64 {
        self.t
    }

    pub fn h(&self) -> f64 {
        self.h
    }

    pub fn set_step_size(&mut self, h_new: f64) -> Result<(), Lsode2RuntimeStateError> {
        if !h_new.is_finite() || h_new == 0.0 {
            return Err(Lsode2RuntimeStateError::InvalidStepSize(h_new));
        }
        self.h = h_new;
        Ok(())
    }

    pub fn order(&self) -> usize {
        self.order
    }

    pub fn max_order(&self) -> usize {
        self.max_order
    }

    pub fn order_cap(&self) -> usize {
        self.order_cap_from_available_history()
    }

    pub fn next_accept_order_cap(&self) -> usize {
        self.order_cap_from_available_history_after_next_accept()
    }

    pub fn y(&self) -> &[f64] {
        &self.y
    }

    pub fn y_pred(&self) -> &[f64] {
        &self.y_pred
    }

    pub fn nordsieck(&self) -> &Lsode2NordsieckHistory {
        &self.nordsieck
    }

    pub fn nordsieck_mut(&mut self) -> &mut Lsode2NordsieckHistory {
        &mut self.nordsieck
    }

    pub fn predicted_nordsieck(&self) -> &Lsode2NordsieckHistory {
        &self.predicted_nordsieck
    }

    pub fn y_history(&self) -> &Lsode2YHistory {
        &self.y_history
    }

    pub fn staged_higher_order_correction(&self) -> Option<&[f64]> {
        self.staged_higher_order_correction.as_deref()
    }

    pub fn first_derivative_refresh_requested(&self) -> bool {
        self.first_derivative_refresh_requested
    }

    pub fn step_control_snapshot(&self) -> Lsode2StepControlSnapshot {
        self.step_controller.snapshot()
    }

    pub fn step_control_config(&self) -> Lsode2StepControlConfig {
        self.step_controller.config()
    }

    pub fn set_order(&mut self, order: usize) -> Result<(), Lsode2RuntimeStateError> {
        if order > self.max_order {
            return Err(Lsode2HistoryError::InvalidOrder {
                order,
                max_order: self.max_order,
            }
            .into());
        }
        self.order = order;
        Ok(())
    }

    pub fn predict_from_nordsieck(&mut self) -> Result<&[f64], Lsode2RuntimeStateError> {
        self.nordsieck
            .predict_into(&mut self.predicted_nordsieck, self.order)?;
        self.y_pred
            .copy_from_slice(self.predicted_nordsieck.col(0)?);
        Ok(&self.y_pred)
    }

    pub fn accept_step(
        &mut self,
        t_new: f64,
        y_new: &[f64],
        suggested_growth: f64,
        error_norm: f64,
        suggested_order: usize,
    ) -> Result<Lsode2AcceptDecision, Lsode2RuntimeStateError> {
        if y_new.len() != self.y.len() {
            return Err(Lsode2HistoryError::DimensionMismatch {
                expected: self.y.len(),
                actual: y_new.len(),
            }
            .into());
        }

        let order_cap = self.order_cap_from_available_history_after_next_accept();
        if suggested_order == 0 || suggested_order > order_cap {
            return Err(Lsode2HistoryError::InvalidOrder {
                order: suggested_order,
                max_order: order_cap,
            }
            .into());
        }
        let decision = self.step_controller.accept(
            self.h,
            self.order,
            order_cap,
            suggested_growth,
            error_norm,
            suggested_order,
        );
        self.t = t_new;
        self.h = decision.h_next;
        self.order = decision.order_new;
        self.y.copy_from_slice(y_new);
        self.y_history.push_front(y_new)?;
        self.staged_higher_order_correction = None;
        let diffs = self.y_history.backward_differences(self.order)?;
        backward_differences_to_nordsieck(diffs.as_slice(), self.order, &mut self.nordsieck)?;
        self.nordsieck.zero_from(self.order + 1)?;
        Ok(decision)
    }

    pub fn accept_step_with_correction(
        &mut self,
        t_new: f64,
        y_new: &[f64],
        accumulated_correction: &[f64],
        suggested_growth: f64,
        error_norm: f64,
        suggested_order: usize,
    ) -> Result<Lsode2AcceptDecision, Lsode2RuntimeStateError> {
        self.accept_step_with_correction_mode(
            t_new,
            y_new,
            accumulated_correction,
            suggested_growth,
            error_norm,
            suggested_order,
            Lsode2CorrectionAcceptUpdateMode::BdfDstodaDirect,
        )
    }

    pub fn accept_step_with_correction_mode(
        &mut self,
        t_new: f64,
        y_new: &[f64],
        accumulated_correction: &[f64],
        suggested_growth: f64,
        error_norm: f64,
        suggested_order: usize,
        update_mode: Lsode2CorrectionAcceptUpdateMode,
    ) -> Result<Lsode2AcceptDecision, Lsode2RuntimeStateError> {
        if y_new.len() != self.y.len() {
            return Err(Lsode2HistoryError::DimensionMismatch {
                expected: self.y.len(),
                actual: y_new.len(),
            }
            .into());
        }
        if accumulated_correction.len() != self.y.len() {
            return Err(Lsode2HistoryError::DimensionMismatch {
                expected: self.y.len(),
                actual: accumulated_correction.len(),
            }
            .into());
        }

        let order_cap = self.order_cap_from_available_history_after_next_accept();
        if suggested_order == 0 || suggested_order > order_cap {
            return Err(Lsode2HistoryError::InvalidOrder {
                order: suggested_order,
                max_order: order_cap,
            }
            .into());
        }

        let order_before_accept = self.order;
        let h_before_accept = self.h;
        let decision = self.step_controller.accept(
            self.h,
            self.order,
            order_cap,
            suggested_growth,
            error_norm,
            suggested_order,
        );

        self.t = t_new;
        self.h = decision.h_next;
        self.order = decision.order_new;
        self.y.copy_from_slice(y_new);
        self.y_history.push_front(y_new)?;

        let can_apply_direct_dstoda_update = matches!(
            update_mode,
            Lsode2CorrectionAcceptUpdateMode::BdfDstodaDirect
                | Lsode2CorrectionAcceptUpdateMode::AdamsDstodaDirect
        );
        if can_apply_direct_dstoda_update {
            match update_mode {
                Lsode2CorrectionAcceptUpdateMode::BdfDstodaDirect => self
                    .apply_bdf_dstoda_accept_update(order_before_accept, accumulated_correction)?,
                Lsode2CorrectionAcceptUpdateMode::AdamsDstodaDirect => self
                    .apply_adams_dstoda_accept_update(
                        order_before_accept,
                        accumulated_correction,
                    )?,
                Lsode2CorrectionAcceptUpdateMode::HistoryRebuild => unreachable!(),
            }
            self.rescale_nordsieck_after_accept_transition(
                h_before_accept,
                decision.h_next,
                order_before_accept,
                decision.order_new,
            )?;
        } else {
            let diffs = self.y_history.backward_differences(self.order)?;
            backward_differences_to_nordsieck(diffs.as_slice(), self.order, &mut self.nordsieck)?;
            self.nordsieck.zero_from(self.order + 1)?;
        }
        self.update_staged_higher_order_correction(accumulated_correction);

        Ok(decision)
    }

    pub fn reject_after_error_test(
        &mut self,
        error_norm: f64,
    ) -> Result<Lsode2RetryDecision, Lsode2RuntimeStateError> {
        let h_old = self.h;
        let order_old = self.order;
        let decision = self
            .step_controller
            .reject_after_error_test(self.h, self.order, error_norm);
        self.h = decision.h_new;
        self.order = decision.order_new;
        self.rescale_nordsieck_after_accept_transition(h_old, self.h, order_old, self.order)?;
        self.staged_higher_order_correction = None;
        Ok(decision)
    }

    pub fn reject_after_error_test_with_hint(
        &mut self,
        h_new: f64,
        order_new: usize,
    ) -> Result<Lsode2RetryDecision, Lsode2RuntimeStateError> {
        let h_old = self.h;
        let order_old = self.order;
        let decision = self
            .step_controller
            .reject_after_error_test_with_hint(self.h, self.order, h_new, order_new);
        self.h = decision.h_new;
        self.order = decision.order_new;
        self.rescale_nordsieck_after_accept_transition(h_old, self.h, order_old, self.order)?;
        self.staged_higher_order_correction = None;
        Ok(decision)
    }

    pub fn reset_after_repeated_error_failures(
        &mut self,
    ) -> Result<Lsode2RetryDecision, Lsode2RuntimeStateError> {
        let h_old = self.h;
        let rh = (self.step_controller.config().h_min / h_old.abs()).max(0.1);
        let h_new = h_old * rh;
        let decision = self
            .step_controller
            .reject_after_error_test_with_hint(self.h, self.order, h_new, 1);
        if decision.action == Lsode2RetryAction::Retry {
            self.step_controller.record_repeated_error_test_reset();
        }

        self.h = decision.h_new;
        self.order = 1;
        self.staged_higher_order_correction = None;
        self.first_derivative_refresh_requested = decision.action == Lsode2RetryAction::Retry;

        if self.max_order >= 1 {
            for value in self.nordsieck.col_mut(1)? {
                *value *= rh;
            }
        }
        self.nordsieck.zero_from(2)?;

        Ok(decision)
    }

    pub fn reject_after_nonlinear_failure(
        &mut self,
    ) -> Result<Lsode2RetryDecision, Lsode2RuntimeStateError> {
        let h_old = self.h;
        let order_old = self.order;
        let decision = self
            .step_controller
            .reject_after_nonlinear_failure(self.h, self.order);
        self.h = decision.h_new;
        self.order = decision.order_new;
        self.rescale_nordsieck_after_accept_transition(h_old, self.h, order_old, self.order)?;
        // DSTODA label-450 parity:
        // when convergence-failure retract path forces order down to 1, the
        // next attempt should refresh first-derivative history (`YH(:,2)`).
        if decision.action == Lsode2RetryAction::RetryWithJacobianRefresh
            && order_old > 1
            && decision.order_new == 1
        {
            self.first_derivative_refresh_requested = true;
        }
        self.staged_higher_order_correction = None;
        Ok(decision)
    }

    pub fn retry_after_stale_jacobian_nonlinear_failure(
        &mut self,
    ) -> Result<Lsode2RetryDecision, Lsode2RuntimeStateError> {
        Ok(self
            .step_controller
            .retry_after_stale_jacobian_nonlinear_failure(self.h, self.order))
    }

    pub fn reconcile_first_nordsieck_derivative(
        &mut self,
        scaled_derivative: &[f64],
    ) -> Result<(), Lsode2RuntimeStateError> {
        reconcile_first_nordsieck_derivative(scaled_derivative, &mut self.nordsieck)?;
        self.first_derivative_refresh_requested = false;
        Ok(())
    }

    pub fn switch_telemetry(&self, stiffness_ratio: Option<f64>) -> Lsode2SwitchTelemetry {
        self.step_controller.switch_telemetry(stiffness_ratio)
    }

    pub fn snapshot(&self) -> Lsode2RuntimeStateSnapshot {
        let step = self.step_controller.snapshot();
        Lsode2RuntimeStateSnapshot {
            t: self.t,
            h: self.h,
            order: self.order,
            order_cap: self.order_cap_from_available_history(),
            max_order: self.max_order,
            n: self.y.len(),
            accepted_steps: step.accepted_steps,
            rejected_steps: step.rejected_steps,
            error_test_failures: step.error_test_failures,
            consecutive_error_test_failures: step.consecutive_error_test_failures,
            convergence_failures: step.convergence_failures,
            consecutive_convergence_failures: step.consecutive_convergence_failures,
            consecutive_rejections: step.consecutive_rejections,
            consecutive_accepts: step.consecutive_accepts,
            jacobian_refresh_requests: step.jacobian_refresh_requests,
            first_derivative_refresh_requested: self.first_derivative_refresh_requested,
            last_failure: step.last_failure,
        }
    }

    fn order_cap_from_available_history(&self) -> usize {
        let accepted_steps = self.step_controller.snapshot().accepted_steps;
        (accepted_steps + 1).min(self.max_order)
    }

    fn order_cap_from_available_history_after_next_accept(&self) -> usize {
        let accepted_steps = self.step_controller.snapshot().accepted_steps;
        (accepted_steps + 2).min(self.max_order)
    }

    fn apply_bdf_dstoda_accept_update(
        &mut self,
        order: usize,
        accumulated_correction: &[f64],
    ) -> Result<(), Lsode2RuntimeStateError> {
        let coeffs = Lsode2BdfDcfodeTables::default().order(order)?;
        for j in 0..=order {
            let predicted_col = self.predicted_nordsieck.col(j)?.to_vec();
            let target_col = self.nordsieck.col_mut(j)?;
            let elj = coeffs.el[j];
            for i in 0..self.y.len() {
                target_col[i] = predicted_col[i] + elj * accumulated_correction[i];
            }
        }
        self.nordsieck.zero_from(order + 1)?;
        Ok(())
    }

    fn apply_adams_dstoda_accept_update(
        &mut self,
        order: usize,
        accumulated_correction: &[f64],
    ) -> Result<(), Lsode2RuntimeStateError> {
        let coeffs = Lsode2AdamsDcfodeTables::default().order(order)?;
        for j in 0..=order {
            let predicted_col = self.predicted_nordsieck.col(j)?.to_vec();
            let target_col = self.nordsieck.col_mut(j)?;
            let elj = coeffs.el[j + 1];
            for i in 0..self.y.len() {
                target_col[i] = predicted_col[i] + elj * accumulated_correction[i];
            }
        }
        self.nordsieck.zero_from(order + 1)?;
        Ok(())
    }

    fn rescale_nordsieck_after_accept_transition(
        &mut self,
        h_old: f64,
        h_new: f64,
        order_old: usize,
        order_new: usize,
    ) -> Result<(), Lsode2RuntimeStateError> {
        if !(h_old.is_finite() && h_new.is_finite()) || h_old == 0.0 {
            return Ok(());
        }
        let rh = h_new / h_old;
        if rh.is_finite() && rh > 0.0 && (rh - 1.0).abs() > 0.0 {
            let max_scale_order = order_old.min(self.max_order);
            for j in 1..=max_scale_order {
                let scale = rh.powi(j as i32);
                let col = self.nordsieck.col_mut(j)?;
                for value in col.iter_mut() {
                    *value *= scale;
                }
            }
        }
        self.nordsieck.zero_from(order_new + 1)?;
        Ok(())
    }

    fn update_staged_higher_order_correction(&mut self, accumulated_correction: &[f64]) {
        let snapshot = self.step_controller.snapshot();
        let should_stage = snapshot.adjustment_wait == 1 && self.order < self.max_order;
        self.staged_higher_order_correction = should_stage.then(|| accumulated_correction.to_vec());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_state_requests_initial_first_derivative_refresh() {
        let state =
            Lsode2RuntimeState::new(0.0, &[1.0], 0.1, 3, Lsode2StepControlConfig::default())
                .unwrap();
        assert!(state.first_derivative_refresh_requested());
        assert!(state.snapshot().first_derivative_refresh_requested);
    }

    #[test]
    fn runtime_state_predicts_from_nordsieck_columns() {
        let mut state =
            Lsode2RuntimeState::new(0.0, &[1.0], 0.1, 3, Lsode2StepControlConfig::default())
                .unwrap();
        state.set_order(3).unwrap();
        state.nordsieck_mut().set_col(1, &[2.0]).unwrap();
        state.nordsieck_mut().set_col(2, &[3.0]).unwrap();
        state.nordsieck_mut().set_col(3, &[4.0]).unwrap();

        let y_pred = state.predict_from_nordsieck().unwrap();

        assert_eq!(y_pred, &[10.0]);
        assert_eq!(state.predicted_nordsieck().col(1).unwrap(), &[20.0]);
    }

    #[test]
    fn runtime_state_accepts_step_and_updates_history_and_counters() {
        let mut state =
            Lsode2RuntimeState::new(0.0, &[1.0, 2.0], 0.1, 2, Lsode2StepControlConfig::default())
                .unwrap();

        let decision = state.accept_step(0.1, &[1.1, 2.2], 2.0, 0.1, 1).unwrap();

        assert_eq!(decision.h_next, 0.1);
        assert_eq!(decision.order_new, 1);
        assert_eq!(state.t(), 0.1);
        assert_eq!(state.h(), 0.1);
        assert_eq!(state.order(), 1);
        assert_eq!(state.order_cap(), 2);
        assert_eq!(state.y(), &[1.1, 2.2]);
        assert_eq!(state.nordsieck().col(0).unwrap(), &[1.1, 2.2]);
        assert_eq!(state.y_history().block(0).unwrap(), &[1.1, 2.2]);
        assert_eq!(state.y_history().block(1).unwrap(), &[1.0, 2.0]);
        assert_eq!(state.snapshot().accepted_steps, 1);
    }

    #[test]
    fn runtime_state_can_raise_order_after_enough_clean_accepts() {
        let mut state =
            Lsode2RuntimeState::new(0.0, &[1.0], 0.1, 3, Lsode2StepControlConfig::default())
                .unwrap();

        let first = state.accept_step(0.1, &[0.9], 1.0, 0.1, 2).unwrap();
        let second = state.accept_step(0.2, &[0.81], 1.0, 0.1, 2).unwrap();
        let third = state.accept_step(0.3, &[0.729], 1.0, 0.1, 2).unwrap();
        let fourth = state.accept_step(0.4, &[0.6561], 1.0, 0.1, 2).unwrap();

        assert_eq!(first.order_new, 1);
        assert_eq!(second.order_new, 2);
        assert_eq!(third.order_new, 2);
        assert_eq!(fourth.order_new, 2);
        let fifth = state.accept_step(0.5, &[0.59049], 1.0, 0.1, 2).unwrap();
        assert_eq!(fifth.order_new, 2);
        assert_eq!(state.order(), 2);
        assert_eq!(state.snapshot().order_cap, 3);
        assert_eq!(state.snapshot().consecutive_accepts, 5);
    }

    #[test]
    fn runtime_state_rejects_and_feeds_switch_telemetry() {
        let mut state =
            Lsode2RuntimeState::new(0.0, &[1.0], 1.0, 3, Lsode2StepControlConfig::default())
                .unwrap();
        state.set_order(3).unwrap();

        let first = state.reject_after_error_test(16.0).unwrap();
        let second = state.reject_after_nonlinear_failure().unwrap();

        assert_eq!(first.action, Lsode2RetryAction::Retry);
        assert_eq!(second.action, Lsode2RetryAction::RetryWithJacobianRefresh);
        assert_eq!(state.order(), 1);
        assert_eq!(state.snapshot().rejected_steps, 2);
        assert_eq!(state.snapshot().error_test_failures, 1);
        assert_eq!(state.snapshot().consecutive_error_test_failures, 1);
        assert_eq!(state.snapshot().convergence_failures, 1);
        assert_eq!(state.snapshot().consecutive_convergence_failures, 1);
        let telemetry = state.switch_telemetry(Some(10.0));
        assert_eq!(telemetry.stiffness_ratio, Some(10.0));
        assert_eq!(telemetry.rejected_steps, 2);
        assert_eq!(telemetry.convergence_failures, 1);
    }

    #[test]
    fn runtime_state_can_reset_to_first_order_after_repeated_error_failures() {
        let mut state =
            Lsode2RuntimeState::new(0.0, &[1.0], 1.0, 3, Lsode2StepControlConfig::default())
                .unwrap();
        state.set_order(3).unwrap();
        state.nordsieck_mut().set_col(1, &[2.0]).unwrap();
        state.nordsieck_mut().set_col(2, &[3.0]).unwrap();
        state.nordsieck_mut().set_col(3, &[4.0]).unwrap();

        let retry = state.reset_after_repeated_error_failures().unwrap();

        assert_eq!(retry.action, Lsode2RetryAction::Retry);
        assert_eq!(retry.order_new, 1);
        assert_eq!(state.order(), 1);
        assert!((state.h() - 0.1).abs() < 1.0e-12);
        assert_eq!(state.nordsieck().col(1).unwrap(), &[0.2]);
        assert_eq!(state.nordsieck().col(2).unwrap(), &[0.0]);
        assert_eq!(state.nordsieck().col(3).unwrap(), &[0.0]);
        let snapshot = state.step_control_snapshot();
        assert_eq!(snapshot.adjustment_wait, 5);
        assert_eq!(snapshot.jacobian_refresh_requests, 1);
        assert!(state.first_derivative_refresh_requested());
        assert!(state.snapshot().first_derivative_refresh_requested);
        state
            .reconcile_first_nordsieck_derivative(&[0.125])
            .unwrap();
        assert_eq!(state.nordsieck().col(1).unwrap(), &[0.125]);
        assert!(!state.first_derivative_refresh_requested());
    }

    #[test]
    fn runtime_state_stale_jacobian_retry_preserves_step_and_order() {
        let mut state =
            Lsode2RuntimeState::new(0.0, &[1.0], 0.1, 3, Lsode2StepControlConfig::default())
                .unwrap();
        state.set_order(2).unwrap();

        let retry = state
            .retry_after_stale_jacobian_nonlinear_failure()
            .unwrap();

        assert_eq!(retry.action, Lsode2RetryAction::RetryWithJacobianRefresh);
        assert_eq!(retry.h_new, 0.1);
        assert_eq!(retry.order_new, 2);
        assert_eq!(state.h(), 0.1);
        assert_eq!(state.order(), 2);
        assert_eq!(state.snapshot().rejected_steps, 0);
        assert_eq!(state.snapshot().convergence_failures, 0);
        assert_eq!(state.snapshot().consecutive_convergence_failures, 0);
        assert_eq!(state.snapshot().jacobian_refresh_requests, 1);
    }

    #[test]
    fn runtime_state_reports_repeated_nonlinear_convergence_failure_limit() {
        let config = Lsode2StepControlConfig {
            max_convergence_failures: 2,
            ..Lsode2StepControlConfig::default()
        };
        let mut state = Lsode2RuntimeState::new(0.0, &[1.0], 1.0, 3, config).unwrap();

        let first = state.reject_after_nonlinear_failure().unwrap();
        let second = state.reject_after_nonlinear_failure().unwrap();

        assert_eq!(first.action, Lsode2RetryAction::RetryWithJacobianRefresh);
        assert_eq!(
            second.action,
            Lsode2RetryAction::FailRepeatedConvergenceFailures
        );
        assert_eq!(state.snapshot().convergence_failures, 2);
        assert_eq!(state.snapshot().consecutive_convergence_failures, 2);
    }

    #[test]
    fn runtime_state_nonlinear_retract_to_first_order_requests_derivative_refresh() {
        let mut state =
            Lsode2RuntimeState::new(0.0, &[1.0], 0.2, 4, Lsode2StepControlConfig::default())
                .unwrap();
        state.set_order(3).unwrap();
        state
            .nordsieck_mut()
            .set_col(1, &[0.5])
            .expect("nordsieck col 1 should be writable");
        state
            .nordsieck_mut()
            .set_col(2, &[0.25])
            .expect("nordsieck col 2 should be writable");
        state
            .nordsieck_mut()
            .set_col(3, &[0.125])
            .expect("nordsieck col 3 should be writable");

        let retry = state.reject_after_nonlinear_failure().unwrap();

        assert_eq!(retry.action, Lsode2RetryAction::RetryWithJacobianRefresh);
        assert_eq!(retry.order_new, 1);
        assert!(state.first_derivative_refresh_requested());
        assert_eq!(state.order(), 1);
        assert_eq!(state.nordsieck().col(2).unwrap(), &[0.0]);
        assert_eq!(state.nordsieck().col(3).unwrap(), &[0.0]);
    }

    #[test]
    fn runtime_state_reports_repeated_error_test_failure_limit() {
        let config = Lsode2StepControlConfig {
            max_error_test_failures: 2,
            ..Lsode2StepControlConfig::default()
        };
        let mut state = Lsode2RuntimeState::new(0.0, &[1.0], 1.0, 3, config).unwrap();

        let first = state.reject_after_error_test(16.0).unwrap();
        let second = state.reject_after_error_test(16.0).unwrap();

        assert_eq!(first.action, Lsode2RetryAction::Retry);
        assert_eq!(
            second.action,
            Lsode2RetryAction::FailRepeatedErrorTestFailures
        );
        assert_eq!(state.snapshot().error_test_failures, 2);
        assert_eq!(state.snapshot().consecutive_error_test_failures, 2);
    }

    #[test]
    fn runtime_state_rejects_invalid_inputs() {
        assert!(matches!(
            Lsode2RuntimeState::new(0.0, &[1.0], 0.0, 1, Lsode2StepControlConfig::default()),
            Err(Lsode2RuntimeStateError::InvalidStepSize(0.0))
        ));

        let mut state =
            Lsode2RuntimeState::new(0.0, &[1.0], 0.1, 1, Lsode2StepControlConfig::default())
                .unwrap();
        let err = state
            .accept_step(0.1, &[1.0, 2.0], 1.0, 0.1, 1)
            .unwrap_err();
        assert!(matches!(
            err,
            Lsode2RuntimeStateError::History(Lsode2HistoryError::DimensionMismatch {
                expected: 1,
                actual: 2
            })
        ));
    }

    #[test]
    fn runtime_state_does_not_raise_order_after_only_marginal_accepts() {
        let mut state =
            Lsode2RuntimeState::new(0.0, &[1.0], 0.1, 3, Lsode2StepControlConfig::default())
                .unwrap();

        let first = state.accept_step(0.1, &[0.9], 1.0, 0.9, 2).unwrap();
        let second = state.accept_step(0.2, &[0.81], 1.0, 0.8, 2).unwrap();
        let third = state.accept_step(0.3, &[0.729], 1.0, 0.75, 2).unwrap();

        assert_eq!(first.order_new, 1);
        assert_eq!(second.order_new, 2);
        assert_eq!(third.order_new, 2);
        assert_eq!(state.order(), 2);
    }

    #[test]
    fn runtime_state_order_cap_tracks_available_history() {
        let config = Lsode2StepControlConfig {
            raise_order_after_accepts: 1,
            ..Lsode2StepControlConfig::default()
        };
        let mut state = Lsode2RuntimeState::new(0.0, &[1.0], 0.1, 5, config).unwrap();

        assert_eq!(state.order_cap(), 1);

        let first = state.accept_step(0.1, &[0.9], 1.0, 0.1, 2).unwrap();
        let second = state.accept_step(0.2, &[0.81], 1.0, 0.1, 2).unwrap();
        let third = state.accept_step(0.3, &[0.729], 1.0, 0.1, 3).unwrap();

        assert_eq!(first.order_new, 1);
        assert_eq!(second.order_new, 2);
        assert_eq!(third.order_new, 2);
        assert_eq!(state.order_cap(), 4);
        assert_eq!(state.snapshot().order_cap, 4);
    }

    #[test]
    fn runtime_state_rebuilds_nordsieck_from_accepted_history() {
        let config = Lsode2StepControlConfig {
            raise_order_after_accepts: 1,
            ..Lsode2StepControlConfig::default()
        };
        let mut state = Lsode2RuntimeState::new(0.0, &[4.0], 0.1, 3, config).unwrap();

        state.accept_step(0.1, &[9.0], 1.0, 0.1, 2).unwrap();
        state.accept_step(0.2, &[16.0], 1.0, 0.1, 2).unwrap();
        state.accept_step(0.3, &[25.0], 1.0, 0.1, 3).unwrap();
        state.accept_step(0.4, &[36.0], 1.0, 0.1, 3).unwrap();
        state.accept_step(0.5, &[49.0], 1.0, 0.1, 3).unwrap();

        assert_eq!(state.order(), 3);
        assert_eq!(state.nordsieck().col(0).unwrap(), &[49.0]);
        assert_eq!(state.nordsieck().col(1).unwrap(), &[12.0]);
        assert_eq!(state.nordsieck().col(2).unwrap(), &[1.0]);
        assert_eq!(state.nordsieck().col(3).unwrap(), &[0.0]);
    }

    #[test]
    fn runtime_state_rejects_invalid_suggested_order() {
        let mut state =
            Lsode2RuntimeState::new(0.0, &[1.0], 0.1, 3, Lsode2StepControlConfig::default())
                .unwrap();

        let err = state.accept_step(0.1, &[0.9], 1.0, 0.1, 3).unwrap_err();
        assert!(matches!(
            err,
            Lsode2RuntimeStateError::History(Lsode2HistoryError::InvalidOrder {
                order: 3,
                max_order: 2
            })
        ));
    }

    #[test]
    fn runtime_state_can_reconcile_first_nordsieck_derivative_after_accept() {
        let config = Lsode2StepControlConfig {
            raise_order_after_accepts: 1,
            ..Lsode2StepControlConfig::default()
        };
        let mut state = Lsode2RuntimeState::new(0.0, &[4.0], 0.1, 3, config).unwrap();

        state.accept_step(0.1, &[9.0], 1.0, 0.1, 2).unwrap();
        state.accept_step(0.2, &[16.0], 1.0, 0.1, 3).unwrap();
        state.accept_step(0.3, &[25.0], 1.0, 0.1, 3).unwrap();
        state.reconcile_first_nordsieck_derivative(&[7.5]).unwrap();

        assert_eq!(state.nordsieck().col(0).unwrap(), &[25.0]);
        assert_eq!(state.nordsieck().col(1).unwrap(), &[7.5]);
        assert_eq!(state.nordsieck().col(2).unwrap(), &[1.0]);
        assert_eq!(state.nordsieck().col(3).unwrap(), &[0.0]);
    }

    #[test]
    fn runtime_state_can_accept_step_via_dstoda_style_elco_acor_update() {
        let mut state =
            Lsode2RuntimeState::new(0.0, &[1.0], 0.1, 3, Lsode2StepControlConfig::default())
                .unwrap();
        state.nordsieck_mut().set_col(1, &[0.1]).unwrap();
        state.predict_from_nordsieck().unwrap();

        let decision = state
            .accept_step_with_correction(0.1, &[1.05], &[-0.05], 1.0, 0.1, 1)
            .unwrap();

        assert_eq!(decision.order_new, 1);
        assert_eq!(state.t(), 0.1);
        assert_eq!(state.y(), &[1.05]);
        assert_eq!(state.nordsieck().col(0).unwrap(), &[1.05]);
        assert_eq!(state.nordsieck().col(1).unwrap(), &[0.05]);
        assert_eq!(state.staged_higher_order_correction(), Some(&[-0.05][..]));
    }

    #[test]
    fn runtime_state_clears_staged_higher_order_correction_after_reject() {
        let mut state =
            Lsode2RuntimeState::new(0.0, &[1.0], 0.1, 3, Lsode2StepControlConfig::default())
                .unwrap();
        state.nordsieck_mut().set_col(1, &[0.1]).unwrap();
        state.predict_from_nordsieck().unwrap();
        state
            .accept_step_with_correction(0.1, &[1.05], &[-0.05], 1.0, 0.1, 1)
            .unwrap();

        assert!(state.staged_higher_order_correction().is_some());

        state.reject_after_error_test(2.0).unwrap();

        assert!(state.staged_higher_order_correction().is_none());
    }

    #[test]
    fn runtime_state_can_accept_step_with_correction_via_history_rebuild_mode() {
        let mut state =
            Lsode2RuntimeState::new(0.0, &[1.0], 0.1, 3, Lsode2StepControlConfig::default())
                .unwrap();
        state.nordsieck_mut().set_col(1, &[0.1]).unwrap();
        state.predict_from_nordsieck().unwrap();

        let decision = state
            .accept_step_with_correction_mode(
                0.1,
                &[1.05],
                &[-0.05],
                1.0,
                0.1,
                1,
                Lsode2CorrectionAcceptUpdateMode::HistoryRebuild,
            )
            .unwrap();

        assert_eq!(decision.order_new, 1);
        assert_eq!(state.t(), 0.1);
        assert_eq!(state.y(), &[1.05]);
        assert_eq!(state.nordsieck().col(0).unwrap(), &[1.05]);
        assert!((state.nordsieck().col(1).unwrap()[0] - 0.05).abs() < 1.0e-12);
        assert_eq!(state.staged_higher_order_correction(), Some(&[-0.05][..]));
    }

    #[test]
    fn runtime_state_can_accept_step_via_adams_dstoda_style_elco_acor_update() {
        let mut state =
            Lsode2RuntimeState::new(0.0, &[1.0], 0.1, 3, Lsode2StepControlConfig::default())
                .unwrap();
        state.nordsieck_mut().set_col(1, &[0.1]).unwrap();
        state.predict_from_nordsieck().unwrap();

        let decision = state
            .accept_step_with_correction_mode(
                0.1,
                &[1.05],
                &[-0.05],
                1.0,
                0.1,
                1,
                Lsode2CorrectionAcceptUpdateMode::AdamsDstodaDirect,
            )
            .unwrap();

        assert_eq!(decision.order_new, 1);
        assert_eq!(state.t(), 0.1);
        assert_eq!(state.y(), &[1.05]);
        assert_eq!(state.nordsieck().col(0).unwrap(), &[1.05]);
        assert_eq!(state.nordsieck().col(1).unwrap(), &[0.05]);
        assert_eq!(state.staged_higher_order_correction(), Some(&[-0.05][..]));
    }
}
