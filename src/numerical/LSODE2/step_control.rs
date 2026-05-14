//! LSODE-style step accept/reject bookkeeping.
//!
//! This module captures the DSTODA-like controller policy for retry decisions,
//! step shrink/growth limits, order lowering after repeated failures, and
//! telemetry used by Adams/BDF method-switch logic.

use super::algorithm::Lsode2SwitchTelemetry;

const DSTODA_HMIN_GUARD_FACTOR: f64 = 1.00001;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lsode2StepControlConfig {
    pub h_min: f64,
    pub error_shrink_safety: f64,
    pub nonlinear_shrink: f64,
    pub min_shrink: f64,
    pub max_error_shrink: f64,
    pub max_growth: f64,
    pub max_growth_after_failure: f64,
    pub lower_order_after_rejects: usize,
    pub raise_order_after_accepts: usize,
    pub repeated_error_adjustment_wait: usize,
    pub raise_order_error_threshold: f64,
    pub max_error_test_failures: usize,
    pub max_convergence_failures: usize,
    pub max_hnil_warnings: usize,
}

impl Default for Lsode2StepControlConfig {
    fn default() -> Self {
        Self {
            h_min: 1.0e-14,
            error_shrink_safety: 0.9,
            nonlinear_shrink: 0.25,
            min_shrink: 0.1,
            max_error_shrink: 0.5,
            max_growth: 5.0,
            max_growth_after_failure: 2.0,
            lower_order_after_rejects: 2,
            raise_order_after_accepts: 3,
            repeated_error_adjustment_wait: 5,
            raise_order_error_threshold: 0.5,
            max_error_test_failures: 10,
            max_convergence_failures: 10,
            max_hnil_warnings: 10,
        }
    }
}

impl Lsode2StepControlConfig {
    pub fn validate(self) -> Result<(), Lsode2StepControlError> {
        if !self.h_min.is_finite() || self.h_min <= 0.0 {
            return Err(Lsode2StepControlError::InvalidConfig(
                "h_min must be finite and positive",
            ));
        }
        if !is_unit_factor(self.error_shrink_safety) {
            return Err(Lsode2StepControlError::InvalidConfig(
                "error_shrink_safety must be in (0, 1]",
            ));
        }
        if !is_unit_factor(self.nonlinear_shrink) {
            return Err(Lsode2StepControlError::InvalidConfig(
                "nonlinear_shrink must be in (0, 1]",
            ));
        }
        if !is_unit_factor(self.min_shrink) || self.min_shrink > self.max_error_shrink {
            return Err(Lsode2StepControlError::InvalidConfig(
                "min_shrink must be in (0, max_error_shrink]",
            ));
        }
        if !is_unit_factor(self.max_error_shrink) {
            return Err(Lsode2StepControlError::InvalidConfig(
                "max_error_shrink must be in (0, 1]",
            ));
        }
        if !self.max_growth.is_finite() || self.max_growth < 1.0 {
            return Err(Lsode2StepControlError::InvalidConfig(
                "max_growth must be finite and >= 1",
            ));
        }
        if !self.max_growth_after_failure.is_finite()
            || self.max_growth_after_failure < 1.0
            || self.max_growth_after_failure > self.max_growth
        {
            return Err(Lsode2StepControlError::InvalidConfig(
                "max_growth_after_failure must be finite and in [1, max_growth]",
            ));
        }
        if self.raise_order_after_accepts == 0 {
            return Err(Lsode2StepControlError::InvalidConfig(
                "raise_order_after_accepts must be at least 1",
            ));
        }
        if self.repeated_error_adjustment_wait == 0 {
            return Err(Lsode2StepControlError::InvalidConfig(
                "repeated_error_adjustment_wait must be at least 1",
            ));
        }
        if !self.raise_order_error_threshold.is_finite() || self.raise_order_error_threshold <= 0.0
        {
            return Err(Lsode2StepControlError::InvalidConfig(
                "raise_order_error_threshold must be finite and positive",
            ));
        }
        if self.max_convergence_failures == 0 {
            return Err(Lsode2StepControlError::InvalidConfig(
                "max_convergence_failures must be at least 1",
            ));
        }
        if self.max_error_test_failures == 0 {
            return Err(Lsode2StepControlError::InvalidConfig(
                "max_error_test_failures must be at least 1",
            ));
        }
        if self.max_hnil_warnings == 0 {
            return Err(Lsode2StepControlError::InvalidConfig(
                "max_hnil_warnings must be at least 1",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Lsode2StepControlError {
    InvalidConfig(&'static str),
    StepSizeUnderflow,
}

impl std::fmt::Display for Lsode2StepControlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(message) => {
                write!(f, "invalid LSODE2 step-control config: {message}")
            }
            Self::StepSizeUnderflow => write!(f, "LSODE2 step size underflow"),
        }
    }
}

impl std::error::Error for Lsode2StepControlError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lsode2StepFailure {
    ErrorTest,
    NonlinearConvergence,
}

impl Lsode2StepFailure {
    pub fn label(self) -> &'static str {
        match self {
            Self::ErrorTest => "error_test",
            Self::NonlinearConvergence => "nonlinear_convergence",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lsode2RetryAction {
    Retry,
    RetryWithJacobianRefresh,
    FailStepSizeUnderflow,
    FailRepeatedErrorTestFailures,
    FailRepeatedConvergenceFailures,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lsode2RetryDecision {
    pub action: Lsode2RetryAction,
    pub failure: Lsode2StepFailure,
    pub h_new: f64,
    pub order_new: usize,
    pub consecutive_rejections: usize,
    pub shrink_factor: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lsode2AcceptDecision {
    pub h_next: f64,
    pub order_new: usize,
    pub consecutive_accepts: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Lsode2StepControlSnapshot {
    pub accepted_steps: usize,
    pub rejected_steps: usize,
    pub error_test_failures: usize,
    pub consecutive_error_test_failures: usize,
    pub convergence_failures: usize,
    pub consecutive_convergence_failures: usize,
    pub consecutive_rejections: usize,
    pub consecutive_accepts: usize,
    pub adjustment_wait: usize,
    pub active_growth_cap: f64,
    pub jacobian_refresh_requests: usize,
    pub null_step_count: usize,
    pub null_step_warning_count: usize,
    pub null_step_warning_cap: usize,
    pub null_step_warning_cap_reached: bool,
    pub last_failure: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lsode2NullStepWarningLevel {
    Warning,
    WarningCapReached,
    SuppressedAfterCap,
}

#[derive(Debug, Clone)]
pub struct Lsode2StepController {
    config: Lsode2StepControlConfig,
    accepted_steps: usize,
    rejected_steps: usize,
    error_test_failures: usize,
    consecutive_error_test_failures: usize,
    convergence_failures: usize,
    consecutive_convergence_failures: usize,
    consecutive_rejections: usize,
    consecutive_accepts: usize,
    adjustment_wait: usize,
    active_growth_cap: f64,
    jacobian_refresh_requests: usize,
    null_step_count: usize,
    null_step_warning_count: usize,
    last_failure: Option<Lsode2StepFailure>,
}

impl Lsode2StepController {
    pub fn new(config: Lsode2StepControlConfig) -> Result<Self, Lsode2StepControlError> {
        config.validate()?;
        Ok(Self {
            config,
            accepted_steps: 0,
            rejected_steps: 0,
            error_test_failures: 0,
            consecutive_error_test_failures: 0,
            convergence_failures: 0,
            consecutive_convergence_failures: 0,
            consecutive_rejections: 0,
            consecutive_accepts: 0,
            adjustment_wait: 2,
            active_growth_cap: config.max_growth,
            jacobian_refresh_requests: 0,
            null_step_count: 0,
            null_step_warning_count: 0,
            last_failure: None,
        })
    }

    pub fn config(&self) -> Lsode2StepControlConfig {
        self.config
    }

    pub fn accept(
        &mut self,
        h_current: f64,
        order_current: usize,
        order_cap: usize,
        suggested_growth: f64,
        accepted_error_norm: f64,
        suggested_order: usize,
    ) -> Lsode2AcceptDecision {
        self.accepted_steps += 1;
        self.consecutive_accepts += 1;
        self.consecutive_rejections = 0;
        self.consecutive_error_test_failures = 0;
        self.consecutive_convergence_failures = 0;
        self.last_failure = None;
        if self.adjustment_wait > 0 {
            self.adjustment_wait -= 1;
        }

        let requested_growth = if suggested_growth.is_finite() {
            suggested_growth.clamp(self.config.min_shrink, self.active_growth_cap)
        } else {
            1.0
        };
        // DSTODA-mirroring:
        // The step-cycle already computes the post-accept order choice via
        // RHUP/RHSM/RHDN choreography. Do not add an extra gate for
        // "enough consecutive accepts" or "small enough accepted error".
        // Those thresholds are handrolled and can block legitimate order raise.
        let requested_order = if suggested_order < order_current {
            suggested_order.max(1)
        } else if suggested_order > order_current && order_current < order_cap {
            suggested_order.min(order_cap)
        } else {
            order_current
        };

        if self.adjustment_wait > 0 {
            return Lsode2AcceptDecision {
                h_next: h_current,
                order_new: order_current,
                consecutive_accepts: self.consecutive_accepts,
            };
        }

        // DSTODA parity:
        // Order/growth choreography (including the 1.1 gate and Adams
        // stability-limited bypass) is already resolved upstream by step-cycle
        // RH selection. Here we should apply any non-unity RH that reaches
        // accept(), rather than re-imposing a second 1.1 threshold.
        let should_change_h_or_order =
            requested_order != order_current || (requested_growth - 1.0).abs() > f64::EPSILON;
        let (h_next, order_new) = if should_change_h_or_order {
            self.adjustment_wait = requested_order + 1;
            (h_current * requested_growth, requested_order)
        } else {
            self.adjustment_wait = 3;
            (h_current, order_current)
        };
        self.active_growth_cap = self.config.max_growth;

        Lsode2AcceptDecision {
            h_next,
            order_new,
            consecutive_accepts: self.consecutive_accepts,
        }
    }

    pub fn reject_after_error_test(
        &mut self,
        h_current: f64,
        order_current: usize,
        error_norm: f64,
    ) -> Lsode2RetryDecision {
        let raw_factor = if error_norm.is_finite() && error_norm > 0.0 {
            self.config.error_shrink_safety / error_norm.powf(1.0 / (order_current + 1) as f64)
        } else {
            self.config.min_shrink
        };
        let shrink_factor = raw_factor.clamp(self.config.min_shrink, self.config.max_error_shrink);
        self.reject(
            h_current,
            order_current,
            Lsode2StepFailure::ErrorTest,
            shrink_factor,
            false,
        )
    }

    pub fn reject_after_error_test_with_hint(
        &mut self,
        h_current: f64,
        _order_current: usize,
        h_new: f64,
        order_new: usize,
    ) -> Lsode2RetryDecision {
        self.rejected_steps += 1;
        self.error_test_failures += 1;
        self.consecutive_error_test_failures += 1;
        self.consecutive_rejections += 1;
        self.consecutive_accepts = 0;
        self.last_failure = Some(Lsode2StepFailure::ErrorTest);
        self.limit_growth_after_failure();

        let shrink_factor = if h_current != 0.0 {
            h_new / h_current
        } else {
            0.0
        };
        let action = if self.consecutive_error_test_failures >= self.config.max_error_test_failures
        {
            Lsode2RetryAction::FailRepeatedErrorTestFailures
        } else if is_hmin_underflow_like_dstoda(h_new.abs(), self.config.h_min) {
            Lsode2RetryAction::FailStepSizeUnderflow
        } else {
            Lsode2RetryAction::Retry
        };

        Lsode2RetryDecision {
            action,
            failure: Lsode2StepFailure::ErrorTest,
            h_new,
            order_new,
            consecutive_rejections: self.consecutive_rejections,
            shrink_factor,
        }
    }

    pub fn reject_after_nonlinear_failure(
        &mut self,
        h_current: f64,
        order_current: usize,
    ) -> Lsode2RetryDecision {
        self.convergence_failures += 1;
        self.consecutive_convergence_failures += 1;
        self.reject(
            h_current,
            order_current,
            Lsode2StepFailure::NonlinearConvergence,
            self.config.nonlinear_shrink,
            true,
        )
    }

    pub fn retry_after_stale_jacobian_nonlinear_failure(
        &mut self,
        h_current: f64,
        order_current: usize,
    ) -> Lsode2RetryDecision {
        // ODEPACK DSTODA parity:
        // first convergence-failure branch with stale Jacobian (`ICF = 1`)
        // requests a matrix refresh and retries same step without incrementing
        // the convergence-failure counter (`NCF` increments only after this
        // refresh path fails and we retract/shrink the step).
        self.consecutive_accepts = 0;
        self.last_failure = Some(Lsode2StepFailure::NonlinearConvergence);
        self.jacobian_refresh_requests += 1;
        self.limit_growth_after_failure();

        let action = Lsode2RetryAction::RetryWithJacobianRefresh;

        Lsode2RetryDecision {
            action,
            failure: Lsode2StepFailure::NonlinearConvergence,
            h_new: h_current,
            order_new: order_current,
            consecutive_rejections: self.consecutive_rejections,
            shrink_factor: 1.0,
        }
    }

    pub fn snapshot(&self) -> Lsode2StepControlSnapshot {
        Lsode2StepControlSnapshot {
            accepted_steps: self.accepted_steps,
            rejected_steps: self.rejected_steps,
            error_test_failures: self.error_test_failures,
            consecutive_error_test_failures: self.consecutive_error_test_failures,
            convergence_failures: self.convergence_failures,
            consecutive_convergence_failures: self.consecutive_convergence_failures,
            consecutive_rejections: self.consecutive_rejections,
            consecutive_accepts: self.consecutive_accepts,
            adjustment_wait: self.adjustment_wait,
            active_growth_cap: self.active_growth_cap,
            jacobian_refresh_requests: self.jacobian_refresh_requests,
            null_step_count: self.null_step_count,
            null_step_warning_count: self.null_step_warning_count,
            null_step_warning_cap: self.config.max_hnil_warnings,
            null_step_warning_cap_reached: self.null_step_count >= self.config.max_hnil_warnings,
            last_failure: self.last_failure.map(Lsode2StepFailure::label),
        }
    }

    pub fn record_null_step_event(&mut self) -> Lsode2NullStepWarningLevel {
        self.null_step_count += 1;
        if self.null_step_count > self.config.max_hnil_warnings {
            return Lsode2NullStepWarningLevel::SuppressedAfterCap;
        }
        self.null_step_warning_count += 1;
        if self.null_step_count == self.config.max_hnil_warnings {
            Lsode2NullStepWarningLevel::WarningCapReached
        } else {
            Lsode2NullStepWarningLevel::Warning
        }
    }

    pub fn record_repeated_error_test_reset(&mut self) {
        self.adjustment_wait = self.config.repeated_error_adjustment_wait;
        self.jacobian_refresh_requests += 1;
    }

    pub fn switch_telemetry(&self, stiffness_ratio: Option<f64>) -> Lsode2SwitchTelemetry {
        Lsode2SwitchTelemetry {
            stiffness_ratio,
            accepted_steps: self.accepted_steps,
            rejected_steps: self.consecutive_rejections,
            convergence_failures: self.convergence_failures,
            adams_step_cost_estimate: None,
            bdf_step_cost_estimate: None,
            adams_cost_samples: 0,
            bdf_cost_samples: 0,
            adams_step_size_cap_estimate: None,
            bdf_step_size_cap_estimate: None,
        }
    }

    fn reject(
        &mut self,
        h_current: f64,
        order_current: usize,
        failure: Lsode2StepFailure,
        shrink_factor: f64,
        request_jacobian_refresh: bool,
    ) -> Lsode2RetryDecision {
        self.rejected_steps += 1;
        if failure == Lsode2StepFailure::ErrorTest {
            self.error_test_failures += 1;
            self.consecutive_error_test_failures += 1;
        }
        self.consecutive_rejections += 1;
        self.consecutive_accepts = 0;
        self.last_failure = Some(failure);
        self.limit_growth_after_failure();
        if request_jacobian_refresh {
            self.jacobian_refresh_requests += 1;
        }

        let h_new = h_current * shrink_factor;
        // ODEPACK DSTODA parity:
        // - error-test failure path may retry with lowered order;
        // - convergence-failure retract path (label 450) forces NQ=1.
        let order_new = if failure == Lsode2StepFailure::ErrorTest
            && order_current > 1
            && self.consecutive_rejections >= self.config.lower_order_after_rejects
        {
            order_current - 1
        } else if failure == Lsode2StepFailure::NonlinearConvergence && order_current > 1 {
            1
        } else {
            order_current
        };
        let action = if failure == Lsode2StepFailure::ErrorTest
            && self.consecutive_error_test_failures >= self.config.max_error_test_failures
        {
            Lsode2RetryAction::FailRepeatedErrorTestFailures
        } else if is_hmin_underflow_like_dstoda(h_new.abs(), self.config.h_min) {
            Lsode2RetryAction::FailStepSizeUnderflow
        } else if failure == Lsode2StepFailure::NonlinearConvergence
            && self.consecutive_convergence_failures >= self.config.max_convergence_failures
        {
            Lsode2RetryAction::FailRepeatedConvergenceFailures
        } else if request_jacobian_refresh {
            Lsode2RetryAction::RetryWithJacobianRefresh
        } else {
            Lsode2RetryAction::Retry
        };

        Lsode2RetryDecision {
            action,
            failure,
            h_new,
            order_new,
            consecutive_rejections: self.consecutive_rejections,
            shrink_factor,
        }
    }

    fn limit_growth_after_failure(&mut self) {
        self.active_growth_cap = self
            .active_growth_cap
            .min(self.config.max_growth_after_failure);
    }
}

impl Default for Lsode2StepController {
    fn default() -> Self {
        Self::new(Lsode2StepControlConfig::default())
            .expect("default LSODE2 step-control config should be valid")
    }
}

fn is_hmin_underflow_like_dstoda(h_abs: f64, h_min: f64) -> bool {
    h_abs <= h_min * DSTODA_HMIN_GUARD_FACTOR
}

fn is_unit_factor(value: f64) -> bool {
    value.is_finite() && value > 0.0 && value <= 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_test_rejection_shrinks_step_and_lowers_order_after_repeated_failures() {
        let mut controller = Lsode2StepController::default();

        let first = controller.reject_after_error_test(1.0, 3, 16.0);
        assert_eq!(first.action, Lsode2RetryAction::Retry);
        assert_eq!(first.order_new, 3);
        assert!(first.h_new > 0.0 && first.h_new <= 0.5);

        let second = controller.reject_after_error_test(first.h_new, 3, 16.0);
        assert_eq!(second.action, Lsode2RetryAction::Retry);
        assert_eq!(second.order_new, 2);
        assert_eq!(second.consecutive_rejections, 2);
        assert_eq!(controller.snapshot().error_test_failures, 2);
        assert_eq!(controller.snapshot().consecutive_error_test_failures, 2);
        assert_eq!(controller.snapshot().last_failure, Some("error_test"));
    }

    #[test]
    fn repeated_error_test_failures_stop_at_odepack_kflag_limit() {
        let config = Lsode2StepControlConfig {
            max_error_test_failures: 2,
            ..Lsode2StepControlConfig::default()
        };
        let mut controller = Lsode2StepController::new(config).unwrap();

        let first = controller.reject_after_error_test(1.0, 2, 16.0);
        let second = controller.reject_after_error_test(first.h_new, first.order_new, 16.0);

        assert_eq!(first.action, Lsode2RetryAction::Retry);
        assert_eq!(
            second.action,
            Lsode2RetryAction::FailRepeatedErrorTestFailures
        );
        let snapshot = controller.snapshot();
        assert_eq!(snapshot.error_test_failures, 2);
        assert_eq!(snapshot.consecutive_error_test_failures, 2);
    }

    #[test]
    fn repeated_error_test_reset_sets_ialth_like_wait_and_jacobian_refresh_request() {
        let mut controller = Lsode2StepController::default();
        controller.reject_after_error_test(1.0, 2, 16.0);
        controller.reject_after_error_test(0.5, 2, 16.0);

        controller.record_repeated_error_test_reset();

        let snapshot = controller.snapshot();
        assert_eq!(snapshot.adjustment_wait, 5);
        assert_eq!(snapshot.jacobian_refresh_requests, 1);
    }

    #[test]
    fn nonlinear_failure_requests_jacobian_refresh_and_feeds_switch_telemetry() {
        let mut controller = Lsode2StepController::default();

        let decision = controller.reject_after_nonlinear_failure(1.0, 2);

        assert_eq!(decision.action, Lsode2RetryAction::RetryWithJacobianRefresh);
        assert_eq!(decision.failure, Lsode2StepFailure::NonlinearConvergence);
        assert_eq!(decision.h_new, 0.25);
        assert_eq!(controller.snapshot().jacobian_refresh_requests, 1);

        let telemetry = controller.switch_telemetry(Some(2.0));
        assert_eq!(telemetry.stiffness_ratio, Some(2.0));
        assert_eq!(telemetry.rejected_steps, 1);
        assert_eq!(telemetry.convergence_failures, 1);
        assert_eq!(controller.snapshot().consecutive_convergence_failures, 1);
    }

    #[test]
    fn stale_jacobian_nonlinear_failure_retries_without_retracting_step() {
        let mut controller = Lsode2StepController::default();

        let decision = controller.retry_after_stale_jacobian_nonlinear_failure(1.0, 2);

        assert_eq!(decision.action, Lsode2RetryAction::RetryWithJacobianRefresh);
        assert_eq!(decision.failure, Lsode2StepFailure::NonlinearConvergence);
        assert_eq!(decision.h_new, 1.0);
        assert_eq!(decision.order_new, 2);
        assert_eq!(decision.shrink_factor, 1.0);
        let snapshot = controller.snapshot();
        assert_eq!(snapshot.rejected_steps, 0);
        assert_eq!(snapshot.consecutive_rejections, 0);
        assert_eq!(snapshot.convergence_failures, 0);
        assert_eq!(snapshot.consecutive_convergence_failures, 0);
        assert_eq!(snapshot.jacobian_refresh_requests, 1);
        assert_eq!(snapshot.last_failure, Some("nonlinear_convergence"));
    }

    #[test]
    fn repeated_nonlinear_failures_stop_at_mxncf_like_limit() {
        let config = Lsode2StepControlConfig {
            max_convergence_failures: 2,
            ..Lsode2StepControlConfig::default()
        };
        let mut controller = Lsode2StepController::new(config).unwrap();

        let first = controller.reject_after_nonlinear_failure(1.0, 2);
        let second = controller.reject_after_nonlinear_failure(first.h_new, first.order_new);

        assert_eq!(first.action, Lsode2RetryAction::RetryWithJacobianRefresh);
        assert_eq!(
            second.action,
            Lsode2RetryAction::FailRepeatedConvergenceFailures
        );
        let snapshot = controller.snapshot();
        assert_eq!(snapshot.convergence_failures, 2);
        assert_eq!(snapshot.consecutive_convergence_failures, 2);
        assert_eq!(snapshot.rejected_steps, 2);
    }

    #[test]
    fn nonlinear_reject_path_drops_to_first_order_like_dstoda_label_450() {
        let mut controller = Lsode2StepController::default();

        let first = controller.reject_after_nonlinear_failure(1.0, 3);
        let second = controller.reject_after_nonlinear_failure(first.h_new, first.order_new);

        assert_eq!(first.order_new, 1);
        assert_eq!(second.order_new, 1);
        assert_eq!(first.action, Lsode2RetryAction::RetryWithJacobianRefresh);
        assert_eq!(second.action, Lsode2RetryAction::RetryWithJacobianRefresh);
    }

    #[test]
    fn accept_clears_mxncf_like_convergence_failure_count() {
        let config = Lsode2StepControlConfig {
            max_convergence_failures: 2,
            ..Lsode2StepControlConfig::default()
        };
        let mut controller = Lsode2StepController::new(config).unwrap();

        controller.reject_after_nonlinear_failure(1.0, 1);
        controller.accept(0.25, 1, 2, 1.0, 0.1, 1);
        let retry = controller.reject_after_nonlinear_failure(0.25, 1);

        assert_eq!(retry.action, Lsode2RetryAction::RetryWithJacobianRefresh);
        let snapshot = controller.snapshot();
        assert_eq!(snapshot.convergence_failures, 2);
        assert_eq!(snapshot.consecutive_convergence_failures, 1);
    }

    #[test]
    fn accept_clears_error_test_failure_streak() {
        let config = Lsode2StepControlConfig {
            max_error_test_failures: 2,
            ..Lsode2StepControlConfig::default()
        };
        let mut controller = Lsode2StepController::new(config).unwrap();

        controller.reject_after_error_test(1.0, 1, 16.0);
        controller.accept(0.5, 1, 2, 1.0, 0.1, 1);
        let retry = controller.reject_after_error_test(0.5, 1, 16.0);

        assert_eq!(retry.action, Lsode2RetryAction::Retry);
        let snapshot = controller.snapshot();
        assert_eq!(snapshot.error_test_failures, 2);
        assert_eq!(snapshot.consecutive_error_test_failures, 1);
    }

    #[test]
    fn accept_resets_consecutive_rejections_and_caps_growth() {
        let mut controller = Lsode2StepController::default();
        controller.reject_after_error_test(1.0, 2, 4.0);

        let accepted = controller.accept(0.5, 2, 5, 100.0, 0.25, 2);

        assert_eq!(accepted.h_next, 0.5);
        assert_eq!(accepted.order_new, 2);
        assert_eq!(accepted.consecutive_accepts, 1);
        let snapshot = controller.snapshot();
        assert_eq!(snapshot.accepted_steps, 1);
        assert_eq!(snapshot.consecutive_rejections, 0);
        assert_eq!(snapshot.adjustment_wait, 1);
        assert_eq!(snapshot.last_failure, None);
    }

    #[test]
    fn failure_growth_cap_persists_until_next_step_change() {
        let mut controller = Lsode2StepController::default();
        let retry = controller.reject_after_error_test(1.0, 2, 4.0);
        assert_eq!(controller.snapshot().active_growth_cap, 2.0);

        let waiting_accept = controller.accept(retry.h_new, retry.order_new, 5, 100.0, 0.1, 2);
        assert_eq!(waiting_accept.h_next, retry.h_new);
        assert_eq!(controller.snapshot().active_growth_cap, 2.0);

        let growth_accept = controller.accept(
            waiting_accept.h_next,
            waiting_accept.order_new,
            5,
            100.0,
            0.1,
            2,
        );
        assert_eq!(growth_accept.h_next, retry.h_new * 2.0);
        assert_eq!(controller.snapshot().active_growth_cap, 5.0);
    }

    #[test]
    fn repeated_accepts_can_raise_order_within_cap() {
        let mut controller = Lsode2StepController::default();

        let first = controller.accept(0.1, 1, 3, 1.2, 0.1, 2);
        let second = controller.accept(first.h_next, first.order_new, 3, 1.2, 0.1, 2);
        let third = controller.accept(second.h_next, second.order_new, 3, 1.2, 0.1, 2);
        let fourth = controller.accept(third.h_next, third.order_new, 3, 1.2, 0.1, 2);

        assert_eq!(first.order_new, 1);
        assert_eq!(second.order_new, 2);
        assert_eq!(third.order_new, 2);
        assert_eq!(fourth.order_new, 2);
        assert_eq!(first.h_next, 0.1);
        assert_eq!(second.h_next, 0.12);
        assert_eq!(third.h_next, 0.12);
        assert_eq!(fourth.h_next, 0.12);
        assert_eq!(controller.snapshot().consecutive_accepts, 4);
    }

    #[test]
    fn underflow_is_reported_as_non_retryable_action() {
        let config = Lsode2StepControlConfig {
            h_min: 0.1,
            ..Lsode2StepControlConfig::default()
        };
        let mut controller = Lsode2StepController::new(config).unwrap();

        let decision = controller.reject_after_nonlinear_failure(0.2, 1);

        assert_eq!(decision.action, Lsode2RetryAction::FailStepSizeUnderflow);
    }

    #[test]
    fn h_equal_to_hmin_is_treated_as_underflow_like_dstoda_guard() {
        let config = Lsode2StepControlConfig {
            h_min: 0.1,
            ..Lsode2StepControlConfig::default()
        };
        let mut controller = Lsode2StepController::new(config).unwrap();

        let decision = controller.reject_after_error_test_with_hint(0.2, 2, 0.1, 2);

        assert_eq!(decision.action, Lsode2RetryAction::FailStepSizeUnderflow);
        assert_eq!(decision.h_new, 0.1);
        assert_eq!(decision.order_new, 2);
    }

    #[test]
    fn h_within_dstoda_hmin_guard_factor_is_treated_as_underflow() {
        let config = Lsode2StepControlConfig {
            h_min: 1.0,
            ..Lsode2StepControlConfig::default()
        };
        let mut controller = Lsode2StepController::new(config).unwrap();

        let decision = controller.reject_after_error_test_with_hint(2.0, 2, 1.000009, 2);

        assert_eq!(decision.action, Lsode2RetryAction::FailStepSizeUnderflow);
        assert_eq!(decision.h_new, 1.000009);
    }

    #[test]
    fn h_above_dstoda_hmin_guard_factor_stays_retryable() {
        let config = Lsode2StepControlConfig {
            h_min: 1.0,
            ..Lsode2StepControlConfig::default()
        };
        let mut controller = Lsode2StepController::new(config).unwrap();

        let decision = controller.reject_after_error_test_with_hint(2.0, 2, 1.00002, 2);

        assert_eq!(decision.action, Lsode2RetryAction::Retry);
        assert_eq!(decision.h_new, 1.00002);
    }

    #[test]
    fn invalid_step_control_config_is_rejected() {
        let config = Lsode2StepControlConfig {
            h_min: 0.0,
            ..Lsode2StepControlConfig::default()
        };

        assert!(Lsode2StepController::new(config).is_err());

        let config = Lsode2StepControlConfig {
            raise_order_after_accepts: 0,
            ..Lsode2StepControlConfig::default()
        };

        assert!(Lsode2StepController::new(config).is_err());

        let config = Lsode2StepControlConfig {
            repeated_error_adjustment_wait: 0,
            ..Lsode2StepControlConfig::default()
        };

        assert!(Lsode2StepController::new(config).is_err());

        let config = Lsode2StepControlConfig {
            raise_order_error_threshold: 0.0,
            ..Lsode2StepControlConfig::default()
        };

        assert!(Lsode2StepController::new(config).is_err());

        let config = Lsode2StepControlConfig {
            max_convergence_failures: 0,
            ..Lsode2StepControlConfig::default()
        };

        assert!(Lsode2StepController::new(config).is_err());

        let config = Lsode2StepControlConfig {
            max_error_test_failures: 0,
            ..Lsode2StepControlConfig::default()
        };

        assert!(Lsode2StepController::new(config).is_err());

        let config = Lsode2StepControlConfig {
            max_growth_after_failure: 6.0,
            ..Lsode2StepControlConfig::default()
        };

        assert!(Lsode2StepController::new(config).is_err());

        let config = Lsode2StepControlConfig {
            max_hnil_warnings: 0,
            ..Lsode2StepControlConfig::default()
        };

        assert!(Lsode2StepController::new(config).is_err());
    }

    #[test]
    fn repeated_accepts_can_raise_order_even_when_error_norm_is_not_clean_enough() {
        let mut controller = Lsode2StepController::default();

        let first = controller.accept(0.1, 1, 3, 1.2, 0.9, 2);
        let second = controller.accept(first.h_next, first.order_new, 3, 1.2, 0.8, 2);
        let third = controller.accept(second.h_next, second.order_new, 3, 1.2, 0.75, 2);

        assert_eq!(first.order_new, 1);
        assert_eq!(second.order_new, 2);
        assert_eq!(third.order_new, 2);
        assert_eq!(controller.snapshot().consecutive_accepts, 3);
    }

    #[test]
    fn accept_respects_history_limited_order_cap() {
        let mut controller = Lsode2StepController::default();

        let accepted = controller.accept(0.1, 2, 2, 1.2, 0.1, 3);

        assert_eq!(accepted.order_new, 2);
    }

    #[test]
    fn accept_can_lower_order_after_adjustment_wait_expires() {
        let mut controller = Lsode2StepController::default();

        let first = controller.accept(0.1, 3, 5, 1.1, 0.8, 2);
        let second = controller.accept(first.h_next, first.order_new, 5, 1.1, 0.8, 2);
        let third = controller.accept(second.h_next, second.order_new, 5, 1.1, 0.8, 2);

        assert_eq!(first.order_new, 3);
        assert_eq!(second.order_new, 2);
        assert_eq!(third.order_new, 2);
    }

    #[test]
    fn null_step_warning_counter_respects_mxhnil_cap() {
        let mut controller = Lsode2StepController::new(Lsode2StepControlConfig {
            max_hnil_warnings: 2,
            ..Lsode2StepControlConfig::default()
        })
        .expect("valid step controller");

        let first = controller.record_null_step_event();
        let second = controller.record_null_step_event();
        let third = controller.record_null_step_event();

        assert_eq!(first, Lsode2NullStepWarningLevel::Warning);
        assert_eq!(second, Lsode2NullStepWarningLevel::WarningCapReached);
        assert_eq!(third, Lsode2NullStepWarningLevel::SuppressedAfterCap);

        let snapshot = controller.snapshot();
        assert_eq!(snapshot.null_step_count, 3);
        assert_eq!(snapshot.null_step_warning_count, 2);
        assert_eq!(snapshot.null_step_warning_cap, 2);
        assert!(snapshot.null_step_warning_cap_reached);
    }

    #[test]
    fn accept_applies_nonunity_growth_without_secondary_ten_percent_gate() {
        let mut controller = Lsode2StepController::default();

        // Warmup call consumes initial IALTH-like wait (2 -> 1), no immediate change.
        let warmup = controller.accept(0.2, 1, 3, 1.0, 0.1, 1);
        assert!((warmup.h_next - 0.2).abs() < 1.0e-14);
        assert_eq!(warmup.order_new, 1);

        // With wait now at 1, second call drops it to 0 and applies RH directly.
        // RH < 1.1 must still be applied when upstream DSTODA selection left RH != 1.
        let applied = controller.accept(0.2, 1, 3, 0.93, 0.1, 1);
        assert!(
            (applied.h_next - 0.186).abs() < 1.0e-14,
            "non-unity RH from DSTODA choreography should be applied directly"
        );
        assert_eq!(applied.order_new, 1);
    }
}
