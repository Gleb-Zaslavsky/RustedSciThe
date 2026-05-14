//! Small explicit DSTODA state-machine flags for LSODE2.
//!
//! ODEPACK's `DSTODA` routine keeps several compact integer flags (`JCUR`,
//! `IPUP`, `ICF`, `IREDO`, `KFLAG`) that drive subtle retry behavior.  Keeping
//! the same concepts explicit in Rust helps us mirror the Fortran control flow
//! without scattering ad-hoc booleans through the step cycle.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lsode2JacobianCurrency {
    Current,
    Stale,
}

impl Lsode2JacobianCurrency {
    pub fn is_current(self) -> bool {
        matches!(self, Self::Current)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lsode2JacobianUpdateRequest {
    None,
    Requested,
}

impl Lsode2JacobianUpdateRequest {
    pub fn is_requested(self) -> bool {
        matches!(self, Self::Requested)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lsode2CorrectorFailureMode {
    None,
    StaleJacobianRetry,
    StepRetraction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lsode2Kflag {
    Ok,
    ErrorTestFailure,
    RepeatedErrorTestFailure,
    ConvergenceFailure,
    RepeatedConvergenceFailure,
}

impl Lsode2Kflag {
    /// ODEPACK-style coarse numeric code used in diagnostics:
    /// - `0`  : normal accepted flow
    /// - `-1` : error-test failure path
    /// - `-2` : convergence-failure path
    pub fn code(self) -> i32 {
        match self {
            Self::Ok => 0,
            Self::ErrorTestFailure | Self::RepeatedErrorTestFailure => -1,
            Self::ConvergenceFailure | Self::RepeatedConvergenceFailure => -2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lsode2Ipup {
    UpToDate,
    NeedsJacobianUpdate,
}

impl Lsode2Ipup {
    pub fn needs_update(self) -> bool {
        matches!(self, Self::NeedsJacobianUpdate)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lsode2IpupTrigger {
    None,
    PredictorRcCcmax,
    PredictorMsbp,
    PredictorRcCcmaxAndMsbp,
    FailurePath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lsode2Icf {
    None,
    /// ODEPACK-style `ICF = 1`: retry after requesting a Jacobian refresh.
    RefreshRequested,
    /// ODEPACK-style `ICF = 2`: convergence failed after refresh path.
    RefreshDidNotRecover,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lsode2Iret {
    /// Normal predictor/corrector flow (`IRET = 0` equivalent in our split design).
    NormalFlow,
    /// History/step-size was rescaled and control should restart from preprocess branch.
    RescaleHistory,
    /// Retry path after regular error-test failure.
    RetryAfterErrorTestFailure,
    /// Restart path after repeated error-test failures with derivative refresh.
    RestartWithDerivativeRefresh,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lsode2RedoStage {
    None,
    CorrectorRefreshSameStep,
    CorrectorFailureRetry,
    ErrorTestRetry,
    RepeatedErrorReset,
    HistoryOrStepSizeChanged,
}

/// Explicit ODEPACK-style `IREDO` marker for diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lsode2Iredo {
    None,
    CorrectorRefreshSameStep,
    CorrectorFailureRetry,
    ErrorTestRetry,
    RepeatedErrorReset,
    HistoryOrStepSizeChanged,
}

impl Lsode2Iredo {
    /// ODEPACK-compatible coarse numeric classes:
    /// - `0`: normal/no redo
    /// - `1`: redo after corrector convergence failure path
    /// - `2`: redo after error-test failure path
    /// - `3`: history/step-size reset style redo path
    pub fn code(self) -> i32 {
        match self {
            Self::None => 0,
            Self::CorrectorRefreshSameStep => 1,
            Self::CorrectorFailureRetry => 1,
            Self::ErrorTestRetry => 2,
            Self::RepeatedErrorReset => 3,
            Self::HistoryOrStepSizeChanged => 3,
        }
    }
}

impl From<Lsode2RedoStage> for Lsode2Iredo {
    fn from(value: Lsode2RedoStage) -> Self {
        match value {
            Lsode2RedoStage::None => Self::None,
            Lsode2RedoStage::CorrectorRefreshSameStep => Self::CorrectorRefreshSameStep,
            Lsode2RedoStage::CorrectorFailureRetry => Self::CorrectorFailureRetry,
            Lsode2RedoStage::ErrorTestRetry => Self::ErrorTestRetry,
            Lsode2RedoStage::RepeatedErrorReset => Self::RepeatedErrorReset,
            Lsode2RedoStage::HistoryOrStepSizeChanged => Self::HistoryOrStepSizeChanged,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lsode2CorrectorFailureDecision {
    RefreshJacobianSameStep,
    RetractAndShrinkStep,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lsode2IterationMode {
    /// ODEPACK-like `MITER != 0`: Jacobian-based (chord/Newton) iteration.
    JacobianBased,
    /// ODEPACK-like `MITER == 0`: functional iteration (no Jacobian refresh path).
    Functional,
}

impl Lsode2IterationMode {
    fn uses_jacobian_iteration(self) -> bool {
        matches!(self, Self::JacobianBased)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lsode2DstodaState {
    jacobian_currency: Lsode2JacobianCurrency,
    jacobian_update_request: Lsode2JacobianUpdateRequest,
    corrector_failure_mode: Lsode2CorrectorFailureMode,
    kflag: Lsode2Kflag,
    ipup: Lsode2Ipup,
    ipup_trigger: Lsode2IpupTrigger,
    icf: Lsode2Icf,
    iret: Lsode2Iret,
    redo_stage: Lsode2RedoStage,
    rc: f64,
    ccmax: f64,
    nslp: usize,
    msbp: usize,
}

impl Default for Lsode2DstodaState {
    fn default() -> Self {
        Self {
            jacobian_currency: Lsode2JacobianCurrency::Current,
            jacobian_update_request: Lsode2JacobianUpdateRequest::None,
            corrector_failure_mode: Lsode2CorrectorFailureMode::None,
            kflag: Lsode2Kflag::Ok,
            ipup: Lsode2Ipup::UpToDate,
            ipup_trigger: Lsode2IpupTrigger::None,
            icf: Lsode2Icf::None,
            iret: Lsode2Iret::NormalFlow,
            redo_stage: Lsode2RedoStage::None,
            rc: 1.0,
            ccmax: 0.3,
            nslp: 0,
            msbp: 20,
        }
    }
}

impl Lsode2DstodaState {
    pub fn jacobian_currency(&self) -> Lsode2JacobianCurrency {
        self.jacobian_currency
    }

    pub fn jacobian_update_request(&self) -> Lsode2JacobianUpdateRequest {
        self.jacobian_update_request
    }

    pub fn corrector_failure_mode(&self) -> Lsode2CorrectorFailureMode {
        self.corrector_failure_mode
    }

    pub fn kflag(&self) -> Lsode2Kflag {
        self.kflag
    }

    pub fn ipup(&self) -> Lsode2Ipup {
        self.ipup
    }

    pub fn ipup_trigger(&self) -> Lsode2IpupTrigger {
        self.ipup_trigger
    }

    pub fn icf(&self) -> Lsode2Icf {
        self.icf
    }

    pub fn iret(&self) -> Lsode2Iret {
        self.iret
    }

    pub fn redo_stage(&self) -> Lsode2RedoStage {
        self.redo_stage
    }

    pub fn iredo(&self) -> Lsode2Iredo {
        self.redo_stage.into()
    }

    pub fn coefficient_ratio(&self) -> f64 {
        self.rc
    }

    pub fn mark_jacobian_current(&mut self, nst: usize) {
        self.jacobian_currency = Lsode2JacobianCurrency::Current;
        self.jacobian_update_request = Lsode2JacobianUpdateRequest::None;
        self.ipup = Lsode2Ipup::UpToDate;
        self.ipup_trigger = Lsode2IpupTrigger::None;
        self.rc = 1.0;
        self.nslp = nst;
    }

    pub fn mark_jacobian_stale(&mut self) {
        self.jacobian_currency = Lsode2JacobianCurrency::Stale;
    }

    pub fn request_jacobian_update(&mut self, reason: Lsode2IpupTrigger) {
        self.jacobian_update_request = Lsode2JacobianUpdateRequest::Requested;
        self.jacobian_currency = Lsode2JacobianCurrency::Stale;
        self.ipup = Lsode2Ipup::NeedsJacobianUpdate;
        self.ipup_trigger = reason;
    }

    /// Mirrors the DSTODA predictor gate:
    /// `if (abs(rc-1) > ccmax) ipup = miter`
    /// `if (nst >= nslp + msbp) ipup = miter`.
    pub fn maybe_request_jacobian_update_before_predict(
        &mut self,
        nst: usize,
        iteration_mode: Lsode2IterationMode,
    ) {
        if !iteration_mode.uses_jacobian_iteration() {
            return;
        }
        let rc_trigger = (self.rc - 1.0).abs() > self.ccmax;
        let msbp_trigger = nst >= self.nslp.saturating_add(self.msbp);
        let reason = match (rc_trigger, msbp_trigger) {
            (false, false) => None,
            (true, false) => Some(Lsode2IpupTrigger::PredictorRcCcmax),
            (false, true) => Some(Lsode2IpupTrigger::PredictorMsbp),
            (true, true) => Some(Lsode2IpupTrigger::PredictorRcCcmaxAndMsbp),
        };
        if let Some(reason) = reason {
            self.request_jacobian_update(reason);
        }
    }

    pub fn set_coefficient_ratio(&mut self, rc: f64) {
        self.rc = if rc.is_finite() { rc } else { 1.0 };
    }

    pub fn record_step_accepted(&mut self) {
        self.corrector_failure_mode = Lsode2CorrectorFailureMode::None;
        self.kflag = Lsode2Kflag::Ok;
        self.icf = Lsode2Icf::None;
        self.iret = Lsode2Iret::NormalFlow;
        self.redo_stage = Lsode2RedoStage::None;
    }

    pub fn record_corrector_converged(&mut self) {
        self.jacobian_currency = Lsode2JacobianCurrency::Stale;
        self.jacobian_update_request = Lsode2JacobianUpdateRequest::None;
        self.ipup = Lsode2Ipup::UpToDate;
        self.ipup_trigger = Lsode2IpupTrigger::None;
        self.icf = Lsode2Icf::None;
    }

    pub fn record_error_test_failure(&mut self) {
        self.corrector_failure_mode = Lsode2CorrectorFailureMode::None;
        self.kflag = Lsode2Kflag::ErrorTestFailure;
        self.iret = Lsode2Iret::RetryAfterErrorTestFailure;
        self.redo_stage = Lsode2RedoStage::ErrorTestRetry;
    }

    pub fn record_repeated_error_test_reset(&mut self) {
        self.corrector_failure_mode = Lsode2CorrectorFailureMode::None;
        self.kflag = Lsode2Kflag::ErrorTestFailure;
        self.iret = Lsode2Iret::RestartWithDerivativeRefresh;
        self.redo_stage = Lsode2RedoStage::RepeatedErrorReset;
        self.request_jacobian_update(Lsode2IpupTrigger::FailurePath);
    }

    pub fn record_repeated_error_test_failure(&mut self) {
        self.corrector_failure_mode = Lsode2CorrectorFailureMode::None;
        self.kflag = Lsode2Kflag::RepeatedErrorTestFailure;
        self.icf = Lsode2Icf::None;
        self.iret = Lsode2Iret::NormalFlow;
        self.redo_stage = Lsode2RedoStage::None;
        self.jacobian_update_request = Lsode2JacobianUpdateRequest::None;
        self.ipup = Lsode2Ipup::UpToDate;
        self.ipup_trigger = Lsode2IpupTrigger::None;
    }

    pub fn record_repeated_convergence_failure(&mut self) {
        self.corrector_failure_mode = Lsode2CorrectorFailureMode::StepRetraction;
        self.kflag = Lsode2Kflag::RepeatedConvergenceFailure;
        self.icf = Lsode2Icf::RefreshDidNotRecover;
        self.iret = Lsode2Iret::NormalFlow;
        self.redo_stage = Lsode2RedoStage::CorrectorFailureRetry;
    }

    pub fn record_history_or_step_size_change(&mut self) {
        self.iret = Lsode2Iret::RescaleHistory;
        self.redo_stage = Lsode2RedoStage::HistoryOrStepSizeChanged;
        self.request_jacobian_update(Lsode2IpupTrigger::FailurePath);
    }

    pub fn decide_after_corrector_failure(
        &mut self,
        iteration_mode: Lsode2IterationMode,
    ) -> Lsode2CorrectorFailureDecision {
        self.kflag = Lsode2Kflag::ConvergenceFailure;
        self.iret = Lsode2Iret::NormalFlow;
        // DSTODA parity:
        // label 410 chooses the one-shot same-step refresh path (`ICF = 1`)
        // only once, then label 430 (`ICF = 2`) retracts/shrinks on the next
        // convergence failure.
        let stale_retry_allowed =
            iteration_mode.uses_jacobian_iteration() && self.icf != Lsode2Icf::RefreshRequested;
        if stale_retry_allowed && !self.jacobian_currency.is_current() {
            self.corrector_failure_mode = Lsode2CorrectorFailureMode::StaleJacobianRetry;
            self.icf = Lsode2Icf::RefreshRequested;
            self.redo_stage = Lsode2RedoStage::CorrectorRefreshSameStep;
            self.request_jacobian_update(Lsode2IpupTrigger::FailurePath);
            Lsode2CorrectorFailureDecision::RefreshJacobianSameStep
        } else {
            self.corrector_failure_mode = Lsode2CorrectorFailureMode::StepRetraction;
            self.icf = Lsode2Icf::RefreshDidNotRecover;
            self.redo_stage = Lsode2RedoStage::CorrectorFailureRetry;
            Lsode2CorrectorFailureDecision::RetractAndShrinkStep
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dstoda_state_defaults_to_current_jacobian_and_clean_flags() {
        let state = Lsode2DstodaState::default();

        assert_eq!(state.jacobian_currency(), Lsode2JacobianCurrency::Current);
        assert_eq!(
            state.jacobian_update_request(),
            Lsode2JacobianUpdateRequest::None
        );
        assert_eq!(
            state.corrector_failure_mode(),
            Lsode2CorrectorFailureMode::None
        );
        assert_eq!(state.kflag(), Lsode2Kflag::Ok);
        assert_eq!(state.ipup(), Lsode2Ipup::UpToDate);
        assert_eq!(state.ipup_trigger(), Lsode2IpupTrigger::None);
        assert_eq!(state.icf(), Lsode2Icf::None);
        assert_eq!(state.iret(), Lsode2Iret::NormalFlow);
        assert_eq!(state.redo_stage(), Lsode2RedoStage::None);
    }

    #[test]
    fn stale_jacobian_corrector_failure_requests_same_step_refresh() {
        let mut state = Lsode2DstodaState::default();
        state.mark_jacobian_stale();

        let decision = state.decide_after_corrector_failure(Lsode2IterationMode::JacobianBased);

        assert_eq!(
            decision,
            Lsode2CorrectorFailureDecision::RefreshJacobianSameStep
        );
        assert_eq!(
            state.corrector_failure_mode(),
            Lsode2CorrectorFailureMode::StaleJacobianRetry
        );
        assert!(state.jacobian_update_request().is_requested());
        assert_eq!(state.kflag(), Lsode2Kflag::ConvergenceFailure);
        assert_eq!(state.ipup(), Lsode2Ipup::NeedsJacobianUpdate);
        assert_eq!(state.ipup_trigger(), Lsode2IpupTrigger::FailurePath);
        assert_eq!(state.icf(), Lsode2Icf::RefreshRequested);
        assert_eq!(
            state.redo_stage(),
            Lsode2RedoStage::CorrectorRefreshSameStep
        );
    }

    #[test]
    fn second_convergence_failure_after_refresh_request_retracts_step() {
        let mut state = Lsode2DstodaState::default();
        state.mark_jacobian_stale();
        let first = state.decide_after_corrector_failure(Lsode2IterationMode::JacobianBased);
        assert_eq!(
            first,
            Lsode2CorrectorFailureDecision::RefreshJacobianSameStep
        );
        assert_eq!(state.icf(), Lsode2Icf::RefreshRequested);

        // Mirror DSTODA 410 -> 430 progression: once `ICF=1` was used,
        // next failure must retract/shrink (`ICF=2`) rather than re-request
        // the same-step refresh indefinitely.
        state.mark_jacobian_stale();
        let second = state.decide_after_corrector_failure(Lsode2IterationMode::JacobianBased);
        assert_eq!(second, Lsode2CorrectorFailureDecision::RetractAndShrinkStep);
        assert_eq!(state.icf(), Lsode2Icf::RefreshDidNotRecover);
        assert_eq!(state.redo_stage(), Lsode2RedoStage::CorrectorFailureRetry);
    }

    #[test]
    fn current_jacobian_corrector_failure_retracts_without_preemptive_ipup_request() {
        let mut state = Lsode2DstodaState::default();

        let decision = state.decide_after_corrector_failure(Lsode2IterationMode::JacobianBased);

        assert_eq!(
            decision,
            Lsode2CorrectorFailureDecision::RetractAndShrinkStep
        );
        assert_eq!(
            state.corrector_failure_mode(),
            Lsode2CorrectorFailureMode::StepRetraction
        );
        // DSTODA parity: label-430 sets IPUP only on the nonterminal retry branch.
        assert_eq!(state.jacobian_currency(), Lsode2JacobianCurrency::Current);
        assert_eq!(
            state.jacobian_update_request(),
            Lsode2JacobianUpdateRequest::None
        );
        assert_eq!(state.ipup(), Lsode2Ipup::UpToDate);
        assert_eq!(state.ipup_trigger(), Lsode2IpupTrigger::None);
        assert_eq!(state.icf(), Lsode2Icf::RefreshDidNotRecover);
        assert_eq!(state.redo_stage(), Lsode2RedoStage::CorrectorFailureRetry);
    }

    #[test]
    fn corrector_convergence_marks_jacobian_stale_without_update_request() {
        let mut state = Lsode2DstodaState::default();
        state.request_jacobian_update(Lsode2IpupTrigger::FailurePath);
        state.mark_jacobian_current(0);

        state.record_corrector_converged();

        assert_eq!(state.jacobian_currency(), Lsode2JacobianCurrency::Stale);
        assert_eq!(
            state.jacobian_update_request(),
            Lsode2JacobianUpdateRequest::None
        );
        assert_eq!(state.kflag(), Lsode2Kflag::Ok);
    }

    #[test]
    fn periodic_msbp_policy_requests_jacobian_update() {
        let mut state = Lsode2DstodaState::default();
        state.mark_jacobian_current(0);

        state.maybe_request_jacobian_update_before_predict(19, Lsode2IterationMode::JacobianBased);
        assert_eq!(state.ipup(), Lsode2Ipup::UpToDate);
        assert_eq!(state.ipup_trigger(), Lsode2IpupTrigger::None);
        assert_eq!(
            state.jacobian_update_request(),
            Lsode2JacobianUpdateRequest::None
        );

        state.maybe_request_jacobian_update_before_predict(20, Lsode2IterationMode::JacobianBased);
        assert_eq!(state.ipup(), Lsode2Ipup::NeedsJacobianUpdate);
        assert_eq!(state.ipup_trigger(), Lsode2IpupTrigger::PredictorMsbp);
        assert!(state.jacobian_update_request().is_requested());
    }

    #[test]
    fn rc_ccmax_policy_requests_jacobian_update_like_dstoda() {
        let mut state = Lsode2DstodaState::default();
        state.mark_jacobian_current(0);

        state.set_coefficient_ratio(1.25);
        state.maybe_request_jacobian_update_before_predict(1, Lsode2IterationMode::JacobianBased);
        assert_eq!(state.ipup(), Lsode2Ipup::UpToDate);
        assert_eq!(state.ipup_trigger(), Lsode2IpupTrigger::None);

        state.set_coefficient_ratio(1.31);
        state.maybe_request_jacobian_update_before_predict(1, Lsode2IterationMode::JacobianBased);
        assert_eq!(state.ipup(), Lsode2Ipup::NeedsJacobianUpdate);
        assert_eq!(state.ipup_trigger(), Lsode2IpupTrigger::PredictorRcCcmax);
        assert!(state.jacobian_update_request().is_requested());
    }

    #[test]
    fn mark_jacobian_current_resets_rc_and_clears_spurious_predictor_refresh() {
        let mut state = Lsode2DstodaState::default();
        state.mark_jacobian_current(0);

        state.set_coefficient_ratio(1.5);
        state.maybe_request_jacobian_update_before_predict(1, Lsode2IterationMode::JacobianBased);
        assert_eq!(state.ipup(), Lsode2Ipup::NeedsJacobianUpdate);
        assert!(state.jacobian_update_request().is_requested());

        state.mark_jacobian_current(1);
        state.maybe_request_jacobian_update_before_predict(2, Lsode2IterationMode::JacobianBased);
        assert_eq!(state.ipup(), Lsode2Ipup::UpToDate);
        assert_eq!(state.ipup_trigger(), Lsode2IpupTrigger::None);
        assert_eq!(
            state.jacobian_update_request(),
            Lsode2JacobianUpdateRequest::None
        );
    }

    #[test]
    fn predictor_refresh_policy_is_disabled_for_functional_iteration_mode() {
        let mut state = Lsode2DstodaState::default();
        state.mark_jacobian_current(0);

        state.set_coefficient_ratio(5.0);
        state.maybe_request_jacobian_update_before_predict(100, Lsode2IterationMode::Functional);

        assert_eq!(state.ipup(), Lsode2Ipup::UpToDate);
        assert_eq!(state.ipup_trigger(), Lsode2IpupTrigger::None);
        assert_eq!(
            state.jacobian_update_request(),
            Lsode2JacobianUpdateRequest::None
        );
    }

    #[test]
    fn error_test_failure_records_iredo_like_retry_stage() {
        let mut state = Lsode2DstodaState::default();

        state.record_error_test_failure();

        assert_eq!(state.kflag(), Lsode2Kflag::ErrorTestFailure);
        assert_eq!(state.iret(), Lsode2Iret::RetryAfterErrorTestFailure);
        assert_eq!(state.redo_stage(), Lsode2RedoStage::ErrorTestRetry);
    }

    #[test]
    fn repeated_error_reset_requests_update_and_records_iredo_like_stage() {
        let mut state = Lsode2DstodaState::default();
        state.mark_jacobian_current(0);

        state.record_repeated_error_test_reset();

        assert_eq!(state.kflag(), Lsode2Kflag::ErrorTestFailure);
        assert_eq!(state.iret(), Lsode2Iret::RestartWithDerivativeRefresh);
        assert_eq!(state.ipup(), Lsode2Ipup::NeedsJacobianUpdate);
        assert_eq!(state.ipup_trigger(), Lsode2IpupTrigger::FailurePath);
        assert_eq!(state.redo_stage(), Lsode2RedoStage::RepeatedErrorReset);
        assert_eq!(state.jacobian_currency(), Lsode2JacobianCurrency::Stale);
        assert!(state.jacobian_update_request().is_requested());
    }

    #[test]
    fn functional_iteration_failure_retracts_without_stale_jacobian_retry() {
        let mut state = Lsode2DstodaState::default();
        state.mark_jacobian_stale();

        let decision = state.decide_after_corrector_failure(Lsode2IterationMode::Functional);

        assert_eq!(
            decision,
            Lsode2CorrectorFailureDecision::RetractAndShrinkStep
        );
        assert_eq!(
            state.corrector_failure_mode(),
            Lsode2CorrectorFailureMode::StepRetraction
        );
        assert_eq!(state.redo_stage(), Lsode2RedoStage::CorrectorFailureRetry);
        assert_eq!(state.icf(), Lsode2Icf::RefreshDidNotRecover);
    }

    #[test]
    fn repeated_convergence_failure_records_terminal_kflag() {
        let mut state = Lsode2DstodaState::default();
        state.record_history_or_step_size_change();
        assert_eq!(state.iret(), Lsode2Iret::RescaleHistory);
        assert_eq!(state.ipup(), Lsode2Ipup::NeedsJacobianUpdate);

        state.record_repeated_convergence_failure();

        assert_eq!(state.kflag(), Lsode2Kflag::RepeatedConvergenceFailure);
        // DSTODA parity: terminal convergence exit itself does not set IPUP;
        // the request belongs to the nonterminal retry branch only.
        assert_eq!(state.ipup(), Lsode2Ipup::NeedsJacobianUpdate);
        assert_eq!(state.ipup_trigger(), Lsode2IpupTrigger::FailurePath);
        assert_eq!(state.icf(), Lsode2Icf::RefreshDidNotRecover);
        assert_eq!(state.iret(), Lsode2Iret::NormalFlow);
        assert_eq!(
            state.corrector_failure_mode(),
            Lsode2CorrectorFailureMode::StepRetraction
        );
    }

    #[test]
    fn convergence_failure_clears_stale_iret_rescale_marker() {
        let mut state = Lsode2DstodaState::default();
        state.record_history_or_step_size_change();
        assert_eq!(state.iret(), Lsode2Iret::RescaleHistory);
        state.mark_jacobian_stale();

        let decision = state.decide_after_corrector_failure(Lsode2IterationMode::JacobianBased);

        assert_eq!(
            decision,
            Lsode2CorrectorFailureDecision::RefreshJacobianSameStep
        );
        assert_eq!(state.kflag(), Lsode2Kflag::ConvergenceFailure);
        assert_eq!(state.icf(), Lsode2Icf::RefreshRequested);
        assert_eq!(state.iret(), Lsode2Iret::NormalFlow);
        assert_eq!(
            state.redo_stage(),
            Lsode2RedoStage::CorrectorRefreshSameStep
        );
    }

    #[test]
    fn history_or_step_change_sets_iret_like_rescale_marker_and_ipup_request() {
        let mut state = Lsode2DstodaState::default();

        state.record_history_or_step_size_change();

        assert_eq!(state.iret(), Lsode2Iret::RescaleHistory);
        assert_eq!(
            state.redo_stage(),
            Lsode2RedoStage::HistoryOrStepSizeChanged
        );
        assert_eq!(state.iredo(), Lsode2Iredo::HistoryOrStepSizeChanged);
        assert_eq!(state.ipup(), Lsode2Ipup::NeedsJacobianUpdate);
        assert_eq!(state.ipup_trigger(), Lsode2IpupTrigger::FailurePath);
        assert!(state.jacobian_update_request().is_requested());
    }

    #[test]
    fn iredo_codes_match_story_table_contract() {
        assert_eq!(Lsode2Iredo::None.code(), 0);
        assert_eq!(Lsode2Iredo::CorrectorRefreshSameStep.code(), 1);
        assert_eq!(Lsode2Iredo::CorrectorFailureRetry.code(), 1);
        assert_eq!(Lsode2Iredo::ErrorTestRetry.code(), 2);
        assert_eq!(Lsode2Iredo::RepeatedErrorReset.code(), 3);
        assert_eq!(Lsode2Iredo::HistoryOrStepSizeChanged.code(), 3);
    }

    #[test]
    fn repeated_error_test_failure_records_terminal_kflag() {
        let mut state = Lsode2DstodaState::default();

        state.record_repeated_error_test_failure();

        assert_eq!(state.kflag(), Lsode2Kflag::RepeatedErrorTestFailure);
        assert_eq!(
            state.corrector_failure_mode(),
            Lsode2CorrectorFailureMode::None
        );
    }

    #[test]
    fn repeated_error_test_failure_clears_sticky_reset_markers() {
        let mut state = Lsode2DstodaState::default();
        state.record_repeated_error_test_reset();
        assert_eq!(state.iret(), Lsode2Iret::RestartWithDerivativeRefresh);
        assert_eq!(state.redo_stage(), Lsode2RedoStage::RepeatedErrorReset);
        assert_eq!(state.ipup(), Lsode2Ipup::NeedsJacobianUpdate);

        state.record_repeated_error_test_failure();

        assert_eq!(state.kflag(), Lsode2Kflag::RepeatedErrorTestFailure);
        assert_eq!(state.iret(), Lsode2Iret::NormalFlow);
        assert_eq!(state.redo_stage(), Lsode2RedoStage::None);
        assert_eq!(state.iredo(), Lsode2Iredo::None);
        assert_eq!(state.ipup(), Lsode2Ipup::UpToDate);
        assert_eq!(state.ipup_trigger(), Lsode2IpupTrigger::None);
        assert_eq!(
            state.jacobian_update_request(),
            Lsode2JacobianUpdateRequest::None
        );
    }

    #[test]
    fn kflag_numeric_codes_match_odepack_style_groups() {
        assert_eq!(Lsode2Kflag::Ok.code(), 0);
        assert_eq!(Lsode2Kflag::ErrorTestFailure.code(), -1);
        assert_eq!(Lsode2Kflag::RepeatedErrorTestFailure.code(), -1);
        assert_eq!(Lsode2Kflag::ConvergenceFailure.code(), -2);
        assert_eq!(Lsode2Kflag::RepeatedConvergenceFailure.code(), -2);
    }
}
