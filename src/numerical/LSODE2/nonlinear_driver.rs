//! Small native nonlinear-step orchestration layer for LSODE2.
//!
//! This driver sits one level above [`crate::numerical::LSODE2::step_cycle`]:
//! it owns Newton-iteration bookkeeping (`iteration`, previous correction norm,
//! active prediction) and pushes every meaningful event into
//! [`crate::numerical::LSODE2::statistics::Lsode2NativeStatistics`].
//!
//! The current version still accepts externally computed correction vectors.
//! That is deliberate: we can validate native step orchestration and telemetry
//! before wiring real residual/Jacobian/linear-solve callbacks into the loop.

use super::adams_engine::Lsode2AdamsDcfodeTables;
use super::algorithm::Lsode2SwitchTelemetry;
use super::correction::{
    Lsode2CorrectionController, Lsode2CorrectionError, Lsode2CorrectionStatus,
    Lsode2DstodaCorrectorContext,
};
use super::dcfode::Lsode2BdfDcfodeTables;
use super::dstoda_state::Lsode2IterationMode;
use super::native_executor::{Lsode2NativeCallbackExecutor, Lsode2NativeExecutorError};
use super::state::Lsode2RuntimeStateSnapshot;
use super::statistics::Lsode2NativeStatistics;
use super::step_control::Lsode2RetryDecision;
use super::step_cycle::{
    Lsode2PredictedStep, Lsode2StepCycle, Lsode2StepCycleError, Lsode2StepCycleOutcome,
};
use crate::numerical::BDF::BDF_solver::BdfLinearBackend;
use nalgebra::DVector;

#[derive(Debug, Clone, PartialEq)]
pub enum Lsode2NonlinearDriverError {
    StepCycle(Lsode2StepCycleError),
    Correction(Lsode2CorrectionError),
    Executor(Lsode2NativeExecutorError),
    NoActivePrediction,
}

impl std::fmt::Display for Lsode2NonlinearDriverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StepCycle(err) => write!(f, "{err}"),
            Self::Correction(err) => write!(f, "{err}"),
            Self::Executor(err) => write!(f, "{err}"),
            Self::NoActivePrediction => write!(
                f,
                "LSODE2 nonlinear driver requires begin_step() before correction submission"
            ),
        }
    }
}

impl std::error::Error for Lsode2NonlinearDriverError {}

impl From<Lsode2StepCycleError> for Lsode2NonlinearDriverError {
    fn from(value: Lsode2StepCycleError) -> Self {
        Self::StepCycle(value)
    }
}

impl From<Lsode2CorrectionError> for Lsode2NonlinearDriverError {
    fn from(value: Lsode2CorrectionError) -> Self {
        Self::Correction(value)
    }
}

impl From<Lsode2NativeExecutorError> for Lsode2NonlinearDriverError {
    fn from(value: Lsode2NativeExecutorError) -> Self {
        Self::Executor(value)
    }
}

#[derive(Debug, Clone)]
pub struct Lsode2NonlinearStepDriver {
    cycle: Lsode2StepCycle,
    correction: Lsode2CorrectionController,
    statistics: Lsode2NativeStatistics,
    active_prediction: Option<Lsode2PredictedStep>,
    active_accumulated_correction: Option<Vec<f64>>,
    current_iteration: usize,
    previous_weighted_norm: Option<f64>,
    previous_rate_estimate: Option<f64>,
    previous_rate_max_estimate: Option<f64>,
}

impl Lsode2NonlinearStepDriver {
    pub fn new(cycle: Lsode2StepCycle, correction: Lsode2CorrectionController) -> Self {
        Self {
            cycle,
            correction,
            statistics: Lsode2NativeStatistics::default(),
            active_prediction: None,
            active_accumulated_correction: None,
            current_iteration: 0,
            previous_weighted_norm: None,
            previous_rate_estimate: None,
            previous_rate_max_estimate: None,
        }
    }

    pub fn cycle(&self) -> &Lsode2StepCycle {
        &self.cycle
    }

    pub fn cycle_mut(&mut self) -> &mut Lsode2StepCycle {
        &mut self.cycle
    }

    pub fn correction(&self) -> &Lsode2CorrectionController {
        &self.correction
    }

    pub fn statistics(&self) -> &Lsode2NativeStatistics {
        &self.statistics
    }

    pub fn statistics_mut(&mut self) -> &mut Lsode2NativeStatistics {
        &mut self.statistics
    }

    pub fn reset_cycle_after_method_switch(
        &mut self,
        cycle: Lsode2StepCycle,
        correction: Lsode2CorrectionController,
    ) {
        self.cycle = cycle;
        self.correction = correction;
        self.active_prediction = None;
        self.active_accumulated_correction = None;
        self.current_iteration = 0;
        self.previous_weighted_norm = None;
        self.previous_rate_estimate = None;
        self.previous_rate_max_estimate = None;
    }

    pub fn reset_iteration_memory_after_method_switch(&mut self) {
        self.active_prediction = None;
        self.active_accumulated_correction = None;
        self.current_iteration = 0;
        self.previous_weighted_norm = None;
        self.previous_rate_estimate = None;
        self.previous_rate_max_estimate = None;
    }

    pub fn current_iteration(&self) -> usize {
        self.current_iteration
    }

    pub fn has_active_prediction(&self) -> bool {
        self.active_prediction.is_some()
    }

    pub fn begin_step(&mut self) -> Result<Lsode2PredictedStep, Lsode2NonlinearDriverError> {
        self.current_iteration = 0;
        self.previous_weighted_norm = None;
        self.previous_rate_estimate = None;
        self.previous_rate_max_estimate = None;
        let predicted = self.cycle.predict()?;
        self.statistics.record_prediction();
        self.statistics
            .record_predictor_ipup_trigger(self.cycle.ipup_trigger());
        self.active_accumulated_correction = Some(vec![0.0; predicted.y_pred.len()]);
        self.active_prediction = Some(predicted.clone());
        Ok(predicted)
    }

    pub fn submit_correction(
        &mut self,
        y_candidate: &[f64],
        correction: &[f64],
    ) -> Result<Lsode2StepCycleOutcome, Lsode2NonlinearDriverError> {
        let prediction = self
            .active_prediction
            .as_ref()
            .ok_or(Lsode2NonlinearDriverError::NoActivePrediction)?;
        let accumulated_correction = self
            .active_accumulated_correction
            .as_mut()
            .ok_or(Lsode2NonlinearDriverError::NoActivePrediction)?;
        self.current_iteration += 1;
        for (acc, delta) in accumulated_correction.iter_mut().zip(correction.iter()) {
            *acc += *delta;
        }

        let h_el1_abs = step_h_el1_abs(
            self.cycle.method(),
            self.cycle.state().h(),
            self.cycle.state().order(),
        );
        let roundoff_tolerance = self.correction.dstoda_roundoff_tolerance(y_candidate)?;
        let tesco2_override = step_tesco2(self.cycle.method(), self.cycle.state().order());
        let assessment = self.correction.assess_iteration_with_dstoda_context(
            self.cycle.state().order(),
            y_candidate,
            correction,
            accumulated_correction.as_slice(),
            self.previous_weighted_norm,
            self.previous_rate_estimate,
            self.current_iteration,
            tesco2_override,
            Some(Lsode2DstodaCorrectorContext {
                method_is_adams: self.cycle.method()
                    == super::step_cycle::Lsode2StepMethod::AdamsLike,
                previous_rate_max: self.previous_rate_max_estimate,
                h_el1_abs,
                roundoff_tolerance: Some(roundoff_tolerance),
            }),
        )?;
        self.statistics.record_correction_assessment(&assessment);

        let outcome = match assessment.status {
            Lsode2CorrectionStatus::Converged => {
                self.cycle
                    .record_adams_lipschitz_estimate_from_assessment(&assessment);
                self.cycle.finish_after_converged_correction(
                    prediction.t_trial,
                    y_candidate,
                    accumulated_correction.as_slice(),
                    &assessment.local_error,
                )?
            }
            Lsode2CorrectionStatus::Continue => {
                self.previous_weighted_norm = Some(assessment.weighted_norm);
                self.previous_rate_estimate = assessment.convergence_rate_estimate;
                self.previous_rate_max_estimate = assessment.rate_max_estimate;
                Lsode2StepCycleOutcome::NonlinearContinue { assessment }
            }
            Lsode2CorrectionStatus::Diverged | Lsode2CorrectionStatus::IterationLimitReached => {
                let retry = self.cycle.reject_after_nonlinear_failure()?;
                Lsode2StepCycleOutcome::NonlinearRejected {
                    assessment,
                    retry,
                    state: self.cycle.state().snapshot(),
                }
            }
        };

        self.statistics.record_step_cycle_outcome(&outcome);
        if !matches!(outcome, Lsode2StepCycleOutcome::NonlinearContinue { .. }) {
            self.reset_iteration_state();
        }
        Ok(outcome)
    }

    pub fn compute_and_submit_correction<L>(
        &mut self,
        y_candidate: &DVector<f64>,
        c: f64,
        executor: &mut Lsode2NativeCallbackExecutor<L>,
    ) -> Result<Lsode2StepCycleOutcome, Lsode2NonlinearDriverError>
    where
        L: BdfLinearBackend,
    {
        self.compute_and_submit_correction_with_refresh_policy(y_candidate, c, executor, false)
    }

    pub fn compute_and_submit_correction_with_refresh_policy<L>(
        &mut self,
        y_candidate: &DVector<f64>,
        c: f64,
        executor: &mut Lsode2NativeCallbackExecutor<L>,
        force_refresh: bool,
    ) -> Result<Lsode2StepCycleOutcome, Lsode2NonlinearDriverError>
    where
        L: BdfLinearBackend,
    {
        let prediction = self
            .active_prediction
            .as_ref()
            .ok_or(Lsode2NonlinearDriverError::NoActivePrediction)?;
        let predictor_refresh_requested = self.cycle.jacobian_update_request().is_requested();
        let effective_force_refresh = force_refresh || predictor_refresh_requested;
        let correction = if self.cycle.iteration_mode() == Lsode2IterationMode::Functional {
            executor.compute_functional_correction(
                prediction.t_trial,
                y_candidate,
                &mut self.statistics,
            )?
        } else {
            let had_current_linearization =
                executor.has_current_linearization(prediction.t_trial, c);
            let correction = executor.compute_newton_correction_with_refresh_policy(
                prediction.t_trial,
                y_candidate,
                c,
                &mut self.statistics,
                effective_force_refresh,
            )?;
            let has_current_linearization =
                executor.has_current_linearization(prediction.t_trial, c);
            if !had_current_linearization && has_current_linearization {
                self.cycle.mark_jacobian_current();
            }
            correction
        };
        self.submit_correction(y_candidate.as_slice(), correction.as_slice())
    }

    pub fn compute_apply_and_submit_correction<L>(
        &mut self,
        y_candidate: &mut DVector<f64>,
        c: f64,
        executor: &mut Lsode2NativeCallbackExecutor<L>,
    ) -> Result<Lsode2StepCycleOutcome, Lsode2NonlinearDriverError>
    where
        L: BdfLinearBackend,
    {
        self.compute_apply_and_submit_correction_with_refresh_policy(
            y_candidate,
            c,
            executor,
            false,
        )
    }

    pub fn compute_apply_and_submit_correction_with_refresh_policy<L>(
        &mut self,
        y_candidate: &mut DVector<f64>,
        c: f64,
        executor: &mut Lsode2NativeCallbackExecutor<L>,
        force_refresh: bool,
    ) -> Result<Lsode2StepCycleOutcome, Lsode2NonlinearDriverError>
    where
        L: BdfLinearBackend,
    {
        let prediction = self
            .active_prediction
            .as_ref()
            .ok_or(Lsode2NonlinearDriverError::NoActivePrediction)?;
        let predictor_refresh_requested = self.cycle.jacobian_update_request().is_requested();
        let effective_force_refresh = force_refresh || predictor_refresh_requested;
        let correction_y = if self.cycle.iteration_mode() == Lsode2IterationMode::Functional {
            executor.compute_functional_correction(
                prediction.t_trial,
                y_candidate,
                &mut self.statistics,
            )?
        } else {
            let had_current_linearization =
                executor.has_current_linearization(prediction.t_trial, c);
            let correction = executor.compute_newton_correction_with_refresh_policy(
                prediction.t_trial,
                y_candidate,
                c,
                &mut self.statistics,
                effective_force_refresh,
            )?;
            let has_current_linearization =
                executor.has_current_linearization(prediction.t_trial, c);
            if !had_current_linearization && has_current_linearization {
                self.cycle.mark_jacobian_current();
            }
            correction
        };
        *y_candidate += &correction_y;

        // DSTODA mirroring:
        // nonlinear correction convergence (`DEL`, `DSM`, error test ACOR/TESCO)
        // is defined in ACOR-space, while executor returns delta in Y-space.
        // For Adams/BDF predictor-corrector we have:
        //   Y = YH(:,1) + EL(1) * ACOR  =>  dY = EL(1) * dACOR
        // therefore assessment/input correction must be scaled by 1/EL(1).
        let el1 = step_el1(self.cycle.method(), self.cycle.state().order())
            .ok_or(Lsode2NonlinearDriverError::NoActivePrediction)?;
        let correction_acor = correction_y
            .iter()
            .map(|value| *value / el1)
            .collect::<Vec<_>>();
        self.submit_correction(y_candidate.as_slice(), correction_acor.as_slice())
    }

    pub fn reject_after_nonlinear_failure(
        &mut self,
    ) -> Result<(Lsode2RetryDecision, Lsode2RuntimeStateSnapshot), Lsode2NonlinearDriverError> {
        let retry = self.cycle.reject_after_nonlinear_failure()?;
        self.statistics.record_native_step_reject_nonlinear();
        self.statistics.record_native_nonlinear_diverged();
        self.statistics.record_jacobian_refresh_request();
        let snapshot = self.cycle.state().snapshot();
        self.reset_iteration_state();
        Ok((retry, snapshot))
    }

    pub fn retry_after_stale_jacobian_nonlinear_failure(
        &mut self,
    ) -> Result<(Lsode2RetryDecision, Lsode2RuntimeStateSnapshot), Lsode2NonlinearDriverError> {
        let retry = self.cycle.retry_after_stale_jacobian_nonlinear_failure()?;
        self.statistics.record_native_nonlinear_diverged();
        self.statistics.record_native_stale_jacobian_retry();
        self.statistics.record_jacobian_refresh_request();
        let snapshot = self.cycle.state().snapshot();
        self.reset_iteration_state();
        Ok((retry, snapshot))
    }

    pub fn switch_telemetry(&self, stiffness_ratio: Option<f64>) -> Lsode2SwitchTelemetry {
        self.cycle.switch_telemetry(stiffness_ratio)
    }

    fn reset_iteration_state(&mut self) {
        self.active_prediction = None;
        self.active_accumulated_correction = None;
        self.current_iteration = 0;
        self.previous_weighted_norm = None;
        self.previous_rate_estimate = None;
        self.previous_rate_max_estimate = None;
    }
}

fn step_h_el1_abs(
    method: super::step_cycle::Lsode2StepMethod,
    h: f64,
    order: usize,
) -> Option<f64> {
    let h_abs = h.abs();
    if !(h_abs.is_finite() && h_abs > 0.0) {
        return None;
    }
    let el1_abs = match method {
        super::step_cycle::Lsode2StepMethod::BdfLike => Lsode2BdfDcfodeTables::default()
            .order(order)
            .ok()
            .map(|c| c.el[0].abs()),
        super::step_cycle::Lsode2StepMethod::AdamsLike => Lsode2AdamsDcfodeTables::default()
            .order(order)
            .ok()
            .map(|c| c.el[1].abs()),
    }?;
    if el1_abs.is_finite() && el1_abs > 0.0 {
        Some(h_abs * el1_abs)
    } else {
        None
    }
}

fn step_el1(method: super::step_cycle::Lsode2StepMethod, order: usize) -> Option<f64> {
    let el1 = match method {
        super::step_cycle::Lsode2StepMethod::BdfLike => Lsode2BdfDcfodeTables::default()
            .order(order)
            .ok()
            .map(|c| c.el[0]),
        super::step_cycle::Lsode2StepMethod::AdamsLike => Lsode2AdamsDcfodeTables::default()
            .order(order)
            .ok()
            .map(|c| c.el[1]),
    }?;
    if el1.is_finite() && el1 != 0.0 {
        Some(el1)
    } else {
        None
    }
}

fn step_tesco2(method: super::step_cycle::Lsode2StepMethod, order: usize) -> Option<f64> {
    let tesco2 = match method {
        super::step_cycle::Lsode2StepMethod::BdfLike => Lsode2BdfDcfodeTables::default()
            .order(order)
            .ok()
            .map(|c| c.tesco2),
        super::step_cycle::Lsode2StepMethod::AdamsLike => Lsode2AdamsDcfodeTables::default()
            .order(order)
            .ok()
            .map(|c| c.tesco2),
    }?;
    if tesco2.is_finite() && tesco2 > 0.0 {
        Some(tesco2)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::BDF::BDF_solver::BdfJacobian;
    use crate::numerical::LSODE2::step_cycle::Lsode2StepMethod;
    use crate::numerical::LSODE2::{
        FaerSparseBdfLinearBackend, Lsode2CorrectionControlConfig, Lsode2ErrorControlConfig,
        Lsode2ErrorController, Lsode2Icf, Lsode2JacobianCurrency, Lsode2RedoStage,
        Lsode2RetryAction, Lsode2RuntimeState, Lsode2StepControlConfig, Lsode2Tolerance,
    };
    use faer::sparse::Triplet;

    fn make_driver() -> Lsode2NonlinearStepDriver {
        let cycle = Lsode2StepCycle::new(
            Lsode2RuntimeState::new(0.0, &[1.0], 0.1, 3, Lsode2StepControlConfig::default())
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
        Lsode2NonlinearStepDriver::new(cycle, correction)
    }

    #[test]
    fn nonlinear_driver_begin_step_records_prediction() {
        let mut driver = make_driver();

        let predicted = driver.begin_step().unwrap();

        assert_eq!(predicted.t_trial, 0.1);
        assert!(driver.has_active_prediction());
        assert_eq!(driver.statistics().native_step_attempts, 1);
    }

    #[test]
    fn nonlinear_driver_can_continue_then_accept() {
        let mut driver = make_driver();
        driver.begin_step().unwrap();

        let continue_outcome = driver.submit_correction(&[0.905], &[1.0e-3]).unwrap();
        match continue_outcome {
            Lsode2StepCycleOutcome::NonlinearContinue { assessment } => {
                assert_eq!(assessment.status, Lsode2CorrectionStatus::Continue);
            }
            other => panic!("expected nonlinear continue, got {other:?}"),
        }
        assert_eq!(driver.current_iteration(), 1);
        assert!(driver.has_active_prediction());

        let accepted = driver.submit_correction(&[0.905], &[1.0e-5]).unwrap();
        match accepted {
            Lsode2StepCycleOutcome::Accepted { state, .. } => {
                assert_eq!(state.accepted_steps, 1);
            }
            other => panic!("expected accepted step, got {other:?}"),
        }

        assert_eq!(driver.current_iteration(), 0);
        assert!(!driver.has_active_prediction());
        assert_eq!(driver.statistics().native_step_attempts, 1);
        assert_eq!(driver.statistics().native_step_accepts, 1);
        assert_eq!(driver.statistics().native_nonlinear_continue_count, 1);
        assert_eq!(driver.statistics().native_nonlinear_converged_count, 1);
    }

    #[test]
    fn nonlinear_driver_rejects_diverged_correction_and_records_refresh() {
        let mut driver = make_driver();
        driver.begin_step().unwrap();
        let continued = driver.submit_correction(&[0.905], &[1.0e-3]).unwrap();
        assert!(matches!(
            continued,
            Lsode2StepCycleOutcome::NonlinearContinue { .. }
        ));

        let rejected = driver.submit_correction(&[0.905], &[1.0e-2]).unwrap();
        match rejected {
            Lsode2StepCycleOutcome::NonlinearRejected {
                assessment, state, ..
            } => {
                assert_eq!(assessment.status, Lsode2CorrectionStatus::Diverged);
                assert_eq!(state.convergence_failures, 1);
            }
            other => panic!("expected nonlinear rejection, got {other:?}"),
        }

        assert_eq!(driver.statistics().native_step_rejects_nonlinear, 1);
        assert_eq!(driver.statistics().native_nonlinear_diverged_count, 1);
        assert_eq!(driver.statistics().native_jacobian_refresh_requests, 1);
        assert!(!driver.has_active_prediction());
    }

    #[test]
    fn nonlinear_driver_supports_direct_failure_before_correction() {
        let mut driver = make_driver();
        driver.begin_step().unwrap();

        let (_retry, snapshot) = driver.reject_after_nonlinear_failure().unwrap();

        assert_eq!(snapshot.convergence_failures, 1);
        assert_eq!(driver.statistics().native_step_rejects_nonlinear, 1);
        assert_eq!(driver.statistics().native_nonlinear_diverged_count, 1);
        assert_eq!(driver.statistics().native_jacobian_refresh_requests, 1);
        assert!(!driver.has_active_prediction());
    }

    #[test]
    fn nonlinear_driver_supports_stale_jacobian_retry_without_step_reject() {
        let mut driver = make_driver();
        driver.begin_step().unwrap();

        let (retry, snapshot) = driver
            .retry_after_stale_jacobian_nonlinear_failure()
            .unwrap();

        assert_eq!(retry.h_new, 0.1);
        assert_eq!(retry.shrink_factor, 1.0);
        assert_eq!(snapshot.rejected_steps, 0);
        assert_eq!(snapshot.convergence_failures, 0);
        assert_eq!(driver.statistics().native_step_rejects_nonlinear, 0);
        assert_eq!(driver.statistics().native_nonlinear_diverged_count, 1);
        assert_eq!(driver.statistics().native_stale_jacobian_retry_count, 1);
        assert_eq!(driver.statistics().native_jacobian_refresh_requests, 1);
        assert!(!driver.has_active_prediction());
    }

    #[test]
    fn nonlinear_driver_submit_path_tracks_one_shot_stale_jacobian_refresh_retry() {
        let mut driver = make_driver();
        driver.begin_step().unwrap();

        let continued = driver.submit_correction(&[0.905], &[1.0e-3]).unwrap();
        assert!(matches!(
            continued,
            Lsode2StepCycleOutcome::NonlinearContinue { .. }
        ));

        driver.cycle_mut().mark_jacobian_stale();
        let rejected = driver.submit_correction(&[0.905], &[1.0e-2]).unwrap();
        match rejected {
            Lsode2StepCycleOutcome::NonlinearRejected {
                retry,
                state,
                assessment,
            } => {
                assert_eq!(assessment.status, Lsode2CorrectionStatus::Diverged);
                assert_eq!(retry.action, Lsode2RetryAction::RetryWithJacobianRefresh);
                assert_eq!(state.convergence_failures, 0);
                assert_eq!(state.rejected_steps, 0);
            }
            other => panic!(
                "expected nonlinear rejection on stale-J path with same-step refresh, got {other:?}"
            ),
        }

        assert_eq!(driver.cycle().icf(), Lsode2Icf::RefreshRequested);
        assert_eq!(
            driver.cycle().redo_stage(),
            Lsode2RedoStage::CorrectorRefreshSameStep
        );
        assert_eq!(driver.statistics().native_stale_jacobian_retry_count, 1);
        assert_eq!(driver.statistics().native_step_rejects_nonlinear, 0);
        assert_eq!(driver.statistics().native_nonlinear_diverged_count, 1);
        assert_eq!(driver.statistics().native_jacobian_refresh_requests, 1);
        assert!(!driver.has_active_prediction());
    }

    #[test]
    fn nonlinear_driver_submit_path_transitions_from_icf1_refresh_to_icf2_retract() {
        let mut driver = make_driver();

        // Attempt #1: converge-rate failure on stale Jacobian -> ICF=1 same-step refresh path.
        driver.begin_step().unwrap();
        let cont1 = driver.submit_correction(&[0.905], &[1.0e-3]).unwrap();
        assert!(matches!(
            cont1,
            Lsode2StepCycleOutcome::NonlinearContinue { .. }
        ));
        driver.cycle_mut().mark_jacobian_stale();
        let first_reject = driver.submit_correction(&[0.905], &[1.0e-2]).unwrap();
        match first_reject {
            Lsode2StepCycleOutcome::NonlinearRejected { retry, state, .. } => {
                assert_eq!(retry.action, Lsode2RetryAction::RetryWithJacobianRefresh);
                assert_eq!(state.convergence_failures, 0);
                assert_eq!(state.rejected_steps, 0);
            }
            other => panic!("expected first nonlinear rejection (ICF=1 refresh), got {other:?}"),
        }
        assert_eq!(driver.cycle().icf(), Lsode2Icf::RefreshRequested);
        assert_eq!(
            driver.cycle().redo_stage(),
            Lsode2RedoStage::CorrectorRefreshSameStep
        );

        // Attempt #2: after refresh path was used, next convergence failure must retract/shrink.
        // Mark J current to mirror refreshed-linearization path consumed before the next attempt.
        driver.cycle_mut().mark_jacobian_current();
        driver.begin_step().unwrap();
        let cont2 = driver.submit_correction(&[0.905], &[1.0e-3]).unwrap();
        assert!(matches!(
            cont2,
            Lsode2StepCycleOutcome::NonlinearContinue { .. }
        ));
        let second_reject = driver.submit_correction(&[0.905], &[1.0e-2]).unwrap();
        match second_reject {
            Lsode2StepCycleOutcome::NonlinearRejected { retry, state, .. } => {
                // In our controller, nonlinear failure retries keep the refresh intent flag.
                // Distinguish ICF=2 by shrink/reject counters and cycle flags, not action enum.
                assert_eq!(retry.action, Lsode2RetryAction::RetryWithJacobianRefresh);
                assert!(retry.shrink_factor < 1.0);
                assert_eq!(state.convergence_failures, 1);
                assert_eq!(state.rejected_steps, 1);
            }
            other => panic!("expected second nonlinear rejection (ICF=2 retract), got {other:?}"),
        }
        assert_eq!(driver.cycle().icf(), Lsode2Icf::RefreshDidNotRecover);
        assert_eq!(
            driver.cycle().redo_stage(),
            Lsode2RedoStage::CorrectorFailureRetry
        );

        // Telemetry parity: first rejection is stale-refresh retry, second is true nonlinear step reject.
        assert_eq!(driver.statistics().native_stale_jacobian_retry_count, 1);
        assert_eq!(driver.statistics().native_step_rejects_nonlinear, 1);
        assert_eq!(driver.statistics().native_nonlinear_diverged_count, 2);
        assert!(driver.statistics().native_jacobian_refresh_requests >= 2);
        assert!(!driver.has_active_prediction());
    }

    #[test]
    fn nonlinear_driver_requires_prediction_before_correction() {
        let mut driver = make_driver();

        let err = driver.submit_correction(&[0.905], &[1.0e-5]).unwrap_err();

        assert!(matches!(
            err,
            Lsode2NonlinearDriverError::NoActivePrediction
        ));
    }

    #[test]
    fn nonlinear_driver_can_compute_and_submit_correction_via_native_executor() {
        let mut driver = make_driver();
        driver.begin_step().unwrap();
        driver.cycle_mut().mark_jacobian_stale();
        let mut executor = Lsode2NativeCallbackExecutor::new(
            |_, y: &DVector<f64>| DVector::from_vec(vec![y[0] - 0.905]),
            |_, _| BdfJacobian::SparseTriplets {
                n: 1,
                triplets: vec![Triplet::new(0, 0, 0.0)],
            },
            FaerSparseBdfLinearBackend::default(),
        );

        let outcome = driver
            .compute_and_submit_correction(
                &DVector::from_vec(vec![0.905 + 1.0e-5]),
                0.0,
                &mut executor,
            )
            .unwrap();

        match outcome {
            Lsode2StepCycleOutcome::Accepted { state, .. } => {
                assert_eq!(state.accepted_steps, 1);
            }
            other => panic!("expected accepted step, got {other:?}"),
        }
        assert_eq!(driver.statistics().native_residual_calls, 1);
        assert_eq!(driver.statistics().native_jacobian_calls, 1);
        assert_eq!(driver.statistics().native_linear_solve_calls, 1);
        assert_eq!(driver.statistics().native_step_accepts, 1);
        assert_eq!(
            driver.cycle().jacobian_currency(),
            Lsode2JacobianCurrency::Stale
        );
    }

    #[test]
    fn nonlinear_driver_honors_dstoda_predictor_refresh_request_on_first_newton_pass() {
        use std::cell::RefCell;
        use std::rc::Rc;

        let mut driver = make_driver();
        let jacobian_calls = Rc::new(RefCell::new(0usize));
        let jacobian_calls_handle = Rc::clone(&jacobian_calls);
        let mut executor = Lsode2NativeCallbackExecutor::new(
            |_, y: &DVector<f64>| DVector::from_vec(vec![y[0] - 0.905]),
            move |_, _| {
                *jacobian_calls_handle.borrow_mut() += 1;
                BdfJacobian::SparseTriplets {
                    n: 1,
                    triplets: vec![Triplet::new(0, 0, 0.0)],
                }
            },
            FaerSparseBdfLinearBackend::default(),
        );

        // Attempt #1 builds the cached linearization.
        driver.begin_step().unwrap();
        let _ = driver
            .compute_and_submit_correction(
                &DVector::from_vec(vec![0.905 + 1.0e-3]),
                0.0,
                &mut executor,
            )
            .unwrap();
        assert_eq!(*jacobian_calls.borrow(), 1);

        // Mirror DSTODA ICF=1 path: same-step refresh requested after convergence trouble.
        let _ = driver
            .retry_after_stale_jacobian_nonlinear_failure()
            .expect("stale-J nonlinear retry should request predictor refresh");
        assert!(driver.cycle().jacobian_update_request().is_requested());

        // Attempt #2 must force a fresh Jacobian on first Newton pass,
        // even though (t, c) stayed the same and executor has cached factors.
        driver.begin_step().unwrap();
        let _ = driver
            .compute_and_submit_correction(
                &DVector::from_vec(vec![0.905 + 1.0e-3]),
                0.0,
                &mut executor,
            )
            .unwrap();
        assert_eq!(*jacobian_calls.borrow(), 2);
    }

    #[test]
    fn nonlinear_driver_can_apply_correction_until_acceptance() {
        let mut driver = make_driver();
        let predicted = driver.begin_step().unwrap();
        let mut candidate = DVector::from_vec(predicted.y_pred.clone());
        let mut executor = Lsode2NativeCallbackExecutor::new(
            |_, y: &DVector<f64>| DVector::from_vec(vec![y[0] - 0.905]),
            |_, _| BdfJacobian::SparseTriplets {
                n: 1,
                triplets: vec![Triplet::new(0, 0, 0.0)],
            },
            FaerSparseBdfLinearBackend::default(),
        );

        let first = driver
            .compute_apply_and_submit_correction(&mut candidate, 0.0, &mut executor)
            .unwrap();
        assert!(matches!(
            first,
            Lsode2StepCycleOutcome::NonlinearContinue { .. }
                | Lsode2StepCycleOutcome::Accepted { .. }
        ));

        if matches!(first, Lsode2StepCycleOutcome::NonlinearContinue { .. }) {
            let mut terminal_outcome = None;
            for _ in 0..3 {
                let next = driver
                    .compute_apply_and_submit_correction(&mut candidate, 0.0, &mut executor)
                    .unwrap();
                if matches!(
                    next,
                    Lsode2StepCycleOutcome::Accepted { .. }
                        | Lsode2StepCycleOutcome::Rejected { .. }
                ) {
                    terminal_outcome = Some(next);
                    break;
                }
                assert!(
                    matches!(next, Lsode2StepCycleOutcome::NonlinearContinue { .. }),
                    "expected additional Newton work or a terminal LSODE-style step outcome, got {next:?}"
                );
            }
            match terminal_outcome {
                Some(Lsode2StepCycleOutcome::Accepted { .. }) => {
                    assert_eq!(driver.statistics().native_step_accepts, 1);
                }
                Some(Lsode2StepCycleOutcome::Rejected { .. }) => {
                    assert_eq!(driver.statistics().native_step_rejects_error_test, 1);
                }
                Some(other) => panic!("unexpected terminal nonlinear-driver outcome: {other:?}"),
                None => panic!(
                    "native nonlinear driver should reach either accept or error-test reject within a few LSODE-style correction iterations"
                ),
            }
        }

        assert!((candidate[0] - 0.905).abs() < 1.0e-8);
        assert!(driver.statistics().native_residual_calls >= 1);
        assert!(driver.statistics().native_jacobian_calls >= 1);
        assert!(driver.statistics().native_linear_solve_calls >= 1);
    }

    #[test]
    fn nonlinear_driver_functional_iteration_uses_residual_only_path() {
        use std::cell::RefCell;
        use std::rc::Rc;

        let mut driver = make_driver();
        driver
            .cycle_mut()
            .set_iteration_mode(Lsode2IterationMode::Functional);
        let predicted = driver.begin_step().unwrap();
        let mut candidate = DVector::from_vec(predicted.y_pred.clone());

        let jacobian_calls = Rc::new(RefCell::new(0usize));
        let jacobian_calls_handle = Rc::clone(&jacobian_calls);
        let mut executor = Lsode2NativeCallbackExecutor::new(
            |_, y: &DVector<f64>| DVector::from_vec(vec![y[0] - 0.905]),
            move |_, _| {
                *jacobian_calls_handle.borrow_mut() += 1;
                BdfJacobian::SparseTriplets {
                    n: 1,
                    triplets: vec![Triplet::new(0, 0, 0.0)],
                }
            },
            FaerSparseBdfLinearBackend::default(),
        );

        let outcome = driver
            .compute_apply_and_submit_correction(&mut candidate, 0.0, &mut executor)
            .expect("functional iteration should compute correction from residual path");

        assert!(matches!(
            outcome,
            Lsode2StepCycleOutcome::NonlinearContinue { .. }
                | Lsode2StepCycleOutcome::Accepted { .. }
        ));
        assert_eq!(driver.statistics().native_residual_calls, 1);
        assert_eq!(driver.statistics().native_jacobian_calls, 0);
        assert_eq!(driver.statistics().native_linear_solve_calls, 0);
        assert_eq!(*jacobian_calls.borrow(), 0);
    }

    #[test]
    fn nonlinear_driver_adams_forces_second_iteration_when_not_roundoff() {
        let cycle = Lsode2StepCycle::new_with_method(
            Lsode2RuntimeState::new(0.0, &[1.0], 0.1, 3, Lsode2StepControlConfig::default())
                .unwrap(),
            Lsode2ErrorController::new(
                Lsode2Tolerance::scalar(1.0e-3, 1.0e-6),
                Lsode2ErrorControlConfig::default(),
            )
            .unwrap(),
            super::super::step_cycle::Lsode2StepMethod::AdamsLike,
        );
        let correction = Lsode2CorrectionController::new(
            Lsode2Tolerance::scalar(1.0e-3, 1.0e-6),
            Lsode2CorrectionControlConfig::default(),
        )
        .unwrap();
        let mut driver = Lsode2NonlinearStepDriver::new(cycle, correction);
        driver.begin_step().unwrap();

        // This increment is small enough to converge by the raw dcon test,
        // but still above roundoff-level threshold.
        let outcome = driver.submit_correction(&[0.905], &[1.0e-10]).unwrap();
        match outcome {
            Lsode2StepCycleOutcome::NonlinearContinue { assessment } => {
                assert_eq!(assessment.status, Lsode2CorrectionStatus::Continue);
            }
            other => panic!("expected forced Adams continue on first pass, got {other:?}"),
        }
        assert_eq!(driver.current_iteration(), 1);
        assert!(driver.has_active_prediction());
    }

    #[test]
    fn nonlinear_driver_adams_high_order_submit_uses_adams_tesco2_without_bdf_overflow() {
        let mut cycle = Lsode2StepCycle::new_with_method(
            Lsode2RuntimeState::new(0.0, &[1.0], 0.1, 12, Lsode2StepControlConfig::default())
                .unwrap(),
            Lsode2ErrorController::new(
                Lsode2Tolerance::scalar(1.0e-3, 1.0e-6),
                Lsode2ErrorControlConfig::default(),
            )
            .unwrap(),
            Lsode2StepMethod::AdamsLike,
        );
        cycle.state_mut().set_order(6).unwrap();

        let correction = Lsode2CorrectionController::new(
            Lsode2Tolerance::scalar(1.0e-3, 1.0e-6),
            Lsode2CorrectionControlConfig::default(),
        )
        .unwrap();
        let mut driver = Lsode2NonlinearStepDriver::new(cycle, correction);
        driver.begin_step().unwrap();

        let outcome = driver.submit_correction(&[0.905], &[1.0e-6]).unwrap();
        assert!(
            matches!(outcome, Lsode2StepCycleOutcome::NonlinearContinue { .. }),
            "Adams q=6 first pass should stay in DSTODA continue-path and must not fail via BDF tesco2 lookup"
        );
    }
}
