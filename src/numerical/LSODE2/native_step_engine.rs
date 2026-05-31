//! One real solver-owned native LSODE2 step attempt.
//!
//! This layer sits between the exploratory preflight loop and the lower-level
//! Newton orchestration pieces:
//! - residual-only generated backend preparation,
//! - native sparse/banded Jacobian callbacks,
//! - nonlinear step driver,
//! - timed callback executor.
//!
//! The current step semantics run through LSODE2-native DSTODA-like
//! predictor/corrector choreography (for both BDF-like and Adams-like families)
//! on top of real residual/Jacobian/linear-solve callbacks.
//! This module focuses on one-step DSTODA choreography and callback execution.

use super::adams_engine::Lsode2AdamsDcfodeTables;
use super::algorithm::{Lsode2ControllerMode, Lsode2SwitchTelemetry};
use super::config::{
    Lsode2JacobianBackend, Lsode2LinearSolverBackend, Lsode2ProblemConfig,
    Lsode2ResidualJacobianSource, Lsode2SymbolicAssemblyBackend, Lsode2SymbolicExecutionMode,
};
use super::correction::{Lsode2CorrectionControlConfig, Lsode2CorrectionController};
use super::dcfode::Lsode2BdfDcfodeTables;
use super::dstoda_state::{
    Lsode2Icf, Lsode2Ipup, Lsode2IpupTrigger, Lsode2Iredo, Lsode2Iret, Lsode2JacobianCurrency,
    Lsode2Kflag, Lsode2RedoStage,
};
use super::error_control::{Lsode2ErrorControlConfig, Lsode2ErrorController};
use super::history::Lsode2Tolerance;
use super::linear_backends::{
    DenseLuBdfLinearBackend, FaerSparseBdfLinearBackend, FaithfulBandedBdfLinearBackend,
};
use super::native_executor::{jacobian_abs_max, Lsode2NativeCallbackExecutor};
use super::native_jacobian::{
    compile_native_sparse_aot_jacobian_with_parameter_handle,
    compile_native_symbolic_jacobian_with_parameter_handle, NativeJacobianStorage,
};
use super::nonlinear_driver::Lsode2NonlinearStepDriver;
use super::state::{Lsode2RuntimeState, Lsode2RuntimeStateSnapshot};
use super::statistics::Lsode2NativeStatistics;
use super::step_control::{Lsode2RetryAction, Lsode2StepControlConfig};
use super::step_cycle::{
    Lsode2PredictedStep, Lsode2StepCycle, Lsode2StepCycleOutcome, Lsode2StepMethod,
};
use crate::numerical::BDF::common::{norm, scale_func, NumberOrVec};
use crate::numerical::BDF::BDF_solver::{BdfJacobian, BdfLinearBackend};
use crate::somelinalg::banded::storage::Banded;
use crate::symbolic::symbolic_ivp::{
    IvpBackendError, IvpSymbolicAssemblyBackend, SymbolicIvpProblemOptions,
};
use crate::symbolic::symbolic_ivp_generated::prepare_generated_symbolic_ivp_residual_problem;
use nalgebra::{DMatrix, DVector};
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

type NativeResidualFn = dyn Fn(f64, &DVector<f64>) -> DVector<f64>;

#[derive(Debug, Clone)]
struct NativeStepResidualContext {
    y_pred: DVector<f64>,
    yh2: DVector<f64>,
    h_trial: f64,
    el1: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lsode2NativeStepMethod {
    BdfLike,
    AdamsLike,
}

impl Lsode2NativeStepMethod {
    fn max_order(self, config: &Lsode2ProblemConfig) -> usize {
        match self {
            Self::BdfLike => config.controller.max_bdf_order,
            Self::AdamsLike => config.controller.max_adams_order,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Lsode2NativeStepAttemptReport {
    pub predicted: Lsode2PredictedStep,
    pub outcome: Lsode2StepCycleOutcome,
    pub iterations: usize,
    pub retry_count: usize,
    pub jacobian_refresh_retry_count: usize,
    pub telemetry: Lsode2SwitchTelemetry,
    pub predictor_jcur: Lsode2JacobianCurrency,
    pub predictor_ipup: Lsode2Ipup,
    pub predictor_ipup_trigger: Lsode2IpupTrigger,
    pub jcur: Lsode2JacobianCurrency,
    pub ipup: Lsode2Ipup,
    pub ipup_trigger: Lsode2IpupTrigger,
    pub kflag: Lsode2Kflag,
    pub kflag_code: i32,
    pub icf: Lsode2Icf,
    pub iret: Lsode2Iret,
    pub redo_stage: Lsode2RedoStage,
    pub iredo: Lsode2Iredo,
    pub ialth: usize,
}

impl Lsode2NativeStepAttemptReport {
    pub fn accepted(&self) -> bool {
        matches!(self.outcome, Lsode2StepCycleOutcome::Accepted { .. })
    }

    pub fn accepted_t(&self) -> Option<f64> {
        match self.outcome {
            Lsode2StepCycleOutcome::Accepted { t_new, .. } => Some(t_new),
            _ => None,
        }
    }

    pub fn outcome_label(&self) -> &'static str {
        match self.outcome {
            Lsode2StepCycleOutcome::Accepted { .. } => "accepted",
            Lsode2StepCycleOutcome::Rejected { .. } => "rejected_error_test",
            Lsode2StepCycleOutcome::NonlinearContinue { .. } => "nonlinear_continue",
            Lsode2StepCycleOutcome::NonlinearRejected { .. } => "rejected_nonlinear",
        }
    }
}

fn should_force_refresh_on_first_correction(
    retry_refresh_requested: bool,
    predictor_ipup: Lsode2Ipup,
) -> bool {
    retry_refresh_requested || predictor_ipup.needs_update()
}

pub enum Lsode2NativeStepEngine {
    Dense(Box<Lsode2NativeStepEngineImpl<DenseLuBdfLinearBackend>>),
    Sparse(Box<Lsode2NativeStepEngineImpl<FaerSparseBdfLinearBackend>>),
    Banded(Box<Lsode2NativeStepEngineImpl<FaithfulBandedBdfLinearBackend>>),
}

impl Lsode2NativeStepEngine {
    pub fn from_problem_config(
        config: &Lsode2ProblemConfig,
    ) -> Result<Option<Self>, IvpBackendError> {
        Self::from_problem_config_with_method(config, Lsode2NativeStepMethod::BdfLike)
    }

    pub fn from_problem_config_with_method(
        config: &Lsode2ProblemConfig,
        method: Lsode2NativeStepMethod,
    ) -> Result<Option<Self>, IvpBackendError> {
        if !matches!(
            config.backend.jacobian_backend,
            Lsode2JacobianBackend::SymbolicGenerated
                | Lsode2JacobianBackend::AnalyticClosure
                | Lsode2JacobianBackend::FiniteDifference
        ) {
            return Err(IvpBackendError::GeneratedBackendFailure {
                message:
                    "LSODE2 native step engine currently supports symbolic-generated and analytical Jacobians only"
                        .to_string(),
            });
        }

        match config.backend.linear_solver_backend {
            Lsode2LinearSolverBackend::Dense => Ok(Some(Self::Dense(Box::new(
                Lsode2NativeStepEngineImpl::from_problem_config(
                    config,
                    method,
                    DenseLuBdfLinearBackend,
                    NativeJacobianStorage::Dense,
                )?,
            )))),
            Lsode2LinearSolverBackend::SparseFaer => Ok(Some(Self::Sparse(Box::new(
                Lsode2NativeStepEngineImpl::from_problem_config(
                    config,
                    method,
                    FaerSparseBdfLinearBackend::default(),
                    NativeJacobianStorage::SparseTriplets,
                )?,
            )))),
            Lsode2LinearSolverBackend::BandedFaithful => Ok(Some(Self::Banded(Box::new(
                Lsode2NativeStepEngineImpl::from_problem_config(
                    config,
                    method,
                    FaithfulBandedBdfLinearBackend::default(),
                    NativeJacobianStorage::Banded,
                )?,
            )))),
        }
    }

    pub fn step_once(&mut self) -> Result<Lsode2NativeStepAttemptReport, IvpBackendError> {
        match self {
            Self::Dense(engine) => engine.step_once(),
            Self::Sparse(engine) => engine.step_once(),
            Self::Banded(engine) => engine.step_once(),
        }
    }

    pub fn statistics(&self) -> &Lsode2NativeStatistics {
        match self {
            Self::Dense(engine) => engine.statistics(),
            Self::Sparse(engine) => engine.statistics(),
            Self::Banded(engine) => engine.statistics(),
        }
    }

    pub fn state_snapshot(&self) -> Lsode2RuntimeStateSnapshot {
        match self {
            Self::Dense(engine) => engine.state_snapshot(),
            Self::Sparse(engine) => engine.state_snapshot(),
            Self::Banded(engine) => engine.state_snapshot(),
        }
    }

    pub fn current_solution(&self) -> Vec<f64> {
        match self {
            Self::Dense(engine) => engine.current_solution(),
            Self::Sparse(engine) => engine.current_solution(),
            Self::Banded(engine) => engine.current_solution(),
        }
    }

    pub fn clamp_step_to_t_bound(&mut self, t_bound: f64) -> Result<(), IvpBackendError> {
        match self {
            Self::Dense(engine) => engine.clamp_step_to_t_bound(t_bound),
            Self::Sparse(engine) => engine.clamp_step_to_t_bound(t_bound),
            Self::Banded(engine) => engine.clamp_step_to_t_bound(t_bound),
        }
    }

    pub fn switch_method(
        &mut self,
        config: &Lsode2ProblemConfig,
        method: Lsode2NativeStepMethod,
    ) -> Result<(), IvpBackendError> {
        match self {
            Self::Dense(engine) => engine.switch_method(config, method),
            Self::Sparse(engine) => engine.switch_method(config, method),
            Self::Banded(engine) => engine.switch_method(config, method),
        }
    }
}

struct Lsode2NativeStepEngineImpl<L> {
    residual: Rc<NativeResidualFn>,
    jacobian: Rc<RefCell<Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian>>>,
    linear_backend: L,
    driver: Lsode2NonlinearStepDriver,
    fallback_stiffness_probe_interval: Option<usize>,
}

impl<L> Lsode2NativeStepEngineImpl<L>
where
    L: BdfLinearBackend + Clone + 'static,
{
    fn from_problem_config(
        config: &Lsode2ProblemConfig,
        method: Lsode2NativeStepMethod,
        linear_backend: L,
        jacobian_storage: NativeJacobianStorage,
    ) -> Result<Self, IvpBackendError> {
        let (residual, jacobian): (
            Rc<NativeResidualFn>,
            Rc<RefCell<Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian>>>,
        ) = match config.backend.jacobian_backend {
            Lsode2JacobianBackend::SymbolicGenerated => {
                let mut options = SymbolicIvpProblemOptions::new();
                if let Some(parameters) = config.equation_parameters.clone() {
                    options = options.with_equation_parameters(parameters);
                }
                if let Some(values) = config.equation_parameter_values.clone() {
                    options = options.with_equation_parameter_values(values);
                }
                let symbolic_assembly_backend = match config.residual_jacobian_source {
                    Lsode2ResidualJacobianSource::Symbolic { assembly, .. } => assembly,
                    Lsode2ResidualJacobianSource::Analytical => {
                        Lsode2SymbolicAssemblyBackend::ExprLegacy
                    }
                };
                options = options.with_symbolic_assembly_backend(match symbolic_assembly_backend {
                    Lsode2SymbolicAssemblyBackend::ExprLegacy => {
                        IvpSymbolicAssemblyBackend::ExprLegacy
                    }
                    Lsode2SymbolicAssemblyBackend::AtomView => IvpSymbolicAssemblyBackend::AtomView,
                });

                let prepared = prepare_generated_symbolic_ivp_residual_problem(
                    config.eq_system.clone(),
                    config.values.clone(),
                    config.arg.clone(),
                    options,
                    config.backend.generated_backend.clone(),
                )
                .map_err(map_generated_backend_error)?;
                let residual_problem = Rc::new(prepared.into_problem());
                let residual = {
                    let residual_problem = Rc::clone(&residual_problem);
                    Rc::new(move |t: f64, y: &DVector<f64>| (residual_problem.residual)(t, y))
                        as Rc<NativeResidualFn>
                };
                let use_sparse_aot_jacobian = matches!(
                    config.residual_jacobian_source,
                    Lsode2ResidualJacobianSource::Symbolic {
                        execution: Lsode2SymbolicExecutionMode::Aot { .. },
                        ..
                    }
                );
                let jacobian = if use_sparse_aot_jacobian {
                    Rc::new(RefCell::new(
                        compile_native_sparse_aot_jacobian_with_parameter_handle(
                            &config.eq_system,
                            &config.values,
                            config.arg.as_str(),
                            config.equation_parameters.as_deref(),
                            config.equation_parameter_values.clone(),
                            residual_problem.parameter_values_handle(),
                            jacobian_storage,
                            config.backend.generated_backend.clone(),
                            match symbolic_assembly_backend {
                                Lsode2SymbolicAssemblyBackend::ExprLegacy => {
                                    IvpSymbolicAssemblyBackend::ExprLegacy
                                }
                                Lsode2SymbolicAssemblyBackend::AtomView => {
                                    IvpSymbolicAssemblyBackend::AtomView
                                }
                            },
                        )?,
                    ))
                } else {
                    Rc::new(RefCell::new(
                        compile_native_symbolic_jacobian_with_parameter_handle(
                            &config.eq_system,
                            &config.values,
                            config.arg.as_str(),
                            config.equation_parameters.as_deref(),
                            residual_problem.parameter_values_handle(),
                            jacobian_storage,
                        ),
                    ))
                };
                (residual, jacobian)
            }
            Lsode2JacobianBackend::AnalyticClosure => {
                let callbacks = config
                    .analytical_callbacks
                    .as_ref()
                    .ok_or_else(|| IvpBackendError::GeneratedBackendFailure {
                        message:
                            "LSODE2 analytical native step engine requires residual/jacobian callbacks"
                                .to_string(),
                    })?
                    .clone();
                let residual_callbacks = callbacks.clone();
                let residual =
                    Rc::new(move |t: f64, y: &DVector<f64>| (residual_callbacks.residual)(t, y))
                        as Rc<NativeResidualFn>;
                let jacobian_callbacks = callbacks;
                let jacobian = Rc::new(RefCell::new(Box::new(move |t: f64, y: &DVector<f64>| {
                    BdfJacobian::from_dense((jacobian_callbacks.jacobian)(t, y))
                })
                    as Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian>));
                (residual, jacobian)
            }
            Lsode2JacobianBackend::FiniteDifference => {
                let callbacks = config
                    .analytical_callbacks
                    .as_ref()
                    .ok_or_else(|| IvpBackendError::GeneratedBackendFailure {
                        message:
                            "LSODE2 finite-difference Jacobian backend requires analytical residual callback"
                                .to_string(),
                    })?
                    .clone();
                let residual_callbacks = callbacks.clone();
                let residual =
                    Rc::new(move |t: f64, y: &DVector<f64>| (residual_callbacks.residual)(t, y))
                        as Rc<NativeResidualFn>;
                let residual_for_jac = Rc::clone(&residual);
                let atol = config.atol.abs().max(1.0e-14);
                let jacobian = Rc::new(RefCell::new(Box::new(move |t: f64, y: &DVector<f64>| {
                    finite_difference_jacobian_from_residual(
                        residual_for_jac.as_ref(),
                        t,
                        y,
                        atol,
                        jacobian_storage,
                    )
                })
                    as Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian>));
                (residual, jacobian)
            }
        };

        let h0 = initial_native_step_size(config, residual.as_ref());
        let max_order = method.max_order(config);
        let step_method = match method {
            Lsode2NativeStepMethod::BdfLike => Lsode2StepMethod::BdfLike,
            Lsode2NativeStepMethod::AdamsLike => Lsode2StepMethod::AdamsLike,
        };
        let cycle = Lsode2StepCycle::new_with_method(
            Lsode2RuntimeState::new(
                config.t0,
                config.y0.as_slice(),
                h0,
                max_order,
                Lsode2StepControlConfig::default(),
            )
            .map_err(map_runtime_state_error)?,
            Lsode2ErrorController::new(
                Lsode2Tolerance::scalar(config.rtol, config.atol),
                Lsode2ErrorControlConfig::default(),
            )
            .map_err(map_error_control_error)?,
            step_method,
        );
        let correction = Lsode2CorrectionController::new(
            Lsode2Tolerance::scalar(config.rtol, config.atol),
            Lsode2CorrectionControlConfig::default(),
        )
        .map_err(map_correction_error)?;

        Ok(Self {
            residual,
            jacobian,
            linear_backend,
            driver: Lsode2NonlinearStepDriver::new(cycle, correction),
            fallback_stiffness_probe_interval: (config.controller.mode
                == Lsode2ControllerMode::AutomaticAdamsBdf)
                .then_some(config.controller.method_switch_probe_steps.max(1)),
        })
    }

    fn step_once(&mut self) -> Result<Lsode2NativeStepAttemptReport, IvpBackendError> {
        let mut first_predicted: Option<Lsode2PredictedStep> = None;
        let mut first_predictor_flags: Option<(
            Lsode2JacobianCurrency,
            Lsode2Ipup,
            Lsode2IpupTrigger,
        )> = None;
        let mut retry_count = 0usize;
        let mut jacobian_refresh_retry_count = 0usize;
        let mut total_iterations = 0usize;
        let mut refresh_requested = false;
        let residual_ctx: Rc<RefCell<Option<NativeStepResidualContext>>> =
            Rc::new(RefCell::new(None));
        let residual = Rc::clone(&self.residual);
        let residual_ctx_for_cb = Rc::clone(&residual_ctx);
        let jacobian = Rc::clone(&self.jacobian);
        let mut executor = Lsode2NativeCallbackExecutor::new(
            move |t: f64, y: &DVector<f64>| {
                let fy = (residual)(t, y);
                let ctx_guard = residual_ctx_for_cb.borrow();
                let ctx = ctx_guard
                    .as_ref()
                    .expect("native residual context should be set before correction pass");

                let mut g = y.clone_owned();
                g -= &ctx.y_pred;

                let mut scaled_f = fy;
                scaled_f *= ctx.h_trial;
                scaled_f -= &ctx.yh2;
                scaled_f *= ctx.el1;

                g -= scaled_f;
                g
            },
            move |t: f64, y: &DVector<f64>| (jacobian.borrow_mut())(t, y),
            self.linear_backend.clone(),
        );

        loop {
            self.refresh_first_derivative_if_requested()?;
            let predicted = self
                .driver
                .begin_step()
                .map_err(map_nonlinear_driver_error)?;
            if first_predictor_flags.is_none() {
                first_predictor_flags = Some((
                    self.driver.cycle().jacobian_currency(),
                    self.driver.cycle().ipup(),
                    self.driver.cycle().ipup_trigger(),
                ));
            }
            if first_predicted.is_none() {
                first_predicted = Some(predicted.clone());
            }
            let h_trial = predicted.h_trial;
            let order = predicted.order;
            let el1 = el1_for_step_method(self.driver.cycle().method(), order)?;
            let hl0 = h_trial * el1;
            let yh2 = self
                .driver
                .cycle()
                .state()
                .predicted_nordsieck()
                .col(1)
                .map_err(map_history_error)?
                .to_vec();
            *residual_ctx.borrow_mut() = Some(NativeStepResidualContext {
                y_pred: DVector::from_vec(predicted.y_pred.clone()),
                yh2: DVector::from_vec(yh2),
                h_trial,
                el1,
            });

            let mut y_candidate = DVector::from_vec(predicted.y_pred.clone());
            let mut force_refresh_on_next_correction = should_force_refresh_on_first_correction(
                refresh_requested,
                self.driver.cycle().ipup(),
            );

            loop {
                total_iterations += 1;
                let outcome = self
                    .driver
                    .compute_apply_and_submit_correction_with_refresh_policy(
                        &mut y_candidate,
                        hl0,
                        &mut executor,
                        force_refresh_on_next_correction,
                    )
                    .map_err(map_nonlinear_driver_error)?;
                force_refresh_on_next_correction = false;

                match &outcome {
                    Lsode2StepCycleOutcome::NonlinearContinue { .. } => continue,
                    Lsode2StepCycleOutcome::Rejected { retry, .. }
                    | Lsode2StepCycleOutcome::NonlinearRejected { retry, .. } => {
                        if let Some(next_refresh_requested) = retry_refresh_requested(retry.action)
                        {
                            retry_count += 1;
                            if next_refresh_requested {
                                jacobian_refresh_retry_count += 1;
                            }
                            refresh_requested = next_refresh_requested;
                            break;
                        }
                    }
                    Lsode2StepCycleOutcome::Accepted { .. } => {}
                }

                let stiffness_ratio = executor
                    .last_jacobian_abs_max()
                    .map(|jac_max| jac_max * h_trial.abs())
                    .or_else(|| {
                        self.fallback_adams_stiffness_probe(&outcome, &y_candidate, h_trial)
                    });
                let telemetry = self.driver.switch_telemetry(stiffness_ratio);
                let jcur = self.driver.cycle().jacobian_currency();
                let ipup = self.driver.cycle().ipup();
                let ipup_trigger = self.driver.cycle().ipup_trigger();
                let kflag = self.driver.cycle().kflag();
                let kflag_code = self.driver.cycle().kflag_code();
                let icf = self.driver.cycle().icf();
                let iret = self.driver.cycle().iret();
                let redo_stage = self.driver.cycle().redo_stage();
                let iredo = self.driver.cycle().iredo();
                let ialth = self
                    .driver
                    .cycle()
                    .state()
                    .step_control_snapshot()
                    .adjustment_wait;
                self.record_dstoda_flags_snapshot();
                // DSTODA mirroring:
                // Do not force-reconcile the first Nordsieck derivative on every
                // accepted step. Accepted-state Nordsieck update is already driven by
                // EL*ACOR choreography in runtime-state accept path. A hard overwrite
                // with h*f at every step perturbs multistep consistency and can
                // artificially bias error-test/retry behavior.
                let (predictor_jcur, predictor_ipup, predictor_ipup_trigger) =
                    first_predictor_flags.expect(
                        "native step report should have predictor JCUR/IPUP flags from begin_step",
                    );
                return Ok(Lsode2NativeStepAttemptReport {
                    predicted: first_predicted.clone().expect(
                        "a native step attempt report should always have an initial prediction",
                    ),
                    outcome,
                    iterations: total_iterations,
                    retry_count,
                    jacobian_refresh_retry_count,
                    telemetry,
                    predictor_jcur,
                    predictor_ipup,
                    predictor_ipup_trigger,
                    jcur,
                    ipup,
                    ipup_trigger,
                    kflag,
                    kflag_code,
                    icf,
                    iret,
                    redo_stage,
                    iredo,
                    ialth,
                });
            }
        }
    }

    fn statistics(&self) -> &Lsode2NativeStatistics {
        self.driver.statistics()
    }

    fn state_snapshot(&self) -> Lsode2RuntimeStateSnapshot {
        self.driver.cycle().state().snapshot()
    }

    fn current_solution(&self) -> Vec<f64> {
        self.driver.cycle().state().y().to_vec()
    }

    fn clamp_step_to_t_bound(&mut self, t_bound: f64) -> Result<(), IvpBackendError> {
        let snapshot = self.state_snapshot();
        let remaining = t_bound - snapshot.t;
        if remaining == 0.0 {
            return Ok(());
        }
        let h = snapshot.h;
        if h.signum() == remaining.signum() && h.abs() > remaining.abs() {
            self.driver
                .cycle_mut()
                .state_mut()
                .set_step_size(remaining)
                .map_err(map_runtime_state_error)?;
        }
        Ok(())
    }

    fn switch_method(
        &mut self,
        config: &Lsode2ProblemConfig,
        method: Lsode2NativeStepMethod,
    ) -> Result<(), IvpBackendError> {
        let max_order = method.max_order(config);
        let step_method = match method {
            Lsode2NativeStepMethod::BdfLike => Lsode2StepMethod::BdfLike,
            Lsode2NativeStepMethod::AdamsLike => Lsode2StepMethod::AdamsLike,
        };
        self.driver
            .cycle_mut()
            .prepare_for_method_switch_handoff(step_method, max_order)
            .map_err(map_step_cycle_error)?;
        self.driver.reset_iteration_memory_after_method_switch();
        Ok(())
    }

    fn fallback_adams_stiffness_probe(
        &mut self,
        outcome: &Lsode2StepCycleOutcome,
        y_candidate: &DVector<f64>,
        h_trial: f64,
    ) -> Option<f64> {
        let interval = self.fallback_stiffness_probe_interval?;
        if self.driver.cycle().method() != Lsode2StepMethod::AdamsLike {
            return None;
        }
        let accepted_steps = self.driver.cycle().state().snapshot().accepted_steps;
        if accepted_steps == 0 || accepted_steps % interval != 0 {
            return None;
        }
        let t_new = match outcome {
            Lsode2StepCycleOutcome::Accepted { t_new, .. } => *t_new,
            _ => return None,
        };

        // Adams functional iteration can accept a step without touching the
        // Newton/Jacobian executor. LSODA still needs a stiffness signal for
        // method switching, so in automatic mode we probe the already available
        // Jacobian callback at the configured switch cadence.
        let jacobian = (self.jacobian.borrow_mut())(t_new, y_candidate);
        jacobian_abs_max(&jacobian).map(|jac_max| jac_max * h_trial.abs())
    }

    fn refresh_first_derivative_if_requested(&mut self) -> Result<(), IvpBackendError> {
        if !self
            .driver
            .cycle()
            .state()
            .first_derivative_refresh_requested()
        {
            return Ok(());
        }

        let snapshot = self.state_snapshot();
        let y = DVector::from_vec(self.driver.cycle().state().y().to_vec());
        let started = Instant::now();
        let rhs = (self.residual)(snapshot.t, &y);
        self.driver
            .statistics_mut()
            .record_native_residual_duration(started.elapsed());
        let scaled_derivative = rhs
            .iter()
            .map(|value| snapshot.h * *value)
            .collect::<Vec<_>>();
        self.driver
            .cycle_mut()
            .state_mut()
            .reconcile_first_nordsieck_derivative(scaled_derivative.as_slice())
            .map_err(map_runtime_state_error)?;
        Ok(())
    }

    fn reconcile_accepted_first_nordsieck_derivative(
        &mut self,
        t_new: f64,
        y_new: &[f64],
        h_trial: f64,
    ) -> Result<(), IvpBackendError> {
        let y_new_vec = DVector::from_vec(y_new.to_vec());
        let started = Instant::now();
        let accepted_rhs = (self.residual)(t_new, &y_new_vec);
        self.driver
            .statistics_mut()
            .record_native_residual_duration(started.elapsed());
        let scaled_derivative = accepted_rhs
            .iter()
            .map(|value| h_trial * *value)
            .collect::<Vec<_>>();
        self.driver
            .cycle_mut()
            .state_mut()
            .reconcile_first_nordsieck_derivative(scaled_derivative.as_slice())
            .map_err(map_runtime_state_error)?;
        Ok(())
    }

    fn record_dstoda_flags_snapshot(&mut self) {
        let cycle = self.driver.cycle();
        let jcur = cycle.jacobian_currency();
        let ipup = cycle.ipup();
        let ipup_trigger = cycle.ipup_trigger();
        let kflag = cycle.kflag();
        let icf = cycle.icf();
        let iret = cycle.iret();
        let redo_stage = cycle.redo_stage();
        let ialth = cycle.state().step_control_snapshot().adjustment_wait;
        self.driver.statistics_mut().record_dstoda_flags(
            jcur,
            ipup,
            ipup_trigger,
            kflag,
            icf,
            iret,
            redo_stage,
        );
        self.driver.statistics_mut().record_ialth(ialth);
    }
}

fn finite_difference_jacobian_from_residual(
    residual: &NativeResidualFn,
    t: f64,
    y: &DVector<f64>,
    atol: f64,
    storage: NativeJacobianStorage,
) -> BdfJacobian {
    let n = y.len();
    let f0 = residual(t, y);
    let mut dense = DMatrix::<f64>::zeros(n, n);
    let eps = f64::EPSILON.sqrt();

    for col in 0..n {
        let yj = y[col];
        let h = eps * yj.abs().max(atol).max(1.0);
        let mut y_pert = y.clone_owned();
        y_pert[col] += h;
        let f_pert = residual(t, &y_pert);
        for row in 0..n {
            dense[(row, col)] = (f_pert[row] - f0[row]) / h;
        }
    }

    match storage {
        NativeJacobianStorage::Dense => BdfJacobian::from_dense(dense),
        NativeJacobianStorage::SparseTriplets => {
            let mut triplets = Vec::new();
            for col in 0..n {
                for row in 0..n {
                    let value = dense[(row, col)];
                    if value != 0.0 {
                        triplets.push(faer::sparse::Triplet::new(row, col, value));
                    }
                }
            }
            BdfJacobian::SparseTriplets { n, triplets }
        }
        NativeJacobianStorage::Banded => {
            let mut kl = 0usize;
            let mut ku = 0usize;
            for col in 0..n {
                for row in 0..n {
                    let value = dense[(row, col)];
                    if value != 0.0 {
                        kl = kl.max(row.saturating_sub(col));
                        ku = ku.max(col.saturating_sub(row));
                    }
                }
            }
            let mut banded = Banded::<f64>::zeros(n, kl, ku)
                .expect("finite-difference Jacobian bandwidth should define valid banded storage");
            banded.fill_from_dense(|i, j| dense[(i, j)]);
            BdfJacobian::Banded(banded)
        }
    }
}

fn retry_refresh_requested(action: Lsode2RetryAction) -> Option<bool> {
    match action {
        Lsode2RetryAction::Retry => Some(false),
        Lsode2RetryAction::RetryWithJacobianRefresh => Some(true),
        Lsode2RetryAction::FailStepSizeUnderflow
        | Lsode2RetryAction::FailRepeatedErrorTestFailures
        | Lsode2RetryAction::FailRepeatedConvergenceFailures => None,
    }
}

fn el1_for_step_method(method: Lsode2StepMethod, order: usize) -> Result<f64, IvpBackendError> {
    let el1 = match method {
        Lsode2StepMethod::BdfLike => {
            Lsode2BdfDcfodeTables::default()
                .order(order)
                .map_err(|err| IvpBackendError::GeneratedBackendFailure {
                    message: err.to_string(),
                })?
                .el[0]
        }
        Lsode2StepMethod::AdamsLike => {
            Lsode2AdamsDcfodeTables::default()
                .order(order)
                .map_err(|err| IvpBackendError::GeneratedBackendFailure {
                    message: err.to_string(),
                })?
                .el[1]
        }
    };
    Ok(el1)
}

fn initial_native_step_size(config: &Lsode2ProblemConfig, residual: &NativeResidualFn) -> f64 {
    let direction = if config.t_bound >= config.t0 {
        1.0
    } else {
        -1.0
    };
    let span = (config.t_bound - config.t0).abs();

    let h0_mag = {
        let f0 = residual(config.t0, &config.y0);
        let scale = DVector::from_vec(scale_func(
            NumberOrVec::Number(config.rtol),
            NumberOrVec::Number(config.atol),
            &config.y0,
        ));
        let d0 = norm(&(config.y0.component_div(&scale)));
        let d1 = norm(&(f0.component_div(&scale)));
        let h0 = if d0 < 1.0e-5 || d1 < 1.0e-5 {
            1.0e-6
        } else {
            0.01 * d0 / d1
        }
        .min(span);

        if h0 > 0.0 {
            let y1 = &config.y0 + h0 * direction * &f0;
            let f1 = residual(config.t0 + h0 * direction, &y1);
            let d2 = norm(&((f1 - f0).component_div(&scale))) / h0;
            let h1 = if d1 <= 1.0e-15 && d2 <= 1.0e-15 {
                1.0e-6_f64.max(h0 * 1.0e-3)
            } else {
                (0.01 / d1.max(d2)).powf(0.5)
            };
            vec![100.0 * h0, h1, span, config.max_step]
                .into_iter()
                .fold(1.0_f64, f64::min)
        } else {
            config.max_step.min(span.max(config.max_step * 0.25))
        }
    };

    let mag = config
        .first_step
        .unwrap_or_else(|| {
            if h0_mag.is_finite() && h0_mag > 0.0 {
                h0_mag.min(config.max_step).min(span.max(f64::EPSILON))
            } else {
                config.max_step.min(span.max(config.max_step * 0.25))
            }
        })
        .abs()
        .max(f64::EPSILON);
    if direction > 0.0 {
        mag
    } else {
        -mag
    }
}

fn map_generated_backend_error(
    err: crate::symbolic::symbolic_ivp_generated::SymbolicIvpGeneratedError,
) -> IvpBackendError {
    match err {
        crate::symbolic::symbolic_ivp_generated::SymbolicIvpGeneratedError::IvpBackend(err) => err,
        other => IvpBackendError::GeneratedBackendFailure {
            message: other.to_string(),
        },
    }
}

fn map_runtime_state_error(err: super::state::Lsode2RuntimeStateError) -> IvpBackendError {
    IvpBackendError::GeneratedBackendFailure {
        message: err.to_string(),
    }
}

fn map_error_control_error(err: super::error_control::Lsode2ErrorControlError) -> IvpBackendError {
    IvpBackendError::GeneratedBackendFailure {
        message: err.to_string(),
    }
}

fn map_correction_error(err: super::correction::Lsode2CorrectionError) -> IvpBackendError {
    IvpBackendError::GeneratedBackendFailure {
        message: err.to_string(),
    }
}

fn map_history_error(err: super::history::Lsode2HistoryError) -> IvpBackendError {
    IvpBackendError::GeneratedBackendFailure {
        message: err.to_string(),
    }
}

fn map_step_cycle_error(err: super::step_cycle::Lsode2StepCycleError) -> IvpBackendError {
    IvpBackendError::GeneratedBackendFailure {
        message: err.to_string(),
    }
}

fn map_nonlinear_driver_error(
    err: super::nonlinear_driver::Lsode2NonlinearDriverError,
) -> IvpBackendError {
    IvpBackendError::GeneratedBackendFailure {
        message: err.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn native_step_engine_builds_dense_backend() {
        let engine = Lsode2NativeStepEngine::from_problem_config(&exponential_decay_config())
            .expect("dense config should not error")
            .expect("dense config should enable native step engine");
        match engine {
            Lsode2NativeStepEngine::Dense(_) => {}
            _ => panic!("expected dense native step engine"),
        }
    }

    #[test]
    fn native_step_engine_dense_attempt_records_native_statistics() {
        let mut engine = Lsode2NativeStepEngine::from_problem_config(&exponential_decay_config())
            .expect("dense config should build a native step engine")
            .expect("dense config should enable native step engine");

        let report = engine
            .step_once()
            .expect("native dense step attempt should succeed");

        assert!(report.iterations > 0);
        assert!(report.predicted.t_trial > 0.0);
        assert!(!report.outcome_label().is_empty());
        assert!(engine.statistics().native_step_attempts > 0);
        assert!(engine.statistics().native_residual_calls > 0);
        assert!(engine.statistics().native_jacobian_calls > 0);
        assert!(engine.statistics().native_linear_solve_calls > 0);
    }

    #[test]
    fn native_step_engine_sparse_attempt_records_native_statistics() {
        let mut engine = Lsode2NativeStepEngine::from_problem_config(
            &exponential_decay_config().with_native_sparse_faer_backend(),
        )
        .expect("sparse config should build a native step engine")
        .expect("sparse config should enable native step engine");

        let report = engine
            .step_once()
            .expect("native step attempt should succeed");

        assert!(report.iterations > 0);
        assert!(report.predicted.t_trial > 0.0);
        assert!(!report.outcome_label().is_empty());
        assert!(engine.statistics().native_step_attempts > 0);
        assert!(engine.statistics().native_residual_calls > 0);
        assert!(engine.statistics().native_jacobian_calls > 0);
        assert!(engine.statistics().native_linear_solve_calls > 0);
    }

    #[test]
    fn retry_refresh_requested_follows_step_retry_policy() {
        assert_eq!(
            retry_refresh_requested(Lsode2RetryAction::Retry),
            Some(false)
        );
        assert_eq!(
            retry_refresh_requested(Lsode2RetryAction::RetryWithJacobianRefresh),
            Some(true)
        );
        assert_eq!(
            retry_refresh_requested(Lsode2RetryAction::FailStepSizeUnderflow),
            None
        );
        assert_eq!(
            retry_refresh_requested(Lsode2RetryAction::FailRepeatedErrorTestFailures),
            None
        );
        assert_eq!(
            retry_refresh_requested(Lsode2RetryAction::FailRepeatedConvergenceFailures),
            None
        );
    }

    #[test]
    fn first_correction_refresh_obeys_retry_or_predictor_ipup() {
        assert!(!should_force_refresh_on_first_correction(
            false,
            Lsode2Ipup::UpToDate
        ));
        assert!(should_force_refresh_on_first_correction(
            true,
            Lsode2Ipup::UpToDate
        ));
        assert!(should_force_refresh_on_first_correction(
            false,
            Lsode2Ipup::NeedsJacobianUpdate
        ));
        assert!(should_force_refresh_on_first_correction(
            true,
            Lsode2Ipup::NeedsJacobianUpdate
        ));
    }

    #[test]
    fn native_step_engine_refreshes_first_derivative_after_repeated_error_reset() {
        let mut engine = Lsode2NativeStepEngine::from_problem_config(
            &exponential_decay_config().with_native_sparse_faer_backend(),
        )
        .expect("sparse config should build a native step engine")
        .expect("sparse config should enable native step engine");

        let inner = match &mut engine {
            Lsode2NativeStepEngine::Sparse(inner) => inner,
            Lsode2NativeStepEngine::Dense(_) => unreachable!("test requested sparse backend"),
            Lsode2NativeStepEngine::Banded(_) => unreachable!("test requested sparse backend"),
        };
        inner.driver.cycle_mut().state_mut().set_order(3).unwrap();
        inner
            .driver
            .cycle_mut()
            .state_mut()
            .reset_after_repeated_error_failures()
            .unwrap();
        assert!(inner
            .driver
            .cycle()
            .state()
            .first_derivative_refresh_requested());

        inner.refresh_first_derivative_if_requested().unwrap();

        let state = inner.driver.cycle().state();
        let expected = -state.h() * state.y()[0];
        assert!((state.nordsieck().col(1).unwrap()[0] - expected).abs() < 1.0e-12);
        assert!(!state.first_derivative_refresh_requested());
        assert!(inner.driver.statistics().native_residual_calls > 0);
    }

    #[test]
    fn native_step_engine_refreshes_first_derivative_after_nonlinear_retract_to_order_one() {
        let mut engine = Lsode2NativeStepEngine::from_problem_config(
            &exponential_decay_config().with_native_sparse_faer_backend(),
        )
        .expect("sparse config should build a native step engine")
        .expect("sparse config should enable native step engine");

        let inner = match &mut engine {
            Lsode2NativeStepEngine::Sparse(inner) => inner,
            Lsode2NativeStepEngine::Dense(_) => unreachable!("test requested sparse backend"),
            Lsode2NativeStepEngine::Banded(_) => unreachable!("test requested sparse backend"),
        };
        inner.driver.cycle_mut().state_mut().set_order(3).unwrap();
        let retry = inner
            .driver
            .cycle_mut()
            .state_mut()
            .reject_after_nonlinear_failure()
            .unwrap();
        assert_eq!(retry.action, Lsode2RetryAction::RetryWithJacobianRefresh);
        assert_eq!(retry.order_new, 1);
        assert!(inner
            .driver
            .cycle()
            .state()
            .first_derivative_refresh_requested());

        inner.refresh_first_derivative_if_requested().unwrap();

        let state = inner.driver.cycle().state();
        let expected = -state.h() * state.y()[0];
        assert!((state.nordsieck().col(1).unwrap()[0] - expected).abs() < 1.0e-12);
        assert!(!state.first_derivative_refresh_requested());
        assert!(inner.driver.statistics().native_residual_calls > 0);
    }

    #[test]
    fn native_step_engine_adams_like_uses_controller_max_adams_order() {
        let config = exponential_decay_config()
            .with_controller(
                super::super::algorithm::Lsode2ControllerConfig::bdf_only().with_max_adams_order(4),
            )
            .with_native_sparse_faer_backend();
        let engine = Lsode2NativeStepEngine::from_problem_config_with_method(
            &config,
            Lsode2NativeStepMethod::AdamsLike,
        )
        .expect("adams-like sparse config should build a native step engine")
        .expect("adams-like sparse config should enable native step engine");

        let snapshot = engine.state_snapshot();
        assert_eq!(snapshot.order, 1);
        assert_eq!(snapshot.max_order, 4);
    }

    #[test]
    fn native_step_engine_el1_is_method_specific_for_higher_order() {
        let bdf_q3 = el1_for_step_method(Lsode2StepMethod::BdfLike, 3)
            .expect("bdf el(1) should be available for q=3");
        let adams_q3 = el1_for_step_method(Lsode2StepMethod::AdamsLike, 3)
            .expect("adams el(1) should be available for q=3");

        assert!(bdf_q3.is_finite() && bdf_q3 > 0.0);
        assert!(adams_q3.is_finite() && adams_q3 > 0.0);
        assert!(
            (bdf_q3 - adams_q3).abs() > 1.0e-12,
            "BDF and Adams EL(1) should differ at q=3: bdf={bdf_q3:e}, adams={adams_q3:e}"
        );
    }
}
