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
    Lsode2JacobianBackend, Lsode2LinearSolverBackend, Lsode2LinearSystemStructure,
    Lsode2ProblemConfig, Lsode2ResidualJacobianSource, Lsode2SparseJacobianPattern,
    Lsode2SymbolicAssemblyBackend, Lsode2SymbolicExecutionMode,
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
use super::native_executor::{Lsode2NativeCallbackExecutor, jacobian_abs_max};
use super::native_jacobian::{
    NativeJacobianStorage, compile_native_sparse_aot_jacobian_from_linked_backend_compat,
    compile_native_sparse_aot_jacobian_with_parameter_handle_and_telemetry,
    compile_native_symbolic_jacobian_with_parameter_handle_and_telemetry_and_policy,
    try_compile_native_atomview_jacobian_with_parameter_handle_and_telemetry_and_policy,
};
use super::nonlinear_driver::Lsode2NonlinearStepDriver;
use super::state::{Lsode2RuntimeState, Lsode2RuntimeStateSnapshot};
use super::statistics::Lsode2NativeStatistics;
use super::step_control::{Lsode2RetryAction, Lsode2StepControlConfig};
use super::step_cycle::{
    Lsode2PredictedStep, Lsode2StepCycle, Lsode2StepCycleOutcome, Lsode2StepMethod,
};
use crate::numerical::BDF::BDF_solver::{BdfJacobian, BdfLinearBackend};
use crate::numerical::BDF::common::{NumberOrVec, norm, scale_func};
use crate::somelinalg::banded::storage::Banded;
use crate::symbolic::ivp_telemetry::{
    IvpTelemetry, IvpTelemetryExecution, IvpTelemetryMatrixBackend, IvpTelemetryRoute, IvpWarmStage,
};
use crate::symbolic::symbolic_ivp::{
    IvpBackendError, IvpSymbolicAssemblyBackend, SharedIvpParameterValues,
    SymbolicIvpProblemOptions,
};
use crate::symbolic::symbolic_ivp_generated::{
    SymbolicIvpAotBuildPolicy, prepare_generated_symbolic_ivp_banded_residual_problem,
    prepare_generated_symbolic_ivp_native_callbacks,
    prepare_generated_symbolic_ivp_residual_problem,
};
use nalgebra::{DMatrix, DVector};
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

type NativeResidualFn = dyn Fn(f64, &DVector<f64>) -> DVector<f64>;

/// Prepared evaluator callbacks shared by fresh native step drivers.
///
/// The callbacks are immutable at the solver lifecycle level.  Jacobian
/// evaluation remains internally mutable because the generated callback may
/// carry scratch state, but solves themselves still create an independent
/// driver and linear backend.
#[derive(Clone)]
pub(crate) struct PreparedNativeCallbacks {
    residual: Rc<NativeResidualFn>,
    jacobian: Rc<RefCell<Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian>>>,
    parameter_values_handle: Option<SharedIvpParameterValues>,
    residual_evaluator_instrumented: bool,
    jacobian_evaluator_instrumented: bool,
}

impl PreparedNativeCallbacks {
    pub(crate) fn parameter_values_handle(&self) -> Option<SharedIvpParameterValues> {
        self.parameter_values_handle.clone()
    }

    pub(crate) fn bridge_residual(&self) -> Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>> {
        let residual = Rc::clone(&self.residual);
        Box::new(move |t, y| residual(t, y))
    }

    pub(crate) fn bridge_jacobian_factory(
        &self,
    ) -> impl Fn(
        Option<SharedIvpParameterValues>,
    ) -> Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian>
    + 'static {
        let jacobian = Rc::clone(&self.jacobian);
        move |_parameter_values_handle| {
            let jacobian = Rc::clone(&jacobian);
            Box::new(move |t, y| (jacobian.borrow_mut())(t, y))
        }
    }

    pub(crate) fn rebind_parameter_values(
        &self,
        values: &DVector<f64>,
    ) -> Result<(), IvpBackendError> {
        if let Some(handle) = self.parameter_values_handle.as_ref() {
            let mut slot = handle
                .write()
                .map_err(|_| IvpBackendError::ParameterStatePoisoned)?;
            *slot = values.clone();
        }
        Ok(())
    }
}

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
#[allow(dead_code)]
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
        config
            .telemetry
            .set_matrix_backend(match config.linear_system_structure {
                Lsode2LinearSystemStructure::Dense => IvpTelemetryMatrixBackend::Dense,
                Lsode2LinearSystemStructure::Sparse => IvpTelemetryMatrixBackend::Sparse,
                Lsode2LinearSystemStructure::Banded { .. } => IvpTelemetryMatrixBackend::Banded,
            });
        config.telemetry.set_problem_shape(
            config.y0.len(),
            config.eq_system.len(),
            config.equation_parameters.as_ref().map_or(0, Vec::len),
        );
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

        if config.jac_sparsity.is_some() && config.sparse_jacobian_pattern.is_some() {
            return Err(IvpBackendError::InvalidArgumentSchema {
                message: "provide either dense jac_sparsity or sparse_jacobian_pattern, not both"
                    .to_string(),
            });
        }
        if config.sparse_jacobian_pattern.is_some()
            && (config.backend.jacobian_backend != Lsode2JacobianBackend::FiniteDifference
                || config.backend.linear_solver_backend != Lsode2LinearSolverBackend::SparseFaer)
        {
            return Err(IvpBackendError::InvalidArgumentSchema {
                message: "sparse_jacobian_pattern is supported only by the native Sparse finite-difference route"
                    .to_string(),
            });
        }

        let sparse_fd_coloring = if config.backend.jacobian_backend
            == Lsode2JacobianBackend::FiniteDifference
            && config.backend.linear_solver_backend == Lsode2LinearSolverBackend::SparseFaer
        {
            if let Some(pattern) = config.sparse_jacobian_pattern.as_ref() {
                Some(SparseFiniteDifferenceColoring::from_sparse_pattern(
                    pattern,
                    config.y0.len(),
                )?)
            } else {
                config
                    .jac_sparsity
                    .as_ref()
                    .map(|pattern| {
                        SparseFiniteDifferenceColoring::from_mask(pattern, config.y0.len())
                    })
                    .transpose()?
            }
        } else {
            None
        };

        match config.backend.linear_solver_backend {
            Lsode2LinearSolverBackend::Dense => Ok(Some(Self::Dense(Box::new(
                Lsode2NativeStepEngineImpl::from_problem_config(
                    config,
                    method,
                    DenseLuBdfLinearBackend,
                    NativeJacobianStorage::Dense,
                    None,
                )?,
            )))),
            Lsode2LinearSolverBackend::SparseFaer => Ok(Some(Self::Sparse(Box::new(
                Lsode2NativeStepEngineImpl::from_problem_config(
                    config,
                    method,
                    FaerSparseBdfLinearBackend::default(),
                    NativeJacobianStorage::SparseTriplets,
                    sparse_fd_coloring,
                )?,
            )))),
            Lsode2LinearSolverBackend::BandedFaithful => Ok(Some(Self::Banded(Box::new(
                Lsode2NativeStepEngineImpl::from_problem_config(
                    config,
                    method,
                    FaithfulBandedBdfLinearBackend::default(),
                    banded_jacobian_storage(config),
                    None,
                )?,
            )))),
        }
    }

    pub(crate) fn prepare_callbacks_for_config(
        config: &Lsode2ProblemConfig,
    ) -> Result<Option<PreparedNativeCallbacks>, IvpBackendError> {
        let engine =
            Self::from_problem_config_with_method(config, Lsode2NativeStepMethod::BdfLike)?;
        Ok(engine.map(Self::into_prepared_callbacks))
    }

    pub(crate) fn from_prepared_callbacks_with_method(
        config: &Lsode2ProblemConfig,
        method: Lsode2NativeStepMethod,
        callbacks: &PreparedNativeCallbacks,
    ) -> Result<Option<Self>, IvpBackendError> {
        config
            .telemetry
            .set_matrix_backend(match config.linear_system_structure {
                Lsode2LinearSystemStructure::Dense => IvpTelemetryMatrixBackend::Dense,
                Lsode2LinearSystemStructure::Sparse => IvpTelemetryMatrixBackend::Sparse,
                Lsode2LinearSystemStructure::Banded { .. } => IvpTelemetryMatrixBackend::Banded,
            });
        config.telemetry.set_problem_shape(
            config.y0.len(),
            config.eq_system.len(),
            config.equation_parameters.as_ref().map_or(0, Vec::len),
        );
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
                Lsode2NativeStepEngineImpl::from_prepared_callbacks(
                    config,
                    method,
                    DenseLuBdfLinearBackend,
                    callbacks,
                )?,
            )))),
            Lsode2LinearSolverBackend::SparseFaer => Ok(Some(Self::Sparse(Box::new(
                Lsode2NativeStepEngineImpl::from_prepared_callbacks(
                    config,
                    method,
                    FaerSparseBdfLinearBackend::default(),
                    callbacks,
                )?,
            )))),
            Lsode2LinearSolverBackend::BandedFaithful => Ok(Some(Self::Banded(Box::new(
                Lsode2NativeStepEngineImpl::from_prepared_callbacks(
                    config,
                    method,
                    FaithfulBandedBdfLinearBackend::default(),
                    callbacks,
                )?,
            )))),
        }
    }

    fn into_prepared_callbacks(self) -> PreparedNativeCallbacks {
        match self {
            Self::Dense(engine) => engine.into_prepared_callbacks(),
            Self::Sparse(engine) => engine.into_prepared_callbacks(),
            Self::Banded(engine) => engine.into_prepared_callbacks(),
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
    parameter_values_handle: Option<SharedIvpParameterValues>,
    linear_backend: L,
    driver: Lsode2NonlinearStepDriver,
    fallback_stiffness_probe_interval: Option<usize>,
    telemetry: IvpTelemetry,
    residual_evaluator_instrumented: bool,
    jacobian_evaluator_instrumented: bool,
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
        sparse_fd_coloring: Option<SparseFiniteDifferenceColoring>,
    ) -> Result<Self, IvpBackendError> {
        match config.backend.jacobian_backend {
            Lsode2JacobianBackend::AnalyticClosure => {
                config
                    .telemetry
                    .set_route(IvpTelemetryRoute::AnalyticalClosure);
            }
            Lsode2JacobianBackend::FiniteDifference => {
                config
                    .telemetry
                    .set_route(IvpTelemetryRoute::FiniteDifference);
            }
            Lsode2JacobianBackend::SymbolicGenerated => {}
        }
        let (
            residual,
            jacobian,
            parameter_values_handle,
            residual_evaluator_instrumented,
            jacobian_evaluator_instrumented,
        ): (
            Rc<NativeResidualFn>,
            Rc<RefCell<Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian>>>,
            Option<SharedIvpParameterValues>,
            bool,
            bool,
        ) = match config.backend.jacobian_backend {
            Lsode2JacobianBackend::SymbolicGenerated => {
                let mut options = SymbolicIvpProblemOptions::new();
                options = options.with_telemetry(config.telemetry.clone());
                options = options.with_lambdify_execution_policy(config.lambdify_execution_policy);
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

                let use_sparse_aot_jacobian = matches!(
                    config.residual_jacobian_source,
                    Lsode2ResidualJacobianSource::Symbolic {
                        execution: Lsode2SymbolicExecutionMode::Aot { .. },
                        ..
                    }
                );
                let use_combined_atom_aot = symbolic_assembly_backend
                    == Lsode2SymbolicAssemblyBackend::AtomView
                    && use_sparse_aot_jacobian
                    && !matches!(
                        config.backend.generated_backend.build_policy,
                        SymbolicIvpAotBuildPolicy::UseIfAvailable
                    );
                let (residual_problem, shared_aot_jacobian, updated_resolver) =
                    if use_combined_atom_aot {
                        let bandwidth = match jacobian_storage {
                            NativeJacobianStorage::Banded {
                                bandwidth: Some((kl, ku)),
                            } => Some((kl, ku)),
                            _ => None,
                        };
                        let combined = prepare_generated_symbolic_ivp_native_callbacks(
                            config.eq_system.clone(),
                            config.values.clone(),
                            config.arg.clone(),
                            bandwidth,
                            options,
                            config.backend.generated_backend.clone(),
                        )
                        .map_err(map_generated_backend_error)?;
                        let linked = combined
                            .sparse_backend
                            .linked_backend
                            .clone()
                            .ok_or_else(|| IvpBackendError::GeneratedBackendFailure {
                                message: "combined AtomView AOT preparation did not retain a linked Jacobian backend"
                                    .to_string(),
                            })?;
                        let structure = combined.sparse_backend.jacobian_structure.clone();
                        (
                            combined.residual_problem,
                            Some((linked, structure)),
                            combined.sparse_backend.updated_resolver,
                        )
                    } else {
                        let prepared = if symbolic_assembly_backend
                            == Lsode2SymbolicAssemblyBackend::AtomView
                        {
                            match jacobian_storage {
                                NativeJacobianStorage::Banded {
                                    bandwidth: Some((kl, ku)),
                                } => prepare_generated_symbolic_ivp_banded_residual_problem(
                                    config.eq_system.clone(),
                                    config.values.clone(),
                                    config.arg.clone(),
                                    (kl, ku),
                                    options,
                                    config.backend.generated_backend.clone(),
                                ),
                                _ => prepare_generated_symbolic_ivp_residual_problem(
                                    config.eq_system.clone(),
                                    config.values.clone(),
                                    config.arg.clone(),
                                    options,
                                    config.backend.generated_backend.clone(),
                                ),
                            }
                        } else {
                            prepare_generated_symbolic_ivp_residual_problem(
                                config.eq_system.clone(),
                                config.values.clone(),
                                config.arg.clone(),
                                options,
                                config.backend.generated_backend.clone(),
                            )
                        }
                        .map_err(map_generated_backend_error)?;
                        let updated_resolver = prepared.updated_resolver.clone();
                        (prepared.into_problem(), None, updated_resolver)
                    };
                if let Some(handoff_path) = config.backend.generated_backend.handoff_path.as_ref() {
                    let resolver = updated_resolver.as_ref().ok_or_else(|| {
                        IvpBackendError::GeneratedBackendFailure {
                            message: "AOT producer did not publish an updated residual resolver"
                                .to_string(),
                        }
                    })?;
                    resolver.merge_handoff(handoff_path).map_err(|err| {
                        IvpBackendError::GeneratedBackendFailure {
                            message: format!(
                                "AOT residual handoff publication failed at {}: {err}",
                                handoff_path.display()
                            ),
                        }
                    })?;
                }
                // Both prepared Lambdify and linked AOT residual wrappers own
                // evaluator-level telemetry. The native executor still owns
                // solver request counters, but must not add a second
                // evaluator count for either generated route.
                let residual_evaluator_instrumented = true;
                config.telemetry.set_execution(
                    if matches!(
                        config.residual_jacobian_source,
                        Lsode2ResidualJacobianSource::Symbolic {
                            execution: Lsode2SymbolicExecutionMode::Aot { .. },
                            ..
                        }
                    ) {
                        IvpTelemetryExecution::Aot
                    } else {
                        IvpTelemetryExecution::Lambdify
                    },
                );
                // ExprLegacy and non-AOT compatibility routes still use the
                // historical Jacobian orchestration. AtomView+AOT instead
                // consumes the linked backend retained above.
                let mut jacobian_generated_backend = config.backend.generated_backend.clone();
                if let Some(updated_resolver) = updated_resolver {
                    jacobian_generated_backend.resolver = Some(updated_resolver);
                }
                let residual_problem = Rc::new(residual_problem);
                let parameter_values_handle = residual_problem.parameter_values_handle();
                let residual = {
                    let residual_problem = Rc::clone(&residual_problem);
                    Rc::new(move |t: f64, y: &DVector<f64>| (residual_problem.residual)(t, y))
                        as Rc<NativeResidualFn>
                };
                let jacobian = if let Some((linked, structure)) = shared_aot_jacobian {
                    Rc::new(RefCell::new(
                        compile_native_sparse_aot_jacobian_from_linked_backend_compat(
                            linked,
                            structure,
                            config.equation_parameters.as_deref(),
                            parameter_values_handle.clone(),
                            jacobian_storage,
                            config.telemetry.clone(),
                        )?,
                    ))
                } else if use_sparse_aot_jacobian {
                    Rc::new(RefCell::new(
                        compile_native_sparse_aot_jacobian_with_parameter_handle_and_telemetry(
                            &config.eq_system,
                            &config.values,
                            config.arg.as_str(),
                            config.equation_parameters.as_deref(),
                            config.equation_parameter_values.clone(),
                            parameter_values_handle.clone(),
                            jacobian_storage,
                            jacobian_generated_backend,
                            match symbolic_assembly_backend {
                                Lsode2SymbolicAssemblyBackend::ExprLegacy => {
                                    IvpSymbolicAssemblyBackend::ExprLegacy
                                }
                                Lsode2SymbolicAssemblyBackend::AtomView => {
                                    IvpSymbolicAssemblyBackend::AtomView
                                }
                            },
                            config.telemetry.clone(),
                        )?,
                    ))
                } else {
                    let parameter_values_handle = residual_problem.parameter_values_handle();
                    let callback = match symbolic_assembly_backend {
                        Lsode2SymbolicAssemblyBackend::ExprLegacy => {
                            compile_native_symbolic_jacobian_with_parameter_handle_and_telemetry_and_policy(
                                &config.eq_system,
                                &config.values,
                                config.arg.as_str(),
                                config.equation_parameters.as_deref(),
                                parameter_values_handle,
                                jacobian_storage,
                                IvpSymbolicAssemblyBackend::ExprLegacy,
                                config.telemetry.clone(),
                                config.lambdify_execution_policy,
                            )
                        }
                        Lsode2SymbolicAssemblyBackend::AtomView => {
                            try_compile_native_atomview_jacobian_with_parameter_handle_and_telemetry_and_policy(
                                &config.eq_system,
                                &config.values,
                                config.arg.as_str(),
                                config.equation_parameters.as_deref(),
                                parameter_values_handle,
                                jacobian_storage,
                                config.telemetry.clone(),
                                config.lambdify_execution_policy,
                            )?
                        }
                    };
                    Rc::new(RefCell::new(callback))
                };
                (
                    residual,
                    jacobian,
                    parameter_values_handle,
                    residual_evaluator_instrumented,
                    // The symbolic Jacobian callback records its own
                    // evaluator invocation for both Lambdify and AOT.
                    true,
                )
            }
            Lsode2JacobianBackend::AnalyticClosure => {
                config
                    .telemetry
                    .set_route(IvpTelemetryRoute::AnalyticalClosure);
                config
                    .telemetry
                    .set_execution(IvpTelemetryExecution::Lambdify);
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
                (residual, jacobian, None, false, false)
            }
            Lsode2JacobianBackend::FiniteDifference => {
                config
                    .telemetry
                    .set_route(IvpTelemetryRoute::FiniteDifference);
                config
                    .telemetry
                    .set_execution(IvpTelemetryExecution::Lambdify);
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
                let telemetry = config.telemetry.clone();
                let sparse_fd_coloring = sparse_fd_coloring.clone();
                let jacobian = Rc::new(RefCell::new(Box::new(move |t: f64, y: &DVector<f64>| {
                    finite_difference_jacobian_from_residual(
                        residual_for_jac.as_ref(),
                        t,
                        y,
                        atol,
                        jacobian_storage,
                        &telemetry,
                        sparse_fd_coloring.as_ref(),
                    )
                })
                    as Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian>));
                (residual, jacobian, None, false, false)
            }
        };

        Self::from_callbacks(
            config,
            method,
            linear_backend,
            residual,
            jacobian,
            parameter_values_handle,
            residual_evaluator_instrumented,
            jacobian_evaluator_instrumented,
        )
    }

    fn from_prepared_callbacks(
        config: &Lsode2ProblemConfig,
        method: Lsode2NativeStepMethod,
        linear_backend: L,
        callbacks: &PreparedNativeCallbacks,
    ) -> Result<Self, IvpBackendError> {
        Self::from_callbacks(
            config,
            method,
            linear_backend,
            Rc::clone(&callbacks.residual),
            Rc::clone(&callbacks.jacobian),
            callbacks.parameter_values_handle.clone(),
            callbacks.residual_evaluator_instrumented,
            callbacks.jacobian_evaluator_instrumented,
        )
    }

    fn from_callbacks(
        config: &Lsode2ProblemConfig,
        method: Lsode2NativeStepMethod,
        linear_backend: L,
        residual: Rc<NativeResidualFn>,
        jacobian: Rc<RefCell<Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian>>>,
        parameter_values_handle: Option<SharedIvpParameterValues>,
        residual_evaluator_instrumented: bool,
        jacobian_evaluator_instrumented: bool,
    ) -> Result<Self, IvpBackendError> {
        let h0 = initial_native_step_size(config, residual.as_ref(), &config.telemetry);
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
            parameter_values_handle,
            linear_backend,
            driver: Lsode2NonlinearStepDriver::new(cycle, correction),
            fallback_stiffness_probe_interval: (config.controller.mode
                == Lsode2ControllerMode::AutomaticAdamsBdf)
                .then_some(config.controller.method_switch_probe_steps.max(1)),
            telemetry: config.telemetry.clone(),
            residual_evaluator_instrumented,
            jacobian_evaluator_instrumented,
        })
    }

    fn into_prepared_callbacks(self) -> PreparedNativeCallbacks {
        PreparedNativeCallbacks {
            residual: self.residual,
            jacobian: self.jacobian,
            parameter_values_handle: self.parameter_values_handle,
            residual_evaluator_instrumented: self.residual_evaluator_instrumented,
            jacobian_evaluator_instrumented: self.jacobian_evaluator_instrumented,
        }
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
        let mut executor = Lsode2NativeCallbackExecutor::new_with_telemetry(
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
            self.telemetry.clone(),
        )
        .with_evaluator_telemetry(
            self.residual_evaluator_instrumented,
            self.jacobian_evaluator_instrumented,
        );

        loop {
            self.refresh_first_derivative_if_requested()?;
            let predicted = {
                let _predictor_scope = self
                    .telemetry
                    .scoped_warm_stage(IvpWarmStage::ControllerPredictor);
                self.driver
                    .begin_step()
                    .map_err(map_nonlinear_driver_error)?
            };
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
            let (h_trial, hl0, mut y_candidate) = {
                let _setup_scope = self
                    .telemetry
                    .scoped_warm_stage(IvpWarmStage::ControllerStepSetup);
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
                (h_trial, hl0, DVector::from_vec(predicted.y_pred.clone()))
            };
            let mut force_refresh_on_next_correction = should_force_refresh_on_first_correction(
                refresh_requested,
                self.driver.cycle().ipup(),
            );

            loop {
                total_iterations += 1;
                let outcome = {
                    let _iteration_scope = self
                        .telemetry
                        .scoped_warm_stage(IvpWarmStage::ControllerIteration);
                    self.driver
                        .compute_apply_and_submit_correction_with_refresh_policy(
                            &mut y_candidate,
                            hl0,
                            &mut executor,
                            force_refresh_on_next_correction,
                        )
                        .map_err(map_nonlinear_driver_error)?
                };
                force_refresh_on_next_correction = false;

                let _outcome_scope = self
                    .telemetry
                    .scoped_warm_stage(IvpWarmStage::ControllerOutcome);
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
        self.telemetry.record_jacobian_auxiliary_evaluation();
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
        self.telemetry.record_residual_auxiliary_evaluation();
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
    #[allow(dead_code)]
    fn reconcile_accepted_first_nordsieck_derivative(
        &mut self,
        t_new: f64,
        y_new: &[f64],
        h_trial: f64,
    ) -> Result<(), IvpBackendError> {
        let y_new_vec = DVector::from_vec(y_new.to_vec());
        let started = Instant::now();
        let accepted_rhs = (self.residual)(t_new, &y_new_vec);
        self.telemetry.record_residual_auxiliary_evaluation();
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

#[derive(Clone, Debug)]
struct SparseFiniteDifferenceColoring {
    groups: Vec<Vec<usize>>,
    rows_by_column: Vec<Vec<usize>>,
}

impl SparseFiniteDifferenceColoring {
    fn from_mask(mask: &DMatrix<f64>, dimension: usize) -> Result<Self, IvpBackendError> {
        if mask.nrows() != dimension || mask.ncols() != dimension {
            return Err(IvpBackendError::InvalidMatrixShape {
                stage: "finite-difference Jacobian sparsity".to_string(),
                expected_rows: dimension,
                expected_cols: dimension,
                actual_rows: mask.nrows(),
                actual_cols: mask.ncols(),
            });
        }

        let mut entries = Vec::new();
        for row in 0..dimension {
            for col in 0..dimension {
                if mask[(row, col)] != 0.0 {
                    entries.push((row, col));
                }
            }
        }

        Self::from_coordinates(dimension, entries)
    }

    fn from_sparse_pattern(
        pattern: &Lsode2SparseJacobianPattern,
        dimension: usize,
    ) -> Result<Self, IvpBackendError> {
        if pattern.dimension() != dimension {
            return Err(IvpBackendError::InvalidMatrixShape {
                stage: "finite-difference Jacobian sparsity".to_string(),
                expected_rows: dimension,
                expected_cols: dimension,
                actual_rows: pattern.dimension(),
                actual_cols: pattern.dimension(),
            });
        }

        for &(row, col) in pattern.entries() {
            if row >= dimension || col >= dimension {
                return Err(IvpBackendError::InvalidSparsePatternEntry {
                    row,
                    col,
                    dimension,
                });
            }
        }

        Ok(Self::from_validated_coordinates(
            dimension,
            pattern.entries().iter().copied(),
        ))
    }

    fn from_coordinates(
        dimension: usize,
        mut entries: Vec<(usize, usize)>,
    ) -> Result<Self, IvpBackendError> {
        for &(row, col) in &entries {
            if row >= dimension || col >= dimension {
                return Err(IvpBackendError::InvalidSparsePatternEntry {
                    row,
                    col,
                    dimension,
                });
            }
        }
        entries.sort_unstable();
        entries.dedup();

        Ok(Self::from_validated_coordinates(
            dimension,
            entries.into_iter(),
        ))
    }

    fn from_validated_coordinates(
        dimension: usize,
        entries: impl Iterator<Item = (usize, usize)>,
    ) -> Self {
        let mut rows_by_column = vec![Vec::new(); dimension];
        let mut columns_by_row = vec![Vec::new(); dimension];
        for (row, col) in entries {
            rows_by_column[col].push(row);
            columns_by_row[row].push(col);
        }

        let mut colors = vec![usize::MAX; dimension];
        let mut color_marks = vec![usize::MAX; dimension];
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for col in 0..dimension {
            for &row in &rows_by_column[col] {
                for &neighbor in &columns_by_row[row] {
                    if neighbor < col {
                        color_marks[colors[neighbor]] = col;
                    }
                }
            }
            let mut color = 0;
            while color_marks[color] == col {
                color += 1;
            }
            if color == groups.len() {
                groups.push(Vec::new());
            }
            colors[col] = color;
            groups[color].push(col);
        }

        Self {
            groups,
            rows_by_column,
        }
    }
}

fn finite_difference_jacobian_from_residual(
    residual: &NativeResidualFn,
    t: f64,
    y: &DVector<f64>,
    atol: f64,
    storage: NativeJacobianStorage,
    telemetry: &IvpTelemetry,
    sparse_coloring: Option<&SparseFiniteDifferenceColoring>,
) -> BdfJacobian {
    let n = y.len();
    let f0 = evaluate_finite_difference_residual(residual, t, y, telemetry);
    let eps = f64::EPSILON.sqrt();

    match storage {
        NativeJacobianStorage::Dense => {
            let mut dense = DMatrix::<f64>::zeros(n, n);
            let mut y_pert = clone_perturbation_state(y, telemetry);
            for col in 0..n {
                let yj = y[col];
                let h = eps * yj.abs().max(atol).max(1.0);
                y_pert[col] += h;
                let f_pert = evaluate_finite_difference_residual(residual, t, &y_pert, telemetry);
                y_pert[col] = yj;
                for row in 0..n {
                    dense[(row, col)] = (f_pert[row] - f0[row]) / h;
                }
            }
            BdfJacobian::from_dense(dense)
        }
        NativeJacobianStorage::SparseTriplets => {
            let mut triplets = Vec::new();
            let mut y_pert = clone_perturbation_state(y, telemetry);
            if let Some(coloring) = sparse_coloring {
                for group in &coloring.groups {
                    for &col in group {
                        let h = eps * y[col].abs().max(atol).max(1.0);
                        y_pert[col] += h;
                    }
                    let f_pert =
                        evaluate_finite_difference_residual(residual, t, &y_pert, telemetry);
                    for &col in group {
                        let h = eps * y[col].abs().max(atol).max(1.0);
                        y_pert[col] = y[col];
                        for &row in &coloring.rows_by_column[col] {
                            let value = (f_pert[row] - f0[row]) / h;
                            if value != 0.0 {
                                triplets.push(faer::sparse::Triplet::new(row, col, value));
                            }
                        }
                    }
                }
                return BdfJacobian::SparseTriplets { n, triplets };
            }
            for col in 0..n {
                let yj = y[col];
                let h = eps * yj.abs().max(atol).max(1.0);
                y_pert[col] += h;
                let f_pert = evaluate_finite_difference_residual(residual, t, &y_pert, telemetry);
                y_pert[col] = yj;
                for row in 0..n {
                    let value = (f_pert[row] - f0[row]) / h;
                    if value != 0.0 {
                        triplets.push(faer::sparse::Triplet::new(row, col, value));
                    }
                }
            }
            BdfJacobian::SparseTriplets { n, triplets }
        }
        NativeJacobianStorage::Banded {
            bandwidth: Some((kl, ku)),
        } => finite_difference_banded_jacobian_from_residual(
            residual, t, y, &f0, atol, kl, ku, telemetry,
        ),
        NativeJacobianStorage::Banded { bandwidth: None } => {
            let mut kl = 0usize;
            let mut ku = 0usize;
            let mut entries = Vec::new();
            let mut y_pert = clone_perturbation_state(y, telemetry);
            for col in 0..n {
                let yj = y[col];
                let h = eps * yj.abs().max(atol).max(1.0);
                y_pert[col] += h;
                let f_pert = evaluate_finite_difference_residual(residual, t, &y_pert, telemetry);
                y_pert[col] = yj;
                for row in 0..n {
                    let value = (f_pert[row] - f0[row]) / h;
                    if value != 0.0 {
                        kl = kl.max(row.saturating_sub(col));
                        ku = ku.max(col.saturating_sub(row));
                        entries.push((row, col, value));
                    }
                }
            }
            let mut banded = Banded::<f64>::zeros(n, kl, ku)
                .expect("finite-difference Jacobian bandwidth should define valid banded storage");
            for (row, col, value) in entries {
                banded
                    .set(row, col, value)
                    .expect("finite-difference entry should be inside inferred band");
            }
            BdfJacobian::Banded(banded)
        }
    }
}

fn finite_difference_banded_jacobian_from_residual(
    residual: &NativeResidualFn,
    t: f64,
    y: &DVector<f64>,
    f0: &DVector<f64>,
    atol: f64,
    kl: usize,
    ku: usize,
    telemetry: &IvpTelemetry,
) -> BdfJacobian {
    let n = y.len();
    let eps = f64::EPSILON.sqrt();
    let mut banded = Banded::<f64>::zeros(n, kl, ku)
        .expect("finite-difference Jacobian bandwidth should define valid banded storage");
    let mut y_pert = clone_perturbation_state(y, telemetry);
    let color_stride = kl.saturating_add(ku).saturating_add(1).max(1);

    // Columns in one color have disjoint row support for the declared band.
    // Perturbing them together reduces residual evaluations to the bandwidth.
    for color in 0..n.min(color_stride) {
        for col in (color..n).step_by(color_stride) {
            let h = eps * y[col].abs().max(atol).max(1.0);
            y_pert[col] += h;
        }
        let f_pert = evaluate_finite_difference_residual(residual, t, &y_pert, telemetry);
        for col in (color..n).step_by(color_stride) {
            let h = eps * y[col].abs().max(atol).max(1.0);
            y_pert[col] = y[col];
            let first_row = col.saturating_sub(ku);
            let last_row = (col + kl).min(n.saturating_sub(1));
            for row in first_row..=last_row {
                let value = (f_pert[row] - f0[row]) / h;
                banded
                    .set(row, col, value)
                    .expect("finite-difference entry should be inside declared band");
            }
        }
    }

    BdfJacobian::Banded(banded)
}

fn clone_perturbation_state(y: &DVector<f64>, telemetry: &IvpTelemetry) -> DVector<f64> {
    let bytes = y.len().saturating_mul(std::mem::size_of::<f64>());
    telemetry.record_copy_bytes(bytes);
    telemetry.record_allocation(bytes);
    y.clone_owned()
}

fn evaluate_finite_difference_residual(
    residual: &NativeResidualFn,
    t: f64,
    y: &DVector<f64>,
    telemetry: &IvpTelemetry,
) -> DVector<f64> {
    let started = telemetry.start_warm_stage(IvpWarmStage::ResidualEvaluation);
    let value = residual(t, y);
    telemetry.record_residual_evaluation(started);
    value
}

fn banded_jacobian_storage(config: &Lsode2ProblemConfig) -> NativeJacobianStorage {
    match config.linear_system_structure {
        Lsode2LinearSystemStructure::Banded { kl: 0, ku: 0 } => {
            NativeJacobianStorage::Banded { bandwidth: None }
        }
        Lsode2LinearSystemStructure::Banded { kl, ku } => NativeJacobianStorage::Banded {
            bandwidth: Some((kl, ku)),
        },
        _ => NativeJacobianStorage::Banded { bandwidth: None },
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

fn initial_native_step_size(
    config: &Lsode2ProblemConfig,
    residual: &NativeResidualFn,
    telemetry: &IvpTelemetry,
) -> f64 {
    let direction = if config.t_bound >= config.t0 {
        1.0
    } else {
        -1.0
    };
    let span = (config.t_bound - config.t0).abs();

    let h0_mag = {
        telemetry.record_residual_preparation_evaluation();
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
            telemetry.record_residual_preparation_evaluation();
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
    if direction > 0.0 { mag } else { -mag }
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
    fn finite_difference_sparse_storage_emits_triplets_without_dense_wrapper() {
        let residual = |_: f64, y: &DVector<f64>| DVector::from_vec(vec![2.0 * y[0], -3.0 * y[1]]);
        let y = DVector::from_vec(vec![1.5, -2.0]);
        let telemetry = IvpTelemetry::counters();

        let jacobian = finite_difference_jacobian_from_residual(
            &residual,
            0.0,
            &y,
            1.0e-8,
            NativeJacobianStorage::SparseTriplets,
            &telemetry,
            None,
        );

        let BdfJacobian::SparseTriplets { n, triplets } = jacobian else {
            panic!("finite-difference sparse storage should not return a dense Jacobian");
        };
        assert_eq!(n, 2);
        assert_eq!(triplets.len(), 2);
        assert!(triplets.iter().any(|triplet| triplet.row == 0
            && triplet.col == 0
            && (triplet.val - 2.0).abs() < 1e-8));
        assert!(triplets.iter().any(|triplet| triplet.row == 1
            && triplet.col == 1
            && (triplet.val + 3.0).abs() < 1e-8));
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.copies, 1);
        assert_eq!(snapshot.copied_bytes, 2 * std::mem::size_of::<f64>() as u64);
        assert_eq!(
            snapshot.allocated_bytes,
            2 * std::mem::size_of::<f64>() as u64
        );
    }

    #[test]
    fn finite_difference_sparse_coloring_uses_declared_pattern_and_checks_shape() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let n = 9;
        let mut mask = DMatrix::zeros(n, n);
        for row in 0..n {
            mask[(row, row)] = 1.0;
            if row > 0 {
                mask[(row, row - 1)] = 1.0;
            }
            if row + 1 < n {
                mask[(row, row + 1)] = 1.0;
            }
        }
        let coloring = SparseFiniteDifferenceColoring::from_mask(&mask, n)
            .expect("tridiagonal sparsity mask should prepare");
        assert_eq!(coloring.groups.len(), 3);

        let calls = Arc::new(AtomicUsize::new(0));
        let observed_calls = Arc::clone(&calls);
        let residual = move |_: f64, y: &DVector<f64>| {
            observed_calls.fetch_add(1, Ordering::Relaxed);
            DVector::from_fn(y.len(), |row, _| {
                let mut value = y[row] * y[row];
                if row > 0 {
                    value += 0.25 * y[row - 1];
                }
                if row + 1 < y.len() {
                    value -= 0.75 * y[row + 1];
                }
                value
            })
        };
        let y = DVector::from_iterator(n, (0..n).map(|index| 1.0 + index as f64));
        let original = y.clone();
        let telemetry = IvpTelemetry::counters();
        let jacobian = finite_difference_jacobian_from_residual(
            &residual,
            0.0,
            &y,
            1.0e-8,
            NativeJacobianStorage::SparseTriplets,
            &telemetry,
            Some(&coloring),
        );

        let BdfJacobian::SparseTriplets {
            n: result_n,
            triplets,
        } = jacobian
        else {
            panic!("colored sparse finite differences should return triplets");
        };
        assert_eq!(result_n, n);
        assert_eq!(
            calls.load(Ordering::Relaxed),
            4,
            "one base plus three colors"
        );
        assert_eq!(
            telemetry
                .snapshot()
                .warm_stage(IvpWarmStage::ResidualEvaluation)
                .calls,
            4
        );
        assert_eq!(
            y, original,
            "colored perturbations must restore caller state"
        );

        let mut actual = DMatrix::zeros(n, n);
        for triplet in triplets {
            actual[(triplet.row, triplet.col)] = triplet.val;
        }
        let eps = f64::EPSILON.sqrt();
        for col in 0..n {
            let h = eps * y[col].abs().max(1.0e-8).max(1.0);
            assert!((actual[(col, col)] - (2.0 * y[col] + h)).abs() < 2.0e-7);
            if col > 0 {
                assert!((actual[(col - 1, col)] + 0.75).abs() < 2.0e-7);
            }
            if col + 1 < n {
                assert!((actual[(col + 1, col)] - 0.25).abs() < 2.0e-7);
            }
        }

        let wrong_shape = DMatrix::zeros(n, n - 1);
        assert!(matches!(
            SparseFiniteDifferenceColoring::from_mask(&wrong_shape, n),
            Err(IvpBackendError::InvalidMatrixShape {
                stage,
                expected_rows,
                expected_cols,
                actual_rows,
                actual_cols,
            }) if stage == "finite-difference Jacobian sparsity"
                && expected_rows == n
                && expected_cols == n
                && actual_rows == n
                && actual_cols == n - 1
        ));

        let coordinates = (0..n)
            .flat_map(|row| {
                let mut entries = vec![(row, row)];
                if row > 0 {
                    entries.push((row, row - 1));
                }
                if row + 1 < n {
                    entries.push((row, row + 1));
                }
                entries
            })
            .collect::<Vec<_>>();
        let compact = Lsode2SparseJacobianPattern::new(n, coordinates.clone());
        let compact_coloring = SparseFiniteDifferenceColoring::from_sparse_pattern(&compact, n)
            .expect("compact tridiagonal pattern should prepare");
        assert_eq!(compact_coloring.groups, coloring.groups);
        assert_eq!(
            compact_coloring.rows_by_column, coloring.rows_by_column,
            "dense and compact patterns must produce the same structure"
        );
        assert_eq!(compact.entries().len(), 3 * n - 2);

        let wrong_dimension = Lsode2SparseJacobianPattern::new(n - 1, coordinates.clone());
        assert!(matches!(
            SparseFiniteDifferenceColoring::from_sparse_pattern(&wrong_dimension, n),
            Err(IvpBackendError::InvalidMatrixShape {
                stage,
                expected_rows,
                expected_cols,
                actual_rows,
                actual_cols,
            }) if stage == "finite-difference Jacobian sparsity"
                && expected_rows == n
                && expected_cols == n
                && actual_rows == n - 1
                && actual_cols == n - 1
        ));

        let invalid_index = Lsode2SparseJacobianPattern::new(n, vec![(n, 0)]);
        assert!(matches!(
            SparseFiniteDifferenceColoring::from_sparse_pattern(&invalid_index, n),
            Err(IvpBackendError::InvalidSparsePatternEntry {
                row,
                col,
                dimension,
            }) if row == n && col == 0 && dimension == n
        ));

        let duplicate = Lsode2SparseJacobianPattern::new(n, {
            let mut entries = coordinates;
            entries.push((0, 0));
            entries
        });
        let duplicate_coloring = SparseFiniteDifferenceColoring::from_sparse_pattern(&duplicate, n)
            .expect("duplicate coordinates should be normalized");
        assert_eq!(duplicate_coloring.rows_by_column, coloring.rows_by_column);
    }

    #[test]
    fn sparse_fd_compact_pattern_storage_tracks_nnz_not_dense_dimension_squared() {
        let n = 2048;
        let entries = (0..n)
            .flat_map(|row| {
                let mut row_entries = vec![(row, row)];
                if row > 0 {
                    row_entries.push((row, row - 1));
                }
                if row + 1 < n {
                    row_entries.push((row, row + 1));
                }
                row_entries
            })
            .collect::<Vec<_>>();
        let nnz = entries.len();
        let pattern = Lsode2SparseJacobianPattern::new(n, entries);
        let coloring = SparseFiniteDifferenceColoring::from_sparse_pattern(&pattern, n)
            .expect("large tridiagonal compact pattern should prepare");

        assert_eq!(nnz, 3 * n - 2);
        assert_eq!(pattern.entries().len(), nnz);
        assert_eq!(
            coloring.rows_by_column.iter().map(Vec::len).sum::<usize>(),
            nnz
        );
        assert_eq!(coloring.groups.len(), 3);
        assert!(pattern.entries().len() < n * n / 100);
    }

    #[test]
    fn sparse_fd_engine_uses_configured_jacobian_sparsity() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let n = 9;
        let mut mask = DMatrix::zeros(n, n);
        for row in 0..n {
            mask[(row, row)] = 1.0;
            if row > 0 {
                mask[(row, row - 1)] = 1.0;
            }
            if row + 1 < n {
                mask[(row, row + 1)] = 1.0;
            }
        }
        let calls = Arc::new(AtomicUsize::new(0));
        let observed_calls = Arc::clone(&calls);
        let y0 = DVector::from_iterator(n, (0..n).map(|index| 1.0 + index as f64));
        let config = Lsode2ProblemConfig::new_numeric_fd(
            (0..n).map(|index| format!("y{index}")).collect(),
            "t".to_string(),
            0.0,
            y0.clone(),
            0.1,
            0.01,
            1.0e-6,
            1.0e-8,
            move |_, y: &DVector<f64>| {
                observed_calls.fetch_add(1, Ordering::Relaxed);
                DVector::from_fn(y.len(), |row, _| {
                    let mut value = y[row] * y[row];
                    if row > 0 {
                        value += 0.25 * y[row - 1];
                    }
                    if row + 1 < y.len() {
                        value -= 0.75 * y[row + 1];
                    }
                    value
                })
            },
        )
        .with_linear_system_structure(Lsode2LinearSystemStructure::Sparse)
        .with_jac_sparsity(Some(mask.clone()));

        let mut engine = Lsode2NativeStepEngine::from_problem_config(&config)
            .expect("sparse FD engine should accept matching sparsity")
            .expect("sparse FD config should create native engine");
        let Lsode2NativeStepEngine::Sparse(engine) = &mut engine else {
            panic!("SparseFaer config should create sparse native engine");
        };
        let calls_before_jacobian = calls.load(Ordering::Relaxed);
        let jacobian = (engine.jacobian.borrow_mut())(0.0, &y0);
        let BdfJacobian::SparseTriplets { triplets, .. } = jacobian else {
            panic!("sparse FD backend should emit triplets");
        };

        assert_eq!(
            calls.load(Ordering::Relaxed) - calls_before_jacobian,
            4,
            "one base plus three colors"
        );
        assert_eq!(triplets.len(), 3 * n - 2);

        let invalid_config = config
            .clone()
            .with_jac_sparsity(Some(DMatrix::zeros(n, n - 1)));
        assert!(matches!(
            Lsode2NativeStepEngine::from_problem_config(&invalid_config),
            Err(IvpBackendError::InvalidMatrixShape {
                stage,
                expected_rows,
                expected_cols,
                actual_rows,
                actual_cols,
            }) if stage == "finite-difference Jacobian sparsity"
                && expected_rows == n
                && expected_cols == n
                && actual_rows == n
                && actual_cols == n - 1
        ));

        let compact_entries = (0..n)
            .flat_map(|row| {
                let mut entries = vec![(row, row)];
                if row > 0 {
                    entries.push((row, row - 1));
                }
                if row + 1 < n {
                    entries.push((row, row + 1));
                }
                entries
            })
            .collect();
        let compact_config = config
            .clone()
            .with_jac_sparsity(None)
            .with_sparse_jacobian_pattern(Some(Lsode2SparseJacobianPattern::new(
                n,
                compact_entries,
            )));
        let mut compact_engine = Lsode2NativeStepEngine::from_problem_config(&compact_config)
            .expect("compact sparse FD engine should prepare")
            .expect("compact Sparse config should create native engine");
        let Lsode2NativeStepEngine::Sparse(compact_engine) = &mut compact_engine else {
            panic!("compact SparseFaer config should create sparse native engine");
        };
        let calls_before_compact_jacobian = calls.load(Ordering::Relaxed);
        let compact_jacobian = (compact_engine.jacobian.borrow_mut())(0.0, &y0);
        let BdfJacobian::SparseTriplets {
            triplets: compact_triplets,
            ..
        } = compact_jacobian
        else {
            panic!("compact sparse FD backend should emit triplets");
        };
        assert_eq!(
            calls.load(Ordering::Relaxed) - calls_before_compact_jacobian,
            4,
            "compact mask should retain one base plus three color evaluations"
        );
        assert_eq!(compact_triplets.len(), 3 * n - 2);
        let canonicalize = |triplets: &[faer::sparse::Triplet<usize, usize, f64>]| {
            let mut entries = triplets
                .iter()
                .map(|triplet| (triplet.row, triplet.col, triplet.val))
                .collect::<Vec<_>>();
            entries.sort_by_key(|&(row, col, _)| (row, col));
            entries
        };
        assert_eq!(
            canonicalize(&compact_triplets),
            canonicalize(&triplets),
            "dense-mask and compact-pattern FD Jacobians should match"
        );

        let conflicting_config = compact_config.clone().with_jac_sparsity(Some(mask));
        assert!(matches!(
            Lsode2NativeStepEngine::from_problem_config(&conflicting_config),
            Err(IvpBackendError::InvalidArgumentSchema { message })
                if message.contains("either dense jac_sparsity or sparse_jacobian_pattern")
        ));
    }

    #[test]
    fn finite_difference_storage_reuses_base_residual_and_one_state_copy() {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };

        for storage in [
            NativeJacobianStorage::Dense,
            NativeJacobianStorage::SparseTriplets,
            NativeJacobianStorage::Banded { bandwidth: None },
            NativeJacobianStorage::Banded {
                bandwidth: Some((0, 0)),
            },
        ] {
            let calls = Arc::new(AtomicUsize::new(0));
            let observed = Arc::clone(&calls);
            let residual = move |_: f64, y: &DVector<f64>| {
                observed.fetch_add(1, Ordering::Relaxed);
                y * 2.0
            };
            let y = DVector::from_vec(vec![1.0, -2.0, 3.0]);
            let original = y.clone();
            let telemetry = IvpTelemetry::counters();
            let _jacobian = finite_difference_jacobian_from_residual(
                &residual, 0.0, &y, 1e-8, storage, &telemetry, None,
            );
            let expected_calls = match storage {
                NativeJacobianStorage::Banded {
                    bandwidth: Some((0, 0)),
                } => 2,
                _ => y.len() + 1,
            };
            assert_eq!(calls.load(Ordering::Relaxed), expected_calls);
            assert_eq!(y, original);
            let snapshot = telemetry.snapshot();
            assert_eq!(
                snapshot.warm_stage(IvpWarmStage::ResidualEvaluation).calls,
                expected_calls as u64
            );
            assert_eq!(snapshot.copies, 1);
            assert_eq!(snapshot.copied_bytes, 24);
            // This is the instrumented perturbation buffer, not allocator-wide usage.
            assert_eq!(snapshot.allocated_bytes, 24);
        }
    }

    #[test]
    fn finite_difference_banded_storage_infers_compact_bandwidth() {
        let residual =
            |_: f64, y: &DVector<f64>| DVector::from_vec(vec![y[0] + 5.0 * y[1], -2.0 * y[1]]);
        let y = DVector::from_vec(vec![1.0, 2.0]);

        let jacobian = finite_difference_jacobian_from_residual(
            &residual,
            0.0,
            &y,
            1.0e-8,
            NativeJacobianStorage::Banded { bandwidth: None },
            &IvpTelemetry::disabled(),
            None,
        );

        let BdfJacobian::Banded(banded) = jacobian else {
            panic!("finite-difference banded storage should not return a dense Jacobian");
        };
        assert_eq!(banded.n(), 2);
        assert_eq!(banded.kl(), 0);
        assert_eq!(banded.ku(), 1);
        assert!((banded[(0, 0)] - 1.0).abs() < 1e-8);
        assert!((banded[(0, 1)] - 5.0).abs() < 1e-8);
        assert!((banded[(1, 1)] + 2.0).abs() < 1e-8);
    }

    #[test]
    fn finite_difference_declared_banded_coloring_reduces_residual_calls() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let calls = Arc::new(AtomicUsize::new(0));
        let observed_calls = Arc::clone(&calls);
        let residual = move |_: f64, y: &DVector<f64>| {
            observed_calls.fetch_add(1, Ordering::Relaxed);
            DVector::from_fn(y.len(), |row, _| {
                let mut value = y[row] * y[row];
                if row > 0 {
                    value += 0.25 * y[row - 1];
                }
                if row + 1 < y.len() {
                    value -= 0.75 * y[row + 1];
                }
                value
            })
        };
        let n = 9;
        let y = DVector::from_iterator(n, (0..n).map(|index| 1.0 + index as f64));
        let original = y.clone();
        let telemetry = IvpTelemetry::counters();
        let jacobian = finite_difference_jacobian_from_residual(
            &residual,
            0.0,
            &y,
            1.0e-8,
            NativeJacobianStorage::Banded {
                bandwidth: Some((1, 1)),
            },
            &telemetry,
            None,
        );

        let BdfJacobian::Banded(banded) = jacobian else {
            panic!("declared finite-difference banded storage should return banded Jacobian");
        };
        assert_eq!(
            calls.load(Ordering::Relaxed),
            4,
            "one base plus three colors"
        );
        assert_eq!(
            telemetry
                .snapshot()
                .warm_stage(IvpWarmStage::ResidualEvaluation)
                .calls,
            4
        );
        assert_eq!(
            y, original,
            "color perturbations must restore the caller state"
        );

        let eps = f64::EPSILON.sqrt();
        for col in 0..y.len() {
            let h = eps * y[col].abs().max(1.0e-8).max(1.0);
            assert!((banded[(col, col)] - (2.0 * y[col] + h)).abs() < 2.0e-7);
            if col > 0 {
                assert!((banded[(col - 1, col)] + 0.75).abs() < 2.0e-7);
            }
            if col + 1 < y.len() {
                assert!((banded[(col + 1, col)] - 0.25).abs() < 2.0e-7);
            }
        }
    }

    #[test]
    fn finite_difference_banded_storage_respects_declared_bandwidth() {
        let residual =
            |_: f64, y: &DVector<f64>| DVector::from_vec(vec![y[0], 2.0 * y[1], 3.0 * y[2]]);
        let y = DVector::from_vec(vec![1.0, 2.0, 3.0]);

        let jacobian = finite_difference_jacobian_from_residual(
            &residual,
            0.0,
            &y,
            1.0e-8,
            NativeJacobianStorage::Banded {
                bandwidth: Some((0, 0)),
            },
            &IvpTelemetry::disabled(),
            None,
        );

        let BdfJacobian::Banded(banded) = jacobian else {
            panic!("declared finite-difference banded storage should return banded Jacobian");
        };
        assert_eq!(banded.n(), 3);
        assert_eq!(banded.kl(), 0);
        assert_eq!(banded.ku(), 0);
        assert!((banded[(0, 0)] - 1.0).abs() < 1e-8);
        assert!((banded[(1, 1)] - 2.0).abs() < 1e-8);
        assert!((banded[(2, 2)] - 3.0).abs() < 1e-8);
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
        assert!(
            inner
                .driver
                .cycle()
                .state()
                .first_derivative_refresh_requested()
        );

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
        assert!(
            inner
                .driver
                .cycle()
                .state()
                .first_derivative_refresh_requested()
        );

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
