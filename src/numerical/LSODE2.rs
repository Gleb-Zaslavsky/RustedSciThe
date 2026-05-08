//! Experimental second-generation LSODE-style IVP solver facade.
//!
//! `LSODE2` is intentionally not a patch over the older [`crate::numerical::LSODE`]
//! prototype.  The first milestone reuses the crate's tested BDF implementation
//! as the time-integration engine, while exposing a cleaner configuration surface
//! for the backends that an LSODE-class solver needs:
//!
//! - generated symbolic RHS/Jacobian backends (`Lambdify` / AOT),
//! - dense, sparse, and banded Jacobian/system matrix policies,
//! - production linear solvers from the rest of the crate instead of local
//!   hand-written factorization code.
//!
//! Today this module is a small, working facade around the tested BDF solver.
//! The backend enums are deliberately present from the start so later work can
//! replace the dense-only BDF internals with `faer` sparse and faithful LAPACK
//! banded adapters without changing the user-facing setup shape again.

pub mod adams_engine;
pub mod algorithm;
pub mod config;
pub mod correction;
pub mod dcfode;
pub mod dstoda_state;
pub mod error_control;
pub mod history;
pub mod linear_backends;
pub mod method_switch;
pub mod native_executor;
pub mod native_integration;
pub mod native_jacobian;
pub mod native_preflight;
pub mod native_step_engine;
pub mod nonlinear_driver;
pub mod order_selection;
pub mod solver;
pub mod state;
pub mod statistics;
pub mod step_control;
pub mod step_cycle;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tests2;
#[cfg(test)]
mod story_tests;

#[cfg(test)]
mod story_tests2;
#[cfg(test)]
mod parity_micro;

pub use adams_engine::{
    Lsode2AdamsDcfodeError, Lsode2AdamsDcfodeTables, Lsode2AdamsOrderCoefficients,
};
pub use algorithm::{
    Lsode2AlgorithmController, Lsode2AlgorithmSnapshot, Lsode2ControllerConfig,
    Lsode2ControllerExecutionPlan, Lsode2ControllerMode, Lsode2MethodFamily, Lsode2SwitchDecision,
    Lsode2SwitchReason, Lsode2SwitchTelemetry,
};
pub use config::{
    Lsode2AnalyticalCallbacks, Lsode2AnalyticalJacobianCallback, Lsode2AnalyticalResidualCallback,
    Lsode2AotProfile, Lsode2AotToolchain, Lsode2BackendConfig, Lsode2JacobianBackend,
    Lsode2LinearSolverBackend, Lsode2LinearSolverChoice, Lsode2LinearSolverPolicy,
    Lsode2LinearSystemStructure, Lsode2Method, Lsode2NativeExecutionConfig, Lsode2ProblemConfig,
    Lsode2ResidualJacobianSource, Lsode2ResolvedPlan, Lsode2SymbolicAssemblyBackend,
    Lsode2SymbolicExecutionMode,
};
pub use correction::{
    Lsode2CorrectionAssessment, Lsode2CorrectionControlConfig, Lsode2CorrectionController,
    Lsode2CorrectionError, Lsode2CorrectionStatus,
};
pub use dcfode::{Lsode2BdfDcfodeTables, Lsode2BdfOrderCoefficients, Lsode2DcfodeError};
pub use dstoda_state::{
    Lsode2CorrectorFailureDecision, Lsode2CorrectorFailureMode, Lsode2DstodaState, Lsode2Icf,
    Lsode2Ipup, Lsode2IpupTrigger, Lsode2Iredo, Lsode2Iret, Lsode2IterationMode,
    Lsode2JacobianCurrency, Lsode2JacobianUpdateRequest, Lsode2Kflag, Lsode2RedoStage,
};
pub use error_control::{
    Lsode2ErrorControlConfig, Lsode2ErrorControlError, Lsode2ErrorController,
    Lsode2ErrorTestAction, Lsode2ErrorTestResult,
};
pub use history::{
    Lsode2HistoryError, Lsode2NordsieckHistory, Lsode2Tolerance, Lsode2YHistory, error_weights,
    weighted_rms_norm,
};
pub use linear_backends::{FaerSparseBdfLinearBackend, FaithfulBandedBdfLinearBackend};
pub use native_executor::{Lsode2NativeCallbackExecutor, Lsode2NativeExecutorError};
pub use native_integration::{
    Lsode2NativeIntegrationLimits, Lsode2NativeIntegrationOutcome, Lsode2NativeIntegrationSummary,
    run_native_integration, run_native_integration_for_method,
};
pub use native_preflight::{Lsode2NativePreflightOutcome, Lsode2NativeStepProbeSummary};
pub use native_step_engine::{
    Lsode2NativeStepAttemptReport, Lsode2NativeStepEngine, Lsode2NativeStepMethod,
};
pub use nonlinear_driver::{Lsode2NonlinearDriverError, Lsode2NonlinearStepDriver};
pub use order_selection::{
    Lsode2OrderCandidate, Lsode2OrderSelectionCandidates, Lsode2OrderSelectionConfig,
    Lsode2OrderSelectionDecision, Lsode2OrderSelectionError, select_bdf_like_order,
};
pub use solver::{Lsode2Error, Lsode2SolveSummary, Lsode2Solver};
pub use state::{Lsode2RuntimeState, Lsode2RuntimeStateError, Lsode2RuntimeStateSnapshot};
pub use statistics::Lsode2NativeStatistics;
pub use step_control::{
    Lsode2AcceptDecision, Lsode2RetryAction, Lsode2RetryDecision, Lsode2StepControlConfig,
    Lsode2StepControlError, Lsode2StepControlSnapshot, Lsode2StepController, Lsode2StepFailure,
};
pub use step_cycle::{
    Lsode2PredictedStep, Lsode2StepCycle, Lsode2StepCycleError, Lsode2StepCycleOutcome,
};
