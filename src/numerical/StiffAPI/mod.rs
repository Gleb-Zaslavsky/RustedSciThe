//! Typed, thin configuration facade for production stiff IVP solvers.
//!
//! The facade only selects and forwards policy. Numerical work, symbolic
//! preparation, AOT lifecycle, telemetry, logging and solver state remain
//! owned by the selected native implementation.

use std::error::Error;
use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use nalgebra::{DMatrix, DVector};

use crate::Utils::postprocessing::{
    PostprocessDataset, PostprocessError, PostprocessPlan, PostprocessReport,
};
use crate::numerical::BDF::BDF_api::{
    BdfSolveError, BdfSolverOptions, BdfTelemetryMode, ODEsolver as BdfSolver,
};
use crate::numerical::BE::{
    BE, BeError, BeSolverOptions, BeSymbolicAssemblyBackend, BeTelemetryMode,
};
use crate::numerical::CancellationToken;
use crate::numerical::LSODE2::{
    Lsode2AotProfile, Lsode2AotToolchain, Lsode2Error, Lsode2LinearSolverChoice,
    Lsode2LinearSolverPolicy, Lsode2LinearSystemStructure, Lsode2NativeStatistics,
    Lsode2ProblemConfig, Lsode2ResidualJacobianSource, Lsode2Solver, Lsode2SymbolicAssemblyBackend,
    Lsode2SymbolicExecutionMode,
};
use crate::numerical::Radau::api::{
    RadauAotConfig, RadauCancellationError, RadauConfig, RadauError, RadauExecution,
    RadauExecutionPolicy, RadauFrontend, RadauMatrixLayout, RadauSolution, RadauSolver,
    RadauTelemetryReport,
};
use crate::numerical::SolverTelemetryMode;
use crate::numerical::ivp_parameter_loop::{
    IvpParameterLoopPoint, IvpParameterLoopSolution, run_ordered,
};
pub use crate::numerical::logging::SolverLogLevel;
use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use crate::symbolic::ivp_telemetry::IvpLambdifyExecutionPolicy;
use crate::symbolic::ivp_telemetry::IvpTelemetrySnapshot;
use crate::symbolic::substitutions::SymbolicSubstitutionError;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp::IvpSymbolicAssemblyBackend;
use crate::symbolic::symbolic_ivp_generated::{
    IvpBackendStatistics, SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
};

/// Native BDF types re-exported at the facade boundary. These remain the
/// solver's canonical configuration and result types; StiffAPI does not clone
/// them into a parallel configuration hierarchy.
pub mod bdf {
    pub use crate::numerical::BDF::BDF_api::{
        BdfSolveError, BdfSolverOptions, BdfTelemetryMode, ODEsolver as Solver,
    };
}

/// Native Radau types re-exported at the facade boundary.
pub mod radau {
    pub use crate::numerical::Radau::api::*;
}

/// Native Backward Euler types re-exported at the facade boundary.
pub mod backward_euler {
    pub use crate::numerical::BE::NR_for_Euler::{NRE, NreError, NreSolverOptions, NreStepMode};
    pub use crate::numerical::BE::{
        BE as Solver, BeDetailedStatistics, BeError, BeSolverOptions, BeStatus,
        BeSymbolicAssemblyBackend, BeTelemetryMode, DEFAULT_BE_MAX_STEPS,
    };
}

/// Production LSODE2 types re-exported at the facade boundary.
pub mod lsode2 {
    pub use crate::numerical::LSODE2::{
        IvpTelemetry, IvpTelemetryMode, Lsode2AotProfile, Lsode2AotToolchain, Lsode2BackendConfig,
        Lsode2ControllerConfig, Lsode2Error, Lsode2JacobianBackend, Lsode2LinearSolverChoice,
        Lsode2LinearSolverPolicy, Lsode2LinearSystemStructure, Lsode2MethodFamily,
        Lsode2NativeExecutionConfig, Lsode2NativeStatistics, Lsode2ProblemConfig,
        Lsode2ResidualJacobianSource, Lsode2SolveSummary, Lsode2Solver as Solver,
        Lsode2StopComparator, Lsode2StopCondition, Lsode2SymbolicAssemblyBackend,
        Lsode2SymbolicExecutionMode,
    };
}

/// Solver family selected by an application-level stiff-solver configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StiffSolverKind {
    Bdf,
    Radau,
    BackwardEuler,
    Lsode2,
}

/// Route-aware request that keeps each solver's native model/configuration
/// contract intact while centralizing application-level solver dispatch.
#[derive(Clone)]
pub enum StiffSolverRequest {
    Bdf {
        options: BdfSolverOptions,
    },
    Radau {
        problem: crate::numerical::Radau::api::RadauProblem,
        config: RadauConfig,
        /// Initial state supplied to Radau's first solve. Other native
        /// adapters keep their initial state in their native configuration.
        initial_state: Vec<f64>,
        /// Ordered value-only parameters declared by `problem`.
        parameters: Vec<f64>,
    },
    BackwardEuler {
        options: BeSolverOptions,
        /// Ordered symbolic parameters declared after native construction.
        equation_parameters: Vec<String>,
        /// Initial values corresponding to `equation_parameters`.
        parameter_values: Vec<f64>,
    },
    /// Preserve LSODE2's independent controller, native/bridge execution,
    /// stop-condition and callback-source configuration.
    Lsode2 {
        config: Lsode2ProblemConfig,
        /// Optional per-request Rust logging threshold, independent of telemetry.
        logging_level: Option<SolverLogLevel>,
    },
}

/// Prepared native solver selected by [`StiffSolverRequest`].
///
/// `try_solve` centralizes first-run dispatch. Native lifecycle,
/// solver-specific results, telemetry and postprocessing remain accessible
/// through typed accessors; the facade does not flatten them into a lossy
/// common representation.
pub enum PreparedStiffSolver {
    Bdf(BdfSolver),
    Radau(PreparedRadau),
    BackwardEuler(BE),
    Lsode2(Lsode2Solver),
}

/// Radau's prepared model and the initial values needed by its first solve.
/// The native solver remains available through [`Self::solver`]; this small
/// adapter only supplies the input Radau requires at execution time.
pub struct PreparedRadau {
    solver: RadauSolver,
    initial_state: Vec<f64>,
    parameters: Vec<f64>,
}

impl PreparedRadau {
    pub fn solver(&self) -> &RadauSolver {
        &self.solver
    }

    pub fn solver_mut(&mut self) -> &mut RadauSolver {
        &mut self.solver
    }
}

/// Solver-native outcome of a common StiffAPI solve dispatch.
///
/// The variants preserve result ownership: Radau returns its dense-output
/// solution, LSODE2 returns its native summary, and BDF/BE retain results on
/// their prepared solver objects.
#[derive(Debug)]
pub enum StiffSolveOutcome {
    Bdf,
    Radau(RadauSolution),
    BackwardEuler,
    Lsode2(crate::numerical::LSODE2::Lsode2SolveSummary),
}

/// Native typed failure from a dispatched solve. The prepared solver remains
/// borrowed by the caller, so native partial results and restart operations
/// remain available after an error.
#[derive(Debug)]
pub enum StiffSolveError {
    Restart(StiffRestartError),
    Bdf(BdfSolveError),
    Radau(RadauError),
    BackwardEuler(BeError),
    Lsode2(Lsode2Error),
}

/// Failure from the opt-in cancellation dispatch on the common stiff facade.
#[derive(Debug)]
pub enum StiffCancellationError {
    Bdf(BdfSolveError),
    Radau(RadauCancellationError),
    BackwardEuler(BeError),
    Lsode2(Lsode2Error),
    Unsupported { solver: &'static str },
}

static NEXT_STIFF_SOLVE_ID: AtomicU64 = AtomicU64::new(1);

/// One owned partial trajectory from a failed or cancelled attempt.
///
/// This type is intentionally distinct from successful solver outputs so it
/// cannot be mistaken for an accepted numerical solution.
#[derive(Debug, Clone)]
pub struct StiffDiagnosticTrajectory {
    pub solve_id: u64,
    pub solver: StiffSolverKind,
    pub axis: DVector<f64>,
    /// Row-oriented values: `(time_points, state_variables)`.
    pub values: DMatrix<f64>,
    pub accepted_steps: Option<usize>,
}

/// Result of exactly one explicit diagnostic solve call.
///
/// `solve_id` identifies both the native telemetry snapshot and any partial
/// trajectory. Collection is opt-in through `try_solve_with_report`; ordinary
/// `try_solve` does not copy trajectories or allocate this report.
#[derive(Debug)]
pub struct StiffSolveAttemptReport {
    pub solve_id: u64,
    pub result: Result<StiffSolveOutcome, StiffAttemptError>,
    pub completed_at: std::time::Instant,
    pub cancellation_at_return: StiffCancellationDisposition,
    pub telemetry: Option<StiffAttemptTelemetry>,
    /// Present only on failure/cancellation, and always diagnostic-only.
    pub partial_trajectory: Option<StiffDiagnosticTrajectory>,
}

/// Structured callback-failure facts that can be recovered from the native
/// typed error chain without inspecting its display text.
///
/// Fields absent from a native error remain `None`; in particular, the last
/// accepted time/state is not substituted for a failed trial's callback
/// inputs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StiffCallbackFailureDiagnostic {
    pub evaluation: StiffCallbackEvaluation,
    pub reason: StiffCallbackFailureReason,
    pub element_index: Option<usize>,
    pub callback_time: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StiffCallbackEvaluation {
    Residual,
    Jacobian,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StiffCallbackFailureReason {
    NonFiniteOutput,
    ShapeMismatch,
    CallbackFailed,
}

impl StiffSolveAttemptReport {
    /// Reclassify a shared token against the attempt's completion timestamp.
    /// This also detects a cancellation request that races just after solve
    /// completion but before the caller inspects the report.
    pub fn cancellation_disposition_for(
        &self,
        token: &CancellationToken,
    ) -> StiffCancellationDisposition {
        if matches!(&self.result, Err(StiffAttemptError::Cancelled(_))) {
            return StiffCancellationDisposition::ObservedBySolver;
        }
        match token.requested_at() {
            None => StiffCancellationDisposition::NotRequested,
            Some(requested_at) if requested_at > self.completed_at => {
                StiffCancellationDisposition::CompletedBeforeRequest
            }
            Some(_) => StiffCancellationDisposition::RequestedButNotObserved,
        }
    }

    /// Return only callback provenance present in the typed native error.
    ///
    /// This accessor does not parse error messages and does not collect or
    /// copy trial states. Native routes that currently flatten callback errors
    /// return `None` until they preserve structured provenance themselves.
    pub fn callback_failure_diagnostic(&self) -> Option<StiffCallbackFailureDiagnostic> {
        let StiffAttemptError::Solve(error) = self.result.as_ref().err()? else {
            return None;
        };
        callback_failure_diagnostic(error)
    }
}

fn callback_failure_diagnostic(error: &StiffSolveError) -> Option<StiffCallbackFailureDiagnostic> {
    use crate::numerical::BDF::BDF_solver::BdfStepError;
    use crate::numerical::BE::NR_for_Euler::NreError;
    use crate::symbolic::symbolic_ivp::IvpBackendError;

    let from_backend = |backend: &IvpBackendError| match backend {
        IvpBackendError::AtomEvaluationFailure { stage, index, .. } => {
            let evaluation = match stage.as_str() {
                "residual" | "Residual" => StiffCallbackEvaluation::Residual,
                "jacobian" | "Jacobian" => StiffCallbackEvaluation::Jacobian,
                _ => StiffCallbackEvaluation::Unknown,
            };
            Some(StiffCallbackFailureDiagnostic {
                evaluation,
                reason: StiffCallbackFailureReason::CallbackFailed,
                element_index: Some(*index),
                callback_time: None,
            })
        }
        IvpBackendError::InvalidOutputShape { stage, .. }
        | IvpBackendError::InvalidMatrixShape { stage, .. } => {
            let evaluation = match stage.as_str() {
                "residual" | "Residual" => StiffCallbackEvaluation::Residual,
                "jacobian" | "Jacobian" => StiffCallbackEvaluation::Jacobian,
                _ => StiffCallbackEvaluation::Unknown,
            };
            Some(StiffCallbackFailureDiagnostic {
                evaluation,
                reason: StiffCallbackFailureReason::ShapeMismatch,
                element_index: None,
                callback_time: None,
            })
        }
        _ => None,
    };

    let from_bdf = |error: &BdfSolveError| match error {
        BdfSolveError::Backend(backend) => from_backend(backend),
        BdfSolveError::Step(step) => match step {
            BdfStepError::NonFiniteRhs => Some(StiffCallbackFailureDiagnostic {
                evaluation: StiffCallbackEvaluation::Residual,
                reason: StiffCallbackFailureReason::NonFiniteOutput,
                element_index: None,
                callback_time: None,
            }),
            BdfStepError::NonFiniteJacobian => Some(StiffCallbackFailureDiagnostic {
                evaluation: StiffCallbackEvaluation::Jacobian,
                reason: StiffCallbackFailureReason::NonFiniteOutput,
                element_index: None,
                callback_time: None,
            }),
            BdfStepError::RhsDimension => Some(StiffCallbackFailureDiagnostic {
                evaluation: StiffCallbackEvaluation::Residual,
                reason: StiffCallbackFailureReason::ShapeMismatch,
                element_index: None,
                callback_time: None,
            }),
            BdfStepError::InvalidJacobianDimension | BdfStepError::InvalidJacobianStructure => {
                Some(StiffCallbackFailureDiagnostic {
                    evaluation: StiffCallbackEvaluation::Jacobian,
                    reason: StiffCallbackFailureReason::ShapeMismatch,
                    element_index: None,
                    callback_time: None,
                })
            }
            BdfStepError::JacobianCallback => Some(StiffCallbackFailureDiagnostic {
                evaluation: StiffCallbackEvaluation::Jacobian,
                reason: StiffCallbackFailureReason::CallbackFailed,
                element_index: None,
                callback_time: None,
            }),
            _ => None,
        },
        _ => None,
    };

    let from_be = |error: &BeError| match error {
        BeError::Backend(backend) => from_backend(backend),
        BeError::GeneratedBackend(
            crate::symbolic::symbolic_ivp_generated::SymbolicIvpGeneratedError::IvpBackend(backend),
        ) => from_backend(backend),
        BeError::Newton(NreError::NonFiniteCallback { stage }) => {
            let evaluation = match *stage {
                "Jacobian" => StiffCallbackEvaluation::Jacobian,
                "residual" | "Residual" => StiffCallbackEvaluation::Residual,
                _ => StiffCallbackEvaluation::Unknown,
            };
            Some(StiffCallbackFailureDiagnostic {
                evaluation,
                reason: StiffCallbackFailureReason::NonFiniteOutput,
                element_index: None,
                callback_time: None,
            })
        }
        BeError::Newton(NreError::InvalidResidualShape { .. }) => {
            Some(StiffCallbackFailureDiagnostic {
                evaluation: StiffCallbackEvaluation::Residual,
                reason: StiffCallbackFailureReason::ShapeMismatch,
                element_index: None,
                callback_time: None,
            })
        }
        BeError::Newton(NreError::InvalidJacobianShape { .. }) => {
            Some(StiffCallbackFailureDiagnostic {
                evaluation: StiffCallbackEvaluation::Jacobian,
                reason: StiffCallbackFailureReason::ShapeMismatch,
                element_index: None,
                callback_time: None,
            })
        }
        BeError::Newton(_)
        | BeError::GeneratedBackend(_)
        | BeError::InvalidConfiguration(_)
        | BeError::StepUnderflow { .. }
        | BeError::StepLimit { .. }
        | BeError::Cancelled { .. } => None,
    };

    match error {
        StiffSolveError::Bdf(error) => from_bdf(error),
        StiffSolveError::BackwardEuler(error) => from_be(error),
        StiffSolveError::Lsode2(crate::numerical::LSODE2::Lsode2Error::GeneratedBackend(
            backend,
        )) => from_backend(backend),
        StiffSolveError::Lsode2(crate::numerical::LSODE2::Lsode2Error::Bridge(error)) => {
            from_bdf(error)
        }
        StiffSolveError::Radau(error) => {
            let evaluation = match error.stage() {
                Some(crate::numerical::Radau::api::RadauErrorStage::Residual) => {
                    StiffCallbackEvaluation::Residual
                }
                Some(crate::numerical::Radau::api::RadauErrorStage::Jacobian) => {
                    StiffCallbackEvaluation::Jacobian
                }
                _ => StiffCallbackEvaluation::Unknown,
            };
            let (evaluation, reason) = match error.kind() {
                crate::numerical::Radau::api::RadauErrorKind::NonFiniteCallback => {
                    (evaluation, StiffCallbackFailureReason::NonFiniteOutput)
                }
                crate::numerical::Radau::api::RadauErrorKind::Callback => {
                    (evaluation, StiffCallbackFailureReason::CallbackFailed)
                }
                crate::numerical::Radau::api::RadauErrorKind::Shape => {
                    (evaluation, StiffCallbackFailureReason::ShapeMismatch)
                }
                _ => return None,
            };
            Some(StiffCallbackFailureDiagnostic {
                evaluation,
                reason,
                element_index: None,
                callback_time: None,
            })
        }
        StiffSolveError::Restart(_) | StiffSolveError::Lsode2(_) => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StiffCancellationDisposition {
    NotRequested,
    ObservedBySolver,
    CompletedBeforeRequest,
    RequestedButNotObserved,
}

#[derive(Debug, Clone)]
pub struct StiffAttemptTelemetry {
    pub solve_id: u64,
    pub scope: StiffTelemetryScope,
    pub snapshot: StiffTelemetrySnapshot,
}

/// Optional per-call LSODE2 telemetry differences, computed from native
/// snapshots immediately before and after an explicit report call.
///
/// This is additive to the native lifecycle-cumulative snapshot in
/// [`StiffSolveAttemptReport`]. `None` symbolic metrics mean their telemetry
/// mode did not request that measurement.
#[derive(Debug, Clone, PartialEq)]
pub struct StiffLsode2AttemptTelemetryDelta {
    pub scope: StiffTelemetryScope,
    /// Full native snapshots bracketing this explicit report call. These let
    /// consumers compute deltas for controller/event counters that are not
    /// duplicated as convenience fields below.
    pub native_before: Lsode2NativeStatistics,
    pub native_after: Lsode2NativeStatistics,
    pub symbolic_residual_evaluations: Option<u64>,
    pub symbolic_jacobian_evaluations: Option<u64>,
    pub symbolic_residual_time: Option<std::time::Duration>,
    pub symbolic_jacobian_time: Option<std::time::Duration>,
    pub native_solve_calls: usize,
    pub native_solve_time_ms: f64,
    pub native_step_attempts: usize,
    pub native_step_accepts: usize,
    pub native_step_rejects: usize,
    pub native_residual_calls: usize,
    pub native_jacobian_calls: usize,
    pub native_linear_solve_calls: usize,
    /// True if a native cumulative counter decreased between snapshots. In
    /// that case its delta is the post-call value, indicating a reset boundary.
    pub native_statistics_reset: bool,
    pub symbolic_counters_reset: bool,
}

/// A standard attempt report plus an optional LSODE2 per-call telemetry delta.
/// Existing report APIs and their snapshot semantics remain unchanged.
#[derive(Debug)]
pub struct StiffSolveAttemptReportWithTelemetryDelta {
    pub report: StiffSolveAttemptReport,
    pub lsode2_delta: Option<StiffLsode2AttemptTelemetryDelta>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StiffTelemetryScope {
    Attempt,
    AttemptAndPreparation,
    SolverLifecycleCumulative,
}

/// Typed failure for an attempt report, preserving native error chains.
#[derive(Debug)]
pub enum StiffAttemptError {
    Solve(StiffSolveError),
    Cancelled(StiffCancellationError),
}

impl fmt::Display for StiffAttemptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Solve(error) => error.fmt(f),
            Self::Cancelled(error) => error.fmt(f),
        }
    }
}

impl Error for StiffAttemptError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Solve(error) => Some(error),
            Self::Cancelled(error) => Some(error),
        }
    }
}

impl fmt::Display for StiffCancellationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bdf(error) => write!(f, "BDF cancellation-aware solve failed: {error}"),
            Self::Radau(error) => write!(f, "Radau cancellation-aware solve failed: {error}"),
            Self::BackwardEuler(error) => {
                write!(f, "Backward Euler cancellation-aware solve failed: {error}")
            }
            Self::Lsode2(error) => write!(f, "LSODE2 cancellation-aware solve failed: {error}"),
            Self::Unsupported { solver } => {
                write!(
                    f,
                    "cooperative cancellation is not yet supported by {solver}"
                )
            }
        }
    }
}

impl Error for StiffCancellationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Bdf(error) => Some(error),
            Self::Radau(error) => Some(error),
            Self::BackwardEuler(error) => Some(error),
            Self::Lsode2(error) => Some(error),
            Self::Unsupported { .. } => None,
        }
    }
}

/// One independent parameter-vector solve through the common stiff facade.
#[derive(Debug)]
pub struct StiffParameterLoopPoint {
    pub index: usize,
    pub solve_id: u64,
    pub parameter_values: Vec<f64>,
    pub result: Result<IvpParameterLoopSolution, StiffParameterLoopError>,
    pub telemetry: Option<StiffAttemptTelemetry>,
    pub partial_trajectory: Option<StiffDiagnosticTrajectory>,
}

struct StiffLoopAttemptMetadata {
    solve_id: u64,
    telemetry: Option<StiffAttemptTelemetry>,
    partial_trajectory: Option<StiffDiagnosticTrajectory>,
}

/// Failure while solving or materializing one StiffAPI loop point.
#[derive(Debug)]
pub enum StiffParameterLoopError {
    Solve(StiffSolveError),
    Cancelled(StiffCancellationError),
    Output(StiffPostprocessError),
}

impl fmt::Display for StiffParameterLoopError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Solve(error) => write!(f, "parameter-point solve failed: {error}"),
            Self::Cancelled(error) => write!(f, "parameter-point solve cancelled: {error}"),
            Self::Output(error) => write!(f, "parameter-point output failed: {error}"),
        }
    }
}

impl Error for StiffParameterLoopError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Solve(error) => Some(error),
            Self::Cancelled(error) => Some(error),
            Self::Output(error) => Some(error),
        }
    }
}

/// Ordered results of an independent stiff-IVP parameter sweep.
#[derive(Debug, Default)]
pub struct StiffParameterLoopReport {
    pub points: Vec<StiffParameterLoopPoint>,
}

fn diagnostic_from_rows(
    solve_id: u64,
    solver: StiffSolverKind,
    times: Vec<f64>,
    states: Vec<Vec<f64>>,
    accepted_steps: Option<usize>,
) -> Option<StiffDiagnosticTrajectory> {
    let state_dimension = states.first()?.len();
    if state_dimension == 0
        || times.len() != states.len()
        || states.iter().any(|state| state.len() != state_dimension)
    {
        return None;
    }
    let values = states.into_iter().flatten().collect::<Vec<_>>();
    Some(StiffDiagnosticTrajectory {
        solve_id,
        solver,
        axis: DVector::from_vec(times),
        values: DMatrix::from_row_slice(values.len() / state_dimension, state_dimension, &values),
        accepted_steps,
    })
}

impl StiffSolveError {
    pub const fn solver_kind(&self) -> StiffSolverKind {
        match self {
            Self::Restart(error) => error.solver_kind(),
            Self::Bdf(_) => StiffSolverKind::Bdf,
            Self::Radau(_) => StiffSolverKind::Radau,
            Self::BackwardEuler(_) => StiffSolverKind::BackwardEuler,
            Self::Lsode2(_) => StiffSolverKind::Lsode2,
        }
    }
}

impl fmt::Display for StiffSolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Restart(error) => write!(f, "IVP restart failed: {error}"),
            Self::Bdf(error) => write!(f, "BDF solve failed: {error}"),
            Self::Radau(error) => write!(f, "Radau solve failed: {error}"),
            Self::BackwardEuler(error) => write!(f, "Backward Euler solve failed: {error}"),
            Self::Lsode2(error) => write!(f, "LSODE2 solve failed: {error}"),
        }
    }
}

impl Error for StiffSolveError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Restart(error) => Some(error),
            Self::Bdf(error) => Some(error),
            Self::Radau(error) => Some(error),
            Self::BackwardEuler(error) => Some(error),
            Self::Lsode2(error) => Some(error),
        }
    }
}

/// Typed failure while resetting a prepared solver to a new initial-value
/// segment. Native solver failures retain their family-specific error type.
#[derive(Debug)]
pub enum StiffRestartError {
    InvalidInterval {
        solver: StiffSolverKind,
    },
    InvalidInitialState {
        solver: StiffSolverKind,
    },
    InitialStateDimension {
        solver: StiffSolverKind,
        expected: usize,
        actual: usize,
    },
    Bdf(BdfSolveError),
    Radau(RadauError),
    BackwardEuler(BeError),
    Lsode2(Lsode2Error),
}

impl StiffRestartError {
    pub const fn solver_kind(&self) -> StiffSolverKind {
        match self {
            Self::InvalidInterval { solver }
            | Self::InvalidInitialState { solver }
            | Self::InitialStateDimension { solver, .. } => *solver,
            Self::Bdf(_) => StiffSolverKind::Bdf,
            Self::Radau(_) => StiffSolverKind::Radau,
            Self::BackwardEuler(_) => StiffSolverKind::BackwardEuler,
            Self::Lsode2(_) => StiffSolverKind::Lsode2,
        }
    }
}

impl fmt::Display for StiffRestartError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInterval { solver } => {
                write!(f, "{solver:?} restart interval must be finite and nonzero")
            }
            Self::InvalidInitialState { solver } => {
                write!(f, "{solver:?} restart state must be nonempty and finite")
            }
            Self::InitialStateDimension {
                solver,
                expected,
                actual,
            } => write!(
                f,
                "{solver:?} restart state has dimension {actual}; expected {expected}"
            ),
            Self::Bdf(error) => write!(f, "BDF restart failed: {error}"),
            Self::Radau(error) => write!(f, "Radau restart failed: {error}"),
            Self::BackwardEuler(error) => write!(f, "Backward Euler restart failed: {error}"),
            Self::Lsode2(error) => write!(f, "LSODE2 restart failed: {error}"),
        }
    }
}

impl Error for StiffRestartError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Bdf(error) => Some(error),
            Self::Radau(error) => Some(error),
            Self::BackwardEuler(error) => Some(error),
            Self::Lsode2(error) => Some(error),
            Self::InvalidInterval { .. }
            | Self::InvalidInitialState { .. }
            | Self::InitialStateDimension { .. } => None,
        }
    }
}

impl PreparedStiffSolver {
    /// Report which native solver was selected by the request.
    pub const fn kind(&self) -> StiffSolverKind {
        match self {
            Self::Bdf(_) => StiffSolverKind::Bdf,
            Self::Radau(_) => StiffSolverKind::Radau,
            Self::BackwardEuler(_) => StiffSolverKind::BackwardEuler,
            Self::Lsode2(_) => StiffSolverKind::Lsode2,
        }
    }

    /// Return the effective native telemetry mode without materializing or
    /// merging solver-specific telemetry reports.
    pub fn telemetry_mode(&self) -> SolverTelemetryMode {
        match self {
            Self::Bdf(solver) => match solver.telemetry_mode() {
                BdfTelemetryMode::Off => SolverTelemetryMode::Off,
                BdfTelemetryMode::Counters => SolverTelemetryMode::Counters,
                BdfTelemetryMode::Timings => SolverTelemetryMode::Timings,
                BdfTelemetryMode::Detailed => SolverTelemetryMode::Detailed,
            },
            Self::Radau(prepared) => match prepared.solver.telemetry_mode() {
                crate::numerical::Radau::api::RadauTelemetryMode::Off => SolverTelemetryMode::Off,
                crate::numerical::Radau::api::RadauTelemetryMode::Counters => {
                    SolverTelemetryMode::Counters
                }
                crate::numerical::Radau::api::RadauTelemetryMode::Timings => {
                    SolverTelemetryMode::Timings
                }
                crate::numerical::Radau::api::RadauTelemetryMode::Detailed => {
                    SolverTelemetryMode::Detailed
                }
            },
            Self::BackwardEuler(solver) => match solver.telemetry_mode() {
                BeTelemetryMode::Off => SolverTelemetryMode::Off,
                BeTelemetryMode::Counters => SolverTelemetryMode::Counters,
                BeTelemetryMode::Timings => SolverTelemetryMode::Timings,
                BeTelemetryMode::Detailed => SolverTelemetryMode::Detailed,
            },
            Self::Lsode2(solver) => match solver.config().telemetry.mode() {
                crate::numerical::LSODE2::IvpTelemetryMode::Off => SolverTelemetryMode::Off,
                crate::numerical::LSODE2::IvpTelemetryMode::Counters => {
                    SolverTelemetryMode::Counters
                }
                crate::numerical::LSODE2::IvpTelemetryMode::Timings => SolverTelemetryMode::Timings,
                crate::numerical::LSODE2::IvpTelemetryMode::Detailed => {
                    SolverTelemetryMode::Detailed
                }
            },
        }
    }

    /// Return the selected native telemetry without flattening solver-specific
    /// counters or timing scopes into a synthetic common schema. Radau keeps
    /// its report on the completed `RadauSolution`, so use
    /// [`StiffTelemetrySnapshot::from_radau_solution`] for that solver.
    pub fn telemetry_snapshot(&self) -> Result<StiffTelemetrySnapshot, StiffTelemetryError> {
        match self {
            Self::Bdf(solver) => Ok(StiffTelemetrySnapshot::Bdf {
                mode: self.telemetry_mode(),
                statistics: solver.get_statistics(),
                symbolic: solver.symbolic_ivp_telemetry_snapshot(),
            }),
            Self::Radau(_) => Err(StiffTelemetryError::RadauSolutionRequired),
            Self::BackwardEuler(solver) => Ok(StiffTelemetrySnapshot::BackwardEuler {
                mode: self.telemetry_mode(),
                statistics: solver.get_statistics(),
                details: solver.detailed_statistics(),
                symbolic: solver.symbolic_ivp_telemetry_snapshot(),
            }),
            Self::Lsode2(solver) => Ok(StiffTelemetrySnapshot::Lsode2 {
                mode: self.telemetry_mode(),
                statistics: solver.statistics(),
                native: solver.native_statistics(),
                symbolic: solver.telemetry_snapshot(),
            }),
        }
    }

    /// Borrow the retained trajectory for BDF, Backward Euler or LSODE2.
    /// Radau returns its sampled output from `solve(y0)` and must instead be
    /// converted from that `RadauSolution` with [`radau_postprocess_dataset`].
    pub fn trajectory_view(&self) -> Result<StiffTrajectoryView<'_>, StiffPostprocessError> {
        let (solver, axis, values) = match self {
            Self::Bdf(solver) => {
                if solver.status_kind() == crate::numerical::BDF::BDF_api::BdfStatus::Running {
                    return Err(StiffPostprocessError::ResultUnavailable {
                        solver: StiffSolverKind::Bdf,
                    });
                }
                let (axis, values) = solver.get_result_ref();
                (StiffSolverKind::Bdf, axis, values)
            }
            Self::Radau(_) => return Err(StiffPostprocessError::RadauSolutionRequired),
            Self::BackwardEuler(solver) => {
                let (axis, values) = solver.trajectory();
                (StiffSolverKind::BackwardEuler, axis, values)
            }
            Self::Lsode2(solver) => {
                if solver.status() == "running" {
                    return Err(StiffPostprocessError::ResultUnavailable {
                        solver: StiffSolverKind::Lsode2,
                    });
                }
                let (axis, values) = solver.get_result_ref();
                (StiffSolverKind::Lsode2, axis, values)
            }
        };
        if axis.is_empty() || values.nrows() != axis.len() {
            return Err(StiffPostprocessError::ResultUnavailable { solver });
        }
        Ok(StiffTrajectoryView {
            solver,
            axis,
            values,
        })
    }

    /// Borrow the native BDF solver when selected.
    pub fn as_bdf(&self) -> Option<&BdfSolver> {
        match self {
            Self::Bdf(solver) => Some(solver),
            Self::Radau(_) | Self::BackwardEuler(_) | Self::Lsode2(_) => None,
        }
    }

    /// Mutably borrow the native BDF solver when selected.
    pub fn as_bdf_mut(&mut self) -> Option<&mut BdfSolver> {
        match self {
            Self::Bdf(solver) => Some(solver),
            Self::Radau(_) | Self::BackwardEuler(_) | Self::Lsode2(_) => None,
        }
    }

    /// Borrow the native Radau solver when this request selected Radau.
    pub fn as_radau(&self) -> Option<&RadauSolver> {
        match self {
            Self::Bdf(_) => None,
            Self::Radau(prepared) => Some(&prepared.solver),
            Self::BackwardEuler(_) | Self::Lsode2(_) => None,
        }
    }

    /// Mutably borrow the native Radau solver when this request selected Radau.
    pub fn as_radau_mut(&mut self) -> Option<&mut RadauSolver> {
        match self {
            Self::Bdf(_) => None,
            Self::Radau(prepared) => Some(&mut prepared.solver),
            Self::BackwardEuler(_) | Self::Lsode2(_) => None,
        }
    }

    /// Borrow the native Backward Euler solver when selected.
    pub fn as_backward_euler(&self) -> Option<&BE> {
        match self {
            Self::Bdf(_) => None,
            Self::Radau(_) => None,
            Self::BackwardEuler(solver) => Some(solver),
            Self::Lsode2(_) => None,
        }
    }

    /// Mutably borrow the native Backward Euler solver when selected.
    pub fn as_backward_euler_mut(&mut self) -> Option<&mut BE> {
        match self {
            Self::Bdf(_) => None,
            Self::Radau(_) => None,
            Self::BackwardEuler(solver) => Some(solver),
            Self::Lsode2(_) => None,
        }
    }

    /// Borrow the native LSODE2 solver when selected.
    pub fn as_lsode2(&self) -> Option<&Lsode2Solver> {
        match self {
            Self::Lsode2(solver) => Some(solver),
            Self::Bdf(_) | Self::Radau(_) | Self::BackwardEuler(_) => None,
        }
    }

    /// Mutably borrow the native LSODE2 solver when selected.
    pub fn as_lsode2_mut(&mut self) -> Option<&mut Lsode2Solver> {
        match self {
            Self::Lsode2(solver) => Some(solver),
            Self::Bdf(_) | Self::Radau(_) | Self::BackwardEuler(_) => None,
        }
    }

    /// Run the solver selected during preparation without application-side
    /// solver dispatch. Native results and failures remain solver-specific.
    /// On failure, this object remains available for status inspection and
    /// native restart/recovery through the typed accessors.
    pub fn try_solve(&mut self) -> Result<StiffSolveOutcome, StiffSolveError> {
        match self {
            Self::Bdf(solver) => {
                solver.try_solve().map_err(StiffSolveError::Bdf)?;
                Ok(StiffSolveOutcome::Bdf)
            }
            Self::Radau(prepared) => {
                let solution = prepared
                    .solver
                    .try_solve_with_parameter_values(&prepared.initial_state, &prepared.parameters)
                    .map_err(StiffSolveError::Radau)?;
                Ok(StiffSolveOutcome::Radau(solution))
            }
            Self::BackwardEuler(solver) => {
                solver.try_solve().map_err(StiffSolveError::BackwardEuler)?;
                Ok(StiffSolveOutcome::BackwardEuler)
            }
            Self::Lsode2(solver) => solver
                .solve_with_summary()
                .map(StiffSolveOutcome::Lsode2)
                .map_err(StiffSolveError::Lsode2),
        }
    }

    /// Solve with opt-in cooperative cancellation where the selected native
    /// route implements safe cancellation checkpoints.
    ///
    /// BDF, Radau, Backward Euler, and LSODE2 observe cancellation at
    /// solver-owned safe checkpoints.
    pub fn try_solve_with_cancellation_token(
        &mut self,
        cancellation: CancellationToken,
    ) -> Result<StiffSolveOutcome, StiffCancellationError> {
        match self {
            Self::Bdf(solver) => solver
                .try_solve_with_cancellation_token(cancellation)
                .map(|()| StiffSolveOutcome::Bdf)
                .map_err(StiffCancellationError::Bdf),
            Self::BackwardEuler(solver) => solver
                .try_solve_with_cancellation_token(cancellation)
                .map(|()| StiffSolveOutcome::BackwardEuler)
                .map_err(StiffCancellationError::BackwardEuler),
            Self::Radau(prepared) => prepared
                .solver
                .try_solve_with_cancellation_token(
                    &prepared.initial_state,
                    &prepared.parameters,
                    &cancellation,
                )
                .map(StiffSolveOutcome::Radau)
                .map_err(StiffCancellationError::Radau),
            Self::Lsode2(solver) => solver
                .solve_with_cancellation_token(&cancellation)
                .map(|()| StiffSolveOutcome::Lsode2(solver.summary()))
                .map_err(StiffCancellationError::Lsode2),
        }
    }

    /// Execute one solve and return an attempt-scoped diagnostic report.
    ///
    /// Unlike `try_solve`, this explicit reporting operation owns a copy of
    /// partial accepted output when the attempt fails or is cancelled. The
    /// regular solve path remains allocation-light and unchanged.
    pub fn try_solve_with_report(
        &mut self,
        cancellation: Option<CancellationToken>,
    ) -> StiffSolveAttemptReport {
        let solve_id = NEXT_STIFF_SOLVE_ID.fetch_add(1, Ordering::Relaxed);
        let token_for_report = cancellation.clone();
        let result = match cancellation {
            Some(token) => self
                .try_solve_with_cancellation_token(token)
                .map_err(StiffAttemptError::Cancelled),
            None => self.try_solve().map_err(StiffAttemptError::Solve),
        };
        let completed_at = std::time::Instant::now();
        let observed = matches!(&result, Err(StiffAttemptError::Cancelled(_)));
        let disposition = match token_for_report.and_then(|token| token.requested_at()) {
            None if !observed => StiffCancellationDisposition::NotRequested,
            _ if observed => StiffCancellationDisposition::ObservedBySolver,
            Some(requested_at) if requested_at > completed_at => {
                StiffCancellationDisposition::CompletedBeforeRequest
            }
            Some(_) => StiffCancellationDisposition::RequestedButNotObserved,
            None => StiffCancellationDisposition::NotRequested,
        };
        self.finish_attempt_report(solve_id, result, completed_at, disposition)
    }

    /// Run one reportable solve and additionally compute LSODE2 telemetry
    /// deltas for this call. The existing report continues to carry the
    /// native cumulative snapshot and its scope. This opt-in wrapper performs
    /// two snapshot reads, never per-step instrumentation.
    pub fn try_solve_with_report_and_telemetry_delta(
        &mut self,
        cancellation: Option<CancellationToken>,
    ) -> StiffSolveAttemptReportWithTelemetryDelta {
        let (before, was_prepared) = match self {
            Self::Lsode2(solver) => (
                Some((solver.native_statistics(), solver.telemetry_snapshot())),
                solver.is_prepared(),
            ),
            _ => (None, false),
        };
        let report = self.try_solve_with_report(cancellation);
        let lsode2_delta = before.and_then(|(native_before, symbolic_before)| {
            let (native_after, symbolic_after) = match self {
                Self::Lsode2(solver) => (solver.native_statistics(), solver.telemetry_snapshot()),
                _ => return None,
            };
            Some(lsode2_attempt_telemetry_delta(
                native_before,
                native_after,
                symbolic_before,
                symbolic_after,
                if was_prepared {
                    StiffTelemetryScope::Attempt
                } else {
                    StiffTelemetryScope::AttemptAndPreparation
                },
            ))
        });
        StiffSolveAttemptReportWithTelemetryDelta {
            report,
            lsode2_delta,
        }
    }

    /// Parameter-rebinding variant of [`Self::try_solve_with_report`].
    pub fn try_solve_with_parameter_values_report(
        &mut self,
        y0: DVector<f64>,
        values: DVector<f64>,
    ) -> StiffSolveAttemptReport {
        let solve_id = NEXT_STIFF_SOLVE_ID.fetch_add(1, Ordering::Relaxed);
        let result = self
            .try_solve_with_parameter_values(y0, values)
            .map_err(StiffAttemptError::Solve);
        self.finish_attempt_report(
            solve_id,
            result,
            std::time::Instant::now(),
            StiffCancellationDisposition::NotRequested,
        )
    }

    /// Explicit-interval parameter-rebinding variant of
    /// [`Self::try_solve_with_report`].
    pub fn try_solve_with_parameter_values_in_interval_report(
        &mut self,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        values: DVector<f64>,
    ) -> StiffSolveAttemptReport {
        let solve_id = NEXT_STIFF_SOLVE_ID.fetch_add(1, Ordering::Relaxed);
        let result = self
            .try_solve_with_parameter_values_in_interval(t0, y0, t_bound, values)
            .map_err(StiffAttemptError::Solve);
        self.finish_attempt_report(
            solve_id,
            result,
            std::time::Instant::now(),
            StiffCancellationDisposition::NotRequested,
        )
    }

    /// Fresh-initial-state variant of [`Self::try_solve_with_report`].
    pub fn try_solve_with_initial_state_report(
        &mut self,
        y0: DVector<f64>,
    ) -> StiffSolveAttemptReport {
        let solve_id = NEXT_STIFF_SOLVE_ID.fetch_add(1, Ordering::Relaxed);
        let result = self
            .try_solve_with_initial_state(y0)
            .map_err(StiffAttemptError::Solve);
        self.finish_attempt_report(
            solve_id,
            result,
            std::time::Instant::now(),
            StiffCancellationDisposition::NotRequested,
        )
    }

    fn finish_attempt_report(
        &self,
        solve_id: u64,
        result: Result<StiffSolveOutcome, StiffAttemptError>,
        completed_at: std::time::Instant,
        cancellation_at_return: StiffCancellationDisposition,
    ) -> StiffSolveAttemptReport {
        let failed = result.is_err();
        let partial_trajectory = failed
            .then(|| self.diagnostic_trajectory(solve_id, result.as_ref().err()))
            .flatten();
        let telemetry_snapshot = match &result {
            Ok(StiffSolveOutcome::Radau(solution)) => {
                Some(StiffTelemetrySnapshot::from_radau_solution(solution))
            }
            Ok(_) => self.telemetry_snapshot().ok(),
            Err(StiffAttemptError::Cancelled(StiffCancellationError::Radau(
                crate::numerical::Radau::api::RadauCancellationError::Cancelled { snapshot },
            ))) => Some(StiffTelemetrySnapshot::Radau(snapshot.telemetry.clone())),
            Err(StiffAttemptError::Cancelled(_)) => self.telemetry_snapshot().ok(),
            Err(StiffAttemptError::Solve(StiffSolveError::Restart(_))) => None,
            Err(StiffAttemptError::Solve(StiffSolveError::Radau(_))) => match self {
                Self::Radau(prepared) => prepared
                    .solver
                    .last_solve_telemetry()
                    .cloned()
                    .map(StiffTelemetrySnapshot::Radau),
                _ => None,
            },
            Err(StiffAttemptError::Solve(_)) if self.is_in_native_failed_state() => {
                self.telemetry_snapshot().ok()
            }
            Err(StiffAttemptError::Solve(_)) => None,
        };
        StiffSolveAttemptReport {
            solve_id,
            result,
            completed_at,
            cancellation_at_return,
            telemetry: telemetry_snapshot.map(|snapshot| StiffAttemptTelemetry {
                solve_id,
                scope: match self.kind() {
                    StiffSolverKind::Bdf | StiffSolverKind::BackwardEuler => {
                        StiffTelemetryScope::Attempt
                    }
                    StiffSolverKind::Radau => StiffTelemetryScope::AttemptAndPreparation,
                    StiffSolverKind::Lsode2 => StiffTelemetryScope::SolverLifecycleCumulative,
                },
                snapshot,
            }),
            partial_trajectory,
        }
    }

    fn diagnostic_trajectory(
        &self,
        solve_id: u64,
        error: Option<&StiffAttemptError>,
    ) -> Option<StiffDiagnosticTrajectory> {
        let solver = self.kind();
        if matches!(
            error?,
            StiffAttemptError::Cancelled(StiffCancellationError::Bdf(_))
                | StiffAttemptError::Cancelled(StiffCancellationError::BackwardEuler(_))
        ) {
            let view = self.trajectory_view().ok()?;
            return Some(StiffDiagnosticTrajectory {
                solve_id,
                solver,
                axis: view.axis.clone(),
                values: view.values.clone(),
                accepted_steps: Some(view.axis.len().saturating_sub(1)),
            });
        }
        let from_native_snapshot = match error? {
            StiffAttemptError::Cancelled(StiffCancellationError::Bdf(
                BdfSolveError::Cancelled { snapshot },
            )) => Some((
                vec![snapshot.current_time],
                snapshot.current_state.clone(),
                Some(snapshot.accepted_steps),
            )),
            StiffAttemptError::Cancelled(StiffCancellationError::BackwardEuler(
                BeError::Cancelled { snapshot },
            )) => Some((
                vec![snapshot.current_time],
                snapshot.current_state.clone(),
                Some(snapshot.completed_steps),
            )),
            StiffAttemptError::Cancelled(StiffCancellationError::Lsode2(
                crate::numerical::LSODE2::Lsode2Error::Cancelled { snapshot },
            )) => {
                let n = snapshot.current_state.len();
                let rows = snapshot.accepted_states.len();
                let values = snapshot
                    .accepted_states
                    .iter()
                    .flat_map(|row| row.iter().copied())
                    .collect::<Vec<_>>();
                if rows > 0 && values.len() == rows.saturating_mul(n) {
                    return Some(StiffDiagnosticTrajectory {
                        solve_id,
                        solver,
                        axis: DVector::from_vec(snapshot.accepted_times.clone()),
                        values: DMatrix::from_row_slice(rows, n, &values),
                        accepted_steps: Some(rows.saturating_sub(1)),
                    });
                }
                Some((
                    vec![snapshot.current_time],
                    snapshot.current_state.clone(),
                    Some(rows.saturating_sub(1)),
                ))
            }
            StiffAttemptError::Cancelled(StiffCancellationError::Radau(
                crate::numerical::Radau::api::RadauCancellationError::Cancelled { snapshot },
            )) => Some((
                vec![snapshot.t],
                snapshot.y.clone(),
                Some(snapshot.accepted_steps),
            )),
            StiffAttemptError::Solve(StiffSolveError::Radau(_)) => {
                let failure = match self {
                    Self::Radau(prepared) => prepared.solver.last_failure_snapshot()?,
                    _ => return None,
                };
                Some((
                    vec![failure.t],
                    failure.y.clone(),
                    Some(failure.accepted_steps),
                ))
            }
            StiffAttemptError::Cancelled(_) => None,
            StiffAttemptError::Solve(_) => None,
        };
        if let Some((times, state, accepted_steps)) = from_native_snapshot {
            return diagnostic_from_rows(solve_id, solver, times, vec![state], accepted_steps);
        }

        let is_current_failure = match self {
            Self::Bdf(native) => {
                native.status_kind() == crate::numerical::BDF::BDF_api::BdfStatus::Failed
            }
            Self::BackwardEuler(native) => {
                native.status() == crate::numerical::BE::BeStatus::Failed
            }
            Self::Lsode2(native) => native.status() == "failed",
            Self::Radau(_) => false,
        };
        if !is_current_failure {
            return None;
        }
        let view = self.trajectory_view().ok()?;
        Some(StiffDiagnosticTrajectory {
            solve_id,
            solver,
            axis: view.axis.clone(),
            values: view.values.clone(),
            accepted_steps: Some(view.axis.len().saturating_sub(1)),
        })
    }

    fn is_in_native_failed_state(&self) -> bool {
        match self {
            Self::Bdf(native) => {
                native.status_kind() == crate::numerical::BDF::BDF_api::BdfStatus::Failed
            }
            Self::BackwardEuler(native) => {
                native.status() == crate::numerical::BE::BeStatus::Failed
            }
            Self::Lsode2(native) => native.status() == "failed",
            Self::Radau(_) => false,
        }
    }

    /// Starts a fresh solve with new parameter values and initial state while
    /// keeping the currently configured time interval. Native prepared
    /// callbacks and AOT artifacts are reused. This is not parameter
    /// continuation from the previous solution.
    pub fn try_solve_with_parameter_values(
        &mut self,
        y0: DVector<f64>,
        values: DVector<f64>,
    ) -> Result<StiffSolveOutcome, StiffSolveError> {
        match self {
            Self::Bdf(solver) => solver
                .try_solve_with_parameter_values(y0, values)
                .map(|()| StiffSolveOutcome::Bdf)
                .map_err(StiffSolveError::Bdf),
            Self::Radau(prepared) => {
                prepared.initial_state = y0.iter().copied().collect();
                prepared.parameters = values.iter().copied().collect();
                prepared
                    .solver
                    .try_solve_with_parameter_values(&prepared.initial_state, &prepared.parameters)
                    .map(StiffSolveOutcome::Radau)
                    .map_err(StiffSolveError::Radau)
            }
            Self::BackwardEuler(solver) => solver
                .try_solve_with_parameter_values(y0, values)
                .map(|()| StiffSolveOutcome::BackwardEuler)
                .map_err(StiffSolveError::BackwardEuler),
            Self::Lsode2(solver) => solver
                .try_solve_with_parameter_values(y0, values)
                .map(StiffSolveOutcome::Lsode2)
                .map_err(StiffSolveError::Lsode2),
        }
    }

    /// Solve each parameter vector independently from the same initial state.
    ///
    /// The prepared solver and symbolic/AOT callbacks are reused, while each
    /// point starts a fresh integration over the already configured interval.
    /// No previous terminal state is used as a guess. Native typed failures
    /// and telemetry are kept in the corresponding point and do not stop the
    /// remaining vectors.
    pub fn try_solve_parameter_loop(
        &mut self,
        y0: DVector<f64>,
        parameter_vectors: &[Vec<f64>],
    ) -> StiffParameterLoopReport {
        let report = run_ordered(parameter_vectors, |parameters| {
            let attempt = self.try_solve_with_parameter_values_report(
                y0.clone(),
                DVector::from_column_slice(parameters),
            );
            let metadata = StiffLoopAttemptMetadata {
                solve_id: attempt.solve_id,
                telemetry: attempt.telemetry,
                partial_trajectory: attempt.partial_trajectory,
            };
            let outcome = attempt.result.map_err(|error| match error {
                StiffAttemptError::Solve(error) => StiffParameterLoopError::Solve(error),
                StiffAttemptError::Cancelled(error) => StiffParameterLoopError::Cancelled(error),
            });
            let result = outcome.and_then(|outcome| match outcome {
                StiffSolveOutcome::Radau(solution) => {
                    let (x_mesh, values) = solution.output_trajectory().map_err(|error| {
                        StiffParameterLoopError::Solve(StiffSolveError::Radau(error))
                    })?;
                    Ok(IvpParameterLoopSolution { x_mesh, values })
                }
                StiffSolveOutcome::Bdf
                | StiffSolveOutcome::BackwardEuler
                | StiffSolveOutcome::Lsode2(_) => {
                    let view = self
                        .trajectory_view()
                        .map_err(StiffParameterLoopError::Output)?;
                    Ok(IvpParameterLoopSolution {
                        x_mesh: view.axis.clone(),
                        values: view.values.clone(),
                    })
                }
            });
            (result, metadata)
        });
        StiffParameterLoopReport {
            points: report
                .points
                .into_iter()
                .map(
                    |point: IvpParameterLoopPoint<
                        StiffLoopAttemptMetadata,
                        StiffParameterLoopError,
                    >| {
                        StiffParameterLoopPoint {
                            index: point.index,
                            solve_id: point.telemetry.solve_id,
                            parameter_values: point.parameter_values,
                            result: point.result,
                            telemetry: point.telemetry.telemetry,
                            partial_trajectory: point.telemetry.partial_trajectory,
                        }
                    },
                )
                .collect(),
        }
    }

    /// Starts a fresh solve with new parameter values, initial state and time
    /// interval. The prepared symbolic/AOT route is retained for the next run.
    pub fn try_solve_with_parameter_values_in_interval(
        &mut self,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        values: DVector<f64>,
    ) -> Result<StiffSolveOutcome, StiffSolveError> {
        match self {
            Self::Bdf(solver) => solver
                .try_solve_with_parameter_values_in_interval(t0, y0, t_bound, values)
                .map(|()| StiffSolveOutcome::Bdf)
                .map_err(StiffSolveError::Bdf),
            Self::Radau(prepared) => {
                prepared.initial_state = y0.iter().copied().collect();
                prepared.parameters = values.iter().copied().collect();
                prepared
                    .solver
                    .try_solve_with_parameter_values_in_interval(
                        t0,
                        &prepared.initial_state,
                        t_bound,
                        &prepared.parameters,
                    )
                    .map(StiffSolveOutcome::Radau)
                    .map_err(StiffSolveError::Radau)
            }
            Self::BackwardEuler(solver) => solver
                .try_solve_with_parameter_values_in_interval(t0, y0, t_bound, values)
                .map(|()| StiffSolveOutcome::BackwardEuler)
                .map_err(StiffSolveError::BackwardEuler),
            Self::Lsode2(solver) => solver
                .try_solve_with_parameter_values_in_interval(t0, y0, t_bound, values)
                .map(StiffSolveOutcome::Lsode2)
                .map_err(StiffSolveError::Lsode2),
        }
    }

    /// Start a fresh solve from `y0` on the supplied interval while retaining
    /// the current symbolic parameter binding and prepared callbacks.
    ///
    /// This resets native integrator history before solving; it is not an
    /// in-time continuation operation.
    pub fn try_solve_with_initial_state_in_interval(
        &mut self,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
    ) -> Result<StiffSolveOutcome, StiffSolveError> {
        self.try_restart_with_initial_state(t0, y0, t_bound)
            .map_err(StiffSolveError::Restart)?;
        self.try_solve()
    }

    /// Starts a fresh solve from `y0` on the currently configured interval,
    /// preserving bound parameter values and prepared callbacks/artifacts.
    /// This resets native integration history; it is not time continuation.
    pub fn try_solve_with_initial_state(
        &mut self,
        y0: DVector<f64>,
    ) -> Result<StiffSolveOutcome, StiffSolveError> {
        match self {
            Self::Bdf(solver) => solver
                .try_solve_with_initial_state(y0)
                .map(|()| StiffSolveOutcome::Bdf)
                .map_err(StiffSolveError::Bdf),
            Self::BackwardEuler(solver) => solver
                .try_solve_with_initial_state(y0)
                .map(|()| StiffSolveOutcome::BackwardEuler)
                .map_err(StiffSolveError::BackwardEuler),
            Self::Lsode2(solver) => solver
                .try_solve_with_initial_state(y0)
                .map(StiffSolveOutcome::Lsode2)
                .map_err(StiffSolveError::Lsode2),
            Self::Radau(prepared) => {
                if y0.len() != prepared.initial_state.len() {
                    return Err(StiffSolveError::Restart(
                        StiffRestartError::InitialStateDimension {
                            solver: StiffSolverKind::Radau,
                            expected: prepared.initial_state.len(),
                            actual: y0.len(),
                        },
                    ));
                }
                if y0.is_empty() || y0.iter().any(|value| !value.is_finite()) {
                    return Err(StiffSolveError::Restart(
                        StiffRestartError::InvalidInitialState {
                            solver: StiffSolverKind::Radau,
                        },
                    ));
                }
                prepared.initial_state = y0.iter().copied().collect();
                prepared
                    .solver
                    .try_solve_with_parameter_values(&prepared.initial_state, &prepared.parameters)
                    .map(StiffSolveOutcome::Radau)
                    .map_err(StiffSolveError::Radau)
            }
        }
    }

    /// Restart the prepared native route on a new initial-value segment while
    /// retaining its symbolic callbacks, generated artifacts and solver policy.
    pub fn try_restart_with_initial_state(
        &mut self,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
    ) -> Result<(), StiffRestartError> {
        let solver = self.kind();
        if !t0.is_finite() || !t_bound.is_finite() || t0 == t_bound {
            return Err(StiffRestartError::InvalidInterval { solver });
        }
        if y0.is_empty() || y0.iter().any(|value| !value.is_finite()) {
            return Err(StiffRestartError::InvalidInitialState { solver });
        }

        match self {
            Self::Bdf(native) => native
                .try_restart_with_initial_state(t0, y0, t_bound)
                .map_err(StiffRestartError::Bdf),
            Self::Radau(prepared) => {
                if y0.len() != prepared.initial_state.len() {
                    return Err(StiffRestartError::InitialStateDimension {
                        solver,
                        expected: prepared.initial_state.len(),
                        actual: y0.len(),
                    });
                }
                prepared
                    .solver
                    .restart(t0, t_bound)
                    .map_err(StiffRestartError::Radau)?;
                prepared.initial_state = y0.iter().copied().collect();
                Ok(())
            }
            Self::BackwardEuler(native) => native
                .try_restart_with_initial_state(t0, y0, t_bound)
                .map_err(StiffRestartError::BackwardEuler),
            Self::Lsode2(native) => native
                .try_restart_with_initial_state(t0, y0, t_bound)
                .map_err(StiffRestartError::Lsode2),
        }
    }
}

/// Solver-native telemetry returned through the common StiffAPI boundary.
///
/// The variants intentionally retain each solver's distinct statistics
/// surface. Timings may be nested or inclusive according to native scope
/// metadata; consumers must not sum them unless their native documentation
/// explicitly says they are disjoint.
#[derive(Debug, Clone)]
pub enum StiffTelemetrySnapshot {
    Bdf {
        mode: SolverTelemetryMode,
        statistics: IvpBackendStatistics,
        symbolic: Option<IvpTelemetrySnapshot>,
    },
    BackwardEuler {
        mode: SolverTelemetryMode,
        statistics: IvpBackendStatistics,
        details: crate::numerical::BE::BeDetailedStatistics,
        symbolic: Option<IvpTelemetrySnapshot>,
    },
    Lsode2 {
        mode: SolverTelemetryMode,
        statistics: IvpBackendStatistics,
        native: Lsode2NativeStatistics,
        symbolic: IvpTelemetrySnapshot,
    },
    Radau(RadauTelemetryReport),
}

impl StiffTelemetrySnapshot {
    /// Preserve the telemetry attached to a completed Radau solve result.
    pub fn from_radau_solution(solution: &RadauSolution) -> Self {
        Self::Radau(solution.telemetry().clone())
    }
}

fn lsode2_attempt_telemetry_delta(
    before: Lsode2NativeStatistics,
    after: Lsode2NativeStatistics,
    symbolic_before: IvpTelemetrySnapshot,
    symbolic_after: IvpTelemetrySnapshot,
    scope: StiffTelemetryScope,
) -> StiffLsode2AttemptTelemetryDelta {
    use crate::symbolic::ivp_telemetry::{IvpTelemetryMode, IvpWarmStage};

    let mut native_statistics_reset = after.solve_ms_total < before.solve_ms_total;
    let mut delta_usize = |before: usize, after: usize| {
        if after < before {
            native_statistics_reset = true;
            after
        } else {
            after - before
        }
    };
    let symbolic_counts = matches!(
        symbolic_after.mode,
        IvpTelemetryMode::Counters | IvpTelemetryMode::Detailed
    );
    let symbolic_timings = matches!(
        symbolic_after.mode,
        IvpTelemetryMode::Timings | IvpTelemetryMode::Detailed
    );
    let symbolic_counters_reset = symbolic_after.residual_evaluations
        < symbolic_before.residual_evaluations
        || symbolic_after.jacobian_evaluations < symbolic_before.jacobian_evaluations;
    let delta_u64 = |before: u64, after: u64| after.saturating_sub(before);
    let delta_duration =
        |before: std::time::Duration, after: std::time::Duration| after.saturating_sub(before);
    let symbolic_stage_delta = |stage| {
        delta_duration(
            symbolic_before.warm_stage(stage).elapsed,
            symbolic_after.warm_stage(stage).elapsed,
        )
    };
    let native_solve_calls = delta_usize(before.solve_calls, after.solve_calls);
    let native_solve_time_ms = (after.solve_ms_total - before.solve_ms_total).max(0.0);
    let native_step_attempts = delta_usize(before.native_step_attempts, after.native_step_attempts);
    let native_step_accepts = delta_usize(before.native_step_accepts, after.native_step_accepts);
    let native_step_rejects = delta_usize(
        before
            .native_step_rejects_error_test
            .saturating_add(before.native_step_rejects_nonlinear),
        after
            .native_step_rejects_error_test
            .saturating_add(after.native_step_rejects_nonlinear),
    );
    let native_residual_calls =
        delta_usize(before.native_residual_calls, after.native_residual_calls);
    let native_jacobian_calls =
        delta_usize(before.native_jacobian_calls, after.native_jacobian_calls);
    let native_linear_solve_calls = delta_usize(
        before.native_linear_solve_calls,
        after.native_linear_solve_calls,
    );
    StiffLsode2AttemptTelemetryDelta {
        scope,
        native_before: before,
        native_after: after,
        symbolic_residual_evaluations: symbolic_counts.then(|| {
            delta_u64(
                symbolic_before.residual_evaluations,
                symbolic_after.residual_evaluations,
            )
        }),
        symbolic_jacobian_evaluations: symbolic_counts.then(|| {
            delta_u64(
                symbolic_before.jacobian_evaluations,
                symbolic_after.jacobian_evaluations,
            )
        }),
        symbolic_residual_time: symbolic_timings
            .then(|| symbolic_stage_delta(IvpWarmStage::ResidualEvaluation)),
        symbolic_jacobian_time: symbolic_timings
            .then(|| symbolic_stage_delta(IvpWarmStage::JacobianEvaluation)),
        native_solve_calls,
        native_solve_time_ms,
        native_step_attempts,
        native_step_accepts,
        native_step_rejects,
        native_residual_calls,
        native_jacobian_calls,
        native_linear_solve_calls,
        native_statistics_reset,
        symbolic_counters_reset,
    }
}

/// Why a prepared solver cannot yet provide its native telemetry report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StiffTelemetryError {
    /// Radau creates its telemetry report together with the solve result.
    RadauSolutionRequired,
}

impl fmt::Display for StiffTelemetryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RadauSolutionRequired => write!(
                f,
                "Radau telemetry is available from the RadauSolution returned by solve"
            ),
        }
    }
}

impl Error for StiffTelemetryError {}

/// Borrowed, row-oriented time-series data retained by a prepared solver.
///
/// `values` has shape `(time_points, state_variables)`. Creating an owned
/// [`PostprocessDataset`] from this view is explicit because it copies both
/// arrays into the dataset's ownership.
pub struct StiffTrajectoryView<'a> {
    pub solver: StiffSolverKind,
    pub axis: &'a DVector<f64>,
    pub values: &'a DMatrix<f64>,
}

impl StiffTrajectoryView<'_> {
    /// Copy the borrowed trajectory into the owned common postprocessing type.
    pub fn to_dataset(
        &self,
        axis_name: impl Into<String>,
        variable_names: Vec<String>,
    ) -> Result<PostprocessDataset, StiffPostprocessError> {
        PostprocessDataset::new(
            axis_name,
            variable_names,
            self.axis.clone(),
            self.values.clone(),
        )
        .map_err(StiffPostprocessError::InvalidDataset)
    }

    /// Materialize the dataset and run the caller's explicit output plan.
    pub fn execute(
        &self,
        axis_name: impl Into<String>,
        variable_names: Vec<String>,
        plan: &PostprocessPlan,
    ) -> Result<PostprocessReport, StiffPostprocessError> {
        let dataset = self.to_dataset(axis_name, variable_names)?;
        plan.execute(&dataset)
            .map_err(|source| StiffPostprocessError::PlanExecution {
                solver: self.solver,
                source,
            })
    }
}

/// Convert Radau's native dense-output samples into an owned common dataset.
/// Sampling times and variable labels are explicit because Radau retains its
/// interpolation result separately from the prepared solver and does not
/// promise the other solvers' accepted-step trajectory layout.
pub fn radau_postprocess_dataset(
    solution: &crate::numerical::Radau::api::RadauSolution,
    sample_times: &[f64],
    axis_name: impl Into<String>,
    variable_names: Vec<String>,
) -> Result<PostprocessDataset, StiffPostprocessError> {
    let component_count = solution.y.len();
    let flattened = solution
        .sample_many(sample_times)
        .map_err(StiffPostprocessError::RadauSampling)?;
    let expected_len = sample_times
        .len()
        .checked_mul(component_count)
        .ok_or_else(|| {
            StiffPostprocessError::InvalidDataset(PostprocessError::InvalidDataset(
                "Radau sample dimensions overflow addressable memory".to_string(),
            ))
        })?;
    if flattened.len() != expected_len {
        return Err(StiffPostprocessError::InvalidDataset(
            PostprocessError::InvalidDataset(format!(
                "Radau returned {} sampled values; expected {expected_len}",
                flattened.len()
            )),
        ));
    }
    let values = DMatrix::from_row_slice(sample_times.len(), component_count, &flattened);
    PostprocessDataset::new(
        axis_name,
        variable_names,
        DVector::from_column_slice(sample_times),
        values,
    )
    .map_err(StiffPostprocessError::InvalidDataset)
}

/// Typed failures from the facade's explicit trajectory materialization and
/// postprocessing helpers. Native solver state remains available after a plan
/// fails; these operations never convert a successful solve into a solve error.
#[derive(Debug)]
pub enum StiffPostprocessError {
    ResultUnavailable {
        solver: StiffSolverKind,
    },
    RadauSolutionRequired,
    RadauSampling(RadauError),
    InvalidDataset(PostprocessError),
    PlanExecution {
        solver: StiffSolverKind,
        source: PostprocessError,
    },
}

impl fmt::Display for StiffPostprocessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ResultUnavailable { solver } => {
                write!(f, "{solver:?} has no retained trajectory to postprocess")
            }
            Self::RadauSolutionRequired => {
                f.write_str("Radau postprocessing requires the RadauSolution returned by solve(y0)")
            }
            Self::RadauSampling(error) => write!(f, "Radau output sampling failed: {error}"),
            Self::InvalidDataset(error) => write!(f, "invalid postprocessing dataset: {error}"),
            Self::PlanExecution { solver, source } => {
                write!(f, "{solver:?} postprocessing plan failed: {source}")
            }
        }
    }
}

impl Error for StiffPostprocessError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::RadauSampling(error) => Some(error),
            Self::InvalidDataset(error) | Self::PlanExecution { source: error, .. } => Some(error),
            Self::ResultUnavailable { .. } | Self::RadauSolutionRequired => None,
        }
    }
}

impl StiffSolverRequest {
    /// Identify the native family represented by this request before applying
    /// parser-produced policy settings.
    pub const fn solver_kind(&self) -> StiffSolverKind {
        match self {
            Self::Bdf { .. } => StiffSolverKind::Bdf,
            Self::Radau { .. } => StiffSolverKind::Radau,
            Self::BackwardEuler { .. } => StiffSolverKind::BackwardEuler,
            Self::Lsode2 { .. } => StiffSolverKind::Lsode2,
        }
    }

    /// Expand named aliases in the selected symbolic IVP before preparation.
    /// Alias expansion is a one-time symbolic input operation, independent of
    /// runtime parameter binding/rebinding and solver continuation.
    pub fn try_with_substitutions<I, N>(mut self, definitions: I) -> Result<Self, StiffApiError>
    where
        I: IntoIterator<Item = (N, Expr)>,
        N: Into<String>,
    {
        let solver = self.solver_kind();
        let definitions = definitions
            .into_iter()
            .map(|(name, expression)| (name.into(), expression))
            .collect::<Vec<_>>();
        let result = match &mut self {
            Self::Bdf { options } => options
                .clone()
                .try_with_substitutions(definitions)
                .map(|updated| *options = updated),
            Self::Radau { problem, .. } => problem
                .clone()
                .try_with_substitutions(definitions)
                .map(|updated| *problem = updated),
            Self::BackwardEuler {
                options,
                equation_parameters,
                ..
            } => {
                let mut options_with_aliases = options.clone();
                let reserved = options_with_aliases
                    .values
                    .iter()
                    .cloned()
                    .chain(std::iter::once(options_with_aliases.arg.clone()))
                    .chain(equation_parameters.iter().cloned())
                    .collect::<Vec<_>>();
                crate::symbolic::substitutions::apply_symbolic_substitutions(
                    &mut options_with_aliases.eq_system,
                    definitions,
                    &reserved,
                )
                .map(|()| *options = options_with_aliases)
            }
            Self::Lsode2 { config, .. } => config
                .clone()
                .try_with_substitutions(definitions)
                .map(|updated| *config = updated),
        };
        result.map_err(|source| StiffApiError::SymbolicSubstitution { solver, source })?;
        Ok(self)
    }

    /// Apply the common profile and prepare the selected native implementation.
    pub fn prepare(
        self,
        preset: Option<&StiffPreset>,
    ) -> Result<PreparedStiffSolver, StiffApiError> {
        match self {
            Self::Bdf { options } => prepare_bdf(options, preset).map(PreparedStiffSolver::Bdf),
            Self::Radau {
                problem,
                config,
                initial_state,
                parameters,
            } => prepare_radau(problem, config, preset).map(|solver| {
                PreparedStiffSolver::Radau(PreparedRadau {
                    solver,
                    initial_state,
                    parameters,
                })
            }),
            Self::BackwardEuler {
                options,
                equation_parameters,
                parameter_values,
            } => {
                let mut solver = prepare_backward_euler(options, preset)?;
                if !equation_parameters.is_empty() {
                    let names = equation_parameters
                        .iter()
                        .map(String::as_str)
                        .collect::<Vec<_>>();
                    solver
                        .try_set_equation_parameters(Some(&names))
                        .map_err(StiffApiError::BackwardEuler)?;
                }
                if !parameter_values.is_empty() {
                    solver
                        .try_set_parameter_values(DVector::from_vec(parameter_values))
                        .map_err(StiffApiError::BackwardEuler)?;
                }
                Ok(PreparedStiffSolver::BackwardEuler(solver))
            }
            Self::Lsode2 {
                config,
                logging_level,
            } => {
                let mut solver = prepare_lsode2(config, preset)?;
                if let Some(level) = preset.and_then(|preset| preset.logging).or(logging_level) {
                    solver.set_logging_level(level);
                }
                Ok(PreparedStiffSolver::Lsode2(solver))
            }
        }
    }
}

/// Stiff family selection plus common policy parsed from an external task
/// document. Family-specific numeric controls remain native request fields.
#[derive(Debug, Clone)]
pub struct StiffSolverSettings {
    solver: StiffSolverKind,
    preset: StiffPreset,
}

impl StiffSolverSettings {
    pub fn new(solver: StiffSolverKind) -> Self {
        Self {
            solver,
            preset: StiffPreset::default(),
        }
    }

    pub fn with_preset(mut self, preset: StiffPreset) -> Self {
        self.preset = preset;
        self
    }

    pub const fn solver_kind(&self) -> StiffSolverKind {
        self.solver
    }

    /// Check the document-selected family and route both programmatic and
    /// hybrid callers through the same native preparation path.
    pub fn prepare(
        self,
        request: StiffSolverRequest,
    ) -> Result<PreparedStiffSolver, StiffApiError> {
        let actual = request.solver_kind();
        if actual != self.solver {
            return Err(StiffApiError::SolverKindMismatch {
                selected: self.solver,
                request: actual,
            });
        }
        request.prepare(Some(&self.preset))
    }
}

/// Common symbolic frontend choice, translated to each native solver's type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StiffFrontend {
    ExprLegacy,
    AtomViewNative,
}

/// Common matrix-layout request. A solver must reject layouts it does not
/// natively support; the facade never densifies a sparse or banded request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StiffMatrixLayout {
    Dense,
    Sparse,
    /// Ask the selected solver to infer bandwidth from its prepared Jacobian
    /// structure, but only when that solver natively supports this policy.
    BandedAuto,
    /// Supply an explicit bandwidth hint when a solver supports it.
    /// Prefer `BandedAuto` when native symbolic inference is available; native
    /// routes validate explicit hints so they cannot truncate known structure.
    Banded {
        lower: usize,
        upper: usize,
    },
}

/// Symbolic execution route requested by a reusable preset.
#[derive(Debug, Clone)]
pub enum StiffExecution {
    Lambdify,
    Aot(SymbolicIvpGeneratedBackendConfig),
}

/// Callback scheduling requested through a solver preset.
///
/// This maps to native callback controls only for routes that expose them
/// (currently Radau and Lambdify callback routes in BDF/LSODE2). Other routes,
/// including Backward Euler and AOT callback routes without such a control,
/// reject it explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StiffCallbackPolicy {
    Sequential,
    Parallel { min_work: usize },
    Auto { min_work: usize },
}

/// Static applicability of a composable preset to one solver family.
///
/// `Conditional` means the selected preset can be applied only when the
/// request's native source/execution route meets the accompanying constraint.
/// Final validation still happens against the actual request before prepare.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StiffPresetApplicability {
    Supported,
    Conditional,
    Unsupported,
}

/// One solver's compatibility entry in a [`StiffPresetDescription`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StiffSolverApplicability {
    pub solver: StiffSolverKind,
    pub applicability: StiffPresetApplicability,
    pub constraint: Option<&'static str>,
}

/// Inspectable summary of the facade policy axes, tools and solver constraints
/// declared by a preset. Field names identify StiffAPI axes, not necessarily
/// fields in each solver's native config. This metadata is not a substitute
/// for request validation or native error reporting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StiffPresetDescription {
    pub fields_changed: Vec<&'static str>,
    /// External tools needed only when the selected AOT policy builds an
    /// artifact instead of loading an already-built one.
    pub tools_if_aot_build_needed: Vec<String>,
    pub required_state_dimension: Option<usize>,
    pub solvers: Vec<StiffSolverApplicability>,
}

/// Static metadata for a named preset constructor. External tools are needed
/// only if the configured AOT artifact must be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StiffPresetCatalogEntry {
    pub name: &'static str,
    pub purpose: &'static str,
    pub fields_changed: &'static [&'static str],
    pub tools_if_aot_build_needed: &'static [&'static str],
    pub requires_output_directory: bool,
}

const PRESET_CATALOG: &[StiffPresetCatalogEntry] = &[
    StiffPresetCatalogEntry {
        name: "solve_1d",
        purpose: "One-state ExprLegacy + Dense; native numerical defaults",
        fields_changed: &[
            "symbolic_frontend",
            "linear_system_layout",
            "state_dimension",
        ],
        tools_if_aot_build_needed: &[],
        requires_output_directory: false,
    },
    StiffPresetCatalogEntry {
        name: "expr_dense",
        purpose: "ExprLegacy frontend with Dense storage",
        fields_changed: &["symbolic_frontend", "linear_system_layout"],
        tools_if_aot_build_needed: &[],
        requires_output_directory: false,
    },
    StiffPresetCatalogEntry {
        name: "atom_dense",
        purpose: "AtomViewNative frontend with Dense storage",
        fields_changed: &["symbolic_frontend", "linear_system_layout"],
        tools_if_aot_build_needed: &[],
        requires_output_directory: false,
    },
    StiffPresetCatalogEntry {
        name: "expr_sparse",
        purpose: "ExprLegacy frontend with Sparse storage where supported",
        fields_changed: &["symbolic_frontend", "linear_system_layout"],
        tools_if_aot_build_needed: &[],
        requires_output_directory: false,
    },
    StiffPresetCatalogEntry {
        name: "atom_sparse",
        purpose: "AtomViewNative frontend with Sparse storage where supported",
        fields_changed: &["symbolic_frontend", "linear_system_layout"],
        tools_if_aot_build_needed: &[],
        requires_output_directory: false,
    },
    StiffPresetCatalogEntry {
        name: "expr_banded_auto",
        purpose: "ExprLegacy frontend with inferred Banded storage",
        fields_changed: &["symbolic_frontend", "linear_system_layout"],
        tools_if_aot_build_needed: &[],
        requires_output_directory: false,
    },
    StiffPresetCatalogEntry {
        name: "atom_banded_auto",
        purpose: "AtomViewNative frontend with inferred Banded storage",
        fields_changed: &["symbolic_frontend", "linear_system_layout"],
        tools_if_aot_build_needed: &[],
        requires_output_directory: false,
    },
    StiffPresetCatalogEntry {
        name: "lambdify",
        purpose: "Explicitly request the Lambdify symbolic route",
        fields_changed: &["symbolic_execution"],
        tools_if_aot_build_needed: &[],
        requires_output_directory: false,
    },
    StiffPresetCatalogEntry {
        name: "sequential_callbacks",
        purpose: "Request sequential callback evaluation where supported",
        fields_changed: &["callback_execution_policy"],
        tools_if_aot_build_needed: &[],
        requires_output_directory: false,
    },
    StiffPresetCatalogEntry {
        name: "parallel_callbacks",
        purpose: "Request parallel callbacks with a caller-supplied work threshold",
        fields_changed: &["callback_execution_policy"],
        tools_if_aot_build_needed: &[],
        requires_output_directory: false,
    },
    StiffPresetCatalogEntry {
        name: "automatic_callbacks",
        purpose: "Delegate callback parallelization to the native Auto policy",
        fields_changed: &["callback_execution_policy"],
        tools_if_aot_build_needed: &[],
        requires_output_directory: false,
    },
    StiffPresetCatalogEntry {
        name: "aot_tcc",
        purpose: "Build missing release AOT artifacts with C+tcc",
        fields_changed: &[
            "symbolic_execution",
            "aot_codegen_backend",
            "aot_c_compiler",
            "aot_build_policy",
            "aot_build_profile",
            "aot_artifact_location",
        ],
        tools_if_aot_build_needed: &["cargo", "rustc", "tcc"],
        requires_output_directory: true,
    },
    StiffPresetCatalogEntry {
        name: "aot_gcc",
        purpose: "Build missing release AOT artifacts with C+GCC",
        fields_changed: &[
            "symbolic_execution",
            "aot_codegen_backend",
            "aot_c_compiler",
            "aot_build_policy",
            "aot_build_profile",
            "aot_artifact_location",
        ],
        tools_if_aot_build_needed: &["cargo", "rustc", "gcc"],
        requires_output_directory: true,
    },
    StiffPresetCatalogEntry {
        name: "aot_zig",
        purpose: "Build missing release AOT artifacts with Zig",
        fields_changed: &[
            "symbolic_execution",
            "aot_codegen_backend",
            "aot_c_compiler",
            "aot_build_policy",
            "aot_build_profile",
            "aot_artifact_location",
        ],
        tools_if_aot_build_needed: &["cargo", "rustc", "zig"],
        requires_output_directory: true,
    },
    StiffPresetCatalogEntry {
        name: "aot_rust",
        purpose: "Build missing release AOT artifacts with Rust codegen",
        fields_changed: &[
            "symbolic_execution",
            "aot_codegen_backend",
            "aot_c_compiler",
            "aot_build_policy",
            "aot_build_profile",
            "aot_artifact_location",
        ],
        tools_if_aot_build_needed: &["cargo", "rustc"],
        requires_output_directory: true,
    },
];

/// A composable set of common solver-policy overrides.
///
/// An unset field leaves the selected solver's native value untouched. Compose
/// independent profiles explicitly; a second profile cannot silently replace
/// an axis already set by the first.
#[derive(Debug, Clone, Default)]
pub struct StiffPreset {
    frontend: Option<StiffFrontend>,
    matrix_layout: Option<StiffMatrixLayout>,
    execution: Option<StiffExecution>,
    callback_policy: Option<StiffCallbackPolicy>,
    telemetry: Option<SolverTelemetryMode>,
    logging: Option<SolverLogLevel>,
    required_state_dimension: Option<usize>,
}

impl StiffPreset {
    /// Return the stable names and metadata for built-in preset constructors.
    /// Parameterized callback/AOT constructors are represented by templates;
    /// their actual thresholds, artifact paths and applicability are available
    /// from [`Self::describe`] on the constructed preset.
    pub fn catalog() -> &'static [StiffPresetCatalogEntry] {
        PRESET_CATALOG
    }

    /// ExprLegacy frontend and dense storage profile.
    ///
    /// This selects no solver family and changes no tolerances, stepping
    /// controls, integration interval or solver-specific defaults.
    pub const fn expr_dense() -> Self {
        Self {
            frontend: Some(StiffFrontend::ExprLegacy),
            matrix_layout: Some(StiffMatrixLayout::Dense),
            execution: None,
            callback_policy: None,
            telemetry: None,
            logging: None,
            required_state_dimension: None,
        }
    }

    /// AtomViewNative frontend with Dense storage.
    pub const fn atom_dense() -> Self {
        Self {
            frontend: Some(StiffFrontend::AtomViewNative),
            matrix_layout: Some(StiffMatrixLayout::Dense),
            execution: None,
            callback_policy: None,
            telemetry: None,
            logging: None,
            required_state_dimension: None,
        }
    }

    /// ExprLegacy frontend with Sparse storage where supported.
    pub const fn expr_sparse() -> Self {
        Self {
            frontend: Some(StiffFrontend::ExprLegacy),
            matrix_layout: Some(StiffMatrixLayout::Sparse),
            execution: None,
            callback_policy: None,
            telemetry: None,
            logging: None,
            required_state_dimension: None,
        }
    }

    /// AtomViewNative frontend with Sparse storage where supported.
    pub const fn atom_sparse() -> Self {
        Self {
            frontend: Some(StiffFrontend::AtomViewNative),
            matrix_layout: Some(StiffMatrixLayout::Sparse),
            execution: None,
            callback_policy: None,
            telemetry: None,
            logging: None,
            required_state_dimension: None,
        }
    }

    /// ExprLegacy frontend with symbolically inferred Banded storage.
    pub const fn expr_banded_auto() -> Self {
        Self {
            frontend: Some(StiffFrontend::ExprLegacy),
            matrix_layout: Some(StiffMatrixLayout::BandedAuto),
            execution: None,
            callback_policy: None,
            telemetry: None,
            logging: None,
            required_state_dimension: None,
        }
    }

    /// AtomViewNative frontend with symbolically inferred Banded storage.
    pub const fn atom_banded_auto() -> Self {
        Self {
            frontend: Some(StiffFrontend::AtomViewNative),
            matrix_layout: Some(StiffMatrixLayout::BandedAuto),
            execution: None,
            callback_policy: None,
            telemetry: None,
            logging: None,
            required_state_dimension: None,
        }
    }

    /// Require an explicit Lambdify route.
    pub const fn lambdify() -> Self {
        Self {
            frontend: None,
            matrix_layout: None,
            execution: Some(StiffExecution::Lambdify),
            callback_policy: None,
            telemetry: None,
            logging: None,
            required_state_dimension: None,
        }
    }

    /// One-state quick profile: ExprLegacy + Dense, with all numerical solver
    /// controls left at their native defaults. The request must contain one
    /// equation and one initial state before any solver preparation begins.
    pub const fn solve_1d() -> Self {
        let mut preset = Self::expr_dense();
        preset.required_state_dimension = Some(1);
        preset
    }

    /// Require AOT and configure a release C+tcc build when the artifact is
    /// missing. The artifact directory is owned by the native AOT lifecycle.
    pub fn aot_tcc(output_parent_dir: impl Into<PathBuf>) -> Self {
        let generated =
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_c_tcc();
        Self::aot(generated)
    }

    /// Require a release C+GCC build when an AOT artifact is missing.
    pub fn aot_gcc(output_parent_dir: impl Into<PathBuf>) -> Self {
        let generated =
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_c_gcc();
        Self::aot(generated)
    }

    /// Require a release Zig build when an AOT artifact is missing.
    pub fn aot_zig(output_parent_dir: impl Into<PathBuf>) -> Self {
        let generated =
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_zig();
        Self::aot(generated)
    }

    /// Require a Rust-codegen release AOT build when an artifact is missing.
    pub fn aot_rust(output_parent_dir: impl Into<PathBuf>) -> Self {
        let generated =
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_rust();
        Self::aot(generated)
    }

    fn aot(generated: SymbolicIvpGeneratedBackendConfig) -> Self {
        Self {
            frontend: None,
            matrix_layout: None,
            execution: Some(StiffExecution::Aot(generated)),
            callback_policy: None,
            telemetry: None,
            logging: None,
            required_state_dimension: None,
        }
    }

    pub const fn with_frontend(mut self, frontend: StiffFrontend) -> Self {
        self.frontend = Some(frontend);
        self
    }

    pub const fn with_matrix_layout(mut self, layout: StiffMatrixLayout) -> Self {
        self.matrix_layout = Some(layout);
        self
    }

    pub fn with_execution(mut self, execution: StiffExecution) -> Self {
        self.execution = Some(execution);
        self
    }

    /// Select a native residual/Jacobian callback scheduling policy where
    /// exposed by the selected solver and execution route.
    pub const fn with_callback_policy(mut self, policy: StiffCallbackPolicy) -> Self {
        self.callback_policy = Some(policy);
        self
    }

    /// Sequential callback evaluation profile.
    pub fn sequential_callbacks() -> Self {
        Self::default().with_callback_policy(StiffCallbackPolicy::Sequential)
    }

    /// Force parallel callback evaluation after the native minimum-work gate.
    pub fn parallel_callbacks(min_work: usize) -> Self {
        Self::default().with_callback_policy(StiffCallbackPolicy::Parallel { min_work })
    }

    /// Let the native callback evaluator decide when parallel dispatch pays off.
    pub fn automatic_callbacks(min_work: usize) -> Self {
        Self::default().with_callback_policy(StiffCallbackPolicy::Auto { min_work })
    }

    pub const fn with_telemetry(mut self, telemetry: SolverTelemetryMode) -> Self {
        self.telemetry = Some(telemetry);
        self
    }

    pub const fn with_logging(mut self, logging: SolverLogLevel) -> Self {
        self.logging = Some(logging);
        self
    }

    /// Combine profiles only when they configure disjoint axes.
    pub fn compose(self, other: Self) -> Result<Self, StiffApiError> {
        macro_rules! merge_axis {
            ($left:expr, $right:expr, $name:literal) => {
                match ($left, $right) {
                    (Some(_), Some(_)) => {
                        return Err(StiffApiError::ConflictingPresetAxis($name));
                    }
                    (Some(value), None) | (None, Some(value)) => Some(value),
                    (None, None) => None,
                }
            };
        }

        Ok(Self {
            frontend: merge_axis!(self.frontend, other.frontend, "frontend"),
            matrix_layout: merge_axis!(self.matrix_layout, other.matrix_layout, "matrix_layout"),
            execution: merge_axis!(self.execution, other.execution, "execution"),
            callback_policy: merge_axis!(
                self.callback_policy,
                other.callback_policy,
                "callback_policy"
            ),
            telemetry: merge_axis!(self.telemetry, other.telemetry, "telemetry"),
            logging: merge_axis!(self.logging, other.logging, "logging"),
            required_state_dimension: merge_axis!(
                self.required_state_dimension,
                other.required_state_dimension,
                "required_state_dimension"
            ),
        })
    }

    /// Describe the exact common axes configured by this profile, its AOT
    /// tool requirements, and solver-specific applicability constraints.
    ///
    /// The request adapter remains authoritative because some constraints
    /// depend on native input (for example, symbolic versus analytical
    /// callbacks) that is not stored in a reusable preset.
    pub fn describe(&self) -> StiffPresetDescription {
        let mut fields_changed = Vec::new();
        if self.frontend.is_some() {
            fields_changed.push("symbolic_frontend");
        }
        if self.matrix_layout.is_some() {
            fields_changed.push("linear_system_layout");
        }
        if let Some(execution) = &self.execution {
            match execution {
                StiffExecution::Lambdify => fields_changed.push("symbolic_execution"),
                StiffExecution::Aot(_) => fields_changed.extend([
                    "symbolic_execution",
                    "aot_codegen_backend",
                    "aot_c_compiler",
                    "aot_build_policy",
                    "aot_build_profile",
                    "aot_artifact_location",
                ]),
            }
        }
        if self.callback_policy.is_some() {
            fields_changed.push("callback_execution_policy");
        }
        if self.telemetry.is_some() {
            fields_changed.push("telemetry_mode");
        }
        if self.logging.is_some() {
            fields_changed.push("logging_level");
        }
        if self.required_state_dimension.is_some() {
            fields_changed.push("state_dimension");
        }

        let mut tools_if_aot_build_needed = Vec::new();
        if let Some(StiffExecution::Aot(generated)) = &self.execution {
            if matches!(
                generated.build_policy,
                SymbolicIvpAotBuildPolicy::BuildIfMissing { .. }
                    | SymbolicIvpAotBuildPolicy::RebuildAlways { .. }
            ) {
                tools_if_aot_build_needed.push("cargo".to_string());
                tools_if_aot_build_needed.push("rustc".to_string());
                match generated.aot_codegen_backend {
                    crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend::C => {
                        tools_if_aot_build_needed.push(
                            generated
                                .aot_c_compiler
                                .clone()
                                .unwrap_or_else(|| "configured C compiler".to_string()),
                        );
                    }
                    crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend::Zig => {
                        tools_if_aot_build_needed.push("zig".to_string());
                    }
                    crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend::Rust => {}
                }
            }
        }

        StiffPresetDescription {
            fields_changed,
            tools_if_aot_build_needed,
            required_state_dimension: self.required_state_dimension,
            solvers: [
                StiffSolverKind::Bdf,
                StiffSolverKind::Radau,
                StiffSolverKind::BackwardEuler,
                StiffSolverKind::Lsode2,
            ]
            .into_iter()
            .map(|solver| self.applicability_for(solver))
            .collect(),
        }
    }

    fn applicability_for(&self, solver: StiffSolverKind) -> StiffSolverApplicability {
        let mut conditional = None;
        let mut unsupported = None;
        match solver {
            StiffSolverKind::Bdf => {
                if self
                    .matrix_layout
                    .is_some_and(|layout| layout != StiffMatrixLayout::Dense)
                {
                    unsupported = Some("BDF supports Dense linear algebra only");
                }
                if self.callback_policy.is_some() {
                    if matches!(self.execution.as_ref(), Some(StiffExecution::Aot(_))) {
                        unsupported =
                            Some("BDF callback scheduling is unavailable for AOT callbacks");
                    } else {
                        conditional = Some("requires effective Lambdify callbacks");
                    }
                }
            }
            StiffSolverKind::Radau => {}
            StiffSolverKind::BackwardEuler => {
                if self
                    .matrix_layout
                    .is_some_and(|layout| layout != StiffMatrixLayout::Dense)
                {
                    unsupported = Some("Backward Euler supports Dense linear algebra only");
                }
                if self.callback_policy.is_some() {
                    unsupported = Some("Backward Euler has no callback scheduling control");
                }
                if matches!(self.execution.as_ref(), Some(StiffExecution::Lambdify)) {
                    unsupported = Some("Backward Euler cannot force Lambdify execution");
                }
            }
            StiffSolverKind::Lsode2 => {
                if self.frontend.is_some() {
                    conditional =
                        Some("requires an LSODE2 symbolic source, not analytical callbacks");
                }
                if self.matrix_layout == Some(StiffMatrixLayout::BandedAuto) {
                    conditional = Some("BandedAuto requires symbolic Jacobian structure");
                }
                if self.callback_policy.is_some() {
                    if matches!(self.execution.as_ref(), Some(StiffExecution::Aot(_))) {
                        unsupported =
                            Some("LSODE2 callback scheduling is unavailable for AOT callbacks");
                    } else {
                        conditional = Some("requires the LSODE2 Lambdify callback route");
                    }
                }
            }
        }

        let (applicability, constraint) = if let Some(reason) = unsupported {
            (StiffPresetApplicability::Unsupported, Some(reason))
        } else if let Some(reason) = conditional {
            (StiffPresetApplicability::Conditional, Some(reason))
        } else {
            (StiffPresetApplicability::Supported, None)
        };
        StiffSolverApplicability {
            solver,
            applicability,
            constraint,
        }
    }

    fn apply_radau(&self, config: &mut RadauConfig) -> Result<(), StiffApiError> {
        if let Some(level) = self.logging {
            config.logging_level = level;
        }
        if let Some(frontend) = self.frontend {
            config.frontend = match frontend {
                StiffFrontend::ExprLegacy => RadauFrontend::ExprLegacy,
                StiffFrontend::AtomViewNative => RadauFrontend::AtomViewNative,
            };
        }
        if let Some(policy) = self.callback_policy {
            config.execution_policy = match policy {
                StiffCallbackPolicy::Sequential => RadauExecutionPolicy::Sequential,
                StiffCallbackPolicy::Parallel { min_work } => {
                    RadauExecutionPolicy::Parallel { min_work }
                }
                StiffCallbackPolicy::Auto { min_work } => RadauExecutionPolicy::Auto { min_work },
            };
        }
        if let Some(layout) = self.matrix_layout {
            config.matrix_layout = match layout {
                StiffMatrixLayout::Dense => RadauMatrixLayout::Dense,
                StiffMatrixLayout::Sparse => RadauMatrixLayout::Sparse,
                StiffMatrixLayout::BandedAuto => RadauMatrixLayout::BandedAuto,
                StiffMatrixLayout::Banded { lower, upper } => {
                    RadauMatrixLayout::Banded { lower, upper }
                }
            };
        }
        if let Some(telemetry) = self.telemetry {
            config.telemetry = match telemetry {
                SolverTelemetryMode::Off => crate::numerical::Radau::api::RadauTelemetryMode::Off,
                SolverTelemetryMode::Counters => {
                    crate::numerical::Radau::api::RadauTelemetryMode::Counters
                }
                SolverTelemetryMode::Timings => {
                    crate::numerical::Radau::api::RadauTelemetryMode::Timings
                }
                SolverTelemetryMode::Detailed => {
                    crate::numerical::Radau::api::RadauTelemetryMode::Detailed
                }
            };
        }
        if let Some(execution) = &self.execution {
            match execution {
                StiffExecution::Lambdify => {
                    config.execution = RadauExecution::Lambdify;
                    config.aot = None;
                }
                StiffExecution::Aot(generated) => {
                    config.execution = RadauExecution::Aot;
                    config.aot = Some(RadauAotConfig {
                        generated: generated.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    fn apply_bdf(&self, options: &mut BdfSolverOptions) -> Result<(), StiffApiError> {
        if let Some(layout) = self.matrix_layout
            && layout != StiffMatrixLayout::Dense
        {
            return Err(StiffApiError::UnsupportedCapability {
                solver: StiffSolverKind::Bdf,
                capability: "BDF currently supports dense linear algebra only",
            });
        }
        if let Some(frontend) = self.frontend {
            options.symbolic_assembly_backend = match frontend {
                StiffFrontend::ExprLegacy => IvpSymbolicAssemblyBackend::ExprLegacy,
                StiffFrontend::AtomViewNative => IvpSymbolicAssemblyBackend::AtomView,
            };
        }
        if let Some(policy) = self.callback_policy {
            options.lambdify_execution_policy = match policy {
                StiffCallbackPolicy::Sequential => {
                    crate::symbolic::ivp_telemetry::IvpLambdifyExecutionPolicy::Sequential
                }
                StiffCallbackPolicy::Parallel { min_work } => {
                    crate::symbolic::ivp_telemetry::IvpLambdifyExecutionPolicy::Parallel {
                        min_work,
                    }
                }
                StiffCallbackPolicy::Auto { min_work } => {
                    crate::symbolic::ivp_telemetry::IvpLambdifyExecutionPolicy::Auto { min_work }
                }
            };
        }
        if let Some(telemetry) = self.telemetry {
            options.telemetry_mode = match telemetry {
                SolverTelemetryMode::Off => BdfTelemetryMode::Off,
                SolverTelemetryMode::Counters => BdfTelemetryMode::Counters,
                SolverTelemetryMode::Timings => BdfTelemetryMode::Timings,
                SolverTelemetryMode::Detailed => BdfTelemetryMode::Detailed,
            };
        }
        if let Some(level) = self.logging {
            options.logging_level = level;
        }
        if let Some(execution) = &self.execution {
            match execution {
                StiffExecution::Lambdify => {
                    // BDF selects generated artifacts through its resolver.
                    // Clear both registry sources to make this explicit preset
                    // an actual Lambdify request rather than a preference.
                    options.generated_backend_config.resolver = None;
                    options.generated_backend_config.handoff_path = None;
                    options.generated_backend_config.build_policy =
                        SymbolicIvpAotBuildPolicy::UseIfAvailable;
                }
                StiffExecution::Aot(generated) => {
                    if self.callback_policy.is_some() {
                        return Err(StiffApiError::UnsupportedCapability {
                            solver: StiffSolverKind::Bdf,
                            capability: "BDF callback scheduling controls apply only to Lambdify, not AOT callbacks",
                        });
                    }
                    if generated.build_policy == SymbolicIvpAotBuildPolicy::UseIfAvailable {
                        return Err(StiffApiError::UnsupportedCapability {
                            solver: StiffSolverKind::Bdf,
                            capability: "AOT selection must use RequirePrebuilt, BuildIfMissing or RebuildAlways; UseIfAvailable permits Lambdify fallback",
                        });
                    }
                    options.generated_backend_config = generated.clone();
                }
            }
        }
        Ok(())
    }

    fn apply_be(&self, options: &mut BeSolverOptions) -> Result<(), StiffApiError> {
        if self.callback_policy.is_some() {
            return Err(StiffApiError::UnsupportedCapability {
                solver: StiffSolverKind::BackwardEuler,
                capability: "callback scheduling policy is not exposed by Backward Euler",
            });
        }
        if let Some(layout) = self.matrix_layout
            && layout != StiffMatrixLayout::Dense
        {
            return Err(StiffApiError::UnsupportedCapability {
                solver: StiffSolverKind::BackwardEuler,
                capability: "only dense linear algebra is implemented by Backward Euler",
            });
        }
        if let Some(level) = self.logging {
            options.logging_level = level;
        }
        if let Some(frontend) = self.frontend {
            options.symbolic_assembly_backend = match frontend {
                StiffFrontend::ExprLegacy => BeSymbolicAssemblyBackend::ExprLegacy,
                StiffFrontend::AtomViewNative => BeSymbolicAssemblyBackend::AtomViewNative,
            };
        }
        if let Some(telemetry) = self.telemetry {
            options.telemetry_mode = match telemetry {
                SolverTelemetryMode::Off => BeTelemetryMode::Off,
                SolverTelemetryMode::Counters => BeTelemetryMode::Counters,
                SolverTelemetryMode::Timings => BeTelemetryMode::Timings,
                SolverTelemetryMode::Detailed => BeTelemetryMode::Detailed,
            };
        }
        if let Some(execution) = &self.execution {
            match execution {
                StiffExecution::Lambdify => {
                    return Err(StiffApiError::UnsupportedCapability {
                        solver: StiffSolverKind::BackwardEuler,
                        capability: "forcing Lambdify is unavailable; the native generated-backend resolver may select a registered AOT artifact",
                    });
                }
                StiffExecution::Aot(generated) => {
                    if generated.build_policy == SymbolicIvpAotBuildPolicy::UseIfAvailable {
                        return Err(StiffApiError::UnsupportedCapability {
                            solver: StiffSolverKind::BackwardEuler,
                            capability: "AOT selection must use RequirePrebuilt, BuildIfMissing or RebuildAlways; UseIfAvailable permits Lambdify fallback",
                        });
                    }
                    options.generated_backend_config = generated.clone();
                }
            }
        }
        Ok(())
    }

    fn apply_lsode2(&self, config: &mut Lsode2ProblemConfig) -> Result<(), StiffApiError> {
        if let Some(frontend) = self.frontend {
            let execution = match config.residual_jacobian_source {
                Lsode2ResidualJacobianSource::Symbolic { execution, .. } => execution,
                Lsode2ResidualJacobianSource::Analytical => {
                    return Err(unsupported_lsode2(
                        "a symbolic frontend preset cannot replace analytical callbacks",
                    ));
                }
            };
            config.residual_jacobian_source = Lsode2ResidualJacobianSource::Symbolic {
                assembly: match frontend {
                    StiffFrontend::ExprLegacy => Lsode2SymbolicAssemblyBackend::ExprLegacy,
                    StiffFrontend::AtomViewNative => Lsode2SymbolicAssemblyBackend::AtomView,
                },
                execution,
            };
            config.sync_legacy_backend_from_policy();
        }

        if let Some(layout) = self.matrix_layout {
            let structure = match layout {
                StiffMatrixLayout::Dense => Lsode2LinearSystemStructure::Dense,
                StiffMatrixLayout::Sparse => Lsode2LinearSystemStructure::Sparse,
                StiffMatrixLayout::Banded { lower, upper } => Lsode2LinearSystemStructure::Banded {
                    kl: lower,
                    ku: upper,
                },
                StiffMatrixLayout::BandedAuto => {
                    if !matches!(
                        config.residual_jacobian_source,
                        Lsode2ResidualJacobianSource::Symbolic { .. }
                    ) {
                        return Err(unsupported_lsode2(
                            "automatic band inference requires symbolic Jacobian structure",
                        ));
                    }
                    Lsode2LinearSystemStructure::BandedAuto
                }
            };
            let expected_solver = match layout {
                StiffMatrixLayout::Dense => Lsode2LinearSolverChoice::DenseLu,
                StiffMatrixLayout::Sparse => Lsode2LinearSolverChoice::FaerSparseLu,
                StiffMatrixLayout::BandedAuto | StiffMatrixLayout::Banded { .. } => {
                    Lsode2LinearSolverChoice::LapackFaithfulBandedLu
                }
            };
            if let Lsode2LinearSolverPolicy::Force(choice) = config.linear_solver_policy
                && choice != expected_solver
            {
                return Err(unsupported_lsode2(
                    "requested matrix layout conflicts with the explicitly forced native linear solver",
                ));
            }
            config.linear_system_structure = structure;
            config.sync_legacy_backend_from_policy();
        }

        if let Some(execution) = &self.execution {
            let assembly = match config.residual_jacobian_source {
                Lsode2ResidualJacobianSource::Symbolic { assembly, .. } => assembly,
                Lsode2ResidualJacobianSource::Analytical => {
                    return Err(unsupported_lsode2(
                        "symbolic execution presets cannot replace analytical callbacks",
                    ));
                }
            };
            let mode = match execution {
                StiffExecution::Lambdify => Lsode2SymbolicExecutionMode::LambdifyExpr,
                StiffExecution::Aot(generated) => {
                    if generated.build_policy == SymbolicIvpAotBuildPolicy::UseIfAvailable {
                        return Err(unsupported_lsode2(
                            "AOT selection must prohibit silent Lambdify fallback",
                        ));
                    }
                    let toolchain = match generated.aot_codegen_backend {
                        crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend::C => {
                            match generated.aot_c_compiler.as_deref() {
                                Some("tcc") => Lsode2AotToolchain::CTcc,
                                Some("gcc") => Lsode2AotToolchain::CGcc,
                                _ => {
                                    return Err(unsupported_lsode2(
                                        "LSODE2 common AOT presets support C+tcc, C+gcc, Zig and Rust toolchains",
                                    ));
                                }
                            }
                        }
                        crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend::Zig => {
                            Lsode2AotToolchain::Zig
                        }
                        crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend::Rust => {
                            Lsode2AotToolchain::Rust
                        }
                    };
                    let profile = match generated.build_policy {
                        SymbolicIvpAotBuildPolicy::RequirePrebuilt => Lsode2AotProfile::Release,
                        SymbolicIvpAotBuildPolicy::BuildIfMissing { profile }
                        | SymbolicIvpAotBuildPolicy::RebuildAlways { profile } => match profile {
                            AotBuildProfile::Debug => Lsode2AotProfile::Debug,
                            AotBuildProfile::Release => Lsode2AotProfile::Release,
                        },
                        SymbolicIvpAotBuildPolicy::UseIfAvailable => unreachable!(),
                    };
                    config.backend.generated_backend = generated.clone();
                    Lsode2SymbolicExecutionMode::Aot { toolchain, profile }
                }
            };
            config.residual_jacobian_source = Lsode2ResidualJacobianSource::Symbolic {
                assembly,
                execution: mode,
            };
            if matches!(mode, Lsode2SymbolicExecutionMode::LambdifyExpr) {
                config.backend.generated_backend.resolver = None;
                config.backend.generated_backend.handoff_path = None;
            }
            config.sync_legacy_backend_from_policy();
        }

        if let Some(policy) = self.callback_policy {
            if !matches!(
                config.residual_jacobian_source,
                Lsode2ResidualJacobianSource::Symbolic {
                    execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
                    ..
                }
            ) {
                return Err(unsupported_lsode2(
                    "callback scheduling is supported only for the Lambdify route",
                ));
            }
            config.lambdify_execution_policy = match policy {
                StiffCallbackPolicy::Sequential => IvpLambdifyExecutionPolicy::Sequential,
                StiffCallbackPolicy::Parallel { min_work } => {
                    IvpLambdifyExecutionPolicy::Parallel { min_work }
                }
                StiffCallbackPolicy::Auto { min_work } => {
                    IvpLambdifyExecutionPolicy::Auto { min_work }
                }
            };
        }

        if let Some(telemetry) = self.telemetry {
            config.telemetry = crate::numerical::LSODE2::IvpTelemetry::with_mode(match telemetry {
                SolverTelemetryMode::Off => crate::numerical::LSODE2::IvpTelemetryMode::Off,
                SolverTelemetryMode::Counters => {
                    crate::numerical::LSODE2::IvpTelemetryMode::Counters
                }
                SolverTelemetryMode::Timings => crate::numerical::LSODE2::IvpTelemetryMode::Timings,
                SolverTelemetryMode::Detailed => {
                    crate::numerical::LSODE2::IvpTelemetryMode::Detailed
                }
            });
        }
        Ok(())
    }
}

fn unsupported_lsode2(capability: &'static str) -> StiffApiError {
    StiffApiError::UnsupportedCapability {
        solver: StiffSolverKind::Lsode2,
        capability,
    }
}

/// Construct a prepared Radau solver after applying the optional common profile.
pub fn prepare_radau(
    problem: crate::numerical::Radau::api::RadauProblem,
    mut config: RadauConfig,
    preset: Option<&StiffPreset>,
) -> Result<RadauSolver, StiffApiError> {
    validate_state_dimension(preset, StiffSolverKind::Radau, problem.dimension())?;
    validate_state_dimension(preset, StiffSolverKind::Radau, problem.state_dimension())?;
    if let Some(preset) = preset {
        preset.apply_radau(&mut config)?;
    }
    RadauSolver::prepare(problem, config).map_err(StiffApiError::Radau)
}

/// Prepare a BDF solver through its fallible native symbolic preparation API.
pub fn prepare_bdf(
    mut options: BdfSolverOptions,
    preset: Option<&StiffPreset>,
) -> Result<BdfSolver, StiffApiError> {
    validate_state_dimension(preset, StiffSolverKind::Bdf, options.eq_system.len())?;
    validate_state_dimension(preset, StiffSolverKind::Bdf, options.y0.len())?;
    if let Some(preset) = preset {
        preset.apply_bdf(&mut options)?;
    }
    let mut solver = BdfSolver::new_with_options(options);
    solver
        .try_generate()
        .map_err(|error| StiffApiError::Bdf(BdfSolveError::Backend(error)))?;
    Ok(solver)
}

/// Construct a Backward Euler solver after applying the optional common profile.
pub fn prepare_backward_euler(
    mut options: BeSolverOptions,
    preset: Option<&StiffPreset>,
) -> Result<BE, StiffApiError> {
    validate_state_dimension(
        preset,
        StiffSolverKind::BackwardEuler,
        options.eq_system.len(),
    )?;
    validate_state_dimension(preset, StiffSolverKind::BackwardEuler, options.y0.len())?;
    if let Some(preset) = preset {
        preset.apply_be(&mut options)?;
    }
    BE::try_new_with_options(options).map_err(StiffApiError::BackwardEuler)
}

/// Construct LSODE2 without flattening its controller or execution lifecycle.
pub fn prepare_lsode2(
    mut config: Lsode2ProblemConfig,
    preset: Option<&StiffPreset>,
) -> Result<Lsode2Solver, StiffApiError> {
    validate_state_dimension(preset, StiffSolverKind::Lsode2, config.eq_system.len())?;
    validate_state_dimension(preset, StiffSolverKind::Lsode2, config.y0.len())?;
    if let Some(preset) = preset {
        preset.apply_lsode2(&mut config)?;
    }
    if matches!(
        config.residual_jacobian_source,
        Lsode2ResidualJacobianSource::Symbolic {
            execution: Lsode2SymbolicExecutionMode::PreparedAtom,
            ..
        }
    ) {
        return Err(unsupported_lsode2(
            "PreparedAtom is experimental and excluded from the production StiffAPI",
        ));
    }
    let logging = preset.and_then(|preset| preset.logging);
    let mut solver = Lsode2Solver::new(config).map_err(StiffApiError::Lsode2)?;
    if let Some(level) = logging {
        solver.set_logging_level(level);
    }
    Ok(solver)
}

fn validate_state_dimension(
    preset: Option<&StiffPreset>,
    solver: StiffSolverKind,
    actual: usize,
) -> Result<(), StiffApiError> {
    if let Some(expected) = preset.and_then(|preset| preset.required_state_dimension)
        && actual != expected
    {
        return Err(StiffApiError::DimensionMismatch {
            solver,
            expected,
            actual,
        });
    }
    Ok(())
}

/// Typed facade failures. Native errors remain intact as source errors.
#[derive(Debug)]
pub enum StiffApiError {
    SymbolicSubstitution {
        solver: StiffSolverKind,
        source: SymbolicSubstitutionError,
    },
    ConflictingPresetAxis(&'static str),
    SolverKindMismatch {
        selected: StiffSolverKind,
        request: StiffSolverKind,
    },
    UnsupportedCapability {
        solver: StiffSolverKind,
        capability: &'static str,
    },
    DimensionMismatch {
        solver: StiffSolverKind,
        expected: usize,
        actual: usize,
    },
    Radau(RadauError),
    Bdf(BdfSolveError),
    BackwardEuler(BeError),
    Lsode2(Lsode2Error),
}

/// Stage at which a StiffAPI request failed. Solver-operation failures after
/// preparation are returned by the native solver and keep their native types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StiffApiErrorStage {
    Configuration,
    Preparation,
}

impl StiffApiError {
    /// Identify the solver involved when the error is solver-specific.
    pub const fn solver_kind(&self) -> Option<StiffSolverKind> {
        match self {
            Self::SymbolicSubstitution { solver, .. } => Some(*solver),
            Self::UnsupportedCapability { solver, .. } | Self::DimensionMismatch { solver, .. } => {
                Some(*solver)
            }
            Self::SolverKindMismatch { selected, .. } => Some(*selected),
            Self::Radau(_) => Some(StiffSolverKind::Radau),
            Self::Bdf(_) => Some(StiffSolverKind::Bdf),
            Self::BackwardEuler(_) => Some(StiffSolverKind::BackwardEuler),
            Self::Lsode2(_) => Some(StiffSolverKind::Lsode2),
            Self::ConflictingPresetAxis(_) => None,
        }
    }

    /// Classify facade validation separately from native preparation errors.
    pub const fn stage(&self) -> StiffApiErrorStage {
        match self {
            Self::SymbolicSubstitution { .. } => StiffApiErrorStage::Configuration,
            Self::ConflictingPresetAxis(_)
            | Self::SolverKindMismatch { .. }
            | Self::UnsupportedCapability { .. }
            | Self::DimensionMismatch { .. } => StiffApiErrorStage::Configuration,
            Self::Radau(_) | Self::Bdf(_) | Self::BackwardEuler(_) | Self::Lsode2(_) => {
                StiffApiErrorStage::Preparation
            }
        }
    }
}

impl fmt::Display for StiffApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SymbolicSubstitution { solver, source } => {
                write!(f, "invalid {solver:?} symbolic substitutions: {source}")
            }
            Self::ConflictingPresetAxis(axis) => {
                write!(f, "multiple StiffAPI presets configure the {axis} axis")
            }
            Self::SolverKindMismatch { selected, request } => write!(
                f,
                "hybrid settings select {selected:?}, but the numerical request is {request:?}"
            ),
            Self::UnsupportedCapability { solver, capability } => {
                write!(
                    f,
                    "{solver:?} does not support this StiffAPI request: {capability}"
                )
            }
            Self::DimensionMismatch {
                solver,
                expected,
                actual,
            } => write!(
                f,
                "{solver:?} request has state dimension {actual}; preset requires {expected}"
            ),
            Self::Radau(error) => write!(f, "Radau failed: {error}"),
            Self::Bdf(error) => write!(f, "BDF failed: {error}"),
            Self::BackwardEuler(error) => write!(f, "Backward Euler failed: {error}"),
            Self::Lsode2(error) => write!(f, "LSODE2 failed: {error}"),
        }
    }
}

impl Error for StiffApiError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::SymbolicSubstitution { source, .. } => Some(source),
            Self::Radau(error) => Some(error),
            Self::Bdf(error) => Some(error),
            Self::BackwardEuler(error) => Some(error),
            Self::Lsode2(error) => Some(error),
            Self::ConflictingPresetAxis(_)
            | Self::SolverKindMismatch { .. }
            | Self::UnsupportedCapability { .. }
            | Self::DimensionMismatch { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests;
