use crate::numerical::Nonlinear_systems::error::SolveError;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::time::Duration;

/// Last accepted parameter vector and available telemetry at cancellation.
#[derive(Debug, Clone, PartialEq)]
pub struct CancelledLeastSquaresSnapshot {
    /// Number of completed outer LM iterations.
    pub completed_iterations: usize,
    /// Last accepted parameter vector (never an unaccepted trial point).
    pub parameters: Vec<f64>,
    /// Residual norm at the last accepted point, when already evaluated.
    pub residual_norm: Option<f64>,
    /// Work counters, absent unless counter telemetry was requested.
    pub residual_evaluations: Option<usize>,
    pub jacobian_evaluations: Option<usize>,
    pub linear_solves: Option<usize>,
    /// Elapsed solve time, absent unless timing telemetry was requested.
    pub elapsed: Option<Duration>,
}

/// Stage at which a least-squares evaluation failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeastSquaresStage {
    Configuration,
    Parameters,
    Residual,
    Jacobian,
    TrustRegion,
}

/// Typed failures exposed by the rectangular least-squares API.
#[derive(Debug, Clone, PartialEq)]
pub enum LeastSquaresError {
    InvalidConfiguration {
        field: &'static str,
        value: f64,
    },
    InvalidLogLevel(String),
    WrongSolverKind,
    CallbackFailed {
        stage: LeastSquaresStage,
    },
    EmptyProblem {
        stage: LeastSquaresStage,
    },
    InvalidProblemShape {
        stage: LeastSquaresStage,
    },
    DomainViolation,
    MaxEvaluationsReached {
        limit: usize,
    },
    MaxIterationsReached {
        limit: usize,
    },
    DimensionMismatch {
        stage: LeastSquaresStage,
        expected: usize,
        actual: usize,
    },
    NonFiniteValue {
        stage: LeastSquaresStage,
        index: usize,
    },
    NumericalBreakdown {
        stage: LeastSquaresStage,
    },
    /// Cooperative cancellation observed at a safe LM checkpoint.
    Cancelled {
        /// Partial state and telemetry from the last accepted point.
        snapshot: Box<CancelledLeastSquaresSnapshot>,
    },
    Problem(SolveError),
}

impl Display for LeastSquaresError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfiguration { field, value } => {
                write!(f, "invalid least-squares configuration {field}={value}")
            }
            Self::InvalidLogLevel(value) => {
                write!(f, "unsupported least-squares log level '{value}'")
            }
            Self::WrongSolverKind => write!(f, "selected solver is not a least-squares method"),
            Self::CallbackFailed { stage } => {
                write!(f, "least-squares {stage:?} callback failed")
            }
            Self::EmptyProblem { stage } => write!(f, "least-squares {stage:?} is empty"),
            Self::InvalidProblemShape { stage } => {
                write!(f, "invalid least-squares problem shape during {stage:?}")
            }
            Self::DomainViolation => write!(f, "least-squares trial domain was exhausted"),
            Self::MaxEvaluationsReached { limit } => {
                write!(
                    f,
                    "least-squares maximum of {limit} residual evaluations reached"
                )
            }
            Self::MaxIterationsReached { limit } => {
                write!(f, "least-squares maximum of {limit} iterations reached")
            }
            Self::DimensionMismatch {
                stage,
                expected,
                actual,
            } => write!(
                f,
                "least-squares {stage:?} dimension mismatch: expected {expected}, got {actual}"
            ),
            Self::NonFiniteValue { stage, index } => {
                write!(f, "non-finite value at {stage:?} index {index}")
            }
            Self::NumericalBreakdown { stage } => {
                write!(f, "numerical breakdown during least-squares {stage:?}")
            }
            Self::Cancelled { snapshot } => write!(
                f,
                "least-squares solve cancelled after {} completed iteration(s)",
                snapshot.completed_iterations
            ),
            Self::Problem(source) => write!(f, "least-squares problem evaluation failed: {source}"),
        }
    }
}

impl Error for LeastSquaresError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Problem(source) => Some(source),
            _ => None,
        }
    }
}

impl From<SolveError> for LeastSquaresError {
    fn from(value: SolveError) -> Self {
        Self::Problem(value)
    }
}
