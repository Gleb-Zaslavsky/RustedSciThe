use crate::numerical::Nonlinear_systems::error::SolveError;
use std::error::Error;
use std::fmt::{Display, Formatter};

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
