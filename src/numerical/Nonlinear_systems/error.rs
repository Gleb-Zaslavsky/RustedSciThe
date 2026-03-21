use std::error::Error;
use std::fmt::{Display, Formatter};

/// Describes why the nonlinear solve stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TerminationReason {
    /// The residual or method-specific convergence check succeeded.
    Converged,
    /// The maximum number of outer iterations was reached.
    MaxIterations,
    /// The trial step became too small to make progress.
    StepTooSmall,
    /// The method stopped because progress stalled.
    Stagnation,
    /// Too many trial steps were rejected.
    RejectedStepLimit,
}

/// Errors returned by the nonlinear solver pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum SolveError {
    /// A vector or matrix has an unexpected dimension.
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
        /// Short description of the failing context.
        context: &'static str,
    },
    /// Solver configuration is invalid.
    InvalidConfig(String),
    /// Initial point violates the provided bounds.
    InfeasibleInitialGuess {
        /// Index of the offending variable.
        index: usize,
        /// Current variable value.
        value: f64,
        /// Lower bound.
        lower: f64,
        /// Upper bound.
        upper: f64,
    },
    /// Residual evaluation failed.
    ResidualEvaluation(String),
    /// Jacobian evaluation failed.
    JacobianEvaluation(String),
    /// Linear system solve failed.
    LinearSolveFailure(String),
    /// Jacobian is singular or too ill-conditioned.
    SingularJacobian,
    /// Numerical breakdown such as NaN or Inf.
    NumericalBreakdown(String),
}

impl Display for SolveError {
    /// Formats a human-readable error message.
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            SolveError::DimensionMismatch {
                expected,
                actual,
                context,
            } => {
                write!(
                    f,
                    "dimension mismatch in {context}: expected {expected}, got {actual}"
                )
            }
            SolveError::InvalidConfig(msg) => write!(f, "invalid solver configuration: {msg}"),
            SolveError::InfeasibleInitialGuess {
                index,
                value,
                lower,
                upper,
            } => write!(
                f,
                "initial guess at index {index} = {value} violates bounds [{lower}, {upper}]"
            ),
            SolveError::ResidualEvaluation(msg) => write!(f, "residual evaluation failed: {msg}"),
            SolveError::JacobianEvaluation(msg) => write!(f, "jacobian evaluation failed: {msg}"),
            SolveError::LinearSolveFailure(msg) => write!(f, "linear solve failed: {msg}"),
            SolveError::SingularJacobian => write!(f, "jacobian is singular or ill-conditioned"),
            SolveError::NumericalBreakdown(msg) => write!(f, "numerical breakdown: {msg}"),
        }
    }
}

impl Error for SolveError {}
