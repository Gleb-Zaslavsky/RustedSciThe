//! Typed failure taxonomy for the new BVP lifecycle.

use thiserror::Error;

/// Machine-readable lifecycle stage used by all currently implemented
/// preparation and callback failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BvpSciStage {
    SymbolicPreparation,
    ExprToAtom,
    SymbolicJacobian,
    PatternConstruction,
    EvaluatorCompilation,
    CallbackInput,
    ArgumentBuffer,
    ResidualCallback,
    JacobianCallback,
    JacobianOutputAssembly,
    BoundaryCallback,
    NumericalCore,
    LinearBackend,
    Continuation,
    Output,
    AotPreparation,
    AotCacheLookup,
    AotBuild,
    AotLink,
    AotPublication,
    AotRuntime,
    AotCallback,
}

/// Machine-readable cause for an AOT preparation failure.
///
/// The human-readable compiler diagnostic remains in the error, but callers
/// should branch on this value instead of parsing that diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BvpSciAotFailureKind {
    SymbolicPreparation,
    MissingArtifact,
    StaleArtifact,
    /// The toolchain process rejected generated source before linking.
    Compiler,
    Build,
    Link,
    Timeout,
    Publication,
    Runtime,
    Unknown,
}

/// SciPy-compatible termination classification.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BvpSciStatus {
    Success = 0,
    MaxNodes = 1,
    SingularJacobian = 2,
    BoundaryTolerance = 3,
}

impl BvpSciStatus {
    pub const fn code(self) -> u8 {
        self as u8
    }
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum BvpSciNewError {
    #[error("invalid BVP configuration: {0}")]
    InvalidConfiguration(String),
    #[error("callback failed during {stage:?}: {message}")]
    Callback { stage: BvpSciStage, message: String },
    #[error(
        "callback output has invalid shape during {stage:?}: expected {expected}, got {actual}"
    )]
    ShapeMismatch {
        stage: BvpSciStage,
        expected: usize,
        actual: usize,
    },
    #[error("callback produced a non-finite value during {stage:?}")]
    NonFinite { stage: BvpSciStage },
    #[error("Newton iteration failed after {iterations} iterations")]
    NewtonFailure { iterations: usize },
    #[error("collocation Jacobian is singular")]
    SingularJacobian,
    #[error("boundary residual did not satisfy tolerance {tolerance:e}")]
    BoundaryToleranceNotMet { tolerance: f64 },
    #[error("mesh refinement exhausted the node budget of {max_nodes}")]
    MaxNodesExceeded { max_nodes: usize },
    #[error("mesh refinement exhausted the refinement budget of {refinements}")]
    MeshRefinementLimit { refinements: usize },
    #[error("linear backend failure: {message}")]
    LinearBackend { message: String },
    #[error("linear backend factorization failed: {message}")]
    LinearFactorization { message: String },
    #[error("linear backend solve failed: {message}")]
    LinearSolve { message: String },
    #[error("unsupported BVP route: {0}")]
    UnsupportedRoute(String),
    #[error("symbolic preparation failed: {0}")]
    SymbolicPreparation(String),
    #[error("AOT preparation failed during {stage:?} ({kind:?}): {message}")]
    AotPreparation {
        stage: BvpSciStage,
        kind: BvpSciAotFailureKind,
        message: String,
    },
    #[error("AOT process handoff failed during {stage:?}: {message}")]
    AotHandoff { stage: BvpSciStage, message: String },
    #[error("AOT runtime failed during {stage:?}: {message}")]
    AotRuntime { stage: BvpSciStage, message: String },
    #[error("AOT callback failed during {stage:?}: {message}")]
    AotCallback { stage: BvpSciStage, message: String },
    #[error("prepared model is incompatible with the requested continuation state")]
    ContinuationMismatch,
    #[error("dense output query is outside the solution domain")]
    OutputOutOfDomain,
    #[error("dense output is unavailable under the selected output policy")]
    OutputUnavailable,
}

impl BvpSciNewError {
    /// Return the corresponding SciPy status for a numerical termination.
    pub const fn status(&self) -> Option<BvpSciStatus> {
        match self {
            Self::MaxNodesExceeded { .. } | Self::MeshRefinementLimit { .. } => {
                Some(BvpSciStatus::MaxNodes)
            }
            Self::SingularJacobian => Some(BvpSciStatus::SingularJacobian),
            Self::BoundaryToleranceNotMet { .. } => Some(BvpSciStatus::BoundaryTolerance),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BvpSciAotFailureKind, BvpSciNewError, BvpSciStage, BvpSciStatus};

    #[test]
    fn aot_failure_taxonomy_keeps_compiler_and_lifecycle_classes_distinct() {
        let kinds = [
            BvpSciAotFailureKind::MissingArtifact,
            BvpSciAotFailureKind::StaleArtifact,
            BvpSciAotFailureKind::Compiler,
            BvpSciAotFailureKind::Build,
            BvpSciAotFailureKind::Link,
            BvpSciAotFailureKind::Timeout,
            BvpSciAotFailureKind::Publication,
            BvpSciAotFailureKind::Runtime,
        ];
        for (left, kind) in kinds.iter().enumerate() {
            for (right, candidate) in kinds.iter().enumerate() {
                assert_eq!(left == right, kind == candidate);
            }
        }
    }

    #[test]
    fn scipy_status_codes_and_error_mapping_are_exact() {
        assert_eq!(BvpSciStatus::Success.code(), 0);
        assert_eq!(BvpSciStatus::MaxNodes.code(), 1);
        assert_eq!(BvpSciStatus::SingularJacobian.code(), 2);
        assert_eq!(BvpSciStatus::BoundaryTolerance.code(), 3);

        assert_eq!(
            BvpSciNewError::MaxNodesExceeded { max_nodes: 8 }.status(),
            Some(BvpSciStatus::MaxNodes)
        );
        assert_eq!(
            BvpSciNewError::MeshRefinementLimit { refinements: 2 }.status(),
            Some(BvpSciStatus::MaxNodes)
        );
        assert_eq!(
            BvpSciNewError::SingularJacobian.status(),
            Some(BvpSciStatus::SingularJacobian)
        );
        assert_eq!(
            BvpSciNewError::BoundaryToleranceNotMet { tolerance: 1e-6 }.status(),
            Some(BvpSciStatus::BoundaryTolerance)
        );
        assert_eq!(
            BvpSciNewError::NonFinite {
                stage: BvpSciStage::NumericalCore
            }
            .status(),
            None
        );
    }
}
