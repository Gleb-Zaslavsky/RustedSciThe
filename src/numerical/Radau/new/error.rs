//! Typed errors for preparation, callbacks, nonlinear failure, and limits.
//!
//! No public `try_*` path in the new API should rely on panic-based failure.

use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Identifies the solver stage that produced an operational error.
pub(crate) enum RadauStage {
    Preparation,
    Residual,
    Jacobian,
    Newton,
    LinearSolve,
    StepControl,
    Output,
}

/// Validation failures that can be reported before numerical work starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub(crate) enum RadauConfigError {
    #[error("time bounds must be finite and distinct")]
    InvalidTimeBounds,
    #[error("relative tolerance must be finite and positive")]
    InvalidRelativeTolerance,
    #[error("absolute tolerance must be finite and positive")]
    InvalidAbsoluteTolerance,
    #[error("maximum step must be positive or infinity")]
    InvalidMaximumStep,
    #[error("first step must be finite and positive")]
    InvalidFirstStep,
    #[error("step, Newton, and retry budgets must be non-zero")]
    EmptyIterationBudget,
    #[error("banded layout must have a non-zero bandwidth")]
    EmptyBandwidth,
    #[error("native execution cannot select a symbolic assembly")]
    NativeWithSymbolicAssembly,
    #[error("symbolic execution requires an assembly frontend")]
    SymbolicWithoutAssembly,
    #[error("state dimension must be non-zero")]
    ZeroDimension,
    #[error("Jacobian pattern entry is outside the state dimensions")]
    JacobianPatternOutOfBounds,
    #[error("sparse Jacobian layout requires a prepared pattern")]
    SparsePatternMissing,
    #[error("sparse pattern entry is outside the state dimensions")]
    SparsePatternOutOfBounds,
    #[error("sparse Jacobian entry is absent from the prepared pattern")]
    SparseEntryMissing,
    #[error("sparse workspace is missing its required diagonal")]
    SparseDiagonalMissing,
    #[error("banded Jacobian bandwidth does not match the prepared backend")]
    BandedBandwidthMismatch,
    #[error("banded compact storage has an invalid slot")]
    InvalidBandedSlot,
    #[error("symbolic callback plan and workspace frontend do not match")]
    CallbackFrontendMismatch,
    #[error("prepared linear backend and workspace variants do not match")]
    LinearWorkspaceMismatch,
    #[error("Lambdify expression contains a variable outside the callback schema")]
    ExpressionVariableOutsideSchema,
    #[error("shared AtomView preparation did not publish a native Jacobian")]
    NativeJacobianNotPublished,
    #[error("Radau step time and step size must be finite and non-zero")]
    InvalidStep,
}

/// Capability or lifecycle gaps that are intentionally reported instead of
/// being hidden behind a conversion or a fallback implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub(crate) enum RadauUnsupportedRoute {
    #[error("Radau AOT preparation is unavailable for the requested route")]
    AotPreparation,
    #[error("Radau AOT configuration is missing")]
    AotConfigurationMissing,
    #[error("Lambdify Jacobian callback was not prepared")]
    LambdifyJacobianNotPrepared,
    #[error("Banded layout excludes a nonzero Jacobian entry")]
    JacobianOutsideBandedLayout,
    #[error("dense backend requires dense Jacobian values")]
    DenseValuesRequired,
    #[error("banded backend requires compact banded Jacobian values")]
    BandedValuesRequired,
    #[error("sparse backend requires sparse Jacobian values")]
    SparseValuesRequired,
    #[error("structured Radau step requires its native linear backend")]
    StructuredStep,
    #[error("dense numerical step requires the dense linear workspace")]
    DenseWorkspaceRequired,
    #[error("structured numerical steps require a layout-aware Jacobian callback")]
    StructuredJacobianCallbackRequired,
    #[error("the initial Radau numerical core supports dense steps only")]
    DenseStepOnly,
    #[error("dense Radau step requires a dense matrix configuration")]
    DenseConfigurationRequired,
}

#[derive(Debug, Error)]
/// Complete typed error surface for preparation, callbacks, and stepping.
///
/// `try_*` APIs return this enum instead of panicking. The nested configuration
/// and capability enums make machine-readable classification possible without
/// parsing display strings.
pub(crate) enum RadauError {
    #[error("invalid Radau configuration: {0}")]
    InvalidConfiguration(#[from] RadauConfigError),
    #[error("callback failed during {stage:?}: {message}")]
    Callback { stage: RadauStage, message: String },
    #[error("non-finite callback output during {stage:?}")]
    NonFiniteCallback { stage: RadauStage },
    #[error("callback output length mismatch during {stage:?}: expected {expected}, got {actual}")]
    ShapeMismatch {
        stage: RadauStage,
        expected: usize,
        actual: usize,
    },
    #[error("Radau Newton iteration failed after {iterations} iterations")]
    NewtonFailure { iterations: usize },
    #[error("Radau step budget exhausted after {steps} steps")]
    StepBudgetExceeded { steps: usize },
    #[error("Radau step underflow at t={t} with h={h}")]
    StepUnderflow { t: f64, h: f64 },
    #[error("Radau step control exhausted after {retries} retries with error norm {error_norm}")]
    StepControlFailure { error_norm: f64, retries: usize },
    #[error("unsupported Radau route: {0}")]
    UnsupportedRoute(#[from] RadauUnsupportedRoute),
    #[error("Radau workspace dimensions overflow for state dimension {dimension}")]
    WorkspaceSizeOverflow { dimension: usize },
    #[error("Radau dense linear solve failed for dimension {dimension}")]
    LinearSolveFailure { dimension: usize },
    #[error("dense output is not available for this solution")]
    OutputNotAvailable,
    #[error("dense output time {t} is outside the stored integration interval")]
    OutputTimeOutsideInterval { t: f64 },
    #[error("dense output segment is invalid")]
    InvalidDenseOutput,
    #[error("Radau AOT lifecycle failed: {message}")]
    AotLifecycle { message: String },
}
