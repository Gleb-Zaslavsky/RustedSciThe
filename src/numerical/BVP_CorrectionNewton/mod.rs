//! Standalone fixed-mesh damped correction-Newton solver.
//!
//! The numerical iteration is isolated from [`super::BVP_Damp`]. The current
//! numerical core accepts reusable coordinate callbacks with Dense, Sparse or
//! automatically inferred Banded linear workspaces. Symbolic [`Expr`]
//! preparation supports ExprLegacy/AtomView and Lambdify/AOT execution, and
//! prepares a fixed trapezoid coordinate pattern without constructing another
//! solver object. Adaptive mesh refinement and pseudo-transient recovery are
//! intentionally outside this fixed-mesh route.
//!
//! The implementation follows a pinned damped correction-Newton core contract:
//! weighted RMS correction norm, fraction-to-boundary restriction, Jacobian
//! age/reuse, and a fixed `sqrt(2)` damping reduction. Those choices are kept
//! local here and are checked against independent transition-contract tests.

use crate::numerical::BVP_sci::new::BvpSciNewError;
use crate::numerical::logging::{
    SolverLogGuard, SolverLogLevel, SolverLoggingError, solver_debug, solver_info, solver_warn,
};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp_generated::SymbolicIvpGeneratedBackendConfig;
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::time::{Duration, Instant};
use thiserror::Error;

mod linear;
use linear::CorrectionNewtonLinearWorkspace;
pub use linear::{
    CorrectionNewtonLinearBackend, CorrectionNewtonLinearError, CorrectionNewtonMatrixLayout,
};
mod jacobian;
pub use jacobian::{CorrectionNewtonJacobianError, CorrectionNewtonJacobianPattern};
mod parameter_loop;
mod symbolic_plan;
pub use parameter_loop::{
    CorrectionNewtonParameterLoopOutcome, CorrectionNewtonParameterLoopPoint,
    CorrectionNewtonParameterLoopReport,
};

#[cfg(test)]
#[path = "tests/mod.rs"]
pub(crate) mod tests;

/// Residual callback for one packed, fixed-mesh state vector.
pub type ResidualCallback =
    dyn FnMut(&DVector<f64>) -> Result<DVector<f64>, CorrectionNewtonCallbackError>;
/// Dense Jacobian callback for one packed, fixed-mesh state vector.
pub type JacobianCallback =
    dyn FnMut(&DVector<f64>) -> Result<DMatrix<f64>, CorrectionNewtonCallbackError>;
/// Coordinate Jacobian callback that fills the prepared pattern's value buffer.
pub type CoordinateJacobianCallback =
    dyn FnMut(&DVector<f64>, &mut [f64]) -> Result<(), CorrectionNewtonCallbackError>;

enum JacobianSource {
    Dense(
        Box<
            dyn FnMut(
                &DVector<f64>,
                &[f64],
            ) -> Result<DMatrix<f64>, CorrectionNewtonCallbackFailure>,
        >,
    ),
    Coordinates {
        pattern: CorrectionNewtonJacobianPattern,
        values: Vec<f64>,
        callback: Box<
            dyn FnMut(
                &DVector<f64>,
                &[f64],
                &mut [f64],
            ) -> Result<(), CorrectionNewtonCallbackFailure>,
        >,
    },
    SymbolicCoordinates {
        pattern: CorrectionNewtonJacobianPattern,
        values: Vec<f64>,
        callback: Box<
            dyn FnMut(
                &DVector<f64>,
                &[f64],
                &mut [f64],
            ) -> Result<(), CorrectionNewtonCallbackFailure>,
        >,
    },
}

enum ResidualSource {
    Allocating(
        Box<
            dyn FnMut(
                &DVector<f64>,
                &[f64],
            ) -> Result<DVector<f64>, CorrectionNewtonCallbackFailure>,
        >,
    ),
    Buffered(
        Box<
            dyn FnMut(
                &DVector<f64>,
                &[f64],
                &mut DVector<f64>,
            ) -> Result<(), CorrectionNewtonCallbackFailure>,
        >,
    ),
}

#[derive(Debug)]
enum JacobianSourceError {
    Callback(CorrectionNewtonCallbackFailure),
    Assembly(CorrectionNewtonJacobianError),
}

/// Classification for a callback failure returned to the Newton core.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CorrectionNewtonCallbackErrorKind {
    /// An external callback failed without a narrower classification.
    External,
    /// The caller explicitly identified an invalid model-domain trial.
    Domain,
    /// A callback returned an output with a shape inconsistent with its contract.
    ShapeMismatch,
    /// A callback returned NaN or infinity.
    NonFinite,
    /// A symbolic frontend returned a typed error.
    Symbolic,
}

/// Shape of a callback-produced output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CorrectionNewtonCallbackShape {
    Vector { length: usize },
    Matrix { rows: usize, columns: usize },
}

impl std::fmt::Display for CorrectionNewtonCallbackShape {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Vector { length } => write!(formatter, "vector(length={length})"),
            Self::Matrix { rows, columns } => write!(formatter, "matrix({rows}x{columns})"),
        }
    }
}

/// Location in a callback output that failed finite-value validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CorrectionNewtonCallbackLocation {
    Residual { index: usize },
    Jacobian { row: usize, column: usize },
}

impl std::fmt::Display for CorrectionNewtonCallbackLocation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Residual { index } => write!(formatter, "residual[{index}]"),
            Self::Jacobian { row, column } => write!(formatter, "Jacobian[{row},{column}]"),
        }
    }
}

/// Typed callback error. Domain failures at a Newton damping trial are
/// recoverable by trying a smaller step; failures at the accepted iterate are
/// returned. Other callback failures are never treated as recoverable.
#[derive(Debug, Error)]
pub enum CorrectionNewtonCallbackError {
    /// An external callback failed for a reason not classified more narrowly.
    #[error("external callback failed: {0}")]
    External(#[source] Box<dyn std::error::Error + Send + Sync>),
    /// The trial input is outside the callback's model domain.
    #[error("callback trial is outside the model domain: {0}")]
    Domain(#[source] Box<dyn std::error::Error + Send + Sync>),
    /// The callback output does not match the required dimensions.
    #[error("callback output shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch {
        expected: CorrectionNewtonCallbackShape,
        actual: CorrectionNewtonCallbackShape,
    },
    /// The callback output contains a non-finite value.
    #[error("callback output contains a non-finite value at {location}")]
    NonFinite {
        location: CorrectionNewtonCallbackLocation,
    },
    /// Preserve the typed symbolic error and its source chain.
    #[error("symbolic callback failed: {0}")]
    Symbolic(#[source] BvpSciNewError),
}

pub(super) type CorrectionNewtonCallbackFailure = CorrectionNewtonCallbackError;

impl CorrectionNewtonCallbackError {
    /// Construct a model-domain error while retaining its concrete source.
    pub fn domain<E>(source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Domain(Box::new(source))
    }

    /// Construct an unclassified callback error while retaining its source.
    pub fn external<E>(source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::External(Box::new(source))
    }

    /// Return the stable high-level classification for this callback failure.
    pub fn kind(&self) -> CorrectionNewtonCallbackErrorKind {
        match self {
            Self::External(_) => CorrectionNewtonCallbackErrorKind::External,
            Self::Domain(_) => CorrectionNewtonCallbackErrorKind::Domain,
            Self::ShapeMismatch { .. } => CorrectionNewtonCallbackErrorKind::ShapeMismatch,
            Self::NonFinite { .. } => CorrectionNewtonCallbackErrorKind::NonFinite,
            Self::Symbolic(BvpSciNewError::ShapeMismatch { .. }) => {
                CorrectionNewtonCallbackErrorKind::ShapeMismatch
            }
            Self::Symbolic(BvpSciNewError::NonFinite { .. }) => {
                CorrectionNewtonCallbackErrorKind::NonFinite
            }
            Self::Symbolic(_) => CorrectionNewtonCallbackErrorKind::Symbolic,
        }
    }
}

/// Configuration and input-contract failures detected before or during setup.
#[derive(Debug, Error)]
pub enum CorrectionNewtonConfigurationError {
    #[error(
        "symbolic BVP dimensions, interval, tolerances, bounds, or iteration limits are invalid"
    )]
    InvalidSymbolicProblem,
    #[error("fixed boundary value for state {state} is outside its bounds [{lower}, {upper}]")]
    FixedBoundaryOutsideBounds {
        state: String,
        value: f64,
        lower: f64,
        upper: f64,
    },
    #[error("AOT execution requires a generated-backend configuration")]
    MissingAotConfiguration,
    #[error("trapezoid Jacobian contribution count overflowed usize")]
    JacobianContributionCountOverflow,
    #[error("failed to construct the trapezoid Jacobian pattern")]
    InvalidJacobianPattern(#[source] CorrectionNewtonJacobianError),
    #[error(
        "Jacobian pattern dimension {pattern_dimension} does not match state size {state_size}"
    )]
    JacobianPatternDimensionMismatch {
        pattern_dimension: usize,
        state_size: usize,
    },
    #[error("Newton state, tolerances, bounds, or iteration limits are inconsistent")]
    InvalidNewtonConfiguration,
    #[error("linear workspace initialization failed")]
    LinearWorkspace(#[source] CorrectionNewtonLinearError),
    #[error("external parameter names must be non-empty and unique")]
    InvalidParameterSchema,
    #[error("parameter schema and finite values differ in length: names={names}, values={values}")]
    ParameterBindingLengthMismatch { names: usize, values: usize },
    #[error("bound parameter names do not match the prepared schema")]
    ParameterSchemaMismatch,
    #[error("expected {expected} finite external parameter values, received {actual}")]
    ParameterValueCountMismatch { expected: usize, actual: usize },
    #[error("expanded CorrectionNewton state size overflowed")]
    ExpandedStateSizeOverflow,
    #[error(
        "expanded CorrectionNewton state size {state_size} is incompatible with component count {components}"
    )]
    ExpandedStateComponentMismatch {
        state_size: usize,
        components: usize,
    },
    #[error("CorrectionNewton state grid mapping is incomplete")]
    IncompleteGridMapping,
    #[error("grid mapping length {mapped} does not match reduced state length {reduced}")]
    GridMappingLengthMismatch { mapped: usize, reduced: usize },
    #[error("grid position {position} is outside expanded state size {state_size}")]
    GridPositionOutOfBounds { position: usize, state_size: usize },
    #[error("fixed boundary position {position} is outside expanded state size {state_size}")]
    BoundaryPositionOutOfBounds { position: usize, state_size: usize },
    #[error("expanded grid contains a non-finite or unmapped value at index {index}")]
    InvalidExpandedGridValue { index: usize },
    #[error("BVP full-grid point count overflowed usize")]
    FullGridPointCountOverflow,
    #[error("boundary condition refers to unknown state {state}")]
    UnknownBoundaryState { state: String },
    #[error("first-order state {state} requires exactly one endpoint condition; got {actual}")]
    EndpointConditionCountMismatch { state: String, actual: usize },
    #[error("state {state} has invalid endpoint condition side={side}, value={value}")]
    InvalidEndpointCondition {
        state: String,
        side: usize,
        value: f64,
    },
    #[error("state {state} is missing its endpoint condition")]
    MissingEndpointCondition { state: String },
    #[error("boundary-condition schema does not correspond one-to-one with states")]
    BoundarySchemaMismatch,
    #[error(
        "boundary value at slot {slot} ({value}) is outside its state bounds [{lower}, {upper}]"
    )]
    BoundaryValueOutsideBounds {
        slot: usize,
        value: f64,
        lower: f64,
        upper: f64,
    },
    #[error("BVP full-grid state size overflowed usize")]
    FullGridStateSizeOverflow,
    #[error("free mesh node {node} for state {state} cannot be mapped into the initial guess")]
    InitialGuessMappingFailure { node: usize, state: String },
    #[error("reduced BVP state has {free_values} values but the initial guess has {guess_values}")]
    InitialGuessLengthMismatch {
        free_values: usize,
        guess_values: usize,
    },
    #[error("parameter loop requires a nonempty sequence and fixed parameter schema")]
    InvalidParameterLoopInput,
    #[error("parameter loop requires a symbolic BVP retaining its physical mesh")]
    ParameterLoopRequiresSymbolicMesh,
}

/// Internal invariant failure where a prepared frontend omitted required data.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum CorrectionNewtonFrontendContractError {
    #[error("parameter-loop mesh and profile row counts differ")]
    ParameterLoopMeshProfileMismatch,
}

enum EvaluatedJacobian<'a> {
    Dense(DMatrix<f64>),
    Coordinates { values: &'a [f64] },
}

fn map_linear_error(
    error: CorrectionNewtonLinearError,
    report: CorrectionNewtonSolveReport,
) -> CorrectionNewtonSolverError {
    match error {
        CorrectionNewtonLinearError::Singular { backend } => {
            CorrectionNewtonSolverError::SingularJacobian { backend, report }
        }
        CorrectionNewtonLinearError::ResidualCheckFailed {
            backend,
            residual_norm,
            tolerance,
        } => CorrectionNewtonSolverError::LinearResidualCheckFailed {
            backend,
            residual_norm,
            tolerance,
            report,
        },
        CorrectionNewtonLinearError::NonFiniteSolution { backend } => {
            CorrectionNewtonSolverError::NonFiniteLinearSolution { backend, report }
        }
        source => CorrectionNewtonSolverError::LinearBackend {
            backend: source.backend_name(),
            source,
            report,
        },
    }
}

impl JacobianSource {
    fn pattern(&self) -> Option<&CorrectionNewtonJacobianPattern> {
        match self {
            Self::Dense(_) => None,
            Self::Coordinates { pattern, .. } | Self::SymbolicCoordinates { pattern, .. } => {
                Some(pattern)
            }
        }
    }

    fn evaluate(
        &mut self,
        state: &DVector<f64>,
        external_parameters: &[f64],
    ) -> Result<EvaluatedJacobian<'_>, JacobianSourceError> {
        match self {
            Self::Dense(callback) => {
                let matrix =
                    callback(state, external_parameters).map_err(JacobianSourceError::Callback)?;
                if matrix.shape() != (state.len(), state.len()) {
                    return Err(JacobianSourceError::Callback(
                        CorrectionNewtonCallbackFailure::ShapeMismatch {
                            expected: CorrectionNewtonCallbackShape::Matrix {
                                rows: state.len(),
                                columns: state.len(),
                            },
                            actual: CorrectionNewtonCallbackShape::Matrix {
                                rows: matrix.nrows(),
                                columns: matrix.ncols(),
                            },
                        },
                    ));
                }
                if let Some((index, _)) = matrix
                    .iter()
                    .enumerate()
                    .find(|(_, value)| !value.is_finite())
                {
                    let row = index % matrix.nrows().max(1);
                    let col = index / matrix.nrows().max(1);
                    return Err(JacobianSourceError::Callback(
                        CorrectionNewtonCallbackFailure::NonFinite {
                            location: CorrectionNewtonCallbackLocation::Jacobian {
                                row,
                                column: col,
                            },
                        },
                    ));
                }
                Ok(EvaluatedJacobian::Dense(matrix))
            }
            Self::Coordinates {
                pattern,
                values,
                callback,
            } => {
                callback(state, external_parameters, values)
                    .map_err(JacobianSourceError::Callback)?;
                pattern
                    .validate_values(values)
                    .map_err(JacobianSourceError::Assembly)?;
                Ok(EvaluatedJacobian::Coordinates { values })
            }
            Self::SymbolicCoordinates {
                pattern,
                values,
                callback,
            } => {
                callback(state, external_parameters, values)
                    .map_err(JacobianSourceError::Callback)?;
                pattern
                    .validate_values(values)
                    .map_err(JacobianSourceError::Assembly)?;
                Ok(EvaluatedJacobian::Coordinates { values })
            }
        }
    }
}

/// Symbolic fixed-mesh BVP input for the CorrectionNewton Newton route.
#[derive(Clone, Debug)]
pub struct CorrectionNewtonProblem {
    /// One symbolic RHS equation per state.
    pub equations: Vec<Expr>,
    /// One initial guess per free mesh value and component. Each component's
    /// values are aligned to its free nodes, with the left endpoint omitted
    /// for a left boundary condition and the right endpoint omitted for a
    /// right boundary condition.
    pub initial_guess: DMatrix<f64>,
    /// State names in equation order.
    pub state_names: Vec<String>,
    /// Independent variable name used by the symbolic equations.
    pub independent_variable: String,
    /// Endpoint conditions keyed by state name; index `0` is left, `1` is right.
    pub boundary_conditions: HashMap<String, Vec<(usize, f64)>>,
    /// Left and right endpoint coordinates.
    pub interval: (f64, f64),
    /// Number of unknown mesh columns supplied in `initial_guess`.
    pub steps: usize,
}

/// Symbolic frontend used to prepare fixed-mesh callbacks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CorrectionNewtonSymbolicFrontend {
    /// Established Expr-based derivative and Lambdify route.
    #[default]
    ExprLegacy,
    /// Native packed-Atom derivative and Lambdify route.
    AtomView,
}

/// Symbolic callback implementation selected during preparation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CorrectionNewtonSymbolicExecution {
    /// Compile reusable Lambdify callbacks during solver preparation.
    #[default]
    Lambdify,
    /// Build or load the shared generated native callback artifact.
    Aot,
}

/// Controls for the selected symbolic pilot route and its Newton core.
#[derive(Clone, Debug)]
pub struct CorrectionNewtonOptions {
    /// Symbolic representation used to prepare RHS and Jacobian callbacks.
    pub symbolic_frontend: CorrectionNewtonSymbolicFrontend,
    /// Callback implementation. AOT requires an explicit generated-backend
    /// configuration supplied through [`Self::with_aot_config`].
    pub symbolic_execution: CorrectionNewtonSymbolicExecution,
    /// Shared AOT compiler/cache lifecycle configuration, when AOT is selected.
    pub aot_config: Option<SymbolicIvpGeneratedBackendConfig>,
    /// Linear storage selected for the symbolic fixed-mesh Jacobian.
    pub matrix_layout: CorrectionNewtonMatrixLayout,
    /// Scalar absolute tolerance for the weighted RMS correction.
    pub absolute_tolerance: f64,
    /// Per-state relative tolerances. Omitted states use
    /// [`Self::default_relative_tolerance`].
    pub relative_tolerances: HashMap<String, f64>,
    /// Positive relative tolerance used for omitted states.
    pub default_relative_tolerance: f64,
    /// Per-state inclusive bounds. Omitted states use wide finite bounds.
    pub bounds: HashMap<String, (f64, f64)>,
    /// Maximum Newton iterations.
    pub max_iterations: usize,
    /// Maximum age of a reused Jacobian.
    pub max_jacobian_age: usize,
    /// Maximum damping trials per iteration.
    pub max_damping_trials: usize,
    /// Optional solver logging level.
    pub logging_level: SolverLogLevel,
    /// Optional core telemetry collection mode.
    pub telemetry_mode: CorrectionNewtonTelemetryMode,
}

impl Default for CorrectionNewtonOptions {
    fn default() -> Self {
        Self {
            symbolic_frontend: CorrectionNewtonSymbolicFrontend::ExprLegacy,
            symbolic_execution: CorrectionNewtonSymbolicExecution::Lambdify,
            aot_config: None,
            matrix_layout: CorrectionNewtonMatrixLayout::Dense,
            absolute_tolerance: 1e-8,
            relative_tolerances: HashMap::new(),
            default_relative_tolerance: 1e-8,
            bounds: HashMap::new(),
            max_iterations: 100,
            max_jacobian_age: 5,
            max_damping_trials: 7,
            logging_level: SolverLogLevel::Off,
            telemetry_mode: CorrectionNewtonTelemetryMode::Off,
        }
    }
}

impl CorrectionNewtonOptions {
    /// Select generated callbacks with the crate-wide AOT build/cache policy.
    pub fn with_aot_config(mut self, config: SymbolicIvpGeneratedBackendConfig) -> Self {
        self.symbolic_execution = CorrectionNewtonSymbolicExecution::Aot;
        self.aot_config = Some(config);
        self
    }
}

/// Detail level for standalone CorrectionNewton-core measurements.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CorrectionNewtonTelemetryMode {
    /// Do not collect counters or durations.
    #[default]
    Off,
    /// Collect operation counters without clocks.
    Counters,
    /// Collect counters and cumulative callback/linear-solve durations.
    Timings,
}

/// Configuration for a fixed-mesh damped correction-Newton solve.
#[derive(Clone, Debug)]
pub struct CorrectionNewtonConfig {
    /// Linear storage; Sparse/BandedAuto require a coordinate Jacobian callback.
    pub matrix_layout: CorrectionNewtonMatrixLayout,
    /// Number of interleaved physical components in the node-major state.
    ///
    /// CorrectionNewton scales each component using its average magnitude over all mesh
    /// points; non-grid systems can keep the default one-component-per-entry.
    pub component_count: usize,
    /// Per-component absolute tolerance used by the weighted correction norm.
    pub absolute_tolerances: DVector<f64>,
    /// Per-component relative tolerance used by the weighted correction norm.
    pub relative_tolerances: DVector<f64>,
    /// Inclusive lower and upper bounds for each packed unknown.
    pub bounds: Vec<(f64, f64)>,
    /// Node-major full-grid position for each packed free unknown. Empty means
    /// the generic callback system is already a complete packed grid.
    pub full_state_positions: Vec<usize>,
    /// Prescribed endpoint values as `(full_grid_position, value)` pairs.
    pub fixed_state_values: Vec<(usize, f64)>,
    /// Maximum accepted Newton iterations.
    pub max_iterations: usize,
    /// Maximum age of a reused Jacobian before refresh.
    pub max_jacobian_age: usize,
    /// Maximum damping trials for one Newton iteration.
    pub max_damping_trials: usize,
    /// Opt-in solver logging level; defaults to silent.
    pub logging_level: SolverLogLevel,
    /// Opt-in telemetry mode; defaults to off.
    pub telemetry_mode: CorrectionNewtonTelemetryMode,
}

impl CorrectionNewtonConfig {
    /// Creates scalar tolerances and unbounded unknowns for a state of `size`.
    pub fn new(size: usize, absolute_tolerance: f64, relative_tolerance: f64) -> Self {
        Self {
            matrix_layout: CorrectionNewtonMatrixLayout::Dense,
            component_count: size,
            absolute_tolerances: DVector::from_element(size, absolute_tolerance),
            relative_tolerances: DVector::from_element(size, relative_tolerance),
            bounds: vec![(f64::NEG_INFINITY, f64::INFINITY); size],
            full_state_positions: Vec::new(),
            fixed_state_values: Vec::new(),
            max_iterations: 100,
            max_jacobian_age: 5,
            max_damping_trials: 7,
            logging_level: SolverLogLevel::Off,
            telemetry_mode: CorrectionNewtonTelemetryMode::Off,
        }
    }
}

/// Counters and optional durations for one solve attempt, including failures.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CorrectionNewtonTelemetrySnapshot {
    /// Number of residual callback invocations.
    pub residual_evaluations: usize,
    /// Number of Jacobian callback invocations.
    pub jacobian_evaluations: usize,
    /// Number of linear solves, including rejected damping trials.
    pub linear_solves: usize,
    /// Number of Jacobian factorization attempts, including failed attempts.
    pub factorizations: usize,
    /// Number of trial states evaluated by the damping loop.
    pub damping_trials: usize,
    /// Number of rejected damping trials.
    pub damping_rejections: usize,
    /// Damping trials rejected because their residual callback was outside the
    /// model domain. This is a subset of `damping_rejections`.
    pub domain_rejections: usize,
    /// Number of accepted Newton updates.
    pub accepted_steps: usize,
    /// Number of Newton steps whose full correction was restricted by bounds.
    pub bound_limited_steps: usize,
    /// Accumulated callback time, populated only in `Timings` mode.
    pub callback_time: Duration,
    /// Residual-only callback time, populated only in `Timings` mode.
    pub residual_callback_time: Duration,
    /// Jacobian-only callback time, populated only in `Timings` mode.
    pub jacobian_callback_time: Duration,
    /// Structural coordinate scatter/native matrix assembly time.
    pub jacobian_assembly_time: Duration,
    /// Accumulated factorization time, populated only in `Timings` mode.
    pub factorization_time: Duration,
    /// Accumulated dense solve time, populated only in `Timings` mode.
    pub linear_solve_time: Duration,
}

/// Result of a converged solve or an unsuccessful attempt.
#[derive(Clone, Debug, PartialEq)]
pub struct CorrectionNewtonSolveReport {
    /// Last accepted packed state; a failed report is diagnostic, not a success result.
    pub state: DVector<f64>,
    /// Effective linear backend selected for this solve.
    pub linear_backend: CorrectionNewtonLinearBackend,
    /// Number of Newton iterations attempted.
    pub iterations: usize,
    /// Weighted RMS norm of the last Newton correction, when available.
    pub correction_norm: Option<f64>,
    /// Infinity norm of the residual at the last accepted state, when available.
    pub residual_norm: Option<f64>,
    /// Last fraction-to-boundary factor computed for a Newton correction.
    pub last_bound_factor: Option<f64>,
    /// Last damping coefficient tried, including the bound restriction.
    pub last_damping_factor: Option<f64>,
    /// Partial telemetry for this attempt.
    pub telemetry: CorrectionNewtonTelemetrySnapshot,
}

/// Typed failure with a partial report for the attempted solve.
#[derive(Debug, Error)]
pub enum CorrectionNewtonSolverError {
    /// Invalid dimensions or non-finite/invalid solver configuration.
    #[error("invalid CorrectionNewton BVP configuration: {source}")]
    InvalidConfiguration {
        #[source]
        source: CorrectionNewtonConfigurationError,
    },
    /// Symbolic equations declare fixed inputs that have not been bound yet.
    #[error("CorrectionNewton BVP external parameters have not been bound: {names:?}")]
    MissingParameterValues { names: Vec<String> },
    /// The selected symbolic frontend failed during preparation.
    #[error("CorrectionNewton BVP symbolic preparation failed: {source}")]
    SymbolicPreparation {
        #[source]
        source: BvpSciNewError,
    },
    /// The frontend returned successfully but did not provide its required callback.
    #[error("CorrectionNewton BVP symbolic frontend contract failed: {source}")]
    FrontendContract {
        #[source]
        source: CorrectionNewtonFrontendContractError,
    },
    /// The shared logging infrastructure could not be entered.
    #[error("CorrectionNewton BVP logging setup failed: {source}")]
    Logging {
        #[source]
        source: SolverLoggingError,
    },
    /// Residual callback failed; `report` retains completed work.
    #[error("CorrectionNewton BVP residual callback failed: {source}")]
    ResidualCallback {
        kind: CorrectionNewtonCallbackErrorKind,
        #[source]
        source: CorrectionNewtonCallbackError,
        report: CorrectionNewtonSolveReport,
    },
    /// Every damping trial left the model domain, so no trial residual could
    /// be evaluated. The accepted state is retained in the partial report.
    #[error("CorrectionNewton BVP exhausted damping trials outside the model domain: {source}")]
    DampingDomainExhausted {
        #[source]
        source: CorrectionNewtonCallbackError,
        report: CorrectionNewtonSolveReport,
    },
    /// Jacobian callback failed; `report` retains completed work.
    #[error("CorrectionNewton BVP Jacobian callback failed: {source}")]
    JacobianCallback {
        kind: CorrectionNewtonCallbackErrorKind,
        #[source]
        source: CorrectionNewtonCallbackError,
        report: CorrectionNewtonSolveReport,
    },
    /// A coordinate callback produced values invalid for its fixed pattern.
    #[error("CorrectionNewton BVP Jacobian assembly failed: {source}")]
    JacobianAssembly {
        #[source]
        source: CorrectionNewtonJacobianError,
        report: CorrectionNewtonSolveReport,
    },
    /// The selected backend found a singular Jacobian at the current iterate.
    #[error("CorrectionNewton BVP {backend} Jacobian is singular")]
    SingularJacobian {
        backend: &'static str,
        report: CorrectionNewtonSolveReport,
    },
    /// A backend returned successfully, but its solution failed backward-error validation.
    #[error(
        "CorrectionNewton BVP {backend} linear solve residual {residual_norm:e} exceeds allowed {tolerance:e}"
    )]
    LinearResidualCheckFailed {
        backend: &'static str,
        residual_norm: f64,
        tolerance: f64,
        report: CorrectionNewtonSolveReport,
    },
    /// A backend returned a solution containing NaN or infinity.
    #[error("CorrectionNewton BVP {backend} linear solve produced a non-finite solution")]
    NonFiniteLinearSolution {
        backend: &'static str,
        report: CorrectionNewtonSolveReport,
    },
    /// The configured linear backend failed during factorization or solve.
    #[error("CorrectionNewton BVP {backend} linear backend failed: {source}")]
    LinearBackend {
        backend: &'static str,
        #[source]
        source: CorrectionNewtonLinearError,
        report: CorrectionNewtonSolveReport,
    },
    /// The undamped Newton step points out of bounds and no safe fraction remains.
    #[error("CorrectionNewton BVP Newton step is blocked by state bounds")]
    BoundsBlocked { report: CorrectionNewtonSolveReport },
    /// The weighted RMS correction could not be represented as a finite value.
    #[error("CorrectionNewton BVP produced a non-finite weighted correction norm")]
    NonFiniteCorrectionNorm { report: CorrectionNewtonSolveReport },
    /// Newton stopped without satisfying the CorrectionNewton weighted correction test.
    #[error(
        "CorrectionNewton BVP Newton did not converge after {} iterations",
        .report.iterations
    )]
    NonConvergence { report: CorrectionNewtonSolveReport },
}

impl CorrectionNewtonSolverError {
    /// Partial numerical report, when failure happened after Newton began.
    pub fn partial_report(&self) -> Option<&CorrectionNewtonSolveReport> {
        match self {
            Self::ResidualCallback { report, .. }
            | Self::DampingDomainExhausted { report, .. }
            | Self::JacobianCallback { report, .. }
            | Self::JacobianAssembly { report, .. }
            | Self::SingularJacobian { report, .. }
            | Self::LinearResidualCheckFailed { report, .. }
            | Self::NonFiniteLinearSolution { report, .. }
            | Self::LinearBackend { report, .. }
            | Self::BoundsBlocked { report }
            | Self::NonFiniteCorrectionNorm { report }
            | Self::NonConvergence { report } => Some(report),
            Self::InvalidConfiguration { .. }
            | Self::MissingParameterValues { .. }
            | Self::SymbolicPreparation { .. }
            | Self::FrontendContract { .. }
            | Self::Logging { .. } => None,
        }
    }
}

/// Fixed-mesh damped Newton core using a pluggable internal linear workspace.
pub struct CorrectionNewtonSolver {
    state: DVector<f64>,
    initial_state: DVector<f64>,
    symbolic_mesh: Option<Vec<f64>>,
    last_successful_checkpoint: Option<parameter_loop::ContinuationCheckpoint>,
    residual: ResidualSource,
    residual_scratch: DVector<f64>,
    base_step_scratch: DVector<f64>,
    trial_state_scratch: DVector<f64>,
    trial_step_scratch: DVector<f64>,
    component_sum_scratch: Vec<f64>,
    jacobian: JacobianSource,
    linear: CorrectionNewtonLinearWorkspace,
    config: CorrectionNewtonConfig,
    external_parameter_names: Vec<String>,
    external_parameter_values: Vec<f64>,
    parameter_binding_order: Vec<usize>,
    external_parameters_bound: bool,
    fixed_boundary_values: Option<Arc<Vec<AtomicU64>>>,
    fixed_boundary_bounds: Option<Vec<(f64, f64)>>,
}

impl CorrectionNewtonSolver {
    /// Builds a callback-driven numerical core from an initial packed state.
    pub fn new<R, J>(
        initial_state: DVector<f64>,
        residual: R,
        jacobian: J,
        config: CorrectionNewtonConfig,
    ) -> Result<Self, CorrectionNewtonSolverError>
    where
        R: FnMut(&DVector<f64>) -> Result<DVector<f64>, CorrectionNewtonCallbackError> + 'static,
        J: FnMut(&DVector<f64>) -> Result<DMatrix<f64>, CorrectionNewtonCallbackError> + 'static,
    {
        let mut residual = residual;
        let mut jacobian = jacobian;
        Self::build(
            initial_state,
            ResidualSource::Allocating(Box::new(move |state, _| residual(state))),
            JacobianSource::Dense(Box::new(move |state, _| jacobian(state))),
            config,
        )
    }

    /// Builds the core with fixed sparse coordinates and a reusable values buffer.
    ///
    /// The coordinate contract is backend-neutral. `config.matrix_layout`
    /// selects Dense, production Sparse, or automatic Banded assembly; the
    /// callback reuses a values buffer and avoids a dense intermediate on the
    /// structural routes.
    pub fn new_with_coordinate_jacobian<R, J>(
        initial_state: DVector<f64>,
        residual: R,
        pattern: CorrectionNewtonJacobianPattern,
        jacobian: J,
        config: CorrectionNewtonConfig,
    ) -> Result<Self, CorrectionNewtonSolverError>
    where
        R: FnMut(&DVector<f64>) -> Result<DVector<f64>, CorrectionNewtonCallbackError> + 'static,
        J: FnMut(&DVector<f64>, &mut [f64]) -> Result<(), CorrectionNewtonCallbackError> + 'static,
    {
        let values = pattern.values_buffer();
        let mut residual = residual;
        let mut jacobian = jacobian;
        Self::build(
            initial_state,
            ResidualSource::Allocating(Box::new(move |state, _| residual(state))),
            JacobianSource::Coordinates {
                pattern,
                values,
                callback: Box::new(move |state, _, output| jacobian(state, output)),
            },
            config,
        )
    }

    /// Builds a coordinate-Jacobian core with a residual that writes into a
    /// solver-owned vector. This avoids one residual allocation per callback.
    pub(super) fn new_with_buffered_callbacks<R, J>(
        initial_state: DVector<f64>,
        residual: R,
        pattern: CorrectionNewtonJacobianPattern,
        jacobian: J,
        config: CorrectionNewtonConfig,
    ) -> Result<Self, CorrectionNewtonSolverError>
    where
        R: FnMut(&DVector<f64>, &mut DVector<f64>) -> Result<(), CorrectionNewtonCallbackFailure>
            + 'static,
        J: FnMut(&DVector<f64>, &mut [f64]) -> Result<(), CorrectionNewtonCallbackFailure>
            + 'static,
    {
        let values = pattern.values_buffer();
        let mut residual = residual;
        let mut jacobian = jacobian;
        Self::build(
            initial_state,
            ResidualSource::Buffered(Box::new(move |state, _, output| residual(state, output))),
            JacobianSource::SymbolicCoordinates {
                pattern,
                values,
                callback: Box::new(move |state, _, output| jacobian(state, output)),
            },
            config,
        )
    }

    pub(super) fn new_with_parameterized_buffered_callbacks<R, J>(
        initial_state: DVector<f64>,
        residual: R,
        pattern: CorrectionNewtonJacobianPattern,
        jacobian: J,
        parameter_names: Vec<String>,
        config: CorrectionNewtonConfig,
    ) -> Result<Self, CorrectionNewtonSolverError>
    where
        R: FnMut(
                &DVector<f64>,
                &[f64],
                &mut DVector<f64>,
            ) -> Result<(), CorrectionNewtonCallbackFailure>
            + 'static,
        J: FnMut(&DVector<f64>, &[f64], &mut [f64]) -> Result<(), CorrectionNewtonCallbackFailure>
            + 'static,
    {
        let values = pattern.values_buffer();
        let solver = Self::build(
            initial_state,
            ResidualSource::Buffered(Box::new(residual)),
            JacobianSource::SymbolicCoordinates {
                pattern,
                values,
                callback: Box::new(jacobian),
            },
            config,
        )?;
        solver.with_parameter_schema(parameter_names)
    }

    fn build(
        initial_state: DVector<f64>,
        residual: ResidualSource,
        jacobian: JacobianSource,
        config: CorrectionNewtonConfig,
    ) -> Result<Self, CorrectionNewtonSolverError> {
        let size = initial_state.len();
        if let JacobianSource::Coordinates { pattern, .. }
        | JacobianSource::SymbolicCoordinates { pattern, .. } = &jacobian
        {
            if pattern.dimension() != size {
                return Err(CorrectionNewtonSolverError::InvalidConfiguration {
                    source: CorrectionNewtonConfigurationError::JacobianPatternDimensionMismatch {
                        pattern_dimension: pattern.dimension(),
                        state_size: size,
                    },
                });
            }
        }
        let valid =
            size > 0
                && config.component_count > 0
                && size % config.component_count == 0
                && config.absolute_tolerances.len() == config.component_count
                && config.relative_tolerances.len() == config.component_count
                && config.bounds.len() == size
                && valid_full_state_layout(size, &config)
                && config.max_iterations > 0
                && config.max_damping_trials > 0
                && initial_state.iter().all(|value| value.is_finite())
                && config
                    .absolute_tolerances
                    .iter()
                    .all(|value| value.is_finite() && *value > 0.0)
                && config
                    .relative_tolerances
                    .iter()
                    .all(|value| value.is_finite() && *value >= 0.0)
                && config.bounds.iter().zip(initial_state.iter()).all(
                    |(&(lower, upper), value)| {
                        !lower.is_nan()
                            && !upper.is_nan()
                            && lower <= upper
                            && *value >= lower
                            && *value <= upper
                    },
                );
        if !valid {
            return Err(CorrectionNewtonSolverError::InvalidConfiguration {
                source: CorrectionNewtonConfigurationError::InvalidNewtonConfiguration,
            });
        }
        let linear =
            CorrectionNewtonLinearWorkspace::new(config.matrix_layout, size, jacobian.pattern())
                .map_err(|error| CorrectionNewtonSolverError::InvalidConfiguration {
                    source: CorrectionNewtonConfigurationError::LinearWorkspace(error),
                })?;
        Ok(Self {
            initial_state: initial_state.clone(),
            symbolic_mesh: None,
            last_successful_checkpoint: None,
            residual_scratch: DVector::zeros(size),
            base_step_scratch: DVector::zeros(size),
            trial_state_scratch: DVector::zeros(size),
            trial_step_scratch: DVector::zeros(size),
            component_sum_scratch: vec![0.0; config.component_count],
            state: initial_state,
            residual,
            jacobian,
            linear,
            config,
            external_parameter_values: vec![0.0; 0],
            external_parameter_names: Vec::new(),
            parameter_binding_order: Vec::new(),
            external_parameters_bound: true,
            fixed_boundary_values: None,
            fixed_boundary_bounds: None,
        })
    }

    fn with_parameter_schema(
        mut self,
        parameter_names: Vec<String>,
    ) -> Result<Self, CorrectionNewtonSolverError> {
        if parameter_names.iter().any(|name| name.trim().is_empty())
            || parameter_names
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len()
                != parameter_names.len()
        {
            return Err(CorrectionNewtonSolverError::InvalidConfiguration {
                source: CorrectionNewtonConfigurationError::InvalidParameterSchema,
            });
        }
        self.external_parameter_values = vec![0.0; parameter_names.len()];
        self.parameter_binding_order = (0..parameter_names.len()).collect();
        self.external_parameters_bound = parameter_names.is_empty();
        self.external_parameter_names = parameter_names;
        Ok(self)
    }

    /// Prepares the selected symbolic Lambdify fixed-mesh trapezoid system.
    ///
    /// The structural Jacobian is compiled once and evaluated into a reusable
    /// coordinate buffer. The selected Dense/Sparse/Banded backend consumes
    /// that structure directly; no other solver's nonlinear loop or solver
    /// object is created.
    pub fn from_symbolic(
        problem: CorrectionNewtonProblem,
        options: CorrectionNewtonOptions,
    ) -> Result<Self, CorrectionNewtonSolverError> {
        symbolic_plan::prepare(problem, options)
    }

    /// Legacy spelling for the configured symbolic frontend.
    pub fn from_expr_legacy(
        problem: CorrectionNewtonProblem,
        options: CorrectionNewtonOptions,
    ) -> Result<Self, CorrectionNewtonSolverError> {
        Self::from_symbolic(problem, options)
    }

    /// Dense-compatible spelling retained for clients of the initial pilot.
    ///
    /// New callers should use [`Self::from_symbolic`] and select their
    /// matrix layout through [`CorrectionNewtonOptions`].
    pub fn from_expr_legacy_dense(
        problem: CorrectionNewtonProblem,
        mut options: CorrectionNewtonOptions,
    ) -> Result<Self, CorrectionNewtonSolverError> {
        options.matrix_layout = CorrectionNewtonMatrixLayout::Dense;
        Self::from_symbolic(problem, options)
    }

    /// Returns the solver's current state. It changes only after an accepted trial.
    pub fn state(&self) -> &DVector<f64> {
        &self.state
    }

    /// Names of fixed symbolic inputs in the deterministic callback order.
    pub fn external_parameter_names(&self) -> &[String] {
        &self.external_parameter_names
    }

    /// Current fixed symbolic inputs in the callback order.
    pub fn external_parameter_values(&self) -> &[f64] {
        &self.external_parameter_values
    }

    /// Bind values to the prepared external-input schema without recompiling callbacks.
    ///
    /// Values may be provided in any order if names cover the prepared schema
    /// exactly. The mapping is resolved once here, outside the Newton loop.
    pub fn bind_parameters(
        &mut self,
        names: Vec<String>,
        values: Vec<f64>,
    ) -> Result<(), CorrectionNewtonSolverError> {
        if names.len() != values.len()
            || values.iter().any(|value| !value.is_finite())
            || names.iter().any(|name| name.trim().is_empty())
        {
            return Err(CorrectionNewtonSolverError::InvalidConfiguration {
                source: CorrectionNewtonConfigurationError::ParameterBindingLengthMismatch {
                    names: names.len(),
                    values: values.len(),
                },
            });
        }
        let schema = self
            .external_parameter_names
            .iter()
            .map(String::as_str)
            .collect::<std::collections::HashSet<_>>();
        let mut seen = std::collections::HashSet::with_capacity(names.len());
        if names.len() != schema.len()
            || names
                .iter()
                .any(|name| !schema.contains(name.as_str()) || !seen.insert(name.as_str()))
        {
            return Err(CorrectionNewtonSolverError::InvalidConfiguration {
                source: CorrectionNewtonConfigurationError::ParameterSchemaMismatch,
            });
        }
        let indices = self
            .external_parameter_names
            .iter()
            .map(|expected| {
                names
                    .iter()
                    .position(|name| name == expected)
                    .expect("validated parameter schema contains every expected name")
            })
            .collect::<Vec<_>>();
        for (target, &input_index) in self
            .external_parameter_values
            .iter_mut()
            .zip(indices.iter())
        {
            *target = values[input_index];
        }
        self.parameter_binding_order = indices;
        self.external_parameters_bound = true;
        Ok(())
    }

    /// Replace values in the order established by the most recent binding.
    pub fn set_parameter_values(
        &mut self,
        values: Vec<f64>,
    ) -> Result<(), CorrectionNewtonSolverError> {
        self.validate_parameter_values(&values)?;
        for (target, &input_index) in self
            .external_parameter_values
            .iter_mut()
            .zip(&self.parameter_binding_order)
        {
            *target = values[input_index];
        }
        Ok(())
    }

    /// Rebind numerical endpoint values without rebuilding symbolic callbacks.
    /// The value order is the fixed endpoint order established during symbolic
    /// preparation; endpoint positions and the reduced Newton layout are fixed.
    pub fn set_boundary_values(
        &mut self,
        values: Vec<f64>,
    ) -> Result<(), CorrectionNewtonSolverError> {
        let invalid = || CorrectionNewtonSolverError::InvalidConfiguration {
            source: CorrectionNewtonConfigurationError::BoundarySchemaMismatch,
        };
        if values.len() != self.config.fixed_state_values.len()
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(invalid());
        }
        let Some(binding) = self.fixed_boundary_values.as_ref() else {
            return Err(invalid());
        };
        if binding.len() != values.len() {
            return Err(invalid());
        }
        let Some(bounds) = self.fixed_boundary_bounds.as_ref() else {
            return Err(invalid());
        };
        if bounds.len() != values.len() {
            return Err(invalid());
        }
        for (slot, (&value, &(lower, upper))) in values.iter().zip(bounds).enumerate() {
            if value < lower || value > upper {
                return Err(CorrectionNewtonSolverError::InvalidConfiguration {
                    source: CorrectionNewtonConfigurationError::BoundaryValueOutsideBounds {
                        slot,
                        value,
                        lower,
                        upper,
                    },
                });
            }
        }
        for ((_, current), value) in self.config.fixed_state_values.iter_mut().zip(&values) {
            *current = *value;
        }
        for (slot, value) in binding.iter().zip(values) {
            slot.store(value.to_bits(), std::sync::atomic::Ordering::Relaxed);
        }
        self.last_successful_checkpoint = None;
        Ok(())
    }

    fn validate_parameter_values(&self, values: &[f64]) -> Result<(), CorrectionNewtonSolverError> {
        if !self.external_parameters_bound {
            return Err(CorrectionNewtonSolverError::MissingParameterValues {
                names: self.external_parameter_names.clone(),
            });
        }
        if values.len() != self.parameter_binding_order.len()
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(CorrectionNewtonSolverError::InvalidConfiguration {
                source: CorrectionNewtonConfigurationError::ParameterValueCountMismatch {
                    expected: self.parameter_binding_order.len(),
                    actual: values.len(),
                },
            });
        }
        Ok(())
    }

    /// Expand a reduced CorrectionNewton Newton state into node-major grid values.
    ///
    /// This allocates the public profile once after a solve; it is not used in
    /// the Newton hot path. Generic callback systems without an explicit map
    /// are treated as one complete row.
    pub fn full_grid_values(
        &self,
        state: &DVector<f64>,
    ) -> Result<DMatrix<f64>, CorrectionNewtonSolverError> {
        let component_count = self.config.component_count;
        let full_size = state
            .len()
            .checked_add(self.config.fixed_state_values.len())
            .ok_or_else(|| CorrectionNewtonSolverError::InvalidConfiguration {
                source: CorrectionNewtonConfigurationError::ExpandedStateSizeOverflow,
            })?;
        if component_count == 0 || full_size == 0 || full_size % component_count != 0 {
            return Err(CorrectionNewtonSolverError::InvalidConfiguration {
                source: CorrectionNewtonConfigurationError::ExpandedStateComponentMismatch {
                    state_size: full_size,
                    components: component_count,
                },
            });
        }
        let node_count = full_size / component_count;
        let mut values = DMatrix::from_element(node_count, component_count, f64::NAN);
        if self.config.full_state_positions.is_empty() {
            if state.len() != full_size {
                return Err(CorrectionNewtonSolverError::InvalidConfiguration {
                    source: CorrectionNewtonConfigurationError::IncompleteGridMapping,
                });
            }
            for (index, &value) in state.iter().enumerate() {
                values[(index / component_count, index % component_count)] = value;
            }
        } else {
            if self.config.full_state_positions.len() != state.len() {
                return Err(CorrectionNewtonSolverError::InvalidConfiguration {
                    source: CorrectionNewtonConfigurationError::GridMappingLengthMismatch {
                        mapped: self.config.full_state_positions.len(),
                        reduced: state.len(),
                    },
                });
            }
            for (&position, &value) in self.config.full_state_positions.iter().zip(state.iter()) {
                if position >= full_size {
                    return Err(CorrectionNewtonSolverError::InvalidConfiguration {
                        source: CorrectionNewtonConfigurationError::GridPositionOutOfBounds {
                            position,
                            state_size: full_size,
                        },
                    });
                }
                values[(position / component_count, position % component_count)] = value;
            }
            for &(position, value) in &self.config.fixed_state_values {
                if position >= full_size {
                    return Err(CorrectionNewtonSolverError::InvalidConfiguration {
                        source: CorrectionNewtonConfigurationError::BoundaryPositionOutOfBounds {
                            position,
                            state_size: full_size,
                        },
                    });
                }
                values[(position / component_count, position % component_count)] = value;
            }
        }
        if values.iter().any(|value| !value.is_finite()) {
            let index = values
                .iter()
                .position(|value| !value.is_finite())
                .expect("non-finite value was found above");
            return Err(CorrectionNewtonSolverError::InvalidConfiguration {
                source: CorrectionNewtonConfigurationError::InvalidExpandedGridValue { index },
            });
        }
        Ok(values)
    }

    /// Runs the standalone fixed-mesh damped correction-Newton loop.
    pub fn solve(&mut self) -> Result<CorrectionNewtonSolveReport, CorrectionNewtonSolverError> {
        if !self.external_parameters_bound {
            return Err(CorrectionNewtonSolverError::MissingParameterValues {
                names: self.external_parameter_names.clone(),
            });
        }
        let _logging = SolverLogGuard::try_enter(self.config.logging_level)
            .map_err(|source| CorrectionNewtonSolverError::Logging { source })?;
        let mut report = self.empty_report();
        let mode = self.config.telemetry_mode;
        let mut jacobian_age = 0_usize;
        let mut force_jacobian = true;
        let mut jacobian_reevaluations_after_failed_damping = 0_usize;

        for iteration in 0..self.config.max_iterations {
            report.iterations = iteration + 1;
            // CorrectionNewton refreshes only once age is strictly greater than max age.
            if force_jacobian || jacobian_age > self.config.max_jacobian_age {
                let previous_jacobian_age = jacobian_age;
                let refresh_reason = if force_jacobian {
                    "damping retry"
                } else {
                    "maximum age"
                };
                // Unlike CorrectionNewton's MultiJac finite-difference assembly, this
                // pilot receives an already assembled symbolic Jacobian.
                let start = (mode == CorrectionNewtonTelemetryMode::Timings).then(Instant::now);
                if mode != CorrectionNewtonTelemetryMode::Off {
                    report.telemetry.jacobian_evaluations += 1;
                }
                let callback_result = self
                    .jacobian
                    .evaluate(&self.state, &self.external_parameter_values);
                if let Some(start) = start {
                    let elapsed = start.elapsed();
                    report.telemetry.callback_time += elapsed;
                    report.telemetry.jacobian_callback_time += elapsed;
                }
                let evaluated = callback_result.map_err(|error| match error {
                    JacobianSourceError::Callback(source) => {
                        CorrectionNewtonSolverError::JacobianCallback {
                            kind: source.kind(),
                            source,
                            report: report.clone(),
                        }
                    }
                    JacobianSourceError::Assembly(source) => {
                        CorrectionNewtonSolverError::JacobianAssembly {
                            source,
                            report: report.clone(),
                        }
                    }
                })?;
                let factor_result = match evaluated {
                    EvaluatedJacobian::Dense(matrix) => {
                        let factor_start =
                            (mode == CorrectionNewtonTelemetryMode::Timings).then(Instant::now);
                        if mode != CorrectionNewtonTelemetryMode::Off {
                            report.telemetry.factorizations += 1;
                        }
                        let result = self.linear.factor_dense(matrix);
                        if let Some(start) = factor_start {
                            report.telemetry.factorization_time += start.elapsed();
                        }
                        result
                    }
                    EvaluatedJacobian::Coordinates { values } => {
                        let assembly_start =
                            (mode == CorrectionNewtonTelemetryMode::Timings).then(Instant::now);
                        let assembly_result = self.linear.assemble_coordinates(values);
                        if let Some(start) = assembly_start {
                            report.telemetry.jacobian_assembly_time += start.elapsed();
                        }
                        assembly_result.map_err(|error| map_linear_error(error, report.clone()))?;

                        let factor_start =
                            (mode == CorrectionNewtonTelemetryMode::Timings).then(Instant::now);
                        if mode != CorrectionNewtonTelemetryMode::Off {
                            report.telemetry.factorizations += 1;
                        }
                        let result = self.linear.factor_assembled();
                        if let Some(start) = factor_start {
                            report.telemetry.factorization_time += start.elapsed();
                        }
                        result
                    }
                };
                factor_result.map_err(|error| map_linear_error(error, report.clone()))?;
                jacobian_age = 0;
                force_jacobian = false;
                solver_debug!(
                    "CorrectionNewton BVP refreshed {} Jacobian at iteration {} (reason={}, previous age={})",
                    self.linear.backend().name(),
                    iteration + 1,
                    refresh_reason,
                    previous_jacobian_age
                );
            }
            Self::eval_residual(
                &mut self.residual,
                &mut self.residual_scratch,
                &self.state,
                &self.external_parameter_values,
                &mut report,
                mode,
            )?;
            report.residual_norm = Some(self.residual_scratch.amax());
            Self::solve_linear(
                &mut self.linear,
                self.residual_scratch.as_slice(),
                &mut self.base_step_scratch,
                true,
                &mut report,
                mode,
            )?;
            let s0 = Self::weighted_rms(
                &self.config,
                &self.state,
                &self.base_step_scratch,
                &mut self.component_sum_scratch,
            )
            .ok_or_else(|| CorrectionNewtonSolverError::NonFiniteCorrectionNorm {
                report: report.clone(),
            })?;
            report.correction_norm = Some(s0);
            jacobian_age += 1;

            let bound_factor = self.bound_factor(&self.base_step_scratch);
            report.last_bound_factor = Some(bound_factor);
            if bound_factor < 1.0 && mode != CorrectionNewtonTelemetryMode::Off {
                report.telemetry.bound_limited_steps += 1;
            }
            // MultiNewton reports this as damping status -3. The caller still
            // gets its stale-Jacobian retry opportunity before terminal failure.
            let bounds_blocked = bound_factor < 1e-10;
            if bounds_blocked {
                solver_warn!(
                    "CorrectionNewton BVP iteration {} blocked by bounds: fraction={:.6e}",
                    iteration + 1,
                    bound_factor
                );
            }

            let mut alpha = bound_factor;
            let mut accepted = false;
            let mut valid_damping_trials = 0_usize;
            let mut last_domain_failure = None;
            let damping_trial_limit = if bounds_blocked {
                0
            } else {
                self.config.max_damping_trials
            };
            for damping_index in 0..damping_trial_limit {
                self.trial_state_scratch.copy_from(&self.state);
                self.trial_state_scratch
                    .axpy(alpha, &self.base_step_scratch, 1.0);
                report.last_damping_factor = Some(alpha);
                if mode != CorrectionNewtonTelemetryMode::Off {
                    report.telemetry.damping_trials += 1;
                }
                let residual_result = Self::eval_residual(
                    &mut self.residual,
                    &mut self.residual_scratch,
                    &self.trial_state_scratch,
                    &self.external_parameter_values,
                    &mut report,
                    mode,
                );
                match residual_result {
                    Ok(()) => valid_damping_trials += 1,
                    Err(CorrectionNewtonSolverError::ResidualCallback {
                        kind: CorrectionNewtonCallbackErrorKind::Domain,
                        source,
                        ..
                    }) => {
                        if mode != CorrectionNewtonTelemetryMode::Off {
                            report.telemetry.damping_rejections += 1;
                            report.telemetry.domain_rejections += 1;
                        }
                        solver_debug!(
                            "CorrectionNewton BVP iteration {} damping trial {} left the model domain; retrying with smaller alpha: {}",
                            iteration + 1,
                            damping_index + 1,
                            source
                        );
                        last_domain_failure = Some(source);
                        alpha /= 2.0_f64.sqrt();
                        continue;
                    }
                    Err(error) => return Err(error),
                }
                let trial_residual_norm = self.residual_scratch.amax();
                Self::solve_linear(
                    &mut self.linear,
                    self.residual_scratch.as_slice(),
                    &mut self.trial_step_scratch,
                    true,
                    &mut report,
                    mode,
                )?;
                let s1 = Self::weighted_rms(
                    &self.config,
                    &self.trial_state_scratch,
                    &self.trial_step_scratch,
                    &mut self.component_sum_scratch,
                )
                .ok_or_else(|| {
                    CorrectionNewtonSolverError::NonFiniteCorrectionNorm {
                        report: report.clone(),
                    }
                })?;
                let improves = s1 < 1.0 || s1 < s0;
                solver_debug!(
                    "CorrectionNewton BVP iteration {} damping trial {}: bound={:.6e}, alpha={:.6e}, correction={:.6e}->{:.6e}, accepted={}",
                    iteration + 1,
                    damping_index + 1,
                    bound_factor,
                    alpha,
                    s0,
                    s1,
                    improves
                );
                if improves {
                    report.correction_norm = Some(s1);
                    report.residual_norm = Some(trial_residual_norm);
                    accepted = true;
                    break;
                }
                if mode != CorrectionNewtonTelemetryMode::Off {
                    report.telemetry.damping_rejections += 1;
                }
                alpha /= 2.0_f64.sqrt();
            }

            if accepted {
                self.state.copy_from(&self.trial_state_scratch);
                report.state.clone_from(&self.state);
                if mode != CorrectionNewtonTelemetryMode::Off {
                    report.telemetry.accepted_steps += 1;
                }
                let correction_norm = report.correction_norm.expect("accepted trial has norm");
                solver_info!(
                    "CorrectionNewton BVP Newton iteration {} accepted alpha={:.6e}, weighted correction={:.6e}",
                    iteration + 1,
                    alpha,
                    correction_norm
                );
                if correction_norm <= 1.0 {
                    self.last_successful_checkpoint = Some(self.capture_checkpoint());
                    return Ok(report);
                }
            // CorrectionNewton retries a failed damping step with a refreshed Jacobian
            // before returning failure when the cached Jacobian is stale.
            } else if jacobian_age > 1 && jacobian_reevaluations_after_failed_damping <= 3 {
                force_jacobian = true;
                jacobian_reevaluations_after_failed_damping += 1;
                let failure_kind = if bounds_blocked {
                    "bounds-blocked step"
                } else {
                    "damping exhaustion"
                };
                solver_info!(
                    "CorrectionNewton BVP iteration {} had {}; refreshing reused Jacobian (retry {})",
                    iteration + 1,
                    failure_kind,
                    jacobian_reevaluations_after_failed_damping
                );
            } else if valid_damping_trials == 0 && last_domain_failure.is_some() {
                let source = last_domain_failure.expect("domain failure was checked above");
                solver_warn!(
                    "CorrectionNewton BVP iteration {} exhausted {} damping trials outside the model domain",
                    iteration + 1,
                    damping_trial_limit
                );
                return Err(CorrectionNewtonSolverError::DampingDomainExhausted { source, report });
            } else {
                let failure_kind = if bounds_blocked {
                    "bounds-blocked step"
                } else {
                    "damping exhaustion"
                };
                solver_warn!(
                    "CorrectionNewton BVP stopped after {failure_kind} at iteration {} (Jacobian age {})",
                    iteration + 1,
                    jacobian_age
                );
                return if bounds_blocked {
                    Err(CorrectionNewtonSolverError::BoundsBlocked { report })
                } else {
                    Err(CorrectionNewtonSolverError::NonConvergence { report })
                };
            }
        }

        // The last accepted trial already supplied this state's residual norm;
        // do not call the residual again just to format a diagnostic report.
        solver_warn!(
            "CorrectionNewton BVP reached the iteration limit ({}) without convergence",
            self.config.max_iterations
        );
        Err(CorrectionNewtonSolverError::NonConvergence { report })
    }

    fn empty_report(&self) -> CorrectionNewtonSolveReport {
        CorrectionNewtonSolveReport {
            state: self.state.clone(),
            linear_backend: self.linear.backend(),
            iterations: 0,
            correction_norm: None,
            residual_norm: None,
            last_bound_factor: None,
            last_damping_factor: None,
            telemetry: CorrectionNewtonTelemetrySnapshot::default(),
        }
    }

    fn eval_residual(
        residual: &mut ResidualSource,
        residual_scratch: &mut DVector<f64>,
        state: &DVector<f64>,
        external_parameters: &[f64],
        report: &mut CorrectionNewtonSolveReport,
        telemetry_mode: CorrectionNewtonTelemetryMode,
    ) -> Result<(), CorrectionNewtonSolverError> {
        let start = (telemetry_mode == CorrectionNewtonTelemetryMode::Timings).then(Instant::now);
        if telemetry_mode != CorrectionNewtonTelemetryMode::Off {
            report.telemetry.residual_evaluations += 1;
        }
        let callback_result = match (&mut *residual, &mut *residual_scratch) {
            (ResidualSource::Allocating(callback), output) => callback(state, external_parameters)
                .and_then(|value| {
                    if value.len() != output.len() {
                        return Err(CorrectionNewtonCallbackFailure::ShapeMismatch {
                            expected: CorrectionNewtonCallbackShape::Vector {
                                length: output.len(),
                            },
                            actual: CorrectionNewtonCallbackShape::Vector {
                                length: value.len(),
                            },
                        });
                    }
                    if let Some(index) = value.iter().position(|value| !value.is_finite()) {
                        return Err(CorrectionNewtonCallbackFailure::NonFinite {
                            location: CorrectionNewtonCallbackLocation::Residual { index },
                        });
                    }
                    output.copy_from(&value);
                    Ok(())
                }),
            (ResidualSource::Buffered(callback), output) => {
                callback(state, external_parameters, output)
            }
        };
        if let Some(start) = start {
            let elapsed = start.elapsed();
            report.telemetry.callback_time += elapsed;
            report.telemetry.residual_callback_time += elapsed;
        }
        callback_result.map_err(|source| CorrectionNewtonSolverError::ResidualCallback {
            kind: source.kind(),
            source,
            report: report.clone(),
        })?;
        if residual_scratch.len() != state.len() {
            return Err(CorrectionNewtonSolverError::ResidualCallback {
                kind: CorrectionNewtonCallbackErrorKind::ShapeMismatch,
                source: CorrectionNewtonCallbackError::ShapeMismatch {
                    expected: CorrectionNewtonCallbackShape::Vector {
                        length: state.len(),
                    },
                    actual: CorrectionNewtonCallbackShape::Vector {
                        length: residual_scratch.len(),
                    },
                },
                report: report.clone(),
            });
        }
        if let Some(index) = residual_scratch.iter().position(|value| !value.is_finite()) {
            return Err(CorrectionNewtonSolverError::ResidualCallback {
                kind: CorrectionNewtonCallbackErrorKind::NonFinite,
                source: CorrectionNewtonCallbackError::NonFinite {
                    location: CorrectionNewtonCallbackLocation::Residual { index },
                },
                report: report.clone(),
            });
        }
        Ok(())
    }

    fn solve_linear(
        factorization: &mut CorrectionNewtonLinearWorkspace,
        rhs: &[f64],
        solution: &mut DVector<f64>,
        negate_rhs: bool,
        report: &mut CorrectionNewtonSolveReport,
        telemetry_mode: CorrectionNewtonTelemetryMode,
    ) -> Result<(), CorrectionNewtonSolverError> {
        let start = (telemetry_mode == CorrectionNewtonTelemetryMode::Timings).then(Instant::now);
        if rhs.len() != solution.len() {
            return Err(map_linear_error(
                CorrectionNewtonLinearError::RightHandSideLengthMismatch {
                    backend: factorization.backend().name(),
                    expected: solution.len(),
                    actual: rhs.len(),
                },
                report.clone(),
            ));
        }
        solution.copy_from_slice(rhs);
        if negate_rhs {
            solution.scale_mut(-1.0);
        }
        let solve_result = factorization.solve_in_place(solution.as_mut_slice());
        if let Some(start) = start {
            report.telemetry.linear_solve_time += start.elapsed();
        }
        if telemetry_mode != CorrectionNewtonTelemetryMode::Off {
            report.telemetry.linear_solves += 1;
        }
        match solve_result {
            Ok(()) if solution.iter().all(|value| value.is_finite()) => Ok(()),
            Err(error) => Err(map_linear_error(error, report.clone())),
            Ok(()) => Err(CorrectionNewtonSolverError::NonFiniteLinearSolution {
                backend: factorization.backend().name(),
                report: report.clone(),
            }),
        }
    }

    fn weighted_rms(
        config: &CorrectionNewtonConfig,
        state: &DVector<f64>,
        correction: &DVector<f64>,
        component_sums: &mut [f64],
    ) -> Option<f64> {
        let components = config.component_count;
        if components == 0
            || state.len() != correction.len()
            || component_sums.len() != components
            || config.absolute_tolerances.len() != components
            || config.relative_tolerances.len() != components
        {
            return None;
        }
        let full_size = state.len().checked_add(config.fixed_state_values.len())?;
        if full_size == 0 || full_size % components != 0 {
            return None;
        }
        let points = full_size / components;
        component_sums.fill(0.0);
        for &(position, value) in &config.fixed_state_values {
            if position >= full_size || !value.is_finite() {
                return None;
            }
            component_sums[position % components] += value.abs();
        }
        for (index, value) in state.iter().enumerate() {
            let position = Self::full_state_position(config, index);
            if position >= full_size || !value.is_finite() {
                return None;
            }
            component_sums[position % components] += value.abs();
        }
        let mut sum = 0.0;
        for component in 0..components {
            // Match CorrectionNewton's norm2: component weights use the mean magnitude
            // over the full grid, including fixed boundary values.
            let mean_abs_state = component_sums[component] / points as f64;
            let scale = config.absolute_tolerances[component]
                + config.relative_tolerances[component] * mean_abs_state;
            if !scale.is_finite() || scale <= 0.0 {
                return None;
            }
        }
        for index in 0..state.len() {
            let component = Self::full_state_position(config, index) % components;
            let scaled = correction[index]
                / (config.absolute_tolerances[component]
                    + config.relative_tolerances[component] * component_sums[component]
                        / points as f64);
            let term = scaled * scaled;
            if !term.is_finite() {
                return None;
            }
            sum += term;
            if !sum.is_finite() {
                return None;
            }
        }
        Some((sum / full_size as f64).sqrt())
    }

    fn full_state_position(config: &CorrectionNewtonConfig, packed_index: usize) -> usize {
        config
            .full_state_positions
            .get(packed_index)
            .copied()
            .unwrap_or(packed_index)
    }

    fn bound_factor(&self, step: &DVector<f64>) -> f64 {
        step.iter()
            .zip(self.state.iter())
            .zip(&self.config.bounds)
            .fold(1.0_f64, |factor, ((delta, value), &(lower, upper))| {
                let candidate = if *delta > 0.0 && value + delta > upper {
                    (upper - value) / delta
                } else if *delta < 0.0 && value + delta < lower {
                    (lower - value) / delta
                } else {
                    1.0
                };
                factor.min(candidate.clamp(0.0, 1.0))
            })
    }
}

/// Evaluate prepared symbolic callbacks on caller-supplied full-grid states.
///
/// This is test-only plumbing for cross-solver residual/Jacobian parity gates;
/// it intentionally does not expose callback internals in the production API.
#[cfg(test)]
pub(crate) fn evaluate_symbolic_callbacks_for_test(
    problem: CorrectionNewtonProblem,
    options: CorrectionNewtonOptions,
    full_profiles: &[DMatrix<f64>],
) -> Vec<(DVector<f64>, DMatrix<f64>)> {
    let mut solver = CorrectionNewtonSolver::from_symbolic(problem, options)
        .expect("symbolic callback parity fixture should prepare");
    let component_count = solver.config.component_count;
    let full_points = solver
        .config
        .full_state_positions
        .len()
        .checked_add(solver.config.fixed_state_values.len())
        .expect("validated full-grid dimensions fit usize")
        / component_count;
    let coordinates = solver
        .jacobian
        .pattern()
        .expect("symbolic BVP Jacobian has a coordinate pattern")
        .coordinates()
        .to_vec();
    let mut results = Vec::with_capacity(full_profiles.len());

    for profile in full_profiles {
        assert_eq!(profile.shape(), (full_points, component_count));
        let state =
            DVector::from_iterator(
                solver.config.full_state_positions.len(),
                solver.config.full_state_positions.iter().map(|&position| {
                    profile[(position / component_count, position % component_count)]
                }),
            );
        let mut report = solver.empty_report();
        CorrectionNewtonSolver::eval_residual(
            &mut solver.residual,
            &mut solver.residual_scratch,
            &state,
            &solver.external_parameter_values,
            &mut report,
            CorrectionNewtonTelemetryMode::Off,
        )
        .expect("symbolic residual callback should evaluate at the supplied profile");
        let residual = solver.residual_scratch.clone();
        let jacobian = solver
            .jacobian
            .evaluate(&state, &solver.external_parameter_values)
            .expect("symbolic Jacobian callback should evaluate at the supplied profile");
        let dense = match jacobian {
            EvaluatedJacobian::Dense(matrix) => matrix,
            EvaluatedJacobian::Coordinates { values } => {
                let mut matrix = DMatrix::zeros(state.len(), state.len());
                for (&(row, column), &value) in coordinates.iter().zip(values) {
                    matrix[(row, column)] = value;
                }
                matrix
            }
        };
        results.push((residual, dense));
    }
    results
}

struct PackedBvpInitialState {
    values: DVector<f64>,
    full_state_positions: Vec<usize>,
    fixed_state_values: Vec<(usize, f64)>,
}

fn pack_bvp_initial_state(
    problem: &CorrectionNewtonProblem,
) -> Result<PackedBvpInitialState, CorrectionNewtonSolverError> {
    let component_count = problem.state_names.len();
    let full_points = problem.steps.checked_add(1).ok_or_else(|| {
        CorrectionNewtonSolverError::InvalidConfiguration {
            source: CorrectionNewtonConfigurationError::FullGridPointCountOverflow,
        }
    })?;
    let mut fixed_by_component = vec![None; component_count];

    for (name, conditions) in &problem.boundary_conditions {
        let Some(component) = problem.state_names.iter().position(|state| state == name) else {
            return Err(CorrectionNewtonSolverError::InvalidConfiguration {
                source: CorrectionNewtonConfigurationError::UnknownBoundaryState {
                    state: name.clone(),
                },
            });
        };
        if conditions.len() != 1 {
            return Err(CorrectionNewtonSolverError::InvalidConfiguration {
                source: CorrectionNewtonConfigurationError::EndpointConditionCountMismatch {
                    state: name.clone(),
                    actual: conditions.len(),
                },
            });
        }
        let (side, value) = conditions[0];
        if side > 1 || !value.is_finite() {
            return Err(CorrectionNewtonSolverError::InvalidConfiguration {
                source: CorrectionNewtonConfigurationError::InvalidEndpointCondition {
                    state: name.clone(),
                    side,
                    value,
                },
            });
        }
        let node = if side == 0 { 0 } else { problem.steps };
        fixed_by_component[component] = Some((node, value));
    }
    if let Some(component) = fixed_by_component.iter().position(Option::is_none) {
        return Err(CorrectionNewtonSolverError::InvalidConfiguration {
            source: CorrectionNewtonConfigurationError::MissingEndpointCondition {
                state: problem.state_names[component].clone(),
            },
        });
    }
    if problem.boundary_conditions.len() != component_count {
        return Err(CorrectionNewtonSolverError::InvalidConfiguration {
            source: CorrectionNewtonConfigurationError::BoundarySchemaMismatch,
        });
    }

    let full_size = component_count.checked_mul(full_points).ok_or_else(|| {
        CorrectionNewtonSolverError::InvalidConfiguration {
            source: CorrectionNewtonConfigurationError::FullGridStateSizeOverflow,
        }
    })?;
    let mut values = Vec::with_capacity(component_count * problem.steps);
    let mut full_state_positions = Vec::with_capacity(component_count * problem.steps);
    let mut fixed_state_values = Vec::with_capacity(component_count);

    for node in 0..full_points {
        for (component, name) in problem.state_names.iter().enumerate() {
            let fixed = fixed_by_component[component]
                .expect("validated boundary condition exists for every state");
            let full_position = node * component_count + component;
            if fixed.0 == node {
                fixed_state_values.push((full_position, fixed.1));
                continue;
            }

            // The public guess matrix has `steps` entries per component. A
            // left-constrained component's free entries begin at mesh node 1;
            // a right-constrained component's free entries begin at node 0.
            let guess_column = if fixed.0 == 0 { node - 1 } else { node };
            if guess_column >= problem.steps || full_position >= full_size {
                return Err(CorrectionNewtonSolverError::InvalidConfiguration {
                    source: CorrectionNewtonConfigurationError::InitialGuessMappingFailure {
                        node,
                        state: name.clone(),
                    },
                });
            }
            values.push(problem.initial_guess[(component, guess_column)]);
            full_state_positions.push(full_position);
        }
    }

    if values.len() != component_count * problem.steps
        || values.len() != problem.initial_guess.len()
    {
        return Err(CorrectionNewtonSolverError::InvalidConfiguration {
            source: CorrectionNewtonConfigurationError::InitialGuessLengthMismatch {
                free_values: values.len(),
                guess_values: problem.initial_guess.len(),
            },
        });
    }
    Ok(PackedBvpInitialState {
        values: DVector::from_vec(values),
        full_state_positions,
        fixed_state_values,
    })
}

fn valid_full_state_layout(state_size: usize, config: &CorrectionNewtonConfig) -> bool {
    if config.component_count == 0 {
        return false;
    }
    let Some(full_size) = state_size.checked_add(config.fixed_state_values.len()) else {
        return false;
    };
    if full_size == 0 || full_size % config.component_count != 0 {
        return false;
    }
    if config.full_state_positions.is_empty() {
        return config.fixed_state_values.is_empty();
    }
    if config.full_state_positions.len() != state_size {
        return false;
    }

    let mut occupied = vec![false; full_size];
    for &position in &config.full_state_positions {
        if position >= full_size || occupied[position] {
            return false;
        }
        occupied[position] = true;
    }
    for &(position, value) in &config.fixed_state_values {
        if position >= full_size || occupied[position] || !value.is_finite() {
            return false;
        }
        occupied[position] = true;
    }
    occupied.into_iter().all(|position| position)
}

fn validate_expr_problem(
    problem: &CorrectionNewtonProblem,
    options: &CorrectionNewtonOptions,
) -> Result<(), CorrectionNewtonSolverError> {
    let state_count = problem.state_names.len();
    let valid = state_count > 0
        && problem.equations.len() == state_count
        && problem.initial_guess.nrows() == state_count
        && problem.initial_guess.ncols() == problem.steps
        && problem.steps > 0
        && !problem.independent_variable.trim().is_empty()
        && problem.interval.0.is_finite()
        && problem.interval.1.is_finite()
        && problem.interval.0 < problem.interval.1
        && options.absolute_tolerance.is_finite()
        && options.absolute_tolerance > 0.0
        && options.default_relative_tolerance.is_finite()
        && options.default_relative_tolerance > 0.0
        && options.max_iterations > 0
        && options.max_damping_trials > 0
        && problem.initial_guess.iter().all(|value| value.is_finite())
        && options
            .relative_tolerances
            .values()
            .all(|value| value.is_finite() && *value > 0.0)
        && options
            .bounds
            .values()
            .all(|(lower, upper)| !lower.is_nan() && !upper.is_nan() && lower <= upper);
    if valid {
        Ok(())
    } else {
        Err(CorrectionNewtonSolverError::InvalidConfiguration {
            source: CorrectionNewtonConfigurationError::InvalidSymbolicProblem,
        })
    }
}
