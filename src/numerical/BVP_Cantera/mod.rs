//! Standalone fixed-mesh Cantera-style damped Newton solver.
//!
//! The numerical iteration is isolated from [`super::BVP_Damp`]. The current
//! numerical core accepts reusable coordinate callbacks with Dense, Sparse or
//! automatically inferred Banded linear workspaces. Symbolic [`Expr`]
//! preparation supports ExprLegacy/AtomView and Lambdify/AOT execution, and
//! prepares a fixed trapezoid coordinate pattern without constructing another
//! solver object. Adaptive mesh refinement and pseudo-transient recovery are
//! intentionally outside this fixed-mesh route.
//!
//! The implementation follows the pinned Cantera `MultiNewton` core contract:
//! weighted RMS correction norm, fraction-to-boundary restriction, Jacobian
//! age/reuse, and a fixed `sqrt(2)` damping reduction. Those choices are kept
//! local here and are checked against independent transition-contract tests.

use crate::numerical::logging::{
    solver_debug, solver_info, solver_warn, SolverLogGuard, SolverLogLevel, SolverLoggingError,
};
use crate::numerical::BVP_sci::new::BvpSciNewError;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp_generated::SymbolicIvpGeneratedBackendConfig;
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use thiserror::Error;

mod linear;
pub use linear::{CanteraLinearBackend, CanteraMatrixLayout};
use linear::{CanteraLinearError, CanteraLinearWorkspace};
mod jacobian;
pub use jacobian::{CanteraJacobianError, CanteraJacobianPattern};
mod parameter_loop;
mod symbolic_plan;
pub use parameter_loop::{
    CanteraParameterLoopOutcome, CanteraParameterLoopPoint, CanteraParameterLoopReport,
};

#[cfg(test)]
#[path = "tests/mod.rs"]
mod tests;

/// Residual callback for one packed, fixed-mesh state vector.
pub type ResidualCallback = dyn FnMut(&DVector<f64>) -> Result<DVector<f64>, String>;
/// Dense Jacobian callback for one packed, fixed-mesh state vector.
pub type JacobianCallback = dyn FnMut(&DVector<f64>) -> Result<DMatrix<f64>, String>;
/// Coordinate Jacobian callback that fills the prepared pattern's value buffer.
pub type CoordinateJacobianCallback = dyn FnMut(&DVector<f64>, &mut [f64]) -> Result<(), String>;

enum JacobianSource {
    Dense(Box<dyn FnMut(&DVector<f64>, &[f64]) -> Result<DMatrix<f64>, String>>),
    Coordinates {
        pattern: CanteraJacobianPattern,
        values: Vec<f64>,
        callback: Box<dyn FnMut(&DVector<f64>, &[f64], &mut [f64]) -> Result<(), String>>,
    },
    SymbolicCoordinates {
        pattern: CanteraJacobianPattern,
        values: Vec<f64>,
        callback:
            Box<dyn FnMut(&DVector<f64>, &[f64], &mut [f64]) -> Result<(), CanteraCallbackFailure>>,
    },
}

enum ResidualSource {
    Allocating(Box<dyn FnMut(&DVector<f64>, &[f64]) -> Result<DVector<f64>, String>>),
    Buffered(
        Box<
            dyn FnMut(
                &DVector<f64>,
                &[f64],
                &mut DVector<f64>,
            ) -> Result<(), CanteraCallbackFailure>,
        >,
    ),
}

#[derive(Debug)]
enum JacobianSourceError {
    Callback(CanteraCallbackFailure),
    Assembly(CanteraJacobianError),
}

#[derive(Debug)]
pub(super) enum CanteraCallbackFailure {
    External(String),
    Symbolic(BvpSciNewError),
}

impl CanteraCallbackFailure {
    fn message(&self) -> String {
        match self {
            Self::External(message) => message.clone(),
            Self::Symbolic(source) => source.to_string(),
        }
    }

    fn symbolic_source(&self) -> Option<BvpSciNewError> {
        match self {
            Self::External(_) => None,
            Self::Symbolic(source) => Some(source.clone()),
        }
    }
}

impl From<String> for CanteraCallbackFailure {
    fn from(message: String) -> Self {
        Self::External(message)
    }
}

enum EvaluatedJacobian<'a> {
    Dense(DMatrix<f64>),
    Coordinates { values: &'a [f64] },
}

fn map_linear_error(error: CanteraLinearError, report: CanteraSolveReport) -> CanteraSolverError {
    match error {
        CanteraLinearError::Singular { backend } => {
            CanteraSolverError::SingularJacobian { backend, report }
        }
        CanteraLinearError::ResidualCheckFailed {
            backend,
            residual_norm,
            tolerance,
        } => CanteraSolverError::LinearResidualCheckFailed {
            backend,
            residual_norm,
            tolerance,
            report,
        },
        CanteraLinearError::NonFiniteSolution { backend } => {
            CanteraSolverError::NonFiniteLinearSolution { backend, report }
        }
        CanteraLinearError::Backend {
            backend,
            message,
            source,
        } => CanteraSolverError::LinearBackend {
            backend,
            message,
            source,
            report,
        },
    }
}

impl JacobianSource {
    fn pattern(&self) -> Option<&CanteraJacobianPattern> {
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
                let matrix = callback(state, external_parameters)
                    .map_err(CanteraCallbackFailure::External)
                    .map_err(JacobianSourceError::Callback)?;
                if matrix.shape() != (state.len(), state.len())
                    || !matrix.iter().all(|value| value.is_finite())
                {
                    return Err(JacobianSourceError::Callback(
                        CanteraCallbackFailure::External(format!(
                            "callback returned shape {}x{} for state size {} (finite={})",
                            matrix.nrows(),
                            matrix.ncols(),
                            state.len(),
                            matrix.iter().all(|value| value.is_finite())
                        )),
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
                    .map_err(CanteraCallbackFailure::External)
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

/// Symbolic fixed-mesh BVP input for the Cantera Newton route.
#[derive(Clone, Debug)]
pub struct CanteraExprBvpProblem {
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
pub enum CanteraSymbolicFrontend {
    /// Established Expr-based derivative and Lambdify route.
    #[default]
    ExprLegacy,
    /// Native packed-Atom derivative and Lambdify route.
    AtomView,
}

/// Symbolic callback implementation selected during preparation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CanteraSymbolicExecution {
    /// Compile reusable Lambdify callbacks during solver preparation.
    #[default]
    Lambdify,
    /// Build or load the shared generated native callback artifact.
    Aot,
}

/// Controls for the selected symbolic pilot route and its Newton core.
#[derive(Clone, Debug)]
pub struct CanteraExprBvpOptions {
    /// Symbolic representation used to prepare RHS and Jacobian callbacks.
    pub symbolic_frontend: CanteraSymbolicFrontend,
    /// Callback implementation. AOT requires an explicit generated-backend
    /// configuration supplied through [`Self::with_aot_config`].
    pub symbolic_execution: CanteraSymbolicExecution,
    /// Shared AOT compiler/cache lifecycle configuration, when AOT is selected.
    pub aot_config: Option<SymbolicIvpGeneratedBackendConfig>,
    /// Linear storage selected for the symbolic fixed-mesh Jacobian.
    pub matrix_layout: CanteraMatrixLayout,
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
    pub telemetry_mode: CanteraTelemetryMode,
}

impl Default for CanteraExprBvpOptions {
    fn default() -> Self {
        Self {
            symbolic_frontend: CanteraSymbolicFrontend::ExprLegacy,
            symbolic_execution: CanteraSymbolicExecution::Lambdify,
            aot_config: None,
            matrix_layout: CanteraMatrixLayout::Dense,
            absolute_tolerance: 1e-8,
            relative_tolerances: HashMap::new(),
            default_relative_tolerance: 1e-8,
            bounds: HashMap::new(),
            max_iterations: 100,
            max_jacobian_age: 5,
            max_damping_trials: 7,
            logging_level: SolverLogLevel::Off,
            telemetry_mode: CanteraTelemetryMode::Off,
        }
    }
}

impl CanteraExprBvpOptions {
    /// Select generated callbacks with the crate-wide AOT build/cache policy.
    pub fn with_aot_config(mut self, config: SymbolicIvpGeneratedBackendConfig) -> Self {
        self.symbolic_execution = CanteraSymbolicExecution::Aot;
        self.aot_config = Some(config);
        self
    }
}

/// Detail level for standalone Cantera-core measurements.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CanteraTelemetryMode {
    /// Do not collect counters or durations.
    #[default]
    Off,
    /// Collect operation counters without clocks.
    Counters,
    /// Collect counters and cumulative callback/linear-solve durations.
    Timings,
}

/// Configuration for a fixed-mesh Cantera-style Newton solve.
#[derive(Clone, Debug)]
pub struct CanteraNewtonConfig {
    /// Linear storage; Sparse/BandedAuto require a coordinate Jacobian callback.
    pub matrix_layout: CanteraMatrixLayout,
    /// Number of interleaved physical components in the node-major state.
    ///
    /// Cantera scales each component using its average magnitude over all mesh
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
    pub telemetry_mode: CanteraTelemetryMode,
}

impl CanteraNewtonConfig {
    /// Creates scalar tolerances and unbounded unknowns for a state of `size`.
    pub fn new(size: usize, absolute_tolerance: f64, relative_tolerance: f64) -> Self {
        Self {
            matrix_layout: CanteraMatrixLayout::Dense,
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
            telemetry_mode: CanteraTelemetryMode::Off,
        }
    }
}

/// Counters and optional durations for one solve attempt, including failures.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CanteraTelemetrySnapshot {
    /// Number of residual callback invocations.
    pub residual_evaluations: usize,
    /// Number of Jacobian callback invocations.
    pub jacobian_evaluations: usize,
    /// Number of linear solves, including rejected damping trials.
    pub linear_solves: usize,
    /// Number of Jacobian factorizations.
    pub factorizations: usize,
    /// Number of trial states evaluated by the damping loop.
    pub damping_trials: usize,
    /// Number of rejected damping trials.
    pub damping_rejections: usize,
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
pub struct CanteraSolveReport {
    /// Last accepted packed state; a failed report is diagnostic, not a success result.
    pub state: DVector<f64>,
    /// Effective linear backend selected for this solve.
    pub linear_backend: CanteraLinearBackend,
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
    pub telemetry: CanteraTelemetrySnapshot,
}

/// Typed failure with a partial report for the attempted solve.
#[derive(Debug, Error)]
pub enum CanteraSolverError {
    /// Invalid dimensions or non-finite/invalid solver configuration.
    #[error("invalid Cantera BVP configuration: {message}")]
    InvalidConfiguration { message: String },
    /// Symbolic equations declare fixed inputs that have not been bound yet.
    #[error("Cantera BVP external parameters have not been bound: {names:?}")]
    MissingParameterValues { names: Vec<String> },
    /// The selected symbolic frontend failed during preparation.
    #[error("Cantera BVP symbolic preparation failed: {source}")]
    SymbolicPreparation {
        #[source]
        source: BvpSciNewError,
    },
    /// The frontend returned successfully but did not provide its required callback.
    #[error("Cantera BVP symbolic frontend contract failed: {message}")]
    FrontendContract { message: String },
    /// The shared logging infrastructure could not be entered.
    #[error("Cantera BVP logging setup failed: {source}")]
    Logging {
        #[source]
        source: SolverLoggingError,
    },
    /// Residual callback failed; `report` retains completed work.
    #[error("Cantera BVP residual callback failed: {message}")]
    ResidualCallback {
        message: String,
        #[source]
        source: Option<BvpSciNewError>,
        report: CanteraSolveReport,
    },
    /// Jacobian callback failed; `report` retains completed work.
    #[error("Cantera BVP Jacobian callback failed: {message}")]
    JacobianCallback {
        message: String,
        #[source]
        source: Option<BvpSciNewError>,
        report: CanteraSolveReport,
    },
    /// A coordinate callback produced values invalid for its fixed pattern.
    #[error("Cantera BVP Jacobian assembly failed: {source}")]
    JacobianAssembly {
        #[source]
        source: CanteraJacobianError,
        report: CanteraSolveReport,
    },
    /// The selected backend found a singular Jacobian at the current iterate.
    #[error("Cantera BVP {backend} Jacobian is singular")]
    SingularJacobian {
        backend: &'static str,
        report: CanteraSolveReport,
    },
    /// A backend returned successfully, but its solution failed backward-error validation.
    #[error(
        "Cantera BVP {backend} linear solve residual {residual_norm:e} exceeds allowed {tolerance:e}"
    )]
    LinearResidualCheckFailed {
        backend: &'static str,
        residual_norm: f64,
        tolerance: f64,
        report: CanteraSolveReport,
    },
    /// A backend returned a solution containing NaN or infinity.
    #[error("Cantera BVP {backend} linear solve produced a non-finite solution")]
    NonFiniteLinearSolution {
        backend: &'static str,
        report: CanteraSolveReport,
    },
    /// The configured linear backend failed during factorization or solve.
    #[error("Cantera BVP {backend} linear backend failed: {message}")]
    LinearBackend {
        backend: &'static str,
        message: String,
        #[source]
        source: Option<BvpSciNewError>,
        report: CanteraSolveReport,
    },
    /// The undamped Newton step points out of bounds and no safe fraction remains.
    #[error("Cantera BVP Newton step is blocked by state bounds")]
    BoundsBlocked { report: CanteraSolveReport },
    /// The weighted RMS correction could not be represented as a finite value.
    #[error("Cantera BVP produced a non-finite weighted correction norm")]
    NonFiniteCorrectionNorm { report: CanteraSolveReport },
    /// Newton stopped without satisfying the Cantera weighted correction test.
    #[error(
        "Cantera BVP Newton did not converge after {} iterations",
        .report.iterations
    )]
    NonConvergence { report: CanteraSolveReport },
}

impl CanteraSolverError {
    /// Partial numerical report, when failure happened after Newton began.
    pub fn partial_report(&self) -> Option<&CanteraSolveReport> {
        match self {
            Self::ResidualCallback { report, .. }
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
pub struct CanteraNewtonSolver {
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
    linear: CanteraLinearWorkspace,
    config: CanteraNewtonConfig,
    external_parameter_names: Vec<String>,
    external_parameter_values: Vec<f64>,
    parameter_binding_order: Vec<usize>,
    external_parameters_bound: bool,
}

impl CanteraNewtonSolver {
    /// Builds a callback-driven numerical core from an initial packed state.
    pub fn new<R, J>(
        initial_state: DVector<f64>,
        residual: R,
        jacobian: J,
        config: CanteraNewtonConfig,
    ) -> Result<Self, CanteraSolverError>
    where
        R: FnMut(&DVector<f64>) -> Result<DVector<f64>, String> + 'static,
        J: FnMut(&DVector<f64>) -> Result<DMatrix<f64>, String> + 'static,
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
        pattern: CanteraJacobianPattern,
        jacobian: J,
        config: CanteraNewtonConfig,
    ) -> Result<Self, CanteraSolverError>
    where
        R: FnMut(&DVector<f64>) -> Result<DVector<f64>, String> + 'static,
        J: FnMut(&DVector<f64>, &mut [f64]) -> Result<(), String> + 'static,
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
        pattern: CanteraJacobianPattern,
        jacobian: J,
        config: CanteraNewtonConfig,
    ) -> Result<Self, CanteraSolverError>
    where
        R: FnMut(&DVector<f64>, &mut DVector<f64>) -> Result<(), CanteraCallbackFailure> + 'static,
        J: FnMut(&DVector<f64>, &mut [f64]) -> Result<(), CanteraCallbackFailure> + 'static,
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
        pattern: CanteraJacobianPattern,
        jacobian: J,
        parameter_names: Vec<String>,
        config: CanteraNewtonConfig,
    ) -> Result<Self, CanteraSolverError>
    where
        R: FnMut(&DVector<f64>, &[f64], &mut DVector<f64>) -> Result<(), CanteraCallbackFailure>
            + 'static,
        J: FnMut(&DVector<f64>, &[f64], &mut [f64]) -> Result<(), CanteraCallbackFailure> + 'static,
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
        config: CanteraNewtonConfig,
    ) -> Result<Self, CanteraSolverError> {
        let size = initial_state.len();
        if let JacobianSource::Coordinates { pattern, .. }
        | JacobianSource::SymbolicCoordinates { pattern, .. } = &jacobian
        {
            if pattern.dimension() != size {
                return Err(CanteraSolverError::InvalidConfiguration {
                    message: format!(
                        "coordinate Jacobian dimension {} does not match state size {size}",
                        pattern.dimension()
                    ),
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
            return Err(CanteraSolverError::InvalidConfiguration {
                message: "state, tolerances, bounds, and iteration limits are inconsistent".into(),
            });
        }
        let linear = CanteraLinearWorkspace::new(config.matrix_layout, size, jacobian.pattern())
            .map_err(|error| CanteraSolverError::InvalidConfiguration {
                message: error.to_string(),
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
        })
    }

    fn with_parameter_schema(
        mut self,
        parameter_names: Vec<String>,
    ) -> Result<Self, CanteraSolverError> {
        if parameter_names.iter().any(|name| name.trim().is_empty())
            || parameter_names
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len()
                != parameter_names.len()
        {
            return Err(CanteraSolverError::InvalidConfiguration {
                message: "external parameter names must be non-empty and unique".into(),
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
        problem: CanteraExprBvpProblem,
        options: CanteraExprBvpOptions,
    ) -> Result<Self, CanteraSolverError> {
        symbolic_plan::prepare(problem, options)
    }

    /// Legacy spelling for the configured symbolic frontend.
    pub fn from_expr_legacy(
        problem: CanteraExprBvpProblem,
        options: CanteraExprBvpOptions,
    ) -> Result<Self, CanteraSolverError> {
        Self::from_symbolic(problem, options)
    }

    /// Dense-compatible spelling retained for clients of the initial pilot.
    ///
    /// New callers should use [`Self::from_symbolic`] and select their
    /// matrix layout through [`CanteraExprBvpOptions`].
    pub fn from_expr_legacy_dense(
        problem: CanteraExprBvpProblem,
        mut options: CanteraExprBvpOptions,
    ) -> Result<Self, CanteraSolverError> {
        options.matrix_layout = CanteraMatrixLayout::Dense;
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
    ) -> Result<(), CanteraSolverError> {
        if names.len() != values.len()
            || values.iter().any(|value| !value.is_finite())
            || names.iter().any(|name| name.trim().is_empty())
        {
            return Err(CanteraSolverError::InvalidConfiguration {
                message: "external parameter names and finite values must have equal lengths"
                    .into(),
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
            return Err(CanteraSolverError::InvalidConfiguration {
                message: format!(
                    "bound external parameter names must match the prepared schema {:?}",
                    self.external_parameter_names
                ),
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
    pub fn set_parameter_values(&mut self, values: Vec<f64>) -> Result<(), CanteraSolverError> {
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

    fn validate_parameter_values(&self, values: &[f64]) -> Result<(), CanteraSolverError> {
        if !self.external_parameters_bound {
            return Err(CanteraSolverError::MissingParameterValues {
                names: self.external_parameter_names.clone(),
            });
        }
        if values.len() != self.parameter_binding_order.len()
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(CanteraSolverError::InvalidConfiguration {
                message: format!(
                    "expected {} finite external parameter values",
                    self.parameter_binding_order.len()
                ),
            });
        }
        Ok(())
    }

    /// Expand a reduced Cantera Newton state into node-major grid values.
    ///
    /// This allocates the public profile once after a solve; it is not used in
    /// the Newton hot path. Generic callback systems without an explicit map
    /// are treated as one complete row.
    pub fn full_grid_values(
        &self,
        state: &DVector<f64>,
    ) -> Result<DMatrix<f64>, CanteraSolverError> {
        let component_count = self.config.component_count;
        let full_size = state
            .len()
            .checked_add(self.config.fixed_state_values.len())
            .ok_or_else(|| CanteraSolverError::InvalidConfiguration {
                message: "expanded Cantera state size overflowed".into(),
            })?;
        if component_count == 0 || full_size == 0 || full_size % component_count != 0 {
            return Err(CanteraSolverError::InvalidConfiguration {
                message: "expanded Cantera state is incompatible with the component count".into(),
            });
        }
        let node_count = full_size / component_count;
        let mut values = DMatrix::from_element(node_count, component_count, f64::NAN);
        if self.config.full_state_positions.is_empty() {
            if state.len() != full_size {
                return Err(CanteraSolverError::InvalidConfiguration {
                    message: "incomplete Cantera grid mapping".into(),
                });
            }
            for (index, &value) in state.iter().enumerate() {
                values[(index / component_count, index % component_count)] = value;
            }
        } else {
            if self.config.full_state_positions.len() != state.len() {
                return Err(CanteraSolverError::InvalidConfiguration {
                    message: "Cantera grid mapping length does not match the reduced state".into(),
                });
            }
            for (&position, &value) in self.config.full_state_positions.iter().zip(state.iter()) {
                if position >= full_size {
                    return Err(CanteraSolverError::InvalidConfiguration {
                        message: "Cantera grid mapping contains an out-of-range position".into(),
                    });
                }
                values[(position / component_count, position % component_count)] = value;
            }
            for &(position, value) in &self.config.fixed_state_values {
                if position >= full_size {
                    return Err(CanteraSolverError::InvalidConfiguration {
                        message: "fixed Cantera boundary position is out of range".into(),
                    });
                }
                values[(position / component_count, position % component_count)] = value;
            }
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(CanteraSolverError::InvalidConfiguration {
                message: "expanded Cantera grid contains unmapped or non-finite values".into(),
            });
        }
        Ok(values)
    }

    /// Runs the standalone Cantera-style fixed-mesh Newton loop.
    pub fn solve(&mut self) -> Result<CanteraSolveReport, CanteraSolverError> {
        if !self.external_parameters_bound {
            return Err(CanteraSolverError::MissingParameterValues {
                names: self.external_parameter_names.clone(),
            });
        }
        let _logging = SolverLogGuard::try_enter(self.config.logging_level)
            .map_err(|source| CanteraSolverError::Logging { source })?;
        let mut report = self.empty_report();
        let mode = self.config.telemetry_mode;
        let mut jacobian_age = 0_usize;
        let mut force_jacobian = true;
        let mut jacobian_reevaluations_after_failed_damping = 0_usize;

        for iteration in 0..self.config.max_iterations {
            report.iterations = iteration + 1;
            // Cantera refreshes only once age is strictly greater than max age.
            if force_jacobian || jacobian_age > self.config.max_jacobian_age {
                let previous_jacobian_age = jacobian_age;
                let refresh_reason = if force_jacobian {
                    "damping retry"
                } else {
                    "maximum age"
                };
                // Unlike Cantera's MultiJac finite-difference assembly, this
                // pilot receives an already assembled symbolic Jacobian.
                let start = (mode == CanteraTelemetryMode::Timings).then(Instant::now);
                if mode != CanteraTelemetryMode::Off {
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
                    JacobianSourceError::Callback(source) => CanteraSolverError::JacobianCallback {
                        message: source.message(),
                        source: source.symbolic_source(),
                        report: report.clone(),
                    },
                    JacobianSourceError::Assembly(source) => CanteraSolverError::JacobianAssembly {
                        source,
                        report: report.clone(),
                    },
                })?;
                let factor_result = match evaluated {
                    EvaluatedJacobian::Dense(matrix) => {
                        let factor_start =
                            (mode == CanteraTelemetryMode::Timings).then(Instant::now);
                        let result = self.linear.factor_dense(matrix);
                        if let Some(start) = factor_start {
                            report.telemetry.factorization_time += start.elapsed();
                        }
                        result
                    }
                    EvaluatedJacobian::Coordinates { values } => {
                        let assembly_start =
                            (mode == CanteraTelemetryMode::Timings).then(Instant::now);
                        let assembly_result = self.linear.assemble_coordinates(values);
                        if let Some(start) = assembly_start {
                            report.telemetry.jacobian_assembly_time += start.elapsed();
                        }
                        assembly_result.map_err(|error| map_linear_error(error, report.clone()))?;

                        let factor_start =
                            (mode == CanteraTelemetryMode::Timings).then(Instant::now);
                        let result = self.linear.factor_assembled();
                        if let Some(start) = factor_start {
                            report.telemetry.factorization_time += start.elapsed();
                        }
                        result
                    }
                };
                factor_result.map_err(|error| map_linear_error(error, report.clone()))?;
                if mode != CanteraTelemetryMode::Off {
                    report.telemetry.factorizations += 1;
                }
                jacobian_age = 0;
                force_jacobian = false;
                solver_debug!(
                    "Cantera BVP refreshed {} Jacobian at iteration {} (reason={}, previous age={})",
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
            .ok_or_else(|| CanteraSolverError::NonFiniteCorrectionNorm {
                report: report.clone(),
            })?;
            report.correction_norm = Some(s0);
            jacobian_age += 1;

            let bound_factor = self.bound_factor(&self.base_step_scratch);
            report.last_bound_factor = Some(bound_factor);
            if bound_factor < 1.0 && mode != CanteraTelemetryMode::Off {
                report.telemetry.bound_limited_steps += 1;
            }
            // MultiNewton reports this as damping status -3. The caller still
            // gets its stale-Jacobian retry opportunity before terminal failure.
            let bounds_blocked = bound_factor < 1e-10;
            if bounds_blocked {
                solver_warn!(
                    "Cantera BVP iteration {} blocked by bounds: fraction={:.6e}",
                    iteration + 1,
                    bound_factor
                );
            }

            let mut alpha = bound_factor;
            let mut accepted = false;
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
                if mode != CanteraTelemetryMode::Off {
                    report.telemetry.damping_trials += 1;
                }
                Self::eval_residual(
                    &mut self.residual,
                    &mut self.residual_scratch,
                    &self.trial_state_scratch,
                    &self.external_parameter_values,
                    &mut report,
                    mode,
                )?;
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
                .ok_or_else(|| CanteraSolverError::NonFiniteCorrectionNorm {
                    report: report.clone(),
                })?;
                let improves = s1 < 1.0 || s1 < s0;
                solver_debug!(
                    "Cantera BVP iteration {} damping trial {}: bound={:.6e}, alpha={:.6e}, correction={:.6e}->{:.6e}, accepted={}",
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
                if mode != CanteraTelemetryMode::Off {
                    report.telemetry.damping_rejections += 1;
                }
                alpha /= 2.0_f64.sqrt();
            }

            if accepted {
                self.state.copy_from(&self.trial_state_scratch);
                report.state.clone_from(&self.state);
                if mode != CanteraTelemetryMode::Off {
                    report.telemetry.accepted_steps += 1;
                }
                let correction_norm = report.correction_norm.expect("accepted trial has norm");
                solver_info!(
                    "Cantera BVP Newton iteration {} accepted alpha={:.6e}, weighted correction={:.6e}",
                    iteration + 1,
                    alpha,
                    correction_norm
                );
                if correction_norm <= 1.0 {
                    self.last_successful_checkpoint = Some(self.capture_checkpoint());
                    return Ok(report);
                }
            // Cantera allows four stale-Jacobian refresh retries: it breaks only
            // after its retry counter has become greater than three.
            } else if jacobian_age > 1 && jacobian_reevaluations_after_failed_damping <= 3 {
                force_jacobian = true;
                jacobian_reevaluations_after_failed_damping += 1;
                let failure_kind = if bounds_blocked {
                    "bounds-blocked step"
                } else {
                    "damping exhaustion"
                };
                solver_info!(
                    "Cantera BVP iteration {} had {}; refreshing reused Jacobian (retry {})",
                    iteration + 1,
                    failure_kind,
                    jacobian_reevaluations_after_failed_damping
                );
            } else {
                let failure_kind = if bounds_blocked {
                    "bounds-blocked step"
                } else {
                    "damping exhaustion"
                };
                solver_warn!(
                    "Cantera BVP stopped after {failure_kind} at iteration {} (Jacobian age {})",
                    iteration + 1,
                    jacobian_age
                );
                return if bounds_blocked {
                    Err(CanteraSolverError::BoundsBlocked { report })
                } else {
                    Err(CanteraSolverError::NonConvergence { report })
                };
            }
        }

        // The last accepted trial already supplied this state's residual norm;
        // do not call the residual again just to format a diagnostic report.
        solver_warn!(
            "Cantera BVP reached the iteration limit ({}) without convergence",
            self.config.max_iterations
        );
        Err(CanteraSolverError::NonConvergence { report })
    }

    fn empty_report(&self) -> CanteraSolveReport {
        CanteraSolveReport {
            state: self.state.clone(),
            linear_backend: self.linear.backend(),
            iterations: 0,
            correction_norm: None,
            residual_norm: None,
            last_bound_factor: None,
            last_damping_factor: None,
            telemetry: CanteraTelemetrySnapshot::default(),
        }
    }

    fn eval_residual(
        residual: &mut ResidualSource,
        residual_scratch: &mut DVector<f64>,
        state: &DVector<f64>,
        external_parameters: &[f64],
        report: &mut CanteraSolveReport,
        telemetry_mode: CanteraTelemetryMode,
    ) -> Result<(), CanteraSolverError> {
        let start = (telemetry_mode == CanteraTelemetryMode::Timings).then(Instant::now);
        if telemetry_mode != CanteraTelemetryMode::Off {
            report.telemetry.residual_evaluations += 1;
        }
        let callback_result = match (&mut *residual, &mut *residual_scratch) {
            (ResidualSource::Allocating(callback), output) => callback(state, external_parameters)
                .map_err(CanteraCallbackFailure::External)
                .and_then(|value| {
                    if value.len() != output.len() {
                        return Err(CanteraCallbackFailure::External(
                            "callback returned a residual with the wrong size".into(),
                        ));
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
        callback_result.map_err(|source| CanteraSolverError::ResidualCallback {
            message: source.message(),
            source: source.symbolic_source(),
            report: report.clone(),
        })?;
        if residual_scratch.len() != state.len() || !residual_scratch.iter().all(|x| x.is_finite())
        {
            return Err(CanteraSolverError::ResidualCallback {
                message: "callback returned a wrong-sized or non-finite residual".into(),
                source: None,
                report: report.clone(),
            });
        }
        Ok(())
    }

    fn solve_linear(
        factorization: &mut CanteraLinearWorkspace,
        rhs: &[f64],
        solution: &mut DVector<f64>,
        negate_rhs: bool,
        report: &mut CanteraSolveReport,
        telemetry_mode: CanteraTelemetryMode,
    ) -> Result<(), CanteraSolverError> {
        let start = (telemetry_mode == CanteraTelemetryMode::Timings).then(Instant::now);
        if rhs.len() != solution.len() {
            return Err(CanteraSolverError::LinearBackend {
                backend: factorization.backend().name(),
                message: format!(
                    "linear RHS length {} does not match solver scratch length {}",
                    rhs.len(),
                    solution.len()
                ),
                source: None,
                report: report.clone(),
            });
        }
        solution.copy_from_slice(rhs);
        if negate_rhs {
            solution.scale_mut(-1.0);
        }
        let solve_result = factorization.solve_in_place(solution.as_mut_slice());
        if let Some(start) = start {
            report.telemetry.linear_solve_time += start.elapsed();
        }
        if telemetry_mode != CanteraTelemetryMode::Off {
            report.telemetry.linear_solves += 1;
        }
        match solve_result {
            Ok(()) if solution.iter().all(|value| value.is_finite()) => Ok(()),
            Err(CanteraLinearError::Singular { backend }) => {
                Err(CanteraSolverError::SingularJacobian {
                    backend,
                    report: report.clone(),
                })
            }
            Err(CanteraLinearError::ResidualCheckFailed {
                backend,
                residual_norm,
                tolerance,
            }) => Err(CanteraSolverError::LinearResidualCheckFailed {
                backend,
                residual_norm,
                tolerance,
                report: report.clone(),
            }),
            Err(CanteraLinearError::NonFiniteSolution { backend }) => {
                Err(CanteraSolverError::NonFiniteLinearSolution {
                    backend,
                    report: report.clone(),
                })
            }
            Err(CanteraLinearError::Backend {
                backend,
                message,
                source,
            }) => Err(CanteraSolverError::LinearBackend {
                backend,
                message,
                source,
                report: report.clone(),
            }),
            Ok(()) => Err(CanteraSolverError::NonFiniteLinearSolution {
                backend: factorization.backend().name(),
                report: report.clone(),
            }),
        }
    }

    fn weighted_rms(
        config: &CanteraNewtonConfig,
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
            // Match Cantera's norm2: component weights use the mean magnitude
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

    fn full_state_position(config: &CanteraNewtonConfig, packed_index: usize) -> usize {
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

struct PackedBvpInitialState {
    values: DVector<f64>,
    full_state_positions: Vec<usize>,
    fixed_state_values: Vec<(usize, f64)>,
}

fn pack_bvp_initial_state(
    problem: &CanteraExprBvpProblem,
) -> Result<PackedBvpInitialState, CanteraSolverError> {
    let component_count = problem.state_names.len();
    let full_points =
        problem
            .steps
            .checked_add(1)
            .ok_or_else(|| CanteraSolverError::InvalidConfiguration {
                message: "BVP full-grid point count overflowed".into(),
            })?;
    let mut fixed_by_component = vec![None; component_count];

    for (name, conditions) in &problem.boundary_conditions {
        let Some(component) = problem.state_names.iter().position(|state| state == name) else {
            return Err(CanteraSolverError::InvalidConfiguration {
                message: format!("boundary condition refers to unknown state '{name}'"),
            });
        };
        if conditions.len() != 1 {
            return Err(CanteraSolverError::InvalidConfiguration {
                message: format!(
                    "state '{name}' must have exactly one endpoint condition for a first-order BVP"
                ),
            });
        }
        let (side, value) = conditions[0];
        if side > 1 || !value.is_finite() {
            return Err(CanteraSolverError::InvalidConfiguration {
                message: format!(
                    "state '{name}' has invalid endpoint condition (side={side}, value={value})"
                ),
            });
        }
        let node = if side == 0 { 0 } else { problem.steps };
        fixed_by_component[component] = Some((node, value));
    }
    if fixed_by_component.iter().any(Option::is_none) {
        return Err(CanteraSolverError::InvalidConfiguration {
            message: "each state must have exactly one left or right endpoint condition".into(),
        });
    }
    if problem.boundary_conditions.len() != component_count {
        return Err(CanteraSolverError::InvalidConfiguration {
            message: "boundary-condition keys must correspond one-to-one with states".into(),
        });
    }

    let full_size = component_count.checked_mul(full_points).ok_or_else(|| {
        CanteraSolverError::InvalidConfiguration {
            message: "BVP full-grid state size overflowed".into(),
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
                return Err(CanteraSolverError::InvalidConfiguration {
                    message: format!(
                        "free mesh node {node} for state '{name}' cannot be mapped into the initial guess"
                    ),
                });
            }
            values.push(problem.initial_guess[(component, guess_column)]);
            full_state_positions.push(full_position);
        }
    }

    if values.len() != component_count * problem.steps
        || values.len() != problem.initial_guess.len()
    {
        return Err(CanteraSolverError::InvalidConfiguration {
            message: format!(
                "reduced BVP state has {} free values but the guess contains {} values",
                values.len(),
                problem.initial_guess.len()
            ),
        });
    }
    Ok(PackedBvpInitialState {
        values: DVector::from_vec(values),
        full_state_positions,
        fixed_state_values,
    })
}

fn valid_full_state_layout(state_size: usize, config: &CanteraNewtonConfig) -> bool {
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
    problem: &CanteraExprBvpProblem,
    options: &CanteraExprBvpOptions,
) -> Result<(), CanteraSolverError> {
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
        Err(CanteraSolverError::InvalidConfiguration {
            message: "symbolic problem dimensions, interval, tolerances, bounds, or iteration limits are invalid".into(),
        })
    }
}
