//! Shared modern symbolic backend for IVP-style ODE solvers.
//!
//! This module sits above the legacy [`crate::symbolic::symbolic_functions::Jacobian`]
//! container and below solver-facing consumers such as:
//! - [`crate::numerical::BE::BE`],
//! - [`crate::numerical::BDF::BDF_api::ODEsolver`],
//! - [`crate::numerical::NR_for_Euler::NRE`].
//!
//! The goal is to give all of those solvers one common contract for:
//! - params-aware symbolic evaluation `f(t, y, p)`,
//! - dense Jacobian evaluation `df/dy`,
//! - typed setup errors instead of ad-hoc panics,
//! - and one prepared AOT bridge that can later be materialized through the
//!   generic codegen lifecycle.

use crate::symbolic::codegen::codegen_aot_runtime_link::{
    LinkedDenseAotBackend, LinkedResidualAotBackend,
};
use crate::symbolic::codegen::codegen_manifest::PreparedProblemManifest;
use crate::symbolic::codegen::codegen_provider_api::{
    BackendKind, MatrixBackend, PreparedDenseProblem,
};
use crate::symbolic::codegen::codegen_runtime_api::{
    DenseJacobianChunkingStrategy, ResidualChunkingStrategy,
};
use crate::symbolic::codegen::codegen_tasks::{IvpJacobianTask, IvpResidualTask};
use crate::symbolic::ivp_telemetry::{
    IvpColdStage, IvpLambdifyExecutionPolicy, IvpTelemetry, IvpTelemetryExecution,
    IvpTelemetryRoute, IvpWarmStage,
};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::View::atom::Atom;
use crate::symbolic::View::conversions::{atom_to_expr, expr_to_atom};
use crate::symbolic::View::evaluate::{
    FunctionMap, PreparedEvaluator, PreparedEvaluatorMetrics, PreparedVariableContext,
};
use crate::symbolic::View::jacobian::{PreparedSparseAtomSystem, SparseAtomJacobianError};
use crate::symbolic::View::state::Symbol;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use std::collections::HashSet;
use std::fmt;
use std::sync::{Arc, RwLock};

/// Shared residual evaluator signature for IVP symbolic backends.
pub type IvpResidualEval = dyn Fn(f64, &DVector<f64>) -> DVector<f64> + Send + Sync;

/// Shared dense Jacobian evaluator signature for IVP symbolic backends.
pub type IvpDenseJacobianEval = dyn Fn(f64, &DVector<f64>) -> DMatrix<f64> + Send + Sync;

/// Fallible residual evaluator used by the typed prepared-runtime boundary.
///
/// The solver-facing [`IvpResidualEval`] remains infallible for compatibility
/// with existing numerical solvers. New callers should prefer the fallible
/// methods on [`PreparedSymbolicIvpProblem`] and
/// [`PreparedSymbolicIvpResidualProblem`].
pub type IvpTryResidualEval =
    dyn Fn(f64, &DVector<f64>) -> Result<DVector<f64>, IvpBackendError> + Send + Sync;

/// Fallible dense Jacobian evaluator used by the typed prepared-runtime
/// boundary. It shares the same compiled closure as the compatibility API.
pub type IvpTryDenseJacobianEval =
    dyn Fn(f64, &DVector<f64>) -> Result<DMatrix<f64>, IvpBackendError> + Send + Sync;

/// Shared parameter storage reused by params-aware IVP evaluators.
pub type SharedIvpParameterValues = Arc<RwLock<DVector<f64>>>;

/// Runs a native callback against the current parameter slice without
/// materializing a temporary `DVector` or flat argument vector. The read lock
/// is held only for the callback's borrowed parameter view; the evaluator
/// never mutates or stores that view.
pub(crate) fn with_shared_parameter_values<T>(
    parameter_values_handle: Option<&SharedIvpParameterValues>,
    on_bound: impl FnOnce(),
    callback: impl FnOnce(&[f64]) -> Result<T, IvpBackendError>,
) -> Result<T, IvpBackendError> {
    match parameter_values_handle {
        Some(handle) => {
            let values = handle
                .read()
                .map_err(|_| IvpBackendError::ParameterStatePoisoned)?;
            on_bound();
            callback(values.as_slice())
        }
        None => {
            on_bound();
            callback(&[])
        }
    }
}

/// Setup/runtime errors for the shared IVP backend layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IvpBackendError {
    /// Parameter names were declared but no initial values were provided.
    MissingParameterValues { expected: usize },
    /// Parameter value vector length does not match declared symbolic names.
    ParameterCountMismatch { expected: usize, actual: usize },
    /// The shared parameter binding was poisoned by a panic in another owner.
    ParameterStatePoisoned,
    /// Time, state, and parameter names do not form a unique callback schema.
    InvalidArgumentSchema { message: String },
    /// High-level generated-backend orchestration failed.
    GeneratedBackendFailure { message: String },
    /// Native AtomView preparation failed before a callback was published.
    AtomPreparationFailure { stage: String, message: String },
    /// Native symbolic differentiation failed at a specific Jacobian entry.
    AtomDifferentiationFailure {
        row: usize,
        col: usize,
        source: crate::symbolic::View::DerivativeError,
    },
    /// Native AtomView callback received an invalid state vector.
    InvalidStateShape { expected: usize, actual: usize },
    /// A caller-owned output buffer has the wrong length for the prepared
    /// residual or Jacobian result.
    InvalidOutputShape {
        stage: String,
        expected: usize,
        actual: usize,
    },
    /// A caller-owned dense matrix has dimensions different from the
    /// prepared Jacobian plan.
    InvalidMatrixShape {
        stage: String,
        expected_rows: usize,
        expected_cols: usize,
        actual_rows: usize,
        actual_cols: usize,
    },
    /// A compact sparse Jacobian pattern contains a coordinate outside its
    /// declared square shape.
    InvalidSparsePatternEntry {
        row: usize,
        col: usize,
        dimension: usize,
    },
    /// A caller selected an output API that does not match the prepared
    /// Jacobian storage plan.
    InvalidJacobianStorage { expected: String, actual: String },
    /// Native AtomView evaluation failed for one residual/Jacobian entry.
    AtomEvaluationFailure {
        stage: String,
        index: usize,
        message: String,
    },
    /// The requested banded storage cannot represent the prepared matrix.
    InvalidBandedStorage {
        rows: usize,
        cols: usize,
        lower: usize,
        upper: usize,
    },
}

impl fmt::Display for IvpBackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingParameterValues { expected } => {
                write!(
                    f,
                    "symbolic IVP backend expected {expected} parameter values, but none were provided"
                )
            }
            Self::ParameterCountMismatch { expected, actual } => {
                write!(
                    f,
                    "symbolic IVP backend expected {expected} parameter values, got {actual}"
                )
            }
            Self::ParameterStatePoisoned => {
                write!(f, "shared IVP parameter state lock is poisoned")
            }
            Self::InvalidArgumentSchema { message } => write!(f, "{message}"),
            Self::GeneratedBackendFailure { message } => write!(f, "{message}"),
            Self::AtomPreparationFailure { stage, message } => {
                write!(f, "AtomView preparation failed during {stage}: {message}")
            }
            Self::AtomDifferentiationFailure { row, col, source } => write!(
                f,
                "AtomView Jacobian differentiation failed at row {row}, column {col}: {source}"
            ),
            Self::InvalidStateShape { expected, actual } => {
                write!(f, "IVP state has length {actual}, expected {expected}")
            }
            Self::InvalidOutputShape {
                stage,
                expected,
                actual,
            } => write!(
                f,
                "IVP {stage} output has length {actual}, expected {expected}"
            ),
            Self::InvalidMatrixShape {
                stage,
                expected_rows,
                expected_cols,
                actual_rows,
                actual_cols,
            } => write!(
                f,
                "IVP {stage} matrix has shape {actual_rows}x{actual_cols}, expected {expected_rows}x{expected_cols}"
            ),
            Self::InvalidSparsePatternEntry {
                row,
                col,
                dimension,
            } => write!(
                f,
                "IVP sparse Jacobian pattern entry ({row}, {col}) is outside {dimension}x{dimension}"
            ),
            Self::InvalidJacobianStorage { expected, actual } => write!(
                f,
                "prepared Jacobian uses {actual} storage, but {expected} output was requested"
            ),
            Self::AtomEvaluationFailure {
                stage,
                index,
                message,
            } => write!(
                f,
                "AtomView {stage} entry {index} evaluation failed: {message}"
            ),
            Self::InvalidBandedStorage {
                rows,
                cols,
                lower,
                upper,
            } => write!(
                f,
                "invalid native banded storage for {rows}x{cols} Jacobian: lower={lower}, upper={upper}"
            ),
        }
    }
}

impl std::error::Error for IvpBackendError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::AtomDifferentiationFailure { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Backend used to evaluate one prepared IVP problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IvpBackendKind {
    /// Existing in-process symbolic lambdify path.
    #[default]
    Lambdify,
    /// Future/optional AOT-generated backend path.
    Aot,
}

/// High-level preparation mode for IVP symbolic backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IvpBackendSelectionPolicy {
    /// Always use the in-process lambdify backend.
    #[default]
    LambdifyOnly,
    /// Prepare the problem for future AOT materialization while keeping the
    /// currently callable backend on the lambdify path.
    PreferAotThenLambdify,
}

/// Symbolic Jacobian assembly backend for IVP preparation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IvpSymbolicAssemblyBackend {
    /// Legacy expression differentiation.
    #[default]
    ExprLegacy,
    /// Native AtomView differentiation and evaluator preparation.
    AtomView,
    /// Test-only compatibility route: AtomView differentiation followed by
    /// `Atom -> Expr` materialization and the legacy Expr closures.
    #[doc(hidden)]
    AtomViewExprCompat,
}

/// One prepared native Atom Jacobian entry. The evaluator owns its immutable
/// compiled IR and can therefore be shared by sequential and Rayon callbacks.
pub(crate) struct PreparedNativeAtomJacobianEntry {
    pub(crate) row: usize,
    pub(crate) col: usize,
    pub(crate) evaluator: PreparedEvaluator,
}

/// Prepared native Atom Jacobian independent of the numerical matrix storage.
pub(crate) struct PreparedNativeAtomJacobian {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) expected_input_len: usize,
    pub(crate) entries: Arc<[PreparedNativeAtomJacobianEntry]>,
}

/// Cold-path shape summary for a native Jacobian evaluator.
///
/// This is intentionally not collected during callbacks. It explains whether
/// a warm-path regression comes from the Atom evaluator interpreter doing more
/// work per entry, rather than from matrix assembly or the linear backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PreparedNativeJacobianMetrics {
    pub(crate) entries: usize,
    pub(crate) total_nodes: usize,
    pub(crate) max_nodes: usize,
    pub(crate) add_nodes: usize,
    pub(crate) mul_nodes: usize,
    pub(crate) powi_nodes: usize,
    pub(crate) pow_nodes: usize,
    pub(crate) builtin_nodes: usize,
    pub(crate) custom_nodes: usize,
}

impl PreparedNativeAtomJacobian {
    pub(crate) fn metrics(&self) -> PreparedNativeJacobianMetrics {
        self.entries.iter().fold(
            PreparedNativeJacobianMetrics::default(),
            |mut total, entry| {
                let metrics: PreparedEvaluatorMetrics = entry.evaluator.metrics();
                total.entries += 1;
                total.total_nodes += metrics.nodes;
                total.max_nodes = total.max_nodes.max(metrics.nodes);
                total.add_nodes += metrics.add_nodes;
                total.mul_nodes += metrics.mul_nodes;
                total.powi_nodes += metrics.powi_nodes;
                total.pow_nodes += metrics.pow_nodes;
                total.builtin_nodes += metrics.builtin_nodes;
                total.custom_nodes += metrics.custom_nodes;
                total
            },
        )
    }

    /// Evaluate native Jacobian entries directly from borrowed IVP segments.
    ///
    /// This is the solver-neutral counterpart of the historical DMatrix
    /// callback.  It keeps Radau and other IVP consumers on the same prepared
    /// Atom plan without forcing a temporary `DVector` or dense matrix.
    pub(crate) fn evaluate_values_into_parts(
        &self,
        time: f64,
        parameters: &[f64],
        state: &[f64],
        execution_policy: IvpLambdifyExecutionPolicy,
        values: &mut [f64],
    ) -> Result<(), IvpBackendError> {
        if values.len() != self.entries.len() {
            return Err(IvpBackendError::InvalidOutputShape {
                stage: "native Jacobian values".to_string(),
                expected: self.entries.len(),
                actual: values.len(),
            });
        }
        let actual_input_len = 1 + parameters.len() + state.len();
        if actual_input_len != self.expected_input_len {
            return Err(IvpBackendError::InvalidStateShape {
                expected: self.expected_input_len.saturating_sub(1 + parameters.len()),
                actual: state.len(),
            });
        }

        let parallel =
            execution_policy.should_parallel_with_tasks(self.entries.len(), self.entries.len());
        if parallel {
            let worker_count = rayon::current_num_threads().max(1);
            let chunk_size = (self.entries.len() + worker_count - 1)
                .checked_div(worker_count)
                .unwrap_or(1)
                .max(1);
            self.entries
                .par_chunks(chunk_size)
                .zip(values.par_chunks_mut(chunk_size))
                .enumerate()
                .try_for_each(|(chunk_index, (entries, values))| {
                    PreparedEvaluator::evaluate_many_thread_local_ivp(
                        entries.iter().map(|entry| &entry.evaluator),
                        time,
                        parameters,
                        state,
                        self.expected_input_len,
                        values,
                    )
                    .map_err(|(index, message)| {
                        IvpBackendError::AtomEvaluationFailure {
                            stage: "Jacobian".to_string(),
                            index: chunk_index * chunk_size + index,
                            message,
                        }
                    })
                })
        } else {
            PreparedEvaluator::evaluate_many_thread_local_ivp(
                self.entries.iter().map(|entry| &entry.evaluator),
                time,
                parameters,
                state,
                self.expected_input_len,
                values,
            )
            .map_err(|(index, message)| IvpBackendError::AtomEvaluationFailure {
                stage: "Jacobian".to_string(),
                index,
                message,
            })
        }
    }
}

/// Immutable native residual runtime shared by owned and caller-owned APIs.
///
/// The evaluator list and parameter binding are read-only from the callback's
/// perspective. Each caller supplies its own output buffer, so parallel
/// evaluation does not need a mutex around reusable result storage.
pub(crate) struct PreparedNativeAtomResidual {
    evaluators: Arc<[PreparedEvaluator]>,
    expected_state_len: usize,
    expected_input_len: usize,
    parameter_values_handle: Option<SharedIvpParameterValues>,
    telemetry: IvpTelemetry,
    execution_policy: IvpLambdifyExecutionPolicy,
}

impl PreparedNativeAtomResidual {
    fn len(&self) -> usize {
        self.evaluators.len()
    }

    fn evaluate_into(
        &self,
        t: f64,
        y: &DVector<f64>,
        out: &mut [f64],
    ) -> Result<(), IvpBackendError> {
        let callback_started = self
            .telemetry
            .start_warm_stage(IvpWarmStage::ResidualCallback);
        if y.len() != self.expected_state_len {
            self.telemetry.record_error();
            self.telemetry
                .record_warm_stage(IvpWarmStage::ResidualCallback, callback_started);
            return Err(IvpBackendError::InvalidStateShape {
                expected: self.expected_state_len,
                actual: y.len(),
            });
        }
        if out.len() != self.evaluators.len() {
            self.telemetry.record_error();
            self.telemetry
                .record_warm_stage(IvpWarmStage::ResidualCallback, callback_started);
            return Err(IvpBackendError::InvalidOutputShape {
                stage: "residual".to_string(),
                expected: self.evaluators.len(),
                actual: out.len(),
            });
        }

        let binding_started = self
            .telemetry
            .start_warm_stage(IvpWarmStage::ArgumentBinding);
        let mut binding_recorded = false;
        let mut evaluate = |parameters: &[f64]| self.evaluate_values(t, parameters, y, out);
        // Most large callback gates are parameter-free. Avoid entering the
        // shared RwLock adapter on that hot path; parameterized continuation
        // still uses the original lock-protected branch below.
        let evaluation = match self.parameter_values_handle.as_ref() {
            Some(handle) => with_shared_parameter_values(
                Some(handle),
                || {
                    self.telemetry
                        .record_warm_stage(IvpWarmStage::ArgumentBinding, binding_started);
                    binding_recorded = true;
                },
                evaluate,
            ),
            None => {
                self.telemetry
                    .record_warm_stage(IvpWarmStage::ArgumentBinding, binding_started);
                binding_recorded = true;
                evaluate(&[])
            }
        };
        if !binding_recorded {
            self.telemetry
                .record_warm_stage(IvpWarmStage::ArgumentBinding, binding_started);
        }
        if let Err(error) = evaluation {
            self.telemetry.record_error();
            self.telemetry
                .record_warm_stage(IvpWarmStage::ResidualCallback, callback_started);
            return Err(error);
        }

        let output_started = self
            .telemetry
            .start_warm_stage(IvpWarmStage::ResidualOutputAssembly);
        self.telemetry
            .record_warm_stage(IvpWarmStage::ResidualOutputAssembly, output_started);
        self.telemetry
            .record_warm_stage(IvpWarmStage::ResidualCallback, callback_started);
        self.telemetry.record_residual_evaluation_count();
        Ok(())
    }

    fn evaluate_values(
        &self,
        t: f64,
        parameters: &[f64],
        y: &DVector<f64>,
        out: &mut [f64],
    ) -> Result<(), IvpBackendError> {
        let parallel = self
            .execution_policy
            .should_parallel_with_tasks(self.evaluators.len(), self.evaluators.len());
        self.telemetry.record_lambdify_dispatch(parallel);
        self.telemetry
            .record_scalar_evaluations(self.evaluators.len());
        let evaluation_started = self
            .telemetry
            .start_warm_stage(IvpWarmStage::ResidualEvaluation);
        let evaluation = if parallel {
            let worker_count = rayon::current_num_threads().max(1);
            let chunk_size = (self.evaluators.len() + worker_count - 1)
                .checked_div(worker_count)
                .unwrap_or(1)
                .max(1);
            self.evaluators
                .par_chunks(chunk_size)
                .zip(out.par_chunks_mut(chunk_size))
                .enumerate()
                .try_for_each(|(chunk_index, (evaluators, values))| {
                    PreparedEvaluator::evaluate_many_thread_local_ivp(
                        evaluators.iter(),
                        t,
                        parameters,
                        y.as_slice(),
                        self.expected_input_len,
                        values,
                    )
                    .map_err(|(index, message)| {
                        IvpBackendError::AtomEvaluationFailure {
                            stage: "residual".to_string(),
                            index: chunk_index * chunk_size + index,
                            message,
                        }
                    })
                })
        } else {
            PreparedEvaluator::evaluate_many_thread_local_ivp(
                self.evaluators.iter(),
                t,
                parameters,
                y.as_slice(),
                self.expected_input_len,
                out,
            )
            .map_err(|(index, message)| IvpBackendError::AtomEvaluationFailure {
                stage: "residual".to_string(),
                index,
                message,
            })
        };
        self.telemetry
            .record_warm_stage(IvpWarmStage::ResidualEvaluation, evaluation_started);
        evaluation
    }

    fn evaluate_values_into_parts(
        &self,
        t: f64,
        parameters: &[f64],
        state: &[f64],
        out: &mut [f64],
    ) -> Result<(), IvpBackendError> {
        if state.len() != self.expected_state_len {
            return Err(IvpBackendError::InvalidStateShape {
                expected: self.expected_state_len,
                actual: state.len(),
            });
        }
        if out.len() != self.evaluators.len() {
            return Err(IvpBackendError::InvalidOutputShape {
                stage: "residual".to_string(),
                expected: self.evaluators.len(),
                actual: out.len(),
            });
        }
        PreparedEvaluator::evaluate_many_thread_local_ivp(
            self.evaluators.iter(),
            t,
            parameters,
            state,
            self.expected_input_len,
            out,
        )
        .map_err(|(index, message)| IvpBackendError::AtomEvaluationFailure {
            stage: "residual".to_string(),
            index,
            message,
        })
    }

    fn evaluate(&self, t: f64, y: &DVector<f64>) -> Result<DVector<f64>, IvpBackendError> {
        let mut values = vec![0.0; self.len()];
        self.evaluate_into(t, y, &mut values)?;
        Ok(DVector::from_vec(values))
    }
}

/// Shared immutable Atom preparation used by the native residual and
/// Jacobian compilers. The numerical callbacks own only their compiled
/// evaluators; this object prevents the two cold paths from converting the
/// same equations independently.
struct PreparedNativeAtomSystem {
    atoms: Arc<[Atom]>,
    context: PreparedVariableContext,
    function_map: FunctionMap,
}

/// AOT preparation settings for dense IVP problems.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SymbolicIvpAotOptions {
    /// Residual chunking used by the generated IVP residual plan.
    pub residual_strategy: ResidualChunkingStrategy,
    /// Row chunking used by the generated dense IVP Jacobian plan.
    pub jacobian_strategy: DenseJacobianChunkingStrategy,
}

impl Default for SymbolicIvpAotOptions {
    fn default() -> Self {
        Self {
            residual_strategy: ResidualChunkingStrategy::Whole,
            jacobian_strategy: DenseJacobianChunkingStrategy::Whole,
        }
    }
}

/// Shared symbolic setup for IVP consumers.
#[derive(Debug, Clone, Default)]
pub struct SymbolicIvpProblemOptions {
    /// Optional symbolic parameter names used during evaluation but not during
    /// differentiation.
    pub equation_parameters: Option<Vec<String>>,
    /// Optional initial parameter values. When present, the values are stored
    /// behind a shared handle so solvers may update them without recompiling
    /// symbolic closures.
    pub equation_parameter_values: Option<DVector<f64>>,
    /// Backend preference for preparation and future codegen.
    pub backend_policy: IvpBackendSelectionPolicy,
    /// Dense AOT preparation settings.
    pub aot_options: SymbolicIvpAotOptions,
    /// Symbolic Jacobian assembly backend.
    pub symbolic_assembly_backend: IvpSymbolicAssemblyBackend,
    /// Runtime policy for independent residual/Jacobian Lambdify entries.
    /// Sequential is the compatibility default; Parallel and Auto are opt-in.
    pub lambdify_execution_policy: IvpLambdifyExecutionPolicy,
    /// Optional row-major Jacobian supplied by the caller.
    ///
    /// When present, preparation must not differentiate the residual graph.
    /// The payload is also carried into the AOT manifest, so an explicit
    /// Jacobian cannot accidentally reuse an artifact generated from a
    /// residual-derived Jacobian.
    pub explicit_jacobian: Option<Vec<Vec<Expr>>>,
    /// Optional preparation/callback telemetry. Disabled by default.
    pub telemetry: IvpTelemetry,
}

impl SymbolicIvpProblemOptions {
    /// Creates one default IVP symbolic setup that stays on lambdify.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declares symbolic parameter names used by `f(t, y, p)`.
    pub fn with_equation_parameters(mut self, parameters: Vec<String>) -> Self {
        self.equation_parameters = Some(parameters);
        self
    }

    /// Installs initial numeric values for symbolic parameters.
    pub fn with_equation_parameter_values(mut self, values: DVector<f64>) -> Self {
        self.equation_parameter_values = Some(values);
        self
    }

    /// Requests AOT-ready preparation while preserving lambdify execution.
    pub fn with_prefer_aot_then_lambdify(mut self) -> Self {
        self.backend_policy = IvpBackendSelectionPolicy::PreferAotThenLambdify;
        self
    }

    /// Overrides dense AOT plan chunking.
    pub fn with_aot_options(mut self, options: SymbolicIvpAotOptions) -> Self {
        self.aot_options = options;
        self
    }

    /// Selects the symbolic Jacobian assembly backend.
    pub fn with_symbolic_assembly_backend(mut self, backend: IvpSymbolicAssemblyBackend) -> Self {
        self.symbolic_assembly_backend = backend;
        self
    }

    /// Selects sequential, forced-parallel, or conservative automatic callback
    /// execution without changing symbolic preparation or solver semantics.
    pub fn with_lambdify_execution_policy(mut self, policy: IvpLambdifyExecutionPolicy) -> Self {
        self.lambdify_execution_policy = policy;
        self
    }

    /// Supplies a validated row-major symbolic Jacobian for all supported
    /// symbolic assembly frontends.
    pub fn with_explicit_jacobian(mut self, jacobian: Vec<Vec<Expr>>) -> Self {
        self.explicit_jacobian = Some(jacobian);
        self
    }

    /// Enables typed preparation and callback telemetry for this problem.
    pub fn with_telemetry(mut self, telemetry: IvpTelemetry) -> Self {
        self.telemetry = telemetry;
        self
    }
}

/// Prepared dense IVP AOT bridge built from symbolic equations.
#[derive(Debug, Clone)]
pub struct PreparedSymbolicIvpAotProblem<'a> {
    equations: &'a [Expr],
    symbolic_jacobian: &'a [Vec<Expr>],
    time_arg: &'a str,
    variable_refs: Vec<&'a str>,
    parameter_refs: Option<Vec<&'a str>>,
    flattened_input_names: Vec<&'a str>,
    residual_fn_name: String,
    jacobian_fn_name: String,
    residual_strategy: ResidualChunkingStrategy,
    jacobian_strategy: DenseJacobianChunkingStrategy,
}

/// Prepared residual-only IVP AOT bridge built from symbolic equations.
#[derive(Debug, Clone)]
pub struct PreparedSymbolicIvpResidualAotProblem<'a> {
    equations: &'a [Expr],
    time_arg: &'a str,
    variable_refs: Vec<&'a str>,
    parameter_refs: Option<Vec<&'a str>>,
    flattened_input_names: Vec<&'a str>,
    residual_fn_name: String,
    residual_strategy: ResidualChunkingStrategy,
}

impl<'a> PreparedSymbolicIvpResidualAotProblem<'a> {
    /// Returns the residual-only runtime plan in flattened IVP order:
    /// time, params..., variables...
    pub fn residual_runtime_plan(
        &self,
    ) -> crate::symbolic::codegen::codegen_runtime_api::ResidualRuntimePlan<'_> {
        IvpResidualTask {
            fn_name: self.residual_fn_name.as_str(),
            time_arg: self.time_arg,
            residuals: self.equations,
            variables: &self.variable_refs,
            params: self.parameter_refs.as_deref(),
        }
        .runtime_plan(self.residual_strategy)
    }

    /// Returns the manifest-derived stable problem key.
    pub fn manifest(&self) -> PreparedProblemManifest {
        PreparedProblemManifest::residual_only(
            BackendKind::Aot,
            MatrixBackend::ValuesOnly,
            &self.residual_runtime_plan(),
        )
    }

    /// Returns the manifest-derived stable problem key.
    pub fn problem_key(&self) -> String {
        self.manifest().problem_key()
    }

    /// Returns the flattened input names in IVP order:
    /// time, params..., variables...
    pub fn flattened_input_names(&self) -> &[&'a str] {
        &self.flattened_input_names
    }
}

impl<'a> PreparedSymbolicIvpAotProblem<'a> {
    fn residual_runtime_plan(
        &self,
    ) -> crate::symbolic::codegen::codegen_runtime_api::ResidualRuntimePlan<'_> {
        IvpResidualTask {
            fn_name: self.residual_fn_name.as_str(),
            time_arg: self.time_arg,
            residuals: self.equations,
            variables: &self.variable_refs,
            params: self.parameter_refs.as_deref(),
        }
        .runtime_plan(self.residual_strategy)
    }

    fn jacobian_runtime_plan(
        &self,
    ) -> crate::symbolic::codegen::codegen_runtime_api::DenseJacobianRuntimePlan<'_> {
        IvpJacobianTask {
            fn_name: self.jacobian_fn_name.as_str(),
            time_arg: self.time_arg,
            jacobian: self.symbolic_jacobian,
            variables: &self.variable_refs,
            params: self.parameter_refs.as_deref(),
        }
        .runtime_plan(self.jacobian_strategy)
    }

    /// Returns the generic prepared dense problem used by the shared AOT lifecycle.
    pub fn as_prepared_problem(&self) -> PreparedDenseProblem<'_> {
        PreparedDenseProblem::new(
            BackendKind::Aot,
            MatrixBackend::Dense,
            self.residual_runtime_plan(),
            self.jacobian_runtime_plan(),
        )
    }

    /// Returns the manifest-derived stable problem key.
    pub fn manifest(&self) -> PreparedProblemManifest {
        PreparedProblemManifest::from(&self.as_prepared_problem())
    }

    /// Returns the manifest-derived stable problem key.
    pub fn problem_key(&self) -> String {
        self.manifest().problem_key()
    }

    /// Returns the flattened input names in IVP order:
    /// time, params..., variables...
    pub fn flattened_input_names(&self) -> &[&'a str] {
        &self.flattened_input_names
    }
}

/// Prepared symbolic IVP backend shared by ODE solvers.
pub struct PreparedSymbolicIvpProblem {
    /// Solver-facing callable residual evaluator.
    pub residual: Box<IvpResidualEval>,
    /// Solver-facing callable dense Jacobian evaluator.
    pub jacobian: Box<IvpDenseJacobianEval>,
    try_residual: Arc<IvpTryResidualEval>,
    try_jacobian: Arc<IvpTryDenseJacobianEval>,
    native_residual: Option<Arc<PreparedNativeAtomResidual>>,
    native_jacobian: Option<Arc<PreparedNativeAtomJacobian>>,
    linked_residual: Option<Arc<PreparedLinkedResidual>>,
    _linked_dense: Option<Arc<PreparedLinkedDense>>,
    /// Packed Atom payload retained for the native AOT handoff. It is built
    /// once in the AtomView preparation branch and never reconstructed from
    /// the public Expr compatibility fields.
    native_atoms: Option<Arc<[Atom]>>,
    /// Symbolic equations used to prepare residuals.
    pub equations: Vec<Expr>,
    /// Symbolic dense Jacobian used by both lambdify and AOT preparation.
    pub symbolic_jacobian: Vec<Vec<Expr>>,
    /// Explicit Jacobian source, when the caller supplied one. This remains
    /// separate from the derived/native representation so AtomView AOT can
    /// lower it directly without converting a derived Jacobian back to Expr.
    pub(crate) explicit_jacobian: Option<Vec<Vec<Expr>>>,
    /// Independent IVP argument name, typically `t`.
    pub time_arg: String,
    /// Differentiable state variables.
    pub variables: Vec<String>,
    /// Optional symbolic parameter names.
    pub equation_parameters: Option<Vec<String>>,
    /// Shared parameter storage used by params-aware closures.
    parameter_values_handle: Option<SharedIvpParameterValues>,
    /// Selected callable backend kind.
    pub backend_kind: IvpBackendKind,
    /// Shared opt-in preparation/callback telemetry stream.
    pub telemetry: IvpTelemetry,
    execution_policy: IvpLambdifyExecutionPolicy,
}

/// Prepared symbolic IVP residual without compiling any Jacobian callback.
///
/// This is useful for solver paths that provide a native sparse/banded
/// Jacobian evaluator separately and should not pay for a dense Jacobian
/// closure during setup.
pub struct PreparedSymbolicIvpResidualProblem {
    pub residual: Box<IvpResidualEval>,
    try_residual: Arc<IvpTryResidualEval>,
    native_residual: Option<Arc<PreparedNativeAtomResidual>>,
    linked_residual: Option<Arc<PreparedLinkedResidual>>,
    /// Packed Atom payload reused by the native AOT handoff.
    native_atoms: Option<Arc<[Atom]>>,
    /// Symbolic equations used to prepare residuals.
    pub equations: Vec<Expr>,
    /// Explicit Jacobian retained for the later AtomView AOT handoff.
    pub(crate) explicit_jacobian: Option<Vec<Vec<Expr>>>,
    /// Independent IVP argument name, typically `t`.
    pub time_arg: String,
    /// Differentiable state variables.
    pub variables: Vec<String>,
    /// Optional symbolic parameter names.
    pub equation_parameters: Option<Vec<String>>,
    parameter_values_handle: Option<SharedIvpParameterValues>,
    /// Selected callable backend kind.
    pub backend_kind: IvpBackendKind,
    /// Shared opt-in preparation/callback telemetry stream.
    pub telemetry: IvpTelemetry,
    execution_policy: IvpLambdifyExecutionPolicy,
}

/// Immutable owner for a linked AOT residual callback.
///
/// The generated callback is `Send + Sync`; reusable argument and output
/// storage belongs to the caller instead of this shared owner. This keeps
/// parallel solver calls free of a mutex around scratch buffers.
struct PreparedLinkedResidual {
    linked: LinkedResidualAotBackend,
    parameter_values_handle: Option<SharedIvpParameterValues>,
    telemetry: IvpTelemetry,
    expected_state_len: usize,
    execution_policy: IvpLambdifyExecutionPolicy,
}

/// Owns one linked dense AOT callback and attributes its warm work to the
/// same typed telemetry stream as native and Lambdify evaluators.
struct PreparedLinkedDense {
    linked: LinkedDenseAotBackend,
    parameter_values_handle: Option<SharedIvpParameterValues>,
    telemetry: IvpTelemetry,
    expected_state_len: usize,
    execution_policy: IvpLambdifyExecutionPolicy,
}

impl PreparedLinkedDense {
    fn new(
        linked: LinkedDenseAotBackend,
        parameter_values_handle: Option<SharedIvpParameterValues>,
        telemetry: IvpTelemetry,
        expected_state_len: usize,
        execution_policy: IvpLambdifyExecutionPolicy,
    ) -> Self {
        Self {
            linked,
            parameter_values_handle,
            telemetry,
            expected_state_len,
            execution_policy,
        }
    }

    fn evaluate_with_output<T>(
        &self,
        t: f64,
        y: &DVector<f64>,
        args: &mut Vec<f64>,
        assemble_output: impl FnOnce(&[f64], usize, usize) -> T,
    ) -> Result<T, IvpBackendError> {
        let callback_scope = self
            .telemetry
            .scoped_warm_stage(IvpWarmStage::JacobianCallback);
        if y.len() != self.expected_state_len {
            self.telemetry.record_error();
            drop(callback_scope);
            return Err(IvpBackendError::InvalidStateShape {
                expected: self.expected_state_len,
                actual: y.len(),
            });
        }
        let (rows, cols) = self.linked.shape;

        let binding_scope = self
            .telemetry
            .scoped_warm_stage(IvpWarmStage::ArgumentBinding);
        let copy_scope = self
            .telemetry
            .scoped_warm_stage(IvpWarmStage::AotArgumentCopy);
        if let Err(error) =
            build_linked_args_into(t, y, self.parameter_values_handle.as_ref(), args)
        {
            self.telemetry.record_error();
            drop(copy_scope);
            drop(binding_scope);
            drop(callback_scope);
            return Err(error);
        }
        self.telemetry
            .record_copy_bytes(args.len() * std::mem::size_of::<f64>());
        drop(copy_scope);
        drop(binding_scope);

        let worker_scope = self
            .telemetry
            .scoped_warm_stage(IvpWarmStage::AotWorkerExecution);
        let mut values = vec![0.0; rows * cols];
        self.telemetry
            .record_allocation(values.len() * std::mem::size_of::<f64>());
        self.linked
            .try_jacobian_eval_with_policy(
                args.as_slice(),
                values.as_mut_slice(),
                self.execution_policy,
                &self.telemetry,
            )
            .map_err(|error| IvpBackendError::GeneratedBackendFailure {
                message: error.to_string(),
            })?;
        drop(worker_scope);

        let output_scope = self
            .telemetry
            .scoped_warm_stage(IvpWarmStage::AotOutputWrite);
        let output = assemble_output(values.as_slice(), rows, cols);
        self.telemetry
            .record_copy_bytes(values.len() * std::mem::size_of::<f64>());
        drop(output_scope);
        self.telemetry.record_jacobian_evaluation_count();
        drop(callback_scope);
        Ok(output)
    }

    fn evaluate_owned(
        &self,
        t: f64,
        y: &DVector<f64>,
        args: &mut Vec<f64>,
    ) -> Result<DMatrix<f64>, IvpBackendError> {
        self.evaluate_with_output(t, y, args, |values, rows, cols| {
            self.telemetry
                .record_allocation(values.len() * std::mem::size_of::<f64>());
            DMatrix::from_row_slice(rows, cols, values)
        })
    }

    /// Evaluates directly into the caller-owned row-major ABI buffer.
    ///
    /// Generated dense Jacobian callbacks already write row-major values. This
    /// boundary deliberately does not materialize a temporary `Vec` or a
    /// `DMatrix`; callers that can consume the generated layout can therefore
    /// reuse both the argument and output buffers across evaluations.
    fn evaluate_row_major_into(
        &self,
        t: f64,
        y: &DVector<f64>,
        out: &mut [f64],
        args: &mut Vec<f64>,
    ) -> Result<(), IvpBackendError> {
        let callback_scope = self
            .telemetry
            .scoped_warm_stage(IvpWarmStage::JacobianCallback);
        if y.len() != self.expected_state_len {
            self.telemetry.record_error();
            drop(callback_scope);
            return Err(IvpBackendError::InvalidStateShape {
                expected: self.expected_state_len,
                actual: y.len(),
            });
        }
        let (rows, cols) = self.linked.shape;
        let expected = rows.saturating_mul(cols);
        if out.len() != expected {
            self.telemetry.record_error();
            drop(callback_scope);
            return Err(IvpBackendError::InvalidOutputShape {
                stage: "jacobian-row-major".to_string(),
                expected,
                actual: out.len(),
            });
        }

        let binding_scope = self
            .telemetry
            .scoped_warm_stage(IvpWarmStage::ArgumentBinding);
        let copy_scope = self
            .telemetry
            .scoped_warm_stage(IvpWarmStage::AotArgumentCopy);
        if let Err(error) =
            build_linked_args_into(t, y, self.parameter_values_handle.as_ref(), args)
        {
            self.telemetry.record_error();
            drop(copy_scope);
            drop(binding_scope);
            drop(callback_scope);
            return Err(error);
        }
        self.telemetry
            .record_copy_bytes(args.len() * std::mem::size_of::<f64>());
        drop(copy_scope);
        drop(binding_scope);

        let worker_scope = self
            .telemetry
            .scoped_warm_stage(IvpWarmStage::AotWorkerExecution);
        self.linked
            .try_jacobian_eval_with_policy(
                args.as_slice(),
                out,
                self.execution_policy,
                &self.telemetry,
            )
            .map_err(|error| IvpBackendError::GeneratedBackendFailure {
                message: error.to_string(),
            })?;
        drop(worker_scope);
        self.telemetry.record_jacobian_evaluation_count();
        drop(callback_scope);
        Ok(())
    }

    /// Evaluates a linked dense Jacobian from borrowed solver slices.
    ///
    /// This is the zero-temporary boundary used by Radau AOT.  Unlike the
    /// historical DVector-facing method it does not create a state vector or
    /// consult a shared parameter lock on every callback.
    fn evaluate_row_major_parts_into(
        &self,
        t: f64,
        parameters: &[f64],
        state: &[f64],
        out: &mut [f64],
        args: &mut Vec<f64>,
    ) -> Result<(), IvpBackendError> {
        if state.len() != self.expected_state_len {
            return Err(IvpBackendError::InvalidStateShape {
                expected: self.expected_state_len,
                actual: state.len(),
            });
        }
        let (rows, cols) = self.linked.shape;
        let expected = rows.saturating_mul(cols);
        if out.len() != expected {
            return Err(IvpBackendError::InvalidOutputShape {
                stage: "jacobian-row-major".to_string(),
                expected,
                actual: out.len(),
            });
        }
        build_linked_args_from_parts(t, parameters, state, args);
        self.linked
            .try_jacobian_eval_with_policy(
                args.as_slice(),
                out,
                self.execution_policy,
                &self.telemetry,
            )
            .map_err(|error| IvpBackendError::GeneratedBackendFailure {
                message: error.to_string(),
            })
    }
}

impl PreparedLinkedResidual {
    fn new(
        linked: LinkedResidualAotBackend,
        parameter_values_handle: Option<SharedIvpParameterValues>,
        telemetry: IvpTelemetry,
        expected_state_len: usize,
        execution_policy: IvpLambdifyExecutionPolicy,
    ) -> Self {
        Self {
            linked,
            parameter_values_handle,
            telemetry,
            expected_state_len,
            execution_policy,
        }
    }

    fn evaluate_into(
        &self,
        t: f64,
        y: &DVector<f64>,
        out: &mut [f64],
        args: &mut Vec<f64>,
    ) -> Result<(), IvpBackendError> {
        let callback_scope = self
            .telemetry
            .scoped_warm_stage(IvpWarmStage::ResidualCallback);
        if y.len() != self.expected_state_len {
            self.telemetry.record_error();
            drop(callback_scope);
            return Err(IvpBackendError::InvalidStateShape {
                expected: self.expected_state_len,
                actual: y.len(),
            });
        }
        if out.len() != self.linked.residual_len {
            self.telemetry.record_error();
            drop(callback_scope);
            return Err(IvpBackendError::InvalidOutputShape {
                stage: "residual".to_string(),
                expected: self.linked.residual_len,
                actual: out.len(),
            });
        }

        let binding_scope = self
            .telemetry
            .scoped_warm_stage(IvpWarmStage::ArgumentBinding);
        let copy_scope = self
            .telemetry
            .scoped_warm_stage(IvpWarmStage::AotArgumentCopy);
        if let Err(error) =
            build_linked_args_into(t, y, self.parameter_values_handle.as_ref(), args)
        {
            self.telemetry.record_error();
            drop(copy_scope);
            drop(binding_scope);
            return Err(error);
        }
        self.telemetry
            .record_copy_bytes(args.len() * std::mem::size_of::<f64>());
        drop(copy_scope);
        drop(binding_scope);

        let worker_scope = self
            .telemetry
            .scoped_warm_stage(IvpWarmStage::AotWorkerExecution);
        let evaluation_scope = self
            .telemetry
            .scoped_warm_stage(IvpWarmStage::ResidualEvaluation);
        if let Err(error) = self
            .linked
            .try_residual_eval_with_policy(
                args.as_slice(),
                out,
                self.execution_policy,
                &self.telemetry,
            )
            .map_err(|error| IvpBackendError::GeneratedBackendFailure {
                message: error.to_string(),
            })
        {
            self.telemetry.record_error();
            drop(worker_scope);
            return Err(error);
        }
        drop(evaluation_scope);
        drop(worker_scope);

        let output_scope = self
            .telemetry
            .scoped_warm_stage(IvpWarmStage::ResidualOutputAssembly);
        let aot_output_scope = self
            .telemetry
            .scoped_warm_stage(IvpWarmStage::AotOutputWrite);
        self.telemetry.record_residual_evaluation_count();
        drop(aot_output_scope);
        drop(output_scope);
        drop(callback_scope);
        Ok(())
    }

    /// Evaluates a linked residual from borrowed solver slices without
    /// materializing a temporary state vector or parameter binding.
    fn evaluate_parts_into(
        &self,
        t: f64,
        parameters: &[f64],
        state: &[f64],
        out: &mut [f64],
        args: &mut Vec<f64>,
    ) -> Result<(), IvpBackendError> {
        if state.len() != self.expected_state_len {
            return Err(IvpBackendError::InvalidStateShape {
                expected: self.expected_state_len,
                actual: state.len(),
            });
        }
        if out.len() != self.linked.residual_len {
            return Err(IvpBackendError::InvalidOutputShape {
                stage: "residual".to_string(),
                expected: self.linked.residual_len,
                actual: out.len(),
            });
        }
        build_linked_args_from_parts(t, parameters, state, args);
        self.linked
            .try_residual_eval_with_policy(
                args.as_slice(),
                out,
                self.execution_policy,
                &self.telemetry,
            )
            .map_err(|error| IvpBackendError::GeneratedBackendFailure {
                message: error.to_string(),
            })
    }
}

impl PreparedSymbolicIvpResidualProblem {
    /// Evaluates the prepared residual through the typed fallible boundary.
    pub fn try_evaluate_residual(
        &self,
        t: f64,
        y: &DVector<f64>,
    ) -> Result<DVector<f64>, IvpBackendError> {
        (self.try_residual)(t, y)
    }

    /// Evaluates the prepared residual into caller-owned storage.
    ///
    /// AtomView uses its immutable native evaluator list directly. Legacy and
    /// compatibility routes retain the owned-result fallback until their
    /// evaluator APIs expose equivalent output-buffer support.
    pub fn try_evaluate_residual_into(
        &self,
        t: f64,
        y: &DVector<f64>,
        out: &mut DVector<f64>,
    ) -> Result<(), IvpBackendError> {
        let mut args = Vec::new();
        self.try_evaluate_residual_into_with_workspace(t, y, out, &mut args)
    }

    /// Evaluates the residual into caller-owned output and argument storage.
    ///
    /// The linked AOT path uses the supplied `args` buffer for the flattened
    /// ABI input, so repeated calls do not allocate an argument vector or an
    /// output vector. Native AtomView already evaluates directly into `out`;
    /// the buffer is intentionally left untouched on that route.
    pub fn try_evaluate_residual_into_with_workspace(
        &self,
        t: f64,
        y: &DVector<f64>,
        out: &mut DVector<f64>,
        args: &mut Vec<f64>,
    ) -> Result<(), IvpBackendError> {
        if let Some(linked) = &self.linked_residual {
            return linked.evaluate_into(t, y, out.as_mut_slice(), args);
        }
        if let Some(native) = &self.native_residual {
            return native.evaluate_into(t, y, out.as_mut_slice());
        }
        let values = self.try_evaluate_residual(t, y)?;
        if out.len() != values.len() {
            return Err(IvpBackendError::InvalidOutputShape {
                stage: "residual".to_string(),
                expected: values.len(),
                actual: out.len(),
            });
        }
        out.copy_from(&values);
        Ok(())
    }

    pub fn parameter_values_handle(&self) -> Option<SharedIvpParameterValues> {
        self.parameter_values_handle.clone()
    }

    /// Updates residual-only parameter values without rebuilding its callback.
    pub fn set_parameter_values(&self, values: DVector<f64>) -> Result<(), IvpBackendError> {
        match (&self.equation_parameters, &self.parameter_values_handle) {
            (Some(parameters), Some(handle)) => {
                if parameters.len() != values.len() {
                    return Err(IvpBackendError::ParameterCountMismatch {
                        expected: parameters.len(),
                        actual: values.len(),
                    });
                }
                let mut slot = handle
                    .write()
                    .map_err(|_| IvpBackendError::ParameterStatePoisoned)?;
                *slot = values;
                self.telemetry.record_parameter_bind();
                Ok(())
            }
            (Some(parameters), None) => Err(IvpBackendError::MissingParameterValues {
                expected: parameters.len(),
            }),
            (None, _) => {
                if values.is_empty() {
                    Ok(())
                } else {
                    Err(IvpBackendError::ParameterCountMismatch {
                        expected: 0,
                        actual: values.len(),
                    })
                }
            }
        }
    }

    pub(crate) fn native_atoms(&self) -> Option<&[Atom]> {
        self.native_atoms.as_deref()
    }

    /// Builds a residual-only IVP prepared AOT bridge from the already
    /// prepared symbolic residual problem.
    pub fn prepare_residual_aot_problem(
        &self,
        options: SymbolicIvpAotOptions,
    ) -> PreparedSymbolicIvpResidualAotProblem<'_> {
        let variable_refs = self
            .variables
            .iter()
            .map(|value| value.as_str())
            .collect::<Vec<_>>();
        let parameter_refs = self.equation_parameters.as_ref().map(|parameters| {
            parameters
                .iter()
                .map(|value| value.as_str())
                .collect::<Vec<_>>()
        });

        let mut flattened_input_names = Vec::with_capacity(
            1 + variable_refs.len() + parameter_refs.as_ref().map_or(0, |params| params.len()),
        );
        flattened_input_names.push(self.time_arg.as_str());
        if let Some(params) = parameter_refs.as_ref() {
            flattened_input_names.extend(params.iter().copied());
        }
        flattened_input_names.extend(variable_refs.iter().copied());

        PreparedSymbolicIvpResidualAotProblem {
            equations: &self.equations,
            time_arg: self.time_arg.as_str(),
            variable_refs,
            parameter_refs,
            flattened_input_names,
            residual_fn_name: "generated_ivp_residual_eval".to_string(),
            residual_strategy: options.residual_strategy,
        }
    }

    /// Rebinds this residual-only problem to one already linked AOT residual
    /// backend.
    pub fn into_linked_residual_backend(self, linked: LinkedResidualAotBackend) -> Self {
        let residual_len = linked.residual_len;
        let parameter_values_handle = self.parameter_values_handle.clone();
        let telemetry = self.telemetry.clone();
        telemetry.set_execution(IvpTelemetryExecution::Aot);
        let execution_policy = self.execution_policy;
        let linked_runtime = Arc::new(PreparedLinkedResidual::new(
            linked.clone(),
            parameter_values_handle.clone(),
            telemetry.clone(),
            self.variables.len(),
            execution_policy,
        ));
        let linked_runtime_for_closure = linked_runtime.clone();
        let try_residual: Arc<IvpTryResidualEval> = Arc::new(move |t, y| {
            let mut out = vec![0.0; residual_len];
            let mut args = Vec::new();
            linked_runtime_for_closure.evaluate_into(t, y, out.as_mut_slice(), &mut args)?;
            Ok(DVector::from_vec(out))
        });
        let residual =
            compatibility_residual(try_residual.clone(), telemetry.clone(), residual_len);

        Self {
            residual,
            try_residual,
            native_residual: None,
            linked_residual: Some(linked_runtime),
            native_atoms: None,
            equations: self.equations,
            explicit_jacobian: self.explicit_jacobian,
            time_arg: self.time_arg,
            variables: self.variables,
            equation_parameters: self.equation_parameters,
            parameter_values_handle: self.parameter_values_handle,
            backend_kind: IvpBackendKind::Aot,
            telemetry,
            execution_policy,
        }
    }
}

impl PreparedSymbolicIvpProblem {
    pub(crate) fn native_atoms(&self) -> Option<&[Atom]> {
        self.native_atoms.as_deref()
    }

    /// Returns whether the prepared problem can write generated dense AOT
    /// Jacobian values into a caller-owned row-major buffer.
    pub(crate) fn supports_jacobian_row_major_workspace(&self) -> bool {
        self._linked_dense.is_some()
    }

    /// Number of entries in the prepared native Atom Jacobian, when this
    /// problem uses the AtomView assembly route.
    pub(crate) fn native_jacobian_entry_count(&self) -> Option<usize> {
        self.native_jacobian
            .as_ref()
            .map(|jacobian| jacobian.entries.len())
    }

    /// Return the immutable native Jacobian entry pattern for structured
    /// solver consumers. This is preparation metadata; callbacks should use
    /// the borrowed values method below rather than rebuild the pattern.
    pub(crate) fn native_jacobian_pattern(&self) -> Option<Vec<(usize, usize)>> {
        self.native_jacobian.as_ref().map(|jacobian| {
            jacobian
                .entries
                .iter()
                .map(|entry| (entry.row, entry.col))
                .collect()
        })
    }

    /// Evaluate the native Atom residual from borrowed IVP slices.
    ///
    /// This is shared by solver adapters so they do not need to construct a
    /// temporary nalgebra vector merely to call the prepared evaluator.
    pub(crate) fn try_evaluate_native_residual_parts(
        &self,
        time: f64,
        parameters: &[f64],
        state: &[f64],
        output: &mut [f64],
    ) -> Result<(), IvpBackendError> {
        let Some(native) = &self.native_residual else {
            return Err(IvpBackendError::GeneratedBackendFailure {
                message: "prepared problem has no native AtomView residual".to_string(),
            });
        };
        native.evaluate_values_into_parts(time, parameters, state, output)
    }

    /// Evaluate the native Atom Jacobian directly into a row-major caller
    /// buffer. The sparse evaluator values are reused through `values` and
    /// never materialized as an Expr or DMatrix.
    pub(crate) fn try_evaluate_native_jacobian_parts(
        &self,
        time: f64,
        parameters: &[f64],
        state: &[f64],
        output: &mut [f64],
        values: &mut Vec<f64>,
    ) -> Result<(), IvpBackendError> {
        let Some(native) = &self.native_jacobian else {
            return Err(IvpBackendError::GeneratedBackendFailure {
                message: "prepared problem has no native AtomView Jacobian".to_string(),
            });
        };
        let rows = self.equations.len();
        let cols = self.variables.len();
        let expected = rows.saturating_mul(cols);
        if output.len() != expected {
            return Err(IvpBackendError::InvalidOutputShape {
                stage: "native Jacobian".to_string(),
                expected,
                actual: output.len(),
            });
        }
        self.try_evaluate_native_jacobian_values_parts(time, parameters, state, values)?;
        output.fill(0.0);
        for (entry, value) in native.entries.iter().zip(values.iter().copied()) {
            output[entry.row * cols + entry.col] = value;
        }
        Ok(())
    }

    /// Evaluate native Jacobian entries in the immutable prepared pattern
    /// order. Structured consumers can write these values directly into
    /// sparse or banded storage without first materializing a dense matrix.
    pub(crate) fn try_evaluate_native_jacobian_values_parts(
        &self,
        time: f64,
        parameters: &[f64],
        state: &[f64],
        values: &mut Vec<f64>,
    ) -> Result<(), IvpBackendError> {
        let Some(native) = &self.native_jacobian else {
            return Err(IvpBackendError::GeneratedBackendFailure {
                message: "prepared problem has no native AtomView Jacobian".to_string(),
            });
        };
        values.resize(native.entries.len(), 0.0);
        native.evaluate_values_into_parts(
            time,
            parameters,
            state,
            self.execution_policy,
            values.as_mut_slice(),
        )
    }

    /// Evaluate a linked dense AOT residual directly from borrowed slices.
    pub(crate) fn try_evaluate_aot_residual_parts(
        &self,
        time: f64,
        parameters: &[f64],
        state: &[f64],
        output: &mut [f64],
        args: &mut Vec<f64>,
    ) -> Result<(), IvpBackendError> {
        let Some(linked) = &self.linked_residual else {
            return Err(IvpBackendError::GeneratedBackendFailure {
                message: "prepared problem has no linked AOT residual".to_string(),
            });
        };
        linked.evaluate_parts_into(time, parameters, state, output, args)
    }

    /// Evaluate a linked dense AOT residual with an already-sized ABI slice.
    ///
    /// The public compatibility method accepts `Vec` because it can resize a
    /// cold caller workspace. Solver adapters already own a correctly sized
    /// buffer, so this boundary keeps the repeated callback path allocation
    /// free.
    pub(crate) fn try_evaluate_aot_residual_parts_with_args(
        &self,
        args: &[f64],
        output: &mut [f64],
    ) -> Result<(), IvpBackendError> {
        let Some(linked) = &self.linked_residual else {
            return Err(IvpBackendError::GeneratedBackendFailure {
                message: "prepared problem has no linked AOT residual".to_string(),
            });
        };
        linked
            .linked
            .try_residual_eval_with_policy(args, output, self.execution_policy, &self.telemetry)
            .map_err(|error| IvpBackendError::GeneratedBackendFailure {
                message: error.to_string(),
            })
    }

    /// Evaluate a linked dense AOT Jacobian directly into row-major storage.
    pub(crate) fn try_evaluate_aot_jacobian_parts(
        &self,
        time: f64,
        parameters: &[f64],
        state: &[f64],
        output: &mut [f64],
        args: &mut Vec<f64>,
    ) -> Result<(), IvpBackendError> {
        let Some(linked) = &self._linked_dense else {
            return Err(IvpBackendError::GeneratedBackendFailure {
                message: "prepared problem has no linked AOT Jacobian".to_string(),
            });
        };
        linked.evaluate_row_major_parts_into(time, parameters, state, output, args)
    }

    /// Evaluate a linked dense AOT Jacobian using caller-owned ABI storage.
    /// This is the zero-allocation adapter used by the BVP collocation loop.
    pub(crate) fn try_evaluate_aot_jacobian_parts_with_args(
        &self,
        args: &[f64],
        output: &mut [f64],
    ) -> Result<(), IvpBackendError> {
        let Some(linked) = &self._linked_dense else {
            return Err(IvpBackendError::GeneratedBackendFailure {
                message: "prepared problem has no linked AOT Jacobian".to_string(),
            });
        };
        linked
            .linked
            .try_jacobian_eval_with_policy(args, output, self.execution_policy, &self.telemetry)
            .map_err(|error| IvpBackendError::GeneratedBackendFailure {
                message: error.to_string(),
            })
    }

    /// Evaluates the prepared residual through the typed fallible boundary.
    pub fn try_evaluate_residual(
        &self,
        t: f64,
        y: &DVector<f64>,
    ) -> Result<DVector<f64>, IvpBackendError> {
        (self.try_residual)(t, y)
    }

    /// Evaluates the prepared residual into caller-owned storage.
    pub fn try_evaluate_residual_into(
        &self,
        t: f64,
        y: &DVector<f64>,
        out: &mut DVector<f64>,
    ) -> Result<(), IvpBackendError> {
        let mut args = Vec::new();
        self.try_evaluate_residual_into_with_workspace(t, y, out, &mut args)
    }

    /// Evaluates the residual using caller-owned flattened ABI storage.
    ///
    /// This is the allocation-free boundary for repeated linked-AOT calls.
    /// The compatibility method above remains available for callers that do
    /// not maintain a workspace between evaluations.
    pub fn try_evaluate_residual_into_with_workspace(
        &self,
        t: f64,
        y: &DVector<f64>,
        out: &mut DVector<f64>,
        args: &mut Vec<f64>,
    ) -> Result<(), IvpBackendError> {
        if let Some(linked) = &self.linked_residual {
            return linked.evaluate_into(t, y, out.as_mut_slice(), args);
        }
        if let Some(native) = &self.native_residual {
            return native.evaluate_into(t, y, out.as_mut_slice());
        }
        let values = self.try_evaluate_residual(t, y)?;
        if out.len() != values.len() {
            return Err(IvpBackendError::InvalidOutputShape {
                stage: "residual".to_string(),
                expected: values.len(),
                actual: out.len(),
            });
        }
        out.copy_from(&values);
        Ok(())
    }

    /// Evaluates the prepared dense Jacobian through the typed fallible
    /// boundary. The compatibility closure remains available as `jacobian`.
    pub fn try_evaluate_jacobian(
        &self,
        t: f64,
        y: &DVector<f64>,
    ) -> Result<DMatrix<f64>, IvpBackendError> {
        (self.try_jacobian)(t, y)
    }

    /// Evaluates a dense Jacobian into caller-owned row-major storage.
    ///
    /// This is the zero-temporary boundary for linked dense AOT callbacks:
    /// `out` is written directly by the generated evaluator and `args` is
    /// reused for flattened input binding. The compatibility `DMatrix`
    /// callback remains available through [`Self::try_evaluate_jacobian`].
    /// For non-linked backends this method preserves the same output contract
    /// through a compatibility fallback.
    pub fn try_evaluate_jacobian_row_major_into_with_workspace(
        &self,
        t: f64,
        y: &DVector<f64>,
        out: &mut [f64],
        args: &mut Vec<f64>,
    ) -> Result<(), IvpBackendError> {
        // Native AtomView preparation keeps the Jacobian in its native plan
        // and leaves the legacy symbolic matrix empty. Dense IVP shape is
        // defined by the validated equation/state schema instead.
        let rows = self.equations.len();
        let cols = self.variables.len();
        let expected = rows.saturating_mul(cols);
        if out.len() != expected {
            return Err(IvpBackendError::InvalidOutputShape {
                stage: "jacobian-row-major".to_string(),
                expected,
                actual: out.len(),
            });
        }
        if let Some(linked) = &self._linked_dense {
            return linked.evaluate_row_major_into(t, y, out, args);
        }

        let matrix = self.try_evaluate_jacobian(t, y)?;
        for row in 0..rows {
            let start = row * cols;
            for col in 0..cols {
                out[start + col] = matrix[(row, col)];
            }
        }
        Ok(())
    }

    /// Evaluates a dense Jacobian into caller-owned nalgebra storage while
    /// reusing the flattened ABI and row-major value workspaces.
    ///
    /// The generated dense ABI is row-major, whereas `DMatrix` stores each
    /// column contiguously. The conversion therefore writes one destination
    /// column at a time, avoiding the strided destination writes used by the
    /// old row-first bridge and keeping the output-assembly scope visible in
    /// the shared IVP telemetry.
    pub(crate) fn try_evaluate_jacobian_into_dmatrix_with_workspace(
        &self,
        t: f64,
        y: &DVector<f64>,
        out: &mut DMatrix<f64>,
        values: &mut Vec<f64>,
        args: &mut Vec<f64>,
    ) -> Result<(), IvpBackendError> {
        // AtomView native preparation intentionally does not retain the
        // legacy Expr Jacobian matrix. The validated IVP schema is the
        // canonical dense shape for both symbolic frontends.
        let rows = self.equations.len();
        let cols = self.variables.len();
        if out.shape() != (rows, cols) {
            return Err(IvpBackendError::InvalidOutputShape {
                stage: "jacobian-dmatrix".to_string(),
                expected: rows.saturating_mul(cols),
                actual: out.len(),
            });
        }
        values.resize(rows.saturating_mul(cols), 0.0);
        self.try_evaluate_jacobian_row_major_into_with_workspace(
            t,
            y,
            values.as_mut_slice(),
            args,
        )?;

        let output_scope = self
            .telemetry
            .scoped_warm_stage(IvpWarmStage::JacobianOutputAssembly);
        let output = out.as_mut_slice();
        for col in 0..cols {
            let destination = &mut output[col * rows..(col + 1) * rows];
            for row in 0..rows {
                destination[row] = values[row * cols + col];
            }
        }
        self.telemetry
            .record_copy_bytes(values.len() * std::mem::size_of::<f64>());
        drop(output_scope);
        Ok(())
    }

    /// Updates parameter values in-place without recompiling closures.
    pub fn set_parameter_values(&self, values: DVector<f64>) -> Result<(), IvpBackendError> {
        match (&self.equation_parameters, &self.parameter_values_handle) {
            (Some(parameters), Some(handle)) => {
                if parameters.len() != values.len() {
                    return Err(IvpBackendError::ParameterCountMismatch {
                        expected: parameters.len(),
                        actual: values.len(),
                    });
                }
                let mut slot = handle
                    .write()
                    .map_err(|_| IvpBackendError::ParameterStatePoisoned)?;
                *slot = values;
                self.telemetry.record_parameter_bind();
                Ok(())
            }
            (Some(parameters), None) => Err(IvpBackendError::MissingParameterValues {
                expected: parameters.len(),
            }),
            (None, _) => {
                if values.is_empty() {
                    Ok(())
                } else {
                    Err(IvpBackendError::ParameterCountMismatch {
                        expected: 0,
                        actual: values.len(),
                    })
                }
            }
        }
    }

    /// Returns a clone of the shared parameter storage handle when params are enabled.
    pub fn parameter_values_handle(&self) -> Option<SharedIvpParameterValues> {
        self.parameter_values_handle.clone()
    }

    /// Builds the dense IVP prepared AOT bridge from the already prepared symbolic problem.
    pub fn prepare_dense_aot_problem(
        &self,
        options: SymbolicIvpAotOptions,
    ) -> PreparedSymbolicIvpAotProblem<'_> {
        let variable_refs = self
            .variables
            .iter()
            .map(|value| value.as_str())
            .collect::<Vec<_>>();
        let parameter_refs = self.equation_parameters.as_ref().map(|parameters| {
            parameters
                .iter()
                .map(|value| value.as_str())
                .collect::<Vec<_>>()
        });

        let mut flattened_input_names = Vec::with_capacity(
            1 + variable_refs.len() + parameter_refs.as_ref().map_or(0, |params| params.len()),
        );
        flattened_input_names.push(self.time_arg.as_str());
        if let Some(params) = parameter_refs.as_ref() {
            flattened_input_names.extend(params.iter().copied());
        }
        flattened_input_names.extend(variable_refs.iter().copied());

        PreparedSymbolicIvpAotProblem {
            equations: &self.equations,
            symbolic_jacobian: &self.symbolic_jacobian,
            time_arg: self.time_arg.as_str(),
            variable_refs,
            parameter_refs,
            flattened_input_names,
            residual_fn_name: "generated_ivp_residual_eval".to_string(),
            jacobian_fn_name: "generated_ivp_jacobian_eval".to_string(),
            residual_strategy: options.residual_strategy,
            jacobian_strategy: options.jacobian_strategy,
        }
    }

    /// Rebinds this prepared problem to one already linked dense AOT backend.
    ///
    /// The resulting residual/Jacobian callbacks stay solver-facing
    /// `f(t, y) / dfdy(t, y)`, while internally flattening arguments into the
    /// AOT order `t, params..., variables...`.
    pub fn into_linked_dense_backend(self, linked: LinkedDenseAotBackend) -> Self {
        let residual_len = linked.residual_len;
        let (rows, cols) = linked.shape;
        let parameter_values_handle = self.parameter_values_handle.clone();
        let telemetry = self.telemetry.clone();
        telemetry.set_execution(IvpTelemetryExecution::Aot);
        let execution_policy = self.execution_policy;
        let linked_runtime = Arc::new(PreparedLinkedResidual::new(
            LinkedResidualAotBackend::new(
                linked.problem_key.clone(),
                linked.residual_len,
                linked.residual_eval.clone(),
            )
            .with_chunked_evaluators(linked.residual_chunks.clone()),
            parameter_values_handle.clone(),
            telemetry.clone(),
            self.variables.len(),
            execution_policy,
        ));
        let linked_dense_runtime = Arc::new(PreparedLinkedDense::new(
            linked,
            parameter_values_handle.clone(),
            telemetry.clone(),
            self.variables.len(),
            execution_policy,
        ));
        let residual_runtime_for_closure = linked_runtime.clone();
        let try_residual: Arc<IvpTryResidualEval> = Arc::new(move |t, y| {
            let mut out = vec![0.0; residual_len];
            let mut args = Vec::new();
            residual_runtime_for_closure.evaluate_into(t, y, out.as_mut_slice(), &mut args)?;
            Ok(DVector::from_vec(out))
        });
        let dense_runtime_for_closure = linked_dense_runtime.clone();
        let try_jacobian: Arc<IvpTryDenseJacobianEval> = Arc::new(move |t, y| {
            let mut args = Vec::new();
            dense_runtime_for_closure.evaluate_owned(t, y, &mut args)
        });
        let residual =
            compatibility_residual(try_residual.clone(), telemetry.clone(), residual_len);
        let jacobian = compatibility_jacobian(try_jacobian.clone(), telemetry.clone(), rows, cols);

        Self {
            residual,
            jacobian,
            try_residual,
            try_jacobian,
            native_residual: None,
            native_jacobian: None,
            linked_residual: Some(linked_runtime),
            _linked_dense: Some(linked_dense_runtime),
            native_atoms: None,
            equations: self.equations,
            symbolic_jacobian: self.symbolic_jacobian,
            explicit_jacobian: self.explicit_jacobian,
            time_arg: self.time_arg,
            variables: self.variables,
            equation_parameters: self.equation_parameters,
            parameter_values_handle,
            backend_kind: IvpBackendKind::Aot,
            telemetry,
            execution_policy,
        }
    }
}

fn prepare_parameter_values_handle(
    equation_parameters: Option<&[String]>,
    equation_parameter_values: Option<DVector<f64>>,
) -> Result<Option<SharedIvpParameterValues>, IvpBackendError> {
    match (equation_parameters, equation_parameter_values) {
        (Some(parameters), Some(values)) => {
            if parameters.len() != values.len() {
                return Err(IvpBackendError::ParameterCountMismatch {
                    expected: parameters.len(),
                    actual: values.len(),
                });
            }
            Ok(Some(Arc::new(RwLock::new(values))))
        }
        (Some(parameters), None) => Err(IvpBackendError::MissingParameterValues {
            expected: parameters.len(),
        }),
        (None, Some(values)) if !values.is_empty() => {
            Err(IvpBackendError::ParameterCountMismatch {
                expected: 0,
                actual: values.len(),
            })
        }
        _ => Ok(None),
    }
}

pub(crate) fn build_symbolic_jacobian(
    equations: &[Expr],
    variables: &[String],
    backend: IvpSymbolicAssemblyBackend,
    telemetry: &IvpTelemetry,
) -> Vec<Vec<Expr>> {
    match backend {
        IvpSymbolicAssemblyBackend::ExprLegacy => {
            let differentiation_started =
                telemetry.start_cold_stage(IvpColdStage::SymbolicDifferentiation);
            let differentiated: Vec<Vec<Expr>> = equations
                .iter()
                .map(|expr| {
                    variables
                        .iter()
                        .map(|variable| expr.diff(variable))
                        .collect::<Vec<_>>()
                })
                .collect();
            telemetry.record_cold_stage(
                IvpColdStage::SymbolicDifferentiation,
                differentiation_started,
            );
            let simplification_started = telemetry.start_cold_stage(IvpColdStage::Simplification);
            let result = differentiated
                .into_iter()
                .map(|row| row.into_iter().map(|expr| expr.simplify()).collect())
                .collect();
            telemetry.record_cold_stage(IvpColdStage::Simplification, simplification_started);
            result
        }
        IvpSymbolicAssemblyBackend::AtomView | IvpSymbolicAssemblyBackend::AtomViewExprCompat => {
            let rows = equations.len();
            let cols = variables.len();
            let variables_for_all_discrete = vec![variables.to_vec(); rows];
            let conversion_started = telemetry.start_cold_stage(IvpColdStage::ExprToAtom);
            let prepared = PreparedSparseAtomSystem::from_exprs(
                equations,
                variables,
                &variables_for_all_discrete,
            );
            telemetry.record_cold_stage(IvpColdStage::ExprToAtom, conversion_started);
            let sparse_started = telemetry.start_cold_stage(IvpColdStage::SparsePattern);
            let sparse_entries =
                prepared.calc_sparse_jacobian_with_bandwidth_and_telemetry(None, telemetry);
            telemetry.record_cold_stage(IvpColdStage::SparsePattern, sparse_started);

            let conversion_started = telemetry.start_cold_stage(IvpColdStage::AtomToExpr);
            let converted = sparse_entries
                .into_iter()
                .map(|entry| {
                    telemetry.record_conversion();
                    (entry.row, entry.col, atom_to_expr(&entry.value))
                })
                .collect::<Vec<_>>();
            telemetry.record_cold_stage(IvpColdStage::AtomToExpr, conversion_started);

            let simplification_started = telemetry.start_cold_stage(IvpColdStage::Simplification);
            let converted = converted
                .into_iter()
                .map(|(row, col, expr)| (row, col, expr.simplify()))
                .collect::<Vec<_>>();
            telemetry.record_cold_stage(IvpColdStage::Simplification, simplification_started);

            let zero = Expr::parse_expression("0");
            let mut dense = vec![vec![zero.clone(); cols]; rows];
            for (row, col, expr) in converted {
                dense[row][col] = expr;
            }
            dense
        }
    }
}

fn native_argument_symbols(
    time_arg: &str,
    variables: &[String],
    equation_parameters: Option<&[String]>,
) -> Vec<Symbol> {
    std::iter::once(time_arg)
        .chain(
            equation_parameters
                .into_iter()
                .flat_map(|values| values.iter().map(String::as_str)),
        )
        .chain(variables.iter().map(String::as_str))
        .map(|name| Symbol::new(crate::wrap_symbol!(name)))
        .collect()
}

/// Prepares the native Atom Jacobian without materializing any `Expr` entries.
pub(crate) fn prepare_native_atom_jacobian(
    equations: &[Expr],
    variables: &[String],
    time_arg: &str,
    equation_parameters: Option<&[String]>,
    telemetry: &IvpTelemetry,
) -> Result<PreparedNativeAtomJacobian, IvpBackendError> {
    let preparation_started = telemetry.start_cold_stage(IvpColdStage::AtomJacobianPreparation);
    let system = prepare_native_atom_system(
        equations,
        time_arg,
        variables,
        equation_parameters,
        telemetry,
    );
    let result = prepare_native_atom_jacobian_from_system(&system, variables, telemetry);
    telemetry.record_cold_stage(IvpColdStage::AtomJacobianPreparation, preparation_started);
    result
}

fn prepare_native_atom_system(
    equations: &[Expr],
    time_arg: &str,
    variables: &[String],
    equation_parameters: Option<&[String]>,
    telemetry: &IvpTelemetry,
) -> PreparedNativeAtomSystem {
    let conversion_started = telemetry.start_cold_stage(IvpColdStage::ExprToAtom);
    let atoms: Arc<[Atom]> = equations
        .iter()
        .map(expr_to_atom)
        .collect::<Vec<_>>()
        .into();
    telemetry.record_cold_stage(IvpColdStage::ExprToAtom, conversion_started);

    PreparedNativeAtomSystem {
        atoms,
        context: PreparedVariableContext::new(&native_argument_symbols(
            time_arg,
            variables,
            equation_parameters,
        )),
        function_map: FunctionMap::new(),
    }
}

fn prepare_native_atom_jacobian_from_system(
    system: &PreparedNativeAtomSystem,
    variables: &[String],
    telemetry: &IvpTelemetry,
) -> Result<PreparedNativeAtomJacobian, IvpBackendError> {
    let rows = system.atoms.len();
    let cols = variables.len();
    let dependency_started = telemetry.start_cold_stage(IvpColdStage::AtomDependencyAnalysis);
    let prepared = PreparedSparseAtomSystem::from_shared_atoms_discovering_dependencies(
        Arc::clone(&system.atoms),
        variables,
    );
    telemetry.record_cold_stage(IvpColdStage::AtomDependencyAnalysis, dependency_started);
    let differentiation_started = telemetry.start_cold_stage(IvpColdStage::SymbolicDifferentiation);
    let sparse_entries = prepared
        .try_calc_sparse_jacobian_with_bandwidth(None)
        .map_err(
            |error: SparseAtomJacobianError| IvpBackendError::AtomDifferentiationFailure {
                row: error.row,
                col: error.col,
                source: error.source,
            },
        );
    telemetry.record_cold_stage(
        IvpColdStage::SymbolicDifferentiation,
        differentiation_started,
    );
    if sparse_entries.is_err() {
        telemetry.record_error();
    }
    let sparse_entries = sparse_entries?;

    let evaluator_started =
        telemetry.start_cold_stage(IvpColdStage::NativeJacobianEvaluatorPreparation);
    let compiled = sparse_entries
        .into_iter()
        .map(|entry| {
            PreparedEvaluator::new_with_context(&entry.value, &system.context, &system.function_map)
                .map(|evaluator| PreparedNativeAtomJacobianEntry {
                    row: entry.row,
                    col: entry.col,
                    evaluator,
                })
                .map_err(|message| IvpBackendError::AtomPreparationFailure {
                    stage: "native Jacobian evaluator compilation".to_string(),
                    message,
                })
        })
        .collect::<Result<Vec<_>, _>>();
    telemetry.record_cold_stage(
        IvpColdStage::NativeJacobianEvaluatorPreparation,
        evaluator_started,
    );

    Ok(PreparedNativeAtomJacobian {
        rows,
        cols,
        expected_input_len: system.context.vars.len(),
        entries: compiled?.into(),
    })
}

/// Compiles a caller-supplied Jacobian directly into the native Atom plan.
///
/// The residual graph remains the source of the residual callback, while the
/// explicit Jacobian expressions are converted to Atom exactly once. Zero
/// entries are omitted from the native entry list and are restored as zeros
/// only when a dense caller asks for the complete matrix.
fn prepare_native_atom_jacobian_from_explicit(
    system: &PreparedNativeAtomSystem,
    jacobian: &[Vec<Expr>],
    variables: &[String],
    telemetry: &IvpTelemetry,
) -> Result<PreparedNativeAtomJacobian, IvpBackendError> {
    let evaluator_started =
        telemetry.start_cold_stage(IvpColdStage::NativeJacobianEvaluatorPreparation);
    let compiled = jacobian
        .iter()
        .enumerate()
        .flat_map(|(row, entries)| {
            entries.iter().enumerate().filter_map(move |(col, expr)| {
                (!expr.is_zero()).then_some((row, col, expr))
            })
        })
        .map(|(row, col, expr)| {
            let atom = expr_to_atom(expr);
            PreparedEvaluator::new_with_context(&atom, &system.context, &system.function_map)
                .map(|evaluator| PreparedNativeAtomJacobianEntry {
                    row,
                    col,
                    evaluator,
                })
                .map_err(|message| IvpBackendError::AtomPreparationFailure {
                    stage: "explicit native Jacobian evaluator compilation".to_string(),
                    message,
                })
        })
        .collect::<Result<Vec<_>, _>>();
    telemetry.record_cold_stage(
        IvpColdStage::NativeJacobianEvaluatorPreparation,
        evaluator_started,
    );

    Ok(PreparedNativeAtomJacobian {
        rows: jacobian.len(),
        cols: variables.len(),
        expected_input_len: system.context.vars.len(),
        entries: compiled?.into(),
    })
}

fn compile_native_atom_residual(
    equations: &[Expr],
    time_arg: &str,
    variables: &[String],
    equation_parameters: Option<&[String]>,
    parameter_values_handle: Option<SharedIvpParameterValues>,
    telemetry: IvpTelemetry,
    execution_policy: IvpLambdifyExecutionPolicy,
) -> Result<Arc<PreparedNativeAtomResidual>, IvpBackendError> {
    let system = prepare_native_atom_system(
        equations,
        time_arg,
        variables,
        equation_parameters,
        &telemetry,
    );
    compile_native_atom_residual_from_system(
        &system,
        variables,
        parameter_values_handle,
        telemetry,
        execution_policy,
    )
}

fn compile_native_atom_residual_from_system(
    system: &PreparedNativeAtomSystem,
    variables: &[String],
    parameter_values_handle: Option<SharedIvpParameterValues>,
    telemetry: IvpTelemetry,
    execution_policy: IvpLambdifyExecutionPolicy,
) -> Result<Arc<PreparedNativeAtomResidual>, IvpBackendError> {
    let lambdification_started = telemetry.start_cold_stage(IvpColdStage::ResidualLambdification);
    let compiled = system
        .atoms
        .iter()
        .map(|atom| {
            PreparedEvaluator::new_with_context(atom, &system.context, &system.function_map)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|message| IvpBackendError::AtomPreparationFailure {
            stage: "native residual evaluator compilation".to_string(),
            message,
        });
    telemetry.record_cold_stage(IvpColdStage::ResidualLambdification, lambdification_started);
    let compiled: Arc<[PreparedEvaluator]> = compiled?.into();
    Ok(Arc::new(PreparedNativeAtomResidual {
        evaluators: compiled,
        expected_state_len: variables.len(),
        expected_input_len: system.context.vars.len(),
        parameter_values_handle,
        telemetry,
        execution_policy,
    }))
}

fn compile_native_atom_dense_jacobian(
    plan: Arc<PreparedNativeAtomJacobian>,
    variables: &[String],
    parameter_values_handle: Option<SharedIvpParameterValues>,
    telemetry: IvpTelemetry,
    execution_policy: IvpLambdifyExecutionPolicy,
) -> Arc<IvpTryDenseJacobianEval> {
    let expected_state_len = variables.len();
    Arc::new(move |t, y| {
        let callback_started = telemetry.start_warm_stage(IvpWarmStage::JacobianCallback);
        if y.len() != expected_state_len {
            telemetry.record_error();
            telemetry.record_warm_stage(IvpWarmStage::JacobianCallback, callback_started);
            return Err(IvpBackendError::InvalidStateShape {
                expected: expected_state_len,
                actual: y.len(),
            });
        }
        let binding_started = telemetry.start_warm_stage(IvpWarmStage::ArgumentBinding);
        let mut binding_recorded = false;
        let values = with_shared_parameter_values(
            parameter_values_handle.as_ref(),
            || {
                telemetry.record_warm_stage(IvpWarmStage::ArgumentBinding, binding_started);
                binding_recorded = true;
            },
            |parameters| {
                let parallel = execution_policy
                    .should_parallel_with_tasks(plan.entries.len(), plan.entries.len());
                telemetry.record_lambdify_dispatch(parallel);
                telemetry.record_scalar_evaluations(plan.entries.len());
                let evaluation_started =
                    telemetry.start_warm_stage(IvpWarmStage::JacobianEvaluation);
                let values = if parallel {
                    let mut values = vec![0.0; plan.entries.len()];
                    let worker_count = rayon::current_num_threads().max(1);
                    let chunk_size = (plan.entries.len() + worker_count - 1)
                        .checked_div(worker_count)
                        .unwrap_or(1)
                        .max(1);
                    let evaluation = plan
                        .entries
                        .par_chunks(chunk_size)
                        .zip(values.par_chunks_mut(chunk_size))
                        .enumerate()
                        .try_for_each(|(chunk_index, (entries, values))| {
                            PreparedEvaluator::evaluate_many_thread_local_ivp(
                                entries.iter().map(|entry| &entry.evaluator),
                                t,
                                parameters,
                                y.as_slice(),
                                plan.expected_input_len,
                                values,
                            )
                            .map_err(|(index, message)| {
                                IvpBackendError::AtomEvaluationFailure {
                                    stage: "Jacobian".to_string(),
                                    index: chunk_index * chunk_size + index,
                                    message,
                                }
                            })
                        });
                    evaluation.map(|()| values)
                } else {
                    plan.entries
                        .iter()
                        .enumerate()
                        .map(|(index, entry)| {
                            entry
                                .evaluator
                                .evaluate_thread_local_ivp(t, parameters, y.as_slice())
                                .map_err(|message| IvpBackendError::AtomEvaluationFailure {
                                    stage: "Jacobian".to_string(),
                                    index,
                                    message,
                                })
                        })
                        .collect::<Result<Vec<_>, _>>()
                };
                telemetry.record_warm_stage(IvpWarmStage::JacobianEvaluation, evaluation_started);
                values
            },
        );
        if !binding_recorded {
            telemetry.record_warm_stage(IvpWarmStage::ArgumentBinding, binding_started);
        }
        let values = values?;
        let output_started = telemetry.start_warm_stage(IvpWarmStage::JacobianOutputAssembly);
        let mut matrix = DMatrix::zeros(plan.rows, plan.cols);
        for (entry, value) in plan.entries.iter().zip(values) {
            matrix[(entry.row, entry.col)] = value;
        }
        telemetry.record_allocation(plan.rows * plan.cols * std::mem::size_of::<f64>());
        telemetry.record_warm_stage(IvpWarmStage::JacobianOutputAssembly, output_started);
        telemetry.record_warm_stage(IvpWarmStage::JacobianCallback, callback_started);
        telemetry.record_jacobian_evaluation_count();
        Ok(matrix)
    })
}

fn build_lambdify_args(
    t: f64,
    y: &DVector<f64>,
    parameter_values_handle: Option<&SharedIvpParameterValues>,
) -> Result<Vec<f64>, IvpBackendError> {
    // Reserve the complete ABI buffer once. Starting with only `1 + y.len()`
    // can reallocate after parameter insertion even though the final size is
    // known while the shared binding is borrowed.
    let mut args = Vec::new();
    build_linked_args_into(t, y, parameter_values_handle, &mut args)?;
    Ok(args)
}

fn build_linked_args(
    t: f64,
    y: &DVector<f64>,
    parameter_values_handle: Option<&SharedIvpParameterValues>,
) -> Result<Vec<f64>, IvpBackendError> {
    // Copy parameter scalars directly from the shared binding into the final
    // ABI buffer. Do not clone a temporary DVector first: prepared native
    // callbacks only need a flat evaluator input buffer.
    let mut args = Vec::with_capacity(1 + y.len());
    build_linked_args_into(t, y, parameter_values_handle, &mut args)?;
    Ok(args)
}

fn build_linked_args_into(
    t: f64,
    y: &DVector<f64>,
    parameter_values_handle: Option<&SharedIvpParameterValues>,
    args: &mut Vec<f64>,
) -> Result<(), IvpBackendError> {
    let parameter_values = parameter_values_handle
        .map(|handle| {
            handle
                .read()
                .map_err(|_| IvpBackendError::ParameterStatePoisoned)
        })
        .transpose()?;
    if let Some(values) = parameter_values.as_ref() {
        build_linked_args_from_parts(t, values.as_slice(), y.as_slice(), args);
    } else {
        build_linked_args_from_parts(t, &[], y.as_slice(), args);
    }
    Ok(())
}

fn build_linked_args_from_parts(
    t: f64,
    parameters: &[f64],
    state: &[f64],
    args: &mut Vec<f64>,
) {
    args.clear();
    args.reserve(1 + parameters.len() + state.len());
    args.push(t);
    args.extend(parameters.iter().copied());
    args.extend(state.iter().copied());
}

fn compatibility_residual(
    evaluator: Arc<IvpTryResidualEval>,
    telemetry: IvpTelemetry,
    residual_len: usize,
) -> Box<IvpResidualEval> {
    Box::new(move |t: f64, y: &DVector<f64>| -> DVector<f64> {
        match evaluator(t, y) {
            Ok(values) => values,
            Err(error) => {
                telemetry.record_error();
                log::warn!(
                    target: "rusted_scithe::symbolic_ivp",
                    "compatibility residual callback failed: {error}"
                );
                DVector::from_element(residual_len, f64::NAN)
            }
        }
    })
}

fn compatibility_jacobian(
    evaluator: Arc<IvpTryDenseJacobianEval>,
    telemetry: IvpTelemetry,
    rows: usize,
    cols: usize,
) -> Box<IvpDenseJacobianEval> {
    Box::new(move |t: f64, y: &DVector<f64>| -> DMatrix<f64> {
        match evaluator(t, y) {
            Ok(values) => values,
            Err(error) => {
                telemetry.record_error();
                log::warn!(
                    target: "rusted_scithe::symbolic_ivp",
                    "compatibility Jacobian callback failed: {error}"
                );
                DMatrix::from_element(rows, cols, f64::NAN)
            }
        }
    })
}

fn validate_explicit_jacobian(
    jacobian: Option<Vec<Vec<Expr>>>,
    rows: usize,
    cols: usize,
) -> Result<Option<Vec<Vec<Expr>>>, IvpBackendError> {
    let Some(jacobian) = jacobian else {
        return Ok(None);
    };
    if jacobian.len() != rows || jacobian.iter().any(|row| row.len() != cols) {
        return Err(IvpBackendError::InvalidMatrixShape {
            stage: "explicit Jacobian".to_string(),
            expected_rows: rows,
            expected_cols: cols,
            actual_rows: jacobian.len(),
            actual_cols: jacobian.first().map_or(0, Vec::len),
        });
    }
    Ok(Some(jacobian))
}

fn compile_ivp_residual(
    equations: &[Expr],
    time_arg: &str,
    variables: &[String],
    equation_parameters: Option<&[String]>,
    parameter_values_handle: Option<SharedIvpParameterValues>,
    telemetry: IvpTelemetry,
    execution_policy: IvpLambdifyExecutionPolicy,
) -> Arc<IvpTryResidualEval> {
    let mut names = Vec::with_capacity(
        1 + variables.len() + equation_parameters.map_or(0, |parameters| parameters.len()),
    );
    names.push(time_arg.to_string());
    if let Some(parameters) = equation_parameters {
        names.extend(parameters.iter().cloned());
    }
    names.extend(variables.iter().cloned());
    let name_refs = names.iter().map(|name| name.as_str()).collect::<Vec<_>>();

    let lambdification_started = telemetry.start_cold_stage(IvpColdStage::ResidualLambdification);
    let compiled = equations
        .iter()
        .map(|expr| Expr::lambdify_borrowed_thread_safe(expr, &name_refs))
        .collect::<Vec<_>>();
    telemetry.record_cold_stage(IvpColdStage::ResidualLambdification, lambdification_started);

    Arc::new(
        move |t: f64, y: &DVector<f64>| -> Result<DVector<f64>, IvpBackendError> {
            let callback_started = telemetry.start_warm_stage(IvpWarmStage::ResidualCallback);
            let binding_started = telemetry.start_warm_stage(IvpWarmStage::ArgumentBinding);
            let args = match build_lambdify_args(t, y, parameter_values_handle.as_ref()) {
                Ok(args) => args,
                Err(error) => {
                    telemetry.record_error();
                    telemetry.record_warm_stage(IvpWarmStage::ArgumentBinding, binding_started);
                    telemetry.record_warm_stage(IvpWarmStage::ResidualCallback, callback_started);
                    return Err(error);
                }
            };
            telemetry.record_copy_bytes(args.len() * std::mem::size_of::<f64>());
            telemetry.record_allocation(args.capacity() * std::mem::size_of::<f64>());
            telemetry.record_warm_stage(IvpWarmStage::ArgumentBinding, binding_started);
            telemetry.record_scalar_evaluations(compiled.len());
            let parallel =
                execution_policy.should_parallel_with_tasks(compiled.len(), compiled.len());
            telemetry.record_lambdify_dispatch(parallel);
            let evaluation_started = telemetry.start_warm_stage(IvpWarmStage::ResidualEvaluation);
            let values = if parallel {
                compiled
                    .par_iter()
                    .map(|func| func(&args))
                    .collect::<Vec<_>>()
            } else {
                compiled.iter().map(|func| func(&args)).collect::<Vec<_>>()
            };
            telemetry.record_warm_stage(IvpWarmStage::ResidualEvaluation, evaluation_started);
            telemetry.record_allocation(values.len() * std::mem::size_of::<f64>());
            let output_started = telemetry.start_warm_stage(IvpWarmStage::ResidualOutputAssembly);
            let result = DVector::from_vec(values);
            telemetry.record_warm_stage(IvpWarmStage::ResidualOutputAssembly, output_started);
            telemetry.record_warm_stage(IvpWarmStage::ResidualCallback, callback_started);
            telemetry.record_residual_evaluation_count();
            Ok(result)
        },
    )
}

fn compile_ivp_dense_jacobian(
    symbolic_jacobian: &[Vec<Expr>],
    time_arg: &str,
    variables: &[String],
    equation_parameters: Option<&[String]>,
    parameter_values_handle: Option<SharedIvpParameterValues>,
    telemetry: IvpTelemetry,
    execution_policy: IvpLambdifyExecutionPolicy,
) -> Arc<IvpTryDenseJacobianEval> {
    let rows = symbolic_jacobian.len();
    let cols = symbolic_jacobian.first().map_or(0, |row| row.len());
    let mut names = Vec::with_capacity(
        1 + variables.len() + equation_parameters.map_or(0, |parameters| parameters.len()),
    );
    names.push(time_arg.to_string());
    if let Some(parameters) = equation_parameters {
        names.extend(parameters.iter().cloned());
    }
    names.extend(variables.iter().cloned());
    let name_refs = names.iter().map(|name| name.as_str()).collect::<Vec<_>>();

    let lambdification_started = telemetry.start_cold_stage(IvpColdStage::JacobianLambdification);
    let compiled = symbolic_jacobian
        .iter()
        .flat_map(|row| {
            row.iter()
                .map(|expr| Expr::lambdify_borrowed_thread_safe(expr, &name_refs))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    telemetry.record_cold_stage(IvpColdStage::JacobianLambdification, lambdification_started);

    Arc::new(
        move |t: f64, y: &DVector<f64>| -> Result<DMatrix<f64>, IvpBackendError> {
            let callback_started = telemetry.start_warm_stage(IvpWarmStage::JacobianCallback);
            let binding_started = telemetry.start_warm_stage(IvpWarmStage::ArgumentBinding);
            let args = match build_lambdify_args(t, y, parameter_values_handle.as_ref()) {
                Ok(args) => args,
                Err(error) => {
                    telemetry.record_error();
                    telemetry.record_warm_stage(IvpWarmStage::ArgumentBinding, binding_started);
                    telemetry.record_warm_stage(IvpWarmStage::JacobianCallback, callback_started);
                    return Err(error);
                }
            };
            telemetry.record_copy_bytes(args.len() * std::mem::size_of::<f64>());
            telemetry.record_allocation(args.capacity() * std::mem::size_of::<f64>());
            telemetry.record_warm_stage(IvpWarmStage::ArgumentBinding, binding_started);
            telemetry.record_scalar_evaluations(compiled.len());
            let parallel =
                execution_policy.should_parallel_with_tasks(compiled.len(), compiled.len());
            telemetry.record_lambdify_dispatch(parallel);
            let evaluation_started = telemetry.start_warm_stage(IvpWarmStage::JacobianEvaluation);
            let values = if parallel {
                compiled
                    .par_iter()
                    .map(|func| func(&args))
                    .collect::<Vec<_>>()
            } else {
                compiled.iter().map(|func| func(&args)).collect::<Vec<_>>()
            };
            telemetry.record_warm_stage(IvpWarmStage::JacobianEvaluation, evaluation_started);
            telemetry.record_allocation(values.len() * std::mem::size_of::<f64>());
            let output_started = telemetry.start_warm_stage(IvpWarmStage::JacobianOutputAssembly);
            let result = DMatrix::from_row_slice(rows, cols, values.as_slice());
            telemetry.record_allocation(rows * cols * std::mem::size_of::<f64>());
            telemetry.record_warm_stage(IvpWarmStage::JacobianOutputAssembly, output_started);
            telemetry.record_warm_stage(IvpWarmStage::JacobianCallback, callback_started);
            telemetry.record_jacobian_evaluation_count();
            Ok(result)
        },
    )
}

/// Prepares one modern shared symbolic IVP backend from equations and options.
pub fn prepare_symbolic_ivp_problem(
    equations: Vec<Expr>,
    variables: Vec<String>,
    time_arg: String,
    options: SymbolicIvpProblemOptions,
) -> Result<PreparedSymbolicIvpProblem, IvpBackendError> {
    let explicit_jacobian = validate_explicit_jacobian(
        options.explicit_jacobian,
        equations.len(),
        variables.len(),
    )?;
    let telemetry = options.telemetry.clone();
    let validation_started = telemetry.start_cold_stage(IvpColdStage::Validation);
    let validation = validate_ivp_argument_schema(
        time_arg.as_str(),
        &variables,
        options.equation_parameters.as_deref(),
    );
    telemetry.record_cold_stage(IvpColdStage::Validation, validation_started);
    validation?;
    telemetry.set_route(match options.symbolic_assembly_backend {
        IvpSymbolicAssemblyBackend::ExprLegacy => IvpTelemetryRoute::ExprLegacy,
        IvpSymbolicAssemblyBackend::AtomViewExprCompat => IvpTelemetryRoute::AtomViewExprCompat,
        IvpSymbolicAssemblyBackend::AtomView => IvpTelemetryRoute::AtomViewNative,
    });
    telemetry.set_execution(IvpTelemetryExecution::Lambdify);
    telemetry.set_lambdify_execution_policy(options.lambdify_execution_policy);
    telemetry.set_problem_shape(
        variables.len(),
        equations.len(),
        options.equation_parameters.as_ref().map_or(0, Vec::len),
    );
    log::debug!(
        target: "rusted_scithe::symbolic_ivp",
        "preparing AtomView/ExprLegacy IVP callbacks: route={}, policy={}, states={}, residuals={}, parameters={}",
        telemetry_route(options.symbolic_assembly_backend).label(),
        options.lambdify_execution_policy.label(),
        variables.len(),
        equations.len(),
        options.equation_parameters.as_ref().map_or(0, Vec::len),
    );
    let binding_started = telemetry.start_cold_stage(IvpColdStage::ParameterBinding);
    let parameter_values_handle = match prepare_parameter_values_handle(
        options.equation_parameters.as_deref(),
        options.equation_parameter_values,
    ) {
        Ok(handle) => {
            telemetry.record_cold_stage(IvpColdStage::ParameterBinding, binding_started);
            handle
        }
        Err(error) => {
            telemetry.record_cold_stage(IvpColdStage::ParameterBinding, binding_started);
            return Err(error);
        }
    };

    let (
        symbolic_jacobian,
        try_residual,
        try_jacobian,
        native_residual,
        native_jacobian,
        native_atoms,
    ) = match options.symbolic_assembly_backend {
        IvpSymbolicAssemblyBackend::AtomView => {
            let symbolic_started = telemetry.start_cold_stage(IvpColdStage::SymbolicJacobian);
            let atom_residual_preparation =
                telemetry.scoped_cold_stage(IvpColdStage::AtomResidualPreparation);
            let atom_system = prepare_native_atom_system(
                &equations,
                time_arg.as_str(),
                &variables,
                options.equation_parameters.as_deref(),
                &telemetry,
            );
            drop(atom_residual_preparation);
            let atom_jacobian_preparation =
                telemetry.scoped_cold_stage(IvpColdStage::AtomJacobianPreparation);
            let native_jacobian = match explicit_jacobian.as_deref() {
                Some(jacobian) => prepare_native_atom_jacobian_from_explicit(
                    &atom_system,
                    jacobian,
                    &variables,
                    &telemetry,
                )?,
                None => {
                    prepare_native_atom_jacobian_from_system(&atom_system, &variables, &telemetry)?
                }
            };
            drop(atom_jacobian_preparation);
            telemetry.record_cold_stage(IvpColdStage::SymbolicJacobian, symbolic_started);
            telemetry.record_symbolic_jacobian_build();
            let residual_started = telemetry.start_cold_stage(IvpColdStage::ResidualCompilation);
            let atom_residual_compilation =
                telemetry.scoped_cold_stage(IvpColdStage::AtomResidualPreparation);
            let native_residual = compile_native_atom_residual_from_system(
                &atom_system,
                &variables,
                parameter_values_handle.clone(),
                telemetry.clone(),
                options.lambdify_execution_policy,
            )?;
            drop(atom_residual_compilation);
            let try_residual: Arc<IvpTryResidualEval> = {
                let native_residual = Arc::clone(&native_residual);
                Arc::new(move |t, y| native_residual.evaluate(t, y))
            };
            telemetry.record_cold_stage(IvpColdStage::ResidualCompilation, residual_started);
            let jacobian_started = telemetry.start_cold_stage(IvpColdStage::JacobianCompilation);
            let native_jacobian = Arc::new(native_jacobian);
            let try_jacobian = compile_native_atom_dense_jacobian(
                Arc::clone(&native_jacobian),
                &variables,
                parameter_values_handle.clone(),
                telemetry.clone(),
                options.lambdify_execution_policy,
            );
            telemetry.record_cold_stage(IvpColdStage::JacobianCompilation, jacobian_started);
            (
                explicit_jacobian.clone().unwrap_or_default(),
                try_residual,
                try_jacobian,
                Some(native_residual),
                Some(native_jacobian),
                Some(Arc::clone(&atom_system.atoms)),
            )
        }
        IvpSymbolicAssemblyBackend::ExprLegacy | IvpSymbolicAssemblyBackend::AtomViewExprCompat => {
            let symbolic_started = telemetry.start_cold_stage(IvpColdStage::SymbolicJacobian);
            let symbolic_jacobian = explicit_jacobian.clone().unwrap_or_else(|| {
                build_symbolic_jacobian(
                    &equations,
                    &variables,
                    options.symbolic_assembly_backend,
                    &telemetry,
                )
            });
            telemetry.record_cold_stage(IvpColdStage::SymbolicJacobian, symbolic_started);
            telemetry.record_symbolic_jacobian_build();
            let residual_started = telemetry.start_cold_stage(IvpColdStage::ResidualCompilation);
            let try_residual = compile_ivp_residual(
                &equations,
                time_arg.as_str(),
                &variables,
                options.equation_parameters.as_deref(),
                parameter_values_handle.clone(),
                telemetry.clone(),
                options.lambdify_execution_policy,
            );
            telemetry.record_cold_stage(IvpColdStage::ResidualCompilation, residual_started);
            let jacobian_started = telemetry.start_cold_stage(IvpColdStage::JacobianCompilation);
            let try_jacobian = compile_ivp_dense_jacobian(
                &symbolic_jacobian,
                time_arg.as_str(),
                &variables,
                options.equation_parameters.as_deref(),
                parameter_values_handle.clone(),
                telemetry.clone(),
                options.lambdify_execution_policy,
            );
            telemetry.record_cold_stage(IvpColdStage::JacobianCompilation, jacobian_started);
            (
                symbolic_jacobian,
                try_residual,
                try_jacobian,
                None,
                None,
                None,
            )
        }
    };
    let residual = compatibility_residual(try_residual.clone(), telemetry.clone(), equations.len());
    let jacobian = compatibility_jacobian(
        try_jacobian.clone(),
        telemetry.clone(),
        equations.len(),
        variables.len(),
    );

    let backend_kind = match options.backend_policy {
        IvpBackendSelectionPolicy::LambdifyOnly => IvpBackendKind::Lambdify,
        IvpBackendSelectionPolicy::PreferAotThenLambdify => IvpBackendKind::Lambdify,
    };

    Ok(PreparedSymbolicIvpProblem {
        residual,
        jacobian,
        try_residual,
        try_jacobian,
        native_residual,
        native_jacobian,
        linked_residual: None,
        _linked_dense: None,
        native_atoms,
        equations,
        symbolic_jacobian,
        explicit_jacobian,
        time_arg,
        variables,
        equation_parameters: options.equation_parameters,
        parameter_values_handle,
        backend_kind,
        telemetry,
        execution_policy: options.lambdify_execution_policy,
    })
}

/// Prepares only the residual callback for IVP symbolic solves.
///
/// Unlike [`prepare_symbolic_ivp_problem`], this does not differentiate or
/// compile a dense Jacobian.  It is intended for LSODE2 native sparse/banded
/// Jacobian paths.
pub fn prepare_symbolic_ivp_residual_problem(
    equations: Vec<Expr>,
    variables: Vec<String>,
    time_arg: String,
    options: SymbolicIvpProblemOptions,
) -> Result<PreparedSymbolicIvpResidualProblem, IvpBackendError> {
    let explicit_jacobian = validate_explicit_jacobian(
        options.explicit_jacobian,
        equations.len(),
        variables.len(),
    )?;
    let telemetry = options.telemetry.clone();
    let validation_started = telemetry.start_cold_stage(IvpColdStage::Validation);
    let validation = validate_ivp_argument_schema(
        time_arg.as_str(),
        &variables,
        options.equation_parameters.as_deref(),
    );
    telemetry.record_cold_stage(IvpColdStage::Validation, validation_started);
    validation?;
    telemetry.set_route(match options.symbolic_assembly_backend {
        IvpSymbolicAssemblyBackend::ExprLegacy => IvpTelemetryRoute::ExprLegacy,
        IvpSymbolicAssemblyBackend::AtomViewExprCompat => IvpTelemetryRoute::AtomViewExprCompat,
        IvpSymbolicAssemblyBackend::AtomView => IvpTelemetryRoute::AtomViewNative,
    });
    telemetry.set_execution(IvpTelemetryExecution::Lambdify);
    telemetry.set_lambdify_execution_policy(options.lambdify_execution_policy);
    telemetry.set_problem_shape(
        variables.len(),
        equations.len(),
        options.equation_parameters.as_ref().map_or(0, Vec::len),
    );
    log::debug!(
        target: "rusted_scithe::symbolic_ivp",
        "preparing residual-only IVP callbacks: route={}, policy={}, states={}, residuals={}, parameters={}",
        telemetry_route(options.symbolic_assembly_backend).label(),
        options.lambdify_execution_policy.label(),
        variables.len(),
        equations.len(),
        options.equation_parameters.as_ref().map_or(0, Vec::len),
    );
    let binding_started = telemetry.start_cold_stage(IvpColdStage::ParameterBinding);
    let parameter_values_handle = match prepare_parameter_values_handle(
        options.equation_parameters.as_deref(),
        options.equation_parameter_values,
    ) {
        Ok(handle) => {
            telemetry.record_cold_stage(IvpColdStage::ParameterBinding, binding_started);
            handle
        }
        Err(error) => {
            telemetry.record_cold_stage(IvpColdStage::ParameterBinding, binding_started);
            return Err(error);
        }
    };
    let residual_started = telemetry.start_cold_stage(IvpColdStage::ResidualCompilation);
    let (try_residual, native_residual, native_atoms) = match options.symbolic_assembly_backend {
        IvpSymbolicAssemblyBackend::AtomView => {
            let _atom_residual_preparation =
                telemetry.scoped_cold_stage(IvpColdStage::AtomResidualPreparation);
            let atom_system = prepare_native_atom_system(
                &equations,
                time_arg.as_str(),
                &variables,
                options.equation_parameters.as_deref(),
                &telemetry,
            );
            let native_atoms = Arc::clone(&atom_system.atoms);
            let native_residual = compile_native_atom_residual_from_system(
                &atom_system,
                &variables,
                parameter_values_handle.clone(),
                telemetry.clone(),
                options.lambdify_execution_policy,
            )?;
            let runtime = Arc::clone(&native_residual);
            let try_residual: Arc<IvpTryResidualEval> =
                Arc::new(move |t, y| runtime.evaluate(t, y));
            (try_residual, Some(native_residual), Some(native_atoms))
        }
        IvpSymbolicAssemblyBackend::ExprLegacy | IvpSymbolicAssemblyBackend::AtomViewExprCompat => {
            (
                compile_ivp_residual(
                    &equations,
                    time_arg.as_str(),
                    &variables,
                    options.equation_parameters.as_deref(),
                    parameter_values_handle.clone(),
                    telemetry.clone(),
                    options.lambdify_execution_policy,
                ),
                None,
                None,
            )
        }
    };
    telemetry.record_cold_stage(IvpColdStage::ResidualCompilation, residual_started);
    let residual = compatibility_residual(try_residual.clone(), telemetry.clone(), equations.len());

    Ok(PreparedSymbolicIvpResidualProblem {
        residual,
        try_residual,
        native_residual,
        linked_residual: None,
        native_atoms,
        equations,
        explicit_jacobian,
        time_arg,
        variables,
        equation_parameters: options.equation_parameters,
        parameter_values_handle,
        backend_kind: IvpBackendKind::Lambdify,
        telemetry,
        execution_policy: options.lambdify_execution_policy,
    })
}

fn telemetry_route(backend: IvpSymbolicAssemblyBackend) -> IvpTelemetryRoute {
    match backend {
        IvpSymbolicAssemblyBackend::ExprLegacy => IvpTelemetryRoute::ExprLegacy,
        IvpSymbolicAssemblyBackend::AtomViewExprCompat => IvpTelemetryRoute::AtomViewExprCompat,
        IvpSymbolicAssemblyBackend::AtomView => IvpTelemetryRoute::AtomViewNative,
    }
}

fn validate_ivp_argument_schema(
    time_arg: &str,
    variables: &[String],
    parameters: Option<&[String]>,
) -> Result<(), IvpBackendError> {
    let mut names =
        HashSet::with_capacity(1 + variables.len() + parameters.map_or(0, |values| values.len()));
    for (kind, name) in std::iter::once(("time", time_arg))
        .chain(variables.iter().map(|name| ("state", name.as_str())))
        .chain(
            parameters
                .into_iter()
                .flat_map(|values| values.iter().map(|name| ("parameter", name.as_str()))),
        )
    {
        if name.trim().is_empty() {
            return Err(IvpBackendError::InvalidArgumentSchema {
                message: format!("{kind} argument name must not be empty"),
            });
        }
        if !names.insert(name) {
            return Err(IvpBackendError::InvalidArgumentSchema {
                message: format!("duplicate IVP callback argument name `{name}`"),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linked_aot_residual_workspace_reuses_flat_arguments() {
        let telemetry = IvpTelemetry::counters();
        let problem = prepare_symbolic_ivp_residual_problem(
            vec![Expr::parse_expression("a*y + t")],
            vec!["y".to_string()],
            "t".to_string(),
            SymbolicIvpProblemOptions::new()
                .with_symbolic_assembly_backend(IvpSymbolicAssemblyBackend::AtomView)
                .with_equation_parameters(vec!["a".to_string()])
                .with_equation_parameter_values(DVector::from_vec(vec![2.0]))
                .with_telemetry(telemetry.clone()),
        )
        .expect("residual-only IVP problem should prepare");
        let linked = LinkedResidualAotBackend::new(
            "workspace-test",
            1,
            Arc::new(|args, out| {
                out[0] = args[0] + args[1] * args[2];
            }),
        );
        let problem = problem.into_linked_residual_backend(linked);
        let state = DVector::from_vec(vec![3.0]);
        let mut output = DVector::zeros(1);
        let mut args = Vec::new();

        problem
            .try_evaluate_residual_into_with_workspace(1.0, &state, &mut output, &mut args)
            .expect("linked residual should evaluate through workspace");
        assert_eq!(output[0], 7.0);
        let capacity_after_first = args.capacity();
        let first_snapshot = telemetry.snapshot();

        problem
            .try_evaluate_residual_into_with_workspace(1.0, &state, &mut output, &mut args)
            .expect("linked residual should reuse workspace");
        assert_eq!(output[0], 7.0);
        assert_eq!(args.capacity(), capacity_after_first);

        problem
            .set_parameter_values(DVector::from_vec(vec![4.0]))
            .expect("parameter rebind should succeed");
        problem
            .try_evaluate_residual_into_with_workspace(1.0, &state, &mut output, &mut args)
            .expect("rebound linked residual should evaluate");
        assert_eq!(output[0], 13.0);

        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.allocated_bytes, first_snapshot.allocated_bytes);
        assert_eq!(snapshot.residual_evaluations, 3);
        assert_eq!(
            snapshot.copied_bytes,
            (3 * 3 * std::mem::size_of::<f64>()) as u64
        );
    }

    #[test]
    fn linked_dense_runtime_rebind_updates_residual_and_jacobian_without_republication() {
        let telemetry = IvpTelemetry::counters();
        let problem = prepare_symbolic_ivp_problem(
            vec![Expr::parse_expression("a*y + t")],
            vec!["y".to_string()],
            "t".to_string(),
            SymbolicIvpProblemOptions::new()
                .with_equation_parameters(vec!["a".to_string()])
                .with_equation_parameter_values(DVector::from_vec(vec![2.0]))
                .with_telemetry(telemetry.clone()),
        )
        .expect("parameterized IVP problem should prepare");
        let linked = LinkedDenseAotBackend::new(
            "dense-rebind-test",
            1,
            (1, 1),
            Arc::new(|args, out| out[0] = args[0] + args[1] * args[2]),
            Arc::new(|args, out| out[0] = args[1]),
        );
        let problem = problem.into_linked_dense_backend(linked);
        let allocation_baseline = telemetry.snapshot().allocated_bytes;
        let state = DVector::from_vec(vec![3.0]);
        let first_residual = problem
            .try_evaluate_residual(1.0, &state)
            .expect("linked Dense residual should evaluate");
        let first_jacobian = problem
            .try_evaluate_jacobian(1.0, &state)
            .expect("linked Dense Jacobian should evaluate");
        assert_eq!(first_residual[0], 7.0);
        assert_eq!(first_jacobian[(0, 0)], 2.0);

        problem
            .set_parameter_values(DVector::from_vec(vec![4.0]))
            .expect("linked Dense parameter rebind should succeed");
        let rebound_residual = problem
            .try_evaluate_residual(1.0, &state)
            .expect("rebound linked Dense residual should evaluate");
        let rebound_jacobian = problem
            .try_evaluate_jacobian(1.0, &state)
            .expect("rebound linked Dense Jacobian should evaluate");
        assert_eq!(rebound_residual[0], 13.0);
        assert_eq!(rebound_jacobian[(0, 0)], 4.0);
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.jacobian_evaluations, 2);
        assert_eq!(
            snapshot.allocated_bytes - allocation_baseline,
            (2 * 2 * std::mem::size_of::<f64>()) as u64,
            "each owned dense Jacobian has one row-major ABI buffer and one DMatrix allocation"
        );
    }

    #[test]
    fn linked_dense_row_major_workspace_matches_owned_matrix_without_output_allocation() {
        let telemetry = IvpTelemetry::counters();
        let problem = prepare_symbolic_ivp_problem(
            vec![
                Expr::parse_expression("a*y + t"),
                Expr::parse_expression("y"),
            ],
            vec!["y".to_string(), "z".to_string()],
            "t".to_string(),
            SymbolicIvpProblemOptions::new()
                .with_equation_parameters(vec!["a".to_string()])
                .with_equation_parameter_values(DVector::from_vec(vec![2.0]))
                .with_telemetry(telemetry.clone()),
        )
        .expect("parameterized dense IVP problem should prepare");
        let linked = LinkedDenseAotBackend::new(
            "dense-row-major-workspace-test",
            2,
            (2, 2),
            Arc::new(|args, out| {
                out[0] = args[1];
                out[1] = args[2];
                out[2] = args[1] + args[2];
                out[3] = args[0];
            }),
            Arc::new(|_args, out| {
                out.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
            }),
        );
        let problem = problem.into_linked_dense_backend(linked);
        let state = DVector::from_vec(vec![3.0, 5.0]);
        let owned = problem
            .try_evaluate_jacobian(1.0, &state)
            .expect("owned linked Jacobian should evaluate");
        let mut row_major = vec![0.0; 4];
        let mut args = Vec::new();
        problem
            .try_evaluate_jacobian_row_major_into_with_workspace(
                1.0,
                &state,
                &mut row_major,
                &mut args,
            )
            .expect("row-major linked Jacobian should evaluate");
        let mut dmatrix = DMatrix::zeros(2, 2);
        let mut values = Vec::new();
        problem
            .try_evaluate_jacobian_into_dmatrix_with_workspace(
                1.0,
                &state,
                &mut dmatrix,
                &mut values,
                &mut args,
            )
            .expect("DMatrix linked Jacobian should evaluate");
        let args_capacity = args.capacity();
        problem
            .try_evaluate_jacobian_row_major_into_with_workspace(
                1.0,
                &state,
                &mut row_major,
                &mut args,
            )
            .expect("row-major linked Jacobian should reuse its workspace");

        assert_eq!(
            row_major,
            vec![owned[(0, 0)], owned[(0, 1)], owned[(1, 0)], owned[(1, 1)]]
        );
        assert_eq!(dmatrix, owned);
        assert_eq!(args.capacity(), args_capacity);
        assert_eq!(telemetry.snapshot().jacobian_evaluations, 4);
    }

    #[test]
    fn parameterized_ivp_backend_updates_values_without_recompiling_callbacks() {
        let equations = vec![Expr::parse_expression("a*y + t")];
        let problem = prepare_symbolic_ivp_problem(
            equations,
            vec!["y".to_string()],
            "t".to_string(),
            SymbolicIvpProblemOptions::new()
                .with_equation_parameters(vec!["a".to_string()])
                .with_equation_parameter_values(DVector::from_vec(vec![2.0])),
        )
        .expect("parameterized IVP backend should prepare");

        let y = DVector::from_vec(vec![3.0]);
        let initial = (problem.residual)(1.0, &y);
        assert_eq!(initial, DVector::from_vec(vec![7.0]));

        problem
            .set_parameter_values(DVector::from_vec(vec![4.0]))
            .expect("parameter update should succeed");
        let updated = (problem.residual)(1.0, &y);
        assert_eq!(updated, DVector::from_vec(vec![13.0]));
    }

    #[test]
    fn prepared_ivp_typed_callbacks_match_compatibility_callbacks() {
        let problem = prepare_symbolic_ivp_problem(
            vec![Expr::parse_expression("a*y + t")],
            vec!["y".to_string()],
            "t".to_string(),
            SymbolicIvpProblemOptions::new()
                .with_equation_parameters(vec!["a".to_string()])
                .with_equation_parameter_values(DVector::from_vec(vec![2.0])),
        )
        .expect("parameterized IVP backend should prepare");
        let y = DVector::from_vec(vec![3.0]);

        let residual = problem.try_evaluate_residual(0.5, &y).unwrap();
        let compatibility_residual = (problem.residual)(0.5, &y);
        assert!((residual[0] - compatibility_residual[0]).abs() < 1e-12);

        let jacobian = problem.try_evaluate_jacobian(0.5, &y).unwrap();
        let compatibility_jacobian = (problem.jacobian)(0.5, &y);
        assert!((jacobian[(0, 0)] - compatibility_jacobian[(0, 0)]).abs() < 1e-12);
    }

    #[test]
    fn parameterized_ivp_backend_rejects_parameter_length_mismatch() {
        let result = prepare_symbolic_ivp_problem(
            vec![Expr::parse_expression("a*y + t")],
            vec!["y".to_string()],
            "t".to_string(),
            SymbolicIvpProblemOptions::new()
                .with_equation_parameters(vec!["a".to_string(), "b".to_string()])
                .with_equation_parameter_values(DVector::from_vec(vec![2.0])),
        );

        match result {
            Err(IvpBackendError::ParameterCountMismatch { expected, actual }) => {
                assert_eq!(expected, 2);
                assert_eq!(actual, 1);
            }
            Err(other) => panic!("expected ParameterCountMismatch, got {other}"),
            Ok(_) => panic!("expected ParameterCountMismatch, got Ok(..)"),
        }
    }

    #[test]
    fn ivp_backend_rejects_duplicate_callback_argument_names() {
        let telemetry = IvpTelemetry::counters();
        let result = prepare_symbolic_ivp_problem(
            vec![Expr::parse_expression("a*y + t")],
            vec!["y".to_string()],
            "t".to_string(),
            SymbolicIvpProblemOptions::new()
                .with_equation_parameters(vec!["y".to_string()])
                .with_telemetry(telemetry.clone()),
        );

        match result {
            Err(IvpBackendError::InvalidArgumentSchema { message }) => {
                assert!(message.contains("duplicate"));
                assert!(message.contains("`y`"));
            }
            Err(other) => panic!("expected InvalidArgumentSchema, got {other}"),
            Ok(_) => panic!("expected InvalidArgumentSchema, got Ok(..)"),
        }
        assert_eq!(
            telemetry
                .snapshot()
                .cold_stage(IvpColdStage::Validation)
                .calls,
            1
        );
    }

    #[test]
    fn ivp_backend_preserves_parameter_binding_stage_on_failure() {
        let telemetry = IvpTelemetry::counters();
        let result = prepare_symbolic_ivp_problem(
            vec![Expr::parse_expression("a*y + t")],
            vec!["y".to_string()],
            "t".to_string(),
            SymbolicIvpProblemOptions::new()
                .with_equation_parameters(vec!["a".to_string(), "b".to_string()])
                .with_equation_parameter_values(DVector::from_vec(vec![2.0]))
                .with_telemetry(telemetry.clone()),
        );

        assert!(matches!(
            result,
            Err(IvpBackendError::ParameterCountMismatch {
                expected: 2,
                actual: 1
            })
        ));
        assert_eq!(
            telemetry
                .snapshot()
                .cold_stage(IvpColdStage::ParameterBinding)
                .calls,
            1
        );
    }

    #[test]
    fn symbolic_ivp_atom_view_and_expr_legacy_jacobians_match() {
        let equations = vec![
            Expr::parse_expression("a*t + y + b*z"),
            Expr::parse_expression("c*y - z + b*t"),
        ];
        let variables = vec!["y".to_string(), "z".to_string()];
        let params = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let pvals = DVector::from_vec(vec![2.0, -0.5, 3.0]);
        let y = DVector::from_vec(vec![1.25, -0.75]);
        let t = 0.6_f64;

        let legacy = prepare_symbolic_ivp_problem(
            equations.clone(),
            variables.clone(),
            "t".to_string(),
            SymbolicIvpProblemOptions::new()
                .with_equation_parameters(params.clone())
                .with_equation_parameter_values(pvals.clone())
                .with_symbolic_assembly_backend(IvpSymbolicAssemblyBackend::ExprLegacy),
        )
        .expect("ExprLegacy IVP backend should prepare");
        let atom = prepare_symbolic_ivp_problem(
            equations,
            variables,
            "t".to_string(),
            SymbolicIvpProblemOptions::new()
                .with_equation_parameters(params)
                .with_equation_parameter_values(pvals)
                .with_symbolic_assembly_backend(IvpSymbolicAssemblyBackend::AtomView),
        )
        .expect("AtomView IVP backend should prepare");

        let j_legacy = (legacy.jacobian)(t, &y);
        let j_atom = (atom.jacobian)(t, &y);
        assert_eq!(
            j_legacy.shape(),
            j_atom.shape(),
            "Jacobian shape should match across symbolic assembly backends"
        );
        let max_diff = j_legacy
            .iter()
            .zip(j_atom.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff <= 1.0e-12,
            "AtomView and ExprLegacy Jacobians should match numerically; max_diff={max_diff:e}"
        );
        assert_eq!(
            atom.telemetry
                .snapshot()
                .cold_stage(IvpColdStage::AtomToExpr)
                .calls,
            0,
            "native AtomView must not materialize Jacobian entries back into Expr"
        );
    }

    #[test]
    fn symbolic_ivp_three_frontends_preserve_componentwise_values_and_nonfinite_behavior() {
        let equations = vec![
            Expr::parse_expression("a*y + t"),
            Expr::parse_expression("y - a*z + exp(t)"),
        ];
        let variables = vec!["y".to_string(), "z".to_string()];
        let parameters = vec!["a".to_string()];
        let make_problem = |backend, telemetry: IvpTelemetry| {
            prepare_symbolic_ivp_problem(
                equations.clone(),
                variables.clone(),
                "t".to_string(),
                SymbolicIvpProblemOptions::new()
                    .with_equation_parameters(parameters.clone())
                    .with_equation_parameter_values(DVector::from_vec(vec![2.5]))
                    .with_symbolic_assembly_backend(backend)
                    .with_telemetry(telemetry),
            )
            .expect("frontend parity problem should prepare")
        };

        let legacy_telemetry = IvpTelemetry::counters();
        let compat_telemetry = IvpTelemetry::counters();
        let native_telemetry = IvpTelemetry::counters();
        let mut legacy = make_problem(IvpSymbolicAssemblyBackend::ExprLegacy, legacy_telemetry);
        let mut compat = make_problem(
            IvpSymbolicAssemblyBackend::AtomViewExprCompat,
            compat_telemetry,
        );
        let mut native = make_problem(IvpSymbolicAssemblyBackend::AtomView, native_telemetry);

        for (parameter_value, time, state) in [
            (2.5_f64, 0.0, DVector::from_vec(vec![1.0, -0.5])),
            (2.5_f64, 0.37, DVector::from_vec(vec![-2.0, 3.25])),
            (-1.25_f64, 0.19, DVector::from_vec(vec![0.75, 2.0])),
        ] {
            if (parameter_value - 2.5).abs() > f64::EPSILON {
                let values = DVector::from_vec(vec![parameter_value]);
                legacy
                    .set_parameter_values(values.clone())
                    .expect("ExprLegacy parameter rebind should succeed");
                compat
                    .set_parameter_values(values.clone())
                    .expect("AtomViewExprCompat parameter rebind should succeed");
                native
                    .set_parameter_values(values)
                    .expect("AtomViewNative parameter rebind should succeed");
            }
            let legacy_residual = legacy
                .try_evaluate_residual(time, &state)
                .expect("ExprLegacy residual should evaluate");
            let compat_residual = compat
                .try_evaluate_residual(time, &state)
                .expect("AtomViewExprCompat residual should evaluate");
            let native_residual = native
                .try_evaluate_residual(time, &state)
                .expect("AtomViewNative residual should evaluate");
            for ((expected, compat_value), native_value) in legacy_residual
                .iter()
                .zip(compat_residual.iter())
                .zip(native_residual.iter())
            {
                assert!((expected - compat_value).abs() <= 1.0e-12);
                assert!((expected - native_value).abs() <= 1.0e-12);
            }

            let legacy_jacobian = legacy
                .try_evaluate_jacobian(time, &state)
                .expect("ExprLegacy Jacobian should evaluate");
            let compat_jacobian = compat
                .try_evaluate_jacobian(time, &state)
                .expect("AtomViewExprCompat Jacobian should evaluate");
            let native_jacobian = native
                .try_evaluate_jacobian(time, &state)
                .expect("AtomViewNative Jacobian should evaluate");
            for ((expected, compat_value), native_value) in legacy_jacobian
                .iter()
                .zip(compat_jacobian.iter())
                .zip(native_jacobian.iter())
            {
                assert!((expected - compat_value).abs() <= 1.0e-12);
                assert!((expected - native_value).abs() <= 1.0e-12);
            }
        }

        let nonfinite_state = DVector::from_vec(vec![f64::NAN, 1.0]);
        let nonfinite_results = [
            legacy
                .try_evaluate_residual(0.5, &nonfinite_state)
                .expect("ExprLegacy should return a typed result for NaN input"),
            compat
                .try_evaluate_residual(0.5, &nonfinite_state)
                .expect("AtomViewExprCompat should return a typed result for NaN input"),
            native
                .try_evaluate_residual(0.5, &nonfinite_state)
                .expect("AtomViewNative should return a typed result for NaN input"),
        ];
        for values in &nonfinite_results {
            assert!(values.iter().all(|value| value.is_nan()));
        }

        let legacy_snapshot = legacy.telemetry.snapshot();
        let compat_snapshot = compat.telemetry.snapshot();
        let native_snapshot = native.telemetry.snapshot();
        assert!(
            legacy_snapshot.cold_stage(IvpColdStage::AtomToExpr).calls == 0
                && compat_snapshot.cold_stage(IvpColdStage::AtomToExpr).calls > 0
                && native_snapshot.cold_stage(IvpColdStage::AtomToExpr).calls == 0
        );
    }

    #[test]
    fn native_atomview_residual_and_jacobian_support_rebind_and_typed_shape_errors() {
        let telemetry = IvpTelemetry::counters();
        let problem = prepare_symbolic_ivp_problem(
            vec![
                Expr::parse_expression("a*y + t"),
                Expr::parse_expression("y*y - a"),
            ],
            vec!["y".to_string(), "z".to_string()],
            "t".to_string(),
            SymbolicIvpProblemOptions::new()
                .with_equation_parameters(vec!["a".to_string()])
                .with_equation_parameter_values(DVector::from_vec(vec![2.0]))
                .with_symbolic_assembly_backend(IvpSymbolicAssemblyBackend::AtomView)
                .with_telemetry(telemetry.clone()),
        )
        .expect("native AtomView problem should prepare");

        let state = DVector::from_vec(vec![3.0, 4.0]);
        let initial = problem
            .try_evaluate_jacobian(1.0, &state)
            .expect("native Jacobian should evaluate");
        assert_eq!(initial[(0, 0)], 2.0);
        assert_eq!(initial[(1, 0)], 6.0);

        problem
            .set_parameter_values(DVector::from_vec(vec![5.0]))
            .expect("native parameter rebind should succeed");
        let rebound = problem
            .try_evaluate_residual(1.0, &state)
            .expect("native residual should use rebound parameters");
        assert_eq!(rebound[0], 16.0);
        assert_eq!(rebound[1], 4.0);

        let error = problem.try_evaluate_residual(1.0, &DVector::from_vec(vec![3.0]));
        assert!(matches!(
            error,
            Err(IvpBackendError::InvalidStateShape {
                expected: 2,
                actual: 1
            })
        ));
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.cold_stage(IvpColdStage::AtomToExpr).calls, 0);
        assert_eq!(
            snapshot.cold_stage(IvpColdStage::ExprToAtom).calls,
            1,
            "native residual and Jacobian preparation should share one Atom payload"
        );
        assert!(snapshot.residual_evaluations > 0);
        assert!(snapshot.jacobian_evaluations > 0);
    }

    #[test]
    fn native_atomview_residual_into_reuses_output_without_flat_parameter_copy() {
        let telemetry = IvpTelemetry::counters();
        let problem = prepare_symbolic_ivp_problem(
            vec![Expr::parse_expression("a*y + t")],
            vec!["y".to_string()],
            "t".to_string(),
            SymbolicIvpProblemOptions::new()
                .with_equation_parameters(vec!["a".to_string()])
                .with_equation_parameter_values(DVector::from_vec(vec![2.0]))
                .with_symbolic_assembly_backend(IvpSymbolicAssemblyBackend::AtomView)
                .with_telemetry(telemetry.clone()),
        )
        .expect("native AtomView residual should prepare");

        let state = DVector::from_vec(vec![3.0]);
        let mut output = DVector::zeros(1);
        problem
            .try_evaluate_residual_into(1.0, &state, &mut output)
            .expect("caller-owned residual output should evaluate");
        assert_eq!(output[0], 7.0);
        let first = telemetry.snapshot();

        problem
            .try_evaluate_residual_into(1.0, &state, &mut output)
            .expect("reused caller-owned residual output should evaluate");
        assert_eq!(output[0], 7.0);
        let second = telemetry.snapshot();
        assert_eq!(second.copied_bytes, first.copied_bytes);
        assert_eq!(second.allocated_bytes, first.allocated_bytes);

        problem
            .set_parameter_values(DVector::from_vec(vec![4.0]))
            .expect("native parameter rebind should succeed");
        problem
            .try_evaluate_residual_into(1.0, &state, &mut output)
            .expect("rebound caller-owned residual output should evaluate");
        assert_eq!(output[0], 13.0);
    }

    #[test]
    fn native_atomview_residual_into_batches_multiple_evaluators() {
        let telemetry = IvpTelemetry::counters();
        let problem = prepare_symbolic_ivp_problem(
            vec![
                Expr::parse_expression("a*y + t"),
                Expr::parse_expression("y*y - a"),
                Expr::parse_expression("exp(y) + a"),
            ],
            vec!["y".to_string()],
            "t".to_string(),
            SymbolicIvpProblemOptions::new()
                .with_equation_parameters(vec!["a".to_string()])
                .with_equation_parameter_values(DVector::from_vec(vec![2.0]))
                .with_symbolic_assembly_backend(IvpSymbolicAssemblyBackend::AtomView)
                .with_telemetry(telemetry.clone()),
        )
        .expect("native AtomView residual batch should prepare");

        let state = DVector::from_vec(vec![3.0]);
        let mut output = DVector::zeros(3);
        problem
            .try_evaluate_residual_into(1.0, &state, &mut output)
            .expect("native residual batch should evaluate");
        assert_eq!(output[0], 7.0);
        assert_eq!(output[1], 7.0);
        assert!((output[2] - (3.0_f64.exp() + 2.0)).abs() <= 1.0e-12);

        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.residual_evaluations, 1);
        assert_eq!(snapshot.scalar_evaluations, 3);

        output.fill(0.0);
        problem
            .try_evaluate_residual_into(1.0, &state, &mut output)
            .expect("reused native residual batch should evaluate");
        assert_eq!(output[0], 7.0);
        assert_eq!(output[1], 7.0);
        assert!((output[2] - (3.0_f64.exp() + 2.0)).abs() <= 1.0e-12);
        assert_eq!(telemetry.snapshot().scalar_evaluations, 6);
    }

    #[test]
    fn failed_parameter_rebind_preserves_previous_native_binding() {
        let telemetry = IvpTelemetry::counters();
        let problem = prepare_symbolic_ivp_problem(
            vec![
                Expr::parse_expression("a*y + t"),
                Expr::parse_expression("a - y"),
            ],
            vec!["y".to_string()],
            "t".to_string(),
            SymbolicIvpProblemOptions::new()
                .with_equation_parameters(vec!["a".to_string()])
                .with_equation_parameter_values(DVector::from_vec(vec![2.0]))
                .with_symbolic_assembly_backend(IvpSymbolicAssemblyBackend::AtomView)
                .with_telemetry(telemetry.clone()),
        )
        .expect("native AtomView problem should prepare");

        let state = DVector::from_vec(vec![3.0]);
        let before = problem
            .try_evaluate_residual(1.0, &state)
            .expect("initial native residual should evaluate");
        assert_eq!(before[0], 7.0);
        assert_eq!(before[1], -1.0);
        let binds_before = telemetry.snapshot().parameter_binds;

        let error = problem.set_parameter_values(DVector::from_vec(vec![5.0, 6.0]));
        assert!(matches!(
            error,
            Err(IvpBackendError::ParameterCountMismatch {
                expected: 1,
                actual: 2
            })
        ));

        let after = problem
            .try_evaluate_residual(1.0, &state)
            .expect("failed rebind must not invalidate native callback");
        assert_eq!(after, before);
        assert_eq!(telemetry.snapshot().parameter_binds, binds_before);
    }

    #[test]
    fn lambdify_execution_policies_preserve_callbacks_and_report_dispatches() {
        let equations = vec![
            Expr::parse_expression("a*t + y + b*z"),
            Expr::parse_expression("c*y - z + b*t"),
        ];
        let variables = vec!["y".to_string(), "z".to_string()];
        let parameters = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let values = DVector::from_vec(vec![2.0, -0.5, 3.0]);
        let state = DVector::from_vec(vec![1.25, -0.75]);
        let time = 0.6_f64;

        let prepare = |policy| {
            prepare_symbolic_ivp_problem(
                equations.clone(),
                variables.clone(),
                "t".to_string(),
                SymbolicIvpProblemOptions::new()
                    .with_equation_parameters(parameters.clone())
                    .with_equation_parameter_values(values.clone())
                    .with_symbolic_assembly_backend(IvpSymbolicAssemblyBackend::AtomView)
                    .with_lambdify_execution_policy(policy)
                    .with_telemetry(IvpTelemetry::counters()),
            )
            .expect("policy-specific IVP backend should prepare")
        };

        let sequential = prepare(IvpLambdifyExecutionPolicy::Sequential);
        let parallel = prepare(IvpLambdifyExecutionPolicy::Parallel { min_work: 0 });
        let automatic = prepare(IvpLambdifyExecutionPolicy::Auto { min_work: 0 });

        let residual = (sequential.residual)(time, &state);
        let jacobian = (sequential.jacobian)(time, &state);
        for candidate in [&parallel, &automatic] {
            assert_eq!(residual, (candidate.residual)(time, &state));
            assert_eq!(jacobian, (candidate.jacobian)(time, &state));
        }

        let sequential_snapshot = sequential.telemetry.snapshot();
        assert_eq!(
            sequential_snapshot.lambdify_execution_policy,
            IvpLambdifyExecutionPolicy::Sequential
        );
        assert_eq!(sequential_snapshot.sequential_dispatches, 2);
        assert_eq!(sequential_snapshot.parallel_dispatches, 0);

        let parallel_snapshot = parallel.telemetry.snapshot();
        assert_eq!(
            parallel_snapshot.lambdify_execution_policy,
            IvpLambdifyExecutionPolicy::Parallel { min_work: 0 }
        );
        assert_eq!(parallel_snapshot.parallel_dispatches, 2);
        assert_eq!(parallel_snapshot.sequential_dispatches, 0);

        let automatic_snapshot = automatic.telemetry.snapshot();
        assert_eq!(
            automatic_snapshot.lambdify_execution_policy,
            IvpLambdifyExecutionPolicy::Auto { min_work: 0 }
        );
        assert_eq!(
            automatic_snapshot.parallel_dispatches + automatic_snapshot.sequential_dispatches,
            2
        );
    }

    #[test]
    fn prepared_ivp_aot_problem_preserves_time_param_variable_input_order() {
        let problem = prepare_symbolic_ivp_problem(
            vec![
                Expr::parse_expression("a*t + y + b*z"),
                Expr::parse_expression("c*y - z + b*t"),
            ],
            vec!["y".to_string(), "z".to_string()],
            "t".to_string(),
            SymbolicIvpProblemOptions::new()
                .with_equation_parameters(vec!["a".to_string(), "b".to_string(), "c".to_string()])
                .with_equation_parameter_values(DVector::from_vec(vec![2.0, -0.5, 3.0]))
                .with_prefer_aot_then_lambdify(),
        )
        .expect("IVP backend should prepare");

        let prepared = problem.prepare_dense_aot_problem(SymbolicIvpAotOptions::default());
        assert_eq!(
            prepared.flattened_input_names(),
            &["t", "a", "b", "c", "y", "z"]
        );
        assert!(!prepared.problem_key().is_empty());
    }
}
