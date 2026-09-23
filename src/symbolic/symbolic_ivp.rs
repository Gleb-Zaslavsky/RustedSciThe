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

use crate::symbolic::View::conversions::atom_to_expr;
use crate::symbolic::View::jacobian::PreparedSparseAtomSystem;
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
        }
    }
}

impl std::error::Error for IvpBackendError {}

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
    /// AtomView symbolic differentiation with an explicit Expr compatibility
    /// boundary before the existing Lambdify closures.
    AtomView,
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
    /// Symbolic equations used to prepare residuals.
    pub equations: Vec<Expr>,
    /// Symbolic dense Jacobian used by both lambdify and AOT preparation.
    pub symbolic_jacobian: Vec<Vec<Expr>>,
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
}

/// Prepared symbolic IVP residual without compiling any Jacobian callback.
///
/// This is useful for solver paths that provide a native sparse/banded
/// Jacobian evaluator separately and should not pay for a dense Jacobian
/// closure during setup.
pub struct PreparedSymbolicIvpResidualProblem {
    pub residual: Box<IvpResidualEval>,
    try_residual: Arc<IvpTryResidualEval>,
    /// Symbolic equations used to prepare residuals.
    pub equations: Vec<Expr>,
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

    pub fn parameter_values_handle(&self) -> Option<SharedIvpParameterValues> {
        self.parameter_values_handle.clone()
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
        let residual_eval = linked.residual_eval.clone();
        let residual_len = linked.residual_len;
        let parameter_values_handle = self.parameter_values_handle.clone();
        let telemetry = self.telemetry.clone();
        let try_residual: Arc<IvpTryResidualEval> = Arc::new(move |t, y| {
            let args = build_linked_args(t, y, parameter_values_handle.as_ref())?;
            let mut out = vec![0.0; residual_len];
            residual_eval(args.as_slice(), out.as_mut_slice());
            Ok(DVector::from_vec(out))
        });
        let residual =
            compatibility_residual(try_residual.clone(), telemetry.clone(), residual_len);

        Self {
            residual,
            try_residual,
            equations: self.equations,
            time_arg: self.time_arg,
            variables: self.variables,
            equation_parameters: self.equation_parameters,
            parameter_values_handle: self.parameter_values_handle,
            backend_kind: IvpBackendKind::Aot,
            telemetry,
        }
    }
}

impl PreparedSymbolicIvpProblem {
    /// Evaluates the prepared residual through the typed fallible boundary.
    pub fn try_evaluate_residual(
        &self,
        t: f64,
        y: &DVector<f64>,
    ) -> Result<DVector<f64>, IvpBackendError> {
        (self.try_residual)(t, y)
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
        let residual_eval = linked.residual_eval.clone();
        let jacobian_eval = linked.jacobian_eval.clone();
        let residual_len = linked.residual_len;
        let (rows, cols) = linked.shape;
        let parameter_values_handle = self.parameter_values_handle.clone();
        let residual_parameter_values_handle = parameter_values_handle.clone();
        let jacobian_parameter_values_handle = parameter_values_handle.clone();
        let try_residual: Arc<IvpTryResidualEval> = Arc::new(move |t, y| {
            let args = build_linked_args(t, y, residual_parameter_values_handle.as_ref())?;
            let mut out = vec![0.0; residual_len];
            residual_eval(&args, &mut out);
            Ok(DVector::from_vec(out))
        });
        let try_jacobian: Arc<IvpTryDenseJacobianEval> = Arc::new(move |t, y| {
            let args = build_linked_args(t, y, jacobian_parameter_values_handle.as_ref())?;
            let mut out = vec![0.0; rows * cols];
            jacobian_eval(&args, &mut out);
            Ok(DMatrix::from_row_slice(rows, cols, out.as_slice()))
        });
        let telemetry = self.telemetry.clone();
        let residual =
            compatibility_residual(try_residual.clone(), telemetry.clone(), residual_len);
        let jacobian = compatibility_jacobian(try_jacobian.clone(), telemetry.clone(), rows, cols);

        Self {
            residual,
            jacobian,
            try_residual,
            try_jacobian,
            equations: self.equations,
            symbolic_jacobian: self.symbolic_jacobian,
            time_arg: self.time_arg,
            variables: self.variables,
            equation_parameters: self.equation_parameters,
            parameter_values_handle,
            backend_kind: IvpBackendKind::Aot,
            telemetry,
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
        IvpSymbolicAssemblyBackend::AtomView => {
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

fn read_parameter_values(
    parameter_values_handle: Option<&SharedIvpParameterValues>,
) -> Result<Option<DVector<f64>>, IvpBackendError> {
    parameter_values_handle
        .map(|handle| {
            handle
                .read()
                .map(|values| values.clone())
                .map_err(|_| IvpBackendError::ParameterStatePoisoned)
        })
        .transpose()
}

fn build_linked_args(
    t: f64,
    y: &DVector<f64>,
    parameter_values_handle: Option<&SharedIvpParameterValues>,
) -> Result<Vec<f64>, IvpBackendError> {
    let parameter_values = read_parameter_values(parameter_values_handle)?;
    let mut args = Vec::with_capacity(
        1 + y.len() + parameter_values.as_ref().map_or(0, |values| values.len()),
    );
    args.push(t);
    if let Some(values) = parameter_values.as_ref() {
        args.extend(values.iter().copied());
    }
    args.extend(y.iter().copied());
    Ok(args)
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
            let parameter_values = match read_parameter_values(parameter_values_handle.as_ref()) {
                Ok(values) => values,
                Err(error) => {
                    telemetry.record_error();
                    return Err(error);
                }
            };
            let mut args = Vec::with_capacity(
                1 + y.len() + parameter_values.as_ref().map_or(0, |values| values.len()),
            );
            args.push(t);
            if let Some(values) = parameter_values.as_ref() {
                args.extend(values.iter().copied());
            }
            args.extend(y.iter().copied());
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
            let parameter_values = match read_parameter_values(parameter_values_handle.as_ref()) {
                Ok(values) => values,
                Err(error) => {
                    telemetry.record_error();
                    return Err(error);
                }
            };
            let mut args = Vec::with_capacity(
                1 + y.len() + parameter_values.as_ref().map_or(0, |values| values.len()),
            );
            args.push(t);
            if let Some(values) = parameter_values.as_ref() {
                args.extend(values.iter().copied());
            }
            args.extend(y.iter().copied());
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
        IvpSymbolicAssemblyBackend::AtomView => IvpTelemetryRoute::AtomViewExprCompat,
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

    let symbolic_started = telemetry.start_cold_stage(IvpColdStage::SymbolicJacobian);
    let symbolic_jacobian = build_symbolic_jacobian(
        &equations,
        &variables,
        options.symbolic_assembly_backend,
        &telemetry,
    );
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
        equations,
        symbolic_jacobian,
        time_arg,
        variables,
        equation_parameters: options.equation_parameters,
        parameter_values_handle,
        backend_kind,
        telemetry,
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
        IvpSymbolicAssemblyBackend::AtomView => IvpTelemetryRoute::AtomViewExprCompat,
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
    let residual = compatibility_residual(try_residual.clone(), telemetry.clone(), equations.len());

    Ok(PreparedSymbolicIvpResidualProblem {
        residual,
        try_residual,
        equations,
        time_arg,
        variables,
        equation_parameters: options.equation_parameters,
        parameter_values_handle,
        backend_kind: IvpBackendKind::Lambdify,
        telemetry,
    })
}

fn telemetry_route(backend: IvpSymbolicAssemblyBackend) -> IvpTelemetryRoute {
    match backend {
        IvpSymbolicAssemblyBackend::ExprLegacy => IvpTelemetryRoute::ExprLegacy,
        IvpSymbolicAssemblyBackend::AtomView => IvpTelemetryRoute::AtomViewExprCompat,
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
