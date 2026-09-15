//! Symbolic frontend for nonlinear systems.
//!
//! This module owns the symbolic side of `Nonlinear_systems` and turns raw
//! equations into one prepared residual/Jacobian provider for the solver
//! engine.
//!
//! The current architecture is intentionally layered:
//! - user code builds one [`SymbolicNonlinearProblem`] from equations plus
//!   [`SymbolicProblemOptions`],
//! - this module prepares one backend that implements the local
//!   residual/Jacobian contract,
//! - solver code later sees only [`NonlinearProblem`] and
//!   [`JacobianProvider`] and does not need to know whether evaluation comes
//!   from `lambdify` or from a linked compiled AOT module.
//!
//! Backend scenarios:
//! - `Lambdify` stays on the existing symbolic Jacobian + lambdify path,
//! - `Aot` is reached through
//!   [`crate::numerical::Nonlinear_systems::symbolic_backend`], which selects,
//!   resolves, and links a compiled dense backend before this module adapts it
//!   to the same solver-facing contract.
//!
//! This makes `Nonlinear_systems` follow the same direction as the newer BVP
//! stack: contracts and prepared bridges are explicit, while generic AOT
//! lifecycle details stay outside the solver engine itself.

use crate::numerical::Nonlinear_systems::error::SolveError;
use crate::numerical::Nonlinear_systems::problem::{JacobianProvider, NonlinearProblem};
use crate::numerical::Nonlinear_systems::symbolic_backend::{
    SelectedSymbolicNonlinearBackendKind, SymbolicBackendSelectionPolicy,
    select_symbolic_nonlinear_backend,
};
use crate::numerical::Nonlinear_systems::symbolic_legacy::LegacyLambdifySymbolicBackend;
use crate::symbolic::codegen::codegen_aot_resolution::AotResolver;
use crate::symbolic::codegen::codegen_aot_runtime_link::{
    LinkedDenseAotBackend, resolve_linked_dense_backend,
};
use crate::symbolic::codegen::codegen_manifest::PreparedProblemManifest;
use crate::symbolic::codegen::codegen_provider_api::{
    BackendKind, MatrixBackend, PreparedDenseProblem,
};
use crate::symbolic::codegen::codegen_runtime_api::{
    DenseJacobianChunkingStrategy, ResidualChunkingStrategy,
};
use crate::symbolic::codegen::codegen_tasks::{JacobianTask, ResidualTask};
use crate::symbolic::symbolic_engine::Expr;
use nalgebra::{DMatrix, DVector};
use std::cell::RefCell;
use std::time::{Duration, Instant};

/// Backend used to turn symbolic equations into callable residual/Jacobian evaluators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SymbolicBackendKind {
    /// Compile symbolic expressions into in-process lambdified closures.
    #[default]
    Lambdify,
    /// Reserve a slot for the future AOT-generated backend path.
    Aot,
}

impl SymbolicBackendKind {
    /// Returns a stable short name used in logs and diagnostics.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lambdify => "lambdify",
            Self::Aot => "aot",
        }
    }
}

/// Lifecycle policy requested for an optional generated artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SymbolicArtifactPolicy {
    /// No generated artifact was requested by the direct Lambdify API.
    #[default]
    NotRequested,
    /// Use an artifact when one is already available, otherwise use Lambdify.
    UseIfAvailable,
    /// Fail unless a compatible prebuilt artifact is available.
    RequirePrebuilt,
    /// Build an artifact when the compatible artifact is missing.
    BuildIfMissing,
    /// Always execute a fresh artifact build.
    RebuildAlways,
}

impl SymbolicArtifactPolicy {
    /// Returns the stable label used by diagnostics and story tables.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotRequested => "not-requested",
            Self::UseIfAvailable => "use-if-available",
            Self::RequirePrebuilt => "require-prebuilt",
            Self::BuildIfMissing => "build-if-missing",
            Self::RebuildAlways => "rebuild-always",
        }
    }
}

fn artifact_policy_for_backend_selection(
    policy: SymbolicBackendSelectionPolicy,
) -> SymbolicArtifactPolicy {
    match policy {
        SymbolicBackendSelectionPolicy::LambdifyOnly => SymbolicArtifactPolicy::NotRequested,
        SymbolicBackendSelectionPolicy::AotOnly => SymbolicArtifactPolicy::RequirePrebuilt,
        SymbolicBackendSelectionPolicy::PreferAotThenLambdify => {
            SymbolicArtifactPolicy::UseIfAvailable
        }
    }
}

/// Action taken with the generated artifact during preparation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SymbolicArtifactAction {
    /// The direct Lambdify API did not request an artifact.
    #[default]
    NotApplicable,
    /// A compatible artifact was reused without a new build.
    Reused,
    /// A new artifact was built and registered.
    Built,
    /// A requested artifact was unavailable and the policy fell back to Lambdify.
    FallbackToLambdify,
}

/// Preparation/build telemetry shared by direct Lambdify and generated AOT
/// entry points.
///
/// Build-related fields are optional by design: `None` means that the stage
/// was not part of this lifecycle call. This keeps an unavailable metric
/// distinct from a measured zero-duration stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolicPreparationReport {
    /// Effective callable backend after preparation and selection.
    pub effective_backend: SymbolicBackendKind,
    /// Artifact policy requested by the caller.
    pub artifact_policy: SymbolicArtifactPolicy,
    /// Action taken for the generated artifact, if applicable.
    pub artifact_action: SymbolicArtifactAction,
    /// Manifest-derived artifact identity, when an AOT plan was prepared.
    pub artifact_key: Option<String>,
    /// End-to-end preparation time, excluding later numerical solves.
    pub preparation_duration: Duration,
    /// Time spent in the AOT materialize/build/register stage performed by
    /// this call. `None` also covers a request satisfied by another process.
    pub build_duration: Option<Duration>,
    /// Number of generated residual jobs/chunks, when an AOT plan exists.
    pub generated_residual_jobs: Option<usize>,
    /// Number of generated Jacobian jobs/chunks, when an AOT plan exists.
    pub generated_jacobian_jobs: Option<usize>,
}

impl SymbolicPreparationReport {
    fn direct_lambdify(duration: Duration, backend: SymbolicBackendKind) -> Self {
        Self {
            effective_backend: backend,
            artifact_policy: SymbolicArtifactPolicy::NotRequested,
            artifact_action: SymbolicArtifactAction::NotApplicable,
            artifact_key: None,
            preparation_duration: duration,
            build_duration: None,
            generated_residual_jobs: None,
            generated_jacobian_jobs: None,
        }
    }
}

/// High-level symbolic backend selection for nonlinear systems.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SymbolicBackendConfig {
    /// Backend used to prepare the symbolic residual/Jacobian provider.
    pub kind: SymbolicBackendKind,
}

/// Runtime policy for evaluating prepared Lambdify callbacks.
///
/// `Sequential` is the conservative default for small systems. `Parallel`
/// uses the already-known evaluator layout and activates only when the
/// selected Jacobian/residual work is at least `min_work`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LambdifyExecutionPolicy {
    /// Evaluate callbacks on the calling thread.
    Sequential,
    /// Evaluate independent callback outputs with Rayon when work is large enough.
    Parallel {
        /// Minimum number of evaluator slots required to use parallel execution.
        min_work: usize,
    },
}

impl Default for LambdifyExecutionPolicy {
    fn default() -> Self {
        Self::Sequential
    }
}

impl SymbolicBackendConfig {
    /// Uses the existing lambdify-based symbolic backend.
    pub fn lambdify() -> Self {
        Self {
            kind: SymbolicBackendKind::Lambdify,
        }
    }

    /// Requests the future AOT symbolic backend.
    pub fn aot() -> Self {
        Self {
            kind: SymbolicBackendKind::Aot,
        }
    }
}

/// Stable, ordered names of the numeric parameters of a nonlinear system.
///
/// The order is part of the evaluation contract: parameter values are passed
/// to Lambdify/AOT backends in this order, before the unknown vector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonlinearParameterSchema {
    names: Vec<String>,
}

impl NonlinearParameterSchema {
    /// Creates a validated parameter schema and preserves the supplied order.
    pub fn new(names: Vec<String>) -> Result<Self, SolveError> {
        for (index, name) in names.iter().enumerate() {
            if name.trim().is_empty() {
                return Err(SolveError::InvalidParameterSchema(format!(
                    "parameter name at index {index} is empty"
                )));
            }
            if name.trim() != name {
                return Err(SolveError::InvalidParameterSchema(format!(
                    "parameter name '{name}' contains surrounding whitespace"
                )));
            }
            if names[..index].iter().any(|previous| previous == name) {
                return Err(SolveError::InvalidParameterSchema(format!(
                    "duplicate parameter name '{name}'"
                )));
            }
        }
        Ok(Self { names })
    }

    /// Returns the number of declared parameters.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Returns true when the schema has no names.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Returns parameter names in their stable evaluation order.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Returns the stable index of a named parameter.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|candidate| candidate == name)
    }

    /// Consumes the schema and returns its ordered names.
    pub fn into_names(self) -> Vec<String> {
        self.names
    }
}

/// Validated numeric values bound to a NonlinearParameterSchema.
#[derive(Debug, Clone, PartialEq)]
pub struct NonlinearParameterValues {
    schema: NonlinearParameterSchema,
    values: DVector<f64>,
}

impl NonlinearParameterValues {
    /// Validates values against the schema without modifying an existing bind.
    pub fn new(
        schema: &NonlinearParameterSchema,
        values: DVector<f64>,
    ) -> Result<Self, SolveError> {
        if schema.len() != values.len() {
            return Err(SolveError::DimensionMismatch {
                expected: schema.len(),
                actual: values.len(),
                context: "nonlinear parameter values",
            });
        }
        for (index, value) in values.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(SolveError::NonFiniteParameterValue { index, value });
            }
        }
        Ok(Self {
            schema: schema.clone(),
            values,
        })
    }

    /// Returns the ordered schema this binding was validated against.
    pub fn schema(&self) -> &NonlinearParameterSchema {
        &self.schema
    }

    /// Returns the validated values in schema order.
    pub fn as_vector(&self) -> &DVector<f64> {
        &self.values
    }

    /// Returns one value by its stable parameter index.
    pub fn get(&self, index: usize) -> Option<f64> {
        self.values.get(index).copied()
    }

    /// Consumes the binding and returns its numeric vector.
    pub fn into_vector(self) -> DVector<f64> {
        self.values
    }
}

fn validate_variable_names(names: &[String]) -> Result<(), SolveError> {
    for (index, name) in names.iter().enumerate() {
        if name.trim().is_empty() {
            return Err(SolveError::InvalidVariableSchema(format!(
                "variable name at index {index} is empty"
            )));
        }
        if name.trim() != name {
            return Err(SolveError::InvalidVariableSchema(format!(
                "variable name '{name}' contains surrounding whitespace"
            )));
        }
        if names[..index].iter().any(|previous| previous == name) {
            return Err(SolveError::InvalidVariableSchema(format!(
                "duplicate variable name '{name}'"
            )));
        }
    }
    Ok(())
}

/// User-facing setup object for symbolic nonlinear problems.
///
/// This keeps symbolic problem construction on one layer:
/// equations are provided separately, while variables, optional parameters,
/// parameter values, and backend choice are grouped into one options object.
#[derive(Debug, Clone, Default)]
pub struct SymbolicProblemOptions {
    /// Explicit variable ordering.
    pub variables: Option<Vec<String>>,
    /// Optional symbolic parameter names.
    pub equation_parameters: Option<Vec<String>>,
    /// Optional parameter values used during evaluation.
    pub equation_parameter_values: Option<DVector<f64>>,
    /// Backend used to prepare the symbolic provider.
    pub backend_config: SymbolicBackendConfig,
    /// Runtime execution policy for the prepared Lambdify backend.
    pub lambdify_execution_policy: LambdifyExecutionPolicy,
}

impl SymbolicProblemOptions {
    /// Creates a default symbolic problem setup that uses the lambdify backend.
    pub fn new() -> Self {
        Self::default()
    }

    /// Installs an explicit variable ordering.
    pub fn with_variables(mut self, variables: Vec<String>) -> Self {
        self.variables = Some(variables);
        self
    }

    /// Installs symbolic parameter names.
    pub fn with_equation_parameters(mut self, parameters: Vec<String>) -> Self {
        self.equation_parameters = Some(parameters);
        self
    }

    /// Installs parameter values used when evaluating the symbolic system.
    pub fn with_equation_parameter_values(mut self, values: DVector<f64>) -> Self {
        self.equation_parameter_values = Some(values);
        self
    }

    /// Overrides the symbolic backend preparation mode.
    pub fn with_backend_config(mut self, backend_config: SymbolicBackendConfig) -> Self {
        self.backend_config = backend_config;
        self
    }

    /// Selects sequential or thresholded parallel Lambdify evaluation.
    pub fn with_lambdify_execution_policy(mut self, policy: LambdifyExecutionPolicy) -> Self {
        self.lambdify_execution_policy = policy;
        self
    }

    /// Convenience preset for the existing lambdify backend.
    pub fn with_lambdify_backend(self) -> Self {
        self.with_backend_config(SymbolicBackendConfig::lambdify())
    }

    /// Convenience preset for the future AOT backend.
    pub fn with_aot_backend(self) -> Self {
        self.with_backend_config(SymbolicBackendConfig::aot())
    }
}

/// AOT preparation settings for dense nonlinear symbolic problems.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SymbolicDenseAotOptions {
    /// Residual chunking used by the dense nonlinear residual plan.
    pub residual_strategy: ResidualChunkingStrategy,
    /// Jacobian row chunking used by the dense nonlinear Jacobian plan.
    pub jacobian_strategy: DenseJacobianChunkingStrategy,
}

impl Default for SymbolicDenseAotOptions {
    fn default() -> Self {
        Self {
            residual_strategy: ResidualChunkingStrategy::Whole,
            jacobian_strategy: DenseJacobianChunkingStrategy::Whole,
        }
    }
}

/// Dense nonlinear prepared AOT bridge built from symbolic equations.
///
/// This is the dense nonlinear-system analogue of the BVP prepared AOT bridge:
/// it carries runtime plans and exposes a manifest-friendly prepared dense
/// problem that can later be handed to the generic AOT lifecycle.
#[derive(Debug, Clone)]
pub struct PreparedSymbolicNonlinearAotProblem<'a> {
    equations: &'a [Expr],
    symbolic_jacobian: &'a [Vec<Expr>],
    variable_refs: Vec<&'a str>,
    parameter_refs: Option<Vec<&'a str>>,
    flattened_input_names: Vec<&'a str>,
    residual_fn_name: String,
    jacobian_fn_name: String,
    residual_strategy: ResidualChunkingStrategy,
    jacobian_strategy: DenseJacobianChunkingStrategy,
}

impl<'a> PreparedSymbolicNonlinearAotProblem<'a> {
    /// Builds the residual runtime plan consumed by the generic AOT pipeline.
    ///
    /// The plan borrows the symbolic equations and the already-fixed flattened
    /// input order stored by this bridge.
    fn residual_runtime_plan(
        &self,
    ) -> crate::symbolic::codegen::codegen_runtime_api::ResidualRuntimePlan<'_> {
        ResidualTask {
            fn_name: self.residual_fn_name.as_str(),
            residuals: self.equations,
            variables: &self.variable_refs,
            params: self.parameter_refs.as_deref(),
        }
        .runtime_plan(self.residual_strategy)
    }

    fn jacobian_runtime_plan(
        &self,
    ) -> crate::symbolic::codegen::codegen_runtime_api::DenseJacobianRuntimePlan<'_> {
        // The dense runtime plan keeps exactly the same flattened input order
        // contract as the residual plan, so codegen/build layers can treat both
        // outputs as one prepared problem.
        JacobianTask {
            fn_name: self.jacobian_fn_name.as_str(),
            jacobian: self.symbolic_jacobian,
            variables: &self.variable_refs,
            params: self.parameter_refs.as_deref(),
        }
        .runtime_plan(self.jacobian_strategy)
    }

    /// Returns the underlying generic prepared dense problem.
    pub fn as_prepared_problem(&self) -> PreparedDenseProblem<'_> {
        PreparedDenseProblem::new(
            BackendKind::Aot,
            MatrixBackend::Dense,
            self.residual_runtime_plan(),
            self.jacobian_runtime_plan(),
        )
    }

    /// Returns the flattened input names shared by residual and Jacobian plans.
    pub fn flattened_input_names(&self) -> &[&'a str] {
        &self.flattened_input_names
    }

    /// Returns the residual length.
    pub fn residual_len(&self) -> usize {
        self.equations.len()
    }

    /// Returns the dense Jacobian shape `(rows, cols)`.
    pub fn jacobian_shape(&self) -> (usize, usize) {
        (
            self.symbolic_jacobian.len(),
            self.symbolic_jacobian.first().map_or(0, |row| row.len()),
        )
    }

    /// Returns an owned manifest for the prepared dense AOT problem.
    pub fn manifest(&self) -> PreparedProblemManifest {
        PreparedProblemManifest::from(&self.as_prepared_problem())
    }

    /// Returns the stable manifest-derived problem key used by registry layers.
    pub fn problem_key(&self) -> String {
        self.manifest().problem_key()
    }
}

/// Solver-facing contract implemented by symbolic nonlinear backends.
///
/// The nonlinear engine should not care whether residuals and Jacobians come from
/// the legacy lambdify path or, later, from generated AOT code. This trait keeps
/// that contract explicit.
pub(crate) trait SymbolicEvaluationBackend: Send + Sync {
    /// Returns the backend kind that produced this provider.
    fn kind(&self) -> SymbolicBackendKind;

    /// Returns symbolic Jacobian metadata when the backend owns it.
    ///
    /// Generated backends do not need to retain symbolic expressions, so the
    /// default is `None`.
    fn symbolic_jacobian(&self) -> Option<&[Vec<Expr>]> {
        None
    }

    /// Returns the configured Lambdify execution policy, if applicable.
    fn lambdify_execution_policy(&self) -> Option<LambdifyExecutionPolicy> {
        None
    }

    /// Returns whether residual output can be written without a temporary.
    fn supports_residual_into(&self) -> bool {
        false
    }

    /// Returns whether Jacobian output can be written into caller-owned storage.
    fn supports_jacobian_into(&self) -> bool {
        false
    }

    /// Evaluates the residual vector at `x`.
    fn residual(
        &self,
        x: &DVector<f64>,
        equation_parameters: Option<&[String]>,
        equation_parameter_values: Option<&DVector<f64>>,
        variables: &[String],
    ) -> Result<DVector<f64>, SolveError>;

    /// Evaluates a residual into caller-owned storage.
    ///
    /// The default preserves the owned-returning backend contract. A backend
    /// with a native `*_into` evaluator may override this to avoid a temporary
    /// vector allocation.
    fn residual_into(
        &self,
        x: &DVector<f64>,
        equation_parameters: Option<&[String]>,
        equation_parameter_values: Option<&DVector<f64>>,
        variables: &[String],
        out: &mut DVector<f64>,
    ) -> Result<(), SolveError> {
        let residual =
            self.residual(x, equation_parameters, equation_parameter_values, variables)?;
        if residual.len() != out.len() {
            return Err(SolveError::DimensionMismatch {
                expected: variables.len(),
                actual: residual.len(),
                context: "symbolic residual evaluation",
            });
        }
        out.copy_from(&residual);
        Ok(())
    }

    /// Evaluates a Jacobian into caller-owned storage.
    fn jacobian_into(
        &self,
        x: &DVector<f64>,
        equation_parameters: Option<&[String]>,
        equation_parameter_values: Option<&DVector<f64>>,
        variables: &[String],
        out: &mut DMatrix<f64>,
    ) -> Result<(), SolveError> {
        let jacobian =
            self.jacobian(x, equation_parameters, equation_parameter_values, variables)?;
        if jacobian.nrows() != out.nrows() || jacobian.ncols() != out.ncols() {
            return Err(SolveError::InvalidConfig(format!(
                "Jacobian evaluation returned {}x{}, expected {}x{}",
                jacobian.nrows(),
                jacobian.ncols(),
                out.nrows(),
                out.ncols()
            )));
        }
        out.copy_from(&jacobian);
        Ok(())
    }

    /// Evaluates the Jacobian matrix at `x`.
    fn jacobian(
        &self,
        x: &DVector<f64>,
        equation_parameters: Option<&[String]>,
        equation_parameter_values: Option<&DVector<f64>>,
        variables: &[String],
    ) -> Result<DMatrix<f64>, SolveError>;
}

/// Concrete backend implementation that uses a linked compiled dense AOT backend.
struct CompiledDenseAotSymbolicBackend {
    /// Process-local linked compiled backend entry.
    linked: LinkedDenseAotBackend,
    /// Ordered symbolic variable names used by the nonlinear problem.
    variables: Vec<String>,
    /// Optional symbolic parameter names used to assemble flattened AOT inputs.
    equation_parameters: Option<Vec<String>>,
}

thread_local! {
    /// Reusable flattened input for one compiled callback on the current thread.
    ///
    /// Generated callbacks intentionally keep a single contiguous ABI. Reusing
    /// this buffer avoids rebuilding `[parameters..., variables...]` on every
    /// AOT residual/Jacobian call without introducing a mutex into the hot path.
    static COMPILED_AOT_ARGS: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    /// Row-major Jacobian scratch used when adapting generated output to nalgebra.
    static COMPILED_AOT_JACOBIAN: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

/// Copies a generated row-major Jacobian into nalgebra's column-major storage.
///
/// The generated ABI is intentionally row-major because it is shared by the
/// C, Zig, Rust, BVP, and IVP integrations. Keeping the conversion here avoids
/// changing that public ABI. Traversing column segments inside small tiles
/// makes writes contiguous in the destination while keeping the source
/// working set cache-friendly.
pub(crate) fn copy_row_major_jacobian_into_column_major(
    row_major: &[f64],
    column_major: &mut [f64],
    rows: usize,
    cols: usize,
) {
    const TILE: usize = 32;
    debug_assert_eq!(row_major.len(), rows * cols);
    debug_assert_eq!(column_major.len(), rows * cols);

    for col_start in (0..cols).step_by(TILE) {
        let col_end = (col_start + TILE).min(cols);
        for row_start in (0..rows).step_by(TILE) {
            let row_end = (row_start + TILE).min(rows);
            for col in col_start..col_end {
                let destination_start = col * rows;
                for row in row_start..row_end {
                    column_major[destination_start + row] = row_major[row * cols + col];
                }
            }
        }
    }
}

impl CompiledDenseAotSymbolicBackend {
    /// Builds a compiled dense backend from a linked runtime entry.
    fn new(
        linked: LinkedDenseAotBackend,
        variables: &[String],
        equation_parameters: Option<&[String]>,
    ) -> Self {
        Self {
            linked,
            variables: variables.to_vec(),
            equation_parameters: equation_parameters.map(|params| params.to_vec()),
        }
    }

    /// Validates the input metadata shared by residual and Jacobian callbacks.
    fn validate_input(
        &self,
        x: &DVector<f64>,
        equation_parameters: Option<&[String]>,
        equation_parameter_values: Option<&DVector<f64>>,
        variables: &[String],
    ) -> Result<(), SolveError> {
        if x.len() != variables.len() {
            return Err(SolveError::DimensionMismatch {
                expected: variables.len(),
                actual: x.len(),
                context: "compiled symbolic residual input",
            });
        }

        if variables != self.variables.as_slice() {
            return Err(SolveError::InvalidConfig(
                "compiled nonlinear AOT backend was prepared for a different variable ordering"
                    .to_string(),
            ));
        }

        if equation_parameters != self.equation_parameters.as_deref() {
            return Err(SolveError::InvalidConfig(
                "compiled nonlinear AOT backend was prepared for a different parameter schema"
                    .to_string(),
            ));
        }

        if let Some(parameters) = &self.equation_parameters {
            let values = equation_parameter_values.ok_or_else(|| {
                SolveError::InvalidConfig(
                    "compiled nonlinear AOT backend requires parameter values".to_string(),
                )
            })?;
            if values.len() != parameters.len() {
                return Err(SolveError::DimensionMismatch {
                    expected: parameters.len(),
                    actual: values.len(),
                    context: "compiled symbolic parameter values",
                });
            }
        }
        Ok(())
    }

    /// Runs one generated callback with a reusable flattened input buffer.
    fn with_flattened_args<T>(
        &self,
        x: &DVector<f64>,
        equation_parameters: Option<&[String]>,
        equation_parameter_values: Option<&DVector<f64>>,
        variables: &[String],
        callback: impl FnOnce(&[f64]) -> Result<T, SolveError>,
    ) -> Result<T, SolveError> {
        self.validate_input(x, equation_parameters, equation_parameter_values, variables)?;
        COMPILED_AOT_ARGS.with(|scratch| {
            let mut args = scratch.borrow_mut();
            args.clear();
            if let Some(values) = equation_parameter_values {
                args.extend(values.iter().copied());
            }
            args.extend(x.iter().copied());
            callback(args.as_slice())
        })
    }
}

impl SymbolicEvaluationBackend for CompiledDenseAotSymbolicBackend {
    fn kind(&self) -> SymbolicBackendKind {
        SymbolicBackendKind::Aot
    }

    fn supports_residual_into(&self) -> bool {
        true
    }

    fn residual(
        &self,
        x: &DVector<f64>,
        equation_parameters: Option<&[String]>,
        equation_parameter_values: Option<&DVector<f64>>,
        variables: &[String],
    ) -> Result<DVector<f64>, SolveError> {
        self.with_flattened_args(
            x,
            equation_parameters,
            equation_parameter_values,
            variables,
            |args| {
                let mut out = vec![0.0; self.linked.residual_len];
                (self.linked.residual_eval)(args, &mut out);
                if out.iter().any(|value| !value.is_finite()) {
                    return Err(SolveError::ResidualEvaluation(
                        "compiled nonlinear AOT residual returned NaN or Inf".to_string(),
                    ));
                }
                Ok(DVector::from_vec(out))
            },
        )
    }

    fn residual_into(
        &self,
        x: &DVector<f64>,
        equation_parameters: Option<&[String]>,
        equation_parameter_values: Option<&DVector<f64>>,
        variables: &[String],
        out: &mut DVector<f64>,
    ) -> Result<(), SolveError> {
        self.with_flattened_args(
            x,
            equation_parameters,
            equation_parameter_values,
            variables,
            |args| {
                if out.len() != self.linked.residual_len {
                    return Err(SolveError::DimensionMismatch {
                        expected: self.linked.residual_len,
                        actual: out.len(),
                        context: "compiled symbolic residual output",
                    });
                }
                (self.linked.residual_eval)(args, out.as_mut_slice());
                if out.iter().any(|value| !value.is_finite()) {
                    return Err(SolveError::ResidualEvaluation(
                        "compiled nonlinear AOT residual returned NaN or Inf".to_string(),
                    ));
                }
                Ok(())
            },
        )
    }

    fn supports_jacobian_into(&self) -> bool {
        true
    }

    fn jacobian(
        &self,
        x: &DVector<f64>,
        equation_parameters: Option<&[String]>,
        equation_parameter_values: Option<&DVector<f64>>,
        variables: &[String],
    ) -> Result<DMatrix<f64>, SolveError> {
        self.with_flattened_args(
            x,
            equation_parameters,
            equation_parameter_values,
            variables,
            |args| {
                let mut out = vec![0.0; self.linked.shape.0 * self.linked.shape.1];
                (self.linked.jacobian_eval)(args, &mut out);
                if out.iter().any(|value| !value.is_finite()) {
                    return Err(SolveError::JacobianEvaluation(
                        "compiled nonlinear AOT jacobian returned NaN or Inf".to_string(),
                    ));
                }
                Ok(DMatrix::from_row_slice(
                    self.linked.shape.0,
                    self.linked.shape.1,
                    &out,
                ))
            },
        )
    }

    fn jacobian_into(
        &self,
        x: &DVector<f64>,
        equation_parameters: Option<&[String]>,
        equation_parameter_values: Option<&DVector<f64>>,
        variables: &[String],
        out: &mut DMatrix<f64>,
    ) -> Result<(), SolveError> {
        let (rows, cols) = self.linked.shape;
        if out.nrows() != rows || out.ncols() != cols {
            return Err(SolveError::InvalidConfig(format!(
                "Jacobian evaluation returned {}x{}, expected {}x{}",
                rows,
                cols,
                out.nrows(),
                out.ncols()
            )));
        }
        self.with_flattened_args(
            x,
            equation_parameters,
            equation_parameter_values,
            variables,
            |args| {
                COMPILED_AOT_JACOBIAN.with(|scratch| {
                    let mut row_major = scratch.borrow_mut();
                    row_major.resize(rows * cols, 0.0);
                    row_major.fill(0.0);
                    (self.linked.jacobian_eval)(args, row_major.as_mut_slice());
                    if row_major.iter().any(|value| !value.is_finite()) {
                        return Err(SolveError::JacobianEvaluation(
                            "compiled nonlinear AOT jacobian returned NaN or Inf".to_string(),
                        ));
                    }
                    copy_row_major_jacobian_into_column_major(
                        &row_major,
                        out.as_mut_slice(),
                        rows,
                        cols,
                    );
                    Ok(())
                })
            },
        )
    }
}

/// Prepared callable backend for a symbolic nonlinear system.
///
/// This mirrors the newer BVP architecture: symbolic equations are first normalized
/// into a backend configuration, then materialized into a prepared provider that the
/// generic solver engine can evaluate without knowing whether the backend came from
/// `lambdify` or, later, from AOT code generation.
struct PreparedSymbolicBackend {
    /// Solver-facing backend contract.
    backend: Box<dyn SymbolicEvaluationBackend>,
}

impl PreparedSymbolicBackend {
    /// Builds a prepared backend from symbolic equations and backend config.
    fn from_expressions(
        equations: &[Expr],
        variables: &[String],
        equation_parameters: Option<&[String]>,
        execution_policy: LambdifyExecutionPolicy,
        config: &SymbolicBackendConfig,
    ) -> Result<Self, SolveError> {
        match config.kind {
            SymbolicBackendKind::Lambdify => Ok(Self {
                backend: Box::new(LegacyLambdifySymbolicBackend::from_expressions(
                    equations,
                    variables,
                    equation_parameters,
                    execution_policy,
                )?),
            }),
            SymbolicBackendKind::Aot => Err(SolveError::InvalidConfig(
                "symbolic nonlinear AOT backend is not wired yet; use SymbolicBackendKind::Lambdify for now".to_string(),
            )),
        }
    }

    /// Returns the prepared backend kind.
    fn kind(&self) -> SymbolicBackendKind {
        self.backend.kind()
    }

    /// Returns whether the selected backend supports direct residual output.
    fn supports_residual_into(&self) -> bool {
        self.backend.supports_residual_into()
    }

    /// Returns whether the prepared backend can fill Jacobian storage directly.
    fn supports_jacobian_into(&self) -> bool {
        self.backend.supports_jacobian_into()
    }

    /// Returns symbolic Jacobian metadata retained by the prepared backend.
    fn symbolic_jacobian(&self) -> Option<&[Vec<Expr>]> {
        self.backend.symbolic_jacobian()
    }

    /// Returns the configured Lambdify execution policy, if applicable.
    fn lambdify_execution_policy(&self) -> Option<LambdifyExecutionPolicy> {
        self.backend.lambdify_execution_policy()
    }

    /// Builds a prepared compiled dense AOT backend from a linked runtime entry.
    ///
    /// By the time this constructor is called, backend selection, artifact
    /// resolution, and process-local runtime linking have already been done by
    /// outer layers. This method only adapts the linked callbacks to the local
    /// solver-facing backend contract.
    fn from_linked_dense(
        linked: LinkedDenseAotBackend,
        variables: &[String],
        equation_parameters: Option<&[String]>,
    ) -> Self {
        Self {
            backend: Box::new(CompiledDenseAotSymbolicBackend::new(
                linked,
                variables,
                equation_parameters,
            )),
        }
    }

    /// Evaluates the residual vector using the prepared backend.
    fn residual(
        &self,
        x: &DVector<f64>,
        equation_parameters: Option<&[String]>,
        equation_parameter_values: Option<&DVector<f64>>,
        variables: &[String],
    ) -> Result<DVector<f64>, SolveError> {
        self.backend
            .residual(x, equation_parameters, equation_parameter_values, variables)
    }

    /// Evaluates the Jacobian matrix using the prepared backend.
    fn jacobian(
        &self,
        x: &DVector<f64>,
        equation_parameters: Option<&[String]>,
        equation_parameter_values: Option<&DVector<f64>>,
        variables: &[String],
    ) -> Result<DMatrix<f64>, SolveError> {
        self.backend
            .jacobian(x, equation_parameters, equation_parameter_values, variables)
    }

    /// Evaluates a residual into reusable caller-owned storage.
    fn residual_into(
        &self,
        x: &DVector<f64>,
        equation_parameters: Option<&[String]>,
        equation_parameter_values: Option<&DVector<f64>>,
        variables: &[String],
        out: &mut DVector<f64>,
    ) -> Result<(), SolveError> {
        self.backend.residual_into(
            x,
            equation_parameters,
            equation_parameter_values,
            variables,
            out,
        )
    }

    /// Evaluates a Jacobian into reusable caller-owned storage.
    fn jacobian_into(
        &self,
        x: &DVector<f64>,
        equation_parameters: Option<&[String]>,
        equation_parameter_values: Option<&DVector<f64>>,
        variables: &[String],
        out: &mut DMatrix<f64>,
    ) -> Result<(), SolveError> {
        self.backend.jacobian_into(
            x,
            equation_parameters,
            equation_parameter_values,
            variables,
            out,
        )
    }
}

/// Adapter that exposes symbolic equations through the generic problem traits.
pub struct SymbolicNonlinearProblem {
    /// Backend configuration used to prepare the symbolic provider.
    backend_config: SymbolicBackendConfig,
    /// Prepared symbolic residual/Jacobian backend.
    backend: PreparedSymbolicBackend,
    /// Original symbolic equations.
    equations: Vec<Expr>,
    /// Dense symbolic Jacobian kept for future AOT preparation and diagnostics.
    symbolic_jacobian: Vec<Vec<Expr>>,
    /// Ordered list of variables.
    variables: Vec<String>,
    /// Optional stable symbolic parameter schema.
    parameter_schema: Option<NonlinearParameterSchema>,
    /// Optional validated parameter values used during evaluation.
    parameter_values: Option<NonlinearParameterValues>,
    /// Immutable preparation telemetry retained for later story/report use.
    preparation_report: SymbolicPreparationReport,
}

/// Immutable prepared symbolic nonlinear problem.
///
/// Preparation owns the equations, symbolic Jacobian, variable order, and
/// callable backend. Runtime parameter values are deliberately not mutable
/// state here; use [`PreparedSymbolicNonlinearProblem::bind`] to create a
/// solver-facing bound view.
pub struct PreparedSymbolicNonlinearProblem {
    problem: SymbolicNonlinearProblem,
}

/// Parameter-bound view of a prepared symbolic nonlinear problem.
///
/// The view borrows the prepared backend and owns only validated numeric
/// parameter values, so changing parameters does not repeat symbolic
/// differentiation, lambdification, AOT selection, or artifact preparation.
pub struct BoundSymbolicNonlinearProblem<'a> {
    prepared: &'a PreparedSymbolicNonlinearProblem,
    parameter_values: Option<NonlinearParameterValues>,
}

impl PreparedSymbolicNonlinearProblem {
    /// Wraps an already prepared compatibility problem without rebuilding it.
    ///
    /// This is useful when a higher-level lifecycle API has already selected
    /// and prepared an AOT backend and the caller wants the immutable
    /// prepared/bound interface afterward.
    pub fn from_problem(problem: SymbolicNonlinearProblem) -> Self {
        Self { problem }
    }

    /// Returns the immutable preparation/build report for this provider.
    pub fn preparation_report(&self) -> &SymbolicPreparationReport {
        self.problem.preparation_report()
    }

    /// Prepares a symbolic problem once from parsed expressions.
    pub fn from_expressions(
        equations: Vec<Expr>,
        options: SymbolicProblemOptions,
    ) -> Result<Self, SolveError> {
        Ok(Self {
            problem: SymbolicNonlinearProblem::from_expressions_with_options(equations, options)?,
        })
    }

    /// Prepares a symbolic problem once from equation strings.
    pub fn from_strings(
        equations: Vec<String>,
        options: SymbolicProblemOptions,
    ) -> Result<Self, SolveError> {
        Ok(Self {
            problem: SymbolicNonlinearProblem::from_strings_with_options(equations, options)?,
        })
    }

    /// Prepares a symbolic problem through the explicit Lambdify/AOT selector.
    pub fn from_expressions_with_backend_selection(
        equations: Vec<Expr>,
        options: SymbolicProblemOptions,
        policy: SymbolicBackendSelectionPolicy,
        resolver: Option<&AotResolver>,
        aot_options: SymbolicDenseAotOptions,
    ) -> Result<Self, SolveError> {
        Ok(Self {
            problem: SymbolicNonlinearProblem::from_expressions_with_backend_selection(
                equations,
                options,
                policy,
                resolver,
                aot_options,
            )?,
        })
    }

    /// String-based backend-selection constructor for the prepared lifecycle.
    pub fn from_strings_with_backend_selection(
        equations: Vec<String>,
        options: SymbolicProblemOptions,
        policy: SymbolicBackendSelectionPolicy,
        resolver: Option<&AotResolver>,
        aot_options: SymbolicDenseAotOptions,
    ) -> Result<Self, SolveError> {
        Ok(Self {
            problem: SymbolicNonlinearProblem::from_strings_with_backend_selection(
                equations,
                options,
                policy,
                resolver,
                aot_options,
            )?,
        })
    }

    /// Returns the prepared solver-facing problem for compatibility adapters.
    pub fn as_problem(&self) -> &SymbolicNonlinearProblem {
        &self.problem
    }

    /// Consumes the prepared object and returns its compatibility problem.
    pub fn into_problem(self) -> SymbolicNonlinearProblem {
        self.problem
    }

    /// Returns the ordered variable names.
    pub fn variables(&self) -> &[String] {
        self.problem.variables()
    }

    /// Returns the declared parameter schema, if any.
    pub fn parameter_schema(&self) -> Option<&NonlinearParameterSchema> {
        self.problem.parameter_schema()
    }

    /// Returns the effective backend selected during preparation.
    pub fn backend_kind(&self) -> SymbolicBackendKind {
        self.problem.backend_kind()
    }

    /// Returns the prepared Lambdify execution policy, if Lambdify is active.
    pub fn lambdify_execution_policy(&self) -> Option<LambdifyExecutionPolicy> {
        self.problem.lambdify_execution_policy()
    }

    /// Builds an AOT manifest view without changing the prepared backend.
    pub fn prepare_dense_aot_problem(
        &self,
        options: SymbolicDenseAotOptions,
    ) -> PreparedSymbolicNonlinearAotProblem<'_> {
        self.problem.prepare_dense_aot_problem(options)
    }

    /// Creates a bound view after verifying that the binding belongs to this schema.
    pub fn bind(
        &self,
        parameter_values: NonlinearParameterValues,
    ) -> Result<BoundSymbolicNonlinearProblem<'_>, SolveError> {
        let schema = self.parameter_schema().ok_or_else(|| {
            SolveError::ParameterSchemaMismatch(
                "cannot bind values because the prepared problem has no parameter schema"
                    .to_string(),
            )
        })?;
        if parameter_values.schema() != schema {
            return Err(SolveError::ParameterSchemaMismatch(
                "parameter values were created for a different ordered schema".to_string(),
            ));
        }
        Ok(BoundSymbolicNonlinearProblem {
            prepared: self,
            parameter_values: Some(parameter_values),
        })
    }

    /// Validates raw values against the prepared schema and creates a bound view.
    pub fn bind_values(
        &self,
        values: DVector<f64>,
    ) -> Result<BoundSymbolicNonlinearProblem<'_>, SolveError> {
        let schema = self.parameter_schema().ok_or_else(|| {
            SolveError::ParameterSchemaMismatch(
                "cannot bind values because the prepared problem has no parameter schema"
                    .to_string(),
            )
        })?;
        self.bind(NonlinearParameterValues::new(schema, values)?)
    }

    /// Creates a bound view for a non-parameterized prepared problem.
    pub fn bind_without_parameters(&self) -> Result<BoundSymbolicNonlinearProblem<'_>, SolveError> {
        if self.parameter_schema().is_some() {
            return Err(SolveError::ParameterSchemaMismatch(
                "the prepared problem declares parameters; bind their values explicitly"
                    .to_string(),
            ));
        }
        Ok(BoundSymbolicNonlinearProblem {
            prepared: self,
            parameter_values: None,
        })
    }

    /// Reuses values supplied through `SymbolicProblemOptions`, if present.
    pub fn bind_initial(&self) -> Result<BoundSymbolicNonlinearProblem<'_>, SolveError> {
        match self.problem.parameter_values.clone() {
            Some(values) => self.bind(values),
            None => self.bind_without_parameters(),
        }
    }
}

impl<'a> BoundSymbolicNonlinearProblem<'a> {
    /// Returns the immutable prepared object behind this binding.
    pub fn prepared(&self) -> &'a PreparedSymbolicNonlinearProblem {
        self.prepared
    }

    /// Returns the currently bound parameter values, if any.
    pub fn parameter_values(&self) -> Option<&DVector<f64>> {
        self.parameter_values
            .as_ref()
            .map(NonlinearParameterValues::as_vector)
    }
}

impl SymbolicNonlinearProblem {
    /// Builds a symbolic problem from parsed expressions.
    pub fn from_expressions(
        equations: Vec<Expr>,
        variables: Option<Vec<String>>,
        equation_parameters: Option<Vec<String>>,
        equation_parameter_values: Option<DVector<f64>>,
    ) -> Result<Self, SolveError> {
        Self::from_expressions_with_backend(
            equations,
            variables,
            equation_parameters,
            equation_parameter_values,
            SymbolicBackendConfig::default(),
        )
    }

    /// Builds a symbolic problem from parsed expressions and grouped symbolic options.
    pub fn from_expressions_with_options(
        equations: Vec<Expr>,
        options: SymbolicProblemOptions,
    ) -> Result<Self, SolveError> {
        Self::from_expressions_with_backend_and_policy(
            equations,
            options.variables,
            options.equation_parameters,
            options.equation_parameter_values,
            options.lambdify_execution_policy,
            options.backend_config,
        )
    }

    /// Builds a symbolic problem through the explicit backend-selection layer.
    ///
    /// This is the nonlinear dense analogue of the newer BVP path:
    /// the symbolic problem is first prepared in a backend-agnostic form, then
    /// the requested backend policy decides whether the callable provider should
    /// use lambdify or a linked compiled dense AOT backend.
    pub fn from_expressions_with_backend_selection(
        equations: Vec<Expr>,
        options: SymbolicProblemOptions,
        policy: SymbolicBackendSelectionPolicy,
        resolver: Option<&AotResolver>,
        aot_options: SymbolicDenseAotOptions,
    ) -> Result<Self, SolveError> {
        let preparation_started = Instant::now();
        let mut problem = Self::from_expressions_with_backend_and_policy(
            equations,
            options.variables,
            options.equation_parameters,
            options.equation_parameter_values,
            options.lambdify_execution_policy,
            SymbolicBackendConfig::lambdify(),
        )?;

        let selected = select_symbolic_nonlinear_backend(&problem, policy, resolver, aot_options);
        match selected.effective_backend {
            SelectedSymbolicNonlinearBackendKind::Lambdify => {
                problem.preparation_report.artifact_policy =
                    artifact_policy_for_backend_selection(policy);
                if policy == SymbolicBackendSelectionPolicy::PreferAotThenLambdify {
                    problem.preparation_report.artifact_action =
                        SymbolicArtifactAction::FallbackToLambdify;
                }
                problem.preparation_report.preparation_duration = preparation_started.elapsed();
                Ok(problem)
            }
            SelectedSymbolicNonlinearBackendKind::AotCompiled => {
                let prepared_key = selected
                    .prepared_aot_problem
                    .as_ref()
                    .map(PreparedSymbolicNonlinearAotProblem::problem_key)
                    .ok_or_else(|| {
                        SolveError::InvalidConfig(
                        "compiled nonlinear AOT selection did not include prepared problem data"
                            .to_string(),
                    )
                    })?;
                let Some(linked) = resolve_linked_dense_backend(&prepared_key) else {
                    return Err(SolveError::CompiledAotRuntimeUnavailable(format!(
                        "no linked dense nonlinear AOT runtime registered for problem key {}",
                        prepared_key
                    )));
                };
                let prepared_manifest = selected
                    .prepared_aot_problem
                    .as_ref()
                    .expect("prepared AOT key was just validated")
                    .manifest();
                drop(selected);
                problem.backend_config = SymbolicBackendConfig::aot();
                problem.backend = PreparedSymbolicBackend::from_linked_dense(
                    linked,
                    &problem.variables,
                    problem
                        .parameter_schema
                        .as_ref()
                        .map(|schema| schema.names()),
                );
                problem.preparation_report.artifact_policy =
                    artifact_policy_for_backend_selection(policy);
                problem.preparation_report.artifact_action = SymbolicArtifactAction::Reused;
                problem.preparation_report.artifact_key = Some(prepared_key);
                problem.preparation_report.generated_residual_jobs =
                    Some(prepared_manifest.functions.residual_chunks.len());
                problem.preparation_report.generated_jacobian_jobs =
                    Some(prepared_manifest.functions.jacobian_chunks.len());
                problem.preparation_report.effective_backend = SymbolicBackendKind::Aot;
                problem.preparation_report.preparation_duration = preparation_started.elapsed();
                Ok(problem)
            }
            SelectedSymbolicNonlinearBackendKind::AotRegisteredButNotBuilt => {
                Err(SolveError::InvalidConfig(
                    "dense nonlinear AOT backend is registered but not built".to_string(),
                ))
            }
            SelectedSymbolicNonlinearBackendKind::AotMissing => Err(SolveError::InvalidConfig(
                "dense nonlinear AOT backend was requested but no artifact was found".to_string(),
            )),
        }
    }

    /// Builds a symbolic problem from parsed expressions and explicit backend config.
    pub fn from_expressions_with_backend(
        equations: Vec<Expr>,
        variables: Option<Vec<String>>,
        equation_parameters: Option<Vec<String>>,
        equation_parameter_values: Option<DVector<f64>>,
        backend_config: SymbolicBackendConfig,
    ) -> Result<Self, SolveError> {
        Self::from_expressions_with_backend_and_policy(
            equations,
            variables,
            equation_parameters,
            equation_parameter_values,
            LambdifyExecutionPolicy::default(),
            backend_config,
        )
    }

    /// Builds a symbolic problem with an explicit prepared Lambdify execution policy.
    fn from_expressions_with_backend_and_policy(
        equations: Vec<Expr>,
        variables: Option<Vec<String>>,
        equation_parameters: Option<Vec<String>>,
        equation_parameter_values: Option<DVector<f64>>,
        lambdify_execution_policy: LambdifyExecutionPolicy,
        backend_config: SymbolicBackendConfig,
    ) -> Result<Self, SolveError> {
        let preparation_started = Instant::now();
        if equations.is_empty() {
            return Err(SolveError::InvalidConfig(
                "equation system must not be empty".to_string(),
            ));
        }

        let parameter_schema = equation_parameters
            .map(NonlinearParameterSchema::new)
            .transpose()?;

        let variables = match variables {
            Some(variables) => variables,
            None => {
                let mut args = equations
                    .iter()
                    .flat_map(|expr| expr.all_arguments_are_variables())
                    .collect::<Vec<_>>();
                args.sort();
                args.dedup();
                if let Some(schema) = &parameter_schema {
                    args.retain(|name| schema.index_of(name).is_none());
                }
                args
            }
        };

        validate_variable_names(&variables)?;
        if variables.is_empty() {
            return Err(SolveError::InvalidConfig(
                "failed to infer variables from symbolic equations".to_string(),
            ));
        }
        if variables.len() != equations.len() {
            return Err(SolveError::DimensionMismatch {
                expected: equations.len(),
                actual: variables.len(),
                context: "symbolic variables vs equations",
            });
        }

        if let Some(schema) = &parameter_schema {
            if let Some(variable) = variables
                .iter()
                .find(|variable| schema.index_of(variable).is_some())
            {
                return Err(SolveError::InvalidParameterSchema(format!(
                    "parameter name '{variable}' is also declared as a variable"
                )));
            }
        }

        let declared_names = variables
            .iter()
            .cloned()
            .chain(
                parameter_schema
                    .as_ref()
                    .into_iter()
                    .flat_map(|schema| schema.names().iter().cloned()),
            )
            .collect::<std::collections::HashSet<_>>();
        for (equation_index, equation) in equations.iter().enumerate() {
            if let Some(name) = equation
                .all_arguments_are_variables()
                .into_iter()
                .find(|name| !declared_names.contains(name))
            {
                return Err(SolveError::UndeclaredSymbol {
                    equation_index,
                    name,
                });
            }
        }

        let parameter_values = match (parameter_schema.as_ref(), equation_parameter_values) {
            (Some(schema), Some(values)) => Some(NonlinearParameterValues::new(schema, values)?),
            (Some(_), None) => None,
            (None, Some(_)) => {
                return Err(SolveError::InvalidParameterSchema(
                    "parameter values were supplied without parameter names".to_string(),
                ));
            }
            (None, None) => None,
        };
        let parameter_names = parameter_schema.as_ref().map(|schema| schema.names());

        let backend = PreparedSymbolicBackend::from_expressions(
            &equations,
            &variables,
            parameter_names,
            lambdify_execution_policy,
            &backend_config,
        )?;
        let symbolic_jacobian = backend
            .symbolic_jacobian()
            .map_or_else(Vec::new, <[Vec<Expr>]>::to_vec);
        let prepared_backend_kind = backend.kind();

        Ok(Self {
            backend_config,
            backend,
            equations,
            symbolic_jacobian,
            variables,
            parameter_schema,
            parameter_values,
            preparation_report: SymbolicPreparationReport::direct_lambdify(
                preparation_started.elapsed(),
                prepared_backend_kind,
            ),
        })
    }

    /// Builds a symbolic problem from equation strings.
    pub fn from_strings(
        equations: Vec<String>,
        variables: Option<Vec<String>>,
        equation_parameters: Option<Vec<String>>,
        equation_parameter_values: Option<DVector<f64>>,
    ) -> Result<Self, SolveError> {
        Self::from_strings_with_backend(
            equations,
            variables,
            equation_parameters,
            equation_parameter_values,
            SymbolicBackendConfig::default(),
        )
    }

    /// Builds a symbolic problem from equation strings and grouped symbolic options.
    pub fn from_strings_with_options(
        equations: Vec<String>,
        options: SymbolicProblemOptions,
    ) -> Result<Self, SolveError> {
        let expressions = equations
            .iter()
            .map(|equation| Expr::parse_expression(equation))
            .collect::<Vec<_>>();
        Self::from_expressions_with_options(expressions, options)
    }

    /// Builds a symbolic problem from equation strings through explicit backend selection.
    pub fn from_strings_with_backend_selection(
        equations: Vec<String>,
        options: SymbolicProblemOptions,
        policy: SymbolicBackendSelectionPolicy,
        resolver: Option<&AotResolver>,
        aot_options: SymbolicDenseAotOptions,
    ) -> Result<Self, SolveError> {
        let expressions = equations
            .iter()
            .map(|equation| Expr::parse_expression(equation))
            .collect::<Vec<_>>();
        Self::from_expressions_with_backend_selection(
            expressions,
            options,
            policy,
            resolver,
            aot_options,
        )
    }

    /// Builds a symbolic problem from equation strings and explicit backend config.
    pub fn from_strings_with_backend(
        equations: Vec<String>,
        variables: Option<Vec<String>>,
        equation_parameters: Option<Vec<String>>,
        equation_parameter_values: Option<DVector<f64>>,
        backend_config: SymbolicBackendConfig,
    ) -> Result<Self, SolveError> {
        let expressions = equations
            .iter()
            .map(|equation| Expr::parse_expression(equation))
            .collect::<Vec<_>>();
        Self::from_expressions_with_backend(
            expressions,
            variables,
            equation_parameters,
            equation_parameter_values,
            backend_config,
        )
    }

    /// Returns the variable names used by the symbolic system.
    pub fn variables(&self) -> &[String] {
        &self.variables
    }

    /// Returns the original symbolic equations.
    pub fn equations(&self) -> &[Expr] {
        &self.equations
    }

    /// Returns the symbolic dense Jacobian retained by the symbolic layer.
    pub fn symbolic_jacobian(&self) -> &[Vec<Expr>] {
        &self.symbolic_jacobian
    }

    /// Returns the symbolic backend configuration used to prepare the problem.
    pub fn backend_config(&self) -> &SymbolicBackendConfig {
        &self.backend_config
    }

    /// Returns immutable preparation/build telemetry for this symbolic provider.
    pub fn preparation_report(&self) -> &SymbolicPreparationReport {
        &self.preparation_report
    }

    /// Replaces preparation telemetry when an outer lifecycle wrapper adds
    /// backend-selection or artifact-build stages to the original report.
    pub(crate) fn replace_preparation_report(&mut self, report: SymbolicPreparationReport) {
        self.preparation_report = report;
    }

    /// Returns the effective prepared backend kind.
    pub fn backend_kind(&self) -> SymbolicBackendKind {
        self.backend.kind()
    }

    /// Returns the configured Lambdify execution policy, if Lambdify is active.
    pub fn lambdify_execution_policy(&self) -> Option<LambdifyExecutionPolicy> {
        self.backend.lambdify_execution_policy()
    }

    /// Builds a dense AOT-ready prepared problem from the symbolic nonlinear system.
    ///
    /// This is the narrow bridge from the nonlinear symbolic frontend into the
    /// generic codegen lifecycle. The returned value fixes:
    /// - parameter-first flattened input order,
    /// - symbolic residual and dense Jacobian sources,
    /// - runtime chunking choices for both outputs,
    /// - manifest-ready metadata such as function names and problem key.
    pub fn prepare_dense_aot_problem(
        &self,
        options: SymbolicDenseAotOptions,
    ) -> PreparedSymbolicNonlinearAotProblem<'_> {
        let variable_refs = self
            .variables
            .iter()
            .map(|name| name.as_str())
            .collect::<Vec<_>>();
        let parameter_refs = self.parameter_schema.as_ref().map(|schema| {
            schema
                .names()
                .iter()
                .map(|name| name.as_str())
                .collect::<Vec<_>>()
        });
        let mut flattened_input_names = Vec::new();
        if let Some(params) = &parameter_refs {
            flattened_input_names.extend(params.iter().copied());
        }
        flattened_input_names.extend(variable_refs.iter().copied());

        PreparedSymbolicNonlinearAotProblem {
            equations: &self.equations,
            symbolic_jacobian: &self.symbolic_jacobian,
            variable_refs,
            parameter_refs,
            flattened_input_names,
            residual_fn_name: "eval_nonlinear_residual".to_string(),
            jacobian_fn_name: "eval_nonlinear_jacobian".to_string(),
            residual_strategy: options.residual_strategy,
            jacobian_strategy: options.jacobian_strategy,
        }
    }

    /// Updates parameter values for parameterized equations.
    pub fn set_parameter_values(&mut self, values: DVector<f64>) -> Result<(), SolveError> {
        let schema = self.parameter_schema.as_ref().ok_or_else(|| {
            SolveError::InvalidParameterSchema(
                "cannot set parameter values because no parameter schema was declared".to_string(),
            )
        })?;
        let validated = NonlinearParameterValues::new(schema, values)?;
        self.parameter_values = Some(validated);
        Ok(())
    }

    /// Returns the ordered symbolic parameter schema, when one was declared.
    pub fn parameter_schema(&self) -> Option<&NonlinearParameterSchema> {
        self.parameter_schema.as_ref()
    }

    /// Returns the currently bound parameter values, when available.
    pub fn parameter_values(&self) -> Option<&DVector<f64>> {
        self.parameter_values
            .as_ref()
            .map(NonlinearParameterValues::as_vector)
    }

    /// Shared implementation of residual evaluation.
    fn residual_impl(&self, x: &DVector<f64>) -> Result<DVector<f64>, SolveError> {
        self.residual_impl_with_parameter_values(
            x,
            self.parameter_values
                .as_ref()
                .map(NonlinearParameterValues::as_vector),
        )
    }

    /// Evaluates the residual with an explicit immutable parameter binding.
    fn residual_impl_with_parameter_values(
        &self,
        x: &DVector<f64>,
        parameter_values: Option<&DVector<f64>>,
    ) -> Result<DVector<f64>, SolveError> {
        self.backend.residual(
            x,
            self.parameter_schema.as_ref().map(|schema| schema.names()),
            parameter_values,
            &self.variables,
        )
    }

    /// Shared implementation of Jacobian evaluation.
    fn jacobian_impl(&self, x: &DVector<f64>) -> Result<DMatrix<f64>, SolveError> {
        self.jacobian_impl_with_parameter_values(
            x,
            self.parameter_values
                .as_ref()
                .map(NonlinearParameterValues::as_vector),
        )
    }

    /// Evaluates the Jacobian with an explicit immutable parameter binding.
    fn jacobian_impl_with_parameter_values(
        &self,
        x: &DVector<f64>,
        parameter_values: Option<&DVector<f64>>,
    ) -> Result<DMatrix<f64>, SolveError> {
        self.backend.jacobian(
            x,
            self.parameter_schema.as_ref().map(|schema| schema.names()),
            parameter_values,
            &self.variables,
        )
    }
}

impl NonlinearProblem for SymbolicNonlinearProblem {
    /// Returns the number of unknowns in the symbolic system.
    fn dimension(&self) -> usize {
        self.variables.len()
    }

    /// Returns whether the selected symbolic backend can fill residual storage.
    fn supports_residual_into(&self) -> bool {
        self.backend.supports_residual_into()
    }

    /// Evaluates the symbolic residual vector.
    fn residual(&self, x: &DVector<f64>) -> Result<DVector<f64>, SolveError> {
        self.residual_impl(x)
    }

    /// Evaluates the symbolic residual into reusable storage.
    fn residual_into(&self, x: &DVector<f64>, out: &mut DVector<f64>) -> Result<(), SolveError> {
        self.backend.residual_into(
            x,
            self.parameter_schema.as_ref().map(|schema| schema.names()),
            self.parameter_values
                .as_ref()
                .map(NonlinearParameterValues::as_vector),
            &self.variables,
            out,
        )
    }
}

impl JacobianProvider for SymbolicNonlinearProblem {
    /// Evaluates the symbolic Jacobian matrix.
    fn jacobian(&self, x: &DVector<f64>) -> Result<DMatrix<f64>, SolveError> {
        self.jacobian_impl(x)
    }

    /// Returns whether the Lambdify backend can fill Jacobian storage directly.
    fn supports_jacobian_into(&self) -> bool {
        self.backend.supports_jacobian_into()
    }

    /// Evaluates the symbolic Jacobian into reusable storage.
    fn jacobian_into(&self, x: &DVector<f64>, out: &mut DMatrix<f64>) -> Result<(), SolveError> {
        self.backend.jacobian_into(
            x,
            self.parameter_schema.as_ref().map(|schema| schema.names()),
            self.parameter_values
                .as_ref()
                .map(NonlinearParameterValues::as_vector),
            &self.variables,
            out,
        )
    }
}

impl<'a> NonlinearProblem for BoundSymbolicNonlinearProblem<'a> {
    /// Returns the number of unknowns in the prepared symbolic system.
    fn dimension(&self) -> usize {
        self.prepared.problem.dimension()
    }

    /// Returns whether the prepared symbolic backend can fill residual storage.
    fn supports_residual_into(&self) -> bool {
        self.prepared.problem.backend.supports_residual_into()
    }

    /// Evaluates the residual using this view's immutable parameter binding.
    fn residual(&self, x: &DVector<f64>) -> Result<DVector<f64>, SolveError> {
        self.prepared
            .problem
            .residual_impl_with_parameter_values(x, self.parameter_values())
    }

    /// Evaluates the bound symbolic residual into reusable storage.
    fn residual_into(&self, x: &DVector<f64>, out: &mut DVector<f64>) -> Result<(), SolveError> {
        self.prepared.problem.backend.residual_into(
            x,
            self.prepared
                .problem
                .parameter_schema
                .as_ref()
                .map(|schema| schema.names()),
            self.parameter_values(),
            &self.prepared.problem.variables,
            out,
        )
    }
}

impl<'a> JacobianProvider for BoundSymbolicNonlinearProblem<'a> {
    /// Evaluates the Jacobian using this view's immutable parameter binding.
    fn jacobian(&self, x: &DVector<f64>) -> Result<DMatrix<f64>, SolveError> {
        self.prepared
            .problem
            .jacobian_impl_with_parameter_values(x, self.parameter_values())
    }

    /// Returns whether the bound Lambdify backend can fill Jacobian storage directly.
    fn supports_jacobian_into(&self) -> bool {
        self.prepared.problem.backend.supports_jacobian_into()
    }

    /// Evaluates the bound symbolic Jacobian into reusable storage.
    fn jacobian_into(&self, x: &DVector<f64>, out: &mut DMatrix<f64>) -> Result<(), SolveError> {
        self.prepared.problem.backend.jacobian_into(
            x,
            self.prepared
                .problem
                .parameter_schema
                .as_ref()
                .map(|schema| schema.names()),
            self.parameter_values(),
            &self.prepared.problem.variables,
            out,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::Nonlinear_systems::LM_Nielsen::NielsenLevenbergMarquardtMethod;
    use crate::numerical::Nonlinear_systems::LM_vanilla::LevenbergMarquardtMethod;
    use crate::numerical::Nonlinear_systems::NR_damped::DampedNewtonMethod;
    use crate::numerical::Nonlinear_systems::engine::{NewtonMethod, SolveOptions, SolverEngine};
    use crate::numerical::Nonlinear_systems::problem::Bounds;
    use crate::numerical::Nonlinear_systems::trust_region::*;
    use crate::symbolic::codegen::codegen_aot_resolution::AotResolver;
    use approx::assert_relative_eq;
    use std::collections::HashMap;
    use std::f64;

    fn elementary_problem() -> SymbolicNonlinearProblem {
        SymbolicNonlinearProblem::from_strings(
            vec!["x^2+y^2-10".to_string(), "x-y-4".to_string()],
            Some(vec!["x".to_string(), "y".to_string()]),
            None,
            None,
        )
        .expect("symbolic problem should build")
    }

    fn elementary_problem_with_options() -> SymbolicNonlinearProblem {
        SymbolicNonlinearProblem::from_strings_with_options(
            vec!["x^2+y^2-10".to_string(), "x-y-4".to_string()],
            SymbolicProblemOptions::new()
                .with_variables(vec!["x".to_string(), "y".to_string()])
                .with_lambdify_backend(),
        )
        .expect("symbolic problem should build")
    }

    fn solve_elementary_with_bounds<M>(method: M) -> DVector<f64>
    where
        M: crate::numerical::Nonlinear_systems::engine::NonlinearMethod,
    {
        let options = SolveOptions {
            bounds: Some(Bounds::new(vec![(-10.0, 10.0), (-10.0, 10.0)]).expect("bounds")),
            ..SolveOptions::default()
        };
        SolverEngine::new(method, options)
            .solve(&elementary_problem(), DVector::from_vec(vec![1.0, 1.0]))
            .expect("solve should succeed")
            .x
    }

    fn chemistry_problem(dgm: f64) -> (SymbolicNonlinearProblem, Vec<String>) {
        let symbolic = Expr::Symbols("N0, N1, N2, Np, Lambda0, Lambda1");
        let d_g0 = Expr::Const(-450.0e3);
        let d_g1 = Expr::Const(-150.0e3);
        let d_g2 = Expr::Const(-50e3);
        let d_gm = Expr::Const(dgm);
        let n0 = symbolic[0].clone();
        let n1 = symbolic[1].clone();
        let n2 = symbolic[2].clone();
        let np = symbolic[3].clone();
        let lambda0 = symbolic[4].clone();
        let lambda1 = symbolic[5].clone();
        let rt = Expr::Const(8.314) * Expr::Const(273.15);

        let eq_mu = vec![
            lambda0.clone()
                + Expr::Const(2.0) * lambda1.clone()
                + (d_g0.clone() + rt.clone() * Expr::ln(n0.clone() / np.clone())) / d_gm.clone(),
            lambda0
                + lambda1.clone()
                + (d_g1 + rt.clone() * Expr::ln(n1.clone() / np.clone())) / d_gm.clone(),
            Expr::Const(2.0) * lambda1 + (d_g2 + rt * Expr::ln(n2.clone() / np.clone())) / d_gm,
        ];
        let eq_sum = vec![n0.clone() + n1.clone() + n2.clone() - np.clone()];
        let eq_comp = vec![
            n0.clone() + n1.clone() - Expr::Const(0.999),
            Expr::Const(2.0) * n0 + n1 + Expr::Const(2.0) * n2 - Expr::Const(1.501),
        ];

        let mut system = Vec::new();
        system.extend(eq_mu);
        system.extend(eq_sum);
        system.extend(eq_comp);
        let variables = symbolic.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        (
            SymbolicNonlinearProblem::from_expressions(system, Some(variables.clone()), None, None)
                .expect("chemistry problem should build"),
            variables,
        )
    }

    fn assert_chemistry_constraints(solution: &DVector<f64>, variables: &[String], tol: f64) {
        let map = variables
            .iter()
            .zip(solution.iter())
            .map(|(k, v)| (k.clone(), *v))
            .collect::<HashMap<String, f64>>();
        let n0 = map["N0"];
        let n1 = map["N1"];
        let n2 = map["N2"];
        let np = map["Np"];
        assert!((n0 + n1 - 0.999).abs() < tol);
        assert!((n0 + n1 + n2 - np).abs() < tol);
        assert!((2.0 * n0 + n1 + 2.0 * n2 - 1.501).abs() < tol);
    }

    #[test]
    fn symbolic_problem_solves_from_expressions() {
        let problem = elementary_problem();
        let result = SolverEngine::new(
            crate::numerical::Nonlinear_systems::engine::NewtonMethod,
            SolveOptions::default(),
        )
        .solve(&problem, DVector::from_vec(vec![1.0, 1.0]))
        .expect("solve should succeed");
        assert_relative_eq!(result.x[0], 3.0, epsilon = 1e-8);
        assert_relative_eq!(result.x[1], -1.0, epsilon = 1e-8);
    }

    #[test]
    fn symbolic_problem_options_builds_problem_through_preferred_path() {
        let problem = elementary_problem_with_options();
        let result = SolverEngine::new(
            crate::numerical::Nonlinear_systems::engine::NewtonMethod,
            SolveOptions::default(),
        )
        .solve(&problem, DVector::from_vec(vec![1.0, 1.0]))
        .expect("solve should succeed");
        assert_relative_eq!(result.x[0], 3.0, epsilon = 1e-8);
        assert_relative_eq!(result.x[1], -1.0, epsilon = 1e-8);
    }

    #[test]
    fn prepared_problem_binds_independent_parameter_views() {
        let prepared = PreparedSymbolicNonlinearProblem::from_strings(
            vec!["a*x-2".to_string(), "y-1".to_string()],
            SymbolicProblemOptions::new()
                .with_variables(vec!["x".to_string(), "y".to_string()])
                .with_equation_parameters(vec!["a".to_string()]),
        )
        .expect("prepared problem should build");

        let first = prepared
            .bind_values(DVector::from_vec(vec![2.0]))
            .expect("first binding should validate");
        let second = prepared
            .bind_values(DVector::from_vec(vec![4.0]))
            .expect("second binding should validate");
        let x0 = DVector::from_vec(vec![0.0, 0.0]);

        let first_residual = first.residual(&x0).expect("first residual");
        let second_residual = second.residual(&x0).expect("second residual");
        assert_relative_eq!(first_residual[0], -2.0, epsilon = 1e-12);
        assert_relative_eq!(second_residual[0], -2.0, epsilon = 1e-12);
        assert_relative_eq!(
            first.jacobian(&x0).expect("first jacobian")[(0, 0)],
            2.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            second.jacobian(&x0).expect("second jacobian")[(0, 0)],
            4.0,
            epsilon = 1e-12
        );
        assert_eq!(prepared.backend_kind(), SymbolicBackendKind::Lambdify);
        assert_eq!(first.prepared().variables(), &["x", "y"]);
    }

    #[test]
    fn lambdify_into_evaluations_match_owned_values_and_reuse_buffers() {
        let prepared = PreparedSymbolicNonlinearProblem::from_strings(
            vec!["a*x^2+y-3".to_string(), "x-y".to_string()],
            SymbolicProblemOptions::new()
                .with_variables(vec!["x".to_string(), "y".to_string()])
                .with_equation_parameters(vec!["a".to_string()]),
        )
        .expect("parameterized Lambdify problem should build");
        let bound = prepared
            .bind_values(DVector::from_vec(vec![2.0]))
            .expect("parameter binding should validate");
        let x = DVector::from_vec(vec![1.25, 0.75]);

        assert!(bound.supports_residual_into());
        assert!(bound.supports_jacobian_into());

        let expected_residual = bound.residual(&x).expect("owned residual");
        let expected_jacobian = bound.jacobian(&x).expect("owned Jacobian");
        let mut residual = DVector::zeros(2);
        let residual_ptr = residual.as_slice().as_ptr();
        let mut jacobian = DMatrix::zeros(2, 2);
        let jacobian_ptr = jacobian.as_slice().as_ptr();

        bound
            .residual_into(&x, &mut residual)
            .expect("residual_into should succeed");
        bound
            .jacobian_into(&x, &mut jacobian)
            .expect("jacobian_into should succeed");

        assert_eq!(residual, expected_residual);
        assert_eq!(jacobian, expected_jacobian);
        assert_eq!(residual.as_slice().as_ptr(), residual_ptr);
        assert_eq!(jacobian.as_slice().as_ptr(), jacobian_ptr);

        let second_x = DVector::from_vec(vec![0.5, -0.25]);
        bound
            .residual_into(&second_x, &mut residual)
            .expect("second residual_into should succeed");
        bound
            .jacobian_into(&second_x, &mut jacobian)
            .expect("second jacobian_into should succeed");
        assert_eq!(residual.as_slice().as_ptr(), residual_ptr);
        assert_eq!(jacobian.as_slice().as_ptr(), jacobian_ptr);
    }

    #[test]
    fn parameterized_lambdify_reuses_input_workspace_capacity() {
        let prepared = PreparedSymbolicNonlinearProblem::from_strings(
            vec!["a*x^2+y-3".to_string(), "x-y".to_string()],
            SymbolicProblemOptions::new()
                .with_variables(vec!["x".to_string(), "y".to_string()])
                .with_equation_parameters(vec!["a".to_string()]),
        )
        .expect("parameterized Lambdify problem should build");
        let bound = prepared
            .bind_values(DVector::from_vec(vec![2.0]))
            .expect("parameter binding should validate");
        let x = DVector::from_vec(vec![1.25, 0.75]);
        let mut residual = DVector::zeros(2);
        let mut jacobian = DMatrix::zeros(2, 2);

        bound
            .residual_into(&x, &mut residual)
            .expect("first parameterized residual should evaluate");
        bound
            .jacobian_into(&x, &mut jacobian)
            .expect("first parameterized Jacobian should evaluate");
        let first_capacity =
            crate::numerical::Nonlinear_systems::symbolic_legacy::lambdify_input_workspace_capacity(
            );

        for _ in 0..8 {
            bound
                .residual_into(&x, &mut residual)
                .expect("repeated parameterized residual should evaluate");
            bound
                .jacobian_into(&x, &mut jacobian)
                .expect("repeated parameterized Jacobian should evaluate");
        }

        let final_capacity =
            crate::numerical::Nonlinear_systems::symbolic_legacy::lambdify_input_workspace_capacity(
            );
        assert!(first_capacity >= 3);
        assert_eq!(final_capacity, first_capacity);
    }

    #[test]
    fn unparameterized_lambdify_into_evaluations_use_the_variable_schema() {
        let prepared = PreparedSymbolicNonlinearProblem::from_strings(
            vec!["x^2+y".to_string(), "x-y".to_string()],
            SymbolicProblemOptions::new()
                .with_variables(vec!["x".to_string(), "y".to_string()])
                .with_lambdify_backend(),
        )
        .expect("unparameterized Lambdify problem should build");
        let bound = prepared
            .bind_without_parameters()
            .expect("unparameterized problem should bind");
        let x = DVector::from_vec(vec![2.0, 0.5]);
        let mut residual = DVector::zeros(2);
        let mut jacobian = DMatrix::zeros(2, 2);

        bound
            .residual_into(&x, &mut residual)
            .expect("residual_into should succeed");
        bound
            .jacobian_into(&x, &mut jacobian)
            .expect("jacobian_into should succeed");

        assert_eq!(residual.as_slice(), &[4.5, 1.5]);
        assert_eq!(
            jacobian,
            DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, -1.0])
        );
    }

    #[test]
    fn row_major_jacobian_adapter_preserves_rectangular_layout() {
        let row_major = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut column_major = vec![0.0; row_major.len()];
        copy_row_major_jacobian_into_column_major(&row_major, &mut column_major, 2, 3);

        assert_eq!(
            DMatrix::from_column_slice(2, 3, &column_major),
            DMatrix::from_row_slice(2, 3, &row_major)
        );
    }

    #[test]
    fn parallel_lambdify_matches_sequential_sparse_jacobian() {
        let variables = (0..8).map(|index| format!("x{index}")).collect::<Vec<_>>();
        let equations = (0..8)
            .map(|index| {
                let next = (index + 1) % 8;
                format!("a*x{index}^2+x{next}-1")
            })
            .collect::<Vec<_>>();
        let base_options = SymbolicProblemOptions::new()
            .with_variables(variables)
            .with_equation_parameters(vec!["a".to_string()]);
        let sequential = PreparedSymbolicNonlinearProblem::from_strings(
            equations.clone(),
            base_options
                .clone()
                .with_lambdify_execution_policy(LambdifyExecutionPolicy::Sequential),
        )
        .expect("sequential Lambdify problem should build");
        let parallel = PreparedSymbolicNonlinearProblem::from_strings(
            equations,
            base_options
                .with_lambdify_execution_policy(LambdifyExecutionPolicy::Parallel { min_work: 1 }),
        )
        .expect("parallel Lambdify problem should build");
        let sequential = sequential
            .bind_values(DVector::from_vec(vec![2.0]))
            .expect("sequential parameter binding should succeed");
        let parallel = parallel
            .bind_values(DVector::from_vec(vec![2.0]))
            .expect("parallel parameter binding should succeed");
        let x = DVector::from_iterator(8, (0..8).map(|index| 0.2 + index as f64 * 0.03));

        assert_eq!(
            sequential.prepared().lambdify_execution_policy(),
            Some(LambdifyExecutionPolicy::Sequential)
        );
        assert_eq!(
            parallel.prepared().lambdify_execution_policy(),
            Some(LambdifyExecutionPolicy::Parallel { min_work: 1 })
        );
        assert_eq!(
            sequential.residual(&x).expect("sequential residual"),
            parallel.residual(&x).expect("parallel residual")
        );
        assert_eq!(
            sequential.jacobian(&x).expect("sequential Jacobian"),
            parallel.jacobian(&x).expect("parallel Jacobian")
        );

        let mut sequential_jacobian = DMatrix::zeros(8, 8);
        let mut parallel_jacobian = DMatrix::zeros(8, 8);
        sequential
            .jacobian_into(&x, &mut sequential_jacobian)
            .expect("sequential Jacobian into");
        parallel
            .jacobian_into(&x, &mut parallel_jacobian)
            .expect("parallel Jacobian into");
        assert_eq!(sequential_jacobian, parallel_jacobian);
    }

    #[test]
    fn large_sparse_parallel_lambdify_is_deterministic_and_thread_safe() {
        const DIMENSION: usize = 64;
        let variables = (0..DIMENSION)
            .map(|index| format!("x{index}"))
            .collect::<Vec<_>>();
        let equations = (0..DIMENSION)
            .map(|index| {
                let current = format!("x{index}");
                let mut equation = format!("a*({current}^2-{:.17})", 1.0 + index as f64 * 0.01);
                if index > 0 {
                    equation.push_str(&format!("+0.08*x{}", index - 1));
                }
                if index + 1 < DIMENSION {
                    equation.push_str(&format!("+0.08*x{}", index + 1));
                }
                equation
            })
            .collect::<Vec<_>>();
        let options = |policy| {
            SymbolicProblemOptions::new()
                .with_variables(variables.clone())
                .with_equation_parameters(vec!["a".to_string()])
                .with_lambdify_execution_policy(policy)
        };
        let sequential_prepared = PreparedSymbolicNonlinearProblem::from_strings(
            equations.clone(),
            options(LambdifyExecutionPolicy::Sequential),
        )
        .expect("large sequential Lambdify problem should build");
        let sequential = sequential_prepared
            .bind_values(DVector::from_vec(vec![1.25]))
            .expect("large sequential binding should succeed");
        let parallel_prepared = std::sync::Arc::new(
            PreparedSymbolicNonlinearProblem::from_strings(
                equations.clone(),
                options(LambdifyExecutionPolicy::Parallel { min_work: 1 }),
            )
            .expect("large parallel Lambdify problem should build"),
        );
        let parallel = parallel_prepared
            .bind_values(DVector::from_vec(vec![1.25]))
            .expect("large parallel binding should succeed");
        let fallback_prepared = PreparedSymbolicNonlinearProblem::from_strings(
            equations,
            options(LambdifyExecutionPolicy::Parallel {
                min_work: usize::MAX,
            }),
        )
        .expect("threshold fallback problem should build");
        let fallback = fallback_prepared
            .bind_values(DVector::from_vec(vec![1.25]))
            .expect("threshold fallback binding should succeed");
        let point = DVector::from_iterator(
            DIMENSION,
            (0..DIMENSION).map(|index| 0.7 + index as f64 * 0.002),
        );
        let expected_residual = sequential.residual(&point).expect("sequential residual");
        let expected_jacobian = sequential.jacobian(&point).expect("sequential Jacobian");

        assert_eq!(
            parallel.residual(&point).expect("parallel residual"),
            expected_residual
        );
        let parallel_jacobian = parallel.jacobian(&point).expect("parallel Jacobian");
        assert_eq!(parallel_jacobian, expected_jacobian);
        assert_eq!(
            fallback.jacobian(&point).expect("fallback Jacobian"),
            expected_jacobian
        );
        for row in 0..DIMENSION {
            for column in 0..DIMENSION {
                if row.abs_diff(column) > 1 {
                    assert_eq!(parallel_jacobian[(row, column)], 0.0);
                }
            }
        }

        let mut repeated_jacobian = DMatrix::zeros(DIMENSION, DIMENSION);
        for _ in 0..4 {
            parallel
                .jacobian_into(&point, &mut repeated_jacobian)
                .expect("repeated parallel Jacobian");
            assert_eq!(repeated_jacobian, expected_jacobian);
        }

        std::thread::scope(|scope| {
            let handles = (0..4).map(|_| {
                let prepared = std::sync::Arc::clone(&parallel_prepared);
                let point = point.clone();
                scope.spawn(move || {
                    let bound = prepared
                        .bind_values(DVector::from_vec(vec![1.25]))
                        .expect("concurrent binding should succeed");
                    let mut residual = DVector::zeros(DIMENSION);
                    let mut jacobian = DMatrix::zeros(DIMENSION, DIMENSION);
                    bound
                        .residual_into(&point, &mut residual)
                        .expect("concurrent residual should succeed");
                    bound
                        .jacobian_into(&point, &mut jacobian)
                        .expect("concurrent Jacobian should succeed");
                    (residual, jacobian)
                })
            });
            for handle in handles {
                let (residual, jacobian) = handle.join().expect("concurrent worker should finish");
                assert_eq!(residual, expected_residual);
                assert_eq!(jacobian, expected_jacobian);
            }
        });
    }

    #[test]
    fn prepared_parallel_lambdify_matches_legacy_callback_values() {
        const DIMENSION: usize = 16;
        let variables = (0..DIMENSION)
            .map(|index| format!("x{index}"))
            .collect::<Vec<_>>();
        let equations = (0..DIMENSION)
            .map(|index| {
                let mut equation = format!("x{index}^2-1.0");
                if index > 0 {
                    equation.push_str(&format!("+0.08*x{}", index - 1));
                }
                if index + 1 < DIMENSION {
                    equation.push_str(&format!("+0.08*x{}", index + 1));
                }
                equation
            })
            .collect::<Vec<_>>();
        let expressions = equations
            .iter()
            .map(|equation| Expr::parse_expression(equation))
            .collect::<Vec<_>>();
        let mut legacy = crate::symbolic::symbolic_functions::Jacobian::new();
        legacy.set_vector_of_functions(expressions);
        legacy.set_variables(variables.iter().map(String::as_str).collect());
        legacy.calc_jacobian();
        legacy.lambdify_vector_funvector_DVector();
        legacy.lambdify_jacobian_DMatrix_parallel();

        let prepared_problem = PreparedSymbolicNonlinearProblem::from_strings(
            equations,
            SymbolicProblemOptions::new()
                .with_variables(variables)
                .with_lambdify_execution_policy(LambdifyExecutionPolicy::Parallel { min_work: 1 }),
        )
        .expect("prepared parity problem should build");
        let prepared = prepared_problem
            .bind_without_parameters()
            .expect("prepared parity problem should bind");
        let point = DVector::from_iterator(
            DIMENSION,
            (0..DIMENSION).map(|index| 0.75 + index as f64 * 0.01),
        );

        assert_eq!(
            prepared.residual(&point).expect("prepared residual"),
            (legacy.lambdified_function_DVector)(&point)
        );
        assert_eq!(
            prepared.jacobian(&point).expect("prepared Jacobian"),
            (legacy.lambdified_jacobian_DMatrix)(&point)
        );
    }

    #[test]
    fn parallel_lambdify_preserves_non_finite_error_semantics() {
        for policy in [
            LambdifyExecutionPolicy::Sequential,
            LambdifyExecutionPolicy::Parallel { min_work: 1 },
        ] {
            let prepared = PreparedSymbolicNonlinearProblem::from_strings(
                vec!["1/x".to_string()],
                SymbolicProblemOptions::new()
                    .with_variables(vec!["x".to_string()])
                    .with_lambdify_execution_policy(policy),
            )
            .expect("non-finite policy problem should build");
            let problem = prepared
                .bind_without_parameters()
                .expect("non-finite policy problem should bind");
            let zero = DVector::from_vec(vec![0.0]);

            assert!(matches!(
                problem.residual(&zero),
                Err(SolveError::ResidualEvaluation(_))
            ));
            assert!(matches!(
                problem.jacobian(&zero),
                Err(SolveError::JacobianEvaluation(_))
            ));
        }
    }

    #[test]
    fn parallel_lambdify_handles_empty_jacobian_columns() {
        let prepared = PreparedSymbolicNonlinearProblem::from_strings(
            vec!["x0^2-1".to_string(), "2*x0+3".to_string()],
            SymbolicProblemOptions::new()
                .with_variables(vec!["x0".to_string(), "x1".to_string()])
                .with_lambdify_execution_policy(LambdifyExecutionPolicy::Parallel { min_work: 1 }),
        )
        .expect("empty-column problem should build");
        let problem = prepared
            .bind_without_parameters()
            .expect("empty-column problem should bind");
        let point = DVector::from_vec(vec![2.0, 7.0]);
        let jacobian = problem.jacobian(&point).expect("empty-column Jacobian");

        assert_eq!(
            jacobian,
            DMatrix::from_row_slice(2, 2, &[4.0, 0.0, 2.0, 0.0])
        );
        assert_eq!(
            problem.residual(&point).expect("empty-column residual")[1],
            7.0
        );

        let threshold_prepared = PreparedSymbolicNonlinearProblem::from_strings(
            vec!["0.00000000000001*x0".to_string(), "x0".to_string()],
            SymbolicProblemOptions::new()
                .with_variables(vec!["x0".to_string(), "x1".to_string()])
                .with_lambdify_execution_policy(LambdifyExecutionPolicy::Parallel { min_work: 1 }),
        )
        .expect("threshold problem should build");
        let threshold_problem = threshold_prepared
            .bind_without_parameters()
            .expect("threshold problem should bind");
        let threshold_jacobian = threshold_problem
            .jacobian(&point)
            .expect("threshold Jacobian");
        assert_eq!(
            threshold_jacobian,
            DMatrix::from_row_slice(2, 2, &[0.0, 0.0, 1.0, 0.0])
        );
    }

    #[test]
    fn parallel_lambdify_large_corpus_matches_sequential() {
        const DIMENSION: usize = 32;
        let variables = (0..DIMENSION)
            .map(|index| format!("x{index}"))
            .collect::<Vec<_>>();
        let targets = [
            (0..DIMENSION)
                .map(|index| 1.0 + index as f64 * 0.01)
                .collect::<Vec<_>>(),
            (0..DIMENSION)
                .map(|index| 0.2 + index as f64 * 0.001)
                .collect::<Vec<_>>(),
            (0..DIMENSION)
                .map(|index| 0.5 + index as f64 * 0.003)
                .collect::<Vec<_>>(),
        ];
        let equations_for = |corpus: usize| {
            (0..DIMENSION)
                .map(|index| {
                    let target = &targets[corpus];
                    let variable = format!("x{index}");
                    let mut equation = match corpus {
                        0 => format!("({variable}^2-{:.17})", target[index] * target[index]),
                        1 => {
                            let h2 = 1.0 / ((DIMENSION + 1) * (DIMENSION + 1)) as f64;
                            format!(
                                "2*({variable}-{:.17})+{h2:.17}*(exp({variable})-exp({:.17}))",
                                target[index], target[index]
                            )
                        }
                        _ => format!("({variable}^2-{:.17})", target[index] * target[index]),
                    };
                    let offsets = match corpus {
                        2 => (1..=5).collect::<Vec<_>>(),
                        _ => vec![1],
                    };
                    for offset in offsets {
                        let coefficient = match corpus {
                            0 => 0.08,
                            1 => -1.0,
                            _ => 0.01 / offset as f64,
                        };
                        if index >= offset {
                            equation.push_str(&format!(
                                "{coefficient:+.17}*(x{}-{:.17})",
                                index - offset,
                                target[index - offset]
                            ));
                        }
                        if index + offset < DIMENSION {
                            equation.push_str(&format!(
                                "{coefficient:+.17}*(x{}-{:.17})",
                                index + offset,
                                target[index + offset]
                            ));
                        }
                    }
                    equation
                })
                .collect::<Vec<_>>()
        };

        for corpus in 0..3 {
            let equations = equations_for(corpus);
            let sequential_prepared = PreparedSymbolicNonlinearProblem::from_strings(
                equations.clone(),
                SymbolicProblemOptions::new()
                    .with_variables(variables.clone())
                    .with_lambdify_execution_policy(LambdifyExecutionPolicy::Sequential),
            )
            .expect("corpus sequential problem should build");
            let parallel_prepared = PreparedSymbolicNonlinearProblem::from_strings(
                equations,
                SymbolicProblemOptions::new()
                    .with_variables(variables.clone())
                    .with_lambdify_execution_policy(LambdifyExecutionPolicy::Parallel {
                        min_work: 1,
                    }),
            )
            .expect("corpus parallel problem should build");
            let sequential = sequential_prepared
                .bind_without_parameters()
                .expect("corpus sequential problem should bind");
            let parallel = parallel_prepared
                .bind_without_parameters()
                .expect("corpus parallel problem should bind");
            let point = DVector::from_iterator(
                DIMENSION,
                targets[corpus].iter().copied().map(|value| value * 0.8),
            );
            assert_eq!(
                parallel.residual(&point).expect("corpus parallel residual"),
                sequential
                    .residual(&point)
                    .expect("corpus sequential residual")
            );
            assert_eq!(
                parallel.jacobian(&point).expect("corpus parallel Jacobian"),
                sequential
                    .jacobian(&point)
                    .expect("corpus sequential Jacobian")
            );
        }
    }

    #[test]
    fn prepared_lambdify_can_be_shared_by_independent_bound_views() {
        fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<PreparedSymbolicNonlinearProblem>();
        assert_send_sync::<BoundSymbolicNonlinearProblem<'static>>();

        let prepared = std::sync::Arc::new(
            PreparedSymbolicNonlinearProblem::from_strings(
                vec!["a*x-2".to_string(), "y-1".to_string()],
                SymbolicProblemOptions::new()
                    .with_variables(vec!["x".to_string(), "y".to_string()])
                    .with_equation_parameters(vec!["a".to_string()]),
            )
            .expect("shared Lambdify problem should build"),
        );

        std::thread::scope(|scope| {
            let handles = [2.0, 4.0].into_iter().map(|parameter| {
                let prepared = std::sync::Arc::clone(&prepared);
                scope.spawn(move || {
                    let bound = prepared
                        .bind_values(DVector::from_vec(vec![parameter]))
                        .expect("thread-local parameter binding should validate");
                    let x = DVector::from_vec(vec![0.5, 1.0]);
                    let residual = bound.residual(&x).expect("thread-local residual");
                    let jacobian = bound.jacobian(&x).expect("thread-local Jacobian");
                    (parameter, residual, jacobian)
                })
            });

            for handle in handles {
                let (parameter, residual, jacobian) = handle.join().expect("worker should finish");
                assert_eq!(residual[0], parameter * 0.5 - 2.0);
                assert_eq!(residual[1], 0.0);
                assert_eq!(jacobian[(0, 0)], parameter);
                assert_eq!(jacobian[(1, 1)], 1.0);
            }
        });
    }

    #[test]
    fn prepared_bound_views_can_solve_without_mutating_preparation() {
        let prepared = PreparedSymbolicNonlinearProblem::from_strings(
            vec!["a*x-2".to_string(), "y-1".to_string()],
            SymbolicProblemOptions::new()
                .with_variables(vec!["x".to_string(), "y".to_string()])
                .with_equation_parameters(vec!["a".to_string()]),
        )
        .expect("prepared problem should build");
        let options = SolveOptions {
            tolerance: 1e-12,
            ..SolveOptions::default()
        };

        let first = prepared
            .bind_values(DVector::from_vec(vec![2.0]))
            .expect("first binding should validate");
        let first_result = SolverEngine::new(NewtonMethod, options.clone())
            .solve(&first, DVector::from_vec(vec![0.0, 0.0]))
            .expect("first bound solve should succeed");
        assert_relative_eq!(first_result.x[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(first_result.x[1], 1.0, epsilon = 1e-10);

        let second = prepared
            .bind_values(DVector::from_vec(vec![4.0]))
            .expect("second binding should validate");
        let second_result = SolverEngine::new(NewtonMethod, options)
            .solve(&second, DVector::from_vec(vec![0.0, 0.0]))
            .expect("second bound solve should succeed");
        assert_relative_eq!(second_result.x[0], 0.5, epsilon = 1e-10);
        assert_relative_eq!(second_result.x[1], 1.0, epsilon = 1e-10);
        assert!(prepared.as_problem().parameter_values().is_none());
    }

    #[test]
    fn prepared_binding_rejects_wrong_schema_and_missing_values() {
        let prepared = PreparedSymbolicNonlinearProblem::from_strings(
            vec!["a*x-1".to_string()],
            SymbolicProblemOptions::new()
                .with_variables(vec!["x".to_string()])
                .with_equation_parameters(vec!["a".to_string()]),
        )
        .expect("prepared problem should build");
        let other_schema = NonlinearParameterSchema::new(vec!["b".to_string()])
            .expect("other schema should build");
        let other_values =
            NonlinearParameterValues::new(&other_schema, DVector::from_vec(vec![2.0]))
                .expect("other values should validate");

        assert!(matches!(
            prepared.bind(other_values),
            Err(SolveError::ParameterSchemaMismatch(_))
        ));
        assert!(matches!(
            prepared.bind_without_parameters(),
            Err(SolveError::ParameterSchemaMismatch(_))
        ));
    }

    #[test]
    fn prepared_backend_selection_preserves_bind_then_solve_contract() {
        let resolver =
            AotResolver::new(crate::symbolic::codegen::codegen_aot_registry::AotRegistry::new());
        let prepared = PreparedSymbolicNonlinearProblem::from_strings_with_backend_selection(
            vec!["a*x-1".to_string()],
            SymbolicProblemOptions::new()
                .with_variables(vec!["x".to_string()])
                .with_equation_parameters(vec!["a".to_string()]),
            SymbolicBackendSelectionPolicy::PreferAotThenLambdify,
            Some(&resolver),
            SymbolicDenseAotOptions::default(),
        )
        .expect("prepared backend selection should fall back cleanly");
        let bound = prepared
            .bind_values(DVector::from_vec(vec![2.0]))
            .expect("binding should validate");

        let residual = bound
            .residual(&DVector::from_vec(vec![0.5]))
            .expect("bound residual should evaluate");
        assert_relative_eq!(residual[0], 0.0, epsilon = 1e-12);
        assert_eq!(prepared.backend_kind(), SymbolicBackendKind::Lambdify);
    }

    #[test]
    fn legacy_symbolic_constructor_matches_typed_options_route() {
        let equations = vec!["a*x+y-3".to_string(), "x-y".to_string()];
        let variables = vec!["x".to_string(), "y".to_string()];
        let parameters = vec!["a".to_string()];
        let values = DVector::from_vec(vec![2.0]);

        let legacy = SymbolicNonlinearProblem::from_strings(
            equations.clone(),
            Some(variables.clone()),
            Some(parameters.clone()),
            Some(values.clone()),
        )
        .expect("legacy constructor should remain usable");
        let typed = SymbolicNonlinearProblem::from_strings_with_options(
            equations,
            SymbolicProblemOptions::new()
                .with_variables(variables)
                .with_equation_parameters(parameters)
                .with_equation_parameter_values(values),
        )
        .expect("typed options constructor should succeed");

        let point = DVector::from_vec(vec![1.25, 0.75]);
        assert_eq!(legacy.backend_kind(), typed.backend_kind());
        assert_eq!(legacy.variables(), typed.variables());
        assert_eq!(legacy.parameter_schema(), typed.parameter_schema());
        assert_eq!(legacy.parameter_values(), typed.parameter_values());
        assert_relative_eq!(
            legacy.residual(&point).expect("legacy residual").as_slice(),
            typed.residual(&point).expect("typed residual").as_slice(),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            legacy.jacobian(&point).expect("legacy Jacobian"),
            typed.jacobian(&point).expect("typed Jacobian"),
            epsilon = 1e-12
        );
    }

    #[test]
    fn symbolic_problem_uses_lambdify_backend_by_default() {
        let problem = elementary_problem();
        assert_eq!(problem.backend_kind(), SymbolicBackendKind::Lambdify);
        assert_eq!(problem.backend_config(), &SymbolicBackendConfig::lambdify());
    }

    #[test]
    fn symbolic_problem_rejects_unwired_aot_backend_for_now() {
        let result = SymbolicNonlinearProblem::from_strings_with_backend(
            vec!["x^2+y^2-10".to_string(), "x-y-4".to_string()],
            Some(vec!["x".to_string(), "y".to_string()]),
            None,
            None,
            SymbolicBackendConfig::aot(),
        );

        match result {
            Err(SolveError::InvalidConfig(message)) => {
                assert!(message.contains("AOT backend"));
            }
            Err(other) => panic!("expected InvalidConfig, got {other:?}"),
            Ok(_) => panic!("AOT backend is not wired yet"),
        }
    }

    #[test]
    fn symbolic_problem_with_parameters_keeps_backend_and_evaluates() {
        let symbolic = Expr::Symbols("x, y, a");
        let x = symbolic[0].clone();
        let y = symbolic[1].clone();
        let a = symbolic[2].clone();
        let problem = SymbolicNonlinearProblem::from_expressions_with_options(
            vec![a.clone() * x.clone() + y.clone() - Expr::Const(3.0), x - y],
            SymbolicProblemOptions::new()
                .with_variables(vec!["x".to_string(), "y".to_string()])
                .with_equation_parameters(vec!["a".to_string()])
                .with_equation_parameter_values(DVector::from_vec(vec![2.0])),
        )
        .expect("parameterized symbolic problem should build");

        assert_eq!(problem.backend_kind(), SymbolicBackendKind::Lambdify);

        let x0 = DVector::from_vec(vec![1.0, 1.0]);
        let residual = problem.residual(&x0).expect("residual");
        let jacobian = problem.jacobian(&x0).expect("jacobian");

        assert_relative_eq!(residual[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(residual[1], 0.0, epsilon = 1e-12);
        assert_relative_eq!(jacobian[(0, 0)], 2.0, epsilon = 1e-12);
        assert_relative_eq!(jacobian[(0, 1)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(jacobian[(1, 0)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(jacobian[(1, 1)], -1.0, epsilon = 1e-12);
    }

    #[test]
    fn parameter_schema_preserves_order_and_supports_index_lookup() {
        let schema = NonlinearParameterSchema::new(vec!["beta".to_string(), "alpha".to_string()])
            .expect("schema");

        assert_eq!(schema.names(), ["beta", "alpha"]);
        assert_eq!(schema.index_of("beta"), Some(0));
        assert_eq!(schema.index_of("alpha"), Some(1));
        assert_eq!(schema.index_of("gamma"), None);

        let values = NonlinearParameterValues::new(&schema, DVector::from_vec(vec![2.5, -4.0]))
            .expect("values");
        assert_eq!(values.get(0), Some(2.5));
        assert_eq!(values.get(1), Some(-4.0));
    }

    #[test]
    fn parameter_update_changes_residual_and_jacobian_without_rebuilding() {
        let mut problem = SymbolicNonlinearProblem::from_strings_with_options(
            vec!["a*x+y-3".to_string(), "x-y".to_string()],
            SymbolicProblemOptions::new()
                .with_variables(vec!["x".to_string(), "y".to_string()])
                .with_equation_parameters(vec!["a".to_string()])
                .with_equation_parameter_values(DVector::from_vec(vec![2.0])),
        )
        .expect("parameterized problem");
        let x = DVector::from_vec(vec![1.0, 1.0]);

        let first_residual = problem.residual(&x).expect("first residual");
        let first_jacobian = problem.jacobian(&x).expect("first jacobian");
        assert_relative_eq!(first_residual[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(first_jacobian[(0, 0)], 2.0, epsilon = 1e-12);

        let schema_before = problem.parameter_schema().cloned().expect("schema");
        let symbolic_jacobian_before = problem.symbolic_jacobian().to_vec();
        let prepared_before = problem.prepare_dense_aot_problem(SymbolicDenseAotOptions::default());
        let artifact_key_before = prepared_before.problem_key().to_string();
        let input_names_before = prepared_before.manifest().io.input_names.clone();

        problem
            .set_parameter_values(DVector::from_vec(vec![4.0]))
            .expect("parameter update");
        let second_residual = problem.residual(&x).expect("second residual");
        let second_jacobian = problem.jacobian(&x).expect("second jacobian");

        assert_relative_eq!(second_residual[0], 2.0, epsilon = 1e-12);
        assert_relative_eq!(second_jacobian[(0, 0)], 4.0, epsilon = 1e-12);
        assert_eq!(problem.parameter_schema(), Some(&schema_before));
        assert_eq!(
            problem.symbolic_jacobian(),
            symbolic_jacobian_before.as_slice()
        );
        assert_eq!(
            problem.parameter_values().expect("bound values").as_slice(),
            &[4.0]
        );

        let prepared_after = problem.prepare_dense_aot_problem(SymbolicDenseAotOptions::default());
        assert_eq!(prepared_after.problem_key(), artifact_key_before);
        assert_eq!(prepared_after.manifest().io.input_names, input_names_before);
    }

    #[test]
    fn invalid_parameter_update_is_atomic_and_rejects_non_finite_values() {
        let mut problem = SymbolicNonlinearProblem::from_strings_with_options(
            vec!["a*x+y-3".to_string(), "x-y".to_string()],
            SymbolicProblemOptions::new()
                .with_variables(vec!["x".to_string(), "y".to_string()])
                .with_equation_parameters(vec!["a".to_string()])
                .with_equation_parameter_values(DVector::from_vec(vec![2.0])),
        )
        .expect("parameterized problem");

        let dimension_error = problem
            .set_parameter_values(DVector::from_vec(vec![1.0, 2.0]))
            .expect_err("wrong parameter length");
        assert!(matches!(
            dimension_error,
            SolveError::DimensionMismatch {
                context: "nonlinear parameter values",
                ..
            }
        ));
        assert_eq!(
            problem.parameter_values().expect("old values").as_slice(),
            &[2.0]
        );

        let finite_error = problem
            .set_parameter_values(DVector::from_vec(vec![f64::NAN]))
            .expect_err("non-finite parameter");
        assert!(matches!(
            finite_error,
            SolveError::NonFiniteParameterValue { index: 0, .. }
        ));
        assert_eq!(
            problem.parameter_values().expect("old values").as_slice(),
            &[2.0]
        );
    }

    #[test]
    fn parameter_schema_is_required_for_parameter_values_and_validated_at_build() {
        let duplicate = NonlinearParameterSchema::new(vec!["a".to_string(), "a".to_string()])
            .expect_err("duplicate schema should fail");
        assert!(matches!(duplicate, SolveError::InvalidParameterSchema(_)));

        let collision = SymbolicNonlinearProblem::from_strings_with_options(
            vec!["a".to_string()],
            SymbolicProblemOptions::new()
                .with_variables(vec!["a".to_string()])
                .with_equation_parameters(vec!["a".to_string()])
                .with_equation_parameter_values(DVector::from_vec(vec![1.0])),
        )
        .err()
        .expect("variable/parameter collision should fail");
        assert!(matches!(collision, SolveError::InvalidParameterSchema(_)));

        let no_schema = SymbolicNonlinearProblem::from_strings_with_options(
            vec!["x".to_string()],
            SymbolicProblemOptions::new()
                .with_variables(vec!["x".to_string()])
                .with_equation_parameter_values(DVector::from_vec(vec![1.0])),
        )
        .err()
        .expect("values without schema should fail");
        assert!(matches!(no_schema, SolveError::InvalidParameterSchema(_)));

        let mut unbound = SymbolicNonlinearProblem::from_strings_with_options(
            vec!["a*x".to_string()],
            SymbolicProblemOptions::new()
                .with_variables(vec!["x".to_string()])
                .with_equation_parameters(vec!["a".to_string()]),
        )
        .expect("a parameterized problem may be prepared before binding");
        let unbound_error = unbound
            .residual(&DVector::from_vec(vec![2.0]))
            .expect_err("unbound parameters must not use a non-parameter closure");
        assert!(matches!(
            unbound_error,
            SolveError::InvalidParameterSchema(_)
        ));

        let undeclared = SymbolicNonlinearProblem::from_strings_with_options(
            vec!["x+q".to_string()],
            SymbolicProblemOptions::new().with_variables(vec!["x".to_string()]),
        )
        .err()
        .expect("undeclared symbols should fail validation");
        assert!(matches!(
            undeclared,
            SolveError::UndeclaredSymbol {
                equation_index: 0,
                name
            } if name == "q"
        ));

        let duplicate_variables = SymbolicNonlinearProblem::from_strings_with_options(
            vec!["x".to_string(), "x".to_string()],
            SymbolicProblemOptions::new().with_variables(vec!["x".to_string(), "x".to_string()]),
        )
        .err()
        .expect("duplicate variables should fail validation");
        assert!(matches!(
            duplicate_variables,
            SolveError::InvalidVariableSchema(_)
        ));

        let empty_variable = SymbolicNonlinearProblem::from_strings_with_options(
            vec!["x".to_string()],
            SymbolicProblemOptions::new().with_variables(vec![" ".to_string()]),
        )
        .err()
        .expect("empty variables should fail validation");
        assert!(matches!(
            empty_variable,
            SolveError::InvalidVariableSchema(_)
        ));
    }

    #[test]
    fn parameters_are_excluded_when_variables_are_inferred() {
        let problem = SymbolicNonlinearProblem::from_strings_with_options(
            vec!["a*x+y-3".to_string(), "x-y".to_string()],
            SymbolicProblemOptions::new()
                .with_equation_parameters(vec!["a".to_string()])
                .with_equation_parameter_values(DVector::from_vec(vec![2.0])),
        )
        .expect("parameterized problem with inferred variables");

        assert_eq!(problem.variables(), ["x", "y"]);
        assert_eq!(
            problem
                .residual(&DVector::from_vec(vec![1.0, 1.0]))
                .expect("residual")[0],
            0.0
        );
    }

    fn central_difference_jacobian(
        problem: &SymbolicNonlinearProblem,
        x: &DVector<f64>,
        step: f64,
    ) -> DMatrix<f64> {
        let baseline = problem.dimension();
        let mut jacobian = DMatrix::zeros(baseline, baseline);
        for column in 0..baseline {
            let mut plus = x.clone();
            let mut minus = x.clone();
            plus[column] += step;
            minus[column] -= step;
            let f_plus = problem.residual(&plus).expect("plus residual");
            let f_minus = problem.residual(&minus).expect("minus residual");
            for row in 0..baseline {
                jacobian[(row, column)] = (f_plus[row] - f_minus[row]) / (2.0 * step);
            }
        }
        jacobian
    }

    #[test]
    fn symbolic_jacobian_matches_central_difference_for_multiple_parameter_scales() {
        let mut problem = SymbolicNonlinearProblem::from_strings_with_options(
            vec!["a*sin(x)+exp(b*y)-c".to_string(), "x*y+a*x^2-b".to_string()],
            SymbolicProblemOptions::new()
                .with_variables(vec!["x".to_string(), "y".to_string()])
                .with_equation_parameters(vec!["a".to_string(), "b".to_string(), "c".to_string()])
                .with_equation_parameter_values(DVector::from_vec(vec![1.3, -0.4, 0.7])),
        )
        .expect("parameterized nonlinear problem");

        let cases = [
            (DVector::from_vec(vec![0.2, -0.3]), 1e-6),
            (DVector::from_vec(vec![0.01, -0.0002]), 1e-6),
        ];
        let parameter_sets = [
            DVector::from_vec(vec![1.3, -0.4, 0.7]),
            DVector::from_vec(vec![1e-3, 2e3, 0.5]),
        ];

        for ((x, step), parameters) in cases.into_iter().zip(parameter_sets) {
            problem
                .set_parameter_values(parameters)
                .expect("parameter update");
            let symbolic = problem.jacobian(&x).expect("symbolic jacobian");
            let finite_difference = central_difference_jacobian(&problem, &x, step);
            for row in 0..symbolic.nrows() {
                for column in 0..symbolic.ncols() {
                    let difference =
                        (symbolic[(row, column)] - finite_difference[(row, column)]).abs();
                    let scale = 1.0
                        + symbolic[(row, column)].abs()
                        + finite_difference[(row, column)].abs();
                    assert!(
                        difference <= 2e-5 * scale,
                        "Jacobian mismatch at ({row}, {column}): symbolic={} finite_difference={} diff={difference}",
                        symbolic[(row, column)],
                        finite_difference[(row, column)]
                    );
                }
            }
        }
    }

    #[test]
    fn symbolic_problem_prepares_dense_aot_bridge_with_variable_input_order() {
        let problem = elementary_problem_with_options();
        let prepared = problem.prepare_dense_aot_problem(SymbolicDenseAotOptions::default());

        assert_eq!(prepared.flattened_input_names(), &["x", "y"]);
        assert_eq!(prepared.residual_len(), 2);
        assert_eq!(prepared.jacobian_shape(), (2, 2));
        assert_eq!(
            prepared.as_prepared_problem().backend_kind,
            BackendKind::Aot
        );
        assert_eq!(
            prepared.as_prepared_problem().matrix_backend,
            MatrixBackend::Dense
        );
    }

    #[test]
    fn symbolic_problem_prepares_dense_aot_bridge_with_parameter_first_order() {
        let symbolic = Expr::Symbols("x, y, a");
        let x = symbolic[0].clone();
        let y = symbolic[1].clone();
        let a = symbolic[2].clone();
        let problem = SymbolicNonlinearProblem::from_expressions_with_options(
            vec![a.clone() * x.clone() + y.clone() - Expr::Const(3.0), x - y],
            SymbolicProblemOptions::new()
                .with_variables(vec!["x".to_string(), "y".to_string()])
                .with_equation_parameters(vec!["a".to_string()])
                .with_equation_parameter_values(DVector::from_vec(vec![2.0])),
        )
        .expect("parameterized symbolic problem should build");

        let prepared = problem.prepare_dense_aot_problem(SymbolicDenseAotOptions::default());
        assert_eq!(prepared.flattened_input_names(), &["a", "x", "y"]);
        assert!(!prepared.problem_key().is_empty());
        assert_eq!(prepared.manifest().io.input_names, vec!["a", "x", "y"]);
    }

    #[test]
    fn symbolic_problem_solves_from_strings() {
        let result = SolverEngine::new(
            crate::numerical::Nonlinear_systems::engine::NewtonMethod,
            SolveOptions::default(),
        )
        .solve(&elementary_problem(), DVector::from_vec(vec![1.0, 1.0]))
        .expect("solve should succeed");
        assert_relative_eq!(result.x[0], 3.0, epsilon = 1e-8);
        assert_relative_eq!(result.x[1], -1.0, epsilon = 1e-8);
    }

    #[test]
    fn various_nonlinear_equations_simple() {
        let problem = SymbolicNonlinearProblem::from_strings(
            vec!["x+y-100".to_string(), "1/x - 1/y - 1/200".to_string()],
            Some(vec!["x".to_string(), "y".to_string()]),
            None,
            None,
        )
        .expect("problem should build");
        let result = SolverEngine::new(
            crate::numerical::Nonlinear_systems::engine::NewtonMethod,
            SolveOptions::default(),
        )
        .solve(&problem, DVector::from_vec(vec![1.0, 1.0]))
        .expect("solve should succeed");
        let x = -50.0 * (f64::sqrt(17.0) - 5.0);
        let y = 50.0 * (f64::sqrt(17.0) - 3.0);
        assert_relative_eq!(result.x[0], x, epsilon = 1e-3);
        assert_relative_eq!(result.x[1], y, epsilon = 1e-3);
    }

    #[test]
    fn symbolic_damped_method_respects_bounds() {
        let solution = solve_elementary_with_bounds(DampedNewtonMethod::default());
        assert_relative_eq!(solution[0], 3.0, epsilon = 1e-6);
        assert_relative_eq!(solution[1], -1.0, epsilon = 1e-6);
    }

    #[test]
    fn symbolic_lm_method_solves_elementary_problem() {
        let solution = solve_elementary_with_bounds(LevenbergMarquardtMethod::default());
        assert_relative_eq!(solution[0], 3.0, epsilon = 1e-6);
        assert_relative_eq!(solution[1], -1.0, epsilon = 1e-6);
    }

    #[test]
    fn symbolic_trust_region_method_solves_elementary_problem() {
        let solution = solve_elementary_with_bounds(TrustRegionMethod::default());
        assert_relative_eq!(solution[0], 3.0, epsilon = 1e-6);
        assert_relative_eq!(solution[1], -1.0, epsilon = 1e-6);
    }

    #[test]
    fn chemistry_problem_solves_with_lm() {
        let (problem, variables) = chemistry_problem(8.314 * 450e4);
        let bounds = Bounds::new(vec![
            (1e-40, 2.0),
            (1e-40, 2.0),
            (1e-40, 2.0),
            (1e-40, 10.0),
            (-1e-1, 1e-2),
            (-1e-1, 1e-2),
        ])
        .expect("bounds");
        let options = SolveOptions {
            tolerance: 2.0e-3,
            max_iterations: 130,
            bounds: Some(bounds),
            ..SolveOptions::default()
        };
        let method = LevenbergMarquardtMethod {
            increase_factor: 11.0,
            decrease_factor: 9.0,
            ..LevenbergMarquardtMethod::default()
        };
        let result = SolverEngine::new(method, options)
            .solve(
                &problem,
                DVector::from_vec(vec![0.9, 0.9, 0.9, 0.6, 0.0, 0.0]),
            )
            .expect("solve should succeed");
        assert_chemistry_constraints(&result.x, &variables, 8e-3);
    }

    #[test]
    fn chemistry_problem_solves_with_nielsen_lm() {
        let (problem, variables) = chemistry_problem(8.314 * 60e5);
        let bounds = Bounds::new(vec![
            (1e-40, 2.0),
            (1e-40, 2.0),
            (1e-40, 2.0),
            (1e-40, 10.0),
            (-10000.0, 1e6),
            (-100000.0, 1e6),
        ])
        .expect("bounds");
        let options = SolveOptions {
            tolerance: 2.0e-3,
            max_iterations: 130,
            bounds: Some(bounds),
            ..SolveOptions::default()
        };
        let result = SolverEngine::new(NielsenLevenbergMarquardtMethod::default(), options)
            .solve(
                &problem,
                DVector::from_vec(vec![0.9, 0.9, 0.9, 0.6, 0.0, 0.0]),
            )
            .expect("solve should succeed");
        assert_chemistry_constraints(&result.x, &variables, 8e-3);
    }
}
