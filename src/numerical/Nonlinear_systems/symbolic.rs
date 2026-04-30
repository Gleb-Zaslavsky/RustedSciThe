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
use crate::symbolic::symbolic_functions::Jacobian;
use nalgebra::{DMatrix, DVector};

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

/// High-level symbolic backend selection for nonlinear systems.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SymbolicBackendConfig {
    /// Backend used to prepare the symbolic residual/Jacobian provider.
    pub kind: SymbolicBackendKind,
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
trait SymbolicEvaluationBackend {
    /// Returns the backend kind that produced this provider.
    fn kind(&self) -> SymbolicBackendKind;

    /// Evaluates the residual vector at `x`.
    fn residual(
        &self,
        x: &DVector<f64>,
        equation_parameters: Option<&[String]>,
        equation_parameter_values: Option<&DVector<f64>>,
        variables: &[String],
    ) -> Result<DVector<f64>, SolveError>;

    /// Evaluates the Jacobian matrix at `x`.
    fn jacobian(
        &self,
        x: &DVector<f64>,
        equation_parameters: Option<&[String]>,
        equation_parameter_values: Option<&DVector<f64>>,
        variables: &[String],
    ) -> Result<DMatrix<f64>, SolveError>;
}

/// Concrete backend implementation that uses the existing `Jacobian` lambdify path.
struct LambdifySymbolicBackend {
    /// Legacy symbolic Jacobian object kept as the callable backend implementation.
    jacobian: Jacobian,
}

impl LambdifySymbolicBackend {
    /// Builds the lambdify backend from symbolic equations.
    ///
    /// This is the legacy-backed execution branch: symbolic equations are
    /// differentiated through [`Jacobian`] and then lambdified into callable
    /// dense residual/Jacobian closures.
    fn from_expressions(
        equations: &[Expr],
        variables: &[String],
        equation_parameters: Option<&[String]>,
    ) -> Result<Self, SolveError> {
        let mut jacobian = Jacobian::new();
        let variable_refs = variables
            .iter()
            .map(|value| value.as_str())
            .collect::<Vec<_>>();
        jacobian.set_vector_of_functions(equations.to_vec());
        jacobian.set_variables(variable_refs);

        if let Some(parameters) = equation_parameters {
            jacobian.set_params(parameters.to_vec());
            jacobian.calc_jacobian();
            jacobian.lambdify_jacobian_DMatrix_with_parameters_parallel();
            jacobian.lambdify_vector_funvector_DVector_with_parameters_parallel();
        } else {
            jacobian.calc_jacobian();
            jacobian.lambdify_jacobian_DMatrix_parallel();
            jacobian.lambdify_vector_funvector_DVector();
        }

        Ok(Self { jacobian })
    }

    /// Returns the symbolic dense Jacobian built during backend preparation.
    fn symbolic_jacobian(&self) -> &[Vec<Expr>] {
        &self.jacobian.symbolic_jacobian
    }
}

impl SymbolicEvaluationBackend for LambdifySymbolicBackend {
    fn kind(&self) -> SymbolicBackendKind {
        SymbolicBackendKind::Lambdify
    }

    fn residual(
        &self,
        x: &DVector<f64>,
        equation_parameters: Option<&[String]>,
        equation_parameter_values: Option<&DVector<f64>>,
        variables: &[String],
    ) -> Result<DVector<f64>, SolveError> {
        if x.len() != variables.len() {
            return Err(SolveError::DimensionMismatch {
                expected: variables.len(),
                actual: x.len(),
                context: "symbolic residual input",
            });
        }

        let result = if let Some(parameter_values) = equation_parameter_values {
            if let Some(parameters) = equation_parameters {
                if parameters.len() != parameter_values.len() {
                    return Err(SolveError::DimensionMismatch {
                        expected: parameters.len(),
                        actual: parameter_values.len(),
                        context: "symbolic parameter values",
                    });
                }
            }
            let residual = &self.jacobian.lambdified_function_with_params;
            residual(parameter_values, x)
        } else {
            let residual = &self.jacobian.lambdified_function_DVector;
            residual(x)
        };

        if result.iter().any(|value| !value.is_finite()) {
            return Err(SolveError::ResidualEvaluation(
                "symbolic residual returned NaN or Inf".to_string(),
            ));
        }
        Ok(result)
    }

    fn jacobian(
        &self,
        x: &DVector<f64>,
        equation_parameters: Option<&[String]>,
        equation_parameter_values: Option<&DVector<f64>>,
        variables: &[String],
    ) -> Result<DMatrix<f64>, SolveError> {
        if x.len() != variables.len() {
            return Err(SolveError::DimensionMismatch {
                expected: variables.len(),
                actual: x.len(),
                context: "symbolic jacobian input",
            });
        }

        let result = if let Some(parameter_values) = equation_parameter_values {
            if let Some(parameters) = equation_parameters {
                if parameters.len() != parameter_values.len() {
                    return Err(SolveError::DimensionMismatch {
                        expected: parameters.len(),
                        actual: parameter_values.len(),
                        context: "symbolic parameter values",
                    });
                }
            }
            let jacobian = &self.jacobian.lambdified_jacobian_DMatrix_with_params;
            jacobian(parameter_values, x)
        } else {
            let jacobian = &self.jacobian.lambdified_jacobian_DMatrix;
            jacobian(x)
        };

        if result.iter().any(|value| !value.is_finite()) {
            return Err(SolveError::JacobianEvaluation(
                "symbolic jacobian returned NaN or Inf".to_string(),
            ));
        }
        Ok(result)
    }
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

    /// Recreates the flattened generated-module input layout.
    ///
    /// Parameter values, when present, are placed before the nonlinear
    /// unknowns. This mirrors the manifest/runtime-plan contract used during
    /// dense AOT preparation.
    fn flattened_args(
        &self,
        x: &DVector<f64>,
        equation_parameter_values: Option<&DVector<f64>>,
        variables: &[String],
    ) -> Result<Vec<f64>, SolveError> {
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

        let mut args = Vec::new();
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
            args.extend(values.iter().copied());
        }
        args.extend(x.iter().copied());
        Ok(args)
    }
}

impl SymbolicEvaluationBackend for CompiledDenseAotSymbolicBackend {
    fn kind(&self) -> SymbolicBackendKind {
        SymbolicBackendKind::Aot
    }

    fn residual(
        &self,
        x: &DVector<f64>,
        _equation_parameters: Option<&[String]>,
        equation_parameter_values: Option<&DVector<f64>>,
        variables: &[String],
    ) -> Result<DVector<f64>, SolveError> {
        let args = self.flattened_args(x, equation_parameter_values, variables)?;
        let mut out = vec![0.0; self.linked.residual_len];
        (self.linked.residual_eval)(&args, &mut out);
        if out.iter().any(|value| !value.is_finite()) {
            return Err(SolveError::ResidualEvaluation(
                "compiled nonlinear AOT residual returned NaN or Inf".to_string(),
            ));
        }
        Ok(DVector::from_vec(out))
    }

    fn jacobian(
        &self,
        x: &DVector<f64>,
        _equation_parameters: Option<&[String]>,
        equation_parameter_values: Option<&DVector<f64>>,
        variables: &[String],
    ) -> Result<DMatrix<f64>, SolveError> {
        let args = self.flattened_args(x, equation_parameter_values, variables)?;
        let mut out = vec![0.0; self.linked.shape.0 * self.linked.shape.1];
        (self.linked.jacobian_eval)(&args, &mut out);
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
        config: &SymbolicBackendConfig,
    ) -> Result<Self, SolveError> {
        match config.kind {
            SymbolicBackendKind::Lambdify => Ok(Self {
                backend: Box::new(LambdifySymbolicBackend::from_expressions(
                    equations,
                    variables,
                    equation_parameters,
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
    /// Optional symbolic parameter names.
    equation_parameters: Option<Vec<String>>,
    /// Optional parameter values used during evaluation.
    equation_parameter_values: Option<DVector<f64>>,
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
        Self::from_expressions_with_backend(
            equations,
            options.variables,
            options.equation_parameters,
            options.equation_parameter_values,
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
        let mut problem = Self::from_expressions_with_backend(
            equations,
            options.variables,
            options.equation_parameters,
            options.equation_parameter_values,
            SymbolicBackendConfig::lambdify(),
        )?;

        let selected = select_symbolic_nonlinear_backend(&problem, policy, resolver, aot_options);
        match selected.effective_backend {
            SelectedSymbolicNonlinearBackendKind::Lambdify => Ok(problem),
            SelectedSymbolicNonlinearBackendKind::AotCompiled => {
                let prepared = selected.prepared_aot_problem.as_ref().ok_or_else(|| {
                    SolveError::InvalidConfig(
                        "compiled nonlinear AOT selection did not include prepared problem data"
                            .to_string(),
                    )
                })?;
                let linked =
                    resolve_linked_dense_backend(&prepared.problem_key()).ok_or_else(|| {
                        SolveError::CompiledAotRuntimeUnavailable(format!(
                            "no linked dense nonlinear AOT runtime registered for problem key {}",
                            prepared.problem_key()
                        ))
                    })?;
                problem.backend_config = SymbolicBackendConfig::aot();
                problem.backend = PreparedSymbolicBackend::from_linked_dense(
                    linked,
                    &problem.variables,
                    problem.equation_parameters.as_deref(),
                );
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
        if equations.is_empty() {
            return Err(SolveError::InvalidConfig(
                "equation system must not be empty".to_string(),
            ));
        }

        let variables = match variables {
            Some(variables) => variables,
            None => {
                let mut args = equations
                    .iter()
                    .flat_map(|expr| expr.all_arguments_are_variables())
                    .collect::<Vec<_>>();
                args.sort();
                args.dedup();
                args
            }
        };

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

        let symbolic_jacobian = match backend_config.kind {
            SymbolicBackendKind::Lambdify => LambdifySymbolicBackend::from_expressions(
                &equations,
                &variables,
                equation_parameters.as_deref(),
            )?
            .symbolic_jacobian()
            .to_vec(),
            SymbolicBackendKind::Aot => Vec::new(),
        };

        let backend = PreparedSymbolicBackend::from_expressions(
            &equations,
            &variables,
            equation_parameters.as_deref(),
            &backend_config,
        )?;

        Ok(Self {
            backend_config,
            backend,
            equations,
            symbolic_jacobian,
            variables,
            equation_parameters,
            equation_parameter_values,
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
        Self::from_strings_with_backend(
            equations,
            options.variables,
            options.equation_parameters,
            options.equation_parameter_values,
            options.backend_config,
        )
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

    /// Returns the effective prepared backend kind.
    pub fn backend_kind(&self) -> SymbolicBackendKind {
        self.backend.kind()
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
        let parameter_refs = self
            .equation_parameters
            .as_ref()
            .map(|params| params.iter().map(|name| name.as_str()).collect::<Vec<_>>());
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
        if let Some(parameters) = &self.equation_parameters {
            if parameters.len() != values.len() {
                return Err(SolveError::DimensionMismatch {
                    expected: parameters.len(),
                    actual: values.len(),
                    context: "symbolic equation parameters",
                });
            }
        }
        self.equation_parameter_values = Some(values);
        Ok(())
    }

    /// Shared implementation of residual evaluation.
    fn residual_impl(&self, x: &DVector<f64>) -> Result<DVector<f64>, SolveError> {
        self.backend.residual(
            x,
            self.equation_parameters.as_deref(),
            self.equation_parameter_values.as_ref(),
            &self.variables,
        )
    }

    /// Shared implementation of Jacobian evaluation.
    fn jacobian_impl(&self, x: &DVector<f64>) -> Result<DMatrix<f64>, SolveError> {
        self.backend.jacobian(
            x,
            self.equation_parameters.as_deref(),
            self.equation_parameter_values.as_ref(),
            &self.variables,
        )
    }
}

impl NonlinearProblem for SymbolicNonlinearProblem {
    /// Returns the number of unknowns in the symbolic system.
    fn dimension(&self) -> usize {
        self.variables.len()
    }

    /// Evaluates the symbolic residual vector.
    fn residual(&self, x: &DVector<f64>) -> Result<DVector<f64>, SolveError> {
        self.residual_impl(x)
    }
}

impl JacobianProvider for SymbolicNonlinearProblem {
    /// Evaluates the symbolic Jacobian matrix.
    fn jacobian(&self, x: &DVector<f64>) -> Result<DMatrix<f64>, SolveError> {
        self.jacobian_impl(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::Nonlinear_systems::LM_Nielsen::NielsenLevenbergMarquardtMethod;
    use crate::numerical::Nonlinear_systems::LM_vanilla::LevenbergMarquardtMethod;
    use crate::numerical::Nonlinear_systems::NR_damped::DampedNewtonMethod;
    use crate::numerical::Nonlinear_systems::engine::{SolveOptions, SolverEngine};
    use crate::numerical::Nonlinear_systems::problem::Bounds;
    use crate::numerical::Nonlinear_systems::trust_region::*;
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
