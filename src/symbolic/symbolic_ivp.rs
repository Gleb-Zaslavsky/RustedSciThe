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
use crate::symbolic::symbolic_engine::Expr;
use nalgebra::{DMatrix, DVector};
use std::fmt;
use std::sync::{Arc, RwLock};

/// Shared residual evaluator signature for IVP symbolic backends.
pub type IvpResidualEval = dyn Fn(f64, &DVector<f64>) -> DVector<f64> + Send + Sync;

/// Shared dense Jacobian evaluator signature for IVP symbolic backends.
pub type IvpDenseJacobianEval = dyn Fn(f64, &DVector<f64>) -> DMatrix<f64> + Send + Sync;

/// Shared parameter storage reused by params-aware IVP evaluators.
pub type SharedIvpParameterValues = Arc<RwLock<DVector<f64>>>;

/// Setup/runtime errors for the shared IVP backend layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IvpBackendError {
    /// Parameter names were declared but no initial values were provided.
    MissingParameterValues { expected: usize },
    /// Parameter value vector length does not match declared symbolic names.
    ParameterCountMismatch { expected: usize, actual: usize },
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
}

/// Prepared symbolic IVP residual without compiling any Jacobian callback.
///
/// This is useful for solver paths that provide a native sparse/banded
/// Jacobian evaluator separately and should not pay for a dense Jacobian
/// closure during setup.
pub struct PreparedSymbolicIvpResidualProblem {
    pub residual: Box<IvpResidualEval>,
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
}

impl PreparedSymbolicIvpResidualProblem {
    pub fn parameter_values_handle(&self) -> Option<SharedIvpParameterValues> {
        self.parameter_values_handle.clone()
    }

    fn linked_args(&self, t: f64, y: &DVector<f64>) -> Vec<f64> {
        let parameter_values = self.parameter_values_handle.as_ref().map(|handle| {
            handle
                .read()
                .expect("shared IVP parameter state lock poisoned")
                .clone()
        });
        let mut args = Vec::with_capacity(
            1 + y.len() + parameter_values.as_ref().map_or(0, |values| values.len()),
        );
        args.push(t);
        if let Some(values) = parameter_values.as_ref() {
            args.extend(values.iter().copied());
        }
        args.extend(y.iter().copied());
        args
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
        let metadata = Self {
            residual: Box::new(|_, _| unreachable!("replaced below")),
            equations: self.equations,
            time_arg: self.time_arg,
            variables: self.variables,
            equation_parameters: self.equation_parameters,
            parameter_values_handle,
            backend_kind: IvpBackendKind::Aot,
        };
        let residual_parameter_values_handle = metadata.parameter_values_handle.clone();
        let residual = Box::new(move |t: f64, y: &DVector<f64>| -> DVector<f64> {
            let parameter_values = residual_parameter_values_handle.as_ref().map(|handle| {
                handle
                    .read()
                    .expect("shared IVP parameter state lock poisoned")
                    .clone()
            });
            let mut args = Vec::with_capacity(
                1 + y.len() + parameter_values.as_ref().map_or(0, |values| values.len()),
            );
            args.push(t);
            if let Some(values) = parameter_values.as_ref() {
                args.extend(values.iter().copied());
            }
            args.extend(y.iter().copied());

            let mut out = vec![0.0; residual_len];
            residual_eval(args.as_slice(), out.as_mut_slice());
            DVector::from_vec(out)
        });

        Self {
            residual,
            ..metadata
        }
    }
}

impl PreparedSymbolicIvpProblem {
    fn linked_args(&self, t: f64, y: &DVector<f64>) -> Vec<f64> {
        let parameter_values = self.parameter_values_handle.as_ref().map(|handle| {
            handle
                .read()
                .expect("shared IVP parameter state lock poisoned")
                .clone()
        });
        let mut args = Vec::with_capacity(
            1 + y.len() + parameter_values.as_ref().map_or(0, |values| values.len()),
        );
        args.push(t);
        if let Some(values) = parameter_values.as_ref() {
            args.extend(values.iter().copied());
        }
        args.extend(y.iter().copied());
        args
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
                    .expect("shared IVP parameter state lock poisoned");
                *slot = values;
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

        let residual = Box::new(move |t: f64, y: &DVector<f64>| -> DVector<f64> {
            let parameter_values = residual_parameter_values_handle.as_ref().map(|handle| {
                handle
                    .read()
                    .expect("shared IVP parameter state lock poisoned")
                    .clone()
            });
            let mut args = Vec::with_capacity(
                1 + y.len() + parameter_values.as_ref().map_or(0, |values| values.len()),
            );
            args.push(t);
            if let Some(values) = parameter_values.as_ref() {
                args.extend(values.iter().copied());
            }
            args.extend(y.iter().copied());
            let mut out = vec![0.0; residual_len];
            residual_eval(&args, &mut out);
            DVector::from_vec(out)
        });

        let jacobian = Box::new(move |t: f64, y: &DVector<f64>| -> DMatrix<f64> {
            let parameter_values = jacobian_parameter_values_handle.as_ref().map(|handle| {
                handle
                    .read()
                    .expect("shared IVP parameter state lock poisoned")
                    .clone()
            });
            let mut args = Vec::with_capacity(
                1 + y.len() + parameter_values.as_ref().map_or(0, |values| values.len()),
            );
            args.push(t);
            if let Some(values) = parameter_values.as_ref() {
                args.extend(values.iter().copied());
            }
            args.extend(y.iter().copied());
            let mut out = vec![0.0; rows * cols];
            jacobian_eval(&args, &mut out);
            DMatrix::from_row_slice(rows, cols, out.as_slice())
        });

        Self {
            residual,
            jacobian,
            equations: self.equations,
            symbolic_jacobian: self.symbolic_jacobian,
            time_arg: self.time_arg,
            variables: self.variables,
            equation_parameters: self.equation_parameters,
            parameter_values_handle,
            backend_kind: IvpBackendKind::Aot,
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

fn build_symbolic_jacobian(equations: &[Expr], variables: &[String]) -> Vec<Vec<Expr>> {
    equations
        .iter()
        .map(|expr| {
            variables
                .iter()
                .map(|variable| expr.diff(variable).simplify())
                .collect::<Vec<_>>()
        })
        .collect()
}

fn compile_ivp_residual(
    equations: &[Expr],
    time_arg: &str,
    variables: &[String],
    equation_parameters: Option<&[String]>,
    parameter_values_handle: Option<SharedIvpParameterValues>,
) -> Box<IvpResidualEval> {
    let mut names = Vec::with_capacity(
        1 + variables.len() + equation_parameters.map_or(0, |parameters| parameters.len()),
    );
    names.push(time_arg.to_string());
    if let Some(parameters) = equation_parameters {
        names.extend(parameters.iter().cloned());
    }
    names.extend(variables.iter().cloned());
    let name_refs = names.iter().map(|name| name.as_str()).collect::<Vec<_>>();

    let compiled = equations
        .iter()
        .map(|expr| Expr::lambdify_borrowed_thread_safe(expr, &name_refs))
        .collect::<Vec<_>>();

    Box::new(move |t: f64, y: &DVector<f64>| -> DVector<f64> {
        let parameter_values = parameter_values_handle.as_ref().map(|handle| {
            handle
                .read()
                .expect("shared IVP parameter state lock poisoned")
                .clone()
        });
        let mut args = Vec::with_capacity(
            1 + y.len() + parameter_values.as_ref().map_or(0, |values| values.len()),
        );
        args.push(t);
        if let Some(values) = parameter_values.as_ref() {
            args.extend(values.iter().copied());
        }
        args.extend(y.iter().copied());
        DVector::from_vec(compiled.iter().map(|func| func(&args)).collect())
    })
}

fn compile_ivp_dense_jacobian(
    symbolic_jacobian: &[Vec<Expr>],
    time_arg: &str,
    variables: &[String],
    equation_parameters: Option<&[String]>,
    parameter_values_handle: Option<SharedIvpParameterValues>,
) -> Box<IvpDenseJacobianEval> {
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

    let compiled = symbolic_jacobian
        .iter()
        .flat_map(|row| {
            row.iter()
                .map(|expr| Expr::lambdify_borrowed_thread_safe(expr, &name_refs))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    Box::new(move |t: f64, y: &DVector<f64>| -> DMatrix<f64> {
        let parameter_values = parameter_values_handle.as_ref().map(|handle| {
            handle
                .read()
                .expect("shared IVP parameter state lock poisoned")
                .clone()
        });
        let mut args = Vec::with_capacity(
            1 + y.len() + parameter_values.as_ref().map_or(0, |values| values.len()),
        );
        args.push(t);
        if let Some(values) = parameter_values.as_ref() {
            args.extend(values.iter().copied());
        }
        args.extend(y.iter().copied());
        let values = compiled.iter().map(|func| func(&args)).collect::<Vec<_>>();
        DMatrix::from_row_slice(rows, cols, values.as_slice())
    })
}

/// Prepares one modern shared symbolic IVP backend from equations and options.
pub fn prepare_symbolic_ivp_problem(
    equations: Vec<Expr>,
    variables: Vec<String>,
    time_arg: String,
    options: SymbolicIvpProblemOptions,
) -> Result<PreparedSymbolicIvpProblem, IvpBackendError> {
    let parameter_values_handle = prepare_parameter_values_handle(
        options.equation_parameters.as_deref(),
        options.equation_parameter_values,
    )?;

    let symbolic_jacobian = build_symbolic_jacobian(&equations, &variables);
    let residual = compile_ivp_residual(
        &equations,
        time_arg.as_str(),
        &variables,
        options.equation_parameters.as_deref(),
        parameter_values_handle.clone(),
    );
    let jacobian = compile_ivp_dense_jacobian(
        &symbolic_jacobian,
        time_arg.as_str(),
        &variables,
        options.equation_parameters.as_deref(),
        parameter_values_handle.clone(),
    );

    let backend_kind = match options.backend_policy {
        IvpBackendSelectionPolicy::LambdifyOnly => IvpBackendKind::Lambdify,
        IvpBackendSelectionPolicy::PreferAotThenLambdify => IvpBackendKind::Lambdify,
    };

    Ok(PreparedSymbolicIvpProblem {
        residual,
        jacobian,
        equations,
        symbolic_jacobian,
        time_arg,
        variables,
        equation_parameters: options.equation_parameters,
        parameter_values_handle,
        backend_kind,
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
    let parameter_values_handle = prepare_parameter_values_handle(
        options.equation_parameters.as_deref(),
        options.equation_parameter_values,
    )?;
    let residual = compile_ivp_residual(
        &equations,
        time_arg.as_str(),
        &variables,
        options.equation_parameters.as_deref(),
        parameter_values_handle.clone(),
    );

    Ok(PreparedSymbolicIvpResidualProblem {
        residual,
        equations,
        time_arg,
        variables,
        equation_parameters: options.equation_parameters,
        parameter_values_handle,
        backend_kind: IvpBackendKind::Lambdify,
    })
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
