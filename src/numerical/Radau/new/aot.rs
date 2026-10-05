//! Radau's thin adapter over the shared generated-IVP AOT lifecycle.
//!
//! The compiler, cache, resolver, dynamic-library lifetime, and generated ABI
//! remain in `symbolic_ivp_generated`.  This module owns only Radau concerns:
//! selected matrix layout, caller-owned callback workspace, typed error
//! translation, and the structural Jacobian pattern consumed by the numerical
//! backend.  Keeping that boundary narrow prevents a second AOT compiler from
//! growing inside the solver.

use crate::symbolic::codegen::codegen_aot_resolution::AotResolver;
use crate::symbolic::codegen::codegen_aot_runtime_link::LinkedSparseAotBackend;
use crate::symbolic::codegen::codegen_runtime_api::{
    recommended_dense_jacobian_chunking_for_auto_parallelism,
    recommended_dense_jacobian_chunking_for_parallelism,
    recommended_residual_chunking_for_auto_parallelism,
    recommended_residual_chunking_for_parallelism,
};
use crate::symbolic::codegen::codegen_tasks::{BandedChunkingStrategy, SparseChunkingStrategy};
use crate::symbolic::ivp_telemetry::IvpLambdifyExecutionPolicy;
use crate::symbolic::ivp_telemetry::IvpTelemetry;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp::{
    IvpSymbolicAssemblyBackend, PreparedSymbolicIvpProblem, SymbolicIvpProblemOptions,
};
use crate::symbolic::symbolic_ivp_generated::{
    SelectedSymbolicIvpBackendKind, SymbolicIvpGeneratedBackendConfig,
    prepare_generated_symbolic_ivp_banded_backend, prepare_generated_symbolic_ivp_problem,
    prepare_generated_symbolic_ivp_sparse_backend,
};
use nalgebra::DVector;

use super::config::{RadauAssembly, RadauMatrixLayout};
use super::error::{RadauError, RadauStage};
use super::telemetry::{RadauFrontendStage, RadauTelemetry, RadauTelemetryMode};

/// Per-session storage for the generated callback ABI.
///
/// The shared linked callbacks accept `[time, parameters..., state...]` and
/// write into caller-owned output.  The vector is allocated once for a solve
/// session and then reused for every residual/Jacobian call.
pub(crate) struct AotWorkspace {
    pub(crate) args: Vec<f64>,
    pub(crate) shared_telemetry: IvpTelemetry,
}

impl AotWorkspace {
    fn new(
        argument_count: usize,
        mode: RadauTelemetryMode,
        policy: IvpLambdifyExecutionPolicy,
    ) -> Self {
        let shared_telemetry = match mode {
            RadauTelemetryMode::Off => IvpTelemetry::disabled(),
            RadauTelemetryMode::Counters => IvpTelemetry::counters(),
            RadauTelemetryMode::Timings => IvpTelemetry::detailed(),
        };
        shared_telemetry.set_lambdify_execution_policy(policy);
        Self {
            args: vec![0.0; argument_count],
            shared_telemetry,
        }
    }

    pub(crate) fn argument_capacity(&self) -> usize {
        self.args.capacity()
    }
}

/// Prepared AOT route selected once before integration starts.
pub(crate) enum AotPlan {
    Dense {
        problem: PreparedSymbolicIvpProblem,
        dimension: usize,
        parameter_count: usize,
        execution_policy: IvpLambdifyExecutionPolicy,
    },
    Structured {
        backend: LinkedSparseAotBackend,
        layout: RadauMatrixLayout,
        pattern: Vec<(usize, usize)>,
        dimension: usize,
        parameter_count: usize,
        execution_policy: IvpLambdifyExecutionPolicy,
    },
}

/// Prepared route plus the resolver snapshot published by the shared AOT
/// lifecycle. The snapshot is returned to the public solver so a producer can
/// hand its provenance to a later `RequirePrebuilt` consumer.
pub(crate) struct PreparedAotPlan {
    pub(crate) plan: AotPlan,
    pub(crate) updated_resolver: Option<AotResolver>,
}

impl AotPlan {
    /// Prepare exactly the requested AOT route.
    ///
    /// A missing artifact or unavailable linked runtime is an error.  Falling
    /// back to Lambdify here would make the public `Aot` selection dishonest
    /// and would invalidate lifecycle telemetry.
    pub(crate) fn prepare(
        assembly: RadauAssembly,
        layout: RadauMatrixLayout,
        residual: Vec<Expr>,
        jacobian: Option<Vec<Expr>>,
        independent_variable: String,
        variables: Vec<String>,
        parameters: Vec<String>,
        config: SymbolicIvpGeneratedBackendConfig,
        telemetry: &mut RadauTelemetry,
    ) -> Result<PreparedAotPlan, RadauError> {
        Self::prepare_with_policy(
            assembly,
            layout,
            residual,
            jacobian,
            independent_variable,
            variables,
            parameters,
            config,
            telemetry,
            IvpLambdifyExecutionPolicy::Sequential,
        )
    }

    /// Prepare an AOT route with the shared Sequential/Parallel/Auto policy.
    pub(crate) fn prepare_with_policy(
        assembly: RadauAssembly,
        layout: RadauMatrixLayout,
        residual: Vec<Expr>,
        jacobian: Option<Vec<Expr>>,
        independent_variable: String,
        variables: Vec<String>,
        parameters: Vec<String>,
        mut config: SymbolicIvpGeneratedBackendConfig,
        telemetry: &mut RadauTelemetry,
        execution_policy: IvpLambdifyExecutionPolicy,
    ) -> Result<PreparedAotPlan, RadauError> {
        let dimension = residual.len();
        if dimension == 0 {
            return Err(super::error::RadauConfigError::ZeroDimension.into());
        }
        let parameter_count = parameters.len();
        let explicit_jacobian = explicit_jacobian_rows(jacobian, dimension, variables.len())?;
        let symbolic_backend = match assembly {
            RadauAssembly::ExprLegacy => IvpSymbolicAssemblyBackend::ExprLegacy,
            RadauAssembly::AtomViewNative => IvpSymbolicAssemblyBackend::AtomView,
        };
        let options = SymbolicIvpProblemOptions::new()
            .with_symbolic_assembly_backend(symbolic_backend)
            .with_equation_parameters(parameters)
            // Shared IVP preparation validates the parameter schema eagerly.
            // Runtime callbacks still receive the current values through the
            // generated ABI; this vector is only the compile-time seed.
            .with_equation_parameter_values(DVector::from_element(parameter_count, 1.0))
            .with_lambdify_execution_policy(execution_policy);
        let options = match explicit_jacobian {
            Some(jacobian) => options.with_explicit_jacobian(jacobian),
            None => options,
        }
        .with_telemetry(match telemetry.mode() {
            RadauTelemetryMode::Off => IvpTelemetry::disabled(),
            RadauTelemetryMode::Counters => IvpTelemetry::counters(),
            RadauTelemetryMode::Timings => IvpTelemetry::detailed(),
        });

        configure_chunking(&mut config, layout, dimension, execution_policy);
        let result = match layout {
            RadauMatrixLayout::Dense => telemetry
                .measure_frontend(RadauFrontendStage::AotPreparation, || {
                    prepare_generated_symbolic_ivp_problem(
                        residual,
                        variables,
                        independent_variable,
                        options,
                        config,
                    )
                })
                .map_err(|error| aot_lifecycle_error(error.to_string()))?,
            RadauMatrixLayout::Sparse => {
                let generated = telemetry
                    .measure_frontend(RadauFrontendStage::AotPreparation, || {
                        prepare_generated_symbolic_ivp_sparse_backend(
                            residual,
                            variables,
                            independent_variable,
                            options,
                            config,
                        )
                    })
                    .map_err(|error| aot_lifecycle_error(error.to_string()))?;
                return Self::from_structured_result(
                    generated,
                    layout,
                    dimension,
                    parameter_count,
                    execution_policy,
                    telemetry,
                );
            }
            RadauMatrixLayout::Banded { lower, upper } => {
                let generated = telemetry
                    .measure_frontend(RadauFrontendStage::AotPreparation, || {
                        prepare_generated_symbolic_ivp_banded_backend(
                            residual,
                            variables,
                            independent_variable,
                            (lower, upper),
                            options,
                            config,
                        )
                    })
                    .map_err(|error| aot_lifecycle_error(error.to_string()))?;
                return Self::from_structured_result(
                    generated,
                    layout,
                    dimension,
                    parameter_count,
                    execution_policy,
                    telemetry,
                );
            }
        };

        if result.selected_backend != SelectedSymbolicIvpBackendKind::AotCompiled {
            return Err(aot_lifecycle_error(
                "generated dense route did not publish a compiled runtime",
            ));
        }
        let updated_resolver = result.updated_resolver.clone();
        let problem = result.into_problem();
        telemetry.absorb_ivp_preparation(&problem.telemetry.snapshot());
        Ok(PreparedAotPlan {
            plan: Self::Dense {
                problem,
                dimension,
                parameter_count,
                execution_policy,
            },
            updated_resolver,
        })
    }

    fn from_structured_result(
        result: crate::symbolic::symbolic_ivp_generated::PreparedGeneratedSymbolicIvpSparseBackend,
        layout: RadauMatrixLayout,
        dimension: usize,
        parameter_count: usize,
        execution_policy: IvpLambdifyExecutionPolicy,
        telemetry: &mut RadauTelemetry,
    ) -> Result<PreparedAotPlan, RadauError> {
        if result.selected_backend != SelectedSymbolicIvpBackendKind::AotCompiled {
            return Err(aot_lifecycle_error(
                "generated structured route did not publish a compiled runtime",
            ));
        }
        let backend = result.linked_backend.ok_or_else(|| {
            aot_lifecycle_error("generated structured route has no linked callback")
        })?;
        let updated_resolver = result.updated_resolver.clone();
        let pattern = match layout {
            RadauMatrixLayout::Sparse => result
                .jacobian_structure
                .row_indices
                .into_iter()
                .zip(result.jacobian_structure.col_indices)
                .collect(),
            RadauMatrixLayout::Banded { lower, upper } => banded_pattern(dimension, lower, upper),
            RadauMatrixLayout::Dense => unreachable!("dense route is handled separately"),
        };
        telemetry.absorb_ivp_preparation(&result.telemetry.snapshot());
        Ok(PreparedAotPlan {
            plan: Self::Structured {
                backend,
                layout,
                pattern,
                dimension,
                parameter_count,
                execution_policy,
            },
            updated_resolver,
        })
    }

    pub(crate) fn workspace(&self) -> AotWorkspace {
        self.workspace_with_telemetry(RadauTelemetryMode::Off)
    }

    pub(crate) fn workspace_with_telemetry(&self, mode: RadauTelemetryMode) -> AotWorkspace {
        AotWorkspace::new(
            1 + self.parameter_count() + self.dimension(),
            mode,
            self.execution_policy(),
        )
    }

    pub(crate) fn dimension(&self) -> usize {
        match self {
            Self::Dense { dimension, .. } | Self::Structured { dimension, .. } => *dimension,
        }
    }

    pub(crate) fn parameter_count(&self) -> usize {
        match self {
            Self::Dense {
                parameter_count, ..
            }
            | Self::Structured {
                parameter_count, ..
            } => *parameter_count,
        }
    }

    pub(crate) fn execution_policy(&self) -> IvpLambdifyExecutionPolicy {
        match self {
            Self::Dense {
                execution_policy, ..
            }
            | Self::Structured {
                execution_policy, ..
            } => *execution_policy,
        }
    }

    /// Import the shared generated runtime counters for this session. Dense
    /// routes own their telemetry inside `PreparedSymbolicIvpProblem`; the
    /// structured routes use the session workspace stream passed to callbacks.
    pub(crate) fn absorb_runtime_telemetry(
        &self,
        workspace: &AotWorkspace,
        telemetry: &mut RadauTelemetry,
    ) {
        let snapshot = match self {
            Self::Dense { problem, .. } => problem.telemetry.snapshot(),
            Self::Structured { .. } => workspace.shared_telemetry.snapshot(),
        };
        telemetry.absorb_ivp_runtime(&snapshot);
    }

    pub(crate) fn jacobian_pattern(&self) -> &[(usize, usize)] {
        match self {
            Self::Dense { problem, .. } => {
                // Dense numerical storage has a complete shape; symbolic zero
                // entries are still valid matrix slots and need no pattern.
                // Returning an empty pattern keeps structured-only callers
                // from accidentally treating dense storage as sparse.
                let _ = problem;
                &[]
            }
            Self::Structured { pattern, .. } => pattern,
        }
    }

    pub(crate) fn layout(&self) -> RadauMatrixLayout {
        match self {
            Self::Dense { .. } => RadauMatrixLayout::Dense,
            Self::Structured { layout, .. } => *layout,
        }
    }

    pub(crate) fn evaluate_residual(
        &self,
        t: f64,
        state: &[f64],
        parameters: &[f64],
        output: &mut [f64],
        workspace: &mut AotWorkspace,
        telemetry: &mut RadauTelemetry,
    ) -> Result<(), RadauError> {
        self.validate_inputs(
            state,
            parameters,
            output.len(),
            self.dimension(),
            RadauStage::Residual,
        )?;
        telemetry.measure_callback(
            super::telemetry::RadauCallbackStage::ArgumentBinding,
            || bind_args(t, state, parameters, workspace),
        );
        let result = telemetry.measure_callback(
            super::telemetry::RadauCallbackStage::ResidualEvaluation,
            || match self {
                Self::Dense { problem, .. } => problem
                    .try_evaluate_aot_residual_parts(
                        t,
                        parameters,
                        state,
                        output,
                        &mut workspace.args,
                    )
                    .map_err(|error| aot_callback_error(RadauStage::Residual, error)),
                Self::Structured { backend, .. } => backend
                    .try_residual_eval_with_policy(
                        &workspace.args,
                        output,
                        self.execution_policy(),
                        &workspace.shared_telemetry,
                    )
                    .map_err(|error| aot_callback_error(RadauStage::Residual, error)),
            },
        );
        if result.is_ok() {
            telemetry.count_output_writes(output.len());
        }
        result
    }

    pub(crate) fn evaluate_jacobian(
        &self,
        t: f64,
        state: &[f64],
        parameters: &[f64],
        layout: RadauMatrixLayout,
        output: &mut [f64],
        workspace: &mut AotWorkspace,
        telemetry: &mut RadauTelemetry,
    ) -> Result<(), RadauError> {
        if self.layout() != layout {
            return Err(aot_lifecycle_error(
                "AOT callback layout differs from Radau configuration",
            ));
        }
        self.validate_inputs(
            state,
            parameters,
            output.len(),
            self.expected_jacobian_len(),
            RadauStage::Jacobian,
        )?;
        telemetry.measure_callback(
            super::telemetry::RadauCallbackStage::ArgumentBinding,
            || bind_args(t, state, parameters, workspace),
        );
        let result = telemetry.measure_callback(
            super::telemetry::RadauCallbackStage::JacobianEvaluation,
            || match self {
                Self::Dense { problem, .. } => problem
                    .try_evaluate_aot_jacobian_parts(
                        t,
                        parameters,
                        state,
                        output,
                        &mut workspace.args,
                    )
                    .map_err(|error| aot_callback_error(RadauStage::Jacobian, error)),
                Self::Structured { backend, .. } => backend
                    .try_jacobian_values_eval_with_policy(
                        &workspace.args,
                        output,
                        self.execution_policy(),
                        &workspace.shared_telemetry,
                    )
                    .map_err(|error| aot_callback_error(RadauStage::Jacobian, error)),
            },
        );
        if result.is_ok() {
            telemetry.count_output_writes(output.len());
        }
        result
    }

    fn expected_jacobian_len(&self) -> usize {
        match self {
            Self::Dense { dimension, .. } => dimension.saturating_mul(*dimension),
            Self::Structured { backend, .. } => backend.jacobian_output_len().unwrap_or(0),
        }
    }

    fn validate_inputs(
        &self,
        state: &[f64],
        parameters: &[f64],
        output_len: usize,
        expected_output: usize,
        stage: RadauStage,
    ) -> Result<(), RadauError> {
        if state.len() != self.dimension() {
            return Err(RadauError::ShapeMismatch {
                stage,
                expected: self.dimension(),
                actual: state.len(),
            });
        }
        if parameters.len() != self.parameter_count() {
            return Err(RadauError::ShapeMismatch {
                stage,
                expected: self.parameter_count(),
                actual: parameters.len(),
            });
        }
        if output_len != expected_output {
            return Err(RadauError::ShapeMismatch {
                stage,
                expected: expected_output,
                actual: output_len,
            });
        }
        Ok(())
    }
}

fn configure_chunking(
    config: &mut SymbolicIvpGeneratedBackendConfig,
    layout: RadauMatrixLayout,
    dimension: usize,
    policy: IvpLambdifyExecutionPolicy,
) {
    let parallel = match policy {
        IvpLambdifyExecutionPolicy::Sequential => return,
        IvpLambdifyExecutionPolicy::Parallel { .. } => true,
        IvpLambdifyExecutionPolicy::Auto { .. } => true,
    };
    if !parallel || dimension == 0 {
        return;
    }
    config.residual_chunking_strategy = match policy {
        IvpLambdifyExecutionPolicy::Parallel { .. } => {
            recommended_residual_chunking_for_parallelism(dimension, 2)
        }
        IvpLambdifyExecutionPolicy::Auto { .. } => {
            recommended_residual_chunking_for_auto_parallelism(dimension)
        }
        IvpLambdifyExecutionPolicy::Sequential => unreachable!(),
    };
    match layout {
        RadauMatrixLayout::Dense => {
            config.aot_options.jacobian_strategy = match policy {
                IvpLambdifyExecutionPolicy::Parallel { .. } => {
                    recommended_dense_jacobian_chunking_for_parallelism(dimension, 2)
                }
                IvpLambdifyExecutionPolicy::Auto { .. } => {
                    recommended_dense_jacobian_chunking_for_auto_parallelism(dimension)
                }
                IvpLambdifyExecutionPolicy::Sequential => unreachable!(),
            };
        }
        RadauMatrixLayout::Sparse | RadauMatrixLayout::Banded { .. } => {
            let workers = std::thread::available_parallelism()
                .map(|value| value.get())
                .unwrap_or(1)
                .max(1);
            let target_chunks = (workers * 2).max(2);
            if matches!(layout, RadauMatrixLayout::Sparse) {
                config.sparse_jacobian_chunking_strategy =
                    SparseChunkingStrategy::ByTargetChunkCount { target_chunks };
            } else {
                config.banded_jacobian_chunking_strategy =
                    BandedChunkingStrategy::ByTargetChunkCount { target_chunks };
            }
        }
    }
}

fn bind_args(t: f64, state: &[f64], parameters: &[f64], workspace: &mut AotWorkspace) {
    workspace.args[0] = t;
    workspace.args[1..1 + parameters.len()].copy_from_slice(parameters);
    workspace.args[1 + parameters.len()..].copy_from_slice(state);
}

fn banded_pattern(dimension: usize, lower: usize, upper: usize) -> Vec<(usize, usize)> {
    let mut pattern = Vec::with_capacity(dimension.saturating_mul(lower + upper + 1));
    for column in 0..dimension {
        let first = column.saturating_sub(upper);
        let last = (column + lower + 1).min(dimension);
        for row in first..last {
            pattern.push((row, column));
        }
    }
    pattern
}

fn aot_lifecycle_error(message: impl Into<String>) -> RadauError {
    RadauError::AotLifecycle {
        message: message.into(),
    }
}

fn explicit_jacobian_rows(
    jacobian: Option<Vec<Expr>>,
    rows: usize,
    cols: usize,
) -> Result<Option<Vec<Vec<Expr>>>, RadauError> {
    let Some(jacobian) = jacobian else {
        return Ok(None);
    };
    let expected = rows
        .checked_mul(cols)
        .ok_or(RadauError::WorkspaceSizeOverflow { dimension: rows })?;
    if jacobian.len() != expected {
        return Err(RadauError::ShapeMismatch {
            stage: RadauStage::Preparation,
            expected,
            actual: jacobian.len(),
        });
    }
    Ok(Some(
        jacobian
            .chunks(cols.max(1))
            .map(ToOwned::to_owned)
            .collect(),
    ))
}

fn aot_callback_error(stage: RadauStage, error: impl std::fmt::Display) -> RadauError {
    RadauError::Callback {
        stage,
        message: error.to_string(),
    }
}
