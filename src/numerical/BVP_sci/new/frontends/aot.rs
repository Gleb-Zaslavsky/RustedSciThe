//! AOT frontend for the second-generation BVP solver.
//!
//! This adapter reuses the crate-wide generated-IVP lifecycle. It does not
//! contain a compiler, cache, dynamic-library loader, or second symbolic
//! differentiation implementation. Its only BVP-specific work is mapping the
//! shared `[x, parameters..., state...]` callback ABI to collocation buffers.

use std::sync::Arc;

use crate::symbolic::bvp::atom_aot::AtomAotBandedSlotMap;
use crate::symbolic::codegen::codegen_aot_resolution::AotResolver;
use crate::symbolic::codegen::codegen_aot_runtime_link::LinkedSparseAotBackend;
use crate::symbolic::codegen::codegen_runtime_api::{
    recommended_dense_jacobian_chunking_for_auto_parallelism,
    recommended_dense_jacobian_chunking_for_parallelism,
    recommended_residual_chunking_for_auto_parallelism,
    recommended_residual_chunking_for_parallelism,
};
use crate::symbolic::codegen::codegen_tasks::{BandedChunkingStrategy, SparseChunkingStrategy};
use crate::symbolic::ivp_telemetry::{IvpLambdifyExecutionPolicy, IvpTelemetry};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp::{
    IvpSymbolicAssemblyBackend, PreparedSymbolicIvpProblem, SymbolicIvpProblemOptions,
};
use crate::symbolic::symbolic_ivp_generated::{
    prepare_generated_symbolic_ivp_banded_backend, prepare_generated_symbolic_ivp_problem,
    prepare_generated_symbolic_ivp_sparse_backend, PreparedGeneratedSymbolicIvpProblem,
    PreparedGeneratedSymbolicIvpSparseBackend, SelectedSymbolicIvpBackendKind,
    SymbolicIvpGeneratedBackendConfig, SymbolicIvpGeneratedError,
};
use nalgebra::DVector;

use super::super::config::{BvpSciAssembly, BvpSciExecutionPolicy, BvpSciMatrixLayout};
use super::super::error::{BvpSciAotFailureKind, BvpSciNewError, BvpSciStage};
use super::super::telemetry::{BvpSciTelemetry, BvpSciTelemetryMode};

/// Prepared generated callback route selected once before numerical solving.
#[derive(Clone)]
pub enum BvpSciAotPlan {
    /// Row-major dense residual and Jacobian callbacks.
    Dense {
        problem: Arc<PreparedSymbolicIvpProblem>,
        assembly: BvpSciAssembly,
        dimension: usize,
        parameter_count: usize,
        execution_policy: IvpLambdifyExecutionPolicy,
        telemetry: BvpSciTelemetry,
        updated_resolver: Option<AotResolver>,
    },
    /// Sparse explicit-value or compact-Banded callback route.
    Structured {
        backend: LinkedSparseAotBackend,
        assembly: BvpSciAssembly,
        layout: BvpSciMatrixLayout,
        pattern: Vec<(usize, usize)>,
        band_slots: Option<AtomAotBandedSlotMap>,
        dimension: usize,
        parameter_count: usize,
        execution_policy: IvpLambdifyExecutionPolicy,
        telemetry: BvpSciTelemetry,
        shared_telemetry: IvpTelemetry,
        updated_resolver: Option<AotResolver>,
    },
}

impl BvpSciAotPlan {
    /// Prepare one exact AOT route using the shared compiler/cache lifecycle.
    /// A selected AOT route never silently falls back to Lambdify.
    pub fn prepare(
        assembly: BvpSciAssembly,
        layout: BvpSciMatrixLayout,
        equations: Vec<Expr>,
        state_names: Vec<String>,
        parameter_names: Vec<String>,
        independent_name: impl Into<String>,
        config: SymbolicIvpGeneratedBackendConfig,
        telemetry: BvpSciTelemetry,
    ) -> Result<Self, BvpSciNewError> {
        Self::prepare_with_policy(
            assembly,
            layout,
            equations,
            state_names,
            parameter_names,
            independent_name,
            config,
            telemetry,
            BvpSciExecutionPolicy::Sequential,
        )
    }

    /// Prepare an AOT route with the selected warm callback policy.
    pub fn prepare_with_policy(
        assembly: BvpSciAssembly,
        layout: BvpSciMatrixLayout,
        equations: Vec<Expr>,
        state_names: Vec<String>,
        parameter_names: Vec<String>,
        independent_name: impl Into<String>,
        mut config: SymbolicIvpGeneratedBackendConfig,
        telemetry: BvpSciTelemetry,
        execution_policy: BvpSciExecutionPolicy,
    ) -> Result<Self, BvpSciNewError> {
        let dimension = equations.len();
        if dimension == 0 || state_names.len() != dimension {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::AotPreparation,
                expected: state_names.len(),
                actual: dimension,
            });
        }
        let parameter_count = parameter_names.len();
        let independent_name = independent_name.into();
        hydrate_handoff_resolver(&mut config)?;
        let symbolic_backend = match assembly {
            BvpSciAssembly::Numerical => {
                return Err(BvpSciNewError::UnsupportedRoute(
                    "AOT preparation requires a symbolic ExprLegacy or AtomView assembly".into(),
                ))
            }
            BvpSciAssembly::ExprLegacy => IvpSymbolicAssemblyBackend::ExprLegacy,
            BvpSciAssembly::AtomViewNative => IvpSymbolicAssemblyBackend::AtomView,
        };
        let ivp_telemetry = match telemetry.mode() {
            BvpSciTelemetryMode::Off => IvpTelemetry::disabled(),
            BvpSciTelemetryMode::Counters => IvpTelemetry::counters(),
            BvpSciTelemetryMode::Timings => IvpTelemetry::detailed(),
        };
        let execution_policy = to_ivp_policy(execution_policy);
        let options = SymbolicIvpProblemOptions::new()
            .with_symbolic_assembly_backend(symbolic_backend)
            .with_equation_parameters(parameter_names.clone())
            .with_equation_parameter_values(DVector::from_element(parameter_count, 1.0))
            .with_lambdify_execution_policy(execution_policy)
            .with_telemetry(ivp_telemetry.clone());
        let calibration_started = ivp_telemetry
            .start_cold_stage(crate::symbolic::ivp_telemetry::IvpColdStage::ParallelCalibration);
        configure_chunking(&mut config, layout, dimension, execution_policy);
        ivp_telemetry.record_cold_stage(
            crate::symbolic::ivp_telemetry::IvpColdStage::ParallelCalibration,
            calibration_started,
        );
        let handoff_path = config.handoff_path.clone();

        let plan_started = ivp_telemetry
            .start_cold_stage(crate::symbolic::ivp_telemetry::IvpColdStage::SolverPreparation);
        let plan = match layout {
            BvpSciMatrixLayout::Dense => {
                let result = prepare_generated_symbolic_ivp_problem(
                    equations,
                    state_names,
                    independent_name,
                    options,
                    config,
                )
                .map_err(aot_generated_error)?;
                let validation_telemetry = result.problem.telemetry.clone();
                let validation_started = validation_telemetry
                    .start_cold_stage(crate::symbolic::ivp_telemetry::IvpColdStage::BackendBinding);
                let validation = require_dense_runtime(&result);
                validation_telemetry.record_cold_stage(
                    crate::symbolic::ivp_telemetry::IvpColdStage::BackendBinding,
                    validation_started,
                );
                validation?;
                let updated_resolver = result.updated_resolver.clone();
                Ok(Self::Dense {
                    problem: Arc::new(result.into_problem()),
                    assembly,
                    dimension,
                    parameter_count,
                    execution_policy,
                    telemetry,
                    updated_resolver,
                })
            }
            BvpSciMatrixLayout::Sparse => {
                let result = prepare_generated_symbolic_ivp_sparse_backend(
                    equations,
                    state_names,
                    independent_name,
                    options,
                    config,
                )
                .map_err(aot_generated_error)?;
                Self::from_structured_result(
                    result,
                    assembly,
                    BvpSciMatrixLayout::Sparse,
                    dimension,
                    parameter_count,
                    execution_policy,
                    telemetry,
                    None,
                )
            }
            BvpSciMatrixLayout::Banded { lower, upper } => {
                let result = prepare_generated_symbolic_ivp_banded_backend(
                    equations,
                    state_names,
                    independent_name,
                    (lower, upper),
                    options,
                    config,
                )
                .map_err(aot_generated_error)?;
                let slot_map = AtomAotBandedSlotMap::new(dimension, dimension, lower, upper)
                    .map_err(|error| aot_preparation_error(error.to_string()))?;
                Self::from_structured_result(
                    result,
                    assembly,
                    BvpSciMatrixLayout::Banded { lower, upper },
                    dimension,
                    parameter_count,
                    execution_policy,
                    telemetry,
                    Some(slot_map),
                )
            }
        }?;
        ivp_telemetry.record_cold_stage(
            crate::symbolic::ivp_telemetry::IvpColdStage::SolverPreparation,
            plan_started,
        );
        publish_handoff(handoff_path.as_deref(), &plan)?;
        Ok(plan)
    }

    fn from_structured_result(
        result: PreparedGeneratedSymbolicIvpSparseBackend,
        assembly: BvpSciAssembly,
        layout: BvpSciMatrixLayout,
        dimension: usize,
        parameter_count: usize,
        execution_policy: IvpLambdifyExecutionPolicy,
        telemetry: BvpSciTelemetry,
        band_slots: Option<AtomAotBandedSlotMap>,
    ) -> Result<Self, BvpSciNewError> {
        let validation_telemetry = result.telemetry.clone();
        let validation_started = validation_telemetry
            .start_cold_stage(crate::symbolic::ivp_telemetry::IvpColdStage::BackendBinding);
        let validation = require_structured_runtime(&result);
        validation_telemetry.record_cold_stage(
            crate::symbolic::ivp_telemetry::IvpColdStage::BackendBinding,
            validation_started,
        );
        validation?;
        let backend = result.linked_backend.ok_or_else(|| {
            aot_preparation_error("generated route did not publish a linked callback")
        })?;
        let pattern = result
            .jacobian_structure
            .row_indices
            .into_iter()
            .zip(result.jacobian_structure.col_indices)
            .collect();
        Ok(Self::Structured {
            backend,
            assembly,
            layout,
            pattern,
            band_slots,
            dimension,
            parameter_count,
            execution_policy,
            telemetry,
            shared_telemetry: result.telemetry,
            updated_resolver: result.updated_resolver,
        })
    }

    pub fn assembly(&self) -> BvpSciAssembly {
        match self {
            Self::Dense { assembly, .. } | Self::Structured { assembly, .. } => *assembly,
        }
    }

    pub fn dimension(&self) -> usize {
        match self {
            Self::Dense { dimension, .. } | Self::Structured { dimension, .. } => *dimension,
        }
    }

    pub fn parameter_dimension(&self) -> usize {
        match self {
            Self::Dense {
                parameter_count, ..
            }
            | Self::Structured {
                parameter_count, ..
            } => *parameter_count,
        }
    }

    pub fn execution_policy(&self) -> BvpSciExecutionPolicy {
        match self {
            Self::Dense {
                execution_policy, ..
            }
            | Self::Structured {
                execution_policy, ..
            } => match execution_policy {
                IvpLambdifyExecutionPolicy::Sequential => BvpSciExecutionPolicy::Sequential,
                IvpLambdifyExecutionPolicy::Parallel { min_work } => {
                    BvpSciExecutionPolicy::Parallel {
                        min_work: *min_work,
                    }
                }
                IvpLambdifyExecutionPolicy::Auto { min_work } => BvpSciExecutionPolicy::Auto {
                    min_work: *min_work,
                },
            },
        }
    }

    pub fn jacobian_nnz(&self) -> usize {
        match self {
            Self::Dense { dimension, .. } => dimension.saturating_mul(*dimension),
            Self::Structured { pattern, .. } => pattern.len(),
        }
    }

    /// Return the native callback buffer length before collocation maps the
    /// values into its dense pointwise block.
    pub fn jacobian_callback_scratch_len(&self) -> Result<usize, BvpSciNewError> {
        match self {
            Self::Dense { dimension, .. } => dimension.checked_mul(*dimension).ok_or_else(|| {
                BvpSciNewError::InvalidConfiguration("Jacobian scratch size overflow".into())
            }),
            Self::Structured { backend, .. } => backend
                .jacobian_output_len()
                .map_err(|error| aot_callback_error(BvpSciStage::JacobianCallback, error)),
        }
    }

    pub fn jacobian_pattern(&self) -> Vec<(usize, usize)> {
        match self {
            Self::Dense { .. } => Vec::new(),
            Self::Structured { pattern, .. } => pattern.clone(),
        }
    }

    pub fn telemetry(&self) -> &BvpSciTelemetry {
        match self {
            Self::Dense { telemetry, .. } | Self::Structured { telemetry, .. } => telemetry,
        }
    }

    /// Return shared compiler/runtime telemetry for compact reports.
    pub fn aot_telemetry_snapshot(&self) -> crate::symbolic::ivp_telemetry::IvpTelemetrySnapshot {
        match self {
            Self::Dense { problem, .. } => problem.telemetry.snapshot(),
            Self::Structured {
                shared_telemetry, ..
            } => shared_telemetry.snapshot(),
        }
    }

    pub fn updated_resolver(&self) -> Option<&AotResolver> {
        match self {
            Self::Dense {
                updated_resolver, ..
            }
            | Self::Structured {
                updated_resolver, ..
            } => updated_resolver.as_ref(),
        }
    }

    fn bind_args(
        &self,
        x: f64,
        state: &[f64],
        parameters: &[f64],
        arguments: &mut [f64],
    ) -> Result<(), BvpSciNewError> {
        if !x.is_finite()
            || state.len() != self.dimension()
            || parameters.len() != self.parameter_dimension()
        {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::AotCallback,
                expected: 1 + self.dimension() + self.parameter_dimension(),
                actual: 1 + state.len() + parameters.len(),
            });
        }
        let expected = 1 + self.parameter_dimension() + self.dimension();
        if arguments.len() != expected {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::ArgumentBuffer,
                expected,
                actual: arguments.len(),
            });
        }
        arguments[0] = x;
        arguments[1..1 + parameters.len()].copy_from_slice(parameters);
        arguments[1 + parameters.len()..].copy_from_slice(state);
        self.telemetry().record_argument_binding(None);
        Ok(())
    }

    pub fn evaluate_rhs(
        &self,
        x: f64,
        state: &[f64],
        parameters: &[f64],
        arguments: &mut [f64],
        output: &mut [f64],
    ) -> Result<(), BvpSciNewError> {
        self.bind_args(x, state, parameters, arguments)?;
        if output.len() != self.dimension() {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::ResidualCallback,
                expected: self.dimension(),
                actual: output.len(),
            });
        }
        let callback_started = self.telemetry().start_timing();
        let evaluation_started = self.telemetry().start_timing();
        let result = match self {
            Self::Dense { problem, .. } => problem
                .try_evaluate_aot_residual_parts_with_args(arguments, output)
                .map_err(|error| aot_callback_error(BvpSciStage::ResidualCallback, error)),
            Self::Structured {
                backend,
                execution_policy,
                shared_telemetry,
                ..
            } => backend
                .try_residual_eval_with_policy(
                    arguments,
                    output,
                    *execution_policy,
                    shared_telemetry,
                )
                .map_err(|error| aot_callback_error(BvpSciStage::ResidualCallback, error)),
        };
        if result.is_ok() {
            // Keep the two diagnostic scopes independent. `rhs` is the public
            // callback boundary, while `residual_evaluation` is the generated
            // evaluator itself; recording both from one timestamp would make
            // the table look additive when the scopes are actually inclusive.
            self.telemetry()
                .record_residual_evaluation(evaluation_started);
            self.telemetry().record_rhs(callback_started);
            self.telemetry().record_output_writes(output.len());
        }
        result
    }

    /// Evaluate the pointwise Jacobian without allocating a temporary values
    /// vector. Structured AOT routes use `scratch` for compact callback output.
    pub fn evaluate_jacobian_dense_with_scratch(
        &self,
        x: f64,
        state: &[f64],
        parameters: &[f64],
        arguments: &mut [f64],
        output: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), BvpSciNewError> {
        self.bind_args(x, state, parameters, arguments)?;
        let expected = self.dimension().saturating_mul(self.dimension());
        if output.len() != expected {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::JacobianCallback,
                expected,
                actual: output.len(),
            });
        }
        let callback_started = self.telemetry().start_timing();
        let evaluation_started = self.telemetry().start_timing();
        match self {
            Self::Dense { problem, .. } => problem
                .try_evaluate_aot_jacobian_parts_with_args(arguments, output)
                .map_err(|error| aot_callback_error(BvpSciStage::JacobianCallback, error))?,
            Self::Structured {
                backend,
                pattern,
                band_slots,
                execution_policy,
                shared_telemetry,
                ..
            } => {
                let value_len = backend
                    .jacobian_output_len()
                    .map_err(|error| aot_callback_error(BvpSciStage::JacobianCallback, error))?;
                if scratch.len() < value_len {
                    return Err(BvpSciNewError::ShapeMismatch {
                        stage: BvpSciStage::JacobianOutputAssembly,
                        expected: value_len,
                        actual: scratch.len(),
                    });
                }
                backend
                    .try_jacobian_values_eval_with_policy(
                        arguments,
                        &mut scratch[..value_len],
                        *execution_policy,
                        shared_telemetry,
                    )
                    .map_err(|error| aot_callback_error(BvpSciStage::JacobianCallback, error))?;
                output.fill(0.0);
                if let Some(slots) = band_slots {
                    for (index, slot) in slots.slots().iter().enumerate() {
                        if let Some(row) = slot.matrix_row {
                            output[row * self.dimension() + slot.column] = scratch[index];
                        }
                    }
                } else {
                    for ((row, column), value) in pattern.iter().copied().zip(scratch.iter()) {
                        output[row * self.dimension() + column] = *value;
                    }
                }
            }
        }
        self.telemetry()
            .record_jacobian_evaluation(evaluation_started);
        self.telemetry()
            .record_jacobian_output_assembly(callback_started);
        self.telemetry().record_jacobian(callback_started);
        self.telemetry().record_output_writes(output.len());
        Ok(())
    }

    pub fn evaluate_jacobian_dense(
        &self,
        x: f64,
        state: &[f64],
        parameters: &[f64],
        arguments: &mut [f64],
        output: &mut [f64],
    ) -> Result<(), BvpSciNewError> {
        match self {
            Self::Dense { .. } => self.evaluate_jacobian_dense_with_scratch(
                x,
                state,
                parameters,
                arguments,
                output,
                &mut [],
            ),
            Self::Structured { backend, .. } => {
                // This convenience API is intentionally allocating. Numerical
                // collocation calls the scratch-taking variant above, while a
                // direct public callback query should still work for every
                // selected layout instead of failing with a scratch error.
                let value_len = backend
                    .jacobian_output_len()
                    .map_err(|error| aot_callback_error(BvpSciStage::JacobianCallback, error))?;
                let mut scratch = vec![0.0; value_len];
                self.evaluate_jacobian_dense_with_scratch(
                    x,
                    state,
                    parameters,
                    arguments,
                    output,
                    &mut scratch,
                )
            }
        }
    }

    pub fn evaluate_jacobian_values(
        &self,
        x: f64,
        state: &[f64],
        parameters: &[f64],
        arguments: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), BvpSciNewError> {
        self.bind_args(x, state, parameters, arguments)?;
        match self {
            Self::Dense { problem, .. } => problem
                .try_evaluate_aot_jacobian_parts_with_args(arguments, values)
                .map_err(|error| aot_callback_error(BvpSciStage::JacobianCallback, error)),
            Self::Structured {
                backend,
                execution_policy,
                shared_telemetry,
                ..
            } => {
                let expected = backend
                    .jacobian_output_len()
                    .map_err(|error| aot_callback_error(BvpSciStage::JacobianCallback, error))?;
                if values.len() != expected {
                    return Err(BvpSciNewError::ShapeMismatch {
                        stage: BvpSciStage::JacobianCallback,
                        expected,
                        actual: values.len(),
                    });
                }
                backend
                    .try_jacobian_values_eval_with_policy(
                        arguments,
                        values,
                        *execution_policy,
                        shared_telemetry,
                    )
                    .map_err(|error| aot_callback_error(BvpSciStage::JacobianCallback, error))
            }
        }
    }
}

fn require_dense_runtime(
    result: &PreparedGeneratedSymbolicIvpProblem,
) -> Result<(), BvpSciNewError> {
    if result.selected_backend != SelectedSymbolicIvpBackendKind::AotCompiled {
        return Err(aot_preparation_error(
            "generated dense route did not publish a compiled runtime",
        ));
    }
    result
        .try_aot_runtime()
        .map_err(|error| aot_runtime_error(error.to_string()))?
        .ok_or_else(|| aot_runtime_error("generated dense route has no runtime owner"))?;
    Ok(())
}

fn require_structured_runtime(
    result: &PreparedGeneratedSymbolicIvpSparseBackend,
) -> Result<(), BvpSciNewError> {
    if result.selected_backend != SelectedSymbolicIvpBackendKind::AotCompiled {
        return Err(aot_preparation_error(
            "generated structured route did not publish a compiled runtime",
        ));
    }
    result
        .try_aot_runtime()
        .map_err(|error| aot_runtime_error(error.to_string()))?
        .ok_or_else(|| aot_runtime_error("generated structured route has no runtime owner"))?;
    Ok(())
}

fn configure_chunking(
    config: &mut SymbolicIvpGeneratedBackendConfig,
    layout: BvpSciMatrixLayout,
    dimension: usize,
    policy: IvpLambdifyExecutionPolicy,
) {
    if matches!(policy, IvpLambdifyExecutionPolicy::Sequential) || dimension == 0 {
        return;
    }
    let residual_strategy = match policy {
        IvpLambdifyExecutionPolicy::Parallel { .. } => {
            recommended_residual_chunking_for_parallelism(dimension, 2)
        }
        IvpLambdifyExecutionPolicy::Auto { .. } => {
            recommended_residual_chunking_for_auto_parallelism(dimension)
        }
        IvpLambdifyExecutionPolicy::Sequential => unreachable!(),
    };
    // Structured ExprLegacy AOT consumes the residual strategy through the
    // historical dense AOT options field, while AtomViewNative consumes the
    // generated-backend field directly. Keep both views synchronized so the
    // selected policy cannot silently produce one chunk for ExprLegacy and
    // many chunks for AtomViewNative on the same BVP workload.
    config.residual_chunking_strategy = residual_strategy;
    config.aot_options.residual_strategy = residual_strategy;
    match layout {
        BvpSciMatrixLayout::Dense => {
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
        BvpSciMatrixLayout::Sparse => {
            let workers = std::thread::available_parallelism()
                .map(|value| value.get())
                .unwrap_or(1);
            let target_chunks = match policy {
                IvpLambdifyExecutionPolicy::Parallel { .. } => workers.saturating_mul(2),
                IvpLambdifyExecutionPolicy::Auto { .. } => workers,
                IvpLambdifyExecutionPolicy::Sequential => unreachable!(),
            }
            .min(dimension)
            .max(1);
            config.sparse_jacobian_chunking_strategy =
                SparseChunkingStrategy::ByTargetChunkCount { target_chunks };
        }
        BvpSciMatrixLayout::Banded { .. } => {
            let workers = std::thread::available_parallelism()
                .map(|value| value.get())
                .unwrap_or(1);
            let target_chunks = match policy {
                IvpLambdifyExecutionPolicy::Parallel { .. } => workers.saturating_mul(2),
                IvpLambdifyExecutionPolicy::Auto { .. } => workers,
                IvpLambdifyExecutionPolicy::Sequential => unreachable!(),
            }
            .min(dimension)
            .max(1);
            config.banded_jacobian_chunking_strategy =
                BandedChunkingStrategy::ByTargetChunkCount { target_chunks };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::codegen::codegen_aot_lifecycle::{
        AotFailureDiagnostics, AotFailureKind, AotLifecycleError, AotLifecycleStage,
    };
    use crate::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy;

    #[test]
    fn policy_chunking_propagates_residual_strategy_to_both_aot_config_views() {
        let mut config = SymbolicIvpGeneratedBackendConfig::default();
        configure_chunking(
            &mut config,
            BvpSciMatrixLayout::Sparse,
            64,
            IvpLambdifyExecutionPolicy::Parallel { min_work: 0 },
        );

        assert_ne!(config.residual_chunking_strategy, ResidualChunkingStrategy::Whole);
        assert_eq!(
            config.aot_options.residual_strategy,
            config.residual_chunking_strategy
        );
    }

    #[test]
    fn generated_aot_errors_keep_lifecycle_kind_at_bvp_boundary() {
        let error = aot_generated_error(SymbolicIvpGeneratedError::AotLifecycle(
            AotLifecycleError::new(AotFailureDiagnostics::new(
                AotLifecycleStage::Link,
                AotFailureKind::Timeout,
                "fixture-key",
                "compiler deadline exceeded",
            )),
        ));
        assert!(matches!(
            error,
            BvpSciNewError::AotPreparation {
                stage: BvpSciStage::AotLink,
                kind: BvpSciAotFailureKind::Timeout,
                ..
            }
        ));
    }

    #[test]
    fn generated_aot_failure_matrix_preserves_compiler_link_timeout_and_artifact_kinds() {
        let cases = [
            (
                AotLifecycleStage::Build,
                AotFailureKind::Compiler,
                BvpSciStage::AotBuild,
                BvpSciAotFailureKind::Compiler,
            ),
            (
                AotLifecycleStage::Link,
                AotFailureKind::Link,
                BvpSciStage::AotLink,
                BvpSciAotFailureKind::Link,
            ),
            (
                AotLifecycleStage::Build,
                AotFailureKind::Timeout,
                BvpSciStage::AotBuild,
                BvpSciAotFailureKind::Timeout,
            ),
            (
                AotLifecycleStage::Planned,
                AotFailureKind::StaleArtifact,
                BvpSciStage::AotPreparation,
                BvpSciAotFailureKind::StaleArtifact,
            ),
        ];
        for (stage, kind, expected_stage, expected_kind) in cases {
            let error = aot_generated_error(SymbolicIvpGeneratedError::AotLifecycle(
                AotLifecycleError::new(AotFailureDiagnostics::new(
                    stage,
                    kind,
                    "matrix-key",
                    "synthetic lifecycle failure",
                )),
            ));
            assert!(matches!(
                error,
                BvpSciNewError::AotPreparation { stage, kind, .. }
                    if stage == expected_stage && kind == expected_kind
            ));
        }
    }

    #[test]
    fn retry_exhaustion_preserves_the_root_compiler_kind() {
        let mut diagnostics = AotFailureDiagnostics::new(
            AotLifecycleStage::Build,
            AotFailureKind::RetryExhausted,
            "retry-key",
            "compiler retry budget exhausted",
        );
        diagnostics.root_kind = Some(AotFailureKind::Compiler);

        let error = aot_generated_error(SymbolicIvpGeneratedError::AotLifecycle(
            AotLifecycleError::new(diagnostics),
        ));

        assert!(matches!(
            error,
            BvpSciNewError::AotPreparation {
                stage: BvpSciStage::AotBuild,
                kind: BvpSciAotFailureKind::Compiler,
                ..
            }
        ));
    }
}

fn aot_preparation_error(message: impl Into<String>) -> BvpSciNewError {
    BvpSciNewError::AotPreparation {
        stage: BvpSciStage::AotPreparation,
        kind: BvpSciAotFailureKind::Unknown,
        message: message.into(),
    }
}

/// Preserve the shared generated-backend classification at the BVP API
/// boundary. The compiler diagnostic is retained as text for humans, while
/// lifecycle consumers receive a stable machine-readable kind and stage.
fn aot_generated_error(error: SymbolicIvpGeneratedError) -> BvpSciNewError {
    let (stage, kind) = match &error {
        SymbolicIvpGeneratedError::IvpBackend(_) => (
            BvpSciStage::SymbolicPreparation,
            BvpSciAotFailureKind::SymbolicPreparation,
        ),
        SymbolicIvpGeneratedError::CompiledAotArtifactMissing(_) => (
            BvpSciStage::AotCacheLookup,
            BvpSciAotFailureKind::MissingArtifact,
        ),
        SymbolicIvpGeneratedError::CompiledAotArtifactNotBuilt(_) => {
            (BvpSciStage::AotBuild, BvpSciAotFailureKind::Build)
        }
        SymbolicIvpGeneratedError::CompiledAotRuntimeUnavailable(_) => {
            (BvpSciStage::AotRuntime, BvpSciAotFailureKind::Runtime)
        }
        SymbolicIvpGeneratedError::AotBuildOutputDirMissing
        | SymbolicIvpGeneratedError::AotBuildFailed(_) => {
            (BvpSciStage::AotBuild, BvpSciAotFailureKind::Build)
        }
        SymbolicIvpGeneratedError::AotLifecycle(lifecycle) => {
            use crate::symbolic::codegen::codegen_aot_lifecycle::{
                AotFailureKind, AotLifecycleStage,
            };
            let stage = match lifecycle.diagnostics.stage {
                AotLifecycleStage::Build => BvpSciStage::AotBuild,
                AotLifecycleStage::Link => BvpSciStage::AotLink,
                AotLifecycleStage::Published => BvpSciStage::AotPublication,
                AotLifecycleStage::RuntimeReady => BvpSciStage::AotRuntime,
                AotLifecycleStage::Planned | AotLifecycleStage::Materialized => {
                    BvpSciStage::AotPreparation
                }
            };
            // RetryExhausted is a transport-level wrapper. Prefer its
            // recorded root kind so repeated compiler/link failures retain
            // their actionable classification at the BVP boundary.
            let failure_kind = lifecycle
                .diagnostics
                .root_kind
                .unwrap_or(lifecycle.diagnostics.kind);
            let kind = match failure_kind {
                AotFailureKind::Timeout => BvpSciAotFailureKind::Timeout,
                AotFailureKind::Link => BvpSciAotFailureKind::Link,
                AotFailureKind::StaleArtifact => BvpSciAotFailureKind::StaleArtifact,
                AotFailureKind::Compiler => BvpSciAotFailureKind::Compiler,
                AotFailureKind::PartialArtifact => BvpSciAotFailureKind::Build,
                AotFailureKind::Io
                | AotFailureKind::Lock
                | AotFailureKind::Manifest
                | AotFailureKind::OutputShape
                | AotFailureKind::RetryExhausted
                | AotFailureKind::Quarantined => BvpSciAotFailureKind::Publication,
            };
            (stage, kind)
        }
    };
    BvpSciNewError::AotPreparation {
        stage,
        kind,
        message: error.to_string(),
    }
}

fn aot_runtime_error(message: impl Into<String>) -> BvpSciNewError {
    BvpSciNewError::AotRuntime {
        stage: BvpSciStage::AotRuntime,
        message: message.into(),
    }
}

fn aot_callback_error(stage: BvpSciStage, error: impl ToString) -> BvpSciNewError {
    BvpSciNewError::AotCallback {
        stage,
        message: error.to_string(),
    }
}

/// Load a producer's durable resolver snapshot before strict consumer lookup.
///
/// The generated-IVP layer deliberately keeps handoff files as metadata. This
/// BVP adapter owns the policy boundary: a missing handoff is left to the
/// selected `RequirePrebuilt` policy to classify as a normal missing artifact,
/// while a malformed handoff is surfaced as a typed handoff failure.
fn hydrate_handoff_resolver(
    config: &mut SymbolicIvpGeneratedBackendConfig,
) -> Result<(), BvpSciNewError> {
    if config.resolver.is_none() {
        if let Some(path) = config.handoff_path.as_ref() {
            if path.exists() {
                config.resolver = Some(AotResolver::read_handoff(path).map_err(|error| {
                    BvpSciNewError::AotHandoff {
                        stage: BvpSciStage::AotCacheLookup,
                        message: format!("cannot read {}: {error}", path.display()),
                    }
                })?);
            }
        }
    }
    Ok(())
}

/// Publish all artifacts returned by one preparation into the durable handoff.
///
/// Handoffs are merged rather than overwritten because a release matrix may
/// publish Dense, Sparse and Banded routes into the same manifest. The
/// compiled files remain in their original output directories; only registry
/// provenance crosses the process boundary.
fn publish_handoff(
    handoff_path: Option<&std::path::Path>,
    plan: &BvpSciAotPlan,
) -> Result<(), BvpSciNewError> {
    let Some(path) = handoff_path else {
        return Ok(());
    };
    let Some(resolver) = plan.updated_resolver() else {
        return Ok(());
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| BvpSciNewError::AotHandoff {
            stage: BvpSciStage::AotPublication,
            message: format!("cannot create {}: {error}", parent.display()),
        })?;
    }
    resolver
        .merge_handoff(path)
        .map_err(|error| BvpSciNewError::AotHandoff {
            stage: BvpSciStage::AotPublication,
            message: format!("cannot publish {}: {error}", path.display()),
        })
}

fn to_ivp_policy(policy: BvpSciExecutionPolicy) -> IvpLambdifyExecutionPolicy {
    match policy {
        BvpSciExecutionPolicy::Sequential => IvpLambdifyExecutionPolicy::Sequential,
        BvpSciExecutionPolicy::Parallel { min_work } => {
            IvpLambdifyExecutionPolicy::Parallel { min_work }
        }
        BvpSciExecutionPolicy::Auto { min_work } => IvpLambdifyExecutionPolicy::Auto { min_work },
    }
}
