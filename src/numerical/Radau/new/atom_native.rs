//! Radau's AtomViewNative adapter over the shared symbolic IVP pipeline.
//!
//! The adapter deliberately contains no second Atom compiler. The shared
//! `symbolic_ivp` preparation owns the `Expr -> Atom` conversion, dependency
//! analysis, native differentiation, and evaluator compilation. Radau only
//! translates its typed callback errors and supplies caller-owned buffers.

use crate::symbolic::ivp_telemetry::IvpLambdifyExecutionPolicy;
use crate::symbolic::ivp_telemetry::IvpTelemetry;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp::{
    IvpBackendError, IvpSymbolicAssemblyBackend, SymbolicIvpProblemOptions,
    prepare_symbolic_ivp_problem,
};

use super::{
    callbacks::validate_callback_output,
    config::RadauMatrixLayout,
    error::{RadauConfigError, RadauError, RadauStage, RadauUnsupportedRoute},
    jacobian::RadauJacobianPattern,
    telemetry::{RadauCallbackStage, RadauTelemetry, RadauTelemetryMode as RadauMode},
};

/// Reusable Radau workspace for native Atom Jacobian values.
///
/// The shared IVP evaluator publishes values in its prepared structural order.
/// This buffer is therefore sized once per session and reused for structured
/// output assembly; it is not a symbolic expression cache.
pub(crate) struct AtomNativeWorkspace {
    values: Vec<f64>,
}

impl AtomNativeWorkspace {
    /// Allocate reusable native Jacobian value storage.
    pub(crate) fn new(jacobian_entries: usize) -> Self {
        Self {
            values: vec![0.0; jacobian_entries],
        }
    }

    /// Report native value-buffer capacity for continuation diagnostics.
    pub(crate) fn value_capacity(&self) -> usize {
        self.values.capacity()
    }
}

/// Radau-owned handle over the shared, verified AtomView IVP plan.
///
/// AtomNative means Expr enters the shared IVP pipeline once and all later
/// differentiation/evaluation stays in Atom form.  Radau does not convert
/// Atom expressions back to Expr and does not maintain a second compiler.
pub(crate) struct AtomNativePlan {
    problem: crate::symbolic::symbolic_ivp::PreparedSymbolicIvpProblem,
    dimension: usize,
    parameter_count: usize,
    execution_policy: IvpLambdifyExecutionPolicy,
}

impl AtomNativePlan {
    /// Prepare the shared AtomView native route.
    ///
    /// AtomView derives the analytic state Jacobian from the residual graph
    /// unless the caller supplies an explicit Jacobian. Explicit entries are
    /// converted to Atom evaluators once and stay on the native path.
    pub(crate) fn from_expressions(
        residual: Vec<Expr>,
        jacobian: Option<Vec<Expr>>,
        independent_variable: &str,
        variables: &[&str],
        parameters: &[&str],
    ) -> Result<Self, RadauError> {
        Self::from_expressions_with_telemetry(
            residual,
            jacobian,
            independent_variable,
            variables,
            parameters,
            RadauMode::Off,
        )
    }

    /// Prepare AtomView directly from the residual graph and record cold stages.
    pub(crate) fn from_expressions_with_telemetry(
        residual: Vec<Expr>,
        jacobian: Option<Vec<Expr>>,
        independent_variable: &str,
        variables: &[&str],
        parameters: &[&str],
        telemetry_mode: RadauMode,
    ) -> Result<Self, RadauError> {
        Self::from_expressions_with_telemetry_and_policy(
            residual,
            jacobian,
            independent_variable,
            variables,
            parameters,
            telemetry_mode,
            IvpLambdifyExecutionPolicy::Sequential,
        )
    }

    pub(crate) fn from_expressions_with_telemetry_and_policy(
        residual: Vec<Expr>,
        jacobian: Option<Vec<Expr>>,
        independent_variable: &str,
        variables: &[&str],
        parameters: &[&str],
        telemetry_mode: RadauMode,
        execution_policy: IvpLambdifyExecutionPolicy,
    ) -> Result<Self, RadauError> {
        let dimension = residual.len();
        if dimension == 0 {
            return Err(RadauConfigError::ZeroDimension.into());
        }
        let explicit_jacobian = if let Some(jacobian) = jacobian {
            let expected = dimension
                .checked_mul(dimension)
                .ok_or(RadauError::WorkspaceSizeOverflow { dimension })?;
            if jacobian.len() != expected {
                return Err(RadauError::ShapeMismatch {
                    stage: RadauStage::Preparation,
                    expected,
                    actual: jacobian.len(),
                });
            }
            Some(
                jacobian
                    .chunks(dimension.max(1))
                    .map(ToOwned::to_owned)
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };

        // Names and parameter slots are preparation data.  Numeric parameter
        // values are rebound later without rebuilding this native plan.
        let variable_names: Vec<String> = variables.iter().map(|name| (*name).to_owned()).collect();
        let parameter_names: Vec<String> =
            parameters.iter().map(|name| (*name).to_owned()).collect();
        let parameter_values = nalgebra::DVector::from_element(parameters.len(), 0.0);
        let options = SymbolicIvpProblemOptions::new()
            .with_symbolic_assembly_backend(IvpSymbolicAssemblyBackend::AtomView)
            .with_equation_parameters(parameter_names)
            .with_equation_parameter_values(parameter_values)
            .with_lambdify_execution_policy(execution_policy);
        let options = match explicit_jacobian {
            Some(jacobian) => options.with_explicit_jacobian(jacobian),
            None => options,
        };
        let options = options.with_telemetry(match telemetry_mode {
            RadauMode::Off => IvpTelemetry::disabled(),
            RadauMode::Counters => IvpTelemetry::counters(),
            RadauMode::Timings => IvpTelemetry::detailed(),
        });
        let problem = prepare_symbolic_ivp_problem(
            residual,
            variable_names,
            independent_variable.to_owned(),
            options,
        )
        .map_err(|error| map_ivp_error(RadauStage::Preparation, error))?;
        if problem.native_jacobian_entry_count().is_none() {
            return Err(RadauConfigError::NativeJacobianNotPublished.into());
        }

        Ok(Self {
            problem,
            dimension,
            parameter_count: parameters.len(),
            execution_policy,
        })
    }

    /// Merge shared IVP preparation stages into Radau-owned telemetry.
    pub(crate) fn absorb_preparation_telemetry(&self, telemetry: &mut RadauTelemetry) {
        telemetry.absorb_ivp_preparation(&self.problem.telemetry.snapshot());
    }

    /// Allocate session-local native Jacobian value storage.
    pub(crate) fn workspace(&self) -> AtomNativeWorkspace {
        AtomNativeWorkspace::new(self.jacobian_entry_count())
    }

    /// Return the number of parameter values accepted by the native evaluator.
    pub(crate) fn parameter_count(&self) -> usize {
        self.parameter_count
    }

    pub(crate) fn execution_policy(&self) -> IvpLambdifyExecutionPolicy {
        self.execution_policy
    }

    /// Return the number of nonzero native Jacobian entries.
    pub(crate) fn jacobian_entry_count(&self) -> usize {
        self.problem.native_jacobian_entry_count().unwrap_or(0)
    }

    /// Return the native Jacobian pattern in callback output order.
    pub(crate) fn jacobian_pattern(&self) -> Vec<(usize, usize)> {
        self.problem.native_jacobian_pattern().unwrap_or_default()
    }

    /// Evaluate the residual using disabled telemetry.
    pub(crate) fn evaluate_residual(
        &self,
        t: f64,
        state: &[f64],
        parameters: &[f64],
        output: &mut [f64],
        workspace: &mut AtomNativeWorkspace,
    ) -> Result<(), RadauError> {
        let mut telemetry = RadauTelemetry::new(Default::default());
        self.evaluate_residual_with_telemetry(
            t,
            state,
            parameters,
            output,
            workspace,
            &mut telemetry,
        )
    }

    /// Evaluate the residual into caller-owned output and measure its stages.
    pub(crate) fn evaluate_residual_with_telemetry(
        &self,
        t: f64,
        state: &[f64],
        parameters: &[f64],
        output: &mut [f64],
        _workspace: &mut AtomNativeWorkspace,
        telemetry: &mut RadauTelemetry,
    ) -> Result<(), RadauError> {
        telemetry.measure_callback(RadauCallbackStage::ArgumentBinding, || {
            self.validate_inputs(state, parameters, RadauStage::Residual)
        })?;
        if output.len() != self.dimension {
            return Err(RadauError::ShapeMismatch {
                stage: RadauStage::Residual,
                expected: self.dimension,
                actual: output.len(),
            });
        }
        let result = telemetry.measure_callback(RadauCallbackStage::ResidualEvaluation, || {
            self.problem
                .try_evaluate_native_residual_parts(t, parameters, state, output)
                .map_err(|error| map_ivp_error(RadauStage::Residual, error))?;
            validate_callback_output(RadauStage::Residual, self.dimension, output)
        });
        if result.is_ok() {
            telemetry.count_output_writes(output.len());
        }
        result
    }

    /// Evaluate native Jacobian values using disabled telemetry.
    pub(crate) fn evaluate_jacobian(
        &self,
        t: f64,
        state: &[f64],
        parameters: &[f64],
        output: &mut [f64],
        workspace: &mut AtomNativeWorkspace,
    ) -> Result<(), RadauError> {
        let mut telemetry = RadauTelemetry::new(Default::default());
        self.evaluate_jacobian_with_telemetry(
            t,
            state,
            parameters,
            output,
            workspace,
            &mut telemetry,
        )
    }

    /// Evaluate native Jacobian values while measuring callback stages.
    pub(crate) fn evaluate_jacobian_with_telemetry(
        &self,
        t: f64,
        state: &[f64],
        parameters: &[f64],
        output: &mut [f64],
        workspace: &mut AtomNativeWorkspace,
        telemetry: &mut RadauTelemetry,
    ) -> Result<(), RadauError> {
        telemetry.measure_callback(RadauCallbackStage::ArgumentBinding, || {
            self.validate_inputs(state, parameters, RadauStage::Jacobian)
        })?;
        let expected = self.dimension.checked_mul(self.dimension).ok_or(
            RadauError::WorkspaceSizeOverflow {
                dimension: self.dimension,
            },
        )?;
        if output.len() != expected {
            return Err(RadauError::ShapeMismatch {
                stage: RadauStage::Jacobian,
                expected,
                actual: output.len(),
            });
        }
        let result = telemetry.measure_callback(RadauCallbackStage::JacobianEvaluation, || {
            self.problem
                .try_evaluate_native_jacobian_parts(
                    t,
                    parameters,
                    state,
                    output,
                    &mut workspace.values,
                )
                .map_err(|error| map_ivp_error(RadauStage::Jacobian, error))?;
            validate_callback_output(RadauStage::Jacobian, expected, output)
        });
        if result.is_ok() {
            telemetry.count_output_writes(output.len());
        }
        result
    }

    /// Evaluate native Jacobian values in the requested storage layout.
    pub(crate) fn evaluate_jacobian_layout(
        &self,
        t: f64,
        state: &[f64],
        parameters: &[f64],
        layout: RadauMatrixLayout,
        output: &mut [f64],
        workspace: &mut AtomNativeWorkspace,
    ) -> Result<(), RadauError> {
        let mut telemetry = RadauTelemetry::new(Default::default());
        self.evaluate_jacobian_layout_with_telemetry(
            t,
            state,
            parameters,
            layout,
            output,
            workspace,
            &mut telemetry,
        )
    }

    /// Evaluate and project native Jacobian values while measuring assembly.
    ///
    /// Dense output can be filled directly by the shared evaluator.  Sparse
    /// and Banded routes use the same native value order but still need a
    /// layout projection today because the shared IVP API owns that value
    /// buffer.  The projection is explicit and measured; it is not hidden
    /// inside factorization and is a candidate for a future direct-writer API.
    pub(crate) fn evaluate_jacobian_layout_with_telemetry(
        &self,
        t: f64,
        state: &[f64],
        parameters: &[f64],
        layout: RadauMatrixLayout,
        output: &mut [f64],
        workspace: &mut AtomNativeWorkspace,
        telemetry: &mut RadauTelemetry,
    ) -> Result<(), RadauError> {
        match layout {
            RadauMatrixLayout::Dense => self.evaluate_jacobian_with_telemetry(
                t, state, parameters, output, workspace, telemetry,
            ),
            RadauMatrixLayout::Sparse => {
                telemetry.measure_callback(RadauCallbackStage::ArgumentBinding, || {
                    self.validate_inputs(state, parameters, RadauStage::Jacobian)
                })?;
                let expected = self.jacobian_entry_count();
                if output.len() != expected {
                    return Err(RadauError::ShapeMismatch {
                        stage: RadauStage::Jacobian,
                        expected,
                        actual: output.len(),
                    });
                }
                telemetry.measure_callback(RadauCallbackStage::JacobianEvaluation, || {
                    self.problem
                        .try_evaluate_native_jacobian_values_parts(
                            t,
                            parameters,
                            state,
                            &mut workspace.values,
                        )
                        .map_err(|error| map_ivp_error(RadauStage::Jacobian, error))
                })?;
                let result =
                    telemetry.measure_callback(RadauCallbackStage::JacobianOutputAssembly, || {
                        // The shared evaluator currently writes its native
                        // value buffer.  Copy into the caller's backend-owned
                        // buffer, keeping this unavoidable bridge visible in
                        // telemetry instead of disguising it as evaluation.
                        output.copy_from_slice(&workspace.values);
                        validate_callback_output(RadauStage::Jacobian, expected, output)
                    });
                if result.is_ok() {
                    telemetry.count_copy();
                    telemetry.count_output_writes(output.len());
                }
                result
            }
            RadauMatrixLayout::Banded { lower, upper } => {
                telemetry.measure_callback(RadauCallbackStage::ArgumentBinding, || {
                    self.validate_inputs(state, parameters, RadauStage::Jacobian)
                })?;
                // The native pattern is sparse; compact banded storage may
                // contain additional structural zero slots.  Validate the
                // promise once per callback and project values by slot.
                let pattern = RadauJacobianPattern::banded(self.dimension, lower, upper)?;
                let expected = pattern.values_len(layout);
                if output.len() != expected {
                    return Err(RadauError::ShapeMismatch {
                        stage: RadauStage::Jacobian,
                        expected,
                        actual: output.len(),
                    });
                }
                let native_pattern = self.jacobian_pattern();
                if native_pattern
                    .iter()
                    .any(|&(row, col)| !pattern.is_in_layout(layout, row, col))
                {
                    return Err(RadauUnsupportedRoute::JacobianOutsideBandedLayout.into());
                }
                telemetry.measure_callback(RadauCallbackStage::JacobianEvaluation, || {
                    self.problem
                        .try_evaluate_native_jacobian_values_parts(
                            t,
                            parameters,
                            state,
                            &mut workspace.values,
                        )
                        .map_err(|error| map_ivp_error(RadauStage::Jacobian, error))
                })?;
                let result =
                    telemetry.measure_callback(RadauCallbackStage::JacobianOutputAssembly, || {
                        output.fill(0.0);
                        for ((row, col), value) in native_pattern
                            .into_iter()
                            .zip(workspace.values.iter().copied())
                        {
                            let Some(slot) = pattern.compact_slot(layout, row, col) else {
                                return Err(RadauConfigError::InvalidBandedSlot.into());
                            };
                            output[slot] = value;
                        }
                        validate_callback_output(RadauStage::Jacobian, expected, output)
                    });
                if result.is_ok() {
                    telemetry.count_output_writes(output.len());
                }
                result
            }
        }
    }

    fn validate_inputs(
        &self,
        state: &[f64],
        parameters: &[f64],
        stage: RadauStage,
    ) -> Result<(), RadauError> {
        if state.len() != self.dimension {
            return Err(RadauError::ShapeMismatch {
                stage,
                expected: self.dimension,
                actual: state.len(),
            });
        }
        if parameters.len() != self.parameter_count {
            return Err(RadauError::ShapeMismatch {
                stage,
                expected: self.parameter_count,
                actual: parameters.len(),
            });
        }
        Ok(())
    }
}

fn map_ivp_error(stage: RadauStage, error: IvpBackendError) -> RadauError {
    RadauError::Callback {
        stage,
        message: error.to_string(),
    }
}
