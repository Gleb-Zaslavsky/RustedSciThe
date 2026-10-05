//! Prepared Lambdify callbacks for the new Radau route.
//!
//! Expressions are compiled once. Evaluation reuses a session-owned argument
//! buffer and writes directly into caller-owned residual/Jacobian storage.
//! The flattened ABI follows LSODE2: `[t, parameters..., state...]`.

use crate::symbolic::ivp_telemetry::IvpLambdifyExecutionPolicy;
use crate::symbolic::symbolic_engine::Expr;

use super::{
    callbacks::validate_callback_output,
    config::RadauMatrixLayout,
    error::{RadauConfigError, RadauError, RadauStage, RadauUnsupportedRoute},
    jacobian::RadauJacobianPattern,
    telemetry::{RadauCallbackStage, RadauTelemetry},
};

type ScalarLambdified = Box<dyn Fn(&[f64]) -> f64 + Send + Sync>;

/// Prepared ExprLegacy residual/Jacobian closures and their structural pattern.
///
/// The closures are immutable after preparation.  Runtime calls only bind
/// values into `LambdifyWorkspace::args` and write evaluator results into
/// caller-owned buffers, which keeps parameter continuation out of the
/// symbolic-construction path.
pub(crate) struct LambdifyPlan {
    residual: Vec<ScalarLambdified>,
    jacobian: Option<Vec<ScalarLambdified>>,
    jacobian_pattern: Vec<(usize, usize)>,
    dimension: usize,
    parameter_count: usize,
    execution_policy: IvpLambdifyExecutionPolicy,
}

/// Caller-owned argument buffer reused by every Lambdify callback invocation.
/// Its ABI is `[time, parameters..., state...]`; changing that order would
/// silently evaluate valid expressions with the wrong values.
pub(crate) struct LambdifyWorkspace {
    args: Vec<f64>,
}

impl LambdifyWorkspace {
    /// Allocate an argument buffer of the exact callback schema size.
    pub(crate) fn new(argument_count: usize) -> Self {
        Self {
            args: vec![0.0; argument_count],
        }
    }

    /// Report capacity for continuation allocation diagnostics.
    pub(crate) fn argument_capacity(&self) -> usize {
        self.args.capacity()
    }
}

impl LambdifyPlan {
    /// Compile ExprLegacy residual/Jacobian closures without evaluating them.
    pub(crate) fn from_expressions(
        residual: Vec<Expr>,
        jacobian: Option<Vec<Expr>>,
        independent_variable: &str,
        variables: &[&str],
        parameters: &[&str],
    ) -> Result<Self, RadauError> {
        Self::from_expressions_with_policy(
            residual,
            jacobian,
            independent_variable,
            variables,
            parameters,
            IvpLambdifyExecutionPolicy::Sequential,
        )
    }

    /// Compile ExprLegacy callbacks and retain the selected warm execution policy.
    pub(crate) fn from_expressions_with_policy(
        residual: Vec<Expr>,
        jacobian: Option<Vec<Expr>>,
        independent_variable: &str,
        variables: &[&str],
        parameters: &[&str],
        execution_policy: IvpLambdifyExecutionPolicy,
    ) -> Result<Self, RadauError> {
        let dimension = residual.len();
        if dimension == 0 {
            return Err(RadauConfigError::ZeroDimension.into());
        }
        if let Some(ref jacobian) = jacobian {
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
        }

        // Build the callback schema once.  Every compiled closure borrows this
        // stable name order, while later calls copy only numeric values.
        let mut names = Vec::with_capacity(1 + variables.len() + parameters.len());
        names.push(independent_variable.to_owned());
        names.extend(parameters.iter().map(|name| (*name).to_owned()));
        names.extend(variables.iter().map(|name| (*name).to_owned()));
        let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();

        for expression in residual.iter().chain(jacobian.iter().flatten()) {
            validate_expression_variables(expression, &name_refs)?;
        }

        let residual = residual
            .iter()
            .map(|expression| Expr::lambdify_borrowed_thread_safe(expression, &name_refs))
            .collect();
        let jacobian_pattern = jacobian
            .as_ref()
            .map(|expressions| {
                expressions
                    .iter()
                    .enumerate()
                    .filter_map(|(index, expression)| {
                        (!expression.is_zero()).then_some((index / dimension, index % dimension))
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let jacobian = jacobian.map(|expressions| {
            expressions
                .iter()
                .map(|expression| Expr::lambdify_borrowed_thread_safe(expression, &name_refs))
                .collect()
        });

        Ok(Self {
            residual,
            jacobian,
            jacobian_pattern,
            dimension,
            parameter_count: parameters.len(),
            execution_policy,
        })
    }

    /// Create fresh reusable callback storage for this prepared plan.
    pub(crate) fn workspace(&self) -> LambdifyWorkspace {
        LambdifyWorkspace::new(1 + self.parameter_count + self.dimension)
    }

    /// Return the number of parameter values expected by callbacks.
    pub(crate) fn parameter_count(&self) -> usize {
        self.parameter_count
    }

    pub(crate) fn execution_policy(&self) -> IvpLambdifyExecutionPolicy {
        self.execution_policy
    }

    /// Evaluate the residual using the default disabled telemetry mode.
    pub(crate) fn evaluate_residual(
        &self,
        t: f64,
        state: &[f64],
        parameters: &[f64],
        output: &mut [f64],
        workspace: &mut LambdifyWorkspace,
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

    /// Evaluate the residual while reporting binding and evaluation stages.
    pub(crate) fn evaluate_residual_with_telemetry(
        &self,
        t: f64,
        state: &[f64],
        parameters: &[f64],
        output: &mut [f64],
        workspace: &mut LambdifyWorkspace,
        telemetry: &mut RadauTelemetry,
    ) -> Result<(), RadauError> {
        telemetry.measure_callback(RadauCallbackStage::ArgumentBinding, || {
            self.fill_args(t, state, parameters, workspace, RadauStage::Residual)
        })?;
        if output.len() != self.dimension {
            return Err(RadauError::ShapeMismatch {
                stage: RadauStage::Residual,
                expected: self.dimension,
                actual: output.len(),
            });
        }
        let result = telemetry.measure_callback(RadauCallbackStage::ResidualEvaluation, || {
            evaluate_scalar_entries(
                &self.residual,
                output,
                &workspace.args,
                self.execution_policy,
            );
            validate_callback_output(RadauStage::Residual, self.dimension, output)
        });
        if result.is_ok() {
            telemetry.count_output_writes(output.len());
        }
        result
    }

    /// Evaluate the dense Jacobian using disabled telemetry.
    pub(crate) fn evaluate_jacobian(
        &self,
        t: f64,
        state: &[f64],
        parameters: &[f64],
        output: &mut [f64],
        workspace: &mut LambdifyWorkspace,
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

    /// Evaluate the dense Jacobian into caller-owned storage with telemetry.
    pub(crate) fn evaluate_jacobian_with_telemetry(
        &self,
        t: f64,
        state: &[f64],
        parameters: &[f64],
        output: &mut [f64],
        workspace: &mut LambdifyWorkspace,
        telemetry: &mut RadauTelemetry,
    ) -> Result<(), RadauError> {
        telemetry.measure_callback(RadauCallbackStage::ArgumentBinding, || {
            self.fill_args(t, state, parameters, workspace, RadauStage::Jacobian)
        })?;
        self.evaluate_jacobian_with_args(output, workspace, telemetry)
    }

    fn evaluate_jacobian_with_args(
        &self,
        output: &mut [f64],
        workspace: &LambdifyWorkspace,
        telemetry: &mut RadauTelemetry,
    ) -> Result<(), RadauError> {
        let Some(jacobian) = self.jacobian.as_ref() else {
            return Err(RadauUnsupportedRoute::LambdifyJacobianNotPrepared.into());
        };
        let expected = self.dimension * self.dimension;
        if output.len() != expected {
            return Err(RadauError::ShapeMismatch {
                stage: RadauStage::Jacobian,
                expected,
                actual: output.len(),
            });
        }
        let result = telemetry.measure_callback(RadauCallbackStage::JacobianEvaluation, || {
            evaluate_scalar_entries(jacobian, output, &workspace.args, self.execution_policy);
            validate_callback_output(RadauStage::Jacobian, expected, output)
        });
        if result.is_ok() {
            telemetry.count_output_writes(output.len());
        }
        result
    }

    /// Return the prepared structural Jacobian pattern.
    pub(crate) fn jacobian_pattern(&self) -> &[(usize, usize)] {
        &self.jacobian_pattern
    }

    /// Evaluate the Jacobian directly in Dense, Sparse, or compact Banded layout.
    pub(crate) fn evaluate_jacobian_layout(
        &self,
        t: f64,
        state: &[f64],
        parameters: &[f64],
        layout: RadauMatrixLayout,
        output: &mut [f64],
        workspace: &mut LambdifyWorkspace,
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

    /// Evaluate and project the Jacobian while measuring output assembly.
    pub(crate) fn evaluate_jacobian_layout_with_telemetry(
        &self,
        t: f64,
        state: &[f64],
        parameters: &[f64],
        layout: RadauMatrixLayout,
        output: &mut [f64],
        workspace: &mut LambdifyWorkspace,
        telemetry: &mut RadauTelemetry,
    ) -> Result<(), RadauError> {
        telemetry.measure_callback(RadauCallbackStage::ArgumentBinding, || {
            self.fill_args(t, state, parameters, workspace, RadauStage::Jacobian)
        })?;
        let Some(jacobian) = self.jacobian.as_ref() else {
            return Err(RadauUnsupportedRoute::LambdifyJacobianNotPrepared.into());
        };
        match layout {
            RadauMatrixLayout::Dense => {
                self.evaluate_jacobian_with_args(output, workspace, telemetry)
            }
            RadauMatrixLayout::Sparse => {
                if output.len() != self.jacobian_pattern.len() {
                    return Err(RadauError::ShapeMismatch {
                        stage: RadauStage::Jacobian,
                        expected: self.jacobian_pattern.len(),
                        actual: output.len(),
                    });
                }
                // Sparse values are selected from the prepared nonzero
                // pattern, so zero entries never incur evaluator calls.
                telemetry.measure_callback(RadauCallbackStage::JacobianEvaluation, || {
                    evaluate_sparse_entries(
                        jacobian,
                        &self.jacobian_pattern,
                        self.dimension,
                        output,
                        &workspace.args,
                        self.execution_policy,
                    );
                });
                let result = telemetry
                    .measure_callback(RadauCallbackStage::JacobianOutputAssembly, || {
                        validate_callback_output(RadauStage::Jacobian, output.len(), output)
                    });
                if result.is_ok() {
                    telemetry.count_output_writes(output.len());
                }
                result
            }
            RadauMatrixLayout::Banded { lower, upper } => {
                let pattern = RadauJacobianPattern::banded(self.dimension, lower, upper)?;
                let expected = pattern.values_len(layout);
                if output.len() != expected {
                    return Err(RadauError::ShapeMismatch {
                        stage: RadauStage::Jacobian,
                        expected,
                        actual: output.len(),
                    });
                }
                if self
                    .jacobian_pattern
                    .iter()
                    .any(|&(row, col)| !pattern.is_in_layout(layout, row, col))
                {
                    return Err(RadauUnsupportedRoute::JacobianOutsideBandedLayout.into());
                }
                // Banded output is compact storage.  We evaluate only entries
                // in the requested band and leave structural zeros explicit;
                // no dense-to-banded conversion is performed afterward.
                output.fill(0.0);
                let entries = pattern.entries().to_vec();
                telemetry.measure_callback(RadauCallbackStage::JacobianEvaluation, || {
                    evaluate_banded_entries(
                        jacobian,
                        &entries,
                        self.dimension,
                        layout,
                        output,
                        &workspace.args,
                        self.execution_policy,
                    )
                })?;
                let result = telemetry
                    .measure_callback(RadauCallbackStage::JacobianOutputAssembly, || {
                        validate_callback_output(RadauStage::Jacobian, output.len(), output)
                    });
                if result.is_ok() {
                    telemetry.count_output_writes(output.len());
                }
                result
            }
        }
    }

    fn fill_args(
        &self,
        t: f64,
        state: &[f64],
        parameters: &[f64],
        workspace: &mut LambdifyWorkspace,
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
        // Keep argument binding centralized.  It is intentionally measured as
        // its own callback stage because continuation performance depends on
        // this copy cost independently of evaluator cost.
        workspace.args[0] = t;
        workspace.args[1..1 + self.parameter_count].copy_from_slice(parameters);
        workspace.args[1 + self.parameter_count..].copy_from_slice(state);
        Ok(())
    }
}

fn evaluate_scalar_entries(
    evaluators: &[ScalarLambdified],
    output: &mut [f64],
    args: &[f64],
    policy: IvpLambdifyExecutionPolicy,
) {
    if policy.should_parallel_with_tasks(evaluators.len(), evaluators.len()) {
        use rayon::prelude::*;
        evaluators
            .par_iter()
            .zip(output.par_iter_mut())
            .for_each(|(evaluator, slot)| *slot = evaluator(args));
    } else {
        for (slot, evaluator) in output.iter_mut().zip(evaluators) {
            *slot = evaluator(args);
        }
    }
}

fn evaluate_sparse_entries(
    jacobian: &[ScalarLambdified],
    pattern: &[(usize, usize)],
    dimension: usize,
    output: &mut [f64],
    args: &[f64],
    policy: IvpLambdifyExecutionPolicy,
) {
    if policy.should_parallel_with_tasks(pattern.len(), pattern.len()) {
        use rayon::prelude::*;
        pattern
            .par_iter()
            .zip(output.par_iter_mut())
            .for_each(|(&(row, col), slot)| *slot = jacobian[row * dimension + col](args));
    } else {
        for (slot, &(row, col)) in output.iter_mut().zip(pattern) {
            *slot = jacobian[row * dimension + col](args);
        }
    }
}

fn evaluate_banded_entries(
    jacobian: &[ScalarLambdified],
    entries: &[(usize, usize)],
    dimension: usize,
    layout: RadauMatrixLayout,
    output: &mut [f64],
    args: &[f64],
    policy: IvpLambdifyExecutionPolicy,
) -> Result<(), RadauError> {
    let pattern = RadauJacobianPattern::banded(
        dimension,
        match layout {
            RadauMatrixLayout::Banded { lower, .. } => lower,
            _ => 0,
        },
        match layout {
            RadauMatrixLayout::Banded { upper, .. } => upper,
            _ => 0,
        },
    )?;
    if policy.should_parallel_with_tasks(entries.len(), entries.len()) {
        // Evaluate into disjoint temporary slots, then place them in compact
        // storage. The copy is explicit so parallel workers never alias output.
        use rayon::prelude::*;
        let values = entries
            .par_iter()
            .map(|&(row, col)| jacobian[row * dimension + col](args))
            .collect::<Vec<_>>();
        for (entry, value) in entries.iter().zip(values) {
            let compact = pattern
                .compact_slot(layout, entry.0, entry.1)
                .ok_or(RadauConfigError::InvalidBandedSlot)?;
            output[compact] = value;
        }
    } else {
        for entry in entries {
            let compact = pattern
                .compact_slot(layout, entry.0, entry.1)
                .ok_or(RadauConfigError::InvalidBandedSlot)?;
            let value = jacobian[entry.0 * dimension + entry.1](args);
            output[compact] = value;
        }
    }
    Ok(())
}

fn validate_expression_variables(expression: &Expr, names: &[&str]) -> Result<(), RadauError> {
    if expression
        .all_arguments_are_variables()
        .iter()
        .all(|variable| names.iter().any(|name| *name == variable))
    {
        Ok(())
    } else {
        Err(RadauConfigError::ExpressionVariableOutsideSchema.into())
    }
}
