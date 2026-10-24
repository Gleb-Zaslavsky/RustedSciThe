//! Prepared Lambdify callbacks for the new Radau route.
//!
//! Expressions are compiled once. Evaluation reuses a session-owned argument
//! buffer and writes directly into caller-owned residual/Jacobian storage.
//! The flattened ABI follows LSODE2: `[t, parameters..., state...]`.

use crate::symbolic::ivp_telemetry::IvpLambdifyExecutionPolicy;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp::SparseSymbolicJacobian;

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
    // Compact row-major nonzero closures, aligned with `jacobian_pattern`.
    jacobian: Option<Vec<ScalarLambdified>>,
    jacobian_pattern: Vec<(usize, usize)>,
    jacobian_row_offsets: Vec<usize>,
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
        let jacobian_expressions = jacobian;
        // Compile only structurally nonzero entries. Large stiff systems often
        // have sparse symbolic Jacobians even when Dense remains the selected
        // linear algebra layout, so compiling closures for known zeros needlessly
        // multiplies preparation time and retained allocations by n^2.
        let nonzero_jacobian_entries = jacobian_expressions
            .as_ref()
            .map(|expressions| expressions.iter().filter(|entry| !entry.is_zero()).count())
            .unwrap_or_default();
        let mut jacobian_pattern = Vec::with_capacity(nonzero_jacobian_entries);
        let mut jacobian: Option<Vec<ScalarLambdified>> = jacobian_expressions
            .as_ref()
            .map(|_| Vec::with_capacity(nonzero_jacobian_entries));
        if let (Some(expressions), Some(compiled)) =
            (jacobian_expressions.as_ref(), jacobian.as_mut())
        {
            for (index, expression) in expressions.iter().enumerate() {
                if expression.is_zero() {
                    continue;
                }
                jacobian_pattern.push((index / dimension, index % dimension));
                compiled.push(Expr::lambdify_borrowed_thread_safe(expression, &name_refs));
            }
        }
        let mut jacobian_row_offsets = if jacobian.is_some() {
            vec![0; dimension + 1]
        } else {
            Vec::new()
        };
        for &(row, _) in &jacobian_pattern {
            jacobian_row_offsets[row + 1] += 1;
        }
        if jacobian.is_some() {
            for row in 0..dimension {
                jacobian_row_offsets[row + 1] += jacobian_row_offsets[row];
            }
        }

        Ok(Self {
            residual,
            jacobian,
            jacobian_pattern,
            jacobian_row_offsets,
            dimension,
            parameter_count: parameters.len(),
            execution_policy,
        })
    }

    /// Compile a coordinate-form Expr Jacobian without materializing `n*n`
    /// symbolic entries. Coordinates must already be in deterministic row-major
    /// order, as published by the shared IVP differentiation builder.
    pub(crate) fn from_sparse_expressions_with_policy(
        residual: Vec<Expr>,
        jacobian: SparseSymbolicJacobian,
        independent_variable: &str,
        variables: &[&str],
        parameters: &[&str],
        execution_policy: IvpLambdifyExecutionPolicy,
    ) -> Result<Self, RadauError> {
        let dimension = residual.len();
        if dimension == 0 {
            return Err(RadauConfigError::ZeroDimension.into());
        }
        if jacobian.rows != dimension || jacobian.cols != dimension {
            return Err(RadauError::ShapeMismatch {
                stage: RadauStage::Preparation,
                expected: dimension,
                actual: if jacobian.rows != dimension {
                    jacobian.rows
                } else {
                    jacobian.cols
                },
            });
        }
        if jacobian
            .entries
            .iter()
            .any(|&(row, col, _)| row >= dimension || col >= dimension)
        {
            return Err(RadauConfigError::JacobianPatternOutOfBounds.into());
        }

        let mut names = Vec::with_capacity(1 + variables.len() + parameters.len());
        names.push(independent_variable.to_owned());
        names.extend(parameters.iter().map(|name| (*name).to_owned()));
        names.extend(variables.iter().map(|name| (*name).to_owned()));
        let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
        for expression in residual
            .iter()
            .chain(jacobian.entries.iter().map(|(_, _, expression)| expression))
        {
            validate_expression_variables(expression, &name_refs)?;
        }

        let residual = residual
            .iter()
            .map(|expression| Expr::lambdify_borrowed_thread_safe(expression, &name_refs))
            .collect();
        let mut entries = jacobian.entries;
        entries.sort_unstable_by_key(|(row, col, _)| (*row, *col));
        if entries
            .windows(2)
            .any(|pair| (pair[0].0, pair[0].1) == (pair[1].0, pair[1].1))
        {
            return Err(RadauConfigError::DuplicateJacobianPatternEntry.into());
        }
        let mut jacobian_pattern = Vec::with_capacity(entries.len());
        let mut compiled = Vec::with_capacity(entries.len());
        let mut jacobian_row_offsets = vec![0; dimension + 1];
        for (row, col, expression) in entries {
            if expression.is_zero() {
                continue;
            }
            jacobian_pattern.push((row, col));
            compiled.push(Expr::lambdify_borrowed_thread_safe(&expression, &name_refs));
            jacobian_row_offsets[row + 1] += 1;
        }
        for row in 0..dimension {
            jacobian_row_offsets[row + 1] += jacobian_row_offsets[row];
        }

        Ok(Self {
            residual,
            jacobian: Some(compiled),
            jacobian_pattern,
            jacobian_row_offsets,
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

    #[cfg(test)]
    pub(crate) fn compiled_jacobian_entry_count(&self) -> usize {
        self.jacobian.as_ref().map_or(0, Vec::len)
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
            if jacobian.len() < output.len() {
                output.fill(0.0);
            }
            evaluate_dense_compact_entries(
                jacobian,
                &self.jacobian_pattern,
                &self.jacobian_row_offsets,
                self.dimension,
                output,
                &workspace.args,
                self.execution_policy,
            );
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
                telemetry.measure_callback(RadauCallbackStage::JacobianEvaluation, || {
                    evaluate_banded_entries(
                        jacobian,
                        &self.jacobian_pattern,
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
    output: &mut [f64],
    args: &[f64],
    policy: IvpLambdifyExecutionPolicy,
) {
    if policy.should_parallel_with_tasks(jacobian.len(), jacobian.len()) {
        use rayon::prelude::*;
        jacobian
            .par_iter()
            .zip(output.par_iter_mut())
            .for_each(|(evaluator, slot)| *slot = evaluator(args));
    } else {
        for (slot, evaluator) in output.iter_mut().zip(jacobian) {
            *slot = evaluator(args);
        }
    }
}

fn evaluate_dense_compact_entries(
    jacobian: &[ScalarLambdified],
    pattern: &[(usize, usize)],
    row_offsets: &[usize],
    dimension: usize,
    output: &mut [f64],
    args: &[f64],
    policy: IvpLambdifyExecutionPolicy,
) {
    if policy.should_parallel_with_tasks(dimension, jacobian.len()) {
        use rayon::prelude::*;
        output
            .par_chunks_mut(dimension)
            .zip(row_offsets.par_windows(2))
            .enumerate()
            .for_each(|(row, (values, bounds))| {
                for index in bounds[0]..bounds[1] {
                    values[pattern[index].1] = jacobian[index](args);
                }
                debug_assert!(
                    pattern[bounds[0]..bounds[1]]
                        .iter()
                        .all(|&(entry_row, _)| entry_row == row)
                );
            });
    } else {
        for (index, evaluator) in jacobian.iter().enumerate() {
            let (row, col) = pattern[index];
            output[row * dimension + col] = evaluator(args);
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
    let (lower, upper) = match layout {
        RadauMatrixLayout::Banded { lower, upper } => (lower, upper),
        _ => return Err(RadauConfigError::InvalidBandedSlot.into()),
    };
    let pattern = RadauJacobianPattern::banded(dimension, lower, upper)?;
    if policy.should_parallel_with_tasks(entries.len(), entries.len()) {
        // Compute only nonzero values into disjoint temporary slots, then
        // scatter them into compact band storage without aliasing output.
        use rayon::prelude::*;
        let values = entries
            .par_iter()
            .zip(jacobian.par_iter())
            .map(|(_, evaluator)| evaluator(args))
            .collect::<Vec<_>>();
        for (entry, value) in entries.iter().zip(values) {
            let compact = pattern
                .compact_slot(layout, entry.0, entry.1)
                .ok_or(RadauConfigError::InvalidBandedSlot)?;
            output[compact] = value;
        }
    } else {
        for (entry, evaluator) in entries.iter().zip(jacobian) {
            let compact = pattern
                .compact_slot(layout, entry.0, entry.1)
                .ok_or(RadauConfigError::InvalidBandedSlot)?;
            output[compact] = evaluator(args);
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
