//! AtomViewNative Lambdify frontend.
//!
//! Expressions are accepted only at the boundary. They are converted once to
//! packed `Atom` values, differentiated on the Atom representation and then
//! compiled directly into evaluators. No Atom-to-Expr round-trip is performed.

use crate::numerical::BVP_sci::new::config::BvpSciExecutionPolicy;
use crate::numerical::BVP_sci::new::error::{BvpSciNewError, BvpSciStage};
use crate::numerical::BVP_sci::new::telemetry::{BvpSciTelemetry, BvpSciTelemetrySnapshot};
use crate::symbolic::codegen::CodegenIR::LinearBlock;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::View::atom::{Atom, AtomView};
use crate::symbolic::View::coefficient::CoefficientView;
use crate::symbolic::View::evaluate::{FunctionMap, PreparedEvaluator, PreparedVariableContext};
use crate::symbolic::View::jacobian::PreparedSparseAtomSystem;
use crate::symbolic::View::state::Symbol;
use crate::symbolic::View::CodegenIR_atom::Lowerer as AtomLowerer;
use std::cell::RefCell;
use std::collections::HashSet;
use std::sync::Arc;

/// Scratch registers for one allocation-free Atom IR callback on one thread.
///
/// Rayon workers get their own instance through TLS. Sequential continuation
/// reuses the same capacity for every residual and Jacobian call.
#[derive(Default)]
struct AtomLinearWorkspace {
    temps: Vec<f64>,
    values: Vec<f64>,
}

thread_local! {
    static ATOM_LINEAR_WORKSPACE: RefCell<AtomLinearWorkspace> =
        RefCell::new(AtomLinearWorkspace::default());
}

/// Prepared AtomView residual and pointwise Jacobian evaluators.
#[derive(Clone)]
pub struct AtomViewNativeLambdifyPlan {
    independent_name: String,
    state_names: Vec<String>,
    parameter_names: Vec<String>,
    // PreparedEvaluator owns its execution plan. Keep it behind Arc so that
    // cloning a prepared frontend remains cheap for continuation/restart
    // users, as it was with the previous callback closures.
    residuals: Vec<Arc<PreparedEvaluator>>,
    jacobian: Vec<(usize, usize, Arc<PreparedEvaluator>)>,
    /// Optional flat Atom IR blocks. They are present only when every
    /// residual/Jacobian expression is supported by the current lowering
    /// subset; otherwise the prepared evaluator path remains authoritative.
    linear_residual: Option<Arc<LinearBlock>>,
    linear_jacobian: Option<Arc<LinearBlock>>,
    jacobian_nnz: usize,
    telemetry: BvpSciTelemetry,
    execution_policy: BvpSciExecutionPolicy,
}

impl AtomViewNativeLambdifyPlan {
    /// Convert, differentiate and compile an Atom-native BVP RHS once.
    pub fn prepare(
        equations: &[Expr],
        state_names: &[String],
        parameter_names: &[String],
        independent_name: impl Into<String>,
        telemetry: BvpSciTelemetry,
    ) -> Result<Self, BvpSciNewError> {
        let independent_name = independent_name.into();
        validate_names(&independent_name, state_names, parameter_names)?;
        if equations.is_empty() || equations.len() != state_names.len() {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::SymbolicPreparation,
                expected: state_names.len(),
                actual: equations.len(),
            });
        }

        // The only Expr work allowed on this route is the input boundary
        // conversion. Everything after this point uses packed Atom values;
        // there is intentionally no Atom-to-Expr round trip.
        let started = telemetry.start_timing();
        let conversion_started = telemetry.start_timing();
        let atoms = equations
            .iter()
            .map(crate::symbolic::View::conversions::expr_to_atom)
            .collect::<Vec<Atom>>();
        let equation_count = atoms.len();
        telemetry.record_atom_conversion(conversion_started, atoms.len() as u64);

        // Dependency discovery and differentiation establish one stable
        // structural order. It is reused for every parameter rebind and every
        // Newton Jacobian refresh.
        let pattern_started = telemetry.start_timing();
        // Transfer the one freshly packed Atom graph into the prepared plan.
        // The slice-based compatibility constructor clones every Atom; using
        // the shared constructor here avoids charging continuation users for
        // a second full graph allocation before differentiation even starts.
        let prepared = PreparedSparseAtomSystem::from_shared_atoms_discovering_dependencies(
            Arc::from(atoms),
            state_names,
        );
        // Keep dependency discovery separate from the derivative walk below.
        // The old attribution recorded the whole nested operation as
        // `pattern_ms`, which made `pattern_ms` and `symbolic_jacobian_ms`
        // report the same work twice.
        let pattern_elapsed = pattern_started.map(|started| started.elapsed());
        let symbolic_started = telemetry.start_timing();
        let symbolic_entries = prepared
            .try_calc_sparse_jacobian_with_bandwidth(None)
            .map_err(|error| BvpSciNewError::SymbolicPreparation(error.to_string()))?;
        // Atom differentiation is a real symbolic-Jacobian stage even though
        // the structural helper also discovers the sparse pattern. Keep both
        // observations: `symbolic_jacobian_ms` answers how long derivation
        // took, while `pattern_ms` remains the inclusive structural scope.
        telemetry.record_symbolic_jacobian(
            symbolic_started,
            (equation_count * state_names.len()) as u64,
        );
        telemetry.record_pattern_elapsed(pattern_elapsed, symbolic_entries.len() as u64);

        let mut argument_names = Vec::with_capacity(1 + state_names.len() + parameter_names.len());
        argument_names.push(independent_name.clone());
        argument_names.extend(state_names.iter().cloned());
        argument_names.extend(parameter_names.iter().cloned());
        let symbols = argument_names
            .iter()
            .map(|name| Symbol::new(crate::wrap_symbol!(name.as_str())))
            .collect::<Vec<_>>();
        let context = PreparedVariableContext::new(symbols.as_slice());
        let function_map = FunctionMap::new();

        // Lower the complete callback families as shared straight-line IR.
        // This removes per-node interpreter dispatch and lets the callback
        // reuse registers without allocating a result vector. Unsupported
        // builtins/custom functions deliberately fall back to PreparedEvaluator.
        let lowering_started = telemetry.start_timing();
        let linear_residual = try_lower_linear_block(
            prepared
                .atoms()
                .iter()
                .map(Atom::as_view)
                .collect::<Vec<_>>(),
            &symbols,
        );
        let linear_jacobian = try_lower_linear_block(
            symbolic_entries
                .iter()
                .map(|entry| entry.value.as_view())
                .collect::<Vec<_>>(),
            &symbols,
        );
        telemetry.record_lowering(lowering_started);

        // Compile residuals and Jacobian entries from the same Atom context so
        // runtime argument binding has one ABI for both callback families.
        // Keep their materialization scopes separate: a combined number can
        // hide duplicated Jacobian work when comparing frontends.
        let residual_evaluator_started = telemetry.start_timing();
        let residuals = prepared
            .atoms()
            .iter()
            .map(|atom| {
                PreparedEvaluator::new_with_context(atom, &context, &function_map)
                    .map(Arc::new)
                    .map_err(BvpSciNewError::SymbolicPreparation)
            })
            .collect::<Result<Vec<_>, _>>()?;
        telemetry.record_residual_evaluator_compilation(
            residual_evaluator_started,
            residuals.len() as u64,
        );

        let jacobian_evaluator_started = telemetry.start_timing();
        let jacobian = symbolic_entries
            .into_iter()
            .map(|entry| {
                let evaluator =
                    PreparedEvaluator::new_with_context(&entry.value, &context, &function_map)
                        .map(Arc::new)
                        .map_err(BvpSciNewError::SymbolicPreparation)?;
                Ok((entry.row, entry.col, evaluator))
            })
            .collect::<Result<Vec<_>, BvpSciNewError>>()?;
        telemetry.record_jacobian_evaluator_compilation(
            jacobian_evaluator_started,
            jacobian.len() as u64,
        );
        telemetry.record_preparation(started);

        Ok(Self {
            independent_name,
            state_names: state_names.to_vec(),
            parameter_names: parameter_names.to_vec(),
            residuals,
            linear_residual,
            linear_jacobian,
            jacobian_nnz: jacobian.len(),
            jacobian,
            telemetry,
            execution_policy: BvpSciExecutionPolicy::Sequential,
        })
    }

    /// Change only warm callback dispatch; Atom preparation is unchanged.
    pub fn with_execution_policy(mut self, policy: BvpSciExecutionPolicy) -> Self {
        self.execution_policy = policy;
        self
    }

    /// Return the number of state equations in the packed Atom plan.
    pub fn dimension(&self) -> usize {
        self.state_names.len()
    }

    /// Return the number of runtime parameters accepted by the packed plan.
    pub fn parameter_dimension(&self) -> usize {
        self.parameter_names.len()
    }

    pub fn independent_name(&self) -> &str {
        &self.independent_name
    }

    pub fn jacobian_nnz(&self) -> usize {
        self.jacobian_nnz
    }

    pub fn telemetry(&self) -> &BvpSciTelemetry {
        &self.telemetry
    }

    pub fn telemetry_snapshot(&self) -> BvpSciTelemetrySnapshot {
        self.telemetry.snapshot()
    }

    /// Evaluate packed Atom residuals into caller-owned storage.
    pub fn evaluate_rhs(
        &self,
        x: f64,
        state: &[f64],
        parameters: &[f64],
        arguments: &mut [f64],
        output: &mut [f64],
    ) -> Result<(), BvpSciNewError> {
        self.fill_arguments(x, state, parameters, arguments)?;
        if output.len() != self.dimension() {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::ResidualCallback,
                expected: self.dimension(),
                actual: output.len(),
            });
        }
        let started = self.telemetry.start_timing();
        let evaluation_started = self.telemetry.start_timing();
        self.telemetry
            .record_dispatch(self.execution_policy, self.residuals.len());
        let evaluation = if self.residuals.len() > 1
            && self
                .execution_policy
                .should_parallel_with_tasks(self.residuals.len(), self.residuals.len())
        {
            use rayon::prelude::*;
            let worker_count = rayon::current_num_threads().max(1);
            let chunk_size = (self.residuals.len() + worker_count - 1)
                .checked_div(worker_count)
                .unwrap_or(1)
                .max(1);
            self.residuals
                .par_chunks(chunk_size)
                .zip(output.par_chunks_mut(chunk_size))
                .enumerate()
                .try_for_each(|(chunk_index, (evaluators, output))| {
                    PreparedEvaluator::evaluate_many_thread_local_flat(
                        evaluators.iter().map(Arc::as_ref),
                        arguments,
                        output,
                    )
                    .map_err(|(index, message)| (chunk_index * chunk_size + index, message))
                })
        } else if let Some(block) = &self.linear_residual {
            ATOM_LINEAR_WORKSPACE.with(|workspace| {
                let mut workspace = workspace.borrow_mut();
                workspace.temps.resize(block.num_temps, 0.0);
                block.eval_into_with_temps(arguments, &mut workspace.temps, output);
                Ok(())
            })
        } else {
            PreparedEvaluator::evaluate_many_thread_local_flat(
                self.residuals.iter().map(Arc::as_ref),
                arguments,
                output,
            )
            .map_err(|(index, message)| (index, message))
        };
        if let Err((index, message)) = evaluation {
            return Err(BvpSciNewError::Callback {
                stage: BvpSciStage::ResidualCallback,
                message: format!("Atom evaluator {index}: {message}"),
            });
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(BvpSciNewError::NonFinite {
                stage: BvpSciStage::ResidualCallback,
            });
        }
        self.telemetry
            .record_residual_evaluation(evaluation_started);
        self.telemetry.record_output_writes(output.len());
        self.telemetry.record_rhs(started);
        Ok(())
    }

    /// Evaluate the sparse Atom Jacobian directly into dense caller storage.
    /// Backend-specific sparse/banded packing will consume the same fixed
    /// `(row, col, evaluator)` plan without rebuilding symbolic structure.
    pub fn evaluate_jacobian_dense(
        &self,
        x: f64,
        state: &[f64],
        parameters: &[f64],
        arguments: &mut [f64],
        output: &mut [f64],
    ) -> Result<(), BvpSciNewError> {
        self.fill_arguments(x, state, parameters, arguments)?;
        let dimension = self.dimension();
        let expected = dimension
            .checked_mul(dimension)
            .ok_or_else(|| BvpSciNewError::InvalidConfiguration("Jacobian size overflow".into()))?;
        if output.len() != expected {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::JacobianCallback,
                expected,
                actual: output.len(),
            });
        }
        output.fill(0.0);
        let started = self.telemetry.start_timing();
        let evaluation_started = self.telemetry.start_timing();
        self.telemetry
            .record_dispatch(self.execution_policy, self.jacobian.len());
        // Fully dense plans are already in row-major order, so they can use
        // the same zero-temporary batch path. Sparse symbolic plans still use
        // the values-only method below and do not pay for dense output.
        let is_dense_ordered = self.jacobian.len() == expected
            && self
                .jacobian
                .iter()
                .enumerate()
                .all(|(index, (row, column, _))| *row * dimension + *column == index);
        if let Some(block) = &self.linear_jacobian {
            ATOM_LINEAR_WORKSPACE.with(|workspace| {
                let mut workspace = workspace.borrow_mut();
                workspace.temps.resize(block.num_temps, 0.0);
                if is_dense_ordered {
                    block.eval_into_with_temps(arguments, &mut workspace.temps, output);
                } else {
                    workspace.values.resize(self.jacobian.len(), 0.0);
                    let AtomLinearWorkspace { temps, values } = &mut *workspace;
                    block.eval_into_with_temps(arguments, temps, values);
                    for ((row, column, _), value) in
                        self.jacobian.iter().zip(values.iter().copied())
                    {
                        output[row * dimension + column] = value;
                    }
                }
            });
        } else if is_dense_ordered {
            let evaluation = PreparedEvaluator::evaluate_many_thread_local_flat(
                self.jacobian
                    .iter()
                    .map(|(_, _, evaluator)| evaluator.as_ref()),
                arguments,
                output,
            );
            if let Err((index, message)) = evaluation {
                return Err(BvpSciNewError::Callback {
                    stage: BvpSciStage::JacobianCallback,
                    message: format!("Atom evaluator {index}: {message}"),
                });
            }
        } else {
            let evaluation = PreparedEvaluator::evaluate_many_thread_local_flat_scatter(
                self.jacobian.iter().map(|(row, column, evaluator)| {
                    (*row * dimension + *column, evaluator.as_ref())
                }),
                arguments,
                output,
            );
            if let Err((index, message)) = evaluation {
                return Err(BvpSciNewError::Callback {
                    stage: BvpSciStage::JacobianCallback,
                    message: format!("Atom evaluator {index}: {message}"),
                });
            }
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(BvpSciNewError::NonFinite {
                stage: BvpSciStage::JacobianCallback,
            });
        }
        self.telemetry
            .record_jacobian_evaluation(evaluation_started);
        self.telemetry.record_output_writes(output.len());
        self.telemetry.record_jacobian(started);
        Ok(())
    }

    /// Evaluate only structural Jacobian values in the prepared Atom pattern.
    /// No dense matrix or temporary value vector is materialized.
    pub fn evaluate_jacobian_values(
        &self,
        x: f64,
        state: &[f64],
        parameters: &[f64],
        arguments: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), BvpSciNewError> {
        self.fill_arguments(x, state, parameters, arguments)?;
        if values.len() != self.jacobian_nnz {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::JacobianCallback,
                expected: self.jacobian_nnz,
                actual: values.len(),
            });
        }
        let started = self.telemetry.start_timing();
        let evaluation_started = self.telemetry.start_timing();
        self.telemetry
            .record_dispatch(self.execution_policy, self.jacobian.len());
        let evaluation = if self.jacobian.len() > 1
            && self
                .execution_policy
                .should_parallel_with_tasks(self.jacobian.len(), self.jacobian.len())
        {
            use rayon::prelude::*;
            let worker_count = rayon::current_num_threads().max(1);
            let chunk_size = (self.jacobian.len() + worker_count - 1)
                .checked_div(worker_count)
                .unwrap_or(1)
                .max(1);
            self.jacobian
                .par_chunks(chunk_size)
                .zip(values.par_chunks_mut(chunk_size))
                .enumerate()
                .try_for_each(|(chunk_index, (entries, output))| {
                    PreparedEvaluator::evaluate_many_thread_local_flat(
                        entries.iter().map(|(_, _, evaluator)| evaluator.as_ref()),
                        arguments,
                        output,
                    )
                    .map_err(|(index, message)| (chunk_index * chunk_size + index, message))
                })
        } else if let Some(block) = &self.linear_jacobian {
            ATOM_LINEAR_WORKSPACE.with(|workspace| {
                let mut workspace = workspace.borrow_mut();
                workspace.temps.resize(block.num_temps, 0.0);
                block.eval_into_with_temps(arguments, &mut workspace.temps, values);
                Ok(())
            })
        } else {
            PreparedEvaluator::evaluate_many_thread_local_flat(
                self.jacobian
                    .iter()
                    .map(|(_, _, evaluator)| evaluator.as_ref()),
                arguments,
                values,
            )
            .map_err(|(index, message)| (index, message))
        };
        if let Err((index, message)) = evaluation {
            return Err(BvpSciNewError::Callback {
                stage: BvpSciStage::JacobianCallback,
                message: format!("Atom evaluator {index}: {message}"),
            });
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(BvpSciNewError::NonFinite {
                stage: BvpSciStage::JacobianCallback,
            });
        }
        self.telemetry
            .record_jacobian_evaluation(evaluation_started);
        self.telemetry.record_output_writes(values.len());
        self.telemetry.record_jacobian(started);
        Ok(())
    }

    /// Fixed structural order used by sparse and banded native storage.
    pub fn jacobian_pattern(&self) -> impl ExactSizeIterator<Item = (usize, usize)> + '_ {
        self.jacobian.iter().map(|(row, column, _)| (*row, *column))
    }

    fn fill_arguments(
        &self,
        x: f64,
        state: &[f64],
        parameters: &[f64],
        arguments: &mut [f64],
    ) -> Result<(), BvpSciNewError> {
        let binding_started = self.telemetry.start_timing();
        if !x.is_finite() {
            return Err(BvpSciNewError::NonFinite {
                stage: BvpSciStage::CallbackInput,
            });
        }
        if state.len() != self.dimension() || parameters.len() != self.parameter_dimension() {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::CallbackInput,
                expected: 1 + self.dimension() + self.parameter_dimension(),
                actual: 1 + state.len() + parameters.len(),
            });
        }
        let expected = 1 + state.len() + parameters.len();
        if arguments.len() != expected {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::ArgumentBuffer,
                expected,
                actual: arguments.len(),
            });
        }
        arguments[0] = x;
        if !state.is_empty() {
            self.telemetry.record_copy();
        }
        arguments[1..1 + state.len()].copy_from_slice(state);
        if !parameters.is_empty() {
            self.telemetry.record_copy();
        }
        arguments[1 + state.len()..].copy_from_slice(parameters);
        self.telemetry.record_argument_binding(binding_started);
        Ok(())
    }
}

/// Try the allocation-free Atom straight-line backend without turning
/// unsupported symbolic constructs into a public preparation failure.
///
/// `CodegenIR_atom::Lowerer` intentionally uses internal assertions for
/// malformed or unsupported nodes because it is also used by code generation.
/// BVP frontend selection must be more forgiving, so the optional fast path is
/// isolated behind `catch_unwind` and the existing typed evaluator remains the
/// compatibility fallback.
fn try_lower_linear_block(
    views: Vec<AtomView<'_>>,
    symbols: &[Symbol],
) -> Option<Arc<LinearBlock>> {
    if views.is_empty()
        || views
            .iter()
            .any(|view| !supports_linear_lowering(*view, symbols))
    {
        return None;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let block = AtomLowerer::new(symbols).lower_many(&views);
        Arc::new(block.peephole_optimize())
    }))
    .ok()
}

/// Conservative capability check for the current Atom straight-line IR.
///
/// Keeping this check explicit avoids using panic-catching as normal feature
/// detection and means unsupported user functions remain quiet compatibility
/// fallbacks rather than producing diagnostic noise on stderr.
fn supports_linear_lowering(view: AtomView<'_>, symbols: &[Symbol]) -> bool {
    match view {
        AtomView::Num(number) => matches!(number.get_coeff_view(), CoefficientView::Natural(_, _)),
        AtomView::Var(variable) => symbols.contains(&variable.get_symbol()),
        AtomView::Pow(power) => {
            supports_linear_lowering(power.get_base(), symbols)
                && supports_linear_lowering(power.get_exp(), symbols)
        }
        AtomView::Mul(product) => product
            .iter()
            .all(|child| supports_linear_lowering(child, symbols)),
        AtomView::Add(sum) => sum
            .iter()
            .all(|child| supports_linear_lowering(child, symbols)),
        AtomView::Fun(function) => {
            function.get_nargs() == 1
                && matches!(
                    function.get_symbol(),
                    symbol if symbol == Atom::EXP
                        || symbol == Atom::LOG
                        || symbol == Atom::SIN
                        || symbol == Atom::COS
                        || symbol == Atom::TAN
                        || symbol == Atom::COT
                        || symbol == Atom::ASIN
                        || symbol == Atom::ACOS
                        || symbol == Atom::ATAN
                        || symbol == Atom::ACOT
                )
                && function
                    .iter()
                    .all(|child| supports_linear_lowering(child, symbols))
        }
    }
}

fn validate_names(
    independent_name: &str,
    state_names: &[String],
    parameter_names: &[String],
) -> Result<(), BvpSciNewError> {
    let mut seen = HashSet::new();
    if independent_name.is_empty() {
        return Err(BvpSciNewError::InvalidConfiguration(
            "independent variable name must not be empty".into(),
        ));
    }
    seen.insert(independent_name);
    for name in state_names.iter().chain(parameter_names) {
        if name.is_empty() || !seen.insert(name.as_str()) {
            return Err(BvpSciNewError::InvalidConfiguration(format!(
                "duplicate or empty symbolic name: {name:?}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{try_lower_linear_block, AtomViewNativeLambdifyPlan};
    use crate::numerical::BVP_sci::new::{BvpSciTelemetry, BvpSciTelemetryMode};
    use crate::symbolic::symbolic_engine::Expr;
    use crate::symbolic::View::atom::Atom;
    use crate::symbolic::View::conversions::expr_to_atom;
    use crate::symbolic::View::evaluate::{
        FunctionMap, PreparedEvaluator, PreparedVariableContext,
    };
    use crate::symbolic::View::jacobian::PreparedSparseAtomSystem;
    use crate::symbolic::View::state::Symbol;
    use crate::Utils::test_reporting::write_test_report;
    use std::sync::Arc;
    use std::time::Instant;
    use tabled::{Table, Tabled};

    #[derive(Tabled)]
    struct PreparationAbRow {
        workload: String,
        variant: String,
        base_ms: String,
        variant_ms: String,
        total_ms: String,
        residuals: usize,
        jacobian_nnz: usize,
        ir_blocks: usize,
        evaluator_count: usize,
        status: String,
    }

    #[derive(Clone, Copy)]
    enum PreparationAbVariant {
        EvaluatorOnly,
        IrOnly,
        DualPath,
    }

    impl PreparationAbVariant {
        fn label(self) -> &'static str {
            match self {
                Self::EvaluatorOnly => "EvaluatorOnly",
                Self::IrOnly => "IrOnly",
                Self::DualPath => "DualPath",
            }
        }
    }

    struct PreparedAbInputs {
        atoms: Arc<[Atom]>,
        system: PreparedSparseAtomSystem,
        jacobian: Vec<crate::symbolic::View::jacobian::SparseAtomJacobianEntry>,
        context: PreparedVariableContext,
        function_map: FunctionMap,
        symbols: Vec<Symbol>,
    }

    fn prepare_ab_inputs(
        equations: &[Expr],
        state_names: &[String],
        parameter_names: &[String],
    ) -> PreparedAbInputs {
        let atoms: Arc<[Atom]> = equations
            .iter()
            .map(expr_to_atom)
            .collect::<Vec<_>>()
            .into();
        let system = PreparedSparseAtomSystem::from_shared_atoms_discovering_dependencies(
            Arc::clone(&atoms),
            state_names,
        );
        let jacobian = system
            .try_calc_sparse_jacobian_with_bandwidth(None)
            .expect("A/B fixture differentiation should succeed");
        let mut argument_names = Vec::with_capacity(1 + state_names.len() + parameter_names.len());
        argument_names.push("x".to_string());
        argument_names.extend(state_names.iter().cloned());
        argument_names.extend(parameter_names.iter().cloned());
        let symbols = argument_names
            .iter()
            .map(|name| Symbol::new(crate::wrap_symbol!(name.as_str())))
            .collect::<Vec<_>>();
        let context = PreparedVariableContext::new(&symbols);
        PreparedAbInputs {
            atoms,
            system,
            jacobian,
            context,
            function_map: FunctionMap::new(),
            symbols,
        }
    }

    fn run_ab_variant(inputs: &PreparedAbInputs, variant: PreparationAbVariant) -> (usize, usize) {
        let mut ir_blocks = 0;
        let mut evaluator_count = 0;
        if matches!(
            variant,
            PreparationAbVariant::IrOnly | PreparationAbVariant::DualPath
        ) {
            let residual_views = inputs.atoms.iter().map(Atom::as_view).collect::<Vec<_>>();
            let jacobian_views = inputs
                .jacobian
                .iter()
                .map(|entry| entry.value.as_view())
                .collect::<Vec<_>>();
            assert!(try_lower_linear_block(residual_views, &inputs.symbols).is_some());
            assert!(try_lower_linear_block(jacobian_views, &inputs.symbols).is_some());
            ir_blocks = 2;
        }
        if matches!(
            variant,
            PreparationAbVariant::EvaluatorOnly | PreparationAbVariant::DualPath
        ) {
            let residuals = inputs
                .system
                .atoms()
                .iter()
                .map(|atom| {
                    PreparedEvaluator::new_with_context(
                        atom,
                        &inputs.context,
                        &inputs.function_map,
                    )
                    .expect("A/B residual evaluator compilation should succeed");
                })
                .count();
            let jacobian_evaluators = inputs
                .jacobian
                .iter()
                .map(|entry| {
                    PreparedEvaluator::new_with_context(
                        &entry.value,
                        &inputs.context,
                        &inputs.function_map,
                    )
                    .expect("A/B Jacobian evaluator compilation should succeed");
                })
                .count();
            evaluator_count = residuals + jacobian_evaluators;
        }
        (ir_blocks, evaluator_count)
    }

    fn median_ms(mut samples: Vec<f64>) -> f64 {
        samples.sort_by(f64::total_cmp);
        samples[samples.len() / 2]
    }

    fn ab_row(
        workload: &str,
        inputs: &PreparedAbInputs,
        base_ms: f64,
        variant: PreparationAbVariant,
        repeats: usize,
    ) -> PreparationAbRow {
        let mut variant_samples = Vec::with_capacity(repeats);
        let mut ir_blocks = 0;
        let mut evaluator_count = 0;
        for _ in 0..repeats {
            let variant_started = Instant::now();
            let counts = run_ab_variant(inputs, variant);
            variant_samples.push(variant_started.elapsed().as_secs_f64() * 1e3);
            ir_blocks = counts.0;
            evaluator_count = counts.1;
        }
        let variant_ms = median_ms(variant_samples);
        PreparationAbRow {
            workload: workload.into(),
            variant: variant.label().into(),
            base_ms: format!("{base_ms:.3}"),
            variant_ms: format!("{variant_ms:.3}"),
            total_ms: format!("{:.3}", base_ms + variant_ms),
            residuals: inputs.atoms.len(),
            jacobian_nnz: inputs.jacobian.len(),
            ir_blocks,
            evaluator_count,
            status: "ok".into(),
        }
    }

    #[test]
    fn atom_native_stays_packed_and_matches_expr_runtime() {
        let plan = AtomViewNativeLambdifyPlan::prepare(
            &[Expr::parse_expression("p*y + x")],
            &["y".into()],
            &["p".into()],
            "x",
            BvpSciTelemetry::timings(),
        )
        .expect("AtomView preparation should succeed");
        let mut arguments = [0.0; 3];
        let mut rhs = [0.0; 1];
        let mut jacobian = [0.0; 1];
        plan.evaluate_rhs(2.0, &[3.0], &[4.0], &mut arguments, &mut rhs)
            .expect("AtomView RHS callback should succeed");
        plan.evaluate_jacobian_dense(2.0, &[3.0], &[4.0], &mut arguments, &mut jacobian)
            .expect("AtomView Jacobian callback should succeed");

        assert_eq!(rhs, [14.0]);
        assert_eq!(jacobian, [4.0]);
        assert_eq!(plan.jacobian_nnz(), 1);
        assert!(plan.linear_residual.is_some());
        assert!(plan.linear_jacobian.is_some());
        let telemetry = plan.telemetry_snapshot();
        assert_eq!(telemetry.mode, BvpSciTelemetryMode::Timings);
        assert_eq!(telemetry.symbolic_jacobian_derivations, 1);
        assert!(telemetry.symbolic_jacobian_ms.is_some());
    }

    #[test]
    fn atom_native_preparation_representation_ab_story() {
        let repeats = std::env::var("BVP_SCI_ATOM_AB_REPEATS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(5)
            .max(3);
        // Atom differentiation uses Rayon. Warm the global pool before
        // sampling so its one-time bootstrap is not tied to one row.
        let _ = rayon::current_num_threads();
        let workloads = [
            (
                "stiff-coupled",
                vec![
                    Expr::parse_expression("-20*y0+10*y1+9"),
                    Expr::parse_expression("-40*y1+20*y2+18-20*x"),
                    Expr::parse_expression("-80*y2+77-240*x"),
                ],
                vec!["y0".into(), "y1".into(), "y2".into()],
                Vec::<String>::new(),
            ),
            (
                "combustion-like",
                vec![
                    Expr::parse_expression("q"),
                    Expr::parse_expression("-0.5*q + 0.2*exp(Teta)*C0"),
                    Expr::parse_expression("J0"),
                    Expr::parse_expression("-0.3*J0 - 0.5*C0 + 0.1*Teta"),
                    Expr::parse_expression("J1"),
                    Expr::parse_expression("-0.2*J1 - 0.3*C1 + 0.05*C0^2"),
                ],
                vec![
                    "Teta".into(),
                    "q".into(),
                    "C0".into(),
                    "J0".into(),
                    "C1".into(),
                    "J1".into(),
                ],
                Vec::<String>::new(),
            ),
        ];
        let mut rows = Vec::new();
        for (name, equations, states, parameters) in workloads {
            let _ = prepare_ab_inputs(&equations, &states, &parameters);
            let mut base_samples = Vec::with_capacity(repeats);
            for _ in 0..repeats {
                let base_started = Instant::now();
                let _ = prepare_ab_inputs(&equations, &states, &parameters);
                base_samples.push(base_started.elapsed().as_secs_f64() * 1e3);
            }
            let base_ms = median_ms(base_samples);
            let variant_inputs = prepare_ab_inputs(&equations, &states, &parameters);
            for variant in [
                PreparationAbVariant::EvaluatorOnly,
                PreparationAbVariant::IrOnly,
                PreparationAbVariant::DualPath,
            ] {
                rows.push(ab_row(name, &variant_inputs, base_ms, variant, repeats));
            }
        }
        let table = Table::new(&rows).to_string();
        let body = format!(
            "# BVP_sci AtomNative preparation A/B\n\n- variants: EvaluatorOnly, IrOnly, DualPath\n- repeats per measured phase: {repeats}; rows report medians\n- base includes Expr-to-Atom conversion, dependency discovery and Atom differentiation\n- variant includes only the selected executable representation\n- no mesh, Newton, callback or linear backend work is included\n- base preparation is sampled independently and is common to all three variants\n- timings are diagnostic samples, not a release performance baseline\n\n{}",
            table
        );
        let path = write_test_report(
            "BVP_sci_Lambdify_Story",
            "atom_native_preparation_representation_ab",
            &body,
        )
        .expect("write AtomNative preparation A/B report");
        println!("report={}", path.display());
        assert_eq!(rows.len(), 6);
    }
}
