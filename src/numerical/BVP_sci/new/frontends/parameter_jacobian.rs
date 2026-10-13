//! Small Lambdify evaluator used for the relatively narrow `df/dp` block.
//!
//! AOT remains responsible for the hot residual and `df/dy` callbacks. This
//! plan compiles only nonzero parameter derivatives, avoiding both an FD
//! parameter sweep and a second full Lambdify residual/Jacobian frontend.

use std::sync::Arc;

use crate::numerical::BVP_sci::new::config::BvpSciAssembly;
use crate::numerical::BVP_sci::new::error::{BvpSciNewError, BvpSciStage};
use crate::numerical::BVP_sci::new::telemetry::BvpSciTelemetry;
use crate::symbolic::View::atom::Atom;
use crate::symbolic::View::evaluate::{FunctionMap, PreparedEvaluator, PreparedVariableContext};
use crate::symbolic::View::state::Symbol;
use crate::symbolic::symbolic_engine::Expr;

type ExprEvaluator = Arc<dyn Fn(&[f64]) -> f64 + Send + Sync>;

#[derive(Clone)]
enum Entries {
    Expr(Vec<(usize, usize, ExprEvaluator)>),
    Atom(Vec<(usize, usize, Arc<PreparedEvaluator>)>),
}

enum SymbolicEntries {
    Expr(Vec<(usize, usize, Expr)>),
    Atom(Vec<(usize, usize, Atom)>),
}

/// Prepared analytic `df/dp` for an AOT route, compiled with its selected
/// ExprLegacy or AtomView symbolic assembly and AOT's `[x, p..., y...]` ABI.
#[derive(Clone)]
pub struct BvpSciParameterJacobianPlan {
    dimension: usize,
    parameter_count: usize,
    entries: Entries,
    telemetry: BvpSciTelemetry,
}

impl BvpSciParameterJacobianPlan {
    pub fn prepare(
        assembly: BvpSciAssembly,
        equations: &[Expr],
        state_names: &[String],
        parameter_names: &[String],
        independent_name: &str,
        telemetry: BvpSciTelemetry,
    ) -> Result<Self, BvpSciNewError> {
        let n = equations.len();
        let k = parameter_names.len();
        if !matches!(
            assembly,
            BvpSciAssembly::ExprLegacy | BvpSciAssembly::AtomViewNative
        ) {
            return Err(BvpSciNewError::UnsupportedRoute(
                "AOT parameter Jacobian requires ExprLegacy or AtomView assembly".into(),
            ));
        }
        if n == 0 || state_names.len() != n {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::SymbolicPreparation,
                expected: state_names.len(),
                actual: n,
            });
        }

        // Keep argument order identical to generated AOT callbacks so the
        // caller-owned AOT argument buffer can be reused without shuffling.
        let mut argument_names = Vec::with_capacity(1 + k + n);
        argument_names.push(independent_name.to_owned());
        argument_names.extend(parameter_names.iter().cloned());
        argument_names.extend(state_names.iter().cloned());
        let unique_names = argument_names
            .iter()
            .collect::<std::collections::HashSet<_>>();
        if unique_names.len() != argument_names.len() {
            return Err(BvpSciNewError::InvalidConfiguration(
                "independent, state, and parameter names must be unique".into(),
            ));
        }

        let derivative_started = telemetry.start_timing();
        let symbolic_entries = match assembly {
            BvpSciAssembly::ExprLegacy => {
                let mut entries = Vec::new();
                for (row, equation) in equations.iter().enumerate() {
                    for (column, parameter) in parameter_names.iter().enumerate() {
                        let derivative = equation.diff(parameter).simplify();
                        if !derivative.is_zero() {
                            entries.push((row, column, derivative));
                        }
                    }
                }
                SymbolicEntries::Expr(entries)
            }
            BvpSciAssembly::AtomViewNative => {
                let atoms = equations
                    .iter()
                    .map(crate::symbolic::View::conversions::expr_to_atom)
                    .collect::<Vec<Atom>>();
                let symbols = argument_names
                    .iter()
                    .map(|name| Symbol::new(crate::wrap_symbol!(name.as_str())))
                    .collect::<Vec<_>>();
                let parameter_symbols = &symbols[1..1 + k];
                let mut entries = Vec::new();
                for (row, atom) in atoms.iter().enumerate() {
                    for (column, &symbol) in parameter_symbols.iter().enumerate() {
                        let derivative = atom
                            .as_view()
                            .try_derivative(symbol)
                            .map_err(|error| {
                                BvpSciNewError::SymbolicPreparation(format!(
                                    "AOT parameter derivative at equation {row}, parameter {column}: {error}"
                                ))
                            })?;
                        if !derivative.is_zero() {
                            entries.push((row, column, derivative));
                        }
                    }
                }
                SymbolicEntries::Atom(entries)
            }
            BvpSciAssembly::Numerical => unreachable!("rejected above"),
        };
        telemetry.record_symbolic_jacobian(derivative_started, (n * k) as u64);

        let evaluator_started = telemetry.start_timing();
        let entries = match symbolic_entries {
            SymbolicEntries::Expr(symbolic_entries) => {
                let refs = argument_names
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>();
                Entries::Expr(
                    symbolic_entries
                        .into_iter()
                        .map(|(row, column, derivative)| {
                            (
                                row,
                                column,
                                Arc::from(Expr::lambdify_borrowed_thread_safe(&derivative, &refs)),
                            )
                        })
                        .collect::<Vec<_>>(),
                )
            }
            SymbolicEntries::Atom(symbolic_entries) => {
                let symbols = argument_names
                    .iter()
                    .map(|name| Symbol::new(crate::wrap_symbol!(name.as_str())))
                    .collect::<Vec<_>>();
                let context = PreparedVariableContext::new(&symbols);
                let functions = FunctionMap::new();
                Entries::Atom(
                    symbolic_entries
                        .into_iter()
                        .map(|(row, column, derivative)| {
                            PreparedEvaluator::new_with_context(&derivative, &context, &functions)
                                .map(|evaluator| (row, column, Arc::new(evaluator)))
                                .map_err(BvpSciNewError::SymbolicPreparation)
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                )
            }
        };
        let entry_count = match &entries {
            Entries::Expr(entries) => entries.len(),
            Entries::Atom(entries) => entries.len(),
        };
        telemetry.record_jacobian_evaluator_compilation(evaluator_started, entry_count as u64);

        Ok(Self {
            dimension: n,
            parameter_count: k,
            entries,
            telemetry,
        })
    }

    pub fn nnz(&self) -> usize {
        match &self.entries {
            Entries::Expr(entries) => entries.len(),
            Entries::Atom(entries) => entries.len(),
        }
    }

    /// Evaluate into row-major `n x k` output, reusing the AOT ABI buffer.
    pub fn evaluate(
        &self,
        arguments: &mut [f64],
        output: &mut [f64],
    ) -> Result<(), BvpSciNewError> {
        let expected_args = 1 + self.parameter_count + self.dimension;
        if arguments.len() != expected_args {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::ArgumentBuffer,
                expected: expected_args,
                actual: arguments.len(),
            });
        }
        let expected_output = self
            .dimension
            .checked_mul(self.parameter_count)
            .ok_or_else(|| {
                BvpSciNewError::InvalidConfiguration("parameter Jacobian size overflow".into())
            })?;
        if output.len() != expected_output {
            return Err(BvpSciNewError::ShapeMismatch {
                stage: BvpSciStage::JacobianCallback,
                expected: expected_output,
                actual: output.len(),
            });
        }
        let started = self.telemetry.start_timing();
        output.fill(0.0);
        match &self.entries {
            Entries::Expr(entries) => {
                for (row, column, evaluator) in entries {
                    write_value(
                        *row,
                        *column,
                        self.parameter_count,
                        evaluator(arguments),
                        output,
                    )?;
                }
            }
            Entries::Atom(entries) => {
                for (row, column, evaluator) in entries {
                    let value = evaluator.evaluate(arguments).map_err(|message| {
                        BvpSciNewError::Callback {
                            stage: BvpSciStage::JacobianCallback,
                            message,
                        }
                    })?;
                    write_value(*row, *column, self.parameter_count, value, output)?;
                }
            }
        }
        self.telemetry.record_parameter_jacobian_callback(started);
        self.telemetry.record_output_writes(output.len());
        Ok(())
    }
}

fn write_value(
    row: usize,
    column: usize,
    parameter_count: usize,
    value: f64,
    output: &mut [f64],
) -> Result<(), BvpSciNewError> {
    if !value.is_finite() {
        return Err(BvpSciNewError::NonFinite {
            stage: BvpSciStage::JacobianCallback,
        });
    }
    output[row * parameter_count + column] = value;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_derivatives_match_for_expr_and_atom_routes() {
        let equations = [
            Expr::parse_expression("p*y0 + x"),
            Expr::parse_expression("y0 - y1 + p*x"),
        ];
        let states = ["y0".to_owned(), "y1".to_owned()];
        let parameters = ["p".to_owned()];
        for assembly in [BvpSciAssembly::ExprLegacy, BvpSciAssembly::AtomViewNative] {
            let plan = BvpSciParameterJacobianPlan::prepare(
                assembly,
                &equations,
                &states,
                &parameters,
                "x",
                BvpSciTelemetry::timings(),
            )
            .expect("parameter-only symbolic route should prepare");
            assert_eq!(plan.nnz(), 2);
            let mut arguments = [0.25, 3.0, 1.0, 2.0]; // AOT ABI: x, p, y0, y1.
            let mut output = [0.0; 2];
            plan.evaluate(&mut arguments, &mut output)
                .expect("parameter derivatives should evaluate");
            assert_eq!(output, [1.0, 0.25]);
        }
    }
}
