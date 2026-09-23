//! Opt-in structural diagnostics for symbolic expressions.
//!
//! This module is intentionally outside the evaluation hot path. It may allocate
//! maps and formatted fingerprints because its purpose is to explain preparation
//! and callback differences on captured fixtures, not to run for every callback.

use super::super::symbolic_engine::Expr;
use super::{Atom, AtomView, coefficient::CoefficientView};
use std::collections::HashMap;

/// Classification of a numeric exponent for evaluator-cost investigations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PowerClass {
    /// Exponent is zero.
    IntegerZero,
    /// Exponent is one.
    IntegerOne,
    /// Exponent is another non-negative integer.
    IntegerPositive,
    /// Exponent is a negative integer.
    IntegerNegative,
    /// Exponent is finite but not an integer.
    Fractional,
    /// Exponent is not a finite numeric value.
    NonFinite,
    /// Exponent is symbolic or cannot be decoded from packed storage.
    NonNumeric,
}

/// Counts of logical operations in one expression collection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OperationCounts {
    pub constants: usize,
    pub variables: usize,
    pub additions: usize,
    pub multiplications: usize,
    pub subtractions: usize,
    pub divisions: usize,
    pub powers: usize,
    pub functions: usize,
}

/// Structural metrics for one or more symbolic expressions.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExpressionMetrics {
    /// Number of root expressions included in the report.
    pub roots: usize,
    /// Number of nonzero roots when the caller filters structural zeros.
    pub nonzero_roots: usize,
    /// Number of expression nodes, counting repeated tree occurrences.
    pub tree_nodes: usize,
    /// Number of distinct formatted subexpression fingerprints.
    pub unique_subexpressions: usize,
    /// `tree_nodes - unique_subexpressions` when fingerprints are available.
    pub repeated_subexpressions: usize,
    /// Maximum root-to-leaf depth.
    pub max_depth: usize,
    /// Character count of formatted roots, for continuity with existing reports.
    pub serialized_chars: usize,
    pub operations: OperationCounts,
    pub power_integer_zero: usize,
    pub power_integer_one: usize,
    pub power_integer_positive: usize,
    pub power_integer_negative: usize,
    pub power_fractional: usize,
    pub power_non_finite: usize,
    pub power_non_numeric: usize,
    /// Function names are diagnostic output and are intentionally owned strings.
    pub function_kinds: HashMap<String, usize>,
}

impl ExpressionMetrics {
    /// Return a stable, allocation-only fingerprint for structural reports.
    ///
    /// This is deliberately not used by evaluators. It makes the distinction
    /// between an explicit `Div` and a negative `Pow` visible in story reports,
    /// while sorted function names keep reports reproducible across runs.
    pub fn operation_fingerprint(&self) -> String {
        let mut functions: Vec<_> = self.function_kinds.iter().collect();
        functions.sort_by(|(left, _), (right, _)| left.cmp(right));
        let functions = functions
            .into_iter()
            .map(|(name, count)| format!("{name}:{count}"))
            .collect::<Vec<_>>()
            .join(",");

        format!(
            "nodes={};unique={};repeated={};add={};mul={};sub={};div={};pow={};pow-1={};pow-frac={};functions=[{}]",
            self.tree_nodes,
            self.unique_subexpressions,
            self.repeated_subexpressions,
            self.operations.additions,
            self.operations.multiplications,
            self.operations.subtractions,
            self.operations.divisions,
            self.operations.powers,
            self.power_integer_negative,
            self.power_fractional,
            functions,
        )
    }

    fn finish(&mut self, fingerprints: HashMap<String, usize>) {
        self.unique_subexpressions = fingerprints.len();
        self.repeated_subexpressions = self.tree_nodes.saturating_sub(self.unique_subexpressions);
    }

    fn record_power(&mut self, class: PowerClass) {
        match class {
            PowerClass::IntegerZero => self.power_integer_zero += 1,
            PowerClass::IntegerOne => self.power_integer_one += 1,
            PowerClass::IntegerPositive => self.power_integer_positive += 1,
            PowerClass::IntegerNegative => self.power_integer_negative += 1,
            PowerClass::Fractional => self.power_fractional += 1,
            PowerClass::NonFinite => self.power_non_finite += 1,
            PowerClass::NonNumeric => self.power_non_numeric += 1,
        }
    }
}

/// Inspect an Expr collection outside the callback hot path.
pub fn inspect_exprs(expressions: &[Expr]) -> ExpressionMetrics {
    let mut metrics = ExpressionMetrics {
        roots: expressions.len(),
        ..ExpressionMetrics::default()
    };
    let mut fingerprints = HashMap::new();
    for expr in expressions {
        if !expr.is_zero() {
            metrics.nonzero_roots += 1;
        }
        metrics.serialized_chars += expr.to_string().len();
        visit_expr(expr, 1, &mut metrics, &mut fingerprints);
    }
    metrics.finish(fingerprints);
    metrics
}

/// Inspect packed Atom expressions without converting them to Expr.
pub fn inspect_atoms(atoms: &[Atom]) -> ExpressionMetrics {
    let mut metrics = ExpressionMetrics {
        roots: atoms.len(),
        ..ExpressionMetrics::default()
    };
    let mut fingerprints = HashMap::new();
    for atom in atoms {
        let view = atom.as_view();
        if !view.is_zero() {
            metrics.nonzero_roots += 1;
        }
        metrics.serialized_chars += view.to_string().len();
        visit_atom(view, 1, &mut metrics, &mut fingerprints);
    }
    metrics.finish(fingerprints);
    metrics
}

fn visit_expr(
    expr: &Expr,
    depth: usize,
    metrics: &mut ExpressionMetrics,
    fingerprints: &mut HashMap<String, usize>,
) {
    metrics.tree_nodes += 1;
    metrics.max_depth = metrics.max_depth.max(depth);
    *fingerprints.entry(format!("{expr:?}")).or_default() += 1;
    match expr {
        Expr::Const(_) => metrics.operations.constants += 1,
        Expr::Var(_) => metrics.operations.variables += 1,
        Expr::Add(left, right) => {
            metrics.operations.additions += 1;
            visit_expr(left, depth + 1, metrics, fingerprints);
            visit_expr(right, depth + 1, metrics, fingerprints);
        }
        Expr::Sub(left, right) => {
            metrics.operations.subtractions += 1;
            visit_expr(left, depth + 1, metrics, fingerprints);
            visit_expr(right, depth + 1, metrics, fingerprints);
        }
        Expr::Mul(left, right) => {
            metrics.operations.multiplications += 1;
            visit_expr(left, depth + 1, metrics, fingerprints);
            visit_expr(right, depth + 1, metrics, fingerprints);
        }
        Expr::Div(left, right) => {
            metrics.operations.divisions += 1;
            visit_expr(left, depth + 1, metrics, fingerprints);
            visit_expr(right, depth + 1, metrics, fingerprints);
        }
        Expr::Pow(base, exponent) => {
            metrics.operations.powers += 1;
            if let Expr::Const(value) = exponent.as_ref() {
                metrics.record_power(classify_power(*value));
            } else {
                metrics.record_power(PowerClass::NonNumeric);
            }
            visit_expr(base, depth + 1, metrics, fingerprints);
            visit_expr(exponent, depth + 1, metrics, fingerprints);
        }
        Expr::Exp(inner)
        | Expr::Ln(inner)
        | Expr::sin(inner)
        | Expr::cos(inner)
        | Expr::tg(inner)
        | Expr::ctg(inner)
        | Expr::arcsin(inner)
        | Expr::arccos(inner)
        | Expr::arctg(inner)
        | Expr::arcctg(inner) => {
            metrics.operations.functions += 1;
            let name = match expr {
                Expr::Exp(_) => "exp",
                Expr::Ln(_) => "ln",
                Expr::sin(_) => "sin",
                Expr::cos(_) => "cos",
                Expr::tg(_) => "tan",
                Expr::ctg(_) => "cot",
                Expr::arcsin(_) => "asin",
                Expr::arccos(_) => "acos",
                Expr::arctg(_) => "atan",
                Expr::arcctg(_) => "acot",
                _ => unreachable!(),
            };
            *metrics.function_kinds.entry(name.to_string()).or_default() += 1;
            visit_expr(inner, depth + 1, metrics, fingerprints);
        }
    }
}

fn visit_atom(
    view: AtomView<'_>,
    depth: usize,
    metrics: &mut ExpressionMetrics,
    fingerprints: &mut HashMap<String, usize>,
) {
    metrics.tree_nodes += 1;
    metrics.max_depth = metrics.max_depth.max(depth);
    *fingerprints.entry(format!("{view:?}")).or_default() += 1;
    match view {
        AtomView::Num(_) => metrics.operations.constants += 1,
        AtomView::Var(_) => metrics.operations.variables += 1,
        AtomView::Add(add) => {
            let args: Vec<_> = add.iter().collect();
            metrics.operations.additions += args.len().saturating_sub(1);
            for child in args {
                visit_atom(child, depth + 1, metrics, fingerprints);
            }
        }
        AtomView::Mul(mul) => {
            let args: Vec<_> = mul.iter().collect();
            metrics.operations.multiplications += args.len().saturating_sub(1);
            for child in args {
                visit_atom(child, depth + 1, metrics, fingerprints);
            }
        }
        AtomView::Pow(pow) => {
            metrics.operations.powers += 1;
            let (base, exponent) = pow.get_base_exp();
            metrics.record_power(match exponent {
                AtomView::Num(num) => match num.get_coeff_view() {
                    CoefficientView::Natural(numerator, denominator) => {
                        classify_power(numerator as f64 / denominator as f64)
                    }
                    CoefficientView::Large(_) => PowerClass::NonNumeric,
                },
                _ => PowerClass::NonNumeric,
            });
            visit_atom(base, depth + 1, metrics, fingerprints);
            visit_atom(exponent, depth + 1, metrics, fingerprints);
        }
        AtomView::Fun(function) => {
            metrics.operations.functions += 1;
            *metrics
                .function_kinds
                .entry(function.get_symbol().get_stripped_name().to_string())
                .or_default() += 1;
            for child in function.iter() {
                visit_atom(child, depth + 1, metrics, fingerprints);
            }
        }
    }
}

fn classify_power(value: f64) -> PowerClass {
    if !value.is_finite() {
        return PowerClass::NonFinite;
    }
    if value.fract() != 0.0 {
        return PowerClass::Fractional;
    }
    match value {
        0.0 => PowerClass::IntegerZero,
        1.0 => PowerClass::IntegerOne,
        value if value < 0.0 => PowerClass::IntegerNegative,
        _ => PowerClass::IntegerPositive,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::View::conversions::expr_to_atom;

    #[test]
    fn expr_and_atom_diagnostics_classify_the_same_power() {
        let expr = Expr::Pow(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Const(-1.0)),
        );
        let expr_metrics = inspect_exprs(std::slice::from_ref(&expr));
        let atom = expr_to_atom(&expr);
        let atom_metrics = inspect_atoms(std::slice::from_ref(&atom));

        assert_eq!(expr_metrics.power_integer_negative, 1);
        assert_eq!(atom_metrics.power_integer_negative, 1);
        assert_eq!(expr_metrics.operations.powers, 1);
        assert_eq!(atom_metrics.operations.powers, 1);
    }

    #[test]
    fn diagnostics_distinguish_repeated_subexpressions_from_node_count() {
        let x = Expr::Var("x".to_string());
        let repeated = Expr::Add(
            Box::new(Expr::Mul(Box::new(x.clone()), Box::new(x.clone()))),
            Box::new(Expr::Mul(Box::new(x), Box::new(Expr::Const(2.0)))),
        );
        let metrics = inspect_exprs(std::slice::from_ref(&repeated));

        assert!(metrics.tree_nodes > metrics.unique_subexpressions);
        assert!(metrics.repeated_subexpressions > 0);
    }

    #[test]
    fn operation_fingerprint_exposes_division_and_power_forms() {
        let expr = Expr::parse_expression("exp(x) / (1+x)^0.5");
        let metrics = inspect_exprs(std::slice::from_ref(&expr));
        let fingerprint = metrics.operation_fingerprint();

        assert!(fingerprint.contains("div=1"));
        assert!(fingerprint.contains("pow-frac=1"));
        assert!(fingerprint.contains("functions=[exp:1]"));
        assert!(fingerprint.contains("nodes="));
        assert!(fingerprint.contains("repeated="));
    }
}
