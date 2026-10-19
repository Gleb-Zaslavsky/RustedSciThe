//! Debug-only numerical fidelity probes for the Expr-to-AtomView boundary.
//!
//! These helpers deliberately sit outside the evaluator hot path. They compare
//! the legacy `Expr` closure with the direct packed-Atom closure and expose the
//! numeric leaves retained by the packed representation. This is useful for
//! distinguishing evaluator round-off from an earlier conversion that has
//! already changed a finite coefficient into zero.

use super::super::symbolic_engine::Expr;
use super::{AtomView, coefficient::CoefficientView, conversions::expr_to_atom};

/// Numerical comparison of one `Expr` and its direct AtomView lowering.
#[derive(Debug, Clone)]
pub struct AtomViewFidelityReport {
    /// Stable source text of the expression under investigation.
    pub expression: String,
    /// Values returned by the two independent lambdify routes.
    pub expr_value: f64,
    pub atom_value: f64,
    /// Absolute and scale-normalized differences between those values.
    pub absolute_error: f64,
    pub relative_error: f64,
    /// Numeric leaves observed in the packed Atom representation.
    pub atom_numeric_constants: Vec<f64>,
    /// Number of packed numeric leaves that use an unsupported large format.
    pub unsupported_large_constants: usize,
}

/// Compare `Expr::lambdify1` with the direct AtomView Lambdify closure.
///
/// The caller supplies the same ordered variables and arguments to both
/// routes. This is intentionally a diagnostic API: it does not run a solver,
/// allocate telemetry, or make a performance claim.
pub fn compare_expr_and_atom(
    expression: &Expr,
    variables: &[&str],
    arguments: &[f64],
) -> Result<AtomViewFidelityReport, String> {
    if variables.len() != arguments.len() {
        return Err(format!(
            "diagnostic argument mismatch: {} variable(s), {} value(s)",
            variables.len(),
            arguments.len()
        ));
    }

    let expr_value = expression.lambdify1(variables)(arguments);
    let atom = expr_to_atom(expression);
    let atom_value = atom.lambdify_borrowed_thread_safe(variables)(arguments);
    let absolute_error = (atom_value - expr_value).abs();
    let relative_error = absolute_error / expr_value.abs().max(f64::MIN_POSITIVE);
    let mut atom_numeric_constants = Vec::new();
    let mut unsupported_large_constants = 0;
    collect_atom_constants(
        atom.as_view(),
        &mut atom_numeric_constants,
        &mut unsupported_large_constants,
    );

    Ok(AtomViewFidelityReport {
        expression: expression.to_string(),
        expr_value,
        atom_value,
        absolute_error,
        relative_error,
        atom_numeric_constants,
        unsupported_large_constants,
    })
}

fn collect_atom_constants(
    view: AtomView<'_>,
    constants: &mut Vec<f64>,
    unsupported_large_constants: &mut usize,
) {
    match view {
        AtomView::Num(num) => match num.get_coeff_view() {
            CoefficientView::Natural(numerator, denominator) => {
                constants.push(numerator as f64 / denominator as f64);
            }
            CoefficientView::Float(value) => constants.push(value),
            CoefficientView::Large(_) => *unsupported_large_constants += 1,
        },
        AtomView::Var(_) => {}
        AtomView::Fun(function) => {
            for child in function.iter() {
                collect_atom_constants(child, constants, unsupported_large_constants);
            }
        }
        AtomView::Pow(power) => {
            let (base, exponent) = power.get_base_exp();
            collect_atom_constants(base, constants, unsupported_large_constants);
            collect_atom_constants(exponent, constants, unsupported_large_constants);
        }
        AtomView::Mul(product) => {
            for child in product.iter() {
                collect_atom_constants(child, constants, unsupported_large_constants);
            }
        }
        AtomView::Add(sum) => {
            for child in sum.iter() {
                collect_atom_constants(child, constants, unsupported_large_constants);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::View::atom::Atom;

    const SMALL_COEFFICIENT: f64 = -0.000000000000000509024475;

    #[test]
    fn diagnostic_reports_small_constant_conversion_without_solver_involvement() {
        let expression = Expr::Const(SMALL_COEFFICIENT);
        let report = compare_expr_and_atom(&expression, &[], &[]).unwrap();

        assert_eq!(report.expr_value, SMALL_COEFFICIENT);
        assert!(
            report
                .atom_numeric_constants
                .iter()
                .any(|value| *value != 0.0),
            "packed Atom constants: {:?}; report={report:?}",
            report.atom_numeric_constants
        );
        assert!(
            report.atom_value != 0.0,
            "nonzero Expr::Const was erased before Atom evaluation: {report:?}"
        );
    }

    #[test]
    fn diagnostic_reports_small_coefficient_inside_power_expression() {
        let expression = Expr::parse_expression("-0.000000000000000509024475 * T ^ 4");
        let report = compare_expr_and_atom(&expression, &["T"], &[2500.0]).unwrap();
        let expected = SMALL_COEFFICIENT * 2500.0_f64.powi(4);

        assert!((report.expr_value - expected).abs() < 1e-15);
        assert!(
            report.atom_value != 0.0,
            "small coefficient disappeared from c*T^4: {report:?}"
        );
        assert!(
            (report.atom_value - expected).abs() < expected.abs() * 1e-12,
            "AtomView drifted beyond f64-level conversion tolerance: {report:?}"
        );
    }

    #[test]
    fn diagnostic_sweep_covers_both_signs_near_observed_boundary() {
        for coefficient in [
            5.09024475e-12,
            5.09024475e-13,
            5.09024475e-14,
            5.09024475e-15,
            SMALL_COEFFICIENT,
        ] {
            for coefficient in [coefficient, -coefficient] {
                let expression = Expr::Const(coefficient);
                let report = compare_expr_and_atom(&expression, &[], &[]).unwrap();
                assert!(
                    report.atom_value != 0.0,
                    "finite coefficient became zero: coefficient={coefficient:e}, report={report:?}"
                );
                assert!(
                    (report.atom_value - coefficient).abs() <= coefficient.abs() * 1e-12,
                    "coefficient fidelity failure: coefficient={coefficient:e}, report={report:?}"
                );
            }
        }
    }

    #[test]
    fn diagnostic_preserves_small_f64_during_packed_numeric_arithmetic() {
        let mut left = match Atom::new_f64(SMALL_COEFFICIENT) {
            Atom::Num(value) => value,
            other => panic!("expected numeric Atom, got {other:?}"),
        };
        let right = match Atom::new_f64(2.0) {
            Atom::Num(value) => value,
            other => panic!("expected numeric Atom, got {other:?}"),
        };

        let right_view = match right.as_view() {
            AtomView::Num(value) => value,
            other => panic!("expected numeric view, got {other:?}"),
        };
        left.mul(&right_view);
        let multiplied = match left.as_view() {
            AtomView::Num(value) => value.get_coeff_view(),
            other => panic!("expected numeric view, got {other:?}"),
        };
        assert_eq!(multiplied, CoefficientView::Float(SMALL_COEFFICIENT * 2.0));

        left.add(&right_view);
        let added = match left.as_view() {
            AtomView::Num(value) => value.get_coeff_view(),
            other => panic!("expected numeric view, got {other:?}"),
        };
        assert_eq!(added, CoefficientView::Float(SMALL_COEFFICIENT * 2.0 + 2.0));
    }

    #[test]
    fn diagnostic_does_not_panic_or_erase_overflowed_float_power() {
        let base = Atom::new_f64(1.0e200);
        let power = base.pow(Atom::new_num(2));
        let exact_power = Atom::new_num(10).pow(Atom::new_num(400));

        assert!(matches!(power.as_view(), AtomView::Pow(_)));
        assert!(power.lambdify_borrowed_thread_safe(&[])(&[]).is_infinite());
        assert!(matches!(exact_power.as_view(), AtomView::Pow(_)));
        assert!(exact_power.lambdify_borrowed_thread_safe(&[])(&[]).is_infinite());
    }

    fn assert_scalar_parity(
        source: &str,
        variables: &[&str],
        arguments: &[f64],
    ) -> AtomViewFidelityReport {
        let expression = Expr::parse_expression(source);
        let report = compare_expr_and_atom(&expression, variables, arguments).unwrap();
        let expr_value = report.expr_value;
        let atom_value = report.atom_value;

        if expr_value.is_nan() || atom_value.is_nan() {
            assert!(
                expr_value.is_nan() && atom_value.is_nan(),
                "NaN classification drift for {source:?}: {report:?}"
            );
        } else if expr_value.is_infinite() || atom_value.is_infinite() {
            assert_eq!(
                expr_value.is_infinite(),
                atom_value.is_infinite(),
                "infinity classification drift for {source:?}: {report:?}"
            );
            assert_eq!(
                expr_value.is_sign_positive(),
                atom_value.is_sign_positive(),
                "infinity sign drift for {source:?}: {report:?}"
            );
        } else {
            let scale = expr_value.abs().max(atom_value.abs()).max(1.0);
            assert!(
                (expr_value - atom_value).abs() <= scale * 2.0e-13,
                "finite AtomView drift for {source:?}: {report:?}"
            );
        }

        report
    }

    #[test]
    fn diagnostic_preserves_subnormal_and_minimum_normal_constants() {
        for coefficient in [
            f64::from_bits(1),
            f64::MIN_POSITIVE,
            -f64::from_bits(1),
            -f64::MIN_POSITIVE,
        ] {
            let source = format!("{coefficient:.17e}");
            let report = assert_scalar_parity(&source, &[], &[]);
            assert_eq!(report.expr_value, coefficient, "source={source}");
            assert_eq!(report.atom_value, coefficient, "source={source}");
        }
    }

    #[test]
    fn diagnostic_preserves_maximum_finite_and_mixed_magnitudes() {
        let max = f64::MAX;
        let max_source = format!("{max:.17e}");
        let max_report = assert_scalar_parity(&max_source, &[], &[]);
        assert_eq!(max_report.expr_value, max);
        assert_eq!(max_report.atom_value, max);

        let mixed = assert_scalar_parity("1e200*x + 1e-200*y", &["x", "y"], &[1.0, 1.0]);
        assert_eq!(mixed.expr_value, mixed.atom_value);
        assert!(mixed.atom_value.is_finite());
    }

    #[test]
    fn diagnostic_preserves_cancellation_and_small_terms_in_compositions() {
        for source in [
            "1 + 5.09024475e-16 - 1",
            "1e16 + 1e-16 - 1e16",
            "(1 + 5.09024475e-16)^2 - 1",
            "5.09024475e-16 / 2 + 5.09024475e-16 / 2",
        ] {
            let report = assert_scalar_parity(source, &[], &[]);
            assert_eq!(report.expr_value, report.atom_value, "source={source}");
        }
    }

    #[test]
    fn diagnostic_covers_signed_integer_fractional_and_reciprocal_powers() {
        for (source, variables, arguments) in [
            ("(-2)^3", &[][..], &[][..]),
            ("(-2)^4", &[][..], &[][..]),
            ("2^-3", &[][..], &[][..]),
            ("(1+x)^0.5", &["x"][..], &[3.0][..]),
            ("(1+x)^-0.5", &["x"][..], &[3.0][..]),
            ("(-2)^0.5", &[][..], &[][..]),
        ] {
            let report = assert_scalar_parity(source, variables, arguments);
            assert_eq!(
                report.expr_value.is_nan(),
                report.atom_value.is_nan(),
                "power domain classification drift for {source:?}: {report:?}"
            );
        }
    }

    #[test]
    fn diagnostic_covers_function_boundaries_and_nonfinite_classification() {
        for (source, variables, arguments) in [
            ("exp(-700*x)", &["x"][..], &[1.0][..]),
            ("exp(700*x)", &["x"][..], &[1.0][..]),
            ("log(x)", &["x"][..], &[f64::MIN_POSITIVE][..]),
            ("x^0.5", &["x"][..], &[f64::MIN_POSITIVE][..]),
            ("sin(1e200*x)", &["x"][..], &[1.0][..]),
            ("log(x)", &["x"][..], &[0.0][..]),
            ("x^0.5", &["x"][..], &[-1.0][..]),
        ] {
            assert_scalar_parity(source, variables, arguments);
        }
    }
}
