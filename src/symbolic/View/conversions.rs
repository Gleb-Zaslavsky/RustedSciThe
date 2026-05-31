//! Conversions between the packed [`Atom`] representation and the boxed [`Expr`] tree.

use super::super::symbolic_engine::Expr;
use super::{
    Atom,
    atom::{AtomView, FunctionBuilder},
    coefficient::{Coefficient, CoefficientView},
    state::{State, Symbol},
};

pub(crate) fn approximate_f64_atom(value: f64) -> Atom {
    let n = value as i64;
    if n as f64 == value {
        return Atom::new_num(n);
    }

    const MAX_COMPACT_DENOMINATOR: i64 = 1_000_000;
    const MAX_COMPACT_ERROR: f64 = 1.0e-12;
    if let Some((num, den)) = bounded_f64_to_ratio(value, MAX_COMPACT_DENOMINATOR) {
        let error = (num as f64 / den as f64 - value).abs();
        if num != 0 && error <= MAX_COMPACT_ERROR * value.abs().max(1.0) {
            return Atom::new_num(Coefficient::reduce(num, den));
        }
    }

    // Very small physical constants cannot be represented accurately under
    // the compact-denominator cap. Preserve their decimal magnitude rather
    // than allowing a compact approximation to collapse them to zero.
    if let Some((num, den)) = decimal_f64_to_ratio(value) {
        return Atom::new_num(Coefficient::reduce(num, den));
    }

    let scale = 1_000_000i64;
    let num = (value * scale as f64).round() as i64;
    Atom::new_num(Coefficient::reduce(num, scale))
}

/// Recovers a compact rational close to `value` without encoding the full
/// binary-float expansion into symbolic arithmetic.
///
/// Small recurring coefficients occur naturally after BVP discretization
/// (`0.01 / 0.000288 = 625 / 18`). Continued fractions preserve those
/// coefficients exactly while the denominator cap prevents nonlinear
/// products from acquiring impractically large rational storage.
fn bounded_f64_to_ratio(value: f64, max_denominator: i64) -> Option<(i64, i64)> {
    if !value.is_finite() {
        return None;
    }
    if max_denominator < 1 {
        return None;
    }

    let sign = if value < 0.0 { -1i128 } else { 1i128 };
    let target = value.abs();
    if target > i64::MAX as f64 {
        return None;
    }

    let max_denominator = max_denominator as i128;
    let mut x = target;
    let mut prev_num = 0i128;
    let mut prev_den = 1i128;
    let mut num = 1i128;
    let mut den = 0i128;

    loop {
        let whole = x.floor() as i128;
        let next_num = whole.checked_mul(num)?.checked_add(prev_num)?;
        let next_den = whole.checked_mul(den)?.checked_add(prev_den)?;

        if next_den > max_denominator {
            if den == 0 {
                return None;
            }
            let scale = (max_denominator - prev_den) / den;
            let bound_num = scale.checked_mul(num)?.checked_add(prev_num)?;
            let bound_den = scale.checked_mul(den)?.checked_add(prev_den)?;
            let current_error = (target - num as f64 / den as f64).abs();
            let bound_error = (target - bound_num as f64 / bound_den as f64).abs();
            if bound_error < current_error {
                num = bound_num;
                den = bound_den;
            }
            break;
        }

        prev_num = num;
        prev_den = den;
        num = next_num;
        den = next_den;

        let fractional = x - whole as f64;
        if fractional.abs() <= f64::EPSILON {
            break;
        }
        x = fractional.recip();
    }

    let num = i64::try_from(sign.checked_mul(num)?).ok()?;
    let den = i64::try_from(den).ok()?;
    Some((num, den))
}

fn decimal_f64_to_ratio(value: f64) -> Option<(i64, i64)> {
    if !value.is_finite() {
        return None;
    }

    let formatted = format!("{value:.6e}");
    let (mantissa, exponent) = formatted.split_once('e')?;
    let exponent = exponent.parse::<i32>().ok()?;
    let negative = mantissa.starts_with('-');
    let mantissa = mantissa.trim_start_matches('-');
    let (whole, fractional) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let digits = format!("{whole}{fractional}").parse::<i128>().ok()?;
    let mut numerator = if negative { -digits } else { digits };
    let mut denominator = pow10_i128(fractional.len() as u32)?;

    if exponent >= 0 {
        numerator = numerator.checked_mul(pow10_i128(exponent as u32)?)?;
    } else {
        denominator = denominator.checked_mul(pow10_i128((-exponent) as u32)?)?;
    }

    let divisor = gcd_i128(numerator.abs(), denominator);
    numerator /= divisor;
    denominator /= divisor;
    Some((
        i64::try_from(numerator).ok()?,
        i64::try_from(denominator).ok()?,
    ))
}

fn pow10_i128(power: u32) -> Option<i128> {
    10i128.checked_pow(power)
}

fn gcd_i128(mut lhs: i128, mut rhs: i128) -> i128 {
    while rhs != 0 {
        let remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    lhs.max(1)
}
// ── Atom → Expr ───────────────────────────────────────────────────────────────

/// Convert a packed [`Atom`] into a classical boxed [`Expr`] tree.
pub fn atom_to_expr(atom: &Atom) -> Expr {
    view_to_expr(atom.as_view())
}

fn view_to_expr(view: AtomView<'_>) -> Expr {
    match view {
        AtomView::Num(n) => match n.get_coeff_view() {
            CoefficientView::Natural(num, den) => {
                if den == 1 {
                    Expr::Const(num as f64)
                } else {
                    Expr::Div(
                        Box::new(Expr::Const(num as f64)),
                        Box::new(Expr::Const(den as f64)),
                    )
                }
            }
            CoefficientView::Large(_) => panic!("Large coefficients not supported"),
        },
        AtomView::Var(v) => {
            let sym = v.get_symbol();
            let name = sym.get_stripped_name().to_string();
            Expr::Var(name)
        }
        AtomView::Fun(f) => {
            let sym = f.get_symbol();
            let mut args: Vec<Expr> = f.iter().map(view_to_expr).collect();
            match sym.get_id() {
                id if id == State::EXP.get_id() => Expr::Exp(Box::new(args.remove(0))),
                id if id == State::LOG.get_id() => Expr::Ln(Box::new(args.remove(0))),
                id if id == State::SIN.get_id() => Expr::sin(Box::new(args.remove(0))),
                id if id == State::COS.get_id() => Expr::cos(Box::new(args.remove(0))),
                id if id == State::TAN.get_id() => Expr::tg(Box::new(args.remove(0))),
                id if id == State::COT.get_id() => Expr::ctg(Box::new(args.remove(0))),
                id if id == State::ASIN.get_id() => Expr::arcsin(Box::new(args.remove(0))),
                id if id == State::ACOS.get_id() => Expr::arccos(Box::new(args.remove(0))),
                id if id == State::ATAN.get_id() => Expr::arctg(Box::new(args.remove(0))),
                id if id == State::ACOT.get_id() => Expr::arcctg(Box::new(args.remove(0))),
                _ => panic!("Unsupported function: {}", sym.get_stripped_name()),
            }
        }
        AtomView::Pow(p) => {
            let (base, exp) = p.get_base_exp();
            Expr::Pow(Box::new(view_to_expr(base)), Box::new(view_to_expr(exp)))
        }
        AtomView::Mul(m) => {
            let mut it = m.iter();
            let first = view_to_expr(it.next().unwrap());
            it.fold(first, |acc, v| {
                Expr::Mul(Box::new(acc), Box::new(view_to_expr(v)))
            })
        }
        AtomView::Add(a) => {
            let mut it = a.iter();
            let first = view_to_expr(it.next().unwrap());
            it.fold(first, |acc, v| {
                Expr::Add(Box::new(acc), Box::new(view_to_expr(v)))
            })
        }
    }
}

// ── Expr → Atom ───────────────────────────────────────────────────────────────

/// Convert a classical boxed [`Expr`] tree into a packed [`Atom`].
pub fn expr_to_atom(expr: &Expr) -> Atom {
    match expr {
        Expr::Var(name) => {
            let sym = Symbol::new(crate::wrap_symbol!(name.as_str()));
            Atom::new_var(sym)
        }
        Expr::Const(v) => approximate_f64_atom(*v),
        Expr::Add(l, r) => expr_to_atom(l) + expr_to_atom(r),
        Expr::Sub(l, r) => expr_to_atom(l) - expr_to_atom(r),
        Expr::Mul(l, r) => expr_to_atom(l) * expr_to_atom(r),
        Expr::Div(l, r) => expr_to_atom(l) / expr_to_atom(r),
        Expr::Pow(b, e) => expr_to_atom(b).pow(expr_to_atom(e)),
        Expr::Exp(x) => FunctionBuilder::new(Atom::EXP)
            .add_arg(expr_to_atom(x))
            .finish(),
        Expr::Ln(x) => FunctionBuilder::new(Atom::LOG)
            .add_arg(expr_to_atom(x))
            .finish(),
        Expr::sin(x) => FunctionBuilder::new(Atom::SIN)
            .add_arg(expr_to_atom(x))
            .finish(),
        Expr::cos(x) => FunctionBuilder::new(Atom::COS)
            .add_arg(expr_to_atom(x))
            .finish(),
        Expr::tg(x) => FunctionBuilder::new(Atom::TAN)
            .add_arg(expr_to_atom(x))
            .finish(),
        Expr::ctg(x) => FunctionBuilder::new(Atom::COT)
            .add_arg(expr_to_atom(x))
            .finish(),
        Expr::arcsin(x) => FunctionBuilder::new(Atom::ASIN)
            .add_arg(expr_to_atom(x))
            .finish(),
        Expr::arccos(x) => FunctionBuilder::new(Atom::ACOS)
            .add_arg(expr_to_atom(x))
            .finish(),
        Expr::arctg(x) => FunctionBuilder::new(Atom::ATAN)
            .add_arg(expr_to_atom(x))
            .finish(),
        Expr::arcctg(x) => FunctionBuilder::new(Atom::ACOT)
            .add_arg(expr_to_atom(x))
            .finish(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parse, symbol};

    // ── helpers ───────────────────────────────────────────────────────────────

    /// Round-trip Expr → Atom → Expr and assert the final Expr matches `expected`.
    fn rt_expr(expr: Expr, expected: Expr) {
        let atom = expr_to_atom(&expr);
        let back = atom_to_expr(&atom);
        assert_eq!(back, expected, "round-trip failed for {:?}", expr);
    }

    /// Round-trip Atom → Expr → Atom and assert the final Atom matches the original.
    fn rt_atom(atom: &Atom) {
        let expr = atom_to_expr(atom);
        let back = expr_to_atom(&expr);
        assert_eq!(*atom, back, "round-trip failed for {}", atom);
    }

    // ── atom_to_expr ─────────────────────────────────────────────────────────

    #[test]
    fn num_integer() {
        let atom = Atom::new_num(42i64);
        assert_eq!(atom_to_expr(&atom), Expr::Const(42.0));
    }

    #[test]
    fn num_rational() {
        let atom = Atom::new_num(Coefficient::reduce(1, 3));
        assert_eq!(
            atom_to_expr(&atom),
            Expr::Div(Box::new(Expr::Const(1.0)), Box::new(Expr::Const(3.0)))
        );
    }

    #[test]
    fn var_roundtrip() {
        let x = symbol!("x");
        let atom = Atom::new_var(x);
        assert_eq!(atom_to_expr(&atom), Expr::Var("x".into()));
    }

    #[test]
    fn add_two_vars() {
        let atom = parse!("x+y").unwrap();
        assert_eq!(
            atom_to_expr(&atom),
            Expr::Add(
                Box::new(Expr::Var("x".into())),
                Box::new(Expr::Var("y".into()))
            )
        );
    }

    #[test]
    fn mul_two_vars() {
        let atom = parse!("x*y").unwrap();
        assert_eq!(
            atom_to_expr(&atom),
            Expr::Mul(
                Box::new(Expr::Var("x".into())),
                Box::new(Expr::Var("y".into()))
            )
        );
    }

    /// x - y normalizes to x + y*-1 in the packed form.
    #[test]
    fn sub_becomes_add_neg() {
        let x = symbol!("x");
        let y = symbol!("y");
        let atom = Atom::new_var(x) - Atom::new_var(y);
        assert_eq!(
            atom_to_expr(&atom),
            Expr::Add(
                Box::new(Expr::Var("x".into())),
                Box::new(Expr::Mul(
                    Box::new(Expr::Var("y".into())),
                    Box::new(Expr::Const(-1.0))
                ))
            )
        );
    }

    /// x / y normalizes to x * y^-1 in the packed form.
    #[test]
    fn div_becomes_mul_pow_neg1() {
        let x = symbol!("x");
        let y = symbol!("y");
        let atom = Atom::new_var(x) / Atom::new_var(y);
        assert_eq!(
            atom_to_expr(&atom),
            Expr::Mul(
                Box::new(Expr::Var("x".into())),
                Box::new(Expr::Pow(
                    Box::new(Expr::Var("y".into())),
                    Box::new(Expr::Const(-1.0))
                ))
            )
        );
    }

    #[test]
    fn pow_var_int() {
        let atom = parse!("x^2").unwrap();
        assert_eq!(
            atom_to_expr(&atom),
            Expr::Pow(Box::new(Expr::Var("x".into())), Box::new(Expr::Const(2.0)))
        );
    }

    #[test]
    fn scaled_sum_2x_plus_3y() {
        // normalizes to x*2 + y*3
        let atom = parse!("2*x+3*y").unwrap();
        assert_eq!(
            atom_to_expr(&atom),
            Expr::Add(
                Box::new(Expr::Mul(
                    Box::new(Expr::Var("x".into())),
                    Box::new(Expr::Const(2.0))
                )),
                Box::new(Expr::Mul(
                    Box::new(Expr::Var("y".into())),
                    Box::new(Expr::Const(3.0))
                ))
            )
        );
    }

    #[test]
    fn trig_functions() {
        let atom = parse!("sin(x)+cos(y)").unwrap();
        assert_eq!(
            atom_to_expr(&atom),
            Expr::Add(
                Box::new(Expr::sin(Box::new(Expr::Var("x".into())))),
                Box::new(Expr::cos(Box::new(Expr::Var("y".into()))))
            )
        );
    }

    #[test]
    fn all_trig_and_inverse_trig() {
        for (src, expected) in [
            ("tan(x)", Expr::tg(Box::new(Expr::Var("x".into())))),
            ("cot(x)", Expr::ctg(Box::new(Expr::Var("x".into())))),
            ("asin(x)", Expr::arcsin(Box::new(Expr::Var("x".into())))),
            ("acos(x)", Expr::arccos(Box::new(Expr::Var("x".into())))),
            ("atan(x)", Expr::arctg(Box::new(Expr::Var("x".into())))),
            ("acot(x)", Expr::arcctg(Box::new(Expr::Var("x".into())))),
            ("exp(x)", Expr::Exp(Box::new(Expr::Var("x".into())))),
            ("log(x)", Expr::Ln(Box::new(Expr::Var("x".into())))),
        ] {
            let atom = parse!(src).unwrap();
            assert_eq!(atom_to_expr(&atom), expected, "failed for {}", src);
        }
    }

    // ── expr_to_atom ─────────────────────────────────────────────────────────

    #[test]
    fn expr_var_to_atom() {
        let expr = Expr::Var("x".into());
        let atom = expr_to_atom(&expr);
        let x = symbol!("x");
        assert_eq!(atom, Atom::new_var(x));
    }

    #[test]
    fn expr_const_integer_to_atom() {
        let expr = Expr::Const(7.0);
        assert_eq!(expr_to_atom(&expr), Atom::new_num(7i64));
    }

    #[test]
    fn expr_const_roundtrip_preserves_bvp_diffusion_coefficient_precision() {
        let value = 34.72222222222222;
        let back = atom_to_expr(&expr_to_atom(&Expr::Const(value))).eval_expression(&[], &[]);
        assert!(
            (back - value).abs() < 1.0e-13,
            "atom conversion changed a solver coefficient: input={value}, output={back}"
        );
    }

    #[test]
    fn expr_const_roundtrip_does_not_erase_small_combustion_scale() {
        let value = 9.0e-8;
        let back = atom_to_expr(&expr_to_atom(&Expr::Const(value))).eval_expression(&[], &[]);
        assert!(
            (back - value).abs() < 1.0e-15,
            "atom conversion erased or distorted a small physical constant: input={value}, output={back}"
        );
    }

    #[test]
    fn expr_add_to_atom() {
        let expr = Expr::Add(
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Var("y".into())),
        );
        assert_eq!(expr_to_atom(&expr), parse!("x+y").unwrap());
    }

    #[test]
    fn expr_sub_to_atom() {
        let expr = Expr::Sub(
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Var("y".into())),
        );
        // x - y normalizes to x + y*-1
        assert_eq!(expr_to_atom(&expr), parse!("x-y").unwrap());
    }

    #[test]
    fn expr_mul_to_atom() {
        let expr = Expr::Mul(
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Var("y".into())),
        );
        assert_eq!(expr_to_atom(&expr), parse!("x*y").unwrap());
    }

    #[test]
    fn expr_div_to_atom() {
        let expr = Expr::Div(
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Var("y".into())),
        );
        assert_eq!(expr_to_atom(&expr), parse!("x/y").unwrap());
    }

    #[test]
    fn expr_pow_to_atom() {
        let expr = Expr::Pow(Box::new(Expr::Var("x".into())), Box::new(Expr::Const(3.0)));
        assert_eq!(expr_to_atom(&expr), parse!("x^3").unwrap());
    }

    #[test]
    fn expr_trig_to_atom() {
        let x = Box::new(Expr::Var("x".into()));
        let cases: &[(Expr, &str)] = &[
            (Expr::sin(x.clone()), "sin(x)"),
            (Expr::cos(x.clone()), "cos(x)"),
            (Expr::tg(x.clone()), "tan(x)"),
            (Expr::ctg(x.clone()), "cot(x)"),
            (Expr::arcsin(x.clone()), "asin(x)"),
            (Expr::arccos(x.clone()), "acos(x)"),
            (Expr::arctg(x.clone()), "atan(x)"),
            (Expr::arcctg(x.clone()), "acot(x)"),
            (Expr::Exp(x.clone()), "exp(x)"),
            (Expr::Ln(x.clone()), "log(x)"),
        ];
        for (expr, src) in cases {
            assert_eq!(
                expr_to_atom(expr),
                parse!(src).unwrap(),
                "failed for {}",
                src
            );
        }
    }

    // ── round-trips ───────────────────────────────────────────────────────────

    #[test]
    fn roundtrip_atom_to_expr_simple_add() {
        rt_atom(&parse!("x+y").unwrap());
    }

    #[test]
    fn roundtrip_atom_to_expr_pow() {
        rt_atom(&parse!("x^2").unwrap());
    }

    #[test]
    fn roundtrip_atom_to_expr_nested_trig() {
        rt_atom(&parse!("sin(x^2+1)").unwrap());
    }

    #[test]
    fn roundtrip_expr_to_atom_add() {
        rt_expr(
            Expr::Add(
                Box::new(Expr::Var("x".into())),
                Box::new(Expr::Var("y".into())),
            ),
            Expr::Add(
                Box::new(Expr::Var("x".into())),
                Box::new(Expr::Var("y".into())),
            ),
        );
    }

    /// Sub normalizes away: the round-trip produces Add(x, Mul(y, -1)).
    #[test]
    fn roundtrip_expr_sub_normalizes() {
        let atom = expr_to_atom(&Expr::Sub(
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Var("y".into())),
        ));
        assert_eq!(
            atom_to_expr(&atom),
            Expr::Add(
                Box::new(Expr::Var("x".into())),
                Box::new(Expr::Mul(
                    Box::new(Expr::Var("y".into())),
                    Box::new(Expr::Const(-1.0))
                ))
            )
        );
    }

    /// Div normalizes away: the round-trip produces Mul(x, Pow(y, -1)).
    #[test]
    fn roundtrip_expr_div_normalizes() {
        let atom = expr_to_atom(&Expr::Div(
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Var("y".into())),
        ));
        assert_eq!(
            atom_to_expr(&atom),
            Expr::Mul(
                Box::new(Expr::Var("x".into())),
                Box::new(Expr::Pow(
                    Box::new(Expr::Var("y".into())),
                    Box::new(Expr::Const(-1.0))
                ))
            )
        );
    }

    #[test]
    fn roundtrip_complex_expression() {
        // sin(x^2 + 1) * cos(y) — atom → expr → atom must be stable
        rt_atom(&parse!("sin(x^2+1)*cos(y)").unwrap());
    }
}
