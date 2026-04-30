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

    if let Some((num, den)) = decimal_f64_to_ratio(value) {
        return Atom::new_num(Coefficient::reduce(num, den));
    }

    let scale = 1_000_000i64;
    let num = (value * scale as f64).round() as i64;
    Atom::new_num(Coefficient::reduce(num, scale))
}

fn decimal_f64_to_ratio(value: f64) -> Option<(i64, i64)> {
    if !value.is_finite() {
        return None;
    }

    let s = format!("{value:.6e}");
    let (negative, unsigned) = if let Some(rest) = s.strip_prefix('-') {
        (true, rest)
    } else if let Some(rest) = s.strip_prefix('+') {
        (false, rest)
    } else {
        (false, s.as_str())
    };

    let (mantissa, exp10) = if let Some((m, e)) = unsigned.split_once(['e', 'E']) {
        (m, e.parse::<i32>().ok()?)
    } else {
        (unsigned, 0)
    };

    let (int_part, frac_part) = if let Some((i, f)) = mantissa.split_once('.') {
        (i, f)
    } else {
        (mantissa, "")
    };

    let mut digits = format!("{int_part}{frac_part}");
    while digits.ends_with('0') && digits.len() > 1 {
        digits.pop();
    }
    if digits.is_empty() {
        return None;
    }

    let frac_len =
        frac_part.len() as i32 - mantissa.chars().rev().take_while(|ch| *ch == '0').count() as i32;
    let scale_exp = frac_len - exp10;
    let mut numerator = digits.parse::<i128>().ok()?;
    let mut denominator = 1i128;

    if scale_exp >= 0 {
        denominator = pow10_i128(scale_exp as u32)?;
    } else {
        numerator = numerator.checked_mul(pow10_i128((-scale_exp) as u32)?)?;
    }

    if negative {
        numerator = -numerator;
    }

    let gcd = gcd_i128(numerator.unsigned_abs() as i128, denominator);
    numerator /= gcd;
    denominator /= gcd;

    let num = i64::try_from(numerator).ok()?;
    let den = i64::try_from(denominator).ok()?;
    Some((num, den))
}

fn pow10_i128(exp: u32) -> Option<i128> {
    let mut acc = 1i128;
    for _ in 0..exp {
        acc = acc.checked_mul(10)?;
    }
    Some(acc)
}

fn gcd_i128(mut a: i128, mut b: i128) -> i128 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a.abs()
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
