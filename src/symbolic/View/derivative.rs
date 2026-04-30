//! Symbolic differentiation over normalized packed atoms.
//!
//! Derivatives are computed directly over [`AtomView`] without constructing a separate
//! symbolic tree. The implementation mirrors the reduced algebra supported by this crate:
//! sums, products, powers, builtin unary functions, and symbolic `der(...)` tags for
//! custom functions. Intermediate expressions are built in a [`Workspace`] and normalized
//! before being exposed.

use std::ops::DerefMut;

use super::{
    atom::{Atom, AtomView},
    coefficient::Coefficient,
    state::{Symbol, Workspace},
};

/// Errors returned by symbolic differentiation when `der(...)` tags are malformed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DerivativeError {
    DerivativeTargetMustBeFunction,
    DerivativeOrdersMustBeNumeric,
}

impl std::fmt::Display for DerivativeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DerivativeError::DerivativeTargetMustBeFunction => {
                f.write_str("Last argument of der function must be a function")
            }
            DerivativeError::DerivativeOrdersMustBeNumeric => f.write_str(
                "Derivative function must contain numbers for all but the last position",
            ),
        }
    }
}

impl std::error::Error for DerivativeError {}

/// Differentiate an owned atom with respect to `x`.
pub fn derivative(atom: &Atom, x: Symbol) -> Atom {
    atom.derivative(x)
}

/// Try to differentiate an owned atom with respect to `x`.
pub fn try_derivative(atom: &Atom, x: Symbol) -> Result<Atom, DerivativeError> {
    atom.try_derivative(x)
}

impl Atom {
    /// Differentiate this owned atom with respect to `x`.
    pub fn derivative(&self, x: Symbol) -> Atom {
        self.try_derivative(x)
            .expect("derivative: malformed der(...) expression")
    }

    /// Try to differentiate this owned atom with respect to `x`.
    pub fn try_derivative(&self, x: Symbol) -> Result<Atom, DerivativeError> {
        self.as_view().try_derivative(x)
    }
}

impl<'a> AtomView<'a> {
    /// Differentiate this borrowed view with respect to `x`.
    pub fn derivative(&self, x: Symbol) -> Atom {
        self.try_derivative(x)
            .expect("derivative: malformed der(...) expression")
    }

    /// Try to differentiate this borrowed view with respect to `x`.
    pub fn try_derivative(&self, x: Symbol) -> Result<Atom, DerivativeError> {
        Workspace::get_local().with(|ws| {
            let mut out = ws.new_atom();
            self.try_derivative_with_ws_into(x, ws, &mut out)?;
            Ok(out.into_inner())
        })
    }

    #[allow(dead_code)]
    pub(crate) fn derivative_into(&self, x: Symbol, out: &mut Atom) -> bool {
        self.try_derivative_into(x, out)
            .expect("derivative: malformed der(...) expression")
    }

    #[allow(dead_code)]
    pub(crate) fn try_derivative_into(
        &self,
        x: Symbol,
        out: &mut Atom,
    ) -> Result<bool, DerivativeError> {
        Workspace::get_local().with(|ws| self.try_derivative_with_ws_into(x, ws, out))
    }

    pub(crate) fn try_derivative_with_ws_into(
        &self,
        x: Symbol,
        workspace: &Workspace,
        out: &mut Atom,
    ) -> Result<bool, DerivativeError> {
        Ok(match self {
            AtomView::Num(_) => {
                out.to_num(Coefficient::zero());
                false
            }
            AtomView::Var(v) => {
                if v.get_symbol() == x {
                    out.to_num(Coefficient::one());
                    true
                } else {
                    out.to_num(Coefficient::zero());
                    false
                }
            }
            AtomView::Fun(f_orig) => {
                let (to_derive, f, is_der) = if f_orig.get_symbol() == Atom::DERIVATIVE {
                    let to_derive = f_orig.iter().last().unwrap();
                    (
                        to_derive,
                        match to_derive {
                            AtomView::Fun(f) => f,
                            _ => Err(DerivativeError::DerivativeTargetMustBeFunction)?,
                        },
                        true,
                    )
                } else {
                    (*self, *f_orig, false)
                };

                let mut args_der = Vec::with_capacity(f.get_nargs());
                for (i, arg) in f.iter().enumerate() {
                    let mut arg_der = workspace.new_atom();
                    if arg.try_derivative_with_ws_into(x, workspace, &mut arg_der)? {
                        args_der.push((i, arg_der));
                    }
                }

                if args_der.is_empty() {
                    out.to_num(Coefficient::zero());
                    return Ok(false);
                }

                if f.get_nargs() == 1 {
                    let arg = f.iter().next().unwrap();
                    let mut fn_der = workspace.new_atom();
                    let recognized = match f.get_symbol() {
                        s if s == Atom::EXP => {
                            fn_der.set_from_view(self);
                            true
                        }
                        s if s == Atom::LOG => {
                            let mut minus_one = workspace.new_atom();
                            minus_one.to_num((-1).into());
                            fn_der.to_pow(arg, minus_one.as_view());
                            true
                        }
                        s if s == Atom::SIN => {
                            fn_der.to_fun(Atom::COS).add_arg(arg);
                            true
                        }
                        s if s == Atom::COS => {
                            let mut sin = workspace.new_atom();
                            sin.to_fun(Atom::SIN).add_arg(arg);

                            let mut minus_one = workspace.new_atom();
                            minus_one.to_num((-1).into());

                            let m = fn_der.to_mul();
                            m.extend(minus_one.as_view());
                            m.extend(sin.as_view());
                            true
                        }
                        s if s == Atom::TAN => {
                            let mut cos = workspace.new_atom();
                            cos.to_fun(Atom::COS).add_arg(arg);

                            let mut minus_two = workspace.new_atom();
                            minus_two.to_num((-2).into());

                            fn_der.to_pow(cos.as_view(), minus_two.as_view());
                            true
                        }
                        s if s == Atom::COT => {
                            let mut sin = workspace.new_atom();
                            sin.to_fun(Atom::SIN).add_arg(arg);

                            let mut minus_two = workspace.new_atom();
                            minus_two.to_num((-2).into());

                            let mut pow = workspace.new_atom();
                            pow.to_pow(sin.as_view(), minus_two.as_view());

                            let mut minus_one = workspace.new_atom();
                            minus_one.to_num((-1).into());

                            let m = fn_der.to_mul();
                            m.extend(minus_one.as_view());
                            m.extend(pow.as_view());
                            true
                        }
                        s if s == Atom::ASIN || s == Atom::ACOS => {
                            let mut one = workspace.new_atom();
                            one.to_num(Coefficient::one());

                            let mut two = workspace.new_atom();
                            two.to_num(2.into());

                            let mut arg_sq = workspace.new_atom();
                            arg_sq.to_pow(arg, two.as_view());

                            let mut inner = workspace.new_atom();
                            let add = inner.to_add();
                            if s == Atom::ASIN {
                                add.extend(one.as_view());
                                let mut minus_one = workspace.new_atom();
                                minus_one.to_num((-1).into());
                                let mut neg_arg_sq = workspace.new_atom();
                                let mul = neg_arg_sq.to_mul();
                                mul.extend(minus_one.as_view());
                                mul.extend(arg_sq.as_view());
                                add.extend(neg_arg_sq.as_view());
                            } else {
                                add.extend(arg_sq.as_view());
                                let mut minus_one = workspace.new_atom();
                                minus_one.to_num((-1).into());
                                add.extend(minus_one.as_view());
                            }

                            let mut minus_half = workspace.new_atom();
                            minus_half.to_num((-1, 2).into());

                            if s == Atom::ASIN {
                                fn_der.to_pow(inner.as_view(), minus_half.as_view());
                            } else {
                                let mut pow = workspace.new_atom();
                                pow.to_pow(inner.as_view(), minus_half.as_view());

                                let mut minus_one = workspace.new_atom();
                                minus_one.to_num((-1).into());
                                let m = fn_der.to_mul();
                                m.extend(minus_one.as_view());
                                m.extend(pow.as_view());
                            }
                            true
                        }
                        s if s == Atom::ATAN || s == Atom::ACOT => {
                            let mut one = workspace.new_atom();
                            one.to_num(Coefficient::one());

                            let mut two = workspace.new_atom();
                            two.to_num(2.into());

                            let mut arg_sq = workspace.new_atom();
                            arg_sq.to_pow(arg, two.as_view());

                            let mut denom = workspace.new_atom();
                            let add = denom.to_add();
                            add.extend(arg_sq.as_view());
                            add.extend(one.as_view());

                            let mut minus_one = workspace.new_atom();
                            minus_one.to_num((-1).into());

                            if s == Atom::ATAN {
                                fn_der.to_pow(denom.as_view(), minus_one.as_view());
                            } else {
                                let mut recip = workspace.new_atom();
                                recip.to_pow(denom.as_view(), minus_one.as_view());

                                let mut neg_one = workspace.new_atom();
                                neg_one.to_num((-1).into());
                                let m = fn_der.to_mul();
                                m.extend(neg_one.as_view());
                                m.extend(recip.as_view());
                            }
                            true
                        }
                        s if s == Atom::SQRT => {
                            let mut half = workspace.new_atom();
                            half.to_num((1, 2).into());

                            let mut minus_half = workspace.new_atom();
                            minus_half.to_num((-1, 2).into());

                            let mut pow = workspace.new_atom();
                            pow.to_pow(arg, minus_half.as_view());

                            let m = fn_der.to_mul();
                            m.extend(half.as_view());
                            m.extend(pow.as_view());
                            true
                        }
                        _ => false,
                    };

                    if recognized {
                        let (_, mut arg_der) = args_der.pop().unwrap();
                        if let Atom::Mul(m) = arg_der.deref_mut() {
                            m.extend(fn_der.as_view());
                            arg_der.as_view().normalize_with_ws(workspace, out);
                        } else {
                            let mut mul = workspace.new_atom();
                            let m = mul.to_mul();
                            m.extend(fn_der.as_view());
                            m.extend(arg_der.as_view());
                            mul.as_view().normalize_with_ws(workspace, out);
                        }

                        return Ok(true);
                    }
                }

                let mut add = workspace.new_atom();
                let a = add.to_add();
                let mut tag_fn = workspace.new_atom();
                let mut tmp_num = workspace.new_atom();
                let mut mul = workspace.new_atom();
                for (index, arg_der) in args_der {
                    let p = tag_fn.to_fun(Atom::DERIVATIVE);

                    if is_der {
                        for (i, x_orig) in f_orig.iter().take(f.get_nargs()).enumerate() {
                            if let AtomView::Num(nn) = x_orig {
                                let num = nn.get_coeff_view() + if i == index { 1 } else { 0 };
                                tmp_num.to_num(num);
                                p.add_arg(tmp_num.as_view());
                            } else {
                                Err(DerivativeError::DerivativeOrdersMustBeNumeric)?
                            }
                        }
                    } else {
                        for i in 0..f.get_nargs() {
                            tmp_num.to_num((if i == index { 1 } else { 0 }, 1).into());
                            p.add_arg(tmp_num.as_view());
                        }
                    }

                    p.add_arg(to_derive);

                    let m = mul.to_mul();
                    m.extend(tag_fn.as_view());
                    m.extend(arg_der.as_view());
                    mul.as_view().normalize_with_ws(workspace, out);

                    a.extend(mul.as_view());
                }

                add.as_view().normalize_with_ws(workspace, out);
                true
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();

                let mut exp_der = workspace.new_atom();
                let exp_der_non_zero =
                    exp.try_derivative_with_ws_into(x, workspace, &mut exp_der)?;

                let mut base_der = workspace.new_atom();
                let base_der_non_zero =
                    base.try_derivative_with_ws_into(x, workspace, &mut base_der)?;

                if !exp_der_non_zero && !base_der_non_zero {
                    out.to_num(0.into());
                    return Ok(false);
                }

                let mut exp_der_contrib = workspace.new_atom();

                if exp_der_non_zero {
                    let mut log_base = workspace.new_atom();
                    log_base.to_fun(Atom::LOG).add_arg(base);

                    if let Atom::Mul(m) = exp_der.deref_mut() {
                        m.extend(*self);
                        m.extend(log_base.as_view());
                        exp_der
                            .as_view()
                            .normalize_with_ws(workspace, &mut exp_der_contrib);
                    } else {
                        let mut mul = workspace.new_atom();
                        let m = mul.to_mul();
                        m.extend(*self);
                        m.extend(exp_der.as_view());
                        m.extend(log_base.as_view());
                        mul.as_view()
                            .normalize_with_ws(workspace, &mut exp_der_contrib);
                    }

                    if !base_der_non_zero {
                        out.set_from_view(&exp_der_contrib.as_view());
                        return Ok(true);
                    }
                }

                let mut mul_h = workspace.new_atom();
                let mul = mul_h.to_mul();
                mul.extend(base_der.as_view());

                let mut new_exp = workspace.new_atom();
                if let AtomView::Num(n) = exp {
                    mul.extend(exp);
                    new_exp.to_num(n.get_coeff_view() + -1);
                } else {
                    mul.extend(exp);
                    let ao = new_exp.to_add();
                    ao.extend(exp);

                    let mut minus_one = workspace.new_atom();
                    minus_one.to_num((-1).into());
                    ao.extend(minus_one.as_view());
                }

                let mut pow_h = workspace.new_atom();
                pow_h.to_pow(base, new_exp.as_view());
                mul.extend(pow_h.as_view());

                if exp_der_non_zero {
                    let mut add = workspace.new_atom();
                    let a = add.to_add();
                    a.extend(mul_h.as_view());
                    a.extend(exp_der_contrib.as_view());
                    add.as_view().normalize_with_ws(workspace, out);
                } else {
                    mul_h.as_view().normalize_with_ws(workspace, out);
                }

                true
            }
            AtomView::Mul(args) => {
                let mut add_h = workspace.new_atom();
                let add = add_h.to_add();
                let mut non_zero = false;

                for (index, arg) in args.iter().enumerate() {
                    let mut arg_der = workspace.new_atom();
                    if arg.try_derivative_with_ws_into(x, workspace, &mut arg_der)? {
                        let mut term = workspace.new_atom();
                        let m = term.to_mul();
                        m.extend(arg_der.as_view());
                        for (other_index, other_arg) in args.iter().enumerate() {
                            if other_index != index {
                                m.extend(other_arg);
                            }
                        }
                        add.extend(term.as_view());
                        non_zero = true;
                    }
                }

                if non_zero {
                    add_h.as_view().normalize_with_ws(workspace, out);
                    true
                } else {
                    out.to_num(0.into());
                    false
                }
            }
            AtomView::Add(args) => {
                let mut add_h = workspace.new_atom();
                let add = add_h.to_add();
                let mut non_zero = false;
                for arg in args.iter() {
                    let mut arg_der = workspace.new_atom();
                    if arg.try_derivative_with_ws_into(x, workspace, &mut arg_der)? {
                        add.extend(arg_der.as_view());
                        non_zero = true;
                    }
                }

                if non_zero {
                    add_h.as_view().normalize_with_ws(workspace, out);
                    true
                } else {
                    out.to_num(0.into());
                    false
                }
            }
        })
    }
}

#[cfg(test)]
mod test {
    use super::DerivativeError;
    use crate::symbolic::View::atom::Atom;
    use crate::{function, parse, symbol};

    #[test]
    fn derivative() {
        let v1 = symbol!("v1");
        let y = symbol!("y");
        let inputs = [
            "(1+2*v1)^(5+v1)",
            "log(2*v1) + exp(3*v1) + sin(4*v1) + cos(y*v1)",
            "f(v1^2,v1)",
            "der(0,1,f(v1,v1^3))",
        ];
        let r = inputs.map(|input| parse!(input).unwrap().derivative(v1));

        let res = [
            "(2*v1+1)^(v1+5)*log(2*v1+1)+2*(v1+5)*(2*v1+1)^(v1+4)",
            "2*(2*v1)^-1+3*exp(3*v1)+4*cos(4*v1)-y*sin(v1*y)",
            "der(0,1,f(v1^2,v1))+2*v1*der(1,0,f(v1^2,v1))",
            "der(1,1,f(v1,v1^3))+3*v1^2*der(0,2,f(v1,v1^3))",
        ];
        let res = res.map(|input| parse!(input).unwrap());

        let _ = y;
        assert_eq!(r, res);
    }

    #[test]
    fn repeated_factor_product_rule() {
        let x = symbol!("x");
        let expr = parse!("x*x").unwrap();
        assert_eq!(expr.derivative(x), parse!("2*x").unwrap());
    }

    #[test]
    fn sqrt_derivative() {
        let x = symbol!("x");
        let expr = parse!("sqrt(x)").unwrap();
        assert_eq!(expr.derivative(x), parse!("1/2*x^(-1/2)").unwrap());
    }
    #[test]
    fn extended_trig_derivatives() {
        let x = symbol!("x");
        assert_eq!(
            parse!("tan(x)").unwrap().derivative(x),
            parse!("cos(x)^-2").unwrap()
        );
        assert_eq!(
            parse!("ctg(x)").unwrap().derivative(x),
            parse!("(-1)*sin(x)^-2").unwrap()
        );
        assert_eq!(
            parse!("arcsin(x)").unwrap().derivative(x),
            parse!("(1-x^2)^(-1/2)").unwrap()
        );
        assert_eq!(
            parse!("arccos(x)").unwrap().derivative(x),
            parse!("(-1)*(x^2-1)^(-1/2)").unwrap()
        );
        assert_eq!(
            parse!("arctg(x)").unwrap().derivative(x),
            parse!("(x^2+1)^-1").unwrap()
        );
        assert_eq!(
            parse!("arcctg(x)").unwrap().derivative(x),
            parse!("(-1)*(x^2+1)^-1").unwrap()
        );
    }

    #[test]
    fn constant_derivative_is_zero() {
        let x = symbol!("x");
        let expr = Atom::new_num((3, 4));
        assert_eq!(expr.derivative(x), Atom::new_num(0));
    }

    #[test]
    fn malformed_derivative_target_returns_error() {
        let x = symbol!("x");
        let expr = parse!("der(0,x)").unwrap();
        assert_eq!(
            expr.try_derivative(x).unwrap_err(),
            DerivativeError::DerivativeTargetMustBeFunction
        );
    }

    #[test]
    fn malformed_derivative_orders_return_error() {
        let x = symbol!("x");
        let expr = parse!("der(x,f(x))").unwrap();
        assert_eq!(
            expr.try_derivative(x).unwrap_err(),
            DerivativeError::DerivativeOrdersMustBeNumeric
        );
    }
}
