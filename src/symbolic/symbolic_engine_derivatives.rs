use crate::symbolic::parse_expr::parse_expression_func;
use crate::symbolic::symbolic_engine::Expr;
use std::f64::consts::PI;
use crate::symbolic::utils::{
    linspace, norm, numerical_derivative, numerical_derivative_multi, transpose,
};
impl Expr {
    /// DIFFERENTIATION

    // differentiate with respect to a variable - partial derivative in case of a function of many variables,
    // a full derivative in case of a function of one variable
    pub fn diff(&self, var: &str) -> Expr {
        match self {
            Expr::Var(name) => {
                if name == var {
                    Expr::Const(1.0)
                } else {
                    Expr::Const(0.0)
                }
            }
            Expr::Const(_) => Expr::Const(0.0),
            Expr::Add(lhs, rhs) => Expr::Add(Box::new(lhs.diff(var)), Box::new(rhs.diff(var))),
            Expr::Sub(lhs, rhs) => Expr::Sub(Box::new(lhs.diff(var)), Box::new(rhs.diff(var))),
            Expr::Mul(lhs, rhs) => Expr::Add(
                Box::new(Expr::Mul(Box::new(lhs.diff(var)), rhs.clone())),
                Box::new(Expr::Mul(lhs.clone(), Box::new(rhs.diff(var)))),
            ),
            Expr::Div(lhs, rhs) => Expr::Div(
                Box::new(Expr::Sub(
                    Box::new(Expr::Mul(Box::new(lhs.diff(var)), rhs.clone())),
                    Box::new(Expr::Mul(Box::new(rhs.diff(var)), lhs.clone())),
                )),
                Box::new(Expr::Mul(rhs.clone(), rhs.clone())),
            ),
            Expr::Pow(base, exp) => Expr::Mul(
                Box::new(Expr::Mul(
                    exp.clone(),
                    Box::new(Expr::Pow(
                        base.clone(),
                        Box::new(Expr::Sub(exp.clone(), Box::new(Expr::Const(1.0)))),
                    )),
                )),
                Box::new(base.diff(var)),
            ),
            Expr::Exp(expr) => {
                Expr::Mul(Box::new(Expr::Exp(expr.clone())), Box::new(expr.diff(var)))
            }
            Expr::Ln(expr) => Expr::Div(Box::new(expr.diff(var)), expr.clone()),
            Expr::sin(expr) => {
                Expr::Mul(Box::new(Expr::cos(expr.clone())), Box::new(expr.diff(var)))
            }
            Expr::cos(expr) => Expr::Mul(
                Box::new(Expr::Mul(
                    Box::new(Expr::Const(-1.0)),
                    Box::new(Expr::sin(expr.clone())),
                )),
                Box::new(expr.diff(var)),
            ),
            Expr::tg(expr) => Expr::Mul(
                Box::new(Expr::Div(
                    Box::new(Expr::Const(1.0)),
                    Box::new(Expr::Pow(
                        Box::new(Expr::cos(expr.clone())),
                        Box::new(Expr::Const(2.0)),
                    )),
                )),
                Box::new(expr.diff(var)),
            ),
            Expr::ctg(expr) => Expr::Mul(
                Box::new(Expr::Div(
                    Box::new(Expr::Const(-1.0)),
                    Box::new(Expr::Pow(
                        Box::new(Expr::sin(expr.clone())),
                        Box::new(Expr::Const(2.0)),
                    )),
                )),
                Box::new(expr.diff(var)),
            ),
            Expr::arcsin(expr) => Expr::Div(
                Box::new(expr.diff(var)),
                Box::new(Expr::Pow(
                    Box::new(Expr::Sub(
                        Box::new(Expr::Const(1.0)),
                        Box::new(Expr::Pow(expr.clone(), Box::new(Expr::Const(2.0)))),
                    )),
                    Box::new(Expr::Const(0.5)),
                )),
            ),
            Expr::arccos(expr) => Expr::Div(
                Box::new(Expr::Mul(
                    Box::new(Expr::Const(-1.0)),
                    Box::new(expr.diff(var)),
                )),
                Box::new(Expr::Pow(
                    Box::new(Expr::Sub(
                        Box::new(Expr::Const(1.0)),
                        Box::new(Expr::Pow(expr.clone(), Box::new(Expr::Const(2.0)))),
                    )),
                    Box::new(Expr::Const(0.5)),
                )),
            ),
            Expr::arctg(expr) => Expr::Div(
                Box::new(expr.diff(var)),
                Box::new(Expr::Add(
                    Box::new(Expr::Const(1.0)),
                    Box::new(Expr::Pow(expr.clone(), Box::new(Expr::Const(2.0)))),
                )),
            ),
            Expr::arcctg(expr) => Expr::Div(
                Box::new(Expr::Mul(
                    Box::new(Expr::Const(1.0)),
                    Box::new(expr.diff(var)),
                )),
                Box::new(Expr::Add(
                    Box::new(Expr::Const(1.0)),
                    Box::new(Expr::Pow(expr.clone(), Box::new(Expr::Const(2.0)))),
                )),
            ),
        }
    } // end of diff

    /// TURN EXP TO STRING
    ///   
    pub fn sym_to_str(&self, var: &str) -> String {
        match self {
            Expr::Var(name) => name.clone(),
            Expr::Const(val) => val.to_string(),
            Expr::Add(lhs, rhs) => format!("({}) + ({})", lhs.sym_to_str(var), rhs.sym_to_str(var)),
            Expr::Sub(lhs, rhs) => format!("({}) - ({})", lhs.sym_to_str(var), rhs.sym_to_str(var)),
            Expr::Mul(lhs, rhs) => format!("({}) * ({})", lhs.sym_to_str(var), rhs.sym_to_str(var)),
            Expr::Div(lhs, rhs) => format!("({}) / ({})", lhs.sym_to_str(var), rhs.sym_to_str(var)),
            Expr::Pow(base, exp) => format!("({}^{})", base.sym_to_str(var), exp.sym_to_str(var)),
            Expr::Exp(expr) => format!("exp({})", expr.sym_to_str(var)),
            Expr::Ln(expr) => format!("ln({})", expr.sym_to_str(var)),
            Expr::sin(expr) => format!("sin({})", expr.sym_to_str(var)),
            Expr::cos(expr) => format!("cos({})", expr.sym_to_str(var)),
            Expr::tg(expr) => format!("tg({})", expr.sym_to_str(var)),
            Expr::ctg(expr) => format!("ctg({})", expr.sym_to_str(var)),
            Expr::arcsin(expr) => format!("arcsin({})", expr.sym_to_str(var)),
            Expr::arccos(expr) => format!("arccos({})", expr.sym_to_str(var)),
            Expr::arctg(expr) => format!("arctg({})", expr.sym_to_str(var)),
            Expr::arcctg(expr) => format!("arcctg({})", expr.sym_to_str(var)),
        } // end of match
    } // end of sym_to_str
    ///LAMBDIFY
    /// function to lambdify the symbolic dunction of one argument = convert it into a rust function
    pub fn lambdify1D(&self) -> Box<dyn Fn(f64) -> f64> {
        match self {
            Expr::Var(_) => Box::new(|x| x),
            Expr::Const(val) => {
                let val = *val;
                Box::new(move |_| val)
            }
            Expr::Add(lhs, rhs) => {
                let lhs_fn = lhs.lambdify1D();
                let rhs_fn = rhs.lambdify1D();
                Box::new(move |x| lhs_fn(x) + rhs_fn(x))
            }

            Expr::Sub(lhs, rhs) => {
                let lhs_fn = lhs.lambdify1D();
                let rhs_fn = rhs.lambdify1D();
                Box::new(move |x| lhs_fn(x) - rhs_fn(x))
            }
            Expr::Mul(lhs, rhs) => {
                let lhs_fn = lhs.lambdify1D();
                let rhs_fn = rhs.lambdify1D();
                Box::new(move |x| lhs_fn(x) * rhs_fn(x))
            }
            Expr::Div(lhs, rhs) => {
                let lhs_fn = lhs.lambdify1D();
                let rhs_fn = rhs.lambdify1D();
                Box::new(move |x| lhs_fn(x) / rhs_fn(x))
            }
            Expr::Pow(base, exp) => {
                let base_fn = base.lambdify1D();
                let exp_fn = exp.lambdify1D();
                Box::new(move |x| base_fn(x).powf(exp_fn(x)))
            }
            Expr::Exp(expr) => {
                let expr_fn = expr.lambdify1D();
                Box::new(move |x| expr_fn(x).exp())
            }
            Expr::Ln(expr) => {
                let expr_fn = expr.lambdify1D();
                Box::new(move |x| expr_fn(x).ln())
            }
            Expr::sin(expr) => {
                let expr_fn = expr.lambdify1D();
                Box::new(move |x| expr_fn(x).sin())
            }
            Expr::cos(expr) => {
                let expr_fn = expr.lambdify1D();
                Box::new(move |x| expr_fn(x).cos())
            }
            Expr::tg(expr) => {
                let expr_fn = expr.lambdify1D();
                Box::new(move |x| expr_fn(x).tan())
            }
            Expr::ctg(expr) => {
                let expr_fn = expr.lambdify1D();
                Box::new(move |x| expr_fn(1.0 / x).tan())
            }
            Expr::arcsin(expr) => {
                let expr_fn = expr.lambdify1D();
                Box::new(move |x| expr_fn(x).asin())
            }
            Expr::arccos(expr) => {
                let expr_fn = expr.lambdify1D();
                Box::new(move |x| expr_fn(x).acos())
            }
            Expr::arctg(expr) => {
                let expr_fn = expr.lambdify1D();
                Box::new(move |x| expr_fn(x).atan())
            }
            Expr::arcctg(expr) => {
                let expr_fn = expr.lambdify1D();
                Box::new(move |x| PI/2.0 - expr_fn( x).atan())
            }
        } // end of match
    } // end of lambdify1D

    /// function to lambdify the symbolic function of multiple variables = convert it into a rust function

    pub fn lambdify(&self, vars: Vec<&str>) -> Box<dyn Fn(Vec<f64>) -> f64> {
        /*
            let var_indices: std::collections::HashMap<&str, usize> = vars
            .iter()
            .enumerate()
            .map(|(i, &name)| (name, i))
            .collect(); // . Pre-computed Variable Indices: Creates a HashMap of
        //variable names to indices at the start, avoiding repeated lookups during evaluation.
        */

        match self {
            Expr::Var(name) => {
                let index = vars.iter().position(|&x| x == name).unwrap();
                Box::new(move |args| args[index])
            }
            Expr::Const(val) => {
                let val = *val;
                Box::new(move |_| val)
            }
            Expr::Add(lhs, rhs) => {
                let lhs_fn = lhs.lambdify(vars.clone());
                let rhs_fn = rhs.lambdify(vars);
                Box::new(move |args| lhs_fn(args.clone()) + rhs_fn(args))
            }
            Expr::Sub(lhs, rhs) => {
                let lhs_fn = lhs.lambdify(vars.clone());
                let rhs_fn = rhs.lambdify(vars);
                Box::new(move |args| lhs_fn(args.clone()) - rhs_fn(args))
            }
            Expr::Mul(lhs, rhs) => {
                let lhs_fn = lhs.lambdify(vars.clone());
                let rhs_fn = rhs.lambdify(vars);
                Box::new(move |args| lhs_fn(args.clone()) * rhs_fn(args))
            }
            Expr::Div(lhs, rhs) => {
                let lhs_fn = lhs.lambdify(vars.clone());
                let rhs_fn = rhs.lambdify(vars);
                Box::new(move |args| lhs_fn(args.clone()) / rhs_fn(args))
            }
            Expr::Pow(base, exp) => {
                let base_fn = base.lambdify(vars.clone());
                let exp_fn = exp.lambdify(vars);
                Box::new(move |args| base_fn(args.clone()).powf(exp_fn(args)))
            }
            Expr::Exp(expr) => {
                let expr_fn = expr.lambdify(vars);
                Box::new(move |args| expr_fn(args).exp())
            }
            Expr::Ln(expr) => {
                let expr_fn = expr.lambdify(vars);
                Box::new(move |args| expr_fn(args).ln())
            }
            Expr::sin(expr) => {
                let expr_fn = expr.lambdify(vars);
                Box::new(move |args| expr_fn(args).sin())
            }
            Expr::cos(expr) => {
                let expr_fn = expr.lambdify(vars);
                Box::new(move |args| expr_fn(args).cos())
            }
            Expr::tg(expr) => {
                let expr_fn = expr.lambdify(vars);
                Box::new(move |args| expr_fn(args).tan())
            }
            Expr::ctg(expr) => {
                let expr_fn = expr.lambdify(vars);
                Box::new(move |args| {
                    let args = args.iter().map(|x| 1.0 / x).collect::<Vec<f64>>();
                    expr_fn(args).tan()
                })
            }
            Expr::arcsin(expr) => {
                let expr_fn = expr.lambdify(vars);
                Box::new(move |args| expr_fn(args).asin())
            }
            Expr::arccos(expr) => {
                let expr_fn = expr.lambdify(vars);
                Box::new(move |args| expr_fn(args).acos())
            }
            Expr::arctg(expr) => {
                let expr_fn = expr.lambdify(vars);
                Box::new(move |args| expr_fn(args).atan())
            }
            Expr::arcctg(expr) => {
                let expr_fn = expr.lambdify(vars);
                Box::new(move |args| {
                  
                    PI/2.0 - expr_fn(args).tan()
                })
            }
        }
    } // end of lambdify

    pub fn lambdify_slice(&self, vars: Vec<&str>) -> Box<dyn Fn(&[f64]) -> f64 + '_> {
        /*
            let var_indices: std::collections::HashMap<&str, usize> = vars
            .iter()
            .enumerate()
            .map(|(i, &name)| (name, i))
            .collect(); // . Pre-computed Variable Indices: Creates a HashMap of
        //variable names to indices at the start, avoiding repeated lookups during evaluation.
        */

        match self {
            Expr::Var(name) => {
                let index = vars.iter().position(|&x| x == name).unwrap();
                Box::new(move |args| args[index])
            }
            Expr::Const(val) => {
                let val = *val;
                Box::new(move |_| val)
            }
            Expr::Add(lhs, rhs) => {
                let lhs_fn = lhs.lambdify_slice(vars.clone());
                let rhs_fn = rhs.lambdify_slice(vars);
                Box::new(move |args| lhs_fn(args) + rhs_fn(args))
            }
            Expr::Sub(lhs, rhs) => {
                let lhs_fn = lhs.lambdify_slice(vars.clone());
                let rhs_fn = rhs.lambdify_slice(vars);
                Box::new(move |args| lhs_fn(args) - rhs_fn(args))
            }
            Expr::Mul(lhs, rhs) => {
                let lhs_fn = lhs.lambdify_slice(vars.clone());
                let rhs_fn = rhs.lambdify_slice(vars);
                Box::new(move |args| lhs_fn(args) * rhs_fn(args))
            }
            Expr::Div(lhs, rhs) => {
                let lhs_fn = lhs.lambdify_slice(vars.clone());
                let rhs_fn = rhs.lambdify_slice(vars);
                Box::new(move |args| lhs_fn(args) / rhs_fn(args))
            }
            Expr::Pow(base, exp) => {
                let base_fn = base.lambdify_slice(vars.clone());
                let exp_fn = exp.lambdify_slice(vars);
                Box::new(move |args| base_fn(args).powf(exp_fn(args)))
            }
            Expr::Exp(expr) => {
                let expr_fn = expr.lambdify_slice(vars);
                Box::new(move |args| expr_fn(args).exp())
            }
            Expr::Ln(expr) => {
                let expr_fn = expr.lambdify_slice(vars);
                Box::new(move |args| expr_fn(args).ln())
            }
            Expr::sin(expr) => {
                let expr_fn = expr.lambdify_slice(vars);
                Box::new(move |args| expr_fn(args).sin())
            }
            Expr::cos(expr) => {
                let expr_fn = expr.lambdify_slice(vars);
                Box::new(move |args| expr_fn(args).cos())
            }
            Expr::tg(expr) => {
                let expr_fn = expr.lambdify_slice(vars);
                Box::new(move |args| expr_fn(args).tan())
            }
            Expr::ctg(expr) => {
                let expr_fn = expr.lambdify_slice(vars);
                Box::new(move |args| 1.0 / expr_fn(args).tan())
            }
            Expr::arcsin(expr) => {
                let expr_fn = expr.lambdify_slice(vars);
                Box::new(move |args| expr_fn(args).asin())
            }
            Expr::arccos(expr) => {
                let expr_fn = expr.lambdify_slice(vars);
                Box::new(move |args| expr_fn(args).acos())
            }
            Expr::arctg(expr) => {
                let expr_fn = expr.lambdify_slice(vars);
                Box::new(move |args| expr_fn(args).atan())
            }
            Expr::arcctg(expr) => {
                let expr_fn = expr.lambdify_slice(vars);
                Box::new(move |args| PI/2.0 - expr_fn(args).atan())
            }
        }
    } // end of lambdify

    pub fn lambdify_owned(self, vars: Vec<&str>) -> Box<dyn Fn(Vec<f64>) -> f64> {
        /*
            let var_indices: std::collections::HashMap<&str, usize> = vars
            .iter()
            .enumerate()
            .map(|(i, &name)| (name, i))
            .collect(); // . Pre-computed Variable Indices: Creates a HashMap of
        //variable names to indices at the start, avoiding repeated lookups during evaluation.
        */
        match self {
            Expr::Var(name) => {
                let index = vars.iter().position(|&x| x == name).unwrap();
                Box::new(move |args| args[index])
            }
            Expr::Const(val) => Box::new(move |_| val),
            /*
            Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) | Expr::Mul(lhs, rhs) | Expr::Div(lhs, rhs) => {
                let lhs_fn = lhs.lambdify_owned(vars.clone());
                let rhs_fn = rhs.lambdify_owned(vars.clone());
                Box::new(move |args| lhs_fn(args.clone()) + rhs_fn(args))
            }
            */
            Expr::Add(lhs, rhs) => {
                let lhs_fn = lhs.lambdify_owned(vars.clone());
                let rhs_fn = rhs.lambdify_owned(vars);
                Box::new(move |args| lhs_fn(args.clone()) + rhs_fn(args))
            }
            Expr::Sub(lhs, rhs) => {
                let lhs_fn = lhs.lambdify_owned(vars.clone());
                let rhs_fn = rhs.lambdify_owned(vars);
                Box::new(move |args| lhs_fn(args.clone()) - rhs_fn(args))
            }
            Expr::Mul(lhs, rhs) => {
                let lhs_fn = lhs.lambdify_owned(vars.clone());
                let rhs_fn = rhs.lambdify_owned(vars);
                Box::new(move |args| lhs_fn(args.clone()) * rhs_fn(args))
            }
            Expr::Div(lhs, rhs) => {
                let lhs_fn = lhs.lambdify_owned(vars.clone());
                let rhs_fn = rhs.lambdify_owned(vars);
                Box::new(move |args| lhs_fn(args.clone()) / rhs_fn(args))
            }
            Expr::Pow(base, exp) => {
                let base_fn = base.lambdify_owned(vars.clone());
                let exp_fn = exp.lambdify_owned(vars);
                Box::new(move |args| base_fn(args.clone()).powf(exp_fn(args)))
            }
            Expr::Exp(expr) => {
                let expr_fn = expr.lambdify_owned(vars);
                Box::new(move |args| expr_fn(args).exp())
            }
            Expr::Ln(expr) => {
                let expr_fn = expr.lambdify_owned(vars);
                Box::new(move |args| expr_fn(args).ln())
            }
            Expr::sin(expr) => {
                let expr_fn = expr.lambdify_owned(vars);
                Box::new(move |args| expr_fn(args).sin())
            }
            Expr::cos(expr) => {
                let expr_fn = expr.lambdify_owned(vars);
                Box::new(move |args| expr_fn(args).cos())
            }
            Expr::tg(expr) => {
                let expr_fn = expr.lambdify_owned(vars);
                Box::new(move |args| expr_fn(args).tan())
            }
            Expr::ctg(expr) => {
                let expr_fn = expr.lambdify_owned(vars);
                Box::new(move |args| 1.0 / expr_fn(args).tan())
            }
            Expr::arcsin(expr) => {
                let expr_fn = expr.lambdify_owned(vars);
                Box::new(move |args| expr_fn(args).asin())
            }
            Expr::arccos(expr) => {
                let expr_fn = expr.lambdify_owned(vars);
                Box::new(move |args| expr_fn(args).acos())
            }
            Expr::arctg(expr) => {
                let expr_fn = expr.lambdify_owned(vars);
                Box::new(move |args| expr_fn(args).atan())
            }
            Expr::arcctg(expr) => {
                let expr_fn = expr.lambdify_owned(vars);
                Box::new(move |args| PI/2.0 - expr_fn(args).atan())
            }
        }
    }

    pub fn lambdify_wrapped(&self) -> Box<dyn Fn(Vec<f64>) -> f64 + '_> {
        let vars_ = self.all_arguments_are_variables();
        let vars = vars_.iter().map(|x| x.as_str()).collect::<Vec<&str>>();
        self.lambdify(vars)
    }

    //extract all the variables present in the symbolic expression. This is achieved through a recursive traversal of the expression tree.
    pub fn extract_variables(&self) -> (Vec<i32>, Vec<String>) {
        let mut vars = Vec::new();
        let mut args = Vec::new();
        match self {
            Expr::Var(name) => {
                vars.push(name.clone());
                args.push(0);
            }
            Expr::Const(_) => {}
            Expr::Add(lhs, rhs)
            | Expr::Sub(lhs, rhs)
            | Expr::Mul(lhs, rhs)
            | Expr::Div(lhs, rhs) => {
                let mut lhs_vars = lhs.all_arguments_are_variables();
                let mut rhs_vars = rhs.all_arguments_are_variables();
                vars.append(&mut lhs_vars);
                vars.append(&mut rhs_vars);
                args.push(0);
                args.push(1);
            }
            Expr::Pow(base, exp) => {
                let mut base_vars = base.all_arguments_are_variables();
                let mut exp_vars = exp.all_arguments_are_variables();
                vars.append(&mut base_vars);
                vars.append(&mut exp_vars);
                args.push(0);
                args.push(1);
            }
            Expr::Exp(expr) => {
                let mut expr_vars = expr.all_arguments_are_variables();
                vars.append(&mut expr_vars);
                args.push(0);
            }
            Expr::Ln(expr) => {
                let mut expr_vars = expr.all_arguments_are_variables();
                vars.append(&mut expr_vars);
                args.push(0);
            }
            Expr::sin(expr) | Expr::cos(expr) | Expr::tg(expr) | Expr::ctg(expr) => {
                let mut expr_vars = expr.all_arguments_are_variables();
                vars.append(&mut expr_vars);
                args.push(0);
            }
            Expr::arcsin(expr) | Expr::arccos(expr) | Expr::arctg(expr) | Expr::arcctg(expr) => {
                let mut expr_vars = expr.all_arguments_are_variables();
                vars.append(&mut expr_vars);
                args.push(0);
            }
        } // end of match
        vars.dedup();
        vars.sort();
        (args, vars)
    } // end of extract_variables
    //TAYLOR SERIES///////////////////////////////////////
    ///Taylor series expansion of a symbolic expression
    pub fn n_th_derivative1D(&self, var_name: &str, n: usize) -> Expr {
        let mut expr = self.clone();
        let mut i = 0;
        while i < n {
            expr = expr.diff(var_name).symplify();
            i += 1;
        }
        return expr.symplify();
    }
    pub fn taylor_series1D(&self, var_name: &str, x0: f64, order: usize) -> Expr {
        let x = Expr::Var(var_name.to_owned());
        let x0_sym = Expr::Const(x0);
        let fun_at_x0 = self.lambdify1D()(x0);
        let fun_at_x0_sym = Expr::Const(fun_at_x0);

        if order == 0 {
            return fun_at_x0_sym.symplify();
        }

        let dfun_dx = self.n_th_derivative1D(var_name, order);
        let dfun_dx_at_x0 = dfun_dx.lambdify1D()(x0);
        let factorial = (1..=order).product::<usize>() as f64;
        let coeff = Expr::Const(dfun_dx_at_x0 / factorial);
        println!("order {}, {:?}, {}", order, coeff, dfun_dx);
        let term = coeff * (x.clone() - x0_sym.clone()).pow(Expr::Const(order as f64));
        if order == 1 {
            let Taylor = fun_at_x0_sym + term;
            return Taylor.symplify();
        } else {
            let Taylor = self.taylor_series1D(var_name, x0, order - 1) + term;
            return Taylor.symplify();
        }
    }

    pub fn taylor_series1D_(&self, var_name: &str, x0: f64, order: usize) -> Expr {
        let x = Expr::Var(var_name.to_owned());
        let x0_sym = Expr::Const(x0);
        let fun_at_x0 = self.lambdify1D()(x0);
        let fun_at_x0_sym = Expr::Const(fun_at_x0);

        if order == 0 {
            return fun_at_x0_sym.symplify();
        }

        fn taylor_term(
            expr: &Expr,
            var_name: &str,
            x0: f64,
            n: usize,
            x: &Expr,
            x0_sym: &Expr,
        ) -> (Expr, Expr) {
            let dfun_dx = expr.diff(var_name).symplify();
            let dfun_dx_at_x0 = dfun_dx.lambdify1D()(x0);
            let factorial = (1..=n).product::<usize>() as f64;
            let coeff = Expr::Const(dfun_dx_at_x0 / factorial);
            //  println!("order {}, {:?}, {}", n, coeff, dfun_dx);
            (
                coeff
                    * (x.clone() - x0_sym.clone())
                        .pow(Expr::Const(n as f64))
                        .symplify(),
                dfun_dx,
            )
        }

        fn taylor_recursive(
            expr: &Expr,
            var_name: &str,
            x0: f64,
            current_order: usize,
            target_order: usize,
            x: &Expr,
            x0_sym: &Expr,
        ) -> Expr {
            if current_order > target_order {
                return Expr::Const(0.0);
            }
            let (term, derivative) = taylor_term(expr, var_name, x0, current_order, x, x0_sym);
            // println!("\n derivative {}, \n term {} \n", derivative, term);
            term + taylor_recursive(
                &derivative,
                var_name,
                x0,
                current_order + 1,
                target_order,
                x,
                x0_sym,
            )
        }

        let Taylor = fun_at_x0_sym + taylor_recursive(self, var_name, x0, 1, order, &x, &x0_sym);
        Taylor.symplify()
    }
    // EVAL EXPRESSIONS //////////////////////////////////////////////////////////
    pub fn eval_expression(&self, vars: Vec<&str>, values: &[f64]) -> f64 {
        /*
            let var_indices: std::collections::HashMap<&str, usize> = vars
            .iter()
            .enumerate()
            .map(|(i, &name)| (name, i))
            .collect(); // . Pre-computed Variable Indices: Creates a HashMap of
        //variable names to indices at the start, avoiding repeated lookups during evaluation.
        */

        match self {
            Expr::Var(name) => {
                let index = vars.iter().position(|&x| x == name).unwrap();
                values[index]
            }
            Expr::Const(val) => {
                let val = *val;
                val
            }
            Expr::Add(lhs, rhs) => {
                let lhs_fn = lhs.eval_expression(vars.clone(), values);
                let rhs_fn = rhs.eval_expression(vars, values);
                lhs_fn + rhs_fn
            }
            Expr::Sub(lhs, rhs) => {
                let lhs_fn = lhs.eval_expression(vars.clone(), values);
                let rhs_fn = rhs.eval_expression(vars, values);
                lhs_fn - rhs_fn
            }
            Expr::Mul(lhs, rhs) => {
                let lhs_fn = lhs.eval_expression(vars.clone(), values);
                let rhs_fn = rhs.eval_expression(vars, values);
                lhs_fn * rhs_fn
            }
            Expr::Div(lhs, rhs) => {
                let lhs_fn = lhs.eval_expression(vars.clone(), values);
                let rhs_fn = rhs.eval_expression(vars, values);
                lhs_fn / rhs_fn
            }
            Expr::Pow(base, exp) => {
                let base_fn = base.eval_expression(vars.clone(), values);
                let exp_fn = exp.eval_expression(vars, values);
                base_fn.powf(exp_fn)
            }
            Expr::Exp(expr) => {
                let expr_fn = expr.eval_expression(vars, values);
                expr_fn.exp()
            }
            Expr::Ln(expr) => {
                let expr_fn = expr.eval_expression(vars, values);
                expr_fn.ln()
            }
            Expr::sin(expr) => {
                let expr_fn = expr.eval_expression(vars, values);
                expr_fn.sin()
            }
            Expr::cos(expr) => {
                let expr_fn = expr.eval_expression(vars, values);
                expr_fn.cos()
            }
            Expr::tg(expr) => {
                let expr_fn = expr.eval_expression(vars, values);
                expr_fn.tan()
            }
            Expr::ctg(expr) => {
                let expr_fn = expr.eval_expression(vars, values);
                (1.0 / expr_fn).tan()
            }
            Expr::arcsin(expr) => {
                let expr_fn = expr.eval_expression(vars, values);
                expr_fn.asin()
            }
            Expr::arccos(expr) => {
                let expr_fn = expr.eval_expression(vars, values);
                expr_fn.acos()
            }
            Expr::arctg(expr) => {
                let expr_fn = expr.eval_expression(vars, values);
                expr_fn.atan()
            }
            Expr::arcctg(expr) => {
                let expr_fn = expr.eval_expression(vars, values);
                PI/2.0 - ( expr_fn).atan()
            }
        }
    } // end of eval_expression

    /// PARSE EXPRESSIONS
    /// function to parse an expression from a string
    /// returns an error if the expression is invalid
    /// returns a symbolic expression if the expression is valid
    /// function is just a wrapper around parse_expression_func
    pub fn parse_expression(input: &str) -> Expr {
        let expr = match parse_expression_func(0, input) {
            Ok(expr) => {
                println!("\n \n found expression: {:?}", expr);
                println!(
                    "\n \n in human readable format {:?} \n \n ",
                    &expr.clone().sym_to_str("x")
                );
                Ok(expr)
            }
            Err(err) => {
                println!("Error: {}", err);
                Err(err)
            }
        };
        expr.unwrap()
    }
    /// function to parse a vector of expressions from a vector of strings
    pub fn parse_vector_expression(input: Vec<&str>) -> Vec<Expr> {
        let mut exprs = Vec::new();
        for i in input {
            let expr = match parse_expression_func(0, i) {
                Ok(expr) => {
                    println!("\n \n found expression: {:?}", expr);
                    println!(
                        "\n \n in human readable format {:?} \n \n ",
                        &expr.clone().sym_to_str("x")
                    );
                    Ok(expr)
                }
                Err(err) => {
                    println!("Error: {}", err);
                    Err(err)
                }
            };
            exprs.push(expr.unwrap());
        }
        exprs
    }
    /// function returns a vector of variables of the symbolic expression
    /// Function to find all symbolic variables in a symbolic expression
    pub fn all_arguments_are_variables(&self) -> Vec<String> {
        let mut vars = Vec::new();

        match self {
            Expr::Var(name) => {
                vars.push(name.clone());
            }
            Expr::Const(_) => {}
            Expr::Add(lhs, rhs)
            | Expr::Sub(lhs, rhs)
            | Expr::Mul(lhs, rhs)
            | Expr::Div(lhs, rhs) => {
                let _lhs_vars = lhs.all_arguments_are_variables();
                vars.extend(lhs.all_arguments_are_variables());
                vars.extend(rhs.all_arguments_are_variables());
            }
            Expr::Pow(base, exp) => {
                vars.extend(base.all_arguments_are_variables());
                vars.extend(exp.all_arguments_are_variables());
            }
            Expr::Exp(expr) | Expr::Ln(expr) => {
                vars.extend(expr.all_arguments_are_variables());
            }
            Expr::sin(expr) | Expr::cos(expr) | Expr::tg(expr) | Expr::ctg(expr) => {
                vars.extend(expr.all_arguments_are_variables());
            }
            Expr::arcsin(expr) | Expr::arccos(expr) | Expr::arctg(expr) | Expr::arcctg(expr) => {
                vars.extend(expr.all_arguments_are_variables());
            }
        }

        vars.sort(); // Add this line to sort the variables
        vars.dedup(); // Remove duplicates
        vars
    } // end of all_arguments_are_variables

    //___________________________________________________________________________________________________________________
    //                    1D  function processing, like y = f(x)
    // _________________________________________________________________________________________________________________
    // function to calculate the symbolic expression for a vector of values
    pub fn calc_vector_lambdified1D(&self, x: &Vec<f64>) -> Vec<f64> {
        let mut result = Vec::new();
        for xi in x {
            result.push(self.lambdify1D()(*xi));
        }
        result
    } // end of calc_vector_lambdified1D
    /// calculate vector of results of 1D lambdified function using linspace
    pub fn lambdify1D_from_linspace(&self, start: f64, end: f64, num_values: usize) -> Vec<f64> {
        let x = linspace(start, end, num_values);
        self.calc_vector_lambdified1D(&x)
    } // end of lambdify1D_from_linspace
    // compare lambdified derivative with numerical derivative on certain x values
    pub fn compare_num1D(
        &self,
        var: &str,
        start: f64,
        end: f64,
        num_values: usize,
        max_norm: f64,
    ) -> (f64, bool) {
        let diff = &self.diff(var); // get the analtical derivative
        let analytical_derivative = diff.lambdify1D_from_linspace(start, end, num_values); // calculate values of the analtical derivative on the linspace
        let analitical_function = &self.lambdify1D(); // get the analtical function
        let step = (1.0 / 1e4) * (end - start) / (num_values as f64 - 1.0); //
        let domain = linspace(start, end, num_values);
        let numerical_derivative = numerical_derivative(analitical_function, domain, step); // calculate values of the numerical derivative on the linspace
        let norma_val = norm(analytical_derivative, numerical_derivative);

        if max_norm > norma_val {
            (norma_val, true)
        } else {
            (norma_val, false)
        }
    }
    //___________________________________________________________________________________________________________________
    //                     processing functions of arbitrary dimension, like y = f(x1, x2, x3)
    // _________________________________________________________________________________________________________________
    //   evaluate the symbolic expression for a vector of vectors of values
    //       for example [[x1, x2, x3], [y1, y2, y3]] for y = f(x, y)
    pub fn evaluate_vector_lambdified(&self, x: &Vec<Vec<f64>>) -> Vec<f64> {
        // Vector passed in the form [[x1, x2, x3], [y1, y2, y3]], and we need for
        // our function f(x, y) to be passed in the form [[x1, y1], [x2, y2], [x3, y3]]
        let x_reshaped = transpose(x.to_owned());
        let all_lengths_equal = x_reshaped.iter().all(|x| x.len() == x_reshaped[0].len());
        assert!(all_lengths_equal, "All vectors must have the same length");
        let mut result = Vec::new();
        let f = self.lambdify_wrapped();
        for xi in x_reshaped {
            result.push(f(xi));
        }
        result
    }
    /// The same as above but arguments defined by linspace
    ///
    pub fn lamdified_from_linspace(
        &self,
        start: Vec<f64>,
        end: Vec<f64>,
        num_values: usize,
    ) -> Vec<f64> {
        let mut vec_of_linspaces = Vec::new();
        for (start_i, end_i) in start.iter().zip(end.iter()) {
            assert!(*start_i <= *end_i);
            assert!(num_values > 1);
            let x = linspace(*start_i, *end_i, num_values);
            vec_of_linspaces.push(x);
        }
        self.evaluate_vector_lambdified(&vec_of_linspaces)
    }
    /// differentiate funcion of multiple variables by all of them - returns vector of Expr
    pub fn diff_multi_args(&self, all_vars: &Vec<&str>) -> Vec<Expr> {
        let vec_of_exprs = all_vars.iter().map(|var| self.diff(var)).collect();
        vec_of_exprs
    }

    /// differentiate funcion of multiple variables by all of them - returns vector of Expr
    pub fn diff_multi(&self) -> Vec<Expr> {
        let all_vars = self.all_arguments_are_variables();
        let vec_of_exprs = all_vars.iter().map(|var| self.diff(var)).collect();
        vec_of_exprs
    }
    /// for every partial derivative calculated numerical result for all variables
    pub fn evaluate_multi_diff(&self, x: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let diff_multi_expression_vec = self.diff_multi();
        let vector_of_results: Vec<Vec<f64>> = diff_multi_expression_vec
            .iter()
            .map(|expr| expr.evaluate_vector_lambdified(x))
            .collect();
        println!(
            "{:?}, {:?}",
            diff_multi_expression_vec.len(),
            &vector_of_results.clone().len()
        );
        vector_of_results
    }
    /// for each parial derivative calulated numerical result for arguments defined by linspace
    pub fn evaluate_multi_diff_from_linspace(
        &self,
        start: Vec<f64>,
        end: Vec<f64>,
        num_values: usize,
    ) -> Vec<Vec<f64>> {
        let diff_multi_expression_vec = self.diff_multi();
        // for each expression of partial derivative calculate result for vector of values
        let vector_of_results: Vec<Vec<f64>> = diff_multi_expression_vec
            .iter()
            .map(|expr| expr.lamdified_from_linspace(start.clone(), end.clone(), num_values))
            .collect();
        // println!("{}, {}",diff_multi_expression_vec.len(),  &vector_of_results.len());
        vector_of_results
    }

    /// compare lambdified derivative with numerical derivative on certain x values
    pub fn compare_num(
        &self,
        start: Vec<f64>,
        end: Vec<f64>,
        num_values: usize,
        max_norm: f64,
    ) -> Vec<(bool, f64)> {
        // for each expression of partial derivative calculate result for vector of values
        let vector_of_vectors_of_dy_dx_analytical =
            self.evaluate_multi_diff_from_linspace(start.clone(), end.clone(), num_values);
        //println!("vector_of_vectors_of_dy_dx_analytical  {:?}", vector_of_vectors_of_dy_dx_analytical);
        let analitical_function = &self.lambdify_wrapped(); // get the analtical function
        // let's define step
        let max_end = end.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
        let min_start = start.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
        let step = (1.0 / 1e4) * (max_end - min_start) / (num_values as f64 - 1.0); //

        // Generate domain
        let mut vec_of_linspaces = Vec::new();
        for (start_i, end_i) in start.iter().zip(end.iter()) {
            assert!(*start_i <= *end_i);
            assert!(num_values > 1);
            let x = linspace(*start_i, *end_i, num_values);
            vec_of_linspaces.push(x);
        }
        // Vector  vec_of_linspaces in the form [[x1, x2, x3], [y1, y2, y3]], and we need for
        // our function f(x, y) to be passed in the form [[x1, y1], [x2, y2], [x3, y3]]
        let domain = transpose(vec_of_linspaces.to_owned());
        let vector_of_dy_dx_nymerical_: Vec<Vec<f64>> = domain
            .iter()
            .map(|x_vector_i| {
                numerical_derivative_multi(analitical_function, x_vector_i.clone(), step)
            })
            .collect();
        let vector_of_dy_dx_nymerical = transpose(vector_of_dy_dx_nymerical_);
        assert_eq!(
            vector_of_vectors_of_dy_dx_analytical.len(),
            vector_of_dy_dx_nymerical.len()
        );
        // calculate vector of norms for each partial derivative
        let norms = vector_of_dy_dx_nymerical
            .iter()
            .zip(&vector_of_vectors_of_dy_dx_analytical)
            .map(|(y_vector_analytical, y_vector_numerical)| {
                norm(y_vector_analytical.clone(), y_vector_numerical.clone())
            })
            .collect::<Vec<_>>();
        // if each norm < max_norm
        let true_or_false = norms
            .iter()
            .map(|norm_i| norm_i < &max_norm)
            .collect::<Vec<_>>();
        // pack true_or_false and norms into pairs
        let pairs = true_or_false
            .iter()
            .zip(norms.iter())
            .map(|(&a, &b)| (a, b))
            .collect::<Vec<_>>();
        pairs
    }
    //____________________________________________________________________________________________________________________________
    /// in IVP you have one argument and one or many unknown variables
    pub fn lambdify_IVP(
        &self,
        arg: &str,
        vars: Vec<&str>,
    ) -> Box<dyn Fn(f64, Vec<f64>) -> f64 + '_> {
        let mut x = vec![arg];
        x.extend(vars); // extend vars.clone();

        let f = self.lambdify(x);

        let f_closure: Box<dyn Fn(f64, Vec<f64>) -> f64> = Box::new(move |x, y_vec| {
            // Assuming y_vec has at least one element
            let mut x_y_vec = vec![x];
            x_y_vec.extend(y_vec); // extend vars.clone();

            // Call the original closure with x and y
            f(x_y_vec)
        });
        f_closure
    }
    pub fn lambdify_IVP_owned(
        self,
        arg: &str,
        vars: Vec<&str>,
    ) -> Box<dyn Fn(f64, Vec<f64>) -> f64> {
        let mut x = vec![arg];
        x.extend(vars); // extend vars.clone();
        //  println!("x {:?}",&x);
        let f = self.lambdify_owned(x);

        let f_closure: Box<dyn Fn(f64, Vec<f64>) -> f64> = Box::new(move |x, y_vec| {
            // Assuming y_vec has at least one element
            let mut x_y_vec = vec![x];
            x_y_vec.extend(y_vec); // extend vars.clone();

            // Call the original closure with x and y
            f(x_y_vec)
        });
        f_closure
    } //lambdify_IVP
}
