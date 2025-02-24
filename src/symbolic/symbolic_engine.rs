#![allow(non_camel_case_types)]
use crate::symbolic::parse_expr::parse_expression_func;
use crate::symbolic::utils::{
    linspace, norm, numerical_derivative, numerical_derivative_multi, transpose,
};
use std::collections::HashMap;
use std::f64;
use std::fmt;
// Define an enum to represent different types of symbolic expressions

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Var(String),
    Const(f64),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Pow(Box<Expr>, Box<Expr>),
    Exp(Box<Expr>),
    Ln(Box<Expr>),
}

// Implement Display for pretty printing

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Expr::Var(name) => write!(f, "{}", name),
            Expr::Const(val) => write!(f, "{}", val),
            Expr::Add(lhs, rhs) => write!(f, "({} + {})", lhs, rhs),
            Expr::Sub(lhs, rhs) => write!(f, "({} - {})", lhs, rhs),
            Expr::Mul(lhs, rhs) => write!(f, "({} * {})", lhs, rhs),
            Expr::Div(lhs, rhs) => write!(f, "({} / {})", lhs, rhs),
            Expr::Pow(base, exp) => write!(f, "({} ^ {})", base, exp),
            Expr::Exp(expr) => write!(f, "exp({})", expr),
            Expr::Ln(expr) => write!(f, "ln({})", expr),
        }
    }
}

impl std::ops::Add for Expr {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Expr::Add(self.boxed(), rhs.boxed())
    }
}

impl std::ops::Sub for Expr {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Expr::Sub(self.boxed(), rhs.boxed())
    }
}

impl std::ops::Mul for Expr {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Expr::Mul(self.boxed(), rhs.boxed())
    }
}

impl std::ops::Div for Expr {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Expr::Div(self.boxed(), rhs.boxed())
    }
}

/*


impl  std::f64:: for Expr {
    type Output = Self;
    fn exp(self) -> Self::Output {
        Expr::Exp(self.boxed())
    }
}

    ipml std::f64::::pow for Expr {
    type Output = Self;

    fn pow(self, rhs: Self) -> Self::Output {
        Expr::Pow(self.boxed(), rhs.boxed())
    }
}

impl f64::ops::ln for Expr {
    type Output = Self;
    fn ln(self) -> Self::Output {
        Expr::Ln(self.boxed())
    }
}
    */
// Implement differentiation, based on the recursive definition

impl Expr {
    /// BASIC FEATURES

    /// create new variables from string
    pub fn Symbols(symbols: &str) -> Vec<Expr> {
        let symbols = symbols.to_string();
        let vec_trimmed: Vec<String> = symbols.split(',').map(|s| s.trim().to_string()).collect();
        let vector_of_symbolic_vars: Vec<Expr> = vec_trimmed
            .iter()
            .filter(|s| !s.is_empty())
            .map(|s| Expr::Var(s.to_string()))
            .collect();
        vector_of_symbolic_vars
    }
    /// change a variable to a constant  
    pub fn set_variable(&self, var: &str, value: f64) -> Expr {
        match self {
            Expr::Var(name) if name == var => Expr::Const(value),
            Expr::Add(lhs, rhs) => Expr::Add(
                Box::new(lhs.set_variable(var, value)),
                Box::new(rhs.set_variable(var, value)),
            ),
            Expr::Sub(lhs, rhs) => Expr::Sub(
                Box::new(lhs.set_variable(var, value)),
                Box::new(rhs.set_variable(var, value)),
            ),
            Expr::Mul(lhs, rhs) => Expr::Mul(
                Box::new(lhs.set_variable(var, value)),
                Box::new(rhs.set_variable(var, value)),
            ),
            Expr::Div(lhs, rhs) => Expr::Div(
                Box::new(lhs.set_variable(var, value)),
                Box::new(rhs.set_variable(var, value)),
            ),
            Expr::Pow(base, exp) => Expr::Pow(
                Box::new(base.set_variable(var, value)),
                Box::new(exp.set_variable(var, value)),
            ),
            Expr::Exp(expr) => Expr::Exp(Box::new(expr.set_variable(var, value))),
            Expr::Ln(expr) => Expr::Ln(Box::new(expr.set_variable(var, value))),
            _ => self.clone(),
        }
    }

    /// change a variables to a constant from a map
    pub fn set_variable_from_map(&self, var_map: &HashMap<String, f64>) -> Expr {
        match self {
            Expr::Var(name) if var_map.contains_key(name) => Expr::Const(var_map[name]),
            Expr::Add(lhs, rhs) => Expr::Add(
                Box::new(lhs.set_variable_from_map(var_map)),
                Box::new(rhs.set_variable_from_map(var_map)),
            ),
            Expr::Sub(lhs, rhs) => Expr::Sub(
                Box::new(lhs.set_variable_from_map(var_map)),
                Box::new(rhs.set_variable_from_map(var_map)),
            ),
            Expr::Mul(lhs, rhs) => Expr::Mul(
                Box::new(lhs.set_variable_from_map(var_map)),
                Box::new(rhs.set_variable_from_map(var_map)),
            ),
            Expr::Div(lhs, rhs) => Expr::Div(
                Box::new(lhs.set_variable_from_map(var_map)),
                Box::new(rhs.set_variable_from_map(var_map)),
            ),
            Expr::Pow(base, exp) => Expr::Pow(
                Box::new(base.set_variable_from_map(var_map)),
                Box::new(exp.set_variable_from_map(var_map)),
            ),
            Expr::Exp(expr) => Expr::Exp(Box::new(expr.set_variable_from_map(var_map))),
            Expr::Ln(expr) => Expr::Ln(Box::new(expr.set_variable_from_map(var_map))),
            _ => self.clone(),
        }
    }
    /// rename variable
    pub fn rename_variable(&self, old_var: &str, new_var: &str) -> Expr {
        match self {
            Expr::Var(name) if name == old_var => Expr::Var(new_var.to_string()),
            Expr::Add(lhs, rhs) => Expr::Add(
                Box::new(lhs.rename_variable(old_var, new_var)),
                Box::new(rhs.rename_variable(old_var, new_var)),
            ),
            Expr::Sub(lhs, rhs) => Expr::Sub(
                Box::new(lhs.rename_variable(old_var, new_var)),
                Box::new(rhs.rename_variable(old_var, new_var)),
            ),
            Expr::Mul(lhs, rhs) => Expr::Mul(
                Box::new(lhs.rename_variable(old_var, new_var)),
                Box::new(rhs.rename_variable(old_var, new_var)),
            ),
            Expr::Div(lhs, rhs) => Expr::Div(
                Box::new(lhs.rename_variable(old_var, new_var)),
                Box::new(rhs.rename_variable(old_var, new_var)),
            ),
            Expr::Pow(base, exp) => Expr::Pow(
                Box::new(base.rename_variable(old_var, new_var)),
                Box::new(exp.rename_variable(old_var, new_var)),
            ),
            Expr::Exp(expr) => Expr::Exp(Box::new(expr.rename_variable(old_var, new_var))),
            Expr::Ln(expr) => Expr::Ln(Box::new(expr.rename_variable(old_var, new_var))),
            _ => self.clone(),
        }
    }
    /// rename variables from a map
    pub fn rename_variables(&self, var_map: &HashMap<String, String>) -> Expr {
        match self {
            Expr::Var(name) if var_map.contains_key(name) => Expr::Var(var_map[name].to_string()),
            Expr::Add(lhs, rhs) => Expr::Add(
                Box::new(lhs.rename_variables(var_map)),
                Box::new(rhs.rename_variables(var_map)),
            ),
            Expr::Sub(lhs, rhs) => Expr::Sub(
                Box::new(lhs.rename_variables(var_map)),
                Box::new(rhs.rename_variables(var_map)),
            ),
            Expr::Mul(lhs, rhs) => Expr::Mul(
                Box::new(lhs.rename_variables(var_map)),
                Box::new(rhs.rename_variables(var_map)),
            ),
            Expr::Div(lhs, rhs) => Expr::Div(
                Box::new(lhs.rename_variables(var_map)),
                Box::new(rhs.rename_variables(var_map)),
            ),
            Expr::Pow(base, exp) => Expr::Pow(
                Box::new(base.rename_variables(var_map)),
                Box::new(exp.rename_variables(var_map)),
            ),
            Expr::Exp(expr) => Expr::Exp(Box::new(expr.rename_variables(var_map))),
            Expr::Ln(expr) => Expr::Ln(Box::new(expr.rename_variables(var_map))),
            _ => self.clone(),
        }
    }
    // just shortcut for box
    fn boxed(self) -> Box<Self> {
        Box::new(self)
    }

    pub fn var_expr(&mut self, var: &str) -> Expr {
        let expr = Expr::Var(var.to_string());
        *self = expr.clone();
        expr
    }

    pub fn const_expr(&mut self, val: f64) {
        *self = Expr::Const(val);
    }
    // implementing different functions that are not part of std
    pub fn exp(mut self) -> Expr {
        self = Expr::Exp(self.boxed());
        self
    }
    pub fn ln(mut self) -> Expr {
        self = Expr::Ln(self.boxed());
        self
    }
    pub fn pow(mut self, rhs: Expr) -> Expr {
        self = Expr::Pow(self.boxed(), rhs.boxed());
        self
    }
    pub fn is_zero(&self) -> bool {
        match self {
            Expr::Const(val) => val == &0.0,
            _ => false,
        }
    }
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
        } // end of match
    } // end of lambdify1D

    /// function to lambdify the symbolic function of multiple variables = convert it into a rust function

    pub fn lambdify(&self, vars: Vec<&str>) -> Box<dyn Fn(Vec<f64>) -> f64 + '_> {
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
        }
    } // end of lambdify
    /* 
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
        }
    } // end of lambdify
    */
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
        } // end of match
        vars.dedup();
        vars.sort();
        (args, vars)
    } // end of extract_variables

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

    ///__________________________________INDEXED VARIABLES____________________________________
    pub fn IndexedVar(index: usize, var_name: &str) -> Expr {
        let indexed_var_name = format!("{}{}", var_name, index);
        Expr::Var(indexed_var_name)
    }

    pub fn IndexedVars(num_vars: usize, var_name: &str) -> (Vec<Expr>, Vec<String>) {
        let vec_of_expr = (0..num_vars)
            .map(|i| Expr::IndexedVar(i, var_name))
            .collect();
        let vec_of_names = (0..num_vars)
            .map(|i| format!("{}_{}", var_name, i))
            .collect();
        (vec_of_expr, vec_of_names)
    }
    pub fn IndexedVarsMatrix(
        num_vars: usize,
        var_names: Vec<String>,
    ) -> (Vec<Vec<Expr>>, Vec<Vec<String>>) {
        let mut matrix = Vec::new();
        let mut matrix_of_expr = Vec::new();
        for i in 0..num_vars {
            let mut matrix_i = Vec::new();
            let mut matrix_of_expr_i = Vec::new();
            for j in 0..var_names.len() {
                let indexed_var_name = format!("{}_{}", var_names[j], i);
                matrix_i.push(indexed_var_name.clone());
                matrix_of_expr_i.push(Expr::Var(indexed_var_name));
            }
            matrix.push(matrix_i);
            matrix_of_expr.push(matrix_of_expr_i);
        }
        (matrix_of_expr, matrix)
    }
    // 2D indexation"x2_315", "Z21_235"
    pub fn IndexedVar2D(index_row: usize, index_col: usize, var_name: &str) -> Expr {
        let indexed_var_name = format!("{}_{}_{}", var_name, index_row, index_col);
        Expr::Var(indexed_var_name)
    }

    pub fn IndexedVars2D(
        num_rows: usize,
        num_cols: usize,
        var_name: &str,
    ) -> (Vec<Vec<Expr>>, Vec<String>) {
        let mut vec_of_names: Vec<String> = Vec::new();
        let matrix = (0..num_rows)
            .map(|i| {
                (0..num_cols)
                    .map(|j| {
                        let indexed_var_name = format!("{}_{}_{}", var_name, i, j);

                        Expr::Var(indexed_var_name)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for i in 0..num_rows {
            for j in 0..num_cols {
                let indexed_var_name = format!("{}_{}_{}", var_name, i, j);
                vec_of_names.push(indexed_var_name);
            }
        }
        (matrix, vec_of_names)
    }
    pub fn IndexedVars2Dflat(num_rows: usize, num_cols: usize, var_name: &str) -> Vec<Expr> {
        (0..num_rows)
            .flat_map(|i| (0..num_cols).map(move |j| Expr::IndexedVar2D(i, j, var_name)))
            .collect()
    }
    //___________________________________SYMPLIFICATE____________________________________
    //  function to symplify the symbolic expression in sych way: if in expression there is a
    // subexpression Mul(Expr::Const(0.0), ...something ) the function turns all subexpression into Expr::Const(0.0)
    #[allow(dead_code)]
    fn nozeros(&self) -> Expr {
        match self {
            Expr::Var(_) => self.clone(),
            Expr::Const(_) => self.clone(),
            Expr::Add(lhs, rhs) => {
                let simplified_lhs = lhs.nozeros();
                let simplified_rhs = rhs.nozeros();
                if simplified_lhs == Expr::Const(0.0) && simplified_rhs == Expr::Const(0.0) {
                    Expr::Const(0.0)
                } else {
                    Expr::Add(Box::new(simplified_lhs), Box::new(simplified_rhs))
                }
            }
            Expr::Sub(lhs, rhs) => {
                let simplified_lhs = lhs.nozeros();
                let simplified_rhs = rhs.nozeros();
                if simplified_lhs == Expr::Const(0.0) && simplified_rhs == Expr::Const(0.0) {
                    Expr::Const(0.0)
                } else {
                    Expr::Sub(Box::new(simplified_lhs), Box::new(simplified_rhs))
                }
            }
            Expr::Mul(lhs, rhs) => {
                let simplified_lhs = lhs.nozeros();
                let simplified_rhs = rhs.nozeros();
                if simplified_lhs == Expr::Const(0.0) || simplified_rhs == Expr::Const(0.0) {
                    Expr::Const(0.0)
                } else {
                    Expr::Mul(Box::new(simplified_lhs), Box::new(simplified_rhs))
                }
            }
            Expr::Div(lhs, rhs) => {
                let simplified_lhs = lhs.nozeros();
                let simplified_rhs = rhs.nozeros();
                if simplified_lhs == Expr::Const(0.0) {
                    Expr::Const(0.0)
                } else {
                    Expr::Div(Box::new(simplified_lhs), Box::new(simplified_rhs))
                }
            }
            Expr::Pow(base, exp) => Expr::Pow(Box::new(base.nozeros()), Box::new(exp.nozeros())),
            Expr::Exp(expr) => Expr::Exp(Box::new(expr.nozeros())),
            Expr::Ln(expr) => Expr::Ln(Box::new(expr.nozeros())),
        }
    } // nozeros

    pub fn simplify_numbers(&self) -> Expr {
        match self {
            Expr::Var(_) => self.clone(),
            Expr::Const(_) => self.clone(),
            Expr::Add(lhs, rhs) => {
                let lhs_simplified = lhs.simplify_numbers();
                let rhs_simplified = rhs.simplify_numbers();
                match (lhs_simplified, rhs_simplified) {
                    (Expr::Const(a), Expr::Const(b)) => Expr::Const(a + b),
                    (lhs, rhs) => Expr::Add(Box::new(lhs), Box::new(rhs)),
                }
            }
            Expr::Sub(lhs, rhs) => {
                let lhs_simplified = lhs.simplify_numbers();
                let rhs_simplified = rhs.simplify_numbers();
                match (lhs_simplified, rhs_simplified) {
                    (Expr::Const(a), Expr::Const(b)) => Expr::Const(a - b),
                    (lhs, rhs) => Expr::Sub(Box::new(lhs), Box::new(rhs)),
                }
            }
            Expr::Mul(lhs, rhs) => {
                let lhs_simplified = lhs.simplify_numbers();
                let rhs_simplified = rhs.simplify_numbers();
                match (lhs_simplified, rhs_simplified) {
                    (Expr::Const(a), Expr::Const(b)) => Expr::Const(a * b),
                    (lhs, rhs) => Expr::Mul(Box::new(lhs), Box::new(rhs)),
                }
            }
            Expr::Div(lhs, rhs) => {
                let lhs_simplified = lhs.simplify_numbers();
                let rhs_simplified = rhs.simplify_numbers();
                match (lhs_simplified, rhs_simplified) {
                    (Expr::Const(a), Expr::Const(b)) => Expr::Const(a / b),
                    (lhs, rhs) => Expr::Div(Box::new(lhs), Box::new(rhs)),
                }
            }
            Expr::Pow(base, exp) => Expr::Pow(
                Box::new(base.simplify_numbers()),
                Box::new(exp.simplify_numbers()),
            ),
            Expr::Exp(expr) => Expr::Exp(Box::new(expr.simplify_numbers())),
            Expr::Ln(expr) => Expr::Ln(Box::new(expr.simplify_numbers())),
        }
    }
    pub fn simplify_(&self) -> Expr {
        match self {
            Expr::Var(_) => self.clone(),
            Expr::Const(_) => self.clone(),
            Expr::Add(lhs, rhs) => {
                let lhs = lhs.simplify_();
                let rhs = rhs.simplify_();
                match (&lhs, &rhs) {
                    (Expr::Const(a), Expr::Const(b)) =>Expr::Const(a + b),// (a) + (b) = (a + b) 
                    (Expr::Const(0.0), _) => rhs, // x + 0 = x
                    (_, Expr::Const(0.0)) => lhs,//  0 + x = x
                    _ => Expr::Add(Box::new(lhs), Box::new(rhs) ),
                }
            }
            Expr::Sub(lhs, rhs) => {
                let lhs = lhs.simplify_();
                let rhs = rhs.simplify_();
                match (&lhs, &rhs) {
                    (Expr::Const(a), Expr::Const(b)) =>Expr::Const(a - b),// (a) - (b) = (a - b)
                    (_, Expr::Const(0.0)) => lhs, // x - 0 = x
                    _ =>Expr::Sub(Box::new(lhs), Box::new(rhs)),
                }
            }
            Expr::Mul(lhs, rhs) => {
                let lhs = lhs.simplify_();
                let rhs = rhs.simplify_();
                match (&lhs, &rhs) {
                    (Expr::Const(a), Expr::Const(b)) => Expr::Const(a * b),// (a) * (b) = (a * b)
                    (Expr::Const(0.0), _) | (_, Expr::Const(0.0)) =>Expr::Const(0.0), // 0 * x = 0 or 0*x = 0
                    (Expr::Const(1.0), _) => rhs, // 1 * x = x
                    (_, Expr::Const(1.0)) => lhs, // x * 1 = x
                    _ => Expr::Mul(Box::new(lhs), Box::new(rhs)),
                }
            }
            Expr::Div(lhs, rhs) => {
                let lhs = lhs.simplify_();
                let rhs = rhs.simplify_();
                match (&lhs, &rhs) {
                    (Expr::Const(a), Expr::Const(b)) if *b != 0.0 => Expr::Const(a / b),// (a) / (b) = (a / b)
                    (Expr::Const(0.0), _) => Expr::Const(0.0),// (0.0) / x = 0.0
                    (_, Expr::Const(1.0)) => lhs,// x / 1.0 = x
                    _ => Expr::Div(Box::new(lhs), Box::new(rhs)),
                }
            }
            Expr::Pow(base, exp) => {
                let base = base.simplify_();
                let exp = exp.simplify_();
                match (&base, &exp) {
                    (Expr::Const(a), Expr::Const(b)) => Expr::Const(a.powf(*b)), // (a) ^ (b) = (a ^ b)
                    (_, Expr::Const(0.0)) => Expr::Const(1.0), // x ^ 0 = 1
                    (_, Expr::Const(1.0)) => base, // x ^ 1 = x
                    (Expr::Const(0.0), _) => Expr::Const(0.0), // 0 ^ x = 0
                    (Expr::Const(1.0), _) => Expr::Const(1.0), // 1 ^ x = 1
                    _ => Expr::Pow(Box::new(base), Box::new(exp)),
                }
            }
            Expr::Exp(expr) => {
                let expr = expr.simplify_();
                match &expr {
                    Expr::Const(a) if a!=&0.0 =>Expr::Const(a.exp()),
                    Expr::Const(0.0) => Expr::Const(1.0),
                    _ => Expr::Exp(Box::new(expr)),
                }
            }
            Expr::Ln(expr) => {
                let expr = expr.simplify_();
                match &expr {
                    Expr::Const(1.0) => Expr::Const(0.0),
                    Expr::Const(a) if *a > 0.0 => Expr::Const(a.ln()),
                    _ => Expr::Ln(Box::new(expr)),
                }
            }
        }
    }
    pub fn symplify(&self) -> Expr {
        //let zeros_proceeded = self.nozeros().simplify_numbers();
        let zeros_proceeded = self.simplify_();
        zeros_proceeded
    }


}

//___________________________________TESTS____________________________________

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diff() {
        let x = Expr::Var("x".to_string());
        let f = Expr::Pow(Box::new(x.clone()), Box::new(Expr::Const(2.0)));
        let df_dx = f.diff("x");
        let _degree = Box::new(Expr::Const(1.0));
        let C = Expr::Const(2.0);
        let C1 = Expr::Const(1.0);

        let expected_result = C.clone() * Expr::pow(x.clone(), C.clone() - C1.clone()) * C1.clone();
        //  Mul(Mul(Const(2.0), Pow(Var("x"), Sub(Const(2.0), Const(1.0)))), Const(1.0)) Box::new(Expr::Mul(Box::new(Expr::Const(2.0)), Box::new(x.clone())))
        println!("df_dx {:?} ", df_dx);
        println!("expected_result {:?} ", expected_result);
        assert_eq!(df_dx, expected_result);
    }

    #[test]
    fn test_sym_to_str() {
        let x = Expr::Var("x".to_string());
        let f = Expr::Pow(Box::new(x.clone()), Box::new(Expr::Const(2.0)));
        let rust_function = f.sym_to_str("x");
        assert_eq!(rust_function, "(x^2)");
    }

    #[test]
    fn test_lambdify1D() {
        let x = Expr::Var("x".to_string());
        let f = Expr::Pow(Box::new(x.clone()), Box::new(Expr::Const(2.0)));
        let fn_closure = f.lambdify1D();
        assert_eq!(fn_closure(2.0), 4.0);
    }
    #[test]
    fn test_constuction_of_expression() {
        let vector_of_symbolic_vars = Expr::Symbols("a, b, c");
        let (a, b, c) = (
            vector_of_symbolic_vars[0].clone(),
            vector_of_symbolic_vars[1].clone(),
            vector_of_symbolic_vars[2].clone(),
        );
        let symbolic_expression = a + Expr::exp(b * c);
        let expression_with_const = symbolic_expression.set_variable("a", 1.0);
        let parsed_function = expression_with_const.sym_to_str("a");
        assert_eq!(parsed_function, "(1) + (exp((b) * (c)))");
    }
    #[test]
    fn test_1D() {
        let input = "log(x)";
        let f = Expr::parse_expression(input);
        let f_res = f.lambdify1D()(1.0);
        assert_eq!(f_res, 0.0);
        let df_dx = f.diff("x");
        let df_dx_str = df_dx.sym_to_str("x");
        assert_eq!(df_dx_str, "(1) / (x)");
    }
    #[test]
    fn test_1D_2() {
        let input = "x+exp(x)";
        let f = Expr::parse_expression(input);
        let f_res = f.lambdify1D()(1.0);
        assert_eq!(f_res, 1.0 + f64::consts::E);
        let start = 0.0;
        let end = 10f64;
        let num_values = 100;
        let max_norm = 1e-6;
        let (_normm, res) = f.compare_num1D("x", start, end, num_values, max_norm);
        assert_eq!(res, true);
    }
    #[test]
    fn test_multi_diff() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let C = Expr::Const(3.0);
        let f = Expr::pow(x.clone(), C.clone()) + Expr::exp(y.clone());
        let df_dx = f.diff("x");
        //  let df_dy = f.diff("y");

        let C1 = Expr::Const(1.0);
        let C0 = Expr::Const(0.0);
        let df_dx_expected_result =
            C.clone() * Expr::pow(x, C - C1.clone()) * C1 + Expr::exp(y.clone()) * C0;
        //  let df_dy_expected_result = C* Expr::exp(y);
        assert_eq!(df_dx, df_dx_expected_result);
        let start = vec![1.0, 1.0];
        let end = vec![2.0, 2.0];
        let comparsion = f.compare_num(start, end, 100, 1e-6);
        let bool_1 = &comparsion[0].0;
        let bool_2 = &comparsion[1].0;

        assert_eq!(*bool_1 && *bool_2, true);
        //    assert_eq!(df_dy, expected_result);
    }

    #[test]
    fn test_set_variable() {
        let x = Expr::Var("x".to_string());
        let f = x.clone() + Expr::Const(2.0);
        let f_with_value = f.set_variable("x", 1.0);
        let expected_result = Expr::Const(1.0) + Expr::Const(2.0);
        assert_eq!(f_with_value, expected_result);
    }

    #[test]
    fn test_calc_vector_lambdified1D() {
        let x = Expr::Var("x".to_string());
        let f = Expr::Pow(Box::new(x.clone()), Box::new(Expr::Const(2.0)));
        let _fn_closureee = f.lambdify1D();
        let x_values = vec![1.0, 2.0, 3.0];
        let result = f.calc_vector_lambdified1D(&x_values);
        let expected_result = vec![1.0, 4.0, 9.0];
        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_lambdify1D_from_linspace() {
        let x = Expr::Var("x".to_string());
        let f = Expr::Pow(Box::new(x.clone()), Box::new(Expr::Const(2.0)));
        let result = f.lambdify1D_from_linspace(1.0, 3.0, 3);
        let expected_result = vec![1.0, 4.0, 9.0];
        assert_eq!(result, expected_result);
    }
    /*
    #[test]
    fn test_evaluate_vector_lambdified() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let f = Expr::Pow(Box::new(x.clone()), Box::new(Expr::Const(2.0))) + Expr::exp(y.clone());
        let x_values = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let y_values = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let result = f.evaluate_vector_lambdified(&x_values, &y_values);
        let expected_result = vec![1.0 + 27.18281828459045, 4.0 + 74.08182845904523, 9.0 + 162.31828459045235];
        assert_eq!(result, expected_result);
    }

    */
    #[test]
    fn test_evaluate_multi_diff_from_linspace() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let f = Expr::Pow(Box::new(x.clone()), Box::new(Expr::Const(2.0))) + Expr::exp(y.clone());
        let result = f.evaluate_multi_diff_from_linspace(vec![1.0, 1.0], vec![2.0, 2.0], 100);
        let last_element = result[0].last().unwrap();

        let expected_result: f64 = 4.0f64; // 2*2
        assert!((last_element - expected_result).abs() < f64::EPSILON);
    }
    #[test]
    fn lambdify_IVP_test() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let z: Expr = Expr::Var("z".to_string());
        let symbolic: Expr = z * x + Expr::exp(y);
        let func = symbolic.lambdify_IVP("x", vec!["y", "z"]);
        let result = func(1.0, vec![0.0, 1.0]);
        println!("result {}", result);
        let expected_result: f64 = 2.0f64; // 2*2

        assert_eq!(result, expected_result);
    }
    #[test]
    fn lambdify_IVP_owned_test() {
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let z: Expr = Expr::Var("z".to_string());
        let symbolic: Expr = z * x + Expr::exp(y);
        let func = symbolic.lambdify_IVP_owned("x", vec!["y", "z"]);
        let result = func(1.0, vec![0.0, 1.0]);
        println!("result {}", result);
        let expected_result: f64 = 2.0f64; // 2*2

        assert_eq!(result, expected_result);
    }
    #[test]
    fn no_zeros_test() {
        let expr = Expr::Mul(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Const(0.0)),
        );

        let simplified_expr = expr.symplify();
        let expected_result = Expr::Const(0.0);
        assert_eq!(simplified_expr, expected_result);
    }
    #[test]
    fn no_zeros_test2() {
        let expr = Expr::Sub(Box::new(Expr::Const(0.0)), Box::new(Expr::Const(0.0)));

        let simplified_expr = expr.symplify();
        let expected_result = Expr::Const(0.0);
        assert_eq!(simplified_expr, expected_result);
    }
    #[test]
    fn no_zeros_test3() {
        let expr = Expr::Add(Box::new(Expr::Const(0.0)), Box::new(Expr::Const(0.0)));

        let simplified_expr = expr.symplify();
        let expected_result = Expr::Const(0.0);
        assert_eq!(simplified_expr, expected_result);
    }

    #[test]
    fn no_zeros_test4() {
        let zero = Box::new(Expr::Const(0.0));
        let added = Expr::Add(zero.clone(), zero.clone()); // 0
        let mulled = Expr::Mul(Box::new(Expr::Const(0.005)), Box::new(added)); //0
        let expr = Box::new(Expr::Sub(
            zero.clone(),
            Box::new(Expr::Add(zero, Box::new(mulled))),
        ));

        let simplified_expr = expr.symplify();
        let expected_result = Expr::Const(0.0);
        assert_eq!(simplified_expr, expected_result);
    }
}
