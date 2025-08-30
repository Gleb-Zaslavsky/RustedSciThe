//! # Symbolic Engine Derivatives Module
//!
//! This module extends the symbolic engine with advanced differentiation, evaluation,
//! and function conversion capabilities. It provides the computational backbone for
//! converting symbolic expressions into executable functions and performing numerical
//! analysis operations.
//!
//! ## Purpose
//!
//! This module enables:
//! - **Analytical Differentiation**: Automatic symbolic differentiation using calculus rules
//! - **Function Lambdification**: Converting symbolic expressions to executable Rust closures
//! - **Numerical Validation**: Comparing analytical derivatives with numerical approximations
//! - **Multi-dimensional Analysis**: Handling functions of multiple variables
//! - **Taylor Series Expansion**: Symbolic polynomial approximations
//! - **Expression Parsing**: Converting string representations to symbolic expressions
//!
//! ## Key Methods
//!
//! ### Differentiation
//! - `diff(var: &str)` - Analytical partial/total derivative
//! - `diff_multi()` - All partial derivatives at once
//! - `n_th_derivative1D()` - Higher-order derivatives
//!
//! ### Function Conversion (Lambdification)
//! - `lambdify1D()` - Single variable functions
//! - `lambdify()` - Multi-variable functions
//! - `lambdify_IVP()` - Initial Value Problem format
//! - `eval_expression()` - Direct evaluation without closure creation
//!
//! ### Numerical Analysis
//! - `compare_num1D()` - Validate 1D derivatives numerically
//! - `compare_num()` - Validate multi-dimensional derivatives
//! - `taylor_series1D()` - Symbolic Taylor expansions
//!
//! ### Parsing and Utilities
//! - `parse_expression()` - String to symbolic expression
//! - `sym_to_str()` - Symbolic expression to string
//! - `all_arguments_are_variables()` - Extract variable names
//!
//! ## Interesting Code Features
//!
//! 1. **Recursive Differentiation Rules**: Implements complete calculus rules including
//!    product rule, quotient rule, chain rule for all supported functions
//!
//! 2. **Closure Generation**: Creates optimized Rust closures from symbolic expressions,
//!    enabling high-performance numerical computation
//!
//! 3. **Memory Management Variants**: Provides both borrowing (`lambdify`) and owned
//!    (`lambdify_owned`) versions for different lifetime requirements
//!
//! 4. **Numerical Validation**: Sophisticated comparison between analytical and numerical
//!    derivatives using configurable tolerance and step sizes
//!
//! 5. **Multi-dimensional Support**: Handles arbitrary-dimension functions with proper
//!    matrix transposition for efficient evaluation
//!
//! 6. **Taylor Series Implementation**: Recursive symbolic Taylor expansion with
//!    factorial computation and term generation
//!
//! 7. **IVP Specialization**: Special lambdification for Initial Value Problems where
//!    one argument (time) is treated differently from state variables

use crate::symbolic::parse_expr::parse_expression_func;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::utils::{
    linspace, norm, numerical_derivative, numerical_derivative_multi, transpose,
};
use std::f64::consts::PI;

impl Expr {
    /// DIFFERENTIATION

    /// Computes the analytical derivative of the expression with respect to a variable.
    ///
    /// Implements all standard differentiation rules from calculus:
    /// - Power rule: d/dx(x^n) = n*x^(n-1)
    /// - Product rule: d/dx(f*g) = f'*g + f*g'
    /// - Quotient rule: d/dx(f/g) = (f'*g - f*g')/g^2
    /// - Chain rule: d/dx(f(g(x))) = f'(g(x))*g'(x)
    ///
    /// For multivariable functions, computes partial derivatives.
    ///
    /// # Arguments
    /// * `var` - Variable name to differentiate with respect to
    ///
    /// # Returns
    /// New symbolic expression representing the derivative
    ///
    /// # Examples
    /// ```rust, ignore
    /// let x = Expr::Var("x".to_string());
    /// let f = x.clone().pow(Expr::Const(2.0)); // x^2
    /// let df_dx = f.diff("x"); // 2*x
    /// ```
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

    /// Converts symbolic expression to human-readable string representation.
    ///
    /// Generates mathematical notation with proper parentheses for precedence.
    /// Uses standard mathematical symbols and function names.
    ///
    /// # Arguments
    /// * `var` - Primary variable name (used for context, but all variables are converted)
    ///
    /// # Returns
    /// String representation of the expression
    ///
    /// # Examples
    /// ```rust, ignore
    /// let expr = Expr::Add(Box::new(Expr::Var("x".to_string())), Box::new(Expr::Const(2.0)));
    /// assert_eq!(expr.sym_to_str("x"), "(x) + (2)");
    /// ```
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
    /// LAMBDIFICATION - Converting Symbolic Expressions to Executable Functions

    /// Converts a single-variable symbolic expression into an executable Rust closure.
    ///
    /// This is the core method for numerical computation, transforming symbolic math
    /// into high-performance executable code. The resulting closure can be called
    /// repeatedly with different input values.
    ///
    /// # Returns
    /// Boxed closure that takes f64 input and returns f64 output
    ///
    /// # Performance Notes
    /// - Creates optimized closure with minimal overhead
    /// - Recursive structure mirrors expression tree for efficiency
    /// - No runtime parsing or interpretation
    ///
    /// # Examples
    /// ```rust, ignore
    /// let x = Expr::Var("x".to_string());
    /// let f = x.pow(Expr::Const(2.0)); // x^2
    /// let func = f.lambdify1D();
    /// assert_eq!(func(3.0), 9.0);
    /// ```
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
                Box::new(move |x| PI / 2.0 - expr_fn(x).atan())
            }
        } // end of match
    } // end of lambdify1D

    /// Converts multi-variable symbolic expression into executable Rust closure.
    ///
    /// Handles functions of arbitrary dimension f(x1, x2, ..., xn). Variables are
    /// matched by position in the vars vector to values in the input vector.
    ///
    /// # Arguments
    /// * `vars` - Ordered list of variable names matching input vector positions
    ///
    /// # Returns
    /// Boxed closure taking Vec<f64> input and returning f64 output
    ///
    /// # Performance Notes
    /// - Pre-computes variable indices for fast lookup
    /// - Avoids repeated string comparisons during evaluation
    /// - Clones input vector for recursive calls (see lambdify_slice for zero-copy)
    ///
    /// # Examples
    /// ```rust, ignore
    /// let expr = x + y; // symbolic expression
    /// let func = expr.lambdify(vec!["x", "y"]);
    /// assert_eq!(func(vec![2.0, 3.0]), 5.0);
    /// ```
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
                Box::new(move |args| PI / 2.0 - expr_fn(args).tan())
            }
        }
    } // end of lambdify

    /// Zero-copy version of lambdify using slice references instead of owned vectors.
    ///
    /// More memory-efficient than lambdify() as it avoids vector cloning during
    /// recursive evaluation. Useful for high-frequency evaluation scenarios.
    ///
    /// # Arguments
    /// * `vars` - Variable names in order matching slice positions
    ///
    /// # Returns
    /// Boxed closure taking &[f64] slice and returning f64
    ///
    /// # Lifetime
    /// Returned closure borrows from self, limiting its lifetime
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
                Box::new(move |args| PI / 2.0 - expr_fn(args).atan())
            }
        }
    } // end of lambdify

    /// Owned version of lambdify that consumes the expression.
    ///
    /// Takes ownership of the expression, allowing the returned closure to have
    /// 'static lifetime. Useful when the closure needs to outlive the original
    /// expression or be moved across thread boundaries.
    ///
    /// # Arguments
    /// * `vars` - Variable names in order matching input vector positions
    ///
    /// # Returns
    /// Boxed closure with 'static lifetime
    ///
    /// # Ownership
    /// Consumes self, so the original expression cannot be used afterward
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
                Box::new(move |args| PI / 2.0 - expr_fn(args).atan())
            }
        }
    }

    pub fn lambdify_thread_safe(
        self,
        vars: Vec<&str>,
    ) -> Box<dyn Fn(Vec<f64>) -> f64 + Send + Sync> {
        match self {
            Expr::Var(name) => {
                let index = vars.iter().position(|&x| x == name).unwrap();
                Box::new(move |args| args[index])
            }
            Expr::Const(val) => Box::new(move |_| val),

            Expr::Add(lhs, rhs) => {
                let lhs_fn = lhs.lambdify_thread_safe(vars.clone());
                let rhs_fn = rhs.lambdify_thread_safe(vars);
                Box::new(move |args| lhs_fn(args.clone()) + rhs_fn(args))
            }
            Expr::Sub(lhs, rhs) => {
                let lhs_fn = lhs.lambdify_thread_safe(vars.clone());
                let rhs_fn = rhs.lambdify_thread_safe(vars);
                Box::new(move |args| lhs_fn(args.clone()) - rhs_fn(args))
            }
            Expr::Mul(lhs, rhs) => {
                let lhs_fn = lhs.lambdify_thread_safe(vars.clone());
                let rhs_fn = rhs.lambdify_thread_safe(vars);
                Box::new(move |args| lhs_fn(args.clone()) * rhs_fn(args))
            }
            Expr::Div(lhs, rhs) => {
                let lhs_fn = lhs.lambdify_thread_safe(vars.clone());
                let rhs_fn = rhs.lambdify_thread_safe(vars);
                Box::new(move |args| lhs_fn(args.clone()) / rhs_fn(args))
            }
            Expr::Pow(base, exp) => {
                let base_fn = base.lambdify_thread_safe(vars.clone());
                let exp_fn = exp.lambdify_thread_safe(vars);
                Box::new(move |args| base_fn(args.clone()).powf(exp_fn(args)))
            }
            Expr::Exp(expr) => {
                let expr_fn = expr.lambdify_thread_safe(vars);
                Box::new(move |args| expr_fn(args).exp())
            }
            Expr::Ln(expr) => {
                let expr_fn = expr.lambdify_thread_safe(vars);
                Box::new(move |args| expr_fn(args).ln())
            }
            Expr::sin(expr) => {
                let expr_fn = expr.lambdify_thread_safe(vars);
                Box::new(move |args| expr_fn(args).sin())
            }
            Expr::cos(expr) => {
                let expr_fn = expr.lambdify_thread_safe(vars);
                Box::new(move |args| expr_fn(args).cos())
            }
            Expr::tg(expr) => {
                let expr_fn = expr.lambdify_thread_safe(vars);
                Box::new(move |args| expr_fn(args).tan())
            }
            Expr::ctg(expr) => {
                let expr_fn = expr.lambdify_thread_safe(vars);
                Box::new(move |args| 1.0 / expr_fn(args).tan())
            }
            Expr::arcsin(expr) => {
                let expr_fn = expr.lambdify_thread_safe(vars);
                Box::new(move |args| expr_fn(args).asin())
            }
            Expr::arccos(expr) => {
                let expr_fn = expr.lambdify_thread_safe(vars);
                Box::new(move |args| expr_fn(args).acos())
            }
            Expr::arctg(expr) => {
                let expr_fn = expr.lambdify_thread_safe(vars);
                Box::new(move |args| expr_fn(args).atan())
            }
            Expr::arcctg(expr) => {
                let expr_fn = expr.lambdify_thread_safe(vars);
                Box::new(move |args| PI / 2.0 - expr_fn(args).atan())
            }
        }
    }

      pub fn lambdify_borrowed_thread_safe(&self, vars: Vec<&str>) -> Box<dyn Fn(Vec<f64>) -> f64 + Send + Sync > {

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
                let lhs_fn = lhs.lambdify_borrowed_thread_safe(vars.clone());
                let rhs_fn = rhs.lambdify_borrowed_thread_safe(vars);
                Box::new(move |args| lhs_fn(args.clone()) + rhs_fn(args))
            }
            Expr::Sub(lhs, rhs) => {
                let lhs_fn = lhs.lambdify_borrowed_thread_safe(vars.clone());
                let rhs_fn = rhs.lambdify_borrowed_thread_safe(vars);
                Box::new(move |args| lhs_fn(args.clone()) - rhs_fn(args))
            }
            Expr::Mul(lhs, rhs) => {
                let lhs_fn = lhs.lambdify_borrowed_thread_safe(vars.clone());
                let rhs_fn = rhs.lambdify_borrowed_thread_safe(vars);
                Box::new(move |args| lhs_fn(args.clone()) * rhs_fn(args))
            }
            Expr::Div(lhs, rhs) => {
                let lhs_fn = lhs.lambdify_borrowed_thread_safe(vars.clone());
                let rhs_fn = rhs.lambdify_borrowed_thread_safe(vars);
                Box::new(move |args| lhs_fn(args.clone()) / rhs_fn(args))
            }
            Expr::Pow(base, exp) => {
                let base_fn = base.lambdify_borrowed_thread_safe(vars.clone());
                let exp_fn = exp.lambdify_borrowed_thread_safe(vars);
                Box::new(move |args| base_fn(args.clone()).powf(exp_fn(args)))
            }
            Expr::Exp(expr) => {
                let expr_fn = expr.lambdify_borrowed_thread_safe(vars);
                Box::new(move |args| expr_fn(args).exp())
            }
            Expr::Ln(expr) => {
                let expr_fn = expr.lambdify_borrowed_thread_safe(vars);
                Box::new(move |args| expr_fn(args).ln())
            }
            Expr::sin(expr) => {
                let expr_fn = expr.lambdify_borrowed_thread_safe(vars);
                Box::new(move |args| expr_fn(args).sin())
            }
            Expr::cos(expr) => {
                let expr_fn = expr.lambdify_borrowed_thread_safe(vars);
                Box::new(move |args| expr_fn(args).cos())
            }
            Expr::tg(expr) => {
                let expr_fn = expr.lambdify_borrowed_thread_safe(vars);
                Box::new(move |args| expr_fn(args).tan())
            }
            Expr::ctg(expr) => {
                let expr_fn = expr.lambdify_borrowed_thread_safe(vars);
                Box::new(move |args| {
                    let args = args.iter().map(|x| 1.0 / x).collect::<Vec<f64>>();
                    expr_fn(args).tan()
                })
            }
            Expr::arcsin(expr) => {
                let expr_fn = expr.lambdify_borrowed_thread_safe(vars);
                Box::new(move |args| expr_fn(args).asin())
            }
            Expr::arccos(expr) => {
                let expr_fn = expr.lambdify_borrowed_thread_safe(vars);
                Box::new(move |args| expr_fn(args).acos())
            }
            Expr::arctg(expr) => {
                let expr_fn = expr.lambdify_borrowed_thread_safe(vars);
                Box::new(move |args| expr_fn(args).atan())
            }
            Expr::arcctg(expr) => {
                let expr_fn = expr.lambdify_borrowed_thread_safe(vars);
                Box::new(move |args| PI / 2.0 - expr_fn(args).tan())
            }
        }
    } // end of lambdify
    /// Convenience method that automatically detects variables and creates a closure.
    ///
    /// Extracts all variables from the expression and creates a lambdified function
    /// with variables ordered alphabetically. Eliminates need to manually specify
    /// variable names.
    ///
    /// # Returns
    /// Boxed closure where input vector positions correspond to alphabetically
    /// sorted variable names
    ///
    /// # Usage
    /// Ideal for quick evaluation when variable order doesn't matter
    pub fn lambdify_wrapped(&self) -> Box<dyn Fn(Vec<f64>) -> f64 + '_> {
        let vars_ = self.all_arguments_are_variables();
        let vars = vars_.iter().map(|x| x.as_str()).collect::<Vec<&str>>();
        self.lambdify(vars)
    }

    /// Extracts all variables from the expression with structural information.
    ///
    /// Performs recursive traversal of the expression tree to find all symbolic
    /// variables. Returns both the variables and structural indices indicating
    /// their positions in the expression tree.
    ///
    /// # Returns
    /// Tuple of (position_indices, variable_names)
    /// - position_indices: Vec<i32> indicating structural positions
    /// - variable_names: Vec<String> of unique variable names (sorted, deduplicated)
    ///
    /// # Note
    /// This method provides more structural information than all_arguments_are_variables()
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
    /// TAYLOR SERIES EXPANSION

    /// Computes the nth derivative of a single-variable expression.
    ///
    /// Repeatedly applies differentiation and simplification to obtain higher-order
    /// derivatives. Used internally for Taylor series expansion.
    ///
    /// # Arguments
    /// * `var_name` - Variable to differentiate with respect to
    /// * `n` - Order of derivative (0 = original function, 1 = first derivative, etc.)
    ///
    /// # Returns
    /// Symbolic expression representing the nth derivative
    pub fn n_th_derivative1D(&self, var_name: &str, n: usize) -> Expr {
        let mut expr = self.clone();
        let mut i = 0;
        while i < n {
            expr = expr.diff(var_name).symplify();
            i += 1;
        }
        return expr.symplify();
    }

    /// Computes Taylor series expansion around a point (recursive implementation).
    ///
    /// Generates polynomial approximation f(x) ≈ Σ(f^(n)(x0)/n!) * (x-x0)^n
    /// Uses recursive approach where each term is computed independently.
    ///
    /// # Arguments
    /// * `var_name` - Variable for expansion
    /// * `x0` - Expansion point
    /// * `order` - Maximum order of expansion
    ///
    /// # Returns
    /// Symbolic polynomial approximating the original function
    ///
    /// # Note
    /// This is the original recursive implementation. See taylor_series1D_() for optimized version.
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

    /// Optimized Taylor series expansion using iterative derivative computation.
    ///
    /// More efficient than taylor_series1D() as it reuses derivative calculations
    /// and avoids redundant recursive calls. Computes derivatives incrementally
    /// rather than from scratch for each term.
    ///
    /// # Arguments
    /// * `var_name` - Variable for expansion
    /// * `x0` - Expansion point
    /// * `order` - Maximum order of expansion
    ///
    /// # Returns
    /// Symbolic polynomial approximating the original function
    ///
    /// # Performance
    /// Preferred over taylor_series1D() for higher-order expansions
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
    /// DIRECT EXPRESSION EVALUATION

    /// Evaluates symbolic expression directly without creating a closure.
    ///
    /// More memory-efficient than lambdification for single-use evaluation.
    /// Recursively evaluates the expression tree with given variable values.
    ///
    /// # Arguments
    /// * `vars` - Variable names in order matching values array
    /// * `values` - Numerical values for each variable
    ///
    /// # Returns
    /// Numerical result of expression evaluation
    ///
    /// # Performance
    /// Use lambdify() for repeated evaluation, eval_expression() for one-time use
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
                PI / 2.0 - (expr_fn).atan()
            }
        }
    } // end of eval_expression

    /// EXPRESSION PARSING FROM STRINGS

    /// Parses a mathematical expression from string representation.
    ///
    /// Converts human-readable mathematical notation into symbolic expression tree.
    /// Supports standard mathematical operators, functions, and parentheses.
    ///
    /// # Arguments
    /// * `input` - String containing mathematical expression (e.g., "x^2 + sin(y)")
    ///
    /// # Returns
    /// Parsed symbolic expression
    ///
    /// # Panics
    /// Panics if the expression cannot be parsed (invalid syntax)
    ///
    /// # Examples
    /// ```rust, ignore
    /// let expr = Expr::parse_expression("x^2 + 2*x + 1");
    /// ```
    ///
    /// # Supported Syntax
    /// - Variables: x, y, var_name
    /// - Constants: 3.14, -2.5, 1e-6
    /// - Operators: +, -, *, /, ^
    /// - Functions: sin, cos, exp, ln, etc.
    /// - Parentheses for grouping
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

    /// Parses multiple expressions from a vector of strings.
    ///
    /// Convenience method for parsing systems of equations or multiple functions.
    /// Each string is parsed independently into a symbolic expression.
    ///
    /// # Arguments
    /// * `input` - Vector of expression strings
    ///
    /// # Returns
    /// Vector of parsed symbolic expressions
    ///
    /// # Panics
    /// Panics if any expression cannot be parsed
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
    /// Extracts all unique variable names from the symbolic expression.
    ///
    /// Recursively traverses the expression tree to collect all symbolic variables.
    /// Returns a sorted, deduplicated list of variable names.
    ///
    /// # Returns
    /// Vector of unique variable names in alphabetical order
    ///
    /// # Examples
    /// ```rust, ignore
    /// let expr = parse_expression("x^2 + y*z + x");
    /// let vars = expr.all_arguments_are_variables();
    /// assert_eq!(vars, vec!["x", "y", "z"]);
    /// ```
    ///
    /// # Usage
    /// Essential for automatic variable detection in lambdify_wrapped()
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
    //                    1D FUNCTION PROCESSING - Single Variable Functions y = f(x)
    // _________________________________________________________________________________________________________________

    /// Evaluates 1D function over a vector of input values.
    ///
    /// Creates a lambdified function once and applies it to each input value.
    /// More efficient than repeated eval_expression calls.
    ///
    /// # Arguments
    /// * `x` - Vector of input values
    ///
    /// # Returns
    /// Vector of function evaluations f(x[i])
    pub fn calc_vector_lambdified1D(&self, x: &Vec<f64>) -> Vec<f64> {
        let mut result = Vec::new();
        for xi in x {
            result.push(self.lambdify1D()(*xi));
        }
        result
    }

    /// Evaluates 1D function over a linearly spaced domain.
    ///
    /// Convenience method combining linspace generation with function evaluation.
    /// Useful for plotting and numerical analysis.
    ///
    /// # Arguments
    /// * `start` - Domain start value
    /// * `end` - Domain end value
    /// * `num_values` - Number of evaluation points
    ///
    /// # Returns
    /// Vector of function values over the specified domain
    pub fn lambdify1D_from_linspace(&self, start: f64, end: f64, num_values: usize) -> Vec<f64> {
        let x = linspace(start, end, num_values);
        self.calc_vector_lambdified1D(&x)
    }

    /// Validates analytical derivative against numerical approximation for 1D functions.
    ///
    /// Computes both analytical and numerical derivatives over a domain and compares
    /// their L2 norm difference. Essential for verifying differentiation correctness.
    ///
    /// # Arguments
    /// * `var` - Variable to differentiate with respect to
    /// * `start` - Domain start
    /// * `end` - Domain end
    /// * `num_values` - Number of test points
    /// * `max_norm` - Maximum acceptable norm difference
    ///
    /// # Returns
    /// Tuple of (actual_norm, is_within_tolerance)
    ///
    /// # Algorithm
    /// Uses finite difference approximation with adaptive step size
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
    //                     MULTI-DIMENSIONAL FUNCTION PROCESSING - Functions y = f(x1, x2, x3, ...)
    // _________________________________________________________________________________________________________________

    /// Evaluates multi-variable function over vectorized input data.
    ///
    /// Handles functions of arbitrary dimension with proper matrix transposition.
    /// Input format: [[x1_vals], [x2_vals], ...] -> Output: [f(x1[i], x2[i], ...)]
    ///
    /// # Arguments
    /// * `x` - Vector of variable value vectors, one per variable
    ///
    /// # Returns
    /// Vector of function evaluations
    ///
    /// # Data Layout
    /// Input: [[x1, x2, x3], [y1, y2, y3]] for f(x,y)
    /// Internally transposed to: [[x1, y1], [x2, y2], [x3, y3]]
    ///
    /// # Panics
    /// Panics if variable vectors have different lengths
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

    /// Evaluates multi-variable function over linearly spaced domains.
    ///
    /// Creates linspace for each variable and evaluates function over the
    /// Cartesian product of the domains.
    ///
    /// # Arguments
    /// * `start` - Start values for each variable
    /// * `end` - End values for each variable
    /// * `num_values` - Number of points per variable
    ///
    /// # Returns
    /// Vector of function evaluations over the multi-dimensional grid
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
    /// Computes partial derivatives with respect to specified variables.
    ///
    /// More controlled than diff_multi() as it allows specifying exactly which
    /// variables to differentiate with respect to and in what order.
    ///
    /// # Arguments
    /// * `all_vars` - Variables to differentiate with respect to
    ///
    /// # Returns
    /// Vector of symbolic partial derivative expressions
    pub fn diff_multi_args(&self, all_vars: &Vec<&str>) -> Vec<Expr> {
        let vec_of_exprs = all_vars.iter().map(|var| self.diff(var)).collect();
        vec_of_exprs
    }

    /// Computes all partial derivatives automatically.
    ///
    /// Finds all variables in the expression and computes partial derivatives
    /// with respect to each. Variables are processed in alphabetical order.
    ///
    /// # Returns
    /// Vector of symbolic partial derivative expressions
    ///
    /// # Usage
    /// Essential for gradient computation and Jacobian matrix construction
    pub fn diff_multi(&self) -> Vec<Expr> {
        let all_vars = self.all_arguments_are_variables();
        let vec_of_exprs = all_vars.iter().map(|var| self.diff(var)).collect();
        vec_of_exprs
    }

    /// Evaluates all partial derivatives over vectorized input data.
    ///
    /// Computes the gradient vector at each point in the input dataset.
    /// Returns matrix where each row corresponds to a partial derivative.
    ///
    /// # Arguments
    /// * `x` - Input data in format [[var1_vals], [var2_vals], ...]
    ///
    /// # Returns
    /// Matrix of partial derivative values [n_vars × n_points]
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

    /// Evaluates all partial derivatives over linearly spaced domains.
    ///
    /// Convenience method for gradient evaluation over regular grids.
    /// Useful for visualization and numerical analysis of multi-variable functions.
    ///
    /// # Arguments
    /// * `start` - Start values for each variable
    /// * `end` - End values for each variable  
    /// * `num_values` - Number of evaluation points per variable
    ///
    /// # Returns
    /// Matrix of partial derivative values over the specified domain
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

    /// Validates all partial derivatives against numerical approximations.
    ///
    /// Comprehensive validation for multi-variable functions. Computes analytical
    /// and numerical partial derivatives, then compares their L2 norms.
    ///
    /// # Arguments
    /// * `start` - Domain start values for each variable
    /// * `end` - Domain end values for each variable
    /// * `num_values` - Number of test points per variable
    /// * `max_norm` - Maximum acceptable norm difference per partial derivative
    ///
    /// # Returns
    /// Vector of (is_valid, actual_norm) for each partial derivative
    ///
    /// # Algorithm
    /// - Generates multi-dimensional test grid
    /// - Computes analytical gradients symbolically
    /// - Approximates gradients numerically using finite differences
    /// - Compares norms for each partial derivative separately
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
    //                    INITIAL VALUE PROBLEM (IVP) SPECIALIZATION
    //____________________________________________________________________________________________________________________________

    /// Creates closure specialized for Initial Value Problems with time-dependent functions.
    ///
    /// In IVPs, functions typically have the form f(t, y1, y2, ...) where t is the
    /// independent variable (time) and y1, y2, ... are state variables. This method
    /// creates a closure that separates the time argument from state variables.
    ///
    /// # Arguments
    /// * `arg` - Independent variable name (typically time "t" or "x")
    /// * `vars` - State variable names in order
    ///
    /// # Returns
    /// Closure taking (time_value, state_vector) and returning f64
    ///
    /// # Usage
    /// Essential for ODE solvers where dy/dt = f(t, y)
    ///
    /// # Examples
    /// ```rust, ignore
    /// let expr = parse_expression("t + y1 + y2"); // f(t, y1, y2)
    /// let func = expr.lambdify_IVP("t", vec!["y1", "y2"]);
    /// let result = func(1.0, vec![2.0, 3.0]); // f(1.0, 2.0, 3.0) = 6.0
    /// ```
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

    /// Owned version of lambdify_IVP that consumes the expression.
    ///
    /// Similar to lambdify_IVP but takes ownership of the expression, allowing
    /// the returned closure to have 'static lifetime. Useful for ODE solvers
    /// that need to store the function long-term.
    ///
    /// # Arguments
    /// * `arg` - Independent variable name (typically time)
    /// * `vars` - State variable names in order
    ///
    /// # Returns
    /// Closure with 'static lifetime taking (time, state_vector) -> f64
    ///
    /// # Ownership
    /// Consumes self, preventing further use of the original expression
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
    }
}
