//! # Symbolic Engine Module
//!
//! This module provides a comprehensive symbolic mathematics engine for creating, manipulating,
//! and evaluating symbolic expressions. It serves as the core foundation for symbolic computation
//! in the RustedSciThe framework, enabling analytical differentiation, expression simplification,
//! and conversion to executable functions.
//!
//! ## Purpose
//!
//! The symbolic engine allows users to:
//! - Parse and create symbolic mathematical expressions
//! - Perform analytical differentiation and integration
//! - Simplify complex expressions algebraically
//! - Convert symbolic expressions to executable Rust functions (lambdification)
//! - Handle indexed variables for matrix/vector operations
//! - Support trigonometric, exponential, and logarithmic functions
//!
//! ## Main Structures and Methods
//!
//! ### `Expr` Enum
//! The core symbolic expression type supporting:
//! - **Variables**: `Var(String)` - symbolic variables like "x", "y"
//! - **Constants**: `Const(f64)` - numerical constants
//! - **Operations**: `Add`, `Sub`, `Mul`, `Div`, `Pow` - basic arithmetic
//! - **Functions**: `Exp`, `Ln`, `sin`, `cos`, etc. - mathematical functions
//!
//! ### Key Methods
//! - `Symbols(symbols: &str)` - Create multiple variables from comma-separated string
//! - `IndexedVar(index: usize, var_name: &str)` - Create indexed variables (x0, x1, etc.)
//! - `diff(var: &str)` - Analytical differentiation
//! - `lambdify()` - Convert to executable function
//! - `simplify_()` - Algebraic simplification
//! - `set_variable()` - Substitute variables with values
//!
//! ## Interesting Code Features
//!
//! 1. **Recursive Expression Tree**: Uses Box<Expr> for nested expressions, enabling
//!    arbitrarily complex mathematical structures
//!
//! 2. **Operator Overloading**: Implements std::ops traits (Add, Sub, Mul, Div) for
//!    natural mathematical syntax: `x + y * z`
//!
//! 3. **Pattern Matching Differentiation**: Uses exhaustive match statements to implement
//!    analytical differentiation rules (product rule, chain rule, etc.)
//!
//! 4. **Smart Simplification**: The `simplify_()` method applies algebraic rules like
//!    `x + 0 = x`, `x * 1 = x`, `0 * x = 0` recursively
//!
//! 5. **Indexed Variable System**: Supports both 1D (x0, x1) and 2D (A_2_3) indexing
//!    for matrix operations and large systems
//!
//! 6. **Macro System**: Provides convenient macros like `symbols!(x, y, z)` for
//!    ergonomic variable creation
//!
//! 7. **Non-standard Function Names**: Uses mathematical notation (tg, ctg) instead
//!    of programming conventions (tan, cot) for trigonometric functions

#![allow(non_camel_case_types)]

use std::collections::HashMap;

use std::f64;
use std::f64::consts::PI;
use std::fmt;

/// Core symbolic expression enum representing mathematical expressions as an abstract syntax tree.
///
/// Each variant represents a different type of mathematical construct, from simple variables
/// and constants to complex nested operations. The enum uses Box<Expr> for recursive structures,
/// allowing arbitrarily deep expression trees.
///
/// # Examples
/// ```rust, ignore
/// use symbolic_engine::Expr;
/// let x = Expr::Var("x".to_string());
/// let expr = Expr::Add(Box::new(x), Box::new(Expr::Const(2.0)));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    /// Symbolic variable with a name (e.g., "x", "y", "velocity")
    Var(String),
    /// Numerical constant value
    Const(f64),
    /// Addition operation: left + right
    Add(Box<Expr>, Box<Expr>),
    /// Subtraction operation: left - right
    Sub(Box<Expr>, Box<Expr>),
    /// Multiplication operation: left * right
    Mul(Box<Expr>, Box<Expr>),
    /// Division operation: left / right
    Div(Box<Expr>, Box<Expr>),
    /// Power operation: base ^ exponent
    Pow(Box<Expr>, Box<Expr>),
    /// Exponential function: e^x
    Exp(Box<Expr>),
    /// Natural logarithm: ln(x)
    Ln(Box<Expr>),
    /// Sine function: sin(x)
    sin(Box<Expr>),
    /// Cosine function: cos(x)
    cos(Box<Expr>),
    /// Tangent function: tan(x) - uses mathematical notation 'tg'
    tg(Box<Expr>),
    /// Cotangent function: cot(x) - uses mathematical notation 'ctg'
    ctg(Box<Expr>),
    /// Arcsine function: arcsin(x)
    arcsin(Box<Expr>),
    /// Arccosine function: arccos(x)
    arccos(Box<Expr>),
    /// Arctangent function: arctan(x) - uses mathematical notation 'arctg'
    arctg(Box<Expr>),
    /// Arccotangent function: arccot(x) - uses mathematical notation 'arcctg'
    arcctg(Box<Expr>),
}

/// Display implementation for pretty printing symbolic expressions.
///
/// Converts expressions to human-readable mathematical notation with parentheses
/// for proper precedence. Uses standard mathematical symbols and function names.
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
            Expr::sin(expr) => write!(f, "sin({})", expr),
            Expr::cos(expr) => write!(f, "cos({})", expr),
            Expr::tg(expr) => write!(f, "tg({})", expr),
            Expr::ctg(expr) => write!(f, "ctg({})", expr),
            Expr::arcsin(expr) => write!(f, "arcsin({})", expr),
            Expr::arccos(expr) => write!(f, "arccos({})", expr),
            Expr::arctg(expr) => write!(f, "arctg({})", expr),
            Expr::arcctg(expr) => write!(f, "arcctg({})", expr),
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
impl std::ops::AddAssign for Expr {
    fn add_assign(&mut self, rhs: Self) {
        *self = Expr::Add(Box::new(self.clone()), Box::new(rhs));
    }
}

impl std::ops::SubAssign for Expr {
    fn sub_assign(&mut self, rhs: Self) {
        *self = Expr::Sub(Box::new(self.clone()), Box::new(rhs));
    }
}

impl std::ops::MulAssign for Expr {
    fn mul_assign(&mut self, rhs: Self) {
        *self = Expr::Mul(Box::new(self.clone()), Box::new(rhs));
    }
}

impl std::ops::DivAssign for Expr {
    fn div_assign(&mut self, rhs: Self) {
        *self = Expr::Div(Box::new(self.clone()), Box::new(rhs));
    }
}

impl std::ops::Neg for Expr {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Expr::Mul(Box::new(Expr::Const(-1.0)), Box::new(self))
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

    /// Creates multiple symbolic variables from a comma-separated string.
    ///
    /// Parses a string containing variable names separated by commas and returns
    /// a vector of Expr::Var instances. Whitespace is automatically trimmed.
    ///
    /// # Arguments
    /// * `symbols` - Comma-separated string of variable names (e.g., "x, y, z")
    ///
    /// # Returns
    /// Vector of Expr::Var instances for each variable name
    ///
    /// # Examples
    /// ```rust, ignore
    /// let vars = Expr::Symbols("x, y, z");
    /// assert_eq!(vars.len(), 3);
    /// ```
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
    /// Substitutes a variable with a constant value throughout the expression.
    ///
    /// Recursively traverses the expression tree and replaces all occurrences
    /// of the specified variable with the given constant value.
    ///
    /// # Arguments
    /// * `var` - Name of the variable to substitute
    /// * `value` - Numerical value to substitute for the variable
    ///
    /// # Returns
    /// New expression with the variable substituted
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
            Expr::sin(expr) => Expr::sin(Box::new(expr.set_variable(var, value))),
            Expr::cos(expr) => Expr::cos(Box::new(expr.set_variable(var, value))),
            Expr::tg(expr) => Expr::tg(Box::new(expr.set_variable(var, value))),
            Expr::ctg(expr) => Expr::ctg(Box::new(expr.set_variable(var, value))),
            Expr::arcsin(expr) => Expr::arcsin(Box::new(expr.set_variable(var, value))),
            Expr::arccos(expr) => Expr::arccos(Box::new(expr.set_variable(var, value))),
            Expr::arctg(expr) => Expr::arctg(Box::new(expr.set_variable(var, value))),
            Expr::arcctg(expr) => Expr::arcctg(Box::new(expr.set_variable(var, value))),
            _ => self.clone(),
        }
    }

    /// Substitutes multiple variables with constant values using a HashMap.
    ///
    /// More efficient than multiple set_variable calls when substituting many variables.
    /// Only variables present in the map are substituted.
    ///
    /// # Arguments
    /// * `var_map` - HashMap mapping variable names to their replacement values
    ///
    /// # Returns
    /// New expression with all mapped variables substituted
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

            Expr::sin(expr) => Expr::sin(Box::new(expr.set_variable_from_map(var_map))),
            Expr::cos(expr) => Expr::cos(Box::new(expr.set_variable_from_map(var_map))),
            Expr::tg(expr) => Expr::tg(Box::new(expr.set_variable_from_map(var_map))),
            Expr::ctg(expr) => Expr::ctg(Box::new(expr.set_variable_from_map(var_map))),
            Expr::arcsin(expr) => Expr::arcsin(Box::new(expr.set_variable_from_map(var_map))),
            Expr::arccos(expr) => Expr::arccos(Box::new(expr.set_variable_from_map(var_map))),
            Expr::arctg(expr) => Expr::arctg(Box::new(expr.set_variable_from_map(var_map))),
            Expr::arcctg(expr) => Expr::arcctg(Box::new(expr.set_variable_from_map(var_map))),
            _ => self.clone(),
        }
    }
    /// Renames a variable throughout the expression.
    ///
    /// Recursively replaces all occurrences of old_var with new_var.
    /// Useful for variable substitution and expression manipulation.
    ///
    /// # Arguments
    /// * `old_var` - Current variable name to replace
    /// * `new_var` - New variable name
    ///
    /// # Returns
    /// New expression with the variable renamed
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

            Expr::sin(expr) => Expr::sin(Box::new(expr.rename_variable(old_var, new_var))),
            Expr::cos(expr) => Expr::cos(Box::new(expr.rename_variable(old_var, new_var))),
            Expr::tg(expr) => Expr::tg(Box::new(expr.rename_variable(old_var, new_var))),
            Expr::ctg(expr) => Expr::ctg(Box::new(expr.rename_variable(old_var, new_var))),
            Expr::arcsin(expr) => Expr::arcsin(Box::new(expr.rename_variable(old_var, new_var))),
            Expr::arccos(expr) => Expr::arccos(Box::new(expr.rename_variable(old_var, new_var))),
            Expr::arctg(expr) => Expr::arctg(Box::new(expr.rename_variable(old_var, new_var))),
            Expr::arcctg(expr) => Expr::arcctg(Box::new(expr.rename_variable(old_var, new_var))),

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

            Expr::sin(expr) => Expr::sin(Box::new(expr.rename_variables(var_map))),
            Expr::cos(expr) => Expr::cos(Box::new(expr.rename_variables(var_map))),
            Expr::tg(expr) => Expr::tg(Box::new(expr.rename_variables(var_map))),
            Expr::ctg(expr) => Expr::ctg(Box::new(expr.rename_variables(var_map))),
            Expr::arcsin(expr) => Expr::arcsin(Box::new(expr.rename_variables(var_map))),
            Expr::arccos(expr) => Expr::arccos(Box::new(expr.rename_variables(var_map))),
            Expr::arctg(expr) => Expr::arctg(Box::new(expr.rename_variables(var_map))),

            _ => self.clone(),
        }
    }
    /// substitute a variable with an expression
    pub fn substitute_variable(&self, var: &str, expr: &Expr) -> Expr {
        match self {
            Expr::Var(name) if name == var => expr.clone(),
            Expr::Add(lhs, rhs) => Expr::Add(
                Box::new(lhs.substitute_variable(var, expr)),
                Box::new(rhs.substitute_variable(var, expr)),
            ),
            Expr::Sub(lhs, rhs) => Expr::Sub(
                Box::new(lhs.substitute_variable(var, expr)),
                Box::new(rhs.substitute_variable(var, expr)),
            ),
            Expr::Mul(lhs, rhs) => Expr::Mul(
                Box::new(lhs.substitute_variable(var, expr)),
                Box::new(rhs.substitute_variable(var, expr)),
            ),
            Expr::Div(lhs, rhs) => Expr::Div(
                Box::new(lhs.substitute_variable(var, expr)),
                Box::new(rhs.substitute_variable(var, expr)),
            ),
            Expr::Pow(base, exp) => Expr::Pow(
                Box::new(base.substitute_variable(var, expr)),
                Box::new(exp.substitute_variable(var, expr)),
            ),
            Expr::Exp(expr_old) => Expr::Exp(Box::new(expr_old.substitute_variable(var, expr))),
            Expr::Ln(expr_old) => Expr::Ln(Box::new(expr_old.substitute_variable(var, expr))),

            Expr::sin(expr) => Expr::sin(Box::new(expr.substitute_variable(var, expr))),
            Expr::cos(expr) => Expr::cos(Box::new(expr.substitute_variable(var, expr))),
            Expr::tg(expr) => Expr::tg(Box::new(expr.substitute_variable(var, expr))),
            Expr::ctg(expr) => Expr::ctg(Box::new(expr.substitute_variable(var, expr))),
            Expr::arcsin(expr) => Expr::arcsin(Box::new(expr.substitute_variable(var, expr))),
            Expr::arccos(expr) => Expr::arccos(Box::new(expr.substitute_variable(var, expr))),
            Expr::arctg(expr) => Expr::arctg(Box::new(expr.substitute_variable(var, expr))),
            Expr::arcctg(expr) => Expr::arcctg(Box::new(expr.substitute_variable(var, expr))),
            _ => self.clone(),
        }
    }
    /// check if the expression contains a variable
    pub fn contains_variable(&self, var_name: &str) -> bool {
        match self {
            Expr::Var(name) => name == var_name,
            Expr::Const(_) => false,
            Expr::Add(left, right)
            | Expr::Sub(left, right)
            | Expr::Mul(left, right)
            | Expr::Div(left, right) => {
                left.contains_variable(var_name) || right.contains_variable(var_name)
            }
            Expr::Pow(base, exp) => {
                base.contains_variable(var_name) || exp.contains_variable(var_name)
            }
            Expr::Exp(expr) => expr.contains_variable(var_name),
            Expr::Ln(expr) => expr.contains_variable(var_name),

            Expr::sin(expr) => expr.contains_variable(var_name),
            Expr::cos(expr) => expr.contains_variable(var_name),
            Expr::tg(expr) => expr.contains_variable(var_name),
            Expr::ctg(expr) => expr.contains_variable(var_name),
            Expr::arcsin(expr) => expr.contains_variable(var_name),
            Expr::arccos(expr) => expr.contains_variable(var_name),
            Expr::arctg(expr) => expr.contains_variable(var_name),
            Expr::arcctg(expr) => expr.contains_variable(var_name),
            _ => false,
        }
    }

    /// Convenience method to wrap expression in Box for recursive structures.
    ///
    /// Essential for creating nested expressions since Expr variants use Box<Expr>.
    pub fn boxed(self) -> Box<Self> {
        Box::new(self)
    }

    /// Mutates current expression to a variable and returns the new variable expression.
    ///
    /// # Arguments
    /// * `var` - Variable name to create
    pub fn var_expr(&mut self, var: &str) -> Expr {
        let expr = Expr::Var(var.to_string());
        *self = expr.clone();
        expr
    }

    /// Mutates current expression to a constant value.
    ///
    /// # Arguments
    /// * `val` - Constant value to set
    pub fn const_expr(&mut self, val: f64) {
        *self = Expr::Const(val);
    }

    /// Creates exponential function e^(self).
    ///
    /// # Returns
    /// New Expr::Exp containing this expression
    pub fn exp(mut self) -> Expr {
        self = Expr::Exp(self.boxed());
        self
    }

    /// Creates natural logarithm ln(self).
    ///
    /// # Returns
    /// New Expr::Ln containing this expression
    pub fn ln(mut self) -> Expr {
        self = Expr::Ln(self.boxed());
        self
    }

    /// Creates base-10 logarithm using ln(x)/ln(10) identity.
    ///
    /// # Returns
    /// Expression equivalent to log10(self)
    pub fn log10(mut self) -> Expr {
        self = Expr::Ln(self.boxed()) / Expr::Const(2.30258509);
        self
    }

    /// Creates power expression self^rhs.
    ///
    /// # Arguments
    /// * `rhs` - Exponent expression
    ///
    /// # Returns
    /// New Expr::Pow with self as base and rhs as exponent
    pub fn pow(mut self, rhs: Expr) -> Expr {
        self = Expr::Pow(self.boxed(), rhs.boxed());
        self
    }

    /// Checks if expression is exactly zero (constant 0.0).
    ///
    /// # Returns
    /// true if expression is Const(0.0), false otherwise
    pub fn is_zero(&self) -> bool {
        match self {
            Expr::Const(val) => val == &0.0,
            _ => false,
        }
    }

    //__________________________________INDEXED VARIABLES____________________________________

    /// Creates a single indexed variable with format "name + index" (e.g., "x5").
    ///
    /// Useful for creating sequences of related variables in mathematical systems.
    ///
    /// # Arguments
    /// * `index` - Numerical index to append
    /// * `var_name` - Base variable name
    ///
    /// # Returns
    /// Expr::Var with indexed name
    pub fn IndexedVar(index: usize, var_name: &str) -> Expr {
        let indexed_var_name = format!("{}{}", var_name, index);
        Expr::Var(indexed_var_name)
    }

    /// Creates multiple indexed variables and their string representations.
    ///
    /// Generates a sequence of variables like x0, x1, x2, ... up to num_vars-1.
    /// Returns both the Expr objects and their string names for convenience.
    ///
    /// # Arguments
    /// * `num_vars` - Number of variables to create
    /// * `var_name` - Base name for variables
    ///
    /// # Returns
    /// Tuple of (Vec<Expr>, Vec<String>) containing expressions and names
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
    /// Creates a 2D indexed variable with format "name_row_col" (e.g., "A_2_3").
    ///
    /// Essential for matrix operations and 2D grid systems in numerical methods.
    ///
    /// # Arguments
    /// * `index_row` - Row index
    /// * `index_col` - Column index  
    /// * `var_name` - Base variable name
    ///
    /// # Returns
    /// Expr::Var with 2D indexed name
    pub fn IndexedVar2D(index_row: usize, index_col: usize, var_name: &str) -> Expr {
        let indexed_var_name = format!("{}_{}_{}", var_name, index_row, index_col);
        Expr::Var(indexed_var_name)
    }

    /// Creates a 2D matrix of indexed variables.
    ///
    /// Generates variables in matrix form like A_0_0, A_0_1, A_1_0, A_1_1, etc.
    /// Returns both the 2D structure and flattened name list.
    ///
    /// # Arguments
    /// * `num_rows` - Number of rows in matrix
    /// * `num_cols` - Number of columns in matrix
    /// * `var_name` - Base name for matrix elements
    ///
    /// # Returns
    /// Tuple of (Vec<Vec<Expr>>, Vec<String>) - matrix structure and flat name list
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

    /// Creates a flattened vector of 2D indexed variables.
    ///
    /// More memory-efficient than IndexedVars2D when matrix structure isn't needed.
    ///
    /// # Arguments
    /// * `num_rows` - Number of rows
    /// * `num_cols` - Number of columns
    /// * `var_name` - Base variable name
    ///
    /// # Returns
    /// Flat Vec<Expr> of all matrix elements
    pub fn IndexedVars2Dflat(num_rows: usize, num_cols: usize, var_name: &str) -> Vec<Expr> {
        (0..num_rows)
            .flat_map(|i| (0..num_cols).map(move |j| Expr::IndexedVar2D(i, j, var_name)))
            .collect()
    }

    /// Creates a symbolic polynomial of specified degree.
    ///
    /// Generates polynomial c0 + c1*x + c2*x^2 + ... + cn*x^n with symbolic coefficients.
    /// Useful for curve fitting and polynomial approximation.
    ///
    /// # Arguments
    /// * `degree` - Highest power in polynomial
    /// * `var_name` - Variable name for polynomial argument
    ///
    /// # Returns
    /// Tuple of (polynomial expression, coefficient names)
    pub fn polyval(degree: usize, var_name: &str) -> (Expr, Vec<String>) {
        let mut eq: Expr = Self::Const(0.0);
        let mut unknowns = Vec::new();
        let arg_expr = Self::Var(var_name.to_string());
        for i in 0..degree + 1 {
            let coeff = Self::Var(format!("c{}", i));
            unknowns.push(format!("c{}", i));
            eq = eq + coeff * (arg_expr.clone().pow(Self::Const(i as f64))).simplify_();
        }
        (eq, unknowns)
    }

    //___________________________________SIMPLIFICATION____________________________________

    /// Internal method to eliminate zero-multiplication subexpressions.
    ///
    /// Recursively identifies patterns like Mul(Const(0.0), anything) and replaces
    /// the entire subexpression with Const(0.0). This is a specialized optimization
    /// for expressions with many zero terms.
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

            Expr::sin(expr) => Expr::sin(Box::new(expr.nozeros())),
            Expr::cos(expr) => Expr::cos(Box::new(expr.nozeros())),
            Expr::tg(expr) => Expr::tg(Box::new(expr.nozeros())),
            Expr::ctg(expr) => Expr::ctg(Box::new(expr.nozeros())),
            Expr::arcsin(expr) => Expr::arcsin(Box::new(expr.nozeros())),
            Expr::arccos(expr) => Expr::arccos(Box::new(expr.nozeros())),
            Expr::arctg(expr) => Expr::arctg(Box::new(expr.nozeros())),
            Expr::arcctg(expr) => Expr::arcctg(Box::new(expr.nozeros())),
        }
    } // nozeros

    /// Simplifies expressions by evaluating constant arithmetic operations.
    ///
    /// Combines numerical constants where possible (e.g., Const(2) + Const(3) = Const(5))
    /// but leaves variable operations unchanged. Less comprehensive than simplify_().
    ///
    /// # Returns
    /// Simplified expression with constant operations evaluated
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
            Expr::sin(expr) => Expr::sin(Box::new(expr.simplify_numbers())),
            Expr::cos(expr) => Expr::cos(Box::new(expr.simplify_numbers())),
            Expr::tg(expr) => Expr::tg(Box::new(expr.simplify_numbers())),
            Expr::ctg(expr) => Expr::ctg(Box::new(expr.simplify_numbers())),
            Expr::arcsin(expr) => Expr::arcsin(Box::new(expr.simplify_numbers())),
            Expr::arccos(expr) => Expr::arccos(Box::new(expr.simplify_numbers())),
            Expr::arctg(expr) => Expr::arctg(Box::new(expr.simplify_numbers())),
            Expr::arcctg(expr) => Expr::arcctg(Box::new(expr.simplify_numbers())),
        }
    }
    /// Comprehensive algebraic simplification using mathematical identities.
    ///
    /// Applies rules like:
    /// - x + 0 = x, x - 0 = x
    /// - x * 1 = x, x * 0 = 0
    /// - x^0 = 1, x^1 = x
    /// - exp(0) = 1, ln(1) = 0
    /// - Constant arithmetic evaluation
    /// - Constant folding in mixed expressions
    ///
    /// More powerful than simplify_numbers() as it handles algebraic identities.
    ///
    /// # Returns
    /// Maximally simplified expression
    pub fn simplify_(&self) -> Expr {
        match self {
            Expr::Var(_) => self.clone(),
            Expr::Const(_) => self.clone(),
            Expr::Add(lhs, rhs) => {
                let lhs = lhs.simplify_();
                let rhs = rhs.simplify_();
                match (&lhs, &rhs) {
                    (Expr::Const(a), Expr::Const(b)) => Expr::Const(a + b), // (a) + (b) = (a + b)
                    (Expr::Const(0.0), _) => rhs,                           // 0 + x = x
                    (_, Expr::Const(0.0)) => lhs,                           // x + 0 = x
                    _ => Expr::Add(Box::new(lhs), Box::new(rhs)),
                }
            }
            Expr::Sub(lhs, rhs) => {
                let lhs = lhs.simplify_();
                let rhs = rhs.simplify_();
                match (&lhs, &rhs) {
                    (Expr::Const(a), Expr::Const(b)) => Expr::Const(a - b), // (a) - (b) = (a - b)
                    (_, Expr::Const(0.0)) => lhs,                           // x - 0 = x
                    _ => Expr::Sub(Box::new(lhs), Box::new(rhs)),
                }
            }
            Expr::Mul(lhs, rhs) => {
                let lhs = lhs.simplify_();
                let rhs = rhs.simplify_();
                match (&lhs, &rhs) {
                    (Expr::Const(a), Expr::Const(b)) => Expr::Const(a * b), // (a) * (b) = (a * b)
                    (Expr::Const(0.0), _) | (_, Expr::Const(0.0)) => Expr::Const(0.0), // 0 * x = 0
                    (Expr::Const(1.0), _) => rhs,                           // 1 * x = x
                    (_, Expr::Const(1.0)) => lhs,                           // x * 1 = x
                    // Handle nested multiplications with constants: (c1 * expr) * c2 = (c1 * c2) * expr
                    (Expr::Mul(inner_lhs, inner_rhs), Expr::Const(c)) => {
                        match (inner_lhs.as_ref(), inner_rhs.as_ref()) {
                            (Expr::Const(c1), _) => {
                                Expr::Mul(Box::new(Expr::Const(c1 * c)), inner_rhs.clone())
                                    .simplify_()
                            }
                            (_, Expr::Const(c1)) => {
                                Expr::Mul(Box::new(Expr::Const(c1 * c)), inner_lhs.clone())
                                    .simplify_()
                            }
                            _ => Expr::Mul(Box::new(lhs), Box::new(rhs)),
                        }
                    }
                    (Expr::Const(c), Expr::Mul(inner_lhs, inner_rhs)) => {
                        match (inner_lhs.as_ref(), inner_rhs.as_ref()) {
                            (Expr::Const(c1), _) => {
                                Expr::Mul(Box::new(Expr::Const(c * c1)), inner_rhs.clone())
                                    .simplify_()
                            }
                            (_, Expr::Const(c1)) => {
                                Expr::Mul(Box::new(Expr::Const(c * c1)), inner_lhs.clone())
                                    .simplify_()
                            }
                            _ => Expr::Mul(Box::new(lhs), Box::new(rhs)),
                        }
                    }
                    _ => Expr::Mul(Box::new(lhs), Box::new(rhs)),
                }
            }
            Expr::Div(lhs, rhs) => {
                let lhs = lhs.simplify_();
                let rhs = rhs.simplify_();
                match (&lhs, &rhs) {
                    (Expr::Const(a), Expr::Const(b)) if *b != 0.0 => Expr::Const(a / b), // (a) / (b) = (a / b)
                    (Expr::Const(0.0), _) => Expr::Const(0.0), // 0 / x = 0
                    (_, Expr::Const(1.0)) => lhs,              // x / 1 = x
                    // Handle division of multiplication by constant: (c1 * expr) / c2 = (c1/c2) * expr
                    (Expr::Mul(inner_lhs, inner_rhs), Expr::Const(c)) if *c != 0.0 => {
                        match (inner_lhs.as_ref(), inner_rhs.as_ref()) {
                            (Expr::Const(c1), _) => {
                                Expr::Mul(Box::new(Expr::Const(c1 / c)), inner_rhs.clone())
                                    .simplify_()
                            }
                            (_, Expr::Const(c1)) => {
                                Expr::Mul(Box::new(Expr::Const(c1 / c)), inner_lhs.clone())
                                    .simplify_()
                            }
                            _ => Expr::Div(Box::new(lhs), Box::new(rhs)),
                        }
                    }
                    // Handle division by multiplication: expr / (c1 * c2) = expr / (c1*c2)
                    (_, Expr::Mul(inner_lhs, inner_rhs)) => {
                        match (inner_lhs.as_ref(), inner_rhs.as_ref()) {
                            (Expr::Const(c1), Expr::Const(c2)) => {
                                Expr::Div(Box::new(lhs), Box::new(Expr::Const(c1 * c2))).simplify_()
                            }
                            _ => Expr::Div(Box::new(lhs), Box::new(rhs)),
                        }
                    }
                    _ => Expr::Div(Box::new(lhs), Box::new(rhs)),
                }
            }
            Expr::Pow(base, exp) => {
                let base = base.simplify_();
                let exp = exp.simplify_();
                match (&base, &exp) {
                    (Expr::Const(a), Expr::Const(b)) => Expr::Const(a.powf(*b)), // (a) ^ (b) = (a ^ b)
                    (_, Expr::Const(0.0)) => Expr::Const(1.0),                   // x ^ 0 = 1
                    (_, Expr::Const(1.0)) => base,                               // x ^ 1 = x
                    (Expr::Const(0.0), _) => Expr::Const(0.0),                   // 0 ^ x = 0
                    (Expr::Const(1.0), _) => Expr::Const(1.0),                   // 1 ^ x = 1
                    _ => Expr::Pow(Box::new(base), Box::new(exp)),
                }
            }
            Expr::Exp(expr) => {
                let expr = expr.simplify_();
                match &expr {
                    Expr::Const(a) if a != &0.0 => Expr::Const(a.exp()),
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
            } // ln

            Expr::sin(expr) => {
                let expr = expr.simplify_();
                match &expr {
                    Expr::Const(a) => Expr::Const(a.sin()),
                    _ => Expr::sin(Box::new(expr)),
                }
            } //sin

            Expr::cos(expr) => {
                let expr = expr.simplify_();
                match &expr {
                    Expr::Const(a) => Expr::Const(a.cos()),
                    _ => Expr::cos(Box::new(expr)),
                }
            } //cos
            Expr::tg(expr) => {
                let expr = expr.simplify_();
                match &expr {
                    Expr::Const(a) => Expr::Const(a.tan()),
                    _ => Expr::tg(Box::new(expr)),
                }
            } //tg
            Expr::ctg(expr) => {
                let expr = expr.simplify_();
                match &expr {
                    Expr::Const(a) => Expr::Const(1.0 / a.tan()),
                    _ => Expr::ctg(Box::new(expr)),
                }
            } //ctg

            Expr::arcsin(expr) => {
                let expr = expr.simplify_();
                match &expr {
                    Expr::Const(a) => Expr::Const(a.asin()),
                    _ => Expr::arcsin(Box::new(expr)),
                }
            } //arcsin
            Expr::arccos(expr) => {
                let expr = expr.simplify_();
                match &expr {
                    Expr::Const(a) => Expr::Const(a.acos()),
                    _ => Expr::arcsin(Box::new(expr)),
                }
            } //arccos
            Expr::arctg(expr) => {
                let expr = expr.simplify_();
                match &expr {
                    Expr::Const(a) => Expr::Const(a.atan()),
                    _ => Expr::arcsin(Box::new(expr)),
                }
            } //arctg

            Expr::arcctg(expr) => {
                let expr = expr.simplify_();
                match &expr {
                    Expr::Const(a) => Expr::Const(PI / 2.0 - (a).atan()),
                    _ => Expr::arcsin(Box::new(expr)),
                }
            } //arctg
        }
    }
    /// Public interface for expression simplification.
    ///
    /// Currently delegates to simplify_() but provides a stable API for future
    /// enhancements. This is the recommended method for users to simplify expressions.
    ///
    /// # Returns
    /// Simplified expression using all available simplification rules
    pub fn symplify(&self) -> Expr {
        //let zeros_proceeded = self.nozeros().simplify_numbers();
        let zeros_proceeded = self.simplify_();
        zeros_proceeded
    }
}

//___________________________________MACROS____________________________________

/// Macro to create symbolic variables from a comma-separated list
/// Usage: symbols!(x, y, z) -> creates variables x, y, z
#[macro_export]
macro_rules! symbols {
    ($($var:ident),+ $(,)?) => {
        {
            let var_names = stringify!($($var),+);
            let vars = Expr::Symbols(var_names);
            let mut iter = vars.into_iter();
            ($(
                {
                    let $var = iter.next().unwrap();
                    $var
                }
            ),+)
        }
    };
}

/// Macro to create indexed variables
/// Usage: indexed_vars!(5, "x") -> creates x0, x1, x2, x3, x4
#[macro_export]
macro_rules! indexed_vars {
    ($count:expr, $name:expr) => {
        Expr::IndexedVars($count, $name)
    };
}

/// Macro to create 2D indexed variables
/// Usage: indexed_vars_2d!(3, 3, "A") -> creates A_0_0, A_0_1, etc.
#[macro_export]
macro_rules! indexed_vars_2d {
    ($rows:expr, $cols:expr, $name:expr) => {
        Expr::IndexedVars2D($rows, $cols, $name)
    };
}

/// Macro to create a single indexed variable
/// Usage: indexed_var!(5, "x") -> creates x5
#[macro_export]
macro_rules! indexed_var {
    ($index:expr, $name:expr) => {
        Expr::IndexedVar($index, $name)
    };
}

/// Macro to create a 2D indexed variable
/// Usage: indexed_var_2d!(2, 3, "A") -> creates A_2_3
#[macro_export]
macro_rules! indexed_var_2d {
    ($row:expr, $col:expr, $name:expr) => {
        Expr::IndexedVar2D($row, $col, $name)
    };
}

//___________________________________TESTS____________________________________

#[cfg(test)]
use approx;
mod tests {
    use super::*;
    #[test]
    fn test_add_assign() {
        let mut expr = Expr::Var("x".to_string());
        expr += Expr::Const(2.0);
        let expected = Expr::Add(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Const(2.0)),
        );
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_sub_assign() {
        let mut expr = Expr::Var("x".to_string());
        expr -= Expr::Const(2.0);
        let expected = Expr::Sub(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Const(2.0)),
        );
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_mul_assign() {
        let mut expr = Expr::Var("x".to_string());
        expr *= Expr::Const(2.0);
        let expected = Expr::Mul(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Const(2.0)),
        );
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_div_assign() {
        let mut expr = Expr::Var("x".to_string());
        expr /= Expr::Const(2.0);
        let expected = Expr::Div(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Const(2.0)),
        );
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_neg() {
        let expr = Expr::Var("x".to_string());
        let neg_expr = -expr;
        let expected = Expr::Mul(
            Box::new(Expr::Const(-1.0)),
            Box::new(Expr::Var("x".to_string())),
        );
        assert_eq!(neg_expr, expected);
    }
    #[test]
    fn test_combined_operations() {
        let mut expr = Expr::Var("x".to_string());
        expr += Expr::Const(2.0);
        expr *= Expr::Const(3.0);
        expr -= Expr::Const(1.0);
        expr /= Expr::Const(2.0);
        let expected = Expr::Div(
            Box::new(Expr::Sub(
                Box::new(Expr::Mul(
                    Box::new(Expr::Add(
                        Box::new(Expr::Var("x".to_string())),
                        Box::new(Expr::Const(2.0)),
                    )),
                    Box::new(Expr::Const(3.0)),
                )),
                Box::new(Expr::Const(1.0)),
            )),
            Box::new(Expr::Const(2.0)),
        );
        assert_eq!(expr, expected);
    }
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

    #[test]
    fn symplify_test4() {
        let zero = Expr::Const(0.0);
        let one = Expr::Const(1.0);
        let exp = Expr::Exp(Box::new(zero.clone()));
        let expr1 = one - exp;
        let expr2 = zero.clone() + (expr1 + zero);
        let simplified_expr = expr2.symplify();
        let expected_result = Expr::Const(0.0);
        assert_eq!(simplified_expr, expected_result);
    }
    #[test]
    fn symplify_test5() {
        let n6 = Expr::Const(6.0);
        let n3 = Expr::Const(3.0);
        let n2 = Expr::Const(2.0);
        let n9 = Expr::Const(9.0);
        let n1 = Expr::Const(1.0);
        let expr2 = n6 * n3 / (n2 * n9) - n1;
        let simplified_expr = expr2.symplify();
        let expected_result = Expr::Const(0.0);
        assert_eq!(simplified_expr, expected_result);
    }
    #[test]
    fn symplify_test6() {
        let n6 = Expr::Const(6.0);
        let n3 = Expr::Const(3.0);
        let n2 = Expr::Const(2.0);
        let n9 = Expr::Const(9.0);
        let n1 = Expr::Const(1.0);
        let x = Expr::Var("x".to_owned());
        let expr2 = x.clone() * n6 * n3 / (n2 * n9) - n1.clone();
        let simplified_expr = expr2.symplify();
        println!("{}", simplified_expr);
        let expected_result = x - n1;

        assert_eq!(simplified_expr, expected_result);
    }
    #[test]
    fn test_eval_expression_var() {
        let expr = Expr::Var("x".to_string());
        let vars = vec!["x"];
        let values = vec![5.0];
        assert_eq!(expr.eval_expression(vars, &values), 5.0);
    }

    #[test]
    fn test_eval_expression_const() {
        let expr = Expr::Const(3.14);
        let vars = vec![];
        let values = vec![];
        assert_eq!(expr.eval_expression(vars, &values), 3.14);
    }

    #[test]
    fn test_eval_expression_add() {
        let expr = Expr::Add(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Var("y".to_string())),
        );
        let vars = vec!["x", "y"];
        let values = vec![2.0, 3.0];
        assert_eq!(expr.eval_expression(vars, &values), 5.0);
    }

    #[test]
    fn test_eval_expression_sub() {
        let expr = Expr::Sub(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Var("y".to_string())),
        );
        let vars = vec!["x", "y"];
        let values = vec![5.0, 3.0];
        assert_eq!(expr.eval_expression(vars, &values), 2.0);
    }

    #[test]
    fn test_eval_expression_mul() {
        let expr = Expr::Mul(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Var("y".to_string())),
        );
        let vars = vec!["x", "y"];
        let values = vec![2.0, 3.0];
        assert_eq!(expr.eval_expression(vars, &values), 6.0);
    }

    #[test]
    fn test_eval_expression_div() {
        let expr = Expr::Div(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Var("y".to_string())),
        );
        let vars = vec!["x", "y"];
        let values = vec![6.0, 2.0];
        assert_eq!(expr.eval_expression(vars, &values), 3.0);
    }

    #[test]
    fn test_eval_expression_pow() {
        let expr = Expr::Pow(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Const(2.0)),
        );
        let vars = vec!["x"];
        let values = vec![3.0];
        assert_eq!(expr.eval_expression(vars, &values), 9.0);
    }

    #[test]
    fn test_eval_expression_exp() {
        let expr = Expr::Exp(Box::new(Expr::Var("x".to_string())));
        let vars = vec!["x"];
        let values = vec![1.0];
        assert!((expr.eval_expression(vars, &values) - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn test_eval_expression_ln() {
        let expr = Expr::Ln(Box::new(Expr::Var("x".to_string())));
        let vars = vec!["x"];
        let values = vec![std::f64::consts::E];
        assert!((expr.eval_expression(vars, &values) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_expression_complex() {
        let expr = Expr::Add(
            Box::new(Expr::Mul(
                Box::new(Expr::Var("x".to_string())),
                Box::new(Expr::Var("y".to_string())),
            )),
            Box::new(Expr::Pow(
                Box::new(Expr::Var("z".to_string())),
                Box::new(Expr::Const(2.0)),
            )),
        );
        let vars = vec!["x", "y", "z"];
        let values = vec![2.0, 3.0, 4.0];
        assert_eq!(expr.eval_expression(vars, &values), 22.0); // (2 * 3) + (4^2) = 22
    }
    #[test]
    fn test_taylor_series1D_constant() {
        let expr = Expr::Const(5.0);
        let result = expr.taylor_series1D("x", 0.0, 3);
        assert_eq!(result, Expr::Const(5.0));
    }
    #[test]
    fn test_taylor_series1D_log() {
        let x = Expr::Var("x".to_string());
        let expr = x.clone().ln();
        let result = expr.taylor_series1D_("x", 5.0, 2);
        let e5 = Expr::Const(5.0);
        let expected = e5.clone().ln() + (x.clone() - e5.clone()) / e5.clone()
            - (x.clone() - e5.clone()).pow(Expr::Const(2.0))
                / (Expr::Const(2.0) * e5.clone().pow(Expr::Const(2.0)));
        println!("{} \n {}", result, expected.symplify());
        let taylor_eval = result.lambdify1D()(3.0);
        let expected_eval = expected.lambdify1D()(3.0);
        approx::assert_relative_eq!(taylor_eval, expected_eval, epsilon = 1e-5);
    }
    #[test]
    fn test_taylor_series1D_exp() {
        let x = Expr::Var("x".to_string());

        let exp_expansion = Expr::Const(1.0)
            + x.clone()
            + x.clone().pow(Expr::Const(2.0)) / Expr::Const(2.0)
            + x.clone().pow(Expr::Const(3.0)) / Expr::Const(6.0);
        let exp_eval = exp_expansion.lambdify1D()(1.0);

        let taylor = exp_expansion.taylor_series1D_("x", 0.0, 3);
        println!("taylor: {}", taylor);
        let taylor_eval = taylor.lambdify1D()(1.0);
        assert_eq!(taylor_eval, exp_eval);
    }
    #[test]
    fn test_substitute() {
        let x = Expr::Var("x".to_string());
        let expr = x.clone() + x.clone().pow(Expr::Const(2.0));
        let result = expr.substitute_variable("x", &Expr::Const(3.0));
        assert_eq!(result.symplify(), Expr::Const(12.0));
    }

    #[test]
    fn test_substitute2() {
        let x = Expr::Var("x".to_string());
        let expr = x.clone() + x.clone().pow(Expr::Const(2.0));
        let result = expr.substitute_variable("x", &Expr::Var("y".to_string()));
        let y = Expr::Var("y".to_string());
        let expected = y.clone() + y.clone().pow(Expr::Const(2.0));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_symbols_macro() {
        let (x, y, z) = symbols!(x, y, z);
        assert_eq!(x, Expr::Var("x".to_string()));
        assert_eq!(y, Expr::Var("y".to_string()));
        assert_eq!(z, Expr::Var("z".to_string()));
    }

    #[test]
    fn test_indexed_vars_macro() {
        let (vars, names) = indexed_vars!(3, "x");
        assert_eq!(vars.len(), 3);
        assert_eq!(names.len(), 3);
        assert_eq!(vars[0], Expr::Var("x0".to_string()));
        assert_eq!(vars[1], Expr::Var("x1".to_string()));
        assert_eq!(vars[2], Expr::Var("x2".to_string()));
    }

    #[test]
    fn test_indexed_var_macro() {
        let var = indexed_var!(5, "x");
        assert_eq!(var, Expr::Var("x5".to_string()));
    }

    #[test]
    fn test_indexed_var_2d_macro() {
        let var = indexed_var_2d!(2, 3, "A");
        assert_eq!(var, Expr::Var("A_2_3".to_string()));
    }
}
