//! # Symbolic Expression Simplification Module
//!
//! This module provides comprehensive algebraic simplification capabilities for symbolic expressions.
//! It implements a multi-layered approach to simplification, from basic constant folding to
//! advanced polynomial term collection and algebraic identity application.
//!
//! ## Simplification Strategy
//!
//! The module employs several complementary simplification techniques:
//!
//! 1. **Constant Folding**: Evaluates arithmetic operations on numerical constants
//! 2. **Algebraic Identities**: Applies mathematical rules like x + 0 = x, x * 1 = x
//! 3. **Polynomial Simplification**: Collects like terms in polynomial expressions
//! 4. **Zero Elimination**: Removes multiplication by zero throughout expressions
//! 5. **Power Rules**: Simplifies expressions involving exponents
//!
//! ## Key Features
//!
//! - **Term Ordering Independence**: Handles expressions like (a + b) and (b + a) equivalently
//! - **Distributive Property**: Correctly expands -1 * (a + b) = -a + -b
//! - **Like Term Collection**: Combines terms such as 3x + 2x = 5x
//! - **Nested Expression Handling**: Recursively simplifies complex nested structures
//!
//! ## Performance Considerations
//!
//! The simplification process is designed to be efficient while thorough:
//! - Early termination when no simplification is possible
//! - Polynomial detection to avoid expensive operations on non-polynomial expressions
//! - Memoization-friendly design for repeated simplification calls

use crate::symbolic::symbolic_engine::Expr;
use std::f64::consts::PI;
use std::collections::{HashMap, BTreeMap};

impl Expr {


    //___________________________________SIMPLIFICATION____________________________________

    /// Internal method to eliminate zero-multiplication subexpressions.
    ///
    /// This method performs a specialized optimization by recursively traversing the expression
    /// tree and identifying patterns where multiplication by zero occurs. It's particularly
    /// useful for expressions that have been generated programmatically and may contain
    /// many zero terms that can be eliminated.
    ///
    /// ## Algorithm
    ///
    /// 1. **Recursive Descent**: Traverses the expression tree depth-first
    /// 2. **Zero Detection**: Identifies Mul(Const(0.0), anything) patterns
    /// 3. **Immediate Replacement**: Replaces entire zero-multiplication subtrees with Const(0.0)
    /// 4. **Preservation**: Leaves non-zero expressions unchanged
    ///
    /// ## Examples
    ///
    /// - `0 * (x + y)` → `0`
    /// - `(a * 0) + b` → `0 + b`
    /// - `sin(0 * x)` → `sin(0)`
    ///
    /// ## Performance Notes
    ///
    /// This is a lightweight optimization that can significantly reduce expression complexity
    /// before applying more expensive simplification algorithms.
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
    /// This method performs **constant folding** - a fundamental optimization technique
    /// that evaluates arithmetic operations between numerical constants at compile time
    /// (or in this case, at simplification time) rather than runtime.
    ///
    /// ## Algorithm
    ///
    /// 1. **Recursive Simplification**: First simplifies all subexpressions
    /// 2. **Pattern Matching**: Identifies operations between two constants
    /// 3. **Arithmetic Evaluation**: Performs the actual computation
    /// 4. **Preservation**: Leaves mixed constant-variable operations unchanged
    ///
    /// ## Supported Operations
    ///
    /// - **Basic Arithmetic**: +, -, *, / between constants
    /// - **Transcendental Functions**: exp, ln, sin, cos, etc. on constants
    /// - **Power Operations**: a^b where both a and b are constants
    ///
    /// ## Examples
    ///
    /// - `Const(2) + Const(3)` → `Const(5)`
    /// - `Const(4) * Const(0.5)` → `Const(2.0)`
    /// - `x + Const(2) + Const(3)` → `x + Const(2) + Const(3)` (unchanged)
    ///
    /// ## Limitations
    ///
    /// This method does NOT apply algebraic identities (like x + 0 = x) or collect
    /// like terms. For comprehensive simplification, use `simplify_()` instead.
    ///
    /// # Returns
    /// Expression with all constant arithmetic operations evaluated
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
    /// This is the core simplification engine that applies a wide range of mathematical
    /// rules and identities to reduce expressions to their simplest form. It combines
    /// constant folding, algebraic identities, and polynomial simplification.
    ///
    /// ## Simplification Rules Applied
    ///
    /// ### Additive Identities
    /// - `x + 0 = x` and `0 + x = x`
    /// - `x - 0 = x`
    /// - `x - x = 0`
    ///
    /// ### Multiplicative Identities  
    /// - `x * 1 = x` and `1 * x = x`
    /// - `x * 0 = 0` and `0 * x = 0`
    ///
    /// ### Power Rules
    /// - `x^0 = 1` (for any x)
    /// - `x^1 = x`
    /// - `0^x = 0` (for x > 0)
    /// - `1^x = 1` (for any x)
    /// - `x^a * x^b = x^(a+b)`
    /// - `x^a / x^b = x^(a-b)`
    /// - `(x^a)^b = x^(a*b)`
    ///
    /// ### Transcendental Functions
    /// - `exp(0) = 1`
    /// - `ln(1) = 0`
    /// - `sin(0) = 0`, `cos(0) = 1`, `tan(0) = 0`
    /// - `arcsin(0) = 0`, `arccos(1) = 0`, `arctan(0) = 0`
    ///
    /// ### Division Rules
    /// - `0 / x = 0` (for x ≠ 0)
    /// - `x / 1 = x`
    /// - `x / x = 1`
    ///
    /// ## Advanced Features
    ///
    /// ### Polynomial Simplification
    /// For addition expressions, attempts to collect like terms using `simplify_polynomial()`.
    /// This handles cases like `3x + 2x = 5x` and `(a + b) - (a + b) = 0`.
    ///
    /// ### Nested Constant Folding
    /// Handles complex nested expressions like `(2 * x) * 3 = 6 * x` by recognizing
    /// and combining constants within multiplication chains.
    ///
    /// ### Subtraction Normalization
    /// Converts subtraction to addition of negated terms: `a - b = a + (-1)*b`
    /// This enables polynomial simplification to work on subtraction expressions.
    ///
    /// ## Algorithm Flow
    ///
    /// 1. **Recursive Simplification**: Simplify all subexpressions first
    /// 2. **Identity Application**: Apply mathematical identities based on expression type
    /// 3. **Polynomial Detection**: For Add/Sub, attempt polynomial simplification
    /// 4. **Fallback**: Return simplified expression even if no further reduction possible
    ///
    /// # Returns
    /// Maximally simplified expression using all available rules
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
                    _ => {
                        let expr = Expr::Add(Box::new(lhs), Box::new(rhs));
                        Self::simplify_polynomial(&expr).unwrap_or(expr)
                    }
                }
            }
            Expr::Sub(lhs, rhs) => {
                let lhs = lhs.simplify_();
                let rhs = rhs.simplify_();
                match (&lhs, &rhs) {
                    (Expr::Const(a), Expr::Const(b)) => Expr::Const(a - b), // (a) - (b) = (a - b)
                    (_, Expr::Const(0.0)) => lhs,                           // x - 0 = x
                    // Handle x - x = 0
                    _ if lhs == rhs => Expr::Const(0.0),
                    _ => {
                        // Convert subtraction to addition: a - b = a + (-1)*b
                        let neg_rhs = Expr::Mul(Box::new(Expr::Const(-1.0)), Box::new(rhs)).simplify_();
                        let add_expr = Expr::Add(Box::new(lhs), Box::new(neg_rhs));
                        Self::simplify_polynomial(&add_expr).unwrap_or(add_expr)
                    }
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
                    // Power rules: x^a * x^b = x^(a+b)
                    (Expr::Pow(base1, exp1), Expr::Pow(base2, exp2)) if base1 == base2 => {
                        let new_exp = Expr::Add(exp1.clone(), exp2.clone()).simplify_();
                        Expr::Pow(base1.clone(), Box::new(new_exp))
                    }
                    (Expr::Var(v1), Expr::Pow(base, exp)) | (Expr::Pow(base, exp), Expr::Var(v1)) => {
                        if let Expr::Var(v2) = base.as_ref() {
                            if v1 == v2 {
                                let new_exp = Expr::Add(Box::new(Expr::Const(1.0)), exp.clone()).simplify_();
                                return Expr::Pow(Box::new(Expr::Var(v1.clone())), Box::new(new_exp));
                            }
                        }
                        Expr::Mul(Box::new(lhs), Box::new(rhs))
                    }
                    (Expr::Var(v1), Expr::Var(v2)) if v1 == v2 => {
                        Expr::Pow(Box::new(Expr::Var(v1.clone())), Box::new(Expr::Const(2.0)))
                    }
                    // Handle nested multiplications with constants: (c1 * expr) * c2 = (c1 * c2) * expr
                    // This is crucial for collecting constants in expressions like (2 * x) * 3 = 6 * x
                    (Expr::Mul(inner_lhs, inner_rhs), Expr::Const(c)) => {
                        match (inner_lhs.as_ref(), inner_rhs.as_ref()) {
                            (Expr::Const(c1), _) => {
                                // Pattern: (c1 * expr) * c2 = (c1 * c2) * expr
                                Expr::Mul(Box::new(Expr::Const(c1 * c)), inner_rhs.clone())
                                    .simplify_()
                            }
                            (_, Expr::Const(c1)) => {
                                // Pattern: (expr * c1) * c2 = (c1 * c2) * expr  
                                Expr::Mul(Box::new(Expr::Const(c1 * c)), inner_lhs.clone())
                                    .simplify_()
                            }
                            _ => Expr::Mul(Box::new(lhs), Box::new(rhs)),
                        }
                    }
                    // Symmetric case: c2 * (c1 * expr) = (c1 * c2) * expr
                    (Expr::Const(c), Expr::Mul(inner_lhs, inner_rhs)) => {
                        match (inner_lhs.as_ref(), inner_rhs.as_ref()) {
                            (Expr::Const(c1), _) => {
                                // Pattern: c2 * (c1 * expr) = (c2 * c1) * expr
                                Expr::Mul(Box::new(Expr::Const(c * c1)), inner_rhs.clone())
                                    .simplify_()
                            }
                            (_, Expr::Const(c1)) => {
                                // Pattern: c2 * (expr * c1) = (c2 * c1) * expr
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
                    // Power rules: x^a / x^b = x^(a-b)
                    (Expr::Pow(base1, exp1), Expr::Pow(base2, exp2)) if base1 == base2 => {
                        let new_exp = Expr::Sub(exp1.clone(), exp2.clone()).simplify_();
                        match new_exp {
                            Expr::Const(0.0) => Expr::Const(1.0),
                            _ => Expr::Pow(base1.clone(), Box::new(new_exp))
                        }
                    }
                    (Expr::Var(v1), Expr::Pow(base, exp)) => {
                        if let Expr::Var(v2) = base.as_ref() {
                            if v1 == v2 {
                                let new_exp = Expr::Sub(Box::new(Expr::Const(1.0)), exp.clone()).simplify_();
                                match new_exp {
                                    Expr::Const(0.0) => return Expr::Const(1.0),
                                    _ => return Expr::Pow(Box::new(Expr::Var(v1.clone())), Box::new(new_exp))
                                }
                            }
                        }
                        Expr::Div(Box::new(lhs), Box::new(rhs))
                    }
                    (Expr::Pow(base, exp), Expr::Var(v2)) => {
                        if let Expr::Var(v1) = base.as_ref() {
                            if v1 == v2 {
                                let new_exp = Expr::Sub(exp.clone(), Box::new(Expr::Const(1.0))).simplify_();
                                match new_exp {
                                    Expr::Const(0.0) => return Expr::Const(1.0),
                                    _ => return Expr::Pow(Box::new(Expr::Var(v1.clone())), Box::new(new_exp))
                                }
                            }
                        }
                        Expr::Div(Box::new(lhs), Box::new(rhs))
                    }
                    (Expr::Var(v1), Expr::Var(v2)) if v1 == v2 => Expr::Const(1.0),
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
                    // (x^a)^b = x^(a*b)
                    (Expr::Pow(inner_base, inner_exp), _) => {
                        let new_exp = Expr::Mul(inner_exp.clone(), Box::new(exp)).simplify_();
                        Expr::Pow(inner_base.clone(), Box::new(new_exp))
                    }
                    _ => Expr::Pow(Box::new(base), Box::new(exp)),
                }
            }
            Expr::Exp(expr) => {
                let expr = expr.simplify_();
                match &expr {
                    Expr::Const(0.0) => Expr::Const(1.0),
                    // Only evaluate exp(0), preserve symbolic form otherwise
                    _ => Expr::Exp(Box::new(expr)),
                }
            }
            Expr::Ln(expr) => {
                let expr = expr.simplify_();
                match &expr {
                    Expr::Const(1.0) => Expr::Const(0.0),
                    // Only evaluate ln for simple cases, preserve symbolic form otherwise
                    _ => Expr::Ln(Box::new(expr)),
                }
            } // ln

            Expr::sin(expr) => {
                let expr = expr.simplify_();
                match &expr {
                    Expr::Const(0.0) => Expr::Const(0.0),
                    // Preserve symbolic form for non-zero constants
                    _ => Expr::sin(Box::new(expr)),
                }
            } //sin

            Expr::cos(expr) => {
                let expr = expr.simplify_();
                match &expr {
                    Expr::Const(0.0) => Expr::Const(1.0),
                    // Preserve symbolic form for non-zero constants
                    _ => Expr::cos(Box::new(expr)),
                }
            } //cos
            Expr::tg(expr) => {
                let expr = expr.simplify_();
                match &expr {
                    Expr::Const(0.0) => Expr::Const(0.0),
                    // Preserve symbolic form for non-zero constants
                    _ => Expr::tg(Box::new(expr)),
                }
            } //tg
            Expr::ctg(expr) => {
                let expr = expr.simplify_();
                match &expr {
                    // Preserve symbolic form for all constants
                    _ => Expr::ctg(Box::new(expr)),
                }
            } //ctg

            Expr::arcsin(expr) => {
                let expr = expr.simplify_();
                match &expr {
                    Expr::Const(0.0) => Expr::Const(0.0),
                    // Preserve symbolic form for non-zero constants
                    _ => Expr::arcsin(Box::new(expr)),
                }
            } //arcsin
            Expr::arccos(expr) => {
                let expr = expr.simplify_();
                match &expr {
                    Expr::Const(1.0) => Expr::Const(0.0),
                    // Preserve symbolic form for other constants
                    _ => Expr::arccos(Box::new(expr)),
                }
            } //arccos
            Expr::arctg(expr) => {
                let expr = expr.simplify_();
                match &expr {
                    Expr::Const(0.0) => Expr::Const(0.0),
                    // Preserve symbolic form for non-zero constants
                    _ => Expr::arctg(Box::new(expr)),
                }
            } //arctg

            Expr::arcctg(expr) => {
                let expr = expr.simplify_();
                match &expr {
                    // Preserve symbolic form for all constants
                    _ => Expr::arcctg(Box::new(expr)),
                }
            } //arcctg
        }
    }


    /// Simplify polynomial expressions by collecting like terms.
    ///
    /// This method implements **polynomial term collection** - a sophisticated algorithm
    /// that identifies and combines terms with identical variable parts but different
    /// coefficients. It's the key to handling expressions like `3x + 2x = 5x` and
    /// more complex cases like `(a + b) - (a + b) = 0`.
    ///
    /// ## Algorithm Overview
    ///
    /// 1. **Flattening**: Convert nested Add/Sub expressions into a flat list of terms
    /// 2. **Monomial Extraction**: Extract the variable part and coefficient from each term
    /// 3. **Grouping**: Group terms by their monomial (variable part)
    /// 4. **Coefficient Addition**: Sum coefficients for identical monomials
    /// 5. **Reconstruction**: Build the simplified expression from collected terms
    ///
    /// ## Key Features
    ///
    /// ### Distributive Property Handling
    /// Correctly expands expressions like `-1 * (a + b)` into `-a + -b` during flattening.
    ///
    /// ### Zero Elimination
    /// Terms with zero coefficients are automatically removed from the result.
    ///
    /// ### Non-Polynomial Detection
    /// If any term cannot be expressed as coefficient * monomial, the method returns
    /// `None` to avoid incorrect simplification.
    ///
    /// ## Examples
    ///
    /// - `3x + 2x` → `5x`
    /// - `x^2 + 2x^2 - x^2` → `2x^2`
    /// - `(a + b) + (-1) * (a + b)` → `0`
    /// - `sin(x) + cos(x)` → `None` (not polynomial)
    ///
    /// ## Performance Notes
    ///
    /// - Early termination if fewer than 2 terms (no simplification possible)
    /// - Polynomial validation prevents expensive operations on non-polynomial expressions
    /// - Only returns `Some` if actual simplification occurred
    ///
    /// # Arguments
    /// * `expr` - The expression to attempt polynomial simplification on
    ///
    /// # Returns
    /// * `Some(simplified_expr)` - If polynomial simplification was successful
    /// * `None` - If expression is not polynomial or no simplification possible
    fn simplify_polynomial(expr: &Expr) -> Option<Expr> {
        let mut terms = Vec::new();
        flatten_add(expr, &mut terms);
        if terms.len() < 2 { return None; }
        
        // Check if all terms are polynomial terms before proceeding
        let mut has_non_poly = false;
        for term in &terms {
            let (_, coeff) = extract_monomial(term);
            if coeff == 0.0 && !matches!(term, Expr::Const(0.0)) {
                has_non_poly = true;
                break;
            }
        }
        
        // Don't apply polynomial simplification if there are non-polynomial terms
        if has_non_poly { return None; }
        
        let poly_map = collect_add_terms(&terms);
        if poly_map.len() == terms.len() { return None; }
        
        let mut result_terms = Vec::new();
        for (monomial, coeff) in poly_map {
            if coeff == 0.0 { continue; }
            let term = Self::build_monomial_term(&monomial, coeff);
            result_terms.push(term);
        }
        
        if result_terms.is_empty() {
            Some(Expr::Const(0.0))
        } else if result_terms.len() == 1 {
            Some(result_terms.into_iter().next().unwrap())
        } else {
            Some(result_terms.into_iter().reduce(|a, b| Expr::Add(Box::new(a), Box::new(b))).unwrap())
        }
    }
    
    /// Build a term from monomial key and coefficient.
    ///
    /// This method reconstructs a symbolic expression from its polynomial representation.
    /// It takes a monomial (variable part) and coefficient, and builds the corresponding
    /// `Expr` that represents their product.
    ///
    /// ## Algorithm
    ///
    /// 1. **Constant Terms**: If monomial is empty, return just the coefficient
    /// 2. **Coefficient Handling**: Add coefficient as a factor (unless it's 1.0)
    /// 3. **Variable Processing**: Add each variable raised to its appropriate power
    /// 4. **Multiplication Chain**: Combine all factors using multiplication
    ///
    /// ## Examples
    ///
    /// - `monomial: {}, coeff: 5.0` → `Const(5.0)`
    /// - `monomial: {"x": 1}, coeff: 3.0` → `3.0 * x`
    /// - `monomial: {"x": 2}, coeff: 1.0` → `x^2`
    /// - `monomial: {"x": 1, "y": 2}, coeff: 2.0` → `2.0 * x * y^2`
    ///
    /// ## Special Cases
    ///
    /// - **Unit Coefficient**: When `coeff = 1.0`, it's omitted from the result
    /// - **Linear Terms**: Variables with exponent 1 are not wrapped in `Pow`
    /// - **Empty Monomial**: Returns the coefficient as a constant
    ///
    /// # Arguments
    /// * `monomial` - The variable part (maps variable names to exponents)
    /// * `coeff` - The numerical coefficient
    ///
    /// # Returns
    /// Symbolic expression representing `coeff * monomial`
    fn build_monomial_term(monomial: &MonomialKey, coeff: f64) -> Expr {
        if monomial.0.is_empty() {
            return Expr::Const(coeff);
        }
        
        let mut factors = Vec::new();
        if coeff != 1.0 {
            factors.push(Expr::Const(coeff));
        }
        
        for (var, exp) in &monomial.0 {
            let var_expr = Expr::Var(var.clone());
            if *exp == 1 {
                factors.push(var_expr);
            } else if *exp > 1 {
                factors.push(Expr::Pow(Box::new(var_expr), Box::new(Expr::Const(*exp as f64))));
            }
        }
        
        if factors.is_empty() {
            Expr::Const(1.0)
        } else if factors.len() == 1 {
            factors.into_iter().next().unwrap()
        } else {
            factors.into_iter().reduce(|a, b| Expr::Mul(Box::new(a), Box::new(b))).unwrap()
        }
    }

    /// Public interface for expression simplification.
    ///
    /// Currently delegates to simplify_() but provides a stable API for future
    /// enhancements. This is the recommended method for users to simplify expressions.
    ///
    /// # Returns
    /// Simplified expression using all available simplification rules
    pub fn simplify(&self) -> Expr {
        //let zeros_proceeded = self.nozeros().simplify_numbers();
        let zeros_proceeded = self.simplify_();
        zeros_proceeded
    }

}


/// Represents the variable part of a polynomial term (monomial).
///
/// A monomial key encodes which variables appear in a term and their respective
/// exponents. For example, the term `3x^2y` has monomial key `{"x": 2, "y": 1}`
/// and coefficient `3`.
///
/// ## Design Rationale
///
/// Using `BTreeMap` instead of `HashMap` ensures:
/// - **Deterministic Ordering**: Same monomials always compare equal
/// - **Canonical Representation**: `x*y` and `y*x` have identical keys
/// - **Efficient Comparison**: Lexicographic ordering for consistent results
///
/// ## Examples
///
/// - `x^2` → `MonomialKey({"x": 2})`
/// - `xy^3` → `MonomialKey({"x": 1, "y": 3})`
/// - `5` (constant) → `MonomialKey({})` (empty map)
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MonomialKey(pub BTreeMap<String, i32>);

/// Flatten nested Add/Sub expressions into a list of terms for polynomial processing.
///
/// This function is crucial for polynomial simplification as it converts complex nested
/// addition and subtraction expressions into a flat list of terms that can be easily
/// processed for like-term collection.
///
/// ## Key Features
///
/// ### Distributive Property Implementation
/// The function correctly handles the distributive property of multiplication over addition:
/// - `-1 * (a + b)` becomes `[-1*a, -1*b]`
/// - `-(a + b)` becomes `[-a, -b]`
///
/// ### Subtraction Normalization  
/// Converts subtraction to addition of negated terms:
/// - `a - b` becomes `[a, -1*b]`
///
/// ### Recursive Flattening
/// Handles arbitrarily nested expressions:
/// - `((a + b) + c) - d` becomes `[a, b, c, -1*d]`
///
/// ## Algorithm
///
/// 1. **Add Expressions**: Recursively flatten both operands
/// 2. **Sub Expressions**: Flatten LHS, negate and flatten RHS
/// 3. **Mul by -1**: Distribute negation over addition if present
/// 4. **Other Expressions**: Add to output list unchanged
///
/// ## Examples
///
/// - `a + b` → `[a, b]`
/// - `a - b` → `[a, -1*b]`
/// - `-1 * (x + y)` → `[-1*x, -1*y]`
/// - `(a + b) - (c + d)` → `[a, b, -1*c, -1*d]`
///
/// # Arguments
/// * `expr` - The expression to flatten
/// * `out` - Vector to collect the flattened terms
fn flatten_add(expr: &Expr, out: &mut Vec<Expr>) {
    match expr {
        Expr::Add(a, b) => {
            flatten_add(a, out);
            flatten_add(b, out);
        }
        Expr::Sub(a, b) => {
            flatten_add(a, out);
            // Convert subtraction to addition of negated term
            let neg_b = Expr::Mul(Box::new(Expr::Const(-1.0)), b.clone());
            flatten_add(&neg_b, out);
        }
        // Handle multiplication by -1 as negation - this implements distributive property
        // Critical for expressions like -(a + b) = -a + -b
        Expr::Mul(lhs, rhs) => {
            if let Expr::Const(-1.0) = lhs.as_ref() {
                // Pattern: -1 * (something)
                match rhs.as_ref() {
                    Expr::Add(a, b) => {
                        // Distribute: -1 * (a + b) = (-1 * a) + (-1 * b)
                        let neg_a = Expr::Mul(Box::new(Expr::Const(-1.0)), a.clone());
                        let neg_b = Expr::Mul(Box::new(Expr::Const(-1.0)), b.clone());
                        flatten_add(&neg_a, out);  // Recursively flatten -1*a
                        flatten_add(&neg_b, out);  // Recursively flatten -1*b
                    }
                    _ => out.push(expr.clone()),  // -1 * (non-addition) stays as is
                }
            } else if let Expr::Const(-1.0) = rhs.as_ref() {
                // Pattern: (something) * -1 (symmetric case)
                match lhs.as_ref() {
                    Expr::Add(a, b) => {
                        // Distribute: (a + b) * -1 = (a * -1) + (b * -1)
                        let neg_a = Expr::Mul(Box::new(Expr::Const(-1.0)), a.clone());
                        let neg_b = Expr::Mul(Box::new(Expr::Const(-1.0)), b.clone());
                        flatten_add(&neg_a, out);
                        flatten_add(&neg_b, out);
                    }
                    _ => out.push(expr.clone()),  // (non-addition) * -1 stays as is
                }
            } else {
                // Regular multiplication (not by -1) - no special handling needed
                out.push(expr.clone());
            }
        }
        _ => out.push(expr.clone()),
    }
}

/// Flatten nested multiplication expressions into a list of factors.
///
/// This helper function converts nested multiplication expressions like `(a * b) * c`
/// into a flat list `[a, b, c]`. This is useful for analyzing the structure of
/// multiplication expressions and extracting constants and variables separately.
///
/// ## Algorithm
///
/// Simple recursive descent that:
/// 1. For `Mul(a, b)`: recursively flatten both `a` and `b`
/// 2. For other expressions: add to the output list
///
/// ## Examples
///
/// - `a * b` → `[a, b]`
/// - `(a * b) * c` → `[a, b, c]`
/// - `2 * x * y` → `[2, x, y]`
///
/// # Arguments
/// * `expr` - The multiplication expression to flatten
/// * `out` - Vector to collect the factors
fn flatten_mul(expr: &Expr, out: &mut Vec<Expr>) {
    match expr {
        Expr::Mul(a, b) => {
            flatten_mul(a, out);
            flatten_mul(b, out);
        }
        _ => out.push(expr.clone()),
    }
}

/// Collect terms in a sum into a polynomial map: monomial → coefficient.
///
/// This function implements the core of polynomial term collection. It takes a list
/// of terms (from `flatten_add`) and groups them by their monomial part, summing
/// the coefficients for identical monomials.
///
/// ## Algorithm
///
/// 1. **Initialize**: Create empty HashMap for monomial → coefficient mapping
/// 2. **Process Each Term**: Extract monomial and coefficient using `extract_monomial`
/// 3. **Accumulate**: Add coefficient to existing entry or create new entry
/// 4. **Return**: Complete mapping of monomials to their total coefficients
///
/// ## Examples
///
/// Input terms: `[3x, 2x, y, -x]`
/// - `3x` → `({"x": 1}, 3.0)`
/// - `2x` → `({"x": 1}, 2.0)` → accumulate to `5.0`
/// - `y` → `({"y": 1}, 1.0)`
/// - `-x` → `({"x": 1}, -1.0)` → accumulate to `4.0`
///
/// Result: `{{"x": 1} → 4.0, {"y": 1} → 1.0}`
///
/// # Arguments
/// * `terms` - List of terms to collect
///
/// # Returns
/// HashMap mapping each unique monomial to its total coefficient
fn collect_add_terms(terms: &[Expr]) -> HashMap<MonomialKey, f64> {
    let mut poly = HashMap::new();
    for t in terms {
        let (mon, coeff) = extract_monomial(t);
        *poly.entry(mon).or_insert(0.0) += coeff;
    }
    poly
}

/// Extract a monomial from an expression if it’s a product of constants and variables/powers
fn extract_monomial(expr: &Expr) -> (MonomialKey, f64) {
    match expr {
        Expr::Const(c) => (MonomialKey(BTreeMap::new()), *c),
        Expr::Var(v) => {
            let mut m = BTreeMap::new();
            m.insert(v.clone(), 1);
            (MonomialKey(m), 1.0)
        }
        Expr::Mul(lhs, rhs) => {
            // Handle simple cases first - this is the fast path for common patterns
            match (lhs.as_ref(), rhs.as_ref()) {
                // Pattern: -1 * something or something * -1
                // Critical for handling negated terms from distributive property
                (Expr::Const(-1.0), other) | (other, Expr::Const(-1.0)) => {
                    let (mon, coeff) = extract_monomial(other);
                    (mon, -coeff)  // Negate the coefficient
                }
                // Pattern: constant * something or something * constant
                // Handles simple coefficient extraction like 3 * x
                (Expr::Const(c), other) | (other, Expr::Const(c)) => {
                    let (mon, coeff) = extract_monomial(other);
                    (mon, c * coeff)  // Multiply coefficients
                }
                _ => {
                    // Fall back to flattening approach
                    let mut factors = Vec::new();
                    flatten_mul(expr, &mut factors);
                    let mut coeff = 1.0;
                    let mut map = BTreeMap::new();
                    let mut has_non_poly = false;
                    
                    for f in factors {
                        match f {
                            Expr::Const(c) => coeff *= c,
                            Expr::Var(v) => *map.entry(v).or_insert(0) += 1,
                            Expr::Pow(base, exp) => {
                                if let Expr::Var(v) = *base {
                                    if let Expr::Const(n) = *exp {
                                        *map.entry(v).or_insert(0) += n as i32;
                                    } else {
                                        has_non_poly = true;
                                    }
                                } else {
                                    has_non_poly = true;
                                }
                            }
                            _ => has_non_poly = true,
                        }
                    }
                    
                    if has_non_poly {
                        (MonomialKey(BTreeMap::new()), 0.0)
                    } else {
                        (MonomialKey(map), coeff)
                    }
                }
            }
        }
        Expr::Pow(base, exp) => {
            if let (Expr::Var(v), Expr::Const(n)) = (base.as_ref(), exp.as_ref()) {
                let mut m = BTreeMap::new();
                m.insert(v.clone(), *n as i32);
                (MonomialKey(m), 1.0)
            } else {
                (MonomialKey(BTreeMap::new()), 0.0)
            }
        }
        _ => (MonomialKey(BTreeMap::new()), 0.0), // non-poly term ignored
    }
}

/// Build Horner form for a univariate polynomial.
///
/// This function converts a polynomial represented as a list of (exponent, coefficient)
/// pairs into **Horner form** - a computationally efficient representation that minimizes
/// the number of multiplications needed for evaluation.
///
/// ## Horner Form
///
/// Horner form transforms a polynomial like `ax³ + bx² + cx + d` into the nested form
/// `((ax + b)x + c)x + d`. This reduces the number of multiplications from 6 to 3.
///
/// ## Algorithm
///
/// 1. **Coefficient Array**: Create array indexed by exponent
/// 2. **Populate**: Fill array with coefficients from input terms
/// 3. **Find Degree**: Locate highest non-zero coefficient
/// 4. **Nest**: Build nested multiplication structure from highest to lowest degree
///
/// ## Examples
///
/// - Input: `[(0, 1), (1, 2), (2, 3)]` (represents `3x² + 2x + 1`)
/// - Output: `((3 * x) + 2) * x + 1` (Horner form)
///
/// ## Performance Benefits
///
/// - **Fewer Operations**: O(n) multiplications instead of O(n²)
/// - **Numerical Stability**: Reduces floating-point errors
/// - **Cache Efficiency**: Sequential memory access pattern
///
/// ## Note
///
/// This function is currently unused but provides a foundation for future
/// polynomial optimization features.
///
/// # Arguments
/// * `var` - Variable name for the polynomial
/// * `terms` - List of (exponent, coefficient) pairs
///
/// # Returns
/// Expression in Horner form
fn horner_univariate(var: &str,  terms: Vec<(i32, f64)>) -> Expr {
    use Expr::*;
    if terms.is_empty() {
        return Const(0.0);
    }
    
    // Find the highest degree to determine array size
    let max_deg = terms.iter().map(|(e, _)| *e).max().unwrap();
    
    // Create coefficient array indexed by exponent
    let mut coeffs = vec![0.0; (max_deg + 1) as usize];
    
    // Populate coefficient array from input terms
    for (exp, c) in terms {
        if exp >= 0 {
            coeffs[exp as usize] += c;  // Handle multiple terms with same exponent
        }
    }
    
    // Find the actual highest degree (skip trailing zeros)
    let mut i = coeffs.len() - 1;
    while i > 0 && coeffs[i] == 0.0 {
        i -= 1;
    }
    
    // Start with the highest degree coefficient
    let mut acc = Const(coeffs[i]);
    let x = Var(var.to_string());
    
    // Build Horner form: (((...((a_n * x) + a_{n-1}) * x) + a_{n-2}) * x) + ... + a_0
    while i > 0 {
        acc = Add(Box::new(Mul(Box::new(acc), Box::new(x.clone()))), Box::new(Const(coeffs[i - 1])));
        i -= 1;
    }
    acc
}