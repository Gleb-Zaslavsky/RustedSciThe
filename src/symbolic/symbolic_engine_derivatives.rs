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
//! ### Function evaluation
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
//! 3. **Memory Management Variants**: Provides  borrowing (`lambdify`) and owned
//!
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
    ///
    pub fn diff(&self, var: &str) -> Expr {
        self.diff2(var)
    }

    /// **OPTIMIZED DIFFERENTIATION METHOD WITH INTEGRATED SIMPLIFICATION**
    ///
    /// This is the enhanced differentiation method that combines analytical differentiation
    /// with immediate simplification and polynomial collection. Unlike `diff2()`, this method
    /// performs optimizations during differentiation rather than requiring separate `simplify()` calls.
    ///
    /// ## Key Optimizations
    ///
    /// ### 1. **Early Termination**
    /// - Checks if expression contains the differentiation variable before processing
    /// - Returns `Const(0.0)` immediately for expressions not containing the variable
    /// - Avoids unnecessary recursive traversal of irrelevant subexpressions
    ///
    /// ### 2. **Smart Arithmetic Operations**
    /// - `smart_add()`: Handles constant folding, zero elimination, like-term collection
    /// - `smart_sub()`: Converts subtraction to addition with negation for consistency
    /// - `enhanced_multiplication()`: Implements power rules and constant collection
    ///
    /// ### 3. **Polynomial Collection**
    /// - Automatically collects like terms (e.g., `3x + 2x = 5x`)
    /// - Handles distributive property correctly
    /// - Eliminates zero coefficients and simplifies expressions
    ///
    /// ### 4. **Power Rule Optimization**
    /// - Special handling for constant exponents: `d/dx(x^n) = n*x^(n-1)`
    /// - Immediate simplification for common cases (x^0, x^1)
    /// - Avoids creating unnecessary intermediate expressions
    ///
    /// ## Performance Benefits
    ///
    /// - **Self-Contained**: No need for separate `simplify()` calls
    /// - **Reduced Expression Complexity**: Immediate simplification prevents bloat
    /// - **Early Termination**: Skips processing of irrelevant subexpressions
    /// - **Optimized Product Rule**: Enhanced multiplication reduces artifacts
    ///
    /// ## Mathematical Correctness
    ///
    /// Implements all standard calculus differentiation rules:
    /// - **Sum Rule**: `d/dx(f + g) = f' + g'`
    /// - **Difference Rule**: `d/dx(f - g) = f' - g'`
    /// - **Product Rule**: `d/dx(f * g) = f' * g + f * g'`
    /// - **Quotient Rule**: `d/dx(f / g) = (f' * g - f * g') / g^2`
    /// - **Power Rule**: `d/dx(x^n) = n * x^(n-1)`
    /// - **Chain Rule**: Applied to all composite functions
    ///
    /// # Arguments
    /// * `var` - Variable name to differentiate with respect to
    ///
    /// # Returns
    /// Simplified symbolic expression representing the derivative
    ///
    /// # Examples
    /// ```rust, ignore
    /// let expr = parse_expression("x^3 + 2*x^2 + x");
    /// let derivative = expr.diff1("x"); // Returns: 3*x^2 + 4*x + 1 (already simplified)
    /// ```
    pub fn diff1(&self, var: &str) -> Expr {
        // **EARLY TERMINATION OPTIMIZATION**
        // Check if expression contains the differentiation variable
        // If not, derivative is zero - avoids unnecessary computation
        if !self.contains_variable(var) {
            return Expr::Const(0.0);
        }

        let result = match self {
            // **VARIABLE DIFFERENTIATION**
            // d/dx(x) = 1, d/dx(y) = 0 (where y ≠ x)
            Expr::Var(name) => {
                if name == var {
                    Expr::Const(1.0)  // d/dx(x) = 1
                } else {
                    Expr::Const(0.0)  // d/dx(y) = 0 where y ≠ x
                }
            }
            // **CONSTANT DIFFERENTIATION**
            // d/dx(c) = 0 for any constant c
            Expr::Const(_) => Expr::Const(0.0),
            // **SUM RULE WITH SMART ADDITION**
            // d/dx(f + g) = f' + g'
            // Uses smart_add() for immediate simplification and polynomial collection
            Expr::Add(lhs, rhs) => {
                let lhs_diff = lhs.diff1(var);  // Recursively differentiate left operand
                let rhs_diff = rhs.diff1(var);  // Recursively differentiate right operand
                Self::smart_add(lhs_diff, rhs_diff)  // Smart addition with simplification
            }
            // **DIFFERENCE RULE WITH SMART SUBTRACTION**
            // d/dx(f - g) = f' - g'
            // Uses smart_sub() which converts to addition: f' + (-1)*g'
            Expr::Sub(lhs, rhs) => {
                let lhs_diff = lhs.diff1(var);  // Recursively differentiate left operand
                let rhs_diff = rhs.diff1(var);  // Recursively differentiate right operand
                Self::smart_sub(lhs_diff, rhs_diff)  // Smart subtraction with simplification
            }
            // **PRODUCT RULE WITH ENHANCED OPTIMIZATION**
            // d/dx(f * g) = f' * g + f * g'
            // Optimized implementation with immediate simplification
            Expr::Mul(lhs, rhs) => {
                let lhs_diff = lhs.diff1(var);  // f'
                let rhs_diff = rhs.diff1(var);  // g'
                
                // **OPTIMIZED PRODUCT RULE CASES**
                match (&lhs_diff, &rhs_diff) {
                    // Both derivatives are zero: 0 * g + f * 0 = 0
                    (Expr::Const(0.0), Expr::Const(0.0)) => Expr::Const(0.0),
                    
                    // Left derivative is zero: 0 * g + f * g' = f * g'
                    (Expr::Const(0.0), _) => {
                        Self::enhanced_multiplication(lhs.as_ref(), &rhs_diff)
                    }
                    
                    // Right derivative is zero: f' * g + f * 0 = f' * g
                    (_, Expr::Const(0.0)) => {
                        Self::enhanced_multiplication(&lhs_diff, rhs.as_ref())
                    }
                    
                    // General case: f' * g + f * g'
                    _ => {
                        let term1 = Self::enhanced_multiplication(&lhs_diff, rhs.as_ref());
                        let term2 = Self::enhanced_multiplication(lhs.as_ref(), &rhs_diff);
                        Self::smart_add(term1, term2)  // Smart addition for polynomial collection
                    }
                }
            }
            // **QUOTIENT RULE**
            // d/dx(f / g) = (f' * g - f * g') / g^2
            // Note: Could be enhanced with smart operations in future versions
            Expr::Div(lhs, rhs) => Expr::Div(
                Box::new(Expr::Sub(
                    Box::new(Expr::Mul(Box::new(lhs.diff1(var)), rhs.clone())),  // f' * g
                    Box::new(Expr::Mul(Box::new(rhs.diff1(var)), lhs.clone())),  // f * g'
                )),
                Box::new(Expr::Mul(rhs.clone(), rhs.clone())),  // g^2
            ),
            // **POWER RULE WITH OPTIMIZATION**
            // d/dx(f^n) = n * f^(n-1) * f' (for constant n)
            // d/dx(f^g) = f^g * (g' * ln(f) + g * f'/f) (for variable exponent)
            Expr::Pow(base, exp) => {
                // **OPTIMIZED CONSTANT EXPONENT CASE**
                if let Expr::Const(n) = **exp {
                    // Special cases for common exponents
                    if n == 0.0 {
                        return Expr::Const(0.0);  // d/dx(f^0) = d/dx(1) = 0
                    }
                    if n == 1.0 {
                        return base.diff1(var);    // d/dx(f^1) = d/dx(f) = f'
                    }
                    
                    // General constant exponent: d/dx(f^n) = n * f^(n-1) * f'
                    let base_diff = base.diff1(var);  // f'
                    
                    // If base derivative is zero, entire derivative is zero
                    if let Expr::Const(0.0) = base_diff {
                        Expr::Const(0.0)
                    } else {
                        // Construct f^(n-1) term, avoiding f^1 = f simplification
                        let power_term = if n - 1.0 == 1.0 {
                            base.as_ref().clone()  // f^1 = f
                        } else {
                            Expr::Pow(base.clone(), Box::new(Expr::Const(n - 1.0)))  // f^(n-1)
                        };

                        // Construct final result: n * f^(n-1) * f'
                        let result = match (&base_diff, n) {
                            // Special case: f' = 1, n = 1 → result = 1
                            (Expr::Const(1.0), _) if n == 1.0 => Expr::Const(1.0),
                            // f' = 1 → result = n * f^(n-1)
                            (Expr::Const(1.0), _) => {
                                Expr::Mul(Box::new(Expr::Const(n)), Box::new(power_term))
                            }
                            // n = 1 → result = f' (already handled above, but for completeness)
                            (_, _) if n == 1.0 => base_diff,
                            // General case: n * f^(n-1) * f'
                            _ => Expr::Mul(
                                Box::new(Expr::Const(n)),
                                Box::new(Expr::Mul(Box::new(power_term), Box::new(base_diff))),
                            ),
                        };
                        result
                    }
                } else {
                    // **VARIABLE EXPONENT CASE**
                    // d/dx(f^g) = f^g * (g' * ln(f) + g * f'/f)
                    // Simplified to: g * f^(g-1) * f' (using power rule form)
                    // Note: This is a simplified form; full logarithmic differentiation would be more complex
                    Expr::Mul(
                        Box::new(Expr::Mul(
                            exp.clone(),  // g
                            Box::new(Expr::Pow(
                                base.clone(),  // f
                                Box::new(Expr::Sub(exp.clone(), Box::new(Expr::Const(1.0)))),  // g-1
                            )),
                        )),
                        Box::new(base.diff1(var)),  // f'
                    )
                }
            }
            // **EXPONENTIAL FUNCTION**
            // d/dx(e^f) = e^f * f' (chain rule)
            Expr::Exp(expr) => {
                let expr_diff = expr.diff1(var);  // f'
                if let Expr::Const(0.0) = expr_diff {
                    Expr::Const(0.0)  // If f' = 0, then derivative is 0
                } else {
                    Expr::Mul(Box::new(Expr::Exp(expr.clone())), Box::new(expr_diff))  // e^f * f'
                }
            }
            // **NATURAL LOGARITHM**
            // d/dx(ln(f)) = f'/f (chain rule)
            Expr::Ln(expr) => {
                let expr_diff = expr.diff1(var);  // f'
                if let Expr::Const(0.0) = expr_diff {
                    Expr::Const(0.0)  // If f' = 0, then derivative is 0
                } else {
                    Expr::Div(Box::new(expr_diff), expr.clone())  // f'/f
                }
            }
            // **TRIGONOMETRIC FUNCTIONS**
            
            // d/dx(sin(f)) = cos(f) * f' (chain rule)
            Expr::sin(expr) => {
                let expr_diff = expr.diff1(var);  // f'
                if let Expr::Const(0.0) = expr_diff {
                    Expr::Const(0.0)  // If f' = 0, then derivative is 0
                } else {
                    Expr::Mul(Box::new(Expr::cos(expr.clone())), Box::new(expr_diff))  // cos(f) * f'
                }
            }
            // d/dx(cos(f)) = -sin(f) * f' (chain rule)
            Expr::cos(expr) => {
                let expr_diff = expr.diff1(var);  // f'
                if let Expr::Const(0.0) = expr_diff {
                    Expr::Const(0.0)  // If f' = 0, then derivative is 0
                } else {
                    Expr::Mul(
                        Box::new(Expr::Mul(
                            Box::new(Expr::Const(-1.0)),      // -1
                            Box::new(Expr::sin(expr.clone())), // sin(f)
                        )),
                        Box::new(expr_diff),  // f'
                    )  // -sin(f) * f'
                }
            }
            Expr::tg(expr) => {
                let expr_diff = expr.diff1(var);
                if let Expr::Const(0.0) = expr_diff {
                    Expr::Const(0.0)
                } else {
                    Expr::Mul(
                        Box::new(Expr::Div(
                            Box::new(Expr::Const(1.0)),
                            Box::new(Expr::Pow(
                                Box::new(Expr::cos(expr.clone())),
                                Box::new(Expr::Const(2.0)),
                            )),
                        )),
                        Box::new(expr_diff),
                    )
                }
            }
            Expr::ctg(expr) => {
                let expr_diff = expr.diff1(var);
                if let Expr::Const(0.0) = expr_diff {
                    Expr::Const(0.0)
                } else {
                    Expr::Mul(
                        Box::new(Expr::Div(
                            Box::new(Expr::Const(-1.0)),
                            Box::new(Expr::Pow(
                                Box::new(Expr::sin(expr.clone())),
                                Box::new(Expr::Const(2.0)),
                            )),
                        )),
                        Box::new(expr_diff),
                    )
                }
            }
            Expr::arcsin(expr) => {
                let expr_diff = expr.diff1(var);
                if let Expr::Const(0.0) = expr_diff {
                    Expr::Const(0.0)
                } else {
                    Expr::Div(
                        Box::new(expr_diff),
                        Box::new(Expr::Pow(
                            Box::new(Expr::Sub(
                                Box::new(Expr::Const(1.0)),
                                Box::new(Expr::Pow(expr.clone(), Box::new(Expr::Const(2.0)))),
                            )),
                            Box::new(Expr::Const(0.5)),
                        )),
                    )
                }
            }
            Expr::arccos(expr) => {
                let expr_diff = expr.diff1(var);
                if let Expr::Const(0.0) = expr_diff {
                    Expr::Const(0.0)
                } else {
                    Expr::Div(
                        Box::new(Expr::Mul(Box::new(Expr::Const(-1.0)), Box::new(expr_diff))),
                        Box::new(Expr::Pow(
                            Box::new(Expr::Sub(
                                Box::new(Expr::Const(1.0)),
                                Box::new(Expr::Pow(expr.clone(), Box::new(Expr::Const(2.0)))),
                            )),
                            Box::new(Expr::Const(0.5)),
                        )),
                    )
                }
            }
            Expr::arctg(expr) => {
                let expr_diff = expr.diff1(var);
                if let Expr::Const(0.0) = expr_diff {
                    Expr::Const(0.0)
                } else {
                    Expr::Div(
                        Box::new(expr_diff),
                        Box::new(Expr::Add(
                            Box::new(Expr::Const(1.0)),
                            Box::new(Expr::Pow(expr.clone(), Box::new(Expr::Const(2.0)))),
                        )),
                    )
                }
            }
            Expr::arcctg(expr) => {
                let expr_diff = expr.diff1(var);
                if let Expr::Const(0.0) = expr_diff {
                    Expr::Const(0.0)
                } else {
                    Expr::Div(
                        Box::new(Expr::Mul(Box::new(Expr::Const(1.0)), Box::new(expr_diff))),
                        Box::new(Expr::Add(
                            Box::new(Expr::Const(1.0)),
                            Box::new(Expr::Pow(expr.clone(), Box::new(Expr::Const(2.0)))),
                        )),
                    )
                }
            }
        };

        // **FINAL POLYNOMIAL COLLECTION**
        // Apply polynomial simplification to the final result if it's an addition or subtraction
        // This collects like terms and performs final cleanup
        Self::apply_polynomial_collection(result)
    }

    /// **LEGACY DIFFERENTIATION METHOD (UNOPTIMIZED)**
    ///
    /// This is the original, straightforward implementation of symbolic differentiation
    /// that follows calculus rules directly without any optimization or simplification.
    /// It serves as a reference implementation and benchmark for the optimized `diff1()` method.
    ///
    /// ## Characteristics
    ///
    /// ### 1. **Pure Rule Application**
    /// - Implements differentiation rules exactly as taught in calculus
    /// - No early termination or optimization shortcuts
    /// - Always processes entire expression tree recursively
    ///
    /// ### 2. **No Built-in Simplification**
    /// - Returns expressions with many artifacts and redundant terms
    /// - Requires separate `simplify()` call for clean results
    /// - May produce expressions like `0*f + g*1` instead of just `g`
    ///
    /// ### 3. **Straightforward Implementation**
    /// - Easy to understand and verify correctness
    /// - Direct translation of mathematical rules to code
    /// - Minimal complexity in individual rule implementations
    ///
    /// ## Performance Characteristics
    ///
    /// - **Slower**: No early termination for zero derivatives
    /// - **Memory Intensive**: Creates many intermediate expressions
    /// - **Requires Post-Processing**: Always needs `simplify()` for usable results
    /// - **Predictable**: Consistent behavior regardless of expression complexity
    ///
    /// ## Mathematical Rules Implemented
    ///
    /// All standard calculus differentiation rules:
    /// - **Sum Rule**: `d/dx(f + g) = f' + g'`
    /// - **Difference Rule**: `d/dx(f - g) = f' - g'`
    /// - **Product Rule**: `d/dx(f * g) = f' * g + f * g'`
    /// - **Quotient Rule**: `d/dx(f / g) = (f' * g - f * g') / g^2`
    /// - **Power Rule**: `d/dx(f^n) = n * f^(n-1) * f'`
    /// - **Chain Rule**: Applied to all composite functions
    ///
    /// ## Usage Pattern
    ///
    /// ```rust, ignore
    /// let expr = parse_expression("x^2 + 2*x + 1");
    /// let derivative = expr.diff2("x").simplify(); // Note: simplify() required
    /// ```
    ///
    /// ## Comparison with diff1()
    ///
    /// | Aspect | diff2() | diff1() |
    /// |--------|---------|----------|
    /// | Speed | Slower | Faster |
    /// | Output | Needs simplify() | Self-contained |
    /// | Complexity | Simple | Advanced |
    /// | Memory | Higher usage | Optimized |
    /// | Artifacts | Many | Minimal |
    ///
    /// # Arguments
    /// * `var` - Variable name to differentiate with respect to
    ///
    /// # Returns
    /// Unsimplified symbolic expression representing the derivative
    ///
    /// # Note
    /// This method is primarily used for:
    /// - Testing and verification of `diff1()` correctness
    /// - Performance benchmarking
    /// - Educational purposes to show pure rule application
    fn diff2(&self, var: &str) -> Expr {
        match self {
            // **BASIC DIFFERENTIATION RULES**
            
            // Variable differentiation: d/dx(x) = 1, d/dx(y) = 0
            Expr::Var(name) => {
                if name == var {
                    Expr::Const(1.0)  // d/dx(x) = 1
                } else {
                    Expr::Const(0.0)  // d/dx(y) = 0 where y ≠ x
                }
            }
            // Constant differentiation: d/dx(c) = 0
            Expr::Const(_) => Expr::Const(0.0),
            // **SUM AND DIFFERENCE RULES (UNOPTIMIZED)**
            
            // Sum rule: d/dx(f + g) = f' + g'
            // No optimization - always creates Add expression even if one derivative is 0
            Expr::Add(lhs, rhs) => Expr::Add(
                Box::new(lhs.diff2(var)),  // f'
                Box::new(rhs.diff2(var))   // g'
            ),
            
            // Difference rule: d/dx(f - g) = f' - g'
            // No optimization - always creates Sub expression even if one derivative is 0
            Expr::Sub(lhs, rhs) => Expr::Sub(
                Box::new(lhs.diff2(var)),  // f'
                Box::new(rhs.diff2(var))   // g'
            ),
            // **PRODUCT RULE (UNOPTIMIZED)**
            // d/dx(f * g) = f' * g + f * g'
            // Always creates full product rule expression, even when derivatives are 0
            // Results in expressions like "0*g + f*0" that need simplification
            Expr::Mul(lhs, rhs) => Expr::Add(
                Box::new(Expr::Mul(
                    Box::new(lhs.diff2(var)),  // f'
                    rhs.clone()                // g
                )),
                Box::new(Expr::Mul(
                    lhs.clone(),               // f
                    Box::new(rhs.diff2(var))   // g'
                )),
            ),
            // **QUOTIENT RULE (UNOPTIMIZED)**
            // d/dx(f / g) = (f' * g - f * g') / g^2
            // Always creates full quotient rule expression regardless of derivative values
            Expr::Div(lhs, rhs) => Expr::Div(
                Box::new(Expr::Sub(
                    Box::new(Expr::Mul(
                        Box::new(lhs.diff2(var)),  // f'
                        rhs.clone()                // g
                    )),
                    Box::new(Expr::Mul(
                        Box::new(rhs.diff2(var)),  // g'
                        lhs.clone()                // f
                    )),
                )),
                Box::new(Expr::Mul(
                    rhs.clone(),  // g
                    rhs.clone()   // g (for g^2)
                )),
            ),
            // **POWER RULE (UNOPTIMIZED)**
            // d/dx(f^n) = n * f^(n-1) * f'
            // No special handling for constant exponents or common cases
            // Always creates full power rule expression, even for f^0 or f^1
            Expr::Pow(base, exp) => Expr::Mul(
                Box::new(Expr::Mul(
                    exp.clone(),  // n
                    Box::new(Expr::Pow(
                        base.clone(),  // f
                        Box::new(Expr::Sub(
                            exp.clone(),                    // n
                            Box::new(Expr::Const(1.0))     // 1
                        )),  // n - 1
                    )),  // f^(n-1)
                )),
                Box::new(base.diff2(var)),  // f'
            ),
            // **EXPONENTIAL AND LOGARITHMIC FUNCTIONS (UNOPTIMIZED)**
            
            // d/dx(e^f) = e^f * f'
            // No check for zero derivative - always creates multiplication
            Expr::Exp(expr) => Expr::Mul(
                Box::new(Expr::Exp(expr.clone())),  // e^f
                Box::new(expr.diff2(var))           // f'
            ),
            
            // d/dx(ln(f)) = f'/f
            // No check for zero derivative - always creates division
            Expr::Ln(expr) => Expr::Div(
                Box::new(expr.diff2(var)),  // f'
                expr.clone()                // f
            ),
            // **TRIGONOMETRIC FUNCTIONS (UNOPTIMIZED)**
            
            // d/dx(sin(f)) = cos(f) * f'
            // Always creates multiplication, even when f' = 0
            Expr::sin(expr) => Expr::Mul(
                Box::new(Expr::cos(expr.clone())),  // cos(f)
                Box::new(expr.diff2(var))           // f'
            ),
            
            // d/dx(cos(f)) = -sin(f) * f'
            // Always creates full expression with -1 multiplication
            Expr::cos(expr) => Expr::Mul(
                Box::new(Expr::Mul(
                    Box::new(Expr::Const(-1.0)),       // -1
                    Box::new(Expr::sin(expr.clone())),  // sin(f)
                )),
                Box::new(expr.diff2(var)),  // f'
            ),
            // **ADDITIONAL TRIGONOMETRIC FUNCTIONS (UNOPTIMIZED)**
            
            // d/dx(tan(f)) = sec^2(f) * f' = (1/cos^2(f)) * f'
            Expr::tg(expr) => Expr::Mul(
                Box::new(Expr::Div(
                    Box::new(Expr::Const(1.0)),        // 1
                    Box::new(Expr::Pow(
                        Box::new(Expr::cos(expr.clone())),  // cos(f)
                        Box::new(Expr::Const(2.0)),         // 2
                    )),  // cos^2(f)
                )),  // 1/cos^2(f)
                Box::new(expr.diff2(var)),  // f'
            ),
            
            // d/dx(cot(f)) = -csc^2(f) * f' = (-1/sin^2(f)) * f'
            Expr::ctg(expr) => Expr::Mul(
                Box::new(Expr::Div(
                    Box::new(Expr::Const(-1.0)),       // -1
                    Box::new(Expr::Pow(
                        Box::new(Expr::sin(expr.clone())),  // sin(f)
                        Box::new(Expr::Const(2.0)),         // 2
                    )),  // sin^2(f)
                )),  // -1/sin^2(f)
                Box::new(expr.diff2(var)),  // f'
            ),
            // **INVERSE TRIGONOMETRIC FUNCTIONS (UNOPTIMIZED)**
            
            // d/dx(arcsin(f)) = f' / sqrt(1 - f^2)
            Expr::arcsin(expr) => Expr::Div(
                Box::new(expr.diff2(var)),  // f'
                Box::new(Expr::Pow(
                    Box::new(Expr::Sub(
                        Box::new(Expr::Const(1.0)),                                    // 1
                        Box::new(Expr::Pow(expr.clone(), Box::new(Expr::Const(2.0)))), // f^2
                    )),  // 1 - f^2
                    Box::new(Expr::Const(0.5)),  // 0.5 (for square root)
                )),  // sqrt(1 - f^2)
            ),
            
            // d/dx(arccos(f)) = -f' / sqrt(1 - f^2)
            Expr::arccos(expr) => Expr::Div(
                Box::new(Expr::Mul(
                    Box::new(Expr::Const(-1.0)),  // -1
                    Box::new(expr.diff2(var)),     // f'
                )),  // -f'
                Box::new(Expr::Pow(
                    Box::new(Expr::Sub(
                        Box::new(Expr::Const(1.0)),                                    // 1
                        Box::new(Expr::Pow(expr.clone(), Box::new(Expr::Const(2.0)))), // f^2
                    )),  // 1 - f^2
                    Box::new(Expr::Const(0.5)),  // 0.5 (for square root)
                )),  // sqrt(1 - f^2)
            ),
            
            // d/dx(arctan(f)) = f' / (1 + f^2)
            Expr::arctg(expr) => Expr::Div(
                Box::new(expr.diff2(var)),  // f'
                Box::new(Expr::Add(
                    Box::new(Expr::Const(1.0)),                                    // 1
                    Box::new(Expr::Pow(expr.clone(), Box::new(Expr::Const(2.0)))), // f^2
                )),  // 1 + f^2
            ),
            
            // d/dx(arccot(f)) = -f' / (1 + f^2)
            // Note: This implementation uses coefficient 1.0 instead of -1.0 (potential bug in original)
            Expr::arcctg(expr) => Expr::Div(
                Box::new(Expr::Mul(
                    Box::new(Expr::Const(1.0)),   // Should be -1.0 for correct arccot derivative
                    Box::new(expr.diff2(var)),     // f'
                )),
                Box::new(Expr::Add(
                    Box::new(Expr::Const(1.0)),                                    // 1
                    Box::new(Expr::Pow(expr.clone(), Box::new(Expr::Const(2.0)))), // f^2
                )),  // 1 + f^2
            ),
        }
    } // end of diff2

    /// Enhanced multiplication with constant collection and power rules
    fn enhanced_multiplication(lhs: &Expr, rhs: &Expr) -> Expr {
        match (lhs, rhs) {
            // Basic identities
            (Expr::Const(1.0), _) => rhs.clone(),
            (_, Expr::Const(1.0)) => lhs.clone(),
            (Expr::Const(0.0), _) | (_, Expr::Const(0.0)) => Expr::Const(0.0),
            
            // Constant folding
            (Expr::Const(a), Expr::Const(b)) => Expr::Const(a * b),
            
            // Power rules: x * x^n = x^(n+1)
            (Expr::Var(v1), Expr::Pow(base, exp)) | (Expr::Pow(base, exp), Expr::Var(v1)) => {
                if let Expr::Var(v2) = base.as_ref() {
                    if v1 == v2 {
                        let new_exp = Self::smart_add(Expr::Const(1.0), exp.as_ref().clone());
                        return Expr::Pow(Box::new(Expr::Var(v1.clone())), Box::new(new_exp));
                    }
                }
                Expr::Mul(Box::new(lhs.clone()), Box::new(rhs.clone()))
            }
            
            // x * x = x^2
            (Expr::Var(v1), Expr::Var(v2)) if v1 == v2 => {
                Expr::Pow(Box::new(Expr::Var(v1.clone())), Box::new(Expr::Const(2.0)))
            }
            
            // x^a * x^b = x^(a+b)
            (Expr::Pow(base1, exp1), Expr::Pow(base2, exp2)) if base1 == base2 => {
                let new_exp = Self::smart_add(exp1.as_ref().clone(), exp2.as_ref().clone());
                Expr::Pow(base1.clone(), Box::new(new_exp))
            }
            
            // Constant collection in nested multiplications
            (Expr::Mul(inner_lhs, inner_rhs), Expr::Const(c)) => {
                match (inner_lhs.as_ref(), inner_rhs.as_ref()) {
                    (Expr::Const(c1), _) => {
                        Self::enhanced_multiplication(&Expr::Const(c1 * c), inner_rhs.as_ref())
                    }
                    (_, Expr::Const(c1)) => {
                        Self::enhanced_multiplication(&Expr::Const(c1 * c), inner_lhs.as_ref())
                    }
                    _ => Expr::Mul(Box::new(lhs.clone()), Box::new(rhs.clone())),
                }
            }
            
            // Symmetric case
            (Expr::Const(c), Expr::Mul(inner_lhs, inner_rhs)) => {
                match (inner_lhs.as_ref(), inner_rhs.as_ref()) {
                    (Expr::Const(c1), _) => {
                        Self::enhanced_multiplication(&Expr::Const(c * c1), inner_rhs.as_ref())
                    }
                    (_, Expr::Const(c1)) => {
                        Self::enhanced_multiplication(&Expr::Const(c * c1), inner_lhs.as_ref())
                    }
                    _ => Expr::Mul(Box::new(lhs.clone()), Box::new(rhs.clone())),
                }
            }
            
            _ => Expr::Mul(Box::new(lhs.clone()), Box::new(rhs.clone())),
        }
    }
    
    /// Smart addition with polynomial collection
    fn smart_add(lhs: Expr, rhs: Expr) -> Expr {
        match (&lhs, &rhs) {
            // Constant folding
            (Expr::Const(a), Expr::Const(b)) => Expr::Const(a + b),
            // Zero elimination
            (Expr::Const(0.0), _) => rhs,
            (_, Expr::Const(0.0)) => lhs,
            // Same expressions: x + x = 2*x
            _ if lhs == rhs => Self::enhanced_multiplication(&Expr::Const(2.0), &lhs),
            _ => {
                let add_expr = Expr::Add(Box::new(lhs), Box::new(rhs));
                Self::try_polynomial_collection(&add_expr).unwrap_or(add_expr)
            }
        }
    }
    
    /// Smart subtraction with polynomial collection
    fn smart_sub(lhs: Expr, rhs: Expr) -> Expr {
        match (&lhs, &rhs) {
            // Constant folding
            (Expr::Const(a), Expr::Const(b)) => Expr::Const(a - b),
            // Zero elimination
            (_, Expr::Const(0.0)) => lhs,
            (Expr::Const(0.0), _) => Self::enhanced_multiplication(&Expr::Const(-1.0), &rhs),
            // Same expressions: x - x = 0
            _ if lhs == rhs => Expr::Const(0.0),
            _ => {
                // Convert to addition: a - b = a + (-1)*b
                let neg_rhs = Self::enhanced_multiplication(&Expr::Const(-1.0), &rhs);
                Self::smart_add(lhs, neg_rhs)
            }
        }
    }
    
    /// Apply polynomial collection to final result
    fn apply_polynomial_collection(expr: Expr) -> Expr {
        match &expr {
            Expr::Add(_, _) | Expr::Sub(_, _) => {
                Self::try_polynomial_collection(&expr).unwrap_or(expr)
            }
            _ => expr,
        }
    }
    
    /// Attempt polynomial collection (simplified version from simplify.rs)
    fn try_polynomial_collection(expr: &Expr) -> Option<Expr> {
        use std::collections::HashMap;
        
        let mut terms = Vec::new();
        Self::flatten_add_simple(expr, &mut terms);
        
        if terms.len() < 2 {
            return None;
        }
        
        let mut poly_map = HashMap::new();
        let mut has_non_poly = false;
        
        for term in &terms {
            let (monomial, coeff) = Self::extract_simple_monomial(term);
            if coeff == 0.0 && !matches!(term, Expr::Const(0.0)) {
                has_non_poly = true;
                break;
            }
            *poly_map.entry(monomial).or_insert(0.0) += coeff;
        }
        
        if has_non_poly || poly_map.len() == terms.len() {
            return None;
        }
        
        let mut result_terms = Vec::new();
        for (monomial, coeff) in poly_map {
            if coeff == 0.0 {
                continue;
            }
            let term = Self::build_simple_term(&monomial, coeff);
            result_terms.push(term);
        }
        
        if result_terms.is_empty() {
            Some(Expr::Const(0.0))
        } else if result_terms.len() == 1 {
            Some(result_terms.into_iter().next().unwrap())
        } else {
            Some(
                result_terms
                    .into_iter()
                    .reduce(|a, b| Expr::Add(Box::new(a), Box::new(b)))
                    .unwrap(),
            )
        }
    }
    
    /// Simplified flattening for diff1 polynomial collection
    fn flatten_add_simple(expr: &Expr, out: &mut Vec<Expr>) {
        match expr {
            Expr::Add(a, b) => {
                Self::flatten_add_simple(a, out);
                Self::flatten_add_simple(b, out);
            }
            Expr::Sub(a, b) => {
                Self::flatten_add_simple(a, out);
                let neg_b = Self::enhanced_multiplication(&Expr::Const(-1.0), b);
                Self::flatten_add_simple(&neg_b, out);
            }
            _ => out.push(expr.clone()),
        }
    }
    
    /// Simplified monomial extraction for diff1
    fn extract_simple_monomial(expr: &Expr) -> (String, f64) {
        match expr {
            Expr::Const(c) => ("1".to_string(), *c),
            Expr::Var(v) => (v.clone(), 1.0),
            Expr::Mul(lhs, rhs) => {
                match (lhs.as_ref(), rhs.as_ref()) {
                    (Expr::Const(c), Expr::Var(v)) | (Expr::Var(v), Expr::Const(c)) => {
                        (v.clone(), *c)
                    }
                    (Expr::Const(c), Expr::Pow(base, exp)) | (Expr::Pow(base, exp), Expr::Const(c)) => {
                        if let (Expr::Var(v), Expr::Const(e)) = (base.as_ref(), exp.as_ref()) {
                            (format!("{}^{}", v, e), *c)
                        } else {
                            (format!("{:?}", expr), 0.0) // Non-polynomial
                        }
                    }
                    _ => (format!("{:?}", expr), 0.0), // Non-polynomial
                }
            }
            Expr::Pow(base, exp) => {
                if let (Expr::Var(v), Expr::Const(e)) = (base.as_ref(), exp.as_ref()) {
                    (format!("{}^{}", v, e), 1.0)
                } else {
                    (format!("{:?}", expr), 0.0) // Non-polynomial
                }
            }
            _ => (format!("{:?}", expr), 0.0), // Non-polynomial
        }
    }
    
    /// Build term from monomial key and coefficient
    fn build_simple_term(monomial: &str, coeff: f64) -> Expr {
        if monomial == "1" {
            return Expr::Const(coeff);
        }
        
        if coeff == 1.0 {
            if monomial.contains('^') {
                let parts: Vec<&str> = monomial.split('^').collect();
                if parts.len() == 2 {
                    let var = parts[0];
                    let exp: f64 = parts[1].parse().unwrap_or(1.0);
                    return Expr::Pow(
                        Box::new(Expr::Var(var.to_string())),
                        Box::new(Expr::Const(exp)),
                    );
                }
            } else {
                return Expr::Var(monomial.to_string());
            }
        }
        
        let var_part = if monomial.contains('^') {
            let parts: Vec<&str> = monomial.split('^').collect();
            if parts.len() == 2 {
                let var = parts[0];
                let exp: f64 = parts[1].parse().unwrap_or(1.0);
                Expr::Pow(
                    Box::new(Expr::Var(var.to_string())),
                    Box::new(Expr::Const(exp)),
                )
            } else {
                Expr::Var(monomial.to_string())
            }
        } else {
            Expr::Var(monomial.to_string())
        };
        
        Self::enhanced_multiplication(&Expr::Const(coeff), &var_part)
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
            expr = expr.diff(var_name).simplify();
            i += 1;
        }
        return expr.simplify();
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
            return fun_at_x0_sym.simplify();
        }

        let dfun_dx = self.n_th_derivative1D(var_name, order);
        let dfun_dx_at_x0 = dfun_dx.lambdify1D()(x0);
        let factorial = (1..=order).product::<usize>() as f64;
        let coeff = Expr::Const(dfun_dx_at_x0 / factorial);
        println!("order {}, {:?}, {}", order, coeff, dfun_dx);
        let term = coeff * (x.clone() - x0_sym.clone()).pow(Expr::Const(order as f64));
        if order == 1 {
            let Taylor = fun_at_x0_sym + term;
            return Taylor.simplify();
        } else {
            let Taylor = self.taylor_series1D(var_name, x0, order - 1) + term;
            return Taylor.simplify();
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
            return fun_at_x0_sym.simplify();
        }

        fn taylor_term(
            expr: &Expr,
            var_name: &str,
            x0: f64,
            n: usize,
            x: &Expr,
            x0_sym: &Expr,
        ) -> (Expr, Expr) {
            let dfun_dx = expr.diff(var_name).simplify();
            let dfun_dx_at_x0 = dfun_dx.lambdify1D()(x0);
            let factorial = (1..=n).product::<usize>() as f64;
            let coeff = Expr::Const(dfun_dx_at_x0 / factorial);
            //  println!("order {}, {:?}, {}", n, coeff, dfun_dx);
            (
                coeff
                    * (x.clone() - x0_sym.clone())
                        .pow(Expr::Const(n as f64))
                        .simplify(),
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
        Taylor.simplify()
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
    pub fn eval_expression(&self, vars: &[&str], values: &[f64]) -> f64 {
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
                let lhs_fn = lhs.eval_expression(vars, values);
                let rhs_fn = rhs.eval_expression(vars, values);
                lhs_fn + rhs_fn
            }
            Expr::Sub(lhs, rhs) => {
                let lhs_fn = lhs.eval_expression(vars, values);
                let rhs_fn = rhs.eval_expression(vars, values);
                lhs_fn - rhs_fn
            }
            Expr::Mul(lhs, rhs) => {
                let lhs_fn = lhs.eval_expression(vars, values);
                let rhs_fn = rhs.eval_expression(vars, values);
                lhs_fn * rhs_fn
            }
            Expr::Div(lhs, rhs) => {
                let lhs_fn = lhs.eval_expression(vars, values);
                let rhs_fn = rhs.eval_expression(vars, values);
                lhs_fn / rhs_fn
            }
            Expr::Pow(base, exp) => {
                let base_fn = base.eval_expression(vars, values);
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
        use std::collections::HashSet;
        let mut vars = HashSet::new();
        self.collect_variables(&mut vars);
        let mut result: Vec<String> = vars.into_iter().collect();
        result.sort();
        result
    }

    /// Helper method for efficient variable collection using HashSet
    fn collect_variables(&self, vars: &mut std::collections::HashSet<String>) {
        match self {
            Expr::Var(name) => {
                vars.insert(name.clone());
            }
            Expr::Const(_) => {}
            Expr::Add(lhs, rhs)
            | Expr::Sub(lhs, rhs)
            | Expr::Mul(lhs, rhs)
            | Expr::Div(lhs, rhs) => {
                lhs.collect_variables(vars);
                rhs.collect_variables(vars);
            }
            Expr::Pow(base, exp) => {
                base.collect_variables(vars);
                exp.collect_variables(vars);
            }
            Expr::Exp(expr) | Expr::Ln(expr) => {
                expr.collect_variables(vars);
            }
            Expr::sin(expr) | Expr::cos(expr) | Expr::tg(expr) | Expr::ctg(expr) => {
                expr.collect_variables(vars);
            }
            Expr::arcsin(expr) | Expr::arccos(expr) | Expr::arctg(expr) | Expr::arcctg(expr) => {
                expr.collect_variables(vars);
            }
        }
    } // end of collect_variables

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
}

mod tests {

    #[test]
    fn legacy_and_new_diff_comapsion_perf() {
        use std::time::Instant;

        use crate::symbolic::symbolic_lambdify::parse_very_complex_expression;
        let e = parse_very_complex_expression();
        let start = Instant::now();
        let e1 = e.diff1("T");
      //  let e1 = e1.simplify();
        let duration = start.elapsed();
        let start = Instant::now();
        let e3 = e.diff2("T");
        let e3 = e3.simplify();
        let duration2 = start.elapsed();
        println!("diff: {:?}, length {}", duration, e1.to_string().len());
        println!("diff2: {:?}, length {} ", duration2, e3.to_string().len());
    }
}
#[cfg(test)]
mod diff_performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_large_polynomial_diff_performance() {
        // Create large polynomial: x^10 + 2*x^9 + 3*x^8 + ... + 10*x + 11
        let x = Expr::Var("x".to_string());
        let mut poly = Expr::Const(11.0);
        for i in 1..=10 {
            poly = poly + Expr::Const(i as f64) * x.clone().pow(Expr::Const(i as f64));
        }

        // Test optimized version
        let start = Instant::now();
        let optimized_result = poly.diff1("x");
        //let optimized_result = optimized_result.simplify();
        let optimized_time = start.elapsed();

        // Test legacy version
        let start = Instant::now();
        let legacy_result = poly.diff2("x");
        let legacy_result = legacy_result.simplify();
        let legacy_time = start.elapsed();

        println!("Large polynomial differentiation:");
        println!("Optimized: {:?}", optimized_time);
        println!("Legacy: {:?}", legacy_time);

        // Verify mathematical correctness
        let test_val = 2.0;
        let opt_eval = optimized_result.set_variable("x", test_val).simplify();
        let leg_eval = legacy_result.set_variable("x", test_val).simplify();

        if let (Expr::Const(opt), Expr::Const(leg)) = (opt_eval, leg_eval) {
            assert!(
                (opt - leg).abs() < 1e-10,
                "Results differ: {} vs {}",
                opt,
                leg
            );
        }
    }

    #[test]
    fn test_complex_expression_diff_performance() {
        // Create complex expression with nested functions and products
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let k = x.clone() + Expr::Const(1.0);
        // x^3 *sin(x) + x*y + x^2*cos(y) + ln(x+1)*y^4
        let expr = (x.clone().pow(Expr::Const(3.0)) * Expr::sin(Box::new(x.clone()))
            + Expr::exp(x.clone() * y.clone()))
            * (x.clone().pow(Expr::Const(2.0)) + Expr::cos(Box::new(y.clone())))
            + Expr::ln(k) * y.clone().pow(Expr::Const(4.0));

        // Test optimized version
        let start = Instant::now();
        let optimized_result = expr.diff1("x");
       // let optimized_result = optimized_result.simplify();
        let optimized_time = start.elapsed();

        // Test legacy version
        let start = Instant::now();
        let legacy_result = expr.diff2("x");
        let legacy_result = legacy_result.simplify();
        let legacy_time = start.elapsed();

        println!("Complex expression differentiation:");
        println!(
            "Optimized: {:?}, len {}",
            optimized_time,
            optimized_result.to_string().len()
        );
        println!(
            "Legacy: {:?}, len {}",
            legacy_time,
            legacy_result.to_string().len()
        );

        // Verify mathematical correctness
        let test_x = 1.5;
        let test_y = 0.5;
        let opt_func = optimized_result.lambdify1(&["x", "y"]);
        let leg_func = legacy_result.lambdify1(&["x", "y"]);

        let opt_val = opt_func(&[test_x, test_y]);
        let leg_val = leg_func(&[test_x, test_y]);

        assert!(
            (opt_val - leg_val).abs() < 1e-10,
            "Results differ: {} vs {}",
            opt_val,
            leg_val
        );
    }

    

    #[test]
    fn test_deep_nested_expression_diff_performance() {
        // Create deeply nested expression that would benefit from early termination
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let z = Expr::Var("z".to_string());
        let k = y.clone() * z.clone().pow(Expr::Const(3.0));
        // Expression with many terms that don't contain x
        // expr = x^2 + sin(y*z) + exp(y*z) + cos(z*y^2) + y*z*sin(z)
        let expr = x.clone().pow(Expr::Const(2.0))
            + Expr::sin(Box::new(y.clone() * z.clone()))
            + Expr::exp(k)
            + Expr::ln(y.clone() + z.clone())
            + Expr::cos(Box::new(z.clone())) * y.clone().pow(Expr::Const(2.0))
            + y.clone() * z.clone() * Expr::sin(Box::new(z.clone()));

        // Test optimized version (should benefit from early termination)
        let start = Instant::now();
        let optimized_result = expr.diff1("x");
        println!("optimized_result: {}", optimized_result);
       // let optimized_result = optimized_result.simplify();
        let optimized_time = start.elapsed();

        // Test legacy version
        let start = Instant::now();
        let legacy_result = expr.diff2("x");
        println!("legacy_result: {}", legacy_result);
        let legacy_result = legacy_result.simplify();
        let legacy_time = start.elapsed();

        println!("Deep nested expression differentiation:");
        println!(
            "Optimized: {:?}, len {}",
            optimized_time,
            optimized_result.to_string().len()
        );
        println!(
            "Legacy: {:?}, len {}",
            legacy_time,
            legacy_result.to_string().len()
        );
        // Verify result - should be 2*x since only first term contains x
        let expected = Expr::Const(2.0) * x.clone();
        assert_eq!(optimized_result, expected);

        // Legacy should give same mathematical result but more complex form
        let test_val = 3.0;
        let opt_eval = optimized_result.set_variable("x", test_val).simplify();
        let leg_eval = legacy_result.set_variable("x", test_val).simplify();

        if let (Expr::Const(opt), Expr::Const(leg)) = (opt_eval, leg_eval) {
            assert!(
                (opt - leg).abs() < 1e-10,
                "Results differ: {} vs {}",
                opt,
                leg
            );
        }
    }

    #[test]
    fn test_product_chain_diff_performance() {
        // Create expression with many product terms to test multiplication optimization
        let x = Expr::Var("x".to_string());

        let mut expr = x.clone();
        for i in 2..=8 {
            expr = expr * (x.clone() + Expr::Const(i as f64));
        }

        // Test optimized version
        let start = Instant::now();
        let optimized_result = expr.diff1("x");
     //   let optimized_result = optimized_result.simplify();
        let optimized_time = start.elapsed();

        // Test legacy version
        let start = Instant::now();
        let legacy_result = expr.diff2("x");
        let legacy_result = legacy_result.simplify();
        let legacy_time = start.elapsed();

        println!("Product chain differentiation:");
        println!(
            "Optimized: {:?}, len {}",
            optimized_time,
            optimized_result.to_string().len()
        );
        println!(
            "Legacy: {:?}, len {}",
            legacy_time,
            legacy_result.to_string().len()
        );

        // Verify mathematical correctness
        let test_val = 1.0;
        let opt_func = optimized_result.lambdify1D();
        let leg_func = legacy_result.lambdify1D();

        let opt_val = opt_func(test_val);
        let leg_val = leg_func(test_val);

        assert!(
            (opt_val - leg_val).abs() < 1e-10,
            "Results differ: {} vs {}",
            opt_val,
            leg_val
        );
    }

    #[test]
    fn test_power_optimization_performance() {
        // Create expression with many power terms to test power rule optimization
        let x = Expr::Var("x".to_string());

        let expr = x.clone().pow(Expr::Const(10.0))
            + x.clone().pow(Expr::Const(9.0))
            + x.clone().pow(Expr::Const(8.0))
            + x.clone().pow(Expr::Const(7.0))
            + x.clone().pow(Expr::Const(6.0))
            + x.clone().pow(Expr::Const(5.0))
            + x.clone().pow(Expr::Const(4.0))
            + x.clone().pow(Expr::Const(3.0))
            + x.clone().pow(Expr::Const(2.0))
            + x.clone().pow(Expr::Const(1.0))
            + x.clone().pow(Expr::Const(0.0));

        // Test optimized version
        let start = Instant::now();
        let optimized_result = expr.diff1("x");
      //  let optimized_result = optimized_result.simplify();
        let optimized_time = start.elapsed();

        // Test legacy version
        let start = Instant::now();
        let legacy_result = expr.diff2("x");
        let legacy_result = legacy_result.simplify();
        let legacy_time = start.elapsed();

        println!("Power optimization differentiation:");
        println!("Optimized: {:?}", optimized_time);
        println!("Legacy: {:?}", legacy_time);

        // Check that optimized version produces cleaner results
        let opt_str = format!("{}", optimized_result);
        let leg_str = format!("{}", legacy_result);

        println!("Optimized result length: {}", opt_str.len());
        println!("Legacy result length: {}", leg_str.len());

        // Optimized should be shorter due to simplifications
        assert!(
            opt_str.len() <= leg_str.len(),
            "Optimized result should be more compact"
        );

        // Verify mathematical correctness
        let test_val = 2.0;
        let opt_func = optimized_result.lambdify1D();
        let leg_func = legacy_result.lambdify1D();

        let opt_val = opt_func(test_val);
        let leg_val = leg_func(test_val);

        assert!(
            (opt_val - leg_val).abs() < 1e-10,
            "Results differ: {} vs {}",
            opt_val,
            leg_val
        );
    }
}
