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

use super::pretty_print::LeveledLayout;
use crate::symbolic::symbolic_metadata::TraversalCache;
use std::collections::HashMap;
use std::f64;

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
/// Converts expressions to human-readable mathematical notation with minimal
/// parentheses based on operator precedence rules.
///
/// ## Key Principles for Bracket Rules
///
/// ### Operator Precedence Hierarchy (highest to lowest):
/// - **Variables/Constants (100)**: Never need brackets (`x`, `5.0`)
/// - **Functions (90)**: Already have parentheses (`sin(x)`, `exp(y)`)
/// - **Power (80)**: Right-associative (`x^y^z` = `x^(y^z)`)
/// - **Multiplication/Division (60)**: Left-associative (`x*y/z` = `(x*y)/z`)
/// - **Addition/Subtraction (40)**: Left-associative (`x+y-z` = `(x+y)-z`)
///
/// ### When Brackets Are Added:
/// 1. **Lower precedence in higher precedence context**: `(x + y) * z`
/// 2. **Right-associative operations with equal/lower precedence**: `x - (y - z)`, `x / (y / z)`, `x ^ (y ^ z)`
/// 3. **Division with complex denominators**: `x / (y + z)`
/// 4. **Power with complex base**: `(x + y) ^ 2`
///
/// ### When Brackets Are Omitted:
/// 1. **Variables and constants**: Never need brackets
/// 2. **Functions**: Already have built-in parentheses
/// 3. **Same or higher precedence**: `x * y + z` (no brackets around `x * y`)
/// 4. **Left-associative with higher precedence**: `x + y * z` (no brackets around `y * z`)
///
/// ## Implementation Logic
///
/// The precedence system uses numerical values where higher numbers indicate
/// higher precedence. Two helper methods determine bracket necessity:
///
/// - `needs_brackets_left()`: Checks if left operand needs brackets (simple precedence comparison)
/// - `needs_brackets_right()`: Handles right-associativity for `-`, `/`, `^` operations
///
/// ### Examples of Output:
/// - `x + y * z` instead of `(x + (y * z))`
/// - `(x + y) * z` instead of `((x + y) * z)`
/// - `x - (y - z)` (preserves necessary brackets for right-associative subtraction)
/// - `sin(x) + y` instead of `(sin(x) + y)`
/// - `x ^ y ^ z` becomes `x ^ (y ^ z)` (right-associative power)
impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Expr::Var(name) => write!(f, "{}", name),
            Expr::Const(val) => write!(f, "{}", val),

            Expr::Add(lhs, rhs) => {
                self.fmt_left_operand(f, lhs)?;
                write!(f, " + ")?;
                self.fmt_right_operand(f, rhs)
            }

            Expr::Sub(lhs, rhs) => {
                self.fmt_left_operand(f, lhs)?;
                write!(f, " - ")?;
                self.fmt_right_operand(f, rhs)
            }

            Expr::Mul(lhs, rhs) => {
                self.fmt_left_operand(f, lhs)?;
                write!(f, " * ")?;
                self.fmt_right_operand(f, rhs)
            }

            Expr::Div(lhs, rhs) => {
                self.fmt_left_operand(f, lhs)?;
                write!(f, " / ")?;
                self.fmt_right_operand(f, rhs)
            }

            Expr::Pow(base, exp) => {
                self.fmt_left_operand(f, base)?;
                write!(f, " ^ ")?;
                self.fmt_right_operand(f, exp)
            }

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
    /// Get operator precedence (higher number = higher precedence)
    fn precedence(&self) -> u8 {
        match self {
            Expr::Var(_) | Expr::Const(_) => 100, // Highest - never need brackets
            Expr::Exp(_)
            | Expr::Ln(_)
            | Expr::sin(_)
            | Expr::cos(_)
            | Expr::tg(_)
            | Expr::ctg(_)
            | Expr::arcsin(_)
            | Expr::arccos(_)
            | Expr::arctg(_)
            | Expr::arcctg(_) => 90, // Functions
            Expr::Pow(_, _) => 80,                // Power
            Expr::Mul(_, _) | Expr::Div(_, _) => 60, // Multiplication/Division
            Expr::Add(_, _) | Expr::Sub(_, _) => 40, // Addition/Subtraction
        }
    }

    /// Check if operand needs brackets when used as left operand
    fn needs_brackets_left(&self, operand: &Expr) -> bool {
        operand.precedence() < self.precedence()
    }

    /// Check if operand needs brackets when used as right operand
    fn needs_brackets_right(&self, operand: &Expr) -> bool {
        match self {
            Expr::Sub(_, _) | Expr::Div(_, _) | Expr::Pow(_, _) => {
                operand.precedence() <= self.precedence()
            }
            _ => operand.precedence() < self.precedence(),
        }
    }

    /// Format left operand with conditional brackets
    fn fmt_left_operand(&self, f: &mut fmt::Formatter, operand: &Expr) -> fmt::Result {
        if self.needs_brackets_left(operand) {
            write!(f, "({})", operand)
        } else {
            write!(f, "{}", operand)
        }
    }

    /// Format right operand with conditional brackets
    fn fmt_right_operand(&self, f: &mut fmt::Formatter, operand: &Expr) -> fmt::Result {
        if self.needs_brackets_right(operand) {
            write!(f, "({})", operand)
        } else {
            write!(f, "{}", operand)
        }
    }

    /// Format function with multi-line content horizontally
    fn _format_function_horizontal(func_name: &str, inner_lines: &[String]) -> Vec<String> {
        if inner_lines.len() <= 1 {
            return vec![format!(
                "{}({})",
                func_name,
                inner_lines.get(0).unwrap_or(&String::new())
            )];
        }

        let mut result = Vec::new();

        // First line: function_name( first_line )
        result.push(format!("{}( {} )", func_name, inner_lines[0]));

        // Middle lines: align content properly with the content position, not the parenthesis
        let content_offset = func_name.len() + 2; // "func( " length
        for i in 1..inner_lines.len() {
            result.push(format!("{}{}", " ".repeat(content_offset), inner_lines[i]));
        }

        result
    }

    /// Convert digit to Unicode superscript
    fn to_superscript_digit(digit: u8) -> char {
        match digit {
            0 => '⁰',
            1 => '¹',
            2 => '²',
            3 => '³',
            4 => '⁴',
            5 => '⁵',
            6 => '⁶',
            7 => '⁷',
            8 => '⁸',
            9 => '⁹',
            _ => '?',
        }
    }

    /// Convert single character to Unicode superscript if possible
    fn to_superscript_char(c: char) -> Option<char> {
        match c {
            'a' => Some('ᵃ'),
            'b' => Some('ᵇ'),
            'c' => Some('ᶜ'),
            'd' => Some('ᵈ'),
            'e' => Some('ᵉ'),
            'f' => Some('ᶠ'),
            'g' => Some('ᵍ'),
            'h' => Some('ʰ'),
            'i' => Some('ⁱ'),
            'j' => Some('ʲ'),
            'k' => Some('ᵏ'),
            'l' => Some('ˡ'),
            'm' => Some('ᵐ'),
            'n' => Some('ⁿ'),
            'o' => Some('ᵒ'),
            'p' => Some('ᵖ'),
            'r' => Some('ʳ'),
            's' => Some('ˢ'),
            't' => Some('ᵗ'),
            'u' => Some('ᵘ'),
            'v' => Some('ᵛ'),
            'w' => Some('ʷ'),
            'x' => Some('ˣ'),
            'y' => Some('ʸ'),
            'z' => Some('ᶻ'),
            '+' => Some('⁺'),
            '-' => Some('⁻'),
            '=' => Some('⁼'),
            '(' => Some('⁽'),
            ')' => Some('⁾'),
            _ => None,
        }
    }

    /// Check if expression can be rendered as simple Unicode superscript
    fn can_use_unicode_superscript(&self) -> bool {
        match self {
            Expr::Const(n) => *n >= 0.0 && *n <= 9.0 && n.fract() == 0.0,
            Expr::Var(name) => {
                name.len() == 1 && Self::to_superscript_char(name.chars().next().unwrap()).is_some()
            }
            _ => false,
        }
    }

    /// Convert expression to Unicode superscript string
    fn to_unicode_superscript(&self) -> String {
        match self {
            Expr::Const(n) => {
                let digit = *n as u8;
                Self::to_superscript_digit(digit).to_string()
            }
            Expr::Var(name) => {
                let c = name.chars().next().unwrap();
                Self::to_superscript_char(c).unwrap().to_string()
            }
            _ => format!("^({})", self),
        }
    }

    /// Convert Greek letter names to Unicode symbols
    fn convert_greek_letters(&self) -> Expr {
        match self {
            Expr::Var(name) => {
                let lower_name = name.to_lowercase();

                // Check for Greek letters with or without indices
                for (greek_name, greek_symbol) in [
                    ("alpha", "α"),
                    ("beta", "β"),
                    ("gamma", "γ"),
                    ("delta", "δ"),
                    ("epsilon", "ε"),
                    ("zeta", "ζ"),
                    ("eta", "η"),
                    ("theta", "θ"),
                    ("iota", "ι"),
                    ("kappa", "κ"),
                    ("lambda", "λ"),
                    ("mu", "μ"),
                    ("nu", "ν"),
                    ("xi", "ξ"),
                    ("omicron", "ο"),
                    ("pi", "π"),
                    ("rho", "ρ"),
                    ("sigma", "σ"),
                    ("tau", "τ"),
                    ("upsilon", "υ"),
                    ("phi", "φ"),
                    ("chi", "χ"),
                    ("psi", "ψ"),
                    ("omega", "ω"),
                ] {
                    if lower_name.starts_with(greek_name) {
                        let suffix = &lower_name[greek_name.len()..];
                        return Expr::Var(format!("{}{}", greek_symbol, suffix));
                    }
                }

                self.clone()
            }
            Expr::Add(lhs, rhs) => Expr::Add(
                Box::new(lhs.convert_greek_letters()),
                Box::new(rhs.convert_greek_letters()),
            ),
            Expr::Sub(lhs, rhs) => Expr::Sub(
                Box::new(lhs.convert_greek_letters()),
                Box::new(rhs.convert_greek_letters()),
            ),
            Expr::Mul(lhs, rhs) => Expr::Mul(
                Box::new(lhs.convert_greek_letters()),
                Box::new(rhs.convert_greek_letters()),
            ),
            Expr::Div(lhs, rhs) => Expr::Div(
                Box::new(lhs.convert_greek_letters()),
                Box::new(rhs.convert_greek_letters()),
            ),
            Expr::Pow(base, exp) => Expr::Pow(
                Box::new(base.convert_greek_letters()),
                Box::new(exp.convert_greek_letters()),
            ),
            Expr::Exp(expr) => Expr::Exp(Box::new(expr.convert_greek_letters())),
            Expr::Ln(expr) => Expr::Ln(Box::new(expr.convert_greek_letters())),
            Expr::sin(expr) => Expr::sin(Box::new(expr.convert_greek_letters())),
            Expr::cos(expr) => Expr::cos(Box::new(expr.convert_greek_letters())),
            Expr::tg(expr) => Expr::tg(Box::new(expr.convert_greek_letters())),
            Expr::ctg(expr) => Expr::ctg(Box::new(expr.convert_greek_letters())),
            Expr::arcsin(expr) => Expr::arcsin(Box::new(expr.convert_greek_letters())),
            Expr::arccos(expr) => Expr::arccos(Box::new(expr.convert_greek_letters())),
            Expr::arctg(expr) => Expr::arctg(Box::new(expr.convert_greek_letters())),
            Expr::arcctg(expr) => Expr::arcctg(Box::new(expr.convert_greek_letters())),
            _ => self.clone(),
        }
    }

    /// Pretty print with mathematical formatting
    pub fn pretty_print(&self) -> String {
        self.convert_greek_letters()
            .pretty_print_internal()
            .join("\n")
    }

    /// Calculate the actual visual width of an expression when pretty printed
    #[allow(dead_code)]
    fn visual_width(&self) -> usize {
        match self {
            Expr::Var(name) => name.chars().count(),
            Expr::Const(val) => format!("{}", val).chars().count(),
            Expr::Div(num, den) => {
                let num_width = num.visual_width();
                let den_width = den.visual_width();
                num_width.max(den_width).max(3)
            }
            Expr::Pow(base, exp) => {
                if exp.can_use_unicode_superscript() {
                    base.visual_width() + exp.to_unicode_superscript().chars().count()
                } else {
                    base.visual_width().max(exp.visual_width())
                }
            }
            _ => {
                // For other operations, calculate based on operands
                match self {
                    Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) | Expr::Mul(lhs, rhs) => {
                        lhs.visual_width() + 3 + rhs.visual_width() // operator + spaces
                    }
                    Expr::Exp(_)
                    | Expr::Ln(_)
                    | Expr::sin(_)
                    | Expr::cos(_)
                    | Expr::tg(_)
                    | Expr::ctg(_)
                    | Expr::arcsin(_)
                    | Expr::arccos(_)
                    | Expr::arctg(_)
                    | Expr::arcctg(_) => {
                        // Function name + parentheses + content
                        let func_name = match self {
                            Expr::Exp(_) => "exp",
                            Expr::Ln(_) => "ln",
                            Expr::sin(_) => "sin",
                            Expr::cos(_) => "cos",
                            Expr::tg(_) => "tg",
                            Expr::ctg(_) => "ctg",
                            Expr::arcsin(_) => "arcsin",
                            Expr::arccos(_) => "arccos",
                            Expr::arctg(_) => "arctg",
                            Expr::arcctg(_) => "arcctg",
                            _ => "",
                        };
                        let inner = match self {
                            Expr::Exp(e)
                            | Expr::Ln(e)
                            | Expr::sin(e)
                            | Expr::cos(e)
                            | Expr::tg(e)
                            | Expr::ctg(e)
                            | Expr::arcsin(e)
                            | Expr::arccos(e)
                            | Expr::arctg(e)
                            | Expr::arcctg(e) => e.visual_width(),
                            _ => 0,
                        };
                        func_name.chars().count() + 2 + inner // name + "()" + content
                    }
                    _ => 10, // fallback
                }
            }
        }
    }

    /// Internal pretty print using level-based layout system
    fn pretty_print_internal(&self) -> Vec<String> {
        let layout = self.analyze_levels_with_brackets(0, false);
        layout.render()
    }

    /// Analyze expression with bracket handling and structural awareness
    fn analyze_levels_with_brackets(
        &self,
        current_level: i32,
        force_brackets: bool,
    ) -> LeveledLayout {
        let mut layout = LeveledLayout::new();
        layout.baseline = current_level;

        match self {
            Expr::Var(name) => {
                let structure_id =
                    layout.create_structure_context(vec!["var".to_string()], None, 0);
                layout.add_element_with_context(name.clone(), current_level, structure_id);
            }
            Expr::Const(val) => {
                let structure_id =
                    layout.create_structure_context(vec!["const".to_string()], None, 0);
                layout.add_element_with_context(format!("{}", val), current_level, structure_id);
            }
            Expr::Add(lhs, rhs) => {
                let left_needs_brackets = self.needs_brackets_left(lhs);
                let right_needs_brackets = self.needs_brackets_right(rhs);

                let mut left_layout = lhs.analyze_levels_with_brackets(current_level, false);
                let right_layout = rhs.analyze_levels_with_brackets(current_level, false);

                if left_needs_brackets {
                    left_layout.add_brackets();
                }

                layout = left_layout;
                layout.merge_horizontal(right_layout, "+", current_level, right_needs_brackets);

                if force_brackets {
                    layout.add_brackets();
                }
            }
            Expr::Sub(lhs, rhs) => {
                let left_needs_brackets = self.needs_brackets_left(lhs);
                let right_needs_brackets = self.needs_brackets_right(rhs);

                let mut left_layout = lhs.analyze_levels_with_brackets(current_level, false);
                let right_layout = rhs.analyze_levels_with_brackets(current_level, false);

                if left_needs_brackets {
                    left_layout.add_brackets();
                }

                layout = left_layout;
                layout.merge_horizontal(right_layout, "-", current_level, right_needs_brackets);

                if force_brackets {
                    layout.add_brackets();
                }
            }
            Expr::Mul(lhs, rhs) => {
                let left_needs_brackets = self.needs_brackets_left(lhs);
                let right_needs_brackets = self.needs_brackets_right(rhs);

                let mut left_layout = lhs.analyze_levels_with_brackets(current_level, false);
                let right_layout = rhs.analyze_levels_with_brackets(current_level, false);

                if left_needs_brackets {
                    left_layout.add_brackets();
                }

                layout = left_layout;
                layout.merge_horizontal(right_layout, "*", current_level, right_needs_brackets);

                if force_brackets {
                    layout.add_brackets();
                }
            }
            Expr::Div(num, den) => {
                // Use structure-aware level assignment for nested divisions
                let depth = self.calculate_nesting_depth();
                let level_offset = if depth > 3 { depth as i32 } else { 1 };

                let num_layout =
                    num.analyze_levels_with_brackets(current_level + level_offset, false);
                let den_layout =
                    den.analyze_levels_with_brackets(current_level - level_offset, false);

                layout.merge_vertical(num_layout, den_layout, current_level);
            }
            Expr::Pow(base, exp) => {
                if exp.can_use_unicode_superscript() {
                    let base_needs_brackets = self.needs_brackets_left(base);
                    let mut base_layout = base.analyze_levels_with_brackets(current_level, false);
                    if base_needs_brackets {
                        base_layout.add_brackets();
                    }
                    layout = base_layout;
                    let structure_id =
                        layout.create_structure_context(vec!["superscript".to_string()], None, 0);
                    layout.add_element_with_context(
                        exp.to_unicode_superscript(),
                        current_level,
                        structure_id,
                    );
                } else {
                    let base_needs_brackets = self.needs_brackets_left(base);
                    let mut base_layout = base.analyze_levels_with_brackets(current_level, false);
                    if base_needs_brackets {
                        base_layout.add_brackets();
                    }

                    // Use isolated level system for complex exponents to prevent conflicts
                    let exp_layout = exp.analyze_levels_with_brackets(0, false);

                    layout = base_layout;
                    layout.merge_power_exponent(exp_layout);
                }

                if force_brackets {
                    layout.add_brackets();
                }
            }
            Expr::Exp(expr) => {
                if expr.can_use_unicode_superscript() {
                    let structure_id =
                        layout.create_structure_context(vec!["exp".to_string()], None, 0);
                    layout.add_element_with_context(
                        format!("e{}", expr.to_unicode_superscript()),
                        current_level,
                        structure_id,
                    );
                } else {
                    let mut base_layout = LeveledLayout::new();
                    base_layout.baseline = current_level;
                    let structure_id =
                        base_layout.create_structure_context(vec!["exp_base".to_string()], None, 0);
                    base_layout.add_element_with_context(
                        "e".to_string(),
                        current_level,
                        structure_id,
                    );

                    // Use isolated level system for exponent to prevent conflicts
                    let exp_layout = expr.analyze_levels_with_brackets(0, false);

                    layout = base_layout;
                    layout.merge_power_exponent(exp_layout);
                }
            }
            _ => {
                let func_name = match self {
                    Expr::Ln(_) => "ln",
                    Expr::sin(_) => "sin",
                    Expr::cos(_) => "cos",
                    Expr::tg(_) => "tg",
                    Expr::ctg(_) => "ctg",
                    Expr::arcsin(_) => "arcsin",
                    Expr::arccos(_) => "arccos",
                    Expr::arctg(_) => "arctg",
                    Expr::arcctg(_) => "arcctg",
                    _ => "func",
                };

                let inner_expr = match self {
                    Expr::Ln(e)
                    | Expr::sin(e)
                    | Expr::cos(e)
                    | Expr::tg(e)
                    | Expr::ctg(e)
                    | Expr::arcsin(e)
                    | Expr::arccos(e)
                    | Expr::arctg(e)
                    | Expr::arcctg(e) => e,
                    _ => return layout,
                };

                let inner_layout = inner_expr.analyze_levels_with_brackets(current_level, false);
                let has_multiple_levels = inner_layout.level_heights.len() > 1;

                if has_multiple_levels {
                    layout.merge_function_with_multiline(func_name, inner_layout);
                } else {
                    let structure_id = layout.create_structure_context(
                        vec!["func".to_string(), func_name.to_string()],
                        None,
                        0,
                    );
                    layout.add_element_with_context(
                        format!("{}(", func_name),
                        current_level,
                        structure_id,
                    );
                    layout.merge_inline(inner_layout);
                    layout.add_element_with_context(")".to_string(), current_level, structure_id);
                }
            }
        }

        layout
    }

    /// Calculate nesting depth to determine appropriate level offsets
    fn calculate_nesting_depth(&self) -> usize {
        match self {
            Expr::Var(_) | Expr::Const(_) => 0,
            Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) | Expr::Mul(lhs, rhs) => {
                1 + lhs
                    .calculate_nesting_depth()
                    .max(rhs.calculate_nesting_depth())
            }
            Expr::Div(num, den) => {
                2 + num
                    .calculate_nesting_depth()
                    .max(den.calculate_nesting_depth())
            }
            Expr::Pow(base, exp) => {
                2 + base
                    .calculate_nesting_depth()
                    .max(exp.calculate_nesting_depth())
            }
            Expr::Exp(e)
            | Expr::Ln(e)
            | Expr::sin(e)
            | Expr::cos(e)
            | Expr::tg(e)
            | Expr::ctg(e)
            | Expr::arcsin(e)
            | Expr::arccos(e)
            | Expr::arctg(e)
            | Expr::arcctg(e) => 1 + e.calculate_nesting_depth(),
        }
    }

    /// BASIC FEATURES

    /// Check if operand needs brackets (continued)
    fn _needs_brackets_continued(&self, operand: &Expr) -> bool {
        operand.precedence() < self.precedence()
    }

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
        let mut cache = TraversalCache::new();
        cache.contains_variable(self, var_name)
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
