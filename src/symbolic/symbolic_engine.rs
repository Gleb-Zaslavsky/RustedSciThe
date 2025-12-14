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
                    ("alpha", "α"), ("beta", "β"), ("gamma", "γ"), ("delta", "δ"),
                    ("epsilon", "ε"), ("zeta", "ζ"), ("eta", "η"), ("theta", "θ"),
                    ("iota", "ι"), ("kappa", "κ"), ("lambda", "λ"), ("mu", "μ"),
                    ("nu", "ν"), ("xi", "ξ"), ("omicron", "ο"), ("pi", "π"),
                    ("rho", "ρ"), ("sigma", "σ"), ("tau", "τ"), ("upsilon", "υ"),
                    ("phi", "φ"), ("chi", "χ"), ("psi", "ψ"), ("omega", "ω"),
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
        self.convert_greek_letters().pretty_print_internal().join("\n")
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
}

/// Structural context for tracking expression hierarchy and preventing layout conflicts.
///
/// Each layout element belongs to a structural family to enable conflict detection
/// and resolution during the rendering process. This prevents mathematical elements
/// from overlapping when complex expressions are combined.
#[derive(Debug, Clone)]
struct StructuralContext {
    /// Optional parent structure ID for hierarchical tracking.
    /// Used to determine structural families and resolve layout conflicts.
    parent_id: Option<usize>,
}

/// A single text element in the mathematical expression layout.
///
/// Represents one piece of text (variable, operator, function name, etc.) with its
/// positioning information. Multiple elements at different levels combine to form
/// the complete mathematical expression.
#[derive(Debug, Clone)]
struct LayoutElement {
    /// The actual text content to be rendered (e.g., "x", "+", "sin", "─")
    content: String,

    /// Vertical level where this element should be placed.
    /// - Positive levels: above baseline (exponents, numerators)
    /// - Zero level: baseline (main expression)
    /// - Negative levels: below baseline (denominators, subscripts)
    level: i32,

    /// Horizontal position (column) where this element starts.
    /// Used for proper alignment when combining multiple elements on the same level.
    position: usize,

    /// Unique identifier linking this element to its structural context.
    /// Enables conflict detection and resolution during rendering.
    structure_id: usize,
}

/// Multi-level layout system for mathematical expression rendering.
///
/// Manages the positioning and rendering of mathematical expressions across multiple
/// vertical levels. Handles complex structures like fractions (numerator/denominator),
/// exponents, and nested functions while maintaining proper mathematical formatting.
///
/// ## Layout Strategy
///
/// - **Level-based positioning**: Each part of the expression is assigned a vertical level
/// - **Structure isolation**: Different mathematical constructs maintain separate structural contexts
/// - **Conservative gap management**: Minimal spacing between levels while preventing overlap
/// - **Conflict resolution**: Structural awareness prevents elements from interfering with each other
#[derive(Debug, Clone)]
struct LeveledLayout {
    /// Collection of all text elements that make up the expression.
    /// Elements are positioned across multiple levels and will be rendered
    /// into a multi-line string representation.
    elements: Vec<LayoutElement>,

    /// The reference level (typically 0) representing the main expression line.
    /// Positive levels are above baseline, negative levels are below.
    baseline: i32,

    /// Total horizontal width needed to accommodate all elements.
    /// Used for proper alignment and spacing calculations.
    total_width: usize,

    /// Maps each level to its height requirement (typically 1 for text lines).
    /// Used during rendering to determine which levels actually contain content.
    level_heights: std::collections::HashMap<i32, usize>,

    /// Maps structure IDs to their contexts for hierarchical tracking.
    /// Enables conflict detection and resolution between different mathematical structures.
    contexts: std::collections::HashMap<usize, StructuralContext>,

    /// Counter for generating unique structure IDs.
    /// Ensures each mathematical construct gets a unique identifier.
    next_structure_id: usize,
}

impl LeveledLayout {
    /// Creates a new empty layout system.
    ///
    /// Initializes all collections and sets baseline to 0 (main expression level).
    /// The layout is ready to accept elements and perform mathematical formatting.
    fn new() -> Self {
        Self {
            elements: Vec::new(),
            baseline: 0,
            total_width: 0,
            level_heights: std::collections::HashMap::new(),
            contexts: std::collections::HashMap::new(),
            next_structure_id: 0,
        }
    }

    /// Creates a new structural context for tracking expression hierarchy.
    ///
    /// Each mathematical construct (variable, operator, function, etc.) gets its own
    /// structural context to enable conflict detection and proper rendering isolation.
    ///
    /// # Arguments
    /// * `_path` - Structural path (currently unused but reserved for future enhancements)
    /// * `parent_id` - Optional parent structure for hierarchical relationships
    /// * `_depth` - Nesting depth (currently unused but reserved for future enhancements)
    ///
    /// # Returns
    /// Unique structure ID for the created context
    fn create_structure_context(
        &mut self,
        _path: Vec<String>,
        parent_id: Option<usize>,
        _depth: usize,
    ) -> usize {
        let structure_id = self.next_structure_id;
        self.next_structure_id += 1;

        let context = StructuralContext { parent_id };

        self.contexts.insert(structure_id, context.clone());
        structure_id
    }

    /// Adds a text element to the layout with specified structural context.
    ///
    /// Places the element at the current total width position and updates layout metrics.
    /// The element will be rendered at the specified level with proper structural tracking.
    ///
    /// # Arguments
    /// * `content` - Text content to be rendered
    /// * `level` - Vertical level for positioning (0=baseline, +above, -below)
    /// * `structure_id` - ID linking this element to its structural context
    fn add_element_with_context(&mut self, content: String, level: i32, structure_id: usize) {
        let position = self.total_width;

        self.elements.push(LayoutElement {
            content: content.clone(),
            level,
            position,
            structure_id,
        });
        self.total_width += content.chars().count();
        *self.level_heights.entry(level).or_insert(0) =
            (*self.level_heights.get(&level).unwrap_or(&0)).max(1);
    }

    /// Wraps the entire current layout in parentheses.
    ///
    /// Shifts all existing elements to the right and adds opening/closing brackets
    /// at the baseline level. Used when operator precedence requires bracketing
    /// for mathematical correctness.
    ///
    /// # Layout Changes
    /// - All existing elements are shifted right by 1 position
    /// - Opening bracket "(" is inserted at position 0
    /// - Closing bracket ")" is added at the end
    /// - Total width increases by 2
    fn add_brackets(&mut self) {
        let bracket_structure_id =
            self.create_structure_context(vec!["bracket".to_string()], None, 0);

        // Shift all existing elements to make room for opening bracket
        for element in &mut self.elements {
            element.position += 1;
        }

        // Add opening bracket at the beginning
        self.elements.insert(
            0,
            LayoutElement {
                content: "(".to_string(),
                level: self.baseline,
                position: 0,
                structure_id: bracket_structure_id,
            },
        );

        // Add closing bracket at the end
        self.elements.push(LayoutElement {
            content: ")".to_string(),
            level: self.baseline,
            position: self.total_width + 1,
            structure_id: bracket_structure_id,
        });

        self.total_width += 2;
    }

    /// Merges another layout horizontally with an operator between them.
    ///
    /// Combines two mathematical expressions side-by-side with an operator (like +, -, *, /)
    /// in between. Handles bracket insertion for the right operand when required by
    /// operator precedence rules.
    ///
    /// # Arguments
    /// * `other` - The right-hand layout to merge
    /// * `operator` - The operator string to place between layouts (e.g., "+", "*")
    /// * `op_level` - Vertical level where the operator should be placed
    /// * `right_needs_brackets` - Whether to wrap the right operand in brackets
    ///
    /// # Layout Process
    /// 1. Optionally add brackets to right operand
    /// 2. Add operator with spacing at current width
    /// 3. Merge structural contexts from both layouts
    /// 4. Offset and add all elements from right layout
    /// 5. Update total width and level heights
    fn merge_horizontal(
        &mut self,
        mut other: LeveledLayout,
        operator: &str,
        op_level: i32,
        right_needs_brackets: bool,
    ) {
        if right_needs_brackets {
            other.add_brackets();
        }

        let current_width = self.total_width;
        let op_width = operator.len() + 2; // operator + spaces

        // Create operator structure context
        let op_structure_id = self.create_structure_context(
            vec!["operator".to_string(), operator.to_string()],
            None,
            0,
        );

        // Add operator with spacing
        self.elements.push(LayoutElement {
            content: format!(" {} ", operator),
            level: op_level,
            position: current_width,
            structure_id: op_structure_id,
        });

        // Merge contexts from other layout
        for (id, context) in other.contexts {
            self.contexts.insert(id, context);
        }

        // Add other elements with proper position offset
        for mut element in other.elements {
            element.position += current_width + op_width;
            self.elements.push(element);
        }

        self.total_width += op_width + other.total_width;

        // Merge level heights
        for (level, height) in other.level_heights {
            *self.level_heights.entry(level).or_insert(0) =
                *self.level_heights.get(&level).unwrap_or(&0).max(&height);
        }
    }

    /// Merges two layouts vertically to create a fraction (numerator/denominator).
    ///
    /// Creates a mathematical fraction by placing the numerator above a division line
    /// and the denominator below. Both numerator and denominator are centered relative
    /// to the division line for proper mathematical formatting.
    ///
    /// # Arguments
    /// * `num_layout` - Layout for the numerator (top part)
    /// * `den_layout` - Layout for the denominator (bottom part)
    /// * `division_level` - Level where the division line should be placed (typically baseline)
    ///
    /// # Layout Structure
    /// ```text
    ///   numerator     <- above division_level
    /// ─────────────   <- at division_level
    ///  denominator    <- below division_level
    /// ```
    ///
    /// # Centering Logic
    /// - Division line width = max(numerator_width, denominator_width, 3)
    /// - Numerator and denominator are centered relative to division line
    /// - All elements maintain their original structural contexts
    fn merge_vertical(
        &mut self,
        num_layout: LeveledLayout,
        den_layout: LeveledLayout,
        division_level: i32,
    ) {
        let max_width = num_layout.total_width.max(den_layout.total_width).max(3);

        // Create division structure context
        let div_structure_id = self.create_structure_context(vec!["division".to_string()], None, 1);

        self.elements.clear();

        // Merge contexts from both layouts
        for (id, context) in num_layout.contexts {
            self.contexts.insert(id, context);
        }
        for (id, context) in den_layout.contexts {
            self.contexts.insert(id, context);
        }

        // Calculate centering offsets for numerator and denominator
        let num_offset = if num_layout.total_width < max_width {
            (max_width - num_layout.total_width) / 2
        } else {
            0
        };

        let den_offset = if den_layout.total_width < max_width {
            (max_width - den_layout.total_width) / 2
        } else {
            0
        };

        // Add numerator elements with centering and structure isolation
        for mut element in num_layout.elements {
            element.position = num_offset + element.position;
            // Preserve original structure context to prevent interference
            self.elements.push(element);
        }

        // Add division line with its own structure
        self.elements.push(LayoutElement {
            content: "─".repeat(max_width),
            level: division_level,
            position: 0,
            structure_id: div_structure_id,
        });

        // Add denominator elements with centering and structure isolation
        for mut element in den_layout.elements {
            element.position = den_offset + element.position;
            // Preserve original structure context to prevent interference
            self.elements.push(element);
        }

        // Merge level heights properly
        self.level_heights.clear();

        // Add numerator levels as-is (they should already be positioned correctly)
        for (&level, &height) in &num_layout.level_heights {
            *self.level_heights.entry(level).or_insert(0) = height;
        }

        // Add denominator levels as-is (they should already be positioned correctly)
        for (&level, &height) in &den_layout.level_heights {
            *self.level_heights.entry(level).or_insert(0) = height;
        }

        // Add division line level
        *self.level_heights.entry(division_level).or_insert(0) = 1;

        self.total_width = max_width;
        self.baseline = division_level;
    }

    /// Merges an exponent layout directly (unused helper method).
    ///
    /// This is a simple merge that adds exponent elements without level adjustment.
    /// Currently unused in favor of the more sophisticated `merge_power_exponent` method.
    ///
    /// # Arguments
    /// * `exp_layout` - The exponent layout to merge
    #[allow(dead_code)]
    fn _merge_exponent(&mut self, exp_layout: LeveledLayout) {
        // Add exponent elements
        self.elements.extend(exp_layout.elements);

        // Merge level heights
        for (level, height) in exp_layout.level_heights {
            *self.level_heights.entry(level).or_insert(0) =
                *self.level_heights.get(&level).unwrap_or(&0).max(&height);
        }

        self.total_width += exp_layout.total_width;
    }

    /// Merges an exponent layout as a superscript to the current base.
    ///
    /// Positions the exponent above the baseline and to the right of the base expression.
    /// Handles level conflicts by adjusting exponent levels to ensure they appear above
    /// the base without interfering with existing layout elements.
    ///
    /// # Arguments
    /// * `exp_layout` - Layout containing the exponent expression
    ///
    /// # Layout Process
    /// 1. Calculate base width and baseline for positioning
    /// 2. Merge structural contexts from exponent
    /// 3. Find minimum level in exponent for offset calculation
    /// 4. Adjust all exponent levels to appear above base
    /// 5. Position exponent elements to the right of base
    /// 6. Update total width and level heights
    ///
    /// # Example
    /// ```text
    /// Before: "x"     Exponent: "2"
    /// After:  "x²"    (if Unicode) or "x" with "2" at higher level
    /// ```
    fn merge_power_exponent(&mut self, exp_layout: LeveledLayout) {
        let base_width = self.total_width;
        let base_baseline = self.baseline;

        // Create power structure context
        let _power_structure_id = self.create_structure_context(
            vec!["power".to_string(), "exponent".to_string()],
            None,
            2,
        );

        // Merge contexts from exponent layout
        for (id, context) in exp_layout.contexts {
            self.contexts.insert(id, context);
        }

        // Find min level in exponent to calculate proper offset
        let exp_min_level = *exp_layout.level_heights.keys().min().unwrap_or(&0);

        // Calculate offset to move all exponent levels above the base
        // Ensures exponent appears as superscript without level conflicts
        let level_offset = (base_baseline + 1) - exp_min_level;

        // Add exponent elements with level adjustment and structure isolation
        for mut element in exp_layout.elements {
            element.level += level_offset;
            element.position += base_width;
            // Preserve original structure context but update level
            self.elements.push(element);
        }

        // Merge level heights with offset
        for (level, height) in exp_layout.level_heights {
            *self.level_heights.entry(level + level_offset).or_insert(0) = height;
        }

        // Update total width to accommodate both base and exponent
        self.total_width = base_width + exp_layout.total_width;
    }

    /// Merges another layout inline (horizontally) without any operator or spacing.
    ///
    /// Directly concatenates the other layout to the right of the current layout.
    /// Used for combining function arguments, parenthetical content, or other
    /// expressions that should appear immediately adjacent.
    ///
    /// # Arguments
    /// * `other` - Layout to merge inline
    ///
    /// # Layout Process
    /// 1. Merge structural contexts
    /// 2. Offset all elements by current width
    /// 3. Add all elements to current layout
    /// 4. Merge level heights (taking maximum for each level)
    /// 5. Update total width
    fn merge_inline(&mut self, other: LeveledLayout) {
        let current_width = self.total_width;

        // Merge contexts from other layout
        for (id, context) in other.contexts {
            self.contexts.insert(id, context);
        }

        for mut element in other.elements {
            element.position += current_width;
            self.elements.push(element);
        }

        for (level, height) in other.level_heights {
            *self.level_heights.entry(level).or_insert(0) =
                *self.level_heights.get(&level).unwrap_or(&0).max(&height);
        }

        self.total_width += other.total_width;
    }

    /// Merges a function with potentially multi-line content inside parentheses.
    ///
    /// Handles mathematical functions like sin(x), ln(complex_expression), etc.
    /// Automatically detects whether the inner content spans multiple levels and
    /// formats accordingly - simple inline for single-level content, or structured
    /// multi-line layout for complex expressions.
    ///
    /// # Arguments
    /// * `func_name` - Name of the function (e.g., "sin", "ln", "exp")
    /// * `inner_layout` - Layout of the function's argument/content
    ///
    /// # Layout Strategies
    ///
    /// ## Single-line content:
    /// ```text
    /// sin(x + y)  <- All on baseline
    /// ```
    ///
    /// ## Multi-line content:
    /// ```text
    /// sin( x + y )  <- Function name and brackets at baseline
    ///      ───      <- Inner content preserves its level structure
    ///       z
    /// ```
    ///
    /// # Implementation Details
    /// - Detects multi-line by comparing min/max levels in inner layout
    /// - Single-line: Concatenates everything into one element
    /// - Multi-line: Preserves inner structure with proper offset positioning
    /// - Maintains structural isolation to prevent layout conflicts
    fn merge_function_with_multiline(&mut self, func_name: &str, inner_layout: LeveledLayout) {
        // Create function structure context
        let func_structure_id = self.create_structure_context(
            vec!["function".to_string(), func_name.to_string()],
            None,
            1,
        );

        // For multi-line content inside functions, we need special handling
        let min_level = *inner_layout.level_heights.keys().min().unwrap_or(&0);
        let max_level = *inner_layout.level_heights.keys().max().unwrap_or(&0);

        // Merge contexts from inner layout
        for (id, context) in inner_layout.contexts {
            self.contexts.insert(id, context);
        }

        // Check if inner layout has multiple levels (is truly multi-line)
        if min_level == max_level {
            // Single line - use simple inline approach
            let content = inner_layout
                .elements
                .iter()
                .map(|e| e.content.clone())
                .collect::<Vec<_>>()
                .join("");
            let func_with_content = format!("{}({})", func_name, content);

            self.elements.push(LayoutElement {
                content: func_with_content.clone(),
                level: self.baseline,
                position: self.total_width,
                structure_id: func_structure_id,
            });

            self.total_width += func_with_content.chars().count();
            self.level_heights.extend(inner_layout.level_heights);
            return;
        }

        // Multi-line case: preserve function structure with isolation
        let current_pos = self.total_width;

        // Add function name and opening parenthesis at baseline
        let opening = format!("{}(", func_name);
        self.elements.push(LayoutElement {
            content: opening.clone(),
            level: self.baseline,
            position: current_pos,
            structure_id: func_structure_id,
        });

        // Add inner elements with proper offset and structure isolation
        let func_offset = opening.chars().count();
        for mut element in inner_layout.elements {
            element.position = current_pos + func_offset + element.position;
            // Preserve original structure context to prevent interference
            self.elements.push(element);
        }

        // Add closing parenthesis at baseline
        let inner_width = inner_layout.total_width;
        self.elements.push(LayoutElement {
            content: ")".to_string(),
            level: self.baseline,
            position: current_pos + func_offset + inner_width,
            structure_id: func_structure_id,
        });

        // Update layout properties
        self.level_heights.extend(inner_layout.level_heights);
        self.total_width = current_pos + func_offset + inner_width + 1;
    }

    /// Renders the layout into a vector of strings representing the final mathematical expression.
    ///
    /// This is the core rendering method that converts the multi-level layout structure
    /// into human-readable mathematical notation. It handles proper alignment, spacing,
    /// and conflict resolution between different structural elements.
    ///
    /// # Returns
    /// Vector of strings where each string represents one line of the mathematical expression
    ///
    /// # Rendering Process
    ///
    /// ## 1. Level Collection and Sorting
    /// - Identifies all levels that contain actual content (non-empty elements)
    /// - Sorts levels from top to bottom (descending order: +2, +1, 0, -1, -2)
    ///
    /// ## 2. Element Positioning
    /// - Groups elements by structural families to prevent conflicts
    /// - Sorts elements within each level by structure family, then by position
    /// - Creates character arrays for each line with proper width
    ///
    /// ## 3. Conflict Resolution
    /// - Detects when multiple elements try to occupy the same position
    /// - Uses structural compatibility checking to resolve conflicts
    /// - Skips conflicting characters when structures are incompatible
    ///
    /// ## 4. Gap Management (Conservative Approach)
    /// - Only adds empty lines between levels with gaps > 2
    /// - Ensures tight rendering for fractions (numerator/line/denominator)
    /// - Preserves spacing for complex nested expressions
    ///
    /// # Example Output
    /// ```text
    /// For expression: x² + sin(y) / e^x + 1
    ///
    ///     2        x
    /// x  + sin(y) / e  + 1
    /// ```
    ///
    /// # Gap Reduction Logic
    /// - gap ≤ 2: No empty lines (tight mathematical formatting)
    /// - gap > 2: One empty line (prevents visual confusion)
    fn render(&self) -> Vec<String> {
        if self.elements.is_empty() {
            return vec![String::new()];
        }

        let _min_level = *self.level_heights.keys().min().unwrap_or(&0);
        let _max_level = *self.level_heights.keys().max().unwrap_or(&0);

        let mut lines = Vec::new();

        // Calculate the maximum width needed for proper alignment
        let max_width = self
            .elements
            .iter()
            .map(|e| e.position + e.content.chars().count())
            .max()
            .unwrap_or(0);

        // Group elements by structural families to prevent interference
        let _structural_families = self.group_by_structural_families();

        // Collect all levels that actually have content
        let mut levels_with_content: Vec<i32> = self
            .level_heights
            .keys()
            .filter(|&&level| {
                self.elements
                    .iter()
                    .any(|e| e.level == level && !e.content.trim().is_empty())
            })
            .cloned()
            .collect();
        levels_with_content.sort_by(|a, b| b.cmp(a)); // Sort descending (top to bottom)

        // Render only levels that have content, with conservative gap reduction
        for (i, level) in levels_with_content.iter().enumerate() {
            // Collect elements at this level, sorted by position and structure
            let mut level_elements: Vec<&LayoutElement> = self
                .elements
                .iter()
                .filter(|e| e.level == *level && !e.content.trim().is_empty())
                .collect();

            // Sort by structure family first, then by position to prevent conflicts
            level_elements.sort_by(|a, b| {
                let family_a = self.get_structure_family(a.structure_id);
                let family_b = self.get_structure_family(b.structure_id);
                family_a.cmp(&family_b).then(a.position.cmp(&b.position))
            });

            if level_elements.is_empty() {
                continue;
            }

            // Build line with structure-aware positioning
            let mut line_chars: Vec<char> = " ".repeat(max_width).chars().collect();

            for element in level_elements {
                let content = &element.content;

                if content.is_empty() {
                    continue;
                }

                let start_pos = element.position;
                let content_chars: Vec<char> = content.chars().collect();

                // Safely place each character with conflict detection
                for (i, &ch) in content_chars.iter().enumerate() {
                    let pos = start_pos + i;
                    if pos < line_chars.len() {
                        // Check for conflicts with different structural families
                        if line_chars[pos] != ' '
                            && !self.are_compatible_structures(element.structure_id, pos)
                        {
                            // Handle conflict by adjusting position or skipping
                            continue;
                        }
                        line_chars[pos] = ch;
                    }
                }
            }

            let line_str: String = line_chars.into_iter().collect();
            lines.push(line_str.trim_end().to_string());

            // DIVISION-AWARE GAP MANAGEMENT: Tight spacing for divisions, conservative for others
            if i < levels_with_content.len() - 1 {
                let current_level = *level;
                let next_level = levels_with_content[i + 1];
                let gap = (current_level - next_level).abs();

                // Check if we're in a division context by looking for division lines
                let has_division_line = self.elements.iter().any(|e| {
                    e.content.contains('─')
                        && (e.level == current_level
                            || e.level == next_level
                            || (e.level > next_level && e.level < current_level))
                });

                // For divisions: no gaps (tight formatting)
                // For other expressions: gaps only when > 2
                if !has_division_line && gap > 2 {
                    lines.push(String::new());
                }
            }
        }

        // Remove trailing empty lines
        while lines.last().map_or(false, |line| line.trim().is_empty()) {
            lines.pop();
        }

        if lines.is_empty() {
            vec![String::new()]
        } else {
            lines
        }
    }

    /// Groups layout elements by their structural families for conflict resolution.
    ///
    /// Elements belonging to the same structural family (sharing a common root structure)
    /// are grouped together. This enables the rendering system to detect and resolve
    /// conflicts between different mathematical constructs.
    ///
    /// # Returns
    /// HashMap mapping family root IDs to vectors of structure IDs in that family
    ///
    /// # Usage
    /// Used during rendering to ensure elements from different structural families
    /// don't interfere with each other's positioning and layout.
    fn group_by_structural_families(&self) -> std::collections::HashMap<usize, Vec<usize>> {
        let mut families = std::collections::HashMap::new();

        for element in &self.elements {
            let family_id = self.get_structure_family(element.structure_id);
            families
                .entry(family_id)
                .or_insert_with(Vec::new)
                .push(element.structure_id);
        }

        families
    }

    /// Finds the root structure ID for a given structure by traversing parent relationships.
    ///
    /// Walks up the structural hierarchy to find the topmost parent structure.
    /// This identifies which "family" a structure belongs to, enabling conflict
    /// detection between different mathematical constructs.
    ///
    /// # Arguments
    /// * `structure_id` - The structure ID to find the family root for
    ///
    /// # Returns
    /// The root structure ID of the family
    ///
    /// # Example
    /// For a complex expression like "sin(x/y)", the division structure and
    /// the sin function structure would have different family roots, preventing
    /// layout conflicts between them.
    fn get_structure_family(&self, structure_id: usize) -> usize {
        // Find the root structure ID by traversing parent relationships
        let mut current_id = structure_id;
        while let Some(context) = self.contexts.get(&current_id) {
            if let Some(parent_id) = context.parent_id {
                current_id = parent_id;
            } else {
                break;
            }
        }
        current_id
    }

    /// Checks if two structures can coexist at the same position without conflict.
    ///
    /// Currently implements a permissive policy allowing all structures to coexist.
    /// This method is designed to be enhanced with more sophisticated conflict
    /// detection logic as needed.
    ///
    /// # Arguments
    /// * `_structure_id` - ID of the structure being placed (currently unused)
    /// * `_position` - Position where the structure is being placed (currently unused)
    ///
    /// # Returns
    /// `true` if structures are compatible, `false` if they conflict
    ///
    /// # Future Enhancements
    /// Could be extended to:
    /// - Detect overlapping mathematical constructs
    /// - Prevent division lines from interfering with exponents
    /// - Resolve bracket placement conflicts
    /// - Handle complex nested expression conflicts
    fn are_compatible_structures(&self, _structure_id: usize, _position: usize) -> bool {
        // For now, allow all structures to coexist
        // This can be enhanced with more sophisticated conflict detection
        true
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
