//! Recursive-descent parser for the simplified symbolic language.
//!
//! The parser is intentionally small and feeds directly into packed [`Atom`] builders.
//! It understands only the reduced language needed by the standalone crate: identifiers,
//! function calls, rational numbers, unary operators, and the standard arithmetic
//! precedence chain. The parser does not build a separate AST; instead it constructs
//! packed atoms immediately and relies on normalization to canonicalize the result.
//!
//! Compared with the first standalone parser, this version carries line/column diagnostics
//! and accepts scientific notation in numeric literals. Decimal and scientific-notation
//! inputs are still converted into exact rational coefficients whenever they fit in the
//! fixed-size numeric model.

use super::{
    atom::{Atom, AtomView, DefaultNamespace},
    coefficient::Coefficient,
    state::Symbol,
};

/// Error type returned by the simplified parser.
pub type ParseError = String;

/// Human-readable source position used in parser diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position {
    /// One-based line number.
    pub line: usize,
    /// One-based column number in Unicode scalar units.
    pub column: usize,
}

/// Parse an expression using the crate name as the default namespace.
pub fn parse(input: &str) -> Result<Atom, ParseError> {
    parse_with_default_namespace(DefaultNamespace {
        namespace: env!("CARGO_CRATE_NAME").into(),
        data: input,
        file: "".into(),
        line: 0,
    })
}

/// Parse an expression with explicit namespace metadata.
pub fn parse_with_default_namespace(input: DefaultNamespace<'_>) -> Result<Atom, ParseError> {
    let mut parser = Parser {
        input: input.data,
        pos: 0,
        namespace: input,
    };
    let expr = parser.parse_expression()?;
    parser.skip_ws();
    if let Some(found) = parser.peek() {
        return Err(parser.error_here(format!("unexpected '{}' after complete expression", found)));
    }
    let mut normalized = Atom::new();
    expr.as_view().normalize(&mut normalized);
    Ok(normalized)
}

/// Streaming parser state over a borrowed input string.
struct Parser<'a> {
    /// Full source buffer being parsed.
    input: &'a str,
    /// Current byte position inside [`Self::input`].
    pos: usize,
    /// Default namespace metadata attached to bare identifiers.
    namespace: DefaultNamespace<'a>,
}

impl<'a> Parser<'a> {
    /// Peek the next Unicode scalar without consuming it.
    fn peek(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    /// Consume the next Unicode scalar and advance the cursor.
    fn next(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    /// Compute the current one-based line and column.
    fn position(&self) -> Position {
        let mut line = 1usize;
        let mut column = 1usize;
        for c in self.input[..self.pos].chars() {
            if c == '\n' {
                line += 1;
                column = 1;
            } else {
                column += 1;
            }
        }
        Position { line, column }
    }

    /// Build a parser error at the current cursor position.
    fn error_here(&self, message: impl Into<String>) -> ParseError {
        let pos = self.position();
        format!(
            "{} at line {} column {}",
            message.into(),
            pos.line,
            pos.column
        )
    }

    /// Skip whitespace between tokens.
    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(c) if c.is_whitespace()) {
            self.next();
        }
    }

    /// Consume a specific token after optional whitespace.
    fn consume(&mut self, expected: char) -> bool {
        self.skip_ws();
        if self.peek() == Some(expected) {
            self.next();
            true
        } else {
            false
        }
    }

    /// Parse a full expression from the current cursor.
    fn parse_expression(&mut self) -> Result<Atom, ParseError> {
        self.parse_add_sub()
    }

    /// Parse left-associative addition and subtraction.
    fn parse_add_sub(&mut self) -> Result<Atom, ParseError> {
        let mut terms = vec![self.parse_mul_div()?];
        loop {
            self.skip_ws();
            if self.consume('+') {
                terms.push(self.parse_mul_div()?);
            } else if self.consume('-') {
                let right = self.parse_mul_div()?;
                terms.push(negate_raw(right.as_view()));
            } else {
                return Ok(build_add_raw(terms));
            }
        }
    }

    /// Parse left-associative multiplication and division.
    fn parse_mul_div(&mut self) -> Result<Atom, ParseError> {
        let mut factors = vec![self.parse_power()?];
        loop {
            self.skip_ws();
            if self.consume('*') {
                factors.push(self.parse_power()?);
            } else if self.consume('/') {
                let right = self.parse_power()?;
                factors.push(reciprocal_raw(right.as_view()));
            } else if self.next_starts_implicit_factor(factors.last().unwrap().as_view()) {
                factors.push(self.parse_power()?);
            } else {
                return Ok(build_mul_raw(factors));
            }
        }
    }

    /// Parse right-associative exponentiation.
    fn parse_power(&mut self) -> Result<Atom, ParseError> {
        let left = self.parse_unary()?;
        if self.consume('^') {
            let right = self.parse_power()?;
            Ok(build_pow_raw(left.as_view(), right.as_view()))
        } else {
            Ok(left)
        }
    }

    /// Parse unary plus and minus.
    fn parse_unary(&mut self) -> Result<Atom, ParseError> {
        self.skip_ws();
        if self.consume('+') {
            self.parse_unary()
        } else if self.consume('-') {
            let inner = self.parse_unary()?;
            Ok(negate_raw(inner.as_view()))
        } else {
            self.parse_primary()
        }
    }

    /// Parse a parenthesized expression, number, variable, or function call.
    fn parse_primary(&mut self) -> Result<Atom, ParseError> {
        self.skip_ws();
        match self.peek() {
            Some('(') => {
                self.next();
                let expr = self.parse_expression()?;
                if !self.consume(')') {
                    return Err(self.error_here("expected ')' to close parenthesized expression"));
                }
                Ok(expr)
            }
            Some(c) if c.is_ascii_digit() || c == '.' => self.parse_number().map(Atom::new_num),
            Some(c) if is_ident_start(c) => self.parse_identifier_or_call(),
            Some(c) => {
                Err(self.error_here(format!("unexpected '{}' while parsing an expression", c)))
            }
            None => Err(self.error_here("unexpected end of input while parsing an expression")),
        }
    }

    /// Parse either a bare variable name or a function call with comma-separated arguments.
    fn parse_identifier_or_call(&mut self) -> Result<Atom, ParseError> {
        let name = self.parse_identifier()?;
        let symbol = Symbol::new(self.namespace.attach_namespace(&name));
        self.skip_ws();
        if self.consume('(') {
            let mut function = Atom::new();
            let builder = function.to_fun(symbol);
            self.skip_ws();
            if !self.consume(')') {
                loop {
                    let arg = self.parse_expression()?;
                    builder.add_arg(arg.as_view());
                    self.skip_ws();
                    if self.consume(')') {
                        break;
                    }
                    if !self.consume(',') {
                        return Err(self.error_here(format!(
                            "expected ',' or ')' after argument to function '{}'",
                            symbol.get_stripped_name()
                        )));
                    }
                }
            }
            Ok(function)
        } else {
            Ok(Atom::new_var(symbol))
        }
    }

    /// Parse an identifier, including namespace separators.
    fn parse_identifier(&mut self) -> Result<String, ParseError> {
        self.skip_ws();
        let mut out = String::new();
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == '_' || c == ':' {
                out.push(c);
                self.next();
            } else {
                break;
            }
        }
        if out.is_empty() || !is_ident_start(out.chars().next().unwrap()) {
            return Err(self.error_here("expected identifier"));
        }
        Ok(out)
    }

    /// Parse an integer, decimal, or scientific-notation literal into an exact rational coefficient.
    fn parse_number(&mut self) -> Result<Coefficient, ParseError> {
        self.skip_ws();
        let start = self.pos;
        let mut text = String::new();
        let mut seen_dot = false;
        let mut seen_exp = false;

        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                text.push(c);
                self.next();
            } else if c == '.' && !seen_dot && !seen_exp {
                seen_dot = true;
                text.push(c);
                self.next();
            } else if (c == 'e' || c == 'E') && !seen_exp {
                seen_exp = true;
                text.push(c);
                self.next();
                if let Some(sign @ ('+' | '-')) = self.peek() {
                    text.push(sign);
                    self.next();
                }
            } else {
                break;
            }
        }

        if text.is_empty()
            || text == "."
            || text.ends_with('e')
            || text.ends_with('E')
            || text.ends_with("e+")
            || text.ends_with("e-")
            || text.ends_with("E+")
            || text.ends_with("E-")
        {
            self.pos = start;
            return Err(self.error_here("expected valid number literal"));
        }

        parse_exact_number(&text).ok_or_else(|| {
            self.error_here(format!("invalid or out-of-range number literal '{}'", text))
        })
    }

    /// Return whether the next token can start an implicit multiplication factor.
    fn next_starts_implicit_factor(&self, previous: AtomView<'_>) -> bool {
        match self.peek() {
            Some('(') => true,
            Some(c) if is_ident_start(c) => true,
            Some(c) if c.is_ascii_digit() || c == '.' => !matches!(previous, AtomView::Num(_)),
            _ => false,
        }
    }
}

/// Return whether `c` can start an identifier.
fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

/// Convert a parsed textual number into an exact fixed-size rational coefficient.
fn parse_exact_number(text: &str) -> Option<Coefficient> {
    let (mantissa, exponent) = match text.find(['e', 'E']) {
        Some(idx) => {
            let mantissa = &text[..idx];
            let exponent = text[idx + 1..].parse::<i32>().ok()?;
            (mantissa, exponent)
        }
        None => (text, 0),
    };

    let (digits, fractional_digits) = match mantissa.split_once('.') {
        Some((int_part, frac_part)) => {
            (format!("{}{}", int_part, frac_part), frac_part.len() as i32)
        }
        None => (mantissa.to_string(), 0),
    };

    if digits.is_empty() {
        return None;
    }

    let raw_num = digits.parse::<i128>().ok()?;
    let net_exp = exponent - fractional_digits;

    let (num, den) = if net_exp >= 0 {
        let scale = checked_pow10(net_exp as u32)?;
        (raw_num.checked_mul(scale)?, 1_i128)
    } else {
        (raw_num, checked_pow10((-net_exp) as u32)?)
    };

    let num_i64 = i64::try_from(num).ok()?;
    let den_i64 = i64::try_from(den).ok()?;
    Some(Coefficient::reduce(num_i64, den_i64))
}

/// Compute `10^exp` with overflow checking.
fn checked_pow10(exp: u32) -> Option<i128> {
    let mut value = 1_i128;
    for _ in 0..exp {
        value = value.checked_mul(10)?;
    }
    Some(value)
}

#[inline]
fn build_add_raw(mut terms: Vec<Atom>) -> Atom {
    if terms.len() == 1 {
        return terms.pop().unwrap();
    }

    let mut out = Atom::new();
    let add = out.to_add();
    for term in terms {
        add.extend(term.as_view());
    }
    out
}

#[inline]
fn build_mul_raw(mut factors: Vec<Atom>) -> Atom {
    if factors.len() == 1 {
        return factors.pop().unwrap();
    }

    let mut out = Atom::new();
    let mul = out.to_mul();
    for factor in factors {
        mul.extend(factor.as_view());
    }
    out
}

#[inline]
fn build_pow_raw(base: AtomView<'_>, exp: AtomView<'_>) -> Atom {
    let mut out = Atom::new();
    out.to_pow(base, exp);
    out
}

#[inline]
fn negate_raw(view: AtomView<'_>) -> Atom {
    let mut out = Atom::new();
    let mul = out.to_mul();
    mul.extend(Atom::new_num(-1).as_view());
    mul.extend(view);
    out
}

#[inline]
fn reciprocal_raw(view: AtomView<'_>) -> Atom {
    build_pow_raw(view, Atom::new_num(-1).as_view())
}

#[cfg(test)]
mod test {
    use crate::symbolic::View::parser::parse;
    use crate::{parse, symbol};

    #[test]
    fn parses_scientific_notation_exactly() {
        assert_eq!(parse!("1e3").unwrap(), parse!("1000").unwrap());
        assert_eq!(parse!("2.5e-1").unwrap(), parse!("1/4").unwrap());
        assert_eq!(parse!("1.25e2").unwrap(), parse!("125").unwrap());
    }

    #[test]
    fn error_reports_line_and_column() {
        let err = parse(
            "x +
(1 + )",
        )
        .unwrap_err();
        assert!(err.contains("line 2"));
        assert!(err.contains("column"));
    }

    #[test]
    fn implicit_multiplication_inside_function_argument_is_supported() {
        assert_eq!(parse!("f(x 1)").unwrap(), parse!("f(x*1)").unwrap());
    }

    #[test]
    fn parses_uppercase_scientific_notation() {
        assert_eq!(parse!("3E2").unwrap(), parse!("300").unwrap());
        assert_eq!(parse!("4.2E-1").unwrap(), parse!("21/50").unwrap());
    }

    #[test]
    fn rejects_incomplete_scientific_notation() {
        let err = parse("1e+").unwrap_err();
        assert!(err.contains("valid number literal") || err.contains("out-of-range"));
    }

    #[test]
    fn namespaced_identifiers_parse() {
        let expr = parse!("symbolica::sin(x)").unwrap();
        assert_eq!(expr.to_string(), "sin(x)");
    }
    #[test]
    fn builtin_aliases_canonicalize_during_parse() {
        assert_eq!(parse!("tg(x)").unwrap(), parse!("tan(x)").unwrap());
        assert_eq!(parse!("arctg(x)").unwrap(), parse!("atan(x)").unwrap());
        assert_eq!(parse!("arcctg(x)").unwrap(), parse!("acot(x)").unwrap());
    }

    #[test]
    fn parses_implicit_multiplication() {
        assert_eq!(parse!("2x").unwrap(), parse!("2*x").unwrap());
        assert_eq!(parse!("x y").unwrap(), parse!("x*y").unwrap());
        assert_eq!(parse!("2(x+1)").unwrap(), parse!("2*(x+1)").unwrap());
        assert_eq!(
            parse!("(x+1)(y+1)").unwrap(),
            parse!("(x+1)*(y+1)").unwrap()
        );
        assert_eq!(parse!("2sin(x)").unwrap(), parse!("2*sin(x)").unwrap());
    }

    #[test]
    fn does_not_merge_adjacent_numbers_as_implicit_multiplication() {
        let err = parse("2 3").unwrap_err();
        assert!(err.contains("unexpected '3' after complete expression"));
    }

    #[test]
    fn parsed_symbol_uses_default_namespace() {
        let x = symbol!("x");
        let parsed = parse!("x").unwrap();
        assert_eq!(parsed, crate::symbolic::View::atom::Atom::new_var(x));
    }
}
