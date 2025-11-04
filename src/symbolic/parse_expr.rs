//use crate::symbolic::shared_expr::Expr;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::utils::{find_char_positions_outside_brackets, find_pair_to_this_bracket};
/// a module turns a String expression into a symbolic expression
///# Example
/// ```
/// use RustedSciThe::symbolic::symbolic_engine::Expr;
///let input = "x^2.3* log(x+y+y^2.6)"; //log(x)/y-x^2.3 log(x+y+y^2.6)-exp(x-y)/(x+y)
/// let parsed_expression = Expr::parse_expression(input);
///println!(" parsed_expression {}", parsed_expression);
/// let parsed_function = parsed_expression.lambdify_borrowed_thread_safe( &["x","y"]);
/// println!("{}, Rust function: {}  \n",input,  parsed_function(&[1.0,2.0])    );
///  
/// ```
//                  search recursion diagram
//                "y^2+exp(x)+log(x)/y-x^2.3"       |
//                |       left  | right             |
//                |_________________________________|
//                |           div by    +           |
//                |_________________________________|
//                |       y^2   |exp(x)+log(x)/y-...|
//                |      (0, 2) |   (4,...)         |
//                |       |     |          |        |
//                |_____ \|/    |          |        |
//                |           div by^      |        |
//                |________________________|________|
//                |       y     |  2       |        |
//                |____________Ok_________\|/_______|
//                |           div by+      |        |
//                |_______________________\|/_______|
//                |       exp(x)| log(x)/y-...      |
//                |      (0, 5) |   (7,...)         |
//                |       |     |          |        |
//                |_____ \|/____|__________|________|
//                |             x          |        |
//                |_____________Ok________\|/_______|
//                  etc...
#[allow(dead_code)]
fn proc_negative<'a>(left: String, flg: usize) -> String {
    let new_left: String = format!("(-1.0)*{}", left.clone());
    let left_ = if flg == 2 { new_left } else { left };
    left_
}

// function to find the rightmost occurrence of operators at the same precedence level,

fn find_rightmost_operator_outside_brackets(
    input: &str,
    operators: &[char],
) -> Option<(usize, char)> {
    let mut bracket_depth = 0;
    let mut last_op_pos = None;
    let mut last_op_char = ' ';

    for (i, c) in input.chars().enumerate() {
        match c {
            '(' => bracket_depth += 1,
            ')' => bracket_depth -= 1,
            _ if bracket_depth == 0 && operators.contains(&c) => {
                last_op_pos = Some(i); // Updates to LAST match
                last_op_char = c; // Remembers which operator
            }
            _ => {}
        }
    }

    last_op_pos.map(|pos| (pos, last_op_char))
}
pub fn parse_expression_func(flg: usize, input: &str) -> Result<Expr, String> {
    let input = input.trim();
    println!("input: {} flag {} \n", input, flg);
    let mut brac_end = 0;
    let mut brac_start = 0;
    // Обработка выражений в скобках
    if let Some(bracket_start) = input.find('(') {
        let mut stack = 0;
        let mut bracket_end = None;
        //    let mut inner_expr: Option<Expr> = None;
        for (i, c) in input.chars().enumerate() {
            // If a '(' is found, it initializes a stack variable to keep track of nested brackets. It also initializes an end_pos variable
            // to store the position of the closing bracket. It then iterates through the characters of the input string, incrementing
            //or decrementing the stack variable based on whether it encounters an opening or closing bracket.
            if c == '(' {
                stack += 1;
            } else if c == ')' {
                stack -= 1;
                if stack == 0 {
                    bracket_end = Some(i);
                    break;
                }
                //  break; // by adding this line, it will stop the loop when the first closing bracket is found, not the last one
            }
        } // end for

        // bracket proceeding: if flg==0, the input expression not in brackets - bracket processing is not needed
        // if flg==1, the input expression in brackets - bracket processing is needed.
        if let Some(end) = bracket_end {
            println!(
                " start bracket: {}, end bracket: {:?}, input: {}",
                bracket_start, end, input
            );
            brac_end = end;
            brac_start = bracket_start;

            if flg == 1 {
                println!("found in brackets: {}", input);
                let inner_str = &input[brac_start + 1..brac_end];
                let inner = parse_expression_func(0, inner_str)?;
                println!("content in brackets parsed");
                let remaining_right = &input[brac_end + 1..];
                let remaining_left = &input[..brac_start];
                println!(
                    "inner brackets expression: {}, remaining content left: {}, remaining content right: {} ",
                    inner, remaining_left, remaining_right
                );
            }
        }
    } // end if brackets
    println!("went out of brackets proceeding with input: {}", input);
    if flg == 2 { // перехват случаев с отрицательными величинами
    }

    // Обработка сложения и вычитания
    {
        if let Some((pos, op)) = find_rightmost_operator_outside_brackets(input, &['+', '-']) {
            let mut left = &input[..pos].trim();
            let right = &input[pos + 1..].trim();

            println!(
                "SIGN '{}' found at position {}: left: {}, right: {}",
                op, pos, left, right
            );

            // Handle negative flag
            let a = left.to_string();
            let new_left = format!("(-1.0)*{}", a);
            let new_left = &new_left.as_str();
            left = if flg == 2 { new_left } else { left };

            // Handle unary minus
            if left.is_empty() && op == '-' {
                println!("negative values found");
                if right.starts_with("(") && right.ends_with(")") {
                    return Ok(Expr::Mul(
                        Box::new(Expr::Const(-1.)),
                        Box::new(parse_expression_func(1, right)?),
                    ));
                } else {
                    let inner = parse_expression_func(2, right)?;
                    return Ok(inner);
                }
            }

            // Handle bracket cases
            let left_in_brackets = left.starts_with("(") && left.ends_with(")");
            let right_in_brackets = right.starts_with("(") && right.ends_with(")");

            return match (left_in_brackets, right_in_brackets, op) {
                (true, false, '+') => Ok(Expr::Add(
                    Box::new(parse_expression_func(1, left)?),
                    Box::new(parse_expression_func(0, right)?),
                )),
                (true, false, '-') => Ok(Expr::Sub(
                    Box::new(parse_expression_func(1, left)?),
                    Box::new(parse_expression_func(0, right)?),
                )),
                (false, true, '+') => Ok(Expr::Add(
                    Box::new(parse_expression_func(0, left)?),
                    Box::new(parse_expression_func(1, right)?),
                )),
                (false, true, '-') => Ok(Expr::Sub(
                    Box::new(parse_expression_func(0, left)?),
                    Box::new(parse_expression_func(1, right)?),
                )),
                (true, true, '+') => Ok(Expr::Add(
                    Box::new(parse_expression_func(1, left)?),
                    Box::new(parse_expression_func(1, right)?),
                )),
                (true, true, '-') => Ok(Expr::Sub(
                    Box::new(parse_expression_func(1, left)?),
                    Box::new(parse_expression_func(1, right)?),
                )),
                (false, false, '+') => Ok(Expr::Add(
                    Box::new(parse_expression_func(0, left)?),
                    Box::new(parse_expression_func(0, right)?),
                )),
                (false, false, '-') => Ok(Expr::Sub(
                    Box::new(parse_expression_func(0, left)?),
                    Box::new(parse_expression_func(0, right)?),
                )),
                _ => unreachable!(),
            };
        }
        // Обработка умножения и деления/Handling multiplication and division
        if let Some(pos) = find_char_positions_outside_brackets(input, '*') {
            let mut left = &input[..pos].trim();
            let right = &input[pos + 1..].trim();
            println!("SIGN '*' at positions {}-{}", pos, pos + 1);
            println!(
                "brackets found in posistions: {} - {},",
                brac_start, brac_end
            );

            let a = left.to_string();
            let new_left = format!("(-1.0)*{}", a);
            let new_left = &new_left.as_str();
            left = if flg == 2 { new_left } else { left };
            return if (left.starts_with("(") & left.ends_with(")"))
                && !(right.starts_with("(") && right.ends_with(")"))
            {
                println!("left in brackets");
                Ok(Expr::Mul(
                    Box::new(parse_expression_func(1, left)?),
                    Box::new(parse_expression_func(0, right)?),
                ))
            } else if (right.starts_with("(") && right.ends_with(")"))
                && !(left.starts_with("(") && left.ends_with(")"))
            {
                println!("right in brackets");
                Ok(Expr::Mul(
                    Box::new(parse_expression_func(0, left)?),
                    Box::new(parse_expression_func(1, right)?),
                ))
            } else if (left.starts_with("(") && left.ends_with(")"))
                && (right.starts_with("(") && right.ends_with(")"))
            {
                println!("both sides have brackets");
                Ok(Expr::Mul(
                    Box::new(parse_expression_func(1, left)?),
                    Box::new(parse_expression_func(1, right)?),
                ))
            } else {
                println!("neither in brackets");
                Ok(Expr::Mul(
                    Box::new(parse_expression_func(0, left)?),
                    Box::new(parse_expression_func(0, right)?),
                ))
            };
        } else if let Some(pos) = find_char_positions_outside_brackets(input, '/') {
            let mut left = &input[..pos].trim();
            let right = &input[pos + 1..].trim();
            println!("SIGN '/' at positions {}-{}", pos, pos + 1);

            let a = left.to_string();
            let new_left = format!("(-1.0)*{}", a);
            let new_left = &new_left.as_str();
            left = if flg == 2 { new_left } else { left };
            return if (left.starts_with("(") && left.ends_with(")"))
                && !(right.starts_with("(") && right.ends_with(")"))
            {
                println!("left in brackets");
                Ok(Expr::Div(
                    Box::new(parse_expression_func(1, left)?),
                    Box::new(parse_expression_func(0, right)?),
                ))
            } else if (right.starts_with("(") && right.ends_with(")"))
                && !(left.starts_with("(") && left.ends_with(")"))
            {
                println!("right in brackets");
                Ok(Expr::Div(
                    Box::new(parse_expression_func(0, left)?),
                    Box::new(parse_expression_func(1, right)?),
                ))
            } else if left.starts_with("(")
                && left.ends_with(")")
                && (right.starts_with("(") && right.ends_with(")"))
            {
                println!("both sides have brackets");
                Ok(Expr::Div(
                    Box::new(parse_expression_func(1, left)?),
                    Box::new(parse_expression_func(1, right)?),
                ))
            } else {
                println!("neither in brackets");
                Ok(Expr::Div(
                    Box::new(parse_expression_func(0, left)?),
                    Box::new(parse_expression_func(0, right)?),
                ))
            };
        } else {
            println!("no matches among +, -, *, /")
        };
        // Обработка возведения в степень
        if let Some(pos) = find_char_positions_outside_brackets(input, '^') {
            let base = &input[..pos].trim();
            let exponent = &input[pos + 1..].trim();
            println!("SIGN '^' at positions {}-{}", pos, pos + 1);
            let base_expr = if base.chars().all(char::is_alphanumeric) {
                println!("base of power: {}", base);
                Expr::Var(base.to_string())
            } else {
                parse_expression_func(0, base)?
                // return Err("Base must be a variable".to_string());
            };
            /*
            let exponent_expr = match exponent.parse::<f64>() {
                Ok(value) => Expr::Const(value),
                Err(_) => return Err("Exponent must be a number".to_string()),
            };
            */
            let exponent_expr = if let Ok(exponent_expr) = exponent.parse::<f64>() {
                Expr::Const(exponent_expr)
            } else {
                parse_expression_func(0, exponent)?
            };
            return Ok(Expr::Pow(Box::new(base_expr), Box::new(exponent_expr)));
        }

        // Обработка экспоненты и логарифма
        if input.starts_with("exp(") && input.ends_with(')') {
            let fisrt_brac_end = find_pair_to_this_bracket(input, 0);
            //let  fisrt_brac_end = input.find(')').unwrap();
            let inner = &input[4..fisrt_brac_end].trim();
            println!(
                "SIGN 'exp' at positions {}-{}",
                fisrt_brac_end - 4,
                fisrt_brac_end
            );
            return Ok(Expr::Exp(Box::new(parse_expression_func(0, inner)?)));
        } else if input.starts_with("log(") || input.starts_with("ln(") && input.ends_with(')') {
            let fisrt_brac_end = find_pair_to_this_bracket(input, 0);
            // let  fisrt_brac_end = input.find(')').unwrap();
            println!(
                "SIGN 'log' at positions {}-{}",
                fisrt_brac_end - 4,
                fisrt_brac_end
            );
            let inner = &input[4..fisrt_brac_end].trim();
            return Ok(Expr::Ln(Box::new(parse_expression_func(0, inner)?)));
        }

        // Обработка тригонометрических функций
        if input.starts_with("sin(") && input.ends_with(')') {
            let fisrt_brac_end = find_pair_to_this_bracket(input, 0);
            let inner = &input[4..fisrt_brac_end].trim();
            return Ok(Expr::sin(Box::new(parse_expression_func(0, inner)?)));
        } else if input.starts_with("cos(") && input.ends_with(')') {
            let fisrt_brac_end = find_pair_to_this_bracket(input, 0);
            let inner = &input[4..fisrt_brac_end].trim();
            return Ok(Expr::cos(Box::new(parse_expression_func(0, inner)?)));
        } else if input.starts_with("tg(") && input.ends_with(')') {
            let fisrt_brac_end = find_pair_to_this_bracket(input, 0);
            let inner = &input[3..fisrt_brac_end].trim();
            return Ok(Expr::tg(Box::new(parse_expression_func(0, inner)?)));
        } else if input.starts_with("tan(") && input.ends_with(')') {
            let fisrt_brac_end = find_pair_to_this_bracket(input, 0);
            let inner = &input[4..fisrt_brac_end].trim();
            return Ok(Expr::tg(Box::new(parse_expression_func(0, inner)?)));
        } else if input.starts_with("ctg(") && input.ends_with(')') {
            let fisrt_brac_end = find_pair_to_this_bracket(input, 0);
            let inner = &input[4..fisrt_brac_end].trim();
            return Ok(Expr::ctg(Box::new(parse_expression_func(0, inner)?)));
        } else if input.starts_with("cot(") && input.ends_with(')') {
            let fisrt_brac_end = find_pair_to_this_bracket(input, 0);
            let inner = &input[4..fisrt_brac_end].trim();
            return Ok(Expr::ctg(Box::new(parse_expression_func(0, inner)?)));
        } else if input.starts_with("arcsin(") && input.ends_with(')') {
            let fisrt_brac_end = find_pair_to_this_bracket(input, 0);
            let inner = &input[7..fisrt_brac_end].trim();
            return Ok(Expr::arcsin(Box::new(parse_expression_func(0, inner)?)));
        } else if input.starts_with("asin(") && input.ends_with(')') {
            let fisrt_brac_end = find_pair_to_this_bracket(input, 0);
            let inner = &input[5..fisrt_brac_end].trim();
            return Ok(Expr::arcsin(Box::new(parse_expression_func(0, inner)?)));
        } else if input.starts_with("arccos(") && input.ends_with(')') {
            let fisrt_brac_end = find_pair_to_this_bracket(input, 0);
            let inner = &input[7..fisrt_brac_end].trim();
            return Ok(Expr::arccos(Box::new(parse_expression_func(0, inner)?)));
        } else if input.starts_with("acos(") && input.ends_with(')') {
            let fisrt_brac_end = find_pair_to_this_bracket(input, 0);
            let inner = &input[5..fisrt_brac_end].trim();
            return Ok(Expr::arccos(Box::new(parse_expression_func(0, inner)?)));
        } else if input.starts_with("arctg(") && input.ends_with(')') {
            let fisrt_brac_end = find_pair_to_this_bracket(input, 0);
            let inner = &input[6..fisrt_brac_end].trim();
            return Ok(Expr::arctg(Box::new(parse_expression_func(0, inner)?)));
        } else if input.starts_with("atan(") && input.ends_with(')') {
            let fisrt_brac_end = find_pair_to_this_bracket(input, 0);
            let inner = &input[5..fisrt_brac_end].trim();
            return Ok(Expr::arctg(Box::new(parse_expression_func(0, inner)?)));
        } else if input.starts_with("arctan(") && input.ends_with(')') {
            let fisrt_brac_end = find_pair_to_this_bracket(input, 0);
            let inner = &input[7..fisrt_brac_end].trim();
            return Ok(Expr::arctg(Box::new(parse_expression_func(0, inner)?)));
        } else if input.starts_with("arcctg(") && input.ends_with(')') {
            let fisrt_brac_end = find_pair_to_this_bracket(input, 0);
            let inner = &input[7..fisrt_brac_end].trim();
            return Ok(Expr::arcctg(Box::new(parse_expression_func(0, inner)?)));
        } else if input.starts_with("acot(") && input.ends_with(')') {
            let fisrt_brac_end = find_pair_to_this_bracket(input, 0);
            let inner = &input[5..fisrt_brac_end].trim();
            return Ok(Expr::arcctg(Box::new(parse_expression_func(0, inner)?)));
        }

        // Обработка констант и переменных
        if let Ok(value) = input.parse::<f64>() {
            println!("found constant: {}", value);
            return if flg != 2 {
                Ok(Expr::Const(value))
            } else {
                Ok(Expr::Const(-value))
            };
        } else if input.chars().all(char::is_alphanumeric) {
            println!("found variable: {}", input);
            return if flg != 2 {
                Ok(Expr::Var(input.to_string()))
            } else {
                Ok(Expr::Mul(
                    Box::new(Expr::Const(-1.0)),
                    Box::new(Expr::Var(input.to_string())),
                ))
            };
        }

        if input.starts_with("(") && input.ends_with(')') {
            let inner_str = &input[brac_start + 1..brac_end];
            let inner = parse_expression_func(0, inner_str)?;

            println!("found expression that is ALL in brackets: {:?}", inner);
            return Ok(inner);
        }
    }
    Err("Invalid expression format".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_parse_exponential() {
        let expr = parse_expression_func(0, "exp(x)").unwrap();
        assert_eq!(expr, Expr::Exp(Box::new(Expr::Var("x".to_string()))));
    }

    #[test]
    fn test_parse_constant() {
        let expr = parse_expression_func(0, "42").unwrap();
        assert_eq!(expr, Expr::Const(42.0));
    }

    #[test]
    fn test_parse_variable() {
        let expr = parse_expression_func(0, "x").unwrap();
        assert_eq!(expr, Expr::Var("x".to_string()));
    }

    #[test]
    fn test_parse_addition() {
        let expr = parse_expression_func(0, "x + 2").unwrap();
        assert_eq!(
            expr,
            Expr::Add(
                Box::new(Expr::Var("x".to_string())),
                Box::new(Expr::Const(2.0))
            )
        );
    }

    #[test]
    fn test_parse_subtraction() {
        let expr = parse_expression_func(0, "x - 2").unwrap();
        assert_eq!(
            expr,
            Expr::Sub(
                Box::new(Expr::Var("x".to_string())),
                Box::new(Expr::Const(2.0))
            )
        );
    }

    #[test]
    fn test_parse_multiplication() {
        let expr = parse_expression_func(0, "x * 2").unwrap();
        assert_eq!(
            expr,
            Expr::Mul(
                Box::new(Expr::Var("x".to_string())),
                Box::new(Expr::Const(2.0))
            )
        );
    }

    #[test]
    fn test_parse_division() {
        let expr = parse_expression_func(0, "x / 2").unwrap();
        assert_eq!(
            expr,
            Expr::Div(
                Box::new(Expr::Var("x".to_string())),
                Box::new(Expr::Const(2.0))
            )
        );
    }

    #[test]
    fn test_parse_power() {
        let expr = parse_expression_func(0, "x^2").unwrap();
        assert_eq!(
            expr,
            Expr::Pow(
                Box::new(Expr::Var("x".to_string())),
                Box::new(Expr::Const(2.0))
            )
        );
    }

    #[test]
    fn test_parse_logarithm() {
        let expr = parse_expression_func(0, "log(x)").unwrap();
        assert_eq!(expr, Expr::Ln(Box::new(Expr::Var("x".to_string()))));
    }

    #[test]
    fn test_parse_expression_func_with_brackets() {
        let expr = parse_expression_func(0, "(x + y) * z").unwrap();
        assert_eq!(
            expr,
            Expr::Mul(
                Box::new(Expr::Add(
                    Box::new(Expr::Var("x".to_string())),
                    Box::new(Expr::Var("y".to_string()))
                )),
                Box::new(Expr::Var("z".to_string()))
            )
        );
    }

    #[test]
    fn test_parse_complex_expression() {
        let expr = parse_expression_func(0, "(x + y) * (z - 2) / exp(w)").unwrap();
        let x = Box::new(Expr::Var("x".to_string()));
        let y = Box::new(Expr::Var("y".to_string()));
        let z = Box::new(Expr::Var("z".to_string()));
        let w = Box::new(Expr::Var("w".to_string()));
        let C = Box::new(Expr::Const(2.0));
        let x_plus_y = Box::new(Expr::Add(x, y));
        let z_minus_C = Box::new(Expr::Sub(z, C));
        let e = Box::new(Expr::Exp(w));
        let z_minus_C_div_e = Box::new(Expr::Div(z_minus_C, e));
        let Res = Expr::Mul(x_plus_y, z_minus_C_div_e);
        assert_eq!(expr, Res);
    }

    #[test]
    fn test_invalid_expression() {
        let result = parse_expression_func(0, "(x +");
        assert_eq!(result.is_err(), true);
    }

    #[test]
    fn test_unmatched_brackets() {
        let result = parse_expression_func(0, "(x + y");
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_addition() {
        let result = parse_expression_func(0, "x^2 - x - 1");
        let x = Box::new(Expr::Var("x".to_string()));
        let to_check = Expr::Pow(x.clone(), Box::new(Expr::Const(2.0))) - *x - Expr::Const(1.0);
        println!("{}", to_check);
        println!("{}", result.clone().unwrap());
        assert_eq!(result.unwrap(), to_check);
    }

    #[test]
    fn test_parse_sin() {
        let expr = parse_expression_func(0, "sin(x)").unwrap();
        assert_eq!(expr, Expr::sin(Box::new(Expr::Var("x".to_string()))));
    }

    #[test]
    fn test_parse_cos() {
        let expr = parse_expression_func(0, "cos(x)").unwrap();
        assert_eq!(expr, Expr::cos(Box::new(Expr::Var("x".to_string()))));
    }

    #[test]
    fn test_parse_tg() {
        let expr = parse_expression_func(0, "tg(x)").unwrap();
        assert_eq!(expr, Expr::tg(Box::new(Expr::Var("x".to_string()))));
    }

    #[test]
    fn test_parse_tan() {
        let expr = parse_expression_func(0, "tan(x)").unwrap();
        assert_eq!(expr, Expr::tg(Box::new(Expr::Var("x".to_string()))));
    }

    #[test]
    fn test_parse_arcsin() {
        let expr = parse_expression_func(0, "arcsin(x)").unwrap();
        assert_eq!(expr, Expr::arcsin(Box::new(Expr::Var("x".to_string()))));
    }

    #[test]
    fn test_parse_complex_trig() {
        let expr = parse_expression_func(0, "sin(x) + cos(y)").unwrap();
        assert_eq!(
            expr,
            Expr::Add(
                Box::new(Expr::sin(Box::new(Expr::Var("x".to_string())))),
                Box::new(Expr::cos(Box::new(Expr::Var("y".to_string()))))
            )
        );
    }

    #[test]
    fn test_parse_nested_trig() {
        let expr = parse_expression_func(0, "sin(cos(x))").unwrap();
        assert_eq!(
            expr,
            Expr::sin(Box::new(Expr::cos(Box::new(Expr::Var("x".to_string())))))
        );
    }
}
