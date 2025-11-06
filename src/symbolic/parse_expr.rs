use std::collections::HashMap;

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
////////////////////////////DIAGNOSTIC TOOLS//////////////////////////////////////////////////
#[allow(dead_code)]
// Optimized single-pass operator finder
fn find_all_operators_outside_brackets(input: &str) -> Vec<(usize, char)> {
    let mut operators = Vec::new();
    let mut bracket_depth = 0;

    for (i, c) in input.chars().enumerate() {
        match c {
            '(' => bracket_depth += 1,
            ')' => bracket_depth -= 1,
            '+' | '-' | '*' | '/' | '^' if bracket_depth == 0 => {
                operators.push((i, c));
            }
            _ => {}
        }
    }
    operators
}
#[allow(dead_code)]
// Find rightmost operator of specific types
fn find_rightmost_of_types(operators: &[(usize, char)], types: &[char]) -> Option<(usize, char)> {
    operators
        .iter()
        .filter(|(_, op)| types.contains(op))
        .last()
        .copied()
}
#[allow(dead_code)]
fn bracket_matching(input: &str) {
    let mut stack = 0;
    let mut closed_bracket_point: HashMap<i32, String> = HashMap::new();
    for (i, c) in input.chars().enumerate() {
        // If a '(' is found, it initializes a stack variable to keep track of nested brackets. It also initializes an end_pos variable
        // to store the position of the closing bracket. It then iterates through the characters of the input string, incrementing
        //or decrementing the stack variable based on whether it encounters an opening or closing bracket.
        if c == '(' {
            stack += 1;
        } else if c == ')' {
            stack -= 1;
            if stack == 0 {
                closed_bracket_point.insert(
                    i as i32,
                    if i + 1 < input.len() {
                        input[i..i + 5].to_string()
                    } else {
                        input[i..].to_string()
                    },
                );
                break;
            }
            //  break; // by adding this line, it will stop the loop when the first closing bracket is found, not the last one
        }
    } // end for
    // If the stack variable is 0, it means that the brackets are balanced and the loop has found the matching closing bracket.
    // If the stack variable is not 0, it means that the brackets are not balanced and the loop has not found the matching closing bracket.
    if stack == 0 {
        println!("Brackets are balanced");
    } else {
        println!("Brackets are not balanced");
    }
    println!("closed_bracket_point: {:?}", closed_bracket_point);
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
    const  S: &'static str = "((1000 * (((0.000002669 * ((28 * T) ^ 0.5)) / (13.3225 * ((((1.16145 / ((T / 98.1) ^ 0.14874)) + (0.52487 / exp((0.7732 * (T / 98.1))))) + (2.16178 / exp((2.43787 * (T / 98.1))))) + ((0.2 * (0 ^ 2)) / (T / 98.1))))) / 0.028)) * ((((2.5 * (1 - ((0.6366197723675814 * (8.314 / (1.5 * 8.314))) * ((2.5 - (((ro * 1) * (((18750000000000000000 * (T ^ 1.5)) * 0.00000000000000000000018973399110000764) / (1349902.3125 * (((((1.06036 / ((T / 98.1) ^ 0.1561)) + (0.193 / exp((0.47635 * (T / 98.1))))) + (1.03587 / exp((1.52996 * (T / 98.1))))) + (1.76474 / exp((3.89411 * (T / 98.1))))) + ((0.19 * (0 ^ 2)) / (T / 98.1)))))) / ((0.000002669 * ((28 * T) ^ 0.5)) / (13.3225 * ((((1.16145 / ((T / 98.1) ^ 0.14874)) + (0.52487 / exp((0.7732 * (T / 98.1))))) + (2.16178 / exp((2.43787 * (T / 98.1))))) + ((0.2 * (0 ^ 2)) / (T / 98.1))))))) / (((1.8 * (((1 + (2.784163998415854 * ((98.1 / T) ^ 0.5))) + (4.4674011002723395 * (98.1 / T))) + (5.568327996831708 * ((98.1 / T) ^ 3.2)))) / (1 + 3.2271366789503664)) + (0.6366197723675814 * ((0.20046507898324112 * 8.314) + (((ro * 1) * (((18750000000000000000 * (T ^ 1.5)) * 0.00000000000000000000018973399110000764) / (1349902.3125 * (((((1.06036 / ((T / 98.1) ^ 0.1561)) + (0.193 / exp((0.47635 * (T / 98.1))))) + (1.03587 / exp((1.52996 * (T / 98.1))))) + (1.76474 / exp((3.89411 * (T / 98.1))))) + ((0.19 * (0 ^ 2)) / (T / 98.1)))))) / ((0.000002669 * ((28 * T) ^ 0.5)) / (13.3225 * ((((1.16145 / ((T / 98.1) ^ 0.14874)) + (0.52487 / exp((0.7732 * (T / 98.1))))) + (2.16178 / exp((2.43787 * (T / 98.1))))) + ((0.2 * (0 ^ 2)) / (T / 98.1))))))))))))) * (1.5 * 8.314)) + (((((ro * 1) * (((18750000000000000000 * (T ^ 1.5)) * 0.00000000000000000000018973399110000764) / (1349902.3125 * (((((1.06036 / ((T / 98.1) ^ 0.1561)) + (0.193 / exp((0.47635 * (T / 98.1))))) + (1.03587 / exp((1.52996 * (T / 98.1))))) + (1.76474 / exp((3.89411 * (T / 98.1))))) + ((0.19 * (0 ^ 2)) / (T / 98.1)))))) * (1 + (0.6366197723675814 * ((2.5 - (((ro * 1) * (((18750000000000000000 * (T ^ 1.5)) * 0.00000000000000000000018973399110000764) / (1349902.3125 * (((((1.06036 / ((T / 98.1) ^ 0.1561)) + (0.193 / exp((0.47635 * (T / 98.1))))) + (1.03587 / exp((1.52996 * (T / 98.1))))) + (1.76474 / exp((3.89411 * (T / 98.1))))) + ((0.19 * (0 ^ 2)) / (T / 98.1)))))) / ((0.000002669 * ((28 * T) ^ 0.5)) / (13.3225 * ((((1.16145 / ((T / 98.1) ^ 0.14874)) + (0.52487 / exp((0.7732 * (T / 98.1))))) + (2.16178 / exp((2.43787 * (T / 98.1))))) + ((0.2 * (0 ^ 2)) / (T / 98.1))))))) / (((1.8 * (((1 + (2.784163998415854 * ((98.1 / T) ^ 0.5))) + (4.4674011002723395 * (98.1 / T))) + (5.568327996831708 * ((98.1 / T) ^ 3.2)))) / (1 + 3.2271366789503664)) + (0.6366197723675814 * ((0.20046507898324112 * 8.314) + (((ro * 1) * (((18750000000000000000 * (T ^ 1.5)) * 0.00000000000000000000018973399110000764) / (1349902.3125 * (((((1.06036 / ((T / 98.1) ^ 0.1561)) + (0.193 / exp((0.47635 * (T / 98.1))))) + (1.03587 / exp((1.52996 * (T / 98.1))))) + (1.76474 / exp((3.89411 * (T / 98.1))))) + ((0.19 * (0 ^ 2)) / (T / 98.1)))))) / ((0.000002669 * ((28 * T) ^ 0.5)) / (13.3225 * ((((1.16145 / ((T / 98.1) ^ 0.14874)) + (0.52487 / exp((0.7732 * (T / 98.1))))) + (2.16178 / exp((2.43787 * (T / 98.1))))) + ((0.2 * (0 ^ 2)) / (T / 98.1))))))))))))) / ((0.000002669 * ((28 * T) ^ 0.5)) / (13.3225 * ((((1.16145 / ((T / 98.1) ^ 0.14874)) + (0.52487 / exp((0.7732 * (T / 98.1))))) + (2.16178 / exp((2.43787 * (T / 98.1))))) + ((0.2 * (0 ^ 2)) / (T / 98.1)))))) * 8.314)) + ((((ro * 1) * (((18750000000000000000 * (T ^ 1.5)) * 0.00000000000000000000018973399110000764) / (1349902.3125 * (((((1.06036 / ((T / 98.1) ^ 0.1561)) + (0.193 / exp((0.47635 * (T / 98.1))))) + (1.03587 / exp((1.52996 * (T / 98.1))))) + (1.76474 / exp((3.89411 * (T / 98.1))))) + ((0.19 * (0 ^ 2)) / (T / 98.1)))))) / ((0.000002669 * ((28 * T) ^ 0.5)) / (13.3225 * ((((1.16145 / ((T / 98.1) ^ 0.14874)) + (0.52487 / exp((0.7732 * (T / 98.1))))) + (2.16178 / exp((2.43787 * (T / 98.1))))) + ((0.2 * (0 ^ 2)) / (T / 98.1)))))) * ((Cp - 8.314) - (2.5 * 8.314)))))";
  
    #[test]
    fn parse_very_complex_expression() {
        bracket_matching(S);
    }

    #[test]
    fn parser_test2() {
    
        let mass_const = 2.635385020221708e-09;
        let K1 = 0.6366197723675814;
        let K2 = 0.20046507898324112;
        let R = 8.3144598;
        let denomA = "1.16145/(T/98.1)^0.14874 + 0.52487/exp(0.7732*(T/98.1)) + 2.16178/exp(2.43787*(T/98.1))";
        let denomA = parse_expression_func(0, denomA).unwrap();
        let mu = "0.000002669 * (28.0*T)^0.5 / (13.3225 * (1.16145/(T/98.1)^0.14874 + 0.52487/exp(0.7732*(T/98.1)) + 2.16178/exp(2.43787*(T/98.1))))";
        let mu = parse_expression_func(0, mu).unwrap();
        let denomB = "1.06036/(T/98.1)^0.1561 + 0.193/exp(0.47635*(T/98.1)) + 1.03587/exp(1.52996*(T/98.1)) + 1.76474/exp(3.89411*(T/98.1))";
        let denomB = parse_expression_func(0, denomB).unwrap();
        let mraw = "ro  * (18750000000000000000 * 0.00000000000000000000018973399110000764 / 1349902.3125)* T^1.5 / denomB";
        let mraw = parse_expression_func(0, mraw).unwrap();
        let mraw = mraw.substitute_variable("denomB", &denomB);
        let denomD = "(1.8 * (1 + 2.784163998415854*(98.1/T)^0.5 + 4.4674011002723395*(98.1/T) + 5.568327996831708*(98.1/T)^3.2)) / (1 + 3.2271366789503664)";
        let denomD = parse_expression_func(0, denomD).unwrap();
        let q = "((ro *mass*(T ^ 1.5))/denomB )/((0.000002669 * (28.0 * T)^0.5)/(13.3225*denomA))";
        let q = parse_expression_func(0, q).unwrap();
        let q = q
            .substitute_variable("denomA", &denomA)
            .substitute_variable("denomB", &denomB)
            .set_variable("mass", mass_const);
        let A = "2.5*(1 - (K1*(R/(1.5*R))*(2.5 - q) / mu)) / ( denomD + K1*(K2*R + q) )";
        let A = parse_expression_func(0, A).unwrap();
        let A = A
            .substitute_variable("denomD", &denomD)
            .substitute_variable("q", &q)
            .substitute_variable("mu", &mu)
            .set_variable("K1", K1)
            .set_variable("K2", K2)
            .set_variable("R", R);
        let B = "( mraw * (1 + K1*( (2.5 - q) / mu )) ) / mu";
        let B = parse_expression_func(0, B).unwrap();
        let B = B
            .substitute_variable("mraw", &mraw)
            .substitute_variable("mu", &mu)
            .substitute_variable("q", &q)
            .set_variable("K1", K1);
        let C = "mraw / mu";
        let C = parse_expression_func(0, C).unwrap();
        let C = C
            .substitute_variable("mraw", &mraw)
            .substitute_variable("mu", &mu);

        let X = "1000 * ( mu / 0.028 )";
        let X = parse_expression_func(0, X).unwrap();
        let X = X.substitute_variable("mu", &mu);

        let Expression = "X * ( A*(1.5*R) + B*R + C*( Cp - 3.5*R ) )";
        let Expression = parse_expression_func(0, Expression).unwrap();
        let Expression = Expression
            .substitute_variable("A", &A)
            .substitute_variable("B", &B)
            .substitute_variable("C", &C)
            .substitute_variable("X", &X)
            .set_variable("R", 8.3144598);

        println!("\n \n \n expr {}", Expression);
        let lambda = Expression.set_variable("Cp", 10.0).set_variable("ro", 0.01);
        let lambda_f = lambda.clone().lambdify1D()(400.0);
        println!("\n \n lambda {} ", lambda_f);
    }
    #[test]
    fn parser_test3(){

                // === Constants ==============================================================
        let R: f64 = 8.3144598;
        let K1: f64 = 0.6366197723675814;
        let K2: f64 = 0.20046507898324112;
        let mass_const: f64 = 2.635385020221708e-09; // 2.635e−9

        // === Denominator A ==========================================================
        let denomA_str = "
            1.16145/(T/98.1)^0.14874
            + 0.52487/exp(0.7732*(T/98.1))
            + 2.16178/exp(2.43787*(T/98.1))
        ";
        let denomA = parse_expression_func(0, denomA_str).unwrap();

        // === Viscosity μ ============================================================
        let mu_str = "
            0.000002669 * (28.0*T)^0.5 /
            (13.3225 * (
                1.16145/(T/98.1)^0.14874
                + 0.52487/exp(0.7732*(T/98.1))
                + 2.16178/exp(2.43787*(T/98.1))
            ))
        ";
        let mu = parse_expression_func(0, mu_str).unwrap();

        // === Denominator B ==========================================================
        let denomB_str = "
            1.06036/(T/98.1)^0.1561
            + 0.193/exp(0.47635*(T/98.1))
            + 1.03587/exp(1.52996*(T/98.1))
            + 1.76474/exp(3.89411*(T/98.1))
        ";
        let denomB = parse_expression_func(0, denomB_str).unwrap();

        // === m_raw ================================================================
        let mraw_str = "
            ro * (18750000000000000000 * 0.00000000000000000000018973399110000764 / 1349902.3125)
            * T^1.5 / denomB
        ";
        let mut mraw = parse_expression_func(0, mraw_str).unwrap();
        mraw = mraw.substitute_variable("denomB", &denomB);

        // === Denominator D =========================================================
        let denomD_str = "
            (1.8 * (
                1
                + 2.784163998415854*(98.1/T)^0.5
                + 4.4674011002723395*(98.1/T)
                + 5.568327996831708*(98.1/T)^3.2
            )) / (1 + 3.2271366789503664)
        ";
        let denomD = parse_expression_func(0, denomD_str).unwrap();

        // === q =====================================================================
        let q_str = "
            ((ro * mass * (T^1.5)) / denomB)
            /
            ((0.000002669 * (28.0 * T)^0.5) / (13.3225 * denomA))
        ";
        let mut q = parse_expression_func(0, q_str).unwrap();
        q = q
            .substitute_variable("denomA", &denomA)
            .substitute_variable("denomB", &denomB)
            .set_variable("mass", mass_const);

        // === A =====================================================================
        let A_str = "
            2.5 * (1 - (K1 * (R/(1.5*R)) * (2.5 - q) / mu))
            / (denomD + K1 * (K2 * R + q))
        ";
        let mut A = parse_expression_func(0, A_str).unwrap();
        A = A
            .substitute_variable("denomD", &denomD)
            .substitute_variable("q", &q)
            .substitute_variable("mu", &mu)
            .set_variable("K1", K1)
            .set_variable("K2", K2)
            .set_variable("R", R);

        // === B =====================================================================
        let B_str = "
            (mraw * (1 + K1 * ((2.5 - q) / mu))) / mu
        ";
        let mut B = parse_expression_func(0, B_str).unwrap();
        B = B
            .substitute_variable("mraw", &mraw)
            .substitute_variable("mu", &mu)
            .substitute_variable("q", &q)
            .set_variable("K1", K1);

        // === C =====================================================================
        let C_str = "mraw / mu";
        let mut C = parse_expression_func(0, C_str).unwrap();
        C = C
            .substitute_variable("mraw", &mraw)
            .substitute_variable("mu", &mu);

        // === X =====================================================================
        let X_str = "1000 * (mu / 0.028)";
        let mut X = parse_expression_func(0, X_str).unwrap();
        X = X.substitute_variable("mu", &mu);

        // === Final Expression ======================================================
        let expr_str = "
            X * (
                A * (1.5 * R)
                + B * R
                + C * (Cp - 3.5 * R)
            )
        ";
        let mut Expression = parse_expression_func(0, expr_str).unwrap();
        Expression = Expression
            .substitute_variable("A", &A)
            .substitute_variable("B", &B)
            .substitute_variable("C", &C)
            .substitute_variable("X", &X)
            .set_variable("R", R);
        let lambda = Expression.set_variable("Cp", 10.0).set_variable("ro", 0.01);
        let lambda_f = lambda.clone().lambdify1D()(400.0);
        println!("\n \n lambda {} ", lambda_f);
    }
}
