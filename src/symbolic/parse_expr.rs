//use crate::symbolic::shared_expr::Expr;
use crate::symbolic::utils::{has_brackets, find_char_positions_outside_brackets, find_pair_to_this_bracket};
use crate::symbolic::symbolic_engine::Expr;
/// a module turns a String expression into a symbolic expression
///# Example
/// ```
/// use RustedSciThe::symbolic::symbolic_engine::Expr;
///let input = "x^2.3* log(x+y+y^2.6)"; //log(x)/y-x^2.3 log(x+y+y^2.6)-exp(x-y)/(x+y)
/// let parsed_expression = Expr::parse_expression(input);
///println!(" parsed_expression {}", parsed_expression);
/// let parsed_function = parsed_expression.lambdify( vec!["x","y"]);
/// println!("{}, Rust function: {}  \n",input,  parsed_function(vec![1.0,2.0])    );
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
    let left_ = if flg==2 {new_left} else {left};
    left_
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
            println!(" start bracket: {}, end bracket: {:?}, input: {}",  bracket_start, end, input);
            brac_end = end ;
            brac_start = bracket_start;
            
            if flg==1{
                println!("found in brackets: {}", input);
                let inner_str = &input[brac_start + 1..brac_end];
                let inner= parse_expression_func(0, inner_str)?;
                println!("content in brackets parsed");
                let remaining_right = &input[brac_end+1..];
                let remaining_left = &input[..brac_start];
                println!("inner brackets expression: {}, remaining content left: {}, remaining content right: {} ", inner, remaining_left,remaining_right);
                
        } 
    }
        

    } // end if brackets
    println!("went out of brackets proceeding with input: {}", input);
    if flg==2 {// перехват случаев с отрицательными величинами


    }
    
    // Обработка сложения и вычитания
    {
                                // we found '+' in the input only if it's not in brackets
    if let Some(pos) = find_char_positions_outside_brackets(input, '+') {
       
        let mut left: &str = &input[..pos].trim();
        let right = &input[pos + 1..].trim();
        println!("SIGN '+' found at positions {}-{}: left from +: {}, right from +: {}", pos, pos+1, left, right);
        println!("brackets found in posistions: {} - {},", brac_start, brac_end);
        println!("left {} has brackets -{}, right {} has brackets - {}", left, has_brackets(left), right,  has_brackets(right));   
    
        // if flag is 2 it means that left part of expression is negative
        let a = left.to_string();
        let new_left = format!("(-1.0)*{}", a);
        let new_left = &new_left.as_str();
        left = if flg==2 {new_left } else {left};

        // if left part is in brackets it will be proceeded by parse_expression_func by setting flg=1, otherwise flag is 1.
        return if left.starts_with("(") && left.ends_with(")") && !(right.starts_with("(") && right.ends_with(")"))
        {
            println!("left is brackets");
            Ok(Expr::Add(Box::new(parse_expression_func(1, left)?), Box::new(parse_expression_func(0, right)?)))
        }
        // same with the right part
        else if (right.starts_with("(") && right.ends_with(")")) && !(left.starts_with("(") && left.ends_with(")"))
        {
            println!("right is brackets");
            Ok(Expr::Add(Box::new(parse_expression_func(0, left)?), Box::new(parse_expression_func(1, right)?)))
        } else if (left.starts_with("(") && left.ends_with(")")) && (right.starts_with("(") && right.ends_with(")"))
        {
            println!("both sides have brackets");
            Ok(Expr::Add(Box::new(parse_expression_func(1, left)?), Box::new(parse_expression_func(1, right)?)))
        } else {  // both parts are not in brackets (but may have brackets inside)
            println!("neither has brackets");
            Ok(Expr::Add(Box::new(parse_expression_func(0, left)?), Box::new(parse_expression_func(0, right)?)))
        } 
   
    } else if let Some(pos) = find_char_positions_outside_brackets(input, '-'){
        let mut left = &input[..pos].trim();
        let right = &input[pos + 1..].trim();
        println!("sign '-' found - at positions {}-{}: left from minus: {}, right from minus: {}", pos, pos+1, left, right);
        println!("brackets found in posistions: {} - {},", brac_start, brac_end);

        let a = left.to_string();
        let new_left = format!("(-1.0)*{}", a);
        let new_left = &new_left.as_str();
        left = if flg==2 {new_left } else {left};

        return if left != &"" {
            if (left.starts_with("(") && left.ends_with(")")) && !(right.starts_with("(") && right.ends_with(")"))
            {
                println!("left in brackets");
                Ok(Expr::Sub(Box::new(parse_expression_func(1, left)?), Box::new(parse_expression_func(0, right)?)))
            } else if (right.starts_with("(") && right.ends_with(")")) && !(left.starts_with("(") && left.ends_with(")"))
            {
                println!("right in brackets");
                Ok(Expr::Sub(Box::new(parse_expression_func(0, left)?), Box::new(parse_expression_func(1, right)?)))
            } else if left.starts_with("(") && left.ends_with(")") && (right.starts_with("(") && right.ends_with(")"))
            {
                println!("both sides have brackets");
                Ok(Expr::Sub(Box::new(parse_expression_func(1, left)?), Box::new(parse_expression_func(1, right)?)))
            } else {
                println!("neither in brackets");
                Ok(Expr::Sub(Box::new(parse_expression_func(0, left)?), Box::new(parse_expression_func(0, right)?)))
            }
        } else { // case of negative values like -y, -x**2, etc
            println!("negative values found");
            if right.starts_with("(") && right.ends_with(")") { // when -(something..)
                Ok(Expr::Mul(Box::new(Expr::Const(-1.)), Box::new(parse_expression_func(1, right)?)))
            } else { // set the flag 2 - that means proceeding of negative values will be used in the next step of recursion
                let inner = parse_expression_func(2, right)?;
                Ok(inner)
            }
        }
    }
  
    // Обработка умножения и деления/Handling multiplication and division
    if let Some(pos) = find_char_positions_outside_brackets(input, '*') {
        let mut left = &input[..pos].trim();
        let right = &input[pos + 1..].trim();
        println!("SIGN '*' at positions {}-{}", pos, pos+1);
        println!("brackets found in posistions: {} - {},", brac_start, brac_end);


        let a = left.to_string();
        let new_left = format!("(-1.0)*{}", a);
        let new_left = &new_left.as_str();
        left = if flg==2 {new_left } else {left};
        return if (left.starts_with("(") & left.ends_with(")")) && !(right.starts_with("(") && right.ends_with(")"))
        {
            println!("left in brackets");
            Ok(Expr::Mul(Box::new(parse_expression_func(1, left)?), Box::new(parse_expression_func(0, right)?)))
        } else if (right.starts_with("(") && right.ends_with(")")) && !(left.starts_with("(") && left.ends_with(")"))
        {
            println!("right in brackets");
            Ok(Expr::Mul(Box::new(parse_expression_func(0, left)?), Box::new(parse_expression_func(1, right)?)))
        } else if (left.starts_with("(") && left.ends_with(")")) && (right.starts_with("(") && right.ends_with(")"))
        {
            println!("both sides have brackets");
            Ok(Expr::Mul(Box::new(parse_expression_func(1, left)?), Box::new(parse_expression_func(1, right)?)))
        } else {
            println!("neither in brackets");
            Ok(Expr::Mul(Box::new(parse_expression_func(0, left)?), Box::new(parse_expression_func(0, right)?)))
        } 

    } else if let Some(pos) = find_char_positions_outside_brackets(input, '/') {
        let mut left = &input[..pos].trim();
        let right = &input[pos + 1..].trim();
        println!("SIGN '/' at positions {}-{}", pos, pos+1);

        let a = left.to_string();
        let new_left = format!("(-1.0)*{}", a);
        let new_left = &new_left.as_str();
        left = if flg==2 {new_left } else {left};
        return if (left.starts_with("(") && left.ends_with(")")) && !(right.starts_with("(") && right.ends_with(")"))
        {
            println!("left in brackets");
            Ok(Expr::Div(Box::new(parse_expression_func(1, left)?), Box::new(parse_expression_func(0, right)?)))
        } else if (right.starts_with("(") && right.ends_with(")")) && !(left.starts_with("(") && left.ends_with(")"))
        {
            println!("right in brackets");
            Ok(Expr::Div(Box::new(parse_expression_func(0, left)?), Box::new(parse_expression_func(1, right)?)))
        } else if left.starts_with("(") && left.ends_with(")") && (right.starts_with("(") && right.ends_with(")"))
        {
            println!("both sides have brackets");
            Ok(Expr::Div(Box::new(parse_expression_func(1, left)?), Box::new(parse_expression_func(1, right)?)))
        } else {
            println!("neither in brackets");
            Ok(Expr::Div(Box::new(parse_expression_func(0, left)?), Box::new(parse_expression_func(0, right)?)))
        } 
        
    } else {println!("no matches among +, -, *, /")};
    // Обработка возведения в степень
    if let Some(pos) =  find_char_positions_outside_brackets(input, '^') {
        let base = &input[..pos].trim();
        let exponent = &input[pos + 1..].trim();
        println!("SIGN '^' at positions {}-{}", pos, pos+1);
        let base_expr = if base.chars().all(char::is_alphanumeric) {
            println!("base of power: {}", base);
            Expr::Var(base.to_string())
        } else  {
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
       let  fisrt_brac_end =  find_pair_to_this_bracket(input, 0);
        //let  fisrt_brac_end = input.find(')').unwrap();
        let inner = &input[4..fisrt_brac_end].trim();
        println!("SIGN 'exp' at positions {}-{}", fisrt_brac_end-4, fisrt_brac_end);
        return Ok(Expr::Exp(Box::new(parse_expression_func(0,inner)?)));
    } else if input.starts_with("log(") || input.starts_with("ln(") && input.ends_with(')') {
        let  fisrt_brac_end =  find_pair_to_this_bracket(input, 0);
      // let  fisrt_brac_end = input.find(')').unwrap();
        println!("SIGN 'log' at positions {}-{}", fisrt_brac_end-4, fisrt_brac_end);
        let inner = &input[4..fisrt_brac_end].trim();
        return Ok(Expr::Ln(Box::new(parse_expression_func(0,inner)?)));
    }

    // Обработка констант и переменных
    if let Ok(value) = input.parse::<f64>() {
        println!("found constant: {}", value);
        return if flg != 2 {
            Ok(Expr::Const(value))
        } else {
            Ok(Expr::Const(-value))
        }
    } else if input.chars().all(char::is_alphanumeric) {
        println!("found variable: {}", input);
        return if flg != 2 {
            Ok(Expr::Var(input.to_string()))
        } else {
            Ok(Expr::Mul(Box::new(Expr::Const(-1.0)), Box::new(Expr::Var(input.to_string()))))
        }
        
    }
    
    if  input.starts_with("(") && input.ends_with(')') { 
          let inner_str = &input[brac_start + 1..brac_end];
           let inner= parse_expression_func(0, inner_str)?;
        
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
            let expr = parse_expression_func(0,"exp(x)").unwrap();
            assert_eq!(expr, Expr::Exp(Box::new(Expr::Var("x".to_string()))));
        }
     

        #[test]
        fn test_parse_constant() {
            let expr = parse_expression_func(0,"42").unwrap();
            assert_eq!(expr, Expr::Const(42.0));
        }
    
        #[test]
        fn test_parse_variable() {
            let expr = parse_expression_func(0,"x").unwrap();
            assert_eq!(expr, Expr::Var("x".to_string()));
        }
    
        #[test]
        fn test_parse_addition() {
            let expr = parse_expression_func(0,"x + 2").unwrap();
            assert_eq!(expr, Expr::Add(Box::new(Expr::Var("x".to_string())), Box::new(Expr::Const(2.0))));
        }
    
        #[test]
        fn test_parse_subtraction() {
            let expr = parse_expression_func(0,"x - 2").unwrap();
            assert_eq!(expr, Expr::Sub(Box::new(Expr::Var("x".to_string())), Box::new(Expr::Const(2.0))));
        }
    
        #[test]
        fn test_parse_multiplication() {
            let expr = parse_expression_func(0,"x * 2").unwrap();
            assert_eq!(expr, Expr::Mul(Box::new(Expr::Var("x".to_string())), Box::new(Expr::Const(2.0))));
        }
    
        #[test]
        fn test_parse_division() {
            let expr = parse_expression_func(0,"x / 2").unwrap();
            assert_eq!(expr, Expr::Div(Box::new(Expr::Var("x".to_string())), Box::new(Expr::Const(2.0))));
        }
    
        #[test]
        fn test_parse_power() {
            let expr = parse_expression_func(0,"x^2").unwrap();
            assert_eq!(expr, Expr::Pow(Box::new(Expr::Var("x".to_string())), Box::new(Expr::Const(2.0))));
        }
    

    
        #[test]
        fn test_parse_logarithm() {
            let expr = parse_expression_func(0,"log(x)").unwrap();
            assert_eq!(expr, Expr::Ln(Box::new(Expr::Var("x".to_string()))));
        }
    
        #[test]
        fn test_parse_expression_func_with_brackets() {
            let expr = parse_expression_func(0,"(x + y) * z").unwrap();
            assert_eq!(
                expr,
                Expr::Mul(
                    Box::new(Expr::Add(Box::new(Expr::Var("x".to_string())), Box::new(Expr::Var("y".to_string())))),
                    Box::new(Expr::Var("z".to_string()))
                )
            );
        }
    
        #[test]
        fn test_parse_complex_expression() {
            let expr = parse_expression_func(0,"(x + y) * (z - 2) / exp(w)").unwrap();
            let x = Box::new(Expr::Var("x".to_string()));
            let y = Box::new(Expr::Var("y".to_string()));
            let z = Box::new(Expr::Var("z".to_string()));
            let w = Box::new(Expr::Var("w".to_string()));
            let C = Box::new(Expr::Const(2.0));
            let x_plus_y = Box::new(Expr::Add(x, y));
            let z_minus_C = Box::new(Expr::Sub(z, C));
            let e = Box::new(Expr::Exp(w));
            let z_minus_C_div_e = 
                Box::new(Expr::Div( z_minus_C, e));
            let Res = Expr::Mul(x_plus_y, z_minus_C_div_e);
            assert_eq!(
                expr,
                Res
            );

             
        }
       
        #[test]
        fn test_invalid_expression() {
            let result = parse_expression_func(0,"(x +");
            assert_eq!(result.is_err(),true);
        }
    
        #[test]
        fn test_unmatched_brackets() {
            
            let result = parse_expression_func(0,"(x + y");
            assert!(result.is_err());
        }
    }

     