
// the collection of utility functions mainly for bracket parsing and proceeding
pub fn has_brackets(s: &str) -> bool {
    let mut stack = Vec::new();
    let mut has_brackets = false;

    for c in s.chars() {
        match c {
            '(' | '{' | '[' => {
                stack.push(c);
                has_brackets = true;
            }
            ')' => {
                if stack.pop() != Some('(') {
                    return false;
                }
                has_brackets = true            }
            '}' => {
                if stack.pop() != Some('{') {
                    return false;
                }
                has_brackets = true;
            }
            ']' => {
                if stack.pop() != Some('[') {
                    return false;
                }
                has_brackets = true;
            }
            _ => {}
        }
    }

    has_brackets && stack.is_empty()
}


pub fn any_brackets(s: &str) -> bool {

    !s.chars().any(|c| c == '(' || c == '{' || c == '[' || c == ')' || c == '}' || c == ']') 
}

pub fn find_char_positions(input: &str, target_char: char) -> Vec<usize> {
    let mut positions = Vec::new();
    let mut start_pos = 0;

    while let Some(pos) = input[start_pos..].find(target_char) {
        positions.push(start_pos + pos);
        start_pos += pos + 1;
    }

    positions
}


pub fn find_char_positions_but_not_inside_brackets(input: &str, target_char: char) -> Vec<usize> {
    let mut positions = Vec::new();
    let mut start_pos = 0;
    let mut inside_brackets = false;

    while let Some(pos) = input[start_pos..].find(target_char) {
        let char_index = start_pos + pos;

        if !inside_brackets {
            if let Some(opening_bracket) = input[..char_index].chars().rfind(|c| c == &'(' || c == &'{' || c == &'[') {
                if let Some(closing_bracket) = input[char_index..].chars().find(|c| c == &')' || c == &'}' || c == &']') {
                    if opening_bracket == '(' && closing_bracket == ')' ||
                       opening_bracket == '{' && closing_bracket == '}' ||
                       opening_bracket == '[' && closing_bracket == ']' {
                        inside_brackets = true;
                    }
                }
            } else {
                positions.push(char_index);
            }
        }

        start_pos += pos + 1;
    }

    positions
}


// find positions of giving char that are  outside brackets only
pub fn find_char_positions_outside_brackets(s: &str, c: char) -> Option<usize> {
    let mut stack = Vec::new();
    let mut positions = Vec::new();
    let _positions_sorted: Vec<usize> = Vec::new();
    for (i, ch) in s.chars().enumerate() {
        if ch == '(' {
            stack.push(i);
        } else if ch == ')' {
            if let Some(_) = stack.pop() {
                // do nothing
            }
        } else if ch == c && stack.is_empty() {
            positions.push(i);
        }
    }
    //let pos =  positions.first().unwrap_or(&0);
 //  _positions_sorted = positions.clone().sort_unstable().cloned();
   
   let res = if  positions.is_empty(){
         None
    } else {
        let pos = *positions.first().unwrap();
        Some(pos)
    };
    res
    //println!("+ at pos: {}", pos);
   // Some(*pos)
}
// code finds the position of 
pub fn find_pair_to_this_bracket(input:&str, bracket_start:usize) -> usize {
    println!("finding closing bracket of {}", input);
    let mut stack = bracket_start;
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
    bracket_end.unwrap()
    
}


pub fn linspace(start: f64, end: f64, num_values: usize) -> Vec<f64> {
    let mut values = Vec::with_capacity(num_values);
    let step = (end - start) / (num_values as f64 - 1.0);

    for i in 0..num_values {
        let value = start + (i as f64 * step);
        values.push(value);
    }

    values
}
/*

    // Define a vector of argument values
    let x_values = vec![0.0, 1.0, 2.0, 3.0, 4.0];

    // Define the step size for the numerical derivative
    let h = 0.001;

    // Compute the numerical derivative
    let derivatives = numerical_derivative(f, x_values, h);
*/
pub fn numerical_derivative<F>(f: F, x_values: Vec<f64>, h: f64) -> Vec<f64>
where
    F: Fn(f64) -> f64,
{
    let mut derivatives = Vec::with_capacity(x_values.len());

    for &x in &x_values {
        let f_x_plus_h = f(x + h);
        let f_x_minus_h = f(x - h);
        let derivative = (f_x_plus_h - f_x_minus_h) / (2.0 * h);
        derivatives.push(derivative);
    }

    derivatives
}
//  y = f(x, y), dy = (f(x+h, y) + f(x-h, y))/2 + (f(x, y+h) + f(x, y-h))/2
//dy = f
pub fn numerical_derivative_multi<F>(f: F, x_values: Vec<f64>, h: f64) -> Vec<f64>
where
    F: Fn(Vec<f64>) -> f64,
{
    let mut derivatives = Vec::with_capacity(x_values.len());
    for i in 0..x_values.len() {
        let mut x_plus_h = x_values.clone();
        let mut x_minus_h = x_values.clone();

        x_plus_h[i] += h;
        x_minus_h[i] -= h;

        let f_x_plus_h = f(x_plus_h);
        let f_x_minus_h = f(x_minus_h);

        let derivative = (f_x_plus_h - f_x_minus_h) / (2.0 * h);
        derivatives.push(derivative);
    }

    derivatives
}



// compute norm of two vectors
pub fn norm(x: Vec<f64>, y: Vec<f64>) -> f64 {
    assert_eq!(x.len(), y.len());
    let norm_res = (1.0 / x.len() as f64)* x.iter().zip(y.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt() ;
    norm_res
}


// transpose a matrix  
pub fn transpose<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>> {
    assert!(!v.is_empty());
    let len = v[0].len();
    let mut iters: Vec<_> = v.into_iter().map(std::iter::IntoIterator::into_iter).collect();
    (0..len)
        .map(|_| iters.iter_mut().map(|n| n.next().unwrap()).collect())
        .collect()
}

pub fn unbox<T>(value: Box<T>) -> T {
    *value
}

/*

let num_cols = matrix.first().unwrap().len();
let mut row_iters: Vec<_> = matrix.into_iter().map(Vec::into_iter).collect();
let mut out: Vec<Vec<_>> = (0..num_cols).map(|_| Vec::new()).collect();

for out_row in out.iter_mut() {
    for it in row_iters.iter_mut() {
        out_row.push(it.next().unwrap());
    }
}

println!("{:?}", out)

*/