use RustedSciThe::symbolic::symbolic_engine::Expr;

fn main() {
    // FUNCTION OF MULTIPLE VARIABLES
    //parse expression from string to symbolic expression
    let input = "exp(x)+log(y)"; //log(x)/y-x^2.3 *log(x+y+y^2.6)-exp(x-y)/(x+y) +  (log((x-y)/(x+y)))^2
    // here you've got symbolic expression
    let parsed_expression = Expr::parse_expression(input);
    println!(" parsed_expression {}", parsed_expression);
    // turn symbolic expression to a pretty human-readable string
    let parsed_function = parsed_expression.sym_to_str("x");
    println!("{}, sym to string: {}  \n", input, parsed_function);
    // return vec of all arguments
    let all = parsed_expression.all_arguments_are_variables();
    println!("all arguments are variables {:?}", all);
    let variables = parsed_expression.extract_variables();
    println!("variables {:?}", variables);

    // differentiate with respect to x and y
    let df_dx = parsed_expression.diff("x");
    let df_dy = parsed_expression.diff("y");
    println!("df_dx = {}, df_dy = {}", df_dx, df_dy);
    //convert symbolic expression to a Rust function and evaluate the function
    let args = vec!["x", "y"];
    let function_of_x_and_y = parsed_expression.lambdify_borrowed_thread_safe(args.as_slice());
    let f_res = function_of_x_and_y(&[1.0, 2.0]);
    println!("f_res = {}", f_res);
    // or you dont want to pass arguments you can use lambdify_wrapped, arguments will be found inside function
    let function_of_x_and_y = parsed_expression.lambdify_wrapped();
    let f_res = function_of_x_and_y(vec![1.0, 2.0]);
    println!("f_res2 = {}", f_res);

    let start = vec![1.0, 1.0];
    let end = vec![2.0, 2.0];
    // evaluate function of 2 or more arguments using linspace for defining vectors of arguments
    let result = parsed_expression.lamdified_from_linspace(start.clone(), end.clone(), 10);
    println!("evaluated function of 2 arguments = {:?}", result);
    //  find vector of derivatives with respect to all arguments
    let vector_of_derivatives = parsed_expression.diff_multi();
    println!(
        "vector_of_derivatives = {:?}, {}",
        vector_of_derivatives,
        vector_of_derivatives.len()
    );
    // compare numerical and analtical derivatives for a given linspace defined by start, end values and number of values.
    // max_norm - maximum norm of the difference between numerical and analtical derivatives
    let comparsion = parsed_expression.compare_num(start, end, 100, 1e-6);
    println!(" result_of compare = {:?}", comparsion);
}