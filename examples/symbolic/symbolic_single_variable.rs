use RustedSciThe::symbolic::symbolic_engine::Expr;

fn main() {
    //  FUNTION OF 1 VARIABLE (processing of them has a slightly easier syntax then for multiple variables)
    // function of 1 argument (1D examples)
    let input = "log(x)";
    let f = Expr::parse_expression(input);
    //convert symbolic expression to a Rust function and evaluate the function
    let f_res = f.lambdify1D()(1.0);
    let df_dx = f.diff("x");
    println!("df_dx = {}, log(1) = {}", df_dx, f_res);

    let input = "x+exp(x)";
    let f = Expr::parse_expression(input);
    let f_res = f.lambdify1D()(1.0);
    println!("f_res = {}", f_res);
    let start = 0.0;
    let end = 10f64;
    let num_values = 100;
    let max_norm = 1e-6;
    // compare numerical and analtical derivatives for a given linspace defined by start, end values and number of values.
    // a norm of the difference between the two of them is returned, and the answer is true if the norm is below max_norm
    let (norm, res) = f.compare_num1D("x", start, end, num_values, max_norm);
    println!("norm = {}, res = {}", norm, res);
}