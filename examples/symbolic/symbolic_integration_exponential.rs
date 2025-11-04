use RustedSciThe::symbolic::symbolic_engine::Expr;

fn main() {
    println!("=== Exponential and Logarithmic Integration Examples ===\n");

    // ∫ e^x dx = e^x
    let x = Expr::Var("x".to_string());
    let expr = x.clone().exp();
    let result = expr.integrate("x").unwrap();
    println!("∫ e^x dx = {}", result);

    // ∫ e^(2x) dx = (1/2) * e^(2x)
    let x = Expr::Var("x".to_string());
    let expr = (Expr::Const(2.0) * x.clone()).exp();
    let result = expr.integrate("x").unwrap();
    println!("∫ e^(2x) dx = {}", result);

    // Test evaluation at a point
    let x_val = 0.5;
    let result_val = result.eval_expression(vec!["x"], &[x_val]);
    println!("Evaluated at x = {}: {}", x_val, result_val);

    // ∫ ln(x) dx = x*ln(x) - x
    let x = Expr::Var("x".to_string());
    let expr = x.clone().ln();
    let result = expr.integrate("x").unwrap();
    println!("∫ ln(x) dx = {}", result);

    // Test evaluation at a point (x > 0)
    let x_val = std::f64::consts::E;
    let result_val = result.eval_expression(vec!["x"], &[x_val]);
    println!("Evaluated at x = e: {}", result_val);

    // Definite integration example
    // ∫₁² ln(x) dx
    let definite = expr.definite_integrate("x", 1.0, 2.0).unwrap();
    println!("∫₁² ln(x) dx = {}", definite);
}