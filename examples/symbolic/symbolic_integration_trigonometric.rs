use RustedSciThe::symbolic::symbolic_engine::Expr;

fn main() {
    println!("=== Trigonometric Function Integration Examples ===\n");

    // ∫ sin(x) dx = -cos(x)
    let x = Expr::Var("x".to_string());
    let expr = Expr::sin(Box::new(x.clone()));
    let result = expr.integrate("x").unwrap();
    println!("∫ sin(x) dx = {}", result);

    // ∫ cos(x) dx = sin(x)
    let x = Expr::Var("x".to_string());
    let expr = Expr::cos(Box::new(x.clone()));
    let result = expr.integrate("x").unwrap();
    println!("∫ cos(x) dx = {}", result);

    // ∫ sin(2x) dx = -(1/2)*cos(2x)
    let x = Expr::Var("x".to_string());
    let expr = Expr::sin(Box::new(Expr::Const(2.0) * x.clone()));
    let result = expr.integrate("x").unwrap();
    println!("∫ sin(2x) dx = {}", result);

    // ∫ cos(3x) dx = (1/3)*sin(3x)
    let x = Expr::Var("x".to_string());
    let expr = Expr::cos(Box::new(Expr::Const(3.0) * x.clone()));
    let result = expr.integrate("x").unwrap();
    println!("∫ cos(3x) dx = {}", result);

    // Test evaluation at specific points
    let x_val = std::f64::consts::PI / 6.0; // π/6
    let result_val = result.eval_expression(vec!["x"], &[x_val]);
    println!("Evaluated at x = π/6: {}", result_val);

    // ∫ tan(x) dx = -ln|cos(x)|
    let x = Expr::Var("x".to_string());
    let expr = Expr::tg(Box::new(x.clone()));
    let result = expr.integrate("x").unwrap();
    println!("\n∫ tan(x) dx = {}", result);

    // ∫ cot(x) dx = ln|sin(x)|
    let x = Expr::Var("x".to_string());
    let expr = Expr::ctg(Box::new(x.clone()));
    let result = expr.integrate("x").unwrap();
    println!("∫ cot(x) dx = {}", result);

    // Definite integral example: ∫₀^(π/2) sin(x) dx = 1
    let x = Expr::Var("x".to_string());
    let expr = Expr::sin(Box::new(x.clone()));
    let definite = expr.definite_integrate("x", 0.0, std::f64::consts::PI / 2.0).unwrap();
    println!("\n∫₀^(π/2) sin(x) dx = {}", definite);
    println!("Expected: 1.0");
    println!("Error: {}", (definite - 1.0).abs());
}