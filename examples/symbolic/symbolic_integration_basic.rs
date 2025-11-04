use RustedSciThe::symbolic::symbolic_engine::Expr;

fn main() {
    println!("=== Basic Symbolic Integration Examples ===\n");

    // ∫ 5 dx = 5x
    let expr = Expr::Const(5.0);
    let result = expr.integrate("x").unwrap();
    println!("∫ 5 dx = {}", result);

    // ∫ x dx = x²/2
    let expr = Expr::Var("x".to_string());
    let result = expr.integrate("x").unwrap();
    println!("∫ x dx = {}", result);

    // ∫ x³ dx = x⁴/4
    let x = Expr::Var("x".to_string());
    let expr = x.clone().pow(Expr::Const(3.0));
    let result = expr.integrate("x").unwrap();
    println!("∫ x³ dx = {}", result);

    // ∫ 1/x dx = ln|x|
    let expr = Expr::Const(1.0) / Expr::Var("x".to_string());
    let result = expr.integrate("x").unwrap();
    println!("∫ 1/x dx = {}", result);

    // ∫ (x + 3) dx = x²/2 + 3x
    let expr = Expr::Var("x".to_string()) + Expr::Const(3.0);
    let result = expr.integrate("x").unwrap();
    println!("∫ (x + 3) dx = {}", result);

    // Test evaluation at a point
    let x_val = 2.0;
    let result_val = result.eval_expression(vec!["x"], &[x_val]);
    println!("Evaluated at x = {}: {}", x_val, result_val);

    // ∫ (x² - x) dx = x³/3 - x²/2
    let x = Expr::Var("x".to_string());
    let expr = x.clone().pow(Expr::Const(2.0)) - x.clone();
    let result = expr.integrate("x").unwrap();
    println!("∫ (x² - x) dx = {}", result);
}