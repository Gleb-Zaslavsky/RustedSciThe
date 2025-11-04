use RustedSciThe::symbolic::symbolic_engine::Expr;

fn main() {
    println!("=== Integration by Parts Examples ===\n");

    // ∫ x * e^x dx using integration by parts
    let x = Expr::Var("x".to_string());
    let expr = x.clone() * x.clone().exp();
    let result = expr.integrate("x").unwrap();
    println!("∫ x * e^x dx = {}", result);

    // Test evaluation at several points
    let test_points = vec![0.0, 1.0, 0.5, -0.5];
    for point in test_points {
        let result_val = result.eval_expression(vec!["x"], &[point]);
        println!("  At x = {}: {}", point, result_val);
    }

    // ∫ x * e^(2x) dx = e^(2x)(x/2 - 1/4)
    let x = Expr::Var("x".to_string());
    let expr = x.clone() * (Expr::Const(2.0) * x.clone()).exp();
    let result = expr.integrate("x").unwrap();
    println!("\n∫ x * e^(2x) dx = {}", result);

    // Test definite integral ∫₀¹ x*e^(2x) dx
    let definite = expr.definite_integrate("x", 0.0, 1.0).unwrap();
    println!("∫₀¹ x * e^(2x) dx = {}", definite);

    // ∫ x² * e^(2x) dx using recursive integration by parts
    let x = Expr::Var("x".to_string());
    let expr = x.clone().pow(Expr::Const(2.0)) * (Expr::Const(2.0) * x.clone()).exp();
    let result = expr.integrate("x").unwrap();
    println!("\n∫ x² * e^(2x) dx = {}", result);

    // ∫ x * ln(x) dx using integration by parts
    let x = Expr::Var("x".to_string());
    let expr = x.clone() * x.clone().ln();
    let result = expr.integrate("x").unwrap();
    println!("\n∫ x * ln(x) dx = {}", result);

    // Test definite integral ∫₁² x*ln(x) dx
    let definite = expr.definite_integrate("x", 1.0, 2.0).unwrap();
    println!("∫₁² x * ln(x) dx = {}", definite);

    // ∫ x² * ln(x) dx
    let x = Expr::Var("x".to_string());
    let expr = x.clone().pow(Expr::Const(2.0)) * x.clone().ln();
    let result = expr.integrate("x").unwrap();
    println!("\n∫ x² * ln(x) dx = {}", result);

    // Test definite integral ∫₁² x²*ln(x) dx
    let definite = expr.definite_integrate("x", 1.0, 2.0).unwrap();
    println!("∫₁² x² * ln(x) dx = {}", definite);
}