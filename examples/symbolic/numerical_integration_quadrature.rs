use RustedSciThe::symbolic::symbolic_engine::Expr;
use RustedSciThe::symbolic::symbolic_integration::QuadMethod;

fn main() { 
    println!("=== Numerical Integration with Gaussian Quadrature ===\n");

    // Test ∫₀¹ x² dx = 1/3 using Gauss-Legendre
    let x = Expr::Var("x".to_string());
    let expr = x.clone().pow(Expr::Const(2.0));
    
    let result = expr.quad(QuadMethod::GaussLegendre, 10, 0.0, 1.0, None).unwrap();
    println!("∫₀¹ x² dx using Gauss-Legendre (10 points) = {}", result);
    println!("Expected: {}", 1.0 / 3.0);
    println!("Error: {}\n", (result - 1.0/3.0).abs());

    // Test ∫₋₁¹ (x³ + 2x² + x + 1) dx = 10/3
    let x = Expr::Var("x".to_string());
    let expr = x.clone().pow(Expr::Const(3.0))
        + Expr::Const(2.0) * x.clone().pow(Expr::Const(2.0))
        + x.clone()
        + Expr::Const(1.0);

    let result = expr.quad(QuadMethod::GaussLegendre, 15, -1.0, 1.0, None).unwrap();
    println!("∫₋₁¹ (x³ + 2x² + x + 1) dx using Gauss-Legendre = {}", result);
    println!("Expected: {}", 10.0 / 3.0);
    println!("Error: {}\n", (result - 10.0/3.0).abs());

    // Test ∫₀¹ e^x dx = e - 1
    let x = Expr::Var("x".to_string());
    let expr = x.clone().exp();

    let result = expr.quad(QuadMethod::GaussLegendre, 20, 0.0, 1.0, None).unwrap();
    let expected = std::f64::consts::E - 1.0;
    println!("∫₀¹ e^x dx using Gauss-Legendre (20 points) = {}", result);
    println!("Expected: {}", expected);
    println!("Error: {}\n", (result - expected).abs());

    // Compare with Simpson's rule
    let simpson_result = expr.numerical_integrate(0.0, 1.0, 1000);
    println!("Same integral using Simpson's rule (1000 intervals) = {}", simpson_result);
    println!("Difference between methods: {}\n", (result - simpson_result).abs());

    // Compare quadrature with analytical integration
    let x = Expr::Var("x".to_string());
    let expr = x.clone().pow(Expr::Const(3.0)) + Expr::Const(2.0) * x.clone();

    let analytical = expr.definite_integrate("x", 0.0, 2.0).unwrap();
    let numerical = expr.quad(QuadMethod::GaussLegendre, 20, 0.0, 2.0, None).unwrap();

    println!("∫₀² (x³ + 2x) dx:");
    println!("Analytical result: {}", analytical);
    println!("Gauss-Legendre result: {}", numerical);
    println!("Error: {}", (analytical - numerical).abs());

    // Show quadrature method descriptions
    println!("\n=== Available Quadrature Methods ===");
    println!("Gauss-Legendre: {}", QuadMethod::GaussLegendre.description());
    println!("Gauss-Hermite: {}", QuadMethod::GaussHermite.description());
    println!("Gauss-Laguerre: {}", QuadMethod::GaussLaguerre.description());
}