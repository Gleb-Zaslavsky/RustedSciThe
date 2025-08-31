use RustedSciThe::symbolic::symbolic_engine::Expr;

fn main() {
    // Test parsing various trigonometric functions
    let test_cases = vec![
        "sin(x)",
        "cos(x)",
        "tg(x)",
        "tan(x)",
        "ctg(x)",
        "cot(x)",
        "arcsin(x)",
        "asin(x)",
        "arccos(x)",
        "acos(x)",
        "arctg(x)",
        "atan(x)",
        "arctan(x)",
        "arcctg(x)",
        "acot(x)",
        "sin(x) + cos(y)",
        "sin(cos(x))",
        "sin(x^2) * cos(y + 1)",
    ];

    println!("Testing trigonometric function parsing:");
    println!("=====================================");

    for test_case in test_cases {
        let expr = Expr::parse_expression(test_case);
        println!("✓ '{}' -> {}", test_case, expr);

        // Test if we can differentiate it
        if test_case.contains('x') {
            let derivative = expr.diff("x");
            println!("  d/dx: {}", derivative);
        }

        // Test if we can convert to function (for simple cases)
        if test_case == "sin(x)" {
            let func = expr.lambdify1D();
            let result = func(std::f64::consts::PI / 2.0);
            println!("  sin(π/2) = {}", result);
        }

        println!();
    }
}
