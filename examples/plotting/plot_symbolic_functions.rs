use RustedSciThe::Utils::bevy_2d::plot_static_multiplot;
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::{DMatrix, DVector};

fn main() {
    println!("Plotting functions using RustedSciThe symbolic engine...");

    // Define symbolic expressions
    let expressions = vec![
        "sin(x) + 0.5*cos(2*x)",
        "exp(-x/5)*sin(x)",
        "x^2*exp(-x)",
        "ln(x+1)",
    ];

    let names: Vec<String> = expressions.iter().map(|s| s.to_string()).collect();

    // Generate x values
    let n = 250;
    let x_start = 0.1;
    let x_end = 10.0;
    let x = DVector::from_iterator(
        n,
        (0..n).map(|i| x_start + (x_end - x_start) * i as f64 / (n - 1) as f64),
    );

    // Evaluate each symbolic expression
    let mut ys = DMatrix::zeros(expressions.len(), n);

    for (expr_idx, expr_str) in expressions.iter().enumerate() {
        println!("Parsing and evaluating: {}", expr_str);

        let expr = Expr::parse_expression(expr_str);
        let func = expr.lambdify1D();

        for i in 0..n {
            ys[(expr_idx, i)] = func(x[i]);
        }
    }

    println!("Launching interactive plot...");
    println!("Controls:");
    println!("- Left mouse button: Pan");
    println!("- Scroll wheel: Zoom");

    plot_static_multiplot(x, ys, names);
}
