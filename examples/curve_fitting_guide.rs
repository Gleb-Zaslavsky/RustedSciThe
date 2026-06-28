//! Comprehensive guide for curve fitting with symbolic expressions
//!
//! This example demonstrates how to use the Fitting solver to fit various
//! mathematical models to data using both string-based and native symbolic
//! construction with the builder pattern.
//!
//! The solver automatically generates analytical Jacobians from symbolic
//! expressions for efficient and accurate parameter estimation.
//!
//! run cargo run --example curve_fitting_guide

use RustedSciThe::numerical::optimization::sym_fitting::Fitting;
use RustedSciThe::symbolic::symbolic_engine::Expr;

fn main() {
    println!("=== Curve Fitting Guide ===\n");

    // Example 1: Linear fitting with string equation
    example_1_linear_string();

    // Example 2: Linear fitting with native symbolic
    example_2_linear_native();

    // Example 3: Quadratic fitting with native symbolic
    example_3_quadratic_native();

    // Example 4: Exponential fitting
    example_4_exponential_native();

    // Example 5: Logarithmic fitting
    example_5_logarithmic_native();

    // Example 6: Trigonometric fitting
    example_6_trigonometric_native();

    // Example 7: Polynomial fitting (convenience method)
    example_7_polynomial();

    // Example 8: Power law fitting
    example_8_power_law_native();

    // Example 9: Rational function fitting
    example_9_rational_native();

    // Example 10: Complex multi-term fitting
    example_10_complex_native();

    println!("\n=== All examples completed ===");
}

fn example_1_linear_string() {
    println!("Example 1: Linear Fitting with String Equation");
    println!("-----------------------------------------------");
    println!("Model: y = a*x + b");
    println!("Data: y = 2x (perfect linear relationship)\n");

    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_data = vec![2.0, 4.0, 6.0, 8.0, 10.0];

    let fitting = Fitting::new()
        .with_data(x_data, y_data)
        .with_equation_str("a * x + b".to_string())
        .with_arg("x".to_string())
        .with_unknowns(vec!["a".to_string(), "b".to_string()])
        .with_initial_guess(vec![1.0, 1.0])
        .build();

    if let Some(map) = fitting.map_of_solutions {
        println!("Fitted parameters:");
        println!("  a (slope) = {:.6}", map["a"]);
        println!("  b (intercept) = {:.6}", map["b"]);
    }

    if let Some(r2) = fitting.r_ssquared {
        println!("  R² = {:.6}", r2);
    }
    println!();
}

fn example_2_linear_native() {
    println!("Example 2: Linear Fitting with Native Symbolic");
    println!("-----------------------------------------------");
    println!("Model: y = a*x + b (using native Expr construction)\n");

    let x_data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y_data = vec![1.0, 3.0, 5.0, 7.0, 9.0];

    // Native symbolic construction
    let vars = Expr::Symbols("a, x, b");
    let a = vars[0].clone();
    let x = vars[1].clone();
    let b = vars[2].clone();
    let eq = a * x + b;

    let fitting = Fitting::new()
        .with_data(x_data, y_data)
        .with_equation(eq)
        .with_arg("x".to_string())
        .with_unknowns(vec!["a".to_string(), "b".to_string()])
        .with_initial_guess(vec![1.0, 1.0])
        .build();

    if let Some(map) = fitting.map_of_solutions {
        println!("Fitted parameters:");
        println!("  a = {:.6}", map["a"]);
        println!("  b = {:.6}", map["b"]);
        println!("  Model: y = {:.3}x + {:.3}", map["a"], map["b"]);
    }
    println!();
}

fn example_3_quadratic_native() {
    println!("Example 3: Quadratic Fitting with Native Symbolic");
    println!("--------------------------------------------------");
    println!("Model: y = a*x² + b*x + c\n");

    let x_data = (0..30).map(|x| x as f64).collect::<Vec<f64>>();
    let y_data = x_data
        .iter()
        .map(|&x| 2.0 * x * x + 3.0 * x + 1.0)
        .collect::<Vec<f64>>();

    // Native symbolic construction
    let vars = Expr::Symbols("a, b, c, x");
    let a = vars[0].clone();
    let b = vars[1].clone();
    let c = vars[2].clone();
    let x = vars[3].clone();

    let eq = a * x.clone().pow(Expr::Const(2.0)) + b * x + c;

    let fitting = Fitting::new()
        .with_data(x_data, y_data)
        .with_equation(eq)
        .with_arg("x".to_string())
        .with_unknowns(vec!["a".to_string(), "b".to_string(), "c".to_string()])
        .with_initial_guess(vec![1.0, 1.0, 1.0])
        .build();

    if let Some(map) = fitting.map_of_solutions {
        println!("Fitted parameters:");
        println!("  a = {:.6}", map["a"]);
        println!("  b = {:.6}", map["b"]);
        println!("  c = {:.6}", map["c"]);
        println!(
            "  Model: y = {:.3}x² + {:.3}x + {:.3}",
            map["a"], map["b"], map["c"]
        );
    }

    if let Some(r2) = fitting.r_ssquared {
        println!("  R² = {:.6}", r2);
    }
    println!();
}

fn example_4_exponential_native() {
    println!("Example 4: Exponential Fitting");
    println!("-------------------------------");
    println!("Model: y = a*exp(b*x) + c\n");

    let x_data = (0..25).map(|x| x as f64 * 0.2).collect::<Vec<f64>>();
    let y_data = x_data
        .iter()
        .map(|&x| 2.0 * (0.5 * x).exp() + 1.0)
        .collect::<Vec<f64>>();

    // Native symbolic construction
    let vars = Expr::Symbols("a, b, c, x");
    let a = vars[0].clone();
    let b = vars[1].clone();
    let c = vars[2].clone();
    let x = vars[3].clone();

    let eq = a * Expr::exp(b * x) + c;

    let fitting = Fitting::new()
        .with_data(x_data, y_data)
        .with_equation(eq)
        .with_arg("x".to_string())
        .with_unknowns(vec!["a".to_string(), "b".to_string(), "c".to_string()])
        .with_initial_guess(vec![1.5, 0.3, 0.5])
        .with_tolerance(1e-8)
        .build();

    if let Some(map) = fitting.map_of_solutions {
        println!("Fitted parameters:");
        println!("  a = {:.6}", map["a"]);
        println!("  b = {:.6}", map["b"]);
        println!("  c = {:.6}", map["c"]);
        println!(
            "  Model: y = {:.3}*exp({:.3}*x) + {:.3}",
            map["a"], map["b"], map["c"]
        );
    }

    if let Some(r2) = fitting.r_ssquared {
        println!("  R² = {:.6}", r2);
    }
    println!();
}

fn example_5_logarithmic_native() {
    println!("Example 5: Logarithmic Fitting");
    println!("-------------------------------");
    println!("Model: y = a*ln(x) + b\n");

    let x_data = (1..40).map(|x| x as f64).collect::<Vec<f64>>();
    let y_data = x_data
        .iter()
        .map(|&x| 3.0 * x.ln() + 2.0)
        .collect::<Vec<f64>>();

    // Native symbolic construction
    let vars = Expr::Symbols("a, x, b");
    let a = vars[0].clone();
    let x = vars[1].clone();
    let b = vars[2].clone();

    let eq = a * Expr::ln(x) + b;

    let fitting = Fitting::new()
        .with_data(x_data, y_data)
        .with_equation(eq)
        .with_arg("x".to_string())
        .with_unknowns(vec!["a".to_string(), "b".to_string()])
        .with_initial_guess(vec![2.0, 1.0])
        .build();

    if let Some(map) = fitting.map_of_solutions {
        println!("Fitted parameters:");
        println!("  a = {:.6}", map["a"]);
        println!("  b = {:.6}", map["b"]);
        println!("  Model: y = {:.3}*ln(x) + {:.3}", map["a"], map["b"]);
    }

    if let Some(r2) = fitting.r_ssquared {
        println!("  R² = {:.6}", r2);
    }
    println!();
}

fn example_6_trigonometric_native() {
    println!("Example 6: Trigonometric Fitting");
    println!("---------------------------------");
    println!("Model: y = a*sin(b*x) + c\n");

    use std::f64::consts::PI;
    let x_data = (0..100)
        .map(|x| x as f64 * 2.0 * PI / 100.0)
        .collect::<Vec<f64>>();
    let y_data = x_data
        .iter()
        .map(|&x| 3.0 * (2.0 * x).sin() + 1.0)
        .collect::<Vec<f64>>();

    // Native symbolic construction
    let vars = Expr::Symbols("a, b, c, x");
    let a = vars[0].clone();
    let b = vars[1].clone();
    let c = vars[2].clone();
    let x = vars[3].clone();

    let eq = a * Expr::sin(Box::new(b * x)) + c;

    let fitting = Fitting::new()
        .with_data(x_data, y_data)
        .with_equation(eq)
        .with_arg("x".to_string())
        .with_unknowns(vec!["a".to_string(), "b".to_string(), "c".to_string()])
        .with_initial_guess(vec![2.0, 1.5, 0.5])
        .with_tolerance(1e-8)
        .with_g_tolerance(1e-8)
        .with_f_tolerance(1e-8)
        .build();

    if let Some(map) = fitting.map_of_solutions {
        println!("Fitted parameters:");
        println!("  a (amplitude) = {:.6}", map["a"]);
        println!("  b (frequency) = {:.6}", map["b"]);
        println!("  c (offset) = {:.6}", map["c"]);
        println!(
            "  Model: y = {:.3}*sin({:.3}*x) + {:.3}",
            map["a"], map["b"], map["c"]
        );
    }

    if let Some(r2) = fitting.r_ssquared {
        println!("  R² = {:.6}", r2);
    }
    println!();
}

fn example_7_polynomial() {
    println!("Example 7: Polynomial Fitting (Convenience Method)");
    println!("---------------------------------------------------");
    println!("Model: y = c₃*x³ + c₂*x² + c₁*x + c₀\n");

    let x_data = (0..40).map(|x| x as f64 * 0.1).collect::<Vec<f64>>();
    let y_data = x_data
        .iter()
        .map(|&x| 2.0 * x.powi(3) + 1.5 * x.powi(2) + 3.0 * x + 0.5)
        .collect::<Vec<f64>>();

    let fitting = Fitting::new()
        .with_data(x_data, y_data)
        .with_polynomial(3, "x".to_string())
        .with_initial_guess(vec![1.0, 1.0, 1.0, 1.0])
        .build();

    if let Some(map) = fitting.map_of_solutions {
        println!("Fitted parameters:");
        println!("  c₃ = {:.6}", map["c3"]);
        println!("  c₂ = {:.6}", map["c2"]);
        println!("  c₁ = {:.6}", map["c1"]);
        println!("  c₀ = {:.6}", map["c0"]);
    }

    if let Some(r2) = fitting.r_ssquared {
        println!("  R² = {:.6}", r2);
    }
    println!();
}

fn example_8_power_law_native() {
    println!("Example 8: Power Law Fitting");
    println!("-----------------------------");
    println!("Model: y = a*x^b + c");
    println!("Note: Power law fitting is sensitive to initial guesses\n");

    let x_data = (1..50).map(|x| x as f64).collect::<Vec<f64>>();
    let y_data = x_data
        .iter()
        .map(|&x| 2.0 * x.powf(0.5) + 1.0)
        .collect::<Vec<f64>>();

    // Native symbolic construction
    let vars = Expr::Symbols("a, b, c, x");
    let a = vars[0].clone();
    let b = vars[1].clone();
    let c = vars[2].clone();
    let x = vars[3].clone();

    let eq = a * x.pow(b) + c;

    let fitting = Fitting::new()
        .with_data(x_data, y_data)
        .with_equation(eq)
        .with_arg("x".to_string())
        .with_unknowns(vec!["a".to_string(), "b".to_string(), "c".to_string()])
        .with_initial_guess(vec![1.5, 0.4, 0.5])
        .with_tolerance(1e-6)
        .with_max_iterations(300)
        .build();

    if let Some(map) = fitting.map_of_solutions {
        println!("Fitted parameters:");
        println!("  a = {:.6}", map["a"]);
        println!("  b = {:.6}", map["b"]);
        println!("  c = {:.6}", map["c"]);
        println!(
            "  Model: y = {:.3}*x^{:.3} + {:.3}",
            map["a"], map["b"], map["c"]
        );
    }

    if let Some(r2) = fitting.r_ssquared {
        println!("  R² = {:.6}", r2);
    }
    println!();
}

fn example_9_rational_native() {
    println!("Example 9: Rational Function Fitting");
    println!("-------------------------------------");
    println!("Model: y = (a*x + b) / (x + c)\n");

    let x_data = (1..40).map(|x| x as f64).collect::<Vec<f64>>();
    let y_data = x_data
        .iter()
        .map(|&x| (3.0 * x + 2.0) / (x + 1.5))
        .collect::<Vec<f64>>();

    // Native symbolic construction
    let vars = Expr::Symbols("a, b, c, x");
    let a = vars[0].clone();
    let b = vars[1].clone();
    let c = vars[2].clone();
    let x = vars[3].clone();

    let eq = (a * x.clone() + b) / (x + c);

    let fitting = Fitting::new()
        .with_data(x_data, y_data)
        .with_equation(eq)
        .with_arg("x".to_string())
        .with_unknowns(vec!["a".to_string(), "b".to_string(), "c".to_string()])
        .with_initial_guess(vec![2.5, 1.5, 1.0])
        .with_tolerance(1e-8)
        .build();

    if let Some(map) = fitting.map_of_solutions {
        println!("Fitted parameters:");
        println!("  a = {:.6}", map["a"]);
        println!("  b = {:.6}", map["b"]);
        println!("  c = {:.6}", map["c"]);
        println!(
            "  Model: y = ({:.3}*x + {:.3}) / (x + {:.3})",
            map["a"], map["b"], map["c"]
        );
    }

    if let Some(r2) = fitting.r_ssquared {
        println!("  R² = {:.6}", r2);
    }
    println!();
}

fn example_10_complex_native() {
    println!("Example 10: Complex Multi-Term Fitting");
    println!("---------------------------------------");
    println!("Model: y = a*ln(x) + b*exp(c*x) + d\n");

    let x_data = (1..30).map(|x| x as f64 * 0.1).collect::<Vec<f64>>();
    let y_data = x_data
        .iter()
        .map(|&x| 2.0 * x.ln() + 0.5 * (0.3 * x).exp() + 1.0)
        .collect::<Vec<f64>>();

    // Native symbolic construction with multiple operations
    let vars = Expr::Symbols("a, b, c, d, x");
    let a = vars[0].clone();
    let b = vars[1].clone();
    let c = vars[2].clone();
    let d = vars[3].clone();
    let x = vars[4].clone();

    let eq = a * Expr::ln(x.clone()) + b * Expr::exp(c * x) + d;

    let fitting = Fitting::new()
        .with_data(x_data, y_data)
        .with_equation(eq)
        .with_arg("x".to_string())
        .with_unknowns(vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ])
        .with_initial_guess(vec![1.5, 0.3, 0.2, 0.5])
        .with_tolerance(1e-6)
        .with_max_iterations(300)
        .build();

    if let Some(map) = fitting.map_of_solutions.as_ref() {
        println!("Fitted parameters:");
        println!("  a = {:.6}", map["a"]);
        println!("  b = {:.6}", map["b"]);
        println!("  c = {:.6}", map["c"]);
        println!("  d = {:.6}", map["d"]);
        println!(
            "  Model: y = {:.3}*ln(x) + {:.3}*exp({:.3}*x) + {:.3}",
            map["a"], map["b"], map["c"], map["d"]
        );
    }

    if let Some(r2) = fitting.r_ssquared {
        println!("  R² = {:.6}", r2);
    }

    // Demonstrate extrapolation
    println!("\n  Extrapolation example:");
    let x_extra = vec![3.5, 4.0, 4.5];
    let y_extra = fitting.extra_interpolate(x_extra.clone());
    for (x, y) in x_extra.iter().zip(y_extra.iter()) {
        println!("    x = {:.1}, predicted y = {:.6}", x, y);
    }
    println!();
}
