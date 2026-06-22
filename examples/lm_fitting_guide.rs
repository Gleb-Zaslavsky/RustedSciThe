//! LM fitting guide.
//!
//! This guide shows the classic symbolic least-squares workflow:
//! - describe the model with symbolic expressions,
//! - provide x/y data,
//! - choose unknown parameter names and an initial guess,
//! - call `build()`, which generates the Jacobian and runs LM.
//!
//! run cargo run --example lm_fitting_guide

use RustedSciThe::numerical::optimization::sym_fitting::Fitting;
use RustedSciThe::symbolic::symbolic_engine::Expr;

fn main() {
    println!("=== LM Fitting Guide ===\n");
    linear_string_model();
    quadratic_native_model();
}

fn linear_string_model() {
    println!("Example 1: Linear model from a string");
    println!("Model: y = a * x + b");
    println!("Data:   y = 3x + 2\n");

    let x_data = (0..10).map(|x| x as f64).collect::<Vec<_>>();
    let y_data = x_data.iter().map(|&x| 3.0 * x + 2.0).collect::<Vec<_>>();

    // The builder collects all model inputs before solving.
    let fitted = Fitting::new()
        .with_data(x_data, y_data)
        .with_equation_str("a * x + b".to_string())
        .with_arg("x".to_string())
        .with_unknowns(vec!["a".to_string(), "b".to_string()])
        .with_initial_guess(vec![1.0, 1.0])
        .with_tolerance(1e-10)
        .with_max_iterations(100)
        .build();

    let map = fitted
        .map_of_solutions
        .expect("LM fitting should produce a solution map");
    println!("Recovered parameters:");
    println!("  a = {:.6}", map["a"]);
    println!("  b = {:.6}", map["b"]);
    println!();
}

fn quadratic_native_model() {
    println!("Example 2: Quadratic model from native Expr values");
    println!("Model: y = a * x^2 + b * x + c");
    println!("Data:   y = 2x^2 + 3x + 1\n");

    let x_data = (0..12).map(|x| x as f64).collect::<Vec<_>>();
    let y_data = x_data
        .iter()
        .map(|&x| 2.0 * x * x + 3.0 * x + 1.0)
        .collect::<Vec<_>>();

    // Native symbolic construction avoids parsing at runtime.
    let vars = Expr::Symbols("a, b, c, x");
    let a = vars[0].clone();
    let b = vars[1].clone();
    let c = vars[2].clone();
    let x = vars[3].clone();
    let eq = a * x.clone().pow(Expr::Const(2.0)) + b * x + c;

    let fitted = Fitting::new()
        .with_data(x_data, y_data)
        .with_equation(eq)
        .with_arg("x".to_string())
        .with_unknowns(vec!["a".to_string(), "b".to_string(), "c".to_string()])
        .with_initial_guess(vec![1.0, 1.0, 1.0])
        .with_tolerance(1e-10)
        .build();

    let map = fitted
        .map_of_solutions
        .expect("LM fitting should produce a solution map");
    println!("Recovered parameters:");
    println!("  a = {:.6}", map["a"]);
    println!("  b = {:.6}", map["b"]);
    println!("  c = {:.6}", map["c"]);
    println!();
}
