//! Руководство по Universal fitting.
//!
//! Это high-level wrapper над двумя backend-ами:
//! - LM для классической symbolic least-squares fitting,
//! - VarPro для separable models, где backend выбирается через `Method`.
//!
//! run cargo run --example universal_fitting_guide

use RustedSciThe::numerical::optimization::{Method, UniversalFitting, UniversalFittingResult};
use RustedSciThe::symbolic::symbolic_engine::Expr;

fn main() {
    println!("=== Руководство по Universal Fitting ===\n");
    universal_lm();
    universal_varpro();
}

fn universal_lm() {
    println!("Пример 1: universal wrapper с классическим LM");
    println!("Модель: y = a * x + b");
    println!("Здесь wrapper использует LM backend.\n");

    let x_data = (0..8).map(|x| x as f64).collect::<Vec<_>>();
    let y_data = x_data.iter().map(|&x| 4.0 * x - 1.0).collect::<Vec<_>>();

    let result = UniversalFitting::new()
        .with_method(Method::LM)
        .with_data(x_data, y_data)
        .with_equation_str("a * x + b".to_string())
        .with_arg("x".to_string())
        .with_unknowns(vec!["a".to_string(), "b".to_string()])
        .with_initial_guess(vec![1.0, 0.0])
        .with_tolerance(1e-10)
        .build()
        .expect("LM build should succeed");

    let lm = match result {
        UniversalFittingResult::LM(fit) => fit,
        UniversalFittingResult::VARPRO(_) => panic!("ожидался результат LM"),
    };

    let map = lm
        .map_of_solutions
        .expect("LM path should produce a solution map");
    println!("Восстановленные параметры:");
    println!("  a = {:.6}", map["a"]);
    println!("  b = {:.6}", map["b"]);
    println!();
}

fn universal_varpro() {
    println!("Пример 2: universal wrapper с VarPro");
    println!("Модель: y = a * exp(-x / tau) + b");
    println!("Backend выбирается через Method::VARPRO.\n");

    let x_data = (0..30).map(|i| i as f64 * 0.2).collect::<Vec<_>>();
    let y_data = x_data
        .iter()
        .map(|&x| 2.5 * (-x / 1.7).exp() + 0.8)
        .collect::<Vec<_>>();

    let result = UniversalFitting::new()
        .with_method(Method::VARPRO)
        .with_data(x_data, y_data)
        .with_arg("x".to_string())
        .with_parameters(vec!["tau".to_string()])
        .with_initial_guess(vec![1.2])
        .with_basis_str("tau", "exp(-x/tau)")
        .with_equations(vec![Expr::parse_expression("1")])
        .build()
        .expect("VarPro build should succeed");

    let fit = match result {
        UniversalFittingResult::VARPRO(fit) => fit,
        UniversalFittingResult::LM(_) => panic!("ожидался результат VarPro"),
    };

    let nonlinear = fit.nonlinear_parameters();
    let linear = fit
        .linear_coefficients()
        .expect("VarPro path should return linear coefficients");

    println!("Восстановленные параметры:");
    println!("  tau = {:.6}", nonlinear[0]);
    println!("  a   = {:.6}", linear[0]);
    println!("  b   = {:.6}", linear[1]);
    println!();
}
