//! Руководство по symbolic VarPro fitting.
//!
//! Здесь показан frontend для variable projection:
//! - один нелинейный параметр на одну basis function,
//! - линейные коэффициенты восстанавливаются solver-ом,
//! - константный фон задаётся отдельно как invariant term.
//!
//! run cargo run --example varpro_fitting_guide

use RustedSciThe::numerical::optimization::varpro::symbolic::SymbolicVarProBuilder;
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DVector;

fn main() {
    println!("=== Руководство по VarPro Fitting ===\n");
    exponential_with_offset();
}

fn exponential_with_offset() {
    println!("Пример: экспоненциальный спад с постоянным смещением");
    println!("Модель: y = a * exp(-x / tau) + b");
    println!("Нелинейный параметр: tau");
    println!("Линейные коэффициенты: a, b\n");

    let x_data = (0..30).map(|i| i as f64 * 0.2).collect::<Vec<_>>();
    let y_data = x_data
        .iter()
        .map(|&x| 2.5 * (-x / 1.7).exp() + 0.8)
        .collect::<Vec<_>>();

    let fit = SymbolicVarProBuilder::new("x")
        .with_parameters(vec!["tau".to_string()])
        .with_initial_parameters(vec![1.2])
        .with_data(DVector::from_vec(x_data), DVector::from_vec(y_data))
        .with_basis_str("tau", "exp(-x/tau)")
        .with_equations(vec![Expr::parse_expression("1")])
        .solve()
        .expect("VarPro fit should succeed");

    let nonlinear = fit.nonlinear_parameters();
    let linear = fit
        .linear_coefficients()
        .expect("VarPro fit should return linear coefficients");

    println!("Восстановленные параметры:");
    println!("  tau = {:.6}", nonlinear[0]);
    println!("  a   = {:.6}", linear[0]);
    println!("  b   = {:.6}", linear[1]);
    println!();
}
