//! Symbolic VarPro fitting guide.
//!
//! This guide shows the VarPro frontend in the crate:
//! - one nonlinear parameter per basis function,
//! - linear coefficients handled by the solver,
//! - invariant/background terms added separately.
//!
//! run cargo run --example varpro_fitting_guide

use RustedSciThe::numerical::optimization::varpro::symbolic::SymbolicVarProBuilder;
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DVector;

fn main() {
    println!("=== VarPro Fitting Guide ===\n");
    exponential_with_offset();
}

fn exponential_with_offset() {
    println!("Example: exponential decay with a constant offset");
    println!("Model: y = a * exp(-x / tau) + b");
    println!("Nonlinear parameter: tau");
    println!("Linear coefficients: a, b\n");

    let x_data = (0..30).map(|i| i as f64 * 0.2).collect::<Vec<_>>();
    let y_data = x_data
        .iter()
        .map(|&x| 2.5 * (-x / 1.7).exp() + 0.8)
        .collect::<Vec<_>>();

    // VarPro separates nonlinear basis shape from linear coefficients.
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

    println!("Recovered parameters:");
    println!("  tau = {:.6}", nonlinear[0]);
    println!("  a   = {:.6}", linear[0]);
    println!("  b   = {:.6}", linear[1]);
    println!();
}
