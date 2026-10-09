//! Universal facade example for nonstiff IVP methods.
//!
//! Stiff first-tier solvers are intentionally not hidden behind this facade:
//! use the native BDF, BE, Radau, or LSODE2 API when their controls are needed.

use RustedSciThe::numerical::ODE_api2::{NonStiffMethod, SolverType, UniversalODESolver};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DVector;

fn main() {
    let equations = vec![Expr::parse_expression("-y")];
    let variables = vec!["y".to_string()];
    let y0 = DVector::from_vec(vec![1.0]);
    let expected = (-1.0_f64).exp();

    let mut rk45 = UniversalODESolver::rk45(
        equations.clone(),
        variables.clone(),
        "t".to_string(),
        0.0,
        y0.clone(),
        1.0,
        1e-4,
    );
    rk45.solve();
    print_result("RK45", &rk45, expected);

    let mut dopri = UniversalODESolver::dopri(
        equations.clone(),
        variables.clone(),
        "t".to_string(),
        0.0,
        y0.clone(),
        1.0,
        1e-4,
    );
    dopri.solve();
    print_result("DOPRI", &dopri, expected);

    let mut custom = UniversalODESolver::new(
        equations,
        variables,
        "t".to_string(),
        SolverType::NonStiff(NonStiffMethod::Rk45),
        0.0,
        y0,
        1.0,
    );
    custom.set_step_size(5e-5);
    custom.solve();
    print_result("custom RK45", &custom, expected);
}

fn print_result(name: &str, solver: &UniversalODESolver, expected: f64) {
    let (times, values) = solver.get_result();
    if let (Some(times), Some(values)) = (times, values) {
        let value = values[(values.nrows() - 1, 0)];
        println!(
            "{name}: final={value:.6}, expected={expected:.6}, error={:.2e}, steps={}",
            (value - expected).abs(),
            times.len()
        );
    }
}
