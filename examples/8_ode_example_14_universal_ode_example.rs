//! Универсальный API для nonstiff IVP.
//!
//! BDF, BE, Radau и LSODE2 не проходят через `UniversalODESolver`: для них
//! нужно использовать native API конкретного солвера, где доступны их
//! специальные настройки и typed errors.

use RustedSciThe::numerical::ODE_api2::{NonStiffMethod, SolverType, UniversalODESolver};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DVector;

fn main() {
    // y' = -y, y(0)=1, точное решение y(t)=exp(-t).
    let equations = vec![Expr::parse_expression("-y")];
    let variables = vec!["y".to_string()];
    let initial = DVector::from_vec(vec![1.0]);
    let expected = (-1.0_f64).exp();

    let mut solver = UniversalODESolver::new(
        equations,
        variables,
        "t".to_string(),
        SolverType::NonStiff(NonStiffMethod::Rk45),
        0.0,
        initial,
        1.0,
    );
    solver.set_step_size(1e-4);
    solver.solve();

    let (_, values) = solver.get_result();
    let final_value = values.expect("RK45 should produce a result")[(0, 0)];
    println!("final={final_value:.6}, expected={expected:.6}");
}
