use RustedSciThe::numerical::ODE_api2::{NonStiffMethod, SolverType, UniversalODESolver};
use RustedSciThe::symbolic::ivp_telemetry::IvpTelemetry;
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DVector;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let equation = Expr::parse_expression("-y");
    let mut solver = UniversalODESolver::new(
        vec![equation],
        vec!["y".to_owned()],
        "t".to_owned(),
        SolverType::NonStiff(NonStiffMethod::Dop853),
        0.0,
        DVector::from_element(1, 1.0),
        1.0,
    )
    .with_nonstiff_symbolic_telemetry(IvpTelemetry::detailed());
    solver.set_rtol(1e-11);
    solver.set_atol(1e-13);
    solver.set_max_step(0.25);
    solver.set_first_step(Some(0.05));
    solver.try_solve()?;

    let (times, states) = solver.get_result();
    let times = times.expect("DOP853 returns accepted mesh points");
    let states = states.expect("DOP853 returns state rows");
    let final_time = times[times.len() - 1];
    let final_state = states[(states.nrows() - 1, 0)];
    let exact = (-final_time).exp();
    assert!((final_state - exact).abs() < 2e-10);
    println!(
        "DOP853 t={final_time:.6}, y={final_state:.12}, error={:.3e}",
        (final_state - exact).abs()
    );
    println!("{}", solver.statistics_report().unwrap_or_default());
    Ok(())
}
