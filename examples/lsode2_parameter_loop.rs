use RustedSciThe::numerical::LSODE2::{Lsode2ProblemConfig, Lsode2Solver};
use RustedSciThe::symbolic::ivp_telemetry::IvpTelemetry;
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DVector;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Lsode2ProblemConfig::new(
        vec![Expr::parse_expression("-rate*y")],
        vec!["y".into()],
        "t".into(),
        0.0,
        DVector::from_element(1, 1.0),
        1.0,
        0.02,
        1e-8,
        1e-10,
    )
    .with_equation_parameters(vec!["rate".into()])
    .with_equation_parameter_values(DVector::from_element(1, 1.0))
    .with_bridge_solve()
    .with_telemetry(IvpTelemetry::detailed());
    let mut solver = Lsode2Solver::new(config)?;

    let report = solver.try_solve_parameter_loop(
        DVector::from_element(1, 1.0),
        &[vec![0.5], vec![1.0], vec![2.0]],
    );
    for point in report.points {
        match point.result {
            Ok(solution) => println!(
                "rate={:?}: {} output points, {} telemetry solve calls",
                point.parameter_values,
                solution.x_mesh.len(),
                point.telemetry.statistics.solve_calls
            ),
            Err(error) => eprintln!("rate={:?}: {error}", point.parameter_values),
        }
    }
    Ok(())
}
