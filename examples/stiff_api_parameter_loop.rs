use RustedSciThe::numerical::BDF::BDF_api::{BdfSolverOptions, BdfTelemetryMode};
use RustedSciThe::numerical::StiffAPI::{StiffPreset, StiffSolverRequest};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DVector;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = BdfSolverOptions::for_bdf(
        vec![Expr::parse_expression("-rate*y")],
        vec!["y".into()],
        "t".into(),
        0.0,
        DVector::from_element(1, 1.0),
        1.0,
        0.02,
        1e-8,
        1e-10,
        None,
        false,
        Some(0.01),
    )
    .with_equation_parameters(vec!["rate".into()])
    .with_equation_parameter_values(DVector::from_element(1, 1.0))
    .with_telemetry_mode(BdfTelemetryMode::Counters);
    let mut solver = StiffSolverRequest::Bdf { options }.prepare(Some(&StiffPreset::solve_1d()))?;

    let report = solver.try_solve_parameter_loop(
        DVector::from_element(1, 1.0),
        &[vec![0.5], vec![1.0], vec![2.0]],
    );
    for point in report.points {
        match point.result {
            Ok(solution) => println!(
                "rate={:?}: final y={} at t={}, telemetry={:?}",
                point.parameter_values,
                solution.values[(solution.values.nrows() - 1, 0)],
                solution.x_mesh[solution.x_mesh.len() - 1],
                point.telemetry
            ),
            Err(error) => eprintln!("rate={:?}: {error}", point.parameter_values),
        }
    }
    Ok(())
}
