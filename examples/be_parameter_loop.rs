use RustedSciThe::numerical::BE::{BE, BeSolverOptions, BeTelemetryMode};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DVector;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = BeSolverOptions::new(
        vec![Expr::parse_expression("-rate*y")],
        vec!["y".into()],
        "t".into(),
        1e-8,
        20,
        Some(0.005),
        0.0,
        1.0,
        DVector::from_element(1, 1.0),
    )
    .with_telemetry_mode(BeTelemetryMode::Counters);
    let mut solver = BE::try_new_with_options(options)?;
    solver.try_set_equation_parameters(Some(&["rate"]))?;

    let report = solver.try_solve_parameter_loop(
        DVector::from_element(1, 1.0),
        &[vec![0.5], vec![1.0], vec![2.0]],
    );
    for point in report.points {
        match point.result {
            Ok(solution) => println!(
                "rate={:?}: final y={}",
                point.parameter_values,
                solution.values[(solution.values.nrows() - 1, 0)]
            ),
            Err(error) => eprintln!("rate={:?}: {error}", point.parameter_values),
        }
    }
    Ok(())
}
