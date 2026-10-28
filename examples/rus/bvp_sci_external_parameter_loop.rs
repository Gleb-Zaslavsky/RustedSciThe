//! Reuse a prepared BVP_sci model with different fixed model parameters.
//! The previous successful profile is the next point's initial guess.

use RustedSciThe::numerical::BVP_sci::new::BvpSciSolver;
use RustedSciThe::symbolic::symbolic_engine::Expr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut solver = BvpSciSolver::builder(vec![Expr::parse_expression("rate")], vec!["y".into()])
        .with_external_parameter_names(vec!["rate".into()])
        .with_external_parameter_values(vec![1.0])
        .with_mesh_and_initial_state(vec![0.0, 0.5, 1.0], vec![0.0; 3])
        .with_boundary_callback(|ya, _yb, _runtime_parameters, output| {
            output[0] = ya[0];
            Ok(())
        })
        .build()?;

    let report = solver.solve_parameter_loop(&[vec![1.0], vec![2.0], vec![3.0]])?;
    for point in report.points {
        let solution = point.result?;
        println!(
            "rate={:.1}, nodes={}, y(1)={:.8}, Newton iterations={}",
            point.external_parameters[0],
            solution.x.len(),
            solution.y[solution.y.len() - 1],
            solution.iterations
        );
    }
    if let Some(error) = report.stop_error {
        return Err(error.into());
    }
    Ok(())
}
