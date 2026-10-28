use RustedSciThe::numerical::Radau::api::{
    RadauConfig, RadauOutputPolicy, RadauProblem, RadauSolver,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let problem = RadauProblem::new(
        vec![Expr::parse_expression("-rate*y")],
        vec!["y".into()],
        "t",
    )
    .with_parameter_names(vec!["rate".into()]);
    let config = RadauConfig {
        t_bound: 1.0,
        output: RadauOutputPolicy::Dense,
        ..RadauConfig::default()
    };
    let mut solver = RadauSolver::prepare(problem, config)?;

    let report = solver.try_solve_parameter_loop(&[1.0], &[vec![0.5], vec![1.0], vec![2.0]]);
    for point in report.points {
        match point.result {
            Ok(solution) => println!(
                "rate={:?}: {} retained output points, final y={}",
                point.parameter_values,
                solution.x_mesh.len(),
                solution.values[(solution.values.nrows() - 1, 0)]
            ),
            Err(error) => eprintln!("rate={:?}: {error}", point.parameter_values),
        }
    }
    Ok(())
}
