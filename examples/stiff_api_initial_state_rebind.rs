use RustedSciThe::numerical::StiffAPI::{StiffPreset, StiffSolverRequest, bdf::BdfSolverOptions};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DVector;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = BdfSolverOptions::for_bdf(
        vec![Expr::parse_expression("-rate*y")],
        vec!["y".into()],
        "t".into(),
        0.0,
        DVector::from_vec(vec![1.0]),
        0.5,
        0.01,
        1e-9,
        1e-11,
        None,
        false,
        Some(0.01),
    )
    .with_equation_parameters(vec!["rate".into()])
    .with_equation_parameter_values(DVector::from_vec(vec![0.7]));
    let mut prepared =
        StiffSolverRequest::Bdf { options }.prepare(Some(&StiffPreset::expr_dense()))?;

    prepared.try_solve()?;
    prepared.try_solve_with_initial_state(DVector::from_vec(vec![2.0]))?;
    let trajectory = prepared.trajectory_view()?;
    let final_value = trajectory.values[(trajectory.values.nrows() - 1, 0)];
    println!("fresh solve from replacement y0=2: y(0.5)={final_value:.9}");
    Ok(())
}
