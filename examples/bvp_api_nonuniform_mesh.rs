use RustedSciThe::numerical::BVP_CorrectionNewton::CorrectionNewtonOptions;
use RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_damped::DampedSolverOptions;
use RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_frozen::FrozenSolverOptions;
use RustedSciThe::numerical::BVP_api::{
    BvpBoundaryCondition, BvpBoundarySide, BvpProblem, BvpSolver, BvpSolverSettings,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DMatrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mesh = vec![0.0, 0.07, 0.31, 0.66, 1.0];
    let mut seed = DMatrix::from_fn(1, mesh.len(), |_, node| 0.5 * mesh[node] * mesh[node]);
    seed[(0, 2)] += 0.05;
    let problem = BvpProblem::new(
        vec![Expr::parse_expression("x")],
        vec!["y".into()],
        "x",
        mesh,
        vec![BvpBoundaryCondition {
            state: 0,
            side: BvpBoundarySide::Left,
            value: 0.0,
        }],
    )?;

    let correction_config = BvpSolverSettings::CorrectionNewton(CorrectionNewtonOptions::default())
        .with_node_aligned_initial_guess(seed.clone());
    let mut correction = BvpSolver::new(problem.clone(), correction_config)?;
    let correction_solution = correction.solve()?;
    println!("CorrectionNewton: {:?}", correction_solution);

    let damped_config = BvpSolverSettings::Damped(
        DampedSolverOptions::default()
            .with_scheme_name("trapezoid")
            .with_abs_tolerance(1e-10)
            .with_rel_tolerance(std::collections::HashMap::from([("y".into(), 1e-10)]))
            .with_bounds(std::collections::HashMap::from([("y".into(), (-1.0, 1.0))])),
    )
    .with_node_aligned_initial_guess(seed.clone());
    let mut damped = BvpSolver::new(problem.clone(), damped_config)?;
    let damped_solution = damped.solve()?;
    println!("Damped: {:?}", damped_solution);

    let frozen_config = BvpSolverSettings::Frozen(
        FrozenSolverOptions::banded_frozen()
            .trapezoid_derivative()
            .with_tolerance(1e-10)
            .with_max_iterations(12),
    )
    .with_node_aligned_initial_guess(seed);
    let mut frozen = BvpSolver::new(problem, frozen_config)?;
    let frozen_solution = frozen.solve()?;
    println!("Frozen: {:?}", frozen_solution);
    Ok(())
}
