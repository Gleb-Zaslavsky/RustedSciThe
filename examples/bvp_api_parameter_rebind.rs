//! Reuse a prepared BVP_Damp route while changing a fixed equation input.

use std::collections::HashMap;

use RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_damped::DampedSolverOptions;
use RustedSciThe::numerical::BVP_api::{
    BvpBoundaryCondition, BvpBoundarySide, BvpProblem, BvpSolution, BvpSolver, BvpSolverConfig,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DMatrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // y' = rate, y(0) = 0, so the exact profile is y(x) = rate*x.
    let problem = BvpProblem::new(
        vec![Expr::parse_expression("rate")],
        vec!["y".into()],
        "x",
        vec![0.0, 0.25, 0.5, 0.75, 1.0],
        vec![BvpBoundaryCondition {
            state: 0,
            side: BvpBoundarySide::Left,
            value: 0.0,
        }],
    )?;
    let config = BvpSolverConfig::Damped {
        options: DampedSolverOptions::default()
            .with_bounds(HashMap::from([("y".to_string(), (-10.0, 10.0))]))
            .with_rel_tolerance(HashMap::from([("y".to_string(), 1e-6)])),
        // Damped stores one Newton-unknown column per mesh interval.
        initial_guess: DMatrix::zeros(1, 4),
    };
    let mut solver = BvpSolver::new(problem, config)?;
    solver.bind_parameters(vec!["rate".into()], vec![1.0])?;

    if let BvpSolution::Damped { values, .. } = solver.solve()? {
        println!("rate=1, y(1)={:.8}", values[(4, 0)]);
    }
    if let BvpSolution::Damped { values, .. } = solver.solve_with_parameter_values(vec![2.0])? {
        println!("rate=2, y(1)={:.8}", values[(4, 0)]);
    }
    Ok(())
}
