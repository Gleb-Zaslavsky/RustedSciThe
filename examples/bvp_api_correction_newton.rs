use RustedSciThe::numerical::BVP_CorrectionNewton::CorrectionNewtonOptions;
use RustedSciThe::numerical::BVP_api::{
    BvpApiTelemetryMode, BvpBoundaryCondition, BvpBoundarySide, BvpProblem, BvpSolver,
    BvpSolverSettings, BvpSolverTelemetrySnapshot,
};
use RustedSciThe::numerical::logging::SolverLogLevel;
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DMatrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let problem = BvpProblem::new(
        vec![Expr::parse_expression("-rate*y")],
        vec!["y".into()],
        "x",
        vec![0.0, 0.25, 0.5, 0.75, 1.0],
        vec![BvpBoundaryCondition {
            state: 0,
            side: BvpBoundarySide::Left,
            value: 1.0,
        }],
    )?
    .with_external_parameters(vec!["rate".into()], vec![0.5])?;
    let settings = BvpSolverSettings::CorrectionNewton(CorrectionNewtonOptions::default())
        .with_telemetry_mode(BvpApiTelemetryMode::Detailed)?
        .with_initial_guess(DMatrix::from_element(1, 4, 1.0))
        .with_logging_level(SolverLogLevel::Off);
    let mut solver = BvpSolver::new(problem, settings)?;

    solver.solve()?;
    let endpoint_at_half = solver
        .solution_view()
        .expect("successful solve publishes profile")
        .value(4, 0)
        .expect("endpoint is in the result");
    solver.solve_with_parameter_values(vec![1.0])?;
    let view = solver
        .solution_view()
        .expect("successful solve publishes profile");
    println!("CorrectionNewton route: {:?}", solver.solution().unwrap());
    println!(
        "profile: {} nodes x {} states",
        view.mesh().len(),
        view.state_count()
    );
    println!("y(1), rate=0.5: {endpoint_at_half:.12e}");
    println!(
        "y(1), rate=1.0: {:.12e}",
        view.value(4, 0).expect("endpoint is in the result")
    );
    if let BvpSolverTelemetrySnapshot::CorrectionNewton(Some(snapshot)) =
        solver.telemetry_snapshot()
    {
        println!(
            "callbacks: residual={}, Jacobian={}, factorization={:?}",
            snapshot.residual_evaluations,
            snapshot.jacobian_evaluations,
            snapshot.factorization_time
        );
    }
    Ok(())
}
