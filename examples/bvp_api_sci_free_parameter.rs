//! Solve a BVP_sci problem whose scalar parameter is a Newton unknown.

use RustedSciThe::numerical::BVP_api::{
    BvpBoundaryCondition, BvpBoundarySide, BvpProblem, BvpSciFreeParameters,
    BvpSciSparseFactorization, BvpSolution, BvpSolver, BvpSolverConfig,
};
use RustedSciThe::numerical::BVP_sci::new::{
    BvpSciAssembly, BvpSciExecution, BvpSciOptions, BvpSciTelemetryMode,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DMatrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // y' = p, y(0) = 0. The extra residual p - 1 = 0 determines free p.
    let problem = BvpProblem::new(
        vec![Expr::parse_expression("p")],
        vec!["y".into()],
        "x",
        vec![0.0, 0.25, 0.5, 0.75, 1.0],
        vec![BvpBoundaryCondition {
            state: 0,
            side: BvpBoundarySide::Left,
            value: 0.0,
        }],
    )?;
    let free_parameters = BvpSciFreeParameters::new(
        vec!["p".into()],
        vec![0.4],
        |_ya, _yb, parameters, residuals| {
            residuals[0] = parameters[0] - 1.0;
            Ok(())
        },
    )
    .with_jacobian(|_ya, _yb, _parameters, dya, dyb, dp| {
        dya[0] = 0.0;
        dyb[0] = 0.0;
        dp[0] = 1.0;
        Ok(())
    });
    let config = BvpSolverConfig::SciWithFreeParameters {
        options: BvpSciOptions::default()
            .with_assembly(BvpSciAssembly::ExprLegacy)
            .with_execution(BvpSciExecution::Lambdify)
            .with_telemetry(BvpSciTelemetryMode::Detailed),
        sparse_factorization: BvpSciSparseFactorization::StandardFaer,
        aot_config: None,
        initial_guess: DMatrix::zeros(1, 5),
        free_parameters,
    };
    let mut solver = BvpSolver::new(problem, config)?;
    let BvpSolution::Sci(solution) = solver.solve()? else {
        unreachable!("the configured route is BVP_sci");
    };
    println!("p={:.8}, y(1)={:.8}", solution.parameters[0], solution.y[4]);

    let repeated_parameter = {
        let BvpSolution::Sci(repeated) = solver.solve_with_parameter_guess(vec![0.8])? else {
            unreachable!("the configured route is BVP_sci");
        };
        repeated.parameters[0]
    };
    let updated_guess = solver.parameter_guess()?[0];
    println!(
        "stored guess after solve={:.8}, solved p={:.8}",
        updated_guess, repeated_parameter
    );
    Ok(())
}
