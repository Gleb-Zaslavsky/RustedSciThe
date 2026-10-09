// Numerical BVP_sci parameter-continuation guide.
// Run with `cargo run --example bvp_sci_numerical_parameters_guide`.
//
// `set_parameters` keeps the prepared callbacks, but it starts a new
// numerical approximation for the new parameter value. This example uses
// two genuinely different members of the family: both the differential
// equation and the boundary residual depend on `p`.

use RustedSciThe::numerical::BVP_sci::{
    BvpSciBoundaryCallbacks, BvpSciExecution, BvpSciLambdifyPlan, BvpSciMatrixLayout,
    BvpSciNumericalPlan, BvpSciOptions, BvpSciSolver, BvpSciTelemetry,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let telemetry = BvpSciTelemetry::counters();
    let plan = BvpSciNumericalPlan::new(
        1,
        1,
        |_x, _state, parameters, output| {
            output[0] = parameters[0];
            Ok(())
        },
        telemetry.clone(),
    )?;
    let plan = BvpSciLambdifyPlan::prepare_numerical(plan);
    let boundary = BvpSciBoundaryCallbacks::new(
        2,
        |ya, _yb, parameters, output| {
            output[0] = ya[0];
            output[1] = parameters[0] - 1.0;
            Ok(())
        },
        telemetry,
    );
    let options = BvpSciOptions::default()
        .with_execution(BvpSciExecution::Numerical)
        .with_matrix_layout(BvpSciMatrixLayout::Dense)
        .with_tolerance(1e-7);
    let mut solver = BvpSciSolver::new(
        plan,
        boundary,
        vec![0.0, 0.5, 1.0],
        vec![0.0, 0.25, 0.5],
        vec![0.25],
        options,
    )?;

    println!("BVP_sci numerical continuation");
    println!("parameter | y(0) | y(1) | residual_norm | continuation_solves");
    for parameter in [0.25, 1.5] {
        if parameter != 0.25 {
            // This is a new BVP member, not a repeated read of p=1 with the
            // unchanged residual p - 1 = 0. The symbolic preparation is reused
            // while the numerical state is re-bound and solved again.
            solver.set_parameters(vec![parameter])?;
        }
        let solution = solver.solve()?;
        let snapshot = solver.plan().telemetry_snapshot();
        println!(
            "{:.6} | {:.6} | {:.6} | {:.3e} | {}",
            solution.parameters[0],
            solution.y[0],
            solution.y[solution.y.len() - 1],
            solution.residual_norm,
            snapshot.continuation_solves,
        );
    }
    Ok(())
}
