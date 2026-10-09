// Numerical BVP_sci guide: analytical callbacks versus finite differences.
// Run with `cargo run --example bvp_sci_numerical_guide`.

use RustedSciThe::numerical::BVP_sci::{
    BvpSciBoundaryCallbacks, BvpSciExecution, BvpSciLambdifyPlan, BvpSciMatrixLayout,
    BvpSciNumericalPlan, BvpSciOptions, BvpSciSolver, BvpSciTelemetry,
};

fn build_solver(with_jacobian: bool) -> BvpSciSolver {
    let telemetry = BvpSciTelemetry::timings();
    let plan = BvpSciNumericalPlan::new(
        1,
        0,
        |_x, _state, _parameters, output| {
            output[0] = 1.0;
            Ok(())
        },
        telemetry.clone(),
    )
    .expect("numerical plan should be valid");
    let plan = if with_jacobian {
        plan.with_rhs_jacobian(|_x, _state, _parameters, output| {
            output[0] = 0.0;
            Ok(())
        })
    } else {
        plan
    };
    let plan = BvpSciLambdifyPlan::prepare_numerical(plan);
    let boundary = if with_jacobian {
        BvpSciBoundaryCallbacks::new_with_jacobian(
            1,
            |_ya, yb, _parameters, output| {
                output[0] = yb[0] - 1.0;
                Ok(())
            },
            |_ya, _yb, _parameters, dya, dyb, _dp| {
                dya[0] = 0.0;
                dyb[0] = 1.0;
                Ok(())
            },
            telemetry,
        )
    } else {
        BvpSciBoundaryCallbacks::new(
            1,
            |_ya, yb, _parameters, output| {
                output[0] = yb[0] - 1.0;
                Ok(())
            },
            telemetry,
        )
    };
    let options = BvpSciOptions::default()
        .with_execution(BvpSciExecution::Numerical)
        .with_matrix_layout(BvpSciMatrixLayout::Dense)
        .with_tolerance(1e-8);
    BvpSciSolver::new(
        plan,
        boundary,
        vec![0.0, 0.5, 1.0],
        vec![0.0, 0.5, 0.0],
        vec![],
        options,
    )
    .expect("numerical solver should construct")
}

fn main() {
    println!("BVP_sci Numerical / Dense");
    println!("route | final_value | residual_norm | fd_probes | jacobian_calls | full_solve_ms");
    for (label, analytical) in [("finite-difference", false), ("analytical", true)] {
        let mut solver = build_solver(analytical);
        let solution = solver.solve().expect("numerical BVP should converge");
        let telemetry = solver.plan().telemetry_snapshot();
        println!(
            "{label} | {:.6e} | {:.3e} | {} | {} | {:.3}",
            solution.y[solution.y.len() - 1],
            solution.residual_norm,
            telemetry.finite_difference_probes,
            telemetry.jacobian_evaluations,
            telemetry.full_solve_ms.unwrap_or_default(),
        );
    }
}
