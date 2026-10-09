//! Русский исполняемый пример чисто числовой ветки новой архитектуры `BVP_sci`.
//!
//! Символьный frontend здесь не используется: residual задаётся замыканием.
//! При наличии аналитического Jacobian солвер вызывает его напрямую, а при
//! отсутствии использует finite differences. Оба случая остаются Dense-only.
//! Полное описание API находится в `BVP_SCI_USER_GUIDE_RU.md`.

use RustedSciThe::numerical::BVP_sci::{
    BvpSciBoundaryCallbacks, BvpSciExecution, BvpSciLambdifyPlan, BvpSciMatrixLayout,
    BvpSciNumericalPlan, BvpSciOptions, BvpSciSolver, BvpSciTelemetry,
};

/// Создаёт одинаковую задачу с FD или аналитическим Jacobian.
fn build_solver(with_jacobian: bool) -> BvpSciSolver {
    let telemetry = BvpSciTelemetry::timings();
    let plan = BvpSciNumericalPlan::new(
        1,
        0,
        |_x, _state, _parameters, output| {
            // Точная задача: y' = 1, y(1) = 1, поэтому y(x) = x.
            output[0] = 1.0;
            Ok(())
        },
        telemetry.clone(),
    )
    .expect("числовой plan должен быть корректным");
    let plan = if with_jacobian {
        plan.with_rhs_jacobian(|_x, _state, _parameters, output| {
            output[0] = 0.0;
            Ok(())
        })
    } else {
        plan
    };
    let plan = BvpSciLambdifyPlan::prepare_numerical(plan);

    // Во второй ветке задаём и Jacobian граничного условия. В FD-ветке этот
    // callback не передаётся, поэтому telemetry показывает FD probes.
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
    .expect("числовой solver должен создаваться")
}

fn main() {
    println!("BVP_sci Numerical / Dense");
    println!("route | final_value | residual_norm | fd_probes | jacobian_calls | full_solve_ms");
    for (label, analytical) in [("finite-difference", false), ("analytical", true)] {
        let mut solver = build_solver(analytical);
        let solution = solver.solve().expect("числовая BVP должна сойтись");
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
