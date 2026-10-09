//! Русский пример Lambdify-ветки новой архитектуры `BVP_sci`.
//!
//! `ExprLegacy` и `AtomViewNative` специально запускаются на одной Dense-задаче.
//! Это не универсальный performance-рейтинг, а понятный пример выбора
//! frontend-а и места, где смотреть preparation/full-solve telemetry.

use RustedSciThe::numerical::BVP_sci::{
    BvpSciAssembly, BvpSciMatrixLayout, BvpSciSolver, BvpSciTelemetryMode,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;

fn run(assembly: BvpSciAssembly) {
    let mut solver = BvpSciSolver::builder(
        vec![Expr::parse_expression("y0"), Expr::parse_expression("y1")],
        vec!["y0".into(), "y1".into()],
    )
    .with_frontend(assembly)
    .with_matrix_layout(BvpSciMatrixLayout::Dense)
    .with_tolerance(1e-8)
    .with_telemetry(BvpSciTelemetryMode::Timings)
    .with_mesh_and_initial_state(vec![0.0, 0.5, 1.0], vec![0.0, 1.0, 0.5, 1.0, 1.0, 1.0])
    .with_boundary_callback(|ya, yb, _parameters, output| {
        output[0] = ya[0];
        output[1] = yb[1] - 1.0;
        Ok(())
    })
    .build()
    .expect("Lambdify solver должен создаваться");
    let solution = solver.solve().expect("Lambdify solve должен сойтись");
    let snapshot = solver.plan().telemetry_snapshot();
    println!(
        "{assembly:?} | {:.6e} | {:.3e} | {:.3} | {:.3} | {}",
        solution.y[solution.y.len() - 2],
        solution.residual_norm,
        snapshot.preparation_ms.unwrap_or_default(),
        snapshot.full_solve_ms.unwrap_or_default(),
        snapshot.jacobian_evaluations,
    );
}

fn main() {
    println!("frontend | y(1) | residual_norm | preparation_ms | full_solve_ms | jacobian_calls");
    run(BvpSciAssembly::ExprLegacy);
    run(BvpSciAssembly::AtomViewNative);
}
