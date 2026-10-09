//! Русский пример выбора matrix layout в новой архитектуре `BVP_sci`.
//!
//! Backend выбирается один раз при создании solver-а. Одна ExprLegacy-задача
//! решается через Dense, Sparse и Banded storage, а таблица показывает
//! residual, сборку матрицы, факторизацию и число solve-операций. Универсально
//! лучшего layout нет: он зависит от структуры Jacobian и размера задачи.

use RustedSciThe::numerical::BVP_sci::{
    BvpSciAssembly, BvpSciMatrixLayout, BvpSciSolver, BvpSciTelemetryMode,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;

fn run(layout: BvpSciMatrixLayout) {
    let mut solver = BvpSciSolver::builder(
        vec![Expr::parse_expression("y0"), Expr::parse_expression("y1")],
        vec!["y0".into(), "y1".into()],
    )
    .with_frontend(BvpSciAssembly::ExprLegacy)
    .with_matrix_layout(layout)
    .with_tolerance(1e-8)
    .with_telemetry(BvpSciTelemetryMode::Timings)
    .with_mesh_and_initial_state(vec![0.0, 0.5, 1.0], vec![0.0, 1.0, 0.5, 1.0, 1.0, 1.0])
    .with_boundary_callback(|ya, yb, _parameters, output| {
        output[0] = ya[0];
        output[1] = yb[1] - 1.0;
        Ok(())
    })
    .build()
    .expect("backend solver должен создаваться");
    let solution = solver.solve().expect("backend solve должен сойтись");
    let snapshot = solver.plan().telemetry_snapshot();
    println!(
        "{layout:?} | {:.6e} | {:.3e} | {:.3} | {:.3} | {}",
        solution.y[solution.y.len() - 2],
        solution.residual_norm,
        snapshot.linear_assembly_ms.unwrap_or_default(),
        snapshot.factorization_ms.unwrap_or_default(),
        snapshot.linear_solves,
    );
}

fn main() {
    println!("layout | y(1) | residual_norm | linear_assembly_ms | factorization_ms | solves");
    run(BvpSciMatrixLayout::Dense);
    run(BvpSciMatrixLayout::Sparse);
    // Для двух state-компонент ширина 1/1 достаточна; 5/5 была бы слишком
    // широкой для этого маленького учебного Jacobian.
    run(BvpSciMatrixLayout::Banded { lower: 1, upper: 1 });
}
