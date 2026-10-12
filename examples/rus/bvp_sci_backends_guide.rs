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

// Необязательный sparse-маршрут SuperLU
// ------------------------------------
// SuperLU — альтернативная политика факторизации для layout Sparse, а не
// четвёртый layout. StandardFaer остаётся настройкой по умолчанию. C ABI
// bridge нужно собрать отдельно для целевой ОС. Нужны Git, CMake, Ninja и C
// compiler; скрипты фиксируют SuperLU 6.0.1 и собирают bridge вместе с
// bundled CBLAS.
//
// Windows (PowerShell; DLL создаётся в игнорируемой папке test_reports/):
//   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_superlu_bridge.ps1
//   $env:SUPERLU_LIBRARY = (Resolve-Path '.\test_reports\somelinalg_sparse_lu_reference\native\superlu_bridge.dll').Path
//
// Linux (shared object создаётся там же):
//   bash scripts/build_superlu_bridge.sh
//   export SUPERLU_LIBRARY="$(pwd)/test_reports/somelinalg_sparse_lu_reference/native/libsuperlu_bridge.so"
//
// Соберите и запустите этот пример с необязательной Cargo feature:
//   cargo run --no-default-features --features superlu --example bvp_sci_backends_guide
//
// Выбор в builder (обычный `new`/builder по-прежнему использует StandardFaer):
//   let solver = BvpSciSolver::builder(rhs, state_names)
//       .with_frontend(BvpSciAssembly::ExprLegacy)
//       .with_superlu() // одновременно выбирает BvpSciMatrixLayout::Sparse
//       .with_mesh_and_initial_state(mesh, initial_state)
//       .with_boundary_callback(boundary)
//       .build()?;
//
// Если Cargo feature выключена или bridge не загрузился/не прошёл проверку,
// BVP_sci предупредит и перейдёт на StandardFaer. Перед замером убедитесь,
// что `telemetry_snapshot().superlu_fallbacks == 0`, иначе это не замер
// SuperLU. Полная инструкция по сборке и release-матрице находится в
// `src/numerical/BVP_sci/SUPERLU_BACKEND.md`.
