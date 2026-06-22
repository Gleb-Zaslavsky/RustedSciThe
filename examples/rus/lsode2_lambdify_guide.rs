//! Гайд по LSODE2 через Lambdify
//!
//! Цель:
//! решить символическую IVP через Lambdify без ahead-of-time компиляции.
//!
//! Обычно это лучший первый symbolic workflow:
//! 1. записать уравнения строками (`Expr::parse_expression`)
//! 2. выбрать структуру якобиана (dense/sparse/banded)
//! 3. дать LSODE2 построить вычислители невязки и якобиана через Lambdify
//! 4. запустить через `UniversalODESolver` (тот же фасад, что и у других ODE-солверов)
//!
//! Задача:
//! y1' = -10 y1 + 9 y2
//! y2' =  y1 - y2
//! with y(0)=[1,0], t in [0,1]
//!
//! Почему здесь sparse:
//! эту маленькую систему можно было бы сделать и dense, но sparse-настройка
//! демонстрирует production-ориентированный путь, который важен на больших якобианах.
//!
//! запуск: cargo run --example lsode2_lambdify_guide

use RustedSciThe::numerical::LSODE2::{
    Lsode2BackendConfig, Lsode2LinearSolverPolicy, Lsode2LinearSystemStructure,
    Lsode2ProblemConfig, Lsode2ResidualJacobianSource, Lsode2SymbolicAssemblyBackend,
    Lsode2SymbolicExecutionMode,
};
use RustedSciThe::numerical::ODE_api2::UniversalODESolver;
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DVector;

fn main() {
    // Шаг 1: задаём символическую систему, переменные и интервал интегрирования.
    let config = Lsode2ProblemConfig::new(
        vec![
            Expr::parse_expression("-10.0*y1 + 9.0*y2"),
            Expr::parse_expression("y1 - y2"),
        ],
        vec!["y1".to_string(), "y2".to_string()],
        "t".to_string(),
        0.0,
        DVector::from_vec(vec![1.0, 0.0]),
        1.0,
        0.02,
        1e-6,
        1e-8,
    )
    // Шаг 2: выбираем символический источник и режим исполнения.
    //
    // `ExprLegacy`:
    // стабильный бэкенд символьной сборки.
    // `LambdifyExpr`:
    // runtime-замыкания на Rust без внешнего компилятора.
    .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
        assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
        execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
    })
    // Опционально:
    // даже на Lambdify-пути можно управлять runtime-chunking.
    // Это полезно для тяжёлых символических ядер на многопроцессорной машине.
    .with_backend(
        Lsode2BackendConfig::native_sparse_faer().with_generated_backend_target_chunks(4, 4),
    )
    // Шаг 3: выбираем структуру и policy линейной алгебры.
    //
    // `Sparse + Auto` в итоговом плане LSODE2 разворачивается в sparse LU бэкенд.
    .with_linear_system_structure(Lsode2LinearSystemStructure::Sparse)
    .with_linear_solver_policy(Lsode2LinearSolverPolicy::Auto)
    // Шаг 4: выбираем faithful native BDF путь интегрирования.
    .with_faithful_bdf_solve(100_000, 100_000);

    // Шаг 5: запускаем через универсальный ODE-фасад.
    let mut solver = UniversalODESolver::lsode2_with_problem_config(config);
    solver.solve();

    let (t, y) = solver.get_result();
    let final_t = t.as_ref().map(|mesh| mesh[mesh.len() - 1]).unwrap_or(0.0);
    let final_y1 = y
        .as_ref()
        .map(|sol| sol[(sol.nrows() - 1, 0)])
        .unwrap_or(f64::NAN);
    let final_y2 = y
        .as_ref()
        .map(|sol| sol[(sol.nrows() - 1, 1)])
        .unwrap_or(f64::NAN);

    println!("LSODE2 Lambdify guide");
    println!("status  = {}", solver.get_status().unwrap_or_default());
    println!("final_t = {final_t:.6}");
    println!("final_y = [{final_y1:.8e}, {final_y2:.8e}]");
    println!(
        "note    = Lambdify route keeps setup simple; move to AOT when Jacobian evaluation becomes expensive."
    );
    println!(
        "note    = Dense/Banded routes use the same pattern: change linear structure + policy."
    );
    if let Some(stats) = solver.get_statistics() {
        println!("stats   = {}", stats.table_report());
    }
}
