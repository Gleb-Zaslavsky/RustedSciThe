//! Гайд по BVP Damped Lambdify
//!
//! Цель:
//! решить символическую BVP через Lambdify без сборки внешнего AOT-артефакта.
//!
//! Lambdify означает, что пользователь пишет уравнения как `Expr`, RustedSciThe
//! дифференцирует и дискретизирует их, а затем привязывает вызываемые Rust
//! closures для Newton-цикла.
//!
//! Задача:
//! `y' = z`, `z' = 0`, `y(0) = 0`, `z(1) = 1`, with exact `y(x) = x`.
//!
//! Почему этот гайд важен:
//! - Lambdify - естественный первый путь для символических моделей;
//! - `AtomView` - оптимизированный символический frontend;
//! - `Banded` соответствует узкой связности, которую даёт локальный BVP stencil.
//! 
//! запуск: cargo run --example bvp_damped_lambdify_guide

use std::collections::HashMap;

use RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_damped::{
    DampedSolverOptions, NRBVP, SolverParams,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DMatrix;

const N_STEPS: usize = 80;

fn initial_guess() -> DMatrix<f64> {
    let h = 1.0 / N_STEPS as f64;
    let mut guess = vec![0.0; 2 * N_STEPS];
    for i in 0..N_STEPS {
        guess[2 * i] = i as f64 * h;
        guess[2 * i + 1] = 1.0;
    }
    DMatrix::from_column_slice(2, N_STEPS, &guess)
}

fn main() {
    // Шаг 1: выбираем нелинейные настройки и символический путь исполнения.
    // `with_banded_lambdify()` выбирает Banded-assembly матрицы с AtomView
    // уравнениями, которые вычисляются через Lambdify-closures.
    let options = DampedSolverOptions::sparse_damped()
        .with_strategy_params(Some(SolverParams::default()))
        .with_abs_tolerance(1e-8)
        .with_rel_tolerance(HashMap::from([
            ("y".to_string(), 1e-6),
            ("z".to_string(), 1e-6),
        ]))
        .with_bounds(HashMap::from([
            ("y".to_string(), (-0.2, 1.2)),
            ("z".to_string(), (-2.0, 2.0)),
        ]))
        .with_banded_lambdify()
        .with_max_iterations(40)
        .with_loglevel(Some("none".to_string()));

    // Шаг 2: строим символическую задачу и её граничные условия.
    let mut solver = NRBVP::new_with_options(
        vec![Expr::parse_expression("z"), Expr::parse_expression("0.0")],
        initial_guess(),
        vec!["y".to_string(), "z".to_string()],
        "x".to_string(),
        HashMap::from([
            ("y".to_string(), vec![(0, 0.0)]),
            ("z".to_string(), vec![(1, 1.0)]),
        ]),
        0.0,
        1.0,
        N_STEPS,
        options,
    );
    solver.dont_save_log(true);
    solver.try_solve().expect("Lambdify BVP must solve");

    let result = solver.get_result().expect("solver must store a result");
    let h = 1.0 / N_STEPS as f64;
    let max_error = (0..result.nrows())
        .map(|i| (result[(i, 0)] - i as f64 * h).abs())
        .fold(0.0_f64, f64::max);
    println!(
        "Damped Banded Lambdify/AtomView: rows={}, max |y-x|={max_error:.3e}",
        result.nrows()
    );
    let stats = solver.get_statistics();
    println!(
        "  selected_backend = {}",
        stats
            .diagnostics
            .get("generated.selected_backend")
            .map(String::as_str)
            .unwrap_or("unknown")
    );
    println!(
        "  assembly         = {}",
        stats
            .diagnostics
            .get("generated.symbolic_assembly_backend")
            .map(String::as_str)
            .unwrap_or("unknown")
    );
    println!(
        "  iterations       = {}",
        stats
            .counters
            .get("number of iterations")
            .copied()
            .unwrap_or(0)
    );
    assert!(max_error < 5e-5, "max error is {max_error:e}");
}
