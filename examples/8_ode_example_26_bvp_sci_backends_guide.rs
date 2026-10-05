//! Руководство по бэкендам BVP_sci.
//!
//! Этот пример показывает три основные пользовательские symbolic-ветки на
//! одной и той же учебной краевой задаче:
//! - `ExprLegacy + Lambdify` как простой базовый маршрут;
//! - `AtomView + AOT sparse` как практический route с generated-backend;
//! - `AtomView + AOT banded` когда матричная структура достаточно узкая и
//!   banded solver действительно уместен.
//!
//! Связанные примеры:
//! - `examples/bvp_sci_numerical_guide.rs` - чисто числовой route без символики;
//! - `examples/bvp_sci_lambdify_guide.rs` - symbolic route через Lambdify;
//! - `examples/bvp_sci_aot_guide.rs` - lifecycle generated-backend.
//!
//! Запуск:
//! `cargo run --example rus_bvp_sci_backends_guide`

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

use nalgebra::DMatrix;

use RustedSciThe::numerical::BVP_sci::BVP_sci_symb::{BVPwrap, BvpSciSolverOptions};
use RustedSciThe::symbolic::symbolic_engine::Expr;

const N_STEPS: usize = 64;

fn command_exists(name: &str) -> bool {
    let locator = if cfg!(windows) { "where" } else { "which" };
    Command::new(locator)
        .arg(name)
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn initial_guess() -> DMatrix<f64> {
    DMatrix::from_element(2, N_STEPS, 0.5)
}

fn base_options() -> BvpSciSolverOptions {
    let equations = vec![Expr::parse_expression("z"), Expr::parse_expression("0.0")];
    let boundary_conditions = HashMap::from([
        ("y".to_string(), vec![(0usize, 0.0f64)]),
        ("z".to_string(), vec![(1usize, 1.0f64)]),
    ]);

    BvpSciSolverOptions::new(
        None,
        Some(0.0),
        Some(1.0),
        Some(N_STEPS),
        equations,
        vec!["y".to_string(), "z".to_string()],
        vec![],
        None,
        boundary_conditions,
        "x".to_string(),
        1e-8,
        256,
        initial_guess(),
    )
    .with_loglevel(Some("none".to_string()))
}

fn run_case(label: &str, mut solver: BVPwrap) {
    solver
        .try_solve()
        .unwrap_or_else(|err| panic!("{label} failed: {err:?}"));
    let result = solver.get_result().expect("solver should store a result");
    let stats = solver.get_statistics();
    let max_error = (0..result.ncols())
        .map(|i| (result[(0, i)] - i as f64 / (N_STEPS - 1) as f64).abs())
        .fold(0.0_f64, f64::max);

    println!("{label}");
    println!("  узлов сетки       = {}", result.ncols());
    println!("  max |y - x|       = {max_error:.3e}");
    println!("  итераций          = {}", stats.number_of_iterations);
    println!("  residual_ms       = {:.3}", stats.residual_ms_total);
    println!("  jacobian_ms       = {:.3}", stats.jacobian_ms_total);
    println!("  linear_system_ms  = {:.3}", stats.linear_system_ms_total);
    println!("  symbolic_ms       = {:.3}", stats.symbolic_prepare_ms_total);
    println!();
}

fn lambdify_solver() -> BVPwrap {
    BVPwrap::new_with_options(base_options()).with_expr_legacy_smart_sparse()
}

fn sparse_aot_solver() -> BVPwrap {
    let output_dir: PathBuf = "target/generated-bvp-sci-guides/sparse-tcc".into();
    let _ = fs::create_dir_all(&output_dir);
    BVPwrap::new_with_options(base_options().with_sparse_atomview_tcc(output_dir))
}

fn banded_aot_solver() -> BVPwrap {
    let output_dir: PathBuf = "target/generated-bvp-sci-guides/banded-tcc".into();
    let _ = fs::create_dir_all(&output_dir);
    BVPwrap::new_with_options(base_options().with_banded_atomview_tcc(output_dir))
}

fn main() {
    println!("Руководство по backend-ам BVP_sci");
    println!("================================");
    println!("В этом примере сравниваются основные symbolic routes на простой BVP.");
    println!("Так проще увидеть, где заканчивается Lambdify и где начинаются generated backends.");
    println!();

    run_case("Lambdify / ExprLegacy", lambdify_solver());

    if command_exists("tcc") {
        run_case("AtomView / sparse AOT / tcc", sparse_aot_solver());
        run_case("AtomView / banded AOT / tcc", banded_aot_solver());
    } else {
        println!("AOT-случаи пропущены: `tcc` не найден в PATH");
    }

    println!("См. также:");
    println!("  - cargo run --example rus_bvp_sci_lambdify_guide");
    println!("  - cargo run --example rus_bvp_sci_aot_guide");
    println!("  - cargo run --example rus_bvp_sci_numerical_guide");
}
