//! Руководство по AOT-ветке новой архитектуры `BVP_sci`.
//!
//! AOT использует тот же collocation solver, что и Lambdify, но публикует
//! скомпилированные callback-артефакты. В примере проверяются Dense, Sparse и
//! Banded layout через AtomViewNative и политика `BuildIfMissing`. При первом
//! запуске компилятор создаёт артефакт, а повторный запуск может использовать
//! тот же каталог вывода.
//!
//! Запуск требует `tcc` в PATH:
//! `cargo run --example 8_ode_example_25_bvp_sci_aot_guide`

use std::path::PathBuf;
use std::process::Command;

use RustedSciThe::numerical::BVP_sci::{
    BvpSciAssembly, BvpSciBoundaryCallbacks, BvpSciExecution, BvpSciLambdifyPlan,
    BvpSciMatrixLayout, BvpSciOptions, BvpSciSolver, BvpSciTelemetry,
    SymbolicIvpGeneratedBackendConfig,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;

fn compiler_available() -> bool {
    let locator = if cfg!(windows) { "where" } else { "which" };
    Command::new(locator)
        .arg("tcc")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn run(layout: BvpSciMatrixLayout, output_dir: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let telemetry = BvpSciTelemetry::timings();
    let plan = BvpSciLambdifyPlan::prepare_aot(
        BvpSciAssembly::AtomViewNative,
        layout,
        // Зависимость от состояния делает Jacobian непустым и проверяет
        // настоящий AOT callback ABI, а не специальный constant-residual case.
        vec![Expr::parse_expression("y0"), Expr::parse_expression("y1")],
        vec!["y0".into(), "y1".into()],
        vec![],
        "x",
        SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_dir).with_c_tcc(),
        telemetry.clone(),
    )?;
    let boundary = BvpSciBoundaryCallbacks::new(
        2,
        |ya, yb, _parameters, output| {
            output[0] = ya[0];
            output[1] = yb[1] - 1.0;
            Ok(())
        },
        telemetry,
    );
    let options = BvpSciOptions::default()
        .with_execution(BvpSciExecution::Aot)
        .with_matrix_layout(layout)
        .with_tolerance(1e-8);
    let mut solver = BvpSciSolver::new(
        plan,
        boundary,
        vec![0.0, 0.5, 1.0],
        vec![0.0, 1.0, 0.5, 1.0, 1.0, 1.0],
        vec![],
        options,
    )?;
    let solution = solver.solve()?;
    let snapshot = solver.plan().telemetry_snapshot();
    println!(
        "{layout:?} | {:.6e} | {:.3e} | {:.3} | {:.3} | {}",
        solution.y[solution.y.len() - 2],
        solution.residual_norm,
        snapshot.preparation_ms.unwrap_or_default(),
        snapshot.full_solve_ms.unwrap_or_default(),
        snapshot.jacobian_evaluations,
    );
    Ok(())
}

fn main() {
    if !compiler_available() {
        println!("AOT guide skipped: tcc was not found in PATH");
        return;
    }
    println!("layout | y(1) | residual_norm | preparation_ms | full_solve_ms | jacobian_calls");
    let root = PathBuf::from("target/generated-bvp-sci-guides/aot");
    for (name, layout) in [
        ("dense", BvpSciMatrixLayout::Dense),
        ("sparse", BvpSciMatrixLayout::Sparse),
        ("banded", BvpSciMatrixLayout::Banded { lower: 1, upper: 1 }),
    ] {
        // Ошибка одного layout не скрывает результаты остальных строк.
        if let Err(error) = run(layout, root.join(name)) {
            eprintln!("AOT {name} failed: {error}");
        }
    }
}
