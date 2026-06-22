//! BVP_sci AOT guide.
//!
//! This example demonstrates the generated-backend route for `BVP_sci`:
//! - sparse AtomView + `tcc`
//! - banded AtomView + `tcc`
//!
//! The first solve may build the native artifact; the second solve reuses the
//! same output directory so you can see the cold-to-warm lifecycle in a small
//! self-contained program.
//!
//! Run with:
//! `cargo run --example bvp_sci_aot_guide`

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
    println!("  grid points      = {}", result.ncols());
    println!("  max |y - x|      = {max_error:.3e}");
    println!("  iterations       = {}", stats.number_of_iterations);
    println!("  residual_ms      = {:.3}", stats.residual_ms_total);
    println!("  jacobian_ms      = {:.3}", stats.jacobian_ms_total);
    println!("  linear_ms        = {:.3}", stats.linear_system_ms_total);
    println!(
        "  symbolic_ms      = {:.3}",
        stats.symbolic_prepare_ms_total
    );
    println!();
}

fn sparse_solver(output_dir: impl Into<PathBuf>) -> BVPwrap {
    let output_dir: PathBuf = output_dir.into();
    let _ = fs::create_dir_all(&output_dir);
    BVPwrap::new_with_options(base_options().with_sparse_atomview_tcc(output_dir))
}

fn banded_solver(output_dir: impl Into<PathBuf>) -> BVPwrap {
    let output_dir: PathBuf = output_dir.into();
    let _ = fs::create_dir_all(&output_dir);
    BVPwrap::new_with_options(base_options().with_banded_atomview_tcc(output_dir))
}

fn main() {
    println!("BVP_sci AOT guide");
    println!("=================");
    println!("This guide demonstrates the generated-backend route.");
    println!("The same tiny BVP is solved in sparse and banded AOT form.");
    println!();

    if !command_exists("tcc") {
        println!("AOT cases skipped: `tcc` was not found in PATH");
        return;
    }

    let sparse_dir: PathBuf = "target/generated-bvp-sci-guides/aot-sparse-tcc".into();
    let banded_dir: PathBuf = "target/generated-bvp-sci-guides/aot-banded-tcc".into();

    println!("Sparse AOT:");
    run_case("  first run  / sparse + tcc", sparse_solver(&sparse_dir));
    run_case("  second run / sparse + tcc", sparse_solver(&sparse_dir));

    println!("Banded AOT:");
    run_case("  first run  / banded + tcc", banded_solver(&banded_dir));
    run_case("  second run / banded + tcc", banded_solver(&banded_dir));
}
