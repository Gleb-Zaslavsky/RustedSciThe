//! IVP backend guide for dense implicit solvers.
//!
//! This example is the user-facing summary of the current IVP backend story:
//! - `Lambdify` is still the safest default for small systems.
//! - `C + tcc` is the first compiled backend to try for larger or stiffer
//!   Backward Euler problems when you want a native Jacobian/residual path
//!   without paying a large startup tax.
//! - `C + gcc` is the runtime-oriented option when repeated solves matter more
//!   than time-to-first-solution.
//! - `BDF` behaves differently from `BE`: many practical BDF scenarios remain
//!   residual-dominated, so `Lambdify` is still a very strong baseline there.
//!
//! Questions this example answers:
//! 1. How do I solve the same IVP through plain `Lambdify`?
//! 2. How do I switch that IVP to `C + tcc` or `C + gcc`?
//! 3. Which backend is a reasonable default for `BE` and `BDF`?
//! 4. Where do I get statistics instead of reading test code?
//! 
//! run cargo run --example ivp_backends_guide


use std::process::Command;
use std::time::Instant;

use RustedSciThe::numerical::BDF::BDF_api::{BdfSolverOptions, ODEsolver};
use RustedSciThe::numerical::BE::{BE, BeSolverOptions};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::{DMatrix, DVector};

fn command_available(command: &str) -> bool {
    let probe = if cfg!(windows) { "where" } else { "which" };
    Command::new(probe)
        .arg(command)
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn small_stiff_equations() -> (Vec<Expr>, Vec<String>, DVector<f64>) {
    (
        vec![
            Expr::parse_expression("-15.0*y1 + 14.0*y2"),
            Expr::parse_expression("y1 - y2"),
        ],
        vec!["y1".to_string(), "y2".to_string()],
        DVector::from_vec(vec![1.0, 0.0]),
    )
}

fn print_guide() {
    println!("IVP backend guide");
    println!("=================");
    println!("Recommended defaults:");
    println!("  - Small BE/BDF systems: Lambdify");
    println!("  - Larger stiff BE systems: try C-tcc first");
    println!("  - Repeated runtime-oriented dense solves: C-gcc");
    println!("  - BDF often stays residual-dominated, so benchmark before");
    println!("    replacing Lambdify by default.");
    println!();
    println!("Where to look for evidence:");
    println!("  - examples/ivp_backends_guide.rs: quick runnable overview");
    println!("  - codegen_ivp_backend_comparison_tests.rs: detailed tables");
    println!("  - solver.get_statistics() / solver.statistics_report():");
    println!("    programmatic setup/solve/callback counters and timings");
    println!();
}

fn run_be_case(label: &str, options: BeSolverOptions) {
    let started = Instant::now();
    let mut solver = BE::new_with_options(options);
    solver.solve();
    let total_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let stats = solver.get_statistics();
    let (_, maybe_y) = solver.get_result();
    let max_abs_solution = maybe_y
        .as_ref()
        .map(|y| y.iter().map(|value| value.abs()).fold(0.0, f64::max))
        .unwrap_or(0.0);
    println!(
        "{label:18} total_ms = {total_ms:9.3}, max_abs_solution = {max_abs_solution:.6e}, stats = {}",
        stats.table_report()
    );
}

fn run_bdf_case(label: &str, options: BdfSolverOptions) {
    let started = Instant::now();
    let mut solver = ODEsolver::new_with_options(options);
    solver.solve();
    let total_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let stats = solver.get_statistics();
    let (_, y) = solver.get_result();
    let max_abs_solution = y.iter().map(|value| value.abs()).fold(0.0, f64::max);
    println!(
        "{label:18} total_ms = {total_ms:9.3}, max_abs_solution = {max_abs_solution:.6e}, stats = {}",
        stats.table_report()
    );
}

fn main() {
    let (equations, values, y0) = small_stiff_equations();
    print_guide();

    println!("Demo problem: small stiff 2x2 IVP");
    println!();

    println!("Backward Euler:");
    let be_lambdify = BeSolverOptions::new(
        equations.clone(),
        values.clone(),
        "t".to_string(),
        1e-9,
        25,
        Some(0.02),
        0.0,
        5.0,
        y0.clone(),
    );
    run_be_case("Lambdify", be_lambdify);

    if command_available("tcc") {
        run_be_case(
            "C-tcc",
            BeSolverOptions::new(
                equations.clone(),
                values.clone(),
                "t".to_string(),
                1e-9,
                25,
                Some(0.02),
                0.0,
                5.0,
                y0.clone(),
            )
            .with_dense_generated_backend_c_tcc("target/generated-ivp-example"),
        );
    } else {
        println!("C-tcc              skipped: `tcc` was not found in PATH");
    }

    if command_available("gcc") {
        run_be_case(
            "C-gcc",
            BeSolverOptions::new(
                equations.clone(),
                values.clone(),
                "t".to_string(),
                1e-9,
                25,
                Some(0.02),
                0.0,
                5.0,
                y0.clone(),
            )
            .with_dense_generated_backend_c_gcc("target/generated-ivp-example"),
        );
    } else {
        println!("C-gcc              skipped: `gcc` was not found in PATH");
    }

    println!();
    println!("BDF:");
    let bdf_lambdify = BdfSolverOptions::new(
        equations.clone(),
        values.clone(),
        "t".to_string(),
        "BDF".to_string(),
        0.0,
        y0.clone(),
        5.0,
        0.05,
        1e-7,
        1e-9,
        None::<DMatrix<f64>>,
        false,
        None,
    );
    run_bdf_case("Lambdify", bdf_lambdify);

    if command_available("tcc") {
        run_bdf_case(
            "C-tcc",
            BdfSolverOptions::new(
                equations.clone(),
                values.clone(),
                "t".to_string(),
                "BDF".to_string(),
                0.0,
                y0.clone(),
                5.0,
                0.05,
                1e-7,
                1e-9,
                None::<DMatrix<f64>>,
                false,
                None,
            )
            .with_dense_generated_backend_c_tcc("target/generated-ivp-example"),
        );
    } else {
        println!("C-tcc              skipped: `tcc` was not found in PATH");
    }

    if command_available("gcc") {
        run_bdf_case(
            "C-gcc",
            BdfSolverOptions::new(
                equations,
                values,
                "t".to_string(),
                "BDF".to_string(),
                0.0,
                y0,
                5.0,
                0.05,
                1e-7,
                1e-9,
                None::<DMatrix<f64>>,
                false,
                None,
            )
            .with_dense_generated_backend_c_gcc("target/generated-ivp-example"),
        );
    } else {
        println!("C-gcc              skipped: `gcc` was not found in PATH");
    }
}
