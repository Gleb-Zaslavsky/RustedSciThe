//! BVP Frozen AOT Guide
//!
//! Goal:
//! show the compiled callback lifecycle while using the Frozen Newton
//! strategy.
//!
//! Frozen supports the same symbolic generated-backend lifecycle as Damped:
//! an AtomView expression assembly pass can create a Banded native callback
//! artifact, and subsequent solves can explicitly require that artifact.
//! This is especially useful when a family of boundary data is solved against
//! one compiled differential system.
//!
//! The guide uses the C/tcc AOT route; `tcc` must be installed and visible on
//! `PATH`.
//! 
//! run cargo run --example bvp_frozen_aot_guide

use std::collections::HashMap;
use std::process::Command;

use RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_frozen::{FrozenSolverOptions, NRBVP};
use RustedSciThe::numerical::BVP_Damp::generated_solver_handoff::{
    AotBuildPolicy, GeneratedBackendConfig,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DMatrix;

const N_STEPS: usize = 40;

fn command_exists(name: &str) -> bool {
    let locator = if cfg!(windows) { "where" } else { "which" };
    Command::new(locator)
        .arg(name)
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn initial_guess() -> DMatrix<f64> {
    let h = 1.0 / N_STEPS as f64;
    let mut guess = vec![0.0; 2 * N_STEPS];
    for i in 0..N_STEPS {
        guess[2 * i] = i as f64 * h;
        guess[2 * i + 1] = 1.0;
    }
    DMatrix::from_column_slice(2, N_STEPS, &guess)
}

fn solver_with(config: GeneratedBackendConfig) -> NRBVP {
    let options = FrozenSolverOptions::banded_frozen()
        .with_generated_backend_config(config)
        .with_tolerance(1e-8)
        .with_max_iterations(40);
    NRBVP::new_with_options(
        vec![Expr::parse_expression("z"), Expr::parse_expression("0.0")],
        initial_guess(),
        vec!["y".to_string(), "z".to_string()],
        "x".to_string(),
        HashMap::from([
            ("y".to_string(), vec![(0, 0.0)]),
            ("z".to_string(), vec![(0, 1.0)]),
        ]),
        0.0,
        1.0,
        N_STEPS,
        options,
    )
}

fn report(label: &str, solver: &NRBVP) {
    let result = solver.get_result().expect("solver must store a result");
    let h = 1.0 / N_STEPS as f64;
    let max_error = (0..result.nrows())
        .map(|i| (result[(i, 0)] - (i + 1) as f64 * h).abs())
        .fold(0.0_f64, f64::max);
    let stats = solver.get_statistics();
    println!("{label}");
    println!("  max |y-x|       = {max_error:.3e}");
    println!(
        "  selected_backend = {}",
        stats
            .diagnostics
            .get("generated.selected_backend")
            .map(String::as_str)
            .unwrap_or("unknown")
    );
    println!(
        "  build_policy     = {}",
        stats
            .diagnostics
            .get("aot.build_policy")
            .map(String::as_str)
            .unwrap_or("unknown")
    );
    assert!(max_error < 1e-7, "max error is {max_error:e}");
}

fn main() {
    if !command_exists("tcc") {
        println!("Install `tcc` and place it on PATH to run the AOT guide.");
        return;
    }

    // Step 1: build an artifact if it is not already present.
    let config = GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc();
    let mut build_or_reuse = solver_with(config.clone());
    build_or_reuse.dont_save_log(true);
    build_or_reuse
        .try_solve()
        .expect("Frozen AOT build-if-missing solve must succeed");
    report("Frozen AOT cold/build-if-missing solve", &build_or_reuse);

    // Step 2: demonstrate the deployment/repeated-solve contract.
    // Seed the second solver with the resolver snapshot returned by the
    // initial build-if-missing pass.
    let strict_config = config
        .with_resolver(build_or_reuse.aot_resolver().cloned())
        .with_aot_build_policy(AotBuildPolicy::RequirePrebuilt);
    let mut prebuilt = solver_with(strict_config);
    prebuilt.dont_save_log(true);
    prebuilt
        .try_solve()
        .expect("Frozen prebuilt AOT artifact must be reusable");
    report("Frozen AOT strict prebuilt solve", &prebuilt);
}
