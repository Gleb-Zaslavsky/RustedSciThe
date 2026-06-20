//! BVP Damped AOT Guide
//!
//! Goal:
//! show a symbolic generated-backend lifecycle: build a native artifact once,
//! then solve again while requiring that prebuilt artifact.
//!
//! Ahead-of-time (AOT) execution lowers the residual and Jacobian callbacks to
//! source code and compiles a native artifact. Building it adds startup work;
//! reuse can pay off for large or repeatedly solved BVPs. This example uses
//! the practical AtomView + Banded + C/tcc route.
//!
//! `BuildIfMissing` creates or locates the artifact on the first solve and
//! returns an updated resolver snapshot.
//! `RequirePrebuilt` on the second solve documents the production deployment
//! contract: do not silently compile when a prepared artifact is expected,
//! but do keep using the snapshot returned by the build phase.
//! The executable example needs `tcc` available on `PATH`.
//! run cargo run --example bvp_damped_aot_guide 
use std::collections::HashMap;
use std::process::Command;

use RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_damped::{
    DampedSolverOptions, NRBVP, SolverParams,
};
use RustedSciThe::numerical::BVP_Damp::generated_solver_handoff::{
    AotBuildPolicy, GeneratedBackendConfig,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DMatrix;

const N_STEPS: usize = 80;

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
        .with_generated_backend_config(config)
        .with_max_iterations(40)
        .with_loglevel(Some("none".to_string()));
    NRBVP::new_with_options(
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
    )
}

fn report(label: &str, solver: &NRBVP) {
    let result = solver.get_result().expect("solver must store a result");
    let h = 1.0 / N_STEPS as f64;
    let max_error = (0..result.nrows())
        .map(|i| (result[(i, 0)] - i as f64 * h).abs())
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
    assert!(max_error < 5e-5, "max error is {max_error:e}");
}

fn main() {
    if !command_exists("tcc") {
        println!("Install `tcc` and place it on PATH to run the AOT guide.");
        return;
    }

    // Step 1: allow cold creation. For production workloads this is commonly
    // performed in an initialization/deployment phase.
    let config = GeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc();
    let mut build_or_reuse = solver_with(config.clone());
    build_or_reuse.dont_save_log(true);
    build_or_reuse
        .try_solve()
        .expect("AOT build-if-missing solve must succeed");
    report("Damped AOT cold/build-if-missing solve", &build_or_reuse);

    // Step 2: require reuse. The second solver must be seeded with the
    // resolver snapshot returned by the first build-if-missing run.
    let strict_config = config
        .with_resolver(build_or_reuse.aot_resolver().cloned())
        .with_aot_build_policy(AotBuildPolicy::RequirePrebuilt);
    let mut prebuilt = solver_with(strict_config);
    prebuilt.dont_save_log(true);
    prebuilt
        .try_solve()
        .expect("prebuilt AOT artifact must be reusable");
    report("Damped AOT strict prebuilt solve", &prebuilt);
}
