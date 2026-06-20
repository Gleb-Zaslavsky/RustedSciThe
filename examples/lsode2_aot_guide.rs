//! LSODE2 AOT Guide
//!
//! Goal:
//! run symbolic LSODE2 with ahead-of-time generated callbacks.
//!
//! AOT route is useful when symbolic residual/Jacobian evaluation is expensive:
//! generated code is compiled once, then reused.
//!
//! In this guide we use:
//! - symbolic assembly: `AtomView`
//! - symbolic execution: `AOT(C + tcc, Release)`
//! - linear structure: sparse
//! - solve path: faithful native BDF
//! - backend chunking tuned for parallel runtime work splitting
//!
//! Note:
//! this example intentionally checks that `tcc` exists before trying to build.
//! 
//! run cargo run --example lsode2_aot_guide

use RustedSciThe::numerical::LSODE2::{
    Lsode2AotProfile, Lsode2AotToolchain, Lsode2BackendConfig, Lsode2LinearSolverPolicy,
    Lsode2LinearSystemStructure, Lsode2ProblemConfig, Lsode2ResidualJacobianSource,
    Lsode2SymbolicAssemblyBackend, Lsode2SymbolicExecutionMode,
};
use RustedSciThe::numerical::ODE_api2::UniversalODESolver;
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DVector;
use std::process::Command;

fn command_available(command: &str) -> bool {
    let probe = if cfg!(windows) { "where" } else { "which" };
    Command::new(probe)
        .arg(command)
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn main() {
    if !command_available("tcc") {
        println!("LSODE2 AOT guide skipped: `tcc` is not available in PATH.");
        println!("Install tcc or switch to another toolchain (gcc/zig/rust) in the config.");
        return;
    }

    // Step 1: symbolic equations and base integration settings.
    let config = Lsode2ProblemConfig::new(
        vec![
            Expr::parse_expression("-15.0*y1 + 14.0*y2"),
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
    // Step 2: configure generated backend artifacts/output.
    //
    // This controls where generated source/build artifacts are placed.
    // Reusing the same folder enables warm/prelinked behavior across runs.
    .with_backend(Lsode2BackendConfig::native_sparse_faer_aot_c_tcc(
        "target/lsode2-aot-guide",
    ))
    // Step 3: tell LSODE2 that symbolic execution should be AOT.
    //
    // Toolchain/profile here define the compilation mode for generated
    // callbacks, not the Rust crate build profile.
    .with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
        assembly: Lsode2SymbolicAssemblyBackend::AtomView,
        execution: Lsode2SymbolicExecutionMode::Aot {
            toolchain: Lsode2AotToolchain::CTcc,
            profile: Lsode2AotProfile::Release,
        },
    })
    // Step 4: sparse Jacobian structure + auto solver policy => sparse LU.
    .with_linear_system_structure(Lsode2LinearSystemStructure::Sparse)
    .with_linear_solver_policy(Lsode2LinearSolverPolicy::Auto)
    // Optional but practical:
    // configure generated backend chunking for parallel runtimes.
    //
    // The same knob affects residual chunking and Jacobian chunking
    // (dense + sparse generated routes). `2` means "two chunks per worker"
    // in the internal recommendation policy.
    .with_aot_parallel_chunking(2)
    // Step 5: faithful native BDF step engine.
    .with_faithful_bdf_solve(100_000, 100_000);

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

    println!("LSODE2 AOT guide (C+tcc)");
    println!("status  = {}", solver.get_status().unwrap_or_default());
    println!("final_t = {final_t:.6}");
    println!("final_y = [{final_y1:.8e}, {final_y2:.8e}]");
    println!(
        "note    = first run may spend time in codegen/compile; repeated runs reuse artifacts."
    );
    println!(
        "note    = for Dense/Banded, keep the same AOT execution and switch linear structure/policy."
    );
    if let Some(stats) = solver.get_statistics() {
        println!("stats   = {}", stats.table_report());
    }
}
