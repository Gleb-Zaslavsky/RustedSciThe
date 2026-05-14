//! Boundary-value problem guide for the damped BVP solver.
//!
//! This example is meant to be the single user-facing entry point for the
//! current BVP backend story:
//! - `Lambdify + ExprLegacy` is the safest one-shot baseline.
//! - `AtomView + C-tcc` is the practical recommendation for repeated solves on
//!   the same symbolic problem.
//! - `AtomView + C-gcc` is useful when native callback throughput matters more
//!   than startup latency.
//!
//! Questions this example answers:
//! 1. How do I solve a simple BVP through the plain lambdify path?
//! 2. How do I switch the same problem to the AOT path?
//! 3. Where do I choose `ExprLegacy` vs `AtomView`?
//! 4. Where do I choose the native compiler (`tcc`, `gcc`)?
//! 5. Where do I override chunking or execution policy?
//!
//! Related runnable examples:
//! - `examples/bvp_sci_numerical_guide.rs` shows the pure numerical no-symbolic
//!   `BVP_sci` API in both `FiniteDifference` and `AnalyticalPointwise` modes.
//! - `examples/bvp_sci_numerical_parameters_guide.rs` shows how the same pure
//!   numerical API solves for unknown scalar parameters together with the state.
//!
//! Practical guidance from current BVP diagnostics:
//! - Use `Lambdify` when you expect to solve the problem only once.
//! - Use `AtomView + C-tcc` when the same symbolic problem will be solved again
//!   with new initial guesses, parameter values, or solver settings.
//! - Use `AtomView + C-gcc` when compile latency is acceptable and you want a
//!   strong native-runtime baseline.
//! - On large combustion grids the picture shifts: on a stronger workstation,
//!   `AtomView + C-tcc/gcc` overtook `Lambdify` by roughly 20-35% around
//!   `n_steps = 5000`, with `C-tcc` slightly ahead. In practice this means the
//!   break-even point for one-shot combustion solves sits somewhere above
//!   `n_steps = 2000` and around the "tens of thousands of discrete equations"
//!   regime, after which AOT keeps pulling away.
//! - Rust and Zig AOT backends exist lower in the stack, but the most polished
//!   BVP-facing choices today are `Lambdify`, `C-tcc`, and `C-gcc`.
/*

|discretization,       |
| sym.jacobian stage   | turning into functions stage |	compilation and linking stage| loop stage|
|______________________|______________________________|______________________________|___________|
|                      |                              |				     |	     	 |
|		               | Classic Lambdification        |				     |		 |
| Classic Symbolic     |______________________________| not needed                   |		 |
|		               |			                      |			             |		 |
|                      | Classic AOT                  |				     |		 |
|______________________|______________________________|______________________________|___________|
|		       |	                      |				     |		 |
| AtomView Symbolic    |  AtomView Lambdification     | C (tcc or gcc),		     |		 |
|		       |______________________________| Zig			     |		 |
|		       |			      | Rust			     |		 |
|		       |  AtomView AOT                |				     |		 |
|		       |______________________________|______________________________|___________|

*/
use std::collections::HashMap;
use std::process::Command;
use std::time::Instant;

use RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_damped::{
    DampedSolverOptions, NRBVP, SolverParams,
};
use RustedSciThe::numerical::BVP_Damp::generated_solver_handoff::{
    AotBuildPolicy, AotBuildProfile, AotChunkingPolicy, AotExecutionPolicy, GeneratedBackendConfig,
};
use RustedSciThe::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
use RustedSciThe::symbolic::codegen::codegen_orchestrator::ParallelExecutorConfig;
use RustedSciThe::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy;
use RustedSciThe::symbolic::codegen::codegen_tasks::SparseChunkingStrategy;
use RustedSciThe::symbolic::symbolic_engine::Expr;
use RustedSciThe::symbolic::symbolic_functions_BVP::BvpSymbolicAssemblyBackend;
use nalgebra::{DMatrix, DVector};

fn uniform_initial_guess(variable_count: usize, n_steps: usize, value: f64) -> DMatrix<f64> {
    DMatrix::from_column_slice(
        variable_count,
        n_steps,
        DVector::from_element(variable_count * n_steps, value).as_slice(),
    )
}

fn simple_linear_bvp(options: DampedSolverOptions, n_steps: usize) -> NRBVP {
    // Solve y'' = 0 with y(0) = 0 and y(1) = 1.
    // First-order form:
    // y' = z
    // z' = 0
    let equations = vec![Expr::parse_expression("z"), Expr::parse_expression("0.0")];
    let boundary_conditions = HashMap::from([
        ("y".to_string(), vec![(0usize, 0.0f64)]),
        ("z".to_string(), vec![(1usize, 1.0f64)]),
    ]);
    let initial_guess = uniform_initial_guess(2, n_steps, 0.5);

    let mut solver = NRBVP::new_with_options(
        equations,
        initial_guess,
        vec!["y".to_string(), "z".to_string()],
        "x".to_string(),
        boundary_conditions,
        0.0,
        1.0,
        n_steps,
        options,
    );
    solver.dont_save_log(true);
    solver
}

fn base_options() -> DampedSolverOptions {
    let rel_tolerance = HashMap::from([("y".to_string(), 1e-7), ("z".to_string(), 1e-7)]);
    let bounds = HashMap::from([
        ("y".to_string(), (-10.0, 10.0)),
        ("z".to_string(), (-10.0, 10.0)),
    ]);
    let strategy_params = SolverParams {
        max_jac: Some(4),
        max_damp_iter: Some(5),
        damp_factor: Some(0.5),
        adaptive: None,
    };

    DampedSolverOptions::sparse_damped()
        .with_strategy_params(Some(strategy_params))
        .with_abs_tolerance(1e-8)
        .with_rel_tolerance(rel_tolerance)
        .with_bounds(bounds)
        .with_max_iterations(40)
        .with_loglevel(Some("error".to_string()))
}

fn lambdify_options() -> DampedSolverOptions {
    let config = GeneratedBackendConfig::default()
        .with_backend_policy_override(Some(BackendSelectionPolicy::LambdifyOnly))
        .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::ExprLegacy);
    base_options().with_generated_backend_config(config)
}

fn atomview_tcc_options() -> DampedSolverOptions {
    base_options().with_sparse_atomview_for_repeated_solves()
}

fn atomview_gcc_options() -> DampedSolverOptions {
    base_options().with_sparse_atomview_c_gcc()
}

fn manual_atomview_tcc_options() -> DampedSolverOptions {
    // This is the "all knobs visible" version.
    //
    // Change these pieces when you need to tune behavior:
    // - symbolic assembly backend: ExprLegacy vs AtomView
    // - native compiler: tcc vs gcc
    // - chunking: residual / sparse Jacobian splitting
    // - execution policy: sequential vs parallel runtime
    // - build policy: build-if-missing vs rebuild-always
    let config = GeneratedBackendConfig::sparse_build_if_missing_release()
        .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
        .with_aot_codegen_backend(
            RustedSciThe::symbolic::codegen::codegen_aot_driver::AotCodegenBackend::C,
        )
        .with_aot_c_compiler("tcc")
        .with_aot_execution_policy(AotExecutionPolicy::Parallel(
            ParallelExecutorConfig::default(),
        ))
        .with_aot_build_policy(AotBuildPolicy::BuildIfMissing {
            profile: AotBuildProfile::Release,
        })
        .with_aot_compile_dev_fastest()
        .with_aot_chunking_policy(AotChunkingPolicy::with_parts(
            Some(ResidualChunkingStrategy::ByTargetChunkCount { target_chunks: 4 }),
            Some(SparseChunkingStrategy::ByRowCount { rows_per_chunk: 16 }),
        ));

    base_options().with_generated_backend_config(config)
}

fn command_available(command: &str) -> bool {
    let probe = if cfg!(windows) { "where" } else { "which" };
    Command::new(probe)
        .arg(command)
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn run_case(label: &str, options: DampedSolverOptions, n_steps: usize) {
    let started = Instant::now();
    let mut solver = simple_linear_bvp(options, n_steps);
    let solution = solver
        .try_solve()
        .unwrap_or_else(|err| panic!("{label} failed: {err:?}"))
        .expect("solver should converge on the simple linear BVP");
    let total_ms = started.elapsed().as_secs_f64() * 1_000.0;

    let max_abs_solution = solution.iter().map(|value| value.abs()).fold(0.0, f64::max);

    println!("{label:24} total_ms = {total_ms:9.3}, max_abs_solution = {max_abs_solution:.6e}");
}

fn print_guide() {
    println!("BVP backend guide");
    println!("=================");
    println!("Recommended defaults:");
    println!("  - One-shot solve: Lambdify + ExprLegacy");
    println!("  - Repeated solves: AtomView + C-tcc");
    println!("  - Throughput-oriented native path: AtomView + C-gcc");
    println!("  - Large combustion-like grids: once you reach the tens-of-thousands");
    println!("    of discrete equations, AtomView + C-tcc/gcc can overtake one-shot");
    println!("    Lambdify end-to-end, with C-tcc currently the most practical choice.");
    println!();
    println!("Important knobs:");
    println!("  - Symbolic assembly: ExprLegacy vs AtomView");
    println!("  - Native compiler: tcc vs gcc");
    println!("  - Chunking: residual and sparse Jacobian chunk strategies");
    println!("  - Build policy: BuildIfMissing vs RebuildAlways");
    println!("  - Compile preset: Production / FastBuild / DevFastest");
    println!();
    println!("Where to look next:");
    println!("  - examples/bvp_sci_numerical_guide.rs:");
    println!("    pure numerical BVP_sci without symbolic APIs");
    println!("  - examples/bvp_sci_numerical_parameters_guide.rs:");
    println!("    pure numerical BVP_sci with solved parameters");
    println!();
    println!("Chunking tips:");
    println!("  - Start with backend defaults.");
    println!("  - Use coarse chunk counts for large problems when native runtime");
    println!("    parallelism is desirable.");
    println!("  - Whole-chunk mode is simplest when investigating correctness.");
    println!();
}

fn main() {
    let n_steps = 64;
    print_guide();

    println!("Demo problem: y'' = 0, y(0) = 0, y(1) = 1, n_steps = {n_steps}");
    println!();

    run_case("Lambdify / ExprLegacy", lambdify_options(), n_steps);

    if command_available("tcc") {
        run_case("AtomView / C-tcc", atomview_tcc_options(), n_steps);
    } else {
        println!("AtomView / C-tcc       skipped: `tcc` was not found in PATH");
    }

    if command_available("gcc") {
        run_case("AtomView / C-gcc", atomview_gcc_options(), n_steps);
    } else {
        println!("AtomView / C-gcc       skipped: `gcc` was not found in PATH");
    }

    println!();
    println!("Manual configuration example:");
    println!("  - same demo problem, but with explicit chunking/build/execution knobs exposed");
    if command_available("tcc") {
        run_case(
            "AtomView / manual C-tcc",
            manual_atomview_tcc_options(),
            n_steps,
        );
    } else {
        println!("AtomView / manual C-tcc skipped: `tcc` was not found in PATH");
    }
}
