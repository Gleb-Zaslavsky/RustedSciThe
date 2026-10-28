// Matrix-layout guide for the new BVP_sci API.
// The backend is selected once when the solver is built. This example uses
// the same ExprLegacy problem for Dense, Sparse and Banded storage; it does
// not use the legacy BVP_sci wrapper or string backend flags.
// Run with `cargo run --example bvp_sci_backends_guide`.

use RustedSciThe::numerical::BVP_sci::{
    BvpSciAssembly, BvpSciBoundaryCallbacks, BvpSciExecution, BvpSciLambdifyPlan,
    BvpSciMatrixLayout, BvpSciOptions, BvpSciSolver, BvpSciTelemetry,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;

fn run(layout: BvpSciMatrixLayout) {
    let telemetry = BvpSciTelemetry::timings();
    let plan = BvpSciLambdifyPlan::prepare(
        BvpSciAssembly::ExprLegacy,
        &[Expr::parse_expression("1"), Expr::parse_expression("1")],
        &["y0".into(), "y1".into()],
        &[],
        "x",
        telemetry.clone(),
    )
    .expect("ExprLegacy plan should prepare");
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
        .with_execution(BvpSciExecution::Lambdify)
        .with_assembly(BvpSciAssembly::ExprLegacy)
        .with_matrix_layout(layout)
        .with_tolerance(1e-8);
    let mut solver = BvpSciSolver::new(
        plan,
        boundary,
        vec![0.0, 0.5, 1.0],
        vec![0.0, 1.0, 0.5, 1.0, 1.0, 1.0],
        vec![],
        options,
    )
    .expect("backend solver should construct");
    let solution = solver.solve().expect("backend solve should converge");
    let snapshot = solver.plan().telemetry_snapshot();
    println!(
        "{layout:?} | {:.6e} | {:.3e} | {:.3} | {:.3} | {}",
        solution.y[solution.y.len() - 2],
        solution.residual_norm,
        snapshot.linear_assembly_ms.unwrap_or_default(),
        snapshot.factorization_ms.unwrap_or_default(),
        snapshot.linear_solves,
    );
}

fn main() {
    println!("layout | y(1) | residual_norm | linear_assembly_ms | factorization_ms | solves");
    run(BvpSciMatrixLayout::Dense);
    run(BvpSciMatrixLayout::Sparse);
    run(BvpSciMatrixLayout::BandedAuto);
}

// Optional SuperLU Sparse route
// -----------------------------
// SuperLU is an alternative factorization policy for the Sparse layout, not a
// fourth matrix layout. StandardFaer remains the default. The small C ABI
// bridge must be built for the target OS before enabling the Rust feature.
// Build prerequisites are Git, CMake, Ninja and a C compiler; the scripts pin
// SuperLU 6.0.1 and build its bundled CBLAS into the bridge.
//
// Windows (PowerShell; the script writes an ignored DLL under test_reports/):
//   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_superlu_bridge.ps1
//   $env:SUPERLU_LIBRARY = (Resolve-Path '.\test_reports\somelinalg_sparse_lu_reference\native\superlu_bridge.dll').Path
//
// Linux (the script writes an ignored shared object under test_reports/):
//   bash scripts/build_superlu_bridge.sh
//   export SUPERLU_LIBRARY="$(pwd)/test_reports/somelinalg_sparse_lu_reference/native/libsuperlu_bridge.so"
//
// Compile and run this guide with the optional feature enabled:
//   cargo run --no-default-features --features superlu --example bvp_sci_backends_guide
//
// Select SuperLU through the builder (the default `new`/builder configuration
// continues to use StandardFaer):
//   let solver = BvpSciSolver::builder(rhs, state_names)
//       .with_frontend(BvpSciAssembly::ExprLegacy)
//       .with_superlu() // also selects BvpSciMatrixLayout::Sparse
//       .with_mesh_and_initial_state(mesh, initial_state)
//       .with_boundary_callback(boundary)
//       .build()?;
//
// If the Cargo feature is disabled or the bridge cannot be loaded/validated,
// BVP_sci warns and falls back to StandardFaer. Check
// `telemetry_snapshot().superlu_fallbacks == 0` before treating a run as
// SuperLU performance evidence. See `src/numerical/BVP_sci/SUPERLU_BACKEND.md`
// for the complete native-build and release-matrix instructions.
