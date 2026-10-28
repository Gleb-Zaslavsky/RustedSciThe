//! Compare the two BVP_Damp routes through the typed BVP facade.
//!
//! Run with `cargo run --example bvp_api_damped_frozen`.

use RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_damped::DampedSolverOptions;
use RustedSciThe::numerical::BVP_Damp::NR_Damp_solver_frozen::FrozenSolverOptions;
use RustedSciThe::numerical::BVP_Damp::telemetry::BvpTelemetryMode;
use RustedSciThe::numerical::BVP_api::{
    BvpBoundaryCondition, BvpBoundarySide, BvpProblem, BvpSolution, BvpSolver, BvpSolverConfig,
    BvpSolverTelemetrySnapshot,
};
use RustedSciThe::symbolic::bvp::legacy::BvpSymbolicAssemblyBackend;
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DMatrix;
use std::collections::HashMap;

fn problem() -> Result<BvpProblem, Box<dyn std::error::Error>> {
    Ok(BvpProblem::new(
        vec![Expr::parse_expression("-y")],
        vec!["y".into()],
        "x",
        vec![0.0, 0.25, 0.5, 0.75, 1.0],
        vec![BvpBoundaryCondition {
            state: 0,
            side: BvpBoundarySide::Left,
            value: 1.0,
        }],
    )?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let damped_config = BvpSolverConfig::Damped {
        options: DampedSolverOptions::default()
            .with_bounds(HashMap::from([("y".to_string(), (-2.0, 2.0))]))
            .with_rel_tolerance(HashMap::from([("y".to_string(), 1e-6)]))
            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
            .with_bvp_telemetry_mode(BvpTelemetryMode::Counters),
        // Damped stores one Newton-unknown column per mesh interval.
        initial_guess: DMatrix::from_row_slice(1, 4, &[0.88, 0.69, 0.54, 0.42]),
    };
    let mut damped = BvpSolver::new(problem()?, damped_config)?;
    let BvpSolution::Damped { mesh, values } = damped.solve()? else {
        unreachable!("the selected config is Damped")
    };
    println!(
        "Damped: nodes={}, y(1)={:.8}",
        mesh.len(),
        values[(mesh.len() - 1, 0)]
    );
    let data = damped.postprocess_dataset()?;
    println!("Damped postprocess table: {:?}", data.values.shape());
    if let BvpSolverTelemetrySnapshot::Damped(snapshot) = damped.telemetry_snapshot() {
        println!("Damped native iterations: {}", snapshot.counters.iterations);
    }

    let frozen_config = BvpSolverConfig::Frozen {
        options: FrozenSolverOptions::default()
            .with_symbolic_assembly_backend(BvpSymbolicAssemblyBackend::AtomView)
            .with_bvp_telemetry_mode(BvpTelemetryMode::Counters),
        // Frozen/Naive stores one initial-guess column per mesh node.
        initial_guess: DMatrix::from_row_slice(1, 5, &[1.0, 0.78, 0.61, 0.47, 0.37]),
    };
    let mut frozen = BvpSolver::new(problem()?, frozen_config)?;
    let BvpSolution::Frozen { mesh, values } = frozen.solve()? else {
        unreachable!("the selected config is Frozen")
    };
    println!(
        "Frozen: nodes={}, y(1)={:.8}",
        mesh.len(),
        values[(mesh.len() - 1, 0)]
    );
    let data = frozen.postprocess_dataset()?;
    println!("Frozen postprocess table: {:?}", data.values.shape());
    if let BvpSolverTelemetrySnapshot::Frozen(snapshot) = frozen.telemetry_snapshot() {
        println!("Frozen native iterations: {}", snapshot.counters.iterations);
    }

    // Logging remains Off by default. Opt in through each native options type
    // with `with_bvp_logging_mode(...)`; the shared logger writes a timestamped
    // text file and setup failures are returned through BvpApiError.
    Ok(())
}
