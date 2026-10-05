//! Dense AtomView + Lambdify BDF with prepared parameter continuation.
//! Run: `cargo run --no-default-features --example bdf_symbolic_continuation_guide`.

use RustedSciThe::numerical::BDF::prelude::{
    BdfSolverOptions, BdfStatus, BdfTelemetryMode, IvpSymbolicAssemblyBackend, ODEsolver,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DVector;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = BdfSolverOptions::for_bdf(
        vec![Expr::parse_expression("-rate*y")],
        vec!["y".into()],
        "t".into(),
        0.0,
        DVector::from_vec(vec![1.0]),
        0.5,
        0.05,
        1e-8,
        1e-11,
        None,
        false,
        Some(1e-3),
    )
    .with_symbolic_assembly_backend(IvpSymbolicAssemblyBackend::AtomView)
    .with_equation_parameters(vec!["rate".into()])
    .with_equation_parameter_values(DVector::from_vec(vec![1.0]))
    .with_telemetry_mode(BdfTelemetryMode::Counters);

    let mut solver = ODEsolver::new_with_options(options);
    solver.try_solve()?;
    assert_eq!(solver.status_kind(), BdfStatus::Finished);
    let first_end = {
        let (_, states) = solver.get_result_ref();
        states[(states.nrows() - 1, 0)]
    };

    // This updates the shared parameter slot and restarts only numerical BDF
    // history. Prepared residual/Jacobian callbacks are retained.
    solver.try_continue_with_parameter_values(DVector::from_vec(vec![2.0]), 1.0)?;
    solver.try_solve()?;
    assert_eq!(solver.status_kind(), BdfStatus::Finished);

    let (times, states) = solver.get_result_ref();
    let final_value = states[(states.nrows() - 1, 0)];
    let exact = (-0.5_f64 - 2.0 * 0.5).exp();
    assert!((final_value - exact).abs() < 2e-8);

    println!("dense BDF AtomView/Lambdify continuation guide");
    println!("first segment y(0.5)={first_end:.10e}");
    println!(
        "continued segment samples={}, y(1)={final_value:.10e}, exact={exact:.10e}",
        times.len()
    );
    println!("{}", solver.statistics_report());
    Ok(())
}
