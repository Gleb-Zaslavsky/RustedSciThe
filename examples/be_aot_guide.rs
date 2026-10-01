//! Dense AtomViewNative AOT route through C/tcc with parameter continuation.
//! Run: `cargo run --no-default-features --example be_aot_guide` (requires tcc).

use RustedSciThe::numerical::BE::{BE, BeSolverOptions, BeStatus, BeSymbolicAssemblyBackend};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DVector;
use std::process::Command;

fn tcc_available() -> bool {
    let locator = if cfg!(windows) { "where" } else { "which" };
    Command::new(locator)
        .arg("tcc")
        .output()
        .is_ok_and(|output| output.status.success())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !tcc_available() {
        println!("Skipping dense AOT example: tcc was not found on PATH.");
        return Ok(());
    }

    let options = BeSolverOptions::new(
        vec![Expr::parse_expression("-rate*y")],
        vec!["y".into()],
        "t".into(),
        1e-12,
        8,
        Some(0.001),
        0.0,
        1.0,
        DVector::from_vec(vec![1.0]),
    )
    .with_symbolic_assembly_backend(BeSymbolicAssemblyBackend::AtomViewNative)
    .with_dense_generated_backend_c_tcc("target/generated-be-guides/aot-tcc");

    let mut solver = BE::try_new_with_options(options)?;
    solver.try_set_equation_parameters(Some(&["rate"]))?;
    for rate in [1.0, 2.0] {
        solver.set_parameter_values(DVector::from_vec(vec![rate]))?;
        solver.try_solve()?;
        assert_eq!(solver.status(), BeStatus::Finished);
        let (times, states) = solver.get_result();
        let times = times.expect("time samples");
        let states = states.expect("state samples");
        let y_end = states[(states.nrows() - 1, 0)];
        println!("AOT rate={rate}, y(1)={y_end:.8e}, samples={}", times.len());
    }
    println!("{}", solver.statistics_report());
    Ok(())
}
