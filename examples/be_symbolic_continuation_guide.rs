//! Symbolic AtomViewNative + Lambdify; change a parameter and continue accepted state.
//! Run: `cargo run --no-default-features --example be_symbolic_continuation_guide`.

use RustedSciThe::numerical::BE::{BE, BeSolverOptions, BeStatus, BeSymbolicAssemblyBackend};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::DVector;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = BeSolverOptions::new(
        vec![Expr::parse_expression("-rate*y")],
        vec!["y".into()],
        "t".into(),
        1e-12,
        8,
        Some(0.001),
        0.0,
        0.5,
        DVector::from_vec(vec![1.0]),
    )
    .with_symbolic_assembly_backend(BeSymbolicAssemblyBackend::AtomViewNative);
    let mut solver = BE::try_new_with_options(options)?;
    solver.try_set_equation_parameters(Some(&["rate"]))?;
    solver.set_parameter_values(DVector::from_vec(vec![1.0]))?;
    solver.try_solve()?;
    assert_eq!(solver.status(), BeStatus::Finished);

    let (_, first_segment) = solver.get_result();
    let y_half = first_segment.expect("first segment").column(0)[500];
    solver.set_parameter_values(DVector::from_vec(vec![2.0]))?;
    solver.try_continue_to(1.0)?;

    let (times, states) = solver.get_result();
    let times = times.expect("time samples");
    let states = states.expect("state samples");
    let y_end = states[(states.nrows() - 1, 0)];
    let continued_exact = y_half * (-2.0_f64 * 0.5).exp();
    assert!((y_end - continued_exact).abs() < 2e-3);
    println!("continued from t=0.5, y={y_half:.8}; y(1)={y_end:.8}");
    println!(
        "samples={}, final_time={:.3}",
        times.len(),
        times[times.len() - 1]
    );
    println!("{}", solver.statistics_report());
    Ok(())
}
