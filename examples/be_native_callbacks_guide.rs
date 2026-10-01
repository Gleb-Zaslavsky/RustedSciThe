//! Minimal native-callback Backward Euler solve with an analytic dense Jacobian.
//! Run: `cargo run --no-default-features --example be_native_callbacks_guide`.

use RustedSciThe::numerical::BE::prelude::{BE, BeStatus, BeTelemetryMode, DMatrix, DVector};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rate = 2.0;
    let mut solver = BE::new();
    solver.try_set_telemetry_mode(BeTelemetryMode::Counters)?;
    solver.try_set_native_initial(
        vec!["y".into()],
        "t".into(),
        1e-12,
        8,
        Some(0.001),
        0.0,
        1.0,
        DVector::from_vec(vec![1.0]),
        move |_t, y| DVector::from_vec(vec![-rate * y[0]]),
        Some(move |_t, _y: &DVector<f64>| DMatrix::from_element(1, 1, -rate)),
    )?;

    solver.try_solve()?;
    assert_eq!(solver.status(), BeStatus::Finished);
    let (times, states) = solver.trajectory();
    let y_end = states[(states.nrows() - 1, 0)];
    let exact = (-rate * times[times.len() - 1]).exp();
    println!("samples={}, y(1)={y_end:.8}, exact={exact:.8}", times.len());
    println!("{}", solver.statistics_report());

    // For finite differences, pass `None::<fn(f64, &DVector<f64>) -> DMatrix<f64>>`
    // as the Jacobian instead of `Some(jacobian)`.
    Ok(())
}
