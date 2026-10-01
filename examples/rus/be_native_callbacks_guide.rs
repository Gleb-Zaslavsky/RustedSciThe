//! Численный Backward Euler с аналитическим плотным якобианом.
//! Запуск: `cargo run --no-default-features --example rus_be_native_callbacks_guide`.

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
    println!(
        "отсчётов={}, y(1)={y_end:.8}, точное={exact:.8}",
        times.len()
    );
    println!("{}", solver.statistics_report());
    Ok(())
}
