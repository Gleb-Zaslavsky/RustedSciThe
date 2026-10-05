//! Плотный BDF с численными callbacks и постоянным аналитическим Jacobian.
//! Запуск: `cargo run --no-default-features --example rus_bdf_native_dense_guide`.

use RustedSciThe::numerical::BDF::BDF_api::{
    BdfNativeJacobianSource, BdfSolverOptions, BdfStatus, BdfTelemetryMode, ODEsolver,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use nalgebra::{DMatrix, DVector};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rate = 20.0;
    let options = BdfSolverOptions::for_bdf(
        vec![Expr::parse_expression("-20*y")],
        vec!["y".into()],
        "t".into(),
        0.0,
        DVector::from_vec(vec![1.0]),
        1.0,
        0.05,
        1e-8,
        1e-11,
        None,
        false,
        Some(1e-3),
    )
    .with_telemetry_mode(BdfTelemetryMode::Counters)
    .with_max_steps(100_000);

    let mut solver = ODEsolver::new_with_options(options);
    solver.set_native_ode_callbacks_with_jacobian_source(
        move |_t, y| DVector::from_vec(vec![-rate * y[0]]),
        BdfNativeJacobianSource::Constant(DMatrix::from_element(1, 1, -rate)),
    );
    solver.try_solve()?;

    assert_eq!(solver.status_kind(), BdfStatus::Finished);
    let (times, states) = solver.get_result_ref();
    let t_end = times[times.len() - 1];
    let y_end = states[(states.nrows() - 1, 0)];
    let exact = (-rate * t_end).exp();
    assert!((y_end - exact).abs() < 2e-8);

    println!("Пример плотного BDF с native callbacks");
    println!(
        "точек={}, y({t_end:.3})={y_end:.10e}, точно={exact:.10e}",
        times.len()
    );
    println!("{}", solver.statistics_report());
    Ok(())
}
