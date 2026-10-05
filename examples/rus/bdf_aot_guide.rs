//! Плотный BDF AtomView AOT через C/tcc с parameter continuation.
//! Запуск: `cargo run --no-default-features --example rus_bdf_aot_guide` (нужен tcc).

use RustedSciThe::numerical::BDF::BDF_api::{
    BdfSolverOptions, BdfStatus, BdfTelemetryMode, ODEsolver,
};
use RustedSciThe::symbolic::symbolic_engine::Expr;
use RustedSciThe::symbolic::symbolic_ivp::IvpSymbolicAssemblyBackend;
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
        println!("Пример пропущен: tcc не найден в PATH.");
        return Ok(());
    }

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
    .with_dense_generated_backend_c_tcc("target/generated-bdf-guides/aot-tcc")
    .with_telemetry_mode(BdfTelemetryMode::Counters);

    let mut solver = ODEsolver::new_with_options(options);
    solver.try_solve()?;
    assert_eq!(solver.status_kind(), BdfStatus::Finished);

    // Скомпилированные callbacks и загруженный артефакт остаются готовыми;
    // меняются только параметр и численная история нового сегмента.
    solver.try_continue_with_parameter_values(DVector::from_vec(vec![2.0]), 1.0)?;
    solver.try_solve()?;
    assert_eq!(solver.status_kind(), BdfStatus::Finished);

    let (times, states) = solver.get_result_ref();
    let final_value = states[(states.nrows() - 1, 0)];
    let exact = (-1.5_f64).exp();
    assert!((final_value - exact).abs() < 2e-8);

    println!("Пример continuation для плотного BDF AtomView AOT");
    println!(
        "точек второго сегмента={}, y(1)={final_value:.10e}, точно={exact:.10e}",
        times.len()
    );
    println!("{}", solver.statistics_report());
    Ok(())
}
