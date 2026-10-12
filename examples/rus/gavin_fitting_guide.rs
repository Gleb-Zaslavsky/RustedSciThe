//! Самостоятельная подгонка данных методом Levenberg-Marquardt Gavin.
//!
//! Модель содержит экспоненциальное затухание и постоянное смещение:
//! `y(x) = a * exp(-x / tau) + b`. Символьная модель подготавливается один
//! раз до численных итераций; ограничение `tau > 0` не допускает физически
//! бессмысленную область параметров.
//!
//! Запуск из корня репозитория:
//! `cargo run --example rus_gavin_fitting_guide --no-default-features`

use RustedSciThe::numerical::Nonlinear_systems::symbolic::SymbolicLambdifyFrontend;
use RustedSciThe::numerical::optimization::gavin_fitting::GavinSymbolicFitting;
use RustedSciThe::numerical::optimization::lm_gavin::GavinLmTermination;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x_data = (0..40).map(|i| i as f64 * 0.15).collect::<Vec<_>>();
    let y_data = x_data
        .iter()
        .map(|&x| 2.4 * (-x / 1.7_f64).exp() + 0.35)
        .collect::<Vec<_>>();

    let result = GavinSymbolicFitting::new()
        .try_with_equation("a * exp(-x / tau) + b")?
        .with_data(x_data, y_data)
        .with_sample_parameter("x")
        .with_unknowns(["a", "tau", "b"])
        .with_initial_guess(vec![1.8, 1.2, 0.2])
        .with_positive_parameters(["tau"])
        .with_frontend(SymbolicLambdifyFrontend::AtomViewNative)
        .with_telemetry(true)
        .try_solve()?;

    println!("Gavin symbolic fit");
    println!(
        "frontend: {:?}; backend: {:?}",
        result.frontend, result.backend
    );
    for (name, value) in result
        .parameter_names
        .iter()
        .zip(result.fit.parameters.iter())
    {
        println!("{name} = {value:.8}");
    }
    println!("termination: {:?}", result.fit.termination);
    println!("weighted SSE: {:.4e}", result.fit.sum_squared_residuals);
    println!("R^2: {:?}", result.fit.r_squared);

    if let Some(telemetry) = result.telemetry {
        println!(
            "iterations={} objective_calls={} jacobian_calls={} accepted={} rejected={} solve_ms={:.3}",
            telemetry.iterations,
            telemetry.objective_evaluations,
            telemetry.analytic_jacobian_evaluations,
            telemetry.accepted_steps,
            telemetry.rejected_steps,
            telemetry.total_time.as_secs_f64() * 1e3,
        );
    }

    // Exact synthetic data should converge rather than merely exhaust budget.
    assert!(matches!(
        result.fit.termination,
        GavinLmTermination::GradientTolerance | GavinLmTermination::ParameterStepTolerance
    ));
    Ok(())
}
