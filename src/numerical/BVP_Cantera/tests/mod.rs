use super::*;

fn scalar_solver(initial: f64, target: f64, jacobian_value: f64) -> CanteraNewtonSolver {
    let mut config = CanteraNewtonConfig::new(1, 1e-8, 0.0);
    config.telemetry_mode = CanteraTelemetryMode::Counters;
    CanteraNewtonSolver::new(
        DVector::from_element(1, initial),
        move |state| {
            Ok(DVector::from_element(
                1,
                jacobian_value * (state[0] - target),
            ))
        },
        move |_| Ok(DMatrix::from_element(1, 1, jacobian_value)),
        config,
    )
    .unwrap()
}

mod large_workloads;
mod newton;
mod parameter_loop;
mod symbolic;
mod telemetry;
