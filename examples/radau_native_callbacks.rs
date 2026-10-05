//! Native Radau callbacks: analytic Jacobian and residual-only FD fallback.

use nalgebra::{DMatrix, DVector};
use RustedSciThe::numerical::Radau::{
    RadauConfig, RadauJacobianSource, RadauNativeSolver, RadauTelemetryMode,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = RadauConfig {
        t_bound: 0.5,
        first_step: Some(0.05),
        max_step: 0.1,
        telemetry: RadauTelemetryMode::Counters,
        ..RadauConfig::default()
    };

    let mut analytic = RadauNativeSolver::prepare(
        config.clone(),
        |_t, y: &DVector<f64>| DVector::from_vec(vec![-y[0]]),
        Some(|_t: f64, _y: &DVector<f64>| DMatrix::from_vec(1, 1, vec![-1.0])),
    )?;
    let analytic_solution = analytic.solve(&[1.0])?;

    let mut finite_difference = RadauNativeSolver::prepare(
        config,
        |_t, y: &DVector<f64>| DVector::from_vec(vec![-y[0]]),
        Option::<fn(f64, &DVector<f64>) -> DMatrix<f64>>::None,
    )?;
    let fd_solution = finite_difference.solve(&[1.0])?;

    println!(
        "analytic: source={:?} final={:.9e}",
        analytic.config().jacobian_source,
        analytic_solution.y[0]
    );
    println!(
        "finite-difference: source={:?} final={:.9e} probes={}",
        finite_difference.config().jacobian_source,
        fd_solution.y[0],
        fd_solution
            .telemetry()
            .counters
            .get("finite_difference_probes")
            .copied()
            .unwrap_or_default()
    );
    assert_eq!(
        analytic.config().jacobian_source,
        RadauJacobianSource::Analytic
    );
    assert_eq!(
        finite_difference.config().jacobian_source,
        RadauJacobianSource::FiniteDifference
    );
    Ok(())
}
