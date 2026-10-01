//! Shared independent IVP reference problems for numerical solver tests.
//!
//! Keep these fixtures solver-agnostic so BE, Radau and BDF can consume the
//! same initial conditions, right-hand sides and reference endpoints.

use nalgebra::{DMatrix, DVector};

pub(crate) struct StiffIvpCase {
    pub name: &'static str,
    pub t_end: f64,
    pub step: f64,
    pub initial: &'static [f64],
    pub reference: Option<&'static [f64]>,
    pub abs_tolerance: f64,
    pub rel_tolerance: f64,
    pub rhs: fn(f64, &DVector<f64>) -> DVector<f64>,
    pub jacobian: Option<fn(f64, &DVector<f64>) -> DMatrix<f64>>,
}

/// Robertson's stiff chemical kinetics problem. The t=1 endpoint is tabulated
/// from an independent RK4 run with h=1e-4 in Akinsola et al., Table 2:
/// <https://aujs.adelekeuniversity.edu.ng/index.php/aujs/article/download/102/74>.
pub(crate) fn robertson() -> StiffIvpCase {
    StiffIvpCase {
        name: "robertson",
        t_end: 1.0,
        step: 1.0e-4,
        initial: &[1.0, 0.0, 0.0],
        reference: Some(&[0.9664597405, 3.074626580e-5, 0.03350951681]),
        abs_tolerance: 2.0e-5,
        rel_tolerance: 2.0e-4,
        rhs: robertson_rhs,
        jacobian: Some(robertson_jacobian),
    }
}

/// HIRES benchmark equations from SciML's standard stiff ODE benchmark:
/// <https://docs.sciml.ai/SciMLBenchmarksOutput/v0.5/StiffODE/Hires/>.
/// The test computes an independent RK4 reference on a shorter stiff interval.
pub(crate) fn hires() -> StiffIvpCase {
    StiffIvpCase {
        name: "hires",
        t_end: 1.0,
        step: 0.001,
        initial: &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0057],
        reference: None,
        abs_tolerance: 1.0e-5,
        rel_tolerance: 5.0e-2,
        rhs: hires_rhs,
        jacobian: None,
    }
}

/// A ten-state stiff reaction/heat-release chain for solver-scale coverage.
/// The reference is generated with explicit RK4 at two step sizes and the
/// caller must establish refinement before using the finer endpoint.
pub(crate) fn combustion_chain_initial() -> DVector<f64> {
    DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.5, 300.0])
}

/// Nine reactive species (eight intermediates plus oxygen) coupled to one
/// thermal state; this deliberately stresses a wider nonlinear Jacobian than
/// the three-state combustion demonstration fixture.
pub(crate) fn combustion_chain_rhs(_: f64, y: &DVector<f64>) -> DVector<f64> {
    let mut dy = DVector::zeros(y.len());
    let temperature_factor = (0.003 * (y[9] - 300.0)).exp();
    let oxygen = y[8].max(0.0);
    let rates = [
        80.0 * temperature_factor * y[0].max(0.0) * oxygen,
        60.0 * temperature_factor * y[1].max(0.0) * oxygen,
        45.0 * temperature_factor * y[2].max(0.0) * oxygen,
        35.0 * temperature_factor * y[3].max(0.0) * oxygen,
        28.0 * temperature_factor * y[4].max(0.0) * oxygen,
        22.0 * temperature_factor * y[5].max(0.0) * oxygen,
        18.0 * temperature_factor * y[6].max(0.0) * oxygen,
        14.0 * temperature_factor * y[7].max(0.0) * oxygen,
    ];

    dy[0] = -rates[0];
    for index in 1..8 {
        dy[index] = rates[index - 1] - rates[index];
    }
    dy[8] = -0.5 * rates.iter().sum::<f64>();
    dy[9] = 0.04 * rates.iter().sum::<f64>() - 0.7 * (y[9] - 300.0);
    dy
}

/// Independent classical RK4 integrator for small deterministic reference
/// problems. It is deliberately not shared with any production solver path.
pub(crate) fn integrate_rk4(
    rhs: fn(f64, &DVector<f64>) -> DVector<f64>,
    t0: f64,
    t_end: f64,
    initial: &DVector<f64>,
    step: f64,
) -> DVector<f64> {
    assert!(step.is_finite() && step > 0.0);
    let steps = ((t_end - t0) / step).ceil() as usize;
    let mut t = t0;
    let mut y = initial.clone();
    for index in 0..steps {
        let h = (t_end - t).min(step);
        let k1 = rhs(t, &y);
        let k2 = rhs(t + 0.5 * h, &(&y + 0.5 * h * &k1));
        let k3 = rhs(t + 0.5 * h, &(&y + 0.5 * h * &k2));
        let k4 = rhs(t + h, &(&y + h * &k3));
        y += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        t = if index + 1 == steps { t_end } else { t + h };
    }
    y
}

fn robertson_rhs(_: f64, y: &DVector<f64>) -> DVector<f64> {
    let r1 = 0.04 * y[0];
    let r2 = 1.0e4 * y[1] * y[2];
    let r3 = 3.0e7 * y[1] * y[1];
    DVector::from_vec(vec![-r1 + r2, r1 - r2 - r3, r3])
}

fn robertson_jacobian(_: f64, y: &DVector<f64>) -> DMatrix<f64> {
    DMatrix::from_row_slice(
        3,
        3,
        &[
            -0.04,
            1.0e4 * y[2],
            1.0e4 * y[1],
            0.04,
            -1.0e4 * y[2] - 6.0e7 * y[1],
            -1.0e4 * y[1],
            0.0,
            6.0e7 * y[1],
            0.0,
        ],
    )
}

fn hires_rhs(_: f64, y: &DVector<f64>) -> DVector<f64> {
    DVector::from_vec(vec![
        -1.71 * y[0] + 0.43 * y[1] + 8.32 * y[2] - 7.0e-4,
        1.71 * y[0] - 8.75 * y[1],
        -10.03 * y[2] + 0.43 * y[3] + 0.035 * y[4],
        8.32 * y[1] + 1.71 * y[2] - 1.12 * y[3],
        -1.745 * y[4] + 0.43 * y[5] + 0.43 * y[6],
        -280.0 * y[5] * y[7] + 0.69 * y[3] + 1.71 * y[4] - 0.43 * y[5] + 0.69 * y[6],
        280.0 * y[5] * y[7] - 1.81 * y[6],
        -1.81 * y[6] + 1.81 * y[7],
    ])
}
