//! Adaptive-step and Newton-controller policy.
//!
//! The controller is kept separate so its SciPy-compatible behavior can be
//! tested independently from callback execution and linear algebra.

use super::config::RadauConfig;

const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 10.0;

/// Predict the next step factor using SciPy Radau's one/two-step controller.
pub(crate) fn predict_factor(
    h_abs: f64,
    h_abs_old: Option<f64>,
    error_norm: f64,
    error_norm_old: Option<f64>,
) -> f64 {
    if !error_norm.is_finite() || error_norm < 0.0 {
        return MIN_FACTOR;
    }
    let multiplier = match (h_abs_old, error_norm_old) {
        (Some(old_h), Some(old_error)) if error_norm != 0.0 => {
            h_abs / old_h * (old_error / error_norm).powf(0.25)
        }
        _ => 1.0,
    };
    (multiplier.min(1.0) * error_norm.powf(-0.25)).clamp(MIN_FACTOR, MAX_FACTOR)
}

pub(crate) fn safety_factor(iterations: usize) -> f64 {
    SAFETY * (2.0 * 6.0 + 1.0) / (2.0 * 6.0 + iterations as f64)
}

pub(crate) fn min_step(t: f64, direction: f64) -> f64 {
    10.0 * (if direction > 0.0 {
        t.next_up() - t
    } else {
        t.next_down() - t
    })
    .abs()
}

/// Select the fallback initial absolute step when no RHS probe is available.
pub(crate) fn initial_step(config: &RadauConfig) -> f64 {
    let span = (config.t_bound - config.t0).abs();
    let candidate = config.first_step.unwrap_or(span * 0.01);
    candidate
        .min(config.max_step)
        .min(span)
        .max(f64::MIN_POSITIVE)
}

/// Select the first step after the two RHS probes used by SciPy's controller.
///
/// `f0` is evaluated at the initial state and `f1` at the short Euler probe
/// `y0 + direction * h0 * f0`.  The numerical runner owns both vectors, so
/// this function performs only scalar reductions and never allocates.
pub(crate) fn select_initial_step_from_probes(
    config: &RadauConfig,
    y0: &[f64],
    f0: &[f64],
    f1: &[f64],
    h0: f64,
) -> f64 {
    let interval = (config.t_bound - config.t0).abs();
    let mut d1_sum = 0.0;
    let mut d2_sum = 0.0;
    for index in 0..y0.len() {
        let scale = config.atol + config.rtol * y0[index].abs();
        let f_ratio = f0[index] / scale;
        d1_sum += f_ratio * f_ratio;
        if h0 > 0.0 {
            let derivative_ratio = (f1[index] - f0[index]) / h0 / scale;
            d2_sum += derivative_ratio * derivative_ratio;
        }
    }
    let d1 = (d1_sum / y0.len() as f64).sqrt();
    let d2 = if h0 > 0.0 {
        (d2_sum / y0.len() as f64).sqrt()
    } else {
        f64::INFINITY
    };
    let h1 = if d1 <= 1.0e-15 && d2 <= 1.0e-15 {
        (h0 * 1.0e-3).max(1.0e-6)
    } else {
        (0.01 / d1.max(d2)).powf(0.25)
    };
    (100.0 * h0)
        .min(h1)
        .min(interval)
        .min(config.max_step)
        .max(f64::MIN_POSITIVE)
}

/// Compute the short Euler probe step before the second RHS evaluation.
pub(crate) fn initial_probe_step(config: &RadauConfig, y0: &[f64], f0: &[f64]) -> f64 {
    let interval = (config.t_bound - config.t0).abs();
    let mut sum_y = 0.0;
    let mut sum_f = 0.0;
    for index in 0..y0.len() {
        let scale = config.atol + config.rtol * y0[index].abs();
        let y_ratio = y0[index] / scale;
        let f_ratio = f0[index] / scale;
        sum_y += y_ratio * y_ratio;
        sum_f += f_ratio * f_ratio;
    }
    let d0 = (sum_y / y0.len() as f64).sqrt();
    let d1 = (sum_f / y0.len() as f64).sqrt();
    let h0 = if d0 < 1.0e-5 || d1 < 1.0e-5 {
        1.0e-6
    } else {
        0.01 * d0 / d1
    };
    h0.min(interval).min(config.max_step).max(f64::MIN_POSITIVE)
}

/// Clip an absolute step to the remaining integration interval.
pub(crate) fn clamp_step_to_bound(t: f64, h_abs: f64, config: &RadauConfig) -> f64 {
    h_abs.min((config.t_bound - t).abs())
}

/// Check whether the directional integration interval has been completed.
pub(crate) fn is_finished(t: f64, config: &RadauConfig) -> bool {
    if config.t_bound > config.t0 {
        t >= config.t_bound
    } else {
        t <= config.t_bound
    }
}
