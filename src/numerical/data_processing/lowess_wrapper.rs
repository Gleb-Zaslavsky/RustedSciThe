//! Small, stable wrapper around the `fastLowess` batch API.
//!
//! `fastLowess` 2.x makes [`Lowess`] itself the in-memory batch entry point;
//! callers no longer select a `Batch` adapter explicitly. The public builder
//! currently exposes CPU parallelism, while its runtime GPU backend type is
//! not exported from the stable prelude. We therefore reject `GPUBack`
//! explicitly instead of silently executing a requested GPU calculation on
//! the CPU.

use fastLowess::prelude::{Lowess, LowessError, LowessResult};

/// Requested execution backend for LOWESS smoothing.
#[derive(Debug, Clone)]
pub enum Backend {
    /// GPU execution. `fastLowess` 2.x does not expose this through its stable
    /// batch builder API yet, so this request currently returns an error.
    GPUBack,
    /// CPU execution, optionally parallelized by `fastLowess`/Rayon.
    CPUBack,
}

/// Configuration forwarded to the `fastLowess` 2.x batch builder.
#[derive(Debug, Clone)]
pub struct LowessConfig {
    pub fraction: f64,
    pub iterations: usize,
    pub delta: f64,
    pub parallel: bool,
    pub backend: Backend,
}

impl Default for LowessConfig {
    fn default() -> Self {
        Self {
            fraction: 0.5,
            iterations: 3,
            delta: 0.01,
            parallel: true,
            backend: Backend::CPUBack,
        }
    }
}

impl LowessConfig {
    /// Validate wrapper-level settings before constructing a LOWESS model.
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.fraction) || self.fraction == 0.0 {
            return Err("LOWESS fraction must be in (0, 1]".to_string());
        }
        if self.delta < 0.0 {
            return Err("LOWESS delta must be >= 0".to_string());
        }
        if matches!(self.backend, Backend::GPUBack) {
            return Err(
                "LOWESS GPU backend is not available through the fastLowess 2.x public batch API"
                    .to_string(),
            );
        }
        Ok(())
    }
}

/// Fit a LOWESS model and return the complete `fastLowess` result.
pub fn lowess_smoothing(
    x: &[f64],
    y: &[f64],
    config: &LowessConfig,
) -> Result<LowessResult<f64>, LowessError> {
    if let Err(msg) = config.validate() {
        return Err(LowessError::InvalidInput(msg));
    }

    let model = Lowess::new()
        .fraction(config.fraction)
        .iterations(config.iterations)
        .delta(config.delta)
        .parallel(config.parallel)
        .build()?;

    model.fit(x, y)
}

/// Fit LOWESS and return only the smoothed ordinate values.
pub fn lowess_smooth_values(
    x: &[f64],
    y: &[f64],
    config: &LowessConfig,
) -> Result<Vec<f64>, LowessError> {
    let result = lowess_smoothing(x, y, config)?;
    Ok(result.y)
}

#[cfg(test)]
mod tests {
    use super::{Backend, LowessConfig, lowess_smooth_values};

    #[test]
    fn cpu_lowess_smoothing_returns_one_finite_value_per_sample() {
        let x: Vec<f64> = (0..12).map(f64::from).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|value| value * value + 0.1 * value.sin())
            .collect();
        let config = LowessConfig {
            fraction: 0.5,
            iterations: 2,
            delta: 0.0,
            parallel: false,
            backend: Backend::CPUBack,
        };

        let smoothed = lowess_smooth_values(&x, &y, &config).expect("CPU LOWESS should fit");
        assert_eq!(smoothed.len(), y.len());
        assert!(smoothed.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn gpu_request_is_rejected_instead_of_falling_back_to_cpu() {
        let config = LowessConfig {
            backend: Backend::GPUBack,
            ..LowessConfig::default()
        };

        let error = config
            .validate()
            .expect_err("GPU must not silently use the CPU backend");
        assert!(error.contains("GPU backend"));
    }
}
