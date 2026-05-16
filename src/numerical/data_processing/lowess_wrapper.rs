use fastLowess::prelude;
use fastLowess::prelude::*;
use fastLowess::prelude::{CPU, GPU};
#[derive(Debug, Clone)]
pub enum Backend {
    GPUBack,
    CPUBack,
}

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
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.fraction) || self.fraction == 0.0 {
            return Err("LOWESS fraction must be in (0, 1]".to_string());
        }
        if self.delta < 0.0 {
            return Err("LOWESS delta must be >= 0".to_string());
        }
        Ok(())
    }
}

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
        /* *
        .iterations(config.iterations)
        .delta(config.delta)
        .weight_function(Tricube)
        .robustness_method(Bisquare)
        .zero_weight_fallback(UseLocalMean)
        .boundary_policy(Extend)
        .confidence_intervals(0.95)
        .prediction_intervals(0.95)
        .return_diagnostics()
        .return_residuals()
        .return_robustness_weights()
        .cross_validate(KFold(5, &[0.3, 0.5, 0.7]).seed(123))
        .auto_converge(1e-4)
          */
        .adapter(Batch)
        .parallel(config.parallel)
        .backend(CPU)
        .build()?;

    model.fit(x, y)
}

pub fn lowess_smooth_values(
    x: &[f64],
    y: &[f64],
    config: &LowessConfig,
) -> Result<Vec<f64>, LowessError> {
    let result = lowess_smoothing(x, y, config)?;
    Ok(result.y)
}
