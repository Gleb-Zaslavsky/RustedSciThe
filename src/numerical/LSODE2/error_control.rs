//! LSODE-style local error test helpers.
//!
//! The native LSODE loop needs a single place that turns a correction/LTE
//! vector into an ODEPACK-shaped weighted norm.  This module is intentionally
//! small: it does not decide how to retry a failed step, it only answers
//! whether the candidate step passes the local error test and what growth would
//! be reasonable after acceptance.

use super::history::{Lsode2HistoryError, Lsode2Tolerance, error_weights, weighted_rms_norm};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lsode2ErrorControlConfig {
    pub accept_threshold: f64,
    pub growth_safety: f64,
    pub max_growth: f64,
}

impl Default for Lsode2ErrorControlConfig {
    fn default() -> Self {
        Self {
            accept_threshold: 1.0,
            growth_safety: 0.9,
            max_growth: 5.0,
        }
    }
}

impl Lsode2ErrorControlConfig {
    pub fn validate(self) -> Result<(), Lsode2ErrorControlError> {
        if !self.accept_threshold.is_finite() || self.accept_threshold <= 0.0 {
            return Err(Lsode2ErrorControlError::InvalidConfig(
                "accept_threshold must be finite and positive",
            ));
        }
        if !self.growth_safety.is_finite() || self.growth_safety <= 0.0 || self.growth_safety > 1.0
        {
            return Err(Lsode2ErrorControlError::InvalidConfig(
                "growth_safety must be in (0, 1]",
            ));
        }
        if !self.max_growth.is_finite() || self.max_growth < 1.0 {
            return Err(Lsode2ErrorControlError::InvalidConfig(
                "max_growth must be finite and >= 1",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Lsode2ErrorControlError {
    InvalidConfig(&'static str),
    InvalidLocalErrorCoefficient(f64),
    History(Lsode2HistoryError),
}

impl std::fmt::Display for Lsode2ErrorControlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(message) => {
                write!(f, "invalid LSODE2 error-control config: {message}")
            }
            Self::InvalidLocalErrorCoefficient(value) => write!(
                f,
                "LSODE2 local error coefficient must be finite, got {value}"
            ),
            Self::History(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for Lsode2ErrorControlError {}

impl From<Lsode2HistoryError> for Lsode2ErrorControlError {
    fn from(value: Lsode2HistoryError) -> Self {
        Self::History(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lsode2ErrorTestAction {
    Accept,
    Reject,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Lsode2ErrorTestResult {
    pub action: Lsode2ErrorTestAction,
    pub error_norm: f64,
    pub suggested_growth: f64,
}

impl Lsode2ErrorTestResult {
    pub fn accepted(&self) -> bool {
        self.action == Lsode2ErrorTestAction::Accept
    }
}

#[derive(Debug, Clone)]
pub struct Lsode2ErrorController {
    config: Lsode2ErrorControlConfig,
    tolerance: Lsode2Tolerance,
}

impl Lsode2ErrorController {
    pub fn new(
        tolerance: Lsode2Tolerance,
        config: Lsode2ErrorControlConfig,
    ) -> Result<Self, Lsode2ErrorControlError> {
        config.validate()?;
        Ok(Self { config, tolerance })
    }

    pub fn scalar(
        rtol: f64,
        atol: f64,
        config: Lsode2ErrorControlConfig,
    ) -> Result<Self, Lsode2ErrorControlError> {
        Self::new(Lsode2Tolerance::scalar(rtol, atol), config)
    }

    pub fn config(&self) -> Lsode2ErrorControlConfig {
        self.config
    }

    pub fn tolerance(&self) -> &Lsode2Tolerance {
        &self.tolerance
    }

    pub fn weights_for(&self, y: &[f64]) -> Result<Vec<f64>, Lsode2ErrorControlError> {
        Ok(error_weights(y, &self.tolerance)?)
    }

    pub fn evaluate_local_error(
        &self,
        y_scale_source: &[f64],
        local_error: &[f64],
        order: usize,
    ) -> Result<Lsode2ErrorTestResult, Lsode2ErrorControlError> {
        self.evaluate_scaled_correction(y_scale_source, local_error, 1.0, order)
    }

    pub fn evaluate_scaled_correction(
        &self,
        y_scale_source: &[f64],
        correction: &[f64],
        local_error_coefficient: f64,
        order: usize,
    ) -> Result<Lsode2ErrorTestResult, Lsode2ErrorControlError> {
        if !local_error_coefficient.is_finite() {
            return Err(Lsode2ErrorControlError::InvalidLocalErrorCoefficient(
                local_error_coefficient,
            ));
        }
        let weights = self.weights_for(y_scale_source)?;
        let mut local_error = Vec::with_capacity(correction.len());
        local_error.extend(
            correction
                .iter()
                .map(|value| local_error_coefficient * *value),
        );
        let error_norm = weighted_rms_norm(&local_error, &weights)?;
        let action = if error_norm <= self.config.accept_threshold {
            Lsode2ErrorTestAction::Accept
        } else {
            Lsode2ErrorTestAction::Reject
        };
        Ok(Lsode2ErrorTestResult {
            action,
            error_norm,
            suggested_growth: self.suggested_growth(error_norm, order),
        })
    }

    fn suggested_growth(&self, error_norm: f64, order: usize) -> f64 {
        if !error_norm.is_finite() || error_norm <= 0.0 {
            return self.config.max_growth;
        }
        let exponent = 1.0 / (order + 1) as f64;
        let growth =
            self.config.growth_safety * (self.config.accept_threshold / error_norm).powf(exponent);
        growth.clamp(1.0, self.config.max_growth)
    }
}

impl Default for Lsode2ErrorController {
    fn default() -> Self {
        Self::scalar(1.0e-6, 1.0e-8, Lsode2ErrorControlConfig::default())
            .expect("default LSODE2 error-control config should be valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_controller_accepts_small_weighted_error_and_suggests_growth() {
        let controller =
            Lsode2ErrorController::scalar(1.0e-3, 1.0e-6, Lsode2ErrorControlConfig::default())
                .unwrap();

        let result = controller
            .evaluate_local_error(&[1.0, -2.0], &[1.0e-5, 1.0e-5], 2)
            .unwrap();

        assert_eq!(result.action, Lsode2ErrorTestAction::Accept);
        assert!(result.accepted());
        assert!(result.error_norm < 1.0);
        assert!(result.suggested_growth >= 1.0);
    }

    #[test]
    fn error_controller_rejects_large_weighted_error() {
        let controller =
            Lsode2ErrorController::scalar(1.0e-3, 1.0e-6, Lsode2ErrorControlConfig::default())
                .unwrap();

        let result = controller
            .evaluate_local_error(&[1.0], &[1.0e-1], 1)
            .unwrap();

        assert_eq!(result.action, Lsode2ErrorTestAction::Reject);
        assert!(result.error_norm > 1.0);
        assert_eq!(result.suggested_growth, 1.0);
    }

    #[test]
    fn correction_coefficient_participates_in_error_norm() {
        let controller =
            Lsode2ErrorController::scalar(1.0e-3, 1.0e-6, Lsode2ErrorControlConfig::default())
                .unwrap();

        let direct = controller
            .evaluate_local_error(&[1.0], &[1.0e-3], 1)
            .unwrap();
        let scaled = controller
            .evaluate_scaled_correction(&[1.0], &[1.0e-3], 0.01, 1)
            .unwrap();

        assert!(direct.error_norm > scaled.error_norm);
        assert_eq!(scaled.action, Lsode2ErrorTestAction::Accept);
    }

    #[test]
    fn error_controller_rejects_bad_config_and_dimension_mismatch() {
        let bad_config = Lsode2ErrorControlConfig {
            max_growth: 0.5,
            ..Lsode2ErrorControlConfig::default()
        };
        assert!(Lsode2ErrorController::scalar(1.0e-3, 1.0e-6, bad_config).is_err());

        let controller = Lsode2ErrorController::default();
        let err = controller
            .evaluate_local_error(&[1.0, 2.0], &[1.0], 1)
            .unwrap_err();
        assert!(matches!(
            err,
            Lsode2ErrorControlError::History(Lsode2HistoryError::DimensionMismatch {
                expected: 1,
                actual: 2
            })
        ));
    }
}
