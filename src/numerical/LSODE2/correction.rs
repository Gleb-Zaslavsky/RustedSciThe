//! LSODE-style Newton correction bookkeeping.
//!
//! This module does not solve linear systems by itself.  Its job is to inspect
//! already computed Newton corrections, track convergence/divergence across
//! iterations, and translate a converged correction into a local-error estimate
//! for the surrounding step controller.

use super::dcfode::{Lsode2BdfDcfodeTables, Lsode2DcfodeError};
use super::history::{Lsode2HistoryError, Lsode2Tolerance, error_weights, weighted_rms_norm};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lsode2CorrectionControlConfig {
    pub divergence_ratio: f64,
    pub rate_memory_weight: f64,
    pub initial_convergence_rate: f64,
    pub max_iterations: usize,
}

impl Default for Lsode2CorrectionControlConfig {
    fn default() -> Self {
        Self {
            divergence_ratio: 2.0,
            rate_memory_weight: 0.2,
            initial_convergence_rate: 0.7,
            max_iterations: 4,
        }
    }
}

impl Lsode2CorrectionControlConfig {
    pub fn validate(self) -> Result<(), Lsode2CorrectionError> {
        if !self.divergence_ratio.is_finite() || self.divergence_ratio < 1.0 {
            return Err(Lsode2CorrectionError::InvalidConfig(
                "divergence_ratio must be finite and >= 1",
            ));
        }
        if !self.rate_memory_weight.is_finite() || !(0.0..=1.0).contains(&self.rate_memory_weight) {
            return Err(Lsode2CorrectionError::InvalidConfig(
                "rate_memory_weight must be finite and lie in [0, 1]",
            ));
        }
        if !self.initial_convergence_rate.is_finite() || self.initial_convergence_rate <= 0.0 {
            return Err(Lsode2CorrectionError::InvalidConfig(
                "initial_convergence_rate must be finite and positive",
            ));
        }
        if self.max_iterations == 0 {
            return Err(Lsode2CorrectionError::InvalidConfig(
                "max_iterations must be at least 1",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Lsode2CorrectionError {
    InvalidConfig(&'static str),
    History(Lsode2HistoryError),
    Dcfode(Lsode2DcfodeError),
}

impl std::fmt::Display for Lsode2CorrectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(message) => {
                write!(f, "invalid LSODE2 correction config: {message}")
            }
            Self::History(err) => write!(f, "{err}"),
            Self::Dcfode(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for Lsode2CorrectionError {}

impl From<Lsode2HistoryError> for Lsode2CorrectionError {
    fn from(value: Lsode2HistoryError) -> Self {
        Self::History(value)
    }
}

impl From<Lsode2DcfodeError> for Lsode2CorrectionError {
    fn from(value: Lsode2DcfodeError) -> Self {
        Self::Dcfode(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lsode2CorrectionStatus {
    Continue,
    Converged,
    Diverged,
    IterationLimitReached,
}

impl Lsode2CorrectionStatus {
    pub fn label(self) -> &'static str {
        match self {
            Self::Continue => "continue",
            Self::Converged => "converged",
            Self::Diverged => "diverged",
            Self::IterationLimitReached => "iteration_limit_reached",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Lsode2CorrectionAssessment {
    pub order: usize,
    pub iteration: usize,
    pub weighted_norm: f64,
    pub accumulated_weighted_norm: f64,
    pub previous_weighted_norm: Option<f64>,
    pub previous_rate_max: Option<f64>,
    pub convergence_ratio: Option<f64>,
    pub convergence_rate_estimate: Option<f64>,
    pub rate_max_estimate: Option<f64>,
    pub pdest_candidate: Option<f64>,
    pub convergence_measure: f64,
    pub status: Lsode2CorrectionStatus,
    pub local_error: Vec<f64>,
    pub needs_jacobian_refresh: bool,
}

impl Lsode2CorrectionAssessment {
    pub fn converged(&self) -> bool {
        self.status == Lsode2CorrectionStatus::Converged
    }
}

#[derive(Debug, Clone)]
pub struct Lsode2CorrectionController {
    tolerance: Lsode2Tolerance,
    config: Lsode2CorrectionControlConfig,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lsode2DstodaCorrectorContext {
    pub method_is_adams: bool,
    pub previous_rate_max: Option<f64>,
    pub h_el1_abs: Option<f64>,
    /// Optional DSTODA-style roundoff threshold `100 * pnorm * uround`.
    ///
    /// When absent, a conservative fallback threshold is used.
    pub roundoff_tolerance: Option<f64>,
}

impl Lsode2CorrectionController {
    pub fn new(
        tolerance: Lsode2Tolerance,
        config: Lsode2CorrectionControlConfig,
    ) -> Result<Self, Lsode2CorrectionError> {
        config.validate()?;
        Ok(Self { tolerance, config })
    }

    pub fn scalar(
        rtol: f64,
        atol: f64,
        config: Lsode2CorrectionControlConfig,
    ) -> Result<Self, Lsode2CorrectionError> {
        Self::new(Lsode2Tolerance::scalar(rtol, atol), config)
    }

    pub fn config(&self) -> Lsode2CorrectionControlConfig {
        self.config
    }

    /// Compute DSTODA-style roundoff threshold `100 * pnorm * uround`, where
    /// `pnorm` is the weighted RMS norm of the predicted solution vector.
    pub fn dstoda_roundoff_tolerance(
        &self,
        y_scale_source: &[f64],
    ) -> Result<f64, Lsode2CorrectionError> {
        let weights = error_weights(y_scale_source, &self.tolerance)?;
        let pnorm = weighted_rms_norm(y_scale_source, &weights)?;
        Ok(100.0 * pnorm * f64::EPSILON)
    }

    pub fn assess_iteration(
        &self,
        order: usize,
        y_scale_source: &[f64],
        correction: &[f64],
        accumulated_correction: &[f64],
        previous_weighted_norm: Option<f64>,
        previous_rate_estimate: Option<f64>,
        iteration: usize,
    ) -> Result<Lsode2CorrectionAssessment, Lsode2CorrectionError> {
        self.assess_iteration_with_tesco2(
            order,
            y_scale_source,
            correction,
            accumulated_correction,
            previous_weighted_norm,
            previous_rate_estimate,
            iteration,
            None,
        )
    }

    pub fn assess_iteration_with_dstoda_context(
        &self,
        order: usize,
        y_scale_source: &[f64],
        correction: &[f64],
        accumulated_correction: &[f64],
        previous_weighted_norm: Option<f64>,
        previous_rate_estimate: Option<f64>,
        iteration: usize,
        tesco2_override: Option<f64>,
        dstoda: Option<Lsode2DstodaCorrectorContext>,
    ) -> Result<Lsode2CorrectionAssessment, Lsode2CorrectionError> {
        self.assess_iteration_inner(
            order,
            y_scale_source,
            correction,
            accumulated_correction,
            previous_weighted_norm,
            previous_rate_estimate,
            iteration,
            tesco2_override,
            dstoda,
        )
    }

    pub fn assess_iteration_with_tesco2(
        &self,
        order: usize,
        y_scale_source: &[f64],
        correction: &[f64],
        accumulated_correction: &[f64],
        previous_weighted_norm: Option<f64>,
        previous_rate_estimate: Option<f64>,
        iteration: usize,
        tesco2_override: Option<f64>,
    ) -> Result<Lsode2CorrectionAssessment, Lsode2CorrectionError> {
        self.assess_iteration_inner(
            order,
            y_scale_source,
            correction,
            accumulated_correction,
            previous_weighted_norm,
            previous_rate_estimate,
            iteration,
            tesco2_override,
            None,
        )
    }

    fn assess_iteration_inner(
        &self,
        order: usize,
        y_scale_source: &[f64],
        correction: &[f64],
        accumulated_correction: &[f64],
        previous_weighted_norm: Option<f64>,
        previous_rate_estimate: Option<f64>,
        iteration: usize,
        tesco2_override: Option<f64>,
        dstoda: Option<Lsode2DstodaCorrectorContext>,
    ) -> Result<Lsode2CorrectionAssessment, Lsode2CorrectionError> {
        let tesco2 = match tesco2_override {
            Some(value) => value,
            None => bdf_tesco2(order)?,
        };
        let conit = 0.5 / (order as f64 + 2.0);
        let weights = error_weights(y_scale_source, &self.tolerance)?;
        let weighted_norm = weighted_rms_norm(correction, &weights)?;
        let accumulated_weighted_norm = weighted_rms_norm(accumulated_correction, &weights)?;
        let convergence_ratio = previous_weighted_norm.and_then(|previous| {
            if previous.is_finite() && previous > 0.0 {
                Some(weighted_norm / previous)
            } else {
                None
            }
        });
        let previous_rate_max = dstoda.and_then(|ctx| ctx.previous_rate_max);
        let rate_max_estimate = if iteration >= 2 {
            let rm = convergence_ratio.map(|ratio| {
                if ratio.is_finite() && ratio > 0.0 {
                    ratio.min(1024.0)
                } else {
                    1024.0
                }
            });
            match (previous_rate_max, rm) {
                (Some(previous), Some(cur)) if previous.is_finite() => Some(previous.max(cur)),
                (None, Some(cur)) => Some(cur),
                (Some(previous), None) if previous.is_finite() => Some(previous),
                _ => None,
            }
        } else {
            previous_rate_max
        };
        let crate_estimate = convergence_ratio
            .map(|ratio| {
                previous_rate_estimate
                    .map(|previous| previous * self.config.rate_memory_weight)
                    .unwrap_or(0.0)
                    .max(ratio)
            })
            .unwrap_or(self.config.initial_convergence_rate);
        let convergence_measure =
            weighted_norm * (1.0_f64).min(1.5 * crate_estimate) / (tesco2 * conit);
        let mut status = if convergence_measure <= 1.0 {
            Lsode2CorrectionStatus::Converged
        } else if iteration >= 2
            && convergence_ratio
                .map(|ratio| ratio.is_finite() && ratio > 2.0)
                .unwrap_or(false)
        {
            // DSTODA parity: once we have at least two correction passes
            // (`M >= 2`), a growth beyond `DEL > 2*DELP` is treated as a
            // failed corrector iteration path.
            Lsode2CorrectionStatus::Diverged
        } else if iteration >= self.config.max_iterations {
            Lsode2CorrectionStatus::IterationLimitReached
        } else {
            Lsode2CorrectionStatus::Continue
        };
        // DSTODA parity:
        // For Adams (`METH = 1`), force at least two corrector passes to form
        // a local Lipschitz estimate, unless the change is already at roundoff level.
        let roundoff_tolerance = dstoda.and_then(|ctx| ctx.roundoff_tolerance);
        if status == Lsode2CorrectionStatus::Converged
            && iteration == 1
            && dstoda.map(|ctx| ctx.method_is_adams).unwrap_or(false)
            && !is_roundoff_level_correction(weighted_norm, roundoff_tolerance)
        {
            status = Lsode2CorrectionStatus::Continue;
        }
        let needs_jacobian_refresh = matches!(
            status,
            Lsode2CorrectionStatus::Diverged | Lsode2CorrectionStatus::IterationLimitReached
        );
        let local_error = accumulated_correction
            .iter()
            .map(|value| *value / tesco2)
            .collect();
        let pdest_candidate = if status == Lsode2CorrectionStatus::Converged {
            if let Some(ctx) = dstoda {
                if ctx.method_is_adams {
                    if let (Some(rate), Some(h_el1_abs)) = (rate_max_estimate, ctx.h_el1_abs) {
                        if rate.is_finite()
                            && rate > 0.0
                            && h_el1_abs.is_finite()
                            && h_el1_abs > 0.0
                        {
                            Some(rate / h_el1_abs)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        Ok(Lsode2CorrectionAssessment {
            order,
            iteration,
            weighted_norm,
            accumulated_weighted_norm,
            previous_weighted_norm,
            previous_rate_max,
            convergence_ratio,
            convergence_rate_estimate: Some(crate_estimate),
            rate_max_estimate,
            pdest_candidate,
            convergence_measure,
            status,
            local_error,
            needs_jacobian_refresh,
        })
    }
}

fn is_roundoff_level_correction(weighted_norm: f64, threshold: Option<f64>) -> bool {
    let bound = threshold
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(100.0 * f64::EPSILON);
    weighted_norm.is_finite() && weighted_norm <= bound
}

impl Default for Lsode2CorrectionController {
    fn default() -> Self {
        Self::scalar(1.0e-6, 1.0e-8, Lsode2CorrectionControlConfig::default())
            .expect("default LSODE2 correction config should be valid")
    }
}

fn bdf_tesco2(order: usize) -> Result<f64, Lsode2CorrectionError> {
    Lsode2BdfDcfodeTables::default()
        .tesco2(order)
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correction_controller_marks_small_correction_as_converged() {
        let controller =
            Lsode2CorrectionController::scalar(1.0e-3, 1.0e-6, Default::default()).unwrap();

        let assessment = controller
            .assess_iteration(
                1,
                &[1.0, 2.0],
                &[1.0e-5, 2.0e-5],
                &[1.0e-5, 2.0e-5],
                None,
                None,
                1,
            )
            .unwrap();

        assert_eq!(assessment.status, Lsode2CorrectionStatus::Converged);
        assert!(assessment.converged());
        assert!(!assessment.needs_jacobian_refresh);
        assert_eq!(assessment.order, 1);
        assert!(assessment.accumulated_weighted_norm > 0.0);
        assert_eq!(assessment.convergence_rate_estimate, Some(0.7));
    }

    #[test]
    fn correction_controller_requests_continue_for_nonconverged_iteration() {
        let controller =
            Lsode2CorrectionController::scalar(1.0e-3, 1.0e-6, Default::default()).unwrap();

        let assessment = controller
            .assess_iteration(2, &[1.0], &[1.0e-3], &[1.5e-3], Some(2.0), Some(0.1), 1)
            .unwrap();

        assert_eq!(assessment.status, Lsode2CorrectionStatus::Continue);
        assert_eq!(
            assessment.convergence_ratio,
            Some(assessment.weighted_norm / 2.0)
        );
        assert!(assessment.convergence_rate_estimate.is_some());
        assert!(assessment.convergence_measure.is_finite());
    }

    #[test]
    fn correction_controller_detects_divergence_and_iteration_limit() {
        let controller =
            Lsode2CorrectionController::scalar(1.0e-3, 1.0e-6, Default::default()).unwrap();

        let diverged = controller
            .assess_iteration(2, &[1.0], &[1.0e-2], &[1.0e-2], Some(1.0e-4), Some(0.5), 2)
            .unwrap();
        assert_eq!(diverged.status, Lsode2CorrectionStatus::Diverged);
        assert!(diverged.needs_jacobian_refresh);

        let limited = controller
            .assess_iteration(2, &[1.0], &[1.0e-3], &[1.0e-3], Some(1.0), Some(0.5), 4)
            .unwrap();
        assert_eq!(
            limited.status,
            Lsode2CorrectionStatus::IterationLimitReached
        );
        assert!(limited.needs_jacobian_refresh);

        let non_diverged_by_large_crate_only = controller
            .assess_iteration(2, &[1.0], &[1.0e-3], &[1.0e-3], Some(0.5), Some(10.0), 1)
            .unwrap();
        assert_eq!(
            non_diverged_by_large_crate_only.status,
            Lsode2CorrectionStatus::Continue,
            "DSTODA parity: convergence-rate magnitude alone should not trigger divergence"
        );
    }

    #[test]
    fn correction_controller_uses_odepack_style_acor_scaling_and_rejects_bad_config() {
        let controller =
            Lsode2CorrectionController::scalar(1.0e-3, 1.0e-6, Default::default()).unwrap();
        let assessment = controller
            .assess_iteration(3, &[1.0], &[2.0e-4], &[6.0e-4], Some(0.3), Some(0.25), 1)
            .unwrap();
        assert!(
            (assessment.local_error[0] - (6.0e-4 / (22.0 / 3.0))).abs() < 1.0e-12,
            "LSODE-style local error should be accumulated correction divided by TESCO(2,q)"
        );

        let bad = Lsode2CorrectionControlConfig {
            initial_convergence_rate: 0.0,
            ..Default::default()
        };
        assert!(Lsode2CorrectionController::scalar(1.0e-3, 1.0e-6, bad).is_err());

        let bad = Lsode2CorrectionControlConfig {
            max_iterations: 0,
            ..Default::default()
        };
        assert!(Lsode2CorrectionController::scalar(1.0e-3, 1.0e-6, bad).is_err());
    }

    #[test]
    fn correction_controller_exposes_bdf_tesco2_constants() {
        assert_eq!(bdf_tesco2(1).unwrap(), 2.0);
        assert_eq!(bdf_tesco2(2).unwrap(), 4.5);
        assert!((bdf_tesco2(3).unwrap() - 22.0 / 3.0).abs() < 1e-12);
        assert!((bdf_tesco2(4).unwrap() - 125.0 / 12.0).abs() < 1e-12);
        assert!((bdf_tesco2(5).unwrap() - 13.7).abs() < 1e-12);
        assert!(bdf_tesco2(6).is_err());
    }

    #[test]
    fn correction_controller_adams_forces_second_pass_except_roundoff() {
        let controller =
            Lsode2CorrectionController::scalar(1.0e-3, 1.0e-6, Default::default()).unwrap();
        let forced = controller
            .assess_iteration_with_dstoda_context(
                1,
                &[1.0],
                &[1.0e-5],
                &[1.0e-5],
                None,
                None,
                1,
                None,
                Some(Lsode2DstodaCorrectorContext {
                    method_is_adams: true,
                    previous_rate_max: None,
                    h_el1_abs: None,
                    roundoff_tolerance: None,
                }),
            )
            .unwrap();
        assert_eq!(forced.status, Lsode2CorrectionStatus::Continue);

        let roundoff = controller
            .assess_iteration_with_dstoda_context(
                1,
                &[1.0],
                &[1.0e-30],
                &[1.0e-30],
                None,
                None,
                1,
                None,
                Some(Lsode2DstodaCorrectorContext {
                    method_is_adams: true,
                    previous_rate_max: None,
                    h_el1_abs: None,
                    roundoff_tolerance: None,
                }),
            )
            .unwrap();
        assert_eq!(roundoff.status, Lsode2CorrectionStatus::Converged);
    }

    #[test]
    fn correction_controller_adams_high_order_requires_method_specific_tesco2_override() {
        let controller =
            Lsode2CorrectionController::scalar(1.0e-3, 1.0e-6, Default::default()).unwrap();
        let adams_tesco2 =
            crate::numerical::LSODE2::adams_engine::Lsode2AdamsDcfodeTables::default()
                .order(6)
                .expect("Adams DCFODE should define coefficients for q=6")
                .tesco2;

        let without_override = controller.assess_iteration_with_dstoda_context(
            6,
            &[1.0],
            &[1.0e-6],
            &[1.0e-6],
            None,
            None,
            1,
            None,
            Some(Lsode2DstodaCorrectorContext {
                method_is_adams: true,
                previous_rate_max: None,
                h_el1_abs: None,
                roundoff_tolerance: None,
            }),
        );
        assert!(
            matches!(without_override, Err(Lsode2CorrectionError::Dcfode(_))),
            "without override we fallback to BDF TESCO(2,q), which is undefined for q=6"
        );

        let with_override = controller.assess_iteration_with_dstoda_context(
            6,
            &[1.0],
            &[1.0e-6],
            &[1.0e-6],
            None,
            None,
            1,
            Some(adams_tesco2),
            Some(Lsode2DstodaCorrectorContext {
                method_is_adams: true,
                previous_rate_max: None,
                h_el1_abs: None,
                roundoff_tolerance: None,
            }),
        );
        match with_override {
            Ok(assessment) => {
                assert_eq!(assessment.order, 6);
                assert_eq!(assessment.status, Lsode2CorrectionStatus::Continue);
            }
            Err(err) => {
                panic!("method-specific Adams TESCO(2,6) should keep assessment valid: {err}")
            }
        }
    }
}
