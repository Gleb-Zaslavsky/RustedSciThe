//! BDF-like order-selection helpers for future LSODE2 native stepping.
//!
//! LSODE/variable-order BDF logic does not decide "raise order or not" from a
//! single boolean.  After an accepted step it compares order candidates around
//! the current order, typically `q-1`, `q`, and `q+1`, and chooses the one
//! with the best admissible growth potential.
//!
//! LSODE2 does not yet compute all three candidate error estimates inside the
//! native step engine, but this module gives that logic an explicit, tested
//! home now so we can wire real candidate norms into it incrementally.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lsode2OrderSelectionConfig {
    pub min_change_factor: f64,
    pub max_growth: f64,
}

impl Default for Lsode2OrderSelectionConfig {
    fn default() -> Self {
        Self {
            min_change_factor: 1.1,
            max_growth: 5.0,
        }
    }
}

impl Lsode2OrderSelectionConfig {
    pub fn validate(self) -> Result<(), Lsode2OrderSelectionError> {
        if !self.min_change_factor.is_finite() || self.min_change_factor < 1.0 {
            return Err(Lsode2OrderSelectionError::InvalidConfig(
                "min_change_factor must be finite and >= 1",
            ));
        }
        if !self.max_growth.is_finite() || self.max_growth < 1.0 {
            return Err(Lsode2OrderSelectionError::InvalidConfig(
                "max_growth must be finite and >= 1",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Lsode2OrderSelectionError {
    InvalidConfig(&'static str),
    InvalidOrder {
        order_current: usize,
        order_cap: usize,
    },
}

impl std::fmt::Display for Lsode2OrderSelectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(message) => {
                write!(f, "invalid LSODE2 order-selection config: {message}")
            }
            Self::InvalidOrder {
                order_current,
                order_cap,
            } => write!(
                f,
                "LSODE2 order selection requires order_current <= order_cap, got {order_current} > {order_cap}"
            ),
        }
    }
}

impl std::error::Error for Lsode2OrderSelectionError {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lsode2OrderSelectionCandidates {
    pub lower_error_norm: f64,
    pub current_error_norm: f64,
    pub higher_error_norm: f64,
}

impl Lsode2OrderSelectionCandidates {
    pub fn current_only(current_error_norm: f64) -> Self {
        Self {
            lower_error_norm: f64::INFINITY,
            current_error_norm,
            higher_error_norm: f64::INFINITY,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lsode2OrderCandidate {
    Lower,
    Current,
    Higher,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lsode2OrderSelectionDecision {
    pub candidate: Lsode2OrderCandidate,
    pub order_new: usize,
    pub suggested_growth: f64,
    pub lower_factor: f64,
    pub current_factor: f64,
    pub higher_factor: f64,
}

pub fn select_bdf_like_order(
    order_current: usize,
    order_cap: usize,
    candidates: Lsode2OrderSelectionCandidates,
    config: Lsode2OrderSelectionConfig,
) -> Result<Lsode2OrderSelectionDecision, Lsode2OrderSelectionError> {
    config.validate()?;
    if order_current == 0 || order_current > order_cap {
        return Err(Lsode2OrderSelectionError::InvalidOrder {
            order_current,
            order_cap,
        });
    }

    let lower_factor = if order_current > 1 {
        lower_rh(candidates.lower_error_norm, order_current)
    } else {
        0.0
    };
    let current_factor = current_rh(candidates.current_error_norm, order_current);
    let higher_factor = if order_current < order_cap {
        higher_rh(candidates.higher_error_norm, order_current)
    } else {
        0.0
    };

    let (candidate, best_factor) = if current_factor >= higher_factor {
        if current_factor < lower_factor {
            (Lsode2OrderCandidate::Lower, lower_factor)
        } else {
            (Lsode2OrderCandidate::Current, current_factor)
        }
    } else if higher_factor > lower_factor {
        (Lsode2OrderCandidate::Higher, higher_factor)
    } else {
        (Lsode2OrderCandidate::Lower, lower_factor)
    };

    if best_factor < config.min_change_factor {
        return Ok(Lsode2OrderSelectionDecision {
            candidate: Lsode2OrderCandidate::Current,
            order_new: order_current,
            suggested_growth: 1.0,
            lower_factor,
            current_factor,
            higher_factor,
        });
    }

    let order_new = match candidate {
        Lsode2OrderCandidate::Lower => order_current.saturating_sub(1).max(1),
        Lsode2OrderCandidate::Current => order_current,
        Lsode2OrderCandidate::Higher => (order_current + 1).min(order_cap),
    };
    let suggested_growth = best_factor.clamp(1.0, config.max_growth);

    Ok(Lsode2OrderSelectionDecision {
        candidate,
        order_new,
        suggested_growth,
        lower_factor,
        current_factor,
        higher_factor,
    })
}

fn lower_rh(error_norm: f64, order_current: usize) -> f64 {
    odepack_rh(error_norm, order_current as f64, 1.3, 1.3e-6)
}

fn current_rh(error_norm: f64, order_current: usize) -> f64 {
    odepack_rh(error_norm, (order_current + 1) as f64, 1.2, 1.2e-6)
}

fn higher_rh(error_norm: f64, order_current: usize) -> f64 {
    odepack_rh(error_norm, (order_current + 2) as f64, 1.4, 1.4e-6)
}

fn odepack_rh(error_norm: f64, exponent_denominator: f64, scale: f64, epsilon: f64) -> f64 {
    if !error_norm.is_finite() || error_norm <= 0.0 {
        return 0.0;
    }
    1.0 / (scale * error_norm.powf(1.0 / exponent_denominator) + epsilon)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_selection_prefers_higher_candidate_when_it_has_best_growth() {
        let decision = select_bdf_like_order(
            2,
            4,
            Lsode2OrderSelectionCandidates {
                lower_error_norm: 0.9,
                current_error_norm: 0.4,
                higher_error_norm: 0.05,
            },
            Lsode2OrderSelectionConfig::default(),
        )
        .unwrap();

        assert_eq!(decision.candidate, Lsode2OrderCandidate::Higher);
        assert_eq!(decision.order_new, 3);
        assert!(decision.higher_factor > decision.current_factor);
        assert!(decision.suggested_growth >= 1.0);
    }

    #[test]
    fn order_selection_prefers_lower_candidate_when_current_order_is_too_ambitious() {
        let decision = select_bdf_like_order(
            3,
            5,
            Lsode2OrderSelectionCandidates {
                lower_error_norm: 0.05,
                current_error_norm: 0.8,
                higher_error_norm: 0.9,
            },
            Lsode2OrderSelectionConfig::default(),
        )
        .unwrap();

        assert_eq!(decision.candidate, Lsode2OrderCandidate::Lower);
        assert_eq!(decision.order_new, 2);
        assert!(decision.lower_factor > decision.current_factor);
    }

    #[test]
    fn order_selection_respects_order_cap_for_higher_candidate() {
        let decision = select_bdf_like_order(
            2,
            2,
            Lsode2OrderSelectionCandidates {
                lower_error_norm: 0.9,
                current_error_norm: 0.5,
                higher_error_norm: 0.01,
            },
            Lsode2OrderSelectionConfig::default(),
        )
        .unwrap();

        assert_ne!(decision.candidate, Lsode2OrderCandidate::Higher);
        assert_eq!(decision.order_new, 2);
        assert_eq!(decision.higher_factor, 0.0);
    }

    #[test]
    fn order_selection_supports_current_only_seed_path() {
        let decision = select_bdf_like_order(
            1,
            3,
            Lsode2OrderSelectionCandidates::current_only(0.2),
            Lsode2OrderSelectionConfig::default(),
        )
        .unwrap();

        assert_eq!(decision.candidate, Lsode2OrderCandidate::Current);
        assert_eq!(decision.order_new, 1);
        assert_eq!(decision.lower_factor, 0.0);
        assert_eq!(decision.higher_factor, 0.0);
    }

    #[test]
    fn order_selection_suppresses_change_when_best_rh_is_too_small() {
        let decision = select_bdf_like_order(
            2,
            4,
            Lsode2OrderSelectionCandidates {
                lower_error_norm: 2.0,
                current_error_norm: 2.0,
                higher_error_norm: 2.0,
            },
            Lsode2OrderSelectionConfig::default(),
        )
        .unwrap();

        assert_eq!(decision.candidate, Lsode2OrderCandidate::Current);
        assert_eq!(decision.order_new, 2);
        assert_eq!(decision.suggested_growth, 1.0);
    }

    #[test]
    fn order_selection_rejects_invalid_config_and_order() {
        let err = select_bdf_like_order(
            2,
            1,
            Lsode2OrderSelectionCandidates::current_only(0.2),
            Lsode2OrderSelectionConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            Lsode2OrderSelectionError::InvalidOrder {
                order_current: 2,
                order_cap: 1
            }
        ));

        let err = select_bdf_like_order(
            1,
            1,
            Lsode2OrderSelectionCandidates::current_only(0.2),
            Lsode2OrderSelectionConfig {
                min_change_factor: 0.0,
                ..Lsode2OrderSelectionConfig::default()
            },
        )
        .unwrap_err();
        assert!(matches!(err, Lsode2OrderSelectionError::InvalidConfig(_)));
    }
}
