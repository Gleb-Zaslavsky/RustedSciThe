//! Committed solver state and transactional candidate-step state.

use super::config::RadauConfig;
use super::controller::initial_step;
use super::error::{RadauError, RadauStage};

/// Mutable adaptive state. The trial output remains separate until a step is
/// accepted, so rejected Newton/error-control attempts cannot corrupt `y`.
#[derive(Debug)]
pub(crate) struct RadauSolverState {
    pub(crate) t: f64,
    pub(crate) y: Vec<f64>,
    pub(crate) h_abs: f64,
    pub(crate) direction: f64,
    pub(crate) attempts: usize,
    pub(crate) accepted_steps: usize,
    pub(crate) rejected_steps: usize,
    /// Previous accepted step and error used by the two-step controller.
    pub(crate) h_abs_old: Option<f64>,
    pub(crate) error_norm_old: Option<f64>,
}

impl RadauSolverState {
    /// Validate the initial state and allocate the two reusable state buffers.
    pub(crate) fn new(config: &RadauConfig, y0: &[f64]) -> Result<Self, RadauError> {
        if y0.is_empty() {
            return Err(RadauError::ShapeMismatch {
                stage: RadauStage::Preparation,
                expected: 1,
                actual: 0,
            });
        }
        if y0.iter().any(|value| !value.is_finite()) {
            return Err(RadauError::NonFiniteCallback {
                stage: RadauStage::Preparation,
            });
        }
        Ok(Self {
            t: config.t0,
            y: y0.to_vec(),
            h_abs: initial_step(config),
            direction: if config.t_bound > config.t0 {
                1.0
            } else {
                -1.0
            },
            attempts: 0,
            accepted_steps: 0,
            rejected_steps: 0,
            h_abs_old: None,
            error_norm_old: None,
        })
    }

    pub(crate) fn commit(&mut self, t: f64, trial: &[f64], h_abs: f64, error_norm: f64) {
        self.h_abs_old = Some(self.h_abs);
        self.error_norm_old = Some(error_norm);
        self.t = t;
        self.y.copy_from_slice(trial);
        self.accepted_steps += 1;
        self.h_abs = h_abs;
    }
}
