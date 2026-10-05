//! Parameter rebind and prepared-model continuation contracts.
//!
//! Rebinding must preserve immutable preparation while isolating mutable
//! values, workspaces, Jacobians, factors, and telemetry between sessions.

use super::callbacks::{DenseJacobianCallback, ResidualCallback};
use super::config::RadauConfig;
use super::error::{RadauError, RadauStage};
use super::prepared::RadauPreparedModel;
use super::session::RadauSession;
use super::step::RadauStepResult;

/// A reusable numerical continuation owner.
///
/// Preparation identity and workspace survive value-only parameter rebinds.
/// Restart changes the numerical interval and initial state, but does not
/// rebuild symbolic plans or backend artifacts.
#[derive(Debug)]
pub(crate) struct RadauContinuation {
    prepared: RadauPreparedModel,
    session: RadauSession,
    initial_state: Vec<f64>,
    t0: f64,
    t_bound: f64,
}

impl RadauContinuation {
    /// Prepare one reusable model/session pair for continuation runs.
    pub(crate) fn prepare(
        dimension: usize,
        parameter_count: usize,
        initial_state: &[f64],
        t0: f64,
        t_bound: f64,
        config: &RadauConfig,
    ) -> Result<Self, RadauError> {
        validate_restart_state(dimension, initial_state, t0, t_bound)?;
        let prepared =
            RadauPreparedModel::prepare_with_parameter_count(dimension, parameter_count, config)?;
        let session = RadauSession::new(prepared)?;
        Ok(Self {
            prepared,
            session,
            initial_state: initial_state.to_vec(),
            t0,
            t_bound,
        })
    }

    pub(crate) const fn prepared(&self) -> RadauPreparedModel {
        self.prepared
    }

    /// Borrow the mutable session that owns numeric workspaces and telemetry.
    pub(crate) fn session(&mut self) -> &mut RadauSession {
        &mut self.session
    }

    /// Borrow the retained initial state used by the next restart.
    pub(crate) fn initial_state(&self) -> &[f64] {
        &self.initial_state
    }

    /// Return the retained initial-state capacity for allocation diagnostics.
    pub(crate) fn initial_state_capacity(&self) -> usize {
        self.initial_state.capacity()
    }

    /// Return the current continuation interval.
    pub(crate) fn interval(&self) -> (f64, f64) {
        (self.t0, self.t_bound)
    }

    /// Rebind values while preserving prepared symbolic artifacts and buffers.
    pub(crate) fn rebind_parameters(&mut self, values: &[f64]) -> Result<(), RadauError> {
        self.session.rebind_parameters(values)
    }

    /// Restart the numerical problem without rebuilding prepared artifacts.
    /// Validation is completed before any session or state is mutated.
    pub(crate) fn restart(
        &mut self,
        initial_state: &[f64],
        t0: f64,
        t_bound: f64,
    ) -> Result<(), RadauError> {
        validate_restart_state(self.prepared.key().dimension, initial_state, t0, t_bound)?;
        self.initial_state.copy_from_slice(initial_state);
        self.t0 = t0;
        self.t_bound = t_bound;
        self.session.restart_numeric_state();
        Ok(())
    }

    /// Advance one dense step through the retained session workspace.
    pub(crate) fn try_dense_step<R, J>(
        &mut self,
        config: &RadauConfig,
        residual: &mut R,
        jacobian: &mut J,
        t: f64,
        h: f64,
        y: &[f64],
        output: &mut [f64],
    ) -> Result<RadauStepResult, RadauError>
    where
        R: ResidualCallback,
        J: DenseJacobianCallback,
    {
        self.session
            .try_dense_step(config, residual, jacobian, t, h, y, output)
    }
}

fn validate_restart_state(
    dimension: usize,
    initial_state: &[f64],
    t0: f64,
    t_bound: f64,
) -> Result<(), RadauError> {
    if initial_state.len() != dimension {
        return Err(RadauError::ShapeMismatch {
            stage: RadauStage::Preparation,
            expected: dimension,
            actual: initial_state.len(),
        });
    }
    if !t0.is_finite() || !t_bound.is_finite() || t0 == t_bound {
        return Err(super::error::RadauConfigError::InvalidTimeBounds.into());
    }
    if initial_state.iter().any(|value| !value.is_finite()) {
        return Err(RadauError::NonFiniteCallback {
            stage: RadauStage::Preparation,
        });
    }
    Ok(())
}
