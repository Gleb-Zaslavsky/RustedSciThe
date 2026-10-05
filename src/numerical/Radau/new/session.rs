//! Mutable solve session layered over immutable preparation.
//!
//! A session owns parameter values, workspace, and numeric generations. The
//! prepared model is copied as a small key-bearing value; symbolic plans and
//! AOT artifacts will be referenced by it rather than rebuilt on rebind.

use super::callbacks::{DenseJacobianCallback, ResidualCallback};
use super::config::RadauMatrixLayout;
use super::error::{RadauError, RadauStage, RadauUnsupportedRoute};
use super::linear::PreparedLinearBackend;
use super::prepared::RadauPreparedModel;
use super::step::{RadauStepResult, try_radau5_step_with_backend};
use super::workspace::RadauWorkspace;

#[derive(Debug)]
/// Mutable numeric state layered over an immutable prepared model.
pub(crate) struct RadauSession {
    prepared: RadauPreparedModel,
    parameter_values: Vec<f64>,
    workspace: RadauWorkspace,
    linear_backend: PreparedLinearBackend,
    numeric_generation: u64,
}

impl RadauSession {
    /// Allocate session-local parameter and backend workspaces.
    pub(crate) fn new(prepared: RadauPreparedModel) -> Result<Self, RadauError> {
        let dimension = prepared.key().dimension;
        let parameter_count = prepared.key().parameter_count;
        let linear_backend = PreparedLinearBackend::from_layout(prepared.key().matrix_layout);
        let mut workspace = RadauWorkspace::default();
        workspace.resize_buffers(dimension, prepared.key().matrix_layout)?;
        workspace.linear = linear_backend.create_workspace(dimension)?;
        Ok(Self {
            prepared,
            parameter_values: vec![0.0; parameter_count],
            workspace,
            linear_backend,
            numeric_generation: 1,
        })
    }

    /// Return the immutable prepared model identity.
    pub(crate) const fn prepared(&self) -> RadauPreparedModel {
        self.prepared
    }

    /// Borrow the reusable numeric workspace.
    pub(crate) fn workspace(&mut self) -> &mut RadauWorkspace {
        &mut self.workspace
    }

    /// Borrow the current parameter values.
    pub(crate) fn parameter_values(&self) -> &[f64] {
        &self.parameter_values
    }

    /// Return the numeric generation changed by rebind/restart operations.
    pub(crate) fn numeric_generation(&self) -> u64 {
        self.numeric_generation
    }

    /// Borrow the backend selected during preparation.
    pub(crate) fn linear_backend(&self) -> &PreparedLinearBackend {
        &self.linear_backend
    }

    /// Rebind values while preserving prepared artifacts and buffer capacity.
    pub(crate) fn rebind_parameters(&mut self, values: &[f64]) -> Result<(), RadauError> {
        if values.len() != self.parameter_values.len() {
            return Err(RadauError::ShapeMismatch {
                stage: RadauStage::Preparation,
                expected: self.parameter_values.len(),
                actual: values.len(),
            });
        }
        self.parameter_values.copy_from_slice(values);
        self.numeric_generation = self.numeric_generation.saturating_add(1);
        self.workspace.invalidate_jacobian();
        self.workspace.discard_dense_output();
        Ok(())
    }

    /// Invalidate numeric history while retaining all allocated storage.
    pub(crate) fn restart_numeric_state(&mut self) {
        self.numeric_generation = self.numeric_generation.saturating_add(1);
        self.workspace.invalidate_jacobian();
        self.workspace.discard_dense_output();
    }

    /// Advance one dense Radau IIA order-5 step through the session workspace.
    pub(crate) fn try_dense_step<R, J>(
        &mut self,
        config: &super::config::RadauConfig,
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
        self.try_dense_step_with_rejection(config, residual, jacobian, t, h, y, output, false)
    }

    /// Internal controller variant carrying SciPy's previous-rejection flag.
    pub(crate) fn try_dense_step_with_rejection<R, J>(
        &mut self,
        config: &super::config::RadauConfig,
        residual: &mut R,
        jacobian: &mut J,
        t: f64,
        h: f64,
        y: &[f64],
        output: &mut [f64],
        rejected: bool,
    ) -> Result<RadauStepResult, RadauError>
    where
        R: ResidualCallback,
        J: DenseJacobianCallback,
    {
        if self.prepared.key().matrix_layout != RadauMatrixLayout::Dense {
            return Err(RadauUnsupportedRoute::DenseStepOnly.into());
        }
        if config.matrix_layout != RadauMatrixLayout::Dense {
            return Err(RadauUnsupportedRoute::DenseConfigurationRequired.into());
        }
        if y.len() != self.prepared.key().dimension {
            return Err(RadauError::ShapeMismatch {
                stage: RadauStage::Preparation,
                expected: self.prepared.key().dimension,
                actual: y.len(),
            });
        }
        try_radau5_step_with_backend(
            config,
            &self.linear_backend,
            residual,
            jacobian,
            t,
            h,
            y,
            output,
            &mut self.workspace,
            rejected,
        )
    }
}
