//! Prepared symbolic and numerical model state.
//!
//! Preparation is intentionally separated from solve/rebind/restart so
//! parameter continuation can reuse immutable model artifacts safely.

use super::config::{
    RadauAssembly, RadauConfig, RadauExecution, RadauJacobianSource, RadauMatrixLayout,
};
use super::error::{RadauConfigError, RadauError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Immutable identity of a prepared Radau problem.
pub(crate) struct RadauProblemKey {
    /// State dimension.
    pub dimension: usize,
    /// Number of value-only continuation parameters.
    pub parameter_count: usize,
    /// Prepared execution family.
    pub execution: RadauExecution,
    /// Selected symbolic assembly frontend, if any.
    pub assembly: Option<RadauAssembly>,
    /// Prepared Jacobian storage layout.
    pub matrix_layout: RadauMatrixLayout,
    /// Jacobian source policy.
    pub jacobian_source: RadauJacobianSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Lightweight prepared model handle retained by sessions and continuations.
pub(crate) struct RadauPreparedModel {
    key: RadauProblemKey,
    preparation_generation: u64,
}

impl RadauPreparedModel {
    /// Prepare a parameter-free model with validated configuration.
    pub(crate) fn prepare(dimension: usize, config: &RadauConfig) -> Result<Self, RadauError> {
        Self::prepare_with_parameter_count(dimension, 0, config)
    }

    /// Prepare a model whose value-only parameter schema is fixed by count.
    pub(crate) fn prepare_with_parameter_count(
        dimension: usize,
        parameter_count: usize,
        config: &RadauConfig,
    ) -> Result<Self, RadauError> {
        config.validate()?;
        if dimension == 0 {
            return Err(RadauConfigError::ZeroDimension.into());
        }
        Ok(Self {
            key: RadauProblemKey {
                dimension,
                parameter_count,
                execution: config.execution,
                assembly: config.assembly,
                matrix_layout: config.matrix_layout,
                jacobian_source: config.jacobian_source,
            },
            preparation_generation: 1,
        })
    }

    /// Return the immutable problem identity used for continuation checks.
    pub(crate) const fn key(&self) -> RadauProblemKey {
        self.key
    }

    /// Return the preparation generation for provenance diagnostics.
    pub(crate) const fn preparation_generation(&self) -> u64 {
        self.preparation_generation
    }

    /// Value-only continuation keeps the immutable preparation identity.
    /// Check whether a candidate key can reuse this prepared model.
    pub(crate) fn can_rebind_values(&self, key: RadauProblemKey) -> bool {
        self.key == key
    }
}
