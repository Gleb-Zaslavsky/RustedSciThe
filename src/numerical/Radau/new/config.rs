//! Public and internal configuration contracts for the new Radau solver.
//!
//! The policy enums are intentionally compact and copyable. Strings stay at
//! compatibility boundaries only; the numerical engine receives validated
//! values and can branch without parsing or allocating.

use super::error::{RadauConfigError, RadauError};
use super::output::RadauOutputPolicy;
use super::telemetry::RadauTelemetryMode;
use crate::symbolic::ivp_telemetry::IvpLambdifyExecutionPolicy;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Selects the runtime family used by the prepared model.
pub(crate) enum RadauExecution {
    Native,
    Lambdify,
    Aot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Selects the symbolic assembly representation for a Lambdify route.
pub(crate) enum RadauAssembly {
    ExprLegacy,
    AtomViewNative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Describes the storage format expected by callback and linear workspaces.
pub(crate) enum RadauMatrixLayout {
    Dense,
    Sparse,
    Banded { lower: usize, upper: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Identifies how Jacobian values are supplied to the numerical core.
pub(crate) enum RadauJacobianSource {
    Analytic,
    Constant,
    FiniteDifference,
}

#[derive(Debug, Clone, PartialEq)]
/// Validated policy inputs shared by preparation, continuation, and stepping.
pub(crate) struct RadauConfig {
    /// Initial integration time.
    pub t0: f64,
    /// Final integration time; the sign of the interval determines direction.
    pub t_bound: f64,
    /// Relative error tolerance.
    pub rtol: f64,
    /// Absolute error tolerance.
    pub atol: f64,
    /// Optional initial step size.
    pub first_step: Option<f64>,
    /// Upper bound for an individual step.
    pub max_step: f64,
    /// Maximum accepted/rejected step attempts for the solve.
    pub max_steps: usize,
    /// Newton iteration budget for one nonlinear step.
    pub max_newton_iterations: usize,
    /// Retry budget after a rejected step.
    pub max_retries: usize,
    /// Runtime family selected during preparation.
    pub execution: RadauExecution,
    /// Symbolic frontend, required for symbolic execution.
    pub assembly: Option<RadauAssembly>,
    /// Matrix storage selected once for the prepared model.
    pub matrix_layout: RadauMatrixLayout,
    /// Jacobian source policy.
    pub jacobian_source: RadauJacobianSource,
    /// Optional diagnostic telemetry mode; defaults to no instrumentation.
    pub telemetry: RadauTelemetryMode,
    pub execution_policy: IvpLambdifyExecutionPolicy,
    /// Retained trajectory policy; final-only is allocation-free by default.
    pub output: RadauOutputPolicy,
}

impl Default for RadauConfig {
    fn default() -> Self {
        Self {
            t0: 0.0,
            t_bound: 1.0,
            rtol: 1e-3,
            atol: 1e-6,
            first_step: None,
            max_step: f64::INFINITY,
            max_steps: 1_000_000,
            max_newton_iterations: 6,
            max_retries: 12,
            execution: RadauExecution::Native,
            assembly: None,
            matrix_layout: RadauMatrixLayout::Dense,
            jacobian_source: RadauJacobianSource::Analytic,
            telemetry: RadauTelemetryMode::Off,
            execution_policy: IvpLambdifyExecutionPolicy::Sequential,
            output: RadauOutputPolicy::FinalOnly,
        }
    }
}

impl RadauConfig {
    /// Validate configuration before allocating solver state or preparing callbacks.
    pub(crate) fn validate(&self) -> Result<(), RadauError> {
        if !self.t0.is_finite() || !self.t_bound.is_finite() || self.t0 == self.t_bound {
            return Err(RadauConfigError::InvalidTimeBounds.into());
        }
        if !self.rtol.is_finite() || self.rtol <= 0.0 {
            return Err(RadauConfigError::InvalidRelativeTolerance.into());
        }
        if !self.atol.is_finite() || self.atol <= 0.0 {
            return Err(RadauConfigError::InvalidAbsoluteTolerance.into());
        }
        if !self.max_step.is_finite() && self.max_step != f64::INFINITY {
            return Err(RadauConfigError::InvalidMaximumStep.into());
        }
        if self.max_step <= 0.0 {
            return Err(RadauConfigError::InvalidMaximumStep.into());
        }
        if let Some(first_step) = self.first_step {
            if !first_step.is_finite() || first_step <= 0.0 {
                return Err(RadauConfigError::InvalidFirstStep.into());
            }
        }
        if self.max_steps == 0 || self.max_newton_iterations == 0 || self.max_retries == 0 {
            return Err(RadauConfigError::EmptyIterationBudget.into());
        }
        if let RadauMatrixLayout::Banded { lower, upper } = self.matrix_layout {
            if lower == 0 && upper == 0 {
                return Err(RadauConfigError::EmptyBandwidth.into());
            }
        }
        if self.execution == RadauExecution::Native && self.assembly.is_some() {
            return Err(RadauConfigError::NativeWithSymbolicAssembly.into());
        }
        if self.execution != RadauExecution::Native && self.assembly.is_none() {
            return Err(RadauConfigError::SymbolicWithoutAssembly.into());
        }
        Ok(())
    }
}
