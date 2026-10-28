// Port based on the Gavin implementation in `PY/levenberg_marquardt.py`.

use crate::numerical::SolverTelemetryMode;
use crate::numerical::logging::{SolverLogGuard, SolverLogLevel};
use nalgebra::{DMatrix, DVector};
use std::error::Error;
use std::time::{Duration, Instant};

/// Owned source error type used by objective and Jacobian callbacks.
pub type GavinLmCallbackError = Box<dyn Error + Send + Sync + 'static>;

/// Input vector categories used by validation errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GavinLmInput {
    SampleLocations,
    Observations,
    InitialParameters,
}

/// Configurable scalar controls validated before solving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GavinLmOption {
    GradientTolerance,
    ParameterStepTolerance,
    GainRatioThreshold,
    InitialDamping,
    DampingIncreaseFactor,
    DampingDecreaseFactor,
}

/// Numerical stages that can report non-finite or invalid intermediates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GavinLmNumericalStage {
    WeightedLinearization,
    TrialObjective,
    QuadraticDampingUpdate,
    QuadraticDampingScale,
    QuadraticTrialObjective,
    GainRatio,
    DampingParameterUpdate,
    ReducedChiSquare,
    ResidualVarianceEstimate,
    FittedOutputUncertainty,
}

/// Failure categories returned by the Gavin Levenberg-Marquardt solver.
#[derive(Debug, thiserror::Error)]
pub enum GavinLmError {
    /// The model's output count does not match the supplied target vector.
    #[error("model/target length mismatch: expected {expected}, got {actual}")]
    ModelTargetLengthMismatch { expected: usize, actual: usize },
    /// The model output count overflowed for the requested number of samples.
    #[error("model output count overflow for {samples} samples")]
    OutputCountOverflow { samples: usize },
    /// The model declares no parameters.
    #[error("the objective model must declare at least one parameter")]
    NoParameters,
    /// Initial parameter vector has the wrong length.
    #[error("initial parameter length mismatch: expected {expected}, got {actual}")]
    ParameterLengthMismatch { expected: usize, actual: usize },
    /// There are fewer observations than fitted parameters.
    #[error("insufficient observations: need at least {parameters}, got {observations}")]
    InsufficientObservations {
        parameters: usize,
        observations: usize,
    },
    /// No observations were supplied.
    #[error("at least one observation is required")]
    EmptyObservations,
    /// An input vector contains a non-finite value.
    #[error("{input:?} contains a non-finite value at index {index}: {value}")]
    NonFiniteInput {
        input: GavinLmInput,
        index: usize,
        value: f64,
    },
    /// A named scalar solver option is invalid.
    #[error("invalid solver option {option:?}={value}")]
    InvalidOption { option: GavinLmOption, value: f64 },
    /// Required field was not configured on the symbolic fitting builder.
    #[error("symbolic fitting configuration is missing required field '{field}'")]
    InvalidFitConfiguration { field: &'static str },
    /// The iteration budget is zero.
    #[error("max_iterations must be greater than zero")]
    ZeroIterationBudget,
    /// Observation normalization cannot produce finite positive weights.
    #[error("observation vector must have a finite, non-zero squared norm")]
    InvalidObservationNorm,
    /// Explicit observation weights have a different length from residuals.
    #[error("weight length mismatch: expected {expected}, got {actual}")]
    WeightLengthMismatch { expected: usize, actual: usize },
    /// An explicit weight is negative or non-finite.
    #[error("invalid least-squares weight at index {index}: {value}")]
    InvalidWeight { index: usize, value: f64 },
    /// At least one residual must participate in the objective.
    #[error("at least one least-squares weight must be positive")]
    AllWeightsZero,
    /// Symbolic fitting data do not match the sample vector.
    #[error("symbolic residual/observation length mismatch: {expected}, got {actual}")]
    SymbolicDataLengthMismatch { expected: usize, actual: usize },
    /// An objective callback returned an error; its typed source is retained.
    #[error("objective callback failed: {0}")]
    ObjectiveCallback(#[source] GavinLmCallbackError),
    /// Symbolic preparation or binding failed; the original typed error is retained.
    #[error("symbolic fitting preparation failed: {0}")]
    SymbolicPreparation(#[source] GavinLmCallbackError),
    /// A symbolic equation string could not be parsed before frontend preparation.
    #[error("failed to parse symbolic equation {equation_index}: {message}")]
    SymbolicParse {
        equation_index: usize,
        message: String,
    },
    /// Objective output length does not match the input sample count.
    #[error("objective output length mismatch: expected {expected}, got {actual}")]
    ObjectiveOutputLengthMismatch { expected: usize, actual: usize },
    /// An objective callback produced non-finite values.
    #[error("objective callback produced a non-finite value at output index {index}: {value}")]
    NonFiniteObjective { index: usize, value: f64 },
    /// An exact Jacobian callback returned an error; its typed source is retained.
    #[error("analytic Jacobian callback failed: {0}")]
    AnalyticJacobianCallback(#[source] GavinLmCallbackError),
    /// Exact Jacobian dimensions do not match (observations, parameters).
    #[error(
        "analytic Jacobian shape mismatch: expected ({expected_rows}, {expected_cols}), got ({actual_rows}, {actual_cols})"
    )]
    AnalyticJacobianShapeMismatch {
        expected_rows: usize,
        expected_cols: usize,
        actual_rows: usize,
        actual_cols: usize,
    },
    /// An exact Jacobian contains a non-finite element.
    #[error("analytic Jacobian contains a non-finite value at ({row}, {column}): {value}")]
    NonFiniteJacobian {
        row: usize,
        column: usize,
        value: f64,
    },
    /// A model advertised an exact Jacobian but did not return one.
    #[error("objective model declared an analytic Jacobian but supplied none")]
    MissingModelJacobian,
    /// The model rejected a parameter vector before evaluating its callbacks.
    #[error("trial parameter vector is outside the objective domain at iteration {iteration}")]
    InfeasibleTrialPoint { iteration: usize },
    /// The symbolic fitting adapter requires at least one formula and one sample variable.
    #[error(
        "symbolic model dimensions are unsupported: expected one or more residual equations and exactly one sample parameter, got {equations} and {parameters}"
    )]
    UnsupportedSymbolicModelDimensions { equations: usize, parameters: usize },
    /// The symbolic residual count overflowed while multiplying equations by samples.
    #[error("symbolic residual count overflow for {equations} equations and {samples} samples")]
    SymbolicResidualCountOverflow { equations: usize, samples: usize },
    /// The damped normal equations could not be solved.
    #[error("damped normal equations are singular at iteration {iteration}")]
    LinearSolveFailure { iteration: usize },
    /// The final normal matrix is singular and covariance cannot be computed.
    #[error("parameter covariance matrix is singular")]
    SingularCovariance,
    /// The covariance matrix has an invalid diagonal element.
    #[error("covariance diagonal at index {index} is invalid: {value}")]
    InvalidCovarianceDiagonal { index: usize, value: f64 },
    /// A computed intermediate became non-finite.
    #[error("numerical breakdown during {stage:?}")]
    NumericalBreakdown { stage: GavinLmNumericalStage },
}

/// Damping update formula used by Gavin's controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GavinLmUpdatePolicy {
    /// Marquardt diagonal scaling.
    Marquardt,
    /// Quadratic damping update.
    Quadratic,
    /// Nielsen damping update.
    Nielsen,
}

fn validate_analytic_jacobian(
    jacobian: &DMatrix<f64>,
    expected_rows: usize,
    expected_columns: usize,
) -> Result<(), GavinLmError> {
    if jacobian.shape() != (expected_rows, expected_columns) {
        return Err(GavinLmError::AnalyticJacobianShapeMismatch {
            expected_rows,
            expected_cols: expected_columns,
            actual_rows: jacobian.nrows(),
            actual_cols: jacobian.ncols(),
        });
    }
    if let Some((flat_index, value)) = jacobian
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        let row = flat_index % jacobian.nrows().max(1);
        let column = flat_index / jacobian.nrows().max(1);
        return Err(GavinLmError::NonFiniteJacobian { row, column, value });
    }
    Ok(())
}

/// How the most recent instrumented solve left its iteration loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GavinLmTermination {
    /// The weighted normal-equation gradient met its configured tolerance.
    GradientTolerance,
    /// The relative parameter update met its configured tolerance.
    ParameterStepTolerance,
    /// The configured iteration budget was exhausted.
    MaxIterations,
    /// A trial evaluation produced a non-finite residual.
    NonFiniteTrial,
}

impl GavinLmTermination {
    /// Whether the controller met one of its convergence tests.
    pub fn is_converged(self) -> bool {
        matches!(self, Self::GradientTolerance | Self::ParameterStepTolerance)
    }
}

/// Complete result from one Gavin LM solve.
///
/// Unlike the historical tuple, this carries the controller termination and
/// represents undefined R-squared explicitly instead of fabricating a value.
#[derive(Debug, Clone)]
pub struct GavinLmResult {
    /// Fitted parameter vector in the model's declared order.
    pub parameters: DVector<f64>,
    /// Reduced weighted chi-square for the accepted state.
    pub reduced_chi_square: f64,
    /// Approximate standard errors of fitted parameters.
    pub parameter_standard_errors: DVector<f64>,
    /// Approximate standard errors of model outputs at the sample locations.
    pub fitted_value_standard_errors: DVector<f64>,
    /// Approximate parameter correlation matrix.
    pub parameter_correlation: DMatrix<f64>,
    /// Weighted coefficient of determination; `None` when target variance is 0.
    pub r_squared: Option<f64>,
    /// Final weighted sum of squared residuals at the accepted parameters.
    pub sum_squared_residuals: f64,
    /// Columns: callback count, reduced chi-square, then accepted parameters.
    pub convergence_history: DMatrix<f64>,
    /// Why the controller stopped, whether or not it converged.
    pub termination: GavinLmTermination,
}

impl GavinLmResult {
    fn into_legacy_tuple(
        self,
    ) -> (
        DVector<f64>,
        f64,
        DVector<f64>,
        DVector<f64>,
        DMatrix<f64>,
        f64,
        DMatrix<f64>,
    ) {
        (
            self.parameters,
            self.reduced_chi_square,
            self.parameter_standard_errors,
            self.fitted_value_standard_errors,
            self.parameter_correlation,
            self.r_squared.unwrap_or(0.0),
            self.convergence_history,
        )
    }
}

/// Jacobian implementation used by the most recent instrumented solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GavinLmJacobianSource {
    /// Finite differences with periodic Broyden updates.
    FiniteDifference,
    /// User-supplied exact derivative callback.
    Analytic,
}

/// Optional aggregate diagnostics for one Gavin LM solve.
///
/// Counters and timers are updated only when collection is enabled. Stage
/// durations are diagnostic scopes and may overlap (for example, objective
/// evaluation occurs inside Jacobian construction).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GavinLmTelemetry {
    /// Requested collection level, needed to interpret zero-valued channels.
    pub telemetry_mode: SolverTelemetryMode,
    /// Number of outer LM iterations entered.
    pub iterations: usize,
    /// Termination reason for the most recent solve.
    pub termination: Option<GavinLmTermination>,
    /// Derivative source selected for this solve.
    pub jacobian_source: Option<GavinLmJacobianSource>,
    /// Number of objective callback invocations.
    pub objective_evaluations: usize,
    /// Number of finite-difference Jacobian rebuilds.
    pub finite_difference_jacobians: usize,
    /// Number of exact Jacobian callback invocations.
    pub analytic_jacobian_evaluations: usize,
    /// Number of rank-one Broyden Jacobian updates.
    pub broyden_updates: usize,
    /// Number of attempted parameter updates accepted by the rho test.
    pub accepted_steps: usize,
    /// Number of attempted parameter updates rejected by the rho test.
    pub rejected_steps: usize,
    /// Number of damped normal-equation solves attempted.
    pub linear_solves: usize,
    /// Cumulative time spent in objective callbacks.
    pub objective_time: Duration,
    /// Cumulative time spent assembling Jacobian and normal-equation data.
    pub linearization_time: Duration,
    /// Cumulative time spent in exact Jacobian callbacks.
    pub analytic_jacobian_time: Duration,
    /// Cumulative time spent solving the damped normal equations.
    pub linear_solve_time: Duration,
    /// Total elapsed time for the most recent successful solve.
    pub total_time: Duration,
}

/// Fallible numerical model callback consumed by the Gavin LM controller.
///
/// Implementations may provide an exact Jacobian directly. Symbolic models
/// prepared by Nonlinear_systems use SymbolicGavinObjective, which adapts the
/// prepared residual and Jacobian callbacks without repeating differentiation.
pub trait ObjectiveFunction {
    /// Evaluates all model outputs at the supplied sample locations and parameters.
    fn evaluate(
        &self,
        t: &DVector<f64>,
        p: &DVector<f64>,
    ) -> Result<DVector<f64>, GavinLmCallbackError>;
    /// Returns the number of fitted coefficients expected by the callback.
    fn parameter_count(&self) -> usize;

    /// Returns the number of scalar outputs for the requested sample count.
    /// Ordinary fitting models produce one output per sample; vector-valued
    /// symbolic systems may produce several residual equations per sample.
    fn output_count(&self, sample_count: usize) -> Option<usize> {
        Some(sample_count)
    }

    /// Validates a trial point before residual or Jacobian evaluation.
    fn validate_trial(&self, p: &DVector<f64>) -> bool {
        p.iter().all(|value| value.is_finite())
    }

    /// Supplies an exact model-output Jacobian when the model owns one.
    ///
    /// Implementations without an exact derivative return None; the solver
    /// then uses its configured closure or the FD/Broyden strategy.
    fn analytic_jacobian(
        &self,
        _t: &DVector<f64>,
        _p: &DVector<f64>,
    ) -> Option<Result<DMatrix<f64>, GavinLmCallbackError>> {
        None
    }

    /// Reports whether this model always provides an analytic Jacobian.
    fn has_analytic_jacobian(&self) -> bool {
        false
    }
}

/// Residual/Jacobian adapter for a prepared scalar symbolic model over samples.
///
/// For each data point and equation the evaluator emits `f_j(x_i, p) - y_ij`,
/// and the exact Jacobian contains `df_j(x_i, p)/dp`. Outputs use equation-major
/// layout, matching `sym_fitting`. The adapter borrows the prepared binding and
/// data, so neither symbolic preparation nor data copies happen inside solve.
pub struct SymbolicGavinObjective<'bound, 'prepared> {
    problem: &'bound crate::numerical::Nonlinear_systems::least_squares::BoundSymbolicLeastSquaresProblem<'prepared>,
    observations: &'bound [f64],
}

impl<'bound, 'prepared> SymbolicGavinObjective<'bound, 'prepared> {
    /// Adapts a prepared symbolic model with one sample parameter to Gavin's API.
    pub fn new(
        problem: &'bound crate::numerical::Nonlinear_systems::least_squares::BoundSymbolicLeastSquaresProblem<'prepared>,
        observations: &'bound [f64],
    ) -> Result<Self, GavinLmError> {
        let equations = problem.residual_count();
        let parameters = problem.equation_parameter_count();
        if equations == 0 || parameters != 1 {
            return Err(GavinLmError::UnsupportedSymbolicModelDimensions {
                equations,
                parameters,
            });
        }
        Ok(Self {
            problem,
            observations,
        })
    }

    /// Exposes one-time symbolic preparation telemetry separately from the
    /// Gavin solve telemetry collected by `LevenbergMarquardt`.
    pub fn preparation_report(
        &self,
    ) -> &crate::numerical::Nonlinear_systems::symbolic::SymbolicPreparationReport {
        self.problem.preparation_report()
    }

    /// Returns the ExprLegacy/AtomNative backend selected during preparation.
    pub fn backend_kind(
        &self,
    ) -> crate::numerical::Nonlinear_systems::symbolic::SymbolicBackendKind {
        self.problem.backend_kind()
    }
}

impl ObjectiveFunction for SymbolicGavinObjective<'_, '_> {
    fn evaluate(
        &self,
        t: &DVector<f64>,
        p: &DVector<f64>,
    ) -> Result<DVector<f64>, GavinLmCallbackError> {
        let expected = t
            .len()
            .checked_mul(self.problem.residual_count())
            .ok_or_else(|| {
                Box::new(GavinLmError::SymbolicResidualCountOverflow {
                    equations: self.problem.residual_count(),
                    samples: t.len(),
                }) as GavinLmCallbackError
            })?;
        if expected != self.observations.len() {
            return Err(Box::new(GavinLmError::SymbolicDataLengthMismatch {
                expected,
                actual: self.observations.len(),
            }));
        }
        let equation_count = self.problem.residual_count();
        let mut output = DVector::zeros(expected);
        let mut parameter_value = DVector::zeros(1);
        let mut point_output = DVector::zeros(equation_count);
        for (index, sample) in t.iter().copied().enumerate() {
            parameter_value[0] = sample;
            self.problem
                .residual_into_with_parameter_values(p, &parameter_value, &mut point_output)
                .map_err(|error| Box::new(error) as GavinLmCallbackError)?;
            for equation in 0..equation_count {
                let output_index = equation * t.len() + index;
                output[output_index] = point_output[equation] - self.observations[output_index];
            }
        }
        Ok(output)
    }

    fn parameter_count(&self) -> usize {
        self.problem.parameter_count()
    }

    fn output_count(&self, sample_count: usize) -> Option<usize> {
        sample_count.checked_mul(self.problem.residual_count())
    }

    fn validate_trial(&self, p: &DVector<f64>) -> bool {
        self.problem.validate_trial(p)
    }

    fn analytic_jacobian(
        &self,
        t: &DVector<f64>,
        p: &DVector<f64>,
    ) -> Option<Result<DMatrix<f64>, GavinLmCallbackError>> {
        let equation_count = self.problem.residual_count();
        let Some(output_rows) = t.len().checked_mul(equation_count) else {
            return Some(Err(Box::new(GavinLmError::SymbolicResidualCountOverflow {
                equations: equation_count,
                samples: t.len(),
            })));
        };
        let mut output = DMatrix::zeros(output_rows, p.len());
        let mut parameter_value = DVector::zeros(1);
        let mut point_jacobian = DMatrix::zeros(equation_count, p.len());
        for (sample_index, sample) in t.iter().copied().enumerate() {
            parameter_value[0] = sample;
            if let Err(error) = self.problem.jacobian_into_with_parameter_values(
                p,
                &parameter_value,
                &mut point_jacobian,
            ) {
                return Some(Err(Box::new(error)));
            }
            for equation in 0..equation_count {
                let output_row = equation * t.len() + sample_index;
                output
                    .row_mut(output_row)
                    .copy_from(&point_jacobian.row(equation));
            }
        }
        Some(Ok(output))
    }

    fn has_analytic_jacobian(&self) -> bool {
        true
    }
}

/// Gavin's damped, finite-difference/Broyden nonlinear least-squares fitter.
///
/// The controller follows the reference implementation in
/// `PY/levenberg_marquardt.py`. The default configuration is silent and does
/// not collect telemetry; callers can opt in per solver instance.
pub struct LevenbergMarquardt<F: ObjectiveFunction> {
    /// Iterations performed by the most recent solve.
    pub iteration: usize,
    /// Objective callback evaluations performed by the most recent solve.
    pub func_calls: usize,
    /// User-provided numerical model.
    pub objective_fn: F,
    /// Termination of the latest solve, available even when telemetry is off.
    pub last_termination: Option<GavinLmTermination>,
    /// Infinity-norm threshold for the weighted normal-equation gradient.
    pub eps_grad: f64,
    /// Relative parameter-step convergence threshold.
    pub eps_coeff: f64,
    /// Minimum gain ratio required to accept a trial step (Gavin epsilon_4).
    pub eps_chi: f64,
    /// Initial damping parameter.
    pub lambda_0: f64,
    /// Multiplier applied after a rejected Marquardt step.
    pub lambda_up_fac: f64,
    /// Divisor applied after an accepted Marquardt step.
    pub lambda_dn_fac: f64,
    /// Damping update formula.
    pub update_policy: GavinLmUpdatePolicy,
    /// Maximum number of controller iterations (1000 by default).
    pub max_iterations: usize,
    /// Enables log-facade messages for this solver instance; disabled by default.
    logging_enabled: bool,
    /// Presence enables the requested collection level; `None` keeps the hot path minimal.
    telemetry: Option<GavinLmTelemetry>,
    telemetry_mode: SolverTelemetryMode,
    /// Optional exact model-output derivative, shaped (samples, parameters).
    analytic_jacobian: Option<
        Box<dyn Fn(&DVector<f64>, &DVector<f64>) -> Result<DMatrix<f64>, GavinLmCallbackError>>,
    >,
}

impl<F: ObjectiveFunction> LevenbergMarquardt<F> {
    pub fn new(objective_fn: F) -> Self {
        let epsilon_1 = 1e-3;
        let epsilon_2 = 1e-3;
        let epsilon_4 = 1e-1;

        Self {
            iteration: 0,
            func_calls: 0,
            objective_fn,
            last_termination: None,
            eps_grad: epsilon_1,
            eps_coeff: epsilon_2,
            eps_chi: epsilon_4,
            lambda_0: 1e-2,
            lambda_up_fac: 11.0,
            lambda_dn_fac: 9.0,
            update_policy: GavinLmUpdatePolicy::Marquardt,
            max_iterations: 1000,
            logging_enabled: false,
            telemetry: None,
            telemetry_mode: SolverTelemetryMode::Off,
            analytic_jacobian: None,
        }
    }

    /// Enables solver diagnostics through the crate's configured `log` logger.
    pub fn with_logging(mut self, enabled: bool) -> Self {
        self.logging_enabled = enabled;
        self
    }

    /// Selects discrete counters, timings, both, or no solve telemetry.
    pub fn with_telemetry(mut self, mode: SolverTelemetryMode) -> Self {
        self.telemetry_mode = mode;
        self.telemetry = mode.is_enabled().then(|| GavinLmTelemetry {
            telemetry_mode: mode,
            ..GavinLmTelemetry::default()
        });
        self
    }

    /// Sets the iteration budget without changing the selected damping policy.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Sets the controller tolerances.
    pub fn with_tolerances(mut self, gradient: f64, parameter_step: f64, gain_ratio: f64) -> Self {
        self.eps_grad = gradient;
        self.eps_coeff = parameter_step;
        self.eps_chi = gain_ratio;
        self
    }

    /// Sets the three damping scale factors.
    pub fn with_damping_factors(mut self, initial: f64, increase: f64, decrease: f64) -> Self {
        self.lambda_0 = initial;
        self.lambda_up_fac = increase;
        self.lambda_dn_fac = decrease;
        self
    }

    /// Selects Gavin's damping update policy.
    pub fn with_update_policy(mut self, policy: GavinLmUpdatePolicy) -> Self {
        self.update_policy = policy;
        self
    }

    /// Selects an exact Jacobian callback for model outputs `f(t, p)`.
    ///
    /// The callback must return `df/dp` with shape `(t.len(), p.len())`.
    /// It is evaluated at each accepted parameter state. Rejected trials reuse
    /// the Jacobian at the unchanged accepted state. The callback is never
    /// invoked from symbolic construction code by this numerical solver.
    pub fn with_analytic_jacobian<J>(mut self, jacobian: J) -> Self
    where
        J: Fn(&DVector<f64>, &DVector<f64>) -> DMatrix<f64> + 'static,
    {
        self.analytic_jacobian = Some(Box::new(move |t, p| Ok(jacobian(t, p))));
        self
    }

    /// Selects a fallible exact Jacobian callback, preserving callback errors.
    ///
    /// This is the adapter point for prepared evaluators whose runtime can
    /// report typed failures, including symbolic Lambdify/AOT evaluators.
    pub fn with_fallible_analytic_jacobian<J, E>(mut self, jacobian: J) -> Self
    where
        J: Fn(&DVector<f64>, &DVector<f64>) -> Result<DMatrix<f64>, E> + 'static,
        E: Error + Send + Sync + 'static,
    {
        self.analytic_jacobian = Some(Box::new(move |t, p| {
            jacobian(t, p).map_err(|error| Box::new(error) as GavinLmCallbackError)
        }));
        self
    }

    /// Restores the default finite-difference/Broyden Jacobian route.
    pub fn with_finite_difference_jacobian(mut self) -> Self {
        self.analytic_jacobian = None;
        self
    }

    /// Returns diagnostics for the most recent solve when collection is enabled.
    pub fn telemetry(&self) -> Option<&GavinLmTelemetry> {
        self.telemetry.as_ref()
    }

    fn evaluate_objective(
        &mut self,
        t: &DVector<f64>,
        p: &DVector<f64>,
        expected_outputs: usize,
    ) -> Result<DVector<f64>, GavinLmError> {
        if !self.objective_fn.validate_trial(p) {
            return Err(GavinLmError::InfeasibleTrialPoint {
                iteration: self.iteration,
            });
        }
        let started = self.telemetry_mode.collects_timings().then(Instant::now);
        let result = self.objective_fn.evaluate(t, p);
        self.func_calls += 1;
        if self.telemetry_mode.collects_counters() {
            if let Some(stats) = self.telemetry.as_mut() {
                stats.objective_evaluations += 1;
            }
        }
        if let (Some(stats), Some(started)) = (self.telemetry.as_mut(), started) {
            stats.objective_time += started.elapsed();
        }
        let value = result.map_err(GavinLmError::ObjectiveCallback)?;
        if value.len() != expected_outputs {
            return Err(GavinLmError::ObjectiveOutputLengthMismatch {
                expected: expected_outputs,
                actual: value.len(),
            });
        }
        if let Some((index, value)) = value
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(GavinLmError::NonFiniteObjective { index, value });
        }
        Ok(value)
    }
    fn lm_fd_j(
        &mut self,
        t: &DVector<f64>,
        p: &DVector<f64>,
        y: &DVector<f64>,
        dp: &DVector<f64>,
    ) -> Result<DMatrix<f64>, GavinLmError> {
        let m = y.len();
        let n = p.len();
        let mut j = DMatrix::zeros(m, n);
        let mut del = DVector::zeros(n);

        // A negative fractional increment selects a backward difference; a
        // positive one selects a centered difference; zero freezes a column.
        for j_idx in 0..n {
            del[j_idx] = dp[j_idx] * (1.0 + p[j_idx].abs());
            if del[j_idx] == 0.0 {
                continue;
            }

            let mut plus = p.clone();
            plus[j_idx] += del[j_idx];
            let plus_valid = self.objective_fn.validate_trial(&plus);
            let mut minus = p.clone();
            minus[j_idx] -= del[j_idx];
            let minus_valid = self.objective_fn.validate_trial(&minus);

            // Keep finite differences inside declared domains. Prefer Gavin's
            // configured direction; use the opposite one-sided difference if
            // a log/positive constraint excludes that perturbation.
            if dp[j_idx] < 0.0 {
                if plus_valid {
                    let y_plus = self.evaluate_objective(t, &plus, y.len())?;
                    for i in 0..m {
                        j[(i, j_idx)] = (y_plus[i] - y[i]) / del[j_idx];
                    }
                } else if minus_valid {
                    let y_minus = self.evaluate_objective(t, &minus, y.len())?;
                    for i in 0..m {
                        j[(i, j_idx)] = (y[i] - y_minus[i]) / del[j_idx];
                    }
                }
            } else if plus_valid && minus_valid {
                let y_plus = self.evaluate_objective(t, &plus, y.len())?;
                let y_minus = self.evaluate_objective(t, &minus, y.len())?;
                for i in 0..m {
                    j[(i, j_idx)] = (y_plus[i] - y_minus[i]) / (2.0 * del[j_idx]);
                }
            } else if plus_valid {
                let y_plus = self.evaluate_objective(t, &plus, y.len())?;
                for i in 0..m {
                    j[(i, j_idx)] = (y_plus[i] - y[i]) / del[j_idx];
                }
            } else if minus_valid {
                let y_minus = self.evaluate_objective(t, &minus, y.len())?;
                for i in 0..m {
                    j[(i, j_idx)] = (y[i] - y_minus[i]) / del[j_idx];
                }
            }
        }
        Ok(j)
    }

    fn lm_broyden_j(
        &self,
        p_old: &DVector<f64>,
        y_old: &DVector<f64>,
        j: &DMatrix<f64>,
        p: &DVector<f64>,
        y: &DVector<f64>,
    ) -> DMatrix<f64> {
        // Rank-one Broyden update enforces the secant condition along the last
        // accepted parameter displacement. The caller must avoid a zero step,
        // for which the denominator below is undefined.
        let h = p - p_old;
        let dy = y - y_old;
        let denominator = h.dot(&h);
        if denominator == 0.0 || !denominator.is_finite() {
            return j.clone();
        }
        let jh = j * &h;
        let a = (&dy - &jh) * h.transpose();
        j + a / denominator
    }

    fn lm_matx(
        &mut self,
        t: &DVector<f64>,
        p_old: &DVector<f64>,
        y_old: &DVector<f64>,
        dx2: f64,
        j: &DMatrix<f64>,
        p: &DVector<f64>,
        y_dat: &DVector<f64>,
        weight: &DVector<f64>,
        dp: &DVector<f64>,
    ) -> Result<(DMatrix<f64>, DVector<f64>, f64, DVector<f64>, DMatrix<f64>), GavinLmError> {
        self.lm_matx_with_output(t, p_old, y_old, dx2, j, p, y_dat, weight, dp, None)
    }

    fn lm_matx_with_output(
        &mut self,
        t: &DVector<f64>,
        p_old: &DVector<f64>,
        y_old: &DVector<f64>,
        dx2: f64,
        j: &DMatrix<f64>,
        p: &DVector<f64>,
        y_dat: &DVector<f64>,
        weight: &DVector<f64>,
        dp: &DVector<f64>,
        cached_output: Option<&DVector<f64>>,
    ) -> Result<(DMatrix<f64>, DVector<f64>, f64, DVector<f64>, DMatrix<f64>), GavinLmError> {
        let linearization_started = self.telemetry_mode.collects_timings().then(Instant::now);
        let npar = p.len();
        let y_hat = match cached_output {
            Some(output) => output.clone(),
            None => self.evaluate_objective(t, p, y_dat.len())?,
        };
        if y_hat.len() != y_dat.len() {
            return Err(GavinLmError::ObjectiveOutputLengthMismatch {
                expected: y_dat.len(),
                actual: y_hat.len(),
            });
        }
        if let Some((index, value)) = y_hat
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(GavinLmError::NonFiniteObjective { index, value });
        }

        let displacement_sq = (p - p_old).norm_squared();
        let j_new = if let Some(analytic_jacobian) = &self.analytic_jacobian {
            let started = self.telemetry_mode.collects_timings().then(Instant::now);
            let exact_result = analytic_jacobian(t, p);
            if self.telemetry_mode.collects_counters() {
                if let Some(stats) = self.telemetry.as_mut() {
                    stats.analytic_jacobian_evaluations += 1;
                }
            }
            if let (Some(stats), Some(started)) = (self.telemetry.as_mut(), started) {
                stats.analytic_jacobian_time += started.elapsed();
            }
            let exact = exact_result.map_err(GavinLmError::AnalyticJacobianCallback)?;
            validate_analytic_jacobian(&exact, y_dat.len(), npar)?;
            exact
        } else if self.objective_fn.has_analytic_jacobian() {
            let started = self.telemetry_mode.collects_timings().then(Instant::now);
            let exact_result = self.objective_fn.analytic_jacobian(t, p);
            if self.telemetry_mode.collects_counters() {
                if let Some(stats) = self.telemetry.as_mut() {
                    stats.analytic_jacobian_evaluations += 1;
                }
            }
            if let (Some(stats), Some(started)) = (self.telemetry.as_mut(), started) {
                stats.analytic_jacobian_time += started.elapsed();
            }
            let exact = exact_result
                .ok_or(GavinLmError::MissingModelJacobian)?
                .map_err(GavinLmError::AnalyticJacobianCallback)?;
            validate_analytic_jacobian(&exact, y_dat.len(), npar)?;
            exact
        } else if self.iteration % (2 * npar) == 0
            || dx2 > 0.0
            || displacement_sq == 0.0
            || !displacement_sq.is_finite()
        {
            if self.telemetry_mode.collects_counters() {
                if let Some(stats) = &mut self.telemetry {
                    stats.finite_difference_jacobians += 1;
                }
            }
            self.lm_fd_j(t, p, &y_hat, dp)?
        } else {
            if self.telemetry_mode.collects_counters() {
                if let Some(stats) = &mut self.telemetry {
                    stats.broyden_updates += 1;
                }
            }
            self.lm_broyden_j(p_old, y_old, j, p, &y_hat)
        };

        let delta_y = y_dat - &y_hat;
        let weighted_delta = delta_y.component_mul(weight);
        let chi_sq = weighted_delta.dot(&delta_y);

        let mut jtw_j = DMatrix::zeros(npar, npar);
        let mut jtw_dy = DVector::zeros(npar);

        for i in 0..npar {
            for j_idx in 0..npar {
                let mut sum = 0.0;
                for k in 0..y_dat.len() {
                    sum += j_new[(k, i)] * j_new[(k, j_idx)] * weight[k];
                }
                jtw_j[(i, j_idx)] = sum;
            }

            let mut sum = 0.0;
            for k in 0..y_dat.len() {
                sum += j_new[(k, i)] * weight[k] * delta_y[k];
            }
            jtw_dy[i] = sum;
        }

        if !chi_sq.is_finite()
            || jtw_j.iter().any(|value| !value.is_finite())
            || jtw_dy.iter().any(|value| !value.is_finite())
        {
            return Err(GavinLmError::NumericalBreakdown {
                stage: GavinLmNumericalStage::WeightedLinearization,
            });
        }

        if let (Some(stats), Some(started)) = (self.telemetry.as_mut(), linearization_started) {
            stats.linearization_time += started.elapsed();
        }
        Ok((jtw_j, jtw_dy, chi_sq, y_hat, j_new))
    }

    /// Fits model predictions to observations using Gavin's reference
    /// normalization `w_i = 1 / ||y||^2`.
    pub fn lm(
        &mut self,
        p: DVector<f64>,
        t: &DVector<f64>,
        y_dat: &DVector<f64>,
    ) -> Result<
        (
            DVector<f64>,
            f64,
            DVector<f64>,
            DVector<f64>,
            DMatrix<f64>,
            f64,
            DMatrix<f64>,
        ),
        GavinLmError,
    > {
        self.lm_impl(p, t, y_dat, None)
            .map(GavinLmResult::into_legacy_tuple)
    }

    /// Fits predictions and observations, returning the explicit controller status.
    pub fn lm_detailed(
        &mut self,
        p: DVector<f64>,
        t: &DVector<f64>,
        y_dat: &DVector<f64>,
    ) -> Result<GavinLmResult, GavinLmError> {
        self.lm_impl(p, t, y_dat, None)
    }

    /// Solves a least-squares problem with caller-supplied nonnegative weights.
    /// The objective is `sum_i weights[i] * (f_i(p) - y_i)^2`.
    pub fn lm_with_weights(
        &mut self,
        p: DVector<f64>,
        t: &DVector<f64>,
        y_dat: &DVector<f64>,
        weights: &DVector<f64>,
    ) -> Result<
        (
            DVector<f64>,
            f64,
            DVector<f64>,
            DVector<f64>,
            DMatrix<f64>,
            f64,
            DMatrix<f64>,
        ),
        GavinLmError,
    > {
        self.lm_impl(p, t, y_dat, Some(weights))
            .map(GavinLmResult::into_legacy_tuple)
    }

    /// Weighted fitting with explicit termination and undefined-statistic states.
    pub fn lm_with_weights_detailed(
        &mut self,
        p: DVector<f64>,
        t: &DVector<f64>,
        y_dat: &DVector<f64>,
        weights: &DVector<f64>,
    ) -> Result<GavinLmResult, GavinLmError> {
        self.lm_impl(p, t, y_dat, Some(weights))
    }

    /// Minimizes residuals directly with unit weights and no observation-norm
    /// normalization. The objective is `sum_i r_i(p)^2`.
    pub fn lm_residuals(
        &mut self,
        p: DVector<f64>,
        t: &DVector<f64>,
    ) -> Result<
        (
            DVector<f64>,
            f64,
            DVector<f64>,
            DVector<f64>,
            DMatrix<f64>,
            f64,
            DMatrix<f64>,
        ),
        GavinLmError,
    > {
        let output_count = self
            .objective_fn
            .output_count(t.len())
            .ok_or(GavinLmError::OutputCountOverflow { samples: t.len() })?;
        let zero_target = DVector::zeros(output_count);
        let unit_weights = DVector::from_element(output_count, 1.0);
        self.lm_impl(p, t, &zero_target, Some(&unit_weights))
            .map(GavinLmResult::into_legacy_tuple)
    }

    /// Direct residual solve with the complete typed result.
    pub fn lm_residuals_detailed(
        &mut self,
        p: DVector<f64>,
        t: &DVector<f64>,
    ) -> Result<GavinLmResult, GavinLmError> {
        let output_count = self
            .objective_fn
            .output_count(t.len())
            .ok_or(GavinLmError::OutputCountOverflow { samples: t.len() })?;
        self.lm_impl(
            p,
            t,
            &DVector::zeros(output_count),
            Some(&DVector::from_element(output_count, 1.0)),
        )
    }

    /// Direct residual solve with explicit nonnegative observation weights.
    pub fn lm_residuals_with_weights_detailed(
        &mut self,
        p: DVector<f64>,
        t: &DVector<f64>,
        weights: &DVector<f64>,
    ) -> Result<GavinLmResult, GavinLmError> {
        let output_count = self
            .objective_fn
            .output_count(t.len())
            .ok_or(GavinLmError::OutputCountOverflow { samples: t.len() })?;
        self.lm_impl(p, t, &DVector::zeros(output_count), Some(weights))
    }

    /// Minimizes direct residuals using explicit nonnegative weights.
    pub fn lm_residuals_with_weights(
        &mut self,
        p: DVector<f64>,
        t: &DVector<f64>,
        weights: &DVector<f64>,
    ) -> Result<
        (
            DVector<f64>,
            f64,
            DVector<f64>,
            DVector<f64>,
            DMatrix<f64>,
            f64,
            DMatrix<f64>,
        ),
        GavinLmError,
    > {
        let output_count = self
            .objective_fn
            .output_count(t.len())
            .ok_or(GavinLmError::OutputCountOverflow { samples: t.len() })?;
        let zero_target = DVector::zeros(output_count);
        self.lm_impl(p, t, &zero_target, Some(weights))
            .map(GavinLmResult::into_legacy_tuple)
    }

    fn lm_impl(
        &mut self,
        mut p: DVector<f64>,
        t: &DVector<f64>,
        y_dat: &DVector<f64>,
        explicit_weights: Option<&DVector<f64>>,
    ) -> Result<GavinLmResult, GavinLmError> {
        let log_level = if self.logging_enabled {
            SolverLogLevel::Info
        } else {
            SolverLogLevel::Off
        };
        let _logging_scope = SolverLogGuard::enter(log_level);
        // Reset per-run counters while preserving the user's diagnostics policy.
        self.iteration = 0;
        self.func_calls = 0;
        self.last_termination = None;
        if let Some(stats) = &mut self.telemetry {
            *stats = GavinLmTelemetry::default();
            stats.telemetry_mode = self.telemetry_mode;
            stats.jacobian_source = Some(
                if self.analytic_jacobian.is_some() || self.objective_fn.has_analytic_jacobian() {
                    GavinLmJacobianSource::Analytic
                } else {
                    GavinLmJacobianSource::FiniteDifference
                },
            );
        }
        let solve_started = self.telemetry_mode.collects_timings().then(Instant::now);

        let expected_outputs = self
            .objective_fn
            .output_count(t.len())
            .ok_or(GavinLmError::OutputCountOverflow { samples: t.len() })?;
        if expected_outputs != y_dat.len() {
            return Err(GavinLmError::ModelTargetLengthMismatch {
                expected: expected_outputs,
                actual: y_dat.len(),
            });
        }

        let expected_parameters = self.objective_fn.parameter_count();
        if expected_parameters == 0 {
            return Err(GavinLmError::NoParameters);
        }
        if p.len() != expected_parameters {
            return Err(GavinLmError::ParameterLengthMismatch {
                expected: expected_parameters,
                actual: p.len(),
            });
        }

        if y_dat.is_empty() {
            return Err(GavinLmError::EmptyObservations);
        }
        if y_dat.len() < p.len() {
            return Err(GavinLmError::InsufficientObservations {
                parameters: p.len(),
                observations: y_dat.len(),
            });
        }

        for (input, values) in [
            (GavinLmInput::SampleLocations, t),
            (GavinLmInput::Observations, y_dat),
            (GavinLmInput::InitialParameters, &p),
        ] {
            if let Some((index, value)) = values
                .iter()
                .copied()
                .enumerate()
                .find(|(_, value)| !value.is_finite())
            {
                return Err(GavinLmError::NonFiniteInput {
                    input,
                    index,
                    value,
                });
            }
        }
        for (option, value, minimum, inclusive) in [
            (GavinLmOption::GradientTolerance, self.eps_grad, 0.0, false),
            (
                GavinLmOption::ParameterStepTolerance,
                self.eps_coeff,
                0.0,
                false,
            ),
            (GavinLmOption::GainRatioThreshold, self.eps_chi, 0.0, true),
            (GavinLmOption::InitialDamping, self.lambda_0, 0.0, false),
            (
                GavinLmOption::DampingIncreaseFactor,
                self.lambda_up_fac,
                1.0,
                false,
            ),
            (
                GavinLmOption::DampingDecreaseFactor,
                self.lambda_dn_fac,
                1.0,
                false,
            ),
        ] {
            if !value.is_finite()
                || if inclusive {
                    value < minimum
                } else {
                    value <= minimum
                }
            {
                return Err(GavinLmError::InvalidOption { option, value });
            }
        }

        let npar = p.len();
        let npnt = y_dat.len();
        let mut p_old = DVector::zeros(npar);
        let mut y_old = DVector::zeros(npnt);
        let j = DMatrix::zeros(npnt, npar);
        let dof = (npnt - npar + 1) as f64;

        let explicit_weighting = explicit_weights.is_some();
        let weight = if let Some(weights) = explicit_weights {
            if weights.len() != npnt {
                return Err(GavinLmError::WeightLengthMismatch {
                    expected: npnt,
                    actual: weights.len(),
                });
            }
            if let Some((index, value)) = weights
                .iter()
                .copied()
                .enumerate()
                .find(|(_, value)| !value.is_finite() || *value < 0.0)
            {
                return Err(GavinLmError::InvalidWeight { index, value });
            }
            if weights.iter().all(|weight| *weight == 0.0) {
                return Err(GavinLmError::AllWeightsZero);
            }
            weights.clone()
        } else {
            let y_dat_norm_sq = y_dat.dot(y_dat);
            if !y_dat_norm_sq.is_finite() || y_dat_norm_sq <= 0.0 {
                return Err(GavinLmError::InvalidObservationNorm);
            }
            // Preserve the Python fitting API's historical normalization.
            DVector::from_element(npnt, 1.0 / y_dat_norm_sq)
        };
        let dp = DVector::from_element(npar, -0.001);
        // These are Gavin's reference bounds, not user-configurable box bounds.
        // As in that reference, an exactly-zero initial coefficient is fixed at
        // zero by this rule; callers needing such a coefficient should start
        // from a small nonzero value.
        let p_min = -100.0 * p.map(|x| x.abs());
        let p_max = 100.0 * p.map(|x| x.abs());

        if self.max_iterations == 0 {
            return Err(GavinLmError::ZeroIterationBudget);
        }
        let max_iter = self.max_iterations;
        let eps_grad = self.eps_grad;
        let eps_coeff = self.eps_coeff;
        let eps_chi = self.eps_chi;
        let lambda_0 = self.lambda_0;
        let lambda_up_fac = self.lambda_up_fac;
        let lambda_dn_fac = self.lambda_dn_fac;
        let update_policy = self.update_policy;

        let idx: Vec<usize> = (0..npar).collect();
        let (mut jtw_j, mut jtw_dy, mut x2_val, mut y_hat, mut j_mat) =
            self.lm_matx(t, &p_old, &y_old, 1.0, &j, &p, y_dat, &weight, &dp)?;

        if jtw_dy.amax() < eps_grad {
            if self.logging_enabled {
                log::info!("Gavin LM: initial guess is already close to stationary");
            }
        }

        let mut lambda = if update_policy == GavinLmUpdatePolicy::Marquardt {
            lambda_0
        } else {
            lambda_0 * jtw_j.diagonal().max()
        };

        let mut nu = 2.0;
        // Gavin's reference keeps the previous accepted objective separately:
        // on rejection, X2 is restored to this value; on acceptance it advances
        // immediately before the new linearization is assembled.
        let mut x2_old = x2_val;
        // The history layout mirrors Gavin's reference: callback count,
        // reduced chi-square, followed by the accepted parameter vector.
        let mut cvg_hst = DMatrix::zeros(max_iter, npar + 2);

        let mut termination = GavinLmTermination::MaxIterations;
        while self.iteration < max_iter {
            self.iteration += 1;
            if self.telemetry_mode.collects_counters() {
                if let Some(stats) = &mut self.telemetry {
                    stats.iterations = self.iteration;
                }
            }

            let linear_solve_started = self.telemetry_mode.collects_timings().then(Instant::now);
            if self.telemetry_mode.collects_counters() {
                if let Some(stats) = &mut self.telemetry {
                    stats.linear_solves += 1;
                }
            }
            // Solve the damped normal equations for the proposed coefficient
            // correction. Marquardt scales each diagonal by its Hessian entry;
            // the quadratic and Nielsen variants add lambda*I instead.
            let mut h = if update_policy == GavinLmUpdatePolicy::Marquardt {
                let mut jtw_j_reg = jtw_j.clone();
                for i in 0..npar {
                    jtw_j_reg[(i, i)] += lambda * jtw_j[(i, i)];
                }
                match jtw_j_reg.lu().solve(&jtw_dy) {
                    Some(solution) => solution,
                    None => {
                        return Err(GavinLmError::LinearSolveFailure {
                            iteration: self.iteration,
                        });
                    }
                }
            } else {
                let mut jtw_j_reg = jtw_j.clone();
                for i in 0..npar {
                    jtw_j_reg[(i, i)] += lambda;
                }
                match jtw_j_reg.lu().solve(&jtw_dy) {
                    Some(solution) => solution,
                    None => {
                        return Err(GavinLmError::LinearSolveFailure {
                            iteration: self.iteration,
                        });
                    }
                }
            };
            if let (Some(stats), Some(started)) = (self.telemetry.as_mut(), linear_solve_started) {
                stats.linear_solve_time += started.elapsed();
            }

            let mut p_try = p.clone();
            for i in &idx {
                p_try[*i] += h[*i];
            }

            for i in 0..npar {
                p_try[i] = p_try[i].max(p_min[i]).min(p_max[i]);
            }

            if !self.objective_fn.validate_trial(&p_try) {
                if self.telemetry_mode.collects_counters() {
                    if let Some(stats) = &mut self.telemetry {
                        stats.rejected_steps += 1;
                    }
                }
                x2_val = x2_old;
                lambda = match update_policy {
                    GavinLmUpdatePolicy::Marquardt => (lambda * lambda_up_fac).min(1e7),
                    GavinLmUpdatePolicy::Quadratic => (lambda * lambda_up_fac).min(1e7),
                    GavinLmUpdatePolicy::Nielsen => {
                        let updated = lambda * nu;
                        nu *= 2.0;
                        updated
                    }
                };
                if !lambda.is_finite() || lambda <= 0.0 {
                    return Err(GavinLmError::NumericalBreakdown {
                        stage: GavinLmNumericalStage::DampingParameterUpdate,
                    });
                }
                cvg_hst[(self.iteration - 1, 0)] = self.func_calls as f64;
                cvg_hst[(self.iteration - 1, 1)] = x2_val / dof;
                for i in 0..npar {
                    cvg_hst[(self.iteration - 1, i + 2)] = p[i];
                }
                continue;
            }

            let mut y_hat_try = match self.evaluate_objective(t, &p_try, npnt) {
                Ok(output) => output,
                Err(GavinLmError::NonFiniteObjective { .. }) => {
                    termination = GavinLmTermination::NonFiniteTrial;
                    cvg_hst[(self.iteration - 1, 0)] = self.func_calls as f64;
                    cvg_hst[(self.iteration - 1, 1)] = x2_val / dof;
                    for i in 0..npar {
                        cvg_hst[(self.iteration - 1, i + 2)] = p[i];
                    }
                    if self.logging_enabled {
                        log::warn!(
                            "Gavin LM: non-finite trial output; keeping last accepted parameters"
                        );
                    }
                    break;
                }
                Err(error) => return Err(error),
            };
            let mut delta_y_try = y_dat - &y_hat_try;

            if !delta_y_try.iter().all(|&x| x.is_finite()) {
                termination = GavinLmTermination::NonFiniteTrial;
                cvg_hst[(self.iteration - 1, 0)] = self.func_calls as f64;
                cvg_hst[(self.iteration - 1, 1)] = x2_val / dof;
                for i in 0..npar {
                    cvg_hst[(self.iteration - 1, i + 2)] = p[i];
                }
                if self.logging_enabled {
                    log::warn!(
                        "Gavin LM: non-finite trial residual; keeping last accepted parameters"
                    );
                }
                break;
            }

            let mut x2_try = delta_y_try.component_mul(&weight).dot(&delta_y_try);
            if !x2_try.is_finite() {
                return Err(GavinLmError::NumericalBreakdown {
                    stage: GavinLmNumericalStage::TrialObjective,
                });
            }

            let mut alpha = 1.0;
            if update_policy == GavinLmUpdatePolicy::Quadratic {
                let numerator = jtw_dy.dot(&h);
                let denominator = (x2_try - x2_val) / 2.0 + 2.0 * jtw_dy.dot(&h);
                if denominator == 0.0 || !denominator.is_finite() {
                    return Err(GavinLmError::NumericalBreakdown {
                        stage: GavinLmNumericalStage::QuadraticDampingUpdate,
                    });
                }
                alpha = numerator / denominator;
                if !alpha.is_finite() {
                    return Err(GavinLmError::NumericalBreakdown {
                        stage: GavinLmNumericalStage::QuadraticDampingScale,
                    });
                }
                h *= alpha;

                for i in &idx {
                    p_try[*i] = p[*i] + h[*i];
                }

                for i in 0..npar {
                    p_try[i] = p_try[i].max(p_min[i]).min(p_max[i]);
                }

                if !self.objective_fn.validate_trial(&p_try) {
                    if self.telemetry_mode.collects_counters() {
                        if let Some(stats) = &mut self.telemetry {
                            stats.rejected_steps += 1;
                        }
                    }
                    x2_val = x2_old;
                    lambda = (lambda * lambda_up_fac).min(1e7);
                    if !lambda.is_finite() || lambda <= 0.0 {
                        return Err(GavinLmError::NumericalBreakdown {
                            stage: GavinLmNumericalStage::DampingParameterUpdate,
                        });
                    }
                    cvg_hst[(self.iteration - 1, 0)] = self.func_calls as f64;
                    cvg_hst[(self.iteration - 1, 1)] = x2_val / dof;
                    for i in 0..npar {
                        cvg_hst[(self.iteration - 1, i + 2)] = p[i];
                    }
                    continue;
                }

                y_hat_try = match self.evaluate_objective(t, &p_try, npnt) {
                    Ok(output) => output,
                    Err(GavinLmError::NonFiniteObjective { .. }) => {
                        termination = GavinLmTermination::NonFiniteTrial;
                        cvg_hst[(self.iteration - 1, 0)] = self.func_calls as f64;
                        cvg_hst[(self.iteration - 1, 1)] = x2_val / dof;
                        for i in 0..npar {
                            cvg_hst[(self.iteration - 1, i + 2)] = p[i];
                        }
                        if self.logging_enabled {
                            log::warn!(
                                "Gavin LM: non-finite quadratic trial output; keeping last accepted parameters"
                            );
                        }
                        break;
                    }
                    Err(error) => return Err(error),
                };
                delta_y_try = y_dat - &y_hat_try;
                if !delta_y_try.iter().all(|value| value.is_finite()) {
                    termination = GavinLmTermination::NonFiniteTrial;
                    cvg_hst[(self.iteration - 1, 0)] = self.func_calls as f64;
                    cvg_hst[(self.iteration - 1, 1)] = x2_val / dof;
                    for i in 0..npar {
                        cvg_hst[(self.iteration - 1, i + 2)] = p[i];
                    }
                    if self.logging_enabled {
                        log::warn!(
                            "Gavin LM: non-finite quadratic trial residual; keeping last accepted parameters"
                        );
                    }
                    break;
                }
                x2_try = delta_y_try.component_mul(&weight).dot(&delta_y_try);
                if !x2_try.is_finite() {
                    return Err(GavinLmError::NumericalBreakdown {
                        stage: GavinLmNumericalStage::QuadraticTrialObjective,
                    });
                }
            }

            // Gain ratio compares the predicted decrease of the local model
            // with the decrease actually observed at the trial parameters.
            let rho_num = h.dot(&(lambda * &h + &jtw_dy));
            let rho_den = x2_val - x2_try;
            let rho = if rho_den == 0.0 {
                0.0
            } else {
                rho_num / rho_den
            };
            if !rho.is_finite() {
                return Err(GavinLmError::NumericalBreakdown {
                    stage: GavinLmNumericalStage::GainRatio,
                });
            }

            if rho > eps_chi {
                if self.telemetry_mode.collects_counters() {
                    if let Some(stats) = &mut self.telemetry {
                        stats.accepted_steps += 1;
                    }
                }
                let dx2 = x2_val - x2_old;
                x2_old = x2_val;
                p_old = p.clone();
                y_old = y_hat.clone();
                p = p_try;

                let (new_jtw_j, new_jtw_dy, new_x2, new_y_hat, new_j) = self.lm_matx_with_output(
                    t,
                    &p_old,
                    &y_old,
                    dx2,
                    &j_mat,
                    &p,
                    y_dat,
                    &weight,
                    &dp,
                    Some(&y_hat_try),
                )?;

                jtw_j = new_jtw_j;
                jtw_dy = new_jtw_dy;
                x2_val = new_x2;
                y_hat = new_y_hat;
                j_mat = new_j;

                if update_policy == GavinLmUpdatePolicy::Marquardt {
                    lambda = (lambda / lambda_dn_fac).max(1e-7);
                } else if update_policy == GavinLmUpdatePolicy::Quadratic {
                    lambda = (lambda / (1.0 + alpha)).max(1e-7);
                } else {
                    lambda =
                        lambda * (1.0 / 3.0 as f64).max((1.0 - (2.0 * rho - 1.0).powi(3)) as f64);
                    nu = 2.0;
                }
            } else {
                if self.telemetry_mode.collects_counters() {
                    if let Some(stats) = &mut self.telemetry {
                        stats.rejected_steps += 1;
                    }
                }
                x2_val = x2_old;

                if self.analytic_jacobian.is_none() && self.iteration % (2 * npar) == 0 {
                    let (new_jtw_j, new_jtw_dy, new_x2, new_y_hat, new_j) =
                        self.lm_matx(t, &p_old, &y_old, -1.0, &j_mat, &p, y_dat, &weight, &dp)?;
                    jtw_j = new_jtw_j;
                    jtw_dy = new_jtw_dy;
                    x2_val = new_x2;
                    y_hat = new_y_hat;
                    j_mat = new_j;
                }

                if update_policy == GavinLmUpdatePolicy::Marquardt {
                    lambda = (lambda * lambda_up_fac).min(1e7);
                } else if update_policy == GavinLmUpdatePolicy::Quadratic {
                    lambda = lambda + ((x2_try - x2_val) / 2.0 / alpha).abs();
                } else {
                    lambda = lambda * nu;
                    nu = 2.0 * nu;
                }
            }
            if !lambda.is_finite() || lambda <= 0.0 {
                return Err(GavinLmError::NumericalBreakdown {
                    stage: GavinLmNumericalStage::DampingParameterUpdate,
                });
            }

            cvg_hst[(self.iteration - 1, 0)] = self.func_calls as f64;
            cvg_hst[(self.iteration - 1, 1)] = x2_val / dof;

            for i in 0..npar {
                cvg_hst[(self.iteration - 1, i + 2)] = p[i];
            }

            if jtw_dy.amax() < eps_grad && self.iteration > 2 {
                termination = GavinLmTermination::GradientTolerance;
                if self.logging_enabled {
                    log::info!(
                        "Gavin LM converged: weighted normal-equation gradient tolerance reached"
                    );
                }
                break;
            }

            let mut max_rel_change = 0.0;
            for i in 0..npar {
                let rel_change = h[i].abs() / (p[i].abs() + 1e-12);
                if rel_change > max_rel_change {
                    max_rel_change = rel_change;
                }
            }

            if max_rel_change < eps_coeff && self.iteration > 2 {
                termination = GavinLmTermination::ParameterStepTolerance;
                if self.logging_enabled {
                    log::info!("Gavin LM converged: relative parameter-step tolerance reached");
                }
                break;
            }

            if self.iteration == max_iter {
                if self.logging_enabled {
                    log::warn!(
                        "Gavin LM reached its iteration limit without satisfying a convergence test"
                    );
                }
            }
        }

        self.last_termination = Some(termination);
        if let Some(stats) = &mut self.telemetry {
            stats.termination = Some(termination);
        }
        let accepted_residual = y_dat - &y_hat;
        let accepted_x2 = accepted_residual
            .component_mul(&weight)
            .dot(&accepted_residual);
        if !accepted_x2.is_finite() {
            return Err(GavinLmError::NumericalBreakdown {
                stage: GavinLmNumericalStage::TrialObjective,
            });
        }

        // The default weights are constructed as a uniform vector. Detect that
        // policy from its source rather than recomputing variance: roundoff in
        // a reduction can make identical floating-point weights appear to
        // have a tiny nonzero variance. Explicit user weights remain absolute.
        // Unlike the Python script, scale from the accepted fit rather than
        // the last rejected trial's local delta_y.
        let final_weight = if !explicit_weighting {
            let residual_norm_sq = accepted_residual.dot(&accepted_residual);
            if !residual_norm_sq.is_finite() || residual_norm_sq <= 0.0 {
                return Err(GavinLmError::NumericalBreakdown {
                    stage: GavinLmNumericalStage::ResidualVarianceEstimate,
                });
            }
            let weight_val = dof / residual_norm_sq;
            DVector::from_element(npnt, weight_val)
        } else {
            weight.clone()
        };

        let red_x2 = accepted_x2 / dof;
        if !red_x2.is_finite() {
            return Err(GavinLmError::NumericalBreakdown {
                stage: GavinLmNumericalStage::ReducedChiSquare,
            });
        }

        // Rebuild the final linearization at the accepted parameter vector so
        // uncertainty estimates do not depend on a stale Broyden approximation.
        let (final_jtw_j, _final_jtw_dy, _final_x2, _final_y_hat, final_j) = self
            .lm_matx_with_output(
                t,
                &p_old,
                &y_old,
                -1.0,
                &j_mat,
                &p,
                y_dat,
                &final_weight,
                &dp,
                Some(&y_hat),
            )?;

        let covar_p = match final_jtw_j.try_inverse() {
            Some(inv) => inv,
            None => return Err(GavinLmError::SingularCovariance),
        };

        // The inverse weighted normal matrix approximates parameter covariance.
        let mut sigma_p = DVector::zeros(npar);
        for i in 0..npar {
            let variance = covar_p[(i, i)];
            if !variance.is_finite() || variance < 0.0 {
                return Err(GavinLmError::InvalidCovarianceDiagonal {
                    index: i,
                    value: variance,
                });
            }
            sigma_p[i] = variance.sqrt();
        }

        let mut sigma_y = DVector::zeros(npnt);
        for i in 0..npnt {
            let mut sum = 0.0;
            for j in 0..npar {
                for k in 0..npar {
                    sum += final_j[(i, j)] * covar_p[(j, k)] * final_j[(i, k)];
                }
            }
            if !sum.is_finite() || sum < 0.0 {
                return Err(GavinLmError::NumericalBreakdown {
                    stage: GavinLmNumericalStage::FittedOutputUncertainty,
                });
            }
            sigma_y[i] = sum.sqrt();
        }

        let mut corr_p = DMatrix::zeros(npar, npar);
        for i in 0..npar {
            for j in 0..npar {
                let scale = sigma_p[i] * sigma_p[j];
                if !scale.is_finite() || scale == 0.0 {
                    return Err(GavinLmError::InvalidCovarianceDiagonal {
                        index: if sigma_p[i] == 0.0 { i } else { j },
                        value: if sigma_p[i] == 0.0 {
                            covar_p[(i, i)]
                        } else {
                            covar_p[(j, j)]
                        },
                    });
                }
                corr_p[(i, j)] = covar_p[(i, j)] / scale;
            }
        }

        let total_weight = weight.sum();
        let r_squared = if total_weight > 0.0 && total_weight.is_finite() {
            let weighted_mean = y_dat.dot(&weight) / total_weight;
            let weighted_total = y_dat
                .iter()
                .zip(weight.iter())
                .map(|(target, weight)| weight * (target - weighted_mean).powi(2))
                .sum::<f64>();
            if weighted_total > 0.0 && weighted_total.is_finite() {
                let residual = y_dat - &y_hat;
                Some(1.0 - residual.component_mul(&weight).dot(&residual) / weighted_total)
            } else {
                None
            }
        } else {
            None
        };

        let cvg_hst_trimmed = cvg_hst.rows(0, self.iteration).into_owned();

        if self.logging_enabled {
            for i in 0..npar {
                let relative_error = if p[i] == 0.0 {
                    f64::INFINITY
                } else {
                    sigma_p[i] / p[i].abs() * 100.0
                };
                log::info!(
                    "Gavin LM parameter p{}={:.6e}, relative standard error={:.3}%",
                    i + 1,
                    p[i],
                    relative_error
                );
            }
        }
        if let (Some(stats), Some(started)) = (self.telemetry.as_mut(), solve_started) {
            stats.total_time = started.elapsed();
            stats.termination = Some(termination);
        }
        Ok(GavinLmResult {
            parameters: p,
            reduced_chi_square: red_x2,
            parameter_standard_errors: sigma_p,
            fitted_value_standard_errors: sigma_y,
            parameter_correlation: corr_p,
            r_squared,
            sum_squared_residuals: accepted_x2,
            convergence_history: cvg_hst_trimmed,
            termination,
        })
    }
}

// Example implementations of ObjectiveFunction trait
pub struct ExponentialSinusoidalModel;

impl ObjectiveFunction for ExponentialSinusoidalModel {
    fn evaluate(
        &self,
        t: &DVector<f64>,
        p: &DVector<f64>,
    ) -> Result<DVector<f64>, GavinLmCallbackError> {
        let mut y_hat = DVector::zeros(t.len());
        for i in 0..t.len() {
            y_hat[i] = p[0] * (-t[i] / p[1]).exp() + p[2] * (t[i] / p[3]).sin();
        }
        Ok(y_hat)
    }

    fn parameter_count(&self) -> usize {
        4
    }
}

pub struct PolynomialModel {
    degree: usize,
}

impl PolynomialModel {
    pub fn new(degree: usize) -> Self {
        Self { degree }
    }
}

impl ObjectiveFunction for PolynomialModel {
    fn evaluate(
        &self,
        t: &DVector<f64>,
        p: &DVector<f64>,
    ) -> Result<DVector<f64>, GavinLmCallbackError> {
        let mut y_hat = DVector::zeros(t.len());
        for i in 0..t.len() {
            let mut sum = 0.0;
            for j in 0..=self.degree {
                sum += p[j] * t[i].powi(j as i32);
            }
            y_hat[i] = sum;
        }
        Ok(y_hat)
    }

    fn parameter_count(&self) -> usize {
        self.degree + 1
    }
}

pub struct GaussianModel;

impl ObjectiveFunction for GaussianModel {
    fn evaluate(
        &self,
        t: &DVector<f64>,
        p: &DVector<f64>,
    ) -> Result<DVector<f64>, GavinLmCallbackError> {
        let mut y_hat = DVector::zeros(t.len());
        for i in 0..t.len() {
            // p[0] = amplitude, p[1] = mean, p[2] = std_dev
            let exponent = -0.5 * ((t[i] - p[1]) / p[2]).powi(2);
            y_hat[i] = p[0] * exponent.exp();
        }
        Ok(y_hat)
    }

    fn parameter_count(&self) -> usize {
        3
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{DVector, dvector};

    struct SquareModel;

    impl ObjectiveFunction for SquareModel {
        fn evaluate(
            &self,
            t: &DVector<f64>,
            p: &DVector<f64>,
        ) -> Result<DVector<f64>, GavinLmCallbackError> {
            Ok(DVector::from_element(t.len(), p[0] * p[0]))
        }

        fn parameter_count(&self) -> usize {
            1
        }
    }

    #[test]
    fn quadratic_update_uses_scaled_step_and_reevaluated_trial_for_acceptance() {
        let mut lm = LevenbergMarquardt::new(SquareModel)
            .with_tolerances(1e-3, 1e-3, 0.6)
            .with_update_policy(GavinLmUpdatePolicy::Quadratic);

        let t = dvector![0.0, 1.0, 2.0];
        let y_dat = dvector![4.0, 4.0, 4.0];
        let (_, _, _, _, _, _, history) = lm
            .lm(dvector![1.0], &t, &y_dat)
            .expect("Quadratic update should finish with a valid covariance");

        // The Python quadratic step scales h by alpha and recomputes X2_try
        // before forming rho. For this fixture rho is below 0.6 on iteration
        // one, then exceeds it after damping rises and the next trial is accepted.
        assert_eq!(history[(0, 2)], 1.0);
        assert!(history[(1, 2)] > 1.0);
    }

    #[test]
    fn nielsen_update_recovers_linear_fit_with_explicit_termination() {
        let t = dvector![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = dvector![1.0, 3.0, 5.0, 7.0, 9.0];
        let mut lm = LevenbergMarquardt::new(PolynomialModel::new(1))
            .with_update_policy(GavinLmUpdatePolicy::Nielsen);
        let result = lm
            .lm_detailed(dvector![0.7, 0.8], &t, &y)
            .expect("Nielsen policy should produce a fit result");

        assert!(result.termination.is_converged());
        assert!((result.parameters[0] - 1.0).abs() < 1e-3);
        assert!((result.parameters[1] - 2.0).abs() < 1e-3);
        assert_eq!(lm.last_termination, Some(result.termination));
    }

    fn linear_jacobian(
        t: DVector<f64>,
    ) -> impl Fn(&DVector<f64>, &DVector<f64>) -> DMatrix<f64> + Clone + 'static {
        move |_, _| DMatrix::from_fn(t.len(), 2, |row, col| if col == 0 { 1.0 } else { t[row] })
    }

    #[test]
    fn covariance_statistics_match_independent_unweighted_linear_oracle() {
        let t = dvector![-2.0, -1.0, 0.0, 1.0, 2.0];
        let noise = dvector![1.0, -2.0, 2.0, -2.0, 1.0];
        let observations = DVector::from_iterator(
            t.len(),
            t.iter()
                .zip(noise.iter())
                .map(|(x, residual)| 4.0 + 3.0 * x + residual),
        );
        let mut solver = LevenbergMarquardt::new(PolynomialModel::new(1))
            .with_analytic_jacobian(linear_jacobian(t.clone()));
        let result = solver
            .lm_detailed(dvector![3.5, 2.5], &t, &observations)
            .expect("full-rank linear fit should produce statistics");

        // The residual is orthogonal to both columns of [1, x], so the exact
        // least-squares solution is (4, 3), SSE=14, and dof=n-p+1=4.
        assert_relative_eq!(result.parameters[0], 4.0, epsilon = 1e-8);
        assert_relative_eq!(result.parameters[1], 3.0, epsilon = 1e-8);
        assert_relative_eq!(
            result.sum_squared_residuals,
            14.0 / observations.dot(&observations),
            epsilon = 1e-12
        );
        let dof = 4.0;
        assert_relative_eq!(
            result.reduced_chi_square,
            14.0 / (observations.dot(&observations) * dof),
            epsilon = 1e-12
        );

        // Cov(p) = (SSE/dof) * inverse(X'X), X'X=diag(5,10).
        assert_relative_eq!(
            result.parameter_standard_errors[0],
            (14.0 / (dof * 5.0_f64)).sqrt(),
            epsilon = 1e-10
        );
        assert_relative_eq!(
            result.parameter_standard_errors[1],
            (14.0 / (dof * 10.0_f64)).sqrt(),
            epsilon = 1e-10
        );
        for (index, x) in t.iter().enumerate() {
            assert_relative_eq!(
                result.fitted_value_standard_errors[index],
                (14.0 / (dof * 5.0) + 14.0 * x * x / (dof * 10.0)).sqrt(),
                epsilon = 1e-10
            );
        }
        assert_relative_eq!(result.parameter_correlation[(0, 0)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(result.parameter_correlation[(1, 1)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(result.parameter_correlation[(0, 1)], 0.0, epsilon = 1e-12);
        assert_relative_eq!(result.parameter_correlation[(1, 0)], 0.0, epsilon = 1e-12);
        assert_relative_eq!(result.r_squared.unwrap(), 45.0 / 52.0, epsilon = 1e-12);
    }

    #[test]
    fn covariance_statistics_match_independent_explicit_weight_oracle() {
        let t = dvector![-2.0, -1.0, 0.0, 1.0, 2.0];
        let weights = dvector![1.0, 2.0, 3.0, 2.0, 1.0];
        let residual = [1.0, -1.0, 2.0 / 3.0, -1.0, 1.0];
        let observations = DVector::from_iterator(
            t.len(),
            t.iter()
                .zip(residual.iter())
                .map(|(x, residual)| 4.0 + 3.0 * x + residual),
        );
        let mut solver = LevenbergMarquardt::new(PolynomialModel::new(1))
            .with_analytic_jacobian(linear_jacobian(t.clone()));
        let result = solver
            .lm_with_weights_detailed(dvector![3.5, 2.5], &t, &observations, &weights)
            .expect("weighted full-rank fit should produce statistics");

        // Weighted residuals are orthogonal to [1, x]. The weighted Gram
        // matrix is diag(9,12), and weighted SSE is 22/3.
        assert_relative_eq!(result.parameters[0], 4.0, epsilon = 1e-8);
        assert_relative_eq!(result.parameters[1], 3.0, epsilon = 1e-8);
        assert_relative_eq!(result.sum_squared_residuals, 22.0 / 3.0, epsilon = 1e-10);
        assert_relative_eq!(result.reduced_chi_square, 11.0 / 6.0, epsilon = 1e-10);
        assert_relative_eq!(
            result.parameter_standard_errors[0],
            1.0 / 3.0,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            result.parameter_standard_errors[1],
            1.0 / 12.0_f64.sqrt(),
            epsilon = 1e-10
        );
        for (index, x) in t.iter().enumerate() {
            assert_relative_eq!(
                result.fitted_value_standard_errors[index],
                (1.0 / 9.0 + x * x / 12.0).sqrt(),
                epsilon = 1e-10
            );
        }
        assert_relative_eq!(result.parameter_correlation[(0, 1)], 0.0, epsilon = 1e-12);
        assert_relative_eq!(result.r_squared.unwrap(), 162.0 / 173.0, epsilon = 1e-12);
    }

    #[test]
    fn rank_deficient_final_jacobian_returns_typed_covariance_error() {
        struct DuplicateColumns;

        impl ObjectiveFunction for DuplicateColumns {
            fn evaluate(
                &self,
                t: &DVector<f64>,
                p: &DVector<f64>,
            ) -> Result<DVector<f64>, GavinLmCallbackError> {
                Ok(DVector::from_element(t.len(), p[0] + p[1]))
            }

            fn parameter_count(&self) -> usize {
                2
            }
        }

        let t = dvector![0.0, 1.0, 2.0, 3.0];
        let target = DVector::from_element(t.len(), 2.0);
        let mut solver = LevenbergMarquardt::new(DuplicateColumns)
            .with_analytic_jacobian(|t: &DVector<f64>, _| DMatrix::from_element(t.len(), 2, 1.0));
        let error = solver
            .lm_detailed(dvector![0.8, 0.8], &t, &target)
            .expect_err("rank-deficient covariance must not be reported as valid");
        assert!(matches!(error, GavinLmError::SingularCovariance));
    }

    #[test]
    fn covariance_after_final_rejection_uses_last_accepted_state() {
        let t = DVector::from_iterator(201, (0..201).map(|index| index as f64 * 0.1));
        let target = ExponentialSinusoidalModel
            .evaluate(&t, &dvector![2.0, 0.5, 1.0, 2.0])
            .unwrap();
        let mut solver = LevenbergMarquardt::new(ExponentialSinusoidalModel)
            .with_telemetry(SolverTelemetryMode::Detailed);
        let result = solver
            .lm_detailed(dvector![1.94, 0.55, 1.0, 1.38], &t, &target)
            .expect("final rejected trial must preserve valid accepted-state statistics");
        let telemetry = solver.telemetry().expect("telemetry enabled");

        assert_eq!(
            result.termination,
            GavinLmTermination::ParameterStepTolerance
        );
        assert_eq!(telemetry.accepted_steps, 1);
        assert_eq!(telemetry.rejected_steps, 7);
        assert_eq!(telemetry.finite_difference_jacobians, 3);

        // Independent NumPy oracle: covariance was formed from the last
        // accepted parameter vector after seven terminal rejected trials.
        assert_relative_eq!(result.parameters[0], 1.2351243514830532, epsilon = 1e-8);
        assert_relative_eq!(result.parameters[1], 1.4209449041658679, epsilon = 1e-8);
        assert_relative_eq!(result.parameters[2], -0.26519741339092007, epsilon = 1e-8);
        assert_relative_eq!(result.parameters[3], 1.3885070236157069, epsilon = 1e-8);
        assert_relative_eq!(
            result.sum_squared_residuals,
            0.7176299191991665,
            epsilon = 1e-8
        );
        assert_relative_eq!(
            result.reduced_chi_square,
            0.7176299191991665 / 198.0,
            epsilon = 1e-10
        );
        assert!(
            (result.parameter_standard_errors[0] - 0.31658562805197854).abs() < 1e-7,
            "parameters={:?}, sse={}, stderr={:?}, corr={:?}",
            result.parameters,
            result.sum_squared_residuals,
            result.parameter_standard_errors,
            result.parameter_correlation,
        );
        assert_relative_eq!(
            result.parameter_standard_errors[1],
            0.55481506077878007,
            epsilon = 1e-7
        );
        assert_relative_eq!(
            result.parameter_standard_errors[2],
            0.06792518979293888,
            epsilon = 1e-7
        );
        assert_relative_eq!(
            result.parameter_standard_errors[3],
            0.041544397391543399,
            epsilon = 1e-7
        );
        assert_relative_eq!(
            result.parameter_correlation[(0, 1)],
            -0.6552129211875944,
            epsilon = 1e-7
        );
        assert_relative_eq!(
            result.r_squared.unwrap(),
            0.20262730362715942,
            epsilon = 1e-7
        );
    }

    #[test]
    fn telemetry_is_opt_in_and_matches_objective_call_counter() {
        let t = dvector![0.0, 1.0, 2.0, 3.0];
        let p_true = dvector![1.0, 2.0];
        let model = LevenbergMarquardt::new(PolynomialModel::new(1));
        let y = model.objective_fn.evaluate(&t, &p_true).unwrap();

        let mut silent = LevenbergMarquardt::new(PolynomialModel::new(1));
        assert!(silent.telemetry().is_none());
        silent
            .lm(dvector![0.8, 1.8], &t, &y)
            .expect("default silent solve should complete");
        assert!(silent.telemetry().is_none());

        let mut measured = LevenbergMarquardt::new(PolynomialModel::new(1))
            .with_telemetry(SolverTelemetryMode::Detailed)
            .with_logging(false);
        measured
            .lm(dvector![0.8, 1.8], &t, &y)
            .expect("instrumented solve should complete");
        let stats = measured.telemetry().expect("telemetry was enabled");
        assert_eq!(stats.objective_evaluations, measured.func_calls);
        assert_eq!(stats.iterations, measured.iteration);
        assert!(stats.finite_difference_jacobians > 0);
        assert!(stats.linear_solves > 0);
        assert_eq!(
            stats.accepted_steps + stats.rejected_steps,
            stats.iterations
        );
        assert!(stats.termination.is_some());
    }

    #[test]
    fn telemetry_modes_separate_gavin_counts_from_clock_reads() {
        let t = dvector![0.0, 1.0, 2.0, 3.0];
        let truth = dvector![1.0, 2.0];
        let y = PolynomialModel::new(1).evaluate(&t, &truth).unwrap();

        let mut counters = LevenbergMarquardt::new(PolynomialModel::new(1))
            .with_telemetry(SolverTelemetryMode::Counters);
        counters.lm(dvector![0.8, 1.8], &t, &y).unwrap();
        let counters = counters.telemetry().unwrap();
        assert_eq!(counters.telemetry_mode, SolverTelemetryMode::Counters);
        assert!(counters.objective_evaluations > 0);
        assert!(counters.linear_solves > 0);
        assert_eq!(counters.objective_time, Duration::ZERO);
        assert_eq!(counters.total_time, Duration::ZERO);

        let mut timings = LevenbergMarquardt::new(PolynomialModel::new(1))
            .with_telemetry(SolverTelemetryMode::Timings);
        timings.lm(dvector![0.8, 1.8], &t, &y).unwrap();
        let timings = timings.telemetry().unwrap();
        assert_eq!(timings.telemetry_mode, SolverTelemetryMode::Timings);
        assert_eq!(timings.objective_evaluations, 0);
        assert_eq!(timings.linear_solves, 0);
        assert!(timings.objective_time > Duration::ZERO);
        assert!(timings.total_time > Duration::ZERO);
    }

    #[test]
    fn analytic_jacobian_path_matches_fd_fit_without_fd_or_broyden_work() {
        let t = dvector![0.0, 1.0, 2.0, 3.0, 4.0];
        let p_true = dvector![1.0, 2.0];
        let y = PolynomialModel::new(1).evaluate(&t, &p_true).unwrap();
        let jacobian_t = t.clone();
        let mut exact = LevenbergMarquardt::new(PolynomialModel::new(1))
            .with_telemetry(SolverTelemetryMode::Detailed)
            .with_analytic_jacobian(move |_, _| {
                DMatrix::from_fn(jacobian_t.len(), 2, |row, col| {
                    if col == 0 { 1.0 } else { jacobian_t[row] }
                })
            });
        let (exact_p, exact_red_chi, ..) = exact
            .lm(dvector![0.8, 1.8], &t, &y)
            .expect("exact-Jacobian fit should complete");

        let mut finite_difference = LevenbergMarquardt::new(PolynomialModel::new(1));
        let (fd_p, fd_red_chi, ..) = finite_difference
            .lm(dvector![0.8, 1.8], &t, &y)
            .expect("finite-difference fit should complete");

        for index in 0..2 {
            assert_relative_eq!(exact_p[index], fd_p[index], epsilon = 1e-8);
        }
        assert_relative_eq!(exact_red_chi, fd_red_chi, epsilon = 1e-12);
        let telemetry = exact.telemetry().expect("telemetry enabled");
        assert_eq!(
            telemetry.jacobian_source,
            Some(GavinLmJacobianSource::Analytic)
        );
        assert!(telemetry.analytic_jacobian_evaluations > 0);
        assert_eq!(telemetry.finite_difference_jacobians, 0);
        assert_eq!(telemetry.broyden_updates, 0);
    }

    #[test]
    fn prepared_symbolic_objective_uses_shared_residual_and_jacobian() {
        use crate::numerical::Nonlinear_systems::least_squares::PreparedSymbolicLeastSquaresProblem;
        use crate::numerical::Nonlinear_systems::symbolic::SymbolicProblemOptions;

        let options = SymbolicProblemOptions::new()
            .with_variables(vec!["a".to_string(), "b".to_string()])
            .with_parameter_names(vec!["x".to_string()])
            .with_parameter_values(dvector![0.0])
            .with_lambdify_backend();
        let prepared =
            PreparedSymbolicLeastSquaresProblem::from_strings(vec!["a + b*x".to_string()], options)
                .expect("symbolic model should prepare once");
        let bound = prepared
            .bind_initial_with_guess(dvector![0.8, 1.8])
            .expect("symbolic model should bind");
        let observations = [1.0, 3.0, 5.0, 7.0, 9.0];
        let model = SymbolicGavinObjective::new(&bound, &observations)
            .expect("scalar formula with one sample input should adapt");
        let t = dvector![0.0, 1.0, 2.0, 3.0, 4.0];
        let mut solver =
            LevenbergMarquardt::new(model).with_telemetry(SolverTelemetryMode::Detailed);
        let (parameters, reduced_chi_square, ..) = solver
            .lm_residuals(dvector![0.8, 1.8], &t)
            .expect("Gavin controller should use the prepared symbolic model");

        assert_relative_eq!(parameters[0], 1.0, epsilon = 1e-7);
        assert_relative_eq!(parameters[1], 2.0, epsilon = 1e-7);
        assert!(reduced_chi_square < 1e-12);
        let telemetry = solver.telemetry().expect("telemetry enabled");
        assert_eq!(
            telemetry.jacobian_source,
            Some(GavinLmJacobianSource::Analytic)
        );
        assert!(telemetry.analytic_jacobian_evaluations > 0);
        assert_eq!(telemetry.finite_difference_jacobians, 0);
        assert_eq!(telemetry.broyden_updates, 0);
    }

    #[test]
    fn symbolic_callback_error_keeps_nonlinear_solve_error_as_source() {
        use crate::numerical::Nonlinear_systems::least_squares::PreparedSymbolicLeastSquaresProblem;
        use crate::numerical::Nonlinear_systems::symbolic::SymbolicProblemOptions;

        let prepared = PreparedSymbolicLeastSquaresProblem::from_strings(
            vec!["ln(a) + x".to_string()],
            SymbolicProblemOptions::new()
                .with_variables(vec!["a".to_string()])
                .with_parameter_names(vec!["x".to_string()])
                .with_parameter_values(dvector![0.0])
                .with_lambdify_backend(),
        )
        .expect("symbolic model should prepare");
        let bound = prepared
            .bind_initial_with_guess(dvector![1.0])
            .expect("symbolic model should bind");
        let observations = [1.0];
        let model = SymbolicGavinObjective::new(&bound, &observations).unwrap();
        let mut solver = LevenbergMarquardt::new(model);
        let error = solver
            .lm_residuals(dvector![-1.0], &dvector![0.0])
            .expect_err("invalid logarithm domain must surface as callback error");
        let callback_error =
            std::error::Error::source(&error).expect("typed callback wrapper retains its source");
        assert!(
            callback_error
                .downcast_ref::<crate::numerical::Nonlinear_systems::error::SolveError>()
                .is_some()
        );
    }

    #[test]
    fn symbolic_positive_domain_is_checked_before_callback_evaluation() {
        use crate::numerical::Nonlinear_systems::least_squares::PreparedSymbolicLeastSquaresProblem;
        use crate::numerical::Nonlinear_systems::symbolic::SymbolicProblemOptions;

        let mut prepared = PreparedSymbolicLeastSquaresProblem::from_strings(
            vec!["ln(a) + x".to_string()],
            SymbolicProblemOptions::new()
                .with_variables(vec!["a".to_string()])
                .with_parameter_names(vec!["x".to_string()])
                .with_parameter_values(dvector![0.0])
                .with_lambdify_backend(),
        )
        .expect("symbolic model should prepare");
        prepared
            .set_positive(&["a"])
            .expect("declared positivity constraint should resolve");
        let bound = prepared
            .bind_initial_with_guess(dvector![1.0])
            .expect("feasible model should bind");
        let observations = [0.0];
        let objective = SymbolicGavinObjective::new(&bound, &observations).unwrap();
        let mut solver =
            LevenbergMarquardt::new(objective).with_telemetry(SolverTelemetryMode::Detailed);
        let error = solver
            .lm_residuals(dvector![-1.0], &dvector![0.0])
            .expect_err("infeasible trial must be rejected before symbolic evaluation");

        assert!(matches!(error, GavinLmError::InfeasibleTrialPoint { .. }));
        assert_eq!(
            solver
                .telemetry()
                .expect("telemetry enabled")
                .objective_evaluations,
            0
        );
    }

    #[test]
    fn prepared_symbolic_gavin_fit_matches_sym_fitting_reference() {
        use crate::numerical::Nonlinear_systems::least_squares::PreparedSymbolicLeastSquaresProblem;
        use crate::numerical::Nonlinear_systems::symbolic::SymbolicProblemOptions;
        use crate::numerical::optimization::sym_fitting::Fitting;

        let x_values = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let y_values = x_values.iter().map(|x| 1.25 + 2.75 * x).collect::<Vec<_>>();
        let fitting = Fitting::new()
            .with_data(x_values.clone(), y_values.clone())
            .with_equation_str("a + b*x".to_string())
            .with_arg("x".to_string())
            .with_unknowns(vec!["a".to_string(), "b".to_string()])
            .with_initial_guess(vec![1.0, 2.0])
            .build();
        let reference = fitting
            .map_of_solutions
            .expect("sym_fitting builder should produce the reference fit");

        let options = SymbolicProblemOptions::new()
            .with_variables(vec!["a".to_string(), "b".to_string()])
            .with_parameter_names(vec!["x".to_string()])
            .with_parameter_values(dvector![0.0])
            .with_lambdify_backend();
        let prepared =
            PreparedSymbolicLeastSquaresProblem::from_strings(vec!["a + b*x".to_string()], options)
                .expect("shared symbolic preparation should succeed");
        let bound = prepared
            .bind_initial_with_guess(dvector![1.0, 2.0])
            .expect("prepared symbolic problem should bind once");
        let objective = SymbolicGavinObjective::new(&bound, &y_values)
            .expect("prepared scalar model should adapt to sampled residuals");
        let mut gavin = LevenbergMarquardt::new(objective);
        let (parameters, ..) = gavin
            .lm_residuals(dvector![1.0, 2.0], &DVector::from_vec(x_values))
            .expect("Gavin LM should solve the same symbolic fitting problem");

        assert_relative_eq!(parameters[0], reference["a"], epsilon = 1e-7);
        assert_relative_eq!(parameters[1], reference["b"], epsilon = 1e-7);
    }

    #[test]
    fn analytic_jacobian_rejects_wrong_shape() {
        let t = dvector![0.0, 1.0, 2.0];
        let y = dvector![1.0, 3.0, 5.0];
        let mut solver = LevenbergMarquardt::new(PolynomialModel::new(1))
            .with_analytic_jacobian(|_, _| DMatrix::zeros(1, 1));
        let error = solver.lm(dvector![0.8, 1.8], &t, &y).unwrap_err();
        assert!(matches!(
            error,
            GavinLmError::AnalyticJacobianShapeMismatch { .. }
        ));
    }

    #[test]
    fn fallible_analytic_jacobian_preserves_callback_failure_context() {
        let t = dvector![0.0, 1.0, 2.0];
        let y = dvector![1.0, 3.0, 5.0];
        let mut solver = LevenbergMarquardt::new(PolynomialModel::new(1))
            .with_fallible_analytic_jacobian(|_, _| {
                Err::<DMatrix<f64>, _>(std::io::Error::other("prepared evaluator failed"))
            });
        let error = solver.lm(dvector![0.8, 1.8], &t, &y).unwrap_err();
        assert!(matches!(&error, GavinLmError::AnalyticJacobianCallback(_)));
        assert!(std::error::Error::source(&error).is_some());
    }

    #[test]
    fn rejects_non_finite_inputs_and_invalid_solver_options() {
        let t = dvector![0.0, 1.0, 2.0];
        let y = dvector![1.0, 2.0, 3.0];
        let mut lm = LevenbergMarquardt::new(PolynomialModel::new(1));
        assert!(matches!(
            lm.lm(dvector![f64::NAN, 1.0], &t, &y),
            Err(GavinLmError::NonFiniteInput { .. })
        ));

        lm.eps_grad = f64::NAN;
        assert!(matches!(
            lm.lm(dvector![1.0, 1.0], &t, &y),
            Err(GavinLmError::InvalidOption {
                option: GavinLmOption::GradientTolerance,
                ..
            })
        ));
    }

    #[test]
    fn test_exponential_sinusoidal_model() {
        let model = ExponentialSinusoidalModel;
        let mut lm = LevenbergMarquardt::new(model);

        let t = DVector::from_vec((0..10).map(|i| i as f64 * 0.1).collect());
        let p_true = dvector![2.0, 0.5, 1.0, 2.0];
        let y_true = lm.objective_fn.evaluate(&t, &p_true).unwrap();

        let p_initial = dvector![1.5, 0.4, 0.8, 1.3];

        match lm.lm(p_initial, &t, &y_true) {
            Ok((_p_fitted, red_x2, _sigma_p, _sigma_y, _corr_p, _r_sq, _cvg_hst)) => {
                assert!(red_x2 < 1e-4);
            }
            Err(e) => panic!("LM fitting failed: {}", e),
        }
    }

    #[test]
    fn challenging_fits_match_gavin_python_reference() {
        // Golden values are emitted by optimization/gavin_reference's Python
        // example using the same model, samples, data, and default controller.
        let exact_parameters = dvector![
            1.998465202538,
            0.500449967249,
            0.999312764688,
            1.999967485164
        ];
        let noisy_parameters = dvector![
            1.998760613259,
            0.501073397130,
            0.999124294596,
            1.999999693341
        ];
        let nonuniform_parameters = dvector![
            1.999498343880,
            0.500247781856,
            0.999639175801,
            1.999979622694
        ];
        let broad_t = DVector::from_iterator(201, (0..201).map(|i| i as f64 * 0.1));
        let noisy_observations = {
            let clean = ExponentialSinusoidalModel
                .evaluate(&broad_t, &dvector![2.0, 0.5, 1.0, 2.0])
                .unwrap();
            DVector::from_iterator(
                broad_t.len(),
                clean.iter().enumerate().map(|(index, value)| {
                    let t = broad_t[index];
                    value + 0.001 * (1.7 * t).sin() + 0.0005 * (2.3 * t).cos()
                }),
            )
        };
        let nonuniform_t = DVector::from_iterator(
            81,
            (0..81).map(|index| 20.0 * (index as f64 / 80.0).powf(1.25)),
        );

        let cases = [
            (
                "broad_exact",
                broad_t.clone(),
                ExponentialSinusoidalModel
                    .evaluate(&broad_t, &dvector![2.0, 0.5, 1.0, 2.0])
                    .unwrap(),
                exact_parameters,
                2.153794225532e-9,
            ),
            (
                "broad_noisy",
                broad_t,
                noisy_observations,
                noisy_parameters,
                7.950234822553e-9,
            ),
            (
                "nonuniform_exact",
                nonuniform_t.clone(),
                ExponentialSinusoidalModel
                    .evaluate(&nonuniform_t, &dvector![2.0, 0.5, 1.0, 2.0])
                    .unwrap(),
                nonuniform_parameters,
                1.253274629481e-9,
            ),
        ];

        for (name, t, observations, python_parameters, python_reduced_chi_square) in cases {
            let mut solver = LevenbergMarquardt::new(ExponentialSinusoidalModel);
            let result = solver
                .lm_detailed(dvector![1.9, 0.55, 0.9, 1.9], &t, &observations)
                .unwrap_or_else(|error| panic!("{name}: Gavin fit failed: {error}"));

            assert_eq!(
                result.termination,
                GavinLmTermination::GradientTolerance,
                "{name}: Python reference termination"
            );
            assert_eq!(result.convergence_history.nrows(), 5, "{name}: iterations");
            assert_eq!(solver.func_calls, 10, "{name}: cached objective calls");
            for (actual, expected) in result.parameters.iter().zip(python_parameters.iter()) {
                assert_relative_eq!(actual, expected, epsilon = 5e-10, max_relative = 2e-7);
            }
            assert_relative_eq!(
                result.reduced_chi_square,
                python_reduced_chi_square,
                epsilon = 1e-12,
                max_relative = 5e-2
            );
        }
    }

    #[test]
    fn test_polynomial_model() {
        let model = PolynomialModel::new(2); // Quadratic
        let mut lm = LevenbergMarquardt::new(model);

        let t = DVector::from_vec((0..10).map(|i| i as f64).collect());
        let p_true = dvector![1.0, 2.0, 0.5]; // 1 + 2x + 0.5x^2
        let y_true = lm.objective_fn.evaluate(&t, &p_true).unwrap();

        let p_initial = dvector![0.8, 1.8, 0.4];

        match lm.lm(p_initial, &t, &y_true) {
            Ok((_p_fitted, red_x2, _sigma_p, _sigma_y, _corr_p, _r_sq, _cvg_hst)) => {
                assert!(red_x2 < 1e-10);
            }
            Err(e) => panic!("Polynomial LM fitting failed: {}", e),
        }
    }

    #[test]
    fn test_gaussian_model() {
        let model = GaussianModel;
        let mut lm = LevenbergMarquardt::new(model);

        let t = DVector::from_vec((-50..50).map(|i| i as f64 * 0.1).collect());
        let p_true = dvector![2.0, 0.0, 1.0]; // amplitude=2, mean=0, std=1
        let y_true = lm.objective_fn.evaluate(&t, &p_true).unwrap();

        let p_initial = dvector![1.8, 0.1, 0.9];

        match lm.lm(p_initial, &t, &y_true) {
            Ok((_p_fitted, red_x2, _sigma_p, _sigma_y, _corr_p, _r_sq, _cvg_hst)) => {
                assert!(red_x2 < 1e-4);
            }
            Err(e) => panic!("Gaussian LM fitting failed: {}", e),
        }
    }

    #[test]
    fn test_custom_closure_model() {
        // Example of how to use a closure as an objective function
        struct ClosureModel<F>
        where
            F: Fn(&DVector<f64>, &DVector<f64>) -> DVector<f64>,
        {
            func: F,
            param_count: usize,
        }

        impl<F> ObjectiveFunction for ClosureModel<F>
        where
            F: Fn(&DVector<f64>, &DVector<f64>) -> DVector<f64>,
        {
            fn evaluate(
                &self,
                t: &DVector<f64>,
                p: &DVector<f64>,
            ) -> Result<DVector<f64>, GavinLmCallbackError> {
                Ok((self.func)(t, p))
            }

            fn parameter_count(&self) -> usize {
                self.param_count
            }
        }

        // Custom exponential decay model
        let custom_func = |t: &DVector<f64>, p: &DVector<f64>| -> DVector<f64> {
            let mut y = DVector::zeros(t.len());
            for i in 0..t.len() {
                y[i] = p[0] * (-p[1] * t[i]).exp();
            }
            y
        };

        let model = ClosureModel {
            func: custom_func,
            param_count: 2,
        };

        let mut lm = LevenbergMarquardt::new(model);

        let t = DVector::from_vec((0..20).map(|i| i as f64 * 0.1).collect());
        let p_true = dvector![3.0, 0.5];
        let y_true = lm.objective_fn.evaluate(&t, &p_true).unwrap();

        let p_initial = dvector![2.5, 0.4];

        match lm.lm(p_initial, &t, &y_true) {
            Ok((_p_fitted, red_x2, _sigma_p, _sigma_y, _corr_p, _r_sq, _cvg_hst)) => {
                assert!(red_x2 < 1e-5);
            }
            Err(e) => panic!("Custom model LM fitting failed: {}", e),
        }
    }
}

#[cfg(test)]
mod additional_tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{DVector, dvector};

    // Test for noisy data fitting
    #[test]
    fn test_exponential_decay_with_noise() {
        struct ExponentialDecay;

        impl ObjectiveFunction for ExponentialDecay {
            fn evaluate(
                &self,
                t: &DVector<f64>,
                p: &DVector<f64>,
            ) -> Result<DVector<f64>, GavinLmCallbackError> {
                let mut y = DVector::zeros(t.len());
                for i in 0..t.len() {
                    y[i] = p[0] * (-p[1] * t[i]).exp();
                }
                Ok(y)
            }

            fn parameter_count(&self) -> usize {
                2
            }
        }

        let model = ExponentialDecay;
        let mut lm = LevenbergMarquardt::new(model);

        // Generate synthetic noisy data
        let t = DVector::from_vec((0..20).map(|i| i as f64 * 0.2).collect());
        let p_true = dvector![5.0, 0.8];
        let y_clean = lm.objective_fn.evaluate(&t, &p_true).unwrap();

        // Add small amount of noise
        let noise = dvector![
            0.01, -0.02, 0.015, -0.01, 0.008, -0.012, 0.005, 0.018, -0.007, 0.011, 0.003, -0.009,
            0.014, -0.006, 0.002, 0.016, -0.004, 0.013, -0.008, 0.001
        ];
        let y_noisy = y_clean + noise;

        let p_initial = dvector![4.0, 0.6];

        match lm.lm(p_initial, &t, &y_noisy) {
            Ok((p_fitted, red_x2, sigma_p, _sigma_y, _corr_p, _r_sq, _cvg_hst)) => {
                // Should recover true parameters within reasonable tolerance
                assert_relative_eq!(p_fitted[0], p_true[0], epsilon = 0.1);
                assert_relative_eq!(p_fitted[1], p_true[1], epsilon = 0.1);
                assert!(red_x2 > 0.0);
                assert!(sigma_p[0] > 0.0 && sigma_p[1] > 0.0);
            }
            Err(e) => panic!("Noisy data fitting failed: {}", e),
        }
    }

    // Test for multi-parameter complex model
    #[test]
    fn test_complex_multi_parameter_model() {
        struct ComplexModel;

        impl ObjectiveFunction for ComplexModel {
            fn evaluate(
                &self,
                t: &DVector<f64>,
                p: &DVector<f64>,
            ) -> Result<DVector<f64>, GavinLmCallbackError> {
                let mut y = DVector::zeros(t.len());
                for i in 0..t.len() {
                    // y = p[0] * exp(-p[1]*t) + p[2] * sin(p[3]*t + p[4]) + p[5]
                    y[i] = p[0] * (-p[1] * t[i]).exp() + p[2] * (p[3] * t[i] + p[4]).sin() + p[5];
                }
                Ok(y)
            }

            fn parameter_count(&self) -> usize {
                6
            }
        }

        let model = ComplexModel;
        let mut lm = LevenbergMarquardt::new(model);

        let t = DVector::from_vec((0..5000).map(|i| i as f64 * 0.001).collect());
        let p_true = dvector![3.0, 0.5, 2.0, 1.2, 0.3, 1.0];
        let y_true = lm.objective_fn.evaluate(&t, &p_true).unwrap();

        let p_initial = dvector![2.5, 0.4, 1.8, 1.0, 0.2, 0.8];

        match lm.lm(p_initial, &t, &y_true) {
            Ok((p_fitted, red_x2, _sigma_p, _sigma_y, corr_p, _r_sq, cvg_hst)) => {
                // Check convergence
                assert!(red_x2 < 1e-3);
                assert!(cvg_hst.nrows() > 0);
                assert!(lm.func_calls > 0);

                // Check parameter recovery
                for i in 0..6 {
                    assert_relative_eq!(p_fitted[i], p_true[i], epsilon = 1e-2);
                }

                // Check correlation matrix properties
                assert_eq!(corr_p.nrows(), 6);
                assert_eq!(corr_p.ncols(), 6);
                for i in 0..6 {
                    assert_relative_eq!(corr_p[(i, i)], 1.0, epsilon = 1e-3);
                }
            }
            Err(e) => panic!("Complex model fitting failed: {}", e),
        }
    }

    // Test error handling for mismatched dimensions
    #[test]
    fn test_dimension_mismatch_error() {
        let model = ExponentialSinusoidalModel;
        let mut lm = LevenbergMarquardt::new(model);

        let t = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let y_dat = DVector::from_vec(vec![1.0, 2.0]); // Different length
        let p_initial = dvector![1.0, 1.0, 1.0, 1.0];

        let error = lm.lm(p_initial, &t, &y_dat).unwrap_err();
        assert!(matches!(
            error,
            GavinLmError::ModelTargetLengthMismatch { .. }
        ));
    }

    #[test]
    fn prepared_multi_equation_gavin_fit_matches_sym_fitting_reference() {
        use crate::numerical::Nonlinear_systems::least_squares::PreparedSymbolicLeastSquaresProblem;
        use crate::numerical::Nonlinear_systems::symbolic::SymbolicProblemOptions;
        use crate::numerical::optimization::sym_fitting::Fitting;

        let x_values = vec![-1.0, 0.0, 1.0, 2.0, 3.0];
        let first = x_values.iter().map(|x| 1.5 + 2.0 * x).collect::<Vec<_>>();
        let second = x_values
            .iter()
            .map(|x| -0.25 + 0.75 * x * x)
            .collect::<Vec<_>>();
        let mut observations = first;
        observations.extend(second);
        let equations = vec!["a + b*x".to_string(), "c + d*x^2".to_string()];
        let unknowns = vec!["a", "b", "c", "d"];
        let reference = Fitting::new()
            .with_data(x_values.clone(), observations.clone())
            .with_equations_str(equations.clone())
            .with_arg("x".to_string())
            .with_unknowns(unknowns.iter().map(|name| (*name).to_string()).collect())
            .with_initial_guess(vec![1.0, 1.0, -0.1, 1.0])
            .build()
            .map_of_solutions
            .expect("sym_fitting should solve the matched multi-equation task");

        let prepared = PreparedSymbolicLeastSquaresProblem::from_strings(
            equations,
            SymbolicProblemOptions::new()
                .with_variables(unknowns.iter().map(|name| (*name).to_string()).collect())
                .with_parameter_names(vec!["x".to_string()])
                .with_parameter_values(dvector![0.0])
                .with_lambdify_backend(),
        )
        .expect("shared symbolic preparation should succeed");
        let bound = prepared
            .bind_initial_with_guess(dvector![1.0, 1.0, -0.1, 1.0])
            .expect("prepared symbolic problem should bind");
        let objective = SymbolicGavinObjective::new(&bound, &observations)
            .expect("multi-equation symbolic model should adapt");
        let mut gavin = LevenbergMarquardt::new(objective);
        let (parameters, ..) = gavin
            .lm_residuals(dvector![1.0, 1.0, -0.1, 1.0], &DVector::from_vec(x_values))
            .expect("Gavin LM should solve the matched multi-equation task");

        for (index, name) in unknowns.iter().enumerate() {
            assert_relative_eq!(parameters[index], reference[*name], epsilon = 1e-6);
        }
    }

    // Test convergence behavior with poor initial guess
    #[test]
    fn test_poor_initial_guess_convergence() {
        let model = GaussianModel;
        let mut lm = LevenbergMarquardt::new(model);
        let t = DVector::from_vec((-300..300).map(|i| i as f64 * 0.02).collect());
        let p_true = dvector![5.0, 2.0, 1.5];
        let y_true = lm.objective_fn.evaluate(&t, &p_true).unwrap();

        // Very poor initial guess
        let p_initial = dvector![4.0, 3.0, 2.1];

        match lm.lm(p_initial, &t, &y_true) {
            Ok((p_fitted, red_x2, _sigma_p, _sigma_y, _corr_p, _r_sq, _cvg_hst)) => {
                // Should still converge, though may take more iterations
                assert!(red_x2 < 1e-3);

                // Parameters should be recovered
                assert_relative_eq!(p_fitted[0], p_true[0], epsilon = 1e-2);
                assert_relative_eq!(p_fitted[1], p_true[1], epsilon = 1e-2);
                assert_relative_eq!(p_fitted[2], p_true[2], epsilon = 1e-2);
            }
            Err(error) => panic!("difficult-but-identifiable Gaussian fit failed: {error}"),
        }
    }

    // Test Jacobian computation methods
    #[test]
    fn test_jacobian_computation() {
        let model = PolynomialModel::new(2);
        let mut lm = LevenbergMarquardt::new(model);

        let t = dvector![0.0, 1.0, 2.0, 3.0];
        let p = dvector![1.0, 2.0, 0.5];
        let y = lm.objective_fn.evaluate(&t, &p).unwrap();
        let dp = dvector![-0.001, -0.001, -0.001];

        // Test finite difference Jacobian
        let j_fd = lm
            .lm_fd_j(&t, &p, &y, &dp)
            .expect("finite-difference Jacobian should be evaluable");

        // Check Jacobian dimensions
        assert_eq!(j_fd.nrows(), t.len());
        assert_eq!(j_fd.ncols(), p.len());

        // For polynomial y = p[0] + p[1]*t + p[2]*t^2
        // dy/dp[0] = 1, dy/dp[1] = t, dy/dp[2] = t^2
        for i in 0..t.len() {
            assert_relative_eq!(j_fd[(i, 0)], 1.0, epsilon = 1e-6);
            assert_relative_eq!(j_fd[(i, 1)], t[i], epsilon = 1e-6);
            assert_relative_eq!(j_fd[(i, 2)], t[i] * t[i], epsilon = 1e-6);
        }
    }

    // Test Broyden update
    #[test]
    fn test_broyden_jacobian_update() {
        let model = PolynomialModel::new(1); // Linear model
        let lm = LevenbergMarquardt::new(model);

        let p_old = dvector![1.0, 2.0];
        let p_new = dvector![1.1, 2.1];
        let y_old = dvector![3.0, 5.0, 7.0];
        let y_new = dvector![3.2, 5.2, 7.2];

        // Initial Jacobian for linear model y = p[0] + p[1]*t
        let j_old = nalgebra::dmatrix![
            1.0, 0.0;
            1.0, 1.0;
            1.0, 2.0;
        ];

        let j_broyden = lm.lm_broyden_j(&p_old, &y_old, &j_old, &p_new, &y_new);

        // Check dimensions
        assert_eq!(j_broyden.nrows(), j_old.nrows());
        assert_eq!(j_broyden.ncols(), j_old.ncols());

        // For a linear model, Broyden update should preserve the Jacobian structure
        assert!(j_broyden.norm() > 0.0);
    }

    #[test]
    fn broyden_zero_step_preserves_finite_jacobian() {
        let solver = LevenbergMarquardt::new(PolynomialModel::new(1));
        let p = dvector![1.0, 2.0];
        let y = dvector![3.0, 5.0, 7.0];
        let jacobian = nalgebra::dmatrix![
            1.0, 0.0;
            1.0, 1.0;
            1.0, 2.0;
        ];

        let updated = solver.lm_broyden_j(&p, &y, &jacobian, &p, &y);
        assert_eq!(updated, jacobian);
        assert!(updated.iter().all(|value| value.is_finite()));
    }

    // Test with singular/ill-conditioned cases
    #[test]
    fn test_ill_conditioned_problem() {
        struct IllConditionedModel;

        impl ObjectiveFunction for IllConditionedModel {
            fn evaluate(
                &self,
                t: &DVector<f64>,
                p: &DVector<f64>,
            ) -> Result<DVector<f64>, GavinLmCallbackError> {
                let mut y = DVector::zeros(t.len());
                for i in 0..t.len() {
                    // Highly correlated parameters: y = p[0] + p[1] + (p[0] - p[1]) * t
                    y[i] = p[0] + p[1] + (p[0] - p[1]) * t[i];
                }
                Ok(y)
            }

            fn parameter_count(&self) -> usize {
                2
            }
        }

        let model = IllConditionedModel;
        let mut lm = LevenbergMarquardt::new(model);

        let t = dvector![0.0, 1.0];
        let p_true = dvector![2.0, 3.0];
        let y_true = lm.objective_fn.evaluate(&t, &p_true).unwrap();

        let p_initial = dvector![1.0, 1.0];

        match lm.lm(p_initial, &t, &y_true) {
            Ok((p_fitted, red_x2, _sigma_p, _sigma_y, _corr_p, _r_sq, _cvg_hst)) => {
                // Should converge but with high parameter uncertainties
                assert!(red_x2 < 1e-10);

                // Check that we get a valid solution (may not be unique)
                let y_fitted = lm.objective_fn.evaluate(&t, &p_fitted).unwrap();
                for i in 0..y_true.len() {
                    assert_relative_eq!(y_fitted[i], y_true[i], epsilon = 1e-5);
                }
            }
            Err(error) => panic!("identifiable ill-conditioned fit failed: {error}"),
        }
    }
}
