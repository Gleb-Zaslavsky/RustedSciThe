use crate::numerical::Nonlinear_systems::error::{SolveError, TerminationReason};
use crate::numerical::Nonlinear_systems::problem::{Bounds, JacobianProvider};
use log::{debug, info, warn};
use nalgebra::{DMatrix, DVector};
use std::time::{Duration, Instant};

/// Linear solver used inside Newton-type methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearSolverKind {
    /// LU factorization.
    Lu,
    /// Explicit inverse. Kept mostly for compatibility.
    Inverse,
}

impl Default for LinearSolverKind {
    fn default() -> Self {
        Self::Lu
    }
}

/// Verbosity level for optional engine logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EngineLogLevel {
    /// Detailed per-iteration logs.
    Debug,
    /// High-level progress logs.
    #[default]
    Info,
    /// Warning-only logs.
    Warn,
}

/// States whether aggregate solve statistics were actually collected.
///
/// The numeric fields in [`SolveStatistics`] intentionally remain zero when
/// collection is disabled for compatibility. Callers must inspect this flag
/// before interpreting those zeros as measurements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StatisticsAvailability {
    /// Statistics were not collected for this solve attempt.
    #[default]
    NotCollected,
    /// Statistics and durations were collected for this solve attempt.
    Collected,
}

impl StatisticsAvailability {
    /// Returns `true` when the statistics fields contain measurements.
    pub fn is_collected(self) -> bool {
        matches!(self, Self::Collected)
    }
}

/// Optional runtime features of the engine.
#[derive(Debug, Clone)]
pub struct DiagnosticsOptions {
    /// Stores per-iteration history in the result.
    pub collect_history: bool,
    /// Collects aggregate counters.
    pub collect_statistics: bool,
    /// Enables `log` output.
    pub enable_logging: bool,
    /// Logging level when logging is enabled.
    pub log_level: EngineLogLevel,
    /// Collects simple memory estimates for the final state.
    pub enable_memory_diagnostics: bool,
}

impl Default for DiagnosticsOptions {
    fn default() -> Self {
        Self {
            collect_history: true,
            collect_statistics: true,
            enable_logging: false,
            log_level: EngineLogLevel::Info,
            enable_memory_diagnostics: false,
        }
    }
}

/// Common options shared by all nonlinear methods.
#[derive(Debug, Clone)]
pub struct SolveOptions {
    /// Residual and small-step tolerance.
    pub tolerance: f64,
    /// Maximum number of nonlinear iterations.
    pub max_iterations: usize,
    /// Linear solver backend.
    pub linear_solver: LinearSolverKind,
    /// Optional box bounds.
    pub bounds: Option<Bounds>,
    /// Optional runtime diagnostics.
    pub diagnostics: DiagnosticsOptions,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-6,
            max_iterations: 100,
            linear_solver: LinearSolverKind::Lu,
            bounds: None,
            diagnostics: DiagnosticsOptions::default(),
        }
    }
}

impl SolveOptions {
    /// Checks that the generic solver options are consistent.
    pub fn validate(&self, dimension: usize) -> Result<(), SolveError> {
        if self.tolerance <= 0.0 {
            return Err(SolveError::InvalidConfig(
                "tolerance must be positive".to_string(),
            ));
        }
        if self.max_iterations == 0 {
            return Err(SolveError::InvalidConfig(
                "max_iterations must be greater than zero".to_string(),
            ));
        }
        if let Some(bounds) = &self.bounds {
            if bounds.len() != dimension {
                return Err(SolveError::DimensionMismatch {
                    expected: dimension,
                    actual: bounds.len(),
                    context: "solve options bounds",
                });
            }
        }
        Ok(())
    }
}

/// One history item collected by the engine.
#[derive(Debug, Clone)]
pub struct IterationRecord {
    /// Zero-based iteration index.
    pub iteration: usize,
    /// Residual norm after this iteration.
    pub residual_norm: f64,
    /// Norm of the trial step.
    pub step_norm: f64,
    /// `true` when the step was accepted.
    pub accepted: bool,
}

/// Aggregate counters for one solve.
#[derive(Debug, Clone, Default)]
pub struct SolveStatistics {
    /// Whether the remaining counter and duration fields are measurements.
    pub availability: StatisticsAvailability,
    /// Number of completed nonlinear iterations.
    pub iterations: usize,
    /// Residual evaluations.
    pub residual_evaluations: usize,
    /// Residual evaluations used to establish the initial or accepted current
    /// iterate. This is a subset of [`Self::residual_evaluations`].
    pub state_residual_evaluations: usize,
    /// Residual evaluations performed for trial points inside method steps.
    pub trial_residual_evaluations: usize,
    /// Jacobian evaluations.
    pub jacobian_evaluations: usize,
    /// Jacobian evaluations used to establish the initial or accepted current
    /// iterate. This is a subset of [`Self::jacobian_evaluations`].
    pub state_jacobian_evaluations: usize,
    /// Jacobian evaluations performed for trial points inside method steps.
    pub trial_jacobian_evaluations: usize,
    /// Linear solves performed by the method.
    pub linear_solves: usize,
    /// Linear factorizations constructed by the method.
    pub linear_factorizations: usize,
    /// Accepted trial steps.
    pub accepted_steps: usize,
    /// Rejected trial steps.
    pub rejected_steps: usize,
    /// Trial points written into a reusable method workspace.
    pub reusable_trial_points: usize,
    /// Cumulative time spent evaluating residuals.
    pub residual_duration: Duration,
    /// Cumulative time spent evaluating Jacobians.
    pub jacobian_duration: Duration,
    /// Cumulative time spent in linear step/subproblem solves.
    pub linear_solve_duration: Duration,
    /// Cumulative time spent constructing a linear factorization.
    ///
    /// This is a diagnostic sub-stage of `linear_solve_duration`. For the
    /// explicit-inverse compatibility backend it is the inverse construction.
    pub linear_factorization_duration: Duration,
    /// Cumulative time spent applying a prepared factorization to a right-hand
    /// side (the back-substitution/matrix-vector part).
    pub linear_system_solve_duration: Duration,
    /// Total numerical solve time, including callbacks and method work.
    pub total_duration: Duration,
    /// Per-iteration telemetry snapshots collected for this solve.
    ///
    /// Entries describe the outer nonlinear step only. Initial state
    /// evaluation is included in the aggregate counters but is not an
    /// iteration attempt.
    pub attempts: Vec<SolveAttemptStatistics>,
}

/// Telemetry for one outer nonlinear iteration attempt.
///
/// Callback counters use the same solver-level meaning as the aggregate
/// [`SolveStatistics`] fields. Backend-internal generated jobs are not
/// counted here. `termination_retries` is `None` because the generic engine
/// has no separate termination-retry concept; a method-specific adapter may
/// publish one when that boundary becomes observable.
#[derive(Debug, Clone, Default)]
pub struct SolveAttemptStatistics {
    /// Zero-based nonlinear iteration index.
    pub iteration: usize,
    /// All residual evaluations made during this attempt.
    pub residual_evaluations: usize,
    /// Residual evaluations that establish the accepted/current state.
    pub state_residual_evaluations: usize,
    /// Residual evaluations for trial points during this attempt.
    pub trial_residual_evaluations: usize,
    /// All Jacobian evaluations made during this attempt.
    pub jacobian_evaluations: usize,
    /// Jacobian evaluations that establish the accepted/current state.
    pub state_jacobian_evaluations: usize,
    /// Jacobian evaluations for trial points during this attempt.
    pub trial_jacobian_evaluations: usize,
    /// Jacobian refreshes after an accepted state transition.
    pub jacobian_refreshes: usize,
    /// Whether the current Jacobian was retained after a rejected attempt.
    pub jacobian_reuses: usize,
    /// Linear factorizations constructed during this attempt.
    pub linear_factorizations: usize,
    /// Linear solves or method subproblems performed during this attempt.
    pub linear_solves: usize,
    /// Accepted steps during this attempt.
    pub accepted_steps: usize,
    /// Rejected trial steps during this attempt.
    pub rejected_steps: usize,
    /// Method-specific termination retries, when observable.
    pub termination_retries: Option<usize>,
    /// Residual callback duration for this attempt.
    pub residual_duration: Duration,
    /// Jacobian callback duration for this attempt.
    pub jacobian_duration: Duration,
    /// Inclusive linear operation duration for this attempt.
    pub linear_solve_duration: Duration,
    /// Linear factorization duration for this attempt.
    pub linear_factorization_duration: Duration,
    /// Linear right-hand-side application duration for this attempt.
    pub linear_system_solve_duration: Duration,
}

/// Memory estimates for the final solver state.
#[derive(Debug, Clone, Default)]
pub struct MemoryDiagnostics {
    /// Bytes used by the final solution vector.
    pub solution_bytes: usize,
    /// Bytes used by the final residual vector.
    pub residual_bytes: usize,
    /// Bytes used by the final Jacobian matrix.
    pub jacobian_bytes: usize,
    /// Bytes used by the stored history.
    pub history_bytes: usize,
    /// Sum of the reported categories.
    pub estimated_total_bytes: usize,
}

/// Final solver output.
#[derive(Debug, Clone)]
pub struct SolveResult {
    /// Final iterate.
    pub x: DVector<f64>,
    /// Residual at the final iterate.
    pub residual: DVector<f64>,
    /// Norm of the final residual.
    pub residual_norm: f64,
    /// Number of nonlinear iterations.
    pub iterations: usize,
    /// Reason why the solve stopped.
    pub termination: TerminationReason,
    /// Optional iteration history.
    pub history: Vec<IterationRecord>,
    /// Aggregate solve counters.
    pub statistics: SolveStatistics,
    /// Optional memory estimates.
    pub memory_diagnostics: Option<MemoryDiagnostics>,
}

/// Read-only snapshot of the current iteration.
#[derive(Debug, Clone)]
pub struct IterationState {
    /// Zero-based iteration index.
    pub iteration: usize,
    /// Current iterate.
    pub x: DVector<f64>,
    /// Current residual vector.
    pub residual: DVector<f64>,
    /// Current Jacobian matrix.
    pub jacobian: DMatrix<f64>,
    /// Current residual norm.
    pub residual_norm: f64,
}

/// Result of one method step.
#[derive(Debug, Clone)]
pub enum StepOutcome {
    /// Continue with a trial point.
    Continue {
        next_x: DVector<f64>,
        accepted: bool,
    },
    /// Accept the trial point and then stop with a method-specific reason.
    ///
    /// Trust-region methods need this outcome for criteria that are evaluated
    /// after MINPACK-style acceptance. Returning only `Terminated` would lose
    /// the already accepted trial point.
    AcceptedAndTerminated {
        next_x: DVector<f64>,
        reason: TerminationReason,
    },
    /// Stop because the method detected convergence.
    Converged,
    /// Stop with an explicit reason.
    Terminated(TerminationReason),
}

/// Mutable counters updated by a concrete method.
#[derive(Debug, Clone, Default)]
pub struct RuntimeDiagnostics {
    /// Trial residual evaluations performed inside a method step.
    pub residual_evaluations: usize,
    /// Trial Jacobian evaluations performed inside a method step.
    pub jacobian_evaluations: usize,
    /// Cumulative time spent evaluating trial residuals.
    pub residual_duration: Duration,
    /// Cumulative time spent evaluating trial Jacobians.
    pub jacobian_duration: Duration,
    /// Linear solves performed by the method.
    pub linear_solves: usize,
    /// Linear factorizations constructed by the method.
    pub linear_factorizations: usize,
    /// Accepted steps.
    pub accepted_steps: usize,
    /// Rejected steps.
    pub rejected_steps: usize,
    /// Cumulative time spent in linear step/subproblem solves.
    pub linear_solve_duration: Duration,
    /// Cumulative time spent constructing a linear factorization.
    pub linear_factorization_duration: Duration,
    /// Cumulative time spent applying a prepared factorization to a right-hand
    /// side.
    pub linear_system_solve_duration: Duration,
    /// Trial points written into a reusable method workspace.
    pub reusable_trial_points: usize,
}

/// Cumulative telemetry values captured at the beginning of one outer step.
#[derive(Debug, Clone, Copy, Default)]
struct AttemptTelemetryStart {
    state_residual_evaluations: usize,
    state_jacobian_evaluations: usize,
    residual_duration: Duration,
    jacobian_duration: Duration,
    trial_residual_evaluations: usize,
    trial_jacobian_evaluations: usize,
    linear_factorizations: usize,
    linear_solves: usize,
    accepted_steps: usize,
    rejected_steps: usize,
    linear_solve_duration: Duration,
    linear_factorization_duration: Duration,
    linear_system_solve_duration: Duration,
}

impl AttemptTelemetryStart {
    fn capture(stats: &SolveStatistics, runtime: &RuntimeDiagnostics) -> Self {
        Self {
            state_residual_evaluations: stats.state_residual_evaluations,
            state_jacobian_evaluations: stats.state_jacobian_evaluations,
            residual_duration: stats.residual_duration,
            jacobian_duration: stats.jacobian_duration,
            trial_residual_evaluations: runtime.residual_evaluations,
            trial_jacobian_evaluations: runtime.jacobian_evaluations,
            linear_factorizations: runtime.linear_factorizations,
            linear_solves: runtime.linear_solves,
            accepted_steps: runtime.accepted_steps,
            rejected_steps: runtime.rejected_steps,
            linear_solve_duration: runtime.linear_solve_duration,
            linear_factorization_duration: runtime.linear_factorization_duration,
            linear_system_solve_duration: runtime.linear_system_solve_duration,
        }
    }
}

/// Interface implemented by all nonlinear methods.
pub trait NonlinearMethod {
    /// Method-specific mutable state.
    type MethodState;

    /// Validates parameters and builds the initial method state.
    fn init<P: JacobianProvider>(
        &self,
        problem: &P,
        x0: &DVector<f64>,
        options: &SolveOptions,
        residual: &DVector<f64>,
        jacobian: &DMatrix<f64>,
    ) -> Result<Self::MethodState, SolveError>;

    /// Computes one trial step.
    fn step<P: JacobianProvider>(
        &self,
        problem: &P,
        state: &IterationState,
        method_state: &mut Self::MethodState,
        options: &SolveOptions,
        runtime: &mut RuntimeDiagnostics,
    ) -> Result<StepOutcome, SolveError>;

    /// Returns whether this method can reuse [`MethodWorkspace`] storage.
    fn supports_step_workspace(&self) -> bool {
        false
    }

    /// Returns whether this method consumes a reusable trial residual buffer.
    ///
    /// This is separate from [`Self::supports_step_workspace`] so methods that
    /// reuse trial points but still use owned residuals do not reserve storage
    /// they never touch.
    fn supports_trial_residual_workspace(&self) -> bool {
        false
    }

    /// Computes one trial step using solve-local reusable storage when the
    /// method supports it. The default preserves the original method contract
    /// for external implementations.
    fn step_with_workspace<P: JacobianProvider>(
        &self,
        problem: &P,
        state: &IterationState,
        method_state: &mut Self::MethodState,
        options: &SolveOptions,
        runtime: &mut RuntimeDiagnostics,
        _workspace: Option<&mut MethodWorkspace>,
    ) -> Result<StepOutcome, SolveError> {
        self.step(problem, state, method_state, options, runtime)
    }
}

/// Plain Newton method.
#[derive(Debug, Clone, Copy, Default)]
pub struct NewtonMethod;

/// Generic outer loop for nonlinear methods.
pub struct SolverEngine<M> {
    method: M,
    options: SolveOptions,
}

/// Reusable storage for temporary trial points created during one solve.
///
/// Built-in methods may use this buffer for rejected/backtracking trials. A
/// successful [`StepOutcome::Continue`] still owns its returned vector, so
/// existing callers and custom methods keep the same ownership semantics.
#[derive(Debug, Clone)]
pub struct MethodWorkspace {
    trial_x: DVector<f64>,
    trial_residual: Option<DVector<f64>>,
    track_usage: bool,
    reusable_trial_points: usize,
}

impl MethodWorkspace {
    /// Creates workspace sized for a problem dimension.
    pub fn new(dimension: usize) -> Self {
        Self::with_usage_tracking_and_residual(dimension, true, false)
    }

    /// Creates solve-local workspace and optionally reserves a residual buffer.
    ///
    /// The residual buffer is enabled only for providers that explicitly opt in
    /// to `residual_into`; compatibility providers keep the old owned path and
    /// therefore do not pay for this additional allocation.
    fn with_usage_tracking_and_residual(
        dimension: usize,
        track_usage: bool,
        reusable_residual: bool,
    ) -> Self {
        Self {
            trial_x: DVector::zeros(dimension),
            trial_residual: reusable_residual.then(|| DVector::zeros(dimension)),
            track_usage,
            reusable_trial_points: 0,
        }
    }

    /// Returns the number of trial points written since the last read.
    pub fn reusable_trial_points(&self) -> usize {
        self.reusable_trial_points
    }

    /// Takes and resets the usage count for the current reporting interval.
    fn take_reusable_trial_points(&mut self) -> usize {
        std::mem::take(&mut self.reusable_trial_points)
    }

    /// Returns the reusable trial vector.
    pub fn trial_x(&self) -> &DVector<f64> {
        &self.trial_x
    }

    /// Returns mutable access to the reusable trial vector.
    pub fn trial_x_mut(&mut self) -> &mut DVector<f64> {
        &mut self.trial_x
    }

    /// Fills `trial_x = base + scale * direction` without allocating.
    pub fn set_affine_trial(
        &mut self,
        base: &DVector<f64>,
        scale: f64,
        direction: &DVector<f64>,
    ) -> Result<(), SolveError> {
        if base.len() != direction.len() || base.len() != self.trial_x.len() {
            return Err(SolveError::DimensionMismatch {
                expected: base.len(),
                actual: direction.len(),
                context: "method trial workspace",
            });
        }
        for ((slot, base_value), direction_value) in self
            .trial_x
            .iter_mut()
            .zip(base.iter())
            .zip(direction.iter())
        {
            *slot = *base_value + scale * *direction_value;
        }
        if self.track_usage {
            self.reusable_trial_points += 1;
        }
        Ok(())
    }

    /// Borrows the trial point and its optional residual output together.
    pub(crate) fn trial_x_and_residual_mut(
        &mut self,
    ) -> (&DVector<f64>, Option<&mut DVector<f64>>) {
        (&self.trial_x, self.trial_residual.as_mut())
    }
}

/// Reusable callback output buffers owned by one solve.
///
/// The public callback traits still return owned values for compatibility.
/// Their additive `*_into` methods let providers that support caller-owned
/// storage fill these buffers without creating a result allocation each time.
struct IterationWorkspace {
    residual: Option<DVector<f64>>,
    jacobian: Option<DMatrix<f64>>,
}

impl IterationWorkspace {
    fn new(problem: &impl JacobianProvider) -> Self {
        let dimension = problem.dimension();
        Self {
            residual: problem
                .supports_residual_into()
                .then(|| DVector::zeros(dimension)),
            jacobian: problem
                .supports_jacobian_into()
                .then(|| DMatrix::zeros(dimension, dimension)),
        }
    }
}

impl<M> SolverEngine<M> {
    /// Creates a new engine.
    pub fn new(method: M, options: SolveOptions) -> Self {
        Self { method, options }
    }
}

impl<M: NonlinearMethod> SolverEngine<M> {
    /// Solves a nonlinear system from `x0`.
    pub fn solve<P>(&self, problem: &P, x0: DVector<f64>) -> Result<SolveResult, SolveError>
    where
        P: JacobianProvider,
    {
        if problem.dimension() != x0.len() {
            return Err(SolveError::DimensionMismatch {
                expected: problem.dimension(),
                actual: x0.len(),
                context: "initial guess",
            });
        }
        self.options.validate(problem.dimension())?;
        if let Some(bounds) = &self.options.bounds {
            bounds.validate(&x0)?;
        }

        let collect_statistics = self.options.diagnostics.collect_statistics;
        // Do not even read the clock when statistics are disabled. The
        // numerical path remains identical; this keeps telemetry-off solves
        // free of timing work rather than merely discarding the result.
        let solve_started = collect_statistics.then(Instant::now);
        let mut stats = SolveStatistics::default();
        stats.availability = if collect_statistics {
            StatisticsAvailability::Collected
        } else {
            StatisticsAvailability::NotCollected
        };
        let mut history = if self.options.diagnostics.collect_history {
            Vec::with_capacity(self.options.max_iterations + 1)
        } else {
            Vec::new()
        };
        let x = x0;
        let residual = eval_residual(problem, &x, &mut stats, collect_statistics)?;
        let jacobian = eval_jacobian(problem, &x, &mut stats, collect_statistics)?;
        let residual_norm = residual.norm();
        let mut state = IterationState {
            iteration: 0,
            x,
            residual,
            jacobian,
            residual_norm,
        };
        let mut workspace = IterationWorkspace::new(problem);
        let mut method_workspace = self.method.supports_step_workspace().then(|| {
            MethodWorkspace::with_usage_tracking_and_residual(
                problem.dimension(),
                collect_statistics,
                self.method.supports_trial_residual_workspace() && problem.supports_residual_into(),
            )
        });
        push_history(
            &self.options,
            &mut history,
            0,
            state.residual_norm,
            0.0,
            true,
        );

        if state.residual_norm < self.options.tolerance {
            return Ok(build_result(
                state.x,
                state.residual,
                state.jacobian,
                state.residual_norm,
                0,
                TerminationReason::Converged,
                history,
                stats,
                self.options.diagnostics.enable_memory_diagnostics,
                solve_started,
            ));
        }

        log_message(
            &self.options,
            EngineLogLevel::Info,
            "starting nonlinear solve",
        );
        let mut method_state = self.method.init(
            problem,
            &state.x,
            &self.options,
            &state.residual,
            &state.jacobian,
        )?;
        let mut runtime = RuntimeDiagnostics::default();

        for iteration in 0..self.options.max_iterations {
            state.iteration = iteration;
            let attempt_start =
                collect_statistics.then(|| AttemptTelemetryStart::capture(&stats, &runtime));
            if self.options.diagnostics.enable_logging
                && self.options.diagnostics.log_level == EngineLogLevel::Debug
            {
                log_message(
                    &self.options,
                    EngineLogLevel::Debug,
                    &format!(
                        "iteration {iteration}, residual = {:.6e}",
                        state.residual_norm
                    ),
                );
            }

            let outcome = self.method.step_with_workspace(
                problem,
                &state,
                &mut method_state,
                &self.options,
                &mut runtime,
                method_workspace.as_mut(),
            )?;
            if let Some(workspace) = method_workspace.as_mut() {
                runtime.reusable_trial_points += workspace.take_reusable_trial_points();
            }

            match outcome {
                StepOutcome::Converged => {
                    stats.iterations = iteration;
                    record_attempt(&mut stats, attempt_start, &runtime, iteration, false);
                    merge_runtime(&mut stats, &runtime, collect_statistics);
                    return Ok(build_result(
                        state.x,
                        state.residual,
                        state.jacobian,
                        state.residual_norm,
                        iteration,
                        TerminationReason::Converged,
                        history,
                        stats,
                        self.options.diagnostics.enable_memory_diagnostics,
                        solve_started,
                    ));
                }
                StepOutcome::Terminated(reason) => {
                    stats.iterations = iteration;
                    record_attempt(&mut stats, attempt_start, &runtime, iteration, false);
                    merge_runtime(&mut stats, &runtime, collect_statistics);
                    return Ok(build_result(
                        state.x,
                        state.residual,
                        state.jacobian,
                        state.residual_norm,
                        iteration,
                        reason,
                        history,
                        stats,
                        self.options.diagnostics.enable_memory_diagnostics,
                        solve_started,
                    ));
                }
                StepOutcome::AcceptedAndTerminated { next_x, reason } => {
                    let mut next_x = next_x;
                    if let Some(bounds) = &self.options.bounds {
                        bounds.project_in_place(&mut next_x);
                    }
                    let step_norm = (&next_x - &state.x).norm();
                    state.x = next_x;
                    if problem.supports_residual_into() {
                        let residual = workspace
                            .residual
                            .as_mut()
                            .expect("residual workspace must exist when enabled");
                        eval_residual_into(
                            problem,
                            &state.x,
                            residual,
                            &mut stats,
                            collect_statistics,
                        )?;
                        std::mem::swap(&mut state.residual, residual);
                    } else {
                        state.residual =
                            eval_residual(problem, &state.x, &mut stats, collect_statistics)?;
                    }
                    if problem.supports_jacobian_into() {
                        let jacobian = workspace
                            .jacobian
                            .as_mut()
                            .expect("Jacobian workspace must exist when enabled");
                        eval_jacobian_into(
                            problem,
                            &state.x,
                            jacobian,
                            &mut stats,
                            collect_statistics,
                        )?;
                        std::mem::swap(&mut state.jacobian, jacobian);
                    } else {
                        state.jacobian =
                            eval_jacobian(problem, &state.x, &mut stats, collect_statistics)?;
                    }
                    state.residual_norm = state.residual.norm();
                    push_history(
                        &self.options,
                        &mut history,
                        iteration + 1,
                        state.residual_norm,
                        step_norm,
                        true,
                    );
                    stats.iterations = iteration + 1;
                    record_attempt(&mut stats, attempt_start, &runtime, iteration, false);
                    merge_runtime(&mut stats, &runtime, collect_statistics);
                    return Ok(build_result(
                        state.x,
                        state.residual,
                        state.jacobian,
                        state.residual_norm,
                        iteration + 1,
                        reason,
                        history,
                        stats,
                        self.options.diagnostics.enable_memory_diagnostics,
                        solve_started,
                    ));
                }
                StepOutcome::Continue { next_x, accepted } => {
                    let mut next_x = next_x;
                    if let Some(bounds) = &self.options.bounds {
                        bounds.project_in_place(&mut next_x);
                    }
                    let step_norm = (&next_x - &state.x).norm();
                    if accepted {
                        state.x = next_x;
                        if problem.supports_residual_into() {
                            let residual = workspace
                                .residual
                                .as_mut()
                                .expect("residual workspace must exist when enabled");
                            eval_residual_into(
                                problem,
                                &state.x,
                                residual,
                                &mut stats,
                                collect_statistics,
                            )?;
                            std::mem::swap(&mut state.residual, residual);
                        } else {
                            state.residual =
                                eval_residual(problem, &state.x, &mut stats, collect_statistics)?;
                        }
                        if problem.supports_jacobian_into() {
                            let jacobian = workspace
                                .jacobian
                                .as_mut()
                                .expect("Jacobian workspace must exist when enabled");
                            eval_jacobian_into(
                                problem,
                                &state.x,
                                jacobian,
                                &mut stats,
                                collect_statistics,
                            )?;
                            std::mem::swap(&mut state.jacobian, jacobian);
                        } else {
                            state.jacobian =
                                eval_jacobian(problem, &state.x, &mut stats, collect_statistics)?;
                        }
                        state.residual_norm = state.residual.norm();
                    }
                    push_history(
                        &self.options,
                        &mut history,
                        iteration + 1,
                        state.residual_norm,
                        step_norm,
                        accepted,
                    );
                    record_attempt(&mut stats, attempt_start, &runtime, iteration, !accepted);
                    if state.residual_norm < self.options.tolerance {
                        stats.iterations = iteration + 1;
                        merge_runtime(&mut stats, &runtime, collect_statistics);
                        return Ok(build_result(
                            state.x,
                            state.residual,
                            state.jacobian,
                            state.residual_norm,
                            iteration + 1,
                            TerminationReason::Converged,
                            history,
                            stats,
                            self.options.diagnostics.enable_memory_diagnostics,
                            solve_started,
                        ));
                    }
                }
            }
        }

        stats.iterations = self.options.max_iterations;
        merge_runtime(&mut stats, &runtime, collect_statistics);
        Ok(build_result(
            state.x,
            state.residual,
            state.jacobian,
            state.residual_norm,
            self.options.max_iterations,
            TerminationReason::MaxIterations,
            history,
            stats,
            self.options.diagnostics.enable_memory_diagnostics,
            solve_started,
        ))
    }
}
//======================================================================================
// classical Newton method
//======================================================================================
impl NonlinearMethod for NewtonMethod {
    type MethodState = ();

    fn init<P: JacobianProvider>(
        &self,
        _problem: &P,
        _x0: &DVector<f64>,
        _options: &SolveOptions,
        _residual: &DVector<f64>,
        _jacobian: &DMatrix<f64>,
    ) -> Result<Self::MethodState, SolveError> {
        Ok(())
    }

    fn step<P: JacobianProvider>(
        &self,
        _problem: &P,
        state: &IterationState,
        _method_state: &mut Self::MethodState,
        options: &SolveOptions,
        runtime: &mut RuntimeDiagnostics,
    ) -> Result<StepOutcome, SolveError> {
        runtime.linear_solves += 1;
        let step = measure_linear_system_operation(
            options.linear_solver,
            &state.jacobian,
            &state.residual,
            runtime,
            options.diagnostics.collect_statistics,
        )?;
        if step.norm() < options.tolerance {
            return Ok(StepOutcome::Terminated(TerminationReason::StepTooSmall));
        }
        runtime.accepted_steps += 1;
        Ok(StepOutcome::Continue {
            next_x: &state.x - step,
            accepted: true,
        })
    }
}

//======================================================================================
/// Solves a dense linear system with the selected backend.
pub fn solve_linear_system(
    solver: LinearSolverKind,
    matrix: &DMatrix<f64>,
    rhs: &DVector<f64>,
) -> Result<DVector<f64>, SolveError> {
    match solver {
        LinearSolverKind::Lu => matrix
            .clone()
            .lu()
            .solve(rhs)
            .ok_or(SolveError::SingularJacobian),
        LinearSolverKind::Inverse => matrix
            .clone()
            .try_inverse()
            .map(|inv| inv * rhs)
            .ok_or(SolveError::SingularJacobian),
    }
}

/// Solves a dense linear system and records its two observable algebra stages.
///
/// The caller still owns the inclusive timer for the complete linear step via
/// [`measure_linear_operation`]. This helper only records factorization and
/// application sub-stages, so the two metrics must not be added to other
/// inclusive timers as if they were disjoint solver stages.
pub(crate) fn solve_linear_system_with_stage_timing(
    solver: LinearSolverKind,
    matrix: &DMatrix<f64>,
    rhs: &DVector<f64>,
    runtime: &mut RuntimeDiagnostics,
    collect_statistics: bool,
) -> Result<DVector<f64>, SolveError> {
    if !collect_statistics {
        return solve_linear_system(solver, matrix, rhs);
    }

    match solver {
        LinearSolverKind::Lu => {
            let factor_started = Instant::now();
            let lu = matrix.clone().lu();
            runtime.linear_factorizations += 1;
            runtime.linear_factorization_duration += factor_started.elapsed();

            let solve_started = Instant::now();
            let result = lu.solve(rhs);
            runtime.linear_system_solve_duration += solve_started.elapsed();
            result.ok_or(SolveError::SingularJacobian)
        }
        LinearSolverKind::Inverse => {
            let factor_started = Instant::now();
            let inverse = matrix.clone().try_inverse();
            runtime.linear_factorizations += 1;
            runtime.linear_factorization_duration += factor_started.elapsed();
            let inverse = inverse.ok_or(SolveError::SingularJacobian)?;

            let solve_started = Instant::now();
            let result = inverse * rhs;
            runtime.linear_system_solve_duration += solve_started.elapsed();
            Ok(result)
        }
    }
}

/// Solves an owned dense linear system and records its algebra stages.
///
/// This variant is for methods that have just constructed a temporary matrix
/// and do not need it after the solve. Consuming that matrix avoids the extra
/// full `DMatrix` clone required by the borrowed compatibility helper.
pub(crate) fn solve_linear_system_owned_with_stage_timing(
    solver: LinearSolverKind,
    matrix: DMatrix<f64>,
    rhs: &DVector<f64>,
    runtime: &mut RuntimeDiagnostics,
    collect_statistics: bool,
) -> Result<DVector<f64>, SolveError> {
    if !collect_statistics {
        return solve_linear_system_owned(solver, matrix, rhs);
    }

    match solver {
        LinearSolverKind::Lu => {
            let factor_started = Instant::now();
            let lu = matrix.lu();
            runtime.linear_factorizations += 1;
            runtime.linear_factorization_duration += factor_started.elapsed();

            let solve_started = Instant::now();
            let result = lu.solve(rhs);
            runtime.linear_system_solve_duration += solve_started.elapsed();
            result.ok_or(SolveError::SingularJacobian)
        }
        LinearSolverKind::Inverse => {
            let factor_started = Instant::now();
            let inverse = matrix.try_inverse();
            runtime.linear_factorizations += 1;
            runtime.linear_factorization_duration += factor_started.elapsed();
            let inverse = inverse.ok_or(SolveError::SingularJacobian)?;

            let solve_started = Instant::now();
            let result = inverse * rhs;
            runtime.linear_system_solve_duration += solve_started.elapsed();
            Ok(result)
        }
    }
}

/// Solves an owned dense linear system without copying its matrix.
fn solve_linear_system_owned(
    solver: LinearSolverKind,
    matrix: DMatrix<f64>,
    rhs: &DVector<f64>,
) -> Result<DVector<f64>, SolveError> {
    match solver {
        LinearSolverKind::Lu => matrix.lu().solve(rhs).ok_or(SolveError::SingularJacobian),
        LinearSolverKind::Inverse => matrix
            .try_inverse()
            .map(|inv| inv * rhs)
            .ok_or(SolveError::SingularJacobian),
    }
}

/// Measures the inclusive linear operation and its factorization sub-stages.
pub(crate) fn measure_linear_system_operation(
    solver: LinearSolverKind,
    matrix: &DMatrix<f64>,
    rhs: &DVector<f64>,
    runtime: &mut RuntimeDiagnostics,
    collect_statistics: bool,
) -> Result<DVector<f64>, SolveError> {
    if !collect_statistics {
        return solve_linear_system(solver, matrix, rhs);
    }
    let started = Instant::now();
    let result = solve_linear_system_with_stage_timing(solver, matrix, rhs, runtime, true);
    runtime.linear_solve_duration += started.elapsed();
    result
}

/// Measures an owned temporary linear operation without cloning its matrix.
pub(crate) fn measure_linear_system_operation_owned(
    solver: LinearSolverKind,
    matrix: DMatrix<f64>,
    rhs: &DVector<f64>,
    runtime: &mut RuntimeDiagnostics,
    collect_statistics: bool,
) -> Result<DVector<f64>, SolveError> {
    if !collect_statistics {
        return solve_linear_system_owned(solver, matrix, rhs);
    }
    let started = Instant::now();
    let result = solve_linear_system_owned_with_stage_timing(solver, matrix, rhs, runtime, true);
    runtime.linear_solve_duration += started.elapsed();
    result
}

/// Builds a column scaling vector from the Jacobian.
pub fn scaling_vector(jacobian: &DMatrix<f64>, use_column_scaling: bool) -> DVector<f64> {
    if !use_column_scaling {
        return DVector::from_element(jacobian.ncols(), 1.0);
    }
    let mut scaling = DVector::zeros(jacobian.ncols());
    for column in 0..jacobian.ncols() {
        let norm = jacobian.column(column).norm();
        scaling[column] = if norm > 0.0 { norm } else { 1.0 };
    }
    scaling
}

/// Computes `||D v||` for a diagonal scaling vector `D`.
pub fn scaled_norm(diag: &DVector<f64>, vector: &DVector<f64>) -> f64 {
    diag.component_mul(vector).norm()
}

/// Evaluates the residual and updates counters.
fn eval_residual<P: JacobianProvider>(
    problem: &P,
    x: &DVector<f64>,
    stats: &mut SolveStatistics,
    collect_statistics: bool,
) -> Result<DVector<f64>, SolveError> {
    if !collect_statistics {
        return problem.residual(x);
    }
    stats.residual_evaluations += 1;
    stats.state_residual_evaluations += 1;
    let started = Instant::now();
    let result = problem.residual(x);
    stats.residual_duration += started.elapsed();
    result
}

/// Evaluates the Jacobian and updates counters.
fn eval_jacobian<P: JacobianProvider>(
    problem: &P,
    x: &DVector<f64>,
    stats: &mut SolveStatistics,
    collect_statistics: bool,
) -> Result<DMatrix<f64>, SolveError> {
    if !collect_statistics {
        return problem.jacobian(x);
    }
    stats.jacobian_evaluations += 1;
    stats.state_jacobian_evaluations += 1;
    let started = Instant::now();
    let result = problem.jacobian(x);
    stats.jacobian_duration += started.elapsed();
    result
}

/// Evaluates a residual into reusable solve-local storage and updates counters.
fn eval_residual_into<P: JacobianProvider>(
    problem: &P,
    x: &DVector<f64>,
    out: &mut DVector<f64>,
    stats: &mut SolveStatistics,
    collect_statistics: bool,
) -> Result<(), SolveError> {
    if !collect_statistics {
        return problem.residual_into(x, out);
    }
    stats.residual_evaluations += 1;
    stats.state_residual_evaluations += 1;
    let started = Instant::now();
    let result = problem.residual_into(x, out);
    stats.residual_duration += started.elapsed();
    result
}

/// Evaluates a Jacobian into reusable solve-local storage and updates counters.
fn eval_jacobian_into<P: JacobianProvider>(
    problem: &P,
    x: &DVector<f64>,
    out: &mut DMatrix<f64>,
    stats: &mut SolveStatistics,
    collect_statistics: bool,
) -> Result<(), SolveError> {
    if !collect_statistics {
        return problem.jacobian_into(x, out);
    }
    stats.jacobian_evaluations += 1;
    stats.state_jacobian_evaluations += 1;
    let started = Instant::now();
    let result = problem.jacobian_into(x, out);
    stats.jacobian_duration += started.elapsed();
    result
}

/// Evaluates a trial residual and records method-local callback telemetry.
pub(crate) fn eval_residual_with_runtime<P: JacobianProvider>(
    problem: &P,
    x: &DVector<f64>,
    runtime: &mut RuntimeDiagnostics,
    collect_statistics: bool,
) -> Result<DVector<f64>, SolveError> {
    if !collect_statistics {
        return problem.residual(x);
    }
    runtime.residual_evaluations += 1;
    let started = Instant::now();
    let result = problem.residual(x);
    runtime.residual_duration += started.elapsed();
    result
}

/// Evaluates only a trial residual norm, optionally reusing caller-owned
/// storage for providers that implement `residual_into`.
pub(crate) fn eval_residual_norm_with_runtime<P: JacobianProvider>(
    problem: &P,
    x: &DVector<f64>,
    runtime: &mut RuntimeDiagnostics,
    collect_statistics: bool,
    output: Option<&mut DVector<f64>>,
) -> Result<f64, SolveError> {
    if problem.supports_residual_into() {
        if let Some(output) = output {
            if !collect_statistics {
                problem.residual_into(x, output)?;
                return Ok(output.norm());
            }
            runtime.residual_evaluations += 1;
            let started = Instant::now();
            let result = problem.residual_into(x, output);
            runtime.residual_duration += started.elapsed();
            result?;
            return Ok(output.norm());
        }
    }

    Ok(eval_residual_with_runtime(problem, x, runtime, collect_statistics)?.norm())
}

/// Evaluates a trial Jacobian and records method-local callback telemetry.
pub(crate) fn eval_jacobian_with_runtime<P: JacobianProvider>(
    problem: &P,
    x: &DVector<f64>,
    runtime: &mut RuntimeDiagnostics,
    collect_statistics: bool,
) -> Result<DMatrix<f64>, SolveError> {
    if !collect_statistics {
        return problem.jacobian(x);
    }
    runtime.jacobian_evaluations += 1;
    let started = Instant::now();
    let result = problem.jacobian(x);
    runtime.jacobian_duration += started.elapsed();
    result
}

/// Publishes one outer-step telemetry snapshot after its state refresh, if any.
fn record_attempt(
    stats: &mut SolveStatistics,
    start: Option<AttemptTelemetryStart>,
    runtime: &RuntimeDiagnostics,
    iteration: usize,
    jacobian_reused: bool,
) {
    let Some(start) = start else {
        return;
    };

    let state_residual_evaluations = stats
        .state_residual_evaluations
        .saturating_sub(start.state_residual_evaluations);
    let state_jacobian_evaluations = stats
        .state_jacobian_evaluations
        .saturating_sub(start.state_jacobian_evaluations);
    let trial_residual_evaluations = runtime
        .residual_evaluations
        .saturating_sub(start.trial_residual_evaluations);
    let trial_jacobian_evaluations = runtime
        .jacobian_evaluations
        .saturating_sub(start.trial_jacobian_evaluations);
    let residual_evaluations =
        state_residual_evaluations.saturating_add(trial_residual_evaluations);
    let jacobian_evaluations =
        state_jacobian_evaluations.saturating_add(trial_jacobian_evaluations);

    stats.attempts.push(SolveAttemptStatistics {
        iteration,
        residual_evaluations,
        state_residual_evaluations,
        trial_residual_evaluations,
        jacobian_evaluations,
        state_jacobian_evaluations,
        trial_jacobian_evaluations,
        jacobian_refreshes: state_jacobian_evaluations,
        jacobian_reuses: usize::from(jacobian_reused),
        linear_factorizations: runtime
            .linear_factorizations
            .saturating_sub(start.linear_factorizations),
        linear_solves: runtime.linear_solves.saturating_sub(start.linear_solves),
        accepted_steps: runtime.accepted_steps.saturating_sub(start.accepted_steps),
        rejected_steps: runtime.rejected_steps.saturating_sub(start.rejected_steps),
        termination_retries: None,
        residual_duration: stats
            .residual_duration
            .saturating_sub(start.residual_duration),
        jacobian_duration: stats
            .jacobian_duration
            .saturating_sub(start.jacobian_duration),
        linear_solve_duration: runtime
            .linear_solve_duration
            .saturating_sub(start.linear_solve_duration),
        linear_factorization_duration: runtime
            .linear_factorization_duration
            .saturating_sub(start.linear_factorization_duration),
        linear_system_solve_duration: runtime
            .linear_system_solve_duration
            .saturating_sub(start.linear_system_solve_duration),
    });
}

/// Measures one linear step or trust-region subproblem without changing its result.
pub(crate) fn measure_linear_operation<T, F>(
    runtime: &mut RuntimeDiagnostics,
    collect_statistics: bool,
    operation: F,
) -> T
where
    F: FnOnce() -> T,
{
    if !collect_statistics {
        return operation();
    }
    let started = Instant::now();
    let result = operation();
    runtime.linear_solve_duration += started.elapsed();
    result
}

/// Stores one history record when history collection is enabled.
fn push_history(
    options: &SolveOptions,
    history: &mut Vec<IterationRecord>,
    iteration: usize,
    residual_norm: f64,
    step_norm: f64,
    accepted: bool,
) {
    if options.diagnostics.collect_history {
        history.push(IterationRecord {
            iteration,
            residual_norm,
            step_norm,
            accepted,
        });
    }
}

/// Merges method-local counters into the final statistics.
fn merge_runtime(
    stats: &mut SolveStatistics,
    runtime: &RuntimeDiagnostics,
    collect_statistics: bool,
) {
    if !collect_statistics {
        return;
    }
    stats.linear_solves = runtime.linear_solves;
    stats.linear_factorizations = runtime.linear_factorizations;
    stats.accepted_steps = runtime.accepted_steps;
    stats.rejected_steps = runtime.rejected_steps;
    stats.linear_solve_duration = runtime.linear_solve_duration;
    stats.residual_evaluations += runtime.residual_evaluations;
    stats.jacobian_evaluations += runtime.jacobian_evaluations;
    stats.trial_residual_evaluations = runtime.residual_evaluations;
    stats.trial_jacobian_evaluations = runtime.jacobian_evaluations;
    stats.residual_duration += runtime.residual_duration;
    stats.jacobian_duration += runtime.jacobian_duration;
    stats.linear_factorization_duration = runtime.linear_factorization_duration;
    stats.linear_system_solve_duration = runtime.linear_system_solve_duration;
    stats.reusable_trial_points = runtime.reusable_trial_points;
}

/// Builds the final solver result.
fn build_result(
    x: DVector<f64>,
    residual: DVector<f64>,
    jacobian: DMatrix<f64>,
    residual_norm: f64,
    iterations: usize,
    termination: TerminationReason,
    history: Vec<IterationRecord>,
    mut statistics: SolveStatistics,
    with_memory: bool,
    solve_started: Option<Instant>,
) -> SolveResult {
    statistics.iterations = iterations;
    if let Some(solve_started) = solve_started {
        statistics.total_duration = solve_started.elapsed();
    }
    let memory_diagnostics = with_memory.then(|| {
        let solution_bytes = x.len() * std::mem::size_of::<f64>();
        let residual_bytes = residual.len() * std::mem::size_of::<f64>();
        let jacobian_bytes = jacobian.nrows() * jacobian.ncols() * std::mem::size_of::<f64>();
        let history_bytes = std::mem::size_of_val(history.as_slice());
        MemoryDiagnostics {
            solution_bytes,
            residual_bytes,
            jacobian_bytes,
            history_bytes,
            estimated_total_bytes: solution_bytes + residual_bytes + jacobian_bytes + history_bytes,
        }
    });
    SolveResult {
        x,
        residual,
        residual_norm,
        iterations,
        termination,
        history,
        statistics,
        memory_diagnostics,
    }
}

/// Emits a log message only when engine logging is enabled.
fn log_message(options: &SolveOptions, level: EngineLogLevel, message: &str) {
    if !options.diagnostics.enable_logging {
        return;
    }
    match options.diagnostics.log_level {
        EngineLogLevel::Debug => match level {
            EngineLogLevel::Debug => debug!("{message}"),
            EngineLogLevel::Info => info!("{message}"),
            EngineLogLevel::Warn => warn!("{message}"),
        },
        EngineLogLevel::Info => match level {
            EngineLogLevel::Debug => {}
            EngineLogLevel::Info => info!("{message}"),
            EngineLogLevel::Warn => warn!("{message}"),
        },
        EngineLogLevel::Warn => {
            if matches!(level, EngineLogLevel::Warn) {
                warn!("{message}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::Nonlinear_systems::problem::{JacobianProvider, NonlinearProblem};
    use std::cell::Cell;

    struct ScalarQuadraticProblem;
    struct CoupledPlainProblem;

    struct ReusableRosenbrockProblem {
        residual_into_calls: Cell<usize>,
    }

    struct ReusableScalarQuadraticProblem {
        residual_into_calls: Cell<usize>,
        jacobian_into_calls: Cell<usize>,
    }

    struct CountingScalarQuadraticProblem {
        residual_calls: Cell<usize>,
        jacobian_calls: Cell<usize>,
    }

    impl CountingScalarQuadraticProblem {
        fn new() -> Self {
            Self {
                residual_calls: Cell::new(0),
                jacobian_calls: Cell::new(0),
            }
        }
    }

    impl NonlinearProblem for ScalarQuadraticProblem {
        fn dimension(&self) -> usize {
            1
        }
        fn residual(&self, x: &DVector<f64>) -> Result<DVector<f64>, SolveError> {
            Ok(DVector::from_vec(vec![x[0] * x[0] - 2.0]))
        }
    }

    impl JacobianProvider for ScalarQuadraticProblem {
        fn jacobian(&self, x: &DVector<f64>) -> Result<DMatrix<f64>, SolveError> {
            Ok(DMatrix::from_row_slice(1, 1, &[2.0 * x[0]]))
        }
    }

    impl NonlinearProblem for ReusableScalarQuadraticProblem {
        fn dimension(&self) -> usize {
            1
        }

        fn supports_residual_into(&self) -> bool {
            true
        }

        fn residual(&self, x: &DVector<f64>) -> Result<DVector<f64>, SolveError> {
            Ok(DVector::from_vec(vec![x[0] * x[0] - 2.0]))
        }

        fn residual_into(
            &self,
            x: &DVector<f64>,
            out: &mut DVector<f64>,
        ) -> Result<(), SolveError> {
            assert_eq!(out.len(), 1);
            self.residual_into_calls
                .set(self.residual_into_calls.get() + 1);
            out[0] = x[0] * x[0] - 2.0;
            Ok(())
        }
    }

    impl JacobianProvider for ReusableScalarQuadraticProblem {
        fn supports_jacobian_into(&self) -> bool {
            true
        }

        fn jacobian(&self, x: &DVector<f64>) -> Result<DMatrix<f64>, SolveError> {
            Ok(DMatrix::from_row_slice(1, 1, &[2.0 * x[0]]))
        }

        fn jacobian_into(
            &self,
            x: &DVector<f64>,
            out: &mut DMatrix<f64>,
        ) -> Result<(), SolveError> {
            assert_eq!(out.shape(), (1, 1));
            self.jacobian_into_calls
                .set(self.jacobian_into_calls.get() + 1);
            out[(0, 0)] = 2.0 * x[0];
            Ok(())
        }
    }

    impl NonlinearProblem for ReusableRosenbrockProblem {
        fn dimension(&self) -> usize {
            2
        }

        fn supports_residual_into(&self) -> bool {
            true
        }

        fn residual(&self, x: &DVector<f64>) -> Result<DVector<f64>, SolveError> {
            Ok(DVector::from_vec(vec![
                10.0 * (x[1] - x[0] * x[0]),
                1.0 - x[0],
            ]))
        }

        fn residual_into(
            &self,
            x: &DVector<f64>,
            out: &mut DVector<f64>,
        ) -> Result<(), SolveError> {
            if out.len() != 2 {
                return Err(SolveError::DimensionMismatch {
                    expected: 2,
                    actual: out.len(),
                    context: "reusable Rosenbrock residual output",
                });
            }
            self.residual_into_calls
                .set(self.residual_into_calls.get() + 1);
            out[0] = 10.0 * (x[1] - x[0] * x[0]);
            out[1] = 1.0 - x[0];
            Ok(())
        }
    }

    impl JacobianProvider for ReusableRosenbrockProblem {
        fn jacobian(&self, x: &DVector<f64>) -> Result<DMatrix<f64>, SolveError> {
            Ok(DMatrix::from_row_slice(
                2,
                2,
                &[-20.0 * x[0], 10.0, -1.0, 0.0],
            ))
        }
    }

    impl NonlinearProblem for CountingScalarQuadraticProblem {
        fn dimension(&self) -> usize {
            1
        }

        fn residual(&self, x: &DVector<f64>) -> Result<DVector<f64>, SolveError> {
            self.residual_calls.set(self.residual_calls.get() + 1);
            Ok(DVector::from_vec(vec![x[0] * x[0] - 2.0]))
        }
    }

    impl JacobianProvider for CountingScalarQuadraticProblem {
        fn jacobian(&self, x: &DVector<f64>) -> Result<DMatrix<f64>, SolveError> {
            self.jacobian_calls.set(self.jacobian_calls.get() + 1);
            Ok(DMatrix::from_row_slice(1, 1, &[2.0 * x[0]]))
        }
    }

    impl NonlinearProblem for CoupledPlainProblem {
        fn dimension(&self) -> usize {
            2
        }
        fn residual(&self, x: &DVector<f64>) -> Result<DVector<f64>, SolveError> {
            Ok(DVector::from_vec(vec![
                x[0] * x[0] + x[1] * x[1] - 1.0,
                x[0] - x[1],
            ]))
        }
    }

    impl JacobianProvider for CoupledPlainProblem {
        fn jacobian(&self, x: &DVector<f64>) -> Result<DMatrix<f64>, SolveError> {
            Ok(DMatrix::from_row_slice(
                2,
                2,
                &[2.0 * x[0], 2.0 * x[1], 1.0, -1.0],
            ))
        }
    }

    fn assert_step_outcomes_match(left: StepOutcome, right: StepOutcome) {
        match (left, right) {
            (
                StepOutcome::Continue {
                    next_x: left_x,
                    accepted: left_accepted,
                },
                StepOutcome::Continue {
                    next_x: right_x,
                    accepted: right_accepted,
                },
            ) => {
                assert_eq!(left_accepted, right_accepted);
                assert_eq!(left_x.len(), right_x.len());
                assert!((left_x - right_x).norm() < 1e-12);
            }
            (StepOutcome::Converged, StepOutcome::Converged) => {}
            (
                StepOutcome::AcceptedAndTerminated {
                    next_x: left_x,
                    reason: left_reason,
                },
                StepOutcome::AcceptedAndTerminated {
                    next_x: right_x,
                    reason: right_reason,
                },
            ) => {
                assert_eq!(left_reason, right_reason);
                assert!((left_x - right_x).norm() < 1e-12);
            }
            (StepOutcome::Terminated(left), StepOutcome::Terminated(right)) => {
                assert_eq!(left, right);
            }
            (left, right) => panic!("workspace changed step outcome: {left:?} vs {right:?}"),
        }
    }

    fn assert_workspace_step_preserves_behavior<M>(method: M)
    where
        M: NonlinearMethod + Clone,
    {
        let problem = ScalarQuadraticProblem;
        let x = DVector::from_vec(vec![1.5]);
        let residual = problem.residual(&x).expect("residual");
        let jacobian = problem.jacobian(&x).expect("jacobian");
        let state = IterationState {
            iteration: 0,
            x,
            residual: residual.clone(),
            jacobian: jacobian.clone(),
            residual_norm: residual.norm(),
        };
        let options = SolveOptions::default();

        let mut legacy_state = method
            .init(&problem, &state.x, &options, &residual, &jacobian)
            .expect("legacy method init");
        let mut workspace_state = method
            .init(&problem, &state.x, &options, &residual, &jacobian)
            .expect("workspace method init");
        let mut legacy_runtime = RuntimeDiagnostics::default();
        let mut workspace_runtime = RuntimeDiagnostics::default();

        let legacy_outcome = method
            .step(
                &problem,
                &state,
                &mut legacy_state,
                &options,
                &mut legacy_runtime,
            )
            .expect("legacy step");
        let mut workspace = MethodWorkspace::new(state.x.len());
        let workspace_outcome = method
            .step_with_workspace(
                &problem,
                &state,
                &mut workspace_state,
                &options,
                &mut workspace_runtime,
                Some(&mut workspace),
            )
            .expect("workspace step");

        assert_step_outcomes_match(legacy_outcome, workspace_outcome);
        assert_eq!(
            legacy_runtime.residual_evaluations,
            workspace_runtime.residual_evaluations
        );
        assert_eq!(
            legacy_runtime.jacobian_evaluations,
            workspace_runtime.jacobian_evaluations
        );
        assert_eq!(
            legacy_runtime.accepted_steps,
            workspace_runtime.accepted_steps
        );
        assert_eq!(
            legacy_runtime.rejected_steps,
            workspace_runtime.rejected_steps
        );
        assert!(workspace.reusable_trial_points() > 0);
    }

    #[test]
    fn newton_engine_converges_for_scalar_problem() {
        let result = SolverEngine::new(NewtonMethod, SolveOptions::default())
            .solve(&ScalarQuadraticProblem, DVector::from_vec(vec![1.5]))
            .expect("solve");
        assert_eq!(result.termination, TerminationReason::Converged);
        assert_eq!(
            result.statistics.availability,
            StatisticsAvailability::Collected
        );
        assert!((result.x[0] - 2.0_f64.sqrt()).abs() < 1e-8);
        assert!(result.statistics.linear_solves > 0);
        assert!(result.statistics.linear_factorizations > 0);
        assert!(result.statistics.residual_evaluations > 0);
        assert!(result.statistics.jacobian_evaluations > 0);
        assert_eq!(
            result.statistics.residual_evaluations,
            result.statistics.state_residual_evaluations
                + result.statistics.trial_residual_evaluations
        );
        assert_eq!(
            result.statistics.jacobian_evaluations,
            result.statistics.state_jacobian_evaluations
                + result.statistics.trial_jacobian_evaluations
        );
        assert_eq!(result.statistics.trial_residual_evaluations, 0);
        assert_eq!(result.statistics.trial_jacobian_evaluations, 0);
        assert!(result.statistics.linear_solve_duration <= result.statistics.total_duration);
        assert!(result.statistics.linear_factorization_duration > Duration::ZERO);
        assert!(result.statistics.linear_system_solve_duration > Duration::ZERO);
        assert!(result.statistics.total_duration >= result.statistics.residual_duration);
        assert!(result.statistics.total_duration >= result.statistics.jacobian_duration);
        assert!(!result.statistics.attempts.is_empty());
        assert_eq!(
            result
                .statistics
                .attempts
                .iter()
                .map(|attempt| attempt.residual_evaluations)
                .sum::<usize>(),
            result.statistics.residual_evaluations - 1
        );
        assert_eq!(
            result
                .statistics
                .attempts
                .iter()
                .map(|attempt| attempt.linear_factorizations)
                .sum::<usize>(),
            result.statistics.linear_factorizations
        );
    }

    #[test]
    fn engine_uses_reusable_callback_buffers_without_changing_solution() {
        let problem = ReusableScalarQuadraticProblem {
            residual_into_calls: Cell::new(0),
            jacobian_into_calls: Cell::new(0),
        };
        let result = SolverEngine::new(NewtonMethod, SolveOptions::default())
            .solve(&problem, DVector::from_vec(vec![1.5]))
            .expect("solve");

        assert_eq!(result.termination, TerminationReason::Converged);
        assert!((result.x[0] - 2.0_f64.sqrt()).abs() < 1e-8);
        assert!(problem.residual_into_calls.get() > 0);
        assert!(problem.jacobian_into_calls.get() > 0);
    }

    #[test]
    fn damped_workspace_reuses_residual_for_rejected_trials() {
        let problem = ReusableRosenbrockProblem {
            residual_into_calls: Cell::new(0),
        };
        let result = SolverEngine::new(
            crate::numerical::Nonlinear_systems::NR_damped::DampedNewtonMethod::default(),
            SolveOptions {
                tolerance: 1e-8,
                max_iterations: 64,
                ..SolveOptions::default()
            },
        )
        .solve(&problem, DVector::from_vec(vec![-1.2, 1.0]))
        .expect("Rosenbrock solve");

        assert_eq!(result.termination, TerminationReason::Converged);
        assert!(result.statistics.rejected_steps > 0);
        assert!((result.x[0] - 1.0).abs() < 1e-7);
        assert!((result.x[1] - 1.0).abs() < 1e-7);
        assert_eq!(
            result.statistics.reusable_trial_points,
            result.statistics.trial_residual_evaluations
        );
        assert!(problem.residual_into_calls.get() >= result.statistics.reusable_trial_points);
        assert!(problem.residual_into_calls.get() > result.statistics.rejected_steps);
    }

    #[test]
    fn lm_workspace_preserves_acceptance_and_trial_values() {
        assert_workspace_step_preserves_behavior(
            crate::numerical::Nonlinear_systems::LM_vanilla::LevenbergMarquardtMethod::default(),
        );
        assert_workspace_step_preserves_behavior(
            crate::numerical::Nonlinear_systems::LM_Nielsen::NielsenLevenbergMarquardtMethod::default(),
        );
        assert_workspace_step_preserves_behavior(
            crate::numerical::Nonlinear_systems::LM_Nielsen::NielsenLevenbergMarquardtMethodAdvanced::default(),
        );
    }

    #[test]
    fn statistics_can_be_disabled_without_publishing_runtime_metrics() {
        let options = SolveOptions {
            diagnostics: DiagnosticsOptions {
                collect_statistics: false,
                ..DiagnosticsOptions::default()
            },
            ..SolveOptions::default()
        };
        let result = SolverEngine::new(NewtonMethod, options)
            .solve(&ScalarQuadraticProblem, DVector::from_vec(vec![1.5]))
            .expect("solve");

        assert_eq!(result.statistics.residual_evaluations, 0);
        assert_eq!(result.statistics.state_residual_evaluations, 0);
        assert_eq!(result.statistics.trial_residual_evaluations, 0);
        assert_eq!(
            result.statistics.availability,
            StatisticsAvailability::NotCollected
        );
        assert_eq!(result.statistics.jacobian_evaluations, 0);
        assert_eq!(result.statistics.state_jacobian_evaluations, 0);
        assert_eq!(result.statistics.trial_jacobian_evaluations, 0);
        assert_eq!(result.statistics.linear_solves, 0);
        assert_eq!(result.statistics.linear_factorizations, 0);
        assert_eq!(result.statistics.residual_duration, Duration::ZERO);
        assert_eq!(result.statistics.jacobian_duration, Duration::ZERO);
        assert_eq!(result.statistics.linear_solve_duration, Duration::ZERO);
        assert_eq!(
            result.statistics.linear_factorization_duration,
            Duration::ZERO
        );
        assert_eq!(
            result.statistics.linear_system_solve_duration,
            Duration::ZERO
        );
        assert_eq!(result.statistics.total_duration, Duration::ZERO);
        assert!(result.statistics.attempts.is_empty());
    }

    #[test]
    fn linear_stage_timing_preserves_lu_and_inverse_results() {
        // A single 2x2 solve is too short for a portable, non-zero wall-clock
        // sample in optimized builds. Repeat a modest block-diagonal workload
        // so the test checks real accumulated telemetry rather than timer
        // resolution.
        let block_count = 32;
        let dimension = 2 * block_count;
        let mut matrix = DMatrix::zeros(dimension, dimension);
        let mut rhs = DVector::zeros(dimension);
        for block in 0..block_count {
            let row = 2 * block;
            matrix[(row, row)] = 4.0;
            matrix[(row, row + 1)] = 1.0;
            matrix[(row + 1, row)] = 2.0;
            matrix[(row + 1, row + 1)] = 3.0;
            rhs[row] = 9.0;
            rhs[row + 1] = 13.0;
        }

        for solver in [LinearSolverKind::Lu, LinearSolverKind::Inverse] {
            let mut runtime = RuntimeDiagnostics::default();
            let mut result = None;
            for _ in 0..8 {
                result = Some(
                    solve_linear_system_with_stage_timing(
                        solver,
                        &matrix,
                        &rhs,
                        &mut runtime,
                        true,
                    )
                    .expect("nonsingular system"),
                );
            }

            let result = result.expect("at least one timed solve");
            for block in 0..block_count {
                assert!((result[2 * block] - 1.4).abs() < 1e-12);
                assert!((result[2 * block + 1] - 3.4).abs() < 1e-12);
            }
            assert!(runtime.linear_factorization_duration > Duration::ZERO);
            assert!(runtime.linear_system_solve_duration > Duration::ZERO);
            assert_eq!(runtime.linear_factorizations, 8);

            let mut borrowed_runtime = RuntimeDiagnostics::default();
            let borrowed = solve_linear_system_with_stage_timing(
                solver,
                &matrix,
                &rhs,
                &mut borrowed_runtime,
                true,
            )
            .expect("borrowed solve");
            let mut owned_runtime = RuntimeDiagnostics::default();
            let owned = solve_linear_system_owned_with_stage_timing(
                solver,
                matrix.clone(),
                &rhs,
                &mut owned_runtime,
                true,
            )
            .expect("owned solve");
            assert!((borrowed - owned).norm() < 1e-12);
            assert_eq!(owned_runtime.linear_factorizations, 1);
        }
    }

    #[test]
    fn trial_callback_telemetry_matches_instrumented_provider_calls() {
        let problem = CountingScalarQuadraticProblem::new();
        let result = SolverEngine::new(
            crate::numerical::Nonlinear_systems::NR_damped::DampedNewtonMethod::default(),
            SolveOptions::default(),
        )
        .solve(&problem, DVector::from_vec(vec![1.5]))
        .expect("solve");

        assert_eq!(
            result.statistics.residual_evaluations,
            problem.residual_calls.get()
        );
        assert_eq!(
            result.statistics.jacobian_evaluations,
            problem.jacobian_calls.get()
        );
        assert_eq!(
            result.statistics.residual_evaluations,
            result.statistics.state_residual_evaluations
                + result.statistics.trial_residual_evaluations
        );
        assert_eq!(
            result.statistics.jacobian_evaluations,
            result.statistics.state_jacobian_evaluations
                + result.statistics.trial_jacobian_evaluations
        );
        assert!(result.statistics.trial_residual_evaluations > 0);
        assert_eq!(result.statistics.trial_jacobian_evaluations, 0);
        assert_eq!(
            result
                .statistics
                .attempts
                .iter()
                .map(|attempt| attempt.trial_residual_evaluations)
                .sum::<usize>(),
            result.statistics.trial_residual_evaluations
        );
        assert!(result.statistics.attempts.iter().any(|attempt| {
            attempt.trial_residual_evaluations > 0
                && attempt.residual_evaluations
                    == attempt.trial_residual_evaluations + attempt.state_residual_evaluations
        }));
        assert!(result.statistics.residual_evaluations > result.statistics.iterations);
        assert!(result.statistics.residual_duration > Duration::ZERO);
        assert!(result.statistics.jacobian_duration > Duration::ZERO);
    }

    #[test]
    fn workspace_usage_is_reported_at_solver_level() {
        let problem = CountingScalarQuadraticProblem::new();
        let result = SolverEngine::new(
            crate::numerical::Nonlinear_systems::NR_damped::DampedNewtonMethod::default(),
            SolveOptions::default(),
        )
        .solve(&problem, DVector::from_vec(vec![1.5]))
        .expect("solve");

        assert_eq!(
            result.statistics.availability,
            StatisticsAvailability::Collected
        );
        assert!(result.statistics.reusable_trial_points > 0);
        assert!(result.statistics.reusable_trial_points <= result.statistics.residual_evaluations);
    }

    #[test]
    fn repeated_solves_start_with_fresh_attempt_state_and_statistics() {
        let engine = SolverEngine::new(NewtonMethod, SolveOptions::default());
        let first = engine
            .solve(&ScalarQuadraticProblem, DVector::from_vec(vec![1.5]))
            .expect("first solve");
        let second = engine
            .solve(&ScalarQuadraticProblem, DVector::from_vec(vec![1.5]))
            .expect("second solve");

        assert_eq!(first.termination, TerminationReason::Converged);
        assert_eq!(second.termination, TerminationReason::Converged);
        assert_eq!(second.statistics.iterations, first.statistics.iterations);
        assert_eq!(
            second.statistics.residual_evaluations,
            first.statistics.residual_evaluations
        );
        assert_eq!(
            second.statistics.jacobian_evaluations,
            first.statistics.jacobian_evaluations
        );
        assert_eq!(
            second.statistics.linear_solves,
            first.statistics.linear_solves
        );
        assert_eq!(second.history.len(), first.history.len());
        assert!((second.x[0] - first.x[0]).abs() < 1e-12);
    }

    #[test]
    fn diagnostics_can_disable_history_and_collect_memory() {
        let options = SolveOptions {
            diagnostics: DiagnosticsOptions {
                collect_history: false,
                enable_memory_diagnostics: true,
                ..DiagnosticsOptions::default()
            },
            ..SolveOptions::default()
        };
        let result = SolverEngine::new(NewtonMethod, options)
            .solve(&ScalarQuadraticProblem, DVector::from_vec(vec![1.5]))
            .expect("solve");
        assert!(result.history.is_empty());
        assert!(result.memory_diagnostics.is_some());
    }
}
