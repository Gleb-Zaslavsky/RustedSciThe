use crate::numerical::Nonlinear_systems::error::{SolveError, TerminationReason};
use crate::numerical::Nonlinear_systems::problem::{Bounds, JacobianProvider};
use log::{debug, info, warn};
use nalgebra::{DMatrix, DVector};

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
    /// Number of completed nonlinear iterations.
    pub iterations: usize,
    /// Residual evaluations.
    pub residual_evaluations: usize,
    /// Jacobian evaluations.
    pub jacobian_evaluations: usize,
    /// Linear solves performed by the method.
    pub linear_solves: usize,
    /// Accepted trial steps.
    pub accepted_steps: usize,
    /// Rejected trial steps.
    pub rejected_steps: usize,
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
    /// Stop because the method detected convergence.
    Converged,
    /// Stop with an explicit reason.
    Terminated(TerminationReason),
}

/// Mutable counters updated by a concrete method.
#[derive(Debug, Clone, Default)]
pub struct RuntimeDiagnostics {
    /// Linear solves performed by the method.
    pub linear_solves: usize,
    /// Accepted steps.
    pub accepted_steps: usize,
    /// Rejected steps.
    pub rejected_steps: usize,
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
}

/// Plain Newton method.
#[derive(Debug, Clone, Copy, Default)]
pub struct NewtonMethod;

/// Generic outer loop for nonlinear methods.
pub struct SolverEngine<M> {
    method: M,
    options: SolveOptions,
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

        let mut stats = SolveStatistics::default();
        let mut history = Vec::with_capacity(self.options.max_iterations + 1);
        let mut x = x0;
        let mut residual = eval_residual(problem, &x, &mut stats)?;
        let mut jacobian = eval_jacobian(problem, &x, &mut stats)?;
        let mut residual_norm = residual.norm();
        push_history(&self.options, &mut history, 0, residual_norm, 0.0, true);

        if residual_norm < self.options.tolerance {
            return Ok(build_result(
                x,
                residual,
                jacobian,
                residual_norm,
                0,
                TerminationReason::Converged,
                history,
                stats,
                self.options.diagnostics.enable_memory_diagnostics,
            ));
        }

        log_message(
            &self.options,
            EngineLogLevel::Info,
            "starting nonlinear solve",
        );
        let mut method_state =
            self.method
                .init(problem, &x, &self.options, &residual, &jacobian)?;
        let mut runtime = RuntimeDiagnostics::default();

        for iteration in 0..self.options.max_iterations {
            log_message(
                &self.options,
                EngineLogLevel::Debug,
                &format!("iteration {iteration}, residual = {:.6e}", residual_norm),
            );
            let state = IterationState {
                iteration,
                x: x.clone(),
                residual: residual.clone(),
                jacobian: jacobian.clone(),
                residual_norm,
            };

            match self.method.step(
                problem,
                &state,
                &mut method_state,
                &self.options,
                &mut runtime,
            )? {
                StepOutcome::Converged => {
                    stats.iterations = iteration;
                    merge_runtime(&mut stats, &runtime);
                    return Ok(build_result(
                        x,
                        residual,
                        jacobian,
                        residual_norm,
                        iteration,
                        TerminationReason::Converged,
                        history,
                        stats,
                        self.options.diagnostics.enable_memory_diagnostics,
                    ));
                }
                StepOutcome::Terminated(reason) => {
                    stats.iterations = iteration;
                    merge_runtime(&mut stats, &runtime);
                    return Ok(build_result(
                        x,
                        residual,
                        jacobian,
                        residual_norm,
                        iteration,
                        reason,
                        history,
                        stats,
                        self.options.diagnostics.enable_memory_diagnostics,
                    ));
                }
                StepOutcome::Continue { next_x, accepted } => {
                    let next_x = if let Some(bounds) = &self.options.bounds {
                        bounds.project(&next_x)
                    } else {
                        next_x
                    };
                    let step_norm = (&next_x - &x).norm();
                    if accepted {
                        x = next_x;
                        residual = eval_residual(problem, &x, &mut stats)?;
                        jacobian = eval_jacobian(problem, &x, &mut stats)?;
                        residual_norm = residual.norm();
                    }
                    push_history(
                        &self.options,
                        &mut history,
                        iteration + 1,
                        residual_norm,
                        step_norm,
                        accepted,
                    );
                    if residual_norm < self.options.tolerance {
                        stats.iterations = iteration + 1;
                        merge_runtime(&mut stats, &runtime);
                        return Ok(build_result(
                            x,
                            residual,
                            jacobian,
                            residual_norm,
                            iteration + 1,
                            TerminationReason::Converged,
                            history,
                            stats,
                            self.options.diagnostics.enable_memory_diagnostics,
                        ));
                    }
                }
            }
        }

        stats.iterations = self.options.max_iterations;
        merge_runtime(&mut stats, &runtime);
        Ok(build_result(
            x,
            residual,
            jacobian,
            residual_norm,
            self.options.max_iterations,
            TerminationReason::MaxIterations,
            history,
            stats,
            self.options.diagnostics.enable_memory_diagnostics,
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
        let step = solve_linear_system(options.linear_solver, &state.jacobian, &state.residual)?;
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
) -> Result<DVector<f64>, SolveError> {
    stats.residual_evaluations += 1;
    problem.residual(x)
}

/// Evaluates the Jacobian and updates counters.
fn eval_jacobian<P: JacobianProvider>(
    problem: &P,
    x: &DVector<f64>,
    stats: &mut SolveStatistics,
) -> Result<DMatrix<f64>, SolveError> {
    stats.jacobian_evaluations += 1;
    problem.jacobian(x)
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
fn merge_runtime(stats: &mut SolveStatistics, runtime: &RuntimeDiagnostics) {
    stats.linear_solves = runtime.linear_solves;
    stats.accepted_steps = runtime.accepted_steps;
    stats.rejected_steps = runtime.rejected_steps;
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
) -> SolveResult {
    statistics.iterations = iterations;
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

    struct ScalarQuadraticProblem;
    struct CoupledPlainProblem;

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

    #[test]
    fn newton_engine_converges_for_scalar_problem() {
        let result = SolverEngine::new(NewtonMethod, SolveOptions::default())
            .solve(&ScalarQuadraticProblem, DVector::from_vec(vec![1.5]))
            .expect("solve");
        assert_eq!(result.termination, TerminationReason::Converged);
        assert!((result.x[0] - 2.0_f64.sqrt()).abs() < 1e-8);
        assert!(result.statistics.linear_solves > 0);
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
