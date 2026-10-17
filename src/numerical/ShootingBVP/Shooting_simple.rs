//! # Shooting Method for Boundary Value Problems (BVP)
//!
//! This module implements the shooting method to solve second-order boundary value problems
//! of the form: y'' = f(x, y, y') with various boundary condition types.
//!
//! ## Supported Boundary Conditions
//! 1. **Dirichlet-Dirichlet**: y(a) = α, y(b) = β
//! 2. **Dirichlet-Neumann**: y(a) = α, y'(b) = γ  
//! 3. **Neumann-Dirichlet**: y'(a) = α, y(b) = β
//! 4. **Neumann-Neumann**: y'(a) = α, y'(b) = γ
//!
//! ## Method Overview
//! The shooting method converts a BVP into an initial value problem (IVP) by:
//! 1. Guessing the unknown initial condition (y'(a) or y(a))
//! 2. Solving the IVP with known initial condition
//! 3. Adjusting the guess until the boundary condition at x = b is satisfied
//!
//! ## Parameters
//! - **a**: Left boundary point (start of domain)
//! - **b**: Right boundary point (end of domain)  
//! - **left_bc**: Left boundary condition (value, type)
//! - **right_bc**: Right boundary condition (value, type)
//! - **initial_guess**: Initial guess for unknown condition
//! - **tolerance**: Convergence tolerance for secant method
//! - **max_iterations**: Maximum iterations for secant method
//! - **step_size**: Step size for RK4 integration
//!
//! ## Usage Example
//! ```rust, ignore
//! use nalgebra::DVector;
//!use RustedSciThe::numerical::ShootingBVP::Shooting_simple::rk4_ivp_solver;
//! // Define ODE system: y'' = y becomes [y', y]
//! let ode_system = |_x: f64, y: &DVector<f64>| -> DVector<f64> {
//!     DVector::from_vec(vec![y[1], y[0]])
//! };
//!
//! // Set up BVP: y'' = y, y(0) = 0, y'(1) = γ (mixed boundary conditions)
//! let problem = BoundaryValueProblem {
//!     ode_system,
//!     a: 0.0,      // Left boundary
//!     b: 1.0,      // Right boundary  
//!     left_bc: BoundaryCondition { value: 0.0, bc_type: BoundaryConditionType::Dirichlet },  // y(0) = 0
//!     right_bc: BoundaryCondition { value: 1.0, bc_type: BoundaryConditionType::Neumann }, // y'(1) = 1
//! };
//!
//! let solver = ShootingMethodSolver {
//!     initial_guess: 1.0,  // Initial guess for y'(0)
//!     tolerance: 1e-6,
//!     max_iterations: 100,
//!     step_size: 0.01,
//! };
//!
//! let (slope, solution) = solver.solve(&problem, |x0, y0, x_end, h| {
//!     rk4_ivp_solver(x0, y0, x_end, h, &problem.ode_system)
//! }).unwrap();
//! ```

use crate::numerical::SolverTelemetryMode;
use crate::numerical::logging::{
    SolverLogGuard, SolverLogLevel, SolverLoggingError, solver_debug as debug, solver_info as info,
};
use nalgebra::{DMatrix, DVector};
use std::error::Error as StdError;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Boundary condition types
#[derive(Debug, Clone, Copy)]
pub enum BoundaryConditionType {
    /// Dirichlet: y = value
    Dirichlet,
    /// Neumann: y' = value  
    Neumann,
}

/// Boundary condition specification
#[derive(Debug, Clone, Copy)]
pub struct BoundaryCondition {
    pub value: f64,
    pub bc_type: BoundaryConditionType,
}

/// Represents the BVP problem: y'' = f(x, y, y') with various boundary conditions.
pub struct BoundaryValueProblem<F> {
    /// The ODE system in 1st-order form: dy/dx = f(x, y)
    pub ode_system: F,
    pub a: f64,
    pub b: f64,
    pub left_bc: BoundaryCondition,
    pub right_bc: BoundaryCondition,
    // Backward compatibility
    pub alpha: f64,
    pub beta: f64,
}

impl<F> BoundaryValueProblem<F> {
    /// Create BVP with new boundary condition format
    pub fn new(
        ode_system: F,
        a: f64,
        b: f64,
        left_bc: BoundaryCondition,
        right_bc: BoundaryCondition,
    ) -> Self {
        Self {
            ode_system,
            a,
            b,
            left_bc,
            right_bc,
            alpha: left_bc.value,
            beta: right_bc.value,
        }
    }

    /// Create BVP with legacy format (Dirichlet-Dirichlet)
    pub fn legacy(ode_system: F, a: f64, b: f64, alpha: f64, beta: f64) -> Self {
        Self {
            ode_system,
            a,
            b,
            left_bc: BoundaryCondition {
                value: alpha,
                bc_type: BoundaryConditionType::Dirichlet,
            },
            right_bc: BoundaryCondition {
                value: beta,
                bc_type: BoundaryConditionType::Dirichlet,
            },
            alpha,
            beta,
        }
    }
}

/// Configuration for the shooting method solver.
pub struct ShootingMethodSolver {
    pub initial_guess: f64,
    pub tolerance: f64,
    pub max_iterations: usize,
    pub step_size: f64,
    pub result: ShootingMethodResult,
}
#[derive(Debug, Clone)]
pub struct ShootingMethodResult {
    pub x_mesh: DVector<f64>,
    pub y: DMatrix<f64>,
    pub s: f64,
    pub bound_values: DVector<f64>,
}

/// Typed failures from validating or executing one scalar shooting solve.
#[derive(Debug, Error)]
pub enum ShootingBvpError {
    /// The requested Rust file logger could not be initialized.
    #[error("cannot enable ShootingBVP logging: {0}")]
    LoggingSetup(#[source] SolverLoggingError),
    /// A required part of the typed BVP definition was not supplied.
    #[error("shooting problem is missing required field '{field}'")]
    MissingProblemField {
        /// Name of the missing definition field.
        field: &'static str,
    },
    /// A symbolic shooting system is not a two-component first-order system.
    #[error(
        "shooting requires exactly two state variables and two equations, got {states} states and {equations} equations"
    )]
    InvalidProblemDimension { states: usize, equations: usize },
    /// A solver option is outside its valid numeric domain.
    #[error("invalid shooting configuration: {field} must be {requirement}")]
    InvalidConfiguration {
        /// Configuration field that failed validation.
        field: &'static str,
        /// Human-readable domain requirement.
        requirement: &'static str,
    },
    /// The integration interval is reversed or non-finite.
    #[error("invalid shooting interval [{start}, {end}]")]
    InvalidInterval { start: f64, end: f64 },
    /// The IVP provider returned a trajectory with an invalid layout.
    #[error(
        "IVP trajectory shape mismatch: expected 2 rows and matching non-empty mesh, got {rows}x{columns} with {mesh_points} mesh points"
    )]
    InvalidTrajectoryShape {
        /// Number of state rows returned by the IVP provider.
        rows: usize,
        /// Number of solution columns returned by the IVP provider.
        columns: usize,
        /// Number of mesh points returned by the IVP provider.
        mesh_points: usize,
    },
    /// The IVP adapter completed without exposing its trajectory.
    #[error("IVP solver completed without a trajectory result")]
    MissingIvpResult,
    /// An IVP trajectory or boundary residual contains NaN or infinity.
    #[error("non-finite numeric value in shooting stage '{stage}'")]
    NonFiniteValue {
        /// Stage where the invalid value was observed.
        stage: &'static str,
    },
    /// The configured IVP solver failed while evaluating a shooting trial.
    #[error("IVP solve failed during shooting: {0}")]
    IvpFailure(#[source] Box<dyn StdError + Send + Sync>),
    /// Shared symbolic Lambdify preparation failed before numerical solving.
    #[error("symbolic shooting preparation failed: {0}")]
    SymbolicPreparation(#[source] Box<dyn StdError + Send + Sync>),
    /// A prepared Lambdify callback failed while the IVP solver evaluated it.
    #[error("symbolic shooting RHS evaluation failed: {0}")]
    SymbolicEvaluation(#[source] Box<dyn StdError + Send + Sync>),
    /// Secant update could not proceed because its residual denominator vanished.
    #[error("secant method breakdown: consecutive boundary residuals are indistinguishable")]
    SecantBreakdown,
    /// Secant iterations ended before the boundary residual met tolerance.
    #[error(
        "secant method did not converge after {max_iterations} iterations (boundary residual {residual:e})"
    )]
    IterationExhaustion {
        /// Configured iteration budget.
        max_iterations: usize,
        /// Final absolute boundary residual.
        residual: f64,
    },
}

/// Discrete work counters collected for one shooting solve.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShootingTelemetryCounters {
    /// Secant updates attempted after evaluating the initial pair.
    pub secant_iterations: usize,
    /// Boundary residual evaluations, each requiring an IVP solve.
    pub boundary_residual_evaluations: usize,
    /// Total IVP solves, including the final trajectory reconstruction.
    pub ivp_solves: usize,
}

/// Stage durations collected for one shooting solve.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ShootingTelemetryTimings {
    /// Time spent integrating IVPs for boundary residuals.
    pub residual_ivp_time: Duration,
    /// Time spent integrating the final trajectory after finding the shot.
    pub final_ivp_time: Duration,
    /// Total wall time for the secant/search phase, including residual IVPs.
    pub secant_phase_time: Duration,
    /// Total wall time for the complete shooting solve.
    pub total_time: Duration,
}

/// Numerical telemetry snapshot with explicit channel availability.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ShootingTelemetry {
    /// Requested collection policy; use it to distinguish disabled channels
    /// from a measured value that happens to be zero.
    pub mode: SolverTelemetryMode,
    /// Discrete call and iteration counts.
    pub counters: ShootingTelemetryCounters,
    /// Timings for the principal solve stages.
    pub timings: ShootingTelemetryTimings,
}

/// Trajectory plus optional telemetry from the typed shooting API.
#[derive(Clone, Debug)]
pub struct ShootingSolveReport {
    /// Computed trajectory and inferred shooting value.
    pub solution: ShootingMethodResult,
    /// Present only when telemetry collection was explicitly enabled.
    pub telemetry: Option<ShootingTelemetry>,
}

/// Outcome form that preserves opt-in telemetry on both success and failure.
///
/// Use this when diagnosing convergence failures: `telemetry` contains all
/// work completed before the error, while `result` retains the typed cause.
#[derive(Debug)]
pub struct ShootingSolveOutcome {
    /// Successful trajectory or the typed reason the solve stopped.
    pub result: Result<ShootingMethodResult, ShootingBvpError>,
    /// Present only when telemetry collection was explicitly enabled.
    pub telemetry: Option<ShootingTelemetry>,
}
impl Default for ShootingMethodResult {
    fn default() -> Self {
        Self {
            x_mesh: DVector::zeros(0),
            y: DMatrix::zeros(0, 0),
            s: 0.0,
            bound_values: DVector::zeros(0),
        }
    }
}

fn validate_shooting_configuration<F>(
    solver: &ShootingMethodSolver,
    problem: &BoundaryValueProblem<F>,
) -> Result<(), ShootingBvpError> {
    if !problem.a.is_finite() || !problem.b.is_finite() || problem.b < problem.a {
        return Err(ShootingBvpError::InvalidInterval {
            start: problem.a,
            end: problem.b,
        });
    }
    for (field, value) in [
        ("initial_guess", solver.initial_guess),
        ("left boundary value", problem.left_bc.value),
        ("right boundary value", problem.right_bc.value),
    ] {
        if !value.is_finite() {
            return Err(ShootingBvpError::InvalidConfiguration {
                field,
                requirement: "be finite",
            });
        }
    }
    if !solver.tolerance.is_finite() || solver.tolerance <= 0.0 {
        return Err(ShootingBvpError::InvalidConfiguration {
            field: "tolerance",
            requirement: "be finite and greater than zero",
        });
    }
    if solver.max_iterations == 0 {
        return Err(ShootingBvpError::InvalidConfiguration {
            field: "max_iterations",
            requirement: "be greater than zero",
        });
    }
    if !solver.step_size.is_finite() || solver.step_size <= 0.0 {
        return Err(ShootingBvpError::InvalidConfiguration {
            field: "step_size",
            requirement: "be finite and greater than zero",
        });
    }
    Ok(())
}

fn initial_state(left: BoundaryCondition, unknown_value: f64) -> DVector<f64> {
    match left.bc_type {
        BoundaryConditionType::Dirichlet => DVector::from_vec(vec![left.value, unknown_value]),
        BoundaryConditionType::Neumann => DVector::from_vec(vec![unknown_value, left.value]),
    }
}

fn validate_trajectory(
    trajectory: &DMatrix<f64>,
    mesh: &DVector<f64>,
) -> Result<(), ShootingBvpError> {
    if trajectory.nrows() != 2 || trajectory.ncols() == 0 || trajectory.ncols() != mesh.len() {
        return Err(ShootingBvpError::InvalidTrajectoryShape {
            rows: trajectory.nrows(),
            columns: trajectory.ncols(),
            mesh_points: mesh.len(),
        });
    }
    if trajectory.iter().any(|value| !value.is_finite())
        || mesh.iter().any(|value| !value.is_finite())
    {
        return Err(ShootingBvpError::NonFiniteValue {
            stage: "IVP output",
        });
    }
    Ok(())
}

fn evaluate_boundary_residual<F, E>(
    problem: &BoundaryValueProblem<F>,
    guess: f64,
    step_size: f64,
    ivp_solver: &mut impl FnMut(f64, DVector<f64>, f64, f64) -> Result<(DMatrix<f64>, DVector<f64>), E>,
    telemetry: &mut Option<ShootingTelemetry>,
    telemetry_mode: SolverTelemetryMode,
) -> Result<f64, ShootingBvpError>
where
    E: StdError + Send + Sync + 'static,
{
    let started = telemetry_mode.collects_timings().then(Instant::now);
    if telemetry_mode.collects_counters()
        && let Some(metrics) = telemetry.as_mut()
    {
        metrics.counters.ivp_solves += 1;
        metrics.counters.boundary_residual_evaluations += 1;
    }
    let ivp_result = ivp_solver(
        problem.a,
        initial_state(problem.left_bc, guess),
        problem.b,
        step_size,
    );
    if telemetry_mode.collects_timings()
        && let (Some(metrics), Some(started)) = (telemetry.as_mut(), started)
    {
        metrics.timings.residual_ivp_time += started.elapsed();
    }
    let (trajectory, mesh) =
        ivp_result.map_err(|error| ShootingBvpError::IvpFailure(Box::new(error)))?;
    validate_trajectory(&trajectory, &mesh)?;
    let final_state = trajectory.column(trajectory.ncols() - 1);
    let component = match problem.right_bc.bc_type {
        BoundaryConditionType::Dirichlet => 0,
        BoundaryConditionType::Neumann => 1,
    };
    let residual = final_state[component] - problem.right_bc.value;
    if !residual.is_finite() {
        return Err(ShootingBvpError::NonFiniteValue {
            stage: "boundary residual",
        });
    }
    debug!("Shooting residual: guess={guess}, residual={residual}");
    Ok(residual)
}
impl ShootingMethodSolver {
    /// Solves the BVP using the shooting method.
    pub fn solve<F>(
        &mut self,
        problem: &BoundaryValueProblem<F>,
        ivp_solver: impl Fn(f64, DVector<f64>, f64, f64) -> (DMatrix<f64>, DVector<f64>),
    ) -> Result<ShootingMethodResult, &'static str>
    where
        F: Fn(f64, &DVector<f64>) -> DVector<f64>,
    {
        self.solve_with_logging_level(problem, ivp_solver, SolverLogLevel::Off)
    }

    /// Solves the BVP with an explicit per-call logging threshold.
    ///
    /// This scopes messages to this invocation and leaves the application's
    /// process-wide logger untouched.
    pub fn solve_with_logging_level<F>(
        &mut self,
        problem: &BoundaryValueProblem<F>,
        ivp_solver: impl Fn(f64, DVector<f64>, f64, f64) -> (DMatrix<f64>, DVector<f64>),
        logging_level: SolverLogLevel,
    ) -> Result<ShootingMethodResult, &'static str>
    where
        F: Fn(f64, &DVector<f64>) -> DVector<f64>,
    {
        let report = self.try_solve_with_options(
            problem,
            |x0, y0, x_end, step| {
                Ok::<_, std::convert::Infallible>(ivp_solver(x0, y0, x_end, step))
            },
            SolverTelemetryMode::Off,
            logging_level,
        );
        report
            .map(|report| report.solution)
            .map_err(|error| match error {
                ShootingBvpError::IterationExhaustion { .. } => "Secant method did not converge.",
                ShootingBvpError::SecantBreakdown => "Secant method did not converge.",
                ShootingBvpError::NonFiniteValue { .. } => "Secant method did not converge.",
                ShootingBvpError::InvalidTrajectoryShape { .. } => {
                    "Secant method did not converge."
                }
                _ => "Invalid shooting problem or IVP trajectory.",
            })
    }

    /// Solves a second-order BVP through a fallible IVP callback.
    ///
    /// Collection uses the shared four-level policy: `Off`, `Counters`,
    /// `Timings`, or `Detailed`. Counter-only mode never reads the clock, and
    /// timing-only mode never updates discrete counters.
    pub fn try_solve_with_telemetry<F, E>(
        &mut self,
        problem: &BoundaryValueProblem<F>,
        ivp_solver: impl FnMut(f64, DVector<f64>, f64, f64) -> Result<(DMatrix<f64>, DVector<f64>), E>,
        telemetry_mode: SolverTelemetryMode,
    ) -> Result<ShootingSolveReport, ShootingBvpError>
    where
        F: Fn(f64, &DVector<f64>) -> DVector<f64>,
        E: StdError + Send + Sync + 'static,
    {
        self.try_solve_with_options(problem, ivp_solver, telemetry_mode, SolverLogLevel::Off)
    }

    /// Fallible solver entry point with explicit telemetry and logging policies.
    pub fn try_solve_with_options<F, E>(
        &mut self,
        problem: &BoundaryValueProblem<F>,
        ivp_solver: impl FnMut(f64, DVector<f64>, f64, f64) -> Result<(DMatrix<f64>, DVector<f64>), E>,
        telemetry_mode: SolverTelemetryMode,
        logging_level: SolverLogLevel,
    ) -> Result<ShootingSolveReport, ShootingBvpError>
    where
        F: Fn(f64, &DVector<f64>) -> DVector<f64>,
        E: StdError + Send + Sync + 'static,
    {
        let outcome = self.try_solve_detailed_with_options(
            problem,
            ivp_solver,
            telemetry_mode,
            logging_level,
        );
        outcome.result.map(|solution| ShootingSolveReport {
            solution,
            telemetry: outcome.telemetry,
        })
    }

    /// Fallible solve that keeps telemetry available when numerical work fails.
    pub fn try_solve_detailed_with_options<F, E>(
        &mut self,
        problem: &BoundaryValueProblem<F>,
        mut ivp_solver: impl FnMut(
            f64,
            DVector<f64>,
            f64,
            f64,
        ) -> Result<(DMatrix<f64>, DVector<f64>), E>,
        telemetry_mode: SolverTelemetryMode,
        logging_level: SolverLogLevel,
    ) -> ShootingSolveOutcome
    where
        F: Fn(f64, &DVector<f64>) -> DVector<f64>,
        E: StdError + Send + Sync + 'static,
    {
        self.result = ShootingMethodResult::default();
        let total_start = telemetry_mode.collects_timings().then(Instant::now);
        let _logging_scope = match SolverLogGuard::try_enter(logging_level) {
            Ok(scope) => scope,
            Err(error) => {
                let mut telemetry = telemetry_mode.is_enabled().then_some(ShootingTelemetry {
                    mode: telemetry_mode,
                    ..Default::default()
                });
                if let (Some(metrics), Some(start)) = (telemetry.as_mut(), total_start) {
                    metrics.timings.total_time = start.elapsed();
                }
                return ShootingSolveOutcome {
                    result: Err(ShootingBvpError::LoggingSetup(error)),
                    telemetry,
                };
            }
        };
        let mut secant_start = None;
        let mut telemetry = telemetry_mode.is_enabled().then_some(ShootingTelemetry {
            mode: telemetry_mode,
            ..Default::default()
        });
        let result = (|| {
            validate_shooting_configuration(self, problem)?;
            secant_start = telemetry_mode.collects_timings().then(Instant::now);
            info!(
                "Starting typed shooting BVP solve on [{}, {}]",
                problem.a, problem.b
            );
            let second_offset = self.initial_guess.abs().max(1.0) * 0.1;
            let mut previous_guess = self.initial_guess;
            let mut current_guess = self.initial_guess + second_offset;
            let mut previous_residual = evaluate_boundary_residual(
                problem,
                previous_guess,
                self.step_size,
                &mut ivp_solver,
                &mut telemetry,
                telemetry_mode,
            )?;
            if previous_residual.abs() <= self.tolerance {
                current_guess = previous_guess;
            } else {
                let mut current_residual = evaluate_boundary_residual(
                    problem,
                    current_guess,
                    self.step_size,
                    &mut ivp_solver,
                    &mut telemetry,
                    telemetry_mode,
                )?;
                if current_residual.abs() > self.tolerance {
                    let mut converged = false;
                    for iteration in 0..self.max_iterations {
                        let denominator = current_residual - previous_residual;
                        let scale = current_residual.abs().max(previous_residual.abs()).max(1.0);
                        if denominator.abs() <= f64::EPSILON * scale {
                            return Err(ShootingBvpError::SecantBreakdown);
                        }
                        let next_guess = current_guess
                            - current_residual * (current_guess - previous_guess) / denominator;
                        if !next_guess.is_finite() {
                            return Err(ShootingBvpError::NonFiniteValue {
                                stage: "secant update",
                            });
                        }
                        previous_guess = current_guess;
                        previous_residual = current_residual;
                        current_guess = next_guess;
                        if telemetry_mode.collects_counters()
                            && let Some(metrics) = telemetry.as_mut()
                        {
                            metrics.counters.secant_iterations = iteration + 1;
                        }
                        current_residual = evaluate_boundary_residual(
                            problem,
                            current_guess,
                            self.step_size,
                            &mut ivp_solver,
                            &mut telemetry,
                            telemetry_mode,
                        )?;
                        if current_residual.abs() <= self.tolerance {
                            converged = true;
                            break;
                        }
                    }
                    if !converged {
                        return Err(ShootingBvpError::IterationExhaustion {
                            max_iterations: self.max_iterations,
                            residual: current_residual.abs(),
                        });
                    }
                }
            }

            if let (Some(metrics), Some(start)) = (telemetry.as_mut(), secant_start) {
                metrics.timings.secant_phase_time = start.elapsed();
            }

            let initial_state = initial_state(problem.left_bc, current_guess);
            let final_start = telemetry_mode.collects_timings().then(Instant::now);
            if telemetry_mode.collects_counters()
                && let Some(metrics) = telemetry.as_mut()
            {
                metrics.counters.ivp_solves += 1;
            }
            let final_result = ivp_solver(problem.a, initial_state, problem.b, self.step_size);
            if telemetry_mode.collects_timings()
                && let (Some(metrics), Some(start)) = (telemetry.as_mut(), final_start)
            {
                metrics.timings.final_ivp_time = start.elapsed();
            }
            let (trajectory, mesh) =
                final_result.map_err(|error| ShootingBvpError::IvpFailure(Box::new(error)))?;
            validate_trajectory(&trajectory, &mesh)?;
            let boundary_values = trajectory.column(trajectory.ncols() - 1).into_owned();
            let solution = ShootingMethodResult {
                x_mesh: mesh,
                y: trajectory,
                s: current_guess,
                bound_values: boundary_values,
            };
            info!(
                "Shooting BVP converged with inferred initial value {}",
                current_guess
            );
            Ok(solution)
        })();

        if let (Some(metrics), Some(start)) = (telemetry.as_mut(), total_start) {
            metrics.timings.total_time = start.elapsed();
        }
        if let (Some(metrics), Some(start)) = (telemetry.as_mut(), secant_start)
            && metrics.timings.secant_phase_time.is_zero()
        {
            metrics.timings.secant_phase_time = start.elapsed();
        }
        if let Ok(solution) = &result {
            self.result = solution.clone();
        }
        ShootingSolveOutcome { result, telemetry }
    }

    pub fn new() -> Self {
        Self {
            initial_guess: 0.0,
            tolerance: 0.0,
            max_iterations: 0,
            step_size: 0.0,
            result: ShootingMethodResult::default(),
        }
    }
    pub fn get_solution(&self) -> ShootingMethodResult {
        return self.result.clone();
    }
    pub fn get_y(&self) -> DMatrix<f64> {
        return self.result.y.clone();
    }
    pub fn get_x(&self) -> DVector<f64> {
        return self.result.x_mesh.clone();
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
pub fn RK45_ivp_solver_and_mesh<F>(
    x0: f64,
    y0: DVector<f64>,
    x_end: f64,
    step_size: f64,

    ode_system: F,
) -> (DMatrix<f64>, DVector<f64>)
where
    F: Fn(f64, &DVector<f64>) -> DVector<f64>,
{
    let ivp_sol = rk4_ivp_solver(x0, y0, x_end, step_size, ode_system);

    let x_mesh = create_mesh(x0, x_end, step_size);
    (ivp_sol, x_mesh)
}
/// Solves an initial value problem (IVP) using the 4th-order Runge-Kutta method.
///
/// # Arguments
/// * `x0` - Initial x value.
/// * `y0` - Initial state vector (y and y' packed into a DVector).
/// * `x_end` - End of the integration interval.
/// * `step_size` - Step size for numerical integration.
/// * `ode_system` - Function defining the ODE system: dy/dx = f(x, y).
///
/// # Returns
/// * `DMatrix<f64>` - Matrix where each column is the solution at a time step
pub fn rk4_ivp_solver<F>(
    x0: f64,
    y0: DVector<f64>,
    x_end: f64,
    step_size: f64,
    ode_system: F,
) -> DMatrix<f64>
where
    F: Fn(f64, &DVector<f64>) -> DVector<f64>,
{
    // Handle case where no integration is needed
    if (x_end - x0).abs() < f64::EPSILON {
        return DMatrix::from_column_slice(y0.len(), 1, y0.as_slice());
    }

    let n_steps = ((x_end - x0) / step_size).ceil() as usize + 1;
    let mut solution = DMatrix::zeros(y0.len(), n_steps);
    let mut x = x0;
    let mut y = y0.clone();
    let mut step_idx = 0;

    // Store initial condition
    solution.set_column(step_idx, &y);
    step_idx += 1;

    while x < x_end && step_idx < n_steps {
        let h = step_size.min(x_end - x);

        let k1 = ode_system(x, &y);
        let k2 = ode_system(x + h / 2.0, &(y.clone() + (h / 2.0) * &k1));
        let k3 = ode_system(x + h / 2.0, &(y.clone() + (h / 2.0) * &k2));
        let k4 = ode_system(x + h, &(y.clone() + h * &k3));

        y += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        x += h;

        solution.set_column(step_idx, &y);
        step_idx += 1;
    }

    // Return only the used columns

    solution.columns(0, step_idx).into_owned()
}

pub fn create_mesh(x0: f64, x_end: f64, step_size: f64) -> DVector<f64> {
    let n_steps = ((x_end - x0) / step_size).ceil() as usize + 1;
    let mut x_mesh: Vec<f64> = Vec::new();
    let mut x = x0;
    x_mesh.push(x);
    let mut step_idx = 0;
    while x < x_end && step_idx < n_steps {
        let h = step_size.min(x_end - x);
        x += h;
        x_mesh.push(x);
        step_idx += 1;
    }
    let x_mesh = DVector::from_vec(x_mesh);
    x_mesh
}

#[cfg(test)]
mod typed_api_tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::convert::Infallible;

    fn linear_problem() -> BoundaryValueProblem<impl Fn(f64, &DVector<f64>) -> DVector<f64>> {
        BoundaryValueProblem::new(
            |_x: f64, y: &DVector<f64>| DVector::from_vec(vec![y[1], 0.0]),
            0.0,
            1.0,
            BoundaryCondition {
                value: 0.0,
                bc_type: BoundaryConditionType::Dirichlet,
            },
            BoundaryCondition {
                value: 2.0,
                bc_type: BoundaryConditionType::Dirichlet,
            },
        )
    }

    fn exact_linear_ivp(
        start: f64,
        y0: DVector<f64>,
        end: f64,
        _step: f64,
    ) -> Result<(DMatrix<f64>, DVector<f64>), Infallible> {
        let h = end - start;
        let y_end = DVector::from_vec(vec![y0[0] + h * y0[1], y0[1]]);
        Ok((
            DMatrix::from_columns(&[y0, y_end]),
            DVector::from_vec(vec![start, end]),
        ))
    }

    fn solver() -> ShootingMethodSolver {
        ShootingMethodSolver {
            initial_guess: 0.0,
            tolerance: 1e-10,
            max_iterations: 8,
            step_size: 0.1,
            result: ShootingMethodResult::default(),
        }
    }

    #[test]
    fn typed_solve_returns_exact_trajectory_and_opt_in_full_telemetry() {
        let problem = linear_problem();
        let mut solver = solver();
        let report = solver
            .try_solve_with_telemetry(&problem, exact_linear_ivp, SolverTelemetryMode::Detailed)
            .unwrap();

        assert_abs_diff_eq!(report.solution.s, 2.0, epsilon = 1e-9);
        assert_abs_diff_eq!(report.solution.bound_values[0], 2.0, epsilon = 1e-9);
        let telemetry = report.telemetry.unwrap();
        assert_eq!(telemetry.mode, SolverTelemetryMode::Detailed);
        assert_eq!(telemetry.counters.boundary_residual_evaluations, 3);
        assert_eq!(telemetry.counters.ivp_solves, 4);
        assert_eq!(telemetry.counters.secant_iterations, 1);
        assert!(telemetry.timings.total_time >= telemetry.timings.final_ivp_time);
    }

    #[test]
    fn typed_solve_leaves_telemetry_absent_when_disabled() {
        let problem = linear_problem();
        let mut solver = solver();
        let report = solver
            .try_solve_with_telemetry(&problem, exact_linear_ivp, SolverTelemetryMode::Off)
            .unwrap();
        assert!(report.telemetry.is_none());
    }

    #[test]
    fn telemetry_modes_collect_only_the_requested_channel() {
        let problem = linear_problem();
        let mut counter_solver = solver();
        let counters = counter_solver
            .try_solve_with_telemetry(&problem, exact_linear_ivp, SolverTelemetryMode::Counters)
            .unwrap()
            .telemetry
            .unwrap();
        assert_eq!(counters.mode, SolverTelemetryMode::Counters);
        assert!(counters.counters.ivp_solves > 0);
        assert_eq!(counters.timings, ShootingTelemetryTimings::default());

        let mut timing_solver = solver();
        let timings = timing_solver
            .try_solve_with_telemetry(&problem, exact_linear_ivp, SolverTelemetryMode::Timings)
            .unwrap()
            .telemetry
            .unwrap();
        assert_eq!(timings.mode, SolverTelemetryMode::Timings);
        assert_eq!(timings.counters, ShootingTelemetryCounters::default());
        assert!(!timings.timings.total_time.is_zero());
    }

    #[test]
    fn typed_solve_rejects_invalid_configuration_before_calling_ivp() {
        let problem = linear_problem();
        let mut solver = solver();
        solver.tolerance = f64::NAN;
        let mut callback_called = false;
        let result = solver.try_solve_with_telemetry(
            &problem,
            |_, _, _, _| {
                callback_called = true;
                exact_linear_ivp(0.0, DVector::from_vec(vec![0.0, 0.0]), 1.0, 0.1)
            },
            SolverTelemetryMode::Off,
        );

        assert!(matches!(
            result,
            Err(ShootingBvpError::InvalidConfiguration {
                field: "tolerance",
                ..
            })
        ));
        assert!(!callback_called);
    }

    #[test]
    fn typed_solve_rejects_malformed_ivp_trajectory() {
        let problem = linear_problem();
        let mut solver = solver();
        let result = solver.try_solve_with_telemetry(
            &problem,
            |_, _, _, _| Ok::<_, Infallible>((DMatrix::zeros(1, 1), DVector::from_vec(vec![0.0]))),
            SolverTelemetryMode::Off,
        );

        assert!(matches!(
            result,
            Err(ShootingBvpError::InvalidTrajectoryShape { rows: 1, .. })
        ));
    }

    #[derive(Debug, thiserror::Error)]
    #[error("fixture IVP failure")]
    struct FixtureIvpError;

    #[test]
    fn typed_solve_preserves_ivp_error_as_source() {
        let problem = linear_problem();
        let mut solver = solver();
        let result = solver.try_solve_with_telemetry(
            &problem,
            |_, _, _, _| Err(FixtureIvpError),
            SolverTelemetryMode::Detailed,
        );

        let error = result.expect_err("the IVP fixture deliberately fails");
        let source = std::error::Error::source(&error).expect("typed source must be retained");
        assert_eq!(source.to_string(), "fixture IVP failure");
        assert_eq!(solver.get_solution().x_mesh.len(), 0);
    }

    #[test]
    fn detailed_outcome_keeps_secant_breakdown_error_and_partial_telemetry() {
        let problem = linear_problem();
        let mut solver = solver();
        let outcome = solver.try_solve_detailed_with_options(
            &problem,
            |_, _, _, _| {
                Ok::<_, Infallible>((
                    DMatrix::from_column_slice(2, 1, &[3.0, 0.0]),
                    DVector::from_vec(vec![0.0]),
                ))
            },
            SolverTelemetryMode::Detailed,
            SolverLogLevel::Off,
        );

        assert!(matches!(
            outcome.result,
            Err(ShootingBvpError::SecantBreakdown)
        ));
        let telemetry = outcome
            .telemetry
            .expect("enabled telemetry is retained on failure");
        assert_eq!(telemetry.counters.boundary_residual_evaluations, 2);
        assert_eq!(telemetry.counters.ivp_solves, 2);
        assert_eq!(telemetry.counters.secant_iterations, 0);
        assert!(!telemetry.timings.total_time.is_zero());
        assert!(!telemetry.timings.secant_phase_time.is_zero());
        assert_eq!(solver.get_solution().x_mesh.len(), 0);
    }
}

/////////////////////////////////////////////////////////////////////////
//          tests
//////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn second_order_simple_bvp() {
        // Define the ODE system: [y1' = y2, y2' = y1] (for y'' = y)
        let ode_system = |_x: f64, y: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![y[1], y[0]]) // dy/dx = [y2, y1]
        };

        // Set up the BVP: y'' = y, y(0) = 0, y(1) = 1
        let problem = BoundaryValueProblem::legacy(ode_system, 0.0, 1.0, 0.0, 1.0);

        // Configure the solver
        let mut solver = ShootingMethodSolver {
            initial_guess: 1.0,
            tolerance: 1e-6,
            max_iterations: 100,
            step_size: 0.01,
            result: ShootingMethodResult::default(),
        };

        // Solve the BVP using RK4
        match solver.solve(&problem, |x0, y0, x_end, h| {
            RK45_ivp_solver_and_mesh(x0, y0, x_end, h, &problem.ode_system)
        }) {
            Ok(result) => {
                println!("Found initial slope s = {}", result.s);
                println!("solution {:?}", result.bound_values);
                println!(
                    "Solution at x = b: y = {}, y' = {}",
                    result.bound_values[0], result.bound_values[1]
                );
            }
            Err(e) => println!("Error: {}", e),
        }
    }

    #[test]
    fn test_linear_ode() {
        let ode_system = |_x: f64, y: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![y[1], y[0]]) // y'' = y
        };

        let problem = BoundaryValueProblem::legacy(ode_system, 0.0, 1.0, 0.0, 1.0_f64.sinh());

        let mut solver = ShootingMethodSolver {
            initial_guess: 1.0,
            tolerance: 1e-8,
            max_iterations: 100,
            step_size: 0.001,
            result: ShootingMethodResult::default(),
        };

        let result = solver
            .solve(&problem, |x0, y0, x_end, h| {
                RK45_ivp_solver_and_mesh(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();

        // Check boundary condition at x = b
        assert_abs_diff_eq!(result.bound_values[0], problem.beta, epsilon = 1e-6);

        // Check initial slope s ≈ y'(0) = 1 (exact for this problem)
        assert_abs_diff_eq!(result.s, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_non_convergence() {
        let ode_system = |_x: f64, y: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![y[1], y[0]]) // y'' = y
        };

        let problem = BoundaryValueProblem::legacy(ode_system, 0.0, 1.0, 0.0, 1.0);

        // Force failure with unrealistic tolerance and low iterations
        let mut solver = ShootingMethodSolver {
            initial_guess: 0.0,
            tolerance: 1e-20,
            max_iterations: 5,
            step_size: 0.1,
            result: ShootingMethodResult::default(),
        };

        let result = solver.solve(&problem, |x0, y0, x_end, h| {
            RK45_ivp_solver_and_mesh(x0, y0, x_end, h, &problem.ode_system)
        });

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Secant method did not converge.");
    }
    #[test]
    fn test_zero_step() {
        let ode_system = |_x: f64, y: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![y[1], 0.0]) // y'' = 0
        };

        let problem = BoundaryValueProblem::legacy(ode_system, 0.0, 0.0, 0.0, 0.0);

        let mut solver = ShootingMethodSolver {
            initial_guess: 0.0,
            tolerance: 1e-6,
            max_iterations: 10,
            step_size: 0.1,
            result: ShootingMethodResult::default(),
        };

        let result = solver
            .solve(&problem, |x0, y0, x_end, h| {
                RK45_ivp_solver_and_mesh(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();
        println!("boundpoints {:?}", result.bound_values);
        // Solution should remain at initial condition
        assert_eq!(result.bound_values[0], problem.alpha);
        assert_eq!(result.bound_values[1], solver.initial_guess);
    }
    #[test]
    fn test_adaptive_step_size() {
        let ode_system = |_x: f64, y: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![y[1], -y[0]]) // y'' = -y (harmonic oscillator)
        };

        let problem = BoundaryValueProblem::legacy(ode_system, 0.0, std::f64::consts::PI, 0.0, 0.0);

        // Test with coarse step size (should still converge)
        let mut solver = ShootingMethodSolver {
            initial_guess: 1.0,
            tolerance: 1e-4,
            max_iterations: 50,
            step_size: 0.1,
            result: ShootingMethodResult::default(),
        };

        let result = solver
            .solve(&problem, |x0, y0, x_end, h| {
                RK45_ivp_solver_and_mesh(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();

        assert_abs_diff_eq!(result.bound_values[0], problem.beta, epsilon = 1e-3);
    }

    #[test]
    fn test_nonlinear_pendulum() {
        // Nonlinear pendulum: y'' + sin(y) = 0
        let ode_system = |_x: f64, y: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![y[1], -y[0].sin()]) // [y', -sin(y)]
        };

        let problem = BoundaryValueProblem::legacy(ode_system, 0.0, 1.0, 0.0, 0.5);

        let mut solver = ShootingMethodSolver {
            initial_guess: 1.0,
            tolerance: 1e-6,
            max_iterations: 100,
            step_size: 0.01,
            result: ShootingMethodResult::default(),
        };

        let result = solver
            .solve(&problem, |x0, y0, x_end, h| {
                RK45_ivp_solver_and_mesh(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();

        assert_abs_diff_eq!(result.bound_values[0], problem.beta, epsilon = 1e-5);
        assert!(result.s.is_finite());
    }
    // stiff problems is not for shooting method
    #[test]
    #[should_panic]
    fn test_stiff_problem() {
        // Stiff problem: y'' - 100*y = 0
        let ode_system = |_x: f64, y: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![y[1], 100.0 * y[0]])
        };

        let problem = BoundaryValueProblem::legacy(ode_system, 0.0, 0.1, 1.0, (10.0_f64).exp());

        let mut solver = ShootingMethodSolver {
            initial_guess: 10.0,
            tolerance: 1e-6,
            max_iterations: 1000,
            step_size: 1e-5, // Small step for stiff problem
            result: ShootingMethodResult::default(),
        };

        let result = solver
            .solve(&problem, |x0, y0, x_end, h| {
                RK45_ivp_solver_and_mesh(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();
        println!("solution {:?}", result.bound_values);
        assert_abs_diff_eq!(result.bound_values[0], problem.beta, epsilon = 1e-3);
        assert_abs_diff_eq!(result.s, 10.0, epsilon = 1e-3);
    }

    #[test]
    fn test_large_domain() {
        // Simple linear problem over large domain
        let ode_system = |_x: f64, y: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![y[1], 0.0]) // y'' = 0
        };

        let problem = BoundaryValueProblem::legacy(ode_system, 0.0, 100.0, 0.0, 50.0);

        let mut solver = ShootingMethodSolver {
            initial_guess: 0.5,
            tolerance: 1e-6,
            max_iterations: 50,
            step_size: 1.0,
            result: ShootingMethodResult::default(),
        };

        let result = solver
            .solve(&problem, |x0, y0, x_end, h| {
                RK45_ivp_solver_and_mesh(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();
        println!("boundpoints {:?}", result.bound_values);
        assert_abs_diff_eq!(result.bound_values[0], problem.beta, epsilon = 1e-6);
        assert_abs_diff_eq!(result.s, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_negative_domain() {
        // Test with negative domain
        let ode_system = |_x: f64, y: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![y[1], y[0]]) // y'' = y
        };
        //y(-1) = sinh(-1), y(0) =0
        let problem = BoundaryValueProblem::legacy(ode_system, -1.0, 0.0, (-1.0_f64).sinh(), 0.0);

        let mut solver = ShootingMethodSolver {
            initial_guess: -1.0,
            tolerance: 1e-6,
            max_iterations: 100,
            step_size: 0.001,
            result: ShootingMethodResult::default(),
        };

        let _ = solver
            .solve(&problem, |x0, y0, x_end, h| {
                RK45_ivp_solver_and_mesh(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();
        // solution y(x) = 1/2 (e^x - e^(-x))
        let x_mesh = solver.get_x();
        let y_mesh = solver.get_y();
        let y: DVector<f64> = y_mesh.row(0).transpose().into_owned();
        let z: DVector<f64> = y_mesh.row(1).transpose().into_owned();
        for i in 0..x_mesh.len() {
            let exect = (x_mesh[i].exp() - (-x_mesh[i]).exp()) / 2.0;
            assert_abs_diff_eq!(y[i], exect, epsilon = 1e-4);
            let exect_z = (x_mesh[i].exp() + (-x_mesh[i]).exp()) / 2.0;
            assert_abs_diff_eq!(z[i], exect_z, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_oscillatory_solution() {
        // Harmonic oscillator with specific frequency
        let omega = 2.0;
        let ode_system = move |_x: f64, y: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![y[1], -omega * omega * y[0]])
        };

        let problem =
            BoundaryValueProblem::legacy(ode_system, 0.0, std::f64::consts::PI / omega, 1.0, -1.0);

        let mut solver = ShootingMethodSolver {
            initial_guess: 0.0,
            tolerance: 1e-6,
            max_iterations: 100,
            step_size: 0.01,
            result: ShootingMethodResult::default(),
        };

        let result = solver
            .solve(&problem, |x0, y0, x_end, h| {
                RK45_ivp_solver_and_mesh(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();

        assert_abs_diff_eq!(result.bound_values[0], problem.beta, epsilon = 1e-5);
        assert_abs_diff_eq!(result.s, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_dirichlet_neumann() {
        // y'' = y, y(0) = 0, y'(1) = cosh(1)
        let ode_system =
            |_x: f64, y: &DVector<f64>| -> DVector<f64> { DVector::from_vec(vec![y[1], y[0]]) };

        let problem = BoundaryValueProblem::new(
            ode_system,
            0.0,
            1.0,
            BoundaryCondition {
                value: 0.0,
                bc_type: BoundaryConditionType::Dirichlet,
            },
            BoundaryCondition {
                value: 1.0_f64.cosh(),
                bc_type: BoundaryConditionType::Neumann,
            },
        );

        let mut solver = ShootingMethodSolver {
            initial_guess: 1.0,
            tolerance: 1e-8,
            max_iterations: 100,
            step_size: 0.001,
            result: ShootingMethodResult::default(),
        };

        let result = solver
            .solve(&problem, |x0, y0, x_end, h| {
                RK45_ivp_solver_and_mesh(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();
        let x_mesh = solver.get_x();
        let y_mesh = solver.get_y();
        let y: DVector<f64> = y_mesh.row(0).transpose().into_owned();
        for i in 0..x_mesh.len() {
            let exect = (x_mesh[i].exp() - (-x_mesh[i]).exp()) / 2.0;
            assert_abs_diff_eq!(y[i], exect, epsilon = 1e-6);
        }
        assert_abs_diff_eq!(
            result.bound_values[1],
            problem.right_bc.value,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(result.s, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_neumann_dirichlet() {
        // y'' = y, y'(0) = 1, y(1) = sinh(1)
        let ode_system =
            |_x: f64, y: &DVector<f64>| -> DVector<f64> { DVector::from_vec(vec![y[1], y[0]]) };

        let problem = BoundaryValueProblem::new(
            ode_system,
            0.0,
            1.0,
            BoundaryCondition {
                value: 1.0,
                bc_type: BoundaryConditionType::Neumann,
            },
            BoundaryCondition {
                value: 1.0_f64.sinh(),
                bc_type: BoundaryConditionType::Dirichlet,
            },
        );

        let mut solver = ShootingMethodSolver {
            initial_guess: 0.0,
            tolerance: 1e-8,
            max_iterations: 100,
            step_size: 0.001,
            result: ShootingMethodResult::default(),
        };

        let result = solver
            .solve(&problem, |x0, y0, x_end, h| {
                RK45_ivp_solver_and_mesh(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();
        println!("boundpoints {:?}", result.bound_values);
        assert_abs_diff_eq!(
            result.bound_values[0],
            problem.right_bc.value,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(result.s, 0.0, epsilon = 1e-6);
        let x_mesh = solver.get_x();
        let y_mesh = solver.get_y();
        let y: DVector<f64> = y_mesh.row(0).transpose().into_owned();
        for i in 0..x_mesh.len() {
            assert_abs_diff_eq!(y[i], x_mesh[i].sinh(), epsilon = 1e-6);
        }
    }

    #[test]
    fn test_neumann_neumann() {
        // y'' = 0, y'(0) = 1, y'(1) = 1 (linear function)
        let ode_system =
            |_x: f64, y: &DVector<f64>| -> DVector<f64> { DVector::from_vec(vec![y[1], 0.0]) };

        let problem = BoundaryValueProblem::new(
            ode_system,
            0.0,
            1.0,
            BoundaryCondition {
                value: 1.0,
                bc_type: BoundaryConditionType::Neumann,
            },
            BoundaryCondition {
                value: 1.0,
                bc_type: BoundaryConditionType::Neumann,
            },
        );

        let mut solver = ShootingMethodSolver {
            initial_guess: 0.0,
            tolerance: 1e-8,
            max_iterations: 100,
            step_size: 0.01,
            result: ShootingMethodResult::default(),
        };

        let result = solver
            .solve(&problem, |x0, y0, x_end, h| {
                RK45_ivp_solver_and_mesh(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();

        assert_abs_diff_eq!(
            result.bound_values[1],
            problem.right_bc.value,
            epsilon = 1e-6
        );
        assert!(result.s.is_finite());
    }

    #[test]
    fn test_dirichlet_neumann_linear() {
        // y'' = 0, y(0) = 2, y'(1) = 3 (linear function y = 3x + 2)
        let ode_system =
            |_x: f64, y: &DVector<f64>| -> DVector<f64> { DVector::from_vec(vec![y[1], 0.0]) };

        let problem = BoundaryValueProblem::new(
            ode_system,
            0.0,
            1.0,
            BoundaryCondition {
                value: 2.0,
                bc_type: BoundaryConditionType::Dirichlet,
            },
            BoundaryCondition {
                value: 3.0,
                bc_type: BoundaryConditionType::Neumann,
            },
        );

        let mut solver = ShootingMethodSolver {
            initial_guess: 3.0,
            tolerance: 1e-8,
            max_iterations: 50,
            step_size: 0.01,
            result: ShootingMethodResult::default(),
        };

        let result = solver
            .solve(&problem, |x0, y0, x_end, h| {
                RK45_ivp_solver_and_mesh(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();
        println!("boundpoints {:?}", result.bound_values);
        assert_abs_diff_eq!(result.bound_values[1], 3.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.s, 3.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.bound_values[0], 5.0, epsilon = 1e-6); // y(1) = 3*1 + 2 = 5
        let x_mesh = solver.get_x();
        let y_mesh = solver.get_y();
        let y: DVector<f64> = y_mesh.row(0).transpose().into_owned();
        for i in 0..x_mesh.len() {
            assert_abs_diff_eq!(y[i], 3.0 * x_mesh[i] + 2.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_dirichlet_neumann_exponential() {
        // y'' - y = 0, y(0) = 1, y'(1) = e^1 (solution: y = e^x)
        let ode_system =
            |_x: f64, y: &DVector<f64>| -> DVector<f64> { DVector::from_vec(vec![y[1], y[0]]) };

        let problem = BoundaryValueProblem::new(
            ode_system,
            0.0,
            1.0,
            BoundaryCondition {
                value: 1.0,
                bc_type: BoundaryConditionType::Dirichlet,
            },
            BoundaryCondition {
                value: 1.0_f64.exp(),
                bc_type: BoundaryConditionType::Neumann,
            },
        );

        let mut solver = ShootingMethodSolver {
            initial_guess: 1.0,
            tolerance: 1e-8,
            max_iterations: 100,
            step_size: 0.001,
            result: ShootingMethodResult::default(),
        };

        let result = solver
            .solve(&problem, |x0, y0, x_end, h| {
                RK45_ivp_solver_and_mesh(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();

        println!("boundpoints {:?}", result.bound_values);
        assert_abs_diff_eq!(result.bound_values[1], 1.0_f64.exp(), epsilon = 1e-6);
        assert_abs_diff_eq!(result.s, 1.0, epsilon = 1e-6);

        let y: DVector<f64> = result.y.row(0).transpose().into_owned();
        let x_mesh = result.x_mesh.clone();
        assert!(y.len() == x_mesh.len());
        let z: DVector<f64> = result.y.row(1).transpose().into_owned();
        for i in 0..y.len() {
            let error = y[i] - (x_mesh[i].exp());
            assert_abs_diff_eq!(error, 0.0, epsilon = 1e-5);
            let error = z[i] - (x_mesh[i].exp());
            assert_abs_diff_eq!(error, 0.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_dirichlet_neumann_oscillator() {
        // y'' + y = 0, y(0) = 0, y'(π) = 1 (solution: y = -sin(x))
        let ode_system =
            |_x: f64, y: &DVector<f64>| -> DVector<f64> { DVector::from_vec(vec![y[1], -y[0]]) };

        let problem = BoundaryValueProblem::new(
            ode_system,
            0.0,
            std::f64::consts::PI,
            BoundaryCondition {
                value: 0.0,
                bc_type: BoundaryConditionType::Dirichlet,
            },
            BoundaryCondition {
                value: 1.0,
                bc_type: BoundaryConditionType::Neumann,
            },
        );

        let mut solver = ShootingMethodSolver {
            initial_guess: 2.0,
            tolerance: 1e-6,
            max_iterations: 100,
            step_size: 0.01,
            result: ShootingMethodResult::default(),
        };

        let result = solver
            .solve(&problem, |x0, y0, x_end, h| {
                RK45_ivp_solver_and_mesh(x0, y0, x_end, h, &problem.ode_system)
            })
            .unwrap();
        let y: DVector<f64> = result.y.row(0).transpose().into_owned();
        let x_mesh = result.x_mesh.clone();
        assert!(y.len() == x_mesh.len());
        let z: DVector<f64> = result.y.row(1).transpose().into_owned();
        for i in 0..y.len() {
            let error = y[i] - (-x_mesh[i].sin());
            assert_abs_diff_eq!(error, 0.0, epsilon = 1e-5);
            let error = z[i] - (-x_mesh[i].cos());
            assert_abs_diff_eq!(error, 0.0, epsilon = 1e-5);
        }

        println!("boundpoints {:?}", result.bound_values);
        assert_abs_diff_eq!(result.bound_values[0], 0.0, epsilon = 1e-5); //y(pi) = -sin(pi)=0
        assert_abs_diff_eq!(result.bound_values[1], 1.0, epsilon = 1e-5); // y'(pi) = cos(pi) =1
        assert_abs_diff_eq!(result.s, -1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_rk4_exponential_growth() {
        // Test y' = y with y(0) = 1, exact solution: y(x) = e^x
        let ode_system = |_x: f64, y: &DVector<f64>| -> DVector<f64> {
            DVector::from_vec(vec![y[0]]) // dy/dx = y
        };

        let y0 = DVector::from_vec(vec![1.0]);
        let result_matrix = RK45_ivp_solver_and_mesh(0.0, y0, 1.0, 0.01, ode_system).0;
        println!("final result {:?}", &result_matrix);
        let final_result = result_matrix.column(result_matrix.ncols() - 1);
        println!(
            "result matrix shape: ({}, {})",
            result_matrix.nrows(),
            result_matrix.ncols()
        );

        assert_abs_diff_eq!(final_result[0], 1.0_f64.exp(), epsilon = 1e-4);
    }

    #[test]
    fn test_rk4_linear_system() {
        // Test system: y1' = y2, y2' = -y1 (harmonic oscillator)
        // With y1(0) = 1, y2(0) = 0, exact: y1(x) = cos(x), y2(x) = -sin(x)
        let ode_system =
            |_x: f64, y: &DVector<f64>| -> DVector<f64> { DVector::from_vec(vec![y[1], -y[0]]) };

        let y0 = DVector::from_vec(vec![1.0, 0.0]);
        println!("Input y0 dimensions: {}", y0.len());
        let result_matrix =
            RK45_ivp_solver_and_mesh(0.0, y0, std::f64::consts::PI / 2.0, 0.001, ode_system).0;
        let final_result = result_matrix.column(result_matrix.ncols() - 1);
        println!(
            "Result matrix shape: ({}, {})",
            result_matrix.nrows(),
            result_matrix.ncols()
        );
        println!("final result {:?}", final_result);
        assert_eq!(result_matrix.nrows(), 2); // Should have 2 state variables
        assert!(result_matrix.ncols() > 1); // Should have multiple time steps
        assert_abs_diff_eq!(final_result[0], 0.0, epsilon = 1e-6); // y'(pi/2)= cos(π/2) = 0
        assert_abs_diff_eq!(final_result[1], -1.0, epsilon = 1e-6); //y(pi/2) = -sin(π/2) = -1
    }
}
