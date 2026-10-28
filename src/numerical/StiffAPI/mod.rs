//! Typed, thin configuration facade for production stiff IVP solvers.
//!
//! The facade only selects and forwards policy. Numerical work, symbolic
//! preparation, AOT lifecycle, telemetry, logging and solver state remain
//! owned by the selected native implementation.

use std::error::Error;
use std::fmt;
use std::path::PathBuf;

use nalgebra::{DMatrix, DVector};

use crate::Utils::postprocessing::{
    PostprocessDataset, PostprocessError, PostprocessPlan, PostprocessReport,
};
use crate::numerical::BDF::BDF_api::{
    BdfSolveError, BdfSolverOptions, BdfTelemetryMode, ODEsolver as BdfSolver,
};
use crate::numerical::BE::{
    BE, BeError, BeSolverOptions, BeSymbolicAssemblyBackend, BeTelemetryMode,
};
use crate::numerical::LSODE2::{
    Lsode2AotProfile, Lsode2AotToolchain, Lsode2Error, Lsode2LinearSolverChoice,
    Lsode2LinearSolverPolicy, Lsode2LinearSystemStructure, Lsode2NativeStatistics,
    Lsode2ProblemConfig, Lsode2ResidualJacobianSource, Lsode2Solver, Lsode2SymbolicAssemblyBackend,
    Lsode2SymbolicExecutionMode,
};
use crate::numerical::Radau::api::{
    RadauAotConfig, RadauConfig, RadauError, RadauExecution, RadauExecutionPolicy, RadauFrontend,
    RadauMatrixLayout, RadauSolution, RadauSolver, RadauTelemetryReport,
};
use crate::numerical::SolverTelemetryMode;
use crate::numerical::ivp_parameter_loop::{
    IvpParameterLoopPoint, IvpParameterLoopSolution, run_ordered,
};
use crate::numerical::logging::SolverLogLevel;
use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
use crate::symbolic::ivp_telemetry::IvpLambdifyExecutionPolicy;
use crate::symbolic::ivp_telemetry::IvpTelemetrySnapshot;
use crate::symbolic::symbolic_ivp::IvpSymbolicAssemblyBackend;
use crate::symbolic::symbolic_ivp_generated::{
    IvpBackendStatistics, SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
};

/// Solver family selected by an application-level stiff-solver configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StiffSolverKind {
    Bdf,
    Radau,
    BackwardEuler,
    Lsode2,
}

/// Route-aware request that keeps each solver's native model/configuration
/// contract intact while centralizing application-level solver dispatch.
pub enum StiffSolverRequest {
    Bdf {
        options: BdfSolverOptions,
    },
    Radau {
        problem: crate::numerical::Radau::api::RadauProblem,
        config: RadauConfig,
        /// Initial state supplied to Radau's first solve. Other native
        /// adapters keep their initial state in their native configuration.
        initial_state: Vec<f64>,
        /// Ordered value-only parameters declared by `problem`.
        parameters: Vec<f64>,
    },
    BackwardEuler {
        options: BeSolverOptions,
        /// Ordered symbolic parameters declared after native construction.
        equation_parameters: Vec<String>,
        /// Initial values corresponding to `equation_parameters`.
        parameter_values: Vec<f64>,
    },
    /// Preserve LSODE2's independent controller, native/bridge execution,
    /// stop-condition and callback-source configuration.
    Lsode2 {
        config: Lsode2ProblemConfig,
        /// Optional per-request Rust logging threshold, independent of telemetry.
        logging_level: Option<SolverLogLevel>,
    },
}

/// Prepared native solver selected by [`StiffSolverRequest`].
///
/// `try_solve` centralizes first-run dispatch. Native lifecycle,
/// solver-specific results, telemetry and postprocessing remain accessible
/// through typed accessors; the facade does not flatten them into a lossy
/// common representation.
pub enum PreparedStiffSolver {
    Bdf(BdfSolver),
    Radau(PreparedRadau),
    BackwardEuler(BE),
    Lsode2(Lsode2Solver),
}

/// Radau's prepared model and the initial values needed by its first solve.
/// The native solver remains available through [`Self::solver`]; this small
/// adapter only supplies the input Radau requires at execution time.
pub struct PreparedRadau {
    solver: RadauSolver,
    initial_state: Vec<f64>,
    parameters: Vec<f64>,
}

impl PreparedRadau {
    pub fn solver(&self) -> &RadauSolver {
        &self.solver
    }

    pub fn solver_mut(&mut self) -> &mut RadauSolver {
        &mut self.solver
    }
}

/// Solver-native outcome of a common StiffAPI solve dispatch.
///
/// The variants preserve result ownership: Radau returns its dense-output
/// solution, LSODE2 returns its native summary, and BDF/BE retain results on
/// their prepared solver objects.
#[derive(Debug)]
pub enum StiffSolveOutcome {
    Bdf,
    Radau(RadauSolution),
    BackwardEuler,
    Lsode2(crate::numerical::LSODE2::Lsode2SolveSummary),
}

/// Native typed failure from a dispatched solve. The prepared solver remains
/// borrowed by the caller, so native partial results and restart operations
/// remain available after an error.
#[derive(Debug)]
pub enum StiffSolveError {
    Bdf(BdfSolveError),
    Radau(RadauError),
    BackwardEuler(BeError),
    Lsode2(Lsode2Error),
}

/// One independent parameter-vector solve through the common stiff facade.
#[derive(Debug)]
pub struct StiffParameterLoopPoint {
    pub index: usize,
    pub parameter_values: Vec<f64>,
    pub result: Result<IvpParameterLoopSolution, StiffParameterLoopError>,
    /// Solver-native telemetry. `None` means the selected native solver did
    /// not publish a snapshot for this attempted point.
    pub telemetry: Option<StiffTelemetrySnapshot>,
}

/// Failure while solving or materializing one StiffAPI loop point.
#[derive(Debug)]
pub enum StiffParameterLoopError {
    Solve(StiffSolveError),
    Output(StiffPostprocessError),
}

impl fmt::Display for StiffParameterLoopError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Solve(error) => write!(f, "parameter-point solve failed: {error}"),
            Self::Output(error) => write!(f, "parameter-point output failed: {error}"),
        }
    }
}

impl Error for StiffParameterLoopError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Solve(error) => Some(error),
            Self::Output(error) => Some(error),
        }
    }
}

/// Ordered results of an independent stiff-IVP parameter sweep.
#[derive(Debug, Default)]
pub struct StiffParameterLoopReport {
    pub points: Vec<StiffParameterLoopPoint>,
}

impl StiffSolveError {
    pub const fn solver_kind(&self) -> StiffSolverKind {
        match self {
            Self::Bdf(_) => StiffSolverKind::Bdf,
            Self::Radau(_) => StiffSolverKind::Radau,
            Self::BackwardEuler(_) => StiffSolverKind::BackwardEuler,
            Self::Lsode2(_) => StiffSolverKind::Lsode2,
        }
    }
}

impl fmt::Display for StiffSolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bdf(error) => write!(f, "BDF solve failed: {error}"),
            Self::Radau(error) => write!(f, "Radau solve failed: {error}"),
            Self::BackwardEuler(error) => write!(f, "Backward Euler solve failed: {error}"),
            Self::Lsode2(error) => write!(f, "LSODE2 solve failed: {error}"),
        }
    }
}

impl Error for StiffSolveError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Bdf(error) => Some(error),
            Self::Radau(error) => Some(error),
            Self::BackwardEuler(error) => Some(error),
            Self::Lsode2(error) => Some(error),
        }
    }
}

/// Typed failure while resetting a prepared solver to a new initial-value
/// segment. Native solver failures retain their family-specific error type.
#[derive(Debug)]
pub enum StiffRestartError {
    InvalidInterval {
        solver: StiffSolverKind,
    },
    InvalidInitialState {
        solver: StiffSolverKind,
    },
    InitialStateDimension {
        solver: StiffSolverKind,
        expected: usize,
        actual: usize,
    },
    Bdf(BdfSolveError),
    Radau(RadauError),
    BackwardEuler(BeError),
    Lsode2(Lsode2Error),
}

impl fmt::Display for StiffRestartError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInterval { solver } => {
                write!(f, "{solver:?} restart interval must be finite and nonzero")
            }
            Self::InvalidInitialState { solver } => {
                write!(f, "{solver:?} restart state must be nonempty and finite")
            }
            Self::InitialStateDimension {
                solver,
                expected,
                actual,
            } => write!(
                f,
                "{solver:?} restart state has dimension {actual}; expected {expected}"
            ),
            Self::Bdf(error) => write!(f, "BDF restart failed: {error}"),
            Self::Radau(error) => write!(f, "Radau restart failed: {error}"),
            Self::BackwardEuler(error) => write!(f, "Backward Euler restart failed: {error}"),
            Self::Lsode2(error) => write!(f, "LSODE2 restart failed: {error}"),
        }
    }
}

impl Error for StiffRestartError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Bdf(error) => Some(error),
            Self::Radau(error) => Some(error),
            Self::BackwardEuler(error) => Some(error),
            Self::Lsode2(error) => Some(error),
            Self::InvalidInterval { .. }
            | Self::InvalidInitialState { .. }
            | Self::InitialStateDimension { .. } => None,
        }
    }
}

impl PreparedStiffSolver {
    /// Report which native solver was selected by the request.
    pub const fn kind(&self) -> StiffSolverKind {
        match self {
            Self::Bdf(_) => StiffSolverKind::Bdf,
            Self::Radau(_) => StiffSolverKind::Radau,
            Self::BackwardEuler(_) => StiffSolverKind::BackwardEuler,
            Self::Lsode2(_) => StiffSolverKind::Lsode2,
        }
    }

    /// Return the effective native telemetry mode without materializing or
    /// merging solver-specific telemetry reports.
    pub fn telemetry_mode(&self) -> SolverTelemetryMode {
        match self {
            Self::Bdf(solver) => match solver.telemetry_mode() {
                BdfTelemetryMode::Off => SolverTelemetryMode::Off,
                BdfTelemetryMode::Counters => SolverTelemetryMode::Counters,
                BdfTelemetryMode::Timings => SolverTelemetryMode::Timings,
                BdfTelemetryMode::Detailed => SolverTelemetryMode::Detailed,
            },
            Self::Radau(prepared) => match prepared.solver.telemetry_mode() {
                crate::numerical::Radau::api::RadauTelemetryMode::Off => SolverTelemetryMode::Off,
                crate::numerical::Radau::api::RadauTelemetryMode::Counters => {
                    SolverTelemetryMode::Counters
                }
                crate::numerical::Radau::api::RadauTelemetryMode::Timings => {
                    SolverTelemetryMode::Timings
                }
                crate::numerical::Radau::api::RadauTelemetryMode::Detailed => {
                    SolverTelemetryMode::Detailed
                }
            },
            Self::BackwardEuler(solver) => match solver.telemetry_mode() {
                BeTelemetryMode::Off => SolverTelemetryMode::Off,
                BeTelemetryMode::Counters => SolverTelemetryMode::Counters,
                BeTelemetryMode::Timings => SolverTelemetryMode::Timings,
                BeTelemetryMode::Detailed => SolverTelemetryMode::Detailed,
            },
            Self::Lsode2(solver) => match solver.config().telemetry.mode() {
                crate::numerical::LSODE2::IvpTelemetryMode::Off => SolverTelemetryMode::Off,
                crate::numerical::LSODE2::IvpTelemetryMode::Counters => {
                    SolverTelemetryMode::Counters
                }
                crate::numerical::LSODE2::IvpTelemetryMode::Timings => SolverTelemetryMode::Timings,
                crate::numerical::LSODE2::IvpTelemetryMode::Detailed => {
                    SolverTelemetryMode::Detailed
                }
            },
        }
    }

    /// Return the selected native telemetry without flattening solver-specific
    /// counters or timing scopes into a synthetic common schema. Radau keeps
    /// its report on the completed `RadauSolution`, so use
    /// [`StiffTelemetrySnapshot::from_radau_solution`] for that solver.
    pub fn telemetry_snapshot(&self) -> Result<StiffTelemetrySnapshot, StiffTelemetryError> {
        match self {
            Self::Bdf(solver) => Ok(StiffTelemetrySnapshot::Bdf {
                mode: self.telemetry_mode(),
                statistics: solver.get_statistics(),
                symbolic: solver.preparation_backend_telemetry().cloned(),
            }),
            Self::Radau(_) => Err(StiffTelemetryError::RadauSolutionRequired),
            Self::BackwardEuler(solver) => Ok(StiffTelemetrySnapshot::BackwardEuler {
                mode: self.telemetry_mode(),
                statistics: solver.get_statistics(),
                details: solver.detailed_statistics(),
                symbolic: solver.symbolic_ivp_telemetry_snapshot(),
            }),
            Self::Lsode2(solver) => Ok(StiffTelemetrySnapshot::Lsode2 {
                mode: self.telemetry_mode(),
                statistics: solver.statistics(),
                native: solver.native_statistics(),
                symbolic: solver.telemetry_snapshot(),
            }),
        }
    }

    /// Borrow the retained trajectory for BDF, Backward Euler or LSODE2.
    /// Radau returns its sampled output from `solve(y0)` and must instead be
    /// converted from that `RadauSolution` with [`radau_postprocess_dataset`].
    pub fn trajectory_view(&self) -> Result<StiffTrajectoryView<'_>, StiffPostprocessError> {
        let (solver, axis, values) = match self {
            Self::Bdf(solver) => {
                if solver.status_kind() == crate::numerical::BDF::BDF_api::BdfStatus::Running {
                    return Err(StiffPostprocessError::ResultUnavailable {
                        solver: StiffSolverKind::Bdf,
                    });
                }
                let (axis, values) = solver.get_result_ref();
                (StiffSolverKind::Bdf, axis, values)
            }
            Self::Radau(_) => return Err(StiffPostprocessError::RadauSolutionRequired),
            Self::BackwardEuler(solver) => {
                let (axis, values) = solver.trajectory();
                (StiffSolverKind::BackwardEuler, axis, values)
            }
            Self::Lsode2(solver) => {
                if solver.status() == "running" {
                    return Err(StiffPostprocessError::ResultUnavailable {
                        solver: StiffSolverKind::Lsode2,
                    });
                }
                let (axis, values) = solver.get_result_ref();
                (StiffSolverKind::Lsode2, axis, values)
            }
        };
        if axis.is_empty() || values.nrows() != axis.len() {
            return Err(StiffPostprocessError::ResultUnavailable { solver });
        }
        Ok(StiffTrajectoryView {
            solver,
            axis,
            values,
        })
    }

    /// Borrow the native BDF solver when selected.
    pub fn as_bdf(&self) -> Option<&BdfSolver> {
        match self {
            Self::Bdf(solver) => Some(solver),
            Self::Radau(_) | Self::BackwardEuler(_) | Self::Lsode2(_) => None,
        }
    }

    /// Mutably borrow the native BDF solver when selected.
    pub fn as_bdf_mut(&mut self) -> Option<&mut BdfSolver> {
        match self {
            Self::Bdf(solver) => Some(solver),
            Self::Radau(_) | Self::BackwardEuler(_) | Self::Lsode2(_) => None,
        }
    }

    /// Borrow the native Radau solver when this request selected Radau.
    pub fn as_radau(&self) -> Option<&RadauSolver> {
        match self {
            Self::Bdf(_) => None,
            Self::Radau(prepared) => Some(&prepared.solver),
            Self::BackwardEuler(_) | Self::Lsode2(_) => None,
        }
    }

    /// Mutably borrow the native Radau solver when this request selected Radau.
    pub fn as_radau_mut(&mut self) -> Option<&mut RadauSolver> {
        match self {
            Self::Bdf(_) => None,
            Self::Radau(prepared) => Some(&mut prepared.solver),
            Self::BackwardEuler(_) | Self::Lsode2(_) => None,
        }
    }

    /// Borrow the native Backward Euler solver when selected.
    pub fn as_backward_euler(&self) -> Option<&BE> {
        match self {
            Self::Bdf(_) => None,
            Self::Radau(_) => None,
            Self::BackwardEuler(solver) => Some(solver),
            Self::Lsode2(_) => None,
        }
    }

    /// Mutably borrow the native Backward Euler solver when selected.
    pub fn as_backward_euler_mut(&mut self) -> Option<&mut BE> {
        match self {
            Self::Bdf(_) => None,
            Self::Radau(_) => None,
            Self::BackwardEuler(solver) => Some(solver),
            Self::Lsode2(_) => None,
        }
    }

    /// Borrow the native LSODE2 solver when selected.
    pub fn as_lsode2(&self) -> Option<&Lsode2Solver> {
        match self {
            Self::Lsode2(solver) => Some(solver),
            Self::Bdf(_) | Self::Radau(_) | Self::BackwardEuler(_) => None,
        }
    }

    /// Mutably borrow the native LSODE2 solver when selected.
    pub fn as_lsode2_mut(&mut self) -> Option<&mut Lsode2Solver> {
        match self {
            Self::Lsode2(solver) => Some(solver),
            Self::Bdf(_) | Self::Radau(_) | Self::BackwardEuler(_) => None,
        }
    }

    /// Run the solver selected during preparation without application-side
    /// solver dispatch. Native results and failures remain solver-specific.
    /// On failure, this object remains available for status inspection and
    /// native restart/recovery through the typed accessors.
    pub fn try_solve(&mut self) -> Result<StiffSolveOutcome, StiffSolveError> {
        match self {
            Self::Bdf(solver) => {
                solver.try_solve().map_err(StiffSolveError::Bdf)?;
                Ok(StiffSolveOutcome::Bdf)
            }
            Self::Radau(prepared) => {
                let solution = prepared
                    .solver
                    .try_solve_with_parameter_values(&prepared.initial_state, &prepared.parameters)
                    .map_err(StiffSolveError::Radau)?;
                Ok(StiffSolveOutcome::Radau(solution))
            }
            Self::BackwardEuler(solver) => {
                solver.try_solve().map_err(StiffSolveError::BackwardEuler)?;
                Ok(StiffSolveOutcome::BackwardEuler)
            }
            Self::Lsode2(solver) => solver
                .solve_with_summary()
                .map(StiffSolveOutcome::Lsode2)
                .map_err(StiffSolveError::Lsode2),
        }
    }

    /// Starts a fresh solve with new parameter values and initial state while
    /// keeping the currently configured time interval. Native prepared
    /// callbacks and AOT artifacts are reused. This is not parameter
    /// continuation from the previous solution.
    pub fn try_solve_with_parameter_values(
        &mut self,
        y0: DVector<f64>,
        values: DVector<f64>,
    ) -> Result<StiffSolveOutcome, StiffSolveError> {
        match self {
            Self::Bdf(solver) => solver
                .try_solve_with_parameter_values(y0, values)
                .map(|()| StiffSolveOutcome::Bdf)
                .map_err(StiffSolveError::Bdf),
            Self::Radau(prepared) => {
                prepared.initial_state = y0.iter().copied().collect();
                prepared.parameters = values.iter().copied().collect();
                prepared
                    .solver
                    .try_solve_with_parameter_values(&prepared.initial_state, &prepared.parameters)
                    .map(StiffSolveOutcome::Radau)
                    .map_err(StiffSolveError::Radau)
            }
            Self::BackwardEuler(solver) => solver
                .try_solve_with_parameter_values(y0, values)
                .map(|()| StiffSolveOutcome::BackwardEuler)
                .map_err(StiffSolveError::BackwardEuler),
            Self::Lsode2(solver) => solver
                .try_solve_with_parameter_values(y0, values)
                .map(StiffSolveOutcome::Lsode2)
                .map_err(StiffSolveError::Lsode2),
        }
    }

    /// Solve each parameter vector independently from the same initial state.
    ///
    /// The prepared solver and symbolic/AOT callbacks are reused, while each
    /// point starts a fresh integration over the already configured interval.
    /// No previous terminal state is used as a guess. Native typed failures
    /// and telemetry are kept in the corresponding point and do not stop the
    /// remaining vectors.
    pub fn try_solve_parameter_loop(
        &mut self,
        y0: DVector<f64>,
        parameter_vectors: &[Vec<f64>],
    ) -> StiffParameterLoopReport {
        let report = run_ordered(parameter_vectors, |parameters| {
            let outcome = self
                .try_solve_with_parameter_values(y0.clone(), DVector::from_column_slice(parameters))
                .map_err(StiffParameterLoopError::Solve);
            let result = outcome.and_then(|outcome| match outcome {
                StiffSolveOutcome::Radau(solution) => {
                    let (x_mesh, values) = solution.output_trajectory().map_err(|error| {
                        StiffParameterLoopError::Solve(StiffSolveError::Radau(error))
                    })?;
                    Ok(IvpParameterLoopSolution { x_mesh, values })
                }
                StiffSolveOutcome::Bdf
                | StiffSolveOutcome::BackwardEuler
                | StiffSolveOutcome::Lsode2(_) => {
                    let view = self
                        .trajectory_view()
                        .map_err(StiffParameterLoopError::Output)?;
                    Ok(IvpParameterLoopSolution {
                        x_mesh: view.axis.clone(),
                        values: view.values.clone(),
                    })
                }
            });
            let telemetry = match self {
                Self::Bdf(_) | Self::BackwardEuler(_) | Self::Lsode2(_) => {
                    self.telemetry_snapshot().ok()
                }
                Self::Radau(prepared) => prepared
                    .solver
                    .last_solve_telemetry()
                    .cloned()
                    .map(StiffTelemetrySnapshot::Radau),
            };
            (result, telemetry)
        });
        StiffParameterLoopReport {
            points: report
                .points
                .into_iter()
                .map(
                    |point: IvpParameterLoopPoint<
                        Option<StiffTelemetrySnapshot>,
                        StiffParameterLoopError,
                    >| {
                        StiffParameterLoopPoint {
                            index: point.index,
                            parameter_values: point.parameter_values,
                            result: point.result,
                            telemetry: point.telemetry,
                        }
                    },
                )
                .collect(),
        }
    }

    /// Starts a fresh solve with new parameter values, initial state and time
    /// interval. The prepared symbolic/AOT route is retained for the next run.
    pub fn try_solve_with_parameter_values_in_interval(
        &mut self,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        values: DVector<f64>,
    ) -> Result<StiffSolveOutcome, StiffSolveError> {
        match self {
            Self::Bdf(solver) => solver
                .try_solve_with_parameter_values_in_interval(t0, y0, t_bound, values)
                .map(|()| StiffSolveOutcome::Bdf)
                .map_err(StiffSolveError::Bdf),
            Self::Radau(prepared) => {
                prepared.initial_state = y0.iter().copied().collect();
                prepared.parameters = values.iter().copied().collect();
                prepared
                    .solver
                    .try_solve_with_parameter_values_in_interval(
                        t0,
                        &prepared.initial_state,
                        t_bound,
                        &prepared.parameters,
                    )
                    .map(StiffSolveOutcome::Radau)
                    .map_err(StiffSolveError::Radau)
            }
            Self::BackwardEuler(solver) => solver
                .try_solve_with_parameter_values_in_interval(t0, y0, t_bound, values)
                .map(|()| StiffSolveOutcome::BackwardEuler)
                .map_err(StiffSolveError::BackwardEuler),
            Self::Lsode2(solver) => solver
                .try_solve_with_parameter_values_in_interval(t0, y0, t_bound, values)
                .map(StiffSolveOutcome::Lsode2)
                .map_err(StiffSolveError::Lsode2),
        }
    }

    /// Restart the prepared native route on a new initial-value segment while
    /// retaining its symbolic callbacks, generated artifacts and solver policy.
    pub fn try_restart_with_initial_state(
        &mut self,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
    ) -> Result<(), StiffRestartError> {
        let solver = self.kind();
        if !t0.is_finite() || !t_bound.is_finite() || t0 == t_bound {
            return Err(StiffRestartError::InvalidInterval { solver });
        }
        if y0.is_empty() || y0.iter().any(|value| !value.is_finite()) {
            return Err(StiffRestartError::InvalidInitialState { solver });
        }

        match self {
            Self::Bdf(native) => native
                .try_restart_with_initial_state(t0, y0, t_bound)
                .map_err(StiffRestartError::Bdf),
            Self::Radau(prepared) => {
                if y0.len() != prepared.initial_state.len() {
                    return Err(StiffRestartError::InitialStateDimension {
                        solver,
                        expected: prepared.initial_state.len(),
                        actual: y0.len(),
                    });
                }
                prepared
                    .solver
                    .restart(t0, t_bound)
                    .map_err(StiffRestartError::Radau)?;
                prepared.initial_state = y0.iter().copied().collect();
                Ok(())
            }
            Self::BackwardEuler(native) => native
                .try_restart_with_initial_state(t0, y0, t_bound)
                .map_err(StiffRestartError::BackwardEuler),
            Self::Lsode2(native) => native
                .try_restart_with_initial_state(t0, y0, t_bound)
                .map_err(StiffRestartError::Lsode2),
        }
    }
}

/// Solver-native telemetry returned through the common StiffAPI boundary.
///
/// The variants intentionally retain each solver's distinct statistics
/// surface. Timings may be nested or inclusive according to native scope
/// metadata; consumers must not sum them unless their native documentation
/// explicitly says they are disjoint.
#[derive(Debug, Clone)]
pub enum StiffTelemetrySnapshot {
    Bdf {
        mode: SolverTelemetryMode,
        statistics: IvpBackendStatistics,
        symbolic: Option<IvpTelemetrySnapshot>,
    },
    BackwardEuler {
        mode: SolverTelemetryMode,
        statistics: IvpBackendStatistics,
        details: crate::numerical::BE::BeDetailedStatistics,
        symbolic: Option<IvpTelemetrySnapshot>,
    },
    Lsode2 {
        mode: SolverTelemetryMode,
        statistics: IvpBackendStatistics,
        native: Lsode2NativeStatistics,
        symbolic: IvpTelemetrySnapshot,
    },
    Radau(RadauTelemetryReport),
}

impl StiffTelemetrySnapshot {
    /// Preserve the telemetry attached to a completed Radau solve result.
    pub fn from_radau_solution(solution: &RadauSolution) -> Self {
        Self::Radau(solution.telemetry().clone())
    }
}

/// Why a prepared solver cannot yet provide its native telemetry report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StiffTelemetryError {
    /// Radau creates its telemetry report together with the solve result.
    RadauSolutionRequired,
}

impl fmt::Display for StiffTelemetryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RadauSolutionRequired => write!(
                f,
                "Radau telemetry is available from the RadauSolution returned by solve"
            ),
        }
    }
}

impl Error for StiffTelemetryError {}

/// Borrowed, row-oriented time-series data retained by a prepared solver.
///
/// `values` has shape `(time_points, state_variables)`. Creating an owned
/// [`PostprocessDataset`] from this view is explicit because it copies both
/// arrays into the dataset's ownership.
pub struct StiffTrajectoryView<'a> {
    pub solver: StiffSolverKind,
    pub axis: &'a DVector<f64>,
    pub values: &'a DMatrix<f64>,
}

impl StiffTrajectoryView<'_> {
    /// Copy the borrowed trajectory into the owned common postprocessing type.
    pub fn to_dataset(
        &self,
        axis_name: impl Into<String>,
        variable_names: Vec<String>,
    ) -> Result<PostprocessDataset, StiffPostprocessError> {
        PostprocessDataset::new(
            axis_name,
            variable_names,
            self.axis.clone(),
            self.values.clone(),
        )
        .map_err(StiffPostprocessError::InvalidDataset)
    }

    /// Materialize the dataset and run the caller's explicit output plan.
    pub fn execute(
        &self,
        axis_name: impl Into<String>,
        variable_names: Vec<String>,
        plan: &PostprocessPlan,
    ) -> Result<PostprocessReport, StiffPostprocessError> {
        let dataset = self.to_dataset(axis_name, variable_names)?;
        plan.execute(&dataset)
            .map_err(|source| StiffPostprocessError::PlanExecution {
                solver: self.solver,
                source,
            })
    }
}

/// Convert Radau's native dense-output samples into an owned common dataset.
/// Sampling times and variable labels are explicit because Radau retains its
/// interpolation result separately from the prepared solver and does not
/// promise the other solvers' accepted-step trajectory layout.
pub fn radau_postprocess_dataset(
    solution: &crate::numerical::Radau::api::RadauSolution,
    sample_times: &[f64],
    axis_name: impl Into<String>,
    variable_names: Vec<String>,
) -> Result<PostprocessDataset, StiffPostprocessError> {
    let component_count = solution.y.len();
    let flattened = solution
        .sample_many(sample_times)
        .map_err(StiffPostprocessError::RadauSampling)?;
    let expected_len = sample_times
        .len()
        .checked_mul(component_count)
        .ok_or_else(|| {
            StiffPostprocessError::InvalidDataset(PostprocessError::InvalidDataset(
                "Radau sample dimensions overflow addressable memory".to_string(),
            ))
        })?;
    if flattened.len() != expected_len {
        return Err(StiffPostprocessError::InvalidDataset(
            PostprocessError::InvalidDataset(format!(
                "Radau returned {} sampled values; expected {expected_len}",
                flattened.len()
            )),
        ));
    }
    let values = DMatrix::from_row_slice(sample_times.len(), component_count, &flattened);
    PostprocessDataset::new(
        axis_name,
        variable_names,
        DVector::from_column_slice(sample_times),
        values,
    )
    .map_err(StiffPostprocessError::InvalidDataset)
}

/// Typed failures from the facade's explicit trajectory materialization and
/// postprocessing helpers. Native solver state remains available after a plan
/// fails; these operations never convert a successful solve into a solve error.
#[derive(Debug)]
pub enum StiffPostprocessError {
    ResultUnavailable {
        solver: StiffSolverKind,
    },
    RadauSolutionRequired,
    RadauSampling(RadauError),
    InvalidDataset(PostprocessError),
    PlanExecution {
        solver: StiffSolverKind,
        source: PostprocessError,
    },
}

impl fmt::Display for StiffPostprocessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ResultUnavailable { solver } => {
                write!(f, "{solver:?} has no retained trajectory to postprocess")
            }
            Self::RadauSolutionRequired => {
                f.write_str("Radau postprocessing requires the RadauSolution returned by solve(y0)")
            }
            Self::RadauSampling(error) => write!(f, "Radau output sampling failed: {error}"),
            Self::InvalidDataset(error) => write!(f, "invalid postprocessing dataset: {error}"),
            Self::PlanExecution { solver, source } => {
                write!(f, "{solver:?} postprocessing plan failed: {source}")
            }
        }
    }
}

impl Error for StiffPostprocessError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::RadauSampling(error) => Some(error),
            Self::InvalidDataset(error) | Self::PlanExecution { source: error, .. } => Some(error),
            Self::ResultUnavailable { .. } | Self::RadauSolutionRequired => None,
        }
    }
}

impl StiffSolverRequest {
    /// Apply the common profile and prepare the selected native implementation.
    pub fn prepare(
        self,
        preset: Option<&StiffPreset>,
    ) -> Result<PreparedStiffSolver, StiffApiError> {
        match self {
            Self::Bdf { options } => prepare_bdf(options, preset).map(PreparedStiffSolver::Bdf),
            Self::Radau {
                problem,
                config,
                initial_state,
                parameters,
            } => prepare_radau(problem, config, preset).map(|solver| {
                PreparedStiffSolver::Radau(PreparedRadau {
                    solver,
                    initial_state,
                    parameters,
                })
            }),
            Self::BackwardEuler {
                options,
                equation_parameters,
                parameter_values,
            } => {
                let mut solver = prepare_backward_euler(options, preset)?;
                if !equation_parameters.is_empty() {
                    let names = equation_parameters
                        .iter()
                        .map(String::as_str)
                        .collect::<Vec<_>>();
                    solver
                        .try_set_equation_parameters(Some(&names))
                        .map_err(StiffApiError::BackwardEuler)?;
                }
                if !parameter_values.is_empty() {
                    solver
                        .try_set_parameter_values(DVector::from_vec(parameter_values))
                        .map_err(StiffApiError::BackwardEuler)?;
                }
                Ok(PreparedStiffSolver::BackwardEuler(solver))
            }
            Self::Lsode2 {
                config,
                logging_level,
            } => {
                let mut solver = prepare_lsode2(config, preset)?;
                if let Some(level) = preset.and_then(|preset| preset.logging).or(logging_level) {
                    solver.set_logging_level(level);
                }
                Ok(PreparedStiffSolver::Lsode2(solver))
            }
        }
    }
}

/// Common symbolic frontend choice, translated to each native solver's type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StiffFrontend {
    ExprLegacy,
    AtomViewNative,
}

/// Common matrix-layout request. A solver must reject layouts it does not
/// natively support; the facade never densifies a sparse or banded request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StiffMatrixLayout {
    Dense,
    Sparse,
    /// Ask the selected solver to infer bandwidth from its prepared Jacobian
    /// structure, but only when that solver natively supports this policy.
    BandedAuto,
    /// Supply an explicit bandwidth hint when a solver supports it.
    /// Prefer `BandedAuto` when native symbolic inference is available; native
    /// routes validate explicit hints so they cannot truncate known structure.
    Banded {
        lower: usize,
        upper: usize,
    },
}

/// Symbolic execution route requested by a reusable preset.
#[derive(Debug, Clone)]
pub enum StiffExecution {
    Lambdify,
    Aot(SymbolicIvpGeneratedBackendConfig),
}

/// Callback scheduling requested through a solver preset.
///
/// This maps to native callback controls only for routes that expose them
/// (currently Radau and Lambdify callback routes in BDF/LSODE2). Other routes,
/// including Backward Euler and AOT callback routes without such a control,
/// reject it explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StiffCallbackPolicy {
    Sequential,
    Parallel { min_work: usize },
    Auto { min_work: usize },
}

/// Static applicability of a composable preset to one solver family.
///
/// `Conditional` means the selected preset can be applied only when the
/// request's native source/execution route meets the accompanying constraint.
/// Final validation still happens against the actual request before prepare.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StiffPresetApplicability {
    Supported,
    Conditional,
    Unsupported,
}

/// One solver's compatibility entry in a [`StiffPresetDescription`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StiffSolverApplicability {
    pub solver: StiffSolverKind,
    pub applicability: StiffPresetApplicability,
    pub constraint: Option<&'static str>,
}

/// Inspectable summary of the facade policy axes, tools and solver constraints
/// declared by a preset. Field names identify StiffAPI axes, not necessarily
/// fields in each solver's native config. This metadata is not a substitute
/// for request validation or native error reporting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StiffPresetDescription {
    pub fields_changed: Vec<&'static str>,
    /// External tools needed only when the selected AOT policy builds an
    /// artifact instead of loading an already-built one.
    pub tools_if_aot_build_needed: Vec<String>,
    pub required_state_dimension: Option<usize>,
    pub solvers: Vec<StiffSolverApplicability>,
}

/// Static metadata for a named preset constructor. External tools are needed
/// only if the configured AOT artifact must be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StiffPresetCatalogEntry {
    pub name: &'static str,
    pub purpose: &'static str,
    pub fields_changed: &'static [&'static str],
    pub tools_if_aot_build_needed: &'static [&'static str],
    pub requires_output_directory: bool,
}

const PRESET_CATALOG: &[StiffPresetCatalogEntry] = &[
    StiffPresetCatalogEntry {
        name: "solve_1d",
        purpose: "One-state ExprLegacy + Dense; native numerical defaults",
        fields_changed: &[
            "symbolic_frontend",
            "linear_system_layout",
            "state_dimension",
        ],
        tools_if_aot_build_needed: &[],
        requires_output_directory: false,
    },
    StiffPresetCatalogEntry {
        name: "expr_dense",
        purpose: "ExprLegacy frontend with Dense storage",
        fields_changed: &["symbolic_frontend", "linear_system_layout"],
        tools_if_aot_build_needed: &[],
        requires_output_directory: false,
    },
    StiffPresetCatalogEntry {
        name: "atom_dense",
        purpose: "AtomViewNative frontend with Dense storage",
        fields_changed: &["symbolic_frontend", "linear_system_layout"],
        tools_if_aot_build_needed: &[],
        requires_output_directory: false,
    },
    StiffPresetCatalogEntry {
        name: "expr_sparse",
        purpose: "ExprLegacy frontend with Sparse storage where supported",
        fields_changed: &["symbolic_frontend", "linear_system_layout"],
        tools_if_aot_build_needed: &[],
        requires_output_directory: false,
    },
    StiffPresetCatalogEntry {
        name: "atom_sparse",
        purpose: "AtomViewNative frontend with Sparse storage where supported",
        fields_changed: &["symbolic_frontend", "linear_system_layout"],
        tools_if_aot_build_needed: &[],
        requires_output_directory: false,
    },
    StiffPresetCatalogEntry {
        name: "expr_banded_auto",
        purpose: "ExprLegacy frontend with inferred Banded storage",
        fields_changed: &["symbolic_frontend", "linear_system_layout"],
        tools_if_aot_build_needed: &[],
        requires_output_directory: false,
    },
    StiffPresetCatalogEntry {
        name: "atom_banded_auto",
        purpose: "AtomViewNative frontend with inferred Banded storage",
        fields_changed: &["symbolic_frontend", "linear_system_layout"],
        tools_if_aot_build_needed: &[],
        requires_output_directory: false,
    },
    StiffPresetCatalogEntry {
        name: "lambdify",
        purpose: "Explicitly request the Lambdify symbolic route",
        fields_changed: &["symbolic_execution"],
        tools_if_aot_build_needed: &[],
        requires_output_directory: false,
    },
    StiffPresetCatalogEntry {
        name: "sequential_callbacks",
        purpose: "Request sequential callback evaluation where supported",
        fields_changed: &["callback_execution_policy"],
        tools_if_aot_build_needed: &[],
        requires_output_directory: false,
    },
    StiffPresetCatalogEntry {
        name: "parallel_callbacks",
        purpose: "Request parallel callbacks with a caller-supplied work threshold",
        fields_changed: &["callback_execution_policy"],
        tools_if_aot_build_needed: &[],
        requires_output_directory: false,
    },
    StiffPresetCatalogEntry {
        name: "automatic_callbacks",
        purpose: "Delegate callback parallelization to the native Auto policy",
        fields_changed: &["callback_execution_policy"],
        tools_if_aot_build_needed: &[],
        requires_output_directory: false,
    },
    StiffPresetCatalogEntry {
        name: "aot_tcc",
        purpose: "Build missing release AOT artifacts with C+tcc",
        fields_changed: &[
            "symbolic_execution",
            "aot_codegen_backend",
            "aot_c_compiler",
            "aot_build_policy",
            "aot_build_profile",
            "aot_artifact_location",
        ],
        tools_if_aot_build_needed: &["cargo", "rustc", "tcc"],
        requires_output_directory: true,
    },
    StiffPresetCatalogEntry {
        name: "aot_gcc",
        purpose: "Build missing release AOT artifacts with C+GCC",
        fields_changed: &[
            "symbolic_execution",
            "aot_codegen_backend",
            "aot_c_compiler",
            "aot_build_policy",
            "aot_build_profile",
            "aot_artifact_location",
        ],
        tools_if_aot_build_needed: &["cargo", "rustc", "gcc"],
        requires_output_directory: true,
    },
    StiffPresetCatalogEntry {
        name: "aot_zig",
        purpose: "Build missing release AOT artifacts with Zig",
        fields_changed: &[
            "symbolic_execution",
            "aot_codegen_backend",
            "aot_c_compiler",
            "aot_build_policy",
            "aot_build_profile",
            "aot_artifact_location",
        ],
        tools_if_aot_build_needed: &["cargo", "rustc", "zig"],
        requires_output_directory: true,
    },
    StiffPresetCatalogEntry {
        name: "aot_rust",
        purpose: "Build missing release AOT artifacts with Rust codegen",
        fields_changed: &[
            "symbolic_execution",
            "aot_codegen_backend",
            "aot_c_compiler",
            "aot_build_policy",
            "aot_build_profile",
            "aot_artifact_location",
        ],
        tools_if_aot_build_needed: &["cargo", "rustc"],
        requires_output_directory: true,
    },
];

/// A composable set of common solver-policy overrides.
///
/// An unset field leaves the selected solver's native value untouched. Compose
/// independent profiles explicitly; a second profile cannot silently replace
/// an axis already set by the first.
#[derive(Debug, Clone, Default)]
pub struct StiffPreset {
    frontend: Option<StiffFrontend>,
    matrix_layout: Option<StiffMatrixLayout>,
    execution: Option<StiffExecution>,
    callback_policy: Option<StiffCallbackPolicy>,
    telemetry: Option<SolverTelemetryMode>,
    logging: Option<SolverLogLevel>,
    required_state_dimension: Option<usize>,
}

impl StiffPreset {
    /// Return the stable names and metadata for built-in preset constructors.
    /// Parameterized callback/AOT constructors are represented by templates;
    /// their actual thresholds, artifact paths and applicability are available
    /// from [`Self::describe`] on the constructed preset.
    pub fn catalog() -> &'static [StiffPresetCatalogEntry] {
        PRESET_CATALOG
    }

    /// ExprLegacy frontend and dense storage profile.
    ///
    /// This selects no solver family and changes no tolerances, stepping
    /// controls, integration interval or solver-specific defaults.
    pub const fn expr_dense() -> Self {
        Self {
            frontend: Some(StiffFrontend::ExprLegacy),
            matrix_layout: Some(StiffMatrixLayout::Dense),
            execution: None,
            callback_policy: None,
            telemetry: None,
            logging: None,
            required_state_dimension: None,
        }
    }

    /// AtomViewNative frontend with Dense storage.
    pub const fn atom_dense() -> Self {
        Self {
            frontend: Some(StiffFrontend::AtomViewNative),
            matrix_layout: Some(StiffMatrixLayout::Dense),
            execution: None,
            callback_policy: None,
            telemetry: None,
            logging: None,
            required_state_dimension: None,
        }
    }

    /// ExprLegacy frontend with Sparse storage where supported.
    pub const fn expr_sparse() -> Self {
        Self {
            frontend: Some(StiffFrontend::ExprLegacy),
            matrix_layout: Some(StiffMatrixLayout::Sparse),
            execution: None,
            callback_policy: None,
            telemetry: None,
            logging: None,
            required_state_dimension: None,
        }
    }

    /// AtomViewNative frontend with Sparse storage where supported.
    pub const fn atom_sparse() -> Self {
        Self {
            frontend: Some(StiffFrontend::AtomViewNative),
            matrix_layout: Some(StiffMatrixLayout::Sparse),
            execution: None,
            callback_policy: None,
            telemetry: None,
            logging: None,
            required_state_dimension: None,
        }
    }

    /// ExprLegacy frontend with symbolically inferred Banded storage.
    pub const fn expr_banded_auto() -> Self {
        Self {
            frontend: Some(StiffFrontend::ExprLegacy),
            matrix_layout: Some(StiffMatrixLayout::BandedAuto),
            execution: None,
            callback_policy: None,
            telemetry: None,
            logging: None,
            required_state_dimension: None,
        }
    }

    /// AtomViewNative frontend with symbolically inferred Banded storage.
    pub const fn atom_banded_auto() -> Self {
        Self {
            frontend: Some(StiffFrontend::AtomViewNative),
            matrix_layout: Some(StiffMatrixLayout::BandedAuto),
            execution: None,
            callback_policy: None,
            telemetry: None,
            logging: None,
            required_state_dimension: None,
        }
    }

    /// Require an explicit Lambdify route.
    pub const fn lambdify() -> Self {
        Self {
            frontend: None,
            matrix_layout: None,
            execution: Some(StiffExecution::Lambdify),
            callback_policy: None,
            telemetry: None,
            logging: None,
            required_state_dimension: None,
        }
    }

    /// One-state quick profile: ExprLegacy + Dense, with all numerical solver
    /// controls left at their native defaults. The request must contain one
    /// equation and one initial state before any solver preparation begins.
    pub const fn solve_1d() -> Self {
        let mut preset = Self::expr_dense();
        preset.required_state_dimension = Some(1);
        preset
    }

    /// Require AOT and configure a release C+tcc build when the artifact is
    /// missing. The artifact directory is owned by the native AOT lifecycle.
    pub fn aot_tcc(output_parent_dir: impl Into<PathBuf>) -> Self {
        let generated =
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_c_tcc();
        Self::aot(generated)
    }

    /// Require a release C+GCC build when an AOT artifact is missing.
    pub fn aot_gcc(output_parent_dir: impl Into<PathBuf>) -> Self {
        let generated =
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_c_gcc();
        Self::aot(generated)
    }

    /// Require a release Zig build when an AOT artifact is missing.
    pub fn aot_zig(output_parent_dir: impl Into<PathBuf>) -> Self {
        let generated =
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_zig();
        Self::aot(generated)
    }

    /// Require a Rust-codegen release AOT build when an artifact is missing.
    pub fn aot_rust(output_parent_dir: impl Into<PathBuf>) -> Self {
        let generated =
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_rust();
        Self::aot(generated)
    }

    fn aot(generated: SymbolicIvpGeneratedBackendConfig) -> Self {
        Self {
            frontend: None,
            matrix_layout: None,
            execution: Some(StiffExecution::Aot(generated)),
            callback_policy: None,
            telemetry: None,
            logging: None,
            required_state_dimension: None,
        }
    }

    pub const fn with_frontend(mut self, frontend: StiffFrontend) -> Self {
        self.frontend = Some(frontend);
        self
    }

    pub const fn with_matrix_layout(mut self, layout: StiffMatrixLayout) -> Self {
        self.matrix_layout = Some(layout);
        self
    }

    pub fn with_execution(mut self, execution: StiffExecution) -> Self {
        self.execution = Some(execution);
        self
    }

    /// Select a native residual/Jacobian callback scheduling policy where
    /// exposed by the selected solver and execution route.
    pub const fn with_callback_policy(mut self, policy: StiffCallbackPolicy) -> Self {
        self.callback_policy = Some(policy);
        self
    }

    /// Sequential callback evaluation profile.
    pub fn sequential_callbacks() -> Self {
        Self::default().with_callback_policy(StiffCallbackPolicy::Sequential)
    }

    /// Force parallel callback evaluation after the native minimum-work gate.
    pub fn parallel_callbacks(min_work: usize) -> Self {
        Self::default().with_callback_policy(StiffCallbackPolicy::Parallel { min_work })
    }

    /// Let the native callback evaluator decide when parallel dispatch pays off.
    pub fn automatic_callbacks(min_work: usize) -> Self {
        Self::default().with_callback_policy(StiffCallbackPolicy::Auto { min_work })
    }

    pub const fn with_telemetry(mut self, telemetry: SolverTelemetryMode) -> Self {
        self.telemetry = Some(telemetry);
        self
    }

    pub const fn with_logging(mut self, logging: SolverLogLevel) -> Self {
        self.logging = Some(logging);
        self
    }

    /// Combine profiles only when they configure disjoint axes.
    pub fn compose(self, other: Self) -> Result<Self, StiffApiError> {
        macro_rules! merge_axis {
            ($left:expr, $right:expr, $name:literal) => {
                match ($left, $right) {
                    (Some(_), Some(_)) => {
                        return Err(StiffApiError::ConflictingPresetAxis($name));
                    }
                    (Some(value), None) | (None, Some(value)) => Some(value),
                    (None, None) => None,
                }
            };
        }

        Ok(Self {
            frontend: merge_axis!(self.frontend, other.frontend, "frontend"),
            matrix_layout: merge_axis!(self.matrix_layout, other.matrix_layout, "matrix_layout"),
            execution: merge_axis!(self.execution, other.execution, "execution"),
            callback_policy: merge_axis!(
                self.callback_policy,
                other.callback_policy,
                "callback_policy"
            ),
            telemetry: merge_axis!(self.telemetry, other.telemetry, "telemetry"),
            logging: merge_axis!(self.logging, other.logging, "logging"),
            required_state_dimension: merge_axis!(
                self.required_state_dimension,
                other.required_state_dimension,
                "required_state_dimension"
            ),
        })
    }

    /// Describe the exact common axes configured by this profile, its AOT
    /// tool requirements, and solver-specific applicability constraints.
    ///
    /// The request adapter remains authoritative because some constraints
    /// depend on native input (for example, symbolic versus analytical
    /// callbacks) that is not stored in a reusable preset.
    pub fn describe(&self) -> StiffPresetDescription {
        let mut fields_changed = Vec::new();
        if self.frontend.is_some() {
            fields_changed.push("symbolic_frontend");
        }
        if self.matrix_layout.is_some() {
            fields_changed.push("linear_system_layout");
        }
        if let Some(execution) = &self.execution {
            match execution {
                StiffExecution::Lambdify => fields_changed.push("symbolic_execution"),
                StiffExecution::Aot(_) => fields_changed.extend([
                    "symbolic_execution",
                    "aot_codegen_backend",
                    "aot_c_compiler",
                    "aot_build_policy",
                    "aot_build_profile",
                    "aot_artifact_location",
                ]),
            }
        }
        if self.callback_policy.is_some() {
            fields_changed.push("callback_execution_policy");
        }
        if self.telemetry.is_some() {
            fields_changed.push("telemetry_mode");
        }
        if self.logging.is_some() {
            fields_changed.push("logging_level");
        }
        if self.required_state_dimension.is_some() {
            fields_changed.push("state_dimension");
        }

        let mut tools_if_aot_build_needed = Vec::new();
        if let Some(StiffExecution::Aot(generated)) = &self.execution {
            if matches!(
                generated.build_policy,
                SymbolicIvpAotBuildPolicy::BuildIfMissing { .. }
                    | SymbolicIvpAotBuildPolicy::RebuildAlways { .. }
            ) {
                tools_if_aot_build_needed.push("cargo".to_string());
                tools_if_aot_build_needed.push("rustc".to_string());
                match generated.aot_codegen_backend {
                    crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend::C => {
                        tools_if_aot_build_needed.push(
                            generated
                                .aot_c_compiler
                                .clone()
                                .unwrap_or_else(|| "configured C compiler".to_string()),
                        );
                    }
                    crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend::Zig => {
                        tools_if_aot_build_needed.push("zig".to_string());
                    }
                    crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend::Rust => {}
                }
            }
        }

        StiffPresetDescription {
            fields_changed,
            tools_if_aot_build_needed,
            required_state_dimension: self.required_state_dimension,
            solvers: [
                StiffSolverKind::Bdf,
                StiffSolverKind::Radau,
                StiffSolverKind::BackwardEuler,
                StiffSolverKind::Lsode2,
            ]
            .into_iter()
            .map(|solver| self.applicability_for(solver))
            .collect(),
        }
    }

    fn applicability_for(&self, solver: StiffSolverKind) -> StiffSolverApplicability {
        let mut conditional = None;
        let mut unsupported = None;
        match solver {
            StiffSolverKind::Bdf => {
                if self
                    .matrix_layout
                    .is_some_and(|layout| layout != StiffMatrixLayout::Dense)
                {
                    unsupported = Some("BDF supports Dense linear algebra only");
                }
                if self.callback_policy.is_some() {
                    if matches!(self.execution.as_ref(), Some(StiffExecution::Aot(_))) {
                        unsupported =
                            Some("BDF callback scheduling is unavailable for AOT callbacks");
                    } else {
                        conditional = Some("requires effective Lambdify callbacks");
                    }
                }
            }
            StiffSolverKind::Radau => {
                if self
                    .logging
                    .is_some_and(|level| level != SolverLogLevel::Off)
                {
                    unsupported = Some("RadauConfig does not expose a logging-level control");
                }
            }
            StiffSolverKind::BackwardEuler => {
                if self
                    .matrix_layout
                    .is_some_and(|layout| layout != StiffMatrixLayout::Dense)
                {
                    unsupported = Some("Backward Euler supports Dense linear algebra only");
                }
                if self.callback_policy.is_some() {
                    unsupported = Some("Backward Euler has no callback scheduling control");
                }
                if matches!(self.execution.as_ref(), Some(StiffExecution::Lambdify)) {
                    unsupported = Some("Backward Euler cannot force Lambdify execution");
                }
            }
            StiffSolverKind::Lsode2 => {
                if self.frontend.is_some() {
                    conditional =
                        Some("requires an LSODE2 symbolic source, not analytical callbacks");
                }
                if self.matrix_layout == Some(StiffMatrixLayout::BandedAuto) {
                    conditional = Some("BandedAuto requires symbolic Jacobian structure");
                }
                if self.callback_policy.is_some() {
                    if matches!(self.execution.as_ref(), Some(StiffExecution::Aot(_))) {
                        unsupported =
                            Some("LSODE2 callback scheduling is unavailable for AOT callbacks");
                    } else {
                        conditional = Some("requires the LSODE2 Lambdify callback route");
                    }
                }
            }
        }

        let (applicability, constraint) = if let Some(reason) = unsupported {
            (StiffPresetApplicability::Unsupported, Some(reason))
        } else if let Some(reason) = conditional {
            (StiffPresetApplicability::Conditional, Some(reason))
        } else {
            (StiffPresetApplicability::Supported, None)
        };
        StiffSolverApplicability {
            solver,
            applicability,
            constraint,
        }
    }

    fn apply_radau(&self, config: &mut RadauConfig) -> Result<(), StiffApiError> {
        if let Some(level) = self.logging
            && level != SolverLogLevel::Off
        {
            return Err(StiffApiError::UnsupportedCapability {
                solver: StiffSolverKind::Radau,
                capability: "logging control is not exposed by the current RadauConfig",
            });
        }
        if let Some(frontend) = self.frontend {
            config.frontend = match frontend {
                StiffFrontend::ExprLegacy => RadauFrontend::ExprLegacy,
                StiffFrontend::AtomViewNative => RadauFrontend::AtomViewNative,
            };
        }
        if let Some(policy) = self.callback_policy {
            config.execution_policy = match policy {
                StiffCallbackPolicy::Sequential => RadauExecutionPolicy::Sequential,
                StiffCallbackPolicy::Parallel { min_work } => {
                    RadauExecutionPolicy::Parallel { min_work }
                }
                StiffCallbackPolicy::Auto { min_work } => RadauExecutionPolicy::Auto { min_work },
            };
        }
        if let Some(layout) = self.matrix_layout {
            config.matrix_layout = match layout {
                StiffMatrixLayout::Dense => RadauMatrixLayout::Dense,
                StiffMatrixLayout::Sparse => RadauMatrixLayout::Sparse,
                StiffMatrixLayout::BandedAuto => RadauMatrixLayout::BandedAuto,
                StiffMatrixLayout::Banded { lower, upper } => {
                    RadauMatrixLayout::Banded { lower, upper }
                }
            };
        }
        if let Some(telemetry) = self.telemetry {
            config.telemetry = match telemetry {
                SolverTelemetryMode::Off => crate::numerical::Radau::api::RadauTelemetryMode::Off,
                SolverTelemetryMode::Counters => {
                    crate::numerical::Radau::api::RadauTelemetryMode::Counters
                }
                SolverTelemetryMode::Timings => {
                    crate::numerical::Radau::api::RadauTelemetryMode::Timings
                }
                SolverTelemetryMode::Detailed => {
                    crate::numerical::Radau::api::RadauTelemetryMode::Detailed
                }
            };
        }
        if let Some(execution) = &self.execution {
            match execution {
                StiffExecution::Lambdify => {
                    config.execution = RadauExecution::Lambdify;
                    config.aot = None;
                }
                StiffExecution::Aot(generated) => {
                    config.execution = RadauExecution::Aot;
                    config.aot = Some(RadauAotConfig {
                        generated: generated.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    fn apply_bdf(&self, options: &mut BdfSolverOptions) -> Result<(), StiffApiError> {
        if let Some(layout) = self.matrix_layout
            && layout != StiffMatrixLayout::Dense
        {
            return Err(StiffApiError::UnsupportedCapability {
                solver: StiffSolverKind::Bdf,
                capability: "BDF currently supports dense linear algebra only",
            });
        }
        if let Some(frontend) = self.frontend {
            options.symbolic_assembly_backend = match frontend {
                StiffFrontend::ExprLegacy => IvpSymbolicAssemblyBackend::ExprLegacy,
                StiffFrontend::AtomViewNative => IvpSymbolicAssemblyBackend::AtomView,
            };
        }
        if let Some(policy) = self.callback_policy {
            options.lambdify_execution_policy = match policy {
                StiffCallbackPolicy::Sequential => {
                    crate::symbolic::ivp_telemetry::IvpLambdifyExecutionPolicy::Sequential
                }
                StiffCallbackPolicy::Parallel { min_work } => {
                    crate::symbolic::ivp_telemetry::IvpLambdifyExecutionPolicy::Parallel {
                        min_work,
                    }
                }
                StiffCallbackPolicy::Auto { min_work } => {
                    crate::symbolic::ivp_telemetry::IvpLambdifyExecutionPolicy::Auto { min_work }
                }
            };
        }
        if let Some(telemetry) = self.telemetry {
            options.telemetry_mode = match telemetry {
                SolverTelemetryMode::Off => BdfTelemetryMode::Off,
                SolverTelemetryMode::Counters => BdfTelemetryMode::Counters,
                SolverTelemetryMode::Timings => BdfTelemetryMode::Timings,
                SolverTelemetryMode::Detailed => BdfTelemetryMode::Detailed,
            };
        }
        if let Some(level) = self.logging {
            options.logging_level = level;
        }
        if let Some(execution) = &self.execution {
            match execution {
                StiffExecution::Lambdify => {
                    // BDF selects generated artifacts through its resolver.
                    // Clear both registry sources to make this explicit preset
                    // an actual Lambdify request rather than a preference.
                    options.generated_backend_config.resolver = None;
                    options.generated_backend_config.handoff_path = None;
                    options.generated_backend_config.build_policy =
                        SymbolicIvpAotBuildPolicy::UseIfAvailable;
                }
                StiffExecution::Aot(generated) => {
                    if self.callback_policy.is_some() {
                        return Err(StiffApiError::UnsupportedCapability {
                            solver: StiffSolverKind::Bdf,
                            capability: "BDF callback scheduling controls apply only to Lambdify, not AOT callbacks",
                        });
                    }
                    if generated.build_policy == SymbolicIvpAotBuildPolicy::UseIfAvailable {
                        return Err(StiffApiError::UnsupportedCapability {
                            solver: StiffSolverKind::Bdf,
                            capability: "AOT selection must use RequirePrebuilt, BuildIfMissing or RebuildAlways; UseIfAvailable permits Lambdify fallback",
                        });
                    }
                    options.generated_backend_config = generated.clone();
                }
            }
        }
        Ok(())
    }

    fn apply_be(&self, options: &mut BeSolverOptions) -> Result<(), StiffApiError> {
        if self.callback_policy.is_some() {
            return Err(StiffApiError::UnsupportedCapability {
                solver: StiffSolverKind::BackwardEuler,
                capability: "callback scheduling policy is not exposed by Backward Euler",
            });
        }
        if let Some(layout) = self.matrix_layout
            && layout != StiffMatrixLayout::Dense
        {
            return Err(StiffApiError::UnsupportedCapability {
                solver: StiffSolverKind::BackwardEuler,
                capability: "only dense linear algebra is implemented by Backward Euler",
            });
        }
        if let Some(level) = self.logging {
            options.logging_level = level;
        }
        if let Some(frontend) = self.frontend {
            options.symbolic_assembly_backend = match frontend {
                StiffFrontend::ExprLegacy => BeSymbolicAssemblyBackend::ExprLegacy,
                StiffFrontend::AtomViewNative => BeSymbolicAssemblyBackend::AtomViewNative,
            };
        }
        if let Some(telemetry) = self.telemetry {
            options.telemetry_mode = match telemetry {
                SolverTelemetryMode::Off => BeTelemetryMode::Off,
                SolverTelemetryMode::Counters => BeTelemetryMode::Counters,
                SolverTelemetryMode::Timings => BeTelemetryMode::Timings,
                SolverTelemetryMode::Detailed => BeTelemetryMode::Detailed,
            };
        }
        if let Some(execution) = &self.execution {
            match execution {
                StiffExecution::Lambdify => {
                    return Err(StiffApiError::UnsupportedCapability {
                        solver: StiffSolverKind::BackwardEuler,
                        capability: "forcing Lambdify is unavailable; the native generated-backend resolver may select a registered AOT artifact",
                    });
                }
                StiffExecution::Aot(generated) => {
                    if generated.build_policy == SymbolicIvpAotBuildPolicy::UseIfAvailable {
                        return Err(StiffApiError::UnsupportedCapability {
                            solver: StiffSolverKind::BackwardEuler,
                            capability: "AOT selection must use RequirePrebuilt, BuildIfMissing or RebuildAlways; UseIfAvailable permits Lambdify fallback",
                        });
                    }
                    options.generated_backend_config = generated.clone();
                }
            }
        }
        Ok(())
    }

    fn apply_lsode2(&self, config: &mut Lsode2ProblemConfig) -> Result<(), StiffApiError> {
        if let Some(frontend) = self.frontend {
            let execution = match config.residual_jacobian_source {
                Lsode2ResidualJacobianSource::Symbolic { execution, .. } => execution,
                Lsode2ResidualJacobianSource::Analytical => {
                    return Err(unsupported_lsode2(
                        "a symbolic frontend preset cannot replace analytical callbacks",
                    ));
                }
            };
            config.residual_jacobian_source = Lsode2ResidualJacobianSource::Symbolic {
                assembly: match frontend {
                    StiffFrontend::ExprLegacy => Lsode2SymbolicAssemblyBackend::ExprLegacy,
                    StiffFrontend::AtomViewNative => Lsode2SymbolicAssemblyBackend::AtomView,
                },
                execution,
            };
            config.sync_legacy_backend_from_policy();
        }

        if let Some(layout) = self.matrix_layout {
            let structure = match layout {
                StiffMatrixLayout::Dense => Lsode2LinearSystemStructure::Dense,
                StiffMatrixLayout::Sparse => Lsode2LinearSystemStructure::Sparse,
                StiffMatrixLayout::Banded { lower, upper } => Lsode2LinearSystemStructure::Banded {
                    kl: lower,
                    ku: upper,
                },
                StiffMatrixLayout::BandedAuto => {
                    if !matches!(
                        config.residual_jacobian_source,
                        Lsode2ResidualJacobianSource::Symbolic { .. }
                    ) {
                        return Err(unsupported_lsode2(
                            "automatic band inference requires symbolic Jacobian structure",
                        ));
                    }
                    Lsode2LinearSystemStructure::BandedAuto
                }
            };
            let expected_solver = match layout {
                StiffMatrixLayout::Dense => Lsode2LinearSolverChoice::DenseLu,
                StiffMatrixLayout::Sparse => Lsode2LinearSolverChoice::FaerSparseLu,
                StiffMatrixLayout::BandedAuto | StiffMatrixLayout::Banded { .. } => {
                    Lsode2LinearSolverChoice::LapackFaithfulBandedLu
                }
            };
            if let Lsode2LinearSolverPolicy::Force(choice) = config.linear_solver_policy
                && choice != expected_solver
            {
                return Err(unsupported_lsode2(
                    "requested matrix layout conflicts with the explicitly forced native linear solver",
                ));
            }
            config.linear_system_structure = structure;
            config.sync_legacy_backend_from_policy();
        }

        if let Some(execution) = &self.execution {
            let assembly = match config.residual_jacobian_source {
                Lsode2ResidualJacobianSource::Symbolic { assembly, .. } => assembly,
                Lsode2ResidualJacobianSource::Analytical => {
                    return Err(unsupported_lsode2(
                        "symbolic execution presets cannot replace analytical callbacks",
                    ));
                }
            };
            let mode = match execution {
                StiffExecution::Lambdify => Lsode2SymbolicExecutionMode::LambdifyExpr,
                StiffExecution::Aot(generated) => {
                    if generated.build_policy == SymbolicIvpAotBuildPolicy::UseIfAvailable {
                        return Err(unsupported_lsode2(
                            "AOT selection must prohibit silent Lambdify fallback",
                        ));
                    }
                    let toolchain = match generated.aot_codegen_backend {
                        crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend::C => {
                            match generated.aot_c_compiler.as_deref() {
                                Some("tcc") => Lsode2AotToolchain::CTcc,
                                Some("gcc") => Lsode2AotToolchain::CGcc,
                                _ => {
                                    return Err(unsupported_lsode2(
                                        "LSODE2 common AOT presets support C+tcc, C+gcc, Zig and Rust toolchains",
                                    ));
                                }
                            }
                        }
                        crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend::Zig => {
                            Lsode2AotToolchain::Zig
                        }
                        crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend::Rust => {
                            Lsode2AotToolchain::Rust
                        }
                    };
                    let profile = match generated.build_policy {
                        SymbolicIvpAotBuildPolicy::RequirePrebuilt => Lsode2AotProfile::Release,
                        SymbolicIvpAotBuildPolicy::BuildIfMissing { profile }
                        | SymbolicIvpAotBuildPolicy::RebuildAlways { profile } => match profile {
                            AotBuildProfile::Debug => Lsode2AotProfile::Debug,
                            AotBuildProfile::Release => Lsode2AotProfile::Release,
                        },
                        SymbolicIvpAotBuildPolicy::UseIfAvailable => unreachable!(),
                    };
                    config.backend.generated_backend = generated.clone();
                    Lsode2SymbolicExecutionMode::Aot { toolchain, profile }
                }
            };
            config.residual_jacobian_source = Lsode2ResidualJacobianSource::Symbolic {
                assembly,
                execution: mode,
            };
            if matches!(mode, Lsode2SymbolicExecutionMode::LambdifyExpr) {
                config.backend.generated_backend.resolver = None;
                config.backend.generated_backend.handoff_path = None;
            }
            config.sync_legacy_backend_from_policy();
        }

        if let Some(policy) = self.callback_policy {
            if !matches!(
                config.residual_jacobian_source,
                Lsode2ResidualJacobianSource::Symbolic {
                    execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
                    ..
                }
            ) {
                return Err(unsupported_lsode2(
                    "callback scheduling is supported only for the Lambdify route",
                ));
            }
            config.lambdify_execution_policy = match policy {
                StiffCallbackPolicy::Sequential => IvpLambdifyExecutionPolicy::Sequential,
                StiffCallbackPolicy::Parallel { min_work } => {
                    IvpLambdifyExecutionPolicy::Parallel { min_work }
                }
                StiffCallbackPolicy::Auto { min_work } => {
                    IvpLambdifyExecutionPolicy::Auto { min_work }
                }
            };
        }

        if let Some(telemetry) = self.telemetry {
            config.telemetry = crate::numerical::LSODE2::IvpTelemetry::with_mode(match telemetry {
                SolverTelemetryMode::Off => crate::numerical::LSODE2::IvpTelemetryMode::Off,
                SolverTelemetryMode::Counters => {
                    crate::numerical::LSODE2::IvpTelemetryMode::Counters
                }
                SolverTelemetryMode::Timings => crate::numerical::LSODE2::IvpTelemetryMode::Timings,
                SolverTelemetryMode::Detailed => {
                    crate::numerical::LSODE2::IvpTelemetryMode::Detailed
                }
            });
        }
        Ok(())
    }
}

fn unsupported_lsode2(capability: &'static str) -> StiffApiError {
    StiffApiError::UnsupportedCapability {
        solver: StiffSolverKind::Lsode2,
        capability,
    }
}

/// Construct a prepared Radau solver after applying the optional common profile.
pub fn prepare_radau(
    problem: crate::numerical::Radau::api::RadauProblem,
    mut config: RadauConfig,
    preset: Option<&StiffPreset>,
) -> Result<RadauSolver, StiffApiError> {
    validate_state_dimension(preset, StiffSolverKind::Radau, problem.dimension())?;
    validate_state_dimension(preset, StiffSolverKind::Radau, problem.state_dimension())?;
    if let Some(preset) = preset {
        preset.apply_radau(&mut config)?;
    }
    RadauSolver::prepare(problem, config).map_err(StiffApiError::Radau)
}

/// Prepare a BDF solver through its fallible native symbolic preparation API.
pub fn prepare_bdf(
    mut options: BdfSolverOptions,
    preset: Option<&StiffPreset>,
) -> Result<BdfSolver, StiffApiError> {
    validate_state_dimension(preset, StiffSolverKind::Bdf, options.eq_system.len())?;
    validate_state_dimension(preset, StiffSolverKind::Bdf, options.y0.len())?;
    if let Some(preset) = preset {
        preset.apply_bdf(&mut options)?;
    }
    let mut solver = BdfSolver::new_with_options(options);
    solver
        .try_generate()
        .map_err(|error| StiffApiError::Bdf(BdfSolveError::Backend(error)))?;
    Ok(solver)
}

/// Construct a Backward Euler solver after applying the optional common profile.
pub fn prepare_backward_euler(
    mut options: BeSolverOptions,
    preset: Option<&StiffPreset>,
) -> Result<BE, StiffApiError> {
    validate_state_dimension(
        preset,
        StiffSolverKind::BackwardEuler,
        options.eq_system.len(),
    )?;
    validate_state_dimension(preset, StiffSolverKind::BackwardEuler, options.y0.len())?;
    if let Some(preset) = preset {
        preset.apply_be(&mut options)?;
    }
    BE::try_new_with_options(options).map_err(StiffApiError::BackwardEuler)
}

/// Construct LSODE2 without flattening its controller or execution lifecycle.
pub fn prepare_lsode2(
    mut config: Lsode2ProblemConfig,
    preset: Option<&StiffPreset>,
) -> Result<Lsode2Solver, StiffApiError> {
    validate_state_dimension(preset, StiffSolverKind::Lsode2, config.eq_system.len())?;
    validate_state_dimension(preset, StiffSolverKind::Lsode2, config.y0.len())?;
    if let Some(preset) = preset {
        preset.apply_lsode2(&mut config)?;
    }
    if matches!(
        config.residual_jacobian_source,
        Lsode2ResidualJacobianSource::Symbolic {
            execution: Lsode2SymbolicExecutionMode::PreparedAtom,
            ..
        }
    ) {
        return Err(unsupported_lsode2(
            "PreparedAtom is experimental and excluded from the production StiffAPI",
        ));
    }
    let logging = preset.and_then(|preset| preset.logging);
    let mut solver = Lsode2Solver::new(config).map_err(StiffApiError::Lsode2)?;
    if let Some(level) = logging {
        solver.set_logging_level(level);
    }
    Ok(solver)
}

fn validate_state_dimension(
    preset: Option<&StiffPreset>,
    solver: StiffSolverKind,
    actual: usize,
) -> Result<(), StiffApiError> {
    if let Some(expected) = preset.and_then(|preset| preset.required_state_dimension)
        && actual != expected
    {
        return Err(StiffApiError::DimensionMismatch {
            solver,
            expected,
            actual,
        });
    }
    Ok(())
}

/// Typed facade failures. Native errors remain intact as source errors.
#[derive(Debug)]
pub enum StiffApiError {
    ConflictingPresetAxis(&'static str),
    UnsupportedCapability {
        solver: StiffSolverKind,
        capability: &'static str,
    },
    DimensionMismatch {
        solver: StiffSolverKind,
        expected: usize,
        actual: usize,
    },
    Radau(RadauError),
    Bdf(BdfSolveError),
    BackwardEuler(BeError),
    Lsode2(Lsode2Error),
}

/// Stage at which a StiffAPI request failed. Solver-operation failures after
/// preparation are returned by the native solver and keep their native types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StiffApiErrorStage {
    Configuration,
    Preparation,
}

impl StiffApiError {
    /// Identify the solver involved when the error is solver-specific.
    pub const fn solver_kind(&self) -> Option<StiffSolverKind> {
        match self {
            Self::UnsupportedCapability { solver, .. } | Self::DimensionMismatch { solver, .. } => {
                Some(*solver)
            }
            Self::Radau(_) => Some(StiffSolverKind::Radau),
            Self::Bdf(_) => Some(StiffSolverKind::Bdf),
            Self::BackwardEuler(_) => Some(StiffSolverKind::BackwardEuler),
            Self::Lsode2(_) => Some(StiffSolverKind::Lsode2),
            Self::ConflictingPresetAxis(_) => None,
        }
    }

    /// Classify facade validation separately from native preparation errors.
    pub const fn stage(&self) -> StiffApiErrorStage {
        match self {
            Self::ConflictingPresetAxis(_)
            | Self::UnsupportedCapability { .. }
            | Self::DimensionMismatch { .. } => StiffApiErrorStage::Configuration,
            Self::Radau(_) | Self::Bdf(_) | Self::BackwardEuler(_) | Self::Lsode2(_) => {
                StiffApiErrorStage::Preparation
            }
        }
    }
}

impl fmt::Display for StiffApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConflictingPresetAxis(axis) => {
                write!(f, "multiple StiffAPI presets configure the {axis} axis")
            }
            Self::UnsupportedCapability { solver, capability } => {
                write!(
                    f,
                    "{solver:?} does not support this StiffAPI request: {capability}"
                )
            }
            Self::DimensionMismatch {
                solver,
                expected,
                actual,
            } => write!(
                f,
                "{solver:?} request has state dimension {actual}; preset requires {expected}"
            ),
            Self::Radau(error) => write!(f, "Radau failed: {error}"),
            Self::Bdf(error) => write!(f, "BDF failed: {error}"),
            Self::BackwardEuler(error) => write!(f, "Backward Euler failed: {error}"),
            Self::Lsode2(error) => write!(f, "LSODE2 failed: {error}"),
        }
    }
}

impl Error for StiffApiError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Radau(error) => Some(error),
            Self::Bdf(error) => Some(error),
            Self::BackwardEuler(error) => Some(error),
            Self::Lsode2(error) => Some(error),
            Self::ConflictingPresetAxis(_)
            | Self::UnsupportedCapability { .. }
            | Self::DimensionMismatch { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests;
