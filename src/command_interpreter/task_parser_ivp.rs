//! IVP task-shell built on top of the generic [`DocumentParser`].
//!
//! This module turns a structured text document into a typed IVP specification
//! and then wires each selected solver into its solver-owned native API.
//!
//! The parser supports two usage modes:
//! - full end-to-end task documents, where equations, initial conditions,
//!   solver selection, solver options, and postprocessing live in one DSL
//!   document
//! - split mode, where the document contributes only solver settings and the
//!   IVP problem itself is assembled from plain Rust data before the solver is
//!   built
//!
//! It is intentionally narrower than the historical damped-BVP task parser:
//! the parser stage only validates and normalizes user input, while solver
//! execution and postprocessing remain explicit follow-up steps.

use crate::command_interpreter::task_parser::{DocumentMap, DocumentParser, ParseError, Value};
use crate::command_interpreter::task_parser_common::{
    parse_continuation_spec, parse_symbolic_equation_system, ContinuationMode,
    ContinuationRestartPolicy, ContinuationSpec, SharedEquationParseError,
};
use crate::numerical::ODE_api2::{
    NonStiffMethod, SolverType, UniversalODESolver, UniversalOdeError,
};
use crate::numerical::Radau::{RadauConfig, RadauError, RadauProblem, RadauSolution, RadauSolver};
use crate::numerical::BDF::BDF_api::{BdfSolveError, BdfSolverOptions, ODEsolver as BdfOdeSolver};
use crate::numerical::BE::{BeError, BeSolverOptions, BE};
use crate::numerical::LSODE2::{
    Lsode2AotProfile, Lsode2AotToolchain, Lsode2ControllerConfig, Lsode2JacobianBackend,
    Lsode2LinearSolverChoice, Lsode2LinearSolverPolicy, Lsode2LinearSystemStructure,
    Lsode2NativeExecutionConfig, Lsode2ProblemConfig, Lsode2ResidualJacobianSource,
    Lsode2StopComparator, Lsode2StopCondition, Lsode2SymbolicAssemblyBackend,
    Lsode2SymbolicExecutionMode,
};
use crate::numerical::LSODE2::{Lsode2Error, Lsode2Solver};
use crate::symbolic::symbolic_engine::Expr;
use crate::Utils::postprocessing::{PostprocessDataset, PostprocessError, PostprocessPlan};
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::time::Instant;

/// Top-level family selector for text-driven task shells.
///
/// For now we only support IVP tasks, but keeping this enum explicit makes it
/// easier to expand the textual interface later without redesigning the whole
/// normalization layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskKindSpec {
    Ivp,
}

/// Method selection normalized out of the user-facing string document.
///
/// We keep a small typed enum instead of storing raw strings all the way down
/// so that validation happens once, early. Nonstiff methods use the universal
/// facade; stiff methods are dispatched by the native adapters below.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IvpMethodSpec {
    NonStiff(String),
    Radau5,
    Bdf,
    BackwardEuler,
    /// Historical fixed-BDF LSODE route.
    Lsode,
    /// LSODA-style automatic Adams/BDF controller route.
    Lsoda,
    /// Explicit LSODE2 route. Its default remains BDF-only for compatibility.
    Lsode2,
}

impl IvpMethodSpec {
    fn to_solver_type(&self) -> Result<SolverType, IvpTaskError> {
        match self {
            Self::NonStiff(name) => Ok(SolverType::NonStiff(NonStiffMethod::from_name(
                name.clone(),
            ))),
            Self::Radau5
            | Self::Bdf
            | Self::BackwardEuler
            | Self::Lsode
            | Self::Lsoda
            | Self::Lsode2 => Err(IvpTaskError::InvalidConfiguration {
                field: "solver.method".to_string(),
                message: "stiff IVP tasks must be built through their solver-owned native API"
                    .to_string(),
            }),
        }
    }

    fn from_str(raw: &str) -> Result<Self, IvpTaskError> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "rk45" => Ok(Self::NonStiff("RK45".to_string())),
            "rk4" => Ok(Self::NonStiff("RK4".to_string())),
            "euler" => Ok(Self::NonStiff("euler".to_string())),
            "ab4" => Ok(Self::NonStiff("AB4".to_string())),
            "radau5" | "radau-5" | "radau_iia_5" => Ok(Self::Radau5),
            "bdf" => Ok(Self::Bdf),
            "backwardeuler" | "backward_euler" | "implicit_euler" => Ok(Self::BackwardEuler),
            "lsode2" => Ok(Self::Lsode2),
            "lsode" => Ok(Self::Lsode),
            "lsoda" => Ok(Self::Lsoda),
            other => Err(IvpTaskError::UnknownMethod(other.to_string())),
        }
    }
}

/// Solver selection extracted from the IVP DSL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolverSelectionSpec {
    pub task_kind: TaskKindSpec,
    pub method: IvpMethodSpec,
}

/// Symbolic IVP system after document normalization.
///
/// At this stage the independent variable, unknown names, symbolic RHS and
/// numeric parameter substitutions are already separated and validated.
#[derive(Debug, Clone, PartialEq)]
pub struct EquationSpec {
    pub arg: String,
    pub unknowns: Vec<String>,
    pub rhs: Vec<Expr>,
    /// Symbolic RHS before numeric parameter binding. This is retained for
    /// prepared continuation and avoids reparsing the task document.
    pub symbolic_rhs: Vec<Expr>,
    pub parameter_names: Vec<String>,
    pub parameter_values: HashMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InitialConditionSpec {
    pub t0: f64,
    pub t_end: f64,
    pub y0: Vec<f64>,
}

/// Solver knobs normalized before dispatch to a native or nonstiff adapter.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct IvpSolverOptionsSpec {
    pub step_size: Option<f64>,
    pub tolerance: Option<f64>,
    pub max_iterations: Option<usize>,
    pub rtol: Option<f64>,
    pub atol: Option<f64>,
    pub max_step: Option<f64>,
    pub first_step: Option<f64>,
    pub vectorized: Option<bool>,
    pub parallel: Option<bool>,
    pub neighborhood_check: Option<f64>,
    pub lsode2: Option<Lsode2TaskOptionsSpec>,
}

/// LSODE2-specific options parsed from `solver_options`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Lsode2TaskOptionsSpec {
    pub controller: Option<Lsode2ControllerConfig>,
    pub symbolic_assembly: Option<Lsode2SymbolicAssemblyBackend>,
    pub symbolic_execution: Option<Lsode2TaskExecutionSpec>,
    pub linear_system_structure: Option<Lsode2LinearSystemStructure>,
    pub linear_solver_policy: Option<Lsode2LinearSolverPolicy>,
    pub native_execution: Option<Lsode2NativeExecutionConfig>,
    /// Optional solver-owned termination condition from the task document.
    pub stop_conditions: Vec<Lsode2StopCondition>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Lsode2TaskExecutionSpec {
    /// Legacy task spelling for the execution route. The symbolic assembly
    /// remains independently selected by `lsode2_symbolic_assembly`, so this
    /// does not imply ExprLegacy and also works with AtomView.
    LambdifyExpr,
    Aot {
        toolchain: Lsode2AotToolchain,
        profile: Lsode2AotProfile,
        output_parent_dir: Option<PathBuf>,
    },
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct PostprocessingSpec {
    pub save_csv: bool,
    pub csv_path: Option<String>,
    pub save_txt: bool,
    pub txt_path: Option<String>,
    pub write_report: bool,
    pub report_path: Option<String>,
    pub plotters_png: bool,
    pub plotters_dir: Option<String>,
    pub gnuplot_png: bool,
    pub gnuplot_dir: Option<String>,
    pub terminal_plot: bool,
    pub plot: bool,
}

impl PostprocessingSpec {
    fn to_plan(&self, default_csv_path: &str) -> PostprocessPlan {
        let mut plan = PostprocessPlan::new();
        if self.save_csv {
            plan = plan.save_csv(
                self.csv_path
                    .clone()
                    .unwrap_or_else(|| default_csv_path.to_string()),
            );
        }
        if self.save_txt {
            plan = plan.save_txt(
                self.txt_path
                    .clone()
                    .unwrap_or_else(|| "ivp_result.txt".to_string()),
            );
        }
        if self.write_report {
            plan = plan.write_report(
                self.report_path
                    .clone()
                    .unwrap_or_else(|| "ivp_report.md".to_string()),
            );
        }
        if self.plotters_png {
            plan = plan.plotters_png(
                self.plotters_dir
                    .clone()
                    .unwrap_or_else(|| "ivp_plotters".to_string()),
            );
        }
        if self.gnuplot_png {
            plan = plan.gnuplot_png(
                self.gnuplot_dir
                    .clone()
                    .unwrap_or_else(|| "ivp_gnuplot".to_string()),
            );
        }
        if self.terminal_plot {
            plan = plan.terminal_plot();
        }
        plan
    }
}

/// Fully normalized textual IVP task ready to be turned into a solver.
#[derive(Debug, Clone, PartialEq)]
pub struct IvpTaskSpec {
    /// Version of the textual IVP schema used to normalize this task.
    /// Version one is the current format; future versions must be rejected
    /// explicitly instead of being silently interpreted as version one.
    pub schema_version: u32,
    pub solver: SolverSelectionSpec,
    pub equations: EquationSpec,
    pub initial_conditions: InitialConditionSpec,
    pub solver_options: IvpSolverOptionsSpec,
    pub postprocessing: PostprocessingSpec,
    /// Optional repeated-parameter plan. The symbolic RHS is retained inside
    /// the plan so continuation runners do not reparse the document.
    pub continuation: Option<ContinuationSpec>,
}

impl IvpTaskSpec {
    /// Extract the problem-only subset from a full IVP task.
    pub fn problem_spec(&self) -> IvpProblemSpec {
        IvpProblemSpec {
            equations: self.equations.clone(),
            initial_conditions: self.initial_conditions.clone(),
        }
    }

    /// Extract the solver-selection and solver-option subset from a full IVP task.
    pub fn solver_settings_spec(&self) -> IvpSolverSettingsSpec {
        IvpSolverSettingsSpec {
            solver: self.solver.clone(),
            solver_options: self.solver_options.clone(),
        }
    }

    /// Extract the postprocessing subset from a full IVP task.
    pub fn postprocessing_spec(&self) -> PostprocessingSpec {
        self.postprocessing.clone()
    }
}

/// Problem-only subset of the IVP task DSL.
#[derive(Debug, Clone, PartialEq)]
pub struct IvpProblemSpec {
    pub equations: EquationSpec,
    pub initial_conditions: InitialConditionSpec,
}

/// Solver-settings-only subset of the IVP task DSL.
#[derive(Debug, Clone, PartialEq)]
pub struct IvpSolverSettingsSpec {
    pub solver: SolverSelectionSpec,
    pub solver_options: IvpSolverOptionsSpec,
}

/// Native execution adapter for a parsed IVP task.
///
/// The text format is only a front-end.  Once a task is normalized, stiff
/// methods must not pass through `ODE_api2`: their options, lifecycle and
/// failure semantics belong to the solver that actually implements them.
pub enum IvpTaskSolver {
    /// Nonstiff compatibility route owned by the universal facade.
    Universal(UniversalODESolver),
    /// Native BDF API.
    Bdf(BdfOdeSolver),
    /// Native backward-Euler API.
    BackwardEuler(BE),
    /// Native Radau API, retaining the initial state until the first solve.
    Radau {
        solver: RadauSolver,
        y0: DVector<f64>,
        parameters: Vec<f64>,
        last_solution: Option<RadauSolution>,
    },
    /// Completed Radau result, retained for the common result adapter.
    RadauSolution(RadauSolution),
    /// Native LSODE2 API.
    Lsode2(Lsode2Solver),
}

impl IvpTaskSolver {
    /// Rebind a prepared symbolic model without reparsing its expressions.
    fn try_set_parameter_values(&mut self, values: &[f64]) -> Result<(), IvpTaskError> {
        match self {
            Self::Bdf(solver) => solver
                .set_parameter_values(DVector::from_column_slice(values))
                .map_err(|error| {
                    IvpTaskError::Native(IvpNativeSolverError::Bdf(BdfSolveError::Backend(error)))
                }),
            Self::BackwardEuler(solver) => solver
                .try_set_parameter_values(DVector::from_column_slice(values))
                .map_err(|error| IvpTaskError::Native(IvpNativeSolverError::BackwardEuler(error))),
            Self::Radau { parameters, .. } => {
                *parameters = values.to_vec();
                Ok(())
            }
            Self::Lsode2(solver) => solver
                .set_parameter_values(DVector::from_column_slice(values))
                .map_err(|error| IvpTaskError::Native(IvpNativeSolverError::Lsode2(error))),
            Self::Universal(_) | Self::RadauSolution(_) => Err(IvpTaskError::Continuation {
                method: "nonstiff/RadauSolution".to_string(),
                message: "parameter rebind is not supported after result extraction".to_string(),
            }),
        }
    }

    /// Apply an explicit segment state to a prepared adapter. Only adapters
    /// with a native restart contract may do this without rebuilding the
    /// symbolic model; unsupported routes return a typed continuation error.
    fn try_restart_with_initial_state(
        &mut self,
        initial: &InitialConditionSpec,
    ) -> Result<(), IvpTaskError> {
        match self {
            Self::Bdf(solver) => {
                if !solver.is_backend_prepared() {
                    solver.try_generate().map_err(|error| {
                        IvpTaskError::Native(IvpNativeSolverError::Bdf(BdfSolveError::Backend(
                            error,
                        )))
                    })?;
                }
                solver
                    .try_restart_with_initial_state(
                        initial.t0,
                        DVector::from_vec(initial.y0.clone()),
                        initial.t_end,
                    )
                    .map_err(|error| IvpTaskError::Native(IvpNativeSolverError::Bdf(error)))
            }
            Self::BackwardEuler(solver) => solver
                .try_restart_with_initial_state(
                    initial.t0,
                    DVector::from_vec(initial.y0.clone()),
                    initial.t_end,
                )
                .map_err(|error| IvpTaskError::Native(IvpNativeSolverError::BackwardEuler(error))),
            Self::Lsode2(solver) => solver
                .prepare()
                .map_err(|error| IvpTaskError::Continuation {
                    method: "LSODE2".to_string(),
                    message: error.to_string(),
                })
                .and_then(|_| {
                    solver
                        .try_restart_with_initial_state(
                            initial.t0,
                            DVector::from_vec(initial.y0.clone()),
                            initial.t_end,
                        )
                        .map_err(|error| IvpTaskError::Continuation {
                            method: "LSODE2".to_string(),
                            message: error.to_string(),
                        })
                }),
            Self::Radau { solver, y0, .. } => {
                *y0 = DVector::from_vec(initial.y0.clone());
                solver
                    .restart(initial.t0, initial.t_end)
                    .map_err(|error| IvpTaskError::Native(IvpNativeSolverError::Radau(error)))
            }
            Self::Universal(_) => Err(IvpTaskError::Continuation {
                method: "UniversalODESolver".to_string(),
                message: "explicit segment y0/t0 restart is not supported by this adapter"
                    .to_string(),
            }),
            Self::RadauSolution(_) => Err(IvpTaskError::Continuation {
                method: "RadauSolution".to_string(),
                message: "explicit segment restart after result extraction is unsupported"
                    .to_string(),
            }),
        }
    }

    /// Continue a prepared native route or restart it from the task's initial
    /// state. Unsupported lifecycle combinations fail explicitly.
    fn try_continue_with_parameters(
        &mut self,
        values: &[f64],
        restart_each: bool,
        initial: &InitialConditionSpec,
    ) -> Result<(), IvpTaskError> {
        match self {
            Self::Bdf(solver) => {
                let values = DVector::from_column_slice(values);
                if restart_each {
                    solver.set_parameter_values(values).map_err(|error| {
                        IvpTaskError::Native(IvpNativeSolverError::Bdf(BdfSolveError::Backend(
                            error,
                        )))
                    })?;
                    solver
                        .try_restart_with_initial_state(
                            initial.t0,
                            DVector::from_vec(initial.y0.clone()),
                            initial.t_end,
                        )
                        .map_err(|error| IvpTaskError::Native(IvpNativeSolverError::Bdf(error)))?;
                } else {
                    solver
                        .try_continue_with_parameter_values(values, initial.t_end)
                        .map_err(|error| IvpTaskError::Native(IvpNativeSolverError::Bdf(error)))?;
                }
                solver
                    .try_solve()
                    .map_err(|error| IvpTaskError::Native(IvpNativeSolverError::Bdf(error)))
            }
            Self::BackwardEuler(solver) => {
                solver
                    .try_set_parameter_values(DVector::from_column_slice(values))
                    .map_err(|error| {
                        IvpTaskError::Native(IvpNativeSolverError::BackwardEuler(error))
                    })?;
                if restart_each {
                    solver
                        .try_restart_with_initial_state(
                            initial.t0,
                            DVector::from_vec(initial.y0.clone()),
                            initial.t_end,
                        )
                        .map_err(|error| {
                            IvpTaskError::Native(IvpNativeSolverError::BackwardEuler(error))
                        })?;
                    solver.try_solve().map_err(|error| {
                        IvpTaskError::Native(IvpNativeSolverError::BackwardEuler(error))
                    })
                } else {
                    solver.try_continue_to(initial.t_end).map_err(|error| {
                        IvpTaskError::Native(IvpNativeSolverError::BackwardEuler(error))
                    })
                }
            }
            Self::Radau {
                solver,
                y0,
                parameters,
                last_solution,
            } => {
                *parameters = values.to_vec();
                *y0 = DVector::from_vec(initial.y0.clone());
                solver
                    .restart(initial.t0, initial.t_end)
                    .map_err(|error| IvpTaskError::Native(IvpNativeSolverError::Radau(error)))?;
                let solution = solver
                    .continue_with_parameters(y0.as_slice(), parameters)
                    .map_err(|error| IvpTaskError::Native(IvpNativeSolverError::Radau(error)))?;
                *last_solution = Some(solution);
                Ok(())
            }
            Self::Lsode2(solver) if restart_each => {
                solver
                    .set_parameter_values(DVector::from_column_slice(values))
                    .map_err(|error| IvpTaskError::Native(IvpNativeSolverError::Lsode2(error)))?;
                solver
                    .try_restart_with_initial_state(
                        initial.t0,
                        DVector::from_vec(initial.y0.clone()),
                        initial.t_end,
                    )
                    .map_err(|error| IvpTaskError::Native(IvpNativeSolverError::Lsode2(error)))?;
                solver
                    .solve()
                    .map_err(|error| IvpTaskError::Native(IvpNativeSolverError::Lsode2(error)))
            }
            Self::Lsode2(_) => Err(IvpTaskError::Continuation {
                method: "LSODE2".to_string(),
                message: "warm/prepared continuation without restart is unsupported; use restart_each or restart_with_state".to_string(),
            }),
            Self::Universal(_) => {
                Err(IvpTaskError::Continuation {
                    method: "UniversalODESolver".to_string(),
                    message: "warm/prepared continuation without restart is unsupported"
                        .to_string(),
                })
            }
            Self::RadauSolution(_) => Err(IvpTaskError::Continuation {
                method: "RadauSolution".to_string(),
                message: "continuation after result extraction is unsupported".to_string(),
            }),
        }
    }

    /// Execute the selected route exactly once.
    ///
    /// Radau consumes its initial vector by value at solve time, so the adapter
    /// replaces the prepared variant with `RadauSolution`. The other native
    /// solver APIs retain their result internally and can be queried after the
    /// call returns.
    fn try_solve(&mut self) -> Result<(), IvpTaskError> {
        match self {
            Self::Universal(solver) => solver
                .try_solve()
                .map_err(|error| IvpTaskError::Native(IvpNativeSolverError::Universal(error))),
            Self::Bdf(solver) => solver
                .try_solve()
                .map_err(|error| IvpTaskError::Native(IvpNativeSolverError::Bdf(error))),
            Self::BackwardEuler(solver) => solver
                .try_solve()
                .map_err(|error| IvpTaskError::Native(IvpNativeSolverError::BackwardEuler(error))),
            Self::Lsode2(solver) => solver
                .solve()
                .map_err(|error| IvpTaskError::Native(IvpNativeSolverError::Lsode2(error))),
            Self::Radau {
                solver,
                y0,
                parameters,
                last_solution,
            } => {
                let solution = solver
                    .solve_with_parameters(y0.as_slice(), parameters)
                    .map_err(|error| IvpTaskError::Native(IvpNativeSolverError::Radau(error)))?;
                *last_solution = Some(solution);
                Ok(())
            }
            Self::RadauSolution(_) => Ok(()),
        }
    }

    /// Adapt solver-owned result storage to the task-runner's common shape.
    ///
    /// This conversion is intentionally at the boundary: the native solver
    /// remains responsible for its own storage and no solver is forced to
    /// expose the universal facade's internal representation.
    fn get_result(&self) -> (Option<DVector<f64>>, Option<DMatrix<f64>>) {
        match self {
            Self::Universal(solver) => solver.get_result(),
            Self::Bdf(solver) => {
                let (time, values) = solver.get_result();
                (Some(time), Some(values))
            }
            Self::BackwardEuler(solver) => solver.get_result(),
            Self::Lsode2(solver) => {
                let (time, values) = solver.get_result();
                (Some(time), Some(values))
            }
            Self::Radau {
                last_solution: Some(solution),
                ..
            } => (
                Some(DVector::from_element(1, solution.t)),
                Some(DMatrix::from_row_slice(1, solution.y.len(), &solution.y)),
            ),
            Self::Radau { .. } => (None, None),
            Self::RadauSolution(solution) => (
                Some(DVector::from_element(1, solution.t)),
                Some(DMatrix::from_row_slice(1, solution.y.len(), &solution.y)),
            ),
        }
    }

    /// Return a stable textual status for task reports without erasing native
    /// failure types. A prepared Radau instance has no trajectory yet, while a
    /// completed Radau result is explicitly marked as `finished`.
    fn status(&self) -> Option<String> {
        match self {
            Self::Universal(solver) => solver.get_status(),
            Self::Bdf(solver) => Some(solver.get_status().to_owned()),
            Self::BackwardEuler(solver) => Some(solver.status().as_str().to_owned()),
            Self::Lsode2(solver) => Some(solver.status().to_owned()),
            Self::Radau {
                last_solution: Some(_),
                ..
            } => Some("finished".to_owned()),
            Self::Radau { .. } => Some("prepared".to_owned()),
            Self::RadauSolution(_) => Some("finished".to_owned()),
        }
    }

    /// Convert the compatibility string status into a typed task-run status.
    fn status_code(&self) -> Option<IvpRunStatus> {
        self.status()
            .map(|status| IvpRunStatus::from_label(&status))
    }

    /// Preserve the solver-owned result shape where possible. In particular,
    /// Radau remains a terminal-state solution instead of being disguised as
    /// a one-row time/state matrix.
    fn trajectory(&self) -> Option<IvpTrajectory> {
        match self {
            Self::Radau {
                last_solution: Some(solution),
                ..
            }
            | Self::RadauSolution(solution) => Some(IvpTrajectory::Radau(solution.clone())),
            _ => {
                let (t, y) = self.get_result();
                match (t, y) {
                    (Some(t), Some(y)) => Some(IvpTrajectory::Grid { t, y }),
                    _ => None,
                }
            }
        }
    }
}

/// Stable status vocabulary exposed by the task runner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IvpRunStatus {
    Prepared,
    Running,
    Finished,
    Failed,
    StoppedByCondition,
    Unknown(String),
}

impl IvpRunStatus {
    fn from_label(label: &str) -> Self {
        let normalized = label.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "prepared" => Self::Prepared,
            value if value == "running" || value.starts_with("running_") => Self::Running,
            value if value == "finished" || value.starts_with("finished_") => Self::Finished,
            value if value == "failed" || value.starts_with("failed_") => Self::Failed,
            "stopped_by_condition" => Self::StoppedByCondition,
            other => Self::Unknown(other.to_string()),
        }
    }
}

/// Solver-owned trajectory adapter used by the typed task result.
///
/// Grid-based solvers expose their native sampled trajectory. Radau exposes
/// its native dense-output solution instead of being forced into an artificial
/// one-row matrix. The legacy `t_result`/`y_result` fields remain available on
/// [`IvpTaskRunResult`] for compatibility with older callers.
#[derive(Debug, Clone, PartialEq)]
pub enum IvpTrajectory {
    Grid { t: DVector<f64>, y: DMatrix<f64> },
    Radau(RadauSolution),
}

/// Result of running a normalized IVP task.
#[derive(Debug)]
pub struct IvpTaskRunResult {
    pub specification: IvpTaskSpec,
    pub t_result: Option<DVector<f64>>,
    pub y_result: Option<DMatrix<f64>>,
    pub status: Option<String>,
    pub status_code: Option<IvpRunStatus>,
    pub trajectory: Option<IvpTrajectory>,
    pub continuation: Option<IvpContinuationRunResult>,
}

/// One completed segment of a parameter-continuation run.
#[derive(Debug)]
pub struct IvpContinuationSegment {
    pub parameter_value: f64,
    pub parameter_values: Vec<f64>,
    pub t_result: Option<DVector<f64>>,
    pub y_result: Option<DMatrix<f64>>,
    pub status: Option<String>,
    pub status_code: Option<IvpRunStatus>,
    pub trajectory: Option<IvpTrajectory>,
    /// Time spent constructing/rebinding preparation for this segment.
    /// It is kept separate from numerical solve time and is zero for later
    /// warm/prepared segments when the prepared model is reused.
    pub prepare_ms: f64,
    /// Time spent inside the solver after preparation/rebinding.
    pub solve_ms: f64,
}

/// Typed result for repeated parameter values. The final segment is also
/// projected into [`IvpTaskRunResult`] for compatibility with single solves.
#[derive(Debug)]
pub struct IvpContinuationRunResult {
    pub parameter: String,
    pub parameters: Vec<String>,
    pub mode: ContinuationMode,
    pub restart_each: bool,
    /// Number of symbolic/native preparations performed by the runner.
    /// Fresh mode increments this for every segment; warm/prepared mode keeps
    /// it at one for the shared prepared model.
    pub fresh_preparations: usize,
    /// Number of later segments that reused the prepared model rather than
    /// rebuilding the symbolic callbacks.
    pub prepared_reuses: usize,
    pub segments: Vec<IvpContinuationSegment>,
}

/// Typed native solver failure preserved by the text-task adapter.
///
/// The adapter deliberately does not flatten these errors into a string:
/// callers can distinguish a BDF step-budget failure from a Radau callback
/// failure and can still inspect the original solver error as the source.
#[derive(Debug)]
pub enum IvpNativeSolverError {
    /// Failure from the nonstiff universal facade.
    Universal(UniversalOdeError),
    /// Failure from the native BDF API.
    Bdf(BdfSolveError),
    /// Failure from the native Backward Euler API.
    BackwardEuler(BeError),
    /// Failure from the native Radau API.
    Radau(RadauError),
    /// Failure from the native LSODE2 API.
    Lsode2(Lsode2Error),
}

impl std::fmt::Display for IvpNativeSolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Universal(error) => write!(f, "nonstiff universal solver failed: {error}"),
            Self::Bdf(error) => write!(f, "native BDF solver failed: {error}"),
            Self::BackwardEuler(error) => write!(f, "native Backward Euler solver failed: {error}"),
            Self::Radau(error) => write!(f, "native Radau solver failed: {error}"),
            Self::Lsode2(error) => write!(f, "native LSODE2 solver failed: {error}"),
        }
    }
}

impl std::error::Error for IvpNativeSolverError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Universal(error) => Some(error),
            Self::Bdf(error) => Some(error),
            Self::BackwardEuler(error) => Some(error),
            Self::Radau(error) => Some(error),
            Self::Lsode2(error) => Some(error),
        }
    }
}

/// Typed failures produced while parsing, preparing, solving, or postprocessing
/// one IVP task document.
#[derive(Debug)]
pub enum IvpTaskError {
    Parser(String),
    Io {
        path: PathBuf,
        source: io::Error,
    },
    Document(ParseError),
    MissingSection(&'static str),
    MissingField {
        section: String,
        field: String,
    },
    InvalidField {
        section: String,
        field: String,
        message: String,
    },
    InconsistentEquationCounts {
        unknowns: usize,
        rhs: usize,
    },
    UnknownMethod(String),
    /// A valid generic option that the selected adapter cannot represent.
    UnsupportedOption {
        method: String,
        option: String,
    },
    /// The selected adapter cannot honor this lifecycle without changing
    /// parameter or state semantics.
    UnsupportedContinuation {
        method: String,
        mode: String,
    },
    Continuation {
        method: String,
        message: String,
    },
    InvalidConfiguration {
        field: String,
        message: String,
    },
    /// Symbol/type diagnostic with a best-effort position in the source task.
    SymbolDiagnostic {
        message: String,
        token: String,
        line: Option<usize>,
        column: Option<usize>,
    },
    Semantic(String),
    /// Failure returned by the selected solver, retaining its native type.
    Native(IvpNativeSolverError),
    /// Failure while constructing or writing a postprocessing dataset.
    Postprocess(PostprocessError),
}

impl IvpTaskError {
    /// Stable error family for compact task-batch reports.
    pub fn category(&self) -> &'static str {
        match self {
            Self::Parser(_) | Self::Document(_) => "parse",
            Self::Io { .. } => "io",
            Self::MissingSection(_)
            | Self::MissingField { .. }
            | Self::InvalidField { .. }
            | Self::InconsistentEquationCounts { .. }
            | Self::UnknownMethod(_)
            | Self::InvalidConfiguration { .. }
            | Self::Semantic(_)
            | Self::SymbolDiagnostic { .. } => "configuration",
            Self::UnsupportedOption { .. } => "unsupported_option",
            Self::UnsupportedContinuation { .. } | Self::Continuation { .. } => "continuation",
            Self::Native(_) => "solver",
            Self::Postprocess(_) => "postprocess",
        }
    }
}

impl std::fmt::Display for IvpTaskError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parser(msg) => write!(f, "parser error: {msg}"),
            Self::Io { path, source } => {
                write!(f, "failed to read IVP task `{}`: {source}", path.display())
            }
            Self::Document(error) => write!(f, "document parser error: {error}"),
            Self::MissingSection(section) => write!(f, "missing section `{section}`"),
            Self::MissingField { section, field } => {
                write!(f, "missing field `{field}` in section `{section}`")
            }
            Self::InvalidField {
                section,
                field,
                message,
            } => write!(
                f,
                "invalid field `{field}` in section `{section}`: {message}"
            ),
            Self::InconsistentEquationCounts { unknowns, rhs } => write!(
                f,
                "number of unknowns ({unknowns}) does not match number of rhs expressions ({rhs})"
            ),
            Self::UnknownMethod(method) => write!(f, "unknown IVP method `{method}`"),
            Self::UnsupportedOption { method, option } => write!(
                f,
                "IVP method `{method}` does not support solver option `{option}`"
            ),
            Self::UnsupportedContinuation { method, mode } => write!(
                f,
                "IVP method `{method}` does not support continuation mode `{mode}`"
            ),
            Self::Continuation { method, message } => {
                write!(f, "continuation failed for `{method}`: {message}")
            }
            Self::InvalidConfiguration { field, message } => {
                write!(f, "invalid IVP configuration `{field}`: {message}")
            }
            Self::SymbolDiagnostic {
                message,
                token,
                line,
                column,
            } => {
                write!(f, "symbol diagnostic for `{token}`: {message}")?;
                if let (Some(line), Some(column)) = (line, column) {
                    write!(f, " at line {line}, column {column}")?;
                }
                Ok(())
            }
            Self::Semantic(message) => write!(f, "{message}"),
            Self::Native(error) => write!(f, "{error}"),
            Self::Postprocess(error) => write!(f, "postprocessing error: {error}"),
        }
    }
}

impl std::error::Error for IvpTaskError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Document(error) => Some(error),
            Self::Native(error) => Some(error),
            Self::Postprocess(error) => Some(error),
            _ => None,
        }
    }
}

type GenericSectionMap = HashMap<String, Option<Vec<Value>>>;

/// Parse a full IVP task document from DSL text.
///
/// The generic `DocumentParser` handles syntax. This function is responsible
/// for IVP-specific normalization:
/// - resolve section/field pseudonyms
/// - preserve pair-style variable names in `equations`
/// - validate counts and types
/// - substitute numeric parameters into symbolic RHS expressions
pub fn parse_ivp_task_from_str(input: &str) -> Result<IvpTaskSpec, IvpTaskError> {
    let mut parser = DocumentParser::new(input.to_string());
    let pseudonyms = default_ivp_pseudonyms();
    parser
        .try_with_pseudonims_typed(Some(pseudonyms.0), Some(pseudonyms.1))
        .map_err(IvpTaskError::Document)?;
    parser
        .parse_document_typed()
        .map_err(IvpTaskError::Document)?;
    parser
        .try_keys_to_lower_case_typed(Some(vec![
            "equations".to_string(),
            "parameters".to_string(),
            "where".to_string(),
            "substitute".to_string(),
        ]))
        .map_err(IvpTaskError::Document)?;
    let document = parser
        .get_result()
        .ok_or_else(|| IvpTaskError::InvalidConfiguration {
            field: "document".to_string(),
            message: "document parser returned no result".to_string(),
        })?;
    parse_ivp_task_from_document(document).map_err(|error| attach_symbol_position(error, input))
}

fn attach_symbol_position(error: IvpTaskError, input: &str) -> IvpTaskError {
    let (message, token) = match error {
        IvpTaskError::Semantic(message) => {
            let token = diagnostic_token(&message);
            (message, token)
        }
        IvpTaskError::InvalidField {
            section,
            field,
            message,
        } => {
            let token = if field == "*" {
                diagnostic_token(&message)
            } else {
                field.clone()
            };
            (
                format!("invalid field `{field}` in section `{section}`: {message}"),
                token,
            )
        }
        other => return other,
    };
    let (line, column) = source_position(input, &token)
        .map(|(line, column)| (Some(line), Some(column)))
        .unwrap_or((None, None));
    IvpTaskError::SymbolDiagnostic {
        message,
        token,
        line,
        column,
    }
}

fn diagnostic_token(message: &str) -> String {
    if let Some((_, remainder)) = message.split_once("symbol(s):") {
        return remainder
            .split(|ch: char| ch == ',' || ch.is_whitespace())
            .find(|token| !token.is_empty())
            .unwrap_or("<unknown>")
            .trim_matches('`')
            .to_string();
    }
    message.split('`').nth(1).unwrap_or("equations").to_string()
}

fn source_position(input: &str, token: &str) -> Option<(usize, usize)> {
    let offset = input.find(token)?;
    let before = &input[..offset];
    let line = before.lines().count().max(1);
    let column = before
        .rsplit('\n')
        .next()
        .map(|line| line.chars().count() + 1)
        .unwrap_or(1);
    Some((line, column))
}

/// Parse a full IVP task document from an on-disk file.
pub fn parse_ivp_task_from_file(path: Option<PathBuf>) -> Result<IvpTaskSpec, IvpTaskError> {
    let file_path = match path {
        Some(path) => path,
        None => {
            let current_dir = std::env::current_dir().map_err(|source| IvpTaskError::Io {
                path: PathBuf::from("."),
                source,
            })?;
            let entries = fs::read_dir(&current_dir).map_err(|source| IvpTaskError::Io {
                path: current_dir.clone(),
                source,
            })?;
            entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .find(|candidate| {
                    candidate
                        .file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.starts_with("problem") && name.ends_with(".txt"))
                })
                .ok_or_else(|| IvpTaskError::Io {
                    path: current_dir.join("problem*.txt"),
                    source: io::Error::new(
                        io::ErrorKind::NotFound,
                        "no problem*.txt task document found",
                    ),
                })?
        }
    };
    let input = fs::read_to_string(&file_path).map_err(|source| IvpTaskError::Io {
        path: file_path,
        source,
    })?;
    parse_ivp_task_from_str(&input)
}

/// Fallible alias for [`parse_ivp_problem_from_document`].
pub fn try_parse_ivp_problem_from_document(
    document: &DocumentMap,
) -> Result<IvpProblemSpec, IvpTaskError> {
    parse_ivp_problem_from_document(document)
}

/// Fallible alias for [`parse_ivp_solver_settings_from_document`].
pub fn try_parse_ivp_solver_settings_from_document(
    document: &DocumentMap,
) -> Result<IvpSolverSettingsSpec, IvpTaskError> {
    parse_ivp_solver_settings_from_document(document)
}

/// Fallible alias for [`parse_ivp_task_from_document`].
pub fn try_parse_ivp_task_from_document(
    document: &DocumentMap,
) -> Result<IvpTaskSpec, IvpTaskError> {
    parse_ivp_task_from_document(document)
}

/// Fallible alias for [`parse_ivp_task_from_str`].
pub fn try_parse_ivp_task_from_str(input: &str) -> Result<IvpTaskSpec, IvpTaskError> {
    parse_ivp_task_from_str(input)
}

/// Fallible alias for [`parse_ivp_task_from_file`].
pub fn try_parse_ivp_task_from_file(path: Option<PathBuf>) -> Result<IvpTaskSpec, IvpTaskError> {
    parse_ivp_task_from_file(path)
}

/// Parse only the equation and initial-condition part of the IVP DSL.
pub fn parse_ivp_problem_from_document(
    document: &DocumentMap,
) -> Result<IvpProblemSpec, IvpTaskError> {
    let equations = parse_equations(document)?;
    let initial_conditions = parse_initial_conditions(document, equations.unknowns.len())?;
    Ok(IvpProblemSpec {
        equations,
        initial_conditions,
    })
}

/// Parse only the solver-selection and solver-option part of the IVP DSL.
pub fn parse_ivp_solver_settings_from_document(
    document: &DocumentMap,
) -> Result<IvpSolverSettingsSpec, IvpTaskError> {
    let solver = parse_solver_selection(document)?;
    let solver_options = parse_solver_options(document)?;
    validate_solver_options(&solver.method, &solver_options)?;
    Ok(IvpSolverSettingsSpec {
        solver,
        solver_options,
    })
}

/// Reject options that an adapter cannot honor instead of silently dropping
/// them. The task DSL is intentionally shared, but solver semantics are not.
fn validate_solver_options(
    method: &IvpMethodSpec,
    options: &IvpSolverOptionsSpec,
) -> Result<(), IvpTaskError> {
    let method_name = ivp_method_name(method);
    let unsupported = match method {
        IvpMethodSpec::NonStiff(_) => None,
        IvpMethodSpec::Bdf => [
            (options.step_size.is_some(), "step_size"),
            (options.tolerance.is_some(), "tolerance"),
            (options.parallel.is_some(), "parallel"),
            (options.neighborhood_check.is_some(), "neighborhood_check"),
        ]
        .into_iter()
        .find_map(|(present, name)| present.then_some(name)),
        IvpMethodSpec::BackwardEuler => [
            (options.rtol.is_some(), "rtol"),
            (options.atol.is_some(), "atol"),
            (options.max_step.is_some(), "max_step"),
            (options.first_step.is_some(), "first_step"),
            (options.vectorized.is_some(), "vectorized"),
            (options.parallel.is_some(), "parallel"),
        ]
        .into_iter()
        .find_map(|(present, name)| present.then_some(name)),
        IvpMethodSpec::Radau5 => [
            (options.step_size.is_some(), "step_size"),
            (options.vectorized.is_some(), "vectorized"),
            (options.parallel.is_some(), "parallel"),
            (options.neighborhood_check.is_some(), "neighborhood_check"),
        ]
        .into_iter()
        .find_map(|(present, name)| present.then_some(name)),
        IvpMethodSpec::Lsode | IvpMethodSpec::Lsoda | IvpMethodSpec::Lsode2 => [
            (options.step_size.is_some(), "step_size"),
            (options.tolerance.is_some(), "tolerance"),
            (options.max_iterations.is_some(), "max_iterations"),
            (options.parallel.is_some(), "parallel"),
            (options.neighborhood_check.is_some(), "neighborhood_check"),
        ]
        .into_iter()
        .find_map(|(present, name)| present.then_some(name)),
    };
    if let Some(option) = unsupported {
        return Err(IvpTaskError::UnsupportedOption {
            method: method_name.to_string(),
            option: option.to_string(),
        });
    }
    Ok(())
}

fn ivp_method_name(method: &IvpMethodSpec) -> &str {
    match method {
        IvpMethodSpec::NonStiff(name) => name.as_str(),
        IvpMethodSpec::Radau5 => "Radau5",
        IvpMethodSpec::Bdf => "BDF",
        IvpMethodSpec::BackwardEuler => "BackwardEuler",
        IvpMethodSpec::Lsode => "LSODE",
        IvpMethodSpec::Lsoda => "LSODA",
        IvpMethodSpec::Lsode2 => "LSODE2",
    }
}

/// Validate numerical invariants before any native solver allocates or
/// prepares symbolic callbacks. Keeping this at the task boundary prevents
/// NaN/Infinity and zero limits from becoming backend-specific failures.
fn validate_ivp_task_spec(spec: &IvpTaskSpec) -> Result<(), IvpTaskError> {
    if spec.schema_version != 1 {
        return Err(IvpTaskError::InvalidConfiguration {
            field: "task.schema_version".to_string(),
            message: format!(
                "unsupported IVP task schema version {}",
                spec.schema_version
            ),
        });
    }
    let initial = &spec.initial_conditions;
    if !initial.t0.is_finite() {
        return Err(IvpTaskError::InvalidConfiguration {
            field: "initial_conditions.t0".to_string(),
            message: "must be finite".to_string(),
        });
    }
    if !initial.t_end.is_finite() {
        return Err(IvpTaskError::InvalidConfiguration {
            field: "initial_conditions.t_end".to_string(),
            message: "must be finite".to_string(),
        });
    }
    if initial.t0 == initial.t_end {
        return Err(IvpTaskError::InvalidConfiguration {
            field: "initial_conditions.t_end".to_string(),
            message: "must differ from t0".to_string(),
        });
    }
    if initial.y0.is_empty() {
        return Err(IvpTaskError::InvalidConfiguration {
            field: "initial_conditions.y0".to_string(),
            message: "must contain at least one state value".to_string(),
        });
    }
    if initial.y0.iter().any(|value| !value.is_finite()) {
        return Err(IvpTaskError::InvalidConfiguration {
            field: "initial_conditions.y0".to_string(),
            message: "all state values must be finite".to_string(),
        });
    }
    if spec
        .equations
        .parameter_values
        .values()
        .any(|value| !value.is_finite())
    {
        return Err(IvpTaskError::InvalidConfiguration {
            field: "equations.parameter_values".to_string(),
            message: "all parameter values must be finite".to_string(),
        });
    }

    let positive_finite = [
        ("step_size", spec.solver_options.step_size),
        ("tolerance", spec.solver_options.tolerance),
        ("rtol", spec.solver_options.rtol),
        ("atol", spec.solver_options.atol),
        ("max_step", spec.solver_options.max_step),
        ("first_step", spec.solver_options.first_step),
        ("neighborhood_check", spec.solver_options.neighborhood_check),
    ];
    if let Some((field, _value)) = positive_finite
        .into_iter()
        .find_map(|(field, value)| value.map(|value| (field, value)))
        .filter(|(_, value)| !value.is_finite() || *value <= 0.0)
    {
        return Err(IvpTaskError::InvalidConfiguration {
            field: format!("solver_options.{field}"),
            message: "must be finite and greater than zero".to_string(),
        });
    }
    if matches!(spec.solver_options.max_iterations, Some(0)) {
        return Err(IvpTaskError::InvalidConfiguration {
            field: "solver_options.max_iterations".to_string(),
            message: "must be greater than zero".to_string(),
        });
    }
    if let Some(continuation) = &spec.continuation {
        if continuation.value_grid.is_empty()
            || continuation
                .value_grid
                .iter()
                .flatten()
                .any(|value| !value.is_finite())
        {
            return Err(IvpTaskError::InvalidConfiguration {
                field: "continuation.value_grid".to_string(),
                message: "must contain finite values and at least one segment".to_string(),
            });
        }
        if continuation.parameters.iter().any(|parameter| {
            !spec
                .equations
                .parameter_names
                .iter()
                .any(|name| name == parameter)
        }) {
            return Err(IvpTaskError::InvalidConfiguration {
                field: "continuation.parameters".to_string(),
                message: "all continuation parameters must be declared".to_string(),
            });
        }
        if continuation
            .y0_values
            .as_ref()
            .is_some_and(|values| values.iter().flatten().any(|value| !value.is_finite()))
            || continuation
                .t0_values
                .as_ref()
                .is_some_and(|values| values.iter().any(|value| !value.is_finite()))
            || continuation
                .t_end_values
                .as_ref()
                .is_some_and(|values| values.iter().any(|value| !value.is_finite()))
        {
            return Err(IvpTaskError::InvalidConfiguration {
                field: "continuation.segment_values".to_string(),
                message: "per-segment state and intervals must be finite".to_string(),
            });
        }
        if continuation
            .value_grid
            .iter()
            .any(|row| row.len() != continuation.parameters.len())
        {
            return Err(IvpTaskError::InvalidConfiguration {
                field: "continuation.value_grid".to_string(),
                message: "each grid row must contain one value per parameter".to_string(),
            });
        }
        for (field, length) in [
            (
                "continuation.y0_values",
                continuation.y0_values.as_ref().map(Vec::len),
            ),
            (
                "continuation.t0_values",
                continuation.t0_values.as_ref().map(Vec::len),
            ),
            (
                "continuation.t_end_values",
                continuation.t_end_values.as_ref().map(Vec::len),
            ),
        ] {
            if let Some(length) = length {
                let segments = continuation.value_grid.len();
                if length != 1 && length != segments {
                    return Err(IvpTaskError::InvalidConfiguration {
                        field: field.to_string(),
                        message: format!(
                            "expected one value or {segments} segment values, got {length}"
                        ),
                    });
                }
            }
        }
        if continuation.y0_values.as_ref().is_some_and(|values| {
            values
                .iter()
                .any(|state| state.len() != spec.equations.unknowns.len())
        }) {
            return Err(IvpTaskError::InvalidConfiguration {
                field: "continuation.y0_values".to_string(),
                message: "each segment state must match the number of unknowns".to_string(),
            });
        }
        if matches!(
            continuation.restart_policy,
            ContinuationRestartPolicy::Continue
        ) && (continuation.y0_values.is_some() || continuation.t0_values.is_some())
        {
            return Err(IvpTaskError::InvalidConfiguration {
                field: "continuation.restart_policy".to_string(),
                message: "per-segment y0/t0 requires restart_each or restart_with_state"
                    .to_string(),
            });
        }
        if let (Some(t0_values), Some(t_end_values)) =
            (&continuation.t0_values, &continuation.t_end_values)
        {
            let count = continuation.value_grid.len();
            for index in 0..count {
                let t0 = t0_values[if t0_values.len() == 1 { 0 } else { index }];
                let t_end = t_end_values[if t_end_values.len() == 1 { 0 } else { index }];
                if t0 == t_end {
                    return Err(IvpTaskError::InvalidConfiguration {
                        field: "continuation.interval".to_string(),
                        message: format!("segment {index} has zero-length interval"),
                    });
                }
            }
        }
    }
    validate_solver_options(&spec.solver.method, &spec.solver_options)
}

/// Parse a full IVP task from an already normalized document map.
pub fn parse_ivp_task_from_document(document: &DocumentMap) -> Result<IvpTaskSpec, IvpTaskError> {
    let problem = parse_ivp_problem_from_document(document)?;
    let solver_settings = parse_ivp_solver_settings_from_document(document)?;
    let postprocessing = parse_postprocessing(document)?;
    let continuation = if document.contains_key("continuation") {
        let parsed = parse_symbolic_equation_system(document, "t").map_err(map_equation_error)?;
        parse_continuation_spec(document, parsed.symbolic_rhs, &parsed.parameter_names)
            .map_err(map_equation_error)?
    } else {
        None
    };

    let spec = IvpTaskSpec {
        schema_version: parse_ivp_schema_version(document)?,
        solver: solver_settings.solver,
        equations: problem.equations,
        initial_conditions: problem.initial_conditions,
        solver_options: solver_settings.solver_options,
        postprocessing,
        continuation,
    };
    validate_ivp_task_spec(&spec)?;
    Ok(spec)
}

/// Convert a typed IVP spec into its native solver-owned execution route.
pub fn build_ivp_solver_from_spec(spec: &IvpTaskSpec) -> Result<IvpTaskSolver, IvpTaskError> {
    build_ivp_solver_from_problem_and_settings(&spec.problem_spec(), &spec.solver_settings_spec())
}

/// Build a solver from Rust-side problem data plus task-doc solver settings.
pub fn build_ivp_solver_from_problem_and_settings(
    problem: &IvpProblemSpec,
    settings: &IvpSolverSettingsSpec,
) -> Result<IvpTaskSolver, IvpTaskError> {
    validate_solver_options(&settings.solver.method, &settings.solver_options)?;
    let spec = IvpTaskSpec {
        schema_version: 1,
        solver: settings.solver.clone(),
        equations: problem.equations.clone(),
        initial_conditions: problem.initial_conditions.clone(),
        solver_options: settings.solver_options.clone(),
        postprocessing: PostprocessingSpec::default(),
        continuation: None,
    };
    validate_ivp_task_spec(&spec)?;
    build_ivp_solver_from_spec_impl(&spec)
}

fn parse_ivp_schema_version(document: &DocumentMap) -> Result<u32, IvpTaskError> {
    let Some(section) = document.get("task") else {
        return Ok(1);
    };
    let raw = section
        .get("schema_version")
        .and_then(|entry| entry.as_ref())
        .and_then(|entries| entries.first());
    let Some(raw) = raw else {
        return Ok(1);
    };
    let version = match raw {
        Value::Usize(value) => *value as u32,
        Value::Integer(value) if *value >= 0 => *value as u32,
        Value::String(value) => {
            value
                .parse::<u32>()
                .map_err(|_| IvpTaskError::InvalidConfiguration {
                    field: "task.schema_version".to_string(),
                    message: "expected a non-negative integer".to_string(),
                })?
        }
        _ => {
            return Err(IvpTaskError::InvalidConfiguration {
                field: "task.schema_version".to_string(),
                message: "expected a non-negative integer".to_string(),
            })
        }
    };
    if version != 1 {
        return Err(IvpTaskError::InvalidConfiguration {
            field: "task.schema_version".to_string(),
            message: format!(
                "unsupported IVP task schema version {version}; supported version is 1"
            ),
        });
    }
    Ok(version)
}

fn build_ivp_solver_from_spec_impl(spec: &IvpTaskSpec) -> Result<IvpTaskSolver, IvpTaskError> {
    if spec.equations.unknowns.len() != spec.initial_conditions.y0.len() {
        return Err(IvpTaskError::InvalidConfiguration {
            field: "initial_conditions.y0".to_string(),
            message: format!(
                "length {} does not match number of unknowns {}",
                spec.initial_conditions.y0.len(),
                spec.equations.unknowns.len()
            ),
        });
    }

    let y0 = DVector::from_vec(spec.initial_conditions.y0.clone());
    let method = &spec.solver.method;
    match method {
        IvpMethodSpec::NonStiff(_) => {
            let mut solver = UniversalODESolver::new(
                spec.equations.rhs.clone(),
                spec.equations.unknowns.clone(),
                spec.equations.arg.clone(),
                method.to_solver_type()?,
                spec.initial_conditions.t0,
                y0,
                spec.initial_conditions.t_end,
            );
            if let Some(step_size) = spec.solver_options.step_size {
                solver.set_step_size(step_size);
            }
            if let Some(value) = spec.solver_options.tolerance {
                solver.set_tolerance(value);
            }
            if let Some(value) = spec.solver_options.max_iterations {
                solver.set_max_iterations(value);
            }
            if let Some(value) = spec.solver_options.rtol {
                solver.set_rtol(value);
            }
            if let Some(value) = spec.solver_options.atol {
                solver.set_atol(value);
            }
            if let Some(value) = spec.solver_options.max_step {
                solver.set_max_step(value);
            }
            if let Some(value) = spec.solver_options.first_step {
                solver.set_first_step(Some(value));
            }
            if let Some(value) = spec.solver_options.vectorized {
                solver.set_vectorized(value);
            }
            if let Some(value) = spec.solver_options.parallel {
                solver.set_parallel(value);
            }
            if let Some(value) = spec.solver_options.neighborhood_check {
                solver.set_neighborhood_check(value);
            }
            Ok(IvpTaskSolver::Universal(solver))
        }
        IvpMethodSpec::Bdf => {
            let mut options = BdfSolverOptions::for_bdf(
                spec.equations.symbolic_rhs.clone(),
                spec.equations.unknowns.clone(),
                spec.equations.arg.clone(),
                spec.initial_conditions.t0,
                y0,
                spec.initial_conditions.t_end,
                spec.solver_options.max_step.unwrap_or(f64::INFINITY),
                spec.solver_options.rtol.unwrap_or(1e-3),
                spec.solver_options.atol.unwrap_or(1e-6),
                None,
                spec.solver_options.vectorized.unwrap_or(false),
                spec.solver_options.first_step,
            );
            if !spec.equations.parameter_names.is_empty() {
                options = options.with_equation_parameters(spec.equations.parameter_names.clone());
                options = options.with_equation_parameter_values(DVector::from_vec(
                    parameter_values_for_spec(spec)?,
                ));
            }
            if let Some(max_steps) = spec.solver_options.max_iterations {
                options.max_steps = max_steps;
            }
            let solver = BdfOdeSolver::new_with_options(options);
            Ok(IvpTaskSolver::Bdf(solver))
        }
        IvpMethodSpec::BackwardEuler => {
            let options = BeSolverOptions::new(
                spec.equations.symbolic_rhs.clone(),
                spec.equations.unknowns.clone(),
                spec.equations.arg.clone(),
                spec.solver_options.tolerance.unwrap_or(1e-6),
                spec.solver_options.max_iterations.unwrap_or(100),
                spec.solver_options.step_size,
                spec.initial_conditions.t0,
                spec.initial_conditions.t_end,
                y0,
            );
            let solver = BE::try_new_with_options(options).map_err(|error| {
                IvpTaskError::Native(IvpNativeSolverError::BackwardEuler(error))
            })?;
            let mut solver = solver;
            if !spec.equations.parameter_names.is_empty() {
                let names = spec
                    .equations
                    .parameter_names
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>();
                solver
                    .try_set_equation_parameters(Some(&names))
                    .map_err(|error| {
                        IvpTaskError::Native(IvpNativeSolverError::BackwardEuler(error))
                    })?;
                solver
                    .try_set_parameter_values(DVector::from_vec(parameter_values_for_spec(spec)?))
                    .map_err(|error| {
                        IvpTaskError::Native(IvpNativeSolverError::BackwardEuler(error))
                    })?;
            }
            Ok(IvpTaskSolver::BackwardEuler(solver))
        }
        IvpMethodSpec::Radau5 => {
            let tolerance = spec.solver_options.tolerance.unwrap_or(1e-6);
            let mut config = RadauConfig {
                t0: spec.initial_conditions.t0,
                t_bound: spec.initial_conditions.t_end,
                rtol: spec.solver_options.rtol.unwrap_or(tolerance),
                atol: spec.solver_options.atol.unwrap_or(tolerance),
                first_step: spec.solver_options.first_step,
                max_step: spec.solver_options.max_step.unwrap_or(f64::INFINITY),
                max_newton_iterations: spec.solver_options.max_iterations.unwrap_or(6),
                ..RadauConfig::default()
            };
            if let Some(max_steps) = spec.solver_options.max_iterations {
                config.max_steps = max_steps;
            }
            let jacobian = spec
                .equations
                .symbolic_rhs
                .iter()
                .flat_map(|equation| {
                    spec.equations
                        .unknowns
                        .iter()
                        .map(move |variable| equation.diff(variable))
                })
                .collect();
            let problem = RadauProblem::new(
                spec.equations.symbolic_rhs.clone(),
                spec.equations.unknowns.clone(),
                spec.equations.arg.clone(),
            )
            .with_jacobian(jacobian)
            .with_parameters(spec.equations.parameter_names.clone());
            let solver = RadauSolver::prepare(problem, config)
                .map_err(|error| IvpTaskError::Native(IvpNativeSolverError::Radau(error)))?;
            Ok(IvpTaskSolver::Radau {
                solver,
                y0,
                parameters: parameter_values_for_spec(spec)?,
                last_solution: None,
            })
        }
        IvpMethodSpec::Lsode | IvpMethodSpec::Lsoda | IvpMethodSpec::Lsode2 => {
            let config = build_lsode2_problem_config_from_spec(spec)?;
            let solver = Lsode2Solver::new(config)
                .map_err(|error| IvpTaskError::Native(IvpNativeSolverError::Lsode2(error)))?;
            Ok(IvpTaskSolver::Lsode2(solver))
        }
    }
}

fn build_lsode2_problem_config_from_spec(
    spec: &IvpTaskSpec,
) -> Result<Lsode2ProblemConfig, IvpTaskError> {
    let max_step = spec.solver_options.max_step.unwrap_or(1e-3);
    let rtol = spec.solver_options.rtol.unwrap_or(1e-5);
    let atol = spec.solver_options.atol.unwrap_or(1e-8);

    let mut config = Lsode2ProblemConfig::new(
        spec.equations.symbolic_rhs.clone(),
        spec.equations.unknowns.clone(),
        spec.equations.arg.clone(),
        spec.initial_conditions.t0,
        DVector::from_vec(spec.initial_conditions.y0.clone()),
        spec.initial_conditions.t_end,
        max_step,
        rtol,
        atol,
    )
    .with_first_step(spec.solver_options.first_step)
    .with_vectorized(spec.solver_options.vectorized.unwrap_or(false))
    .with_faithful_bdf_solve(200_000, 200_000);

    // Preserve the historical aliases instead of collapsing them at parse
    // time: LSODE is fixed BDF, while LSODA requests automatic Adams/BDF.
    config = if matches!(spec.solver.method, IvpMethodSpec::Lsoda) {
        config.with_automatic_adams_bdf_controller()
    } else {
        config.with_controller(Lsode2ControllerConfig::bdf_only())
    };

    if !spec.equations.parameter_names.is_empty() {
        let mut parameter_values = Vec::with_capacity(spec.equations.parameter_names.len());
        for name in &spec.equations.parameter_names {
            let value = spec
                .equations
                .parameter_values
                .get(name)
                .copied()
                .ok_or_else(|| IvpTaskError::MissingField {
                    section: "equations".to_string(),
                    field: format!("parameter_values[{name}]"),
                })?;
            parameter_values.push(value);
        }
        config = config
            .with_equation_parameters(spec.equations.parameter_names.clone())
            .with_equation_parameter_values(DVector::from_vec(parameter_values));
    }

    if let Some(options) = spec.solver_options.lsode2.as_ref() {
        if let Some(controller) = options.controller {
            config = config.with_controller(controller);
        }
        let symbolic_assembly = options
            .symbolic_assembly
            .unwrap_or(Lsode2SymbolicAssemblyBackend::ExprLegacy);
        let symbolic_execution = options
            .symbolic_execution
            .clone()
            .unwrap_or(Lsode2TaskExecutionSpec::LambdifyExpr);

        match symbolic_execution {
            Lsode2TaskExecutionSpec::LambdifyExpr => {
                config =
                    config.with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
                        assembly: symbolic_assembly,
                        execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
                    });
            }
            Lsode2TaskExecutionSpec::Aot {
                toolchain,
                profile,
                output_parent_dir,
            } => {
                config =
                    config.with_residual_jacobian_source(Lsode2ResidualJacobianSource::Symbolic {
                        assembly: symbolic_assembly,
                        execution: Lsode2SymbolicExecutionMode::Aot { toolchain, profile },
                    });
                if let Some(output_dir) = output_parent_dir {
                    let mut backend = config.backend.clone();
                    backend.generated_backend.output_parent_dir = Some(output_dir);
                    config = config.with_backend(backend);
                }
            }
        }

        if let Some(structure) = options.linear_system_structure {
            config = config.with_linear_system_structure(structure);
        }
        if let Some(policy) = options.linear_solver_policy {
            config = config.with_linear_solver_policy(policy);
        }
        if let Some(native_execution) = options.native_execution {
            config = config.with_native_execution(native_execution);
        }
        for condition in &options.stop_conditions {
            config = match condition.comparator {
                Lsode2StopComparator::GreaterEqual => {
                    config.with_stop_condition_ge(condition.variable.clone(), condition.target)
                }
                Lsode2StopComparator::LessEqual => {
                    config.with_stop_condition_le(condition.variable.clone(), condition.target)
                }
                Lsode2StopComparator::AbsDistance => config.with_stop_condition_abs(
                    condition.variable.clone(),
                    condition.target,
                    condition.tolerance,
                ),
            };
        }
    }

    // Keep parser route symbolic-only to avoid hybrid "document + closures" mode.
    let source = config.residual_jacobian_source;
    config = config.with_residual_jacobian_source(match source {
        Lsode2ResidualJacobianSource::Symbolic {
            assembly,
            execution,
        } => Lsode2ResidualJacobianSource::Symbolic {
            assembly,
            execution,
        },
        Lsode2ResidualJacobianSource::Analytical => Lsode2ResidualJacobianSource::Symbolic {
            assembly: Lsode2SymbolicAssemblyBackend::ExprLegacy,
            execution: Lsode2SymbolicExecutionMode::LambdifyExpr,
        },
    });
    // The task-document route always provides equations, not closures, so the
    // Jacobian backend must be symbolic. Do not call `with_backend` here:
    // that legacy compatibility API re-infers linear policy from backend fields
    // and would turn an explicit `lsode2_linear_solver_policy: auto` into a
    // forced policy after parser-side structure/policy resolution.
    config.backend.jacobian_backend = Lsode2JacobianBackend::SymbolicGenerated;

    Ok(config)
}

fn parameter_values_for_spec(spec: &IvpTaskSpec) -> Result<Vec<f64>, IvpTaskError> {
    spec.equations
        .parameter_names
        .iter()
        .map(|name| {
            spec.equations
                .parameter_values
                .get(name)
                .copied()
                .ok_or_else(|| IvpTaskError::MissingField {
                    section: "equations".to_string(),
                    field: format!("parameter_values[{name}]"),
                })
        })
        .collect()
}

pub fn run_ivp_task_from_str(input: &str) -> Result<IvpTaskRunResult, IvpTaskError> {
    let spec = parse_ivp_task_from_str(input)?;
    run_ivp_task(spec)
}

/// Solve a normalized IVP task and optionally execute lightweight postprocessing.
///
/// Parser-side postprocessing is routed through the unified
/// [`PostprocessPlan`] facade. The historical `plot` flag is still parsed and
/// preserved, but graphical output is only executed for explicit modern
/// actions such as `plotters_png` or `gnuplot_png`.
pub fn run_ivp_task(spec: IvpTaskSpec) -> Result<IvpTaskRunResult, IvpTaskError> {
    if spec.continuation.is_some() {
        let continuation = run_ivp_continuation(spec.clone())?;
        let last = continuation
            .segments
            .last()
            .ok_or_else(|| IvpTaskError::Continuation {
                method: ivp_method_name(&spec.solver.method).to_string(),
                message: "continuation produced no segments".to_string(),
            })?;
        return Ok(IvpTaskRunResult {
            specification: spec,
            t_result: last.t_result.clone(),
            y_result: last.y_result.clone(),
            status: last.status.clone(),
            status_code: last.status_code.clone(),
            trajectory: last.trajectory.clone(),
            continuation: Some(continuation),
        });
    }
    let mut solver = build_ivp_solver_from_spec(&spec)?;
    solver.try_solve()?;
    let (t_result, y_result) = solver.get_result();
    let status = solver.status();
    let status_code = solver.status_code();
    let trajectory = solver.trajectory();
    let plan = spec.postprocessing.to_plan("ivp_result.csv");
    if !plan.actions.is_empty() {
        let dataset = PostprocessDataset::new(
            spec.equations.arg.clone(),
            spec.equations.unknowns.clone(),
            t_result.clone().ok_or_else(|| {
                IvpTaskError::Postprocess(PostprocessError::InvalidDataset(
                    "cannot write postprocessing output because t_result is missing".to_string(),
                ))
            })?,
            y_result.clone().ok_or_else(|| {
                IvpTaskError::Postprocess(PostprocessError::InvalidDataset(
                    "cannot write postprocessing output because y_result is missing".to_string(),
                ))
            })?,
        )
        .map_err(IvpTaskError::Postprocess)?;
        plan.execute(&dataset).map_err(IvpTaskError::Postprocess)?;
    }
    Ok(IvpTaskRunResult {
        specification: spec,
        t_result,
        y_result,
        status,
        status_code,
        trajectory,
        continuation: None,
    })
}

/// Execute a parsed continuation plan while keeping prepared symbolic models
/// alive for `warm` and `prepared` modes. `fresh` deliberately rebuilds every
/// segment and is useful as an independent lifecycle reference.
pub fn run_ivp_continuation(spec: IvpTaskSpec) -> Result<IvpContinuationRunResult, IvpTaskError> {
    let continuation = spec
        .continuation
        .clone()
        .ok_or_else(|| IvpTaskError::Continuation {
            method: ivp_method_name(&spec.solver.method).to_string(),
            message: "continuation section is required".to_string(),
        })?;
    let mut segments = Vec::with_capacity(continuation.value_grid.len());
    let mut fresh_preparations = 0usize;
    let mut prepared_reuses = 0usize;

    match continuation.mode {
        ContinuationMode::Fresh => {
            for (index, parameter_values) in continuation.value_grid.iter().enumerate() {
                let prepare_started = Instant::now();
                let mut segment_spec = spec.clone();
                let rhs = continuation_rhs_for_values(&spec, &continuation, parameter_values);
                segment_spec.equations.rhs = rhs.clone();
                segment_spec.equations.symbolic_rhs = rhs;
                segment_spec.equations.parameter_names.clear();
                segment_spec.equations.parameter_values.clear();
                segment_spec.continuation = None;
                segment_spec.initial_conditions = continuation_segment_initial(
                    &spec.initial_conditions,
                    &continuation,
                    index,
                    continuation.restart_policy,
                );
                let mut solver = build_ivp_solver_from_spec(&segment_spec)?;
                fresh_preparations += 1;
                let prepare_ms = prepare_started.elapsed().as_secs_f64() * 1_000.0;
                let solve_started = Instant::now();
                solver.try_solve()?;
                let solve_ms = solve_started.elapsed().as_secs_f64() * 1_000.0;
                let (t_result, y_result) = solver.get_result();
                segments.push(IvpContinuationSegment {
                    parameter_value: parameter_values[0],
                    parameter_values: parameter_values.clone(),
                    t_result,
                    y_result,
                    status: solver.status(),
                    status_code: solver.status_code(),
                    trajectory: solver.trajectory(),
                    prepare_ms,
                    solve_ms,
                });
            }
        }
        ContinuationMode::Warm | ContinuationMode::Prepared => {
            let prepare_started = Instant::now();
            let mut solver = build_ivp_solver_from_spec(&spec)?;
            fresh_preparations = 1;
            let initial_prepare_ms = prepare_started.elapsed().as_secs_f64() * 1_000.0;
            let initial = &spec.initial_conditions;
            for (index, grid_values) in continuation.value_grid.iter().enumerate() {
                let segment_prepare_ms = if index == 0 { initial_prepare_ms } else { 0.0 };
                let parameter_values =
                    continuation_parameter_values(&spec, &continuation, grid_values)?;
                let segment_initial = continuation_segment_initial(
                    initial,
                    &continuation,
                    index,
                    continuation.restart_policy,
                );
                let restart_each = !matches!(
                    continuation.restart_policy,
                    ContinuationRestartPolicy::Continue
                );
                if index == 0 {
                    let solve_started = Instant::now();
                    solver.try_set_parameter_values(&parameter_values)?;
                    if continuation.y0_values.is_some() || continuation.t0_values.is_some() {
                        solver.try_restart_with_initial_state(&segment_initial)?;
                    }
                    solver.try_solve()?;
                    let solve_ms = solve_started.elapsed().as_secs_f64() * 1_000.0;
                    let (t_result, y_result) = solver.get_result();
                    segments.push(IvpContinuationSegment {
                        parameter_value: grid_values[0],
                        parameter_values: grid_values.clone(),
                        t_result,
                        y_result,
                        status: solver.status(),
                        status_code: solver.status_code(),
                        trajectory: solver.trajectory(),
                        prepare_ms: segment_prepare_ms,
                        solve_ms,
                    });
                    continue;
                } else {
                    prepared_reuses += 1;
                    let solve_started = Instant::now();
                    solver.try_continue_with_parameters(
                        &parameter_values,
                        restart_each,
                        &segment_initial,
                    )?;
                    let solve_ms = solve_started.elapsed().as_secs_f64() * 1_000.0;
                    let (t_result, y_result) = solver.get_result();
                    segments.push(IvpContinuationSegment {
                        parameter_value: grid_values[0],
                        parameter_values: grid_values.clone(),
                        t_result,
                        y_result,
                        status: solver.status(),
                        status_code: solver.status_code(),
                        trajectory: solver.trajectory(),
                        prepare_ms: segment_prepare_ms,
                        solve_ms,
                    });
                }
            }
        }
    }

    Ok(IvpContinuationRunResult {
        parameter: continuation.parameter,
        parameters: continuation.parameters,
        mode: continuation.mode,
        restart_each: !matches!(
            continuation.restart_policy,
            ContinuationRestartPolicy::Continue
        ),
        fresh_preparations,
        prepared_reuses,
        segments,
    })
}

fn continuation_parameter_values(
    spec: &IvpTaskSpec,
    continuation: &ContinuationSpec,
    grid_values: &[f64],
) -> Result<Vec<f64>, IvpTaskError> {
    let mut values = parameter_values_for_spec(spec)?;
    for (parameter, value) in continuation.parameters.iter().zip(grid_values) {
        let index = spec
            .equations
            .parameter_names
            .iter()
            .position(|name| name == parameter)
            .ok_or_else(|| IvpTaskError::Continuation {
                method: ivp_method_name(&spec.solver.method).to_string(),
                message: format!(
                    "continuation parameter `{parameter}` disappeared from the prepared model"
                ),
            })?;
        values[index] = *value;
    }
    Ok(values)
}

fn continuation_segment_initial(
    initial: &InitialConditionSpec,
    continuation: &ContinuationSpec,
    index: usize,
    restart_policy: ContinuationRestartPolicy,
) -> InitialConditionSpec {
    let y0 = continuation
        .y0_values
        .as_ref()
        .and_then(|values| values.get(if values.len() == 1 { 0 } else { index }))
        .cloned()
        .unwrap_or_else(|| initial.y0.clone());
    let t0 = continuation
        .t0_values
        .as_ref()
        .and_then(|values| values.get(if values.len() == 1 { 0 } else { index }))
        .copied()
        .unwrap_or(initial.t0);
    if !matches!(restart_policy, ContinuationRestartPolicy::Continue)
        || index == 0
        || continuation.t_end_values.is_some()
        || continuation.t0_values.is_some()
        || continuation.y0_values.is_some()
    {
        return InitialConditionSpec {
            t0,
            t_end: continuation
                .t_end_values
                .as_ref()
                .and_then(|values| values.get(if values.len() == 1 { 0 } else { index }))
                .copied()
                .unwrap_or(initial.t_end),
            y0,
        };
    }
    let duration = initial.t_end - initial.t0;
    InitialConditionSpec {
        t0: initial.t0,
        t_end: initial.t0 + duration * (index as f64 + 1.0),
        y0,
    }
}

fn continuation_rhs_for_values(
    spec: &IvpTaskSpec,
    continuation: &ContinuationSpec,
    values: &[f64],
) -> Vec<Expr> {
    let mut bindings = spec.equations.parameter_values.clone();
    for (parameter, value) in continuation.parameters.iter().zip(values) {
        bindings.insert(parameter.clone(), *value);
    }
    continuation
        .symbolic_rhs
        .iter()
        .cloned()
        .map(|expr| expr.set_variable_from_map(&bindings))
        .collect()
}

/// Write a starter IVP task document template to disk or to the current folder.
pub fn create_ivp_template_file(path: Option<PathBuf>) {
    use std::env;
    use std::fs::File;
    use std::io::Write;

    let template = r#"
task
solver: IVP
method: BDF

equations
arg: t
parameters: a
parameter_values: 1.0
y: -a*y

initial_conditions
t0: 0.0
t_end: 1.0
y0: 1.0

solver_options
rtol: 1e-6
atol: 1e-8
max_step: 0.1
first_step: Some(1e-4)
parallel: false

postprocessing
save_csv: false
csv_path: ivp_result.csv
save_txt: false
txt_path: ivp_result.txt
write_report: false
report_path: ivp_report.md
plotters_png: false
plotters_dir: ivp_plotters
gnuplot_png: false
gnuplot_dir: ivp_gnuplot
terminal_plot: false
plot: false
"#;

    let file_path = path.unwrap_or_else(|| {
        let mut default_path = env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
        default_path.push("ivp_task_template.txt");
        default_path
    });

    match File::create(&file_path) {
        Ok(mut file) => {
            if let Err(err) = file.write_all(template.as_bytes()) {
                eprintln!("Failed to write IVP template file: {err}");
            }
        }
        Err(err) => eprintln!("Failed to create IVP template file: {err}"),
    }
}

/// Parse the `task` section and select the IVP method family.
fn parse_solver_selection(document: &DocumentMap) -> Result<SolverSelectionSpec, IvpTaskError> {
    let task_section = get_required_section(document, "task")?;
    let solver_name = get_required_string(task_section, "task", "solver")?;
    if !solver_name.eq_ignore_ascii_case("ivp") {
        return Err(IvpTaskError::InvalidField {
            section: "task".to_string(),
            field: "solver".to_string(),
            message: format!("expected `IVP`, got `{solver_name}`"),
        });
    }
    let method = IvpMethodSpec::from_str(&get_required_string(task_section, "task", "method")?)?;
    Ok(SolverSelectionSpec {
        task_kind: TaskKindSpec::Ivp,
        method,
    })
}

/// Parse the symbolic ODE system and substitute constant parameters.
///
/// We support two equivalent user-facing forms:
/// - `unknowns` + `rhs`
/// - pair-style `y: -a*y`
///
/// The second form is often nicer for humans, so we preserve variable names in
/// the `equations` section when lowercasing other configuration keys.
fn parse_equations(document: &DocumentMap) -> Result<EquationSpec, IvpTaskError> {
    let parsed = parse_symbolic_equation_system(document, "t").map_err(map_equation_error)?;
    Ok(EquationSpec {
        arg: parsed.arg,
        unknowns: parsed.unknowns,
        rhs: parsed.rhs,
        symbolic_rhs: parsed.symbolic_rhs,
        parameter_names: parsed.parameter_names,
        parameter_values: parsed.parameter_values,
    })
}

fn map_equation_error(error: SharedEquationParseError) -> IvpTaskError {
    match error {
        SharedEquationParseError::MissingSection(section) => IvpTaskError::MissingSection(section),
        SharedEquationParseError::MissingField { section, field } => {
            IvpTaskError::MissingField { section, field }
        }
        SharedEquationParseError::InvalidField {
            section,
            field,
            message,
        } => IvpTaskError::InvalidField {
            section,
            field,
            message,
        },
        SharedEquationParseError::InconsistentEquationCounts { unknowns, rhs } => {
            IvpTaskError::InconsistentEquationCounts { unknowns, rhs }
        }
        SharedEquationParseError::Semantic(message) => IvpTaskError::Semantic(message),
    }
}

/// Parse `t0`, `t_end`, and the initial state vector.
fn parse_initial_conditions(
    document: &DocumentMap,
    expected_dimension: usize,
) -> Result<InitialConditionSpec, IvpTaskError> {
    let section = get_required_section(document, "initial_conditions")?;
    let t0 = get_required_float(section, "initial_conditions", "t0")?;
    let t_end = get_required_float(section, "initial_conditions", "t_end")?;
    let y0 = get_required_float_list(section, "initial_conditions", "y0")?;
    if y0.len() != expected_dimension {
        return Err(IvpTaskError::InvalidField {
            section: "initial_conditions".to_string(),
            field: "y0".to_string(),
            message: format!(
                "expected {expected_dimension} initial values, got {}",
                y0.len()
            ),
        });
    }
    Ok(InitialConditionSpec { t0, t_end, y0 })
}

/// Parse optional solver controls shared by the nonstiff facade and native
/// stiff adapters. Each adapter consumes only the options it supports.
fn parse_solver_options(document: &DocumentMap) -> Result<IvpSolverOptionsSpec, IvpTaskError> {
    let section = match document.get("solver_options") {
        Some(section) => section,
        None => return Ok(IvpSolverOptionsSpec::default()),
    };
    let lsode2 = parse_lsode2_options(section)?;
    Ok(IvpSolverOptionsSpec {
        step_size: get_optional_float(section, "step_size")?,
        tolerance: get_optional_float(section, "tolerance")?,
        max_iterations: get_optional_usize(section, "max_iterations")?,
        rtol: get_optional_float(section, "rtol")?,
        atol: get_optional_float(section, "atol")?,
        max_step: get_optional_float(section, "max_step")?,
        first_step: get_optional_float_or_option(section, "first_step")?,
        vectorized: get_optional_bool(section, "vectorized")?,
        parallel: get_optional_bool(section, "parallel")?,
        neighborhood_check: get_optional_float(section, "neighborhood_check")?,
        lsode2,
    })
}

fn parse_lsode2_options(
    section: &GenericSectionMap,
) -> Result<Option<Lsode2TaskOptionsSpec>, IvpTaskError> {
    let controller = parse_lsode2_controller(section)?;
    let assembly = match get_optional_string(section, "lsode2_symbolic_assembly", "solver_options")?
    {
        Some(raw) => Some(parse_lsode2_symbolic_assembly(&raw)?),
        None => None,
    };
    let execution = parse_lsode2_execution(section)?;
    let linear_structure = parse_lsode2_linear_structure(section)?;
    let linear_policy = parse_lsode2_linear_solver_policy(section)?;
    let native_execution = parse_lsode2_native_execution(section)?;
    let stop_conditions = parse_lsode2_stop_conditions(section)?;

    let has_any = assembly.is_some()
        || execution.is_some()
        || controller.is_some()
        || linear_structure.is_some()
        || linear_policy.is_some()
        || native_execution.is_some()
        || !stop_conditions.is_empty();
    if !has_any {
        return Ok(None);
    }

    Ok(Some(Lsode2TaskOptionsSpec {
        controller,
        symbolic_assembly: assembly,
        symbolic_execution: execution,
        linear_system_structure: linear_structure,
        linear_solver_policy: linear_policy,
        native_execution,
        stop_conditions,
    }))
}

/// Parse lightweight output controls.
fn parse_postprocessing(document: &DocumentMap) -> Result<PostprocessingSpec, IvpTaskError> {
    let section = match document.get("postprocessing") {
        Some(section) => section,
        None => return Ok(PostprocessingSpec::default()),
    };
    Ok(PostprocessingSpec {
        save_csv: get_optional_bool(section, "save_csv")?.unwrap_or(false),
        csv_path: get_optional_string(section, "csv_path", "postprocessing")?,
        save_txt: get_optional_bool(section, "save_txt")?.unwrap_or(false),
        txt_path: get_optional_string(section, "txt_path", "postprocessing")?,
        write_report: get_optional_bool(section, "write_report")?.unwrap_or(false),
        report_path: get_optional_string(section, "report_path", "postprocessing")?,
        plotters_png: get_optional_bool(section, "plotters_png")?.unwrap_or(false),
        plotters_dir: get_optional_string(section, "plotters_dir", "postprocessing")?,
        gnuplot_png: get_optional_bool(section, "gnuplot_png")?.unwrap_or(false),
        gnuplot_dir: get_optional_string(section, "gnuplot_dir", "postprocessing")?,
        terminal_plot: get_optional_bool(section, "terminal_plot")?.unwrap_or(false),
        plot: get_optional_bool(section, "plot")?.unwrap_or(false),
    })
}

/// Provide a small set of forgiving aliases for the text format.
fn default_ivp_pseudonyms() -> (HashMap<String, Vec<String>>, HashMap<String, Vec<String>>) {
    let headers = HashMap::from([
        (
            "task".to_string(),
            vec!["problem".to_string(), "solver_selection".to_string()],
        ),
        (
            "equations".to_string(),
            vec!["system".to_string(), "ode_system".to_string()],
        ),
        (
            "where".to_string(),
            vec!["substitute".to_string(), "aliases".to_string()],
        ),
        (
            "initial_conditions".to_string(),
            vec!["initial".to_string(), "iv".to_string()],
        ),
        (
            "solver_options".to_string(),
            vec!["solver_settings".to_string(), "options".to_string()],
        ),
    ]);
    let fields = HashMap::from([
        (
            "method".to_string(),
            vec!["ivp_method".to_string(), "solver_method".to_string()],
        ),
        (
            "parameters".to_string(),
            vec!["params".to_string(), "parameter_names".to_string()],
        ),
        (
            "parameter_values".to_string(),
            vec!["params_values".to_string(), "param_values".to_string()],
        ),
        (
            "t_end".to_string(),
            vec!["tbound".to_string(), "t_bound".to_string()],
        ),
    ]);
    (headers, fields)
}

fn parse_lsode2_symbolic_assembly(
    raw: &str,
) -> Result<Lsode2SymbolicAssemblyBackend, IvpTaskError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "exprlegacy" | "expr_legacy" => Ok(Lsode2SymbolicAssemblyBackend::ExprLegacy),
        "atomview" | "atom_view" => Ok(Lsode2SymbolicAssemblyBackend::AtomView),
        other => Err(IvpTaskError::InvalidField {
            section: "solver_options".to_string(),
            field: "lsode2_symbolic_assembly".to_string(),
            message: format!(
                "unknown LSODE2 symbolic assembly `{other}` (use ExprLegacy or AtomView)"
            ),
        }),
    }
}

fn parse_lsode2_controller(
    section: &GenericSectionMap,
) -> Result<Option<Lsode2ControllerConfig>, IvpTaskError> {
    let raw = match get_optional_string(section, "lsode2_method_family", "solver_options")? {
        Some(value) => value,
        None => match get_optional_string(section, "lsode2_controller", "solver_options")? {
            Some(value) => value,
            None => return Ok(None),
        },
    };
    let value = raw.trim().to_ascii_lowercase().replace(['-', ' '], "_");
    let controller = match value.as_str() {
        "auto" | "automatic" | "automatic_adams_bdf" | "lsoda" | "adams_bdf" => {
            Lsode2ControllerConfig::automatic_adams_bdf()
        }
        "adams" | "adams_only" | "lsode_adams" => Lsode2ControllerConfig::adams_only(),
        "bdf" | "bdf_only" | "lsode_bdf" | "lsode" => Lsode2ControllerConfig::bdf_only(),
        other => {
            return Err(IvpTaskError::InvalidField {
                section: "solver_options".to_string(),
                field: "lsode2_method_family".to_string(),
                message: format!("unknown LSODE2 method family `{other}` (use auto, adams or bdf)"),
            });
        }
    };
    Ok(Some(controller))
}

fn parse_lsode2_execution(
    section: &GenericSectionMap,
) -> Result<Option<Lsode2TaskExecutionSpec>, IvpTaskError> {
    let raw = match get_optional_string(section, "lsode2_symbolic_execution", "solver_options")? {
        Some(value) => value,
        None => return Ok(None),
    };
    let key = raw.trim().to_ascii_lowercase();
    if matches!(key.as_str(), "lambdify" | "lambdifyexpr" | "lambdify_expr") {
        return Ok(Some(Lsode2TaskExecutionSpec::LambdifyExpr));
    }
    if !matches!(key.as_str(), "aot") {
        return Err(IvpTaskError::InvalidField {
            section: "solver_options".to_string(),
            field: "lsode2_symbolic_execution".to_string(),
            message: format!("unknown LSODE2 symbolic execution `{raw}` (use LambdifyExpr or AOT)"),
        });
    }

    let toolchain_raw = get_optional_string(section, "lsode2_aot_toolchain", "solver_options")?
        .unwrap_or_else(|| "c_gcc".to_string());
    let profile_raw = get_optional_string(section, "lsode2_aot_profile", "solver_options")?
        .unwrap_or_else(|| "release".to_string());
    let output_parent_dir =
        get_optional_string(section, "lsode2_aot_output_dir", "solver_options")?.map(PathBuf::from);

    let toolchain = match toolchain_raw.trim().to_ascii_lowercase().as_str() {
        "c_tcc" | "ctcc" => Lsode2AotToolchain::CTcc,
        "c_gcc" | "cgcc" | "gcc" => Lsode2AotToolchain::CGcc,
        "zig" => Lsode2AotToolchain::Zig,
        "rust" => Lsode2AotToolchain::Rust,
        other => {
            return Err(IvpTaskError::InvalidField {
                section: "solver_options".to_string(),
                field: "lsode2_aot_toolchain".to_string(),
                message: format!("unknown LSODE2 AOT toolchain `{other}`"),
            });
        }
    };
    let profile = match profile_raw.trim().to_ascii_lowercase().as_str() {
        "debug" => Lsode2AotProfile::Debug,
        "release" => Lsode2AotProfile::Release,
        other => {
            return Err(IvpTaskError::InvalidField {
                section: "solver_options".to_string(),
                field: "lsode2_aot_profile".to_string(),
                message: format!("unknown LSODE2 AOT profile `{other}`"),
            });
        }
    };

    Ok(Some(Lsode2TaskExecutionSpec::Aot {
        toolchain,
        profile,
        output_parent_dir,
    }))
}

fn parse_lsode2_linear_structure(
    section: &GenericSectionMap,
) -> Result<Option<Lsode2LinearSystemStructure>, IvpTaskError> {
    let raw = match get_optional_string(section, "lsode2_linear_structure", "solver_options")? {
        Some(value) => value,
        None => return Ok(None),
    };
    let value = raw.trim().to_ascii_lowercase();
    match value.as_str() {
        "dense" => Ok(Some(Lsode2LinearSystemStructure::Dense)),
        "sparse" => Ok(Some(Lsode2LinearSystemStructure::Sparse)),
        "banded" => {
            let kl = get_optional_usize(section, "lsode2_banded_kl")?.unwrap_or(0);
            let ku = get_optional_usize(section, "lsode2_banded_ku")?.unwrap_or(0);
            Ok(Some(Lsode2LinearSystemStructure::Banded { kl, ku }))
        }
        other => Err(IvpTaskError::InvalidField {
            section: "solver_options".to_string(),
            field: "lsode2_linear_structure".to_string(),
            message: format!(
                "unknown LSODE2 linear structure `{other}` (use dense, sparse or banded)"
            ),
        }),
    }
}

fn parse_lsode2_linear_solver_policy(
    section: &GenericSectionMap,
) -> Result<Option<Lsode2LinearSolverPolicy>, IvpTaskError> {
    let raw = match get_optional_string(section, "lsode2_linear_solver_policy", "solver_options")? {
        Some(value) => value,
        None => return Ok(None),
    };
    let value = raw.trim().to_ascii_lowercase();
    let policy = match value.as_str() {
        "auto" => Lsode2LinearSolverPolicy::Auto,
        "dense_lu" | "denselu" => {
            Lsode2LinearSolverPolicy::Force(Lsode2LinearSolverChoice::DenseLu)
        }
        "faer_sparse_lu" | "faersparselu" => {
            Lsode2LinearSolverPolicy::Force(Lsode2LinearSolverChoice::FaerSparseLu)
        }
        "lapack_faithful_banded_lu" | "lapackfaithfulbandedlu" => {
            Lsode2LinearSolverPolicy::Force(Lsode2LinearSolverChoice::LapackFaithfulBandedLu)
        }
        other => {
            return Err(IvpTaskError::InvalidField {
                section: "solver_options".to_string(),
                field: "lsode2_linear_solver_policy".to_string(),
                message: format!("unknown LSODE2 linear solver policy `{other}`"),
            });
        }
    };
    Ok(Some(policy))
}

fn parse_lsode2_native_execution(
    section: &GenericSectionMap,
) -> Result<Option<Lsode2NativeExecutionConfig>, IvpTaskError> {
    let raw = match get_optional_string(section, "lsode2_native_execution", "solver_options")? {
        Some(value) => value,
        None => return Ok(None),
    };
    let max_step_attempts =
        get_optional_usize(section, "lsode2_native_max_step_attempts")?.unwrap_or(200_000);
    let max_accepted_steps =
        get_optional_usize(section, "lsode2_native_max_accepted_steps")?.unwrap_or(200_000);
    let value = raw.trim().to_ascii_lowercase();
    let mode = match value.as_str() {
        "faithful_bdf_solve" | "native_solve" => {
            Lsode2NativeExecutionConfig::faithful_bdf_solve(max_step_attempts, max_accepted_steps)
        }
        "probe_before_bridge" | "native_probe_before_bridge" => {
            Lsode2NativeExecutionConfig::probe_before_bridge(max_step_attempts, max_accepted_steps)
        }
        "bridge_solve" => Lsode2NativeExecutionConfig::bridge_solve(),
        other => {
            return Err(IvpTaskError::InvalidField {
                section: "solver_options".to_string(),
                field: "lsode2_native_execution".to_string(),
                message: format!("unknown LSODE2 native execution mode `{other}`"),
            });
        }
    };
    Ok(Some(mode))
}

/// Parse one LSODE2 stop condition from task-document fields.
///
/// The condition is intentionally explicit rather than encoded in one compact
/// string so malformed target values and unknown comparators remain typed
/// task-parser errors:
///
/// ```text
/// lsode2_stop_variable: eta_ox
/// lsode2_stop_comparator: le
/// lsode2_stop_target: 1e-3
/// lsode2_stop_tolerance: 0.0
/// ```
fn parse_lsode2_stop_conditions(
    section: &GenericSectionMap,
) -> Result<Vec<Lsode2StopCondition>, IvpTaskError> {
    const FIELDS: [&str; 4] = [
        "lsode2_stop_variable",
        "lsode2_stop_comparator",
        "lsode2_stop_target",
        "lsode2_stop_tolerance",
    ];
    let variable = get_optional_string(section, "lsode2_stop_variable", "solver_options")?;
    let Some(variable) = variable else {
        if FIELDS
            .iter()
            .skip(1)
            .any(|field| section.contains_key(*field))
        {
            return Err(IvpTaskError::MissingField {
                section: "solver_options".to_string(),
                field: "lsode2_stop_variable".to_string(),
            });
        }
        return Ok(Vec::new());
    };

    let target = get_optional_float(section, "lsode2_stop_target")?.ok_or_else(|| {
        IvpTaskError::MissingField {
            section: "solver_options".to_string(),
            field: "lsode2_stop_target".to_string(),
        }
    })?;
    let comparator_raw = get_optional_string(section, "lsode2_stop_comparator", "solver_options")?
        .unwrap_or_else(|| "ge".to_string());
    let comparator = match comparator_raw.trim().to_ascii_lowercase().as_str() {
        "ge" | ">=" | "greater_equal" | "greater_or_equal" => Lsode2StopComparator::GreaterEqual,
        "le" | "<=" | "less_equal" | "less_or_equal" => Lsode2StopComparator::LessEqual,
        "abs" | "abs_distance" | "distance" => Lsode2StopComparator::AbsDistance,
        other => {
            return Err(IvpTaskError::InvalidField {
                section: "solver_options".to_string(),
                field: "lsode2_stop_comparator".to_string(),
                message: format!(
                    "unknown LSODE2 stop comparator `{other}` (use ge, le, or abs_distance)"
                ),
            });
        }
    };
    let tolerance = get_optional_float(section, "lsode2_stop_tolerance")?.unwrap_or(0.0);

    Ok(vec![Lsode2StopCondition {
        variable,
        target,
        comparator,
        tolerance: tolerance.abs(),
    }])
}

fn get_required_section<'a>(
    document: &'a DocumentMap,
    section: &'static str,
) -> Result<&'a GenericSectionMap, IvpTaskError> {
    document
        .get(section)
        .ok_or(IvpTaskError::MissingSection(section))
}

fn get_required_values<'a>(
    section: &'a GenericSectionMap,
    section_name: &str,
    field: &str,
) -> Result<&'a Vec<Value>, IvpTaskError> {
    section
        .get(field)
        .ok_or_else(|| IvpTaskError::MissingField {
            section: section_name.to_string(),
            field: field.to_string(),
        })?
        .as_ref()
        .ok_or_else(|| IvpTaskError::MissingField {
            section: section_name.to_string(),
            field: field.to_string(),
        })
}

fn get_required_string(
    section: &GenericSectionMap,
    section_name: &str,
    field: &str,
) -> Result<String, IvpTaskError> {
    let values = get_required_values(section, section_name, field)?;
    if values.len() != 1 {
        return Err(IvpTaskError::InvalidField {
            section: section_name.to_string(),
            field: field.to_string(),
            message: "expected a single string value".to_string(),
        });
    }
    value_to_string(&values[0], section_name, field)
}

fn get_optional_string(
    section: &GenericSectionMap,
    field: &str,
    section_name: &str,
) -> Result<Option<String>, IvpTaskError> {
    match section.get(field) {
        Some(Some(values)) if !values.is_empty() => {
            if values.len() != 1 {
                return Err(IvpTaskError::InvalidField {
                    section: section_name.to_string(),
                    field: field.to_string(),
                    message: "expected a single string value".to_string(),
                });
            }
            Ok(Some(value_to_string(&values[0], section_name, field)?))
        }
        _ => Ok(None),
    }
}

fn get_required_float(
    section: &GenericSectionMap,
    section_name: &str,
    field: &str,
) -> Result<f64, IvpTaskError> {
    let values = get_required_values(section, section_name, field)?;
    if values.len() != 1 {
        return Err(IvpTaskError::InvalidField {
            section: section_name.to_string(),
            field: field.to_string(),
            message: "expected a single numeric value".to_string(),
        });
    }
    value_to_float(&values[0], section_name, field)
}

fn get_optional_float(
    section: &GenericSectionMap,
    field: &str,
) -> Result<Option<f64>, IvpTaskError> {
    match section.get(field) {
        Some(Some(values)) if !values.is_empty() => {
            if values.len() != 1 {
                return Err(IvpTaskError::InvalidField {
                    section: "solver_options".to_string(),
                    field: field.to_string(),
                    message: "expected a single numeric value".to_string(),
                });
            }
            Ok(Some(value_to_float(&values[0], "solver_options", field)?))
        }
        _ => Ok(None),
    }
}

fn get_optional_float_or_option(
    section: &GenericSectionMap,
    field: &str,
) -> Result<Option<f64>, IvpTaskError> {
    match section.get(field) {
        Some(Some(values)) if !values.is_empty() => {
            if values.len() != 1 {
                return Err(IvpTaskError::InvalidField {
                    section: "solver_options".to_string(),
                    field: field.to_string(),
                    message: "expected a single numeric or optional numeric value".to_string(),
                });
            }
            match &values[0] {
                Value::Optional(None) => Ok(None),
                Value::Optional(Some(_)) => Ok(values[0].as_option_float()),
                _ => Ok(Some(value_to_float(&values[0], "solver_options", field)?)),
            }
        }
        _ => Ok(None),
    }
}

fn get_optional_usize(
    section: &GenericSectionMap,
    field: &str,
) -> Result<Option<usize>, IvpTaskError> {
    match section.get(field) {
        Some(Some(values)) if !values.is_empty() => {
            if values.len() != 1 {
                return Err(IvpTaskError::InvalidField {
                    section: "solver_options".to_string(),
                    field: field.to_string(),
                    message: "expected a single integer value".to_string(),
                });
            }
            values[0]
                .as_usize()
                .ok_or_else(|| IvpTaskError::InvalidField {
                    section: "solver_options".to_string(),
                    field: field.to_string(),
                    message: "expected usize".to_string(),
                })
                .map(Some)
        }
        _ => Ok(None),
    }
}

fn get_optional_bool(
    section: &GenericSectionMap,
    field: &str,
) -> Result<Option<bool>, IvpTaskError> {
    match section.get(field) {
        Some(Some(values)) if !values.is_empty() => {
            if values.len() != 1 {
                return Err(IvpTaskError::InvalidField {
                    section: "solver_options".to_string(),
                    field: field.to_string(),
                    message: "expected a single boolean value".to_string(),
                });
            }
            values[0]
                .as_boolean()
                .ok_or_else(|| IvpTaskError::InvalidField {
                    section: "solver_options".to_string(),
                    field: field.to_string(),
                    message: "expected bool".to_string(),
                })
                .map(Some)
        }
        _ => Ok(None),
    }
}

fn get_required_float_list(
    section: &GenericSectionMap,
    section_name: &str,
    field: &str,
) -> Result<Vec<f64>, IvpTaskError> {
    let values = get_required_values(section, section_name, field)?;
    values_to_float_list(values, section_name, field)
}

fn values_to_float_list(
    values: &[Value],
    section_name: &str,
    field: &str,
) -> Result<Vec<f64>, IvpTaskError> {
    if values.len() == 1 {
        if let Some(vector) = values[0].as_vector() {
            return Ok(vector.clone());
        }
    }
    values
        .iter()
        .map(|value| value_to_float(value, section_name, field))
        .collect()
}

fn value_to_string(value: &Value, section_name: &str, field: &str) -> Result<String, IvpTaskError> {
    if let Some(text) = value.as_string() {
        Ok(text.clone())
    } else {
        Err(IvpTaskError::InvalidField {
            section: section_name.to_string(),
            field: field.to_string(),
            message: "expected string".to_string(),
        })
    }
}

fn value_to_float(value: &Value, section_name: &str, field: &str) -> Result<f64, IvpTaskError> {
    if let Some(number) = value.as_float() {
        Ok(number)
    } else if let Some(integer) = value.as_usize() {
        Ok(integer as f64)
    } else {
        Err(IvpTaskError::InvalidField {
            section: section_name.to_string(),
            field: field.to_string(),
            message: "expected numeric value".to_string(),
        })
    }
}

#[cfg(test)]
#[path = "task_parser_ivp_tests.rs"]
mod task_parser_ivp_tests;
