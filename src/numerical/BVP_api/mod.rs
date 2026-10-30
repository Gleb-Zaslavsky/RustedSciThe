//! Typed orchestration over the production boundary-value solver APIs.
//!
//! This facade owns routing and shared problem validation only. Solver
//! configuration, diagnostics, telemetry, and native result types stay with
//! their respective solver families. Task-document parsing is intentionally a
//! separate subsystem and calls native solvers directly.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use crate::Utils::postprocessing::{
    PostprocessDataset, PostprocessError, PostprocessPlan, PostprocessReport,
};
use crate::numerical::BVP_Damp::NR_Damp_solver_damped::{
    DampedParameterLoopReport, DampedSolverOptions, NRBVP as DampedSolver,
};
use crate::numerical::BVP_Damp::NR_Damp_solver_frozen::{
    FrozenParameterLoopReport, FrozenSolverOptions, NRBVP as FrozenSolver,
};
use crate::numerical::BVP_Damp::telemetry::BvpTelemetrySnapshot;
use crate::numerical::BVP_sci::new::{
    BvpSciAssembly, BvpSciExecution, BvpSciNewError, BvpSciOptions, BvpSciParameterLoopReport,
    BvpSciSolution, BvpSciSolver, BvpSciSparseFactorization as NativeBvpSciSparseFactorization,
    BvpSciTelemetryMode, BvpSciTelemetrySnapshot,
};
use crate::numerical::logging::SolverLogLevel;
use crate::symbolic::codegen::codegen_backend_selection::BackendSelectionPolicy;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions_BVP::BvpBackendIntegrationError;
use crate::symbolic::symbolic_ivp_generated::SymbolicIvpGeneratedBackendConfig;
use nalgebra::{DMatrix, DVector};
use thiserror::Error;

#[cfg(test)]
#[path = "test.rs"]
mod tests;

/// Endpoint of a scalar boundary condition for a first-order system.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BvpBoundarySide {
    Left,
    Right,
}

/// One state value constrained at one endpoint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BvpBoundaryCondition {
    /// Index into [`BvpProblem`]'s ordered state list.
    pub state: usize,
    /// Boundary endpoint where the state is constrained.
    pub side: BvpBoundarySide,
    /// Required state value at that endpoint.
    pub value: f64,
}

/// Production-supported sparse factorization choices exposed by this facade.
///
/// Experimental reusable-faer variants remain available through the direct
/// BVP_sci API, but are intentionally excluded here until release evidence
/// justifies promoting them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BvpSciSparseFactorization {
    /// Established native faer sparse factorization (the default).
    #[default]
    StandardFaer,
    /// Optional native SuperLU bridge with its documented fallback behavior.
    SuperLu,
}

impl From<BvpSciSparseFactorization> for NativeBvpSciSparseFactorization {
    fn from(value: BvpSciSparseFactorization) -> Self {
        match value {
            BvpSciSparseFactorization::StandardFaer => Self::StandardFaer,
            BvpSciSparseFactorization::SuperLu => Self::SuperLu,
        }
    }
}

type ExtraBoundaryResidual =
    dyn Fn(&[f64], &[f64], &[f64], &mut [f64]) -> Result<(), String> + Send + Sync;
type ExtraBoundaryJacobian = dyn Fn(&[f64], &[f64], &[f64], &mut [f64], &mut [f64], &mut [f64]) -> Result<(), String>
    + Send
    + Sync;

/// Free BVP_sci parameters and the extra boundary equations that identify them.
///
/// The residual writes exactly one value per parameter. The optional Jacobian
/// writes row-major blocks shaped `p x n_state`, `p x n_state`, and `p x p`.
#[derive(Clone)]
pub struct BvpSciFreeParameters {
    names: Vec<String>,
    initial_guess: Vec<f64>,
    residual: Arc<ExtraBoundaryResidual>,
    jacobian: Option<Arc<ExtraBoundaryJacobian>>,
}

impl BvpSciFreeParameters {
    /// Create a free-parameter block with its additional boundary residuals.
    pub fn new<F>(names: Vec<String>, initial_guess: Vec<f64>, residual: F) -> Self
    where
        F: Fn(&[f64], &[f64], &[f64], &mut [f64]) -> Result<(), String> + Send + Sync + 'static,
    {
        Self {
            names,
            initial_guess,
            residual: Arc::new(residual),
            jacobian: None,
        }
    }

    /// Supply analytical Jacobian rows for the additional parameter equations.
    pub fn with_jacobian<J>(mut self, jacobian: J) -> Self
    where
        J: Fn(&[f64], &[f64], &[f64], &mut [f64], &mut [f64], &mut [f64]) -> Result<(), String>
            + Send
            + Sync
            + 'static,
    {
        self.jacobian = Some(Arc::new(jacobian));
        self
    }

    fn validate(&self, problem: &BvpProblem) -> Result<(), BvpApiError> {
        if self.names.is_empty() || self.names.len() != self.initial_guess.len() {
            return Err(BvpApiError::InvalidProblem(
                "BVP_sci free parameter names and guesses must have equal non-zero lengths".into(),
            ));
        }
        if self.initial_guess.iter().any(|value| !value.is_finite()) {
            return Err(BvpApiError::InvalidProblem(
                "BVP_sci free parameter guesses must be finite".into(),
            ));
        }
        let mut seen = HashSet::with_capacity(self.names.len());
        if self.names.iter().any(|name| {
            name.trim().is_empty()
                || name == &problem.independent_variable
                || problem.state_names.contains(name)
                || problem
                    .external_parameters
                    .as_ref()
                    .is_some_and(|external| external.names.contains(name))
                || !seen.insert(name.as_str())
        }) {
            return Err(BvpApiError::InvalidProblem(
                "BVP_sci free parameter names must be unique and distinct from states and the independent variable".into(),
            ));
        }
        Ok(())
    }
}

impl fmt::Debug for BvpSciFreeParameters {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BvpSciFreeParameters")
            .field("names", &self.names)
            .field("initial_guess", &self.initial_guess)
            .field("has_analytical_jacobian", &self.jacobian.is_some())
            .finish_non_exhaustive()
    }
}

/// Shared symbolic problem data independent of solver-specific guess layout.
///
/// Equations correspond to states by index. The mesh is explicit and strictly
/// increasing. Solver-specific initial guesses live in [`BvpSolverConfig`]
/// because native routes intentionally use different column conventions.
#[derive(Clone, Debug)]
pub struct BvpProblem {
    equations: Vec<Expr>,
    state_names: Vec<String>,
    independent_variable: String,
    mesh: Vec<f64>,
    boundary_conditions: Vec<BvpBoundaryCondition>,
    external_parameters: Option<BvpSciExternalParameters>,
}

/// Fixed symbolic model inputs for BVP_sci. The schema is supplied before
/// preparation; parameter-loop points replace values without rebuilding it.
#[derive(Clone, Debug, PartialEq)]
pub struct BvpSciExternalParameters {
    names: Vec<String>,
    values: Vec<f64>,
}

impl BvpProblem {
    /// Validate a first-order BVP with one scalar endpoint constraint per equation.
    ///
    /// Conditions may constrain the same state at both endpoints. The facade
    /// validates the total condition count and duplicate endpoint/state pairs;
    /// the native solver remains responsible for problem-specific solvability.
    pub fn new(
        equations: Vec<Expr>,
        state_names: Vec<String>,
        independent_variable: impl Into<String>,
        mesh: Vec<f64>,
        boundary_conditions: Vec<BvpBoundaryCondition>,
    ) -> Result<Self, BvpApiError> {
        let problem = Self {
            equations,
            state_names,
            independent_variable: independent_variable.into(),
            mesh,
            boundary_conditions,
            external_parameters: None,
        };
        problem.validate()?;
        Ok(problem)
    }

    /// Declare fixed BVP_sci inputs before constructing the solver.
    ///
    /// This is separate from [`BvpSciFreeParameters`], which adds unknowns to
    /// the Newton system. The declared evaluator schema is prepared once;
    /// `solve_parameter_loop` supplies successive values for that schema.
    pub fn with_external_parameters(
        mut self,
        names: Vec<String>,
        values: Vec<f64>,
    ) -> Result<Self, BvpApiError> {
        if names.is_empty() || names.len() != values.len() {
            return Err(BvpApiError::InvalidProblem(
                "BVP_sci external parameter names and values must have equal non-zero lengths"
                    .into(),
            ));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(BvpApiError::InvalidProblem(
                "BVP_sci external parameter values must be finite".into(),
            ));
        }
        let mut seen = HashSet::with_capacity(names.len());
        if names.iter().any(|name| {
            name.trim().is_empty()
                || name == &self.independent_variable
                || self.state_names.contains(name)
                || !seen.insert(name.as_str())
        }) {
            return Err(BvpApiError::InvalidProblem(
                "BVP_sci external parameter names must be unique and distinct from state and independent-variable names".into(),
            ));
        }
        self.external_parameters = Some(BvpSciExternalParameters { names, values });
        Ok(self)
    }

    fn validate(&self) -> Result<(), BvpApiError> {
        let state_count = self.state_names.len();
        if state_count == 0 || self.equations.len() != state_count {
            return Err(BvpApiError::InvalidProblem(
                "equation and state dimensions must be non-zero and equal".into(),
            ));
        }
        if self.independent_variable.trim().is_empty() {
            return Err(BvpApiError::InvalidProblem(
                "independent variable name cannot be empty".into(),
            ));
        }
        if self.state_names.iter().any(|name| name.trim().is_empty()) {
            return Err(BvpApiError::InvalidProblem(
                "state names cannot be empty".into(),
            ));
        }
        let mut names = HashSet::with_capacity(state_count + 1);
        if !names.insert(self.independent_variable.as_str())
            || self
                .state_names
                .iter()
                .any(|name| !names.insert(name.as_str()))
        {
            return Err(BvpApiError::InvalidProblem(
                "state names and independent variable must be unique".into(),
            ));
        }
        validate_mesh(&self.mesh)?;
        if self.boundary_conditions.len() != state_count {
            return Err(BvpApiError::InvalidProblem(
                "this facade requires exactly state_count scalar boundary conditions".into(),
            ));
        }
        let mut constrained = vec![[false; 2]; state_count];
        for condition in &self.boundary_conditions {
            if condition.state >= state_count || !condition.value.is_finite() {
                return Err(BvpApiError::InvalidProblem(
                    "boundary state index and value must be valid and finite".into(),
                ));
            }
            let endpoint = match condition.side {
                BvpBoundarySide::Left => 0,
                BvpBoundarySide::Right => 1,
            };
            if std::mem::replace(&mut constrained[condition.state][endpoint], true) {
                return Err(BvpApiError::InvalidProblem(
                    "a state cannot have duplicate conditions at the same endpoint".into(),
                ));
            }
        }
        Ok(())
    }

    fn legacy_boundary_conditions(&self) -> HashMap<String, Vec<(usize, f64)>> {
        let mut conditions = HashMap::<String, Vec<(usize, f64)>>::new();
        for condition in &self.boundary_conditions {
            let side = match condition.side {
                BvpBoundarySide::Left => 0,
                BvpBoundarySide::Right => 1,
            };
            conditions
                .entry(self.state_names[condition.state].clone())
                .or_default()
                .push((side, condition.value));
        }
        conditions
    }
}

/// Solver-specific native configuration. No shared lossy option bag is used.
pub enum BvpSolverConfig {
    /// Damped Newton route. Its guess has one column per mesh interval.
    Damped {
        options: DampedSolverOptions,
        /// State-major initial guess with shape `state_count x (mesh.len() - 1)`.
        initial_guess: DMatrix<f64>,
    },
    /// Frozen/Naive route. Its guess has one column per mesh point.
    Frozen {
        options: FrozenSolverOptions,
        /// State-major initial guess with shape `state_count x mesh.len()`.
        initial_guess: DMatrix<f64>,
    },
    /// Collocation route, retaining its full native option and AOT policies.
    Sci {
        options: BvpSciOptions,
        sparse_factorization: BvpSciSparseFactorization,
        aot_config: Option<SymbolicIvpGeneratedBackendConfig>,
        /// State-major initial guess with shape `state_count x mesh.len()`.
        initial_guess: DMatrix<f64>,
    },
    /// BVP_sci with free Newton parameters and additional boundary equations.
    SciWithFreeParameters {
        options: BvpSciOptions,
        sparse_factorization: BvpSciSparseFactorization,
        aot_config: Option<SymbolicIvpGeneratedBackendConfig>,
        /// State-major initial guess with shape `state_count x mesh.len()`.
        initial_guess: DMatrix<f64>,
        free_parameters: BvpSciFreeParameters,
    },
}

/// Solver-family settings without problem-owned data such as the initial guess.
///
/// Hybrid applications can parse or construct this value independently from
/// their physical model, then attach the caller-owned guess when building a
/// [`BvpSolverConfig`]. Native option structs are retained intact; this type is
/// not a normalized, lossy option bag.
pub enum BvpSolverSettings {
    /// Damped Newton route with its complete native options.
    Damped(DampedSolverOptions),
    /// Frozen/Naive route with its complete native options.
    Frozen(FrozenSolverOptions),
    /// Collocation route with its native options and generated-backend policy.
    Sci {
        options: BvpSciOptions,
        sparse_factorization: BvpSciSparseFactorization,
        aot_config: Option<SymbolicIvpGeneratedBackendConfig>,
    },
}

/// Common telemetry request exposed by hybrid BVP clients.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BvpApiTelemetryMode {
    /// Disable runtime collection.
    #[default]
    Off,
    /// Collect counters without reading clocks.
    Counters,
    /// Collect timing data without counters, when supported by the route.
    Timings,
    /// Collect both counters and timing data.
    Detailed,
}

/// Typed configuration failure while applying a facade-wide policy.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum BvpSettingsError {
    /// A selected native route cannot represent the requested telemetry mode.
    #[error("{solver} does not support telemetry mode {mode:?}: {reason}")]
    UnsupportedTelemetry {
        solver: &'static str,
        mode: BvpApiTelemetryMode,
        reason: &'static str,
    },
}

impl BvpSolverSettings {
    /// Apply one telemetry policy without conflating logging or decision traces.
    ///
    /// BVP_Damp AOT callback telemetry has no Timings-only mode. Damped/Frozen
    /// accept this request only when their backend is explicitly restricted to
    /// Lambdify; routes that may select AOT are rejected instead of silently
    /// enabling Detailed collection. BVP_sci supports all four modes natively.
    pub fn with_telemetry_mode(
        mut self,
        mode: BvpApiTelemetryMode,
    ) -> Result<Self, BvpSettingsError> {
        match &mut self {
            Self::Damped(options) => {
                if mode == BvpApiTelemetryMode::Timings
                    && options.generated_backend_config.backend_policy_override
                        != Some(BackendSelectionPolicy::LambdifyOnly)
                {
                    return Err(BvpSettingsError::UnsupportedTelemetry {
                        solver: "BVP_Damp Damped",
                        mode,
                        reason: "Timings-only requires an explicit LambdifyOnly backend policy; AOT telemetry has no timings-only mode",
                    });
                }
                let native = match mode {
                    BvpApiTelemetryMode::Off => {
                        crate::numerical::BVP_Damp::telemetry::BvpTelemetryMode::Off
                    }
                    BvpApiTelemetryMode::Counters => {
                        crate::numerical::BVP_Damp::telemetry::BvpTelemetryMode::Counters
                    }
                    BvpApiTelemetryMode::Timings => {
                        crate::numerical::BVP_Damp::telemetry::BvpTelemetryMode::Timings
                    }
                    BvpApiTelemetryMode::Detailed => {
                        crate::numerical::BVP_Damp::telemetry::BvpTelemetryMode::Detailed
                    }
                };
                let callbacks = match mode {
                    BvpApiTelemetryMode::Off => {
                        crate::symbolic::bvp::telemetry::BvpLambdifyTelemetryMode::Off
                    }
                    BvpApiTelemetryMode::Counters => {
                        crate::symbolic::bvp::telemetry::BvpLambdifyTelemetryMode::Counters
                    }
                    BvpApiTelemetryMode::Timings => {
                        crate::symbolic::bvp::telemetry::BvpLambdifyTelemetryMode::Timings
                    }
                    BvpApiTelemetryMode::Detailed => {
                        crate::symbolic::bvp::telemetry::BvpLambdifyTelemetryMode::Detailed
                    }
                };
                let aot = match mode {
                    BvpApiTelemetryMode::Off => {
                        crate::symbolic::bvp::aot_telemetry::BvpAotTelemetryMode::Off
                    }
                    BvpApiTelemetryMode::Counters => {
                        crate::symbolic::bvp::aot_telemetry::BvpAotTelemetryMode::Counters
                    }
                    BvpApiTelemetryMode::Timings => {
                        crate::symbolic::bvp::aot_telemetry::BvpAotTelemetryMode::Off
                    }
                    BvpApiTelemetryMode::Detailed => {
                        crate::symbolic::bvp::aot_telemetry::BvpAotTelemetryMode::Detailed
                    }
                };
                options.generated_backend_config.bvp_telemetry_mode = native;
                options.generated_backend_config.lambdify_telemetry_mode = callbacks;
                options.generated_backend_config.aot_telemetry_mode = aot;
            }
            Self::Frozen(options) => {
                if mode == BvpApiTelemetryMode::Timings
                    && options.generated_backend_config.backend_policy_override
                        != Some(BackendSelectionPolicy::LambdifyOnly)
                {
                    return Err(BvpSettingsError::UnsupportedTelemetry {
                        solver: "BVP_Damp Frozen/Naive",
                        mode,
                        reason: "Timings-only requires an explicit LambdifyOnly backend policy; AOT telemetry has no timings-only mode",
                    });
                }
                options.generated_backend_config.bvp_telemetry_mode = match mode {
                    BvpApiTelemetryMode::Off => {
                        crate::numerical::BVP_Damp::telemetry::BvpTelemetryMode::Off
                    }
                    BvpApiTelemetryMode::Counters => {
                        crate::numerical::BVP_Damp::telemetry::BvpTelemetryMode::Counters
                    }
                    BvpApiTelemetryMode::Timings => {
                        crate::numerical::BVP_Damp::telemetry::BvpTelemetryMode::Timings
                    }
                    BvpApiTelemetryMode::Detailed => {
                        crate::numerical::BVP_Damp::telemetry::BvpTelemetryMode::Detailed
                    }
                };
                options.generated_backend_config.lambdify_telemetry_mode = match mode {
                    BvpApiTelemetryMode::Off => {
                        crate::symbolic::bvp::telemetry::BvpLambdifyTelemetryMode::Off
                    }
                    BvpApiTelemetryMode::Counters => {
                        crate::symbolic::bvp::telemetry::BvpLambdifyTelemetryMode::Counters
                    }
                    BvpApiTelemetryMode::Timings => {
                        crate::symbolic::bvp::telemetry::BvpLambdifyTelemetryMode::Timings
                    }
                    BvpApiTelemetryMode::Detailed => {
                        crate::symbolic::bvp::telemetry::BvpLambdifyTelemetryMode::Detailed
                    }
                };
                options.generated_backend_config.aot_telemetry_mode = match mode {
                    BvpApiTelemetryMode::Off => {
                        crate::symbolic::bvp::aot_telemetry::BvpAotTelemetryMode::Off
                    }
                    BvpApiTelemetryMode::Counters => {
                        crate::symbolic::bvp::aot_telemetry::BvpAotTelemetryMode::Counters
                    }
                    BvpApiTelemetryMode::Timings => {
                        crate::symbolic::bvp::aot_telemetry::BvpAotTelemetryMode::Off
                    }
                    BvpApiTelemetryMode::Detailed => {
                        crate::symbolic::bvp::aot_telemetry::BvpAotTelemetryMode::Detailed
                    }
                };
            }
            Self::Sci { options, .. } => {
                options.telemetry = match mode {
                    BvpApiTelemetryMode::Off => BvpSciTelemetryMode::Off,
                    BvpApiTelemetryMode::Counters => BvpSciTelemetryMode::Counters,
                    BvpApiTelemetryMode::Timings => BvpSciTelemetryMode::Timings,
                    BvpApiTelemetryMode::Detailed => BvpSciTelemetryMode::Detailed,
                };
            }
        }
        Ok(self)
    }

    /// Attach a state-major initial guess and construct the matching route.
    ///
    /// The guess shape is route-specific and is validated by
    /// [`BvpSolver::new`]: Damped expects one column per interval, while Frozen
    /// and Sci expect one column per mesh node.
    pub fn with_initial_guess(self, initial_guess: DMatrix<f64>) -> BvpSolverConfig {
        match self {
            Self::Damped(options) => BvpSolverConfig::Damped {
                options,
                initial_guess,
            },
            Self::Frozen(options) => BvpSolverConfig::Frozen {
                options,
                initial_guess,
            },
            Self::Sci {
                options,
                sparse_factorization,
                aot_config,
            } => BvpSolverConfig::Sci {
                options,
                sparse_factorization,
                aot_config,
                initial_guess,
            },
        }
    }
}

impl BvpSolverConfig {
    /// Applies one consistent Rust text-logging level to the selected route.
    /// Native telemetry and bounded typed decision traces remain independent.
    pub fn with_logging_level(mut self, level: SolverLogLevel) -> Self {
        match &mut self {
            Self::Damped { options, .. } => {
                *options = options.clone().with_logging_level(level);
            }
            Self::Frozen { options, .. } => {
                *options = options.clone().with_logging_level(level);
            }
            Self::Sci { options, .. } | Self::SciWithFreeParameters { options, .. } => {
                *options = options.clone().with_logging_level(level);
            }
        }
        self
    }
}

/// Native solver retained by the facade so callers can reach solver-specific
/// lifecycle and diagnostics without those details being erased.
pub enum BvpNativeSolver {
    /// Native BVP_Damp damped Newton solver.
    Damped(DampedSolver),
    /// Native BVP_Damp Frozen/Naive solver.
    Frozen(FrozenSolver),
    /// Native BVP_sci collocation solver.
    Sci(BvpSciSolver),
}

/// One successful native result, preserving the route's result contract.
#[derive(Clone, Debug)]
pub enum BvpSolution {
    /// Damped result; values are node-major (`mesh.len() x state_count`).
    Damped {
        mesh: DVector<f64>,
        values: DMatrix<f64>,
    },
    /// Frozen result; values are node-major (`mesh.len() x state_count`).
    Frozen {
        mesh: DVector<f64>,
        values: DMatrix<f64>,
    },
    Sci(BvpSciSolution),
}

/// Allocation-free common access to a retained route-native solution.
///
/// The view normalizes indexing only; it does not flatten, copy, or replace
/// the underlying result. Solver-specific diagnostics remain available from
/// the route-native solution and the native solver.
#[derive(Clone, Copy, Debug)]
pub struct BvpSolutionView<'a> {
    solution: &'a BvpSolution,
}

impl<'a> BvpSolutionView<'a> {
    fn new(solution: &'a BvpSolution) -> Self {
        Self { solution }
    }

    /// Borrow the actual mesh retained by the selected solver.
    pub fn mesh(&self) -> &'a [f64] {
        match self.solution {
            BvpSolution::Damped { mesh, .. } | BvpSolution::Frozen { mesh, .. } => mesh.as_slice(),
            BvpSolution::Sci(solution) => &solution.x,
        }
    }

    /// Number of state variables represented in each mesh row.
    pub fn state_count(&self) -> usize {
        match self.solution {
            BvpSolution::Damped { values, .. } | BvpSolution::Frozen { values, .. } => {
                values.ncols()
            }
            BvpSolution::Sci(solution) => solution.dimension(),
        }
    }

    /// Read one value by mesh-node and state index without materializing a
    /// common dense buffer.
    pub fn value(&self, node: usize, state: usize) -> Option<f64> {
        if node >= self.mesh().len() || state >= self.state_count() {
            return None;
        }
        Some(match self.solution {
            BvpSolution::Damped { values, .. } | BvpSolution::Frozen { values, .. } => {
                values[(node, state)]
            }
            BvpSolution::Sci(solution) => solution.y[node * solution.dimension() + state],
        })
    }
}

/// Route-tagged native telemetry. Each variant retains its solver family's
/// original fields, scopes, units, and absent-versus-zero semantics.
#[derive(Clone, Debug, PartialEq)]
pub enum BvpSolverTelemetrySnapshot {
    /// Damped Newton counters, timings, scopes, storage, and logging trace.
    Damped(BvpTelemetrySnapshot),
    /// Frozen/Naive counters, timings, scopes, storage, and logging trace.
    Frozen(BvpTelemetrySnapshot),
    /// BVP_sci callback, Newton, mesh, and linear-backend telemetry.
    Sci(BvpSciTelemetrySnapshot),
}

/// Ordered continuation output retaining the route's complete native report.
#[derive(Debug)]
pub enum BvpParameterLoopReport {
    /// Adaptive Damped loop outcomes, meshes and typed errors.
    Damped(DampedParameterLoopReport),
    /// Frozen/Naive loop outcomes and telemetry snapshots.
    Frozen(FrozenParameterLoopReport),
    /// BVP_sci fixed external-input outcomes, distinct from free Newton values.
    Sci(BvpSciParameterLoopReport),
}

/// Typed facade-owned errors with native errors retained as sources.
#[derive(Debug, Error)]
pub enum BvpApiError {
    #[error("invalid BVP problem: {0}")]
    InvalidProblem(String),
    #[error("unsupported {solver} BVP route: {reason}")]
    UnsupportedRoute {
        solver: &'static str,
        reason: &'static str,
    },
    #[error("Damped BVP route failed")]
    Damped(#[source] BvpBackendIntegrationError),
    #[error("Frozen/Naive BVP route failed")]
    Frozen(#[source] BvpBackendIntegrationError),
    #[error("BVP_sci route failed")]
    Sci(#[from] BvpSciNewError),
    #[error("postprocessing failed for {solver}")]
    Postprocessing {
        solver: &'static str,
        #[source]
        source: PostprocessError,
    },
    #[error("native solver completed without a result")]
    MissingResult,
}

fn require_uniform_legacy_mesh(mesh: &[f64], solver: &'static str) -> Result<(), BvpApiError> {
    let first_step = mesh[1] - mesh[0];
    let domain_width = mesh[mesh.len() - 1] - mesh[0];
    let scale = domain_width.abs().max(first_step.abs()).max(1.0);
    let tolerance = 32.0 * f64::EPSILON * scale;
    if mesh
        .windows(2)
        .any(|pair| ((pair[1] - pair[0]) - first_step).abs() > tolerance)
    {
        return Err(BvpApiError::UnsupportedRoute {
            solver,
            reason: "native Damped and Frozen constructors accept uniform endpoints/count, not an arbitrary mesh",
        });
    }
    Ok(())
}

fn validate_mesh(mesh: &[f64]) -> Result<(), BvpApiError> {
    if mesh.len() < 2
        || mesh.iter().any(|value| !value.is_finite())
        || !mesh.windows(2).all(|pair| {
            let step = pair[1] - pair[0];
            step.is_finite() && step > 0.0
        })
    {
        return Err(BvpApiError::InvalidProblem(
            "mesh must be finite and strictly increasing".into(),
        ));
    }
    Ok(())
}

/// Thin typed selector which forwards complete native options unchanged.
pub struct BvpSolver {
    problem: BvpProblem,
    native: BvpNativeSolver,
    last_solution: Option<BvpSolution>,
}

impl BvpSolver {
    /// Construct a solver using an explicit solver-family configuration.
    pub fn new(problem: BvpProblem, config: BvpSolverConfig) -> Result<Self, BvpApiError> {
        problem.validate()?;
        let native = match config {
            BvpSolverConfig::Damped {
                options,
                initial_guess,
            } => {
                if problem.external_parameters.is_some() {
                    return Err(BvpApiError::UnsupportedRoute {
                        solver: "BVP_Damp Damped",
                        reason: "external_parameters are supported by the symbolic BVP_sci route, not BVP_Damp",
                    });
                }
                require_uniform_legacy_mesh(&problem.mesh, "BVP_Damp Damped")?;
                validate_initial_guess(&problem, &initial_guess, problem.mesh.len() - 1)?;
                let solver = DampedSolver::new_with_options(
                    problem.equations.clone(),
                    initial_guess,
                    problem.state_names.clone(),
                    problem.independent_variable.clone(),
                    problem.legacy_boundary_conditions(),
                    problem.mesh[0],
                    *problem.mesh.last().expect("validated mesh is non-empty"),
                    problem.mesh.len() - 1,
                    options,
                );
                BvpNativeSolver::Damped(solver)
            }
            BvpSolverConfig::Frozen {
                options,
                initial_guess,
            } => {
                if problem.external_parameters.is_some() {
                    return Err(BvpApiError::UnsupportedRoute {
                        solver: "BVP_Damp Frozen/Naive",
                        reason: "external_parameters are supported by the symbolic BVP_sci route, not BVP_Damp",
                    });
                }
                require_uniform_legacy_mesh(&problem.mesh, "BVP_Damp Frozen/Naive")?;
                validate_initial_guess(&problem, &initial_guess, problem.mesh.len() - 1)?;
                let solver = FrozenSolver::new_with_options(
                    problem.equations.clone(),
                    initial_guess,
                    problem.state_names.clone(),
                    problem.independent_variable.clone(),
                    problem.legacy_boundary_conditions(),
                    problem.mesh[0],
                    *problem.mesh.last().expect("validated mesh is non-empty"),
                    problem.mesh.len() - 1,
                    options,
                );
                BvpNativeSolver::Frozen(solver)
            }
            BvpSolverConfig::Sci {
                options,
                sparse_factorization,
                aot_config,
                initial_guess,
            } => BvpNativeSolver::Sci(build_sci_solver(
                &problem,
                options,
                sparse_factorization,
                aot_config,
                initial_guess,
                None,
                problem.external_parameters.as_ref(),
            )?),
            BvpSolverConfig::SciWithFreeParameters {
                options,
                sparse_factorization,
                aot_config,
                initial_guess,
                free_parameters,
            } => {
                free_parameters.validate(&problem)?;
                BvpNativeSolver::Sci(build_sci_solver(
                    &problem,
                    options,
                    sparse_factorization,
                    aot_config,
                    initial_guess,
                    Some(free_parameters),
                    problem.external_parameters.as_ref(),
                )?)
            }
        };
        Ok(Self {
            problem,
            native,
            last_solution: None,
        })
    }

    /// Access the selected native solver for its complete specialized API.
    pub fn native_mut(&mut self) -> &mut BvpNativeSolver {
        self.last_solution = None;
        &mut self.native
    }

    /// Access the selected native solver without mutation.
    pub fn native(&self) -> &BvpNativeSolver {
        &self.native
    }

    /// Declare fixed model inputs for BVP_Damp routes before their first solve.
    ///
    /// These values are evaluator inputs, not Newton unknowns. BVP_sci's
    /// `parameters` have different semantics and are deliberately rejected by
    /// this fixed-input API.
    pub fn bind_parameters(
        &mut self,
        names: Vec<String>,
        values: Vec<f64>,
    ) -> Result<(), BvpApiError> {
        if names.len() != values.len()
            || values.iter().any(|value| !value.is_finite())
            || names.iter().any(|name| name.trim().is_empty())
        {
            return Err(BvpApiError::InvalidProblem(
                "fixed parameter names and finite values must have equal lengths".into(),
            ));
        }
        let mut seen = HashSet::with_capacity(names.len());
        if names.iter().any(|name| {
            name == &self.problem.independent_variable
                || self.problem.state_names.contains(name)
                || !seen.insert(name.as_str())
        }) {
            return Err(BvpApiError::InvalidProblem(
                "fixed parameter names must be unique and distinct from state/independent-variable names".into(),
            ));
        }
        let name_refs = names.iter().map(String::as_str).collect::<Vec<_>>();
        match &mut self.native {
            BvpNativeSolver::Damped(solver) => {
                solver
                    .try_set_parameter_names(Some(&name_refs))
                    .map_err(BvpApiError::Damped)?;
                solver
                    .try_set_parameter_values(Some(values))
                    .map_err(BvpApiError::Damped)?;
            }
            BvpNativeSolver::Frozen(solver) => {
                solver
                    .try_set_parameter_names(Some(&name_refs))
                    .map_err(BvpApiError::Frozen)?;
                solver
                    .try_set_parameter_values(Some(values))
                    .map_err(BvpApiError::Frozen)?;
            }
            BvpNativeSolver::Sci(_) => {
                return Err(BvpApiError::UnsupportedRoute {
                    solver: "BVP_sci",
                    reason: "BVP_sci parameters are free Newton unknowns, not fixed model inputs",
                });
            }
        }
        self.last_solution = None;
        Ok(())
    }

    /// Replace values in an already declared BVP_Damp fixed-input schema.
    pub fn set_parameter_values(&mut self, values: Vec<f64>) -> Result<(), BvpApiError> {
        if values.iter().any(|value| !value.is_finite()) {
            return Err(BvpApiError::InvalidProblem(
                "fixed parameter values must be finite".into(),
            ));
        }
        match &mut self.native {
            BvpNativeSolver::Damped(solver) => solver
                .try_set_parameter_values(Some(values))
                .map_err(BvpApiError::Damped)?,
            BvpNativeSolver::Frozen(solver) => solver
                .try_set_parameter_values(Some(values))
                .map_err(BvpApiError::Frozen)?,
            BvpNativeSolver::Sci(_) => {
                return Err(BvpApiError::UnsupportedRoute {
                    solver: "BVP_sci",
                    reason: "BVP_sci parameters are free Newton unknowns, not fixed model inputs",
                });
            }
        }
        self.last_solution = None;
        Ok(())
    }

    /// Execute one solve and publish the route-native result on success.
    ///
    /// A failed retry leaves the last successfully published solution intact;
    /// inspect the returned typed error and the current telemetry snapshot for
    /// diagnostics from the failed attempt.
    pub fn solve(&mut self) -> Result<&BvpSolution, BvpApiError> {
        let solution = match &mut self.native {
            BvpNativeSolver::Damped(solver) => {
                solver.try_solver().map_err(BvpApiError::Damped)?;
                BvpSolution::Damped {
                    mesh: solver.x_mesh.clone(),
                    values: solver.get_result().ok_or(BvpApiError::MissingResult)?,
                }
            }
            BvpNativeSolver::Frozen(solver) => {
                solver.try_solver().map_err(BvpApiError::Frozen)?;
                if solver.result.is_none() {
                    return Err(BvpApiError::MissingResult);
                }
                BvpSolution::Frozen {
                    mesh: solver.x_mesh.clone(),
                    values: solver.get_result().ok_or(BvpApiError::MissingResult)?,
                }
            }
            BvpNativeSolver::Sci(solver) => BvpSolution::Sci(solver.solve()?),
        };
        self.last_solution = Some(solution);
        Ok(self
            .last_solution
            .as_ref()
            .expect("solution was just stored"))
    }

    /// Return the current BVP_sci free-parameter Newton guesses.
    pub fn parameter_guess(&self) -> Result<Vec<f64>, BvpApiError> {
        match &self.native {
            BvpNativeSolver::Sci(solver) => Ok(solver.state().2.to_vec()),
            BvpNativeSolver::Damped(_) | BvpNativeSolver::Frozen(_) => {
                Err(BvpApiError::UnsupportedRoute {
                    solver: "BVP_Damp",
                    reason: "Damped and Frozen parameters are fixed evaluator inputs, not free Newton unknowns",
                })
            }
        }
    }

    /// Set BVP_sci free-parameter Newton guesses without changing their schema.
    pub fn set_parameter_guess(&mut self, values: Vec<f64>) -> Result<(), BvpApiError> {
        match &mut self.native {
            BvpNativeSolver::Sci(solver) => {
                solver
                    .set_parameter_guess(values)
                    .map_err(BvpApiError::Sci)?;
                self.last_solution = None;
                Ok(())
            }
            BvpNativeSolver::Damped(_) | BvpNativeSolver::Frozen(_) => {
                Err(BvpApiError::UnsupportedRoute {
                    solver: "BVP_Damp",
                    reason: "Damped and Frozen parameters are fixed evaluator inputs; use set_parameter_values",
                })
            }
        }
    }

    /// Update free-parameter guesses and solve on BVP_sci's retained mesh/profile.
    pub fn solve_with_parameter_guess(
        &mut self,
        values: Vec<f64>,
    ) -> Result<&BvpSolution, BvpApiError> {
        self.set_parameter_guess(values)?;
        self.solve()
    }

    /// Re-solve a prepared BVP_Damp model with new fixed inputs.
    ///
    /// When a previous result exists, the current native Newton iterate is
    /// retained as the next guess. BVP_Damp owns adaptive refinement and may
    /// rebuild mesh-sized symbolic work if the solve grows the grid further.
    pub fn solve_with_parameter_values(
        &mut self,
        values: Vec<f64>,
    ) -> Result<&BvpSolution, BvpApiError> {
        let has_previous_result = self.last_solution.is_some();
        self.set_parameter_values(values)?;
        if !has_previous_result {
            return self.solve();
        }

        let solution = match &mut self.native {
            BvpNativeSolver::Damped(solver) => {
                let current = solver.y.to_DVectorType();
                let expected = solver.values.len() * solver.n_steps;
                if current.len() != expected {
                    return Err(BvpApiError::InvalidProblem(
                        "Damped solver's retained iterate no longer matches its refined mesh"
                            .into(),
                    ));
                }
                let iterate = DMatrix::from_column_slice(
                    solver.values.len(),
                    solver.n_steps,
                    current.as_slice(),
                );
                solver
                    .try_set_prepared_iterate(iterate)
                    .map_err(BvpApiError::Damped)?;
                solver.try_solver_prepared().map_err(BvpApiError::Damped)?;
                BvpSolution::Damped {
                    mesh: solver.x_mesh.clone(),
                    values: solver.get_result().ok_or(BvpApiError::MissingResult)?,
                }
            }
            BvpNativeSolver::Frozen(solver) => {
                let previous = solver.result.as_ref().ok_or(BvpApiError::MissingResult)?;
                let expected = solver.values.len() * solver.n_steps;
                if previous.len() != expected {
                    return Err(BvpApiError::InvalidProblem(
                        "Frozen solver's retained iterate no longer matches its fixed mesh".into(),
                    ));
                }
                let iterate = DMatrix::from_column_slice(
                    solver.values.len(),
                    solver.n_steps,
                    previous.as_slice(),
                );
                solver
                    .try_set_prepared_iterate(iterate)
                    .map_err(BvpApiError::Frozen)?;
                solver.try_solver_prepared().map_err(BvpApiError::Frozen)?;
                BvpSolution::Frozen {
                    mesh: solver.x_mesh.clone(),
                    values: solver.get_result().ok_or(BvpApiError::MissingResult)?,
                }
            }
            BvpNativeSolver::Sci(_) => {
                return Err(BvpApiError::UnsupportedRoute {
                    solver: "BVP_sci",
                    reason: "use BVP_sci's free-parameter guess and solve lifecycle; fixed-input rebinding is not supported by this facade route",
                });
            }
        };
        self.last_solution = Some(solution);
        Ok(self
            .last_solution
            .as_ref()
            .expect("solution was just stored"))
    }

    /// Solve an ordered list of fixed parameter vectors through the selected
    /// native continuation loop.
    ///
    /// Damped/Frozen schemas must first be declared with `bind_parameters`.
    /// BVP_sci external inputs must be declared on `BvpProblem` before solver
    /// construction. The native report is retained by route, including its
    /// telemetry, typed point failures, and any unrecoverable stop error.
    pub fn solve_parameter_loop(
        &mut self,
        parameter_vectors: &[Vec<f64>],
    ) -> Result<BvpParameterLoopReport, BvpApiError> {
        self.last_solution = None;
        match &mut self.native {
            BvpNativeSolver::Damped(solver) => solver
                .try_solve_parameter_loop(parameter_vectors)
                .map(BvpParameterLoopReport::Damped)
                .map_err(BvpApiError::Damped),
            BvpNativeSolver::Frozen(solver) => solver
                .try_solve_parameter_loop(parameter_vectors)
                .map(BvpParameterLoopReport::Frozen)
                .map_err(BvpApiError::Frozen),
            BvpNativeSolver::Sci(solver) => solver
                .solve_parameter_loop(parameter_vectors)
                .map(BvpParameterLoopReport::Sci)
                .map_err(BvpApiError::Sci),
        }
    }

    /// Explicitly restart a prepared BVP_sci model on a supplied mesh/profile.
    ///
    /// BVP_sci preserves its symbolic preparation across this operation. This
    /// is not adaptive mesh refinement: BVP_Damp performs enabled refinement
    /// internally as part of its solve lifecycle, so it has no facade restart
    /// operation.
    pub fn restart(
        &mut self,
        mesh: Vec<f64>,
        initial_guess: DMatrix<f64>,
    ) -> Result<(), BvpApiError> {
        if !matches!(&self.native, BvpNativeSolver::Sci(_)) {
            return Err(BvpApiError::UnsupportedRoute {
                solver: match &self.native {
                    BvpNativeSolver::Damped(_) => "BVP_Damp Damped",
                    BvpNativeSolver::Frozen(_) => "BVP_Damp Frozen/Naive",
                    BvpNativeSolver::Sci(_) => unreachable!("Sci route was checked above"),
                },
                reason: "explicit mesh restart is BVP_sci-only; BVP_Damp refinement is solver-managed during solve",
            });
        }
        validate_mesh(&mesh)?;
        validate_initial_guess(&self.problem, &initial_guess, mesh.len())?;

        let dimension = self.problem.state_names.len();
        let state_size = dimension.checked_mul(mesh.len()).ok_or_else(|| {
            BvpApiError::InvalidProblem("restart state size overflows usize".into())
        })?;
        let mut node_major = vec![0.0; state_size];
        for node in 0..mesh.len() {
            for state in 0..dimension {
                node_major[node * dimension + state] = initial_guess[(state, node)];
            }
        }
        self.last_solution = None;
        let BvpNativeSolver::Sci(solver) = &mut self.native else {
            unreachable!("Sci route was checked above")
        };
        solver
            .restart(mesh.clone(), node_major)
            .map_err(BvpApiError::Sci)?;
        self.problem.mesh = mesh;
        Ok(())
    }

    /// Return the last facade result, if a solve has completed successfully.
    pub fn solution(&self) -> Option<&BvpSolution> {
        self.last_solution.as_ref()
    }

    /// Borrow a common mesh/state view of the last successful result.
    ///
    /// This view performs no allocation and preserves the native result for
    /// callers that need solver-specific diagnostics.
    pub fn solution_view(&self) -> Option<BvpSolutionView<'_>> {
        self.last_solution.as_ref().map(BvpSolutionView::new)
    }

    /// Return the route-native telemetry snapshot without merging its fields
    /// into a facade-defined counter or timing schema.
    pub fn telemetry_snapshot(&self) -> BvpSolverTelemetrySnapshot {
        match &self.native {
            BvpNativeSolver::Damped(solver) => {
                BvpSolverTelemetrySnapshot::Damped(solver.get_statistics().telemetry)
            }
            BvpNativeSolver::Frozen(solver) => {
                BvpSolverTelemetrySnapshot::Frozen(solver.get_statistics().telemetry)
            }
            BvpNativeSolver::Sci(solver) => {
                BvpSolverTelemetrySnapshot::Sci(solver.plan().telemetry_snapshot())
            }
        }
    }

    /// Build a common dataset view for plotting/export while retaining the
    /// route-native solution separately.
    pub fn postprocess_dataset(&self) -> Result<PostprocessDataset, PostprocessError> {
        let solution = self
            .last_solution
            .as_ref()
            .ok_or_else(|| PostprocessError::InvalidDataset("BVP has not been solved".into()))?;
        let (mesh, values) = match solution {
            BvpSolution::Damped { mesh, values } | BvpSolution::Frozen { mesh, values } => {
                (mesh.as_slice().to_vec(), values.clone())
            }
            BvpSolution::Sci(solution) => {
                let dimension = solution.dimension();
                let node_count = solution.x.len();
                let values = DMatrix::from_fn(node_count, dimension, |node, state| {
                    solution.y[node * dimension + state]
                });
                (solution.x.clone(), values)
            }
        };
        PostprocessDataset::new(
            self.problem.independent_variable.clone(),
            self.problem.state_names.clone(),
            DVector::from_vec(mesh),
            values,
        )
    }

    /// Execute the shared postprocessing plan on the retained mesh solution.
    ///
    /// This intentionally exports stored mesh values. BVP_sci dense-output
    /// sampling remains available through its route-native solution API.
    pub fn execute_postprocessing(
        &self,
        plan: &PostprocessPlan,
    ) -> Result<PostprocessReport, BvpApiError> {
        if self.last_solution.is_none() {
            return Err(BvpApiError::MissingResult);
        }
        let solver = match &self.native {
            BvpNativeSolver::Damped(_) => "BVP_Damp Damped",
            BvpNativeSolver::Frozen(_) => "BVP_Damp Frozen/Naive",
            BvpNativeSolver::Sci(_) => "BVP_sci",
        };
        let dataset = self
            .postprocess_dataset()
            .map_err(|source| BvpApiError::Postprocessing { solver, source })?;
        plan.execute(&dataset)
            .map_err(|source| BvpApiError::Postprocessing { solver, source })
    }
}

fn build_sci_solver(
    problem: &BvpProblem,
    options: BvpSciOptions,
    sparse_factorization: BvpSciSparseFactorization,
    aot_config: Option<SymbolicIvpGeneratedBackendConfig>,
    initial_guess: DMatrix<f64>,
    free_parameters: Option<BvpSciFreeParameters>,
    external_parameters: Option<&BvpSciExternalParameters>,
) -> Result<BvpSciSolver, BvpApiError> {
    if options.execution == BvpSciExecution::Numerical
        || options.assembly == Some(BvpSciAssembly::Numerical)
    {
        return Err(BvpApiError::UnsupportedRoute {
            solver: "BVP_sci",
            reason: "BVP_api accepts symbolic equations; numerical callback plans must use the native BVP_sci builder",
        });
    }
    let dimension = problem.state_names.len();
    validate_initial_guess(problem, &initial_guess, problem.mesh.len())?;
    let mut initial_state = vec![0.0; dimension * problem.mesh.len()];
    for node in 0..problem.mesh.len() {
        for state in 0..dimension {
            initial_state[node * dimension + state] = initial_guess[(state, node)];
        }
    }
    let conditions = problem.boundary_conditions.clone();
    let sparse_factorization = NativeBvpSciSparseFactorization::from(sparse_factorization);
    let residual_callback = free_parameters
        .as_ref()
        .map(|parameters| Arc::clone(&parameters.residual));
    let mut builder = BvpSciSolver::builder(problem.equations.clone(), problem.state_names.clone())
        .with_independent_variable(problem.independent_variable.clone())
        .with_mesh_and_initial_state(problem.mesh.clone(), initial_state)
        .with_boundary_callback(move |ya, yb, parameters, output| {
            output.fill(0.0);
            for (index, condition) in conditions.iter().enumerate() {
                output[index] = match condition.side {
                    BvpBoundarySide::Left => ya[condition.state] - condition.value,
                    BvpBoundarySide::Right => yb[condition.state] - condition.value,
                };
            }
            if let Some(callback) = &residual_callback {
                callback(ya, yb, parameters, &mut output[conditions.len()..])?;
            }
            Ok(())
        })
        .with_options(options)
        .with_sparse_factorization(sparse_factorization);
    if let Some(parameters) = external_parameters {
        builder = builder
            .with_external_parameter_names(parameters.names.clone())
            .with_external_parameter_values(parameters.values.clone());
    }
    if let Some(parameters) = free_parameters {
        let parameter_count = parameters.names.len();
        builder = builder
            .with_parameter_names(parameters.names)
            .with_parameter_guess(parameters.initial_guess);
        if let Some(extra_jacobian) = parameters.jacobian {
            let conditions = problem.boundary_conditions.clone();
            let state_dimension = dimension;
            let state_row_offset = conditions.len() * state_dimension;
            let parameter_row_offset = conditions.len() * parameter_count;
            builder = builder.with_boundary_jacobian(move |ya, yb, parameters, dya, dyb, dp| {
                dya.fill(0.0);
                dyb.fill(0.0);
                dp.fill(0.0);
                for (row, condition) in conditions.iter().enumerate() {
                    let block = match condition.side {
                        BvpBoundarySide::Left => &mut *dya,
                        BvpBoundarySide::Right => &mut *dyb,
                    };
                    block[row * state_dimension + condition.state] = 1.0;
                }
                extra_jacobian(
                    ya,
                    yb,
                    parameters,
                    &mut dya[state_row_offset..],
                    &mut dyb[state_row_offset..],
                    &mut dp[parameter_row_offset..],
                )
            });
        }
    }
    if let Some(config) = aot_config {
        builder = builder.with_aot_generated_backend(config);
    }
    builder.build().map_err(BvpApiError::Sci)
}

fn validate_initial_guess(
    problem: &BvpProblem,
    initial_guess: &DMatrix<f64>,
    expected_columns: usize,
) -> Result<(), BvpApiError> {
    if initial_guess.nrows() != problem.state_names.len()
        || initial_guess.ncols() != expected_columns
        || initial_guess.iter().any(|value| !value.is_finite())
    {
        return Err(BvpApiError::InvalidProblem(format!(
            "initial guess must be finite with shape {} x {expected_columns}",
            problem.state_names.len()
        )));
    }
    Ok(())
}
