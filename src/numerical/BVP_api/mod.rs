//! Typed orchestration over the production boundary-value solver APIs.
//!
//! This facade owns routing and shared problem validation only. Solver
//! configuration, diagnostics, telemetry, and native result types stay with
//! their respective solver families. Task-document parsing is intentionally a
//! separate subsystem and calls native solvers directly.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::Utils::postprocessing::{
    PostprocessDataset, PostprocessError, PostprocessPlan, PostprocessReport,
};
use crate::numerical::BVP_CorrectionNewton::CorrectionNewtonProblem;
use crate::numerical::BVP_CorrectionNewton::{
    CorrectionNewtonOptions, CorrectionNewtonParameterLoopOutcome,
    CorrectionNewtonParameterLoopReport, CorrectionNewtonSolveReport, CorrectionNewtonSolver,
    CorrectionNewtonSolverError, CorrectionNewtonTelemetryMode, CorrectionNewtonTelemetrySnapshot,
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

    /// Declare fixed symbolic inputs before constructing a BVP_sci or
    /// BVP_CorrectionNewton solver.
    ///
    /// This is separate from [`BvpSciFreeParameters`], which adds unknowns to
    /// the Newton system. On BVP_sci, the declared evaluator schema is prepared
    /// once and its parameter loop supplies successive values. BVP_CorrectionNewton
    /// supports binding/rebinding the same fixed-input schema but not a loop.
    pub fn with_external_parameters(
        mut self,
        names: Vec<String>,
        values: Vec<f64>,
    ) -> Result<Self, BvpApiError> {
        if names.is_empty() || names.len() != values.len() {
            return Err(BvpApiError::InvalidProblem(
                "external parameter names and values must have equal non-zero lengths".into(),
            ));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(BvpApiError::InvalidProblem(
                "external parameter values must be finite".into(),
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
                "external parameter names must be unique and distinct from state and independent-variable names".into(),
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
    /// Fixed-mesh damped correction-Newton core using ExprLegacy/Lambdify.
    CorrectionNewton {
        options: CorrectionNewtonOptions,
        /// State-major initial guess with one value per non-fixed node.
        initial_guess: DMatrix<f64>,
    },
    /// CorrectionNewton route with a full state-major profile aligned to every
    /// node of [`BvpProblem::mesh`]. Fixed endpoint entries must match the BCs.
    CorrectionNewtonNodeAligned {
        options: CorrectionNewtonOptions,
        initial_guess: DMatrix<f64>,
    },
    /// Damped Newton route. Its guess has one column per mesh interval.
    Damped {
        options: DampedSolverOptions,
        /// State-major initial guess with shape `state_count x (mesh.len() - 1)`.
        initial_guess: DMatrix<f64>,
    },
    /// Damped route with a full state-major profile aligned to every node.
    /// Fixed endpoint entries must match the BCs; enabled refinement remains
    /// owned by the native Damped solve lifecycle.
    DampedNodeAligned {
        options: DampedSolverOptions,
        initial_guess: DMatrix<f64>,
    },
    /// Frozen/Naive route. Its reduced guess has one column per free interval
    /// coordinate; use `FrozenNodeAligned` to supply all mesh nodes.
    Frozen {
        options: FrozenSolverOptions,
        /// State-major reduced guess with shape `state_count x interval_count`.
        initial_guess: DMatrix<f64>,
    },
    /// Frozen/Naive route with a complete state-major profile aligned to every
    /// mesh node. Fixed endpoint entries must match the boundary conditions.
    FrozenNodeAligned {
        options: FrozenSolverOptions,
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
    /// Fixed-mesh correction-Newton route with its native options.
    CorrectionNewton(CorrectionNewtonOptions),
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
    /// Applies the shared text-logging policy before attaching problem data.
    /// Logging defaults to `Off`; enabling it uses the crate-wide logger.
    pub fn with_logging_level(mut self, level: SolverLogLevel) -> Self {
        match &mut self {
            Self::CorrectionNewton(options) => options.logging_level = level,
            Self::Damped(options) => *options = options.clone().with_logging_level(level),
            Self::Frozen(options) => *options = options.clone().with_logging_level(level),
            Self::Sci { options, .. } => *options = options.clone().with_logging_level(level),
        }
        self
    }

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
            Self::CorrectionNewton(options) => {
                if mode == BvpApiTelemetryMode::Timings {
                    return Err(BvpSettingsError::UnsupportedTelemetry {
                        solver: "BVP_CorrectionNewton",
                        mode,
                        reason: "CorrectionNewton currently collects counters together with timings; request Detailed instead",
                    });
                }
                options.telemetry_mode = match mode {
                    BvpApiTelemetryMode::Off => CorrectionNewtonTelemetryMode::Off,
                    BvpApiTelemetryMode::Counters => CorrectionNewtonTelemetryMode::Counters,
                    BvpApiTelemetryMode::Timings | BvpApiTelemetryMode::Detailed => {
                        CorrectionNewtonTelemetryMode::Timings
                    }
                };
            }
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
    /// [`BvpSolver::new`]: Damped, Frozen and CorrectionNewton expect one value
    /// per free state/interval coordinate, while Sci expects one column per
    /// mesh node. For Damped, each state row follows increasing mesh order with
    /// its constrained endpoint omitted; the facade converts it to the native
    /// compact node-major Newton ordering.
    pub fn with_initial_guess(self, initial_guess: DMatrix<f64>) -> BvpSolverConfig {
        match self {
            Self::CorrectionNewton(options) => BvpSolverConfig::CorrectionNewton {
                options,
                initial_guess,
            },
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

    /// Attach a full state-major profile with one column per node in the
    /// caller-supplied mesh. CorrectionNewton and Damped validate the
    /// constrained endpoint entries and pack only free values internally;
    /// Frozen validates the constrained endpoint entries and packs only free
    /// coordinates; Sci already consumes the full profile directly.
    pub fn with_node_aligned_initial_guess(self, initial_guess: DMatrix<f64>) -> BvpSolverConfig {
        match self {
            Self::CorrectionNewton(options) => BvpSolverConfig::CorrectionNewtonNodeAligned {
                options,
                initial_guess,
            },
            Self::Damped(options) => BvpSolverConfig::DampedNodeAligned {
                options,
                initial_guess,
            },
            Self::Frozen(options) => BvpSolverConfig::FrozenNodeAligned {
                options,
                initial_guess,
            },
            settings @ Self::Sci { .. } => settings.with_initial_guess(initial_guess),
        }
    }
}

impl BvpSolverConfig {
    /// Applies one consistent Rust text-logging level to the selected route.
    /// Native telemetry and bounded typed decision traces remain independent.
    pub fn with_logging_level(mut self, level: SolverLogLevel) -> Self {
        match &mut self {
            Self::CorrectionNewton { options, .. }
            | Self::CorrectionNewtonNodeAligned { options, .. } => options.logging_level = level,
            Self::Damped { options, .. } | Self::DampedNodeAligned { options, .. } => {
                *options = options.clone().with_logging_level(level);
            }
            Self::Frozen { options, .. } | Self::FrozenNodeAligned { options, .. } => {
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
    /// Standalone fixed-mesh CorrectionNewton Newton solver.
    CorrectionNewton(CorrectionNewtonSolver),
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
    /// CorrectionNewton fixed-mesh Newton result with its complete native report.
    CorrectionNewton {
        mesh: DVector<f64>,
        values: DMatrix<f64>,
        report: CorrectionNewtonSolveReport,
    },
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
            BvpSolution::CorrectionNewton { mesh, .. } => mesh.as_slice(),
            BvpSolution::Damped { mesh, .. } | BvpSolution::Frozen { mesh, .. } => mesh.as_slice(),
            BvpSolution::Sci(solution) => &solution.x,
        }
    }

    /// Number of state variables represented in each mesh row.
    pub fn state_count(&self) -> usize {
        match self.solution {
            BvpSolution::CorrectionNewton { values, .. } => values.ncols(),
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
            BvpSolution::CorrectionNewton { values, .. } => values[(node, state)],
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
    /// CorrectionNewton core counters/timings and their native scopes.
    CorrectionNewton(Option<CorrectionNewtonTelemetrySnapshot>),
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
    /// Fixed-mesh CorrectionNewton outcomes; failed points never become the next guess.
    CorrectionNewton(CorrectionNewtonParameterLoopReport),
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
    #[error("CorrectionNewton BVP route failed")]
    CorrectionNewton(#[from] CorrectionNewtonSolverError),
    #[error("postprocessing failed for {solver}")]
    Postprocessing {
        solver: &'static str,
        #[source]
        source: PostprocessError,
    },
    #[error("native solver completed without a result")]
    MissingResult,
}

fn is_uniform_mesh(mesh: &[f64]) -> bool {
    let first_step = mesh[1] - mesh[0];
    let domain_width = mesh[mesh.len() - 1] - mesh[0];
    let scale = domain_width.abs().max(first_step.abs()).max(1.0);
    let tolerance = 32.0 * f64::EPSILON * scale;
    !mesh
        .windows(2)
        .any(|pair| ((pair[1] - pair[0]) - first_step).abs() > tolerance)
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
    last_correction_newton_telemetry: Option<CorrectionNewtonTelemetrySnapshot>,
    boundary_values_binding: Option<Arc<Vec<AtomicU64>>>,
    damped_explicit_mesh: bool,
}

impl BvpSolver {
    /// Construct a solver using an explicit solver-family configuration.
    pub fn new(problem: BvpProblem, config: BvpSolverConfig) -> Result<Self, BvpApiError> {
        problem.validate()?;
        let mut boundary_values_binding = None;
        let mut damped_explicit_mesh = false;
        let native = match config {
            BvpSolverConfig::CorrectionNewton {
                options,
                initial_guess,
            } => {
                validate_initial_guess(&problem, &initial_guess, problem.mesh.len() - 1)?;
                let correction_newton_problem = CorrectionNewtonProblem {
                    equations: problem.equations.clone(),
                    initial_guess,
                    state_names: problem.state_names.clone(),
                    independent_variable: problem.independent_variable.clone(),
                    boundary_conditions: problem.legacy_boundary_conditions(),
                    interval: (
                        problem.mesh[0],
                        *problem.mesh.last().expect("validated mesh"),
                    ),
                    steps: problem.mesh.len() - 1,
                };
                let mut solver = if is_uniform_mesh(&problem.mesh) {
                    CorrectionNewtonSolver::from_symbolic(correction_newton_problem, options)?
                } else {
                    CorrectionNewtonSolver::from_symbolic_on_mesh(
                        correction_newton_problem,
                        problem.mesh.clone(),
                        options,
                    )?
                };
                if let Some(parameters) = problem.external_parameters.as_ref() {
                    solver
                        .bind_parameters(parameters.names.clone(), parameters.values.clone())
                        .map_err(BvpApiError::CorrectionNewton)?;
                }
                BvpNativeSolver::CorrectionNewton(solver)
            }
            BvpSolverConfig::CorrectionNewtonNodeAligned {
                options,
                initial_guess,
            } => {
                let reduced_guess = reduce_node_aligned_guess(&problem, &initial_guess)?;
                let correction_newton_problem = CorrectionNewtonProblem {
                    equations: problem.equations.clone(),
                    initial_guess: reduced_guess,
                    state_names: problem.state_names.clone(),
                    independent_variable: problem.independent_variable.clone(),
                    boundary_conditions: problem.legacy_boundary_conditions(),
                    interval: (
                        problem.mesh[0],
                        *problem.mesh.last().expect("validated mesh"),
                    ),
                    steps: problem.mesh.len() - 1,
                };
                let mut solver = if is_uniform_mesh(&problem.mesh) {
                    CorrectionNewtonSolver::from_symbolic(correction_newton_problem, options)?
                } else {
                    CorrectionNewtonSolver::from_symbolic_on_mesh(
                        correction_newton_problem,
                        problem.mesh.clone(),
                        options,
                    )?
                };
                if let Some(parameters) = problem.external_parameters.as_ref() {
                    solver
                        .bind_parameters(parameters.names.clone(), parameters.values.clone())
                        .map_err(BvpApiError::CorrectionNewton)?;
                }
                BvpNativeSolver::CorrectionNewton(solver)
            }
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
                validate_initial_guess(&problem, &initial_guess, problem.mesh.len() - 1)?;
                let compact_initial_iterate = pack_damped_initial_guess(&problem, &initial_guess)?;
                let mut solver = DampedSolver::new_with_options(
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
                damped_explicit_mesh = !is_uniform_mesh(&problem.mesh);
                if damped_explicit_mesh {
                    solver.x_mesh = DVector::from_vec(problem.mesh.clone());
                }
                solver
                    .try_set_initial_compact_node_major_iterate(compact_initial_iterate)
                    .map_err(BvpApiError::Damped)?;
                BvpNativeSolver::Damped(solver)
            }
            BvpSolverConfig::DampedNodeAligned {
                options,
                initial_guess,
            } => {
                if problem.external_parameters.is_some() {
                    return Err(BvpApiError::UnsupportedRoute {
                        solver: "BVP_Damp Damped",
                        reason: "external_parameters are supported by the symbolic BVP_sci route, not BVP_Damp",
                    });
                }
                let reduced_guess = reduce_node_aligned_guess(&problem, &initial_guess)?;
                let compact_initial_iterate = pack_damped_initial_guess(&problem, &reduced_guess)?;
                let mut solver = DampedSolver::new_with_options(
                    problem.equations.clone(),
                    reduced_guess,
                    problem.state_names.clone(),
                    problem.independent_variable.clone(),
                    problem.legacy_boundary_conditions(),
                    problem.mesh[0],
                    *problem.mesh.last().expect("validated mesh"),
                    problem.mesh.len() - 1,
                    options,
                );
                solver.x_mesh = DVector::from_vec(problem.mesh.clone());
                damped_explicit_mesh = true;
                solver
                    .try_set_initial_compact_node_major_iterate(compact_initial_iterate)
                    .map_err(BvpApiError::Damped)?;
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
                BvpNativeSolver::Frozen(build_frozen_solver(
                    &problem,
                    options,
                    initial_guess,
                    false,
                )?)
            }
            BvpSolverConfig::FrozenNodeAligned {
                options,
                initial_guess,
            } => {
                if problem.external_parameters.is_some() {
                    return Err(BvpApiError::UnsupportedRoute {
                        solver: "BVP_Damp Frozen/Naive",
                        reason: "external_parameters are supported by the symbolic BVP_sci route, not BVP_Damp",
                    });
                }
                BvpNativeSolver::Frozen(build_frozen_solver(
                    &problem,
                    options,
                    initial_guess,
                    true,
                )?)
            }
            BvpSolverConfig::Sci {
                options,
                sparse_factorization,
                aot_config,
                initial_guess,
            } => {
                let (solver, binding) = build_sci_solver(
                    &problem,
                    options,
                    sparse_factorization,
                    aot_config,
                    initial_guess,
                    None,
                    problem.external_parameters.as_ref(),
                )?;
                boundary_values_binding = Some(binding);
                BvpNativeSolver::Sci(solver)
            }
            BvpSolverConfig::SciWithFreeParameters {
                options,
                sparse_factorization,
                aot_config,
                initial_guess,
                free_parameters,
            } => {
                free_parameters.validate(&problem)?;
                let (solver, binding) = build_sci_solver(
                    &problem,
                    options,
                    sparse_factorization,
                    aot_config,
                    initial_guess,
                    Some(free_parameters),
                    problem.external_parameters.as_ref(),
                )?;
                boundary_values_binding = Some(binding);
                BvpNativeSolver::Sci(solver)
            }
        };
        Ok(Self {
            problem,
            native,
            last_solution: None,
            last_correction_newton_telemetry: None,
            boundary_values_binding,
            damped_explicit_mesh,
        })
    }

    /// Access the selected native solver for its complete specialized API.
    pub fn native_mut(&mut self) -> &mut BvpNativeSolver {
        self.last_solution = None;
        self.last_correction_newton_telemetry = None;
        &mut self.native
    }

    /// Access the selected native solver without mutation.
    pub fn native(&self) -> &BvpNativeSolver {
        &self.native
    }

    /// Declare fixed model inputs for BVP_Damp or BVP_CorrectionNewton before solving.
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
            BvpNativeSolver::CorrectionNewton(solver) => solver
                .bind_parameters(names, values)
                .map_err(BvpApiError::CorrectionNewton)?,
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
            BvpNativeSolver::CorrectionNewton(solver) => solver
                .set_parameter_values(values)
                .map_err(BvpApiError::CorrectionNewton)?,
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

    /// Rebind numeric values for the existing boundary-condition slots.
    ///
    /// Values follow `BvpProblem::boundary_conditions` order. State/endpoint
    /// topology is immutable; symbolic callbacks and AOT code are retained.
    pub fn set_boundary_values(&mut self, values: Vec<f64>) -> Result<(), BvpApiError> {
        if values.len() != self.problem.boundary_conditions.len()
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(BvpApiError::InvalidProblem(format!(
                "boundary values must contain {} finite entries in the original condition order",
                self.problem.boundary_conditions.len()
            )));
        }
        match &mut self.native {
            BvpNativeSolver::CorrectionNewton(solver) => {
                let state_count = self.problem.state_names.len();
                let mut ordered = self
                    .problem
                    .boundary_conditions
                    .iter()
                    .zip(&values)
                    .map(|(condition, value)| {
                        let node = match condition.side {
                            BvpBoundarySide::Left => 0,
                            BvpBoundarySide::Right => self.problem.mesh.len() - 1,
                        };
                        (node * state_count + condition.state, *value)
                    })
                    .collect::<Vec<_>>();
                ordered.sort_by_key(|(position, _)| *position);
                solver
                    .set_boundary_values(ordered.into_iter().map(|(_, value)| value).collect())
                    .map_err(BvpApiError::CorrectionNewton)?;
            }
            BvpNativeSolver::Sci(_) => {
                let Some(binding) = self.boundary_values_binding.as_ref() else {
                    return Err(BvpApiError::InvalidProblem(
                        "BVP_sci boundary value binding was not retained".into(),
                    ));
                };
                for (slot, value) in binding.iter().zip(&values) {
                    slot.store(value.to_bits(), Ordering::Relaxed);
                }
            }
            BvpNativeSolver::Damped(solver) => {
                let mut schema_ordered = self
                    .problem
                    .boundary_conditions
                    .iter()
                    .zip(&values)
                    .map(|(condition, value)| {
                        let endpoint = match condition.side {
                            BvpBoundarySide::Left => 0,
                            BvpBoundarySide::Right => 1,
                        };
                        (condition.state, endpoint, *value)
                    })
                    .collect::<Vec<_>>();
                schema_ordered.sort_by_key(|(state, endpoint, _)| (*state, *endpoint));
                solver
                    .try_set_boundary_values(
                        schema_ordered
                            .into_iter()
                            .map(|(_, _, value)| value)
                            .collect(),
                    )
                    .map_err(BvpApiError::Damped)?;
            }
            BvpNativeSolver::Frozen(solver) => {
                let mut schema_ordered = self
                    .problem
                    .boundary_conditions
                    .iter()
                    .zip(&values)
                    .map(|(condition, value)| {
                        let endpoint = match condition.side {
                            BvpBoundarySide::Left => 0,
                            BvpBoundarySide::Right => 1,
                        };
                        (condition.state, endpoint, *value)
                    })
                    .collect::<Vec<_>>();
                schema_ordered.sort_by_key(|(state, endpoint, _)| (*state, *endpoint));
                solver
                    .try_set_boundary_values(
                        schema_ordered
                            .into_iter()
                            .map(|(_, _, value)| value)
                            .collect(),
                    )
                    .map_err(BvpApiError::Frozen)?;
            }
        }
        for (condition, value) in self.problem.boundary_conditions.iter_mut().zip(values) {
            condition.value = value;
        }
        self.last_solution = None;
        Ok(())
    }

    /// Solve again with new values for the existing boundary-condition slots.
    pub fn solve_with_boundary_values(
        &mut self,
        values: Vec<f64>,
    ) -> Result<&BvpSolution, BvpApiError> {
        self.set_boundary_values(values)?;
        self.solve()
    }

    /// Execute one solve and publish the route-native result on success.
    ///
    /// A failed retry leaves the last successfully published solution intact;
    /// inspect the returned typed error and the current telemetry snapshot for
    /// diagnostics from the failed attempt.
    pub fn solve(&mut self) -> Result<&BvpSolution, BvpApiError> {
        let solution = match &mut self.native {
            BvpNativeSolver::CorrectionNewton(solver) => match solver.solve() {
                Ok(report) => {
                    self.last_correction_newton_telemetry = Some(report.telemetry.clone());
                    let values = solver.full_grid_values(&report.state)?;
                    BvpSolution::CorrectionNewton {
                        mesh: DVector::from_vec(self.problem.mesh.clone()),
                        values,
                        report,
                    }
                }
                Err(error) => {
                    if let Some(report) = error.partial_report() {
                        self.last_correction_newton_telemetry = Some(report.telemetry.clone());
                    }
                    return Err(BvpApiError::CorrectionNewton(error));
                }
            },
            BvpNativeSolver::Damped(solver) => {
                let solve_result = if self.damped_explicit_mesh {
                    let active_mesh = solver.x_mesh.iter().copied().collect();
                    solver.try_solver_on_mesh(active_mesh)
                } else {
                    solver.try_solver()
                };
                if solve_result.map_err(BvpApiError::Damped)?.is_none() {
                    return Err(BvpApiError::MissingResult);
                }
                BvpSolution::Damped {
                    mesh: solver.x_mesh.clone(),
                    values: solver.get_result().ok_or(BvpApiError::MissingResult)?,
                }
            }
            BvpNativeSolver::Frozen(solver) => {
                if solver.try_solver().map_err(BvpApiError::Frozen)?.is_none() {
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
            BvpNativeSolver::CorrectionNewton(_) => Err(BvpApiError::UnsupportedRoute {
                solver: "BVP_CorrectionNewton",
                reason: "the fixed-mesh CorrectionNewton route has no free Newton parameters",
            }),
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
            BvpNativeSolver::CorrectionNewton(_) => Err(BvpApiError::UnsupportedRoute {
                solver: "BVP_CorrectionNewton",
                reason: "the fixed-mesh CorrectionNewton route has no free Newton parameters",
            }),
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

    /// Re-solve a prepared BVP_Damp or BVP_CorrectionNewton model with new fixed inputs.
    ///
    /// BVP_Damp retains its current Newton iterate and may rebuild mesh-sized
    /// work after adaptive refinement. BVP_CorrectionNewton reuses its fixed-mesh
    /// symbolic callbacks and current packed state.
    pub fn solve_with_parameter_values(
        &mut self,
        values: Vec<f64>,
    ) -> Result<&BvpSolution, BvpApiError> {
        if let BvpNativeSolver::CorrectionNewton(solver) = &mut self.native {
            match solver.continue_with_parameter_values(values) {
                Ok(report) => {
                    let values = solver.full_grid_values(&report.state)?;
                    self.last_correction_newton_telemetry = Some(report.telemetry.clone());
                    self.last_solution = Some(BvpSolution::CorrectionNewton {
                        mesh: DVector::from_vec(self.problem.mesh.clone()),
                        values,
                        report,
                    });
                    return Ok(self
                        .last_solution
                        .as_ref()
                        .expect("converged profile stored"));
                }
                Err(error) => {
                    self.last_correction_newton_telemetry = error
                        .partial_report()
                        .map(|report| report.telemetry.clone());
                    return Err(BvpApiError::CorrectionNewton(error));
                }
            }
        }
        let has_previous_result = self.last_solution.is_some();
        self.set_parameter_values(values)?;
        if !has_previous_result {
            return self.solve();
        }
        let solution = match &mut self.native {
            BvpNativeSolver::CorrectionNewton(_) => {
                unreachable!("CorrectionNewton route returned above")
            }
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
                if solver
                    .try_solver_prepared()
                    .map_err(BvpApiError::Damped)?
                    .is_none()
                {
                    return Err(BvpApiError::MissingResult);
                }
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
                if solver
                    .try_solver_prepared()
                    .map_err(BvpApiError::Frozen)?
                    .is_none()
                {
                    return Err(BvpApiError::MissingResult);
                }
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
        if let BvpNativeSolver::CorrectionNewton(solver) = &mut self.native {
            let report = solver.solve_parameter_loop(parameter_vectors)?;
            // Native recovery owns the checkpoint. The facade publishes only
            // converged profiles; telemetry describes the last attempted point.
            if let Some(point) = report.points.last() {
                self.last_correction_newton_telemetry = match &point.outcome {
                    CorrectionNewtonParameterLoopOutcome::Converged { report, .. } => {
                        Some(report.telemetry.clone())
                    }
                    CorrectionNewtonParameterLoopOutcome::Failed { error } => error
                        .partial_report()
                        .map(|report| report.telemetry.clone()),
                };
            }
            if let Some((mesh, values, solve_report)) =
                report
                    .points
                    .iter()
                    .rev()
                    .find_map(|point| match &point.outcome {
                        CorrectionNewtonParameterLoopOutcome::Converged {
                            mesh,
                            values,
                            report,
                        } => Some((mesh, values, report)),
                        CorrectionNewtonParameterLoopOutcome::Failed { .. } => None,
                    })
            {
                self.last_solution = Some(BvpSolution::CorrectionNewton {
                    mesh: DVector::from_vec(mesh.clone()),
                    values: values.clone(),
                    report: solve_report.clone(),
                });
            }
            return Ok(BvpParameterLoopReport::CorrectionNewton(report));
        }
        self.last_solution = None;
        match &mut self.native {
            BvpNativeSolver::CorrectionNewton(_) => {
                unreachable!("CorrectionNewton loop returned above")
            }
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
                    BvpNativeSolver::CorrectionNewton(_) => "BVP_CorrectionNewton",
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
            BvpNativeSolver::CorrectionNewton(_) => BvpSolverTelemetrySnapshot::CorrectionNewton(
                self.last_correction_newton_telemetry.clone(),
            ),
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

    /// Return the opt-in Damped Newton decision trace, if this facade owns a
    /// Damped solver. The trace is empty when detailed BVP logging was off.
    pub fn damped_decision_trace_snapshot(
        &self,
    ) -> Option<crate::numerical::BVP_Damp::telemetry::BvpDampedDecisionTrace> {
        match &self.native {
            BvpNativeSolver::Damped(solver) => Some(solver.damped_decision_trace_snapshot()),
            _ => None,
        }
    }

    /// Return the opt-in Frozen Newton iteration trace, if this facade owns a
    /// Frozen solver. The trace is empty unless detailed BVP logging is on.
    pub fn frozen_iteration_trace_snapshot(
        &self,
    ) -> Option<crate::numerical::BVP_Damp::telemetry::BvpFrozenIterationTrace> {
        match &self.native {
            BvpNativeSolver::Frozen(solver) => Some(solver.frozen_iteration_trace_snapshot()),
            _ => None,
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
            BvpSolution::CorrectionNewton { mesh, values, .. } => {
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
            BvpNativeSolver::CorrectionNewton(_) => "BVP_CorrectionNewton",
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
) -> Result<(BvpSciSolver, Arc<Vec<AtomicU64>>), BvpApiError> {
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
    let boundary_values = Arc::new(
        conditions
            .iter()
            .map(|condition| AtomicU64::new(condition.value.to_bits()))
            .collect::<Vec<_>>(),
    );
    let callback_boundary_values = Arc::clone(&boundary_values);
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
                let value = f64::from_bits(callback_boundary_values[index].load(Ordering::Relaxed));
                output[index] = match condition.side {
                    BvpBoundarySide::Left => ya[condition.state] - value,
                    BvpBoundarySide::Right => yb[condition.state] - value,
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
    builder
        .build()
        .map(|solver| (solver, boundary_values))
        .map_err(BvpApiError::Sci)
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

fn build_frozen_solver(
    problem: &BvpProblem,
    options: FrozenSolverOptions,
    initial_guess: DMatrix<f64>,
    node_aligned: bool,
) -> Result<FrozenSolver, BvpApiError> {
    let (native_guess, full_guess) = if node_aligned {
        (
            DMatrix::zeros(problem.state_names.len(), problem.mesh.len() - 1),
            Some(initial_guess),
        )
    } else {
        validate_initial_guess(problem, &initial_guess, problem.mesh.len() - 1)?;
        (initial_guess, None)
    };
    let mut solver = FrozenSolver::new_with_options(
        problem.equations.clone(),
        native_guess,
        problem.state_names.clone(),
        problem.independent_variable.clone(),
        problem.legacy_boundary_conditions(),
        problem.mesh[0],
        *problem.mesh.last().expect("validated mesh is non-empty"),
        problem.mesh.len() - 1,
        options,
    );
    if !is_uniform_mesh(&problem.mesh) {
        solver
            .try_set_explicit_mesh(problem.mesh.clone())
            .map_err(BvpApiError::Frozen)?;
    }
    if let Some(full_guess) = full_guess {
        solver
            .try_set_node_aligned_initial_guess(full_guess)
            .map_err(BvpApiError::Frozen)?;
    }
    Ok(solver)
}

fn reduce_node_aligned_guess(
    problem: &BvpProblem,
    initial_guess: &DMatrix<f64>,
) -> Result<DMatrix<f64>, BvpApiError> {
    validate_initial_guess(problem, initial_guess, problem.mesh.len())?;
    let state_count = problem.state_names.len();
    let interval_count = problem.mesh.len() - 1;
    let mut constrained_side = vec![None; state_count];
    for condition in &problem.boundary_conditions {
        if constrained_side[condition.state]
            .replace(condition.side)
            .is_some()
        {
            return Err(BvpApiError::InvalidProblem(
                "node-aligned Newton guesses require exactly one endpoint condition per state"
                    .into(),
            ));
        }
        let node = match condition.side {
            BvpBoundarySide::Left => 0,
            BvpBoundarySide::Right => interval_count,
        };
        let value = initial_guess[(condition.state, node)];
        let scale = value.abs().max(condition.value.abs()).max(1.0);
        if (value - condition.value).abs() > 64.0 * f64::EPSILON * scale {
            return Err(BvpApiError::InvalidProblem(format!(
                "node-aligned initial guess violates the {:?} boundary value for state {}: supplied {}, expected {}",
                condition.side, problem.state_names[condition.state], value, condition.value
            )));
        }
    }
    if constrained_side.iter().any(Option::is_none) {
        return Err(BvpApiError::InvalidProblem(
            "node-aligned Newton guesses require exactly one endpoint condition per state".into(),
        ));
    }

    let mut reduced = DMatrix::zeros(state_count, interval_count);
    for state in 0..state_count {
        let constrained_node = match constrained_side[state].expect("checked endpoint condition") {
            BvpBoundarySide::Left => 0,
            BvpBoundarySide::Right => interval_count,
        };
        let mut column = 0;
        for node in 0..=interval_count {
            if node == constrained_node {
                continue;
            }
            reduced[(state, column)] = initial_guess[(state, node)];
            column += 1;
        }
    }
    Ok(reduced)
}

/// Convert the public state-major Damped profile to its native compact
/// node-major iterate, omitting each state's fixed endpoint coordinate.
fn pack_damped_initial_guess(
    problem: &BvpProblem,
    state_major: &DMatrix<f64>,
) -> Result<Vec<f64>, BvpApiError> {
    let state_count = problem.state_names.len();
    let interval_count = problem.mesh.len() - 1;
    let mut constrained_side = vec![None; state_count];
    for condition in &problem.boundary_conditions {
        if constrained_side[condition.state]
            .replace(condition.side)
            .is_some()
        {
            return Err(BvpApiError::InvalidProblem(
                "BVP_Damp state-major initial guesses require exactly one endpoint condition per state".into(),
            ));
        }
    }
    if constrained_side.iter().any(Option::is_none) {
        return Err(BvpApiError::InvalidProblem(
            "BVP_Damp state-major initial guesses require exactly one endpoint condition per state"
                .into(),
        ));
    }

    let mut compact_node_major = Vec::with_capacity(state_count * interval_count);
    for node in 0..=interval_count {
        for (state, side) in constrained_side.iter().enumerate() {
            let column = match side.expect("validated endpoint condition") {
                BvpBoundarySide::Left if node == 0 => None,
                BvpBoundarySide::Left => Some(node - 1),
                BvpBoundarySide::Right if node == interval_count => None,
                BvpBoundarySide::Right => Some(node),
            };
            if let Some(column) = column {
                compact_node_major.push(state_major[(state, column)]);
            }
        }
    }

    debug_assert_eq!(compact_node_major.len(), state_count * interval_count);
    Ok(compact_node_major)
}
