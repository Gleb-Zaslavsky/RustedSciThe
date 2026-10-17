//! Fluent public construction API for the Lambdify BVP solver.
//!
//! The low-level `BvpSciLambdifyPlan::prepare` plus `BvpSciSolver::new`
//! sequence remains available for advanced integrations. This builder keeps
//! the common user path flat: symbolic equations, boundary callbacks, mesh,
//! frontend, matrix layout and telemetry are configured in one chain.

use super::{
    BvpSciBoundaryCallbacks, BvpSciExecution, BvpSciExecutionPolicy, BvpSciLambdifyPlan,
    BvpSciMatrixLayout, BvpSciNewError, BvpSciNumericalPlan, BvpSciOptions, BvpSciOutputPolicy,
    BvpSciSingularTerm, BvpSciSolver, BvpSciSparseFactorization, BvpSciTelemetry,
    BvpSciTelemetryMode,
};
use crate::numerical::logging::{SolverLogGuard, SolverLogLevel};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp_generated::SymbolicIvpGeneratedBackendConfig;

type BoundaryCallback =
    Box<dyn Fn(&[f64], &[f64], &[f64], &mut [f64]) -> Result<(), String> + Send + Sync>;
type BoundaryJacobianCallback = Box<
    dyn Fn(&[f64], &[f64], &[f64], &mut [f64], &mut [f64], &mut [f64]) -> Result<(), String>
        + Send
        + Sync,
>;

/// Fluent builder for a symbolic BVP_sci Lambdify solve.
pub struct BvpSciSolverBuilder {
    equations: Vec<Expr>,
    state_names: Vec<String>,
    parameter_names: Vec<String>,
    independent_name: String,
    boundary_callback: Option<BoundaryCallback>,
    boundary_jacobian: Option<BoundaryJacobianCallback>,
    mesh: Option<Vec<f64>>,
    initial_state: Option<Vec<f64>>,
    parameters: Vec<f64>,
    options: BvpSciOptions,
    sparse_factorization: BvpSciSparseFactorization,
    aot_config: Option<SymbolicIvpGeneratedBackendConfig>,
    numerical_plan: Option<BvpSciNumericalPlan>,
}

impl BvpSciSolverBuilder {
    /// Start a symbolic builder from first-order equations and state names.
    ///
    /// The default frontend is `ExprLegacy`, the default storage is `Dense`,
    /// telemetry is disabled, and runtime parameters default to zero.
    pub fn new(equations: Vec<Expr>, state_names: Vec<String>) -> Self {
        let mut options = BvpSciOptions::default();
        options.execution = BvpSciExecution::Lambdify;
        options.assembly = Some(super::BvpSciAssembly::ExprLegacy);
        Self {
            equations,
            state_names,
            parameter_names: Vec::new(),
            independent_name: "x".into(),
            boundary_callback: None,
            boundary_jacobian: None,
            mesh: None,
            initial_state: None,
            parameters: Vec::new(),
            options,
            sparse_factorization: BvpSciSparseFactorization::StandardFaer,
            aot_config: None,
            numerical_plan: None,
        }
    }

    /// Select `ExprLegacy` or `AtomViewNative` for Lambdify preparation.
    pub fn with_frontend(mut self, assembly: super::BvpSciAssembly) -> Self {
        self.options.assembly = Some(assembly);
        self
    }

    /// Select `ExprLegacy` as the symbolic frontend.
    pub fn with_expr_legacy(self) -> Self {
        self.with_frontend(super::BvpSciAssembly::ExprLegacy)
    }

    /// Select `AtomViewNative` as the symbolic frontend.
    pub fn with_atom_native(self) -> Self {
        self.with_frontend(super::BvpSciAssembly::AtomViewNative)
    }

    /// Configure Rust-side solver logging; logging is disabled unless enabled.
    pub fn with_logging_level(mut self, level: SolverLogLevel) -> Self {
        self.options.logging_level = level;
        self
    }

    /// Set the independent variable name used by symbolic expressions.
    pub fn with_independent_variable(mut self, name: impl Into<String>) -> Self {
        self.independent_name = name.into();
        self
    }

    /// Set symbolic parameter names. Values can be supplied with
    /// [`Self::with_parameters`]; omitted values default to zero.
    pub fn with_parameter_names(mut self, names: Vec<String>) -> Self {
        self.parameter_names = names;
        self
    }

    /// Set the initial parameter values for the first solve.
    pub fn with_parameters(mut self, parameters: Vec<f64>) -> Self {
        self.parameters = parameters;
        self
    }

    /// Supply boundary residuals. The output dimension is inferred as
    /// `state_dimension + parameter_dimension`, matching the collocation API.
    pub fn with_boundary_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(&[f64], &[f64], &[f64], &mut [f64]) -> Result<(), String> + Send + Sync + 'static,
    {
        self.boundary_callback = Some(Box::new(callback));
        self
    }

    /// Supply analytical endpoint Jacobians. If omitted, the numerical core
    /// keeps its finite-difference boundary-Jacobian fallback.
    pub fn with_boundary_jacobian<J>(mut self, jacobian: J) -> Self
    where
        J: Fn(&[f64], &[f64], &[f64], &mut [f64], &mut [f64], &mut [f64]) -> Result<(), String>
            + Send
            + Sync
            + 'static,
    {
        self.boundary_jacobian = Some(Box::new(jacobian));
        self
    }

    /// Set the strictly increasing mesh and node-major initial state.
    pub fn with_mesh_and_initial_state(mut self, mesh: Vec<f64>, initial_state: Vec<f64>) -> Self {
        self.mesh = Some(mesh);
        self.initial_state = Some(initial_state);
        self
    }

    /// Set only the mesh. An initial state is still required before `build`.
    pub fn with_mesh(mut self, mesh: Vec<f64>) -> Self {
        self.mesh = Some(mesh);
        self
    }

    /// Set only the node-major initial state.
    pub fn with_initial_state(mut self, initial_state: Vec<f64>) -> Self {
        self.initial_state = Some(initial_state);
        self
    }

    /// Select Dense, Sparse or Banded native linear storage.
    pub fn with_matrix_layout(mut self, layout: BvpSciMatrixLayout) -> Self {
        self.options.matrix_layout = layout;
        self
    }

    /// Select an explicit sparse factorization policy.
    /// StandardFaer remains the default.
    pub fn with_sparse_factorization(mut self, strategy: BvpSciSparseFactorization) -> Self {
        self.sparse_factorization = strategy;
        self
    }

    /// Select the runtime-loaded SuperLU backend for Sparse layout.
    ///
    /// Build the bridge for the current platform and set `SUPERLU_LIBRARY` to
    /// its path. If loading fails, the solver warns and uses StandardFaer.
    pub fn with_superlu(self) -> Self {
        self.with_matrix_layout(BvpSciMatrixLayout::Sparse)
            .with_sparse_factorization(BvpSciSparseFactorization::SuperLu)
    }

    /// Compatibility alias for [`Self::with_superlu`].
    #[deprecated(note = "use with_superlu")]
    pub fn with_superlu_experimental(self) -> Self {
        self.with_superlu()
    }

    /// Set the collocation defect tolerance.
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.options.tolerance = tolerance;
        self
    }

    /// Set an independent boundary residual tolerance.
    pub fn with_boundary_tolerance(mut self, tolerance: f64) -> Self {
        self.options.boundary_tolerance = Some(tolerance);
        self
    }

    /// Set mesh, Newton and refinement limits.
    pub fn with_limits(
        mut self,
        max_nodes: usize,
        max_newton_iterations: usize,
        max_mesh_refinements: usize,
    ) -> Self {
        self.options.max_nodes = max_nodes;
        self.options.max_newton_iterations = max_newton_iterations;
        self.options.max_mesh_refinements = max_mesh_refinements;
        self
    }

    /// Set the modified-Newton Jacobian refresh budget independently from the
    /// Newton and adaptive-mesh iteration limits.
    pub fn with_jacobian_refresh_limit(mut self, max_refreshes: usize) -> Self {
        self.options.max_jacobian_refreshes = max_refreshes;
        self
    }

    /// Set the maximum number of line-search trial evaluations per Newton
    /// correction.
    pub fn with_backtracking_limit(mut self, max_trials: usize) -> Self {
        self.options.max_backtracking_steps = max_trials;
        self
    }

    /// Set the callback execution policy for warm residual/Jacobian entries.
    pub fn with_execution_policy(mut self, policy: BvpSciExecutionPolicy) -> Self {
        self.options.execution_policy = policy;
        self
    }

    /// Select the generated AOT lifecycle for this symbolic builder.
    ///
    /// The symbolic assembly (`ExprLegacy` or `AtomViewNative`) and native
    /// matrix layout remain independent choices. AOT preparation is performed
    /// exactly once in `build`; a missing output directory or compiler/link
    /// failure is returned as the original typed BVP error rather than falling
    /// back silently to Lambdify.
    pub fn with_aot_generated_backend(mut self, config: SymbolicIvpGeneratedBackendConfig) -> Self {
        self.aot_config = Some(config);
        self.options.execution = BvpSciExecution::Aot;
        self
    }

    /// Use the Dense numerical-callback frontend instead of symbolic RHS
    /// expressions. Its telemetry handle is shared with boundary and solver
    /// stages; the configured execution policy is applied when building.
    pub fn with_numerical_plan(mut self, plan: BvpSciNumericalPlan) -> Self {
        self.numerical_plan = Some(plan);
        self.options.execution = BvpSciExecution::Numerical;
        self.options.assembly = Some(super::BvpSciAssembly::Numerical);
        self
    }

    /// Enable disabled, counter-only or detailed telemetry.
    pub fn with_telemetry(mut self, mode: BvpSciTelemetryMode) -> Self {
        self.options.telemetry = mode;
        self
    }

    /// Select mesh/value or dense-output retention.
    pub fn with_output_policy(mut self, policy: BvpSciOutputPolicy) -> Self {
        self.options.output_policy = policy;
        self
    }

    /// Configure the optional SciPy singular term.
    pub fn with_singular_term(mut self, singular: BvpSciSingularTerm) -> Self {
        self.options.singular_term = Some(singular);
        self
    }

    /// Prepare callbacks and construct the numerical solver.
    pub fn build(self) -> Result<BvpSciSolver, BvpSciNewError> {
        let _logging_scope = SolverLogGuard::enter(self.options.logging_level);
        let Self {
            equations,
            state_names,
            parameter_names,
            independent_name,
            boundary_callback,
            boundary_jacobian,
            mesh,
            initial_state,
            parameters,
            mut options,
            sparse_factorization,
            aot_config,
            numerical_plan,
        } = self;
        let mesh = mesh.ok_or_else(|| {
            BvpSciNewError::InvalidConfiguration("BVP builder requires a mesh".into())
        })?;
        let initial_state = initial_state.ok_or_else(|| {
            BvpSciNewError::InvalidConfiguration("BVP builder requires an initial state".into())
        })?;
        let boundary_callback = boundary_callback.ok_or_else(|| {
            BvpSciNewError::InvalidConfiguration("BVP builder requires boundary callbacks".into())
        })?;
        let telemetry = match options.telemetry {
            BvpSciTelemetryMode::Off => BvpSciTelemetry::disabled(),
            BvpSciTelemetryMode::Counters => BvpSciTelemetry::counters(),
            BvpSciTelemetryMode::Timings => BvpSciTelemetry::timings(),
            BvpSciTelemetryMode::Detailed => BvpSciTelemetry::detailed(),
        };
        let assembly = options
            .assembly
            .unwrap_or(super::BvpSciAssembly::ExprLegacy);
        options.assembly = Some(assembly);
        let plan = match (options.execution, assembly, numerical_plan, aot_config) {
            (BvpSciExecution::Numerical, super::BvpSciAssembly::Numerical, Some(plan), None) => {
                if plan.dimension() != state_names.len()
                    || plan.parameter_dimension() != parameter_names.len()
                {
                    return Err(BvpSciNewError::InvalidConfiguration(
                        "numerical plan dimensions must match declared state and parameter names"
                            .into(),
                    ));
                }
                if !matches!(options.matrix_layout, super::BvpSciMatrixLayout::Dense) {
                    return Err(BvpSciNewError::UnsupportedRoute(
                        "the numerical callback frontend currently supports Dense layout only"
                            .into(),
                    ));
                }
                BvpSciLambdifyPlan::prepare_numerical(
                    plan.with_telemetry(telemetry.clone())
                        .with_execution_policy(options.execution_policy),
                )
            }
            (BvpSciExecution::Numerical, super::BvpSciAssembly::Numerical, None, _) => {
                return Err(BvpSciNewError::InvalidConfiguration(
                    "numerical execution requires a numerical callback plan".into(),
                ));
            }
            (BvpSciExecution::Numerical, _, None, _) => {
                return Err(BvpSciNewError::InvalidConfiguration(
                    "numerical execution requires Numerical assembly and a callback plan".into(),
                ));
            }
            (_, _, Some(_), _) => {
                return Err(BvpSciNewError::InvalidConfiguration(
                    "numerical callback plan cannot be combined with symbolic or AOT execution"
                        .into(),
                ));
            }
            (_, super::BvpSciAssembly::Numerical, None, _) => {
                return Err(BvpSciNewError::InvalidConfiguration(
                    "Numerical assembly requires a numerical callback plan".into(),
                ));
            }
            (BvpSciExecution::Aot, _, None, Some(config)) => {
                BvpSciLambdifyPlan::prepare_aot_with_policy(
                    assembly,
                    options.matrix_layout,
                    equations.clone(),
                    state_names.clone(),
                    parameter_names.clone(),
                    independent_name.clone(),
                    config,
                    telemetry.clone(),
                    options.execution_policy,
                )?
            }
            (BvpSciExecution::Aot, _, None, None) => {
                return Err(BvpSciNewError::InvalidConfiguration(
                    "AOT execution requires a generated backend configuration".into(),
                ));
            }
            (_, _, None, Some(_)) => {
                return Err(BvpSciNewError::InvalidConfiguration(
                    "generated backend configuration requires AOT execution".into(),
                ));
            }
            (_, _, None, None) => BvpSciLambdifyPlan::prepare(
                assembly,
                &equations,
                &state_names,
                &parameter_names,
                independent_name,
                telemetry.clone(),
            )?,
        };
        let boundary_dimension = state_names.len() + parameter_names.len();
        let boundary = match boundary_jacobian {
            Some(jacobian) => BvpSciBoundaryCallbacks::new_with_jacobian(
                boundary_dimension,
                boundary_callback,
                jacobian,
                telemetry,
            ),
            None => BvpSciBoundaryCallbacks::new(boundary_dimension, boundary_callback, telemetry),
        };
        BvpSciSolver::new_with_sparse_factorization(
            plan,
            boundary,
            mesh,
            initial_state,
            parameters,
            options,
            sparse_factorization,
        )
    }
}
