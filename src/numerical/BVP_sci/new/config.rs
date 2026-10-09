//! Validated policy inputs for the new BVP lifecycle.

use super::{output::BvpSciOutputPolicy, singular::BvpSciSingularTerm};
use crate::symbolic::bvp::telemetry::BvpLambdifyExecutionPolicy;

/// Execution policy for independent warm Lambdify callback entries.
///
/// This alias keeps BVP_sci on the crate-wide BVP policy contract. It does not
/// make the numerical collocation loop itself concurrent: only independent
/// residual/Jacobian scalar evaluators may dispatch to Rayon workers.
pub type BvpSciExecutionPolicy = BvpLambdifyExecutionPolicy;

/// SciPy `_bvp.solve_newton` controller constants.
pub const SCIPY_MAX_JACOBIAN_REFRESHES: usize = 4;
pub const SCIPY_MAX_NEWTON_ITERATIONS: usize = 8;
pub const SCIPY_MAX_MESH_ITERATIONS: usize = 10;
pub const SCIPY_MAX_BACKTRACKING_TRIALS: usize = 4;
pub const SCIPY_ARMIJO_SIGMA: f64 = 0.2;
pub const SCIPY_BACKTRACKING_TAU: f64 = 0.5;

/// Runtime family selected during preparation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BvpSciExecution {
    /// Plain numerical callbacks, including finite-difference Jacobians.
    Numerical,
    /// Interpreted symbolic callback evaluation.
    Lambdify,
    /// Generated native evaluation, to be enabled after the Lambdify core.
    Aot,
}

/// Symbolic representation used by the Lambdify frontend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BvpSciAssembly {
    /// User-supplied numerical callbacks; no symbolic representation exists.
    Numerical,
    ExprLegacy,
    AtomViewNative,
}

/// Native storage selected once for a prepared collocation system.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BvpSciMatrixLayout {
    Dense,
    Sparse,
    Banded { lower: usize, upper: usize },
}

/// Solver policies which do not belong in the numerical hot loop.
///
/// The two iteration budgets deliberately remain explicit configuration rather
/// than hidden constants.  `max_newton_iterations` limits work on one mesh;
/// `max_mesh_refinements` limits the outer collocation controller.  A separate
/// Jacobian budget makes modified-Newton reuse observable and bounded.
#[derive(Clone, Debug)]
pub struct BvpSciOptions {
    /// Runtime family selected before numerical solving.
    pub execution: BvpSciExecution,
    /// Symbolic assembly family, when the Lambdify route is used.
    pub assembly: Option<BvpSciAssembly>,
    /// Native matrix storage selected once for each prepared mesh.
    pub matrix_layout: BvpSciMatrixLayout,
    /// Collocation defect tolerance used by Newton and mesh refinement.
    pub tolerance: f64,
    /// Boundary residual tolerance; defaults to `tolerance` when absent.
    pub boundary_tolerance: Option<f64>,
    /// Hard upper bound for adaptive mesh nodes.
    pub max_nodes: usize,
    /// Maximum Newton iterations on one mesh.
    pub max_newton_iterations: usize,
    /// Maximum number of Jacobian assembly/factorization refreshes per mesh.
    pub max_jacobian_refreshes: usize,
    /// Maximum number of accepted adaptive mesh refinements.
    pub max_mesh_refinements: usize,
    /// Maximum affine-cost trial evaluations for one Newton step.
    ///
    /// The first evaluation is the full Newton step. This matches SciPy's
    /// `n_trial` contract; the historical field name is retained for API
    /// compatibility.
    pub max_backtracking_steps: usize,
    /// Disabled by default; enabled modes add counters or timings only.
    pub telemetry: super::telemetry::BvpSciTelemetryMode,
    /// Warm callback dispatch policy; sequential is the safe default.
    pub execution_policy: BvpSciExecutionPolicy,
    /// Output retention policy. Dense output is built only when requested.
    pub output_policy: BvpSciOutputPolicy,
    /// Optional SciPy-style singular term `S*y/(x-a)`.
    pub singular_term: Option<BvpSciSingularTerm>,
}

impl Default for BvpSciOptions {
    fn default() -> Self {
        Self {
            execution: BvpSciExecution::Numerical,
            assembly: None,
            matrix_layout: BvpSciMatrixLayout::Dense,
            tolerance: 1e-3,
            boundary_tolerance: None,
            max_nodes: 1_000,
            max_newton_iterations: SCIPY_MAX_NEWTON_ITERATIONS,
            max_jacobian_refreshes: SCIPY_MAX_JACOBIAN_REFRESHES,
            max_mesh_refinements: SCIPY_MAX_MESH_ITERATIONS,
            max_backtracking_steps: SCIPY_MAX_BACKTRACKING_TRIALS,
            telemetry: super::telemetry::BvpSciTelemetryMode::Off,
            execution_policy: BvpSciExecutionPolicy::Sequential,
            output_policy: BvpSciOutputPolicy::MeshAndValues,
            singular_term: None,
        }
    }
}

impl BvpSciOptions {
    /// Set the symbolic frontend family used by the prepared model.
    pub fn with_execution(mut self, execution: BvpSciExecution) -> Self {
        self.execution = execution;
        self
    }

    /// Select ExprLegacy or AtomViewNative symbolic assembly.
    pub fn with_assembly(mut self, assembly: BvpSciAssembly) -> Self {
        self.assembly = Some(assembly);
        self
    }

    /// Select the native Dense, Sparse or Banded linear storage.
    pub fn with_matrix_layout(mut self, matrix_layout: BvpSciMatrixLayout) -> Self {
        self.matrix_layout = matrix_layout;
        self
    }

    /// Set the collocation defect tolerance.
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set the independent boundary residual tolerance.
    pub fn with_boundary_tolerance(mut self, tolerance: f64) -> Self {
        self.boundary_tolerance = Some(tolerance);
        self
    }

    /// Set the adaptive mesh and Newton work budgets in one place.
    pub fn with_limits(
        mut self,
        max_nodes: usize,
        max_newton_iterations: usize,
        max_mesh_refinements: usize,
    ) -> Self {
        self.max_nodes = max_nodes;
        self.max_newton_iterations = max_newton_iterations;
        self.max_mesh_refinements = max_mesh_refinements;
        self
    }

    /// Set the maximum number of trial halvings for one Newton correction.
    pub fn with_backtracking_limit(mut self, max_backtracking_steps: usize) -> Self {
        self.max_backtracking_steps = max_backtracking_steps;
        self
    }

    /// Set the callback execution policy.
    pub fn with_execution_policy(mut self, policy: BvpSciExecutionPolicy) -> Self {
        self.execution_policy = policy;
        self
    }

    /// Enable the requested telemetry collection level.
    pub fn with_telemetry(mut self, telemetry: super::telemetry::BvpSciTelemetryMode) -> Self {
        self.telemetry = telemetry;
        self
    }

    /// Set the output retention policy.
    pub fn with_output_policy(mut self, output_policy: BvpSciOutputPolicy) -> Self {
        self.output_policy = output_policy;
        self
    }

    /// Enable the optional SciPy singular term `S*y/(x-a)`.
    pub fn with_singular_term(mut self, singular_term: BvpSciSingularTerm) -> Self {
        self.singular_term = Some(singular_term);
        self
    }
}
