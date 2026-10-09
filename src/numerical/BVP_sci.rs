//! Public Boundary Value Problem solver API.
//!
//! The root exposes the second-generation collocation implementation from
//! [`new`]. The previous implementation is retained under [`legacy`] as a
//! reference and compatibility route, but it is deliberately not the default
//! public solver anymore.
//!
//! Keeping the old implementation in a named namespace prevents accidental
//! coupling to its internal modules while preserving a migration path for old
//! examples and downstream code that still needs the historical baseline.

/// Second-generation BVP architecture and the public solver facade.
pub mod new;

/// Historical BVP implementation retained for reference and compatibility.
///
/// New code should use the re-exported items from [`new`] instead. This
/// namespace is intentionally explicit so benchmarks and parity tests can
/// identify when they are exercising the old baseline.
#[doc(hidden)]
pub mod legacy {
    #[path = "../BVP_sci_aot.rs"]
    pub mod BVP_sci_aot;
    #[path = "../BVP_sci_banded.rs"]
    pub mod BVP_sci_banded;
    #[path = "../BVP_sci_bordered_banded.rs"]
    pub mod BVP_sci_bordered_banded;
    #[path = "../BVP_sci_bordered_solver.rs"]
    pub mod BVP_sci_bordered_solver;
    /// Historical solver using `faer` matrix and vector operations.
    #[path = "../BVP_sci_faer.rs"]
    pub mod BVP_sci_faer;
    /// Historical alternative solver using `nalgebra` matrix operations.
    #[path = "../BVP_sci_nalgebra.rs"]
    pub mod BVP_sci_nalgebra;
    #[path = "../BVP_sci_numerical.rs"]
    pub mod BVP_sci_numerical;
    #[path = "../BVP_sci_symb.rs"]
    pub mod BVP_sci_symb;
    #[path = "../BVP_sci_symbolic_functions.rs"]
    pub mod BVP_sci_symbolic_functions;
    #[path = "../BVP_sci_utils.rs"]
    pub(crate) mod BVP_sci_utils;

    #[cfg(test)]
    #[path = "../BVP_sci_aot_tests.rs"]
    mod BVP_sci_aot_tests;
    #[cfg(test)]
    #[path = "../tests/banded_story.rs"]
    mod BVP_sci_banded_story_tests;
    #[cfg(test)]
    #[path = "../BVP_sci_banded_tests.rs"]
    mod BVP_sci_banded_tests;
    #[cfg(test)]
    #[path = "../BVP_sci_bordered_banded_tests.rs"]
    mod BVP_sci_bordered_banded_tests;
    #[cfg(test)]
    #[path = "../BVP_sci_bordered_solver_tests.rs"]
    mod BVP_sci_bordered_solver_tests;
    #[cfg(test)]
    #[path = "../BVP_sci_faer_tests.rs"]
    mod BVP_sci_faer_tests;
    #[cfg(test)]
    #[path = "../BVP_sci_generated_compare_tests.rs"]
    mod BVP_sci_generated_compare_tests;
    #[cfg(test)]
    #[path = "../BVP_sci_nalgebra_tests.rs"]
    mod BVP_sci_nalgebra_tests;
    #[cfg(test)]
    #[path = "../BVP_sci_numerical_tests.rs"]
    mod BVP_sci_numerical_tests;
    #[cfg(test)]
    #[path = "../BVP_sci_story_tests.rs"]
    mod BVP_sci_story_tests;
    #[cfg(test)]
    #[path = "../BVP_sci_symb_tests.rs"]
    mod BVP_sci_symb_tests;
    #[cfg(test)]
    #[path = "../BVP_sci_symb_tests2.rs"]
    mod BVP_sci_symb_tests2;
    #[cfg(test)]
    #[path = "../tests/common.rs"]
    pub(crate) mod test_common;
}

// The old source files use the historical root paths internally. Keep these
// crate-local aliases while making the old modules externally visible only via
// `BVP_sci::legacy`.
pub(crate) use legacy::BVP_sci_aot;
pub(crate) use legacy::BVP_sci_banded;
pub(crate) use legacy::BVP_sci_bordered_banded;
pub(crate) use legacy::BVP_sci_bordered_solver;
pub(crate) use legacy::BVP_sci_faer;
pub(crate) use legacy::BVP_sci_nalgebra;
pub(crate) use legacy::BVP_sci_numerical;
pub(crate) use legacy::BVP_sci_symb;
pub(crate) use legacy::BVP_sci_symbolic_functions;
pub(crate) use legacy::BVP_sci_utils;
#[cfg(test)]
pub(crate) use legacy::test_common;

pub use new::{
    AtomViewNativeLambdifyPlan, BvpSciAotPlan, BvpSciAssembly, BvpSciBoundary,
    BvpSciBoundaryCallbacks, BvpSciDenseOutput, BvpSciExecution,
    BvpSciExecutionPolicy, BvpSciLambdifyPlan, BvpSciMatrixLayout, BvpSciOptions,
    BvpSciSolverBuilder,
    BvpSciOutputPolicy, BvpSciSingularTerm, BvpSciSolution, BvpSciSolver,
    BvpSciNumericalPlan, NumericalJacobianCallback, NumericalParameterJacobianCallback,
    NumericalRhsCallback,
    BvpSciAotFailureKind, BvpSciNewError, BvpSciStage, BvpSciStatus, BvpSciTelemetryMode,
    BvpSciTelemetry,
    BvpSciTelemetryScope, BvpSciTelemetryScopeKind, BvpSciTelemetrySnapshot,
    BvpSciTelemetryStage, BvpSciNewtonTraceEntry, ExprLegacyLambdifyPlan,
    SCIPY_ARMIJO_SIGMA, SCIPY_BACKTRACKING_TAU, SCIPY_MAX_BACKTRACKING_TRIALS,
    SCIPY_MAX_JACOBIAN_REFRESHES, SCIPY_MAX_MESH_ITERATIONS, SCIPY_MAX_NEWTON_ITERATIONS,
    SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
};
