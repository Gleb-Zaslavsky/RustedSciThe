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

/// Serializes tests that mutate the process-wide linked generated-backend
/// registry, preventing AOT lifecycle stories from racing with test cleanup.
#[cfg(test)]
pub(crate) fn generated_backend_registry_test_guard() -> std::sync::MutexGuard<'static, ()> {
    use std::sync::{Mutex, OnceLock};

    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .expect("generated backend registry test lock should not be poisoned")
}

/// Second-generation BVP architecture and the public solver facade.
pub mod new;

// Keep the historical implementation files as conventional child modules of
// `BVP_sci`; `legacy` below is only the compatibility namespace. This avoids
// path attributes whose `..` resolution differs across build/package layouts.
#[doc(hidden)]
pub mod BVP_sci_aot;
#[doc(hidden)]
pub mod BVP_sci_banded;
#[doc(hidden)]
pub mod BVP_sci_bordered_banded;
#[doc(hidden)]
pub mod BVP_sci_bordered_solver;
/// Historical solver using `faer` matrix and vector operations.
#[doc(hidden)]
pub mod BVP_sci_faer;
/// Historical alternative solver using `nalgebra` matrix operations.
#[doc(hidden)]
pub mod BVP_sci_nalgebra;
#[doc(hidden)]
pub mod BVP_sci_numerical;
#[doc(hidden)]
pub mod BVP_sci_symb;
#[doc(hidden)]
pub mod BVP_sci_symbolic_functions;
#[doc(hidden)]
pub(crate) mod BVP_sci_utils;

/// Historical BVP implementation retained for reference and compatibility.
///
/// New code should use the re-exported items from [`new`] instead. This
/// namespace is intentionally explicit so benchmarks and parity tests can
/// identify when they are exercising the old baseline.
#[doc(hidden)]
pub mod legacy;

#[cfg(test)]
pub(crate) use legacy::test_common;

pub use new::{
    AtomViewNativeLambdifyPlan, BvpSciAotFailureKind, BvpSciAotPlan, BvpSciAssembly,
    BvpSciBoundary, BvpSciBoundaryCallbacks, BvpSciDenseOutput, BvpSciExecution,
    BvpSciExecutionPolicy, BvpSciLambdifyPlan, BvpSciMatrixLayout, BvpSciNewError,
    BvpSciNewtonTraceEntry, BvpSciNumericalPlan, BvpSciOptions, BvpSciOutputPolicy,
    BvpSciSingularTerm, BvpSciSolution, BvpSciSolver, BvpSciSolverBuilder, BvpSciStage,
    BvpSciStatus, BvpSciTelemetry, BvpSciTelemetryMode, BvpSciTelemetryScope,
    BvpSciTelemetryScopeKind, BvpSciTelemetrySnapshot, BvpSciTelemetryStage,
    ExprLegacyLambdifyPlan, NumericalJacobianCallback, NumericalParameterJacobianCallback,
    NumericalRhsCallback, SCIPY_ARMIJO_SIGMA, SCIPY_BACKTRACKING_TAU,
    SCIPY_MAX_BACKTRACKING_TRIALS, SCIPY_MAX_JACOBIAN_REFRESHES, SCIPY_MAX_MESH_ITERATIONS,
    SCIPY_MAX_NEWTON_ITERATIONS, SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
};
