//! Radau solver facade.
//!
//! The current public module names are retained as compatibility shims while
//! the second-generation implementation is built in [`new`]. The old source
//! files are intentionally kept under `*_old` names as executable references
//! during the migration.

/// Archived main solver, retained inside the crate for reference tests only.
#[path = "Radau/Radau_main_old.rs"]
pub(crate) mod Radau_main;
/// Archived Newton solver, retained inside the crate for reference tests.
#[path = "Radau/Radau_newton_old.rs"]
pub(crate) mod Radau_newton;
/// Archived parallel Newton solver, retained inside the crate for reference tests.
#[path = "Radau/Radau_newton_par_old.rs"]
pub(crate) mod Radau_newton_par;

/// Stable second-generation public facade.
pub mod api;
pub use api::{
    RadauAotConfig, RadauConfig, RadauError, RadauErrorKind, RadauExecution, RadauExecutionPolicy,
    RadauFrontend, RadauJacobianSource, RadauMatrixLayout, RadauNativeSolver, RadauOutputPolicy,
    RadauProblem, RadauSolution, RadauSolver, RadauTelemetryMode, RadauTelemetryReport,
    RadauTelemetryScopeKind, RadauTelemetryScopeMetadata,
};

/// Convenient import set for ordinary symbolic Radau applications.
pub mod prelude {
    pub use super::{
        RadauAotConfig, RadauConfig, RadauError, RadauErrorKind, RadauExecution,
        RadauExecutionPolicy, RadauFrontend, RadauJacobianSource, RadauMatrixLayout,
        RadauNativeSolver, RadauOutputPolicy, RadauProblem, RadauSolution, RadauSolver,
        RadauTelemetryMode, RadauTelemetryReport, RadauTelemetryScopeKind,
        RadauTelemetryScopeMetadata,
    };
}

/// Internal second-generation implementation.
pub(crate) mod new;

/// Diagnostic harness shared by Radau stories and Criterion benches.
///
/// This module is intentionally hidden from the stable API. It exists so
/// story tests and Criterion benches share the same prepared workloads while
/// the public solver facade is still being migrated.
#[doc(hidden)]
pub mod benchmark {
    pub use super::new::benchmark::{
        BenchmarkAssembly, BenchmarkExecutionPolicy, BenchmarkLayout, PreparedAotCallbacks,
        PreparedBenchmark,
    };
}

/// Keep the archived implementation as a correctness and behavior reference.
#[cfg(test)]
#[path = "Radau/Radau_test_old.rs"]
mod Radau_test;

/// New story-test layout. The archived test module remains a numerical
/// reference while the new route grows its independent coverage.
#[cfg(test)]
#[path = "Radau/tests/mod.rs"]
mod tests;
