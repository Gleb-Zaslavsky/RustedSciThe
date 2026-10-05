/// api for BDF solver
pub mod BDF_api;
/// Stable convenience imports for applications using the dense BDF solver.
///
/// The prelude intentionally contains only the high-level solver API, typed
/// errors, telemetry controls, and symbolic backend selectors. Low-level
/// linear backend traits remain available from [`BDF_solver`] when an
/// application needs to provide a custom implementation.
pub mod prelude {
    pub use super::BDF_api::{
        BdfAotProvenance, BdfNativeJacobianSource, BdfPreparationTimings, BdfSolveError,
        BdfSolverOptions, BdfStatus, BdfStopConditionError, BdfTelemetryMode, ODEsolver,
    };
    pub use crate::numerical::BDF::BDF_solver::{
        BdfConfigurationError, BdfJacobian, BdfJacobianCallbackError, BdfJacobianSource,
        BdfOperationCounters, BdfStepError,
    };
    pub use crate::symbolic::ivp_telemetry::IvpLambdifyExecutionPolicy;
    pub use crate::symbolic::symbolic_ivp::{IvpBackendError, IvpSymbolicAssemblyBackend};
    pub use crate::symbolic::symbolic_ivp_generated::{
        DenseIvpGeneratedBackendMode, SymbolicIvpGeneratedBackendConfig,
    };
}
///pub mod BDF;
/// SOLVER OF STIFF IVP
/// direct rewrite to Rust python code from SciPy
pub mod BDF_solver;
/// some utilities for BDF solver
mod BDF_utils;
/// some utilities for ODE solvers (now written only BDF)
///
pub mod common;

#[cfg(test)]
#[path = "BDF/tests/telemetry.rs"]
mod telemetry_tests;

#[cfg(test)]
#[path = "BDF/tests/performance_story_tests.rs"]
mod performance_story_tests;

#[cfg(test)]
#[path = "BDF/tests/numerical_fidelity_story_tests.rs"]
mod numerical_fidelity_story_tests;
