//! Second-generation `BVP_sci` architecture.
//!
//! This namespace is the public replacement for the fragmented legacy
//! implementation. The Lambdify numerical slice includes reusable
//! collocation buffers, native linear backends, a bounded modified-Newton
//! controller and adaptive mesh refinement; remaining SciPy parity work is
//! tracked in the local TODO ledger.
//!
//! Design rules:
//!
//! - the collocation algorithm is independent of matrix storage;
//! - Dense, Sparse and Banded backends write directly to native storage;
//! - ExprLegacy and AtomView are independent Lambdify frontends;
//! - AOT is a generated callback frontend over the same numerical core, not a
//!   second solver algorithm;
//! - telemetry is disabled by default and never substitutes for correctness;
//! - preparation, continuation and solve work are separate lifecycle phases.
#![allow(dead_code)]

pub mod backends;
pub mod builder;
pub mod callbacks;
pub mod collocation;
pub mod config;
pub mod continuation;
pub mod error;
pub mod frontends;
pub mod jacobian;
pub mod linear;
pub mod numerical;
pub mod output;
pub mod prepared;
pub mod problem;
pub mod singular;
pub mod solver;
pub mod telemetry;
pub mod workspace;

pub use callbacks::{BvpSciBoundary, BvpSciBoundaryCallbacks};
pub use builder::BvpSciSolverBuilder;
pub use config::{
    BvpSciAssembly, BvpSciExecution, BvpSciExecutionPolicy, BvpSciMatrixLayout, BvpSciOptions,
    SCIPY_ARMIJO_SIGMA, SCIPY_BACKTRACKING_TAU, SCIPY_MAX_BACKTRACKING_TRIALS,
    SCIPY_MAX_JACOBIAN_REFRESHES, SCIPY_MAX_MESH_ITERATIONS, SCIPY_MAX_NEWTON_ITERATIONS,
};
pub use error::{BvpSciAotFailureKind, BvpSciNewError, BvpSciStage, BvpSciStatus};
pub use frontends::{AtomViewNativeLambdifyPlan, BvpSciAotPlan, ExprLegacyLambdifyPlan};
pub use numerical::{
    BvpSciNumericalPlan, NumericalJacobianCallback, NumericalParameterJacobianCallback,
    NumericalRhsCallback,
};
pub use crate::symbolic::symbolic_ivp_generated::{
    SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
};
pub use output::{BvpSciDenseOutput, BvpSciOutputPolicy, BvpSciSolution};
pub use prepared::BvpSciLambdifyPlan;
pub use singular::BvpSciSingularTerm;

#[cfg(test)]
mod story_tests;
pub use solver::BvpSciSolver;
pub use telemetry::{
    BvpSciBandedRoute, BvpSciNewtonTraceEntry, BvpSciTelemetry, BvpSciTelemetryMode, BvpSciTelemetryScope,
    BvpSciTelemetryScopeKind, BvpSciTelemetrySnapshot, BvpSciTelemetryStage, BvpSciTelemetryTimer,
};
