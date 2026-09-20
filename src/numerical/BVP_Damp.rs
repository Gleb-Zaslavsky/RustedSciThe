/// interface  abstracting lineat algebra operations and solvers
pub mod BVP_traits;
/// utilities for BVP solver in general
pub mod BVP_utils;
/// utilities for BVP solver NR_Damp_solver_damped;
pub mod BVP_utils_damped;
/// main module for damped modified Newton-Raphson solver with analytic Jacobian
pub mod NR_Damp_solver_damped;
/// main module for frozen Newton-Raphson solver with analytic Jacobian
pub mod NR_Damp_solver_frozen;
/// Internal Dense/faer factor-owner runtime for Frozen; not a legacy API.
pub(crate) mod factor_runtime;
/// Shared revision tracking for prepared callback/runtime invalidation.
pub(crate) mod prepared_runtime;
/// Typed read-only runtime plan shared by Damped and Frozen configuration.
pub mod resolved_plan;
/// typed runtime counters and timing snapshots for BVP_Damp
pub mod telemetry;
/// AOT, symbolic-assembly, and backend diagnostic story tests
#[cfg(test)]
#[path = "BVP_Damp/tests/aot_diagnostics.rs"]
mod test_aot_diagnostics;
/// end-to-end race and stress tables for sparse/banded generated backends
#[cfg(test)]
#[path = "BVP_Damp/tests/aot_race_stress.rs"]
mod test_aot_race_stress;
/// focused backend comparison diagnostics for solver-facing BVP generated pipelines
#[cfg(test)]
#[path = "BVP_Damp/tests/backend_compare.rs"]
mod test_backend_compare;
/// classic symbolic BVP examples and grid-refinement tests
#[cfg(test)]
#[path = "BVP_Damp/tests/classic_examples.rs"]
mod test_classic_examples;
/// baseline correctness and pure-numeric BVP_Damp tests
#[cfg(test)]
#[path = "BVP_Damp/tests/basic_correctness.rs"]
mod test_correctness;
/// factorization reuse correctness and telemetry semantics
#[cfg(test)]
#[path = "BVP_Damp/tests/factorization_cache.rs"]
mod test_factorization_cache;
/// provisional release timing story for repeated Frozen Dense/faer/Banded solves
#[cfg(test)]
#[path = "BVP_Damp/tests/frozen_runtime_story.rs"]
mod test_frozen_runtime_story;
/// exact-solution and nonlinear solver-level pure-Lambdify acceptance gates
#[cfg(test)]
#[path = "BVP_Damp/tests/lambdify_acceptance.rs"]
mod test_lambdify_acceptance;
/// broad pure-Lambdify frontend, matrix-backend and execution-policy correctness corpus
#[cfg(test)]
#[path = "BVP_Damp/tests/lambdify_cross_product.rs"]
mod test_lambdify_cross_product;
/// prepared-runtime invalidation and solver-level typed callback-error gates
#[cfg(test)]
#[path = "BVP_Damp/tests/lambdify_lifecycle.rs"]
mod test_lambdify_lifecycle;
/// typed Dense/faer/Banded linear-solve boundary gates
#[cfg(test)]
#[path = "BVP_Damp/tests/linear_solve_boundary.rs"]
mod test_linear_solve_boundary;
/// prepared numeric-parameter rebinding across Dense/faer/Banded Lambdify
#[cfg(test)]
#[path = "BVP_Damp/tests/parameter_rebind.rs"]
mod test_parameter_rebind;
/// deterministic ExprLegacy/AtomView parity corpus for the pure Lambdify path
#[cfg(test)]
#[path = "BVP_Damp/tests/parity_corpus.rs"]
mod test_parity_corpus;
/// release-only telemetry price and adaptive scope story
#[cfg(test)]
#[path = "BVP_Damp/tests/telemetry_story.rs"]
mod test_telemetry_story;
/// fallible Damped/Frozen task and strategy validation gates
#[cfg(test)]
#[path = "BVP_Damp/tests/validation.rs"]
mod test_validation;
/// Typed matrix backend selector for new BVP_Damp callers.
///
/// The historical `DampedSolverOptions::method` and
/// `FrozenSolverOptions::method` string fields remain available as
/// compatibility surfaces; new code should prefer this enum-based API.
pub use crate::symbolic::codegen::codegen_provider_api::MatrixBackend;
pub use NR_Damp_solver_damped::BvpDerivativeScheme;
/// module for basic adaptive grid for NR method
pub mod adaptive_grid_basic;
/// module for more advanced adaptive grid for NR method
pub mod adaptive_grid_twopoint;
/// shared solver handoff types for generated residual/Jacobian callbacks
pub mod generated_solver_handoff;
/// module of interface for creating a new grid
pub mod grid_api;
/// pure numeric BVP discretization helpers used by NumericOnly runtime path
pub mod numeric_discretization;
pub mod solver_common;
/// shared helpers for BVP_Damp test modules
#[cfg(test)]
#[path = "BVP_Damp/tests/common.rs"]
mod test_common;

pub mod task_parser_damped;

/// Runtime collection policy for pure-Lambdify residual/Jacobian callbacks.
///
/// Re-exported here so users do not need to depend on the internal symbolic
/// module path when selecting `Off`, `Counters` or `Detailed` telemetry.
pub use crate::symbolic::bvp::telemetry::{BvpLambdifyExecutionPolicy, BvpLambdifyTelemetryMode};
pub use resolved_plan::{
    BvpEvaluatorKind, BvpLinearAlgorithmKind, BvpParameterBinding, BvpResolvedPlan, BvpStrategyKind,
};
pub use telemetry::{
    BvpLogEvent, BvpLogEventKind, BvpLogLevel, BvpLoggingConfig, BvpLoggingMode, BvpTelemetryMode,
};
