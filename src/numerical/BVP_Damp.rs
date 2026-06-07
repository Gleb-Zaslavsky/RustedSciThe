/// focused backend comparison diagnostics for solver-facing BVP generated pipelines
#[cfg(test)]
#[path = "BVP_Damp/tests/backend_compare.rs"]
mod BVP_Damp_codegen_compare_tests;
/// baseline correctness and pure-numeric BVP_Damp tests
#[cfg(test)]
#[path = "BVP_Damp/tests/basic_correctness.rs"]
mod BVP_Damp_tests;
/// classic symbolic BVP examples and grid-refinement tests
#[cfg(test)]
#[path = "BVP_Damp/tests/classic_examples.rs"]
mod BVP_Damp_tests2;
/// AOT, symbolic-assembly, and backend diagnostic story tests
#[cfg(test)]
#[path = "BVP_Damp/tests/aot_diagnostics.rs"]
mod BVP_Damp_tests3;
/// end-to-end race and stress tables for sparse/banded generated backends
#[cfg(test)]
#[path = "BVP_Damp/tests/aot_race_stress.rs"]
mod BVP_Damp_tests4;
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
