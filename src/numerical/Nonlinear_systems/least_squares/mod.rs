//! Dedicated rectangular least-squares core for nonlinear fitting.
//!
//! This module is the rectangular least-squares counterpart of the square
//! nonlinear-system solver. It owns the numerical LM/QR/trust-region
//! implementation and symbolic least-squares adapter. Data fitting, VarPro,
//! and statistics adapters remain in `numerical::optimization`.
//!
//! The old duplicate numerical core under `numerical::optimization` has been
//! removed. This module is now the canonical controller for rectangular
//! least-squares problems.

pub mod errors;
pub mod lm;
pub mod problem;
pub mod symbolic;
pub mod symbolic_solver;
pub mod trust_region;
pub mod utils;

#[cfg(test)]
mod parity_tests;

pub use errors::{LeastSquaresError, LeastSquaresStage};
pub use lm::{
    LeastSquaresTelemetryMode, LevenbergMarquardt, MinimizationReport,
    TerminationReason as LeastSquaresTerminationReason,
};
pub use problem::{ClosureLeastSquaresProblem, LeastSquaresProblem};
pub use symbolic::{BoundSymbolicLeastSquaresProblem, PreparedSymbolicLeastSquaresProblem};
pub use symbolic_solver::SymbolicLeastSquaresSolver;
