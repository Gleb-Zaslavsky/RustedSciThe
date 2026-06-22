//! Common imports that are frequently used when working with this crate.
//!
//! This module re-exports the most commonly used types and traits to make
//! it easier to get started with the library.

/// The local f64 least-squares trait used by the VarPro Levenberg-Marquardt bridge
pub use crate::numerical::optimization::problem_LM::LeastSquaresProblem;
/// The trait for describing basis functions
pub use crate::numerical::optimization::varpro::basis_function::BasisFunction;
/// The trait for describing separable nonlinear models
pub use crate::numerical::optimization::varpro::model::SeparableNonlinearModel;
/// The builder for creating separable models
pub use crate::numerical::optimization::varpro::model::builder::SeparableModelBuilder;
/// Symbolic helpers and boxed closure types for building VarPro basis functions
pub use crate::numerical::optimization::varpro::symbolic::{
    InvariantBasisFn, SymbolicVarProBuilder, SymbolicVarProError, SymbolicVarProFit, UnaryBasisFn,
    invariant_basis, unary_basis, unary_basis_and_derivative,
};
