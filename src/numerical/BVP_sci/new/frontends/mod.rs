//! Symbolic and numerical frontend boundaries.

pub mod atom_native;
pub mod expr_legacy;

/// Generated native callbacks reuse the crate-wide AOT lifecycle. Compiler
/// policy stays in the preparation boundary and never leaks into collocation.
pub mod aot;

pub use aot::BvpSciAotPlan;
pub use atom_native::AtomViewNativeLambdifyPlan;
pub use expr_legacy::ExprLegacyLambdifyPlan;
