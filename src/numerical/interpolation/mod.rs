//! Interpolation and extrapolation utilities shared by numerical solvers.

pub mod inter_n_extrapolate;
pub mod ppoly;

/// Compatibility module name retained for existing callers.
pub use ppoly as PPoly;
