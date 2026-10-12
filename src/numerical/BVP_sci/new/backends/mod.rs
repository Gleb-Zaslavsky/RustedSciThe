//! Native linear backends for the collocation Jacobian.

pub mod banded;
pub mod dense;
pub mod sparse;
/// Experimental reusable-faer alternatives; not the default sparse backend.
pub mod sparse_reusable_experimental;
#[cfg(feature = "superlu")]
pub mod superlu;

pub use banded::BandedBackend;
pub use dense::DenseBackend;
pub use sparse::SparseBackend;
pub use sparse_reusable_experimental::ReusableSparseExperimentalBackend;
#[cfg(feature = "superlu")]
pub use superlu::SuperLuBackend;
