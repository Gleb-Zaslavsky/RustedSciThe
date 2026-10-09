//! Native linear backends for the collocation Jacobian.

pub mod banded;
pub mod dense;
pub mod sparse;

pub use banded::BandedBackend;
pub use dense::DenseBackend;
pub use sparse::SparseBackend;
