//! Compatibility facade for the direct BVP symbolic API.
//!
//! The implementation now lives in [`crate::symbolic::bvp::direct`].  New
//! code should import that explicit path; this facade preserves the historical
//! public module name.
//!
//! The direct Banded assembly API remains supported and is not a deprecated
//! experiment. BVP_Damp's canonical matrix backends remain Dense, faer Sparse
//! and Banded; deprecation is applied only to obsolete backend-specific
//! generators in the legacy facade.

pub use crate::symbolic::bvp::direct::*;
