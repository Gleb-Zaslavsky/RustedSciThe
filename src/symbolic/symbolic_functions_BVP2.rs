//! Compatibility facade for the direct BVP symbolic API.
//!
//! The implementation now lives in [`crate::symbolic::bvp::direct`].  New
//! code should import that explicit path; this facade preserves the historical
//! public module name.

pub use crate::symbolic::bvp::direct::*;
