//! Compatibility facade for the historical BVP symbolic API.
//!
//! The implementation now lives in [`crate::symbolic::bvp::legacy`].  This
//! module name remains public so downstream users and old story tests do not
//! need to change imports during the architecture migration.

pub use crate::symbolic::bvp::legacy::*;
