//! Compatibility facade for the historical BVP symbolic API.
//!
//! The implementation now lives in [`crate::symbolic::bvp::legacy`].  This
//! module name remains public so downstream users and old story tests do not
//! need to change imports during the architecture migration.
//!
//! This facade is not deprecated as a whole: BVP_Damp and other internal
//! consumers still use supported shared BVP functionality through it. Only
//! obsolete `sprs` and removed nalgebra `CsMatrix` Jacobian callbacks are
//! deprecated individually. The canonical BVP_Damp matrix backends are Dense,
//! faer Sparse and Banded.

pub use crate::symbolic::bvp::legacy::*;
