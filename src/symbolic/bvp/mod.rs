//! Explicit module boundaries for the BVP symbolic pipeline.
//!
//! The BVP symbolic implementation is being migrated in stages.  New code
//! must enter through [`direct`] and [`telemetry`], while compatibility code
//! remains available through [`legacy`] and the historical
//! `symbolic_functions_BVP` module. `legacy_symbolic` owns ExprLegacy
//! differentiation, `legacy_lambdify` owns ExprLegacy callback builders, and
//! `atom_lambdify` owns callbacks compiled directly from packed Atom graphs.
//! Their callback telemetry uses one schema but separate handles. Keeping these
//! names explicit prevents a new no-Mutex callback from accidentally invoking a
//! legacy callback. The legacy solver crosses into the direct path only by
//! creating an owned prepared snapshot; remaining native Banded Atom work is
//! tracked in BVP_Damp's TODO.

pub(crate) mod aot_telemetry;
pub(crate) mod atom_aot;
pub(crate) mod atom_lambdify;
pub mod direct;
pub mod legacy;
pub mod legacy_lambdify;
pub(crate) mod legacy_symbolic;
pub(crate) mod parameter_binding;
pub mod telemetry;
