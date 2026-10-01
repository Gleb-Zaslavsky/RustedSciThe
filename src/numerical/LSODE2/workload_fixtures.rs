//! Backwards-compatible import path for the shared solver-neutral IVP corpus.
//!
//! New solver tests and benches should import `numerical::ivp_workloads` so
//! workload ownership does not imply an LSODE2 dependency.

pub use crate::numerical::ivp_workloads::*;
