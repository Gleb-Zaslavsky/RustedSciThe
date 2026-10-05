//! Second-generation Radau architecture.
//!
//! The namespace contains the adaptive numerical core plus explicit
//! callback/backend contracts. The archived implementation remains a
//! compatibility and numerical reference; the public route now uses this
//! namespace, including shared generated-IVP AOT preparation.
//!
//! The temporary allowance below is limited to the migration namespace. It
//! must be removed when the public route is switched from the archive; until
//! then it prevents dead-code warnings from hiding the old compatibility gate.
#![allow(dead_code)]

pub(crate) mod analytic_jacobian;
pub(crate) mod aot;
pub(crate) mod atom_native;
pub(crate) mod benchmark;
pub(crate) mod callbacks;
pub(crate) mod coefficients;
pub(crate) mod collocation;
pub(crate) mod config;
pub(crate) mod continuation;
pub(crate) mod controller;
pub(crate) mod dense_output;
pub(crate) mod error;
pub(crate) mod finite_difference;
pub(crate) mod jacobian;
pub(crate) mod lambdify;
pub(crate) mod linear;
pub(crate) mod native_callbacks;
pub(crate) mod output;
pub(crate) mod prepared;
pub(crate) mod session;
pub(crate) mod solver;
pub(crate) mod state;
pub(crate) mod step;
pub(crate) mod telemetry;
pub(crate) mod workspace;
