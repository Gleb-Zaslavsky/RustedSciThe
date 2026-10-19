//! some linear algebra functions used throughout the code
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

pub mod iterative_solvers_cpu;
/// diagnostics for linear systems and matrices: if it is singular
/// or poorly conditioned
pub mod linear_sys_diagnostics;

pub mod RustedLINPACK;

pub mod banded;
/// Experimental reusable sparse LU kernel isolated for BVP_sci work.
pub(crate) mod bvp_sparse_lu;
#[cfg(test)]
mod bvp_sparse_lu_tests;
pub mod iterative_solvers_gpu;
/// Dense column-pivoted QR and diagonal least-squares kernels.
pub mod least_squares_qr;
#[cfg(test)]
mod sparse_lu_reference;
#[cfg(test)]
mod sparse_lu_reference_tests;
/// Safe Rust interface to the optional, dynamically loaded SuperLU backend.
#[cfg(feature = "superlu")]
mod sparse_lu_superlu_experimental;
/// Shared safe API for optional SuperLU sparse factorizations.
#[cfg(feature = "superlu")]
pub mod superlu;
