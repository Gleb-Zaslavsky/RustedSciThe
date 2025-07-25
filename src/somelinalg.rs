//! some linear algebra functions used throughout the code
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
/// BiConjugate Gradient Stabilized (BICGSTAB) method for solving linear systems
pub mod BICGSTAB;

pub mod GMRES_mult_api;
pub mod GMRESapi;
pub mod LUsolver;
pub mod Lx_eq_b;
/// matrix inversion algorithms
pub mod some_matrix_inv;

pub mod RustedLINPACK;
/// diagnostics for linear systems and matrices: if it is singular
/// or poorly conditioned
pub mod linear_sys_diagnostics;
