//! some linear algebra functions used throughout the code
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

pub mod iterative_solvers_cpu;
/// diagnostics for linear systems and matrices: if it is singular
/// or poorly conditioned
pub mod linear_sys_diagnostics;

pub mod RustedLINPACK;

pub mod iterative_solvers_gpu;
