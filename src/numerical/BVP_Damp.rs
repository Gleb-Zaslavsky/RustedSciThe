/// tests for BVP_Damp module
mod BVP_Damp_tests;
/// interface  abstracting lineat algebra operations and solvers
pub mod BVP_traits;
/// utilities for BVP solver in general
pub mod BVP_utils;
/// utilities for BVP solver NR_Damp_solver_damped;
mod BVP_utils_damped;
/// main module for damped modified Newton-Raphson solver with analytic Jacobian 
pub mod NR_Damp_solver_damped;
/// main module for frozen Newton-Raphson solver with analytic Jacobian 
pub mod NR_Damp_solver_frozen;
/// module for basic adaptive grid for NR method 
mod adaptive_grid_basic;
/// module for more advanced adaptive grid for NR method
mod adaptive_grid_twopoint;
/// module of interface for creating a new grid
mod grid_api;
/// module for linear system solvers 
mod linear_sys_solvers_depot;
