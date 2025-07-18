#[allow(non_snake_case)]
///here is main loop for solving nonlinear equation system with Levenberg-Marquardt algorithm
pub mod LM_optimization;
/// some special cases of fitting
pub mod fitting_features;
#[allow(non_snake_case)]
/// main function to solve nonlinear equation system with Levenberg-Marquardt algorithm
pub mod problem_LM;
/// some linear algebra functions for solving nonlinear equation system
#[allow(non_snake_case)]
pub mod qr_LM;
/// nonlinear equation system solver implementation for fittting data. Fiitting function is symbolic expression.
pub mod sym_fitting;
/// solver of nonlinear equation system with Levenberg-Marquardt algorithm is
/// used with residual functions that are symbolic expressions and jacobian functions that are calculated analytically
pub mod sym_wrapper;
/// trust region subproblem solver for Levenberg-Marquardt algorithm
#[allow(non_snake_case)]
pub mod trust_region_LM;
/// some utility functions for solving nonlinear equation system with Levenberg-Marquardt algorithm
pub mod utils;

pub mod Gavin_chi;

/// interpolation and extrapolation of data
pub mod inter_n_extrapolate;
pub mod lm_gavin;
/// using Bisection, Secant,, Newton, and Brent methods to find the minimum of a scalar function of one variable
pub mod minimize_scalar;
