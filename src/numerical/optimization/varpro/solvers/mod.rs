/// Contains solvers using the [levenberg-marquardt](https://crates.io/crates/levenberg-marquardt)
/// crate.
///
/// This module provides implementations of optimization algorithms for solving the
/// nonlinear least squares problem in variable projection. Currently, it contains
/// the [`levmar`] module which implements the Levenberg-Marquardt algorithm
/// with multiple linear solver backends (SVD and column-pivoted QR decomposition).
pub mod levmar;
