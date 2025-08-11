//!
//! # BVP_sci - Advanced Boundary Value Problem Solver Module
//!
//! This module provides a comprehensive suite of high-performance solvers for Boundary Value Problems (BVP)
//! in ordinary differential equations. It implements state-of-the-art numerical methods with both symbolic
//! and direct numerical approaches.
//!
//! ## Key Features
//! - **Multiple Linear Algebra Backends**: Support for both `faer` and `nalgebra` crates
//! - **Symbolic Integration**: Automatic conversion from symbolic expressions to numerical functions
//! - **Analytical Jacobians**: Symbolic differentiation for faster convergence
//! - **Adaptive Mesh Refinement**: Automatic grid adaptation for optimal accuracy
//! - **Collocation Methods**: 4th-order collocation algorithms with residual control
//! - **Sparse Matrix Support**: Efficient handling of large, sparse systems
//! - **Variable Bounds**: Numerical stability through variable constraints
//! - **Comprehensive Testing**: Extensive test suites for reliability
//!
//! ## Module Structure
//! - `BVP_sci_faer`: Core BVP solver using faer linear algebra (recommended for performance)
//! - `BVP_sci_nalgebra`: Alternative implementation using nalgebra
//! - `BVP_sci_symb`: High-level symbolic wrapper for easy problem setup
//! - `BVP_sci_symbolic_functions`: Symbolic-to-numerical conversion utilities
//! - `BVP_sci_utils`: Common utilities and helper functions
//!
//! ## Supported Problem Types
//! - Linear and nonlinear BVPs
//! - Systems of coupled ODEs
//! - Problems with parameters
//! - Stiff and non-stiff equations
//! - Multi-point boundary conditions
//!
//! ## Performance Characteristics
//! - Parallel symbolic differentiation
//! - Sparse matrix optimizations
//! - Bandwidth-aware Jacobian storage
//! - Memory-efficient mesh handling
//! - Adaptive error control
//!
///BVP solver using faer crate for matrix&vector operations
pub mod BVP_sci_faer;
mod BVP_sci_faer_tests;
/// BVP solver using nalgebra crate for matrix&vector operations
pub mod BVP_sci_nalgebra;
mod BVP_sci_nalgebra_tests;
pub mod BVP_sci_symb;
mod BVP_sci_symb_tests;
mod BVP_sci_symb_tests2;
pub mod BVP_sci_symbolic_functions;
mod BVP_sci_utils;
