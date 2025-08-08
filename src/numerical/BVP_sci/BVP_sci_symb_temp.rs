//!
//! # BVP Symbolic Wrapper Module
//!
//! This module provides a high-level wrapper for solving Boundary Value Problems (BVP)
//! using symbolic expressions. It combines symbolic computation with numerical BVP solving
//! to provide an easy-to-use interface for complex differential equation systems.
//!
//! ## Key Features
//! - **Symbolic to Numerical**: Converts symbolic ODE expressions to efficient numerical functions
//! - **Automatic Jacobian**: Generates analytical Jacobians from symbolic expressions
//! - **Bounds Support**: Applies variable bounds to prevent numerical issues (e.g., log(0))
//! - **Flexible Mesh**: Supports both uniform and custom mesh definitions
//! - **Multiple Solvers**: Integrates with advanced BVP solvers with error control
//! - **Result Analysis**: Provides plotting, saving, and statistical analysis of solutions
//!
//! ## Main Structure: `BVPwrap`
//! The `BVPwrap` struct is the main interface that:
//! 1. Takes symbolic ODE expressions as strings or `Expr` objects
//! 2. Converts them to numerical functions with bounds handling
//! 3. Generates analytical Jacobians for faster convergence
//! 4. Solves the BVP using state-of-the-art numerical methods
//! 5. Provides results in various formats (matrices, plots, files)
//!
//! ## Usage Example
//! ```rust,ignore
//! // Define ODE system: y'' = -y, y(0) = 0, y'(0) = 1
//! let equations = vec!["z".to_string(), "-y".to_string()];
//! let variables = vec!["y".to_string(), "z".to_string()];
//! let boundary_conditions = HashMap::from([
//!     ("y".to_string(), vec![(0, 0.0)]),  // y(0) = 0
//!     ("z".to_string(), vec![(0, 1.0)]),  // z(0) = 1
//! ]);
//! 
//! let mut solver = BVPwrap::new(
//!     None, Some(0.0), Some(π), Some(100),
//!     Expr::parse_vector_expression(equations),
//!     variables, vec![], None, boundary_conditions,
//!     "x".to_string(), 1e-6, 1000, initial_guess
//! );
//! 
//! solver.solve();
//! solver.plot_result();
//! ```
//!
//! ## Function Overview
//! - `new()`: Creates a new BVP solver instance with mesh and boundary conditions
//! - `set_additional_parameters()`: Configures Jacobian usage and variable bounds
//! - `solve()`: Main solving function with logging support
//! - `eq_generate()`: Converts symbolic expressions to numerical functions
//! - `BC_closure_creater()`: Creates boundary condition functions from HashMap
//! - `plot_result()`, `save_to_file()`: Result visualization and export
//!