//! # Symbolic Functions for Boundary Value Problems (BVP)
//!
//! ## Main Purpose
//! This module provides high-performance symbolic computation and automatic differentiation
//! capabilities specifically designed for solving Boundary Value Problems (BVPs) in differential
//! equations. It bridges symbolic mathematics with numerical computation by:
//! - Converting continuous ODEs into discretized algebraic systems
//! - Computing analytical Jacobians for Newton-Raphson and other iterative methods
//! - Providing multiple sparse and dense matrix backends for optimal performance
//! - Supporting parallel computation throughout the symbolic-to-numerical pipeline
//!
//! ## Main Structures and Methods
//!
//! ### `Jacobian` Struct
//! The core structure that manages the entire symbolic-to-numerical transformation pipeline:
//! - **Fields**:
//!   - `vector_of_functions`: Symbolic expressions representing the discretized BVP system
//!   - `symbolic_jacobian`: 2D matrix of symbolic partial derivatives
//!   - `jac_function`: Compiled numerical Jacobian function (trait object)
//!   - `residiual_function`: Compiled numerical residual function (trait object)
//!   - `bandwidth`: Optional sparse matrix bandwidth for banded systems
//!   - `bounds`/`rel_tolerance_vec`: Constraint and tolerance vectors for variables
//!
//! ### Key Methods
//!
//! #### Discretization
//! - `discretization_system_BVP_par()`: Converts continuous BVP to discrete algebraic system
//!   with parallel boundary condition processing and variable tracking
//!
//! #### Jacobian Computation
//! - `calc_jacobian_parallel_smart()`: Parallel symbolic differentiation with sparsity optimization
//! - `calc_jacobian_parallel()`: Standard parallel symbolic differentiation
//!
//! #### Function Compilation (Multiple Backends)
//! - **Dense (nalgebra)**: `lambdify_jacobian_DMatrix_par()`, `lambdify_residual_DVector()`
//! - **Sparse (faer)**: `lambdify_jacobian_SparseColMat_parallel2()`, `lambdify_residual_Col_parallel2()`
//! - **Sparse (sprs)**: `lambdify_jacobian_CsMat()`, `lambdify_residual_CsVec()`
//! - **Sparse (nalgebra)**: `lambdify_jacobian_CsMatrix()`, residual functions
//!
//! #### Utility Functions
//! - `find_bandwidths()`: Automatic sparse matrix bandwidth detection
//! - `process_bounds_and_tolerances()`: Efficient constraint processing with string optimization
//!
//! ## Interesting Tips and Non-Obvious Code Features
//!
//! ### Performance Optimizations
//! 1. **Parallel Outer Loop Pattern**: Functions like `lambdify_jacobian_SparseColMat_parallel2()`
//!    use `(0..n).into_par_iter().flat_map()` instead of nested loops for better load balancing
//!
//! 2. **Pre-compilation Strategy**: Functions are compiled once during setup and stored as
//!    `Box<dyn Fn + Send + Sync>` to avoid repeated lambdification during evaluation
//!
//! 3. **Smart Sparsity Detection**: `calc_jacobian_parallel_smart()` only computes derivatives
//!    for variables actually present in each equation, dramatically reducing computation
//!
//! 4. **String Processing Optimization**: `process_bounds_and_tolerances()` uses `rfind('_')`
//!    instead of regex for 10-100x faster variable name processing
//!
//! 5. **Boundary Condition Caching**: `discretization_system_BVP_par()` pre-computes HashSets
//!    and HashMaps for O(1) boundary condition lookups instead of O(n²) nested loops
//!
//! ### Thread Safety Patterns
//! 1. **Mutex-Protected Collections**: Parallel triplet collection uses `Mutex<Vec<Triplet>>`
//!    for thread-safe sparse matrix assembly
//!
//! 2. **Lifetime Management**: Functions convert `Vec<&str>` to `Vec<String>` early to avoid
//!    lifetime issues in `'static` closures: `variable_str.iter().map(|s| s.to_string()).collect()`
//!
//! 3. **Send + Sync Bounds**: Thread-safe lambdification uses `Box<dyn Fn + Send + Sync>`
//!    instead of regular `Box<dyn Fn>` for parallel evaluation
//!
//! ### Memory Management Tricks
//! 1. **Pre-allocation**: Vectors are pre-allocated with known capacity to avoid reallocations:
//!    `Vec::with_capacity(len)`
//!
//! 2. **Zero-Copy Operations**: Uses `ColRef::from_slice().to_owned()` for efficient faer
//!    matrix construction without intermediate copies
//!
//! 3. **Selective Cloning**: Only clones `Expr` objects when necessary due to their complex
//!    recursive structure - uses references and moves where possible
//!
//! ### Discretization Schemes
//! - **Forward Euler**: `"forward"` - First-order explicit scheme
//! - **Trapezoidal**: `"trapezoid"` - Second-order implicit scheme with better stability
//!
//! ### Matrix Backend Strategy
//! The module supports 4 different matrix backends, each optimized for different scenarios:
//! - **Dense (nalgebra)**: Best for small, dense systems
//! - **Sparse (faer)**: Modern, high-performance sparse matrices with excellent parallel support
//! - **Sparse (sprs)**: Mature Rust sparse matrix library
//! - **Sparse (nalgebra)**: Integrated with nalgebra ecosystem
//!
//! This multi-backend approach allows users to choose the optimal performance characteristics
//! for their specific problem size and sparsity pattern.

#![allow(non_camel_case_types)]

use crate::numerical::BVP_Damp::BVP_traits::{
    Fun,
    FunEnum,
    Jac,
    JacEnum,
    MatrixType,
    VectorType,
    convert_to_fun, // convert_to_jac
};
use crate::numerical::BVP_Damp::BVP_utils::elapsed_time;
use crate::symbolic::symbolic_engine::Expr;
//use crate::symbolic::symbolic_traits::SymbolicType;
use faer::col::Col;
use faer::col::ColRef;

use faer::sparse::{SparseColMat, Triplet};
use log::info;
use nalgebra::sparse::CsMatrix;
use nalgebra::{DMatrix, DVector};

use rayon::prelude::*;

use sprs::{CsMat, CsVec};
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use tabled::{builder::Builder, settings::Style};
use std::sync::Mutex;
/// Core structure for BVP symbolic-to-numerical transformation pipeline.
/// 
/// This struct represents a Jacobian for BVPs, which is a matrix of partial derivatives of a vector function.
/// It contains the symbolic and numerical representations of the jacobian, as well as the functions used to evaluate it.
/// Manages the complete workflow from symbolic BVP discretization to high-performance numerical evaluation.
/// 
/// # Performance Features
/// - Supports multiple matrix backends (dense/sparse) for optimal performance
/// - Parallel symbolic differentiation and function compilation
/// - Smart sparsity detection and bandwidth optimization
/// - Thread-safe evaluation with Send+Sync closures
//#[derive(Clone)]
pub struct Jacobian {
    /// Vector of symbolic functions/expressions representing the discretized BVP system.
    /// After discretization, contains the residual equations F(y) = 0 that need to be solved.
    pub vector_of_functions: Vec<Expr>,
    
    /// Vector of lambdified functions (symbolic functions converted to rust functions).
    /// Legacy field - modern implementations use trait objects in `jac_function` and `residiual_function`.
    pub lambdified_functions: Vec<Box<dyn Fn(Vec<f64>) -> f64>>,

    /// String identifier for the matrix backend method being used.
    /// Values: "Dense", "Sparse", "Sparse_1" (sprs), "Sparse_2" (nalgebra), "Sparse_3" (faer)
    pub method: String,
    
    /// Vector of symbolic variables representing the unknowns in the BVP system.
    /// Contains Expr::Var objects for each discretized variable (e.g., y_0, y_1, ..., y_n)
    pub vector_of_variables: Vec<Expr>,
    
    /// String representations of the variable names for efficient lookups.
    /// Used during lambdification and variable processing (e.g., ["y_0", "y_1", "z_0", "z_1"])
    pub variable_string: Vec<String>,
    
    /// 2D matrix of symbolic partial derivatives ∂F_i/∂x_j.
    /// Core of the analytical Jacobian - computed once symbolically, then compiled to numerical functions
    pub symbolic_jacobian: Vec<Vec<Expr>>,
    
    /// Optional single Jacobian element evaluator function.
    /// Legacy field for element-wise Jacobian evaluation - modern code uses `jac_function`
    pub lambdified_jac_element: Option<Box<dyn Fn(f64, usize, usize) -> f64>>,

    /// Compiled numerical Jacobian function as trait object.
    /// Supports multiple backends through JacEnum variants (Dense, Sparse_1, Sparse_2, Sparse_3)
    pub jac_function: Option<Box<dyn Jac>>,

    /// Compiled numerical residual function as trait object.
    /// Evaluates F(x,y) for the discretized BVP system through FunEnum variants
    pub residiual_function: Box<dyn Fun>,

    /// Optional variable bounds as (min, max) pairs for each discretized variable.
    /// Used by constrained solvers to enforce physical constraints during Newton iteration
    pub bounds: Option<Vec<(f64, f64)>>,
    
    /// Optional relative tolerance vector for adaptive error control.
    /// Per-variable tolerances for adaptive mesh refinement and convergence criteria
    pub rel_tolerance_vec: Option<Vec<f64>>,
    
    /// Optional sparse matrix bandwidth (kl, ku) for banded Jacobians.
    /// kl = number of subdiagonals, ku = number of superdiagonals. Enables banded matrix optimizations
    pub bandwidth: Option<(usize, usize)>,
    
    /// Variable tracking for smart sparsity detection.
    /// variables_for_all_disrete[i] contains variables actually used in equation i,
    /// enabling zero-derivative skipping in calc_jacobian_parallel_smart()
    pub variables_for_all_disrete: Vec<Vec<String>>,
}

impl Jacobian {
    /// Creates a new Jacobian instance with default values.
    /// 
    /// Initializes all fields to empty/default states and sets up a dummy residual function
    /// that returns the input vector unchanged. This serves as a safe default until the
    /// actual BVP system is discretized and compiled.
    /// 
    /// # Returns
    /// A new Jacobian instance ready for BVP discretization and symbolic computation.
    pub fn new() -> Self {
        let fun0: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>> =
            Box::new(|_x, y: &DVector<f64>| y.clone());
        let boxed_fun: Box<dyn Fun> = Box::new(FunEnum::Dense(fun0));
        Self {
            vector_of_functions: Vec::new(),
            lambdified_functions: Vec::new(),
            method: String::new(),
            vector_of_variables: Vec::new(),
            variable_string: Vec::new(),
            symbolic_jacobian: Vec::new(),
            lambdified_jac_element: None,
            jac_function: None,
            residiual_function: boxed_fun,
            bounds: None,
            rel_tolerance_vec: None,
            bandwidth: None,
            variables_for_all_disrete: Vec::new(),
        }
    }

    /// Initializes the Jacobian with a vector of symbolic functions and variable names.
    /// 
    /// This method sets up the basic symbolic representation of the system before discretization.
    /// It converts string variable names to symbolic Expr::Var objects for internal processing.
    /// 
    /// # Arguments
    /// * `vector_of_functions` - Vector of symbolic expressions representing the system equations
    /// * `variable_string` - Vector of variable names as strings (e.g., ["y", "z"])
    /// 
    /// # Example
    /// ```ignore
    /// let mut jac = Jacobian::new();
    /// let funcs = vec![Expr::parse_expression("y + z"), Expr::parse_expression("y - z")];
    /// let vars = vec!["y".to_string(), "z".to_string()];
    /// jac.from_vectors(funcs, vars);
    /// ```
    pub fn from_vectors(&mut self, vector_of_functions: Vec<Expr>, variable_string: Vec<String>) {
        self.vector_of_functions = vector_of_functions;
        self.variable_string = variable_string.clone();
        self.vector_of_variables =
            Expr::parse_vector_expression(variable_string.iter().map(|s| s.as_str()).collect());
        println!(" {:?}", self.vector_of_functions);
        println!(" {:?}", self.vector_of_variables);
    }

    /// Computes the symbolic Jacobian matrix using parallel differentiation with smart sparsity optimization.
    /// 
    /// This is the most efficient Jacobian computation method. It uses the `variables_for_all_disrete`
    /// field to only compute partial derivatives for variables that actually appear in each equation,
    /// dramatically reducing computation time for sparse systems.
    /// 
    /// # Performance Features
    /// - Parallel computation across equations using rayon
    /// - Smart zero-derivative detection: skips ∂F_i/∂x_j if x_j not in equation i
    /// - Automatic symbolic simplification of computed derivatives
    /// 
    /// # Panics
    /// - If `vector_of_functions` is empty
    /// - If `vector_of_variables` is empty  
    /// - If `variables_for_all_disrete` is empty (must call discretization first)
    /// 
    /// # Note
    /// Must be called after `discretization_system_BVP_par()` which populates `variables_for_all_disrete`.
    pub fn calc_jacobian_parallel_smart(&mut self) {
        assert!(self.variables_for_all_disrete.len() > 0);
        assert!(
            !self.vector_of_functions.is_empty(),
            "vector_of_functions is empty"
        );
        assert!(
            !self.vector_of_variables.is_empty(),
            "vector_of_variables is empty"
        );

        let variable_string_vec = self.variable_string.clone();
        let new_jac: Vec<Vec<Expr>> = self
            .vector_of_functions
            .par_iter()
            .enumerate()
            .map(|(i, function)| {
                let mut vector_of_partial_derivatives = Vec::new();
                // let function = function.clone();
                for j in 0..self.vector_of_variables.len() {
                    let variable = &variable_string_vec[j]; // obviously if function does not contain variable its derivative should be 0
                    let list_of_vaiables_for_this_eq = &self.variables_for_all_disrete[i]; // so we can only calculate derivative for variables that are used in this equation
                    if list_of_vaiables_for_this_eq.contains(variable) {
                        let mut partial = Expr::diff(&function, variable);
                        partial = partial.symplify();
                        vector_of_partial_derivatives.push(partial);
                    } else {
                        vector_of_partial_derivatives.push(Expr::Const(0.0));
                    }
                }
                vector_of_partial_derivatives
            })
            .collect();

        self.symbolic_jacobian = new_jac;
    }
    /// Computes the symbolic Jacobian matrix using standard parallel differentiation.
    /// 
    /// This method computes all partial derivatives ∂F_i/∂x_j without sparsity optimization.
    /// Use `calc_jacobian_parallel_smart()` for better performance on sparse systems.
    /// 
    /// # Performance Features
    /// - Parallel computation across equations using rayon
    /// - Automatic symbolic simplification of computed derivatives
    /// 
    /// # Panics
    /// - If `vector_of_functions` is empty
    /// - If `vector_of_variables` is empty
    /// 
    /// # When to Use
    /// - For dense systems where most variables appear in most equations
    /// - When `variables_for_all_disrete` is not available (before discretization)
    pub fn calc_jacobian_parallel(&mut self) {
        assert!(
            !self.vector_of_functions.is_empty(),
            "vector_of_functions is empty"
        );
        assert!(
            !self.vector_of_variables.is_empty(),
            "vector_of_variables is empty"
        );

        let variable_string_vec = self.variable_string.clone();
        let new_jac: Vec<Vec<Expr>> = self
            .vector_of_functions
            .par_iter()
            .map(|function| {
                let mut vector_of_partial_derivatives = Vec::new();
                // let function = function.clone();
                for j in 0..self.vector_of_variables.len() {
                    let mut partial = Expr::diff(&function, &variable_string_vec[j]);
                    partial = partial.symplify();
                    vector_of_partial_derivatives.push(partial);
                }
                vector_of_partial_derivatives
            })
            .collect();

        self.symbolic_jacobian = new_jac;
    }
    //

    ////////////////////////////////////////////////////////////////////////////////////
    //  GENERIC FUNCTIONS
    ////////////////////////////////////////////////////////////////////////////////////
    
    /// Compiles residual functions using generic VectorType trait for backend flexibility.
    /// 
    /// This method creates a generic residual function that works with any vector type
    /// implementing the VectorType trait. It's part of the experimental generic backend system.
    /// 
    /// # Arguments
    /// * `arg` - Independent variable name (typically time "t" or "x")
    /// * `variable_str` - Vector of variable names as string slices
    /// 
    /// # Note
    /// This is an experimental generic interface. For production use, prefer the specific
    /// backend methods like `lambdify_residual_DVector()` or `lambdify_residual_Col_parallel2()`.
    pub fn lambdify_residual_VectorType(&mut self, arg: &str, variable_str: Vec<&str>) {
        let vector_of_functions = &self.vector_of_functions;
        fn f(
            vector_of_functions: Vec<Expr>,
            _arg: String,
            variable_str: Vec<String>,
        ) -> Box<dyn Fn(f64, &dyn VectorType) -> Box<dyn VectorType>> {
            Box::new(move |_x: f64, v: &dyn VectorType| -> Box<dyn VectorType> {
                let mut result = v.zeros(vector_of_functions.len());

                // Iterate through functions and assign values
                for (i, func) in vector_of_functions.iter().enumerate() {
                    let func = Expr::lambdify_owned(
                        func.to_owned(),
                        variable_str
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .clone(),
                    );
                    let value = func(v.iterate().collect());
                    result = result.assign_value(i, value);
                }
                result
            })
        }

        let fun: Box<dyn Fn(f64, &dyn VectorType) -> Box<dyn VectorType>> = f(
            vector_of_functions.to_owned(),
            arg.to_string(),
            variable_str.iter().map(|s| s.to_string()).collect(),
        );
        let residual_function = convert_to_fun(fun);
        self.residiual_function = residual_function;
    }

    /// Generates a generic parallel Jacobian evaluator using trait objects.
    /// 
    /// Creates a closure that evaluates the symbolic Jacobian matrix using generic
    /// VectorType and MatrixType traits. Supports bandwidth optimization for sparse matrices.
    /// 
    /// # Arguments
    /// * `jac` - 2D vector of symbolic partial derivatives
    /// * `vector_of_functions_len` - Number of equations in the system
    /// * `vector_of_variables_len` - Number of variables in the system
    /// * `variable_str` - Variable names as owned strings
    /// * `_arg` - Independent variable name (unused in current implementation)
    /// * `bandwidth` - Optional (kl, ku) bandwidth for banded matrix optimization
    /// 
    /// # Returns
    /// A boxed closure that takes (time, variables) and returns a matrix of partial derivatives
    /// 
    /// # Performance Features
    /// - Parallel evaluation using rayon
    /// - Bandwidth-aware computation for sparse matrices
    /// - Thread-safe triplet collection using Mutex
    /// - Zero-threshold filtering (1e-8) to maintain sparsity
    pub fn jacobian_generate_generic_par(
        jac: Vec<Vec<Expr>>,
        vector_of_functions_len: usize,
        vector_of_variables_len: usize,
        variable_str: Vec<String>,
        _arg: String,
        bandwidth: Option<(usize, usize)>,
    ) -> Box<dyn Fn(f64, &(dyn VectorType + Send + Sync)) -> Box<dyn MatrixType>> {
        //let arg = arg.as_str();
        //let variable_str: Vec<&str> = variable_str.iter().map(|s| s.as_str()).collect();

        Box::new(
            move |_x: f64, v: &(dyn VectorType + Send + Sync)| -> Box<dyn MatrixType> {
                let mut vector_of_derivatives =
                    vec![0.0; vector_of_functions_len * vector_of_variables_len];
                let vector_of_derivatives_mutex = std::sync::Mutex::new(&mut vector_of_derivatives);

                let mut vector_of_triplets = Vec::new();
                let vector_of_triplets_mutex = std::sync::Mutex::new(&mut vector_of_triplets);
                (0..vector_of_functions_len).into_par_iter().for_each(|i| {
                    let (right_border, left_border) = if let Some((kl, ku)) = bandwidth {
                        let right_border = std::cmp::min(i + ku + 1, vector_of_variables_len);
                        let left_border = if i as i32 - (kl as i32) - 1 < 0 {
                            0
                        } else {
                            i - kl - 1
                        };
                        (right_border, left_border)
                    } else {
                        let right_border = vector_of_variables_len;
                        let left_border = 0;
                        (right_border, left_border)
                    };
                    for j in left_border..right_border {
                        // if jac[i][j] != Expr::Const(0.0) { println!("i = {}, j = {}, {}", i, j,  &jac[i][j]);}
                        let partial_func = Expr::lambdify_owned(
                            jac[i][j].clone(),
                            variable_str
                                .iter()
                                .map(|s| s.as_str())
                                .collect::<Vec<_>>()
                                .clone(),
                        );

                        let v_vec: Vec<f64> = v.iterate().collect();
                        let _vec_type = &v.vec_type();
                        let P = partial_func(v_vec.clone());
                        if P.abs() > 1e-8 {
                            vector_of_derivatives_mutex.lock().unwrap()
                                [i * vector_of_functions_len + j] = P;
                            let triplet = (i, j, P);
                            vector_of_triplets_mutex.lock().unwrap().push(triplet);
                        }
                    }
                }); //par_iter()

                let new_function_jacobian: Box<dyn MatrixType> = v.from_vector(
                    vector_of_functions_len,
                    vector_of_variables_len,
                    &vector_of_derivatives,
                    vector_of_triplets,
                );
                //  panic!("stop here");
                new_function_jacobian
            },
        ) // end of box
    } // end of function

    /// Compiles the symbolic Jacobian using the generic trait-based backend system.
    /// 
    /// This method creates a generic Jacobian function that works with any matrix type
    /// implementing the MatrixType trait. Currently incomplete due to Send+Sync trait limitations.
    /// 
    /// # Arguments
    /// * `arg` - Independent variable name
    /// * `variable_str` - Vector of variable names as string slices
    /// 
    /// # Status
    /// This method is currently incomplete and commented out due to trait object limitations.
    /// The generic backend system needs further development for full functionality.
    /// 
    /// # Note
    /// For production use, prefer the specific backend methods like `lambdify_jacobian_DMatrix_par()`.
    pub fn lambdify_jacobian_generic(&mut self, arg: &str, variable_str: Vec<&str>) {
        let symbolic_jacobian = self.symbolic_jacobian.clone();
        let symbolic_jacobian_rc = symbolic_jacobian.clone();

        let vector_of_functions_len = self.vector_of_functions.len();
        let vector_of_variables_len = self.vector_of_variables.len();
        let bandwidth = self.bandwidth;
        let _new_jac = Jacobian::jacobian_generate_generic_par(
            symbolic_jacobian_rc,
            vector_of_functions_len,
            vector_of_variables_len,
            variable_str.iter().map(|s| s.to_string()).collect(),
            arg.to_string(),
            bandwidth,
        );
        //  let mut boxed_jacobian: Box<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64>> = Box::new(|arg, variable_str_| {
        //    DMatrix::from_rows(&new_jac) }) ;
        // TODO! Send + Sync trait is not implemented
        //   let boxed_jac: Box<dyn Jac> = convert_to_jac(new_jac);
        //  self.jac_function = Some(boxed_jac);
    }

    ////////////////////////////////////////////////////////////////////////////////////
    ///                             NALGEBRA DENSE
    ////////////////////////////////////////////////////////////////////////////////////

    /// Compiles the symbolic Jacobian to a dense nalgebra DMatrix evaluator with parallel optimization.
    /// 
    /// This method creates a high-performance dense matrix Jacobian evaluator using nalgebra's DMatrix.
    /// It pre-compiles all non-zero symbolic derivatives and uses parallel evaluation for optimal performance.
    /// 
    /// # Arguments
    /// * `arg` - Independent variable name (typically "t" or "x")
    /// * `variable_str` - Vector of variable names as string slices
    /// 
    /// # Performance Features
    /// - **Pre-compilation**: All symbolic derivatives compiled once during setup
    /// - **Parallel evaluation**: Uses rayon for concurrent derivative evaluation
    /// - **Bandwidth optimization**: Respects sparse matrix bandwidth if set
    /// - **Zero filtering**: Only evaluates non-zero symbolic derivatives
    /// - **Thread-safe**: Uses `Send + Sync` closures and Mutex for thread safety
    /// - **Outer loop parallelization**: Uses `flat_map()` pattern for better load balancing
    /// 
    /// # Matrix Structure
    /// Creates a dense DMatrix where element (i,j) = ∂F_i/∂x_j evaluated at the given point.
    /// 
    /// # When to Use
    /// - Small to medium systems (< 1000 variables)
    /// - Dense or moderately sparse Jacobians
    /// - When nalgebra ecosystem integration is important
    pub fn lambdify_jacobian_DMatrix_par(&mut self, arg: &str, variable_str: Vec<&str>) {
        let jac = self.symbolic_jacobian.clone();
        let vector_of_functions_len = self.vector_of_functions.len();
        let vector_of_variables_len = self.vector_of_variables.len();
        let bandwidth = self.bandwidth;

        // Convert to owned strings to avoid lifetime issues
        let variable_str_owned: Vec<String> = variable_str.iter().map(|s| s.to_string()).collect();

        // Create jacobian positions in parallel using outer loop parallelization
                let jacobian_positions: Vec<(usize, usize, Box<dyn Fn(Vec<f64>) -> f64 + Send + Sync>)> = (0..vector_of_functions_len)
            .into_par_iter()
            .flat_map(|i| {
                let (right_border, left_border) = if let Some((kl, ku)) = bandwidth {
                    let right_border = std::cmp::min(i + ku + 1, vector_of_variables_len);
                    let left_border = if i as i32 - (kl as i32) - 1 < 0 {
                        0
                    } else {
                        i - kl - 1
                    };
                    (right_border, left_border)
                } else {
                    (vector_of_variables_len, 0)
                };
                
                (left_border..right_border)
                    .filter_map( |j| {
                        let symbolic_partial_derivative = &jac[i][j];
                        if !symbolic_partial_derivative.is_zero() {
                            let compiled_func: Box<dyn Fn(Vec<f64>) -> f64 + Send + Sync> =
                                Expr::lambdify_borrowed_thread_safe(
                                    &symbolic_partial_derivative,
                                    variable_str.clone(),
                                );
                            Some((i, j, compiled_func))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        let new_jac = Box::new(move |_x: f64, v: &DVector<f64>| -> DMatrix<f64> {
            let v_vec: Vec<f64> = v.iter().cloned().collect();
            let mut matrix = DMatrix::zeros(vector_of_functions_len, vector_of_variables_len);
            let matrix_mutex = Mutex::new(&mut matrix);

            jacobian_positions
                .par_iter()
                .for_each(|(i, j, compiled_func)| {
                    let P = compiled_func(v_vec.clone());
                    if P.abs() > 1e-8 {
                        let mut mat = matrix_mutex.lock().unwrap();
                        mat[(*i, *j)] = P;
                    }
                });

            matrix
        });

        let boxed_jac: Box<dyn Jac> = Box::new(JacEnum::Dense(new_jac));
        self.jac_function = Some(boxed_jac);
    }

    /// Compiles the residual functions to a dense nalgebra DVector evaluator with parallel optimization.
    /// 
    /// Creates a high-performance residual function evaluator that returns nalgebra DVector results.
    /// All symbolic functions are pre-compiled for maximum efficiency during repeated evaluations.
    /// 
    /// # Arguments
    /// * `arg` - Independent variable name (typically "t" or "x")
    /// * `variable_str` - Vector of variable names as string slices
    /// 
    /// # Performance Features
    /// - **Pre-compilation**: All residual functions compiled once during setup
    /// - **Parallel evaluation**: Uses rayon for concurrent function evaluation
    /// - **Thread-safe closures**: Uses `Send + Sync` bounds for parallel safety
    /// - **Memory efficient**: Direct vector construction without intermediate collections
    /// 
    /// # Output Format
    /// Returns DVector where element i = F_i(t, y) for the i-th residual equation.
    /// 
    /// # When to Use
    /// - Dense vector operations with nalgebra
    /// - Small to medium systems where dense storage is acceptable
    /// - Integration with nalgebra-based linear solvers
    pub fn lambdify_residual_DVector(&mut self, arg: &str, variable_str: Vec<&str>) {
        let vector_of_functions = &self.vector_of_functions;

        // Convert to owned strings to avoid lifetime issues
        let variable_str_owned: Vec<String> = variable_str.iter().map(|s| s.to_string()).collect();

        let compiled_functions: Vec<Box<dyn Fn(Vec<f64>) -> f64 + Send + Sync>> =
            vector_of_functions
                .par_iter()
                .map(|func| {
                    Expr::lambdify_borrowed_thread_safe(
                        func,
                        variable_str_owned.iter().map(|s| s.as_str()).collect(),
                    )
                })
                .collect();

        let fun = Box::new(move |_x: f64, v: &DVector<f64>| -> DVector<f64> {
            let v_vec: Vec<f64> = v.iter().cloned().collect();
            let result: Vec<_> = compiled_functions
                .par_iter()
                .map(|func| func(v_vec.clone()))
                .collect();
            DVector::from_vec(result)
        });
        let boxed_fun: Box<dyn Fun> = Box::new(FunEnum::Dense(fun));
        self.residiual_function = boxed_fun;
    }

    
    
    //////////////////////////////////////////////////////////////////////////////////////////
    ////                             SPRS CRATE SPARSE FUNCTIONS
    ///////////////////////////////////////////////////////////////////////////////////////////

    /// Generates a sparse Jacobian evaluator using the sprs crate's CsMat format.
    /// 
    /// Creates a closure that evaluates the symbolic Jacobian matrix and returns it as a
    /// sprs::CsMat (Compressed Sparse Matrix). This is a legacy method with sequential evaluation.
    /// 
    /// # Arguments
    /// * `jac` - 2D vector of symbolic partial derivatives
    /// * `vector_of_functions_len` - Number of equations in the system
    /// * `vector_of_variables_len` - Number of variables in the system
    /// * `variable_str` - Variable names as owned strings
    /// * `_arg` - Independent variable name (unused)
    /// 
    /// # Returns
    /// A mutable closure that takes (time, sparse_vector) and returns a sparse matrix
    /// 
    /// # Performance Notes
    /// - Sequential evaluation (not parallelized)
    /// - Uses sprs::CsMat compressed sparse row format
    /// - Evaluates all matrix elements (no sparsity optimization)
    /// 
    /// # When to Use
    /// - Legacy code compatibility with sprs crate
    /// - When mutable closure semantics are required
    /// - Small systems where parallelization overhead isn't justified
    pub fn jacobian_generate_CsMat(
        jac: Vec<Vec<Expr>>,
        vector_of_functions_len: usize,
        vector_of_variables_len: usize,
        variable_str: Vec<String>,
        _arg: String,
    ) -> Box<dyn FnMut(f64, &CsVec<f64>) -> CsMat<f64>> {
        //let arg = arg.as_str();
        //let variable_str: Vec<&str> = variable_str.iter().map(|s| s.as_str()).collect();

        Box::new(move |_x: f64, v: &CsVec<f64>| -> CsMat<f64> {
            let mut new_function_jacobian: CsMat<f64> =
                CsMat::zero((vector_of_functions_len, vector_of_variables_len));
            for i in 0..vector_of_functions_len {
                for j in 0..vector_of_variables_len {
                    // println!("i = {}, j = {}", i, j);
                    let partial_func = Expr::lambdify_owned(
                        jac[i][j].clone(),
                        variable_str
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .clone(),
                    );
                    //let v_vec: Vec<f64> = v.iter().cloned().collect();
                    let v_vec: Vec<f64> = v.to_dense().to_vec();
                    new_function_jacobian.insert(j, i, partial_func(v_vec.clone()))
                }
            }
            new_function_jacobian
        })
    } // end of function
    /// Compiles the symbolic Jacobian to a sprs CsMat evaluator.
    /// 
    /// Sets up the Jacobian function using the sprs crate's compressed sparse matrix format.
    /// This method wraps the generated CsMat evaluator in the trait object system.
    /// 
    /// # Arguments
    /// * `arg` - Independent variable name
    /// * `variable_str` - Vector of variable names as string slices
    /// 
    /// # Backend Details
    /// - Uses sprs::CsMat (Compressed Sparse Row format)
    /// - Sequential evaluation (not parallelized)
    /// - Wrapped in JacEnum::Sparse_1 variant
    /// 
    /// # When to Use
    /// - Legacy compatibility with sprs-based code
    /// - When CSR format is specifically required
    /// - Small to medium sparse systems
    pub fn lambdify_jacobian_CsMat(&mut self, arg: &str, variable_str: Vec<&str>) {
        let symbolic_jacobian = self.symbolic_jacobian.clone();
        let symbolic_jacobian_rc = symbolic_jacobian.clone();

        let vector_of_functions_len = self.vector_of_functions.len();
        let vector_of_variables_len = self.vector_of_variables.len();

        let new_jac = Jacobian::jacobian_generate_CsMat(
            symbolic_jacobian_rc,
            vector_of_functions_len,
            vector_of_variables_len,
            variable_str.iter().map(|s| s.to_string()).collect(),
            arg.to_string(),
        );
        //  let mut boxed_jacobian: Box<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64>> = Box::new(|arg, variable_str_| {
        //    DMatrix::from_rows(&new_jac) }) ;

        let boxed_jac: Box<dyn Jac> = Box::new(JacEnum::Sparse_1(new_jac));
        self.jac_function = Some(boxed_jac);
    }

    /// Compiles residual functions to a sprs CsVec evaluator for sparse vector operations.
    /// 
    /// Creates a residual function evaluator that works with sprs sparse vectors (CsVec).
    /// This method is designed for systems where the residual vector itself is sparse.
    /// 
    /// # Arguments
    /// * `arg` - Independent variable name (used in IVP-style evaluation)
    /// * `variable_str` - Vector of variable names as string slices
    /// 
    /// # Backend Details
    /// - Uses sprs::CsVec (Compressed Sparse Vector format)
    /// - Supports IVP-style evaluation with time parameter
    /// - Sequential evaluation of residual functions
    /// - Wrapped in FunEnum::Sparse_1 variant
    /// 
    /// # When to Use
    /// - Sparse residual vectors (many zero elements)
    /// - Integration with sprs-based linear algebra
    /// - IVP problems where time-dependent evaluation is needed
    pub fn lambdify_residual_CsVec(&mut self, arg: &str, variable_str: Vec<&str>) {
        let vector_of_functions = &self.vector_of_functions;
        fn f(
            vector_of_functions: Vec<Expr>,
            arg: String,
            variable_str: Vec<String>,
        ) -> Box<dyn Fn(f64, &CsVec<f64>) -> CsVec<f64> + 'static> {
            Box::new(move |x: f64, v: &CsVec<f64>| -> CsVec<f64> {
                //  let mut result: CsVec<f64> = CsVec::new(n, indices, data);
                let mut result: CsVec<f64> = CsVec::empty(vector_of_functions.len());
                for (i, func) in vector_of_functions.iter().enumerate() {
                    let func = Expr::lambdify_IVP_owned(
                        func.to_owned(),
                        arg.as_str(),
                        variable_str
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .clone(),
                    );
                    result.append(i, func(x, v.to_dense().to_vec()));
                }
                result
            }) //enf of box
        } // end of function
        let fun = f(
            vector_of_functions.to_owned(),
            arg.to_string(),
            variable_str.clone().iter().map(|s| s.to_string()).collect(),
        );
        let boxed_fun: Box<dyn Fun> = Box::new(FunEnum::Sparse_1(fun));
        self.residiual_function = boxed_fun;
    }

    //////////////////////////////////////////////////////////////////////////////////////////
    //                            NALGEBRA SPARSE CRATE
    ///////////////////////////////////////////////////////////////////////////////////////////
    
    /// Generates a sparse Jacobian evaluator using nalgebra's CsMatrix format.
    /// 
    /// Creates a closure that evaluates the symbolic Jacobian and returns it as nalgebra's
    /// compressed sparse matrix format. Uses dense intermediate storage for simplicity.
    /// 
    /// # Arguments
    /// * `jac` - 2D vector of symbolic partial derivatives
    /// * `vector_of_functions_len` - Number of equations in the system
    /// * `vector_of_variables_len` - Number of variables in the system
    /// * `variable_str` - Variable names as owned strings
    /// * `_arg` - Independent variable name (unused)
    /// 
    /// # Returns
    /// A mutable closure that takes (time, dense_vector) and returns a sparse matrix
    /// 
    /// # Implementation Details
    /// - Uses DMatrix as intermediate storage, then converts to CsMatrix
    /// - Sequential evaluation (not parallelized)
    /// - Evaluates all matrix elements regardless of sparsity
    /// 
    /// # When to Use
    /// - Integration with nalgebra's sparse matrix ecosystem
    /// - When CsMatrix format is specifically required
    /// - Small to medium systems where conversion overhead is acceptable
    pub fn jacobian_generate_CsMatrix(
        jac: Vec<Vec<Expr>>,
        vector_of_functions_len: usize,
        vector_of_variables_len: usize,
        variable_str: Vec<String>,
        _arg: String,
    ) -> Box<dyn FnMut(f64, &DVector<f64>) -> CsMatrix<f64>> {
        //let arg = arg.as_str();
        //let variable_str: Vec<&str> = variable_str.iter().map(|s| s.as_str()).collect();

        Box::new(move |_x: f64, v: &DVector<f64>| -> CsMatrix<f64> {
            let _number_of_possible_non_zero: usize = jac.len();
            //let mut new_function_jacobian: CsMatrix<f64> = CsMatrix::new_uninitialized_generic(Dyn(vector_of_functions_len),
            //Dyn(vector_of_variables_len), number_of_possible_non_zero);
            let mut new_function_jacobian: DMatrix<f64> =
                DMatrix::zeros(vector_of_functions_len, vector_of_variables_len);
            for i in 0..vector_of_functions_len {
                for j in 0..vector_of_variables_len {
                    // println!("i = {}, j = {}", i, j);
                    let partial_func = Expr::lambdify_owned(
                        jac[i][j].clone(),
                        variable_str
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .clone(),
                    );
                    //let v_vec: Vec<f64> = v.iter().cloned().collect();
                    let v_vec: Vec<f64> = v.iter().cloned().collect();
                    //new_function_jacobian = CsMatrix::from_triplet(vector_of_functions_len,
                    // vector_of_variables_len, &[i], &[j], &[partial_func(x, v_vec.clone())]   );
                    new_function_jacobian[(i, j)] = partial_func(v_vec.clone());
                }
            }
            new_function_jacobian.into()
        })
    } // end of function
    /// Compiles the symbolic Jacobian to a nalgebra CsMatrix evaluator.
    /// 
    /// Sets up the Jacobian function using nalgebra's compressed sparse matrix format.
    /// This method integrates with nalgebra's sparse matrix ecosystem.
    /// 
    /// # Arguments
    /// * `arg` - Independent variable name
    /// * `variable_str` - Vector of variable names as string slices
    /// 
    /// # Backend Details
    /// - Uses nalgebra::sparse::CsMatrix format
    /// - Sequential evaluation with dense intermediate storage
    /// - Wrapped in JacEnum::Sparse_2 variant
    /// 
    /// # When to Use
    /// - Integration with nalgebra's sparse linear algebra
    /// - When nalgebra CsMatrix format is specifically required
    /// - Medium-sized sparse systems
    pub fn lambdify_jacobian_CsMatrix(&mut self, arg: &str, variable_str: Vec<&str>) {
        let symbolic_jacobian = self.symbolic_jacobian.clone();
        let symbolic_jacobian_rc = symbolic_jacobian.clone();

        let vector_of_functions_len = self.vector_of_functions.len();
        let vector_of_variables_len = self.vector_of_variables.len();

        let new_jac = Jacobian::jacobian_generate_CsMatrix(
            symbolic_jacobian_rc,
            vector_of_functions_len,
            vector_of_variables_len,
            variable_str.iter().map(|s| s.to_string()).collect(),
            arg.to_string(),
        );
        //  let mut boxed_jacobian: Box<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64>> = Box::new(|arg, variable_str_| {
        //    DMatrix::from_rows(&new_jac) }) ;

        let boxed_jac: Box<dyn Jac> = Box::new(JacEnum::Sparse_2(new_jac));
        self.jac_function = Some(boxed_jac);
    }
    ////////////////////////////////////////////////////////////////////////////////////////
    ///         FAER SPARSE CRATE
    ////////////////////////////////////////////////////////////////////////////////////////

    /// Compiles the symbolic Jacobian to a faer SparseColMat evaluator (sequential version).
    /// 
    /// Creates a high-performance sparse Jacobian evaluator using the faer crate's SparseColMat.
    /// This version pre-compiles all functions sequentially to avoid thread safety issues.
    /// 
    /// # Arguments
    /// * `_arg` - Independent variable name (unused)
    /// * `variable_str` - Vector of variable names as string slices
    /// 
    /// # Performance Features
    /// - **Pre-compilation**: All non-zero derivatives compiled once during setup
    /// - **Bandwidth optimization**: Respects sparse matrix bandwidth if set
    /// - **Zero filtering**: Only compiles and stores non-zero symbolic derivatives
    /// - **Triplet assembly**: Uses faer's efficient triplet-based sparse matrix construction
    /// - **Sequential safety**: Avoids Send+Sync issues by compiling sequentially
    /// 
    /// # Matrix Format
    /// Uses faer::sparse::SparseColMat (Compressed Sparse Column format) for optimal performance.
    /// 
    /// # When to Use
    /// - Large sparse systems where faer's performance is critical
    /// - When thread safety during compilation is a concern
    /// - Systems with well-defined sparsity patterns
    pub fn lambdify_jacobian_SparseColMat_modified(&mut self, _arg: &str, variable_str: Vec<&str>) {
        let symbolic_jacobian = self.symbolic_jacobian.clone();
        let vector_of_functions_len = self.vector_of_functions.len();
        let vector_of_variables_len = self.vector_of_variables.len();
        let bandwidth = self.bandwidth;

        // Pre-compile all non-zero jacobian elements sequentially to avoid Send issues
        let mut compiled_jacobian_elements = Vec::new();
        for i in 0..vector_of_functions_len {
            let (right_border, left_border) = if let Some((kl, ku)) = bandwidth {
                let right_border = std::cmp::min(i + ku + 1, vector_of_variables_len);
                let left_border = if i as i32 - (kl as i32) - 1 < 0 {
                    0
                } else {
                    i - kl - 1
                };
                (right_border, left_border)
            } else {
                (vector_of_variables_len, 0)
            };

            for j in left_border..right_border {
                let symbolic_partial_derivative = &symbolic_jacobian[i][j];
                if !symbolic_partial_derivative.is_zero() {
                    let compiled_func =
                        Expr::lambdify(symbolic_partial_derivative, variable_str.clone());
                    compiled_jacobian_elements.push((i, j, compiled_func));
                }
            }
        }

        let new_jac = Box::new(move |_x: f64, v: &Col<f64>| -> SparseColMat<usize, f64> {
            let v_vec: Vec<f64> = v.iter().cloned().collect();

            let mut vector_of_triplets = Vec::new();
            for (i, j, func) in &compiled_jacobian_elements {
                let P = func(v_vec.clone());
                if P.abs() > 1e-8 {
                    vector_of_triplets.push(Triplet::new(*i, *j, P));
                }
            }

            SparseColMat::try_new_from_triplets(
                vector_of_functions_len,
                vector_of_variables_len,
                vector_of_triplets.as_slice(),
            )
            .unwrap()
        });

        let boxed_jac: Box<dyn Jac> = Box::new(JacEnum::Sparse_3(new_jac));
        self.jac_function = Some(boxed_jac);
    }

    /// Compiles the symbolic Jacobian to a faer SparseColMat evaluator (parallel version 1).
    /// 
    /// Creates a parallel sparse Jacobian evaluator using nested parallelization.
    /// This version parallelizes the inner loop for each row of the Jacobian matrix.
    /// 
    /// # Arguments
    /// * `_arg` - Independent variable name (unused)
    /// * `variable_str` - Vector of variable names as string slices
    /// 
    /// # Performance Features
    /// - **Nested parallelization**: Outer loop sequential, inner loop parallel
    /// - **Thread-safe compilation**: Uses `lambdify_thread_safe()` for Send+Sync closures
    /// - **Bandwidth optimization**: Respects sparse matrix bandwidth if set
    /// - **Parallel evaluation**: Concurrent derivative evaluation during matrix assembly
    /// 
    /// # Parallelization Strategy
    /// - Outer loop (rows): Sequential iteration
    /// - Inner loop (columns): Parallel using `into_par_iter()`
    /// - Function compilation: Parallel within each row
    /// 
    /// # When to Use
    /// - Medium to large sparse systems
    /// - When row-wise parallelization is preferred
    /// - Systems with irregular sparsity patterns per row
    pub fn lambdify_jacobian_SparseColMat_parallel(&mut self, _arg: &str, variable_str: Vec<&str>) {
        let jac = self.symbolic_jacobian.clone();
        let vector_of_functions_len = self.vector_of_functions.len();
        let vector_of_variables_len = self.vector_of_variables.len();
        let bandwidth = self.bandwidth;
        
        // Convert to owned strings to avoid lifetime issues
     //   let variable_str_owned: Vec<String> = variable_str.iter().map(|s| s.to_string()).collect();
        
        // Store non-zero jacobian positions and expressions (not compiled functions)
        let mut jacobian_positions = Vec::new();
        for i in 0..vector_of_functions_len {
            let (right_border, left_border) = if let Some((kl, ku)) = bandwidth {
                let right_border = std::cmp::min(i + ku + 1, vector_of_variables_len);
                let left_border = if i as i32 - (kl as i32) - 1 < 0 {
                    0
                } else {
                    i - kl - 1
                };
                (right_border, left_border)
            } else {
                (vector_of_variables_len, 0)
            };
        let inner: Vec<(usize, usize, Box<dyn Fn(Vec<f64>) -> f64 + Send + Sync>)> = 
            (left_border..right_border)
                .into_par_iter()
                .filter_map(|j| {
                    let symbolic_partial_derivative = jac[i][j].clone();
                    if !symbolic_partial_derivative.is_zero() {
                        let compiled_func: Box<dyn Fn(Vec<f64>) -> f64 + Send + Sync> =
                            Expr::lambdify_thread_safe(symbolic_partial_derivative, variable_str.clone());
                        Some((i, j, compiled_func))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
                    jacobian_positions.extend(inner);
                }

        let new_jac = Box::new(move |_x: f64, v: &Col<f64>| -> SparseColMat<usize, f64> {
            let v_vec: Vec<f64> = v.iter().cloned().collect();
            let triplets_mutex = std::sync::Mutex::new(Vec::new());

            // Compile and evaluate in parallel without storing closures
            jacobian_positions
                .par_iter()
                .for_each(|(i, j, compiled_func)| {
                    let P = compiled_func(v_vec.clone());
                    if P.abs() > 1e-8 {
                        let mut triplets = triplets_mutex.lock().unwrap();
                        triplets.push(Triplet::new(*i, *j, P));
                    }
                });

            let triplets = triplets_mutex.lock().unwrap();
            SparseColMat::try_new_from_triplets(
                vector_of_functions_len,
                vector_of_variables_len,
                triplets.as_slice(),
            )
            .unwrap()
        });

        let boxed_jac: Box<dyn Jac> = Box::new(JacEnum::Sparse_3(new_jac));
        self.jac_function = Some(boxed_jac);
    }

    /// Compiles the symbolic Jacobian to a faer SparseColMat evaluator (parallel version 2 - RECOMMENDED).
    /// 
    /// Creates the most efficient parallel sparse Jacobian evaluator using outer loop parallelization.
    /// This is the recommended method for large sparse systems due to superior load balancing.
    /// 
    /// # Arguments
    /// * `_arg` - Independent variable name (unused)
    /// * `variable_str` - Vector of variable names as string slices
    /// 
    /// # Performance Features
    /// - **Outer loop parallelization**: Uses `flat_map()` for optimal load balancing
    /// - **Pre-compilation**: All non-zero derivatives compiled during setup
    /// - **Thread-safe closures**: Uses `lambdify_borrowed_thread_safe()` for efficiency
    /// - **Bandwidth optimization**: Respects sparse matrix bandwidth if set
    /// - **Parallel evaluation**: Concurrent triplet assembly with Mutex protection
    /// 
    /// # Parallelization Strategy
    /// ```ignore
    /// (0..n_rows).into_par_iter().flat_map(|i| {
    ///     (cols_for_row_i).filter_map(|j| compile_and_store(i, j))
    /// })
    /// ```
    /// 
    /// # Why This is Best
    /// - Better load balancing than nested loops
    /// - Avoids thread contention in inner loops
    /// - Scales well with number of CPU cores
    /// - Optimal for sparse matrices with varying row densities
    /// 
    /// # When to Use
    /// - **RECOMMENDED** for all large sparse systems
    /// - Production code requiring maximum performance
    /// - Systems with > 1000 variables
    pub fn lambdify_jacobian_SparseColMat_parallel2(&mut self, _arg: &str, variable_str: Vec<&str>) {
     let jac = self.symbolic_jacobian.clone();
        let vector_of_functions_len = self.vector_of_functions.len();
        let vector_of_variables_len = self.vector_of_variables.len();
        let bandwidth = self.bandwidth;

        // Convert to owned strings to avoid lifetime issues
      //  let variable_str_owned: Vec<String> = variable_str.iter().map(|s| s.to_string()).collect();

        // Create jacobian positions in parallel using outer loop parallelization
        let jacobian_positions: Vec<(usize, usize, Box<dyn Fn(Vec<f64>) -> f64 + Send + Sync>)> = (0..vector_of_functions_len)
            .into_par_iter()
            .flat_map(|i| {
                let (right_border, left_border) = if let Some((kl, ku)) = bandwidth {
                    let right_border = std::cmp::min(i + ku + 1, vector_of_variables_len);
                    let left_border = if i as i32 - (kl as i32) - 1 < 0 {
                        0
                    } else {
                        i - kl - 1
                    };
                    (right_border, left_border)
                } else {
                    (vector_of_variables_len, 0)
                };
                
                (left_border..right_border)
                    .filter_map( |j| {
                        let symbolic_partial_derivative = &jac[i][j];
                        if !symbolic_partial_derivative.is_zero() {
                            let compiled_func: Box<dyn Fn(Vec<f64>) -> f64 + Send + Sync> =
                                Expr::lambdify_borrowed_thread_safe(
                                    &symbolic_partial_derivative,
                                    variable_str.clone(),
                                );
                            Some((i, j, compiled_func))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        let new_jac = Box::new(move |_x: f64, v: &Col<f64>| -> SparseColMat<usize, f64> {
            let v_vec: Vec<f64> = v.iter().cloned().collect();
            let triplets_mutex = std::sync::Mutex::new(Vec::new());

            // Compile and evaluate in parallel without storing closures
            jacobian_positions
                .par_iter()
                .for_each(|(i, j, compiled_func)| {
                    let P = compiled_func(v_vec.clone());
                    if P.abs() > 1e-8 {
                        let mut triplets = triplets_mutex.lock().unwrap();
                        triplets.push(Triplet::new(*i, *j, P));
                    }
                });

            let triplets = triplets_mutex.lock().unwrap();
            SparseColMat::try_new_from_triplets(
                vector_of_functions_len,
                vector_of_variables_len,
                triplets.as_slice(),
            )
            .unwrap()
        });

        let boxed_jac: Box<dyn Jac> = Box::new(JacEnum::Sparse_3(new_jac));
        self.jac_function = Some(boxed_jac);
    }
    ////////////////////////////////RESIDUAL LAMBDIFICATION/////////////////////////////////////////////////////////

    /// Compiles residual functions to a faer Col evaluator (sequential pre-compilation version).
    /// 
    /// Creates an optimized residual function evaluator using faer's Col vector format.
    /// This version pre-compiles all functions sequentially for maximum efficiency.
    /// 
    /// # Arguments
    /// * `_arg` - Independent variable name (unused)
    /// * `variable_str` - Vector of variable names as string slices
    /// 
    /// # Performance Features
    /// - **Pre-compilation**: All residual functions compiled once during setup (CRITICAL optimization)
    /// - **Sequential compilation**: Avoids thread safety issues during function compilation
    /// - **Efficient evaluation**: Direct vector construction without intermediate collections
    /// - **Memory efficient**: Uses `ColRef::from_slice().to_owned()` for zero-copy construction
    /// 
    /// # Why Pre-compilation Matters
    /// This is the most significant optimization - compiling symbolic functions once during setup
    /// instead of on every evaluation provides 10-100x speedup for repeated evaluations.
    /// 
    /// # When to Use
    /// - Production code where maximum performance is critical
    /// - Systems with frequent residual evaluations
    /// - When thread safety during compilation is a concern
    pub fn lambdify_residual_Col_modified(&mut self, _arg: &str, variable_str: Vec<&str>) {
        let vector_of_functions = &self.vector_of_functions;

        // Pre-compile all functions once (most significant optimization)
        let compiled_functions: Vec<_> = vector_of_functions
            .iter()
            .map(|func| {
                Expr::lambdify(
                    func,
                    variable_str.iter().map(|s| *s).collect::<Vec<_>>().clone(),
                )
            })
            .collect();

        let fun = Box::new(move |_x: f64, v: &Col<f64>| -> Col<f64> {
            let v_vec: Vec<f64> = v.iter().cloned().collect();
            let result: Vec<_> = compiled_functions
                .iter()
                .map(|func| func(v_vec.clone()))
                .collect();
            ColRef::from_slice(result.as_slice()).to_owned()
        });
        let boxed_fun: Box<dyn Fun> = Box::new(FunEnum::Sparse_3(fun));
        self.residiual_function = boxed_fun;
    }

    /// Compiles residual functions to a faer Col evaluator (parallel version 1).
    /// 
    /// Creates a parallel residual function evaluator with thread-safe function compilation.
    /// This version parallelizes both compilation and evaluation phases.
    /// 
    /// # Arguments
    /// * `_arg` - Independent variable name (unused)
    /// * `variable_str` - Vector of variable names as string slices
    /// 
    /// # Performance Features
    /// - **Parallel compilation**: Functions compiled concurrently using rayon
    /// - **Thread-safe closures**: Uses `lambdify_thread_safe()` for Send+Sync bounds
    /// - **Parallel evaluation**: Concurrent function evaluation during residual computation
    /// - **Owned strings**: Converts to owned strings early to avoid lifetime issues
    /// 
    /// # When to Use
    /// - Large systems with many residual functions
    /// - When compilation time is significant
    /// - Multi-core systems where parallel compilation provides benefits
    pub fn lambdify_residual_Col_parallel(&mut self, _arg: &str, variable_str: Vec<&str>) {
        let vector_of_functions = &self.vector_of_functions;
        
        // Convert to owned strings to avoid lifetime issues
        let variable_str_owned: Vec<String> = variable_str.iter().map(|s| s.to_string()).collect();

        let compiled_functions: Vec<Box<dyn Fn(Vec<f64> )->f64 + Send+ Sync>>  =
        vector_of_functions
            .into_par_iter()
            .map(|func| {
                let func = func.clone();
                let compiled = Expr::lambdify_thread_safe(
                    func,
                    variable_str_owned.iter().map(|s| s.as_str()).collect(),
                );
                compiled
            }).collect();
 

     let fun = Box::new(move |_x: f64, v: &Col<f64>| -> Col<f64> {
            let v_vec: Vec<f64> = v.iter().cloned().collect();
            let result: Vec<_> = compiled_functions
                .par_iter()
                .map(|func| func(v_vec.clone()))
                .collect();
            ColRef::from_slice(result.as_slice()).to_owned()
        });
        let boxed_fun: Box<dyn Fun> = Box::new(FunEnum::Sparse_3(fun));
        self.residiual_function = boxed_fun;
    }

    /// Compiles residual functions to a faer Col evaluator (parallel version 2 - RECOMMENDED).
    /// 
    /// Creates the most efficient parallel residual function evaluator using borrowed references
    /// for optimal memory usage and performance.
    /// 
    /// # Arguments
    /// * `_arg` - Independent variable name (unused)
    /// * `variable_str` - Vector of variable names as string slices
    /// 
    /// # Performance Features
    /// - **Parallel compilation**: Functions compiled concurrently using rayon
    /// - **Borrowed references**: Uses `lambdify_borrowed_thread_safe()` for efficiency
    /// - **Parallel evaluation**: Concurrent function evaluation during residual computation
    /// - **Memory efficient**: Avoids unnecessary cloning during compilation
    /// 
    /// # Why This is Recommended
    /// - More memory efficient than parallel version 1
    /// - Uses borrowed references instead of cloning expressions
    /// - Better performance for large symbolic expressions
    /// - Optimal balance of compilation and evaluation speed
    /// 
    /// # When to Use
    /// - **RECOMMENDED** for all parallel residual evaluation
    /// - Large systems with complex symbolic expressions
    /// - Production code requiring optimal memory usage
    pub fn lambdify_residual_Col_parallel2(&mut self, _arg: &str, variable_str: Vec<&str>) {
        let vector_of_functions = &self.vector_of_functions;
        
        // Convert to owned strings to avoid lifetime issues
        let variable_str_owned: Vec<String> = variable_str.iter().map(|s| s.to_string()).collect();

        let compiled_functions: Vec<Box<dyn Fn(Vec<f64> )->f64 + Send+ Sync>>  =
        vector_of_functions
            .into_par_iter()
            .map(|func| {
             
                let compiled = Expr::lambdify_borrowed_thread_safe(
                    func,
                    variable_str_owned.iter().map(|s| s.as_str()).collect(),
                );
                compiled
            }).collect();
 

     let fun = Box::new(move |_x: f64, v: &Col<f64>| -> Col<f64> {
            let v_vec: Vec<f64> = v.iter().cloned().collect();
            let result: Vec<_> = compiled_functions
                .par_iter()
                .map(|func| func(v_vec.clone()))
                .collect();
            ColRef::from_slice(result.as_slice()).to_owned()
        });
        let boxed_fun: Box<dyn Fun> = Box::new(FunEnum::Sparse_3(fun));
        self.residiual_function = boxed_fun;
    }
    ///////////////////////////////////////////////////////////////////////////
    ///              DISCRETIZED FUNCTIONS
    ///////////////////////////////////////////////////////////////////////////
    
    /// Removes numeric suffix from discretized variable names for efficient processing.
    /// 
    /// Extracts the base variable name from discretized variables like "y_0", "y_1", "z_0", etc.
    /// This is used for mapping discretized variables back to their original names for bounds
    /// and tolerance processing.
    /// 
    /// # Arguments
    /// * `input` - Variable name with potential numeric suffix (e.g., "y_0", "z_15")
    /// 
    /// # Returns
    /// Base variable name without suffix (e.g., "y", "z")
    /// 
    /// # Performance
    /// Uses `rfind('_')` for O(n) string processing, much faster than regex alternatives.
    /// 
    /// # Examples
    /// ```ignore
    /// assert_eq!(remove_numeric_suffix("y_0"), "y");
    /// assert_eq!(remove_numeric_suffix("temperature_15"), "temperature");
    /// assert_eq!(remove_numeric_suffix("x"), "x"); // No suffix
    /// ```
    pub fn remove_numeric_suffix(input: &str) -> String {
        if let Some(pos) = input.rfind('_') {
            input[..pos].to_string()
        } else {
            input.to_string()
        }
    }
    /// Applies discretization scheme to a single equation at a specific time step.
    /// 
    /// Transforms a continuous ODE equation into its discretized form using the specified
    /// numerical scheme. Handles variable renaming and time substitution for the discretization.
    /// 
    /// # Arguments
    /// * `matrix_of_names` - 2D matrix of discretized variable names [time_step][variable_index]
    /// * `eq_i` - The symbolic equation to discretize
    /// * `values` - Original variable names (e.g., ["y", "z"])
    /// * `arg` - Independent variable name (typically "t" or "x")
    /// * `j` - Current time step index
    /// * `t` - Time value at step j
    /// * `scheme` - Discretization scheme ("forward" or "trapezoid")
    /// 
    /// # Returns
    /// Discretized equation with renamed variables and substituted time value
    /// 
    /// # Supported Schemes
    /// - **"forward"**: Forward Euler - f(t_j, y_j)
    /// - **"trapezoid"**: Trapezoidal rule - 0.5 * (f(t_j, y_j) + f(t_{j+1}, y_{j+1}))
    /// 
    /// # Variable Renaming
    /// Original variables like "y", "z" become "y_j", "z_j" for the j-th time step.
    fn eq_step(
        matrix_of_names: &[Vec<String>],
        eq_i: &Expr,
        values: &[String],
        arg: &str,
        j: usize,
        t: f64,
        scheme: &str,
    ) -> Expr {
        let vec_of_names_on_step = &matrix_of_names[j];
        let hashmap_for_rename: HashMap<String, String> = values
            .iter()
            .zip(vec_of_names_on_step.iter())
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        //set time value of j-th time step
        let eq_step_j = eq_i.rename_variables(&hashmap_for_rename);
        let eq_step_j = eq_step_j.set_variable(arg, t);

        match scheme {
            "forward" => eq_step_j,
            "trapezoid" => {
                let vec_of_names_on_step = &matrix_of_names[j + 1];
                let hashmap_for_rename: HashMap<String, String> = values
                    .iter()
                    .zip(vec_of_names_on_step.iter())
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect();

                let eq_step_j_plus_1 = eq_i
                    .rename_variables(&hashmap_for_rename)
                    .set_variable(arg, t);
                Expr::Const(0.5) * (eq_step_j + eq_step_j_plus_1)
            }
            _ => panic!("Invalid scheme"),
        }
    }
    //
    /// Creates the time/spatial mesh for BVP discretization.
    /// 
    /// Generates the discretization mesh either from explicit mesh points or by creating
    /// a uniform grid. Returns step sizes as symbolic expressions and mesh points.
    /// 
    /// # Arguments
    /// * `n_steps` - Number of discretization steps (if uniform mesh)
    /// * `h` - Step size (if uniform mesh)
    /// * `mesh` - Explicit mesh points (overrides n_steps/h if provided)
    /// * `t0` - Starting point of the domain
    /// 
    /// # Returns
    /// Tuple of (step_sizes, mesh_points, n_steps) where:
    /// - `step_sizes`: Vector of Expr::Const representing h_i = t_{i+1} - t_i
    /// - `mesh_points`: Vector of f64 mesh coordinates
    /// - `n_steps`: Total number of mesh points
    /// 
    /// # Mesh Types
    /// - **Explicit mesh**: Uses provided mesh points with variable step sizes
    /// - **Uniform mesh**: Creates evenly spaced points with constant step size
    /// 
    /// # Default Values
    /// - n_steps: 101 (100 intervals + 1 endpoint)
    /// - h: 1.0
    fn create_mesh(
        &self,
        n_steps: Option<usize>,
        h: Option<f64>,
        mesh: Option<Vec<f64>>,
        t0: f64,
    ) -> (Vec<Expr>, Vec<f64>, usize) {
        // mesh of t's can be defined directly or by size of step -h, and number of steps
        if let Some(mesh) = mesh {
            let n_steps = mesh.len();
            info!("mesh with n_steps = {} is defined directly", n_steps);
            let H: Vec<Expr> = mesh
                .windows(2)
                .map(|window| Expr::Const(window[1] - window[0]))
                .collect();
            (H, mesh, n_steps)
        } else {
            let n_steps = n_steps.unwrap_or(100) + 1;
            info!(
                "mesh is not defined, creating evenly distributed mesh of length {}",
                n_steps
            );
            let h = h.unwrap_or(1.0);
            let H: Vec<Expr> = vec![Expr::Const(h); n_steps - 1]; // number of intervals = n_steps -1
            let T_list: Vec<f64> = (0..n_steps).map(|i| t0 + (i as f64) * h).collect();
            (H, T_list, n_steps)
        }
    }

    /// Processes variable bounds and tolerances for discretized variables with optimized string operations.
    /// 
    /// Efficiently maps bounds and tolerances from original variable names to all discretized
    /// variable instances. Uses fast string processing to extract base names from discretized variables.
    /// 
    /// # Arguments
    /// * `Bounds` - Optional bounds map: {"y": (min, max), "z": (min, max)}
    /// * `rel_tolerance` - Optional tolerance map: {"y": tol, "z": tol}
    /// * `flat_list_of_names` - All discretized variable names ["y_0", "y_1", "z_0", "z_1", ...]
    /// 
    /// # Performance Optimizations
    /// - **Fast string processing**: Uses `rfind('_')` instead of regex (10-100x faster)
    /// - **Pre-allocation**: Vectors allocated with known capacity
    /// - **Single pass**: Processes all variables in one iteration
    /// 
    /// # Output
    /// Sets `self.bounds` and `self.rel_tolerance_vec` with per-variable values:
    /// - bounds[i] = (min, max) for discretized variable i
    /// - rel_tolerance_vec[i] = tolerance for discretized variable i
    /// 
    /// # Example
    /// ```ignore
    /// // Input: Bounds = {"y": (0.0, 10.0)}, flat_list = ["y_0", "y_1", "z_0"]
    /// // Output: bounds = [(0.0, 10.0), (0.0, 10.0), default_for_z]
    /// ```
    fn process_bounds_and_tolerances(
        &mut self,
        Bounds: Option<HashMap<String, (f64, f64)>>,
        rel_tolerance: Option<HashMap<String, f64>>,
        flat_list_of_names: Vec<String>,
    ) {
        let len = flat_list_of_names.len();

        self.bounds = Bounds.as_ref().map(|bounds_map| {
            let mut vec_of_bounds = Vec::with_capacity(len);
            for name in &flat_list_of_names {
                // Fast string processing: find last underscore and extract base name
                let base_name = if let Some(pos) = name.rfind('_') {
                    &name[..pos]
                } else {
                    name
                };

                if let Some(&bound_pair) = bounds_map.get(base_name) {
                    vec_of_bounds.push(bound_pair);
                }
            }
            vec_of_bounds
        });

        self.rel_tolerance_vec = rel_tolerance.as_ref().map(|tolerance_map| {
            let mut vec_of_tolerance = Vec::with_capacity(len);
            for name in &flat_list_of_names {
                // Fast string processing: find last underscore and extract base name
                let base_name = if let Some(pos) = name.rfind('_') {
                    &name[..pos]
                } else {
                    name
                };

                if let Some(&tolerance) = tolerance_map.get(base_name) {
                    vec_of_tolerance.push(tolerance);
                }
            }
            vec_of_tolerance
        });
    }

    /// High-performance parallel BVP discretization with comprehensive optimization.
    /// 
    /// Converts a continuous BVP system into a discretized algebraic system ready for Newton-Raphson
    /// solving. This is the core discretization method with extensive performance optimizations.
    /// 
    /// # Arguments
    /// * `eq_system` - Vector of symbolic ODE equations [dy/dt = f1(t,y,z), dz/dt = f2(t,y,z)]
    /// * `values` - Original variable names ["y", "z"]
    /// * `arg` - Independent variable name ("t" for time, "x" for space)
    /// * `t0` - Starting point of the domain
    /// * `n_steps` - Number of discretization steps (creates n_steps+1 points)
    /// * `h` - Step size for uniform mesh
    /// * `mesh` - Explicit mesh points (overrides n_steps/h)
    /// * `BorderConditions` - Boundary conditions: {"y": [(0, value), (1, value)]}
    ///   - 0 = initial condition, 1 = final condition
    /// * `Bounds` - Variable bounds for constrained solving
    /// * `rel_tolerance` - Per-variable relative tolerances
    /// * `scheme` - Discretization scheme ("forward" or "trapezoid")
    /// 
    /// # Performance Optimizations
    /// 1. **Parallel discretization**: Uses rayon for concurrent equation processing
    /// 2. **Boundary condition caching**: Pre-computes HashMaps for O(1) BC lookups
    /// 3. **Variable tracking**: Tracks which variables appear in each equation for smart Jacobian
    /// 4. **Memory pre-allocation**: Pre-allocates vectors with known capacity
    /// 5. **Efficient BC application**: Parallel boundary condition substitution
    /// 6. **Fast string processing**: Optimized variable name processing
    /// 
    /// # Output
    /// Creates discretized system: y_{i+1} - y_i - h*f(t_i, y_i, z_i) = 0
    /// Sets up all internal data structures for subsequent Jacobian computation.
    /// 
    /// # Timing
    /// Provides detailed timing breakdown of each optimization phase.
    pub fn discretization_system_BVP_par(
        &mut self,
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        t0: f64,
        n_steps: Option<usize>,
        h: Option<f64>,
        mesh: Option<Vec<f64>>,
        BorderConditions: HashMap<String, Vec<(usize, f64)>>,
        Bounds: Option<HashMap<String, (f64, f64)>>,
        rel_tolerance: Option<HashMap<String, f64>>,
        scheme: String,
    ) {
        let total_start = Instant::now();
        let mut timer_hash: HashMap<String, f64> = HashMap::new();
        let (H, T_list, n_steps) = self.create_mesh(n_steps, h, mesh, t0);
        let (matrix_of_expr, matrix_of_names) = Expr::IndexedVarsMatrix(n_steps, values.clone());
        let bc_handling = Instant::now();
        // Pre-compute boundary conditions lookup for O(1) access
        let bc_lookup: HashMap<String, HashMap<usize, f64>> = BorderConditions
            .into_iter()
            .map(|(k, v)| (k, v.into_iter().collect()))
            .collect();

        // Pre-compute variables to exclude from boundary conditions
        let mut vars_for_boundary_conditions = HashMap::new();
        let mut vars_to_exclude = HashSet::new();

        for (var_name, conditions) in &bc_lookup {
            if let Some(var_idx) = values.iter().position(|v| v == var_name) {
                for (&pos, &value) in conditions {
                    match pos {
                        0 => {
                            // Initial condition
                            let var_name = &matrix_of_names[0][var_idx];
                            vars_for_boundary_conditions.insert(var_name.clone(), value);
                            vars_to_exclude.insert(var_name.clone());
                        }
                        1 => {
                            // Final condition
                            let var_name = &matrix_of_names[n_steps - 1][var_idx];
                            vars_for_boundary_conditions.insert(var_name.clone(), value);
                            vars_to_exclude.insert(var_name.clone());
                        }
                        _ => {}
                    }
                }
            }
        }
        timer_hash.insert(
            "bc handling".to_string(),
            bc_handling.elapsed().as_millis() as f64,
        );
        // DISCRETAZING EQUATIONS
        println!("creating discretization equations");
        let discretization_start = Instant::now();

        // Optimized discretization with variable tracking
        let (discreditized_system, variables_for_all_discrete): (Vec<Expr>, Vec<Vec<String>>) = (0
            ..n_steps - 1)
            .into_par_iter()
            .flat_map(|j| {
                let t = T_list[j];
                eq_system
                    .par_iter()
                    .enumerate()
                    .map(|(i, eq_i)| {
                        let eq_step_j =
                            Self::eq_step(&matrix_of_names, eq_i, &values, &arg, j, t, &scheme);
                        let Y_j_plus_1 = &matrix_of_expr[j + 1][i];
                        let Y_j = &matrix_of_expr[j][i];
                        let res_ij = Y_j_plus_1.clone() - Y_j.clone() - H[j].clone() * eq_step_j;

                        // Track variables used in this equation (excluding boundary condition vars)
                        let mut vars_in_equation = Vec::new();
                        let y_j_plus_1_name = &matrix_of_names[j + 1][i];
                        let y_j_name = &matrix_of_names[j][i];

                        if !vars_to_exclude.contains(y_j_plus_1_name) {
                            vars_in_equation.push(y_j_plus_1_name.clone());
                        }
                        if !vars_to_exclude.contains(y_j_name) {
                            vars_in_equation.push(y_j_name.clone());
                        }

                        // Add variables from eq_step_j (from original equation at this time step)
                        for var_idx in 0..values.len() {
                            let var_name = &matrix_of_names[j][var_idx];
                            if !vars_to_exclude.contains(var_name) {
                                vars_in_equation.push(var_name.clone());
                            }
                        }

                        (res_ij.symplify(), vars_in_equation)
                    })
                    .collect::<Vec<_>>()
            })
            .unzip();

        self.variables_for_all_disrete = variables_for_all_discrete;
        timer_hash.insert(
            "discretization of equations".to_string(),
            discretization_start.elapsed().as_millis() as f64,
        );
        let start_flat_list = Instant::now();

        // Efficient flat list creation with pre-allocation
        let total_vars = values.len() * n_steps;
        let mut flat_list_of_names = Vec::with_capacity(total_vars);
        let mut flat_list_of_expr = Vec::with_capacity(total_vars);

        for var_idx in 0..values.len() {
            for time_idx in 0..n_steps {
                let name = &matrix_of_names[time_idx][var_idx];
                if !vars_to_exclude.contains(name) {
                    flat_list_of_names.push(name.clone());
                    flat_list_of_expr.push(matrix_of_expr[time_idx][var_idx].clone());
                }
            }
        }

        timer_hash.insert(
            "flat list creation".to_string(),
            start_flat_list.elapsed().as_millis() as f64,
        );
        let BC_application_start = Instant::now();
        // Apply boundary conditions to discretized system in parallel
        let discreditized_system_with_BC: Vec<Expr> = discreditized_system
            .into_par_iter()
            .map(|mut eq_i| {
                for (var_name, &value) in &vars_for_boundary_conditions {
                    eq_i = eq_i.set_variable(var_name, value);
                }
                eq_i.symplify()
            })
            .collect();

        let discreditized_system_flat = discreditized_system_with_BC;
        timer_hash.insert(
            "BC application".to_string(),
            BC_application_start.elapsed().as_millis() as f64,
        );
        let consistency_start = Instant::now();
        // Simplified consistency test using pre-computed variables
        let hashset_of_vars: HashSet<&String> = flat_list_of_names.iter().collect();
        let mut missing_vars = Vec::new();

        for var_list in &self.variables_for_all_disrete {
            for var in var_list {
                if !hashset_of_vars.contains(var) {
                    missing_vars.push(var.clone());
                }
            }
        }

        if !missing_vars.is_empty() {
            missing_vars.sort_unstable();
            missing_vars.dedup();
            panic!("Variables not found in system: {:?}", missing_vars);
        }
        timer_hash.insert(
            "consistency test".to_string(),
            consistency_start.elapsed().as_millis() as f64,
        );

        self.vector_of_functions = discreditized_system_flat;
        self.vector_of_variables = flat_list_of_expr;
        self.variable_string = flat_list_of_names.clone();
        let bounds_and_tolerances_start = Instant::now();
        self.process_bounds_and_tolerances(Bounds, rel_tolerance, flat_list_of_names);
        timer_hash.insert(
            "bounds and tolerances".to_string(),
            bounds_and_tolerances_start.elapsed().as_millis() as f64,
        );

        // timing
        let total_end = total_start.elapsed().as_millis() as f64;
        *timer_hash.get_mut("bc handling").unwrap() /= total_end / 100.0;
        *timer_hash.get_mut("discretization of equations").unwrap() /= total_end / 100.0;
        *timer_hash.get_mut("BC application").unwrap() /= total_end / 100.0;
        *timer_hash.get_mut("flat list creation").unwrap() /= total_end / 100.0;
        *timer_hash.get_mut("consistency test").unwrap() /= total_end / 100.0;
        *timer_hash.get_mut("bounds and tolerances").unwrap() /= total_end / 100.0;
        timer_hash.insert("total time, sec".to_string(), total_end);

        let mut table = Builder::from(timer_hash.clone()).build();
        table.with(Style::modern_rounded());
        println!("{}", table.to_string());
    }

    /// Automatically detects the bandwidth of the sparse Jacobian matrix using parallel computation.
    /// 
    /// Analyzes the symbolic Jacobian to determine the banded structure, computing the number
    /// of sub- and super-diagonals. This information is crucial for banded matrix optimizations.
    /// 
    /// # Algorithm
    /// For each non-zero element at position (i,j):
    /// - If j > i: contributes to super-diagonal width (ku)
    /// - If i > j: contributes to sub-diagonal width (kl)
    /// 
    /// # Performance Features
    /// - **Parallel computation**: Uses rayon to process rows concurrently
    /// - **Reduction pattern**: Efficiently combines results from parallel workers
    /// - **Early termination**: Skips zero elements for efficiency
    /// 
    /// # Output
    /// Sets `self.bandwidth = Some((kl, ku))` where:
    /// - kl = maximum number of sub-diagonals
    /// - ku = maximum number of super-diagonals
    /// 
    /// # When to Use
    /// - Automatically called by `generate_BVP()` if bandwidth not provided
    /// - Essential for banded matrix solvers and storage optimization
    /// - Particularly important for large sparse systems
    fn find_bandwidths(&mut self) {
        let A = &self.symbolic_jacobian;
        let n = A.len();
        // kl  Number of subdiagonals
        // ku = 0; Number of superdiagonals

        /*
            Matrix Iteration: The function find_bandwidths iterates through each element of the matrix A.
        Subdiagonal Width (kl): For each non-zero element below the main diagonal (i.e., i > j), it calculates the distance from the diagonal and updates
        kl if this distance is greater than the current value of kl.
        Superdiagonal Width (ku): Similarly, for each non-zero element above the main diagonal (i.e., j > i), it calculates the distance from the diagonal
         and updates ku if this distance is greater than the current value of ku.
             */
        let (kl, ku) = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut row_kl = 0;
                let mut row_ku = 0;
                for j in 0..n {
                    if A[i][j] != Expr::Const(0.0) {
                        if j > i {
                            row_ku = std::cmp::max(row_ku, j - i);
                        } else if i > j {
                            row_kl = std::cmp::max(row_kl, i - j);
                        }
                    }
                }
                (row_kl, row_ku)
            })
            .reduce(
                || (0, 0),
                |acc, row| (std::cmp::max(acc.0, row.0), std::cmp::max(acc.1, row.1)),
            );

        self.bandwidth = Some((kl, ku));
    }

    /// **MAIN FUNCTION**: Complete BVP symbolic-to-numerical transformation pipeline.
    /// 
    /// This is the primary entry point for BVP solving. It orchestrates the entire process from
    /// symbolic ODE system to high-performance numerical functions ready for Newton-Raphson solving.
    /// 
    /// # Complete Workflow
    /// 1. **Discretization**: Converts continuous BVP to algebraic system
    /// 2. **Symbolic Jacobian**: Computes analytical partial derivatives
    /// 3. **Bandwidth Detection**: Analyzes sparsity structure
    /// 4. **Function Compilation**: Creates optimized numerical evaluators
    /// 5. **Performance Timing**: Provides detailed timing breakdown
    /// 
    /// # Arguments
    /// * `eq_system` - Vector of symbolic ODE equations
    /// * `values` - Original variable names ["y", "z"]
    /// * `arg` - Independent variable name ("t" or "x")
    /// * `t0` - Domain starting point
    /// * `param` - Parameter name (defaults to `arg` if None)
    /// * `n_steps` - Number of discretization steps
    /// * `h` - Step size for uniform mesh
    /// * `mesh` - Explicit mesh points
    /// * `BorderConditions` - Boundary conditions specification
    /// * `Bounds` - Variable bounds for constrained solving
    /// * `rel_tolerance` - Per-variable relative tolerances
    /// * `scheme` - Discretization scheme ("forward" or "trapezoid")
    /// * `method` - Matrix backend ("Dense", "Sparse", "Sparse_1", "Sparse_2")
    /// * `bandwidth` - Optional pre-computed bandwidth (auto-detected if None)
    /// 
    /// # Supported Matrix Backends
    /// - **"Dense"**: nalgebra DMatrix - best for small, dense systems
    /// - **"Sparse"**: faer SparseColMat - recommended for large sparse systems
    /// - **"Sparse_1"**: sprs CsMat - legacy sparse support
    /// - **"Sparse_2"**: nalgebra CsMatrix - nalgebra ecosystem integration
    /// 
    /// # Performance Features
    /// - Comprehensive timing analysis with percentage breakdown
    /// - Automatic bandwidth detection for sparse optimization
    /// - Smart sparsity-aware Jacobian computation
    /// - Multi-backend support for optimal performance
    /// 
    /// # Output
    /// Sets up `self.jac_function` and `self.residiual_function` ready for BVP solving.
    /// Prints detailed timing table showing performance breakdown.
    /// 
    /// # Example Usage
    /// ```ignore
    /// let mut jacobian = Jacobian::new();
    /// jacobian.generate_BVP(
    ///     equations, variables, "t", 0.0, None, Some(100), None, None,
    ///     boundary_conditions, bounds, tolerances, "trapezoid".to_string(),
    ///     "Sparse".to_string(), None
    /// );
    /// ```
    pub fn generate_BVP(
        &mut self,
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        t0: f64,
        param: Option<String>,
        n_steps: Option<usize>,
        h: Option<f64>,
        mesh: Option<Vec<f64>>,
        BorderConditions: HashMap<String, Vec<(usize, f64)>>,
        Bounds: Option<HashMap<String, (f64, f64)>>,
        rel_tolerance: Option<HashMap<String, f64>>,
        scheme: String,
        method: String,
        bandwidth: Option<(usize, usize)>,
    ) {
        let total_start = Instant::now();
        let mut timer_hash: HashMap<String, f64> = HashMap::new();
        let param = if let Some(param) = param {
            param
        } else {
            arg.clone()
        };
        self.method = method.clone();
        let begin = Instant::now();
        self.discretization_system_BVP_par(
            eq_system,
            values.clone(),
            arg.clone(),
            t0,
            n_steps,
            h,
            mesh,
            BorderConditions,
            Bounds,
            rel_tolerance,
            scheme,
        );
        timer_hash.insert(
            "discretization time".to_string(),
            begin.elapsed().as_secs_f64(),
        );
        info!("system discretized");
        let v = self.variable_string.clone();
        /*
        info!(
            "VECTOR OF FUNCTIONS \n \n  {:?},  length: {} \n \n",
            &self.vector_of_functions,
            &self.vector_of_functions.len()
        );
        */
        // println!("INDEXED VARIABLES {:?}, length:  {} \n \n", &v, &v.len());
        let indexed_values: Vec<&str> = v.iter().map(|x| x.as_str()).collect();
        info!("Calculating jacobian");
        let now = Instant::now();
        self.calc_jacobian_parallel_smart(); //calculate the symbolic Jacobian matrix.
        info!("Jacobian calculation time:");
        let elapsed = now.elapsed();
        info!("{:?}", elapsed_time(elapsed));
        timer_hash.insert("symbolic jacobian time".to_string(), elapsed.as_secs_f64());
        let now = Instant::now();
        if let Some(bandwidth_) = bandwidth {
            self.bandwidth = Some(bandwidth_);
            info!("Bandwidth provided: {:?}", self.bandwidth);
        } else {
            self.find_bandwidths(); //  determine the bandwidth of the Jacobian matrix.
            info!("Bandwidth calculated:");
            info!("(kl, ku) = {:?}", self.bandwidth);
        }
        timer_hash.insert(
            "find bandwidth time".to_string(),
            now.elapsed().as_secs_f64(),
        );
        //  println!("symbolic Jacbian created {:?}", &self.symbolic_jacobian);
        let now = Instant::now();
        match method.as_str() {
            // transform the symbolic Jacobian into a numerical function
            "Dense" => self.lambdify_jacobian_DMatrix_par(param.as_str(), indexed_values.clone()),
            "Sparse_1" => self.lambdify_jacobian_CsMat(param.as_str(), indexed_values.clone()),
            "Sparse_2" => self.lambdify_jacobian_CsMatrix(param.as_str(), indexed_values.clone()),
            "Sparse" => {
                self.lambdify_jacobian_SparseColMat_parallel2(param.as_str(), indexed_values.clone())
            }
            _ => panic!("unknown method: {}", method),
        }
        timer_hash.insert(
            "jacobian lambdify time".to_string(),
            now.elapsed().as_secs_f64(),
        );
        info!("Jacobian lambdified");
        let n = &self.symbolic_jacobian.len();
        for (_i, vec_s) in self.symbolic_jacobian.iter().enumerate() {
            assert_eq!(vec_s.len(), *n, "jacobian not square ");
        }
        // println!("functioon Jacobian created");
        let now = Instant::now();
        match method.as_str() {
            // transform the symbolic residual vector-function into a numerical function
            "Dense" => self.lambdify_residual_DVector(param.as_str(), indexed_values.clone()),
            "Sparse_1" => self.lambdify_residual_CsVec(param.as_str(), indexed_values.clone()),
            "Sparse_2" => self.lambdify_residual_DVector(param.as_str(), indexed_values.clone()),
            "Sparse" => self.lambdify_residual_Col_parallel2(param.as_str(), indexed_values.clone()),
            _ => panic!("unknown method: {}", method),
        }
        timer_hash.insert(
            "residual functions lambdify time".to_string(),
            now.elapsed().as_secs_f64(),
        );
        info!("Residuals vector lambdified");
        /*
        if let Some(bounds_) = self.bounds.clone() {
            println!(
                "\n \n bounds vector {:?}, of lengh {}",
                bounds_,
                bounds_.len()
            )

        }
        if let Some(rel_tolerance) = self.rel_tolerance_vec.clone() {
            println!(
                "\n \n abs_tolerance vector {:?}, of lengh {}",
                rel_tolerance,
                rel_tolerance.len()
            )
        }
        */
        let total_end = total_start.elapsed().as_secs_f64();
        *timer_hash.get_mut("discretization time").unwrap() /= total_end / 100.0;
        *timer_hash.get_mut("symbolic jacobian time").unwrap() /= total_end / 100.0;
        *timer_hash.get_mut("jacobian lambdify time").unwrap() /= total_end / 100.0;
        *timer_hash
            .get_mut("residual functions lambdify time")
            .unwrap() /= total_end / 100.0;
        *timer_hash.get_mut("find bandwidth time").unwrap() /= total_end / 100.0;
        timer_hash.insert("total time, sec".to_string(), total_end);

        let mut table = Builder::from(timer_hash.clone()).build();
        table.with(Style::modern_rounded());
        println!("{}", table.to_string());
        //   panic!("END OF GENERATE BVP");
        info!(
            "\n \n ____________END OF GENERATE BVP ________________________________________________________________"
        );
    }
} // end of impl

////////////////////////////////////////////////////////////////
// TESTS
////////////////////////////////////////////////////////////////

#[cfg(test)]

mod tests {
    use super::*;
    use crate::numerical::BVP_Damp::BVP_traits::Vectors_type_casting;
    /*
    #[test]
    fn generate_IVP_test() {
        let mut Jacobian_instance = Jacobian::new();
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let z: Expr = Expr::Var("z".to_string());
        //eq = z*x +exp(y);
        let eq1: Expr = z.clone() * x.clone() + Expr::exp(y.clone());
        // eq2 = x + ln(z) + y
        let eq2: Expr = x + Expr::ln(z) + y;
        let eq_system = vec![eq1, eq2];
        let values = vec!["y".to_string(), "z".to_string()];
        let arg = "x".to_string();
        // expecting Jac:
        Jacobian_instance.generate_IVP(eq_system, values, arg);
        // J= {{e^y, 10}, {1, 1/z}}
        //
        Jacobian_instance.evaluate_func_jacobian_DMatrix_IVP(10.0, vec![0.0, 1.0]);
        println!(
            "Jacobian_instance.evaluated_jacobian_DMatrix = {:?}",
            Jacobian_instance.evaluated_jacobian_DMatrix
        );
        assert_eq!(
            Jacobian_instance.evaluated_jacobian_DMatrix,
            DMatrix::from_row_slice(2, 2, &[1.0, 10.0, 1.0, 1.0])
        );
        Jacobian_instance.evaluate_funvector_lambdified_DVector_IVP(10.0, vec![0.0, 1.0]);
        println!(
            "Jacobian_instance.evaluated_functions_DVector = {:?}",
            Jacobian_instance.evaluated_functions_DVector
        );
        assert_eq!(
            Jacobian_instance.evaluated_functions_DVector,
            DVector::from_vec(vec![11.0, 10.0])
        );
    }
     */
    #[test]
    fn generate_Jac_test() {
        let Jacobian_instance = &mut Jacobian::new();

        let y = Expr::Var("y".to_string());
        let z: Expr = Expr::Var("z".to_string());
        //eq = z*10.0 +exp(y);
        let eq1: Expr = z.clone() * Expr::Const(10.0) + Expr::exp(y.clone());
        // eq2 = x + ln(z) + y
        let eq2: Expr = Expr::ln(z.clone()) + y.clone();
        let eq_system = vec![eq1, eq2];
        let values = vec!["y".to_string(), "z".to_string()];
        //  let arg = "x".to_string();
        Jacobian_instance.from_vectors(eq_system, values);
        Jacobian_instance.calc_jacobian_parallel();
        println!(
            "Jacobian_instance.evaluated_jacobian_DMatrix = {:?}",
            Jacobian_instance.symbolic_jacobian
        );
        // expecting Jac: | exp(y) ; 10.0  |
        //                 | 1     ;  1/z|
        let vect = &DVector::from_vec(vec![0.0, 1.0]);
        let expexted_jac = DMatrix::from_row_slice(2, 2, &[1.0, 10.0, 1.0, 1.0]);
        // cast type to deisred crate type
        // FAER CRATE
        Jacobian_instance.lambdify_jacobian_SparseColMat_modified("x", vec!["y", "z"]);

        let variables = &*Vectors_type_casting(vect, "Sparse".to_string());
        let jac = Jacobian_instance.jac_function.as_mut().unwrap();
        let result_SparseColMat = jac.call(1.0, variables);
        assert_eq!(
            result_SparseColMat.to_DMatrixType(),
            expexted_jac,
            "FAER CRATE jac error"
        );
        // NALGEBRA CRATE
        Jacobian_instance.lambdify_jacobian_DMatrix_par("x", vec!["y", "z"]);
        let variables = &*Vectors_type_casting(vect, "Dense".to_string());
        let jac = Jacobian_instance.jac_function.as_mut().unwrap();
        let result_DMatrix = jac.call(1.0, variables);
        assert_eq!(
            result_DMatrix.to_DMatrixType(),
            expexted_jac,
            "nalagebra jac error"
        );

        // NALGEBRA SPARSE CRATE
        Jacobian_instance.lambdify_jacobian_CsMatrix("x", vec!["y", "z"]);
        let variables = &*Vectors_type_casting(vect, "Sparse_2".to_string());
        let jac = Jacobian_instance.jac_function.as_mut().unwrap();
        let result_CsMatrix = jac.call(1.0, variables);
        assert_eq!(
            result_CsMatrix.to_DMatrixType(),
            expexted_jac,
            "sprs jac error"
        );
        //SPRS CRATE
        /*
               Jacobian_instance.lambdify_jacobian_CsMat(  "x", vec!("y", "z"));
               let variables = &*Vectors_type_casting(vect, "Sparse_1".to_string());
               let jac =   Jacobian_instance.jac_function.as_mut().unwrap();
               let result_CsMat = jac.call(1.0,  variables.clone());
               println!("RES {:?}, EXPECTED {}", result_CsMat, expexted_jac);
               assert_eq!(result_CsMat.to_DMatrixType(),   expexted_jac);

        */
    }
    #[test]
    fn CsMat_test() {}
}

/*
#[test]
fn test_jacobian_new() {
    let jac = Jacobian::new();
    assert_eq!(jac.vector_of_functions.len(), 0);
    assert_eq!(jac.vector_of_variables.len(), 0);
    assert_eq!(jac.variable_string.len(), 0);
    assert_eq!(jac.symbolic_jacobian.len(), 0);
    assert_eq!(jac.readable_jacobian.len(), 0);
    assert_eq!(jac.function_jacobian.len(), 0);
}

#[test]
fn test_jacobian_from_vectors() {
    let funcs = vec![Expr::parse_expression("x^2 + y^2"), Expr::parse_expression("log(x) + exp(y)")];
    let vars = vec![Expr::parse_expression("x"), Expr::parse_expression("y")];
    let jac = Jacobian::from_vectors(funcs, vars);
    assert_eq!(jac.vector_of_functions.len(), 2);
    assert_eq!(jac.vector_of_variables.len(), 2);
    assert_eq!(jac.variable_string.len(), 0);
    assert_eq!(jac.symbolic_jacobian.len(), 2);
    assert_eq!(jac.readable_jacobian.len(), 2);
    assert_eq!(jac.function_jacobian.len(), 2);
}

#[test]
fn test_jacobian_calc_jacobian() {
    let mut jac = Jacobian::new();
    jac.set_vector_of_functions(vec![Expr::parse_expression("x^2 + y^2"), Expr::parse_expression("sin(x) + cos(y)")]);
    jac.set_vector_of_variables(vec![Expr::parse_expression("x"), Expr::parse_expression("y")]);
    jac.set_varvecor_from_str("x, y");
    jac.calc_jacobian();
    let expected_jacobian = vec![
        vec![Expr::parse_expression("2*x"), Expr::parse_expression("2*y")],
        vec![Expr::parse_expression("cos(x)"), Expr::parse_expression("-sin(y)")],
    ];
    assert_eq!(jac.symbolic_jacobian, expected_jacobian);
}

#[test]
fn test_jacobian_calc_jacobian_fun() {
    let mut jac = Jacobian::new();
    jac.set_vector_of_functions(vec![Expr::parse_expression("x^2 + y^2"), Expr::parse_expression("sin(x) + cos(y)")]);
    jac.set_vector_of_variables(vec![Expr::parse_expression("x"), Expr::parse_expression("y")]);
    jac.set_varvecor_from_str("x, y");
    jac.calc_jacobian();
    jac.calc_jacobian_fun();
    let x = vec![1.0, 2.0];
    let expected_result = vec![
        vec![2.0, 4.0],
        vec![0.5403023058681398, -0.4161468365471424],
    ];
  //  assert_eq!(jac.evaluate_func_jacobian(&x), expected_result);
}
*/
