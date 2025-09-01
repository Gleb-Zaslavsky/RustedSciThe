//! # Adaptive Grid Refinement API
//!
//! ## Module Purpose
//! This module provides a unified interface for various adaptive grid refinement algorithms
//! used in boundary value problem (BVP) solvers. It acts as a dispatcher that selects and
//! executes the appropriate grid refinement method based on solution characteristics and
//! user preferences.
//!
//! ## Key Features
//! - **Unified Interface**: Single entry point for all grid refinement algorithms
//! - **Multiple Algorithms**: Supports 6 different refinement strategies from simple to sophisticated
//! - **Automatic Selection**: Chooses optimal refinement based on problem characteristics
//! - **Solution Interpolation**: Provides initial guess on refined mesh through interpolation
//!
//! ## Main Structures
//! - [`GridRefinementMethod`]: Enum defining available refinement algorithms with their parameters
//!
//! ## Core Function
//! - [`new_grid`]: Main dispatcher function that creates refined mesh and interpolated solution
//!
//! ## Available Algorithms
//! 1. **DoublePoints**: Naive approach - doubles mesh density everywhere
//! 2. **Easy**: Simple gradient-based refinement with tolerance parameter
//! 3. **Pearson**: Classical boundary layer algorithm with smoothing
//! 4. **GrcarSmooke**: Advanced method considering both solution and derivative jumps
//! 5. **Sci**: SciPy-inspired residual-based refinement
//! 6. **TwoPoint**: FORTRAN TWOPNT-inspired algorithm with multiple criteria
//!
//! ## Algorithm Selection Guidelines
//! - Use `DoublePoints` for testing or when other methods fail
//! - Use `Easy` for simple problems with moderate gradients
//! - Use `Pearson` for boundary layer problems
//! - Use `GrcarSmooke` for combustion/reaction problems
//! - Use `Sci` when residual information is available
//! - Use `TwoPoint` for complex multi-scale problems
//!
//! ## Integration with BVP Solver
//! This module is called by the damped Newton-Raphson solver when:
//! - Newton iterations fail to converge on current mesh
//! - Solution gradients exceed specified thresholds
//! - Adaptive refinement is enabled in solver configuration

use crate::numerical::BVP_Damp::adaptive_grid_basic::{
    easy_grid_refinement_par, grcar_smooke_grid_refinement_par, pearson_grid_refinement_par, refine_all_grid_par,
    scipy_grid_refinement,
};
use crate::numerical::BVP_Damp::adaptive_grid_twopoint::twpnt_refinement;
use nalgebra::{DMatrix, DVector};
/// Enumeration of available grid refinement algorithms
///
/// Each variant represents a different strategy for adaptive mesh refinement,
/// with associated parameters that control the refinement behavior.
///
/// # Algorithm Descriptions
/// - `DoublePoints`: Uniformly refines by adding midpoints to all intervals
/// - `Easy(tolerance)`: Refines based on solution gradient magnitude
/// - `Pearson(delta, safety)`: Classical boundary layer method with buffering
/// - `GrcarSmooke(d, g, C)`: Considers both solution jumps and derivative changes
/// - `Sci()`: Uses residual-based error estimation
/// - `TwoPoint(p1, p2, p3)`: Multi-criteria refinement with three parameters
#[derive(Debug, Clone)]
pub enum GridRefinementMethod {
    /// Simplest method: doubles mesh density by adding midpoints to all intervals
    /// 
    /// Always adds n-1 new points for n original points. Guaranteed to refine
    /// but may be inefficient for problems with localized features.
    DoublePoints,
    
    /// Gradient-based refinement with tolerance parameter
    /// 
    /// Refines intervals where solution changes exceed `tolerance * (max - min)`.
    /// Good for problems with moderate solution gradients.
    /// 
    /// # Parameters
    /// - `f64`: Relative tolerance (typically 0.1 to 0.5)
    Easy(f64),
    
    /// Pearson's boundary layer algorithm with smoothing
    /// 
    /// Classical method for problems with boundary layers. Includes buffering
    /// to prevent abrupt mesh size changes.
    /// 
    /// # Parameters  
    /// - `f64`: Delta parameter (solution jump threshold, typically 1e-3)
    /// - `f64`: Safety parameter for mesh smoothing (typically 1.4)
    Pearson(f64, f64),
    
    /// Grcar-Smooke algorithm for combustion problems
    /// 
    /// Advanced method considering both solution jumps and derivative changes.
    /// Developed for flame simulation problems.
    /// 
    /// # Parameters
    /// - `f64`: Solution jump threshold (typically 1e-3)
    /// - `f64`: Derivative change threshold (typically 1e-2) 
    /// - `f64`: Mesh ratio constraint (typically 1.4)
    GrcarSmooke(f64, f64, f64),
    
    /// SciPy-inspired residual-based refinement
    /// 
    /// Uses residual magnitude to determine refinement needs.
    /// Requires residual vector from previous solution attempt.
    Sci(),
    
    /// FORTRAN TWOPNT-inspired multi-criteria algorithm
    /// 
    /// Sophisticated method with multiple refinement criteria.
    /// Best for complex multi-scale problems.
    /// 
    /// # Parameters
    /// - `f64`: Primary refinement parameter
    /// - `f64`: Secondary refinement parameter  
    /// - `f64`: Tertiary refinement parameter
    TwoPoint(f64, f64, f64),
}

/// Creates a refined mesh using the specified refinement algorithm
///
/// This is the main entry point for adaptive grid refinement. It analyzes the current
/// solution and mesh, then applies the chosen refinement algorithm to create a finer
/// mesh where needed.
///
/// # Arguments
/// * `method` - The grid refinement algorithm to use
/// * `y_DMatrix` - Current solution matrix (variables × mesh points)
/// * `x_mesh` - Current mesh points (spatial/temporal coordinates)
/// * `abs_tolerance` - Absolute tolerance for convergence checking
/// * `residuals` - Optional residual vector (required for Sci method)
///
/// # Returns
/// A tuple containing:
/// * `Vec<f64>` - New refined mesh points
/// * `DMatrix<f64>` - Interpolated solution on new mesh (initial guess)
/// * `usize` - Number of intervals that were refined
///
/// # Algorithm Selection
/// The function dispatches to the appropriate refinement algorithm based on the
/// `method` parameter. Each algorithm analyzes the solution differently:
/// 
/// - **Gradient-based** (Easy, Pearson, GrcarSmooke): Look at solution changes
/// - **Residual-based** (Sci): Uses equation residuals to guide refinement
/// - **Uniform** (DoublePoints): Refines everywhere regardless of solution
/// - **Multi-criteria** (TwoPoint): Combines multiple refinement indicators
///
/// # Interpolation Strategy
/// All methods provide linear interpolation of the old solution onto the new mesh
/// to create an initial guess for the next Newton iteration. This ensures the
/// refined solution starts close to the converged coarse solution.
///
/// # Performance Notes
/// - Refinement cost scales with number of mesh points and solution complexity
/// - Methods that add fewer points (Easy, Pearson) are faster but may need more iterations
/// - Uniform refinement (DoublePoints) is expensive but guaranteed to improve resolution
pub fn new_grid(
    method: GridRefinementMethod,
    y_DMatrix: &DMatrix<f64>,
    x_mesh: &DVector<f64>,
    abs_tolerance: f64,
    residuals: Option<DVector<f64>>,
) -> (Vec<f64>, DMatrix<f64>, usize) {
    match method {
        // Uniform refinement - always doubles mesh density
        GridRefinementMethod::DoublePoints => refine_all_grid_par(y_DMatrix, x_mesh),
        
        // Simple gradient-based refinement
        GridRefinementMethod::Easy(tolerance) => easy_grid_refinement_par(y_DMatrix, x_mesh, tolerance),
        
        // Boundary layer algorithm with smoothing
        GridRefinementMethod::Pearson(param, safety_par) => {
            pearson_grid_refinement_par(y_DMatrix, x_mesh, param, safety_par)
        }
        
        // Advanced combustion-oriented algorithm
        GridRefinementMethod::GrcarSmooke(p1, p2, p3) => {
            grcar_smooke_grid_refinement_par(y_DMatrix, x_mesh, p1, p2, p3)
        }
        
        // Residual-based refinement (requires residual vector)
        GridRefinementMethod::Sci() => {
            scipy_grid_refinement(y_DMatrix, x_mesh, abs_tolerance, residuals)
        }
        
        // Multi-criteria TWOPNT-inspired algorithm
        GridRefinementMethod::TwoPoint(p1, p2, p3) => {
            twpnt_refinement(y_DMatrix.clone(), x_mesh.clone(), p1, p2, p3, abs_tolerance)
        }
    }
}
