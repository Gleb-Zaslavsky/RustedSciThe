use crate::numerical::BVP_Damp::adaptive_grid_basic::{
    easy_grid_refinement, grcar_smooke_grid_refinement, pearson_grid_refinement, refine_all_grid,
    scipy_grid_refinement,
};
use crate::numerical::BVP_Damp::adaptive_grid_twopoint::twpnt_refinement;
use nalgebra::{DMatrix, DVector};
/// interface for creating a new grid
#[derive(Debug, Clone)]
pub enum GridRefinementMethod {
    /// easiest method: double the number of points
    DoublePoints, //
    /// more advanced method: comare differences between adjacent points and refine where large
    Easy(f64), // tolerance,
    /// advanced method: Pearson's algorithm
    Pearson(f64, f64), // single parameter
    /// advanced method: Grcar-Smooke algorithm
    GrcarSmooke(f64, f64, f64), // three parameters
    /// advanced method: SciPy's method (not implemented yet)
    Sci(),
    /// advanced method: inspired Two-Point FORTAN algorithm
    TwoPoint(f64, f64, f64), // three parameters
}

pub fn new_grid(
    method: GridRefinementMethod,
    y_DMatrix: &DMatrix<f64>,
    x_mesh: &DVector<f64>,
    abs_tolerance: f64,
    residuals: Option<DVector<f64>>,
) -> (Vec<f64>, DMatrix<f64>, usize) {
    match method {
        GridRefinementMethod::DoublePoints => refine_all_grid(y_DMatrix, x_mesh),
        GridRefinementMethod::Easy(tolerance) => easy_grid_refinement(y_DMatrix, x_mesh, tolerance),
        GridRefinementMethod::Pearson(param, safety_par) => {
            pearson_grid_refinement(y_DMatrix, x_mesh, param, safety_par)
        }
        GridRefinementMethod::GrcarSmooke(p1, p2, p3) => {
            grcar_smooke_grid_refinement(y_DMatrix, x_mesh, p1, p2, p3)
        }
        GridRefinementMethod::Sci() => {
            scipy_grid_refinement(y_DMatrix, x_mesh, abs_tolerance, residuals)
        }
        GridRefinementMethod::TwoPoint(p1, p2, p3) => {
            twpnt_refinement(y_DMatrix.clone(), x_mesh.clone(), p1, p2, p3, abs_tolerance)
        }
    }
}
