use crate::numerical::BVP_Damp::adaptive_grid_easiest::{
    easiest_grid_refinement, grcar_smooke_grid_refinement, pearson_grid_refinement,
};
use nalgebra::{DMatrix, DVector};
/// interface for creating a new grid
pub enum GridRefinementMethod {
    Easiest,
    Pearson,
    GrcarSmooke,
}
trait GridRefinement {
    fn refine(
        &self,
        y_DMatrix: &DMatrix<f64>,
        x_mesh: &DVector<f64>,
        params: Vec<f64>,
    ) -> (Vec<f64>, DMatrix<f64>, usize);
}

struct EasiestGridRefinement;
impl GridRefinement for EasiestGridRefinement {
    fn refine(
        &self,
        y_DMatrix: &DMatrix<f64>,
        x_mesh: &DVector<f64>,
        params: Vec<f64>,
    ) -> (Vec<f64>, DMatrix<f64>, usize) {
        let (tolerance, safety_param) = (params[0], params[1]);
        let (new_grid, new_solution, number_of_nonzero_keys) =
            easiest_grid_refinement(y_DMatrix, x_mesh, tolerance, safety_param);
        (new_grid, new_solution, number_of_nonzero_keys)
    }
}

struct PearsonGridRefinement;

impl GridRefinement for PearsonGridRefinement {
    fn refine(
        &self,
        y_DMatrix: &DMatrix<f64>,
        x_mesh: &DVector<f64>,
        params: Vec<f64>,
    ) -> (Vec<f64>, DMatrix<f64>, usize) {
        let (new_grid, new_solution, number_of_nonzero_keys) =
            pearson_grid_refinement(y_DMatrix, x_mesh, params[0]);
        (new_grid, new_solution, number_of_nonzero_keys)
    }
}
struct GrcarSmookeRefinement;
impl GridRefinement for GrcarSmookeRefinement {
    fn refine(
        &self,
        y_DMatrix: &DMatrix<f64>,
        x_mesh: &DVector<f64>,
        params: Vec<f64>,
    ) -> (Vec<f64>, DMatrix<f64>, usize) {
        let (new_grid, new_solution, number_of_nonzero_keys) =
            grcar_smooke_grid_refinement(y_DMatrix, x_mesh, params[0], params[1]);
        (new_grid, new_solution, number_of_nonzero_keys)
    }
}
pub fn new_grid(
    method: GridRefinementMethod,
    y_DMatrix: &DMatrix<f64>,
    x_mesh: &DVector<f64>,
    params: Vec<f64>,
) -> (Vec<f64>, DMatrix<f64>, usize) {
    let refiner: Box<dyn GridRefinement> = match method {
        GridRefinementMethod::Easiest => Box::new(EasiestGridRefinement),
        GridRefinementMethod::Pearson => Box::new(PearsonGridRefinement),
        GridRefinementMethod::GrcarSmooke => Box::new(GrcarSmookeRefinement),
    };
    let (new_grid, new_initial_guess, number_of_nonzero_keys) =
        refiner.refine(&y_DMatrix, &x_mesh, params);
    (new_grid, new_initial_guess, number_of_nonzero_keys)
}
