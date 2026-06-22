use crate::numerical::optimization::problem_LM::LeastSquaresProblem;
use crate::numerical::optimization::varpro::{
    model::SeparableNonlinearModel,
    problem::{RhsType, SeparableProblem},
};
use nalgebra::DMatrix;

mod svd;

pub use svd::SvdLinearSolver;

/// VarPro LM problem using the SVD backend for the linear subproblem.
pub type LevMarProblemSvd<Model, Rhs> = LevMarProblem<Model, Rhs, SvdLinearSolver>;

/// Reduced nonlinear least-squares problem passed to Levenberg-Marquardt.
#[derive(Debug)]
pub struct LevMarProblem<Model, Rhs, Solver>
where
    Model: SeparableNonlinearModel<ScalarType = f64>,
    Solver: LinearSolver<ScalarType = f64>,
    Rhs: RhsType,
{
    pub(crate) separable_problem: SeparableProblem<Model, Rhs>,
    pub(crate) cached: Option<Solver>,
}

impl<Model, Rhs, Solver> From<SeparableProblem<Model, Rhs>> for LevMarProblem<Model, Rhs, Solver>
where
    Model: SeparableNonlinearModel<ScalarType = f64>,
    Solver: LinearSolver<ScalarType = f64>,
    Rhs: RhsType,
    Self: LeastSquaresProblem,
{
    fn from(problem: SeparableProblem<Model, Rhs>) -> Self {
        let mut this = Self {
            separable_problem: problem,
            cached: None,
        };
        this.set_params(&this.separable_problem.model.params());
        this
    }
}

/// Linear solver cache used by the reduced VarPro problem.
pub trait LinearSolver: std::fmt::Debug + sealed::Sealed {
    /// Numeric type used in this solver.
    type ScalarType;

    /// Return the linear coefficients in matrix form.
    fn linear_coefficients_matrix(self) -> DMatrix<Self::ScalarType>;
}

impl sealed::Sealed for SvdLinearSolver {}

pub mod sealed {
    pub trait Sealed {}
}
