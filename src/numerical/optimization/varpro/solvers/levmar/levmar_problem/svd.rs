use super::{LevMarProblem, LinearSolver};
use crate::numerical::optimization::problem_LM::LeastSquaresProblem;
use crate::numerical::optimization::varpro::{
    model::SeparableNonlinearModel, problem::RhsType, util::to_vector,
};
use nalgebra::{DMatrix, DVector, Dyn, Matrix, MatrixViewMut, SVD};

/// Cached SVD solution of the linear subproblem for the current nonlinear parameters.
#[derive(Debug, Clone)]
pub struct SvdLinearSolver {
    /// Current weighted residual matrix for all right-hand sides.
    pub(crate) current_residuals: DMatrix<f64>,
    /// Singular value decomposition of the weighted basis matrix.
    pub(crate) decomposition: SVD<f64, Dyn, Dyn>,
    /// Linear coefficients that solve the current linear least-squares subproblem.
    pub(crate) linear_coefficients: DMatrix<f64>,
}

impl LinearSolver for SvdLinearSolver {
    type ScalarType = f64;

    fn linear_coefficients_matrix(self) -> DMatrix<f64> {
        self.linear_coefficients
    }
}

impl<Model, Rhs> LeastSquaresProblem for LevMarProblem<Model, Rhs, SvdLinearSolver>
where
    Model: SeparableNonlinearModel<ScalarType = f64>,
    Rhs: RhsType,
{
    #[allow(non_snake_case)]
    fn set_params(&mut self, params: &DVector<f64>) {
        if self
            .separable_problem
            .model
            .set_params(params.clone())
            .is_err()
        {
            self.cached = None;
            return;
        }

        let Ok(Phi) = self.separable_problem.model.eval() else {
            self.cached = None;
            return;
        };

        let Phi_w = &self.separable_problem.weights * Phi;
        let max_dim_phiw = Phi_w.ncols().max(Phi_w.nrows()) as f64;
        let current_svd = Phi_w.clone().svd(true, true);
        let svd_epsilon = current_svd.singular_values.max() * max_dim_phiw * f64::EPSILON;

        let Ok(linear_coefficients) = current_svd.solve(&self.separable_problem.Y_w, svd_epsilon)
        else {
            self.cached = None;
            return;
        };

        // The local LM implementation follows the usual convention
        // `residual = model - data`, so the Jacobian below is the direct
        // derivative of the projected model residual.
        let current_residuals = Phi_w * &linear_coefficients - &self.separable_problem.Y_w;

        self.cached = Some(SvdLinearSolver {
            current_residuals,
            decomposition: current_svd,
            linear_coefficients,
        })
    }

    fn params(&self) -> DVector<f64> {
        self.separable_problem.model.params()
    }

    fn residuals(&self) -> Option<DVector<f64>> {
        self.cached
            .as_ref()
            .map(|cached| to_vector(cached.current_residuals.clone()))
    }

    #[allow(non_snake_case)]
    fn jacobian(&self) -> Option<DMatrix<f64>> {
        let SvdLinearSolver {
            current_residuals: _,
            decomposition,
            linear_coefficients,
        } = self.cached.as_ref()?;

        let data_cols = self.separable_problem.Y_w.ncols();
        let parameter_count = self.separable_problem.model.parameter_count();
        let mut jacobian_matrix = DMatrix::zeros(
            self.separable_problem.model.output_len() * data_cols,
            parameter_count,
        );

        let U_full = decomposition.u.as_ref()?;
        let singular_values = &decomposition.singular_values;
        let max_singular_value = singular_values.max();
        let max_dim = U_full.nrows().max(U_full.ncols()) as f64;
        let rank_threshold = max_singular_value * max_dim * f64::EPSILON;
        let rank = singular_values
            .iter()
            .filter(|singular_value| **singular_value > rank_threshold)
            .count();

        // nalgebra stores a full U matrix. The VarPro projector must only use
        // the numerical column space of Phi, otherwise U * U^T becomes identity
        // and the reduced Jacobian incorrectly vanishes.
        let U = U_full.columns(0, rank);
        let U_t = U.transpose();

        let result: Result<Vec<()>, Model::Error> = jacobian_matrix
            .column_iter_mut()
            .enumerate()
            .map(|(k, mut jacobian_col)| {
                let Dk = &self.separable_problem.weights
                    * self.separable_problem.model.eval_partial_deriv(k)?;

                let view: MatrixViewMut<f64, Dyn, Dyn, _, _> = jacobian_col.as_view_mut();
                let mut dkc_shaped_jacobian: Matrix<f64, Dyn, Dyn, _> = view
                    .reshape_generic::<Dyn, Dyn>(
                        Dk.shape_generic().0,
                        linear_coefficients.shape_generic().1,
                    );

                if data_cols <= parameter_count {
                    Dk.mul_to(linear_coefficients, &mut dkc_shaped_jacobian);
                    let Ut_DkC = &U_t * &dkc_shaped_jacobian;
                    dkc_shaped_jacobian.gemm(1.0, &U, &Ut_DkC, -1.0);
                } else {
                    let Ut_Dk = &U_t * &Dk;
                    let mut Dk = Dk;
                    Dk.gemm(1.0, &U, &Ut_Dk, -1.0);
                    Dk.mul_to(linear_coefficients, &mut dkc_shaped_jacobian);
                };

                Ok(())
            })
            .collect::<Result<_, _>>();

        result.ok()?;

        // The Kaufman approximation above is written for `data - model`.
        // Our residuals are `model - data`, so the reduced Jacobian needs the
        // opposite sign.
        jacobian_matrix *= -1.0;

        Some(jacobian_matrix)
    }
}
