use crate::numerical::optimization::LM_optimization::LevenbergMarquardt;
use crate::numerical::optimization::problem_LM::LeastSquaresProblem;
use crate::numerical::optimization::varpro::fit::FitResult;
use crate::numerical::optimization::varpro::prelude::*;
use crate::numerical::optimization::varpro::problem::{RhsType, SeparableProblem};

mod levmar_problem;

pub use levmar_problem::LevMarProblem;
pub use levmar_problem::LevMarProblemSvd;
pub use levmar_problem::LinearSolver;
pub use levmar_problem::SvdLinearSolver;

/// Variable projection solver backed by this crate's f64 Levenberg-Marquardt implementation.
///
/// The original upstream `varpro` crate supports several scalar types and optional LAPACK QR
/// backends. This in-tree version is intentionally narrower: it solves f64 problems and uses the
/// SVD backend for the linear subproblem.
#[derive(Debug)]
pub struct LevMarSolver<Model>
where
    Model: SeparableNonlinearModel<ScalarType = f64>,
{
    solver: LevenbergMarquardt,
    _model: std::marker::PhantomData<Model>,
}

impl<Model> LevMarSolver<Model>
where
    Model: SeparableNonlinearModel<ScalarType = f64>,
{
    /// Construct a VarPro solver using an already configured LM solver.
    pub fn with_solver(solver: LevenbergMarquardt) -> Self {
        Self {
            solver,
            _model: Default::default(),
        }
    }

    #[allow(clippy::result_large_err)]
    /// Solve a prepared VarPro LM problem.
    pub fn solve_generic<Rhs, Solver>(
        &self,
        problem: LevMarProblem<Model, Rhs, Solver>,
    ) -> Result<FitResult<Model, Rhs>, FitResult<Model, Rhs>>
    where
        Rhs: RhsType,
        Solver: LinearSolver<ScalarType = f64>,
        LevMarProblem<Model, Rhs, Solver>: LeastSquaresProblem,
    {
        let (problem, report) = self.solver.minimize(problem);

        let LevMarProblem {
            separable_problem,
            cached,
        } = problem;

        let linear_coefficients = cached.map(|cached| cached.linear_coefficients_matrix());
        let result = FitResult::new(separable_problem, linear_coefficients, report);

        if result.was_successful() {
            Ok(result)
        } else {
            Err(result)
        }
    }

    #[allow(clippy::result_large_err)]
    /// Solve using the SVD-based linear subproblem solver.
    pub fn solve_with_svd<Rhs: RhsType>(
        &self,
        problem: SeparableProblem<Model, Rhs>,
    ) -> Result<FitResult<Model, Rhs>, FitResult<Model, Rhs>> {
        self.solve_generic(LevMarProblemSvd::from(problem))
    }

    #[allow(clippy::result_large_err)]
    /// Solve using the default VarPro backend.
    pub fn solve<Rhs: RhsType>(
        &self,
        problem: SeparableProblem<Model, Rhs>,
    ) -> Result<FitResult<Model, Rhs>, FitResult<Model, Rhs>> {
        self.solve_with_svd(problem)
    }
}

impl<Model> Default for LevMarSolver<Model>
where
    Model: SeparableNonlinearModel<ScalarType = f64>,
{
    fn default() -> Self {
        // VarPro residuals are projected through a linear least-squares solve.
        // The projected residual can make the orthogonality test too eager, so
        // the default wrapper lets ftol/xtol drive convergence.
        Self::with_solver(LevenbergMarquardt::default().with_gtol(0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::optimization::varpro::model::builder::SeparableModelBuilder;
    use crate::numerical::optimization::varpro::problem::SeparableProblemBuilder;
    use approx::assert_relative_eq;
    use nalgebra::{DMatrix, DVector};

    fn exp_decay(x: &DVector<f64>, tau: f64) -> DVector<f64> {
        x.map(|value| (-value / tau).exp())
    }

    fn exp_decay_dtau(x: &DVector<f64>, tau: f64) -> DVector<f64> {
        x.map(|value| (-value / tau).exp() * value / tau.powi(2))
    }

    fn linspace(start: f64, end: f64, len: usize) -> DVector<f64> {
        let step = (end - start) / (len - 1) as f64;
        DVector::from_iterator(len, (0..len).map(|idx| start + idx as f64 * step))
    }

    fn constant_offset(x: &DVector<f64>) -> DVector<f64> {
        DVector::from_element(x.len(), 1.0)
    }

    fn double_exponential_model(
        x: DVector<f64>,
        initial_tau1: f64,
        initial_tau2: f64,
    ) -> crate::numerical::optimization::varpro::model::SeparableModel<f64> {
        SeparableModelBuilder::<f64>::new(["tau1", "tau2"])
            .function(["tau1"], exp_decay)
            .partial_deriv("tau1", exp_decay_dtau)
            .function(["tau2"], exp_decay)
            .partial_deriv("tau2", exp_decay_dtau)
            .invariant_function(constant_offset)
            .independent_variable(x)
            .initial_parameters(vec![initial_tau1, initial_tau2])
            .build()
            .expect("A valid double-exponential model should build successfully.")
    }

    #[test]
    fn svd_solver_fits_exponential_with_constant_offset() {
        let x = linspace(0.0, 12.5, 128);
        let tau = 2.0;
        let amplitude = 4.0;
        let offset = 1.0;
        let y = x.map(|value| amplitude * (-value / tau).exp() + offset);

        let model = SeparableModelBuilder::<f64>::new(["tau"])
            .function(["tau"], exp_decay)
            .partial_deriv("tau", exp_decay_dtau)
            .invariant_function(constant_offset)
            .independent_variable(x)
            .initial_parameters(vec![3.0])
            .build()
            .expect("A valid separable model should build successfully.");

        let problem = SeparableProblemBuilder::new(model)
            .observations(y)
            .build()
            .expect("A valid single-RHS VarPro problem should build successfully.");

        let fit = LevMarSolver::default()
            .solve(problem)
            .expect("The SVD-backed VarPro solver should converge on noiseless data.");

        let params = fit.nonlinear_parameters();
        let coeffs = fit
            .linear_coefficients()
            .expect("Successful fitting should expose the linear coefficients.");

        assert_relative_eq!(params[0], tau, epsilon = 1e-6);
        assert_relative_eq!(coeffs[0], amplitude, epsilon = 1e-6);
        assert_relative_eq!(coeffs[1], offset, epsilon = 1e-6);
    }

    #[test]
    fn svd_solver_fits_weighted_exponential_with_constant_offset() {
        let x = linspace(0.0, 10.0, 96);
        let tau = 1.5;
        let amplitude = 2.2;
        let offset = -0.4;
        let y = x.map(|value| amplitude * (-value / tau).exp() + offset);
        let weights = DVector::from_iterator(
            x.len(),
            (0..x.len()).map(|idx| 0.75 + idx as f64 / x.len() as f64),
        );

        let model = SeparableModelBuilder::<f64>::new(["tau"])
            .function(["tau"], exp_decay)
            .partial_deriv("tau", exp_decay_dtau)
            .invariant_function(constant_offset)
            .independent_variable(x)
            .initial_parameters(vec![2.3])
            .build()
            .expect("A valid separable model should build successfully.");

        let problem = SeparableProblemBuilder::new(model)
            .observations(y)
            .weights(weights)
            .build()
            .expect("A valid weighted VarPro problem should build successfully.");

        let fit = LevMarSolver::default()
            .solve(problem)
            .expect("Weighted noiseless data should converge.");
        let params = fit.nonlinear_parameters();
        let coeffs = fit
            .linear_coefficients()
            .expect("Successful fitting should expose the linear coefficients.");

        assert_relative_eq!(params[0], tau, epsilon = 1e-6);
        assert_relative_eq!(coeffs[0], amplitude, epsilon = 1e-6);
        assert_relative_eq!(coeffs[1], offset, epsilon = 1e-6);
    }

    #[test]
    fn svd_solver_fits_double_exponential_with_constant_offset() {
        let x = linspace(0.0, 12.5, 160);
        let tau1 = 1.0;
        let tau2 = 3.0;
        let c1 = 4.0;
        let c2 = 2.5;
        let c3 = 1.0;
        let y = x.map(|value| c1 * (-value / tau1).exp() + c2 * (-value / tau2).exp() + c3);

        let model = double_exponential_model(x, 1.2, 2.8);
        let problem = SeparableProblemBuilder::new(model)
            .observations(y)
            .build()
            .expect("A valid double-exponential problem should build successfully.");

        let fit = LevMarSolver::default()
            .solve(problem)
            .expect("The double-exponential VarPro problem should converge.");
        let params = fit.nonlinear_parameters();
        let coeffs = fit
            .linear_coefficients()
            .expect("Successful fitting should expose the linear coefficients.");

        let (fast_idx, slow_idx) = if params[0] < params[1] {
            (0usize, 1usize)
        } else {
            (1usize, 0usize)
        };

        assert_relative_eq!(params[fast_idx], tau1, epsilon = 1e-6);
        assert_relative_eq!(params[slow_idx], tau2, epsilon = 1e-6);
        assert_relative_eq!(coeffs[fast_idx], c1, epsilon = 1e-6);
        assert_relative_eq!(coeffs[slow_idx], c2, epsilon = 1e-6);
        assert_relative_eq!(coeffs[2], c3, epsilon = 1e-6);
    }

    #[test]
    fn svd_solver_fits_multiple_rhs_with_shared_nonlinear_parameter() {
        let x = linspace(0.0, 8.0, 96);
        let tau = 2.5;
        let mut y = DMatrix::zeros(x.len(), 2);
        y.set_column(0, &x.map(|value| 1.5 * (-value / tau).exp() + 0.2));
        y.set_column(1, &x.map(|value| -0.7 * (-value / tau).exp() + 2.0));

        let model = SeparableModelBuilder::<f64>::new(["tau"])
            .function(["tau"], exp_decay)
            .partial_deriv("tau", exp_decay_dtau)
            .invariant_function(constant_offset)
            .independent_variable(x)
            .initial_parameters(vec![1.8])
            .build()
            .expect("A valid shared-parameter model should build successfully.");

        let problem = SeparableProblemBuilder::mrhs(model)
            .observations(y)
            .build()
            .expect("A valid MRHS VarPro problem should build successfully.");

        let fit = LevMarSolver::default()
            .solve(problem)
            .expect("The MRHS VarPro problem should converge.");
        let params = fit.nonlinear_parameters();
        let coeffs = fit
            .linear_coefficients()
            .expect("Successful fitting should expose the linear coefficient matrix.");

        assert_relative_eq!(params[0], tau, epsilon = 1e-6);
        assert_relative_eq!(coeffs[(0, 0)], 1.5, epsilon = 1e-6);
        assert_relative_eq!(coeffs[(1, 0)], 0.2, epsilon = 1e-6);
        assert_relative_eq!(coeffs[(0, 1)], -0.7, epsilon = 1e-6);
        assert_relative_eq!(coeffs[(1, 1)], 2.0, epsilon = 1e-6);
    }
}
