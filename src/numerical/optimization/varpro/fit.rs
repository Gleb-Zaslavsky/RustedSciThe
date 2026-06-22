use crate::numerical::optimization::LM_optimization::MinimizationReport;
use crate::numerical::optimization::varpro::model::SeparableNonlinearModel;
use crate::numerical::optimization::varpro::problem::{
    MultiRhs, RhsType, SeparableProblem, SingleRhs,
};
use nalgebra::{DMatrix, DMatrixView, DVectorView, Dyn, OMatrix, OVector};
use std::collections::HashMap;

/// A helper type that contains the fitting problem after the
/// minimization, as well as a report and some convenience functions.
///
/// This structure is returned by the [`LevMarSolver::solve`](crate::solvers::levmar::LevMarSolver::solve)
/// and other solve methods of [`LevMarSolver`](crate::solvers::levmar::LevMarSolver).
#[derive(Debug)]
pub struct FitResult<Model, Rhs>
where
    Model: SeparableNonlinearModel<ScalarType = f64>,
    Rhs: RhsType,
{
    /// The final state of the fitting problem after the
    /// minimization finished (regardless of whether fitting was successful or not).
    pub problem: SeparableProblem<Model, Rhs>,

    /// the linear coefficient matrix (in case of single rhs: matrix has
    /// only one column) at the solution.
    pub(crate) linear_coefficients: Option<DMatrix<Model::ScalarType>>,

    /// The minimization report of the underlying solver.
    /// It contains information about the minimization process
    /// and should be queried to see whether the minimization
    /// was considered successful.
    pub minimization_report: MinimizationReport,
}

impl<Model, Rhs> FitResult<Model, Rhs>
where
    Model: SeparableNonlinearModel<ScalarType = f64>,
    Rhs: RhsType,
{
    /// this returns the linear coefficients in matrix form. For a single
    /// right hand side, this is a matrix with one column. Typically,
    /// the [`linear_coefficients`](Self::linear_coefficients) function should be called to get the
    /// appropriate view for single or multiple right hand sides out of
    /// the box. This function exists to overcome some limitations on how
    /// generics work.
    pub fn linear_coefficients_generic(&self) -> Option<DMatrixView<'_, Model::ScalarType>> {
        self.linear_coefficients.as_ref().map(|c| c.as_view())
    }
}

impl<Model> FitResult<Model, MultiRhs>
// take trait bounds from above:
where
    Model: SeparableNonlinearModel<ScalarType = f64>,
{
    /// **Note** This implementation is for fitting problems with multiple right hand sides.
    ///
    /// Convenience function to get the linear coefficients after the fit has
    /// finished. Will return None if there was an error during fitting.
    ///
    /// The coefficients vectors for the individual
    /// members of the datasets are the colums of the returned matrix. That means
    /// one coefficient vector for each right hand side.
    pub fn linear_coefficients(&self) -> Option<DMatrixView<'_, Model::ScalarType>> {
        self.linear_coefficients.as_ref().map(|c| c.as_view())
    }

    /// **Note** This implementation is for fitting problems with multiple right hand sides.
    ///
    /// Calculate the values of the model at the best fit parameters.
    /// Returns None if there was an error during fitting.
    /// Since this is for multiple right hand sides, the output is a matrix
    /// where each column corresponds to one right hand side.
    pub fn best_fit(&self) -> Option<OMatrix<Model::ScalarType, Dyn, Dyn>> {
        let coeff = self.linear_coefficients()?;
        let eval = self.problem.model().eval().ok()?;
        Some(eval * coeff)
    }
}

impl<Model> FitResult<Model, SingleRhs>
// take trait bounds from above:
where
    Model: SeparableNonlinearModel<ScalarType = f64>,
{
    /// **Note** This implementation is for fitting problems with a single right hand side.
    ///
    /// Convenience function to get the linear coefficients after the fit has
    /// finished. Will return None if there was an error during fitting.
    /// The coefficients are given as a single vector.
    pub fn linear_coefficients(&self) -> Option<DVectorView<'_, Model::ScalarType>> {
        let coeff = self.linear_coefficients.as_ref()?;
        debug_assert_eq!(
            coeff.ncols(),
            1,
            "Coefficient matrix must have exactly one colum for problem with single right hand side. This indicates a programming error inside this library!"
        );
        Some(coeff.column(0))
    }
    /// **Note** This implementation is for fitting problems with multiple right hand sides
    ///
    /// Calculate the values of the model at the best fit parameters.
    /// Returns None if there was an error during fitting.
    /// Since this is for a single right hand side, the output is a vector.
    pub fn best_fit(&self) -> Option<OVector<Model::ScalarType, Dyn>> {
        let coeff = self.linear_coefficients()?;
        let eval = self.problem.model().eval().ok()?;
        Some(eval * coeff)
    }

    /// Return all fitted parameters in a single map.
    ///
    /// Nonlinear parameters keep their original symbolic names. Linear
    /// coefficients are exposed as `c0`, `c1`, ... in basis-function order.
    pub fn solution_map(&self) -> Option<HashMap<String, f64>> {
        let nonlinear_names = self.problem.model().parameter_names();
        let nonlinear_values = self.nonlinear_parameters();
        let linear_coefficients = self.linear_coefficients()?;

        let mut map = HashMap::with_capacity(nonlinear_names.len() + linear_coefficients.len());
        for (name, value) in nonlinear_names.iter().zip(nonlinear_values.iter()) {
            map.insert(name.clone(), *value);
        }
        for (index, value) in linear_coefficients.iter().enumerate() {
            map.insert(format!("c{index}"), *value);
        }

        Some(map)
    }

    /// Return the coefficient of determination for the fitted data.
    ///
    /// This uses the problem data stored by the solver. For unweighted
    /// problems this matches the ordinary `r^2` value.
    pub fn r_squared(&self) -> Option<f64> {
        let observed: Vec<f64> = self.problem.weighted_data().iter().copied().collect();
        let predicted: Vec<f64> = self.best_fit()?.iter().copied().collect();

        if observed.len() != predicted.len() || observed.is_empty() {
            return None;
        }

        let mean = observed.iter().sum::<f64>() / observed.len() as f64;
        let ss_tot = observed.iter().map(|y| (y - mean).powi(2)).sum::<f64>();
        if ss_tot == 0.0 {
            return None;
        }
        let ss_res = observed
            .iter()
            .zip(predicted.iter())
            .map(|(y, y_pred)| (y - y_pred).powi(2))
            .sum::<f64>();

        Some(1.0 - ss_res / ss_tot)
    }

    /// Short alias for [`r_squared`](Self::r_squared).
    pub fn r2(&self) -> Option<f64> {
        self.r_squared()
    }
}

impl<Model, Rhs: RhsType> FitResult<Model, Rhs>
// take trait bounds from above:
where
    Model: SeparableNonlinearModel<ScalarType = f64>,
{
    /// internal helper for constructing an instance
    pub(crate) fn new(
        problem: SeparableProblem<Model, Rhs>,
        linear_coefficients: Option<DMatrix<Model::ScalarType>>,
        minimization_report: MinimizationReport,
    ) -> Self {
        Self {
            problem,
            minimization_report,
            linear_coefficients,
        }
    }

    /// convenience function to get the nonlinear parameters of the model after
    /// the fitting process has finished.
    pub fn nonlinear_parameters(&self) -> OVector<Model::ScalarType, Dyn> {
        self.problem.model().params()
    }

    /// whether the fit was deemeed successful. The fit might still be not
    /// be optimal for numerical reasons, but the minimization process
    /// terminated successfully.
    pub fn was_successful(&self) -> bool {
        self.minimization_report.termination.was_successful()
    }
}
