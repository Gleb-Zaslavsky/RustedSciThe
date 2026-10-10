//! Rectangular symbolic frontend for least-squares problems.
//!
//! The backend selection, preparation telemetry, parameter schema, ExprLegacy,
//! AtomNative, Lambdify, and AOT lifecycle remain implemented by the shared
//! `Nonlinear_systems::symbolic` infrastructure. This module only adapts its
//! prepared problem to a least-squares-shaped API where the residual count may
//! differ from the parameter count.

use super::problem::LeastSquaresProblem;
use crate::numerical::Nonlinear_systems::error::SolveError;
use crate::numerical::Nonlinear_systems::problem::{JacobianProvider, NonlinearProblem};
use crate::numerical::Nonlinear_systems::symbolic::{
    BoundSymbolicNonlinearProblem, PreparedSymbolicNonlinearAotProblem,
    PreparedSymbolicNonlinearProblem, SymbolicBackendKind, SymbolicDenseAotOptions,
    SymbolicLambdifyFrontend, SymbolicPreparationReport, SymbolicProblemOptions,
};
use crate::numerical::Nonlinear_systems::symbolic_backend::SymbolicBackendSelectionPolicy;
use crate::symbolic::codegen::codegen_aot_resolution::AotResolver;
use crate::symbolic::symbolic_engine::Expr;
use nalgebra::{DMatrix, DVector};
use std::collections::HashSet;

/// Prepared symbolic rectangular least-squares problem.
///
/// Preparation is delegated to the shared nonlinear symbolic infrastructure;
/// no second Expr/Atom differentiation or frontend-specific compilation is
/// performed here.
pub struct PreparedSymbolicLeastSquaresProblem {
    prepared: PreparedSymbolicNonlinearProblem,
    positive_indices: Vec<usize>,
}

/// Parameter-bound view of a prepared symbolic least-squares problem.
pub struct BoundSymbolicLeastSquaresProblem<'a> {
    bound: BoundSymbolicNonlinearProblem<'a>,
    params: DVector<f64>,
    positive_indices: Vec<usize>,
}

impl PreparedSymbolicLeastSquaresProblem {
    /// Prepares a rectangular symbolic problem through the default policy.
    pub fn from_expressions(
        equations: Vec<Expr>,
        options: SymbolicProblemOptions,
    ) -> Result<Self, SolveError> {
        Ok(Self {
            prepared: PreparedSymbolicNonlinearProblem::from_expressions_rectangular(
                equations, options,
            )?,
            positive_indices: Vec::new(),
        })
    }

    /// Prepares a rectangular symbolic problem from source strings.
    pub fn from_strings(
        equations: Vec<String>,
        options: SymbolicProblemOptions,
    ) -> Result<Self, SolveError> {
        let expressions = equations
            .into_iter()
            .enumerate()
            .map(|(index, equation)| {
                Expr::try_parse_expression(&equation).map_err(|error| {
                    SolveError::InvalidConfig(format!(
                        "failed to parse symbolic equation {index}: {error}"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_expressions(expressions, options)
    }

    /// Applies the explicit Lambdify/AOT selection policy used by the shared backend layer.
    pub fn from_expressions_with_backend_selection(
        equations: Vec<Expr>,
        options: SymbolicProblemOptions,
        policy: SymbolicBackendSelectionPolicy,
        resolver: Option<&AotResolver>,
        aot_options: SymbolicDenseAotOptions,
    ) -> Result<Self, SolveError> {
        Ok(Self {
            prepared:
                PreparedSymbolicNonlinearProblem::from_expressions_with_backend_selection_rectangular(
                    equations,
                    options,
                    policy,
                    resolver,
                    aot_options,
                )?,
            positive_indices: Vec::new(),
        })
    }

    /// Declares selected symbolic variables strictly positive.
    ///
    /// The names are resolved once against the prepared variable schema. A
    /// trial point violating any declared constraint is rejected before a
    /// residual or Jacobian callback is invoked. This is useful for domains
    /// such as `ln(x)` without making the numerical core chemistry-specific.
    pub fn set_positive<S: AsRef<str>>(&mut self, names: &[S]) -> Result<(), SolveError> {
        let mut seen = HashSet::with_capacity(names.len());
        let mut indices = Vec::with_capacity(names.len());
        let variables = self.prepared.variables();
        for name in names {
            let name = name.as_ref();
            if !seen.insert(name) {
                return Err(SolveError::InvalidVariableSchema(format!(
                    "positive variable '{name}' was declared more than once"
                )));
            }
            let Some(index) = variables.iter().position(|variable| variable == name) else {
                return Err(SolveError::InvalidVariableSchema(format!(
                    "positive variable '{name}' is absent from the prepared variable schema"
                )));
            };
            indices.push(index);
        }
        self.positive_indices = indices;
        Ok(())
    }

    /// Returns the immutable preparation report, including backend lifecycle stages.
    pub fn preparation_report(&self) -> &SymbolicPreparationReport {
        self.prepared.preparation_report()
    }

    /// Returns the number of residual equations.
    pub fn residual_count(&self) -> usize {
        self.prepared.as_problem().equations().len()
    }

    /// Returns the number of fitted parameters.
    pub fn parameter_count(&self) -> usize {
        self.prepared.variables().len()
    }

    /// Returns the `(residual_count, parameter_count)` Jacobian shape.
    pub fn jacobian_shape(&self) -> (usize, usize) {
        (self.residual_count(), self.parameter_count())
    }

    /// Returns the selected symbolic backend kind.
    pub fn backend_kind(&self) -> SymbolicBackendKind {
        self.prepared.backend_kind()
    }

    /// Returns the selected Lambdify frontend, when the Lambdify route is active.
    pub fn lambdify_frontend(&self) -> SymbolicLambdifyFrontend {
        self.prepared.lambdify_frontend()
    }

    /// Returns the prepared dense AOT manifest bridge without rebuilding the problem.
    pub fn prepare_dense_aot_problem(
        &self,
        options: SymbolicDenseAotOptions,
    ) -> PreparedSymbolicNonlinearAotProblem<'_> {
        self.prepared.prepare_dense_aot_problem(options)
    }

    /// Binds the initial or configured parameter values.
    pub fn bind_initial(&self) -> Result<BoundSymbolicLeastSquaresProblem<'_>, SolveError> {
        self.bind_initial_with_guess(DVector::zeros(self.parameter_count()))
    }

    /// Binds the prepared problem and installs the initial LM parameter vector.
    pub fn bind_initial_with_guess(
        &self,
        initial_guess: DVector<f64>,
    ) -> Result<BoundSymbolicLeastSquaresProblem<'_>, SolveError> {
        if initial_guess.len() != self.parameter_count() {
            return Err(SolveError::DimensionMismatch {
                expected: self.parameter_count(),
                actual: initial_guess.len(),
                context: "least-squares initial guess",
            });
        }
        for (index, value) in initial_guess.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(SolveError::NonFiniteInitialGuess { index, value });
            }
        }
        for &index in &self.positive_indices {
            let value = initial_guess[index];
            if value <= 0.0 {
                return Err(SolveError::InfeasibleInitialGuess {
                    index,
                    value,
                    lower: 0.0,
                    upper: f64::INFINITY,
                });
            }
        }
        Ok(BoundSymbolicLeastSquaresProblem {
            bound: self.prepared.bind_initial()?,
            params: initial_guess,
            positive_indices: self.positive_indices.clone(),
        })
    }

    /// Binds an explicit parameter vector against the prepared schema.
    pub fn bind_values(
        &self,
        values: DVector<f64>,
    ) -> Result<BoundSymbolicLeastSquaresProblem<'_>, SolveError> {
        self.bind_values_with_guess(values, DVector::zeros(self.parameter_count()))
    }

    /// Binds explicit equation parameters and installs a caller-provided LM
    /// initial guess without repeating symbolic preparation.
    pub fn bind_values_with_guess(
        &self,
        values: DVector<f64>,
        initial_guess: DVector<f64>,
    ) -> Result<BoundSymbolicLeastSquaresProblem<'_>, SolveError> {
        if initial_guess.len() != self.parameter_count() {
            return Err(SolveError::DimensionMismatch {
                expected: self.parameter_count(),
                actual: initial_guess.len(),
                context: "least-squares initial guess",
            });
        }
        if initial_guess.iter().any(|value| !value.is_finite()) {
            return Err(SolveError::NonFiniteInitialGuess {
                index: initial_guess
                    .iter()
                    .position(|value| !value.is_finite())
                    .unwrap_or(0),
                value: initial_guess
                    .iter()
                    .copied()
                    .find(|value| !value.is_finite())
                    .unwrap_or(f64::NAN),
            });
        }
        for &index in &self.positive_indices {
            let value = initial_guess[index];
            if value <= 0.0 {
                return Err(SolveError::InfeasibleInitialGuess {
                    index,
                    value,
                    lower: 0.0,
                    upper: f64::INFINITY,
                });
            }
        }
        Ok(BoundSymbolicLeastSquaresProblem {
            bound: self.prepared.bind_values(values)?,
            params: initial_guess,
            positive_indices: self.positive_indices.clone(),
        })
    }
}

/// Compatibility implementation for the staged LM controller.
///
/// The symbolic methods above preserve typed errors for the new frontend API.
/// The copied legacy LM contract still uses `Option`, so this adapter maps a
/// failed callback to `None` only at that compatibility boundary.
impl<'a> LeastSquaresProblem for BoundSymbolicLeastSquaresProblem<'a> {
    fn set_params(&mut self, x: &DVector<f64>) {
        self.params.copy_from(x);
    }

    fn params(&self) -> DVector<f64> {
        self.params.clone()
    }

    fn residuals(&self) -> Option<DVector<f64>> {
        self.residual(&self.params).ok()
    }

    fn jacobian(&self) -> Option<DMatrix<f64>> {
        self.jacobian(&self.params).ok()
    }

    fn try_residuals(&self) -> Result<DVector<f64>, super::errors::LeastSquaresError> {
        self.residual(&self.params).map_err(Into::into)
    }

    fn try_jacobian(&self) -> Result<DMatrix<f64>, super::errors::LeastSquaresError> {
        self.jacobian(&self.params).map_err(Into::into)
    }

    fn validate_trial(&self, x: &DVector<f64>) -> bool {
        x.len() == self.params.len()
            && x.iter().all(|value| value.is_finite())
            && self
                .positive_indices
                .iter()
                .all(|&index| x.get(index).is_some_and(|value| *value > 0.0))
    }
}

impl<'a> BoundSymbolicLeastSquaresProblem<'a> {
    /// Evaluates the residual vector with typed symbolic/runtime errors preserved.
    pub fn residual(&self, x: &DVector<f64>) -> Result<DVector<f64>, SolveError> {
        NonlinearProblem::residual(&self.bound, x)
    }

    /// Evaluates the dense rectangular Jacobian with typed errors preserved.
    pub fn jacobian(&self, x: &DVector<f64>) -> Result<DMatrix<f64>, SolveError> {
        JacobianProvider::jacobian(&self.bound, x)
    }

    /// Evaluates this prepared model with explicit numeric equation parameters.
    pub fn residual_with_parameter_values(
        &self,
        x: &DVector<f64>,
        parameter_values: &DVector<f64>,
    ) -> Result<DVector<f64>, SolveError> {
        self.bound
            .residual_with_parameter_values(x, parameter_values)
    }

    /// Evaluates the Jacobian with explicit numeric equation parameters.
    pub fn jacobian_with_parameter_values(
        &self,
        x: &DVector<f64>,
        parameter_values: &DVector<f64>,
    ) -> Result<DMatrix<f64>, SolveError> {
        self.bound
            .jacobian_with_parameter_values(x, parameter_values)
    }

    /// Evaluates the residual into caller-owned storage.
    pub fn residual_into(
        &self,
        x: &DVector<f64>,
        out: &mut DVector<f64>,
    ) -> Result<(), SolveError> {
        NonlinearProblem::residual_into(&self.bound, x, out)
    }

    /// Evaluates the Jacobian into caller-owned storage.
    pub fn jacobian_into(
        &self,
        x: &DVector<f64>,
        out: &mut DMatrix<f64>,
    ) -> Result<(), SolveError> {
        JacobianProvider::jacobian_into(&self.bound, x, out)
    }

    /// Fills residual storage with explicit numeric equation parameters.
    pub fn residual_into_with_parameter_values(
        &self,
        x: &DVector<f64>,
        parameter_values: &DVector<f64>,
        out: &mut DVector<f64>,
    ) -> Result<(), SolveError> {
        self.bound
            .residual_into_with_parameter_values(x, parameter_values, out)
    }

    /// Fills Jacobian storage with explicit numeric equation parameters.
    pub fn jacobian_into_with_parameter_values(
        &self,
        x: &DVector<f64>,
        parameter_values: &DVector<f64>,
        out: &mut DMatrix<f64>,
    ) -> Result<(), SolveError> {
        self.bound
            .jacobian_into_with_parameter_values(x, parameter_values, out)
    }

    /// Returns whether the selected backend supports direct residual output.
    pub fn supports_residual_into(&self) -> bool {
        NonlinearProblem::supports_residual_into(&self.bound)
    }

    /// Returns whether the selected backend supports direct Jacobian output.
    pub fn supports_jacobian_into(&self) -> bool {
        JacobianProvider::supports_jacobian_into(&self.bound)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::Nonlinear_systems::error::SolveError;
    use crate::numerical::Nonlinear_systems::least_squares::LevenbergMarquardt;
    use crate::numerical::Nonlinear_systems::symbolic::SymbolicLambdifyFrontend;

    fn equations() -> Vec<Expr> {
        vec![
            Expr::parse_expression("x + y - 1"),
            Expr::parse_expression("2*x - y"),
            Expr::parse_expression("x + 3*y - 2"),
        ]
    }

    fn options() -> SymbolicProblemOptions {
        SymbolicProblemOptions::new()
            .with_variables(vec!["x".to_string(), "y".to_string()])
            .with_lambdify_backend()
    }

    #[test]
    fn rectangular_expr_frontend_preserves_shape_and_evaluates() {
        let prepared =
            PreparedSymbolicLeastSquaresProblem::from_expressions(equations(), options())
                .expect("rectangular Expr frontend should prepare");
        assert_eq!(prepared.jacobian_shape(), (3, 2));
        assert_eq!(prepared.backend_kind(), SymbolicBackendKind::Lambdify);

        let bound = prepared
            .bind_initial()
            .expect("unparameterized problem should bind");
        let x = DVector::from_vec(vec![0.5, 0.25]);
        assert_eq!(bound.residual(&x).expect("residual").len(), 3);
        assert_eq!(bound.jacobian(&x).expect("jacobian").shape(), (3, 2));
    }

    #[test]
    fn rectangular_atom_frontend_uses_shared_backend_preparation() {
        let prepared = PreparedSymbolicLeastSquaresProblem::from_expressions(
            equations(),
            options().with_atom_native_frontend(),
        )
        .expect("rectangular AtomNative frontend should prepare");
        assert_eq!(prepared.jacobian_shape(), (3, 2));
        assert_eq!(
            prepared.lambdify_frontend(),
            SymbolicLambdifyFrontend::AtomViewNative
        );
        let bound = prepared
            .bind_initial()
            .expect("unparameterized problem should bind");
        assert_eq!(
            bound
                .jacobian(&DVector::from_vec(vec![0.5, 0.25]))
                .unwrap()
                .shape(),
            (3, 2)
        );
    }

    #[test]
    fn positive_symbolic_variables_validate_names_and_initial_guess() {
        let equations = vec![Expr::parse_expression("ln(x) - 1")];
        let mut prepared = PreparedSymbolicLeastSquaresProblem::from_expressions(
            equations,
            SymbolicProblemOptions::new()
                .with_variables(vec!["x".to_string()])
                .with_lambdify_backend(),
        )
        .expect("positive-domain symbolic problem should prepare");

        prepared
            .set_positive(&["x"])
            .expect("declared positive variable should resolve");
        assert!(matches!(
            prepared.set_positive(&["missing"]),
            Err(SolveError::InvalidVariableSchema(_))
        ));
        assert!(matches!(
            prepared.set_positive(&["x", "x"]),
            Err(SolveError::InvalidVariableSchema(_))
        ));
        assert!(matches!(
            prepared.bind_initial_with_guess(DVector::from_vec(vec![0.0])),
            Err(SolveError::InfeasibleInitialGuess { index: 0, .. })
        ));
        assert!(matches!(
            prepared.bind_initial_with_guess(DVector::from_vec(vec![f64::NAN])),
            Err(SolveError::NonFiniteInitialGuess { index: 0, .. })
        ));
    }

    #[test]
    fn positive_symbolic_trial_is_rejected_before_log_callback() {
        let mut prepared = PreparedSymbolicLeastSquaresProblem::from_expressions(
            vec![Expr::parse_expression("ln(x) - 1")],
            SymbolicProblemOptions::new()
                .with_variables(vec!["x".to_string()])
                .with_lambdify_backend(),
        )
        .expect("positive-domain symbolic problem should prepare");
        prepared
            .set_positive(&["x"])
            .expect("positive variable should resolve");
        let bound = prepared
            .bind_initial_with_guess(DVector::from_vec(vec![0.5]))
            .expect("positive initial guess should bind");

        assert!(bound.validate_trial(&DVector::from_vec(vec![0.25])));
        assert!(!bound.validate_trial(&DVector::from_vec(vec![0.0])));
        assert!(!bound.validate_trial(&DVector::from_vec(vec![-1.0])));
        assert!(!bound.validate_trial(&DVector::from_vec(vec![f64::INFINITY])));
    }

    #[test]
    fn positive_symbolic_problem_solves_with_staged_lambdify_frontend() {
        let mut prepared = PreparedSymbolicLeastSquaresProblem::from_expressions(
            vec![Expr::parse_expression("ln(x) - 1")],
            SymbolicProblemOptions::new()
                .with_variables(vec!["x".to_string()])
                .with_lambdify_backend(),
        )
        .expect("positive-domain symbolic problem should prepare");
        prepared
            .set_positive(&["x"])
            .expect("positive variable should resolve");
        let bound = prepared
            .bind_initial_with_guess(DVector::from_vec(vec![0.5]))
            .expect("positive initial guess should bind");

        let (result, report) = LevenbergMarquardt::new().with_tol(1e-10).minimize(bound);
        assert!(report.termination.was_successful());
        assert!((result.params()[0] - std::f64::consts::E).abs() < 1e-7);
    }
}
