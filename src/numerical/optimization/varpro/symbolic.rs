//! Symbolic helpers for building VarPro basis functions.
//!
//! This module is intentionally a thin frontend over the existing VarPro model
//! API. It converts symbolic expressions into native f64 closures, while the
//! actual fitting still goes through `SeparableModelBuilder`, `SeparableProblem`
//! and the standard VarPro solver.

use nalgebra::DVector;
use thiserror::Error as ThisError;

use crate::numerical::optimization::varpro::fit::FitResult;
use crate::numerical::optimization::varpro::model::SeparableModel;
use crate::numerical::optimization::varpro::model::builder::SeparableModelBuilder;
use crate::numerical::optimization::varpro::model::builder::error::ModelBuildError;
use crate::numerical::optimization::varpro::problem::{
    SeparableProblemBuilder, SeparableProblemBuilderError, SingleRhs,
};
use crate::numerical::optimization::varpro::solvers::levmar::LevMarSolver;
use crate::symbolic::symbolic_engine::Expr;

/// Boxed VarPro basis function with one nonlinear scalar parameter.
pub type UnaryBasisFn = Box<dyn Fn(&DVector<f64>, f64) -> DVector<f64> + Send + Sync>;

/// Boxed VarPro invariant basis function that depends only on the independent variable.
pub type InvariantBasisFn = Box<dyn Fn(&DVector<f64>) -> DVector<f64> + Send + Sync>;

/// Fit result returned by the high-level symbolic VarPro builder.
pub type SymbolicVarProFit = FitResult<SeparableModel<f64>, SingleRhs>;

/// Errors produced while building or solving a symbolic VarPro problem.
#[derive(Debug, ThisError)]
pub enum SymbolicVarProError {
    /// No independent variable data was supplied.
    #[error("independent variable data is missing")]
    XDataMissing,

    /// No observed data was supplied.
    #[error("observed data is missing")]
    YDataMissing,

    /// The symbolic model has no nonlinear parameter names.
    #[error("at least one nonlinear parameter name is required")]
    ParameterNamesMissing,

    /// The symbolic model has no initial nonlinear parameter values.
    #[error("initial nonlinear parameter values are missing")]
    InitialParametersMissing,

    /// The symbolic model has no basis functions.
    #[error("at least one symbolic basis function is required")]
    BasisFunctionsMissing,

    /// Model construction failed before the nonlinear least-squares problem was built.
    #[error("failed to build separable model: {0}")]
    ModelBuild(#[from] ModelBuildError),

    /// Problem construction failed after the separable model was built.
    #[error("failed to build separable problem: {0}")]
    ProblemBuild(#[from] SeparableProblemBuilderError),

    /// The underlying nonlinear solver returned an unsuccessful report.
    #[error("symbolic VarPro solver did not converge: {termination}")]
    SolverFailed {
        /// Debug representation of the solver termination reason.
        termination: String,
    },
}

#[derive(Clone, Debug)]
struct UnaryBasisSpec {
    parameter_name: String,
    expression: Expr,
}

/// High-level builder for symbolic single-RHS VarPro fitting problems.
///
/// This wrapper follows the style of `sym_fitting`: the user describes the
/// symbolic model and the data in one fluent chain, while the builder creates
/// VarPro basis closures, symbolic derivatives, a separable model, a fitting
/// problem and the default Levenberg-Marquardt solver under the hood.
///
/// The current API intentionally supports one nonlinear parameter per symbolic
/// basis function. That covers common models such as sums of exponentials:
/// `c1 * exp(-x/tau1) + c2 * exp(-x/tau2) + c3`. Basis functions that depend on
/// multiple nonlinear parameters should use future arity-specific helpers.
#[derive(Clone, Debug)]
pub struct SymbolicVarProBuilder {
    x_name: String,
    parameter_names: Vec<String>,
    initial_parameters: Vec<f64>,
    x_data: Option<DVector<f64>>,
    y_data: Option<DVector<f64>>,
    basis_functions: Vec<UnaryBasisSpec>,
    invariant_functions: Vec<Expr>,
}

impl SymbolicVarProBuilder {
    /// Create an empty symbolic VarPro builder.
    ///
    /// `x_name` is the name used by symbolic expressions for the independent
    /// variable, for example `"x"` in `exp(-x/tau)`.
    pub fn new(x_name: impl Into<String>) -> Self {
        Self {
            x_name: x_name.into(),
            parameter_names: Vec::new(),
            initial_parameters: Vec::new(),
            x_data: None,
            y_data: None,
            basis_functions: Vec::new(),
            invariant_functions: Vec::new(),
        }
    }

    /// Set the full ordered list of nonlinear parameter names.
    ///
    /// The order must match the order of values passed to
    /// [`with_initial_parameters`](Self::with_initial_parameters).
    pub fn with_parameters<Names>(mut self, parameter_names: Names) -> Self
    where
        Names: IntoIterator,
        Names::Item: AsRef<str>,
    {
        self.parameter_names = parameter_names
            .into_iter()
            .map(|name| name.as_ref().to_string())
            .collect();
        self
    }

    /// Alias for [`with_parameters`](Self::with_parameters).
    ///
    /// This keeps the symbolic VarPro API closer to the fitting-oriented style
    /// used elsewhere in the crate.
    pub fn with_unknowns<Names>(self, unknowns: Names) -> Self
    where
        Names: IntoIterator,
        Names::Item: AsRef<str>,
    {
        self.with_parameters(unknowns)
    }

    /// Set the initial guess for all nonlinear parameters.
    pub fn with_initial_parameters(mut self, initial_parameters: Vec<f64>) -> Self {
        self.initial_parameters = initial_parameters;
        self
    }

    /// Set independent variable and observed data vectors together.
    pub fn with_data(mut self, x_data: DVector<f64>, y_data: DVector<f64>) -> Self {
        self.x_data = Some(x_data);
        self.y_data = Some(y_data);
        self
    }

    /// Set independent variable data.
    pub fn with_x_data(mut self, x_data: DVector<f64>) -> Self {
        self.x_data = Some(x_data);
        self
    }

    /// Set observed data.
    pub fn with_y_data(mut self, y_data: DVector<f64>) -> Self {
        self.y_data = Some(y_data);
        self
    }

    /// Set the symbolic name of the independent variable.
    ///
    /// This is the variable that appears inside symbolic basis expressions,
    /// for example `x` in `exp(-x/tau)`.
    pub fn with_x_name(mut self, x_name: impl Into<String>) -> Self {
        self.x_name = x_name.into();
        self
    }

    /// Alias for [`with_x_name`](Self::with_x_name).
    pub fn with_arg(self, x_name: impl Into<String>) -> Self {
        self.with_x_name(x_name)
    }

    /// Add a symbolic target equation set.
    ///
    /// For VarPro the target usually comes from basis functions plus invariant
    /// background terms, so this alias simply treats the expressions as basis
    /// functions with no extra nonlinear parameters.
    pub fn with_equations(mut self, equations: Vec<Expr>) -> Self {
        self.invariant_functions.extend(equations);
        self
    }

    /// Alias for [`with_equations`](Self::with_equations) that accepts strings.
    pub fn with_equations_str(self, equations: Vec<String>) -> Self {
        let equations = equations
            .into_iter()
            .map(|expr| Expr::parse_expression(&expr))
            .collect();
        self.with_equations(equations)
    }

    /// Add a one-parameter symbolic basis function from an already parsed expression.
    ///
    /// For example, `parameter_name = "tau"` and
    /// `expression = Expr::parse_expression("exp(-x/tau)")`.
    pub fn with_basis_expr(mut self, parameter_name: impl Into<String>, expression: Expr) -> Self {
        self.basis_functions.push(UnaryBasisSpec {
            parameter_name: parameter_name.into(),
            expression,
        });
        self
    }

    /// Add a one-parameter symbolic basis function from a string expression.
    pub fn with_basis_str(self, parameter_name: impl Into<String>, expression: &str) -> Self {
        self.with_basis_expr(parameter_name, Expr::parse_expression(expression))
    }

    /// Add an invariant symbolic basis function from an already parsed expression.
    ///
    /// Invariant basis functions depend only on the independent variable, or are
    /// constants. They are useful for offsets and background terms.
    pub fn with_invariant_expr(mut self, expression: Expr) -> Self {
        self.invariant_functions.push(expression);
        self
    }

    /// Add an invariant symbolic basis function from a string expression.
    pub fn with_invariant_str(self, expression: &str) -> Self {
        self.with_invariant_expr(Expr::parse_expression(expression))
    }

    /// Build the underlying separable model without solving it.
    pub fn build_model(&self) -> Result<SeparableModel<f64>, SymbolicVarProError> {
        self.validate()?;

        let mut model_builder = SeparableModelBuilder::<f64>::new(self.parameter_names.clone());

        for basis_spec in &self.basis_functions {
            let (basis, derivative) = unary_basis_and_derivative(
                basis_spec.expression.clone(),
                &self.x_name,
                &basis_spec.parameter_name,
            );
            model_builder = model_builder
                .function([basis_spec.parameter_name.as_str()], basis)
                .partial_deriv(&basis_spec.parameter_name, derivative);
        }

        for invariant in &self.invariant_functions {
            model_builder =
                model_builder.invariant_function(invariant_basis(invariant.clone(), &self.x_name));
        }

        let x_data = self
            .x_data
            .clone()
            .ok_or(SymbolicVarProError::XDataMissing)?;

        Ok(model_builder
            .independent_variable(x_data)
            .initial_parameters(self.initial_parameters.clone())
            .build()?)
    }

    /// Build the model and solve the symbolic VarPro fitting problem.
    pub fn solve(self) -> Result<SymbolicVarProFit, SymbolicVarProError> {
        let y_data = self
            .y_data
            .clone()
            .ok_or(SymbolicVarProError::YDataMissing)?;
        let model = self.build_model()?;
        let problem = SeparableProblemBuilder::new(model)
            .observations(y_data)
            .build()?;

        LevMarSolver::default()
            .solve(problem)
            .map_err(|fit| SymbolicVarProError::SolverFailed {
                termination: format!("{:?}", fit.minimization_report.termination),
            })
    }

    /// Alias for [`solve`](Self::solve) for users coming from fitting-style APIs.
    pub fn fit(self) -> Result<SymbolicVarProFit, SymbolicVarProError> {
        self.solve()
    }

    fn validate(&self) -> Result<(), SymbolicVarProError> {
        if self.x_data.is_none() {
            return Err(SymbolicVarProError::XDataMissing);
        }
        if self.y_data.is_none() {
            return Err(SymbolicVarProError::YDataMissing);
        }
        if self.parameter_names.is_empty() {
            return Err(SymbolicVarProError::ParameterNamesMissing);
        }
        if self.initial_parameters.is_empty() {
            return Err(SymbolicVarProError::InitialParametersMissing);
        }
        if self.basis_functions.is_empty() {
            return Err(SymbolicVarProError::BasisFunctionsMissing);
        }

        Ok(())
    }
}

/// Build a VarPro basis function and its symbolic derivative.
///
/// The expression must use `x_name` for the independent variable and
/// `parameter_name` for the nonlinear parameter. The returned pair is ordered as
/// `(basis_function, derivative_with_respect_to_parameter)`, so it can be passed
/// directly to `SeparableModelBuilder::function(...).partial_deriv(...)`.
///
/// This helper covers the common separable fitting case
/// `basis(x, alpha)`, for example `exp(-x/tau)`. Multi-parameter basis
/// functions should be added as explicit arity-specific helpers rather than as
/// a runtime variadic closure, because the VarPro builder intentionally uses
/// Rust function arity to validate user models.
pub fn unary_basis_and_derivative(
    expression: Expr,
    x_name: &str,
    parameter_name: &str,
) -> (UnaryBasisFn, UnaryBasisFn) {
    let derivative = expression.diff(parameter_name);
    (
        unary_basis(expression, x_name, parameter_name),
        unary_basis(derivative, x_name, parameter_name),
    )
}

/// Build a VarPro basis function from a symbolic expression with one parameter.
///
/// The generated closure evaluates the expression for each element of `x` using
/// the variable order `[x_name, parameter_name]`.
pub fn unary_basis(expression: Expr, x_name: &str, parameter_name: &str) -> UnaryBasisFn {
    let evaluator = expression.lambdify_borrowed_thread_safe(&[x_name, parameter_name]);

    Box::new(move |x: &DVector<f64>, parameter: f64| {
        x.map(|x_value| evaluator(&[x_value, parameter]))
    })
}

/// Build an invariant VarPro basis function from a symbolic expression.
///
/// The expression may depend only on `x_name` or be a constant. Invariant
/// functions are useful for offsets and polynomial background terms.
pub fn invariant_basis(expression: Expr, x_name: &str) -> InvariantBasisFn {
    let evaluator = expression.lambdify_borrowed_thread_safe(&[x_name]);

    Box::new(move |x: &DVector<f64>| x.map(|x_value| evaluator(&[x_value])))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::optimization::varpro::model::builder::SeparableModelBuilder;
    use crate::numerical::optimization::varpro::problem::SeparableProblemBuilder;
    use crate::numerical::optimization::varpro::solvers::levmar::LevMarSolver;
    use approx::assert_relative_eq;

    fn linspace(start: f64, end: f64, len: usize) -> DVector<f64> {
        let step = (end - start) / (len - 1) as f64;
        DVector::from_iterator(len, (0..len).map(|idx| start + idx as f64 * step))
    }

    #[test]
    fn unary_symbolic_basis_matches_expected_derivative() {
        let expression = Expr::parse_expression("exp(-x/tau)");
        let (basis, derivative) = unary_basis_and_derivative(expression, "x", "tau");
        let x = DVector::from_vec(vec![0.0, 1.5, 3.0]);
        let tau = 2.0;

        let values = basis(&x, tau);
        let derivatives = derivative(&x, tau);

        for idx in 0..x.len() {
            let expected_value = (-x[idx] / tau).exp();
            let expected_derivative = expected_value * x[idx] / tau.powi(2);
            assert_relative_eq!(values[idx], expected_value, epsilon = 1e-12);
            assert_relative_eq!(derivatives[idx], expected_derivative, epsilon = 1e-12);
        }
    }

    #[test]
    fn varpro_solver_fits_symbolic_exponential_with_offset() {
        let x = linspace(0.0, 10.0, 96);
        let tau = 1.7;
        let amplitude = 3.5;
        let offset = -0.25;
        let y = x.map(|value| amplitude * (-value / tau).exp() + offset);

        let (exp_decay, exp_decay_dtau) =
            unary_basis_and_derivative(Expr::parse_expression("exp(-x/tau)"), "x", "tau");
        let constant_offset = invariant_basis(Expr::parse_expression("1"), "x");

        let model = SeparableModelBuilder::<f64>::new(["tau"])
            .function(["tau"], exp_decay)
            .partial_deriv("tau", exp_decay_dtau)
            .invariant_function(constant_offset)
            .independent_variable(x)
            .initial_parameters(vec![2.4])
            .build()
            .expect("A valid symbolic VarPro model should build successfully.");

        let problem = SeparableProblemBuilder::new(model)
            .observations(y)
            .build()
            .expect("A valid symbolic VarPro problem should build successfully.");

        let fit = LevMarSolver::default()
            .solve(problem)
            .expect("The symbolic VarPro model should converge on noiseless data.");
        let params = fit.nonlinear_parameters();
        let coeffs = fit
            .linear_coefficients()
            .expect("Successful fitting should expose linear coefficients.");

        assert_relative_eq!(params[0], tau, epsilon = 1e-6);
        assert_relative_eq!(coeffs[0], amplitude, epsilon = 1e-6);
        assert_relative_eq!(coeffs[1], offset, epsilon = 1e-6);
    }

    #[test]
    fn builder_api_fits_symbolic_exponential_with_offset() {
        let x = linspace(0.0, 10.0, 96);
        let tau = 1.7;
        let amplitude = 3.5;
        let offset = -0.25;
        let y = x.map(|value| amplitude * (-value / tau).exp() + offset);

        let fit = SymbolicVarProBuilder::new("x")
            .with_parameters(["tau"])
            .with_initial_parameters(vec![2.4])
            .with_data(x, y)
            .with_basis_str("tau", "exp(-x/tau)")
            .with_invariant_str("1")
            .fit()
            .expect("The high-level symbolic VarPro API should converge.");

        let params = fit.nonlinear_parameters();
        let coeffs = fit
            .linear_coefficients()
            .expect("Successful fitting should expose linear coefficients.");

        assert_relative_eq!(params[0], tau, epsilon = 1e-6);
        assert_relative_eq!(coeffs[0], amplitude, epsilon = 1e-6);
        assert_relative_eq!(coeffs[1], offset, epsilon = 1e-6);
    }

    #[test]
    fn builder_api_fits_symbolic_double_exponential_with_offset() {
        let x = linspace(0.0, 12.0, 144);
        let tau_fast = 0.9;
        let tau_slow = 3.2;
        let c_fast = 2.75;
        let c_slow = -1.4;
        let offset = 0.65;
        let y = x.map(|value| {
            c_fast * (-value / tau_fast).exp() + c_slow * (-value / tau_slow).exp() + offset
        });

        let fit = SymbolicVarProBuilder::new("x")
            .with_parameters(["tau_fast", "tau_slow"])
            .with_initial_parameters(vec![1.1, 2.9])
            .with_data(x, y)
            .with_basis_str("tau_fast", "exp(-x/tau_fast)")
            .with_basis_str("tau_slow", "exp(-x/tau_slow)")
            .with_invariant_str("1")
            .solve()
            .expect("The builder should fit a two-basis symbolic VarPro model.");

        let params = fit.nonlinear_parameters();
        let coeffs = fit
            .linear_coefficients()
            .expect("Successful fitting should expose linear coefficients.");

        assert_relative_eq!(params[0], tau_fast, epsilon = 1e-6);
        assert_relative_eq!(params[1], tau_slow, epsilon = 1e-6);
        assert_relative_eq!(coeffs[0], c_fast, epsilon = 1e-6);
        assert_relative_eq!(coeffs[1], c_slow, epsilon = 1e-6);
        assert_relative_eq!(coeffs[2], offset, epsilon = 1e-6);
    }

    #[test]
    fn symbolic_varpro_exposes_solution_map_and_r2() {
        let x = linspace(0.0, 10.0, 96);
        let tau = 1.7;
        let amplitude = 3.5;
        let offset = -0.25;
        let y = x.map(|value| amplitude * (-value / tau).exp() + offset);

        let fit = SymbolicVarProBuilder::new("x")
            .with_parameters(["tau"])
            .with_initial_parameters(vec![2.4])
            .with_data(x, y)
            .with_basis_str("tau", "exp(-x/tau)")
            .with_invariant_str("1")
            .fit()
            .expect("The high-level symbolic VarPro API should converge.");

        let map = fit
            .solution_map()
            .expect("A successful fit should expose a map.");
        assert_relative_eq!(map["tau"], tau, epsilon = 1e-6);
        assert_relative_eq!(map["c0"], amplitude, epsilon = 1e-6);
        assert_relative_eq!(map["c1"], offset, epsilon = 1e-6);
        assert_relative_eq!(fit.r_squared().unwrap(), 1.0, epsilon = 1e-10);
    }
}
