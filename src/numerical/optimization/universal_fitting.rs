//! High-level fitting facade over the symbolic LM and VarPro frontends.
//!
//! `UniversalFitting` is a small depot-style wrapper that lets callers choose
//! the backend late in the builder chain. By default it uses the classic
//! symbolic Levenberg-Marquardt path from `sym_fitting`. If `with_varpro()` or
//! `with_method(Method::VARPRO)` is used, the same wrapper dispatches to the
//! symbolic VarPro frontend.

use crate::numerical::optimization::sym_fitting::Fitting;
use crate::numerical::optimization::varpro::symbolic::{
    SymbolicVarProBuilder, SymbolicVarProError, SymbolicVarProFit,
};
use crate::symbolic::symbolic_engine::Expr;
use nalgebra::DVector;
use std::collections::HashMap;
use thiserror::Error;

/// Selects the fitting backend used by [`UniversalFitting`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// Classic symbolic Levenberg-Marquardt fitting.
    LM,
    /// Symbolic variable projection fitting.
    VARPRO,
}

/// Result returned by [`UniversalFitting::build`].
pub enum UniversalFittingResult {
    /// The classic symbolic Levenberg-Marquardt pipeline.
    LM(Fitting),
    /// The symbolic VarPro pipeline.
    VARPRO(SymbolicVarProFit),
}

impl UniversalFittingResult {
    /// Return the fitted parameter map regardless of backend.
    ///
    /// LM returns its symbolic parameter map directly. VarPro returns the
    /// nonlinear parameters plus linear coefficients labeled `c0`, `c1`, ...
    pub fn solution_map(&self) -> Option<HashMap<String, f64>> {
        match self {
            UniversalFittingResult::LM(fitting) => fitting.solution_map(),
            UniversalFittingResult::VARPRO(fit) => fit.solution_map(),
        }
    }

    /// Return the coefficient of determination for the fit.
    ///
    /// For unweighted problems this is the usual `r^2` quality metric.
    pub fn r_squared(&self) -> Option<f64> {
        match self {
            UniversalFittingResult::LM(fitting) => fitting.r_squared(),
            UniversalFittingResult::VARPRO(fit) => fit.r_squared(),
        }
    }

    /// Short alias for [`r_squared`](Self::r_squared).
    pub fn r2(&self) -> Option<f64> {
        self.r_squared()
    }
}

/// Errors returned by the universal fitting wrapper.
#[derive(Debug, Error)]
pub enum UniversalFittingError {
    /// The VarPro backend failed while building or solving the problem.
    #[error(transparent)]
    VarPro(#[from] SymbolicVarProError),
}

/// Depot-style wrapper over the symbolic LM and VarPro builders.
///
/// Common configuration such as data, unknown names, initial guesses and the
/// independent variable name is mirrored into both backends. Backend-specific
/// model description still uses the methods of the corresponding solver:
/// `with_equation*` for LM and `with_basis_*` / `with_invariant_*` for VarPro.
pub struct UniversalFitting {
    method: Method,
    equations: Vec<Expr>,
    lm: Fitting,
    varpro: SymbolicVarProBuilder,
}

impl UniversalFitting {
    /// Create a new wrapper with the classic LM backend selected.
    pub fn new() -> Self {
        Self {
            method: Method::LM,
            equations: Vec::new(),
            lm: Fitting::new(),
            varpro: SymbolicVarProBuilder::new("x"),
        }
    }

    /// Select the fitting backend explicitly.
    pub fn with_method(mut self, method: Method) -> Self {
        self.method = method;
        self
    }

    /// Shortcut for [`with_method`](Self::with_method) with [`Method::VARPRO`].
    pub fn with_varpro(self) -> Self {
        self.with_method(Method::VARPRO)
    }

    /// Set the independent variable data.
    pub fn with_x_data(mut self, x_data: Vec<f64>) -> Self {
        self.varpro = self.varpro.with_x_data(DVector::from_vec(x_data.clone()));
        self.lm = self.lm.with_x_data(x_data);
        self
    }

    /// Set the observed data.
    pub fn with_y_data(mut self, y_data: Vec<f64>) -> Self {
        self.varpro = self.varpro.with_y_data(DVector::from_vec(y_data.clone()));
        self.lm = self.lm.with_y_data(y_data);
        self
    }

    /// Set independent variable and observed data together.
    pub fn with_data(mut self, x_data: Vec<f64>, y_data: Vec<f64>) -> Self {
        self.varpro = self.varpro.with_data(
            DVector::from_vec(x_data.clone()),
            DVector::from_vec(y_data.clone()),
        );
        self.lm = self.lm.with_data(x_data, y_data);
        self
    }

    /// Set the unknown names for both backends.
    pub fn with_unknowns(mut self, unknowns: Vec<String>) -> Self {
        self.varpro = self.varpro.with_parameters(unknowns.clone());
        self.lm = self.lm.with_unknowns(unknowns);
        self
    }

    /// Alias for [`with_unknowns`](Self::with_unknowns).
    pub fn with_parameters(self, parameters: Vec<String>) -> Self {
        self.with_unknowns(parameters)
    }

    /// Set the initial guess for both backends.
    pub fn with_initial_guess(mut self, initial_guess: Vec<f64>) -> Self {
        self.varpro = self.varpro.with_initial_parameters(initial_guess.clone());
        self.lm = self.lm.with_initial_guess(initial_guess);
        self
    }

    /// Set the symbolic name of the independent variable.
    pub fn with_arg(mut self, arg: String) -> Self {
        self.varpro = self.varpro.with_arg(arg.clone());
        self.lm = self.lm.with_arg(arg);
        self
    }

    /// Set one symbolic model expression.
    ///
    /// For the LM backend, the expressions are solved as residual equations.
    /// For the VarPro backend, they are treated as invariant/background terms.
    pub fn with_equation(mut self, eq: Expr) -> Self {
        self.equations = vec![eq];
        self
    }

    /// Set a symbolic expression system.
    ///
    /// For the LM backend, the expressions are solved as residual equations.
    /// For the VarPro backend, they are treated as invariant/background terms.
    pub fn with_equations(mut self, eq_system: Vec<Expr>) -> Self {
        self.equations = eq_system;
        self
    }

    /// Set one symbolic model expression from a string.
    pub fn with_equation_str(mut self, eq_string: String) -> Self {
        self.equations = vec![Expr::parse_expression(&eq_string)];
        self
    }

    /// Set a symbolic expression system from strings.
    pub fn with_equations_str(mut self, eq_system_string: Vec<String>) -> Self {
        self.equations = eq_system_string
            .into_iter()
            .map(|expr| Expr::parse_expression(&expr))
            .collect();
        self
    }

    /// Set the LM polynomial model.
    pub fn with_polynomial(mut self, degree: usize, arg: String) -> Self {
        self.lm = self.lm.with_polynomial(degree, arg);
        self
    }

    /// Set a symbolic basis function for VarPro.
    pub fn with_basis_expr(mut self, parameter_name: impl Into<String>, expression: Expr) -> Self {
        self.varpro = self.varpro.with_basis_expr(parameter_name, expression);
        self
    }

    /// Set a symbolic basis function for VarPro from a string expression.
    pub fn with_basis_str(self, parameter_name: impl Into<String>, expression: &str) -> Self {
        self.with_basis_expr(parameter_name, Expr::parse_expression(expression))
    }

    /// Mirror the LM tolerance setting.
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.lm = self.lm.with_tolerance(tolerance);
        self
    }

    /// Mirror the LM function tolerance setting.
    pub fn with_f_tolerance(mut self, f_tolerance: f64) -> Self {
        self.lm = self.lm.with_f_tolerance(f_tolerance);
        self
    }

    /// Mirror the LM gradient tolerance setting.
    pub fn with_g_tolerance(mut self, g_tolerance: f64) -> Self {
        self.lm = self.lm.with_g_tolerance(g_tolerance);
        self
    }

    /// Mirror the LM diagonal scaling setting.
    pub fn with_scale_diag(mut self, scale_diag: bool) -> Self {
        self.lm = self.lm.with_scale_diag(scale_diag);
        self
    }

    /// Mirror the LM iteration budget.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.lm = self.lm.with_max_iterations(max_iterations);
        self
    }

    /// Build and solve the selected backend.
    pub fn build(self) -> Result<UniversalFittingResult, UniversalFittingError> {
        match self.method {
            Method::LM => {
                let lm = self.lm.with_equations(self.equations);
                Ok(UniversalFittingResult::LM(lm.build()))
            }
            Method::VARPRO => {
                let varpro = self.varpro.with_equations(self.equations);
                Ok(UniversalFittingResult::VARPRO(varpro.solve()?))
            }
        }
    }
}

impl Default for UniversalFitting {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn universal_lm_fits_linear_model() {
        let x_data = (0..20).map(|x| x as f64).collect::<Vec<_>>();
        let y_data = x_data.iter().map(|&x| 3.0 * x + 2.0).collect::<Vec<_>>();

        let result = UniversalFitting::new()
            .with_data(x_data, y_data)
            .with_equation_str("a*x + b".to_string())
            .with_arg("x".to_string())
            .with_unknowns(vec!["a".to_string(), "b".to_string()])
            .with_initial_guess(vec![1.0, 1.0])
            .build()
            .unwrap();

        let fitting = match result {
            UniversalFittingResult::LM(fitting) => fitting,
            UniversalFittingResult::VARPRO(_) => panic!("expected LM result"),
        };

        let map = fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map["a"], 3.0, epsilon = 1e-6);
        assert_relative_eq!(map["b"], 2.0, epsilon = 1e-6);
    }

    #[test]
    fn universal_lm_exposes_map_and_r2() {
        let x_data = (0..20).map(|x| x as f64).collect::<Vec<_>>();
        let y_data = x_data.iter().map(|&x| 3.0 * x + 2.0).collect::<Vec<_>>();

        let result = UniversalFitting::new()
            .with_data(x_data, y_data)
            .with_equation_str("a*x + b".to_string())
            .with_arg("x".to_string())
            .with_unknowns(vec!["a".to_string(), "b".to_string()])
            .with_initial_guess(vec![1.0, 1.0])
            .build()
            .unwrap();

        let map = result.solution_map().unwrap();
        assert_relative_eq!(map["a"], 3.0, epsilon = 1e-6);
        assert_relative_eq!(map["b"], 2.0, epsilon = 1e-6);
        assert_relative_eq!(result.r_squared().unwrap(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn universal_varpro_fits_symbolic_exponential_with_offset() {
        let x_data = (0..25).map(|x| x as f64 * 0.2).collect::<Vec<_>>();
        let y_data = x_data
            .iter()
            .map(|&x| 2.5 * (-x / 1.7).exp() + 0.8)
            .collect::<Vec<_>>();

        let result = UniversalFitting::new()
            .with_varpro()
            .with_data(x_data, y_data)
            .with_arg("x".to_string())
            .with_parameters(vec!["tau".to_string()])
            .with_initial_guess(vec![1.2])
            .with_basis_str("tau", "exp(-x/tau)")
            .with_equation_str("1".to_string())
            .build()
            .unwrap();

        let fit = match result {
            UniversalFittingResult::VARPRO(fit) => fit,
            UniversalFittingResult::LM(_) => panic!("expected VarPro result"),
        };

        let nonlinear = fit.nonlinear_parameters();
        assert_relative_eq!(nonlinear[0], 1.7, epsilon = 1e-2);

        let linear = fit.linear_coefficients().unwrap();
        assert_relative_eq!(linear[0], 2.5, epsilon = 1e-2);
        assert_relative_eq!(linear[1], 0.8, epsilon = 1e-2);
    }

    #[test]
    fn universal_varpro_exposes_map_and_r2() {
        let x_data = (0..25).map(|x| x as f64 * 0.2).collect::<Vec<_>>();
        let y_data = x_data
            .iter()
            .map(|&x| 2.5 * (-x / 1.7).exp() + 0.8)
            .collect::<Vec<_>>();

        let result = UniversalFitting::new()
            .with_varpro()
            .with_data(x_data, y_data)
            .with_arg("x".to_string())
            .with_parameters(vec!["tau".to_string()])
            .with_initial_guess(vec![1.2])
            .with_basis_str("tau", "exp(-x/tau)")
            .with_equation_str("1".to_string())
            .build()
            .unwrap();

        let map = result.solution_map().unwrap();
        assert_relative_eq!(map["tau"], 1.7, epsilon = 1e-2);
        assert_relative_eq!(map["c0"], 2.5, epsilon = 1e-2);
        assert_relative_eq!(map["c1"], 0.8, epsilon = 1e-2);
        assert_relative_eq!(result.r_squared().unwrap(), 1.0, epsilon = 1e-10);
    }
}
