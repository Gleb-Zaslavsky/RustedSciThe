//! Python adapter for the native symbolic variable-projection fitter.
//!
//! VarPro is exposed as a fluent one-shot builder because the native
//! `SymbolicVarProBuilder` owns its model and consumes it during `solve()`.
//! This is intentionally separate from the prepared canonical fitting route.

use std::collections::HashMap;

use pyo3::prelude::*;

use rustedscithe_core::numerical::optimization::varpro::symbolic::{
    SymbolicVarProBuilder, SymbolicVarProError,
};

use crate::errors::{
    InvalidConfigurationError, InvalidExpressionError, SolverError, annotate_native, catch_panic,
    native_variant,
};
use crate::expr::PyExpr;

fn map_error(error: SymbolicVarProError) -> PyErr {
    let native_kind = native_variant(&error);
    let message = error.to_string();
    let mapped = match error {
        SymbolicVarProError::XDataMissing
        | SymbolicVarProError::YDataMissing
        | SymbolicVarProError::ParameterNamesMissing
        | SymbolicVarProError::InitialParametersMissing
        | SymbolicVarProError::BasisFunctionsMissing
        | SymbolicVarProError::ModelBuild(_)
        | SymbolicVarProError::ProblemBuild(_) => InvalidConfigurationError::new_err(message),
        SymbolicVarProError::SolverFailed { .. } => SolverError::new_err(message),
    };
    annotate_native(mapped, "solve", "varpro", "lambdify", &native_kind)
}

/// Detached result from one symbolic variable-projection fit.
///
/// The solution maps named nonlinear parameters to fitted values. `best_fit` contains
/// model values at the supplied data points, in their input order. The result does not
/// retain the builder or symbolic expressions.
#[pyclass(name = "VarProSolution", module = "rustedscithe")]
pub struct VarProSolution {
    solution: HashMap<String, f64>,
    best_fit: Vec<f64>,
    r_squared: Option<f64>,
    successful: bool,
    termination: String,
}

impl VarProSolution {
    fn from_native(
        result: rustedscithe_core::numerical::optimization::varpro::symbolic::SymbolicVarProFit,
    ) -> Self {
        let best_fit = result
            .best_fit()
            .map(|values| values.iter().copied().collect())
            .unwrap_or_default();
        Self {
            solution: result.solution_map().unwrap_or_default(),
            best_fit,
            r_squared: result.r_squared(),
            successful: result.was_successful(),
            termination: format!("{:?}", result.minimization_report.termination),
        }
    }
}

#[pymethods]
impl VarProSolution {
    /// Return fitted parameter values keyed by their declared names.
    pub fn solution(&self) -> HashMap<String, f64> {
        self.solution.clone()
    }
    /// Return model values evaluated at the input sample points.
    pub fn best_fit(&self) -> Vec<f64> {
        self.best_fit.clone()
    }
    #[getter]
    /// Return the coefficient of determination when defined by the data.
    pub fn r_squared(&self) -> Option<f64> {
        self.r_squared
    }
    #[getter]
    /// Whether the native minimizer reported successful termination.
    pub fn successful(&self) -> bool {
        self.successful
    }
    #[getter]
    /// Native minimizer termination label.
    pub fn termination(&self) -> &str {
        &self.termination
    }
    /// Return a compact string-valued summary of fit status and quality.
    pub fn diagnostics(&self) -> HashMap<String, String> {
        HashMap::from([
            ("successful".to_owned(), self.successful.to_string()),
            ("termination".to_owned(), self.termination.clone()),
            (
                "r_squared".to_owned(),
                self.r_squared
                    .map_or_else(|| "undefined".to_owned(), |v| v.to_string()),
            ),
        ])
    }
}

/// Fluent one-shot symbolic variable-projection fitter.
///
/// Build the problem by supplying named nonlinear parameters, initial values, data,
/// and one basis expression per parameter. Optional invariant expressions can be
/// added before solving. `solve()` consumes the builder; construct a new instance for
/// another independent fit. This API is distinct from the reusable canonical
/// `FittingSolver` lifecycle.
#[pyclass(name = "VarProSolver", module = "rustedscithe")]
pub struct VarProSolver {
    builder: Option<SymbolicVarProBuilder>,
}

impl VarProSolver {
    fn update<F>(&mut self, update: F) -> PyResult<()>
    where
        F: FnOnce(SymbolicVarProBuilder) -> SymbolicVarProBuilder,
    {
        let builder = self.builder.take().ok_or_else(|| {
            SolverError::new_err("VarProSolver is already solved; construct a new solver")
        })?;
        self.builder = Some(update(builder));
        Ok(())
    }
}

#[pymethods]
impl VarProSolver {
    #[new]
    /// Create an empty builder and declare the independent-variable name.
    pub fn new(x_name: String) -> Self {
        Self {
            builder: Some(SymbolicVarProBuilder::new(x_name)),
        }
    }

    /// Declare the ordered names of nonlinear parameters to fit.
    pub fn with_parameters(&mut self, names: Vec<String>) -> PyResult<()> {
        self.update(|builder| builder.with_parameters(names))
    }

    /// Set initial values corresponding to `with_parameters` order.
    pub fn with_initial_parameters(&mut self, values: Vec<f64>) -> PyResult<()> {
        self.update(|builder| builder.with_initial_parameters(values))
    }

    /// Set paired independent-variable and observation vectors.
    pub fn with_data(&mut self, x_data: Vec<f64>, y_data: Vec<f64>) -> PyResult<()> {
        self.update(|builder| {
            builder.with_data(
                nalgebra::DVector::from_vec(x_data),
                nalgebra::DVector::from_vec(y_data),
            )
        })
    }

    /// Add a symbolic basis expression for a named parameter.
    ///
    /// Basis expressions define the linear-in-parameter portion of the model.
    pub fn with_basis(&mut self, parameter_name: String, expression: PyExpr) -> PyResult<()> {
        self.update(|builder| builder.with_basis_expr(parameter_name, expression.inner))
    }

    /// Parse and add a basis expression for a named parameter.
    pub fn with_basis_str(&mut self, parameter_name: String, expression: &str) -> PyResult<()> {
        let expression =
            rustedscithe_core::symbolic::symbolic_engine::Expr::try_parse_expression(expression)
                .map_err(|error| InvalidExpressionError::new_err(error))?;
        self.update(|builder| builder.with_basis_expr(parameter_name, expression))
    }

    /// Add a symbolic invariant or constraint expression to the fit definition.
    pub fn with_invariant(&mut self, expression: PyExpr) -> PyResult<()> {
        self.update(|builder| builder.with_invariant_expr(expression.inner))
    }

    /// Parse and add an invariant or constraint expression to the fit definition.
    pub fn with_invariant_str(&mut self, expression: &str) -> PyResult<()> {
        let expression =
            rustedscithe_core::symbolic::symbolic_engine::Expr::try_parse_expression(expression)
                .map_err(|error| InvalidExpressionError::new_err(error))?;
        self.update(|builder| builder.with_invariant_expr(expression))
    }

    /// Run the fit and consume this one-shot builder.
    pub fn solve(&mut self) -> PyResult<VarProSolution> {
        catch_panic("solve", "varpro", "lambdify", || {
            let builder = self.builder.take().ok_or_else(|| {
                SolverError::new_err(
                    "VarProSolver is one-shot; construct a new solver for another fit",
                )
            })?;
            builder
                .solve()
                .map(VarProSolution::from_native)
                .map_err(map_error)
        })
    }

    /// Return whether the builder has been consumed by `solve()`.
    pub fn is_solved(&self) -> bool {
        self.builder.is_none()
    }
}
