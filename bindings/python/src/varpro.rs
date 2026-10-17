//! Python adapter for the native symbolic variable-projection fitter.
//!
//! VarPro is exposed as a fluent one-shot builder because the native
//! `SymbolicVarProBuilder` owns its model and consumes it during `solve()`.
//! This is intentionally separate from the prepared canonical fitting route.

use std::collections::HashMap;

use pyo3::prelude::*;

use rustedscithe_core::numerical::Nonlinear_systems::least_squares::{
    LeastSquaresTelemetryMode, LevenbergMarquardt,
};
use rustedscithe_core::numerical::optimization::varpro::symbolic::{
    SymbolicVarProBuilder, SymbolicVarProError,
};

use crate::errors::{
    ExhaustionError, InvalidConfigurationError, InvalidExpressionError, SolverError,
    annotate_native, catch_panic, native_variant,
};
use crate::expr::PyExpr;

fn map_error(error: SymbolicVarProError) -> PyErr {
    let error = match error {
        SymbolicVarProError::SolverFailed {
            source: Some(source),
            ..
        } => {
            let native_kind = native_variant(&source);
            return annotate_native(
                crate::nonlinear::map_least_squares_error(source),
                "solve",
                "varpro",
                "lambdify",
                &native_kind,
            );
        }
        SymbolicVarProError::SolverFailed {
            source: None,
            termination,
            exhausted: true,
        } => {
            return annotate_native(
                ExhaustionError::new_err(format!(
                    "symbolic VarPro solver exhausted: {termination}"
                )),
                "solve",
                "varpro",
                "lambdify",
                "SolverFailed",
            );
        }
        other => other,
    };
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
    objective_function: f64,
    function_evaluations: usize,
    statistics: rustedscithe_core::numerical::Nonlinear_systems::engine::SolveStatistics,
    telemetry_enabled: bool,
}

impl VarProSolution {
    fn from_native(
        result: rustedscithe_core::numerical::optimization::varpro::symbolic::SymbolicVarProFit,
        telemetry_enabled: bool,
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
            objective_function: result.minimization_report.objective_function,
            function_evaluations: result.minimization_report.number_of_evaluations,
            statistics: result.minimization_report.statistics,
            telemetry_enabled,
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
    #[getter]
    /// Objective value from the underlying least-squares controller.
    pub fn objective_function(&self) -> f64 {
        self.objective_function
    }
    #[getter]
    /// Number of objective evaluations reported by the native solver.
    pub fn function_evaluations(&self) -> usize {
        self.function_evaluations
    }
    /// Return native counters when telemetry was enabled, otherwise an empty map.
    pub fn telemetry_counters(&self) -> HashMap<String, u64> {
        if self.telemetry_enabled {
            crate::nonlinear::statistics_counters(&self.statistics)
        } else {
            HashMap::new()
        }
    }
    /// Return native timing stages when detailed telemetry was enabled.
    pub fn telemetry_timings_ms(&self) -> HashMap<String, f64> {
        if self.telemetry_enabled && self.statistics.timings_collected {
            crate::nonlinear::statistics_timings(&self.statistics)
        } else {
            HashMap::new()
        }
    }
    /// Return a compact string-valued summary of fit status and quality.
    pub fn diagnostics(&self) -> HashMap<String, String> {
        HashMap::from([
            ("successful".to_owned(), self.successful.to_string()),
            ("termination".to_owned(), self.termination.clone()),
            (
                "objective_function".to_owned(),
                self.objective_function.to_string(),
            ),
            (
                "function_evaluations".to_owned(),
                self.function_evaluations.to_string(),
            ),
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
    solver: LevenbergMarquardt,
    telemetry_mode: LeastSquaresTelemetryMode,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_less_budget_failure_uses_typed_exhaustion_flag() {
        Python::initialize();
        let error = map_error(SymbolicVarProError::SolverFailed {
            termination: "budget exhausted".to_owned(),
            exhausted: true,
            source: None,
        });
        Python::attach(|py| {
            let value = error.value(py);
            assert!(value.is_instance_of::<ExhaustionError>());
            assert_eq!(
                value
                    .getattr("native_kind")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "SolverFailed"
            );
            assert_eq!(
                value.getattr("stage").unwrap().extract::<String>().unwrap(),
                "solve"
            );
        });

        // A diagnostic that happens to resemble an exhaustion variant must
        // not override the typed classification.
        let error = map_error(SymbolicVarProError::SolverFailed {
            termination: "MaxIterationsReached { limit: 1 }".to_owned(),
            exhausted: false,
            source: None,
        });
        Python::attach(|py| assert!(error.value(py).is_instance_of::<SolverError>()));
    }
}

#[pymethods]
impl VarProSolver {
    #[new]
    /// Create an empty builder and declare the independent-variable name.
    pub fn new(x_name: String) -> Self {
        Self {
            builder: Some(SymbolicVarProBuilder::new(x_name)),
            solver: LevenbergMarquardt::new().with_gtol(0.0),
            telemetry_mode: LeastSquaresTelemetryMode::Off,
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

    /// Set the objective, parameter-step and gradient convergence tolerances.
    pub fn with_tolerances(&mut self, ftol: f64, xtol: f64, gtol: f64) -> PyResult<()> {
        let solver = self
            .solver
            .try_with_ftol(ftol)
            .and_then(|solver| solver.try_with_xtol(xtol))
            .and_then(|solver| solver.try_with_gtol(gtol))
            .map_err(crate::nonlinear::map_least_squares_error)?;
        self.solver = solver;
        Ok(())
    }

    /// Set an exact cap on outer LM iterations.
    pub fn with_max_iterations(&mut self, max_iterations: usize) -> PyResult<()> {
        self.solver = self
            .solver
            .try_with_max_iterations(max_iterations)
            .map_err(crate::nonlinear::map_least_squares_error)?;
        Ok(())
    }

    /// Set the function-evaluation budget (`patience`) of the native LM solver.
    pub fn with_max_function_evaluations(&mut self, evaluations: usize) -> PyResult<()> {
        self.solver = self
            .solver
            .try_with_patience(evaluations)
            .map_err(crate::nonlinear::map_least_squares_error)?;
        Ok(())
    }

    /// Configure the maximum trust-region step bound.
    pub fn with_step_bound(&mut self, step_bound: f64) -> PyResult<()> {
        self.solver = self
            .solver
            .try_with_stepbound(step_bound)
            .map_err(crate::nonlinear::map_least_squares_error)?;
        Ok(())
    }

    /// Enable or disable native diagonal scaling.
    pub fn with_scale_diag(&mut self, enabled: bool) {
        self.solver = self.solver.with_scale_diag(enabled);
    }

    /// Enable opt-in native counters or detailed timing telemetry.
    pub fn with_telemetry(&mut self, mode: &str) -> PyResult<()> {
        self.telemetry_mode = match mode.trim().to_ascii_lowercase().as_str() {
            "off" | "none" => LeastSquaresTelemetryMode::Off,
            "counters" => LeastSquaresTelemetryMode::Counters,
            "detailed" | "timings" => LeastSquaresTelemetryMode::Detailed,
            other => {
                return Err(InvalidConfigurationError::new_err(format!(
                    "unsupported VarPro telemetry '{other}'"
                )));
            }
        };
        self.solver = self.solver.with_telemetry(self.telemetry_mode);
        Ok(())
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
                .solve_with_solver(self.solver)
                .map(|result| {
                    VarProSolution::from_native(
                        result,
                        self.telemetry_mode != LeastSquaresTelemetryMode::Off,
                    )
                })
                .map_err(map_error)
        })
    }

    /// Return whether the builder has been consumed by `solve()`.
    pub fn is_solved(&self) -> bool {
        self.builder.is_none()
    }
}
