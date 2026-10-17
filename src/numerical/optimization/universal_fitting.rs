//! High-level fitting facade over classic LM, Gavin LM, and VarPro frontends.
//!
//! `UniversalFitting` is a small depot-style wrapper that lets callers choose
//! the backend late in the builder chain. By default it uses the classic
//! symbolic Levenberg-Marquardt path from `sym_fitting`. `with_gavin()` selects
//! Gavin's distinct damping/Broyden controller, while `with_varpro()` selects
//! symbolic variable projection. Controller-specific options remain explicit
//! and unsupported shared options return a typed error.

use crate::numerical::Nonlinear_systems::least_squares::{
    LeastSquaresError, LeastSquaresTelemetryMode, LevenbergMarquardt,
};
use crate::numerical::Nonlinear_systems::symbolic::{
    LambdifyExecutionPolicy, SymbolicLambdifyFrontend,
};
use crate::numerical::SolverTelemetryMode;
use crate::numerical::logging::{SolverLogGuard, SolverLogLevel};
use crate::numerical::optimization::gavin_fitting::{GavinSymbolicFitResult, GavinSymbolicFitting};
use crate::numerical::optimization::lm_gavin::GavinLmUpdatePolicy;
use crate::numerical::optimization::sym_fitting::Fitting;
use crate::numerical::optimization::varpro::symbolic::{
    SymbolicVarProBuilder, SymbolicVarProError, SymbolicVarProFit,
};
use crate::symbolic::symbolic_engine::Expr;
use log::Level;
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
    /// H.P. Gavin's symbolic LM controller with FD/Broyden updates.
    GAVIN,
}

/// Result returned by [`UniversalFitting::build`].
pub enum UniversalFittingResult {
    /// The classic symbolic Levenberg-Marquardt pipeline.
    LM(Fitting),
    /// The symbolic VarPro pipeline.
    VARPRO(SymbolicVarProFit),
    /// Gavin's independently controlled symbolic LM fit.
    GAVIN(GavinSymbolicFitResult),
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
            UniversalFittingResult::GAVIN(fit) => Some(
                fit.parameter_names
                    .iter()
                    .cloned()
                    .zip(fit.fit.parameters.iter().copied())
                    .collect(),
            ),
        }
    }

    /// Return the coefficient of determination for the fit.
    ///
    /// For unweighted problems this is the usual `r^2` quality metric.
    pub fn r_squared(&self) -> Option<f64> {
        match self {
            UniversalFittingResult::LM(fitting) => fitting.r_squared(),
            UniversalFittingResult::VARPRO(fit) => fit.r_squared(),
            UniversalFittingResult::GAVIN(fit) => fit.fit.r_squared,
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
    /// A shared solver setting could not be applied to the selected backend.
    #[error("invalid fitting solver configuration: {0}")]
    SolverConfiguration(#[from] LeastSquaresError),
    /// A Gavin-specific symbolic preparation or solve failed.
    #[error(transparent)]
    Gavin(#[from] crate::numerical::optimization::lm_gavin::GavinLmError),
    /// A requested shared option has no equivalent in the selected controller.
    #[error("option '{option}' is not supported by fitting method {method:?}")]
    UnsupportedOption {
        method: Method,
        option: &'static str,
    },
}

/// Unified facade over the classic LM, Gavin LM, and VarPro fitting builders.
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
    gavin: GavinSymbolicFitting,
    frontend: SymbolicLambdifyFrontend,
    execution_policy: LambdifyExecutionPolicy,
    telemetry_mode: LeastSquaresTelemetryMode,
    gavin_telemetry_mode: SolverTelemetryMode,
    logging_enabled: bool,
    tolerance: Option<f64>,
    f_tolerance: Option<f64>,
    g_tolerance: Option<f64>,
    max_iterations: Option<usize>,
    scale_diag: Option<bool>,
    gavin_controls: Option<(usize, f64, f64, f64)>,
}

impl UniversalFitting {
    /// Create a new wrapper with the classic LM backend selected.
    pub fn new() -> Self {
        Self {
            method: Method::LM,
            equations: Vec::new(),
            lm: Fitting::new(),
            varpro: SymbolicVarProBuilder::new("x"),
            gavin: GavinSymbolicFitting::new(),
            frontend: SymbolicLambdifyFrontend::ExprLegacy,
            execution_policy: LambdifyExecutionPolicy::Sequential,
            telemetry_mode: LeastSquaresTelemetryMode::Off,
            gavin_telemetry_mode: SolverTelemetryMode::Off,
            logging_enabled: false,
            tolerance: None,
            f_tolerance: None,
            g_tolerance: None,
            max_iterations: None,
            scale_diag: None,
            gavin_controls: None,
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

    /// Shortcut for [`with_method`](Self::with_method) with [`Method::GAVIN`].
    pub fn with_gavin(self) -> Self {
        self.with_method(Method::GAVIN)
    }

    /// Set the independent variable data.
    pub fn with_x_data(mut self, x_data: Vec<f64>) -> Self {
        self.varpro = self.varpro.with_x_data(DVector::from_vec(x_data.clone()));
        self.gavin = self.gavin.with_x_data(x_data.clone());
        self.lm = self.lm.with_x_data(x_data);
        self
    }

    /// Set the observed data.
    pub fn with_y_data(mut self, y_data: Vec<f64>) -> Self {
        self.varpro = self.varpro.with_y_data(DVector::from_vec(y_data.clone()));
        self.gavin = self.gavin.with_observations(y_data.clone());
        self.lm = self.lm.with_y_data(y_data);
        self
    }

    /// Set independent variable and observed data together.
    pub fn with_data(mut self, x_data: Vec<f64>, y_data: Vec<f64>) -> Self {
        self.gavin = self.gavin.with_data(x_data.clone(), y_data.clone());
        self.varpro = self.varpro.with_data(
            DVector::from_vec(x_data.clone()),
            DVector::from_vec(y_data.clone()),
        );
        self.lm = self.lm.with_data(x_data, y_data);
        self
    }

    /// Set the unknown names for both backends.
    pub fn with_unknowns<I, S>(mut self, unknowns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let unknowns = unknowns.into_iter().map(Into::into).collect::<Vec<_>>();
        self.gavin = self.gavin.with_unknowns(unknowns.clone());
        self.varpro = self.varpro.with_parameters(unknowns.clone());
        self.lm = self.lm.with_unknowns(unknowns);
        self
    }

    /// Alias for [`with_unknowns`](Self::with_unknowns).
    pub fn with_parameters<I, S>(self, parameters: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.with_unknowns(parameters)
    }

    /// Set the initial guess for both backends.
    pub fn with_initial_guess(mut self, initial_guess: Vec<f64>) -> Self {
        self.gavin = self.gavin.with_initial_guess(initial_guess.clone());
        self.varpro = self.varpro.with_initial_parameters(initial_guess.clone());
        self.lm = self.lm.with_initial_guess(initial_guess);
        self
    }

    /// Set the symbolic name of the independent variable.
    pub fn with_arg(mut self, arg: impl Into<String>) -> Self {
        let arg = arg.into();
        self.gavin = self.gavin.with_sample_parameter(arg.clone());
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
        self.gavin = self.gavin.with_equations(self.equations.clone());
        self
    }

    /// Set a symbolic expression system.
    ///
    /// For the LM backend, the expressions are solved as residual equations.
    /// For the VarPro backend, they are treated as invariant/background terms.
    pub fn with_equations(mut self, eq_system: Vec<Expr>) -> Self {
        self.equations = eq_system;
        self.gavin = self.gavin.with_equations(self.equations.clone());
        self
    }

    /// Set one symbolic model expression from a string.
    pub fn with_equation_str(mut self, eq_string: impl AsRef<str>) -> Self {
        self.equations = vec![Expr::parse_expression(eq_string.as_ref())];
        self.gavin = self.gavin.with_equations(self.equations.clone());
        self
    }

    /// Set a symbolic expression system from strings.
    pub fn with_equations_str<I, S>(mut self, eq_system_string: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.equations = eq_system_string
            .into_iter()
            .map(|expr| Expr::parse_expression(expr.as_ref()))
            .collect();
        self.gavin = self.gavin.with_equations(self.equations.clone());
        self
    }

    /// Set the polynomial model and select the LM backend.
    pub fn with_polynomial(mut self, degree: usize, arg: String) -> Self {
        self.method = Method::LM;
        self.lm = self.lm.with_polynomial(degree, arg);
        self.equations = self.lm.equations.clone();
        self.gavin = self.gavin.with_equations(self.equations.clone());
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

    /// Set the parameter-step tolerance for the selected fitting backend.
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = Some(tolerance);
        self.lm = self.lm.with_tolerance(tolerance);
        self
    }

    /// Set the objective tolerance for the selected fitting backend.
    pub fn with_f_tolerance(mut self, f_tolerance: f64) -> Self {
        self.f_tolerance = Some(f_tolerance);
        self.lm = self.lm.with_f_tolerance(f_tolerance);
        self
    }

    /// Set the gradient tolerance for the selected fitting backend.
    pub fn with_g_tolerance(mut self, g_tolerance: f64) -> Self {
        self.g_tolerance = Some(g_tolerance);
        self.lm = self.lm.with_g_tolerance(g_tolerance);
        self
    }

    /// Set diagonal scaling for the selected fitting backend.
    pub fn with_scale_diag(mut self, scale_diag: bool) -> Self {
        self.scale_diag = Some(scale_diag);
        self.lm = self.lm.with_scale_diag(scale_diag);
        self
    }

    /// Set the nonlinear iteration budget for the selected fitting backend.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = Some(max_iterations);
        self.lm = self.lm.with_max_iterations(max_iterations);
        self
    }

    /// Selects the symbolic representation for supported Lambdify solvers.
    /// ExprLegacy remains the default. VarPro currently accepts ExprLegacy only.
    pub fn with_symbolic_frontend(mut self, frontend: SymbolicLambdifyFrontend) -> Self {
        self.frontend = frontend;
        self.lm = self.lm.with_symbolic_frontend(frontend);
        self.gavin = self.gavin.with_frontend(frontend);
        self
    }

    /// Short alias for [`with_symbolic_frontend`](Self::with_symbolic_frontend).
    pub fn with_frontend(self, frontend: SymbolicLambdifyFrontend) -> Self {
        self.with_symbolic_frontend(frontend)
    }

    /// Configures sequential or thresholded parallel prepared callback evaluation.
    pub fn with_lambdify_execution_policy(mut self, policy: LambdifyExecutionPolicy) -> Self {
        self.execution_policy = policy;
        self.lm = self.lm.with_lambdify_execution_policy(policy);
        self.gavin = self.gavin.with_lambdify_execution_policy(policy);
        self
    }

    /// Configures classic LM/VarPro telemetry. Gavin's distinct controller
    /// uses [`SolverTelemetryMode`] via [`with_gavin_telemetry`](Self::with_gavin_telemetry).
    /// Requesting this least-squares telemetry mode for Gavin is rejected at build.
    pub fn with_telemetry(mut self, mode: LeastSquaresTelemetryMode) -> Self {
        self.telemetry_mode = mode;
        self.lm = self.lm.with_telemetry(mode);
        self
    }

    /// Enables or disables logging to the shared timestamped solver log.
    /// Logging is off by default; VarPro emits a completion summary while LM
    /// and Gavin retain their solver-specific details.
    pub fn with_logging(mut self, enabled: bool) -> Self {
        self.logging_enabled = enabled;
        self.lm = if enabled {
            self.lm.with_logging(Level::Info)
        } else {
            self.lm.without_logging()
        };
        self.gavin = self.gavin.with_logging(enabled);
        self
    }

    /// Select Gavin's damping update policy when using [`Method::GAVIN`].
    pub fn with_gavin_update_policy(mut self, policy: GavinLmUpdatePolicy) -> Self {
        self.gavin = self.gavin.with_update_policy(policy);
        self
    }

    /// Select Gavin solve telemetry when using [`Method::GAVIN`].
    pub fn with_gavin_telemetry(mut self, mode: SolverTelemetryMode) -> Self {
        self.gavin_telemetry_mode = mode;
        self.gavin = self.gavin.with_telemetry(mode);
        self
    }

    /// Select Gavin's ExprLegacy or AtomView symbolic frontend.
    pub fn with_gavin_frontend(self, frontend: SymbolicLambdifyFrontend) -> Self {
        self.with_symbolic_frontend(frontend)
    }

    /// Select Gavin's optional instance-local log output.
    pub fn with_gavin_logging(self, enabled: bool) -> Self {
        self.with_logging(enabled)
    }

    /// Set Gavin's iteration budget, gradient tolerance, step tolerance, and
    /// gain-ratio threshold without borrowing similarly named controls from
    /// the other methods.
    pub fn with_gavin_solver_controls(
        mut self,
        max_iterations: usize,
        gradient_tolerance: f64,
        parameter_step_tolerance: f64,
        gain_ratio_threshold: f64,
    ) -> Self {
        self.gavin_controls = Some((
            max_iterations,
            gradient_tolerance,
            parameter_step_tolerance,
            gain_ratio_threshold,
        ));
        self
    }

    /// Set nonnegative per-observation weights on the Gavin route.
    pub fn with_gavin_weights(mut self, weights: Vec<f64>) -> Self {
        self.gavin = self.gavin.with_weights(weights);
        self
    }

    /// Require selected fitted parameters to remain strictly positive.
    pub fn with_gavin_positive_parameters<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.gavin = self.gavin.with_positive_parameters(names);
        self
    }

    /// Build and solve the selected backend.
    pub fn build(self) -> Result<UniversalFittingResult, UniversalFittingError> {
        let log_level = if self.logging_enabled {
            SolverLogLevel::Info
        } else {
            SolverLogLevel::Off
        };
        let _logging_scope = SolverLogGuard::enter(log_level);
        match self.method {
            Method::LM => {
                let lm = self
                    .lm
                    .with_equations(self.equations)
                    .with_symbolic_frontend(self.frontend)
                    .with_lambdify_execution_policy(self.execution_policy)
                    .with_telemetry(self.telemetry_mode);
                Ok(UniversalFittingResult::LM(lm.build()))
            }
            Method::VARPRO => {
                if self.frontend != SymbolicLambdifyFrontend::ExprLegacy {
                    return Err(UniversalFittingError::UnsupportedOption {
                        method: Method::VARPRO,
                        option: "symbolic_frontend=AtomViewNative",
                    });
                }
                if self.execution_policy != LambdifyExecutionPolicy::Sequential {
                    return Err(UniversalFittingError::UnsupportedOption {
                        method: Method::VARPRO,
                        option: "lambdify_execution_policy",
                    });
                }
                let varpro = self.varpro.with_equations(self.equations);
                let mut solver = LevenbergMarquardt::new()
                    .with_gtol(0.0)
                    .with_telemetry(self.telemetry_mode);
                if let Some(max_iterations) = self.max_iterations {
                    solver = solver.try_with_max_iterations(max_iterations)?;
                }
                if let Some(tolerance) = self.tolerance {
                    solver = solver.try_with_xtol(tolerance)?;
                }
                if let Some(f_tolerance) = self.f_tolerance {
                    solver = solver.try_with_ftol(f_tolerance)?;
                }
                if let Some(g_tolerance) = self.g_tolerance {
                    solver = solver.try_with_gtol(g_tolerance)?;
                }
                if let Some(scale_diag) = self.scale_diag {
                    solver = solver.with_scale_diag(scale_diag);
                }
                let fit = varpro.solve_with_solver(solver)?;
                if self.logging_enabled {
                    log::info!(
                        "UniversalFitting VarPro finished: termination={:?}, objective={:.6e}, evaluations={}",
                        fit.minimization_report.termination,
                        fit.minimization_report.objective_function,
                        fit.minimization_report.number_of_evaluations
                    );
                }
                Ok(UniversalFittingResult::VARPRO(fit))
            }
            Method::GAVIN => {
                if self.telemetry_mode != LeastSquaresTelemetryMode::Off {
                    return Err(UniversalFittingError::UnsupportedOption {
                        method: Method::GAVIN,
                        option: "use with_gavin_telemetry(SolverTelemetryMode)",
                    });
                }
                if self.f_tolerance.is_some() {
                    return Err(UniversalFittingError::UnsupportedOption {
                        method: Method::GAVIN,
                        option: "f_tolerance",
                    });
                }
                if self.scale_diag.is_some() {
                    return Err(UniversalFittingError::UnsupportedOption {
                        method: Method::GAVIN,
                        option: "scale_diag",
                    });
                }
                let controls = self.gavin_controls.unwrap_or((
                    self.max_iterations.unwrap_or(1000),
                    self.g_tolerance.unwrap_or(1e-3),
                    self.tolerance.unwrap_or(1e-3),
                    1e-1,
                ));
                Ok(UniversalFittingResult::GAVIN(
                    self.gavin
                        .with_equations(self.equations)
                        .with_frontend(self.frontend)
                        .with_lambdify_execution_policy(self.execution_policy)
                        .with_telemetry(self.gavin_telemetry_mode)
                        .with_logging(self.logging_enabled)
                        .with_solver_controls(controls.0, controls.1, controls.2, controls.3)
                        .try_solve()?,
                ))
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
            UniversalFittingResult::GAVIN(_) => panic!("expected LM result"),
        };

        let map = fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map["a"], 3.0, epsilon = 1e-6);
        assert_relative_eq!(map["b"], 2.0, epsilon = 1e-6);
    }

    #[test]
    fn universal_lm_accepts_common_frontend_telemetry_and_slice_builders() {
        let x = (0..12).map(|index| index as f64).collect::<Vec<_>>();
        let y = x.iter().map(|&value| 2.0 * value + 1.0).collect::<Vec<_>>();
        let result = UniversalFitting::new()
            .with_data(x, y)
            .with_equation_str("a*x+b")
            .with_arg("x")
            .with_unknowns(["a", "b"])
            .with_initial_guess(vec![1.0, 0.0])
            .with_frontend(SymbolicLambdifyFrontend::AtomViewNative)
            .with_telemetry(LeastSquaresTelemetryMode::Counters)
            .build()
            .unwrap();

        let UniversalFittingResult::LM(fit) = result else {
            panic!("expected classic LM result");
        };
        assert_eq!(
            fit.symbolic_frontend(),
            SymbolicLambdifyFrontend::AtomViewNative
        );
        assert!(fit.last_statistics.is_some());
        assert_relative_eq!(
            fit.map_of_solutions.as_ref().unwrap()["a"],
            2.0,
            epsilon = 1e-8
        );
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
    fn universal_polynomial_model_survives_build_dispatch() {
        let x_data = (0..12).map(|x| x as f64 * 0.25).collect::<Vec<_>>();
        let y_data = x_data
            .iter()
            .map(|&x| 2.0 * x * x + 3.0 * x + 1.0)
            .collect::<Vec<_>>();

        let result = UniversalFitting::new()
            .with_data(x_data, y_data)
            .with_polynomial(2, "x".to_string())
            .with_initial_guess(vec![0.0; 3])
            .build()
            .unwrap();

        let map = result.solution_map().unwrap();
        assert_relative_eq!(map["c2"], 2.0, epsilon = 1e-7);
        assert_relative_eq!(map["c1"], 3.0, epsilon = 1e-7);
        assert_relative_eq!(map["c0"], 1.0, epsilon = 1e-7);
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
            .with_telemetry(LeastSquaresTelemetryMode::Detailed)
            .with_logging(true)
            .build()
            .unwrap();

        let fit = match result {
            UniversalFittingResult::VARPRO(fit) => fit,
            UniversalFittingResult::LM(_) => panic!("expected VarPro result"),
            UniversalFittingResult::GAVIN(_) => panic!("expected VarPro result"),
        };

        let nonlinear = fit.nonlinear_parameters();
        assert_relative_eq!(nonlinear[0], 1.7, epsilon = 1e-2);

        let linear = fit.linear_coefficients().unwrap();
        assert_relative_eq!(linear[0], 2.5, epsilon = 1e-2);
        assert_relative_eq!(linear[1], 0.8, epsilon = 1e-2);
        assert!(fit.minimization_report.statistics.timings_collected);
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

    #[test]
    fn universal_varpro_applies_shared_iteration_budget() {
        let x_data = (0..30).map(|x| x as f64 * 0.1).collect::<Vec<_>>();
        let y_data = x_data
            .iter()
            .map(|&x| 3.0 * (-x / 1.7).exp() + 0.4)
            .collect::<Vec<_>>();

        let result = UniversalFitting::new()
            .with_varpro()
            .with_data(x_data, y_data)
            .with_arg("x".to_string())
            .with_parameters(vec!["tau".to_string()])
            .with_initial_guess(vec![20.0])
            .with_basis_str("tau", "exp(-x/tau)")
            .with_equation_str("1".to_string())
            .with_max_iterations(1)
            .build();

        assert!(matches!(
            result,
            Err(UniversalFittingError::VarPro(
                SymbolicVarProError::SolverFailed { .. }
            ))
        ));
    }

    #[test]
    fn universal_varpro_rejects_invalid_shared_tolerance() {
        let result = UniversalFitting::new()
            .with_varpro()
            .with_data(vec![0.0, 1.0], vec![1.0, 0.5])
            .with_arg("x".to_string())
            .with_parameters(vec!["tau".to_string()])
            .with_initial_guess(vec![1.0])
            .with_basis_str("tau", "exp(-x/tau)")
            .with_tolerance(-1.0)
            .build();

        assert!(matches!(
            result,
            Err(UniversalFittingError::SolverConfiguration(
                LeastSquaresError::InvalidConfiguration { field: "xtol", .. }
            ))
        ));
    }

    #[test]
    fn universal_varpro_rejects_atom_frontend_with_typed_error() {
        let result = UniversalFitting::new()
            .with_varpro()
            .with_frontend(SymbolicLambdifyFrontend::AtomViewNative)
            .with_data(vec![0.0, 1.0], vec![1.0, 0.5])
            .with_arg("x")
            .with_parameters(["tau"])
            .with_initial_guess(vec![1.0])
            .with_basis_str("tau", "exp(-x/tau)")
            .with_equation_str("1")
            .build();

        assert!(matches!(
            result,
            Err(UniversalFittingError::UnsupportedOption {
                method: Method::VARPRO,
                option: "symbolic_frontend=AtomViewNative"
            })
        ));
    }

    #[test]
    fn universal_gavin_fits_nonlinear_model_and_preserves_diagnostics() {
        let x = (0..40).map(|i| i as f64 * 0.1).collect::<Vec<_>>();
        let y = x
            .iter()
            .map(|&x| 1.7 * (-x / 1.3).exp() + 0.25)
            .collect::<Vec<_>>();

        let result = UniversalFitting::new()
            .with_gavin()
            .with_data(x, y)
            .with_equation_str("a*exp(-x/tau)+b".to_owned())
            .with_arg("x".to_owned())
            .with_unknowns(vec!["a".to_owned(), "tau".to_owned(), "b".to_owned()])
            .with_initial_guess(vec![1.4, 1.6, 0.1])
            .with_gavin_telemetry(SolverTelemetryMode::Detailed)
            .with_logging(true)
            .build()
            .expect("Gavin universal route should fit the nonlinear model");

        let fit = match &result {
            UniversalFittingResult::GAVIN(fit) => fit,
            _ => panic!("expected Gavin result"),
        };
        assert!(fit.fit.termination.is_converged());
        assert_eq!(
            fit.frontend,
            SymbolicLambdifyFrontend::ExprLegacy,
            "Gavin defaults to ExprLegacy unless a frontend is explicitly selected"
        );
        assert!((fit.fit.parameters[0] - 1.7).abs() < 2e-2);
        assert!((fit.fit.parameters[1] - 1.3).abs() < 2e-2);
        assert!((fit.fit.parameters[2] - 0.25).abs() < 2e-2);
        assert!(fit.telemetry.as_ref().unwrap().objective_evaluations > 0);
        let map = result.solution_map().unwrap();
        assert!((map["tau"] - fit.fit.parameters[1]).abs() < f64::EPSILON);
        assert!(result.r_squared().unwrap() > 0.999);
    }

    #[test]
    fn universal_gavin_common_settings_select_atom_and_gate_telemetry() {
        let x = (0..10).map(|index| index as f64 * 0.2).collect::<Vec<_>>();
        let y = x.iter().map(|&value| 1.7 * value + 0.4).collect::<Vec<_>>();
        let result = UniversalFitting::new()
            .with_gavin()
            .with_data(x, y)
            .with_equation_str("a*x+b")
            .with_arg("x")
            .with_unknowns(["a", "b"])
            .with_initial_guess(vec![1.2, 0.2])
            .with_frontend(SymbolicLambdifyFrontend::AtomViewNative)
            .with_telemetry(LeastSquaresTelemetryMode::Off)
            .with_logging(false)
            .build()
            .unwrap();

        let UniversalFittingResult::GAVIN(fit) = result else {
            panic!("expected Gavin result");
        };
        assert_eq!(fit.frontend, SymbolicLambdifyFrontend::AtomViewNative);
        assert!(fit.telemetry.is_none());
        assert_relative_eq!(fit.fit.parameters[0], 1.7, epsilon = 1e-6);
        assert_relative_eq!(fit.fit.parameters[1], 0.4, epsilon = 1e-6);
    }

    #[test]
    fn universal_gavin_uses_explicit_telemetry_mode_and_rejects_lm_modes() {
        let invalid = UniversalFitting::new()
            .with_gavin()
            .with_telemetry(LeastSquaresTelemetryMode::Counters)
            .build();
        assert!(matches!(
            invalid,
            Err(UniversalFittingError::UnsupportedOption {
                method: Method::GAVIN,
                option: "use with_gavin_telemetry(SolverTelemetryMode)"
            })
        ));

        let x = vec![0.0, 1.0, 2.0, 3.0];
        let result = UniversalFitting::new()
            .with_gavin()
            .with_data(x.clone(), x.iter().map(|value| 2.0 * value + 1.0).collect())
            .with_equation_str("a*x+b")
            .with_arg("x")
            .with_unknowns(["a", "b"])
            .with_initial_guess(vec![1.5, 0.5])
            .with_gavin_telemetry(SolverTelemetryMode::Detailed)
            .build()
            .unwrap();
        let UniversalFittingResult::GAVIN(fit) = result else {
            panic!("expected Gavin result");
        };
        assert!(fit.telemetry.is_some());
    }

    #[test]
    fn universal_gavin_rejects_options_without_equivalent_semantics() {
        let result = UniversalFitting::new()
            .with_gavin()
            .with_f_tolerance(1e-8)
            .build();
        assert!(matches!(
            result,
            Err(UniversalFittingError::UnsupportedOption {
                method: Method::GAVIN,
                option: "f_tolerance"
            })
        ));
    }
}
