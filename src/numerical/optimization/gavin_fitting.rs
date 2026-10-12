//! Builder-style symbolic data fitting for Gavin's LM controller.
//!
//! The symbolic frontend is prepared once in `try_solve`; the resulting
//! bound problem is borrowed by the objective adapter for the complete
//! numerical solve. No symbolic differentiation or lowering occurs inside
//! the iteration loop.
//!
//! ```no_run
//! use RustedSciThe::numerical::optimization::gavin_fitting::GavinSymbolicFitting;
//!
//! let x_values = vec![0.0, 1.0, 2.0, 3.0];
//! let observed_values = vec![1.1, 0.8, 0.6, 0.5];
//! let report = GavinSymbolicFitting::new()
//!     .try_with_equation("a*exp(-x/tau)+b")?
//!     .with_data(x_values, observed_values)
//!     .with_sample_parameter("x")
//!     .with_unknowns(["a", "tau", "b"])
//!     .with_initial_guess(vec![1.0, 2.0, 0.1])
//!     .with_positive_parameters(["tau"])
//!     .with_atom_native_frontend()
//!     .with_telemetry(true)
//!     .try_solve()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use super::lm_gavin::{
    GavinLmError, GavinLmResult, GavinLmTelemetry, LevenbergMarquardt, SymbolicGavinObjective,
};
use crate::numerical::Nonlinear_systems::least_squares::{
    LeastSquaresProblem, PreparedSymbolicLeastSquaresProblem,
};
use crate::numerical::Nonlinear_systems::symbolic::{
    LambdifyExecutionPolicy, PreparationTelemetryMode, SymbolicLambdifyFrontend,
    SymbolicPreparationReport, SymbolicProblemOptions,
};
use crate::symbolic::symbolic_engine::Expr;
use nalgebra::DVector;

/// Symbolic fitting configuration and one-shot builder for Gavin LM.
///
/// This gives Gavin the same user-facing flow as `sym_fitting::Fitting` while
/// keeping Gavin's distinct damping/Broyden controller. Data are equation-major
/// for systems with multiple formulas: all observations for equation 0, then
/// equation 1, and so on.
pub struct GavinSymbolicFitting {
    equations: Vec<Expr>,
    sample_parameter: Option<String>,
    unknowns: Vec<String>,
    x_data: Vec<f64>,
    observations: Vec<f64>,
    weights: Option<Vec<f64>>,
    initial_guess: Vec<f64>,
    positive_parameters: Vec<String>,
    frontend: SymbolicLambdifyFrontend,
    execution_policy: LambdifyExecutionPolicy,
    max_iterations: usize,
    tolerances: (f64, f64, f64),
    damping_factors: (f64, f64, f64),
    update_policy: super::lm_gavin::GavinLmUpdatePolicy,
    telemetry: bool,
    logging: bool,
    preparation_telemetry: PreparationTelemetryMode,
}

/// Symbolic preparation provenance and numerical result from one builder solve.
#[derive(Debug, Clone)]
pub struct GavinSymbolicFitResult {
    /// Numerical fit output and explicit controller termination.
    pub fit: GavinLmResult,
    /// Parameter names in the same order as `fit.parameters`.
    pub parameter_names: Vec<String>,
    /// Optional solve counters/timing; absent when not requested.
    pub telemetry: Option<GavinLmTelemetry>,
    /// Effective symbolic backend selected during preparation.
    pub backend: crate::numerical::Nonlinear_systems::symbolic::SymbolicBackendKind,
    /// Selected ExprLegacy/AtomView representation for the Lambdify route.
    pub frontend: SymbolicLambdifyFrontend,
    /// One-time symbolic preparation report, separate from solve timings.
    pub preparation: SymbolicPreparationReport,
}

impl GavinSymbolicFitting {
    /// Starts a symbolic fitting configuration with reference Gavin defaults.
    pub fn new() -> Self {
        Self {
            equations: Vec::new(),
            sample_parameter: None,
            unknowns: Vec::new(),
            x_data: Vec::new(),
            observations: Vec::new(),
            weights: None,
            initial_guess: Vec::new(),
            positive_parameters: Vec::new(),
            frontend: SymbolicLambdifyFrontend::ExprLegacy,
            execution_policy: LambdifyExecutionPolicy::Sequential,
            max_iterations: 1000,
            tolerances: (1e-3, 1e-3, 1e-1),
            damping_factors: (1e-2, 11.0, 9.0),
            update_policy: super::lm_gavin::GavinLmUpdatePolicy::Marquardt,
            telemetry: false,
            logging: false,
            preparation_telemetry: PreparationTelemetryMode::Disabled,
        }
    }

    /// Sets one equation using the checked expression parser.
    pub fn try_with_equation(mut self, equation: &str) -> Result<Self, GavinLmError> {
        let parsed =
            Expr::try_parse_expression(equation).map_err(|error| GavinLmError::SymbolicParse {
                equation_index: 0,
                message: error,
            })?;
        self.equations = vec![parsed];
        Ok(self)
    }

    /// Sets a system of checked symbolic equation strings.
    pub fn try_with_equations<I, S>(mut self, equations: I) -> Result<Self, GavinLmError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.equations = equations
            .into_iter()
            .enumerate()
            .map(|(index, equation)| {
                Expr::try_parse_expression(equation.as_ref()).map_err(|error| {
                    GavinLmError::SymbolicParse {
                        equation_index: index,
                        message: error,
                    }
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(self)
    }

    /// Sets already-parsed symbolic formulas.
    pub fn with_equations(mut self, equations: Vec<Expr>) -> Self {
        self.equations = equations;
        self
    }

    /// Sets sample coordinates and observed values.
    pub fn with_data(mut self, x_data: Vec<f64>, observations: Vec<f64>) -> Self {
        self.x_data = x_data;
        self.observations = observations;
        self
    }

    /// Replaces only the independent-variable samples.
    pub fn with_x_data(mut self, x_data: Vec<f64>) -> Self {
        self.x_data = x_data;
        self
    }

    /// Replaces only the observed values.
    pub fn with_observations(mut self, observations: Vec<f64>) -> Self {
        self.observations = observations;
        self
    }

    /// Supplies one nonnegative least-squares weight per flattened observation.
    pub fn with_weights(mut self, weights: Vec<f64>) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Names the independent/sample variable used in each equation.
    pub fn with_sample_parameter(mut self, name: impl Into<String>) -> Self {
        self.sample_parameter = Some(name.into());
        self
    }

    /// Names the fitted symbolic variables in their numeric vector order.
    pub fn with_unknowns<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.unknowns = names.into_iter().map(Into::into).collect();
        self
    }

    /// Sets the initial parameter values in the same order as `with_unknowns`.
    pub fn with_initial_guess(mut self, initial_guess: Vec<f64>) -> Self {
        self.initial_guess = initial_guess;
        self
    }

    /// Marks selected fitted variables as strictly positive.
    pub fn with_positive_parameters<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.positive_parameters = names.into_iter().map(Into::into).collect();
        self
    }

    /// Selects the symbolic expression representation used by Lambdify.
    pub fn with_frontend(mut self, frontend: SymbolicLambdifyFrontend) -> Self {
        self.frontend = frontend;
        self
    }

    /// Selects sequential or thresholded parallel callback evaluation.
    pub fn with_lambdify_execution_policy(mut self, policy: LambdifyExecutionPolicy) -> Self {
        self.execution_policy = policy;
        self
    }

    /// Selects the packed AtomView-native symbolic route.
    pub fn with_atom_native_frontend(self) -> Self {
        self.with_frontend(SymbolicLambdifyFrontend::AtomViewNative)
    }

    /// Sets the iteration budget and Gavin's three controller tolerances.
    pub fn with_solver_controls(
        mut self,
        max_iterations: usize,
        gradient_tolerance: f64,
        parameter_step_tolerance: f64,
        gain_ratio_threshold: f64,
    ) -> Self {
        self.max_iterations = max_iterations;
        self.tolerances = (
            gradient_tolerance,
            parameter_step_tolerance,
            gain_ratio_threshold,
        );
        self
    }

    /// Selects Marquardt, Quadratic, or Nielsen damping updates.
    pub fn with_update_policy(mut self, policy: super::lm_gavin::GavinLmUpdatePolicy) -> Self {
        self.update_policy = policy;
        self
    }

    /// Sets initial damping and the rejection/acceptance scale factors.
    pub fn with_damping_factors(mut self, initial: f64, increase: f64, decrease: f64) -> Self {
        self.damping_factors = (initial, increase, decrease);
        self
    }

    /// Enables numerical counters and timing for the solve.
    pub fn with_telemetry(mut self, enabled: bool) -> Self {
        self.telemetry = enabled;
        self
    }

    /// Enables log-facade output for this solver instance.
    pub fn with_logging(mut self, enabled: bool) -> Self {
        self.logging = enabled;
        self
    }

    /// Opts into detailed one-time symbolic preparation-stage timing.
    pub fn with_preparation_telemetry(mut self, mode: PreparationTelemetryMode) -> Self {
        self.preparation_telemetry = mode;
        self
    }

    /// Prepares, binds, and solves the configured symbolic fit in one call.
    pub fn try_solve(self) -> Result<GavinSymbolicFitResult, GavinLmError> {
        if self.equations.is_empty() {
            return Err(GavinLmError::InvalidFitConfiguration { field: "equations" });
        }
        if self.x_data.is_empty() || self.observations.is_empty() {
            return Err(GavinLmError::EmptyObservations);
        }
        if self.unknowns.is_empty() {
            return Err(GavinLmError::NoParameters);
        }
        if self.initial_guess.len() != self.unknowns.len() {
            return Err(GavinLmError::ParameterLengthMismatch {
                expected: self.unknowns.len(),
                actual: self.initial_guess.len(),
            });
        }
        let expected_observations = self.equations.len().checked_mul(self.x_data.len()).ok_or(
            GavinLmError::OutputCountOverflow {
                samples: self.x_data.len(),
            },
        )?;
        if self.observations.len() != expected_observations {
            return Err(GavinLmError::SymbolicDataLengthMismatch {
                expected: expected_observations,
                actual: self.observations.len(),
            });
        }
        let sample_parameter =
            self.sample_parameter
                .ok_or(GavinLmError::InvalidFitConfiguration {
                    field: "sample_parameter",
                })?;

        let options = SymbolicProblemOptions::new()
            .with_variables(self.unknowns.clone())
            .with_equation_parameters(vec![sample_parameter])
            .with_lambdify_backend()
            .with_lambdify_frontend(self.frontend)
            .with_lambdify_execution_policy(self.execution_policy)
            .with_preparation_telemetry(self.preparation_telemetry);
        let mut prepared =
            PreparedSymbolicLeastSquaresProblem::from_expressions(self.equations, options)
                .map_err(|error| GavinLmError::SymbolicPreparation(Box::new(error)))?;
        if !self.positive_parameters.is_empty() {
            prepared
                .set_positive(&self.positive_parameters)
                .map_err(|error| GavinLmError::SymbolicPreparation(Box::new(error)))?;
        }
        let preparation = prepared.preparation_report().clone();
        let backend = prepared.backend_kind();
        let bound = prepared
            .bind_values_with_guess(
                DVector::from_element(1, 0.0),
                DVector::from_vec(self.initial_guess),
            )
            .map_err(|error| GavinLmError::SymbolicPreparation(Box::new(error)))?;
        let objective = SymbolicGavinObjective::new(&bound, &self.observations)?;
        let mut solver = LevenbergMarquardt::new(objective)
            .with_max_iterations(self.max_iterations)
            .with_tolerances(self.tolerances.0, self.tolerances.1, self.tolerances.2)
            .with_damping_factors(
                self.damping_factors.0,
                self.damping_factors.1,
                self.damping_factors.2,
            )
            .with_update_policy(self.update_policy)
            .with_telemetry(self.telemetry)
            .with_logging(self.logging);
        let initial = bound.params().clone();
        let sample_locations = DVector::from_vec(self.x_data.clone());
        let fit = if let Some(weights) = &self.weights {
            solver.lm_residuals_with_weights_detailed(
                initial,
                &sample_locations,
                &DVector::from_vec(weights.clone()),
            )?
        } else {
            solver.lm_residuals_detailed(initial, &sample_locations)?
        };
        let telemetry = solver.telemetry().cloned();

        // Residual mode has a zero target by definition. For fitting, R² is
        // instead computed against the observed data using the same final SSE.
        let unit_weights = vec![1.0; self.observations.len()];
        let weights = self.weights.as_deref().unwrap_or(&unit_weights);
        let weight_sum = weights.iter().sum::<f64>();
        let mean = self
            .observations
            .iter()
            .zip(weights)
            .map(|(value, weight)| value * weight)
            .sum::<f64>()
            / weight_sum;
        let total = self
            .observations
            .iter()
            .zip(weights)
            .map(|(value, weight)| weight * (value - mean).powi(2))
            .sum::<f64>();
        let mut fit = fit;
        fit.r_squared =
            (total > 0.0 && total.is_finite()).then(|| 1.0 - fit.sum_squared_residuals / total);

        Ok(GavinSymbolicFitResult {
            fit,
            parameter_names: self.unknowns,
            telemetry,
            backend,
            frontend: self.frontend,
            preparation,
        })
    }
}

impl Default for GavinSymbolicFitting {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::Nonlinear_systems::symbolic::SymbolicBackendKind;

    #[test]
    fn builder_prepares_and_solves_symbolic_fit_in_one_flow() {
        let x = (0..10).map(|index| index as f64 * 0.2).collect::<Vec<_>>();
        let y = x.iter().map(|x| 1.7 * x + 0.4).collect::<Vec<_>>();
        let report = GavinSymbolicFitting::new()
            .try_with_equation("a*x+b")
            .unwrap()
            .with_data(x, y)
            .with_sample_parameter("x")
            .with_unknowns(["a", "b"])
            .with_initial_guess(vec![1.2, 0.2])
            .with_telemetry(true)
            .with_preparation_telemetry(PreparationTelemetryMode::Collect)
            .try_solve()
            .unwrap();
        let atom_report = GavinSymbolicFitting::new()
            .try_with_equation("a*x+b")
            .unwrap()
            .with_data(
                (0..10).map(|index| index as f64 * 0.2).collect(),
                (0..10)
                    .map(|index| 1.7 * (index as f64 * 0.2) + 0.4)
                    .collect(),
            )
            .with_sample_parameter("x")
            .with_unknowns(["a", "b"])
            .with_initial_guess(vec![1.2, 0.2])
            .with_atom_native_frontend()
            .try_solve()
            .unwrap();

        assert_eq!(report.backend, SymbolicBackendKind::Lambdify);
        assert_eq!(report.frontend, SymbolicLambdifyFrontend::ExprLegacy);
        assert_eq!(report.fit.termination.is_converged(), true);
        assert!((report.fit.parameters[0] - 1.7).abs() < 1e-6);
        assert!((report.fit.parameters[1] - 0.4).abs() < 1e-6);
        assert!(report.fit.r_squared.unwrap() > 1.0 - 1e-12);
        assert!(report.preparation.preparation_duration.as_nanos() > 0);
        assert_eq!(
            atom_report.backend,
            crate::numerical::Nonlinear_systems::symbolic::SymbolicBackendKind::Lambdify
        );
        assert!((atom_report.fit.parameters[0] - report.fit.parameters[0]).abs() < 1e-10);
        assert!((atom_report.fit.parameters[1] - report.fit.parameters[1]).abs() < 1e-10);
    }

    #[test]
    fn builder_rejects_bad_observation_layout_with_typed_error() {
        let result = GavinSymbolicFitting::new()
            .try_with_equation("a*x")
            .unwrap()
            .with_data(vec![0.0, 1.0], vec![1.0])
            .with_sample_parameter("x")
            .with_unknowns(["a"])
            .with_initial_guess(vec![1.0])
            .try_solve();
        assert!(matches!(
            result,
            Err(GavinLmError::SymbolicDataLengthMismatch { .. })
        ));
    }

    #[test]
    fn builder_reports_iteration_exhaustion_without_telemetry() {
        let report = GavinSymbolicFitting::new()
            .try_with_equation("a*x+b")
            .unwrap()
            .with_data(vec![0.0, 1.0, 2.0], vec![1.0, 3.0, 5.0])
            .with_sample_parameter("x")
            .with_unknowns(["a", "b"])
            .with_initial_guess(vec![0.5, 0.5])
            .with_solver_controls(1, 1e-12, 1e-12, 0.1)
            .try_solve()
            .unwrap();

        assert_eq!(
            report.fit.termination,
            super::super::lm_gavin::GavinLmTermination::MaxIterations
        );
        assert!(!report.fit.termination.is_converged());
    }

    #[test]
    fn constant_observations_have_explicitly_undefined_r_squared() {
        let report = GavinSymbolicFitting::new()
            .try_with_equation("a*x+b")
            .unwrap()
            .with_data(vec![0.0, 1.0, 2.0, 3.0], vec![2.0; 4])
            .with_sample_parameter("x")
            .with_unknowns(["a", "b"])
            .with_initial_guess(vec![0.2, 1.8])
            .try_solve()
            .unwrap();

        assert!(report.fit.termination.is_converged());
        assert_eq!(report.fit.r_squared, None);
    }

    #[test]
    fn builder_applies_explicit_weights_to_fit_and_statistics() {
        let report = GavinSymbolicFitting::new()
            .try_with_equation("a*x+b")
            .unwrap()
            .with_data(vec![0.0, 1.0, 2.0], vec![1.0, 100.0, 5.0])
            .with_weights(vec![1.0, 0.0, 1.0])
            .with_sample_parameter("x")
            .with_unknowns(["a", "b"])
            .with_initial_guess(vec![1.0, 1.0])
            .try_solve()
            .unwrap();

        assert!((report.fit.parameters[0] - 2.0).abs() < 1e-6);
        assert!((report.fit.parameters[1] - 1.0).abs() < 1e-6);
        assert!(report.fit.r_squared.unwrap() > 1.0 - 1e-12);
    }

    #[test]
    fn positive_domain_rejects_trials_before_symbolic_evaluation() {
        let report = GavinSymbolicFitting::new()
            .try_with_equation("a*x")
            .unwrap()
            .with_data(vec![1.0, 2.0, 3.0], vec![-1.0, -2.0, -3.0])
            .with_sample_parameter("x")
            .with_unknowns(["a"])
            .with_initial_guess(vec![0.1])
            .with_positive_parameters(["a"])
            .with_solver_controls(12, 1e-12, 1e-12, 0.1)
            .with_telemetry(true)
            .try_solve()
            .unwrap();

        assert!(
            report.fit.termination.is_converged()
                || report.fit.termination
                    == super::super::lm_gavin::GavinLmTermination::MaxIterations
        );
        assert!(report.fit.parameters[0] > 0.0);
        assert!(report.telemetry.unwrap().rejected_steps > 0);
    }

    #[test]
    fn nonlinear_symbolic_fit_tracks_sym_fitting_solution() {
        use crate::numerical::optimization::sym_fitting::Fitting;

        let x = (0..30).map(|index| index as f64 * 0.1).collect::<Vec<_>>();
        let y = x.iter().map(|x| (0.65 * x).exp() + 1.4).collect::<Vec<_>>();
        let gavin = GavinSymbolicFitting::new()
            .try_with_equation("exp(a*x)+b")
            .unwrap()
            .with_data(x.clone(), y.clone())
            .with_sample_parameter("x")
            .with_unknowns(["a", "b"])
            .with_initial_guess(vec![0.45, 1.0])
            .try_solve()
            .unwrap();
        let reference = Fitting::new()
            .with_data(x, y)
            .with_equation_str("exp(a*x)+b".to_string())
            .with_arg("x".to_string())
            .with_unknowns(vec!["a".to_string(), "b".to_string()])
            .with_initial_guess(vec![0.45, 1.0])
            .try_build()
            .unwrap();

        let reference = reference.result.expect("sym_fitting should converge");
        assert!(gavin.fit.termination.is_converged());
        assert!((gavin.fit.parameters[0] - reference[0]).abs() < 2e-4);
        assert!((gavin.fit.parameters[1] - reference[1]).abs() < 2e-4);
    }
}
