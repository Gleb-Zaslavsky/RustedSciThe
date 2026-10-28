use crate::numerical::Nonlinear_systems::engine::SolveStatistics;
use crate::numerical::Nonlinear_systems::least_squares::{
    LeastSquaresError, LeastSquaresProblem, LeastSquaresTelemetryMode,
    LeastSquaresTerminationReason, LevenbergMarquardt, MinimizationReport,
    PreparedSymbolicLeastSquaresProblem,
};
use crate::numerical::Nonlinear_systems::symbolic::{
    LambdifyExecutionPolicy, SymbolicLambdifyFrontend, SymbolicProblemOptions,
};
use crate::symbolic::symbolic_engine::Expr;
use log::{Level, log};
use nalgebra::DVector;
use rayon::prelude::*;
use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Controls how independent data observations are evaluated by [`Fitting`].
///
/// This is separate from [`LambdifyExecutionPolicy`], which controls work
/// inside a single symbolic callback. The production default batches large
/// data sets without nesting Rayon dispatch. See `FITTING_STORY_TESTS.md`
/// (2026-10-24): two model families retain exact per-frontend solve counters
/// and results, with lower full-fit time at the cost of staging memory. The
/// existing solver telemetry schema is unchanged; callers can inspect the
/// selected observation policy separately.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FittingObservationPolicy {
    /// Batch at least 20,001 observations when the current Rayon pool has
    /// multiple workers and symbolic callbacks are sequential. Otherwise use
    /// the scalar path. The threshold is the smallest measured winning scale,
    /// not a universal hardware-independent break-even estimate.
    #[default]
    Auto,
    /// Evaluate observations serially, reusing one callback scratch per pass.
    /// This remains supported for small jobs and memory-constrained fits.
    Sequential,
    /// Evaluate observation blocks in parallel once the sample count reaches
    /// `min_observations`. A value of zero is treated as one. Work is split
    /// into 1,024-observation blocks. This currently allocates a full
    /// sample-major staging buffer before restoring the equation-major
    /// least-squares layout; its payload grows with observations, equations,
    /// and fitted parameters.
    Parallel { min_observations: usize },
}

const FITTING_OBSERVATION_BATCH_SIZE: usize = 1024;
const FITTING_OBSERVATION_AUTO_MIN: usize = 20_001;

impl FittingObservationPolicy {
    fn resolve(
        self,
        samples: usize,
        workers: usize,
        callback_policy: LambdifyExecutionPolicy,
    ) -> Result<Self, FittingError> {
        let callback_parallel = matches!(callback_policy, LambdifyExecutionPolicy::Parallel { .. });
        match self {
            Self::Parallel { .. } if callback_parallel => Err(FittingError::InvalidInput {
                field: "nested observation and callback parallelism",
            }),
            Self::Auto
                if !callback_parallel && workers > 1 && samples >= FITTING_OBSERVATION_AUTO_MIN =>
            {
                Ok(Self::Parallel {
                    min_observations: FITTING_OBSERVATION_AUTO_MIN,
                })
            }
            Self::Parallel { min_observations } if samples >= min_observations.max(1) => Ok(self),
            _ => Ok(Self::Sequential),
        }
    }
}

/// Typed failures from input validation, symbolic preparation, and fitting.
#[derive(Debug)]
pub enum FittingError {
    InvalidInput {
        field: &'static str,
    },
    EquationParse(String),
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },
    NonFiniteInput {
        field: &'static str,
        index: usize,
    },
    NoUnknowns,
    Preparation(LeastSquaresError),
    Solver(LeastSquaresError),
    DidNotConverge {
        termination: LeastSquaresTerminationReason,
        evaluations: usize,
        objective: f64,
    },
}

impl Display for FittingError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput { field } => write!(f, "invalid fitting input: {field}"),
            Self::EquationParse(message) => {
                write!(f, "failed to parse fitting equation: {message}")
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "fitting data length mismatch: expected {expected}, got {actual}"
                )
            }
            Self::NonFiniteInput { field, index } => {
                write!(f, "non-finite fitting input {field}[{index}]")
            }
            Self::NoUnknowns => write!(f, "no fitting coefficients were found"),
            Self::Preparation(source) => write!(f, "fitting preparation failed: {source}"),
            Self::Solver(source) => write!(f, "fitting solve failed: {source}"),
            Self::DidNotConverge {
                termination,
                evaluations,
                objective,
            } => write!(
                f,
                "fitting did not converge: {termination:?}; evaluations={evaluations}, objective={objective}"
            ),
        }
    }
}

impl Error for FittingError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Preparation(source) | Self::Solver(source) => Some(source),
            _ => None,
        }
    }
}

/// 1D fitting using Levenberg-Marquardt algorithm
/// This is a wrapper around the Levenberg-Marquardt algorithm.
/// It takes a symbolic expression and a set of data points and fits the expression to the data points.
pub struct Fitting {
    pub x_data: Vec<f64>,     // x data
    pub y_data: Vec<f64>,     // y data
    pub equations: Vec<Expr>, // equations to fit, flattened row-major when there are many
    pub arg: String,
    pub unknown_coeffs: Vec<String>,   // vector of variables
    pub initial_guess: Vec<f64>,       // initial guess
    pub max_iterations: Option<usize>, // maximum number of iterations
    pub tolerance: Option<f64>,        // tolerance
    pub f_tolerance: Option<f64>,
    pub g_tolerance: Option<f64>, // gradient tolerance
    pub scale_diag: Option<bool>,
    pub result: Option<DVector<f64>>,
    pub map_of_solutions: Option<HashMap<String, f64>>,
    pub r_ssquared: Option<f64>,
    /// Telemetry snapshot for the most recent numerical solve attempt.
    /// Check `availability` and `timings_collected` before interpreting zeros.
    pub last_statistics: Option<SolveStatistics>,
    telemetry_mode: LeastSquaresTelemetryMode,
    log_level: Option<Level>,
    symbolic_frontend: SymbolicLambdifyFrontend,
    lambdify_execution_policy: LambdifyExecutionPolicy,
    observation_policy: FittingObservationPolicy,
    /// Prepared rectangular frontend retained between repeated solves.
    prepared_least_squares: Option<PreparedSymbolicLeastSquaresProblem>,
}

impl Fitting {
    /// Invalidates the prepared frontend when its symbolic schema changes.
    ///
    /// Observation coordinates and target values are numeric callback inputs,
    /// so changing data clears the previous result but keeps symbolic preparation.
    fn invalidate_prepared(&mut self) {
        self.prepared_least_squares = None;
        self.clear_last_solution();
    }

    fn clear_last_solution(&mut self) {
        self.result = None;
        self.map_of_solutions = None;
        self.r_ssquared = None;
        self.last_statistics = None;
    }

    pub fn new() -> Self {
        Fitting {
            x_data: Vec::new(),
            y_data: Vec::new(),
            equations: vec![Expr::parse_expression("0")],
            unknown_coeffs: Vec::new(),
            arg: String::new(),
            initial_guess: Vec::new(),
            tolerance: None,
            f_tolerance: None,
            g_tolerance: None,
            scale_diag: None,
            max_iterations: None,
            result: None,
            map_of_solutions: None,
            r_ssquared: None,
            prepared_least_squares: None,
            last_statistics: None,
            telemetry_mode: LeastSquaresTelemetryMode::Off,
            log_level: None,
            symbolic_frontend: SymbolicLambdifyFrontend::ExprLegacy,
            lambdify_execution_policy: LambdifyExecutionPolicy::Sequential,
            observation_policy: FittingObservationPolicy::default(),
        }
    }

    /// Builder pattern: Set x data
    pub fn with_x_data(mut self, x_data: Vec<f64>) -> Self {
        self.clear_last_solution();
        self.x_data = x_data;
        self
    }

    /// Builder pattern: Set y data
    pub fn with_y_data(mut self, y_data: Vec<f64>) -> Self {
        self.clear_last_solution();
        self.y_data = y_data;
        self
    }

    /// Builder pattern: Set data (x and y together)
    pub fn with_data(mut self, x_data: Vec<f64>, y_data: Vec<f64>) -> Self {
        self.clear_last_solution();
        self.x_data = x_data;
        self.y_data = y_data;
        self
    }

    /// Builder pattern: Set equation from Expr
    pub fn with_equation(mut self, eq: Expr) -> Self {
        self.invalidate_prepared();
        self.equations = vec![eq];
        self
    }

    /// Builder pattern: Set target equations from a vector of Expr values.
    ///
    /// The equations are flattened in row-major order when residuals and
    /// predictions are generated.
    pub fn with_equations(mut self, eq_system: Vec<Expr>) -> Self {
        self.invalidate_prepared();
        self.equations = eq_system;
        self
    }

    /// Builder pattern: Set equation from string
    pub fn with_equation_str(mut self, eq_string: String) -> Self {
        self.invalidate_prepared();
        self.equations = vec![Expr::parse_expression(&eq_string)];
        self
    }

    /// Fallible equation-string builder for input that may be malformed.
    pub fn try_with_equation_str(mut self, equation: &str) -> Result<Self, FittingError> {
        let equation = Expr::try_parse_expression(equation)
            .map_err(|error| FittingError::EquationParse(error.to_string()))?;
        self.invalidate_prepared();
        self.equations = vec![equation];
        Ok(self)
    }

    /// Builder pattern: Set target equations from a vector of strings.
    pub fn with_equations_str(mut self, eq_system_string: Vec<String>) -> Self {
        self.invalidate_prepared();
        self.equations = eq_system_string
            .iter()
            .map(|x| Expr::parse_expression(x))
            .collect::<Vec<_>>();
        self
    }

    /// Builder pattern: Set polynomial equation of given degree
    pub fn with_polynomial(mut self, degree: usize, arg: String) -> Self {
        self.invalidate_prepared();
        let (eq, unknowns) = Expr::polyval(degree, &arg);
        self.equations = vec![eq];
        self.unknown_coeffs = unknowns;
        self.arg = arg;
        self
    }

    /// Builder pattern: Set unknown coefficients
    pub fn with_unknowns(mut self, unknowns: Vec<String>) -> Self {
        self.invalidate_prepared();
        self.unknown_coeffs = unknowns;
        self
    }

    /// Builder pattern: Set argument variable
    pub fn with_arg(mut self, arg: String) -> Self {
        self.invalidate_prepared();
        self.arg = arg;
        self
    }

    /// Builder pattern: Set initial guess
    pub fn with_initial_guess(mut self, initial_guess: Vec<f64>) -> Self {
        self.clear_last_solution();
        self.initial_guess = initial_guess;
        self
    }

    /// Builder pattern: Set tolerance
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.clear_last_solution();
        self.tolerance = Some(tolerance);
        self
    }

    /// Builder pattern: Set function tolerance
    pub fn with_f_tolerance(mut self, f_tolerance: f64) -> Self {
        self.clear_last_solution();
        self.f_tolerance = Some(f_tolerance);
        self
    }

    /// Builder pattern: Set gradient tolerance
    pub fn with_g_tolerance(mut self, g_tolerance: f64) -> Self {
        self.clear_last_solution();
        self.g_tolerance = Some(g_tolerance);
        self
    }

    /// Builder pattern: Set scale diagonal
    pub fn with_scale_diag(mut self, scale_diag: bool) -> Self {
        self.clear_last_solution();
        self.scale_diag = Some(scale_diag);
        self
    }

    /// Builder pattern: Set max iterations
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.clear_last_solution();
        self.max_iterations = Some(max_iterations);
        self
    }

    /// Enables optional counters or detailed least-squares timings.
    /// Telemetry is disabled by default and Off adds no clock/counter work.
    pub fn with_telemetry(mut self, mode: LeastSquaresTelemetryMode) -> Self {
        self.telemetry_mode = mode;
        self
    }

    /// Selects the in-process symbolic representation used during preparation.
    pub fn with_symbolic_frontend(mut self, frontend: SymbolicLambdifyFrontend) -> Self {
        self.invalidate_prepared();
        self.symbolic_frontend = frontend;
        self
    }

    /// Selects sequential or thresholded parallel callback evaluation.
    pub fn with_lambdify_execution_policy(mut self, policy: LambdifyExecutionPolicy) -> Self {
        self.invalidate_prepared();
        self.lambdify_execution_policy = policy;
        self
    }

    /// Selects production block-parallel evaluation across independent samples.
    ///
    /// This does not change the execution policy within a symbolic callback.
    /// `Auto` (the default) preserves explicit inner callback parallelism by
    /// keeping observations sequential. Explicit `Parallel` with parallel
    /// callbacks is a typed configuration error. Choose `Sequential` to avoid
    /// the extra sample-major staging buffer even for large data sets.
    pub fn with_observation_policy(mut self, policy: FittingObservationPolicy) -> Self {
        self.observation_policy = policy;
        self.clear_last_solution();
        self
    }

    /// Returns the requested observation policy, before automatic resolution.
    pub fn observation_policy(&self) -> FittingObservationPolicy {
        self.observation_policy
    }

    /// Resolves the policy for the current data and Rayon pool without solving.
    /// Call inside the same `ThreadPool::install` scope as `try_solve` when
    /// using a custom pool. This is configuration provenance, not telemetry.
    pub fn resolved_observation_policy(&self) -> Result<FittingObservationPolicy, FittingError> {
        // Small/sequential fits must not initialize a global Rayon pool merely
        // to discover that no observation dispatch is needed.
        let workers = if self.observation_policy == FittingObservationPolicy::Auto
            && self.x_data.len() >= FITTING_OBSERVATION_AUTO_MIN
            && self.lambdify_execution_policy == LambdifyExecutionPolicy::Sequential
        {
            rayon::current_num_threads()
        } else {
            1
        };
        self.observation_policy
            .resolve(self.x_data.len(), workers, self.lambdify_execution_policy)
    }

    /// Returns the symbolic frontend selected for the current prepared model.
    pub fn symbolic_frontend(&self) -> SymbolicLambdifyFrontend {
        self.symbolic_frontend
    }

    /// Enables fitting log records through the host application's `log` facade.
    /// No logger is installed or configured by this library.
    pub fn with_logging(mut self, level: Level) -> Self {
        self.log_level = Some(level);
        self
    }

    /// Disables fitting log records (the default).
    pub fn without_logging(mut self) -> Self {
        self.log_level = None;
        self
    }

    /// Statistics from the latest solve, if telemetry was enabled.
    pub fn last_statistics(&self) -> Option<&SolveStatistics> {
        self.last_statistics.as_ref()
    }

    /// Builder pattern: validate, prepare through the symbolic frontend, and solve.
    pub fn build(self) -> Self {
        self.try_build()
            .expect("fitting build failed; use try_build for typed errors")
    }

    /// Fallible builder that validates, prepares, and solves the fit.
    pub fn try_build(mut self) -> Result<Self, FittingError> {
        self.try_solve()?;
        Ok(self)
    }

    /// Validate the fitting data and infer unknown coefficient names.
    fn try_validate_and_infer(&mut self) -> Result<(), FittingError> {
        if self.x_data.is_empty() {
            return Err(FittingError::InvalidInput { field: "x_data" });
        }
        if self.y_data.is_empty() {
            return Err(FittingError::InvalidInput { field: "y_data" });
        }
        if self.initial_guess.is_empty() {
            return Err(FittingError::InvalidInput {
                field: "initial_guess",
            });
        }
        if self.arg.is_empty() {
            return Err(FittingError::InvalidInput { field: "arg" });
        }
        if self.equations.is_empty() {
            return Err(FittingError::InvalidInput { field: "equations" });
        }

        let equations = self.active_equations();
        let expected_y_len = self
            .x_data
            .len()
            .checked_mul(equations.len())
            .ok_or(FittingError::InvalidInput { field: "data size" })?;
        if self.y_data.len() != expected_y_len {
            return Err(FittingError::DimensionMismatch {
                expected: expected_y_len,
                actual: self.y_data.len(),
            });
        }
        for (field, values) in [
            ("x_data", self.x_data.as_slice()),
            ("y_data", self.y_data.as_slice()),
            ("initial_guess", self.initial_guess.as_slice()),
        ] {
            if let Some(index) = values.iter().position(|value| !value.is_finite()) {
                return Err(FittingError::NonFiniteInput { field, index });
            }
        }

        if self.unknown_coeffs.is_empty() {
            let mut args: Vec<String> = equations
                .iter()
                .flat_map(|eq| eq.all_arguments_are_variables())
                .collect();
            args.sort();
            args.dedup();
            // Remove the independent variable from unknowns
            args.retain(|x| x != &self.arg);
            if args.is_empty() {
                return Err(FittingError::NoUnknowns);
            }
            self.unknown_coeffs = args;
        }

        if self.unknown_coeffs.len() != self.initial_guess.len() {
            return Err(FittingError::DimensionMismatch {
                expected: self.unknown_coeffs.len(),
                actual: self.initial_guess.len(),
            });
        }
        Ok(())
    }
    pub fn set_fitting(
        &mut self,
        x_data: Vec<f64>,
        y_data: Vec<f64>,
        eq: Expr,
        unknowns: Option<Vec<String>>,
        arg: String,
        initial_guess: Vec<f64>,
        tolerance: Option<f64>,
        f_tolerance: Option<f64>,
        g_tolerance: Option<f64>,
        scale_diag: Option<bool>,
        max_iterations: Option<usize>,
    ) {
        self.invalidate_prepared();
        self.x_data = x_data;
        self.y_data = y_data;
        self.equations = vec![eq];
        self.arg = arg;
        self.initial_guess = initial_guess;
        self.tolerance = tolerance;
        self.g_tolerance = g_tolerance;
        self.max_iterations = max_iterations;
        self.f_tolerance = f_tolerance;
        self.scale_diag = scale_diag;
        let values = if let Some(values) = unknowns {
            values
        } else {
            let mut args: Vec<String> = self.equations[0].all_arguments_are_variables();
            args.sort();
            args.dedup();

            args
        };
        self.unknown_coeffs = values;
    }
    /// set fitting function as a vector of expressions
    pub fn set_fitting_system(
        &mut self,
        x_data: Vec<f64>,
        y_data: Vec<f64>,
        eq_system: Vec<Expr>,
        unknowns: Option<Vec<String>>,
        arg: String,
        initial_guess: Vec<f64>,
        tolerance: Option<f64>,
        f_tolerance: Option<f64>,
        g_tolerance: Option<f64>,
        scale_diag: Option<bool>,
        max_iterations: Option<usize>,
    ) {
        self.invalidate_prepared();
        self.x_data = x_data;
        self.y_data = y_data;
        self.equations = eq_system;
        self.arg = arg;
        self.initial_guess = initial_guess;
        self.tolerance = tolerance;
        self.g_tolerance = g_tolerance;
        self.max_iterations = max_iterations;
        self.f_tolerance = f_tolerance;
        self.scale_diag = scale_diag;
        let values = if let Some(values) = unknowns {
            values
        } else {
            let mut args: Vec<String> = self
                .equations
                .iter()
                .flat_map(|x| x.all_arguments_are_variables())
                .collect();
            args.sort();
            args.dedup();
            args
        };
        self.unknown_coeffs = values;
    }
    /// set fitting function as a string
    pub fn fitting_generate_from_str(
        &mut self,
        x_data: Vec<f64>,
        y_data: Vec<f64>,
        eq_string: String,
        unknowns: Option<Vec<String>>,
        arg: String,
        initial_guess: Vec<f64>,
        tolerance: Option<f64>, // tolerance: f64
        f_tolerance: Option<f64>,
        g_tolerance: Option<f64>,
        scale_diag: Option<bool>,
        max_iterations: Option<usize>, // max_iterations: usize,
    ) {
        self.try_fitting_generate_from_str(
            x_data,
            y_data,
            eq_string,
            unknowns,
            arg,
            initial_guess,
            tolerance,
            f_tolerance,
            g_tolerance,
            scale_diag,
            max_iterations,
        )
        .expect("fitting input is invalid; use try_fitting_generate_from_str for typed errors");
    }

    /// Fallible counterpart that preserves symbolic parse failures.
    #[allow(clippy::too_many_arguments)]
    pub fn try_fitting_generate_from_str(
        &mut self,
        x_data: Vec<f64>,
        y_data: Vec<f64>,
        eq_string: String,
        unknowns: Option<Vec<String>>,
        arg: String,
        initial_guess: Vec<f64>,
        tolerance: Option<f64>,
        f_tolerance: Option<f64>,
        g_tolerance: Option<f64>,
        scale_diag: Option<bool>,
        max_iterations: Option<usize>,
    ) -> Result<(), FittingError> {
        self.invalidate_prepared();
        let eq = Expr::try_parse_expression(&eq_string)
            .map_err(|error| FittingError::EquationParse(error.to_string()))?;
        self.set_fitting(
            x_data,
            y_data,
            eq,
            unknowns,
            arg,
            initial_guess,
            tolerance,
            f_tolerance,
            g_tolerance,
            scale_diag,
            max_iterations,
        );
        Ok(())
    }
    /// set fitting function as a vector of expressions
    pub fn fitting_generate_from_vec(
        &mut self,
        x_data: Vec<f64>,
        y_data: Vec<f64>,
        eq_system: Vec<Expr>,
        unknowns: Option<Vec<String>>,
        arg: String,
        initial_guess: Vec<f64>,
        tolerance: Option<f64>,
        f_tolerance: Option<f64>,
        g_tolerance: Option<f64>,
        scale_diag: Option<bool>,
        max_iterations: Option<usize>,
    ) {
        self.set_fitting_system(
            x_data,
            y_data,
            eq_system,
            unknowns,
            arg,
            initial_guess,
            tolerance,
            f_tolerance,
            g_tolerance,
            scale_diag,
            max_iterations,
        );
    }
    /// fit with polynomial of certain degree
    pub fn poly_fitting(
        &mut self,
        x_data: Vec<f64>,
        y_data: Vec<f64>,
        degree: usize,
        arg: String,
        initial_guess: Vec<f64>,
        tolerance: Option<f64>,
        f_tolerance: Option<f64>,
        g_tolerance: Option<f64>,
        scale_diag: Option<bool>,
        max_iterations: Option<usize>,
    ) {
        // create polynomial equation
        let (eq, unknowns) = Expr::polyval(degree, &arg);
        if let Some(level) = self.log_level {
            log!(level, "generated fitting polynomial: {eq}");
        }
        self.set_fitting(
            x_data,
            y_data,
            eq,
            Some(unknowns),
            arg,
            initial_guess,
            tolerance,
            f_tolerance,
            g_tolerance,
            scale_diag,
            max_iterations,
        );
    }
    pub fn fit_linear(&mut self, x_data: Vec<f64>, y_data: Vec<f64>, guess: (f64, f64)) {
        self.poly_fitting(
            x_data,
            y_data,
            1,
            "x".to_string(),
            vec![guess.0, guess.1],
            None,
            None,
            None,
            None,
            None,
        );
    }
    fn try_prepare_least_squares(&mut self) -> Result<(), FittingError> {
        self.resolved_observation_policy()?;
        let equations = self.active_equations();
        let options = SymbolicProblemOptions::new()
            .with_variables(self.unknown_coeffs.clone())
            .with_equation_parameters(vec![self.arg.clone()])
            .with_lambdify_backend()
            .with_lambdify_frontend(self.symbolic_frontend)
            .with_lambdify_execution_policy(self.lambdify_execution_policy);
        self.prepared_least_squares = Some(
            PreparedSymbolicLeastSquaresProblem::from_expressions(equations.to_vec(), options)
                .map_err(|error| FittingError::Preparation(error.into()))?,
        );
        Ok(())
    }

    /// Fallible fitting entry point. A failed attempt clears any previous
    /// solution before doing validation/preparation, so stale results cannot
    /// be observed as the output of the latest request.
    pub fn try_solve(&mut self) -> Result<MinimizationReport, FittingError> {
        self.clear_last_solution();
        self.try_validate_and_infer()?;
        // Validate every attempt, including reuse after a policy-only change.
        // Such a change does not invalidate the prepared symbolic model.
        let observation_policy = self.resolved_observation_policy()?;
        let mut solver = LevenbergMarquardt::new();
        if let Some(max_iterations) = self.max_iterations {
            solver = solver
                .try_with_max_iterations(max_iterations)
                .map_err(FittingError::Solver)?;
        }
        if let Some(tolerance) = self.tolerance {
            solver = solver
                .try_with_xtol(tolerance)
                .map_err(FittingError::Solver)?;
        }
        if let Some(g_tolerance) = self.g_tolerance {
            solver = solver
                .try_with_gtol(g_tolerance)
                .map_err(FittingError::Solver)?;
        }
        if let Some(f_tolerance) = self.f_tolerance {
            solver = solver
                .try_with_ftol(f_tolerance)
                .map_err(FittingError::Solver)?;
        }
        solver = solver
            .with_scale_diag(self.scale_diag.unwrap_or(true))
            .with_telemetry(self.telemetry_mode);

        if self.prepared_least_squares.is_none() {
            self.try_prepare_least_squares()?;
        }
        let prepared = self
            .prepared_least_squares
            .as_ref()
            .ok_or(FittingError::InvalidInput {
                field: "prepared model",
            })?;
        let bound = prepared
            .bind_values_with_guess(
                DVector::zeros(1),
                DVector::from_vec(self.initial_guess.clone()),
            )
            .map_err(|error| FittingError::Solver(error.into()))?;
        let problem = FittingLeastSquaresProblem::new(
            bound,
            &self.x_data,
            &self.y_data,
            self.active_equations().len(),
            observation_policy,
        );

        let (problem, report) = solver.minimize(problem);
        self.last_statistics = Some(report.statistics.clone());
        if let Some(error) = report.error.clone() {
            return Err(FittingError::Solver(error));
        }
        if !report.termination.was_successful() {
            return Err(FittingError::DidNotConverge {
                termination: report.termination,
                evaluations: report.number_of_evaluations,
                objective: report.objective_function,
            });
        }

        let solution = problem.params();
        let solution_map = self
            .unknown_coeffs
            .iter()
            .cloned()
            .zip(solution.iter().copied())
            .collect::<HashMap<_, _>>();
        self.result = Some(solution);
        self.map_of_solutions = Some(solution_map);
        self.r_ssquared = Some(r_squared(
            &self.y_data,
            &evaluate_equations(
                self.active_equations(),
                &self.x_data,
                self.map_of_solutions.as_ref().expect("just assigned"),
            ),
        ));
        if let Some(level) = self.log_level {
            log!(level, "fitting termination: {:?}", report.termination);
            log!(
                level,
                "fitting evaluations: {}",
                report.number_of_evaluations
            );
            log!(level, "fitting objective: {}", report.objective_function);
        }
        Ok(report)
    }

    /// Compatibility-friendly concise solve name; failures are returned,
    /// never printed and discarded.
    pub fn solve(&mut self) -> Result<MinimizationReport, FittingError> {
        self.try_solve()
    }
    /// for those who din't want to mess with multiple parameters

    pub fn easy_fitting(
        &mut self,
        x_data: Vec<f64>,
        y_data: Vec<f64>,
        eq_string: String,
        unknowns: Option<Vec<String>>,
        arg: String,
        initial_guess: Vec<f64>,
    ) -> Result<MinimizationReport, FittingError> {
        self.fitting_generate_from_str(
            x_data,
            y_data,
            eq_string,
            unknowns,
            arg,
            initial_guess,
            None,
            None,
            None,
            None,
            None,
        );
        self.try_solve()
    }
    /// extrapolate or interpolate function for arbitrary x values
    pub fn extra_interpolate(&self, x_values: Vec<f64>) -> Vec<f64> {
        evaluate_equations(
            self.active_equations(),
            &x_values,
            self.map_of_solutions.as_ref().expect("fit has no solution"),
        )
    }
    pub fn get_r_squared(&self) -> Option<f64> {
        self.r_ssquared
    }

    /// Return the coefficient of determination for the last successful fit.
    pub fn r_squared(&self) -> Option<f64> {
        self.get_r_squared()
    }

    /// Short alias for [`r_squared`](Self::r_squared).
    pub fn r2(&self) -> Option<f64> {
        self.r_squared()
    }

    pub fn get_map_of_solutions(&self) -> Option<HashMap<String, f64>> {
        self.map_of_solutions.clone()
    }

    /// Return the fitted parameter map in a backend-friendly form.
    ///
    /// This is the preferred accessor for new code because it reads the same
    /// way on the classic LM path and on the higher-level wrappers.
    pub fn solution_map(&self) -> Option<HashMap<String, f64>> {
        self.get_map_of_solutions()
    }

    /// Borrow the fitted parameter map without cloning it.
    pub fn solution_map_ref(&self) -> Option<&HashMap<String, f64>> {
        self.map_of_solutions.as_ref()
    }

    fn active_equations(&self) -> &[Expr] {
        &self.equations
    }
}

/// Evaluates one prepared symbolic model across all observations.
///
/// The symbolic equation count stays independent of the number of data points;
/// each observation is supplied as a numeric equation-parameter value.
struct FittingLeastSquaresProblem<'a> {
    model: crate::numerical::Nonlinear_systems::least_squares::BoundSymbolicLeastSquaresProblem<'a>,
    params: DVector<f64>,
    x_data: &'a [f64],
    y_data: &'a [f64],
    equation_count: usize,
    observation_policy: FittingObservationPolicy,
}

impl<'a> FittingLeastSquaresProblem<'a> {
    fn new(
        model: crate::numerical::Nonlinear_systems::least_squares::BoundSymbolicLeastSquaresProblem<
            'a,
        >,
        x_data: &'a [f64],
        y_data: &'a [f64],
        equation_count: usize,
        observation_policy: FittingObservationPolicy,
    ) -> Self {
        let params = model.params();
        Self {
            model,
            params,
            x_data,
            y_data,
            equation_count,
            observation_policy,
        }
    }

    fn evaluate_residuals(&self, params: &DVector<f64>) -> Result<DVector<f64>, LeastSquaresError> {
        if self.should_parallelize_observations() {
            return self.evaluate_residuals_parallel(params);
        }
        let mut residuals = DVector::zeros(self.y_data.len());
        let mut point_parameter = DVector::zeros(1);
        let mut point_residual = DVector::zeros(self.equation_count);
        for (point_index, &x) in self.x_data.iter().enumerate() {
            point_parameter[0] = x;
            self.model
                .residual_into_with_parameter_values(params, &point_parameter, &mut point_residual)
                .map_err(|error| LeastSquaresError::Problem(error.into()))?;
            for equation_index in 0..self.equation_count {
                let output_index = equation_index * self.x_data.len() + point_index;
                residuals[output_index] =
                    point_residual[equation_index] - self.y_data[output_index];
            }
        }
        Ok(residuals)
    }

    fn should_parallelize_observations(&self) -> bool {
        match self.observation_policy {
            // Public Auto is resolved once before constructing this adapter.
            FittingObservationPolicy::Auto => false,
            FittingObservationPolicy::Sequential => false,
            FittingObservationPolicy::Parallel { min_observations } => {
                self.x_data.len() >= min_observations.max(1)
            }
        }
    }

    fn evaluate_residuals_parallel(
        &self,
        params: &DVector<f64>,
    ) -> Result<DVector<f64>, LeastSquaresError> {
        // Rayon workers need disjoint contiguous output chunks. Stage in
        // sample-major order, then scatter once into the public equation-major
        // layout; this trades an additional N * equation_count buffer for
        // lock-free block writes.
        let mut sample_major = vec![0.0; self.x_data.len() * self.equation_count];
        self.x_data
            .par_chunks(FITTING_OBSERVATION_BATCH_SIZE)
            .enumerate()
            .zip(sample_major.par_chunks_mut(FITTING_OBSERVATION_BATCH_SIZE * self.equation_count))
            .try_for_each_init(
                || (DVector::zeros(1), DVector::zeros(self.equation_count)),
                |(point_parameter, point_residual), ((block_index, x_block), output_block)| {
                    let block_start = block_index * FITTING_OBSERVATION_BATCH_SIZE;
                    for (sample_index, &x) in x_block.iter().enumerate() {
                        point_parameter[0] = x;
                        self.model
                            .residual_into_with_parameter_values(
                                params,
                                point_parameter,
                                point_residual,
                            )
                            .map_err(|error| LeastSquaresError::Problem(error.into()))?;
                        for equation_index in 0..self.equation_count {
                            output_block[sample_index * self.equation_count + equation_index] =
                                point_residual[equation_index]
                                    - self.y_data[equation_index * self.x_data.len()
                                        + block_start
                                        + sample_index];
                        }
                    }
                    Ok::<_, LeastSquaresError>(())
                },
            )?;

        let mut residuals = DVector::zeros(self.y_data.len());
        for sample_index in 0..self.x_data.len() {
            for equation_index in 0..self.equation_count {
                residuals[equation_index * self.x_data.len() + sample_index] =
                    sample_major[sample_index * self.equation_count + equation_index];
            }
        }
        Ok(residuals)
    }

    fn evaluate_jacobian(
        &self,
        params: &DVector<f64>,
    ) -> Result<nalgebra::DMatrix<f64>, LeastSquaresError> {
        if self.should_parallelize_observations() {
            return self.evaluate_jacobian_parallel(params);
        }
        let mut jacobian = nalgebra::DMatrix::zeros(self.y_data.len(), params.len());
        let mut point_parameter = DVector::zeros(1);
        let mut point_jacobian = nalgebra::DMatrix::zeros(self.equation_count, params.len());
        for (point_index, &x) in self.x_data.iter().enumerate() {
            point_parameter[0] = x;
            self.model
                .jacobian_into_with_parameter_values(params, &point_parameter, &mut point_jacobian)
                .map_err(|error| LeastSquaresError::Problem(error.into()))?;
            for equation_index in 0..self.equation_count {
                let output_index = equation_index * self.x_data.len() + point_index;
                for parameter_index in 0..params.len() {
                    jacobian[(output_index, parameter_index)] =
                        point_jacobian[(equation_index, parameter_index)];
                }
            }
        }
        Ok(jacobian)
    }

    fn evaluate_jacobian_parallel(
        &self,
        params: &DVector<f64>,
    ) -> Result<nalgebra::DMatrix<f64>, LeastSquaresError> {
        let row_width = self.equation_count * params.len();
        // The contiguous staging layout lets workers write independent blocks;
        // restoring the existing row order costs one extra buffer of
        // observations * equations * parameters values.
        let mut sample_major = vec![0.0; self.x_data.len() * row_width];
        self.x_data
            .par_chunks(FITTING_OBSERVATION_BATCH_SIZE)
            .enumerate()
            .zip(sample_major.par_chunks_mut(FITTING_OBSERVATION_BATCH_SIZE * row_width))
            .try_for_each_init(
                || {
                    (
                        DVector::zeros(1),
                        nalgebra::DMatrix::zeros(self.equation_count, params.len()),
                    )
                },
                |(point_parameter, point_jacobian), ((_, x_block), output_block)| {
                    for (sample_index, &x) in x_block.iter().enumerate() {
                        point_parameter[0] = x;
                        self.model
                            .jacobian_into_with_parameter_values(
                                params,
                                point_parameter,
                                point_jacobian,
                            )
                            .map_err(|error| LeastSquaresError::Problem(error.into()))?;
                        for equation_index in 0..self.equation_count {
                            for parameter_index in 0..params.len() {
                                let index = sample_index * row_width
                                    + equation_index * params.len()
                                    + parameter_index;
                                output_block[index] =
                                    point_jacobian[(equation_index, parameter_index)];
                            }
                        }
                    }
                    Ok::<_, LeastSquaresError>(())
                },
            )?;

        let mut jacobian = nalgebra::DMatrix::zeros(self.y_data.len(), params.len());
        for sample_index in 0..self.x_data.len() {
            for equation_index in 0..self.equation_count {
                let output_row = equation_index * self.x_data.len() + sample_index;
                let sample_offset = sample_index * row_width + equation_index * params.len();
                for parameter_index in 0..params.len() {
                    jacobian[(output_row, parameter_index)] =
                        sample_major[sample_offset + parameter_index];
                }
            }
        }
        Ok(jacobian)
    }
}

impl LeastSquaresProblem for FittingLeastSquaresProblem<'_> {
    fn set_params(&mut self, params: &DVector<f64>) {
        self.params.copy_from(params);
    }
    fn params(&self) -> DVector<f64> {
        self.params.clone()
    }
    fn residuals(&self) -> Option<DVector<f64>> {
        self.try_residuals().ok()
    }
    fn jacobian(&self) -> Option<nalgebra::DMatrix<f64>> {
        self.try_jacobian().ok()
    }
    fn try_residuals(&self) -> Result<DVector<f64>, LeastSquaresError> {
        self.evaluate_residuals(&self.params)
    }
    fn try_jacobian(&self) -> Result<nalgebra::DMatrix<f64>, LeastSquaresError> {
        self.evaluate_jacobian(&self.params)
    }
}

fn evaluate_equations(
    eq_system: &[Expr],
    x_values: &[f64],
    map_of_solutions: &HashMap<String, f64>,
) -> Vec<f64> {
    let mut y_pred = Vec::with_capacity(eq_system.len() * x_values.len());
    for eq in eq_system {
        let eq = eq.clone().set_variable_from_map(map_of_solutions);
        let eq_fun = eq.lambdify1D();
        for x in x_values {
            y_pred.push(eq_fun(*x));
        }
    }
    y_pred
}

pub fn r_squared(y_data: &[f64], y_pred: &[f64]) -> f64 {
    let y_mean = y_data.iter().sum::<f64>() / y_data.len() as f64;
    let ss_tot = y_data.iter().map(|y| (y - y_mean).powi(2)).sum::<f64>();
    let ss_res = y_data
        .iter()
        .zip(y_pred.iter())
        .map(|(y, y_pred)| (y - y_pred).powi(2))
        .sum::<f64>();
    let r_squared = 1.0 - ss_res / ss_tot;
    r_squared
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn observation_auto_respects_scale_workers_and_explicit_callback_policy() {
        use FittingObservationPolicy::{Auto, Parallel, Sequential};
        let callback_seq = LambdifyExecutionPolicy::Sequential;
        let callback_par = LambdifyExecutionPolicy::Parallel { min_work: 1 };
        assert_eq!(Fitting::new().observation_policy(), Auto);
        for (samples, workers, callback, expected) in [
            (20_000, 8, callback_seq, Sequential),
            (20_001, 1, callback_seq, Sequential),
            (20_001, 8, callback_par, Sequential),
            (
                20_001,
                8,
                callback_seq,
                Parallel {
                    min_observations: 20_001,
                },
            ),
        ] {
            assert_eq!(Auto.resolve(samples, workers, callback).unwrap(), expected);
        }
        assert_eq!(
            Sequential.resolve(100_001, 8, callback_seq).unwrap(),
            Sequential
        );
        assert_eq!(
            Parallel {
                min_observations: 20
            }
            .resolve(19, 8, callback_seq)
            .unwrap(),
            Sequential
        );
        // Explicit nested policies are rejected even below the threshold.
        assert!(
            Parallel {
                min_observations: 20
            }
            .resolve(1, 8, callback_par)
            .is_err()
        );
    }

    #[test]
    fn default_large_fit_matches_scalar_results_and_all_telemetry_modes() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        pool.install(|| {
            let x = (0..20_001).map(|i| i as f64 / 20_000.0).collect::<Vec<_>>();
            for frontend in [
                SymbolicLambdifyFrontend::ExprLegacy,
                SymbolicLambdifyFrontend::AtomViewNative,
            ] {
                for mode in [
                    LeastSquaresTelemetryMode::Off,
                    LeastSquaresTelemetryMode::Counters,
                    LeastSquaresTelemetryMode::Timings,
                    LeastSquaresTelemetryMode::Detailed,
                ] {
                    let make = || {
                        Fitting::new()
                            .with_data(x.clone(), x.iter().map(|v| 2.0 * v + 1.0).collect())
                            .with_equation(Expr::parse_expression("a*x+b"))
                            .with_arg("x".into())
                            .with_unknowns(vec!["a".into(), "b".into()])
                            .with_initial_guess(vec![1.5, 0.7])
                            .with_symbolic_frontend(frontend)
                            .with_telemetry(mode)
                    };
                    let mut automatic = make();
                    assert!(matches!(
                        automatic.resolved_observation_policy().unwrap(),
                        FittingObservationPolicy::Parallel { .. }
                    ));
                    let mut scalar =
                        make().with_observation_policy(FittingObservationPolicy::Sequential);
                    let actual = automatic.try_solve().unwrap();
                    let expected = scalar.try_solve().unwrap();
                    assert_eq!(automatic.result, scalar.result);
                    assert_eq!(actual.termination, expected.termination);
                    assert_eq!(actual.number_of_evaluations, expected.number_of_evaluations);
                    let a = actual.statistics;
                    let b = expected.statistics;
                    assert_eq!(a.availability, b.availability);
                    assert_eq!(a.timings_collected, b.timings_collected);
                    assert_eq!(
                        (
                            a.iterations,
                            a.residual_evaluations,
                            a.jacobian_evaluations,
                            a.linear_factorizations,
                            a.linear_solves
                        ),
                        (
                            b.iterations,
                            b.residual_evaluations,
                            b.jacobian_evaluations,
                            b.linear_factorizations,
                            b.linear_solves
                        )
                    );
                }
            }
        });
    }

    #[test]
    fn observation_policy_change_revalidates_a_prepared_fit() {
        let mut fit = Fitting::new()
            .with_data(vec![0.0, 1.0, 2.0], vec![0.0, 2.0, 4.0])
            .with_equation(Expr::parse_expression("a*x"))
            .with_arg("x".into())
            .with_unknowns(vec!["a".into()])
            .with_initial_guess(vec![0.5])
            .with_lambdify_execution_policy(LambdifyExecutionPolicy::Parallel { min_work: 1 })
            .try_build()
            .unwrap();
        assert!(fit.prepared_least_squares.is_some());
        fit = fit.with_observation_policy(FittingObservationPolicy::Parallel {
            min_observations: 1,
        });
        assert!(matches!(
            fit.try_solve(),
            Err(FittingError::InvalidInput {
                field: "nested observation and callback parallelism"
            })
        ));
        assert!(fit.result.is_none());
        // Recover without rebuilding symbolic expressions after the invalid attempt.
        fit = fit.with_observation_policy(FittingObservationPolicy::Auto);
        fit.try_solve().unwrap();
        assert!(fit.result.is_some());
    }

    #[test]
    fn fitting_prepares_equations_once_and_evaluates_observations_numerically() {
        let mut fitting = Fitting::new()
            .with_data(vec![0.0, 1.0, 2.0], vec![1.0, 3.0, 5.0, -2.0, -1.0, 0.0])
            .with_equations(vec![
                Expr::parse_expression("a*x + b"),
                Expr::parse_expression("c*x + d"),
            ])
            .with_arg("x".to_string())
            .with_unknowns(vec!["a".into(), "b".into(), "c".into(), "d".into()])
            .with_initial_guess(vec![0.0; 4]);

        fitting.try_validate_and_infer().unwrap();
        fitting.try_prepare_least_squares().unwrap();
        let prepared = fitting.prepared_least_squares.as_ref().unwrap();

        assert_eq!(prepared.residual_count(), 2);
        assert_eq!(prepared.jacobian_shape(), (2, 4));

        let bound = prepared
            .bind_values_with_guess(DVector::zeros(1), DVector::from_element(4, 0.0))
            .unwrap();
        let problem = FittingLeastSquaresProblem::new(
            bound,
            &fitting.x_data,
            &fitting.y_data,
            fitting.equations.len(),
            FittingObservationPolicy::Sequential,
        );
        assert_eq!(
            problem.try_residuals().unwrap(),
            DVector::from_vec(vec![-1.0, -3.0, -5.0, 2.0, 1.0, 0.0])
        );
        assert_eq!(
            problem.try_jacobian().unwrap(),
            nalgebra::DMatrix::from_row_slice(
                6,
                4,
                &[
                    0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                    0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 1.0,
                ],
            )
        );
        assert!(matches!(
            problem.model.residual_with_parameter_values(
                &DVector::from_element(4, 0.0),
                &DVector::from_element(1, f64::NAN),
            ),
            Err(
                crate::numerical::Nonlinear_systems::error::SolveError::NonFiniteParameterValue {
                    index: 0,
                    ..
                }
            )
        ));
    }

    #[test]
    fn parallel_observation_blocks_match_scalar_residual_jacobian_and_fit() {
        // Cross multiple internal 1,024-point chunks so this exercises actual
        // worker-level parallelism, not only the staging/scatter path.
        let x_data = (0..2049)
            .map(|index| index as f64 / 100.0)
            .collect::<Vec<_>>();
        let y_data = x_data
            .iter()
            .map(|&x| 2.0 * x - 1.0)
            .chain(x_data.iter().map(|&x| -0.5 * x + 3.0))
            .collect::<Vec<_>>();
        let equations = vec![
            Expr::parse_expression("a*x+b"),
            Expr::parse_expression("c*x+d"),
        ];
        let variables = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ];
        for frontend in [
            SymbolicLambdifyFrontend::ExprLegacy,
            SymbolicLambdifyFrontend::AtomViewNative,
        ] {
            let initial = DVector::from_vec(vec![1.7, -0.7, -0.3, 2.7]);
            let prepared = PreparedSymbolicLeastSquaresProblem::from_expressions(
                equations.clone(),
                SymbolicProblemOptions::new()
                    .with_variables(variables.clone())
                    .with_equation_parameters(vec!["x".to_string()])
                    .with_lambdify_backend()
                    .with_lambdify_frontend(frontend)
                    .with_lambdify_execution_policy(LambdifyExecutionPolicy::Sequential),
            )
            .unwrap();
            let sequential = FittingLeastSquaresProblem::new(
                prepared
                    .bind_values_with_guess(DVector::zeros(1), initial.clone())
                    .unwrap(),
                &x_data,
                &y_data,
                equations.len(),
                FittingObservationPolicy::Sequential,
            );
            let parallel = FittingLeastSquaresProblem::new(
                prepared
                    .bind_values_with_guess(DVector::zeros(1), initial)
                    .unwrap(),
                &x_data,
                &y_data,
                equations.len(),
                FittingObservationPolicy::Parallel {
                    min_observations: 1,
                },
            );

            assert_eq!(
                sequential.try_residuals().unwrap(),
                parallel.try_residuals().unwrap()
            );
            assert_eq!(
                sequential.try_jacobian().unwrap(),
                parallel.try_jacobian().unwrap()
            );

            let make_fit = |observation_policy| {
                Fitting::new()
                    .with_data(x_data.clone(), y_data.clone())
                    .with_equations(equations.clone())
                    .with_arg("x".to_string())
                    .with_unknowns(variables.clone())
                    .with_initial_guess(vec![1.7, -0.7, -0.3, 2.7])
                    .with_symbolic_frontend(frontend)
                    .with_observation_policy(observation_policy)
                    .with_telemetry(LeastSquaresTelemetryMode::Counters)
            };
            let scalar_fit = make_fit(FittingObservationPolicy::Sequential)
                .try_build()
                .unwrap();
            let parallel_fit = make_fit(FittingObservationPolicy::Parallel {
                min_observations: 1,
            })
            .try_build()
            .unwrap();
            assert_eq!(scalar_fit.result, parallel_fit.result);
            let scalar_stats = scalar_fit.last_statistics().unwrap();
            let parallel_stats = parallel_fit.last_statistics().unwrap();
            assert_eq!(scalar_stats.availability, parallel_stats.availability);
            assert_eq!(
                scalar_stats.timings_collected,
                parallel_stats.timings_collected
            );
            assert_eq!(scalar_stats.iterations, parallel_stats.iterations);
            assert_eq!(
                scalar_stats.residual_evaluations,
                parallel_stats.residual_evaluations
            );
            assert_eq!(
                scalar_stats.jacobian_evaluations,
                parallel_stats.jacobian_evaluations
            );
            assert_eq!(
                scalar_stats.linear_factorizations,
                parallel_stats.linear_factorizations
            );
            assert_eq!(scalar_stats.linear_solves, parallel_stats.linear_solves);
            assert_relative_eq!(
                scalar_fit.r_squared().unwrap(),
                parallel_fit.r_squared().unwrap()
            );
        }
    }

    #[test]
    fn fitting_rejects_nested_observation_and_callback_parallelism() {
        let result = Fitting::new()
            .with_data(vec![0.0, 1.0], vec![0.0, 1.0])
            .with_equation(Expr::parse_expression("a*x"))
            .with_arg("x".to_string())
            .with_unknowns(vec!["a".to_string()])
            .with_initial_guess(vec![0.5])
            .with_observation_policy(FittingObservationPolicy::Parallel {
                min_observations: 1,
            })
            .with_lambdify_execution_policy(LambdifyExecutionPolicy::Parallel { min_work: 1 })
            .try_build();

        assert!(matches!(
            result,
            Err(FittingError::InvalidInput {
                field: "nested observation and callback parallelism"
            })
        ));
    }

    #[test]
    fn parallel_observation_blocks_preserve_typed_callback_failures() {
        let mut x_data = vec![1.0; 2049];
        x_data[1024] = -1.0;
        let y_data = vec![0.0; x_data.len()];
        let model = PreparedSymbolicLeastSquaresProblem::from_expressions(
            vec![Expr::parse_expression("a*ln(x)")],
            SymbolicProblemOptions::new()
                .with_variables(vec!["a".to_string()])
                .with_equation_parameters(vec!["x".to_string()])
                .with_lambdify_frontend(SymbolicLambdifyFrontend::ExprLegacy)
                .with_lambdify_execution_policy(LambdifyExecutionPolicy::Sequential),
        )
        .expect("log model should prepare");

        let evaluate = |observation_policy| {
            let bound = model
                .bind_values_with_guess(DVector::zeros(1), DVector::from_vec(vec![1.0]))
                .expect("model should bind");
            FittingLeastSquaresProblem::new(bound, &x_data, &y_data, 1, observation_policy)
                .try_residuals()
        };

        let sequential_error = evaluate(FittingObservationPolicy::Sequential)
            .expect_err("negative logarithm input should fail");
        let parallel_error = evaluate(FittingObservationPolicy::Parallel {
            min_observations: 1,
        })
        .expect_err("negative logarithm input should fail in observation workers");
        assert!(matches!(
            sequential_error,
            LeastSquaresError::Problem(
                crate::numerical::Nonlinear_systems::error::SolveError::ResidualEvaluation(_)
            )
        ));
        assert!(matches!(
            parallel_error,
            LeastSquaresError::Problem(
                crate::numerical::Nonlinear_systems::error::SolveError::ResidualEvaluation(_)
            )
        ));
    }

    #[test]
    #[ignore = "release A/B story: large fitting workloads, sequential versus observation batches"]
    fn fitting_observation_batch_ab_large_workloads() {
        use std::time::Instant;

        for sample_count in [20_001, 100_001] {
            let x_data = (0..sample_count)
                .map(|index| 20.0 * index as f64 / (sample_count - 1) as f64)
                .collect::<Vec<_>>();
            let y_data = x_data
                .iter()
                .map(|&x| 2.0 * (-x / 0.5_f64).exp() + (x / 2.0_f64).sin())
                .collect::<Vec<_>>();

            for frontend in [
                SymbolicLambdifyFrontend::ExprLegacy,
                SymbolicLambdifyFrontend::AtomViewNative,
            ] {
                for observation_policy in [
                    FittingObservationPolicy::Sequential,
                    FittingObservationPolicy::Parallel {
                        min_observations: 256,
                    },
                ] {
                    let mut fit = Fitting::new()
                        .try_with_equation_str("a*exp(-x/tau)+c*sin(x/omega)")
                        .unwrap()
                        .with_data(x_data.clone(), y_data.clone())
                        .with_arg("x".into())
                        .with_unknowns(["a", "tau", "c", "omega"].map(str::to_owned).to_vec())
                        .with_initial_guess(vec![1.9, 0.55, 0.9, 1.9])
                        .with_symbolic_frontend(frontend)
                        .with_lambdify_execution_policy(LambdifyExecutionPolicy::Sequential)
                        .with_observation_policy(observation_policy)
                        .with_telemetry(LeastSquaresTelemetryMode::Detailed);
                    fit.try_validate_and_infer().unwrap();
                    let preparation_started = Instant::now();
                    fit.try_prepare_least_squares().unwrap();
                    let preparation_ms = preparation_started.elapsed().as_secs_f64() * 1_000.0;
                    let solve_started = Instant::now();
                    fit.try_solve().unwrap();
                    let solve_wall_ms = solve_started.elapsed().as_secs_f64() * 1_000.0;
                    let stats = fit.last_statistics().unwrap();
                    println!(
                        "FIT_BATCH_AB samples={sample_count} frontend={frontend:?} policy={observation_policy:?} preparation_ms={preparation_ms:.3} solve_wall_ms={solve_wall_ms:.3} residual_ms={:.3} jacobian_ms={:.3} iterations={} residual_evaluations={} jacobian_evaluations={} factorizations={} solves={}",
                        stats.residual_duration.as_secs_f64() * 1_000.0,
                        stats.jacobian_duration.as_secs_f64() * 1_000.0,
                        stats.iterations,
                        stats.residual_evaluations,
                        stats.jacobian_evaluations,
                        stats.linear_factorizations,
                        stats.linear_solves,
                    );
                    assert!(fit.r_squared().is_some_and(|r2| r2 > 0.99));
                }
            }
        }
    }

    #[test]
    #[ignore = "release A/B story: second nonlinear, multi-equation fitting family"]
    fn fitting_observation_batch_ab_two_response_model() {
        use std::time::Instant;

        let sample_count = 100_001;
        let x_data = (0..sample_count)
            .map(|index| 4.0 * index as f64 / (sample_count - 1) as f64)
            .collect::<Vec<_>>();
        let mut y_data = Vec::with_capacity(2 * sample_count);
        y_data.extend(x_data.iter().map(|&x| 1.7 * (-0.35 * x).exp() + 0.2));
        y_data.extend(x_data.iter().map(|&x| 0.8 * (1.2 * x).sin() - 0.1));
        let equations = vec![
            Expr::parse_expression("a*exp(b*x)+c"),
            Expr::parse_expression("d*sin(e*x)+f"),
        ];
        let unknowns = ["a", "b", "c", "d", "e", "f"].map(str::to_owned).to_vec();
        let initial_guess = vec![1.6, -0.32, 0.15, 0.75, 1.15, -0.08];

        for frontend in [
            SymbolicLambdifyFrontend::ExprLegacy,
            SymbolicLambdifyFrontend::AtomViewNative,
        ] {
            let mut sequential_results = Vec::new();
            let mut parallel_results = Vec::new();
            // Alternate route order to reduce systematic drift from always
            // measuring the same policy first in a long-running release test.
            for repetition in 0..7 {
                let policies = if repetition % 2 == 0 {
                    [
                        FittingObservationPolicy::Sequential,
                        FittingObservationPolicy::Parallel {
                            min_observations: 256,
                        },
                    ]
                } else {
                    [
                        FittingObservationPolicy::Parallel {
                            min_observations: 256,
                        },
                        FittingObservationPolicy::Sequential,
                    ]
                };

                for policy in policies {
                    let mut fit = Fitting::new()
                        .with_data(x_data.clone(), y_data.clone())
                        .with_equations(equations.clone())
                        .with_arg("x".into())
                        .with_unknowns(unknowns.clone())
                        .with_initial_guess(initial_guess.clone())
                        .with_symbolic_frontend(frontend)
                        .with_lambdify_execution_policy(LambdifyExecutionPolicy::Sequential)
                        .with_observation_policy(policy)
                        .with_telemetry(LeastSquaresTelemetryMode::Detailed);
                    fit.try_validate_and_infer().unwrap();

                    let preparation_started = Instant::now();
                    fit.try_prepare_least_squares().unwrap();
                    let preparation_ms = preparation_started.elapsed().as_secs_f64() * 1_000.0;
                    let solve_started = Instant::now();
                    fit.try_solve().unwrap();
                    let solve_ms = solve_started.elapsed().as_secs_f64() * 1_000.0;
                    let stats = fit.last_statistics().unwrap();
                    let solution = fit.result.as_ref().unwrap().as_slice().to_vec();
                    assert!(fit.r_squared().is_some_and(|r_squared| r_squared > 0.999));

                    println!(
                        "FIT_BATCH_TWO_RESPONSE repetition={repetition} samples={sample_count} equations=2 parameters=6 frontend={frontend:?} policy={policy:?} preparation_ms={preparation_ms:.3} solve_ms={solve_ms:.3} residual_ms={:.3} jacobian_ms={:.3} iterations={} residual_evaluations={} jacobian_evaluations={} factorizations={} linear_solves={}",
                        stats.residual_duration.as_secs_f64() * 1_000.0,
                        stats.jacobian_duration.as_secs_f64() * 1_000.0,
                        stats.iterations,
                        stats.residual_evaluations,
                        stats.jacobian_evaluations,
                        stats.linear_factorizations,
                        stats.linear_solves,
                    );
                    let row = (solution, stats.clone());
                    match policy {
                        FittingObservationPolicy::Auto => unreachable!("explicit A/B policies"),
                        FittingObservationPolicy::Sequential => sequential_results.push(row),
                        FittingObservationPolicy::Parallel { .. } => parallel_results.push(row),
                    }
                }
            }

            for ((sequential_solution, sequential_stats), (parallel_solution, parallel_stats)) in
                sequential_results.iter().zip(&parallel_results)
            {
                assert_eq!(sequential_solution, parallel_solution);
                assert_eq!(sequential_stats.iterations, parallel_stats.iterations);
                assert_eq!(
                    sequential_stats.residual_evaluations,
                    parallel_stats.residual_evaluations
                );
                assert_eq!(
                    sequential_stats.jacobian_evaluations,
                    parallel_stats.jacobian_evaluations
                );
                assert_eq!(
                    sequential_stats.linear_factorizations,
                    parallel_stats.linear_factorizations
                );
                assert_eq!(sequential_stats.linear_solves, parallel_stats.linear_solves);
            }
        }
    }

    #[test]
    fn linear_fitting_test() {
        let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_data = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let initial_guess = vec![1.0, 1.0];
        let unknown_coeffs = vec!["a".to_string(), "b".to_string()];
        let eq = "a * x + b".to_string();
        let mut sym_fitting = Fitting::new();
        sym_fitting.fitting_generate_from_str(
            x_data,
            y_data,
            eq,
            Some(unknown_coeffs),
            "x".to_string(),
            initial_guess,
            None,
            None,
            None,
            None,
            None,
        );
        sym_fitting.solve().unwrap();
        let map_of_solutions = sym_fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map_of_solutions["a"], 2.0, epsilon = 1e-6);
        assert_relative_eq!(map_of_solutions["b"], 0.0, epsilon = 1e-6);
    }
    #[test]
    fn quadratic_fitting_test() {
        let x_data = (0..100).map(|x| x as f64).collect::<Vec<f64>>();
        let quadratic_function = |x: f64| 5.0 * x * x + 2.0 * x + 100.0;
        let y_data = x_data
            .iter()
            .map(|&x| quadratic_function(x))
            .collect::<Vec<f64>>();
        let initial_guess = vec![1.0, 1.0, 1.0];
        let unknown_coeffs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let eq = "a * x^2.0 + b * x + c".to_string();
        let mut sym_fitting = Fitting::new();
        sym_fitting.fitting_generate_from_str(
            x_data,
            y_data,
            eq,
            Some(unknown_coeffs),
            "x".to_string(),
            initial_guess,
            None,
            None,
            None,
            None,
            None,
        );
        sym_fitting.solve().unwrap();
        let map_of_solutions = sym_fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map_of_solutions["a"], 5.0, epsilon = 1e-6);
        assert_relative_eq!(map_of_solutions["b"], 2.0, epsilon = 1e-6);
        assert_relative_eq!(map_of_solutions["c"], 100.0, epsilon = 1e-6);
    }
    #[test]
    fn exp_fitting_test() {
        let x_data = (0..20).map(|x| x as f64).collect::<Vec<f64>>();
        let exp_function = |x: f64| (1e-1 * x).exp() + 10.0;
        let y_data = x_data
            .iter()
            .map(|&x| exp_function(x))
            .collect::<Vec<f64>>();
        let initial_guess = vec![1.0, 1.0];
        let unknown_coeffs = vec!["a".to_string(), "b".to_string()];
        let eq = " exp(a*x) + b".to_string();
        let mut sym_fitting = Fitting::new();
        sym_fitting.fitting_generate_from_str(
            x_data,
            y_data,
            eq,
            Some(unknown_coeffs),
            "x".to_string(),
            initial_guess,
            None,
            None,
            None,
            None,
            None,
        );
        sym_fitting.solve().unwrap();
        let map_of_solutions = sym_fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map_of_solutions["a"], 1e-1, epsilon = 1e-6);
        assert_relative_eq!(map_of_solutions["b"], 10.0, epsilon = 1e-6);
    }
    #[test]
    fn easy_fitting_test() {
        let x_data = (0..100).map(|x| x as f64).collect::<Vec<f64>>();
        let quadratic_function = |x: f64| 5.0 * x * x + 2.0 * x + 100.0;
        let y_data = x_data
            .iter()
            .map(|&x| quadratic_function(x))
            .collect::<Vec<f64>>();
        let initial_guess = vec![1.0, 1.0, 1.0];
        let unknown_coeffs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let eq = "a * x^2.0 + b * x + c".to_string();
        let mut sym_fitting = Fitting::new();
        sym_fitting
            .easy_fitting(
                x_data,
                y_data,
                eq,
                Some(unknown_coeffs),
                "x".to_string(),
                initial_guess,
            )
            .unwrap();
        let map_of_solutions = sym_fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map_of_solutions["a"], 5.0, epsilon = 1e-6);
        assert_relative_eq!(map_of_solutions["b"], 2.0, epsilon = 1e-6);
        assert_relative_eq!(map_of_solutions["c"], 100.0, epsilon = 1e-6);
    }

    #[test]
    fn polinomial_fitting_test() {
        let x_data = (0..100).map(|x| x as f64).collect::<Vec<f64>>();
        let polynomial_function = |x: f64| 5.0 * x * x * x + 2.0 * x * x + 100.0 * x + 1000.0;
        let y_data = x_data
            .iter()
            .map(|&x| polynomial_function(x))
            .collect::<Vec<f64>>();
        let initial_guess = vec![1.0, 1.0, 1.0, 1.0];

        let mut sym_fitting = Fitting::new();
        sym_fitting.poly_fitting(
            x_data,
            y_data,
            3,
            "x".to_string(),
            initial_guess,
            None,
            None,
            None,
            None,
            None,
        );
        sym_fitting.solve().unwrap();
        let map_of_solutions = sym_fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map_of_solutions["c3"], 5.0, epsilon = 1e-6);
        assert_relative_eq!(map_of_solutions["c2"], 2.0, epsilon = 1e-6);
        assert_relative_eq!(map_of_solutions["c1"], 100.0, epsilon = 1e-6);
        assert_relative_eq!(map_of_solutions["c0"], 1000.0, epsilon = 1e-6);
    }
    #[test]
    fn test_linear_fit() {
        let x_data = (0..100).map(|x| x as f64).collect::<Vec<f64>>();
        let y_data = x_data.iter().map(|&x| 5.0 * x + 2.0).collect::<Vec<f64>>();
        let initial_guess = (1.0, 1.0);
        let mut sym_fitting = Fitting::new();
        sym_fitting.fit_linear(x_data, y_data, initial_guess);
        sym_fitting.solve().unwrap();
        let map_of_solutions = sym_fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map_of_solutions["c1"], 5.0, epsilon = 1e-6);
        assert_relative_eq!(map_of_solutions["c0"], 2.0, epsilon = 1e-6);
    }
    #[test]
    fn test_noisy_linear_fit() {
        let x_data = (0..1000).map(|x| x as f64).collect::<Vec<f64>>();
        // to y data add some noise from -0.05 to 0.05
        let y_data = x_data
            .iter()
            .map(|&x| 5.0 * x + 2.0 + rand::random_range(-0.1..0.1))
            .collect::<Vec<f64>>();
        println!("noisy y_data: {:?}", y_data);
        let initial_guess = (1.0, 1.0);
        let mut sym_fitting = Fitting::new();
        sym_fitting.fit_linear(x_data, y_data, initial_guess);
        sym_fitting.solve().unwrap();
        let map_of_solutions = sym_fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map_of_solutions["c1"], 5.0, epsilon = 1e-2);
        assert_relative_eq!(map_of_solutions["c0"], 2.0, epsilon = 1e-2);
    }

    #[test]
    fn test_noisy_linear_experimatal() {
        let x_data = vec![
            0.0 + 41.35,
            1.75 + 41.35,
            4.85 + 41.35,
            6.0 + 41.35,
            11.2 + 41.35,
        ];

        let y_data = vec![-0.69, -2.24, -5.47, -6.47, -11.86];
        println!("noisy y_data: {:?}", y_data);
        let initial_guess = (1.0, 1.0);
        let mut sym_fitting = Fitting::new();
        sym_fitting.fit_linear(x_data, y_data, initial_guess);
        sym_fitting.solve().unwrap();
        let map_of_solutions = sym_fitting.map_of_solutions.unwrap();
        println!("{:?}", map_of_solutions);
    }

    // ========== Builder Pattern Tests ==========

    #[test]
    fn test_builder_linear_string() {
        let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_data = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let fitting = Fitting::new()
            .with_data(x_data, y_data)
            .with_equation_str("a * x + b".to_string())
            .with_arg("x".to_string())
            .with_unknowns(vec!["a".to_string(), "b".to_string()])
            .with_initial_guess(vec![1.0, 1.0])
            .build();

        let map = fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map["a"], 2.0, epsilon = 1e-6);
        assert_relative_eq!(map["b"], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_builder_linear_native_expr() {
        let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_data = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        // Native symbolic construction
        let vars = Expr::Symbols("a, x, b");
        let a = vars[0].clone();
        let x = vars[1].clone();
        let b = vars[2].clone();
        let eq = a * x + b;

        let fitting = Fitting::new()
            .with_data(x_data, y_data)
            .with_equation(eq)
            .with_arg("x".to_string())
            .with_unknowns(vec!["a".to_string(), "b".to_string()])
            .with_initial_guess(vec![1.0, 1.0])
            .build();

        let map = fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map["a"], 2.0, epsilon = 1e-6);
        assert_relative_eq!(map["b"], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_builder_vector_equations_native_expr() {
        let x_data = vec![1.0, 2.0, 3.0, 4.0];
        let mut y_data = x_data.iter().map(|&x| 2.0 * x + 1.0).collect::<Vec<f64>>();
        y_data.extend(x_data.iter().map(|&x| 3.0 * x * x + 4.0));

        let vars = Expr::Symbols("a, b, c, d, x");
        let a = vars[0].clone();
        let b = vars[1].clone();
        let c = vars[2].clone();
        let d = vars[3].clone();
        let x = vars[4].clone();

        let eq_system = vec![
            a.clone() * x.clone() + b.clone(),
            c * x.clone().pow(Expr::Const(2.0)) + d,
        ];

        let fitting = Fitting::new()
            .with_data(x_data, y_data)
            .with_equations(eq_system)
            .with_arg("x".to_string())
            .with_unknowns(vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ])
            .with_initial_guess(vec![1.0, 1.0, 1.0, 1.0])
            .build();

        let map = fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map["a"], 2.0, epsilon = 1e-6);
        assert_relative_eq!(map["b"], 1.0, epsilon = 1e-6);
        assert_relative_eq!(map["c"], 3.0, epsilon = 1e-6);
        assert_relative_eq!(map["d"], 4.0, epsilon = 1e-6);
    }

    #[test]
    fn test_vector_equations_imperative_api() {
        let x_data = vec![1.0, 2.0, 3.0];
        let mut y_data = x_data.iter().map(|&x| 1.5 * x + 0.5).collect::<Vec<f64>>();
        y_data.extend(x_data.iter().map(|&x| 2.0 * x * x + 3.0));

        let vars = Expr::Symbols("a, b, c, d, x");
        let a = vars[0].clone();
        let b = vars[1].clone();
        let c = vars[2].clone();
        let d = vars[3].clone();
        let x = vars[4].clone();

        let eq_system = vec![
            a.clone() * x.clone() + b.clone(),
            c * x.clone().pow(Expr::Const(2.0)) + d,
        ];

        let mut fitting = Fitting::new();
        fitting.fitting_generate_from_vec(
            x_data,
            y_data,
            eq_system,
            Some(vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ]),
            "x".to_string(),
            vec![1.0, 1.0, 1.0, 1.0],
            None,
            None,
            None,
            None,
            None,
        );
        fitting.solve().unwrap();

        let map = fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map["a"], 1.5, epsilon = 1e-6);
        assert_relative_eq!(map["b"], 0.5, epsilon = 1e-6);
        assert_relative_eq!(map["c"], 2.0, epsilon = 1e-6);
        assert_relative_eq!(map["d"], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_builder_quadratic_native_expr() {
        let x_data = (0..50).map(|x| x as f64).collect::<Vec<f64>>();
        let y_data = x_data
            .iter()
            .map(|&x| 3.0 * x * x + 2.0 * x + 1.0)
            .collect::<Vec<f64>>();

        // Native symbolic construction
        let vars = Expr::Symbols("a, b, c, x");
        let a = vars[0].clone();
        let b = vars[1].clone();
        let c = vars[2].clone();
        let x = vars[3].clone();

        let eq = a * x.clone().pow(Expr::Const(2.0)) + b * x + c;

        let fitting = Fitting::new()
            .with_data(x_data, y_data)
            .with_equation(eq)
            .with_arg("x".to_string())
            .with_unknowns(vec!["a".to_string(), "b".to_string(), "c".to_string()])
            .with_initial_guess(vec![1.0, 1.0, 1.0])
            .build();

        let map = fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map["a"], 3.0, epsilon = 1e-6);
        assert_relative_eq!(map["b"], 2.0, epsilon = 1e-6);
        assert_relative_eq!(map["c"], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_builder_exponential_native_expr() {
        let x_data = (0..20).map(|x| x as f64).collect::<Vec<f64>>();
        let y_data = x_data
            .iter()
            .map(|&x| (0.1 * x).exp() + 5.0)
            .collect::<Vec<f64>>();

        // Native symbolic construction
        let vars = Expr::Symbols("a, x, b");
        let a = vars[0].clone();
        let x = vars[1].clone();
        let b = vars[2].clone();

        let eq = Expr::exp(a * x) + b;

        let fitting = Fitting::new()
            .with_data(x_data, y_data)
            .with_equation(eq)
            .with_arg("x".to_string())
            .with_unknowns(vec!["a".to_string(), "b".to_string()])
            .with_initial_guess(vec![0.05, 1.0])
            .with_tolerance(1e-8)
            .build();

        let map = fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map["a"], 0.1, epsilon = 1e-6);
        assert_relative_eq!(map["b"], 5.0, epsilon = 1e-6);
    }

    #[test]
    fn test_builder_logarithmic_native_expr() {
        let x_data = (1..30).map(|x| x as f64).collect::<Vec<f64>>();
        let y_data = x_data
            .iter()
            .map(|&x| 2.0 * x.ln() + 3.0)
            .collect::<Vec<f64>>();

        // Native symbolic construction
        let vars = Expr::Symbols("a, x, b");
        let a = vars[0].clone();
        let x = vars[1].clone();
        let b = vars[2].clone();

        let eq = a * Expr::ln(x) + b;

        let fitting = Fitting::new()
            .with_data(x_data, y_data)
            .with_equation(eq)
            .with_arg("x".to_string())
            .with_unknowns(vec!["a".to_string(), "b".to_string()])
            .with_initial_guess(vec![1.0, 1.0])
            .build();

        let map = fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map["a"], 2.0, epsilon = 1e-6);
        assert_relative_eq!(map["b"], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_builder_trigonometric_native_expr() {
        use std::f64::consts::PI;
        let x_data = (0..100).map(|x| x as f64 * PI / 50.0).collect::<Vec<f64>>();
        let y_data = x_data
            .iter()
            .map(|&x| 2.0 * x.sin() + 1.0)
            .collect::<Vec<f64>>();

        // Native symbolic construction
        let vars = Expr::Symbols("a, x, b");
        let a = vars[0].clone();
        let x = vars[1].clone();
        let b = vars[2].clone();
        let eq = a * Expr::sin(Box::new(x)) + b;

        let fitting = Fitting::new()
            .with_data(x_data, y_data)
            .with_equation(eq)
            .with_arg("x".to_string())
            .with_unknowns(vec!["a".to_string(), "b".to_string()])
            .with_initial_guess(vec![1.0, 0.5])
            .with_tolerance(1e-8)
            .with_g_tolerance(1e-8)
            .with_f_tolerance(1e-8)
            .with_max_iterations(100)
            .build();

        let map = fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map["a"], 2.0, epsilon = 1e-6);
        assert_relative_eq!(map["b"], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_builder_polynomial() {
        let x_data = (0..50).map(|x| x as f64).collect::<Vec<f64>>();
        let y_data = x_data
            .iter()
            .map(|&x| 2.0 * x.powi(3) + 3.0 * x.powi(2) + 4.0 * x + 5.0)
            .collect::<Vec<f64>>();

        let fitting = Fitting::new()
            .with_data(x_data, y_data)
            .with_polynomial(3, "x".to_string())
            .with_initial_guess(vec![1.0, 1.0, 1.0, 1.0])
            .build();

        let map = fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map["c3"], 2.0, epsilon = 1e-6);
        assert_relative_eq!(map["c2"], 3.0, epsilon = 1e-6);
        assert_relative_eq!(map["c1"], 4.0, epsilon = 1e-6);
        assert_relative_eq!(map["c0"], 5.0, epsilon = 1e-6);
    }

    #[test]
    fn test_builder_complex_native_expr() {
        let x_data = (1..30).map(|x| x as f64 * 0.1).collect::<Vec<f64>>();
        let y_data = x_data
            .iter()
            .map(|&x| 2.0 * x.ln() + 0.5 * (0.3 * x).exp() + 1.0)
            .collect::<Vec<f64>>();

        // Native symbolic construction with multiple operations
        let vars = Expr::Symbols("a, b, c, d, x");
        let a = vars[0].clone();
        let b = vars[1].clone();
        let c = vars[2].clone();
        let d = vars[3].clone();
        let x = vars[4].clone();

        let eq = a * Expr::ln(x.clone()) + b * Expr::exp(c * x) + d;

        let fitting = Fitting::new()
            .with_data(x_data, y_data)
            .with_equation(eq)
            .with_arg("x".to_string())
            .with_unknowns(vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ])
            .with_initial_guess(vec![1.5, 0.3, 0.2, 0.5])
            .with_tolerance(1e-6)
            .with_max_iterations(300)
            .build();

        let map = fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map["a"], 2.0, epsilon = 1e-3);
        assert_relative_eq!(map["b"], 0.5, epsilon = 1e-3);
        assert_relative_eq!(map["c"], 0.3, epsilon = 1e-3);
        assert_relative_eq!(map["d"], 1.0, epsilon = 1e-3);
    }

    #[test]
    fn test_builder_power_law_native_expr() {
        // Use smaller range and simpler power for more stable fitting
        let x_data = (1..50).map(|x| x as f64).collect::<Vec<f64>>();
        let y_data = x_data
            .iter()
            .map(|&x| 2.0 * x.powf(0.5) + 1.0)
            .collect::<Vec<f64>>();

        // Native symbolic construction: a * x^b + c
        let vars = Expr::Symbols("a, b, c, x");
        let a = vars[0].clone();
        let b = vars[1].clone();
        let c = vars[2].clone();
        let x = vars[3].clone();

        let eq = a * x.pow(b) + c;

        let fitting = Fitting::new()
            .with_data(x_data, y_data)
            .with_equation(eq)
            .with_arg("x".to_string())
            .with_unknowns(vec!["a".to_string(), "b".to_string(), "c".to_string()])
            .with_initial_guess(vec![1.5, 0.4, 0.5])
            .with_tolerance(1e-6)
            .with_max_iterations(300)
            .build();

        let map = fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map["a"], 2.0, epsilon = 1e-4);
        assert_relative_eq!(map["b"], 0.5, epsilon = 1e-4);
        assert_relative_eq!(map["c"], 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_builder_power_law() {
        // Use smaller range and simpler power for more stable fitting
        let x_data = (1..50).map(|x| x as f64).collect::<Vec<f64>>();
        let y_data = x_data
            .iter()
            .map(|&x| 2.0 * x.powf(0.5) + 1.0)
            .collect::<Vec<f64>>();

        let fitting = Fitting::new()
            .with_data(x_data, y_data)
            .with_equation_str("a*x^b + c".to_string())
            .with_arg("x".to_string())
            .with_unknowns(vec!["a".to_string(), "b".to_string(), "c".to_string()])
            .with_initial_guess(vec![1.5, 0.4, 0.5])
            .with_tolerance(1e-6)
            .with_max_iterations(3000)
            .build();

        let map = fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map["a"], 2.0, epsilon = 1e-4);
        assert_relative_eq!(map["b"], 0.5, epsilon = 1e-4);
        assert_relative_eq!(map["c"], 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_builder_rational_function_native_expr() {
        let x_data = (1..30).map(|x| x as f64).collect::<Vec<f64>>();
        let y_data = x_data
            .iter()
            .map(|&x| (2.0 * x + 3.0) / (x + 1.0))
            .collect::<Vec<f64>>();

        // Native symbolic construction: (a*x + b) / (x + c)
        let vars = Expr::Symbols("a, b, c, x");
        let a = vars[0].clone();
        let b = vars[1].clone();
        let c = vars[2].clone();
        let x = vars[3].clone();

        let eq = (a * x.clone() + b) / (x + c);

        let fitting = Fitting::new()
            .with_data(x_data, y_data)
            .with_equation(eq)
            .with_arg("x".to_string())
            .with_unknowns(vec!["a".to_string(), "b".to_string(), "c".to_string()])
            .with_initial_guess(vec![1.5, 2.0, 0.5])
            .with_tolerance(1e-8)
            .build();

        let map = fitting.map_of_solutions.unwrap();
        assert_relative_eq!(map["a"], 2.0, epsilon = 1e-5);
        assert_relative_eq!(map["b"], 3.0, epsilon = 1e-5);
        assert_relative_eq!(map["c"], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn symbolic_input_changes_invalidate_prepared_frontend() {
        let mut fitting = Fitting::new()
            .with_data(vec![0.0, 1.0], vec![1.0, 2.0])
            .with_equation_str("a*x + b".to_string())
            .with_arg("x".to_string())
            .with_unknowns(vec!["a".to_string(), "b".to_string()])
            .with_initial_guess(vec![1.0, 1.0]);

        fitting.solve().unwrap();
        assert!(fitting.prepared_least_squares.is_some());

        let fitting = fitting.with_equation_str("a*x^2 + b".to_string());
        assert!(fitting.prepared_least_squares.is_none());
    }

    #[test]
    fn new_observations_reuse_symbolic_preparation() {
        let mut fitting = Fitting::new()
            .with_data(vec![0.0, 1.0], vec![1.0, 3.0])
            .with_equation_str("a*x + b".to_string())
            .with_arg("x".to_string())
            .with_unknowns(vec!["a".to_string(), "b".to_string()])
            .with_initial_guess(vec![0.0, 0.0]);
        fitting.try_validate_and_infer().unwrap();
        fitting.try_prepare_least_squares().unwrap();

        let mut fitting = fitting.with_data(vec![2.0, 3.0], vec![5.0, 7.0]);
        assert!(fitting.prepared_least_squares.is_some());
        fitting.try_solve().unwrap();
        let solution = fitting.solution_map().unwrap();
        assert_relative_eq!(solution["a"], 2.0, epsilon = 1e-8);
        assert_relative_eq!(solution["b"], 1.0, epsilon = 1e-8);
    }

    #[test]
    fn failed_repeat_solve_clears_previous_solution() {
        let mut fitting = Fitting::new()
            .with_data(vec![0.0, 1.0], vec![1.0, 2.0])
            .with_equation_str("a*x + b".to_string())
            .with_arg("x".to_string())
            .with_unknowns(vec!["a".to_string(), "b".to_string()])
            .with_initial_guess(vec![1.0, 1.0])
            .try_build()
            .unwrap();
        assert!(fitting.result.is_some());
        assert!(fitting.map_of_solutions.is_some());

        fitting.y_data.pop();
        let error = fitting.try_solve().unwrap_err();
        assert!(matches!(error, FittingError::DimensionMismatch { .. }));
        assert!(fitting.result.is_none());
        assert!(fitting.map_of_solutions.is_none());
        assert!(fitting.r_squared().is_none());
    }

    #[test]
    fn max_iterations_is_an_exact_iteration_cap_and_keeps_telemetry() {
        let mut fitting = Fitting::new()
            .with_data(
                (0..30).map(|i| i as f64 * 0.1).collect(),
                (0..30)
                    .map(|i| (0.7 * i as f64 * 0.1).exp() + 2.0)
                    .collect(),
            )
            .with_equation_str("exp(a*x) + b".to_string())
            .with_arg("x".to_string())
            .with_unknowns(vec!["a".to_string(), "b".to_string()])
            .with_initial_guess(vec![-1.0, 0.0])
            .with_max_iterations(1)
            .with_telemetry(LeastSquaresTelemetryMode::Counters);

        let error = match fitting.try_solve() {
            Err(error) => error,
            Ok(_) => panic!("one iteration should not converge for this initial guess"),
        };
        assert!(matches!(
            error,
            FittingError::DidNotConverge {
                termination: LeastSquaresTerminationReason::MaxIterationsReached { limit: 1 },
                ..
            }
        ));
        let stats = fitting.last_statistics().expect("counters were requested");
        assert!(stats.availability.is_collected());
        assert!(!stats.timings_collected);
        assert_eq!(stats.iterations, 1);
    }

    #[test]
    fn fitting_exposes_opt_in_least_squares_telemetry() {
        let fitting = Fitting::new()
            .with_data(vec![0.0, 1.0, 2.0], vec![1.0, 3.0, 5.0])
            .with_equation_str("a*x + b".to_string())
            .with_arg("x".to_string())
            .with_unknowns(vec!["a".to_string(), "b".to_string()])
            .with_initial_guess(vec![0.0, 0.0])
            .with_scale_diag(false)
            .with_telemetry(LeastSquaresTelemetryMode::Detailed)
            .try_build()
            .unwrap();

        let stats = fitting.last_statistics().expect("telemetry was requested");
        assert!(stats.availability.is_collected());
        assert!(stats.timings_collected);
        assert!(stats.jacobian_evaluations > 0);
        assert!(stats.residual_evaluations > 0);
        assert_relative_eq!(fitting.solution_map().unwrap()["a"], 2.0, epsilon = 1e-8);
    }

    #[test]
    fn malformed_equation_has_a_typed_parse_error() {
        let result = Fitting::new().try_with_equation_str("(");
        assert!(matches!(result, Err(FittingError::EquationParse(_))));
    }
}
