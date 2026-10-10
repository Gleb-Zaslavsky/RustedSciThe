//! High-level symbolic API for rectangular least-squares problems.
//!
//! Symbolic preparation is delegated to [`PreparedSymbolicLeastSquaresProblem`],
//! while all numerical iterations go through the canonical least-squares solver
//! selected by [`crate::numerical::Nonlinear_systems::solver::NonlinearSolver`].

use crate::numerical::Nonlinear_systems::engine::SolveStatistics;
use crate::numerical::Nonlinear_systems::least_squares::{
    LeastSquaresError, LeastSquaresProblem, LeastSquaresTelemetryMode, LevenbergMarquardt,
    PreparedSymbolicLeastSquaresProblem,
};
use crate::numerical::Nonlinear_systems::solver::NonlinearSolver;
use crate::numerical::Nonlinear_systems::symbolic::SymbolicProblemOptions;
use crate::symbolic::symbolic_engine::Expr;
use log::{log, Level};
use nalgebra::DVector;
use std::collections::HashMap;

/// Symbolic wrapper for Levenberg-Marquardt algorithm.
/// Solves nonlinear least squares problems using symbolic expressions with analytical Jacobians.
pub struct SymbolicLeastSquaresSolver {
    /// Vector of symbolic equations to solve
    pub eq_system: Vec<Expr>,
    /// Variable names in the equations
    pub values: Vec<String>,
    /// Optional parameter names for parametric equations
    pub parameters: Option<Vec<String>>,
    /// Initial guess for the solution
    pub initial_guess: Vec<f64>,
    /// Maximum number of iterations
    pub max_iterations: Option<usize>,
    /// Convergence tolerance for parameters
    pub tolerance: Option<f64>,
    /// Function value tolerance
    pub f_tolerance: Option<f64>,
    /// Gradient tolerance
    pub g_tolerance: Option<f64>,
    /// Whether to scale diagonal elements
    pub scale_diag: Option<bool>,
    /// Solution vector
    pub result: Option<DVector<f64>>,
    /// Solution mapped to variable names
    pub map_of_solutions: Option<HashMap<String, f64>>,
    /// Statistics from the most recent mutable solve.
    pub last_statistics: Option<SolveStatistics>,
    /// Optional log level (debug, info, warn, error, off, none); `None` is silent.
    /// This library wrapper never installs or changes a process-global logger.
    pub loglevel: Option<String>,
    /// Symbolic variables whose numeric domain is strictly positive.
    ///
    /// This is required for expressions such as `ln(N0 / Np)`: the
    /// trust-region controller can reject an invalid trial before it reaches
    /// the symbolic callback.
    positive_variables: Vec<String>,
    /// Prepared shared symbolic frontend used by all solves.
    prepared: Option<PreparedSymbolicLeastSquaresProblem>,
    telemetry_mode: LeastSquaresTelemetryMode,
}

impl SymbolicLeastSquaresSolver {
    /// Creates a new LM solver instance with default settings.
    pub fn new() -> Self {
        SymbolicLeastSquaresSolver {
            eq_system: Vec::new(),
            values: Vec::new(),
            parameters: None,
            initial_guess: Vec::new(),
            tolerance: None,
            f_tolerance: None,
            g_tolerance: None,
            scale_diag: None,
            max_iterations: None,
            result: None,
            map_of_solutions: None,
            last_statistics: None,
            loglevel: None,
            positive_variables: Vec::new(),
            prepared: None,
            telemetry_mode: LeastSquaresTelemetryMode::Off,
        }
    }

    /// Builder pattern: Set equations from Expr vector
    pub fn with_equations(mut self, eq_system: Vec<Expr>) -> Self {
        self.prepared = None;
        self.eq_system = eq_system;
        self
    }

    /// Builder pattern: Set equations from string vector
    pub fn with_equations_str(mut self, eq_system_string: Vec<String>) -> Self {
        self.prepared = None;
        self.eq_system = eq_system_string
            .iter()
            .map(|x| Expr::parse_expression(x))
            .collect();
        self
    }

    /// Fallible string-equation builder for callers that want parse errors
    /// returned through the typed least-squares error channel.
    pub fn try_with_equations_str(
        mut self,
        equations: Vec<String>,
    ) -> Result<Self, LeastSquaresError> {
        self.prepared = None;
        self.eq_system = equations
            .into_iter()
            .enumerate()
            .map(|(index, equation)| {
                Expr::try_parse_expression(&equation).map_err(|error| {
                    LeastSquaresError::Problem(
                        crate::numerical::Nonlinear_systems::error::SolveError::InvalidConfig(
                            format!("failed to parse least-squares equation {index}: {error}"),
                        ),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self)
    }

    /// Builder pattern: Set unknowns
    pub fn with_unknowns(mut self, unknowns: Vec<String>) -> Self {
        self.prepared = None;
        self.values = unknowns;
        self
    }

    /// Builder pattern: Set parameters
    pub fn with_parameters(mut self, parameters: Vec<String>) -> Self {
        self.prepared = None;
        self.parameters = Some(parameters);
        self
    }

    /// Builder pattern: Set initial guess
    pub fn with_initial_guess(mut self, initial_guess: Vec<f64>) -> Self {
        self.initial_guess = initial_guess;
        self
    }

    /// Builder pattern: Set tolerance
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = Some(tolerance);
        self
    }

    /// Builder pattern: Set function tolerance
    pub fn with_f_tolerance(mut self, f_tolerance: f64) -> Self {
        self.f_tolerance = Some(f_tolerance);
        self
    }

    /// Builder pattern: Set gradient tolerance
    pub fn with_g_tolerance(mut self, g_tolerance: f64) -> Self {
        self.g_tolerance = Some(g_tolerance);
        self
    }

    /// Builder pattern: Set scale diagonal
    pub fn with_scale_diag(mut self, scale_diag: bool) -> Self {
        self.scale_diag = Some(scale_diag);
        self
    }

    /// Builder pattern: Set max iterations
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = Some(max_iterations);
        self
    }

    /// Builder pattern: Set log level
    pub fn with_loglevel(mut self, loglevel: String) -> Self {
        self.loglevel = Some(loglevel);
        self
    }

    /// Enables optional LM telemetry for solves through this symbolic wrapper.
    pub fn with_telemetry(mut self, mode: LeastSquaresTelemetryMode) -> Self {
        self.telemetry_mode = mode;
        self
    }

    /// Fallible logging-level builder. `off` and `none` disable solver logs.
    pub fn try_with_loglevel(mut self, loglevel: &str) -> Result<Self, LeastSquaresError> {
        parse_log_level(loglevel)?;
        self.loglevel = Some(loglevel.to_ascii_lowercase());
        Ok(self)
    }

    /// Builder pattern: Build and prepare solver (generates Jacobian)
    pub fn build(mut self) -> Self {
        self.validate_and_infer();
        self.prepare_symbolic()
            .expect("symbolic least-squares preparation should succeed");
        self
    }

    /// Validates, prepares, and returns a symbolic solver without panic-based
    /// input or preparation failures.
    pub fn try_build(mut self) -> Result<Self, LeastSquaresError> {
        self.try_validate_and_infer()?;
        self.prepare_symbolic()?;
        Ok(self)
    }

    fn try_validate_and_infer(&mut self) -> Result<(), LeastSquaresError> {
        if self.eq_system.is_empty() {
            return Err(LeastSquaresError::EmptyProblem {
                stage:
                    crate::numerical::Nonlinear_systems::least_squares::LeastSquaresStage::Residual,
            });
        }
        if self.values.is_empty() {
            let mut variables: Vec<String> = self
                .eq_system
                .iter()
                .flat_map(Expr::all_arguments_are_variables)
                .collect();
            variables.sort();
            variables.dedup();
            self.values = variables;
        }
        if self.values.is_empty() {
            return Err(LeastSquaresError::EmptyProblem {
                stage: crate::numerical::Nonlinear_systems::least_squares::LeastSquaresStage::Parameters,
            });
        }
        if self.initial_guess.len() != self.values.len() {
            return Err(LeastSquaresError::DimensionMismatch {
                stage: crate::numerical::Nonlinear_systems::least_squares::LeastSquaresStage::Parameters,
                expected: self.values.len(),
                actual: self.initial_guess.len(),
            });
        }
        if let Some((index, _)) = self
            .initial_guess
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(LeastSquaresError::NonFiniteValue {
                stage: crate::numerical::Nonlinear_systems::least_squares::LeastSquaresStage::Parameters,
                index,
            });
        }
        Ok(())
    }

    /// Validate inputs and infer unknowns if not provided
    fn validate_and_infer(&mut self) {
        assert!(
            !self.eq_system.is_empty(),
            "Equation system cannot be empty."
        );
        assert!(
            !self.initial_guess.is_empty(),
            "Initial guess cannot be empty."
        );

        if self.values.is_empty() {
            let mut args: Vec<String> = self
                .eq_system
                .iter()
                .flat_map(|x| x.all_arguments_are_variables())
                .collect();
            args.sort();
            args.dedup();
            assert!(!args.is_empty(), "No variables found in equations.");
            self.values = args;
        }

        assert_eq!(
            self.values.len(),
            self.eq_system.len(),
            "Number of unknowns must equal number of equations."
        );
        assert_eq!(
            self.values.len(),
            self.initial_guess.len(),
            "Initial guess length must match number of unknowns."
        );
    }

    /// Prepares the shared symbolic residual/Jacobian frontend once.
    ///
    /// The wrapper deliberately keeps the historical infallible builder
    /// surface, so typed preparation errors are reported at this compatibility
    /// boundary with the same explicit context as the old API.
    fn prepare_symbolic(
        &mut self,
    ) -> Result<(), crate::numerical::Nonlinear_systems::error::SolveError> {
        let mut options = SymbolicProblemOptions::new()
            .with_variables(self.values.clone())
            .with_lambdify_backend();
        if let Some(parameters) = &self.parameters {
            options = options.with_equation_parameters(parameters.clone());
        }
        let mut prepared =
            PreparedSymbolicLeastSquaresProblem::from_expressions(self.eq_system.clone(), options)?;
        if !self.positive_variables.is_empty() {
            prepared.set_positive(&self.positive_variables)?;
        }
        self.prepared = Some(prepared);
        Ok(())
    }

    fn ensure_prepared(&mut self) {
        self.validate_and_infer();
        if self.prepared.is_none() {
            self.prepare_symbolic()
                .expect("symbolic least-squares preparation should succeed");
        }
    }

    pub fn set_loglevel(&mut self, loglevel: String) {
        self.loglevel = Some(loglevel);
    }

    /// Fallible setter for opt-in logging; invalid levels do not panic.
    pub fn try_set_loglevel(&mut self, loglevel: &str) -> Result<(), LeastSquaresError> {
        parse_log_level(loglevel)?;
        self.loglevel = Some(loglevel.to_ascii_lowercase());
        Ok(())
    }

    fn log_level(&self) -> Option<Level> {
        self.loglevel
            .as_deref()
            .and_then(|value| parse_log_level(value).ok().flatten())
    }

    /// Declares symbolic variables that must remain strictly positive.
    ///
    /// The declaration is retained when the equation system is rebuilt and
    /// is applied to an already prepared frontend immediately. This keeps
    /// domain handling explicit while avoiding any legacy callback generator.
    pub fn set_positive_variables<S: AsRef<str>>(
        &mut self,
        names: &[S],
    ) -> Result<(), crate::numerical::Nonlinear_systems::error::SolveError> {
        let names = names
            .iter()
            .map(|name| name.as_ref().to_string())
            .collect::<Vec<_>>();
        if let Some(prepared) = &mut self.prepared {
            prepared.set_positive(&names)?;
        }
        self.positive_variables = names;
        Ok(())
    }
    /// Sets up the equation system with unknowns, parameters, and solver options.
    pub fn set_equation_system(
        &mut self,
        eq_system: Vec<Expr>,
        unknowns: Option<Vec<String>>,
        parameters: Option<Vec<String>>,
        initial_guess: Vec<f64>,
        tolerance: Option<f64>,
        f_tolerance: Option<f64>,
        g_tolerance: Option<f64>,
        scale_diag: Option<bool>,
        max_iterations: Option<usize>,
    ) {
        self.prepared = None;
        self.eq_system = eq_system.clone();
        self.initial_guess = initial_guess;
        self.tolerance = tolerance;
        self.g_tolerance = g_tolerance;
        self.max_iterations = max_iterations;
        self.f_tolerance = f_tolerance;
        self.scale_diag = scale_diag;
        self.parameters = parameters;
        let values = if let Some(values) = unknowns {
            values
        } else {
            let mut args: Vec<String> = eq_system
                .iter()
                .map(|x| x.all_arguments_are_variables())
                .flatten()
                .collect::<Vec<String>>();
            args.sort();
            args.dedup();

            assert!(!args.is_empty(), "No variables found in the equations.");
            assert_eq!(
                args.len() == eq_system.len(),
                true,
                "Equation system and vector of variables should have the same length."
            );

            args
        };
        self.values = values.clone();
        assert!(
            !self.initial_guess.is_empty(),
            "Initial guess should not be empty."
        );
        if let Some(tolerance) = tolerance {
            assert!(
                tolerance >= 0.0,
                "Tolerance should be a non-negative number."
            );
        }
        if let Some(max_iterations) = max_iterations {
            assert!(
                max_iterations > 0,
                "Max iterations should be a positive number."
            );
        }
        if let Some(g_tolerance) = g_tolerance {
            assert!(
                g_tolerance >= 0.0,
                "Gradient tolerance should be a non-negative number."
            );
        }
        if let Some(f_tolerance) = f_tolerance {
            assert!(
                f_tolerance >= 0.0,
                "Function tolerance should be a non-negative number."
            );
        }
    }

    /// Parses string equations and sets up the system.
    pub fn eq_generate_from_str(
        &mut self,
        eq_system_string: Vec<String>,
        unknowns: Option<Vec<String>>,
        parameters: Option<Vec<String>>,
        initial_guess: Vec<f64>,
        tolerance: Option<f64>, // tolerance: f64
        f_tolerance: Option<f64>,
        g_tolerance: Option<f64>,
        scale_diag: Option<bool>,
        max_iterations: Option<usize>, // max_iterations: usize,
    ) {
        let eq_system = eq_system_string
            .iter()
            .map(|x| Expr::parse_expression(x))
            .collect::<Vec<Expr>>();
        self.set_equation_system(
            eq_system,
            unknowns,
            parameters,
            initial_guess,
            tolerance,
            f_tolerance,
            g_tolerance,
            scale_diag,
            max_iterations,
        );
        self.prepare_symbolic()
            .expect("symbolic least-squares preparation should succeed");
    }

    fn configured_solver(&self) -> LevenbergMarquardt {
        let mut solver = LevenbergMarquardt::new();
        if let Some(max_iterations) = self.max_iterations {
            solver = solver.with_patience(max_iterations);
        }
        if let Some(tolerance) = self.tolerance {
            solver = solver.with_xtol(tolerance);
        }
        if let Some(g_tolerance) = self.g_tolerance {
            solver = solver.with_gtol(g_tolerance);
        }
        if let Some(f_tolerance) = self.f_tolerance {
            solver = solver.with_ftol(f_tolerance);
        }
        solver = solver.with_telemetry(self.telemetry_mode);
        solver
    }

    fn try_configured_solver(&self) -> Result<LevenbergMarquardt, LeastSquaresError> {
        let mut solver = LevenbergMarquardt::new();
        if let Some(max_iterations) = self.max_iterations {
            solver = solver.try_with_patience(max_iterations)?;
        }
        if let Some(tolerance) = self.tolerance {
            solver = solver.try_with_xtol(tolerance)?;
        }
        if let Some(g_tolerance) = self.g_tolerance {
            solver = solver.try_with_gtol(g_tolerance)?;
        }
        if let Some(f_tolerance) = self.f_tolerance {
            solver = solver.try_with_ftol(f_tolerance)?;
        }
        Ok(solver.with_telemetry(self.telemetry_mode))
    }

    fn try_ensure_prepared(&mut self) -> Result<(), LeastSquaresError> {
        self.try_validate_and_infer()?;
        if self.prepared.is_none() {
            self.prepare_symbolic()?;
        }
        Ok(())
    }

    /// Solves through a typed, panic-free path and stores the latest result and
    /// telemetry snapshot on this wrapper.
    pub fn try_solve(
        &mut self,
    ) -> Result<
        crate::numerical::Nonlinear_systems::least_squares::MinimizationReport,
        LeastSquaresError,
    > {
        self.result = None;
        self.map_of_solutions = None;
        self.last_statistics = None;
        self.try_ensure_prepared()?;
        let prepared = self.prepared.as_ref().ok_or_else(|| {
            LeastSquaresError::InvalidProblemShape {
                stage: crate::numerical::Nonlinear_systems::least_squares::LeastSquaresStage::Configuration,
            }
        })?;
        let problem =
            prepared.bind_initial_with_guess(DVector::from_vec(self.initial_guess.clone()))?;
        let solver = self.try_configured_solver()?;
        let (problem, report) = solver.try_minimize(problem)?;
        self.last_statistics = Some(report.statistics.clone());
        if report.termination.was_successful() {
            let solution = problem.params();
            self.result = Some(solution.clone());
            self.map_of_solutions = Some(
                self.values
                    .iter()
                    .cloned()
                    .zip(solution.iter().copied())
                    .collect(),
            );
            if let Some(level) = self.log_level() {
                log!(level, "Least-squares termination: {:?}", report.termination);
                log!(
                    level,
                    "Least-squares objective: {}",
                    report.objective_function
                );
            }
        }
        Ok(report)
    }

    /// Typed parameter-rebind solve. Symbolic preparation is reused, and the
    /// returned report contains this solve's optional telemetry snapshot.
    pub fn try_solve_with_params(
        &mut self,
        equation_parameters: Vec<f64>,
    ) -> Result<
        crate::numerical::Nonlinear_systems::least_squares::MinimizationReport,
        LeastSquaresError,
    > {
        self.result = None;
        self.map_of_solutions = None;
        self.last_statistics = None;
        self.try_ensure_prepared()?;
        let prepared = self.prepared.as_ref().ok_or_else(|| {
            LeastSquaresError::InvalidProblemShape {
                stage: crate::numerical::Nonlinear_systems::least_squares::LeastSquaresStage::Configuration,
            }
        })?;
        let problem = prepared.bind_values_with_guess(
            DVector::from_vec(equation_parameters),
            DVector::from_vec(self.initial_guess.clone()),
        )?;
        let solver = self.try_configured_solver()?;
        let (problem, report) = solver.try_minimize(problem)?;
        self.last_statistics = Some(report.statistics.clone());
        if report.termination.was_successful() {
            let solution = problem.params();
            self.result = Some(solution.clone());
            self.map_of_solutions = Some(
                self.values
                    .iter()
                    .cloned()
                    .zip(solution.iter().copied())
                    .collect(),
            );
        }
        Ok(report)
    }

    /// Solves the nonlinear system with optional logging.
    pub fn solve(&mut self) {
        self.ensure_prepared();
        self.solve_internal();
    }

    /// Internal solver implementation without logging setup.
    fn solve_internal(&mut self) {
        let (solution, report) = {
            let prepared = self
                .prepared
                .as_ref()
                .expect("symbolic least-squares problem must be prepared");
            let problem = prepared
                .bind_initial_with_guess(DVector::from_vec(self.initial_guess.clone()))
                .expect("symbolic initial guess should bind");
            let solver = NonlinearSolver::LeastSquares(self.configured_solver());
            let (result, report) = solver
                .minimize_least_squares(problem)
                .expect("least-squares selector should accept its prepared problem");
            (result.params(), report)
        };
        self.last_statistics = Some(report.statistics.clone());
        if let Some(level) = self.log_level() {
            log!(level, "Least-squares termination: {:?}", report.termination);
            log!(
                level,
                "Least-squares evaluations: {}",
                report.number_of_evaluations
            );
            log!(
                level,
                "Least-squares final objective: {}",
                report.objective_function
            );
            log!(level, "Least-squares final parameters: {:?}", solution);
        }
        if report.termination.was_successful() {
            self.result = Some(solution.clone());
            let solution: Vec<f64> = solution.data.into();
            let unknowns = self.values.clone();
            let map_of_solutions: HashMap<String, f64> = unknowns
                .iter()
                .zip(solution.iter())
                .map(|(k, v)| (k.to_string(), *v))
                .collect();

            let map_of_solutions = map_of_solutions;
            if let Some(level) = self.log_level() {
                log!(level, "Least-squares solution map: {:?}", map_of_solutions);
            }
            self.map_of_solutions = Some(map_of_solutions);
        }
    }

    /// Solves parametric system without modifying self, returns solution map and vector.
    pub fn solve_with_params_unmut_internal(
        &self,
        params: Vec<f64>,
    ) -> (Option<HashMap<String, f64>>, Option<DVector<f64>>) {
        let (solution, report) = {
            let prepared = self
                .prepared
                .as_ref()
                .expect("symbolic least-squares problem must be prepared");
            let problem = prepared
                .bind_values_with_guess(
                    DVector::from_vec(params),
                    DVector::from_vec(self.initial_guess.clone()),
                )
                .expect("symbolic parameter values and initial guess should bind");
            let solver = NonlinearSolver::LeastSquares(self.configured_solver());
            let (result, report) = solver
                .minimize_least_squares(problem)
                .expect("least-squares selector should accept its prepared problem");
            (result.params(), report)
        };
        if let Some(level) = self.log_level() {
            log!(level, "Least-squares termination: {:?}", report.termination);
            log!(
                level,
                "Least-squares evaluations: {}",
                report.number_of_evaluations
            );
            log!(
                level,
                "Least-squares final objective: {}",
                report.objective_function
            );
            log!(level, "Least-squares final parameters: {:?}", solution);
        }
        if report.termination.was_successful() {
            let solution_: DVector<f64> = solution;
            // self.result = Some(solution.clone());
            let solution: Vec<f64> = solution_.clone().data.into();
            let unknowns = self.values.clone();
            let map_of_solutions: HashMap<String, f64> = unknowns
                .iter()
                .zip(solution.iter())
                .map(|(k, v)| (k.to_string(), *v))
                .collect();

            let map_of_solutions: HashMap<String, f64> = map_of_solutions;
            if let Some(level) = self.log_level() {
                log!(level, "Least-squares solution map: {:?}", map_of_solutions);
            }
            return (Some(map_of_solutions), Some(solution_));
        } else {
            (None, None)
        }
    }

    /// Solves parametric system with given parameter values and optional logging.
    pub fn solve_with_params_unmut(
        &self,
        params: Vec<f64>,
    ) -> (Option<HashMap<String, f64>>, Option<DVector<f64>>) {
        self.solve_with_params_unmut_internal(params)
    }

    pub fn solve_with_params(&mut self, params: Vec<f64>) {
        self.ensure_prepared();
        let (map_of_solutions, solution) = self.solve_with_params_unmut(params);
        self.map_of_solutions = map_of_solutions;
        self.result = solution;
    }
}

fn parse_log_level(value: &str) -> Result<Option<Level>, LeastSquaresError> {
    match value.to_ascii_lowercase().as_str() {
        "off" | "none" => Ok(None),
        "debug" => Ok(Some(Level::Debug)),
        "info" => Ok(Some(Level::Info)),
        "warn" => Ok(Some(Level::Warn)),
        "error" => Ok(Some(Level::Error)),
        _ => Err(LeastSquaresError::InvalidLogLevel(value.to_string())),
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////TESTS////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::Nonlinear_systems::least_squares::{
        ClosureLeastSquaresProblem as NonlinearSystem, LevenbergMarquardt,
    };
    use nalgebra::DMatrix;

    #[test]
    fn test_nonlinear_system_example() {
        // Example: Solve the system:
        // x^2 + y^2 - 1 = 0
        // x - y = 0
        // Solution should be approximately (√2/2, √2/2) or (-√2/2, -√2/2)

        let initial_guess = DVector::from_vec(vec![0.5, 0.3]);

        // Define residuals function
        let residuals_fn = |params: &DVector<f64>| -> DVector<f64> {
            let x = params[0];
            let y = params[1];
            DVector::from_vec(vec![
                x * x + y * y - 1.0, // x^2 + y^2 - 1 = 0
                x - y,               // x - y = 0
            ])
        };

        // Define Jacobian function
        let jacobian_fn = |params: &DVector<f64>| -> DMatrix<f64> {
            let x = params[0];
            let y = params[1];
            DMatrix::from_row_slice(
                2,
                2,
                &[
                    2.0 * x,
                    2.0 * y, // ∂/∂x(x^2+y^2-1), ∂/∂y(x^2+y^2-1)
                    1.0,
                    -1.0, // ∂/∂x(x-y),       ∂/∂y(x-y)
                ],
            )
        };

        let problem = NonlinearSystem::new(initial_guess, residuals_fn, jacobian_fn);
        let (result, report) = LevenbergMarquardt::new().minimize(problem);

        println!("Nonlinear System Example:");
        println!("Termination: {:?}", report.termination);
        println!("Evaluations: {}", report.number_of_evaluations);
        println!("Final objective: {}", report.objective_function);
        println!("Final params: {:?}", result.params());

        let final_params = result.params();
        let expected = (2.0_f64).sqrt() / 2.0; // √2/2 ≈ 0.707

        // Check that we found a solution close to (√2/2, √2/2)
        assert!((final_params[0].abs() - expected).abs() < 1e-6);
        assert!((final_params[1].abs() - expected).abs() < 1e-6);
        assert!((final_params[0] - final_params[1]).abs() < 1e-10); // x ≈ y
    }

    #[test]
    fn test_simple_quadratic_system() {
        // Solve: x^2 - 4 = 0, solution should be x = ±2
        let initial_guess = DVector::from_vec(vec![1.0]);

        let residuals_fn = |params: &DVector<f64>| -> DVector<f64> {
            let x = params[0];
            DVector::from_vec(vec![x * x - 4.0])
        };

        let jacobian_fn = |params: &DVector<f64>| -> DMatrix<f64> {
            let x = params[0];
            DMatrix::from_row_slice(1, 1, &[2.0 * x])
        };

        let problem = NonlinearSystem::new(initial_guess, residuals_fn, jacobian_fn);
        let (result, report) = LevenbergMarquardt::new().minimize(problem);

        println!("\nSimple Quadratic System:");
        println!("Termination: {:?}", report.termination);
        println!("Final params: {:?}", result.params());

        let final_params = result.params();
        assert!((final_params[0].abs() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_complex_nonlinear_system() {
        // More complex system:
        // sin(x) + cos(y) - 1 = 0
        // x^2 + y^2 - 1 = 0

        let initial_guess = DVector::from_vec(vec![0.5, 0.5]);

        let residuals_fn = |params: &DVector<f64>| -> DVector<f64> {
            let x = params[0];
            let y = params[1];
            DVector::from_vec(vec![x.sin() + y.cos() - 1.0, x * x + y * y - 1.0])
        };

        let jacobian_fn = |params: &DVector<f64>| -> DMatrix<f64> {
            let x = params[0];
            let y = params[1];
            DMatrix::from_row_slice(
                2,
                2,
                &[
                    x.cos(),
                    -y.sin(), // ∂/∂x(sin(x)+cos(y)-1), ∂/∂y(sin(x)+cos(y)-1)
                    2.0 * x,
                    2.0 * y, // ∂/∂x(x^2+y^2-1),       ∂/∂y(x^2+y^2-1)
                ],
            )
        };

        let problem = NonlinearSystem::new(initial_guess, residuals_fn, jacobian_fn);
        let (result, report) = LevenbergMarquardt::new().with_tol(1e-12).minimize(problem);

        println!("\nComplex Nonlinear System:");
        println!("Termination: {:?}", report.termination);
        println!("Final params: {:?}", result.params());
        println!("Final objective: {}", report.objective_function);

        // Verify the solution satisfies the equations
        let final_params = result.params();
        let x = final_params[0];
        let y = final_params[1];

        let residual1 = x.sin() + y.cos() - 1.0;
        let residual2 = x * x + y * y - 1.0;

        assert!(residual1.abs() < 1e-10);
        assert!(residual2.abs() < 1e-10);
    }

    #[test]
    fn logging_is_opt_in_and_invalid_levels_are_typed() {
        let solver = SymbolicLeastSquaresSolver::new();
        assert!(solver.loglevel.is_none());
        assert!(solver.log_level().is_none());

        let result = SymbolicLeastSquaresSolver::new().try_with_loglevel("verbose");
        assert!(matches!(result, Err(LeastSquaresError::InvalidLogLevel(_))));

        let solver = SymbolicLeastSquaresSolver::new()
            .try_with_loglevel("debug")
            .expect("debug is supported");
        assert_eq!(solver.log_level(), Some(Level::Debug));
    }

    #[test]
    fn symbolic_try_build_and_solve_return_typed_input_errors() {
        let build_result = SymbolicLeastSquaresSolver::new().try_build();
        assert!(matches!(
            build_result,
            Err(LeastSquaresError::EmptyProblem { .. })
        ));

        let mut solver = SymbolicLeastSquaresSolver::new()
            .with_equations_str(vec!["x - 1".to_string()])
            .with_unknowns(vec!["x".to_string()])
            .with_initial_guess(vec![]);
        let error = solver
            .try_solve()
            .expect_err("initial-guess shape must fail before preparation");
        assert!(matches!(error, LeastSquaresError::DimensionMismatch { .. }));

        let parse_result =
            SymbolicLeastSquaresSolver::new().try_with_equations_str(vec!["x + (".to_string()]);
        assert!(matches!(parse_result, Err(LeastSquaresError::Problem(_))));
    }

    #[test]
    fn symbolic_typed_parameter_solve_rebinds_without_repreparing() {
        let mut solver = SymbolicLeastSquaresSolver::new()
            .with_equations_str(vec!["x - a".to_string()])
            .with_unknowns(vec!["x".to_string()])
            .with_parameters(vec!["a".to_string()])
            .with_initial_guess(vec![0.0])
            .with_telemetry(LeastSquaresTelemetryMode::Counters)
            .build();
        let report = solver
            .try_solve_with_params(vec![2.0])
            .expect("parameter rebind should solve");
        assert!(report.termination.was_successful());
        assert!((solver.result.as_ref().expect("result")[0] - 2.0).abs() < 1e-8);
        assert!(
            solver
                .last_statistics
                .as_ref()
                .expect("telemetry")
                .residual_evaluations
                > 0
        );
    }

    #[test]
    fn symbolic_wrapper_collects_opt_in_solver_statistics() {
        let mut solver = SymbolicLeastSquaresSolver::new()
            .with_equations_str(vec!["x - 1".to_string()])
            .with_unknowns(vec!["x".to_string()])
            .with_initial_guess(vec![0.0])
            .with_telemetry(LeastSquaresTelemetryMode::Counters)
            .build();
        solver.solve();

        let statistics = solver
            .last_statistics
            .as_ref()
            .expect("solve should publish its telemetry snapshot");
        assert!(statistics.availability.is_collected());
        assert!(statistics.residual_evaluations > 0);
        assert!(!statistics.timings_collected);
    }
}
/////////////////////////////////////////////////////////////////////////////////////
///   
#[cfg(test)]
mod tests2 {
    use super::*;
    use crate::symbolic::symbolic_engine::Expr;
    use std::vec;

    #[test]
    fn test_builder_pattern_basic() {
        let solver = SymbolicLeastSquaresSolver::new()
            .with_equations_str(vec!["x^2 + y^2 - 1".to_string(), "x - y".to_string()])
            .with_unknowns(vec!["x".to_string(), "y".to_string()])
            .with_initial_guess(vec![0.5, 0.5])
            .with_loglevel("none".to_string())
            .build();

        let mut solver = solver;
        solver.solve();
        let map = solver.map_of_solutions.unwrap();
        let expected = (2.0_f64).sqrt() / 2.0;
        assert!((map["x"].abs() - expected).abs() < 1e-6);
        assert!((map["y"].abs() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_builder_pattern_with_expr() {
        let eq1 = Expr::parse_expression("x^2 + y^2 - 1");
        let eq2 = Expr::parse_expression("x - y");

        let solver = SymbolicLeastSquaresSolver::new()
            .with_equations(vec![eq1, eq2])
            .with_unknowns(vec!["x".to_string(), "y".to_string()])
            .with_initial_guess(vec![0.5, 0.5])
            .with_tolerance(1e-8)
            .with_loglevel("none".to_string())
            .build();

        let mut solver = solver;
        solver.solve();
        assert!(solver.map_of_solutions.is_some());
    }

    #[test]
    fn test_builder_rosenbrock() {
        // Rosenbrock function: f1 = 10*(y - x^2), f2 = 1 - x
        let solver = SymbolicLeastSquaresSolver::new()
            .with_equations_str(vec!["10*(y - x^2)".to_string(), "1 - x".to_string()])
            .with_initial_guess(vec![-1.2, 1.0])
            .with_tolerance(1e-8)
            .with_max_iterations(200)
            .with_loglevel("none".to_string())
            .build();

        let mut solver = solver;
        solver.solve();
        let map = solver.map_of_solutions.unwrap();
        assert!((map["x"] - 1.0).abs() < 1e-5);
        assert!((map["y"] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_builder_exponential_system() {
        // exp(x) + y - 3 = 0, x + exp(y) - 3 = 0
        let solver = SymbolicLeastSquaresSolver::new()
            .with_equations_str(vec![
                "exp(x) + y - 3".to_string(),
                "x + exp(y) - 3".to_string(),
            ])
            .with_initial_guess(vec![0.5, 0.5])
            .with_f_tolerance(1e-8)
            .with_g_tolerance(1e-8)
            .with_loglevel("none".to_string())
            .build();

        let mut solver = solver;
        solver.solve();
        let map = solver.map_of_solutions.unwrap();
        // Solution should be symmetric: x ≈ y
        assert!((map["x"] - map["y"]).abs() < 1e-5);
    }

    #[test]
    fn test_builder_trigonometric_system() {
        // sin(x) + cos(y) - 1 = 0, cos(x) - sin(y) = 0
        let solver = SymbolicLeastSquaresSolver::new()
            .with_equations_str(vec![
                "sin(x) + cos(y) - 1".to_string(),
                "cos(x) - sin(y)".to_string(),
            ])
            .with_initial_guess(vec![0.5, 0.5])
            .with_tolerance(1e-7)
            .with_loglevel("none".to_string())
            .build();

        let mut solver = solver;
        solver.solve();
        assert!(solver.map_of_solutions.is_some());
    }

    #[test]
    fn test_builder_3d_system() {
        // x^2 + y^2 + z^2 - 1 = 0, x + y + z - 1 = 0, x - y = 0
        let solver = SymbolicLeastSquaresSolver::new()
            .with_equations_str(vec![
                "x^2 + y^2 + z^2 - 1".to_string(),
                "x + y + z - 1".to_string(),
                "x - y".to_string(),
            ])
            .with_initial_guess(vec![0.3, 0.3, 0.3])
            .with_tolerance(1e-7)
            .with_max_iterations(150)
            .with_loglevel("none".to_string())
            .build();

        let mut solver = solver;
        solver.solve();
        let map = solver.map_of_solutions.unwrap();
        // Check x = y
        assert!((map["x"] - map["y"]).abs() < 1e-5);
        // Check x + y + z = 1
        assert!((map["x"] + map["y"] + map["z"] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_nonlinear_system_example() {
        let vec_of_str = vec!["x^2 + y^2 - 1".to_string(), "x - y".to_string()];
        let initial_guess = vec![0.5, 0.5];
        let values = vec!["x".to_string(), "y".to_string()];
        let mut LM = SymbolicLeastSquaresSolver::new();
        LM.eq_generate_from_str(
            vec_of_str,
            Some(values),
            None,
            initial_guess,
            None,
            None,
            None,
            None,
            None,
        );
        LM.solve();
    }

    #[test]
    fn test_with_params() {
        // Solve: a*x^2 + b*y^2 - 1 = 0, x - y = 0 with params a=1, b=1
        let vec_of_str = vec!["a*x^2 + b*y^2 - 1".to_string(), "x - y".to_string()];
        let initial_guess = vec![0.5, 0.5];
        let values = vec!["x".to_string(), "y".to_string()];
        let params = vec!["a".to_string(), "b".to_string()];
        let mut LM = SymbolicLeastSquaresSolver::new();
        LM.eq_generate_from_str(
            vec_of_str,
            Some(values),
            Some(params),
            initial_guess,
            None,
            None,
            None,
            None,
            None,
        );
        LM.set_loglevel("info".to_string());
        LM.solve_with_params(vec![1.0, 1.0]);
        let map = LM.map_of_solutions.unwrap();
        let expected = (2.0_f64).sqrt() / 2.0;
        assert!((map["x"].abs() - expected).abs() < 1e-6);
        assert!((map["y"].abs() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_builder_with_params() {
        // Builder pattern with parameters
        let solver = SymbolicLeastSquaresSolver::new()
            .with_equations_str(vec!["a*x^2 + b*y^2 - 1".to_string(), "x - y".to_string()])
            .with_unknowns(vec!["x".to_string(), "y".to_string()])
            .with_parameters(vec!["a".to_string(), "b".to_string()])
            .with_initial_guess(vec![0.5, 0.5])
            .with_loglevel("none".to_string())
            .build();

        let mut solver = solver;
        solver.solve_with_params(vec![1.0, 1.0]);
        let map = solver.map_of_solutions.unwrap();
        let expected = (2.0_f64).sqrt() / 2.0;
        assert!((map["x"].abs() - expected).abs() < 1e-6);
        assert!((map["y"].abs() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_native_symbolic_construction() {
        // Using native Expr construction without string parsing
        let vars = Expr::Symbols("x, y");
        let x = vars[0].clone();
        let y = vars[1].clone();

        // Build equations: x^2 + y^2 - 1 = 0, x - y = 0
        let eq1 =
            x.clone().pow(Expr::Const(2.0)) + y.clone().pow(Expr::Const(2.0)) - Expr::Const(1.0);
        let eq2 = x.clone() - y.clone();

        let solver = SymbolicLeastSquaresSolver::new()
            .with_equations(vec![eq1, eq2])
            .with_unknowns(vec!["x".to_string(), "y".to_string()])
            .with_initial_guess(vec![0.5, 0.5])
            .with_loglevel("none".to_string())
            .build();

        let mut solver = solver;
        solver.solve();
        let map = solver.map_of_solutions.unwrap();
        let expected = (2.0_f64).sqrt() / 2.0;
        assert!((map["x"].abs() - expected).abs() < 1e-6);
        assert!((map["y"].abs() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_native_symbolic_exponential() {
        // Native symbolic construction with exponentials
        let vars = Expr::Symbols("x, y");
        let x = vars[0].clone();
        let y = vars[1].clone();

        // exp(x) + y - 3 = 0, x + exp(y) - 3 = 0
        let eq1 = Expr::exp(x.clone()) + y.clone() - Expr::Const(3.0);
        let eq2 = x.clone() + Expr::exp(y.clone()) - Expr::Const(3.0);

        let solver = SymbolicLeastSquaresSolver::new()
            .with_equations(vec![eq1, eq2])
            .with_initial_guess(vec![0.5, 0.5])
            .with_tolerance(1e-8)
            .with_loglevel("none".to_string())
            .build();

        let mut solver = solver;
        solver.solve();
        let map = solver.map_of_solutions.unwrap();
        // Solution should be symmetric
        assert!((map["x"] - map["y"]).abs() < 1e-5);
    }

    #[test]
    fn test_native_symbolic_trigonometric() {
        // Native symbolic construction with trig functions
        let vars = Expr::Symbols("x, y");
        let x = vars[0].clone();
        let y = vars[1].clone();

        // sin(x) + cos(y) - 1 = 0, cos(x) - sin(y) = 0
        let eq1 =
            Expr::sin(Box::new(x.clone())) + Expr::cos(Box::new(y.clone())) - Expr::Const(1.0);
        let eq2 = Expr::cos(Box::new(x.clone())) - Expr::sin(Box::new(y.clone()));

        let solver = SymbolicLeastSquaresSolver::new()
            .with_equations(vec![eq1, eq2])
            .with_initial_guess(vec![0.5, 0.5])
            .with_tolerance(1e-7)
            .with_loglevel("none".to_string())
            .build();

        let mut solver = solver;
        solver.solve();
        assert!(solver.map_of_solutions.is_some());
    }

    #[test]
    fn test_native_symbolic_logarithmic() {
        // Native symbolic construction with logarithms
        let vars = Expr::Symbols("x, y");
        let x = vars[0].clone();
        let y = vars[1].clone();

        // ln(x) + y - 2 = 0, x + ln(y) - 2 = 0
        let eq1 = Expr::ln(x.clone()) + y.clone() - Expr::Const(2.0);
        let eq2 = x.clone() + Expr::ln(y.clone()) - Expr::Const(2.0);

        let solver = SymbolicLeastSquaresSolver::new()
            .with_equations(vec![eq1, eq2])
            .with_initial_guess(vec![1.0, 1.0])
            .with_tolerance(1e-8)
            .with_loglevel("none".to_string())
            .build();

        let mut solver = solver;
        solver.solve();
        let map = solver.map_of_solutions.unwrap();
        // Solution should be symmetric
        assert!((map["x"] - map["y"]).abs() < 1e-5);
    }

    #[test]
    fn test_native_symbolic_complex_expression() {
        // Complex expression with multiple operations
        let vars = Expr::Symbols("x, y, z");
        let x = vars[0].clone();
        let y = vars[1].clone();
        let z = vars[2].clone();

        // x^2 + y^2 + z^2 - 1 = 0
        let eq1 = x.clone().pow(Expr::Const(2.0))
            + y.clone().pow(Expr::Const(2.0))
            + z.clone().pow(Expr::Const(2.0))
            - Expr::Const(1.0);

        // x*y + z - 0.5 = 0
        let eq2 = x.clone() * y.clone() + z.clone() - Expr::Const(0.5);

        // x + y + z - 1 = 0
        let eq3 = x.clone() + y.clone() + z.clone() - Expr::Const(1.0);

        let solver = SymbolicLeastSquaresSolver::new()
            .with_equations(vec![eq1, eq2, eq3])
            .with_unknowns(vec!["x".to_string(), "y".to_string(), "z".to_string()])
            .with_initial_guess(vec![0.3, 0.3, 0.4])
            .with_tolerance(1e-7)
            .with_max_iterations(200)
            .with_loglevel("none".to_string())
            .build();

        let mut solver = solver;
        solver.solve();
        let map = solver.map_of_solutions.unwrap();

        // Verify constraints
        let sphere = map["x"].powi(2) + map["y"].powi(2) + map["z"].powi(2);
        let product = map["x"] * map["y"] + map["z"];
        let sum = map["x"] + map["y"] + map["z"];

        assert!((sphere - 1.0).abs() < 1e-5);
        assert!((product - 0.5).abs() < 1e-5);
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_native_symbolic_with_parameters() {
        // Native symbolic construction with parameters
        let vars = Expr::Symbols("x, y");
        let params = Expr::Symbols("a, b");
        let x = vars[0].clone();
        let y = vars[1].clone();
        let a = params[0].clone();
        let b = params[1].clone();

        // a*x^2 + b*y^2 - 1 = 0, x - y = 0
        let eq1 = a * x.clone().pow(Expr::Const(2.0)) + b * y.clone().pow(Expr::Const(2.0))
            - Expr::Const(1.0);
        let eq2 = x.clone() - y.clone();

        let solver = SymbolicLeastSquaresSolver::new()
            .with_equations(vec![eq1, eq2])
            .with_unknowns(vec!["x".to_string(), "y".to_string()])
            .with_parameters(vec!["a".to_string(), "b".to_string()])
            .with_initial_guess(vec![0.5, 0.5])
            .with_tolerance(1e-8)
            .with_loglevel("none".to_string())
            .build();

        let mut solver = solver;
        solver.solve_with_params(vec![1.0, 1.0]);
        let map = solver.map_of_solutions.unwrap();
        let expected = (2.0_f64).sqrt() / 2.0;
        assert!((map["x"].abs() - expected).abs() < 1e-6);
        assert!((map["y"].abs() - expected).abs() < 1e-6);
    }

    #[test]
    fn chemical_equations() {
        let symbolic = Expr::Symbols("N0, N1, N2, Np, Lambda0, Lambda1");

        let dGm0 = Expr::Const(8.314 * 8.0e4); //8.314 * 8.0e3
        let dG0 = Expr::Const(-450.0e3);
        let dG1 = Expr::Const(-150.0e3);
        let dG2 = Expr::Const(-50e3);
        let N0 = symbolic[0].clone();
        let N1 = symbolic[1].clone();
        let N2 = symbolic[2].clone();
        let Np = symbolic[3].clone();
        let Lambda0 = symbolic[4].clone();
        let Lambda1 = symbolic[5].clone();

        let RT = Expr::Const(8.314) * Expr::Const(3250.0);
        let eq_mu = vec![
            Lambda0.clone()
                + Expr::Const(2.0) * Lambda1.clone()
                + (dG0.clone() + RT.clone() * Expr::ln(N0.clone() / Np.clone())) / dGm0.clone(),
            Lambda0
                + Lambda1.clone()
                + (dG1 + RT.clone() * Expr::ln(N1.clone() / Np.clone())) / dGm0.clone(),
            Expr::Const(2.0) * Lambda1
                + (dG2 + RT * Expr::ln(N2.clone() / Np.clone())) / dGm0.clone(),
        ];
        let eq_sum_mole_numbers = vec![N0.clone() + N1.clone() + N2.clone() - Np.clone()];
        let composition_eq = vec![
            N0.clone() + N1.clone() - Expr::Const(0.999),
            Expr::Const(2.0) * N0.clone() + N1.clone() + Expr::Const(2.0) * N2 - Expr::Const(1.501),
        ];

        let mut full_system_sym = Vec::new();
        full_system_sym.extend(eq_mu.clone());
        full_system_sym.extend(eq_sum_mole_numbers.clone());
        full_system_sym.extend(composition_eq.clone());

        let full_system_sym: Vec<Expr> = full_system_sym
            .iter()
            .map(|x| x.clone().simplify())
            .collect();

        for eq in &full_system_sym {
            println!("eq: {}", eq.clone().pretty_print());
        }
        // solver
        let initial_guess = vec![0.1, 0.1, 0.2, 0.3, 2.0, 2.0];
        let unknowns: Vec<String> = symbolic.iter().map(|x| x.to_string()).collect();
        let mut LM = SymbolicLeastSquaresSolver::new();
        LM.set_loglevel("none".to_string());
        LM.set_equation_system(
            full_system_sym.clone(),
            Some(unknowns.clone()),
            None,
            initial_guess,
            None,
            Some(1e-6),
            Some(1e-6),
            Some(true),
            None,
        );
        LM.set_positive_variables(&["N0", "N1", "N2", "Np"])
            .expect("chemical logarithm variables should have a valid domain");
        LM.solve();
        let map_of_solutions = LM.map_of_solutions.unwrap();

        let N0 = map_of_solutions.get("N0").unwrap();
        let N1 = map_of_solutions.get("N1").unwrap();
        let N2 = map_of_solutions.get("N2").unwrap();
        let Np = map_of_solutions.get("Np").unwrap();
        let _Lambda0 = map_of_solutions.get("Lambda0").unwrap();
        let _Lambda1 = map_of_solutions.get("Lambda1").unwrap();
        let d1 = *N0 + *N1 - 0.999;
        let d2 = N0 + N1 + N2 - Np;
        let d3 = 2.0 * N0 + N1 + 2.0 * N2 - 1.501;
        println!("d1: {}", d1);
        println!("d2: {}", d2);
        println!("d3: {}", d3);
        println!("map_of_solutions: {:?}", map_of_solutions);
        assert!(d1.abs() < 1e-3);
        assert!(d2.abs() < 1e-2);
        assert!(d3.abs() < 1e-2);
    }

    #[test]
    fn test_native_symbolic_division_operations() {
        // Test with division and complex operations
        let vars = Expr::Symbols("x, y");
        let x = vars[0].clone();
        let y = vars[1].clone();

        // x/y - 2 = 0, x + y - 3 = 0
        let eq1 = x.clone() / y.clone() - Expr::Const(2.0);
        let eq2 = x.clone() + y.clone() - Expr::Const(3.0);

        let solver = SymbolicLeastSquaresSolver::new()
            .with_equations(vec![eq1, eq2])
            .with_initial_guess(vec![1.5, 1.0])
            .with_tolerance(1e-8)
            .with_loglevel("none".to_string())
            .build();

        let mut solver = solver;
        solver.solve();
        let map = solver.map_of_solutions.unwrap();

        // x = 2, y = 1
        assert!((map["x"] - 2.0).abs() < 1e-5);
        assert!((map["y"] - 1.0).abs() < 1e-5);
    }
}
