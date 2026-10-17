//! # Symbolic Shooting Method Wrapper for Boundary Value Problems
//!
//! This module provides a high-level interface for solving boundary value problems (BVPs)
//! using the shooting method with symbolic equation definitions and various IVP solvers.
//!
//! ## Overview
//!
//! The shooting method converts a BVP into an initial value problem (IVP) by guessing
//! unknown initial conditions and iteratively adjusting them until boundary conditions
//! are satisfied. This wrapper allows you to:
//!
//! - Define ODEs symbolically using string expressions
//! - Use any available IVP solver (RK45, BDF, Radau, etc.)
//! - Handle mixed boundary conditions (Dirichlet/Neumann)
//! - Automatically generate equation closures from symbolic expressions
//!
//! ## Supported Boundary Conditions
//!
//! - **Dirichlet**: y(x) = value (position specified)
//! - **Neumann**: y'(x) = value (derivative specified)
//! - **Mixed**: Combinations of Dirichlet and Neumann at different boundaries
//!
//! ## Variable Convention
//!
//! For second-order ODEs converted to first-order systems:
//! - `y0` represents the function value y(x)
//! - `y1` represents the derivative y'(x)
//!
//! ## Usage Example
//!
//! ```rust, ignore
//! use std::collections::HashMap;
//! use RustedSciThe::numerical::ShootingBVP::Shooting_sym_wrap::BVPShooting;
//! use RustedSciThe::numerical::ODE_api2::{NonStiffMethod, SolverType, SolverParam};
//! use RustedSciThe::symbolic::symbolic_engine::Expr;
//!
//! // Define second-order ODE: y'' + y = 0
//! // Convert to first-order system: y0' = y1, y1' = -y0
//! let eq_vec = vec![
//!     Expr::parse_expression("y1"),    // y0' = y1
//!     Expr::parse_expression("-y0")   // y1' = -y0
//! ];
//! let values = vec!["y0".to_string(), "y1".to_string()];
//! let arg = "x".to_string();
//!
//! // Set boundary conditions: y(0) = 1, y(π/2) = 0
//! let mut boundary_conditions = HashMap::new();
//! boundary_conditions.insert("y0".to_string(), vec![(0, 1.0), (1, 0.0)]);
//!
//! let borders = (0.0, std::f64::consts::PI / 2.0);
//!
//! // Create BVP instance
//! let mut bvp = BVPShooting::new(eq_vec, values, arg, boundary_conditions, borders);
//!
//! // Solve with RK45 (simple method)
//! bvp.simple_solve(0.0, 1e-6, 100, 0.01);
//!
//! // Or solve with specific IVP solver
//! let params = HashMap::new();
//! bvp.solve_with_certain_ivp(
//!     0.0, 1e-6, 100, 0.01,
//!     SolverType::NonStiff(NonStiffMethod::Rk45),
//!     params
//! );
//!
//! // Get results
//! let solution = bvp.get_solution();
//! let x_mesh = bvp.get_x();
//! let y_matrix = bvp.get_y();
//! ```

use crate::numerical::ODE_api2::{SolverParam, SolverType, UniversalODESolver};
use crate::numerical::ShootingBVP::Shooting_simple::{
    BoundaryCondition, BoundaryConditionType, BoundaryValueProblem, RK45_ivp_solver_and_mesh,
    ShootingBvpError, ShootingMethodResult, ShootingMethodSolver, ShootingSolveReport,
};
use crate::numerical::ShootingBVP::symbolic::{
    PreparedShootingRhs, ShootingSymbolicOptions, ShootingSymbolicSolveReport,
};

use crate::numerical::SolverTelemetryMode;
use crate::numerical::logging::{
    SolverLogLevel, solver_debug as debug, solver_error as error, solver_info as info,
};
use crate::symbolic::symbolic_engine::Expr;
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

type NativeShootingRhs = Arc<dyn Fn(f64, &DVector<f64>) -> DVector<f64> + Send + Sync>;
/// Main structure for solving boundary value problems using the shooting method
/// with symbolic equation definitions and configurable IVP solvers.
pub struct BVPShooting {
    /// Vector of symbolic expressions representing the ODE system
    pub eq_vec: Vec<Expr>,
    /// Names of the dependent variables (e.g., ["y0", "y1"] for y and y')
    pub values: Vec<String>,
    /// Name of the independent variable (e.g., "x" or "t")
    pub arg: String,
    /// Boundary conditions map: variable_name -> [(boundary_index, value)]
    /// boundary_index: 0 = left boundary, 1 = right boundary
    pub BoundaryConditions: HashMap<String, Vec<(usize, f64)>>,
    /// Internal shooting method solver instance
    pub solver: ShootingMethodSolver,
    /// Type of IVP solver to use (RK45, BDF, Radau, etc.)
    pub ivp_solver: SolverType,
    /// Parameters for the IVP solver
    pub solver_params: HashMap<String, SolverParam>,
    /// Domain boundaries (start, end)
    pub borders: (f64, f64),
    /// Solution matrix (rows = variables, columns = time steps)
    pub solution: DMatrix<f64>,
    /// Complete solution result including mesh and boundary values
    pub full_sol: ShootingMethodResult,
    /// Per-instance Rust log threshold. Logging is disabled by default.
    logging_level: SolverLogLevel,
}
impl BVPShooting {
    /// Create a new BVP shooting method solver instance.
    ///
    /// # Arguments
    /// * `eq_vec` - Vector of symbolic expressions representing the ODE system
    /// * `values` - Names of dependent variables (e.g., ["y0", "y1"])
    /// * `arg` - Name of independent variable (e.g., "x")
    /// * `BoundaryConditions` - Map of boundary conditions
    /// * `borders` - Domain boundaries (start, end)
    pub fn new(
        eq_vec: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        BoundaryConditions: HashMap<String, Vec<(usize, f64)>>,
        borders: (f64, f64),
    ) -> Self {
        BVPShooting {
            eq_vec,
            values,
            arg,
            solver: ShootingMethodSolver::new(),
            ivp_solver: SolverType::Bdf,
            solver_params: HashMap::new(),
            BoundaryConditions: BoundaryConditions,
            borders: borders,
            solution: DMatrix::zeros(0, 0),
            full_sol: ShootingMethodResult::default(),
            logging_level: SolverLogLevel::Off,
        }
    }

    /// Sets the maximum Rust log level emitted while solving this BVP.
    ///
    /// The host application remains responsible for installing a `log`
    /// backend; the default [`SolverLogLevel::Off`] keeps the library silent.
    pub fn set_logging_level(&mut self, level: SolverLogLevel) {
        self.logging_level = level;
    }

    /// Returns this solver's configured Rust log threshold.
    pub fn logging_level(&self) -> SolverLogLevel {
        self.logging_level
    }

    /// Builder-style configuration for the solver's Rust log threshold.
    pub fn with_logging_level(mut self, level: SolverLogLevel) -> Self {
        self.logging_level = level;
        self
    }
    /// Set the IVP solver type and parameters.
    ///
    /// # Arguments
    /// * `ivp_solver` - Type of IVP solver (RK45, BDF, Radau, etc.)
    /// * `params` - Solver-specific parameters
    pub fn set_parameters(&mut self, ivp_solver: SolverType, params: HashMap<String, SolverParam>) {
        self.ivp_solver = ivp_solver;
        self.solver_params = params;
    }
    pub fn generate_eq_closure(&self) -> Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>> {
        let eq_vec = self.eq_vec.clone();
        let values = self.values.clone();
        let arg = self.arg.clone();
        let n = eq_vec.len();
        let residual = Box::new(move |t: f64, y: &DVector<f64>| -> DVector<f64> {
            let mut args = Vec::with_capacity(1 + n);
            args.push(t);
            args.extend(y.iter().cloned());
            let mut res = DVector::zeros(n);
            let mut all_var_names = vec![arg.as_str()];
            all_var_names.extend(values.iter().map(|s| s.as_str()));
            for i in 0..n {
                res[i] = eq_vec[i].eval_expression(all_var_names.as_slice(), &args);
            }
            res
        });
        residual
    }
    //

    pub fn BC_create(&self) -> (BoundaryCondition, BoundaryCondition) {
        let values = &self.values;
        let map_of_bc = &self.BoundaryConditions;

        let mut left_bc: Option<BoundaryCondition> = None;
        let mut right_bc: Option<BoundaryCondition> = None;

        // Process all boundary conditions
        for var_name in values {
            if let Some(conditions) = map_of_bc.get(var_name) {
                let var_idx = values.iter().position(|v| v == var_name).unwrap();
                for (boundary_idx, value) in conditions {
                    // For shooting method: y0 is position (Dirichlet), y1 is derivative (Neumann)
                    let bc_type: BoundaryConditionType = if var_idx == 0 {
                        BoundaryConditionType::Dirichlet // y0 -> position boundary condition
                    } else {
                        BoundaryConditionType::Neumann // y1 -> derivative boundary condition
                    };

                    let bc = BoundaryCondition {
                        value: *value,
                        bc_type: bc_type,
                    };

                    if *boundary_idx == 0 {
                        // 0 means left boundary - only set if not already set
                        if left_bc.is_none() {
                            left_bc = Some(bc);
                        }
                    } else {
                        // 1 means right boundary - only set if not already set
                        if right_bc.is_none() {
                            right_bc = Some(bc);
                        }
                    }
                }
            }
        }

        // Return with defaults if not set
        let left = left_bc.unwrap_or(BoundaryCondition {
            value: 0.0,
            bc_type: BoundaryConditionType::Dirichlet,
        });
        let right = right_bc.unwrap_or(BoundaryCondition {
            value: 0.0,
            bc_type: BoundaryConditionType::Dirichlet,
        });

        debug!("Left BC: value={}, type={:?}", left.value, left.bc_type);
        debug!("Right BC: value={}, type={:?}", right.value, right.bc_type);

        (left, right)
    }
    /// Solve the BVP using the shooting method with built-in RK45 IVP solver.
    ///
    /// This compatibility interface uses the legacy fixed-step RK4 IVP helper
    /// (despite its historical `RK45_ivp_solver_and_mesh` name).
    ///
    /// # Arguments
    /// * `initial_guess` - Initial guess for unknown boundary condition
    /// * `tolerance` - Convergence tolerance for shooting method
    /// * `max_iterations` - Maximum iterations for shooting method
    /// * `step_size` - Step size for RK45 integration
    pub fn simple_solve(
        &mut self,
        initial_guess: f64,
        tolerance: f64,
        max_iterations: usize,
        step_size: f64,
    ) {
        if let Err(solve_error) = self.try_simple_solve(
            initial_guess,
            tolerance,
            max_iterations,
            step_size,
            SolverTelemetryMode::Off,
        ) {
            error!("Shooting BVP solve failed: {solve_error}");
        }
    }

    /// Fallible RK45 shooting solve with the shared telemetry collection mode.
    ///
    /// On failure, the previously stored trajectory is cleared so callers
    /// cannot mistake stale output for the result of this solve attempt.
    pub fn try_simple_solve(
        &mut self,
        initial_guess: f64,
        tolerance: f64,
        max_iterations: usize,
        step_size: f64,
        telemetry_mode: SolverTelemetryMode,
    ) -> Result<ShootingSolveReport, ShootingBvpError> {
        let ode_system = self.generate_eq_closure();
        let (a, b) = self.borders;
        let (left_bc, right_bc) = self.BC_create();

        let problem = BoundaryValueProblem::new(ode_system, a, b, left_bc, right_bc);

        let mut solver = ShootingMethodSolver {
            initial_guess,
            tolerance,
            max_iterations,
            step_size,
            result: ShootingMethodResult::default(),
        };
        self.full_sol = ShootingMethodResult::default();
        self.solution = DMatrix::zeros(0, 0);
        let report = solver.try_solve_with_options(
            &problem,
            |x0, y0, x_end, h| {
                Ok::<_, std::convert::Infallible>(RK45_ivp_solver_and_mesh(
                    x0,
                    y0,
                    x_end,
                    h,
                    &problem.ode_system,
                ))
            },
            telemetry_mode,
            self.logging_level,
        )?;
        self.full_sol = report.solution.clone();
        self.solution = report.solution.y.clone();
        Ok(report)
    }

    /// Solve the BVP using the shooting method with a specified IVP solver.
    ///
    /// This method allows you to choose any available IVP solver (RK45, BDF, Radau, etc.)
    /// and configure its parameters.
    ///
    /// # Arguments
    /// * `initial_guess` - Initial guess for unknown boundary condition
    /// * `tolerance` - Convergence tolerance for shooting method
    /// * `max_iterations` - Maximum iterations for shooting method
    /// * `step_size` - Step size for integration
    /// * `ivp_solver` - Type of IVP solver to use
    /// * `params` - Parameters specific to the chosen IVP solver
    pub fn solve_with_certain_ivp(
        &mut self,
        initial_guess: f64,
        tolerance: f64,
        max_iterations: usize,
        step_size: f64,
        ivp_solver: SolverType,
        params: HashMap<String, SolverParam>,
    ) {
        if let Err(solve_error) = self.try_solve_with_certain_ivp(
            initial_guess,
            tolerance,
            max_iterations,
            step_size,
            ivp_solver,
            params,
            SolverTelemetryMode::Off,
        ) {
            error!("Shooting BVP solve failed: {solve_error}");
        }
    }

    /// Fallible native-IVP solve that preserves typed backend failures.
    ///
    /// Counter and timing collection are independently controlled by
    /// [`SolverTelemetryMode`].
    pub fn try_solve_with_certain_ivp(
        &mut self,
        initial_guess: f64,
        tolerance: f64,
        max_iterations: usize,
        step_size: f64,
        ivp_solver: SolverType,
        params: HashMap<String, SolverParam>,
        telemetry_mode: SolverTelemetryMode,
    ) -> Result<ShootingSolveReport, ShootingBvpError> {
        self.try_solve_native_internal(
            initial_guess,
            tolerance,
            max_iterations,
            step_size,
            ivp_solver,
            params,
            telemetry_mode,
            None,
        )
    }

    fn try_solve_native_internal(
        &mut self,
        initial_guess: f64,
        tolerance: f64,
        max_iterations: usize,
        step_size: f64,
        ivp_solver: SolverType,
        params: HashMap<String, SolverParam>,
        telemetry_mode: SolverTelemetryMode,
        native_rhs: Option<NativeShootingRhs>,
    ) -> Result<ShootingSolveReport, ShootingBvpError> {
        let ode_system = self.generate_eq_closure();
        let (t0, tend) = self.borders;
        let (left_bc, right_bc) = self.BC_create();

        let problem = BoundaryValueProblem::new(ode_system, t0, tend, left_bc, right_bc);
        let mut solver = ShootingMethodSolver {
            initial_guess,
            tolerance,
            max_iterations,
            step_size,
            result: ShootingMethodResult::default(),
        };
        self.full_sol = ShootingMethodResult::default();
        self.solution = DMatrix::zeros(0, 0);
        let report = solver.try_solve_with_options(
            &problem,
            |x0, y0, x_end, h| {
                Self::try_ivp_solver_with_native_rhs(
                    self.eq_vec.clone(),
                    self.values.clone(),
                    self.arg.clone(),
                    x0,
                    y0,
                    x_end,
                    ivp_solver.clone(),
                    params.clone(),
                    h,
                    native_rhs.clone(),
                )
                .map(|(mesh, trajectory)| (trajectory.transpose(), mesh))
            },
            telemetry_mode,
            self.logging_level,
        )?;
        self.full_sol = report.solution.clone();
        self.solution = report.solution.y.clone();
        Ok(report)
    }

    /// Solves through the shared symbolic Lambdify preparation path.
    ///
    /// The symbolic system is prepared once before the secant iterations and
    /// reused for every native IVP solve. ExprLegacy is the default; select
    /// AtomViewNative in `options` for the packed Atom route. AOT is not used.
    pub fn try_solve_symbolic(
        &mut self,
        initial_guess: f64,
        tolerance: f64,
        max_iterations: usize,
        step_size: f64,
        ivp_solver: SolverType,
        params: HashMap<String, SolverParam>,
        options: ShootingSymbolicOptions,
        telemetry_mode: SolverTelemetryMode,
    ) -> Result<ShootingSymbolicSolveReport, ShootingBvpError> {
        let prepared = Arc::new(
            PreparedShootingRhs::prepare(
                self.eq_vec.clone(),
                self.values.clone(),
                self.arg.clone(),
                options,
            )
            .map_err(|error| ShootingBvpError::SymbolicPreparation(Box::new(error)))?,
        );
        let callback_prepared = Arc::clone(&prepared);
        let callback_error = Arc::new(Mutex::new(None));
        let callback_error_sink = Arc::clone(&callback_error);
        let rhs: NativeShootingRhs = Arc::new(move |coordinate, state| {
            match callback_prepared.evaluate(coordinate, state) {
                Ok(values) => values,
                Err(error) => {
                    let mut first_error = callback_error_sink
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    if first_error.is_none() {
                        *first_error = Some(error);
                    }
                    DVector::from_element(state.len(), f64::NAN)
                }
            }
        });

        let solve = self.try_solve_native_internal(
            initial_guess,
            tolerance,
            max_iterations,
            step_size,
            ivp_solver,
            params,
            telemetry_mode,
            Some(rhs),
        );
        let solve = match solve {
            Err(error) => {
                let symbolic_error = callback_error
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .take();
                if let Some(symbolic_error) = symbolic_error {
                    return Err(ShootingBvpError::SymbolicEvaluation(Box::new(
                        symbolic_error,
                    )));
                }
                return Err(error);
            }
            Ok(report) => report,
        };
        if let Some(symbolic_error) = callback_error
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            return Err(ShootingBvpError::SymbolicEvaluation(Box::new(
                symbolic_error,
            )));
        }

        Ok(ShootingSymbolicSolveReport {
            solve,
            frontend: prepared.frontend(),
            preparation: prepared.preparation_report().clone(),
            preparation_wall_time: prepared.preparation_wall_time(),
        })
    }

    /// Solve the BVP using the shooting method with a generic typed IVP
    /// configurator instead of the legacy `HashMap<String, SolverParam>` bag.
    pub fn solve_with_certain_ivp_generic<F>(
        &mut self,
        initial_guess: f64,
        tolerance: f64,
        max_iterations: usize,
        step_size: f64,
        ivp_solver: SolverType,
        configure: F,
    ) where
        F: Fn(UniversalODESolver) -> UniversalODESolver + Clone + 'static,
    {
        if let Err(solve_error) = self.try_solve_with_certain_ivp_generic(
            initial_guess,
            tolerance,
            max_iterations,
            step_size,
            ivp_solver,
            configure,
            SolverTelemetryMode::Off,
        ) {
            error!("Shooting BVP solve failed: {solve_error}");
        }
    }

    /// Fallible generic-configurator route with preserved IVP error sources.
    pub fn try_solve_with_certain_ivp_generic<F>(
        &mut self,
        initial_guess: f64,
        tolerance: f64,
        max_iterations: usize,
        step_size: f64,
        ivp_solver: SolverType,
        configure: F,
        telemetry_mode: SolverTelemetryMode,
    ) -> Result<ShootingSolveReport, ShootingBvpError>
    where
        F: Fn(UniversalODESolver) -> UniversalODESolver + Clone + 'static,
    {
        let ode_system = self.generate_eq_closure();
        let (t0, tend) = self.borders;
        let (left_bc, right_bc) = self.BC_create();

        let problem = BoundaryValueProblem::new(ode_system, t0, tend, left_bc, right_bc);
        let mut solver = ShootingMethodSolver {
            initial_guess,
            tolerance,
            max_iterations,
            step_size,
            result: ShootingMethodResult::default(),
        };
        self.full_sol = ShootingMethodResult::default();
        self.solution = DMatrix::zeros(0, 0);
        let report = solver.try_solve_with_options(
            &problem,
            |x0, y0, x_end, _step| {
                Self::try_ivp_solver_generic(
                    self.eq_vec.clone(),
                    self.values.clone(),
                    self.arg.clone(),
                    x0,
                    y0,
                    x_end,
                    ivp_solver.clone(),
                    configure.clone(),
                )
                .map(|(mesh, trajectory, _stats)| (trajectory.transpose(), mesh))
            },
            telemetry_mode,
            self.logging_level,
        )?;
        self.full_sol = report.solution.clone();
        self.solution = report.solution.y.clone();
        Ok(report)
    }
    /// Get the complete solution result including mesh, solution matrix, and boundary values.
    pub fn get_solution(&self) -> ShootingMethodResult {
        return self.full_sol.clone();
    }

    /// Get the spatial mesh (x-coordinates) of the solution.
    pub fn get_x(&self) -> DVector<f64> {
        return self.full_sol.x_mesh.clone();
    }

    /// Get the solution matrix where rows are variables and columns are spatial points.
    pub fn get_y(&self) -> DMatrix<f64> {
        return self.full_sol.y.clone();
    }
    pub fn IVP_solver(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        solver_type: SolverType,
        params: HashMap<String, SolverParam>,
        step: f64,
    ) -> Result<(DVector<f64>, DMatrix<f64>), String> {
        let mut solver =
            UniversalODESolver::new_legacy(eq_system, values, arg, solver_type, t0, y0, t_bound);
        let mut params = params;
        let add_step_size = HashMap::from([("step_size".to_string(), SolverParam::Float(step))]);
        params.extend(add_step_size);
        info!("solving with params {:?}", params);
        solver.set_parameters_legacy(params);
        solver
            .try_initialize()
            .map_err(|e| format!("failed to initialize universal IVP solver: {e}"))?;
        solver
            .try_solve()
            .map_err(|e| format!("failed to solve universal IVP problem: {e}"))?;
        let (x, y) = solver.get_result();
        match (x, y) {
            (Some(x_vec), Some(y_mat)) => Ok((x_vec, y_mat)),
            _ => Err("IVP solver failed to produce results".to_string()),
        }
    }

    /// Typed counterpart of [`Self::IVP_solver`] used by fallible solve APIs.
    pub fn try_ivp_solver(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        solver_type: SolverType,
        params: HashMap<String, SolverParam>,
        step: f64,
    ) -> Result<(DVector<f64>, DMatrix<f64>), ShootingBvpError> {
        Self::try_ivp_solver_with_native_rhs(
            eq_system,
            values,
            arg,
            t0,
            y0,
            t_bound,
            solver_type,
            params,
            step,
            None,
        )
    }

    fn try_ivp_solver_with_native_rhs(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        solver_type: SolverType,
        mut params: HashMap<String, SolverParam>,
        step: f64,
        native_rhs: Option<NativeShootingRhs>,
    ) -> Result<(DVector<f64>, DMatrix<f64>), ShootingBvpError> {
        let mut solver =
            UniversalODESolver::new_legacy(eq_system, values, arg, solver_type, t0, y0, t_bound);
        params.insert("step_size".to_string(), SolverParam::Float(step));
        solver.set_parameters_legacy(params);
        if let Some(rhs) = native_rhs {
            solver = solver.with_native_ode_callbacks(
                move |time, state| rhs(time, state),
                None::<fn(f64, &DVector<f64>) -> DMatrix<f64>>,
            );
        }
        solver
            .try_initialize()
            .map_err(|error| ShootingBvpError::IvpFailure(Box::new(error)))?;
        solver
            .try_solve()
            .map_err(|error| ShootingBvpError::IvpFailure(Box::new(error)))?;
        let (mesh, trajectory) = solver.get_result();
        match (mesh, trajectory) {
            (Some(mesh), Some(trajectory)) => Ok((mesh, trajectory)),
            _ => Err(ShootingBvpError::MissingIvpResult),
        }
    }

    pub fn IVP_solver_generic<F>(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        solver_type: SolverType,
        configure: F,
    ) -> Result<(DVector<f64>, DMatrix<f64>, Option<String>), String>
    where
        F: FnOnce(UniversalODESolver) -> UniversalODESolver,
    {
        let solver = UniversalODESolver::new(eq_system, values, arg, solver_type, t0, y0, t_bound);
        let mut solver = configure(solver);
        solver
            .try_initialize()
            .map_err(|e| format!("failed to initialize universal IVP solver: {e}"))?;
        solver
            .try_solve()
            .map_err(|e| format!("failed to solve universal IVP problem: {e}"))?;
        let stats_report = solver.statistics_report();
        let (x, y) = solver.get_result();
        match (x, y) {
            (Some(x_vec), Some(y_mat)) => Ok((x_vec, y_mat, stats_report)),
            _ => Err("IVP solver failed to produce results".to_string()),
        }
    }

    /// Typed generic-IVP adapter used by the fallible shooting API.
    pub fn try_ivp_solver_generic<F>(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        solver_type: SolverType,
        configure: F,
    ) -> Result<(DVector<f64>, DMatrix<f64>, Option<String>), ShootingBvpError>
    where
        F: FnOnce(UniversalODESolver) -> UniversalODESolver,
    {
        let solver = UniversalODESolver::new(eq_system, values, arg, solver_type, t0, y0, t_bound);
        let mut solver = configure(solver);
        solver
            .try_initialize()
            .map_err(|error| ShootingBvpError::IvpFailure(Box::new(error)))?;
        solver
            .try_solve()
            .map_err(|error| ShootingBvpError::IvpFailure(Box::new(error)))?;
        let stats_report = solver.statistics_report();
        let (mesh, trajectory) = solver.get_result();
        match (mesh, trajectory) {
            (Some(mesh), Some(trajectory)) => Ok((mesh, trajectory, stats_report)),
            _ => Err(ShootingBvpError::MissingIvpResult),
        }
    }

    pub fn creating_IVP_closure(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        solver_type: SolverType,
        params: HashMap<String, SolverParam>,
    ) -> Box<dyn Fn(f64, DVector<f64>, f64, f64) -> (DMatrix<f64>, DVector<f64>)> {
        let f: Box<dyn Fn(f64, DVector<f64>, f64, f64) -> (DMatrix<f64>, DVector<f64>) + '_> =
            Box::new(
                move |t0: f64, ivp_initial_condition: DVector<f64>, t_end, step| {
                    info!("\n ivp_initial_condition: {:}", ivp_initial_condition);
                    match Self::IVP_solver(
                        eq_system.clone(),
                        values.clone(),
                        arg.clone(),
                        t0,
                        ivp_initial_condition,
                        t_end,
                        solver_type.clone(),
                        params.clone(),
                        step,
                    ) {
                        Ok((x_vec, y_mat)) => {
                            let y_mat = y_mat.transpose();
                            (y_mat, x_vec)
                        }
                        Err(e) => {
                            error!("IVP solver error: {e}");
                            panic!("IVP solver failed: {}", e);
                        }
                    }
                },
            );
        f
    }

    pub fn creating_IVP_closure_generic<F>(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        solver_type: SolverType,
        configure: F,
    ) -> Box<dyn Fn(f64, DVector<f64>, f64, f64) -> (DMatrix<f64>, DVector<f64>)>
    where
        F: Fn(UniversalODESolver) -> UniversalODESolver + Clone + 'static,
    {
        Box::new(
            move |t0: f64, ivp_initial_condition: DVector<f64>, t_end, _step| {
                info!("\n ivp_initial_condition: {:}", ivp_initial_condition);
                match Self::IVP_solver_generic(
                    eq_system.clone(),
                    values.clone(),
                    arg.clone(),
                    t0,
                    ivp_initial_condition,
                    t_end,
                    solver_type.clone(),
                    configure.clone(),
                ) {
                    Ok((x_vec, y_mat, stats_report)) => {
                        if let Some(stats_report) = stats_report {
                            debug!("IVP stats: {}", stats_report);
                        }
                        (y_mat.transpose(), x_vec)
                    }
                    Err(e) => {
                        error!("IVP solver error: {e}");
                        panic!("IVP solver failed: {}", e);
                    }
                }
            },
        )
    }
}

#[cfg(test)]
mod logging_tests {
    use super::BVPShooting;
    use crate::numerical::Nonlinear_systems::symbolic::SymbolicLambdifyFrontend;
    use crate::numerical::ODE_api2::{SolverParam, SolverType};
    use crate::numerical::ShootingBVP::Shooting_simple::{ShootingBvpError, ShootingMethodResult};
    use crate::numerical::ShootingBVP::symbolic::ShootingSymbolicOptions;
    use crate::numerical::SolverTelemetryMode;
    use crate::numerical::logging::SolverLogLevel;
    use crate::symbolic::symbolic_engine::Expr;
    use nalgebra::{DMatrix, DVector};
    use std::collections::HashMap;

    #[test]
    fn symbolic_shooting_logging_is_off_by_default_and_configurable() {
        let solver = BVPShooting::new(
            Vec::new(),
            Vec::new(),
            "t".to_string(),
            HashMap::new(),
            (0.0, 1.0),
        );
        assert_eq!(solver.logging_level(), SolverLogLevel::Off);

        let configured = solver.with_logging_level(SolverLogLevel::Info);
        assert_eq!(configured.logging_level(), SolverLogLevel::Info);
    }

    #[test]
    fn typed_solve_failure_clears_previous_result() {
        let mut solver = BVPShooting::new(
            Vec::new(),
            Vec::new(),
            "t".to_string(),
            HashMap::new(),
            (0.0, 1.0),
        );
        solver.full_sol = ShootingMethodResult {
            x_mesh: DVector::from_vec(vec![0.0, 1.0]),
            y: DMatrix::from_element(2, 2, 7.0),
            s: 7.0,
            bound_values: DVector::from_element(2, 7.0),
        };
        solver.solution = solver.full_sol.y.clone();

        let result = solver.try_simple_solve(0.0, 0.0, 10, 0.1, SolverTelemetryMode::Off);
        assert!(matches!(
            result,
            Err(ShootingBvpError::InvalidConfiguration {
                field: "tolerance",
                ..
            })
        ));
        assert_eq!(solver.full_sol.x_mesh.len(), 0);
        assert_eq!(solver.solution.ncols(), 0);
    }

    #[test]
    fn typed_native_ivp_route_returns_solution_and_stage_telemetry() {
        let mut boundary_conditions = HashMap::new();
        boundary_conditions.insert("y0".to_string(), vec![(0, 0.0), (1, 1.0)]);
        let mut solver = BVPShooting::new(
            vec![Expr::parse_expression("y1"), Expr::parse_expression("0")],
            vec!["y0".to_string(), "y1".to_string()],
            "x".to_string(),
            boundary_conditions,
            (0.0, 1.0),
        );
        let params = HashMap::from([
            ("rtol".to_string(), SolverParam::Float(1e-6)),
            ("atol".to_string(), SolverParam::Float(1e-8)),
            ("max_step".to_string(), SolverParam::Float(0.1)),
        ]);

        let report = solver
            .try_solve_with_certain_ivp(
                0.5,
                1e-8,
                20,
                0.1,
                SolverType::Bdf,
                params,
                SolverTelemetryMode::Detailed,
            )
            .expect("typed BDF shooting route should converge");

        assert_eq!(report.solution.y.nrows(), 2);
        assert!((report.solution.bound_values[0] - 1.0).abs() < 1e-5);
        let telemetry = report.telemetry.expect("telemetry was requested");
        assert!(telemetry.counters.boundary_residual_evaluations >= 2);
        assert_eq!(
            telemetry.counters.ivp_solves,
            telemetry.counters.boundary_residual_evaluations + 1
        );
        assert!(telemetry.timings.total_time >= telemetry.timings.final_ivp_time);
    }

    #[test]
    fn symbolic_lambdify_frontends_solve_the_same_bvp() {
        for frontend in [
            SymbolicLambdifyFrontend::ExprLegacy,
            SymbolicLambdifyFrontend::AtomViewNative,
        ] {
            let mut boundary_conditions = HashMap::new();
            boundary_conditions.insert("y0".to_string(), vec![(0, 0.0), (1, 1.0)]);
            let mut solver = BVPShooting::new(
                vec![Expr::parse_expression("y1"), Expr::parse_expression("0")],
                vec!["y0".to_string(), "y1".to_string()],
                "x".to_string(),
                boundary_conditions,
                (0.0, 1.0),
            );
            let params = HashMap::from([
                ("rtol".to_string(), SolverParam::Float(1e-6)),
                ("atol".to_string(), SolverParam::Float(1e-8)),
                ("max_step".to_string(), SolverParam::Float(0.1)),
            ]);

            let report = solver
                .try_solve_symbolic(
                    0.5,
                    1e-8,
                    20,
                    0.1,
                    SolverType::Bdf,
                    params,
                    ShootingSymbolicOptions {
                        frontend,
                        ..ShootingSymbolicOptions::default()
                    },
                    SolverTelemetryMode::Detailed,
                )
                .expect("symbolic shooting route should converge");

            assert_eq!(report.frontend, frontend);
            assert!(report.preparation.preparation_duration > std::time::Duration::ZERO);
            assert!(report.solve.telemetry.is_some());
            assert!((report.solve.solution.bound_values[0] - 1.0).abs() < 1e-5);
        }
    }
}
