use core::fmt::Display;

/// Radau IIA method for solving systems of ordinary differential equations
/// This is an implicit Runge-Kutta method with high order accuracy
/// Newton-Raphson calculation on each step is made using analytic jacobian
use crate::numerical::Radau::Radau_newton::RadauNewton;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_vectors::ExprVector;
use crate::{Utils::plots::plots, symbolic::symbolic_vectors::ExprMatrix};
use log::{info, warn};
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use simplelog::*;
use std::collections::HashMap;
use std::fs::File;
use std::time::Instant;

/// Radau IIA method orders and corresponding coefficients
#[derive(Debug, Clone)]
pub enum RadauOrder {
    Order3, // 2-stage, order 3
    Order5, // 3-stage, order 5
    Order7, // 4-stage, order 7
}

/// Butcher tableau coefficients for Radau IIA methods
#[derive(Debug, Clone)]
pub struct RadauCoefficients {
    pub c: DVector<f64>, // nodes
    pub a: DMatrix<f64>, // Runge-Kutta matrix
    pub b: DVector<f64>, // weights
    pub stages: usize,   // number of stages
}

impl RadauCoefficients {
    pub fn new(order: RadauOrder) -> Self {
        match order {
            RadauOrder::Order3 => {
                // 2-stage Radau IIA (order 3)
                let c = DVector::from_vec(vec![1.0 / 3.0, 1.0]);
                let a =
                    DMatrix::from_row_slice(2, 2, &[5.0 / 12.0, -1.0 / 12.0, 3.0 / 4.0, 1.0 / 4.0]);
                let b = DVector::from_vec(vec![3.0 / 4.0, 1.0 / 4.0]);
                RadauCoefficients { c, a, b, stages: 2 }
            }
            RadauOrder::Order5 => {
                // 3-stage Radau IIA (order 5)
                let c = DVector::from_vec(vec![
                    (4.0 - 6_f64.sqrt()) / 10.0,
                    (4.0 + 6_f64.sqrt()) / 10.0,
                    1.0,
                ]);
                let a = DMatrix::from_row_slice(
                    3,
                    3,
                    &[
                        (88.0 - 7.0 * 6_f64.sqrt()) / 360.0,
                        (296.0 - 169.0 * 6_f64.sqrt()) / 1800.0,
                        (-2.0 + 3.0 * 6_f64.sqrt()) / 225.0,
                        (296.0 + 169.0 * 6_f64.sqrt()) / 1800.0,
                        (88.0 + 7.0 * 6_f64.sqrt()) / 360.0,
                        (-2.0 - 3.0 * 6_f64.sqrt()) / 225.0,
                        (16.0 - 6_f64.sqrt()) / 36.0,
                        (16.0 + 6_f64.sqrt()) / 36.0,
                        1.0 / 9.0,
                    ],
                );
                let b = DVector::from_vec(vec![
                    (16.0 - 6_f64.sqrt()) / 36.0,
                    (16.0 + 6_f64.sqrt()) / 36.0,
                    1.0 / 9.0,
                ]);
                RadauCoefficients { c, a, b, stages: 3 }
            }
            RadauOrder::Order7 => {
                // 4-stage Radau IIA (order 7) - coefficients would be more complex
                // For now, fallback to order 5
                Self::new(RadauOrder::Order5)
            }
        }
    }
}

//#[derive(Debug)]
pub struct Radau {
    pub newton: RadauNewton,
    pub coefficients: RadauCoefficients,
    pub order: RadauOrder,

    // Initial conditions and bounds
    pub y0: DVector<f64>,
    pub t0: f64,
    pub t_bound: f64,

    // Current state
    pub t: f64,
    pub y: DVector<f64>,
    pub t_old: Option<f64>,

    // Results storage
    pub t_result: DVector<f64>,
    pub y_result: DMatrix<f64>,

    // Status and control
    pub status: String,
    pub message: Option<String>,
    pub h: Option<f64>,
    pub global_timestepping: bool,

    pub K_matrix: ExprMatrix,
    pub y_vector: ExprVector,
    // Radau-specific working arrays
    pub k_stages: DMatrix<f64>, // Stage derivatives k_i
    pub y_stages: DMatrix<f64>, // Stage values Y_i
    values: Vec<String>,
    pub parallel: bool,
    // Add logging configuration
    pub log_level: Option<LevelFilter>,
    pub log_to_file: Option<String>,
    pub log_to_console: bool,

    // Stop condition
    pub stop_condition: Option<HashMap<String, f64>>,
    pub tolerance: f64,
}

impl Display for Radau {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Radau {{ order: {:?}, t0: {}, t_bound: {}, t: {}, y: {:?} }}",
            self.order, self.t0, self.t_bound, self.t, self.y
        )
    }
}

impl Radau {
    pub fn new(order: RadauOrder) -> Radau {
        let coefficients = RadauCoefficients::new(order.clone());
        let stages = coefficients.stages;

        // Create empty RadauNewton solver
        let nr_new = RadauNewton::new(
            Vec::new(),
            Vec::new(),
            None,
            Vec::new(),
            "".to_string(),
            1e-6,
            50,
            0,
            0,
        );

        Radau {
            newton: nr_new,
            coefficients,
            order,
            y0: DVector::zeros(0),
            t0: 0.0,
            t_bound: 0.0,
            t: 0.0,
            y: DVector::zeros(0),
            t_old: None,
            t_result: DVector::zeros(0),
            y_result: DMatrix::zeros(0, 0),
            status: "running".to_string(),
            message: None,
            h: None,
            global_timestepping: true,
            K_matrix: ExprMatrix::zeros(0, 0),
            y_vector: ExprVector::zeros(0),
            k_stages: DMatrix::zeros(0, stages),
            y_stages: DMatrix::zeros(0, stages),
            values: Vec::new(),
            parallel: false,
            log_level: Some(LevelFilter::Warn),
            log_to_file: None,
            log_to_console: true,
            stop_condition: None,
            tolerance: 1e-6,
        }
    }

    pub fn set_initial(
        &mut self,
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        tolerance: f64,
        max_iterations: usize,
        h: Option<f64>,
        t0: f64,
        t_bound: f64,
        y0: DVector<f64>,
    ) -> () {
        self.tolerance = tolerance;
        // Initialize logging first
        self.init_logger();
        let n_vars = y0.len();
        let n_stages = self.coefficients.stages;

        // Initialize working arrays
        self.k_stages = DMatrix::zeros(n_vars, n_stages);
        self.y_stages = DMatrix::zeros(n_vars, n_stages);
        info!(
            "initialized stage variables with shape: {:?}",
            self.k_stages.shape()
        );
        self.values = values.clone();

        //  info!("Radau system: {:?}", radau_system);
        // Create variable names for stage derivatives K_{i,j} values. Total number of variables is n_vars * n_stages
        // K {rows = n_vars, columns = n_stages}
        let K_matrix = ExprMatrix::indexed_vars_matrix(n_vars, n_stages, "K");
        self.K_matrix = K_matrix.clone();
        // for n_stages = 3, n_vars = 2 => K_parameters_str =[ "K_0_0", "K_0_1", "K_0_2", "K_1_0", "K_1_1", "K_1_2"]
        let K_parameters_str = K_matrix.to_strings();
        let radau_variables = K_parameters_str.clone();

        // Create variable names for y_n values
        // Create parameter names for y_n values and h
        let mut parameters = Vec::new();
        let y_vector = ExprVector::indexed_vars_vector(n_vars, "y_n");
        self.y_vector = y_vector.clone();
        info!(" \n created vector of unknowns: {:?}", y_vector);
        info!("created matrix of stage variables: {:?}", K_matrix);
        info!("list of stage variables: {:?}", K_parameters_str.clone());
        let y_parameters_str = y_vector.to_strings();
        parameters.extend(y_parameters_str);
        parameters.push("h".to_string());

        // Create the nonlinear system for Radau method
        let radau_system = self.construct_radau_system(eq_system, values.clone(), arg.clone());
        // Create Newton solver to fint the stage values
        let nr = RadauNewton::new(
            radau_system,
            radau_variables,
            None,
            parameters,
            arg,
            tolerance,
            max_iterations,
            n_vars,
            n_stages,
        );

        self.newton = nr;
        self.t0 = t0;
        self.t_bound = t_bound;
        self.y0 = y0.clone();
        self.t = t0;
        self.y = y0.clone();

        // Set global timestepping
        if let Some(dt) = h {
            self.global_timestepping = true;
            self.h = Some(dt);
        } else {
            info!("global_timestepping = false");
            self.global_timestepping = false;
            self.h = Some(1e-4); // Default step size
        }

        self.check();
    }

    pub fn set_stop_condition(&mut self, stop_condition: HashMap<String, f64>) {
        self.stop_condition = Some(stop_condition);
    }

    fn check_stop_condition(&self, y: &DVector<f64>) -> bool {
        if let Some(ref conditions) = self.stop_condition {
            for (var_name, target_value) in conditions {
                if let Some(var_index) = self.values.iter().position(|v| v == var_name) {
                    let current_value = y[var_index];
                    if (current_value - target_value).abs() <= self.tolerance {
                        return true;
                    }
                }
            }
        }
        false
    }

    #[allow(dead_code)]
    pub fn set_stage_variables(&mut self, K_matrix: ExprMatrix, y_vector: ExprVector) {
        self.K_matrix = K_matrix;
        self.y_vector = y_vector;
    }
    /// Enable or disable parallel processing
    pub fn set_parallel(&mut self, parallel: bool) {
        self.parallel = parallel;
        self.newton.set_parallel(parallel);
        info!(
            "Radau parallel processing: {}",
            if parallel { "enabled" } else { "disabled" }
        );
    }
    /// Construct the nonlinear system for Radau method
    /// For s stages, we need to solve: K_i - f(t_n + c_i*h, y_n + h*sum(a_ij*K_j)) = 0
    /// where K_i are the stage derivatives we're solving for
    //
    pub fn construct_radau_system(
        &self,
        original_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
    ) -> Vec<Expr> {
        let n_vars = original_system.len();
        let n_stages = self.coefficients.stages;
        let mut radau_equations: ExprMatrix = ExprMatrix::zeros(n_vars, n_stages);
        // radau_equations.show();
        info!(
            "Constructing Radau nonlinear system with {} stages and {} variables",
            n_stages, n_vars
        );
        let h_var = Expr::Var("h".to_string());
        let t_var = Expr::Var(arg.clone());

        let K_matrix = self.K_matrix.clone();
        let y_vector = self.y_vector.clone();
        let y_vector_str = y_vector.to_strings();

        let original_system = ExprVector::new(original_system);

        let f_expr: &dyn Fn(Expr, &ExprVector) -> ExprVector = &|t: Expr, y_i: &ExprVector| {
            let f_expr = original_system.clone();
            let mut f_expr = f_expr.substitute(&arg, &t);

            for i in 0..n_vars {
                info!("variable {} replaced by  {}", y_vector_str[i], &y_i[i]);
                f_expr = f_expr.substitute(&values[i], &y_i[i]);
            }
            info!("RHS for Newton equations:");
            f_expr.show();
            f_expr
        };
        // For each stage i and each variable j, create equation:
        // K_{i,j} - f_j(t_n + c_i*h, y_n + h*sum_{k=1}^s(a_{i,k}*K_{k,j})) = 0

        for stage_i in 0..n_stages {
            let mut y_stage = y_vector.clone();
            // Compute y_i = y_n + h * sum_j a_ij * k_j
            for stage_k in 0..n_stages {
                let a_ik = Expr::Const(self.coefficients.a[(stage_i, stage_k)]);
                let k_j = K_matrix.column(stage_k);

                assert_eq!(k_j.len(), y_stage.len(), "Vector dimensions must match");
                // axpy operation y_stage = y_stage + a_ik * k_j
                y_stage.axpy(&(h_var.clone() * a_ik.clone()), &k_j, &Expr::Const(1.0));
            }

            for y_i in y_stage.clone().iter() {
                info!("\n y to substitute into Radau stage equation {}", y_i);
            }
            // Create the argument for f: t_n + c_i*h
            let c_i = Expr::Const(self.coefficients.c[stage_i]);
            let t_stage = t_var.clone() + c_i * h_var.clone();
            info!("\n t for rhe Radau atage {}", &t_stage);
            let f_expr = f_expr(t_stage, &y_stage);

            let k_i = K_matrix.column(stage_i);
            let radau_eq = k_i - f_expr;
            radau_equations.set_column(stage_i, &radau_eq);
        }
        info!("\n Radau equations:");

        let radau_equations = radau_equations.flatten().as_vec();
        for (i, radau_equation) in radau_equations.clone().iter().enumerate() {
            info!("Radau equation {}: {}", i, radau_equation.clone());
        }
        info!("Created {} Radau equations", radau_equations.len());
        radau_equations
    }

    pub fn check(&self) -> () {
        assert_eq!(!self.y.is_empty(), true, "initial y is empty");
        assert_eq!(!self.newton.eq_system.is_empty(), true, "system is empty");
        assert_eq!(
            !self.newton.k_variables.is_empty(),
            true,
            "K variables are empty"
        );
        assert_eq!(
            !self.newton.parameters.is_empty(),
            true,
            "parameters are empty"
        );
        assert_eq!(!self.newton.arg.is_empty(), true, "arg is empty");
        assert_eq!(self.newton.tolerance >= 0.0, true, "tolerance is invalid");
        assert_eq!(
            self.newton.max_iterations >= 1,
            true,
            "max_iterations is invalid"
        );
        assert_eq!(
            self.coefficients.stages >= 2,
            true,
            "Radau method needs at least 2 stages"
        );
    }

    pub fn _step_impl(&mut self) -> (bool, Option<String>) {
        let n_vars = self.y.len();
        let n_stages = self.coefficients.stages;

        // Calculate step size
        let dt = if self.global_timestepping {
            self.h.unwrap()
        } else {
            // Adaptive step size logic could go here
            (self.t_bound - self.t).min(self.h.unwrap())
        };

        // Set up Newton solver for this step
        let t_new = self.t + dt;
        self.newton.set_parameters(self.y.clone(), dt, t_new);

        // Set initial guess for stage derivatives
        let mut initial_guess = DVector::zeros(n_vars * n_stages);
        // use previous solution or simple estimate

        for stage_i in 0..n_stages {
            for var_j in 0..n_vars {
                let idx = stage_i * n_vars + var_j;
                // Simple initial guess - could be improved
                initial_guess[idx] = self.k_stages[(var_j, stage_i)];
            }
        }

        self.newton.set_initial_guess(initial_guess);

        info!("Solving Radau system at t = {}, h = {}", t_new, dt);

        // Solve the nonlinear system for all stage derivatives
        let result = self.newton.solve();

        if result.is_none() {
            warn!("Radau Newton solver failed to converge");
            return (
                false,
                Some("maximum number of iterations reached in Radau solver".to_string()),
            );
        }

        let k_solution = result.unwrap();
        assert_eq!(
            k_solution.len(),
            n_vars * n_stages,
            "Solution vector has wrong size, expected {} elements",
            n_vars * n_stages
        );

        // Extract stage derivatives from solution vector and reconstruct matrix
        // for n_stages = 3, n_vars = 2 => k_solution = [K_0_0, K_0_1, K_0_2, K_1_0, K_1_1, K_1_2]
        // first index is variable, second is stage
        info!("Radau Newton solver converged {:?}", k_solution.as_slice());
        // Extract stage derivatives from solution vector
        self.k_stages = DMatrix::zeros(n_vars, n_stages);
        for var_j in 0..n_vars {
            for stage_i in 0..n_stages {
                let idx = var_j * n_stages + stage_i; // Row-major indexing
                self.k_stages[(var_j, stage_i)] = k_solution[idx];
            }
        }

        // Compute stage values Y_i = y_n + h * sum(a_{i,j} * K_j)
        for stage_i in 0..n_stages {
            for var_j in 0..n_vars {
                let mut y_stage = self.y[var_j];
                for stage_k in 0..n_stages {
                    let a_ik = self.coefficients.a[(stage_i, stage_k)];
                    let k_k = self.k_stages[(var_j, stage_k)];
                    y_stage += dt * a_ik * k_k;
                }
                self.y_stages[(var_j, stage_i)] = y_stage;
            }
        }

        // Compute final solution: y_{n+1} = y_n + h * sum(b_i * K_i)
        let mut y_new = self.y.clone();
        assert!(self.k_stages.ncols() == n_stages);
        assert!(
            self.k_stages.nrows() == y_new.len(),
            "y_new.len() != k_stages.nrows()"
        );
        for i in 0..n_stages {
            let k_i = self.k_stages.column(i);
            let b_i = self.coefficients.b[i];
            y_new.axpy(dt * b_i, &k_i, 1.0); // y_new += h * b_i * k_i
        }
        /*
        the same as:
        let mut y_new = self.y.clone();
        for var_j in 0..n_vars {
            let mut increment = 0.0;
            for stage_i in 0..n_stages {
                let b_i = self.coefficients.b[stage_i];
                let k_i = self.k_stages[(var_j, stage_i)];
                increment += b_i * k_i;
            }
            y_new[var_j] += dt * increment;
        }
        */
        info!("y new = {}", y_new);
        // Update solution
        self.y = y_new;
        self.t = t_new;

        info!(
            "Radau step completed successfully: t = {}, y = {:?}",
            self.t, self.y
        );

        (true, None)
    }
    #[allow(dead_code)]
    fn update_y(&mut self, y_new: &mut DVector<f64>, dt: f64, n_stages: usize, n_vars: usize) {
        let parallel_flag = self.parallel;
        if parallel_flag {
            // Parallel computation of the weighted sum
            let increments: Vec<f64> = (0..n_vars)
                .into_par_iter()
                .map(|var_j| {
                    (0..n_stages)
                        .map(|stage_i| {
                            let b_i = self.coefficients.b[stage_i];
                            let k_i = self.k_stages[(var_j, stage_i)];
                            dt * b_i * k_i
                        })
                        .sum()
                })
                .collect();

            // Apply increments
            for (var_j, increment) in increments.iter().enumerate() {
                y_new[var_j] += increment;
            }
        } else {
            for i in 0..n_stages {
                let k_i = self.k_stages.column(i);
                let b_i = self.coefficients.b[i];
                y_new.axpy(dt * b_i, &k_i, 1.0); // y_new += h * b_i * k_i
            }
        }
    }
    #[allow(dead_code)]
    fn update_Y_stages(&mut self, dt: f64, n_stages: usize, n_vars: usize) {
        // Parallel computation of stage values Y_i = y_n + h * sum(a_{i,j} * K_j)
        if self.parallel && n_stages > 1 {
            let stage_values: Vec<Vec<f64>> = (0..n_stages)
                .into_par_iter()
                .map(|stage_i| {
                    (0..n_vars)
                        .map(|var_j| {
                            let mut y_stage = self.y[var_j];
                            for stage_k in 0..n_stages {
                                let a_ik = self.coefficients.a[(stage_i, stage_k)];
                                let k_k = self.k_stages[(var_j, stage_k)];
                                y_stage += dt * a_ik * k_k;
                            }
                            y_stage
                        })
                        .collect()
                })
                .collect();

            // Fill y_stages matrix
            for (stage_i, stage_values) in stage_values.iter().enumerate() {
                for (var_j, &value) in stage_values.iter().enumerate() {
                    self.y_stages[(var_j, stage_i)] = value;
                }
            }
        } else {
            // Sequential computation
            for stage_i in 0..n_stages {
                for var_j in 0..n_vars {
                    let mut y_stage = self.y[var_j];
                    for stage_k in 0..n_stages {
                        let a_ik = self.coefficients.a[(stage_i, stage_k)];
                        let k_k = self.k_stages[(var_j, stage_k)];
                        y_stage += dt * a_ik * k_k;
                    }
                    self.y_stages[(var_j, stage_i)] = y_stage;
                }
            }
        }
    }
    pub fn step(&mut self) {
        // Similar to BE::step but for Radau
        let t = self.t;
        if t == self.t_bound {
            self.t_old = Some(t);
            self.status = "finished".to_string();
        } else {
            let (success, message_) = self._step_impl();

            if let Some(message_str) = message_ {
                self.message = Some(message_str.to_string());
            } else {
                self.message = None;
            }

            if success == false {
                self.status = "failed".to_string();
            } else {
                self.t_old = Some(t);
                if (self.t - self.t_bound) >= 0.0 {
                    self.status = "finished".to_string();
                }
            }
        }
    }

    pub fn main_loop(&mut self) -> () {
        let start = Instant::now();
        info!("Starting Radau main loop with {:?} order", self.order);

        // Analogue of https://github.com/scipy/scipy/blob/main/scipy/integrate/_ivp/ivp.py
        let mut integr_status: Option<i8> = None;
        let mut y: Vec<DVector<f64>> = Vec::new();
        let mut t: Vec<f64> = Vec::new();
        let mut iteration_count: i64 = 0;

        // Store initial conditions
        t.push(self.t);
        y.push(self.y.clone());

        while integr_status.is_none() {
            self.step();
            iteration_count += 1;

            info!(
                "Radau iteration: {}, t: {:.6}, status: {}",
                iteration_count, self.t, self.status
            );

            // Check termination conditions
            if self.status == "finished".to_string() {
                integr_status = Some(0);
                info!(
                    "Radau integration finished successfully at t = {:.6}",
                    self.t
                );
            } else if self.status == "failed".to_string() {
                integr_status = Some(-1);
                if let Some(ref msg) = self.message {
                    info!("Radau integration failed: {}", msg);
                } else {
                    info!("Radau integration failed with unknown error");
                }
                break;
            }

            // Check stop condition before storing solution
            if self.check_stop_condition(&self.y) {
                self.status = "stopped_by_condition".to_string();
                integr_status = Some(0);
                info!(
                    "Radau integration stopped by condition at t = {:.6}",
                    self.t
                );
            }

            // Store current solution
            t.push(self.t);
            y.push(self.y.clone());

            // Safety check to prevent infinite loops
            if iteration_count > 1000000 {
                info!("Radau integration stopped: maximum iterations exceeded");
                self.status = "failed".to_string();
                self.message = Some("Maximum iteration count exceeded".to_string());
                integr_status = Some(-1);
                break;
            }

            // Progress logging for long integrations
            if iteration_count % 1000 == 0 {
                let progress = (self.t - self.t0) / (self.t_bound - self.t0) * 100.0;
                info!(
                    "Radau progress: {:.1}% (t = {:.6}/{:.6})",
                    progress, self.t, self.t_bound
                );
            }
        }

        // Process results
        let rows = y.len();
        let cols = if rows > 0 { y[0].len() } else { 0 };

        if rows == 0 || cols == 0 {
            info!("Warning: No solution points collected");
            self.t_result = DVector::zeros(0);
            self.y_result = DMatrix::zeros(0, 0);
            return;
        }

        // Convert solution vectors to matrices
        let mut flat_vec: Vec<f64> = Vec::new();
        for vector in y.iter() {
            flat_vec.extend(vector.iter());
        }

        // Create result matrices: y_result[variable, time_point]
        let y_res: DMatrix<f64> = DMatrix::from_vec(cols, rows, flat_vec).transpose();
        let t_res = DVector::from_vec(t);

        // Log integration statistics
        let duration = start.elapsed();
        info!("Radau integration completed:");
        info!("  - Method: {:?}", self.order);
        info!("  - Time span: [{:.6}, {:.6}]", self.t0, self.t_bound);
        info!("  - Total steps: {}", iteration_count);
        info!("  - Solution points: {}", rows);
        info!("  - Variables: {}", cols);
        info!("  - Final time: {:.6}", self.t);
        info!(
            "  - Integration time: {:.3} seconds",
            duration.as_secs_f64()
        );
        info!(
            "  - Average time per step: {:.6} ms",
            duration.as_millis() as f64 / iteration_count as f64
        );

        // Calculate and log step size statistics if using global timestepping
        if self.global_timestepping && rows > 1 {
            let step_sizes: Vec<f64> = t_res
                .iter()
                .zip(t_res.iter().skip(1))
                .map(|(t1, t2)| t2 - t1)
                .collect();

            if !step_sizes.is_empty() {
                let avg_step = step_sizes.iter().sum::<f64>() / step_sizes.len() as f64;
                let min_step = step_sizes.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max_step = step_sizes.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                info!("  - Step size statistics:");
                info!("    * Average: {:.6}", avg_step);
                info!("    * Minimum: {:.6}", min_step);
                info!("    * Maximum: {:.6}", max_step);
            }
        }

        // Store results
        self.t_result = t_res;
        self.y_result = y_res;

        // Final status check
        match integr_status {
            Some(0) => info!("Radau integration successful"),
            Some(-1) => info!("Radau integration failed"),
            _ => info!("Radau integration status unknown"),
        }
    }

    pub fn solve(&mut self) -> () {
        info!("Initializing Radau solver with {:?} order", self.order);

        // Generate equations and jacobian for Newton solver
        self.newton.eq_generate();

        // Run the main integration loop
        self.main_loop();

        // Final status report
        match self.status.as_str() {
            "finished" => info!("Radau solver completed successfully"),
            "failed" => {
                if let Some(ref msg) = self.message {
                    info!("Radau solver failed: {}", msg);
                } else {
                    info!("Radau solver failed with unknown error");
                }
            }
            _ => info!("Radau solver finished with status: {}", self.status),
        }
    }

    pub fn get_result(&self) -> (Option<DVector<f64>>, Option<DMatrix<f64>>) {
        (Some(self.t_result.clone()), Some(self.y_result.clone()))
    }

    pub fn get_status(&self) -> &String {
        &self.status
    }

    pub fn plot_result(&self) -> () {
        plots(
            self.newton.arg.clone(),
            self.values.clone(),
            self.t_result.clone(),
            self.y_result.clone(),
        );
        info!("Radau result plotted");
    }

    ////////////////////////////////logging functions
    /// Set logging level (Off, Error, Warn, Info, Debug, Trace)
    pub fn set_log_level(&mut self, level: LevelFilter) {
        self.log_level = Some(level);
        self.init_logger();
    }

    /// Enable logging to file
    pub fn set_log_file(&mut self, filename: String) {
        self.log_to_file = Some(filename);
        self.init_logger();
    }

    /// Enable/disable console logging
    pub fn set_console_logging(&mut self, enabled: bool) {
        self.log_to_console = enabled;
        self.init_logger();
    }

    /// Initialize the logger based on current settings
    fn init_logger(&self) {
        let level = self.log_level.unwrap_or(LevelFilter::Info);

        let mut loggers: Vec<Box<dyn SharedLogger>> = Vec::new();

        // Console logger
        if self.log_to_console {
            loggers.push(TermLogger::new(
                level,
                Config::default(),
                TerminalMode::Mixed,
                ColorChoice::Auto,
            ));
        }

        // File logger
        if let Some(ref filename) = self.log_to_file {
            if let Ok(file) = File::create(filename) {
                loggers.push(WriteLogger::new(level, Config::default(), file));
            }
        }

        // Initialize combined logger
        if !loggers.is_empty() {
            let _ = CombinedLogger::init(loggers);
        }
    }

    /// Enable debug logging
    pub fn enable_debug_logging(&mut self) {
        self.set_log_level(LevelFilter::Debug);
    }

    /// Enable info logging (default)
    pub fn enable_info_logging(&mut self) {
        self.set_log_level(LevelFilter::Info);
    }

    /// Disable logging
    pub fn disable_logging(&mut self) {
        self.set_log_level(LevelFilter::Off);
    }

    /// Enable verbose logging with file output
    pub fn enable_verbose_logging(&mut self, log_file: Option<String>) {
        self.set_log_level(LevelFilter::Debug);
        if let Some(filename) = log_file {
            self.set_log_file(filename);
        }
    }
}
