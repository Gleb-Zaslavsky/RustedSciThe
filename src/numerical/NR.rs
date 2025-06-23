use crate::numerical::BVP_Damp::BVP_utils::checkmem;
use crate::numerical::BVP_Damp::BVP_utils::{CustomTimer, elapsed_time};
/// A framework for solving system of nonlinear equations using
/// - Newton-Raphson method;
/// - damped Newton-Raphson method;
/// - trust region method;
/// - Levenberg-Marquardt method;
///
///  Example#1
/// ```
///  use RustedSciThe::numerical::NR::NR;
/// use nalgebra::DVector;
/// //use the shortest way to solve system of equations
///    // first define system of equations and initial guess
///    let mut NR_instanse = NR::new();
///
///    let vec_of_expressions = vec![ "x^2+y^2-10".to_string(), "x-y-4".to_string()];
///   let initial_guess = vec![1.0, 1.0];
///    // solve
///    NR_instanse.eq_generate_from_str(vec_of_expressions, None, initial_guess, 1e-6, 100);
///    NR_instanse.main_loop();
///    println!("result = {:?} \n", NR_instanse.get_result().unwrap());
///  ```
/// Example#2
///  ```
///    // or more verbose way...
///    // first define system of equations
///     use RustedSciThe::numerical::NR::NR;
///     use RustedSciThe::symbolic::symbolic_engine::Expr;
///     use RustedSciThe::symbolic::symbolic_functions::Jacobian;
/// use nalgebra::DVector;
///     let vec_of_expressions = vec!["x^2+y^2-10", "x-y-4"];
///
///     let initial_guess = vec![1.0, 1.0];
///     let mut NR_instanse = NR::new();
///     let vec_of_expr = Expr::parse_vector_expression(vec_of_expressions.clone());
///     let values = vec!["x".to_string(), "y".to_string()];
///     NR_instanse.set_equation_system(vec_of_expr, Some(values.clone()), initial_guess, 1e-6, 100);
///     NR_instanse.eq_generate();
///     NR_instanse.main_loop();
///     let solution = NR_instanse.get_result().unwrap();
///     assert_eq!(solution, DVector::from(vec![-1.0, 3.0] ));
///     println!("result = {:?} \n", NR_instanse.get_result().unwrap());
///  ```
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions::Jacobian;

use crate::numerical::LM_utils::{
    ConvergenceCriteria, ReductionRatio, ScalingMethod, SubproblemMethod, UpdateMethod,
};
use log::{error, info, warn};
use nalgebra::{DMatrix, DVector, Matrix};
use simplelog::LevelFilter;
use simplelog::*;
use std::collections::HashMap;
use std::error::Error;
use std::time::Instant;
use std::vec;
use tabled::{builder::Builder, settings::Style};
#[derive(Debug, Clone)]
pub enum Method {
    simple,
    damped,

    trust_region,
    LM,
    LM_Nielsen,
}
pub struct NR {
    pub jacobian: Jacobian, // instance of Jacobian struct, contains jacobian matrix function and equation functions
    pub eq_system: Vec<Expr>, // vector of equations
    pub values: Vec<String>, // vector of variables
    pub initial_guess: Vec<f64>, // initial guess
    pub method: Method,
    pub Bounds: Option<HashMap<String, (f64, f64)>>,
    pub bounds_vec: Vec<(f64, f64)>,
    pub tolerance: f64,        // tolerance
    pub max_iterations: usize, // max number of iterations

    pub parameters: Option<HashMap<String, f64>>, // parameters
    pub max_error: f64,                           // max error
    pub dumping_factor: f64,
    pub i: usize,                     // iteration counter
    pub jac: DMatrix<f64>,            // jacobian matrix
    pub fun_vector: DVector<f64>,     // vector of functions
    pub y: DVector<f64>,              // current iteration
    pub step: DVector<f64>,           // step
    pub result: Option<DVector<f64>>, // result of the iteration
    pub loglevel: Option<String>,
    pub linear_sys_method: Option<String>, // method for solving linear system
    pub custom_timer: CustomTimer,
    pub calc_statistics: HashMap<String, usize>,
    /// module NR_LM_Nielsen contains the most advanced version of the LM method.
    /// This code needs many parameters to work properly.
    pub scales_vec: DVector<f64>,
    pub scaling_method: Option<ScalingMethod>,
    pub subproblem_method: Option<SubproblemMethod>,
    pub reduction_ratio: Option<ReductionRatio>,
    pub update_method: Option<UpdateMethod>,
    pub convergence_criteria: Option<ConvergenceCriteria>,
    pub f_tolerance: Option<f64>,
    pub g_tolerance: Option<f64>,
}

impl NR {
    pub fn new() -> NR {
        //jacobian: Jacobian, initial_guess: Vec<f64>, tolerance: f64, max_iterations: usize, max_error: f64, result: Option<Vec<f64>>
        NR {
            jacobian: Jacobian::new(),
            eq_system: Vec::new(),
            values: Vec::new(),
            initial_guess: Vec::new(),
            method: Method::simple,
            Bounds: None,
            bounds_vec: Vec::new(),
            tolerance: 1e-6,
            max_iterations: 100,
            parameters: None,
            max_error: 0.0,
            dumping_factor: 1.0,
            i: 0,
            jac: DMatrix::zeros(0, 0),
            fun_vector: DVector::zeros(0),
            y: DVector::zeros(0),
            step: DVector::zeros(0),
            result: None,
            loglevel: Some("info".to_string()),
            linear_sys_method: Some("lu".to_string()),
            custom_timer: CustomTimer::new(),
            calc_statistics: HashMap::new(),
            scales_vec: DVector::zeros(0),
            scaling_method: None,
            subproblem_method: None,
            reduction_ratio: None,
            update_method: None,
            convergence_criteria: None,
            f_tolerance: None,
            g_tolerance: None,
        }
    }
    ////////////////////////////SETTERS///////////////////////////////////////////////////////////////////
    /// Basic methods to set the equation system
    pub fn set_equation_system(
        &mut self,
        eq_system: Vec<Expr>,
        unknowns: Option<Vec<String>>,
        initial_guess: Vec<f64>,
        tolerance: f64,
        max_iterations: usize,
    ) {
        self.eq_system = eq_system.clone();
        self.initial_guess = initial_guess;
        self.tolerance = tolerance;
        self.max_iterations = max_iterations;
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
        assert!(
            tolerance >= 0.0,
            "Tolerance should be a non-negative number."
        );
        assert!(
            max_iterations > 0,
            "Max iterations should be a positive number."
        );
        self.step = DVector::zeros(self.initial_guess.len());
    }

    pub fn eq_generate_from_str(
        &mut self,
        eq_system_string: Vec<String>,
        unknowns: Option<Vec<String>>,
        initial_guess: Vec<f64>,
        tolerance: f64,
        max_iterations: usize,
    ) {
        let eq_system = eq_system_string
            .iter()
            .map(|x| Expr::parse_expression(x))
            .collect::<Vec<Expr>>();
        self.set_equation_system(
            eq_system,
            unknowns,
            initial_guess,
            tolerance,
            max_iterations,
        );
        self.eq_generate();
    }

    /// check if solver parameters are set correctly if not set default values
    pub fn parameters_handle(&mut self, parameters: Option<HashMap<String, f64>>) {
        // set default parameters for each method if not set by user
        macro_rules! merge_parameters {
            ($self:expr, $parameters:expr, $default_parameters:expr) => {
                if let Some(user_defined_parameters) = $parameters {
                    let mut parameters = user_defined_parameters.clone();
                    for (key, value) in $default_parameters.iter() {
                        if !user_defined_parameters.contains_key(key) {
                            parameters.insert(key.clone(), *value);
                        }
                    }
                    $self.parameters = Some(parameters);
                } else {
                    info!(
                        "Setting default parameters for method {:?}",
                        $default_parameters
                    );
                    $self.parameters = Some($default_parameters);
                }
            };
        }
        let method = self.method.clone();
        match method {
            Method::simple => {
                // this method does not have parameters
            }
            Method::damped => {
                let default_parameters = HashMap::from([("maxDampIter".to_string(), 50.0)]);
                merge_parameters!(self, parameters, default_parameters);
            } // clipping
            Method::trust_region => {
                let default_parameters: HashMap<String, f64> = HashMap::from([
                    ("eta_min".to_string(), 0.25),       // C5 in original paper
                    ("eta_max".to_string(), 8.0),        // C6 in original paper
                    ("ro_threshold0".to_string(), 0.25), // C2 in original paper
                    ("ro_threshold1".to_string(), 0.75), //C4 in original paper
                    ("C0".to_string(), 1e-4),
                    ("M".to_string(), 0.1 * 10.0 * 8.0),
                    ("d".to_string(), 0.8),  // little delta in original paper
                    ("mu".to_string(), 0.1), // mu0 in original paper
                    ("m".to_string(), 1e-6), // m in original paper
                ]);
                merge_parameters!(self, parameters, default_parameters);
            }
            Method::LM => {
                let default_parameters = HashMap::from([
                    ("diag".to_string(), 1.0),
                    ("increase_factor".to_string(), 3.0),
                    ("decrease_factor".to_string(), 10.0),
                    ("max_lambda".to_string(), 1000.0),
                    ("min_lambda".to_string(), 1e-6),
                ]);
                merge_parameters!(self, parameters, default_parameters);
            }
            Method::LM_Nielsen => {
                let default_parameters = HashMap::from([
                    ("tau".to_string(), 1e-6),
                    ("nu".to_string(), 2.0),
                    ("factor_up".to_string(), 3.0),
                    ("factor_down".to_string(), 2.0),
                    ("rho_threshold".to_string(), 1e-4),
                ]);
                self.scaling_method = Some(ScalingMethod::Marquardt);
                self.reduction_ratio = Some(ReductionRatio::More);
                self.update_method = Some(UpdateMethod::Nielsen);
                self.subproblem_method = Some(SubproblemMethod::Direct);
                self.convergence_criteria = Some(ConvergenceCriteria::SimpleScaled);
                self.f_tolerance = Some(1e-3);
                self.g_tolerance = Some(1e-3);
                merge_parameters!(self, parameters, default_parameters);
            }
            _ => {
                panic!("Method not implemented")
            }
        }
    }
    pub fn set_solver_params(
        &mut self,
        loglevel: Option<String>,
        linear_sys_method: Option<String>,
        damping_factor: Option<f64>,
        Bounds: Option<HashMap<String, (f64, f64)>>,
        method: Option<Method>,
        parameters: Option<HashMap<String, f64>>,
    ) {
        self.loglevel = if let Some(level) = loglevel {
            assert!(
                level == "debug"
                    || level == "info"
                    || level == "warn"
                    || level == "error"
                    || level == "off"
                    || level == "none",
                "loglevel must be none/off, debug/info, warn or error"
            );
            Some(level.to_string())
        } else {
            self.loglevel.clone()
        };
        self.linear_sys_method = if let Some(method) = linear_sys_method {
            let method = method.to_lowercase();
            assert!(
                method == "lu" || method == "inv",
                "linear_sys_method must be lu or inv"
            );

            Some(method.to_string())
        } else {
            self.linear_sys_method.clone()
        };
        self.dumping_factor = if let Some(dumping_factor) = damping_factor {
            assert!(
                dumping_factor >= 0.0 && dumping_factor <= 1.0,
                "Dumping factor should be between 0.0 and 1.0."
            );
            dumping_factor
        } else {
            self.dumping_factor
        };
        if Bounds.is_some() {
            if_initial_guess_inside_bounds(
                &DVector::from_vec(self.initial_guess.clone()),
                &Bounds.clone(),
                &self.values.clone(),
            );
            self.Bounds = Bounds;
            let mut vec_bounds = Vec::new();
            for values in &self.values {
                let (lower, upper) = self.Bounds.as_ref().unwrap().get(values).unwrap();
                vec_bounds.push((*lower, *upper));
            }
            self.bounds_vec = vec_bounds;
        } // if Nounds

        match method {
            Some(method) => self.method = method,
            None => self.method = Method::simple,
        };
        self.parameters_handle(parameters);
    }
    /// module NR_LM_Nielsen contains the most advanced version of the LM method.
    /// This code needs many parameters to work properly.
    fn set_additional_params(
        &mut self,
        // enum to select the scaling method: Levenberg, Marquardt or More (see LM_utils)
        scaling_method: Option<ScalingMethod>,
        // enum to select the reduction ratio: from Nielsen book, or from More paper
        reduction_ratio: Option<ReductionRatio>,

        update_method: Option<UpdateMethod>,
        // enum to select the subproblem method: direct or dogleg (see LM_utils)
        subproblem_method: Option<SubproblemMethod>,

        comvergence_criteria: Option<ConvergenceCriteria>,
        tau: Option<f64>,
        nu: Option<f64>,
        factor_up: Option<f64>,
        factor_down: Option<f64>,
        rho_threshold: Option<f64>,
        f_tolerance: Option<f64>,
        g_tolerance: Option<f64>,
    ) {
        if let Some(scaling_method) = scaling_method {
            self.scaling_method = Some(scaling_method);
        }
        if let Some(reduction_ratio) = reduction_ratio {
            self.reduction_ratio = Some(reduction_ratio);
        }
        if let Some(update_method) = update_method {
            self.update_method = Some(update_method);
        }
        if let Some(subproblem_method) = subproblem_method {
            self.subproblem_method = Some(subproblem_method);
        }
        if let Some(comvergence_criteria) = comvergence_criteria {
            self.convergence_criteria = Some(comvergence_criteria);
        }

        if let Some(params) = self.parameters.as_mut() {
            if let Some(tau) = tau {
                assert!(tau > 0.0, "tau must be positive");
                params.insert("tau".to_string(), tau);
            }
            if let Some(nu) = nu {
                assert!(nu > 0.0, "nu must be positive");
                params.insert("nu".to_string(), nu);
            }
            if let Some(factor_up) = factor_up {
                assert!(factor_up > 0.0, "factor_up must be positive");
                params.insert("factor_up".to_string(), factor_up);
            }
            if let Some(factor_down) = factor_down {
                assert!(factor_down > 0.0, "factor_down must be positive");
                params.insert("factor_down".to_string(), factor_down);
            }
            if let Some(rho_threshold) = rho_threshold {
                assert!(rho_threshold > 0.0, "rho_threshold must be positive");
                params.insert("rho_threshold".to_string(), rho_threshold);
            }
            if let Some(f_tolerance) = f_tolerance {
                assert!(f_tolerance > 0.0, "f_tolerance must be positive");
                params.insert("f_tolerance".to_string(), f_tolerance);
            }
            if let Some(g_tolerance) = g_tolerance {
                assert!(g_tolerance > 0.0, "g_tolerance must be positive");
                params.insert("g_tolerance".to_string(), g_tolerance);
            }
        }
    }

    pub fn implement_weights(&mut self) {
        info!("\n implementing weights!");

        let eq_system = self.eq_system.clone();
        let args = self.values.clone();
        let args: Vec<&str> = args.iter().map(|x| x.as_str()).collect();
        let mut Jacobian_instance_for_scaling = Jacobian::new();
        Jacobian_instance_for_scaling.set_vector_of_functions(eq_system.clone());
        Jacobian_instance_for_scaling.lambdify_funcvector(args);
        let y_data = self.initial_guess.clone();
        Jacobian_instance_for_scaling.evaluate_funvector_lambdified_DVector(y_data);
        let weights = Jacobian_instance_for_scaling.evaluated_functions_DVector;
        let weights = weights.map(|x| if x == 0.0 { 1.0 } else { x.abs() });
        let weights_abs = weights.map(|x| 1.0 / x.abs());
        let weights_abs_vec: Vec<f64> = weights_abs.data.into();
        info!("\n weights_abs_vec: {:#?}", weights_abs_vec);
        println!("\n weights_abs_vec: {:#?}", weights_abs_vec); // .iter().max_by(|a, b| a.partial_cmp(b).unwrap()) 
        let weighted_resuduals: Vec<Expr> = eq_system
            .clone()
            .iter()
            .zip(weights_abs_vec)
            .map(|(eq, weight)| eq.clone() * Expr::Const(weight))
            .collect();
        info!("\n weighted_resuduals: {:?}", weighted_resuduals);
        println!("\n weighted_resuduals: {:?}", weighted_resuduals);
        //self.eq_system = weighted_resuduals;
    }
    ///Set system of equations with vector of symbolic expressions
    pub fn eq_generate(&mut self) {
        let eq_system = self.eq_system.clone();
        let mut Jacobian_instance = Jacobian::new();
        let args = self.values.clone();
        let args: Vec<&str> = args.iter().map(|x| x.as_str()).collect();
        Jacobian_instance.set_vector_of_functions(eq_system);
        Jacobian_instance.set_variables(args.clone());
        Jacobian_instance.calc_jacobian();
        Jacobian_instance.jacobian_generate(args.clone());
        Jacobian_instance.lambdify_funcvector(args);
        assert_eq!(
            Jacobian_instance.vector_of_variables.len(),
            self.initial_guess.len(),
            "Initial guess and vector of variables should have the same length."
        );
        self.jacobian = Jacobian_instance;
    }
    /////////////////////////////////////////////////////////////////////////////////////////////
    //                ITERATIONS
    /////////////////////////////////////////////////////////////////////////////////////////////
    pub fn evaluate_function(&mut self, y: DVector<f64>) -> DVector<f64> {
        let y_data = y.data.into();

        // Evaluate functions
        self.custom_timer.fun_tic();
        self.jacobian.evaluate_funvector_lambdified_DVector(y_data);
        self.custom_timer.fun_tac();

        assert!(
            !self.jacobian.evaluated_functions_DVector.is_empty(),
            "Functions should not be empty."
        );

        // Return cloned values to avoid borrow checker issues

        self.jacobian.evaluated_functions_DVector.clone()
    }
    pub fn evaluate_jacobian(&mut self, y: DVector<f64>) -> DMatrix<f64> {
        let y_data = y.data.into();
        // Evaluate jacobian
        self.custom_timer.jac_tic();
        self.jacobian.evaluate_func_jacobian_DMatrix(y_data);
        self.custom_timer.jac_tac();
        assert!(
            !self.jacobian.evaluated_jacobian_DMatrix.is_empty(),
            "Jacobian should not be empty."
        );
        // Return cloned values to avoid borrow checker issues
        self.jacobian.evaluated_jacobian_DMatrix.clone()
    }
    pub fn step(&mut self, y: DVector<f64>) -> (DVector<f64>, DVector<f64>) {
        let method = self.linear_sys_method.clone().unwrap();
        // let previous_step: DVector<f64> = self.step.clone();
        let Jacobian_instance = &mut self.jacobian;
        // evaluate jacobian and functions
        self.custom_timer.jac_tic();
        Jacobian_instance.evaluate_func_jacobian_DMatrix(y.clone().data.into());
        self.custom_timer.jac_tac();
        self.custom_timer.fun_tic();
        Jacobian_instance.evaluate_funvector_lambdified_DVector(y.clone().data.into());
        self.custom_timer.fun_tac();
        assert!(
            !Jacobian_instance.evaluated_jacobian_DMatrix.is_empty(),
            "Jacobian should not be empty."
        );
        assert!(
            !Jacobian_instance.evaluated_functions_DVector.is_empty(),
            "Functions should not be empty."
        );
        self.custom_timer.linear_system_tic();
        let J_k = &Jacobian_instance.evaluated_jacobian_DMatrix;
        self.jac = J_k.clone();
        let F_k = &Jacobian_instance.evaluated_functions_DVector;
        info!("\n J_k: {}", J_k);
        info!("\n F_k: {}", F_k);
        for (i, el) in F_k.iter().enumerate() {
            if el.is_nan() {
                let issue_func = Jacobian_instance.vector_of_functions[i].clone();
                let issue_value = y.clone();
                error!(
                    "\n \n NaN in undamped step residual function {} with valus {} \n \n",
                    issue_func, issue_value
                );
                panic!();
                //return previous_step;
            }
        }
        let undamped_step_k = solve_linear_system(method, J_k, F_k).unwrap();
        for el in undamped_step_k.iter() {
            if el.is_nan() {
                log::error!("\n \n NaN in damped step deltaY \n \n");
                panic!();
                //return previous_step;
            }
        }
        //    self.step = undamped_step_k.clone();
        (undamped_step_k, F_k.clone())
    }
    pub fn simple_newton_step(&mut self) -> (i32, Option<DVector<f64>>) {
        let now = Instant::now();
        let y_k_minus_1 = self.y.clone();
        let (undamped_step_k_minus_1, F_k_minus_1) = self.step(y_k_minus_1.clone());
        let lambda = self.dumping_factor;
        let dy = lambda * undamped_step_k_minus_1;
        let damped_step_result: DVector<f64> = y_k_minus_1 - dy.clone();
        let damped_step_result = if self.Bounds.is_some() {
            self.clip(&damped_step_result, &self.bounds_vec.clone())
        } else {
            damped_step_result
        };
        self.custom_timer.linear_system_tac();
        *self
            .calc_statistics
            .entry("number of solving linear systems".to_string())
            .or_insert(0) += 1;

        let error = Matrix::norm(&F_k_minus_1);
        info!("norm of residual = {}", error);
        if (error > self.max_error) && (self.i > 0) {
            warn!("Error is increasing");
        }
        let elapsed = now.elapsed();
        elapsed_time(elapsed);
        if error < self.tolerance {
            return (1, Some(damped_step_result));
        } else {
            info!("iteration = {}, error = {}", self.i, error);
            self.max_error = error;
            return (0, Some(damped_step_result));
        }
    }
    /// a control function that is used to
    /// redirects the flow of execution depending on the method: simple newton iteration,  damped newton iteration
    pub fn extended_step(&mut self) -> (i32, Option<DVector<f64>>) {
        // SIMPLE NEWTON STEPS - NO BOUNDS OF THE VARIABLES
        match self.method {
            Method::simple => self.simple_newton_step(),
            Method::damped => self.step_damped(),

            Method::trust_region => self.step_trust_region(),
            Method::LM => self.step_lm(),
            Method::LM_Nielsen => self.step_trust_region_Nielsen(),
        }
    }
    /// main function to solve the system of equations  
    pub fn main_loop(&mut self) -> Option<DVector<f64>> {
        info!("\n \n solving system of equations with Newton-Raphson method! \n \n");
        let y: DVector<f64> = DVector::from_vec(self.initial_guess.clone());
        self.result = Some(y.clone()); // save into result in case the vary first iteration
        self.y = y.clone();
        while self.i < self.max_iterations {
            info!(
                "\n_____________________________________start of iteration = {}_______________________________\n",
                self.i
            );
            let (status, damped_step_result) = self.extended_step();

            if status == 0 {
                let y_k_plus_1 = match damped_step_result {
                    Some(y_k_plus_1) => y_k_plus_1,
                    _ => {
                        error!("\n \n y_k_plus_1 is None");
                        panic!()
                    }
                };
                self.y = y_k_plus_1;

                info!(
                    "\n_____________________________________end of iteration = {}, error = {}_______________________________\n",
                    self.i, self.max_error
                );
                self.i += 1;
            } else if status == 1 {
                // status == 1 means convergence is reached, save the result
                info!("\n \n Solution has converged, breaking the loop!");

                let y_k_plus_1 = match damped_step_result {
                    Some(y_k_plus_1) => y_k_plus_1,
                    _ => {
                        panic!(" \n \n y_k_plus_1 is None")
                    }
                };
                let result = Some(y_k_plus_1); // save the successful result of the iteration
                // before refining in case it will go wrong
                self.result = result.clone();
                info!(
                    "\n \n solutioon found for {}",
                    &self.result.clone().unwrap()
                );
                return result;
            }

            /*
            if error < self.tolerance {
                self.result = Some(new_x.clone());
                self.max_error = error;
                return Some(new_x);
            } else {
                x = new_x;
                self.i += 1;
                info!("iteration = {}, error = {}", self.i, error)
            }
            */
        }
        error!("Maximum number of iterations reached. No solution found.");
        None
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                       main functions to start the solver and caclulate statistics
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //

    pub fn solver(&mut self) -> Option<DVector<f64>> {
        self.custom_timer.start();
        self.custom_timer.symbolic_operations_tic();
        self.eq_generate();
        self.custom_timer.symbolic_operations_tac();
        let begin = Instant::now();
        let res = self.main_loop();
        self.custom_timer.get_all();
        let end = begin.elapsed();
        elapsed_time(end);
        let time = end.as_secs_f64() as usize;

        self.calc_statistics
            .insert("time elapsed, s".to_string(), time);
        self.calc_statistics();

        self.result = res;
        self.result.clone()
    }
    // wrapper around solver function to implement logging
    pub fn solve(&mut self) -> Option<DVector<f64>> {
        let is_logging_disabled = self
            .loglevel
            .as_ref()
            .map(|level| level == "off" || level == "none")
            .unwrap_or(false);

        if is_logging_disabled {
            let res = self.solver();
            res
        } else {
            let loglevel = self.loglevel.clone();
            let log_option = if let Some(level) = loglevel {
                match level.as_str() {
                    "debug" => LevelFilter::Info,
                    "info" => LevelFilter::Info,
                    "warn" => LevelFilter::Warn,
                    "error" => LevelFilter::Error,
                    _ => panic!("loglevel must be debug, info, warn or error"),
                }
            } else {
                LevelFilter::Info
            };
            println!(" \n \n Program started with loglevel: {}", log_option);
            //  let date_and_time = Local::now().format("%Y-%m-%d_%H-%M-%S");
            //  let name = format!("log_{}.txt", date_and_time);
            let logger_instance = CombinedLogger::init(vec![TermLogger::new(
                log_option,
                Config::default(),
                TerminalMode::Mixed,
                ColorChoice::Auto,
            )]);

            match logger_instance {
                Ok(()) => {
                    let res = self.solver();
                    info!(" \n \n Program ended");
                    res
                }
                Err(_) => {
                    let res = self.solver();
                    res
                } //end Error
            } // end mat 
        }
    }
    pub fn get_result(&self) -> Option<DVector<f64>> {
        self.result.clone()
    }
    fn calc_statistics(&self) {
        let mut stats = self.calc_statistics.clone();
        let jac = &self.jac;
        let jac_shape = jac.shape();
        let matrix_weight = checkmem(jac);
        stats.insert("jacobian memory, MB".to_string(), matrix_weight as usize);
        stats.insert(
            "number of jacobian elements".to_string(),
            jac_shape.0 * jac_shape.1,
        );

        stats.insert("length of y vector".to_string(), self.values.len() as usize);
        stats.insert("number of iterations".to_string(), self.i as usize);
        let mut table = Builder::from(stats).build();
        table.with(Style::modern_rounded());
        info!("\n \n CALC STATISTICS \n \n {}", table.to_string());
    }

    /*
        pub fn get_error(&mut self, x: Vec<f64>) -> f64 {
            let Jacobian_instance = &mut self.jacobian;
            Jacobian_instance.evaluate_funvector_lambdified_DVector(x.clone());
            let new_x = &Jacobian_instance.evaluated_functions_DVector;
            let dx = new_x
                .iter()
                .zip(&x)
                .map(|(x_i, x_j)| (x_i - x_j).abs())
                .collect::<Vec<f64>>();
            let dx_matrix = DVector::from_vec(dx);
            let error = Matrix::norm(&dx_matrix);
            error
        }

        pub fn test_correction(&mut self) -> f64 {
            let result = self.get_result().clone().unwrap().clone();
            let norm = self.get_error(result);
            norm.clone()
        }

        // Gauss-Jordan elimination method. The function takes two parameters: matrix, which is a reference to a vector of vectors representing the coefficients of the linear equations,
        // and constants, which is a reference to a vector containing the constants on the right-hand side of the equations.

        pub fn solve_linear_system(matrix: &[Vec<f64>], constants: &[f64]) -> Vec<f64> {
            // Implement a linear system solver (e.g., LU decomposition, Gauss-Jordan elimination, etc.)
            // Here, we'll use a simple implementation for demonstration purposes
            let n = matrix.len();
            let mut augmented_matrix = matrix
                .iter()
                .cloned()
                .zip(constants.iter().cloned())
                .map(|(row, constant)| {
                    row.into_iter()
                        .chain(std::iter::once(constant))
                        .collect::<Vec<f64>>()
                })
                .collect::<Vec<Vec<f64>>>();

            for i in 0..n {
                let pivot = augmented_matrix[i][i];
                for j in i..n + 1 {
                    augmented_matrix[i][j] /= pivot;
                }

                for k in 0..n {
                    if k != i {
                        let factor = augmented_matrix[k][i];
                        for j in i..n + 1 {
                            augmented_matrix[k][j] -= factor * augmented_matrix[i][j];
                        }
                    }
                }
            }

            augmented_matrix.iter().map(|row| row[n]).collect()
        }
        pub fn solve_linear_LU(
            coeffs: Vec<Vec<f64>>,
            constants: Vec<f64>,
        ) -> Result<Vec<f64>, &'static str> {
            let mut res: Vec<f64> = Vec::new();
            let n = coeffs.len();
            let a: DMatrix<f64> = DMatrix::from_fn(n, n, |i, j| coeffs[i][j]);
            let b: DVector<f64> = DVector::from_vec(constants);
            match a.lu().solve(&b) {
                Some(x) => {
                    info!("Solution: {}", x);
                    res = x.data.into();
                    Ok(res)
                }
                None => {
                    info!("No solution found");
                    Err("no solution")
                }
            }
        }
    */
}
//////////////////////////////////////////////////////////////////////////////////////////////
///                 LINEAR SYSTEM SOLVERS
//////////////////////////////////////////////////////////////////////////////////////////////
pub fn solve_linear_system(
    solver: String,
    A: &DMatrix<f64>,
    b: &DVector<f64>,
) -> Result<DVector<f64>, Box<dyn Error>> {
    //  use crate::somelinalg::linear_sys_diagnostics::poorly_conditioned;
    // poorly_conditioned(A.clone(), 1e5);
    match solver.as_str() {
        "lu" => {
            let lu = A.clone().lu();
            let x = lu.solve(&b);
            match x {
                Some(x) => Ok(x),
                None => Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Failed to solve the system",
                ))),
            }
        }
        "inv" => {
            let A_inv = A.clone().try_inverse().unwrap();
            let f = A_inv * b;
            Ok(f)
        }
        _ => Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Failed to solve the system",
        ))),
    } // match solver.as_str() {
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///                                         MISC
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
pub fn if_initial_guess_inside_bounds(
    initial_guess: &DVector<f64>,
    Bounds: &Option<HashMap<String, (f64, f64)>>,
    values: &Vec<String>,
) -> () {
    //  initial_guess - vector
    //  Bounds - hashmap where keys are variable names and values are tuples with lower and upper bounds.
    //  Function checks if initial guess is inside the bounds of the variables. If not, it panics.
    for (i, el_i) in initial_guess.iter().enumerate() {
        let var_name = values[i].clone();
        let bounds = Bounds
            .as_ref()
            .expect("No bounds specified")
            .get(&var_name)
            .unwrap();
        let (lower, upper) = bounds;
        if el_i < lower || el_i > upper {
            panic!(
                "\n, \n, Initial guess  of the variable {} is outside the bounds {:?}.",
                var_name, &bounds
            );
        }
    }
}

// This function calculates the minimum damping factor necessary to keep the solution within specified bounds after taking a Newton step.
pub fn bound_step(y: &DVector<f64>, step: &DVector<f64>, bounds: &Vec<(f64, f64)>) -> f64 {
    // Initialize no damping
    let mut fbound = 1.0;
    let mut _entry = 0;
    let mut _force = false;
    let mut _value = 0.0;
    let s0 = step;
    for (i, y_i) in y.iter().enumerate() {
        let below = bounds[i].0;
        let above = bounds[i].1;

        let s_i = s0[i];
        if *y_i == below {
            warn!(
                "Solution is on a lower bound, y[{}] = {} but bound is {}",
                i, *y_i, below
            );
        };
        if *y_i == above {
            warn!(
                "Solution is on an upper bound, y[{}] = {} but bound is {}",
                i, *y_i, above
            );
        }
        if s_i > f64::max(*y_i - below, 0.0) {
            let temp = (*y_i - below) / s_i;
            if temp < fbound {
                fbound = temp;
                _entry = i + 1; //
                _force = true;
                _value = below;
            }
        } else if s_i < f64::min(*y_i - above, 0.0) {
            let temp = (*y_i - above) / s_i;
            if temp < fbound {
                fbound = temp;
                _entry = i + 1; //
                _force = true;
                _value = above;
            }
        }
    }
    fbound
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                     TESTS
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    #[test]
    fn test_NR_elem_example_simple() {
        let vec_of_expressions = vec!["x^2+y^2-10", "x-y-4"];

        let initial_guess = vec![1.0, 1.0];
        let mut NR_instanse = NR::new();
        let vec_of_expr = Expr::parse_vector_expression(vec_of_expressions.clone());
        let values = vec!["x".to_string(), "y".to_string()];
        NR_instanse.set_equation_system(
            vec_of_expr,
            Some(values.clone()),
            initial_guess,
            1e-6,
            100,
        );
        NR_instanse.eq_generate();
        NR_instanse.solve();
        let solution = NR_instanse.get_result().unwrap();
        assert_eq!(solution, DVector::from(vec![-1.0, 3.0]));
    }

    #[test]
    fn test_NR_elem_example_simple_str() {
        let mut NR_instanse = NR::new();
        let vec_of_expressions = vec!["x^2+y^2-10".to_string(), "x-y-4".to_string()];
        let initial_guess = vec![1.0, 1.0];
        // solve
        NR_instanse.eq_generate_from_str(vec_of_expressions, None, initial_guess, 1e-6, 100);
        NR_instanse.main_loop();
        let solution = NR_instanse.get_result().unwrap();
        assert_eq!(solution, DVector::from(vec![-1.0, 3.0]));
    }
    #[test]
    fn various_nonlinear_equations_simple() {
        use std::f64;
        // 1) x+y-100 =0, 1/x - 1/y  - 1/200=0
        let mut NR_instanse = NR::new();
        let vec_of_expressions = vec!["x+y-100", "1/x - 1/y  - 1/200"];
        let initial_guess = vec![1.0, 1.0];
        let vec_of_expr = Expr::parse_vector_expression(vec_of_expressions.clone());
        let values = vec!["x".to_string(), "y".to_string()];
        NR_instanse.set_equation_system(
            vec_of_expr,
            Some(values.clone()),
            initial_guess,
            1e-6,
            100,
        );
        NR_instanse.eq_generate();
        NR_instanse.main_loop();
        let solution = NR_instanse.get_result().unwrap();
        let x = -50.0 * (f64::sqrt(17.0) - 5.0);
        let y = 50.0 * (f64::sqrt(17.0) - 3.0);
        assert_relative_eq!(solution[0], y, epsilon = 1e-3);
        assert_relative_eq!(solution[1], x, epsilon = 1e-3);
        // 2)
    }
    fn elemntary_example_test(method: Method, Bounds: Option<HashMap<String, (f64, f64)>>) {
        let vec_of_expressions = vec!["x^2+y^2-10", "x-y-4"];

        let initial_guess = vec![1.0, 1.0];
        let mut NR_instanse = NR::new();
        let vec_of_expr = Expr::parse_vector_expression(vec_of_expressions.clone());
        let values = vec!["x".to_string(), "y".to_string()];
        NR_instanse.set_equation_system(vec_of_expr, Some(values.clone()), initial_guess, 1e-6, 20);
        NR_instanse.set_solver_params(
            Some("info".to_string()),
            None,
            None,
            Bounds,
            Some(method),
            None,
        );
        NR_instanse.eq_generate();
        NR_instanse.solve();
        let solution = NR_instanse.get_result().unwrap();
        println!("solution: {:?}", solution);

        assert_relative_eq!(solution, DVector::from(vec![-1.0, 3.0]), epsilon = 1e-3);
    }
    #[test]
    fn test_NR_elementary_example_simple2() {
        elemntary_example_test(Method::simple, None);
    }
    #[test]
    fn test_chem_equlibrium_simple_scaled() {
        // equations
        let symbolic = Expr::Symbols("N0, N1, N2, Np, Lambda0, Lambda1");
        let dG0 = Expr::Const(-450.0e3);
        let dG1 = Expr::Const(-150.0e3);
        let dG2 = Expr::Const(-50e3);
        // scaling constants for each equation
        let dGm0 = Expr::Const(8.314 * 450e5);
        let dGm1 = Expr::Const(8.314 * 150e5);
        let dGm2 = Expr::Const(8.314 * 50e5);
        let N0 = symbolic[0].clone();
        let N1 = symbolic[1].clone();
        let N2 = symbolic[2].clone();
        let Np = symbolic[3].clone();
        let Lambda0 = symbolic[4].clone();
        let Lambda1 = symbolic[5].clone();

        let RT = Expr::Const(8.314) * Expr::Const(273.15);
        let eq_mu = vec![
            Lambda0.clone()
                + Expr::Const(2.0) * Lambda1.clone()
                + (dG0.clone() + RT.clone() * Expr::ln(N0.clone() / Np.clone())) / dGm0.clone(),
            Lambda0
                + Lambda1.clone()
                + (dG1 + RT.clone() * Expr::ln(N1.clone() / Np.clone())) / dGm1.clone(),
            Expr::Const(2.0) * Lambda1
                + (dG2 + RT * Expr::ln(N2.clone() / Np.clone())) / dGm2.clone(),
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

        for eq in &full_system_sym {
            println!("eq: {}", eq);
        }
        // solver
        let initial_guess = vec![0.5, 0.5, 0.5, 1.0, 2.0, 2.0];
        let unknowns: Vec<String> = symbolic.iter().map(|x| x.to_string()).collect();
        let mut solver = NR::new();
        solver.set_equation_system(
            full_system_sym.clone(),
            Some(unknowns.clone()),
            initial_guess,
            1e-2,
            1000,
        );
        solver.set_solver_params(
            Some("info".to_string()),
            None,
            Some(0.009),
            None,
            None,
            None,
        );
        solver.eq_generate();
        solver.solve();
        let solution = solver.get_result().expect("Failed to get result");
        let solution: Vec<f64> = solution.data.into();
        let map_of_solutions: HashMap<String, f64> = unknowns
            .iter()
            .zip(solution.iter())
            .map(|(k, v)| (k.to_string(), *v))
            .collect();

        let map_of_solutions = map_of_solutions;
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
    fn test_NR_elementary_example_with_bounds() {
        let Bounds = HashMap::from([
            ("x".to_string(), (-10.0, 10.0)),
            ("y".to_string(), (-10.0, 10.0)),
        ]);
        elemntary_example_test(Method::damped, Some(Bounds));
    }

    fn full_system_sym(dGm0: Expr) -> Vec<Expr> {
        let symbolic = Expr::Symbols("N0, N1, N2, Np, Lambda0, Lambda1");
        let dG0 = Expr::Const(-450.0e3);
        let dG1 = Expr::Const(-150.0e3);
        let dG2 = Expr::Const(-50e3);

        let N0 = symbolic[0].clone();
        let N1 = symbolic[1].clone();
        let N2 = symbolic[2].clone();
        let Np = symbolic[3].clone();
        let Lambda0 = symbolic[4].clone();
        let Lambda1 = symbolic[5].clone();

        let RT = Expr::Const(8.314) * Expr::Const(273.15);
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
        full_system_sym
    }

    fn test_solver_with_certain_method(
        method: Method,
        parameters: Option<HashMap<String, f64>>,
        dGm0: Expr,
        Bounds: HashMap<String, (f64, f64)>,
        enable_weighting: bool,
        initial_guess: Vec<f64>,
        scaling_method: Option<ScalingMethod>,
        subproblem_method: Option<SubproblemMethod>,
        convergence_criteria: Option<ConvergenceCriteria>,
    ) {
        // equations
        let symbolic = Expr::Symbols("N0, N1, N2, Np, Lambda0, Lambda1");

        let full_system_sym = full_system_sym(dGm0);
        for eq in &full_system_sym {
            println!("eq: {}", eq);
        }
        // solver

        let unknowns: Vec<String> = symbolic.iter().map(|x| x.to_string()).collect();
        let mut solver = NR::new();
        solver.set_equation_system(
            full_system_sym.clone(),
            Some(unknowns.clone()),
            initial_guess,
            2.0 * 1e-3,
            130,
        );
        solver.set_solver_params(
            Some("info".to_string()),
            None,
            None,
            Some(Bounds),
            Some(method),
            parameters,
        );
        solver.set_additional_params(
            scaling_method,
            None,
            None,
            subproblem_method,
            convergence_criteria,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        if enable_weighting {
            solver.implement_weights();
        }
        solver.eq_generate();

        solver.solve();
        let solution = solver.get_result().expect("Failed to get result");
        let solution: Vec<f64> = solution.data.into();
        let map_of_solutions: HashMap<String, f64> = unknowns
            .iter()
            .zip(solution.iter())
            .map(|(k, v)| (k.to_string(), *v))
            .collect();

        let map_of_solutions = map_of_solutions;
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
        println!("map_of_solutions: {:? }", map_of_solutions);
        assert!(d1.abs() < 8e-3);
        assert!(d2.abs() < 8e-3);
        assert!(d3.abs() < 8e-3);
    }
    #[test]

    fn test_solver_with_clipping_method() {
        let dGm0 = Expr::Const(8.314 * 8.0e7);

        let params = HashMap::from([("maxDampIter".to_string(), 18.0)]);
        let Boubds = HashMap::from([
            ("N0".to_string(), (1e-40, 2.0)),
            ("N1".to_string(), (1e-40, 2.0)),
            ("N2".to_string(), (1e-40, 2.0)),
            ("Np".to_string(), (1e-40, 10.0)),
            ("Lambda0".to_string(), (-1e-1, 1e-2)),
            ("Lambda1".to_string(), (-1e-1, 1e-2)),
        ]);
        let initial_guess = vec![0.9, 0.9, 0.9, 0.6, 0.0, 0.0];
        test_solver_with_certain_method(
            Method::damped,
            Some(params),
            dGm0,
            Boubds,
            false,
            initial_guess,
            None,
            None,
            None,
        );
    }

    #[test]
    fn test_solver_with_levenberg_marquardt_method() {
        let Bounds = HashMap::from([
            ("x".to_string(), (-10.0, 10.0)),
            ("y".to_string(), (-10.0, 10.0)),
        ]);
        elemntary_example_test(Method::LM, Some(Bounds));
        let dGm0 = Expr::Const(8.314 * 450e4);

        let params = HashMap::from([
            ("diag".to_string(), 1.0),
            ("increase_factor".to_string(), 11.0),
            ("decrease_factor".to_string(), 9.0),
        ]);
        let Boubds = HashMap::from([
            ("N0".to_string(), (1e-40, 2.0)),
            ("N1".to_string(), (1e-40, 2.0)),
            ("N2".to_string(), (1e-40, 2.0)),
            ("Np".to_string(), (1e-40, 10.0)),
            ("Lambda0".to_string(), (-1e-1, 1e-2)),
            ("Lambda1".to_string(), (-1e-1, 1e-2)),
        ]);
        let initial_guess = vec![0.9, 0.9, 0.9, 0.6, 0.0, 0.0];
        test_solver_with_certain_method(
            Method::LM,
            Some(params),
            dGm0,
            Boubds,
            false,
            initial_guess,
            None,
            None,
            None,
        );
    }
    #[test]
    fn test_simple_solver() {
        let dGm0 = Expr::Const(8.314 * 60e5); //  8.314 * 40e5

        let Boubds = HashMap::from([
            ("N0".to_string(), (1e-40, 2.0)),
            ("N1".to_string(), (1e-40, 2.0)),
            ("N2".to_string(), (1e-40, 2.0)),
            ("Np".to_string(), (1e-40, 10.0)),
            ("Lambda0".to_string(), (-10000.0, 1e6)),
            ("Lambda1".to_string(), (-100000.0, 1e6)),
        ]);
        let initial_guess = vec![0.9, 0.9, 0.9, 0.6, 0.0, 0.0];
        test_solver_with_certain_method(
            Method::LM_Nielsen,
            None,
            dGm0,
            Boubds,
            false,
            initial_guess,
            Some(ScalingMethod::More),
            None,
            Some(ConvergenceCriteria::SimpleScaled),
        );
    }
    #[test]
    fn test_w_residuals() {
        let vec_of_expressions = vec!["x^2+y^2-10", "x-y-4"];

        let initial_guess = vec![1.0, 1.0];
        let mut NR_instanse = NR::new();
        let vec_of_expr = Expr::parse_vector_expression(vec_of_expressions.clone());
        let values = vec!["x".to_string(), "y".to_string()];
        NR_instanse.set_equation_system(vec_of_expr, Some(values.clone()), initial_guess, 1e-6, 20);
        NR_instanse.set_solver_params(
            Some("info".to_string()),
            None,
            None,
            None,
            Some(Method::simple),
            None,
        );
        NR_instanse.implement_weights();
        println!("weigted residuals: {:?}", NR_instanse.eq_system);
        NR_instanse.eq_generate();
        NR_instanse.solve();
    }
}
