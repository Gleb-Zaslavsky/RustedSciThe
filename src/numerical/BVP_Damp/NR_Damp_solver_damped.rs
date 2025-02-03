/*
Modified Newton method or Damped Newton method for solving a system of nonlinear equations.
This code implements a modified Newton method for solving a system of non-linear boundary value problems..
The code mostly inspired by sources listed below:
-  Cantera MultiNewton solver (MultiNewton.cpp )
- TWOPNT fortran solver (see "The Twopnt Program for Boundary Value Problems" by J. F. Grcar and Chemkin Theory Manual p.261)
*/
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions_BVP::Jacobian;
use chrono::Local;

use crate::numerical::BVP_Damp::BVP_traits::{
    Fun, FunEnum, Jac, MatrixType, VectorType, Vectors_type_casting,
};
use crate::numerical::BVP_Damp::BVP_utils::{
    construct_full_solution, elapsed_time, extract_unknown_variables, task_check_mem,
};
use crate::numerical::BVP_Damp::BVP_utils_damped::{
    bound_step, convergence_condition, if_initial_guess_inside_bounds, interchange_columns,
    jac_recalc,
};
use crate::Utils::logger::save_matrix_to_file;
use crate::Utils::plots::plots;
use nalgebra::{DMatrix, DVector};
use simplelog::LevelFilter;
use simplelog::*;
use std::collections::HashMap;
use std::fs::File;
use std::time::Instant;
use tabled::{builder::Builder, settings::Style};

use crate::numerical::BVP_Damp::grid_api::{new_grid, GridRefinementMethod};

use log::{error, info, warn};

use super::BVP_utils::checkmem;

pub struct NRBVP {
    pub eq_system: Vec<Expr>, // the system of ODEs defined in the symbolic format
    pub initial_guess: DMatrix<f64>, // initial guess s - matrix with number of rows equal to the number of unknown vars, and number of columns equal to the number of steps
    pub values: Vec<String>,         //unknown variables
    pub arg: String,                 // time or coordinate
    pub BorderConditions: HashMap<String, (usize, f64)>, // hashmap where keys are variable names and values are tuples with the index of the boundary condition (0 for inititial condition 1 for ending condition) and the value.
    pub t0: f64,                                         // initial value of argument
    pub t_end: f64,                                      // end of argument
    pub n_steps: usize,                                  // number of  steps
    pub scheme: String,                                  // name of the numerical scheme
    pub strategy: String,                                // name of the strategy
    pub strategy_params: Option<HashMap<String, Option<Vec<f64>>>>, // solver parameters
    pub linear_sys_method: Option<String>,               // method for solving linear system
    pub method: String,     // define crate using for matrices and vectors
    pub abs_tolerance: f64, // relative tolerance

    pub rel_tolerance: Option<HashMap<String, f64>>, // absolute tolerance - hashmap of the var names and values of tolerance for them
    pub max_iterations: usize,                       // maximum number of iterations
    pub max_error: f64,
    pub Bounds: Option<HashMap<String, (f64, f64)>>, // hashmap where keys are variable names and values are tuples with lower and upper bounds.
    pub loglevel: Option<String>,
    // thets all user defined  parameters
    //
    pub result: Option<DVector<f64>>, // result vector of calculation
    pub x_mesh: DVector<f64>,
    pub fun: Box<dyn Fun>, // vector representing the discretized sysytem
    pub jac: Option<Box<dyn Jac>>, // matrix function of Jacobian
    pub p: f64,            // parameter
    pub y: Box<dyn VectorType>, // iteration vector
    m: usize,              // iteration counter without jacobian recalculation
    old_jac: Option<Box<dyn MatrixType>>,
    jac_recalc: bool,            //flag indicating if jacobian should be recalculated
    error_old: f64,              // error of previous iteration
    bounds_vec: Vec<(f64, f64)>, //vector of bounds for each of the unkown variables (discretized vector)
    rel_tolerance_vec: Vec<f64>, // vector of relative tolerance for each of the unkown variables
    variable_string: Vec<String>, // vector of indexed variable names
    #[allow(dead_code)]
    adaptive: bool,              // flag indicating if adaptive grid should be used
    new_grid_enabled: bool,      //flag indicating if the grid should be refined
    grid_refinemens: usize,      //
    number_of_refined_intervals: usize, //number of refined intervals
    bandwidth: (usize, usize),   //bandwidth
    calc_statistics: HashMap<String, usize>,
}

impl NRBVP {
    pub fn new(
        eq_system: Vec<Expr>,        //
        initial_guess: DMatrix<f64>, // initial guess
        values: Vec<String>,
        arg: String,
        BorderConditions: HashMap<String, (usize, f64)>,
        t0: f64,
        t_end: f64,
        n_steps: usize,
        scheme: String,
        strategy: String,
        strategy_params: Option<HashMap<String, Option<Vec<f64>>>>,
        linear_sys_method: Option<String>,
        method: String,
        abs_tolerance: f64, // tolerance
        rel_tolerance: Option<HashMap<String, f64>>,
        max_iterations: usize, // max number of iterations

        Bounds: Option<HashMap<String, (f64, f64)>>,
        loglevel: Option<String>,
    ) -> NRBVP {
        //jacobian: Jacobian, initial_guess: Vec<f64>, tolerance: f64, max_iterations: usize, max_error: f64, result: Option<Vec<f64>>
        let y0 = Box::new(DVector::from_vec(vec![0.0, 0.0]));

        let fun0: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>> =
            Box::new(|_x, y: &DVector<f64>| y.clone());
        let boxed_fun: Box<dyn Fun> = Box::new(FunEnum::Dense(fun0));
        let h = (t_end - t0) / n_steps as f64;
        let T_list: Vec<f64> = (0..n_steps + 1)
            .map(|i| t0 + (i as f64) * h)
            .collect::<Vec<_>>();

        // let fun0 =  Box::new( |x, y: &DVector<f64>| y.clone() );
        let new_grid_enabled_: bool = if strategy_params.clone().unwrap().get("adaptive").is_some()
        {
            info!("problem with adaptive grid");
            true
        } else {
            false
        };
        let vec_of_tuples = vec![
            ("number of iterations".to_string(), 0),
            ("number of solving linear systems".to_string(), 0),
            ("number of jacobians recalculations".to_string(), 0),
            ("number of grid refinements".to_string(), 0),
        ];
        let Hashmap_statistics: HashMap<String, usize> = vec_of_tuples.into_iter().collect();
        NRBVP {
            eq_system,
            initial_guess: initial_guess.clone(),
            values,
            arg,
            BorderConditions,
            t0,
            t_end,
            n_steps,
            abs_tolerance,
            rel_tolerance,
            scheme,
            strategy,
            strategy_params,
            linear_sys_method,
            method,
            max_iterations,
            max_error: 0.0,
            Bounds,
            loglevel,
            result: None,
            x_mesh: DVector::from_vec(T_list),
            fun: boxed_fun,
            jac: None,
            p: 0.0,
            y: y0,
            m: 0,
            old_jac: None,
            jac_recalc: true,
            error_old: 0.0,

            bounds_vec: Vec::new(),
            rel_tolerance_vec: Vec::new(),
            variable_string: Vec::new(),
            adaptive: false,
            new_grid_enabled: new_grid_enabled_,
            grid_refinemens: 0,
            number_of_refined_intervals: 0,
            bandwidth: (0, 0),
            calc_statistics: Hashmap_statistics,
        }
    }
    /// Basic methods to set the equation system

    // check if user specified task is correct
    pub fn task_check(&self) {
        assert_eq!(
            self.initial_guess.len(), //grid length =  number of unknowns
            self.n_steps * self.values.len(),
            "lenght of initial guess {} should be equal to n_steps*values, {}, {} ",
            self.initial_guess.len(),
            self.x_mesh.len(),
            self.values.len()
        );
        assert!(self.t_end > self.t0, "t_end must be greater than t0");
        assert!(self.n_steps > 1, "n_steps must be greater than 1");
        assert!(
            self.max_iterations > 1,
            "max_iterations must be greater than 1"
        );
        let (m, n) = self.initial_guess.shape();
        if m != self.values.len() {
            panic!(
                "m must be equal to the length of the argument, m= {}, arg = {}",
                m,
                self.arg.len()
            );
        }
        assert_eq!(n, self.n_steps, "n must be equal to the number of steps");
        assert!(
            self.abs_tolerance > 0.0,
            "tolerance must be greater than 0.0"
        );

        assert!(
            !self.BorderConditions.is_empty(),
            "BorderConditions must be specified"
        );
        assert_eq!(
            self.BorderConditions.len(),
            self.values.len(),
            "BorderConditions must be specified for each value"
        );
        assert!(
            !self.Bounds.is_none(),
            "Bounds must be specified for each value"
        );
        let bound_keys_vec = self
            .Bounds
            .clone()
            .unwrap()
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(
            bound_keys_vec.len(),
            self.values.len(),
            "Bounds must be specified for each value"
        );
        // check if initial guess values are inside bunds defined for certain values
        if self.result.is_none() {
            // check of does the guess fits into bounds must be enable only at the beginning (at result == None)
            // we will find ourselves in this place again when the command to recalculate the lattice is given, and the result of the previous
            //iteration may go beyond the boundaries and we must make sure that this fact does not stop the program, therefore
            if_initial_guess_inside_bounds(&self.initial_guess, &self.Bounds, self.values.clone());
        }
        assert!(
            !self.rel_tolerance.is_none(),
            "rel_tolerance must be specified for each value"
        );

        let required_keys = vec!["max_jac", "maxDampIter", "DampFacor", "adaptive"];
        for key in required_keys {
            assert!(
                self.strategy_params.as_ref().unwrap().contains_key(key),
                "Key '{}' must be present in strategy_params",
                key
            );
        }
        if self
            .strategy_params
            .as_ref()
            .unwrap()
            .get("adaptive")
            .unwrap()
            .is_some()
        {
            // Check that exactly one of the required keys is present
            let strategy_keys = vec![
                "easy".to_string(),
                "pearson".to_string(),
                "grcar_smooke".to_string(),
                "two_point".to_string(),
            ];
            let present_keys: Vec<_> = strategy_keys
                .iter()
                .filter(|&key| self.strategy_params.as_ref().unwrap().contains_key(key))
                .collect();
            assert_eq!(
                present_keys.len(), 
                1, 
                "Exactly one of {:?} must be present in strategy_params when 'adaptive' is Some, found: {:?}", 
                strategy_keys, 
                present_keys
                );
            let vec_of_params_len = self
                .strategy_params
                .as_ref()
                .unwrap()
                .get("adaptive")
                .unwrap()
                .clone()
                .unwrap()
                .len();
            assert_eq!(
                vec_of_params_len, 2,
                "vector of 2 elements must be present in  strategy_params when 'adaptive' is Some"
            );
        };
    }
    ///Set system of equations with vector of symbolic expressions
    pub fn eq_generate(&mut self, mesh_: Option<Vec<f64>>, bandwidth: Option<(usize, usize)>) {
        // check if memory is enough for
        task_check_mem(self.n_steps, self.values.len(), &self.method);
        // check if user specified task is correct
        self.task_check();
        // strategy_check(&self.strategy, &self.strategy_params);
        let mut jacobian_instance = Jacobian::new();
        // mesh of t's can be defined directly or by size of step -h, and number of points
        let (h, n_steps, mesh) = if mesh_.is_none() {
            // case of mesh not defined directly
            let h = Some((self.t_end - self.t0) / self.n_steps as f64);
            let n_steps = Some(self.n_steps);
            (h, n_steps, None)
        } else {
            // case of mesh defined directly
            self.x_mesh = DVector::from_vec(mesh_.clone().unwrap());
            (None, None, mesh_)
        };
        let scheme = self.scheme.clone();

        jacobian_instance.generate_BVP(
            self.eq_system.clone(),
            self.values.clone(),
            self.arg.clone(),
            self.t0.clone(),
            None,
            n_steps.clone(),
            h,
            mesh,
            self.BorderConditions.clone(),
            self.Bounds.clone(),
            self.rel_tolerance.clone(),
            scheme.clone(),
            self.method.clone(),
            bandwidth,
        );

        //     info("Jacobian = {:?}", jacobian_instance.readable_jacobian);
        let fun = jacobian_instance.residiual_function;

        let jac = jacobian_instance.jac_function;

        self.fun = fun;

        self.jac = jac;
        self.bounds_vec = jacobian_instance.bounds.unwrap();
        self.rel_tolerance_vec = jacobian_instance.rel_tolerance_vec.unwrap();
        self.variable_string = jacobian_instance.variable_string;
        self.bandwidth = jacobian_instance.bandwidth.unwrap();
    } // end of method eq_generate
    pub fn set_new_step(&mut self, p: f64, y: Box<dyn VectorType>, initial_guess: DMatrix<f64>) {
        self.p = p;
        self.y = y;
        self.initial_guess = initial_guess;
    }
    pub fn set_p(&mut self, p: f64) {
        self.p = p;
    }
    /////////////////////
    pub fn step_with_inv_Jac(&self, p: f64, y: &dyn VectorType) -> Box<dyn VectorType> {
        let fun = &self.fun;
        let F_k = fun.call(p, y);
        let inv_J_k = self.old_jac.as_ref().unwrap().clone_box();
        let undamped_step_k: Box<dyn VectorType> = inv_J_k.mul(&*F_k);
        undamped_step_k
    }
    pub fn recalc_and_inverse_Jac(&mut self) {
        if self.jac_recalc {
            let p = self.p;
            let y = &*self.y;
            log::info!("\n \n JACOBIAN (RE)CALCULATED! \n \n");
            let jac_function = self.jac.as_mut().unwrap();
            let jac_matrix = jac_function.call(p, y);
            let inv_J_k = jac_function.inv(&*jac_matrix, self.abs_tolerance, self.max_iterations);
            self.old_jac = Some(inv_J_k);
            self.m = 0;
            *self
                .calc_statistics
                .entry("number of jacobians recalculations".to_string())
                .or_insert(0) += 1;
        }
    }
    ////////////////////////////////////////////////////////////////

    // jacobian recalculation
    fn recalc_jacobian(&mut self) {
        if self.jac_recalc {
            let p = self.p;
            let y = &*self.y;
            info!("\n \n JACOBIAN (RE)CALCULATED! \n \n");
            let begin = Instant::now();
            let jac_function = self.jac.as_mut().unwrap();
            let jac_matrix = jac_function.call(p, y);
            // println!(" \n \n new_j = {:?} ", jac_rowwise_printing(&*&new_j) );
            info!("jacobian recalculation time: ");
            let elapsed = begin.elapsed();
            elapsed_time(elapsed);
            self.old_jac = Some(jac_matrix);
            self.m = 0;
            *self
                .calc_statistics
                .entry("number of jacobians recalculations".to_string())
                .or_insert(0) += 1;
        }
    }

    //undamped step without jacobian recalculation
    pub fn step(&self, p: f64, y: &dyn VectorType) -> Box<dyn VectorType> {
        let fun = &self.fun;
        let F_k = fun.call(p, y);
        let J_k = self.old_jac.as_ref().unwrap();
        assert_eq!(
            F_k.len(),
            J_k.shape().0,
            "length of F_k {} and number of rows in J_k {} must be equal",
            F_k.len(),
            J_k.shape().0
        );
        let residual_norm = F_k.norm();
        info!("\n \n residual norm = {:?} ", residual_norm);
        // jac_rowwise_printing(&J_k);
        //    println!(" \n \n F_k = {:?} \n \n", F_k.to_DVectorType());
        for el in F_k.iterate() {
            if el.is_nan() {
                error!("\n \n NaN in undamped step residual function \n \n");
                panic!()
            }
        }
        // solving equation J_k*dy_k=-F_k for undamped dy_k, but Lambda*dy_k - is dumped step
        let undamped_step_k: Box<dyn VectorType> = J_k.solve_sys(
            &*F_k,
            self.linear_sys_method.clone(),
            self.abs_tolerance,
            self.max_iterations,
            self.bandwidth,
            y,
        );
        for el in undamped_step_k.iterate() {
            if el.is_nan() {
                log::error!("\n \n NaN in damped step deltaY \n \n");
                panic!()
            }
        }
        undamped_step_k
    }

    pub fn damped_step(&mut self) -> (i32, Option<Box<dyn VectorType>>) {
        let p = self.p;
        let now = Instant::now();
        // compute the undamped Newton step
        let y_k_minus_1 = &*self.y;
        let undamped_step_k_minus_1 = self.step(p, y_k_minus_1);
        *self
            .calc_statistics
            .entry("number of solving linear systems".to_string())
            .or_insert(0) += 1;
        let y_k = y_k_minus_1 - &*undamped_step_k_minus_1;
        let fbound = bound_step(y_k_minus_1, &*undamped_step_k_minus_1, &self.bounds_vec);
        if fbound.is_nan() {
            error!("\n \n fbound is NaN \n \n");
            panic!()
        }
        if fbound.is_infinite() {
            error!("\n \n fbound is infinite \n \n");
            panic!()
        }
        // let fbound =1.0;
        info!("\n \n fboundary  = {}", fbound);
        let mut lambda = 1.0 * fbound;
        // if fbound is very small, then x0 is already close to the boundary and
        // step0 points out of the allowed domain. In this case, the Newton
        // algorithm fails, so return an error condition.
        if fbound < 1e-10 {
            log::warn!(
                "\n  No damped step can be taken without violating solution component bounds."
            );
            return (-3, None);
        }

        let maxDampIter = if let Some(maxDampIter_) = self
            .strategy_params
            .clone()
            .unwrap()
            .get("maxDampIter")
            .unwrap()
        {
            maxDampIter_[0] as usize
        } else {
            5
        };
        let DampFacor: f64 = if let Some(DampFacor_) = self
            .strategy_params
            .clone()
            .unwrap()
            .get("DampFacor")
            .unwrap()
        {
            DampFacor_[0]
        } else {
            0.5
        };

        let mut k_: usize = 0;
        let mut S_k_plus_1: Option<f64> = None;
        let mut damped_step_result: Option<Box<dyn VectorType>> = None;
        let mut conv: f64 = 0.0;
        //   let fun = &self.fun;
        // calculate damped step
        let undamped_step_k = self.step(p, &*y_k);
        *self
            .calc_statistics
            .entry("number of solving linear systems".to_string())
            .or_insert(0) += 1;
        //
        for mut k in 0..maxDampIter {
            if k > 1 {
                info!("\n \n damped_step number {} ", k);
            }
            info!("\n \n Damping coefficient = {}", lambda);
            let damped_step_k = undamped_step_k.mul_float(lambda);
            let S_k = &damped_step_k.norm();

            let y_k_plus_1: Box<dyn VectorType> = &*y_k - &*damped_step_k;

            // / compute the next undamped step that would result if x1 is accepted
            // J(x_k)^-1 F(x_k+1)
            let undamped_step_k_plus_1 = self.step(p, &*y_k_plus_1); //???????????????
            let error = &undamped_step_k_plus_1.norm();
            self.error_old = *error;
            info!("\n \n L2 norm of undamped step = {}", error);
            let convergence_cond_for_step =
                convergence_condition(&*y_k_plus_1, &self.abs_tolerance, &self.rel_tolerance_vec);
            let S_k_plus_1_temp = &undamped_step_k_plus_1.norm();
            // If the norm of S_k_plus_1 is less than the norm of S_k, then accept this
            // damping coefficient. Also accept it if this step would result in a
            // converged solution. Otherwise, decrease the damping coefficient and
            // try again.
            let elapsed = now.elapsed();
            elapsed_time(elapsed);
            if (S_k_plus_1_temp < S_k) || (S_k_plus_1 < Some(convergence_cond_for_step)) {
                // The  criterion for accepting is that the undamped steps decrease in
                // magnitude, This prevents the iteration from stepping away from the region where there is good reason to believe a solution lies

                k_ = k;
                S_k_plus_1 = Some(*S_k_plus_1_temp);
                damped_step_result = Some(y_k_plus_1.clone_box());
                conv = convergence_cond_for_step;
                break;
            }
            // if fail this criterion we must reject it and retries the step with a reduced (often halved) damping parameter trying again until
            // criterion is met  or max damping iterations is reached

            lambda = lambda / (2.0f64.powf(k as f64 + DampFacor));
            k_ = k;
            S_k_plus_1 = Some(*S_k_plus_1_temp);

            k = k + 1;
        }

        if k_ < maxDampIter {
            // if there is a damping coefficient found (so max damp steps not exceeded)
            if S_k_plus_1.unwrap() > conv {
                //found damping coefficient but not converged yet
                info!("\n \n  Damping coefficient found (solution has not converged yet)");
                info!(
                    "\n \n  step norm =  {}, weight norm = {}, convergence condition = {}",
                    self.error_old,
                    S_k_plus_1.unwrap(),
                    conv
                );
                (0, damped_step_result)
            } else {
                info!("\n \n  Damping coefficient found (solution has converged)");
                info!(
                    "\n \n step norm =  {}, weight norm = {}, convergence condition = {}",
                    self.error_old,
                    S_k_plus_1.unwrap(),
                    conv
                );
                (1, damped_step_result)
            }
        } else {
            //  if we have reached max damping iterations without finding a damping coefficient we must reject the step
            warn!("\n \n  No damping coefficient found (max damping iterations reached)");
            (-2, None)
        }
    } // end of damped step
    pub fn main_loop_damped(&mut self) -> Option<DVector<f64>> {
        info!("\n \n solving system of equations with Newton-Raphson method! \n \n");
        let y: DMatrix<f64> = self.initial_guess.clone();
        //  println!("new y = {} \n \n", &y);
        let y: Vec<f64> = y.iter().cloned().collect();
        let y: DVector<f64> = DVector::from_vec(y);
        self.result = Some(y.clone()); // save into result in case the vary first iteration
                                       // with the current n_steps will go wrong and we shall need frid refinement
        self.y = Vectors_type_casting(&y.clone(), self.method.clone());
        // println!("y = {:?}", &y);
        let mut nJacReeval = 0;
        let mut i = 0;
        while i < self.max_iterations {
            self.jac_recalc = jac_recalc(
                &self.strategy_params,
                self.m,
                &self.old_jac,
                &mut self.jac_recalc,
            );
            self.recalc_jacobian();
            self.m += 1;
            i += 1; // increment the number of iterations
            *self
                .calc_statistics
                .entry("number of iterations".to_string())
                .or_insert(0) += 1;
            let (status, damped_step_result) = self.damped_step();

            if status == 0 {
                // status == 0 means convergence is not reached yet we're going to another iteration
                let y_k_plus_1 = if let Some(y_k_plus_1) = damped_step_result {
                    y_k_plus_1
                } else {
                    error!("\n \n y_k_plus_1 is None");
                    panic!()
                };
                self.y = y_k_plus_1;
                self.jac_recalc = false;
            }
            // status == 0
            else if status == 1 {
                // status == 1 means convergence is reached, save the result
                info!("\n \n Solution has converged, breaking the loop!");

                let y_k_plus_1 = if let Some(y_k_plus_1) = damped_step_result {
                    y_k_plus_1
                } else {
                    panic!(" \n \n y_k_plus_1 is None")
                };

                let result = Some(y_k_plus_1.to_DVectorType()); // save the successful result of the iteration
                                                                // before refining in case it will go wrong
                self.result = result.clone();
                info!(
                    "\n \n solutioon found for the current grid {}",
                    &self.result.clone().unwrap()
                );

                // if flag for new grid is up we must call adaptive grid refinement
                if self.new_grid_enabled
                    && self
                        .strategy_params
                        .clone()
                        .unwrap()
                        .get("adaptive")
                        .unwrap()
                        .clone()
                        .is_some()
                {
                    info!("solving with new grid!");
                    self.solve_with_new_grid()
                } else {
                    // if adapive is None then we just return the result
                    info!("returning the result");

                    return result;
                };
            //  self.max_error = error; // ???
            }
            // status == 1
            else if status < 0 {
                //negative means convergence is not reached yet, damped step is not accepted
                if self.m > 1 {
                    // if we have already tried 2 times with same Jacobian we must recalculate Jacobian
                    self.jac_recalc = true;
                    info!(
                        "\n \n status <0, recalculating Jacobian flag up! Jacobian age = {} \n \n",
                        self.m
                    );
                    if nJacReeval > 3 {
                        break;
                    }
                    nJacReeval += 1;
                } else {
                    info!("\n \n Jacobian age {} =<1 \n \n", self.m);
                    //  self.new_grid_enabled = true;
                    break;
                }
            } // status <0

            info!("\n \n end of iteration {} with jac age {} \n \n", i, self.m);
        }

        // all iterations, recalculations of Jacobian were unsuccessful
        // only that can help - grid refinement

        if self.new_grid_enabled
            && self
                .strategy_params
                .clone()
                .unwrap()
                .get("adaptive")
                .unwrap()
                .clone()
                .is_some()
        {
            info!("\n \n iterations unsuccessful, calling solve_with_new_grid \n \n");
            self.solve_with_new_grid()
        }

        None
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                      functions to create a new grid and recalculate with new grid
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // function to choose if we need to refine the grid in next iteration
    pub fn we_need_refinement(&mut self) {
        let vec_of_params = self
            .strategy_params
            .as_ref()
            .unwrap()
            .get("adaptive")
            .unwrap()
            .clone()
            .unwrap();
        let version = vec_of_params[0] as usize;
        let mut res = match version {
            1 => {
                if self.number_of_refined_intervals == 0 {
                    log::info!(
                        "\n \n number of marked intervals is 0, no new grid is needed \n \n"
                    );
                    false
                } else {
                    log::info!(
                        "\n \n number of marked intervals is {}, new grid is needed \n \n",
                        self.number_of_refined_intervals
                    );
                    true
                }
            }

            _ => {
                panic!()
            }
        };
        let max_grid_refinenents = vec_of_params[1] as usize;
        if max_grid_refinenents <= self.grid_refinemens {
            info!(
                "maximum number of grid refinements {} reached  {} ",
                max_grid_refinenents, self.grid_refinemens
            );
            res = false;
        }
        self.new_grid_enabled = res;
    }
    fn create_new_grid(&mut self) -> (Vec<f64>, DMatrix<f64>, usize) {
        info!("\n \n creating new grid \n \n");
        let y = self.result.clone().unwrap().clone_box();
        let y_DVector = y.to_DVectorType();
        let nrows = self.values.len();
        let ncols = self.n_steps;
        let y_DMatrix = DMatrix::from_column_slice(nrows, ncols, y_DVector.as_slice());

        // unpack the adaptive strategy parameters
        let params = self.strategy_params.clone().unwrap();

        //there are several approaches to create a new grid, so we provide some of them for the user to choose from
        let (vector_of_params, method) = if params.contains_key("easy") {
            let res = params.get("easy").clone().unwrap().clone().unwrap();
            if res.len() != 2 {
                panic!("parameters for adaptive strategy not found")
            };
            let method = GridRefinementMethod::Easiest;
            (res, method)
        } else if params.contains_key("pearson") {
            let res = params.get("pearson").clone().unwrap().clone().unwrap();
            if res.len() != 1 {
                panic!("this strategy requires only one parameter")
            };
            let method = GridRefinementMethod::Pearson;
            (res, method)
        } else if params.contains_key("grcar_smooke") {
            let res = params.get("grcar_smooke").clone().unwrap().clone().unwrap();
            if res.len() != 3 {
                panic!("this strategy requires 3 parameters")
            };
            let method = GridRefinementMethod::GrcarSmooke;
            (res, method)
        } else if params.contains_key("two_point") {
            let res = params.get("two_point").clone().unwrap().clone().unwrap();
            if res.len() != 3 {
                panic!("this strategy requires 3 parameters")
            };

            let method = GridRefinementMethod::TwoPoint;
            (res, method)
        } else {
            panic!("parameters for adaptive strategy not found")
        };
        //create a new mesh with a chosen algorithm a
        // API of new grid returns a new mesh, initial guess and number of intervals that doesnt meet the criteria and wac subdivided
        // if number_of_nonzero_keys==0 it means that no need to create a new grid

        let y_DMatrix = construct_full_solution(
            y_DMatrix,
            &self.BorderConditions,
            &self.variable_string,
            &self.values,
        );
        let (new_mesh, initial_guess, number_of_nonzero_keys) = new_grid(
            method,
            &y_DMatrix,
            &self.x_mesh,
            vector_of_params.clone(),
            self.abs_tolerance,
        );

        //   info!("\n \n new grid enabled! \n \n");
        let initial_guess = extract_unknown_variables(
            initial_guess,
            &self.BorderConditions,
            &self.variable_string,
            &self.values,
        );
        (new_mesh, initial_guess, number_of_nonzero_keys)
    }

    fn solve_with_new_grid(&mut self) {
        let (new_mesh, initial_guess, number_of_nonzero_keys) = self.create_new_grid();
        self.number_of_refined_intervals = number_of_nonzero_keys;
        // mock data
        //   self.initial_guess = DMatrix::from_column_slice(self.values.len(), self.n_steps, self.y.to_DVectorType().as_slice());
        //  let binding = self.x_mesh.clone();
        //  let new_mesh = binding.data.as_vec();

        self.jac_recalc = true; // to avoid using old (low dimension) jacobian with new data
        self.n_steps = new_mesh.len() - 1;
        self.initial_guess = initial_guess;
        self.grid_refinemens += 1;
        info!(
            "\n \n grid refinement counter = {} \n \n",
            self.grid_refinemens
        );
        *self
            .calc_statistics
            .entry("number of grid refinements".to_string())
            .or_insert(0) += 1;
        // here we go again... running the code wtih new grid
        self.eq_generate(Some(new_mesh.clone()), Some(self.bandwidth));
        //               bandwidth doesnt change with gerid recalc so we pass into function
        // previously calculated  bandwidth to avoid O(N^2) calculation of bandwidth
        self.we_need_refinement();

        self.main_loop_damped();
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                       main functions to start the solver
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  // main function to solve the system of equations
    //Newton-Raphson method
    // realize iteration of Newton-Raphson - calculate new iteration vector by using Jacobian matrix
    pub fn solver(&mut self) -> Option<DVector<f64>> {
        self.eq_generate(None, None);
        let begin = Instant::now();
        let res = self.main_loop_damped();
        let end = begin.elapsed();
        elapsed_time(end);
        let time = end.as_secs_f64() as usize;
        self.calc_statistics
            .insert("time elapsed, s".to_string(), time);
        self.calc_statistics();
        res
    }
    // wrapper around solver function to implement logging
    pub fn solve(&mut self) -> Option<DVector<f64>> {
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
        let date_and_time = Local::now().format("%Y-%m-%d_%H-%M-%S");
        let name = format!("log_{}.txt", date_and_time);
        let logger_instance = CombinedLogger::init(vec![
            TermLogger::new(
                log_option,
                Config::default(),
                TerminalMode::Mixed,
                ColorChoice::Auto,
            ),
            WriteLogger::new(log_option, Config::default(), File::create(name).unwrap()),
        ]);

        match logger_instance {
            Ok(()) => {
                let res = self.solver();
                info!(" \n \n Program ended");
                res
            }
            Err(_) => {
                let res = self.solver();
                res
            }
        }
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                     functions to return and save result in different formats
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    pub fn save_to_file(&self, filename: Option<String>) {
        //let date_and_time = Local::now().format("%Y-%m-%d_%H-%M-%S");
        let name = if let Some(name) = filename {
            format!("{}.txt", name)
        } else {
            "result.txt".to_string()
        };
        let result_DMatrix = self.get_result().unwrap();
        let _ = save_matrix_to_file(&result_DMatrix, &self.values, &name);
    }

    pub fn get_result(&self) -> Option<DMatrix<f64>> {
        let number_of_Ys = self.values.len();
        let n_steps = self.n_steps;
        let vector_of_results = self.result.clone().unwrap().clone();
        let matrix_of_results: DMatrix<f64> =
            DMatrix::from_column_slice(number_of_Ys, n_steps, vector_of_results.clone().as_slice());

        let full_results = construct_full_solution(
            matrix_of_results,
            &self.BorderConditions,
            &self.variable_string,
            &self.values.clone(),
        );

        let permutted_results = interchange_columns(
            full_results.transpose(),
            self.values.clone(),
            self.variable_string.clone(),
        );

        Some(permutted_results)
    }

    pub fn plot_result(&self) {
        let number_of_Ys = self.values.len();
        let n_steps = self.n_steps;
        let vector_of_results = self.result.clone().unwrap().clone();
        let matrix_of_results: DMatrix<f64> =
            DMatrix::from_column_slice(number_of_Ys, n_steps, vector_of_results.clone().as_slice());

        for _col in matrix_of_results.column_iter() {
            //   println!( "{:?}", DVector::from_column_slice(_col.as_slice()) );
        }

        let full_results = construct_full_solution(
            matrix_of_results,
            &self.BorderConditions,
            &self.variable_string,
            &self.values.clone(),
        )
        .transpose();
        let permutted_results = interchange_columns(
            full_results.transpose(),
            self.values.clone(),
            self.variable_string.clone(),
        );

        info!(
            "matrix of results has shape {:?}",
            permutted_results.shape()
        );
        info!("length of x mesh : {:?}", n_steps);
        info!("number of Ys: {:?}", number_of_Ys);
        plots(
            self.arg.clone(),
            self.values.clone(),
            self.x_mesh.clone(),
            permutted_results.transpose(),
        );
        info!("result plotted");
    }

    fn calc_statistics(&self) {
        let mut stats = self.calc_statistics.clone();
        if let Some(jac) = &self.old_jac {
            let jac_shape = self.old_jac.as_ref().unwrap().shape();
            let matrix_weight = checkmem(&**jac);
            stats.insert("jacobian memory, MB".to_string(), matrix_weight as usize);
            stats.insert(
                "number of jacobian elements".to_string(),
                jac_shape.0 * jac_shape.1,
            );
        }
        stats.insert("length of y vector".to_string(), self.y.len() as usize);
        stats.insert(
            "number of grid points".to_string(),
            self.x_mesh.len() as usize,
        );
        let mut table = Builder::from(stats).build();
        table.with(Style::modern_rounded());
        info!("\n \n CALC STATISTICS \n \n {}", table.to_string());
    }
}

/* */
