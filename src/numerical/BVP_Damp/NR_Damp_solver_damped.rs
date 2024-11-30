/*
Modified Newton method or Damped Newton method for solving a system of nonlinear equations.
This code implements a modified Newton method for solving a system of non-linear boundary value problems..
The code mostly inspired by sources listed below:
-  Cantera MultiNewton solver (MultiNewton.cpp )
- TWOPNT fortran solver (see "The Twopnt Program for Boundary Value Problems" by J. F. Grcar and Chemkin Theory Manual p.261)
*/
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions::Jacobian;
use chrono::Local;
use nalgebra::{DMatrix, DVector};

use crate::numerical::BVP_Damp::BVP_traits::{
    Fun, FunEnum, Jac, JacEnum, MatrixType, VectorType, Vectors_type_casting,
};
use crate::numerical::BVP_Damp::BVP_utils::elapsed_time;
use crate::numerical::BVP_Damp::BVP_utils_damped::{
    bound_step, convergence_condition, if_initial_guess_inside_bounds, interchange_columns,
    jac_recalc,
};
use crate::Utils::logger::save_matrix_to_file;
use crate::Utils::plots::plots;
use std::collections::HashMap;
use std::fs::File;
use std::time::Instant;

use crate::numerical::BVP_Damp::grid_api::{new_grid, GridRefinementMethod};
use faer::col::Col;
use faer::sparse::SparseColMat;
use log::{info,warn,error};
use nalgebra::sparse::CsMatrix;
use sprs::{CsMat, CsVec};
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
    adaptive: bool,              // flag indicating if adaptive grid should be used
    new_grid_enabled: bool,      //flag indicating if the grid should be refined
    grid_refinemens: usize,      //
    number_of_refined_intervals: usize, //number of refined intervals
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
    ) -> NRBVP {
        //jacobian: Jacobian, initial_guess: Vec<f64>, tolerance: f64, max_iterations: usize, max_error: f64, result: Option<Vec<f64>>
        let y0 = Box::new(DVector::from_vec(vec![0.0, 0.0]));

        let fun0: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>> =
            Box::new(|_x, y: &DVector<f64>| y.clone());
        let boxed_fun: Box<dyn Fun> = Box::new(FunEnum::Dense(fun0));
        let h = (t_end - t0) / (n_steps - 1) as f64;
        let T_list: Vec<f64> = (0..n_steps)
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
        }
    }
    /// Basic methods to set the equation system

    // check if user specified task is correct
    pub fn task_check(&self) {
        assert_eq!(
            self.initial_guess.len(),
            self.n_steps * self.values.len(),
            "lenght of initial guess should be equal to n_steps*values"
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
        if_initial_guess_inside_bounds(&self.initial_guess, &self.Bounds, self.values.clone());
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
    pub fn eq_generate(&mut self, mesh_: Option<Vec<f64>>) {
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
        match self.method.as_str() {
            "Dense" => {
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
                );

                //     info("Jacobian = {:?}", jacobian_instance.readable_jacobian);
                let fun: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>> =
                    jacobian_instance.lambdified_functions_IVP_DVector;

                let jac = jacobian_instance.function_jacobian_IVP_DMatrix;

                let jac_wrapped: Box<dyn FnMut(f64, &DVector<f64>) -> DMatrix<f64>> =
                    Box::new(move |t: f64, y: &DVector<f64>| -> DMatrix<f64> { jac(t, &y) });

                let boxed_fun: Box<dyn Fun> = Box::new(FunEnum::Dense(fun));
                self.fun = boxed_fun;
                let boxed_jac: Box<dyn Jac> = Box::new(JacEnum::Dense(jac_wrapped));
                self.jac = Some(boxed_jac);
                self.bounds_vec = jacobian_instance.bounds.unwrap();
                self.rel_tolerance_vec = jacobian_instance.rel_tolerance_vec.unwrap();
                self.variable_string = jacobian_instance.variable_string;
            } // end of method Dense

            "Sparse 1" => {
                jacobian_instance.generate_BVP_CsMat(
                    self.eq_system.clone(),
                    self.values.clone(),
                    self.arg.clone(),
                    self.t0.clone(),
                    None,
                    n_steps,
                    h,
                    mesh,
                    self.BorderConditions.clone(),
                    self.Bounds.clone(),
                    self.rel_tolerance.clone(),
                    scheme.clone(),
                );

                //     info("Jacobian = {:?}", jacobian_instance.readable_jacobian);
                let fun: Box<dyn Fn(f64, &CsVec<f64>) -> CsVec<f64>> =
                    jacobian_instance.lambdified_functions_IVP_CsVec;

                let jac = jacobian_instance.function_jacobian_IVP_CsMat;

                let jac_wrapped: Box<dyn FnMut(f64, &CsVec<f64>) -> CsMat<f64>> =
                    Box::new(move |t: f64, y: &CsVec<f64>| -> CsMat<f64> { jac(t, &y) });
                // test
                let y: DMatrix<f64> = self.initial_guess.clone();
                let y: Vec<f64> = y.iter().cloned().collect();
                let y: DVector<f64> = DVector::from_vec(y);
                let y_0 = Vectors_type_casting(&y.clone(), "Sparse 1".to_string());
                let y_0 = y_0.as_any().downcast_ref::<CsVec<f64>>().unwrap();
                let test = fun(self.p, &y_0.clone());
                info!("test = {:?}", test);
                // panic!("test");
                //
                let boxed_fun: Box<dyn Fun> = Box::new(FunEnum::Sparse_1(fun));
                self.fun = boxed_fun;
                let boxed_jac: Box<dyn Jac> = Box::new(JacEnum::Sparse_1(jac_wrapped));
                self.jac = Some(boxed_jac);
                self.bounds_vec = jacobian_instance.bounds.unwrap();
                self.rel_tolerance_vec = jacobian_instance.rel_tolerance_vec.unwrap();
                self.variable_string = jacobian_instance.variable_string;
            }
            "Sparse 2" => {
                panic!("method not ready");
                jacobian_instance.generate_BVP_CsMatrix(
                    self.eq_system.clone(),
                    self.values.clone(),
                    self.arg.clone(),
                    self.t0.clone(),
                    None,
                    n_steps,
                    h,
                    mesh,
                    self.BorderConditions.clone(),
                    self.Bounds.clone(),
                    self.rel_tolerance.clone(),
                    scheme.clone(),
                );

                //     info("Jacobian = {:?}", jacobian_instance.readable_jacobian);
                let fun: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>> =
                    jacobian_instance.lambdified_functions_IVP_DVector;

                let jac = jacobian_instance.function_jacobian_IVP_CsMatrix;

                let jac_wrapped: Box<dyn FnMut(f64, &DVector<f64>) -> CsMatrix<f64>> =
                    Box::new(move |t: f64, y: &DVector<f64>| -> CsMatrix<f64> { jac(t, &y) });

                let boxed_fun: Box<dyn Fun> = Box::new(FunEnum::Sparse_2(fun));
                self.fun = boxed_fun;
                let boxed_jac: Box<dyn Jac> = Box::new(JacEnum::Sparse_2(jac_wrapped));
                self.jac = Some(boxed_jac);
                self.bounds_vec = jacobian_instance.bounds.unwrap();
                self.rel_tolerance_vec = jacobian_instance.rel_tolerance_vec.unwrap();
                self.variable_string = jacobian_instance.variable_string;
            }
            "Sparse" => {
                jacobian_instance.generate_BVP_SparseColMat(
                    self.eq_system.clone(),
                    self.values.clone(),
                    self.arg.clone(),
                    self.t0.clone(),
                    None,
                    n_steps,
                    h,
                    mesh,
                    self.BorderConditions.clone(),
                    self.Bounds.clone(),
                    self.rel_tolerance.clone(),
                    scheme.clone(),
                );

                //     info("Jacobian = {:?}", jacobian_instance.readable_jacobian);
                let fun: Box<dyn Fn(f64, &Col<f64>) -> Col<f64>> =
                    jacobian_instance.lambdified_functions_IVP_Col;

                let jac = jacobian_instance.function_jacobian_IVP_SparseColMat;

                let jac_wrapped: Box<dyn FnMut(f64, &Col<f64>) -> SparseColMat<usize, f64>> =
                    Box::new(move |t: f64, y: &Col<f64>| -> SparseColMat<usize, f64> {
                        jac(t, &y)
                    });

                let boxed_fun: Box<dyn Fun> = Box::new(FunEnum::Sparse_3(fun));
                self.fun = boxed_fun;
                let boxed_jac: Box<dyn Jac> = Box::new(JacEnum::Sparse_3(jac_wrapped));
                self.jac = Some(boxed_jac);
                self.bounds_vec = jacobian_instance.bounds.unwrap();
                self.rel_tolerance_vec = jacobian_instance.rel_tolerance_vec.unwrap();
                self.variable_string = jacobian_instance.variable_string;
            }
            _ => {
                info!("Method not implemented");
                std::process::exit(1);
            } // assert_eq!(self.bo)
        } // end of match
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
            info!("JACOBIAN (RE)CALCULATED!");
            let jac_function = self.jac.as_mut().unwrap();
            let jac_matrix = jac_function.call(p, y);
            let inv_J_k = jac_function.inv(&*jac_matrix, self.abs_tolerance, self.max_iterations);
            self.old_jac = Some(inv_J_k);
            self.m = 0;
        }
    }
    ////////////////////////////////////////////////////////////////

    // jacobian recalculation
    fn recalc_jacobian(&mut self) {
        if self.jac_recalc {
            let p = self.p;
            let y = &*self.y;
            info!("JACOBIAN (RE)CALCULATED!");
            let jac_function = self.jac.as_mut().unwrap();
            let jac_matrix = jac_function.call(p, y);
            // info(" new_j = {:?} ", jac_rowwise_printing(&*&new_j) );
            self.old_jac = Some(jac_matrix);
            self.m = 0;
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
        info!("residual norm = {:?} ", residual_norm);
        // jac_rowwise_printing(&J_k);
        //    info(" F_k = {:?} ", F_k.to_DVectorType());
        for el in F_k.iterate() {
            if el.is_nan() {
                error!("NaN in undamped step residual function");
                panic!()
            }
        }
        // solving equation J_k*dy_k=-F_k for undamped dy_k, but Lambda*dy_k - is dumped step
        let undamped_step_k: Box<dyn VectorType> = J_k.solve_sys(
            &*F_k,
            self.linear_sys_method.clone(),
            self.abs_tolerance,
            self.max_iterations,
            y,
        );
        for el in undamped_step_k.iterate() {
            if el.is_nan() {
                error!("NaN in damped step deltaY ");
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
        let y_k = y_k_minus_1 - &*undamped_step_k_minus_1;
        let fbound = bound_step(y_k_minus_1, &*undamped_step_k_minus_1, &self.bounds_vec);
        if fbound.is_nan() {
            error!("fbound is NaN");
            panic!()
        }
        if fbound.is_infinite() {
            error!("fbound is infinite");
            panic!()
        }
        // let fbound =1.0;
        info!("fboundary  = {}", fbound);
        let mut lambda = 1.0 * fbound;
        // if fbound is very small, then x0 is already close to the boundary and
        // step0 points out of the allowed domain. In this case, the Newton
        // algorithm fails, so return an error condition.
        if fbound < 1e-10 {
            warn!(
                "No damped step can be taken without violating solution component bounds."
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
        //
        for mut k in 0..maxDampIter {
            if k > 1 {
                info!("damped_step number {} ", k);
            }
            info!("Damping coefficient = {}", lambda);
            let damped_step_k = undamped_step_k.mul_float(lambda);
            let S_k = &damped_step_k.norm();

            let y_k_plus_1: Box<dyn VectorType> = &*y_k - &*damped_step_k;

            // / compute the next undamped step that would result if x1 is accepted
            // J(x_k)^-1 F(x_k+1)
            let undamped_step_k_plus_1 = self.step(p, &*y_k_plus_1); //???????????????
            let error = &undamped_step_k_plus_1.norm();
            self.error_old = *error;
            info!("L2 norm of undamped step = {}", error);
            let convergence_cond_for_step = convergence_condition(
                &*undamped_step_k_plus_1,
                &self.abs_tolerance,
                &self.rel_tolerance_vec,
            );
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
                info!(" Damping coefficient found (solution has not converged yet)");
                info!(
                    " step norm =  {}, weight norm = {}, convergence condition = {}",
                    self.error_old,
                    S_k_plus_1.unwrap(),
                    conv
                );
                (0, damped_step_result)
            } else {
                info!(" Damping coefficient found (solution has converged)");
                info!(
                    "step norm =  {}, weight norm = {}, convergence condition = {}",
                    self.error_old,
                    S_k_plus_1.unwrap(),
                    conv
                );
                (1, damped_step_result)
            }
        } else {
            //  if we have reached max damping iterations without finding a damping coefficient we must reject the step
            warn!("No damping coefficient found (max damping iterations reached)");
            (-2, None)
        }
    } // end of damped step
    pub fn main_loop_damped(&mut self) -> Option<DVector<f64>> {
        info!("solving system of equations with Newton-Raphson method!");
        let y: DMatrix<f64> = self.initial_guess.clone();
        let y: Vec<f64> = y.iter().cloned().collect();
        let y: DVector<f64> = DVector::from_vec(y);

        self.y = Vectors_type_casting(&y.clone(), self.method.clone());

        // info("y = {:?}", &y);
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
            let (status, damped_step_result) = self.damped_step();

            if status == 0 {
                // status == 0 means convergence is not reached yet we're going to another iteration
                let y_k_plus_1 = if let Some(y_k_plus_1) = damped_step_result {
                    y_k_plus_1
                } else {
                    error!("y_k_plus_1 is None");
                    panic!()
                };
                self.y = y_k_plus_1;
                self.jac_recalc = false;
            }
            // status == 0
            else if status == 1 {
                // status == 1 means convergence is reached, save the result
                info!("Solution has converged, breaking the loop!");
                let y_k_plus_1 = if let Some(y_k_plus_1) = damped_step_result {
                    y_k_plus_1
                } else {
                    panic!(" y_k_plus_1 is None")
                };
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
                    self.solve_with_new_grid()
                } else {
                    // if adapive is None then we just return the result
                    let result = Some(y_k_plus_1.to_DVectorType()); // save the result in the format of DVector
                    self.result = result.clone();
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
                        "status <0, recalculating Jacobian flag up! Jacobian age = {} ",
                        self.m
                    );
                    if nJacReeval > 3 {
                        break;
                    }
                    nJacReeval += 1;
                } else {
                    info!("Jacobian age {} =<1 ", self.m);
                    break;
                }
            } // status <0

            info!("end of iteration {} with jac age {} ", i, self.m);
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
                    info!(
                        "number of marked intervals is 0, no new grid is needed "
                    );
                    false
                } else {
                    info!(
                        "number of marked intervals is {}, new grid is needed ",
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
                max_grid_refinenents,
                self.grid_refinemens
            );
            res = false;
        }
        self.new_grid_enabled = res;
    }
    fn create_new_grid(&mut self) -> (Vec<f64>, DMatrix<f64>, usize) {
        info!("creating new grid ");
        let y = self.y.clone_box();
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
            if res.len() != 2 {
                panic!("this strategy requires two parameters")
            };
            let method = GridRefinementMethod::GrcarSmooke;
            (res, method)
        } else {
            panic!("parameters for adaptive strategy not found")
        };
        //create a new mesh with a chosen algorithm a
        // API of new grid returns a new mesh, initial guess and number of intervals that doesnt meet the criteria and wac subdivided
        // if number_of_nonzero_keys==0 it means that no need to create a new grid
        let (new_mesh, initial_guess, number_of_nonzero_keys) =
            new_grid(method, &y_DMatrix, &self.x_mesh, vector_of_params.clone());

        info!("new grid enabled! ");
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
        self.n_steps = new_mesh.len();
        self.initial_guess = initial_guess;
        self.grid_refinemens += 1;
        info!(
            "grid refinement counter = {} ",
            self.grid_refinemens
        );

        // here we go again... running the code wtih new grid
        self.eq_generate(Some(new_mesh.clone()));
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
        // TODO! сравнить явный мэш с неявным
        // let test_mesh = Some((0..100).map(|x| 0.01 * x as f64).collect::<Vec<f64>>());
        self.eq_generate(None);
        let begin = Instant::now();
        let res = self.main_loop_damped();
        let end = begin.elapsed();
        elapsed_time(end);

        res
    }
    // wrapper around solver function to implement logging
    pub fn solve(&mut self) -> Option<DVector<f64>> {
        let date_and_time = Local::now().format("%Y-%m-%d_%H-%M-%S");
        let name = format!("log_{}.txt", date_and_time);
        let res = self.solver();
        info!(" Program ended");
        res

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
    // function creates a new instance of solver but with new mesh

    pub fn get_result(&self) -> Option<DMatrix<f64>> {
        let number_of_Ys = self.values.len();
        let n_steps = self.n_steps;
        let vector_of_results = self.result.clone().unwrap().clone();
        let matrix_of_results: DMatrix<f64> =
            DMatrix::from_column_slice(number_of_Ys, n_steps, vector_of_results.clone().as_slice())
                .transpose();
        let permutted_results = interchange_columns(
            matrix_of_results,
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
            DMatrix::from_column_slice(number_of_Ys, n_steps, vector_of_results.clone().as_slice())
                .transpose();
        for _col in matrix_of_results.column_iter() {
            //   info!( "{:?}", DVector::from_column_slice(_col.as_slice()) );
        }
        info!(
            "matrix of results has shape {:?}",
            matrix_of_results.shape()
        );
        info!("length of x mesh : {:?}", n_steps);
        info!("number of Ys: {:?}", number_of_Ys);
        let permutted_results = interchange_columns(
            matrix_of_results,
            self.values.clone(),
            self.variable_string.clone(),
        );
        plots(
            self.arg.clone(),
            self.values.clone(),
            self.x_mesh.clone(),
            permutted_results,
        );
        info!("result plotted");
    }
}

/* */
