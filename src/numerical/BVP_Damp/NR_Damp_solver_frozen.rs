use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions::Jacobian;

use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;
use std::time::Instant;

use crate::numerical::plots::plots;

use nalgebra::sparse::CsMatrix;
use sprs::{CsMat, CsVec};

use crate::numerical::BVP_Damp::BVP_traits::{
    Fun, FunEnum, Jac, JacEnum, MatrixType, VectorType, Vectors_type_casting,
};
use crate::numerical::BVP_Damp::BVP_utils::*;
use faer::sparse::SparseColMat;

use faer::col::Col;

pub struct NRBVP {
    pub eq_system: Vec<Expr>,
    pub initial_guess: DMatrix<f64>,
    pub values: Vec<String>,
    pub arg: String,
    pub BorderConditions: HashMap<String, (usize, f64)>,
    pub t0: f64,
    pub t_end: f64,
    pub n_steps: usize,
    pub strategy: String,
    pub strategy_params: Option<HashMap<String, Option<Vec<f64>>>>,
    pub linear_sys_method: Option<String>,
    pub method: String,
    pub tolerance: f64,
    pub max_iterations: usize,
    pub max_error: f64,
    pub result: Option<DVector<f64>>,
    pub x_mesh: DVector<f64>,

    pub fun: Box<dyn Fun>,
    pub jac: Option<Box<dyn Jac>>,
    pub p: f64,
    pub y: Box<dyn VectorType>,
    m: usize, // iteration counter without jacobian recalculation
    old_jac: Option<Box<dyn MatrixType>>,
    jac_recalc: bool,
    error_old: f64,
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
        strategy: String,
        strategy_params: Option<HashMap<String, Option<Vec<f64>>>>,
        linear_sys_method: Option<String>,
        method: String,
        tolerance: f64,        // tolerance
        max_iterations: usize, // max number of iterations
        max_error: f64,        // max error
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
        NRBVP {
            eq_system,
            initial_guess: initial_guess.clone(),
            values,
            arg,
            BorderConditions,
            t0,
            t_end,
            n_steps,
            tolerance,
            strategy,
            strategy_params,
            linear_sys_method,
            method,
            max_iterations,
            max_error,
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
        }
    }
    /// Basic methods to set the equation system

    ///Set system of equations with vector of symbolic expressions
    pub fn task_check(&self) {
        if self.t_end < self.t0 {
            panic!("t_end must be greater than t0");
        }

        if self.n_steps < 1 {
            panic!("n_steps must be greater than 1");
        }
        if self.max_iterations < 1 {
            panic!("max_iterations must be greater than 1");
        }
        let (m, n) = self.initial_guess.shape();
        if m != self.values.len() {
            panic!(
                "m must be equal to the length of the argument, m= {}, arg = {}",
                m,
                self.arg.len()
            );
        }
        if n != self.n_steps {
            panic!("n must be equal to the number of steps");
        }
        if self.tolerance < 0.0 {
            panic!("tolerance must be greater than 0.0");
        }
        if self.max_error < 0.0 {
            panic!("max_error must be greater than 0.0");
        }
        if self.BorderConditions.is_empty() {
            panic!("BorderConditions must be specified");
        }
        if self.BorderConditions.len() != self.values.len() {
            panic!("BorderConditions must be specified for each value");
        }
    }

    pub fn eq_generate(&mut self) {
        self.task_check();
        strategy_check(&self.strategy, &self.strategy_params);
        let mut jacobian_instance = Jacobian::new();
        let h = (self.t_end - self.t0) / self.n_steps as f64;
        match self.method.as_str() {
            "Dense" => {
                jacobian_instance.generate_BVP(
                    self.eq_system.clone(),
                    self.values.clone(),
                    self.arg.clone(),
                    self.t0.clone(),
                    None,
                    Some(self.n_steps),
                    Some(h),
                    None,
                    self.BorderConditions.clone(),
                    None,
                    None,
                );

                //     println!("Jacobian = {:?}", jacobian_instance.readable_jacobian);
                let fun: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>> =
                    jacobian_instance.lambdified_functions_IVP_DVector;

                let jac = jacobian_instance.function_jacobian_IVP_DMatrix;

                let jac_wrapped: Box<dyn FnMut(f64, &DVector<f64>) -> DMatrix<f64>> =
                    Box::new(move |t: f64, y: &DVector<f64>| -> DMatrix<f64> { jac(t, &y) });

                let boxed_fun: Box<dyn Fun> = Box::new(FunEnum::Dense(fun));
                self.fun = boxed_fun;
                let boxed_jac: Box<dyn Jac> = Box::new(JacEnum::Dense(jac_wrapped));
                self.jac = Some(boxed_jac);
            } // end of method Dense

            "Sparse 1" => {
                jacobian_instance.generate_BVP_CsMat(
                    self.eq_system.clone(),
                    self.values.clone(),
                    self.arg.clone(),
                    self.t0.clone(),
                    None,
                    Some(self.n_steps),
                    Some(h),
                    None,
                    self.BorderConditions.clone(),
                    None,
                    None,
                );

                //     println!("Jacobian = {:?}", jacobian_instance.readable_jacobian);
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
                println!("test = {:?}", test);
                // panic!("test");
                //
                let boxed_fun: Box<dyn Fun> = Box::new(FunEnum::Sparse_1(fun));
                self.fun = boxed_fun;
                let boxed_jac: Box<dyn Jac> = Box::new(JacEnum::Sparse_1(jac_wrapped));
                self.jac = Some(boxed_jac);
            }
            "Sparse 2" => {
                panic!("method not ready");
                jacobian_instance.generate_BVP_CsMatrix(
                    self.eq_system.clone(),
                    self.values.clone(),
                    self.arg.clone(),
                    self.t0.clone(),
                    None,
                    Some(self.n_steps),
                    Some(h),
                    None,
                    self.BorderConditions.clone(),
                    None,
                    None,
                );

                //     println!("Jacobian = {:?}", jacobian_instance.readable_jacobian);
                let fun: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>> =
                    jacobian_instance.lambdified_functions_IVP_DVector;

                let jac = jacobian_instance.function_jacobian_IVP_CsMatrix;

                let jac_wrapped: Box<dyn FnMut(f64, &DVector<f64>) -> CsMatrix<f64>> =
                    Box::new(move |t: f64, y: &DVector<f64>| -> CsMatrix<f64> { jac(t, &y) });

                let boxed_fun: Box<dyn Fun> = Box::new(FunEnum::Sparse_2(fun));
                self.fun = boxed_fun;
                let boxed_jac: Box<dyn Jac> = Box::new(JacEnum::Sparse_2(jac_wrapped));
                self.jac = Some(boxed_jac);
            }
            "Sparse" => {
                jacobian_instance.generate_BVP_SparseColMat(
                    self.eq_system.clone(),
                    self.values.clone(),
                    self.arg.clone(),
                    self.t0.clone(),
                    None,
                    Some(self.n_steps),
                    Some(h),
                    None,
                    self.BorderConditions.clone(),
                    None,
                    None,
                );

                //     println!("Jacobian = {:?}", jacobian_instance.readable_jacobian);
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
            }
            _ => {
                println!("Method not implemented");
                std::process::exit(1);
            }
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

    ///Newton-Raphson method
    /// realize iteration of Newton-Raphson - calculate new iteration vector by using Jacobian matrix
    pub fn iteration(&mut self) -> Box<dyn VectorType> {
        let p = self.p;
        let y = &*self.y;
        let fun = &self.fun;
        let new_fun = fun.call(p, y);
        let jac = self.jac.as_mut().unwrap();
        let now = Instant::now();

        let new_j = if self.jac_recalc {
            println!("\n \n JACOBIAN (RE)CALCULATED! \n \n");
            let new_j = jac.call(p, y);
            // println!(" \n \n new_j = {:?} ", jac_rowwise_printing(&*&new_j) );
            self.old_jac = Some(new_j.clone_box());
            self.m = 0;
            new_j
        } else {
            self.m = self.m + 1;
            self.old_jac.as_ref().unwrap().clone_box()
        };

        //   println!("new fun = {:?}", &new_fun);
        let delta: Box<dyn VectorType> = new_j.solve_sys(
            &*new_fun,
            self.linear_sys_method.clone(),
            self.tolerance,
            self.max_iterations,
            y,
        );
        let elapsed = now.elapsed();
        elapsed_time(elapsed);
        //  println!(" \n \n dy= {:?}", &delta);
        // element wise subtraction
        let new_y: Box<dyn VectorType> = y - &*delta;

        new_y
    }
    // main function to solve the system of equations

    pub fn main_loop(&mut self) -> Option<DVector<f64>> {
        println!("solving system of equations with Newton-Raphson method! \n \n");
        let y: DMatrix<f64> = self.initial_guess.clone();
        let y: Vec<f64> = y.iter().cloned().collect();
        let y: DVector<f64> = DVector::from_vec(y);
        //  self.y = Box::new(y.clone());
        self.y = Vectors_type_casting(&y.clone(), self.method.clone());

        //self.y = Box::new(y_);
        //  println!("y = {:?}", &y);
        let mut i = 0;

        while i < self.max_iterations {
            let _y = &self.y;

            let new_y = self.iteration();
            let y1 = new_y.subtract(&*self.y);
            let dy: Box<dyn VectorType> = y1.clone_box();

            let error = dy.norm();
            self.jac_recalc = frozen_jac_recalc(
                &self.strategy,
                &self.strategy_params,
                &self.old_jac,
                self.m,
                error,
                self.error_old,
            );
            self.error_old = error;
            //    println!("new_x = {:?} \n \n, x = {:?} \n \n ", &new_y.clone(), &_y );
            println!(" \n \n error = {:?} \n \n", &error);
            if error < self.tolerance {
                println!("converged in {} iterations, error = {}", i, error);
                self.result = Some(new_y.to_DVectorType());
                self.max_error = error;
                return Some(new_y.to_DVectorType());
            } else {
                //   _y = new_y.clone();
                let new_y: Box<dyn VectorType> = new_y.clone_box(); //Box::new(new_y);
                self.y = new_y;
                i += 1;
                //   println!("iteration = {}, error = {}, tol = {} \n \n", i, error, self.tolerance );
            }
        }
        None
    }
    pub fn solve(&mut self) -> Option<DVector<f64>> {
        self.eq_generate();
        let begin = Instant::now();
        let res = self.main_loop();
        let end = begin.elapsed();
        elapsed_time(end);
        res
    }

    pub fn get_result(&self) -> Option<DVector<f64>> {
        self.result.clone()
    }

    pub fn plot_result(&self) {
        let number_of_Ys = self.values.len();
        let n_steps = self.n_steps;
        let vector_of_results = self.result.clone().unwrap().clone();
        let matrix_of_results: DMatrix<f64> =
            DMatrix::from_column_slice(number_of_Ys, n_steps, vector_of_results.clone().as_slice())
                .transpose();
        for _col in matrix_of_results.column_iter() {
            //   println!( "{:?}", DVector::from_column_slice(_col.as_slice()) );
        }
        println!(
            "matrix of results has shape {:?}",
            matrix_of_results.shape()
        );
        println!("length of x mesh : {:?}", n_steps);
        println!("number of Ys: {:?}", number_of_Ys);
        plots(
            self.arg.clone(),
            self.values.clone(),
            self.x_mesh.clone(),
            matrix_of_results,
        );
        println!("result plotted");
    }
}

/* */

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_newton_raphson_solver() {
        // Define a simple equation: x^2 - 4 = 0
        let eq1 = Expr::parse_expression("y-z");
        let eq2 = Expr::parse_expression("-z");
        let eq_system = vec![eq1, eq2];

        let values = vec!["z".to_string(), "y".to_string()];
        let arg = "x".to_string();
        let tolerance = 1e-2;
        let max_iterations = 100;
        let max_error = 0.0;
        let t0 = 0.0;
        let t_end = 1.0;
        let n_steps = 100;
        let method = "Dense".to_string();
        let strategy = "Naive".to_string();
        let ones = vec![1.0; values.len() * n_steps];
        let initial_guess: DMatrix<f64> =
            DMatrix::from_column_slice(values.len(), n_steps, DVector::from_vec(ones).as_slice());
        let mut BorderConditions = HashMap::new();
        BorderConditions.insert("z".to_string(), (0usize, 1.0f64));
        BorderConditions.insert("y".to_string(), (1usize, 1.0f64));
        assert_eq!(&eq_system.len(), &2);
        let mut nr = NRBVP::new(
            eq_system,
            initial_guess,
            values,
            arg,
            BorderConditions,
            t0,
            t_end,
            n_steps,
            strategy,
            None,
            None,
            method,
            tolerance,
            max_iterations,
            max_error,
        );
        nr.eq_generate();

        assert_eq!(nr.eq_system.len(), 2);

        // Solve the equation at t=0 with initial guess y=[2.0]
        //    nr.set_new_step(0.0, DVector::from_element(1, 2.0), DVector::from_element(1, 2.0));
        let _solution = nr.solve().unwrap();
        /*
        // Check if the solution is close to the expected value
        let expected_solution = DVector::from_vec(vec![3.0, 4.0]);
        assert!((solution - expected_solution).norm() < tolerance);
         */
    }
}
