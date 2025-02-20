#![allow(non_camel_case_types)]

use crate::numerical::BVP_Damp::BVP_traits::{
    Fun, FunEnum, Jac, JacEnum, VectorType, MatrixType,  convert_to_fun, // convert_to_jac
};
use crate::numerical::BVP_Damp::BVP_utils::elapsed_time;
use crate::symbolic::symbolic_engine::Expr;
//use crate::symbolic::symbolic_traits::SymbolicType;
use faer::col::{from_slice, Col};
use faer::sparse::SparseColMat;
use log::info;
use nalgebra::sparse::CsMatrix;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use regex::Regex;
use sprs::{CsMat, CsVec};
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use tabled::{builder::Builder, settings::Style};

/*
 * This struct represents a Jacobian for BVPs, which is a matrix of partial derivatives of a vector function.
 * It contains the symbolic and numerical representations of the jacobian, as well as the functions used to evaluate it.
 * Fields:
   - vector_of_functions: A vector of symbolic functions/expressions representing the vector function.
   - lambdified_functions: A vector of lambdified functions (symbolic functions converted to rust functions) that can be used to evaluate the vector function.
   - vector_of_variables: A vector of symbolic variables representing the vector function.
   - variable_string: A vector of string representations of the variables.
   - symbolic_jacobian: A vector of vectors representing the symbolic jacobian of the vector function.
   - function_jacobian: A vector of vectors representing the functions used to evaluate the symbolic jacobian.
*/
//#[derive(Clone)]
pub struct Jacobian {
    pub vector_of_functions: Vec<Expr>, // vector of symbolic functions/expressions
    pub lambdified_functions: Vec<Box<dyn Fn(Vec<f64>) -> f64>>, // vector of lambdified functions (symbolic functions converted to rust functions)

    pub method: String,
    pub vector_of_variables: Vec<Expr>, // vector of symbolic variables
    pub variable_string: Vec<String>,   // vector of string representation of variables
    pub symbolic_jacobian: Vec<Vec<Expr>>, // vector of symbolic jacobian
    pub lambdified_jac_element: Option<Box<dyn Fn(f64, usize, usize) -> f64>>,

    pub jac_function: Option<Box<dyn Jac>>,

    pub residiual_function: Box<dyn Fun>,

    pub bounds: Option<Vec<(f64, f64)>>,
    pub rel_tolerance_vec: Option<Vec<f64>>,
    pub bandwidth: Option<(usize, usize)>,
    pub variables_for_all_disrete: Vec<Vec<String>>
}

impl Jacobian {
    pub fn new() -> Self {
        let fun0: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>> =
            Box::new(|_x, y: &DVector<f64>| y.clone());
        let boxed_fun: Box<dyn Fun> = Box::new(FunEnum::Dense(fun0));
        Self {
            vector_of_functions: Vec::new(),
            lambdified_functions: Vec::new(),
            method: String::new(),
            vector_of_variables: Vec::new(),
            variable_string: Vec::new(),
            symbolic_jacobian: Vec::new(),
            lambdified_jac_element: None,
            jac_function: None,
            residiual_function: boxed_fun,
            bounds: None,
            rel_tolerance_vec: None,
            bandwidth: None,
            variables_for_all_disrete: Vec::new()
        }
    }

    pub fn from_vectors(&mut self, vector_of_functions: Vec<Expr>, variable_string: Vec<String>) {
        self.vector_of_functions = vector_of_functions;
        self.variable_string = variable_string.clone();
        self.vector_of_variables =
            Expr::parse_vector_expression(variable_string.iter().map(|s| s.as_str()).collect());
        println!(" {:?}", self.vector_of_functions);
        println!(" {:?}", self.vector_of_variables);
    }

    // using paretial differentiation a jacobian is calculated from the vector of functions and variables
    pub fn calc_jacobian(&mut self) {
        assert!(
            !self.vector_of_functions.is_empty(),
            "vector_of_functions is empty"
        );
        assert!(
            !self.vector_of_variables.is_empty(),
            "vector_of_variables is empty"
        );

        let mut new_jac: Vec<Vec<Expr>> = Vec::new();
        let variable_string_vec = self.variable_string.clone();
        for i in 0..self.vector_of_functions.len() {
            let mut vector_of_partial_derivatives = Vec::new();
            for j in 0..self.vector_of_variables.len() {
                let mut partial = Expr::diff(
                    &self.vector_of_functions[i].clone(),
                    &variable_string_vec[j],
                );
                partial = partial.symplify();
                vector_of_partial_derivatives.push(partial);
            }
            new_jac.push(vector_of_partial_derivatives);
        }

        self.symbolic_jacobian = new_jac;
    }

    // parallel version
    pub fn calc_jacobian_parallel(&mut self) {
        assert!(
            !self.vector_of_functions.is_empty(),
            "vector_of_functions is empty"
        );
        assert!(
            !self.vector_of_variables.is_empty(),
            "vector_of_variables is empty"
        );

        let variable_string_vec = self.variable_string.clone();
        let new_jac: Vec<Vec<Expr>> = self
            .vector_of_functions
            .par_iter()
            .map(|function| {
                let mut vector_of_partial_derivatives = Vec::new();
               // let function = function.clone();
                for j in 0..self.vector_of_variables.len() {
                    let mut partial = Expr::diff(&function, &variable_string_vec[j]);
                    partial = partial.symplify();
                    vector_of_partial_derivatives.push(partial);
                }
                vector_of_partial_derivatives
            })
            .collect();

        self.symbolic_jacobian = new_jac;
    }
    //
    pub fn calc_jacobian_parallel_smart(&mut self) {
        assert!(
            !self.vector_of_functions.is_empty(),
            "vector_of_functions is empty"
        );
        assert!(
            !self.vector_of_variables.is_empty(),
            "vector_of_variables is empty"
        );

        let variable_string_vec = self.variable_string.clone();
        let new_jac: Vec<Vec<Expr>> = self
            .vector_of_functions
            .par_iter().enumerate()
            .map(|(i,function)| {
                let mut vector_of_partial_derivatives = Vec::new();
               // let function = function.clone();
                for j in 0..self.vector_of_variables.len() {
                    let variable = &variable_string_vec[j];// obviously if function does not contain variable its derivative should be 0
                    let list_of_vaiables_for_this_eq = &self.variables_for_all_disrete[i];// so we can only calculate derivative for variables that are used in this equation
                    if list_of_vaiables_for_this_eq.contains(variable) {
                    let mut partial = Expr::diff(&function, variable);
                    partial = partial.symplify();
                    vector_of_partial_derivatives.push(partial);
                    } else {
                        vector_of_partial_derivatives.push(Expr::Const(0.0));
                    }
                   
                }
                vector_of_partial_derivatives
            })
            .collect();

        self.symbolic_jacobian = new_jac;
    }


    ////////////////////////////////////////////////////////////////////////////////////
    //  GENERIC FUNCTIONS
    ////////////////////////////////////////////////////////////////////////////////////
    pub fn lambdify_residual_VectorType(&mut self, arg: &str, variable_str: Vec<&str>) {
        let vector_of_functions = &self.vector_of_functions;
        fn f(
            vector_of_functions: Vec<Expr>,
            _arg: String,
            variable_str: Vec<String>,
        ) -> Box<dyn Fn(f64, &dyn VectorType) -> Box<dyn VectorType>> {
            Box::new(move |_x: f64, v: &dyn VectorType| -> Box<dyn VectorType> {
                let mut result = v.zeros(vector_of_functions.len());
                
                // Iterate through functions and assign values
                for (i, func) in vector_of_functions.iter().enumerate() {
                    let func = Expr::lambdify_owned(
                        func.to_owned(),
                        variable_str
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .clone(),
                    );
                    let value = func(v.iterate().collect());
                    result = result.assign_value(i, value);
                }
                result
            })
        }
    
        let fun: Box<dyn Fn(f64, &dyn VectorType) -> Box<dyn VectorType>> = f(
            vector_of_functions.to_owned(),
            arg.to_string(),
            variable_str.iter().map(|s| s.to_string()).collect(),
        );
        let residual_function = convert_to_fun(fun);
        self.residiual_function = residual_function;
    }
    

    pub fn jacobian_generate_generic_par(
        jac: Vec<Vec<Expr>>,
        vector_of_functions_len: usize,
        vector_of_variables_len: usize,
        variable_str: Vec<String>,
        _arg: String,
        bandwidth: Option<(usize, usize)>,
    ) -> Box<dyn Fn(f64, &(dyn VectorType + Send + Sync) ) -> Box<dyn MatrixType>> {
        //let arg = arg.as_str();
        //let variable_str: Vec<&str> = variable_str.iter().map(|s| s.as_str()).collect();

        Box::new(move |_x: f64, v: &(dyn VectorType + Send + Sync) | -> Box<dyn MatrixType> {
            let mut vector_of_derivatives =
                vec![0.0; vector_of_functions_len * vector_of_variables_len];
            let vector_of_derivatives_mutex = std::sync::Mutex::new(&mut vector_of_derivatives);
            
            let mut vector_of_triplets = Vec::new();
            let  vector_of_triplets_mutex = std::sync::Mutex::new(&mut vector_of_triplets);
            (0..vector_of_functions_len).into_par_iter().for_each(|i| {
                let (right_border, left_border) = if let Some((kl, ku)) = bandwidth {
                    let right_border = std::cmp::min(i + ku + 1, vector_of_variables_len);
                    let left_border = if i as i32 - (kl as i32) - 1 < 0 {
                        0
                    } else {
                        i - kl - 1
                    };
                    (right_border, left_border)
                } else {
                    let right_border = vector_of_variables_len;
                    let left_border = 0;
                    (right_border, left_border)
                };
                for j in left_border..right_border {
                    // if jac[i][j] != Expr::Const(0.0) { println!("i = {}, j = {}, {}", i, j,  &jac[i][j]);}
                    let partial_func = Expr::lambdify_owned(
                        jac[i][j].clone(),
                        variable_str
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .clone(),
                    );

                    let v_vec: Vec<f64> = v.iterate().collect();
                    let _vec_type = &v.vec_type();
                    let P = partial_func(v_vec.clone());
                    if P.abs() > 1e-8 {
                        vector_of_derivatives_mutex.lock().unwrap()
                            [i * vector_of_functions_len + j] = P;
                            let triplet = (i, j, P);
                            vector_of_triplets_mutex.lock().unwrap().push(triplet);
                    }
         

                }
            }); //par_iter()

            let new_function_jacobian: Box<dyn MatrixType> =    v.from_vector(
                vector_of_functions_len,
                vector_of_variables_len,
                &vector_of_derivatives,
                vector_of_triplets
            );
            //  panic!("stop here");
           new_function_jacobian
        })// end of box
    } // end of function


    pub fn lambdify_jacobian_generic(&mut self, arg: &str, variable_str: Vec<&str>) {
        let symbolic_jacobian = self.symbolic_jacobian.clone();
        let symbolic_jacobian_rc = symbolic_jacobian.clone();

        let vector_of_functions_len = self.vector_of_functions.len();
        let vector_of_variables_len = self.vector_of_variables.len();
        let bandwidth = self.bandwidth;
        let _new_jac = Jacobian::jacobian_generate_generic_par(
            symbolic_jacobian_rc,
            vector_of_functions_len,
            vector_of_variables_len,
            variable_str.iter().map(|s| s.to_string()).collect(),
            arg.to_string(),
            bandwidth,
        );
        //  let mut boxed_jacobian: Box<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64>> = Box::new(|arg, variable_str_| {
        //    DMatrix::from_rows(&new_jac) }) ;
            // TODO! Send + Sync trait is not implemented
     //   let boxed_jac: Box<dyn Jac> = convert_to_jac(new_jac);
      //  self.jac_function = Some(boxed_jac);
    }

    ////////////////////////////////////////////////////////////////////////////////////
    ///                             NALGEBRA DENSE
    ////////////////////////////////////////////////////////////////////////////////////

    pub fn jacobian_generate_DMatrix_par(
        jac: Vec<Vec<Expr>>,
        vector_of_functions_len: usize,
        vector_of_variables_len: usize,
        variable_str: Vec<String>,
        _arg: String,
        bandwidth: Option<(usize, usize)>,
    ) -> Box<dyn FnMut(f64, &DVector<f64>) -> DMatrix<f64>> {
        //let arg = arg.as_str();
        //let variable_str: Vec<&str> = variable_str.iter().map(|s| s.as_str()).collect();

        Box::new(move |_x: f64, v: &DVector<f64>| -> DMatrix<f64> {
            let mut vector_of_derivatives =
                vec![0.0; vector_of_functions_len * vector_of_variables_len];
            let vector_of_derivatives_mutex = std::sync::Mutex::new(&mut vector_of_derivatives);
            (0..vector_of_functions_len).into_par_iter().for_each(|i| {
                let (right_border, left_border) = if let Some((kl, ku)) = bandwidth {
                    let right_border = std::cmp::min(i + ku + 1, vector_of_variables_len);
                    let left_border = if i as i32 - (kl as i32) - 1 < 0 {
                        0
                    } else {
                        i - kl - 1
                    };
                    (right_border, left_border)
                } else {
                    let right_border = vector_of_variables_len;
                    let left_border = 0;
                    (right_border, left_border)
                };
                for j in left_border..right_border {
                    // if jac[i][j] != Expr::Const(0.0) { println!("i = {}, j = {}, {}", i, j,  &jac[i][j]);}
                    let symbolic_partoal_derivative = &jac[i][j];
                    if !symbolic_partoal_derivative.is_zero() {
                        let partial_func = Expr::lambdify(
                            symbolic_partoal_derivative,
                            variable_str
                                .iter()
                                .map(|s| s.as_str())
                                .collect::<Vec<_>>()
                                .clone(),
                        );
                        let v_vec: Vec<f64> = v.iter().cloned().collect();

                        let P = partial_func(v_vec.clone());
                        if P.abs() > 1e-8 {
                            vector_of_derivatives_mutex.lock().unwrap()
                                [i * vector_of_functions_len + j] = P;
                        }// if P.abs() > 1e-8
                    }

                }
            }); //par_iter()

            let  new_function_jacobian: DMatrix<f64> = DMatrix::from_row_slice(
                vector_of_functions_len,
                vector_of_variables_len,
                &vector_of_derivatives,
            );
            //  panic!("stop here");
            new_function_jacobian
        })
    } // end of function

    pub fn jacobian_generate_DMatrix(
        jac: Vec<Vec<Expr>>,
        vector_of_functions_len: usize,
        vector_of_variables_len: usize,
        variable_str: Vec<String>,
        _arg: String,
        bandwidth: Option<(usize, usize)>,
    ) -> Box<dyn FnMut(f64, &DVector<f64>) -> DMatrix<f64>> {
        //let arg = arg.as_str();
        //let variable_str: Vec<&str> = variable_str.iter().map(|s| s.as_str()).collect();

        Box::new(move |_x: f64, v: &DVector<f64>| -> DMatrix<f64> {
            let mut new_function_jacobian: DMatrix<f64> =
                DMatrix::zeros(vector_of_functions_len, vector_of_variables_len);
            for i in 0..vector_of_functions_len {
                let (right_border, left_border) = if let Some((kl, ku)) = bandwidth {
                    let right_border = std::cmp::min(i + ku + 1, vector_of_variables_len);
                    let left_border = if i as i32 - (kl as i32) - 1 < 0 {
                        0
                    } else {
                        i - kl - 1
                    };
                    (right_border, left_border)
                } else {
                    let right_border = vector_of_variables_len;
                    let left_border = 0;
                    (right_border, left_border)
                };
                for j in left_border..right_border {
                    // println!("i = {}, j = {}", i, j);
                    //  let now = std::time::Instant::now();
                    let symbolic_partoal_derivative = &jac[i][j];
                    if !symbolic_partoal_derivative.is_zero() {
                        let partial_func = Expr::lambdify(
                            symbolic_partoal_derivative,
                            variable_str
                                .iter()
                                .map(|s| s.as_str())
                                .collect::<Vec<_>>()
                                .clone(),
                        );
                        let v_vec: Vec<f64> = v.iter().cloned().collect();
                        new_function_jacobian[(i, j)] = partial_func(v_vec.clone());
                    } else {
                        new_function_jacobian[(i, j)] = 0.0;
                       };
             
                    //    let time_test = now.elapsed().as_micros();
                    //   if time_test > 100 {println!("Elapsed time: {:?} micrs, {:?} ", now.elapsed().as_micros(), jac[i][j].clone() );}
                }
            }
            new_function_jacobian
        })
    } // end of function

    pub fn lambdify_jacobian_DMatrix(&mut self, arg: &str, variable_str: Vec<&str>) {
        let symbolic_jacobian = self.symbolic_jacobian.clone();
        let symbolic_jacobian_rc = symbolic_jacobian.clone();

        let vector_of_functions_len = self.vector_of_functions.len();
        let vector_of_variables_len = self.vector_of_variables.len();
        let bandwidth = self.bandwidth;
        let new_jac = Jacobian::jacobian_generate_DMatrix_par(
            symbolic_jacobian_rc,
            vector_of_functions_len,
            vector_of_variables_len,
            variable_str.iter().map(|s| s.to_string()).collect(),
            arg.to_string(),
            bandwidth,
        );
        //  let mut boxed_jacobian: Box<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64>> = Box::new(|arg, variable_str_| {
        //    DMatrix::from_rows(&new_jac) }) ;

        let boxed_jac: Box<dyn Jac> = Box::new(JacEnum::Dense(new_jac));
        self.jac_function = Some(boxed_jac);
    }

    pub fn lambdify_residual_DVector(&mut self, arg: &str, variable_str: Vec<&str>) {
        let vector_of_functions = &self.vector_of_functions;
        fn f(
            vector_of_functions: Vec<Expr>,
            _arg: String,
            variable_str: Vec<String>,
        ) -> Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64> + 'static> {
            Box::new(move |_x: f64, v: &DVector<f64>| -> DVector<f64> {
                let result: Vec<_> = vector_of_functions
                    .par_iter()
                    .map(|func| {
                        let func = Expr::lambdify(
                            func,
                            variable_str
                                .iter()
                                .map(|s| s.as_str())
                                .collect::<Vec<_>>()
                                .clone(),
                        );
                        let v_vec: Vec<f64> = v.iter().cloned().collect();
                        func( v_vec)
                    })
                    .collect();
                let result = DVector::from_vec(result);
                result
            }) //enf of box
        } // end of function
        let fun = f(
            vector_of_functions.to_owned(),
            arg.to_string(),
            variable_str.clone().iter().map(|s| s.to_string()).collect(),
        );
        let boxed_fun: Box<dyn Fun> = Box::new(FunEnum::Dense(fun));
        self.residiual_function = boxed_fun;
    }

    /*
    pub fn lambdify_residual_VectorType_parallel(&mut self, arg: &str, variable_str: Vec<&str>) {
        let vector_of_functions = &self.vector_of_functions;
        fn f(
            vector_of_functions: Vec<Expr>,
            arg: String,
            variable_str: Vec<String>,
        ) -> Box<dyn Fn(f64, &dyn VectorType) -> Box<dyn VectorType> + Send + Sync> {
            Box::new(move |x: f64, v: &dyn VectorType| -> Box<dyn VectorType> {
                let res = vector_of_functions
                    .par_iter()
                    .enumerate()
                    .map(|(i, func)| {
                        let func = Expr::lambdify_IVP_owned(
                            func.to_owned(),
                            arg.as_str(),
                            variable_str.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                        );
                        (i, func(x, v.iterate().collect()))
                    })
                    .fold(
                        || v.zeros(vector_of_functions.len()),
                        |mut acc, (i, result)| {
                            acc.assign_value(i, result);
                            acc
                        },
                    )
                    .reduce(
                        || v.zeros(vector_of_functions.len()),
                        |mut acc, res| {
                            acc.add(res);
                            acc
                        },
                    );
                res
            })
        }
        let fun = f(
            vector_of_functions.to_owned(),
            arg.to_string(),
            variable_str.clone().iter().map(|s| s.to_string()).collect(),
        );
    }
    */
    //////////////////////////////////////////////////////////////////////////////////////////
    ////                             SPRS CRATE SPARSE FUNCTIONS
    ///////////////////////////////////////////////////////////////////////////////////////////

    pub fn jacobian_generate_CsMat(
        jac: Vec<Vec<Expr>>,
        vector_of_functions_len: usize,
        vector_of_variables_len: usize,
        variable_str: Vec<String>,
       _arg: String,
    ) -> Box<dyn FnMut(f64, &CsVec<f64>) -> CsMat<f64>> {
        //let arg = arg.as_str();
        //let variable_str: Vec<&str> = variable_str.iter().map(|s| s.as_str()).collect();

        Box::new(move |_x: f64, v: &CsVec<f64>| -> CsMat<f64> {
            let mut new_function_jacobian: CsMat<f64> =
                CsMat::zero((vector_of_functions_len, vector_of_variables_len));
            for i in 0..vector_of_functions_len {
                for j in 0..vector_of_variables_len {
                    // println!("i = {}, j = {}", i, j);
                    let partial_func = Expr::lambdify_owned(
                        jac[i][j].clone(),
                        variable_str
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .clone(),
                    );
                    //let v_vec: Vec<f64> = v.iter().cloned().collect();
                    let v_vec: Vec<f64> = v.to_dense().to_vec();
                    new_function_jacobian.insert(j, i, partial_func(v_vec.clone()))
                }
            }
            new_function_jacobian
        })
    } // end of function
    pub fn lambdify_jacobian_CsMat(&mut self, arg: &str, variable_str: Vec<&str>) {
        let symbolic_jacobian = self.symbolic_jacobian.clone();
        let symbolic_jacobian_rc = symbolic_jacobian.clone();

        let vector_of_functions_len = self.vector_of_functions.len();
        let vector_of_variables_len = self.vector_of_variables.len();

        let new_jac = Jacobian::jacobian_generate_CsMat(
            symbolic_jacobian_rc,
            vector_of_functions_len,
            vector_of_variables_len,
            variable_str.iter().map(|s| s.to_string()).collect(),
            arg.to_string(),
        );
        //  let mut boxed_jacobian: Box<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64>> = Box::new(|arg, variable_str_| {
        //    DMatrix::from_rows(&new_jac) }) ;

        let boxed_jac: Box<dyn Jac> = Box::new(JacEnum::Sparse_1(new_jac));
        self.jac_function = Some(boxed_jac);
    }

    pub fn lambdify_residual_CsVec(&mut self, arg: &str, variable_str: Vec<&str>) {
        let vector_of_functions = &self.vector_of_functions;
        fn f(
            vector_of_functions: Vec<Expr>,
            arg: String,
            variable_str: Vec<String>,
        ) -> Box<dyn Fn(f64, &CsVec<f64>) -> CsVec<f64> + 'static> {
            Box::new(move |x: f64, v: &CsVec<f64>| -> CsVec<f64> {
                //  let mut result: CsVec<f64> = CsVec::new(n, indices, data);
                let mut result: CsVec<f64> = CsVec::empty(vector_of_functions.len());
                for (i, func) in vector_of_functions.iter().enumerate() {
                    let func = Expr::lambdify_IVP_owned(
                        func.to_owned(),
                        arg.as_str(),
                        variable_str
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .clone(),
                    );
                    result.append(i, func(x, v.to_dense().to_vec()));
                }
                result
            }) //enf of box
        } // end of function
        let fun = f(
            vector_of_functions.to_owned(),
            arg.to_string(),
            variable_str.clone().iter().map(|s| s.to_string()).collect(),
        );
        let boxed_fun: Box<dyn Fun> = Box::new(FunEnum::Sparse_1(fun));
        self.residiual_function = boxed_fun;
    }

    //////////////////////////////////////////////////////////////////////////////////////////
    //                            NALGEBRA SPARSE CRATE
    ///////////////////////////////////////////////////////////////////////////////////////////
    pub fn jacobian_generate_CsMatrix(
        jac: Vec<Vec<Expr>>,
        vector_of_functions_len: usize,
        vector_of_variables_len: usize,
        variable_str: Vec<String>,
        _arg: String,
    ) -> Box<dyn FnMut(f64, &DVector<f64>) -> CsMatrix<f64>> {
        //let arg = arg.as_str();
        //let variable_str: Vec<&str> = variable_str.iter().map(|s| s.as_str()).collect();

        Box::new(move |_x: f64, v: &DVector<f64>| -> CsMatrix<f64> {
            let _number_of_possible_non_zero: usize = jac.len();
            //let mut new_function_jacobian: CsMatrix<f64> = CsMatrix::new_uninitialized_generic(Dyn(vector_of_functions_len),
            //Dyn(vector_of_variables_len), number_of_possible_non_zero);
            let mut new_function_jacobian: DMatrix<f64> =
                DMatrix::zeros(vector_of_functions_len, vector_of_variables_len);
            for i in 0..vector_of_functions_len {
                for j in 0..vector_of_variables_len {
                    // println!("i = {}, j = {}", i, j);
                    let partial_func = Expr::lambdify_owned(
                        jac[i][j].clone(),
                        variable_str
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .clone(),
                    );
                    //let v_vec: Vec<f64> = v.iter().cloned().collect();
                    let v_vec: Vec<f64> = v.iter().cloned().collect();
                    //new_function_jacobian = CsMatrix::from_triplet(vector_of_functions_len,
                    // vector_of_variables_len, &[i], &[j], &[partial_func(x, v_vec.clone())]   );
                    new_function_jacobian[(i, j)] = partial_func(v_vec.clone());
                }
            }
            new_function_jacobian.into()
        })
    } // end of function
    pub fn lambdify_jacobian_CsMatrix(&mut self, arg: &str, variable_str: Vec<&str>) {
        let symbolic_jacobian = self.symbolic_jacobian.clone();
        let symbolic_jacobian_rc = symbolic_jacobian.clone();

        let vector_of_functions_len = self.vector_of_functions.len();
        let vector_of_variables_len = self.vector_of_variables.len();

        let new_jac = Jacobian::jacobian_generate_CsMatrix(
            symbolic_jacobian_rc,
            vector_of_functions_len,
            vector_of_variables_len,
            variable_str.iter().map(|s| s.to_string()).collect(),
            arg.to_string(),
        );
        //  let mut boxed_jacobian: Box<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64>> = Box::new(|arg, variable_str_| {
        //    DMatrix::from_rows(&new_jac) }) ;

        let boxed_jac: Box<dyn Jac> = Box::new(JacEnum::Sparse_2(new_jac));
        self.jac_function = Some(boxed_jac);
    }
    ////////////////////////////////////////////////////////////////////////////////////////
    ///         FAER SPARCE CRATE
    ////////////////////////////////////////////////////////////////////////////////////////
    pub fn jacobian_generate_SparseColMat(
        jac: Vec<Vec<Expr>>,
        vector_of_functions_len: usize,
        vector_of_variables_len: usize,
        variable_str: Vec<String>,
        _arg: String,
        bandwidth: Option<(usize, usize)>,
    ) -> Box<dyn FnMut(f64, &Col<f64>) -> SparseColMat<usize, f64>> {
        //let arg = arg.as_str();
        //let variable_str: Vec<&str> = variable_str.iter().map(|s| s.as_str()).collect();

        Box::new(move |_x: f64, v: &Col<f64>| -> SparseColMat<usize, f64> {
            let _number_of_possible_non_zero: usize = jac.len();
            //let mut new_function_jacobian: CsMatrix<f64> = CsMatrix::new_uninitialized_generic(Dyn(vector_of_functions_len),
            //Dyn(vector_of_variables_len), number_of_possible_non_zero);
            let mut vector_of_triplets = Vec::new();
            for i in 0..vector_of_functions_len {
                let (right_border, left_border) = if let Some((kl, ku)) = bandwidth {
                    let right_border = std::cmp::min(i + ku + 1, vector_of_variables_len);
                    let left_border = if i as i32 - (kl as i32) - 1 < 0 {
                        0
                    } else {
                        i - kl - 1
                    };
                    (right_border, left_border)
                } else {
                    let right_border = vector_of_variables_len;
                    let left_border = 0;
                    (right_border, left_border)
                };
                for j in left_border..right_border {
                    // if jac[i][j] != Expr::Const(0.0) { println!("i = {}, j = {}, {}", i, j,  &jac[i][j]);}
                    let partial_func = Expr::lambdify_owned(
                        jac[i][j].clone(),
                        variable_str
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .clone(),
                    );
                    //let v_vec: Vec<f64> = v.iter().cloned().collect();
                    let v_vec: Vec<f64> = v.iter().cloned().collect();
                    //new_function_jacobian = CsMatrix::from_triplet(vector_of_functions_len,
                    // vector_of_variables_len, &[i], &[j], &[partial_func(x, v_vec.clone())]   );
                    let P = partial_func(v_vec.clone());
                    if P.abs() > 1e-8 {
                        let triplet = (i, j, P);
                        vector_of_triplets.push(triplet);
                    }
                }
            }
            let new_function_jacobian: SparseColMat<usize, f64> =
                SparseColMat::try_new_from_triplets(
                    vector_of_functions_len,
                    vector_of_variables_len,
                    vector_of_triplets.as_slice(),
                )
                .unwrap();
            //  panic!("stop here");
            new_function_jacobian
        })
    } // end of function

    ///
    ///
    pub fn jacobian_generate_SparseColMat_par(
        jac: Vec<Vec<Expr>>,
        vector_of_functions_len: usize,
        vector_of_variables_len: usize,
        variable_str: Vec<String>,
        _arg: String,
        bandwidth: Option<(usize, usize)>,
    ) -> Box<dyn FnMut(f64, &Col<f64>) -> SparseColMat<usize, f64>> {
        //let arg = arg.as_str();
        //let variable_str: Vec<&str> = variable_str.iter().map(|s| s.as_str()).collect();

        Box::new(move |_x: f64, v: &Col<f64>| -> SparseColMat<usize, f64> {
            let _number_of_possible_non_zero: usize = jac.len();
            //let mut new_function_jacobian: CsMatrix<f64> = CsMatrix::new_uninitialized_generic(Dyn(vector_of_functions_len),
            //Dyn(vector_of_variables_len), number_of_possible_non_zero);
            let mut vector_of_triplets = Vec::new();
            let vector_of_triplets_mutex = std::sync::Mutex::new(&mut vector_of_triplets);
            (0..vector_of_functions_len).into_par_iter().for_each(|i| {
                let (right_border, left_border) = if let Some((kl, ku)) = bandwidth {
                    let right_border = std::cmp::min(i + ku + 1, vector_of_variables_len);
                    let left_border = if i as i32 - (kl as i32) - 1 < 0 {
                        0
                    } else {
                        i - kl - 1
                    };
                    (right_border, left_border)
                } else {
                    let right_border = vector_of_variables_len;
                    let left_border = 0;
                    (right_border, left_border)
                };
                for j in left_border..right_border {
                    // if jac[i][j] != Expr::Const(0.0) { println!("i = {}, j = {}, {}", i, j,  &jac[i][j]);}
                    let symbolic_partoal_derivative = &jac[i][j];
                    if !symbolic_partoal_derivative.is_zero() {
                        let partial_func = Expr::lambdify(
                            symbolic_partoal_derivative,
                            variable_str
                                .iter()
                                .map(|s| s.as_str())
                                .collect::<Vec<_>>()
                                .clone(),
                        );
                        let v_vec: Vec<f64> = v.iter().cloned().collect();
                        let P = partial_func(v_vec.clone());
                        if P.abs() > 1e-8 {
                            let triplet = (i, j, P);
                            vector_of_triplets_mutex.lock().unwrap().push(triplet);
                        }
                    }
                   
                    //let v_vec: Vec<f64> = v.iter().cloned().collect();
                   
                    //new_function_jacobian = CsMatrix::from_triplet(vector_of_functions_len,
                    // vector_of_variables_len, &[i], &[j], &[partial_func(x, v_vec.clone())]   );
  
                }
            });
            let new_function_jacobian: SparseColMat<usize, f64> =
                SparseColMat::try_new_from_triplets(
                    vector_of_functions_len,
                    vector_of_variables_len,
                    vector_of_triplets.as_slice(),
                )
                .unwrap();
            //  panic!("stop here");
            new_function_jacobian
        })
    } // end of function

    pub fn lambdify_jacobian_SparseColMat(&mut self, arg: &str, variable_str: Vec<&str>) {
        let symbolic_jacobian = self.symbolic_jacobian.clone();
        let symbolic_jacobian_rc = symbolic_jacobian.clone();

        let vector_of_functions_len = self.vector_of_functions.len();
        let vector_of_variables_len = self.vector_of_variables.len();

        let new_jac = Jacobian::jacobian_generate_SparseColMat_par(
            symbolic_jacobian_rc,
            vector_of_functions_len,
            vector_of_variables_len,
            variable_str.iter().map(|s| s.to_string()).collect(),
            arg.to_string(),
            self.bandwidth,
        );
        //  let mut boxed_jacobian: Box<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64>> = Box::new(|arg, variable_str_| {
        //    DMatrix::from_rows(&new_jac) }) ;

        let boxed_jac: Box<dyn Jac> = Box::new(JacEnum::Sparse_3(new_jac));
        self.jac_function = Some(boxed_jac);
    }
    pub fn lambdify_residual_Col(&mut self, arg: &str, variable_str: Vec<&str>) {
        let vector_of_functions = &self.vector_of_functions;
        fn f(
            vector_of_functions: Vec<Expr>,
            _arg: String,
            variable_str: Vec<String>,
        ) -> Box<dyn Fn(f64, &Col<f64>) -> Col<f64> + 'static> {
            Box::new(move |_x: f64, v: &Col<f64>| -> Col<f64> {
                //  let mut result: Col<f64> = Col::with_capacity(vector_of_functions.len());
                let result: Vec<_> = vector_of_functions
                    .par_iter()
                    .map(|func| {
                        let func = Expr::lambdify(
                            func,
                       
                            variable_str
                                .iter()
                                .map(|s| s.as_str())
                                .collect::<Vec<_>>()
                                .clone(),
                        );
                        let v_vec: Vec<f64> = v.iter().cloned().collect();
                        func( v_vec)
                    })
                    .collect();
                let res = from_slice(result.as_slice()).to_owned();
                res
            }) //enf of box
        } // end of function
        let fun = f(
            vector_of_functions.to_owned(),
            arg.to_string(),
            variable_str.clone().iter().map(|s| s.to_string()).collect(),
        );
        let boxed_fun: Box<dyn Fun> = Box::new(FunEnum::Sparse_3(fun));
        self.residiual_function = boxed_fun;
    }
    ///////////////////////////////////////////////////////////////////////////
    ///              DISCRETZED FUNCTIONS
    ///////////////////////////////////////////////////////////////////////////
    pub fn remove_numeric_suffix(input: &str) -> String {
        let re = Regex::new(r"_\d+$").unwrap();
        re.replace(input, "").to_string()
    }
    fn eq_step(
        matrix_of_names: &[Vec<String>],
        eq_i: &Expr,
        values: &[String],
        arg: &str,
        j: usize,
        t: f64,
        scheme: &str,
    ) -> Expr {
        let vec_of_names_on_step = &matrix_of_names[j];
        let hashmap_for_rename: HashMap<String, String> = values
            .iter()
            .zip(vec_of_names_on_step.iter())
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
     //set time value of j-th time step
        let eq_step_j = eq_i.rename_variables(&hashmap_for_rename);
        let eq_step_j = eq_step_j.set_variable(arg, t);
    
        match scheme {
            "forward" => eq_step_j,
            "trapezoid" => {
                let vec_of_names_on_step = &matrix_of_names[j + 1];
                let hashmap_for_rename: HashMap<String, String> = values
                    .iter()
                    .zip(vec_of_names_on_step.iter())
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect();
    
                let eq_step_j_plus_1 = eq_i.rename_variables(&hashmap_for_rename).set_variable(arg, t);
                Expr::Const(0.5) * (eq_step_j + eq_step_j_plus_1)
            }
            _ => panic!("Invalid scheme"),
        }
    }
    //
    fn create_mesh(&self, n_steps: Option<usize>, h: Option<f64>, mesh: Option<Vec<f64>>, t0: f64) -> (Vec<Expr>, Vec<f64>, usize) {
        // mesh of t's can be defined directly or by size of step -h, and number of steps
        if let Some(mesh) = mesh {
            info!("mesh is not defined, creating evenly distributed mesh");
            let n_steps = mesh.len();
            let H: Vec<Expr> = mesh.windows(2)
                .map(|window| Expr::Const(window[1] - window[0]))
                .collect();
            (H, mesh, n_steps)
        } else {
            
            let n_steps = n_steps.unwrap_or(100) + 1;
            info!("mesh with n_steps = {} is defined directly", n_steps);
            let h = h.unwrap_or(1.0);
            let H: Vec<Expr> = vec![Expr::Const(h); n_steps - 1];  // number of intervals = n_steps -1
            let T_list: Vec<f64> = (0..n_steps).map(|i| t0 + (i as f64) * h).collect();
            (H, T_list, n_steps)
        }
    }
    
    fn process_bounds_and_tolerances(&mut self, 
    Bounds: Option<HashMap<String, (f64, f64)>>, 
    rel_tolerance: Option<HashMap<String, f64>>, 
    flat_list_of_names:Vec<String>) {
        let mut vec_of_bounds: Vec<(f64, f64)> = Vec::new();
        let mut vec_of_rel_tolerance: Vec<f64> = Vec::new();
        if let Some(ref Bounds_) = Bounds.clone() {
            for Y_i in flat_list_of_names.iter() {
                let name_without_index = Self::remove_numeric_suffix(&Y_i);

                let bound_pair = Bounds_.get(&name_without_index);
                if let Some(bound_pair) = bound_pair {
                    vec_of_bounds.push(*bound_pair)
                }
            }
            self.bounds = Some(vec_of_bounds.clone());
        } else {
            self.bounds = None
        }

        if let Some(ref rel_tolerance_) = rel_tolerance {
            for Y_i in flat_list_of_names.iter() {
                let name_without_index = Self::remove_numeric_suffix(&Y_i);
                let rel_tolerance = rel_tolerance_.get(&name_without_index);
                if let Some(rel_tolerance) = rel_tolerance {
                    vec_of_rel_tolerance.push(*rel_tolerance)
                }
            }
            self.rel_tolerance_vec = Some(vec_of_rel_tolerance.clone());
        } else {
            self.rel_tolerance_vec = None
        }
    }

    fn consistency_test(&mut self, discreditized_system_flat: Vec<Expr>, flat_list_of_names:Vec<String> ) {
        // varaibles from every expression in the system
        let vars_from_flat: Vec<Vec<String>> = discreditized_system_flat
        .clone()
        .iter()
        .map(|exp_i| Expr::all_arguments_are_variables(exp_i))
        .collect(); 
    self.variables_for_all_disrete = vars_from_flat.clone();
    let flat_vec_of_vars_extracted: Vec<String> =
        vars_from_flat.into_iter().flatten().collect();
    let hashset_of_vars_extracted: HashSet<String> =
        flat_vec_of_vars_extracted.into_iter().collect();
    let hashset_of_vars: HashSet<String> = flat_list_of_names.clone().into_iter().collect();
    let found_difference: HashSet<String> = hashset_of_vars_extracted
        .difference(&hashset_of_vars)
        .cloned()
        .collect();
  //  println!("hashset_of_vars_extracted: {:?}", hashset_of_vars_extracted);

    if !found_difference.is_empty() {
        println!("found error: {:?}", found_difference);
        panic!(
            "\n \n \n Some variables are not found in the system {:?} ! \n \n \n",
            found_difference
        );
    }
    }


    pub fn discretization_system_BVP(
        &mut self,
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        t0: f64,
        n_steps: Option<usize>,
        h: Option<f64>,
        mesh: Option<Vec<f64>>,
        BorderConditions: HashMap<String, (usize, f64)>,
        Bounds: Option<HashMap<String, (f64, f64)>>,
        rel_tolerance: Option<HashMap<String, f64>>,
        scheme: String,
    ) {
      //  let total_time = Instant::now();
        let (H, T_list, n_steps) = self.create_mesh(n_steps, h, mesh, t0);
      
     //   info!("creating discretization equations with n_steps = {}, H = {:?} \n ({}), \n T_list = {:?}, \n ({})", n_steps, H, H.len(),T_list, T_list.len());
        // variables on each time slice [[x_0, y_0, z_0], [x_1, y_1, z_1], [x_2, y_2, z_2]]
        let (matrix_of_expr, matrix_of_names) = Expr::IndexedVarsMatrix(n_steps, values.clone());
        /* 
        println!(
            "matrix of names = {:?}, matrix of expr = {:?}",
            &matrix_of_names, &matrix_of_expr
        );
            */
        // DISCRETAZING EQUATIONS
        println!("creating discretization equations");
        /* non parallel version
         let now = Instant::now();
        let mut discreditized_system: Vec<Expr> = Vec::with_capacity(n_steps * eq_system.len());
        // iterate over eq_system and for each eq_i create a vector of discretization equations for all time steps
        for j in 0..n_steps - 1 {
            for (i, eq_i) in eq_system.iter().enumerate() {
                    //  println!("eq_i = {:?}", eq_i);                    
                    //current time step
                    let t = T_list[j];
                    let eq_step_j =Self::eq_step(&matrix_of_names, &eq_i, &values, &arg, j, t, &scheme);
                   
                    // defining residuals for each equation on each time step
                    let Y_j_plus_1 = &matrix_of_expr[j + 1][i];
                    let Y_j = &matrix_of_expr[j][i].clone();
                    let res_ij = Y_j_plus_1.clone() - Y_j.clone() - H[j].clone() * eq_step_j;
                    // println!( "equation {:?} for  {} -th timestep \n", res_ij, j);

                    let res_ij = res_ij.symplify();
                    discreditized_system.push( res_ij);
           
            } // end of for loop eq_i
        } // end of for j in 0..n_steps
        */
      //  let now = Instant::now();
      
        // iterate over eq_system and for each eq_i create a vector of discretization equations for all time steps
        let discreditized_system: Vec<Expr> = (0..n_steps - 1)
        .into_par_iter()
        .flat_map(|j| {
            eq_system.par_iter()
                .enumerate()
                .map(|(i, eq_i)| {
                    let t = T_list[j]; //set time value of j-th time step
                      // defining residuals for each equation on each time step
                    let eq_step_j = Self::eq_step(&matrix_of_names, &eq_i, &values, &arg, j, t, &scheme);
                    let Y_j_plus_1 = &matrix_of_expr[j + 1][i];
                    let Y_j = &matrix_of_expr[j][i].clone();   
                    let res_ij = Y_j_plus_1.clone() - Y_j.clone() - H[j].clone() * eq_step_j;
                    res_ij.symplify()
                })
                .collect::<Vec<_>>()
        })
        .collect();   
     //   println!("discretization system created in {} seconds", now.elapsed().as_secs_f64());
      
    // SETTING UP BOUNDARY CONDITION
    let mut flat_list_of_names = matrix_of_names
    .clone()
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    let mut flat_list_of_expr = matrix_of_expr
        .clone()
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let mut vars_for_boundary_conditions = HashMap::new();
    for j in 0..n_steps - 1 {
        for (i, _) in eq_system.iter().enumerate() {
            let Y_name = &values[i]; //y_i
                //check if there is a border condition. var initial or final ==0 if initial condition, ==1 if final
            if let Some((initial_or_final, condition)) = BorderConditions.get(Y_name) {
                // defining residuals for each equation on each time step
                let Y_j_plus_1 = &matrix_of_expr[j + 1][i];
                let Y_j = &matrix_of_expr[j][i].clone();
                let Y_j_plus_1_str = &matrix_of_names[j + 1][i].clone();
                let Y_j_str = &matrix_of_names[j][i].clone();
                // println!( "equation {:?} for  {} -th timestep \n", res_ij, j);
                if j == 0 && *initial_or_final == 0 {
                    println!("found initial condition");
                    vars_for_boundary_conditions.insert(Y_j_str.clone(), *condition);
                    // delete the variable name from list of variables because we dont want to differentiate on this variable because it is initial condition
                    println!("variable {:?} deleted from list", Y_j_str);
                    flat_list_of_names.retain(|name| *name != *Y_j_str);
                    flat_list_of_expr.retain(|expr| expr != Y_j);
                }

                if j == n_steps - 2 && *initial_or_final == 1 {
                    println!("found final condition");
                    vars_for_boundary_conditions.insert(Y_j_plus_1_str.clone(), *condition);
                    println!("variable {:?} deleted from list", Y_j_plus_1_str);
                    flat_list_of_names.retain(|name| name != Y_j_plus_1_str);
                    flat_list_of_expr.retain(|expr| expr != Y_j_plus_1);
                }
                //  println!("{:?}",vars_for_boundary_conditions);
            }
            // end of if let Some BorderCondition
            else {
                panic!("Border condition for variable {Y_name} not found")
            }
        } // end of for loop eq_i
    } // end of for j in 0..n_steps

    let mut discreditized_system_with_BC = Vec::new();
    for mut eq_i in discreditized_system {
        for (Y, k) in vars_for_boundary_conditions.iter() {
            //  println!("y= {}, eq_i = {:?}", Y,Expr::to_string( &eq_i));
            eq_i = eq_i.set_variable(Y, *k);
            eq_i = eq_i.symplify();
        }
        // println!(" \n eq_i+BC {:?}",Expr::to_string( &eq_i));
        discreditized_system_with_BC.push(eq_i.clone());
    }
    //
    let discreditized_system_flat = discreditized_system_with_BC;

    self.consistency_test(discreditized_system_flat.clone(), flat_list_of_names.clone());
    //
    self.vector_of_functions = discreditized_system_flat;
    self.vector_of_variables = flat_list_of_expr;
    self.variable_string = flat_list_of_names.clone();
        self.process_bounds_and_tolerances(
            Bounds,
            rel_tolerance,
            flat_list_of_names );
        //

   // println!("total time _elapsed_ {}", total_time.elapsed().as_secs());

    } // end of discretization_system



    fn find_bandwidths(&mut self) {
        let A = &self.symbolic_jacobian;
        let n = A.len();
        let mut kl = 0; // Number of subdiagonals
        let mut ku = 0; // Number of superdiagonals
                        /*
                            Matrix Iteration: The function find_bandwidths iterates through each element of the matrix A.
                        Subdiagonal Width (kl): For each non-zero element below the main diagonal (i.e., i > j), it calculates the distance from the diagonal and updates
                        kl if this distance is greater than the current value of kl.
                        Superdiagonal Width (ku): Similarly, for each non-zero element above the main diagonal (i.e., j > i), it calculates the distance from the diagonal
                         and updates ku if this distance is greater than the current value of ku.
                             */
        for i in 0..n {
            for j in 0..n {
                if A[i][j] != Expr::Const(0.0) {
                    if j > i {
                        ku = std::cmp::max(ku, j - i);
                    } else if i > j {
                        kl = std::cmp::max(kl, i - j);
                    }
                }
            }
        }

        self.bandwidth = Some((kl, ku));
    }
    // main function of this module
    // This function essentially sets up all the necessary components for solving a Boundary Value Problem, including discretization,
    // Jacobian calculation, and preparation of numerical evaluation functions. It allows for different methods of handling sparse or dense matrices,
    // making it flexible for various types of problems and computational approaches.
    pub fn generate_BVP(
        &mut self,
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        t0: f64,
        param: Option<String>,
        n_steps: Option<usize>,
        h: Option<f64>,
        mesh: Option<Vec<f64>>,
        BorderConditions: HashMap<String, (usize, f64)>,
        Bounds: Option<HashMap<String, (f64, f64)>>,
        rel_tolerance: Option<HashMap<String, f64>>,
        scheme: String,
        method: String,
        bandwidth: Option<(usize, usize)>,
    ) {
        let total_start = Instant::now();
        let mut timer_hash: HashMap<String, f64> = HashMap::new();
        let param = if let Some(param) = param {
            param
        } else {
            arg.clone()
        };
        self.method = method.clone();
        let begin = Instant::now();
        self.discretization_system_BVP(
            eq_system,
            values.clone(),
            arg.clone(),
            t0,
            n_steps,
            h,
            mesh,
            BorderConditions,
            Bounds,
            rel_tolerance,
            scheme,
        );
        timer_hash.insert("discretization time".to_string(), begin.elapsed().as_secs_f64());
        info!("system discretized");
        let v = self.variable_string.clone();
        /* 
        info!(
            "VECTOR OF FUNCTIONS \n \n  {:?},  length: {} \n \n",
            &self.vector_of_functions,
            &self.vector_of_functions.len()
        );
        */
       // println!("INDEXED VARIABLES {:?}, length:  {} \n \n", &v, &v.len());
        let indexed_values: Vec<&str> = v.iter().map(|x| x.as_str()).collect();
        info!("Calculating jacobian");
        let now = Instant::now();
        self.calc_jacobian_parallel_smart(); //calculate the symbolic Jacobian matrix.
        info!("Jacobian calculation time:");
        let elapsed = now.elapsed();
        info!("{:?}", elapsed_time(elapsed));
        timer_hash.insert("symbolic jacobian time".to_string(), elapsed.as_secs_f64());
        let now = Instant::now();
        if let Some(bandwidth_) = bandwidth {
            self.bandwidth = Some(bandwidth_);
            info!("Bandwidth provided: {:?}", self.bandwidth);
        } else {
            self.find_bandwidths(); //  determine the bandwidth of the Jacobian matrix.
            info!("Bandwidth calculated:");
            info!("(kl, ku) = {:?}", self.bandwidth);
        }
        timer_hash.insert("find bandwidth time".to_string(), now.elapsed().as_secs_f64());
        //  println!("symbolic Jacbian created {:?}", &self.symbolic_jacobian);
        let now = Instant::now();
        match method.as_str() {
            // transform the symbolic Jacobian into a numerical function
            "Dense" => self.lambdify_jacobian_DMatrix(param.as_str(), indexed_values.clone()),
            "Sparse_1" => self.lambdify_jacobian_CsMat(param.as_str(), indexed_values.clone()),
            "Sparse_2" => self.lambdify_jacobian_CsMatrix(param.as_str(), indexed_values.clone()),
            "Sparse" => self.lambdify_jacobian_SparseColMat(param.as_str(), indexed_values.clone()),
            _ => panic!("unknown method: {}", method),
        }
        timer_hash.insert("jacobian lambdify time".to_string(), now.elapsed().as_secs_f64());
        info!("Jacobian lambdified");
        let n = &self.symbolic_jacobian.len();
        for (_i, vec_s) in self.symbolic_jacobian.iter().enumerate() {
            assert_eq!(vec_s.len(), *n, "jacobian not square ");
        }
        // println!("functioon Jacobian created");
        let now = Instant::now();
        match method.as_str() {
            // transform the symbolic residual vector-function into a numerical function
            "Dense" => self.lambdify_residual_DVector(param.as_str(), indexed_values.clone()),
            "Sparse_1" => self.lambdify_residual_CsVec(param.as_str(), indexed_values.clone()),
            "Sparse_2" => self.lambdify_residual_DVector(param.as_str(), indexed_values.clone()),
            "Sparse" => self.lambdify_residual_Col(param.as_str(), indexed_values.clone()),
            _ => panic!("unknown method: {}", method),
        }
        timer_hash.insert("residual functions lambdify time".to_string(), now.elapsed().as_secs_f64());
        info!("Residuals vector lambdified");
        /* 
        if let Some(bounds_) = self.bounds.clone() {
            println!(
                "\n \n bounds vector {:?}, of lengh {}",
                bounds_,
                bounds_.len()
            )
            
        }
        if let Some(rel_tolerance) = self.rel_tolerance_vec.clone() {
            println!(
                "\n \n abs_tolerance vector {:?}, of lengh {}",
                rel_tolerance,
                rel_tolerance.len()
            )
        }
        */
        let total_end = total_start.elapsed().as_secs_f64();
        *timer_hash.get_mut("discretization time").unwrap() /= total_end/100.0;
        *timer_hash.get_mut("symbolic jacobian time").unwrap() /= total_end/100.0;
        *timer_hash.get_mut("jacobian lambdify time").unwrap() /= total_end/100.0;
        *timer_hash.get_mut("residual functions lambdify time").unwrap() /= total_end/100.0;
        *timer_hash.get_mut("find bandwidth time").unwrap() /= total_end/100.0;
        timer_hash.insert("total time, sec".to_string(), total_end);

        let mut table = Builder::from(timer_hash.clone()).build();
        table.with(Style::modern_rounded());
        println!("{}", table.to_string());
     //   panic!("END OF GENERATE BVP");
        info!("\n \n ____________END OF GENERATE BVP ________________________________________________________________");
    }
} // end of impl
  ////////////////////////////////////////////////////////////////
  // TESTS
  ////////////////////////////////////////////////////////////////

#[cfg(test)]

mod tests {
    use super::*;
    use crate::numerical::BVP_Damp::BVP_traits::
        Vectors_type_casting;
    /*
    #[test]
    fn generate_IVP_test() {
        let mut Jacobian_instance = Jacobian::new();
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let z: Expr = Expr::Var("z".to_string());
        //eq = z*x +exp(y);
        let eq1: Expr = z.clone() * x.clone() + Expr::exp(y.clone());
        // eq2 = x + ln(z) + y
        let eq2: Expr = x + Expr::ln(z) + y;
        let eq_system = vec![eq1, eq2];
        let values = vec!["y".to_string(), "z".to_string()];
        let arg = "x".to_string();
        // expecting Jac:
        Jacobian_instance.generate_IVP(eq_system, values, arg);
        // J= {{e^y, 10}, {1, 1/z}}
        //
        Jacobian_instance.evaluate_func_jacobian_DMatrix_IVP(10.0, vec![0.0, 1.0]);
        println!(
            "Jacobian_instance.evaluated_jacobian_DMatrix = {:?}",
            Jacobian_instance.evaluated_jacobian_DMatrix
        );
        assert_eq!(
            Jacobian_instance.evaluated_jacobian_DMatrix,
            DMatrix::from_row_slice(2, 2, &[1.0, 10.0, 1.0, 1.0])
        );
        Jacobian_instance.evaluate_funvector_lambdified_DVector_IVP(10.0, vec![0.0, 1.0]);
        println!(
            "Jacobian_instance.evaluated_functions_DVector = {:?}",
            Jacobian_instance.evaluated_functions_DVector
        );
        assert_eq!(
            Jacobian_instance.evaluated_functions_DVector,
            DVector::from_vec(vec![11.0, 10.0])
        );
    }
     */
    #[test]
    fn generate_Jac_test() {
        let Jacobian_instance = &mut Jacobian::new();

        let y = Expr::Var("y".to_string());
        let z: Expr = Expr::Var("z".to_string());
        //eq = z*10.0 +exp(y);
        let eq1: Expr = z.clone() * Expr::Const(10.0) + Expr::exp(y.clone());
        // eq2 = x + ln(z) + y
        let eq2: Expr = Expr::ln(z.clone()) + y.clone();
        let eq_system = vec![eq1, eq2];
        let values = vec!["y".to_string(), "z".to_string()];
        //  let arg = "x".to_string();
        Jacobian_instance.from_vectors(eq_system, values);
        Jacobian_instance.calc_jacobian();

        // expecting Jac: | exp(y) ; 10.0  |
        //                 | 1     ;  1/z|
        let vect = &DVector::from_vec(vec![0.0, 1.0]);
        let expexted_jac = DMatrix::from_row_slice(2, 2, &[1.0, 10.0, 1.0, 1.0]);
        // cast type to deisred crate type
        // FAER CRATE
        Jacobian_instance.lambdify_jacobian_SparseColMat("x", vec!["y", "z"]);

        let variables = &*Vectors_type_casting(vect, "Sparse".to_string());
        let jac = Jacobian_instance.jac_function.as_mut().unwrap();
        let result_SparseColMat = jac.call(1.0, variables);
        assert_eq!(result_SparseColMat.to_DMatrixType(), expexted_jac);
        // NALGEBRA CRATE
        Jacobian_instance.lambdify_jacobian_DMatrix("x", vec!["y", "z"]);
        let variables = &*Vectors_type_casting(vect, "Dense".to_string());
        let jac = Jacobian_instance.jac_function.as_mut().unwrap();
        let result_DMatrix = jac.call(1.0, variables);
        assert_eq!(result_DMatrix.to_DMatrixType(), expexted_jac);

        // NALGEBRA SPARSE CRATE
        Jacobian_instance.lambdify_jacobian_CsMatrix("x", vec!["y", "z"]);
        let variables = &*Vectors_type_casting(vect, "Sparse_2".to_string());
        let jac = Jacobian_instance.jac_function.as_mut().unwrap();
        let result_CsMatrix = jac.call(1.0, variables);
        assert_eq!(result_CsMatrix.to_DMatrixType(), expexted_jac);
        //SPRS CRATE
        /*
               Jacobian_instance.lambdify_jacobian_CsMat(  "x", vec!("y", "z"));
               let variables = &*Vectors_type_casting(vect, "Sparse_1".to_string());
               let jac =   Jacobian_instance.jac_function.as_mut().unwrap();
               let result_CsMat = jac.call(1.0,  variables.clone());
               println!("RES {:?}, EXPECTED {}", result_CsMat, expexted_jac);
               assert_eq!(result_CsMat.to_DMatrixType(),   expexted_jac);

        */
    }
    #[test]
    fn CsMat_test() {}
}

/*
#[test]
fn test_jacobian_new() {
    let jac = Jacobian::new();
    assert_eq!(jac.vector_of_functions.len(), 0);
    assert_eq!(jac.vector_of_variables.len(), 0);
    assert_eq!(jac.variable_string.len(), 0);
    assert_eq!(jac.symbolic_jacobian.len(), 0);
    assert_eq!(jac.readable_jacobian.len(), 0);
    assert_eq!(jac.function_jacobian.len(), 0);
}

#[test]
fn test_jacobian_from_vectors() {
    let funcs = vec![Expr::parse_expression("x^2 + y^2"), Expr::parse_expression("log(x) + exp(y)")];
    let vars = vec![Expr::parse_expression("x"), Expr::parse_expression("y")];
    let jac = Jacobian::from_vectors(funcs, vars);
    assert_eq!(jac.vector_of_functions.len(), 2);
    assert_eq!(jac.vector_of_variables.len(), 2);
    assert_eq!(jac.variable_string.len(), 0);
    assert_eq!(jac.symbolic_jacobian.len(), 2);
    assert_eq!(jac.readable_jacobian.len(), 2);
    assert_eq!(jac.function_jacobian.len(), 2);
}

#[test]
fn test_jacobian_calc_jacobian() {
    let mut jac = Jacobian::new();
    jac.set_vector_of_functions(vec![Expr::parse_expression("x^2 + y^2"), Expr::parse_expression("sin(x) + cos(y)")]);
    jac.set_vector_of_variables(vec![Expr::parse_expression("x"), Expr::parse_expression("y")]);
    jac.set_varvecor_from_str("x, y");
    jac.calc_jacobian();
    let expected_jacobian = vec![
        vec![Expr::parse_expression("2*x"), Expr::parse_expression("2*y")],
        vec![Expr::parse_expression("cos(x)"), Expr::parse_expression("-sin(y)")],
    ];
    assert_eq!(jac.symbolic_jacobian, expected_jacobian);
}

#[test]
fn test_jacobian_calc_jacobian_fun() {
    let mut jac = Jacobian::new();
    jac.set_vector_of_functions(vec![Expr::parse_expression("x^2 + y^2"), Expr::parse_expression("sin(x) + cos(y)")]);
    jac.set_vector_of_variables(vec![Expr::parse_expression("x"), Expr::parse_expression("y")]);
    jac.set_varvecor_from_str("x, y");
    jac.calc_jacobian();
    jac.calc_jacobian_fun();
    let x = vec![1.0, 2.0];
    let expected_result = vec![
        vec![2.0, 4.0],
        vec![0.5403023058681398, -0.4161468365471424],
    ];
  //  assert_eq!(jac.evaluate_func_jacobian(&x), expected_result);
}
*/
