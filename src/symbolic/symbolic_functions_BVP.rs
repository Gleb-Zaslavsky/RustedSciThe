#![allow(non_camel_case_types)]

use crate::symbolic::symbolic_engine::Expr;
use crate::numerical::BVP_Damp::BVP_utils::elapsed_time;
use std::time::{Instant, Duration};
use faer::col::{from_slice, Col};
use faer::sparse::SparseColMat;
use nalgebra::sparse::CsMatrix;
use nalgebra::{DMatrix, DVector};
use log::info;
use regex::Regex;
use sprs::{CsMat, CsVec};
use std::collections::{HashMap, HashSet};
use crate::numerical::BVP_Damp::BVP_traits::{ Fun, FunEnum, Jac, JacEnum,  VectorType, Vectors_type_casting};
use rayon::prelude::*;
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
    pub vector_of_variables: Vec<Expr>,            // vector of symbolic variables
    pub variable_string: Vec<String>,              // vector of string representation of variables
    pub symbolic_jacobian: Vec<Vec<Expr>>,         // vector of symbolic jacobian
    pub lambdified_jac_element:Option<Box<dyn Fn(f64, usize, usize) -> f64>  >,


    pub jac_function:  Option<Box<dyn Jac>>,

    pub residiual_function: Box<dyn Fun>,

    pub bounds: Option<Vec<(f64, f64)>>,
    pub rel_tolerance_vec: Option<Vec<f64>>,
    pub bandwidth: (usize, usize)
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
            bandwidth: (0, 0),
        }
    }

    pub fn from_vectors(&mut self,
        vector_of_functions: Vec<Expr>,
        variable_string: Vec<String>,
    )  {
       self.vector_of_functions = vector_of_functions;
       self.variable_string = variable_string.clone();
       self.vector_of_variables = Expr::parse_vector_expression(variable_string.iter().map(|s| s.as_str()).collect());
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
        let new_jac: Vec<Vec<Expr>> = self.vector_of_functions.par_iter().map(|function| {
            let mut vector_of_partial_derivatives = Vec::new();
            let function = function.clone();
            for j in 0..self.vector_of_variables.len() {
                let mut partial = Expr::diff(
                    &function,
                    &variable_string_vec[j],
                );
                partial = partial.symplify();
                vector_of_partial_derivatives.push(partial);
            }
            vector_of_partial_derivatives
        }).collect();

        self.symbolic_jacobian = new_jac;
    }
    
////////////////////////////////////////////////////////////////////////////////////
//  GENERIC FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////
pub fn lambdify_residual_VectorType(&mut self, arg: &str, variable_str: Vec<&str>) {
    let vector_of_functions = &self.vector_of_functions;
    fn f(
        vector_of_functions: Vec<Expr>,
        arg: String,
        variable_str: Vec<String>,
    ) -> Box<dyn Fn(f64, &dyn VectorType) ->Box<dyn VectorType>> {
        Box::new(move |x: f64, v: &dyn VectorType| -> Box<dyn VectorType>  {
            let mut  res =  v.zeros(vector_of_functions.len());
            let result =  v.zeros(vector_of_functions.len());
            for (i, func) in vector_of_functions.iter().enumerate() {
                let func = Expr::lambdify_owned(
                    func.to_owned(),
             
                    variable_str
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .clone(),
                );
                res = result.assign_value(i,func( v.iterate().collect())   );
            }
         
          res
        }) //enf of box
    } // end of function
    let fun:Box<dyn Fn(f64, &dyn VectorType) -> Box<dyn VectorType>>= f(
        vector_of_functions.to_owned(),
        arg.to_string(),
        variable_str.clone().iter().map(|s| s.to_string()).collect(),
    );
   // self.residiual_function = fun;
   

} 


////////////////////////////////////////////////////////////////////////////////////
///                             NALGEBRA DENSE
////////////////////////////////////////////////////////////////////////////////////
/*
pub fn jacobian_generate_DMatrix_(
    jac: Vec<Vec<Expr>>,
    vector_of_functions_len: usize,
    vector_of_variables_len: usize,
    variable_str: Vec<String>,
    arg: String,
    bandwidth: (usize, usize),
) -> Box<dyn FnMut(f64, &DVector<f64>) -> DMatrix<f64>> {
    //let arg = arg.as_str();
    //let variable_str: Vec<&str> = variable_str.iter().map(|s| s.as_str()).collect();
    let (kl, ku) = bandwidth;
    Box::new(move |x: f64, v: &DVector<f64>| -> DMatrix<f64> {
   
    let new_function_jacobian:Vec<DVector<f64>> =
    jac.par_iter().enumerate().map(|(i, vec_of_derivs)| {
            let mut vec_od_jac = DVector::zeros(vec_of_derivs.len());
            let (right_border, left_border) =
            if kl==0&&ku==0 {
                let right_border = vector_of_variables_len;
                let left_border = 0;
                (right_border, left_border)
            } else {
                let right_border = std::cmp::min(i+ku+1, vector_of_variables_len );
                let left_border =  if  i as i32 - (kl as i32) - 1  <0 {0} else { i - kl- 1} ;
                (right_border, left_border)
            } ;

            for j in left_border.. right_border {
                // println!("i = {}, j = {}", i, j);
                //  let now = std::time::Instant::now();
                let partial_func = Expr::lambdify_owned(
                    vec_of_derivs[j].clone(),
            
                    variable_str
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .clone(),
                );
                let v_vec: Vec<f64> = v.iter().cloned().collect();
                vec_od_jac[ j] = if vec_of_derivs[j].clone() == Expr::Const(0.0) {
                    0.0
                } else {
                    partial_func( v_vec.clone())
                };

                //    let time_test = now.elapsed().as_micros();
                //   if time_test > 100 {println!("Elapsed time: {:?} micrs, {:?} ", now.elapsed().as_micros(), jac[i][j].clone() );}
            }// for j im left...
            vec_od_jac
        }).collect_into_vec();
        new_function_jacobian
    })
} // end of function


pub fn jacobian_generate_DMatrix2(
    jac: Vec<Vec<Expr>>,
    vector_of_functions_len: usize,
    vector_of_variables_len: usize,
    variable_str: Vec<String>,
    arg: String,
    bandwidth: (usize, usize),
) -> Box<dyn FnMut(f64, &DVector<f64>) -> DMatrix<f64>> {
    let (kl, ku) = bandwidth;
    Box::new(move |x: f64, v: &DVector<f64>| -> DMatrix<f64> {
        let mut new_function_jacobian: DMatrix<f64> =
            DMatrix::zeros(vector_of_functions_len, vector_of_variables_len);

        new_function_jacobian
            .into_iter()
            .enumerate()
            .par_bridge()
            .for_each(|(i, mut row)| {
                let (right_border, left_border) = if kl == 0 && ku == 0 {
                    let right_border = vector_of_variables_len;
                    let left_border = 0;
                    (right_border, left_border)
                } else {
                    let right_border = std::cmp::min(i + ku + 1, vector_of_variables_len);
                    let left_border = if i as i32 - (kl as i32) - 1 < 0 {
                        0
                    } else {
                        i - kl - 1
                    };
                    (right_border, left_border)
                };

                for j in left_border..right_border {
                    let partial_func = Expr::lambdify_owned(
                        jac[i][j].clone(),
                        variable_str
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .clone(),
                    );
                    let v_vec: Vec<f64> = v.iter().cloned().collect();
                    row[j] = if jac[i][j].clone() == Expr::Const(0.0) {
                        0.0
                    } else {
                        partial_func(v_vec.clone())
                    };
                }
            });

        new_function_jacobian
    })
}
*/
    pub fn jacobian_generate_DMatrix(
        jac: Vec<Vec<Expr>>,
        vector_of_functions_len: usize,
        vector_of_variables_len: usize,
        variable_str: Vec<String>,
        arg: String,
        bandwidth: (usize, usize),
    ) -> Box<dyn FnMut(f64, &DVector<f64>) -> DMatrix<f64>> {
        //let arg = arg.as_str();
        //let variable_str: Vec<&str> = variable_str.iter().map(|s| s.as_str()).collect();
        let (kl, ku) = bandwidth;
        Box::new(move |x: f64, v: &DVector<f64>| -> DMatrix<f64> {
            let mut new_function_jacobian: DMatrix<f64> =
                DMatrix::zeros(vector_of_functions_len, vector_of_variables_len);
            for i in 0..vector_of_functions_len {
                let (right_border, left_border) =
                if kl==0&&ku==0 {
                    let right_border = vector_of_variables_len;
                    let left_border = 0;
                    (right_border, left_border)
                } else {
                    let right_border = std::cmp::min(i+ku+1, vector_of_variables_len );
                    let left_border =  if  i as i32 - (kl as i32) - 1  <0 {0} else { i - kl- 1} ;
                    (right_border, left_border)
                } ;
                for j in left_border.. right_border {
                    // println!("i = {}, j = {}", i, j);
                    //  let now = std::time::Instant::now();
                    let partial_func = Expr::lambdify_owned(
                        jac[i][j].clone(),
                
                        variable_str
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .clone(),
                    );
                    let v_vec: Vec<f64> = v.iter().cloned().collect();
                    new_function_jacobian[(i, j)] = if jac[i][j].clone() == Expr::Const(0.0) {
                        0.0
                    } else {
                        partial_func( v_vec.clone())
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
        let new_jac = Jacobian::jacobian_generate_DMatrix(
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
        self.jac_function = Some( boxed_jac);
    }


    pub fn lambdify_residual_DVector(&mut self, arg: &str, variable_str: Vec<&str>) {
        let vector_of_functions = &self.vector_of_functions;
        fn f(
            vector_of_functions: Vec<Expr>,
            arg: String,
            variable_str: Vec<String>,
        ) -> Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64> + 'static> {
            Box::new(move |x: f64, v: &DVector<f64>| -> DVector<f64> {
                let mut result: DVector<f64> = DVector::zeros(vector_of_functions.len());
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
                    result[i] = func(x, v.iter().cloned().collect());
                }
                result
            }) //enf of box
        } // end of function
        let fun= f(
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
        arg: String,
    ) -> Box<dyn FnMut(f64, &CsVec<f64>) -> CsMat<f64>> {
        //let arg = arg.as_str();
        //let variable_str: Vec<&str> = variable_str.iter().map(|s| s.as_str()).collect();

        Box::new(move |x: f64, v: &CsVec<f64>| -> CsMat<f64> {
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
                    new_function_jacobian.insert(j, i, partial_func( v_vec.clone()))
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
    arg: String,
) -> Box<dyn FnMut(f64, &DVector<f64>) -> CsMatrix<f64>> {
    //let arg = arg.as_str();
    //let variable_str: Vec<&str> = variable_str.iter().map(|s| s.as_str()).collect();

    Box::new(move |x: f64, v: &DVector<f64>| -> CsMatrix<f64> {
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
                new_function_jacobian[(i, j)] = partial_func( v_vec.clone());
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
        arg: String,
        bandwidth: (usize, usize),
    ) -> Box<dyn FnMut(f64, &Col<f64>) -> SparseColMat<usize, f64>> {
        //let arg = arg.as_str();
        //let variable_str: Vec<&str> = variable_str.iter().map(|s| s.as_str()).collect();

        Box::new(move |x: f64, v: &Col<f64>| -> SparseColMat<usize, f64> {
            let (kl, ku) = bandwidth;
            let _number_of_possible_non_zero: usize = jac.len();
            //let mut new_function_jacobian: CsMatrix<f64> = CsMatrix::new_uninitialized_generic(Dyn(vector_of_functions_len),
            //Dyn(vector_of_variables_len), number_of_possible_non_zero);
            let mut vector_of_triplets = Vec::new();
            for i in 0..vector_of_functions_len {
                let (right_border, left_border) =
                if kl==0&&ku==0 {
                    let right_border = vector_of_variables_len;
                    let left_border = 0;
                    (right_border, left_border)
                } else {
                    let right_border = std::cmp::min(i+ku+1, vector_of_variables_len );
                    let left_border =  if  i as i32 - (kl as i32) - 1  <0 {0} else { i - kl- 1} ;
                    (right_border, left_border)
                } ;
                for j in left_border.. right_border {
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
                    let P = partial_func( v_vec.clone());
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

    pub fn lambdify_jacobian_SparseColMat(&mut self, arg: &str, variable_str: Vec<&str>) {
        let symbolic_jacobian = self.symbolic_jacobian.clone();
        let symbolic_jacobian_rc = symbolic_jacobian.clone();

        let vector_of_functions_len = self.vector_of_functions.len();
        let vector_of_variables_len = self.vector_of_variables.len();

        let new_jac = Jacobian::jacobian_generate_SparseColMat(
            symbolic_jacobian_rc,
            vector_of_functions_len,
            vector_of_variables_len,
            variable_str.iter().map(|s| s.to_string()).collect(),
            arg.to_string(),
            self.bandwidth
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
            arg: String,
            variable_str: Vec<String>,
        ) -> Box<dyn Fn(f64, &Col<f64>) -> Col<f64> + 'static> {
            Box::new(move |x: f64, v: &Col<f64>| -> Col<f64> {
                //  let mut result: Col<f64> = Col::with_capacity(vector_of_functions.len());
                let mut result = Vec::new();
                for (_i, func) in vector_of_functions.iter().enumerate() {
                    let func = Expr::lambdify_IVP_owned(
                        func.to_owned(),
                        arg.as_str(),
                        variable_str
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .clone(),
                    );
                    let v_vec: Vec<f64> = v.iter().cloned().collect();
                    result.push(func(x, v_vec));
                }
                let res = from_slice(result.as_slice()).to_owned();
                res
            }) //enf of box
        } // end of function
        let fun= f(
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
        matrix_of_names: &Vec<Vec<String>>,
        eq_i: &Expr,
        values: &Vec<String>,
        arg: &String,
        j: usize,
        t: f64,
        scheme: &String,
    ) -> Expr {
        // vector of variables of j-th time step
        let vec_of_names_on_step = matrix_of_names[j].clone();
        // for each time step rename the default variable name to a name of variable on j-th time step
        // like x,y,z->  x_10, y_10, z_10, ...
        let hashmap_for_rename: HashMap<String, String> = values
            .iter()
            .zip(vec_of_names_on_step.iter())
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let eq_step_j = eq_i.rename_variables(&hashmap_for_rename);
        //set time value of j-th time step
        let eq_step_j_ = eq_step_j.set_variable(arg.as_str(), t);
        let eq_step_j = match scheme.as_str() {
            "forward" => eq_step_j_,
            "trapezoid" => {
                let vec_of_names_on_step = matrix_of_names[j + 1].clone();
                // for each time step rename the default variable name to a name of variable on j-th time step
                // like x,y,z->  x_10, y_10, z_10, ...
                let hashmap_for_rename: HashMap<String, String> = values
                    .iter()
                    .zip(vec_of_names_on_step.iter())
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect();
                let eq_step_j = eq_i.rename_variables(&hashmap_for_rename);
                //set time value of j-th time step
                let eq_step_j_plus_1 = eq_step_j.set_variable(arg.as_str(), t);
                Expr::Const(0.5) * (eq_step_j_ + eq_step_j_plus_1)
            }
            _ => panic!("Invalid scheme"),
        };
        eq_step_j
    }
    //
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
        let mut vec_of_bounds: Vec<(f64, f64)> = Vec::new();
        let _map_of_bounds: HashMap<String, (f64, f64)> = HashMap::new();
        let mut vec_of_rel_tolerance: Vec<f64> = Vec::new();
        let _map_of_rel_tolerance: HashMap<String, f64> = HashMap::new();
        // mesh of t's can be defined directly or by size of step -h, and number of steps

        let n_steps = if let Some(n_steps) = n_steps {
            n_steps
        } else {
            mesh.clone().unwrap().len()
        };
        // say mesh is defined  by size of step -h, and number of steps
        let h = if let Some(h) = h {
            h
        } else {
            1.0 // its nothing
        };

        let n_steps = n_steps + 1;
        // if there are no mesh in the explicit form we construct evenly distributed H and list of times
        let (H, T_list) = if mesh.clone().is_none() {
            let H_: Expr = Expr::Const(h);
            let H: Vec<Expr> = vec![H_; n_steps]; // H_0, H_1, H_2>
            let T_list: Vec<f64> = (0..n_steps)
                .map(|i| t0 + (i as f64) * h)
                .collect::<Vec<_>>();
            (H, T_list)
        } else {
            (
                mesh.clone()
                    .unwrap()
                    .iter()
                    .map(|x| Expr::Const(*x))
                    .collect(),
                mesh.unwrap(),
            )
        };
        info!("creating discretization equations with n_steps = {}, H = {:?} ({}), T_list = {:?} ({})", n_steps, H, H.len(),T_list, T_list.len());

        let mut discreditized_system: Vec<Vec<Expr>> = Vec::new();
        // variables on each time slice [[x_0, y_0, z_0], [x_1, y_1, z_1], [x_2, y_2, z_2]]
        let (matrix_of_expr, matrix_of_names) = Expr::IndexedVarsMatrix(n_steps, values.clone());
        println!(
            "matrix of names = {:?}, matrix of expr = {:?}",
            &matrix_of_names, &matrix_of_expr
        );

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
        println!("creating discretization equations");
        let mut vars_for_boundary_conditions = HashMap::new();

        // iterate over eq_system and for each eq_i create a vector of discretization equations for all time steps
         for j in 0..n_steps - 1 {
                for (i, eq_i) in eq_system.clone().into_iter().enumerate() {
                    let Y_name = values[i].clone(); //y_i
                                                    //  println!("eq_i = {:?}", eq_i);
                                                    //check if there is a border condition. var initial or final ==0 if initial condition, ==1 if final
                    if let Some((initial_or_final, condition)) = BorderConditions.get(&Y_name) {
                        let mut vec_of_res_for_each_eq = Vec::new();

                    
                            //currwbt time step
                            let t = T_list[j];

                            let mut eq_step_j =
                                Self::eq_step(&matrix_of_names, &eq_i, &values, &arg, j, t, &scheme);
                            //set time value of j-th time step
                            eq_step_j = eq_step_j.set_variable(arg.as_str(), t);

                            // defining residuals for each equation on each time step

                            let Y_j_plus_1 = matrix_of_expr[j + 1][i].clone();
                            let Y_j = matrix_of_expr[j][i].clone();
                            let Y_j_plus_1_str = matrix_of_names[j + 1][i].clone();
                            let Y_j_str = matrix_of_names[j][i].clone();
                            let mut res_ij = Y_j_plus_1.clone() - Y_j.clone() - H[j].clone() * eq_step_j;
                            // println!( "equation {:?} for  {} -th timestep \n", res_ij, j);

                            if j == 0 && initial_or_final.to_owned() == 0 {
                                println!("found initial condition");
                                res_ij = res_ij
                                    .set_variable(Expr::to_string(&Y_j).as_str(), condition.to_owned());
                                vars_for_boundary_conditions.insert(Y_j_str.clone(), condition);
                                // delete the variable name from list of variables because we dont want to differentiate on this variable because it is initial condition
                                println!("variable {:?} deleted from list", Y_j_str);
                                flat_list_of_names.retain(|name| *name != *Y_j_str);
                                flat_list_of_expr.retain(|expr| *expr != Y_j);
                            }

                            if j == n_steps - 2 && initial_or_final.to_owned() == 1 {
                                println!("found final condition");
                                res_ij = res_ij.set_variable(
                                    Expr::to_string(&Y_j_plus_1).as_str(),
                                    condition.to_owned(),
                                );
                                vars_for_boundary_conditions.insert(Y_j_plus_1_str.clone(), condition);
                                println!("variable {:?} deleted from list", Y_j_plus_1_str);
                                flat_list_of_names.retain(|name| *name != *Y_j_plus_1_str);
                                flat_list_of_expr.retain(|expr| *expr != Y_j_plus_1);
                            }
                            //  println!("{:?}",vars_for_boundary_conditions);
                            for (Y, k) in vars_for_boundary_conditions.iter() {
                                res_ij = res_ij.set_variable(Y, **k);
                                // println!(" boundary conditions {:?}",Expr::to_string( &res_ij));
                            }
                            //    vars_for_boundary_conditions.iter().map(|  (Y, condition)|res_ij.set_variable(Y, **condition)  );
                            res_ij = res_ij.symplify();
                            vec_of_res_for_each_eq.push(res_ij);
                    

                        discreditized_system.push(vec_of_res_for_each_eq);
                    }
                    // end of if let Some BorderCondition
                    else {
                        panic!("Border condition for variable {Y_name} not found")
                    }
                } // end of for loop eq_i
         } // end of for j in 0..n_steps
        let discreditized_system_flat = discreditized_system
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        let mut discreditized_system_flat_filtered = Vec::new();
        for mut eq_i in discreditized_system_flat {
            for (Y, k) in vars_for_boundary_conditions.iter() {
                //  println!("y= {}, eq_i = {:?}", Y,Expr::to_string( &eq_i));
                eq_i = eq_i.set_variable(Y, **k);
                eq_i = eq_i.symplify();
            }
            // println!(" \n eq_i+BC {:?}",Expr::to_string( &eq_i));
            discreditized_system_flat_filtered.push(eq_i.clone());
        }
        //
        let discreditized_system_flat = discreditized_system_flat_filtered;

        let vars_from_flat: Vec<Vec<String>> = discreditized_system_flat
            .clone()
            .iter()
            .map(|exp_i| Expr::all_arguments_are_variables(exp_i))
            .collect();
        let flat_vec_of_vars_extracted: Vec<String> =
            vars_from_flat.into_iter().flatten().collect();
        let hashset_of_vars_extracted: HashSet<String> =
            flat_vec_of_vars_extracted.into_iter().collect();
        let hashset_of_vars: HashSet<String> = flat_list_of_names.clone().into_iter().collect();
        let found_difference: HashSet<String> = hashset_of_vars_extracted
            .difference(&hashset_of_vars)
            .cloned()
            .collect();
        println!("hashset_of_vars_extracted: {:?}", hashset_of_vars_extracted);

        if !found_difference.is_empty() {
            println!("found error: {:?}", found_difference);
            panic!(
                "\n \n \n Some variables are not found in the system {:?} ! \n \n \n",
                found_difference
            );
        }
        //
        self.vector_of_functions = discreditized_system_flat;
        self.vector_of_variables = flat_list_of_expr;
        self.variable_string = flat_list_of_names.clone();
        //
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
    } // end of discretization_system

    fn find_bandwidths(&mut self)  {
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
    
        self.bandwidth = (kl, ku);
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
        method: String
    ) {
        let param = if let Some(param) = param {
            param
        } else {
            arg.clone()
        };
        self.method = method.clone();
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
        info!("system discretized");
        let v = self.variable_string.clone();
        println!(
            "VECTOR OF FUNCTIONS \n \n  {:?},  length: {} \n \n",
            &self.vector_of_functions,
            &self.vector_of_functions.len()
        );
        println!("INDEXED VARIABLES {:?}, length:  {} \n \n", &v, &v.len());
        let indexed_values: Vec<&str> = v.iter().map(|x| x.as_str()).collect();
        info!("Calculating jacobian");
        let now = Instant::now();
        self.calc_jacobian_parallel(); //calculate the symbolic Jacobian matrix.
        info!("Jacobian calculation time:");
        let elapsed = now.elapsed();
        info!("{:?}",  elapsed_time(elapsed) );
        self.find_bandwidths();//  determine the bandwidth of the Jacobian matrix.
        info!("Bandwidth calculated:");
        info!("(kl, ku) = {:?}", self.bandwidth);
 
        //  println!("symbolic Jacbian created {:?}", &self.symbolic_jacobian);
        match method.as_str() { // transform the symbolic Jacobian into a numerical function
            "Dense" => self.lambdify_jacobian_DMatrix(param.as_str(), indexed_values.clone()),
            "Sparse_1" => self.lambdify_jacobian_CsMat(param.as_str(), indexed_values.clone()),
            "Sparse_2" =>self.lambdify_jacobian_CsMatrix(param.as_str(), indexed_values.clone()),
            "Sparse" => self.lambdify_jacobian_SparseColMat(param.as_str(), indexed_values.clone()),
             _ => panic!("unknown method: {}", method),
         }
        info!("Jacobian lambdified");
        let n = &self.symbolic_jacobian.len();
        for (_i, vec_s) in self.symbolic_jacobian.iter().enumerate() {
            assert_eq!(vec_s.len(), *n, "jacobian not square ");
        }
        // println!("functioon Jacobian created");
        match method.as_str() { // transform the symbolic residual vector-function into a numerical function
            "Dense" => self.lambdify_residual_DVector(param.as_str(), indexed_values.clone()),
            "Sparse_1" => self.lambdify_residual_CsVec(param.as_str(), indexed_values.clone()),
            "Sparse_2" =>self.lambdify_residual_DVector(param.as_str(), indexed_values.clone()),
            "Sparse" => self.lambdify_residual_Col(param.as_str(), indexed_values.clone()),
            _ => panic!("unknown method: {}", method),
         }
        info!("Residuals vector lambdified");
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
        info!("\n \n ____________END OF GENERATE BVP ________________________________________________________________");

    }

} // end of impl
////////////////////////////////////////////////////////////////
// TESTS
//////////////////////////////////////////////////////////////// 

#[cfg(test)]

mod tests {
    use super::*;
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
        let eq2: Expr =  Expr::ln(z.clone()) + y.clone();
        let eq_system = vec![eq1, eq2];
        let values = vec!["y".to_string(), "z".to_string()];
      //  let arg = "x".to_string();
        Jacobian_instance.from_vectors(eq_system, values);
        Jacobian_instance.calc_jacobian();
       // expecting Jac: | exp(y) ; 10.0  |
       //                 | 1     ;  1/z|
       let vect =&DVector::from_vec(vec![0.0, 1.0]);
       let expexted_jac = DMatrix::from_row_slice(2, 2, &[1.0, 10.0, 1.0, 1.0]);
       // cast type to deisred crate type
       // FAER CRATE
        Jacobian_instance.lambdify_jacobian_SparseColMat(  "x", vec!("y", "z"));

        let variables = &*Vectors_type_casting(vect, "Sparse".to_string());
        let jac = Jacobian_instance.jac_function.as_mut().unwrap();
        let result_SparseColMat = jac.call(1.0,  variables.clone());
        assert_eq!(result_SparseColMat.to_DMatrixType(), expexted_jac);
       // NALGEBRA CRATE
        Jacobian_instance.lambdify_jacobian_DMatrix(  "x", vec!("y", "z"));
        let variables = &*Vectors_type_casting(vect, "Dense".to_string());
        let jac =  Jacobian_instance.jac_function.as_mut().unwrap();
        let result_DMatrix = jac.call(1.0,  variables.clone());
        assert_eq!(result_DMatrix.to_DMatrixType(),   expexted_jac);

        // NALGEBRA SPARSE CRATE
        Jacobian_instance.lambdify_jacobian_CsMatrix(  "x", vec!("y", "z"));
        let variables = &*Vectors_type_casting(vect, "Sparse_2".to_string());
        let jac =  Jacobian_instance.jac_function.as_mut().unwrap();
        let result_CsMatrix = jac.call(1.0,  variables.clone());
        assert_eq!(result_CsMatrix.to_DMatrixType(),   expexted_jac);
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

