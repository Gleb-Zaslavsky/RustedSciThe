#![allow(non_camel_case_types)]

use crate::symbolic::symbolic_engine::Expr;
use faer::col::{Col, ColRef};
use faer::sparse::{SparseColMat, Triplet};
use log::info;
use nalgebra::sparse::CsMatrix;
use nalgebra::{DMatrix, DVector, Dyn};
use regex::Regex;
use sprs::{CsMat, CsVec};
use std::collections::{HashMap, HashSet};
///
/// calculate symbolic jacobian and evaluate it
/// Example#
/// ```
/// use RustedSciThe::symbolic::symbolic_functions::Jacobian;
///  let mut Jacobian_instance = Jacobian::new();
///       // function of 2 or more arguments
///       let vec_of_expressions = vec![ "2*x^3+y".to_string(), "1".to_string()];
///       // set vector of functions
///      Jacobian_instance.set_funcvecor_from_str(vec_of_expressions);
///      // set vector of variables
///      Jacobian_instance.set_varvecor_from_str("x, y");
///      // calculate symbolic jacobian
///      Jacobian_instance.calc_jacobian();
///      // transform into human...kind of readable form
///      Jacobian_instance.readable_jacobian();
///      // generate jacobian made of regular rust functions
///      Jacobian_instance.jacobian_generate(vec!["x", "y"]);

///     println!("Jacobian_instance: functions  {:?}. Variables {:?}", Jacobian_instance.vector_of_functions, Jacobian_instance.vector_of_variables);
///      println!("Jacobian_instance: Jacobian  {:?} readable {:?}.", Jacobian_instance.symbolic_jacobian, Jacobian_instance.readable_jacobian);
///     for i in 0.. Jacobian_instance.symbolic_jacobian.len() {
///       for j in 0.. Jacobian_instance.symbolic_jacobian[i].len() {
///        println!("Jacobian_instance: Jacobian  {} row  {} colomn {:?}", i, j, Jacobian_instance.symbolic_jacobian[i][j]);
///      }  
///    }
///    // calculate element of jacobian (just for control)
///    let ij_element = Jacobian_instance.calc_ij_element(0, 0,  vec!["x", "y"],vec![10.0, 2.0]) ;
///    println!("ij_element = {:?} \n", ij_element);
///    // evaluate jacobian to numerical values
///     Jacobian_instance. lambdify_and_ealuate_funcvector(vec!["x", "y"], vec![10.0, 2.0]);
///     println!("function vector = {:?} \n", Jacobian_instance.evaluated_functions);
///    // lambdify and evaluate function vector to numerical values
///    Jacobian_instance. lambdify_and_ealuate_funcvector(vec!["x", "y"], vec![10.0, 2.0]);
///    println!("function vector = {:?} \n", Jacobian_instance.evaluated_functions);
///     // or first lambdify
///     Jacobian_instance.lambdify_funcvector(vec!["x", "y"]);
///     // then evaluate
///     Jacobian_instance.evaluate_funvector_lambdified(vec![10.0, 2.0]);
///     println!("function vector after evaluate_funvector_lambdified = {:?} \n", Jacobian_instance.evaluated_functions);
///        // evaluate jacobian to nalgebra matrix format
///     Jacobian_instance.evaluate_func_jacobian_DMatrix(vec![10.0, 2.0]);
///     println!("Jacobian_DMatrix = {:?} \n", Jacobian_instance.evaluated_jacobian_DMatrix);
///        Jacobian_instance.evaluate_funvector_lambdified_DVector(vec![10.0, 2.0]);
///     println!("function vector after evaluate_funvector_lambdified_DMatrix = {:?} \n", Jacobian_instance.evaluated_functions_DVector);
/// ```

pub struct Jacobian {
    pub vector_of_functions: Vec<Expr>, // vector of symbolic functions/expressions
    pub lambdified_functions: Vec<Box<dyn Fn(Vec<f64>) -> f64>>, // vector of lambdified functions (symbolic functions converted to rust functions)

    pub evaluated_functions: Vec<f64>, // vector of numerical results of evaluated functions
    pub evaluated_functions_DVector: DVector<f64>, // vector of DVector of numerical results of evaluated functions
    pub vector_of_variables: Vec<Expr>,            // vector of symbolic variables
    pub variable_string: Vec<String>,              // vector of string representation of variables
    pub symbolic_jacobian: Vec<Vec<Expr>>,         // vector of symbolic jacobian
    pub readable_jacobian: Vec<Vec<String>>,       // human readable jacobian
    pub function_jacobian: Vec<Vec<Box<dyn Fn(Vec<f64>) -> f64>>>,
    pub evaluated_jacobian: Vec<Vec<f64>>, // vector of numerical results of evaluated jacobian
    pub evaluated_jacobian_DMatrix: DMatrix<f64>, // vector of DMatrix of numerical results of evaluated jacobian

    pub function_jacobian_IVP: Vec<Vec<Box<dyn Fn(f64, Vec<f64>) -> f64>>>,
    pub lambdified_functions_IVP: Vec<Box<dyn Fn(f64, Vec<f64>) -> f64>>,
    pub function_jacobian_IVP_DMatrix: Box<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64>>,
    pub lambdified_functions_IVP_DVector: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,

    pub function_jacobian_IVP_CsMat: Box<dyn Fn(f64, &CsVec<f64>) -> CsMat<f64>>,
    pub lambdified_functions_IVP_CsVec: Box<dyn Fn(f64, &CsVec<f64>) -> CsVec<f64>>,

    pub function_jacobian_IVP_CsMatrix: Box<dyn Fn(f64, &DVector<f64>) -> CsMatrix<f64>>,

    pub function_jacobian_IVP_SparseColMat: Box<dyn Fn(f64, &Col<f64>) -> SparseColMat<usize, f64>>,
    pub lambdified_functions_IVP_Col: Box<dyn Fn(f64, &Col<f64>) -> Col<f64>>,
    pub bounds: Option<Vec<(f64, f64)>>,
    pub rel_tolerance_vec: Option<Vec<f64>>,
    pub bandwidth: (usize, usize),
}

impl Jacobian {
    pub fn new() -> Self {
        Self {
            vector_of_functions: Vec::new(),
            lambdified_functions: Vec::new(),
            evaluated_functions: Vec::new(),
            evaluated_functions_DVector: Vec::new().into(),
            vector_of_variables: Vec::new(),
            variable_string: Vec::new(),
            symbolic_jacobian: Vec::new(),
            readable_jacobian: Vec::new(),
            function_jacobian: Vec::new(),
            evaluated_jacobian: Vec::new(),
            evaluated_jacobian_DMatrix: DMatrix::from_row_slice(2, 2, &vec![0.0, 0.0, 0.0, 0.0]),

            function_jacobian_IVP: Vec::new(),
            lambdified_functions_IVP: Vec::new(),
            function_jacobian_IVP_DMatrix: Box::new(|_xx: f64, _yy: &DVector<f64>| {
                DMatrix::from_element(2, 2, 0.0)
            }),
            lambdified_functions_IVP_DVector: Box::new(|_xx: f64, _y: &DVector<f64>| {
                DVector::from_element(2, 0.0)
            }),

            function_jacobian_IVP_CsMat: Box::new(|_xx: f64, _y: &CsVec<f64>| CsMat::zero((1, 1))),
            lambdified_functions_IVP_CsVec: Box::new(|_xx: f64, _y: &CsVec<f64>| CsVec::empty(1)),

            function_jacobian_IVP_CsMatrix: Box::new(|_xx: f64, _y: &DVector<f64>| {
                CsMatrix::new_uninitialized_generic(Dyn(2), Dyn(2), 1)
            }),
            function_jacobian_IVP_SparseColMat: Box::new(|_xx: f64, _y: &Col<f64>| {
                SparseColMat::<usize, f64>::try_new_from_triplets(1, 1, &[Triplet::new(0, 0, 0.0)])
                    .unwrap()
            }),
            lambdified_functions_IVP_Col: Box::new(|_xx: f64, _y: &Col<f64>| Col::zeros(0)),
            bounds: None,
            rel_tolerance_vec: None,
            bandwidth: (0, 0),
        }
    }
    pub fn from_vectors(vector_of_functions: Vec<Expr>, vector_of_variables: Vec<Expr>) -> Self {
        let mut symbolic_jacobian = Vec::new();
        let lambdified_functions = Vec::new();
        let evaluated_functions_DVector = Vec::new().into();
        let mut readable_jacobian = Vec::new();
        let evaluated_functions = Vec::new();
        let mut function_jacobian = Vec::new();
        let variable_string = Vec::new();
        let evaluated_jacobian = Vec::new();
        let evaluated_jacobian_DMatrix = DMatrix::from_row_slice(2, 2, &vec![0.0, 0.0, 0.0, 0.0]);

        let function_jacobian_IVP = Vec::new();
        let lambdified_functions_IVP = Vec::new();
        let function_jacobian_IVP_DMatrix =
            Box::new(|_xx: f64, _y: &DVector<f64>| DMatrix::from_element(2, 2, 0.0));
        let lambdified_functions_IVP_DVector =
            Box::new(|_xx: f64, _y: &DVector<f64>| DVector::from_element(2, 0.0));

        let function_jacobian_IVP_CsMat = Box::new(|_xx: f64, _y: &CsVec<f64>| CsMat::zero((1, 1)));
        let lambdified_functions_IVP_CsVec = Box::new(|_xx: f64, _y: &CsVec<f64>| CsVec::empty(1));
        let function_jacobian_IVP_CsMatrix = Box::new(|_xx: f64, _y: &DVector<f64>| {
            CsMatrix::new_uninitialized_generic(Dyn(2), Dyn(2), 1)
        });
        let function_jacobian_IVP_SparseColMat = Box::new(|_xx: f64, _y: &Col<f64>| {
            SparseColMat::<usize, f64>::try_new_from_triplets(1, 1, &[Triplet::new(0, 0, 0.0)])
                .unwrap()
        });
        let lambdified_functions_IVP_Col = Box::new(|_xx: f64, _y: &Col<f64>| Col::zeros(0));
        let bounds = Some(Vec::new());
        let rel_tolerance_vec = Some(Vec::new());
        for _ in 0..vector_of_functions.len() {
            symbolic_jacobian.push(Vec::new());
            function_jacobian.push(Vec::new());
            readable_jacobian.push(Vec::new());
        }
        Self {
            vector_of_functions,
            lambdified_functions,
            evaluated_functions,
            evaluated_functions_DVector,
            vector_of_variables,
            variable_string,
            symbolic_jacobian,
            readable_jacobian,
            function_jacobian,
            evaluated_jacobian,
            evaluated_jacobian_DMatrix,
            function_jacobian_IVP,
            lambdified_functions_IVP,
            function_jacobian_IVP_DMatrix,
            lambdified_functions_IVP_DVector,
            function_jacobian_IVP_CsMat,
            lambdified_functions_IVP_CsVec,
            function_jacobian_IVP_CsMatrix,
            function_jacobian_IVP_SparseColMat,
            lambdified_functions_IVP_Col,
            bounds,
            rel_tolerance_vec,
            bandwidth: (0, 0),
        }
    }
    /// Basic functionality: setting, pushing, and getting variables and functions
    pub fn set_i_variable(&mut self, index: usize, value: Expr) {
        self.vector_of_variables[index] = value;
    }
    pub fn set_j_function(&mut self, index: usize, value: Expr) {
        self.vector_of_functions[index] = value;
    }
    pub fn set_vector_of_variables(&mut self, value: Vec<Expr>) {
        self.vector_of_variables = value;
    }
    pub fn set_vector_of_functions(&mut self, value: Vec<Expr>) {
        self.vector_of_functions = value;
    }
    pub fn push_to_funcvector(&mut self, value: Expr) {
        self.vector_of_functions.push(value);
    }
    pub fn push_to_varvector(&mut self, value: Expr) {
        self.vector_of_variables.push(value);
    }
    pub fn set_funcvecor_from_str(&mut self, value: Vec<String>) {
        let vec = value
            .iter()
            .map(|x| Expr::parse_expression(x))
            .collect::<Vec<Expr>>();
        self.vector_of_functions = vec;
    }
    pub fn set_varvecor_from_str(&mut self, symbols: &str) {
        let vec_trimmed: Vec<String> = symbols.split(',').map(|s| s.trim().to_string()).collect();
        self.variable_string = vec_trimmed;
        let enter = Expr::Symbols(symbols);
        self.vector_of_variables = enter;
    }
    pub fn set_variables(&mut self, varvec: Vec<&str>) {
        let vec_trimmed: Vec<String> = varvec.iter().map(|s| s.trim().to_string()).collect();
        let symbols = vec_trimmed.join(",");
        self.variable_string = vec_trimmed;
        let enter = Expr::Symbols(&symbols);
        self.vector_of_variables = enter;
    }
    /// turn jacobian into readable format
    pub fn readable_jacobian(&mut self) {
        let mut readable_jac: Vec<Vec<String>> = Vec::new();
        for i in 0..self.symbolic_jacobian.len() {
            let mut readable_jac_string: Vec<String> = Vec::new();
            for j in 0..self.symbolic_jacobian[i].len() {
                let element_to_str =
                    Expr::sym_to_str(&self.symbolic_jacobian[i][j], &self.variable_string[0]);
                readable_jac_string.push(element_to_str);
            }
            readable_jac.push(readable_jac_string);
        }
        self.readable_jacobian = readable_jac;
    }
    /// calculate the symbolic jacobian  
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
    /// calculate the element of the jacobian - test function for correctness
    pub fn calc_ij_element(
        &self,
        i: usize,
        j: usize,
        variable_str: Vec<&str>,
        values: Vec<f64>,
    ) -> f64 {
        let partial_func_from_symbolic_jacobian =
            Expr::lambdify_owned(self.symbolic_jacobian[i][j].clone(), variable_str.clone());
        let partial_func_row = &self.function_jacobian[i];
        let partial_func_element_from_function_jacobian = &partial_func_row[j];
        let ij_element_from_function_jacobian =
            partial_func_element_from_function_jacobian(values.clone());
        let ij_element_from_symbolic_jacobian = partial_func_from_symbolic_jacobian(values);
        assert_eq!(
            ij_element_from_symbolic_jacobian,
            ij_element_from_function_jacobian
        );
        ij_element_from_symbolic_jacobian
    }

    pub fn calc_jacobian_fun(
        jac: Vec<Vec<Expr>>,
        vector_of_functions_len: usize,
        vector_of_variables_len: usize,
        variable_str: Vec<&str>,
    ) -> Vec<Vec<Box<dyn Fn(Vec<f64>) -> f64>>> {
        let mut new_function_jacobian: Vec<Vec<Box<dyn Fn(Vec<f64>) -> f64>>> = Vec::new();

        for i in 0..vector_of_functions_len {
            let mut vector_of_partial_derivatives_func: Vec<Box<dyn Fn(Vec<f64>) -> f64>> =
                Vec::new();

            for j in 0..vector_of_variables_len {
                //  println!("i = {}, j = {}", i, j);
                let partial_func = Expr::lambdify_owned(jac[i][j].clone(), variable_str.clone());

                //  println!("partial_func = {:?}", partial_func(vec![10.0,1.0]));
                vector_of_partial_derivatives_func.push(partial_func);
            }
            // println!("vector_of_partial_derivatives_func = {:?}, {:?}", vector_of_partial_derivatives_func[0](vec![10.0,1.0]), vector_of_partial_derivatives_func[1](vec![10.0,1.0])   );
            new_function_jacobian.push(vector_of_partial_derivatives_func);
        }

        new_function_jacobian
    }

    /// creating function jacobian a matrix of functions with partial derivatives
    pub fn jacobian_generate(&mut self, variable_str: Vec<&str>) {
        let symbolic_jacobian = self.symbolic_jacobian.clone();
        let symbolic_jacobian_rc = symbolic_jacobian.clone();

        let vector_of_functions_len = self.vector_of_functions.len();
        let vector_of_variables_len = self.vector_of_variables.len();

        let new_jac = Jacobian::calc_jacobian_fun(
            symbolic_jacobian_rc,
            vector_of_functions_len,
            vector_of_variables_len,
            variable_str,
        );

        self.function_jacobian = new_jac;
    }

    /// numeical evaluate jacobian function
    pub fn evaluate_func_jacobian(&mut self, x: &Vec<f64>) {
        let mut result: Vec<Vec<f64>> = Vec::new();
        for i in 0..self.vector_of_functions.len() {
            let mut result_row: Vec<f64> = Vec::new();
            for j in 0..self.vector_of_variables.len() {
                let ij_res = self.function_jacobian[i][j](x.clone());
                //  println!("jacobian element {} {} = {}", i, j, ij_res);
                result_row.push(ij_res);
            }
            result.push(result_row);
        }
        self.evaluated_jacobian = result
    }

    // evaluate jacobian to nalgebra DMatrix
    pub fn evaluate_func_jacobian_DMatrix(&mut self, x: Vec<f64>) {
        let mut result: Vec<f64> = Vec::new();
        let vecfunc_len = self.vector_of_functions.len();
        let var_len = self.vector_of_variables.len();
        for i in 0..vecfunc_len {
            for j in 0..var_len {
                let ij_res = self.function_jacobian[i][j](x.clone());
                result.push(ij_res);
            }
        }
        self.evaluated_jacobian_DMatrix = DMatrix::from_row_slice(vecfunc_len, var_len, &result);
    }

    // when you need to lambdify and evaluate a vector of function in one place
    pub fn lambdify_and_ealuate_funcvector(
        &mut self,
        variable_str: Vec<&str>,
        arg_values: Vec<f64>,
    ) {
        let mut result: Vec<f64> = Vec::new();
        for func in &self.vector_of_functions {
            let func = Expr::lambdify(func, variable_str.clone());
            result.push(func(arg_values.clone()));
        }
        self.evaluated_functions = result.clone();
        //result
    }

    pub fn lambdify_funcvector(&mut self, variable_str: Vec<&str>) {
        let mut result: Vec<Box<dyn Fn(Vec<f64>) -> f64>> = Vec::new();
        for func in self.vector_of_functions.clone() {
            let func = Expr::lambdify_owned(func, variable_str.clone());
            result.push(func);
        }
        self.lambdified_functions = result;
        //result
    }

    pub fn evaluate_funvector_lambdified(&mut self, arg_values: Vec<f64>) {
        let mut result: Vec<f64> = Vec::new();
        for func in &self.lambdified_functions {
            result.push(func(arg_values.clone()));
        }

        self.evaluated_functions = result.clone();
    }

    pub fn evaluate_funvector_lambdified_DVector(&mut self, arg_values: Vec<f64>) {
        let mut result: Vec<f64> = Vec::new();
        for func in &self.lambdified_functions {
            result.push(func(arg_values.clone()));
        }

        self.evaluated_functions_DVector = DVector::from_vec(result);
    }

    //_____________________________________________________________________________
    //                            IVP section:all you need for IVP
    /// creating function jacobian a matrix of functions with partial derivatives
    /// generates Jac in the form of a vector of vector of functions  Vec<Vec<Box<dyn Fn(f64, Vec<f64>) -> f64>>>
    pub fn calc_jacobian_fun_IVP(
        jac: Vec<Vec<Expr>>,
        vector_of_functions_len: usize,
        vector_of_variables_len: usize,
        variable_str: Vec<&str>,
        arg: &str,
    ) -> Vec<Vec<Box<dyn Fn(f64, Vec<f64>) -> f64>>> {
        let mut new_function_jacobian: Vec<Vec<Box<dyn Fn(f64, Vec<f64>) -> f64>>> = Vec::new();
        for i in 0..vector_of_functions_len {
            let mut vector_of_partial_derivatives_func: Vec<Box<dyn Fn(f64, Vec<f64>) -> f64>> =
                Vec::new();
            for j in 0..vector_of_variables_len {
                //  println!("i = {}, j = {}", i, j);
                let partial_func =
                    Expr::lambdify_IVP_owned(jac[i][j].clone(), arg, variable_str.clone());

                //  println!("partial_func = {:?}", partial_func(vec![10.0,1.0]));
                vector_of_partial_derivatives_func.push(partial_func);
            }
            // println!("vector_of_partial_derivatives_func = {:?}, {:?}", vector_of_partial_derivatives_func[0](vec![10.0,1.0]), vector_of_partial_derivatives_func[1](vec![10.0,1.0])   );
            new_function_jacobian.push(vector_of_partial_derivatives_func);
        }
        new_function_jacobian
    } // end of function

    pub fn jacobian_generate_IVP(&mut self, arg: &str, variable_str: Vec<&str>) {
        let symbolic_jacobian = self.symbolic_jacobian.clone();
        let symbolic_jacobian_rc = symbolic_jacobian.clone();

        let vector_of_functions_len = self.vector_of_functions.len();
        let vector_of_variables_len = self.vector_of_variables.len();

        let new_jac = Jacobian::calc_jacobian_fun_IVP(
            symbolic_jacobian_rc,
            vector_of_functions_len,
            vector_of_variables_len,
            variable_str,
            arg,
        );
        //  let mut boxed_jacobian: Box<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64>> = Box::new(|arg, variable_str_| {
        //    DMatrix::from_rows(&new_jac) }) ;
        self.function_jacobian_IVP = new_jac;
    }

    /// creating function jacobian a matrix of functions with partial derivatives
    /// generates Jac in the form of  Box<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64> >
    pub fn calc_jacobian_fun_IVP_DMatrix(
        jac: Vec<Vec<Expr>>,
        vector_of_functions_len: usize,
        vector_of_variables_len: usize,
        variable_str: Vec<String>,
        arg: String,
        bandwidth: (usize, usize),
    ) -> Box<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64>> {
        //let arg = arg.as_str();
        //let variable_str: Vec<&str> = variable_str.iter().map(|s| s.as_str()).collect();
        let (kl, ku) = bandwidth;
        Box::new(move |x: f64, v: &DVector<f64>| -> DMatrix<f64> {
            let mut new_function_jacobian: DMatrix<f64> =
                DMatrix::zeros(vector_of_functions_len, vector_of_variables_len);
            for i in 0..vector_of_functions_len {
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
                    // println!("i = {}, j = {}", i, j);
                    //  let now = std::time::Instant::now();
                    let partial_func = Expr::lambdify_IVP_owned(
                        jac[i][j].clone(),
                        arg.as_str(),
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
                        partial_func(x, v_vec.clone())
                    };

                    //    let time_test = now.elapsed().as_micros();
                    //   if time_test > 100 {println!("Elapsed time: {:?} micrs, {:?} ", now.elapsed().as_micros(), jac[i][j].clone() );}
                }
            }
            new_function_jacobian
        })
    } // end of function

    pub fn jacobian_generate_IVP_DMatrix(&mut self, arg: &str, variable_str: Vec<&str>) {
        let symbolic_jacobian = self.symbolic_jacobian.clone();
        let symbolic_jacobian_rc = symbolic_jacobian.clone();

        let vector_of_functions_len = self.vector_of_functions.len();
        let vector_of_variables_len = self.vector_of_variables.len();
        let bandwidth = self.bandwidth;
        let new_jac = Jacobian::calc_jacobian_fun_IVP_DMatrix(
            symbolic_jacobian_rc,
            vector_of_functions_len,
            vector_of_variables_len,
            variable_str.iter().map(|s| s.to_string()).collect(),
            arg.to_string(),
            bandwidth,
        );
        //  let mut boxed_jacobian: Box<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64>> = Box::new(|arg, variable_str_| {
        //    DMatrix::from_rows(&new_jac) }) ;
        self.function_jacobian_IVP_DMatrix = new_jac;
    }

    /// evaluate the function jacobian when it is defined as vec of vecs
    pub fn evaluate_func_jacobian_DMatrix_IVP(&mut self, arg: f64, x: Vec<f64>) {
        let mut result: Vec<f64> = Vec::new();
        let vecfunc_len = self.vector_of_functions.len();
        let var_len = self.vector_of_variables.len();
        for i in 0..vecfunc_len {
            for j in 0..var_len {
                let ij_res = self.function_jacobian_IVP[i][j](arg, x.clone());
                result.push(ij_res);
            }
        }
        self.evaluated_jacobian_DMatrix = DMatrix::from_row_slice(vecfunc_len, var_len, &result);
    }
    // evaluate the function jacobian when it is defined as Box<dyn Fn(f64, DVector<f64>) -> DMatrix<f64> >
    pub fn evaluate_func_jacobian_box_DMatrix_IVP(&mut self, arg: f64, x: Vec<f64>) {
        self.evaluated_jacobian_DMatrix =
            (self.function_jacobian_IVP_DMatrix)(arg, &DVector::from_vec(x));
    }

    pub fn lambdify_funcvector_IVP(&mut self, arg: &str, variable_str: Vec<&str>) {
        let mut result: Vec<Box<dyn Fn(f64, Vec<f64>) -> f64>> = Vec::new();
        for func in self.vector_of_functions.clone() {
            let func = Expr::lambdify_IVP_owned(func, arg, variable_str.clone());
            result.push(func);
        }
        self.lambdified_functions_IVP = result;
        //result
    }
    pub fn vector_funvector_IVP_DVector(&mut self, arg: &str, variable_str: Vec<&str>) {
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
        self.lambdified_functions_IVP_DVector = f(
            vector_of_functions.to_owned(),
            arg.to_string(),
            variable_str.clone().iter().map(|s| s.to_string()).collect(),
        );
    }
    /// evaluate RHS when it is defined as vec of functions
    pub fn evaluate_funvector_lambdified_DVector_IVP(&mut self, arg: f64, values: Vec<f64>) {
        let mut result: Vec<f64> = Vec::new();
        for func in &self.lambdified_functions_IVP {
            result.push(func(arg, values.clone()));
        }

        self.evaluated_functions_DVector = DVector::from_vec(result);
    }
    // evaluate RHS when it is defined as Box<dyn Fn(f64, DVector<f64>) -> DVector<f64> >
    pub fn evaluate_funvector_lambdified_box_DVector_IVP(&mut self, arg: f64, values: Vec<f64>) {
        self.evaluated_functions_DVector =
            (self.lambdified_functions_IVP_DVector)(arg, &DVector::from_vec(values));
    }

    // shortcut for all functions to create jacobain as a vector of vectors and RHS as vector of functions
    pub fn generate_IVP(&mut self, eq_system: Vec<Expr>, values: Vec<String>, arg: String) {
        let values: Vec<&str> = values.iter().map(|x| x.as_str()).collect();
        self.set_vector_of_functions(eq_system);
        self.set_variables(values.clone());
        self.calc_jacobian();
        self.jacobian_generate_IVP(arg.as_str(), values.clone());
        // self.generate_IVP_DMatrix_jacobian()
        self.lambdify_funcvector_IVP(arg.as_str(), values);
    }

    pub fn generate_IVP_ODEsolver(
        &mut self,
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
    ) {
        let values: Vec<&str> = values.iter().map(|x| x.as_str()).collect();
        self.set_vector_of_functions(eq_system);
        self.set_variables(values.clone());
        self.calc_jacobian();
        // println!("symbolic Jacbian created {:?}", &self.symbolic_jacobian);
        self.jacobian_generate_IVP_DMatrix(arg.as_str(), values.clone());
        // println!("functioon Jacobian created");
        self.lambdify_funcvector_IVP(arg.as_str(), values.clone());
        //  println!("lambdified functions created");
        self.vector_funvector_IVP_DVector(arg.as_str(), values);
        // println!("function vector created");
    }
    //_________________________SPARSE_MATRICES SECTION____________________________

    pub fn calc_jacobian_fun_IVP_CsMat(
        jac: Vec<Vec<Expr>>,
        vector_of_functions_len: usize,
        vector_of_variables_len: usize,
        variable_str: Vec<String>,
        arg: String,
    ) -> Box<dyn Fn(f64, &CsVec<f64>) -> CsMat<f64>> {
        //let arg = arg.as_str();
        //let variable_str: Vec<&str> = variable_str.iter().map(|s| s.as_str()).collect();

        Box::new(move |x: f64, v: &CsVec<f64>| -> CsMat<f64> {
            let mut new_function_jacobian: CsMat<f64> =
                CsMat::zero((vector_of_functions_len, vector_of_variables_len));
            for i in 0..vector_of_functions_len {
                for j in 0..vector_of_variables_len {
                    // println!("i = {}, j = {}", i, j);
                    let partial_func = Expr::lambdify_IVP_owned(
                        jac[i][j].clone(),
                        arg.as_str(),
                        variable_str
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .clone(),
                    );
                    //let v_vec: Vec<f64> = v.iter().cloned().collect();
                    let v_vec: Vec<f64> = v.to_dense().to_vec();
                    new_function_jacobian.insert(j, i, partial_func(x, v_vec.clone()))
                }
            }
            new_function_jacobian
        })
    } // end of function

    pub fn jacobian_generate_IVP_CsMat(&mut self, arg: &str, variable_str: Vec<&str>) {
        let symbolic_jacobian = self.symbolic_jacobian.clone();
        let symbolic_jacobian_rc = symbolic_jacobian.clone();

        let vector_of_functions_len = self.vector_of_functions.len();
        let vector_of_variables_len = self.vector_of_variables.len();

        let new_jac = Jacobian::calc_jacobian_fun_IVP_CsMat(
            symbolic_jacobian_rc,
            vector_of_functions_len,
            vector_of_variables_len,
            variable_str.iter().map(|s| s.to_string()).collect(),
            arg.to_string(),
        );
        //  let mut boxed_jacobian: Box<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64>> = Box::new(|arg, variable_str_| {
        //    DMatrix::from_rows(&new_jac) }) ;

        self.function_jacobian_IVP_CsMat = new_jac;
    }

    pub fn vector_funvector_IVP_CsVec(&mut self, arg: &str, variable_str: Vec<&str>) {
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
        self.lambdified_functions_IVP_CsVec = f(
            vector_of_functions.to_owned(),
            arg.to_string(),
            variable_str.clone().iter().map(|s| s.to_string()).collect(),
        );
    }

    pub fn calc_jacobian_fun_IVP_CsMatrix(
        jac: Vec<Vec<Expr>>,
        vector_of_functions_len: usize,
        vector_of_variables_len: usize,
        variable_str: Vec<String>,
        arg: String,
    ) -> Box<dyn Fn(f64, &DVector<f64>) -> CsMatrix<f64>> {
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
                    let partial_func = Expr::lambdify_IVP_owned(
                        jac[i][j].clone(),
                        arg.as_str(),
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
                    new_function_jacobian[(i, j)] = partial_func(x, v_vec.clone());
                }
            }
            new_function_jacobian.into()
        })
    } // end of function

    pub fn jacobian_generate_IVP_CsMatrix(&mut self, arg: &str, variable_str: Vec<&str>) {
        let symbolic_jacobian = self.symbolic_jacobian.clone();
        let symbolic_jacobian_rc = symbolic_jacobian.clone();

        let vector_of_functions_len = self.vector_of_functions.len();
        let vector_of_variables_len = self.vector_of_variables.len();

        let new_jac = Jacobian::calc_jacobian_fun_IVP_CsMatrix(
            symbolic_jacobian_rc,
            vector_of_functions_len,
            vector_of_variables_len,
            variable_str.iter().map(|s| s.to_string()).collect(),
            arg.to_string(),
        );
        //  let mut boxed_jacobian: Box<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64>> = Box::new(|arg, variable_str_| {
        //    DMatrix::from_rows(&new_jac) }) ;
        self.function_jacobian_IVP_CsMatrix = new_jac;
    }

    pub fn calc_jacobian_fun_IVP_SparseColMat(
        jac: Vec<Vec<Expr>>,
        vector_of_functions_len: usize,
        vector_of_variables_len: usize,
        variable_str: Vec<String>,
        arg: String,
        bandwidth: (usize, usize),
    ) -> Box<dyn Fn(f64, &Col<f64>) -> SparseColMat<usize, f64>> {
        //let arg = arg.as_str();
        //let variable_str: Vec<&str> = variable_str.iter().map(|s| s.as_str()).collect();

        Box::new(move |x: f64, v: &Col<f64>| -> SparseColMat<usize, f64> {
            let (kl, ku) = bandwidth;
            let _number_of_possible_non_zero: usize = jac.len();
            //let mut new_function_jacobian: CsMatrix<f64> = CsMatrix::new_uninitialized_generic(Dyn(vector_of_functions_len),
            //Dyn(vector_of_variables_len), number_of_possible_non_zero);
            let mut vector_of_triplets = Vec::new();
            for i in 0..vector_of_functions_len {
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
                    // if jac[i][j] != Expr::Const(0.0) { println!("i = {}, j = {}, {}", i, j,  &jac[i][j]);}
                    let partial_func = Expr::lambdify_IVP_owned(
                        jac[i][j].clone(),
                        arg.as_str(),
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
                    let P = partial_func(x, v_vec.clone());
                    if P.abs() > 1e-8 {
                        let triplet = Triplet::new(i, j, P);
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

    pub fn jacobian_generate_IVP_SparseColMat(&mut self, arg: &str, variable_str: Vec<&str>) {
        let symbolic_jacobian = self.symbolic_jacobian.clone();
        let symbolic_jacobian_rc = symbolic_jacobian.clone();

        let vector_of_functions_len = self.vector_of_functions.len();
        let vector_of_variables_len = self.vector_of_variables.len();

        let new_jac = Jacobian::calc_jacobian_fun_IVP_SparseColMat(
            symbolic_jacobian_rc,
            vector_of_functions_len,
            vector_of_variables_len,
            variable_str.iter().map(|s| s.to_string()).collect(),
            arg.to_string(),
            self.bandwidth,
        );
        //  let mut boxed_jacobian: Box<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64>> = Box::new(|arg, variable_str_| {
        //    DMatrix::from_rows(&new_jac) }) ;
        self.function_jacobian_IVP_SparseColMat = new_jac;
    }
    pub fn vector_funvector_IVP_Col(&mut self, arg: &str, variable_str: Vec<&str>) {
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
                let res = ColRef::from_slice(result.as_slice()).to_owned();
                res
            }) //enf of box
        } // end of function
        self.lambdified_functions_IVP_Col = f(
            vector_of_functions.to_owned(),
            arg.to_string(),
            variable_str.clone().iter().map(|s| s.to_string()).collect(),
        );
    }

    //__________________________BVP SECTION____________________________
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
        log::info!(
            "creating discretization equations with n_steps = {}, H = {:?} ({}), T_list = {:?} ({})",
            n_steps,
            H,
            H.len(),
            T_list,
            T_list.len()
        );

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

        self.bandwidth = (kl, ku);
    }

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
    ) {
        let param = if let Some(param) = param {
            param
        } else {
            arg.clone()
        };

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

        let v = self.variable_string.clone();
        println!(
            "VECTOR OF FUNCTIONS \n \n  {:?},  length: {} \n \n",
            &self.vector_of_functions,
            &self.vector_of_functions.len()
        );
        println!("INDEXED VARIABLES {:?}, length:  {} \n \n", &v, &v.len());
        let indexed_values: Vec<&str> = v.iter().map(|x| x.as_str()).collect();
        self.calc_jacobian();
        self.find_bandwidths();
        println!("kl, ku {:?}", self.bandwidth);
        //  println!("symbolic Jacbian created {:?}", &self.symbolic_jacobian);
        self.jacobian_generate_IVP_DMatrix(param.as_str(), indexed_values.clone());
        let n = &self.symbolic_jacobian.len();
        for (_i, vec_s) in self.symbolic_jacobian.iter().enumerate() {
            assert_eq!(vec_s.len(), *n, "jacobian not square ");
        }
        // println!("functioon Jacobian created");
        self.lambdify_funcvector_IVP(param.as_str(), indexed_values.clone());
        //  println!("lambdified functions created");
        self.vector_funvector_IVP_DVector(param.as_str(), indexed_values);
        // println!("function vector created");
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
    }

    pub fn generate_BVP_CsMat(
        &mut self,
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        t0: f64,
        _param: Option<String>,
        n_steps: Option<usize>,
        h: Option<f64>,
        mesh: Option<Vec<f64>>,
        BorderConditions: HashMap<String, (usize, f64)>,
        Bounds: Option<HashMap<String, (f64, f64)>>,
        rel_tolerance: Option<HashMap<String, f64>>,
        scheme: String,
    ) {
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
        println!("{:?}", &self.vector_of_functions);
        let v = self.variable_string.clone();
        let indexed_values: Vec<&str> = v.iter().map(|x| x.as_str()).collect();
        self.calc_jacobian();
        self.find_bandwidths();
        self.jacobian_generate_IVP_CsMat(arg.as_str(), indexed_values.clone());
        self.lambdify_funcvector_IVP(arg.as_str(), indexed_values.clone());
        self.vector_funvector_IVP_CsVec(arg.as_str(), indexed_values);
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
    }

    pub fn generate_BVP_CsMatrix(
        &mut self,
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        t0: f64,
        _param: Option<String>,
        n_steps: Option<usize>,
        h: Option<f64>,
        mesh: Option<Vec<f64>>,
        BorderConditions: HashMap<String, (usize, f64)>,
        Bounds: Option<HashMap<String, (f64, f64)>>,
        rel_tolerance: Option<HashMap<String, f64>>,
        scheme: String,
    ) {
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
        println!("{:?}", &self.vector_of_functions);
        let v = self.variable_string.clone();
        let indexed_values: Vec<&str> = v.iter().map(|x| x.as_str()).collect();
        self.calc_jacobian();
        let n = &self.symbolic_jacobian.len();
        for (_i, vec_s) in self.symbolic_jacobian.iter().enumerate() {
            assert_eq!(vec_s.len(), *n, "jacobian not square ");
        }
        self.find_bandwidths();
        self.jacobian_generate_IVP_CsMatrix(arg.as_str(), indexed_values.clone());
        self.lambdify_funcvector_IVP(arg.as_str(), indexed_values.clone());
        self.vector_funvector_IVP_DVector(arg.as_str(), indexed_values);
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
    }

    pub fn generate_BVP_SparseColMat(
        &mut self,
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        t0: f64,
        _param: Option<String>,
        n_steps: Option<usize>,
        h: Option<f64>,
        mesh: Option<Vec<f64>>,
        BorderConditions: HashMap<String, (usize, f64)>,
        Bounds: Option<HashMap<String, (f64, f64)>>,
        rel_tolerance: Option<HashMap<String, f64>>,
        scheme: String,
    ) {
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
        info!("system disretized");
        println!(
            "\n \n vector of functions \n \n {:?} \n \n of lengh {}",
            &self.vector_of_functions,
            &self.vector_of_functions.len()
        );
        let v = self.variable_string.clone();
        let indexed_values: Vec<&str> = v.iter().map(|x| x.as_str()).collect();
        println!(
            "\n \n indexed values \n \n {:?} of length {}",
            &self.variable_string,
            &self.variable_string.len()
        );
        self.calc_jacobian();
        info!("jacobian calculated");
        // println!("\n \n symbolic jacobian \n \n{:?}", &self.symbolic_jacobian);
        let n = &self.symbolic_jacobian.len();
        for (_i, vec_s) in self.symbolic_jacobian.iter().enumerate() {
            if n != &vec_s.len() {
                println!(
                    "\n \n symbolic jacobian consists of {:?} vectors, each of length {}\n \n it means it is not square!",
                    n,
                    vec_s.len()
                );
            }
            assert_eq!(
                vec_s.len(),
                *n,
                "jacobian not square! symbolic jacobian consists of {:?} vectors, each of length {}",
                n,
                vec_s.len()
            );
        }

        self.find_bandwidths();
        println!("bandwidths found");
        println!("kl, ku {:?}", self.bandwidth);
        self.jacobian_generate_IVP_SparseColMat(arg.as_str(), indexed_values.clone());
        self.lambdify_funcvector_IVP(arg.as_str(), indexed_values.clone());
        self.vector_funvector_IVP_Col(arg.as_str(), indexed_values);
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
    }
} // end of impl

#[cfg(test)]

mod tests {
    use super::*;

    #[test]
    fn generate_IVP_test() {
        let mut Jacobian_instance = Jacobian::new();
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let z: Expr = Expr::Var("z".to_string());
        let eq1: Expr = z.clone() * x.clone() + Expr::exp(y.clone());
        let eq2: Expr = x + Expr::ln(z) + y;
        let eq_system = vec![eq1, eq2];
        let values = vec!["y".to_string(), "z".to_string()];
        let arg = "x".to_string();
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

    #[test]
    fn generate_IVP_ODEsolver_test() {
        let mut Jacobian_instance = Jacobian::new();
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let z: Expr = Expr::Var("z".to_string());
        let eq1: Expr = z.clone() * x.clone() + Expr::exp(y.clone());
        let eq2: Expr = x + Expr::ln(z) + y;
        let eq_system = vec![eq1, eq2];
        let values = vec!["y".to_string(), "z".to_string()];
        let arg = "x".to_string();
        Jacobian_instance.generate_IVP_ODEsolver(eq_system, values, arg);
        // J= {{e^y, 10}, {1, 1/z}}
        //
        Jacobian_instance.evaluate_func_jacobian_box_DMatrix_IVP(10.0, vec![0.0, 1.0]);
        println!(
            "Jacobian_instance.evaluated_jacobian_DMatrix = {:?}",
            Jacobian_instance.evaluated_jacobian_DMatrix
        );
        assert_eq!(
            Jacobian_instance.evaluated_jacobian_DMatrix,
            DMatrix::from_row_slice(2, 2, &[1.0, 10.0, 1.0, 1.0])
        );
        Jacobian_instance.evaluate_funvector_lambdified_box_DVector_IVP(10.0, vec![0.0, 1.0]);
        println!(
            "Jacobian_instance.evaluated_functions_DVector = {:?}",
            Jacobian_instance.evaluated_functions_DVector
        );
        assert_eq!(
            Jacobian_instance.evaluated_functions_DVector,
            DVector::from_vec(vec![11.0, 10.0])
        );
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
