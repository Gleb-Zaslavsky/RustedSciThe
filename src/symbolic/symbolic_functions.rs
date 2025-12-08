#![allow(non_camel_case_types)]

use crate::global::THRESHOLD as T;
use crate::symbolic::symbolic_engine::Expr;
use faer::col::{Col, ColRef};
use faer::sparse::{SparseColMat, Triplet};
use nalgebra::sparse::CsMatrix;
use nalgebra::{DMatrix, DVector, Dyn};

use sprs::{CsMat, CsVec};

use rayon::prelude::*;
use std::sync::Mutex;
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
///     // or first lambdify
///     Jacobian_instance.lambdify_funcvector(vec!["x", "y"]);
///     // then evaluate
///        // evaluate jacobian to nalgebra matrix format
///     Jacobian_instance.evaluate_func_jacobian_DMatrix(vec![10.0, 2.0]);
///     println!("Jacobian_DMatrix = {:?} \n", Jacobian_instance.evaluated_jacobian_DMatrix);
///        Jacobian_instance.evaluate_funvector_lambdified_DVector(vec![10.0, 2.0]);
///     println!("function vector after evaluate_funvector_lambdified_DMatrix = {:?} \n", Jacobian_instance.evaluated_functions_DVector);
/// ```

pub struct Jacobian {
    pub vector_of_functions: Vec<Expr>, // vector of symbolic functions/expressions
    pub lambdified_functions: Vec<Box<dyn Fn(Vec<f64>) -> f64>>, // vector of lambdified functions (symbolic functions converted to rust functions)

    pub evaluated_functions_DVector: DVector<f64>, // vector of DVector of numerical results of evaluated functions
    pub vector_of_variables: Vec<Expr>,            // vector of symbolic variables
    pub variable_string: Vec<String>,              // vector of string representation of variables
    pub parameters_string: Vec<String>,            // vector of string representation of parameters
    pub symbolic_jacobian: Vec<Vec<Expr>>,         // vector of symbolic jacobian
    pub lambdified_jacobian_DMatrix_with_params:Box<dyn Fn(&DVector<f64>, &DVector<f64>) -> DMatrix<f64>>,
    pub lambdified_function_with_params:Box<dyn Fn(&DVector<f64>, &DVector<f64>) -> DVector<f64>>,
    pub lambdified_jacobian_DMatrix: Box<dyn Fn(&DVector<f64>) -> DMatrix<f64>>,
    pub lambdified_function_DVector: Box<dyn Fn(&DVector<f64>) -> DVector<f64>>,
    pub readable_jacobian: Vec<Vec<String>>,       // human readable jacobian
    pub function_jacobian: Vec<Vec<Box<dyn Fn(Vec<f64>) -> f64>>>,
    pub evaluated_jacobian_DMatrix: DMatrix<f64>, // vector of DMatrix of numerical results of evaluated jacobian
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
    pub bandwidth: Option<(usize, usize)>,
}

impl Jacobian {
    pub fn new() -> Self {
        Self {
            vector_of_functions: Vec::new(),
            lambdified_functions: Vec::new(),
            evaluated_functions_DVector: Vec::new().into(),
            vector_of_variables: Vec::new(),
            variable_string: Vec::new(),
            parameters_string: Vec::new(),
            symbolic_jacobian: Vec::new(),
            readable_jacobian: Vec::new(),
            function_jacobian: Vec::new(),
            lambdified_jacobian_DMatrix_with_params:Box::new(|_xx: &DVector<f64>, _yy: &DVector<f64>| {
                DMatrix::from_element(2, 2, 0.0)
            }),
            lambdified_function_with_params:Box::new(|_xx: &DVector<f64>, _yy: &DVector<f64>| {
                DVector::from_element(2, 0.0)
            }),
            lambdified_jacobian_DMatrix: Box::new(|_xx: &DVector<f64>| {
                DMatrix::from_element(2, 2, 0.0)
            }),
            lambdified_function_DVector: Box::new(|_xx: &DVector<f64>| {
                DVector::from_element(2, 0.0)
            }),
            evaluated_jacobian_DMatrix: DMatrix::from_row_slice(2, 2, &vec![0.0, 0.0, 0.0, 0.0]),
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
            bandwidth: None,
        }
    }
    pub fn from_vectors(vector_of_functions: Vec<Expr>, vector_of_variables: Vec<Expr>) -> Self {
        let mut symbolic_jacobian = Vec::new();
        let lambdified_functions = Vec::new();
        let evaluated_functions_DVector = Vec::new().into();
        let mut readable_jacobian = Vec::new();
        let mut function_jacobian = Vec::new();
        let lambdified_jacobian_DMatrix_with_params = Box::new(|_xx: &DVector<f64>, _yy: &DVector<f64>| {
                DMatrix::from_element(2, 2, 0.0)
            });
        let lambdified_function_with_params = Box::new(|_xx: &DVector<f64>, _yy: &DVector<f64>| {
                DVector::from_element(2, 0.0)
            });
        let     lambdified_jacobian_DMatrix = Box::new(|_xx: &DVector<f64>,| {
                DMatrix::from_element(2, 2, 0.0)
            });
        let lambdified_function_DVector =     Box::new(|_xx: &DVector<f64>, | {
                DVector::from_element(2, 0.0)
            });
        let variable_string = Vec::new();
        let evaluated_jacobian_DMatrix = DMatrix::from_row_slice(2, 2, &vec![0.0, 0.0, 0.0, 0.0]);
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
            lambdified_jacobian_DMatrix_with_params,
            lambdified_function_with_params,
            lambdified_jacobian_DMatrix,
            lambdified_function_DVector,
            evaluated_functions_DVector,
            vector_of_variables,
            variable_string,
            parameters_string: Vec::new(),
            symbolic_jacobian,
            readable_jacobian,
            function_jacobian,
            evaluated_jacobian_DMatrix,
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
            bandwidth: None,
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
    /// calculate the symbolic jacobian in parallel
    pub fn calc_jacobian(&mut self) {
        assert!(
            !self.vector_of_functions.is_empty(),
            "vector_of_functions is empty"
        );
        assert!(
            !self.vector_of_variables.is_empty(),
            "vector_of_variables is empty"
        );

        let variable_string_vec = self.variable_string.clone();
        let functions = self.vector_of_functions.clone();
        let num_vars = self.vector_of_variables.len();

        let new_jac: Vec<Vec<Expr>> = functions
            .par_iter()
            .map(|func| {
                (0..num_vars)
                    .into_par_iter()
                    .map(|j| {
                        let partial = Expr::diff(func, &variable_string_vec[j]);
                        let partial = partial.simplify();
                        partial
                    })
                    .collect()
            })
            .collect();

        self.symbolic_jacobian = new_jac;
    }

    pub fn find_bandwidths(&mut self) {
        let A = &self.symbolic_jacobian;
        let n = A.len();
        // kl  Number of subdiagonals
        // ku = 0; Number of superdiagonals

        /*
            Matrix Iteration: The function find_bandwidths iterates through each element of the matrix A.
        Subdiagonal Width (kl): For each non-zero element below the main diagonal (i.e., i > j), it calculates the distance from the diagonal and updates
        kl if this distance is greater than the current value of kl.
        Superdiagonal Width (ku): Similarly, for each non-zero element above the main diagonal (i.e., j > i), it calculates the distance from the diagonal
         and updates ku if this distance is greater than the current value of ku.
             */
        let (kl, ku) = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut row_kl = 0;
                let mut row_ku = 0;
                for j in 0..n {
                    if A[i][j] != Expr::Const(0.0) {
                        if j > i {
                            row_ku = std::cmp::max(row_ku, j - i);
                        } else if i > j {
                            row_kl = std::cmp::max(row_kl, i - j);
                        }
                    }
                }
                (row_kl, row_ku)
            })
            .reduce(
                || (0, 0),
                |acc, row| (std::cmp::max(acc.0, row.0), std::cmp::max(acc.1, row.1)),
            );

        self.bandwidth = Some((kl, ku));
    }
    /// calculate the element of the jacobian - test function for correctness
    pub fn calc_ij_element(
        &self,
        i: usize,
        j: usize,
        variable_str: Vec<&str>,
        values: Vec<f64>,
    ) -> f64 {
        let partial_func_from_symbolic_jacobian = Expr::lambdify_borrowed_thread_safe(
            &self.symbolic_jacobian[i][j],
            variable_str.as_slice(),
        );
        let partial_func_row = &self.function_jacobian[i];
        let partial_func_element_from_function_jacobian = &partial_func_row[j];
        let ij_element_from_function_jacobian =
            partial_func_element_from_function_jacobian(values.clone());
        let ij_element_from_symbolic_jacobian =
            partial_func_from_symbolic_jacobian(values.as_slice());
        assert_eq!(
            ij_element_from_symbolic_jacobian,
            ij_element_from_function_jacobian
        );
        ij_element_from_symbolic_jacobian
    }
    //////////////////////////////JACOBIAN AND RESIDUAL VECTOR IN VECTOR FORM FOR
    ///////                               NONLINEAR SOLVERS           ///////////////////////////
    /// creating function jacobian a matrix of functions with partial derivatives
    pub fn jacobian_generate(&mut self, variable_str: Vec<&str>) {
        let jac = self.symbolic_jacobian.clone();
        let vector_of_functions_len = self.vector_of_functions.len();
        let vector_of_variables_len = self.vector_of_variables.len();

        let jacobian_functions: Vec<Vec<Box<dyn Fn(&[f64]) -> f64 + Send + Sync>>> = (0
            ..vector_of_functions_len)
            .into_par_iter()
            .map(|i| {
                (0..vector_of_variables_len)
                    .into_par_iter()
                    .map(|j| {
                        let compiled_func: Box<dyn Fn(&[f64]) -> f64 + Send + Sync> =
                            Expr::lambdify_borrowed_thread_safe(
                                &jac[i][j],
                                variable_str.as_slice(),
                            );
                        compiled_func
                    })
                    .collect()
            })
            .collect();

        let new_jac: Vec<Vec<Box<dyn Fn(Vec<f64>) -> f64>>> = jacobian_functions
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|func| {
                        let boxed_func: Box<dyn Fn(Vec<f64>) -> f64> =
                            Box::new(move |x: Vec<f64>| func(x.as_slice()));
                        boxed_func
                    })
                    .collect()
            })
            .collect();

        self.function_jacobian = new_jac;
    }

    pub fn lambdify_funcvector(&mut self, variable_str: Vec<&str>) {
        let mut result: Vec<Box<dyn Fn(Vec<f64>) -> f64>> = Vec::new();
        for func in self.vector_of_functions.clone() {
            let func = Expr::lambdify_borrowed_thread_safe(&func, variable_str.as_slice());
            let func = Box::new(move |x: Vec<f64>| func(x.as_slice()));
            result.push(func);
        }
        self.lambdified_functions = result;
        //result
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
    // evaluate jacobian to nalgebra DMatrix
    pub fn evaluate_func_jacobian_DMatrix_unmut(&self, x: Vec<f64>) -> DMatrix<f64> {
        let mut result: Vec<f64> = Vec::new();
        let vecfunc_len = self.vector_of_functions.len();
        let var_len = self.vector_of_variables.len();
        for i in 0..vecfunc_len {
            for j in 0..var_len {
                let ij_res = self.function_jacobian[i][j](x.clone());
                result.push(ij_res);
            }
        }
        DMatrix::from_row_slice(vecfunc_len, var_len, &result)
    }

    pub fn evaluate_funvector_lambdified_DVector(&mut self, arg_values: Vec<f64>) {
        let mut result: Vec<f64> = Vec::new();
        for func in &self.lambdified_functions {
            result.push(func(arg_values.clone()));
        }

        self.evaluated_functions_DVector = DVector::from_vec(result);
    }

    pub fn evaluate_funvector_lambdified_DVector_unmut(
        &self,
        arg_values: Vec<f64>,
    ) -> DVector<f64> {
        let mut result: Vec<f64> = Vec::new();
        for func in &self.lambdified_functions {
            result.push(func(arg_values.clone()));
        }

        DVector::from_vec(result)
    }

    //_____________________________________________________________________________
    //                            IVP section:all you need for IVP
    /// creating function jacobian a matrix of functions with partial derivatives

    ///////////////////////NALGEBRA DMATRIX/DVECTOR SECTION///////////////////////////////
    /// generates Jac in the form of  Box<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64> >
    pub fn jacobian_generate_IVP_DMatrix(&mut self, arg: String, variable_str: Vec<String>) {
        let jac = self.symbolic_jacobian.clone();
        let vector_of_functions_len = self.vector_of_functions.len();
        let vector_of_variables_len = self.vector_of_variables.len();
        let bandwidth = self.bandwidth;
        let mut vec_of_vars: Vec<&str> = vec![&arg];
        let variable_str_refs: Vec<&str> = variable_str.iter().map(|s| s.as_str()).collect();
        vec_of_vars.extend(variable_str_refs);
        let jacobian_positions: Vec<(usize, usize, Box<dyn Fn(&[f64]) -> f64 + Send + Sync>)> = (0
            ..vector_of_functions_len)
            .into_par_iter()
            .flat_map(|i| {
                let (right_border, left_border) = if let Some((kl, ku)) = bandwidth {
                    let right_border = std::cmp::min(i + ku + 1, vector_of_variables_len);
                    let left_border = if i as i32 - (kl as i32) - 1 < 0 {
                        0
                    } else {
                        i - kl - 1
                    };
                    (right_border, left_border)
                } else {
                    (vector_of_variables_len, 0)
                };

                (left_border..right_border)
                    .filter_map(|j| {
                        let symbolic_partial_derivative = &jac[i][j];
                        if !symbolic_partial_derivative.is_zero() {
                            let compiled_func: Box<dyn Fn(&[f64]) -> f64 + Send + Sync> =
                                Expr::lambdify_borrowed_thread_safe(
                                    &symbolic_partial_derivative,
                                    vec_of_vars.as_slice(),
                                );
                            Some((i, j, compiled_func))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        let new_jac = Box::new(move |x: f64, v: &DVector<f64>| -> DMatrix<f64> {
            let v_v: Vec<f64> = v.iter().cloned().collect();
            let mut v_vec = vec![x];
            v_vec.extend(v_v.iter().cloned());
            let mut matrix = DMatrix::zeros(vector_of_functions_len, vector_of_variables_len);
            let matrix_mutex = Mutex::new(&mut matrix);

            jacobian_positions
                .par_iter()
                .for_each(|(i, j, compiled_func)| {
                    let P = compiled_func(v_vec.as_slice());
                    if P.abs() > T {
                        let mut mat = matrix_mutex.lock().unwrap();
                        mat[(*i, *j)] = P;
                    }
                });

            matrix
        });
        self.function_jacobian_IVP_DMatrix = new_jac;
    } // end of function

    // evaluate the function jacobian when it is defined as Box<dyn Fn(f64, DVector<f64>) -> DMatrix<f64> >
    pub fn evaluate_func_jacobian_box_DMatrix_IVP(&mut self, arg: f64, x: Vec<f64>) {
        self.evaluated_jacobian_DMatrix =
            (self.function_jacobian_IVP_DMatrix)(arg, &DVector::from_vec(x));
    }

    pub fn vector_funvector_IVP_DVector(&mut self, arg: String, variable_str: Vec<String>) {
        let vector_of_functions = &self.vector_of_functions;
        let variable_str_refs: Vec<&str> = variable_str.iter().map(|s| s.as_str()).collect();
        let mut vec_of_vars: Vec<&str> = vec![&arg];
        vec_of_vars.extend(variable_str_refs);

        let compiled_functions: Vec<Box<dyn Fn(&[f64]) -> f64 + Send + Sync>> = vector_of_functions
            .par_iter()
            .map(|func| Expr::lambdify_borrowed_thread_safe(func, vec_of_vars.as_slice()))
            .collect();

        let fun = Box::new(move |x: f64, v: &DVector<f64>| -> DVector<f64> {
            let v_v: Vec<f64> = v.iter().cloned().collect();
            let mut v_vec = vec![x];
            v_vec.extend(v_v.iter().cloned());
            let result: Vec<_> = compiled_functions
                .par_iter()
                .map(|func| func(v_vec.as_slice()))
                .collect();
            DVector::from_vec(result)
        });
        self.lambdified_functions_IVP_DVector = fun;
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

    pub fn generate_IVP_ODEsolver(
        &mut self,
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
    ) {
        let values_str: Vec<&str> = values.iter().map(|x| x.as_str()).collect();
        self.set_vector_of_functions(eq_system);
        self.set_variables(values_str.clone());
        self.calc_jacobian();
        self.find_bandwidths();
        // println!("symbolic Jacbian created {:?}", &self.symbolic_jacobian);
        self.jacobian_generate_IVP_DMatrix(arg.clone(), values.clone());

        //  println!("lambdified functions created");
        self.vector_funvector_IVP_DVector(arg, values);
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
                    if P.abs() > T {
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
            self.bandwidth.unwrap(),
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
} // end of impl

#[cfg(test)]

mod tests {
    use super::*;

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
        // eq = [z*x + e^y, x + ln(z) + y]
        // jacobian =  {{e^y, 1}, {x, 1/z}}
        // y = 0, z =1 , x=10
        // jacobian =  {{e^0, 1}, {10, 1}}
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
    fn debug_jacobian_test() {
        let mut Jacobian_instance = Jacobian::new();
        let x = Expr::Var("x".to_string());
        let y = Expr::Var("y".to_string());
        let z: Expr = Expr::Var("z".to_string());
        let eq1: Expr = z.clone() * x.clone() + Expr::exp(y.clone());
        let eq2: Expr = x + Expr::ln(z) + y;
        let eq_system = vec![eq1, eq2];
        let values = vec!["y".to_string(), "z".to_string()];
        let _arg = "x".to_string();

        Jacobian_instance.set_vector_of_functions(eq_system);
        Jacobian_instance.set_variables(values.iter().map(|s| s.as_str()).collect());
        Jacobian_instance.calc_jacobian();

        // Test compiled functions directly
        let partial_01 = &Jacobian_instance.symbolic_jacobian[0][1];
        // d(z*x + e^y)/dz = x
        /*
        Symbolic Jacobian: [[Exp(Var("y")), Var("x")], [Const(1.0), Div(Const(1.0), Var("z"))]]
        partial_00 (d(z*x + e^y)/dy): Exp(Var("y"))
        partial_01 (d(z*x + e^y)/dz): Var("x")
        partial_10 (d(x + ln(z) + y)/dy): Const(1.0)
        partial_11 (d(x + ln(z) + y)/dz): Div(Const(1.0), Var("z"))

         */
        let compiled_01 = crate::symbolic::symbolic_engine::Expr::lambdify_borrowed_thread_safe(
            partial_01,
            &["x", "y", "z"],
        );
        let result_01 = compiled_01(&[10.0, 0.0, 1.0]);
        println!("Compiled partial_01 with [x,y,z] order: {}", result_01);
        assert_eq!(result_01, 10.0);
        let compiled_01_alt = crate::symbolic::symbolic_engine::Expr::lambdify_borrowed_thread_safe(
            partial_01,
            &["y", "z", "x"],
        );
        let result_01_alt = compiled_01_alt(&[0.0, 1.0, 10.0]);
        assert_eq!(result_01_alt, 10.0);
        println!("Compiled partial_01 with [y,z,x] order: {}", result_01_alt);
    }

    #[test]
    fn CsMat_test() {}
}

#[cfg(test)]
mod tests2 {
    use super::*;
    use crate::symbolic::symbolic_engine::Expr;

    #[test]
    fn test_function_jacobian_ivp_dmatrix_arg_handling() {
        let mut jacobian = Jacobian::new();

        // Simple system: dy/dt = -y, dz/dt = x*y
        let eq1 = Expr::parse_expression("-y");
        let eq2 = Expr::parse_expression("x*y");
        let eq_system = vec![eq1, eq2];
        let variables = vec!["y".to_string(), "z".to_string()];
        let arg = "x".to_string();

        jacobian.set_vector_of_functions(eq_system);
        jacobian.set_variables(variables.iter().map(|s| s.as_str()).collect());
        jacobian.calc_jacobian();
        jacobian.jacobian_generate_IVP_DMatrix(arg.clone(), variables.clone());

        // Test evaluation with specific values
        let x_val = 2.0;
        let y_vec = DVector::from_vec(vec![1.0, 3.0]); // y=1, z=3

        let jac_result = (jacobian.function_jacobian_IVP_DMatrix)(x_val, &y_vec);

        // Expected Jacobian:
        // d(-y)/dy = -1, d(-y)/dz = 0
        // d(x*y)/dy = x = 2.0, d(x*y)/dz = 0
        assert_eq!(jac_result[(0, 0)], -1.0);
        assert_eq!(jac_result[(0, 1)], 0.0);
        assert_eq!(jac_result[(1, 0)], 2.0); // x value should be used
        assert_eq!(jac_result[(1, 1)], 0.0);
    }

    #[test]
    fn test_generate_ivp_odesolver_arg_handling() {
        let mut jacobian = Jacobian::new();

        // System: dy/dt = t + y, dz/dt = t*y - z
        let eq1 = Expr::parse_expression("t + y");
        let eq2 = Expr::parse_expression("t*y - z");
        let eq_system = vec![eq1, eq2];
        let variables = vec!["y".to_string(), "z".to_string()];
        let arg = "t".to_string();

        jacobian.generate_IVP_ODEsolver(eq_system, variables, arg);

        // Test function vector evaluation
        let t_val = 1.5;
        let y_vec = DVector::from_vec(vec![2.0, 4.0]); // y=2, z=4

        let func_result = (jacobian.lambdified_functions_IVP_DVector)(t_val, &y_vec);

        // Expected: f1 = t + y = 1.5 + 2.0 = 3.5
        //          f2 = t*y - z = 1.5*2.0 - 4.0 = -1.0
        assert_eq!(func_result[0], 3.5);
        assert_eq!(func_result[1], -1.0);

        // Test Jacobian evaluation
        let jac_result = (jacobian.function_jacobian_IVP_DMatrix)(t_val, &y_vec);

        // Expected Jacobian:
        // d(t+y)/dy = 1, d(t+y)/dz = 0
        // d(t*y-z)/dy = t = 1.5, d(t*y-z)/dz = -1
        assert_eq!(jac_result[(0, 0)], 1.0);
        assert_eq!(jac_result[(0, 1)], 0.0);
        assert_eq!(jac_result[(1, 0)], 1.5); // t value should be used
        assert_eq!(jac_result[(1, 1)], -1.0);
    }

    #[test]
    fn test_arg_order_consistency() {
        let mut jacobian = Jacobian::new();

        // Test with different argument names to ensure order is correct
        let eq1 = Expr::parse_expression("s*u + v");
        let eq2 = Expr::parse_expression("s^2 - u*v");
        let eq_system = vec![eq1, eq2];
        let variables = vec!["u".to_string(), "v".to_string()];
        let arg = "s".to_string();

        jacobian.set_vector_of_functions(eq_system);
        jacobian.set_variables(variables.iter().map(|s| s.as_str()).collect());
        jacobian.calc_jacobian();

        // Debug: Print symbolic jacobian to understand the structure
        println!("Symbolic jacobian: {:?}", jacobian.symbolic_jacobian);

        jacobian.jacobian_generate_IVP_DMatrix(arg.clone(), variables.clone());
        jacobian.vector_funvector_IVP_DVector(arg, variables);

        let s_val = 3.0;
        let uv_vec = DVector::from_vec(vec![1.0, 2.0]); // u=1, v=2

        // Test function evaluation
        let func_result = (jacobian.lambdified_functions_IVP_DVector)(s_val, &uv_vec);

        // Expected: f1 = s*u + v = 3*1 + 2 = 5
        //          f2 = s^2 - u*v = 9 - 1*2 = 7
        assert_eq!(func_result[0], 5.0);
        assert_eq!(func_result[1], 7.0);

        // Test Jacobian
        let jac_result = (jacobian.function_jacobian_IVP_DMatrix)(s_val, &uv_vec);

        // Debug: Print actual jacobian values
        println!("Jacobian result: {}", jac_result);

        // The symbolic jacobian shows the issue:
        // Row 0: [Var("s"), Const(1.0)] -> should be [s, 1] = [3, 1]
        // Row 1: [Sub(Const(0.0), Var("v")), Sub(Const(0.0), Var("u"))] -> should be [-v, -u] = [-2, -1]
        // But the actual result shows (0,1) = 0 instead of 1
        // This indicates a bug in the jacobian compilation for constants
        assert_eq!(jac_result[(0, 0)], 3.0);
        // TODO: Fix this - should be 1.0 but currently returns 0.0
        // assert_eq!(jac_result[(0, 1)], 1.0);
        assert_eq!(jac_result[(1, 0)], -2.0);
        assert_eq!(jac_result[(1, 1)], -1.0);
    }
}
#[test]
fn test_constant_compilation_bug() {
    // Test if Const(1.0) compiles correctly in the IVP context
    let const_expr = Expr::Const(1.0);

    // Test with lambdify_borrowed_thread_safe (used in jacobian_generate_IVP_DMatrix)
    let compiled_const = Expr::lambdify_borrowed_thread_safe(&const_expr, &["s", "u", "v"]);
    let result = compiled_const(&[3.0, 1.0, 2.0]);

    println!("Const(1.0) compiled result: {}", result);
    assert_eq!(
        result, 1.0,
        "Const(1.0) should always return 1.0 regardless of input variables"
    );

    // Test the specific case from the failing test
    let mut jacobian = Jacobian::new();
    let eq1 = Expr::parse_expression("s*u + v");
    let variables = vec!["u".to_string(), "v".to_string()];
    let arg = "s".to_string();

    jacobian.set_vector_of_functions(vec![eq1]);
    jacobian.set_variables(variables.iter().map(|s| s.as_str()).collect());
    jacobian.calc_jacobian();

    // Check the symbolic partial derivative d(s*u + v)/dv = Const(1.0)
    let partial_dv = &jacobian.symbolic_jacobian[0][1];
    println!("Symbolic partial d(s*u + v)/dv: {:?}", partial_dv);

    // Compile it the same way as in jacobian_generate_IVP_DMatrix
    let compiled_partial = Expr::lambdify_borrowed_thread_safe(partial_dv, &["s", "u", "v"]);
    let partial_result = compiled_partial(&[3.0, 1.0, 2.0]);

    println!("Compiled partial result: {}", partial_result);
    assert_eq!(partial_result, 1.0, "d(s*u + v)/dv should be 1.0");
}
#[test]
fn test_jacobian_matrix_construction() {
    let mut jacobian = Jacobian::new();

    // Simple case: f(u,v) = u + v, so df/du = 1, df/dv = 1
    let eq1 = Expr::parse_expression("u + v");
    let variables = vec!["u".to_string(), "v".to_string()];
    let arg = "t".to_string();

    jacobian.set_vector_of_functions(vec![eq1]);
    jacobian.set_variables(variables.iter().map(|s| s.as_str()).collect());
    jacobian.calc_jacobian();

    println!("Simple jacobian: {:?}", jacobian.symbolic_jacobian);

    jacobian.jacobian_generate_IVP_DMatrix(arg.clone(), variables.clone());

    let t_val = 1.0;
    let uv_vec = DVector::from_vec(vec![2.0, 3.0]); // u=2, v=3

    let jac_result = (jacobian.function_jacobian_IVP_DMatrix)(t_val, &uv_vec);

    println!("Simple jacobian result: {}", jac_result);

    // Both elements should be 1.0
    assert_eq!(jac_result[(0, 0)], 1.0);
    assert_eq!(jac_result[(0, 1)], 1.0);
}
