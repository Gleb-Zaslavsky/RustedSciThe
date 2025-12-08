//! Parallel implementations for Jacobian and function vector compilation.
//!
//! This module provides optimized parallel versions of Jacobian matrix and function vector
//! compilation with support for parameterized systems. All functions use Rayon for parallel
//! processing and pre-compile symbolic expressions for efficient evaluation.

#![allow(non_camel_case_types)]

use crate::global::THRESHOLD as T;
use crate::symbolic::symbolic_engine::Expr;
use rayon::prelude::*;
use crate::symbolic::symbolic_functions::Jacobian;
use nalgebra::{DMatrix, DVector};
use std::sync::Mutex;

impl Jacobian {
    /// Sets the parameter names for the Jacobian system.
    ///
    /// # Arguments
    /// * `params` - Vector of parameter name strings
    pub fn set_params(&mut self, params: Vec<String>) {
        self.parameters_string = params;
    }

    /// Creates a parallel-compiled Jacobian matrix function with parameters.
    ///
    /// Pre-compiles all non-zero Jacobian elements in parallel and returns a closure
    /// that evaluates the Jacobian for given parameter and variable values.
    ///
    /// # Arguments
    /// * `jac` - Symbolic Jacobian matrix
    /// * `variable_str` - Variable names
    /// * `parameters` - Parameter names
    ///
    /// # Returns
    /// Boxed closure taking (parameters, variables) and returning DMatrix
    pub fn calc_jacobian_DMatrix_with_parameters_parallel_local(
        jac: Vec<Vec<Expr>>,
        variable_str: Vec<String>,
        parameters: Vec<String>,
    ) -> Box<dyn Fn(&DVector<f64>, &DVector<f64>) -> DMatrix<f64>> {
        // Concatenate all variables: arg + variables + parameters
        let mut all_variables: Vec<String> = vec![];
        all_variables.extend(parameters.clone());
        all_variables.extend(variable_str.clone());
        let vector_of_variables_len = jac.len();
        let vector_of_functions_len = variable_str.len();

        // Pre-compile jacobian positions
        let jacobian_positions: Vec<(usize, usize, Box<dyn Fn(&[f64]) -> f64 + Send + Sync>)> = (0
            ..vector_of_functions_len)
            .into_par_iter()
            .flat_map(|i| {
                (0..vector_of_variables_len)
                    .filter_map(|j| {
                        let symbolic_partial_derivative = &jac[i][j];
                        if !symbolic_partial_derivative.is_zero() {
                            let compiled_func: Box<dyn Fn(&[f64]) -> f64 + Send + Sync> =
                                Expr::lambdify_borrowed_thread_safe(
                                    &symbolic_partial_derivative,
                                    all_variables
                                        .iter()
                                        .map(|s| s.as_str())
                                        .collect::<Vec<_>>()
                                        .as_slice(),
                                );
                            Some((i, j, compiled_func))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        Box::new(move | params: &DVector<f64>, values: &DVector<f64>| -> DMatrix<f64> {
            // Concatenate x with v to match all_variables order
            let mut v_vec: Vec<f64> = vec![];
            v_vec.extend(params.iter().cloned());
            v_vec.extend(values.iter().cloned());
           

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
        })
    }

    /// Creates a parallel-compiled function vector with parameters.
    ///
    /// Pre-compiles all functions in parallel and returns a closure that evaluates
    /// the function vector for given parameter and variable values.
    ///
    /// # Arguments
    /// * `vector_of_functions` - Vector of symbolic expressions
    /// * `variable_str` - Variable names
    /// * `parameters` - Parameter names
    ///
    /// # Returns
    /// Boxed closure taking (parameters, variables) and returning DVector
    pub fn vector_funvector_DVector_with_parameters_parallel_local(
        vector_of_functions:Vec<Expr>,
        variable_str: Vec<String>,
        parameters: Vec<String>,
    ) -> Box<dyn Fn( &DVector<f64>, &DVector<f64>) -> DVector<f64>> {
         let mut all_variables: Vec<String> = vec![];
        all_variables.extend(parameters.clone());
        all_variables.extend(variable_str.clone());
        let all_variables: Vec<&str> = all_variables.iter().map(|x| x.as_str()).collect();
        let compiled_functions: Vec<Box<dyn Fn(&[f64]) -> f64 + Send + Sync>> = vector_of_functions
            .par_iter()
            .map(|func| Expr::lambdify_borrowed_thread_safe(func,  all_variables.as_slice()))
            .collect();
         Box::new(move |params: &DVector<f64>, values: &DVector<f64>| -> DVector<f64> {
            let mut v_vec: Vec<f64> = vec![];
            v_vec.extend(params.iter().cloned());
            v_vec.extend(values.iter().cloned());
         
            let result: Vec<_> = compiled_functions
                .par_iter()
                .map(|func| func(v_vec.as_slice()))
                .collect();
            DVector::from_vec(result)
        })

        
       
    }

    /// Compiles the Jacobian matrix with parameters and stores it in the instance.
    ///
    /// Uses parallel compilation and stores the result in `lambdified_jacobian_DMatrix_with_params`.
    pub fn lambdify_jacobian_DMatrix_with_parameters_parallel(&mut self) {
        let jac = self.symbolic_jacobian.clone();
        let variable_str = self.variable_string.clone();
        let parameters = self.parameters_string.clone();
        let jacobian_DMatrix_with_parameters = Jacobian::calc_jacobian_DMatrix_with_parameters_parallel_local(
            jac,
            variable_str,
            parameters,
        );
        self.lambdified_jacobian_DMatrix_with_params = jacobian_DMatrix_with_parameters;
    }
    /// Compiles and returns the Jacobian matrix with parameters without mutating the instance.
    ///
    /// # Returns
    /// Boxed closure for Jacobian evaluation with parameters
    pub fn lambdify_jacobian_DMatrix_with_parameters_parallel_unmut(&self) -> Box<dyn Fn(&DVector<f64>, &DVector<f64>) -> DMatrix<f64>> {
        let jac = self.symbolic_jacobian.clone();
        let variable_str = self.variable_string.clone();
        let parameters = self.parameters_string.clone();
        let jacobian_DMatrix_with_parameters = Jacobian::calc_jacobian_DMatrix_with_parameters_parallel_local(
            jac,
            variable_str,
            parameters,
        );
        jacobian_DMatrix_with_parameters
    }

    /// Compiles the function vector with parameters and stores it in the instance.
    ///
    /// Uses parallel compilation and stores the result in `lambdified_function_with_params`.
    pub fn lambdify_vector_funvector_DVector_with_parameters_parallel(&mut self) {
        let vector_of_functions = self.vector_of_functions.clone();
        let variable_str = self.variable_string.clone();
        let parameters = self.parameters_string.clone();
        let vector_funvector_DVector_with_parameters = Jacobian::vector_funvector_DVector_with_parameters_parallel_local(
            vector_of_functions,
            variable_str,
            parameters,
        );
        self.lambdified_function_with_params = vector_funvector_DVector_with_parameters;
    }

    
    /// Compiles the Jacobian matrix without parameters and stores it in the instance.
    ///
    /// Uses parallel compilation and stores the result in `lambdified_jacobian_DMatrix`.
    pub fn lambdify_jacobian_DMatrix_parallel(&mut self) {
        let jac = self.symbolic_jacobian.clone();
        let variable_str = self.variable_string.clone();
        let jacobian_DMatrix_with_parameters = Jacobian::calc_jacobian_DMatrix_with_parameters_parallel_local(
            jac,
            variable_str,
            vec![],
        );
        let jacobian_DMatrix = move |x: &DVector<f64>| jacobian_DMatrix_with_parameters(&DVector::from_vec(vec![]), x);
        self.lambdified_jacobian_DMatrix = Box::new(jacobian_DMatrix);
    }

    /// Compiles the function vector without parameters and stores it in the instance.
    ///
    /// Uses parallel compilation and stores the result in `lambdified_function_DVector`.
    pub fn lambdify_vector_funvector_DVector(&mut self) {
        let vector_of_functions = self.vector_of_functions.clone();
        let variable_str = self.variable_string.clone();
        let vector_funvector_DVector_with_parameters = Jacobian::vector_funvector_DVector_with_parameters_parallel_local(
            vector_of_functions,
            variable_str,
            vec![],
        );
        let vector_funvector_DVector = move |x: &DVector<f64>| vector_funvector_DVector_with_parameters(&DVector::from_vec(vec![]), x);
        self.lambdified_function_DVector = Box::new(vector_funvector_DVector);
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::symbolic_engine::Expr;

    #[test]
    fn test_lambdify_jacobian_with_parameters() {
        let mut jacobian = Jacobian::new();
        
        // System: f1 = a*x + b*y, f2 = c*x*y
        // where a, b, c are parameters and x, y are variables
        let eq1 = Expr::parse_expression("a*x + b*y");
        let eq2 = Expr::parse_expression("c*x*y");
        let eq_system = vec![eq1, eq2];
        
        jacobian.set_vector_of_functions(eq_system);
        jacobian.variable_string = vec!["x".to_string(), "y".to_string()];
        jacobian.parameters_string = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        jacobian.set_variables(vec!["x", "y"]);
        jacobian.calc_jacobian();
        
        jacobian.lambdify_jacobian_DMatrix_with_parameters_parallel();
        
        let params = DVector::from_vec(vec![2.0, 3.0, 4.0]); // a=2, b=3, c=4
        let values = DVector::from_vec(vec![1.0, 2.0]); // x=1, y=2
        
        let jac_result = (jacobian.lambdified_jacobian_DMatrix_with_params)(&params, &values);
        
        // Expected Jacobian:
        // df1/dx = a = 2, df1/dy = b = 3
        // df2/dx = c*y = 4*2 = 8, df2/dy = c*x = 4*1 = 4
        assert_eq!(jac_result[(0, 0)], 2.0);
        assert_eq!(jac_result[(0, 1)], 3.0);
        assert_eq!(jac_result[(1, 0)], 8.0);
        assert_eq!(jac_result[(1, 1)], 4.0);
    }

    #[test]
    fn test_lambdify_funcvector_with_parameters() {
        let mut jacobian = Jacobian::new();
        
        // System: f1 = a*x + b, f2 = c*x^2
        let eq1 = Expr::parse_expression("a*x + b");
        let eq2 = Expr::parse_expression("c*x^2");
        let eq_system = vec![eq1, eq2];
        
        jacobian.set_vector_of_functions(eq_system);
        jacobian.variable_string = vec!["x".to_string()];
        jacobian.parameters_string = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        
        jacobian.lambdify_vector_funvector_DVector_with_parameters_parallel();
        
        let params = DVector::from_vec(vec![2.0, 3.0, 4.0]); // a=2, b=3, c=4
        let values = DVector::from_vec(vec![5.0]); // x=5
        
        let func_result = (jacobian.lambdified_function_with_params)(&params, &values);
        
        // Expected: f1 = 2*5 + 3 = 13, f2 = 4*25 = 100
        assert_eq!(func_result[0], 13.0);
        assert_eq!(func_result[1], 100.0);
    }

    #[test]
    fn test_lambdify_jacobian_parallel() {
        let mut jacobian = Jacobian::new();
        
        // System: f1 = x^2 + y, f2 = x*y^2
        let eq1 = Expr::parse_expression("x^2 + y");
        let eq2 = Expr::parse_expression("x*y^2");
        let eq_system = vec![eq1, eq2];
        
        jacobian.set_vector_of_functions(eq_system);
        jacobian.variable_string = vec!["x".to_string(), "y".to_string()];
        jacobian.set_variables(vec!["x", "y"]);
        jacobian.calc_jacobian();
        
        jacobian.lambdify_jacobian_DMatrix_parallel();
        
        let values = DVector::from_vec(vec![3.0, 2.0]); // x=3, y=2
        let jac_result = (jacobian.lambdified_jacobian_DMatrix)(&values);
        
        // Expected Jacobian:
        // df1/dx = 2*x = 6, df1/dy = 1
        // df2/dx = y^2 = 4, df2/dy = 2*x*y = 12
        assert_eq!(jac_result[(0, 0)], 6.0);
        assert_eq!(jac_result[(0, 1)], 1.0);
        assert_eq!(jac_result[(1, 0)], 4.0);
        assert_eq!(jac_result[(1, 1)], 12.0);
    }

    #[test]
    fn test_lambdify_funcvector() {
        let mut jacobian = Jacobian::new();
        
        // System: f1 = x^2 + 1, f2 = 2*x
        let eq1 = Expr::parse_expression("x^2 + 1");
        let eq2 = Expr::parse_expression("2*x");
        let eq_system = vec![eq1, eq2];
        
        jacobian.set_vector_of_functions(eq_system);
        jacobian.variable_string = vec!["x".to_string()];
        
        jacobian.lambdify_vector_funvector_DVector();
        
        let values = DVector::from_vec(vec![4.0]); // x=4
        let func_result = (jacobian.lambdified_function_DVector)(&values);
        
        // Expected: f1 = 16 + 1 = 17, f2 = 8
        assert_eq!(func_result[0], 17.0);
        assert_eq!(func_result[1], 8.0);
    }
}
