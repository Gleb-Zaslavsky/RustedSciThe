use rayon::prelude::*;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions::Jacobian;
use nalgebra::{DMatrix, DVector, };
use std::sync::{Arc, Mutex};
// Update the Jacobian implementation with parallel versions
impl Jacobian {
    pub fn generate_NR_solver_for_Radau_parallel(
        &mut self,
        eq_system: Vec<Expr>,
        values: Vec<String>,
        parameters: Vec<String>,
        arg: String,
    ) {
        self.set_vector_of_functions(eq_system);
        self.set_variables(values.iter().map(|x| x.as_str()).collect());
        self.calc_jacobian(); // Use the smart parallel version
        let ncols = self.symbolic_jacobian.len();
        let nrows = self.symbolic_jacobian[0].len();
        assert!(nrows == ncols);
        self.jacobian_generate_IVP_Radau_parallel(arg.as_str(), values.clone(), parameters.clone());
        let values_str = values.iter().map(|x| x.as_str()).collect::<Vec<&str>>();
        let parameters_str = parameters.iter().map(|x| x.as_str()).collect::<Vec<&str>>();
        self.lambdify_funcvector_with_parameters_parallel(
            arg.as_str(),
            values_str.clone(),
            parameters_str.clone(),
        );
        self.vector_funvector_with_parameters_DVector_parallel(
            arg.as_str(),
            values_str.clone(),
            parameters_str.clone(),
        );
    }

    /// Parallel jacobian generation following the BVP pattern
    pub fn jacobian_generate_IVP_Radau_parallel(
        &mut self,
        arg: &str,
        variable_str: Vec<String>,
        parameters: Vec<String>,
    ) {
        let symbolic_jacobian = self.symbolic_jacobian.clone();
        let vector_of_functions_len = self.vector_of_functions.len();
        let vector_of_variables_len = self.vector_of_variables.len();
        let bandwidth = self.bandwidth;
        
        let new_jac = Jacobian::calc_jacobian_fun_with_parameters_parallel(
            symbolic_jacobian,
            vector_of_functions_len,
            vector_of_variables_len,
            variable_str.iter().map(|s| s.to_string()).collect(),
            parameters.iter().map(|s| s.to_string()).collect(),
            arg.to_string(),
            bandwidth,
        );

        self.function_jacobian_IVP_DMatrix = new_jac;
    }

    /// Parallel jacobian computation following the BVP pattern
    pub fn calc_jacobian_fun_with_parameters_parallel(
        jac: Vec<Vec<Expr>>,
        vector_of_functions_len: usize,
        vector_of_variables_len: usize,
        variable_str: Vec<String>,
        parameters: Vec<String>,
        arg: String,
        bandwidth: (usize, usize),
    ) -> Box<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64>> {
        let mut variables_and_parameters: Vec<String> = variable_str.clone();
        variables_and_parameters.extend(parameters.clone());

        let (kl, ku) = bandwidth;
        Box::new(move |x: f64, v: &DVector<f64>| -> DMatrix<f64> {
            let mut vector_of_derivatives = vec![0.0; vector_of_functions_len * vector_of_variables_len];
            let vector_of_derivatives_mutex = Mutex::new(&mut vector_of_derivatives);
            
            let variales_and_parameters: Vec<&str> = variables_and_parameters
                .iter()
                .map(|s| s.as_str())
                .collect();

            // Parallel computation of jacobian elements
            (0..vector_of_functions_len).into_par_iter().for_each(|i| {
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
                    let symbolic_partial_derivative = &jac[i][j];
                    if !symbolic_partial_derivative.is_zero() {
                        let partial_func = Expr::lambdify_IVP_owned(
                            jac[i][j].clone(),
                            arg.as_str(),
                            variales_and_parameters.clone(),
                        );
                        let v_vec: Vec<f64> = v.iter().cloned().collect();
                        let P = partial_func(x, v_vec.clone());
                        if P.abs() > 1e-8 {
                            vector_of_derivatives_mutex.lock().unwrap()
                                [i * vector_of_functions_len + j] = P;
                        }
                    }
                }
            });

            DMatrix::from_row_slice(
                vector_of_functions_len,
                vector_of_variables_len,
                &vector_of_derivatives,
            )
        })
    }

    /// Parallel function vector evaluation
    /// 
    
    pub fn lambdify_funcvector_with_parameters_parallel(
        &mut self,
        arg: &str,
        variable_str: Vec<&str>,
        parameters: Vec<&str>,
    ) {
        let mut variable_and_parameters: Vec<&str> = variable_str.clone();
        variable_and_parameters.extend(parameters.clone());
        println!("variable_and_parameters = {:?} \n", variable_and_parameters);
        
        // Create thread-safe lambdified functions
        let lambdified_funcs: Vec<_> = self.vector_of_functions
            .iter()
            .map(|func| {
                Expr::lambdify_IVP_owned(func.clone(), arg, variable_and_parameters.clone())
            })
            .collect();

        // Store as thread-safe functions
        self.lambdified_functions_IVP = lambdified_funcs;
    }

/// Alternative: Store functions that can be safely sent between threads
    pub fn lambdify_funcvector_with_parameters_parallel_safe(
        &mut self,
        arg: &str,
        variable_str: Vec<&str>,
        parameters: Vec<&str>,
    ) {
        let mut variable_and_parameters: Vec<&str> = variable_str.clone();
        variable_and_parameters.extend(parameters.clone());
        
        // Instead of storing the functions, we'll recreate them when needed
        // This avoids the Send/Sync issue entirely
        println!("variable_and_parameters = {:?} \n", variable_and_parameters);
    }


    /// Parallel vector function evaluation following BVP pattern
    pub fn vector_funvector_with_parameters_DVector_parallel(
        &mut self,
        arg: &str,
        variable_str: Vec<&str>,
        parameters: Vec<&str>,
    ) {
        let vector_of_functions = &self.vector_of_functions;
        
        fn f_parallel(
            vector_of_functions: Vec<Expr>,
            arg: String,
            variable_and_parameters: Vec<String>,
        ) -> Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64> + 'static> {
            Box::new(move |x: f64, v: &DVector<f64>| -> DVector<f64> {
                let v_vec: Vec<f64> = v.iter().cloned().collect();
                
                // Parallel evaluation following BVP pattern
                let result: Vec<_> = vector_of_functions
                    .par_iter()
                    .map(|func| {
                        let func = Expr::lambdify_IVP_owned(
                            func.to_owned(),
                            arg.as_str(),
                            variable_and_parameters
                                .iter()
                                .map(|s| s.as_str())
                                .collect::<Vec<_>>()
                                .clone(),
                        );
                        func(x, v_vec.clone())
                    })
                    .collect();
                
                DVector::from_vec(result)
            })
        }
        
        let mut variable_and_parameters = variable_str.clone();
        variable_and_parameters.extend(parameters.clone());

        self.lambdified_functions_IVP_DVector = f_parallel(
            vector_of_functions.to_owned(),
            arg.to_string(),
            variable_and_parameters
                .clone()
                .iter()
                .map(|s| s.to_string())
                .collect(),
        );
    }
}
