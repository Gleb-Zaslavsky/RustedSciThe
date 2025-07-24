use crate::numerical::Radau::Radau_newton::RadauNewton;
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions::Jacobian;
use log::info;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use std::sync::Mutex;
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
            let mut vector_of_derivatives =
                vec![0.0; vector_of_functions_len * vector_of_variables_len];
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
        let lambdified_funcs: Vec<_> = self
            .vector_of_functions
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
// Update the RadauNewton implementation
impl RadauNewton {
    /// Generate parallel function and Jacobian closures with proper thread safety
    pub fn eq_generate_parallel(&mut self) {
        info!("Generating parallel Radau Newton equations and jacobian");
        let mut jacobian_instance = Jacobian::new();

        jacobian_instance.set_vector_of_functions(self.eq_system.clone());
        jacobian_instance.set_variables(self.k_variables.iter().map(|x| x.as_str()).collect());
        jacobian_instance.calc_jacobian();

        let values_str = self
            .k_variables
            .iter()
            .map(|x| x.as_str())
            .collect::<Vec<&str>>();
        let parameters_str = self
            .parameters
            .iter()
            .map(|x| x.as_str())
            .collect::<Vec<&str>>();

        // Use the safe parallel jacobian generation
        let symbolic_jacobian = jacobian_instance.symbolic_jacobian.clone();
        let new_jac = Jacobian::calc_jacobian_fun_with_parameters_parallel(
            symbolic_jacobian,
            jacobian_instance.vector_of_functions.len(),
            jacobian_instance.vector_of_variables.len(),
            self.k_variables.clone(),
            self.parameters.clone(),
            self.arg.clone(),
            jacobian_instance.bandwidth,
        );

        // Use the safe parallel vector function generation
        jacobian_instance.vector_funvector_with_parameters_DVector_parallel(
            self.arg.as_str(),
            values_str.clone(),
            parameters_str.clone(),
        );

        self.jacobian_symbolic = Some(jacobian_instance.symbolic_jacobian.clone());

        let fun = jacobian_instance.lambdified_functions_IVP_DVector;
        let jac_wrapped: Box<dyn FnMut(f64, &DVector<f64>) -> DMatrix<f64>> =
            Box::new(move |t: f64, y: &DVector<f64>| -> DMatrix<f64> { new_jac(t, y) });

        self.fun = fun;
        self.jac = Some(jac_wrapped);
        self.n = self.eq_system.len();

        info!("Parallel Radau Newton functions generated successfully");
    }
}

#[cfg(test)]
mod tests_parallel {

    use crate::symbolic::symbolic_engine::Expr;
    use crate::symbolic::symbolic_functions::Jacobian;
    use approx::assert_relative_eq;
    use log::info;
    use nalgebra::DVector;

    #[test]
    fn test_generate_nr_solver_for_radau_parallel_basic() {
        let mut jacobian = Jacobian::new();
        // Simple test system: K_0_0 - y_0 = 0
        let eq1 = Expr::parse_expression("K01 - y");
        let eq_system = vec![eq1];

        let values = vec!["K01".to_string()]; // Stage derivatives (unknowns)
        jacobian.vector_of_functions = eq_system;
        jacobian.set_variables(values.iter().map(|x| x.as_str()).collect());
        jacobian.calc_jacobian();
        info!("Parallel Jacobian: {:?}", jacobian.symbolic_jacobian);
    }

    #[test]
    fn test_generate_nr_solver_for_radau_parallel_multiple_stages() {
        let mut jacobian = Jacobian::new();

        // Two-stage system for one variable
        let eq1 = Expr::parse_expression("K00 - y0 - h*K10");
        let eq2 = Expr::parse_expression("K10 - y0 - h*K00");
        let eq_system = vec![eq1, eq2];

        let values = vec!["K00".to_string(), "K10".to_string()]; // Stage derivatives
        let parameters = vec!["y0".to_string(), "h".to_string()]; // Parameters
        let arg = "t".to_string();

        jacobian.generate_NR_solver_for_Radau_parallel(
            eq_system,
            values.clone(),
            parameters.clone(),
            arg.clone(),
        );

        // Test jacobian function
        let K_values = DVector::from_vec(vec![0.0, 0.0]);
        let h = 0.0;
        let y_0 = 0.0;
        let parameters_val = vec![h, y_0];
        let mut values_and_parameters = K_values.clone();
        values_and_parameters.extend(parameters_val);
        info!(
            "Parallel values and parameters = {:?} \n",
            values_and_parameters
        );

        let J = (jacobian.function_jacobian_IVP_DMatrix)(0.0, &values_and_parameters);
        info!("Parallel J = {:?}", J.clone());
        assert_eq!(J.shape(), (2, 2));
        assert_eq!(J.data.as_vec().to_owned(), vec![1.0, 0.0, 0.0, 1.0]);

        // Test the vector function
        let result = (jacobian.lambdified_functions_IVP_DVector)(0.0, &values_and_parameters);
        info!("Parallel result = {:?} \n", result);
    }

    #[test]
    fn test_generate_nr_solver_for_radau_parallel_multiple_variables() {
        let mut jacobian = Jacobian::new();

        // Two variables, one stage each
        let eq1 = Expr::parse_expression("K00 - y0");
        let eq2 = Expr::parse_expression("K01 - y1");
        let eq_system = vec![eq1, eq2];

        let values = vec!["K00".to_string(), "K01".to_string()]; // Stage derivatives
        let parameters = vec!["y0".to_string(), "y1".to_string()];
        let arg = "t".to_string();

        jacobian.generate_NR_solver_for_Radau_parallel(eq_system, values, parameters, arg);

        assert_eq!(jacobian.lambdified_functions_IVP.len(), 2);
    }

    #[test]
    fn test_jacobian_generate_ivp_radau_parallel() {
        let mut jacobian = Jacobian::new();

        // Set up a simple system
        let eq1 = Expr::parse_expression("K00 - y0");
        jacobian.set_vector_of_functions(vec![eq1]);
        jacobian.set_variables(vec!["K00"]);
        jacobian.calc_jacobian();

        let values = vec!["K00".to_string()];
        let parameters = vec!["y0".to_string(), "h".to_string()];

        jacobian.jacobian_generate_IVP_Radau_parallel("t", values, parameters);

        // Test that jacobian function was created
        let jac_fn = &jacobian.function_jacobian_IVP_DMatrix;
        let test_vars = DVector::from_vec(vec![1.0, 2.0, 0.1]); // K_0_0, y_0, h
        let jac_result = jac_fn(0.0, &test_vars);

        assert_eq!(jac_result.nrows(), 1);
        assert_eq!(jac_result.ncols(), 1);
        // For K_0_0 - y_0, derivative w.r.t. K_0_0 should be 1
        assert_relative_eq!(jac_result[(0, 0)], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_calc_jacobian_fun_with_parameters_parallel() {
        // Create a simple jacobian matrix symbolically
        let jac_symbolic = vec![
            vec![Expr::Const(1.0), Expr::Const(0.0)], // d/dK_0_0, d/dK_1_0 of first equation
            vec![Expr::Const(0.0), Expr::Const(1.0)], // d/dK_0_0, d/dK_1_0 of second equation
        ];

        let variable_str = vec!["K00".to_string(), "K10".to_string()];
        let parameters = vec!["y0".to_string(), "h".to_string()];
        let arg = "t".to_string();
        let bandwidth = (0, 0); // Dense matrix

        let jac_fn = Jacobian::calc_jacobian_fun_with_parameters_parallel(
            jac_symbolic,
            2, // vector_of_functions_len
            2, // vector_of_variables_len
            variable_str,
            parameters,
            arg,
            bandwidth,
        );

        // Test the jacobian function
        let test_vars = DVector::from_vec(vec![1.0, 2.0, 3.0, 0.1]); // K_0_0, K_1_0, y_0, h
        let jac_result = jac_fn(0.0, &test_vars);

        assert_eq!(jac_result.nrows(), 2);
        assert_eq!(jac_result.ncols(), 2);
        assert_relative_eq!(jac_result[(0, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(jac_result[(0, 1)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(jac_result[(1, 0)], 0.0, epsilon = 1e-10);
        assert_relative_eq!(jac_result[(1, 1)], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_lambdify_funcvector_with_parameters_parallel() {
        let mut jacobian = Jacobian::new();

        // Simple system: K_0_0 - y_0 = 0, K_1_0 - y_1 = 0
        let eq1 = Expr::parse_expression("K00 - y0");
        let eq2 = Expr::parse_expression("K10 - y1");
        jacobian.set_vector_of_functions(vec![eq1, eq2]);

        let variable_str = vec!["K00", "K10"];
        let parameters = vec!["y0", "y1", "h"];

        jacobian.lambdify_funcvector_with_parameters_parallel("t", variable_str, parameters);

        assert_eq!(jacobian.lambdified_functions_IVP.len(), 2);

        // Test the functions
        let test_values = vec![1.0, 2.0, 3.0, 4.0, 0.1]; // K_0_0, K_1_0, y_0, y_1, h
        let result1 = jacobian.lambdified_functions_IVP[0](0.0, test_values.clone());
        let result2 = jacobian.lambdified_functions_IVP[1](0.0, test_values.clone());

        // K_0_0 - y_0 = 1.0 - 3.0 = -2.0
        assert_relative_eq!(result1, -2.0, epsilon = 1e-10);
        // K_1_0 - y_1 = 2.0 - 4.0 = -2.0
        assert_relative_eq!(result2, -2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_vector_funvector_with_parameters_dvector_parallel() {
        let mut jacobian = Jacobian::new();

        // System: K_0_0 - y_0, K_1_0 - y_1
        let eq1 = Expr::parse_expression("K00 - y0");
        let eq2 = Expr::parse_expression("K10 - y1");
        jacobian.set_vector_of_functions(vec![eq1, eq2]);

        let variable_str = vec!["K00", "K10"];
        let parameters = vec!["y0", "y1", "h"];

        jacobian.vector_funvector_with_parameters_DVector_parallel("t", variable_str, parameters);

        // Test the vector function
        let test_vars = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 0.1]); // K_0_0, K_1_0, y_0, y_1, h
        let result = (jacobian.lambdified_functions_IVP_DVector)(0.0, &test_vars);

        assert_eq!(result.len(), 2);
        assert_relative_eq!(result[0], -2.0, epsilon = 1e-10); // K_0_0 - y_0 = 1.0 - 3.0
        assert_relative_eq!(result[1], -2.0, epsilon = 1e-10); // K_1_0 - y_1 = 2.0 - 4.0
    }

    #[test]
    fn test_vector_funvector_with_parameters_dvector_parallel_with_step_size() {
        let mut jacobian = Jacobian::new();

        // System: K_0_0 - h*y_0, K_1_0 - h*y_1
        let eq1 = Expr::parse_expression("K00 - h*y0");
        let eq2 = Expr::parse_expression("K10 - h*y1");
        jacobian.set_vector_of_functions(vec![eq1, eq2]);

        let variable_str = vec!["K00", "K10"];
        let parameters = vec!["y0", "y1", "h"];

        jacobian.vector_funvector_with_parameters_DVector_parallel("t", variable_str, parameters);

        // Test the vector function
        let test_vars = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 1.0]); // K_0_0, K_1_0, y_0, y_1, h
        let result = (jacobian.lambdified_functions_IVP_DVector)(0.0, &test_vars);

        assert_eq!(result.len(), 2);
        assert_relative_eq!(result[0], -2.0, epsilon = 1e-10); // K_0_0 - h*y_0 = 1.0 - 1.0*3.0
        assert_relative_eq!(result[1], -2.0, epsilon = 1e-10); // K_1_0 - h*y_1 = 2.0 - 1.0*4.0
    }

    #[test]
    fn test_radau_system_with_time_dependency_parallel() {
        let mut jacobian = Jacobian::new();

        // System with time dependency: K_0_0 - t*y_0
        let eq1 = Expr::parse_expression("K00 - t*y0");
        let eq_system = vec![eq1];

        let values = vec!["K00".to_string()];
        let parameters = vec!["y0".to_string(), "h".to_string()];
        let arg = "t".to_string();

        jacobian.generate_NR_solver_for_Radau_parallel(eq_system, values, parameters, arg);

        // Test at different times
        let test_vars1 = DVector::from_vec(vec![2.0, 1.0, 0.1]); // K_0_0, y_0, h
        let result1 = (jacobian.lambdified_functions_IVP_DVector)(1.0, &test_vars1);
        // K_0_0 - t*y_0 = 2.0 - 1.0*1.0 = 1.0
        assert_relative_eq!(result1[0], 1.0, epsilon = 1e-10);

        let result2 = (jacobian.lambdified_functions_IVP_DVector)(2.0, &test_vars1);
        // K_0_0 - t*y_0 = 2.0 - 2.0*1.0 = 0.0
        assert_relative_eq!(result2[0], 0.0, epsilon = 1e-10);
    }
}
