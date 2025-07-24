use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions::Jacobian;
use log::info;
use nalgebra::{DMatrix, DVector, Matrix};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
impl Jacobian {
    pub fn generate_NR_solver_for_Radau(
        &mut self,
        eq_system: Vec<Expr>,
        values: Vec<String>,
        parameters: Vec<String>,
        arg: String,
    ) {
        self.set_vector_of_functions(eq_system);
        self.set_variables(values.iter().map(|x| x.as_str()).collect());
        self.calc_jacobian();
        let ncols = self.symbolic_jacobian.len();
        let nrows = self.symbolic_jacobian[0].len();
        assert!(nrows == ncols);
        self.jacobian_generate_IVP_Radau(arg.as_str(), values.clone(), parameters.clone());
        let values_str = values.iter().map(|x| x.as_str()).collect::<Vec<&str>>();
        let parameters_str = parameters.iter().map(|x| x.as_str()).collect::<Vec<&str>>();
        self.lambdify_funcvector_with_parameters(
            arg.as_str(),
            values_str.clone(),
            parameters_str.clone(),
        );
        self.vector_funvector_with_parameters_DVector(
            arg.as_str(),
            values_str.clone(),
            parameters_str.clone(),
        );
    }
    pub fn jacobian_generate_IVP_Radau(
        &mut self,
        arg: &str,
        variable_str: Vec<String>,
        parameters: Vec<String>,
    ) {
        let symbolic_jacobian = self.symbolic_jacobian.clone();
        let symbolic_jacobian_rc = symbolic_jacobian.clone();

        let vector_of_functions_len = self.vector_of_functions.len();
        let vector_of_variables_len = self.vector_of_variables.len();
        let bandwidth = self.bandwidth;
        let new_jac = Jacobian::calc_jacobian_fun_with_parameters(
            symbolic_jacobian_rc,
            vector_of_functions_len,
            vector_of_variables_len,
            variable_str.iter().map(|s| s.to_string()).collect(),
            parameters.iter().map(|s| s.to_string()).collect(),
            arg.to_string(),
            bandwidth,
        );

        self.function_jacobian_IVP_DMatrix = new_jac;
    }
    /// creating function jacobian a matrix of functions with partial derivatives
    /// generates Jac in the form of  Box<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64> >
    pub fn calc_jacobian_fun_with_parameters(
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
            let mut new_function_jacobian: DMatrix<f64> =
                DMatrix::zeros(vector_of_functions_len, vector_of_variables_len);
            let variales_and_parameters: Vec<&str> = variables_and_parameters
                .iter()
                .map(|s| s.as_str())
                .collect();
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
                    let partial_func = Expr::lambdify_IVP_owned(
                        jac[i][j].clone(),
                        arg.as_str(),
                        variales_and_parameters.clone(),
                    );
                    let v_vec: Vec<f64> = v.iter().cloned().collect();
                    new_function_jacobian[(i, j)] = if jac[i][j].clone() == Expr::Const(0.0) {
                        0.0
                    } else {
                        partial_func(x, v_vec.clone())
                    };
                }
            }
            new_function_jacobian
        })
    } // end of function

    pub fn lambdify_funcvector_with_parameters(
        &mut self,
        arg: &str,
        variable_str: Vec<&str>,
        parameters: Vec<&str>,
    ) {
        let mut variable_and_parameters: Vec<&str> = variable_str.clone();
        variable_and_parameters.extend(parameters.clone());
        println!("variable_and_parameters = {:?} \n", variable_and_parameters);
        let mut result: Vec<Box<dyn Fn(f64, Vec<f64>) -> f64>> = Vec::new();
        for func in self.vector_of_functions.clone() {
            let func = Expr::lambdify_IVP_owned(func, arg, variable_and_parameters.clone());
            result.push(func);
        }
        self.lambdified_functions_IVP = result;
        //result
    }
    pub fn vector_funvector_with_parameters_DVector(
        &mut self,
        arg: &str,
        variable_str: Vec<&str>,
        parameters: Vec<&str>,
    ) {
        let vector_of_functions = &self.vector_of_functions;
        fn f(
            vector_of_functions: Vec<Expr>,
            arg: String,
            variable_and_parameters: Vec<String>,
        ) -> Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64> + 'static> {
            Box::new(move |x: f64, v: &DVector<f64>| -> DVector<f64> {
                let mut result: DVector<f64> = DVector::zeros(vector_of_functions.len());
                for (i, func) in vector_of_functions.iter().enumerate() {
                    let func = Expr::lambdify_IVP_owned(
                        func.to_owned(),
                        arg.as_str(),
                        variable_and_parameters
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
        let mut variable_and_parameters = variable_str.clone();
        variable_and_parameters.extend(parameters.clone());

        self.lambdified_functions_IVP_DVector = f(
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

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Newton-Raphson solver specifically designed for Radau method
/// Solves for stage derivatives K_{i,j} while treating y_n, h, t as parameters
/// Newton-Raphson solver specifically designed for Radau method
/// Solves for stage derivatives K_{i,j} while treating y_n, h, t as parameters
pub struct RadauNewton {
    pub eq_system: Vec<Expr>,     // Radau equations with K variables as unknowns
    pub k_values: DVector<f64>,   // Initial guess for K variables
    pub k_variables: Vec<String>, // K variable names ["K_0_0", "K_0_1", ...]
    pub parameters: Vec<String>,  // Parameter names ["y_0", "y_1", "h"]
    pub arg: String,              // Independent variable (usually "t")
    pub tolerance: f64,
    pub max_iterations: usize,
    pub max_error: f64,
    pub result: Option<DVector<f64>>, // Solution for K variables
    pub jacobian_symbolic: Option<Vec<Vec<Expr>>>, // Jacobian of eq_system with respect to k_variables and parameters (set before each Newton solve>,
    // Radau-specific parameters (set before each Newton solve)
    pub y_n: DVector<f64>, // Current solution y_n
    pub h: f64,            // Current step size
    pub t_n: f64,          // Current time
    pub n_vars: usize,     // Number of ODE variables
    pub n_stages: usize,   // Number of Radau stages
    pub parallel: bool,
    // Function closures (generated from symbolic expressions)
    pub fun: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
    pub jac: Option<Box<dyn FnMut(f64, &DVector<f64>) -> DMatrix<f64>>>,
    pub n: usize,
}

impl RadauNewton {
    pub fn new(
        eq_system: Vec<Expr>,
        k_variables: Vec<String>,
        initial_guess: Option<DVector<f64>>,
        parameters: Vec<String>,
        arg: String,
        tolerance: f64,
        max_iterations: usize,
        n_vars: usize,
        n_stages: usize,
    ) -> RadauNewton {
        let initial_guess = initial_guess.unwrap_or_else(|| DVector::zeros(k_variables.len()));

        RadauNewton {
            eq_system,
            k_values: initial_guess.clone(),
            k_variables: k_variables.clone(),
            parameters,
            arg,
            tolerance,
            max_iterations,
            max_error: 1e-3,
            result: None,
            jacobian_symbolic: None,
            y_n: DVector::zeros(n_vars),
            h: 0.0,
            t_n: 0.0,
            n_vars,
            n_stages,
            parallel: false,
            fun: Box::new(|_t, y| y.clone()),
            jac: None,
            n: k_variables.len(),
        }
    }

    /// Generate function and Jacobian closures using the infrastructure
    pub fn eq_generate(&mut self) {
        info!("Generating Radau Newton equations and jacobian");
        let mut jacobian_instance = Jacobian::new();
        match self.parallel {
            true => {
                println!("\n Parallel version enabled \n");
                jacobian_instance.generate_NR_solver_for_Radau_parallel(
                    self.eq_system.clone(),
                    self.k_variables.clone(),
                    self.parameters.clone(),
                    self.arg.clone(),
                );
            }
            false => {
                jacobian_instance.generate_NR_solver_for_Radau(
                    self.eq_system.clone(),
                    self.k_variables.clone(),
                    self.parameters.clone(),
                    self.arg.clone(),
                );
            }
        }

        self.jacobian_symbolic = Some(jacobian_instance.symbolic_jacobian.clone());
        // Extract the lambdified functions and Jacobian
        let fun: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>> =
            jacobian_instance.lambdified_functions_IVP_DVector;

        let jac = jacobian_instance.function_jacobian_IVP_DMatrix;

        let jac_wrapped: Box<dyn FnMut(f64, &DVector<f64>) -> DMatrix<f64>> =
            Box::new(move |t: f64, y: &DVector<f64>| -> DMatrix<f64> { jac(t, &y) });

        self.fun = fun;
        self.jac = Some(jac_wrapped);
        self.n = self.eq_system.len();

        info!("Radau Newton functions generated successfully");
    }

    /// Set parameters for the current Newton solve (y_n, h, t_n)
    pub fn set_parameters(&mut self, y_n: DVector<f64>, h: f64, t_n: f64) {
        self.y_n = y_n;
        self.h = h;
        self.t_n = t_n;

        info!(
            "Setting Radau Newton parameters: t_n = {:.6}, h = {:.6}",
            t_n, h
        );
    }

    pub fn set_parallel(&mut self, parallel: bool) {
        self.parallel = parallel;
    }

    /// Set initial guess for K variables
    pub fn set_initial_guess(&mut self, guess: DVector<f64>) {
        self.k_values = guess;
    }

    /// Perform one Newton-Raphson iteration
    pub fn iteration(&mut self) -> DVector<f64> {
        //  assert!(!self.initial_guess.is_empty(), "initial guess is empty");

        let t = self.t_n;
        let y_n: DVector<f64> = self.y_n.clone();
        // Create parameter vector: [K_variables, y_n_variables, h]
        let k_vector: DVector<f64> = self.k_values.clone();
        info!("Radau Newton iteration: k_vector = {:?}", k_vector);
        let mut param_vector: DVector<f64> = k_vector.clone();
        param_vector.extend(y_n.iter().cloned());
        let param_vector: DVector<f64> = param_vector.push(self.h);

        let f = (self.fun)(t, &param_vector);
        let new_j = &self.jac.as_mut().unwrap()(t, &param_vector);

        // For Radau, we solve: J * delta_k = -f
        // where J is the Jacobian w.r.t. K variables only
        let lu = new_j.clone().lu();
        let neg_f = -1.0 * f;
        let delta_k = lu.solve(&neg_f).expect("Jacobian should be invertible");
        info!("Radau Newton iteration: step = {:?}", delta_k);
        let new_k = k_vector + delta_k;
        new_k
    }

    /// Solve the Newton system for K variables
    pub fn solve(&mut self) -> Option<DVector<f64>> {
        info!(
            "Starting Radau Newton solve with {} unknowns",
            self.k_values.len()
        );

        let mut k = self.k_values.clone();
        let mut iteration_count = 0;

        while iteration_count < self.max_iterations {
            info!("Radau Newton iteration {}", iteration_count);
            // Update initial_guess for iteration method

            self.k_values = k.clone();

            let k_new = self.iteration();
            info!(
                "Radau Newton iteration {}: k_new = {:?}",
                iteration_count, k_new
            );
            let delta_k = &k_new - &k;
            let error = Matrix::norm(&delta_k);

            info!(
                "Radau Newton iteration {}: error = {:.2e}",
                iteration_count, error
            );
            info!("end of iteration {} \n", iteration_count);
            // self.k_values = k_new.clone();
            if error < self.tolerance {
                info!("Radau Newton converged in {} iterations", iteration_count);
                self.result = Some(k_new.clone());
                self.max_error = error;
                return Some(k_new);
            }

            k = k_new;
            iteration_count += 1;
        }

        info!(
            "Radau Newton failed to converge after {} iterations",
            self.max_iterations
        );
        None
    }

    pub fn get_result(&self) -> Option<DVector<f64>> {
        self.result.clone()
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
///                       TESTS
/// ///////////////////////////////////////////////////////////////////////////////////////////////////////////
#[cfg(test)]
mod tests2 {
    use super::*;

    use approx::assert_relative_eq;
    use nalgebra::DVector;

    #[test]
    fn test_generate_nr_solver_for_radau_basic() {
        let mut jacobian = Jacobian::new();
        // Simple test system: K_0_0 - y_0 = 0
        let eq1 = Expr::parse_expression("K01 - y");
        let eq_system = vec![eq1];

        let values = vec!["K01".to_string()]; // Stage derivatives (unknowns)tem;
        jacobian.vector_of_functions = eq_system;
        jacobian.set_variables(values.iter().map(|x| x.as_str()).collect());
        jacobian.calc_jacobian();
        info!("Jacobian: {:?}", jacobian.symbolic_jacobian);
    }

    #[test]
    fn test_generate_nr_solver_for_radau_multiple_stages() {
        let mut jacobian = Jacobian::new();

        // Two-stage system for one variable
        let eq1 = Expr::parse_expression("K00 - y0 - h*K10");
        let eq2 = Expr::parse_expression("K10 - y0 - h*K00");
        let eq_system = vec![eq1, eq2];

        let values = vec!["K00".to_string(), "K10".to_string()]; // Stage derivatives
        let parameters = vec!["y0".to_string(), "h".to_string()]; // Parameters
        let arg = "t".to_string();

        jacobian.vector_of_functions = eq_system;
        jacobian.set_variables(values.iter().map(|x| x.as_str()).collect());

        jacobian.calc_jacobian();
        info!("sym jac");
        for (i, row) in jacobian.symbolic_jacobian.iter().enumerate() {
            for (j, col) in row.iter().enumerate() {
                print!("\n({}, {}), \n {} ", i, j, col);
            }
        }
        jacobian.jacobian_generate_IVP_Radau(arg.as_str(), values.clone(), parameters.clone());
        let J_fun = jacobian.function_jacobian_IVP_DMatrix.as_ref();
        let K_values = DVector::from_vec(vec![0.0, 0.0]);
        let h = 0.0;
        let y_0 = 0.0;
        let parameters_val = vec![h, y_0];
        let mut values_and_parameters = K_values.clone();
        values_and_parameters.extend(parameters_val);
        info!("values and parameters = {:?} \n", values_and_parameters);

        let J = J_fun(0.0, &values_and_parameters);
        info!("J = {:?}", J.clone());
        assert_eq!(J.shape(), (2, 2));
        assert_eq!(J.data.as_vec().to_owned(), vec![1.0, 0.0, 0.0, 1.0]);

        let values_str = values.iter().map(|x| x.as_str()).collect::<Vec<&str>>();
        let parameters_str = parameters.iter().map(|x| x.as_str()).collect::<Vec<&str>>();
        // info!("function vector {:?}", jacobian.vector_of_functions );
        //  jacobian.lambdify_funcvector_with_parameters( arg.as_str(), values_str.clone(), parameters_str.clone());
        //  let r = jacobian.lambdified_functions_IVP[0](0.0, values_and_parameters.data.clone().as_vec().to_owned());
        //   info!("r = {:?}", r);
        jacobian.vector_funvector_with_parameters_DVector(
            arg.as_str(),
            values_str.clone(),
            parameters_str.clone(),
        );
        // Test the vector function

        let result = (jacobian.lambdified_functions_IVP_DVector)(0.0, &values_and_parameters);
        info!("result = {:?} \n", result);
    }

    #[test]
    fn test_generate_nr_solver_for_radau_multiple_variables() {
        let mut jacobian = Jacobian::new();

        // Two variables, one stage each
        let eq1 = Expr::parse_expression("K00 - y0");
        let eq2 = Expr::parse_expression("K01 - y1");
        let eq_system = vec![eq1, eq2];

        let values = vec!["K00".to_string(), "K01".to_string()]; // Stage derivatives
        let parameters = vec!["y0".to_string(), "y1".to_string()];
        let arg = "t".to_string();

        jacobian.generate_NR_solver_for_Radau(eq_system, values, parameters, arg);

        assert_eq!(jacobian.lambdified_functions_IVP.len(), 2);
    }

    #[test]
    fn test_jacobian_generate_ivp_radau() {
        let mut jacobian = Jacobian::new();

        // Set up a simple system
        let eq1 = Expr::parse_expression("K00 - y0");
        jacobian.set_vector_of_functions(vec![eq1]);
        jacobian.set_variables(vec!["K00"]);
        jacobian.calc_jacobian();

        let values = vec!["K00".to_string()];
        let parameters = vec!["y0".to_string(), "h".to_string()];

        jacobian.jacobian_generate_IVP_Radau("t", values, parameters);

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
    fn test_calc_jacobian_fun_with_parameters() {
        // Create a simple jacobian matrix symbolically
        let jac_symbolic = vec![
            vec![Expr::Const(1.0), Expr::Const(0.0)], // d/dK_0_0, d/dK_1_0 of first equation
            vec![Expr::Const(0.0), Expr::Const(1.0)], // d/dK_0_0, d/dK_1_0 of second equation
        ];

        let variable_str = vec!["K00".to_string(), "K10".to_string()];
        let parameters = vec!["y0".to_string(), "h".to_string()];
        let arg = "t".to_string();
        let bandwidth = (0, 0); // Dense matrix

        let jac_fn = Jacobian::calc_jacobian_fun_with_parameters(
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
    fn test_lambdify_funcvector_with_parameters() {
        let mut jacobian = Jacobian::new();

        // Simple system: K_0_0 - y_0 = 0, K_1_0 - y_1 = 0
        let eq1 = Expr::parse_expression("K00 - y0");
        let eq2 = Expr::parse_expression("K10 - y1");
        jacobian.set_vector_of_functions(vec![eq1, eq2]);

        let variable_str = vec!["K00", "K10"];
        let parameters = vec!["y0", "y1", "h"];

        jacobian.lambdify_funcvector_with_parameters("t", variable_str, parameters);

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
    fn test_vector_funvector_with_parameters_dvector() {
        let mut jacobian = Jacobian::new();

        // System: K_0_0 - y_0, K_1_0 - y_1
        let eq1 = Expr::parse_expression("K00 - y0");
        let eq2 = Expr::parse_expression("K10 - y1");
        jacobian.set_vector_of_functions(vec![eq1, eq2]);

        let variable_str = vec!["K00", "K10"];
        let parameters = vec!["y0", "y1", "h"];

        jacobian.vector_funvector_with_parameters_DVector("t", variable_str, parameters);

        // Test the vector function
        let test_vars = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 0.1]); // K_0_0, K_1_0, y_0, y_1, h
        let result = (jacobian.lambdified_functions_IVP_DVector)(0.0, &test_vars);

        assert_eq!(result.len(), 2);
        assert_relative_eq!(result[0], -2.0, epsilon = 1e-10); // K_0_0 - y_0 = 1.0 - 3.0
        assert_relative_eq!(result[1], -2.0, epsilon = 1e-10); // K_1_0 - y_1 = 2.0 - 4.0
    }
    #[test]
    fn test_vector_funvector_with_parameters_dvector2() {
        let mut jacobian = Jacobian::new();

        // System: K_0_0 - y_0, K_1_0 - y_1
        let eq1 = Expr::parse_expression("K00 - h*y0");
        let eq2 = Expr::parse_expression("K10 - h*y1");
        jacobian.set_vector_of_functions(vec![eq1, eq2]);

        let variable_str = vec!["K00", "K10"];
        let parameters = vec!["y0", "y1", "h"];

        jacobian.vector_funvector_with_parameters_DVector("t", variable_str, parameters);

        // Test the vector function
        let test_vars = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 1.0]); // K_0_0, K_1_0, y_0, y_1, h
        let result = (jacobian.lambdified_functions_IVP_DVector)(0.0, &test_vars);

        assert_eq!(result.len(), 2);
        assert_relative_eq!(result[0], -2.0, epsilon = 1e-10); // K_0_0 - y_0 = 1.0 - 3.0
        assert_relative_eq!(result[1], -2.0, epsilon = 1e-10); // K_1_0 - y_1 = 2.0 - 4.0
    }
    #[test]
    fn test_radau_system_with_time_dependency() {
        let mut jacobian = Jacobian::new();

        // System with time dependency: K_0_0 - t*y_0
        let eq1 = Expr::parse_expression("K00 - t*y0");
        let eq_system = vec![eq1];

        let values = vec!["K00".to_string()];
        let parameters = vec!["y0".to_string(), "h".to_string()];
        let arg = "t".to_string();

        jacobian.generate_NR_solver_for_Radau(eq_system, values, parameters, arg);

        // Test at different times
        let test_vars1 = DVector::from_vec(vec![2.0, 1.0, 0.1]); // K_0_0, y_0, h
        let result1 = (jacobian.lambdified_functions_IVP_DVector)(1.0, &test_vars1);
        // K_0_0 - t*y_0 = 2.0 - 1.0*1.0 = 1.0
        assert_relative_eq!(result1[0], 1.0, epsilon = 1e-10);

        let result2 = (jacobian.lambdified_functions_IVP_DVector)(2.0, &test_vars1);
        // K_0_0 - t*y_0 = 2.0 - 2.0*1.0 = 0.0
        assert_relative_eq!(result2[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_radau_system_with_step_size_dependency() {
        let mut jacobian = Jacobian::new();

        // System with step size dependency: K_0_0 - h*y_0
        let eq1 = Expr::parse_expression("K00 - h*y0");
        let eq_system = vec![eq1];

        let values = vec!["K00".to_string()];
        let parameters = vec!["y0".to_string(), "h".to_string()];
        let arg = "t".to_string();

        jacobian.generate_NR_solver_for_Radau(eq_system, values, parameters, arg);

        // Test with different step sizes
        let test_vars1 = DVector::from_vec(vec![1.0, 2.0, 0.1]); // K_0_0, y_0, h
        let result1 = (jacobian.lambdified_functions_IVP_DVector)(0.0, &test_vars1);
        // K_0_0 - h*y_0 = 1.0 - 0.1*2.0 = 0.8
        assert_relative_eq!(result1[0], 0.8, epsilon = 1e-10);

        let test_vars2 = DVector::from_vec(vec![1.0, 2.0, 0.5]); // K_0_0, y_0, h
        let result2 = (jacobian.lambdified_functions_IVP_DVector)(0.0, &test_vars2);
        // K_0_0 - h*y_0 = 1.0 - 0.5*2.0 = 0.0
        assert_relative_eq!(result2[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_radau_jacobian_with_coupling() {
        let mut jacobian = Jacobian::new();

        // Coupled system: K_0_0 - K_1_0, K_1_0 - K_0_0
        let eq1 = Expr::parse_expression("K00 - K10");
        let eq2 = Expr::parse_expression("K10 - K00");
        jacobian.set_vector_of_functions(vec![eq1, eq2]);
        jacobian.set_variables(vec!["K00", "K10"]);
        jacobian.calc_jacobian();

        let values = vec!["K00".to_string(), "K10".to_string()];
        let parameters = vec!["y0".to_string(), "h".to_string()];

        jacobian.jacobian_generate_IVP_Radau("t", values.clone(), parameters.clone());
        info!("residual: {:?}", jacobian.vector_of_functions);
        // Test jacobian
        let test_vars = DVector::from_vec(vec![1.0, 2.0, 3.0, 0.1]); // K_0_0, K_1_
        jacobian.vector_funvector_with_parameters_DVector(
            "t",
            values.iter().map(|x| x.as_str()).collect(),
            parameters.iter().map(|x| x.as_str()).collect(),
        );
        let result = (jacobian.lambdified_functions_IVP_DVector)(0.0, &test_vars);

        assert_eq!(result.len(), 2);
        info!("result: {:?}", result);
        assert_relative_eq!(result[0], -1.0, epsilon = 1e-10); // K_0_0 - K_1_0 = 1.0 - 3.0
        assert_relative_eq!(result[1], 1.0, epsilon = 1e-10); // K_1_0 - K_0_0 = 2.0 - 1.0
    }
}
//////////////////////////////////////////////////////////////////////////////////////////////////
// Radau solver tests
//////////////////////////////////////////////////////////////////////////////////////////////////
#[cfg(test)]
mod tests_radau_newton {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::DVector;

    #[test]
    fn test_radau_newton_new() {
        let eq1 = Expr::parse_expression("K00 - y0");
        let eq_system = vec![eq1];
        let k_variables = vec!["K00".to_string()];
        let parameters = vec!["y0".to_string(), "h".to_string()];
        let arg = "t".to_string();

        let radau_newton = RadauNewton::new(
            eq_system,
            k_variables,
            None,
            parameters,
            arg,
            1e-6,
            50,
            1,
            1,
        );

        assert_eq!(radau_newton.k_variables.len(), 1);
        assert_eq!(radau_newton.n_vars, 1);
        assert_eq!(radau_newton.n_stages, 1);
        assert_eq!(radau_newton.parameters.len(), 2);
    }

    #[test]
    fn test_radau_newton_simple_solve() {
        // Simple system: K_0_0 - y_0 = 0
        // Solution should be K_0_0 = y_0
        let eq1 = Expr::parse_expression("K00 - y0");
        let eq_system = vec![eq1];
        let k_variables = vec!["K00".to_string()];
        let parameters = vec!["y0".to_string(), "h".to_string()];
        let arg = "t".to_string();

        let mut radau_newton = RadauNewton::new(
            eq_system,
            k_variables,
            None,
            parameters,
            arg,
            1e-10,
            50,
            1,
            1,
        );

        // Generate functions
        radau_newton.eq_generate();

        // Set parameters: y_0 = 2.0, h = 0.1, t = 0.0
        let y_n = DVector::from_vec(vec![2.0]);
        radau_newton.set_parameters(y_n, 0.1, 0.0);

        // Set initial guess
        radau_newton.set_initial_guess(DVector::from_vec(vec![0.0]));

        // Solve
        let result = radau_newton.solve();

        assert!(result.is_some());
        let k_solution = result.unwrap();
        assert_eq!(k_solution.len(), 1);
        // K_0_0 should equal y_0 = 2.0
        assert_relative_eq!(k_solution[0], 2.0, epsilon = 1e-8);
    }

    #[test]
    fn test_radau_newton_two_variables() {
        // System: K_0_0 - y_0 = 0, K_0_1 - y_1 = 0
        // Solution should be K_0_0 = y_0, K_0_1 = y_1
        let eq1 = Expr::parse_expression("K00 - y0");
        let eq2 = Expr::parse_expression("K01 - y1");
        let eq_system = vec![eq1, eq2];
        let k_variables = vec!["K00".to_string(), "K01".to_string()];
        let parameters = vec!["y0".to_string(), "y1".to_string(), "h".to_string()];
        let arg = "t".to_string();

        let mut radau_newton = RadauNewton::new(
            eq_system,
            k_variables,
            None,
            parameters,
            arg,
            1e-10,
            50,
            2,
            1,
        );

        radau_newton.eq_generate();

        // Set parameters: y_0 = 1.5, y_1 = 3.0, h = 0.1
        let y_n = DVector::from_vec(vec![1.5, 3.0]);
        radau_newton.set_parameters(y_n, 0.1, 0.0);

        radau_newton.set_initial_guess(DVector::from_vec(vec![0.0, 0.0]));

        let result = radau_newton.solve();

        assert!(result.is_some());
        let k_solution = result.unwrap();
        assert_eq!(k_solution.len(), 2);
        assert_relative_eq!(k_solution[0], 1.5, epsilon = 1e-8);
        assert_relative_eq!(k_solution[1], 3.0, epsilon = 1e-8);
    }

    #[test]
    fn test_radau_newton_with_step_size() {
        // System: K_0_0 - h*y_0 = 0
        // Solution should be K_0_0 = h*y_0
        let eq1 = Expr::parse_expression("K00 - h*y0");
        let eq_system = vec![eq1];
        let k_variables = vec!["K00".to_string()];
        let parameters = vec!["y0".to_string(), "h".to_string()];
        let arg = "t".to_string();

        let mut radau_newton = RadauNewton::new(
            eq_system,
            k_variables,
            None,
            parameters,
            arg,
            1e-10,
            50,
            1,
            1,
        );

        radau_newton.eq_generate();

        // Set parameters: y_0 = 2.0, h = 0.5
        let y_n = DVector::from_vec(vec![2.0]);
        radau_newton.set_parameters(y_n, 0.5, 0.0);

        radau_newton.set_initial_guess(DVector::from_vec(vec![0.0]));

        let result = radau_newton.solve();

        assert!(result.is_some());
        let k_solution = result.unwrap();
        // K_0_0 should equal h*y_0 = 0.5*2.0 = 1.0
        assert_relative_eq!(k_solution[0], 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_radau_newton_with_time_dependency() {
        // System: K_0_0 - t*y_0 = 0
        // Solution should be K_0_0 = t*y_0
        let eq1 = Expr::parse_expression("K00 - t*y0");
        let eq_system = vec![eq1];
        let k_variables = vec!["K00".to_string()];
        let parameters = vec!["y0".to_string(), "h".to_string()];
        let arg = "t".to_string();

        let mut radau_newton = RadauNewton::new(
            eq_system,
            k_variables,
            None,
            parameters,
            arg,
            1e-10,
            50,
            1,
            1,
        );

        radau_newton.eq_generate();

        // Set parameters: y_0 = 3.0, h = 0.1, t = 2.0
        let y_n = DVector::from_vec(vec![3.0]);
        radau_newton.set_parameters(y_n, 0.1, 2.0);

        radau_newton.set_initial_guess(DVector::from_vec(vec![0.0]));

        let result = radau_newton.solve();

        assert!(result.is_some());
        let k_solution = result.unwrap();
        // K_0_0 should equal t*y_0 = 2.0*3.0 = 6.0
        assert_relative_eq!(k_solution[0], 6.0, epsilon = 1e-8);
    }

    #[test]
    fn test_radau_newton_coupled_system() {
        // Coupled system: K_0_0 - K_1_0 - y_0 = 0, K_1_0 - y_1 = 0
        // Solution: K_1_0 = y_1, K_0_0 = K_1_0 + y_0 = y_1 + y_0
        let eq1 = Expr::parse_expression("K00 - (K10 + y0)");
        let eq2 = Expr::parse_expression("K10 - y1");
        let eq_system = vec![eq1, eq2];
        let k_variables = vec!["K00".to_string(), "K10".to_string()];
        let parameters = vec!["y0".to_string(), "y1".to_string(), "h".to_string()];
        let arg = "t".to_string();

        let mut radau_newton = RadauNewton::new(
            eq_system,
            k_variables,
            Some(DVector::from_vec(vec![20.0, 20.0])),
            parameters,
            arg,
            1e-10,
            50,
            2,
            2,
        );

        radau_newton.eq_generate();
        let sym_jac: &Vec<Vec<Expr>> = &radau_newton.jacobian_symbolic.clone().unwrap().clone();
        info!("sym_jac: {:?}", sym_jac);
        // Set parameters: y_0 = 1.0, y_1 = 2.0, h = 0.1
        let y_n = DVector::from_vec(vec![1.0, 50.0]);
        radau_newton.set_parameters(y_n.clone(), 0.1, 0.0);

        // radau_newton.set_initial_guess(DVector::from_vec(vec![1.0, 1.0]));

        let result = radau_newton.solve();

        assert!(result.is_some());
        let k_solution = result.unwrap();
        info!("solution is: {:?}", k_solution);
        assert_eq!(k_solution.len(), 2);
        let res = k_solution[0] - k_solution[1] - y_n[0];
        info!("residual: {:?}", res);
        let res2 = k_solution[1] - y_n[1];
        info!("residual: {:?}", res2);
        assert_relative_eq!(k_solution[1], 50.0, epsilon = 1e-8); // K_1_0 = y_1
        assert_relative_eq!(k_solution[0], 51.0, epsilon = 1e-8); // K00 = y_1 + y_0
    }

    #[test]
    fn test_radau_newton_nonlinear_system() {
        // Nonlinear system: K_0_0^2 - y_0 = 0
        // Solution: K_0_0 = sqrt(y_0)
        let eq1 = Expr::parse_expression("K00*K00 - y0");
        let eq_system = vec![eq1];
        let k_variables = vec!["K00".to_string()];
        let parameters = vec!["y0".to_string(), "h".to_string()];
        let arg = "t".to_string();

        let mut radau_newton = RadauNewton::new(
            eq_system,
            k_variables,
            None,
            parameters,
            arg,
            1e-10,
            50,
            1,
            1,
        );

        radau_newton.eq_generate();

        // Set parameters: y_0 = 4.0, h = 0.1
        let y_n = DVector::from_vec(vec![4.0]);
        radau_newton.set_parameters(y_n, 0.1, 0.0);

        // Initial guess close to solution
        radau_newton.set_initial_guess(DVector::from_vec(vec![1.5]));

        let result = radau_newton.solve();

        assert!(result.is_some());
        let k_solution = result.unwrap();
        // K_0_0 should equal sqrt(4.0) = 2.0
        assert_relative_eq!(k_solution[0], 2.0, epsilon = 1e-8);
    }

    #[test]
    fn test_radau_newton_parameter_changes() {
        // Test that solver works correctly when parameters change
        let eq1 = Expr::parse_expression("K00 - y0");
        let eq_system = vec![eq1];
        let k_variables = vec!["K00".to_string()];
        let parameters = vec!["y0".to_string(), "h".to_string()];
        let arg = "t".to_string();

        let mut radau_newton = RadauNewton::new(
            eq_system,
            k_variables,
            None,
            parameters,
            arg,
            1e-10,
            50,
            1,
            1,
        );

        radau_newton.eq_generate();

        // First solve with y_0 = 1.0
        let y_n1 = DVector::from_vec(vec![1.0]);
        radau_newton.set_parameters(y_n1, 0.1, 0.0);
        radau_newton.set_initial_guess(DVector::from_vec(vec![0.0]));

        let result1 = radau_newton.solve();
        assert!(result1.is_some());
        assert_relative_eq!(result1.unwrap()[0], 1.0, epsilon = 1e-8);

        // Second solve with y_0 = 5.0
        let y_n2 = DVector::from_vec(vec![5.0]);
        radau_newton.set_parameters(y_n2, 0.1, 0.0);
        radau_newton.set_initial_guess(DVector::from_vec(vec![0.0]));

        let result2 = radau_newton.solve();
        assert!(result2.is_some());
        assert_relative_eq!(result2.unwrap()[0], 5.0, epsilon = 1e-8);
    }

    #[test]
    fn test_radau_newton_convergence_failure() {
        // System that should fail to converge with bad initial guess
        let eq1 = Expr::parse_expression("K00*K00*K00 - y0");
        let eq_system = vec![eq1];
        let k_variables = vec!["K00".to_string()];
        let parameters = vec!["y0".to_string(), "h".to_string()];
        let arg = "t".to_string();

        let mut radau_newton = RadauNewton::new(
            eq_system,
            k_variables,
            None,
            parameters,
            arg,
            1e-10,
            5, // Very low max iterations
            1,
            1,
        );

        radau_newton.eq_generate();

        let y_n = DVector::from_vec(vec![8.0]);
        radau_newton.set_parameters(y_n, 0.1, 0.0);

        // Bad initial guess far from solution
        radau_newton.set_initial_guess(DVector::from_vec(vec![100.0]));

        let result = radau_newton.solve();
        assert!(result.is_none());
    }
}
