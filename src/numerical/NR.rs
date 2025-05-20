use crate::numerical::BVP_Damp::BVP_utils::checkmem;
use crate::numerical::BVP_Damp::BVP_utils::{CustomTimer, elapsed_time};
///  Example#1
/// ```
///  use RustedSciThe::numerical::NR::NR;
/// use nalgebra::DVector;
/// //use the shortest way to solve system of equations
///    // first define system of equations and initial guess
///    let mut NR_instanse = NR::new();
/// 
///    let vec_of_expressions = vec![ "x^2+y^2-10".to_string(), "x-y-4".to_string()];
///   let initial_guess = vec![1.0, 1.0];
///    // solve
///    NR_instanse.eq_generate_from_str(vec_of_expressions, None, initial_guess, 1e-6, 100);
///    NR_instanse.main_loop();
///    println!("result = {:?} \n", NR_instanse.get_result().unwrap());
///  ```
/// Example#2
///  ```
///    // or more verbose way...
///    // first define system of equations
///     use RustedSciThe::numerical::NR::NR;
///     use RustedSciThe::symbolic::symbolic_engine::Expr;
///     use RustedSciThe::symbolic::symbolic_functions::Jacobian;
/// use nalgebra::DVector;
///     let vec_of_expressions = vec!["x^2+y^2-10", "x-y-4"];
///
///     let initial_guess = vec![1.0, 1.0];
///     let mut NR_instanse = NR::new();
///     let vec_of_expr = Expr::parse_vector_expression(vec_of_expressions.clone());
///     let values = vec!["x".to_string(), "y".to_string()];
///     NR_instanse.set_equation_system(vec_of_expr, Some(values.clone()), initial_guess, 1e-6, 100);
///     NR_instanse.eq_generate();
///     NR_instanse.main_loop();
///     let solution = NR_instanse.get_result().unwrap();
///     assert_eq!(solution, DVector::from(vec![-1.0, 3.0] ));
///     println!("result = {:?} \n", NR_instanse.get_result().unwrap());
///  ```
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions::Jacobian;
use chrono::Local;
use log::{error, info, warn};
use simplelog::LevelFilter;
use simplelog::*;
use tabled::assert;
use std::collections::HashMap;
use std::time::Instant;
use tabled::{builder::Builder, settings::Style};
use std::error::Error;
use nalgebra::{DMatrix, DVector, Matrix};
pub struct NR {
    pub jacobian: Jacobian, // instance of Jacobian struct, contains jacobian matrix function and equation functions
    pub eq_system: Vec<Expr>, // vector of equations
    pub values: Vec<String>, // vector of variables
    pub initial_guess: Vec<f64>, // initial guess
    pub tolerance: f64,     // tolerance
    pub max_iterations: usize, // max number of iterations

    max_error: f64, // max error
    pub dumping_factor: f64,
    pub i: usize,                 // iteration counter
    pub jac: DMatrix<f64>,        // jacobian matrix
    pub fun_vector: DVector<f64>,     // vector of functions
    pub result: Option<DVector<f64>>, // result of the iteration

    pub loglevel: Option<String>,
    pub linear_sys_method: Option<String>, // method for solving linear system
    pub custom_timer: CustomTimer,
    calc_statistics: HashMap<String, usize>,
}

impl NR {
    pub fn new() -> NR {
        //jacobian: Jacobian, initial_guess: Vec<f64>, tolerance: f64, max_iterations: usize, max_error: f64, result: Option<Vec<f64>>
        NR {
            jacobian: Jacobian::new(),
            eq_system: Vec::new(),
            values: Vec::new(),
            initial_guess: Vec::new(),
            tolerance: 1e-6,
            max_iterations: 100,
            max_error: 0.0,
            dumping_factor: 1.0,
            i: 0,
            jac: DMatrix::zeros(0, 0),
            fun_vector: DVector::zeros(0),
            result: None,
            loglevel: Some("info".to_string()),
            linear_sys_method: Some("lu".to_string()),
            custom_timer: CustomTimer::new(),
            calc_statistics: HashMap::new(),
        }
    }
    ////////////////////////////SETTERS///////////////////////////////////////////////////////////////////
    /// Basic methods to set the equation system
    pub fn set_equation_system(
        &mut self,
        eq_system: Vec<Expr>,
        unknowns: Option<Vec<String>>,
        initial_guess: Vec<f64>,
        tolerance: f64,
        max_iterations: usize,
    ) {
        self.eq_system = eq_system.clone();
        self.initial_guess = initial_guess;
        self.tolerance = tolerance;
        self.max_iterations = max_iterations;
        let values = if let Some(values) = unknowns {
            values
        } else {
            let mut args: Vec<String> = eq_system
                .iter()
                .map(|x| x.all_arguments_are_variables())
                .flatten()
                .collect::<Vec<String>>();
            args.sort();
            args.dedup();

            assert!(!args.is_empty(), "No variables found in the equations.");
            assert_eq!(
                args.len() == eq_system.len(),
                true,
                "Equation system and vector of variables should have the same length."
            );

            args
        };
        self.values = values.clone();
        assert!(
            !self.initial_guess.is_empty(),
            "Initial guess should not be empty."
        );
        assert!(
            tolerance >= 0.0,
            "Tolerance should be a non-negative number."
        );
        assert!(
            max_iterations > 0,
            "Max iterations should be a positive number."
        );
    }

    pub fn eq_generate_from_str(
        &mut self,
        eq_system_string: Vec<String>,
        unknowns: Option<Vec<String>>,
        initial_guess: Vec<f64>,
        tolerance: f64,
        max_iterations: usize,
    ) {
        let eq_system = eq_system_string
            .iter()
            .map(|x| Expr::parse_expression(x))
            .collect::<Vec<Expr>>();
        self.set_equation_system(
            eq_system,
            unknowns,
            initial_guess,
            tolerance,
            max_iterations,
        );
        self.eq_generate();
    }
    pub fn set_solver_params(
        &mut self,
        loglevel: Option<String>,
        linear_sys_method: Option<String>,
        damping_factor: Option<f64>,
    ) {
        self.loglevel = if let Some(level) = loglevel {
            assert!(level == "debug" || level == "info" || level == "warn" || level == "error", "loglevel must be debug/info, warn or error");
            Some(level.to_string())
        } else {
            self.loglevel.clone()
        };
        self.linear_sys_method = if let Some(method) = linear_sys_method {
            let method = method.to_lowercase();
            assert!(method == "lu" || method == "inv", "linear_sys_method must be lu or inv");

            Some(method.to_string())
        } else {
            self.linear_sys_method.clone()
        };
        self.dumping_factor = if let Some(dumping_factor) = damping_factor {
            assert!(
                dumping_factor >= 0.0 && dumping_factor <= 1.0,
                "Dumping factor should be between 0.0 and 1.0."
            );
            dumping_factor
        } else {
            self.dumping_factor
        };
    }
    ///Set system of equations with vector of symbolic expressions
    pub fn eq_generate(&mut self) {
        let eq_system = self.eq_system.clone();
        let mut Jacobian_instance = Jacobian::new();
        let args = self.values.clone();
        let args: Vec<&str> = args.iter().map(|x| x.as_str()).collect();
        Jacobian_instance.set_vector_of_functions(eq_system);
        Jacobian_instance.set_variables(args.clone());
        Jacobian_instance.calc_jacobian();
        Jacobian_instance.jacobian_generate(args.clone());
        Jacobian_instance.lambdify_funcvector(args);
        assert_eq!(
            Jacobian_instance.vector_of_variables.len(),
            self.initial_guess.len(),
            "Initial guess and vector of variables should have the same length."
        );
        self.jacobian = Jacobian_instance;
    }
    /////////////////////////////////////////////////////////////////////////////////////////////
    //                ITERATIONS
    /////////////////////////////////////////////////////////////////////////////////////////////
    // Newton-Raphson method
    /// realize iteration of Newton-Raphson - calculate new iteration vector by using Jacobian matrix
    pub fn iteration(&mut self, x: DVector<f64>) -> DVector<f64> {
        let method = self.linear_sys_method.clone().unwrap();
        let Jacobian_instance = &mut self.jacobian;
        // evaluate jacobian and functions
        self.custom_timer.jac_tic();
        Jacobian_instance.evaluate_func_jacobian_DMatrix(x.clone().data.into());
        self.custom_timer.jac_tac();
        self.custom_timer.fun_tic();
        Jacobian_instance.evaluate_funvector_lambdified_DVector(x.clone().data.into());
        self.custom_timer.fun_tac();
        assert!(
            !Jacobian_instance.evaluated_jacobian_DMatrix.is_empty(),
            "Jacobian should not be empty."
        );
        assert!(
            !Jacobian_instance.evaluated_functions_DVector.is_empty(),
            "Functions should not be empty."
        );
        self.custom_timer.linear_system_tic();
        let new_j = &Jacobian_instance.evaluated_jacobian_DMatrix;
        self.jac = new_j.clone();
        let new_f = &Jacobian_instance.evaluated_functions_DVector;
        let delta = Self:: solve_linear_system(method, new_j, new_f).unwrap();

        let lambda = self.dumping_factor;
        let new_x: DVector<f64> = x - lambda* delta;
       // let dx: Vec<f64> = delta.data.into(); //.iter().map(|x| *x).collect();
        // element wise subtraction

        self.custom_timer.linear_system_tac();

        new_x
    }
    /// main function to solve the system of equations  
    pub fn main_loop(&mut self) -> Option<DVector<f64>> {
        let  x = self.initial_guess.clone();
        let mut x = DVector::from_vec(x);
        self.result = Some(x.clone()); // save into result in case the very first iteration
        while self.i < self.max_iterations {
            let new_x = self.iteration(x.clone());

            let dx: DVector<f64> = new_x.clone() - x;
            let error = Matrix::norm(&dx);
            if (error > self.max_error) && (self.i > 0) {
                warn!("Error is increasing");
            }
            self.max_error = error;
            if error < self.tolerance {
                self.result = Some(new_x.clone());
                self.max_error = error;
                return Some(new_x);
            } else {
                x = new_x;
                self.i += 1;
                info!("iteration = {}, error = {}", self.i, error)
            }
        }
        error!("Maximum number of iterations reached. No solution found.");
        None
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                       main functions to start the solver and caclulate statistics
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //

    pub fn solver(&mut self) -> Option<DVector<f64>> {
        self.custom_timer.start();
        self.custom_timer.symbolic_operations_tic();
        self.eq_generate();
        self.custom_timer.symbolic_operations_tac();
        let begin = Instant::now();
        let res = self.main_loop();
        self.custom_timer.get_all();
        let end = begin.elapsed();
        elapsed_time(end);
        let time = end.as_secs_f64() as usize;

        self.calc_statistics
            .insert("time elapsed, s".to_string(), time);
        self.calc_statistics();

        self.result = res;
        self.result.clone()
    }
    // wrapper around solver function to implement logging
    pub fn solve(&mut self) -> Option<DVector<f64>> {
        let is_logging_disabled = self
            .loglevel
            .as_ref()
            .map(|level| level == "off" || level == "none")
            .unwrap_or(false);

        if is_logging_disabled {
            let res = self.solver();
            res
        } else {
            let loglevel = self.loglevel.clone();
            let log_option = if let Some(level) = loglevel {
                match level.as_str() {
                    "debug" => LevelFilter::Info,
                    "info" => LevelFilter::Info,
                    "warn" => LevelFilter::Warn,
                    "error" => LevelFilter::Error,
                    _ => panic!("loglevel must be debug, info, warn or error"),
                }
            } else {
                LevelFilter::Info
            };
            println!(" \n \n Program started with loglevel: {}", log_option);
          //  let date_and_time = Local::now().format("%Y-%m-%d_%H-%M-%S");
          //  let name = format!("log_{}.txt", date_and_time);
            let logger_instance = CombinedLogger::init(vec![TermLogger::new(
                log_option,
                Config::default(),
                TerminalMode::Mixed,
                ColorChoice::Auto,
            )]);

            match logger_instance {
                Ok(()) => {
                    let res = self.solver();
                    info!(" \n \n Program ended");
                    res
                }
                Err(_) => {
                    let res = self.solver();
                    res
                } //end Error
            } // end mat 
        }
    }
    pub fn get_result(&self) -> Option<DVector<f64>> {
        self.result.clone()
    }
    fn calc_statistics(&self) {
        let mut stats = self.calc_statistics.clone();
        let jac = &self.jac;
        let jac_shape = jac.shape();
        let matrix_weight = checkmem(jac);
        stats.insert("jacobian memory, MB".to_string(), matrix_weight as usize);
        stats.insert(
            "number of jacobian elements".to_string(),
            jac_shape.0 * jac_shape.1,
        );

        stats.insert("length of y vector".to_string(), self.values.len() as usize);
        stats.insert("number of iterations".to_string(), self.i as usize);
        let mut table = Builder::from(stats).build();
        table.with(Style::modern_rounded());
        info!("\n \n CALC STATISTICS \n \n {}", table.to_string());
    }
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///                 LINEAR SYSTEM SOLVERS
    //////////////////////////////////////////////////////////////////////////////////////////////
    /* 
    pub fn get_error(&mut self, x: Vec<f64>) -> f64 {
        let Jacobian_instance = &mut self.jacobian;
        Jacobian_instance.evaluate_funvector_lambdified_DVector(x.clone());
        let new_x = &Jacobian_instance.evaluated_functions_DVector;
        let dx = new_x
            .iter()
            .zip(&x)
            .map(|(x_i, x_j)| (x_i - x_j).abs())
            .collect::<Vec<f64>>();
        let dx_matrix = DVector::from_vec(dx);
        let error = Matrix::norm(&dx_matrix);
        error
    }

    pub fn test_correction(&mut self) -> f64 {
        let result = self.get_result().clone().unwrap().clone();
        let norm = self.get_error(result);
        norm.clone()
    }

    // Gauss-Jordan elimination method. The function takes two parameters: matrix, which is a reference to a vector of vectors representing the coefficients of the linear equations,
    // and constants, which is a reference to a vector containing the constants on the right-hand side of the equations.

    pub fn solve_linear_system(matrix: &[Vec<f64>], constants: &[f64]) -> Vec<f64> {
        // Implement a linear system solver (e.g., LU decomposition, Gauss-Jordan elimination, etc.)
        // Here, we'll use a simple implementation for demonstration purposes
        let n = matrix.len();
        let mut augmented_matrix = matrix
            .iter()
            .cloned()
            .zip(constants.iter().cloned())
            .map(|(row, constant)| {
                row.into_iter()
                    .chain(std::iter::once(constant))
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>();

        for i in 0..n {
            let pivot = augmented_matrix[i][i];
            for j in i..n + 1 {
                augmented_matrix[i][j] /= pivot;
            }

            for k in 0..n {
                if k != i {
                    let factor = augmented_matrix[k][i];
                    for j in i..n + 1 {
                        augmented_matrix[k][j] -= factor * augmented_matrix[i][j];
                    }
                }
            }
        }

        augmented_matrix.iter().map(|row| row[n]).collect()
    }
    pub fn solve_linear_LU(
        coeffs: Vec<Vec<f64>>,
        constants: Vec<f64>,
    ) -> Result<Vec<f64>, &'static str> {
        let mut res: Vec<f64> = Vec::new();
        let n = coeffs.len();
        let a: DMatrix<f64> = DMatrix::from_fn(n, n, |i, j| coeffs[i][j]);
        let b: DVector<f64> = DVector::from_vec(constants);
        match a.lu().solve(&b) {
            Some(x) => {
                info!("Solution: {}", x);
                res = x.data.into();
                Ok(res)
            }
            None => {
                info!("No solution found");
                Err("no solution")
            }
        }
    }
*/
    pub fn solve_linear_system(solver: String, A:&DMatrix<f64>, b: &DVector<f64>) -> Result<DVector<f64>, Box<dyn Error> > {
    
        match solver.as_str() {
            "lu" => {
                let lu = A.clone().lu();
                let x = lu.solve(&b);
                match x {
                    Some(x) => Ok(x),
                    None => Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        "Failed to solve the system",
                    ))),
                }
            }
            "inv" =>{
                let A_inv = A.clone().try_inverse().unwrap(); 
                let f = A_inv * b; 
                Ok(f)
            }
         _ => Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Failed to solve the system",
         ))) 
                
        }// match solver.as_str() {

    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                     TESTS
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#[test]
fn test_NR_set_equation_sysytem() {
    let vec_of_expressions = vec!["x^2+y^2-10", "x-y-4"];

    let initial_guess = vec![1.0, 1.0];
    let mut NR_instanse = NR::new();
    let vec_of_expr = Expr::parse_vector_expression(vec_of_expressions.clone());
    let values = vec!["x".to_string(), "y".to_string()];
    NR_instanse.set_equation_system(vec_of_expr, Some(values.clone()), initial_guess, 1e-6, 100);
    NR_instanse.eq_generate();
    NR_instanse.main_loop();
    let solution = NR_instanse.get_result().unwrap();
   assert_eq!(solution, DVector::from(vec![-1.0, 3.0] ));
}

#[test]
fn test_NR_eq_generate_from_str() {
    let mut NR_instanse = NR::new();
    let vec_of_expressions = vec!["x^2+y^2-10".to_string(), "x-y-4".to_string()];
    let initial_guess = vec![1.0, 1.0];
    // solve
    NR_instanse.eq_generate_from_str(vec_of_expressions, None, initial_guess, 1e-6, 100);
    NR_instanse.main_loop();
    let solution = NR_instanse.get_result().unwrap();
    assert_eq!(solution, DVector::from(vec![-1.0, 3.0] ));
}

#[test]
fn test_NR_set_equation_sysytem_with_features() {
    let vec_of_expressions = vec!["x^2+y^2-10", "x-y-4"];

    let initial_guess = vec![1.0, 1.0];
    let mut NR_instanse = NR::new();
    let vec_of_expr = Expr::parse_vector_expression(vec_of_expressions.clone());
    let values = vec!["x".to_string(), "y".to_string()];
    NR_instanse.set_equation_system(vec_of_expr, Some(values.clone()), initial_guess, 1e-6, 100);
    NR_instanse.set_solver_params(Some("info".to_string()), None, None);
    NR_instanse.eq_generate();
    NR_instanse.solve();
    let solution = NR_instanse.get_result().unwrap();
    println!("solution: {:?}", solution);

    assert_eq!(solution, DVector::from(vec![-1.0, 3.0] ));
}
