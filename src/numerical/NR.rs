///  Example#1
/// ```
///  use RustedSciThe::numerical::NR::NR;
/// //use the shortest way to solve system of equations
///    // first define system of equations and initial guess
///    let mut NR_instanse = NR::new();
///    let vec_of_expressions = vec![ "x^2+y^2-10".to_string(), "x-y-4".to_string()];
///   let initial_guess = vec![1.0, 1.0];
///    // solve
///    NR_instanse.eq_generate_from_str(vec_of_expressions,initial_guess, 1e-6, 100, );
///    NR_instanse.solve();
///    println!("result = {:?} \n", NR_instanse.get_result().unwrap());
///  ```
/// Example#2
///     ```
///    // or more verbose way...
///    // first define system of equations
///     use RustedSciThe::numerical::NR::NR;
///     use RustedSciThe::symbolic::symbolic_engine::Expr;
///     use RustedSciThe::symbolic::symbolic_functions::Jacobian;
///    let vec_of_expressions = vec![ "x^2+y^2-10".to_string(), "x-y-4".to_string()];
///    let mut Jacobian_instance = Jacobian::new();
///     Jacobian_instance.set_funcvecor_from_str(vec_of_expressions);
///     Jacobian_instance.set_variables(vec!["x", "y"]);
///     Jacobian_instance.calc_jacobian();
///     Jacobian_instance.jacobian_generate(vec!["x", "y"]);
///     Jacobian_instance.lambdify_funcvector(vec!["x", "y"]);
///     Jacobian_instance.readable_jacobian();
///     println!("Jacobian_instance: functions  {:?}. Variables {:?}", Jacobian_instance.vector_of_functions, Jacobian_instance.vector_of_variables);
///      println!("Jacobian_instance: Jacobian  {:?} readable {:?}. \n", Jacobian_instance.symbolic_jacobian, Jacobian_instance.readable_jacobian);
///     let initial_guess = vec![1.0, 1.0];
///     // in case you are interested in Jacobian value at initial guess
///     Jacobian_instance.evaluate_func_jacobian_DMatrix(initial_guess.clone());
///     Jacobian_instance.evaluate_funvector_lambdified_DVector(initial_guess.clone());
///     let guess_jacobian = (Jacobian_instance.evaluated_jacobian_DMatrix).clone();
///     println!("guess Jacobian = {:?} \n", guess_jacobian.try_inverse());
///     // defining NR method instance and solving
///     let mut NR_instanse = NR::new();
///     NR_instanse.set_equation_sysytem(Jacobian_instance, initial_guess, 1e-6, 100, );
///     NR_instanse.solve();
///     println!("result = {:?} \n", NR_instanse.get_result().unwrap());
///     ```
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions::Jacobian;
use log::info;
use nalgebra::{DMatrix, DVector, Matrix};
pub struct NR {
    jacobian: Jacobian, // instance of Jacobian struct, contains jacobian matrix function and equation functions
    initial_guess: Vec<f64>, // initial guess
    tolerance: f64,     // tolerance
    max_iterations: usize, // max number of iterations
    max_error: f64,     // max error
    result: Option<Vec<f64>>, // result of the iteration
}

impl NR {
    pub fn new() -> NR {
        //jacobian: Jacobian, initial_guess: Vec<f64>, tolerance: f64, max_iterations: usize, max_error: f64, result: Option<Vec<f64>>
        NR {
            jacobian: Jacobian::new(),
            initial_guess: Vec::new(),
            tolerance: 1e-6,
            max_iterations: 100,
            max_error: 0.0,
            result: None,
        }
    }
    /// Basic methods to set the equation system
    pub fn set_equation_sysytem(
        &mut self,
        jacobian: Jacobian,
        initial_guess: Vec<f64>,
        tolerance: f64,
        max_iterations: usize,
    ) {
        assert_eq!(
            jacobian.vector_of_variables.len(),
            initial_guess.len(),
            "Initial guess and vector of variables should have the same length."
        );

        self.jacobian = jacobian;
        self.initial_guess = initial_guess;
        self.tolerance = tolerance;
        self.max_iterations = max_iterations;

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
    ///Set system of equations with vector of symbolic expressions
    pub fn eq_generate(
        &mut self,
        eq_system: Vec<Expr>,
        initial_guess: Vec<f64>,
        tolerance: f64,
        max_iterations: usize,
    ) {
        let mut args_: Vec<String> = eq_system
            .iter()
            .map(|x| x.all_arguments_are_variables())
            .flatten()
            .collect::<Vec<String>>();
        args_.sort();
        args_.dedup();
        let args: Vec<&str> = args_.iter().map(|x| x.as_str()).collect();
        assert!(!args.is_empty(), "No variables found in the equations.");
        assert_eq!(
            args.len() == eq_system.len(),
            true,
            "Equation system and vector of variables should have the same length."
        );
        let mut Jacobian_instance = Jacobian::new();
        Jacobian_instance.set_vector_of_functions(eq_system);
        Jacobian_instance.set_variables(args.clone());
        Jacobian_instance.calc_jacobian();
        Jacobian_instance.jacobian_generate(args.clone());
        Jacobian_instance.lambdify_funcvector(args);
        self.jacobian = Jacobian_instance;
        self.initial_guess = initial_guess;
        self.tolerance = tolerance;
        self.max_iterations = max_iterations;
    }

    pub fn eq_generate_from_str(
        &mut self,
        eq_system_string: Vec<String>,
        initial_guess: Vec<f64>,
        tolerance: f64,
        max_iterations: usize,
    ) {
        let eq_system = eq_system_string
            .iter()
            .map(|x| Expr::parse_expression(x))
            .collect::<Vec<Expr>>();
        assert!(
            !eq_system.is_empty(),
            "Equation system should not be empty."
        );
        assert!(
            !initial_guess.is_empty(),
            "Initial guess should not be empty."
        );
        assert_eq!(
            eq_system.len() == initial_guess.len(),
            true,
            "Equation system and initial guess should have the same length."
        );
        self.eq_generate(eq_system, initial_guess, tolerance, max_iterations);
    }

    ///Newton-Raphson method
    /// realize iteration of Newton-Raphson - calculate new iteration vector by using Jacobian matrix
    pub fn iteration(&mut self, x: Vec<f64>) -> Vec<f64> {
        let Jacobian_instance = &mut self.jacobian;
        // evaluate jacobian and functions
        Jacobian_instance.evaluate_func_jacobian_DMatrix(x.clone());
        Jacobian_instance.evaluate_funvector_lambdified_DVector(x.clone());
        assert!(
            !Jacobian_instance.evaluated_jacobian_DMatrix.is_empty(),
            "Jacobian should not be empty."
        );
        assert!(
            !Jacobian_instance.evaluated_functions_DVector.is_empty(),
            "Functions should not be empty."
        );
        let new_j = &Jacobian_instance.evaluated_jacobian_DMatrix;
        let new_f = &Jacobian_instance.evaluated_functions_DVector;
        let j_inverse = new_j.clone().try_inverse().unwrap();
        //  let j_inverse = <Matrix<f64, Dyn, Dyn, VecStorage<f64, Dyn, Dyn>> as Clone>::clone(&new_j).try_inverse().unwrap();
        let delta: DVector<f64> = j_inverse * new_f;
        let dx: Vec<f64> = delta.data.into(); //.iter().map(|x| *x).collect();
                                              // element wise subtraction
        let new_x = x
            .iter()
            .zip(&dx)
            .map(|(x_i, dx_i)| x_i - dx_i)
            .collect::<Vec<f64>>();

        new_x
    }
    /// main function to solve the system of equations  
    pub fn solve(&mut self) -> Option<Vec<f64>> {
        let mut x = self.initial_guess.clone();
        let mut i = 0;
        while i < self.max_iterations {
            let new_x = self.iteration(x.clone());

            let dx = new_x
                .iter()
                .zip(&x)
                .map(|(x_i, x_j)| (x_i - x_j).abs())
                .collect::<Vec<f64>>();
            let dx_matrix = DVector::from_vec(dx);
            let error = Matrix::norm(&dx_matrix);
            if error < self.tolerance {
                self.result = Some(new_x.clone());
                self.max_error = error;
                return Some(new_x);
            } else {
                x = new_x;
                i += 1;
                info!("iteration = {}, error = {}", i, error)
            }
        }
        None
    }

    pub fn get_result(&self) -> Option<Vec<f64>> {
        self.result.clone()
    }

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
}

#[test]
fn test_NR_set_equation_sysytem() {
    let vec_of_expressions = vec!["x^2+y^2-10".to_string(), "x-y-4".to_string()];
    let mut Jacobian_instance = Jacobian::new();
    Jacobian_instance.set_funcvecor_from_str(vec_of_expressions);
    Jacobian_instance.set_variables(vec!["x", "y"]);
    Jacobian_instance.calc_jacobian();
    Jacobian_instance.jacobian_generate(vec!["x", "y"]);
    Jacobian_instance.lambdify_funcvector(vec!["x", "y"]);
    Jacobian_instance.readable_jacobian();
    let initial_guess = vec![1.0, 1.0];
    let mut NR_instanse = NR::new();
    NR_instanse.set_equation_sysytem(Jacobian_instance, initial_guess, 1e-6, 100);
    NR_instanse.solve();
    let solution = NR_instanse.get_result().unwrap();
    assert_eq!(solution, vec![-1.0, 3.0]);
}

#[test]
fn test_NR_eq_generate_from_str() {
    let mut NR_instanse = NR::new();
    let vec_of_expressions = vec!["x^2+y^2-10".to_string(), "x-y-4".to_string()];
    let initial_guess = vec![1.0, 1.0];
    // solve
    NR_instanse.eq_generate_from_str(vec_of_expressions, initial_guess, 1e-6, 100);
    NR_instanse.solve();
    let solution = NR_instanse.get_result().unwrap();
    assert_eq!(solution, vec![-1.0, 3.0]);
}
