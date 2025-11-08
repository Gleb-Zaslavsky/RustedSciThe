//!
//! # BVP Symbolic Functions Module
//!
//! This module provides symbolic-to-numerical conversion for Boundary Value Problems (BVP)
//! using the faer linear algebra library. It converts symbolic expressions into efficient
//! numerical closures for use with BVP solvers.
//!
//! ## Key Features
//! - Convert symbolic ODE systems to numerical residual functions
//! - Generate analytical Jacobians from symbolic expressions
//! - Parallel evaluation using rayon for performance
//! - Sparse matrix support for large systems
//! - Bandwidth optimization for banded Jacobians
//!
//! ## Usage
//! ```rust,ignore
//! let mut jacobian_instance = Jacobian_sci_faer { /* ... */ };
//! jacobian_instance.generate_BVP(equations, variables, params, arg, boundary_conditions);
//! // Use jacobian_instance.residiual_function and jacobian_instance.jac_function
//! ```

use crate::numerical::BVP_Damp::BVP_utils::elapsed_time;
use crate::numerical::BVP_sci::BVP_sci_faer::{
    ODEFunction, ODEJacobian, faer_col, faer_dense_mat, faer_mat,
};
use crate::symbolic::symbolic_engine::Expr;
use faer::mat::Mat;
use log::info;
use std::collections::HashMap;
use std::time::Instant;

use faer::sparse::{SparseColMat, Triplet};
use rayon::prelude::*;
use tabled::{builder::Builder, settings::Style};
/// Main structure for symbolic BVP processing
///
/// Handles conversion of symbolic ODE systems to numerical functions
/// suitable for BVP solvers using faer sparse matrices.
pub struct Jacobian_sci_faer {
    /// Vector of symbolic ODE expressions (RHS of dy/dx = f(x,y,p))
    pub vector_of_functions: Vec<Expr>,
    /// Vector of symbolic variables (unknowns in the system)
    pub vector_of_variables: Vec<Expr>,
    /// String representation of variable names
    pub variable_string: Vec<String>,
    /// Symbolic Jacobian matrix (df/dy)
    pub symbolic_jacobian: Vec<Vec<Expr>>,
    /// Numerical Jacobian function closure
    pub jac_function: Option<Box<ODEJacobian>>,
    /// Numerical residual function closure
    pub residual_function: Box<ODEFunction>,
    /// Jacobian bandwidth (kl, ku) for banded matrices
    pub bandwidth: Option<(usize, usize)>,
    /// Variables used in each equation (for optimization)
    pub variables_for_all_eq: Vec<Vec<String>>,
}
impl Jacobian_sci_faer {
    /// Create new empty instance
    pub fn new() -> Self {
        Self {
            vector_of_functions: vec![],
            vector_of_variables: vec![],
            variable_string: vec![],
            symbolic_jacobian: vec![],
            jac_function: None,
            residual_function: Box::new(|_, _, _| Mat::zeros(0, 0)),
            bandwidth: None,
            variables_for_all_eq: vec![],
        }
    }
    //
    /// Initialize the structure from symbolic expressions and variable names
    ///
    /// # Arguments
    /// * `vector_of_functions` - Vector of symbolic ODE expressions
    /// * `variable_string` - Names of unknown variables
    pub fn from_vectors(&mut self, vector_of_functions: Vec<Expr>, variable_string: Vec<String>) {
        self.vector_of_functions = vector_of_functions.clone();
        self.variable_string = variable_string.clone();
        self.vector_of_variables =
            Expr::parse_vector_expression(variable_string.iter().map(|s| s.as_str()).collect());

        // varaibles from every expression in the system
        let vars_from_flat: Vec<Vec<String>> = vector_of_functions
            .clone()
            .iter()
            .map(|exp_i| Expr::all_arguments_are_variables(exp_i))
            .collect();
        self.variables_for_all_eq = vars_from_flat.clone();
        println!(" {:?}", self.vector_of_functions);
        println!(" {:?}", self.vector_of_variables);
    }
    /// Calculate symbolic Jacobian matrix using parallel differentiation
    ///
    /// Computes df/dy for each function with respect to each variable.
    /// Uses parallel processing and smart optimization to skip zero derivatives.
    pub fn calc_jacobian_parallel_smart(&mut self) {
        assert!(self.variables_for_all_eq.len() > 0);
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
            .enumerate()
            .map(|(i, function)| {
                let mut vector_of_partial_derivatives = Vec::new();
                // let function = function.clone();
                for j in 0..self.vector_of_variables.len() {
                    let variable = &variable_string_vec[j]; // obviously if function does not contain variable its derivative should be 0
                    let list_of_variables_for_this_eq = &self.variables_for_all_eq[i]; // so we can only calculate derivative for variables that are used in this equation
                    if list_of_variables_for_this_eq.contains(variable) {
                        let mut partial = Expr::diff(&function, variable);
                        partial = partial.simplify();
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
    /// Determine Jacobian matrix bandwidth for sparse storage optimization
    ///
    /// Calculates the number of sub/super-diagonals (kl, ku) in the Jacobian
    /// to optimize sparse matrix operations.
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
    /// Convert symbolic vector to BVP residual function closure
    /// Convert symbolic expressions to numerical residual function closure
    ///
    /// Creates a closure that evaluates the ODE system at mesh points.
    ///
    /// # Arguments
    /// * `arg` - Independent variable name (e.g., "x")
    /// * `var_names` - Unknown variable names
    /// * `par_names` - Parameter names
    pub fn symbolic_to_ode_function2(
        &mut self,
        arg: String,
        var_names: Vec<String>,
        par_names: Vec<String>,
        Bounds: Option<HashMap<String, Vec<(usize, f64)>>>,
    ) {
        let vector_of_functions = self.vector_of_functions.clone();
        //let var_names: Vec<String> = variable_names.iter().map(|s| s.to_string()).collect();
        //let par_names: Vec<String> = param_names.iter().map(|s| s.to_string()).collect();
        let residual = Box::new(
            move |x: &faer_col, y: &faer_dense_mat, p: &faer_col| -> faer_dense_mat {
                let (n, m) = (y.nrows(), y.ncols());
                let mut result = faer_dense_mat::zeros(n, m);

                let columns: Vec<Vec<f64>> = (0..m) // m is the number of time steps
                    .into_par_iter()
                    .map(|j| {
                        let mut args = Vec::with_capacity(1 + n + p.nrows()); // n is the number of variables and p is the number of parameters
                        args.push(x[j]);
                        for i in 0..n {
                            let y = *y.get(i, j);
                            let bounded_value =
                                Self::handle_bounds(i, y, Bounds.clone(), var_names.clone());
                            args.push(bounded_value);
                        }
                        for i in 0..p.nrows() {
                            args.push(p[i]);
                        }

                        let mut all_var_names = vec![arg.as_str()];
                        all_var_names.extend(var_names.iter().map(|s| s.as_str()));
                        all_var_names.extend(par_names.iter().map(|s| s.as_str()));

                        vector_of_functions
                            .iter()
                            .map(|func| func.eval_expression(all_var_names.as_slice(), &args))
                            .collect()
                    })
                    .collect();

                for (j, col) in columns.iter().enumerate() {
                    // j is the index of the time step
                    for (i, value) in col.iter().enumerate() {
                        // i is the index of the variable
                        // let bounded_value = Self::handle_bounds(i, *value, Bounds.clone(), var_names.clone());
                        *result.get_mut(i, j) = *value;
                    }
                }

                result
            },
        );
        self.residual_function = residual;
    }

    /// Apply variable bounds to residual values to prevent numerical issues
    ///
    /// # Arguments
    /// * `var_index` - Index of the variable in the system
    /// * `value` - Computed residual value
    /// * `bounds` - Optional bounds map: variable_name -> [(bound_type, bound_value)]
    ///   where bound_type: 0=minimum, 1=maximum
    /// * `var_names` - Variable names corresponding to indices
    ///
    /// # Returns
    /// Bounded value within specified limits
    ///
    pub fn symbolic_to_ode_function(
        &mut self,
        arg: String,
        var_names: Vec<String>,
        par_names: Vec<String>,
        Bounds: Option<HashMap<String, Vec<(usize, f64)>>>,
    ) {
        let vector_of_functions = self.vector_of_functions.clone();
        //let var_names: Vec<String> = variable_names.iter().map(|s| s.to_string()).collect();
        //let par_names: Vec<String> = param_names.iter().map(|s| s.to_string()).collect();
        let residual = Box::new(
            move |x: &faer_col, y: &faer_dense_mat, p: &faer_col| -> faer_dense_mat {
                let (n, m) = (y.nrows(), y.ncols());
                let mut result = faer_dense_mat::zeros(n, m);

                let columns: Vec<Vec<f64>> = (0..m) // m is the number of time steps
                    .into_par_iter()
                    .map(|j| {
                        let mut args = Vec::with_capacity(1 + n + p.nrows()); // n is the number of variables and p is the number of parameters
                        args.push(x[j]);
                        for i in 0..n {
                            let y = *y.get(i, j);
                            let bounded_value =
                                Self::handle_bounds(i, y, Bounds.clone(), var_names.clone());
                            args.push(bounded_value);
                        }
                        for i in 0..p.nrows() {
                            args.push(p[i]);
                        }

                        let mut all_var_names = vec![arg.as_str()];
                        all_var_names.extend(var_names.iter().map(|s| s.as_str()));
                        all_var_names.extend(par_names.iter().map(|s| s.as_str()));

                        let result: Vec<_> = vector_of_functions
                            .iter()
                            .map(|func| {
                                let func = Expr::lambdify_borrowed_thread_safe(
                                    &func,
                                    all_var_names.as_slice(),
                                );

                                func(args.as_slice())
                            })
                            .collect();
                        result
                    })
                    .collect();

                for (j, col) in columns.iter().enumerate() {
                    // j is the index of the time step
                    for (i, value) in col.iter().enumerate() {
                        // i is the index of the variable
                        // let bounded_value = Self::handle_bounds(i, *value, Bounds.clone(), var_names.clone());
                        *result.get_mut(i, j) = *value;
                    }
                }

                result
            },
        );
        self.residual_function = residual;
    }

    pub fn handle_bounds(
        var_index: usize,
        value: f64,
        bounds: Option<HashMap<String, Vec<(usize, f64)>>>,
        var_names: Vec<String>,
    ) -> f64 {
        let Some(bounds) = bounds else { return value };

        let var_name = &var_names[var_index];

        let Some(var_bounds) = bounds.get(var_name) else {
            return value;
        };

        let mut bounded_value = value;
        for &(bound_type, bound_val) in var_bounds {
            // println!("Applying bounds {:?} for variable {}", var_bounds.clone(), var_name);
            bounded_value = match bound_type {
                0 => bounded_value.max(bound_val), // minimum bound
                1 => bounded_value.min(bound_val), // maximum bound
                _ => {
                    eprintln!(
                        "Warning: Unknown bound type {} for variable {}",
                        bound_type, var_name
                    );
                    bounded_value
                }
            };
        }
        bounded_value
    }
    /// Generate sparse Jacobian function with parallel evaluation
    ///
    /// Creates a closure that computes sparse Jacobian matrices at mesh points.
    /// Supports both df/dy and df/dp derivatives with bandwidth optimization.
    ///
    /// # Arguments
    /// * `jac_dy` - Symbolic Jacobian w.r.t. variables
    /// * `jac_dp` - Optional symbolic Jacobian w.r.t. parameters
    /// * `variable_str` - Variable names
    /// * `param_str` - Parameter names
    /// * `arg` - Independent variable name
    /// * `bandwidth` - Optional bandwidth (kl, ku) for optimization
    pub fn jacobian_generate_SparseColMat_par_sci(
        jac_dy: Vec<Vec<Expr>>,
        jac_dp: Option<Vec<Vec<Expr>>>,
        variable_str: Vec<String>,
        param_str: Vec<String>,
        arg: String,
        bandwidth: Option<(usize, usize)>,
        bounds: Option<HashMap<String, Vec<(usize, f64)>>>,
    ) -> Box<dyn Fn(&faer_col, &faer_dense_mat, &faer_col) -> (Vec<faer_mat>, Option<Vec<faer_mat>>)>
    {
        Box::new(
            move |x: &faer_col,
                  y: &faer_dense_mat,
                  p: &faer_col|
                  -> (Vec<faer_mat>, Option<Vec<faer_mat>>) {
                let (n, m) = (y.nrows(), y.ncols());
                let k = p.nrows();

                let df_dy: Vec<faer_mat> = (0..m)
                    .into_par_iter()
                    .map(|j| {
                        let mut triplets = Vec::new();
                        let triplets_mutex = std::sync::Mutex::new(&mut triplets);

                        (0..n).into_par_iter().for_each(|i| {
                            let (right_border, left_border) = if let Some((kl, ku)) = bandwidth {
                                let right_border = std::cmp::min(i + ku + 1, n);
                                let left_border = if i as i32 - (kl as i32) - 1 < 0 {
                                    0
                                } else {
                                    i - kl - 1
                                };
                                (right_border, left_border)
                            } else {
                                (n, 0)
                            };

                            for col in left_border..right_border {
                                if !jac_dy[i][col].is_zero() {
                                    let mut args = Vec::with_capacity(1 + n + k);
                                    args.push(x[j]);
                                    for row in 0..n {
                                        let y_val = *y.get(row, j);
                                        let bounded_y = Self::handle_bounds(
                                            row,
                                            y_val,
                                            bounds.clone(),
                                            variable_str.clone(),
                                        );
                                        args.push(bounded_y);
                                    }
                                    for param in 0..k {
                                        args.push(p[param]);
                                    }

                                    let mut all_var_names = vec![arg.as_str()];
                                    all_var_names.extend(variable_str.iter().map(|s| s.as_str()));
                                    all_var_names.extend(param_str.iter().map(|s| s.as_str()));

                                    let value = jac_dy[i][col]
                                        .eval_expression(all_var_names.as_slice(), &args);
                                    if value.abs() > 1e-15 {
                                        triplets_mutex
                                            .lock()
                                            .unwrap()
                                            .push(Triplet::new(i, col, value));
                                    }
                                }
                            }
                        });

                        SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap()
                    })
                    .collect();

                let df_dp = if let Some(jac_dp_vec) = &jac_dp {
                    Some(
                        (0..m)
                            .into_par_iter()
                            .map(|j| {
                                let mut triplets = Vec::new();

                                for i in 0..n {
                                    for param_idx in 0..k {
                                        if !jac_dp_vec[i][param_idx].is_zero() {
                                            let mut args = Vec::with_capacity(1 + n + k);
                                            args.push(x[j]);
                                            for row in 0..n {
                                                let y_val = *y.get(row, j);
                                                let bounded_y = Self::handle_bounds(
                                                    row,
                                                    y_val,
                                                    bounds.clone(),
                                                    variable_str.clone(),
                                                );
                                                args.push(bounded_y);
                                            }
                                            for param in 0..k {
                                                args.push(p[param]);
                                            }

                                            let mut all_var_names = vec![arg.as_str()];
                                            all_var_names
                                                .extend(variable_str.iter().map(|s| s.as_str()));
                                            all_var_names
                                                .extend(param_str.iter().map(|s| s.as_str()));

                                            let value = jac_dp_vec[i][param_idx]
                                                .eval_expression(all_var_names.as_slice(), &args);
                                            if value.abs() > 1e-15 {
                                                triplets.push(Triplet::new(i, param_idx, value));
                                            }
                                        }
                                    }
                                }

                                SparseColMat::try_new_from_triplets(n, k, &triplets).unwrap()
                            })
                            .collect(),
                    )
                } else {
                    None
                };

                (df_dy, df_dp)
            },
        )
    }

    /// Convert symbolic Jacobian to numerical function and store in struct
    ///
    /// # Arguments
    /// * `jac_dp` - Optional parameter derivatives
    /// * `variable_names` - Variable names
    /// * `param_names` - Parameter names
    /// * `arg` - Independent variable name
    pub fn symbolic_to_ode_jacobian(
        &mut self,
        jac_dp: Option<Vec<Vec<Expr>>>,
        variable_names: Vec<String>,
        param_names: Vec<String>,
        arg: String,
        bounds: Option<HashMap<String, Vec<(usize, f64)>>>,
    ) {
        self.jac_function = Some(Self::jacobian_generate_SparseColMat_par_sci(
            self.symbolic_jacobian.clone(),
            jac_dp,
            variable_names,
            param_names,
            arg,
            self.bandwidth,
            bounds,
        ));
    }

    // main function of this module
    // This function essentially sets up all the necessary components for solving a Boundary Value Problem, including discretization,
    // Jacobian calculation, and preparation of numerical evaluation functions. It allows for different methods of handling sparse or dense matrices,
    // making it flexible for various types of problems and computational approaches.
    /// Main function: Generate complete BVP system from symbolic expressions
    ///
    /// This is the primary interface that converts a symbolic ODE system into
    /// numerical functions ready for BVP solving. It performs:
    /// 1. Symbolic Jacobian calculation
    /// 2. Bandwidth optimization
    /// 3. Function closure generation
    /// 4. Performance timing
    ///
    /// # Arguments
    /// * `eq_system` - Vector of symbolic ODE expressions
    /// * `values` - Names of unknown variables
    /// * `param` - Names of parameters
    /// * `arg` - Independent variable name
    /// * `BorderConditions` - Boundary conditions: variable -> [(position, value), ...]
    ///   where position: 0=start, 1=end, value=boundary value
    pub fn generate_BVP(
        &mut self,
        eq_system: Vec<Expr>,
        values: Vec<String>,
        param: Vec<String>,
        arg: String,

        Bounds: Option<HashMap<String, Vec<(usize, f64)>>>,
    ) {
        self.from_vectors(eq_system, values.clone());
        let total_start = Instant::now();
        let mut timer_hash: HashMap<String, f64> = HashMap::new();
        info!(
            "\n \n ____________START OF GENERATE BVP ________________________________________________________________"
        );

        info!("Calculating jacobian");
        let now = Instant::now();
        self.calc_jacobian_parallel_smart(); //calculate the symbolic Jacobian matrix.
        info!("Jacobian calculation time:");
        let elapsed = now.elapsed();
        info!("{:?}", elapsed_time(elapsed));
        timer_hash.insert("symbolic jacobian time".to_string(), elapsed.as_secs_f64());
        let now = Instant::now();
        self.find_bandwidths(); //  determine the bandwidth of the Jacobian matrix.
        info!("Bandwidth calculated:");
        timer_hash.insert(
            "find bandwidth time".to_string(),
            now.elapsed().as_secs_f64(),
        );
        //  println!("symbolic Jacbian created {:?}", &self.symbolic_jacobian);
        let now = Instant::now();

        self.symbolic_to_ode_jacobian(
            Some(self.symbolic_jacobian.clone()),
            self.variable_string.clone(),
            param.clone(),
            arg.clone(),
            Bounds.clone(),
        );
        timer_hash.insert(
            "jacobian lambdify time".to_string(),
            now.elapsed().as_secs_f64(),
        );
        info!("Jacobian lambdified");
        let n = &self.symbolic_jacobian.len();
        for (_i, vec_s) in self.symbolic_jacobian.iter().enumerate() {
            assert_eq!(vec_s.len(), *n, "jacobian not square ");
        }
        // println!("functioon Jacobian created");
        let now = Instant::now();
        let _ = self.symbolic_to_ode_function(arg, values, param, Bounds);
        timer_hash.insert(
            "residual functions lambdify time".to_string(),
            now.elapsed().as_secs_f64(),
        );
        info!("Residuals vector lambdified");
        let total_end = total_start.elapsed().as_secs_f64();

        *timer_hash.get_mut("symbolic jacobian time").unwrap() /= total_end / 100.0;
        *timer_hash.get_mut("jacobian lambdify time").unwrap() /= total_end / 100.0;
        *timer_hash
            .get_mut("residual functions lambdify time")
            .unwrap() /= total_end / 100.0;
        *timer_hash.get_mut("find bandwidth time").unwrap() /= total_end / 100.0;
        timer_hash.insert("total time, sec".to_string(), total_end);

        let mut table = Builder::from(timer_hash.clone()).build();
        table.with(Style::modern_rounded());
        println!("{}", table.to_string());
        //   panic!("END OF GENERATE BVP");
        info!(
            "\n \n ____________END OF GENERATE BVP ________________________________________________________________"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::mat::Mat;

    #[test]
    fn test_symbolic_to_ode_function_simple() {
        // Create symbolic expressions: dy/dx = z, dz/dx = -y
        let eq1 = Expr::Var("z".to_string());
        let eq2 = -Expr::Var("y".to_string());
        let vector_of_functions = vec![eq1, eq2];

        let mut jacobian_instance = Jacobian_sci_faer {
            vector_of_functions,
            vector_of_variables: vec![],
            variable_string: vec![],
            symbolic_jacobian: vec![],
            jac_function: None,
            residual_function: Box::new(|_, _, _| Mat::zeros(0, 0)),
            bandwidth: None,
            variables_for_all_eq: vec![],
        };

        jacobian_instance.symbolic_to_ode_function(
            "x".to_string(),
            vec!["y".to_owned(), "z".to_owned()],
            vec![],
            None,
        );
        let symbolic_fun = jacobian_instance.residual_function;
        // Direct function for comparison
        let direct_fun = |_x: &faer_col, y: &faer_dense_mat, _p: &faer_col| {
            let mut f = faer_dense_mat::zeros(2, y.ncols());
            for j in 0..y.ncols() {
                *f.get_mut(0, j) = *y.get(1, j);
                *f.get_mut(1, j) = -*y.get(0, j);
            }
            f
        };

        // Test data
        let x = faer_col::from_fn(3, |i| i as f64);
        let mut y = faer_dense_mat::zeros(2, 3);
        *y.get_mut(0, 0) = 1.0;
        *y.get_mut(1, 0) = 2.0;
        *y.get_mut(0, 1) = 3.0;
        *y.get_mut(1, 1) = 4.0;
        *y.get_mut(0, 2) = 5.0;
        *y.get_mut(1, 2) = 6.0;
        let p = faer_col::zeros(0);

        let result_symbolic = symbolic_fun(&x, &y, &p);
        let result_direct = direct_fun(&x, &y, &p);

        // Compare results
        for i in 0..2 {
            for j in 0..3 {
                assert!((result_symbolic.get(i, j) - result_direct.get(i, j)).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_symbolic_to_ode_function_with_params() {
        // Create symbolic expressions: dy/dx = a*z, dz/dx = -b*y
        let eq1 = Expr::Var("a".to_string()) * Expr::Var("z".to_string());
        let eq2 = -Expr::Var("b".to_string()) * Expr::Var("y".to_string());
        let vector_of_functions = vec![eq1, eq2];

        let mut jacobian_instance = Jacobian_sci_faer {
            vector_of_functions,
            vector_of_variables: vec![],
            variable_string: vec![],
            symbolic_jacobian: vec![],
            jac_function: None,
            residual_function: Box::new(|_, _, _| Mat::zeros(0, 0)),
            bandwidth: None,
            variables_for_all_eq: vec![],
        };

        jacobian_instance.symbolic_to_ode_function(
            "x".to_string(),
            vec!["y".to_owned(), "z".to_owned()],
            vec!["a".to_owned(), "b".to_owned()],
            None,
        );
        let symbolic_fun = jacobian_instance.residual_function;
        // Test data
        let x = faer_col::from_fn(2, |i| i as f64);
        let mut y = faer_dense_mat::zeros(2, 2);
        *y.get_mut(0, 0) = 1.0;
        *y.get_mut(1, 0) = 2.0;
        *y.get_mut(0, 1) = 3.0;
        *y.get_mut(1, 1) = 4.0;
        let p = faer_col::from_fn(2, |i| (i + 1) as f64); // a=1, b=2

        let result = symbolic_fun(&x, &y, &p);

        // Expected: f[0] = a*z, f[1] = -b*y
        assert!((result.get(0, 0) - 2.0).abs() < 1e-10); // 1*2
        assert!((result.get(1, 0) - (-2.0)).abs() < 1e-10); // -2*1
        assert!((result.get(0, 1) - 4.0).abs() < 1e-10); // 1*4
        assert!((result.get(1, 1) - (-6.0)).abs() < 1e-10); // -2*3
    }

    #[test]
    fn test_symbolic_to_ode_jacobian_simple() {
        // System: dy/dx = z, dz/dx = -y
        // Jacobian df/dy = [[0, 1], [-1, 0]]
        let mut jacobian_instance = Jacobian_sci_faer {
            vector_of_functions: vec![Expr::Var("z".to_string()), -Expr::Var("y".to_string())],
            vector_of_variables: vec![Expr::Var("y".to_string()), Expr::Var("z".to_string())],
            variable_string: vec!["y".to_string(), "z".to_string()],
            symbolic_jacobian: vec![
                vec![Expr::Const(0.0), Expr::Const(1.0)],
                vec![Expr::Const(-1.0), Expr::Const(0.0)],
            ],
            jac_function: None,
            residual_function: Box::new(|_, _, _| Mat::zeros(0, 0)),
            bandwidth: None,
            variables_for_all_eq: vec![],
        };

        jacobian_instance.symbolic_to_ode_jacobian(
            None,
            vec!["y".to_string(), "z".to_string()],
            vec![],
            "x".to_string(),
            None,
        );

        // Direct jacobian for comparison
        let direct_jac = |_x: &faer_col,
                          _y: &faer_dense_mat,
                          _p: &faer_col|
         -> (Vec<faer_mat>, Option<Vec<faer_mat>>) {
            let jac_matrices = (0.._y.ncols())
                .map(|_| {
                    let triplets = vec![Triplet::new(0, 1, 1.0), Triplet::new(1, 0, -1.0)];
                    SparseColMat::try_new_from_triplets(2, 2, &triplets).unwrap()
                })
                .collect();
            (jac_matrices, None)
        };

        // Test data
        let x = faer_col::from_fn(2, |i| i as f64);
        let mut y = faer_dense_mat::zeros(2, 2);
        *y.get_mut(0, 0) = 1.0;
        *y.get_mut(1, 0) = 2.0;
        *y.get_mut(0, 1) = 3.0;
        *y.get_mut(1, 1) = 4.0;
        let p = faer_col::zeros(0);
        let jac_from_sym = jacobian_instance.jac_function.as_ref().unwrap();
        let (result_symbolic, _) = jac_from_sym(&x, &y, &p);
        let (result_direct, _) = direct_jac(&x, &y, &p);

        // Compare jacobians
        for j in 0..2 {
            assert!((result_symbolic[j].get(0, 1).unwrap() - 1.0).abs() < 1e-10);
            assert!((result_symbolic[j].get(1, 0).unwrap() - (-1.0)).abs() < 1e-10);
            assert_eq!(result_symbolic[j].get(0, 0), result_direct[j].get(0, 0));
            assert_eq!(result_symbolic[j].get(1, 1), result_direct[j].get(1, 1));
        }
    }

    #[test]
    fn test_symbolic_to_ode_jacobian_with_params() {
        // System: dy/dx = a*z, dz/dx = -b*y
        // df/dy = [[0, a], [-b, 0]], df/dp = [[z, 0], [0, -y]]
        let mut jacobian_instance = Jacobian_sci_faer {
            vector_of_functions: vec![
                Expr::Var("a".to_string()) * Expr::Var("z".to_string()),
                -Expr::Var("b".to_string()) * Expr::Var("y".to_string()),
            ],
            vector_of_variables: vec![Expr::Var("y".to_string()), Expr::Var("z".to_string())],
            variable_string: vec!["y".to_string(), "z".to_string()],
            symbolic_jacobian: vec![
                vec![Expr::Const(0.0), Expr::Var("a".to_string())],
                vec![-Expr::Var("b".to_string()), Expr::Const(0.0)],
            ],
            jac_function: None,
            residual_function: Box::new(|_, _, _| Mat::zeros(0, 0)),
            bandwidth: None,
            variables_for_all_eq: vec![],
        };

        let jac_dp = Some(vec![
            vec![Expr::Var("z".to_string()), Expr::Const(0.0)],
            vec![Expr::Const(0.0), -Expr::Var("y".to_string())],
        ]);

        jacobian_instance.symbolic_to_ode_jacobian(
            jac_dp,
            vec!["y".to_string(), "z".to_string()],
            vec!["a".to_string(), "b".to_string()],
            "x".to_string(),
            None,
        );

        // Test data
        let x = faer_col::from_fn(1, |i| i as f64);
        let mut y = faer_dense_mat::zeros(2, 1);
        *y.get_mut(0, 0) = 2.0;
        *y.get_mut(1, 0) = 3.0;
        let p = faer_col::from_fn(2, |i| (i + 1) as f64); // a=1, b=2
        let jac_from_sym = jacobian_instance.jac_function.as_ref().unwrap();
        let (df_dy, df_dp) = jac_from_sym(&x, &y, &p);

        // Check df/dy: [[0, a], [-b, 0]] = [[0, 1], [-2, 0]]
        assert!((df_dy[0].get(0, 1).unwrap() - 1.0).abs() < 1e-10);
        assert!((df_dy[0].get(1, 0).unwrap() - (-2.0)).abs() < 1e-10);

        // Check df/dp: [[z, 0], [0, -y]] = [[3, 0], [0, -2]]
        let df_dp_unwrap = df_dp.unwrap();
        assert!((df_dp_unwrap[0].get(0, 0).unwrap() - 3.0).abs() < 1e-10);
        assert!((df_dp_unwrap[0].get(1, 1).unwrap() - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_generate_bvp_simple_system() {
        let eq1 = Expr::Var("z".to_string());
        let eq2 = -Expr::Var("y".to_string());
        let eq_system = vec![eq1, eq2];
        let values = vec!["y".to_string(), "z".to_string()];
        let params = vec![];
        let arg = "x".to_string();

        let mut jacobian_instance = Jacobian_sci_faer::new();

        jacobian_instance.generate_BVP(eq_system, values, params, arg, None);

        // Test residual function
        let x = faer_col::from_fn(2, |i| i as f64);
        let mut y = faer_dense_mat::zeros(2, 2);
        *y.get_mut(0, 0) = 1.0;
        *y.get_mut(1, 0) = 2.0;
        *y.get_mut(0, 1) = 3.0;
        *y.get_mut(1, 1) = 4.0;
        let p = faer_col::zeros(0);

        let result = (jacobian_instance.residual_function)(&x, &y, &p);
        assert!((result.get(0, 0) - 2.0).abs() < 1e-10);
        assert!((result.get(1, 0) - (-1.0)).abs() < 1e-10);
        assert!((result.get(0, 1) - 4.0).abs() < 1e-10);
        assert!((result.get(1, 1) - (-3.0)).abs() < 1e-10);

        // Test jacobian structure
        assert_eq!(jacobian_instance.symbolic_jacobian.len(), 2);
        assert_eq!(jacobian_instance.symbolic_jacobian[0].len(), 2);
        assert!(jacobian_instance.jac_function.is_some());
    }

    #[test]
    fn test_generate_bvp_with_parameters() {
        let eq1 = Expr::Var("a".to_string()) * Expr::Var("z".to_string());
        let eq2 = -Expr::Var("b".to_string()) * Expr::Var("y".to_string());
        let eq_system = vec![eq1, eq2];
        let values = vec!["y".to_string(), "z".to_string()];
        let params = vec!["a".to_string(), "b".to_string()];
        let arg = "x".to_string();

        let mut jacobian_instance = Jacobian_sci_faer::new();

        jacobian_instance.generate_BVP(eq_system, values, params, arg, None);

        // Test residual function with parameters
        let x = faer_col::from_fn(1, |i| i as f64);
        let mut y = faer_dense_mat::zeros(2, 1);
        *y.get_mut(0, 0) = 2.0;
        *y.get_mut(1, 0) = 3.0;
        let p = faer_col::from_fn(2, |i| (i + 1) as f64); // a=1, b=2

        let result = (jacobian_instance.residual_function)(&x, &y, &p);
        assert!((result.get(0, 0) - 3.0).abs() < 1e-10); // a*z = 1*3
        assert!((result.get(1, 0) - (-4.0)).abs() < 1e-10); // -b*y = -2*2
    }

    #[test]
    fn test_generate_bvp_nonlinear_system() {
        let eq1 = Expr::Var("y".to_string()).pow(Expr::Const(2.0)) + Expr::Var("z".to_string());
        let eq2 = Expr::Var("y".to_string()) * Expr::Var("z".to_string());
        let eq_system = vec![eq1, eq2];
        let values = vec!["y".to_string(), "z".to_string()];
        let params = vec![];
        let arg = "x".to_string();

        let mut jacobian_instance = Jacobian_sci_faer::new();

        jacobian_instance.generate_BVP(eq_system, values, params, arg, None);

        // Test nonlinear residual
        let x = faer_col::from_fn(1, |i| i as f64);
        let mut y = faer_dense_mat::zeros(2, 1);
        *y.get_mut(0, 0) = 2.0;
        *y.get_mut(1, 0) = 3.0;
        let p = faer_col::zeros(0);

        let result = (jacobian_instance.residual_function)(&x, &y, &p);
        assert!((result.get(0, 0) - 7.0).abs() < 1e-10); // y^2 + z = 4 + 3
        assert!((result.get(1, 0) - 6.0).abs() < 1e-10); // y*z = 2*3

        // Check jacobian was computed
        assert!(!jacobian_instance.symbolic_jacobian.is_empty());
        assert!(jacobian_instance.bandwidth.is_some());
    }

    #[test]
    fn test_generate_bvp_bandwidth_calculation() {
        let eq1 = Expr::Var("y".to_string());
        let eq2 = Expr::Var("z".to_string());
        let eq_system = vec![eq1, eq2];
        let values = vec!["y".to_string(), "z".to_string()];
        let params = vec![];
        let arg = "x".to_string();

        let mut jacobian_instance = Jacobian_sci_faer::new();

        jacobian_instance.generate_BVP(eq_system, values, params, arg, None);

        // Check bandwidth was calculated
        assert!(jacobian_instance.bandwidth.is_some());
        let (kl, ku) = jacobian_instance.bandwidth.unwrap();
        assert_eq!(kl, 0); // No subdiagonals for diagonal jacobian
        assert_eq!(ku, 0); // No superdiagonals for diagonal jacobian
    }

    #[test]
    fn test_handle_bounds_minimum() {
        let bounds = Some(HashMap::from([
            ("y".to_string(), vec![(0, 1e-10)]), // minimum bound
        ]));
        let var_names = vec!["y".to_string(), "z".to_string()];

        // Test minimum bound enforcement
        let result = Jacobian_sci_faer::handle_bounds(0, -1.0, bounds.clone(), var_names.clone());
        assert_eq!(result, 1e-10);

        // Test value above minimum
        let result = Jacobian_sci_faer::handle_bounds(0, 5.0, bounds, var_names);
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_handle_bounds_maximum() {
        let bounds = Some(HashMap::from([
            ("z".to_string(), vec![(1, 100.0)]), // maximum bound
        ]));
        let var_names = vec!["y".to_string(), "z".to_string()];

        // Test maximum bound enforcement
        let result = Jacobian_sci_faer::handle_bounds(1, 200.0, bounds.clone(), var_names.clone());
        assert_eq!(result, 100.0);

        // Test value below maximum
        let result = Jacobian_sci_faer::handle_bounds(1, 50.0, bounds, var_names);
        assert_eq!(result, 50.0);
    }

    #[test]
    fn test_handle_bounds_both_min_max() {
        let bounds = Some(HashMap::from([
            ("y".to_string(), vec![(0, 1e-10), (1, 10.0)]), // both min and max
        ]));
        let var_names = vec!["y".to_string()];

        // Test minimum enforcement
        let result = Jacobian_sci_faer::handle_bounds(0, -5.0, bounds.clone(), var_names.clone());
        assert_eq!(result, 1e-10);

        // Test maximum enforcement
        let result = Jacobian_sci_faer::handle_bounds(0, 15.0, bounds.clone(), var_names.clone());
        assert_eq!(result, 10.0);

        // Test value within bounds
        let result = Jacobian_sci_faer::handle_bounds(0, 5.0, bounds, var_names);
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_generate_bvp_with_bounds() {
        // System with logarithm that needs bounds: dy/dx = log(y), dz/dx = z
        let eq1 = Expr::ln(Expr::Var("y".to_string()));
        let eq2 = Expr::Var("z".to_string());
        let eq_system = vec![eq1, eq2];
        let values = vec!["y".to_string(), "z".to_string()];
        let params = vec![];
        let arg = "x".to_string();

        // Set bounds to prevent log(0)
        let bounds = Some(HashMap::from([
            ("y".to_string(), vec![(0, 1e-10)]), // minimum bound for y
        ]));

        let mut jacobian_instance = Jacobian_sci_faer::new();
        jacobian_instance.generate_BVP(eq_system, values, params, arg, bounds);

        // Test with y=0 (should be bounded to 1e-10)
        let x = faer_col::from_fn(1, |i| i as f64);
        let mut y = faer_dense_mat::zeros(2, 1);
        *y.get_mut(0, 0) = 0.0; // This should trigger the bound
        *y.get_mut(1, 0) = 2.0;
        let p = faer_col::zeros(0);

        let result = (jacobian_instance.residual_function)(&x, &y, &p);
        println!("Result: {:?}", result);
        // log(1e-10) ≈ -23.03, z = 2.0
        assert!(*result.get(0, 0) < -20.0); // Should be log(1e-10), not log(0)
        assert!((*result.get(1, 0) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_generate_bvp_with_multiple_variable_bounds() {
        // System: dy/dx = 1/y, dz/dx = sqrt(z)
        let eq1 = Expr::Const(1.0) / Expr::Var("y".to_string());
        let eq2 = Expr::Pow(
            Box::new(Expr::Var("z".to_string())),
            Box::new(Expr::Const(0.5)),
        );
        let eq_system = vec![eq1, eq2];
        let values = vec!["y".to_string(), "z".to_string()];
        let params = vec![];
        let arg = "x".to_string();

        // Set bounds for both variables
        let bounds = Some(HashMap::from([
            ("y".to_string(), vec![(0, 1e-10)]), // minimum bound for y (avoid division by zero)
            ("z".to_string(), vec![(0, 0.0)]),   // minimum bound for z (avoid sqrt of negative)
        ]));

        let mut jacobian_instance = Jacobian_sci_faer::new();
        jacobian_instance.generate_BVP(eq_system, values, params, arg, bounds);

        // Test with problematic values
        let x = faer_col::from_fn(1, |i| i as f64);
        let mut y = faer_dense_mat::zeros(2, 1);
        *y.get_mut(0, 0) = -1.0; // Should be bounded to 1e-10
        *y.get_mut(1, 0) = -4.0; // Should be bounded to 0.0
        let p = faer_col::zeros(0);

        let result = (jacobian_instance.residual_function)(&x, &y, &p);
        println!("Result: {:?}", result);
        // 1/(1e-10) = 1e10, sqrt(0) = 0
        assert!(*result.get(0, 0) > 1e9); // Should be 1/(1e-10)
        assert!((*result.get(1, 0) - 0.0).abs() < 1e-10); // Should be sqrt(0)
    }

    #[test]
    fn test_generate_bvp_bounds_with_parameters() {
        // System with parameters and bounds: dy/dx = a*log(y), dz/dx = b*z
        let eq1 = Expr::Var("a".to_string()) * Expr::ln(Expr::Var("y".to_string()));
        let eq2 = Expr::Var("b".to_string()) * Expr::Var("z".to_string());
        let eq_system = vec![eq1, eq2];
        let values = vec!["y".to_string(), "z".to_string()];
        let params = vec!["a".to_string(), "b".to_string()];
        let arg = "x".to_string();

        let bounds = Some(HashMap::from([
            ("y".to_string(), vec![(0, 1e-8)]), // minimum bound for y
        ]));

        let mut jacobian_instance = Jacobian_sci_faer::new();
        jacobian_instance.generate_BVP(eq_system, values, params, arg, bounds);

        // Test with parameters
        let x = faer_col::from_fn(1, |i| i as f64);
        let mut y = faer_dense_mat::zeros(2, 1);
        *y.get_mut(0, 0) = 0.0; // Should be bounded
        *y.get_mut(1, 0) = 3.0;
        let p = faer_col::from_fn(2, |i| (i + 1) as f64 * 2.0); // a=2, b=4

        let result = (jacobian_instance.residual_function)(&x, &y, &p);
        println!("Result: {:?}", result);
        // a*log(1e-8) = 2*(-18.42) ≈ -36.84, b*z = 4*3 = 12
        assert!(*result.get(0, 0) < -30.0);
        assert!((*result.get(1, 0) - 12.0).abs() < 1e-10);
    }
}
