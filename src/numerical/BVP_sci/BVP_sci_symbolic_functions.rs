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
    faer_col, faer_dense_mat, faer_mat, ODEFunction, ODEJacobian,
};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_functions_BVP::{
    bvp_symbolic_bandwidth, bvp_symbolic_jacobian_smart, bvp_symbolic_parse_variables,
    bvp_symbolic_variables_for_functions,
};
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
    /// Sparse symbolic Jacobian entries (row, col, expr) for nonzero df/dy
    pub symbolic_jacobian_sparse: Vec<(usize, usize, Expr)>,
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
            symbolic_jacobian_sparse: vec![],
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
        self.vector_of_functions = vector_of_functions;
        self.variable_string = variable_string;
        self.vector_of_variables = bvp_symbolic_parse_variables(&self.variable_string);
        self.variables_for_all_eq = bvp_symbolic_variables_for_functions(&self.vector_of_functions);
        info!(
            "Initialized BVP symbolic functions for {} equations",
            self.vector_of_functions.len()
        );
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

        let (symbolic_jacobian, symbolic_sparse) = bvp_symbolic_jacobian_smart(
            &self.vector_of_functions,
            self.vector_of_variables.len(),
            &self.variable_string,
            &self.variables_for_all_eq,
        );
        self.symbolic_jacobian = symbolic_jacobian;
        self.symbolic_jacobian_sparse = symbolic_sparse;
    }
    /// Determine Jacobian matrix bandwidth for sparse storage optimization
    ///
    /// Calculates the number of sub/super-diagonals (kl, ku) in the Jacobian
    /// to optimize sparse matrix operations.
    pub(crate) fn find_bandwidths(&mut self) {
        self.bandwidth = Some(bvp_symbolic_bandwidth(&self.symbolic_jacobian));
    }
    fn compile_residual_function(
        &self,
        arg: String,
        var_names: Vec<String>,
        par_names: Vec<String>,
        bounds: Option<HashMap<String, Vec<(usize, f64)>>>,
    ) -> Box<ODEFunction> {
        let mut all_var_names = vec![arg.as_str()];
        all_var_names.extend(var_names.iter().map(|s| s.as_str()));
        all_var_names.extend(par_names.iter().map(|s| s.as_str()));
        let compiled_functions: Vec<_> = self
            .vector_of_functions
            .iter()
            .map(|func| Expr::lambdify_borrowed_thread_safe(func, all_var_names.as_slice()))
            .collect();
        Box::new(
            move |x: &faer_col, y: &faer_dense_mat, p: &faer_col| -> faer_dense_mat {
                let (n, m) = (y.nrows(), y.ncols());
                let mut result = faer_dense_mat::zeros(n, m);
                let bounds_ref = bounds.as_ref();
                let var_names_ref = var_names.as_slice();

                let columns: Vec<Vec<f64>> = (0..m) // m is the number of time steps
                    .into_par_iter()
                    .map(|j| {
                        let mut args = Vec::with_capacity(1 + n + p.nrows()); // n is the number of variables and p is the number of parameters
                        args.push(x[j]);
                        for i in 0..n {
                            let y = *y.get(i, j);
                            let bounded_value =
                                Self::handle_bounds(i, y, bounds_ref, var_names_ref);
                            args.push(bounded_value);
                        }
                        for i in 0..p.nrows() {
                            args.push(p[i]);
                        }

                        compiled_functions
                            .iter()
                            .map(|func| func(args.as_slice()))
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
        )
    }

    /// Convert symbolic expressions to numerical residual function closure
    ///
    /// Creates a closure that evaluates the ODE system at mesh points.
    ///
    /// # Arguments
    /// * `arg` - Independent variable name (e.g., "x")
    /// * `var_names` - Unknown variable names
    /// * `par_names` - Parameter names
    pub fn symbolic_to_ode_function(
        &mut self,
        arg: String,
        var_names: Vec<String>,
        par_names: Vec<String>,
        Bounds: Option<HashMap<String, Vec<(usize, f64)>>>,
    ) {
        self.residual_function = self.compile_residual_function(arg, var_names, par_names, Bounds);
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

    pub fn handle_bounds(
        var_index: usize,
        value: f64,
        bounds: Option<&HashMap<String, Vec<(usize, f64)>>>,
        var_names: &[String],
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
                    log::warn!(
                        "Warning: Unknown bound type {} for variable {}",
                        bound_type,
                        var_name
                    );
                    bounded_value
                }
            };
        }
        bounded_value
    }

    fn bounded_runtime_args_for_column(
        x: &faer_col,
        y: &faer_dense_mat,
        p: &faer_col,
        column: usize,
        variable_names: &[String],
        bounds: &Option<HashMap<String, Vec<(usize, f64)>>>,
    ) -> Vec<f64> {
        let n = y.nrows();
        let k = p.nrows();
        let mut args = Vec::with_capacity(1 + n + k);
        args.push(x[column]);
        let bounds_ref = bounds.as_ref();
        for row in 0..n {
            let y_val = *y.get(row, column);
            let bounded_y = Self::handle_bounds(row, y_val, bounds_ref, variable_names);
            args.push(bounded_y);
        }
        for param in 0..k {
            args.push(p[param]);
        }
        args
    }

    fn compile_sparse_entry_evaluators(
        jacobian: &[Vec<Expr>],
        row_range_for_bandwidth: Option<&[(usize, usize)]>,
        input_names: &[String],
    ) -> Vec<Vec<(usize, Box<dyn Fn(&[f64]) -> f64 + Send + Sync>)>> {
        let input_name_refs: Vec<&str> = input_names.iter().map(|s| s.as_str()).collect();
        jacobian
            .iter()
            .enumerate()
            .map(|(row_idx, row)| {
                let (left_border, right_border) = row_range_for_bandwidth
                    .map(|ranges| ranges[row_idx])
                    .unwrap_or((0, row.len()));
                (left_border..right_border)
                    .filter_map(|col_idx| {
                        let expr = &row[col_idx];
                        if expr.is_zero() {
                            None
                        } else {
                            Some((
                                col_idx,
                                Expr::lambdify_borrowed_thread_safe(
                                    expr,
                                    input_name_refs.as_slice(),
                                ),
                            ))
                        }
                    })
                    .collect()
            })
            .collect()
    }

    fn compile_sparse_triplet_evaluators(
        entries: &[(usize, usize, Expr)],
        input_names: &[String],
    ) -> Vec<(usize, usize, Box<dyn Fn(&[f64]) -> f64 + Send + Sync>)> {
        let input_name_refs: Vec<&str> = input_names.iter().map(|s| s.as_str()).collect();
        entries
            .iter()
            .map(|(row, col, expr)| {
                (
                    *row,
                    *col,
                    Expr::lambdify_borrowed_thread_safe(expr, input_name_refs.as_slice()),
                )
            })
            .collect()
    }

    fn sparse_entries_from_dense(symbolic_matrix: &[Vec<Expr>]) -> Vec<(usize, usize, Expr)> {
        symbolic_matrix
            .iter()
            .enumerate()
            .flat_map(|(row, dense_row)| {
                dense_row.iter().enumerate().filter_map(move |(col, expr)| {
                    if expr.is_zero() {
                        None
                    } else {
                        Some((row, col, expr.clone()))
                    }
                })
            })
            .collect()
    }

    fn build_symbolic_param_jacobian_sparse(
        equations: &[Expr],
        param_names: &[String],
    ) -> Option<Vec<(usize, usize, Expr)>> {
        if param_names.is_empty() {
            return None;
        }

        Some(
            equations
                .iter()
                .enumerate()
                .flat_map(|(row, expr)| {
                    param_names
                        .iter()
                        .enumerate()
                        .filter_map(move |(col, parameter)| {
                            let partial = expr.diff(parameter).simplify();
                            if partial.is_zero() {
                                None
                            } else {
                                Some((row, col, partial))
                            }
                        })
                })
                .collect(),
        )
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
        let mut all_var_names = vec![arg];
        all_var_names.extend(variable_str.iter().cloned());
        all_var_names.extend(param_str.iter().cloned());
        let row_ranges_for_bandwidth: Option<Vec<(usize, usize)>> = bandwidth.map(|(kl, ku)| {
            let n = jac_dy.len();
            (0..n)
                .map(|i| {
                    let right_border = std::cmp::min(i + ku + 1, n);
                    let left_border = if i == 0 { 0 } else { i.saturating_sub(kl + 1) };
                    (left_border, right_border)
                })
                .collect()
        });
        let compiled_jac_dy = Self::compile_sparse_entry_evaluators(
            &jac_dy,
            row_ranges_for_bandwidth.as_deref(),
            all_var_names.as_slice(),
        );
        let compiled_jac_dp = jac_dp.as_ref().map(|jac_dp_vec| {
            Self::compile_sparse_entry_evaluators(jac_dp_vec, None, all_var_names.as_slice())
        });
        let nnz_dy_capacity: usize = compiled_jac_dy.iter().map(Vec::len).sum();
        let nnz_dp_capacity: usize = compiled_jac_dp
            .as_ref()
            .map(|rows| rows.iter().map(Vec::len).sum())
            .unwrap_or(0);

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
                        let args = Self::bounded_runtime_args_for_column(
                            x,
                            y,
                            p,
                            j,
                            variable_str.as_slice(),
                            &bounds,
                        );
                        let mut triplets = Vec::with_capacity(nnz_dy_capacity);
                        for (i, row_entries) in compiled_jac_dy.iter().enumerate() {
                            for (col_idx, evaluator) in row_entries {
                                let value = evaluator(args.as_slice());
                                if value.abs() > 1e-15 {
                                    triplets.push(Triplet::new(i, *col_idx, value));
                                }
                            }
                        }

                        SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap()
                    })
                    .collect();

                let df_dp = if let Some(compiled_jac_dp_rows) = &compiled_jac_dp {
                    Some(
                        (0..m)
                            .into_par_iter()
                            .map(|j| {
                                let args = Self::bounded_runtime_args_for_column(
                                    x,
                                    y,
                                    p,
                                    j,
                                    variable_str.as_slice(),
                                    &bounds,
                                );
                                let mut triplets = Vec::with_capacity(nnz_dp_capacity);

                                for (i, row_entries) in compiled_jac_dp_rows.iter().enumerate() {
                                    for (param_idx, evaluator) in row_entries {
                                        let value = evaluator(args.as_slice());
                                        if value.abs() > 1e-15 {
                                            triplets.push(Triplet::new(i, *param_idx, value));
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

    pub fn jacobian_generate_SparseColMat_par_sci_from_sparse_entries(
        jac_dy_sparse: Vec<(usize, usize, Expr)>,
        jac_dp_sparse: Option<Vec<(usize, usize, Expr)>>,
        variable_str: Vec<String>,
        param_str: Vec<String>,
        arg: String,
        bounds: Option<HashMap<String, Vec<(usize, f64)>>>,
    ) -> Box<dyn Fn(&faer_col, &faer_dense_mat, &faer_col) -> (Vec<faer_mat>, Option<Vec<faer_mat>>)>
    {
        let mut all_var_names = vec![arg];
        all_var_names.extend(variable_str.iter().cloned());
        all_var_names.extend(param_str.iter().cloned());
        let compiled_jac_dy = Self::compile_sparse_triplet_evaluators(
            jac_dy_sparse.as_slice(),
            all_var_names.as_slice(),
        );
        let compiled_jac_dp = jac_dp_sparse.as_ref().map(|entries| {
            Self::compile_sparse_triplet_evaluators(entries.as_slice(), all_var_names.as_slice())
        });
        let nnz_dp_capacity = compiled_jac_dp
            .as_ref()
            .map(|entries| entries.len())
            .unwrap_or(0);

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
                        let args = Self::bounded_runtime_args_for_column(
                            x,
                            y,
                            p,
                            j,
                            variable_str.as_slice(),
                            &bounds,
                        );
                        let mut triplets = Vec::with_capacity(compiled_jac_dy.len());
                        for (row, col, evaluator) in &compiled_jac_dy {
                            let value = evaluator(args.as_slice());
                            if value.abs() > 1e-15 {
                                triplets.push(Triplet::new(*row, *col, value));
                            }
                        }
                        SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap()
                    })
                    .collect();

                let df_dp = compiled_jac_dp.as_ref().map(|compiled_param_entries| {
                    (0..m)
                        .into_par_iter()
                        .map(|j| {
                            let args = Self::bounded_runtime_args_for_column(
                                x,
                                y,
                                p,
                                j,
                                variable_str.as_slice(),
                                &bounds,
                            );
                            let mut triplets = Vec::with_capacity(nnz_dp_capacity);
                            for (row, col, evaluator) in compiled_param_entries {
                                let value = evaluator(args.as_slice());
                                if value.abs() > 1e-15 {
                                    triplets.push(Triplet::new(*row, *col, value));
                                }
                            }
                            SparseColMat::try_new_from_triplets(n, k, &triplets).unwrap()
                        })
                        .collect()
                });

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
        let jac_dp_sparse = jac_dp
            .as_ref()
            .map(|dense_param_jacobian| Self::sparse_entries_from_dense(dense_param_jacobian));
        self.symbolic_to_ode_jacobian_from_sparse_entries(
            jac_dp_sparse,
            variable_names,
            param_names,
            arg,
            bounds,
        );
    }

    pub fn symbolic_to_ode_jacobian_from_sparse_entries(
        &mut self,
        jac_dp_sparse: Option<Vec<(usize, usize, Expr)>>,
        variable_names: Vec<String>,
        param_names: Vec<String>,
        arg: String,
        bounds: Option<HashMap<String, Vec<(usize, f64)>>>,
    ) {
        self.jac_function = Some(
            Self::jacobian_generate_SparseColMat_par_sci_from_sparse_entries(
                self.symbolic_jacobian_sparse.clone(),
                jac_dp_sparse,
                variable_names,
                param_names,
                arg,
                bounds,
            ),
        );
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

        let symbolic_param_jacobian_sparse =
            Self::build_symbolic_param_jacobian_sparse(&self.vector_of_functions, &param);
        self.symbolic_to_ode_jacobian_from_sparse_entries(
            symbolic_param_jacobian_sparse,
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
        info!("{}", table.to_string());
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
            symbolic_jacobian_sparse: vec![],
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
            symbolic_jacobian_sparse: vec![],
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
            symbolic_jacobian_sparse: vec![(0, 1, Expr::Const(1.0)), (1, 0, Expr::Const(-1.0))],
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
            symbolic_jacobian_sparse: vec![
                (0, 1, Expr::Var("a".to_string())),
                (1, 0, -Expr::Var("b".to_string())),
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
    fn test_sparse_entry_jacobian_builder_matches_dense_builder() {
        let jac_dy = vec![
            vec![Expr::Const(0.0), Expr::Var("a".to_string())],
            vec![-Expr::Var("b".to_string()), Expr::Const(0.0)],
        ];
        let jac_dp_dense = Some(vec![
            vec![Expr::Var("z".to_string()), Expr::Const(0.0)],
            vec![Expr::Const(0.0), -Expr::Var("y".to_string())],
        ]);
        let jac_dy_sparse = vec![
            (0, 1, Expr::Var("a".to_string())),
            (1, 0, -Expr::Var("b".to_string())),
        ];
        let jac_dp_sparse = Some(vec![
            (0, 0, Expr::Var("z".to_string())),
            (1, 1, -Expr::Var("y".to_string())),
        ]);

        let dense_builder = Jacobian_sci_faer::jacobian_generate_SparseColMat_par_sci(
            jac_dy,
            jac_dp_dense,
            vec!["y".to_string(), "z".to_string()],
            vec!["a".to_string(), "b".to_string()],
            "x".to_string(),
            None,
            None,
        );
        let sparse_builder =
            Jacobian_sci_faer::jacobian_generate_SparseColMat_par_sci_from_sparse_entries(
                jac_dy_sparse,
                jac_dp_sparse,
                vec!["y".to_string(), "z".to_string()],
                vec!["a".to_string(), "b".to_string()],
                "x".to_string(),
                None,
            );

        let x = faer_col::from_fn(1, |_| 0.0);
        let mut y = faer_dense_mat::zeros(2, 1);
        *y.get_mut(0, 0) = 2.0;
        *y.get_mut(1, 0) = 3.0;
        let p = faer_col::from_fn(2, |i| if i == 0 { 1.5 } else { 2.5 });

        let (dense_dy, dense_dp) = dense_builder(&x, &y, &p);
        let (sparse_dy, sparse_dp) = sparse_builder(&x, &y, &p);

        assert!((dense_dy[0].get(0, 1).unwrap() - sparse_dy[0].get(0, 1).unwrap()).abs() < 1e-12);
        assert!((dense_dy[0].get(1, 0).unwrap() - sparse_dy[0].get(1, 0).unwrap()).abs() < 1e-12);

        let dense_dp = dense_dp.expect("dense builder should produce df/dp");
        let sparse_dp = sparse_dp.expect("sparse builder should produce df/dp");
        assert!((dense_dp[0].get(0, 0).unwrap() - sparse_dp[0].get(0, 0).unwrap()).abs() < 1e-12);
        assert!((dense_dp[0].get(1, 1).unwrap() - sparse_dp[0].get(1, 1).unwrap()).abs() < 1e-12);
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
        let result = Jacobian_sci_faer::handle_bounds(0, -1.0, bounds.as_ref(), &var_names);
        assert_eq!(result, 1e-10);

        // Test value above minimum
        let result = Jacobian_sci_faer::handle_bounds(0, 5.0, bounds.as_ref(), &var_names);
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_handle_bounds_maximum() {
        let bounds = Some(HashMap::from([
            ("z".to_string(), vec![(1, 100.0)]), // maximum bound
        ]));
        let var_names = vec!["y".to_string(), "z".to_string()];

        // Test maximum bound enforcement
        let result = Jacobian_sci_faer::handle_bounds(1, 200.0, bounds.as_ref(), &var_names);
        assert_eq!(result, 100.0);

        // Test value below maximum
        let result = Jacobian_sci_faer::handle_bounds(1, 50.0, bounds.as_ref(), &var_names);
        assert_eq!(result, 50.0);
    }

    #[test]
    fn test_handle_bounds_both_min_max() {
        let bounds = Some(HashMap::from([
            ("y".to_string(), vec![(0, 1e-10), (1, 10.0)]), // both min and max
        ]));
        let var_names = vec!["y".to_string()];

        // Test minimum enforcement
        let result = Jacobian_sci_faer::handle_bounds(0, -5.0, bounds.as_ref(), &var_names);
        assert_eq!(result, 1e-10);

        // Test maximum enforcement
        let result = Jacobian_sci_faer::handle_bounds(0, 15.0, bounds.as_ref(), &var_names);
        assert_eq!(result, 10.0);

        // Test value within bounds
        let result = Jacobian_sci_faer::handle_bounds(0, 5.0, bounds.as_ref(), &var_names);
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
