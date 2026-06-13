//!
//! # BVP Symbolic Wrapper Module
//!
//! This module provides a high-level wrapper for solving Boundary Value Problems (BVP)
//! using symbolic expressions. It combines symbolic computation with numerical BVP solving
//! to provide an easy-to-use interface for complex differential equation systems.
//!
//! ## Key Features
//! - **Symbolic to Numerical**: Converts symbolic ODE expressions to efficient numerical functions
//! - **Automatic Jacobian**: Generates analytical Jacobians from symbolic expressions
//! - **Bounds Support**: Applies variable bounds to prevent numerical issues (e.g., log(0))
//! - **Flexible Mesh**: Supports both uniform and custom mesh definitions
//! - **Multiple Solvers**: Integrates with advanced BVP solvers with error control
//! - **Result Analysis**: Provides plotting, saving, and statistical analysis of solutions
//!
//! ## Main Structure: `BVPwrap`
//! The `BVPwrap` struct is the main interface that:
//! 1. Takes symbolic ODE expressions as strings or `Expr` objects
//! 2. Converts them to numerical functions with bounds handling
//! 3. Generates analytical Jacobians for faster convergence
//! 4. Solves the BVP using state-of-the-art numerical methods
//! 5. Provides results in various formats (matrices, plots, files)
//!
//! ## Usage Example
//! ```rust,ignore
//! // Define ODE system: y'' = -y, y(0) = 0, y'(0) = 1
//! let equations = vec!["z".to_string(), "-y".to_string()];
//! let variables = vec!["y".to_string(), "z".to_string()];
//! let boundary_conditions = HashMap::from([
//!     ("y".to_string(), vec![(0, 0.0)]),  // y(0) = 0
//!     ("z".to_string(), vec![(0, 1.0)]),  // z(0) = 1
//! ]);
//!
//! let mut solver = BVPwrap::new(
//!     None, Some(0.0), Some(π), Some(100),
//!     Expr::parse_vector_expression(equations),
//!     variables, vec![], None, boundary_conditions,
//!     "x".to_string(), 1e-6, 1000, initial_guess
//! );
//!
//! solver.solve();
//! solver.plot_result();
//! ```
//!
//! ## Function Overview
//! - `new()`: Creates a new BVP solver instance with mesh and boundary conditions
//! - `set_additional_parameters()`: Configures Jacobian usage and variable bounds
//! - `solve()`: Main solving function with logging support
//! - `eq_generate()`: Converts symbolic expressions to numerical functions
//! - `BC_closure_creater()`: Creates boundary condition functions from HashMap
//! - `plot_result()`, `save_to_file()`: Result visualization and export
//!
//! ## Supported Workflows
//! `BVP_sci` intentionally supports three distinct workflows:
//! 1. `ExprLegacySmartSparseLambdify`
//!    Smart parallel symbolic differentiation with sparse/banded Jacobian metadata,
//!    followed by `lambdify_borrowed_thread_safe(...)` runtime closures.
//! 2. `AtomViewAotSparse`
//!    Symbolic differentiation through the codegen/AOT pipeline with sparse runtime plans
//!    and compiled residual/Jacobian modules.
//! 3. `DirectNumericFaer`
//!    No symbolic machinery at all: users provide ordinary numerical closures directly to
//!    the `faer` solver core in [`BVP_sci_faer`].
//!
use crate::Utils::logger::{save_matrix_to_csv, save_matrix_to_file};
use crate::Utils::plots::{plots, plots_gnulot};
use crate::numerical::BVP_Damp::BVP_utils::{CustomTimer, elapsed_time};
use crate::numerical::BVP_sci::BVP_sci_aot::{
    BvpSciGeneratedBackendConfig, BvpSciGeneratedBackendMode,
};
use crate::numerical::BVP_sci::BVP_sci_faer::{
    BCFunction, BCJacobian, BVPResult, BvpSciLinearSolvePolicy, ODEBandedJacobian, ODEFunction,
    ODEJacobian, faer_col, faer_dense_mat, faer_mat,
    solve_bvp_with_strategy_linear_policy_and_banded_jacobian,
};
use crate::numerical::BVP_sci::BVP_sci_symbolic_functions::Jacobian_sci_faer;
use crate::numerical::BVP_sci::BVP_sci_utils::size_of_jacobian;
use crate::symbolic::codegen::codegen_runtime_api::ResidualChunkingStrategy;
use crate::symbolic::codegen::codegen_tasks::SparseChunkingStrategy;
use crate::symbolic::symbolic_engine::Expr;
use chrono::Local;
use faer::mat::Mat;
use log::{error, info};
use nalgebra::{DMatrix, DVector};
use simplelog::LevelFilter;
use simplelog::*;
use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;
use tabled::{builder::Builder, settings::Style};

#[derive(Clone, Debug)]
pub struct BvpSciSolverOptions {
    pub x_mesh_set: Option<DVector<f64>>,
    pub t0: Option<f64>,
    pub t_end: Option<f64>,
    pub n_steps: Option<usize>,
    pub eq_system: Vec<Expr>,
    pub values: Vec<String>,
    pub param: Vec<String>,
    pub param_values: Option<DVector<f64>>,
    pub boundary_conditions: HashMap<String, Vec<(usize, f64)>>,
    pub arg: String,
    pub tolerance: f64,
    pub max_nodes: usize,
    pub initial_guess: DMatrix<f64>,
    pub use_analytical_jacobian: bool,
    pub bounds: Option<HashMap<String, Vec<(usize, f64)>>>,
    pub loglevel: Option<String>,
    pub generated_backend_config: BvpSciGeneratedBackendConfig,
    pub linear_solve_policy: BvpSciLinearSolvePolicy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BvpSciWorkflow {
    ExprLegacySmartSparseLambdify,
    AtomViewAotSparse,
    AtomViewAotBanded,
    DirectNumericFaer,
}

impl BvpSciSolverOptions {
    pub fn new(
        x_mesh_set: Option<DVector<f64>>,
        t0: Option<f64>,
        t_end: Option<f64>,
        n_steps: Option<usize>,
        eq_system: Vec<Expr>,
        values: Vec<String>,
        param: Vec<String>,
        param_values: Option<DVector<f64>>,
        boundary_conditions: HashMap<String, Vec<(usize, f64)>>,
        arg: String,
        tolerance: f64,
        max_nodes: usize,
        initial_guess: DMatrix<f64>,
    ) -> Self {
        Self {
            x_mesh_set,
            t0,
            t_end,
            n_steps,
            eq_system,
            values,
            param,
            param_values,
            boundary_conditions,
            arg,
            tolerance,
            max_nodes,
            initial_guess,
            use_analytical_jacobian: true,
            bounds: None,
            loglevel: Some("info".to_string()),
            generated_backend_config: BvpSciGeneratedBackendConfig::default(),
            linear_solve_policy: BvpSciLinearSolvePolicy::Sparse,
        }
    }

    pub fn with_analytical_jacobian(mut self, enabled: bool) -> Self {
        self.use_analytical_jacobian = enabled;
        self
    }

    pub fn with_bounds(mut self, bounds: Option<HashMap<String, Vec<(usize, f64)>>>) -> Self {
        self.bounds = bounds;
        self
    }

    pub fn with_loglevel(mut self, loglevel: Option<String>) -> Self {
        self.loglevel = loglevel;
        self
    }

    pub fn with_generated_backend_config(mut self, config: BvpSciGeneratedBackendConfig) -> Self {
        self.generated_backend_config = config;
        self
    }

    pub fn with_sparse_jacobian_chunking_strategy(
        mut self,
        sparse_jacobian_chunking_strategy: SparseChunkingStrategy,
    ) -> Self {
        self.generated_backend_config = self
            .generated_backend_config
            .with_sparse_jacobian_chunking_strategy(sparse_jacobian_chunking_strategy);
        self
    }

    pub fn with_residual_chunking_strategy(
        mut self,
        residual_chunking_strategy: ResidualChunkingStrategy,
    ) -> Self {
        self.generated_backend_config = self
            .generated_backend_config
            .with_residual_chunking_strategy(residual_chunking_strategy);
        self
    }

    pub fn with_linear_solve_policy(mut self, policy: BvpSciLinearSolvePolicy) -> Self {
        self.linear_solve_policy = policy;
        self
    }

    pub fn with_auto_banded_linear_solver(self) -> Self {
        self.with_linear_solve_policy(BvpSciLinearSolvePolicy::AutoBanded)
    }

    pub fn with_experimental_bordered_banded_linear_solver(self) -> Self {
        self.with_linear_solve_policy(BvpSciLinearSolvePolicy::ExperimentalBorderedBanded)
    }

    pub fn with_generated_backend_mode(self, mode: BvpSciGeneratedBackendMode) -> Self {
        self.with_generated_backend_config(BvpSciGeneratedBackendConfig::from_mode(mode))
    }

    pub fn with_expr_legacy_smart_sparse(mut self) -> Self {
        self.generated_backend_config = BvpSciGeneratedBackendConfig::default();
        self.use_analytical_jacobian = true;
        self
    }

    pub fn with_sparse_atomview_c_gcc(self, output_parent_dir: impl Into<PathBuf>) -> Self {
        self.with_generated_backend_config(
            BvpSciGeneratedBackendConfig::sparse_atomview_build_if_missing_release_gcc(
                output_parent_dir,
            ),
        )
    }

    pub fn with_sparse_atomview_rust(self, output_parent_dir: impl Into<PathBuf>) -> Self {
        self.with_generated_backend_config(
            BvpSciGeneratedBackendConfig::sparse_atomview_build_if_missing_release_rust(
                output_parent_dir,
            ),
        )
    }

    pub fn with_sparse_atomview_c_tcc(self, output_parent_dir: impl Into<PathBuf>) -> Self {
        self.with_generated_backend_config(
            BvpSciGeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc(
                output_parent_dir,
            ),
        )
    }

    pub fn with_banded_atomview_c_tcc(self, output_parent_dir: impl Into<PathBuf>) -> Self {
        self.with_generated_backend_config(
            BvpSciGeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc(
                output_parent_dir,
            ),
        )
        .with_auto_banded_linear_solver()
    }

    pub fn with_sparse_atomview_zig(self, output_parent_dir: impl Into<PathBuf>) -> Self {
        self.with_generated_backend_config(
            BvpSciGeneratedBackendConfig::sparse_atomview_build_if_missing_release_zig(
                output_parent_dir,
            ),
        )
    }

    pub fn with_sparse_atomview_for_repeated_solves(
        self,
        output_parent_dir: impl Into<PathBuf>,
    ) -> Self {
        self.with_generated_backend_config(
            BvpSciGeneratedBackendConfig::sparse_atomview_for_repeated_solves(output_parent_dir),
        )
    }
}

#[derive(Debug, Clone)]
pub enum BvpSciBackendError {
    GeneratedBackendNotYetImplemented { mode: BvpSciGeneratedBackendMode },
    GeneratedBackendFailure { message: String },
}

impl Display for BvpSciBackendError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GeneratedBackendNotYetImplemented { mode } => {
                write!(
                    f,
                    "BVP_sci generated backend mode {:?} is not wired into the AOT pipeline yet",
                    mode
                )
            }
            Self::GeneratedBackendFailure { message } => write!(f, "{message}"),
        }
    }
}

impl Error for BvpSciBackendError {}

#[derive(Clone, Debug)]
pub struct BvpSciStatistics {
    pub symbolic_prepare_ms_total: f64,
    pub residual_ms_total: f64,
    pub jacobian_ms_total: f64,
    pub linear_system_ms_total: f64,
    pub grid_refinement_ms_total: f64,
    pub measured_total_ms: f64,
    pub number_of_iterations: usize,
    pub number_of_linear_solves: usize,
    pub number_of_jacobian_recalculations: usize,
    pub number_of_grid_refinements: usize,
    pub number_of_grid_points: usize,
    /// Extensible counters (e.g. "number of backtracking steps", "number of jacobian evaluations")
    pub counters: HashMap<String, usize>,
    /// Extensible timers as "key" → "value_ms" strings (e.g. "aot_build_ms" → "123.456")
    pub timers: HashMap<String, String>,
    /// Extensible diagnostics (e.g. "aot_backend" → "CTcc", "matrix_nnz" → "1024")
    pub diagnostics: HashMap<String, String>,
}

impl Default for BvpSciStatistics {
    fn default() -> Self {
        Self {
            symbolic_prepare_ms_total: 0.0,
            residual_ms_total: 0.0,
            jacobian_ms_total: 0.0,
            linear_system_ms_total: 0.0,
            grid_refinement_ms_total: 0.0,
            measured_total_ms: 0.0,
            number_of_iterations: 0,
            number_of_linear_solves: 0,
            number_of_jacobian_recalculations: 0,
            number_of_grid_refinements: 0,
            number_of_grid_points: 0,
            counters: HashMap::new(),
            timers: HashMap::new(),
            diagnostics: HashMap::new(),
        }
    }
}

impl BvpSciStatistics {
    fn duration_ms(duration: Duration) -> f64 {
        duration.as_secs_f64() * 1_000.0
    }

    fn from_result_and_mesh(result: &BVPResult, grid_points: usize) -> Self {
        let timer = &result.custom_timer;
        let get_count = |key: &str| result.calc_statistics.get(key).copied().unwrap_or(0);
        let symbolic_prepare_ms_total = Self::duration_ms(timer.symbolic_operations);
        let residual_ms_total = Self::duration_ms(timer.fun);
        let jacobian_ms_total = Self::duration_ms(timer.jac);
        let linear_system_ms_total = Self::duration_ms(timer.linear_system);
        let grid_refinement_ms_total = Self::duration_ms(timer.grid_refinement);
        let measured_total_ms = symbolic_prepare_ms_total
            + residual_ms_total
            + jacobian_ms_total
            + linear_system_ms_total
            + grid_refinement_ms_total;

        // Populate counters from calc_statistics HashMap
        let mut counters = HashMap::new();
        for (k, v) in &result.calc_statistics {
            counters.insert(k.clone(), *v);
        }

        // Populate timers from CustomTimer::get_all()
        let timers = result.custom_timer.get_all();

        // Populate diagnostics (currently empty — BVPResult has no diagnostics field)
        let diagnostics = HashMap::new();

        Self {
            symbolic_prepare_ms_total,
            residual_ms_total,
            jacobian_ms_total,
            linear_system_ms_total,
            grid_refinement_ms_total,
            measured_total_ms,
            number_of_iterations: get_count("number of iterations"),
            number_of_linear_solves: get_count("number of solving linear systems"),
            number_of_jacobian_recalculations: get_count("number of jacobians recalculations"),
            number_of_grid_refinements: get_count("number of grid refinements"),
            number_of_grid_points: grid_points,
            counters,
            timers,
            diagnostics,
        }
    }

    /// Set a timer value (formatted as ms with 3 decimal places).
    pub fn set_timer(&mut self, key: impl Into<String>, value_ms: f64) {
        self.timers.insert(key.into(), format!("{:.3}", value_ms));
    }

    /// Set a diagnostic string.
    pub fn set_diagnostic(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.diagnostics.insert(key.into(), value.into());
    }

    /// Increment a counter by `delta` (default 1).
    pub fn increment_counter(&mut self, key: impl Into<String>, delta: usize) {
        *self.counters.entry(key.into()).or_insert(0) += delta;
    }

    /// Get a timer value parsed back to f64, if present.
    pub fn get_timer_ms(&self, key: &str) -> Option<f64> {
        self.timers.get(key).and_then(|v| v.parse::<f64>().ok())
    }

    /// Get a diagnostic string by key.
    pub fn get_diagnostic_string(&self, key: &str) -> Option<&str> {
        self.diagnostics.get(key).map(|s| s.as_str())
    }

    /// Get a counter value by key.
    pub fn get_counter(&self, key: &str) -> Option<usize> {
        self.counters.get(key).copied()
    }

    pub fn table_report(&self) -> String {
        let mut parts = vec![
            format!(
                "symbolic_prepare_ms_total={:.3}",
                self.symbolic_prepare_ms_total
            ),
            format!("residual_ms_total={:.3}", self.residual_ms_total),
            format!("jacobian_ms_total={:.3}", self.jacobian_ms_total),
            format!("linear_system_ms_total={:.3}", self.linear_system_ms_total),
            format!(
                "grid_refinement_ms_total={:.3}",
                self.grid_refinement_ms_total
            ),
            format!("measured_total_ms={:.3}", self.measured_total_ms),
            format!("iterations={}", self.number_of_iterations),
            format!("linear_solves={}", self.number_of_linear_solves),
            format!(
                "jacobian_recalculations={}",
                self.number_of_jacobian_recalculations
            ),
            format!("grid_refinements={}", self.number_of_grid_refinements),
            format!("grid_points={}", self.number_of_grid_points),
        ];
        // Append counters
        for (k, v) in &self.counters {
            parts.push(format!("counter_{}={}", k, v));
        }
        // Append timers
        for (k, v) in &self.timers {
            parts.push(format!("timer_{}={}ms", k, v));
        }
        // Append diagnostics
        for (k, v) in &self.diagnostics {
            parts.push(format!("diag_{}={}", k, v));
        }
        parts.join(" ")
    }
}
pub struct BVPwrap {
    pub eq_system: Vec<Expr>,
    ///equation system
    pub values: Vec<String>,
    ///unknown variables
    pub param: Vec<String>,
    ///parameters
    pub arg: String,
    /// time or coordinate
    pub t0: Option<f64>,
    /// start time or coordinate
    pub t_end: Option<f64>,
    /// end time or coordinate
    pub n_steps: Option<usize>,
    /// number of steps
    pub x_mesh_col: faer_col,
    pub param_values: Option<DVector<f64>>,
    /// parameters values
    pub BoundaryConditions: HashMap<String, Vec<(usize, f64)>>,
    /// boundary conditions function
    pub BC_function: Option<Box<BCFunction>>,
    /// boundary condition function
    pub tolerance: f64,
    ///  Map{var_name: {index, value}} index 0=minimum of variable, 1=maximum of variable
    pub Bounds: Option<HashMap<String, Vec<(usize, f64)>>>,
    /// maximum nodes of mesh
    pub max_nodes: usize,
    pub initial_guess: DMatrix<f64>,
    /// initial guess for the solution
    /// Numerical Jacobian function closure
    pub jac_function: Option<Box<ODEJacobian>>,
    pub banded_jac_function: Option<Arc<ODEBandedJacobian>>,
    /// Numerical residual function closure
    pub residual_function: Box<ODEFunction>,
    pub use_analytical_jacobian: bool,
    pub loglevel: Option<String>,
    pub generated_backend_config: BvpSciGeneratedBackendConfig,
    pub linear_solve_policy: BvpSciLinearSolvePolicy,
    pub result: BVPResult,
    full_result: Option<DMatrix<f64>>,
    custom_timer: CustomTimer,
    generated_backend_last_action: Option<String>,
    generated_backend_last_problem_key: Option<String>,
    generated_backend_last_policy: Option<String>,
    generated_backend_last_toolchain: Option<String>,
    generated_backend_runtime_diagnostics: HashMap<String, String>,
}

struct PreparedBvpSciProblem {
    pointwise: Arc<ExprLegacyPreparedBvpSciProblem>,
    use_analytical_jacobian: bool,
    boundary_conditions: HashMap<String, Vec<(usize, f64)>>,
    values: Vec<String>,
    bounds: Option<HashMap<String, Vec<(usize, f64)>>>,
}

pub(crate) struct ExprLegacyPreparedBvpSciProblem {
    pub(crate) legacy_residual: Arc<ODEFunction>,
    pub(crate) legacy_sparse_jacobian: Arc<ODEJacobian>,
    pub(crate) equations: Vec<Expr>,
    pub(crate) symbolic_jacobian_sparse: Vec<(usize, usize, Expr)>,
    pub(crate) symbolic_param_jacobian_sparse: Option<Vec<(usize, usize, Expr)>>,
    pub(crate) bandwidth: Option<(usize, usize)>,
    pub(crate) time_arg: String,
    pub(crate) variables: Vec<String>,
    pub(crate) equation_parameters: Option<Vec<String>>,
}

fn prepare_bvp_sci_symbolic_jacobian_smart(
    equations: &[Expr],
    variables: &[String],
) -> (Vec<(usize, usize, Expr)>, Option<(usize, usize)>) {
    let mut legacy = Jacobian_sci_faer::new();
    legacy.from_vectors(equations.to_vec(), variables.to_vec());
    legacy.calc_jacobian_parallel_smart();
    legacy.find_bandwidths();
    (legacy.symbolic_jacobian_sparse, legacy.bandwidth)
}

fn build_bvp_sci_symbolic_param_jacobian_sparse(
    equations: &[Expr],
    parameters: Option<&[String]>,
) -> Option<Vec<(usize, usize, Expr)>> {
    let parameters = parameters?;
    Some(
        equations
            .iter()
            .enumerate()
            .flat_map(|(row, expr)| {
                parameters
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

impl BVPwrap {
    pub fn expr_legacy_smart_sparse_workflow() -> BvpSciWorkflow {
        BvpSciWorkflow::ExprLegacySmartSparseLambdify
    }

    pub fn direct_numeric_faer_workflow() -> BvpSciWorkflow {
        BvpSciWorkflow::DirectNumericFaer
    }

    pub fn workflow(&self) -> BvpSciWorkflow {
        self.generated_backend_config.workflow()
    }

    pub fn set_expr_legacy_smart_sparse(&mut self) {
        self.generated_backend_config = BvpSciGeneratedBackendConfig::default();
        self.use_analytical_jacobian = true;
    }

    pub fn with_expr_legacy_smart_sparse(mut self) -> Self {
        self.set_expr_legacy_smart_sparse();
        self
    }

    fn faer_col_from_dvector(values: &DVector<f64>) -> faer_col {
        faer_col::from_iter(values.iter().copied())
    }

    fn dvector_from_faer_col(values: &faer_col) -> DVector<f64> {
        DVector::from_iterator(values.shape().0, values.iter().copied())
    }

    fn faer_dense_from_dmatrix(values: &DMatrix<f64>) -> faer_dense_mat {
        faer_dense_mat::from_fn(values.nrows(), values.ncols(), |i, j| values[(i, j)])
    }

    fn dmatrix_from_faer_dense(values: &faer_dense_mat) -> DMatrix<f64> {
        let (nrows, ncols) = values.shape();
        let mut dmatrix = DMatrix::zeros(ncols, nrows);
        for (i, row) in values.row_iter().enumerate() {
            let row = row.to_owned().iter().copied().collect::<Vec<f64>>();
            dmatrix.column_mut(i).copy_from(&DVector::from_vec(row));
        }
        dmatrix
    }

    pub fn runtime_initial_guess_mat(&self) -> faer_dense_mat {
        Self::faer_dense_from_dmatrix(&self.initial_guess)
    }

    pub fn runtime_param_col(&self) -> Option<faer_col> {
        self.param_values.as_ref().map(Self::faer_col_from_dvector)
    }

    pub fn mesh(&self) -> DVector<f64> {
        Self::dvector_from_faer_col(&self.x_mesh_col)
    }

    pub(crate) fn prepare_exprlegacy_pointwise_problem(
        &self,
    ) -> Result<Arc<ExprLegacyPreparedBvpSciProblem>, BvpSciBackendError> {
        if self.param.is_empty() {
            if self
                .param_values
                .as_ref()
                .is_some_and(|values| !values.is_empty())
            {
                return Err(BvpSciBackendError::GeneratedBackendFailure {
                    message: format!(
                        "BVP_sci pointwise prepare received {} parameter values but no parameter names",
                        self.param_values.as_ref().map_or(0, |values| values.len())
                    ),
                });
            }
        } else {
            let values = self.param_values.as_ref().ok_or_else(|| {
                BvpSciBackendError::GeneratedBackendFailure {
                    message: format!(
                        "BVP_sci pointwise prepare expected {} parameter values but none were provided",
                        self.param.len()
                    ),
                }
            })?;
            if values.len() != self.param.len() {
                return Err(BvpSciBackendError::GeneratedBackendFailure {
                    message: format!(
                        "BVP_sci pointwise prepare expected {} parameter values, got {}",
                        self.param.len(),
                        values.len()
                    ),
                });
            }
        }

        let (symbolic_jacobian_sparse, bandwidth) =
            prepare_bvp_sci_symbolic_jacobian_smart(&self.eq_system, &self.values);
        let eq_system = self.eq_system.clone();
        let values = self.values.clone();
        let param = self.param.clone();
        let arg = self.arg.clone();
        let bounds = self.Bounds.clone();
        let symbolic_param_jacobian_sparse = build_bvp_sci_symbolic_param_jacobian_sparse(
            &eq_system,
            if param.is_empty() {
                None
            } else {
                Some(param.as_slice())
            },
        );
        let legacy_residual = {
            let mut legacy = Jacobian_sci_faer::new();
            legacy.from_vectors(eq_system.clone(), values.clone());
            legacy.symbolic_to_ode_function(
                arg.clone(),
                values.clone(),
                param.clone(),
                bounds.clone(),
            );
            Arc::from(legacy.residual_function)
        };
        let legacy_sparse_jacobian = Arc::from(
            Jacobian_sci_faer::jacobian_generate_SparseColMat_par_sci_from_sparse_entries(
                symbolic_jacobian_sparse.clone(),
                symbolic_param_jacobian_sparse.clone(),
                values.clone(),
                param.clone(),
                arg.clone(),
                bounds,
            ),
        );

        Ok(Arc::new(ExprLegacyPreparedBvpSciProblem {
            legacy_residual,
            legacy_sparse_jacobian,
            equations: eq_system,
            symbolic_jacobian_sparse,
            symbolic_param_jacobian_sparse,
            bandwidth,
            time_arg: arg,
            variables: values,
            equation_parameters: if param.is_empty() { None } else { Some(param) },
        }))
    }

    fn wrap_prepared_problem(
        &self,
        prepared: PreparedBvpSciProblem,
    ) -> (
        Option<Box<ODEJacobian>>,
        Box<ODEFunction>,
        Option<Box<BCFunction>>,
    ) {
        let legacy_residual = Arc::clone(&prepared.pointwise.legacy_residual);
        let residual = Box::new(
            move |x: &faer_col, y: &faer_dense_mat, p: &faer_col| -> faer_dense_mat {
                (legacy_residual)(x, y, p)
            },
        ) as Box<ODEFunction>;

        let jacobian = if prepared.use_analytical_jacobian {
            let legacy_sparse_jacobian = Arc::clone(&prepared.pointwise.legacy_sparse_jacobian);
            Some(Box::new(
                move |x: &faer_col,
                      y: &faer_dense_mat,
                      p: &faer_col|
                      -> (Vec<faer_mat>, Option<Vec<faer_mat>>) {
                    (legacy_sparse_jacobian)(x, y, p)
                },
            ) as Box<ODEJacobian>)
        } else {
            None
        };

        let bc_func = Self::BC_closure_creater(prepared.boundary_conditions, prepared.values);
        (jacobian, residual, bc_func)
    }

    pub fn new_with_options(options: BvpSciSolverOptions) -> Self {
        let mut solver = Self::new(
            options.x_mesh_set,
            options.t0,
            options.t_end,
            options.n_steps,
            options.eq_system,
            options.values,
            options.param,
            options.param_values,
            options.boundary_conditions,
            options.arg,
            options.tolerance,
            options.max_nodes,
            options.initial_guess,
        );
        solver.set_additional_parameters(
            Some(options.use_analytical_jacobian),
            options.bounds,
            options.loglevel,
        );
        solver.set_generated_backend_config(options.generated_backend_config);
        solver.set_linear_solve_policy(options.linear_solve_policy);
        solver
    }

    pub fn new(
        x_mesh_set: Option<DVector<f64>>,
        t0: Option<f64>,
        t_end: Option<f64>,
        n_steps: Option<usize>,
        eq_system: Vec<Expr>,
        values: Vec<String>,
        param: Vec<String>,
        param_values: Option<DVector<f64>>,
        BoundaryConditions: HashMap<String, Vec<(usize, f64)>>,
        arg: String,
        tolerance: f64,
        max_nodes: usize,
        initial_guess: DMatrix<f64>,
    ) -> Self {
        // mesh can be set either by (start, end, steps) or by a vector of points
        let x_mesh_col = if let Some(x_mesh) = x_mesh_set {
            Self::faer_col_from_dvector(&x_mesh)
        } else {
            if t0.is_some() && t_end.is_some() && n_steps.is_some() {
                let start = t0.unwrap();
                let end = t_end.unwrap();
                let steps = n_steps.unwrap();
                faer_col::from_fn(steps, |i| {
                    start + i as f64 * (end - start) / (steps - 1) as f64
                })
            } else {
                panic!("Either provide a mesh vector or t0, t_end, and n_steps");
            }
        };
        let bc_closure = Self::BC_closure_creater(
            BoundaryConditions.clone(), // default empty HashMap for boundary conditions
            values.clone(),
        );
        BVPwrap {
            eq_system: eq_system,
            values: values,
            param: param,
            arg: arg,

            t0: None,
            t_end: None,
            n_steps: None,
            x_mesh_col: x_mesh_col,
            param_values: param_values,
            tolerance: tolerance,
            Bounds: None,
            BoundaryConditions: BoundaryConditions,
            BC_function: bc_closure,
            max_nodes: max_nodes,
            initial_guess: initial_guess,
            jac_function: None,
            banded_jac_function: None,
            residual_function: Box::new(|_, _, _| Mat::zeros(0, 0)),
            result: BVPResult::default(),
            full_result: None,
            use_analytical_jacobian: true,
            loglevel: Some("info".to_string()), // default log level
            generated_backend_config: BvpSciGeneratedBackendConfig::default(),
            linear_solve_policy: BvpSciLinearSolvePolicy::Sparse,
            custom_timer: CustomTimer::new(),
            generated_backend_last_action: None,
            generated_backend_last_problem_key: None,
            generated_backend_last_policy: None,
            generated_backend_last_toolchain: None,
            generated_backend_runtime_diagnostics: HashMap::new(),
        }
    }

    pub fn options_from_current_state(&self) -> BvpSciSolverOptions {
        BvpSciSolverOptions {
            x_mesh_set: Some(self.mesh()),
            t0: self.t0,
            t_end: self.t_end,
            n_steps: self.n_steps,
            eq_system: self.eq_system.clone(),
            values: self.values.clone(),
            param: self.param.clone(),
            param_values: self.param_values.clone(),
            boundary_conditions: self.BoundaryConditions.clone(),
            arg: self.arg.clone(),
            tolerance: self.tolerance,
            max_nodes: self.max_nodes,
            initial_guess: self.initial_guess.clone(),
            use_analytical_jacobian: self.use_analytical_jacobian,
            bounds: self.Bounds.clone(),
            loglevel: self.loglevel.clone(),
            generated_backend_config: self.generated_backend_config.clone(),
            linear_solve_policy: self.linear_solve_policy,
        }
    }

    pub fn set_linear_solve_policy(&mut self, policy: BvpSciLinearSolvePolicy) {
        self.linear_solve_policy = policy;
    }

    pub fn with_linear_solve_policy(mut self, policy: BvpSciLinearSolvePolicy) -> Self {
        self.set_linear_solve_policy(policy);
        self
    }

    pub fn with_auto_banded_linear_solver(self) -> Self {
        self.with_linear_solve_policy(BvpSciLinearSolvePolicy::AutoBanded)
    }

    pub fn with_experimental_bordered_banded_linear_solver(self) -> Self {
        self.with_linear_solve_policy(BvpSciLinearSolvePolicy::ExperimentalBorderedBanded)
    }

    pub fn set_generated_backend_config(&mut self, config: BvpSciGeneratedBackendConfig) {
        self.generated_backend_config = config;
        self.jac_function = None;
        self.banded_jac_function = None;
        self.generated_backend_last_action = None;
        self.generated_backend_last_problem_key = None;
        self.generated_backend_last_policy = None;
        self.generated_backend_last_toolchain = None;
        self.generated_backend_runtime_diagnostics.clear();
    }

    pub fn generated_backend_config(&self) -> &BvpSciGeneratedBackendConfig {
        &self.generated_backend_config
    }

    pub(crate) fn record_generated_backend_lifecycle(
        &mut self,
        action: impl Into<String>,
        problem_key: impl Into<String>,
        policy: impl Into<String>,
        toolchain: impl Into<String>,
    ) {
        self.generated_backend_last_action = Some(action.into());
        self.generated_backend_last_problem_key = Some(problem_key.into());
        self.generated_backend_last_policy = Some(policy.into());
        self.generated_backend_last_toolchain = Some(toolchain.into());
    }

    pub(crate) fn record_generated_backend_runtime_diagnostics(
        &mut self,
        diagnostics: HashMap<String, String>,
    ) {
        self.generated_backend_runtime_diagnostics = diagnostics;
    }

    pub fn with_generated_backend_config(mut self, config: BvpSciGeneratedBackendConfig) -> Self {
        self.set_generated_backend_config(config);
        self
    }

    pub fn set_sparse_generated_backend_mode(&mut self, mode: BvpSciGeneratedBackendMode) {
        let mut config = BvpSciGeneratedBackendConfig::from_mode(mode);
        config.output_parent_dir = self.generated_backend_config.output_parent_dir.clone();
        self.set_generated_backend_config(config);
    }

    pub fn with_sparse_generated_backend_mode(mut self, mode: BvpSciGeneratedBackendMode) -> Self {
        self.set_sparse_generated_backend_mode(mode);
        self
    }

    pub fn with_banded_atomview_c_tcc(mut self, output_parent_dir: impl Into<PathBuf>) -> Self {
        self.set_generated_backend_config(
            BvpSciGeneratedBackendConfig::banded_atomview_build_if_missing_release_tcc(
                output_parent_dir,
            ),
        );
        self.set_linear_solve_policy(BvpSciLinearSolvePolicy::AutoBanded);
        self
    }

    pub fn set_additional_parameters(
        &mut self,
        use_analytical_jacobian: Option<bool>,
        Bounds: Option<HashMap<String, Vec<(usize, f64)>>>,
        loglevel: Option<String>,
    ) {
        if let Some(use_analytical_jacobian) = use_analytical_jacobian {
            info!(
                "Using {} Jacobian",
                if use_analytical_jacobian {
                    "analytical"
                } else {
                    "numerical"
                }
            );
            self.use_analytical_jacobian = use_analytical_jacobian;
        } else {
            info!("Using analytical Jacobian");
        }
        self.Bounds = Bounds;
        if let Some(level) = loglevel {
            self.loglevel = Some(level)
        }
    }
    pub fn check_task(&self) {
        assert_eq!(
            self.eq_system.len(),
            self.values.len(),
            "eq_system and values must have the same length"
        );

        assert!(!self.values.is_empty(), "values must not be empty");

        assert!(!self.arg.is_empty(), "arg must not be empty");

        assert!(
            self.x_mesh_col.shape().0 > 1,
            "x_mesh must have at least 2 points"
        );

        if self.t_end.is_some() && self.t0.is_some() {
            assert!(self.t_end > self.t0, "t_end must be greater than t0");
        }

        if self.n_steps.is_some() {
            assert!(self.n_steps.unwrap() > 1, "n_steps must be greater than 1");
        }

        assert!(self.tolerance > 0.0, "tolerance must be greater than 0.0");
        assert!(
            !self.BoundaryConditions.is_empty(),
            "BoundaryConditions must be specified"
        );
        let total_bcs: usize = self.BoundaryConditions.values().map(|v| v.len()).sum();
        assert_eq!(
            total_bcs,
            self.values.len(),
            "Total boundary conditions must equal number of variables"
        );

        // Check if params are provided, param_values should match
        if !self.param.is_empty() {
            if let Some(param_values) = &self.param_values {
                assert_eq!(
                    param_values.len(),
                    self.param.len(),
                    "param_values must be specified for each param"
                );
            } else {
                panic!("param_values must be provided when params are specified");
            }
        }

        assert_eq!(
            self.initial_guess.nrows(),
            self.values.len(),
            "initial_guess rows must match number of variables"
        );
        assert_eq!(
            self.initial_guess.ncols(),
            self.x_mesh_col.shape().0,
            "initial_guess cols must match mesh points"
        );
    }
    /// Check if all necessary components are created
    pub fn is_all_created(&self) {
        assert!(self.x_mesh_col.shape().0 > 0, "x_mesh_col must be created");
        let initial_guess_mat = self.runtime_initial_guess_mat();
        assert!(
            initial_guess_mat.shape().0 > 0,
            "initial_guess_mat must be created"
        );
        assert!(
            initial_guess_mat.shape().1 > 0,
            "initial_guess_mat must have columns"
        );

        assert!(self.BC_function.is_some(), "BC_function must be created");
        if self.use_analytical_jacobian {
            assert!(self.jac_function.is_some(), "jac_function must be created");
        }

        // Check parameter consistency (params are optional)
        if !self.param.is_empty() {
            let runtime_param_col = self.runtime_param_col();
            assert!(
                runtime_param_col.is_some(),
                "param_col must be created when params exist"
            );
            if let Some(param_col) = &runtime_param_col {
                assert_eq!(
                    param_col.shape().0,
                    self.param.len(),
                    "param_col size must match param names"
                );
            }
        }
    }
    /// BC set as HashMap with key as variable name and value as Vec<(position, value)>
    /// where position: 0=left boundary, 1=right boundary
    pub fn BC_closure_creater(
        map_of_bc: HashMap<String, Vec<(usize, f64)>>,
        values: Vec<String>,
    ) -> Option<Box<BCFunction>> {
        if map_of_bc.is_empty() {
            return None;
        }

        let bc_closure = move |ya: &faer_col, yb: &faer_col, _p: &faer_col| {
            let mut bc_residuals = Vec::new();

            for var_name in &values {
                if let Some(conditions) = map_of_bc.get(var_name) {
                    let var_idx = values.iter().position(|v| v == var_name).unwrap();
                    for &(boundary_idx, target_value) in conditions {
                        let residual = match boundary_idx {
                            0 => ya[var_idx] - target_value, // left boundary
                            1 => yb[var_idx] - target_value, // right boundary
                            _ => panic!("Boundary index must be 0 (left) or 1 (right)"),
                        };
                        bc_residuals.push(residual);
                    }
                }
            }

            faer_col::from_fn(bc_residuals.len(), |i| bc_residuals[i])
        };

        Some(Box::new(bc_closure))
    }
    /// Wrapper function to solve BVP with error handling
    pub fn solve_bvp_wrap(
        &mut self,
        fun: &ODEFunction,
        bc: &BCFunction,
        x: faer_col,
        y: faer_dense_mat,
        p: Option<faer_col>,
        _s: Option<faer_dense_mat>, // Singular term not implemented
        fun_jac: Option<&ODEJacobian>,
        banded_fun_jac: Option<&ODEBandedJacobian>,
        bc_jac: Option<&BCJacobian>,
        tol: f64,
        max_nodes: usize,
        verbose: u8,
        bc_tol: Option<f64>,
        custom_timer: CustomTimer,
        strategy_params: Option<&HashMap<String, Vec<f64>>>,
    ) {
        let begin = Instant::now();

        info!("BVP solver started");
        let bvpres = solve_bvp_with_strategy_linear_policy_and_banded_jacobian(
            fun,
            bc,
            x,
            y,
            p,
            _s,
            fun_jac,
            banded_fun_jac,
            bc_jac,
            tol,
            max_nodes,
            verbose,
            bc_tol,
            Some(custom_timer),
            strategy_params,
            self.linear_solve_policy,
        );

        match bvpres {
            Ok(res) => {
                info!("BVP solved successfully");
                self.result = res.clone();
                self.convert_result();
                let timer = self.result.custom_timer.clone();
                timer.get_all();
                let end = begin.elapsed();
                elapsed_time(end);
                let time = end.as_secs_f64() as usize;
                // println!("{:?}", self.result.calc_statistics);
                let calc_statistics = &mut self.result.calc_statistics;
                calc_statistics.insert("time elapsed, s".to_string(), time);
                self.calc_statistics();
            }
            Err(e) => {
                error!("Error solving BVP: {}", e);
            }
        }
    }
    pub fn solver(&mut self) {
        let begin = Instant::now();
        self.custom_timer.start();
        self.custom_timer.symbolic_operations_tic();
        let (jacobian, residual_function, bc_func) = self.eq_generate();
        self.custom_timer.symbolic_operations_tac();

        // Extract needed fields before mutable borrow

        let x_mesh_col = self.x_mesh_col.clone();
        let initial_guess_mat = self.runtime_initial_guess_mat();
        let param_col = self.runtime_param_col();
        // let jac_func = Some(self.jac_function.as_ref().unwrap());
        let tolerance = self.tolerance;
        let max_nodes = self.max_nodes;
        let custom_timer = self.custom_timer.clone();
        self.solve_bvp_wrap(
            &residual_function,
            bc_func.as_ref().unwrap(),
            x_mesh_col,
            initial_guess_mat,
            param_col,
            None, // Singular term not implemented
            jacobian.as_deref(),
            None,
            None,
            tolerance,
            max_nodes,
            2,    // verbose level
            None, // bc_tol
            custom_timer.clone(),
            None, // strategy_params
        );
        let end = begin.elapsed();
        // let's now calculate paramters of jacobian
        let x_mesh = &self.result.x;
        let y = &self.result.y;
        let empty_p = faer_col::zeros(0);
        let p = self.result.p.as_ref().unwrap_or(&empty_p);
        if let Some(jacobian_fn) = jacobian.as_deref() {
            let jacfunc = jacobian_fn(x_mesh, y, p).0;
            size_of_jacobian(jacfunc);
        }
        elapsed_time(end);
    }

    /// wrapper around solver function to implement logging

    pub fn solve(&mut self) {
        self.try_solve().expect("BVP_sci solve should succeed");
    }

    pub fn try_solve(&mut self) -> Result<(), BvpSciBackendError> {
        self.validate_generated_backend_mode()?;
        let is_logging_disabled = self
            .loglevel
            .as_ref()
            .map(|level| level == "off" || level == "none")
            .unwrap_or(false);

        if is_logging_disabled {
            self.try_solver()?;
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
            let date_and_time = Local::now().format("%Y-%m-%d_%H-%M-%S");
            let name = format!("log_{}.txt", date_and_time);
            let logger_instance = CombinedLogger::init(vec![
                TermLogger::new(
                    log_option,
                    Config::default(),
                    TerminalMode::Mixed,
                    ColorChoice::Auto,
                ),
                WriteLogger::new(log_option, Config::default(), File::create(name).unwrap()),
            ]);

            match logger_instance {
                Ok(()) => {
                    self.try_solver()?;
                    info!(" \n \n Program ended");
                }
                Err(_) => {
                    self.try_solver()?;
                } //end Error
            } // end mat
        }
        Ok(())
    }

    fn try_solver(&mut self) -> Result<(), BvpSciBackendError> {
        let begin = Instant::now();
        self.custom_timer.start();
        self.custom_timer.symbolic_operations_tic();
        let (jacobian, residual_function, bc_func) = self.try_eq_generate()?;
        self.custom_timer.symbolic_operations_tac();

        let x_mesh_col = self.x_mesh_col.clone();
        let initial_guess_mat = self.runtime_initial_guess_mat();
        let param_col = self.runtime_param_col();
        let tolerance = self.tolerance;
        let max_nodes = self.max_nodes;
        let custom_timer = self.custom_timer.clone();
        let banded_jacobian = self.banded_jac_function.clone();
        self.solve_bvp_wrap(
            &residual_function,
            bc_func.as_ref().unwrap(),
            x_mesh_col,
            initial_guess_mat,
            param_col,
            None,
            jacobian.as_deref(),
            banded_jacobian.as_deref(),
            None,
            tolerance,
            max_nodes,
            2,
            None,
            custom_timer.clone(),
            None, // strategy_params
        );
        let end = begin.elapsed();
        let x_mesh = &self.result.x;
        let y = &self.result.y;
        let empty_p = faer_col::zeros(0);
        let p = self.result.p.as_ref().unwrap_or(&empty_p);
        if banded_jacobian.is_none() {
            if let Some(jacobian_fn) = jacobian.as_deref() {
                let jacfunc = jacobian_fn(x_mesh, y, p).0;
                size_of_jacobian(jacfunc);
            }
        }
        elapsed_time(end);
        Ok(())
    }

    pub fn try_eq_generate(
        &mut self,
    ) -> Result<
        (
            Option<Box<ODEJacobian>>,
            Box<ODEFunction>,
            Option<Box<BCFunction>>,
        ),
        BvpSciBackendError,
    > {
        match self.generated_backend_config.mode {
            BvpSciGeneratedBackendMode::LambdifyOnly => Ok(self.eq_generate()),
            _ => self.eq_generate_generated(),
        }
    }

    fn validate_generated_backend_mode(&self) -> Result<(), BvpSciBackendError> {
        match self.generated_backend_config.mode {
            BvpSciGeneratedBackendMode::LambdifyOnly => Ok(()),
            _ => {
                let _ = self.bvp_generated_backend_config()?;
                Ok(())
            }
        }
    }

    pub fn eq_generate(
        &mut self,
    ) -> (
        Option<Box<ODEJacobian>>,
        Box<ODEFunction>,
        Option<Box<BCFunction>>,
    ) {
        self.check_task();
        if !matches!(
            self.generated_backend_config.mode,
            BvpSciGeneratedBackendMode::LambdifyOnly
        ) {
            info!("BVP_sci generated backend closures installed");
            let generated = self.eq_generate_generated().unwrap_or_else(|err| {
                panic!("BVP_sci generated backend generation should succeed: {err}")
            });
            self.is_all_created();
            return generated;
        }
        let prepared = PreparedBvpSciProblem {
            pointwise: self
                .prepare_exprlegacy_pointwise_problem()
                .unwrap_or_else(|err| panic!("BVP_sci lambdify preparation should succeed: {err}")),
            use_analytical_jacobian: self.use_analytical_jacobian,
            boundary_conditions: self.BoundaryConditions.clone(),
            values: self.values.clone(),
            bounds: self.Bounds.clone(),
        };
        let (self_jac, self_residual, self_bc) =
            self.wrap_prepared_problem(PreparedBvpSciProblem {
                pointwise: Arc::clone(&prepared.pointwise),
                use_analytical_jacobian: prepared.use_analytical_jacobian,
                boundary_conditions: prepared.boundary_conditions.clone(),
                values: prepared.values.clone(),
                bounds: prepared.bounds.clone(),
            });
        self.jac_function = self_jac;
        self.residual_function = self_residual;
        self.BC_function = self_bc;
        let (jacobian, residual, bc_func) = self.wrap_prepared_problem(prepared);
        self.is_all_created();
        info!("BVP equations generated");
        info!("BVP symbolic Jacobian generated");
        (jacobian, residual, bc_func)
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                     functions to return and save result in different formats
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // returns a tuple of a dense matrix and a vector
    fn convert_result(&mut self) {
        let x = self.result.x.clone();
        let y = self.result.y.clone();
        self.x_mesh_col = x;
        let dmatrix = Self::dmatrix_from_faer_dense(&y);
        self.full_result = Some(dmatrix);
    }

    pub fn save_to_file(&self, filename: Option<String>) {
        //let date_and_time = Local::now().format("%Y-%m-%d_%H-%M-%S");
        let name = if let Some(name) = filename {
            format!("{}.txt", name)
        } else {
            "result.txt".to_string()
        };
        let result_DMatrix = self.get_result().unwrap();
        let _ = save_matrix_to_file(
            &result_DMatrix,
            &self.values,
            &name,
            &self.mesh(),
            &self.arg,
        );
    }
    pub fn save_to_csv(&self, filename: Option<String>) {
        let name = if let Some(name) = filename {
            name
        } else {
            "result_table".to_string()
        };
        let result_DMatrix = self.get_result().unwrap();
        let _ = save_matrix_to_csv(
            &result_DMatrix,
            &self.values,
            &name,
            &self.mesh(),
            &self.arg,
        );
    }
    pub fn get_result(&self) -> Option<DMatrix<f64>> {
        self.full_result.clone()
    }

    pub fn gnuplot_result(&self) {
        let permutted_results = self.full_result.clone().unwrap();
        plots_gnulot(
            self.arg.clone(),
            self.values.clone(),
            self.mesh(),
            permutted_results,
        );
        info!("result plotted");
    }
    pub fn plot_result(&self) {
        let permutted_results = self.full_result.clone().unwrap();
        plots(
            self.arg.clone(),
            self.values.clone(),
            self.mesh(),
            permutted_results,
        );
        info!("result plotted");
    }

    pub fn get_statistics(&self) -> BvpSciStatistics {
        let mut statistics =
            BvpSciStatistics::from_result_and_mesh(&self.result, self.x_mesh_col.nrows());
        if let Some(action) = &self.generated_backend_last_action {
            statistics.set_diagnostic("generated_backend_action", action.clone());
        }
        if let Some(problem_key) = &self.generated_backend_last_problem_key {
            statistics.set_diagnostic("generated_backend_problem_key", problem_key.clone());
        }
        if let Some(policy) = &self.generated_backend_last_policy {
            statistics.set_diagnostic("generated_backend_policy", policy.clone());
        }
        if let Some(toolchain) = &self.generated_backend_last_toolchain {
            statistics.set_diagnostic("generated_backend_toolchain", toolchain.clone());
        }
        for (key, value) in &self.generated_backend_runtime_diagnostics {
            statistics.set_diagnostic(key.clone(), value.clone());
        }
        statistics
    }

    pub fn statistics_report(&self) -> String {
        self.get_statistics().table_report()
    }

    fn calc_statistics(&self) {
        let mut stats = self.result.calc_statistics.clone();
        let mut typed_stats = self.get_statistics();

        stats.insert(
            "number of grid points".to_string(),
            self.x_mesh_col.nrows() as usize,
        );

        // Record diagnostics
        let n_variables = self.result.y.nrows();
        let n_mesh_points = self.x_mesh_col.nrows();
        typed_stats.set_diagnostic("n_variables", n_variables.to_string());
        typed_stats.set_diagnostic("n_mesh_points", n_mesh_points.to_string());
        if let Some(nnz) = stats.get("matrix_nnz") {
            typed_stats.set_diagnostic("matrix_nnz", nnz.to_string());
        }
        if let Some(strategy) = stats.get("solver_strategy") {
            typed_stats.set_diagnostic("solver_strategy", strategy.to_string());
        }

        let mut table = Builder::from(stats).build();
        table.with(Style::modern_rounded());
        info!("\n \n CALC STATISTICS \n \n {}", table.to_string());
        info!(
            "\n \n TYPED STATISTICS \n \n {}",
            typed_stats.table_report()
        );
    }
}

#[cfg(test)]
mod tests_phase1 {
    use super::{
        BVPwrap, BvpSciBackendError, BvpSciGeneratedBackendConfig, BvpSciGeneratedBackendMode,
        BvpSciSolverOptions, BvpSciWorkflow, Jacobian_sci_faer,
    };
    use crate::numerical::BVP_sci::BVP_sci_faer::faer_col;
    use crate::symbolic::codegen::codegen_aot_runtime_link::{
        resolve_linked_sparse_backend, unregister_linked_sparse_backend,
    };
    use crate::symbolic::symbolic_engine::Expr;
    use nalgebra::{DMatrix, DVector};
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    fn generated_backend_registry_guard() -> std::sync::MutexGuard<'static, ()> {
        static GENERATED_BACKEND_REGISTRY_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        GENERATED_BACKEND_REGISTRY_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("generated backend registry test lock should not be poisoned")
    }
    use std::process::Command;

    fn tcc_is_available() -> bool {
        if let Ok(explicit) = std::env::var("RUSTEDSCITHE_TCC") {
            return std::path::Path::new(&explicit).is_file();
        }
        Command::new("tcc").arg("-v").output().is_ok()
    }

    fn gcc_is_available() -> bool {
        if let Ok(explicit) = std::env::var("RUSTEDSCITHE_GCC") {
            return std::path::Path::new(&explicit).is_file();
        }
        if let Ok(explicit) = std::env::var("RUSTEDSCITHE_C_COMPILER") {
            if std::path::Path::new(&explicit).is_file() {
                return explicit.to_ascii_lowercase().contains("gcc");
            }
        }
        Command::new("gcc").arg("-v").output().is_ok()
    }

    const GENERATED_TEST_ARTIFACT_REV: &str = "rev2-bounds";

    fn generated_test_artifact_dir(name: &str) -> String {
        format!("target/generated-bvp-sci-tests/{GENERATED_TEST_ARTIFACT_REV}/{name}")
    }

    fn linear_problem_options() -> BvpSciSolverOptions {
        let eq_system = vec![Expr::parse_expression("z"), Expr::parse_expression("0")];
        let values = vec!["y".to_string(), "z".to_string()];
        let mut boundary_conditions = HashMap::new();
        boundary_conditions.insert("y".to_string(), vec![(0, 0.0), (1, 1.0)]);
        BvpSciSolverOptions::new(
            None,
            Some(0.0),
            Some(1.0),
            Some(8),
            eq_system,
            values,
            vec![],
            None,
            boundary_conditions,
            "x".to_string(),
            1e-4,
            100,
            DMatrix::zeros(2, 8),
        )
        .with_loglevel(Some("off".to_string()))
    }

    fn isolated_missing_backend_options() -> BvpSciSolverOptions {
        let eq_system = vec![Expr::parse_expression("z"), Expr::parse_expression("1")];
        let values = vec!["y".to_string(), "z".to_string()];
        let mut boundary_conditions = HashMap::new();
        boundary_conditions.insert("y".to_string(), vec![(0, 0.25), (1, 0.75)]);
        BvpSciSolverOptions::new(
            None,
            Some(0.0),
            Some(1.0),
            Some(8),
            eq_system,
            values,
            vec![],
            None,
            boundary_conditions,
            "x".to_string(),
            1e-4,
            100,
            DMatrix::zeros(2, 8),
        )
        .with_loglevel(Some("off".to_string()))
    }

    fn exponential_problem_options() -> BvpSciSolverOptions {
        let eq_system = Expr::parse_vector_expression(vec!["z", "-(2.0/4.0)*(1+2.0*ln((y)))*y"]);
        let values = vec!["y".to_string(), "z".to_string()];
        let mut boundary_conditions = HashMap::new();
        let a = 4.0_f64;
        let bc_val = (-1.0 / a).exp();
        boundary_conditions.insert("y".to_string(), vec![(0, bc_val), (1, bc_val)]);
        let mut bounds = HashMap::new();
        bounds.insert("y".to_string(), vec![(0, 1e-10)]);
        BvpSciSolverOptions::new(
            None,
            Some(-1.0),
            Some(1.0),
            Some(32),
            eq_system,
            values,
            vec![],
            None,
            boundary_conditions,
            "x".to_string(),
            1e-6,
            2000,
            DMatrix::zeros(2, 32),
        )
        .with_bounds(Some(bounds))
        .with_loglevel(Some("off".to_string()))
    }

    fn max_abs_matrix(matrix: &DMatrix<f64>) -> f64 {
        matrix
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()))
    }

    fn max_abs_diff(lhs: &DMatrix<f64>, rhs: &DMatrix<f64>) -> f64 {
        lhs.iter()
            .zip(rhs.iter())
            .fold(0.0_f64, |acc, (a, b)| acc.max((a - b).abs()))
    }

    #[test]
    fn bvp_sci_new_with_options_preserves_main_knobs() {
        let options = linear_problem_options().with_analytical_jacobian(false);
        let solver = BVPwrap::new_with_options(options.clone());
        assert_eq!(solver.values, options.values);
        assert_eq!(solver.arg, options.arg);
        assert_eq!(
            solver.use_analytical_jacobian,
            options.use_analytical_jacobian
        );
        assert_eq!(solver.max_nodes, options.max_nodes);
    }

    #[test]
    fn bvp_sci_exprlegacy_workflow_is_explicitly_exposed() {
        let solver =
            BVPwrap::new_with_options(linear_problem_options()).with_expr_legacy_smart_sparse();
        assert_eq!(
            solver.workflow(),
            BvpSciWorkflow::ExprLegacySmartSparseLambdify
        );
        assert_eq!(
            solver.generated_backend_config().mode,
            BvpSciGeneratedBackendMode::LambdifyOnly
        );
    }

    #[test]
    fn bvp_sci_statistics_are_exposed_after_solve() {
        let options = linear_problem_options();
        let mut solver = BVPwrap::new_with_options(options);
        solver.solve();
        let stats = solver.get_statistics();
        assert!(stats.number_of_grid_points > 0);
        assert!(!solver.statistics_report().is_empty());
    }

    #[test]
    fn bvp_sci_generated_backend_mode_is_exposed_on_surface() {
        let solver = BVPwrap::new_with_options(linear_problem_options())
            .with_sparse_generated_backend_mode(
                BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseTcc,
            );
        assert_eq!(
            solver.generated_backend_config().mode,
            BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseTcc
        );
        assert_eq!(solver.workflow(), BvpSciWorkflow::AtomViewAotSparse);
    }

    #[test]
    fn bvp_sci_generated_aot_modes_are_sparse_atomview_only() {
        let aot_modes = [
            BvpSciGeneratedBackendMode::RequirePrebuiltAot,
            BvpSciGeneratedBackendMode::BuildIfMissingRelease,
            BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseRust,
            BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseGcc,
            BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseTcc,
            BvpSciGeneratedBackendMode::AtomViewBuildIfMissingReleaseZig,
            BvpSciGeneratedBackendMode::AtomViewForRepeatedSolves,
        ];

        for mode in aot_modes {
            let config = BvpSciGeneratedBackendConfig::from_mode(mode);
            assert_eq!(
                config.workflow(),
                BvpSciWorkflow::AtomViewAotSparse,
                "{mode:?} must stay an explicitly sparse AtomView AOT route"
            );
        }
    }

    #[test]
    fn bvp_sci_generated_backend_options_aliases_are_preserved() {
        let options = linear_problem_options()
            .with_sparse_atomview_for_repeated_solves("target/test-artifacts/bvp-sci");
        let solver = BVPwrap::new_with_options(options.clone());
        assert_eq!(
            solver.generated_backend_config().mode,
            BvpSciGeneratedBackendMode::AtomViewForRepeatedSolves
        );
        assert_eq!(
            solver.generated_backend_config().output_parent_dir,
            options.generated_backend_config.output_parent_dir
        );
    }

    #[test]
    fn bvp_sci_try_eq_generate_surfaces_unimplemented_generated_backend() {
        let mut solver = BVPwrap::new_with_options(isolated_missing_backend_options())
            .with_sparse_generated_backend_mode(BvpSciGeneratedBackendMode::RequirePrebuiltAot);

        match solver.try_eq_generate() {
            Err(BvpSciBackendError::GeneratedBackendFailure { message }) => {
                assert!(message.contains("missing") || message.contains("artifact"));
            }
            Err(BvpSciBackendError::GeneratedBackendNotYetImplemented { .. }) => {
                panic!("generated backend path should no longer stop at NotYetImplemented")
            }
            Ok(_) => panic!("missing prebuilt backend should surface a typed error"),
        }
    }

    #[test]
    fn bvp_sci_try_solve_surfaces_missing_prebuilt_generated_backend() {
        let mut solver = BVPwrap::new_with_options(isolated_missing_backend_options())
            .with_sparse_generated_backend_mode(BvpSciGeneratedBackendMode::RequirePrebuiltAot);

        match solver.try_solve() {
            Err(BvpSciBackendError::GeneratedBackendFailure { message }) => {
                assert!(message.contains("missing") || message.contains("artifact"));
            }
            Err(BvpSciBackendError::GeneratedBackendNotYetImplemented { .. }) => {
                panic!("generated backend path should no longer stop at NotYetImplemented")
            }
            Ok(_) => panic!("missing prebuilt backend should surface a typed error"),
        }
    }

    #[test]
    fn bvp_sci_generated_backend_ctcc_smoke_solve() {
        let _registry_guard = generated_backend_registry_guard();
        if !tcc_is_available() {
            println!(
                "[BVP_sci generated backend] skipping C-tcc smoke test: compiler not available"
            );
            return;
        }

        let mut solver = BVPwrap::new_with_options(
            linear_problem_options()
                .with_sparse_atomview_c_tcc(generated_test_artifact_dir("ctcc")),
        );

        solver
            .try_solve()
            .expect("BVP_sci generated backend C-tcc smoke solve should succeed");

        assert!(
            solver.result.success,
            "generated BVP_sci solve should finish successfully"
        );
        assert!(
            solver.get_result().is_some(),
            "generated BVP_sci solve should produce a result matrix"
        );
        assert!(
            solver.get_statistics().number_of_grid_points > 0,
            "generated BVP_sci solve should expose non-empty statistics"
        );
        let stats = solver.get_statistics();
        if let Some(problem_key) = stats.get_diagnostic_string("generated_backend_problem_key") {
            let _ = unregister_linked_sparse_backend(problem_key);
        }
    }

    #[test]
    fn bvp_sci_generated_banded_ctcc_uses_native_bordered_route() {
        let _registry_guard = generated_backend_registry_guard();
        if !tcc_is_available() {
            println!(
                "[BVP_sci generated backend] skipping native banded C-tcc smoke test: compiler not available"
            );
            return;
        }

        let mut solver = BVPwrap::new_with_options(
            linear_problem_options()
                .with_banded_atomview_c_tcc(generated_test_artifact_dir("banded-ctcc-native")),
        );

        solver
            .try_solve()
            .expect("BVP_sci native banded C-tcc solve should succeed");

        assert!(solver.result.success);
        assert!(solver.get_result().is_some());
        let stats = solver.get_statistics();
        assert!(
            stats
                .get_counter("bvp sci direct banded assembly calls")
                .unwrap_or(0)
                > 0,
            "true banded AOT must assemble bordered blocks directly"
        );
        assert!(
            stats
                .get_counter("bvp sci global sparse jacobian bypasses")
                .unwrap_or(0)
                > 0,
            "true banded AOT must bypass global sparse Jacobian assembly"
        );
        assert!(
            stats
                .get_counter("bvp sci bordered factorization calls")
                .unwrap_or(0)
                > 0,
            "true banded AOT must use the native bordered factorization"
        );
        assert_eq!(
            stats
                .get_counter("bvp sci sparse linear solves")
                .unwrap_or(0),
            0,
            "native banded smoke route must not silently fall back to Sparse LU"
        );
        if let Some(problem_key) = stats.get_diagnostic_string("generated_backend_problem_key") {
            let _ = unregister_linked_sparse_backend(problem_key);
        }
    }

    #[test]
    fn bvp_sci_generated_backend_cgcc_smoke_solve() {
        let _registry_guard = generated_backend_registry_guard();
        if !gcc_is_available() {
            println!(
                "[BVP_sci generated backend] skipping C-gcc smoke test: compiler not available"
            );
            return;
        }

        let mut solver = BVPwrap::new_with_options(
            linear_problem_options()
                .with_sparse_atomview_c_gcc(generated_test_artifact_dir("cgcc")),
        );

        solver
            .try_solve()
            .expect("BVP_sci generated backend C-gcc smoke solve should succeed");

        assert!(
            solver.result.success,
            "generated BVP_sci solve should finish successfully"
        );
        assert!(
            solver.get_result().is_some(),
            "generated BVP_sci solve should produce a result matrix"
        );
        assert!(
            solver.get_statistics().number_of_grid_points > 0,
            "generated BVP_sci solve should expose non-empty statistics"
        );
        let stats = solver.get_statistics();
        if let Some(problem_key) = stats.get_diagnostic_string("generated_backend_problem_key") {
            let _ = unregister_linked_sparse_backend(problem_key);
        }
    }

    #[test]
    fn bvp_sci_generated_backend_ctcc_registers_sparse_runtime() {
        let _registry_guard = generated_backend_registry_guard();
        if !tcc_is_available() {
            println!(
                "[BVP_sci generated backend] skipping sparse runtime registration test: compiler not available"
            );
            return;
        }

        let mut solver = BVPwrap::new_with_options(
            linear_problem_options()
                .with_sparse_atomview_c_tcc(generated_test_artifact_dir("ctcc-registry")),
        );
        let pointwise = solver
            .prepare_atomview_pointwise_problem()
            .expect("pointwise AtomView problem should prepare");
        let config = solver
            .bvp_generated_backend_config()
            .expect("generated backend config lookup should succeed")
            .expect("generated backend config should exist");
        let problem_key = solver
            .ensure_sparse_generated_runtime(&pointwise, &config)
            .expect("sparse generated runtime build/link should succeed");

        let linked = resolve_linked_sparse_backend(problem_key.as_str())
            .expect("generated BVP_sci should register sparse runtime backend");
        assert_eq!(linked.residual_len, solver.values.len());
        assert!(
            linked.nnz > 0,
            "linked sparse backend should expose nonzero Jacobian entries"
        );
        let stats = solver.get_statistics();
        assert_eq!(
            stats.get_diagnostic_string("generated_backend_action"),
            Some("built_and_linked")
        );
        assert_eq!(
            stats.get_diagnostic_string("generated_backend_policy"),
            Some("BuildIfMissing")
        );
        let _ = unregister_linked_sparse_backend(problem_key.as_str());
    }

    #[test]
    fn bvp_sci_symbolic_sparse_structure_matches_linear_problem_jacobian_pattern() {
        let solver = BVPwrap::new_with_options(linear_problem_options());
        let prepared = solver
            .prepare_atomview_pointwise_problem()
            .expect("pointwise AtomView problem should prepare");
        let structure = BVPwrap::symbolic_sparse_structure(&prepared);

        assert_eq!(structure.rows, 2);
        assert_eq!(structure.cols, 2);
        assert_eq!(structure.nnz(), 1);
        assert_eq!(structure.row_indices, vec![0]);
        assert_eq!(structure.col_indices, vec![1]);
    }

    #[test]
    fn bvp_sci_exprlegacy_prepare_preserves_parameter_jacobian_and_sparse_entries() {
        let equations = Expr::parse_vector_expression(vec!["a * y", "b * z"]);
        let values = vec!["y".to_string(), "z".to_string()];
        let params = vec!["a".to_string(), "b".to_string()];
        let boundary_conditions = HashMap::from([
            ("y".to_string(), vec![(0, 1.0)]),
            ("z".to_string(), vec![(1, 2.0)]),
        ]);
        let initial_guess = DMatrix::from_element(2, 5, 1.0);
        let solver = BVPwrap::new(
            None,
            Some(0.0),
            Some(1.0),
            Some(5),
            equations,
            values,
            params,
            Some(DVector::from_vec(vec![3.0, 4.0])),
            boundary_conditions,
            "x".to_string(),
            1e-6,
            32,
            initial_guess,
        );
        let prepared = solver
            .prepare_exprlegacy_pointwise_problem()
            .expect("parameterized ExprLegacy pointwise problem should prepare");

        assert_eq!(
            prepared.symbolic_jacobian_sparse.len(),
            2,
            "df/dy sparse entries should keep only nonzero derivatives"
        );
        let param_sparse = prepared
            .symbolic_param_jacobian_sparse
            .as_ref()
            .expect("df/dp sparse entries should be present for parameterized problem");
        assert_eq!(param_sparse.len(), 2);

        let x = faer_col::from_fn(5, |i| i as f64 * 0.25);
        let y = solver.runtime_initial_guess_mat();
        let p = faer_col::from_fn(2, |i| if i == 0 { 3.0 } else { 4.0 });
        let (_, df_dp) = (prepared.legacy_sparse_jacobian)(&x, &y, &p);
        let df_dp = df_dp.expect("legacy ExprLegacy sparse Jacobian should preserve df/dp");

        assert!((df_dp[0].get(0, 0).copied().unwrap_or(0.0) - 1.0).abs() < 1e-12);
        assert!(df_dp[0].get(0, 1).copied().unwrap_or(0.0).abs() < 1e-12);
        assert!(df_dp[0].get(1, 0).copied().unwrap_or(0.0).abs() < 1e-12);
        assert!((df_dp[0].get(1, 1).copied().unwrap_or(0.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn bvp_sci_pointwise_prepare_reuses_legacy_smart_jacobian_and_bandwidth() {
        let equations = Expr::parse_vector_expression(vec!["z", "-y + z", "u"]);
        let values = vec!["y".to_string(), "z".to_string(), "u".to_string()];

        let solver = BVPwrap::new(
            None,
            Some(0.0),
            Some(1.0),
            Some(5),
            equations.clone(),
            values.clone(),
            vec![],
            None,
            HashMap::from([
                ("y".to_string(), vec![(0, 1.0)]),
                ("z".to_string(), vec![(1, 0.0)]),
                ("u".to_string(), vec![(0, 0.0)]),
            ]),
            "x".to_string(),
            1e-6,
            32,
            DMatrix::from_element(3, 5, 1.0),
        );

        let prepared = solver
            .prepare_exprlegacy_pointwise_problem()
            .expect("pointwise lambdify problem should prepare");

        let mut legacy = Jacobian_sci_faer::new();
        legacy.from_vectors(equations, values);
        legacy.calc_jacobian_parallel_smart();
        legacy.find_bandwidths();

        assert_eq!(prepared.bandwidth, legacy.bandwidth);
        assert_eq!(
            prepared.symbolic_jacobian_sparse,
            legacy.symbolic_jacobian_sparse
        );
    }

    #[test]
    fn bvp_sci_exprlegacy_jacobian_uses_runtime_parameter_vector() {
        let equations = Expr::parse_vector_expression(vec!["a * y", "b * z"]);
        let values = vec!["y".to_string(), "z".to_string()];
        let params = vec!["a".to_string(), "b".to_string()];
        let boundary_conditions = HashMap::from([
            ("y".to_string(), vec![(0, 1.0)]),
            ("z".to_string(), vec![(1, 2.0)]),
        ]);
        let mut solver = BVPwrap::new(
            None,
            Some(0.0),
            Some(1.0),
            Some(5),
            equations,
            values,
            params,
            Some(DVector::from_vec(vec![3.0, 4.0])),
            boundary_conditions,
            "x".to_string(),
            1e-6,
            32,
            DMatrix::from_element(2, 5, 1.0),
        )
        .with_expr_legacy_smart_sparse();

        let (jacobian, _, _) = solver.eq_generate();
        let jacobian = jacobian.expect("ExprLegacy Jacobian callback should exist");

        let x = faer_col::from_fn(5, |i| i as f64 * 0.25);
        let y = solver.runtime_initial_guess_mat();
        let p = faer_col::from_fn(2, |i| if i == 0 { 10.0 } else { 20.0 });
        let (df_dy, df_dp) = jacobian(&x, &y, &p);

        assert!((df_dy[0].get(0, 0).copied().unwrap_or(0.0) - 10.0).abs() < 1e-12);
        assert!((df_dy[0].get(1, 1).copied().unwrap_or(0.0) - 20.0).abs() < 1e-12);

        let df_dp = df_dp.expect("df/dp should be present for parameterized ExprLegacy path");
        assert!((df_dp[0].get(0, 0).copied().unwrap_or(0.0) - 1.0).abs() < 1e-12);
        assert!((df_dp[0].get(1, 1).copied().unwrap_or(0.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn bvp_sci_exprlegacy_residual_uses_runtime_parameter_vector() {
        let equations = Expr::parse_vector_expression(vec!["a * y", "b * z"]);
        let values = vec!["y".to_string(), "z".to_string()];
        let params = vec!["a".to_string(), "b".to_string()];
        let boundary_conditions = HashMap::from([
            ("y".to_string(), vec![(0, 1.0)]),
            ("z".to_string(), vec![(1, 2.0)]),
        ]);
        let mut solver = BVPwrap::new(
            None,
            Some(0.0),
            Some(1.0),
            Some(5),
            equations,
            values,
            params,
            Some(DVector::from_vec(vec![3.0, 4.0])),
            boundary_conditions,
            "x".to_string(),
            1e-6,
            32,
            DMatrix::from_element(2, 5, 1.0),
        )
        .with_expr_legacy_smart_sparse();

        let (_, residual, _) = solver.eq_generate();
        let x = faer_col::from_fn(5, |i| i as f64 * 0.25);
        let y = solver.runtime_initial_guess_mat();
        let p = faer_col::from_fn(2, |i| if i == 0 { 10.0 } else { 20.0 });
        let values = residual(&x, &y, &p);

        assert!((values.get(0, 0) - 10.0).abs() < 1e-12);
        assert!((values.get(1, 0) - 20.0).abs() < 1e-12);
    }

    #[test]
    fn bvp_sci_generated_backend_ctcc_preserves_parameter_jacobian_callback() {
        let _registry_guard = generated_backend_registry_guard();
        if !tcc_is_available() {
            println!(
                "[BVP_sci generated backend] skipping parameter Jacobian callback test: C-tcc not available"
            );
            return;
        }

        let equations = Expr::parse_vector_expression(vec!["a * y", "b * z"]);
        let values = vec!["y".to_string(), "z".to_string()];
        let params = vec!["a".to_string(), "b".to_string()];
        let boundary_conditions = HashMap::from([
            ("y".to_string(), vec![(0, 1.0)]),
            ("z".to_string(), vec![(1, 2.0)]),
        ]);
        let initial_guess = DMatrix::from_element(2, 5, 1.0);
        let mut solver = BVPwrap::new(
            None,
            Some(0.0),
            Some(1.0),
            Some(5),
            equations,
            values,
            params,
            Some(DVector::from_vec(vec![3.0, 4.0])),
            boundary_conditions,
            "x".to_string(),
            1e-6,
            32,
            initial_guess,
        )
        .with_generated_backend_config(
            BvpSciGeneratedBackendConfig::sparse_atomview_build_if_missing_release_tcc(
                generated_test_artifact_dir("ctcc-param-dfdp"),
            ),
        );

        let (jacobian, _, _) = solver
            .try_eq_generate()
            .expect("generated parameterized callbacks should prepare");
        let jacobian = jacobian.expect("generated Jacobian callback should exist");

        let x = faer_col::from_fn(5, |i| i as f64 * 0.25);
        let y = solver.runtime_initial_guess_mat();
        let p = solver
            .runtime_param_col()
            .expect("parameterized problem should expose parameter values");
        let (_, df_dp) = jacobian(&x, &y, &p);
        let df_dp = df_dp.expect("generated path should preserve df/dp callback");
        assert_eq!(df_dp.len(), 5);
        assert!((df_dp[0].get(0, 0).copied().unwrap_or(0.0) - 1.0).abs() < 1e-12);
        assert!(df_dp[0].get(0, 1).copied().unwrap_or(0.0).abs() < 1e-12);
        assert!(df_dp[0].get(1, 0).copied().unwrap_or(0.0).abs() < 1e-12);
        assert!((df_dp[0].get(1, 1).copied().unwrap_or(0.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn bvp_sci_exponential_generated_callbacks_match_lambdify_ctcc() {
        if !tcc_is_available() {
            println!(
                "[BVP_sci generated backend] skipping exponential callback consistency test: C-tcc not available"
            );
            return;
        }

        let baseline_solver = BVPwrap::new_with_options(exponential_problem_options());
        let generated_solver = BVPwrap::new_with_options(
            exponential_problem_options()
                .with_sparse_atomview_c_tcc(generated_test_artifact_dir("exponential-ctcc")),
        );

        let x = baseline_solver.x_mesh_col.clone();
        let y = baseline_solver.runtime_initial_guess_mat();
        let p = baseline_solver
            .runtime_param_col()
            .unwrap_or_else(|| faer_col::zeros(0));

        let (baseline_jac, baseline_res, baseline_bc) = {
            let mut solver = baseline_solver;
            solver.eq_generate()
        };
        let (generated_jac, generated_res, generated_bc) = {
            let mut solver = generated_solver;
            solver
                .try_eq_generate()
                .expect("generated exponential callbacks should be prepared")
        };

        let baseline_residual = baseline_res(&x, &y, &p);
        let generated_residual = generated_res(&x, &y, &p);
        let residual_diff = {
            let mut max_diff = 0.0_f64;
            for i in 0..baseline_residual.nrows() {
                for j in 0..baseline_residual.ncols() {
                    max_diff = max_diff
                        .max((*baseline_residual.get(i, j) - *generated_residual.get(i, j)).abs());
                }
            }
            max_diff
        };

        let baseline_ya = faer_col::from_fn(y.nrows(), |i| *y.get(i, 0));
        let baseline_yb = faer_col::from_fn(y.nrows(), |i| *y.get(i, y.ncols() - 1));
        let baseline_bc_values =
            baseline_bc.expect("baseline BC callback should exist")(&baseline_ya, &baseline_yb, &p);
        let generated_bc_values = generated_bc.expect("generated BC callback should exist")(
            &baseline_ya,
            &baseline_yb,
            &p,
        );
        let bc_diff = baseline_bc_values
            .iter()
            .zip(generated_bc_values.iter())
            .fold(0.0_f64, |acc, (lhs, rhs)| acc.max((lhs - rhs).abs()));

        let baseline_jac_values =
            baseline_jac.expect("baseline jacobian callback should exist")(&x, &y, &p).0;
        let generated_jac_values =
            generated_jac.expect("generated jacobian callback should exist")(&x, &y, &p).0;
        let jacobian_diff = baseline_jac_values
            .iter()
            .zip(generated_jac_values.iter())
            .fold(0.0_f64, |acc, (lhs, rhs)| {
                let mut local = 0.0_f64;
                for row in 0..lhs.nrows() {
                    for col in 0..lhs.ncols() {
                        let l = lhs.get(row, col).copied().unwrap_or(0.0);
                        let r = rhs.get(row, col).copied().unwrap_or(0.0);
                        local = local.max((l - r).abs());
                    }
                }
                acc.max(local)
            });

        println!(
            "[BVP_sci exponential callbacks] residual_diff={:.6e} bc_diff={:.6e} jacobian_diff={:.6e}",
            residual_diff, bc_diff, jacobian_diff
        );

        assert!(
            residual_diff <= 1e-9,
            "exponential residual callbacks should match between Lambdify and C-tcc"
        );
        assert!(
            bc_diff <= 1e-12,
            "exponential BC callbacks should match between Lambdify and C-tcc"
        );
        assert!(
            jacobian_diff <= 1e-9,
            "exponential Jacobian callbacks should match between Lambdify and C-tcc"
        );
    }

    #[test]
    fn bvp_sci_exponential_solution_trajectory_diagnostics_ctcc() {
        if !tcc_is_available() {
            println!(
                "[BVP_sci generated backend] skipping exponential trajectory diagnostics: C-tcc not available"
            );
            return;
        }

        let mut baseline_solver = BVPwrap::new_with_options(exponential_problem_options());
        baseline_solver
            .try_solve()
            .expect("baseline exponential solve should complete");

        let mut generated_solver = BVPwrap::new_with_options(
            exponential_problem_options()
                .with_sparse_atomview_c_tcc(generated_test_artifact_dir("exponential-ctcc")),
        );
        generated_solver
            .try_solve()
            .expect("generated exponential solve should complete");

        let baseline_solution = baseline_solver
            .get_result()
            .expect("baseline exponential solve should produce solution");
        let generated_solution = generated_solver
            .get_result()
            .expect("generated exponential solve should produce solution");
        let baseline_max_abs = max_abs_matrix(&baseline_solution);
        let generated_max_abs = max_abs_matrix(&generated_solution);
        let solution_diff = max_abs_diff(&baseline_solution, &generated_solution);

        let baseline_mesh = baseline_solver.mesh();
        let generated_mesh = generated_solver.mesh();
        let mesh_diff = baseline_mesh
            .iter()
            .zip(generated_mesh.iter())
            .fold(0.0_f64, |acc, (lhs, rhs)| acc.max((lhs - rhs).abs()));

        let (baseline_jac, baseline_res, _) =
            BVPwrap::new_with_options(exponential_problem_options()).eq_generate();
        let (generated_jac, generated_res, _) = BVPwrap::new_with_options(
            exponential_problem_options()
                .with_sparse_atomview_c_tcc(generated_test_artifact_dir("exponential-ctcc")),
        )
        .try_eq_generate()
        .expect("generated exponential callbacks should prepare on a fresh solver");

        let baseline_x = baseline_solver.x_mesh_col.clone();
        let baseline_y = baseline_solver.result.y.clone();
        let generated_x = generated_solver.x_mesh_col.clone();
        let generated_y = generated_solver.result.y.clone();
        let baseline_p = baseline_solver
            .runtime_param_col()
            .unwrap_or_else(|| faer_col::zeros(0));
        let generated_p = generated_solver
            .runtime_param_col()
            .unwrap_or_else(|| faer_col::zeros(0));

        let baseline_on_baseline = baseline_res(&baseline_x, &baseline_y, &baseline_p);
        let generated_on_baseline = generated_res(&baseline_x, &baseline_y, &generated_p);
        let residual_diff_on_baseline = {
            let mut max_diff = 0.0_f64;
            for i in 0..baseline_on_baseline.nrows() {
                for j in 0..baseline_on_baseline.ncols() {
                    max_diff = max_diff.max(
                        (*baseline_on_baseline.get(i, j) - *generated_on_baseline.get(i, j)).abs(),
                    );
                }
            }
            max_diff
        };

        let baseline_on_generated = baseline_res(&generated_x, &generated_y, &baseline_p);
        let generated_on_generated = generated_res(&generated_x, &generated_y, &generated_p);
        let residual_diff_on_generated = {
            let mut max_diff = 0.0_f64;
            for i in 0..baseline_on_generated.nrows() {
                for j in 0..baseline_on_generated.ncols() {
                    max_diff = max_diff.max(
                        (*baseline_on_generated.get(i, j) - *generated_on_generated.get(i, j))
                            .abs(),
                    );
                }
            }
            max_diff
        };

        let baseline_jacobian_on_baseline = baseline_jac
            .expect("baseline jacobian callback should exist")(
            &baseline_x,
            &baseline_y,
            &baseline_p,
        )
        .0;
        let generated_jacobian_on_baseline = generated_jac
            .expect("generated jacobian callback should exist")(
            &baseline_x,
            &baseline_y,
            &generated_p,
        )
        .0;
        let jacobian_diff_on_baseline = baseline_jacobian_on_baseline
            .iter()
            .zip(generated_jacobian_on_baseline.iter())
            .fold(0.0_f64, |acc, (lhs, rhs)| {
                let mut local = 0.0_f64;
                for row in 0..lhs.nrows() {
                    for col in 0..lhs.ncols() {
                        let l = lhs.get(row, col).copied().unwrap_or(0.0);
                        let r = rhs.get(row, col).copied().unwrap_or(0.0);
                        local = local.max((l - r).abs());
                    }
                }
                acc.max(local)
            });

        println!(
            "[BVP_sci exponential trajectory] baseline(status={}, niter={}, nodes={}, max_abs={:.6e}) generated(status={}, niter={}, nodes={}, max_abs={:.6e}) solution_diff={:.6e} mesh_diff={:.6e} residual_diff_on_baseline={:.6e} residual_diff_on_generated={:.6e} jacobian_diff_on_baseline={:.6e}",
            baseline_solver.result.status,
            baseline_solver.result.niter,
            baseline_solver.result.x.nrows(),
            baseline_max_abs,
            generated_solver.result.status,
            generated_solver.result.niter,
            generated_solver.result.x.nrows(),
            generated_max_abs,
            solution_diff,
            mesh_diff,
            residual_diff_on_baseline,
            residual_diff_on_generated,
            jacobian_diff_on_baseline,
        );
    }
}
