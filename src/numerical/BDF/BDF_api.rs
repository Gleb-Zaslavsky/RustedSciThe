//! # BDF ODE Solver API
//!
//! High-level interface for solving ordinary differential equations using the
//! Backward Differentiation Formula (BDF) method with symbolic expression support.
//!
//! ## Overview
//!
//! This module provides a user-friendly API for solving stiff and non-stiff ODEs
//! by combining symbolic expression parsing with the robust BDF numerical solver.
//! The solver automatically generates analytical Jacobians from symbolic expressions,
//! significantly improving performance and accuracy for stiff systems.
//!
//! ## Key Features
//!
//! - **Symbolic Integration**: Parse string expressions into ODEs
//! - **Automatic Jacobian**: Generate analytical Jacobians from symbolic expressions
//! - **Adaptive Methods**: Variable order (1-5) and step size control
//! - **Stop Conditions**: Terminate integration when variables reach target values
//! - **Result Export**: Save results to CSV and generate plots
//! - **Comprehensive Testing**: Extensive test suite with analytical comparisons
//!
//! ## Main Components
//!
//! ### ODEsolver
//! The primary struct that orchestrates the entire solving process:
//! - Parses symbolic expressions into numerical functions
//! - Generates analytical Jacobians automatically
//! - Manages BDF solver instance and integration loop
//! - Handles result storage and export
//!
//! ### Key Methods
//! - `new()`: Create solver with problem parameters
//! - `solve()`: Complete integration from t0 to t_bound
//! - `set_stop_condition()`: Set early termination conditions
//! - `get_result()`: Retrieve time and solution arrays
//! - `plot_result()`: Generate solution plots
//! - `save_result()`: Export results to CSV
//!
//! ## Usage Example
//!
//! ```rust
//! use crate::numerical::BDF::BDF_api::ODEsolver;
//! use crate::symbolic::symbolic_engine::Expr;
//! use nalgebra::DVector;
//!
//! // Define Van der Pol oscillator: y1' = y2, y2' = μ(1-y1²)y2 - y1
//! let eq1 = Expr::parse_expression("y2");
//! let eq2 = Expr::parse_expression("5*(1-y1*y1)*y2 - y1");
//! let eq_system = vec![eq1, eq2];
//! let values = vec!["y1".to_string(), "y2".to_string()];
//!
//! // Create solver
//! let mut solver = ODEsolver::new(
//!     eq_system,                    // System of ODEs
//!     values,                       // Variable names
//!     "t".to_string(),             // Independent variable
//!     "BDF".to_string(),           // Method
//!     0.0,                         // t0
//!     DVector::from_vec(vec![2.0, 0.0]), // y0
//!     10.0,                        // t_bound
//!     0.01,                        // max_step
//!     1e-6,                        // rtol
//!     1e-8,                        // atol
//!     None,                        // jac_sparsity
//!     false,                       // vectorized
//!     None,                        // first_step
//! );
//!
//! // Solve and get results
//! solver.solve();
//! let (t_result, y_result) = solver.get_result();
//!
//! // Optional: plot and save results
//! solver.plot_result();
//! solver.save_result().unwrap();
//! ```
//!
//! ## Advanced Features
//!
//! ### Stop Conditions
//! Terminate integration when variables reach specific values:
//! ```rust, ignore
//! let mut stop_condition = HashMap::new();
//! stop_condition.insert("y1".to_string(), 0.0);
//! solver.set_stop_condition(stop_condition);
//! ```
//!
//! ### Symbolic Expression Support
//! The solver supports complex mathematical expressions:
//! - Basic operations: `+`, `-`, `*`, `/`, `^`
//! - Functions: `sin`, `cos`, `exp`, `log`, `sqrt`
//! - Variables: Any alphanumeric identifier
//! - Constants: Numerical values
//!
//! ### Automatic Jacobian Generation
//! The solver automatically computes analytical Jacobians ∂f/∂y from symbolic
//! expressions, which is crucial for:
//! - **Stiff systems**: Implicit methods require accurate Jacobians
//! - **Performance**: Analytical Jacobians are much faster than finite differences
//! - **Accuracy**: Exact derivatives improve Newton convergence
//!
//! ## Implementation Notes
//!
//! ### Matrix Flattening Strategy
//! The solver uses an efficient matrix flattening approach for result storage:
//! ```rust, ignore
//! // Convert Vec<DVector<f64>> to DMatrix<f64>
//! let mut flat_vec: Vec<f64> = Vec::new();
//! for vector in y.iter() {
//!     flat_vec.extend(vector);
//! }
//! let y_res = DMatrix::from_vec(cols, rows, flat_vec).transpose();
//! ```
//!
//! ### Status Management
//! Integration status is tracked throughout the process:
//! - `"running"`: Integration in progress
//! - `"finished"`: Successfully reached t_bound
//! - `"failed"`: Integration failed (step size too small, etc.)
//! - `"stopped_by_condition"`: Terminated by user-defined stop condition
//!
//! ### Error Handling
//! The solver implements robust error handling:
//! - Graceful degradation when Newton iteration fails
//! - Automatic step size reduction for difficult regions
//! - Clear error messages for debugging

use crate::numerical::BDF::BDF_solver::{BDF, BdfJacobian, BdfLinearBackend};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp::{
    IvpBackendError, IvpSymbolicAssemblyBackend, SharedIvpParameterValues,
    SymbolicIvpProblemOptions,
};
use crate::symbolic::symbolic_ivp_generated::{
    DenseIvpGeneratedBackendMode, IvpBackendStatistics, SymbolicIvpGeneratedBackendConfig,
    prepare_generated_symbolic_ivp_problem, prepare_generated_symbolic_ivp_residual_problem,
};
extern crate nalgebra as na;
use crate::Utils::plots::plots;
use crate::numerical::BDF::common::NumberOrVec;
use na::{DMatrix, DVector};

use csv::Writer;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

type BdfNativeJacobianFactory =
    dyn Fn(Option<SharedIvpParameterValues>) -> Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian>;
type BdfNativeRhs = Arc<dyn Fn(f64, &DVector<f64>) -> DVector<f64> + Send + Sync>;
type BdfNativeDenseJac = Arc<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64> + Send + Sync>;

/// Grouped setup for one symbolic BDF solve.
#[derive(Clone)]
pub struct BdfSolverOptions {
    pub eq_system: Vec<Expr>,
    pub values: Vec<String>,
    pub arg: String,
    pub method: String,
    pub t0: f64,
    pub y0: DVector<f64>,
    pub t_bound: f64,
    pub max_step: f64,
    pub rtol: f64,
    pub atol: f64,
    pub jac_sparsity: Option<DMatrix<f64>>,
    pub vectorized: bool,
    pub first_step: Option<f64>,
    pub max_bdf_order: usize,
    pub equation_parameters: Option<Vec<String>>,
    pub equation_parameter_values: Option<DVector<f64>>,
    pub generated_backend_config: SymbolicIvpGeneratedBackendConfig,
    pub symbolic_assembly_backend: IvpSymbolicAssemblyBackend,
}

impl BdfSolverOptions {
    /// Creates grouped BDF options.
    pub fn new(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        method: String,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        max_step: f64,
        rtol: f64,
        atol: f64,
        jac_sparsity: Option<DMatrix<f64>>,
        vectorized: bool,
        first_step: Option<f64>,
    ) -> Self {
        Self {
            eq_system,
            values,
            arg,
            method,
            t0,
            y0,
            t_bound,
            max_step,
            rtol,
            atol,
            jac_sparsity,
            vectorized,
            first_step,
            max_bdf_order: 5,
            equation_parameters: None,
            equation_parameter_values: None,
            generated_backend_config: SymbolicIvpGeneratedBackendConfig::defaults(),
            symbolic_assembly_backend: IvpSymbolicAssemblyBackend::ExprLegacy,
        }
    }

    /// Applies one explicit generated-backend config.
    pub fn with_generated_backend_config(
        mut self,
        config: SymbolicIvpGeneratedBackendConfig,
    ) -> Self {
        self.generated_backend_config = config;
        self
    }

    /// Selects symbolic Jacobian assembly backend for generated IVP preparation.
    pub fn with_symbolic_assembly_backend(mut self, backend: IvpSymbolicAssemblyBackend) -> Self {
        self.symbolic_assembly_backend = backend;
        self
    }

    /// Declares symbolic parameter names used by the IVP right-hand side.
    pub fn with_equation_parameters(mut self, parameters: Vec<String>) -> Self {
        self.equation_parameters = Some(parameters);
        self
    }

    /// Installs initial numeric values for declared symbolic parameters.
    pub fn with_equation_parameter_values(mut self, values: DVector<f64>) -> Self {
        self.equation_parameter_values = Some(values);
        self
    }

    /// Caps adaptive BDF order selection.
    ///
    /// Valid values are `1..=5`, matching the tested variable-order BDF range.
    pub fn with_max_bdf_order(mut self, max_bdf_order: usize) -> Self {
        self.max_bdf_order = max_bdf_order;
        self
    }

    /// Applies one high-level dense generated backend mode.
    pub fn with_dense_generated_backend_mode(mut self, mode: DenseIvpGeneratedBackendMode) -> Self {
        let mut config = SymbolicIvpGeneratedBackendConfig::from_mode(mode);
        config.resolver = self.generated_backend_config.resolver.clone();
        config.aot_options = self.generated_backend_config.aot_options;
        config.aot_codegen_backend = self.generated_backend_config.aot_codegen_backend;
        config.aot_c_compiler = self.generated_backend_config.aot_c_compiler.clone();
        config.output_parent_dir = self.generated_backend_config.output_parent_dir.clone();
        config.crate_name_override = self.generated_backend_config.crate_name_override.clone();
        config.module_name_override = self.generated_backend_config.module_name_override.clone();
        self.generated_backend_config = config;
        self
    }

    /// Uses compiled dense IVP path via `C + tcc` when startup latency matters most.
    ///
    /// Practical note:
    /// for `BDF` this is an optional optimization, not a universal default.
    /// Many BDF scenarios stay residual-dominated, so benchmark against
    /// `Lambdify` before assuming the generated path will win.
    pub fn with_dense_generated_backend_c_tcc(self, output_parent_dir: impl Into<PathBuf>) -> Self {
        self.with_generated_backend_config(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_c_tcc(),
        )
    }

    /// Uses compiled dense IVP path via `C + gcc` when runtime throughput matters more.
    ///
    /// Practical note:
    /// this is the runtime-oriented dense IVP option. It is worth trying on
    /// larger repeated runs, but for `BDF` the dominant cost is often still the
    /// residual path rather than Jacobian generation itself.
    pub fn with_dense_generated_backend_c_gcc(self, output_parent_dir: impl Into<PathBuf>) -> Self {
        self.with_generated_backend_config(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_c_gcc(),
        )
    }

    /// Uses compiled dense IVP path via Zig.
    pub fn with_dense_generated_backend_zig(self, output_parent_dir: impl Into<PathBuf>) -> Self {
        self.with_generated_backend_config(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_zig(),
        )
    }

    /// Recommended generated-backend preset for dense IVP repeated solves.
    ///
    /// For `BDF`, treat this as a "benchmark me on your problem" preset rather
    /// than a blanket replacement for `Lambdify`.
    pub fn with_dense_generated_backend_for_repeated_solves(
        self,
        output_parent_dir: impl Into<PathBuf>,
    ) -> Self {
        self.with_generated_backend_config(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .for_repeated_solves(),
        )
    }
}

/// High-level ODE solver interface with symbolic expression support.
///
/// This struct provides a complete solution for solving ODEs defined as symbolic
/// expressions, automatically generating numerical functions and analytical Jacobians.
///
/// # Workflow
/// 1. **Setup**: Define ODE system as symbolic expressions
/// 2. **Generation**: Convert expressions to numerical functions and Jacobians
/// 3. **Integration**: Use BDF method with adaptive order/step control
/// 4. **Results**: Store, plot, and export solution data
///
/// # Key Features
/// - Automatic Jacobian generation from symbolic expressions
/// - Adaptive BDF method (orders 1-5) for stiff problems
/// - Stop conditions for event detection
/// - Built-in plotting and CSV export capabilities
pub struct ODEsolver {
    /// System of ODEs as symbolic expressions (e.g., ["y2", "-y1"])
    eq_system: Vec<Expr>,
    /// Variable names corresponding to solution components (e.g., ["y1", "y2"])
    values: Vec<String>,
    /// Independent variable name (typically "t" for time)
    arg: String,
    /// Numerical method identifier ("BDF" for this implementation)
    method: String,
    /// Initial time t₀
    t0: f64,
    /// Initial solution vector y₀
    y0: DVector<f64>,
    /// Final integration time
    t_bound: f64,
    /// Maximum allowed step size
    max_step: f64,
    /// Relative error tolerance
    rtol: f64,
    /// Absolute error tolerance
    atol: f64,
    /// Optional Jacobian sparsity pattern (not currently used)
    #[allow(dead_code)]
    jac_sparsity: Option<DMatrix<f64>>,
    /// Whether the ODE function supports vectorized evaluation
    vectorized: bool,
    /// Optional initial step size (auto-selected if None)
    first_step: Option<f64>,
    /// Maximum adaptive BDF order allowed for the low-level BDF engine.
    max_bdf_order: usize,

    /// Current integration status: "running", "finished", "failed", "stopped_by_condition"
    status: String,
    /// Internal BDF solver instance
    Solver_instance: BDF,
    /// Optional error message from failed integration
    message: Option<String>,

    /// Time points of computed solution
    t_result: DVector<f64>,
    /// Solution matrix: rows = time points, columns = variables
    y_result: DMatrix<f64>,
    /// Optional stop conditions: variable_name → target_value
    stop_condition: Option<HashMap<String, f64>>,
    /// Optional symbolic equation parameters used by `f(t, y, p)`.
    equation_parameters: Option<Vec<String>>,
    /// Current numeric values for `equation_parameters`.
    equation_parameter_values: Option<DVector<f64>>,
    /// Shared parameter storage reused by params-aware symbolic closures.
    parameter_values_handle: Option<SharedIvpParameterValues>,
    /// Whether the current symbolic backend has already been prepared.
    backend_prepared: bool,
    /// High-level generated-backend orchestration config reused across solves.
    generated_backend_config: SymbolicIvpGeneratedBackendConfig,
    symbolic_assembly_backend: IvpSymbolicAssemblyBackend,
    statistics: Arc<Mutex<IvpBackendStatistics>>,
    /// Optional factory for replacing the default dense Newton linear backend
    /// after each generated BDF instance is initialized.
    bdf_linear_backend_factory: Option<Box<dyn Fn() -> Box<dyn BdfLinearBackend>>>,
    /// Optional factory for replacing the dense generated Jacobian callback
    /// with a native sparse/banded Jacobian callback.
    bdf_native_jacobian_factory: Option<Box<BdfNativeJacobianFactory>>,
    /// Optional pure numerical RHS callback `f(t, y)`.
    native_rhs: Option<BdfNativeRhs>,
    /// Optional pure numerical dense Jacobian callback `df/dy`.
    native_jacobian: Option<BdfNativeDenseJac>,
}
impl ODEsolver {
    /// Creates a new ODE solver with the specified parameters.
    ///
    /// # Parameters
    /// * `eq_system` - Vector of symbolic expressions defining dy/dt = f(t,y)
    /// * `values` - Variable names corresponding to solution components
    /// * `arg` - Independent variable name (usually "t")
    /// * `method` - Solver method ("BDF" for this implementation)
    /// * `t0` - Initial time
    /// * `y0` - Initial solution vector
    /// * `t_bound` - Final integration time
    /// * `max_step` - Maximum step size
    /// * `rtol` - Relative tolerance for error control
    /// * `atol` - Absolute tolerance for error control
    /// * `jac_sparsity` - Optional Jacobian sparsity pattern
    /// * `vectorized` - Whether ODE function supports vectorized calls
    /// * `first_step` - Optional initial step size
    ///
    /// # Returns
    /// New ODEsolver instance ready for integration
    ///
    /// # Example
    /// ```rust, ignore
    /// let eq_system = vec![Expr::parse_expression("y2"),
    ///                      Expr::parse_expression("-y1")];
    /// let values = vec!["y1".to_string(), "y2".to_string()];
    /// let solver = ODEsolver::new(
    ///     eq_system, values, "t".to_string(), "BDF".to_string(),
    ///     0.0, DVector::from_vec(vec![1.0, 0.0]), 10.0,
    ///     0.01, 1e-6, 1e-8, None, false, None
    /// );
    /// ```
    pub fn new(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        method: String,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
        max_step: f64,
        rtol: f64,
        atol: f64,
        jac_sparsity: Option<DMatrix<f64>>,
        vectorized: bool,
        first_step: Option<f64>,
    ) -> Self {
        let New = BDF::new();

        ODEsolver {
            eq_system,
            values,
            arg,
            method,
            t0,
            y0,
            t_bound,
            max_step,
            rtol,
            atol,

            jac_sparsity,
            vectorized,
            first_step,
            max_bdf_order: 5,
            status: "running".to_string(),
            Solver_instance: New,
            message: None,

            t_result: DVector::zeros(1),
            y_result: DMatrix::zeros(1, 1),
            stop_condition: None,
            equation_parameters: None,
            equation_parameter_values: None,
            parameter_values_handle: None,
            backend_prepared: false,
            generated_backend_config: SymbolicIvpGeneratedBackendConfig::defaults(),
            symbolic_assembly_backend: IvpSymbolicAssemblyBackend::ExprLegacy,
            statistics: Arc::new(Mutex::new(IvpBackendStatistics::default())),
            bdf_linear_backend_factory: None,
            bdf_native_jacobian_factory: None,
            native_rhs: None,
            native_jacobian: None,
        }
    }

    /// Preferred grouped setup path for symbolic BDF solves.
    pub fn new_with_options(options: BdfSolverOptions) -> Self {
        let mut solver = Self::new(
            options.eq_system,
            options.values,
            options.arg,
            options.method,
            options.t0,
            options.y0,
            options.t_bound,
            options.max_step,
            options.rtol,
            options.atol,
            options.jac_sparsity,
            options.vectorized,
            options.first_step,
        )
        .with_generated_backend_config(options.generated_backend_config);
        solver.max_bdf_order = options.max_bdf_order;
        solver.equation_parameters = options.equation_parameters;
        solver.equation_parameter_values = options.equation_parameter_values;
        solver.symbolic_assembly_backend = options.symbolic_assembly_backend;
        solver
    }

    /// Installs one high-level generated-backend orchestration config.
    pub fn set_generated_backend_config(&mut self, config: SymbolicIvpGeneratedBackendConfig) {
        self.generated_backend_config = config;
        self.backend_prepared = false;
    }

    /// Returns the current generated-backend orchestration config.
    pub fn generated_backend_config(&self) -> &SymbolicIvpGeneratedBackendConfig {
        &self.generated_backend_config
    }

    pub fn symbolic_assembly_backend(&self) -> IvpSymbolicAssemblyBackend {
        self.symbolic_assembly_backend
    }

    pub fn set_symbolic_assembly_backend(&mut self, backend: IvpSymbolicAssemblyBackend) {
        self.symbolic_assembly_backend = backend;
        self.backend_prepared = false;
    }

    pub fn get_statistics(&self) -> IvpBackendStatistics {
        self.statistics
            .lock()
            .expect("IVP statistics lock poisoned")
            .clone()
    }

    pub fn statistics_report(&self) -> String {
        self.get_statistics().table_report()
    }

    pub fn bdf_max_order_cap(&self) -> usize {
        self.Solver_instance.max_order_cap()
    }

    pub fn bdf_current_order(&self) -> usize {
        self.Solver_instance.current_order()
    }

    pub fn bdf_equal_step_count(&self) -> usize {
        self.Solver_instance.equal_step_count()
    }

    /// Installs a factory for the BDF Newton linear backend.
    ///
    /// `ODEsolver::try_generate` creates a fresh low-level BDF instance, so a
    /// factory is used instead of a single backend object.  This is the bridge
    /// LSODE2 will use for sparse/banded Newton solves while preserving the
    /// existing symbolic/generated IVP setup path.
    pub fn set_bdf_linear_backend_factory<F>(&mut self, factory: F)
    where
        F: Fn() -> Box<dyn BdfLinearBackend> + 'static,
    {
        self.bdf_linear_backend_factory = Some(Box::new(factory));
        self.backend_prepared = false;
    }

    /// Builder-style alias for [`Self::set_bdf_linear_backend_factory`].
    pub fn with_bdf_linear_backend_factory<F>(mut self, factory: F) -> Self
    where
        F: Fn() -> Box<dyn BdfLinearBackend> + 'static,
    {
        self.set_bdf_linear_backend_factory(factory);
        self
    }

    /// Installs a factory for native BDF Jacobian evaluators.
    ///
    /// The regular IVP path still prepares dense Jacobian closures. This hook
    /// lets LSODE2 override that closure with a sparse triplet or banded
    /// evaluator after the low-level BDF instance has been initialized.
    pub fn set_bdf_native_jacobian_factory<F>(&mut self, factory: F)
    where
        F: Fn(
                Option<SharedIvpParameterValues>,
            ) -> Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian>
            + 'static,
    {
        self.bdf_native_jacobian_factory = Some(Box::new(factory));
        self.backend_prepared = false;
    }

    /// Builder-style alias for [`Self::set_bdf_native_jacobian_factory`].
    pub fn with_bdf_native_jacobian_factory<F>(mut self, factory: F) -> Self
    where
        F: Fn(
                Option<SharedIvpParameterValues>,
            ) -> Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian>
            + 'static,
    {
        self.set_bdf_native_jacobian_factory(factory);
        self
    }

    /// Installs pure numerical ODE callbacks for BDF.
    ///
    /// If `jac` is `None`, BDF falls back to finite-difference Jacobians.
    pub fn set_native_ode_callbacks<F, J>(&mut self, rhs: F, jac: Option<J>)
    where
        F: Fn(f64, &DVector<f64>) -> DVector<f64> + Send + Sync + 'static,
        J: Fn(f64, &DVector<f64>) -> DMatrix<f64> + Send + Sync + 'static,
    {
        self.native_rhs = Some(Arc::new(rhs));
        self.native_jacobian = jac.map(|j| Arc::new(j) as BdfNativeDenseJac);
        self.backend_prepared = false;
    }

    /// Builder-style generated backend setup.
    pub fn with_generated_backend_config(
        mut self,
        config: SymbolicIvpGeneratedBackendConfig,
    ) -> Self {
        self.set_generated_backend_config(config);
        self
    }

    /// Applies one high-level dense generated backend mode.
    pub fn set_dense_generated_backend_mode(&mut self, mode: DenseIvpGeneratedBackendMode) {
        let mut config = SymbolicIvpGeneratedBackendConfig::from_mode(mode);
        config.resolver = self.generated_backend_config.resolver.clone();
        config.aot_options = self.generated_backend_config.aot_options;
        config.aot_codegen_backend = self.generated_backend_config.aot_codegen_backend;
        config.aot_c_compiler = self.generated_backend_config.aot_c_compiler.clone();
        config.output_parent_dir = self.generated_backend_config.output_parent_dir.clone();
        config.crate_name_override = self.generated_backend_config.crate_name_override.clone();
        config.module_name_override = self.generated_backend_config.module_name_override.clone();
        self.set_generated_backend_config(config);
    }

    /// Builder-style preset for the dense generated backend mode.
    pub fn with_dense_generated_backend_mode(mut self, mode: DenseIvpGeneratedBackendMode) -> Self {
        self.set_dense_generated_backend_mode(mode);
        self
    }

    /// Uses compiled dense IVP path via `C + tcc` when startup latency matters most.
    ///
    /// In `BDF` this is primarily a low-startup native option to compare
    /// against `Lambdify`, not an always-better default.
    pub fn set_dense_generated_backend_c_tcc(&mut self, output_parent_dir: impl Into<PathBuf>) {
        self.set_generated_backend_config(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_c_tcc(),
        );
    }

    /// Uses compiled dense IVP path via `C + gcc` for runtime-oriented repeated solves.
    ///
    /// Prefer this only when you expect enough repeated dense residual work to
    /// amortize native build/setup cost.
    pub fn set_dense_generated_backend_c_gcc(&mut self, output_parent_dir: impl Into<PathBuf>) {
        self.set_generated_backend_config(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_c_gcc(),
        );
    }

    /// Uses compiled dense IVP path via Zig.
    pub fn set_dense_generated_backend_zig(&mut self, output_parent_dir: impl Into<PathBuf>) {
        self.set_generated_backend_config(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_zig(),
        );
    }

    /// Recommended generated-backend preset for dense IVP repeated solves.
    ///
    /// `BDF` often remains residual-dominated, so keep `Lambdify` in mind as a
    /// strong baseline when end-to-end latency is the priority.
    pub fn set_dense_generated_backend_for_repeated_solves(
        &mut self,
        output_parent_dir: impl Into<PathBuf>,
    ) {
        self.set_generated_backend_config(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .for_repeated_solves(),
        );
    }

    /// Builder-style alias for `C + tcc` dense generated backend setup.
    pub fn with_dense_generated_backend_c_tcc(
        mut self,
        output_parent_dir: impl Into<PathBuf>,
    ) -> Self {
        self.set_dense_generated_backend_c_tcc(output_parent_dir);
        self
    }

    /// Builder-style alias for `C + gcc` dense generated backend setup.
    pub fn with_dense_generated_backend_c_gcc(
        mut self,
        output_parent_dir: impl Into<PathBuf>,
    ) -> Self {
        self.set_dense_generated_backend_c_gcc(output_parent_dir);
        self
    }

    /// Builder-style alias for Zig dense generated backend setup.
    pub fn with_dense_generated_backend_zig(
        mut self,
        output_parent_dir: impl Into<PathBuf>,
    ) -> Self {
        self.set_dense_generated_backend_zig(output_parent_dir);
        self
    }

    /// Builder-style alias for the recommended repeated-solve IVP preset.
    pub fn with_dense_generated_backend_for_repeated_solves(
        mut self,
        output_parent_dir: impl Into<PathBuf>,
    ) -> Self {
        self.set_dense_generated_backend_for_repeated_solves(output_parent_dir);
        self
    }

    /// Sets stop conditions for early termination of integration.
    ///
    /// Integration will stop when any variable reaches its target value
    /// within the absolute tolerance.
    ///
    /// # Parameters
    /// * `stop_condition` - Map of variable names to target values
    ///
    /// # Example
    /// ```rust, ignore
    /// let mut stop_condition = HashMap::new();
    /// stop_condition.insert("y1".to_string(), 0.0);
    /// solver.set_stop_condition(stop_condition);
    /// ```
    pub fn set_stop_condition(&mut self, stop_condition: HashMap<String, f64>) {
        self.stop_condition = Some(stop_condition);
    }

    /// Declares symbolic parameter names used by the IVP right-hand side.
    pub fn set_equation_parameters(&mut self, params: Option<&[&str]>) {
        self.equation_parameters =
            params.map(|params| params.iter().map(|p| (*p).to_string()).collect());
        self.backend_prepared = false;
    }

    /// Updates numeric values of symbolic equation parameters without recompiling
    /// already prepared closures.
    pub fn set_parameter_values(&mut self, values: DVector<f64>) -> Result<(), IvpBackendError> {
        if let Some(parameters) = self.equation_parameters.as_ref() {
            if parameters.len() != values.len() {
                return Err(IvpBackendError::ParameterCountMismatch {
                    expected: parameters.len(),
                    actual: values.len(),
                });
            }
        } else if !values.is_empty() {
            return Err(IvpBackendError::ParameterCountMismatch {
                expected: 0,
                actual: values.len(),
            });
        }

        if let Some(handle) = self.parameter_values_handle.as_ref() {
            let mut slot = handle
                .write()
                .expect("shared IVP parameter state lock poisoned");
            *slot = values.clone();
        }
        self.equation_parameter_values = Some(values);
        Ok(())
    }

    /// Checks if any stop condition has been met.
    ///
    /// # Parameters
    /// * `y` - Current solution vector
    ///
    /// # Returns
    /// `true` if any variable has reached its target value within tolerance
    fn check_stop_condition(&self, y: &DVector<f64>) -> bool {
        if let Some(ref conditions) = self.stop_condition {
            for (var_name, target_value) in conditions {
                if let Some(var_index) = self.values.iter().position(|v| v == var_name) {
                    let current_value = y[var_index];
                    if (current_value - target_value).abs() <= self.atol {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Generates numerical functions and Jacobian from symbolic expressions.
    ///
    /// This method:
    /// 1. Creates a Jacobian instance for symbolic processing
    /// 2. Converts symbolic expressions to numerical functions
    /// 3. Generates analytical Jacobian matrix function
    /// 4. Initializes the BDF solver with these functions
    ///
    /// # Implementation Details
    /// Uses the symbolic engine to automatically compute ∂f/∂y analytically,
    /// which is crucial for stiff problem performance.
    pub fn try_generate(&mut self) -> Result<(), IvpBackendError> {
        if self.native_rhs.is_some() {
            return self.try_generate_native_numeric();
        }
        let start = Instant::now();
        let mut options = SymbolicIvpProblemOptions::new();
        if let Some(parameters) = self.equation_parameters.clone() {
            options = options.with_equation_parameters(parameters);
        }
        if let Some(values) = self.equation_parameter_values.clone() {
            options = options.with_equation_parameter_values(values);
        }
        options = options.with_symbolic_assembly_backend(self.symbolic_assembly_backend);

        if self.bdf_native_jacobian_factory.is_some() {
            return self.try_generate_with_native_jacobian(start, options);
        }

        let prepared = prepare_generated_symbolic_ivp_problem(
            self.eq_system.clone(),
            self.values.clone(),
            self.arg.clone(),
            options.with_aot_options(self.generated_backend_config.aot_options),
            self.generated_backend_config.clone(),
        )
        .map_err(|err| IvpBackendError::GeneratedBackendFailure {
            message: err.to_string(),
        })?;
        self.generated_backend_config.resolver = prepared.updated_resolver.clone();
        let prepared_problem = prepared.into_problem();
        let parameter_values_handle = prepared_problem.parameter_values_handle();
        let fun = prepared_problem.residual;
        let jac = prepared_problem.jacobian;
        let stats_for_fun = Arc::clone(&self.statistics);
        let wrapped_fun = Box::new(move |t: f64, y: &DVector<f64>| -> DVector<f64> {
            let start = Instant::now();
            let out = fun(t, y);
            stats_for_fun
                .lock()
                .expect("IVP statistics lock poisoned")
                .record_residual_duration(start.elapsed());
            out
        });
        let stats_for_jac = Arc::clone(&self.statistics);
        let wrapped_jac = Box::new(move |t: f64, y: &DVector<f64>| -> DMatrix<f64> {
            let start = Instant::now();
            let out = jac(t, y);
            stats_for_jac
                .lock()
                .expect("IVP statistics lock poisoned")
                .record_jacobian_duration(start.elapsed());
            out
        });
        self.parameter_values_handle = parameter_values_handle.clone();

        if self.method == "BDF" {
            let mut Solver_instance = BDF::new();
            Solver_instance.set_max_order_cap(self.max_bdf_order);
            Solver_instance.set_initial(
                wrapped_fun,
                self.t0,
                self.y0.clone(),
                self.t_bound,
                self.max_step,
                NumberOrVec::Number(self.rtol),
                NumberOrVec::Number(self.atol),
                Some(wrapped_jac),
                None,
                self.vectorized,
                self.first_step,
            );
            if let Some(factory) = self.bdf_native_jacobian_factory.as_ref() {
                Solver_instance.set_native_jacobian(
                    self.timed_native_jacobian(factory(parameter_values_handle.clone())),
                );
            }
            if let Some(factory) = self.bdf_linear_backend_factory.as_ref() {
                Solver_instance.set_linear_backend(factory());
            }
            self.Solver_instance = Solver_instance;
        }
        self.backend_prepared = true;
        self.statistics
            .lock()
            .expect("IVP statistics lock poisoned")
            .record_backend_prepare_duration(start.elapsed());
        Ok(())
    }

    fn try_generate_native_numeric(&mut self) -> Result<(), IvpBackendError> {
        let start = Instant::now();
        if self.method == "BDF" {
            let rhs = self
                .native_rhs
                .clone()
                .expect("native_rhs must exist in native numeric generation path");
            let stats_for_fun = Arc::clone(&self.statistics);
            let wrapped_fun = Box::new(move |t: f64, y: &DVector<f64>| -> DVector<f64> {
                let start = Instant::now();
                let out = rhs(t, y);
                stats_for_fun
                    .lock()
                    .expect("IVP statistics lock poisoned")
                    .record_residual_duration(start.elapsed());
                out
            });

            let wrapped_jac = self.native_jacobian.clone().map(|jac| {
                let stats_for_jac = Arc::clone(&self.statistics);
                Box::new(move |t: f64, y: &DVector<f64>| -> DMatrix<f64> {
                    let start = Instant::now();
                    let out = jac(t, y);
                    stats_for_jac
                        .lock()
                        .expect("IVP statistics lock poisoned")
                        .record_jacobian_duration(start.elapsed());
                    out
                }) as Box<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64>>
            });

            let mut solver_instance = BDF::new();
            solver_instance.set_max_order_cap(self.max_bdf_order);
            solver_instance.set_initial(
                wrapped_fun,
                self.t0,
                self.y0.clone(),
                self.t_bound,
                self.max_step,
                NumberOrVec::Number(self.rtol),
                NumberOrVec::Number(self.atol),
                wrapped_jac,
                self.jac_sparsity.clone(),
                self.vectorized,
                self.first_step,
            );
            if let Some(factory) = self.bdf_linear_backend_factory.as_ref() {
                solver_instance.set_linear_backend(factory());
            }
            self.Solver_instance = solver_instance;
        }
        self.backend_prepared = true;
        self.statistics
            .lock()
            .expect("IVP statistics lock poisoned")
            .record_backend_prepare_duration(start.elapsed());
        Ok(())
    }

    fn try_generate_with_native_jacobian(
        &mut self,
        start: std::time::Instant,
        options: SymbolicIvpProblemOptions,
    ) -> Result<(), IvpBackendError> {
        let prepared = prepare_generated_symbolic_ivp_residual_problem(
            self.eq_system.clone(),
            self.values.clone(),
            self.arg.clone(),
            options.with_aot_options(self.generated_backend_config.aot_options),
            self.generated_backend_config.clone(),
        )
        .map_err(|err| IvpBackendError::GeneratedBackendFailure {
            message: err.to_string(),
        })?;
        self.generated_backend_config.resolver = prepared.updated_resolver.clone();
        let prepared_problem = prepared.into_problem();
        let parameter_values_handle = prepared_problem.parameter_values_handle();
        let fun = prepared_problem.residual;
        let stats_for_fun = Arc::clone(&self.statistics);
        let wrapped_fun = Box::new(move |t: f64, y: &DVector<f64>| -> DVector<f64> {
            let start = Instant::now();
            let out = fun(t, y);
            stats_for_fun
                .lock()
                .expect("IVP statistics lock poisoned")
                .record_residual_duration(start.elapsed());
            out
        });
        self.parameter_values_handle = parameter_values_handle.clone();

        if self.method == "BDF" {
            let mut solver_instance = BDF::new();
            solver_instance.set_max_order_cap(self.max_bdf_order);
            solver_instance.set_initial(
                wrapped_fun,
                self.t0,
                self.y0.clone(),
                self.t_bound,
                self.max_step,
                NumberOrVec::Number(self.rtol),
                NumberOrVec::Number(self.atol),
                None,
                None,
                self.vectorized,
                self.first_step,
            );
            if let Some(factory) = self.bdf_native_jacobian_factory.as_ref() {
                solver_instance.set_native_jacobian(
                    self.timed_native_jacobian(factory(parameter_values_handle.clone())),
                );
            }
            if let Some(factory) = self.bdf_linear_backend_factory.as_ref() {
                solver_instance.set_linear_backend(factory());
            }
            self.Solver_instance = solver_instance;
        }

        self.backend_prepared = true;
        self.statistics
            .lock()
            .expect("IVP statistics lock poisoned")
            .record_backend_prepare_duration(start.elapsed());
        Ok(())
    }

    fn timed_native_jacobian(
        &self,
        mut jacobian: Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian>,
    ) -> Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian> {
        let stats_for_jac = Arc::clone(&self.statistics);
        Box::new(move |t: f64, y: &DVector<f64>| -> BdfJacobian {
            let start = Instant::now();
            let out = jacobian(t, y);
            stats_for_jac
                .lock()
                .expect("IVP statistics lock poisoned")
                .record_jacobian_duration(start.elapsed());
            out
        })
    }

    pub fn generate(&mut self) {
        self.try_generate()
            .expect("BDF symbolic IVP backend generation should succeed");
    }
    /// Performs a single integration step.
    ///
    /// This method wraps the BDF solver's step implementation and manages
    /// the integration status. It handles:
    /// - Boundary detection (reaching t_bound)
    /// - Error handling and status updates
    /// - Direction checking for integration completion
    ///
    /// # Status Updates
    /// - "finished": Successfully reached t_bound or boundary
    /// - "failed": Step failed (convergence issues, step size too small)
    /// - "running": Integration continues normally
    pub fn step(&mut self) {
        //  let (success, message_) =self.Solver_instance._step_impl();

        // Analogue of step function in https://github.com/scipy/scipy/blob/main/scipy/integrate/_ivp/base.py
        let t = self.Solver_instance.t;
        if t == self.t_bound {
            self.Solver_instance.t_old = Some(t);

            self.status = "finished".to_string();
        } else {
            let (success, message_) = self.Solver_instance._step_impl();
            if let Some(message_str) = message_ {
                self.message = Some(message_str.to_string());
            } else {
                self.message = None;
            }

            if success == false {
                self.status = "failed".to_string();
            } else {
                self.Solver_instance.t_old = Some(t);
                let _status: String = "running".to_string();
                if self.Solver_instance.direction * (self.Solver_instance.t - self.t_bound) >= 0.0 {
                    self.status = "finished".to_string();
                }
            }
        }
    }
    #[warn(unused_assignments)]
    /// Main integration loop that drives the solution from t0 to t_bound.
    ///
    /// This method implements the complete integration algorithm:
    /// 1. **Step Loop**: Repeatedly calls step() until completion
    /// 2. **Status Monitoring**: Tracks integration progress and failures
    /// 3. **Stop Conditions**: Checks user-defined termination criteria
    /// 4. **Data Collection**: Stores solution points for output
    /// 5. **Matrix Assembly**: Converts solution vectors to result matrices
    ///
    /// # Performance Features
    /// - **Efficient Storage**: Uses vector extension for minimal allocations
    /// - **Matrix Flattening**: Optimized conversion from Vec<DVector> to DMatrix
    /// - **Timing**: Reports integration time for performance analysis
    ///
    /// # Matrix Assembly Algorithm
    /// ```text
    /// flat_vec = [y₁(t₁), y₂(t₁), ..., yₙ(t₁), y₁(t₂), y₂(t₂), ..., yₙ(tₘ)]
    /// y_result = reshape(flat_vec, n_vars, n_times).transpose()
    /// ```
    pub fn main_loop(&mut self) -> () {
        // Analogue of https://github.com/scipy/scipy/blob/main/scipy/integrate/_ivp/ivp.py
        let start = Instant::now();
        let mut integr_status: Option<i8> = None;
        let mut y: Vec<DVector<f64>> = Vec::new();
        let mut t: Vec<f64> = Vec::new();
        let mut _i: i64 = 0;
        let (nfev_before, njev_before, nlu_before) = self.Solver_instance.counters();
        while integr_status.is_none() {
            self.step();
            self.statistics
                .lock()
                .expect("IVP statistics lock poisoned")
                .step_calls += 1;
            let _status: i8 = 0;
            _i += 1;
            if self.status == "finished".to_string() {
                integr_status = Some(0)
            } else if self.status == "failed".to_string() {
                integr_status = Some(-1);
                break;
            }
            // Check stop condition before storing solution
            if self.check_stop_condition(&self.Solver_instance.y) {
                self.status = "stopped_by_condition".to_string();
                integr_status = Some(0);
            }

            t.push(self.Solver_instance.t);
            y.push(self.Solver_instance.y.clone());
        }

        let rows = &y.len();
        let cols = &y[0].len();

        let mut flat_vec: Vec<f64> = Vec::new();
        for vector in y.iter() {
            flat_vec.extend(vector)
        }
        let y_res: DMatrix<f64> = DMatrix::from_vec(*cols, *rows, flat_vec).transpose();
        let t_res = DVector::from_vec(t);
        let duration = start.elapsed();
        println!("Program took {} milliseconds to run", duration.as_millis());
        let (nfev_after, njev_after, nlu_after) = self.Solver_instance.counters();
        let mut stats = self
            .statistics
            .lock()
            .expect("IVP statistics lock poisoned");
        stats.record_solve_duration(duration);
        stats.bdf_nfev_total += nfev_after.saturating_sub(nfev_before);
        stats.bdf_njev_total += njev_after.saturating_sub(njev_before);
        stats.bdf_nlu_total += nlu_after.saturating_sub(nlu_before);

        self.t_result = t_res.clone();
        self.y_result = y_res.clone();
    }

    /// Solves the ODE system from t0 to t_bound.
    ///
    /// This is the main entry point that orchestrates the complete solution process:
    /// 1. **Generate**: Convert symbolic expressions to numerical functions
    /// 2. **Integrate**: Run the main integration loop
    ///
    /// After calling this method, use `get_result()` to retrieve the solution.
    ///
    /// # Example
    /// ```rust, ignore
    /// solver.solve();
    /// let (t_result, y_result) = solver.get_result();
    /// ```
    pub fn solve(&mut self) -> () {
        if !self.backend_prepared {
            self.generate();
        }
        self.main_loop();
    }

    /// Generates plots of the solution using the built-in plotting utility.
    ///
    /// Creates time-series plots for all solution variables.
    /// Requires the solution to be computed first via `solve()`.
    pub fn plot_result(&self) -> () {
        plots(
            self.arg.clone(),
            self.values.clone(),
            self.t_result.clone(),
            self.y_result.clone(),
        );
        println!("result plotted");
    }

    /// Returns the computed solution data.
    ///
    /// # Returns
    /// * `DVector<f64>` - Time points
    /// * `DMatrix<f64>` - Solution matrix (rows = time, columns = variables)
    ///
    /// # Example
    /// ```rust, ignore
    /// let (t_result, y_result) = solver.get_result();
    /// println!("Final time: {}", t_result[t_result.len()-1]);
    /// println!("Final solution: {:?}", y_result.row(y_result.nrows()-1));
    /// ```
    pub fn get_result(&self) -> (DVector<f64>, DMatrix<f64>) {
        (self.t_result.clone(), self.y_result.clone())
    }

    /// Returns the current integration status.
    ///
    /// # Possible Values
    /// - `"running"`: Integration in progress
    /// - `"finished"`: Successfully completed
    /// - `"failed"`: Integration failed
    /// - `"stopped_by_condition"`: Terminated by stop condition
    ///
    /// # Returns
    /// Reference to the status string
    pub fn get_status(&self) -> &String {
        &self.status
    }

    /// Saves the solution results to a CSV file.
    ///
    /// The CSV format includes:
    /// - First row: Column headers (time variable + solution variables)
    /// - Subsequent rows: Time points and corresponding solution values
    ///
    /// # File Format
    /// ```csv
    /// t,y1,y2,...
    /// 0.0,1.0,0.0,...
    /// 0.01,0.999,0.01,...
    /// ```
    ///
    /// # Returns
    /// `Result<(), Box<dyn std::error::Error>>` - Success or file I/O error

    pub fn save_result(&self) -> Result<(), Box<dyn std::error::Error>> {
        let path = format!(
            "f:\\RUST\\RustProjects_\\RustedSciThe3\\src\\numerical\\results\\{}+{}.csv",
            self.arg,
            self.values.join("+")
        );
        let mut wtr = Writer::from_path(path)?;

        // Write column titles
        wtr.write_record(&[&self.arg, "values"])?;

        // Write time column
        wtr.write_record(self.t_result.iter().map(|&x| x.to_string()))?;

        // Write y columns
        for (i, col) in self.y_result.column_iter().enumerate() {
            let col_name = format!("{}", &self.values[i]);
            wtr.write_record(&[
                &col_name,
                &col.iter()
                    .map(|&x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            ])?;
        }
        print!("result saved");
        wtr.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic::symbolic_engine::Expr;
    use crate::symbolic::symbolic_ivp_generated::SymbolicIvpAotBuildPolicy;
    use std::collections::HashMap;

    #[test]
    fn bdf_new_with_options_installs_generated_backend_mode() {
        let solver = ODEsolver::new_with_options(
            BdfSolverOptions::new(
                vec![Expr::parse_expression("y")],
                vec!["y".to_string()],
                "t".to_string(),
                "BDF".to_string(),
                0.0,
                DVector::from_vec(vec![1.0]),
                1.0,
                0.1,
                1e-6,
                1e-8,
                None,
                false,
                None,
            )
            .with_dense_generated_backend_mode(DenseIvpGeneratedBackendMode::BuildIfMissingRelease),
        );

        assert_eq!(
            solver.generated_backend_config().build_policy,
            SymbolicIvpAotBuildPolicy::BuildIfMissing {
                profile: crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile::Release
            }
        );
    }

    #[test]
    fn generated_backend_surface_mode_updates_bdf_config() {
        let solver = ODEsolver::new(
            vec![Expr::parse_expression("y")],
            vec!["y".to_string()],
            "t".to_string(),
            "BDF".to_string(),
            0.0,
            DVector::from_vec(vec![1.0]),
            1.0,
            0.1,
            1e-6,
            1e-8,
            None,
            false,
            None,
        )
        .with_dense_generated_backend_mode(DenseIvpGeneratedBackendMode::RequirePrebuilt);

        assert_eq!(
            solver.generated_backend_config().build_policy,
            SymbolicIvpAotBuildPolicy::RequirePrebuilt
        );
    }

    #[test]
    fn bdf_generated_backend_surface_keeps_selected_zig_backend() {
        let solver = ODEsolver::new(
            vec![Expr::parse_expression("y")],
            vec!["y".to_string()],
            "t".to_string(),
            "BDF".to_string(),
            0.0,
            DVector::from_vec(vec![1.0]),
            1.0,
            0.1,
            1e-6,
            1e-8,
            None,
            false,
            None,
        )
        .with_dense_generated_backend_zig("target/generated-ivp-tests")
        .with_dense_generated_backend_mode(DenseIvpGeneratedBackendMode::BuildIfMissingRelease);

        assert_eq!(
            solver.generated_backend_config().aot_codegen_backend,
            crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend::Zig
        );
        assert_eq!(solver.generated_backend_config().aot_c_compiler, None);
    }

    #[test]
    fn bdf_generated_backend_repeated_solves_alias_prefers_c_gcc() {
        let solver = ODEsolver::new(
            vec![Expr::parse_expression("y")],
            vec!["y".to_string()],
            "t".to_string(),
            "BDF".to_string(),
            0.0,
            DVector::from_vec(vec![1.0]),
            1.0,
            0.1,
            1e-6,
            1e-8,
            None,
            false,
            None,
        )
        .with_dense_generated_backend_for_repeated_solves("target/generated-ivp-tests");

        assert_eq!(
            solver.generated_backend_config().aot_codegen_backend,
            crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend::C
        );
        assert_eq!(
            solver.generated_backend_config().aot_c_compiler.as_deref(),
            Some("gcc")
        );
    }

    #[test]
    fn bdf_options_can_set_symbolic_assembly_backend() {
        let solver = ODEsolver::new_with_options(
            BdfSolverOptions::new(
                vec![Expr::parse_expression("y")],
                vec!["y".to_string()],
                "t".to_string(),
                "BDF".to_string(),
                0.0,
                DVector::from_vec(vec![1.0]),
                1.0,
                0.1,
                1e-6,
                1e-8,
                None,
                false,
                None,
            )
            .with_symbolic_assembly_backend(IvpSymbolicAssemblyBackend::AtomView),
        );

        assert_eq!(
            solver.symbolic_assembly_backend(),
            IvpSymbolicAssemblyBackend::AtomView
        );
    }

    #[test]
    fn test_bdf_riccati_equation() {
        // Riccati equation: y' = y^2 - t^2, y(0) = 1
        // Highly nonlinear with known analytical behavior
        let eq1 = Expr::parse_expression("y*y - t*t");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let method = "BDF".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 0.5;
        let max_step = 0.001;
        let rtol = 1e-8;
        let atol = 1e-10;

        let mut solver = ODEsolver::new(
            eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false,
            None,
        );

        solver.solve();
        assert_eq!(solver.get_status(), "finished");

        let (t_result, y_result) = solver.get_result();
        // Verify solution remains bounded and smooth
        for i in 0..t_result.len() {
            assert!(y_result[(i, 0)].is_finite());
            assert!(y_result[(i, 0)] > 0.0); // Should remain positive
        }
    }

    #[test]
    fn test_bdf_van_der_pol_oscillator() {
        // Van der Pol oscillator: y1' = y2, y2' = μ(1-y1^2)y2 - y1
        // Highly nonlinear system with μ = 5 (stiff)
        let eq1 = Expr::parse_expression("y2");
        let eq2 = Expr::parse_expression("5*(1-y1*y1)*y2 - y1");
        let eq_system = vec![eq1, eq2];
        let values = vec!["y1".to_string(), "y2".to_string()];
        let arg = "t".to_string();
        let method = "BDF".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![2.0, 0.0]);
        let t_bound = 5.0;
        let max_step = 0.01;
        let rtol = 1e-6;
        let atol = 1e-8;

        let mut solver = ODEsolver::new(
            eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false,
            None,
        );

        solver.solve();
        assert_eq!(solver.get_status(), "finished");

        let (_, y_result) = solver.get_result();
        // Van der Pol should exhibit limit cycle behavior
        assert!(y_result[(y_result.nrows() - 1, 0)].abs() < 3.0); // Bounded oscillation
    }

    #[test]
    fn test_bdf_bernoulli_equation() {
        // Bernoulli equation: y' + y = y^3, y(0) = 0.5
        // Analytical solution: y = 1/sqrt(3*exp(2*t) + 1)
        let eq1 = Expr::parse_expression("y*y*y - y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let method = "BDF".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![0.5]);
        let t_bound = 0.3;
        let max_step = 0.001;
        let rtol = 1e-8;
        let atol = 1e-10;

        let mut solver = ODEsolver::new(
            eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false,
            None,
        );

        solver.solve();
        assert_eq!(solver.get_status(), "finished");

        let (t_result, y_result) = solver.get_result();

        // Compare with analytical solution at final time
        let t_final = t_result[t_result.len() - 1];
        let y_analytical = 1.0 / (3.0 * (2.0 * t_final).exp() + 1.0).sqrt();
        let y_numerical = y_result[(y_result.nrows() - 1, 0)];

        assert!(
            (y_numerical - y_analytical).abs() < 1e-4,
            "Numerical: {}, Analytical: {}, Error: {}",
            y_numerical,
            y_analytical,
            (y_numerical - y_analytical).abs()
        );
    }

    #[test]
    fn test_bdf_logistic_equation() {
        // Logistic equation: y' = r*y*(1-y/K), y(0) = y0
        // Analytical solution: y = K*y0*exp(r*t)/(K + y0*(exp(r*t) - 1))
        let eq1 = Expr::parse_expression("2*y*(1-y/10)");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let method = "BDF".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 3.0;
        let max_step = 0.01;
        let rtol = 1e-8;
        let atol = 1e-10;

        let mut solver = ODEsolver::new(
            eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false,
            None,
        );

        solver.solve();
        assert_eq!(solver.get_status(), "finished");

        let (t_result, y_result) = solver.get_result();

        // Compare with analytical solution
        let r = 2.0;
        let k = 10.0;
        let y0_val = 1.0;

        for i in 0..t_result.len() {
            let t = t_result[i];
            let y_analytical = k * y0_val * (r * t).exp() / (k + y0_val * ((r * t).exp() - 1.0));
            let y_numerical = y_result[(i, 0)];

            assert!(
                (y_numerical - y_analytical).abs() < 1e-5,
                "At t={}: Numerical: {}, Analytical: {}, Error: {}",
                t,
                y_numerical,
                y_analytical,
                (y_numerical - y_analytical).abs()
            );
        }
    }

    #[test]
    fn test_bdf_pendulum_equation() {
        // Nonlinear pendulum: θ'' + sin(θ) = 0
        // Rewritten as system: θ' = ω, ω' = -sin(θ)
        let eq1 = Expr::parse_expression("omega");
        let eq2 = Expr::parse_expression("-sin(theta)");
        let eq_system = vec![eq1, eq2];
        let values = vec!["theta".to_string(), "omega".to_string()];
        let arg = "t".to_string();
        let method = "BDF".to_string();
        let t0 = 0.0;
        // at t = 0 θ(0)=1, omega(0) = θ'(0)=1
        let y0 = DVector::from_vec(vec![1.0, 0.0]); // Small angle approximation
        let t_bound = 1.0;
        let max_step = 0.001;
        let rtol = 1e-6;
        let atol = 1e-8;

        let mut solver = ODEsolver::new(
            eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false,
            None,
        );

        solver.solve();
        assert_eq!(solver.get_status(), "finished");

        let (_, y_result) = solver.get_result();

        // Energy conservation 0.5 *θ'^2 = C+cos(θ), C = - cos(1)
        // 0.5 *θ'^2 = cos(θ)- cos(1)
        //  so 0.5 *θ'^2 - (cos(θ)- cos(1)) must be close to 0 evarywhere
        println!(
            "1st and last teta {}, {}",
            y_result[(0, 0)],
            y_result[(y_result.nrows() - 1, 0)]
        );
        println!(
            "1st and last omega {}, {}",
            y_result[(0, 1)],
            y_result[(y_result.nrows() - 1, 1)]
        );
        let final_theta = y_result[(y_result.nrows() - 1, 0)];
        let final_omega = y_result[(y_result.nrows() - 1, 1)];
        let final_energy = 0.5 * final_omega.powi(2) - (1.0_f64.cos() - final_theta.cos());

        assert!(
            final_energy.abs() < 1e-3,
            "Energy not conserved: Initial: {}",
            final_energy
        );
    }

    #[test]
    fn test_bdf_lorenz_system() {
        // Lorenz system: x' = σ(y-x), y' = x(ρ-z)-y, z' = xy-βz
        let eq1 = Expr::parse_expression("10*(y-x)");
        let eq2 = Expr::parse_expression("x*(28-z)-y");
        let eq3 = Expr::parse_expression("x*y-8*z/3");
        let eq_system = vec![eq1, eq2, eq3];
        let values = vec!["x".to_string(), "y".to_string(), "z".to_string()];
        let arg = "t".to_string();
        let method = "BDF".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        let t_bound = 5.0;
        let max_step = 0.001;
        let rtol = 1e-8;
        let atol = 1e-10;

        let mut solver = ODEsolver::new(
            eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false,
            None,
        );

        solver.solve();
        assert_eq!(solver.get_status(), "finished");

        let (_, y_result) = solver.get_result();

        // Verify chaotic behavior remains bounded
        for i in 0..y_result.nrows() {
            assert!(y_result[(i, 0)].abs() < 50.0); // x bounded
            assert!(y_result[(i, 1)].abs() < 50.0); // y bounded
            assert!(y_result[(i, 2)] > 0.0 && y_result[(i, 2)] < 50.0); // z positive and bounded
        }
    }

    #[test]
    fn test_bdf_stiff_chemical_reaction() {
        // Stiff chemical kinetics: A -> B -> C
        // y1' = -k1*y1, y2' = k1*y1 - k2*y2, y3' = k2*y2
        // with k1 = 1, k2 = 1000 (stiff)
        let eq1 = Expr::parse_expression("-y1");
        let eq2 = Expr::parse_expression("y1 - 1000*y2");
        let eq3 = Expr::parse_expression("1000*y2");
        let eq_system = vec![eq1, eq2, eq3];
        let values = vec!["y1".to_string(), "y2".to_string(), "y3".to_string()];
        let arg = "t".to_string();
        let method = "BDF".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let t_bound = 2.0;
        let max_step = 0.01;
        let rtol = 1e-6;
        let atol = 1e-8;

        let mut solver = ODEsolver::new(
            eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false,
            None,
        );

        solver.solve();
        assert_eq!(solver.get_status(), "finished");

        let (t_result, y_result) = solver.get_result();

        // Mass conservation: y1 + y2 + y3 = 1
        let final_sum = y_result[(y_result.nrows() - 1, 0)]
            + y_result[(y_result.nrows() - 1, 1)]
            + y_result[(y_result.nrows() - 1, 2)];
        assert!(
            (final_sum - 1.0).abs() < 1e-6,
            "Mass not conserved: {}",
            final_sum
        );

        // At t_bound, y1 should be approximately exp(-t_bound)
        let t_final = t_result[t_result.len() - 1];
        let y1_analytical = (-t_final).exp();
        let y1_numerical = y_result[(y_result.nrows() - 1, 0)];
        assert!((y1_numerical - y1_analytical).abs() < 1e-4);
    }

    // Stop condition tests
    #[test]
    fn test_bdf_stop_condition_single_variable() {
        let eq1 = Expr::parse_expression("y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let method = "BDF".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 10.0;
        let max_step = 0.01;
        let rtol = 1e-6;
        let atol = 1e-3;

        let mut solver = ODEsolver::new(
            eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false,
            None,
        );

        let mut stop_condition = HashMap::new();
        stop_condition.insert("y".to_string(), 2.0);
        solver.set_stop_condition(stop_condition);

        solver.solve();

        assert_eq!(solver.get_status(), "stopped_by_condition");
        let (_, y_result) = solver.get_result();
        let final_y = y_result[(y_result.nrows() - 1, 0)];
        assert!((final_y - 2.0).abs() <= atol);
    }

    #[test]
    fn test_bdf_stop_condition_multiple_variables() {
        let eq1 = Expr::parse_expression("y2");
        let eq2 = Expr::parse_expression("-y1");
        let eq_system = vec![eq1, eq2];
        let values = vec!["y1".to_string(), "y2".to_string()];
        let arg = "t".to_string();
        let method = "BDF".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0, 0.0]);
        let t_bound = 10.0;
        let max_step = 0.01;
        let rtol = 1e-6;
        let atol = 1e-3;

        let mut solver = ODEsolver::new(
            eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false,
            None,
        );

        let mut stop_condition = HashMap::new();
        stop_condition.insert("y1".to_string(), 0.0);
        solver.set_stop_condition(stop_condition);

        solver.solve();

        assert_eq!(solver.get_status(), "stopped_by_condition");
        let (_, y_result) = solver.get_result();
        let final_y1 = y_result[(y_result.nrows() - 1, 0)];
        assert!(final_y1.abs() <= atol);
    }

    #[test]
    fn test_bdf_no_stop_condition() {
        let eq1 = Expr::parse_expression("-y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let method = "BDF".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 1.0;
        let max_step = 0.1;
        let rtol = 1e-6;
        let atol = 1e-6;

        let mut solver = ODEsolver::new(
            eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false,
            None,
        );

        solver.solve();

        assert_eq!(solver.get_status(), "finished");
        let (t_result, _) = solver.get_result();
        let final_t = t_result[t_result.len() - 1];
        assert!((final_t - t_bound).abs() <= max_step);
    }

    #[test]
    fn test_bdf_stop_condition_nonlinear() {
        let eq1 = Expr::parse_expression("y*y");
        let eq_system = vec![eq1];
        let values = vec!["y".to_string()];
        let arg = "t".to_string();
        let method = "BDF".to_string();
        let t0 = 0.0;
        let y0 = DVector::from_vec(vec![1.0]);
        let t_bound = 10.0;
        let max_step = 0.01;
        let rtol = 1e-6;
        let atol = 1e-3;

        let mut solver = ODEsolver::new(
            eq_system, values, arg, method, t0, y0, t_bound, max_step, rtol, atol, None, false,
            None,
        );

        let mut stop_condition = HashMap::new();
        stop_condition.insert("y".to_string(), 1.5);
        solver.set_stop_condition(stop_condition);

        solver.solve();

        assert_eq!(solver.get_status(), "stopped_by_condition");
        let (_, y_result) = solver.get_result();
        let final_y = y_result[(y_result.nrows() - 1, 0)];
        assert!((final_y - 1.5).abs() <= atol);
    }
}

#[cfg(test)]
mod tests_generated_backend_heavy_dense_aot {
    use super::*;
    use crate::symbolic::codegen::codegen_runtime_api::{
        recommended_dense_jacobian_chunking_for_parallelism,
        recommended_residual_chunking_for_parallelism,
    };
    use crate::symbolic::codegen::rust_backend::codegen_aot_build::AotBuildProfile;
    use crate::symbolic::symbolic_ivp::SymbolicIvpAotOptions;
    use crate::symbolic::symbolic_ivp_generated::{
        DenseIvpGeneratedBackendMode, SymbolicIvpAotBuildPolicy, SymbolicIvpGeneratedBackendConfig,
    };
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::path::PathBuf;
    use std::process::Command;
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    #[derive(Clone)]
    struct BdfScenario {
        label: &'static str,
        equations: Vec<Expr>,
        values: Vec<String>,
        y0: DVector<f64>,
        t0: f64,
        t_bound: f64,
        max_step: f64,
        rtol: f64,
        atol: f64,
    }

    #[derive(Clone, Copy)]
    enum Toolchain {
        Ctcc,
        Cgcc,
        Zig,
        Rust,
    }

    impl Toolchain {
        fn label(self) -> &'static str {
            match self {
                Self::Ctcc => "AOT-C-tcc",
                Self::Cgcc => "AOT-C-gcc",
                Self::Zig => "AOT-Zig",
                Self::Rust => "AOT-Rust",
            }
        }
    }

    #[derive(Clone, Copy)]
    enum ChunkingMode {
        Whole,
        Parallel2,
    }

    impl ChunkingMode {
        fn label(self) -> &'static str {
            match self {
                Self::Whole => "whole",
                Self::Parallel2 => "parallel(auto,x2)",
            }
        }
    }

    struct CompareRow {
        scenario: &'static str,
        route: String,
        chunking: &'static str,
        total: Duration,
        prepare_ms: f64,
        solve_ms: f64,
        residual_calls: usize,
        jacobian_calls: usize,
        nlu: usize,
        final_diff: f64,
        status: String,
    }

    fn command_exists(cmd: &str, probe_arg: &str) -> bool {
        Command::new(cmd).arg(probe_arg).output().is_ok()
    }

    fn tcc_available() -> bool {
        if let Ok(explicit) = std::env::var("RUSTEDSCITHE_TCC") {
            return std::path::Path::new(&explicit).is_file();
        }
        command_exists("tcc", "-v")
    }

    fn gcc_available() -> bool {
        if let Ok(explicit) = std::env::var("RUSTEDSCITHE_GCC") {
            return std::path::Path::new(&explicit).is_file();
        }
        command_exists("gcc", "--version")
    }

    fn zig_available() -> bool {
        command_exists("zig", "version")
    }

    fn toolchain_available(toolchain: Toolchain) -> bool {
        match toolchain {
            Toolchain::Ctcc => tcc_available(),
            Toolchain::Cgcc => gcc_available(),
            Toolchain::Zig => zig_available(),
            Toolchain::Rust => true,
        }
    }

    fn robertson_3_scenario() -> BdfScenario {
        BdfScenario {
            label: "robertson-3",
            equations: vec![
                Expr::parse_expression("-0.04*y1 + 1.0e4*y2*y3"),
                Expr::parse_expression("0.04*y1 - 1.0e4*y2*y3 - 3.0e7*y2^2"),
                Expr::parse_expression("3.0e7*y2^2"),
            ],
            values: vec!["y1".to_string(), "y2".to_string(), "y3".to_string()],
            y0: DVector::from_vec(vec![1.0, 0.0, 0.0]),
            t0: 0.0,
            t_bound: 20.0,
            max_step: 0.001,
            rtol: 1e-9,
            atol: 1e-12,
        }
    }

    fn hires_8_scenario() -> BdfScenario {
        BdfScenario {
            label: "hires-8",
            equations: vec![
                Expr::parse_expression("-1.71*y1 + 0.43*y2 + 8.32*y3 + 0.0007"),
                Expr::parse_expression("1.71*y1 - 8.75*y2"),
                Expr::parse_expression("-10.03*y3 + 0.43*y4 + 0.035*y5"),
                Expr::parse_expression("8.32*y2 + 1.71*y3 - 1.12*y4"),
                Expr::parse_expression("-1.745*y5 + 0.43*y6 + 0.43*y7"),
                Expr::parse_expression("-280.0*y6*y8 + 0.69*y4 + 1.71*y5 - 0.43*y6 + 0.69*y7"),
                Expr::parse_expression("280.0*y6*y8 - 1.81*y7"),
                Expr::parse_expression("-280.0*y6*y8 + 1.81*y7"),
            ],
            values: vec![
                "y1".to_string(),
                "y2".to_string(),
                "y3".to_string(),
                "y4".to_string(),
                "y5".to_string(),
                "y6".to_string(),
                "y7".to_string(),
                "y8".to_string(),
            ],
            y0: DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0057]),
            t0: 0.0,
            t_bound: 20.0,
            max_step: 0.002,
            rtol: 1e-8,
            atol: 1e-11,
        }
    }

    fn max_abs_diff(a: &DVector<f64>, b: &DVector<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .fold(0.0_f64, |acc, (lhs, rhs)| acc.max((lhs - rhs).abs()))
    }

    fn unique_output_root(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        PathBuf::from(format!(
            "target/generated-bdf-aot-story/{prefix}/pid{}_{}",
            std::process::id(),
            nanos
        ))
    }

    fn chunking_options(var_count: usize, mode: ChunkingMode) -> SymbolicIvpAotOptions {
        match mode {
            ChunkingMode::Whole => SymbolicIvpAotOptions::default(),
            ChunkingMode::Parallel2 => SymbolicIvpAotOptions {
                residual_strategy: recommended_residual_chunking_for_parallelism(var_count, 2),
                jacobian_strategy: recommended_dense_jacobian_chunking_for_parallelism(
                    var_count, 2,
                ),
            },
        }
    }

    fn make_backend_config(
        out_dir: PathBuf,
        toolchain: Toolchain,
        chunking: ChunkingMode,
        var_count: usize,
    ) -> SymbolicIvpGeneratedBackendConfig {
        let base = SymbolicIvpGeneratedBackendConfig::from_mode(
            DenseIvpGeneratedBackendMode::BuildIfMissingRelease,
        )
        .with_output_parent_dir(Some(out_dir))
        .with_build_policy(SymbolicIvpAotBuildPolicy::BuildIfMissing {
            profile: AotBuildProfile::Release,
        })
        .with_aot_options(chunking_options(var_count, chunking));

        match toolchain {
            Toolchain::Ctcc => base.with_c_tcc(),
            Toolchain::Cgcc => base.with_c_gcc(),
            Toolchain::Zig => base.with_zig(),
            Toolchain::Rust => base.with_rust(),
        }
    }

    fn scenario_options(s: &BdfScenario) -> BdfSolverOptions {
        BdfSolverOptions::new(
            s.equations.clone(),
            s.values.clone(),
            "t".to_string(),
            "BDF".to_string(),
            s.t0,
            s.y0.clone(),
            s.t_bound,
            s.max_step,
            s.rtol,
            s.atol,
            None,
            false,
            None,
        )
        .with_max_bdf_order(5)
    }

    fn run_case(
        scenario: &BdfScenario,
        route_label: &str,
        chunking: ChunkingMode,
        options: BdfSolverOptions,
        baseline_solution: Option<&DVector<f64>>,
    ) -> (CompareRow, DVector<f64>) {
        let mut solver = ODEsolver::new_with_options(options);
        let start = Instant::now();
        let solve_result = catch_unwind(AssertUnwindSafe(|| {
            solver.solve();
            let stats = solver.get_statistics();
            let (_, y) = solver.get_result();
            let final_solution = if y.nrows() == 0 {
                DVector::from_element(scenario.values.len(), f64::NAN)
            } else {
                y.row(y.nrows() - 1).transpose().into_owned()
            };
            (
                solver.get_status().clone(),
                stats.backend_prepare_ms_total,
                stats.solve_ms_total,
                stats.bdf_nfev_total,
                stats.bdf_njev_total,
                stats.bdf_nlu_total,
                final_solution,
            )
        }));
        let total = start.elapsed();

        match solve_result {
            Ok((status, prepare_ms, solve_ms, residual_calls, jacobian_calls, nlu, solution)) => {
                let final_diff = baseline_solution
                    .map(|baseline| max_abs_diff(&solution, baseline))
                    .unwrap_or(0.0);
                (
                    CompareRow {
                        scenario: scenario.label,
                        route: route_label.to_string(),
                        chunking: chunking.label(),
                        total,
                        prepare_ms,
                        solve_ms,
                        residual_calls,
                        jacobian_calls,
                        nlu,
                        final_diff,
                        status,
                    },
                    solution,
                )
            }
            Err(_) => (
                CompareRow {
                    scenario: scenario.label,
                    route: route_label.to_string(),
                    chunking: chunking.label(),
                    total,
                    prepare_ms: f64::NAN,
                    solve_ms: f64::NAN,
                    residual_calls: 0,
                    jacobian_calls: 0,
                    nlu: 0,
                    final_diff: f64::NAN,
                    status: "panic".to_string(),
                },
                DVector::from_element(scenario.values.len(), f64::NAN),
            ),
        }
    }

    #[test]
    #[ignore]
    fn bdf_dense_aot_heavy_toolchain_chunking_matrix_story() {
        let scenarios = vec![robertson_3_scenario(), hires_8_scenario()];
        let mut rows = Vec::<CompareRow>::new();

        for scenario in &scenarios {
            let (baseline_row, baseline_solution) = run_case(
                scenario,
                "Lambdify",
                ChunkingMode::Whole,
                scenario_options(scenario),
                None,
            );
            rows.push(baseline_row);

            for toolchain in [
                Toolchain::Ctcc,
                Toolchain::Cgcc,
                Toolchain::Zig,
                Toolchain::Rust,
            ] {
                if !toolchain_available(toolchain) {
                    println!(
                        "[BDF AOT heavy] skipping {} on scenario {}: compiler/runtime unavailable",
                        toolchain.label(),
                        scenario.label
                    );
                    continue;
                }

                for chunking in [ChunkingMode::Whole, ChunkingMode::Parallel2] {
                    let out_dir = unique_output_root(&format!(
                        "{}_{}_{}",
                        scenario.label,
                        toolchain.label(),
                        chunking.label()
                    ));
                    let config = make_backend_config(
                        out_dir,
                        toolchain,
                        chunking,
                        scenario.values.len().max(1),
                    );
                    let options = scenario_options(scenario).with_generated_backend_config(config);
                    let (row, _) = run_case(
                        scenario,
                        toolchain.label(),
                        chunking,
                        options,
                        Some(&baseline_solution),
                    );
                    rows.push(row);
                }
            }
        }

        println!(
            "[BDF AOT heavy] dense toolchain+chunking matrix; all time columns are milliseconds"
        );
        println!(
            "scenario    | route        | chunking         | total_ms | prepare_ms | solve_ms | final_diff_vs_lambdify | residual_calls | jacobian_calls | nlu | status"
        );
        println!(
            "---------------------------------------------------------------------------------------------------------------------------------------------------------------"
        );
        for row in &rows {
            println!(
                "{:<11} | {:<12} | {:<16} | {:>8.3} | {:>10.3} | {:>8.3} | {:>22.3e} | {:>14} | {:>14} | {:>3} | {}",
                row.scenario,
                row.route,
                row.chunking,
                row.total.as_secs_f64() * 1_000.0,
                row.prepare_ms,
                row.solve_ms,
                row.final_diff,
                row.residual_calls,
                row.jacobian_calls,
                row.nlu,
                row.status
            );
        }

        let finished: Vec<&CompareRow> =
            rows.iter().filter(|row| row.status == "finished").collect();
        assert!(
            !finished.is_empty(),
            "at least one dense BDF heavy AOT route should finish"
        );

        for row in rows.iter().filter(|row| row.route != "Lambdify") {
            assert_eq!(
                row.status, "finished",
                "dense BDF AOT route failed: scenario={} route={} chunking={}",
                row.scenario, row.route, row.chunking
            );
            assert!(
                row.final_diff <= 1e-6,
                "dense BDF AOT parity drift is too large: scenario={} route={} chunking={} diff={:e}",
                row.scenario,
                row.route,
                row.chunking,
                row.final_diff
            );
        }
    }
}
