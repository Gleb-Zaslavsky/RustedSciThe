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

use crate::numerical::BDF::BDF_solver::{
    BdfConfigurationError, BdfJacobian, BdfLinearBackend, BdfOperationCounters, BdfStepError, BDF,
};
pub use crate::numerical::BDF::BDF_solver::{BdfJacobianCallbackError, BdfJacobianSource};
use crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend;
use crate::symbolic::ivp_telemetry::{
    IvpLambdifyExecutionPolicy, IvpTelemetry, IvpTelemetrySnapshot,
};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp::{
    IvpBackendError, IvpSymbolicAssemblyBackend, SharedIvpParameterValues,
    SymbolicIvpProblemOptions,
};
use crate::symbolic::symbolic_ivp_generated::{
    prepare_generated_symbolic_ivp_problem, prepare_generated_symbolic_ivp_residual_problem,
    DenseIvpGeneratedBackendMode, IvpBackendStatistics, SymbolicIvpAotBuildPolicy,
    SymbolicIvpGeneratedBackendConfig,
};
extern crate nalgebra as na;
use crate::numerical::BDF::common::NumberOrVec;
use crate::Utils::plots::plots_ref;
use na::{DMatrix, DVector};

use csv::Writer;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

fn elapsed_ms(start: Option<Instant>) -> f64 {
    start.map_or(0.0, |start| start.elapsed().as_secs_f64() * 1_000.0)
}

type BdfNativeJacobianFactory =
    dyn Fn(Option<SharedIvpParameterValues>) -> Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian>;
type BdfPreparedResidual = dyn Fn(f64, &DVector<f64>) -> DVector<f64>;
type BdfPreparedDenseJacobianInto =
    dyn FnMut(f64, &DVector<f64>, &mut DMatrix<f64>) -> Result<(), BdfJacobianCallbackError>;
type BdfNativeRhs = Arc<dyn Fn(f64, &DVector<f64>) -> DVector<f64> + Send + Sync>;
pub type BdfNativeDenseJacobian = Arc<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64> + Send + Sync>;

/// Jacobian lifecycle for pure numerical callback problems.
#[derive(Clone)]
pub enum BdfNativeJacobianSource {
    FiniteDifference,
    StateDependent(BdfNativeDenseJacobian),
    Constant(DMatrix<f64>),
}

/// Runtime instrumentation level for BDF. The default performs no telemetry
/// allocation, callback wrapping, clock reads, or statistics locking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BdfTelemetryMode {
    #[default]
    Off,
    Counters,
    Timings,
}

/// Disjoint diagnostic stages inside one BDF backend preparation call.
/// Timings are populated only when [`BdfTelemetryMode::Timings`] is enabled.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BdfPreparationTimings {
    pub configuration_ms: f64,
    pub symbolic_backend_ms: f64,
    pub callback_wiring_ms: f64,
    pub runtime_initialization_ms: f64,
    pub linear_backend_setup_ms: f64,
    pub runtime_publication_ms: f64,
    pub total_ms: f64,
}

impl BdfPreparationTimings {
    pub fn table_report(&self) -> String {
        format!(
            "configuration_ms={:.3} symbolic_backend_ms={:.3} callback_wiring_ms={:.3} runtime_initialization_ms={:.3} linear_backend_setup_ms={:.3} runtime_publication_ms={:.3} total_ms={:.3} (stages are disjoint diagnostic scopes)",
            self.configuration_ms,
            self.symbolic_backend_ms,
            self.callback_wiring_ms,
            self.runtime_initialization_ms,
            self.linear_backend_setup_ms,
            self.runtime_publication_ms,
            self.total_ms,
        )
    }
}

/// Provenance for the latest generated AOT preparation.
///
/// The lifecycle identity is kept next to the immutable telemetry snapshot so
/// reports cannot accidentally compare counters from one policy with metadata
/// from another. The snapshot remains the source of truth for cache, build,
/// link, publication, and timing values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BdfAotProvenance {
    pub build_policy: SymbolicIvpAotBuildPolicy,
    pub codegen_backend: AotCodegenBackend,
    pub c_compiler: Option<String>,
    pub telemetry: IvpTelemetrySnapshot,
}

impl BdfAotProvenance {
    pub fn backend_label(&self) -> &'static str {
        match self.codegen_backend {
            AotCodegenBackend::Rust => "rust",
            AotCodegenBackend::C => "c",
            AotCodegenBackend::Zig => "zig",
        }
    }

    pub fn execution_label(&self) -> &'static str {
        self.telemetry.execution.label()
    }
}

/// Stable solver status; the string getter is retained for API compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BdfStatus {
    #[default]
    Running,
    Finished,
    Failed,
    StoppedByCondition,
}

/// Typed failure from preparing or integrating one BDF problem.
#[derive(Debug)]
pub enum BdfSolveError {
    Backend(IvpBackendError),
    Configuration(BdfConfigurationError),
    Step(BdfStepError),
    MaxStepsExceeded { max_steps: usize },
    InvalidMaxSteps,
    InvalidContinuationBound,
    ContinuationRequiresPreparedBackend,
}

impl std::fmt::Display for BdfSolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Backend(error) => write!(f, "BDF backend preparation failed: {error}"),
            Self::Configuration(error) => write!(f, "invalid BDF configuration: {error}"),
            Self::Step(error) => write!(f, "BDF integration step failed: {error}"),
            Self::MaxStepsExceeded { max_steps } => {
                write!(f, "BDF exceeded the configured limit of {max_steps} steps")
            }
            Self::InvalidMaxSteps => f.write_str("BDF max_steps must be greater than zero"),
            Self::InvalidContinuationBound => {
                f.write_str("continuation bound must be finite and differ from the current time")
            }
            Self::ContinuationRequiresPreparedBackend => f.write_str(
                "prepare and solve the BDF model before starting a continuation segment",
            ),
        }
    }
}

impl std::error::Error for BdfSolveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Backend(error) => Some(error),
            Self::Configuration(error) => Some(error),
            Self::Step(error) => Some(error),
            Self::MaxStepsExceeded { .. }
            | Self::InvalidMaxSteps
            | Self::InvalidContinuationBound
            | Self::ContinuationRequiresPreparedBackend => None,
        }
    }
}

impl BdfStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Finished => "finished",
            Self::Failed => "failed",
            Self::StoppedByCondition => "stopped_by_condition",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BdfStopConditionError {
    UnknownVariable(String),
    NonFiniteTarget(String),
}

impl std::fmt::Display for BdfStopConditionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownVariable(name) => {
                write!(f, "stop condition refers to unknown variable `{name}`")
            }
            Self::NonFiniteTarget(name) => {
                write!(f, "stop condition target for `{name}` must be finite")
            }
        }
    }
}

impl std::error::Error for BdfStopConditionError {}

fn validate_legacy_method(method: &str) {
    assert_eq!(
        method, "BDF",
        "BDF API supports only the BDF method; other methods require their own solver API"
    );
}

/// Grouped setup for one symbolic BDF solve.
#[derive(Clone)]
pub struct BdfSolverOptions {
    pub eq_system: Vec<Expr>,
    pub values: Vec<String>,
    pub arg: String,
    pub t0: f64,
    pub y0: DVector<f64>,
    pub t_bound: f64,
    pub max_step: f64,
    pub rtol: f64,
    pub atol: f64,
    /// Reserved for grouped finite-difference Jacobians; rejected when BDF
    /// would otherwise estimate the Jacobian by finite differences.
    pub jac_sparsity: Option<DMatrix<f64>>,
    /// Compatibility placeholder; `true` is rejected because callbacks are
    /// currently scalar, not batched.
    pub vectorized: bool,
    pub first_step: Option<f64>,
    pub max_bdf_order: usize,
    pub equation_parameters: Option<Vec<String>>,
    pub equation_parameter_values: Option<DVector<f64>>,
    pub generated_backend_config: SymbolicIvpGeneratedBackendConfig,
    pub symbolic_assembly_backend: IvpSymbolicAssemblyBackend,
    pub lambdify_execution_policy: IvpLambdifyExecutionPolicy,
    pub telemetry_mode: BdfTelemetryMode,
    /// Maximum number of high-level integration attempts per solve.
    pub max_steps: usize,
}

impl BdfSolverOptions {
    /// Creates grouped BDF options through the legacy method-string interface.
    /// Prefer [`Self::for_bdf`] for new code; any method other than `"BDF"`
    /// is rejected immediately.
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
        validate_legacy_method(&method);
        Self::for_bdf(
            eq_system,
            values,
            arg,
            t0,
            y0,
            t_bound,
            max_step,
            rtol,
            atol,
            jac_sparsity,
            vectorized,
            first_step,
        )
    }

    /// Creates options for this module's sole supported method, BDF.
    pub fn for_bdf(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
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
            lambdify_execution_policy: IvpLambdifyExecutionPolicy::default(),
            telemetry_mode: BdfTelemetryMode::Off,
            max_steps: 1_000_000,
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

    /// Selects the runtime policy for independent Lambdify callback entries.
    pub fn with_lambdify_execution_policy(mut self, policy: IvpLambdifyExecutionPolicy) -> Self {
        self.lambdify_execution_policy = policy;
        self
    }

    /// Enables optional solver telemetry. It is fully disabled by default.
    pub fn with_telemetry_mode(mut self, mode: BdfTelemetryMode) -> Self {
        self.telemetry_mode = mode;
        self
    }

    /// Sets a bounded number of BDF step attempts for each solve.
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
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
    /// Reserved for grouped finite-difference Jacobians, which are not implemented.
    jac_sparsity: Option<DMatrix<f64>>,
    /// Compatibility option; `true` is rejected until batched RHS callbacks exist.
    vectorized: bool,
    /// Optional initial step size (auto-selected if None)
    first_step: Option<f64>,
    /// Maximum adaptive BDF order allowed for the low-level BDF engine.
    max_bdf_order: usize,
    max_steps: usize,

    /// Current integration status.
    status: BdfStatus,
    /// Internal BDF solver instance
    Solver_instance: BDF,
    /// Optional error message from failed integration
    message: Option<String>,

    /// Time points of computed solution
    t_result: DVector<f64>,
    /// Solution matrix: rows = time points, columns = variables
    y_result: DMatrix<f64>,
    /// Optional stop conditions: variable_name → target_value
    stop_condition: Option<Vec<(usize, f64)>>,
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
    lambdify_execution_policy: IvpLambdifyExecutionPolicy,
    telemetry_mode: BdfTelemetryMode,
    statistics: Option<Arc<Mutex<IvpBackendStatistics>>>,
    preparation_timings: BdfPreparationTimings,
    preparation_backend_telemetry: Option<IvpTelemetrySnapshot>,
    reported_bdf_counters: BdfOperationCounters,
    /// Optional factory for the Newton linear backend of each generated BDF instance.
    bdf_linear_backend_factory: Option<Box<dyn Fn() -> Box<dyn BdfLinearBackend>>>,
    /// Optional factory for a generated Jacobian callback. When present, the
    /// residual-only preparation path is used and no dense symbolic/FD Jacobian
    /// is constructed before the native callback is installed.
    bdf_native_jacobian_factory: Option<Box<BdfNativeJacobianFactory>>,
    /// Prepared generated residual supplied by an owning solver lifecycle.
    ///
    /// LSODE2 prepares residual and native Jacobian callbacks as one unit. The
    /// bridge must consume that unit instead of rebuilding the symbolic
    /// problem before installing its BDF instance.
    prepared_generated_residual: Option<Box<BdfPreparedResidual>>,
    /// Retained numerical callbacks let parameter continuation restart solver
    /// history without repeating symbolic differentiation or backend lowering.
    prepared_residual_callback: Option<Arc<BdfPreparedResidual>>,
    prepared_jacobian_callback: Option<Arc<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64>>>,
    prepared_jacobian_into_factory: Option<Arc<dyn Fn() -> Box<BdfPreparedDenseJacobianInto>>>,
    /// Optional pure numerical RHS callback `f(t, y)`.
    native_rhs: Option<BdfNativeRhs>,
    /// Optional pure numerical Jacobian lifecycle for `f(t, y)` callbacks.
    native_jacobian_source: Option<BdfNativeJacobianSource>,
}
impl ODEsolver {
    /// Creates a new ODE solver through the legacy method-string interface.
    /// Prefer [`BdfSolverOptions::for_bdf`] and [`Self::new_with_options`] for
    /// new code. Any method other than `"BDF"` is rejected immediately.
    ///
    /// # Parameters
    /// * `eq_system` - Vector of symbolic expressions defining dy/dt = f(t,y)
    /// * `values` - Variable names corresponding to solution components
    /// * `arg` - Independent variable name (usually "t")
    /// * `method` - Legacy selector; only "BDF" is accepted.
    /// * `t0` - Initial time
    /// * `y0` - Initial solution vector
    /// * `t_bound` - Final integration time
    /// * `max_step` - Maximum step size
    /// * `rtol` - Relative tolerance for error control
    /// * `atol` - Absolute tolerance for error control
    /// * `jac_sparsity` - Reserved for grouped FD; rejected when no analytic Jacobian is available
    /// * `vectorized` - Must be false; BDF currently invokes scalar RHS callbacks
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
        validate_legacy_method(&method);
        Self::new_bdf(
            eq_system,
            values,
            arg,
            t0,
            y0,
            t_bound,
            max_step,
            rtol,
            atol,
            jac_sparsity,
            vectorized,
            first_step,
        )
    }

    fn new_bdf(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
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
            max_steps: 1_000_000,
            status: BdfStatus::Running,
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
            lambdify_execution_policy: IvpLambdifyExecutionPolicy::default(),
            telemetry_mode: BdfTelemetryMode::Off,
            statistics: None,
            preparation_timings: BdfPreparationTimings::default(),
            preparation_backend_telemetry: None,
            reported_bdf_counters: BdfOperationCounters::default(),
            bdf_linear_backend_factory: None,
            bdf_native_jacobian_factory: None,
            prepared_generated_residual: None,
            prepared_residual_callback: None,
            prepared_jacobian_callback: None,
            prepared_jacobian_into_factory: None,
            native_rhs: None,
            native_jacobian_source: None,
        }
    }

    /// Preferred grouped setup path for symbolic BDF solves.
    pub fn new_with_options(options: BdfSolverOptions) -> Self {
        let mut solver = Self::new_bdf(
            options.eq_system,
            options.values,
            options.arg,
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
        solver.max_steps = options.max_steps;
        solver.equation_parameters = options.equation_parameters;
        solver.equation_parameter_values = options.equation_parameter_values;
        solver.symbolic_assembly_backend = options.symbolic_assembly_backend;
        solver.lambdify_execution_policy = options.lambdify_execution_policy;
        solver.set_telemetry_mode(options.telemetry_mode);
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
            .as_ref()
            .and_then(|stats| stats.lock().ok().map(|stats| stats.clone()))
            .unwrap_or_default()
    }

    /// Sets telemetry mode. Select it before generation/solve when possible;
    /// changing mode invalidates the prepared callback wrappers. Turning it off
    /// drops all shared counters and restores the uninstrumented callback path.
    pub fn set_telemetry_mode(&mut self, mode: BdfTelemetryMode) {
        if mode == self.telemetry_mode {
            return;
        }
        self.telemetry_mode = mode;
        self.statistics = (mode != BdfTelemetryMode::Off)
            .then(|| Arc::new(Mutex::new(IvpBackendStatistics::default())));
        self.preparation_timings = BdfPreparationTimings::default();
        self.preparation_backend_telemetry = None;
        self.prepared_jacobian_into_factory = None;
        self.backend_prepared = false;
    }

    pub fn telemetry_mode(&self) -> BdfTelemetryMode {
        self.telemetry_mode
    }

    /// Returns disjoint backend-preparation timings from the latest generation.
    /// Values are zero unless telemetry mode is `Timings`.
    pub fn preparation_timings(&self) -> BdfPreparationTimings {
        self.preparation_timings
    }

    /// Detailed symbolic-backend cold stages from the latest preparation.
    /// Available only when BDF timing telemetry was enabled.
    pub fn preparation_backend_telemetry(&self) -> Option<&IvpTelemetrySnapshot> {
        self.preparation_backend_telemetry.as_ref()
    }

    /// Returns the lifecycle identity and telemetry for the latest generated
    /// AOT preparation. Returns `None` for native/Lambdify preparation or
    /// before a generated backend has been prepared.
    pub fn aot_provenance(&self) -> Option<BdfAotProvenance> {
        let telemetry = self.preparation_backend_telemetry.as_ref()?;
        (telemetry.execution == crate::symbolic::ivp_telemetry::IvpTelemetryExecution::Aot).then(
            || BdfAotProvenance {
                build_policy: self.generated_backend_config.build_policy,
                codegen_backend: self.generated_backend_config.aot_codegen_backend,
                c_compiler: self.generated_backend_config.aot_c_compiler.clone(),
                telemetry: telemetry.clone(),
            },
        )
    }

    fn record_prepare_duration(&self, start: Option<Instant>) {
        if let Some(stats) = self.statistics.as_ref() {
            let Ok(mut stats) = stats.lock() else {
                return;
            };
            stats.backend_prepare_calls += 1;
            if let Some(start) = start {
                stats.backend_prepare_ms_total += start.elapsed().as_secs_f64() * 1_000.0;
            }
        }
    }

    fn instrument_residual(
        &self,
        residual: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
    ) -> Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>> {
        let Some(stats) = self.statistics.as_ref().map(Arc::clone) else {
            return residual;
        };
        let mode = self.telemetry_mode;
        Box::new(move |t, y| {
            let start = (mode == BdfTelemetryMode::Timings).then(Instant::now);
            let out = residual(t, y);
            let Ok(mut stats) = stats.lock() else {
                return out;
            };
            if let Some(start) = start {
                stats.record_residual_duration(start.elapsed());
            } else {
                stats.residual_calls += 1;
            }
            out
        })
    }

    fn instrument_jacobian(
        &self,
        jacobian: Box<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64>>,
    ) -> Box<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64>> {
        let Some(stats) = self.statistics.as_ref().map(Arc::clone) else {
            return jacobian;
        };
        let mode = self.telemetry_mode;
        Box::new(move |t, y| {
            let start = (mode == BdfTelemetryMode::Timings).then(Instant::now);
            let out = jacobian(t, y);
            let Ok(mut stats) = stats.lock() else {
                return out;
            };
            if let Some(start) = start {
                stats.record_jacobian_duration(start.elapsed());
            } else {
                stats.jacobian_calls += 1;
            }
            out
        })
    }

    fn instrument_jacobian_into(
        &self,
        mut jacobian: Box<BdfPreparedDenseJacobianInto>,
    ) -> Box<BdfPreparedDenseJacobianInto> {
        let Some(stats) = self.statistics.as_ref().map(Arc::clone) else {
            return jacobian;
        };
        let mode = self.telemetry_mode;
        Box::new(move |t, y, out| {
            let start = (mode == BdfTelemetryMode::Timings).then(Instant::now);
            let result = jacobian(t, y, out);
            let Ok(mut stats) = stats.lock() else {
                return result;
            };
            if let Some(start) = start {
                stats.record_jacobian_duration(start.elapsed());
            } else {
                stats.jacobian_calls += 1;
            }
            result
        })
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
    /// When this factory is present, BDF prepares the residual separately and
    /// uses the factory result as the initial and refresh Jacobian source. This
    /// avoids constructing and then discarding a dense Jacobian.
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

    /// Installs callbacks prepared by an owning symbolic lifecycle.
    ///
    /// This is intentionally a single operation: installing the residual and
    /// Jacobian separately would allow `try_generate` to prepare the symbolic
    /// residual a second time before it invokes the native Jacobian factory.
    pub(crate) fn set_prepared_generated_callbacks<F>(
        &mut self,
        residual: Box<BdfPreparedResidual>,
        parameter_values_handle: Option<SharedIvpParameterValues>,
        jacobian_factory: F,
    ) where
        F: Fn(
                Option<SharedIvpParameterValues>,
            ) -> Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian>
            + 'static,
    {
        self.prepared_generated_residual = Some(residual);
        self.parameter_values_handle = parameter_values_handle;
        self.bdf_native_jacobian_factory = Some(Box::new(jacobian_factory));
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
    /// Compatibility adapter: `None` selects finite differences and `Some`
    /// selects a state-dependent analytic Jacobian.
    pub fn set_native_ode_callbacks<F, J>(&mut self, rhs: F, jac: Option<J>)
    where
        F: Fn(f64, &DVector<f64>) -> DVector<f64> + Send + Sync + 'static,
        J: Fn(f64, &DVector<f64>) -> DMatrix<f64> + Send + Sync + 'static,
    {
        let jacobian_source = match jac {
            Some(jacobian) => BdfNativeJacobianSource::StateDependent(Arc::new(jacobian)),
            None => BdfNativeJacobianSource::FiniteDifference,
        };
        self.set_native_ode_callbacks_with_jacobian_source(rhs, jacobian_source);
    }

    /// Installs pure numerical callbacks with an explicit Jacobian lifecycle.
    pub fn set_native_ode_callbacks_with_jacobian_source<F>(
        &mut self,
        rhs: F,
        jacobian_source: BdfNativeJacobianSource,
    ) where
        F: Fn(f64, &DVector<f64>) -> DVector<f64> + Send + Sync + 'static,
    {
        self.native_rhs = Some(Arc::new(rhs));
        self.native_jacobian_source = Some(jacobian_source);
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
        self.try_set_stop_condition(stop_condition)
            .expect("BDF stop conditions must use known variables and finite targets");
    }

    /// Validates and pre-resolves stop conditions once, outside the step loop.
    pub fn try_set_stop_condition(
        &mut self,
        stop_condition: HashMap<String, f64>,
    ) -> Result<(), BdfStopConditionError> {
        let mut resolved = Vec::with_capacity(stop_condition.len());
        for (name, target) in stop_condition {
            if !target.is_finite() {
                return Err(BdfStopConditionError::NonFiniteTarget(name));
            }
            let Some(index) = self.values.iter().position(|value| value == &name) else {
                return Err(BdfStopConditionError::UnknownVariable(name));
            };
            resolved.push((index, target));
        }
        self.stop_condition = Some(resolved);
        Ok(())
    }

    pub fn clear_stop_condition(&mut self) {
        self.stop_condition = None;
    }

    /// Declares symbolic parameter names used by the IVP right-hand side.
    pub fn set_equation_parameters(&mut self, params: Option<&[&str]>) {
        self.equation_parameters =
            params.map(|params| params.iter().map(|p| (*p).to_string()).collect());
        self.backend_prepared = false;
    }

    /// Updates numeric values of symbolic equation parameters without recompiling
    /// already prepared closures. This only changes the shared parameter slot;
    /// use [`Self::try_continue_with_parameter_values`] to restart solver history
    /// at a segment boundary before integrating with the new values.
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
                .map_err(|_| IvpBackendError::ParameterStatePoisoned)?;
            *slot = values.clone();
        }
        self.equation_parameter_values = Some(values);
        Ok(())
    }

    /// Starts a new parameter segment at the current accepted `(t, y)` state.
    /// The prepared numerical callbacks and generated artifact are reused; only
    /// the BDF history, Jacobian and factorization are reinitialized.
    pub fn try_continue_with_parameter_values(
        &mut self,
        values: DVector<f64>,
        t_bound: f64,
    ) -> Result<(), BdfSolveError> {
        if !self.backend_prepared {
            return Err(BdfSolveError::ContinuationRequiresPreparedBackend);
        }
        if self.max_steps == 0 {
            return Err(BdfSolveError::InvalidMaxSteps);
        }
        let t0 = self.Solver_instance.t;
        if !t_bound.is_finite() || t_bound == t0 {
            return Err(BdfSolveError::InvalidContinuationBound);
        }

        self.set_parameter_values(values)
            .map_err(BdfSolveError::Backend)?;
        self.t0 = t0;
        self.y0 = self.Solver_instance.y.clone();
        self.t_bound = t_bound;

        self.try_restart_prepared_runtime()
    }

    /// Restarts a prepared model with a new initial state and integration
    /// interval without repeating symbolic preparation or AOT publication.
    ///
    /// The prepared residual/Jacobian callbacks, generated artifact and
    /// selected linear-backend factory are retained. Only the BDF history,
    /// initial Jacobian and factorization are rebuilt for the new segment.
    pub fn try_restart_with_initial_state(
        &mut self,
        t0: f64,
        y0: DVector<f64>,
        t_bound: f64,
    ) -> Result<(), BdfSolveError> {
        if !self.backend_prepared {
            return Err(BdfSolveError::ContinuationRequiresPreparedBackend);
        }
        if self.max_steps == 0 {
            return Err(BdfSolveError::InvalidMaxSteps);
        }
        if !t0.is_finite() || !t_bound.is_finite() || t0 == t_bound {
            return Err(BdfSolveError::InvalidContinuationBound);
        }
        if y0.is_empty() {
            return Err(BdfSolveError::Configuration(
                BdfConfigurationError::EmptyInitialState,
            ));
        }
        if y0.len() != self.y0.len() {
            return Err(BdfSolveError::Configuration(
                BdfConfigurationError::InitialStateDimension,
            ));
        }
        if y0.iter().any(|value| !value.is_finite()) {
            return Err(BdfSolveError::Configuration(
                BdfConfigurationError::NonFiniteInitialState,
            ));
        }

        self.t0 = t0;
        self.y0 = y0;
        self.t_bound = t_bound;
        self.try_restart_prepared_runtime()
    }

    fn try_restart_prepared_runtime(&mut self) -> Result<(), BdfSolveError> {
        if self.native_rhs.is_some() {
            self.try_generate_native_numeric()
                .map_err(BdfSolveError::Configuration)?;
            return Ok(());
        }

        let residual = self
            .prepared_residual_callback
            .as_ref()
            .cloned()
            .ok_or(BdfSolveError::ContinuationRequiresPreparedBackend)?;
        let mut solver_instance = BDF::new();
        solver_instance
            .try_set_max_order_cap(self.max_bdf_order)
            .map_err(BdfSolveError::Configuration)?;
        solver_instance
            .set_operation_counters_enabled(self.telemetry_mode != BdfTelemetryMode::Off);
        solver_instance
            .set_operation_timings_enabled(self.telemetry_mode == BdfTelemetryMode::Timings);
        let residual_for_runtime = Arc::clone(&residual);
        let wrapped_fun =
            self.instrument_residual(Box::new(move |t, y| residual_for_runtime(t, y)));
        let jacobian_source = if let Some(factory) = self.bdf_native_jacobian_factory.as_ref() {
            let handle = self.parameter_values_handle.clone();
            BdfJacobianSource::StateDependent(self.timed_native_jacobian(factory(handle)))
        } else if let Some(factory) = self.prepared_jacobian_into_factory.as_ref().cloned() {
            BdfJacobianSource::StateDependentDenseInto(self.instrument_jacobian_into(factory()))
        } else if let Some(jacobian) = self.prepared_jacobian_callback.as_ref().cloned() {
            let jacobian_for_runtime = Arc::clone(&jacobian);
            let jacobian =
                self.instrument_jacobian(Box::new(move |t, y| jacobian_for_runtime(t, y)));
            BdfJacobianSource::StateDependent(Box::new(move |t, y| {
                BdfJacobian::from_dense(jacobian(t, y))
            }))
        } else {
            BdfJacobianSource::FiniteDifference
        };
        solver_instance
            .try_set_initial_with_jacobian_source(
                wrapped_fun,
                self.t0,
                self.y0.clone(),
                self.t_bound,
                self.max_step,
                NumberOrVec::Number(self.rtol),
                NumberOrVec::Number(self.atol),
                jacobian_source,
                self.jac_sparsity.clone(),
                self.vectorized,
                self.first_step
                    .filter(|first_step| *first_step <= (self.t_bound - self.t0).abs()),
            )
            .map_err(BdfSolveError::Configuration)?;
        if let Some(factory) = self.bdf_linear_backend_factory.as_ref() {
            solver_instance.set_linear_backend(factory());
        }
        self.reported_bdf_counters = BdfOperationCounters::default();
        self.Solver_instance = solver_instance;
        self.mark_backend_prepared();
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
        self.stop_condition.as_ref().is_some_and(|conditions| {
            conditions
                .iter()
                .any(|&(index, target)| (y[index] - target).abs() <= self.atol)
        })
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
        self.validate_callback_options().map_err(|error| {
            IvpBackendError::InvalidArgumentSchema {
                message: error.to_string(),
            }
        })?;
        if self.native_rhs.is_some() {
            return self.try_generate_native_numeric().map_err(|error| {
                IvpBackendError::InvalidArgumentSchema {
                    message: error.to_string(),
                }
            });
        }
        self.preparation_timings = BdfPreparationTimings::default();
        self.preparation_backend_telemetry = None;
        let start = (self.telemetry_mode == BdfTelemetryMode::Timings).then(Instant::now);
        let configuration_started = start.map(|_| Instant::now());
        let mut options = SymbolicIvpProblemOptions::new();
        if let Some(parameters) = self.equation_parameters.clone() {
            options = options.with_equation_parameters(parameters);
        }
        if let Some(values) = self.equation_parameter_values.clone() {
            options = options.with_equation_parameter_values(values);
        }
        options = options.with_symbolic_assembly_backend(self.symbolic_assembly_backend);
        options = options.with_lambdify_execution_policy(self.lambdify_execution_policy);
        let backend_telemetry =
            (self.telemetry_mode == BdfTelemetryMode::Timings).then(IvpTelemetry::detailed);
        if let Some(telemetry) = &backend_telemetry {
            options = options.with_telemetry(telemetry.clone());
        }
        self.preparation_timings.configuration_ms = elapsed_ms(configuration_started);

        if self.bdf_native_jacobian_factory.is_some() {
            return self.try_generate_with_native_jacobian(start, options);
        }

        let symbolic_started = start.map(|_| Instant::now());
        let prepared_result = prepare_generated_symbolic_ivp_problem(
            self.eq_system.clone(),
            self.values.clone(),
            self.arg.clone(),
            options.with_aot_options(self.generated_backend_config.aot_options),
            self.generated_backend_config.clone(),
        );
        self.preparation_timings.symbolic_backend_ms = elapsed_ms(symbolic_started);
        let prepared = prepared_result.map_err(|err| IvpBackendError::GeneratedBackendFailure {
            message: err.to_string(),
        })?;
        self.preparation_backend_telemetry =
            backend_telemetry.map(|telemetry| telemetry.snapshot());

        let callback_wiring_started = start.map(|_| Instant::now());
        self.generated_backend_config.resolver = prepared.updated_resolver.clone();
        let prepared_problem = Arc::new(prepared.into_problem());
        let parameter_values_handle = prepared_problem.parameter_values_handle();
        let residual_problem = Arc::clone(&prepared_problem);
        let residual_callback: Arc<BdfPreparedResidual> =
            Arc::new(move |t, y| (residual_problem.residual)(t, y));
        let jacobian_problem = Arc::clone(&prepared_problem);
        let jacobian_callback: Arc<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64>> =
            Arc::new(move |t, y| (jacobian_problem.jacobian)(t, y));
        let jacobian_into_factory = if prepared_problem.supports_jacobian_row_major_workspace() {
            let jacobian_into_problem = Arc::clone(&prepared_problem);
            Some(Arc::new(move || -> Box<BdfPreparedDenseJacobianInto> {
                let problem = Arc::clone(&jacobian_into_problem);
                let mut args = Vec::new();
                // AtomView keeps the native Jacobian plan outside the legacy
                // symbolic matrix field. Use the validated IVP shape rather
                // than inferring it from that compatibility-only field.
                let rows = problem.equations.len();
                let cols = problem.variables.len();
                let mut values = vec![0.0; rows.saturating_mul(cols)];
                Box::new(move |t, y, out: &mut DMatrix<f64>| {
                    problem
                        .try_evaluate_jacobian_into_dmatrix_with_workspace(
                            t,
                            y,
                            out,
                            &mut values,
                            &mut args,
                        )
                        .map_err(|_| BdfJacobianCallbackError::EvaluationFailed)?;
                    Ok(())
                })
            })
                as Arc<dyn Fn() -> Box<BdfPreparedDenseJacobianInto>>)
        } else {
            None
        };
        self.prepared_residual_callback = Some(Arc::clone(&residual_callback));
        self.prepared_jacobian_callback = Some(Arc::clone(&jacobian_callback));
        self.prepared_jacobian_into_factory = jacobian_into_factory.clone();
        let residual_for_runtime = Arc::clone(&residual_callback);
        let wrapped_fun =
            self.instrument_residual(Box::new(move |t, y| residual_for_runtime(t, y)));
        let jacobian_source = if let Some(factory) = jacobian_into_factory.as_ref() {
            BdfJacobianSource::StateDependentDenseInto(self.instrument_jacobian_into(factory()))
        } else {
            let jacobian_for_runtime = Arc::clone(&jacobian_callback);
            let instrumented =
                self.instrument_jacobian(Box::new(move |t, y| jacobian_for_runtime(t, y)));
            BdfJacobianSource::StateDependent(Box::new(move |t, y| {
                BdfJacobian::from_dense(instrumented(t, y))
            }))
        };
        self.parameter_values_handle = parameter_values_handle.clone();
        self.preparation_timings.callback_wiring_ms = elapsed_ms(callback_wiring_started);

        let runtime_initialization_started = start.map(|_| Instant::now());
        let mut Solver_instance = BDF::new();
        self.reported_bdf_counters = BdfOperationCounters::default();
        Solver_instance
            .try_set_max_order_cap(self.max_bdf_order)
            .map_err(|error| IvpBackendError::InvalidArgumentSchema {
                message: error.to_string(),
            })?;
        Solver_instance
            .set_operation_counters_enabled(self.telemetry_mode != BdfTelemetryMode::Off);
        Solver_instance
            .set_operation_timings_enabled(self.telemetry_mode == BdfTelemetryMode::Timings);
        Solver_instance
            .try_set_initial_with_jacobian_source(
                wrapped_fun,
                self.t0,
                self.y0.clone(),
                self.t_bound,
                self.max_step,
                NumberOrVec::Number(self.rtol),
                NumberOrVec::Number(self.atol),
                jacobian_source,
                None,
                self.vectorized,
                self.first_step,
            )
            .map_err(|error| IvpBackendError::InvalidArgumentSchema {
                message: error.to_string(),
            })?;
        self.preparation_timings.runtime_initialization_ms =
            elapsed_ms(runtime_initialization_started);

        let linear_backend_started = start.map(|_| Instant::now());
        if let Some(factory) = self.bdf_linear_backend_factory.as_ref() {
            Solver_instance.set_linear_backend(factory());
        }
        self.preparation_timings.linear_backend_setup_ms = elapsed_ms(linear_backend_started);

        let publication_started = start.map(|_| Instant::now());
        self.Solver_instance = Solver_instance;
        self.mark_backend_prepared();
        self.preparation_timings.runtime_publication_ms = elapsed_ms(publication_started);
        self.preparation_timings.total_ms = elapsed_ms(start);
        self.record_prepare_duration(start);
        Ok(())
    }

    /// Reports whether symbolic/native callbacks and the selected linear
    /// backend have already been prepared for continuation or restart.
    pub fn is_backend_prepared(&self) -> bool {
        self.backend_prepared
    }

    fn try_generate_native_numeric(&mut self) -> Result<(), BdfConfigurationError> {
        self.preparation_timings = BdfPreparationTimings::default();
        self.preparation_backend_telemetry = None;
        self.prepared_jacobian_into_factory = None;
        let start = (self.telemetry_mode == BdfTelemetryMode::Timings).then(Instant::now);
        let callback_wiring_started = start.map(|_| Instant::now());
        let rhs = self
            .native_rhs
            .clone()
            .ok_or(BdfConfigurationError::MissingNativeRhs)?;
        let wrapped_fun = self.instrument_residual(Box::new(move |t, y| rhs(t, y)));

        let jacobian_source = match self
            .native_jacobian_source
            .clone()
            .unwrap_or(BdfNativeJacobianSource::FiniteDifference)
        {
            BdfNativeJacobianSource::FiniteDifference => BdfJacobianSource::FiniteDifference,
            BdfNativeJacobianSource::StateDependent(jacobian) => {
                let jacobian = self.instrument_jacobian(Box::new(move |t, y| jacobian(t, y)));
                BdfJacobianSource::StateDependent(Box::new(move |t, y| {
                    BdfJacobian::from_dense(jacobian(t, y))
                }))
            }
            BdfNativeJacobianSource::Constant(jacobian) => {
                BdfJacobianSource::Constant(BdfJacobian::from_dense(jacobian))
            }
        };
        self.preparation_timings.callback_wiring_ms = elapsed_ms(callback_wiring_started);

        let runtime_initialization_started = start.map(|_| Instant::now());
        let mut solver_instance = BDF::new();
        self.reported_bdf_counters = BdfOperationCounters::default();
        solver_instance.try_set_max_order_cap(self.max_bdf_order)?;
        solver_instance
            .set_operation_counters_enabled(self.telemetry_mode != BdfTelemetryMode::Off);
        solver_instance
            .set_operation_timings_enabled(self.telemetry_mode == BdfTelemetryMode::Timings);
        solver_instance.try_set_initial_with_jacobian_source(
            wrapped_fun,
            self.t0,
            self.y0.clone(),
            self.t_bound,
            self.max_step,
            NumberOrVec::Number(self.rtol),
            NumberOrVec::Number(self.atol),
            jacobian_source,
            self.jac_sparsity.clone(),
            self.vectorized,
            self.first_step,
        )?;
        self.preparation_timings.runtime_initialization_ms =
            elapsed_ms(runtime_initialization_started);

        let linear_backend_started = start.map(|_| Instant::now());
        if let Some(factory) = self.bdf_linear_backend_factory.as_ref() {
            solver_instance.set_linear_backend(factory());
        }
        self.preparation_timings.linear_backend_setup_ms = elapsed_ms(linear_backend_started);

        let publication_started = start.map(|_| Instant::now());
        self.Solver_instance = solver_instance;
        self.mark_backend_prepared();
        self.preparation_timings.runtime_publication_ms = elapsed_ms(publication_started);
        self.preparation_timings.total_ms = elapsed_ms(start);
        self.record_prepare_duration(start);
        Ok(())
    }

    fn try_generate_with_native_jacobian(
        &mut self,
        start: Option<Instant>,
        options: SymbolicIvpProblemOptions,
    ) -> Result<(), IvpBackendError> {
        self.prepared_jacobian_into_factory = None;
        let symbolic_started = start.map(|_| Instant::now());
        let (parameter_values_handle, fun) =
            if let Some(prepared_residual) = self.prepared_generated_residual.take() {
                (self.parameter_values_handle.clone(), prepared_residual)
            } else {
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
                let residual = prepared_problem.residual;
                let residual = Box::new(move |t: f64, y: &DVector<f64>| residual(t, y))
                    as Box<BdfPreparedResidual>;
                (parameter_values_handle, residual)
            };
        self.preparation_timings.symbolic_backend_ms = elapsed_ms(symbolic_started);

        let callback_wiring_started = start.map(|_| Instant::now());
        let residual_callback: Arc<BdfPreparedResidual> = Arc::new(move |t, y| fun(t, y));
        self.prepared_residual_callback = Some(Arc::clone(&residual_callback));
        let residual_for_runtime = Arc::clone(&residual_callback);
        let wrapped_fun =
            self.instrument_residual(Box::new(move |t, y| residual_for_runtime(t, y)));
        self.parameter_values_handle = parameter_values_handle.clone();
        self.preparation_timings.callback_wiring_ms = elapsed_ms(callback_wiring_started);

        let runtime_initialization_started = start.map(|_| Instant::now());
        let mut solver_instance = BDF::new();
        self.reported_bdf_counters = BdfOperationCounters::default();
        solver_instance
            .try_set_max_order_cap(self.max_bdf_order)
            .map_err(|error| IvpBackendError::InvalidArgumentSchema {
                message: error.to_string(),
            })?;
        solver_instance
            .set_operation_counters_enabled(self.telemetry_mode != BdfTelemetryMode::Off);
        solver_instance
            .set_operation_timings_enabled(self.telemetry_mode == BdfTelemetryMode::Timings);
        let Some(factory) = self.bdf_native_jacobian_factory.as_ref() else {
            return Err(IvpBackendError::InvalidArgumentSchema {
                message: "native Jacobian route has no registered factory".to_string(),
            });
        };
        let jacobian_source = BdfJacobianSource::StateDependent(
            self.timed_native_jacobian(factory(parameter_values_handle.clone())),
        );
        solver_instance
            .try_set_initial_with_jacobian_source(
                wrapped_fun,
                self.t0,
                self.y0.clone(),
                self.t_bound,
                self.max_step,
                NumberOrVec::Number(self.rtol),
                NumberOrVec::Number(self.atol),
                jacobian_source,
                None,
                self.vectorized,
                self.first_step,
            )
            .map_err(|error| IvpBackendError::InvalidArgumentSchema {
                message: error.to_string(),
            })?;
        self.preparation_timings.runtime_initialization_ms =
            elapsed_ms(runtime_initialization_started);

        let linear_backend_started = start.map(|_| Instant::now());
        if let Some(factory) = self.bdf_linear_backend_factory.as_ref() {
            solver_instance.set_linear_backend(factory());
        }
        self.preparation_timings.linear_backend_setup_ms = elapsed_ms(linear_backend_started);

        let publication_started = start.map(|_| Instant::now());
        self.Solver_instance = solver_instance;

        self.mark_backend_prepared();
        self.preparation_timings.runtime_publication_ms = elapsed_ms(publication_started);
        self.preparation_timings.total_ms = elapsed_ms(start);
        self.record_prepare_duration(start);
        Ok(())
    }

    fn mark_backend_prepared(&mut self) {
        self.backend_prepared = true;
        self.status = BdfStatus::Running;
        self.message = None;
        self.t_result = DVector::zeros(0);
        self.y_result = DMatrix::zeros(0, self.y0.len());
    }

    fn timed_native_jacobian(
        &self,
        mut jacobian: Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian>,
    ) -> Box<dyn FnMut(f64, &DVector<f64>) -> BdfJacobian> {
        let Some(stats_for_jac) = self.statistics.as_ref().map(Arc::clone) else {
            return jacobian;
        };
        let mode = self.telemetry_mode;
        Box::new(move |t: f64, y: &DVector<f64>| -> BdfJacobian {
            let start = (mode == BdfTelemetryMode::Timings).then(Instant::now);
            let out = jacobian(t, y);
            let Ok(mut stats) = stats_for_jac.lock() else {
                return out;
            };
            if let Some(start) = start {
                stats.record_jacobian_duration(start.elapsed());
            } else {
                stats.jacobian_calls += 1;
            }
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
        let _ = self.try_step();
    }

    fn try_step(&mut self) -> Result<(), BdfStepError> {
        let t = self.Solver_instance.t;
        if t == self.t_bound {
            self.Solver_instance.t_old = Some(t);
            self.status = BdfStatus::Finished;
            return Ok(());
        }

        let (success, error) = self.Solver_instance._step_impl();
        if let Some(error_message) = error {
            self.message = Some(format!("{error_message:?}"));
        } else {
            self.message = None;
        }

        if !success {
            let error = error.unwrap_or(BdfStepError::StepSizeUnderflow);
            self.status = BdfStatus::Failed;
            return Err(error);
        }

        self.Solver_instance.t_old = Some(t);
        if self.Solver_instance.direction * (self.Solver_instance.t - self.t_bound) >= 0.0 {
            self.status = BdfStatus::Finished;
        }
        Ok(())
    }

    /// Executes a complete solve and returns a typed failure while preserving
    /// the initial state and every accepted point in `get_result_ref()`.
    pub fn try_solve(&mut self) -> Result<(), BdfSolveError> {
        if self.max_steps == 0 {
            return Err(BdfSolveError::InvalidMaxSteps);
        }
        self.validate_callback_options()
            .map_err(BdfSolveError::Configuration)?;
        if !self.backend_prepared {
            if self.native_rhs.is_some() {
                self.try_generate_native_numeric()
                    .map_err(BdfSolveError::Configuration)?;
            } else {
                self.try_generate().map_err(BdfSolveError::Backend)?;
            }
        }
        self.try_main_loop()
    }

    fn validate_callback_options(&self) -> Result<(), BdfConfigurationError> {
        if let Some(pattern) = &self.jac_sparsity {
            if pattern.shape() != (self.y0.len(), self.y0.len()) {
                return Err(BdfConfigurationError::SparsityDimension);
            }
            if self.native_rhs.is_some()
                && matches!(
                    &self.native_jacobian_source,
                    None | Some(BdfNativeJacobianSource::FiniteDifference)
                )
            {
                return Err(BdfConfigurationError::UnsupportedJacobianSparsity);
            }
        }
        if let Some(BdfNativeJacobianSource::Constant(jacobian)) = &self.native_jacobian_source {
            if jacobian.shape() != (self.y0.len(), self.y0.len()) {
                return Err(BdfConfigurationError::JacobianDimension);
            }
            if jacobian.iter().any(|value| !value.is_finite()) {
                return Err(BdfConfigurationError::NonFiniteJacobian);
            }
        }
        if self.vectorized {
            return Err(BdfConfigurationError::UnsupportedVectorizedRhs);
        }
        Ok(())
    }

    fn try_main_loop(&mut self) -> Result<(), BdfSolveError> {
        if self.integration_is_finished() {
            return Ok(());
        }

        let solve_start = (self.telemetry_mode == BdfTelemetryMode::Timings).then(Instant::now);
        let integration_start = solve_start;
        let mut times = Vec::new();
        let mut states = Vec::new();
        self.append_current_state(&mut times, &mut states);

        if self.check_stop_condition(&self.Solver_instance.y) {
            self.status = BdfStatus::StoppedByCondition;
        } else if self.Solver_instance.t == self.t_bound {
            self.status = BdfStatus::Finished;
        }

        let mut step_calls = 0usize;
        let mut bdf_step_ms = 0.0;
        let mut output_collection_ms = 0.0;
        let mut failure = None;

        while !self.integration_is_finished() {
            if step_calls >= self.max_steps {
                self.status = BdfStatus::Failed;
                let error = BdfSolveError::MaxStepsExceeded {
                    max_steps: self.max_steps,
                };
                self.message = Some(error.to_string());
                failure = Some(error);
                break;
            }

            step_calls += 1;
            let step_start = (self.telemetry_mode == BdfTelemetryMode::Timings).then(Instant::now);
            let step_result = self.try_step();
            if let Some(start) = step_start {
                bdf_step_ms += start.elapsed().as_secs_f64() * 1_000.0;
            }
            if let Err(error) = step_result {
                failure = Some(BdfSolveError::Step(error));
                break;
            }

            let collection_start =
                (self.telemetry_mode == BdfTelemetryMode::Timings).then(Instant::now);
            self.append_current_state(&mut times, &mut states);
            if let Some(start) = collection_start {
                output_collection_ms += start.elapsed().as_secs_f64() * 1_000.0;
            }
            if self.status != BdfStatus::Finished
                && self.check_stop_condition(&self.Solver_instance.y)
            {
                self.status = BdfStatus::StoppedByCondition;
            }
        }

        let integration_ms = integration_start.map(|start| start.elapsed().as_secs_f64() * 1_000.0);
        let assembly_start = (self.telemetry_mode == BdfTelemetryMode::Timings).then(Instant::now);
        let cols = self.y0.len();
        let rows = states.len();
        let mut flat = Vec::with_capacity(rows * cols);
        for state in states {
            flat.extend(state.iter().copied());
        }
        self.y_result = DMatrix::from_vec(cols, rows, flat).transpose();
        self.t_result = DVector::from_vec(times);
        let assembly_ms = assembly_start.map(|start| start.elapsed().as_secs_f64() * 1_000.0);
        let solve_ms = solve_start.map(|start| start.elapsed().as_secs_f64() * 1_000.0);

        if let Some(stats) = self.statistics.as_ref() {
            let current = self.Solver_instance.operation_counters();
            let previous = self.reported_bdf_counters;
            self.reported_bdf_counters = current;
            let Ok(mut stats) = stats.lock() else {
                return failure.map_or(Ok(()), Err);
            };
            stats.solve_calls += 1;
            stats.step_calls += step_calls;
            stats.accepted_steps_total += current
                .accepted_steps
                .saturating_sub(previous.accepted_steps);
            stats.candidate_step_attempts_total += current
                .candidate_step_attempts
                .saturating_sub(previous.candidate_step_attempts);
            stats.rejected_step_attempts_total += current
                .rejected_step_attempts
                .saturating_sub(previous.rejected_step_attempts);
            stats.linear_solve_attempts_total += current
                .linear_solve_attempts
                .saturating_sub(previous.linear_solve_attempts);
            stats.nonlinear_solve_calls += current
                .nonlinear_solves
                .saturating_sub(previous.nonlinear_solves);
            stats.nonlinear_iterations_total += current
                .nonlinear_iterations
                .saturating_sub(previous.nonlinear_iterations);
            stats.bdf_nfev_total += current
                .rhs_evaluations
                .saturating_sub(previous.rhs_evaluations);
            stats.bdf_njev_total += current
                .jacobian_evaluations
                .saturating_sub(previous.jacobian_evaluations);
            stats.bdf_nlu_total += current
                .factorization_attempts
                .saturating_sub(previous.factorization_attempts);
            stats.linear_factorization_ms_total += (current.linear_factorization_ms_total
                - previous.linear_factorization_ms_total)
                .max(0.0);
            stats.linear_matrix_assembly_ms_total += (current.linear_matrix_assembly_ms_total
                - previous.linear_matrix_assembly_ms_total)
                .max(0.0);
            stats.linear_solve_ms_total +=
                (current.linear_solve_ms_total - previous.linear_solve_ms_total).max(0.0);
            stats.bdf_step_snapshot_ms_total +=
                (current.step_snapshot_ms_total - previous.step_snapshot_ms_total).max(0.0);
            stats.bdf_step_predictor_setup_ms_total += (current.step_predictor_setup_ms_total
                - previous.step_predictor_setup_ms_total)
                .max(0.0);
            stats.bdf_newton_rhs_assembly_ms_total += (current.newton_rhs_assembly_ms_total
                - previous.newton_rhs_assembly_ms_total)
                .max(0.0);
            stats.bdf_newton_correction_norm_ms_total += (current.newton_correction_norm_ms_total
                - previous.newton_correction_norm_ms_total)
                .max(0.0);
            stats.bdf_newton_state_update_ms_total += (current.newton_state_update_ms_total
                - previous.newton_state_update_ms_total)
                .max(0.0);
            stats.bdf_step_error_estimate_ms_total += (current.step_error_estimate_ms_total
                - previous.step_error_estimate_ms_total)
                .max(0.0);
            stats.bdf_step_nordsieck_update_ms_total += (current.step_nordsieck_update_ms_total
                - previous.step_nordsieck_update_ms_total)
                .max(0.0);
            if let Some(ms) = assembly_ms {
                stats.result_assembly_ms_total += ms;
            }
            if let Some(ms) = integration_ms {
                stats.integration_loop_ms_total += ms;
            }
            stats.bdf_step_ms_total += bdf_step_ms;
            stats.output_collection_ms_total += output_collection_ms;
            if let Some(ms) = solve_ms {
                stats.solve_ms_total += ms;
            }
        }

        failure.map_or(Ok(()), Err)
    }

    /// Compatibility adapter. Use [`Self::try_solve`] to handle typed errors.
    pub fn solve(&mut self) {
        self.try_solve()
            .expect("BDF integration should complete successfully");
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
    /// - **Timing**: Optional timings are returned through `get_statistics`.
    ///
    /// # Matrix Assembly Algorithm
    /// ```text
    /// flat_vec = [y₁(t₁), y₂(t₁), ..., yₙ(t₁), y₁(t₂), y₂(t₂), ..., yₙ(tₘ)]
    /// y_result = reshape(flat_vec, n_vars, n_times).transpose()
    /// ```
    pub fn main_loop(&mut self) {
        self.try_main_loop()
            .expect("BDF integration loop should complete successfully");
    }

    fn append_current_state(&self, t: &mut Vec<f64>, y: &mut Vec<DVector<f64>>) {
        t.push(self.Solver_instance.t);
        y.push(self.Solver_instance.y.clone());
    }

    fn integration_is_finished(&self) -> bool {
        matches!(
            self.status,
            BdfStatus::Finished | BdfStatus::StoppedByCondition
        )
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
    /// Generates plots of the solution using the built-in plotting utility.
    ///
    /// Creates time-series plots for all solution variables.
    /// Requires the solution to be computed first via `solve()`.
    pub fn plot_result(&self) -> () {
        plots_ref(&self.arg, &self.values, &self.t_result, &self.y_result);
    }

    /// Returns the computed solution data. Rows of `y` correspond to entries
    /// in `t`; a completed or partial trajectory begins with the initial state.
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

    /// Borrows the computed trajectory without cloning its time/state arrays.
    /// Rows of `y` correspond to entries in `t`, including the initial state.
    pub fn get_result_ref(&self) -> (&DVector<f64>, &DMatrix<f64>) {
        (&self.t_result, &self.y_result)
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
    /// Stable status enum for callers that want to avoid string matching.
    pub fn status_kind(&self) -> BdfStatus {
        self.status
    }

    /// Compatibility label for the current integration status.
    pub fn get_status(&self) -> &'static str {
        self.status.as_str()
    }

    /// Saves the solution to `bdf_result.csv` in the current directory.
    /// Use [`Self::save_result_to`] to select an explicit path.
    pub fn save_result(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.save_result_to("bdf_result.csv")
    }

    /// Saves a conventional time-by-state CSV table to `path`.
    ///
    /// The header contains the independent-variable name followed by the
    /// state-variable names. Each subsequent row contains one time point.
    pub fn save_result_to(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        let mut writer = Writer::from_path(path)?;
        let mut header = Vec::with_capacity(self.values.len() + 1);
        header.push(self.arg.as_str());
        header.extend(self.values.iter().map(String::as_str));
        writer.write_record(header)?;

        for (row_index, &time) in self.t_result.iter().enumerate() {
            let mut row = Vec::with_capacity(self.values.len() + 1);
            row.push(time.to_string());
            row.extend(self.y_result.row(row_index).iter().map(ToString::to_string));
            writer.write_record(row)?;
        }
        writer.flush()?;
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
    fn bdf_methodless_options_default_to_telemetry_off() {
        let options = BdfSolverOptions::for_bdf(
            vec![Expr::parse_expression("y")],
            vec!["y".to_string()],
            "t".to_string(),
            0.0,
            DVector::from_vec(vec![1.0]),
            1.0,
            0.1,
            1e-6,
            1e-8,
            None,
            false,
            None,
        );
        let solver = ODEsolver::new_with_options(options);

        assert_eq!(solver.telemetry_mode, BdfTelemetryMode::Off);
        assert_eq!(solver.status_kind(), BdfStatus::Running);
        assert_eq!(solver.get_status(), "running");
    }

    #[test]
    fn unsupported_vectorized_rhs_fails_before_backend_preparation() {
        let options = BdfSolverOptions::for_bdf(
            vec![Expr::parse_expression("y")],
            vec!["y".to_string()],
            "t".to_string(),
            0.0,
            DVector::from_vec(vec![1.0]),
            1.0,
            0.1,
            1e-6,
            1e-8,
            None,
            true,
            None,
        );
        let mut solver = ODEsolver::new_with_options(options);

        assert!(matches!(
            solver.try_solve(),
            Err(BdfSolveError::Configuration(
                BdfConfigurationError::UnsupportedVectorizedRhs
            ))
        ));
        assert!(!solver.backend_prepared);
    }

    #[test]
    fn unsupported_jacobian_sparsity_fails_before_backend_preparation() {
        let options = BdfSolverOptions::for_bdf(
            vec![Expr::parse_expression("y")],
            vec!["y".to_string()],
            "t".to_string(),
            0.0,
            DVector::from_vec(vec![1.0]),
            1.0,
            0.1,
            1e-6,
            1e-8,
            Some(DMatrix::from_element(1, 1, 1.0)),
            false,
            None,
        );
        let mut solver = ODEsolver::new_with_options(options);
        solver.set_native_ode_callbacks(
            |_, y: &DVector<f64>| DVector::from_element(y.len(), -y[0]),
            None::<fn(f64, &DVector<f64>) -> DMatrix<f64>>,
        );

        assert!(matches!(
            solver.try_solve(),
            Err(BdfSolveError::Configuration(
                BdfConfigurationError::UnsupportedJacobianSparsity
            ))
        ));
        assert!(!solver.backend_prepared);
    }

    #[test]
    fn native_callback_api_accepts_constant_jacobian_source() {
        let options = BdfSolverOptions::for_bdf(
            vec![Expr::parse_expression("-rate*y")],
            vec!["y".to_string()],
            "t".to_string(),
            0.0,
            DVector::from_vec(vec![1.0]),
            0.2,
            0.05,
            1e-7,
            1e-10,
            None,
            false,
            Some(0.01),
        )
        .with_equation_parameters(vec!["rate".to_string()])
        .with_equation_parameter_values(DVector::from_element(1, 1.0))
        .with_telemetry_mode(BdfTelemetryMode::Counters);
        let mut solver = ODEsolver::new_with_options(options);
        solver.set_native_ode_callbacks_with_jacobian_source(
            |_, y: &DVector<f64>| -y,
            BdfNativeJacobianSource::Constant(DMatrix::from_element(1, 1, -1.0)),
        );

        solver.try_solve().unwrap();

        assert_eq!(solver.get_statistics().bdf_njev_total, 1);
        let (_, trajectory) = solver.get_result_ref();
        let final_state = trajectory[(trajectory.nrows() - 1, 0)];
        assert!((final_state - (-0.2_f64).exp()).abs() < 2e-6);
    }

    #[test]
    fn native_callback_jacobian_shape_error_remains_typed() {
        let options = BdfSolverOptions::for_bdf(
            vec![Expr::parse_expression("-y")],
            vec!["y".to_string()],
            "t".to_string(),
            0.0,
            DVector::from_vec(vec![1.0]),
            0.2,
            0.05,
            1e-7,
            1e-10,
            None,
            false,
            Some(0.01),
        );
        let mut solver = ODEsolver::new_with_options(options);
        solver.set_native_ode_callbacks_with_jacobian_source(
            |_, y: &DVector<f64>| -y,
            BdfNativeJacobianSource::StateDependent(Arc::new(|_, _| DMatrix::zeros(2, 2))),
        );

        assert!(matches!(
            solver.try_solve(),
            Err(BdfSolveError::Configuration(
                BdfConfigurationError::JacobianDimension
            ))
        ));
        assert!(!solver.backend_prepared);
    }

    #[test]
    fn generated_native_jacobian_factory_skips_initial_finite_difference_work() {
        let options = BdfSolverOptions::for_bdf(
            vec![Expr::parse_expression("-rate*y")],
            vec!["y".to_string()],
            "t".to_string(),
            0.0,
            DVector::from_vec(vec![1.0]),
            0.2,
            0.05,
            1e-7,
            1e-10,
            None,
            false,
            Some(0.01),
        )
        .with_equation_parameters(vec!["rate".to_string()])
        .with_equation_parameter_values(DVector::from_element(1, 1.0))
        .with_telemetry_mode(BdfTelemetryMode::Counters);
        let mut solver = ODEsolver::new_with_options(options);
        solver.set_bdf_native_jacobian_factory(|_| {
            Box::new(|_, _| BdfJacobian::from_dense(DMatrix::from_element(1, 1, -1.0)))
        });

        solver.try_generate().unwrap();

        let counters = solver.Solver_instance.operation_counters();
        assert_eq!(
            counters.rhs_evaluations, 1,
            "only the initial RHS is needed"
        );
        assert_eq!(counters.jacobian_evaluations, 1);

        solver
            .try_continue_with_parameter_values(DVector::from_element(1, 2.0), 0.4)
            .unwrap();
        let continuation_counters = solver.Solver_instance.operation_counters();
        assert_eq!(continuation_counters.rhs_evaluations, 1);
        assert_eq!(continuation_counters.jacobian_evaluations, 1);
    }

    #[test]
    fn jacobian_sparsity_does_not_reject_an_analytic_jacobian_route() {
        let mut solver = BDF::new();
        let result = solver.try_set_initial(
            Box::new(|_, y| DVector::from_element(y.len(), -y[0])),
            0.0,
            DVector::from_vec(vec![1.0]),
            1.0,
            0.1,
            NumberOrVec::Number(1e-6),
            NumberOrVec::Number(1e-8),
            Some(Box::new(|_, _| DMatrix::from_element(1, 1, -1.0))),
            Some(DMatrix::from_element(1, 1, 1.0)),
            false,
            None,
        );

        assert_eq!(result, Ok(()));
    }

    #[test]
    #[should_panic(expected = "only the BDF method")]
    fn legacy_method_selector_rejects_unsupported_methods() {
        validate_legacy_method("Radau");
    }

    #[test]
    fn stop_conditions_are_validated_and_pre_resolved() {
        let mut solver = ODEsolver::new_with_options(BdfSolverOptions::for_bdf(
            vec![Expr::parse_expression("y1"), Expr::parse_expression("y2")],
            vec!["y1".to_string(), "y2".to_string()],
            "t".to_string(),
            0.0,
            DVector::from_vec(vec![0.0, 0.0]),
            1.0,
            0.1,
            1e-6,
            1e-8,
            None,
            false,
            None,
        ));

        solver
            .try_set_stop_condition(HashMap::from([("y2".to_string(), 3.0)]))
            .unwrap();
        assert_eq!(solver.stop_condition, Some(vec![(1, 3.0)]));
        assert!(solver.check_stop_condition(&DVector::from_vec(vec![0.0, 3.0])));
        assert!(!solver.check_stop_condition(&DVector::from_vec(vec![3.0, 0.0])));

        assert_eq!(
            solver.try_set_stop_condition(HashMap::from([("missing".to_string(), 1.0)])),
            Err(BdfStopConditionError::UnknownVariable(
                "missing".to_string()
            ))
        );
        assert_eq!(
            solver.try_set_stop_condition(HashMap::from([("y1".to_string(), f64::NAN)])),
            Err(BdfStopConditionError::NonFiniteTarget("y1".to_string()))
        );
        assert_eq!(solver.stop_condition, Some(vec![(1, 3.0)]));
    }

    #[test]
    fn save_result_to_writes_time_by_state_csv() {
        let mut solver = ODEsolver::new_with_options(BdfSolverOptions::for_bdf(
            vec![Expr::parse_expression("y1"), Expr::parse_expression("y2")],
            vec!["y1".to_string(), "y2".to_string()],
            "time".to_string(),
            0.0,
            DVector::from_vec(vec![1.0, 2.0]),
            1.0,
            0.1,
            1e-6,
            1e-8,
            None,
            false,
            None,
        ));
        solver.t_result = DVector::from_vec(vec![0.0, 0.5]);
        solver.y_result = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 0.5, 1.0]);
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("bdf-result-{unique}.csv"));

        solver.save_result_to(&path).unwrap();

        let mut reader = csv::Reader::from_path(&path).unwrap();
        assert_eq!(
            reader.headers().unwrap().iter().collect::<Vec<_>>(),
            vec!["time", "y1", "y2"]
        );
        let records = reader.records().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].iter().collect::<Vec<_>>(), vec!["0", "1", "2"]);
        assert_eq!(
            records[1].iter().collect::<Vec<_>>(),
            vec!["0.5", "0.5", "1"]
        );
        std::fs::remove_file(path).unwrap();
    }

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
        let final_energy = 0.5 * final_omega.powi(2) - (final_theta.cos() - 1.0_f64.cos());

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
        // Stop conditions are sampled at accepted steps, so use a cap fine
        // enough for the requested 1e-3 state neighborhood.
        let max_step = 0.001;
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
        let max_step = 0.001;
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
        let max_step = 0.001;
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
    use std::panic::{catch_unwind, AssertUnwindSafe};
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
                solver.get_status().to_string(),
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

#[cfg(test)]
#[path = "tests/backend_story_tests.rs"]
mod backend_story_tests;
