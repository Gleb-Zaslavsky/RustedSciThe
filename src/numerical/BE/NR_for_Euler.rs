use super::{BeDetailedStatistics, BeFailureKind, BeSymbolicAssemblyBackend, BeTelemetryMode};
use crate::symbolic::ivp_telemetry::{IvpTelemetry, IvpTelemetryMode, IvpTelemetrySnapshot};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp::{
    IvpBackendError, PreparedSymbolicIvpProblem, SharedIvpParameterValues,
    SymbolicIvpProblemOptions,
};
use crate::symbolic::symbolic_ivp_generated::{
    prepare_generated_symbolic_ivp_problem, DenseIvpGeneratedBackendMode, IvpBackendStatistics,
    SymbolicIvpGeneratedBackendConfig,
};
use log::info;
use nalgebra::{DMatrix, DVector, Matrix};
use std::fmt::Display;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Debug, Clone, PartialEq)]
pub enum NreError {
    InvalidConfiguration(&'static str),
    InvalidResidualShape {
        expected: usize,
        actual: usize,
    },
    InvalidJacobianShape {
        expected_rows: usize,
        expected_cols: usize,
        actual_rows: usize,
        actual_cols: usize,
    },
    NonFiniteCallback {
        stage: &'static str,
    },
    SingularNewtonMatrix,
    NonFiniteNewtonState,
    NonConvergence {
        max_iterations: usize,
    },
}

impl std::fmt::Display for NreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfiguration(message) => {
                write!(f, "invalid Newton configuration: {message}")
            }
            Self::InvalidResidualShape { expected, actual } => {
                write!(
                    f,
                    "Newton residual has length {actual}, expected {expected}"
                )
            }
            Self::InvalidJacobianShape {
                expected_rows,
                expected_cols,
                actual_rows,
                actual_cols,
            } => write!(
                f,
                "Newton Jacobian has shape {actual_rows}x{actual_cols}, expected {expected_rows}x{expected_cols}"
            ),
            Self::NonFiniteCallback { stage } => {
                write!(f, "Newton {stage} callback returned a non-finite value")
            }
            Self::SingularNewtonMatrix => write!(f, "Newton matrix is singular"),
            Self::NonFiniteNewtonState => write!(f, "Newton iteration produced a non-finite state"),
            Self::NonConvergence { max_iterations } => {
                write!(
                    f,
                    "Newton iteration did not converge in {max_iterations} iterations"
                )
            }
        }
    }
}

impl std::error::Error for NreError {}

/// Selects how NRE obtains the timestep used by a Newton solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NreStepMode {
    /// Use the configured `dt` unchanged.
    Fixed,
    /// Recompute the legacy heuristic from the current state and time bound.
    LegacyHeuristic,
}

impl NreStepMode {
    const fn from_legacy_global_timestepping(global_timestepping: bool) -> Self {
        if global_timestepping {
            Self::Fixed
        } else {
            Self::LegacyHeuristic
        }
    }
}

fn finite_difference_jacobian(
    fun: &dyn Fn(f64, &DVector<f64>) -> DVector<f64>,
    t: f64,
    y: &DVector<f64>,
    mut on_rhs_evaluation: impl FnMut(),
) -> Result<DMatrix<f64>, NreError> {
    let n = y.len();
    if n == 0 {
        return Ok(DMatrix::zeros(0, 0));
    }
    on_rhs_evaluation();
    let f0 = fun(t, y);
    validate_residual(&f0, n)?;
    let mut jac = DMatrix::zeros(n, n);
    let mut y_pert = y.clone();
    let eps_base = f64::EPSILON.sqrt();
    for col in 0..n {
        let h = eps_base * (1.0 + y[col].abs());
        y_pert[col] += h;
        on_rhs_evaluation();
        let f1 = fun(t, &y_pert);
        y_pert[col] = y[col];
        validate_residual(&f1, n)?;
        for row in 0..n {
            jac[(row, col)] = (f1[row] - f0[row]) / h;
        }
    }
    Ok(jac)
}
// solve algebraic nonlinear system with free parameter t
//#[derive(Debug)]
#[derive(Clone)]
pub struct NreSolverOptions {
    pub eq_system: Vec<Expr>,
    pub initial_guess: DVector<f64>,
    pub values: Vec<String>,
    pub arg: String,
    pub tolerance: f64,
    pub max_iterations: usize,
    pub dt: f64,
    pub step_mode: NreStepMode,
    pub t_bound: Option<f64>,
    pub generated_backend_config: SymbolicIvpGeneratedBackendConfig,
    /// Symbolic expression frontend used when callbacks are prepared.
    pub symbolic_assembly_backend: BeSymbolicAssemblyBackend,
}

impl NreSolverOptions {
    pub fn new(
        eq_system: Vec<Expr>,
        initial_guess: DVector<f64>,
        values: Vec<String>,
        arg: String,
        tolerance: f64,
        max_iterations: usize,
        dt: f64,
        global_timestepping: bool,
        t_bound: Option<f64>,
    ) -> Self {
        Self {
            eq_system,
            initial_guess,
            values,
            arg,
            tolerance,
            max_iterations,
            dt,
            step_mode: NreStepMode::from_legacy_global_timestepping(global_timestepping),
            t_bound,
            generated_backend_config: SymbolicIvpGeneratedBackendConfig::defaults(),
            symbolic_assembly_backend: BeSymbolicAssemblyBackend::default(),
        }
    }

    pub fn with_step_mode(mut self, step_mode: NreStepMode) -> Self {
        self.step_mode = step_mode;
        self
    }

    /// Selects the symbolic frontend used to prepare residual/Jacobian callbacks.
    pub fn with_symbolic_assembly_backend(mut self, backend: BeSymbolicAssemblyBackend) -> Self {
        self.symbolic_assembly_backend = backend;
        self
    }

    pub fn with_generated_backend_config(
        mut self,
        config: SymbolicIvpGeneratedBackendConfig,
    ) -> Self {
        self.generated_backend_config = config;
        self
    }

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
    /// This is usually the first compiled backend worth trying for larger stiff
    /// Backward Euler problems.
    pub fn with_dense_generated_backend_c_tcc(self, output_parent_dir: impl Into<PathBuf>) -> Self {
        self.with_generated_backend_config(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_c_tcc(),
        )
    }

    /// Uses compiled dense IVP path via `C + gcc` when runtime throughput matters more.
    ///
    /// Prefer this when repeated dense Newton solves matter more than startup
    /// latency.
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
    /// For Backward Euler this is a reasonable compiled preset once the system
    /// is large enough that Jacobian throughput starts to matter.
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

pub struct NRE {
    pub eq_system: Vec<Expr>,        //
    pub initial_guess: DVector<f64>, // initial guess
    pub values: Vec<String>,
    pub arg: String,
    pub tolerance: f64,               // tolerance
    pub max_iterations: usize,        // max number of iterations
    pub max_error: f64,               // max error
    pub result: Option<DVector<f64>>, // result of the iteration
    pub jacobian: Option<Vec<Vec<Expr>>>,
    pub fun: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
    pub jac: Option<Box<dyn FnMut(f64, &DVector<f64>) -> DMatrix<f64>>>,
    pub equation_parameters: Option<Vec<String>>,
    pub equation_parameter_values: Option<DVector<f64>>,
    pub t: f64,
    pub y: DVector<f64>,
    pub dt: f64,
    n: usize,
    step_mode: NreStepMode,
    pub t_bound: Option<f64>,
    parameter_values_handle: Option<SharedIvpParameterValues>,
    generated_backend_config: SymbolicIvpGeneratedBackendConfig,
    statistics: Arc<Mutex<IvpBackendStatistics>>,
    finite_difference_error: Arc<Mutex<Option<NreError>>>,
    uses_finite_difference_jacobian: bool,
    telemetry_mode: BeTelemetryMode,
    symbolic_ivp_telemetry: IvpTelemetry,
    symbolic_assembly_backend: BeSymbolicAssemblyBackend,
    detailed_statistics: Arc<Mutex<BeDetailedStatistics>>,
}

impl Display for NRE {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        //   write!(f, "{}", self.eq_system);
        write!(
            f,
            "Initial guess: {:?}, tolerance: {}, max_iterations: {}, max_error: {}, result: {:?}",
            self.initial_guess, self.tolerance, self.max_iterations, self.max_error, self.result
        )
    }
}

impl NRE {
    pub fn new(
        eq_system: Vec<Expr>,        //
        initial_guess: DVector<f64>, // initial guess
        values: Vec<String>,
        arg: String,
        tolerance: f64,        // tolerance
        max_iterations: usize, // max number of iterations

        dt: f64,
        global_timestepping: bool,
        t_bound: Option<f64>,
    ) -> NRE {
        //jacobian: Jacobian, initial_guess: Vec<f64>, tolerance: f64, max_iterations: usize, max_error: f64, result: Option<Vec<f64>>
        NRE {
            eq_system,
            initial_guess: initial_guess.clone(),
            values,
            arg,
            tolerance,
            max_iterations,

            dt,
            step_mode: NreStepMode::from_legacy_global_timestepping(global_timestepping),
            t_bound,
            result: None,
            jacobian: None,
            fun: Box::new(|_t, y| y.clone()),
            jac: None,
            equation_parameters: None,
            equation_parameter_values: None,
            t: 0.0,
            y: initial_guess.clone(),
            max_error: 1e-3,
            n: 0,
            parameter_values_handle: None,
            generated_backend_config: SymbolicIvpGeneratedBackendConfig::defaults(),
            statistics: Arc::new(Mutex::new(IvpBackendStatistics::default())),
            finite_difference_error: Arc::new(Mutex::new(None)),
            uses_finite_difference_jacobian: false,
            telemetry_mode: BeTelemetryMode::Timings,
            symbolic_ivp_telemetry: IvpTelemetry::disabled(),
            symbolic_assembly_backend: BeSymbolicAssemblyBackend::default(),
            detailed_statistics: Arc::new(Mutex::new(BeDetailedStatistics::default())),
        }
    }

    /// Preferred grouped setup path for Newton-Raphson-for-Euler backend.
    pub fn new_with_options(options: NreSolverOptions) -> Self {
        let step_mode = options.step_mode;
        let mut solver = Self::new(
            options.eq_system,
            options.initial_guess,
            options.values,
            options.arg,
            options.tolerance,
            options.max_iterations,
            options.dt,
            true,
            options.t_bound,
        )
        .with_generated_backend_config(options.generated_backend_config);
        solver.set_step_mode(step_mode);
        solver.set_symbolic_assembly_backend(options.symbolic_assembly_backend);
        solver
    }

    pub fn set_generated_backend_config(&mut self, config: SymbolicIvpGeneratedBackendConfig) {
        self.generated_backend_config = config;
        self.jac = None;
        self.uses_finite_difference_jacobian = false;
        self.symbolic_ivp_telemetry = IvpTelemetry::disabled();
    }

    /// Selects the symbolic frontend used for subsequent callback preparation.
    pub fn set_symbolic_assembly_backend(&mut self, backend: BeSymbolicAssemblyBackend) {
        if self.symbolic_assembly_backend != backend {
            self.symbolic_assembly_backend = backend;
            self.jac = None;
            self.uses_finite_difference_jacobian = false;
            self.symbolic_ivp_telemetry = IvpTelemetry::disabled();
        }
    }

    /// Returns the selected symbolic frontend.
    pub fn symbolic_assembly_backend(&self) -> BeSymbolicAssemblyBackend {
        self.symbolic_assembly_backend
    }

    pub fn step_mode(&self) -> NreStepMode {
        self.step_mode
    }

    pub fn set_step_mode(&mut self, step_mode: NreStepMode) {
        self.step_mode = step_mode;
    }

    pub fn generated_backend_config(&self) -> &SymbolicIvpGeneratedBackendConfig {
        &self.generated_backend_config
    }

    pub fn statistics(&self) -> IvpBackendStatistics {
        if !self.telemetry_mode.collects_counters() {
            return IvpBackendStatistics::default();
        }
        self.statistics
            .lock()
            .expect("IVP statistics lock poisoned")
            .clone()
    }

    pub(crate) fn detailed_statistics(&self) -> BeDetailedStatistics {
        if !self.telemetry_mode.collects_counters() {
            return BeDetailedStatistics::default();
        }
        self.detailed_statistics
            .lock()
            .expect("BE detailed statistics lock poisoned")
            .clone()
    }

    fn update_detailed_statistics(&self, update: impl FnOnce(&mut BeDetailedStatistics)) {
        if self.telemetry_mode.collects_counters() {
            update(
                &mut self
                    .detailed_statistics
                    .lock()
                    .expect("BE detailed statistics lock poisoned"),
            );
        }
    }

    pub(crate) fn record_parameter_bind_result(
        &self,
        elapsed: Option<std::time::Duration>,
        succeeded: bool,
    ) {
        self.update_detailed_statistics(|stats| stats.record_parameter_bind(elapsed, succeeded));
    }

    pub(crate) fn set_telemetry_mode(&mut self, mode: BeTelemetryMode) {
        self.telemetry_mode = mode;
    }

    pub(crate) fn record_accepted_step(&self) {
        self.update_detailed_statistics(BeDetailedStatistics::record_accepted_step);
    }

    pub(crate) fn record_failed_step(&self) {
        self.update_detailed_statistics(BeDetailedStatistics::record_failed_step);
    }

    pub(crate) fn record_failure(&self, kind: BeFailureKind) {
        self.update_detailed_statistics(|stats| stats.record_failure(kind));
    }

    pub(crate) fn record_output_assembly(&self, elapsed: Option<std::time::Duration>) {
        self.update_detailed_statistics(|stats| stats.record_output_assembly(elapsed));
    }

    pub(crate) fn statistics_handle(&self) -> Arc<Mutex<IvpBackendStatistics>> {
        Arc::clone(&self.statistics)
    }

    pub fn with_generated_backend_config(
        mut self,
        config: SymbolicIvpGeneratedBackendConfig,
    ) -> Self {
        self.set_generated_backend_config(config);
        self
    }

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

    pub fn with_dense_generated_backend_mode(mut self, mode: DenseIvpGeneratedBackendMode) -> Self {
        self.set_dense_generated_backend_mode(mode);
        self
    }

    /// Uses compiled dense IVP path via `C + tcc` when startup latency matters most.
    pub fn set_dense_generated_backend_c_tcc(&mut self, output_parent_dir: impl Into<PathBuf>) {
        self.set_generated_backend_config(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_c_tcc(),
        );
    }

    /// Uses compiled dense IVP path via `C + gcc` for runtime-oriented repeated solves.
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
    /// Basic methods to set the equation system

    ///Set system of equations with vector of symbolic expressions
    pub fn set_equation_parameters(&mut self, params: Option<&[&str]>) {
        self.try_set_equation_parameters(params)
            .expect("Newton parameter schema should be valid");
    }

    pub fn try_set_equation_parameters(&mut self, params: Option<&[&str]>) -> Result<(), NreError> {
        if let Some(params) = params {
            let mut names = std::collections::HashSet::with_capacity(params.len());
            for name in params {
                if name.trim().is_empty()
                    || *name == self.arg
                    || self.values.iter().any(|state_name| state_name == *name)
                    || !names.insert(*name)
                {
                    return Err(NreError::InvalidConfiguration(
                        "parameter names must be nonempty, unique, and distinct from state/time names",
                    ));
                }
            }
        }
        self.equation_parameters =
            params.map(|params| params.iter().map(|p| (*p).to_string()).collect());
        self.equation_parameter_values = None;
        self.parameter_values_handle = None;
        self.jac = None;
        self.uses_finite_difference_jacobian = false;
        self.symbolic_ivp_telemetry = IvpTelemetry::disabled();
        self.result = None;
        self.max_error = f64::INFINITY;
        Ok(())
    }

    pub fn set_parameter_values(&mut self, values: DVector<f64>) -> Result<(), IvpBackendError> {
        match self.telemetry_mode {
            BeTelemetryMode::Off => self.set_parameter_values_inner(values),
            BeTelemetryMode::Counters => {
                let result = self.set_parameter_values_inner(values);
                self.record_parameter_bind_result(None, result.is_ok());
                result
            }
            BeTelemetryMode::Timings => {
                let start = Instant::now();
                let result = self.set_parameter_values_inner(values);
                self.record_parameter_bind_result(Some(start.elapsed()), result.is_ok());
                result
            }
        }
    }

    fn set_parameter_values_inner(&mut self, values: DVector<f64>) -> Result<(), IvpBackendError> {
        if !values.iter().all(|value| value.is_finite()) {
            return Err(IvpBackendError::InvalidArgumentSchema {
                message: "equation parameter values must be finite".to_string(),
            });
        }
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

    pub(crate) fn install_prepared_backend(&mut self, prepared: PreparedSymbolicIvpProblem) {
        self.uses_finite_difference_jacobian = false;
        self.symbolic_ivp_telemetry = prepared.telemetry.clone();
        self.jacobian = Some(prepared.symbolic_jacobian.clone());
        self.parameter_values_handle = prepared.parameter_values_handle();
        self.equation_parameters = prepared.equation_parameters.clone();

        let residual = prepared.residual;
        let jacobian = prepared.jacobian;
        let stats_for_residual = self.statistics_handle();
        self.fun = match self.telemetry_mode {
            BeTelemetryMode::Off => Box::new(residual),
            BeTelemetryMode::Counters => Box::new(move |t, y| {
                let out = residual(t, y);
                let mut stats = stats_for_residual
                    .lock()
                    .expect("IVP statistics lock poisoned");
                stats.residual_calls = stats.residual_calls.saturating_add(1);
                out
            }),
            BeTelemetryMode::Timings => Box::new(move |t, y| {
                let start = Instant::now();
                let out = residual(t, y);
                stats_for_residual
                    .lock()
                    .expect("IVP statistics lock poisoned")
                    .record_residual_duration(start.elapsed());
                out
            }),
        };
        let stats_for_jacobian = self.statistics_handle();
        self.jac = Some(match self.telemetry_mode {
            BeTelemetryMode::Off => Box::new(jacobian),
            BeTelemetryMode::Counters => Box::new(move |t, y| {
                let out = jacobian(t, y);
                let mut stats = stats_for_jacobian
                    .lock()
                    .expect("IVP statistics lock poisoned");
                stats.jacobian_calls = stats.jacobian_calls.saturating_add(1);
                out
            }),
            BeTelemetryMode::Timings => Box::new(move |t, y| {
                let start = Instant::now();
                let out = jacobian(t, y);
                stats_for_jacobian
                    .lock()
                    .expect("IVP statistics lock poisoned")
                    .record_jacobian_duration(start.elapsed());
                out
            }),
        });
        self.n = self.eq_system.len();
    }

    pub(crate) fn symbolic_ivp_telemetry_for_preparation(&mut self) -> IvpTelemetry {
        let telemetry = match self.telemetry_mode {
            BeTelemetryMode::Off => IvpTelemetry::disabled(),
            BeTelemetryMode::Counters => IvpTelemetry::counters(),
            BeTelemetryMode::Timings => IvpTelemetry::detailed(),
        };
        self.symbolic_ivp_telemetry = telemetry.clone();
        telemetry
    }

    pub fn symbolic_ivp_telemetry_snapshot(&self) -> Option<IvpTelemetrySnapshot> {
        if self.telemetry_mode == BeTelemetryMode::Off {
            return None;
        }
        let snapshot = self.symbolic_ivp_telemetry.snapshot();
        (snapshot.mode != IvpTelemetryMode::Off).then_some(snapshot)
    }

    pub(crate) fn clear_symbolic_ivp_telemetry(&mut self) {
        self.symbolic_ivp_telemetry = IvpTelemetry::disabled();
    }

    pub fn try_eq_generate(&mut self) -> Result<(), IvpBackendError> {
        info!("generating equations and jacobian");
        let start = self.telemetry_mode.collects_timings().then(Instant::now);
        let mut options = SymbolicIvpProblemOptions::new();
        if let Some(parameters) = self.equation_parameters.clone() {
            options = options.with_equation_parameters(parameters);
        }
        if let Some(values) = self.equation_parameter_values.clone() {
            options = options.with_equation_parameter_values(values);
        }
        let telemetry = self.symbolic_ivp_telemetry_for_preparation();
        options = options.with_telemetry(telemetry);
        options = options.with_symbolic_assembly_backend(match self.symbolic_assembly_backend {
            BeSymbolicAssemblyBackend::ExprLegacy => {
                crate::symbolic::symbolic_ivp::IvpSymbolicAssemblyBackend::ExprLegacy
            }
            BeSymbolicAssemblyBackend::AtomViewNative => {
                crate::symbolic::symbolic_ivp::IvpSymbolicAssemblyBackend::AtomView
            }
        });

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
        self.install_prepared_backend(prepared.into_problem());
        if self.telemetry_mode.collects_counters() {
            let mut stats = self
                .statistics
                .lock()
                .expect("IVP statistics lock poisoned");
            if let Some(start) = start {
                stats.record_backend_prepare_duration(start.elapsed());
            } else {
                stats.backend_prepare_calls = stats.backend_prepare_calls.saturating_add(1);
            }
        }
        assert_eq!(&self.eq_system.len(), &self.n);
        Ok(())
    }

    pub fn eq_generate(&mut self) {
        self.try_eq_generate()
            .expect("NR_for_Euler symbolic IVP backend generation should succeed");
    }

    pub fn set_new_step(&mut self, t: f64, y: DVector<f64>, initial_guess: DVector<f64>) {
        self.t = t;
        self.y = y;
        self.initial_guess = initial_guess;
    }
    pub fn set_t(&mut self, t: f64) {
        self.t = t;
    }
    pub fn set_initial_guess(&mut self, initial_guess: DVector<f64>) {
        self.initial_guess = initial_guess;
        self.result = None;
        self.max_error = f64::INFINITY;
    }

    /// Chooses one legacy heuristic step from the current state.
    /// BE recomputes it before each step; the returned value is capped by
    /// `max_dt` and remains fixed throughout that step's Newton solve.
    pub fn suggest_step_size(
        &mut self,
        t: f64,
        y: &DVector<f64>,
        max_dt: f64,
    ) -> Result<f64, NreError> {
        if !max_dt.is_finite() || max_dt <= 0.0 {
            return Err(NreError::InvalidConfiguration(
                "maximum step must be finite and positive",
            ));
        }
        self.clear_finite_difference_error();
        let f = (self.fun)(t, y);
        let jac = self.jac.as_mut().ok_or(NreError::InvalidConfiguration(
            "Jacobian callback is not prepared",
        ))?(t, y);
        if let Some(error) = self.take_finite_difference_error() {
            return Err(error);
        }
        let n = y.len();
        validate_callbacks(&f, &jac, n)?;
        let scale = (&jac * &f).amax();
        let dt = if scale > 0.0 {
            (2.0 * self.tolerance / scale).sqrt().min(max_dt)
        } else {
            max_dt
        };
        if !dt.is_finite() || dt <= 0.0 {
            return Err(NreError::InvalidConfiguration(
                "step heuristic produced a non-positive or non-finite step",
            ));
        }
        Ok(dt)
    }

    /// Installs pure numerical callbacks for Newton iterations.
    ///
    /// `fun` is interpreted as ODE RHS `f(t, y)`. If `jac` is `None`, a finite
    /// difference Jacobian of `f` is used.
    pub fn set_native_callbacks(
        &mut self,
        fun: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>>,
        jac: Option<Box<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64>>>,
    ) {
        use std::sync::Arc;
        self.clear_symbolic_ivp_telemetry();
        self.uses_finite_difference_jacobian = jac.is_none();
        let fun = Arc::new(fun);
        let wrapped_fun: Box<dyn Fn(f64, &DVector<f64>) -> DVector<f64>> = match self.telemetry_mode
        {
            BeTelemetryMode::Off => {
                let fun_for_eval = Arc::clone(&fun);
                Box::new(move |t, y| fun_for_eval(t, y))
            }
            BeTelemetryMode::Counters => {
                let stats = self.statistics_handle();
                let fun_for_eval = Arc::clone(&fun);
                Box::new(move |t, y| {
                    let out = fun_for_eval(t, y);
                    let mut stats = stats.lock().expect("IVP statistics lock poisoned");
                    stats.residual_calls = stats.residual_calls.saturating_add(1);
                    out
                })
            }
            BeTelemetryMode::Timings => {
                let stats = self.statistics_handle();
                let fun_for_eval = Arc::clone(&fun);
                Box::new(move |t, y| {
                    let start = Instant::now();
                    let out = fun_for_eval(t, y);
                    stats
                        .lock()
                        .expect("IVP statistics lock poisoned")
                        .record_residual_duration(start.elapsed());
                    out
                })
            }
        };

        let fd_error = Arc::clone(&self.finite_difference_error);
        let wrapped_jac: Box<dyn FnMut(f64, &DVector<f64>) -> DMatrix<f64>> =
            match jac {
                Some(jac_fun) => match self.telemetry_mode {
                    BeTelemetryMode::Off => Box::new(move |t, y| jac_fun(t, y)),
                    BeTelemetryMode::Counters => {
                        let stats = self.statistics_handle();
                        Box::new(move |t, y| {
                            let out = jac_fun(t, y);
                            let mut stats = stats.lock().expect("IVP statistics lock poisoned");
                            stats.jacobian_calls = stats.jacobian_calls.saturating_add(1);
                            out
                        })
                    }
                    BeTelemetryMode::Timings => {
                        let stats = self.statistics_handle();
                        Box::new(move |t, y| {
                            let start = Instant::now();
                            let out = jac_fun(t, y);
                            stats
                                .lock()
                                .expect("IVP statistics lock poisoned")
                                .record_jacobian_duration(start.elapsed());
                            out
                        })
                    }
                },
                None => {
                    let fd_fun = Arc::clone(&fun);
                    match self.telemetry_mode {
                        BeTelemetryMode::Off => Box::new(move |t, y| {
                            match finite_difference_jacobian(fd_fun.as_ref().as_ref(), t, y, || {})
                            {
                                Ok(out) => out,
                                Err(error) => {
                                    *fd_error
                                        .lock()
                                        .expect("finite-difference callback error lock poisoned") =
                                        Some(error);
                                    DMatrix::from_element(y.len(), y.len(), f64::NAN)
                                }
                            }
                        }),
                        BeTelemetryMode::Counters => {
                            let stats = self.statistics_handle();
                            let detailed_stats = Arc::clone(&self.detailed_statistics);
                            Box::new(move |t, y| {
                                let out = match finite_difference_jacobian(
                                    fd_fun.as_ref().as_ref(),
                                    t,
                                    y,
                                    || {
                                        detailed_stats
                                            .lock()
                                            .expect("BE detailed statistics lock poisoned")
                                            .record_fd_rhs_evaluation();
                                    },
                                ) {
                                    Ok(out) => out,
                                    Err(error) => {
                                        *fd_error.lock().expect(
                                            "finite-difference callback error lock poisoned",
                                        ) = Some(error);
                                        DMatrix::from_element(y.len(), y.len(), f64::NAN)
                                    }
                                };
                                let mut stats = stats.lock().expect("IVP statistics lock poisoned");
                                stats.jacobian_calls = stats.jacobian_calls.saturating_add(1);
                                out
                            })
                        }
                        BeTelemetryMode::Timings => {
                            let stats = self.statistics_handle();
                            let detailed_stats = Arc::clone(&self.detailed_statistics);
                            Box::new(move |t, y| {
                                let start = Instant::now();
                                let out = match finite_difference_jacobian(
                                    fd_fun.as_ref().as_ref(),
                                    t,
                                    y,
                                    || {
                                        detailed_stats
                                            .lock()
                                            .expect("BE detailed statistics lock poisoned")
                                            .record_fd_rhs_evaluation();
                                    },
                                ) {
                                    Ok(out) => out,
                                    Err(error) => {
                                        *fd_error.lock().expect(
                                            "finite-difference callback error lock poisoned",
                                        ) = Some(error);
                                        DMatrix::from_element(y.len(), y.len(), f64::NAN)
                                    }
                                };
                                stats
                                    .lock()
                                    .expect("IVP statistics lock poisoned")
                                    .record_jacobian_duration(start.elapsed());
                                out
                            })
                        }
                    }
                }
            };

        self.fun = wrapped_fun;
        self.jac = Some(wrapped_jac);
        self.n = self.values.len().max(self.initial_guess.len());
        self.jacobian = None;
    }

    ///Newton-Raphson method
    /// realize iteration of Newton-Raphson - calculate new iteration vector by using Jacobian matrix
    pub fn iteration(&mut self) -> DVector<f64> {
        let result = match self.telemetry_mode {
            BeTelemetryMode::Off => self.try_iteration::<false, false>(self.dt),
            BeTelemetryMode::Counters => self.try_iteration::<true, false>(self.dt),
            BeTelemetryMode::Timings => self.try_iteration::<true, true>(self.dt),
        };
        result.expect("Newton iteration should succeed")
    }

    fn try_iteration<const COUNTERS: bool, const TIMINGS: bool>(
        &mut self,
        dt: f64,
    ) -> Result<DVector<f64>, NreError> {
        if self.y.is_empty() {
            return Err(NreError::InvalidConfiguration("current state is empty"));
        }
        let t = self.t;
        let y = &self.y;
        self.clear_finite_difference_error();
        let f = (self.fun)(t, &y);
        let mut new_j = self.jac.as_mut().ok_or(NreError::InvalidConfiguration(
            "Jacobian callback is not prepared",
        ))?(t, &y);
        if let Some(error) = self.take_finite_difference_error() {
            return Err(error);
        }
        validate_callbacks(&f, &new_j, y.len())?;
        if !dt.is_finite() || dt <= 0.0 {
            return Err(NreError::InvalidConfiguration(
                "step must be finite and positive",
            ));
        }

        let y_k_minus_1 = &self.initial_guess;

        //   println!("Newton-Raphson iteration {}", &y);
        let new_G = y - y_k_minus_1 - dt * f;
        //   println!("new_f = {:?}", &new_G);

        // Build I - dt*J in the owned callback result to avoid two dense temporaries.
        new_j *= -dt;
        for diagonal in 0..self.n {
            new_j[(diagonal, diagonal)] += 1.0;
        }
        //equation J*deltay  = -G
        let factorization_start = TIMINGS.then(Instant::now);
        let lu = new_j.lu();
        if COUNTERS {
            self.update_detailed_statistics(|stats| {
                stats.record_factorization(factorization_start.map(|start| start.elapsed()));
            });
        }
        let neg_f = -1.0 * new_G;
        let linear_solve_start = TIMINGS.then(Instant::now);
        let solve_result = lu.solve(&neg_f);
        if COUNTERS {
            self.update_detailed_statistics(|stats| {
                stats.record_linear_solve(linear_solve_start.map(|start| start.elapsed()));
            });
        }
        let delta_y = solve_result.ok_or(NreError::SingularNewtonMatrix)?;
        //    println!("delta_y = {:?},\n", &delta_y );
        let new_y: DVector<f64> = y + delta_y;

        if !new_y.iter().all(|value| value.is_finite()) {
            return Err(NreError::NonFiniteNewtonState);
        }
        Ok(new_y)
    }
    // main function to solve the system of equations

    pub fn solve(&mut self) -> Option<DVector<f64>> {
        self.try_solve().ok()
    }

    pub fn try_solve(&mut self) -> Result<DVector<f64>, NreError> {
        match self.telemetry_mode {
            BeTelemetryMode::Off => self.try_solve_impl::<false, false>(),
            BeTelemetryMode::Counters => self.try_solve_impl::<true, false>(),
            BeTelemetryMode::Timings => self.try_solve_impl::<true, true>(),
        }
    }

    fn try_solve_impl<const COUNTERS: bool, const TIMINGS: bool>(
        &mut self,
    ) -> Result<DVector<f64>, NreError> {
        //  println!("solving system of equations with Newton-Raphson method");
        if COUNTERS {
            self.statistics
                .lock()
                .expect("IVP statistics lock poisoned")
                .nonlinear_solve_calls += 1;
        }
        let mut y: DVector<f64> = self.initial_guess.clone();
        self.y = y.clone();
        self.result = None;
        self.max_error = f64::INFINITY;
        let dt = if self.step_mode == NreStepMode::Fixed {
            self.dt
        } else {
            let remaining = self.t_bound.ok_or(NreError::InvalidConfiguration(
                "legacy step heuristic requires a time bound",
            ))? - self.t;
            let initial_y = self.y.clone();
            if remaining > 0.0 {
                self.suggest_step_size(self.t, &initial_y, remaining)?
            } else {
                self.dt
            }
        };
        if !dt.is_finite() || dt <= 0.0 {
            return Err(NreError::InvalidConfiguration(
                "step must be finite and positive",
            ));
        }
        self.dt = dt;
        let mut i = 0;
        while i < self.max_iterations {
            let new_y = self.try_iteration::<COUNTERS, TIMINGS>(dt)?;

            let dy = &new_y - &y;

            let error = Matrix::norm(&dy);
            //  println!("new_y = {:?}, dy = {:?}, error = {}", &new_y, &dy, error);
            if error < self.tolerance {
                //  println!("converged in {} iterations", i);
                self.result = Some(new_y.clone());
                self.max_error = error;
                if COUNTERS {
                    self.statistics
                        .lock()
                        .expect("IVP statistics lock poisoned")
                        .nonlinear_iterations_total += i + 1;
                }
                return Ok(new_y);
            } else {
                y = new_y.clone();
                self.y = new_y;
                i += 1;
                //  if i==5 {panic!("Too many iterations")}
                //  println!("\n \n iteration = {}, error = {}", i, error)
            }
        }
        if COUNTERS {
            self.statistics
                .lock()
                .expect("IVP statistics lock poisoned")
                .nonlinear_iterations_total += i;
        }
        Err(NreError::NonConvergence {
            max_iterations: self.max_iterations,
        })
    }

    pub fn get_result(&self) -> Option<DVector<f64>> {
        self.result.clone()
    }

    fn clear_finite_difference_error(&self) {
        if self.uses_finite_difference_jacobian {
            *self
                .finite_difference_error
                .lock()
                .expect("finite-difference callback error lock poisoned") = None;
        }
    }

    fn take_finite_difference_error(&self) -> Option<NreError> {
        if self.uses_finite_difference_jacobian {
            self.finite_difference_error
                .lock()
                .expect("finite-difference callback error lock poisoned")
                .take()
        } else {
            None
        }
    }
}

fn validate_callbacks(f: &DVector<f64>, jac: &DMatrix<f64>, n: usize) -> Result<(), NreError> {
    validate_residual(f, n)?;
    if jac.shape() != (n, n) {
        return Err(NreError::InvalidJacobianShape {
            expected_rows: n,
            expected_cols: n,
            actual_rows: jac.nrows(),
            actual_cols: jac.ncols(),
        });
    }
    if !jac.iter().all(|value| value.is_finite()) {
        return Err(NreError::NonFiniteCallback { stage: "Jacobian" });
    }
    Ok(())
}

fn validate_residual(f: &DVector<f64>, n: usize) -> Result<(), NreError> {
    if f.len() != n {
        return Err(NreError::InvalidResidualShape {
            expected: n,
            actual: f.len(),
        });
    }
    if !f.iter().all(|value| value.is_finite()) {
        return Err(NreError::NonFiniteCallback { stage: "residual" });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    fn zero_rhs(_: f64, y: &DVector<f64>) -> DVector<f64> {
        DVector::zeros(y.len())
    }

    fn constant_rhs(_: f64, y: &DVector<f64>) -> DVector<f64> {
        DVector::from_element(y.len(), 1.0)
    }

    fn zero_jac(_: f64, y: &DVector<f64>) -> DMatrix<f64> {
        DMatrix::zeros(y.len(), y.len())
    }

    fn wrong_shape_rhs(_: f64, _: &DVector<f64>) -> DVector<f64> {
        DVector::zeros(2)
    }

    fn non_finite_rhs(_: f64, y: &DVector<f64>) -> DVector<f64> {
        DVector::from_element(y.len(), f64::NAN)
    }

    fn decay_rhs(_: f64, y: &DVector<f64>) -> DVector<f64> {
        -y
    }

    fn decay_jac(_: f64, _: &DVector<f64>) -> DMatrix<f64> {
        DMatrix::from_element(1, 1, -1.0)
    }

    #[test]
    fn nre_finite_difference_jacobian_reuses_perturbation_and_restores_state() {
        let state = DVector::from_vec(vec![2.0, 3.0]);
        let mut rhs_calls = 0;
        let jacobian = finite_difference_jacobian(
            &|_, y| DVector::from_vec(vec![y[0] * y[0] + y[1], y[0] * y[1]]),
            0.0,
            &state,
            || rhs_calls += 1,
        )
        .unwrap();

        assert_eq!(rhs_calls, 3);
        assert_eq!(state, DVector::from_vec(vec![2.0, 3.0]));
        assert!((jacobian[(0, 0)] - 4.0).abs() < 1e-7);
        assert!((jacobian[(0, 1)] - 1.0).abs() < 1e-7);
        assert!((jacobian[(1, 0)] - 3.0).abs() < 1e-7);
        assert!((jacobian[(1, 1)] - 2.0).abs() < 1e-7);
    }

    fn make_nre() -> NRE {
        NRE::new(
            vec![Expr::parse_expression("0")],
            DVector::from_vec(vec![1.0]),
            vec!["y".to_string()],
            "t".to_string(),
            1e-8,
            4,
            0.1,
            true,
            None,
        )
    }

    #[test]
    fn nre_failure_clears_previous_result_and_validates_callback_shape() {
        let mut solver = make_nre();
        solver.set_native_callbacks(Box::new(zero_rhs), Some(Box::new(zero_jac)));
        solver.try_solve().unwrap();
        assert!(solver.get_result().is_some());

        solver.set_native_callbacks(Box::new(wrong_shape_rhs), Some(Box::new(zero_jac)));
        assert!(matches!(
            solver.try_solve(),
            Err(NreError::InvalidResidualShape {
                expected: 1,
                actual: 2
            })
        ));
        assert!(solver.get_result().is_none());
    }

    #[test]
    fn nre_finite_difference_callback_errors_are_typed() {
        let mut solver = make_nre();
        solver.set_native_callbacks(Box::new(wrong_shape_rhs), None);
        assert!(solver.uses_finite_difference_jacobian);
        assert!(matches!(
            solver.try_solve(),
            Err(NreError::InvalidResidualShape {
                expected: 1,
                actual: 2
            })
        ));

        solver.set_native_callbacks(Box::new(non_finite_rhs), None);
        assert!(matches!(
            solver.try_solve(),
            Err(NreError::NonFiniteCallback { stage: "residual" })
        ));
    }

    #[test]
    fn nre_analytic_jacobian_skips_finite_difference_error_channel() {
        let mut solver = make_nre();
        solver.set_native_callbacks(Box::new(zero_rhs), Some(Box::new(zero_jac)));

        assert!(!solver.uses_finite_difference_jacobian);
        solver.try_solve().unwrap();
    }

    #[test]
    fn nre_native_callback_replacement_clears_symbolic_telemetry() {
        let mut solver = make_nre();
        solver.try_eq_generate().unwrap();
        assert!(solver.symbolic_ivp_telemetry_snapshot().is_some());

        solver.set_native_callbacks(Box::new(zero_rhs), Some(Box::new(zero_jac)));

        assert!(solver.symbolic_ivp_telemetry_snapshot().is_none());
    }

    #[test]
    fn nre_iteration_limit_returns_typed_nonconvergence_without_result() {
        let mut solver = make_nre();
        solver.max_iterations = 1;
        solver.set_native_callbacks(Box::new(constant_rhs), Some(Box::new(zero_jac)));

        assert!(matches!(
            solver.try_solve(),
            Err(NreError::NonConvergence { max_iterations: 1 })
        ));
        assert!(solver.get_result().is_none());
    }

    #[test]
    fn nre_legacy_step_heuristic_is_positive_bounded_and_uses_newton_tolerance() {
        let mut solver = make_nre();
        solver.tolerance = 0.5;
        solver.set_native_callbacks(Box::new(decay_rhs), Some(Box::new(decay_jac)));

        let capped = solver
            .suggest_step_size(0.0, &DVector::from_vec(vec![1.0]), 0.25)
            .unwrap();
        let uncapped = solver
            .suggest_step_size(0.0, &DVector::from_vec(vec![1.0]), 2.0)
            .unwrap();

        assert_eq!(capped, 0.25);
        assert_eq!(uncapped, 1.0);
        assert_eq!(solver.statistics().residual_calls, 2);
        assert_eq!(solver.statistics().jacobian_calls, 2);
        assert!(solver
            .suggest_step_size(0.0, &DVector::from_vec(vec![1.0]), 0.0)
            .is_err());
    }

    #[test]
    fn nre_parameter_schema_rejects_collisions_without_mutation() {
        let mut solver = make_nre();
        solver.try_set_equation_parameters(Some(&["rate"])).unwrap();
        solver
            .set_parameter_values(DVector::from_vec(vec![2.0]))
            .unwrap();

        assert!(solver.try_set_equation_parameters(Some(&["y"])).is_err());
        assert_eq!(
            solver.equation_parameters.as_deref(),
            Some(&["rate".to_string()][..])
        );
        assert_eq!(
            solver
                .equation_parameter_values
                .as_ref()
                .unwrap()
                .as_slice(),
            &[2.0]
        );
        assert!(solver
            .set_parameter_values(DVector::from_vec(vec![f64::INFINITY]))
            .is_err());
    }

    #[test]
    fn nre_new_with_options_installs_generated_backend_mode() {
        let nr = NRE::new_with_options(
            NreSolverOptions::new(
                vec![Expr::parse_expression("y")],
                DVector::from_vec(vec![1.0]),
                vec!["y".to_string()],
                "t".to_string(),
                1e-6,
                20,
                1e-3,
                true,
                None,
            )
            .with_dense_generated_backend_mode(DenseIvpGeneratedBackendMode::RequirePrebuilt),
        );

        assert_eq!(
            nr.generated_backend_config().build_policy,
            crate::symbolic::symbolic_ivp_generated::SymbolicIvpAotBuildPolicy::RequirePrebuilt
        );
        assert_eq!(nr.step_mode(), NreStepMode::Fixed);
    }

    #[test]
    fn nre_typed_step_mode_overrides_legacy_constructor_flag() {
        let solver = NRE::new_with_options(
            NreSolverOptions::new(
                vec![Expr::parse_expression("y")],
                DVector::from_vec(vec![1.0]),
                vec!["y".to_string()],
                "t".to_string(),
                1e-6,
                20,
                1e-3,
                true,
                Some(1.0),
            )
            .with_step_mode(NreStepMode::LegacyHeuristic),
        );
        assert_eq!(solver.step_mode(), NreStepMode::LegacyHeuristic);
    }

    #[test]
    fn nre_generated_backend_surface_keeps_selected_c_backend() {
        let nr = NRE::new_with_options(
            NreSolverOptions::new(
                vec![Expr::parse_expression("y")],
                DVector::from_vec(vec![1.0]),
                vec!["y".to_string()],
                "t".to_string(),
                1e-6,
                20,
                1e-3,
                true,
                None,
            )
            .with_dense_generated_backend_c_gcc("target/generated-ivp-tests")
            .with_dense_generated_backend_mode(DenseIvpGeneratedBackendMode::BuildIfMissingRelease),
        );

        assert_eq!(
            nr.generated_backend_config().aot_codegen_backend,
            crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend::C
        );
        assert_eq!(
            nr.generated_backend_config().aot_c_compiler.as_deref(),
            Some("gcc")
        );
    }

    #[test]
    fn nre_generated_backend_repeated_solves_alias_prefers_c_gcc() {
        let nr = NRE::new_with_options(
            NreSolverOptions::new(
                vec![Expr::parse_expression("y")],
                DVector::from_vec(vec![1.0]),
                vec!["y".to_string()],
                "t".to_string(),
                1e-6,
                20,
                1e-3,
                true,
                None,
            )
            .with_dense_generated_backend_for_repeated_solves("target/generated-ivp-tests"),
        );

        assert_eq!(
            nr.generated_backend_config().aot_codegen_backend,
            crate::symbolic::codegen::codegen_aot_driver::AotCodegenBackend::C
        );
        assert_eq!(
            nr.generated_backend_config().aot_c_compiler.as_deref(),
            Some("gcc")
        );
    }

    #[test]
    fn test_newton_raphson_solver_for_Euler() {
        let eq1 = Expr::parse_expression("z+y-10.0*x");
        let eq2 = Expr::parse_expression("z*y-4.0*x");
        let eq_system = vec![eq1, eq2];
        info!("eq_system = {:?}", eq_system);
        let initial_guess = DVector::from_vec(vec![1.0, 1.0]);
        let values = vec!["z".to_string(), "y".to_string()];
        let arg = "x".to_string();
        let tolerance = 1e-3;
        let max_iterations = 50;

        let h = 1e-5;

        assert_eq!(&eq_system.len(), &2);
        let mut nr = NRE::new(
            eq_system,
            initial_guess,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            true,
            None,
        );
        nr.eq_generate();

        assert_eq!(nr.eq_system.len(), 2);
        nr.set_t(1.0);
        let solution = nr.solve().unwrap();
        assert_eq!(solution.len(), 2);
        // assert_eq!()
        /*
        // Check if the solution is close to the expected value
        let expected_solution = DVector::from_vec(vec![3.0, 4.0]);
        assert!((solution - expected_solution).norm() < tolerance);
         */
    }

    #[test]
    fn test_newton_raphson_solver_for_Euler_2() {
        let eq1 = Expr::parse_expression("z+y-10.0*x");
        let eq2 = Expr::parse_expression("z*y-4.0*x");
        let eq_system = vec![eq1, eq2];
        info!("eq_system = {:?}", eq_system);
        let initial_guess = DVector::from_vec(vec![1.0, 1.0]);
        let values = vec!["z".to_string(), "y".to_string()];
        let arg = "x".to_string();
        let tolerance = 1e-3;
        let max_iterations = 50;

        let h = 1e-5;

        assert_eq!(&eq_system.len(), &2);
        let mut nr = NRE::new(
            eq_system,
            initial_guess,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            false,
            Some(1.0),
        );
        nr.eq_generate();

        assert_eq!(nr.eq_system.len(), 2);
        nr.set_t(1.0);
        let solution = nr.solve().unwrap();
        assert_eq!(solution.len(), 2);
        // assert_eq!()
        /*
        // Check if the solution is close to the expected value
        let expected_solution = DVector::from_vec(vec![3.0, 4.0]);
        assert!((solution - expected_solution).norm() < tolerance);
         */
    }
}
