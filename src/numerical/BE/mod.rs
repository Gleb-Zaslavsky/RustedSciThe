use core::fmt::Display;

use crate::Utils::plots::plots;
/// Backward Euler method for solving systems of ordinary differential equation
/// Newton-Raphson calculation on each step of the method is made by using the analytic jacobian
pub mod NR_for_Euler;

use self::NR_for_Euler::{NreError, NreStepMode, NRE};
use crate::symbolic::symbolic_engine::Expr;
use crate::symbolic::symbolic_ivp::IvpBackendError;
use crate::symbolic::symbolic_ivp_generated::{
    prepare_generated_symbolic_ivp_problem, DenseIvpGeneratedBackendMode, IvpBackendStatistics,
    SymbolicIvpGeneratedBackendConfig, SymbolicIvpGeneratedError,
};
use log::info;
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

/// Default upper bound for BE step attempts in one integration segment.
pub const DEFAULT_BE_MAX_STEPS: usize = 1_000_000;

/// Controls collection of BE/NRE lifecycle and callback telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BeTelemetryMode {
    /// Do not install instrumentation or collect counters.
    Off,
    /// Collect work counters without reading the clock.
    Counters,
    /// Collect counters and wall-clock stage timings.
    #[default]
    Timings,
}

impl BeTelemetryMode {
    /// Compatibility alias for the former fully disabled mode.
    #[allow(non_upper_case_globals)]
    pub const Disabled: Self = Self::Off;

    /// Compatibility alias for the former counters-and-timings mode.
    #[allow(non_upper_case_globals)]
    pub const Enabled: Self = Self::Timings;

    pub(crate) const fn collects_counters(self) -> bool {
        !matches!(self, Self::Off)
    }

    pub(crate) const fn collects_timings(self) -> bool {
        matches!(self, Self::Timings)
    }
}

/// Typed lifecycle state for Backward Euler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BeStatus {
    /// Configured and ready, or currently advancing.
    #[default]
    Running,
    /// Reached the configured final time.
    Finished,
    /// Stopped at a configured sampled stop condition.
    StoppedByCondition,
    /// Integration failed.
    Failed,
}

impl BeStatus {
    /// Stable text representation retained for compatibility APIs.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Finished => "finished",
            Self::StoppedByCondition => "stopped_by_condition",
            Self::Failed => "failed",
        }
    }
}

#[derive(Debug)]
pub enum BeError {
    InvalidConfiguration(&'static str),
    Backend(IvpBackendError),
    GeneratedBackend(SymbolicIvpGeneratedError),
    Newton(NreError),
    StepUnderflow { time: f64, step: f64 },
    StepLimit { max_steps: usize },
}

/// Telemetry classification for failures returned by BE operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeFailureKind {
    Configuration,
    Backend,
    GeneratedBackend,
    Newton,
    StepUnderflow,
    StepLimit,
}

impl BeError {
    fn failure_kind(&self) -> BeFailureKind {
        match self {
            Self::InvalidConfiguration(_) => BeFailureKind::Configuration,
            Self::Backend(_) => BeFailureKind::Backend,
            Self::GeneratedBackend(_) => BeFailureKind::GeneratedBackend,
            Self::Newton(_) => BeFailureKind::Newton,
            Self::StepUnderflow { .. } => BeFailureKind::StepUnderflow,
            Self::StepLimit { .. } => BeFailureKind::StepLimit,
        }
    }
}

/// Counters for explicit accepted-state continuation requests.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BeContinuationStatistics {
    /// Number of continuation calls, including requests rejected at validation.
    pub attempts: usize,
    /// Number of continuation calls ending in a successful terminal status.
    pub completed: usize,
    /// Number of continuation calls returning an error.
    pub failures: usize,
    /// Cumulative wall-clock time spent in continuation calls, in milliseconds.
    pub elapsed_ms_total: f64,
}

impl BeContinuationStatistics {
    fn record(&mut self, elapsed: Option<std::time::Duration>, succeeded: bool) {
        self.attempts = self.attempts.saturating_add(1);
        if succeeded {
            self.completed = self.completed.saturating_add(1);
        } else {
            self.failures = self.failures.saturating_add(1);
        }
        if let Some(elapsed) = elapsed {
            self.elapsed_ms_total += elapsed.as_secs_f64() * 1_000.0;
        }
    }
}

/// BE/NRE work that is not represented by the shared IVP callback counters.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BeDetailedStatistics {
    /// Parameter-value binding attempts, including rejected values.
    pub parameter_bind_attempts: usize,
    /// Successful parameter-value bindings.
    pub parameter_bind_successes: usize,
    /// Rejected parameter-value bindings.
    pub parameter_bind_failures: usize,
    /// Cumulative time validating and applying parameter values, in milliseconds.
    pub parameter_bind_ms_total: f64,
    /// Failed BE operations classified by their returned error type.
    pub configuration_failures: usize,
    pub backend_failures: usize,
    pub generated_backend_failures: usize,
    pub newton_failures: usize,
    pub step_underflow_failures: usize,
    pub step_limit_failures: usize,
    /// RHS evaluations performed only to construct a finite-difference Jacobian.
    pub fd_rhs_evaluations: usize,
    /// Newton matrix factorizations attempted.
    pub factorization_calls: usize,
    /// Cumulative Newton matrix factorization time in milliseconds.
    pub factorization_ms_total: f64,
    /// Linear-system solve attempts after factorization.
    pub linear_solve_calls: usize,
    /// Cumulative triangular/linear solve time in milliseconds.
    pub linear_solve_ms_total: f64,
    /// Steps whose Newton solve completed and whose state was committed.
    pub accepted_steps: usize,
    /// Step attempts that failed before committing a state.
    pub failed_steps: usize,
    /// Number of final/continuation trajectory assembly operations.
    pub output_assembly_calls: usize,
    /// Cumulative trajectory assembly time in milliseconds.
    pub output_assembly_ms_total: f64,
}

impl BeDetailedStatistics {
    pub(crate) fn record_parameter_bind(
        &mut self,
        elapsed: Option<std::time::Duration>,
        succeeded: bool,
    ) {
        self.parameter_bind_attempts = self.parameter_bind_attempts.saturating_add(1);
        if succeeded {
            self.parameter_bind_successes = self.parameter_bind_successes.saturating_add(1);
        } else {
            self.parameter_bind_failures = self.parameter_bind_failures.saturating_add(1);
        }
        if let Some(elapsed) = elapsed {
            self.parameter_bind_ms_total += elapsed.as_secs_f64() * 1_000.0;
        }
    }

    pub(crate) fn record_failure(&mut self, kind: BeFailureKind) {
        let count = match kind {
            BeFailureKind::Configuration => &mut self.configuration_failures,
            BeFailureKind::Backend => &mut self.backend_failures,
            BeFailureKind::GeneratedBackend => &mut self.generated_backend_failures,
            BeFailureKind::Newton => &mut self.newton_failures,
            BeFailureKind::StepUnderflow => &mut self.step_underflow_failures,
            BeFailureKind::StepLimit => &mut self.step_limit_failures,
        };
        *count = count.saturating_add(1);
    }

    pub(crate) fn record_fd_rhs_evaluation(&mut self) {
        self.fd_rhs_evaluations = self.fd_rhs_evaluations.saturating_add(1);
    }

    pub(crate) fn record_factorization(&mut self, elapsed: Option<std::time::Duration>) {
        self.factorization_calls = self.factorization_calls.saturating_add(1);
        if let Some(elapsed) = elapsed {
            self.factorization_ms_total += elapsed.as_secs_f64() * 1_000.0;
        }
    }

    pub(crate) fn record_linear_solve(&mut self, elapsed: Option<std::time::Duration>) {
        self.linear_solve_calls = self.linear_solve_calls.saturating_add(1);
        if let Some(elapsed) = elapsed {
            self.linear_solve_ms_total += elapsed.as_secs_f64() * 1_000.0;
        }
    }

    pub(crate) fn record_accepted_step(&mut self) {
        self.accepted_steps = self.accepted_steps.saturating_add(1);
    }

    pub(crate) fn record_failed_step(&mut self) {
        self.failed_steps = self.failed_steps.saturating_add(1);
    }

    pub(crate) fn record_output_assembly(&mut self, elapsed: Option<std::time::Duration>) {
        self.output_assembly_calls = self.output_assembly_calls.saturating_add(1);
        if let Some(elapsed) = elapsed {
            self.output_assembly_ms_total += elapsed.as_secs_f64() * 1_000.0;
        }
    }
}

impl std::fmt::Display for BeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfiguration(message) => {
                write!(f, "invalid Backward Euler configuration: {message}")
            }
            Self::Backend(error) => write!(f, "Backward Euler backend failed: {error}"),
            Self::GeneratedBackend(error) => {
                write!(f, "Backward Euler generated backend failed: {error}")
            }
            Self::Newton(error) => write!(f, "Backward Euler Newton solve failed: {error}"),
            Self::StepUnderflow { time, step } => {
                write!(f, "Backward Euler step {step} cannot advance time {time}")
            }
            Self::StepLimit { max_steps } => {
                write!(
                    f,
                    "Backward Euler exceeded the maximum of {max_steps} steps"
                )
            }
        }
    }
}

impl std::error::Error for BeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Backend(error) => Some(error),
            Self::GeneratedBackend(error) => Some(error),
            Self::Newton(error) => Some(error),
            _ => None,
        }
    }
}

impl From<IvpBackendError> for BeError {
    fn from(value: IvpBackendError) -> Self {
        Self::Backend(value)
    }
}

impl From<NreError> for BeError {
    fn from(value: NreError) -> Self {
        Self::Newton(value)
    }
}

type BeNativeRhs = Arc<dyn Fn(f64, &DVector<f64>) -> DVector<f64> + Send + Sync>;
type BeNativeJac = Arc<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64> + Send + Sync>;
pub enum Equation {
    LHS(Vec<Expr>),
    RHS(Vec<Expr>),
}

/// Grouped setup for one Backward Euler solve.
#[derive(Clone)]
pub struct BeSolverOptions {
    pub eq_system: Vec<Expr>,
    pub values: Vec<String>,
    pub arg: String,
    pub tolerance: f64,
    pub max_iterations: usize,
    /// Fixed integration step. `None` recomputes the legacy local heuristic
    /// before each step from current `(t, y)` and the remaining interval. The
    /// selected step is fixed during that step's Newton solve; this is not an
    /// error estimator and does not guarantee integration accuracy.
    pub h: Option<f64>,
    pub t0: f64,
    pub t_bound: f64,
    pub y0: DVector<f64>,
    pub generated_backend_config: SymbolicIvpGeneratedBackendConfig,
    /// Maximum integration step attempts in one solve/continuation segment.
    pub max_steps: usize,
    /// Controls solver and callback instrumentation.
    pub telemetry_mode: BeTelemetryMode,
}

impl BeSolverOptions {
    /// Creates grouped Backward Euler options.
    pub fn new(
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        tolerance: f64,
        max_iterations: usize,
        h: Option<f64>,
        t0: f64,
        t_bound: f64,
        y0: DVector<f64>,
    ) -> Self {
        Self {
            eq_system,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            t0,
            t_bound,
            y0,
            generated_backend_config: SymbolicIvpGeneratedBackendConfig::defaults(),
            max_steps: DEFAULT_BE_MAX_STEPS,
            telemetry_mode: BeTelemetryMode::Timings,
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

    /// Sets the maximum attempted steps per integration segment.
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Enables or disables lifecycle and callback telemetry.
    pub fn with_telemetry_mode(mut self, mode: BeTelemetryMode) -> Self {
        self.telemetry_mode = mode;
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
    /// for Backward Euler this is usually the first compiled backend worth
    /// trying once the nonlinear system stops being toy-sized. Current IVP
    /// comparisons show that `C + tcc` is often the best practical compromise
    /// for medium and larger stiff systems because setup stays cheap while the
    /// generated Jacobian/residual path becomes faster than `Lambdify`.
    pub fn with_dense_generated_backend_c_tcc(self, output_parent_dir: impl Into<PathBuf>) -> Self {
        self.with_generated_backend_config(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_c_tcc(),
        )
    }

    /// Uses compiled dense IVP path via `C + gcc` when runtime throughput matters more.
    ///
    /// Practical note:
    /// this is the "pay a larger setup cost for stronger native code" option.
    /// It is worth trying for long repeated Backward Euler runs when startup
    /// latency is secondary.
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
    /// For `BE` the practical recommendation is:
    /// - small systems: keep `Lambdify`;
    /// - larger stiff systems: try this preset first;
    /// - if startup latency becomes a problem, drop to explicit `C + tcc`.
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
//#[derive(Debug)]
pub struct BE {
    pub newton: NRE,
    y0: DVector<f64>,
    t0: f64,
    t_bound: f64,
    t: f64,
    y: DVector<f64>,
    t_old: Option<f64>,
    t_result: DVector<f64>,
    y_result: DMatrix<f64>,
    status: BeStatus,
    message: Option<String>,
    h: Option<f64>,
    stop_conditions: Vec<(usize, f64)>,
    neighborhood_check: f64,
    generated_backend_config: SymbolicIvpGeneratedBackendConfig,
    statistics: IvpBackendStatistics,
    native_rhs: Option<BeNativeRhs>,
    native_jac: Option<BeNativeJac>,
    last_error: Option<BeError>,
    continuation_statistics: BeContinuationStatistics,
    max_steps: usize,
    telemetry_mode: BeTelemetryMode,
    telemetry_locked: bool,
}
impl Display for BE {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BE {{ t0: {}, t_bound: {}, t: {}, y: {:?} }}",
            self.t0, self.t_bound, self.t, self.y
        )
    }
}

impl BE {
    pub fn new() -> BE {
        let nr_new = NRE::new(
            Vec::new(),
            DVector::zeros(0),
            Vec::new(),
            String::new(),
            0.0,
            0,
            0.0,
            true,
            None,
        );
        BE {
            newton: nr_new,
            y0: DVector::zeros(0),
            t0: 0.0,
            t_bound: 0.0,
            t: 0.0,
            y: DVector::zeros(0),
            t_old: None,
            t_result: DVector::zeros(0),
            y_result: DMatrix::zeros(0, 0),
            status: BeStatus::Running,
            message: None,
            h: None,
            stop_conditions: Vec::new(),
            neighborhood_check: 1e-6,
            generated_backend_config: SymbolicIvpGeneratedBackendConfig::defaults(),
            statistics: IvpBackendStatistics::default(),
            native_rhs: None,
            native_jac: None,
            last_error: None,
            continuation_statistics: BeContinuationStatistics::default(),
            max_steps: DEFAULT_BE_MAX_STEPS,
            telemetry_mode: BeTelemetryMode::Timings,
            telemetry_locked: false,
        }
    }

    /// Preferred grouped setup path for Backward Euler.
    pub fn new_with_options(options: BeSolverOptions) -> Self {
        Self::try_new_with_options(options)
            .expect("Backward Euler options should describe a valid problem")
    }

    pub fn try_new_with_options(options: BeSolverOptions) -> Result<Self, BeError> {
        let mut solver =
            Self::new().with_generated_backend_config(options.generated_backend_config);
        solver.try_set_initial(
            options.eq_system,
            options.values,
            options.arg,
            options.tolerance,
            options.max_iterations,
            options.h,
            options.t0,
            options.t_bound,
            options.y0,
        )?;
        solver.try_set_max_steps(options.max_steps)?;
        solver.try_set_telemetry_mode(options.telemetry_mode)?;
        Ok(solver)
    }

    /// Installs one high-level generated-backend orchestration config.
    pub fn set_generated_backend_config(&mut self, config: SymbolicIvpGeneratedBackendConfig) {
        self.generated_backend_config = config;
        self.newton.jac = None;
    }

    /// Returns the current generated-backend orchestration config.
    pub fn generated_backend_config(&self) -> &SymbolicIvpGeneratedBackendConfig {
        &self.generated_backend_config
    }

    pub fn get_statistics(&self) -> IvpBackendStatistics {
        if !self.telemetry_mode.collects_counters() {
            return IvpBackendStatistics::default();
        }
        let mut stats = self.statistics.clone();
        let newton_stats = self.newton.statistics();
        stats.nonlinear_solve_calls = newton_stats.nonlinear_solve_calls;
        stats.nonlinear_iterations_total = newton_stats.nonlinear_iterations_total;
        stats.residual_calls = newton_stats.residual_calls;
        stats.residual_ms_total = newton_stats.residual_ms_total;
        stats.jacobian_calls = newton_stats.jacobian_calls;
        stats.jacobian_ms_total = newton_stats.jacobian_ms_total;
        stats
    }

    pub fn statistics_report(&self) -> String {
        let mut report = self.get_statistics().table_report();
        let detailed = self.detailed_statistics();
        report.push_str(&format!(
            "\ntelemetry={}; continuation: attempts={} completed={} failures={} elapsed_ms_total={:.6}\nBE work: parameter_bind_attempts={} parameter_bind_successes={} parameter_bind_failures={} parameter_bind_ms={:.6} failures=[configuration:{},backend:{},generated_backend:{},newton:{},step_underflow:{},step_limit:{}] fd_rhs={} factorization_calls={} factorization_ms={:.6} linear_solve_calls={} linear_solve_ms={:.6} accepted_steps={} failed_steps={} output_assembly_calls={} output_assembly_ms={:.6}",
            match self.telemetry_mode {
                BeTelemetryMode::Off => "off",
                BeTelemetryMode::Counters => "counters",
                BeTelemetryMode::Timings => "timings",
            },
            self.continuation_statistics.attempts,
            self.continuation_statistics.completed,
            self.continuation_statistics.failures,
            self.continuation_statistics.elapsed_ms_total,
            detailed.parameter_bind_attempts,
            detailed.parameter_bind_successes,
            detailed.parameter_bind_failures,
            detailed.parameter_bind_ms_total,
            detailed.configuration_failures,
            detailed.backend_failures,
            detailed.generated_backend_failures,
            detailed.newton_failures,
            detailed.step_underflow_failures,
            detailed.step_limit_failures,
            detailed.fd_rhs_evaluations,
            detailed.factorization_calls,
            detailed.factorization_ms_total,
            detailed.linear_solve_calls,
            detailed.linear_solve_ms_total,
            detailed.accepted_steps,
            detailed.failed_steps,
            detailed.output_assembly_calls,
            detailed.output_assembly_ms_total,
        ));
        report
    }

    /// Returns BE/NRE-specific work counters and stage timings.
    pub fn detailed_statistics(&self) -> BeDetailedStatistics {
        self.newton.detailed_statistics()
    }

    /// Returns the configured maximum attempted steps per integration segment.
    pub fn max_steps(&self) -> usize {
        self.max_steps
    }

    /// Configures a nonzero maximum number of attempted steps per segment.
    pub fn try_set_max_steps(&mut self, max_steps: usize) -> Result<(), BeError> {
        if max_steps == 0 {
            return Err(BeError::InvalidConfiguration(
                "maximum step count must be nonzero",
            ));
        }
        self.max_steps = max_steps;
        Ok(())
    }

    /// Returns the telemetry mode selected for this solver.
    pub fn telemetry_mode(&self) -> BeTelemetryMode {
        self.telemetry_mode
    }

    /// Changes telemetry mode before integration starts.
    pub fn try_set_telemetry_mode(&mut self, mode: BeTelemetryMode) -> Result<(), BeError> {
        if (self.telemetry_locked || self.newton.jac.is_some()) && mode != self.telemetry_mode {
            return Err(BeError::InvalidConfiguration(
                "telemetry mode cannot change after integration has started; reinitialize the problem first",
            ));
        }
        self.telemetry_mode = mode;
        self.newton.set_telemetry_mode(mode);
        Ok(())
    }

    /// Returns counters for accepted-state continuation requests.
    pub fn continuation_statistics(&self) -> &BeContinuationStatistics {
        &self.continuation_statistics
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
    /// This is often the most practical compiled `BE` choice once the problem
    /// is large enough for Jacobian throughput to matter.
    pub fn set_dense_generated_backend_c_tcc(&mut self, output_parent_dir: impl Into<PathBuf>) {
        self.set_generated_backend_config(
            SymbolicIvpGeneratedBackendConfig::build_if_missing_release(output_parent_dir)
                .with_c_tcc(),
        );
    }

    /// Uses compiled dense IVP path via `C + gcc` for runtime-oriented repeated solves.
    ///
    /// Prefer this when you expect many repeated Backward Euler solves on the
    /// same symbolic problem and startup latency is acceptable.
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
    /// For `BE` this points to the runtime-oriented generated path. Benchmark
    /// against `Lambdify` on small systems before enabling it by default in
    /// application code.
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
    pub fn set_initial(
        &mut self,
        eq_system: Vec<Expr>, //
        values: Vec<String>,
        arg: String,
        tolerance: f64,        // tolerance
        max_iterations: usize, // max number of iterations
        h: Option<f64>,
        t0: f64,
        t_bound: f64,
        y0: DVector<f64>,
    ) -> () {
        self.try_set_initial(
            eq_system,
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            t0,
            t_bound,
            y0,
        )
        .expect("Backward Euler initial configuration should be valid");
    }

    pub fn try_set_initial(
        &mut self,
        eq_system: Vec<Expr>,
        values: Vec<String>,
        arg: String,
        tolerance: f64,
        max_iterations: usize,
        h: Option<f64>,
        t0: f64,
        t_bound: f64,
        y0: DVector<f64>,
    ) -> Result<(), BeError> {
        validate_be_configuration(
            Some(eq_system.len()),
            &values,
            &arg,
            tolerance,
            max_iterations,
            h,
            t0,
            t_bound,
            &y0,
        )?;
        let nr = make_be_newton(
            eq_system,
            y0.clone(),
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            t_bound,
        );
        self.reset_initial_state(nr, h, t0, t_bound, y0);
        Ok(())
    }

    /// Configures a purely numerical Backward Euler problem without symbolic
    /// equations or symbolic-backend preparation.
    pub fn try_set_native_initial<F, J>(
        &mut self,
        values: Vec<String>,
        arg: String,
        tolerance: f64,
        max_iterations: usize,
        h: Option<f64>,
        t0: f64,
        t_bound: f64,
        y0: DVector<f64>,
        rhs: F,
        jac: Option<J>,
    ) -> Result<(), BeError>
    where
        F: Fn(f64, &DVector<f64>) -> DVector<f64> + Send + Sync + 'static,
        J: Fn(f64, &DVector<f64>) -> DMatrix<f64> + Send + Sync + 'static,
    {
        validate_be_configuration(
            None,
            &values,
            &arg,
            tolerance,
            max_iterations,
            h,
            t0,
            t_bound,
            &y0,
        )?;
        let nr = make_be_newton(
            Vec::new(),
            y0.clone(),
            values,
            arg,
            tolerance,
            max_iterations,
            h,
            t_bound,
        );
        self.reset_initial_state(nr, h, t0, t_bound, y0);
        self.set_native_ode_callbacks(rhs, jac);
        Ok(())
    }

    fn reset_initial_state(
        &mut self,
        newton: NRE,
        h: Option<f64>,
        t0: f64,
        t_bound: f64,
        y0: DVector<f64>,
    ) {
        self.newton = newton;
        self.t0 = t0;
        self.t_bound = t_bound;
        self.y0 = y0.clone();
        self.t = t0;
        self.y = y0.clone();
        self.h = h;
        self.t_old = None;
        self.t_result = DVector::from_vec(vec![t0]);
        self.y_result = DMatrix::from_row_slice(1, y0.len(), y0.as_slice());
        self.status = BeStatus::Running;
        self.message = None;
        self.native_rhs = None;
        self.native_jac = None;
        self.stop_conditions.clear();
        self.neighborhood_check = 1e-6;
        self.statistics = IvpBackendStatistics::default();
        self.last_error = None;
        self.continuation_statistics = BeContinuationStatistics::default();
        self.telemetry_locked = false;
        self.newton.set_telemetry_mode(self.telemetry_mode);
    }

    pub fn set_stop_condition(&mut self, stop_condition: HashMap<String, f64>) {
        self.try_set_stop_condition(stop_condition)
            .expect("Backward Euler stop condition should be valid");
    }

    pub fn set_equation_parameters(&mut self, params: Option<&[&str]>) {
        self.try_set_equation_parameters(params)
            .expect("Backward Euler parameter schema should be valid");
    }

    /// Stops after an accepted sample enters the configured neighborhood.
    /// This is a sampled threshold check, not event localization; the solver
    /// does not interpolate a crossing between time steps.
    pub fn try_set_stop_condition(
        &mut self,
        stop_condition: HashMap<String, f64>,
    ) -> Result<(), BeError> {
        let mut compiled = Vec::with_capacity(stop_condition.len());
        for (name, target) in stop_condition {
            if !target.is_finite() {
                return Err(BeError::InvalidConfiguration(
                    "stop conditions must reference known states and finite targets",
                ));
            }
            let Some(index) = self
                .newton
                .values
                .iter()
                .position(|state_name| state_name == &name)
            else {
                return Err(BeError::InvalidConfiguration(
                    "stop conditions must reference known states and finite targets",
                ));
            };
            compiled.push((index, target));
        }
        self.stop_conditions = compiled;
        Ok(())
    }

    pub fn try_set_equation_parameters(&mut self, params: Option<&[&str]>) -> Result<(), BeError> {
        if self.newton.eq_system.is_empty() {
            return Err(BeError::InvalidConfiguration(
                "equation parameter schemas require a symbolic equation system",
            ));
        }
        self.newton.try_set_equation_parameters(params)?;
        Ok(())
    }

    pub fn set_parameter_values(&mut self, values: DVector<f64>) -> Result<(), BeError> {
        if self.newton.eq_system.is_empty() {
            let start = self.telemetry_mode.collects_timings().then(Instant::now);
            if self.telemetry_mode.collects_counters() {
                self.newton
                    .record_parameter_bind_result(start.map(|start| start.elapsed()), false);
            }
            return Err(BeError::InvalidConfiguration(
                "symbolic parameter values cannot be rebound on a numerical-callback problem",
            ));
        }
        self.newton.set_parameter_values(values)?;
        Ok(())
    }

    /// Fallible alias emphasizing that parameter validation can fail.
    pub fn try_set_parameter_values(&mut self, values: DVector<f64>) -> Result<(), BeError> {
        self.set_parameter_values(values)
    }

    pub fn set_neighborhood_check(&mut self, tolerance: f64) {
        self.try_set_neighborhood_check(tolerance)
            .expect("Backward Euler stop-condition tolerance should be valid");
    }

    pub fn try_set_neighborhood_check(&mut self, tolerance: f64) -> Result<(), BeError> {
        if tolerance.is_finite() && tolerance > 0.0 {
            self.neighborhood_check = tolerance;
            Ok(())
        } else {
            Err(BeError::InvalidConfiguration(
                "stop-condition tolerance must be positive and finite",
            ))
        }
    }

    /// Installs pure numerical ODE callbacks for Backward Euler.
    ///
    /// If `jac` is `None`, Newton iterations use a finite-difference Jacobian.
    /// Symbolic equation parameter setters do not apply to numerical callbacks;
    /// manage values captured by such callbacks through their own shared state.
    pub fn set_native_ode_callbacks<F, J>(&mut self, rhs: F, jac: Option<J>)
    where
        F: Fn(f64, &DVector<f64>) -> DVector<f64> + Send + Sync + 'static,
        J: Fn(f64, &DVector<f64>) -> DMatrix<f64> + Send + Sync + 'static,
    {
        self.native_rhs = Some(Arc::new(rhs));
        self.native_jac = jac.map(|j| Arc::new(j) as BeNativeJac);
        self.newton.jac = None;
    }

    fn check_stop_condition(&self, y: &DVector<f64>) -> bool {
        self.stop_conditions
            .iter()
            .any(|(index, target)| (y[*index] - target).abs() <= self.neighborhood_check)
    }
    /// Validates the current problem configuration without panicking.
    pub fn try_check(&self) -> Result<(), BeError> {
        validate_be_configuration(
            if self.native_rhs.is_some() {
                None
            } else {
                Some(self.newton.eq_system.len())
            },
            &self.newton.values,
            &self.newton.arg,
            self.newton.tolerance,
            self.newton.max_iterations,
            self.h,
            self.t0,
            self.t_bound,
            &self.y0,
        )?;
        Ok(())
    }

    /// Compatibility wrapper; prefer [`BE::try_check`] in fallible code.
    pub fn check(&self) {
        self.try_check()
            .expect("Backward Euler configuration should be valid");
    }

    pub fn _step_impl(&mut self) -> (bool, Option<String>) {
        match self.try_step_impl() {
            Ok(()) => (true, None),
            Err(error) => (false, Some(error.to_string())),
        }
    }

    fn try_step_impl(&mut self) -> Result<(), BeError> {
        let remaining = self.t_bound - self.t;
        if remaining <= 0.0 {
            return Ok(());
        }
        let dt = match self.h {
            Some(h) => h.min(remaining),
            None => self.newton.suggest_step_size(self.t, &self.y, remaining)?,
        };
        let t_next = self.t + dt;
        if !t_next.is_finite() || t_next <= self.t {
            return Err(BeError::StepUnderflow {
                time: self.t,
                step: dt,
            });
        }

        self.newton.dt = dt;
        self.newton.set_step_mode(NreStepMode::Fixed);
        self.newton.t_bound = None;
        self.newton.set_t(t_next);
        self.newton.set_initial_guess(self.y.clone());
        self.y = self.newton.try_solve()?;
        self.t = if dt >= remaining {
            self.t_bound
        } else {
            t_next
        };
        Ok(())
    }

    fn try_step_with_telemetry(&mut self) -> Result<(), BeError> {
        self.install_native_callbacks();
        match self.telemetry_mode {
            BeTelemetryMode::Off => self.try_step_inner::<false>(),
            BeTelemetryMode::Counters | BeTelemetryMode::Timings => self.try_step_inner::<true>(),
        }
    }

    /// Advances by one step using already-prepared callbacks.
    ///
    /// Native callbacks configured through BE are installed automatically. For
    /// symbolic equations, prepare the NRE callbacks first or use [`BE::try_solve`].
    pub fn try_step(&mut self) -> Result<(), BeError> {
        let result = self.try_step_with_telemetry();
        if self.telemetry_mode.collects_counters() {
            self.statistics.step_calls = self.statistics.step_calls.saturating_add(1);
        }
        if let Err(error) = &result {
            self.message = Some(error.to_string());
            self.status = BeStatus::Failed;
            if self.telemetry_mode.collects_counters() {
                self.newton.record_failure(error.failure_kind());
            }
        }
        result
    }

    /// Compatibility wrapper; use [`BE::try_step`] to handle failures explicitly.
    pub fn step(&mut self) {
        if let Err(error) = self.try_step() {
            self.last_error = Some(error);
        }
    }

    fn try_step_inner<const COUNTERS: bool>(&mut self) -> Result<(), BeError> {
        if self.t >= self.t_bound {
            self.t = self.t_bound;
            self.status = BeStatus::Finished;
            return Ok(());
        }
        let old_t = self.t;
        if let Err(error) = self.try_step_impl() {
            if COUNTERS {
                self.newton.record_failed_step();
            }
            return Err(error);
        }
        if COUNTERS {
            self.newton.record_accepted_step();
        }
        self.t_old = Some(old_t);
        if self.check_stop_condition(&self.y) {
            self.status = BeStatus::StoppedByCondition;
        } else if self.t >= self.t_bound {
            self.status = BeStatus::Finished;
        } else {
            self.status = BeStatus::Running;
        }
        Ok(())
    }

    fn run_main_loop(&mut self) {
        self.telemetry_locked = true;
        match self.telemetry_mode {
            BeTelemetryMode::Off => self.main_loop_impl::<false, false>(),
            BeTelemetryMode::Counters => self.main_loop_impl::<true, false>(),
            BeTelemetryMode::Timings => self.main_loop_impl::<true, true>(),
        }
    }

    /// Runs the integration loop using callbacks/backend already installed on BE.
    /// Prefer [`BE::try_solve`] when preparation and integration should be combined.
    pub fn try_main_loop(&mut self) -> Result<(), BeError> {
        self.last_error = None;
        self.install_native_callbacks();
        self.run_main_loop();
        let result = if let Some(error) = self.last_error.take() {
            Err(error)
        } else if matches!(
            self.status,
            BeStatus::Finished | BeStatus::StoppedByCondition
        ) {
            Ok(())
        } else {
            Err(BeError::InvalidConfiguration(
                "solver stopped without a successful terminal status",
            ))
        };
        if let Err(error) = &result {
            if self.telemetry_mode.collects_counters() {
                self.newton.record_failure(error.failure_kind());
            }
        }
        result
    }

    /// Compatibility wrapper; use [`BE::try_main_loop`] to handle failures explicitly.
    pub fn main_loop(&mut self) {
        if let Err(error) = self.try_main_loop() {
            self.message = Some(error.to_string());
            self.status = BeStatus::Failed;
            self.last_error = Some(error);
        }
    }

    fn main_loop_impl<const COUNTERS: bool, const TIMINGS: bool>(&mut self) {
        let start = if TIMINGS { Some(Instant::now()) } else { None };
        // Analogue of https://github.com/scipy/scipy/blob/main/scipy/integrate/_ivp/ivp.py

        let mut integr_status: Option<i8> = None;
        let mut y: Vec<DVector<f64>> = vec![self.y.clone()];
        let mut t: Vec<f64> = vec![self.t];
        let mut step_count = 0usize;
        while integr_status.is_none() {
            if step_count >= self.max_steps {
                self.status = BeStatus::Failed;
                let error = BeError::StepLimit {
                    max_steps: self.max_steps,
                };
                self.message = Some(error.to_string());
                self.last_error = Some(error);
                integr_status = Some(-1);
                break;
            }
            let result = self.try_step_inner::<COUNTERS>();
            if let Err(error) = result {
                self.message = Some(error.to_string());
                self.status = BeStatus::Failed;
                self.last_error = Some(error);
            }
            if COUNTERS {
                self.statistics.step_calls = self.statistics.step_calls.saturating_add(1);
            }
            let _status: i8 = 0;
            //   info("\n iteration: {}", i);
            //if i == 100 {panic!()}
            step_count += 1;
            if self.status == BeStatus::Finished {
                integr_status = Some(0)
            } else if self.status == BeStatus::Failed {
                integr_status = Some(-1);
                break;
            }
            // Check stop condition before storing solution
            if self.check_stop_condition(&self.y) {
                self.status = BeStatus::StoppedByCondition;
                integr_status = Some(0);
            }

            //  info("i: {}, t: {}, y: {:?}, _status: {}", i, self.Solver_instance.t, self.Solver_instance.y, _status);
            if self.t > *t.last().unwrap() {
                t.push(self.t);
                y.push(self.y.clone());
            }
            // info("time  {:?}, len {}", t, t.len())
        }

        let output_assembly_start = TIMINGS.then(Instant::now);
        let rows = &y.len();
        let cols = &y[0].len();

        let mut flat_vec: Vec<f64> = Vec::new();
        for vector in y.iter() {
            flat_vec.extend(vector)
        }
        let y_res: DMatrix<f64> = DMatrix::from_vec(*cols, *rows, flat_vec).transpose();
        let t_res = DVector::from_vec(t);

        // info("time  {:?}, len {}", &t_res, t_res.len());
        //info("y  {:?}, len {:?}", &y_res, y_res.shape());
        if COUNTERS {
            if let Some(start) = start {
                let duration = start.elapsed();
                info!("Program took {} milliseconds to run", duration.as_millis());
                self.statistics.record_solve_duration(duration);
            } else {
                self.statistics.solve_calls = self.statistics.solve_calls.saturating_add(1);
            }
        }
        self.t_result = t_res;
        self.y_result = y_res;
        if COUNTERS {
            self.newton
                .record_output_assembly(output_assembly_start.map(|start| start.elapsed()));
        }
    } //

    pub fn try_solve(&mut self) -> Result<(), BeError> {
        self.t = self.t0;
        self.y = self.y0.clone();
        self.t_old = None;
        self.t_result = DVector::from_vec(vec![self.t0]);
        self.y_result = DMatrix::from_row_slice(1, self.y0.len(), self.y0.as_slice());
        self.status = BeStatus::Running;
        self.message = None;
        self.last_error = None;
        self.newton.set_initial_guess(self.y0.clone());

        let result = self.run_current_interval();
        if let Err(error) = &result {
            if self.telemetry_mode.collects_counters() {
                self.newton.record_failure(error.failure_kind());
            }
        }
        result
    }

    /// Continues integration from the latest accepted state to `new_t_bound`.
    ///
    /// The existing trajectory and prepared callbacks are retained. The bound
    /// must be finite and strictly later than the current accepted time.
    /// Calling [`Self::try_solve`] afterwards restarts from `(t0, y0)` and uses
    /// the currently configured bound, including this extended bound.
    pub fn try_continue_to(&mut self, new_t_bound: f64) -> Result<(), BeError> {
        if self.telemetry_mode == BeTelemetryMode::Off {
            return self.try_continue_to_inner(new_t_bound);
        }
        let start = self.telemetry_mode.collects_timings().then(Instant::now);
        let result = self.try_continue_to_inner(new_t_bound);
        self.continuation_statistics
            .record(start.map(|start| start.elapsed()), result.is_ok());
        if let Err(error) = &result {
            self.newton.record_failure(error.failure_kind());
        }
        result
    }

    fn try_continue_to_inner(&mut self, new_t_bound: f64) -> Result<(), BeError> {
        if self.y.is_empty() || self.t_result.is_empty() {
            return Err(BeError::InvalidConfiguration(
                "configure an initial value problem before continuing",
            ));
        }
        if !new_t_bound.is_finite() || new_t_bound <= self.t {
            return Err(BeError::InvalidConfiguration(
                "continuation bound must be finite and greater than the current accepted time",
            ));
        }

        let prefix_times = std::mem::replace(&mut self.t_result, DVector::from_vec(vec![self.t]));
        let prefix_states = std::mem::replace(
            &mut self.y_result,
            DMatrix::from_row_slice(1, self.y.len(), self.y.as_slice()),
        );
        self.t_bound = new_t_bound;
        self.t_old = None;
        self.status = BeStatus::Running;
        self.message = None;
        self.last_error = None;
        self.newton.set_initial_guess(self.y.clone());

        let result = self.run_current_interval();
        let output_assembly_start = self.telemetry_mode.collects_timings().then(Instant::now);
        self.append_trajectory(prefix_times, prefix_states);
        if self.telemetry_mode.collects_counters() {
            self.newton
                .record_output_assembly(output_assembly_start.map(|start| start.elapsed()));
        }
        result
    }

    fn append_trajectory(&mut self, prefix_times: DVector<f64>, prefix_states: DMatrix<f64>) {
        let prefix_len = prefix_times.len();
        let segment_len = self.t_result.len();
        debug_assert_eq!(prefix_states.nrows(), prefix_len);
        debug_assert_eq!(prefix_states.ncols(), self.y_result.ncols());
        debug_assert!(segment_len > 0);

        let mut times = Vec::with_capacity(prefix_len + segment_len.saturating_sub(1));
        times.extend(prefix_times.iter().copied());
        times.extend(self.t_result.iter().skip(1).copied());

        let rows = times.len();
        let cols = self.y_result.ncols();
        let states = DMatrix::from_fn(rows, cols, |row, col| {
            if row < prefix_len {
                prefix_states[(row, col)]
            } else {
                self.y_result[(row - prefix_len + 1, col)]
            }
        });
        self.t_result = DVector::from_vec(times);
        self.y_result = states;
    }

    fn run_current_interval(&mut self) -> Result<(), BeError> {
        self.telemetry_locked = true;
        if self.t_bound == self.t {
            self.status = BeStatus::Finished;
            return Ok(());
        }
        if self.check_stop_condition(&self.y) {
            self.status = BeStatus::StoppedByCondition;
            return Ok(());
        }

        self.install_native_callbacks();
        if self.newton.jac.is_none() {
            let start = self.telemetry_mode.collects_timings().then(Instant::now);
            let mut options = crate::symbolic::symbolic_ivp::SymbolicIvpProblemOptions::new();
            if let Some(parameters) = self.newton.equation_parameters.clone() {
                options = options.with_equation_parameters(parameters);
            }
            if let Some(values) = self.newton.equation_parameter_values.clone() {
                options = options.with_equation_parameter_values(values);
            }
            let prepared = match prepare_generated_symbolic_ivp_problem(
                self.newton.eq_system.clone(),
                self.newton.values.clone(),
                self.newton.arg.clone(),
                options.with_aot_options(self.generated_backend_config.aot_options),
                self.generated_backend_config.clone(),
            ) {
                Ok(prepared) => prepared,
                Err(error) => {
                    self.status = BeStatus::Failed;
                    self.message = Some(error.to_string());
                    return Err(BeError::GeneratedBackend(error));
                }
            };
            self.generated_backend_config.resolver = prepared.updated_resolver.clone();
            self.newton
                .install_prepared_backend(prepared.into_problem());
            if self.telemetry_mode.collects_counters() {
                if let Some(start) = start {
                    self.statistics
                        .record_backend_prepare_duration(start.elapsed());
                } else {
                    self.statistics.backend_prepare_calls =
                        self.statistics.backend_prepare_calls.saturating_add(1);
                }
            }
        }
        self.run_main_loop();
        if let Some(error) = self.last_error.take() {
            return Err(error);
        }
        if self.status != BeStatus::Finished && self.status != BeStatus::StoppedByCondition {
            return Err(BeError::InvalidConfiguration(
                "solver stopped without a successful terminal status",
            ));
        }
        Ok(())
    }

    fn install_native_callbacks(&mut self) {
        if let Some(rhs) = self.native_rhs.clone() {
            let jac = self.native_jac.clone();
            self.newton.set_native_callbacks(
                Box::new(move |t: f64, y: &DVector<f64>| rhs(t, y)),
                jac.map(|jac_fun| {
                    Box::new(move |t: f64, y: &DVector<f64>| jac_fun(t, y))
                        as Box<dyn Fn(f64, &DVector<f64>) -> DMatrix<f64>>
                }),
            );
        }
    }

    pub fn solve(&mut self) -> () {
        self.try_solve()
            .expect("Backward Euler symbolic IVP backend generation should succeed");
    }
    pub fn plot_result(&self) -> () {
        plots(
            self.newton.arg.clone(),
            self.newton.values.clone(),
            self.t_result.clone(),
            self.y_result.clone(),
        );
        info!("result plotted");
    }

    /// Returns sampled times and states as `(times, states)`.
    ///
    /// `states` has one row per time sample and one column per state variable.
    /// The initial state is included, and after failure the matrices contain
    /// only the initial sample plus states from successfully accepted steps.
    pub fn get_result(&self) -> (Option<DVector<f64>>, Option<DMatrix<f64>>) {
        (Some(self.t_result.clone()), Some(self.y_result.clone()))
    }

    /// Returns the typed lifecycle state.
    pub fn status(&self) -> BeStatus {
        self.status
    }

    /// Returns the stable legacy text form of the lifecycle state.
    ///
    /// BE uses [`BeStatus`] internally; this allocating conversion is kept only
    /// for callers that still consume textual lifecycle flags.
    pub fn get_status(&self) -> String {
        self.status.as_str().to_owned()
    }
}

fn validate_be_configuration(
    equation_count: Option<usize>,
    values: &[String],
    arg: &str,
    tolerance: f64,
    max_iterations: usize,
    h: Option<f64>,
    t0: f64,
    t_bound: f64,
    y0: &DVector<f64>,
) -> Result<(), BeError> {
    if y0.is_empty() || !y0.iter().all(|value| value.is_finite()) {
        return Err(BeError::InvalidConfiguration(
            "initial state must be nonempty and finite",
        ));
    }
    if values.len() != y0.len() || equation_count.is_some_and(|count| count != y0.len()) {
        let dimension_message = if equation_count.is_some() {
            "state, variable names, and equations must have equal dimensions"
        } else {
            "state and variable names must have equal dimensions"
        };
        return Err(BeError::InvalidConfiguration(dimension_message));
    }
    if arg.trim().is_empty()
        || values.iter().any(|name| name.trim().is_empty())
        || values
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            != values.len()
    {
        return Err(BeError::InvalidConfiguration(
            "time argument and state variable names must be nonempty and unique",
        ));
    }
    if !tolerance.is_finite() || tolerance <= 0.0 || max_iterations == 0 {
        return Err(BeError::InvalidConfiguration(
            "tolerance must be positive and finite, and max_iterations must be nonzero",
        ));
    }
    if !t0.is_finite() || !t_bound.is_finite() || t_bound < t0 {
        return Err(BeError::InvalidConfiguration(
            "time bounds must be finite and backward integration is not supported",
        ));
    }
    if h.is_some_and(|step| !step.is_finite() || step <= 0.0) {
        return Err(BeError::InvalidConfiguration(
            "fixed step must be positive and finite",
        ));
    }
    Ok(())
}

fn make_be_newton(
    eq_system: Vec<Expr>,
    initial_guess: DVector<f64>,
    values: Vec<String>,
    arg: String,
    tolerance: f64,
    max_iterations: usize,
    h: Option<f64>,
    t_bound: f64,
) -> NRE {
    if let Some(dt) = h {
        NRE::new(
            eq_system,
            initial_guess,
            values,
            arg,
            tolerance,
            max_iterations,
            dt,
            true,
            None,
        )
    } else {
        info!("Backward Euler uses the legacy local step-size heuristic");
        NRE::new(
            eq_system,
            initial_guess,
            values,
            arg,
            tolerance,
            max_iterations,
            1e-4,
            false,
            Some(t_bound),
        )
    }
}

#[cfg(test)]
#[path = "tests/be_tests.rs"]
mod tests;
