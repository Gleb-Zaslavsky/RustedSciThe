//! Optional, typed telemetry for symbolic IVP preparation and callbacks.
//!
//! `Off` has no shared allocation and all record methods are no-ops.
//! `Counters` records only cheap atomic counters. `Detailed` additionally
//! records elapsed wall-clock time. The hot path uses typed arrays rather than
//! a `HashMap` so stage names cannot allocate during callback execution.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Runtime policy for independent Lambdify residual/Jacobian entries.
///
/// The policy applies only to warm callback evaluation. Symbolic preparation
/// keeps its existing implementation, and `Parallel` uses disjoint result
/// slots rather than a shared `Mutex`. `Auto` is deliberately conservative so
/// small IVP systems do not pay Rayon scheduling overhead.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IvpLambdifyExecutionPolicy {
    /// Evaluate entries on the calling thread.
    #[default]
    Sequential,
    /// Dispatch once the callback exposes at least `min_work` entries.
    Parallel { min_work: usize },
    /// Dispatch only when work and worker count meet a conservative threshold.
    Auto { min_work: usize },
}

impl IvpLambdifyExecutionPolicy {
    const AUTO_WORK_PER_TASK: usize = 8;

    pub const fn label(self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
            Self::Parallel { .. } => "parallel",
            Self::Auto { .. } => "auto",
        }
    }

    /// Decides whether `work` independent callback entries should be split.
    ///
    /// `parallel_tasks` allows callers with a structured layout to report the
    /// actual number of independent jobs instead of treating every scalar as
    /// a separate task. The decision is deterministic for a given Rayon pool.
    #[inline]
    pub(crate) fn should_parallel_with_tasks(self, work: usize, parallel_tasks: usize) -> bool {
        match self {
            Self::Sequential => false,
            Self::Parallel { min_work } => work >= min_work,
            Self::Auto { min_work } => {
                let workers = rayon::current_num_threads().max(1);
                workers > 1
                    && parallel_tasks > 1
                    && work >= min_work.max(workers.saturating_mul(Self::AUTO_WORK_PER_TASK))
            }
        }
    }
}

/// Runtime collection level for symbolic IVP work.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IvpTelemetryMode {
    /// No allocation and no runtime measurements.
    #[default]
    Off,
    /// Record counters but do not call `Instant::now`.
    Counters,
    /// Record counters and elapsed wall-clock durations.
    Detailed,
}

impl IvpTelemetryMode {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Counters => "counters",
            Self::Detailed => "detailed",
        }
    }
}

/// Symbolic/evaluator route associated with one telemetry stream.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum IvpTelemetryRoute {
    #[default]
    Unknown,
    ExprLegacy,
    AtomViewExprCompat,
    AtomViewNative,
    Aot,
    AnalyticalClosure,
    FiniteDifference,
}

impl IvpTelemetryRoute {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::ExprLegacy => "expr_legacy",
            Self::AtomViewExprCompat => "atom_view_expr_compat",
            Self::AtomViewNative => "atom_view_native",
            Self::Aot => "aot",
            Self::AnalyticalClosure => "analytical_closure",
            Self::FiniteDifference => "finite_difference",
        }
    }
}

/// Callable execution route associated with one telemetry stream.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum IvpTelemetryExecution {
    #[default]
    Lambdify,
    Aot,
}

/// Matrix storage selected by the numerical IVP runtime.
///
/// This is deliberately separate from [`IvpTelemetryRoute`]: the same
/// Lambdify callback can feed dense, sparse, or banded Newton systems.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum IvpTelemetryMatrixBackend {
    #[default]
    Unknown,
    Dense,
    Sparse,
    Banded,
}

impl IvpTelemetryMatrixBackend {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Dense => "dense",
            Self::Sparse => "sparse",
            Self::Banded => "banded",
        }
    }
}

impl IvpTelemetryExecution {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Lambdify => "lambdify",
            Self::Aot => "aot",
        }
    }
}

/// Cold preparation stages measured by [`IvpTelemetry`].
///
/// `SymbolicJacobian` is an aggregate around the backend-specific children
/// (`SymbolicDifferentiation`, `Simplification`, `ExprToAtom`, `SparsePattern`
/// and `AtomToExpr`). For AtomView, `SparsePattern` is itself an inclusive
/// aggregate around the parallel sparse builder and its differentiation child.
/// `ResidualCompilation` and `JacobianCompilation` likewise
/// contain their corresponding lambdification stage. These are inclusive
/// scopes; child timings must not be summed with their parent as independent
/// wall-clock work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum IvpColdStage {
    Validation = 0,
    ParameterBinding,
    ExprToAtom,
    SymbolicJacobian,
    SymbolicDifferentiation,
    Simplification,
    AtomToExpr,
    SparsePattern,
    LayoutPlanning,
    ResidualCompilation,
    ResidualLambdification,
    JacobianCompilation,
    JacobianLambdification,
    BackendBinding,
    AotMaterialization,
    AotBuild,
    AotLink,
}

impl IvpColdStage {
    pub const COUNT: usize = 17;

    const fn from_index(index: usize) -> Self {
        match index {
            0 => Self::Validation,
            1 => Self::ParameterBinding,
            2 => Self::ExprToAtom,
            3 => Self::SymbolicJacobian,
            4 => Self::SymbolicDifferentiation,
            5 => Self::Simplification,
            6 => Self::AtomToExpr,
            7 => Self::SparsePattern,
            8 => Self::LayoutPlanning,
            9 => Self::ResidualCompilation,
            10 => Self::ResidualLambdification,
            11 => Self::JacobianCompilation,
            12 => Self::JacobianLambdification,
            13 => Self::BackendBinding,
            14 => Self::AotMaterialization,
            15 => Self::AotBuild,
            _ => Self::AotLink,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Validation => "validation",
            Self::ParameterBinding => "parameter_binding",
            Self::ExprToAtom => "expr_to_atom",
            Self::SymbolicJacobian => "symbolic_jacobian",
            Self::SymbolicDifferentiation => "symbolic_differentiation",
            Self::Simplification => "symbolic_simplification",
            Self::AtomToExpr => "atom_to_expr",
            Self::SparsePattern => "sparse_pattern",
            Self::LayoutPlanning => "layout_planning",
            Self::ResidualCompilation => "residual_compilation",
            Self::ResidualLambdification => "residual_lambdification",
            Self::JacobianCompilation => "jacobian_compilation",
            Self::JacobianLambdification => "jacobian_lambdification",
            Self::BackendBinding => "backend_binding",
            Self::AotMaterialization => "aot_materialization",
            Self::AotBuild => "aot_build",
            Self::AotLink => "aot_link",
        }
    }
}

/// Warm callback/linear stages measured by [`IvpTelemetry`].
///
/// `Controller` is the whole integration scope. The `controller_*` entries
/// are nested scopes inside it, and `ControllerIteration` is intentionally
/// inclusive of callback and linear-solve work performed by the nonlinear
/// correction. The report exposes both levels so controller overhead can be
/// separated without pretending nested durations are additive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum IvpWarmStage {
    ArgumentBinding = 0,
    ResidualEvaluation,
    ResidualOutputAssembly,
    JacobianEvaluation,
    JacobianOutputAssembly,
    Factorization,
    RhsSolve,
    Controller,
    ControllerStepSetup,
    ControllerPredictor,
    ControllerIteration,
    ControllerOutcome,
    ControllerStopCondition,
    ControllerMethodPolicy,
    ControllerMethodSwitch,
    ResidualCallback,
    JacobianCallback,
}

impl IvpWarmStage {
    pub const COUNT: usize = 17;

    const fn from_index(index: usize) -> Self {
        match index {
            0 => Self::ArgumentBinding,
            1 => Self::ResidualEvaluation,
            2 => Self::ResidualOutputAssembly,
            3 => Self::JacobianEvaluation,
            4 => Self::JacobianOutputAssembly,
            5 => Self::Factorization,
            6 => Self::RhsSolve,
            7 => Self::Controller,
            8 => Self::ControllerStepSetup,
            9 => Self::ControllerPredictor,
            10 => Self::ControllerIteration,
            11 => Self::ControllerOutcome,
            12 => Self::ControllerStopCondition,
            13 => Self::ControllerMethodPolicy,
            14 => Self::ControllerMethodSwitch,
            15 => Self::ResidualCallback,
            _ => Self::JacobianCallback,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::ArgumentBinding => "argument_binding",
            Self::ResidualEvaluation => "residual_evaluation",
            Self::ResidualOutputAssembly => "residual_output_assembly",
            Self::JacobianEvaluation => "jacobian_evaluation",
            Self::JacobianOutputAssembly => "jacobian_output_assembly",
            Self::Factorization => "factorization",
            Self::RhsSolve => "rhs_solve",
            Self::Controller => "controller",
            Self::ControllerStepSetup => "controller_step_setup",
            Self::ControllerPredictor => "controller_predictor",
            Self::ControllerIteration => "controller_iteration_inclusive",
            Self::ControllerOutcome => "controller_outcome",
            Self::ControllerStopCondition => "controller_stop_condition",
            Self::ControllerMethodPolicy => "controller_method_policy",
            Self::ControllerMethodSwitch => "controller_method_switch",
            Self::ResidualCallback => "residual_callback_inclusive",
            Self::JacobianCallback => "jacobian_callback_inclusive",
        }
    }
}

/// One typed stage timing entry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IvpStageTiming {
    pub calls: u64,
    pub elapsed: Duration,
}

/// Immutable telemetry snapshot suitable for story reports and tests.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IvpTelemetrySnapshot {
    pub mode: IvpTelemetryMode,
    pub route: IvpTelemetryRoute,
    pub execution: IvpTelemetryExecution,
    pub lambdify_execution_policy: IvpLambdifyExecutionPolicy,
    /// Rayon worker count observed when the evaluator policy was selected.
    pub lambdify_worker_count: usize,
    pub matrix_backend: IvpTelemetryMatrixBackend,
    pub state_dimension: usize,
    pub residual_dimension: usize,
    pub parameter_count: usize,
    pub cold: [IvpStageTiming; IvpColdStage::COUNT],
    pub warm: [IvpStageTiming; IvpWarmStage::COUNT],
    pub residual_requests: u64,
    pub residual_evaluations: u64,
    pub jacobian_requests: u64,
    pub jacobian_evaluations: u64,
    pub symbolic_jacobian_builds: u64,
    pub jacobian_rebuilds: u64,
    pub factorization_requests: u64,
    pub steps_using_current_jacobian: u64,
    pub linear_solve_requests: u64,
    pub accepted_steps: u64,
    pub rejected_steps: u64,
    pub parameter_binds: u64,
    pub method_switches: u64,
    pub scalar_evaluations: u64,
    pub conversions: u64,
    pub copies: u64,
    pub copied_bytes: u64,
    pub allocated_bytes: u64,
    pub errors: u64,
    pub parallel_dispatches: u64,
    pub sequential_dispatches: u64,
}

impl IvpTelemetrySnapshot {
    pub fn cold_stage(&self, stage: IvpColdStage) -> IvpStageTiming {
        self.cold[stage as usize]
    }

    pub fn warm_stage(&self, stage: IvpWarmStage) -> IvpStageTiming {
        self.warm[stage as usize]
    }

    /// Returns a stable, human-readable report for story tests and diagnostics.
    ///
    /// Formatting is intentionally performed from an immutable snapshot, never
    /// from a callback. This keeps file/console reporting outside measured
    /// solver work and makes reports easy to diff between dated runs.
    pub fn pretty_report(&self) -> String {
        self.to_string()
    }
}

impl fmt::Display for IvpTelemetrySnapshot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(formatter, "# LSODE2 IVP telemetry")?;
        writeln!(formatter, "")?;
        writeln!(formatter, "- mode: `{}`", self.mode.label())?;
        writeln!(formatter, "- route: `{}`", self.route.label())?;
        writeln!(formatter, "- execution: `{}`", self.execution.label())?;
        writeln!(
            formatter,
            "- lambdify_execution_policy: `{}`",
            policy_report(self.lambdify_execution_policy)
        )?;
        writeln!(
            formatter,
            "- lambdify_worker_count: `{}`",
            self.lambdify_worker_count
        )?;
        writeln!(
            formatter,
            "- matrix_backend: `{}`",
            self.matrix_backend.label()
        )?;
        writeln!(formatter, "- state_dimension: `{}`", self.state_dimension)?;
        writeln!(
            formatter,
            "- residual_dimension: `{}`",
            self.residual_dimension
        )?;
        writeln!(formatter, "- parameter_count: `{}`", self.parameter_count)?;
        writeln!(formatter, "")?;
        writeln!(formatter, "## Counters")?;
        writeln!(formatter, "")?;
        writeln!(formatter, "| counter | value |")?;
        writeln!(formatter, "|---|---:|")?;
        for (name, value) in [
            ("residual_requests", self.residual_requests),
            ("residual_evaluations", self.residual_evaluations),
            ("jacobian_requests", self.jacobian_requests),
            ("jacobian_evaluations", self.jacobian_evaluations),
            ("symbolic_jacobian_builds", self.symbolic_jacobian_builds),
            ("jacobian_rebuilds", self.jacobian_rebuilds),
            ("factorization_requests", self.factorization_requests),
            (
                "steps_using_current_jacobian",
                self.steps_using_current_jacobian,
            ),
            ("linear_solve_requests", self.linear_solve_requests),
            ("accepted_steps", self.accepted_steps),
            ("rejected_steps", self.rejected_steps),
            ("parameter_binds", self.parameter_binds),
            ("method_switches", self.method_switches),
            ("scalar_evaluations", self.scalar_evaluations),
            ("conversions", self.conversions),
            ("copies", self.copies),
            ("copied_bytes", self.copied_bytes),
            ("allocated_bytes", self.allocated_bytes),
            ("errors", self.errors),
            ("parallel_dispatches", self.parallel_dispatches),
            ("sequential_dispatches", self.sequential_dispatches),
        ] {
            writeln!(formatter, "| `{name}` | {value} |")?;
        }
        write_stage_report(formatter, "Cold stages", &self.cold, cold_stage_label)?;
        write_stage_report(formatter, "Warm stages", &self.warm, warm_stage_label)
    }
}

fn cold_stage_label(index: usize) -> &'static str {
    // The snapshot array is typed at the public boundary; this conversion is
    // only used while formatting a completed report.
    IvpColdStage::from_index(index).label()
}

fn warm_stage_label(index: usize) -> &'static str {
    IvpWarmStage::from_index(index).label()
}

fn policy_report(policy: IvpLambdifyExecutionPolicy) -> String {
    match policy {
        IvpLambdifyExecutionPolicy::Sequential => "sequential".to_string(),
        IvpLambdifyExecutionPolicy::Parallel { min_work } => {
            format!("parallel(min_work={min_work})")
        }
        IvpLambdifyExecutionPolicy::Auto { min_work } => {
            format!("auto(min_work={min_work})")
        }
    }
}

fn write_stage_report<const N: usize>(
    formatter: &mut fmt::Formatter<'_>,
    title: &str,
    stages: &[IvpStageTiming; N],
    label: fn(usize) -> &'static str,
) -> fmt::Result {
    writeln!(formatter, "")?;
    writeln!(formatter, "## {title}")?;
    writeln!(formatter, "")?;
    writeln!(formatter, "| stage | calls | elapsed_ms |")?;
    writeln!(formatter, "|---|---:|---:|")?;
    for (index, timing) in stages.iter().enumerate() {
        writeln!(
            formatter,
            "| `{}` | {} | {:.6} |",
            label(index),
            timing.calls,
            timing.elapsed.as_secs_f64() * 1_000.0
        )?;
    }
    Ok(())
}

#[derive(Debug)]
struct IvpTelemetryInner {
    route: AtomicU8,
    execution: AtomicU8,
    lambdify_execution_policy: AtomicU8,
    lambdify_execution_min_work: AtomicU64,
    lambdify_worker_count: AtomicU64,
    matrix_backend: AtomicU8,
    state_dimension: AtomicU64,
    residual_dimension: AtomicU64,
    parameter_count: AtomicU64,
    cold_calls: [AtomicU64; IvpColdStage::COUNT],
    cold_nanos: [AtomicU64; IvpColdStage::COUNT],
    warm_calls: [AtomicU64; IvpWarmStage::COUNT],
    warm_nanos: [AtomicU64; IvpWarmStage::COUNT],
    residual_requests: AtomicU64,
    residual_evaluations: AtomicU64,
    jacobian_requests: AtomicU64,
    jacobian_evaluations: AtomicU64,
    symbolic_jacobian_builds: AtomicU64,
    jacobian_rebuilds: AtomicU64,
    factorization_requests: AtomicU64,
    steps_using_current_jacobian: AtomicU64,
    linear_solve_requests: AtomicU64,
    accepted_steps: AtomicU64,
    rejected_steps: AtomicU64,
    parameter_binds: AtomicU64,
    method_switches: AtomicU64,
    scalar_evaluations: AtomicU64,
    conversions: AtomicU64,
    copies: AtomicU64,
    copied_bytes: AtomicU64,
    allocated_bytes: AtomicU64,
    errors: AtomicU64,
    parallel_dispatches: AtomicU64,
    sequential_dispatches: AtomicU64,
}

impl Default for IvpTelemetryInner {
    fn default() -> Self {
        Self {
            route: AtomicU8::new(IvpTelemetryRoute::Unknown as u8),
            execution: AtomicU8::new(IvpTelemetryExecution::Lambdify as u8),
            lambdify_execution_policy: AtomicU8::new(lambdify_policy_tag(
                IvpLambdifyExecutionPolicy::Sequential,
            )),
            lambdify_execution_min_work: AtomicU64::new(0),
            lambdify_worker_count: AtomicU64::new(0),
            matrix_backend: AtomicU8::new(IvpTelemetryMatrixBackend::Unknown as u8),
            state_dimension: AtomicU64::new(0),
            residual_dimension: AtomicU64::new(0),
            parameter_count: AtomicU64::new(0),
            cold_calls: std::array::from_fn(|_| AtomicU64::new(0)),
            cold_nanos: std::array::from_fn(|_| AtomicU64::new(0)),
            warm_calls: std::array::from_fn(|_| AtomicU64::new(0)),
            warm_nanos: std::array::from_fn(|_| AtomicU64::new(0)),
            residual_requests: AtomicU64::new(0),
            residual_evaluations: AtomicU64::new(0),
            jacobian_requests: AtomicU64::new(0),
            jacobian_evaluations: AtomicU64::new(0),
            symbolic_jacobian_builds: AtomicU64::new(0),
            jacobian_rebuilds: AtomicU64::new(0),
            factorization_requests: AtomicU64::new(0),
            steps_using_current_jacobian: AtomicU64::new(0),
            linear_solve_requests: AtomicU64::new(0),
            accepted_steps: AtomicU64::new(0),
            rejected_steps: AtomicU64::new(0),
            parameter_binds: AtomicU64::new(0),
            method_switches: AtomicU64::new(0),
            scalar_evaluations: AtomicU64::new(0),
            conversions: AtomicU64::new(0),
            copies: AtomicU64::new(0),
            copied_bytes: AtomicU64::new(0),
            allocated_bytes: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            parallel_dispatches: AtomicU64::new(0),
            sequential_dispatches: AtomicU64::new(0),
        }
    }
}

/// Cheap cloneable handle for one symbolic IVP telemetry stream.
#[derive(Clone, Debug)]
pub struct IvpTelemetry {
    mode: IvpTelemetryMode,
    inner: Option<Arc<IvpTelemetryInner>>,
}

/// RAII guard for a detailed warm-stage scope.
///
/// The guard is deliberately tiny and becomes a no-op when telemetry is off
/// or counters-only. It also closes the stage on early `Result` returns, so a
/// partial diagnostic still contains elapsed time up to the failure point.
#[must_use = "a telemetry scope must stay alive until the measured operation ends"]
pub struct IvpTelemetryScope {
    telemetry: IvpTelemetry,
    stage: IvpWarmStage,
    started: Option<Instant>,
}

impl Drop for IvpTelemetryScope {
    fn drop(&mut self) {
        self.telemetry
            .record_warm_stage(self.stage, self.started.take());
    }
}

impl Default for IvpTelemetry {
    fn default() -> Self {
        Self::disabled()
    }
}

impl IvpTelemetry {
    pub fn disabled() -> Self {
        Self {
            mode: IvpTelemetryMode::Off,
            inner: None,
        }
    }

    pub fn counters() -> Self {
        Self::with_mode(IvpTelemetryMode::Counters)
    }

    pub fn detailed() -> Self {
        Self::with_mode(IvpTelemetryMode::Detailed)
    }

    pub fn with_mode(mode: IvpTelemetryMode) -> Self {
        Self {
            mode,
            inner: (mode != IvpTelemetryMode::Off).then(|| Arc::new(IvpTelemetryInner::default())),
        }
    }

    pub fn mode(&self) -> IvpTelemetryMode {
        self.mode
    }

    pub fn set_route(&self, route: IvpTelemetryRoute) {
        if let Some(inner) = &self.inner {
            inner.route.store(route as u8, Ordering::Relaxed);
        }
    }

    pub fn set_execution(&self, execution: IvpTelemetryExecution) {
        if let Some(inner) = &self.inner {
            inner.execution.store(execution as u8, Ordering::Relaxed);
        }
    }

    pub fn set_lambdify_execution_policy(&self, policy: IvpLambdifyExecutionPolicy) {
        if let Some(inner) = &self.inner {
            inner
                .lambdify_execution_policy
                .store(lambdify_policy_tag(policy), Ordering::Relaxed);
            inner
                .lambdify_execution_min_work
                .store(lambdify_policy_min_work(policy) as u64, Ordering::Relaxed);
            inner
                .lambdify_worker_count
                .store(rayon::current_num_threads() as u64, Ordering::Relaxed);
        }
    }

    pub fn set_matrix_backend(&self, backend: IvpTelemetryMatrixBackend) {
        if let Some(inner) = &self.inner {
            inner.matrix_backend.store(backend as u8, Ordering::Relaxed);
        }
    }

    /// Records static problem shape once during preparation/configuration.
    /// These values are metadata and never participate in callback decisions.
    pub fn set_problem_shape(
        &self,
        state_dimension: usize,
        residual_dimension: usize,
        parameter_count: usize,
    ) {
        if let Some(inner) = &self.inner {
            inner
                .state_dimension
                .store(state_dimension as u64, Ordering::Relaxed);
            inner
                .residual_dimension
                .store(residual_dimension as u64, Ordering::Relaxed);
            inner
                .parameter_count
                .store(parameter_count as u64, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn start_cold_stage(&self, _stage: IvpColdStage) -> Option<Instant> {
        (self.mode == IvpTelemetryMode::Detailed).then(Instant::now)
    }

    #[inline]
    pub fn start_warm_stage(&self, _stage: IvpWarmStage) -> Option<Instant> {
        (self.mode == IvpTelemetryMode::Detailed).then(Instant::now)
    }

    pub fn scoped_warm_stage(&self, stage: IvpWarmStage) -> IvpTelemetryScope {
        IvpTelemetryScope {
            telemetry: self.clone(),
            stage,
            started: self.start_warm_stage(stage),
        }
    }

    pub fn record_cold_stage(&self, stage: IvpColdStage, started: Option<Instant>) {
        self.record_stage(stage as usize, started, true);
    }

    pub fn record_cold_stage_duration(&self, stage: IvpColdStage, elapsed: Duration) {
        self.record_stage_duration(stage as usize, elapsed, true);
    }

    pub fn record_warm_stage(&self, stage: IvpWarmStage, started: Option<Instant>) {
        self.record_stage(stage as usize, started, false);
    }

    pub fn record_warm_stage_duration(&self, stage: IvpWarmStage, elapsed: Duration) {
        self.record_stage_duration(stage as usize, elapsed, false);
    }

    pub fn record_residual_request(&self) {
        self.record_counter(|inner| &inner.residual_requests);
    }

    pub fn record_residual_evaluation(&self, started: Option<Instant>) {
        self.record_counter(|inner| &inner.residual_evaluations);
        self.record_warm_stage(IvpWarmStage::ResidualEvaluation, started);
    }

    pub fn record_residual_evaluation_count(&self) {
        self.record_counter(|inner| &inner.residual_evaluations);
    }

    pub fn record_jacobian_request(&self) {
        self.record_counter(|inner| &inner.jacobian_requests);
    }

    pub fn record_jacobian_evaluation(&self, started: Option<Instant>) {
        self.record_counter(|inner| &inner.jacobian_evaluations);
        self.record_warm_stage(IvpWarmStage::JacobianEvaluation, started);
    }

    pub fn record_jacobian_evaluation_count(&self) {
        self.record_counter(|inner| &inner.jacobian_evaluations);
    }

    pub fn record_symbolic_jacobian_build(&self) {
        self.record_counter(|inner| &inner.symbolic_jacobian_builds);
    }

    pub fn record_jacobian_rebuild(&self) {
        self.record_counter(|inner| &inner.jacobian_rebuilds);
    }

    pub fn record_factorization_request(&self) {
        self.record_counter(|inner| &inner.factorization_requests);
    }

    pub fn record_step_using_current_jacobian(&self) {
        self.record_counter(|inner| &inner.steps_using_current_jacobian);
    }

    pub fn record_linear_solve_request(&self) {
        self.record_counter(|inner| &inner.linear_solve_requests);
    }

    pub fn record_accepted_step(&self) {
        self.record_counter(|inner| &inner.accepted_steps);
    }

    pub fn record_rejected_step(&self) {
        self.record_counter(|inner| &inner.rejected_steps);
    }

    pub fn record_parameter_bind(&self) {
        self.record_counter(|inner| &inner.parameter_binds);
    }

    pub fn record_method_switch(&self) {
        self.record_counter(|inner| &inner.method_switches);
    }

    pub fn record_scalar_evaluations(&self, count: usize) {
        self.add_counter(|inner| &inner.scalar_evaluations, count as u64);
    }

    pub fn record_conversion(&self) {
        self.record_counter(|inner| &inner.conversions);
    }

    pub fn record_copy_bytes(&self, bytes: usize) {
        self.record_counter(|inner| &inner.copies);
        self.add_counter(|inner| &inner.copied_bytes, bytes as u64);
    }

    pub fn record_allocation(&self, bytes: usize) {
        self.add_counter(|inner| &inner.allocated_bytes, bytes as u64);
    }

    pub fn record_error(&self) {
        self.record_counter(|inner| &inner.errors);
    }

    /// Records the dispatch decision without timing the scheduler itself.
    pub fn record_lambdify_dispatch(&self, parallel: bool) {
        if parallel {
            self.record_counter(|inner| &inner.parallel_dispatches);
        } else {
            self.record_counter(|inner| &inner.sequential_dispatches);
        }
    }

    pub fn snapshot(&self) -> IvpTelemetrySnapshot {
        let Some(inner) = &self.inner else {
            return IvpTelemetrySnapshot {
                mode: self.mode,
                route: IvpTelemetryRoute::Unknown,
                execution: IvpTelemetryExecution::Lambdify,
                lambdify_execution_policy: IvpLambdifyExecutionPolicy::Sequential,
                lambdify_worker_count: 0,
                matrix_backend: IvpTelemetryMatrixBackend::Unknown,
                state_dimension: 0,
                residual_dimension: 0,
                parameter_count: 0,
                cold: [IvpStageTiming::default(); IvpColdStage::COUNT],
                warm: [IvpStageTiming::default(); IvpWarmStage::COUNT],
                residual_requests: 0,
                residual_evaluations: 0,
                jacobian_requests: 0,
                jacobian_evaluations: 0,
                symbolic_jacobian_builds: 0,
                jacobian_rebuilds: 0,
                factorization_requests: 0,
                steps_using_current_jacobian: 0,
                linear_solve_requests: 0,
                accepted_steps: 0,
                rejected_steps: 0,
                parameter_binds: 0,
                method_switches: 0,
                scalar_evaluations: 0,
                conversions: 0,
                copies: 0,
                copied_bytes: 0,
                allocated_bytes: 0,
                errors: 0,
                parallel_dispatches: 0,
                sequential_dispatches: 0,
            };
        };

        let cold = std::array::from_fn(|index| IvpStageTiming {
            calls: inner.cold_calls[index].load(Ordering::Relaxed),
            elapsed: Duration::from_nanos(inner.cold_nanos[index].load(Ordering::Relaxed)),
        });
        let warm = std::array::from_fn(|index| IvpStageTiming {
            calls: inner.warm_calls[index].load(Ordering::Relaxed),
            elapsed: Duration::from_nanos(inner.warm_nanos[index].load(Ordering::Relaxed)),
        });
        IvpTelemetrySnapshot {
            mode: self.mode,
            route: decode_route(inner.route.load(Ordering::Relaxed)),
            execution: decode_execution(inner.execution.load(Ordering::Relaxed)),
            lambdify_execution_policy: decode_lambdify_execution_policy(
                inner.lambdify_execution_policy.load(Ordering::Relaxed),
                inner.lambdify_execution_min_work.load(Ordering::Relaxed) as usize,
            ),
            lambdify_worker_count: inner.lambdify_worker_count.load(Ordering::Relaxed) as usize,
            matrix_backend: decode_matrix_backend(inner.matrix_backend.load(Ordering::Relaxed)),
            state_dimension: inner.state_dimension.load(Ordering::Relaxed) as usize,
            residual_dimension: inner.residual_dimension.load(Ordering::Relaxed) as usize,
            parameter_count: inner.parameter_count.load(Ordering::Relaxed) as usize,
            cold,
            warm,
            residual_requests: inner.residual_requests.load(Ordering::Relaxed),
            residual_evaluations: inner.residual_evaluations.load(Ordering::Relaxed),
            jacobian_requests: inner.jacobian_requests.load(Ordering::Relaxed),
            jacobian_evaluations: inner.jacobian_evaluations.load(Ordering::Relaxed),
            symbolic_jacobian_builds: inner.symbolic_jacobian_builds.load(Ordering::Relaxed),
            jacobian_rebuilds: inner.jacobian_rebuilds.load(Ordering::Relaxed),
            factorization_requests: inner.factorization_requests.load(Ordering::Relaxed),
            steps_using_current_jacobian: inner
                .steps_using_current_jacobian
                .load(Ordering::Relaxed),
            linear_solve_requests: inner.linear_solve_requests.load(Ordering::Relaxed),
            accepted_steps: inner.accepted_steps.load(Ordering::Relaxed),
            rejected_steps: inner.rejected_steps.load(Ordering::Relaxed),
            parameter_binds: inner.parameter_binds.load(Ordering::Relaxed),
            method_switches: inner.method_switches.load(Ordering::Relaxed),
            scalar_evaluations: inner.scalar_evaluations.load(Ordering::Relaxed),
            conversions: inner.conversions.load(Ordering::Relaxed),
            copies: inner.copies.load(Ordering::Relaxed),
            copied_bytes: inner.copied_bytes.load(Ordering::Relaxed),
            allocated_bytes: inner.allocated_bytes.load(Ordering::Relaxed),
            errors: inner.errors.load(Ordering::Relaxed),
            parallel_dispatches: inner.parallel_dispatches.load(Ordering::Relaxed),
            sequential_dispatches: inner.sequential_dispatches.load(Ordering::Relaxed),
        }
    }

    fn record_stage(&self, index: usize, started: Option<Instant>, cold: bool) {
        let Some(inner) = &self.inner else {
            return;
        };
        let calls = if cold {
            &inner.cold_calls[index]
        } else {
            &inner.warm_calls[index]
        };
        calls.fetch_add(1, Ordering::Relaxed);
        if let Some(started) = started {
            let nanos = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            let elapsed = if cold {
                &inner.cold_nanos[index]
            } else {
                &inner.warm_nanos[index]
            };
            elapsed.fetch_add(nanos, Ordering::Relaxed);
        }
    }

    fn record_stage_duration(&self, index: usize, elapsed: Duration, cold: bool) {
        let Some(inner) = &self.inner else {
            return;
        };
        let calls = if cold {
            &inner.cold_calls[index]
        } else {
            &inner.warm_calls[index]
        };
        calls.fetch_add(1, Ordering::Relaxed);
        if self.mode == IvpTelemetryMode::Detailed {
            let nanos = elapsed.as_nanos().min(u64::MAX as u128) as u64;
            let target = if cold {
                &inner.cold_nanos[index]
            } else {
                &inner.warm_nanos[index]
            };
            target.fetch_add(nanos, Ordering::Relaxed);
        }
    }

    #[inline]
    fn record_counter(&self, select: impl FnOnce(&IvpTelemetryInner) -> &AtomicU64) {
        self.add_counter(select, 1);
    }

    #[inline]
    fn add_counter(&self, select: impl FnOnce(&IvpTelemetryInner) -> &AtomicU64, value: u64) {
        if let Some(inner) = &self.inner {
            select(inner).fetch_add(value, Ordering::Relaxed);
        }
    }
}

fn decode_route(value: u8) -> IvpTelemetryRoute {
    match value {
        1 => IvpTelemetryRoute::ExprLegacy,
        2 => IvpTelemetryRoute::AtomViewExprCompat,
        3 => IvpTelemetryRoute::AtomViewNative,
        4 => IvpTelemetryRoute::Aot,
        5 => IvpTelemetryRoute::AnalyticalClosure,
        6 => IvpTelemetryRoute::FiniteDifference,
        _ => IvpTelemetryRoute::Unknown,
    }
}

fn decode_execution(value: u8) -> IvpTelemetryExecution {
    match value {
        1 => IvpTelemetryExecution::Aot,
        _ => IvpTelemetryExecution::Lambdify,
    }
}

const fn lambdify_policy_tag(policy: IvpLambdifyExecutionPolicy) -> u8 {
    match policy {
        IvpLambdifyExecutionPolicy::Sequential => 0,
        IvpLambdifyExecutionPolicy::Parallel { .. } => 1,
        IvpLambdifyExecutionPolicy::Auto { .. } => 2,
    }
}

const fn lambdify_policy_min_work(policy: IvpLambdifyExecutionPolicy) -> usize {
    match policy {
        IvpLambdifyExecutionPolicy::Sequential => 0,
        IvpLambdifyExecutionPolicy::Parallel { min_work }
        | IvpLambdifyExecutionPolicy::Auto { min_work } => min_work,
    }
}

fn decode_lambdify_execution_policy(value: u8, min_work: usize) -> IvpLambdifyExecutionPolicy {
    match value {
        1 => IvpLambdifyExecutionPolicy::Parallel { min_work },
        2 => IvpLambdifyExecutionPolicy::Auto { min_work },
        _ => IvpLambdifyExecutionPolicy::Sequential,
    }
}

fn decode_matrix_backend(value: u8) -> IvpTelemetryMatrixBackend {
    match value {
        1 => IvpTelemetryMatrixBackend::Dense,
        2 => IvpTelemetryMatrixBackend::Sparse,
        3 => IvpTelemetryMatrixBackend::Banded,
        _ => IvpTelemetryMatrixBackend::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_telemetry_has_no_measurements() {
        let telemetry = IvpTelemetry::disabled();
        telemetry.record_residual_request();
        telemetry
            .record_cold_stage_duration(IvpColdStage::SymbolicJacobian, Duration::from_secs(1));
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.mode, IvpTelemetryMode::Off);
        assert_eq!(snapshot.residual_requests, 0);
        assert_eq!(
            snapshot.cold_stage(IvpColdStage::SymbolicJacobian),
            IvpStageTiming::default()
        );
    }

    #[test]
    fn counters_record_work_without_timing() {
        let telemetry = IvpTelemetry::counters();
        telemetry.record_residual_request();
        telemetry.record_residual_evaluation(None);
        telemetry.record_jacobian_request();
        telemetry.record_symbolic_jacobian_build();
        telemetry.record_scalar_evaluations(12);
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.residual_requests, 1);
        assert_eq!(snapshot.residual_evaluations, 1);
        assert_eq!(snapshot.jacobian_requests, 1);
        assert_eq!(snapshot.symbolic_jacobian_builds, 1);
        assert_eq!(snapshot.jacobian_rebuilds, 0);
        assert_eq!(snapshot.scalar_evaluations, 12);
        assert_eq!(
            snapshot
                .warm_stage(IvpWarmStage::ResidualEvaluation)
                .elapsed,
            Duration::ZERO
        );
    }

    #[test]
    fn detailed_telemetry_keeps_typed_stage_breakdown_and_route() {
        let telemetry = IvpTelemetry::detailed();
        telemetry.set_route(IvpTelemetryRoute::AtomViewExprCompat);
        telemetry
            .record_cold_stage_duration(IvpColdStage::SymbolicJacobian, Duration::from_millis(7));
        telemetry
            .record_warm_stage_duration(IvpWarmStage::ArgumentBinding, Duration::from_micros(11));
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.route, IvpTelemetryRoute::AtomViewExprCompat);
        assert_eq!(snapshot.cold_stage(IvpColdStage::SymbolicJacobian).calls, 1);
        assert_eq!(
            snapshot.cold_stage(IvpColdStage::SymbolicJacobian).elapsed,
            Duration::from_millis(7)
        );
        assert_eq!(
            snapshot.warm_stage(IvpWarmStage::ArgumentBinding).elapsed,
            Duration::from_micros(11)
        );
    }

    #[test]
    fn scoped_warm_stage_closes_on_drop() {
        let telemetry = IvpTelemetry::counters();
        {
            let _scope = telemetry.scoped_warm_stage(IvpWarmStage::Controller);
        }
        assert_eq!(
            telemetry
                .snapshot()
                .warm_stage(IvpWarmStage::Controller)
                .calls,
            1
        );
    }

    #[test]
    fn pretty_report_contains_typed_route_counters_and_stage_tables() {
        let telemetry = IvpTelemetry::detailed();
        telemetry.set_route(IvpTelemetryRoute::AtomViewExprCompat);
        telemetry.set_execution(IvpTelemetryExecution::Lambdify);
        telemetry.set_lambdify_execution_policy(IvpLambdifyExecutionPolicy::Auto { min_work: 32 });
        telemetry.set_matrix_backend(IvpTelemetryMatrixBackend::Banded);
        telemetry.set_problem_shape(4, 4, 2);
        telemetry.record_residual_request();
        telemetry.record_lambdify_dispatch(false);
        telemetry.record_lambdify_dispatch(true);
        telemetry
            .record_cold_stage_duration(IvpColdStage::SymbolicJacobian, Duration::from_micros(12));
        telemetry.record_cold_stage_duration(
            IvpColdStage::SymbolicDifferentiation,
            Duration::from_micros(3),
        );
        telemetry.record_cold_stage_duration(
            IvpColdStage::ResidualLambdification,
            Duration::from_micros(2),
        );
        telemetry
            .record_warm_stage_duration(IvpWarmStage::ResidualEvaluation, Duration::from_micros(4));
        telemetry.record_warm_stage_duration(
            IvpWarmStage::ControllerPredictor,
            Duration::from_micros(1),
        );

        let snapshot = telemetry.snapshot();
        assert!(snapshot.lambdify_worker_count >= 1);
        let report = snapshot.pretty_report();
        assert!(report.contains("route: `atom_view_expr_compat`"));
        assert!(report.contains("lambdify_execution_policy: `auto(min_work=32)`"));
        assert!(report.contains("lambdify_worker_count: `"));
        assert!(report.contains("matrix_backend: `banded`"));
        assert!(report.contains("state_dimension: `4`"));
        assert!(report.contains("| `residual_requests` | 1 |"));
        assert!(report.contains("| `symbolic_jacobian` | 1 | 0.012000 |"));
        assert!(report.contains("| `symbolic_differentiation` | 1 | 0.003000 |"));
        assert!(report.contains("| `residual_lambdification` | 1 | 0.002000 |"));
        assert!(report.contains("| `residual_evaluation` | 1 | 0.004000 |"));
        assert!(report.contains("| `controller_predictor` | 1 | 0.001000 |"));
        assert!(report.contains("| `parallel_dispatches` | 1 |"));
        assert!(report.contains("| `sequential_dispatches` | 1 |"));
    }
}
