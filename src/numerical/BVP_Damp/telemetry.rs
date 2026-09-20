//! Typed runtime telemetry for the BVP_Damp solvers.
//!
//! The solver hot path must not use stringly-typed maps as its primary state.
//! The legacy map representation is produced only at the public compatibility
//! boundary, where existing story tests and downstream callers still consume it.

use crate::numerical::BVP_Damp::resolved_plan::BvpResolvedPlan;
use crate::symbolic::bvp::telemetry::BvpAtomDiscretizationTelemetrySnapshot;
use crate::symbolic::bvp::telemetry::BvpGenerationTelemetrySnapshot;
use crate::symbolic::bvp::telemetry::BvpLambdifyTelemetrySnapshot;
use std::cell::Cell;
use std::cell::RefCell;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Logical ownership scope of a BVP telemetry event.
///
/// These scopes are deliberately an enum rather than labels in a map.  This
/// keeps the production schema stable while allowing presentation code to
/// format names differently for tables, logs, or structured output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BvpTelemetryScope {
    Solve,
    MeshRevision,
    Iteration,
    DampingTrial,
}

/// Semantic stage of a BVP operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BvpTelemetryStage {
    ScalarEvaluation,
    ResidualRequest,
    JacobianRequest,
    ResidualChunk,
    JacobianChunk,
    Conversion,
    Copy,
    Factorization,
    RhsSolve,
}

/// Runtime logging policy for solver decision events.
///
/// This is separate from the process-wide `log` level. `Off` keeps the
/// solver-local event recorder out of the hot path; `Warnings` retains only
/// recoverable/fatal events; `Detailed` also records normal adaptive choices.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BvpLoggingMode {
    #[default]
    Off,
    Warnings,
    Detailed,
}

/// Collection policy for solver counters and stage timings.
///
/// `Off` is a real fast path: no solver counters are incremented and the
/// stage timer is not started. `Counters` preserves the historical default
/// and collects cheap typed counters and major timings. `Detailed` is for
/// diagnostic runs that also need callback-stage timings. Decision logging is
/// controlled independently by [`BvpLoggingConfig`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BvpTelemetryMode {
    Off,
    #[default]
    Counters,
    Detailed,
}

/// Bounded configuration for the typed decision trace.
///
/// A finite buffer is intentional: detailed logging must remain useful on a
/// large adaptive solve and must not turn a difficult numerical case into an
/// unbounded allocation. `max_events = 0` disables retained events while
/// keeping the mode available for an external logger.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BvpLoggingConfig {
    pub mode: BvpLoggingMode,
    pub max_events: usize,
}

impl Default for BvpLoggingConfig {
    fn default() -> Self {
        Self {
            mode: BvpLoggingMode::Off,
            max_events: 256,
        }
    }
}

impl BvpLoggingConfig {
    pub const fn new(mode: BvpLoggingMode) -> Self {
        Self {
            mode,
            max_events: 256,
        }
    }

    pub const fn with_max_events(mut self, max_events: usize) -> Self {
        self.max_events = max_events;
        self
    }
}

/// Severity of a typed solver decision event.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum BvpLogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

/// Stable event vocabulary for numerical decisions and diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BvpLogEventKind {
    BackendSelected,
    BackendFallback,
    MeshRevision,
    JacobianRefresh,
    Factorization,
    FactorizationCacheHit,
    FactorizationInvalidated,
    DampingTrial,
    DampingRejected,
    BoundLimitedStep,
    NonFiniteValue,
    NearSingularSystem,
    Termination,
}

/// Allocation-free payload for one decision event.
///
/// Numeric payload slots intentionally avoid a string map in the hot path.
/// Their interpretation is defined by `kind`; presentation layers can turn
/// them into structured JSON, tables, or human-readable log lines.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BvpLogEvent {
    pub solve_id: u64,
    pub sequence: u64,
    pub level: BvpLogLevel,
    pub kind: BvpLogEventKind,
    pub scope: BvpTelemetryScope,
    pub stage: Option<BvpTelemetryStage>,
    pub iteration: u64,
    pub damping_trial: u64,
    pub mesh_revision: u64,
    pub value_a: f64,
    pub value_b: f64,
}

/// Fixed vocabulary for callback sub-stages.
///
/// Callback instrumentation uses this enum internally so a callback does not
/// allocate or depend on a presentation string. The old string map remains a
/// compatibility adapter at the reporting boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BvpCallbackStage {
    ResidualInputPrep,
    ResidualArgs,
    ResidualValues,
    ResidualBoxing,
    JacobianInputPrep,
    JacobianArgs,
    JacobianValues,
    JacobianAssembly,
    Other,
}

impl BvpCallbackStage {
    pub const ALL: [Self; 9] = [
        Self::ResidualInputPrep,
        Self::ResidualArgs,
        Self::ResidualValues,
        Self::ResidualBoxing,
        Self::JacobianInputPrep,
        Self::JacobianArgs,
        Self::JacobianValues,
        Self::JacobianAssembly,
        Self::Other,
    ];

    pub fn from_label(label: &str) -> Self {
        match label {
            "Callback Residual Input Prep" => Self::ResidualInputPrep,
            "Callback Residual Args" => Self::ResidualArgs,
            "Callback Residual Values" => Self::ResidualValues,
            "Callback Residual Boxing" => Self::ResidualBoxing,
            "Callback Jacobian Input Prep" => Self::JacobianInputPrep,
            "Callback Jacobian Args" => Self::JacobianArgs,
            "Callback Jacobian Values" => Self::JacobianValues,
            "Callback Jacobian Assembly" => Self::JacobianAssembly,
            _ => Self::Other,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::ResidualInputPrep => "Callback Residual Input Prep",
            Self::ResidualArgs => "Callback Residual Args",
            Self::ResidualValues => "Callback Residual Values",
            Self::ResidualBoxing => "Callback Residual Boxing",
            Self::JacobianInputPrep => "Callback Jacobian Input Prep",
            Self::JacobianArgs => "Callback Jacobian Args",
            Self::JacobianValues => "Callback Jacobian Values",
            Self::JacobianAssembly => "Callback Jacobian Assembly",
            Self::Other => "Callback Other",
        }
    }

    pub(crate) const fn index(self) -> usize {
        match self {
            Self::ResidualInputPrep => 0,
            Self::ResidualArgs => 1,
            Self::ResidualValues => 2,
            Self::ResidualBoxing => 3,
            Self::JacobianInputPrep => 4,
            Self::JacobianArgs => 5,
            Self::JacobianValues => 6,
            Self::JacobianAssembly => 7,
            Self::Other => 8,
        }
    }
}

/// One typed callback-stage duration in a timing snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BvpCallbackStageTiming {
    pub stage: BvpCallbackStage,
    pub elapsed: Duration,
}

/// Storage accounting for one runtime snapshot.
///
/// Values are optional because sparse and opaque third-party allocations cannot
/// always be measured without forcing a conversion.  Estimated values must be
/// explicitly marked as such; they are never presented as process RSS.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BvpStorageBytes {
    pub jacobian_values: Option<u64>,
    pub jacobian_indices: Option<u64>,
    pub factor: Option<u64>,
    pub scratch: Option<u64>,
    pub estimated: bool,
}

impl BvpStorageBytes {
    /// Dense f64 storage estimate.  This is an estimate, not an allocator
    /// measurement and therefore remains clearly marked in the snapshot.
    pub fn dense_f64(rows: usize, cols: usize) -> Self {
        Self {
            jacobian_values: rows
                .checked_mul(cols)
                .and_then(|items| items.checked_mul(std::mem::size_of::<f64>()))
                .map(|bytes| bytes as u64),
            estimated: true,
            ..Self::default()
        }
    }

    /// Compact scalar-banded storage estimate including the pivot workspace.
    pub fn banded_f64(rows: usize, lower: usize, upper: usize) -> Self {
        let diagonals = lower.saturating_add(upper).saturating_add(1);
        let values = rows
            .saturating_mul(diagonals)
            .saturating_mul(std::mem::size_of::<f64>());
        Self {
            jacobian_values: Some(values as u64),
            factor: Some(values as u64),
            estimated: true,
            ..Self::default()
        }
    }

    /// Selects the conservative storage estimate for a solver matrix label.
    /// Sparse values/indices stay unavailable here until the native sparse
    /// owner exposes its exact buffers; a dense-equivalent size is never
    /// silently reported as sparse storage.
    pub fn for_solver_method(
        method: &str,
        rows: usize,
        cols: usize,
        bandwidth: (usize, usize),
    ) -> Self {
        match method.to_ascii_lowercase().as_str() {
            "dense" => Self::dense_f64(rows, cols),
            "banded" => Self::banded_f64(rows, bandwidth.0, bandwidth.1),
            _ => Self::default(),
        }
    }
}

/// Counts of semantically distinct solver operations.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BvpTelemetryCounters {
    pub iterations: u64,
    /// Number of vector/scalar callback requests, distinct from solver-level
    /// residual and Jacobian requests.
    pub scalar_evaluations: u64,
    pub residual_calls: u64,
    pub residual_requests: u64,
    pub jacobian_requests: u64,
    pub jacobian_recalculations: u64,
    pub residual_chunks: u64,
    pub jacobian_chunks: u64,
    pub conversions: u64,
    pub copies: u64,
    pub damping_trials: u64,
    pub damping_rejections: u64,
    pub linear_solves: u64,
    pub factorizations: u64,
    pub factorization_cache_hits: u64,
    pub factorization_invalidations: u64,
    pub rhs_solves: u64,
    pub grid_refinements: u64,
}

impl BvpTelemetryCounters {
    pub fn record_iteration(&mut self) {
        self.iterations += 1;
    }

    pub fn record_scalar_evaluation(&mut self) {
        self.scalar_evaluations += 1;
    }

    pub fn record_residual_call(&mut self) {
        self.residual_calls += 1;
        self.residual_requests += 1;
    }

    pub fn record_residual_request(&mut self) {
        self.residual_requests += 1;
    }

    pub fn record_jacobian_request(&mut self) {
        self.jacobian_requests += 1;
    }

    pub fn record_residual_chunk(&mut self) {
        self.residual_chunks += 1;
    }

    pub fn record_jacobian_chunk(&mut self) {
        self.jacobian_chunks += 1;
    }

    pub fn record_conversion(&mut self) {
        self.conversions += 1;
    }

    pub fn record_copy(&mut self) {
        self.copies += 1;
    }

    pub fn record_damping_trial(&mut self) {
        self.damping_trials += 1;
    }

    pub fn record_damping_rejection(&mut self) {
        self.damping_rejections += 1;
    }

    pub fn record_jacobian_recalculation(&mut self) {
        self.jacobian_recalculations += 1;
        self.jacobian_requests += 1;
    }

    pub fn record_linear_solve(&mut self) {
        self.linear_solves += 1;
    }

    /// Records the current baseline behaviour: one factorization per solve.
    ///
    /// The explicit pair is intentional. Once a reusable factorization is
    /// introduced, `factorizations` and `rhs_solves` can diverge without
    /// changing the meaning of `linear_solves`.
    pub fn record_factorization_and_rhs_solve(&mut self) {
        self.factorizations += 1;
        self.rhs_solves += 1;
    }

    pub fn record_factorization(&mut self) {
        self.factorizations += 1;
    }

    pub fn record_factorization_cache_hit(&mut self) {
        self.factorization_cache_hits += 1;
    }

    pub fn record_factorization_invalidation(&mut self) {
        self.factorization_invalidations += 1;
    }

    pub fn record_rhs_solve(&mut self) {
        self.rhs_solves += 1;
    }

    pub fn record_grid_refinement(&mut self) {
        self.grid_refinements += 1;
    }

    /// Compatibility projection for existing BVP_Damp consumers and stories.
    pub fn to_legacy_map(&self) -> HashMap<String, usize> {
        HashMap::from([
            ("number of iterations".to_string(), self.iterations as usize),
            (
                "number of residual calls".to_string(),
                self.residual_calls as usize,
            ),
            (
                "number of jacobians recalculations".to_string(),
                self.jacobian_recalculations as usize,
            ),
            (
                "number of solving linear systems".to_string(),
                self.linear_solves as usize,
            ),
            (
                "number of factorizations".to_string(),
                self.factorizations as usize,
            ),
            (
                "number of factorization cache hits".to_string(),
                self.factorization_cache_hits as usize,
            ),
            (
                "number of factorization invalidations".to_string(),
                self.factorization_invalidations as usize,
            ),
            ("number of RHS solves".to_string(), self.rhs_solves as usize),
            (
                "number of grid refinements".to_string(),
                self.grid_refinements as usize,
            ),
        ])
    }
}

/// Allocation-free mutable counter storage owned by a solver instance.
///
/// Solver callbacks are sometimes reached through an `&self` boundary. Using
/// `Cell` here lets those boundaries record residual requests without keeping
/// a mutable borrow of the solver alive or introducing a lock. The public
/// report still exposes the immutable `BvpTelemetryCounters` value.
#[derive(Clone, Debug, Default)]
pub(crate) struct BvpTelemetryRecorder {
    telemetry_mode: Cell<BvpTelemetryMode>,
    logging_config: Cell<BvpLoggingConfig>,
    solve_id: Cell<u64>,
    next_log_sequence: Cell<u64>,
    dropped_log_events: Cell<u64>,
    log_events: RefCell<Vec<BvpLogEvent>>,
    iterations: Cell<u64>,
    scalar_evaluations: Cell<u64>,
    residual_calls: Cell<u64>,
    residual_requests: Cell<u64>,
    jacobian_requests: Cell<u64>,
    jacobian_recalculations: Cell<u64>,
    residual_chunks: Cell<u64>,
    jacobian_chunks: Cell<u64>,
    conversions: Cell<u64>,
    copies: Cell<u64>,
    damping_trials: Cell<u64>,
    damping_rejections: Cell<u64>,
    linear_solves: Cell<u64>,
    factorizations: Cell<u64>,
    factorization_cache_hits: Cell<u64>,
    factorization_invalidations: Cell<u64>,
    rhs_solves: Cell<u64>,
    grid_refinements: Cell<u64>,
    iteration_elapsed: Cell<Duration>,
    damping_trial_elapsed: Cell<Duration>,
}

impl BvpTelemetryRecorder {
    /// Selects the collection policy without affecting numerical behaviour.
    pub(crate) fn set_telemetry_mode(&self, mode: BvpTelemetryMode) {
        self.telemetry_mode.set(mode);
    }

    pub(crate) fn telemetry_mode(&self) -> BvpTelemetryMode {
        self.telemetry_mode.get()
    }

    #[inline]
    fn telemetry_enabled(&self) -> bool {
        self.telemetry_mode.get() != BvpTelemetryMode::Off
    }

    /// Configures event retention without changing numerical behaviour.
    pub(crate) fn set_logging_mode(&self, mode: BvpLoggingMode) {
        self.set_logging_config(BvpLoggingConfig {
            mode,
            ..self.logging_config.get()
        });
    }

    pub(crate) fn set_logging_config(&self, config: BvpLoggingConfig) {
        self.logging_config.set(config);
        if config.mode == BvpLoggingMode::Off {
            self.log_events.borrow_mut().clear();
        }
    }

    pub(crate) fn logging_mode(&self) -> BvpLoggingMode {
        self.logging_config.get().mode
    }

    pub(crate) fn logging_config(&self) -> BvpLoggingConfig {
        self.logging_config.get()
    }

    /// Starts a new trace without resetting numerical counters.
    ///
    /// Counters intentionally retain their historical solver semantics, while
    /// decision events are solve-local and therefore begin at sequence zero.
    pub(crate) fn begin_solve(&self) {
        self.solve_id.set(self.solve_id.get().saturating_add(1));
        self.next_log_sequence.set(0);
        self.dropped_log_events.set(0);
        self.log_events.borrow_mut().clear();
        self.iteration_elapsed.set(Duration::ZERO);
        self.damping_trial_elapsed.set(Duration::ZERO);
    }

    /// Starts a solve-owned nonlinear iteration scope.
    ///
    /// The `Option` keeps the disabled path free of `Instant::now()` calls.
    #[inline]
    pub(crate) fn start_iteration_scope(&self) -> Option<Instant> {
        self.telemetry_enabled().then(Instant::now)
    }

    /// Completes an iteration scope and accumulates its elapsed duration.
    #[inline]
    pub(crate) fn finish_iteration_scope(&self, started: Option<Instant>) {
        if self.telemetry_enabled() {
            if let Some(started) = started {
                self.iteration_elapsed.set(
                    self.iteration_elapsed
                        .get()
                        .saturating_add(started.elapsed()),
                );
            }
        }
    }

    /// Starts a damping-trial scope without allocating or locking.
    #[inline]
    pub(crate) fn start_damping_trial_scope(&self) -> Option<Instant> {
        self.telemetry_enabled().then(Instant::now)
    }

    /// Completes a damping-trial scope and accumulates its elapsed duration.
    #[inline]
    pub(crate) fn finish_damping_trial_scope(&self, started: Option<Instant>) {
        if self.telemetry_enabled() {
            if let Some(started) = started {
                self.damping_trial_elapsed.set(
                    self.damping_trial_elapsed
                        .get()
                        .saturating_add(started.elapsed()),
                );
            }
        }
    }

    fn record_log_event(
        &self,
        level: BvpLogLevel,
        kind: BvpLogEventKind,
        iteration: u64,
        damping_trial: u64,
        mesh_revision: u64,
        value_a: f64,
        value_b: f64,
    ) {
        let config = self.logging_config.get();
        let mode = config.mode;
        let retain = match mode {
            BvpLoggingMode::Off => false,
            BvpLoggingMode::Warnings => level >= BvpLogLevel::Warn,
            BvpLoggingMode::Detailed => true,
        };
        if !retain {
            return;
        }

        if config.max_events == 0 {
            self.dropped_log_events
                .set(self.dropped_log_events.get().saturating_add(1));
            return;
        }

        let sequence = self.next_log_sequence.get();
        self.next_log_sequence.set(sequence.saturating_add(1));
        if self.log_events.borrow().len() >= config.max_events {
            self.dropped_log_events
                .set(self.dropped_log_events.get().saturating_add(1));
            return;
        }
        let event = BvpLogEvent {
            solve_id: self.solve_id.get(),
            sequence,
            level,
            kind,
            scope: match kind {
                BvpLogEventKind::MeshRevision => BvpTelemetryScope::MeshRevision,
                BvpLogEventKind::DampingTrial | BvpLogEventKind::DampingRejected => {
                    BvpTelemetryScope::DampingTrial
                }
                _ => BvpTelemetryScope::Solve,
            },
            stage: match kind {
                BvpLogEventKind::JacobianRefresh => Some(BvpTelemetryStage::JacobianRequest),
                BvpLogEventKind::Factorization
                | BvpLogEventKind::FactorizationCacheHit
                | BvpLogEventKind::FactorizationInvalidated => {
                    Some(BvpTelemetryStage::Factorization)
                }
                _ => None,
            },
            iteration,
            damping_trial,
            mesh_revision,
            value_a,
            value_b,
        };
        self.log_events.borrow_mut().push(event);

        match level {
            BvpLogLevel::Debug => log::debug!(
                target: "RustedSciThe::BVP_Damp",
                "event={:?} iteration={} damping_trial={} mesh_revision={} value_a={} value_b={}",
                kind, iteration, damping_trial, mesh_revision, value_a, value_b
            ),
            BvpLogLevel::Info => log::info!(
                target: "RustedSciThe::BVP_Damp",
                "event={:?} iteration={} damping_trial={} mesh_revision={} value_a={} value_b={}",
                kind, iteration, damping_trial, mesh_revision, value_a, value_b
            ),
            BvpLogLevel::Warn => log::warn!(
                target: "RustedSciThe::BVP_Damp",
                "event={:?} iteration={} damping_trial={} mesh_revision={} value_a={} value_b={}",
                kind, iteration, damping_trial, mesh_revision, value_a, value_b
            ),
            BvpLogLevel::Error => log::error!(
                target: "RustedSciThe::BVP_Damp",
                "event={:?} iteration={} damping_trial={} mesh_revision={} value_a={} value_b={}",
                kind, iteration, damping_trial, mesh_revision, value_a, value_b
            ),
        }
    }

    pub(crate) fn log_events_snapshot(&self) -> Vec<BvpLogEvent> {
        self.log_events.borrow().clone()
    }

    pub(crate) fn dropped_log_events(&self) -> u64 {
        self.dropped_log_events.get()
    }

    pub(crate) fn solve_id(&self) -> u64 {
        self.solve_id.get()
    }

    #[inline]
    pub(crate) fn iterations(&self) -> u64 {
        self.iterations.get()
    }

    #[inline]
    pub(crate) fn record_iteration(&self) {
        if !self.telemetry_enabled() {
            return;
        }
        self.iterations.set(self.iterations.get() + 1);
    }

    #[inline]
    pub(crate) fn record_scalar_evaluation(&self) {
        if !self.telemetry_enabled() {
            return;
        }
        self.scalar_evaluations
            .set(self.scalar_evaluations.get() + 1);
    }

    #[inline]
    pub(crate) fn record_residual_call(&self) {
        if !self.telemetry_enabled() {
            return;
        }
        self.residual_calls.set(self.residual_calls.get() + 1);
        self.residual_requests.set(self.residual_requests.get() + 1);
    }

    #[inline]
    pub(crate) fn record_residual_request(&self) {
        if !self.telemetry_enabled() {
            return;
        }
        self.residual_requests.set(self.residual_requests.get() + 1);
    }

    #[inline]
    pub(crate) fn record_jacobian_request(&self) {
        if !self.telemetry_enabled() {
            return;
        }
        self.jacobian_requests.set(self.jacobian_requests.get() + 1);
    }

    #[inline]
    pub(crate) fn record_residual_chunk(&self) {
        if !self.telemetry_enabled() {
            return;
        }
        self.residual_chunks.set(self.residual_chunks.get() + 1);
    }

    #[inline]
    pub(crate) fn record_jacobian_chunk(&self) {
        if !self.telemetry_enabled() {
            return;
        }
        self.jacobian_chunks.set(self.jacobian_chunks.get() + 1);
    }

    #[inline]
    pub(crate) fn record_conversion(&self) {
        if !self.telemetry_enabled() {
            return;
        }
        self.conversions.set(self.conversions.get() + 1);
    }

    #[inline]
    pub(crate) fn record_copy(&self) {
        if !self.telemetry_enabled() {
            return;
        }
        self.copies.set(self.copies.get() + 1);
    }

    #[inline]
    pub(crate) fn record_damping_trial(&self) {
        self.record_damping_trial_at(self.iterations.get(), f64::NAN, f64::NAN);
    }

    pub(crate) fn record_damping_trial_at(
        &self,
        iteration: u64,
        damping_coefficient: f64,
        residual_norm: f64,
    ) {
        if !self.telemetry_enabled() {
            return;
        }
        self.damping_trials.set(self.damping_trials.get() + 1);
        self.record_log_event(
            BvpLogLevel::Debug,
            BvpLogEventKind::DampingTrial,
            iteration,
            self.damping_trials.get(),
            0,
            damping_coefficient,
            residual_norm,
        );
    }

    #[inline]
    pub(crate) fn record_damping_rejection(&self) {
        self.record_damping_rejection_at(self.iterations.get(), f64::NAN, f64::NAN);
    }

    pub(crate) fn record_damping_rejection_at(
        &self,
        iteration: u64,
        damping_coefficient: f64,
        residual_norm: f64,
    ) {
        if !self.telemetry_enabled() {
            return;
        }
        self.damping_rejections
            .set(self.damping_rejections.get() + 1);
        self.record_log_event(
            BvpLogLevel::Warn,
            BvpLogEventKind::DampingRejected,
            iteration,
            self.damping_rejections.get(),
            0,
            damping_coefficient,
            residual_norm,
        );
    }

    pub(crate) fn record_backend_selection(&self, backend_code: u64, fallback: bool) {
        self.record_log_event(
            if fallback {
                BvpLogLevel::Warn
            } else {
                BvpLogLevel::Info
            },
            if fallback {
                BvpLogEventKind::BackendFallback
            } else {
                BvpLogEventKind::BackendSelected
            },
            self.iterations.get(),
            0,
            0,
            backend_code as f64,
            f64::NAN,
        );
    }

    pub(crate) fn record_termination(&self, converged: bool, value: f64) {
        self.record_log_event(
            if converged {
                BvpLogLevel::Info
            } else {
                BvpLogLevel::Error
            },
            BvpLogEventKind::Termination,
            self.iterations.get(),
            0,
            0,
            if converged { 1.0 } else { 0.0 },
            value,
        );
    }

    pub(crate) fn record_bound_limited_step(&self, factor: f64) {
        self.record_log_event(
            BvpLogLevel::Debug,
            BvpLogEventKind::BoundLimitedStep,
            self.iterations.get(),
            0,
            0,
            factor,
            f64::NAN,
        );
    }

    pub(crate) fn record_non_finite_value(&self, index: usize) {
        self.record_log_event(
            BvpLogLevel::Error,
            BvpLogEventKind::NonFiniteValue,
            self.iterations.get(),
            0,
            0,
            index as f64,
            f64::NAN,
        );
    }

    #[inline]
    pub(crate) fn record_jacobian_recalculation(&self) {
        if !self.telemetry_enabled() {
            return;
        }
        self.jacobian_recalculations
            .set(self.jacobian_recalculations.get() + 1);
        self.jacobian_requests.set(self.jacobian_requests.get() + 1);
        self.record_log_event(
            BvpLogLevel::Info,
            BvpLogEventKind::JacobianRefresh,
            self.iterations.get(),
            0,
            0,
            self.jacobian_recalculations.get() as f64,
            f64::NAN,
        );
    }

    #[inline]
    pub(crate) fn record_linear_solve(&self) {
        if !self.telemetry_enabled() {
            return;
        }
        self.linear_solves.set(self.linear_solves.get() + 1);
    }

    #[inline]
    pub(crate) fn record_factorization(&self) {
        if !self.telemetry_enabled() {
            return;
        }
        self.factorizations.set(self.factorizations.get() + 1);
        self.record_log_event(
            BvpLogLevel::Debug,
            BvpLogEventKind::Factorization,
            self.iterations.get(),
            0,
            0,
            self.factorizations.get() as f64,
            f64::NAN,
        );
    }

    #[inline]
    pub(crate) fn record_factorization_cache_hit(&self) {
        if !self.telemetry_enabled() {
            return;
        }
        self.factorization_cache_hits
            .set(self.factorization_cache_hits.get() + 1);
        self.record_log_event(
            BvpLogLevel::Debug,
            BvpLogEventKind::FactorizationCacheHit,
            self.iterations.get(),
            0,
            0,
            self.factorization_cache_hits.get() as f64,
            f64::NAN,
        );
    }

    #[inline]
    pub(crate) fn record_factorization_invalidation(&self) {
        if !self.telemetry_enabled() {
            return;
        }
        self.factorization_invalidations
            .set(self.factorization_invalidations.get() + 1);
        self.record_log_event(
            BvpLogLevel::Info,
            BvpLogEventKind::FactorizationInvalidated,
            self.iterations.get(),
            0,
            0,
            self.factorization_invalidations.get() as f64,
            f64::NAN,
        );
    }

    #[inline]
    pub(crate) fn record_rhs_solve(&self) {
        if !self.telemetry_enabled() {
            return;
        }
        self.rhs_solves.set(self.rhs_solves.get() + 1);
    }

    #[inline]
    pub(crate) fn record_grid_refinement(&self) {
        if !self.telemetry_enabled() {
            return;
        }
        self.grid_refinements.set(self.grid_refinements.get() + 1);
        self.record_log_event(
            BvpLogLevel::Info,
            BvpLogEventKind::MeshRevision,
            self.iterations.get(),
            0,
            self.grid_refinements.get(),
            self.grid_refinements.get() as f64,
            f64::NAN,
        );
    }

    pub(crate) fn snapshot(&self) -> BvpTelemetryCounters {
        if !self.telemetry_enabled() {
            return BvpTelemetryCounters::default();
        }
        BvpTelemetryCounters {
            iterations: self.iterations.get(),
            scalar_evaluations: self.scalar_evaluations.get(),
            residual_calls: self.residual_calls.get(),
            residual_requests: self.residual_requests.get(),
            jacobian_requests: self.jacobian_requests.get(),
            jacobian_recalculations: self.jacobian_recalculations.get(),
            residual_chunks: self.residual_chunks.get(),
            jacobian_chunks: self.jacobian_chunks.get(),
            conversions: self.conversions.get(),
            copies: self.copies.get(),
            damping_trials: self.damping_trials.get(),
            damping_rejections: self.damping_rejections.get(),
            linear_solves: self.linear_solves.get(),
            factorizations: self.factorizations.get(),
            factorization_cache_hits: self.factorization_cache_hits.get(),
            factorization_invalidations: self.factorization_invalidations.get(),
            rhs_solves: self.rhs_solves.get(),
            grid_refinements: self.grid_refinements.get(),
        }
    }

    pub(crate) fn scopes_snapshot(
        &self,
        timings: &BvpTimingSnapshot,
        counters: &BvpTelemetryCounters,
    ) -> BvpTelemetryScopes {
        let mut scopes = BvpTelemetryScopes::from_snapshot(timings, counters);
        if self.telemetry_enabled() {
            scopes.iteration.elapsed = self.iteration_elapsed.get();
            scopes.damping_trial.elapsed = self.damping_trial_elapsed.get();
        }
        scopes
    }
}

/// One scope in the telemetry hierarchy.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BvpTelemetryScopeMetrics {
    pub elapsed: Duration,
    pub events: u64,
}

/// Typed scope hierarchy.  Empty scopes are meaningful: they distinguish
/// "not entered" from a missing field in an older report.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BvpTelemetryScopes {
    pub solve: BvpTelemetryScopeMetrics,
    pub mesh_revision: BvpTelemetryScopeMetrics,
    pub iteration: BvpTelemetryScopeMetrics,
    pub damping_trial: BvpTelemetryScopeMetrics,
}

impl BvpTelemetryScopes {
    /// Builds the scope facts available from major timers and counters.
    /// Nested iteration/damping durations are supplied by the recorder when
    /// the solver has instrumented those runtime scopes.
    pub fn from_snapshot(timings: &BvpTimingSnapshot, counters: &BvpTelemetryCounters) -> Self {
        Self {
            solve: BvpTelemetryScopeMetrics {
                elapsed: timings.total,
                events: u64::from(timings.total > Duration::ZERO),
            },
            mesh_revision: BvpTelemetryScopeMetrics {
                elapsed: timings.grid_refinement,
                events: counters.grid_refinements,
            },
            iteration: BvpTelemetryScopeMetrics {
                elapsed: Duration::ZERO,
                events: counters.iterations,
            },
            damping_trial: BvpTelemetryScopeMetrics {
                elapsed: Duration::ZERO,
                events: counters.damping_trials,
            },
        }
    }
}

/// Typed duration snapshot for the major solver stages.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BvpTimingSnapshot {
    pub total: Duration,
    pub residual: Duration,
    pub jacobian: Duration,
    pub linear_system: Duration,
    pub factorization: Duration,
    pub rhs_solve: Duration,
    pub symbolic_operations: Duration,
    pub grid_refinement: Duration,
    /// Typed callback stages. This is the primary field for new consumers.
    pub callback_stage_timings: Vec<BvpCallbackStageTiming>,
    /// String presentation retained for old tables and log consumers.
    pub callback_stages: Vec<(String, Duration)>,
}

/// Stable typed telemetry returned alongside legacy presentation fields.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BvpTelemetrySnapshot {
    pub telemetry_mode: BvpTelemetryMode,
    pub counters: BvpTelemetryCounters,
    pub timings: BvpTimingSnapshot,
    pub scopes: BvpTelemetryScopes,
    pub storage: BvpStorageBytes,
    /// Normalized requested/selected runtime plan, when available.
    pub plan: Option<BvpResolvedPlan>,
    /// Typed decision trace retained when `BvpLoggingMode` is enabled.
    pub log_events: Vec<BvpLogEvent>,
    /// Number of events discarded after the bounded trace reached capacity.
    pub log_events_dropped: u64,
    /// Correlation id for the current solve-local decision trace.
    pub solve_id: u64,
    pub logging_config: BvpLoggingConfig,
    /// Optional Atom-native preparation telemetry captured before the solve.
    ///
    /// `None` is intentional for legacy Expr and pure-numeric routes.
    pub atom_discretization: Option<BvpAtomDiscretizationTelemetrySnapshot>,
    /// Typed cold-preparation stages, when the solver used the generated
    /// symbolic/backend handoff.
    pub generation: Option<BvpGenerationTelemetrySnapshot>,
    /// Runtime callback measurements for ExprLegacy Lambdify, if enabled.
    pub legacy_lambdify: Option<BvpLambdifyTelemetrySnapshot>,
    /// Runtime callback measurements for AtomView Lambdify, if enabled.
    pub atom_lambdify: Option<BvpLambdifyTelemetrySnapshot>,
    /// Runtime stages for the direct no-Mutex Banded Jacobian callback.
    ///
    /// This is a solver-level immutable projection of the symbolic callback
    /// telemetry. `None` means that the active route did not install the
    /// direct Banded callback.
    pub direct_banded_jacobian:
        Option<crate::symbolic::bvp::telemetry::BvpDirectJacobianTelemetrySnapshot>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_project_to_legacy_names_without_stringly_internal_state() {
        let mut counters = BvpTelemetryCounters::default();
        counters.record_iteration();
        counters.record_scalar_evaluation();
        counters.record_residual_call();
        counters.record_residual_chunk();
        counters.record_jacobian_chunk();
        counters.record_conversion();
        counters.record_copy();
        counters.record_damping_trial();
        counters.record_damping_rejection();
        counters.record_jacobian_recalculation();
        counters.record_linear_solve();
        counters.record_factorization_and_rhs_solve();
        counters.record_factorization_cache_hit();
        counters.record_factorization_invalidation();
        counters.record_grid_refinement();

        let legacy = counters.to_legacy_map();
        assert_eq!(legacy["number of iterations"], 1);
        assert_eq!(legacy["number of residual calls"], 1);
        assert_eq!(legacy["number of jacobians recalculations"], 1);
        assert_eq!(counters.scalar_evaluations, 1);
        assert_eq!(counters.residual_requests, 1);
        assert_eq!(counters.jacobian_requests, 1);
        assert_eq!(counters.residual_chunks, 1);
        assert_eq!(counters.jacobian_chunks, 1);
        assert_eq!(counters.conversions, 1);
        assert_eq!(counters.copies, 1);
        assert_eq!(counters.damping_trials, 1);
        assert_eq!(counters.damping_rejections, 1);
        assert_eq!(legacy["number of solving linear systems"], 1);
        assert_eq!(legacy["number of factorizations"], 1);
        assert_eq!(legacy["number of factorization cache hits"], 1);
        assert_eq!(legacy["number of factorization invalidations"], 1);
        assert_eq!(legacy["number of RHS solves"], 1);
        assert_eq!(legacy["number of grid refinements"], 1);
    }

    #[test]
    fn solver_snapshot_keeps_atom_preparation_optional_and_typed() {
        let snapshot = BvpTelemetrySnapshot {
            telemetry_mode: BvpTelemetryMode::Counters,
            counters: BvpTelemetryCounters::default(),
            timings: BvpTimingSnapshot::default(),
            scopes: BvpTelemetryScopes::default(),
            storage: BvpStorageBytes::default(),
            plan: None,
            log_events: Vec::new(),
            log_events_dropped: 0,
            solve_id: 0,
            logging_config: BvpLoggingConfig::default(),
            atom_discretization: Some(BvpAtomDiscretizationTelemetrySnapshot {
                total: Duration::from_millis(2),
                ..Default::default()
            }),
            generation: None,
            legacy_lambdify: None,
            atom_lambdify: None,
            direct_banded_jacobian: None,
        };

        assert_eq!(
            snapshot
                .atom_discretization
                .expect("typed preparation snapshot")
                .total,
            Duration::from_millis(2)
        );
    }

    #[test]
    fn storage_estimates_are_explicit_and_scope_snapshot_is_typed() {
        let dense = BvpStorageBytes::dense_f64(10, 20);
        assert_eq!(dense.jacobian_values, Some(10 * 20 * 8));
        assert!(dense.estimated);

        let banded = BvpStorageBytes::banded_f64(10, 1, 2);
        assert_eq!(banded.jacobian_values, Some(10 * 4 * 8));
        assert_eq!(banded.factor, banded.jacobian_values);
        let sparse = BvpStorageBytes::for_solver_method("Sparse", 10, 20, (1, 2));
        assert_eq!(sparse.jacobian_values, None);
        assert!(!sparse.estimated);

        let timings = BvpTimingSnapshot {
            total: Duration::from_millis(4),
            grid_refinement: Duration::from_millis(1),
            callback_stage_timings: Vec::new(),
            callback_stages: Vec::new(),
            ..Default::default()
        };
        let mut counters = BvpTelemetryCounters::default();
        counters.iterations = 3;
        counters.grid_refinements = 1;
        counters.damping_trials = 2;
        let scopes = BvpTelemetryScopes::from_snapshot(&timings, &counters);
        assert_eq!(scopes.solve.events, 1);
        assert_eq!(scopes.iteration.events, 3);
        assert_eq!(scopes.mesh_revision.elapsed, Duration::from_millis(1));
        assert_eq!(scopes.damping_trial.events, 2);
    }

    #[test]
    fn decision_logging_is_opt_in_and_preserves_typed_event_order() {
        let recorder = BvpTelemetryRecorder::default();
        recorder.record_damping_rejection();
        assert!(recorder.log_events_snapshot().is_empty());

        recorder.set_logging_mode(BvpLoggingMode::Detailed);
        recorder.record_jacobian_recalculation();
        recorder.record_factorization_cache_hit();
        recorder.record_damping_rejection();

        let events = recorder.log_events_snapshot();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].sequence, 0);
        assert_eq!(events[0].kind, BvpLogEventKind::JacobianRefresh);
        assert_eq!(events[1].kind, BvpLogEventKind::FactorizationCacheHit);
        assert_eq!(events[2].kind, BvpLogEventKind::DampingRejected);
        assert_eq!(events[2].level, BvpLogLevel::Warn);

        recorder.set_logging_mode(BvpLoggingMode::Warnings);
        recorder.record_factorization();
        recorder.record_factorization_invalidation();
        recorder.record_damping_rejection();
        let events = recorder.log_events_snapshot();
        assert_eq!(events.len(), 4);
        assert_eq!(events[3].kind, BvpLogEventKind::DampingRejected);
    }

    #[test]
    fn decision_logging_is_bounded_and_carries_solve_context() {
        let recorder = BvpTelemetryRecorder::default();
        recorder
            .set_logging_config(BvpLoggingConfig::new(BvpLoggingMode::Detailed).with_max_events(2));
        recorder.begin_solve();
        recorder.record_jacobian_recalculation();
        recorder.record_factorization();
        recorder.record_grid_refinement();

        let events = recorder.log_events_snapshot();
        assert_eq!(events.len(), 2);
        assert_eq!(recorder.dropped_log_events(), 1);
        assert!(events.iter().all(|event| event.solve_id == 1));
        assert_eq!(events[0].scope, BvpTelemetryScope::Solve);
        assert_eq!(events[0].stage, Some(BvpTelemetryStage::JacobianRequest));

        recorder.begin_solve();
        assert!(recorder.log_events_snapshot().is_empty());
        recorder.record_damping_trial_at(7, 0.25, 1.5);
        let event = recorder.log_events_snapshot()[0];
        assert_eq!(event.solve_id, 2);
        assert_eq!(event.iteration, 7);
        assert_eq!(event.scope, BvpTelemetryScope::DampingTrial);
        assert_eq!(event.value_a, 0.25);
        assert_eq!(event.value_b, 1.5);
    }

    #[test]
    fn decision_logging_preserves_fallback_and_failure_events() {
        let recorder = BvpTelemetryRecorder::default();
        recorder
            .set_logging_config(BvpLoggingConfig::new(BvpLoggingMode::Detailed).with_max_events(8));
        recorder.begin_solve();
        recorder.record_backend_selection(2, true);
        recorder.record_termination(false, 3.5);

        let events = recorder.log_events_snapshot();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].kind, BvpLogEventKind::BackendFallback);
        assert_eq!(events[0].level, BvpLogLevel::Warn);
        assert_eq!(events[1].kind, BvpLogEventKind::Termination);
        assert_eq!(events[1].level, BvpLogLevel::Error);
        assert_eq!(events[1].value_a, 0.0);
        assert_eq!(events[1].value_b, 3.5);
    }

    #[test]
    fn telemetry_off_does_not_collect_counters_but_logging_remains_independent() {
        let recorder = BvpTelemetryRecorder::default();
        recorder.set_telemetry_mode(BvpTelemetryMode::Off);
        recorder.set_logging_mode(BvpLoggingMode::Detailed);
        recorder.begin_solve();
        recorder.record_iteration();
        recorder.record_residual_call();
        recorder.record_jacobian_recalculation();
        recorder.record_linear_solve();
        recorder.record_termination(false, 2.5);

        assert_eq!(recorder.snapshot(), BvpTelemetryCounters::default());
        let events = recorder.log_events_snapshot();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, BvpLogEventKind::Termination);
        assert_eq!(recorder.telemetry_mode(), BvpTelemetryMode::Off);
    }

    #[test]
    fn iteration_and_damping_scopes_report_real_elapsed_time() {
        let recorder = BvpTelemetryRecorder::default();
        let iteration_started = recorder.start_iteration_scope();
        std::thread::sleep(Duration::from_millis(1));
        recorder.finish_iteration_scope(iteration_started);

        let trial_started = recorder.start_damping_trial_scope();
        std::thread::sleep(Duration::from_millis(1));
        recorder.finish_damping_trial_scope(trial_started);

        let counters = recorder.snapshot();
        let scopes = recorder.scopes_snapshot(&BvpTimingSnapshot::default(), &counters);
        assert!(scopes.iteration.elapsed > Duration::ZERO);
        assert!(scopes.damping_trial.elapsed > Duration::ZERO);
    }

    #[test]
    fn off_mode_does_not_start_iteration_or_damping_scopes() {
        let recorder = BvpTelemetryRecorder::default();
        recorder.set_telemetry_mode(BvpTelemetryMode::Off);
        let iteration_started = recorder.start_iteration_scope();
        let trial_started = recorder.start_damping_trial_scope();
        recorder.finish_iteration_scope(iteration_started);
        recorder.finish_damping_trial_scope(trial_started);

        let counters = recorder.snapshot();
        let scopes = recorder.scopes_snapshot(&BvpTimingSnapshot::default(), &counters);
        assert_eq!(iteration_started, None);
        assert_eq!(trial_started, None);
        assert_eq!(scopes.iteration, BvpTelemetryScopeMetrics::default());
        assert_eq!(scopes.damping_trial, BvpTelemetryScopeMetrics::default());
    }

    #[test]
    fn begin_solve_resets_iteration_and_damping_scope_elapsed() {
        let recorder = BvpTelemetryRecorder::default();
        recorder.finish_iteration_scope(recorder.start_iteration_scope());
        recorder.finish_damping_trial_scope(recorder.start_damping_trial_scope());

        let before = recorder.scopes_snapshot(&BvpTimingSnapshot::default(), &recorder.snapshot());
        assert!(before.iteration.elapsed > Duration::ZERO);
        assert!(before.damping_trial.elapsed > Duration::ZERO);

        recorder.begin_solve();
        let after = recorder.scopes_snapshot(&BvpTimingSnapshot::default(), &recorder.snapshot());
        assert_eq!(after.iteration.elapsed, Duration::ZERO);
        assert_eq!(after.damping_trial.elapsed, Duration::ZERO);
    }
}
