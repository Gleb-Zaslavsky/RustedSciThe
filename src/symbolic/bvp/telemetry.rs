//! Typed telemetry for direct BVP symbolic callbacks.
//!
//! The direct Jacobian path deliberately uses atomics rather than a `Mutex`:
//! callbacks may evaluate rows/diagonals in parallel, and telemetry must not
//! reintroduce the lock contention that the callback migration removes.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Selects how independent pure-Lambdify callback work is evaluated.
///
/// This policy affects runtime residual/Jacobian evaluation only. Symbolic
/// preparation remains free to use its own parallel implementation. The
/// parallel callbacks write into disjoint temporary results (or scatter in a
/// single owner thread), so selecting `Parallel` never requires a `Mutex`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BvpLambdifyExecutionPolicy {
    /// Evaluate callback entries on the calling thread.
    Sequential,
    /// Use Rayon once the callback has at least `min_work` independent items.
    Parallel { min_work: usize },
    /// Select sequential or Rayon execution from work size and worker count.
    ///
    /// `min_work` is an explicit caller-provided lower bound.  `Auto` also
    /// requires a conservative eight scalar items per Rayon worker.  This is
    /// intentionally a deterministic gate, not a runtime calibration: the
    /// first residual/Jacobian callback must not pay a hidden measurement cost.
    Auto { min_work: usize },
}

impl Default for BvpLambdifyExecutionPolicy {
    fn default() -> Self {
        Self::Parallel { min_work: 0 }
    }
}

impl BvpLambdifyExecutionPolicy {
    const AUTO_WORK_PER_TASK: usize = 8;

    /// Returns whether a callback with this amount of work should dispatch to
    /// worker threads.
    #[inline]
    pub(crate) fn should_parallel(self, work: usize) -> bool {
        self.should_parallel_with_tasks(work, work)
    }

    /// Returns whether a callback should dispatch after accounting for the
    /// number of independent jobs exposed by its storage layout.
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

    /// Estimates the number of coarse jobs available to Auto for a scalar
    /// workload. Layout-specific callers may sum this over independent groups.
    #[inline]
    pub(crate) fn auto_task_count(work: usize) -> usize {
        work.div_ceil(Self::AUTO_WORK_PER_TASK)
    }
}

#[derive(Default)]
struct Inner {
    calls: AtomicU64,
    work_items: AtomicU64,
    elapsed_ns: AtomicU64,
    errors: AtomicU64,
    argument_prepare_ns: AtomicU64,
    evaluator_ns: AtomicU64,
    storage_write_ns: AtomicU64,
    assembly_alloc_ns: AtomicU64,
    evaluator_calls: AtomicU64,
    storage_writes: AtomicU64,
    parallel_dispatches: AtomicU64,
    sequential_dispatches: AtomicU64,
    diagonal_dispatches: AtomicU64,
    entry_dispatches: AtomicU64,
    effective_task_count: AtomicU64,
    prepared_atom_evaluators: AtomicU64,
    prepared_atom_nodes: AtomicU64,
    prepared_atom_add_nodes: AtomicU64,
    prepared_atom_mul_nodes: AtomicU64,
    prepared_atom_powi_nodes: AtomicU64,
    prepared_atom_pow_nodes: AtomicU64,
    prepared_atom_builtin_nodes: AtomicU64,
    prepared_atom_custom_nodes: AtomicU64,
}

/// Shared recording handle attached to a direct BVP Jacobian callback.
///
/// The low-level constructor remains detailed for direct callers and
/// compatibility tests. Solver-owned callbacks can use [`Self::disabled`] so
/// they do not allocate an `Arc`, read an atomic, or create an `Instant` per
/// Jacobian request when diagnostics are off.
#[derive(Clone)]
pub struct BvpDirectJacobianTelemetry {
    inner: Option<Arc<Inner>>,
    timing_enabled: bool,
}

impl Default for BvpDirectJacobianTelemetry {
    fn default() -> Self {
        Self::new()
    }
}

/// Immutable telemetry snapshot suitable for solver diagnostics and story tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BvpDirectJacobianTelemetrySnapshot {
    pub calls: u64,
    pub work_items: u64,
    pub elapsed: Duration,
    pub errors: u64,
    /// Time spent preparing `[parameters..., unknowns...]` callback arguments.
    pub argument_prepare_elapsed: Duration,
    /// Time spent evaluating compiled scalar Jacobian entries. For the
    /// diagonal layout this includes the fused write into the diagonal slot,
    /// because separating that write would require a temporary value buffer.
    pub evaluator_elapsed: Duration,
    /// Time spent writing evaluated values into native band storage.
    pub storage_write_elapsed: Duration,
    /// Time spent allocating and zero-initializing the BandedAssembly.
    pub assembly_alloc_elapsed: Duration,
    /// Number of scalar evaluator invocations.
    pub evaluator_calls: u64,
    /// Number of numeric writes into native band storage.
    /// Structural zero slots are not counted: the assembly is zero-initialized
    /// once per callback and the diagonal runtime skips those slots entirely.
    pub storage_writes: u64,
    /// Number of callbacks dispatched through the parallel policy.
    pub parallel_dispatches: u64,
    /// Number of callbacks evaluated sequentially.
    pub sequential_dispatches: u64,
    /// Number of callbacks using diagonal work decomposition.
    pub diagonal_dispatches: u64,
    /// Number of callbacks using entry work decomposition.
    pub entry_dispatches: u64,
    /// Sum of independent evaluator tasks exposed to the dispatch policy.
    ///
    /// For a long diagonal this is larger than one even though the storage
    /// layout contains a single diagonal. It makes Auto decisions auditable
    /// instead of reporting only the selected branch.
    pub effective_task_count: u64,
    /// Number of scalar Atom callbacks compiled into the prepared plan.
    /// Zero means that this route is ExprLegacy or does not expose Atom IR.
    pub prepared_atom_evaluators: u64,
    /// Total prepared Atom IR node count across scalar callbacks.
    pub prepared_atom_nodes: u64,
    pub prepared_atom_add_nodes: u64,
    pub prepared_atom_mul_nodes: u64,
    pub prepared_atom_powi_nodes: u64,
    pub prepared_atom_pow_nodes: u64,
    pub prepared_atom_builtin_nodes: u64,
    pub prepared_atom_custom_nodes: u64,
}

#[derive(Debug, Default)]
struct LambdifyInner {
    residual_calls: AtomicU64,
    residual_elapsed_ns: AtomicU64,
    jacobian_calls: AtomicU64,
    jacobian_elapsed_ns: AtomicU64,
    parallel_dispatches: AtomicU64,
    sequential_dispatches: AtomicU64,
}

/// Controls the amount of runtime work collected for Lambdify callbacks.
///
/// `Off` is the production default and does not allocate an `Arc`, touch an
/// atomic counter, or call `Instant::now()` on the callback hot path.
/// `Counters` adds only relaxed call counters. `Detailed` additionally records
/// callback wall-clock time with relaxed atomics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BvpLambdifyTelemetryMode {
    /// No runtime callback measurements.
    #[default]
    Off,
    /// Count residual and Jacobian callback requests without timing them.
    Counters,
    /// Count callback requests and accumulate elapsed wall-clock time.
    Detailed,
}

/// Lock-free runtime counters shared by prepared Lambdify callback pairs.
///
/// The stream is deliberately backend-neutral: `ExprLegacy` and `AtomView`
/// each receive a different handle, so their callback costs cannot be mixed,
/// while story tests can still consume one stable snapshot schema.
#[derive(Clone, Debug)]
pub struct BvpLambdifyTelemetry {
    mode: BvpLambdifyTelemetryMode,
    inner: Option<Arc<LambdifyInner>>,
}

impl Default for BvpLambdifyTelemetry {
    fn default() -> Self {
        Self::disabled()
    }
}

/// Immutable callback-cost snapshot for one prepared Lambdify route.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BvpLambdifyTelemetrySnapshot {
    /// Collection mode used by this callback stream.
    pub mode: BvpLambdifyTelemetryMode,
    pub residual_calls: u64,
    pub residual_elapsed: Duration,
    pub jacobian_calls: u64,
    pub jacobian_elapsed: Duration,
    pub parallel_dispatches: u64,
    pub sequential_dispatches: u64,
}

impl BvpLambdifyTelemetry {
    /// Creates a detailed stream for direct callers and compatibility tests.
    ///
    /// Solver configurations use [`Self::disabled`] by default; this
    /// constructor preserves the historical behaviour of the low-level
    /// telemetry helper.
    pub fn new() -> Self {
        Self::with_mode(BvpLambdifyTelemetryMode::Detailed)
    }

    /// Creates a runtime-disabled stream with no shared allocation.
    pub fn disabled() -> Self {
        Self {
            mode: BvpLambdifyTelemetryMode::Off,
            inner: None,
        }
    }

    /// Creates a low-overhead call-counter stream.
    pub fn counters() -> Self {
        Self::with_mode(BvpLambdifyTelemetryMode::Counters)
    }

    /// Creates a stream with callback calls and elapsed wall-clock timings.
    pub fn detailed() -> Self {
        Self::with_mode(BvpLambdifyTelemetryMode::Detailed)
    }

    /// Creates a stream for an explicit collection mode.
    pub fn with_mode(mode: BvpLambdifyTelemetryMode) -> Self {
        Self {
            mode,
            inner: (mode != BvpLambdifyTelemetryMode::Off)
                .then(|| Arc::new(LambdifyInner::default())),
        }
    }

    /// Returns the configured collection mode.
    pub fn mode(&self) -> BvpLambdifyTelemetryMode {
        self.mode
    }

    /// Starts timing only in `Detailed` mode.
    #[inline]
    pub fn start_timing(&self) -> Option<Instant> {
        (self.mode == BvpLambdifyTelemetryMode::Detailed).then(Instant::now)
    }

    #[inline]
    fn record(&self, residual: bool, started: Option<Instant>) {
        let Some(inner) = &self.inner else {
            return;
        };
        if residual {
            inner.residual_calls.fetch_add(1, Ordering::Relaxed);
        } else {
            inner.jacobian_calls.fetch_add(1, Ordering::Relaxed);
        }
        if self.mode == BvpLambdifyTelemetryMode::Detailed {
            let elapsed_ns = started
                .map(|started| started.elapsed().as_nanos().min(u64::MAX as u128) as u64)
                .unwrap_or(0);
            if residual {
                inner
                    .residual_elapsed_ns
                    .fetch_add(elapsed_ns, Ordering::Relaxed);
            } else {
                inner
                    .jacobian_elapsed_ns
                    .fetch_add(elapsed_ns, Ordering::Relaxed);
            }
        }
    }

    /// Records a residual callback, using a timer started by `start_timing`.
    #[inline]
    pub fn record_residual_sample(&self, started: Option<Instant>) {
        self.record(true, started);
    }

    /// Records a Jacobian callback, using a timer started by `start_timing`.
    #[inline]
    pub fn record_jacobian_sample(&self, started: Option<Instant>) {
        self.record(false, started);
    }

    /// Records one residual callback invocation.
    #[inline]
    pub fn record_residual(&self, elapsed: Duration) {
        let Some(inner) = &self.inner else {
            return;
        };
        inner.residual_calls.fetch_add(1, Ordering::Relaxed);
        if self.mode == BvpLambdifyTelemetryMode::Detailed {
            inner.residual_elapsed_ns.fetch_add(
                elapsed.as_nanos().min(u64::MAX as u128) as u64,
                Ordering::Relaxed,
            );
        }
    }

    /// Records one Jacobian callback invocation.
    #[inline]
    pub fn record_jacobian(&self, elapsed: Duration) {
        let Some(inner) = &self.inner else {
            return;
        };
        inner.jacobian_calls.fetch_add(1, Ordering::Relaxed);
        if self.mode == BvpLambdifyTelemetryMode::Detailed {
            inner.jacobian_elapsed_ns.fetch_add(
                elapsed.as_nanos().min(u64::MAX as u128) as u64,
                Ordering::Relaxed,
            );
        }
    }

    /// Records which execution branch handled one callback request.
    ///
    /// The disabled mode returns before touching atomics, preserving the
    /// zero-overhead default for production solves without diagnostics.
    #[inline]
    pub fn record_dispatch(&self, policy: BvpLambdifyExecutionPolicy, work: usize) {
        self.record_dispatch_selected(policy.should_parallel(work));
    }

    /// Records a dispatch decision using layout-aware task information.
    #[inline]
    pub fn record_dispatch_with_tasks(
        &self,
        policy: BvpLambdifyExecutionPolicy,
        work: usize,
        parallel_tasks: usize,
    ) {
        self.record_dispatch_selected(policy.should_parallel_with_tasks(work, parallel_tasks));
    }

    #[inline]
    fn record_dispatch_selected(&self, parallel: bool) {
        let Some(inner) = &self.inner else {
            return;
        };
        if parallel {
            inner.parallel_dispatches.fetch_add(1, Ordering::Relaxed);
        } else {
            inner.sequential_dispatches.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Reads counters without taking a mutex.
    pub fn snapshot(&self) -> BvpLambdifyTelemetrySnapshot {
        let Some(inner) = &self.inner else {
            return BvpLambdifyTelemetrySnapshot::default();
        };
        BvpLambdifyTelemetrySnapshot {
            mode: self.mode,
            residual_calls: inner.residual_calls.load(Ordering::Relaxed),
            residual_elapsed: Duration::from_nanos(
                inner.residual_elapsed_ns.load(Ordering::Relaxed),
            ),
            jacobian_calls: inner.jacobian_calls.load(Ordering::Relaxed),
            jacobian_elapsed: Duration::from_nanos(
                inner.jacobian_elapsed_ns.load(Ordering::Relaxed),
            ),
            parallel_dispatches: inner.parallel_dispatches.load(Ordering::Relaxed),
            sequential_dispatches: inner.sequential_dispatches.load(Ordering::Relaxed),
        }
    }
}

/// Flattens fixed parameter values followed by the current unknown vector.
///
/// Both Lambdify routes use the same argument ABI. Keeping this helper in the
/// shared support module prevents ExprLegacy and AtomView from drifting apart.
pub(crate) fn flatten_lambdify_args(
    parameter_values: Option<&[f64]>,
    unknowns: &[f64],
) -> Vec<f64> {
    let mut args =
        Vec::with_capacity(parameter_values.map_or(0, |values| values.len()) + unknowns.len());
    if let Some(values) = parameter_values {
        args.extend_from_slice(values);
    }
    args.extend_from_slice(unknowns);
    args
}

/// Typed symbolic-preparation timings for one BVP Jacobian build.
///
/// The historical API still exposes the raw string-keyed timing map for
/// compatibility, but new diagnostics can consume this schema directly and
/// compare ExprLegacy with AtomView without parsing presentation text.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BvpSymbolicPreparationTelemetrySnapshot {
    pub variable_sets: Duration,
    pub row_differentiation: Duration,
    pub dense_cache_materialization: Duration,
    pub sparse_cache_flatten: Duration,
}

/// Typed wall-clock breakdown for one complete symbolic BVP preparation.
///
/// The legacy timing table still remains available for compatibility, but the
/// solver handoff uses this snapshot so a slow route can be diagnosed without
/// parsing presentation strings or relying on a `HashMap` key convention.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BvpGenerationTelemetrySnapshot {
    pub total: Duration,
    pub discretization: Duration,
    pub symbolic_jacobian: Duration,
    pub find_bandwidth: Duration,
    pub backend_selection: Duration,
    pub runtime_binding: Duration,
    pub lambdify_jacobian_compile: Duration,
    pub lambdify_residual_compile: Duration,
}

/// Exact stage timings produced by Atom-native BVP discretization.
///
/// The snapshot is intentionally typed and keeps sub-millisecond precision.
/// The older `timer_hash` on `DiscretizedBvpAtomSystem` remains available as a
/// presentation/compatibility projection, but it is not the primary telemetry
/// representation for new code or story-test assertions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BvpAtomDiscretizationTelemetrySnapshot {
    pub boundary_conditions: Duration,
    pub discretization: Duration,
    pub boundary_application: Duration,
    pub flat_list: Duration,
    pub consistency: Duration,
    pub bounds_and_tolerances: Duration,
    pub total: Duration,
}

impl BvpAtomDiscretizationTelemetrySnapshot {
    /// Returns whether each measured stage fits within the total measurement.
    ///
    /// Discretization itself may use worker threads, so this checks individual
    /// stage containment rather than summing stage durations.
    pub fn stages_fit_total(&self) -> bool {
        self.boundary_conditions <= self.total
            && self.discretization <= self.total
            && self.boundary_application <= self.total
            && self.flat_list <= self.total
            && self.consistency <= self.total
            && self.bounds_and_tolerances <= self.total
    }
}

impl BvpSymbolicPreparationTelemetrySnapshot {
    /// Converts legacy seconds-based stage keys into the typed snapshot.
    pub(crate) fn from_seconds_map(timings: &HashMap<String, f64>) -> Self {
        fn duration(timings: &HashMap<String, f64>, key: &str) -> Duration {
            Duration::from_secs_f64(timings.get(key).copied().unwrap_or(0.0).max(0.0))
        }

        Self {
            variable_sets: duration(timings, "symbolic jacobian variable sets time"),
            row_differentiation: duration(timings, "symbolic jacobian row differentiation time"),
            dense_cache_materialization: duration(
                timings,
                "symbolic jacobian dense cache materialize time",
            ),
            sparse_cache_flatten: duration(timings, "symbolic jacobian sparse cache flatten time"),
        }
    }
}

impl BvpGenerationTelemetrySnapshot {
    /// Converts the raw seconds snapshot into the typed preparation contract.
    pub(crate) fn from_seconds_map(timings: &HashMap<String, f64>) -> Self {
        fn duration(timings: &HashMap<String, f64>, key: &str) -> Duration {
            Duration::from_secs_f64(timings.get(key).copied().unwrap_or(0.0).max(0.0))
        }

        Self {
            total: duration(timings, "total time, sec"),
            discretization: duration(timings, "discretization time"),
            symbolic_jacobian: duration(timings, "symbolic jacobian time"),
            find_bandwidth: duration(timings, "find bandwidth time"),
            backend_selection: duration(timings, "backend selection time"),
            runtime_binding: duration(timings, "runtime binding time"),
            lambdify_jacobian_compile: duration(timings, "lambdify jacobian callback compile time"),
            lambdify_residual_compile: duration(timings, "lambdify residual callback compile time"),
        }
    }
}

impl BvpDirectJacobianTelemetry {
    /// Creates an independent telemetry stream for one prepared callback.
    pub fn new() -> Self {
        Self {
            inner: Some(Arc::new(Inner::default())),
            timing_enabled: true,
        }
    }

    /// Creates a counter-only stream without callback wall-clock timers.
    pub fn counters() -> Self {
        Self {
            inner: Some(Arc::new(Inner::default())),
            timing_enabled: false,
        }
    }

    /// Creates a disabled stream without a shared allocation.
    pub fn disabled() -> Self {
        Self {
            inner: None,
            timing_enabled: false,
        }
    }

    /// Returns whether the callback should collect direct-path telemetry.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.inner.is_some()
    }

    /// Returns whether the callback should create wall-clock timestamps.
    #[inline]
    pub fn timing_enabled(&self) -> bool {
        self.timing_enabled
    }

    /// Records one completed callback invocation.
    #[inline]
    pub fn record_call(&self, elapsed: Duration, work_items: usize) {
        let Some(inner) = &self.inner else {
            return;
        };
        inner.calls.fetch_add(1, Ordering::Relaxed);
        inner
            .work_items
            .fetch_add(work_items as u64, Ordering::Relaxed);
        if self.timing_enabled {
            inner.elapsed_ns.fetch_add(
                elapsed.as_nanos().min(u64::MAX as u128) as u64,
                Ordering::Relaxed,
            );
        }
    }

    /// Records a callback rejected before assembly.
    #[inline]
    pub fn record_error(&self) {
        if let Some(inner) = &self.inner {
            inner.errors.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Records the typed stage split for one Banded Jacobian callback.
    ///
    /// These fields intentionally use atomics and fixed slots. The hot path
    /// never creates a stage label or a map entry; callers may disable reading
    /// the snapshot entirely when diagnostics are not needed.
    #[inline]
    pub fn record_stage_breakdown(
        &self,
        argument_prepare: Duration,
        evaluator: Duration,
        storage_write: Duration,
        assembly_alloc: Duration,
        evaluator_calls: usize,
        storage_writes: usize,
    ) {
        let Some(inner) = &self.inner else {
            return;
        };
        if self.timing_enabled {
            inner.argument_prepare_ns.fetch_add(
                argument_prepare.as_nanos().min(u64::MAX as u128) as u64,
                Ordering::Relaxed,
            );
            inner.evaluator_ns.fetch_add(
                evaluator.as_nanos().min(u64::MAX as u128) as u64,
                Ordering::Relaxed,
            );
            inner.storage_write_ns.fetch_add(
                storage_write.as_nanos().min(u64::MAX as u128) as u64,
                Ordering::Relaxed,
            );
            inner.assembly_alloc_ns.fetch_add(
                assembly_alloc.as_nanos().min(u64::MAX as u128) as u64,
                Ordering::Relaxed,
            );
        }
        inner
            .evaluator_calls
            .fetch_add(evaluator_calls as u64, Ordering::Relaxed);
        inner
            .storage_writes
            .fetch_add(storage_writes as u64, Ordering::Relaxed);
    }

    /// Records the selected policy and work decomposition without allocating.
    #[inline]
    pub fn record_dispatch(&self, parallel: bool, diagonal: bool) {
        self.record_dispatch_with_tasks(parallel, diagonal, 0);
    }

    /// Records dispatch selection together with the number of independent
    /// tasks exposed by the prepared layout.
    #[inline]
    pub fn record_dispatch_with_tasks(
        &self,
        parallel: bool,
        diagonal: bool,
        effective_task_count: usize,
    ) {
        let Some(inner) = &self.inner else {
            return;
        };
        if parallel {
            inner.parallel_dispatches.fetch_add(1, Ordering::Relaxed);
        } else {
            inner.sequential_dispatches.fetch_add(1, Ordering::Relaxed);
        }
        if diagonal {
            inner.diagonal_dispatches.fetch_add(1, Ordering::Relaxed);
        } else {
            inner.entry_dispatches.fetch_add(1, Ordering::Relaxed);
        }
        inner
            .effective_task_count
            .fetch_add(effective_task_count as u64, Ordering::Relaxed);
    }

    /// Records the compile-time Atom IR shape once per prepared callback plan.
    ///
    /// This is intentionally separate from callback counters: repeated
    /// Jacobian requests must not multiply structural metrics, and disabled
    /// telemetry returns before touching any atomic field.
    #[inline]
    pub fn record_prepared_atom_metrics(
        &self,
        evaluators: usize,
        nodes: usize,
        add_nodes: usize,
        mul_nodes: usize,
        powi_nodes: usize,
        pow_nodes: usize,
        builtin_nodes: usize,
        custom_nodes: usize,
    ) {
        let Some(inner) = &self.inner else {
            return;
        };
        inner
            .prepared_atom_evaluators
            .fetch_add(evaluators as u64, Ordering::Relaxed);
        inner
            .prepared_atom_nodes
            .fetch_add(nodes as u64, Ordering::Relaxed);
        inner
            .prepared_atom_add_nodes
            .fetch_add(add_nodes as u64, Ordering::Relaxed);
        inner
            .prepared_atom_mul_nodes
            .fetch_add(mul_nodes as u64, Ordering::Relaxed);
        inner
            .prepared_atom_powi_nodes
            .fetch_add(powi_nodes as u64, Ordering::Relaxed);
        inner
            .prepared_atom_pow_nodes
            .fetch_add(pow_nodes as u64, Ordering::Relaxed);
        inner
            .prepared_atom_builtin_nodes
            .fetch_add(builtin_nodes as u64, Ordering::Relaxed);
        inner
            .prepared_atom_custom_nodes
            .fetch_add(custom_nodes as u64, Ordering::Relaxed);
    }

    /// Returns a consistent-enough lock-free snapshot for diagnostics.
    pub fn snapshot(&self) -> BvpDirectJacobianTelemetrySnapshot {
        let Some(inner) = &self.inner else {
            return BvpDirectJacobianTelemetrySnapshot::default();
        };
        BvpDirectJacobianTelemetrySnapshot {
            calls: inner.calls.load(Ordering::Relaxed),
            work_items: inner.work_items.load(Ordering::Relaxed),
            elapsed: Duration::from_nanos(inner.elapsed_ns.load(Ordering::Relaxed)),
            errors: inner.errors.load(Ordering::Relaxed),
            argument_prepare_elapsed: Duration::from_nanos(
                inner.argument_prepare_ns.load(Ordering::Relaxed),
            ),
            evaluator_elapsed: Duration::from_nanos(inner.evaluator_ns.load(Ordering::Relaxed)),
            storage_write_elapsed: Duration::from_nanos(
                inner.storage_write_ns.load(Ordering::Relaxed),
            ),
            assembly_alloc_elapsed: Duration::from_nanos(
                inner.assembly_alloc_ns.load(Ordering::Relaxed),
            ),
            evaluator_calls: inner.evaluator_calls.load(Ordering::Relaxed),
            storage_writes: inner.storage_writes.load(Ordering::Relaxed),
            parallel_dispatches: inner.parallel_dispatches.load(Ordering::Relaxed),
            sequential_dispatches: inner.sequential_dispatches.load(Ordering::Relaxed),
            diagonal_dispatches: inner.diagonal_dispatches.load(Ordering::Relaxed),
            entry_dispatches: inner.entry_dispatches.load(Ordering::Relaxed),
            effective_task_count: inner.effective_task_count.load(Ordering::Relaxed),
            prepared_atom_evaluators: inner.prepared_atom_evaluators.load(Ordering::Relaxed),
            prepared_atom_nodes: inner.prepared_atom_nodes.load(Ordering::Relaxed),
            prepared_atom_add_nodes: inner.prepared_atom_add_nodes.load(Ordering::Relaxed),
            prepared_atom_mul_nodes: inner.prepared_atom_mul_nodes.load(Ordering::Relaxed),
            prepared_atom_powi_nodes: inner.prepared_atom_powi_nodes.load(Ordering::Relaxed),
            prepared_atom_pow_nodes: inner.prepared_atom_pow_nodes.load(Ordering::Relaxed),
            prepared_atom_builtin_nodes: inner.prepared_atom_builtin_nodes.load(Ordering::Relaxed),
            prepared_atom_custom_nodes: inner.prepared_atom_custom_nodes.load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lambdify_execution_policy_respects_work_threshold() {
        assert!(!BvpLambdifyExecutionPolicy::Sequential.should_parallel(10));
        assert!(BvpLambdifyExecutionPolicy::Parallel { min_work: 0 }.should_parallel(0));
        assert!(!BvpLambdifyExecutionPolicy::Parallel { min_work: 4 }.should_parallel(3));
        assert!(BvpLambdifyExecutionPolicy::Parallel { min_work: 4 }.should_parallel(4));

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("test Rayon pool should build");
        pool.install(|| {
            assert!(!BvpLambdifyExecutionPolicy::Auto { min_work: 0 }.should_parallel(15));
            assert!(BvpLambdifyExecutionPolicy::Auto { min_work: 0 }.should_parallel(16));
            assert!(!BvpLambdifyExecutionPolicy::Auto { min_work: 17 }.should_parallel(16));
            assert!(BvpLambdifyExecutionPolicy::Auto { min_work: 17 }.should_parallel(17));
            assert!(
                !BvpLambdifyExecutionPolicy::Auto {
                    min_work: usize::MAX
                }
                .should_parallel(usize::MAX - 1)
            );
            assert!(
                !BvpLambdifyExecutionPolicy::Auto { min_work: 0 }.should_parallel_with_tasks(16, 1)
            );
            assert!(
                BvpLambdifyExecutionPolicy::Auto { min_work: 0 }.should_parallel_with_tasks(16, 2)
            );
            assert_eq!(BvpLambdifyExecutionPolicy::auto_task_count(16), 2);
        });
    }

    #[test]
    fn direct_jacobian_telemetry_is_copy_free_and_thread_safe() {
        let telemetry = BvpDirectJacobianTelemetry::new();
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let telemetry = telemetry.clone();
                scope.spawn(move || {
                    telemetry.record_call(Duration::from_nanos(10), 3);
                });
            }
        });
        telemetry.record_error();

        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.calls, 4);
        assert_eq!(snapshot.work_items, 12);
        assert_eq!(snapshot.elapsed, Duration::from_nanos(40));
        assert_eq!(snapshot.errors, 1);
    }

    #[test]
    fn direct_jacobian_telemetry_off_has_no_runtime_measurements() {
        let telemetry = BvpDirectJacobianTelemetry::disabled();
        assert!(!telemetry.is_enabled());
        assert!(!telemetry.timing_enabled());
        telemetry.record_call(Duration::from_millis(1), 4);
        telemetry.record_error();
        telemetry.record_dispatch(true, true);
        telemetry.record_stage_breakdown(
            Duration::from_millis(1),
            Duration::from_millis(1),
            Duration::from_millis(1),
            Duration::from_millis(1),
            4,
            4,
        );
        assert_eq!(
            telemetry.snapshot(),
            BvpDirectJacobianTelemetrySnapshot::default()
        );
    }

    #[test]
    fn direct_jacobian_counters_skip_timing_but_keep_work_counts() {
        let telemetry = BvpDirectJacobianTelemetry::counters();
        assert!(telemetry.is_enabled());
        assert!(!telemetry.timing_enabled());
        telemetry.record_call(Duration::from_millis(1), 4);
        telemetry.record_stage_breakdown(
            Duration::from_millis(1),
            Duration::from_millis(1),
            Duration::from_millis(1),
            Duration::from_millis(1),
            4,
            4,
        );
        telemetry.record_dispatch(true, true);

        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.calls, 1);
        assert_eq!(snapshot.work_items, 4);
        assert_eq!(snapshot.evaluator_calls, 4);
        assert_eq!(snapshot.storage_writes, 4);
        assert_eq!(snapshot.parallel_dispatches, 1);
        assert_eq!(snapshot.elapsed, Duration::ZERO);
        assert_eq!(snapshot.evaluator_elapsed, Duration::ZERO);
        assert_eq!(snapshot.assembly_alloc_elapsed, Duration::ZERO);
    }

    #[test]
    fn direct_jacobian_telemetry_keeps_prepared_atom_shape_separate_from_calls() {
        let telemetry = BvpDirectJacobianTelemetry::counters();
        telemetry.record_prepared_atom_metrics(2, 10, 3, 4, 1, 1, 1, 0);
        telemetry.record_call(Duration::ZERO, 2);
        telemetry.record_call(Duration::ZERO, 2);

        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.calls, 2);
        assert_eq!(snapshot.evaluator_calls, 0);
        assert_eq!(snapshot.prepared_atom_evaluators, 2);
        assert_eq!(snapshot.prepared_atom_nodes, 10);
        assert_eq!(snapshot.prepared_atom_add_nodes, 3);
        assert_eq!(snapshot.prepared_atom_mul_nodes, 4);
        assert_eq!(snapshot.prepared_atom_powi_nodes, 1);
        assert_eq!(snapshot.prepared_atom_pow_nodes, 1);
        assert_eq!(snapshot.prepared_atom_builtin_nodes, 1);
        assert_eq!(snapshot.prepared_atom_custom_nodes, 0);
    }

    #[test]
    fn lambdify_telemetry_keeps_residual_and_jacobian_streams_separate() {
        let telemetry = BvpLambdifyTelemetry::new();
        telemetry.record_residual(Duration::from_nanos(7));
        telemetry.record_jacobian(Duration::from_nanos(11));

        assert_eq!(
            telemetry.snapshot(),
            BvpLambdifyTelemetrySnapshot {
                mode: BvpLambdifyTelemetryMode::Detailed,
                residual_calls: 1,
                residual_elapsed: Duration::from_nanos(7),
                jacobian_calls: 1,
                jacobian_elapsed: Duration::from_nanos(11),
                parallel_dispatches: 0,
                sequential_dispatches: 0,
            }
        );
    }

    #[test]
    fn lambdify_telemetry_off_has_no_runtime_measurements() {
        let telemetry = BvpLambdifyTelemetry::disabled();
        assert_eq!(telemetry.mode(), BvpLambdifyTelemetryMode::Off);
        assert!(telemetry.start_timing().is_none());
        telemetry.record_residual_sample(None);
        telemetry.record_jacobian_sample(None);
        assert_eq!(
            telemetry.snapshot(),
            BvpLambdifyTelemetrySnapshot::default()
        );
    }

    #[test]
    fn lambdify_telemetry_counters_skip_timing_but_keep_calls() {
        let telemetry = BvpLambdifyTelemetry::counters();
        assert_eq!(telemetry.mode(), BvpLambdifyTelemetryMode::Counters);
        assert!(telemetry.start_timing().is_none());
        telemetry.record_residual_sample(None);
        telemetry.record_jacobian_sample(None);
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.residual_calls, 1);
        assert_eq!(snapshot.jacobian_calls, 1);
        assert_eq!(snapshot.residual_elapsed, Duration::ZERO);
        assert_eq!(snapshot.jacobian_elapsed, Duration::ZERO);
    }

    #[test]
    fn lambdify_telemetry_detailed_records_sampled_elapsed_time() {
        let telemetry = BvpLambdifyTelemetry::detailed();
        let residual_started = telemetry.start_timing();
        assert!(residual_started.is_some());
        // The production callback can legitimately complete within one clock
        // tick and report zero nanoseconds. This test is about the timed
        // sampling path, so create a deterministic non-zero interval instead
        // of treating a zero-resolution instant as a telemetry failure.
        std::thread::sleep(Duration::from_millis(1));
        telemetry.record_residual_sample(residual_started);
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.residual_calls, 1);
        assert!(snapshot.residual_elapsed > Duration::ZERO);
    }

    #[test]
    fn atom_discretization_snapshot_has_typed_stage_contract() {
        let snapshot = BvpAtomDiscretizationTelemetrySnapshot {
            boundary_conditions: Duration::from_nanos(1),
            discretization: Duration::from_nanos(2),
            boundary_application: Duration::from_nanos(1),
            flat_list: Duration::from_nanos(1),
            consistency: Duration::from_nanos(1),
            bounds_and_tolerances: Duration::from_nanos(1),
            total: Duration::from_nanos(3),
        };

        assert!(snapshot.stages_fit_total());
        assert_eq!(snapshot.total, Duration::from_nanos(3));
    }
}
