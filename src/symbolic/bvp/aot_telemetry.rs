//! Typed, low-overhead telemetry for the BVP AOT lifecycle.
//!
//! Cold preparation and warm callback work intentionally have separate fields.
//! A compiler build must never be mistaken for residual/Jacobian execution,
//! and the disabled mode must not add a timer, map or lock to the hot path.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Amount of AOT telemetry collected by one prepared runtime.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BvpAotTelemetryMode {
    /// No allocation, timing or atomic updates on the AOT runtime path.
    #[default]
    Off,
    /// Collect counters but do not read a clock for stage/callback timings.
    Counters,
    /// Collect counters and elapsed wall-clock timings.
    Detailed,
}

/// Cold AOT stages with stable names and no string-keyed storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BvpAotColdStage {
    Validation,
    AtomPreparation,
    JacobianPreparation,
    Lowering,
    Optimization,
    SourceEmission,
    Materialization,
    Build,
    Link,
    Publication,
}

/// Lifecycle transitions emitted by artifact management and linking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BvpAotLifecycleEvent {
    Planned,
    SourceEmitted,
    Materialized,
    BuildStarted,
    BuildSucceeded,
    BuildFailed,
    LinkFailed,
    Published,
    Linked,
    RuntimeReady,
    CacheHit,
    CacheMiss,
    Retry,
    Quarantined,
}

/// Symbolic frontend that owns a prepared AOT payload.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BvpAotFrontend {
    #[default]
    Unknown,
    ExprLegacy,
    AtomView,
}

impl BvpAotFrontend {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::ExprLegacy => "expr-legacy",
            Self::AtomView => "atom-view",
        }
    }
}

/// Matrix layout carried by the prepared AOT callback contract.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BvpAotMatrixLayout {
    #[default]
    Unknown,
    Dense,
    SparseCsc,
    Banded,
    BandedCompact,
}

impl BvpAotMatrixLayout {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Dense => "dense",
            Self::SparseCsc => "sparse-csc",
            Self::Banded => "banded",
            Self::BandedCompact => "banded-compact",
        }
    }
}

/// Coarse evaluator policy attached to a prepared callback route.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BvpAotEvaluatorPolicy {
    #[default]
    Unknown,
    Sequential,
    Parallel,
    Auto,
}

impl BvpAotEvaluatorPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Sequential => "sequential",
            Self::Parallel => "parallel",
            Self::Auto => "auto",
        }
    }
}

/// Chunking identity recorded without storing strategy strings.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BvpAotChunking {
    #[default]
    Unknown,
    Whole,
    Chunked,
}

impl BvpAotChunking {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Whole => "whole",
            Self::Chunked => "chunked",
        }
    }
}

/// Typed route metadata shared by cold lifecycle and warm callback reports.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BvpAotTelemetryIdentity {
    pub frontend: BvpAotFrontend,
    pub matrix_layout: BvpAotMatrixLayout,
    pub evaluator_policy: BvpAotEvaluatorPolicy,
    pub residual_chunking: BvpAotChunking,
    pub jacobian_chunking: BvpAotChunking,
}

#[derive(Debug, Default)]
struct Inner {
    validation_ns: AtomicU64,
    atom_preparation_ns: AtomicU64,
    jacobian_preparation_ns: AtomicU64,
    lowering_ns: AtomicU64,
    optimization_ns: AtomicU64,
    source_emission_ns: AtomicU64,
    materialization_ns: AtomicU64,
    build_ns: AtomicU64,
    link_ns: AtomicU64,
    publication_ns: AtomicU64,
    residual_calls: AtomicU64,
    residual_ns: AtomicU64,
    jacobian_calls: AtomicU64,
    jacobian_ns: AtomicU64,
    residual_chunks: AtomicU64,
    jacobian_chunks: AtomicU64,
    parameter_binds: AtomicU64,
    conversions: AtomicU64,
    copies: AtomicU64,
    copy_bytes: AtomicU64,
    allocation_events: AtomicU64,
    allocation_bytes: AtomicU64,
    worker_threads: AtomicU64,
    worker_batches: AtomicU64,
    errors: AtomicU64,
    build_attempts: AtomicU64,
    build_failures: AtomicU64,
    link_failures: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    retries: AtomicU64,
    quarantines: AtomicU64,
    last_lifecycle_event: AtomicU64,
}

/// Shared AOT telemetry handle owned by a prepared AtomView/ExprLegacy plan.
#[derive(Clone, Debug)]
pub struct BvpAotTelemetry {
    mode: BvpAotTelemetryMode,
    inner: Option<Arc<Inner>>,
    identity: BvpAotTelemetryIdentity,
}

impl Default for BvpAotTelemetry {
    fn default() -> Self {
        Self::disabled()
    }
}

/// Immutable typed AOT telemetry snapshot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BvpAotTelemetrySnapshot {
    pub mode: BvpAotTelemetryMode,
    pub identity: BvpAotTelemetryIdentity,
    pub validation: Duration,
    pub atom_preparation: Duration,
    pub jacobian_preparation: Duration,
    pub lowering: Duration,
    pub optimization: Duration,
    pub source_emission: Duration,
    pub materialization: Duration,
    pub build: Duration,
    pub link: Duration,
    pub publication: Duration,
    pub residual_calls: u64,
    pub residual_elapsed: Duration,
    pub jacobian_calls: u64,
    pub jacobian_elapsed: Duration,
    pub residual_chunks: u64,
    pub jacobian_chunks: u64,
    pub parameter_binds: u64,
    pub conversions: u64,
    pub copies: u64,
    pub copy_bytes: u64,
    pub allocation_events: u64,
    pub allocation_bytes: u64,
    pub worker_threads: u64,
    pub worker_batches: u64,
    pub errors: u64,
    pub build_attempts: u64,
    pub build_failures: u64,
    pub link_failures: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub retries: u64,
    pub quarantines: u64,
    /// Last typed lifecycle transition, if telemetry is enabled and at least
    /// one transition was recorded.  This is a scalar code internally; the
    /// enum is reconstructed only when a snapshot is requested.
    pub last_lifecycle_event: Option<BvpAotLifecycleEvent>,
}

impl BvpAotTelemetrySnapshot {
    /// Canonical symbolic/preparation bucket used by AOT story reports.
    ///
    /// The historical reports called this stage `symbolic_ms`.  AtomView has
    /// more precise sub-stages, but this aggregate keeps old and new rows
    /// directly comparable without parsing diagnostic maps.
    pub fn symbolic_preparation(&self) -> Duration {
        self.validation + self.atom_preparation + self.jacobian_preparation
    }

    /// Canonical generated-fixture bucket: lowering, source emission and
    /// artifact materialization before an external compiler is run.
    pub fn fixture_generation(&self) -> Duration {
        self.lowering + self.source_emission + self.materialization
    }

    /// Canonical compiler bucket corresponding to the historical
    /// `compile_ms`/`build_ms` column.
    pub fn compilation(&self) -> Duration {
        self.build
    }

    /// Canonical linker/publication bucket.  Runtime registration is kept
    /// separate from compilation so toolchain and loader costs are visible.
    pub fn linking(&self) -> Duration {
        self.link + self.publication
    }

    /// Publishes the typed snapshot through the historical string-keyed
    /// diagnostics boundary.
    ///
    /// The compatibility map is intentionally built only when a caller asks
    /// for diagnostics.  The AOT callback hot path remains typed and does not
    /// allocate or update a `HashMap`.
    pub fn append_compatibility_diagnostics(&self, diagnostics: &mut HashMap<String, String>) {
        diagnostics.insert(
            "generated.aot.identity.frontend".to_string(),
            self.identity.frontend.as_str().to_string(),
        );
        diagnostics.insert(
            "generated.aot.identity.matrix_layout".to_string(),
            self.identity.matrix_layout.as_str().to_string(),
        );
        diagnostics.insert(
            "generated.aot.identity.evaluator_policy".to_string(),
            self.identity.evaluator_policy.as_str().to_string(),
        );
        diagnostics.insert(
            "generated.aot.identity.residual_chunking".to_string(),
            self.identity.residual_chunking.as_str().to_string(),
        );
        diagnostics.insert(
            "generated.aot.identity.jacobian_chunking".to_string(),
            self.identity.jacobian_chunking.as_str().to_string(),
        );
        for (key, value) in [
            (
                "symbolic_preparation_ms",
                self.symbolic_preparation().as_secs_f64() * 1_000.0,
            ),
            (
                "fixture_generation_ms",
                self.fixture_generation().as_secs_f64() * 1_000.0,
            ),
            ("compile_ms", self.compilation().as_secs_f64() * 1_000.0),
            ("link_ms", self.linking().as_secs_f64() * 1_000.0),
            ("validation_ms", self.validation.as_secs_f64() * 1_000.0),
            (
                "atom_preparation_ms",
                self.atom_preparation.as_secs_f64() * 1_000.0,
            ),
            (
                "jacobian_preparation_ms",
                self.jacobian_preparation.as_secs_f64() * 1_000.0,
            ),
            ("lowering_ms", self.lowering.as_secs_f64() * 1_000.0),
            (
                "source_emission_ms",
                self.source_emission.as_secs_f64() * 1_000.0,
            ),
            (
                "materialization_ms",
                self.materialization.as_secs_f64() * 1_000.0,
            ),
            ("build_ms", self.build.as_secs_f64() * 1_000.0),
            ("typed_link_ms", self.link.as_secs_f64() * 1_000.0),
            ("publication_ms", self.publication.as_secs_f64() * 1_000.0),
        ] {
            diagnostics.insert(format!("generated.aot.typed.{key}"), format!("{value:.6}"));
        }
        for (key, value) in [
            ("residual_calls", self.residual_calls),
            ("jacobian_calls", self.jacobian_calls),
            ("residual_chunks", self.residual_chunks),
            ("jacobian_chunks", self.jacobian_chunks),
            ("errors", self.errors),
            ("build_attempts", self.build_attempts),
            ("build_failures", self.build_failures),
            ("link_failures", self.link_failures),
            ("copies", self.copies),
            ("copy_bytes", self.copy_bytes),
            ("allocation_events", self.allocation_events),
            ("allocation_bytes", self.allocation_bytes),
            ("worker_threads", self.worker_threads),
            ("worker_batches", self.worker_batches),
        ] {
            diagnostics.insert(format!("generated.aot.typed.{key}"), value.to_string());
        }
        if let Some(event) = self.last_lifecycle_event {
            diagnostics.insert(
                "generated.aot.typed.last_lifecycle_event".to_string(),
                format!("{event:?}"),
            );
        }
    }
}

impl BvpAotTelemetry {
    pub fn disabled() -> Self {
        Self {
            mode: BvpAotTelemetryMode::Off,
            inner: None,
            identity: BvpAotTelemetryIdentity::default(),
        }
    }

    pub fn counters() -> Self {
        Self::with_mode(BvpAotTelemetryMode::Counters)
    }

    pub fn detailed() -> Self {
        Self::with_mode(BvpAotTelemetryMode::Detailed)
    }

    pub fn with_mode(mode: BvpAotTelemetryMode) -> Self {
        Self {
            mode,
            inner: (mode != BvpAotTelemetryMode::Off).then(|| Arc::new(Inner::default())),
            identity: BvpAotTelemetryIdentity::default(),
        }
    }

    /// Attaches route metadata during cold preparation.
    ///
    /// The identity is copied and never stored in the atomic hot-path state,
    /// so it adds no callback synchronization or allocation.
    pub fn with_identity(mut self, identity: BvpAotTelemetryIdentity) -> Self {
        self.identity = identity;
        self
    }

    /// Returns the typed route metadata attached to this stream.
    pub const fn identity(&self) -> BvpAotTelemetryIdentity {
        self.identity
    }

    pub const fn mode(&self) -> BvpAotTelemetryMode {
        self.mode
    }

    /// Starts a timer only for the detailed mode.
    #[inline]
    pub fn start_timing(&self) -> Option<Instant> {
        (self.mode == BvpAotTelemetryMode::Detailed).then(Instant::now)
    }

    /// Records a completed cold stage. Counters mode deliberately skips time.
    #[inline]
    pub fn record_cold_stage(&self, stage: BvpAotColdStage, started: Option<Instant>) {
        let Some(inner) = &self.inner else { return };
        if self.mode != BvpAotTelemetryMode::Detailed {
            return;
        }
        let elapsed = started
            .map(|value| duration_ns(value.elapsed()))
            .unwrap_or(0);
        cold_stage_counter(inner, stage).fetch_add(elapsed, Ordering::Relaxed);
    }

    /// Records an already measured cold-stage duration.
    ///
    /// Codegen lowering already exposes a typed stage breakdown, so taking a
    /// second `Instant` around each sub-stage would both duplicate work and
    /// blur the boundary between symbolic preparation and IR lowering.
    #[inline]
    pub fn record_cold_stage_duration(&self, stage: BvpAotColdStage, elapsed: Duration) {
        let Some(inner) = &self.inner else { return };
        if self.mode != BvpAotTelemetryMode::Detailed {
            return;
        }
        cold_stage_counter(inner, stage).fetch_add(duration_ns(elapsed), Ordering::Relaxed);
    }

    /// Records one warm residual callback and its chunk count.
    #[inline]
    pub fn record_residual(&self, started: Option<Instant>, chunks: usize) {
        let Some(inner) = &self.inner else { return };
        inner.residual_calls.fetch_add(1, Ordering::Relaxed);
        inner
            .residual_chunks
            .fetch_add(chunks as u64, Ordering::Relaxed);
        if self.mode == BvpAotTelemetryMode::Detailed {
            let elapsed = started
                .map(|value| duration_ns(value.elapsed()))
                .unwrap_or(0);
            inner.residual_ns.fetch_add(elapsed, Ordering::Relaxed);
        }
    }

    /// Records one warm Jacobian callback and its chunk count.
    #[inline]
    pub fn record_jacobian(&self, started: Option<Instant>, chunks: usize) {
        let Some(inner) = &self.inner else { return };
        inner.jacobian_calls.fetch_add(1, Ordering::Relaxed);
        inner
            .jacobian_chunks
            .fetch_add(chunks as u64, Ordering::Relaxed);
        if self.mode == BvpAotTelemetryMode::Detailed {
            let elapsed = started
                .map(|value| duration_ns(value.elapsed()))
                .unwrap_or(0);
            inner.jacobian_ns.fetch_add(elapsed, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn record_parameter_bind(&self) {
        self.record_counter(|inner| &inner.parameter_binds);
    }

    #[inline]
    pub fn record_conversion(&self) {
        self.record_counter(|inner| &inner.conversions);
    }

    #[inline]
    pub fn record_copy(&self) {
        self.record_counter(|inner| &inner.copies);
    }

    /// Records a copy with a known byte count. The count is optional because
    /// compiler/FFI boundaries sometimes expose only a logical copy event.
    #[inline]
    pub fn record_copy_bytes(&self, bytes: usize) {
        let Some(inner) = &self.inner else { return };
        inner.copies.fetch_add(1, Ordering::Relaxed);
        inner
            .copy_bytes
            .fetch_add(bytes.min(u64::MAX as usize) as u64, Ordering::Relaxed);
    }

    /// Records an owned allocation outside the callback hot path. This is a
    /// counter, not a global allocator hook, so disabled telemetry remains
    /// allocation-free and callers can instrument only known buffers.
    #[inline]
    pub fn record_allocation(&self, bytes: usize) {
        let Some(inner) = &self.inner else { return };
        inner.allocation_events.fetch_add(1, Ordering::Relaxed);
        inner
            .allocation_bytes
            .fetch_add(bytes.min(u64::MAX as usize) as u64, Ordering::Relaxed);
    }

    /// Aggregates work performed by worker threads without a per-worker map.
    #[inline]
    pub fn record_worker_batch(&self, workers: usize, batches: usize) {
        let Some(inner) = &self.inner else { return };
        inner
            .worker_threads
            .fetch_max(workers.max(1) as u64, Ordering::Relaxed);
        inner
            .worker_batches
            .fetch_add(batches as u64, Ordering::Relaxed);
    }

    #[inline]
    pub fn record_error(&self) {
        self.record_counter(|inner| &inner.errors);
    }

    /// Records a typed artifact lifecycle event without storing strings.
    #[inline]
    pub fn record_lifecycle(&self, event: BvpAotLifecycleEvent) {
        let Some(inner) = &self.inner else { return };
        inner
            .last_lifecycle_event
            .store(lifecycle_event_code(event), Ordering::Relaxed);
        match event {
            BvpAotLifecycleEvent::BuildStarted => {
                inner.build_attempts.fetch_add(1, Ordering::Relaxed);
            }
            BvpAotLifecycleEvent::BuildFailed => {
                inner.build_failures.fetch_add(1, Ordering::Relaxed);
            }
            BvpAotLifecycleEvent::LinkFailed => {
                inner.link_failures.fetch_add(1, Ordering::Relaxed);
            }
            BvpAotLifecycleEvent::CacheHit => {
                inner.cache_hits.fetch_add(1, Ordering::Relaxed);
            }
            BvpAotLifecycleEvent::CacheMiss => {
                inner.cache_misses.fetch_add(1, Ordering::Relaxed);
            }
            BvpAotLifecycleEvent::Retry => {
                inner.retries.fetch_add(1, Ordering::Relaxed);
            }
            BvpAotLifecycleEvent::Quarantined => {
                inner.quarantines.fetch_add(1, Ordering::Relaxed);
            }
            BvpAotLifecycleEvent::Planned
            | BvpAotLifecycleEvent::SourceEmitted
            | BvpAotLifecycleEvent::Materialized
            | BvpAotLifecycleEvent::BuildSucceeded
            | BvpAotLifecycleEvent::Published
            | BvpAotLifecycleEvent::Linked
            | BvpAotLifecycleEvent::RuntimeReady => {}
        }
    }

    /// Records and emits one cold lifecycle event without using a string map.
    ///
    /// The logger is consulted only for lifecycle transitions, never for warm
    /// residual/Jacobian callbacks. The artifact key is kept at this boundary
    /// because it identifies the external build resource rather than a hot
    /// numeric operation.
    pub fn record_lifecycle_with_log(&self, event: BvpAotLifecycleEvent, artifact_key: &str) {
        if self.mode == BvpAotTelemetryMode::Off {
            return;
        }
        self.record_lifecycle(event);
        log::debug!(
            target: "rustedscithe::bvp::aot",
            "AOT artifact lifecycle transition: event={event:?} artifact_key={artifact_key}"
        );
    }

    pub fn snapshot(&self) -> BvpAotTelemetrySnapshot {
        let Some(inner) = &self.inner else {
            // Route identity is configuration metadata, not collected
            // telemetry. Preserve it even when counters and timings are off.
            return BvpAotTelemetrySnapshot {
                mode: self.mode,
                identity: self.identity,
                ..BvpAotTelemetrySnapshot::default()
            };
        };
        BvpAotTelemetrySnapshot {
            mode: self.mode,
            identity: self.identity,
            validation: duration(inner.validation_ns.load(Ordering::Relaxed)),
            atom_preparation: duration(inner.atom_preparation_ns.load(Ordering::Relaxed)),
            jacobian_preparation: duration(inner.jacobian_preparation_ns.load(Ordering::Relaxed)),
            lowering: duration(inner.lowering_ns.load(Ordering::Relaxed)),
            optimization: duration(inner.optimization_ns.load(Ordering::Relaxed)),
            source_emission: duration(inner.source_emission_ns.load(Ordering::Relaxed)),
            materialization: duration(inner.materialization_ns.load(Ordering::Relaxed)),
            build: duration(inner.build_ns.load(Ordering::Relaxed)),
            link: duration(inner.link_ns.load(Ordering::Relaxed)),
            publication: duration(inner.publication_ns.load(Ordering::Relaxed)),
            residual_calls: inner.residual_calls.load(Ordering::Relaxed),
            residual_elapsed: duration(inner.residual_ns.load(Ordering::Relaxed)),
            jacobian_calls: inner.jacobian_calls.load(Ordering::Relaxed),
            jacobian_elapsed: duration(inner.jacobian_ns.load(Ordering::Relaxed)),
            residual_chunks: inner.residual_chunks.load(Ordering::Relaxed),
            jacobian_chunks: inner.jacobian_chunks.load(Ordering::Relaxed),
            parameter_binds: inner.parameter_binds.load(Ordering::Relaxed),
            conversions: inner.conversions.load(Ordering::Relaxed),
            copies: inner.copies.load(Ordering::Relaxed),
            copy_bytes: inner.copy_bytes.load(Ordering::Relaxed),
            allocation_events: inner.allocation_events.load(Ordering::Relaxed),
            allocation_bytes: inner.allocation_bytes.load(Ordering::Relaxed),
            worker_threads: inner.worker_threads.load(Ordering::Relaxed),
            worker_batches: inner.worker_batches.load(Ordering::Relaxed),
            errors: inner.errors.load(Ordering::Relaxed),
            build_attempts: inner.build_attempts.load(Ordering::Relaxed),
            build_failures: inner.build_failures.load(Ordering::Relaxed),
            link_failures: inner.link_failures.load(Ordering::Relaxed),
            cache_hits: inner.cache_hits.load(Ordering::Relaxed),
            cache_misses: inner.cache_misses.load(Ordering::Relaxed),
            retries: inner.retries.load(Ordering::Relaxed),
            quarantines: inner.quarantines.load(Ordering::Relaxed),
            last_lifecycle_event: lifecycle_event_from_code(
                inner.last_lifecycle_event.load(Ordering::Relaxed),
            ),
        }
    }

    #[inline]
    fn record_counter(&self, counter: impl FnOnce(&Inner) -> &AtomicU64) {
        if let Some(inner) = &self.inner {
            counter(inner).fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn duration_ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

fn duration(value: u64) -> Duration {
    Duration::from_nanos(value)
}

#[inline]
fn lifecycle_event_code(event: BvpAotLifecycleEvent) -> u64 {
    match event {
        BvpAotLifecycleEvent::Planned => 1,
        BvpAotLifecycleEvent::SourceEmitted => 2,
        BvpAotLifecycleEvent::Materialized => 3,
        BvpAotLifecycleEvent::BuildStarted => 4,
        BvpAotLifecycleEvent::BuildSucceeded => 5,
        BvpAotLifecycleEvent::BuildFailed => 6,
        BvpAotLifecycleEvent::LinkFailed => 7,
        BvpAotLifecycleEvent::Published => 8,
        BvpAotLifecycleEvent::Linked => 9,
        BvpAotLifecycleEvent::RuntimeReady => 10,
        BvpAotLifecycleEvent::CacheHit => 11,
        BvpAotLifecycleEvent::CacheMiss => 12,
        BvpAotLifecycleEvent::Retry => 13,
        BvpAotLifecycleEvent::Quarantined => 14,
    }
}

#[inline]
fn lifecycle_event_from_code(code: u64) -> Option<BvpAotLifecycleEvent> {
    Some(match code {
        1 => BvpAotLifecycleEvent::Planned,
        2 => BvpAotLifecycleEvent::SourceEmitted,
        3 => BvpAotLifecycleEvent::Materialized,
        4 => BvpAotLifecycleEvent::BuildStarted,
        5 => BvpAotLifecycleEvent::BuildSucceeded,
        6 => BvpAotLifecycleEvent::BuildFailed,
        7 => BvpAotLifecycleEvent::LinkFailed,
        8 => BvpAotLifecycleEvent::Published,
        9 => BvpAotLifecycleEvent::Linked,
        10 => BvpAotLifecycleEvent::RuntimeReady,
        11 => BvpAotLifecycleEvent::CacheHit,
        12 => BvpAotLifecycleEvent::CacheMiss,
        13 => BvpAotLifecycleEvent::Retry,
        14 => BvpAotLifecycleEvent::Quarantined,
        _ => return None,
    })
}

fn cold_stage_counter(inner: &Inner, stage: BvpAotColdStage) -> &AtomicU64 {
    match stage {
        BvpAotColdStage::Validation => &inner.validation_ns,
        BvpAotColdStage::AtomPreparation => &inner.atom_preparation_ns,
        BvpAotColdStage::JacobianPreparation => &inner.jacobian_preparation_ns,
        BvpAotColdStage::Lowering => &inner.lowering_ns,
        BvpAotColdStage::Optimization => &inner.optimization_ns,
        BvpAotColdStage::SourceEmission => &inner.source_emission_ns,
        BvpAotColdStage::Materialization => &inner.materialization_ns,
        BvpAotColdStage::Build => &inner.build_ns,
        BvpAotColdStage::Link => &inner.link_ns,
        BvpAotColdStage::Publication => &inner.publication_ns,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_mode_has_zero_snapshot_and_no_timing() {
        let telemetry = BvpAotTelemetry::disabled();
        assert_eq!(telemetry.mode(), BvpAotTelemetryMode::Off);
        assert!(telemetry.start_timing().is_none());
        telemetry.record_lifecycle(BvpAotLifecycleEvent::BuildStarted);
        telemetry.record_error();
        assert_eq!(telemetry.snapshot(), BvpAotTelemetrySnapshot::default());
    }

    #[test]
    fn route_identity_is_typed_and_only_rendered_at_compatibility_boundary() {
        let identity = BvpAotTelemetryIdentity {
            frontend: BvpAotFrontend::AtomView,
            matrix_layout: BvpAotMatrixLayout::BandedCompact,
            evaluator_policy: BvpAotEvaluatorPolicy::Auto,
            residual_chunking: BvpAotChunking::Whole,
            jacobian_chunking: BvpAotChunking::Chunked,
        };
        let snapshot = BvpAotTelemetry::counters()
            .with_identity(identity)
            .snapshot();

        assert_eq!(snapshot.identity, identity);

        let mut diagnostics = HashMap::new();
        snapshot.append_compatibility_diagnostics(&mut diagnostics);
        assert_eq!(
            diagnostics.get("generated.aot.identity.frontend"),
            Some(&"atom-view".to_string())
        );
        assert_eq!(
            diagnostics.get("generated.aot.identity.matrix_layout"),
            Some(&"banded-compact".to_string())
        );
        assert_eq!(
            diagnostics.get("generated.aot.identity.evaluator_policy"),
            Some(&"auto".to_string())
        );
        assert_eq!(
            diagnostics.get("generated.aot.identity.jacobian_chunking"),
            Some(&"chunked".to_string())
        );
    }

    #[test]
    fn counters_keep_lifecycle_and_callback_semantics_without_timers() {
        let telemetry = BvpAotTelemetry::counters();
        telemetry.record_lifecycle(BvpAotLifecycleEvent::BuildStarted);
        telemetry.record_lifecycle(BvpAotLifecycleEvent::BuildFailed);
        telemetry.record_lifecycle(BvpAotLifecycleEvent::LinkFailed);
        telemetry.record_lifecycle(BvpAotLifecycleEvent::Quarantined);
        telemetry.record_lifecycle(BvpAotLifecycleEvent::CacheMiss);
        telemetry.record_residual(None, 2);
        telemetry.record_jacobian(None, 3);
        telemetry.record_parameter_bind();
        let snapshot = telemetry.snapshot();

        assert_eq!(snapshot.residual_calls, 1);
        assert_eq!(snapshot.residual_chunks, 2);
        assert_eq!(snapshot.jacobian_calls, 1);
        assert_eq!(snapshot.jacobian_chunks, 3);
        assert_eq!(snapshot.build_attempts, 1);
        assert_eq!(snapshot.build_failures, 1);
        assert_eq!(snapshot.link_failures, 1);
        assert_eq!(snapshot.quarantines, 1);
        assert_eq!(snapshot.cache_misses, 1);
        assert_eq!(snapshot.parameter_binds, 1);
        assert_eq!(
            snapshot.last_lifecycle_event,
            Some(BvpAotLifecycleEvent::CacheMiss)
        );
        assert_eq!(snapshot.residual_elapsed, Duration::ZERO);
        assert_eq!(snapshot.jacobian_elapsed, Duration::ZERO);
    }

    #[test]
    fn worker_and_buffer_metrics_are_fixed_field_and_aggregate_across_calls() {
        let telemetry = BvpAotTelemetry::counters();
        telemetry.record_worker_batch(4, 8);
        telemetry.record_worker_batch(2, 3);
        telemetry.record_copy_bytes(128);
        telemetry.record_allocation(256);
        let snapshot = telemetry.snapshot();

        assert_eq!(snapshot.worker_threads, 4);
        assert_eq!(snapshot.worker_batches, 11);
        assert_eq!(snapshot.copies, 1);
        assert_eq!(snapshot.copy_bytes, 128);
        assert_eq!(snapshot.allocation_events, 1);
        assert_eq!(snapshot.allocation_bytes, 256);
    }

    #[test]
    fn detailed_mode_separates_cold_and_warm_streams() {
        let telemetry = BvpAotTelemetry::detailed();
        let cold = telemetry.start_timing();
        std::thread::sleep(Duration::from_micros(50));
        telemetry.record_cold_stage(BvpAotColdStage::Lowering, cold);
        let residual = telemetry.start_timing();
        std::thread::sleep(Duration::from_micros(50));
        telemetry.record_residual(residual, 4);
        let jacobian = telemetry.start_timing();
        std::thread::sleep(Duration::from_micros(50));
        telemetry.record_jacobian(jacobian, 5);
        let snapshot = telemetry.snapshot();

        assert_eq!(snapshot.residual_calls, 1);
        assert_eq!(snapshot.jacobian_calls, 1);
        assert_eq!(snapshot.residual_chunks, 4);
        assert_eq!(snapshot.jacobian_chunks, 5);
        assert!(snapshot.lowering > Duration::ZERO);
        assert!(snapshot.residual_elapsed > Duration::ZERO);
        assert!(snapshot.jacobian_elapsed > Duration::ZERO);
    }

    #[test]
    fn canonical_story_buckets_preserve_cold_stage_boundaries() {
        let telemetry = BvpAotTelemetry::detailed();
        telemetry.record_cold_stage_duration(BvpAotColdStage::Validation, Duration::from_millis(1));
        telemetry
            .record_cold_stage_duration(BvpAotColdStage::AtomPreparation, Duration::from_millis(2));
        telemetry.record_cold_stage_duration(
            BvpAotColdStage::JacobianPreparation,
            Duration::from_millis(3),
        );
        telemetry.record_cold_stage_duration(BvpAotColdStage::Lowering, Duration::from_millis(4));
        telemetry
            .record_cold_stage_duration(BvpAotColdStage::SourceEmission, Duration::from_millis(5));
        telemetry
            .record_cold_stage_duration(BvpAotColdStage::Materialization, Duration::from_millis(6));
        telemetry.record_cold_stage_duration(BvpAotColdStage::Build, Duration::from_millis(7));
        telemetry.record_cold_stage_duration(BvpAotColdStage::Link, Duration::from_millis(8));
        telemetry
            .record_cold_stage_duration(BvpAotColdStage::Publication, Duration::from_millis(9));

        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.symbolic_preparation(), Duration::from_millis(6));
        assert_eq!(snapshot.fixture_generation(), Duration::from_millis(15));
        assert_eq!(snapshot.compilation(), Duration::from_millis(7));
        assert_eq!(snapshot.linking(), Duration::from_millis(17));
    }
}
