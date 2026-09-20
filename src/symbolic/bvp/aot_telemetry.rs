//! Typed, low-overhead telemetry for the BVP AOT lifecycle.
//!
//! Cold preparation and warm callback work intentionally have separate fields.
//! A compiler build must never be mistaken for residual/Jacobian execution,
//! and the disabled mode must not add a timer, map or lock to the hot path.

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
    Published,
    Linked,
    RuntimeReady,
    CacheHit,
    CacheMiss,
    Retry,
    Quarantined,
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
    errors: AtomicU64,
    build_attempts: AtomicU64,
    build_failures: AtomicU64,
    link_failures: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    retries: AtomicU64,
    quarantines: AtomicU64,
}

/// Shared AOT telemetry handle owned by a prepared AtomView/ExprLegacy plan.
#[derive(Clone, Debug)]
pub struct BvpAotTelemetry {
    mode: BvpAotTelemetryMode,
    inner: Option<Arc<Inner>>,
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
    pub errors: u64,
    pub build_attempts: u64,
    pub build_failures: u64,
    pub link_failures: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub retries: u64,
    pub quarantines: u64,
}

impl BvpAotTelemetry {
    pub fn disabled() -> Self {
        Self {
            mode: BvpAotTelemetryMode::Off,
            inner: None,
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
        }
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

    #[inline]
    pub fn record_error(&self) {
        self.record_counter(|inner| &inner.errors);
    }

    /// Records a typed artifact lifecycle event without storing strings.
    #[inline]
    pub fn record_lifecycle(&self, event: BvpAotLifecycleEvent) {
        let Some(inner) = &self.inner else { return };
        match event {
            BvpAotLifecycleEvent::BuildStarted => {
                inner.build_attempts.fetch_add(1, Ordering::Relaxed);
            }
            BvpAotLifecycleEvent::BuildFailed => {
                inner.build_failures.fetch_add(1, Ordering::Relaxed);
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
        self.record_lifecycle(event);
        log::debug!(
            target: "rustedscithe::bvp::aot",
            "AOT artifact lifecycle transition: event={event:?} artifact_key={artifact_key}"
        );
    }

    pub fn snapshot(&self) -> BvpAotTelemetrySnapshot {
        let Some(inner) = &self.inner else {
            return BvpAotTelemetrySnapshot::default();
        };
        BvpAotTelemetrySnapshot {
            mode: self.mode,
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
            errors: inner.errors.load(Ordering::Relaxed),
            build_attempts: inner.build_attempts.load(Ordering::Relaxed),
            build_failures: inner.build_failures.load(Ordering::Relaxed),
            link_failures: inner.link_failures.load(Ordering::Relaxed),
            cache_hits: inner.cache_hits.load(Ordering::Relaxed),
            cache_misses: inner.cache_misses.load(Ordering::Relaxed),
            retries: inner.retries.load(Ordering::Relaxed),
            quarantines: inner.quarantines.load(Ordering::Relaxed),
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
    fn counters_keep_lifecycle_and_callback_semantics_without_timers() {
        let telemetry = BvpAotTelemetry::counters();
        telemetry.record_lifecycle(BvpAotLifecycleEvent::BuildStarted);
        telemetry.record_lifecycle(BvpAotLifecycleEvent::BuildFailed);
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
        assert_eq!(snapshot.cache_misses, 1);
        assert_eq!(snapshot.parameter_binds, 1);
        assert_eq!(snapshot.residual_elapsed, Duration::ZERO);
        assert_eq!(snapshot.jacobian_elapsed, Duration::ZERO);
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
}
