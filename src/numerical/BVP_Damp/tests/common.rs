#![allow(dead_code)]

use nalgebra::DMatrix;
use std::env;
use std::thread;
use std::time::Duration;

/// Test-only route labels used to keep the migration corpus explicit.
///
/// These labels describe the symbolic/runtime route under test, not a solver
/// algorithm. The old ExprLegacy route remains the numerical regression oracle;
/// AtomView and parity tests are added alongside it during the migration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SymbolicTestRoute {
    ExprLegacy,
    AtomView,
    Parity,
}

impl SymbolicTestRoute {
    pub(crate) const fn symbolic_frontend(self) -> &'static str {
        match self {
            Self::ExprLegacy => "ExprLegacy",
            Self::AtomView => "AtomView",
            Self::Parity => "ExprLegacy+AtomView",
        }
    }

    pub(crate) const fn runtime_route(self) -> &'static str {
        match self {
            Self::ExprLegacy => "ExprLegacy+legacy-callback",
            Self::AtomView => "AtomView+direct-no-Mutex",
            Self::Parity => "comparison-only",
        }
    }
}

pub(crate) const DEFAULT_COLD_STORY_COOLDOWN_MS: u64 = 5_000;
pub(crate) const DEFAULT_WARM_STORY_COOLDOWN_MS: u64 = 1_000;

/// One reproducible protocol for cold/warm AOT story comparisons.
///
/// The protocol is test infrastructure, not solver configuration. Keeping it
/// in one value prevents a Rust/C/Zig or Lambdify/AOT row from silently using
/// different repetitions, cooldowns or artifact hygiene. The protocol is
/// intentionally copyable so a story can attach the same value to every row
/// without allocating or consulting a global test registry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AotStoryProtocol {
    pub(crate) n_steps: usize,
    pub(crate) cold_repetitions: usize,
    pub(crate) warm_repetitions: usize,
    pub(crate) cold_cooldown_ms: u64,
    pub(crate) warm_cooldown_ms: u64,
    pub(crate) clean_artifacts: bool,
    /// `0` means inherit the process/thread-pool policy. A non-zero value is
    /// recorded as the requested worker budget; the harness remains
    /// responsible for applying it before a worker pool is initialized.
    pub(crate) worker_threads: usize,
}

impl AotStoryProtocol {
    /// Reads the common story controls from the environment.
    ///
    /// Environment overrides are useful for release runs, while the defaults
    /// keep debug correctness tests short. Invalid or zero values fall back to
    /// the supplied defaults rather than changing the comparison silently.
    pub(crate) fn from_env(n_steps: usize, default_repetitions: usize) -> Self {
        let cold_repetitions =
            story_repetitions("BVP_AOT_COLD_REPETITIONS", default_repetitions.max(1));
        let warm_repetitions =
            story_repetitions("BVP_AOT_WARM_REPETITIONS", default_repetitions.max(1));
        let worker_threads = env::var("BVP_AOT_WORKER_THREADS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        let clean_artifacts = env::var("BVP_AOT_COLD_CLEAN_ARTIFACTS")
            .map(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false);

        Self {
            n_steps,
            cold_repetitions,
            warm_repetitions,
            cold_cooldown_ms: env_u64("BVP_AOT_COLD_COOLDOWN_MS", 0),
            warm_cooldown_ms: env_u64("BVP_AOT_WARM_COOLDOWN_MS", DEFAULT_WARM_STORY_COOLDOWN_MS),
            clean_artifacts,
            worker_threads,
        }
    }

    /// Rejects protocol values that cannot produce a meaningful comparison.
    pub(crate) fn validate(self) -> Result<(), &'static str> {
        if self.n_steps < 2 {
            return Err("AOT story protocol requires at least two mesh steps");
        }
        if self.cold_repetitions == 0 || self.warm_repetitions == 0 {
            return Err("AOT story protocol requires non-zero repetitions");
        }
        Ok(())
    }

    /// Stable text embedded in story reports so old rows are auditable.
    pub(crate) fn summary(self) -> String {
        format!(
            "n_steps={}; cold_repetitions={}; warm_repetitions={}; cold_cooldown_ms={}; warm_cooldown_ms={}; clean_artifacts={}; worker_threads={}",
            self.n_steps,
            self.cold_repetitions,
            self.warm_repetitions,
            self.cold_cooldown_ms,
            self.warm_cooldown_ms,
            self.clean_artifacts,
            self.worker_threads,
        )
    }
}

/// Shared architecture marker printed by migration stories.
pub(crate) const TEST_SUITE_ARCHITECTURE: &str =
    "legacy-oracle + atomview-parity + isolated-performance";

pub(crate) fn env_u64(name: &str, default: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default)
}

pub(crate) fn sleep_ms(ms: u64) {
    if ms > 0 {
        thread::sleep(Duration::from_millis(ms));
    }
}

pub(crate) fn story_repetitions(env_name: &str, default: usize) -> usize {
    env::var(env_name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

/// Builds the common uniform initial state used by combustion stories.
pub(crate) fn uniform_initial_guess(
    variable_count: usize,
    n_steps: usize,
    value: f64,
) -> DMatrix<f64> {
    DMatrix::from_element(variable_count, n_steps, value)
}

/// Returns the maximum componentwise absolute difference between two vectors.
pub(crate) fn max_abs_slice_diff(lhs: &[f64], rhs: &[f64]) -> f64 {
    assert_eq!(lhs.len(), rhs.len(), "vectors must have equal lengths");
    lhs.iter()
        .zip(rhs)
        .map(|(&left, &right)| (left - right).abs())
        .fold(0.0, f64::max)
}
