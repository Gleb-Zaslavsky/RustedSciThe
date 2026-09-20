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
