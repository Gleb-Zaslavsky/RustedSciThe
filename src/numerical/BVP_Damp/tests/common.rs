#![allow(dead_code)]

use std::env;
use std::thread;
use std::time::Duration;

pub(crate) const DEFAULT_COLD_STORY_COOLDOWN_MS: u64 = 5_000;
pub(crate) const DEFAULT_WARM_STORY_COOLDOWN_MS: u64 = 1_000;

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
