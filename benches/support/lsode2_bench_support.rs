use chrono::{SecondsFormat, Utc};
use std::time::Duration;

use RustedSciThe::numerical::LSODE2::workload_fixtures::WorkloadKind;

pub fn dimensions_from_env(variable: &str, default: &[usize]) -> Vec<usize> {
    let Some(value) = std::env::var_os(variable) else {
        return default.to_vec();
    };

    let mut dimensions = value
        .to_string_lossy()
        .split(',')
        .filter_map(|item| item.trim().parse::<usize>().ok())
        .filter(|dimension| *dimension > 0)
        .collect::<Vec<_>>();
    dimensions.sort_unstable();
    dimensions.dedup();

    if dimensions.is_empty() {
        default.to_vec()
    } else {
        dimensions
    }
}

#[allow(dead_code)]
pub fn positive_usizes_from_env(variable: &str, default: &[usize]) -> Vec<usize> {
    let Some(value) = std::env::var_os(variable) else {
        return default.to_vec();
    };

    let mut values = value
        .to_string_lossy()
        .split(',')
        .filter_map(|item| item.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .collect::<Vec<_>>();
    values.sort_unstable();
    values.dedup();

    if values.is_empty() {
        default.to_vec()
    } else {
        values
    }
}

pub fn sample_size() -> usize {
    std::env::var("LSODE2_BENCH_SAMPLE_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|sample_size| *sample_size >= 10)
        .unwrap_or(10)
}

pub fn measurement_time() -> Option<Duration> {
    std::env::var("LSODE2_BENCH_MEASUREMENT_TIME_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .map(Duration::from_secs)
}

#[allow(dead_code)]
pub fn workloads_from_env(variable: &str, default: &[WorkloadKind]) -> Vec<WorkloadKind> {
    let Some(value) = std::env::var_os(variable) else {
        return default.to_vec();
    };

    let mut workloads = Vec::new();
    for label in value.to_string_lossy().split(',') {
        let label = label.trim();
        let workload = WorkloadKind::ALL
            .iter()
            .copied()
            .find(|workload| workload.label() == label)
            .unwrap_or_else(|| {
                panic!(
                    "invalid {variable} value {label:?}; expected one of: {}",
                    WorkloadKind::ALL
                        .iter()
                        .map(|workload| workload.label())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            });
        if !workloads.contains(&workload) {
            workloads.push(workload);
        }
    }

    if workloads.is_empty() {
        default.to_vec()
    } else {
        workloads
    }
}

pub fn print_metadata(bench: &str, dimensions: &[usize], extra: &str) {
    eprintln!(
        "[LSODE2 bench metadata] bench={bench}; recorded_at_utc={}; profile={}; os={}; arch={}; package={}; dimensions={dimensions:?}; sample_size={}; measurement_time_secs={:?}; {extra}",
        Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true),
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        std::env::consts::OS,
        std::env::consts::ARCH,
        env!("CARGO_PKG_VERSION"),
        sample_size(),
        measurement_time().map(|duration| duration.as_secs()),
    );
}
