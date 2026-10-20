use crate::comparison::{MetricComparison, Verdict};
use crate::environment::{EnvironmentFingerprint, MachineProfile};
use crate::measurement::MeasurementSeries;
use crate::registry::Scenario;
use tabled::Tabled;

#[derive(Tabled)]
struct MetricRow {
    metric: String,
    baseline: String,
    current: String,
    change: String,
    threshold: String,
    result: String,
}

pub fn print_list(
    scenarios: &[&Scenario],
    machine: Option<&MachineProfile>,
    env: &EnvironmentFingerprint,
    has_baseline: impl Fn(&Scenario) -> bool,
) {
    println!(
        "Performance scenarios | target={} | profile={} | machine={}",
        env.target,
        env.profile,
        machine.map_or("unregistered", |m| m.alias.as_str())
    );
    for scenario in scenarios {
        println!(
            "{} | solver={} | workload={} | frontend={} | execution={} | linalg={} | parallel={} | telemetry={} | groups={} | baseline={} | {}",
            scenario.id,
            scenario.solver,
            scenario.workload,
            scenario.frontend,
            scenario.execution,
            scenario.linear_algebra,
            scenario.parallel_policy,
            scenario.telemetry,
            scenario.groups.join(","),
            if machine.is_some() {
                if has_baseline(scenario) {
                    "baseline available"
                } else {
                    "NO BASELINE"
                }
            } else {
                "register machine first"
            },
            scenario.description
        );
    }
}

pub fn print_record(
    machine: &MachineProfile,
    env: &EnvironmentFingerprint,
    scenario: &Scenario,
    series: &MeasurementSeries,
    path: &std::path::Path,
) {
    println!(
        "Recording {} baseline on {} / {} ({} samples)",
        scenario.id, machine.alias, env.id, series.measurement_runs
    );
    println!(
        "source={}{} | rustc={} | Cargo.lock={}",
        env.source_revision,
        if env.source_dirty { " (dirty)" } else { "" },
        env.rustc,
        env.lockfile_fingerprint
    );
    for metric in &series.metrics {
        println!(
            "  {} [{}; scope={}{}]: median={:.4} {}, MAD={:.4}",
            metric.id,
            metric.phase,
            metric.scope,
            metric
                .parent_scope
                .as_deref()
                .map_or(String::new(), |parent| format!(", parent={parent}")),
            metric.median,
            metric.unit,
            metric.mad
        );
    }
    println!(
        "{} baseline created: scenario={} | machine={} | environment={} | path={}",
        scenario.solver,
        scenario.id,
        machine.alias,
        env.id,
        path.display()
    );
}

pub fn print_compare(
    machine: &MachineProfile,
    env: &EnvironmentFingerprint,
    scenario: &Scenario,
    current: &MeasurementSeries,
    rows: &[MetricComparison],
    confirmation: Option<&[MetricComparison]>,
    baseline: &toml::Value,
) -> Verdict {
    println!(
        "\n{} | machine={} | environment={} | samples={}\nsource={}{} | rustc={} | Cargo.lock={}",
        scenario.id,
        machine.alias,
        env.id,
        current.measurement_runs,
        env.source_revision,
        if env.source_dirty { " (dirty)" } else { "" },
        env.rustc,
        env.lockfile_fingerprint
    );
    println!(
        "baseline source={}{} | rustc={} | Cargo.lock={}",
        baseline
            .get("source_revision")
            .and_then(toml::Value::as_str)
            .unwrap_or("unavailable"),
        if baseline
            .get("source_dirty")
            .and_then(toml::Value::as_bool)
            .unwrap_or(true)
        {
            " (dirty)"
        } else {
            ""
        },
        baseline
            .get("rustc")
            .and_then(toml::Value::as_str)
            .unwrap_or("unavailable"),
        baseline
            .get("lockfile_fingerprint")
            .and_then(toml::Value::as_str)
            .unwrap_or("unavailable")
    );
    let table = rows.iter().map(row).collect::<Vec<_>>();
    println!("{}", tabled::Table::new(table));
    for metric in &current.metrics {
        if current.gate_metrics.contains(&metric.id) {
            continue;
        }
        let saved = baseline
            .get("metrics")
            .and_then(toml::Value::as_table)
            .and_then(|metrics| metrics.get(&metric.id))
            .and_then(toml::Value::as_table);
        let previous_median = saved
            .and_then(|metric| metric.get("median"))
            .and_then(|value| {
                value
                    .as_float()
                    .or_else(|| value.as_integer().map(|n| n as f64))
            });
        let previous_mad = saved
            .and_then(|metric| metric.get("mad"))
            .and_then(|value| {
                value
                    .as_float()
                    .or_else(|| value.as_integer().map(|n| n as f64))
            });
        println!(
            "diagnostic {} ({}; phase={}, scope={}): baseline median/MAD={:?}/{:?}, current={:.4}/{:.4}",
            metric.id,
            metric.unit,
            metric.phase,
            metric.scope,
            previous_median,
            previous_mad,
            metric.median,
            metric.mad
        );
    }
    let saved_diagnostics = baseline.get("diagnostics").and_then(toml::Value::as_array);
    for (name, unit, value) in &current.diagnostics {
        let prior = saved_diagnostics
            .and_then(|items| {
                items
                    .iter()
                    .find(|item| item.get("id").and_then(toml::Value::as_str) == Some(name))
            })
            .and_then(|item| item.get("value"))
            .and_then(|value| {
                value
                    .as_float()
                    .or_else(|| value.as_integer().map(|n| n as f64))
            });
        if let Some(prior) = prior {
            println!("correctness {name}: baseline={prior:.6e}, current={value:.6e} {unit}");
        } else {
            println!("correctness {name}: {value:.6e} {unit} (no baseline value)");
        }
    }
    let saved_counters = baseline.get("counters").and_then(toml::Value::as_table);
    for counter in &current.counters {
        let prior = saved_counters
            .and_then(|table| table.get(&counter.id))
            .and_then(toml::Value::as_table);
        let prior_median = prior
            .and_then(|table| table.get("median"))
            .and_then(toml::Value::as_float);
        let prior_range = prior.and_then(|table| {
            Some((
                table.get("min")?.as_integer()?,
                table.get("max")?.as_integer()?,
            ))
        });
        let state = match (prior_median, prior_range) {
            (Some(median), Some((min, max)))
                if median == counter.median
                    && min == counter.min as i64
                    && max == counter.max as i64 =>
            {
                "same"
            }
            (Some(_), Some(_)) => "changed",
            _ => "new",
        };
        let prior_text = prior_median.map_or_else(
            || "unavailable".to_owned(),
            |median| {
                let (min, max) = prior_range.unwrap_or((0, 0));
                format!("median={median}, range={min}..{max}")
            },
        );
        println!(
            "counter {}: {} -> median={}, range={}..{} ({state})",
            counter.id, prior_text, counter.median, counter.min, counter.max
        );
    }
    if let Some(saved_counters) = saved_counters {
        for (id, saved) in saved_counters {
            if current.counters.iter().any(|counter| counter.id == *id) {
                continue;
            }
            let saved = saved.as_table();
            let median = saved
                .and_then(|table| table.get("median"))
                .and_then(toml::Value::as_float);
            let min = saved
                .and_then(|table| table.get("min"))
                .and_then(toml::Value::as_integer);
            let max = saved
                .and_then(|table| table.get("max"))
                .and_then(toml::Value::as_integer);
            println!(
                "counter {id}: baseline median={median:?}, range={min:?}..{max:?} -> unavailable (removed)"
            );
        }
    }
    let mut verdict = aggregate_verdict(rows.iter().map(|row| row.verdict));
    if let Some(confirm) = confirmation {
        println!("Confirmation series:");
        println!(
            "{}",
            tabled::Table::new(confirm.iter().map(row).collect::<Vec<_>>())
        );
        let confirmed_failure = rows.iter().any(|first| {
            first.verdict == Verdict::Fail
                && confirm
                    .iter()
                    .any(|second| second.id == first.id && second.verdict == Verdict::Fail)
        });
        let conflicting = rows.iter().any(|first| {
            confirm
                .iter()
                .find(|second| second.id == first.id)
                .is_some_and(|second| first.verdict != second.verdict)
        });
        verdict = if verdict == Verdict::Unstable || conflicting {
            Verdict::Unstable
        } else if confirmed_failure {
            Verdict::Fail
        } else {
            Verdict::Pass
        };
    }
    println!("RESULT: {}", verdict_name(verdict));
    verdict
}

fn row(comparison: &MetricComparison) -> MetricRow {
    MetricRow {
        metric: format!("{} ({})", comparison.id, comparison.unit),
        baseline: format!("{:.4}", comparison.baseline),
        current: format!("{:.4}", comparison.current),
        change: format!("{:+.1}%", comparison.change * 100.0),
        threshold: format!("+{:.1}%", comparison.tolerance * 100.0),
        result: verdict_name(comparison.verdict).into(),
    }
}

fn verdict_name(verdict: Verdict) -> &'static str {
    match verdict {
        Verdict::Pass => "PASS",
        Verdict::Fail => "FAIL",
        Verdict::Unstable => "UNSTABLE",
    }
}

fn aggregate_verdict(verdicts: impl Iterator<Item = Verdict>) -> Verdict {
    let mut saw_failure = false;
    for verdict in verdicts {
        match verdict {
            Verdict::Unstable => return Verdict::Unstable,
            Verdict::Fail => saw_failure = true,
            Verdict::Pass => {}
        }
    }
    if saw_failure {
        Verdict::Fail
    } else {
        Verdict::Pass
    }
}
