use crate::environment::{EnvironmentFingerprint, MachineProfile};
use crate::error::RunnerError;
use crate::measurement::{MeasurementSeries, MetricStats};
use crate::registry::Scenario;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("performance")
        .join("baselines")
}

pub fn path(
    machine: &MachineProfile,
    environment: &EnvironmentFingerprint,
    scenario: &Scenario,
) -> PathBuf {
    root()
        .join(&machine.alias)
        .join(&environment.id)
        .join(format!("{}.toml", scenario.id))
}

pub fn write(
    path: &Path,
    machine: &MachineProfile,
    environment: &EnvironmentFingerprint,
    scenario: &Scenario,
    series: &MeasurementSeries,
    tolerance: f64,
    force: bool,
) -> Result<(), RunnerError> {
    if path.exists() && !force {
        return Err(RunnerError::Configuration(format!(
            "baseline exists at {}; pass --force to replace it",
            path.display()
        )));
    }
    let value = encode(machine, environment, scenario, series, tolerance);
    let body = toml::to_string_pretty(&value)?;
    let parent = path
        .parent()
        .ok_or_else(|| RunnerError::Configuration("baseline path has no parent".into()))?;
    fs::create_dir_all(parent)?;
    publish_no_clobber(path, &body, force)?;
    Ok(())
}

pub fn read(path: &Path) -> Result<toml::Value, RunnerError> {
    let contents = fs::read_to_string(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            RunnerError::MissingBaseline(path.display().to_string())
        } else {
            RunnerError::Io(error)
        }
    })?;
    let value = toml::Value::Table(toml::from_str::<toml::Table>(&contents)?);
    if value
        .get("schema_version")
        .and_then(toml::Value::as_integer)
        != Some(3)
    {
        return Err(RunnerError::IncompatibleBaseline(format!(
            "unsupported baseline schema at {}",
            path.display()
        )));
    }
    const ROOT_FIELDS: &[&str] = &[
        "schema_version",
        "machine_id",
        "machine_alias",
        "environment_id",
        "scenario_id",
        "scenario_version",
        "solver",
        "target",
        "profile",
        "cpu",
        "rustc",
        "cpu_flags",
        "worker_config",
        "compiled_features",
        "source_revision",
        "source_dirty",
        "lockfile_fingerprint",
        "gate_metrics",
        "warmup_runs",
        "measurement_runs",
        "metrics",
        "counters",
        "diagnostics",
    ];
    let root = value
        .as_table()
        .ok_or_else(|| RunnerError::IncompatibleBaseline("baseline root is not a table".into()))?;
    if let Some(unknown) = root.keys().find(|key| !ROOT_FIELDS.contains(&key.as_str())) {
        return Err(RunnerError::IncompatibleBaseline(format!(
            "unknown baseline field {unknown:?}; bump schema version for format changes"
        )));
    }
    for required in ROOT_FIELDS {
        if value.get(required).is_none() {
            return Err(RunnerError::IncompatibleBaseline(format!(
                "baseline {} lacks {required}",
                path.display()
            )));
        }
    }
    for field in ["source_revision", "lockfile_fingerprint"] {
        if value
            .get(field)
            .and_then(toml::Value::as_str)
            .is_none_or(str::is_empty)
        {
            return Err(RunnerError::IncompatibleBaseline(format!(
                "baseline {} has invalid {field}",
                path.display()
            )));
        }
    }
    if value
        .get("source_dirty")
        .and_then(toml::Value::as_bool)
        .is_none()
    {
        return Err(RunnerError::IncompatibleBaseline(format!(
            "baseline {} has invalid source_dirty flag",
            path.display()
        )));
    }
    let runs = value
        .get("measurement_runs")
        .and_then(toml::Value::as_integer)
        .filter(|runs| *runs >= 3)
        .ok_or_else(|| {
            RunnerError::IncompatibleBaseline(
                "baseline measurement_runs must be at least three".into(),
            )
        })? as usize;
    let warmups = value
        .get("warmup_runs")
        .and_then(toml::Value::as_integer)
        .filter(|runs| *runs >= 0)
        .ok_or_else(|| {
            RunnerError::IncompatibleBaseline("baseline warmup_runs is invalid".into())
        })?;
    if warmups > 1000 || runs > 1000 {
        return Err(RunnerError::IncompatibleBaseline(
            "baseline warmup/sample count exceeds the supported limit".into(),
        ));
    }
    let metrics = value
        .get("metrics")
        .and_then(toml::Value::as_table)
        .ok_or_else(|| {
            RunnerError::Configuration(format!("baseline {} has invalid metrics", path.display()))
        })?;
    if metrics.is_empty() {
        return Err(RunnerError::IncompatibleBaseline(
            "baseline has no metrics".into(),
        ));
    }
    let gates = value
        .get("gate_metrics")
        .and_then(toml::Value::as_array)
        .filter(|gates| !gates.is_empty())
        .ok_or_else(|| RunnerError::IncompatibleBaseline("baseline has no gate_metrics".into()))?;
    let mut gate_ids: Vec<String> = Vec::with_capacity(gates.len());
    for gate in gates {
        let id = gate
            .as_str()
            .filter(|id| !id.is_empty())
            .ok_or_else(|| RunnerError::IncompatibleBaseline("invalid gate metric ID".into()))?;
        if gate_ids.iter().any(|seen| seen.as_str() == id) || !metrics.contains_key(id) {
            return Err(RunnerError::IncompatibleBaseline(format!(
                "duplicate or unknown gate metric {id:?}"
            )));
        }
        gate_ids.push(id.to_owned());
    }
    for (id, metric) in metrics {
        validate_metric(id, metric, runs, gate_ids.iter().any(|gate| gate == id))?;
    }
    let counters = value
        .get("counters")
        .and_then(toml::Value::as_table)
        .ok_or_else(|| RunnerError::IncompatibleBaseline("baseline has invalid counters".into()))?;
    for (id, counter) in counters {
        validate_counter(id, counter, runs)?;
    }
    let diagnostics = value
        .get("diagnostics")
        .and_then(toml::Value::as_array)
        .ok_or_else(|| {
            RunnerError::IncompatibleBaseline("baseline has invalid diagnostics".into())
        })?;
    for diagnostic in diagnostics {
        let table = diagnostic
            .as_table()
            .ok_or_else(|| RunnerError::IncompatibleBaseline("diagnostic is not a table".into()))?;
        if table
            .keys()
            .any(|key| !["id", "unit", "value"].contains(&key.as_str()))
            || table
                .get("id")
                .and_then(toml::Value::as_str)
                .is_none_or(str::is_empty)
            || table
                .get("unit")
                .and_then(toml::Value::as_str)
                .is_none_or(str::is_empty)
        {
            return Err(RunnerError::IncompatibleBaseline(
                "diagnostic has invalid fields".into(),
            ));
        }
        metric_number(table.get("value"), "diagnostic", "value")?;
    }
    Ok(value)
}

fn validate_metric(
    id: &str,
    value: &toml::Value,
    expected_samples: usize,
    is_gate: bool,
) -> Result<(), RunnerError> {
    let invalid =
        |reason: &str| RunnerError::IncompatibleBaseline(format!("metric {id:?} {reason}"));
    let table = value.as_table().ok_or_else(|| invalid("is not a table"))?;
    const FIELDS: &[&str] = &[
        "unit",
        "phase",
        "scope",
        "parent_scope",
        "median",
        "mad",
        "min",
        "max",
        "tolerance",
        "samples",
    ];
    if let Some(unknown) = table.keys().find(|key| !FIELDS.contains(&key.as_str())) {
        return Err(invalid(&format!("has unknown field {unknown:?}")));
    }
    let unit = table.get("unit").and_then(toml::Value::as_str);
    if unit.is_none_or(str::is_empty) {
        return Err(invalid("has an empty or missing unit"));
    }
    if table
        .get("phase")
        .and_then(toml::Value::as_str)
        .is_none_or(str::is_empty)
        || table
            .get("scope")
            .and_then(toml::Value::as_str)
            .is_none_or(str::is_empty)
        || table
            .get("parent_scope")
            .is_some_and(|value| value.as_str().is_none_or(str::is_empty))
    {
        return Err(invalid("has invalid measurement scope metadata"));
    }
    let median = metric_number(table.get("median"), id, "median")?;
    let mad = metric_number(table.get("mad"), id, "mad")?;
    let min = metric_number(table.get("min"), id, "min")?;
    let max = metric_number(table.get("max"), id, "max")?;
    let tolerance = metric_number(table.get("tolerance"), id, "tolerance")?;
    if (is_gate && median <= 0.0)
        || (!is_gate && median < 0.0)
        || mad < 0.0
        || min < 0.0
        || max < min
        || !(0.0..=10.0).contains(&tolerance)
    {
        return Err(invalid("contains invalid summary values"));
    }
    let samples = table
        .get("samples")
        .and_then(toml::Value::as_array)
        .ok_or_else(|| invalid("has no raw sample array"))?;
    if samples.is_empty()
        || samples.len() > expected_samples
        || (is_gate && samples.len() != expected_samples)
    {
        return Err(invalid(
            "raw sample count is invalid for its gate/optional diagnostic role",
        ));
    }
    let mut raw = samples
        .iter()
        .map(|sample| metric_number(Some(sample), id, "sample"))
        .collect::<Result<Vec<_>, _>>()?;
    if raw.iter().any(|sample| *sample < 0.0) {
        return Err(invalid("contains a negative sample"));
    }
    raw.sort_by(f64::total_cmp);
    let calculated_median = sample_median(&raw);
    let mut deviations = raw
        .iter()
        .map(|sample| (sample - calculated_median).abs())
        .collect::<Vec<_>>();
    deviations.sort_by(f64::total_cmp);
    let calculated_mad = sample_median(&deviations);
    if !approximately_equal(raw[0], min)
        || !approximately_equal(*raw.last().expect("samples checked nonempty"), max)
        || !approximately_equal(calculated_median, median)
        || !approximately_equal(calculated_mad, mad)
    {
        return Err(invalid("stored summaries do not match raw samples"));
    }
    Ok(())
}

fn metric_number(value: Option<&toml::Value>, id: &str, field: &str) -> Result<f64, RunnerError> {
    value
        .and_then(toml::Value::as_float)
        .or_else(|| {
            value
                .and_then(toml::Value::as_integer)
                .map(|number| number as f64)
        })
        .filter(|number| number.is_finite())
        .ok_or_else(|| {
            RunnerError::IncompatibleBaseline(format!("metric {id:?} has invalid {field}"))
        })
}

fn sample_median(sorted: &[f64]) -> f64 {
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}

fn approximately_equal(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-12 * left.abs().max(right.abs()).max(f64::MIN_POSITIVE)
}

fn encode(
    machine: &MachineProfile,
    env: &EnvironmentFingerprint,
    scenario: &Scenario,
    series: &MeasurementSeries,
    tolerance: f64,
) -> toml::Value {
    let mut root = toml::map::Map::new();
    root.insert("schema_version".into(), toml::Value::Integer(3));
    root.insert(
        "machine_id".into(),
        toml::Value::String(machine.installation_id.clone()),
    );
    root.insert(
        "machine_alias".into(),
        toml::Value::String(machine.alias.clone()),
    );
    root.insert("environment_id".into(), toml::Value::String(env.id.clone()));
    root.insert(
        "scenario_id".into(),
        toml::Value::String(scenario.id.into()),
    );
    root.insert(
        "scenario_version".into(),
        toml::Value::Integer(scenario.version as i64),
    );
    root.insert("solver".into(), toml::Value::String(scenario.solver.into()));
    root.insert("target".into(), toml::Value::String(env.target.clone()));
    root.insert("profile".into(), toml::Value::String(env.profile.clone()));
    root.insert("cpu".into(), toml::Value::String(env.cpu.clone()));
    root.insert("rustc".into(), toml::Value::String(env.rustc.clone()));
    root.insert(
        "cpu_flags".into(),
        toml::Value::String(env.cpu_flags.clone()),
    );
    root.insert(
        "worker_config".into(),
        toml::Value::String(env.worker_config.clone()),
    );
    root.insert(
        "compiled_features".into(),
        toml::Value::String(env.compiled_features.clone()),
    );
    root.insert(
        "source_revision".into(),
        toml::Value::String(env.source_revision.clone()),
    );
    root.insert(
        "source_dirty".into(),
        toml::Value::Boolean(env.source_dirty),
    );
    root.insert(
        "lockfile_fingerprint".into(),
        toml::Value::String(env.lockfile_fingerprint.clone()),
    );
    root.insert(
        "warmup_runs".into(),
        toml::Value::Integer(series.warmup_runs as i64),
    );
    root.insert(
        "measurement_runs".into(),
        toml::Value::Integer(series.measurement_runs as i64),
    );
    root.insert(
        "gate_metrics".into(),
        toml::Value::Array(
            series
                .gate_metrics
                .iter()
                .cloned()
                .map(toml::Value::String)
                .collect(),
        ),
    );
    let metrics = series
        .metrics
        .iter()
        .map(|metric| (metric.id.clone(), metric_value(metric, tolerance)))
        .collect();
    root.insert("metrics".into(), toml::Value::Table(metrics));
    let counters = series
        .counters
        .iter()
        .map(|counter| (counter.id.clone(), counter_value(counter)))
        .collect();
    root.insert("counters".into(), toml::Value::Table(counters));
    let diagnostics = series
        .diagnostics
        .iter()
        .map(|(id, unit, value)| {
            let mut diagnostic = toml::map::Map::new();
            diagnostic.insert("id".into(), toml::Value::String(id.clone()));
            diagnostic.insert("unit".into(), toml::Value::String(unit.clone()));
            diagnostic.insert("value".into(), toml::Value::Float(*value));
            toml::Value::Table(diagnostic)
        })
        .collect();
    root.insert("diagnostics".into(), toml::Value::Array(diagnostics));
    toml::Value::Table(root)
}

fn counter_value(counter: &crate::measurement::CounterStats) -> toml::Value {
    let mut value = toml::map::Map::new();
    value.insert("median".into(), toml::Value::Float(counter.median));
    value.insert("min".into(), toml::Value::Integer(counter.min as i64));
    value.insert("max".into(), toml::Value::Integer(counter.max as i64));
    value.insert(
        "samples".into(),
        toml::Value::Array(
            counter
                .samples
                .iter()
                .map(|sample| toml::Value::Integer(*sample as i64))
                .collect(),
        ),
    );
    toml::Value::Table(value)
}

fn validate_counter(
    id: &str,
    value: &toml::Value,
    expected_samples: usize,
) -> Result<(), RunnerError> {
    let invalid =
        |reason: &str| RunnerError::IncompatibleBaseline(format!("counter {id:?} {reason}"));
    if id.is_empty() {
        return Err(invalid("has an empty identifier"));
    }
    let table = value.as_table().ok_or_else(|| invalid("is not a table"))?;
    const FIELDS: &[&str] = &["median", "min", "max", "samples"];
    if let Some(unknown) = table.keys().find(|key| !FIELDS.contains(&key.as_str())) {
        return Err(invalid(&format!("has unknown field {unknown:?}")));
    }
    let median = metric_number(table.get("median"), id, "median")?;
    let integer = |field: &str| {
        table
            .get(field)
            .and_then(toml::Value::as_integer)
            .filter(|value| *value >= 0)
            .map(|value| value as u64)
            .ok_or_else(|| invalid(&format!("has invalid {field}")))
    };
    let min = integer("min")?;
    let max = integer("max")?;
    let samples = table
        .get("samples")
        .and_then(toml::Value::as_array)
        .ok_or_else(|| invalid("has no raw sample array"))?;
    if samples.len() != expected_samples {
        return Err(invalid("raw sample count differs from measurement_runs"));
    }
    let mut raw = samples
        .iter()
        .map(|sample| {
            sample
                .as_integer()
                .filter(|value| *value >= 0)
                .map(|value| value as u64)
                .ok_or_else(|| invalid("contains an invalid sample"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    raw.sort_unstable();
    let middle = raw.len() / 2;
    let calculated_median = if raw.len().is_multiple_of(2) {
        raw[middle - 1] as f64 / 2.0 + raw[middle] as f64 / 2.0
    } else {
        raw[middle] as f64
    };
    if raw[0] != min
        || *raw.last().expect("sample count validated") != max
        || !approximately_equal(calculated_median, median)
    {
        return Err(invalid("stored summary does not match raw samples"));
    }
    Ok(())
}

fn metric_value(metric: &MetricStats, tolerance: f64) -> toml::Value {
    let mut table = toml::map::Map::new();
    table.insert("unit".into(), toml::Value::String(metric.unit.clone()));
    table.insert("phase".into(), toml::Value::String(metric.phase.clone()));
    table.insert("scope".into(), toml::Value::String(metric.scope.clone()));
    if let Some(parent_scope) = &metric.parent_scope {
        table.insert(
            "parent_scope".into(),
            toml::Value::String(parent_scope.clone()),
        );
    }
    table.insert("median".into(), toml::Value::Float(metric.median));
    table.insert("mad".into(), toml::Value::Float(metric.mad));
    table.insert("min".into(), toml::Value::Float(metric.min));
    table.insert("max".into(), toml::Value::Float(metric.max));
    table.insert("tolerance".into(), toml::Value::Float(tolerance));
    table.insert(
        "samples".into(),
        toml::Value::Array(
            metric
                .samples
                .iter()
                .copied()
                .map(toml::Value::Float)
                .collect(),
        ),
    );
    toml::Value::Table(table)
}

fn publish_no_clobber(path: &Path, contents: &str, force: bool) -> Result<(), RunnerError> {
    publish_no_clobber_with_hook(path, contents, force, || Ok(()))
}

fn publish_no_clobber_with_hook(
    path: &Path,
    contents: &str,
    force: bool,
    before_publish: impl FnOnce() -> std::io::Result<()>,
) -> Result<(), RunnerError> {
    let temporary = create_temporary_file(path, contents)?;
    if force {
        let backup = unique_sibling_path(path, "backup");
        let had_previous = path.exists();
        if had_previous {
            fs::rename(path, &backup)?;
        }
        if let Err(error) = before_publish() {
            if had_previous {
                let _ = fs::rename(&backup, path);
            }
            let _ = fs::remove_file(&temporary);
            return Err(RunnerError::Io(error));
        }
        if let Err(error) = fs::rename(&temporary, path) {
            if had_previous {
                let _ = fs::rename(&backup, path);
            }
            let _ = fs::remove_file(&temporary);
            return Err(RunnerError::Io(error));
        }
        if had_previous {
            fs::remove_file(backup)?;
        }
        return Ok(());
    }
    match fs::hard_link(&temporary, path) {
        Ok(()) => {
            fs::remove_file(&temporary)?;
            Ok(())
        }
        Err(error) => {
            let _ = fs::remove_file(&temporary);
            Err(RunnerError::Io(error))
        }
    }
}

fn create_temporary_file(path: &Path, contents: &str) -> Result<PathBuf, RunnerError> {
    for _ in 0..128 {
        let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let temporary = path.with_extension(format!("toml.tmp-{}-{sequence}", std::process::id()));
        let mut file = match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
        {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(RunnerError::Io(error)),
        };
        if let Err(error) = file.write_all(contents.as_bytes()) {
            drop(file);
            let _ = fs::remove_file(&temporary);
            return Err(RunnerError::Io(error));
        }
        return Ok(temporary);
    }
    Err(RunnerError::Configuration(format!(
        "could not allocate a unique temporary file beside {}",
        path.display()
    )))
}

fn unique_sibling_path(path: &Path, role: &str) -> PathBuf {
    let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    path.with_extension(format!("toml.{role}-{}-{sequence}", std::process::id()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measurement::CounterStats;

    #[test]
    fn validates_raw_samples_against_persisted_statistics() {
        let valid = toml::Value::Table(
            toml::from_str::<toml::Table>(
                r#"
            unit = "ms"
            phase = "gate_off"
            scope = "scenario_outer_wall"
            median = 10.0
            mad = 2.0
            min = 8.0
            max = 12.0
            tolerance = 0.1
            samples = [8.0, 10.0, 12.0]
        "#,
            )
            .unwrap(),
        );
        validate_metric("solve_ms", &valid, 3, true).unwrap();

        let optional_diagnostic = toml::Value::Table(
            toml::from_str::<toml::Table>(
                r#"
            unit = "ms/timing"
            phase = "instrumented_diagnostic"
            scope = "native_stage_inclusive"
            parent_scope = "solver_integration"
            median = 0.0
            mad = 0.0
            min = 0.0
            max = 0.0
            tolerance = 0.1
            samples = [0.0]
        "#,
            )
            .unwrap(),
        );
        validate_metric("stage_optional", &optional_diagnostic, 3, false).unwrap();
        assert!(validate_metric("stage_optional", &optional_diagnostic, 3, true).is_err());

        let inconsistent = toml::Value::Table(
            toml::from_str::<toml::Table>(
                r#"
            unit = "ms"
            phase = "gate_off"
            scope = "scenario_outer_wall"
            median = 11.0
            mad = 2.0
            min = 8.0
            max = 12.0
            tolerance = 0.1
            samples = [8.0, 10.0, 12.0]
        "#,
            )
            .unwrap(),
        );
        assert!(validate_metric("solve_ms", &inconsistent, 3, true).is_err());
    }

    #[test]
    fn rejects_incomplete_raw_sample_history() {
        let metric = toml::Value::Table(
            toml::from_str::<toml::Table>(
                r#"
            unit = "ms"
            phase = "gate_off"
            scope = "scenario_outer_wall"
            median = 10.0
            mad = 0.0
            min = 10.0
            max = 10.0
            tolerance = 0.1
            samples = [10.0]
        "#,
            )
            .unwrap(),
        );
        assert!(validate_metric("solve_ms", &metric, 3, true).is_err());
    }

    #[test]
    fn validates_counter_range_and_median_from_raw_samples() {
        let counter = toml::Value::Table(
            toml::from_str::<toml::Table>(
                r#"
                    median = 5.0
                    min = 4
                    max = 6
                    samples = [4, 5, 6]
                "#,
            )
            .unwrap(),
        );
        validate_counter("accepted_steps", &counter, 3).unwrap();
    }

    #[test]
    fn versioned_baseline_roundtrips_gate_and_zero_diagnostic_metric() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("fixture.toml");
        let machine = MachineProfile {
            alias: "test-machine".into(),
            installation_id: "test-installation".into(),
        };
        let environment = EnvironmentFingerprint {
            id: "test-environment".into(),
            target: "test-target".into(),
            profile: "release".into(),
            cpu: "test-cpu".into(),
            rustc: "rustc test".into(),
            cpu_flags: "test-flags".into(),
            worker_config: "1".into(),
            compiled_features: String::new(),
            source_revision: "abc123".into(),
            source_dirty: false,
            lockfile_fingerprint: "fnv1a-test".into(),
        };
        let scenario = &crate::registry::SCENARIOS[0];
        let series = MeasurementSeries {
            scenario_id: scenario.id.into(),
            scenario_version: scenario.version,
            gate_metrics: vec!["scenario_wall_ms".into()],
            warmup_runs: 1,
            measurement_runs: 3,
            metrics: vec![
                MetricStats {
                    id: "scenario_wall_ms".into(),
                    unit: "ms".into(),
                    phase: "gate_off".into(),
                    scope: "scenario_outer_wall".into(),
                    parent_scope: None,
                    samples: vec![8.0, 10.0, 12.0],
                    median: 10.0,
                    mad: 2.0,
                    min: 8.0,
                    max: 12.0,
                },
                MetricStats {
                    id: "setup_ms".into(),
                    unit: "ms".into(),
                    phase: "gate_off".into(),
                    scope: "scenario_component".into(),
                    parent_scope: Some("scenario_outer_wall".into()),
                    samples: vec![0.0, 0.0, 0.0],
                    median: 0.0,
                    mad: 0.0,
                    min: 0.0,
                    max: 0.0,
                },
            ],
            counters: vec![CounterStats {
                id: "factorizations".into(),
                samples: vec![2, 2, 2],
                median: 2.0,
                min: 2,
                max: 2,
            }],
            diagnostics: Vec::new(),
        };

        write(&path, &machine, &environment, scenario, &series, 0.1, false).unwrap();
        let original_contents = fs::read_to_string(&path).unwrap();
        assert!(write(&path, &machine, &environment, scenario, &series, 0.1, false).is_err());
        assert_eq!(fs::read_to_string(&path).unwrap(), original_contents);

        let mut replacement = series.clone();
        let gate = replacement
            .metrics
            .iter_mut()
            .find(|metric| metric.id == "scenario_wall_ms")
            .unwrap();
        gate.samples = vec![18.0, 20.0, 22.0];
        gate.median = 20.0;
        gate.mad = 2.0;
        gate.min = 18.0;
        gate.max = 22.0;
        write(
            &path,
            &machine,
            &environment,
            scenario,
            &replacement,
            0.1,
            true,
        )
        .unwrap();

        let decoded = read(&path).unwrap();
        assert_eq!(decoded["schema_version"].as_integer(), Some(3));
        assert_eq!(
            decoded["gate_metrics"][0].as_str(),
            Some("scenario_wall_ms")
        );
        assert_eq!(
            decoded["metrics"]["setup_ms"]["median"].as_float(),
            Some(0.0)
        );
        assert_eq!(
            decoded["metrics"]["scenario_wall_ms"]["median"].as_float(),
            Some(20.0)
        );
        assert_eq!(decoded["source_dirty"].as_bool(), Some(false));
        let entries = fs::read_dir(directory.path()).unwrap().count();
        assert_eq!(
            entries, 1,
            "temporary/backup artifacts should be cleaned up"
        );
        let before_read = fs::read(&path).unwrap();
        let _ = read(&path).unwrap();
        assert_eq!(
            fs::read(&path).unwrap(),
            before_read,
            "read must not mutate baseline"
        );
    }

    #[test]
    fn concurrent_no_clobber_writers_publish_one_complete_file() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("race.toml");
        let writers = 8;
        let results = std::thread::scope(|scope| {
            (0..writers)
                .map(|index| {
                    let path = &path;
                    scope.spawn(move || {
                        publish_no_clobber(path, &format!("complete-payload-{index}"), false)
                    })
                })
                .map(|writer| writer.join().unwrap())
                .collect::<Vec<_>>()
        });

        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        let published = fs::read_to_string(&path).unwrap();
        assert!((0..writers).any(|index| published == format!("complete-payload-{index}")));
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
    }

    #[test]
    fn failed_force_publication_restores_previous_file_and_cleans_staging_files() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("baseline.toml");
        fs::write(&path, "old-complete-baseline").unwrap();

        let result = publish_no_clobber_with_hook(&path, "new-baseline", true, || {
            Err(std::io::Error::other(
                "simulated interruption before publish",
            ))
        });

        assert!(result.is_err());
        assert_eq!(fs::read_to_string(&path).unwrap(), "old-complete-baseline");
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
    }
}
