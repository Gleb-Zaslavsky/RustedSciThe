use crate::error::RunnerError;
use crate::measurement::MeasurementSeries;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Pass,
    Fail,
    Unstable,
}

#[derive(Clone, Debug)]
pub struct MetricComparison {
    pub id: String,
    pub unit: String,
    pub baseline: f64,
    pub current: f64,
    pub tolerance: f64,
    pub change: f64,
    pub verdict: Verdict,
}

pub fn compare(
    baseline: &toml::Value,
    current: &MeasurementSeries,
    max_relative_mad: f64,
) -> Result<Vec<MetricComparison>, RunnerError> {
    let baseline_metrics = baseline
        .get("metrics")
        .and_then(toml::Value::as_table)
        .ok_or_else(|| RunnerError::IncompatibleBaseline("missing metrics table".into()))?;
    if baseline_metrics.len() != current.metrics.len() {
        return Err(RunnerError::IncompatibleBaseline(
            "metric set changed".into(),
        ));
    }
    let baseline_gates = baseline
        .get("gate_metrics")
        .and_then(toml::Value::as_array)
        .ok_or_else(|| RunnerError::IncompatibleBaseline("missing gate_metrics array".into()))?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| RunnerError::IncompatibleBaseline("invalid gate metric ID".into()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if baseline_gates != current.gate_metrics {
        return Err(RunnerError::IncompatibleBaseline(
            "regression-gate metric set changed".into(),
        ));
    }
    let mut comparisons = Vec::with_capacity(current.metrics.len());
    for metric in &current.metrics {
        let saved = baseline_metrics
            .get(&metric.id)
            .and_then(toml::Value::as_table)
            .ok_or_else(|| {
                RunnerError::IncompatibleBaseline(format!(
                    "required metric {:?} is missing",
                    metric.id
                ))
            })?;
        let baseline_median = get_float(saved, "median", &metric.id)?;
        let tolerance = get_float(saved, "tolerance", &metric.id)?;
        let saved_mad = get_float(saved, "mad", &metric.id)?;
        if saved_mad < 0.0 || !(0.0..=10.0).contains(&tolerance) {
            return Err(RunnerError::IncompatibleBaseline(format!(
                "metric {:?} has invalid MAD or tolerance",
                metric.id
            )));
        }
        let unit = saved
            .get("unit")
            .and_then(toml::Value::as_str)
            .ok_or_else(|| {
                RunnerError::IncompatibleBaseline(format!("metric {:?} has no unit", metric.id))
            })?;
        if unit != metric.unit {
            return Err(RunnerError::IncompatibleBaseline(format!(
                "metric {:?} unit changed",
                metric.id
            )));
        }
        if saved.get("phase").and_then(toml::Value::as_str) != Some(metric.phase.as_str())
            || saved.get("scope").and_then(toml::Value::as_str) != Some(metric.scope.as_str())
            || saved.get("parent_scope").and_then(toml::Value::as_str)
                != metric.parent_scope.as_deref()
        {
            return Err(RunnerError::IncompatibleBaseline(format!(
                "metric {:?} scope metadata changed",
                metric.id
            )));
        }
        if !baseline_gates.iter().any(|gate| gate == &metric.id) {
            continue;
        }
        if baseline_median <= 0.0 {
            return Err(RunnerError::IncompatibleBaseline(format!(
                "metric {:?} baseline is not positive",
                metric.id
            )));
        }
        let change = (metric.median - baseline_median) / baseline_median;
        let relative_mad = metric.mad / metric.median.max(f64::MIN_POSITIVE);
        let baseline_relative_mad = saved_mad / baseline_median;
        // A zero wall-time median is below the timer's useful resolution, not
        // evidence of an infinite speedup. Keep it visible as unstable data.
        let verdict = if metric.median <= 0.0
            || relative_mad > max_relative_mad
            || baseline_relative_mad > max_relative_mad
        {
            Verdict::Unstable
        } else if change > tolerance {
            Verdict::Fail
        } else {
            Verdict::Pass
        };
        comparisons.push(MetricComparison {
            id: metric.id.clone(),
            unit: metric.unit.clone(),
            baseline: baseline_median,
            current: metric.median,
            tolerance,
            change,
            verdict,
        });
    }
    Ok(comparisons)
}

fn get_float(
    table: &toml::map::Map<String, toml::Value>,
    key: &str,
    metric: &str,
) -> Result<f64, RunnerError> {
    table
        .get(key)
        .and_then(toml::Value::as_float)
        .or_else(|| {
            table
                .get(key)
                .and_then(toml::Value::as_integer)
                .map(|n| n as f64)
        })
        .filter(|value| value.is_finite())
        .ok_or_else(|| {
            RunnerError::IncompatibleBaseline(format!("metric {metric:?} has invalid {key}"))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measurement::{MeasurementSeries, MetricStats};

    fn baseline_value() -> toml::Value {
        toml::Value::Table(
            toml::from_str::<toml::Table>(
                r#"
            gate_metrics = ["solve_ms"]
            [metrics.solve_ms]
            unit = "ms"
            phase = "gate_off"
            scope = "scenario_component"
            parent_scope = "scenario_outer_wall"
            median = 100.0
            mad = 1.0
            tolerance = 0.1
        "#,
            )
            .unwrap(),
        )
    }

    fn current(median: f64, mad: f64) -> MeasurementSeries {
        MeasurementSeries {
            scenario_id: "test".into(),
            scenario_version: 1,
            gate_metrics: vec!["solve_ms".into()],
            warmup_runs: 0,
            measurement_runs: 3,
            metrics: vec![MetricStats {
                id: "solve_ms".into(),
                unit: "ms".into(),
                phase: "gate_off".into(),
                scope: "scenario_component".into(),
                parent_scope: Some("scenario_outer_wall".into()),
                samples: vec![median - mad, median, median + mad],
                median,
                mad,
                min: median - mad,
                max: median + mad,
            }],
            counters: Vec::new(),
            diagnostics: Vec::new(),
        }
    }

    #[test]
    fn flags_only_a_stable_change_above_the_saved_threshold() {
        let baseline = baseline_value();
        let pass = compare(&baseline, &current(105.0, 1.0), 0.1).unwrap();
        assert_eq!(pass[0].verdict, Verdict::Pass);
        let improvement = compare(&baseline, &current(90.0, 1.0), 0.1).unwrap();
        assert_eq!(improvement[0].verdict, Verdict::Pass);
        let fail = compare(&baseline, &current(111.0, 1.0), 0.1).unwrap();
        assert_eq!(fail[0].verdict, Verdict::Fail);
        let passing_median_but_noisy = compare(&baseline, &current(105.0, 20.0), 0.1).unwrap();
        assert_eq!(passing_median_but_noisy[0].verdict, Verdict::Unstable);
        let noisy = compare(&baseline, &current(111.0, 20.0), 0.1).unwrap();
        assert_eq!(noisy[0].verdict, Verdict::Unstable);
    }

    #[test]
    fn rejects_metric_set_and_unit_mismatches() {
        let baseline = baseline_value();
        let mut current = current(100.0, 1.0);
        current.metrics[0].unit = "seconds".into();
        assert!(compare(&baseline, &current, 0.1).is_err());
        current.metrics[0].id = "other".into();
        let error = compare(&baseline, &current, 0.1).unwrap_err();
        assert!(matches!(error, RunnerError::IncompatibleBaseline(_)));
    }

    #[test]
    fn zero_gate_duration_is_unstable_but_positive_tiny_duration_is_finite() {
        let baseline = baseline_value();
        let zero = compare(&baseline, &current(0.0, 0.0), 0.1).unwrap();
        assert_eq!(zero[0].verdict, Verdict::Unstable);

        let tiny = compare(&baseline, &current(1.0e-300, 0.0), 0.1).unwrap();
        assert_eq!(tiny[0].verdict, Verdict::Pass);
        assert!(tiny[0].change.is_finite());
    }

    #[test]
    fn rejects_missing_and_extra_gate_metrics() {
        let baseline = baseline_value();
        let mut missing = current(100.0, 1.0);
        missing.metrics.clear();
        assert!(matches!(
            compare(&baseline, &missing, 0.1),
            Err(RunnerError::IncompatibleBaseline(_))
        ));

        let mut extra = current(100.0, 1.0);
        extra.metrics.push(MetricStats {
            id: "extra_ms".into(),
            unit: "ms".into(),
            phase: "gate_off".into(),
            scope: "scenario_component".into(),
            parent_scope: Some("scenario_outer_wall".into()),
            samples: vec![1.0, 1.0, 1.0],
            median: 1.0,
            mad: 0.0,
            min: 1.0,
            max: 1.0,
        });
        assert!(matches!(
            compare(&baseline, &extra, 0.1),
            Err(RunnerError::IncompatibleBaseline(_))
        ));
    }

    #[test]
    fn diagnostic_timings_are_retained_but_do_not_decide_the_verdict() {
        let mut baseline = baseline_value();
        let metrics = baseline["metrics"].as_table_mut().unwrap();
        metrics.insert(
            "setup_ms".into(),
            toml::Value::Table(
                toml::from_str::<toml::Table>(
                    r#"
                    unit = "ms"
                    phase = "instrumented_diagnostic"
                    scope = "solver_diagnostic"
                    parent_scope = "solver_integration"
                    median = 5.0
                    mad = 0.1
                    tolerance = 0.1
                    "#,
                )
                .unwrap(),
            ),
        );
        let mut current = current(111.0, 1.0);
        current.metrics.push(MetricStats {
            id: "setup_ms".into(),
            unit: "ms".into(),
            phase: "instrumented_diagnostic".into(),
            scope: "solver_diagnostic".into(),
            parent_scope: Some("solver_integration".into()),
            samples: vec![100.0, 101.0, 102.0],
            median: 101.0,
            mad: 1.0,
            min: 100.0,
            max: 102.0,
        });

        let rows = compare(&baseline, &current, 0.1).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, "solve_ms");
        assert_eq!(rows[0].verdict, Verdict::Fail);
    }
}
