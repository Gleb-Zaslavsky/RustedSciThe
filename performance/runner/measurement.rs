use crate::error::RunnerError;
use crate::registry::{NativeMetricScope, Scenario};

#[derive(Clone, Debug)]
pub struct MetricSamples {
    pub id: String,
    pub unit: String,
    pub phase: String,
    pub scope: String,
    pub parent_scope: Option<String>,
    pub samples: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct MetricStats {
    pub id: String,
    pub unit: String,
    /// Whether values came from uninstrumented gate runs or diagnostic runs.
    pub phase: String,
    /// Scope labels are runner-level descriptions; they do not alter solver telemetry.
    pub scope: String,
    pub parent_scope: Option<String>,
    pub samples: Vec<f64>,
    pub median: f64,
    pub mad: f64,
    pub min: f64,
    pub max: f64,
}

#[derive(Clone, Debug)]
pub struct CounterStats {
    pub id: String,
    pub samples: Vec<u64>,
    pub median: f64,
    pub min: u64,
    pub max: u64,
}

#[derive(Clone, Debug)]
pub struct MeasurementSeries {
    pub scenario_id: String,
    pub scenario_version: u32,
    pub gate_metrics: Vec<String>,
    pub warmup_runs: usize,
    pub measurement_runs: usize,
    pub metrics: Vec<MetricStats>,
    pub counters: Vec<CounterStats>,
    pub diagnostics: Vec<(String, String, f64)>,
}

pub fn measure(
    scenario: &Scenario,
    warmups: usize,
    runs: usize,
) -> Result<MeasurementSeries, RunnerError> {
    if runs < 3 {
        return Err(RunnerError::Configuration(
            "measurement runs must be at least three".into(),
        ));
    }
    if scenario.execution.starts_with("AOT") && warmups == 0 {
        return Err(RunnerError::Configuration(format!(
            "AOT scenario {} requires at least one warmup so cold compilation is not mixed into measured warm samples",
            scenario.id
        )));
    }
    if scenario.gate_metrics.is_empty() {
        return Err(RunnerError::Scenario(format!(
            "scenario {} declares no regression-gate metrics",
            scenario.id
        )));
    }
    let mut gate_ids = scenario
        .gate_metrics
        .iter()
        .map(|id| (*id).to_owned())
        .collect::<Vec<_>>();
    gate_ids.sort_unstable();
    if gate_ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(RunnerError::Scenario(format!(
            "scenario {} declares duplicate regression-gate metrics",
            scenario.id
        )));
    }
    for _ in 0..warmups {
        (scenario.run)()?;
    }
    let gate = collect_phase(scenario, Phase::Gate, runs, &gate_ids)?;
    let diagnostic = match scenario.diagnose {
        Some(_) => collect_phase(scenario, Phase::Diagnostic, runs, &[])?,
        None => CollectedPhase::default(),
    };

    let mut all_metrics = gate.metrics;
    for metric in diagnostic.metrics {
        if all_metrics.iter().any(|existing| existing.id == metric.id) {
            return Err(RunnerError::Scenario(format!(
                "scenario {} emitted metric {} in both gate and diagnostic passes",
                scenario.id, metric.id
            )));
        }
        all_metrics.push(metric);
    }
    let mut metrics = all_metrics
        .into_iter()
        .map(stats)
        .collect::<Result<Vec<_>, _>>()?;
    metrics.sort_by(|left, right| left.id.cmp(&right.id));
    let counters = if diagnostic.counters.is_empty() {
        gate.counters
    } else {
        diagnostic.counters
    };
    let diagnostics = if diagnostic.diagnostics.is_empty() {
        gate.diagnostics
    } else {
        diagnostic.diagnostics
    };
    Ok(MeasurementSeries {
        scenario_id: scenario.id.into(),
        scenario_version: scenario.version,
        gate_metrics: gate_ids,
        warmup_runs: warmups,
        measurement_runs: runs,
        metrics,
        counters,
        diagnostics,
    })
}

#[derive(Default)]
struct CollectedPhase {
    metrics: Vec<MetricSamples>,
    counters: Vec<CounterStats>,
    diagnostics: Vec<(String, String, f64)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Gate,
    Diagnostic,
}

impl Phase {
    fn label(self) -> &'static str {
        match self {
            Self::Gate => "gate_off",
            Self::Diagnostic => "instrumented_diagnostic",
        }
    }
}

fn metric_scope(
    id: &str,
    phase: Phase,
    native_scope: Option<&NativeMetricScope>,
) -> (String, Option<String>) {
    if let Some(native) = native_scope {
        return (
            format!("native_{}_{}", native.kind, native.stage),
            native.parent.clone(),
        );
    }
    if id == "scenario_wall_ms" {
        ("scenario_outer_wall".into(), None)
    } else if id.starts_with("stage_") {
        (
            "native_stage_inclusive".into(),
            Some("solver_integration".into()),
        )
    } else if phase == Phase::Gate {
        (
            "scenario_component".into(),
            Some("scenario_outer_wall".into()),
        )
    } else {
        (
            "solver_diagnostic".into(),
            Some("solver_integration".into()),
        )
    }
}

fn collect_phase(
    scenario: &Scenario,
    phase: Phase,
    runs: usize,
    required_metrics: &[String],
) -> Result<CollectedPhase, RunnerError> {
    let mut metric_samples: Vec<MetricSamples> = Vec::new();
    let mut counter_samples: Vec<(String, Vec<u64>)> = Vec::new();
    let mut expected_metrics: Option<Vec<(String, String)>> = None;
    let mut expected_counters: Option<Vec<String>> = None;
    let mut expected_diagnostics: Option<Vec<(String, String)>> = None;
    let mut last_diagnostics = Vec::new();

    for _ in 0..runs {
        let output = match phase {
            Phase::Gate => (scenario.run)()?,
            Phase::Diagnostic => (scenario
                .diagnose
                .expect("diagnostic phase requires a registered diagnostic"))(
                scenario
            )?,
        };
        if phase == Phase::Gate && output.metrics.is_empty() {
            return Err(RunnerError::Scenario(format!(
                "scenario {} returned no gate metrics",
                scenario.id
            )));
        }
        let metric_schema = unique_schema(
            scenario,
            output.metrics.iter().map(|(id, unit, _)| (id, unit)),
            "metric",
        )?;
        if phase == Phase::Gate {
            if expected_metrics
                .as_ref()
                .is_some_and(|expected| expected != &metric_schema)
            {
                return Err(RunnerError::Scenario(format!(
                    "scenario {} changed its gate metric set between samples",
                    scenario.id
                )));
            }
            expected_metrics.get_or_insert(metric_schema.clone());
        }
        for gate_id in required_metrics {
            if phase == Phase::Gate && !metric_schema.iter().any(|(id, _)| id == gate_id) {
                return Err(RunnerError::Scenario(format!(
                    "scenario {} gate metric {gate_id:?} is not emitted by the scenario",
                    scenario.id
                )));
            }
        }

        let mut counter_schema = output
            .counters
            .iter()
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        counter_schema.sort();
        if counter_schema.windows(2).any(|pair| pair[0] == pair[1])
            || counter_schema.iter().any(String::is_empty)
        {
            return Err(RunnerError::Scenario(format!(
                "scenario {} returned duplicate or empty counter IDs",
                scenario.id
            )));
        }
        if expected_counters
            .as_ref()
            .is_some_and(|expected| expected != &counter_schema)
        {
            return Err(RunnerError::Scenario(format!(
                "scenario {} changed its {phase:?} counter set between samples",
                scenario.id
            )));
        }
        expected_counters.get_or_insert(counter_schema);
        for (id, value) in &output.counters {
            if *value > i64::MAX as u64 {
                return Err(RunnerError::Scenario(format!(
                    "counter {id:?} exceeds the supported TOML integer range"
                )));
            }
            if let Some((_, samples)) = counter_samples.iter_mut().find(|(key, _)| key == id) {
                samples.push(*value);
            } else {
                counter_samples.push((id.clone(), vec![*value]));
            }
        }

        let diagnostic_schema = unique_schema(
            scenario,
            output.diagnostics.iter().map(|(id, unit, _)| (id, unit)),
            "correctness diagnostic",
        )?;
        if expected_diagnostics
            .as_ref()
            .is_some_and(|expected| expected != &diagnostic_schema)
        {
            return Err(RunnerError::Scenario(format!(
                "scenario {} changed its {phase:?} correctness diagnostic set between samples",
                scenario.id
            )));
        }
        expected_diagnostics.get_or_insert(diagnostic_schema);

        for (index, native_scope) in output.native_scopes.iter().enumerate() {
            if native_scope.metric_id.is_empty()
                || native_scope.stage.is_empty()
                || native_scope.kind.is_empty()
                || !output
                    .metrics
                    .iter()
                    .any(|(id, _, _)| id == &native_scope.metric_id)
                || output.native_scopes[..index]
                    .iter()
                    .any(|previous| previous.metric_id == native_scope.metric_id)
            {
                return Err(RunnerError::Scenario(format!(
                    "scenario {} returned invalid or duplicate native timing-scope metadata for {:?}",
                    scenario.id, native_scope.metric_id
                )));
            }
        }

        for (id, unit, value) in output.metrics {
            if id.is_empty() || unit.is_empty() || !value.is_finite() || value < 0.0 {
                return Err(RunnerError::Scenario(format!(
                    "metric {id:?} has an empty identifier/unit or is not finite and nonnegative"
                )));
            }
            let native_scope = output
                .native_scopes
                .iter()
                .find(|scope| scope.metric_id == id);
            let (scope, parent_scope) = metric_scope(&id, phase, native_scope);
            let metric =
                if let Some(existing) = metric_samples.iter_mut().find(|item| item.id == id) {
                    if existing.unit != unit
                        || existing.scope != scope
                        || existing.parent_scope != parent_scope
                    {
                        return Err(RunnerError::Scenario(format!(
                            "metric {id} changed unit or scope between samples"
                        )));
                    }
                    existing
                } else {
                    metric_samples.push(MetricSamples {
                        id: id.clone(),
                        unit: unit.clone(),
                        phase: phase.label().into(),
                        scope,
                        parent_scope,
                        samples: Vec::with_capacity(runs),
                    });
                    metric_samples.last_mut().expect("just inserted")
                };
            metric.samples.push(value);
        }
        if output
            .diagnostics
            .iter()
            .any(|(id, unit, value)| id.is_empty() || unit.is_empty() || !value.is_finite())
        {
            return Err(RunnerError::Scenario(format!(
                "scenario {} returned an invalid correctness diagnostic",
                scenario.id
            )));
        }
        last_diagnostics = output.diagnostics;
    }

    Ok(CollectedPhase {
        metrics: metric_samples,
        counters: counter_samples.into_iter().map(counter_stats).collect(),
        diagnostics: last_diagnostics,
    })
}

fn unique_schema<'a>(
    scenario: &Scenario,
    entries: impl Iterator<Item = (&'a String, &'a String)>,
    kind: &str,
) -> Result<Vec<(String, String)>, RunnerError> {
    let mut schema = entries
        .map(|(id, unit)| (id.clone(), unit.clone()))
        .collect::<Vec<_>>();
    schema.sort();
    if schema.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(RunnerError::Scenario(format!(
            "scenario {} returned duplicate {kind} IDs",
            scenario.id
        )));
    }
    Ok(schema)
}

fn stats(mut metric: MetricSamples) -> Result<MetricStats, RunnerError> {
    if metric.samples.is_empty() {
        return Err(RunnerError::Scenario(format!(
            "metric {} had no samples",
            metric.id
        )));
    }
    metric.samples.sort_by(f64::total_cmp);
    let median_value = median(&metric.samples);
    let mut deviations: Vec<_> = metric
        .samples
        .iter()
        .map(|sample| (sample - median_value).abs())
        .collect();
    deviations.sort_by(f64::total_cmp);
    let mad = median(&deviations);
    Ok(MetricStats {
        id: metric.id,
        unit: metric.unit,
        phase: metric.phase,
        scope: metric.scope,
        parent_scope: metric.parent_scope,
        min: metric.samples[0],
        max: *metric.samples.last().expect("nonempty checked"),
        median: median_value,
        mad,
        samples: metric.samples,
    })
}

fn median(sorted: &[f64]) -> f64 {
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        sorted[middle - 1] + (sorted[middle] - sorted[middle - 1]) / 2.0
    } else {
        sorted[middle]
    }
}

fn counter_stats((id, mut samples): (String, Vec<u64>)) -> CounterStats {
    samples.sort_unstable();
    let middle = samples.len() / 2;
    let median = if samples.len().is_multiple_of(2) {
        samples[middle - 1] as f64 / 2.0 + samples[middle] as f64 / 2.0
    } else {
        samples[middle] as f64
    };
    CounterStats {
        id,
        min: samples[0],
        max: *samples
            .last()
            .expect("counter samples are collected per run"),
        median,
        samples,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::ScenarioOutput;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn fixed_sample() -> Result<ScenarioOutput, RunnerError> {
        Ok(ScenarioOutput {
            metrics: vec![("solve_ms".into(), "ms".into(), 10.0)],
            native_scopes: Vec::new(),
            counters: vec![("steps".into(), 7)],
            diagnostics: vec![("residual".into(), "absolute".into(), 0.0)],
        })
    }

    static CHANGING_SAMPLE: AtomicUsize = AtomicUsize::new(0);

    fn changing_metric_schema() -> Result<ScenarioOutput, RunnerError> {
        let index = CHANGING_SAMPLE.fetch_add(1, Ordering::Relaxed);
        let id = if index.is_multiple_of(2) {
            "solve_ms"
        } else {
            "total_ms"
        };
        Ok(ScenarioOutput {
            metrics: vec![(id.into(), "ms".into(), 10.0)],
            ..ScenarioOutput::default()
        })
    }

    #[test]
    fn aggregates_median_mad_and_keeps_sample_values() {
        let scenario = Scenario {
            id: "fixed",
            solver: "test",
            groups: &[],
            workload: "fixed fixture",
            frontend: "not-applicable",
            execution: "test",
            linear_algebra: "not-applicable",
            parallel_policy: "sequential",
            telemetry: "off",
            version: 1,
            gate_metrics: &["solve_ms"],
            description: "test-only fixed sample",
            run: fixed_sample,
            diagnose: None,
        };
        let result = measure(&scenario, 0, 3).unwrap();
        assert_eq!(result.metrics.len(), 1);
        assert_eq!(result.metrics[0].median, 10.0);
        assert_eq!(result.metrics[0].mad, 0.0);
        assert_eq!(result.metrics[0].samples, [10.0, 10.0, 10.0]);
        assert_eq!(result.counters[0].id, "steps");
        assert_eq!(result.counters[0].median, 7.0);
        assert_eq!(result.counters[0].min, 7);
        assert_eq!(result.counters[0].max, 7);
    }

    #[test]
    fn rejects_a_metric_schema_that_changes_between_samples() {
        CHANGING_SAMPLE.store(0, Ordering::Relaxed);
        let scenario = Scenario {
            id: "changing",
            solver: "test",
            groups: &[],
            workload: "changing fixture",
            frontend: "not-applicable",
            execution: "test",
            linear_algebra: "not-applicable",
            parallel_policy: "sequential",
            telemetry: "off",
            version: 1,
            gate_metrics: &["solve_ms"],
            description: "test-only changing metric schema",
            run: changing_metric_schema,
            diagnose: None,
        };
        let error = measure(&scenario, 0, 3).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("changed its") && message.contains("metric set"));
    }

    #[test]
    fn aot_warm_route_requires_a_warmup_to_exclude_cold_compilation() {
        let scenario = Scenario {
            id: "aot-warm",
            solver: "test",
            groups: &[],
            workload: "fixed fixture",
            frontend: "ExprLegacy",
            execution: "AOT(warm-cache)",
            linear_algebra: "Dense",
            parallel_policy: "sequential",
            telemetry: "off",
            version: 1,
            gate_metrics: &["solve_ms"],
            description: "test-only AOT route",
            run: fixed_sample,
            diagnose: None,
        };
        let error = measure(&scenario, 0, 3).unwrap_err();
        assert!(error.to_string().contains("requires at least one warmup"));
    }

    static PHASE_ORDER: AtomicUsize = AtomicUsize::new(0);

    fn off_gate_sample() -> Result<ScenarioOutput, RunnerError> {
        let order = PHASE_ORDER.fetch_add(1, Ordering::SeqCst);
        assert!(
            (2..5).contains(&order) || order < 2,
            "warmups and all gate samples must precede diagnostics"
        );
        Ok(ScenarioOutput {
            metrics: vec![("scenario_wall_ms".into(), "ms/off".into(), 2.0)],
            ..ScenarioOutput::default()
        })
    }

    fn timed_diagnostic(_: &Scenario) -> Result<ScenarioOutput, RunnerError> {
        let order = PHASE_ORDER.fetch_add(1, Ordering::SeqCst);
        assert!(
            order >= 5,
            "diagnostics must run after warmups and gate sampling"
        );
        let metrics = if order == 5 {
            vec![("stage_factorization".into(), "ms/timing".into(), 0.0)]
        } else {
            Vec::new()
        };
        let native_scopes = if order == 5 {
            vec![crate::registry::NativeMetricScope {
                metric_id: "stage_factorization".into(),
                stage: "Factorization".into(),
                kind: "child".into(),
                parent: Some("Linear".into()),
            }]
        } else {
            Vec::new()
        };
        Ok(ScenarioOutput {
            metrics,
            native_scopes,
            counters: vec![("factorizations".into(), 0)],
            ..ScenarioOutput::default()
        })
    }

    #[test]
    fn gate_samples_precede_diagnostics_and_zero_is_not_missing() {
        PHASE_ORDER.store(0, Ordering::SeqCst);
        let scenario = Scenario {
            id: "phased",
            solver: "test",
            groups: &[],
            workload: "phase-order fixture",
            frontend: "not-applicable",
            execution: "test",
            linear_algebra: "not-applicable",
            parallel_policy: "sequential",
            telemetry: "off plus timed diagnostic",
            version: 1,
            gate_metrics: &["scenario_wall_ms"],
            description: "ensures diagnostics cannot perturb gate samples",
            run: off_gate_sample,
            diagnose: Some(timed_diagnostic),
        };
        let result = measure(&scenario, 2, 3).unwrap();
        assert_eq!(PHASE_ORDER.load(Ordering::SeqCst), 8);
        let wall = result
            .metrics
            .iter()
            .find(|metric| metric.id == "scenario_wall_ms")
            .unwrap();
        assert_eq!(wall.phase, "gate_off");
        assert_eq!(wall.scope, "scenario_outer_wall");
        let factor = result
            .metrics
            .iter()
            .find(|metric| metric.id == "stage_factorization")
            .unwrap();
        assert_eq!(factor.phase, "instrumented_diagnostic");
        assert_eq!(factor.scope, "native_child_Factorization");
        assert_eq!(factor.parent_scope.as_deref(), Some("Linear"));
        assert_eq!(factor.samples, [0.0]);
        assert_eq!(result.counters[0].samples, [0, 0, 0]);
    }
}
