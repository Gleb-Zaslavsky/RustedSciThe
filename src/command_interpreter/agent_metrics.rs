//! Typed catalog and comparison helpers for solver-reported numerical metrics.
//!
//! This module deliberately does not extend Agent Job v1. A future versioned
//! worker contract can serialize these requests after its wire semantics are
//! selected; Rust callers can use the typed evaluator today.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct NumericalMetricDefinition {
    pub name: &'static str,
    pub unit: &'static str,
    pub scope: &'static str,
    pub description: &'static str,
}

/// Stable common metrics emitted by the bounded task-result telemetry schema.
pub const NUMERICAL_METRIC_CATALOG: &[NumericalMetricDefinition] = &[
    NumericalMetricDefinition {
        name: "solve_elapsed",
        unit: "ms",
        scope: "solver_solve_call",
        description: "Elapsed time inside the solver solve call.",
    },
    NumericalMetricDefinition {
        name: "nonlinear_iterations",
        unit: "count",
        scope: "nonlinear_solver",
        description: "Native nonlinear iterations or updates reported by the route.",
    },
    NumericalMetricDefinition {
        name: "residual_evaluations",
        unit: "count",
        scope: "native_solver",
        description: "Residual evaluations reported by the native solver route.",
    },
    NumericalMetricDefinition {
        name: "jacobian_evaluations",
        unit: "count",
        scope: "native_solver",
        description: "Jacobian evaluations reported by the native solver route.",
    },
    NumericalMetricDefinition {
        name: "linear_solves",
        unit: "count",
        scope: "native_solver",
        description: "Linear-system solve calls reported by the native solver route.",
    },
    NumericalMetricDefinition {
        name: "factorizations",
        unit: "count",
        scope: "native_solver",
        description: "Linear factorization calls reported by the native solver route.",
    },
    NumericalMetricDefinition {
        name: "accepted_time_steps",
        unit: "count",
        scope: "ivp_solver",
        description: "Accepted IVP time steps.",
    },
    NumericalMetricDefinition {
        name: "rejected_time_steps",
        unit: "count",
        scope: "ivp_solver",
        description: "Rejected IVP time steps.",
    },
    NumericalMetricDefinition {
        name: "accepted_nonlinear_updates",
        unit: "count",
        scope: "nonlinear_solver",
        description: "Accepted nonlinear updates reported by the route.",
    },
    NumericalMetricDefinition {
        name: "rejected_nonlinear_trials",
        unit: "count",
        scope: "nonlinear_solver",
        description: "Rejected nonlinear trials reported by the route.",
    },
    NumericalMetricDefinition {
        name: "mesh_refinements",
        unit: "count",
        scope: "bvp_solver",
        description: "BVP mesh refinements reported by the route.",
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricComparisonOperator {
    LessThan,
    LessThanOrEqual,
    Equal,
    GreaterThanOrEqual,
    GreaterThan,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricComparisonRequest {
    pub metric: String,
    pub operator: MetricComparisonOperator,
    pub threshold: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricAvailability {
    Measured,
    NotRequested,
    Unsupported,
    Partial,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricComparisonStatus {
    Passed,
    Failed,
    NotEvaluated,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricComparisonResult {
    pub metric: String,
    pub operator: MetricComparisonOperator,
    pub threshold: f64,
    pub observed: Option<f64>,
    pub availability: MetricAvailability,
    pub status: MetricComparisonStatus,
    pub reason: Option<String>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum MetricComparisonError {
    #[error("unknown numerical metric `{0}`")]
    UnknownMetric(String),
    #[error("threshold for numerical metric `{0}` must be finite")]
    NonFiniteThreshold(String),
}

/// Return a comparison result for each request. Only fully measured metrics
/// can pass or fail a comparison; partial, unsupported, and unrequested values
/// remain explicitly not evaluated rather than being coerced to zero.
pub fn evaluate_metric_comparisons(
    telemetry: Option<&Value>,
    requests: &[MetricComparisonRequest],
) -> Result<Vec<MetricComparisonResult>, MetricComparisonError> {
    requests
        .iter()
        .map(|request| {
            if !NUMERICAL_METRIC_CATALOG
                .iter()
                .any(|definition| definition.name == request.metric)
            {
                return Err(MetricComparisonError::UnknownMetric(request.metric.clone()));
            }
            if !request.threshold.is_finite() {
                return Err(MetricComparisonError::NonFiniteThreshold(
                    request.metric.clone(),
                ));
            }

            let metric = telemetry
                .and_then(|report| report.get("metrics"))
                .and_then(|metrics| metrics.get(&request.metric));
            let availability = metric
                .and_then(|metric| metric.get("availability"))
                .and_then(Value::as_str)
                .and_then(parse_availability)
                .unwrap_or(if telemetry.is_some() {
                    MetricAvailability::Unsupported
                } else {
                    MetricAvailability::NotRequested
                });
            let observed = metric
                .and_then(|metric| metric.get("value"))
                .and_then(Value::as_f64)
                .filter(|value| value.is_finite());
            let (status, reason) = if availability == MetricAvailability::Measured {
                if let Some(observed) = observed {
                    (
                        if compare(observed, request.operator, request.threshold) {
                            MetricComparisonStatus::Passed
                        } else {
                            MetricComparisonStatus::Failed
                        },
                        None,
                    )
                } else {
                    (
                        MetricComparisonStatus::NotEvaluated,
                        Some("measured_metric_has_no_finite_value".to_owned()),
                    )
                }
            } else {
                (
                    MetricComparisonStatus::NotEvaluated,
                    Some(
                        match availability {
                            MetricAvailability::NotRequested => "telemetry_not_requested",
                            MetricAvailability::Unsupported => "metric_unsupported_by_route",
                            MetricAvailability::Partial => "metric_value_is_partial",
                            MetricAvailability::Measured => unreachable!(),
                        }
                        .to_owned(),
                    ),
                )
            };
            Ok(MetricComparisonResult {
                metric: request.metric.clone(),
                operator: request.operator,
                threshold: request.threshold,
                observed,
                availability,
                status,
                reason,
            })
        })
        .collect()
}

fn parse_availability(value: &str) -> Option<MetricAvailability> {
    match value {
        "measured" => Some(MetricAvailability::Measured),
        "not_requested" => Some(MetricAvailability::NotRequested),
        "unsupported" => Some(MetricAvailability::Unsupported),
        "partial" => Some(MetricAvailability::Partial),
        _ => None,
    }
}

fn compare(observed: f64, operator: MetricComparisonOperator, threshold: f64) -> bool {
    match operator {
        MetricComparisonOperator::LessThan => observed < threshold,
        MetricComparisonOperator::LessThanOrEqual => observed <= threshold,
        MetricComparisonOperator::Equal => observed == threshold,
        MetricComparisonOperator::GreaterThanOrEqual => observed >= threshold,
        MetricComparisonOperator::GreaterThan => observed > threshold,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MetricAvailability, MetricComparisonError, MetricComparisonOperator,
        MetricComparisonRequest, MetricComparisonStatus, NUMERICAL_METRIC_CATALOG,
        evaluate_metric_comparisons,
    };
    use serde_json::json;

    fn request(
        metric: &str,
        operator: MetricComparisonOperator,
        threshold: f64,
    ) -> MetricComparisonRequest {
        MetricComparisonRequest {
            metric: metric.into(),
            operator,
            threshold,
        }
    }

    #[test]
    fn catalog_names_match_the_compact_telemetry_contract() {
        let expected = [
            "solve_elapsed",
            "nonlinear_iterations",
            "residual_evaluations",
            "jacobian_evaluations",
            "linear_solves",
            "factorizations",
            "accepted_time_steps",
            "rejected_time_steps",
            "accepted_nonlinear_updates",
            "rejected_nonlinear_trials",
            "mesh_refinements",
        ];
        assert_eq!(NUMERICAL_METRIC_CATALOG.len(), expected.len());
        for (definition, name) in NUMERICAL_METRIC_CATALOG.iter().zip(expected) {
            assert_eq!(definition.name, name);
            assert!(!definition.unit.is_empty());
            assert!(!definition.scope.is_empty());
        }
    }

    #[test]
    fn compares_only_fully_measured_values_and_preserves_absence_states() {
        let telemetry = json!({"metrics":{
            "residual_evaluations":{"availability":"measured", "value":12},
            "jacobian_evaluations":{"availability":"partial", "value":4},
            "factorizations":{"availability":"unsupported", "value":null},
            "linear_solves":{"availability":"not_requested", "value":null}
        }});
        let results = evaluate_metric_comparisons(
            Some(&telemetry),
            &[
                request(
                    "residual_evaluations",
                    MetricComparisonOperator::LessThanOrEqual,
                    12.0,
                ),
                request(
                    "jacobian_evaluations",
                    MetricComparisonOperator::GreaterThan,
                    1.0,
                ),
                request("factorizations", MetricComparisonOperator::Equal, 0.0),
                request("linear_solves", MetricComparisonOperator::GreaterThan, 0.0),
                request("solve_elapsed", MetricComparisonOperator::LessThan, 1.0),
            ],
        )
        .unwrap();
        assert_eq!(results[0].status, MetricComparisonStatus::Passed);
        assert_eq!(results[1].availability, MetricAvailability::Partial);
        assert_eq!(results[1].status, MetricComparisonStatus::NotEvaluated);
        assert_eq!(results[2].availability, MetricAvailability::Unsupported);
        assert_eq!(results[2].status, MetricComparisonStatus::NotEvaluated);
        assert_eq!(results[3].availability, MetricAvailability::NotRequested);
        assert_eq!(results[3].status, MetricComparisonStatus::NotEvaluated);
        assert_eq!(results[4].availability, MetricAvailability::Unsupported);
        assert_eq!(results[4].status, MetricComparisonStatus::NotEvaluated);

        let no_telemetry = evaluate_metric_comparisons(
            None,
            &[request(
                "solve_elapsed",
                MetricComparisonOperator::LessThan,
                1.0,
            )],
        )
        .unwrap();
        assert_eq!(
            no_telemetry[0].availability,
            MetricAvailability::NotRequested
        );
        assert_eq!(no_telemetry[0].status, MetricComparisonStatus::NotEvaluated);
    }

    #[test]
    fn rejects_unknown_metrics_and_non_finite_thresholds() {
        assert_eq!(
            evaluate_metric_comparisons(
                None,
                &[request(
                    "physical_temperature",
                    MetricComparisonOperator::GreaterThan,
                    1.0
                )],
            ),
            Err(MetricComparisonError::UnknownMetric(
                "physical_temperature".into()
            ))
        );
        assert_eq!(
            evaluate_metric_comparisons(
                None,
                &[request(
                    "solve_elapsed",
                    MetricComparisonOperator::LessThan,
                    f64::INFINITY
                )],
            ),
            Err(MetricComparisonError::NonFiniteThreshold(
                "solve_elapsed".into()
            ))
        );
    }
}
