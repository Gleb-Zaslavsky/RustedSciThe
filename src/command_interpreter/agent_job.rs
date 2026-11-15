//! Versioned outer job metadata for external-agent orchestration.
//!
//! The embedded `TaskDocument` remains the executable numerical request.
//! This layer validates job metadata and acceptance declarations but does not
//! enforce worker resource budgets or run the task itself.

use crate::command_interpreter::task_contract::{TaskInspection, inspect_task_json};
use crate::command_interpreter::task_json::{TaskJsonError, json_value_from_json};
use crate::command_interpreter::task_runner::TaskRunnerError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;

/// Draft 2020-12 schema for the versioned external-agent job wrapper.
pub const AGENT_JOB_JSON_SCHEMA: &str = include_str!("agent_job.schema.json");
/// Draft 2020-12 schema for acceptance-evaluation reports.
pub const ACCEPTANCE_REPORT_JSON_SCHEMA: &str = include_str!("acceptance_report.schema.json");

/// Outer worker-job contract, version 1.
#[derive(Debug, Clone, PartialEq)]
pub struct AgentJobSpec {
    pub job_id: Option<String>,
    /// The canonical task payload, validated by the native family parser.
    pub task: Value,
    pub task_inspection: TaskInspection,
    pub acceptance: AcceptancePolicy,
    /// Worker metadata only; this crate does not apply these limits.
    pub resource_budget: ResourceBudget,
    pub output_policy: OutputPolicy,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AgentJobDocument {
    job_contract_version: u32,
    #[serde(default)]
    job_id: Option<String>,
    task: Value,
    #[serde(default)]
    acceptance: AcceptancePolicy,
    #[serde(default)]
    resource_budget: ResourceBudget,
    #[serde(default)]
    output_policy: OutputPolicy,
}

/// Generic, solver-independent checks available in job contract v1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcceptanceCheck {
    /// The native task result reports successful solver termination.
    SolverConverged,
    /// All reported numeric result values are finite.
    FiniteResult,
}

/// Checks required for this job to be considered accepted by the caller.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcceptancePolicy {
    #[serde(default)]
    pub checks: Vec<AcceptanceCheck>,
}

/// Resource limits that must be enforced by the worker/scheduler.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceBudget {
    pub wall_time_ms: Option<u64>,
    pub memory_bytes: Option<u64>,
    pub cpu_threads: Option<usize>,
}

/// Output handling currently supported by the RST task envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LargeResultPolicy {
    /// Keep small values inline and explicitly omit oversized payloads.
    #[default]
    Omit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputPolicy {
    #[serde(default)]
    pub large_result: LargeResultPolicy,
}

/// Result of evaluating one generic acceptance check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcceptanceCheckStatus {
    Passed,
    Failed,
    NotEvaluated,
}

impl AcceptanceCheckStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Passed => "passed",
            Self::Failed => "failed",
            Self::NotEvaluated => "not_evaluated",
        }
    }
}

/// One evaluated check and its machine-readable evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct AcceptanceCheckResult {
    pub check: AcceptanceCheck,
    pub status: AcceptanceCheckStatus,
    pub observed: Option<Value>,
}

/// Aggregate caller-acceptance outcome, kept separate from solver status.
/// A proven failed check rejects the job even if another check lacks evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct AcceptanceReport {
    /// `not_requested`, `accepted`, `rejected`, or `not_evaluated`.
    pub status: &'static str,
    pub checks: Vec<AcceptanceCheckResult>,
}

impl AcceptanceReport {
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "status": self.status,
            "checks": self.checks.iter().map(|check| serde_json::json!({
                "name": check.check.as_str(),
                "status": check.status.as_str(),
                "observed": check.observed,
            })).collect::<Vec<_>>(),
        })
    }
}

impl AcceptanceCheck {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SolverConverged => "solver_converged",
            Self::FiniteResult => "finite_result",
        }
    }
}

/// Parse the outer job, validate its metadata, and preflight the embedded task.
///
/// Resource budgets are returned for the caller-owned worker to enforce. This
/// function does not launch a solver, spawn a worker, or claim those budgets
/// were applied.
pub fn parse_agent_job_json(input: &str) -> Result<AgentJobSpec, AgentJobError> {
    let value = json_value_from_json(input).map_err(AgentJobError::Json)?;
    let document: AgentJobDocument =
        serde_json::from_value(value).map_err(AgentJobError::Envelope)?;
    if document.job_contract_version != 1 {
        return Err(AgentJobError::UnsupportedVersion {
            found: document.job_contract_version,
        });
    }
    if let Some(job_id) = &document.job_id
        && (job_id.trim().is_empty() || job_id.chars().count() > 128)
    {
        return Err(AgentJobError::InvalidJobId);
    }
    for (name, value) in [
        ("wall_time_ms", document.resource_budget.wall_time_ms),
        ("memory_bytes", document.resource_budget.memory_bytes),
    ] {
        if value == Some(0) {
            return Err(AgentJobError::InvalidResourceBudget { field: name });
        }
    }
    if document.resource_budget.cpu_threads == Some(0) {
        return Err(AgentJobError::InvalidResourceBudget {
            field: "cpu_threads",
        });
    }
    let mut unique_checks = HashSet::new();
    for check in &document.acceptance.checks {
        if !unique_checks.insert(*check) {
            return Err(AgentJobError::DuplicateAcceptanceCheck {
                check: check.as_str(),
            });
        }
    }

    let task_json = serde_json::to_string(&document.task).map_err(AgentJobError::Envelope)?;
    let task_inspection = inspect_task_json(&task_json).map_err(AgentJobError::Task)?;

    Ok(AgentJobSpec {
        job_id: document.job_id,
        task: document.task,
        task_inspection,
        acceptance: document.acceptance,
        resource_budget: document.resource_budget,
        output_policy: document.output_policy,
    })
}

/// Evaluate only the generic, versioned checks declared by contract v1.
///
/// Domain-specific residual, boundary, invariant, and physical checks remain
/// caller-owned. The input is the ordinary task execution envelope.
pub fn evaluate_acceptance(envelope: &Value, policy: &AcceptancePolicy) -> AcceptanceReport {
    if policy.checks.is_empty() {
        return AcceptanceReport {
            status: "not_requested",
            checks: Vec::new(),
        };
    }
    let status = envelope.get("status").and_then(Value::as_str);
    let result = envelope.get("result").filter(|value| !value.is_null());
    let checks = policy
        .checks
        .iter()
        .copied()
        .map(|check| evaluate_one(check, status, result))
        .collect::<Vec<_>>();
    let report_status = if checks
        .iter()
        .any(|check| check.status == AcceptanceCheckStatus::Failed)
    {
        "rejected"
    } else if result.is_none()
        || checks
            .iter()
            .any(|check| check.status == AcceptanceCheckStatus::NotEvaluated)
    {
        "not_evaluated"
    } else if checks
        .iter()
        .all(|check| check.status == AcceptanceCheckStatus::Passed)
    {
        "accepted"
    } else {
        "rejected"
    };
    AcceptanceReport {
        status: report_status,
        checks,
    }
}

fn evaluate_one(
    check: AcceptanceCheck,
    envelope_status: Option<&str>,
    result: Option<&Value>,
) -> AcceptanceCheckResult {
    let (status, observed) = match check {
        AcceptanceCheck::SolverConverged => match envelope_status {
            Some("succeeded") => (AcceptanceCheckStatus::Passed, Some(Value::Bool(true))),
            Some("not_converged") => (AcceptanceCheckStatus::Failed, Some(Value::Bool(false))),
            _ => (AcceptanceCheckStatus::NotEvaluated, None),
        },
        AcceptanceCheck::FiniteResult => {
            let count = result
                .and_then(|result| result.get("non_finite_values_encoded_as_null"))
                .and_then(Value::as_u64);
            match count {
                Some(0) => (AcceptanceCheckStatus::Passed, Some(Value::from(0))),
                Some(count) => (AcceptanceCheckStatus::Failed, Some(Value::from(count))),
                None => (AcceptanceCheckStatus::NotEvaluated, None),
            }
        }
    };
    AcceptanceCheckResult {
        check,
        status,
        observed,
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AgentJobError {
    #[error(transparent)]
    Json(#[from] TaskJsonError),
    #[error("invalid agent job envelope: {0}")]
    Envelope(#[source] serde_json::Error),
    #[error("unsupported agent job contract version {found}; this build supports version 1")]
    UnsupportedVersion { found: u32 },
    #[error("job_id must contain 1 to 128 non-whitespace characters")]
    InvalidJobId,
    #[error("resource budget field `{field}` must be greater than zero")]
    InvalidResourceBudget { field: &'static str },
    #[error("acceptance check `{check}` is specified more than once")]
    DuplicateAcceptanceCheck { check: &'static str },
    #[error("embedded task is invalid: {0}")]
    Task(#[source] TaskRunnerError),
}

#[cfg(test)]
#[path = "agent_job_tests.rs"]
mod tests;
